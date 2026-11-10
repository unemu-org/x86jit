use super::*;
use crate::{HostTarget, JitBackend, OptLevel};
use cranelift::codegen::Context;
use cranelift_module::Module;
use x86jit_core::jit_abi::cpu_offsets;
use x86jit_core::lift::{lift_block, FetchAddr};
use x86jit_core::{Backend, CachedBlock, CpuMode, Memory, MemoryModel, Prot, RegionKind};

const BASE: u64 = 0x1000;
const HIGH_BASE: u64 = 0x1234_5678_0000_1000;

thread_local! {
    static CHECKED_HOSTS: std::cell::RefCell<Vec<Value>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub(super) fn record_checked_addr(host: Value) {
    CHECKED_HOSTS.with(|hosts| hosts.borrow_mut().push(host));
}

fn audit_guest_accesses(func: &ir::Function, hosts: &[Value]) -> Vec<(u64, GuestAccessKind, u8)> {
    let mut accesses = Vec::new();
    for inst in func
        .layout
        .blocks()
        .flat_map(|b| func.layout.block_insts(b))
    {
        use ir::InstructionData::*;
        let data = &func.dfg.insts[inst];
        if !data.opcode().can_load() && !data.opcode().can_store() {
            continue;
        }
        let Some(flags) = data.memflags() else {
            continue;
        };
        let flags = func.dfg.mem_flags[flags];
        let (host, access, ty) = match data {
            Load { arg, .. } | LoadNoOffset { arg, .. } => (
                *arg,
                GuestAccessKind::Read,
                func.dfg.value_type(func.dfg.first_result(inst)),
            ),
            Store { args, .. } | StoreNoOffset { args, .. } => (
                args[1],
                GuestAccessKind::Write,
                func.dfg.value_type(args[0]),
            ),
            AtomicRmw { args, op, .. } => {
                let ty = func.dfg.value_type(args[1]);
                let probe = ty == types::I8
                    && *op == ir::AtomicRmwOp::Add
                    && matches!(func.dfg.insts[func.dfg.value_def(args[1]).unwrap_inst()],
                        UnaryImm { imm, .. } if imm.bits() == 0);
                (
                    args[0],
                    if probe {
                        GuestAccessKind::Write
                    } else {
                        GuestAccessKind::ReadWrite
                    },
                    ty,
                )
            }
            AtomicCas { args, .. } => (
                args[0],
                GuestAccessKind::ReadWrite,
                func.dfg.value_type(args[1]),
            ),
            _ => unreachable!(),
        };
        // Membership comes only from checked_addr, independently of the flags
        // under test. In particular, making a guest access trusted cannot hide it.
        if hosts
            .iter()
            .any(|h| func.dfg.resolve_aliases(*h) == func.dfg.resolve_aliases(host))
        {
            let width = ty.bytes() as u8;
            assert_eq!(
                flags.trap_code().and_then(guest_trap_access),
                Some((access, width)),
                "guest memory flags: {data:?}"
            );
            assert!(!flags.notrap() && !flags.aligned());
            accesses.push((func.srcloc(inst).bits() as u64, access, width));
        } else {
            assert!(flags.notrap(), "engine memory must be trusted: {data:?}");
        }
    }
    accesses
}

fn high_block(bytes: &[u8]) -> IrBlock {
    let mut mem = Memory::new(MemoryModel::Flat { size: 0x10000 });
    mem.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    mem.write_bytes(BASE, bytes).unwrap();
    let mut ir = lift_block(&mem, FetchAddr::flat(BASE), CpuMode::Long64).unwrap();
    for op in &mut ir.ops {
        if let IrOp::InsnStart { guest_addr } = op {
            *guest_addr += HIGH_BASE - BASE;
        }
    }
    ir.guest_start = HIGH_BASE;
    ir
}

fn compile(bytes: &[u8], region: bool, target: HostTarget) -> (Context, Box<[u64]>, usize) {
    compile_at(bytes, region, target, OptLevel::None);
    compile_at(bytes, region, target, OptLevel::Speed)
}

fn compile_at(
    bytes: &[u8],
    region: bool,
    target: HostTarget,
    opt: OptLevel,
) -> (Context, Box<[u64]>, usize) {
    let jit = JitBackend::build_jit(target, opt);
    compile_for_isa(bytes, region, jit.module.isa())
}

fn compile_for_isa(
    bytes: &[u8],
    region: bool,
    isa: &dyn cranelift::codegen::isa::TargetIsa,
) -> (Context, Box<[u64]>, usize) {
    compile_ir_for_isa(high_block(bytes), bytes, region, isa)
}

fn compile_ir_for_isa(
    ir: IrBlock,
    bytes: &[u8],
    region: bool,
    isa: &dyn cranelift::codegen::isa::TargetIsa,
) -> (Context, Box<[u64]>, usize) {
    compile_ir_for_isa_with_split_stores(ir, bytes, region, isa, true)
}

fn compile_ir_for_isa_with_split_stores(
    ir: IrBlock,
    bytes: &[u8],
    region: bool,
    isa: &dyn cranelift::codegen::isa::TargetIsa,
    precise: bool,
) -> (Context, Box<[u64]>, usize) {
    CHECKED_HOSTS.with(|hosts| hosts.borrow_mut().clear());
    let mut ctx = Context::new();
    ctx.func
        .signature
        .params
        .extend([AbiParam::new(types::I64); 2]);
    ctx.func.signature.returns.push(AbiParam::new(types::I64));
    let mut fbctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
    let sig = |n, returns| {
        let mut sig = ir::Signature::new(isa.default_call_conv());
        sig.params = vec![AbiParam::new(types::I64); n];
        if returns {
            sig.returns.push(AbiParam::new(types::I64));
        }
        sig
    };
    let dummy = builder.import_signature(sig(6, true));
    let watch = builder.import_signature(sig(3, false));
    let mut helpers = Helpers::all_dummy(&mut || (dummy, 0, 0), (watch, 0, 0));
    helpers.precise_split_stores = precise;
    helpers.cpuid = (builder.import_signature(sig(1, true)), 0, 0);
    helpers.vmaskmov_mem = (builder.import_signature(sig(11, true)), 0, 0);
    let offsets = cpu_offsets();
    let rips = if region {
        translate_region(
            &mut builder,
            &IrRegion {
                entry: HIGH_BASE,
                blocks: vec![ir],
                has_loop: false,
            },
            &offsets,
            &mut || 0,
            helpers,
            MemConsistency::Fast,
            None,
            0,
            false,
        )
    } else {
        translate_block(
            &mut builder,
            &ir,
            &offsets,
            &mut || 0,
            helpers,
            MemConsistency::Fast,
            None,
            0,
            false,
        )
    };
    builder.finalize(isa.frontend_config());
    ctx.set_disasm(true);
    let hosts = CHECKED_HOSTS.with(|hosts| std::mem::take(&mut *hosts.borrow_mut()));
    let original_accesses = audit_guest_accesses(&ctx.func, &hosts);
    let metadata = crate::prepare_guest_traps(&mut ctx.func, &rips).unwrap_or_else(|error| {
        panic!("metadata refused: {error:?}, {bytes:02x?}, region={region}")
    });
    let started = std::time::Instant::now();
    ctx.compile(isa, &mut Default::default()).unwrap();
    let cc = ctx.compiled_code().unwrap();
    crate::libcalls::assert_supported_relocations(cc);
    let compile_ns = started.elapsed().as_nanos();
    let started = std::time::Instant::now();
    let sites = crate::collect_guest_traps(cc, &metadata).unwrap_or_else(|()| {
        panic!(
            "unclassified machine traps: {:?}, bytes={bytes:02x?}",
            cc.buffer.traps()
        )
    });
    if std::env::var_os("X86JIT_TRAP_TEST_TIMINGS").is_some() {
        eprintln!(
        "MEASURED: region={region}: {} code bytes, {} traps, {compile_ns} ns compile, {} record bytes, {} ns collect",
        cc.code_info().total_size,
        cc.buffer.traps().len(),
        sites.len() * std::mem::size_of::<x86jit_core::GuestTrapSite>(),
        started.elapsed().as_nanos(),
    );
    }
    assert_eq!(
        sites.len(),
        cc.buffer.traps().len(),
        "every reported trap needs a record"
    );
    // Count optimized IR, so removed accesses need no record. The lifted
    // repeated-read fixture retains both loads because CpuState stores separate
    // them; its synthetic sibling pins elimination without an intervening commit.
    let accesses: Vec<_> = ctx
        .func
        .layout
        .blocks()
        .flat_map(|b| ctx.func.layout.block_insts(b))
        .filter_map(|inst| {
            let data = &ctx.func.dfg.insts[inst];
            if !data.opcode().can_load() && !data.opcode().can_store() {
                return None;
            }
            let code = ctx.func.dfg.mem_flags[data.memflags()?].trap_code()?;
            let descriptor = metadata[code.as_raw().get() as usize - 1];
            Some((descriptor.guest_rip, descriptor.access, descriptor.width))
        })
        .collect();
    for site in &sites {
        assert!(
            accesses.contains(&(site.guest_rip, site.access, site.width)),
            "record must describe an IR guest access: {site:?}"
        );
    }
    for access in &accesses {
        let ir_count = accesses.iter().filter(|a| *a == access).count();
        let site_count = sites
            .iter()
            .filter(|s| (s.guest_rip, s.access, s.width) == *access)
            .count();
        assert!(
            site_count >= ir_count,
            "each IR access must retain a trap site: {access:?}"
        );
    }
    (ctx, rips, original_accesses.len())
}

#[test]
fn shifts_and_rotates_compile_with_guest_traps_on_both_isas() {
    for target in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for opt in [OptLevel::None, OptLevel::Speed] {
            let mut flags = settings::builder();
            flags.set("opt_level", opt.flag()).unwrap();
            let isa = cranelift::codegen::isa::lookup(target.parse().unwrap())
                .unwrap()
                .finish(settings::Flags::new(flags))
                .unwrap();
            for region in [false, true] {
                for prefix in [&[][..], &[0x66], &[0x48]] {
                    for opcode in [0xd0, 0xd1, 0xd2, 0xd3, 0xc0, 0xc1] {
                        for op in [0, 1, 2, 3, 4, 5, 7] {
                            let mut bytes = prefix.to_vec();
                            bytes.extend([opcode, 0xc0 | (op << 3)]);
                            if matches!(opcode, 0xc0 | 0xc1) {
                                bytes.push(1);
                            }
                            bytes.extend([0x48, 0x8b, 0x07, 0xf4]);
                            compile_for_isa(&bytes, region, &*isa);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn guest_trap_sites_codes_pin_direction_and_width() {
    for (access, first_code) in [
        (GuestAccessKind::Read, 1),
        (GuestAccessKind::Write, 6),
        (GuestAccessKind::ReadWrite, 11),
    ] {
        for (index, width) in [1, 2, 4, 8, 16].into_iter().enumerate() {
            let flags = guest_mem_flags(access, width);
            let code = ir::TrapCode::user(first_code + index as u8).unwrap();
            assert_eq!(flags.trap_code(), Some(code));
            assert_eq!(guest_trap_access(code), Some((access, width)));
            assert!(!flags.notrap() && !flags.aligned());
        }
    }
    for code in ir::TrapCode::non_user_traps() {
        assert_eq!(guest_trap_access(*code), None);
    }
    assert_eq!(guest_trap_access(ir::TrapCode::user(16).unwrap()), None);
}

#[test]
fn guest_trap_sites_cover_access_shapes() {
    // Scalar, vector, stack, control-flow and atomic guest transfers.
    let lane_sites = if cfg!(target_arch = "aarch64") { 1 } else { 2 };
    let cases: &[(&[u8], usize)] = &[
        (&[0x8a, 0x07, 0xf4], 1),
        (&[0x66, 0x8b, 0x07, 0xf4], 1),
        (&[0x8b, 0x07, 0xf4], 1),
        (&[0x48, 0x8b, 0x07, 0xf4], 1),
        (&[0x88, 0x07, 0xf4], 1),
        (&[0x66, 0x89, 0x07, 0xf4], 1),
        (&[0x89, 0x07, 0xf4], 1),
        (&[0x48, 0x89, 0x07, 0xf4], 1),
        (&[0x0f, 0x10, 0x07, 0xf4], lane_sites),
        (&[0xf3, 0x0f, 0x7f, 0x07, 0xf4], lane_sites),
        (&[0xf0, 0x48, 0x01, 0x07, 0xf4], 1),
        (&[0xf0, 0x48, 0xf7, 0x1f, 0xf4], 2),
        (&[0xf0, 0x48, 0x0f, 0xb1, 0x0f, 0xf4], 1),
        // Stack and control-flow transfers.
        (&[0x50, 0xf4], 1),
        (&[0x58, 0xf4], 1),
        (&[0xff, 0xd0], 1),
        (&[0xe8, 0, 0, 0, 0], 1),
        (&[0xc3], 1),
        // Wide transfers, broadcast, narrow extension and half-register moves.
        (&[0xc5, 0xfe, 0x6f, 0x07, 0xf4], 2 * lane_sites),
        (&[0xc5, 0xfe, 0x7f, 0x07, 0xf4], 2 * lane_sites),
        (&[0x62, 0xf1, 0xfe, 0x48, 0x6f, 0x07, 0xf4], 4 * lane_sites),
        (&[0x62, 0xf1, 0xfe, 0x48, 0x7f, 0x07, 0xf4], 4 * lane_sites),
        (&[0xc4, 0xe2, 0x7d, 0x18, 0x07, 0xf4], 1),
        (&[0x66, 0x0f, 0x38, 0x32, 0x07, 0xf4], 1),
        (&[0x0f, 0x16, 0x07, 0xf4], 1),
        (&[0x0f, 0x17, 0x07, 0xf4], 1),
        (&[0x0f, 0x12, 0x07, 0xf4], 1),
        (&[0x0f, 0x13, 0x07, 0xf4], 1),
        // Unmasked opmask compare/test sources use inline memory loads.
        (&[0x62, 0xf3, 0x7d, 0x08, 0x1f, 0x0f, 6, 0xf4], lane_sites),
        (&[0x62, 0xf2, 0x7e, 0x08, 0x26, 0x0f, 0xf4], lane_sites),
        // Repeated reads of an identical address.
        (&[0x48, 0x8b, 0x07, 0x48, 0x8b, 0x1f, 0xf4], 1),
        // Scalar FP loads, and an inline packed VEX memory operand.
        (&[0xf3, 0x0f, 0x58, 0x07, 0xf4], 1),
        (&[0xf2, 0x0f, 0x58, 0x07, 0xf4], 1),
        (&[0xc5, 0xf8, 0x58, 0x07, 0xf4], lane_sites),
    ];
    let mut missing = Vec::new();
    for region in [false, true] {
        for target in [HostTarget::Baseline, HostTarget::Native] {
            for &(bytes, minimum_sites) in cases {
                let (ctx, _, access_count) = compile(bytes, region, target);
                if ctx.compiled_code().unwrap().buffer.traps().len() < minimum_sites
                    || access_count < minimum_sites
                {
                    missing.push((bytes, region, target));
                }
            }
            // Masked sources use the fault-reporting memory helper, whose
            // accesses have no inline guest trap sites in the compiled caller.
            for bytes in [
                &[0x62, 0xf3, 0x7d, 0x0a, 0x1f, 0x0f, 6, 0xf4][..],
                &[0x62, 0xf2, 0x7e, 0x0a, 0x26, 0x0f, 0xf4],
            ] {
                assert!(high_block(bytes)
                    .ops
                    .iter()
                    .any(|op| { matches!(op, IrOp::VMaskLoadMem { k: 2, .. }) }));
                let (_, _, access_count) = compile(bytes, region, target);
                assert_eq!(access_count, 0);
            }
        }
    }
    assert!(
        missing.is_empty(),
        "guest accesses must have trap sites: {missing:02x?}"
    );
}

#[test]
fn guest_trap_sites_are_registered_for_blocks_and_regions() {
    // The two instruction RIPs share their low 32 bits across two region blocks.
    // Metadata must distinguish them and preserve the widths of split transfers.
    let mut first = high_block(&[0x48, 0x8b, 0x07, 0xeb, 0x00]);
    let mut second = high_block(&[0x0f, 0x11, 0x07, 0xf4]);
    let second_rip = HIGH_BASE + (1 << 32);
    for op in &mut first.ops {
        if let IrOp::Jump { target } = op {
            *target = Val::Imm(second_rip);
        }
    }
    second.guest_start = second_rip;
    for op in &mut second.ops {
        if let IrOp::InsnStart { guest_addr } = op {
            *guest_addr += 1 << 32;
        }
    }
    for region in [false, true] {
        let jit = JitBackend::new();
        let block = if region {
            jit.materialize_region(
                &IrRegion {
                    entry: HIGH_BASE,
                    blocks: vec![first.clone(), second.clone()],
                    has_loop: false,
                },
                MemConsistency::Fast,
                None,
                0,
            )
        } else {
            jit.materialize(&first, MemConsistency::Fast, None, 0)
        };
        let CachedBlock::Compiled { entry } = block else {
            panic!("must compile")
        };
        let start = entry.0 as usize;
        let sites: Vec<_> = (0..4096)
            .filter_map(|off| {
                let s = x86jit_core::codemap::lookup_trap(start + off)?;
                (s.host_offset as usize == off).then_some(s)
            })
            .collect();
        let lane_sites = if cfg!(target_arch = "aarch64") { 1 } else { 2 };
        assert_eq!(sites.len(), if region { 1 + lane_sites } else { 1 });
        assert_eq!(
            sites
                .iter()
                .filter(|s| s.guest_rip == HIGH_BASE
                    && s.access == GuestAccessKind::Read
                    && s.width == 8)
                .count(),
            1
        );
        if region {
            assert_eq!(
                sites
                    .iter()
                    .filter(|s| s.guest_rip == second_rip
                        && s.access == GuestAccessKind::Write
                        && s.width == if cfg!(target_arch = "aarch64") { 16 } else { 8 })
                    .count(),
                lane_sites
            );
        }
        for s in &sites {
            assert_eq!(
                x86jit_core::codemap::lookup(start + s.host_offset as usize),
                Some(s.guest_rip)
            );
        }
    }
}

#[test]
fn guest_trap_sites_preserve_dead_loads() {
    // Guard against a Cranelift upgrade deleting a dead guest probe. In 0.115.1
    // these loads happen to survive even with notrap (src/inst_predicates.rs,
    // has_lowering_side_effect); their fault must remain observable in any version.
    let mut missing = Vec::new();
    for bytes in [
        // test [rdi], rax; xor eax, eax; hlt: all TEST flags are overwritten.
        &[0x48, 0x85, 0x07, 0x31, 0xc0, 0xf4][..],
        // mov rax, [rdi]; mov eax, 42; hlt: region GPR Variable is redefined.
        &[0x48, 0x8b, 0x07, 0xb8, 42, 0, 0, 0, 0xf4][..],
    ] {
        for region in [false, true] {
            let (ctx, _, _) = compile(bytes, region, HostTarget::Native);
            if ctx.compiled_code().unwrap().buffer.traps().is_empty() {
                missing.push((bytes, region));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "dead guest loads must still have trap sites: {missing:02x?}"
    );
}

#[test]
fn guest_trap_sites_synthetic_ir_uses_block_rip() {
    let mut ir = high_block(&[0x48, 0x8b, 0x07, 0xf4]);
    ir.ops.retain(|op| !matches!(op, IrOp::InsnStart { .. }));
    for region in [false, true] {
        let jit = JitBackend::new();
        let block = if region {
            jit.materialize_region(
                &IrRegion {
                    entry: HIGH_BASE,
                    blocks: vec![ir.clone()],
                    has_loop: false,
                },
                MemConsistency::Fast,
                None,
                0,
            )
        } else {
            jit.materialize(&ir, MemConsistency::Fast, None, 0)
        };
        let CachedBlock::Compiled { entry } = block else {
            panic!("must compile")
        };
        assert!((0..4096).any(
            |off| x86jit_core::codemap::lookup_trap(entry.0 as usize + off).is_some_and(|s| s
                .host_offset
                as usize
                == off
                && s.guest_rip == HIGH_BASE
                && s.access == GuestAccessKind::Read
                && s.width == 8)
        ));
    }
}

#[test]
fn guest_trap_sites_hot_extending_loads() {
    for bytes in [
        &[0x48, 0x0f, 0xbe, 0x07, 0xf4][..], // movsx rax, byte [rdi]
        &[0x48, 0x0f, 0xbf, 0x07, 0xf4][..],
        &[0x48, 0x63, 0x07, 0xf4][..], // movsxd rax, dword [rdi]
        &[0x48, 0x0f, 0xb6, 0x07, 0xf4][..],
        &[0x48, 0x0f, 0xb7, 0x07, 0xf4][..],
    ] {
        for region in [false, true] {
            compile(bytes, region, HostTarget::Native);
        }
    }
}

#[test]
fn guest_trap_sites_engine_permute_loads_are_trusted() {
    for bytes in [
        &[0xc4, 0xe2, 0x75, 0x36, 0xc2, 0xf4][..], // vpermd ymm0, ymm1, ymm2
        &[0xc4, 0xe2, 0x71, 0x0c, 0xc2, 0xf4][..], // vpermilps xmm0, xmm1, xmm2
        &[0xc4, 0xe2, 0x71, 0x0d, 0xc2, 0xf4][..], // vpermilpd xmm0, xmm1, xmm2
    ] {
        let (ctx, _, accesses) = compile(bytes, false, HostTarget::Native);
        assert_eq!(accesses, 0);
        assert!(ctx.compiled_code().unwrap().buffer.traps().is_empty());
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn guest_trap_sites_aarch64_single_16_byte_transfer() {
    for bytes in [&[0x0f, 0x10, 0x07, 0xf4][..], &[0x0f, 0x11, 0x07, 0xf4][..]] {
        for region in [false, true] {
            let (ctx, _, _) = compile(bytes, region, HostTarget::Native);
            let cc = ctx.compiled_code().unwrap();
            let width16: Vec<_> = cc
                .buffer
                .traps()
                .iter()
                .filter(|trap| {
                    let word = u32::from_le_bytes(
                        cc.code_buffer()[trap.offset as usize..trap.offset as usize + 4]
                            .try_into()
                            .unwrap(),
                    );
                    word & 0xff00_0000 == 0xa900_0000 || word & 0xff80_0000 == 0x3d80_0000
                })
                .collect();
            assert_eq!(
                cc.buffer.traps().len(),
                if bytes[1] == 0x11 { 3 } else { 1 }
            );
            assert_eq!(width16.len(), 1);
            let trap = width16[0];
            let word = u32::from_le_bytes(
                cc.code_buffer()[trap.offset as usize..trap.offset as usize + 4]
                    .try_into()
                    .unwrap(),
            );
            // AArch64 integer pair (LDP/STP) or SIMD Q (LDR/STR), each transferring
            // all 16 bytes in one instruction. Ignore registers, offset and L bit.
            assert!(
                word & 0xff00_0000 == 0xa900_0000 || word & 0xff80_0000 == 0x3d80_0000,
                "expected 16-byte transfer: {word:08x}\n{}",
                cc.vcode.as_deref().unwrap()
            );
        }
    }
}

#[test]
fn guest_trap_sites_x64_merged_load_keeps_owner() {
    use cranelift::codegen::{isa, settings};
    let Ok(builder) = isa::lookup_by_name("x86_64") else {
        // Enable the dependency's x86 backend when running this on an ARM host:
        // CARGO_FEATURE_X86=1 RUSTFLAGS='--cfg feature="x86"' cargo test -p x86jit-cranelift --lib guest_trap_sites_x64
        return;
    };
    for opt in [OptLevel::None, OptLevel::Speed] {
        let mut flags = settings::builder();
        flags.set("opt_level", opt.flag()).unwrap();
        let isa = builder.finish(settings::Flags::new(flags)).unwrap();
        let mut ctx = memory_fixture(&*isa, guest_mem_flags(GuestAccessKind::Read, 8));
        let metadata =
            crate::prepare_guest_traps(&mut ctx.func, &[HIGH_BASE, HIGH_BASE + 3]).unwrap();
        ctx.compile(&*isa, &mut Default::default()).unwrap();
        let cc = ctx.compiled_code().unwrap();
        let sites = crate::collect_guest_traps(cc, &metadata).unwrap();
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].guest_rip, HIGH_BASE);
        let sources = cc.buffer.get_srclocs_sorted();
        let source = sources
            .iter()
            .find(|s| s.start <= sites[0].host_offset && sites[0].host_offset < s.end)
            .unwrap();
        assert_eq!(
            source.loc.bits(),
            1,
            "pin a merged load carrying the consumer's source location"
        );

        for region in [false, true] {
            // mov rax,[rdi]; add rbx,rax; xor ecx,ecx; mov eax,42; hlt.
            // Include the register redefinition that permits load sinking.
            let (ctx, rips, _) = compile_for_isa(
                &[
                    0x48, 0x8b, 0x07, 0x48, 0x01, 0xc3, 0x31, 0xc9, 0xb8, 42, 0, 0, 0, 0xf4,
                ],
                region,
                &*isa,
            );
            let cc = ctx.compiled_code().unwrap();
            assert_eq!(cc.buffer.traps().len(), 1);
            let sources = cc.buffer.get_srclocs_sorted();
            let trap = &cc.buffer.traps()[0];
            let source = sources
                .iter()
                .find(|s| s.start <= trap.offset && trap.offset < s.end)
                .unwrap();
            assert!(!source.loc.is_default());
            assert!([HIGH_BASE, HIGH_BASE + 3].contains(&rips[source.loc.bits() as usize]));
            // This lifted shape currently keeps the load separate: flag/register
            // stores prevent sinking. The record must retain the load's owner
            // even if a later Cranelift version merges it into the ADD.
        }
    }
}

fn memory_fixture(isa: &dyn cranelift::codegen::isa::TargetIsa, flags: MemFlagsData) -> Context {
    let mut ctx = Context::new();
    ctx.func.signature.call_conv = isa.default_call_conv();
    ctx.func
        .signature
        .params
        .extend([AbiParam::new(types::I64); 2]);
    ctx.func.signature.returns.push(AbiParam::new(types::I64));
    let mut fbctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    let params = builder.block_params(entry).to_vec();
    builder.set_srcloc(ir::SourceLoc::new(0));
    let value = builder.ins().load(types::I64, flags, params[0], 0);
    builder.set_srcloc(ir::SourceLoc::new(1));
    let result = builder.ins().bor(params[1], value);
    builder.ins().return_(&[result]);
    builder.seal_all_blocks();
    builder.finalize(isa.frontend_config());
    ctx
}

#[test]
fn wide_transfer_materializes_flags_once_in_both_modes() {
    // Deferred ADD, SHL, SHLD and SHRD writers followed by a split transfer.
    // SDM Vol 2B, SAL/SAR/SHL/SHR and SHLD/SHRD, Operation: masked count 3
    // always writes flags, so recovery needs no count-zero branch.
    // Count hot-path flag stores with optimization disabled as well as enabled.
    let mut failures = Vec::new();
    for writer in [
        &[0x48, 0x01, 0xd8][..],      // add rax,rbx
        &[0x48, 0xc1, 0xe0, 3],       // shl rax,3
        &[0x48, 0x0f, 0xa4, 0xd8, 3], // shld rax,rbx,3
        &[0x48, 0x0f, 0xac, 0xd8, 3], // shrd rax,rbx,3
    ] {
        let mut bytes = writer.to_vec();
        bytes.extend_from_slice(&[0xc5, 0xfe, 0x6f, 0x07, 0x31, 0xc0, 0xf4]);
        for region in [false, true] {
            for opt in [OptLevel::None, OptLevel::Speed] {
                let (ctx, _, _) = compile_at(&bytes, region, HostTarget::Native, opt);
                let offsets = cpu_offsets();
                let fields = [
                    offsets.cf,
                    offsets.pf_src,
                    offsets.af_src,
                    offsets.zf,
                    offsets.sf,
                    offsets.of,
                ];
                let mut stores = [0; 6];
                for block in ctx
                    .func
                    .layout
                    .blocks()
                    .filter(|&b| !ctx.func.layout.is_cold(b))
                {
                    for inst in ctx.func.layout.block_insts(block) {
                        if let ir::InstructionData::Store { offset, .. } = ctx.func.dfg.insts[inst]
                        {
                            if let Some(i) = fields.iter().position(|&o| o == i32::from(offset)) {
                                if ctx.func.srcloc(inst).bits() == 1 {
                                    stores[i] += 1;
                                }
                            }
                        }
                    }
                }
                let expected = if writer[1] == 0x01 {
                    [1; 6]
                } else {
                    [1, 1, 0, 1, 1, 1]
                };
                if stores != expected {
                    failures.push(format!("region={region}, {opt:?}, writer={writer:x?}: stores={stores:?}, expected={expected:?}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "one recovery per flag across the split transfer: {failures:?}"
    );
}

#[test]
fn region_fault_flush_only_dirty_gprs() {
    // mov eax,42; mov [rdi],rax; mov [rsi],rax; hlt. No GPR changes
    // between accesses: only RAX needs one store, even at opt_level=none.
    for opt in [OptLevel::None, OptLevel::Speed] {
        let (ctx, _, _) = compile_at(
            &[0xb8, 42, 0, 0, 0, 0x48, 0x89, 0x07, 0x48, 0x89, 0x06, 0xf4],
            true,
            HostTarget::Native,
            opt,
        );
        let cpu = ctx
            .func
            .dfg
            .block_params(ctx.func.layout.entry_block().unwrap())[0];
        let offsets = cpu_offsets();
        let mut stores = [0; 16];
        for inst in ctx
            .func
            .layout
            .blocks()
            .flat_map(|b| ctx.func.layout.block_insts(b))
        {
            if let ir::InstructionData::Store { args, offset, .. } = ctx.func.dfg.insts[inst] {
                if ctx.func.dfg.resolve_aliases(args[1]) == cpu {
                    if let Some(i) = (0..16).find(|&i| offsets.gpr(i) == i32::from(offset)) {
                        stores[i] += 1;
                    }
                }
            }
        }
        // A bounds-check exit before the first access also flushes RAX. The
        // successful path and subsequent exits must not store any clean GPR.
        // The first cold crossing probe also needs instruction-entry RAX.
        assert_eq!(
            stores[0],
            if cfg!(target_arch = "aarch64") { 3 } else { 2 },
            "{opt:?}: {stores:?}"
        );
        assert_eq!(&stores[1..], &[0; 15]);
    }
}

#[test]
fn guest_trap_sites_refuse_missing_owner_without_panicking() {
    let jit = JitBackend::build_jit(HostTarget::Native, OptLevel::Speed);
    let mut ctx = memory_fixture(jit.module.isa(), guest_mem_flags(GuestAccessKind::Read, 8));
    assert_eq!(
        crate::prepare_guest_traps(&mut ctx.func, &[]),
        Err(crate::CompileError::Unavailable)
    );
    for inst in ctx
        .func
        .layout
        .blocks()
        .flat_map(|b| ctx.func.layout.block_insts(b))
        .collect::<Vec<_>>()
    {
        ctx.func.set_srcloc(inst, ir::SourceLoc::default());
    }
    assert_eq!(
        crate::prepare_guest_traps(&mut ctx.func, &[HIGH_BASE]),
        Err(crate::CompileError::Unavailable)
    );
}

#[test]
fn guest_trap_sites_reject_unknown_machine_codes() {
    for flags in [
        MemFlagsData::new(),
        MemFlagsData::new().with_trap_code(ir::TrapCode::user(250)),
    ] {
        let jit = JitBackend::build_jit(HostTarget::Native, OptLevel::None);
        let mut ctx = memory_fixture(jit.module.isa(), flags);
        ctx.compile(jit.module.isa(), &mut Default::default())
            .unwrap();
        let refusal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::collect_guest_traps(ctx.compiled_code().unwrap(), &[])
        }));
        assert_eq!(refusal.expect("unknown trap codes must not panic"), Err(()));
    }
}

#[test]
fn guest_trap_sites_reject_unknown_preparation_codes_without_panicking() {
    let jit = JitBackend::build_jit(HostTarget::Native, OptLevel::None);
    let mut ctx = memory_fixture(
        jit.module.isa(),
        MemFlagsData::new().with_trap_code(ir::TrapCode::user(250)),
    );
    assert_eq!(
        crate::prepare_guest_traps(&mut ctx.func, &[HIGH_BASE]),
        Err(crate::CompileError::Unavailable)
    );
}

#[test]
fn guest_trap_sites_audit_detects_trusted_guest_access() {
    let jit = JitBackend::build_jit(HostTarget::Native, OptLevel::None);
    let ctx = memory_fixture(jit.module.isa(), MemFlagsData::trusted());
    let host = ctx
        .func
        .dfg
        .block_params(ctx.func.layout.entry_block().unwrap())[0];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        audit_guest_accesses(&ctx.func, &[host])
    }));
    assert!(
        result.is_err(),
        "a trusted guest load must not disappear from the audit"
    );
}

#[test]
fn guest_trap_sites_repeated_read_counts_optimized_ir() {
    for region in [false, true] {
        let (ctx, _, accesses) = compile(
            &[0x48, 0x8b, 0x07, 0x48, 0x8b, 0x1f, 0xf4],
            region,
            HostTarget::Native,
        );
        assert_eq!(accesses, 2);
        // The pre-access RAX store separates the reads, so both retain owners.
        if region {
            let traps = ctx.compiled_code().unwrap().buffer.traps();
            assert_eq!(traps.len(), 2);
            assert_eq!(traps[0].code, ir::TrapCode::user(1).unwrap());
            assert_eq!(traps[1].code, ir::TrapCode::user(2).unwrap());
        }
        // Two identical loads before any architectural write, within one owner.
        // This IR-only shape isolates dominated-load elimination from GPR sync.
        let mut block = high_block(&[0xf4]);
        block.temp_count = 3;
        block.ops = vec![
            IrOp::InsnStart {
                guest_addr: HIGH_BASE,
            },
            IrOp::Load {
                fault: Default::default(),
                dst: 0,
                addr: Val::Imm(0x8000),
                size: 8,
            },
            IrOp::Load {
                fault: Default::default(),
                dst: 1,
                addr: Val::Imm(0x8000),
                size: 8,
            },
            IrOp::Add {
                dst: 2,
                a: Val::Temp(0),
                b: Val::Temp(1),
                size: 8,
                set_flags: FlagMask::NONE,
            },
            IrOp::WriteReg {
                reg: Reg::Rax,
                src: Val::Temp(2),
                size: 8,
            },
            IrOp::Hlt,
        ];
        let jit = JitBackend::build_jit(HostTarget::Native, OptLevel::Speed);
        let (ctx, _, original) = compile_ir_for_isa(block, &[], region, jit.module.isa());
        assert_eq!(original, 2);
        assert_eq!(ctx.compiled_code().unwrap().buffer.traps().len(), 1);
    }
}

#[test]
fn guest_trap_sites_lifter_capped_blocks_never_overflow() {
    // Repeated vector chunks share descriptors; RMWs and indirect calls can
    // contribute multiple descriptors for one instruction owner.
    for (insn, count, terminal) in [
        (&[0x48, 0x8b, 0x07][..], 250, &[0xf4][..]),
        (&[0xf3, 0x0f, 0x51, 0x07][..], 250, &[0xf4][..]),
        (
            &[0xc5, 0xfe, 0x6f, 0x07, 0xc5, 0xfe, 0x7f, 0x06][..],
            128,
            &[0xf4][..],
        ),
        (&[0x48, 0x83, 0x07, 1][..], 125, &[0xf4][..]),
        (&[0xf0, 0x48, 0xf7, 0x1f][..], 125, &[0xf4][..]),
        (&[0x48, 0x8b, 0x07][..], 249, &[0xff, 0x17][..]),
        (&[0x48, 0x8b, 0x07][..], 249, &[0xc3][..]),
    ] {
        let mut bytes = insn.repeat(count);
        bytes.extend(terminal);
        let first = high_block(&bytes);
        assert!(
            first.guest_len < bytes.len() as u32,
            "lifter must cap this sequence"
        );
        let mut offset = 0;
        while offset < bytes.len() {
            let block = high_block(&bytes[offset..]);
            for target in [HostTarget::Baseline, HostTarget::Native] {
                for region in [false, true] {
                    for opt in [OptLevel::None, OptLevel::Speed] {
                        let jit = JitBackend::build_jit(target, opt);
                        compile_ir_for_isa_with_split_stores(
                            high_block(&bytes[offset..]),
                            &bytes[offset..],
                            region,
                            jit.module.isa(),
                            false,
                        );
                    }
                }
            }
            offset += block.guest_len as usize;
        }
    }
}

#[test]
fn guest_trap_sites_oversized_synthetic_ir_is_refused_without_panicking() {
    let mut block = high_block(&[0x48, 0x8b, 0x07, 0xf4]);
    let load = IrOp::Load {
        fault: Default::default(),
        dst: 0,
        addr: Val::Imm(0x8000),
        size: 8,
    };
    block.ops.clear();
    for index in 0..250 {
        block.ops.push(IrOp::InsnStart {
            guest_addr: HIGH_BASE + index * 3,
        });
        block.ops.push(load.clone());
    }
    block.icount = 250;
    block.guest_len = 750;
    for opt in [OptLevel::None, OptLevel::Speed] {
        let jit = JitBackend::with_opt_level(opt);
        assert!(
            matches!(jit.materialize(&block, MemConsistency::Fast, None, 0), CachedBlock::Interpreted(ir) if ir.guest_len == 750)
        );
        assert!(!jit.code_space_exhausted());
        assert!(matches!(
            jit.materialize(
                &high_block(&[0x48, 0x8b, 0x07, 0xf4]),
                MemConsistency::Fast,
                None,
                0
            ),
            CachedBlock::Compiled { .. }
        ));
    }
}

#[test]
fn alignment_checks_share_one_cold_exception_tail() {
    // Eight aligned accesses must not duplicate register/flag recovery exits.
    // Address and state semantics are pinned by address_exceptions on every tier.
    let mut bytes = [0x0f, 0x28, 0x03, 0x0f, 0x29, 0x07].repeat(4);
    bytes.push(0xf4);
    for region in [false, true] {
        for target in [HostTarget::Baseline, HostTarget::Native] {
            for opt in [OptLevel::None, OptLevel::Speed] {
                let (ctx, _, _) = compile_at(&bytes, region, target, opt);
                let tails = ctx
                    .func
                    .layout
                    .blocks()
                    .flat_map(|b| ctx.func.layout.block_insts(b))
                    .filter(|inst| {
                        matches!(ctx.func.dfg.insts[*inst],
                        ir::InstructionData::Store { offset, .. }
                            if i32::from(offset) == MEMCTX_EXCEPTION_VECTOR)
                    })
                    .count();
                assert_eq!(tails, 1, "region={region}: {}", ctx.func.display());
            }
        }
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn alignment_hot_path_has_only_mask_and_branch() {
    // Test actual encoded instructions, including a PIE-sized RIP and a region
    // back-edge. RIP materialisation must not sit between AND and CBNZ.
    let cases = [
        (false, &[0x0f, 0x28, 0x06, 0x0f, 0x29, 0x07, 0xf4][..]),
        (
            true,
            &[0x0f, 0x28, 0x06, 0x0f, 0x29, 0x07, 0xff, 0xc9, 0x75, 0xf6][..],
        ),
    ];
    for (region, bytes) in cases {
        for opt in [OptLevel::None, OptLevel::Speed] {
            let mut block = high_block(bytes);
            for op in &mut block.ops {
                if let IrOp::Branch {
                    taken, fallthrough, ..
                } = op
                {
                    *taken += HIGH_BASE - BASE;
                    *fallthrough += HIGH_BASE - BASE;
                }
            }
            let mut control = block.clone();
            control
                .ops
                .retain(|op| !matches!(op, IrOp::CheckAlign { .. }));
            let jit = JitBackend::build_jit(HostTarget::Native, opt);
            let (ctx, _, _) = compile_ir_for_isa(block, bytes, region, jit.module.isa());
            let cc = ctx.compiled_code().unwrap();
            let words: Vec<_> = cc
                .code_buffer()
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let mut checks = 0;
            for pair in words.windows(2) {
                if pair[0] & 0xffff_fc00 == 0x9240_0c00 {
                    // AND Xd,Xn,#15.
                    checks += 1;
                    assert_eq!(
                        pair[1] & 0x7f00_0000,
                        0x3500_0000,
                        "AND must be followed immediately by CBNZ: {:?}",
                        cc.vcode
                    );
                    assert_eq!(pair[0] & 31, pair[1] & 31);
                }
            }
            assert_eq!(checks, 2, "{:?}", cc.vcode);
            if region {
                let (plain, _, _) = compile_ir_for_isa(control, bytes, true, jit.module.isa());
                assert_eq!(
                    loop_register_moves(cc.code_buffer()),
                    loop_register_moves(plain.compiled_code().unwrap().code_buffer()),
                    "alignment must add no loop register shuffles: {:?}",
                    cc.vcode
                );
            }
            if let Ok(dir) = std::env::var("X86JIT_ALIGNMENT_DUMP_DIR") {
                std::fs::create_dir_all(&dir).unwrap();
                let name = format!("{dir}/{}-{opt:?}", if region { "region" } else { "block" });
                std::fs::write(format!("{name}.bin"), cc.code_buffer()).unwrap();
                std::fs::write(format!("{name}.vcode"), cc.vcode.as_deref().unwrap()).unwrap();
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn loop_register_moves(code: &[u8]) -> usize {
    let words: Vec<_> = code
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    for (i, word) in words.iter().enumerate() {
        if word & 0xffff_0000 == 0xd65f_0000 {
            break;
        }
        if word & 0xfc00_0000 == 0x1400_0000 {
            let delta = ((*word << 6) as i32 >> 6) as isize;
            if delta < 0 && i as isize + delta >= 0 {
                let start = (i as isize + delta) as usize;
                return words[start..=i]
                    .iter()
                    .filter(|w| **w & 0x7fe0_ffe0 == 0x2a00_03e0)
                    .count();
            }
        }
    }
    panic!("expected a region back-edge");
}

#[test]
#[ignore = "diagnostic dump"]
fn dump_split_store_paths() {
    let jit = JitBackend::build_jit(HostTarget::Native, OptLevel::Speed);
    for (name, store) in [
        ("scalar8", &[0x48, 0x89, 0x07][..]),
        ("vector16", &[0xf3, 0x0f, 0x7f, 0x07][..]),
        ("vector32", &[0xc5, 0xfe, 0x7f, 0x07][..]),
        ("vector64", &[0x62, 0xf1, 0x7e, 0x48, 0x7f, 0x07][..]),
    ] {
        for count in [1, 8] {
            for looping in [false, true] {
                let mut bytes = store.repeat(count);
                if looping {
                    bytes.extend_from_slice(&[0xff, 0xc9, 0x75, (-(bytes.len() as i8 + 4)) as u8]);
                } else {
                    bytes.push(0xf4);
                }
                for region in [false, true] {
                    let mut block = high_block(&bytes);
                    for op in &mut block.ops {
                        if let IrOp::Branch {
                            taken, fallthrough, ..
                        } = op
                        {
                            *taken += HIGH_BASE - BASE;
                            *fallthrough += HIGH_BASE - BASE;
                        }
                    }
                    let (ctx, _, _) = compile_ir_for_isa(block, &bytes, region, jit.module.isa());
                    let cc = ctx.compiled_code().unwrap();
                    let dir = std::env::var("X86JIT_STORE_DUMP_DIR").unwrap();
                    std::fs::create_dir_all(&dir).unwrap();
                    let path = format!(
                        "{dir}/{name}-{count}-{}{}",
                        if region { "region" } else { "block" },
                        if looping { "-loop" } else { "" }
                    );
                    std::fs::write(format!("{path}.clif"), ctx.func.display().to_string()).unwrap();
                    std::fs::write(format!("{path}.vcode"), cc.vcode.as_deref().unwrap()).unwrap();
                    std::fs::write(format!("{path}.bin"), cc.code_buffer()).unwrap();
                }
            }
        }
    }
}

#[test]
fn split_store_preflight_reuses_only_dominating_proofs() {
    let store = [0xf3, 0x0f, 0x7f, 0x07]; // movdqu [rdi],xmm0
    let mut repeated = store.repeat(8);
    repeated.push(0xf4);
    let mut changed = store.to_vec();
    changed.extend_from_slice(&[0x48, 0xff, 0xc7]); // inc rdi
    changed.extend_from_slice(&store);
    changed.push(0xf4);
    let mut helper = store.to_vec();
    helper.extend_from_slice(&[0x0f, 0xa2]); // cpuid: can reload guest registers
    helper.extend_from_slice(&store);
    helper.push(0xf4);
    for region in [false, true] {
        for opt in [OptLevel::None, OptLevel::Speed] {
            for (bytes, expected) in [(&repeated, 2), (&changed, 4), (&helper, 4)] {
                let (ctx, _, _) = compile_at(bytes, region, HostTarget::Native, opt);
                let probes = ctx
                    .func
                    .layout
                    .blocks()
                    .flat_map(|b| ctx.func.layout.block_insts(b))
                    .filter(|&i| {
                        matches!(ctx.func.dfg.insts[i], ir::InstructionData::AtomicRmw { .. })
                            && ctx.func.dfg.insts[i]
                                .memflags()
                                .and_then(|f| ctx.func.dfg.mem_flags[f].trap_code())
                                .is_some()
                    })
                    .count();
                assert_eq!(
                    probes,
                    expected,
                    "region={region}, {opt:?}: {}",
                    ctx.func.display()
                );
            }
        }
    }
}

#[test]
fn split_store_preflight_omits_proven_alignment_only() {
    // Backing preserves alignment modulo 16. No proof about 32/64-byte host alignment.
    let cases: &[(&[u8], usize)] = &[
        (&[0x88, 0x07, 0xf4], 0), // byte stores never cross
        (
            &[0x48, 0x83, 0xe7, 0xf0, 0xf3, 0x0f, 0x7f, 0x47, 0x10, 0xf4],
            0,
        ),
        (
            &[0x48, 0x83, 0xe7, 0xf0, 0xf3, 0x0f, 0x7f, 0x47, 0x11, 0xf4],
            2,
        ),
        (&[0xbf, 0, 0x40, 0, 0, 0xf3, 0x0f, 0x7f, 0x07, 0xf4], 0),
        (&[0xbf, 0, 0x40, 0, 0, 0xc5, 0xfe, 0x7f, 0x07, 0xf4], 2),
        // A prior aligned load proves RDI aligned even across instruction boundaries.
        (&[0x0f, 0x28, 0x07, 0xf3, 0x0f, 0x7f, 0x47, 0x10, 0xf4], 0),
    ];
    for region in [false, true] {
        for opt in [OptLevel::None, OptLevel::Speed] {
            for &(bytes, expected) in cases {
                let (ctx, _, _) = compile_at(bytes, region, HostTarget::Native, opt);
                let probes = ctx
                    .func
                    .layout
                    .blocks()
                    .flat_map(|b| ctx.func.layout.block_insts(b))
                    .filter(|&i| {
                        matches!(ctx.func.dfg.insts[i], ir::InstructionData::AtomicRmw { .. })
                            && ctx.func.dfg.insts[i]
                                .memflags()
                                .and_then(|f| ctx.func.dfg.mem_flags[f].trap_code())
                                .is_some()
                    })
                    .count();
                assert_eq!(
                    probes,
                    expected,
                    "region={region}, {opt:?}, bytes={bytes:x?}: {}",
                    ctx.func.display()
                );
            }
        }
    }
}

#[test]
fn split_store_policy_controls_probes_on_both_isas() {
    use cranelift::codegen::{isa, settings};
    for target in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for opt in [OptLevel::None, OptLevel::Speed] {
            let mut flags = settings::builder();
            flags.set("opt_level", opt.flag()).unwrap();
            flags.set("preserve_frame_pointers", "true").unwrap();
            flags.set("enable_probestack", "false").unwrap();
            let isa = isa::lookup(target.parse().unwrap())
                .unwrap()
                .finish(settings::Flags::new(flags))
                .unwrap();
            for (name, bytes) in [
                ("scalar8", &[0x48, 0x89, 0x07, 0xf4][..]),
                ("vector16", &[0xf3, 0x0f, 0x7f, 0x07, 0xf4][..]),
                ("vector32", &[0xc5, 0xfe, 0x7f, 0x07, 0xf4][..]),
                ("vector64", &[0x62, 0xf1, 0x7e, 0x48, 0x7f, 0x07, 0xf4][..]),
            ] {
                for region in [false, true] {
                    for precise in [false, true] {
                        let (ctx, _, _) = compile_ir_for_isa_with_split_stores(
                            high_block(bytes),
                            bytes,
                            region,
                            isa.as_ref(),
                            precise,
                        );
                        let probes = ctx
                            .func
                            .layout
                            .blocks()
                            .flat_map(|b| ctx.func.layout.block_insts(b))
                            .filter(|&i| {
                                matches!(
                                    ctx.func.dfg.insts[i],
                                    ir::InstructionData::AtomicRmw { .. }
                                ) && ctx.func.dfg.insts[i]
                                    .memflags()
                                    .and_then(|f| ctx.func.dfg.mem_flags[f].trap_code())
                                    .is_some()
                            })
                            .count();
                        // This diagnostic uses native-host lowering with two machine
                        // encoders; runtime host selection is covered separately.
                        assert_eq!(
                            probes,
                            if precise && (cfg!(target_arch = "aarch64") || name != "scalar8") {
                                2
                            } else {
                                0
                            }
                        );
                        if !precise {
                            if let Some(dir) = std::env::var_os("X86JIT_STORE_DUMP_DIR") {
                                let dir = std::path::PathBuf::from(dir);
                                std::fs::create_dir_all(&dir).unwrap();
                                let file = format!(
                                    "{target}-{opt:?}-{name}-{}.bin",
                                    if region { "region" } else { "block" }
                                );
                                std::fs::write(
                                    dir.join(file),
                                    ctx.compiled_code().unwrap().code_buffer(),
                                )
                                .unwrap();
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn precise_split_store_descriptor_limit_interprets_without_exhausting_code() {
    let bytes = [0xc5, 0xfe, 0x7f, 0x07].repeat(249);
    let ir = high_block(&bytes);
    for region in [false, true] {
        let jit = JitBackend::with_opt_level(OptLevel::None);
        let materialize = |block: &IrBlock, precise| {
            if region {
                jit.materialize_region_with_split_stores(
                    &IrRegion {
                        entry: block.guest_start,
                        blocks: vec![block.clone()],
                        has_loop: false,
                    },
                    MemConsistency::Fast,
                    None,
                    0,
                    0,
                    precise,
                )
            } else {
                jit.materialize_with_split_stores(block, MemConsistency::Fast, None, 0, 0, precise)
            }
        };
        assert!(matches!(
            materialize(&ir, false),
            CachedBlock::Compiled { .. }
        ));
        assert!(matches!(
            materialize(&ir, true),
            CachedBlock::Interpreted(_)
        ));
        assert!(!jit.code_space_exhausted());
        assert!(matches!(
            materialize(&high_block(&[0xc5, 0xfe, 0x7f, 0x07, 0xf4]), true),
            CachedBlock::Compiled { .. }
        ));
    }
}
