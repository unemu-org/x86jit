//! Fence bytes use the production translator at Speed; alias controls use hand-built IR.
use super::*;
use cranelift::codegen::{isa, settings, Context};
use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
use x86jit_core::{Prot, RegionKind, Vm, VmConfig};

const DMB: &[u8] = &[0xbf, 0x3b, 0x03, 0xd5];
const MFENCE: &[u8] = &[0x0f, 0xae, 0xf0];
const LOAD: &[u8] = &[0x48, 0x8b, 0x03]; // mov rax,[rbx]
const STORE: &[u8] = &[0x48, 0xc7, 0x03, 1, 0, 0, 0]; // mov qword [rbx],1

fn compile(bytes: &[u8], target: &str, tier: MemConsistency, opt: crate::OptLevel) -> Context {
    let mut vm = Vm::new(VmConfig::flat(0x10000));
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, bytes).unwrap();
    let ir = lift_block(&vm.mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    compile_ir(&ir, target, tier, opt)
}

fn compile_ir(ir: &IrBlock, target: &str, tier: MemConsistency, opt: crate::OptLevel) -> Context {
    let mut flags = settings::builder();
    flags.set("opt_level", opt.flag()).unwrap();
    let isa = isa::lookup(target.parse().unwrap())
        .unwrap()
        .finish(settings::Flags::new(flags))
        .unwrap();
    let mut ctx = Context::new();
    ctx.func
        .signature
        .params
        .extend([AbiParam::new(types::I64); 2]);
    ctx.func.signature.returns.push(AbiParam::new(types::I64));
    let mut fbctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
    let mut signature = ir::Signature::new(isa.default_call_conv());
    signature.params = vec![AbiParam::new(types::I64); 6];
    signature.returns.push(AbiParam::new(types::I64));
    let dummy = builder.import_signature(signature);
    let mut signature = ir::Signature::new(isa.default_call_conv());
    signature.params = vec![AbiParam::new(types::I64); 3];
    let watch = builder.import_signature(signature);
    let mut helpers = Helpers::all_dummy(&mut || (dummy, 0, 0), (watch, 0, 0));
    let mut signature = ir::Signature::new(isa.default_call_conv());
    signature.params = vec![AbiParam::new(types::I64); 8];
    signature.returns.push(AbiParam::new(types::I64));
    helpers.string = (builder.import_signature(signature), 0, 0);
    let mut signature = ir::Signature::new(isa.default_call_conv());
    signature.params.push(AbiParam::new(types::I64));
    helpers.cpuid = (builder.import_signature(signature), 0, 0);
    for (slot, params) in [
        (&mut helpers.x87, 6),
        (&mut helpers.fxstate, 5),
        (&mut helpers.vmaskmov_mem, 11),
        (&mut helpers.vec_maskmov_mem, 9),
        (&mut helpers.gather, 13),
        (&mut helpers.vpermt2_mem, 12),
        (&mut helpers.vperm1_mem, 11),
        (&mut helpers.gf2p8_mem, 11),
        (&mut helpers.vpmov_narrow_mem, 8),
        (&mut helpers.fma_mem, 18),
        (&mut helpers.broadcast_lane_mem, 11),
    ] {
        let mut signature = ir::Signature::new(isa.default_call_conv());
        signature.params = vec![AbiParam::new(types::I64); params];
        signature.returns.push(AbiParam::new(types::I64));
        *slot = (builder.import_signature(signature), 0, 0);
    }
    translate_block_for_target(
        &mut builder,
        ir,
        &x86jit_core::jit_abi::cpu_offsets(),
        &mut || 0,
        helpers,
        tier,
        None,
        0,
        false,
        target.starts_with("aarch64"),
    );
    builder.finalize();
    ctx.set_disasm(true);
    ctx.compile(&*isa, &mut Default::default()).unwrap();
    ctx
}

fn snippet(first: &[u8], modrm: Option<u8>, second: &[u8]) -> Vec<u8> {
    let mut bytes = first.to_vec();
    if let Some(m) = modrm {
        bytes.extend([0x0f, 0xae, m]);
    }
    bytes.extend(second);
    bytes.push(0xf4);
    bytes
}

fn access_offsets(ctx: &Context, code: u8) -> Vec<usize> {
    ctx.compiled_code()
        .unwrap()
        .buffer
        .traps()
        .iter()
        .filter(|trap| Some(trap.code) == ir::TrapCode::user(code))
        .map(|trap| trap.offset as usize)
        .collect()
}

#[test]
fn mfence_machine_barrier_between_store_and_load() {
    // SDM Vol 2B §4.3 MFENCE: 0F AE F0 orders stores before subsequent loads.
    // Use different guest addresses: [rbx] and [rdx].
    for target in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for tier in [
            MemConsistency::Fast,
            MemConsistency::AcqRel,
            MemConsistency::FullTso,
        ] {
            let ctx = compile(
                &snippet(STORE, Some(0xf0), &[0x48, 0x8b, 0x02]),
                target,
                tier,
                crate::OptLevel::Speed,
            );
            let code = ctx.compiled_code().unwrap().code_buffer();
            let store = access_offsets(&ctx, 9)[0];
            let load = access_offsets(&ctx, 4)[0];
            assert!(store < load);
            let base = compile(
                &snippet(STORE, None, &[0x48, 0x8b, 0x02]),
                target,
                tier,
                crate::OptLevel::Speed,
            );
            let barrier = if target.starts_with("aarch64") {
                DMB
            } else {
                MFENCE
            };
            assert_eq!(
                code.windows(barrier.len())
                    .filter(|w| *w == barrier)
                    .count(),
                base.compiled_code()
                    .unwrap()
                    .code_buffer()
                    .windows(barrier.len())
                    .filter(|w| *w == barrier)
                    .count()
                    + 1,
                "MFENCE must add its own barrier in {target}/{tier:?}",
            );
            assert!(
                code[store..load]
                    .windows(barrier.len())
                    .any(|w| w == barrier),
                "{target}/{tier:?}: missing barrier between guest accesses\n{}",
                ctx.compiled_code().unwrap().vcode.as_ref().unwrap()
            );
        }
    }
}

#[test]
fn lfence_sfence_follow_host_and_tier_rule() {
    // SDM Vol 3A §11.2.2, §11.2.5: ordinary WB loads and stores already
    // provide LoadLoad/StoreStore ordering on x86; our non-temporal lifts use
    // ordinary stores too. ARM needs a barrier in every tier, including for helper accesses.
    for target in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for tier in [
            MemConsistency::Fast,
            MemConsistency::AcqRel,
            MemConsistency::FullTso,
        ] {
            for (modrm, op, trap) in [(0xe8, LOAD, 4), (0xf8, STORE, 9)] {
                let ctx = compile(
                    &snippet(op, Some(modrm), op),
                    target,
                    tier,
                    crate::OptLevel::Speed,
                );
                let base = compile(&snippet(op, None, op), target, tier, crate::OptLevel::Speed);
                let code = ctx.compiled_code().unwrap().code_buffer();
                if target.starts_with("aarch64") {
                    let accesses = access_offsets(&ctx, trap);
                    assert_eq!(accesses.len(), 2);
                    let base_code = base.compiled_code().unwrap().code_buffer();
                    assert_eq!(
                        code.windows(4).filter(|w| *w == DMB).count(),
                        base_code.windows(4).filter(|w| *w == DMB).count() + 1
                    );
                    assert!(code[accesses[0]..accesses[1]].windows(4).any(|w| w == DMB));
                } else {
                    // RIP changes when a fence is present, so compare barrier counts,
                    // not the instruction-boundary bookkeeping bytes.
                    for barrier in [DMB, MFENCE] {
                        assert_eq!(
                            code.windows(barrier.len())
                                .filter(|w| *w == barrier)
                                .count(),
                            base.compiled_code()
                                .unwrap()
                                .code_buffer()
                                .windows(barrier.len())
                                .filter(|w| *w == barrier)
                                .count()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn guest_loads_have_a_machine_barrier_between_them() {
    // This checks guest barrier placement only. CpuState write-through stores
    // already prevent these loads from merging without a fence; the separate
    // hand-built control below tests the alias-analysis clobber property.
    for target in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for opt in [crate::OptLevel::Speed, crate::OptLevel::SpeedAndSize] {
            let ctx = compile(
                &snippet(LOAD, Some(0xf0), &[0x48, 0x8b, 0x13]),
                target,
                MemConsistency::Fast,
                opt,
            );
            let accesses = access_offsets(&ctx, 4);
            assert_eq!(accesses.len(), 2, "both guest loads must survive");
            let barrier = if target.starts_with("aarch64") {
                DMB
            } else {
                MFENCE
            };
            let code = ctx.compiled_code().unwrap().code_buffer();
            assert!(
                code[accesses[0]..accesses[1]]
                    .windows(barrier.len())
                    .any(|w| w == barrier),
                "second load moved above fence: {}",
                ctx.compiled_code().unwrap().vcode.as_ref().unwrap()
            );
        }
    }
}

#[test]
fn fence_clobbers_alias_analysis_loads() {
    // Both values remain live, with no intervening CpuState store. The no-fence
    // control must merge, otherwise this cannot witness a memory clobber.
    for target in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for opt in [
            crate::OptLevel::for_tiering(true),
            crate::OptLevel::SpeedAndSize,
        ] {
            for fenced in [false, true] {
                // Delay register write-through until both loads finish. The
                // production Fence arm must participate in this witness too.
                let mut ops = vec![
                    IrOp::ReadReg {
                        dst: 0,
                        reg: Reg::Rbx,
                    },
                    IrOp::Load {
                        fault: x86jit_core::ir::MemoryAccess::new(8, true, false),
                        dst: 1,
                        addr: Val::Temp(0),
                        size: 8,
                    },
                ];
                if fenced {
                    ops.push(IrOp::Fence {
                        kind: FenceKind::Full,
                    });
                }
                ops.extend([
                    IrOp::Load {
                        fault: x86jit_core::ir::MemoryAccess::new(8, true, false),
                        dst: 2,
                        addr: Val::Temp(0),
                        size: 8,
                    },
                    IrOp::WriteReg {
                        reg: Reg::Rax,
                        src: Val::Temp(1),
                        size: 8,
                    },
                    IrOp::WriteReg {
                        reg: Reg::Rdx,
                        src: Val::Temp(2),
                        size: 8,
                    },
                    IrOp::Hlt,
                ]);
                let ctx = compile_ir(
                    &IrBlock {
                        ops,
                        temp_count: 3,
                        guest_start: 0x1000,
                        guest_len: 1,
                        icount: 1,
                    },
                    target,
                    MemConsistency::Fast,
                    opt,
                );
                assert_eq!(
                    access_offsets(&ctx, 4).len(),
                    if fenced { 2 } else { 1 },
                    "alias-analysis witness: {target}/{opt:?}/fenced={fenced}"
                );
                if fenced {
                    let accesses = access_offsets(&ctx, 4);
                    let code = ctx.compiled_code().unwrap().code_buffer();
                    let barrier = if target.starts_with("aarch64") {
                        DMB
                    } else {
                        MFENCE
                    };
                    assert!(code[accesses[0]..accesses[1]]
                        .windows(barrier.len())
                        .any(|w| w == barrier));
                }
            }
        }
    }
}

#[test]
fn helper_accesses_are_ordered_by_lfence_and_sfence() {
    // REP LODS/STOS execute in helpers, outside gload/gstore's tier barriers.
    // SDM Vol 2A §3.3 LFENCE; Vol 2B §4.3 SFENCE.
    for tier in [
        MemConsistency::Fast,
        MemConsistency::AcqRel,
        MemConsistency::FullTso,
    ] {
        for (helper, modrm, access, trap) in [
            (&[0xf3, 0xac][..], 0xe8, LOAD, 4),
            (&[0xf3, 0xaa][..], 0xf8, STORE, 9),
        ] {
            let ctx = compile(
                &snippet(helper, Some(modrm), access),
                "aarch64-unknown-linux-gnu",
                tier,
                crate::OptLevel::Speed,
            );
            let code = ctx.compiled_code().unwrap().code_buffer();
            let access = access_offsets(&ctx, trap)[0];
            // BLR Xn: ignore the register field. The string helper is the last
            // indirect call before the following inline guest access.
            let call = code[..access]
                .chunks_exact(4)
                .enumerate()
                .filter(|(_, bytes)| {
                    u32::from_le_bytes((*bytes).try_into().unwrap()) & 0xffff_fc1f == 0xd63f_0000
                })
                .map(|(index, _)| index * 4)
                .next_back()
                .unwrap();
            assert!(
                code[call + 4..access].windows(4).any(|w| w == DMB),
                "helper then fence lacks barrier before guest access: {tier:?}/{modrm:x}"
            );
        }
    }
}

#[test]
fn cpuid_has_an_explicit_full_barrier() {
    // SDM Vol 3A §11.3: CPUID drains preceding buffered writes.
    for target in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        for tier in [
            MemConsistency::Fast,
            MemConsistency::AcqRel,
            MemConsistency::FullTso,
        ] {
            let bytes = [STORE, &[0x0f, 0xa2], &[0x48, 0x8b, 0x02, 0xf4]].concat();
            let ctx = compile(&bytes, target, tier, crate::OptLevel::Speed);
            let code = ctx.compiled_code().unwrap().code_buffer();
            let store = access_offsets(&ctx, 9)[0];
            let load = access_offsets(&ctx, 4)[0];
            let barrier = if target.starts_with("aarch64") {
                DMB
            } else {
                MFENCE
            };
            assert!(code[store..load]
                .windows(barrier.len())
                .any(|w| w == barrier));
        }
    }
}

#[test]
fn guest_memory_helpers_follow_host_and_tier_rule() {
    // SDM Vol 3A §11.2.2 and §11.2.4.1: order ordinary guest accesses,
    // including string operations, across the helper boundary.
    let mut cases = Vec::new();
    for (name, bytes, loads, stores) in [
        ("stos", &[0xf3, 0xaa][..], false, true),
        ("movs", &[0xf3, 0xa4][..], true, true),
        ("lods", &[0xf3, 0xac][..], true, false),
        ("scas", &[0xf2, 0xae][..], true, false),
        ("cmps", &[0xf3, 0xa6][..], true, false),
        ("fld", &[0xd9, 0x03][..], true, false),
        ("fstp", &[0xd9, 0x1b][..], false, true),
        ("fld1", &[0xd9, 0xe8][..], false, false),
        ("fxsave", &[0x0f, 0xae, 0x03][..], false, true),
        ("fxrstor", &[0x0f, 0xae, 0x0b][..], true, false),
        (
            "gather",
            &[0xc4, 0xe2, 0x6d, 0x90, 0x04, 0x88][..],
            true,
            false,
        ),
        (
            "vmaskmov-load",
            &[0xc4, 0xe2, 0x71, 0x2c, 0x03][..],
            true,
            false,
        ),
        (
            "vmaskmov-store",
            &[0xc4, 0xe2, 0x71, 0x2e, 0x03][..],
            false,
            true,
        ),
    ] {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &snippet(bytes, None, &[])).unwrap();
        let ir = lift_block(&vm.mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
        cases.push((name, ir, loads, stores));
    }
    let fault = x86jit_core::ir::MemoryAccess::new(16, true, false);
    let addr = Val::Imm(0x8000);
    for (name, op, loads, stores) in [
        (
            "masked-load",
            IrOp::VMaskLoadMem {
                broadcast: None,
                fault,
                dst: 0,
                addr,
                k: 1,
                elem: 4,
                zeroing: false,
                bytes: 16,
            },
            true,
            false,
        ),
        (
            "masked-store",
            IrOp::VMaskStoreMem {
                fault,
                src: 0,
                addr,
                k: 1,
                elem: 4,
                bytes: 16,
            },
            false,
            true,
        ),
        (
            "permute-two",
            IrOp::VPermT2M {
                fault,
                dst: 0,
                idx: 1,
                addr,
                elem: 4,
                writemask: None,
                zeroing: false,
                bytes: 16,
                imode: false,
            },
            true,
            false,
        ),
        (
            "permute-one",
            IrOp::VPerm1M {
                fault,
                dst: 0,
                idx: 1,
                addr,
                elem: 4,
                bytes: 16,
                writemask: None,
                zeroing: false,
            },
            true,
            false,
        ),
        (
            "gf2p8",
            IrOp::VGf2p8M {
                fault,
                dst: 0,
                a: 1,
                addr,
                imm: 0,
                mode: 0,
                k: 0,
                zeroing: false,
                bytes: 16,
            },
            true,
            false,
        ),
        (
            "narrow-store",
            IrOp::VPmovNarrowMem {
                fault,
                src: 0,
                addr,
                from: 4,
                to: 2,
                src_width: 16,
            },
            false,
            true,
        ),
        (
            "fma",
            IrOp::VFmaM {
                fault,
                dst: 0,
                x: 0,
                y: 1,
                z: 2,
                addr,
                mem_role: 1,
                prec: FPrec::F32,
                scalar: false,
                neg_prod: false,
                neg_add: false,
                bytes: 16,
                alt_sign: 0,
                writemask: None,
                zeroing: false,
            },
            true,
            false,
        ),
        (
            "broadcast",
            IrOp::VBroadcastLaneM {
                fault,
                dst: 0,
                addr,
                chunk: 8,
                elem: 4,
                dst_width: 16,
                writemask: None,
                zeroing: false,
            },
            true,
            false,
        ),
        (
            "cas-pair",
            IrOp::AtomicCasPair {
                fault,
                addr,
                size: 16,
                len: 4,
            },
            true,
            true,
        ),
    ] {
        cases.push((
            name,
            IrBlock {
                ops: vec![op, IrOp::Hlt],
                temp_count: 0,
                guest_start: 0x1000,
                guest_len: 1,
                icount: 1,
            },
            loads,
            stores,
        ));
    }
    for target in ["aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"] {
        for tier in [
            MemConsistency::Fast,
            MemConsistency::AcqRel,
            MemConsistency::FullTso,
        ] {
            for (name, ir, loads, stores) in &cases {
                let (loads, stores) = (*loads, *stores);
                let ctx = compile_ir(ir, target, tier, crate::OptLevel::Speed);
                let code = ctx.compiled_code().unwrap().code_buffer();
                let arm = target.starts_with("aarch64");
                let (before, after) = if !arm {
                    (false, false)
                } else {
                    match tier {
                        MemConsistency::Fast => (false, false),
                        MemConsistency::AcqRel => (stores, loads),
                        MemConsistency::FullTso => (loads, loads || stores),
                    }
                };
                assert_eq!(
                    code.windows(4).filter(|w| *w == DMB).count(),
                    usize::from(before) + usize::from(after),
                    "{name}/{target}/{tier:?}"
                );
                assert_eq!(code.windows(3).filter(|w| *w == MFENCE).count(), 0);
                if arm {
                    let calls: Vec<_> = code
                        .chunks_exact(4)
                        .enumerate()
                        .filter_map(|(i, w)| {
                            let instruction = u32::from_le_bytes(w.try_into().unwrap());
                            (instruction & 0xffff_fc1f == 0xd63f_0000).then_some(i * 4)
                        })
                        .collect();
                    assert_eq!(calls.len(), 1, "{name}: expected one helper call");
                    let call = calls[0];
                    assert_eq!(
                        code[..call].windows(4).filter(|w| *w == DMB).count(),
                        usize::from(before),
                        "{name}/{tier:?}: leading barrier"
                    );
                    assert_eq!(
                        code[call + 4..].windows(4).filter(|w| *w == DMB).count(),
                        usize::from(after),
                        "{name}/{tier:?}: trailing barrier"
                    );
                }
            }
        }
    }
}
