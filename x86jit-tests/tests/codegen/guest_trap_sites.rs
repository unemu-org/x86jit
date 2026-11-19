//! Alignment witnesses for guest transfers (SDM Vol 1 §14.9, Table 14-21).
use iced_x86::code_asm::*;
use x86jit_core::{Backend, CachedBlock, InterpreterBackend, IrBlock, IrRegion, MemConsistency};
use x86jit_cranelift::{HostTarget, JitBackend, OptLevel};
use x86jit_tests::compare::compare;
use x86jit_tests::oracle::{run_with_backend, RunOutcome, VectorInput};
use x86jit_tests::vector::{CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec};

const CODE: u64 = 0x21_0000;
const DATA: u64 = 0x23_0000;

struct CompiledBackend {
    jit: JitBackend,
    region: bool,
}

impl Backend for CompiledBackend {
    fn code_lifetime(&self) -> Option<std::sync::Arc<x86jit_core::code_lifetime::CodeLifetime>> {
        self.jit.code_lifetime()
    }

    fn materialize(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> CachedBlock {
        let block = if self.region {
            self.jit.materialize_region(
                &IrRegion {
                    entry: ir.guest_start,
                    blocks: vec![ir.clone()],
                    has_loop: false,
                },
                consistency,
                mmio,
                guest_base,
            )
        } else {
            self.jit.materialize(ir, consistency, mmio, guest_base)
        };
        let CachedBlock::Compiled { entry } = &block else {
            panic!("alignment witness must execute compiled code")
        };
        // Pin observability as well as results: the flags regression can be latent
        // on hosts where the selected instruction tolerates a misaligned address.
        assert!(
            (0..4096).any(
                |off| x86jit_core::codemap::lookup_trap(entry.0 as usize + off)
                    .is_some_and(|s| s.host_offset as usize == off)
            ),
            "guest access needs a trap site"
        );
        block
    }
}

fn input(asm: CodeAssembler, data: Vec<u8>) -> VectorInput {
    let mut asm = asm;
    asm.hlt().unwrap();
    VectorInput {
        cpu_init: CpuSnapshot {
            rip: CODE,
            ..Default::default()
        },
        mem_init: vec![
            MemChunk {
                addr: CODE,
                bytes: asm.assemble(CODE).unwrap(),
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: DATA,
                bytes: data,
                kind: MemKind::Ram,
            },
        ],
        entry: CODE,
        run: RunSpec::UntilExit,
    }
}

#[test]
fn guest_trap_sites_native_fixture_addresses_are_mappable() {
    let program = input(CodeAssembler::new(64).unwrap(), vec![0; 128]);
    assert!(program.mem_init.iter().all(|chunk| chunk.addr >= 0x10000));
}

fn matrix(input: &VectorInput, native: bool) -> RunOutcome {
    matrix_exit(input, native, ExitKind::Hlt)
}

fn matrix_exit(input: &VectorInput, native: bool, expected: ExitKind) -> RunOutcome {
    let interp = run_with_backend(input, Box::new(InterpreterBackend));
    assert_eq!(interp.exit, expected, "architectural exit");
    for target in [HostTarget::Baseline, HostTarget::Native] {
        for region in [false, true] {
            for opt in [OptLevel::None, OptLevel::Speed] {
                let jit = run_with_backend(
                    input,
                    Box::new(CompiledBackend {
                        jit: JitBackend::with_options(None, target, opt),
                        region,
                    }),
                );
                assert!(
                    compare(&interp, &jit, &[]).is_none(),
                    "{target:?}, region={region}, opt={opt:?}: {:?}",
                    compare(&interp, &jit, &[])
                );
            }
        }
    }
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if native {
        let hardware = if matches!(expected, ExitKind::Exception { vector: 13, .. }) {
            x86jit_tests::native::run_native_fault(
                input,
                x86jit_tests::native::NativeFault::GeneralProtection,
            )
        } else {
            x86jit_tests::native::run_native(input)
        }
        .expect("native alignment witness must run");
        assert!(
            compare(&hardware, &interp, &[]).is_none(),
            "{:?}",
            compare(&hardware, &interp, &[])
        );
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    let _ = native;
    interp
}

#[test]
fn guest_trap_sites_misaligned_movups_movdqu() {
    // SDM Vol 1 §14.9: explicitly unaligned SIMD loads and stores accept every
    // byte alignment. Pin the transferred bytes independently of either tier.
    for offset in [1u64, 3, 7, 15] {
        for integer in [false, true] {
            let mut asm = CodeAssembler::new(64).unwrap();
            asm.mov(rdi, DATA + offset).unwrap();
            if integer {
                asm.movdqu(xmm0, xmmword_ptr(rdi)).unwrap();
                asm.movdqu(xmmword_ptr(rdi + 32), xmm0).unwrap();
            } else {
                asm.movups(xmm0, xmmword_ptr(rdi)).unwrap();
                asm.movups(xmmword_ptr(rdi + 32), xmm0).unwrap();
            }
            let mut data = vec![0; 128];
            let pattern = std::array::from_fn::<_, 16, _>(|i| (i * 13 + 1) as u8);
            data[offset as usize..offset as usize + 16].copy_from_slice(&pattern);
            let outcome = matrix(&input(asm, data), true);
            assert_eq!(outcome.cpu.xmm[0], u128::from_le_bytes(pattern));
            let memory = outcome.mem.iter().find(|m| m.addr == DATA).unwrap();
            assert_eq!(
                &memory.bytes[offset as usize + 32..offset as usize + 48],
                &pattern
            );
        }
    }
}

fn arithmetic(vex: bool) {
    for offset in [1u64, 3, 7, 15] {
        let mut asm = CodeAssembler::new(64).unwrap();
        asm.mov(rdi, DATA + offset).unwrap();
        asm.movups(xmm1, xmmword_ptr(rdi)).unwrap();
        if vex {
            asm.vaddps(xmm0, xmm1, xmmword_ptr(rdi)).unwrap();
        } else {
            asm.movaps(xmm0, xmm1).unwrap();
            asm.addps(xmm0, xmmword_ptr(rdi)).unwrap();
        }
        let mut data = vec![0; 128];
        let values = [1f32, 2., 4., 8.].map(f32::to_le_bytes).concat();
        data[offset as usize..offset as usize + 16].copy_from_slice(&values);
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        let native = !vex || std::is_x86_feature_detected!("avx");
        #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
        let native = false;
        let input = input(asm, data);
        let exit = if vex {
            ExitKind::Hlt
        } else {
            // ADDPS is the final three-byte instruction before HLT.
            ExitKind::Exception {
                addr: CODE + input.mem_init[0].bytes.len() as u64 - 4,
                vector: 13,
            }
        };
        let outcome = matrix_exit(&input, native, exit);
        let expected: [u8; 16] = if vex {
            [2f32, 4., 8., 16.]
        } else {
            [1f32, 2., 4., 8.]
        }
        .map(f32::to_le_bytes)
        .concat()
        .try_into()
        .unwrap();
        assert_eq!(outcome.cpu.xmm[0], u128::from_le_bytes(expected));
    }
}

#[test]
fn guest_trap_sites_misaligned_vex_arithmetic() {
    // SDM Vol 1 §14.9: VEX arithmetic m128 operands allow byte alignment.
    arithmetic(true);
}

#[test]
fn guest_trap_sites_misaligned_legacy_arithmetic_raises_gp() {
    // SDM Vol 1 §14.9, Table 14-21 requires guest #GP(0) for legacy ADDPS.
    arithmetic(false);
}

#[test]
fn guest_trap_sites_hot_extending_load_results() {
    // SDM Vol 2B §4, MOVSX/MOVSXD and MOVZX, Operation: sign/zero extension.
    let expected = [
        0xffff_ffff_ffff_ff81,
        0xffff_ffff_ffff_fe81,
        0xffff_ffff_fcfd_fe81,
        0x81,
        0xfe81,
    ];
    for (shape, expected) in expected.into_iter().enumerate() {
        let mut asm = CodeAssembler::new(64).unwrap();
        asm.mov(rdi, DATA).unwrap();
        match shape {
            0 => asm.movsx(rax, byte_ptr(rdi)),
            1 => asm.movsx(rax, word_ptr(rdi)),
            2 => asm.movsxd(rax, dword_ptr(rdi)),
            3 => asm.movzx(rax, byte_ptr(rdi)),
            4 => asm.movzx(rax, word_ptr(rdi)),
            _ => unreachable!(),
        }
        .unwrap();
        let outcome = matrix(&input(asm, vec![0x81, 0xfe, 0xfd, 0xfc]), true);
        assert_eq!(outcome.cpu.gpr[0], expected);
    }
}

fn descriptor_overflow_case(shape: usize) {
    let mut asm = CodeAssembler::new(64).unwrap();
    asm.mov(rdi, DATA).unwrap();
    match shape {
        0 => {
            for _ in 0..320 {
                asm.mov(rax, qword_ptr(rdi)).unwrap();
            }
        }
        1 => {
            // A 4 KiB copy: 128 pairs, 256 instruction owners.
            for offset in (0..4096).step_by(32) {
                asm.vmovdqu(ymm0, ymmword_ptr(rdi + offset)).unwrap();
                asm.vmovdqu(ymmword_ptr(rdi + offset + 4096), ymm0).unwrap();
            }
        }
        2 => {
            for _ in 0..125 {
                asm.add(qword_ptr(rdi), 1).unwrap();
            }
        }
        _ => unreachable!(),
    }
    let data = (0..8192)
        .map(|i| if i < 4096 { (i * 13 + 1) as u8 } else { 0 })
        .collect();
    // The matrix asserts compilation of every dispatched block, then compares
    // complete register, flag, memory and exit results with the interpreter.
    let program = input(asm, data);
    let outcome = matrix(&program, false);
    for budget in 1..=3 {
        let snapshot = |backend: Box<dyn Backend>| {
            let vm = budget_vm(backend, &program.mem_init[0].bytes);
            vm.write_bytes(DATA, &program.mem_init[1].bytes).unwrap();
            let snapshot = budget_snapshot(&vm, budget);
            let mut data = vec![0; 8192];
            vm.mem.read_bytes(DATA, &mut data).unwrap();
            (snapshot, data)
        };
        let mut expected = snapshot(Box::new(InterpreterBackend));
        assert_eq!(expected.0 .4, expected.0 .3);
        if budget == 1 {
            assert_eq!(expected.0 .3, if shape == 2 { 125 } else { 250 });
        }
        expected.0 .4 = 0;
        for opt in [OptLevel::None, OptLevel::Speed] {
            let jit = JitBackend::with_opt_level(opt);
            jit.enable_icount();
            assert_eq!(
                snapshot(Box::new(CompiledBackend { jit, region: false })),
                expected,
                "shape={shape}, budget={budget}, {opt:?}"
            );
        }
    }
    if shape == 1 {
        // SDM Vol 2B §4, MOVDQU/VMOVDQU, Operation (VEX.256 load/store):
        // each pair copies all 256 bits, including across ordinary blocks.
        let memory = outcome.mem.iter().find(|m| m.addr == DATA).unwrap();
        assert_eq!(&memory.bytes[4096..8192], &memory.bytes[..4096]);
    }
}

#[test]
fn guest_trap_sites_descriptor_overflow_320_loads_compiles() {
    descriptor_overflow_case(0);
}

#[test]
fn guest_trap_sites_descriptor_overflow_4k_ymm_copy_compiles() {
    descriptor_overflow_case(1);
}

#[test]
fn guest_trap_sites_descriptor_overflow_125_rmw_compiles() {
    descriptor_overflow_case(2);
}

struct RegionOverflowBackend {
    jit: JitBackend,
    regions: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Backend for RegionOverflowBackend {
    fn code_lifetime(&self) -> Option<std::sync::Arc<x86jit_core::code_lifetime::CodeLifetime>> {
        self.jit.code_lifetime()
    }

    fn tier_up_async(&self, req: x86jit_core::TierUpRequest) -> x86jit_core::TierUpSubmit {
        assert!(matches!(
            &req.unit,
            x86jit_core::TierUpUnit::Region(region) if region.blocks.len() == 2 && region.has_loop
        ));
        self.jit.tier_up_async(req)
    }

    fn tier_up_finished(&self) -> Vec<x86jit_core::TierUpFinished> {
        self.jit.tier_up_finished()
    }

    fn region_caps(&self) -> Option<x86jit_core::RegionCaps> {
        Some(x86jit_core::RegionCaps {
            max_blocks: 16,
            max_icount: 128,
        })
    }

    fn materialize(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> CachedBlock {
        let block = self.jit.materialize(ir, consistency, mmio, guest_base);
        assert!(
            matches!(block, CachedBlock::Compiled { .. }),
            "every block must compile"
        );
        block
    }

    fn materialize_region(
        &self,
        region: &IrRegion,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> CachedBlock {
        assert_eq!(
            region.blocks.len(),
            2,
            "descriptor cap excludes the third block"
        );
        assert!(region.has_loop);
        self.regions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let block = self
            .jit
            .materialize_region(region, consistency, mmio, guest_base);
        assert!(
            matches!(block, CachedBlock::Compiled { .. }),
            "capped loop region must compile"
        );
        block
    }
}

#[test]
fn guest_trap_sites_descriptor_limit_caps_multiblock_region() {
    let code = overflowing_region_code();
    let mut input = VectorInput {
        cpu_init: CpuSnapshot {
            rip: CODE,
            ..Default::default()
        },
        mem_init: vec![
            MemChunk {
                addr: CODE,
                bytes: code,
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: DATA,
                bytes: vec![0; 8],
                kind: MemKind::Ram,
            },
        ],
        entry: CODE,
        run: RunSpec::UntilExit,
    };
    input.cpu_init.gpr[1] = 2;
    input.cpu_init.gpr[7] = DATA;
    let interp = run_with_backend(&input, Box::new(InterpreterBackend));
    assert_eq!(interp.exit, ExitKind::Hlt);
    for opt in [OptLevel::None, OptLevel::Speed] {
        let regions = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let jit = run_with_backend(
            &input,
            Box::new(RegionOverflowBackend {
                jit: JitBackend::with_opt_level(opt),
                regions: regions.clone(),
            }),
        );
        assert!(regions.load(std::sync::atomic::Ordering::Relaxed) > 0);
        assert!(
            compare(&interp, &jit, &[]).is_none(),
            "{opt:?}: {:?}",
            compare(&interp, &jit, &[])
        );
    }
}

fn budget_vm(backend: Box<dyn Backend>, code: &[u8]) -> x86jit_core::Vm {
    use x86jit_core::{Prot, RegionKind, Vm, VmConfig};
    let span = DATA + 0x10000;
    let mut vm = Vm::with_backend(VmConfig::flat(span), backend);
    vm.map(CODE, (span - CODE) as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.write_bytes(CODE, code).unwrap();
    vm
}

fn budget_snapshot(vm: &x86jit_core::Vm, budget: u64) -> (String, CpuSnapshot, u64, u64, u64) {
    use x86jit_core::Reg;
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rdi, DATA);
    cpu.set_reg(Reg::Rcx, 2);
    let exit = cpu.run(vm, Some(budget));
    let retired = cpu.retired_instructions();
    let executed = cpu.executed_instructions();
    // Retired instructions remain interpreter-only; compiled execution contributes
    // through the separately enabled executed-instruction counter.
    assert!(retired <= executed);
    (
        format!("{exit:?}"),
        CpuSnapshot {
            gpr: cpu.cpu.gpr,
            rip: cpu.cpu.rip,
            flags: cpu.cpu.flags.into(),
            xmm: std::array::from_fn(|i| cpu.xmm(i)),
            ymm_hi: std::array::from_fn(|i| cpu.ymm_hi(i)),
            ..Default::default()
        },
        vm.mem.read(DATA, 8).unwrap(),
        executed,
        retired,
    )
}

fn overflowing_region_code() -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    let mut entry = a.create_label();
    let mut body = a.create_label();
    a.set_label(&mut entry).unwrap();
    a.mov(rax, qword_ptr(rdi)).unwrap();
    a.jmp(body).unwrap();
    a.set_label(&mut body).unwrap();
    for _ in 0..100 {
        a.add(qword_ptr(rdi), 1).unwrap();
    }
    a.dec(ecx).unwrap();
    a.jnz(entry).unwrap();
    for _ in 0..60 {
        a.mov(rax, qword_ptr(rdi)).unwrap();
    }
    a.hlt().unwrap();
    a.assemble(CODE).unwrap()
}

#[test]
fn descriptor_capped_region_finite_budget_parity() {
    let code = overflowing_region_code();
    for budget in 1..=3 {
        let mut interp = budget_snapshot(&budget_vm(Box::new(InterpreterBackend), &code), budget);
        assert_eq!(interp.4, interp.3, "every interpreted instruction retires");
        interp.4 = 0;
        for opt in [OptLevel::None, OptLevel::Speed] {
            let jit = JitBackend::with_opt_level(opt);
            jit.enable_icount();
            let vm = budget_vm(
                Box::new(RegionOverflowBackend {
                    jit,
                    regions: Default::default(),
                }),
                &code,
            );
            assert_eq!(
                budget_snapshot(&vm, budget),
                interp,
                "budget={budget}, {opt:?}"
            );
        }
    }
}

#[test]
fn descriptor_capped_region_records_region() {
    let code = overflowing_region_code();
    let vm = budget_vm(
        Box::new(RegionOverflowBackend {
            jit: JitBackend::new(),
            regions: Default::default(),
        }),
        &code,
    );
    budget_snapshot(&vm, 1);
    assert_eq!(vm.cache.regions(), 1);
}

#[test]
fn descriptor_background_capped_region_records_region() {
    let code = overflowing_region_code();
    for opt in [OptLevel::None, OptLevel::Speed] {
        let jit = JitBackend::with_opt_level(opt);
        let handle = jit.tier_up_handle();
        let mut vm = budget_vm(
            Box::new(RegionOverflowBackend {
                jit,
                regions: Default::default(),
            }),
            &code,
        );
        vm.set_tier_up_after(Some(1));
        vm.set_tier_up_background(true);
        budget_snapshot(&vm, 1);
        budget_snapshot(&vm, 1);
        handle.wait_idle();
        budget_snapshot(&vm, 1);
        assert_eq!(vm.cache.tier_bg_published(), 1);
        assert_eq!(
            vm.cache.regions(),
            1,
            "background publication retains the capped region"
        );
    }
}

#[test]
fn cached_block_public_variants_remain_exhaustive() {
    fn embedder_match(block: CachedBlock) -> bool {
        match block {
            CachedBlock::Interpreted(_) => false,
            CachedBlock::Compiled { .. } => true,
        }
    }
    let ir = IrBlock {
        legacy_context: None,
        mode: x86jit_core::CpuMode::Long64,
        guest_start: CODE,
        ops: vec![],
        temp_count: 0,
        guest_len: 0,
        icount: 0,
    };
    assert!(!embedder_match(CachedBlock::Interpreted(
        std::sync::Arc::new(ir)
    )));
}

#[test]
fn descriptor_boundary_smc_and_finite_budget_parity() {
    // Same-block SMC takes effect at the next ordinary block boundary (§10).
    let mut code = vec![0xc6, 0x04, 0x25, 0, 0, 0, 0, 9]; // mov byte ptr [patch],9
    code.extend([0x48, 0x8b, 0x07].repeat(248));
    let suffix = CODE + code.len() as u64;
    code.extend([0x48, 0x8b, 0x07, 0xb8, 7, 0, 0, 0, 0xf4]);
    let patch = (suffix + 4) as u32;
    code[3..7].copy_from_slice(&patch.to_le_bytes());
    for budget in 1..=3 {
        let vm = budget_vm(Box::new(InterpreterBackend), &code);
        let mut expected = budget_snapshot(&vm, budget);
        assert_eq!(vm.mem.read(u64::from(patch), 1).unwrap(), 9);
        if budget == 1 {
            assert_eq!(expected.1.rip, suffix);
            assert_eq!(expected.3, 249);
        } else {
            assert_eq!(expected.1.gpr[0], 9);
        }
        expected.4 = 0;
        for opt in [OptLevel::None, OptLevel::Speed] {
            let jit = JitBackend::with_opt_level(opt);
            jit.enable_icount();
            let vm = budget_vm(Box::new(jit), &code);
            assert_eq!(
                budget_snapshot(&vm, budget),
                expected,
                "budget={budget}, {opt:?}"
            );
            assert_eq!(vm.mem.read(u64::from(patch), 1).unwrap(), 9);
        }
    }
}
