//! Decision-18 contract witnesses. These test engine configuration/refusal;
//! they claim no legacy instruction execution or descriptor-loading support.
use x86jit_core::{
    Backend, CachedBlock, CpuGeneration, CpuMode, CpuState, ExecutionProfile, Exit,
    ForceRegionError, InterpreterBackend, IrBlock, LegacyContextError, LegacyCpuModel,
    LegacyCpuState, MemConsistency, Prot, RegionCaps, RegionKind, StepOptions, StepResult,
    StepRunError, StepStop, UnsupportedSystemOperation, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const MODEL: LegacyCpuModel = LegacyCpuModel {
    generation: CpuGeneration::I386,
    model_semantics_id: 19,
};

struct NoBackend;
impl Backend for NoBackend {
    fn materialize(
        &self,
        _: &IrBlock,
        _: MemConsistency,
        _: Option<(u64, u64)>,
        _: u64,
    ) -> CachedBlock {
        panic!("unsupported system entry reached backend");
    }
}

fn setup(backend: Box<dyn Backend>) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x4000), backend);
    vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xb8, 0x34, 0x12, 0, 0, 0xf4])
        .unwrap();
    vm
}

fn memory(vm: &Vm) -> Vec<u8> {
    let mut bytes = vec![0; 0x4000];
    vm.read_bytes(0, &mut bytes).unwrap();
    bytes
}

fn unchanged(cpu: &Vcpu, state: &str, vm: &Vm, bytes: &[u8]) {
    assert_eq!(format!("{:?}", cpu.cpu), state);
    assert_eq!(memory(vm), bytes);
    assert_eq!(cpu.retired_instructions(), 0);
    assert_eq!(cpu.executed_instructions(), 0);
    assert!(cpu.has_pending_irq());
    assert!(!cpu.irq_ready());
}

#[test]
fn profile_selection_preserves_application_defaults_and_forks() {
    let mut vm = setup(Box::new(InterpreterBackend));
    assert_eq!(
        vm.execution_profile(),
        ExecutionProfile::Application(CpuMode::Long64)
    );
    assert_eq!(vm.legacy_cpu_model(), None);
    vm.set_legacy_cpu_model(MODEL);
    assert_eq!(vm.cpu_mode(), CpuMode::Long64);
    vm.set_cpu_mode(CpuMode::Compat32);
    assert_eq!(
        vm.execution_profile(),
        ExecutionProfile::Application(CpuMode::Compat32)
    );
    assert_eq!(vm.legacy_cpu_model(), None);
    let cpu = vm.new_vcpu();
    assert_eq!(cpu.execution_profile(), vm.execution_profile());
    assert_eq!(cpu.legacy_cpu_model(), None);
    assert_eq!(
        format!("{:?}", cpu.cpu),
        format!("{:?}", CpuState::default())
    );
    vm.set_cpu_mode(CpuMode::Real16);
    assert_eq!(
        cpu.execution_profile(),
        ExecutionProfile::Application(CpuMode::Compat32)
    );
    assert_eq!(
        vm.new_vcpu().execution_profile(),
        ExecutionProfile::Application(CpuMode::Real16)
    );
    drop(cpu);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || vm.set_legacy_cpu_model(MODEL)
    ))
    .is_err());

    let mut legacy = setup(Box::new(NoBackend));
    legacy.set_cpu_mode(CpuMode::Real16);
    legacy.set_legacy_cpu_model(MODEL);
    let child = legacy.fork_with_backend(Box::new(NoBackend)).unwrap();
    assert_eq!(child.execution_profile(), ExecutionProfile::LegacySystem);
    assert_eq!(child.legacy_cpu_model(), Some(MODEL));
    let cpu = child.new_vcpu();
    assert_eq!(cpu.execution_profile(), ExecutionProfile::LegacySystem);
    assert_eq!(cpu.legacy_cpu_model(), Some(MODEL));
    assert_eq!(cpu.cpu.legacy, CpuState::legacy_real_entry([0; 6]).legacy);
    assert_ne!(cpu.cpu.legacy, LegacyCpuState::default());
    drop(cpu);
    let mut child = child;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || child.set_cpu_mode(CpuMode::Long64)
    ))
    .is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || child.set_legacy_cpu_model(MODEL)
    ))
    .is_err());
}

#[test]
fn every_application_mode_still_ignores_legacy_storage() {
    for mode in [CpuMode::Long64, CpuMode::Compat32, CpuMode::Real16] {
        let mut vm = setup(Box::new(InterpreterBackend));
        vm.set_cpu_mode(mode);
        let mut cpu = vm.new_vcpu();
        assert_eq!(cpu.cpu.legacy, LegacyCpuState::default());
        cpu.cpu.rip = 0x1000;
        cpu.cpu.legacy.cr0 = u32::MAX;
        cpu.cpu.legacy.cpl = 255;
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.cpu.gpr[0], 0x1234);
    }
}

#[test]
fn legacy_run_and_step_refuse_before_any_guest_effect_or_backend_entry() {
    for backend in [
        Box::new(NoBackend) as Box<dyn Backend>,
        Box::new(JitBackend::new()),
    ] {
        let mut vm = setup(backend);
        vm.set_legacy_cpu_model(MODEL);
        vm.set_tier_up_after(Some(1));
        assert!(matches!(
            vm.test_support_force_region(
                0x1000,
                RegionCaps {
                    max_blocks: 2,
                    max_icount: 8
                },
                0
            ),
            Err(ForceRegionError::UnsupportedMode)
        ));
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[4] = 0x3000;
        cpu.cpu.flags.if_ = true;
        // Nonzero-CPL execution awaits the complete privilege-enforcement seam.
        cpu.cpu.legacy.cr0 = 1;
        cpu.cpu.legacy.cpl = 1;
        // Outstanding and completed device work must remain untouched on refusal.
        cpu.cpu.pending_mmio = Some(0xabcd);
        cpu.cpu.pending_mmio_write = true;
        cpu.cpu.pending_port_in = Some(1);
        cpu.cpu.mmio_retry_active = true;
        cpu.inject_irq(0x40);
        let before = format!("{:?}", cpu.cpu);
        let bytes = memory(&vm);
        for budget in [Some(0), Some(1), None] {
            assert!(matches!(
                cpu.run(&vm, budget),
                Exit::UnsupportedSystemContext {
                    addr: 0x1000,
                    operation: UnsupportedSystemOperation::LegacyEntryFacility,
                    ..
                }
            ));
            unchanged(&cpu, &before, &vm, &bytes);
        }
        assert!(matches!(
            cpu.step_instruction(&vm),
            StepResult::Exit(Exit::UnsupportedSystemContext {
                operation: UnsupportedSystemOperation::LegacyEntryFacility,
                ..
            })
        ));
        unchanged(&cpu, &before, &vm, &bytes);
        for budget in [0, 1] {
            let run = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::UnsupportedSystemContext {
                    operation: UnsupportedSystemOperation::LegacyEntryFacility,
                    ..
                })
            ));
            unchanged(&cpu, &before, &vm, &bytes);
        }
        cpu.cpu.legacy.cr0 = 1 << 31;
        assert!(matches!(
            cpu.run(&vm, None),
            Exit::UnsupportedSystemContext {
                operation: UnsupportedSystemOperation::InvalidLegacyContext(
                    LegacyContextError::PagingRequiresProtectedMode
                ),
                ..
            }
        ));
    }
}

#[test]
fn application_irq_guard_retains_vector_stack_memory_and_completions() {
    for mode in [CpuMode::Compat32, CpuMode::Long64] {
        for if_ in [false, true] {
            for backend in [
                Box::new(NoBackend) as Box<dyn Backend>,
                Box::new(JitBackend::new()),
            ] {
                let mut vm = setup(backend);
                vm.set_cpu_mode(mode);
                let mut cpu = vm.new_vcpu();
                cpu.cpu.rip = 0x1000;
                cpu.cpu.gpr[4] = 0x3000;
                cpu.cpu.flags.if_ = if_;
                cpu.inject_irq(0x40);
                let before = format!("{:?}", cpu.cpu);
                let bytes = memory(&vm);
                assert!(matches!(cpu.run(&vm, Some(0)), Exit::BudgetExhausted));
                unchanged(&cpu, &before, &vm, &bytes);
                for budget in [Some(1), None] {
                    assert!(matches!(
                        cpu.run(&vm, budget),
                        Exit::UnsupportedSystemContext {
                            addr: 0x1000,
                            operation: UnsupportedSystemOperation::ApplicationIrqDelivery,
                            ..
                        }
                    ));
                    unchanged(&cpu, &before, &vm, &bytes);
                }
                for budget in [0, 1] {
                    assert!(matches!(
                        cpu.run_steps(&vm, budget, StepOptions::default()),
                        Err(StepRunError::UnsupportedMode)
                    ));
                    unchanged(&cpu, &before, &vm, &bytes);
                }
                cpu.cpu.pending_mmio = Some(42);
                cpu.cpu.pending_mmio_write = true;
                cpu.cpu.pending_port_in = Some(1);
                cpu.cpu.mmio_retry_active = true;
                let before = format!("{:?}", cpu.cpu);
                assert!(matches!(
                    cpu.run(&vm, Some(1)),
                    Exit::UnsupportedSystemContext {
                        operation: UnsupportedSystemOperation::ApplicationIrqDelivery,
                        ..
                    }
                ));
                unchanged(&cpu, &before, &vm, &bytes);
            }
        }
    }
}

#[test]
fn legacy_ir_is_never_compiled_even_in_a_mixed_region() {
    use x86jit_core::{
        derive_legacy_context,
        lift::{lift_block, FetchAddr},
        IrRegion,
    };
    let vm = setup(Box::new(NoBackend));
    let application = lift_block(&vm.mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    assert!(application.legacy_context.is_none());
    let mut legacy = application.clone();
    legacy.legacy_context =
        Some(derive_legacy_context(MODEL, &CpuState::legacy_real_entry([0; 6])).unwrap());
    let backend = JitBackend::new();
    backend.set_compiled_code_observer(|_| panic!("legacy IR was compiled"));
    for mxcsr in [0, 0x2000] {
        assert!(matches!(
            backend.materialize_with_split_stores(
                &legacy,
                MemConsistency::Fast,
                None,
                0,
                mxcsr,
                false
            ),
            CachedBlock::Interpreted(_)
        ));
        let region = IrRegion {
            entry: application.guest_start,
            blocks: vec![application.clone(), legacy.clone()],
            has_loop: true,
        };
        assert!(matches!(
            backend.materialize_region_with_split_stores(
                &region,
                MemConsistency::Fast,
                None,
                0,
                mxcsr,
                false
            ),
            CachedBlock::Interpreted(_)
        ));
    }
}

#[test]
fn public_interpreter_refuses_legacy_ir_including_backend_fallbacks() {
    use x86jit_core::{
        derive_legacy_context,
        interp::{interpret_block, RetireInfo},
        lift::{lift_block, FetchAddr},
    };
    for mode in [CpuMode::Long64, CpuMode::Compat32, CpuMode::Real16] {
        let vm = setup(Box::new(NoBackend));
        // A store and register write would expose an accidental application walk.
        vm.write_bytes(0x1000, &[0x89, 0x03, 0xb8, 0x78, 0x56, 0, 0, 0xf4])
            .unwrap();
        let mut ir = lift_block(&vm.mem, FetchAddr::flat(0x1000), mode).unwrap();
        ir.legacy_context =
            Some(derive_legacy_context(MODEL, &CpuState::legacy_real_entry([0; 6])).unwrap());
        let backend = JitBackend::new();
        for mxcsr in [0, 0x2000] {
            let CachedBlock::Interpreted(fallback) =
                backend.materialize_with_mxcsr(&ir, MemConsistency::Fast, None, 0, mxcsr)
            else {
                panic!("legacy IR was compiled")
            };
            for block in [&ir, fallback.as_ref()] {
                let mut cpu = CpuState {
                    rip: 0x1000,
                    ..CpuState::default()
                };
                cpu.gpr[3] = 0x2000;
                cpu.gpr[0] = 0x1234;
                cpu.mxcsr |= mxcsr;
                cpu.pending_mmio = Some(42);
                cpu.pending_port_in = Some(1);
                cpu.mmio_retry_active = true;
                let before = format!("{cpu:?}");
                let bytes = memory(&vm);
                let mut scratch = vec![99, 88];
                let mut info = RetireInfo {
                    retired: 17,
                    sti_shadow: true,
                    ss_shadow: true,
                    ..RetireInfo::default()
                };
                let bookkeeping = format!("{info:?}");
                assert!(matches!(
                    interpret_block(block, &mut cpu, &vm.mem, &mut scratch, &mut info),
                    StepResult::Exit(Exit::UnsupportedSystemContext {
                        addr: 0x1000,
                        operation: UnsupportedSystemOperation::LegacyExecution,
                        ..
                    })
                ));
                assert_eq!(format!("{cpu:?}"), before);
                assert_eq!(memory(&vm), bytes);
                assert_eq!(scratch, [99, 88]);
                assert_eq!(format!("{info:?}"), bookkeeping);
            }
        }
    }
}
