//! Decision-18 contract witnesses. These test engine configuration/refusal;
//! they claim no legacy instruction execution or descriptor-loading support.
use x86jit_core::{
    Backend, CachedBlock, CpuGeneration, CpuMode, CpuState, ExecutionProfile, Exit,
    ForceRegionError, InterpreterBackend, IrBlock, LegacyCpuModel, LegacyCpuState,
    MachineContextError, MemConsistency, Prot, RegionCaps, RegionKind, StepOptions, StepResult,
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
#[allow(deprecated)] // Compatibility API witness.
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
    assert_eq!(
        legacy.exception_delivery(),
        x86jit_core::ExceptionDelivery::Outward
    );
    legacy.set_exception_delivery(x86jit_core::ExceptionDelivery::InGuest);
    let child = legacy.fork_with_backend(Box::new(NoBackend)).unwrap();
    assert_eq!(child.execution_profile(), ExecutionProfile::LegacySystem);
    assert_eq!(child.legacy_cpu_model(), Some(MODEL));
    assert_eq!(
        child.exception_delivery(),
        x86jit_core::ExceptionDelivery::InGuest
    );
    let cpu = child.new_vcpu();
    assert_eq!(cpu.execution_profile(), ExecutionProfile::LegacySystem);
    assert_eq!(cpu.legacy_cpu_model(), Some(MODEL));
    assert_eq!(cpu.cpu.legacy, CpuState::legacy_real_entry([0; 6]).legacy);
    assert_ne!(cpu.cpu.legacy, LegacyCpuState::default());
    drop(cpu);
    let mut child = child;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || child.set_exception_delivery(x86jit_core::ExceptionDelivery::Outward)
    ))
    .is_err());
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
#[allow(deprecated)] // Compatibility API witness.
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
                    MachineContextError::PagingRequiresProtectedMode
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

#[test]
fn catalogue_pins_current_machine_profiles() {
    use x86jit_core::{CpuModel, FpuModel, ModelIsa, ModelSystem};
    // Decision-23 §2; I286 PRM §6.3 Fig. 6-3, Appendix B General
    // Protection; 80386 PRM §14.7 item 18; SDM Vol 2 §2.1.
    for (model, isa, generation, system, physical_bits, max_insn_len, id, flags_mask) in [
        (
            CpuModel::I80286,
            ModelIsa::I80286,
            CpuGeneration::I286,
            ModelSystem::Msw286,
            24,
            10,
            // xj-50w.11 adds the LOCK and coprocessor-overrun quirk fields,
            // xj-oog the LOADALL encoding (0x24d34f81af8da20b before it).
            0x7357723a3f162910,
            0xf000,
        ),
        (
            CpuModel::I80386,
            ModelIsa::I80386,
            CpuGeneration::I386,
            ModelSystem::Cr386,
            32,
            15,
            0xe8d00c17f97eb71f,
            0x8000,
        ),
    ] {
        let profile = model.profile();
        assert!(std::ptr::eq(profile, model.profile()));
        assert_eq!(profile.isa, isa);
        assert_eq!(profile.isa.generation(), generation);
        assert_eq!(profile.system, system);
        assert_eq!(profile.physical_bits, physical_bits);
        assert_eq!(profile.max_insn_len, Some(max_insn_len));
        assert_eq!(profile.semantics_id, id);
        assert_eq!(profile.fpu, &[FpuModel::Present, FpuModel::None]);
        assert!(profile.timing.is_none());
        // SDM Vol 3B §§25.17.1–25.17.2; Vol 2B §4.2, SAL/SAR/SHL/SHR.
        assert!(!profile.quirks.push_sp_after_decrement);
        assert_eq!(profile.quirks.shift_count_mask, Some(31));
        assert_eq!(profile.quirks.real_flags_clear_mask, flags_mask);
        assert_eq!(profile.quirks.flags_fixed_zero, (1 << 18) | (1 << 21));
    }
}

#[test]
#[allow(deprecated)] // Compatibility API witness.
fn model_selectors_default_exception_delivery_and_allow_overrides() {
    use x86jit_core::{CpuModel, ExceptionDelivery};

    for model in [CpuModel::I80286, CpuModel::I80386] {
        let mut vm = setup(Box::new(NoBackend));
        vm.set_cpu_model(model);
        assert_eq!(vm.exception_delivery(), ExceptionDelivery::InGuest);
        vm.set_exception_delivery(ExceptionDelivery::Outward);
        assert_eq!(vm.exception_delivery(), ExceptionDelivery::Outward);
        let child = vm.fork_with_backend(Box::new(NoBackend)).unwrap();
        assert_eq!(child.exception_delivery(), ExceptionDelivery::Outward);

        vm.set_cpu_model(model);
        assert_eq!(vm.exception_delivery(), ExceptionDelivery::InGuest);
        vm.set_legacy_cpu_model(vm.legacy_cpu_model().unwrap());
        assert_eq!(vm.exception_delivery(), ExceptionDelivery::Outward);
        vm.set_exception_delivery(ExceptionDelivery::InGuest);
        assert_eq!(vm.exception_delivery(), ExceptionDelivery::InGuest);
        let child = vm.fork_with_backend(Box::new(NoBackend)).unwrap();
        assert_eq!(child.exception_delivery(), ExceptionDelivery::InGuest);

        vm.set_legacy_cpu_model(vm.legacy_cpu_model().unwrap());
        assert_eq!(vm.exception_delivery(), ExceptionDelivery::Outward);
    }
}

#[test]
fn model_and_fpu_defaults_are_inherited_and_forked() {
    use x86jit_core::{CpuModel, FpuModel};
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for fpu in [FpuModel::Present, FpuModel::None] {
            let mut vm = setup(Box::new(NoBackend));
            assert_eq!(vm.fpu_model(), FpuModel::Present);
            vm.set_fpu_model(fpu);
            vm.set_cpu_model(model);
            assert_eq!(vm.execution_profile(), ExecutionProfile::Machine);
            assert_eq!(vm.cpu_model(), Some(model));
            assert_eq!(vm.cpu_mode(), CpuMode::Long64);
            let mut cpu = vm.new_vcpu();
            assert_eq!(cpu.cpu_model(), Some(model));
            assert_eq!(cpu.execution_profile(), ExecutionProfile::Machine);
            assert_eq!(cpu.fpu_model(), fpu);
            assert_eq!(cpu.cpu.legacy, CpuState::legacy_real_entry([0; 6]).legacy);
            cpu.set_fpu_model(FpuModel::None);
            assert_eq!(vm.new_vcpu().fpu_model(), fpu);
            let child = vm.fork_with_backend(Box::new(NoBackend)).unwrap();
            assert_eq!(child.cpu_model(), Some(model));
            assert_eq!(child.fpu_model(), fpu);
            assert_eq!(child.new_vcpu().fpu_model(), fpu);
            drop(cpu);
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || vm.set_cpu_model(model)
            ))
            .is_err());
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || vm.set_fpu_model(fpu)
            ))
            .is_err());
        }
    }
}

#[test]
#[allow(deprecated)] // Compatibility API witness.
fn model_selection_can_be_replaced_before_creation_and_cleared_by_application() {
    use x86jit_core::CpuModel;
    let mut vm = setup(Box::new(NoBackend));
    vm.set_cpu_model(CpuModel::I80286);
    vm.set_cpu_model(CpuModel::I80386);
    assert_eq!(vm.cpu_model(), Some(CpuModel::I80386));
    vm.set_legacy_cpu_model(MODEL);
    assert_eq!(vm.cpu_model(), None);
    assert_eq!(vm.legacy_cpu_model(), Some(MODEL));
    vm.set_cpu_model(CpuModel::I80286);
    vm.set_cpu_mode(CpuMode::Real16);
    assert_eq!(vm.cpu_model(), None);
    assert_eq!(vm.legacy_cpu_model(), None);
    assert_eq!(
        vm.execution_profile(),
        ExecutionProfile::Application(CpuMode::Real16)
    );
    assert_eq!(vm.new_vcpu().cpu.legacy, LegacyCpuState::default());
}

#[test]
fn fpu_default_applies_to_application_and_all_vcpu_constructors() {
    use x86jit_core::FpuModel;
    for mode in [CpuMode::Long64, CpuMode::Compat32, CpuMode::Real16] {
        let mut vm = setup(Box::new(InterpreterBackend));
        vm.set_cpu_mode(mode);
        vm.set_fpu_model(FpuModel::None);
        for cpu in [
            vm.new_vcpu(),
            vm.new_vcpu_with_mxcsr(0x1f80),
            vm.new_vcpu_with_cpuid_overrides(Default::default()),
            vm.new_vcpu_with_source_id(100).unwrap(),
        ] {
            assert_eq!(cpu.fpu_model(), FpuModel::None);
            assert_eq!(cpu.execution_profile(), ExecutionProfile::Application(mode));
            assert_eq!(cpu.cpu.legacy, LegacyCpuState::default());
        }
    }
}

#[test]
#[allow(deprecated)] // Compatibility API witness.
fn deprecated_profile_name_remains_usable_in_patterns() {
    let mut vm = setup(Box::new(NoBackend));
    vm.set_legacy_cpu_model(MODEL);
    // This exhaustive match is an external-crate compile witness for the
    // associated-constant pattern and the unchanged enum exhaustiveness.
    let profile = match vm.execution_profile() {
        ExecutionProfile::LegacySystem => ExecutionProfile::Machine,
        ExecutionProfile::Application(mode) => ExecutionProfile::Application(mode),
    };
    assert_eq!(profile, ExecutionProfile::Machine);
}

#[test]
#[allow(deprecated)] // Compatibility API witness.
fn deprecated_context_types_remain_aliases_of_machine_types() {
    use x86jit_core::{
        derive_legacy_context, LegacyContextError, LegacyDecodeContext, LegacyRegime,
        MachineContext, MachineContextError, MachineRegime,
    };

    let cpu = CpuState::legacy_real_entry([0; 6]);
    let context: MachineContext = derive_legacy_context(MODEL, &cpu).unwrap();
    let old_context: LegacyDecodeContext = LegacyDecodeContext {
        regime: LegacyRegime::Real,
        ..context
    };
    let module_context: x86jit_core::legacy::LegacyDecodeContext = old_context;
    assert_eq!(context, module_context);

    let regime: MachineRegime = LegacyRegime::Protected;
    let old_regime: x86jit_core::legacy::LegacyRegime = regime;
    assert!(matches!(old_regime, MachineRegime::Protected));

    let mut invalid = cpu;
    invalid.legacy.cpl = 4;
    let error: MachineContextError = derive_legacy_context(MODEL, &invalid).unwrap_err();
    let old_error: LegacyContextError = error;
    let module_error: x86jit_core::legacy::LegacyContextError = old_error;
    assert!(matches!(module_error, LegacyContextError::InvalidPrivilege));
}
