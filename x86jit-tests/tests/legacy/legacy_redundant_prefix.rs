//! Redundant-prefix witnesses (spec.md §17.10).
//! SDM Vol 2B §4.3 PAUSE p. 4-226: F3 90 is NOP before Pentium 4.
//! Intel486 PRM §26 REP pp. 26-244–246: ignores REP on non-strings
//! except LOOP, whose behavior is not guaranteed (not covered here).
//! 8086 UM §2.7 pp. 2-42–43, 286 PRM App. B REP pp. B-92–93,
//! NEC V20/V30 UM §12 p. 12-15 describe string uses only: ignoring
//! REP/REPNE on ordinary non-strings is ASSUMED for those models.
//! 386 PRM §17 REP p. 17-141 says #UD on non-strings; compatibility
//! follows SDM PAUSE for F3 90 and the explicit BSF/BSR compatibility
//! encodings below. F2 90 is ASSUMED NOP; other non-strings raise #UD.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use x86jit_core::{
    CpuMode, CpuModel, ExceptionDelivery, I386RepQuirkError, InterpreterBackend, ModelSystem, Prot,
    RegionKind, SegmentCache, Vcpu, Vm, VmConfig,
};

const MODELS: [CpuModel; 8] = [
    CpuModel::I8088,
    CpuModel::I8086,
    CpuModel::NecV20,
    CpuModel::NecV30,
    CpuModel::I80286,
    CpuModel::I80386,
    CpuModel::I80486Dx,
    CpuModel::I80486DxCpuid,
];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Real,
    P16,
    P32,
}
fn modes(model: CpuModel) -> &'static [Mode] {
    if model.profile().system == ModelSystem::None {
        &[Mode::Real]
    } else if model == CpuModel::I80286 {
        &[Mode::Real, Mode::P16]
    } else {
        &[Mode::Real, Mode::P16, Mode::P32]
    }
}
fn setup(model: CpuModel, mode: Mode, code: &[u8]) -> (Vm, Vcpu) {
    setup_quirk(model, mode, code, None)
}
fn setup_quirk(model: CpuModel, mode: Mode, code: &[u8], ignore: Option<bool>) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x10000));
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.set_cpu_model(model);
    if let Some(ignore) = ignore {
        vm.set_i386_ignore_rep_non_string(ignore).unwrap();
    }
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    let protected = mode != Mode::Real;
    cpu.cpu.legacy.cr0 = u32::from(protected);
    cpu.cpu.cs = if protected { 8 } else { 0 };
    let data = if protected { 16 } else { 0 };
    cpu.cpu.ds = data;
    cpu.cpu.es = data;
    cpu.cpu.ss = data;
    cpu.cpu.fs = data;
    cpu.cpu.gs = data;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        *cache = SegmentCache {
            base: 0,
            limit: if mode == Mode::P32 { u32::MAX } else { 0xffff },
            attributes: if i == 0 { 0x9b } else { 0x93 }
                | if mode == Mode::P32 { 0xc000 } else { 0 },
            ..Default::default()
        };
    }
    cpu.cpu.gpr[..8].copy_from_slice(&[0x1234, 0x27, 0x40, 0x123, 0x8000, 0x8100, 0x6000, 0x7000]);
    cpu.cpu.flags.cf = true;
    cpu.cpu.flags.of = true;
    cpu.cpu.flags.df = true;
    (vm, cpu)
}

#[test]
fn rep_nop_retires_once_without_changing_state() {
    for model in MODELS {
        for &mode in modes(model) {
            for code in [&[0xf3, 0x90][..], &[0xf2, 0x90]] {
                let (vm, mut cpu) = setup(model, mode, code);
                let before = cpu.cpu.clone();
                fixture::steps(&mut cpu, &vm, 1);
                assert_eq!(
                    cpu.cpu.rip,
                    before.rip + 2,
                    "{model:?}/{mode:?}/{code:02x?}"
                );
                let mut after = cpu.cpu.clone();
                after.rip = before.rip;
                assert_eq!(
                    format!("{after:?}"),
                    format!("{before:?}"),
                    "{model:?}/{mode:?}/{code:02x?}"
                );
            }
        }
    }
}

#[test]
fn ignored_prefixes_preserve_the_unprefixed_operation() {
    // Segment prefixes select memory only: 8086 UM §2.3 p. 2-13;
    // 286 PRM §2.4.3.1 pp. 2-18–19; 386 PRM §2.5.3.1 p. 2-16;
    // Intel486 PRM §2.4 p. 2-16; NEC UM §12 p. 12-176 (Segment Override Prefixes).
    // 66/67 apply only to the respective size attributes: 386 PRM
    // §17.1.2 p. 17-1; Intel486 PRM §2.4 p. 2-16.
    for model in MODELS {
        for &mode in modes(model) {
            let mut prefixes = vec![0xf2, 0xf3, 0x26, 0x2e, 0x36, 0x3e];
            if matches!(
                model,
                CpuModel::I80386 | CpuModel::I80486Dx | CpuModel::I80486DxCpuid
            ) {
                prefixes.extend([0x64, 0x65, 0x66, 0x67]);
            }
            // NOP, MOV AL,BL, ADD AL,BL, CLC. Operand/address sizes
            // have no effect on these byte or operand-free operations.
            for bare in [&[0x90][..], &[0x88, 0xd8], &[0x00, 0xd8], &[0xf8]] {
                for &prefix in &prefixes {
                    let mut code = vec![prefix];
                    code.extend(bare);
                    let (plain_vm, mut plain) = setup(model, mode, bare);
                    let (vm, mut cpu) = setup(model, mode, &code);
                    if model == CpuModel::I80386 && matches!(prefix, 0xf2 | 0xf3) && bare != [0x90]
                    {
                        fixture::fault(&mut cpu, &vm, 6, None);
                        continue;
                    }
                    fixture::steps(&mut plain, &plain_vm, 1);
                    fixture::steps(&mut cpu, &vm, 1);
                    assert_eq!(cpu.cpu.rip, plain.cpu.rip + 1);
                    cpu.cpu.rip = plain.cpu.rip;
                    assert_eq!(
                        format!("{:?}", cpu.cpu),
                        format!("{:?}", plain.cpu),
                        "{model:?}/{mode:?}/{code:02x?}"
                    );
                }
            }
        }
    }
}

#[test]
fn rep_bit_scans_keep_the_historical_operation() {
    // SDM Vol 2A §3.3 LZCNT p. 3-578, Vol 2B §4.3 TZCNT p. 4-730:
    // without the feature these encodings execute BSR/BSF.
    for model in [
        CpuModel::I80386,
        CpuModel::I80486Dx,
        CpuModel::I80486DxCpuid,
    ] {
        for &mode in modes(model) {
            for (opcode, expected) in [(0xbc, 4), (0xbd, 12)] {
                let (vm, mut cpu) = setup(model, mode, &[0xf3, 0x0f, opcode, 0xc3]);
                cpu.cpu.gpr[3] = 0x1010;
                fixture::steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.gpr[0], expected, "{model:?}/{mode:?}/{opcode:x}");
                assert!(!cpu.cpu.flags.zf);
            }
        }
    }
}

#[test]
fn rep_wbinvd_keeps_the_486_operation() {
    // Intel486 PRM §26 REP p. 26-246, WBINVD p. 26-281.
    for model in [CpuModel::I80486Dx, CpuModel::I80486DxCpuid] {
        for &mode in modes(model) {
            let (vm, mut cpu) = setup(model, mode, &[0xf3, 0x0f, 0x09]);
            fixture::steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.rip, 0x1003);
        }
    }
}

#[test]
fn i386_rep_non_string_quirk_matches_the_unprefixed_instruction() {
    // Default: 386 PRM §17 REP p. 17-141, in real and protected mode.
    // Ignoring is ASSUMED: SDM Vol 2A §2.1.1 p. 2-2 calls this use reserved
    // with unpredictable behavior; decision-23 §3 permits this compatibility option.
    for &mode in modes(CpuModel::I80386) {
        for bare in [&[0xc3][..], &[0x40], &[0x88, 0xd8], &[0x00, 0xd8], &[0xf8]] {
            for prefix in [0xf3, 0xf2] {
                let mut code = vec![prefix];
                code.extend(bare);
                for ignore in [None, Some(false), Some(true)] {
                    let (vm, mut cpu) = setup_quirk(CpuModel::I80386, mode, &code, ignore);
                    vm.write_bytes(0x8000, &0x1800u32.to_le_bytes()).unwrap();
                    if ignore != Some(true) {
                        fixture::fault(&mut cpu, &vm, 6, None);
                        continue;
                    }
                    let (plain_vm, mut plain) = setup(CpuModel::I80386, mode, bare);
                    plain_vm
                        .write_bytes(0x8000, &0x1800u32.to_le_bytes())
                        .unwrap();
                    fixture::steps(&mut cpu, &vm, 1);
                    fixture::steps(&mut plain, &plain_vm, 1);
                    assert_eq!(cpu.cpu.rip, plain.cpu.rip + u64::from(bare != [0xc3]));
                    if bare == [0xc3] {
                        assert_eq!(cpu.cpu.rip, 0x1800);
                    }
                    cpu.cpu.rip = plain.cpu.rip;
                    assert_eq!(
                        format!("{:?}", cpu.cpu),
                        format!("{:?}", plain.cpu),
                        "{mode:?}/{code:02x?}"
                    );
                }
            }
        }
    }
}

#[test]
fn i386_rep_compatibility_exceptions_survive_both_quirk_settings() {
    // SDM Vol 2B §4.3 PAUSE p. 4-226 / TZCNT p. 4-730;
    // Vol 2A §3.3 LZCNT p. 3-578. F2 90 remains ASSUMED NOP.
    for &mode in modes(CpuModel::I80386) {
        for ignore in [false, true] {
            for bare in [&[0x90][..], &[0x0f, 0xbc, 0xc3], &[0x0f, 0xbd, 0xc3]] {
                let prefixes: &[u8] = if bare == [0x90] {
                    &[0xf3, 0xf2]
                } else {
                    &[0xf3]
                };
                for &prefix in prefixes {
                    let mut code = vec![prefix];
                    code.extend(bare);
                    let (vm, mut cpu) = setup_quirk(CpuModel::I80386, mode, &code, Some(ignore));
                    let (plain_vm, mut plain) = setup(CpuModel::I80386, mode, bare);
                    fixture::steps(&mut cpu, &vm, 1);
                    fixture::steps(&mut plain, &plain_vm, 1);
                    assert_eq!(cpu.cpu.rip, plain.cpu.rip + 1);
                    cpu.cpu.rip = plain.cpu.rip;
                    assert_eq!(
                        format!("{:?}", cpu.cpu),
                        format!("{:?}", plain.cpu),
                        "{mode:?}/{code:02x?}/{ignore}"
                    );
                }
            }
        }
    }
}

#[test]
fn i386_rep_quirk_rejects_other_models_and_application_profiles() {
    for model in MODELS.into_iter().map(Some).chain([None]) {
        let mut vm = Vm::new(VmConfig::flat(0x1000));
        if let Some(model) = model {
            vm.set_cpu_model(model);
        }
        for ignore in [false, true] {
            let result = vm.set_i386_ignore_rep_non_string(ignore);
            if model == Some(CpuModel::I80386) {
                assert_eq!(result, Ok(()));
            } else {
                assert_eq!(result, Err(I386RepQuirkError { model }));
            }
        }
    }
    #[allow(deprecated)]
    {
        let mut vm = Vm::new(VmConfig::flat(0x1000));
        vm.set_legacy_cpu_model(x86jit_core::LegacyCpuModel {
            generation: x86jit_core::CpuGeneration::I386,
            model_semantics_id: CpuModel::I80386.profile().semantics_id,
        });
        assert_eq!(
            vm.set_i386_ignore_rep_non_string(true),
            Err(I386RepQuirkError { model: None })
        );
    }
    for mode in [CpuMode::Long64, CpuMode::Compat32, CpuMode::Real16] {
        let mut vm = Vm::new(VmConfig::flat(0x1000));
        vm.set_cpu_mode(mode);
        assert_eq!(
            vm.set_i386_ignore_rep_non_string(true),
            Err(I386RepQuirkError { model: None })
        );
    }
}

#[test]
fn i386_rep_quirk_identity_is_inherited_and_resets_with_model_selection() {
    let mut vm = Vm::new(VmConfig::flat(0x1000));
    vm.set_cpu_model(CpuModel::I80386);
    let default = vm.legacy_cpu_model().unwrap();
    assert_eq!(
        default.model_semantics_id,
        CpuModel::I80386.profile().semantics_id
    );
    vm.set_i386_ignore_rep_non_string(true).unwrap();
    let ignored = vm.legacy_cpu_model().unwrap();
    assert_ne!(ignored.model_semantics_id, default.model_semantics_id);
    assert_eq!(ignored.generation, default.generation);
    let fork = vm.fork_with_backend(Box::new(InterpreterBackend)).unwrap();
    assert_eq!(fork.legacy_cpu_model(), Some(ignored));
    assert_eq!(fork.new_vcpu().legacy_cpu_model(), Some(ignored));
    vm.set_i386_ignore_rep_non_string(false).unwrap();
    assert_eq!(vm.legacy_cpu_model(), Some(default));
    vm.set_i386_ignore_rep_non_string(true).unwrap();
    vm.set_cpu_model(CpuModel::I80386);
    assert_eq!(vm.legacy_cpu_model(), Some(default));
    vm.set_i386_ignore_rep_non_string(true).unwrap();
    vm.set_cpu_model(CpuModel::I80486Dx);
    assert_eq!(
        vm.legacy_cpu_model().unwrap().model_semantics_id,
        CpuModel::I80486Dx.profile().semantics_id
    );
}

#[test]
#[should_panic(expected = "select model quirks before vCPU creation")]
fn i386_rep_quirk_locks_after_vcpu_creation_even_after_drop() {
    let mut vm = Vm::new(VmConfig::flat(0x1000));
    vm.set_cpu_model(CpuModel::I80386);
    drop(vm.new_vcpu());
    vm.set_i386_ignore_rep_non_string(true).unwrap();
}

#[test]
fn i386_rep_quirk_fork_preserves_decode_policy() {
    let (vm, cpu) = setup_quirk(CpuModel::I80386, Mode::Real, &[0xf3, 0x40], Some(true));
    let fork = vm.fork_with_backend(Box::new(InterpreterBackend)).unwrap();
    let mut child = fork.new_vcpu();
    child.cpu = cpu.cpu.clone();
    fixture::steps(&mut child, &fork, 1);
    assert_eq!(child.cpu.gpr[0], 0x1235);
    assert_eq!(child.cpu.rip, 0x1002);
}
