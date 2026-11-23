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
use x86jit_core::{CpuModel, ModelSystem, SegmentCache, Vcpu, Vm};

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
    let (vm, mut cpu) = fixture::setup(model, code);
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
    // Intel486 PRM §2.4 p. 2-16; NEC UM §12 p. 12-177.
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
