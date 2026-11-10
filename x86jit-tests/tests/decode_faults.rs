//! Decode faults preserve the completed prefix (spec.md §5.2, §7.3).

use x86jit_core::interp::{step_one, RetireInfo};
use x86jit_core::lift::{lift_block, lift_one, lift_region, FetchAddr, LiftError};
use x86jit_core::{
    AccessKind, Backend, CpuMode, CpuState, Exit, InterpreterBackend, Prot, Reg, RegionCaps,
    RegionKind, StepResult, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;
const END: u64 = 0x2000;
const CAPS: RegionCaps = RegionCaps {
    max_blocks: 16,
    max_icount: 256,
};
const INVALID: &[u8] = &[0xf0, 0x90]; // LOCK NOP: iced-invalid in both flat modes.
const GAPS: &[&[u8]] = &[
    &[0x0f, 0x08],                   // INVD: decoded, but not lifted.
    &[0xf0, 0x11, 0x54, 0x88, 0x08], // LOCK ADC [RAX+RCX*4+8], EDX: emits EA ops before failing.
];

fn vm(tier: usize, mode: CpuMode) -> Vm {
    let backend: Box<dyn Backend> = match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        2 => Box::new(JitBackend::with_superblocks(CAPS)),
        _ => unreachable!(),
    };
    let mut vm = Vm::with_backend(VmConfig::flat(0x4000), backend);
    vm.set_cpu_mode(mode);
    vm.set_tier_up_after(None);
    vm.map(CODE, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
    vm
}

fn prefix(n: usize, region: bool) -> Vec<u8> {
    let mut bytes = if region {
        // JMP loop; loop: INC EDX; CMP EDX,3; JB loop.
        // The back-edge forces actual region compilation, including its exit arm.
        vec![0xeb, 0x00, 0xff, 0xc2, 0x83, 0xfa, 0x03, 0x72, 0xf9]
    } else {
        vec![]
    };
    for _ in 0..n {
        bytes.extend_from_slice(&[0xff, 0xc3]); // INC EBX
    }
    bytes
}

fn bad_instruction(tier: usize, bad: &[u8], vector: Option<u8>) {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for n in [0, 1, 3] {
            let vm = vm(tier, mode);
            let mut bytes = prefix(n, tier == 2);
            let fault = CODE + bytes.len() as u64;
            bytes.extend_from_slice(bad);
            vm.write_bytes(CODE, &bytes).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rbx, 10);
            for _ in 0..2 {
                let exit = cpu.run(&vm, Some(30));
                if let Some(expected) = vector {
                    // SDM Vol 3A §7.15, Events 6 and 13: #UD and #GP are
                    // faults with saved IP on the instruction.
                    assert!(
                        matches!(exit, Exit::Exception { addr, vector, .. } if addr == fault && vector == expected),
                        "{mode:?}, N={n}: {exit:?}"
                    );
                } else {
                    match exit {
                        Exit::UnknownInstruction {
                            addr, bytes, len, ..
                        } => {
                            assert_eq!(addr, fault);
                            assert_eq!(&bytes[..len as usize], bad);
                        }
                        other => panic!("{mode:?}, N={n}: {other:?}"),
                    }
                }
                assert_eq!(cpu.reg(Reg::Rip), fault);
                assert_eq!(cpu.reg(Reg::Rbx), 10 + n as u64);
                if tier == 0 {
                    assert_eq!(cpu.retired_instructions(), n as u64);
                }
            }
            if tier == 2 {
                assert!(vm.cache.regions() > 0, "fixture must compile a region");
                assert_eq!(cpu.reg(Reg::Rdx), 3);
            }
        }
    }
}

macro_rules! tiers {
    ($interp:ident, $jit:ident, $region:ident, $body:expr) => {
        #[test]
        fn $interp() {
            ($body)(0);
        }
        #[test]
        fn $jit() {
            ($body)(1);
        }
        #[test]
        fn $region() {
            ($body)(2);
        }
    };
}

tiers!(
    invalid_opcode_interp,
    invalid_opcode_jit,
    invalid_opcode_region,
    |tier| bad_instruction(tier, INVALID, Some(6))
);
tiers!(
    unsupported_prefix_interp,
    unsupported_prefix_jit,
    unsupported_prefix_region,
    |tier| {
        for gap in GAPS {
            bad_instruction(tier, gap, None);
        }
    }
);

// SDM Vol 2A §§2.2.1, 2.3.11; Vol 3A §7.15, Event 13:
// exceeding 15 bytes raises #GP(0), with saved IP on the instruction.
fn overlength(tier: usize) {
    for prefix in [0x66, 0x67, 0x2e, 0xf3] {
        let mut prefixes = vec![prefix; 15];
        prefixes.push(0x90);
        bad_instruction(tier, &prefixes, Some(13));
    }

    // The opcode fits; its immediate would extend beyond byte 15.
    let mut immediate = vec![0x66; 13];
    immediate.extend_from_slice(&[0xb8, 0x34, 0x12]);
    bad_instruction(tier, &immediate, Some(13));
}

tiers!(
    overlength_interp,
    overlength_jit,
    overlength_region,
    overlength
);

#[test]
fn overlength_lift_one_and_step_one() {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        let vm = vm(0, mode);
        // No byte 16 is mapped: exhausting the length limit still raises #GP.
        let addr = END - 15;
        vm.write_bytes(addr, &[0x66; 15]).unwrap();
        let block = lift_one(&vm.mem, FetchAddr::flat(addr), mode).unwrap();
        assert!(matches!(
            block.ops.last(),
            Some(x86jit_core::ir::IrOp::Trap {
                vector: 13,
                advance: 0
            })
        ));
        let mut cpu = CpuState {
            rip: addr,
            ..CpuState::default()
        };
        assert!(matches!(
            step_one(&vm.mem, &mut cpu, mode, &mut vec![], &mut RetireInfo::default()),
            StepResult::Exit(Exit::Exception { addr: fault, vector: 13, .. }) if fault == addr
        ));
        assert_eq!(cpu.rip, addr);
    }
}

#[test]
fn invalid_opcode_at_length_limit_stays_ud() {
    // SDM Vol 3A §7.15, Event 6: LOCK on NOP is invalid, even when its
    // complete encoding occupies exactly 15 bytes (Vol 2B, NOP, #UD).
    for invalid in [INVALID, &[0x0f, 0x0b]] {
        // UD2: SDM Vol 2B, UD, Operation.
        let mut bytes = vec![0x66; 13];
        bytes.extend_from_slice(invalid);
        for tier in 0..3 {
            bad_instruction(tier, &bytes, Some(6));
        }
    }
}

#[test]
fn valid_instruction_at_length_limit_executes() {
    // SDM Vol 2A §2.3.11: the maximum length is inclusive.
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for tier in 0..3 {
            let vm = vm(tier, mode);
            let mut bytes = prefix(1, tier == 2);
            bytes.extend_from_slice(&[0x66; 14]);
            bytes.extend_from_slice(&[0x90, 0xf4]);
            vm.write_bytes(CODE, &bytes).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            assert!(matches!(cpu.run(&vm, Some(30)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rbx), 1);
            assert_eq!(cpu.reg(Reg::Rip), CODE + bytes.len() as u64);
            if tier == 2 {
                assert!(vm.cache.regions() > 0);
            }
        }
    }
}

fn length_boundary(tier: usize) {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for kind in [
            None,
            Some(RegionKind::Trap),
            Some(RegionKind::Ram),
            Some(RegionKind::Rom),
        ] {
            for available in [14, 15] {
                let mut vm = vm(tier, mode);
                if let Some(kind) = kind {
                    vm.map(END, 0x1000, Prot::RX, kind).unwrap();
                    vm.write_bytes(END, &[0x90, 0xf4]).unwrap();
                }
                let mut bytes = prefix(1, tier == 2);
                let start = END - available - bytes.len() as u64;
                let fault = start + bytes.len() as u64;
                // SDM Vol 2A §2.2.1: 40h is a REX prefix only in Long64.
                let prefix_byte = if mode == CpuMode::Long64 { 0x40 } else { 0x66 };
                bytes.extend(std::iter::repeat_n(prefix_byte, available as usize));
                vm.write_bytes(start, &bytes).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, start);
                let exit = cpu.run(&vm, Some(30));
                // SDM Vol 3A §7.15, Events 13 and 14; Vol 2A §2.3.11.
                // Fifteen prefixes exhaust the length limit without fetching byte 16.
                let expected_rip = if available == 15 {
                    assert!(
                        matches!(exit, Exit::Exception { addr, vector: 13, .. } if addr == fault),
                        "{mode:?}, {kind:?}: {exit:?}"
                    );
                    fault
                } else if matches!(kind, Some(RegionKind::Ram | RegionKind::Rom)) {
                    assert!(matches!(exit, Exit::Hlt), "{mode:?}, {kind:?}: {exit:?}");
                    END + 2
                } else {
                    assert!(
                        matches!(
                            exit,
                            Exit::UnmappedMemory {
                                addr: END,
                                access: AccessKind::Execute,
                                ..
                            }
                        ),
                        "{mode:?}, {kind:?}: {exit:?}"
                    );
                    fault
                };
                assert_eq!(cpu.reg(Reg::Rip), expected_rip);
                assert_eq!(cpu.reg(Reg::Rbx), 1);
                if tier == 2 {
                    assert!(vm.cache.regions() > 0);
                }
            }
        }
    }
}

tiers!(
    length_boundary_interp,
    length_boundary_jit,
    length_boundary_region,
    length_boundary
);

fn fetch_boundary(tier: usize, adjacent: bool) {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for n in [0, 3] {
            let mut vm = vm(tier, mode);
            if adjacent {
                // Deliberately map out of order and split one instruction three ways.
                for (addr, len) in [(END + 1, 2), (END, 1), (END + 3, 0xffd)] {
                    vm.map(addr, len, Prot::RX, RegionKind::Ram).unwrap();
                }
            }
            let mut bytes = prefix(n, tier == 2);
            let start = END - 2 - bytes.len() as u64;
            let fault = END - 2;
            bytes.extend_from_slice(if adjacent {
                &[0xb8, 0x78, 0x56, 0x34, 0x12, 0xf4] // MOV EAX,12345678h; HLT
            } else {
                &[0xb8, 0x78]
            });
            for (i, byte) in bytes.iter().enumerate() {
                vm.write_bytes(start + i as u64, &[*byte]).unwrap();
            }
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, start);
            cpu.set_reg(Reg::Rax, 0x42);
            let exit = cpu.run(&vm, Some(30));
            if adjacent {
                assert!(matches!(exit, Exit::Hlt), "{mode:?}: {exit:?}");
                assert_eq!(cpu.reg(Reg::Rax), 0x1234_5678);
                assert_eq!(cpu.reg(Reg::Rip), fault + 6);
            } else {
                // SDM Vol 3A §7.15, Event 14; AMD APM Vol 2 §8.2.15:
                // saved IP names the instruction, CR2 names the inaccessible byte.
                for exit in [exit, cpu.run(&vm, Some(30))] {
                    assert!(
                        matches!(
                            exit,
                            Exit::UnmappedMemory {
                                addr: END,
                                access: AccessKind::Execute,
                                ..
                            }
                        ),
                        "{mode:?}: {exit:?}"
                    );
                }
                assert_eq!(cpu.reg(Reg::Rip), fault);
                assert_eq!(cpu.reg(Reg::Rax), 0x42);
            }
            assert_eq!(cpu.reg(Reg::Rbx), n as u64);
            if tier == 2 {
                assert!(vm.cache.regions() > 0, "fixture must compile a region");
            }
        }
    }
}

tiers!(
    fetch_unmapped_interp,
    fetch_unmapped_jit,
    fetch_unmapped_region,
    |tier| fetch_boundary(tier, false)
);
tiers!(
    fetch_adjacent_interp,
    fetch_adjacent_jit,
    fetch_adjacent_region,
    |tier| fetch_boundary(tier, true)
);

#[test]
fn lift_prefix_discards_partial_instruction_ops() {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        let vm = vm(0, mode);
        let mut bytes = prefix(1, false);
        bytes.extend_from_slice(GAPS[1]);
        vm.write_bytes(CODE, &bytes).unwrap();
        let one = lift_one(&vm.mem, FetchAddr::flat(CODE), mode).unwrap();
        let block = lift_block(&vm.mem, FetchAddr::flat(CODE), mode).unwrap();
        assert_eq!((block.guest_len, block.icount), (2, 1));
        assert_eq!(format!("{:?}", block.ops), format!("{:?}", one.ops));
        let region = lift_region(&vm.mem, CODE, CAPS, mode).unwrap();
        assert_eq!(region.blocks[0].guest_len, 2);
    }
}

#[test]
fn lift_one_and_step_one_classify_decode_failures() {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        let vm = vm(0, mode);
        vm.write_bytes(CODE, INVALID).unwrap();
        vm.write_bytes(CODE + 16, GAPS[0]).unwrap();
        vm.write_bytes(END - 2, &[0xb8, 0x78]).unwrap();
        assert!(matches!(
            lift_one(&vm.mem, FetchAddr::flat(CODE + 16), mode),
            Err(LiftError::Unsupported { .. })
        ));
        assert!(matches!(
            lift_one(&vm.mem, FetchAddr::flat(END - 2), mode),
            Err(LiftError::DecodeFault { addr: END })
        ));
        for addr in [CODE, CODE + 16, END - 2] {
            let mut cpu = CpuState {
                rip: addr,
                ..CpuState::default()
            };
            let result = step_one(
                &vm.mem,
                &mut cpu,
                mode,
                &mut vec![],
                &mut RetireInfo::default(),
            );
            match (addr, result) {
                (
                    CODE,
                    StepResult::Exit(Exit::Exception {
                        addr: CODE,
                        vector: 6,
                        ..
                    }),
                ) => {}
                (a, StepResult::Exit(Exit::UnknownInstruction { addr: b, .. }))
                    if a == CODE + 16 && a == b => {}
                (
                    a,
                    StepResult::Exit(Exit::UnmappedMemory {
                        addr: END,
                        access: AccessKind::Execute,
                        ..
                    }),
                ) if a == END - 2 => {}
                (_, StepResult::Exit(exit)) => panic!("{addr:#x}: {exit:?}"),
                _ => panic!("expected an exit at {addr:#x}"),
            }
            assert_eq!(cpu.rip, addr);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod native {
    use super::*;
    use x86jit_tests::native::{run_native_fault, NativeFault};
    use x86jit_tests::oracle::{run_with_backend, VectorInput};
    use x86jit_tests::vector::{Access, CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec};

    const PAGE_END: u64 = 0x21_1000;

    fn witness(fault: NativeFault) {
        let fetch = matches!(fault, NativeFault::PageFault { .. });
        let overlength = matches!(fault, NativeFault::GeneralProtection);
        for n in [0, 3] {
            let mut bytes = prefix(n, false);
            let entry = if fetch {
                PAGE_END - 2 - bytes.len() as u64
            } else {
                0x21_0000
            };
            let rip = entry + bytes.len() as u64;
            if overlength {
                bytes.extend_from_slice(&[0x66; 15]);
                bytes.push(0x90);
            } else {
                bytes.extend_from_slice(if fetch { &[0xb8, 0x78] } else { INVALID });
            }
            let mut cpu_init = CpuSnapshot::default();
            cpu_init.gpr[3] = 10;
            cpu_init.gpr[0] = 0x42;
            let input = VectorInput {
                cpu_init,
                mem_init: vec![MemChunk {
                    addr: entry,
                    bytes,
                    kind: MemKind::Ram,
                }],
                entry,
                run: RunSpec::UntilExit,
            };
            // SDM Vol 3A §7.15, Events 6, 13 and 14; AMD APM Vol 2 §§8.2.7,
            // 8.2.15. The oracle checks signal, si_code and vector before accepting.
            let native = run_native_fault(&input, fault).expect("capture the designated CPU fault");
            if fetch {
                assert!(matches!(
                    native.exit,
                    ExitKind::UnmappedMemory {
                        addr: PAGE_END,
                        access: Access::Execute
                    }
                ));
            } else {
                assert!(
                    matches!(native.exit, ExitKind::Exception { addr, vector } if addr == rip && vector == if overlength { 13 } else { 6 })
                );
            }
            assert_eq!(native.cpu.rip, rip);
            assert_eq!(native.cpu.gpr[3], 10 + n as u64);
            assert_eq!(native.cpu.gpr[0], 0x42);
            for backend in [
                Box::new(InterpreterBackend) as Box<dyn Backend>,
                Box::new(JitBackend::new()),
            ] {
                let engine = run_with_backend(&input, backend);
                assert_eq!(engine.exit, native.exit);
                assert_eq!(engine.cpu.rip, native.cpu.rip);
                assert_eq!(engine.cpu.gpr, native.cpu.gpr);
            }
        }
    }

    #[test]
    fn native_invalid_opcode_is_sigill() {
        witness(NativeFault::InvalidOpcode);
    }

    #[test]
    fn native_overlength_is_sigsegv() {
        witness(NativeFault::GeneralProtection);
    }

    #[test]
    fn native_truncated_fetch_is_sigsegv() {
        witness(NativeFault::PageFault { addr: PAGE_END });
    }
}
