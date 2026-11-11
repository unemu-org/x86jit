//! Physical A20 witnesses: SDM Vol 3A §11.7.13.4 (physical bit 20),
//! Vol 3B §§23.1.1, 25.33.1 (real-address wrap); 80386 PRM §14.7 item 18;
//! I286 PRM Appendix C item 16. Decision-22 applies the board gate to I286/I386.
use x86jit_core::{
    AccessKind, AddressSpace, Backend, CachedBlock, CpuGeneration, CpuMode, CpuState, Exit,
    IrBlock, Knowledge, LegacyCpuModel, MemConsistency, Prot, RegionKind, ResetOptions, SegmentId,
    StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

const GENERATIONS: [CpuGeneration; 2] = [CpuGeneration::I286, CpuGeneration::I386];
struct NoBackend;
impl Backend for NoBackend {
    fn materialize(
        &self,
        _: &IrBlock,
        _: MemConsistency,
        _: Option<(u64, u64)>,
        _: u64,
    ) -> CachedBlock {
        panic!("legacy execution reached application backend")
    }
}
fn setup(generation: CpuGeneration, code: &[u8], selectors: [u16; 6]) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x400000), Box::new(NoBackend));
    vm.set_legacy_cpu_model(LegacyCpuModel {
        generation,
        model_semantics_id: 1,
    });
    vm.map(0, 0x400000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry(selectors);
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}
fn steps(cpu: &mut Vcpu, vm: &Vm, count: u64) {
    let run = cpu.run_steps(vm, count, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), count, "{:?}", run.stop());
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{:?}",
        run.stop()
    );
}
fn bytes(vm: &Vm, addr: u64, count: usize) -> Vec<u8> {
    let mut result = vec![0; count];
    vm.read_bytes(addr, &mut result).unwrap();
    result
}

#[test]
fn fetch_ffff_0010_and_toggle_between_runs() {
    for generation in GENERATIONS {
        let (vm, mut cpu) = setup(generation, &[], [0xffff, 0, 0, 0, 0, 0]);
        vm.write_bytes(0, b"\xb8\x34\x12").unwrap();
        vm.write_bytes(0x100000, b"\xb8\x78\x56").unwrap();
        for enabled in [true, false, true, false] {
            cpu.cpu.rip = 0x10;
            cpu.set_a20_enabled(enabled);
            assert_eq!(cpu.a20_enabled(), enabled);
            assert_eq!(cpu.pending_a20_enabled(), None);
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[0], if enabled { 0x5678 } else { 0x1234 });
        }
        // Returning to either retained context must still revalidate physical bytes.
        vm.write_bytes(0, b"\xb8\xbc\x9a").unwrap();
        cpu.cpu.rip = 0x10;
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x9abc);
    }
}

#[test]
fn fetch_instruction_straddling_ffff_000f() {
    for generation in GENERATIONS {
        let (vm, mut cpu) = setup(generation, &[], [0xffff, 0, 0, 0, 0, 0]);
        vm.write_bytes(0xfffff, b"\xb8").unwrap();
        vm.write_bytes(0, b"\x34\x12").unwrap();
        vm.write_bytes(0x100000, b"\x78\x56").unwrap();
        for enabled in [false, true] {
            cpu.cpu.rip = 0xf;
            cpu.set_a20_enabled(enabled);
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[0], if enabled { 0x5678 } else { 0x1234 });
        }
    }
}

#[test]
fn fetch_policy_and_fault_address_follow_the_physical_alias() {
    for generation in GENERATIONS {
        let (mut vm, mut cpu) = setup(generation, &[], [0xffff, 0, 0, 0, 0, 0]);
        vm.write_bytes(0, b"\x43").unwrap();
        vm.write_bytes(0x100000, b"\x43").unwrap();
        vm.set_fetch_policy(Box::new(|page| {
            if page == 0 {
                x86jit_core::FetchVerdict::DenyNotPresent
            } else {
                x86jit_core::FetchVerdict::Allow
            }
        }));
        cpu.cpu.rip = 0x10;
        steps(&mut cpu, &vm, 1);
        cpu.cpu.rip = 0x10;
        cpu.set_a20_enabled(false);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::UnmappedMemory {
                    addr: 0,
                    access: AccessKind::Execute,
                    ..
                })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!(cpu.cpu.gpr[3], 1);
        assert_bus_fault(run.stop(), 0x100000, 0);
        cpu.set_a20_enabled(true);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[3], 2);
    }
}

fn assert_bus_fault(stop: &StepStop, linear: u64, physical: u64) {
    let StepStop::Exit(exit) = stop else {
        panic!("{stop:?}")
    };
    let metadata = exit.fault_metadata().expect("bus fault evidence");
    let Knowledge::Known(address) = metadata.linear_address() else {
        panic!("missing linear address: {metadata:?}")
    };
    assert_eq!(address.value(), linear);
    assert_eq!(address.space(), AddressSpace::Linear);
    assert_eq!(metadata.physical_address(), Knowledge::Known(physical));
}

#[test]
fn gated_fetch_and_data_faults_keep_both_address_spaces() {
    for generation in GENERATIONS {
        // Fetch a later byte across the physical discontinuity as well as the
        // first byte at linear 1 MiB. Test absent mappings and trap fetches.
        for offset in [0xf, 0x10] {
            for denied in [false, true] {
                let (mut vm, mut cpu) = setup(generation, &[], [0xffff, 0, 0, 0, 0, 0]);
                vm.write_bytes(0xfffff, b"\xb8").unwrap();
                vm.unmap(0, 0x400000).unwrap();
                vm.map(0x1000, 0x3ff000, Prot::RWX, RegionKind::Ram)
                    .unwrap();
                if denied {
                    vm.map(0, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                }
                cpu.cpu.rip = offset;
                cpu.set_a20_enabled(false);
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 0);
                assert!(matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnmappedMemory {
                        addr: 0,
                        access: AccessKind::Execute,
                        ..
                    })
                ));
                assert_bus_fault(run.stop(), 0x100000, 0);
            }
        }
        for code in [
            b"\xa1\x10\x00",
            b"\xa3\x10\x00",
            b"\xa1\x0f\x00",
            b"\xa3\x0f\x00",
        ] {
            let (mut vm, mut cpu) = setup(generation, code, [0, 0xffff, 0, 0, 0, 0]);
            vm.unmap(0, 0x400000).unwrap();
            vm.map(0x1000, 0x3ff000, Prot::RWX, RegionKind::Ram)
                .unwrap();
            cpu.set_a20_enabled(false);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::UnmappedMemory { addr: 0, .. })
            ));
            assert_bus_fault(run.stop(), 0x100000, 0);
        }
    }
}

#[test]
fn data_reads_and_writes_at_and_across_one_mib() {
    for generation in GENERATIONS {
        for offset in [0xfu16, 0x10] {
            for enabled in [false, true] {
                let (vm, mut cpu) = setup(
                    generation,
                    b"\xa1\x00\x00\xa3\x00\x00",
                    [0, 0xffff, 0, 0, 0, 0],
                );
                vm.write_bytes(0x1001, &offset.to_le_bytes()).unwrap();
                vm.write_bytes(0x1004, &offset.to_le_bytes()).unwrap();
                vm.write_bytes(0xfffff, b"\xaa\xbb\xcc").unwrap();
                vm.write_bytes(0, b"\x11\x22").unwrap();
                cpu.set_a20_enabled(enabled);
                steps(&mut cpu, &vm, 1);
                let expected = match (offset, enabled) {
                    (0xf, false) => 0x11aa,
                    (0xf, true) => 0xbbaa,
                    (_, false) => 0x2211,
                    (_, true) => 0xccbb,
                };
                assert_eq!(cpu.cpu.gpr[0], expected);
                cpu.cpu.gpr[0] = 0x5678;
                steps(&mut cpu, &vm, 1);
                let low = if offset == 0xf {
                    0xfffff
                } else if enabled {
                    0x100000
                } else {
                    0
                };
                let high = if enabled {
                    0xffff0 + u64::from(offset) + 1
                } else if offset == 0xf {
                    0
                } else {
                    1
                };
                assert_eq!(bytes(&vm, low, 1), [0x78]);
                assert_eq!(bytes(&vm, high, 1), [0x56]);
                if !enabled {
                    assert_eq!(bytes(&vm, 0x100000, 2), [0xbb, 0xcc]);
                }
            }
        }
    }
}

#[test]
fn strings_read_write_and_rep_crossing() {
    for generation in GENERATIONS {
        for enabled in [false, true] {
            let (vm, mut cpu) = setup(generation, b"\xf3\xa5", [0, 0xffff, 0xffff, 0, 0, 0]);
            vm.write_bytes(0xfffff, b"\xaa\xbb\xcc\xdd\xee").unwrap();
            vm.write_bytes(0, b"\x11\x22\x33\x44").unwrap();
            cpu.set_a20_enabled(enabled);
            cpu.cpu.gpr[1] = 2;
            cpu.cpu.gpr[6] = 0xf;
            cpu.cpu.gpr[7] = 0x20;
            steps(&mut cpu, &vm, 2);
            assert_eq!(cpu.cpu.gpr[1], 0);
            assert_eq!(cpu.cpu.gpr[6], 0x13);
            let destination = if enabled { 0x100010 } else { 0x10 };
            assert_eq!(
                bytes(&vm, destination, 4),
                if enabled {
                    vec![0xaa, 0xbb, 0xcc, 0xdd]
                } else {
                    vec![0xaa, 0x11, 0x22, 0x33]
                }
            );
            // LODSW and STOSW at FFFF:0010 use the same gate.
            vm.write_bytes(0x1000, b"\xad\xab").unwrap();
            cpu.cpu.rip = 0x1000;
            cpu.cpu.gpr[6] = 0x10;
            cpu.cpu.gpr[7] = 0x30;
            steps(&mut cpu, &vm, 2);
            assert_eq!(cpu.cpu.gpr[0], if enabled { 0xccbb } else { 0x2211 });
            assert_eq!(
                bytes(&vm, if enabled { 0x100020 } else { 0x20 }, 2),
                if enabled {
                    vec![0xbb, 0xcc]
                } else {
                    vec![0x11, 0x22]
                }
            );
        }
    }
}

#[test]
fn stack_push_and_pop_at_and_across_wrap() {
    for generation in GENERATIONS {
        for offset in [0xf, 0x10] {
            for enabled in [false, true] {
                let (vm, mut cpu) = setup(generation, b"\x50\x5b", [0, 0, 0, 0xffff, 0, 0]);
                cpu.set_a20_enabled(enabled);
                cpu.cpu.gpr[0] = 0x1234;
                cpu.cpu.gpr[4] = offset + 2;
                steps(&mut cpu, &vm, 2);
                assert_eq!(cpu.cpu.gpr[3], 0x1234);
                assert_eq!(cpu.cpu.gpr[4], offset + 2);
                let address = 0xffff0 + offset;
                let low = if enabled {
                    address
                } else {
                    address & !(1 << 20)
                };
                let high = if enabled {
                    address + 1
                } else {
                    (address + 1) & !(1 << 20)
                };
                assert_eq!(bytes(&vm, low, 1), [0x34]);
                assert_eq!(bytes(&vm, high, 1), [0x12]);
            }
        }
    }
}

#[test]
fn table_operand_and_descriptor_reads_and_accessed_write_are_gated() {
    for generation in GENERATIONS {
        for enabled in [false, true] {
            // LGDT's base dword straddles physical FFFFFh.
            let (vm, mut cpu) = setup(generation, b"\x0f\x01\x16\x0d\x00", [0, 0xffff, 0, 0, 0, 0]);
            vm.write_bytes(0xffffd, b"\x0f\x00\xf8").unwrap();
            vm.write_bytes(0, b"\xff\x0f\x00").unwrap();
            vm.write_bytes(0x100000, b"\x2f\x00\x00").unwrap();
            cpu.set_a20_enabled(enabled);
            steps(&mut cpu, &vm, 1);
            assert_eq!(
                cpu.cpu.legacy.gdtr.base,
                if enabled { 0x2ff8 } else { 0xffff8 }
            );
            assert_eq!(cpu.cpu.legacy.gdtr.limit, 15);
            // SDM Vol 3A §§3.4.3–3.4.5; I286 PRM §§6.3, 7.4:
            // MOV DS loads the descriptor and writes its accessed bit through A20.
            vm.write_bytes(0x1000, b"\xb8\x08\x00\x8e\xd8").unwrap();
            vm.write_bytes(0, b"\xff\xff\x00\x20\x00\x92\x00\x00")
                .unwrap();
            vm.write_bytes(0x100000, b"\xff\xff\x00\x30\x00\x92\x00\x00")
                .unwrap();
            cpu.cpu.rip = 0x1000;
            cpu.cpu.legacy.cr0 = 1;
            cpu.cpu.legacy.gdtr.base = 0xffff8;
            steps(&mut cpu, &vm, 2);
            assert_eq!(
                cpu.cpu.legacy.segments[SegmentId::Ds as usize].base,
                if enabled { 0x3000 } else { 0x2000 }
            );
            assert_eq!(bytes(&vm, if enabled { 0x100005 } else { 5 }, 1), [0x93]);
            assert_eq!(bytes(&vm, if enabled { 5 } else { 0x100005 }, 1), [0x92]);
        }
    }
}

#[test]
fn relocated_ivt_and_interrupt_stack_use_the_gate() {
    // SDM Vol 3B §23.1.4; I286 PRM §10.2: retained IDTR base.
    for generation in GENERATIONS {
        for enabled in [false, true] {
            let (vm, mut cpu) = setup(generation, b"\xcd\x20", [0, 0, 0, 0xffff, 0, 0]);
            vm.write_bytes(0x80, b"\x00\x20\x00\x00").unwrap();
            vm.write_bytes(0x100080, b"\x00\x30\x00\x00").unwrap();
            cpu.cpu.legacy.idtr.base = 0x100000;
            cpu.cpu.gpr[4] = 0x16;
            cpu.set_a20_enabled(enabled);
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.rip, if enabled { 0x3000 } else { 0x2000 });
            assert_eq!(cpu.cpu.gpr[4], 0x10);
            assert_eq!(bytes(&vm, if enabled { 0x100000 } else { 0 }, 2), [2, 0x10]);
        }
    }
}

#[test]
fn only_bit_twenty_is_masked_in_protected_mode() {
    for generation in GENERATIONS {
        let (vm, mut cpu) = setup(generation, b"\xa1\x10\x00", [0; 6]);
        cpu.cpu.legacy.cr0 = 1;
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].base = 0x2ffff0;
        vm.write_bytes(0x200000, b"\x34\x12").unwrap();
        vm.write_bytes(0x300000, b"\x78\x56").unwrap();
        for enabled in [false, true] {
            cpu.cpu.rip = 0x1000;
            cpu.set_a20_enabled(enabled);
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[0], if enabled { 0x5678 } else { 0x1234 });
        }
    }
}

#[test]
fn crossing_two_mib_preserves_the_next_higher_address_bit() {
    for generation in GENERATIONS {
        let (vm, mut cpu) = setup(generation, b"\xa1\x0f\x00\xa3\x0f\x00", [0; 6]);
        cpu.cpu.legacy.cr0 = 1;
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].base = 0x1ffff0;
        vm.write_bytes(0xfffff, &[0xaa]).unwrap();
        vm.write_bytes(0x1fffff, &[0xbb]).unwrap();
        vm.write_bytes(0x200000, &[0xcc]).unwrap();
        cpu.set_a20_enabled(false);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[0], 0xccaa);
        cpu.cpu.gpr[0] = 0x1234;
        steps(&mut cpu, &vm, 1);
        assert_eq!(bytes(&vm, 0xfffff, 1), [0x34]);
        assert_eq!(bytes(&vm, 0x200000, 1), [0x12]);
        assert_eq!(bytes(&vm, 0x1fffff, 1), [0xbb]);
    }
}

#[test]
fn dword_split_handles_every_boundary_position() {
    for offset in [0xdu16, 0xe, 0xf, 0x10] {
        let (vm, mut cpu) = setup(
            CpuGeneration::I386,
            b"\x66\xa1\x00\x00\x66\xa3\x00\x00",
            [0, 0xffff, 0, 0, 0, 0],
        );
        vm.write_bytes(0x1002, &offset.to_le_bytes()).unwrap();
        vm.write_bytes(0x1006, &offset.to_le_bytes()).unwrap();
        vm.write_bytes(0xffffd, b"\x11\x22\x33").unwrap();
        vm.write_bytes(0, b"\x44\x55\x66\x77").unwrap();
        cpu.set_a20_enabled(false);
        steps(&mut cpu, &vm, 1);
        let expected = match offset {
            0xd => 0x44332211,
            0xe => 0x55443322,
            0xf => 0x66554433,
            _ => 0x77665544,
        };
        assert_eq!(cpu.cpu.gpr[0], expected);
        cpu.cpu.gpr[0] = 0xaabbccdd;
        steps(&mut cpu, &vm, 1);
        for (i, byte) in [0xdd, 0xcc, 0xbb, 0xaa].into_iter().enumerate() {
            let address = (0xffff0 + u64::from(offset) + i as u64) & !(1 << 20);
            assert_eq!(bytes(&vm, address, 1), [byte]);
        }
    }
}

#[test]
fn split_store_faults_before_committing_ram_and_names_physical_address() {
    for generation in GENERATIONS {
        let (mut vm, mut cpu) = setup(generation, b"\xa3\x0f\x00", [0, 0xffff, 0, 0, 0, 0]);
        vm.write_bytes(0xfffff, &[0xaa]).unwrap();
        vm.unmap(0, 0x400000).unwrap();
        vm.map(0x1000, 0x3ff000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        cpu.cpu.gpr[0] = 0x1234;
        cpu.set_a20_enabled(false);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::UnmappedMemory {
                    addr: 0,
                    access: AccessKind::Write,
                    ..
                })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!(bytes(&vm, 0xfffff, 1), [0xaa]);
        assert_eq!(cpu.cpu.rip, 0x1000);
    }
}

#[test]
fn split_mmio_read_and_write_complete_with_physical_parts() {
    for generation in GENERATIONS {
        let (mut vm, mut cpu) = setup(
            generation,
            b"\xa1\x0f\x00\xa3\x0f\x00",
            [0, 0xffff, 0, 0, 0, 0],
        );
        vm.unmap(0, 0x400000).unwrap();
        vm.map(0x1000, 0x3ff000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.map(0, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.write_bytes(0xfffff, &[0xaa]).unwrap();
        cpu.set_a20_enabled(false);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioRead {
                    addr: 0,
                    size: 1,
                    ..
                })
            ),
            "{:?}",
            run.stop()
        );
        cpu.complete_mmio_read(0xbb);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[0], 0xbbaa);
        cpu.cpu.gpr[0] = 0x1234;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0,
                    size: 1,
                    value: 0x12,
                    ..
                })
            ),
            "{:?}",
            run.stop()
        );
        cpu.complete_mmio_write();
        steps(&mut cpu, &vm, 1);
        assert_eq!(bytes(&vm, 0xfffff, 1), [0x34]);
    }
}

#[test]
fn changing_line_defers_until_mmio_read_retires_and_same_value_is_a_noop() {
    for generation in GENERATIONS {
        let (mut vm, mut cpu) = setup(generation, b"\xa1\x10\x00", [0, 0xffff, 0, 0, 0, 0]);
        vm.unmap(0, 0x400000).unwrap();
        vm.map(0, 0x100000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x100000, 0x1000, Prot::RW, RegionKind::Trap)
            .unwrap();
        vm.write_bytes(0, b"\x34\x12").unwrap();
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(
            run.stop(),
            StepStop::Exit(Exit::MmioRead {
                addr: 0x100000,
                size: 2,
                ..
            })
        ));
        cpu.complete_mmio_read(0x5678);
        cpu.set_a20_enabled(true);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x5678);
        cpu.cpu.rip = 0x1000;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })));
        cpu.complete_mmio_read(0x9abc);
        cpu.set_a20_enabled(false);
        assert!(cpu.a20_enabled());
        assert_eq!(cpu.pending_a20_enabled(), Some(false));
        cpu.set_a20_enabled(false);
        cpu.set_a20_enabled(true);
        assert_eq!(cpu.pending_a20_enabled(), Some(false));
        assert_eq!(
            cpu.run_steps(&vm, 0, StepOptions::default())
                .unwrap()
                .steps(),
            0
        );
        assert!(cpu.a20_enabled());
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x9abc);
        assert!(!cpu.a20_enabled());
        assert_eq!(cpu.pending_a20_enabled(), None);
        cpu.cpu.rip = 0x1000;
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x1234);
    }
}

#[test]
fn changing_line_keeps_completed_device_write_and_pending_acknowledgment() {
    for generation in GENERATIONS {
        for acknowledge_first in [false, true] {
            let (mut vm, mut cpu) = setup(
                generation,
                b"\xa3\x0f\x00\xa1\x10\x00",
                [0, 0xffff, 0, 0, 0, 0],
            );
            vm.unmap(0, 0x400000).unwrap();
            vm.map(0x1000, 0xfefff, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x100000, 0x300000, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.map(0, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
            vm.map(0xfffff, 1, Prot::RW, RegionKind::Trap).unwrap();
            vm.write_bytes(0x100000, b"\x78\x56").unwrap();
            cpu.cpu.gpr[0] = 0x1234;
            cpu.set_a20_enabled(false);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::MmioWrite {
                        addr: 0xfffff,
                        size: 1,
                        value,
                        ..
                    }) if *value as u8 == 0x34
                ),
                "{:?}",
                run.stop()
            );
            cpu.complete_mmio_write();
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0,
                    size: 1,
                    value: 0x12,
                    ..
                })
            ));
            if acknowledge_first {
                cpu.complete_mmio_write();
            }
            cpu.set_a20_enabled(true);
            assert!(!cpu.a20_enabled());
            assert_eq!(cpu.pending_a20_enabled(), Some(true));
            if !acknowledge_first {
                cpu.complete_mmio_write();
            }
            // No further device exit: the first write is not replayed, the
            // second acknowledgment is kept, and the next load uses the new line.
            steps(&mut cpu, &vm, 2);
            assert_eq!(cpu.cpu.gpr[0], 0x5678);
            assert!(cpu.a20_enabled());
            assert_eq!(cpu.pending_a20_enabled(), None);
        }
    }
}

#[test]
fn changing_line_keeps_movsw_consumed_device_read() {
    for generation in GENERATIONS {
        for acknowledge_first in [false, true] {
            let (mut vm, mut cpu) = setup(
                generation,
                b"\xa5\xa1\x10\x00",
                [0, 0xffff, 0xffff, 0, 0, 0],
            );
            vm.unmap(0, 0x400000).unwrap();
            vm.map(0, 0x100000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x100000, 0x1000, Prot::RW, RegionKind::Trap)
                .unwrap();
            vm.write_bytes(0, b"\x34\x12").unwrap();
            vm.write_bytes(0x10, b"\xaa\xbb").unwrap();
            cpu.cpu.gpr[6] = 0x10;
            cpu.cpu.gpr[7] = 0x20;
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioRead {
                    addr: 0x100000,
                    size: 2,
                    ..
                })
            ));
            cpu.complete_mmio_read(0x5678);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0x100010,
                    size: 2,
                    value: 0x5678,
                    ..
                })
            ));
            if acknowledge_first {
                cpu.complete_mmio_write();
            }
            cpu.set_a20_enabled(false);
            assert!(cpu.a20_enabled());
            assert_eq!(cpu.pending_a20_enabled(), Some(false));
            if !acknowledge_first {
                cpu.complete_mmio_write();
            }
            steps(&mut cpu, &vm, 2);
            assert_eq!((cpu.cpu.gpr[6], cpu.cpu.gpr[7]), (0x12, 0x22));
            assert_eq!(bytes(&vm, 0x10, 2), [0xaa, 0xbb]);
            assert_eq!(cpu.cpu.gpr[0], 0x1234);
            assert!(!cpu.a20_enabled());
            assert_eq!(cpu.pending_a20_enabled(), None);
        }
    }
}

// SDM Vol 2B, REP/REPE/REPZ/REPNE/REPNZ, Description and Operation,
// pp. 4-564–568: suspension preserves the next element and updated count;
// REP services pending interrupts before executing the next element.
#[test]
fn line_change_at_rep_iteration_boundary_uses_new_line_for_next_element() {
    for generation in GENERATIONS {
        for queued_before_pause in [false, true] {
            for same_run in [false, true] {
                let (mut vm, mut cpu) =
                    setup(generation, b"\xf3\xa5\xa1\x10\x00", [0, 0xffff, 0, 0, 0, 0]);
                vm.unmap(0, 0x400000).unwrap();
                vm.map(0, 0x100000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.map(0x100000, 0x1000, Prot::RW, RegionKind::Trap)
                    .unwrap();
                vm.write_bytes(0, b"\x34\x12\xaa\xbb").unwrap();
                cpu.cpu.gpr[1] = 2;
                cpu.cpu.gpr[6] = 0x10;
                cpu.cpu.gpr[7] = 0x2000;
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(matches!(
                    run.stop(),
                    StepStop::Exit(Exit::MmioRead { addr: 0x100000, .. })
                ));
                if queued_before_pause {
                    cpu.set_a20_enabled(false);
                    assert_eq!(cpu.pending_a20_enabled(), Some(false));
                }
                cpu.complete_mmio_read(0x5678);
                // A queued change can also take effect between iterations in one run.
                if same_run && queued_before_pause {
                    steps(&mut cpu, &vm, 2);
                } else {
                    steps(&mut cpu, &vm, 1);
                    assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[1]), (0x1000, 1));
                    assert_eq!((cpu.cpu.gpr[6], cpu.cpu.gpr[7]), (0x12, 0x2002));
                    assert_eq!(cpu.retired_instructions(), 0);
                    if !queued_before_pause {
                        cpu.set_a20_enabled(false);
                    }
                    assert!(!cpu.a20_enabled());
                    assert_eq!(cpu.pending_a20_enabled(), None);
                    steps(&mut cpu, &vm, 1);
                }
                assert_eq!(bytes(&vm, 0x2000, 4), [0x78, 0x56, 0xaa, 0xbb]);
                assert_eq!(cpu.cpu.gpr[1], 0);
                assert_eq!(cpu.retired_instructions(), 1);
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.gpr[0], 0x1234);
                assert!(!cpu.a20_enabled());
                assert_eq!(cpu.pending_a20_enabled(), None);
            }
        }
    }
}

#[test]
fn irq_handler_changes_line_between_rep_iterations() {
    // SDM Vol 2B REP, p. 4-564; Vol 3B §23.1.4: the saved IP resumes
    // the REP at its next element after the handler returns with IRET.
    for generation in GENERATIONS {
        for queued_before_pause in [false, true] {
            let (mut vm, mut cpu) = setup(generation, b"\xf3\xa5", [0, 0xffff, 0, 0, 0, 0]);
            vm.unmap(0, 0x400000).unwrap();
            vm.map(0, 0x100000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x100000, 0x1000, Prot::RW, RegionKind::Trap)
                .unwrap();
            vm.write_bytes(0, b"\x34\x12\xaa\xbb").unwrap();
            vm.write_bytes(0x80, b"\x00\x18\x00\x00").unwrap();
            vm.write_bytes(0x1800, b"\xe6\x92\xcf").unwrap(); // OUT; IRET
            cpu.cpu.gpr[1] = 2;
            cpu.cpu.gpr[6] = 0x10;
            cpu.cpu.gpr[7] = 0x2000;
            cpu.cpu.flags.if_ = true;
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })));
            if queued_before_pause {
                cpu.set_a20_enabled(false);
            }
            cpu.complete_mmio_read(0x5678);
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.pending_a20_enabled(), None);
            assert_eq!(cpu.a20_enabled(), !queued_before_pause);
            assert!(cpu.irq_ready());
            cpu.inject_irq(0x20);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            assert!(matches!(run.stop(), StepStop::Exit(Exit::PortIo { .. })));
            assert!(!cpu.has_pending_irq());
            assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[1]), (0x1802, 1));
            assert_eq!(bytes(&vm, 0x7ffa, 2), [0, 0x10]);
            // The controller OUT toggles the effective line in both cases.
            let handler_enabled = queued_before_pause;
            cpu.set_a20_enabled(handler_enabled);
            assert_eq!(cpu.a20_enabled(), handler_enabled);
            assert_eq!(cpu.pending_a20_enabled(), None);
            let run = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
            let second = if handler_enabled {
                assert_eq!(run.steps(), 1); // IRET, then the REP's device read.
                assert!(matches!(
                    run.stop(),
                    StepStop::Exit(Exit::MmioRead { addr: 0x100002, .. })
                ));
                cpu.complete_mmio_read(0x9abc);
                steps(&mut cpu, &vm, 1);
                [0xbc, 0x9a]
            } else {
                assert_eq!(run.steps(), 2); // IRET and the REP's RAM element.
                assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
                [0xaa, 0xbb]
            };
            assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[1]), (0x1002, 0));
            assert_eq!(bytes(&vm, 0x2000, 4), [0x78, 0x56, second[0], second[1]]);
            assert_eq!(cpu.retired_instructions(), 3);
        }
    }
}

#[test]
fn mmio_divide_fault_applies_queued_line_and_discards_retry() {
    // SDM Vol 2A DIV, pp. 3-264–266; Vol 3A §7.15, Event 0, p. 7-25:
    // a zero divisor raises #DE with the faulting IP and unchanged registers.
    for generation in GENERATIONS {
        let (mut vm, mut cpu) = setup(generation, b"\xf7\x36\x10\x00", [0, 0xffff, 0, 0, 0, 0]);
        vm.unmap(0, 0x400000).unwrap();
        vm.map(0, 0x100000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x100000, 0x1000, Prot::RW, RegionKind::Trap)
            .unwrap();
        vm.write_bytes(0, b"\x02\x00").unwrap();
        cpu.cpu.gpr[0] = 6;
        let before = cpu.cpu.gpr;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(
            run.stop(),
            StepStop::Exit(Exit::MmioRead { addr: 0x100000, .. })
        ));
        cpu.set_a20_enabled(false);
        cpu.complete_mmio_read(0);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(matches!(
            run.stop(),
            StepStop::Exit(Exit::Exception {
                addr: 0x1000,
                vector: 0,
                ..
            })
        ));
        assert_eq!(cpu.cpu.gpr, before);
        assert_eq!(cpu.retired_instructions(), 0);
        assert!(!cpu.a20_enabled());
        assert_eq!(cpu.pending_a20_enabled(), None);
        assert!(!cpu.cpu.mmio_retry_active);
        assert_eq!(cpu.cpu.mmio_parts_len, 0);
        assert_eq!(cpu.cpu.pending_mmio, None);
        // Retrying the faulting instruction now reads the RAM alias, not the
        // consumed device answer. A fresh fetch must also observe changed code.
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[0], 3);
        cpu.cpu.rip = 0x1000;
        vm.write_bytes(0x1000, b"\xb8\x34\x12").unwrap();
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[0]), (0x1003, 0x1234));
    }
}

#[test]
fn event_delivery_fault_applies_queued_line_and_discards_retry() {
    // Decision-22: an architectural fault abandons the in-flight transfer.
    // I286 PRM Appendix B MOV p. B-74; SDM Vol 3B §23.1.4;
    // 80386 PRM §14.6 Tables 14-1/14-2 pp. 14-5, 14-8:
    // a short IVT limit or invalid stack faults before committing the frame.
    for generation in GENERATIONS {
        for debug in [false, true] {
            for short_idtr in [false, true] {
                let (mut vm, mut cpu) = setup(generation, b"\x90\x90", [0; 6]);
                vm.unmap(0, 0x400000).unwrap();
                vm.map(0, 0x30000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.write_bytes(0x1000, b"\x90\x90").unwrap();
                vm.map(0x30000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                cpu.cpu.legacy.idtr.base = 0x30000;
                let vector: u8 = if debug {
                    cpu.cpu.rflags_system = 0x100;
                    1
                } else {
                    cpu.cpu.flags.if_ = true;
                    cpu.inject_irq(0x20);
                    0x20
                };
                let address = 0x30000 + u64::from(vector) * 4;
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), u64::from(debug));
                assert!(matches!(
                    run.stop(),
                    StepStop::Exit(Exit::MmioRead { addr, .. }) if *addr == address
                ));
                cpu.set_a20_enabled(false);
                assert!(cpu.a20_enabled());
                assert_eq!(cpu.pending_a20_enabled(), Some(false));
                cpu.complete_mmio_read(0x1800);
                let fault = if short_idtr {
                    cpu.cpu.legacy.idtr.limit = u16::from(vector) * 4 + 2;
                    if generation == CpuGeneration::I386 {
                        8
                    } else {
                        13
                    }
                } else {
                    cpu.cpu.legacy.segments[SegmentId::Ss as usize].limit = 0x7ff0;
                    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    assert_eq!(run.steps(), 0);
                    assert!(matches!(
                        run.stop(),
                        StepStop::Exit(Exit::MmioRead { addr, .. }) if *addr == address + 2
                    ));
                    cpu.complete_mmio_read(0);
                    if generation == CpuGeneration::I386 {
                        12
                    } else {
                        13
                    }
                };
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 0);
                assert!(matches!(
                    run.stop(),
                    StepStop::Exit(Exit::Exception { vector, .. }) if *vector == fault
                ));
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
                    (0, 0x1000 + u64::from(debug), 0x8000)
                );
                assert_eq!(cpu.retired_instructions(), u64::from(debug));
                assert_eq!(cpu.has_pending_irq(), !debug);
                assert!(!cpu.a20_enabled());
                assert_eq!(cpu.pending_a20_enabled(), None);
                assert!(!cpu.cpu.mmio_retry_active);
                assert_eq!(cpu.cpu.mmio_parts_len, 0);
                assert_eq!(cpu.cpu.pending_mmio, None);
                assert!(!cpu.cpu.pending_mmio_write);
                cpu.cpu.flags.if_ = false;
                cpu.cpu.rflags_system = 0;
                cpu.cpu.legacy.idtr.limit = 0x3ff;
                cpu.cpu.legacy.segments[SegmentId::Ss as usize].limit = 0xffff;
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.rip, 0x1001 + u64::from(debug));
                assert_eq!(cpu.cpu.gpr[4], 0x8000);
                assert_eq!(cpu.has_pending_irq(), !debug);
            }
        }
    }
}

#[test]
fn divide_fault_handler_fetch_failure_does_not_leave_line_pending() {
    // SDM Vol 3A §7.15, Event 0; Vol 3B §23.1.4. Legacy exceptions
    // return outward (decision-18 §4); the embedder explicitly routes vector 0
    // through the supported real interrupt-delivery seam before handler fetch.
    for generation in GENERATIONS {
        let (mut vm, mut cpu) = setup(generation, b"\xf7\x36\x10\x00", [0, 0xffff, 0, 0, 0, 0]);
        vm.unmap(0, 0x400000).unwrap();
        vm.map(0, 0x100000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x100000, 0x1000, Prot::RW, RegionKind::Trap)
            .unwrap();
        vm.write_bytes(0, b"\x10\x00\xff\xff").unwrap(); // Vector 0: FFFF:0010.
        vm.set_fetch_policy(Box::new(|page| {
            if page == 0 {
                x86jit_core::FetchVerdict::DenyNotPresent
            } else {
                x86jit_core::FetchVerdict::Allow
            }
        }));
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })));
        cpu.set_a20_enabled(false);
        cpu.complete_mmio_read(0);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(
            run.stop(),
            StepStop::Exit(Exit::Exception { vector: 0, .. })
        ));
        assert!(!cpu.a20_enabled());
        assert_eq!(cpu.pending_a20_enabled(), None);
        cpu.cpu.flags.if_ = true;
        cpu.inject_irq(0);
        for _ in 0..2 {
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::UnmappedMemory {
                    addr: 0,
                    access: AccessKind::Execute,
                    ..
                })
            ));
            assert_bus_fault(run.stop(), 0x100000, 0);
            assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0xffff, 0x10));
            assert!(!cpu.has_pending_irq());
            assert!(!cpu.a20_enabled());
            assert_eq!(cpu.pending_a20_enabled(), None);
        }
    }
}

#[test]
fn reset_applies_queued_board_line_after_abandoning_retry() {
    let (mut vm, mut cpu) = setup(
        CpuGeneration::I386,
        b"\xa1\x10\x00",
        [0, 0xffff, 0, 0, 0, 0],
    );
    vm.unmap(0, 0x400000).unwrap();
    vm.map(0, 0x100000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x100000, 0x1000, Prot::RW, RegionKind::Trap)
        .unwrap();
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert!(matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })));
    cpu.set_a20_enabled(false);
    cpu.complete_mmio_read(0x5678);
    cpu.reset_execution_state(ResetOptions::default());
    assert!(!cpu.a20_enabled());
    assert_eq!(cpu.pending_a20_enabled(), None);
    vm.write_bytes(0, b"\x34\x12").unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[0], 0x1234);
}

#[test]
fn port_exit_can_toggle_line_and_reset_preserves_it() {
    for generation in GENERATIONS {
        let (vm, mut cpu) = setup(generation, b"\xe6\x80\xa1\x10\x00", [0, 0xffff, 0, 0, 0, 0]);
        assert!(cpu.a20_enabled());
        vm.write_bytes(0, b"\x34\x12").unwrap();
        vm.write_bytes(0x100000, b"\x78\x56").unwrap();
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(run.stop(), StepStop::Exit(Exit::PortIo { .. })));
        cpu.set_a20_enabled(false);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x1234);
        cpu.reset_execution_state(ResetOptions::default());
        assert!(!cpu.a20_enabled());
        let other = vm.new_vcpu();
        assert!(other.a20_enabled());
        cpu.set_a20_enabled(true);
        cpu.reset_execution_state(ResetOptions::default());
        assert!(cpu.a20_enabled());
    }
}

#[test]
fn application_real16_ignores_the_legacy_line_for_fetch_data_string_and_stack() {
    let mut vm = Vm::new(VmConfig::flat(0x200000));
    vm.set_cpu_mode(CpuMode::Real16);
    vm.map(0, 0x200000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0, b"\x11\x22\x33\x44").unwrap();
    vm.write_bytes(0x100000, b"\xa1\x20\x00\xad\xab\x50\x5b")
        .unwrap();
    vm.write_bytes(0x100010, b"\x78\x56").unwrap();
    let mut cpu = vm.new_vcpu();
    for enabled in [false, true] {
        cpu.cpu.cs = 0xffff;
        cpu.cpu.ds = 0xffff;
        cpu.cpu.es = 0xffff;
        cpu.cpu.ss = 0xffff;
        cpu.cpu.rip = 0x10;
        cpu.cpu.gpr[6] = 0x20;
        cpu.cpu.gpr[7] = 0x30;
        cpu.cpu.gpr[4] = 0x42;
        cpu.set_a20_enabled(enabled);
        steps(&mut cpu, &vm, 5);
        assert_eq!(cpu.cpu.gpr[0], 0x5678);
        assert_eq!(cpu.cpu.gpr[3], 0x5678);
        assert_eq!(bytes(&vm, 0x100020, 2), [0x78, 0x56]);
        assert_eq!(bytes(&vm, 0x100030, 2), [0x78, 0x56]);
        assert_eq!(bytes(&vm, 0, 4), [0x11, 0x22, 0x33, 0x44]);
    }
}

#[test]
fn application_long64_and_compat32_ignore_the_line() {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        let mut vm = Vm::new(VmConfig::flat(0x200000));
        vm.set_cpu_mode(mode);
        vm.map(0, 0x200000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x100000, b"\xbb\x20\x00\x10\x00\x8b\x03\xf4")
            .unwrap();
        vm.write_bytes(0x100020, b"\x78\x56\x34\x12").unwrap();
        vm.write_bytes(0x20, b"\xaa\xbb\xcc\xdd").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_a20_enabled(false);
        cpu.cpu.rip = 0x100000;
        assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
        assert_eq!(cpu.cpu.gpr[0], 0x12345678);
    }
}
