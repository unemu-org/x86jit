//! Decision-16 witnesses. Hardware restart/shadow rules: SDM Vol 3A §§7.6,
//! 7.8.1, 7.8.3; string progress: Vol 2B §4.3 REP/REPE/REPNE. Budgets and
//! port-handoff attribution are engine contracts, independent of an oracle.
use x86jit_core::{
    CpuMode, Exit, InterpreterBackend, PortDir, Prot, Reg, RegionKind, StepOptions, StepRun,
    StepRunError, StepStop, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

fn setup(code: &[u8]) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x40000), Box::new(InterpreterBackend));
    vm.set_cpu_mode(CpuMode::Real16);
    for page in (0..0x40000).step_by(4096) {
        vm.map(page, 4096, Prot::RWX, RegionKind::Ram).unwrap();
    }
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Cs, 0x100);
    cpu.set_reg(Reg::Ss, 0x300);
    cpu.set_reg(Reg::Rsp, 0x800);
    (vm, cpu)
}
fn run(vm: &Vm, cpu: &mut Vcpu, n: u64) -> StepRun {
    let r = cpu.run_steps(vm, n, StepOptions::default()).unwrap();
    assert!(r.steps() <= n, "budget={n}, {r:?}");
    r
}
fn budget(vm: &Vm, cpu: &mut Vcpu, n: u64) {
    let r = run(vm, cpu, n);
    assert_eq!(r.steps(), n);
    assert!(
        matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{r:?}"
    );
}
fn handler(vm: &mut Vm, vector: u8, bytes: &[u8]) {
    vm.write_bytes(0x6000, bytes).unwrap();
    vm.write_bytes(u64::from(vector) * 4, &[0, 0, 0, 6])
        .unwrap();
}
fn flags(cpu: &mut Vcpu, if_: bool, df: bool) {
    let mut f = cpu.flags();
    f.if_ = if_;
    f.df = df;
    cpu.set_flags(f);
}
fn trap(vm: &mut Vm, addr: u64) {
    vm.unmap(addr, 4096).unwrap();
    vm.map(addr, 4096, Prot::RW, RegionKind::Trap).unwrap();
}
fn byte(vm: &Vm, addr: u64) -> u8 {
    let mut b = [0];
    vm.read_bytes(addr, &mut b).unwrap();
    b[0]
}

#[test]
fn cut_cached_hundred_instruction_block() {
    let mut code = vec![0x40; 100]; // inc ax
    code.extend_from_slice(&[0xe6, 0x80, 0xf4]);
    let (vm, mut cpu) = setup(&code);
    // Populate the ordinary cached block, then reuse precisely that block.
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::PortIo { .. }));
    cpu.set_reg(Reg::Rip, 0);
    cpu.set_reg(Reg::Rax, 0);
    let before = cpu.retired_instructions();
    budget(&vm, &mut cpu, 7);
    assert_eq!(cpu.reg(Reg::Rip), 7);
    assert_eq!(cpu.reg(Reg::Rax), 7);
    assert_eq!(cpu.retired_instructions() - before, 7);
}

#[test]
fn cut_before_fault_and_device_and_future_decode() {
    for tail in [&[0xa0, 0, 0x70][..], &[0xe6, 0x80], &[0x0f, 0x01, 0xc1]] {
        let mut code = vec![0x40; 7];
        code.extend_from_slice(tail);
        let (mut vm, mut cpu) = setup(&code);
        vm.unmap(0x7000, 4096).unwrap();
        budget(&vm, &mut cpu, 7);
        assert_eq!(cpu.reg(Reg::Rip), 7);
        assert_eq!(cpu.reg(Reg::Rax), 7);
    }
}

#[test]
fn partitions_include_zero_and_device_events() {
    type Snapshot = (Vec<u64>, u64, Vec<u8>, Vec<(u16, PortDir, u64)>);
    fn execute(slices: &[u64]) -> Snapshot {
        // inc ax; stosb repeated in source, with an IN and OUT in the stream.
        let mut code = vec![];
        for _ in 0..35 {
            code.extend_from_slice(&[0x40, 0xaa]);
        }
        code.extend_from_slice(&[0xe4, 0x80, 0xe6, 0x81, 0xf4]);
        let (vm, mut cpu) = setup(&code);
        cpu.set_reg(Reg::Rdi, 0x8000);
        cpu.set_reg(Reg::Rcx, 0x1234);
        let mut steps = 0;
        let mut events = vec![];
        for n in slices.iter().copied().cycle().take(1000) {
            let r = run(&vm, &mut cpu, n);
            steps += r.steps();
            assert_eq!(cpu.reg(Reg::Rcx), 0x1234);
            assert_eq!(cpu.reg(Reg::Rdi), 0x8000 + steps.min(70) / 2);
            assert_eq!(
                cpu.reg(Reg::Rip),
                match steps {
                    0..=70 => steps,
                    71 => 72,
                    72 => 74,
                    73 => 75,
                    _ => panic!("unexpected progress: {steps}"),
                }
            );
            if matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)) {
                assert_eq!(r.steps(), n);
            }
            let expected_events = [(0x80, PortDir::In, 0), (0x81, PortDir::Out, 0x5a)];
            let issued = steps.saturating_sub(70).min(2) as usize;
            let returned_event =
                usize::from(matches!(r.stop(), StepStop::Exit(Exit::PortIo { .. })));
            assert_eq!(events, expected_events[..issued - returned_event]);
            if let StepStop::Exit(Exit::PortIo {
                port, dir, value, ..
            }) = r.stop()
            {
                assert_eq!((*port, *dir, *value), expected_events[issued - 1]);
            }
            match r.into_stop() {
                StepStop::Exit(Exit::BudgetExhausted) => {}
                StepStop::Exit(Exit::PortIo {
                    port, dir, value, ..
                }) => {
                    events.push((port, dir, value));
                    if dir == PortDir::In {
                        cpu.complete_port_in(0x5a);
                    }
                }
                StepStop::Exit(Exit::Hlt) => {
                    assert_eq!(steps, 73);
                    assert_eq!(cpu.retired_instructions(), steps);
                    let regs = [Reg::Rax, Reg::Rdi, Reg::Rip].map(|r| cpu.reg(r)).to_vec();
                    let mut mem = vec![0; 35];
                    vm.read_bytes(0x8000, &mut mem).unwrap();
                    return (regs, cpu.flags().to_rflags(), mem, events);
                }
                other => panic!("{other:?}"),
            }
        }
        panic!("no halt")
    }
    let expected = execute(&[100]);
    for slices in [&[1][..], &[7], &[0, 1, 0, 7, 100, 0, 3]] {
        assert_eq!(execute(slices), expected);
    }
}

#[test]
fn self_branch_int_fault_delivery_and_self_iret() {
    let (vm, mut cpu) = setup(&[0xeb, 0xfe]);
    for n in [1, 7, 100] {
        let before = cpu.retired_instructions();
        budget(&vm, &mut cpu, n);
        assert_eq!(cpu.reg(Reg::Rip), 0);
        assert_eq!(cpu.retired_instructions() - before, n);
    }
    for (code, vector, total) in [
        (&[0xcd, 0x40][..], 0x40, 2),
        (&[0xcc][..], 3, 2),
        (&[0x0f, 0x0b][..], 6, 1),
        (&[0xf6, 0xf3][..], 0, 1),
    ] {
        let (mut vm, mut cpu) = setup(code);
        handler(&mut vm, vector, &[0xf4]);
        let r = run(&vm, &mut cpu, 10);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(r.steps(), total);
        assert_eq!(cpu.retired_instructions(), total);
    }
    // SDM Vol 2A §3.3 IRET: successful self-target return still completes.
    let (vm, mut cpu) = setup(&[0xcf]);
    vm.write_bytes(0x3800, &[0, 0, 0, 1, 2, 0]).unwrap();
    budget(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Rip), 0);
    assert_eq!(cpu.retired_instructions(), 1);
    assert_eq!(cpu.reg(Reg::Rsp), 0x806);
}

#[test]
fn rep_cut_resume_all_string_ops_both_directions() {
    for op in [0xa4, 0xaa, 0xac, 0xa6, 0xae] {
        // movs/stos/lods/cmps/scas
        for df in [false, true] {
            for count in [0, 1, 2, 7] {
                let (vm, mut cpu) = setup(&[0xf3, op, 0xf4]);
                vm.write_bytes(0x8000, &[0x55; 32]).unwrap();
                vm.write_bytes(0x9000, &[0x55; 32]).unwrap();
                flags(&mut cpu, false, df);
                cpu.set_reg(Reg::Rax, 0x55);
                cpu.set_reg(Reg::Rcx, 0xabc0000 | count);
                let start = if df { 0x8010 } else { 0x8000 };
                cpu.set_reg(Reg::Rsi, start);
                cpu.set_reg(Reg::Rdi, start + 0x1000);
                if op == 0xac {
                    cpu.set_reg(Reg::Rax, 0x00ab_cdee);
                    for i in 0..count {
                        let addr = if df { start - i } else { start + i };
                        vm.write_bytes(addr, &[0x10 + i as u8]).unwrap();
                    }
                }
                budget(&vm, &mut cpu, 0);
                for i in 0..count.max(1) {
                    budget(&vm, &mut cpu, 1);
                    let last = i + 1 == count.max(1);
                    assert_eq!(
                        cpu.reg(Reg::Rip),
                        if last { 2 } else { 0 },
                        "op={op:x}, count={count}, i={i}"
                    );
                    assert_eq!(cpu.retired_instructions(), u64::from(last));
                    assert_eq!(cpu.reg(Reg::Rcx), 0xabc0000 | count.saturating_sub(i + 1));
                    if op == 0xac {
                        assert_eq!(
                            cpu.reg(Reg::Rax),
                            if count == 0 {
                                0x00ab_cdee
                            } else {
                                0x00ab_cd00 | (0x10 + i)
                            }
                        );
                    }
                }
                let advanced = if df { start - count } else { start + count };
                if matches!(op, 0xa4 | 0xac | 0xa6) {
                    assert_eq!(cpu.reg(Reg::Rsi), advanced);
                }
                if matches!(op, 0xa4 | 0xaa | 0xa6 | 0xae) {
                    assert_eq!(cpu.reg(Reg::Rdi), advanced + 0x1000);
                }
                let r = run(&vm, &mut cpu, 1);
                assert_eq!(r.steps(), 1);
                assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
            }
        }
    }
}

#[test]
fn repe_repne_early_end_on_last_allowed_iteration() {
    for op in [0xa6, 0xae] {
        for prefix in [0xf2, 0xf3] {
            let (vm, mut cpu) = setup(&[prefix, op, 0xf4]);
            vm.write_bytes(0x8000, &[3; 4]).unwrap();
            let data = if prefix == 0xf3 {
                [3, 3, 4, 3]
            } else {
                [4, 4, 3, 4]
            };
            vm.write_bytes(0x9000, &data).unwrap();
            cpu.set_reg(Reg::Rsi, 0x8000);
            cpu.set_reg(Reg::Rdi, 0x9000);
            cpu.set_reg(Reg::Rax, 3);
            cpu.set_reg(Reg::Rcx, 4);
            budget(&vm, &mut cpu, 2);
            assert_eq!(cpu.reg(Reg::Rip), 0);
            budget(&vm, &mut cpu, 1);
            assert_eq!(cpu.reg(Reg::Rip), 2);
            assert_eq!(cpu.reg(Reg::Rcx), 1);
            assert_eq!(cpu.retired_instructions(), 1);
        }
    }
}

#[test]
fn rep_address_wrap_and_count_width() {
    // SDM Vol 2B §4.3 REP: CX selects the repeat count at address size 16.
    let (mut vm, mut cpu) = setup(&[0xf3, 0xaa, 0xf4]);
    cpu.set_reg(Reg::Es, 0x1000);
    cpu.set_reg(Reg::Rdi, 0xffff);
    cpu.set_reg(Reg::Rcx, 0x1234ffff);
    cpu.set_reg(Reg::Rax, 0x6a);
    budget(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Rdi), 0);
    assert_eq!(cpu.reg(Reg::Rcx), 0x1234fffe);
    assert_eq!(byte(&vm, 0x1ffff), 0x6a);
    flags(&mut cpu, false, true);
    cpu.set_reg(Reg::Rcx, 1);
    budget(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Rdi), 0xffff);
    assert_eq!(cpu.reg(Reg::Rip), 2);
    // Empty low CX must complete once even when upper RCX is nonzero.
    cpu.set_reg(Reg::Rip, 0);
    cpu.set_reg(Reg::Rcx, 0xffff0000);
    vm.unmap(0x1f000, 4096).unwrap();
    budget(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Rcx), 0xffff0000);
    assert_eq!(cpu.reg(Reg::Rip), 2);
}

#[test]
fn rep_word_cut_resume_and_pointer_wrap_both_directions() {
    // SDM Vol 2A §3.3 CMPS/LODS; Vol 2B §4.3 MOVS/SCAS/STOS, Operation:
    // word strings advance SI/DI by two and LODSW writes AX, preserving upper bits.
    for op in [0xa5, 0xab, 0xad, 0xa7, 0xaf] {
        for df in [false, true] {
            let (vm, mut cpu) = setup(&[0xf3, op, 0xf4]);
            flags(&mut cpu, false, df);
            cpu.set_reg(Reg::Ds, 0x1000);
            cpu.set_reg(Reg::Es, 0x2000);
            cpu.set_reg(Reg::Rax, 0x7654_0000_1234);
            cpu.set_reg(Reg::Rcx, 0xabc0_0002);
            let start = if df { 0u16 } else { 0xfffe };
            let next = |offset: u16| {
                if df {
                    offset.wrapping_sub(2)
                } else {
                    offset.wrapping_add(2)
                }
            };
            cpu.set_reg(Reg::Rsi, u64::from(start));
            cpu.set_reg(Reg::Rdi, u64::from(start));
            for (i, offset) in [start, next(start)].into_iter().enumerate() {
                let value = if i == 0 || matches!(op, 0xab | 0xaf) {
                    0x1234u16
                } else {
                    0xabcd
                };
                vm.write_bytes(0x10000 + u64::from(offset), &value.to_le_bytes())
                    .unwrap();
                let destination = if matches!(op, 0xa5 | 0xab) {
                    0xdead_u16
                } else {
                    value
                };
                vm.write_bytes(0x20000 + u64::from(offset), &destination.to_le_bytes())
                    .unwrap();
            }
            budget(&vm, &mut cpu, 0);
            for (i, offset) in [start, next(start)].into_iter().enumerate() {
                budget(&vm, &mut cpu, 1);
                assert_eq!(
                    cpu.reg(Reg::Rip),
                    if i == 0 { 0 } else { 2 },
                    "op={op:x}, DF={df}"
                );
                assert_eq!(cpu.reg(Reg::Rcx), 0xabc0_0001 - i as u64);
                assert_eq!(cpu.retired_instructions(), i as u64);
                assert_eq!(
                    cpu.reg(Reg::Rsi),
                    u64::from(if matches!(op, 0xa5 | 0xad | 0xa7) {
                        next(offset)
                    } else {
                        start
                    })
                );
                assert_eq!(
                    cpu.reg(Reg::Rdi),
                    u64::from(if op != 0xad { next(offset) } else { start })
                );
                let value = if i == 0 || matches!(op, 0xab | 0xaf) {
                    0x1234u16
                } else {
                    0xabcd
                };
                if op == 0xad {
                    assert_eq!(cpu.reg(Reg::Rax), 0x7654_0000_0000 | u64::from(value));
                } else {
                    assert_eq!(cpu.reg(Reg::Rax), 0x7654_0000_1234);
                }
                if matches!(op, 0xa5 | 0xab) {
                    let mut bytes = [0; 2];
                    vm.read_bytes(0x20000 + u64::from(offset), &mut bytes)
                        .unwrap();
                    assert_eq!(u16::from_le_bytes(bytes), value);
                }
                if matches!(op, 0xa7 | 0xaf) {
                    assert!(cpu.flags().zf);
                }
            }
        }
    }
}

#[test]
fn rep_word_second_element_fault_and_cut_both_directions() {
    use x86jit_core::Knowledge::Known;
    for op in [0xa5, 0xab, 0xad, 0xa7, 0xaf] {
        for df in [false, true] {
            for source in [false, true] {
                if (source && matches!(op, 0xab | 0xaf)) || (!source && op == 0xad) {
                    continue;
                }
                for cut in [false, true] {
                    let (mut vm, mut cpu) = setup(&[0xf3, op, 0xf4]);
                    flags(&mut cpu, false, df);
                    let start = if df { 0x9000 } else { 0x8ffe };
                    let other = if df { 0xb000 } else { 0xaffe };
                    cpu.set_reg(Reg::Rsi, if source { start } else { other });
                    cpu.set_reg(Reg::Rdi, if source { other } else { start });
                    cpu.set_reg(Reg::Rcx, 2);
                    cpu.set_reg(Reg::Rax, 0x1234);
                    for addr in [0x8ffe, 0x9000, 0xaffe, 0xb000] {
                        vm.write_bytes(addr, &0x1234u16.to_le_bytes()).unwrap();
                    }
                    let fault_addr = if df { 0x8ffe } else { 0x9000 };
                    let hole = if df { 0x8000 } else { 0x9000 };
                    vm.unmap(hole, 4096).unwrap();
                    if cut {
                        budget(&vm, &mut cpu, 1);
                        assert_eq!(cpu.reg(Reg::Rip), 0);
                        assert_eq!(cpu.reg(Reg::Rcx), 1);
                    }
                    let r = run(&vm, &mut cpu, 10);
                    assert_eq!(
                        r.steps(),
                        u64::from(!cut),
                        "op={op:x}, DF={df}, source={source}"
                    );
                    let StepStop::Exit(exit @ Exit::UnmappedMemory { addr, .. }) = r.into_stop()
                    else {
                        panic!("expected word fault")
                    };
                    assert_eq!(addr, fault_addr);
                    let m = exit.fault_metadata().unwrap();
                    assert_eq!(m.transfer_size(), Known(2));
                    let restart = m.progress().rep().unwrap();
                    assert_eq!(restart.remaining(), 1);
                    assert_eq!(restart.element_size(), 2);
                    assert_eq!(restart.address_bits(), 16);
                    assert_eq!(restart.direction_flag(), df);
                    assert_eq!(restart.source(), cpu.reg(Reg::Rsi));
                    assert_eq!(restart.destination(), cpu.reg(Reg::Rdi));
                    assert_eq!(cpu.reg(Reg::Rip), 0);
                    assert_eq!(cpu.reg(Reg::Rcx), 1);
                    assert_eq!(cpu.retired_instructions(), 0);
                    vm.map(hole, 4096, Prot::RW, RegionKind::Ram).unwrap();
                    vm.write_bytes(fault_addr, &0x1234u16.to_le_bytes())
                        .unwrap();
                    budget(&vm, &mut cpu, 1);
                    assert_eq!(cpu.reg(Reg::Rip), 2);
                    assert_eq!(cpu.reg(Reg::Rcx), 0);
                    assert_eq!(cpu.retired_instructions(), 1);
                    assert_eq!(restart.remaining(), 1);
                }
            }
        }
    }
}

#[test]
fn conditional_strings_terminate_backwards_at_cut() {
    for op in [0xa6, 0xa7, 0xae, 0xaf] {
        for prefix in [0xf2, 0xf3] {
            let (vm, mut cpu) = setup(&[prefix, op, 0xf4]);
            flags(&mut cpu, false, true);
            let width = if op & 1 == 0 { 1 } else { 2 };
            cpu.set_reg(Reg::Rax, 0x1234);
            cpu.set_reg(Reg::Rsi, 0x8010);
            cpu.set_reg(Reg::Rdi, 0x9010);
            cpu.set_reg(Reg::Rcx, 4);
            for i in 0..4 {
                let equal = (i < 2) == (prefix == 0xf3);
                vm.write_bytes(
                    0x8010 - i * width,
                    &0x1234u16.to_le_bytes()[..width as usize],
                )
                .unwrap();
                let rhs = if equal { 0x1234u16 } else { 0x5678 };
                vm.write_bytes(0x9010 - i * width, &rhs.to_le_bytes()[..width as usize])
                    .unwrap();
            }
            for i in 1..=3 {
                budget(&vm, &mut cpu, 1);
                assert_eq!(cpu.reg(Reg::Rcx), 4 - i);
                assert_eq!(cpu.reg(Reg::Rip), if i == 3 { 2 } else { 0 });
                assert_eq!(cpu.flags().zf, (i < 3) == (prefix == 0xf3));
                assert_eq!(cpu.reg(Reg::Rdi), 0x9010 - i * width);
                assert_eq!(
                    cpu.reg(Reg::Rsi),
                    if matches!(op, 0xa6 | 0xa7) {
                        0x8010 - i * width
                    } else {
                        0x8010
                    }
                );
                assert_eq!(cpu.retired_instructions(), u64::from(i == 3));
            }
        }
    }
}

#[test]
fn lods_accumulator_changes_each_ram_and_mmio_iteration() {
    // SDM Vol 2A §3.3 LODS, Operation: byte/word loads replace AL/AX only.
    for (op, width, mask) in [(0xac, 1, 0xff), (0xad, 2, 0xffff)] {
        for df in [false, true] {
            for mmio in [false, true] {
                let (mut vm, mut cpu) = setup(&[0xf3, op, 0xf4]);
                flags(&mut cpu, false, df);
                if mmio {
                    trap(&mut vm, 0x7000);
                }
                let start = if df { 0x7010 } else { 0x7000 };
                cpu.set_reg(Reg::Rsi, start);
                cpu.set_reg(Reg::Rcx, 2);
                let initial = 0x9876_5432_ffff;
                cpu.set_reg(Reg::Rax, initial);
                for (i, answer) in [0x1234u64, 0xabcd].into_iter().enumerate() {
                    let addr = if df {
                        start - i as u64 * width
                    } else {
                        start + i as u64 * width
                    };
                    let before = cpu.reg(Reg::Rax);
                    if mmio {
                        let r = run(&vm, &mut cpu, 1);
                        assert_eq!(r.steps(), 0);
                        assert!(
                            matches!(r.stop(), StepStop::Exit(Exit::MmioRead { addr: a, size, .. }) if *a == addr && u64::from(*size) == width)
                        );
                        assert_eq!(cpu.reg(Reg::Rax), before);
                        cpu.complete_mmio_read(answer);
                        budget(&vm, &mut cpu, 0);
                        assert_eq!(cpu.reg(Reg::Rax), before);
                    } else {
                        vm.write_bytes(addr, &answer.to_le_bytes()[..width as usize])
                            .unwrap();
                    }
                    budget(&vm, &mut cpu, 1);
                    assert_eq!(cpu.reg(Reg::Rax), (initial & !mask) | (answer & mask));
                    assert_eq!(cpu.reg(Reg::Rcx), 1 - i as u64);
                    assert_eq!(cpu.reg(Reg::Rip), if i == 0 { 0 } else { 2 });
                    assert_eq!(
                        cpu.reg(Reg::Rsi),
                        if df { addr - width } else { addr + width }
                    );
                }
            }
        }
    }
}

#[test]
fn rep_fault_counts_only_completed_prefix_and_retries() {
    for source in [false, true] {
        let (mut vm, mut cpu) = setup(&[0xf3, 0xa4, 0xf4]);
        vm.write_bytes(0x8ffe, &[1, 2]).unwrap();
        vm.write_bytes(0xaffe, &[1, 2]).unwrap();
        cpu.set_reg(Reg::Rsi, if source { 0x8ffe } else { 0xaffe });
        cpu.set_reg(Reg::Rdi, if source { 0xaffe } else { 0x8ffe });
        cpu.set_reg(Reg::Rcx, 3);
        vm.unmap(0x9000, 4096).unwrap();
        budget(&vm, &mut cpu, 2); // No speculative access to the next element.
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 0);
        assert!(matches!(
            r.stop(),
            StepStop::Exit(Exit::UnmappedMemory { addr: 0x9000, .. })
        ));
        assert_eq!(cpu.reg(Reg::Rcx), 1);
        assert_eq!(cpu.reg(Reg::Rip), 0);
        assert_eq!(cpu.retired_instructions(), 0);
        vm.map(0x9000, 4096, Prot::RW, RegionKind::Ram).unwrap();
        budget(&vm, &mut cpu, 1);
        assert_eq!(cpu.reg(Reg::Rip), 2);
        assert_eq!(cpu.retired_instructions(), 1);
    }
}

#[test]
fn exit_attribution_and_pending_in() {
    for (op, kind) in [(0xe6, PortDir::Out), (0xe4, PortDir::In)] {
        let (vm, mut cpu) = setup(&[0x90, op, 0x80, 0xf4]);
        let r = run(&vm, &mut cpu, 2);
        assert_eq!(r.steps(), 2);
        assert!(matches!(r.stop(),StepStop::Exit(Exit::PortIo{dir,..}) if *dir==kind));
        assert_eq!(cpu.reg(Reg::Rip), 3);
        assert_eq!(cpu.retired_instructions(), 2);
        if kind == PortDir::In {
            budget(&vm, &mut cpu, 0);
            assert_eq!(
                cpu.run_steps(&vm, 1, StepOptions::default()).unwrap_err(),
                StepRunError::PendingPortIn
            );
            assert_eq!(cpu.reg(Reg::Rip), 3);
            cpu.complete_port_in(0x42);
            assert_eq!(cpu.retired_instructions(), 2);
        }
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 1);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
    }
    let (mut vm, mut cpu) = setup(&[0x90, 0xa0, 0, 0x70]);
    vm.unmap(0x7000, 4096).unwrap();
    let r = run(&vm, &mut cpu, 7);
    assert_eq!(r.steps(), 1);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::UnmappedMemory { .. })
    ));
    assert_eq!(cpu.reg(Reg::Rip), 1);
}

#[test]
fn scalar_mmio_replay_zero_preserves_answer_and_host_request() {
    let (mut vm, mut cpu) = setup(&[0x90, 0xa0, 0, 0x70, 0x40, 0xf4]);
    trap(&mut vm, 0x7000);
    let r = run(&vm, &mut cpu, 9);
    assert_eq!(r.steps(), 1);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::MmioRead { addr: 0x7000, .. })
    ));
    cpu.complete_mmio_read(0x55);
    cpu.exit_request().request_exit();
    budget(&vm, &mut cpu, 0);
    budget(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Rax), 0x55);
    assert_eq!(cpu.reg(Reg::Rip), 4);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 0);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Interrupted)));
    assert!(cpu.interruption_evidence().is_some());
    budget(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Rax), 0x56);
}

#[test]
fn scalar_mmio_write_ack_replay_respects_budget() {
    let (mut vm, mut cpu) = setup(&[0x90, 0xa2, 0, 0x70, 0x40, 0xf4]);
    trap(&mut vm, 0x7000);
    cpu.set_reg(Reg::Rax, 0x9a);
    let r = run(&vm, &mut cpu, 9);
    assert_eq!(r.steps(), 1);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::MmioWrite {
            addr: 0x7000,
            size: 1,
            value: 0x9a,
            ..
        })
    ));
    assert_eq!(cpu.reg(Reg::Rip), 1);
    assert_eq!(cpu.retired_instructions(), 1);
    budget(&vm, &mut cpu, 0);
    cpu.complete_mmio_write();
    cpu.exit_request().request_exit();
    budget(&vm, &mut cpu, 0);
    assert_eq!(cpu.reg(Reg::Rip), 1);
    assert_eq!(cpu.retired_instructions(), 1);
    budget(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Rip), 4);
    assert_eq!(cpu.reg(Reg::Rax), 0x9a);
    assert_eq!(cpu.retired_instructions(), 2);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 0);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Interrupted)));
    budget(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Rip), 5);
    assert_eq!(cpu.reg(Reg::Rax), 0x9b);
}

#[test]
fn conditional_mmio_distinct_operands_and_early_termination() {
    for op in [0xa6, 0xa7, 0xae, 0xaf] {
        for prefix in [0xf2, 0xf3] {
            for df in [false, true] {
                let (mut vm, mut cpu) = setup(&[prefix, op, 0xf4]);
                trap(&mut vm, 0x7000);
                flags(&mut cpu, false, df);
                cpu.set_reg(Reg::Rsi, 0x7010);
                cpu.set_reg(Reg::Rdi, 0x7010);
                cpu.set_reg(Reg::Rcx, 3);
                cpu.set_reg(Reg::Rax, 0x11);
                let cmps = matches!(op, 0xa6 | 0xa7);
                let width = if op & 1 == 0 { 1 } else { 2 };
                let iterations = if prefix == 0xf3 { 1 } else { 2 };
                for i in 0..iterations {
                    let addr = if df {
                        0x7010 - i * width
                    } else {
                        0x7010 + i * width
                    };
                    let before_flags = cpu.flags().to_rflags();
                    for part in 0..if cmps { 2 } else { 1 } {
                        let r = run(&vm, &mut cpu, 1);
                        assert_eq!(r.steps(), 0);
                        assert!(
                            matches!(r.stop(), StepStop::Exit(Exit::MmioRead { addr: a, size, .. }) if *a == addr && u64::from(*size) == width)
                        );
                        assert_eq!(cpu.reg(Reg::Rcx), 3 - i);
                        assert_eq!(cpu.reg(Reg::Rip), 0);
                        assert_eq!(cpu.flags().to_rflags(), before_flags);
                        // Even at the same address, CMPS has two distinct transfers.
                        let answer = if i == 0 && (!cmps || part == 1) {
                            0x22
                        } else {
                            0x11
                        };
                        cpu.complete_mmio_read(answer);
                        budget(&vm, &mut cpu, 0);
                    }
                    budget(&vm, &mut cpu, 1);
                    assert_eq!(cpu.flags().zf, i != 0);
                    assert_eq!(cpu.reg(Reg::Rcx), 2 - i);
                    let last = i + 1 == iterations;
                    assert_eq!(cpu.reg(Reg::Rip), if last { 2 } else { 0 });
                    assert_eq!(cpu.retired_instructions(), u64::from(last));
                    let next = if df { addr - width } else { addr + width };
                    assert_eq!(cpu.reg(Reg::Rdi), next);
                    assert_eq!(cpu.reg(Reg::Rsi), if cmps { next } else { 0x7010 });
                }
            }
        }
    }
}

#[test]
fn rep_mmio_multitransfer_same_address_and_fresh_iterations() {
    for op in [0xa4, 0xa6] {
        let (mut vm, mut cpu) = setup(&[0xf3, op, 0xf4]);
        trap(&mut vm, 0x7000);
        cpu.set_reg(Reg::Rsi, 0x7000);
        cpu.set_reg(Reg::Rdi, 0x7000);
        cpu.set_reg(Reg::Rcx, 2);
        let mut events = vec![];
        for iteration in 0..2 {
            for part in 0..2 {
                let r = run(&vm, &mut cpu, 1);
                assert_eq!(r.steps(), 0);
                assert_eq!(cpu.retired_instructions(), 0);
                match r.into_stop() {
                    StepStop::Exit(Exit::MmioRead { addr, .. }) => {
                        events.push((addr, false));
                        cpu.complete_mmio_read(0x33);
                    }
                    StepStop::Exit(Exit::MmioWrite { addr, value, .. }) => {
                        assert_eq!(value, 0x33);
                        events.push((addr, true));
                        cpu.complete_mmio_write();
                    }
                    other => panic!("{iteration}/{part}: {other:?}"),
                }
                budget(&vm, &mut cpu, 0);
            }
            budget(&vm, &mut cpu, 1);
            assert_eq!(cpu.reg(Reg::Rcx), 1 - iteration);
            assert_eq!(cpu.reg(Reg::Rip), if iteration == 1 { 2 } else { 0 });
        }
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].0, events[1].0);
        assert_eq!(events[2].0, events[3].0);
        assert_eq!(cpu.retired_instructions(), 1);
    }
}

#[test]
fn rep_recorded_read_survives_ram_fault_without_answer_slot() {
    use x86jit_core::{FaultAccess, Knowledge::Known};
    let (mut vm, mut cpu) = setup(&[0xf3, 0xa4, 0xf4]);
    trap(&mut vm, 0x7000);
    vm.unmap(0x8000, 4096).unwrap();
    cpu.set_reg(Reg::Rsi, 0x7000);
    cpu.set_reg(Reg::Rdi, 0x8000);
    cpu.set_reg(Reg::Rcx, 1);
    flags(&mut cpu, true, false);
    assert!(matches!(
        run(&vm, &mut cpu, 1).stop(),
        StepStop::Exit(Exit::MmioRead { .. })
    ));
    cpu.complete_mmio_read(0x9a);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 0);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::UnmappedMemory { .. })
    ));
    let StepStop::Exit(fault) = r.into_stop() else {
        unreachable!()
    };
    let metadata = fault.fault_metadata().unwrap();
    assert_eq!(metadata.transfer_size(), Known(1));
    assert_eq!(metadata.transfer_access(), Known(FaultAccess::Write));
    assert!(matches!(metadata.linear_address(), Known(a) if a.value() == 0x8000));
    assert_eq!(
        metadata.progress().acknowledged_transfers(),
        &[(0x7000, 0x9a)]
    );
    let restart = metadata.progress().rep().unwrap();
    assert_eq!(restart.source(), 0x7000);
    assert_eq!(restart.destination(), 0x8000);
    assert_eq!(restart.remaining(), 1);
    assert_eq!(restart.element_size(), 1);
    assert_eq!(restart.address_bits(), 16);
    let saved = format!("{metadata:?}");
    cpu.exit_request().request_exit();
    cpu.inject_irq(0x40);
    budget(&vm, &mut cpu, 0);
    let retry = run(&vm, &mut cpu, 1);
    assert_eq!(retry.steps(), 0);
    let StepStop::Exit(retry_fault @ Exit::UnmappedMemory { addr: 0x8000, .. }) = retry.into_stop()
    else {
        panic!("expected replay fault")
    };
    assert_eq!(
        format!("{:?}", retry_fault.fault_metadata().unwrap()),
        saved
    );
    vm.map(0x8000, 4096, Prot::RW, RegionKind::Ram).unwrap();
    budget(&vm, &mut cpu, 1);
    assert_eq!(byte(&vm, 0x8000), 0x9a);
    assert_eq!(format!("{:?}", fault.fault_metadata().unwrap()), saved);
    assert_eq!(
        format!("{:?}", retry_fault.fault_metadata().unwrap()),
        saved
    );
    assert!(cpu.has_pending_irq());
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 0);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Interrupted)));
}

#[test]
fn rep_mmio_fault_metadata_resolves_segment_override_and_word_transfer() {
    use x86jit_core::{
        AddressSpace, FaultAccess,
        Knowledge::{Known, Unknown},
    };
    let (mut vm, mut cpu) = setup(&[0x36, 0xf3, 0xa7, 0xf4]); // ss repe cmpsw
    trap(&mut vm, 0x7000);
    vm.unmap(0x8000, 4096).unwrap();
    cpu.set_reg(Reg::Ds, 0x900);
    cpu.set_reg(Reg::Ss, 0x700);
    cpu.set_reg(Reg::Es, 0x800);
    cpu.set_reg(Reg::Rsi, 0x20);
    cpu.set_reg(Reg::Rdi, 0x20);
    cpu.set_reg(Reg::Rcx, 2);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 0);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::MmioRead {
            addr: 0x7020,
            size: 2,
            ..
        })
    ));
    cpu.complete_mmio_read(0x1234);
    budget(&vm, &mut cpu, 0);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 0);
    let StepStop::Exit(fault @ Exit::UnmappedMemory { addr: 0x8020, .. }) = r.into_stop() else {
        panic!("expected second-operand fault")
    };
    let m = fault.fault_metadata().unwrap();
    assert_eq!(m.real_mode_cs_ip(), Known((0x100, 0)));
    assert_eq!(m.saved_rip(), 0);
    assert_eq!(m.operand_size(), Known(2));
    assert_eq!(m.transfer_size(), Known(2));
    assert_eq!(m.transfer_access(), Known(FaultAccess::Read));
    assert!(
        matches!(m.linear_address(), Known(a) if a.value() == 0x8020 && a.space() == AddressSpace::RealModePhysical)
    );
    assert_eq!(m.progress().acknowledged_transfers(), &[(0x7020, 0x1234)]);
    assert_eq!(m.progress().device_history_complete(), Unknown);
    let rep = m.progress().rep().unwrap();
    assert_eq!(rep.source(), 0x20);
    assert_eq!(rep.destination(), 0x20);
    assert_eq!(rep.remaining(), 2);
    assert_eq!(rep.element_size(), 2);
    assert_eq!(rep.completed_iterations(), Unknown);
    let saved = format!("{m:?}");
    budget(&vm, &mut cpu, 0);
    vm.map(0x8000, 4096, Prot::RW, RegionKind::Ram).unwrap();
    vm.write_bytes(0x8020, &0x5678u16.to_le_bytes()).unwrap();
    budget(&vm, &mut cpu, 1);
    assert!(!cpu.flags().zf);
    assert_eq!(cpu.reg(Reg::Rcx), 1);
    assert_eq!(cpu.reg(Reg::Rip), 3);
    assert_eq!(cpu.reg(Reg::Rsi), 0x22);
    assert_eq!(cpu.reg(Reg::Rdi), 0x22);
    assert_eq!(format!("{:?}", fault.fault_metadata().unwrap()), saved);
}

#[test]
fn queued_irq_sti_hlt_and_budget_precedence() {
    let (mut vm, mut cpu) = setup(&[0xfb, 0xf4, 0x40]);
    handler(&mut vm, 0x40, &[0xf4]);
    cpu.inject_irq(0x40);
    let r = run(&vm, &mut cpu, 2);
    assert_eq!(r.steps(), 2);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
    assert!(cpu.has_pending_irq());
    assert_eq!(cpu.reg(Reg::Rip), 2);
    budget(&vm, &mut cpu, 0);
    assert!(cpu.has_pending_irq());
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 1);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
    assert!(!cpu.has_pending_irq());
    assert_eq!(byte(&vm, 0x3000 + cpu.reg(Reg::Rsp)), 2);
    let (vm, mut cpu) = setup(&[0xf4]);
    cpu.inject_irq(0x40);
    let r = run(&vm, &mut cpu, 1);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
    assert!(cpu.has_pending_irq());
}

#[test]
fn sliced_shadow_successor_and_trapped_successor() {
    for load in [&[0x8e, 0xd0][..], &[0x17][..], &[0xfb][..]] {
        let mut code = load.to_vec();
        code.extend_from_slice(&[0x89, 0xdc, 0x40, 0xf4]);
        let (mut vm, mut cpu) = setup(&code);
        handler(&mut vm, 0x40, &[0xf4]);
        cpu.set_reg(Reg::Rax, 0x300);
        cpu.set_reg(Reg::Rbx, 0x900);
        vm.write_bytes(0x3800, &[0, 3]).unwrap();
        flags(&mut cpu, load != [0xfb], false);
        budget(&vm, &mut cpu, 1);
        cpu.inject_irq(0x40);
        budget(&vm, &mut cpu, 0);
        budget(&vm, &mut cpu, 1);
        assert!(cpu.has_pending_irq());
        assert_eq!(cpu.reg(Reg::Rsp), 0x900);
        let r = run(&vm, &mut cpu, 1);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(cpu.reg(Reg::Rax), 0x300);
    }
    let (mut vm, mut cpu) = setup(&[0xfb, 0xa0, 0, 0x70, 0x40]);
    handler(&mut vm, 0x40, &[0xf4]);
    vm.unmap(0x7000, 4096).unwrap();
    budget(&vm, &mut cpu, 1);
    cpu.inject_irq(0x40);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 0);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::UnmappedMemory { .. })
    ));
    assert!(cpu.has_pending_irq());
    vm.map(0x7000, 4096, Prot::RW, RegionKind::Ram).unwrap();
    budget(&vm, &mut cpu, 1);
    assert!(cpu.has_pending_irq());
    assert!(matches!(
        run(&vm, &mut cpu, 1).stop(),
        StepStop::Exit(Exit::Hlt)
    ));
}

#[test]
fn shadowed_rep_consumes_shadow_on_first_iteration() {
    for load in [&[0xfb][..], &[0x8e, 0xd0][..], &[0x17][..]] {
        let mut code = load.to_vec();
        code.extend_from_slice(&[0xf3, 0xaa, 0x40]);
        let (mut vm, mut cpu) = setup(&code);
        handler(&mut vm, 0x40, &[0xf4]);
        cpu.set_reg(Reg::Rax, 0x300);
        vm.write_bytes(0x3800, &[0, 3]).unwrap();
        flags(&mut cpu, load != [0xfb], false);
        cpu.set_reg(Reg::Rcx, 3);
        cpu.set_reg(Reg::Rdi, 0x8000);
        budget(&vm, &mut cpu, 1);
        cpu.inject_irq(0x40);
        let r = run(&vm, &mut cpu, 10);
        assert_eq!(r.steps(), 2);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(cpu.reg(Reg::Rcx), 2);
        assert_eq!(byte(&vm, 0x3000 + cpu.reg(Reg::Rsp)), load.len() as u8);
    }
}

#[test]
fn shadowed_rep_mmio_subtransfers_preserve_inhibition_until_completion() {
    // SDM Vol 3A §§7.6, 7.8.1, 7.8.3: the first completed REP iteration
    // ends the incoming STI/SS shadow; an unfinished device transfer does not.
    for load in [&[0xfb][..], &[0x8e, 0xd0][..], &[0x17][..]] {
        for op in [0xa4, 0xa5] {
            let mut code = load.to_vec();
            code.extend_from_slice(&[0xf3, op, 0x40]);
            let (mut vm, mut cpu) = setup(&code);
            handler(&mut vm, 0x40, &[0xf4]);
            trap(&mut vm, 0x7000);
            trap(&mut vm, 0x8000);
            cpu.set_reg(Reg::Rax, 0x300);
            vm.write_bytes(0x3800, &[0, 3]).unwrap();
            flags(&mut cpu, load != [0xfb], false);
            cpu.set_reg(Reg::Rsi, 0x7000);
            cpu.set_reg(Reg::Rdi, 0x8000);
            cpu.set_reg(Reg::Rcx, 2);
            budget(&vm, &mut cpu, 1);
            assert!(!cpu.irq_ready());
            cpu.inject_irq(0x40);
            let r = run(&vm, &mut cpu, 10);
            assert_eq!(r.steps(), 0);
            let width = if op == 0xa4 { 1 } else { 2 };
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::MmioRead { addr: 0x7000, size, .. }) if *size == width)
            );
            assert!(cpu.has_pending_irq());
            assert!(!cpu.irq_ready());
            budget(&vm, &mut cpu, 0);
            cpu.complete_mmio_read(0x66);
            assert!(!cpu.irq_ready());
            budget(&vm, &mut cpu, 0);
            let r = run(&vm, &mut cpu, 10);
            assert_eq!(r.steps(), 0);
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::MmioWrite { addr: 0x8000, size, value: 0x66, .. }) if *size == width)
            );
            assert_eq!(cpu.reg(Reg::Rip), load.len() as u64);
            assert_eq!(cpu.reg(Reg::Rcx), 2);
            assert_eq!(cpu.reg(Reg::Rsi), 0x7000);
            assert_eq!(cpu.reg(Reg::Rdi), 0x8000);
            assert!(cpu.has_pending_irq());
            assert!(!cpu.irq_ready());
            cpu.complete_mmio_write();
            budget(&vm, &mut cpu, 0);
            assert!(!cpu.irq_ready());
            budget(&vm, &mut cpu, 1);
            assert_eq!(cpu.reg(Reg::Rip), load.len() as u64);
            assert_eq!(cpu.reg(Reg::Rcx), 1);
            assert_eq!(cpu.reg(Reg::Rsi), 0x7000 + u64::from(width));
            assert_eq!(cpu.reg(Reg::Rdi), 0x8000 + u64::from(width));
            assert_eq!(cpu.retired_instructions(), 1);
            assert!(cpu.has_pending_irq());
            assert!(cpu.irq_ready());
            let r = run(&vm, &mut cpu, 10);
            assert_eq!(r.steps(), 1);
            assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
            assert!(!cpu.has_pending_irq());
            assert_eq!(cpu.reg(Reg::Rcx), 1);
            assert_eq!(byte(&vm, 0x3000 + cpu.reg(Reg::Rsp)), load.len() as u8);
        }
    }
}

#[test]
fn mode_refusal_and_compiling_backend_real16() {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        let mut vm = Vm::new(VmConfig::flat(4096));
        vm.set_cpu_mode(mode);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rax, 0x123);
        cpu.exit_request().request_exit();
        for n in [0, 1] {
            assert_eq!(
                cpu.run_steps(&vm, n, StepOptions::default()).unwrap_err(),
                StepRunError::UnsupportedMode
            );
            assert_eq!(cpu.reg(Reg::Rax), 0x123);
            assert_eq!(cpu.retired_instructions(), 0);
        }
    }
    let mut vm = Vm::with_backend(VmConfig::flat(4096), Box::new(JitBackend::new()));
    vm.set_cpu_mode(CpuMode::Real16);
    vm.map(0, 4096, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0, &[0x40, 0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    budget(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Rax), 1);
    assert!(matches!(
        run(&vm, &mut cpu, 1).stop(),
        StepStop::Exit(Exit::Hlt)
    ));
}

#[test]
fn stopped_marker_does_not_enter_trace() {
    use std::{
        io::Write,
        sync::{Arc, Mutex},
    };
    #[derive(Clone)]
    struct Sink(Arc<Mutex<Vec<u64>>>);
    impl Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap()
                .push(u64::from_le_bytes(bytes[..8].try_into().unwrap()));
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut code = [0xff, 0xc3].repeat(100); // inc bx (also decodes as a scalar in the trace)
    code.push(0xf4);
    let (mut vm, mut cpu) = setup(&code);
    cpu.set_reg(Reg::Cs, 0);
    cpu.set_reg(Reg::Rip, 0x1000);
    let records = Arc::new(Mutex::new(vec![]));
    vm.set_lockstep_trace(Some(Arc::new(x86jit_core::lockstep::Trace::new(
        Sink(Arc::clone(&records)),
        x86jit_core::lockstep::TraceOptions {
            window: Some((0x1000, 0x2000)),
            max_records: None,
        },
    ))));
    budget(&vm, &mut cpu, 7);
    assert_eq!(cpu.reg(Reg::Rip), 0x100e);
    assert_eq!(cpu.reg(Reg::Rbx), 7);
    // Brackets flush at the next entered marker; the seventh stays unflushed.
    assert_eq!(
        *records.lock().unwrap(),
        (0..6).map(|i| 0x1000 + 2 * i).collect::<Vec<_>>()
    );
}

#[test]
fn fetch_and_unsupported_attribution_and_zero_fetch() {
    let (mut vm, mut cpu) = setup(&[0x40]);
    vm.unmap(0x2000, 4096).unwrap();
    cpu.set_reg(Reg::Rip, 0xfff);
    vm.write_bytes(0x1fff, &[0x40]).unwrap();
    budget(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Rip), 0x1000);
    budget(&vm, &mut cpu, 0);
    let r = run(&vm, &mut cpu, 5);
    assert_eq!(r.steps(), 0);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::UnmappedMemory { .. })
    ));
    let (vm, mut cpu) = setup(&[0x90, 0x0f, 0x01, 0xc1]); // VMCALL: unsupported in this interpreter
    let r = run(&vm, &mut cpu, 5);
    assert_eq!(r.steps(), 1);
    assert_eq!(cpu.reg(Reg::Rip), 1);
    assert!(
        matches!(r.stop(), StepStop::Exit(Exit::UnknownInstruction { .. })),
        "{r:?}"
    );
}

#[test]
fn rep_single_call_fault_attribution() {
    for op in [0xa4, 0xaa, 0xac, 0xa6, 0xae] {
        let (mut vm, mut cpu) = setup(&[0xf3, op, 0xf4]);
        vm.write_bytes(0x8ffe, &[0x55; 2]).unwrap();
        vm.write_bytes(0xaffe, &[0x55; 2]).unwrap();
        cpu.set_reg(Reg::Rax, 0x55);
        cpu.set_reg(Reg::Rsi, if op == 0xac { 0x8ffe } else { 0xaffe });
        cpu.set_reg(Reg::Rdi, 0x8ffe);
        cpu.set_reg(Reg::Rcx, 3);
        vm.unmap(0x9000, 4096).unwrap();
        let r = run(&vm, &mut cpu, 10);
        assert_eq!(r.steps(), 2, "op={op:x}");
        assert_eq!(cpu.retired_instructions(), 0);
        assert_eq!(cpu.reg(Reg::Rcx), 1);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::UnmappedMemory { .. })),
            "{r:?}"
        );
    }
}

#[test]
fn all_string_mmio_answers_and_iteration_ledger() {
    for op in [0xaa, 0xac, 0xae] {
        let (mut vm, mut cpu) = setup(&[0xf3, op, 0xf4]);
        trap(&mut vm, 0x7000);
        cpu.set_reg(Reg::Rax, 0x44);
        cpu.set_reg(Reg::Rsi, 0x7000);
        cpu.set_reg(Reg::Rdi, 0x7000);
        cpu.set_reg(Reg::Rcx, 2);
        for iteration in 0..2 {
            let r = run(&vm, &mut cpu, 1);
            assert_eq!(r.steps(), 0);
            match r.into_stop() {
                StepStop::Exit(Exit::MmioRead { addr: 0x7000, .. }) => {
                    cpu.complete_mmio_read(if op == 0xac { 0x11 + iteration } else { 0x44 });
                }
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0x7000,
                    value: 0x44,
                    ..
                }) => cpu.complete_mmio_write(),
                other => panic!("{other:?}"),
            }
            budget(&vm, &mut cpu, 0);
            budget(&vm, &mut cpu, 1);
            assert_eq!(cpu.reg(Reg::Rcx), 1 - iteration);
            if op == 0xac {
                assert_eq!(cpu.reg(Reg::Rax), 0x11 + iteration);
            }
            // Force the next iteration back to the same device address.
            cpu.set_reg(Reg::Rsi, 0x7000);
            cpu.set_reg(Reg::Rdi, 0x7000);
        }
        assert_eq!(cpu.reg(Reg::Rip), 2);
        assert_eq!(cpu.retired_instructions(), 1);
    }
}

#[test]
fn mmio_subtransfers_defer_irq_and_host_until_iteration_completion() {
    let (mut vm, mut cpu) = setup(&[0xf3, 0xa4, 0xf4]);
    handler(&mut vm, 0x40, &[0xf4]);
    trap(&mut vm, 0x7000);
    trap(&mut vm, 0x8000);
    cpu.set_reg(Reg::Rsi, 0x7000);
    cpu.set_reg(Reg::Rdi, 0x8000);
    cpu.set_reg(Reg::Rcx, 2);
    flags(&mut cpu, true, false);
    assert!(matches!(
        run(&vm, &mut cpu, 1).stop(),
        StepStop::Exit(Exit::MmioRead { .. })
    ));
    cpu.inject_irq(0x40);
    cpu.exit_request().request_exit();
    budget(&vm, &mut cpu, 0);
    cpu.complete_mmio_read(0x66);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 0);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::MmioWrite { value: 0x66, .. })
    ));
    assert!(cpu.has_pending_irq());
    cpu.complete_mmio_write();
    budget(&vm, &mut cpu, 1);
    assert!(cpu.has_pending_irq());
    assert_eq!(cpu.reg(Reg::Rcx), 1);
    let r = run(&vm, &mut cpu, 5);
    assert_eq!(r.steps(), 0);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Interrupted)));
    assert!(cpu.has_pending_irq());
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 1);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
    assert!(!cpu.has_pending_irq());
    assert_eq!(byte(&vm, 0x3000 + cpu.reg(Reg::Rsp)), 0); // REP restart IP
}

#[test]
fn ram_rep_host_request_after_observed_write_stops_at_iteration_boundary() {
    // The request races the REP: on a fast or idle host the whole string can
    // finish before the watcher observes a write. Retry; one mid-REP stop is
    // the witness.
    for _ in 0..20 {
        if host_request_mid_rep_attempt() {
            return;
        }
    }
    panic!("REP finished before the host request in 20 attempts");
}

fn host_request_mid_rep_attempt() -> bool {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Barrier,
    };
    use std::time::{Duration, Instant};
    // Observe the public dirty-page stream, avoiding concurrent reads of guest RAM.
    let count = 0xffff;
    let (vm, mut cpu) = setup(&[0xf3, 0xaa, 0xf4]);
    cpu.set_reg(Reg::Es, 0x1000);
    cpu.set_reg(Reg::Rdi, 0);
    cpu.set_reg(Reg::Rcx, count);
    cpu.set_reg(Reg::Rax, 0x6a);
    vm.mem.watch_range(0x10000, count);
    assert!(vm.mem.take_dirty_ranges().is_empty());
    let request = cpu.exit_request();
    let ready = Barrier::new(2);
    let finished = AtomicBool::new(false);
    let r = std::thread::scope(|scope| {
        let watcher = scope.spawn(|| {
            ready.wait();
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                if !vm.mem.take_dirty_ranges().is_empty() {
                    request.request_exit();
                    return;
                }
                if finished.load(Ordering::Acquire) {
                    return;
                }
                assert!(Instant::now() < deadline, "REP produced no watched write");
                std::thread::yield_now();
            }
        });
        ready.wait();
        let r = run(&vm, &mut cpu, count);
        finished.store(true, Ordering::Release);
        watcher.join().unwrap();
        r
    });
    if !matches!(r.stop(), StepStop::Exit(Exit::Interrupted)) || r.steps() >= count {
        return false;
    }
    assert!(r.steps() > 0, "{r:?}");
    assert_eq!(cpu.reg(Reg::Rip), 0);
    assert_eq!(cpu.reg(Reg::Rcx), count - r.steps());
    assert_eq!(cpu.reg(Reg::Rdi), r.steps());
    assert_eq!(cpu.retired_instructions(), 0);
    assert!(cpu.interruption_evidence().is_some());
    assert_eq!(byte(&vm, 0x10000), 0x6a);
    assert_eq!(byte(&vm, 0x10000 + cpu.reg(Reg::Rdi) - 1), 0x6a);
    assert_eq!(byte(&vm, 0x10000 + cpu.reg(Reg::Rdi)), 0);
    budget(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Rcx), count - r.steps() - 1);
    assert_eq!(byte(&vm, 0x10000 + r.steps()), 0x6a);
    true
}

#[test]
fn shadow_overlap_consecutive_ss_and_sti_entry_if() {
    // The IRQ is queued only after the first unit, exposing the protected boundary.
    for (code, entry_if, ax_delta, remaining_steps) in [
        (&[0x8e, 0xd0, 0x8e, 0xd0, 0x40][..], true, 0, 2),
        (&[0xfb, 0x8e, 0xd0, 0x40][..], false, 1, 3),
        (&[0x8e, 0xd0, 0xfb, 0x40][..], false, 1, 3),
        (&[0x8e, 0xd0, 0xfb, 0x40][..], true, 0, 2),
        (&[0xfb, 0x90, 0x40][..], false, 0, 2),
        (&[0xfb, 0x90, 0x40][..], true, 0, 1),
    ] {
        let (mut vm, mut cpu) = setup(code);
        handler(&mut vm, 0x40, &[0xf4]);
        cpu.set_reg(Reg::Rax, 0x300);
        flags(&mut cpu, entry_if, false);
        budget(&vm, &mut cpu, 1);
        cpu.inject_irq(0x40);
        let r = run(&vm, &mut cpu, 10);
        assert_eq!(r.steps(), remaining_steps, "{code:x?}, IF={entry_if}");
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(cpu.reg(Reg::Rax), 0x300 + ax_delta);
    }
    let (vm, mut cpu) = setup(&[0xfb, 0xfa, 0xf4]);
    budget(&vm, &mut cpu, 1);
    cpu.inject_irq(0x40);
    budget(&vm, &mut cpu, 1);
    assert!(!cpu.flags().if_);
    assert!(matches!(
        run(&vm, &mut cpu, 1).stop(),
        StepStop::Exit(Exit::Hlt)
    ));
    assert!(cpu.has_pending_irq());
}

#[test]
fn faulted_ss_load_has_no_shadow_and_pending_in_completion_consumes_shadow() {
    let (mut vm, mut cpu) = setup(&[0x8e, 0x16, 0, 0x70]);
    handler(&mut vm, 0x40, &[0xf4]);
    vm.unmap(0x7000, 4096).unwrap();
    flags(&mut cpu, true, false);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 0);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::UnmappedMemory { .. })
    ));
    cpu.inject_irq(0x40);
    assert!(matches!(
        run(&vm, &mut cpu, 1).stop(),
        StepStop::Exit(Exit::Hlt)
    ));
    let (mut vm, mut cpu) = setup(&[0xfb, 0xe4, 0x80, 0x40]);
    handler(&mut vm, 0x40, &[0xf4]);
    budget(&vm, &mut cpu, 1);
    cpu.inject_irq(0x40);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 1);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::PortIo {
            dir: PortDir::In,
            ..
        })
    ));
    budget(&vm, &mut cpu, 0);
    assert_eq!(
        cpu.run_steps(&vm, 1, StepOptions::default()).unwrap_err(),
        StepRunError::PendingPortIn
    );
    assert!(cpu.has_pending_irq());
    cpu.complete_port_in(0x55);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 1);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
    assert_eq!(cpu.reg(Reg::Rax), 0x55);
}

#[test]
fn zero_preserves_requests_vectors_and_exit_evidence() {
    let (mut vm, mut cpu) = setup(&[0x40, 0xf4]);
    handler(&mut vm, 0x40, &[0xf4]);
    flags(&mut cpu, true, false);
    cpu.inject_irq(0x40);
    cpu.exit_request().request_exit();
    budget(&vm, &mut cpu, 0);
    assert!(cpu.has_pending_irq());
    assert_eq!(cpu.reg(Reg::Rip), 0);
    assert_eq!(cpu.retired_instructions(), 0);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 0);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Interrupted)));
    let evidence = cpu.interruption_evidence().unwrap();
    budget(&vm, &mut cpu, 0);
    assert_eq!(cpu.interruption_evidence(), Some(evidence));
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 1);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
    assert_eq!(cpu.reg(Reg::Rax), 0);
}

#[test]
fn rep_device_stream_partitions_and_call_local_retirement_relation() {
    fn execute(slices: &[u64]) -> (u64, Vec<u64>, Vec<(u64, bool)>) {
        let (mut vm, mut cpu) = setup(&[0xf3, 0xa4, 0xf4]);
        trap(&mut vm, 0x7000);
        trap(&mut vm, 0x8000);
        cpu.set_reg(Reg::Rsi, 0x7000);
        cpu.set_reg(Reg::Rdi, 0x8000);
        cpu.set_reg(Reg::Rcx, 11);
        let mut total = 0;
        let mut events = vec![];
        for n in slices.iter().copied().cycle().take(1000) {
            let before_cx = cpu.reg(Reg::Rcx);
            let before_ret = cpu.retired_instructions();
            let before_ip = cpu.reg(Reg::Rip);
            let r = run(&vm, &mut cpu, n);
            total += r.steps();
            let iterations = before_cx - cpu.reg(Reg::Rcx);
            let completed_rep = u64::from(before_ip == 0 && cpu.reg(Reg::Rip) != 0);
            assert_eq!(
                r.steps(),
                cpu.retired_instructions() - before_ret + iterations - completed_rep
            );
            let completed = total.min(11);
            assert_eq!(cpu.reg(Reg::Rcx), 11 - completed);
            assert_eq!(cpu.reg(Reg::Rsi), 0x7000 + completed);
            assert_eq!(cpu.reg(Reg::Rdi), 0x8000 + completed);
            assert_eq!(
                cpu.reg(Reg::Rip),
                if total < 11 {
                    0
                } else if total == 11 {
                    2
                } else {
                    3
                }
            );
            let expected_events: Vec<_> = (0..11)
                .flat_map(|i| [(0x7000 + i, false), (0x8000 + i, true)])
                .collect();
            assert_eq!(events, expected_events[..events.len()]);
            match r.stop() {
                StepStop::Exit(Exit::BudgetExhausted) => {
                    assert_eq!(r.steps(), n);
                    if n > 0 {
                        assert_eq!(events.len() as u64, 2 * completed);
                    } else {
                        assert_eq!(cpu.reg(Reg::Rip), before_ip);
                        assert_eq!(cpu.reg(Reg::Rcx), before_cx);
                    }
                }
                StepStop::Exit(Exit::MmioRead { addr, size, .. }) => {
                    assert!(r.steps() < n, "next iteration started at a budget cut");
                    assert_eq!(events.len() as u64, 2 * completed);
                    assert_eq!((*addr, *size), (0x7000 + completed, 1));
                }
                StepStop::Exit(Exit::MmioWrite { addr, size, .. }) => {
                    assert!(r.steps() < n, "next iteration started at a budget cut");
                    assert_eq!(events.len() as u64, 2 * completed + 1);
                    assert_eq!((*addr, *size), (0x8000 + completed, 1));
                }
                StepStop::Exit(Exit::Hlt) => assert_eq!(events.len(), 22),
                other => panic!("{other:?}"),
            }
            match r.into_stop() {
                StepStop::Exit(Exit::BudgetExhausted) => {}
                StepStop::Exit(Exit::MmioRead { addr, .. }) => {
                    events.push((addr, false));
                    cpu.complete_mmio_read(addr & 0xff);
                }
                StepStop::Exit(Exit::MmioWrite { addr, value, .. }) => {
                    assert_eq!(value, addr & 0xff);
                    events.push((addr, true));
                    cpu.complete_mmio_write();
                }
                StepStop::Exit(Exit::Hlt) => {
                    assert_eq!(total, 12);
                    assert_eq!(cpu.retired_instructions(), 2);
                    assert_eq!(events.len(), 22);
                    return (
                        total,
                        [Reg::Rsi, Reg::Rdi, Reg::Rcx, Reg::Rip]
                            .map(|r| cpu.reg(r))
                            .to_vec(),
                        events,
                    );
                }
                other => panic!("{other:?}"),
            }
        }
        panic!("no halt")
    }
    let expected = execute(&[100]);
    for slices in [&[1][..], &[7], &[0, 1, 0, 7, 100, 0, 3]] {
        assert_eq!(execute(slices), expected);
    }
}

#[test]
fn smc_cut_resolves_modified_successor() {
    let (vm, mut cpu) = setup(&[0xa2, 0x03, 0x10, 0x40, 0xf4]); // mov [0x1003],al; inc ax; hlt
    cpu.set_reg(Reg::Rax, 0x90);
    budget(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Rip), 3);
    budget(&vm, &mut cpu, 1);
    assert_eq!(cpu.reg(Reg::Rax), 0x90);
    assert_eq!(cpu.reg(Reg::Rip), 4);
}

#[test]
fn stepped_walk_specializations_keep_identical_budget_behavior() {
    use x86jit_core::MemConsistency;
    for consistency in [
        MemConsistency::Fast,
        MemConsistency::AcqRel,
        MemConsistency::FullTso,
    ] {
        for routed in [false, true] {
            for mxcsr in [0x1f80, 0x3f80] {
                let mut config = VmConfig::flat(0x4000);
                config.consistency = consistency;
                let mut vm = Vm::new(config);
                vm.set_cpu_mode(CpuMode::Real16);
                vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.write_bytes(0x1000, &[0x40, 0xf3, 0xaa, 0xf4]).unwrap();
                vm.mem.set_host_fault_recovery(routed);
                let mut cpu = vm.new_vcpu_with_mxcsr(mxcsr);
                cpu.set_reg(Reg::Cs, 0x100);
                cpu.set_reg(Reg::Rdi, 0x2000);
                cpu.set_reg(Reg::Rcx, 3);
                budget(&vm, &mut cpu, 2);
                assert_eq!(cpu.reg(Reg::Rip), 1);
                assert_eq!(cpu.reg(Reg::Rcx), 2);
                assert_eq!(cpu.retired_instructions(), 1);
                budget(&vm, &mut cpu, 2);
                assert_eq!(cpu.reg(Reg::Rip), 3);
                assert_eq!(cpu.retired_instructions(), 2);
                assert_eq!(byte(&vm, 0x2002), 1);
                assert_eq!(cpu.mxcsr(), mxcsr);
            }
        }
    }
}

#[test]
fn failed_irq_delivery_and_faulted_pop_ss_add_no_unit() {
    let (mut vm, mut cpu) = setup(&[0x40]);
    handler(&mut vm, 0x40, &[0xf4]);
    vm.unmap(0x3000, 4096).unwrap();
    flags(&mut cpu, true, false);
    cpu.inject_irq(0x40);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 0);
    assert_eq!(cpu.retired_instructions(), 0);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::UnmappedMemory { .. })
    ));
    assert_eq!(cpu.reg(Reg::Rax), 0);
    let (mut vm, mut cpu) = setup(&[0x17]);
    handler(&mut vm, 0x40, &[0xf4]);
    vm.unmap(0x3000, 4096).unwrap();
    flags(&mut cpu, true, false);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 0);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::UnmappedMemory { .. })
    ));
    vm.map(0x3000, 4096, Prot::RW, RegionKind::Ram).unwrap();
    cpu.inject_irq(0x40);
    let r = run(&vm, &mut cpu, 1);
    assert_eq!(r.steps(), 1);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
}
