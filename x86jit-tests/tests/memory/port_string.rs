//! Hardware witnesses: SDM Vol 2A §3.3 INS/INSB/INSW/INSD, pp. 3-464–466;
//! Vol 2B §4.3 OUTS/OUTSB/OUTSW/OUTSD, pp. 4-173–176; REP, pp. 4-563–564.
//! Restart/IRQ boundaries: Vol 3A §§7.6, 7.8.1, 7.8.3. Handoff accounting
//! and buffered input completion follow decision-16 §2 (engine contracts).
use x86jit_core::{
    AccessKind, Backend, CpuMode, Exit, FaultAccess, InterpreterBackend, Knowledge, PortDir, Prot,
    Reg, RegionKind, StepOptions, StepRun, StepRunError, StepStop, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

fn setup(code: &[u8], mode: CpuMode, jit: bool) -> (Vm, Vcpu) {
    let backend: Box<dyn Backend> = if jit {
        Box::new(JitBackend::new())
    } else {
        Box::new(InterpreterBackend)
    };
    let mut vm = Vm::with_backend(VmConfig::flat(0x40000), backend);
    vm.set_cpu_mode(mode);
    for page in (0..0x40000).step_by(4096) {
        vm.map(page, 4096, Prot::RWX, RegionKind::Ram).unwrap();
    }
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    if mode == CpuMode::Real16 {
        cpu.set_reg(Reg::Cs, 0x100);
        cpu.set_reg(Reg::Ds, 0x1000);
        cpu.set_reg(Reg::Es, 0x2000);
        cpu.set_reg(Reg::Ss, 0x3000);
    } else {
        cpu.set_reg(Reg::Rip, 0x1000);
    }
    cpu.set_reg(Reg::Rsp, 0x800);
    cpu.set_reg(Reg::Rdx, 0xffff_1234);
    cpu.set_reg(Reg::Rax, 0xabcd_8765);
    (vm, cpu)
}
fn steps(vm: &Vm, cpu: &mut Vcpu, n: u64) -> StepRun {
    let r = cpu.run_steps(vm, n, StepOptions::default()).unwrap();
    assert!(r.steps() <= n);
    r
}
fn port(exit: &Exit, out: bool, elem: u8, value: u64) {
    assert!(
        matches!(exit, Exit::PortIo { port: 0x1234, size, dir, value: v, .. }
        if *size == elem && *dir == if out { PortDir::Out } else { PortDir::In } && *v == value),
        "{exit:?}"
    );
}
fn port_step(r: &StepRun, out: bool, elem: u8, value: u64, n: u64) {
    assert_eq!(r.steps(), n, "{r:?}");
    let StepStop::Exit(exit) = r.stop() else {
        panic!("{r:?}")
    };
    port(exit, out, elem, value);
}
fn bytes(vm: &Vm, addr: u64, len: usize) -> Vec<u8> {
    let mut b = vec![0; len];
    vm.read_bytes(addr, &mut b).unwrap();
    b
}

#[test]
fn real16_elements_df_rep_counts_and_completed_prefix() {
    for jit in [false, true] {
        for out in [false, true] {
            for elem in [1, 2] {
                for df in [false, true] {
                    for rep in [false, true] {
                        for count in [0, 1, 3] {
                            let mut code = vec![];
                            if rep {
                                code.push(0xf3);
                            }
                            code.extend_from_slice(&[
                                if out { 0x6e } else { 0x6c } + elem - 1,
                                0xf4,
                            ]);
                            let (vm, mut cpu) = setup(&code, CpuMode::Real16, jit);
                            cpu.set_reg(Reg::Rsi, 0xcafe_0040);
                            cpu.set_reg(Reg::Rdi, 0xbeef_0040);
                            cpu.set_reg(Reg::Rcx, 0x1234_0000 | count);
                            let mut f = cpu.flags();
                            f.df = df;
                            cpu.set_flags(f);
                            let initial_flags = cpu.cpu.rflags();
                            vm.write_bytes(0x10030, &[0x5a; 32]).unwrap();
                            let iterations = if rep { count } else { 1 };
                            for i in 0..iterations {
                                let r = steps(&vm, &mut cpu, 7);
                                port_step(
                                    &r,
                                    out,
                                    elem,
                                    if out {
                                        if elem == 1 {
                                            0x5a
                                        } else {
                                            0x5a5a
                                        }
                                    } else {
                                        0
                                    },
                                    1,
                                );
                                let delta = (i + u64::from(out)) * u64::from(elem);
                                let offset = if df { 0x40 - delta } else { 0x40 + delta };
                                assert_eq!(
                                    cpu.reg(if out { Reg::Rsi } else { Reg::Rdi }) & 0xffff,
                                    offset
                                );
                                if rep {
                                    assert_eq!(
                                        cpu.reg(Reg::Rcx) & 0xffff,
                                        count - i - u64::from(out)
                                    );
                                }
                                assert_eq!(
                                    cpu.reg(Reg::Rip),
                                    if out && i + 1 == iterations {
                                        (code.len() - 1) as u64
                                    } else {
                                        0
                                    }
                                );
                                if !out {
                                    assert!(!cpu.irq_ready());
                                    assert!(matches!(
                                        cpu.run_steps(&vm, 1, StepOptions::default()),
                                        Err(StepRunError::PendingPortIn)
                                    ));
                                    cpu.complete_port_in(0xffff_a55a);
                                }
                            }
                            let r = steps(&vm, &mut cpu, 7);
                            assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{r:?}");
                            assert_eq!(r.steps(), if rep && count == 0 { 2 } else { 1 });
                            let delta = iterations * u64::from(elem);
                            let offset = if df { 0x40 - delta } else { 0x40 + delta };
                            assert_eq!(
                                cpu.reg(if out { Reg::Rsi } else { Reg::Rdi }),
                                if out { 0xcafe_0000 } else { 0xbeef_0000 } | offset
                            );
                            assert_eq!(
                                cpu.reg(if out { Reg::Rdi } else { Reg::Rsi }),
                                if out { 0xbeef_0040 } else { 0xcafe_0040 }
                            );
                            assert_eq!(
                                cpu.reg(Reg::Rcx),
                                0x1234_0000 | if rep { 0 } else { count }
                            );
                            assert_eq!(cpu.reg(Reg::Rax), 0xabcd_8765);
                            assert_eq!(cpu.cpu.rflags(), initial_flags);
                            assert_eq!(cpu.retired_instructions(), 2);
                            if !out {
                                for i in 0..iterations {
                                    let offset = if df {
                                        0x40 - i * u64::from(elem)
                                    } else {
                                        0x40 + i * u64::from(elem)
                                    };
                                    assert_eq!(
                                        bytes(&vm, 0x20000 + offset, elem as usize),
                                        [0x5a, 0xa5][..elem as usize]
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn segment_override_outs_and_fixed_es_ins() {
    for out in [false, true] {
        // CS override differs from DS and ES.
        let (vm, mut cpu) = setup(
            &[0x2e, if out { 0x6f } else { 0x6d }, 0xf4],
            CpuMode::Real16,
            false,
        );
        cpu.set_reg(Reg::Rsi, 0x80);
        cpu.set_reg(Reg::Rdi, 0x80);
        vm.write_bytes(0x1080, &[0x34, 0x12]).unwrap();
        vm.write_bytes(0x10080, &[0xef, 0xbe]).unwrap();
        port_step(
            &steps(&vm, &mut cpu, 1),
            out,
            2,
            if out { 0x1234 } else { 0 },
            1,
        );
        if !out {
            cpu.complete_port_in(0x5678);
        }
        assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
        assert_eq!(bytes(&vm, 0x1080, 2), [0x34, 0x12]);
        if !out {
            assert_eq!(bytes(&vm, 0x20080, 2), [0x78, 0x56]);
        }
    }
}

#[test]
fn pending_input_zero_budget_and_answer_are_stable() {
    let (vm, mut cpu) = setup(&[0xf3, 0x6d, 0xf4], CpuMode::Real16, false);
    cpu.set_reg(Reg::Rcx, 2);
    cpu.set_reg(Reg::Rdi, 0x80);
    port_step(&steps(&vm, &mut cpu, 1), false, 2, 0, 1);
    for answered in [false, true] {
        if answered {
            cpu.complete_port_in(0x1234);
            cpu.complete_port_in(0xffff);
        }
        let before = (
            cpu.reg(Reg::Rip),
            cpu.reg(Reg::Rdi),
            cpu.reg(Reg::Rcx),
            cpu.retired_instructions(),
        );
        let r = steps(&vm, &mut cpu, 0);
        assert_eq!(r.steps(), 0);
        assert_eq!(
            before,
            (
                cpu.reg(Reg::Rip),
                cpu.reg(Reg::Rdi),
                cpu.reg(Reg::Rcx),
                cpu.retired_instructions()
            )
        );
        assert_eq!(bytes(&vm, 0x20080, 2), [0, 0]);
    }
    port_step(&steps(&vm, &mut cpu, 1), false, 2, 0, 1);
    assert_eq!(bytes(&vm, 0x20080, 2), [0x34, 0x12]);
    assert_eq!(cpu.reg(Reg::Rdi), 0x82);
    assert_eq!(cpu.reg(Reg::Rcx), 1);
    cpu.complete_port_in(0x5678);
    assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
    assert_eq!(bytes(&vm, 0x20080, 4), [0x34, 0x12, 0x78, 0x56]);
}

#[test]
fn steps_cut_before_port_and_attribute_final_budget_unit() {
    for out in [false, true] {
        let (vm, mut cpu) = setup(
            &[0x40, 0xf3, if out { 0x6e } else { 0x6c }, 0x43, 0xf4],
            CpuMode::Real16,
            false,
        );
        cpu.set_reg(Reg::Rcx, 2);
        cpu.set_reg(Reg::Rsi, 0x80);
        cpu.set_reg(Reg::Rdi, 0x80);
        let r = steps(&vm, &mut cpu, 1);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
        assert_eq!(cpu.reg(Reg::Rip), 1);
        for _ in 0..2 {
            port_step(&steps(&vm, &mut cpu, 1), out, 1, 0, 1);
            assert_eq!(cpu.reg(Reg::Rbx), 0);
            if !out {
                cpu.complete_port_in(0x42);
            }
        }
        let r = steps(&vm, &mut cpu, 1);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)),
            "{r:?}"
        );
        assert_eq!(r.steps(), 1);
        assert_eq!(cpu.reg(Reg::Rbx), 1);
        assert_eq!(cpu.reg(Reg::Rcx), 0);
        assert_eq!(cpu.retired_instructions(), 3);
    }
}

#[test]
fn real16_index_and_instruction_pointer_wrap() {
    for out in [false, true] {
        for elem in [1, 2] {
            for df in [false, true] {
                let (vm, mut cpu) = setup(&[0xf4], CpuMode::Real16, false);
                vm.write_bytes(0x10ffe, &[0xf3, if out { 0x6e } else { 0x6c } + elem - 1])
                    .unwrap();
                cpu.set_reg(Reg::Rip, 0xfffe);
                let mut f = cpu.flags();
                f.df = df;
                cpu.set_flags(f);
                let start = if df { 0 } else { 0x10000 - u64::from(elem) };
                cpu.set_reg(Reg::Rcx, 2);
                cpu.set_reg(Reg::Rsi, start);
                cpu.set_reg(Reg::Rdi, start);
                for _ in 0..2 {
                    port_step(&steps(&vm, &mut cpu, 1), out, elem, 0, 1);
                    if !out {
                        cpu.complete_port_in(0);
                    }
                }
                assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
                assert_eq!(cpu.reg(Reg::Rip), 1);
                assert_eq!(
                    cpu.reg(if out { Reg::Rsi } else { Reg::Rdi }),
                    if df {
                        0x10000 - 2 * u64::from(elem)
                    } else {
                        u64::from(elem)
                    }
                );
            }
        }
    }
}

#[test]
fn irq_between_elements_preserves_restart_ip_and_input_store() {
    for out in [false, true] {
        for shadow in [&[][..], &[0xfb][..], &[0x8e, 0xd0][..]] {
            // STI / MOV SS,AX
            let mut code = shadow.to_vec();
            code.extend_from_slice(&[0xf3, if out { 0x6f } else { 0x6d }, 0xf4]);
            let (vm, mut cpu) = setup(&code, CpuMode::Real16, false);
            vm.write_bytes(0x80, &[0, 0, 0, 6]).unwrap();
            vm.write_bytes(0x6000, &[0xcf]).unwrap(); // IRET
            cpu.set_reg(Reg::Rcx, 2);
            cpu.set_reg(Reg::Rdi, 0x80);
            cpu.set_reg(Reg::Rsi, 0x80);
            cpu.set_reg(Reg::Rax, 0x3000);
            let mut f = cpu.flags();
            f.if_ = shadow != [0xfb];
            cpu.set_flags(f);
            if !shadow.is_empty() {
                steps(&vm, &mut cpu, 1);
            }
            if !shadow.is_empty() {
                assert!(!cpu.irq_ready());
            }
            port_step(&steps(&vm, &mut cpu, 1), out, 2, 0, 1);
            cpu.inject_irq(0x20);
            if !out {
                assert!(!cpu.irq_ready());
                cpu.complete_port_in(0xabcd);
                assert!(!cpu.irq_ready());
            }
            // Complete buffered INS, then deliver IRQ and execute exactly IRET.
            let r = steps(&vm, &mut cpu, 1);
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                "{r:?}"
            );
            assert_eq!(cpu.reg(Reg::Cs), 0x100);
            assert_eq!(cpu.reg(Reg::Rip), shadow.len() as u64);
            assert_eq!(cpu.reg(Reg::Rcx), 1);
            assert_eq!(bytes(&vm, 0x307fa, 2), (shadow.len() as u16).to_le_bytes());
            if !out {
                assert_eq!(bytes(&vm, 0x20080, 2), [0xcd, 0xab]);
            }
            port_step(&steps(&vm, &mut cpu, 1), out, 2, 0, 1);
            if !out {
                cpu.complete_port_in(0x5678);
            }
            assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
        }
    }
}

#[test]
fn memory_faults_keep_prefix_and_input_answer() {
    for out in [false, true] {
        let (mut vm, mut cpu) = setup(
            &[0xf3, if out { 0x6f } else { 0x6d }, 0xf4],
            CpuMode::Real16,
            false,
        );
        cpu.set_reg(Reg::Rcx, 3);
        cpu.set_reg(Reg::Rsi, 0xffe);
        cpu.set_reg(Reg::Rdi, 0xffe);
        let base = if out { 0x10000 } else { 0x20000 };
        vm.unmap(base + 0x1000, 4096).unwrap();
        port_step(&steps(&vm, &mut cpu, 1), out, 2, 0, 1);
        if !out {
            cpu.complete_port_in(0x1234);
            port_step(&steps(&vm, &mut cpu, 1), false, 2, 0, 1);
            cpu.complete_port_in(0x5678);
        }
        let r = steps(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 0);
        let StepStop::Exit(exit) = r.stop() else {
            panic!("{r:?}")
        };
        assert!(
            matches!(exit, Exit::UnmappedMemory { addr, access, .. } if *addr == base + 0x1000 && *access == if out { AccessKind::Read } else { AccessKind::Write }),
            "{exit:?}"
        );
        let m = exit.fault_metadata().unwrap();
        assert_eq!(m.instruction_rip(), Knowledge::Known(0));
        assert_eq!(m.instruction_len(), Knowledge::Known(2));
        assert_eq!(m.operand_size(), Knowledge::Known(2));
        assert_eq!(m.transfer_size(), Knowledge::Known(2));
        assert_eq!(
            m.access(),
            Knowledge::Known(if out {
                FaultAccess::Read
            } else {
                FaultAccess::Write
            })
        );
        let restart = m.progress().rep().unwrap();
        assert_eq!(restart.remaining(), 2);
        assert_eq!(restart.element_size(), 2);
        assert_eq!(restart.address_bits(), 16);
        assert_eq!(cpu.reg(Reg::Rip), 0);
        assert_eq!(cpu.reg(Reg::Rcx), 2);
        vm.map(base + 0x1000, 4096, Prot::RW, RegionKind::Ram)
            .unwrap();
        // INS finishes the same retained answer, then requests only the third element.
        port_step(&steps(&vm, &mut cpu, 1), out, 2, 0, 1);
        if !out {
            assert_eq!(bytes(&vm, base + 0x1000, 2), [0x78, 0x56]);
            cpu.complete_port_in(0xabcd);
        }
    }
}

#[test]
fn memory_mmio_answers_are_consumed_once_per_element() {
    for out in [false, true] {
        let (mut vm, mut cpu) = setup(
            &[0xf3, if out { 0x6e } else { 0x6c }, 0xf4],
            CpuMode::Real16,
            false,
        );
        cpu.set_reg(Reg::Rcx, 2);
        cpu.set_reg(Reg::Rsi, 0x80);
        cpu.set_reg(Reg::Rdi, 0x80);
        let base = if out { 0x10000 } else { 0x20000 };
        vm.unmap(base, 4096).unwrap();
        vm.map(base, 4096, Prot::RW, RegionKind::Trap).unwrap();
        for value in [0x12, 0x34] {
            if !out {
                port_step(&steps(&vm, &mut cpu, 1), false, 1, 0, 1);
                cpu.complete_port_in(value);
            }
            let r = steps(&vm, &mut cpu, 1);
            assert_eq!(r.steps(), 0);
            assert!(
                matches!(
                    r.stop(),
                    StepStop::Exit(Exit::MmioRead { .. } | Exit::MmioWrite { .. })
                ),
                "{r:?}"
            );
            if out {
                cpu.complete_mmio_read(value);
                port_step(&steps(&vm, &mut cpu, 1), true, 1, value, 1);
            } else {
                cpu.complete_mmio_write();
                assert!(matches!(
                    cpu.step_instruction(&vm),
                    x86jit_core::StepResult::Continue
                ));
            }
        }
        assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
    }
}

#[test]
fn protected_modes_address_and_operand_sizes_on_both_backends() {
    for mode in [CpuMode::Compat32, CpuMode::Long64] {
        for jit in [false, true] {
            for addr_bits in if mode == CpuMode::Compat32 {
                [16, 32]
            } else {
                [32, 64]
            } {
                for elem in [1, 2, 4] {
                    for out in [false, true] {
                        let mut code = vec![0xf3];
                        if addr_bits != if mode == CpuMode::Compat32 { 32 } else { 64 } {
                            code.push(0x67);
                        }
                        if elem == 2 {
                            code.push(0x66);
                        }
                        code.extend_from_slice(&[
                            if out { 0x6e } else { 0x6c } + u8::from(elem != 1),
                            0xf4,
                        ]);
                        let (vm, mut cpu) = setup(&code, mode, jit);
                        let upper = if addr_bits < 64 { 0xcafe_0000_0000 } else { 0 };
                        cpu.set_reg(Reg::Rsi, upper | 0x8000);
                        cpu.set_reg(Reg::Rdi, upper | 0x9000);
                        cpu.set_reg(Reg::Rcx, upper | 2);
                        vm.write_bytes(0x8000, &[0x5a; 16]).unwrap();
                        for _ in 0..2 {
                            let value = if out {
                                match elem {
                                    1 => 0x5a,
                                    2 => 0x5a5a,
                                    _ => 0x5a5a5a5a,
                                }
                            } else {
                                0
                            };
                            port(&cpu.run(&vm, Some(32)), out, elem, value);
                            if !out {
                                cpu.complete_port_in(0x1234_5678);
                            }
                        }
                        assert!(matches!(cpu.run(&vm, Some(32)), Exit::Hlt));
                        let final_upper = if addr_bits == 16 { upper } else { 0 };
                        assert_eq!(cpu.reg(Reg::Rcx), final_upper);
                        assert_eq!(
                            cpu.reg(if out { Reg::Rsi } else { Reg::Rdi }),
                            final_upper | (if out { 0x8000 } else { 0x9000 } + 2 * u64::from(elem))
                        );
                        assert_eq!(cpu.reg(Reg::Rax), 0xabcd_8765);
                        if !out {
                            assert_eq!(
                                bytes(&vm, 0x9000, elem as usize),
                                0x1234_5678u32.to_le_bytes()[..elem as usize]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn protected_source_fs_override_and_rex_w_keeps_dword() {
    for mode in [CpuMode::Compat32, CpuMode::Long64] {
        for jit in [false, true] {
            for out in [false, true] {
                let mut code = vec![0x64]; // FS source override; ignored by INS.
                if mode == CpuMode::Long64 {
                    code.push(0x48);
                } // REX.W
                code.extend_from_slice(&[if out { 0x6f } else { 0x6d }, 0xf4]);
                let (vm, mut cpu) = setup(&code, mode, jit);
                cpu.set_reg(Reg::FsBase, 0x10000);
                cpu.set_reg(Reg::Rsi, 0x8000);
                cpu.set_reg(Reg::Rdi, 0x8000);
                vm.write_bytes(0x18000, &[0x12, 0x34, 0x56, 0x78]).unwrap();
                port(
                    &cpu.run(&vm, Some(16)),
                    out,
                    4,
                    if out { 0x78563412 } else { 0 },
                );
                if !out {
                    cpu.complete_port_in(0xabcdef01);
                }
                assert!(matches!(cpu.run(&vm, Some(16)), Exit::Hlt));
                assert_eq!(bytes(&vm, 0x18000, 4), [0x12, 0x34, 0x56, 0x78]);
                if !out {
                    assert_eq!(bytes(&vm, 0x8000, 4), [1, 0xef, 0xcd, 0xab]);
                }
            }
        }
    }
}

#[test]
fn real16_segment_limit_fault_precedes_port_handoff() {
    for prefix in [None, Some(0x36)] {
        for out in [false, true] {
            let mut code = prefix.into_iter().collect::<Vec<_>>();
            code.extend_from_slice(&[0xf3, if out { 0x6f } else { 0x6d }, 0xf4]);
            let (vm, mut cpu) = setup(&code, CpuMode::Real16, false);
            let vector = if prefix == Some(0x36) && out { 12 } else { 13 };
            vm.write_bytes(vector * 4, &[0, 0, 0, 6]).unwrap();
            vm.write_bytes(0x6000, &[0xf4]).unwrap();
            cpu.set_reg(Reg::Rsi, 0xffff);
            cpu.set_reg(Reg::Rdi, 0xffff);
            cpu.set_reg(Reg::Rcx, 2);
            let r = steps(&vm, &mut cpu, 1);
            assert_eq!(r.steps(), 1); // handler HLT; the faulting element costs zero.
            assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{r:?}");
            assert_eq!(cpu.reg(Reg::Rcx), 2);
            assert_eq!(cpu.reg(Reg::Rsi), 0xffff);
            assert_eq!(cpu.reg(Reg::Rdi), 0xffff);
            assert_eq!(bytes(&vm, 0x307fa, 2), [0, 0]);
        }
    }
}

#[test]
fn block_run_resumes_real16_one_element_at_a_time() {
    for jit in [false, true] {
        for out in [false, true] {
            let (vm, mut cpu) = setup(
                &[0xf3, if out { 0x6f } else { 0x6d }, 0xf4],
                CpuMode::Real16,
                jit,
            );
            cpu.set_reg(Reg::Rcx, 3);
            cpu.set_reg(Reg::Rsi, 0x80);
            cpu.set_reg(Reg::Rdi, 0x80);
            for i in 0..3 {
                port(&cpu.run(&vm, Some(16)), out, 2, 0);
                assert_eq!(cpu.reg(Reg::Rcx), 3 - i - u64::from(out));
                if !out {
                    cpu.complete_port_in(i + 1);
                }
            }
            assert!(matches!(cpu.run(&vm, Some(16)), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rcx), 0);
            assert_eq!(cpu.retired_instructions(), 2);
            if !out {
                assert_eq!(bytes(&vm, 0x20080, 6), [1, 0, 2, 0, 3, 0]);
            }
        }
    }
}

#[test]
fn input_store_finishes_before_ready_stop_without_another_step() {
    let (vm, mut cpu) = setup(&[0xf3, 0x6c, 0xf4], CpuMode::Real16, false);
    cpu.set_reg(Reg::Rcx, 2);
    cpu.set_reg(Reg::Rdi, 0x80);
    let mut f = cpu.flags();
    f.if_ = true;
    cpu.set_flags(f);
    port_step(&steps(&vm, &mut cpu, 1), false, 1, 0, 1);
    cpu.complete_port_in(0x42);
    assert!(!cpu.irq_ready());
    let r = cpu
        .run_steps(
            &vm,
            1,
            StepOptions::default().with_stop_when_irq_ready(true),
        )
        .unwrap();
    assert_eq!(r.steps(), 0);
    assert!(matches!(r.stop(), StepStop::IrqReady));
    assert!(cpu.irq_ready());
    assert_eq!(cpu.reg(Reg::Rip), 0);
    assert_eq!(cpu.reg(Reg::Rdi), 0x81);
    assert_eq!(cpu.reg(Reg::Rcx), 1);
    assert_eq!(bytes(&vm, 0x20080, 1), [0x42]);
}

#[test]
fn long64_noncanonical_memory_faults_preserve_indices() {
    // SDM Vol 2A §3.3 INS / Vol 2B §4.3 OUTS, 64-Bit Mode Exceptions.
    for jit in [false, true] {
        for out in [false, true] {
            for stack in [false, true] {
                let code = [
                    if stack { 0x36 } else { 0x3e },
                    0xf3,
                    if out { 0x6f } else { 0x6d },
                    0xf4,
                ];
                let (vm, mut cpu) = setup(&code, CpuMode::Long64, jit);
                cpu.set_reg(Reg::Rcx, 2);
                cpu.set_reg(if out { Reg::Rsi } else { Reg::Rdi }, 0x8000_0000_0000);
                if !out {
                    port(&cpu.run(&vm, Some(16)), false, 4, 0);
                    cpu.complete_port_in(0x12345678);
                }
                let exit = cpu.run(&vm, Some(16));
                assert!(
                    matches!(exit, Exit::Exception { vector, addr: 0x1000, .. } if vector == 13),
                    "{exit:?}"
                );
                assert_eq!(cpu.reg(Reg::Rip), 0x1000);
                assert_eq!(cpu.reg(Reg::Rcx), 2);
                assert_eq!(
                    cpu.reg(if out { Reg::Rsi } else { Reg::Rdi }),
                    0x8000_0000_0000
                );
            }
        }
    }
}
