//! Seg16 cached access proofs and exact block boundaries (decision-27).
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::descriptor;
use x86jit_core::{
    derive_legacy_context, CpuModel, ExceptionDelivery, Exit, Prot, RegionKind, SegmentClass,
    StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

fn class(cpu: &Vcpu) -> SegmentClass {
    derive_legacy_context(cpu.legacy_cpu_model().unwrap(), &cpu.cpu)
        .unwrap()
        .segment_class
}

fn setup(model: CpuModel, code: &[u8], cpl: u8, paged: bool, checked: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x20000));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1100, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_a20_enabled(true);
    cpu.cpu.rip = 0x100;
    cpu.cpu.gpr[4] = 0x1000;
    cpu.cpu.legacy.cr0 = if paged { 0x80000001 } else { 1 };
    cpu.cpu.legacy.cr3 = 0x1e000;
    cpu.cpu.legacy.cpl = cpl;
    cpu.cpu.legacy.gdtr.base = 0x3000;
    cpu.cpu.legacy.gdtr.limit = 0x7f;
    cpu.cpu.cs = 8 | u16::from(cpl);
    cpu.cpu.ds = 16 | u16::from(cpl);
    cpu.cpu.es = cpu.cpu.ds;
    cpu.cpu.ss = 24 | u16::from(cpl);
    cpu.cpu.fs = cpu.cpu.ds;
    cpu.cpu.gs = cpu.cpu.ds;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        cache.base = [0x1000, 0x5000, 0x6000, 0x7000, 0x8000, 0x9000][i];
        cache.limit = if i == 1 { 0xff } else { 0x1fff };
        cache.attributes = (if i == 0 { 0x9b } else { 0x93 }) | (u16::from(cpl) << 5);
        cache.unusable = 0;
    }
    descriptor(&vm, 0x3008, 0x1000, 0x1fff, 0x9b | (u16::from(cpl) << 5));
    descriptor(&vm, 0x3010, 0xa000, 0xff, 0x93 | (u16::from(cpl) << 5));
    descriptor(&vm, 0x3018, 0x7000, 0x1fff, 0x93 | (u16::from(cpl) << 5));
    if paged {
        vm.write_bytes(0x1e000, &0x1f007u32.to_le_bytes()).unwrap();
        for page in 0..32u32 {
            vm.write_bytes(
                0x1f000 + u64::from(page * 4),
                &((page * 4096) | 7).to_le_bytes(),
            )
            .unwrap();
        }
    }
    if checked {
        // An unobserved non-present hidden GS cache rejects Seg16 admission.
        // Checked's access policy uses cached type/limit/unusable, not P:
        // SDM Vol 3A §3.4.5 (P is checked on descriptor load), §§6.3–6.4.
        cpu.cpu.legacy.segments[5].attributes &= !0x80;
    }
    assert_eq!(
        class(&cpu),
        if checked {
            SegmentClass::Checked
        } else {
            SegmentClass::Seg16
        }
    );
    (vm, cpu)
}

fn same(a: &Vcpu, b: &mut Vcpu, avm: &Vm, bvm: &Vm) {
    b.cpu.legacy.segments[5].attributes |= 0x80;
    assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu));
    b.cpu.legacy.segments[5].attributes &= !0x80;
    assert_eq!(a.retired_instructions(), b.retired_instructions());
    let mut am = vec![0; 0x20000];
    let mut bm = vec![0; 0x20000];
    avm.read_bytes(0, &mut am).unwrap();
    bvm.read_bytes(0, &mut bm).unwrap();
    assert_eq!(am, bm);
}

#[test]
fn budgets_fault_ordinals_and_segment_loads_match_checked() {
    let bodies: &[&[u8]] = &[
        b"\x40\x01\xc3\x31\xda",
        b"\xa1\x00\x00\x40\xa3\x02\x00",
        b"\x50\x5b\x66\x50\x66\x59",
        b"\x64\xa1\x00\x00\x65\xa3\x04\x00",
        b"\x67\x8b\x03\x66\x01\xc3",
        b"\x8e\xdb\xa1\x00\x00",
        b"\xf3\xa4",
        b"\xe2\xfe",
        b"\xe8\x01\x00\x90\xc3",
    ];
    let faults: &[&[u8]] = &[
        b"\xf4",
        b"\x0f\x0b",
        b"\xf7\xf1",
        b"\xa1\xff\x00",
        b"\x2e\xa3\x00\x00",
        b"\x36\x67\xa1\xff\x1f\x00\x00",
        b"\x8e\xd8\xa1\x00\x00", // null DS load then access
        b"\xfa",                 // privileged/IOPL-sensitive singleton
    ];
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for cpl in [0, 3] {
            for paged in [false, true] {
                for body in bodies {
                    for fault in faults {
                        for ordinal in 0..3 {
                            let mut code = body.to_vec();
                            code.extend(vec![0x42; ordinal]);
                            code.extend_from_slice(fault);
                            for budget in [0, 1, 2, 3, 5, 9] {
                                let (avm, mut a) = setup(model, &code, cpl, paged, false);
                                let (bvm, mut b) = setup(model, &code, cpl, paged, true);
                                for cpu in [&mut a, &mut b] {
                                    cpu.cpu.gpr[1] = 2;
                                    cpu.cpu.gpr[3] = 16 | u64::from(cpl);
                                }
                                let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                                let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                                assert_eq!(
                                    format!("{ar:?}"),
                                    format!("{br:?}"),
                                    "{model:?} CPL{cpl} PG={paged} {code:02x?} budget={budget}"
                                );
                                same(&a, &mut b, &avm, &bvm);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn segment_load_faults_keep_selector_codes_and_prior_retirement() {
    // SDM Vol 2B MOV, Protected Mode Exceptions; Vol 3A §§6.3–6.4:
    // null SS #GP(0), not-present DS #NP(selector), not-present SS #SS(selector).
    for cpl in [0, 1, 2, 3] {
        for paged in [false, true] {
            for (reg, selector, attr, vector, error) in [
                (0xd0, 0, 0x93, 13, 0),
                (0xd8, 0x28 | u16::from(cpl), 0x13, 11, 0x28),
                (0xd0, 0x28 | u16::from(cpl), 0x13, 12, 0x28),
            ] {
                let (vm, mut cpu) = setup(
                    CpuModel::I80386,
                    &[0x43, 0x8e, reg, 0x42],
                    cpl,
                    paged,
                    false,
                );
                descriptor(&vm, 0x3028, 0x5000, 0xff, attr | (u16::from(cpl) << 5));
                cpu.cpu.gpr[0] = u64::from(selector);
                let run = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1);
                assert_eq!(cpu.retired_instructions(), 1);
                assert_eq!(cpu.cpu.rip, 0x101);
                let StepStop::Exit(exit @ Exit::Exception { vector: actual, .. }) = run.stop()
                else {
                    panic!("{run:?}")
                };
                assert_eq!(*actual, vector);
                assert_eq!(
                    exit.fault_metadata().unwrap().error_code(),
                    x86jit_core::fault::ErrorCode::bits(error, 0xffff)
                );
            }
        }
    }
}

#[test]
fn far_calls_and_call_gates_end_blocks_and_reselect() {
    // SDM Vol 3A §6.8.5; Vol 2A CALL and Vol 2B RET: same-ring
    // direct and 16-bit call-gate transfers preserve a balanced word stack.
    for cpl in [0, 3] {
        for paged in [false, true] {
            for gate in [false, true] {
                for budget in 0..9 {
                    let selector = if gate { 0x20 } else { 8 } | cpl;
                    let code = [0x43, 0x9a, 0x80, 1, selector, 0, 0x42, 0xf4];
                    let (avm, mut a) = setup(CpuModel::I80386, &code, cpl, paged, false);
                    let (bvm, mut b) = setup(CpuModel::I80386, &code, cpl, paged, true);
                    for vm in [&avm, &bvm] {
                        vm.write_bytes(0x1180, b"\x40\xcb").unwrap();
                        vm.write_bytes(0x3020, &[0x80, 1, 8 | cpl, 0, 0, 0x84 | (cpl << 5), 0, 0])
                            .unwrap();
                    }
                    let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                    assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                    assert_eq!(
                        a.cpu.gpr[4],
                        if (2..4).contains(&budget) {
                            0xffc
                        } else {
                            0x1000
                        }
                    );
                    same(&a, &mut b, &avm, &bvm);
                    assert_eq!(class(&a), SegmentClass::Seg16);
                    assert_eq!(class(&b), SegmentClass::Checked);
                }
            }
        }
    }
}

#[test]
fn smc_and_device_retries_stop_before_successors() {
    for paged in [false, true] {
        let (avm, mut a) = setup(
            CpuModel::I80386,
            b"\x40\xc6\x06\x07\x01\xf4\x40\x40\xf4",
            0,
            paged,
            false,
        );
        let (bvm, mut b) = setup(
            CpuModel::I80386,
            b"\x40\xc6\x06\x07\x01\xf4\x40\x40\xf4",
            0,
            paged,
            true,
        );
        for cpu in [&mut a, &mut b] {
            cpu.cpu.legacy.segments[1].base = 0x1000;
            cpu.cpu.legacy.segments[1].limit = 0x1fff;
        }
        let ar = a.run_steps(&avm, 20, StepOptions::default()).unwrap();
        let br = b.run_steps(&bvm, 20, StepOptions::default()).unwrap();
        assert_eq!(ar.steps(), 4);
        assert_eq!(a.cpu.gpr[0], 2);
        assert_eq!(format!("{ar:?}"), format!("{br:?}"));
        same(&a, &mut b, &avm, &bvm);

        let (mut avm, mut a) = setup(
            CpuModel::I80386,
            b"\x43\xa1\x00\x00\x40\xa3\x02\x00\x42\xf4",
            3,
            paged,
            false,
        );
        let (mut bvm, mut b) = setup(
            CpuModel::I80386,
            b"\x43\xa1\x00\x00\x40\xa3\x02\x00\x42\xf4",
            3,
            paged,
            true,
        );
        for vm in [&mut avm, &mut bvm] {
            let mut contents = vec![0; 0x20000];
            vm.read_bytes(0, &mut contents).unwrap();
            vm.unmap(0, 0x20000).unwrap();
            vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x6000, 0x1a000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0, &contents[..0x5000]).unwrap();
            vm.write_bytes(0x6000, &contents[0x6000..]).unwrap();
            vm.map(0x5000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        }
        for cpu in [&mut a, &mut b] {
            let vm = if class(cpu) == SegmentClass::Seg16 {
                &avm
            } else {
                &bvm
            };
            let run = cpu.run_steps(vm, 20, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            assert!(matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })));
            cpu.complete_mmio_read(41);
            let run = cpu.run_steps(vm, 20, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 2);
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioWrite { value: 42, .. })
            ));
            cpu.complete_mmio_write();
            let run = cpu.run_steps(vm, 20, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 2);
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::Exception { vector: 13, .. })
            ));
            assert_eq!(cpu.retired_instructions(), 5);
        }
        b.cpu.legacy.segments[5].attributes |= 0x80;
        assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu));
    }
}

#[test]
fn whole_operand_edges_and_expand_down_bounds() {
    // SDM Vol 3A §§3.4.5.1, 6.3–6.4: inclusive expand-up limit;
    // expand-down lower bound is exclusive, B=0 upper bound is FFFFh.
    for cpl in [0, 3] {
        for paged in [false, true] {
            for (segment, prefix) in [0x2e, 0x3e, 0x26, 0x36, 0x64, 0x65].into_iter().enumerate() {
                for down in [false, true] {
                    if segment == 0 && down {
                        continue;
                    }
                    let limit = 0x1ff;
                    for bytes in [1u8, 2, 4] {
                        for offset in [
                            0,
                            limit - 3,
                            limit - 2,
                            limit - 1,
                            limit,
                            limit + 1,
                            0xfffc,
                            0xfffd,
                            0xfffe,
                            0xffff,
                        ] {
                            let mut code = vec![0x43, prefix];
                            if bytes == 4 {
                                code.push(0x66);
                            }
                            code.push(if bytes == 1 { 0xa0 } else { 0xa1 });
                            code.extend_from_slice(&(offset as u16).to_le_bytes());
                            let (avm, mut a) = setup(CpuModel::I80386, &code, cpl, paged, false);
                            let (bvm, mut b) = setup(CpuModel::I80386, &code, cpl, paged, true);
                            for cpu in [&mut a, &mut b] {
                                cpu.cpu.legacy.segments[segment].limit = limit;
                                if down {
                                    cpu.cpu.legacy.segments[segment].attributes |= 4;
                                }
                            }
                            let ar = a.run_steps(&avm, 2, StepOptions::default()).unwrap();
                            let br = b.run_steps(&bvm, 2, StepOptions::default()).unwrap();
                            let allowed = if down {
                                offset > limit && offset + u32::from(bytes) - 1 <= 0xffff
                            } else {
                                offset + u32::from(bytes) - 1 <= limit
                            };
                            assert_eq!(
                                ar.steps(),
                                if allowed { 2 } else { 1 },
                                "segment={segment} down={down} offset={offset:x} bytes={bytes}"
                            );
                            if !allowed {
                                assert!(
                                    matches!(ar.stop(), StepStop::Exit(Exit::Exception { vector, .. }) if *vector == if segment == 3 { 12 } else { 13 })
                                );
                            }
                            assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                            same(&a, &mut b, &avm, &bvm);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn null_access_segment_precedence_and_second_page_cr2() {
    // SDM Vol 3A §§6.3–6.4, 5.6.1, 7.15 (#PF): segmentation
    // precedes paging; a crossing operand faults on its first missing byte.
    for cpl in [0, 3] {
        for store in [false, true] {
            let code = [0x43, if store { 0xa3 } else { 0xa1 }, 0xff, 0x0f, 0x42];
            for limit in [0xff, 0x1fff] {
                let (avm, mut a) = setup(CpuModel::I80386, &code, cpl, true, false);
                let (bvm, mut b) = setup(CpuModel::I80386, &code, cpl, true, true);
                for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                    cpu.cpu.legacy.segments[1].limit = limit;
                    cpu.cpu.legacy.cr2 = 0x1234;
                    vm.write_bytes(0x1f018, &0u32.to_le_bytes()).unwrap();
                }
                let ar = a.run_steps(&avm, 10, StepOptions::default()).unwrap();
                let br = b.run_steps(&bvm, 10, StepOptions::default()).unwrap();
                assert_eq!(ar.steps(), 1);
                assert_eq!(
                    a.cpu.legacy.cr2,
                    if limit == 0xff { 0x1234 } else { 0x6000 }
                );
                assert!(
                    matches!(ar.stop(), StepStop::Exit(Exit::Exception { vector, .. }) if *vector == if limit == 0xff { 13 } else { 14 })
                );
                if limit != 0xff {
                    let StepStop::Exit(exit) = ar.stop() else {
                        unreachable!()
                    };
                    assert_eq!(
                        exit.fault_metadata().unwrap().error_code(),
                        x86jit_core::fault::ErrorCode::bits(
                            u32::from(store) * 2 + u32::from(cpl == 3) * 4,
                            7
                        )
                    );
                }
                assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                same(&a, &mut b, &avm, &bvm);
            }
        }
        for paged in [false, true] {
            let (vm, mut cpu) = setup(
                CpuModel::I80386,
                b"\x43\x8e\xd8\xa1\x00\x00",
                cpl,
                paged,
                false,
            );
            let run = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 2);
            assert_eq!(cpu.cpu.legacy.segments[1].unusable, 1);
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::Exception { vector: 13, .. })
            ));
        }
    }
}

#[test]
fn nonaligned_cs_pages_and_straddling_instructions_fetch_only_required_bytes() {
    // SDM Vol 3A §§5.3, 5.6.1, 7.15 (#PF): bound blocks by the
    // linear page after CS.base addition; only required fetch bytes may fault.
    for cpl in [0, 3] {
        for straddle in [false, true] {
            let (avm, mut a) = setup(CpuModel::I80386, b"", cpl, true, false);
            let (bvm, mut b) = setup(CpuModel::I80386, b"", cpl, true, true);
            for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                cpu.cpu.legacy.segments[0].base = 0x1001;
                cpu.cpu.rip = 0xffe;
                vm.write_bytes(0x1fff, if straddle { b"\xb8" } else { b"\x40" })
                    .unwrap();
                vm.write_bytes(0x1f008, &0u32.to_le_bytes()).unwrap();
            }
            let ar = a.run_steps(&avm, 1, StepOptions::default()).unwrap();
            let br = b.run_steps(&bvm, 1, StepOptions::default()).unwrap();
            assert_eq!(ar.steps(), u64::from(!straddle));
            assert_eq!(a.cpu.legacy.cr2, if straddle { 0x2000 } else { 0 });
            assert_eq!(format!("{ar:?}"), format!("{br:?}"));
            same(&a, &mut b, &avm, &bvm);
            if !straddle {
                let run = a.run_steps(&avm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 0);
                assert_eq!(a.cpu.legacy.cr2, 0x2000);
                assert!(matches!(
                    run.stop(),
                    StepStop::Exit(Exit::Exception { vector: 14, .. })
                ));
            }
        }
    }
}

#[test]
fn stack_width_and_near_target_faults_preserve_predecessors() {
    // SDM Vol 3A §§6.3, 6.8.3; Vol 2A CALL/Jcc, Vol 2B RET/LOOP:
    // taken near targets check CS; failing CALL/RET/LOOP preserve stack/count.
    for cpl in [0, 3] {
        for paged in [false, true] {
            for wide in [false, true] {
                for code in [
                    &b"\x43\xc3"[..],
                    &b"\x43\xe8\xfc\x1e"[..],
                    &b"\x43\xe2\x10"[..],
                    &b"\x43\x75\x10"[..],
                ] {
                    let (avm, mut a) = setup(CpuModel::I80486Dx, code, cpl, paged, false);
                    let (bvm, mut b) = setup(CpuModel::I80486Dx, code, cpl, paged, true);
                    for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                        if wide {
                            cpu.cpu.legacy.segments[3].attributes |= 0x4000;
                        } else {
                            cpu.cpu.gpr[4] |= 0x12340000;
                        }
                        if code[1] != 0xc3 && code[1] != 0xe8 {
                            cpu.cpu.legacy.segments[0].limit = 0x108;
                        }
                        cpu.cpu.gpr[1] = 2;
                        vm.write_bytes(0x8000, &0x2000u16.to_le_bytes()).unwrap();
                    }
                    let ar = a.run_steps(&avm, 10, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, 10, StepOptions::default()).unwrap();
                    assert_eq!(ar.steps(), 1, "{code:02x?}");
                    assert_eq!(a.cpu.rip, 0x101);
                    assert_eq!(a.cpu.gpr[4], if wide { 0x1000 } else { 0x12341000 });
                    assert_eq!(a.cpu.gpr[1], 2);
                    assert!(matches!(
                        ar.stop(),
                        StepStop::Exit(Exit::Exception { vector: 13, .. })
                    ));
                    assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                    same(&a, &mut b, &avm, &bvm);
                }
            }
        }
    }
}

#[test]
fn paged_cpl3_segment_reloads_and_far_returns_retain_context_blocks() {
    // SDM Vol 2A CALL/MOV, Vol 2B RET; spec.md §17.10 Seg16.
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        let code = b"\x8e\xd8\x8e\xc0\x9a\x20\x01\x0b\x00";
        let (vm, mut cpu) = setup(model, code, 3, true, false);
        vm.write_bytes(0x1120, b"\xcb").unwrap();
        let mut warm = None;
        for round in 0..4 {
            cpu.cpu.rip = 0x100;
            cpu.cpu.gpr[0] = 19;
            fixture::steps(&mut cpu, &vm, 4);
            assert_eq!(cpu.cpu.rip, 0x109);
            assert_eq!(cpu.cpu.legacy.segments[1].base, 0xa000);
            assert_eq!(cpu.cpu.legacy.segments[2].base, 0xa000);
            let proof = cpu.paging_cache_for_test();
            assert!(proof.1 > 0, "{proof:?}");
            if round == 1 {
                warm = Some(proof);
            } else if round > 1 {
                assert_eq!(Some(proof), warm);
            }
        }
    }
}
