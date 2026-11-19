//! Legacy restart and frozen-alias witnesses. Values are derived from the cited
//! manuals, or explicitly from the parent implementation for compatibility.
use x86jit_core::{
    CpuMode, CpuModel, ExceptionDelivery, Exit, Knowledge, Prot, Reg, RegionKind, SegmentId,
    StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

fn setup(model: Option<CpuModel>, checked: bool, code: &[u8]) -> (Vm, Vcpu) {
    setup_with_memory(model, checked, code, 0x40000)
}
fn setup_with_memory(
    model: Option<CpuModel>,
    checked: bool,
    code: &[u8],
    bytes: u64,
) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(bytes));
    if let Some(model) = model {
        vm.set_cpu_model(model);
        vm.set_exception_delivery(ExceptionDelivery::Outward);
    } else {
        vm.set_cpu_mode(CpuMode::Real16);
    }
    for page in (0..0x40000).step_by(0x1000) {
        vm.map(page, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    }
    vm.write_bytes(0x2000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x2000;
    cpu.cpu.gpr[4] = 0x8000;
    if checked {
        cpu.cpu.legacy.segments[SegmentId::Es as usize].attributes = 0x92;
    }
    (vm, cpu)
}
fn trap(vm: &mut Vm, page: u64) {
    vm.unmap(page, 0x1000).unwrap();
    vm.map(page, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
}
fn read(vm: &Vm, cpu: &mut Vcpu, at: u64, size: u8, value: u64) {
    let result = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!((result.steps(), cpu.retired_instructions()), (0, 0));
    assert!(
        matches!(result.stop(), StepStop::Exit(Exit::MmioRead { addr, size: n, .. }) if *addr == at && *n == size),
        "{:?}",
        result.stop()
    );
    cpu.complete_mmio_read(value);
}
fn done(vm: &Vm, cpu: &mut Vcpu) {
    let result = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert!(
        matches!(result.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{:?}",
        result.stop()
    );
    assert_eq!((result.steps(), cpu.retired_instructions()), (1, 1));
}
fn word(vm: &Vm, at: u64) -> u16 {
    let mut bytes = [0; 2];
    vm.read_bytes(at, &mut bytes).unwrap();
    u16::from_le_bytes(bytes)
}

#[test]
fn far_pointer_two_mmio_reads_complete_once() {
    // SDM Vol 2A §3.3 LDS/LES/LSS/LFS/LGS; 80386 PRM §9.1:
    // both pointer parts must survive restart before destinations commit.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        for (code, segment) in [
            (&b"\x66\xc5\x06\x00\x40"[..], SegmentId::Ds),
            (&b"\x66\xc4\x06\x00\x40"[..], SegmentId::Es),
            (&b"\x66\x0f\xb2\x06\x00\x40"[..], SegmentId::Ss),
            (&b"\x66\x0f\xb4\x06\x00\x40"[..], SegmentId::Fs),
            (&b"\x66\x0f\xb5\x06\x00\x40"[..], SegmentId::Gs),
        ] {
            let (mut vm, mut cpu) = setup(model, checked, code);
            trap(&mut vm, 0x4000);
            if model.is_none() && matches!(segment, SegmentId::Ss | SegmentId::Fs | SegmentId::Gs) {
                // Decision-23 §6 freezes the alias's accepted decoder forms:
                // LSS/LFS/LGS remain embedder exits; I386 executes them below.
                let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(matches!(
                    result.stop(),
                    StepStop::Exit(Exit::UnknownInstruction { .. })
                ));
                assert_eq!((result.steps(), cpu.retired_instructions()), (0, 0));
                continue;
            }
            read(&vm, &mut cpu, 0x4000, 4, 0x1234_5678);
            read(&vm, &mut cpu, 0x4004, 2, 0x200);
            assert_eq!(cpu.cpu.gpr[0], 0);
            done(&vm, &mut cpu);
            assert_eq!(cpu.cpu.gpr[0], 0x1234_5678);
            assert_eq!(
                match segment {
                    SegmentId::Cs => cpu.cpu.cs,
                    SegmentId::Ds => cpu.cpu.ds,
                    SegmentId::Es => cpu.cpu.es,
                    SegmentId::Ss => cpu.cpu.ss,
                    SegmentId::Fs => cpu.cpu.fs,
                    SegmentId::Gs => cpu.cpu.gs,
                },
                0x200
            );
        }
    }
}

#[test]
fn enter_preserves_ram_display_across_frame_write_retry() {
    // SDM Vol 2A §3.3 ENTER; 80386 PRM §17 ENTER:
    // nesting level 3 copies two display words before the final frame pointer.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80286), false),
        (Some(CpuModel::I80286), true),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\xc8\x00\x00\x03");
        trap(&mut vm, 0);
        cpu.cpu.gpr[4] = 0x1006;
        cpu.cpu.gpr[5] = 0x1002;
        vm.write_bytes(0x1000, &0x1111u16.to_le_bytes()).unwrap();
        read(&vm, &mut cpu, 0xffe, 2, 0x2222);
        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(
                result.stop(),
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0xffe,
                    size: 2,
                    value: 0x1004,
                    ..
                })
            ),
            "{:?}",
            result.stop()
        );
        assert_eq!(
            (word(&vm, 0x1004), word(&vm, 0x1002), word(&vm, 0x1000)),
            (0x1002, 0x1111, 0x2222)
        );
        cpu.complete_mmio_write();
        done(&vm, &mut cpu);
        assert_eq!(word(&vm, 0x1002), 0x1111, "{model:?} checked={checked}");
        assert_eq!((cpu.cpu.gpr[4], cpu.cpu.gpr[5]), (0xffe, 0x1004));
    }
}

// Retry snapshots are engine storage; only answered device transfers are evidence.
fn enter_read_fault(vm: &Vm, cpu: &mut Vcpu, expected: &[(u64, u64)]) {
    let registers = cpu.cpu.gpr;
    let selectors = (
        cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs,
    );
    let flags = cpu.cpu.rflags();
    let result = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    let StepStop::Exit(exit @ Exit::UnmappedMemory { .. }) = result.stop() else {
        panic!("{:?}", result.stop());
    };
    assert_eq!((result.steps(), cpu.retired_instructions()), (0, 0));
    assert_eq!(cpu.cpu.gpr, registers);
    assert_eq!(cpu.cpu.rip, 0x2000);
    assert_eq!(cpu.cpu.rflags(), flags);
    assert_eq!(
        (cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs),
        selectors
    );
    let progress = exit.fault_metadata().unwrap().progress();
    assert_eq!(progress.acknowledged_transfers(), expected);
    if expected.is_empty() {
        assert_ne!(progress.had_partial_progress(), Knowledge::Known(true));
    } else {
        assert_eq!(progress.had_partial_progress(), Knowledge::Known(true));
    }
}

#[test]
fn enter_ram_display_snapshot_is_not_device_evidence() {
    // SDM Vol 2A §3.3 ENTER; 80386 PRM §9.1: a later display-read
    // fault precedes frame writes and architectural register commits.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80286), false),
        (Some(CpuModel::I80286), true),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\xc8\x00\x00\x03");
        cpu.cpu.gpr[5] = 0x1002;
        vm.write_bytes(0x1000, &0x1111u16.to_le_bytes()).unwrap();
        vm.unmap(0, 0x1000).unwrap();
        enter_read_fault(&vm, &mut cpu, &[]);
        assert_eq!(word(&vm, 0x7ffe), 0);
        assert_eq!(word(&vm, 0x1000), 0x1111);
    }
}

#[test]
fn enter_device_display_retains_bus_address_evidence() {
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80286), false),
        (Some(CpuModel::I80286), true),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\xc8\x00\x00\x03");
        cpu.cpu.gpr[5] = 0x1002;
        trap(&mut vm, 0x1000);
        vm.unmap(0, 0x1000).unwrap();
        read(&vm, &mut cpu, 0x1000, 2, 0x1111);
        enter_read_fault(&vm, &mut cpu, &[(0x1000, 0x1111)]);
        assert_eq!(word(&vm, 0x7ffe), 0);
    }
}

#[test]
fn enter_partial_device_display_retains_each_answer_as_evidence() {
    // SDM Vol 2A §3.3 ENTER: earlier staged pushes can overlap a display.
    // Three device bytes plus one staged byte become one internal snapshot.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\x66\xc8\x00\x00\x04");
        cpu.cpu.gpr[4] = 0x1008;
        cpu.cpu.gpr[5] = 0x1005;
        trap(&mut vm, 0);
        for (at, value) in [(0xffd, 0x11), (0xffe, 0x22), (0xfff, 0x33)] {
            read(&vm, &mut cpu, at, 1, value);
        }
        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(
            result.stop(),
            StepStop::Exit(Exit::MmioRead { addr: 0xff9, .. })
        ));
        vm.unmap(0, 0x1000).unwrap();
        enter_read_fault(
            &vm,
            &mut cpu,
            &[(0xffd, 0x11), (0xffe, 0x22), (0xfff, 0x33)],
        );
        assert_eq!(word(&vm, 0x1004), 0);
        assert_eq!(word(&vm, 0x1000), 0);
    }
}

#[test]
fn enter_changed_bp_rereads_display_at_new_address() {
    // SDM Vol 2A §3.3 ENTER: display addresses follow the current frame pointer.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80286), false),
        (Some(CpuModel::I80286), true),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\xc8\x00\x00\x03");
        cpu.cpu.gpr[5] = 0x1002;
        vm.write_bytes(0x1000, &0x1111u16.to_le_bytes()).unwrap();
        vm.write_bytes(0x3002, &0x2222u16.to_le_bytes()).unwrap();
        vm.write_bytes(0x3000, &0x3333u16.to_le_bytes()).unwrap();
        vm.unmap(0, 0x1000).unwrap();
        enter_read_fault(&vm, &mut cpu, &[]);
        cpu.set_reg(Reg::Rbp, 0x3004);
        done(&vm, &mut cpu);
        assert_eq!((word(&vm, 0x7ffc), word(&vm, 0x7ffa)), (0x2222, 0x3333));
    }
}

#[test]
fn enter_changed_a20_policy_rereads_display() {
    for (model, checked) in [
        (Some(CpuModel::I80286), false),
        (Some(CpuModel::I80286), true),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\xc8\x00\x00\x03");
        cpu.cpu.gpr[5] = 0x1002;
        vm.write_bytes(0x1000, &0x1111u16.to_le_bytes()).unwrap();
        vm.unmap(0, 0x1000).unwrap();
        enter_read_fault(&vm, &mut cpu, &[]);
        cpu.cpu.legacy.a20_enabled ^= 1;
        vm.write_bytes(0x1000, &0x2222u16.to_le_bytes()).unwrap();
        vm.map(0, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
        vm.write_bytes(0xffe, &0x3333u16.to_le_bytes()).unwrap();
        done(&vm, &mut cpu);
        assert_eq!((word(&vm, 0x7ffc), word(&vm, 0x7ffa)), (0x2222, 0x3333));
    }
}

fn incomplete_enter_display(model: Option<CpuModel>, checked: bool) {
    // SDM Vol 2A §3.3 ENTER: the first display dword overlaps one staged
    // frame byte. A fault after two device bytes leaves no complete snapshot.
    let (mut vm, mut cpu) = setup(model, checked, b"\x66\xc8\x00\x00\x02");
    cpu.cpu.gpr[4] = 0x1000;
    cpu.cpu.gpr[5] = 0x1003;
    vm.unmap(0x1000, 0x1000).unwrap();
    vm.map(0x1000, 2, Prot::RW, RegionKind::Trap).unwrap();
    read(&vm, &mut cpu, 0x1000, 1, 0x11);
    read(&vm, &mut cpu, 0x1001, 1, 0x22);
    enter_read_fault(&vm, &mut cpu, &[(0x1000, 0x11), (0x1001, 0x22)]);
    assert_eq!(word(&vm, 0xffc), 0);
}

#[test]
fn enter_incomplete_display_evidence_alias() {
    incomplete_enter_display(None, false);
}

#[test]
fn enter_incomplete_display_evidence_direct() {
    incomplete_enter_display(Some(CpuModel::I80386), false);
}

#[test]
fn enter_incomplete_display_evidence_checked() {
    incomplete_enter_display(Some(CpuModel::I80386), true);
}

#[test]
fn enter_changed_bp_abandons_prior_device_evidence() {
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\x66\xc8\x00\x00\x03");
        cpu.cpu.gpr[4] = 0x1000;
        cpu.cpu.gpr[5] = 0x1003;
        vm.unmap(0x1000, 0x1000).unwrap();
        vm.map(0x1000, 2, Prot::RW, RegionKind::Trap).unwrap();
        read(&vm, &mut cpu, 0x1000, 1, 0x11);
        read(&vm, &mut cpu, 0x1001, 1, 0x22);
        enter_read_fault(&vm, &mut cpu, &[(0x1000, 0x11), (0x1001, 0x22)]);
        cpu.set_reg(Reg::Rbp, 0x4004);
        vm.write_bytes(0x4000, &0x7777u32.to_le_bytes()).unwrap();
        vm.unmap(0x3000, 0x1000).unwrap();
        enter_read_fault(&vm, &mut cpu, &[]);
    }
}

#[test]
fn enter_display_word_spans_device_page_boundary() {
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80286), false),
        (Some(CpuModel::I80286), true),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\xc8\x00\x00\x03");
        cpu.cpu.gpr[5] = 0x1001;
        vm.unmap(0, 0x1000).unwrap();
        vm.unmap(0x1000, 0x1000).unwrap();
        vm.map(0, 0x2000, Prot::RW, RegionKind::Trap).unwrap();
        // One bus answer can span two pages of the same device region.
        read(&vm, &mut cpu, 0xfff, 2, 0x2211);
        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(
            result.stop(),
            StepStop::Exit(Exit::MmioRead { addr: 0xffd, .. })
        ));
        vm.unmap(0, 0x2000).unwrap();
        enter_read_fault(&vm, &mut cpu, &[(0xfff, 0x2211)]);
        vm.map(0, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
        vm.write_bytes(0xffd, &0x3333u16.to_le_bytes()).unwrap();
        done(&vm, &mut cpu);
        assert_eq!((word(&vm, 0x7ffc), word(&vm, 0x7ffa)), (0x2211, 0x3333));
    }
}

#[test]
fn enter_display_word_split_across_a20_device_pages() {
    // SDM Vol 3A §11.7.13.4: the board gate splits the word at 1 MiB.
    // SDM Vol 2A §3.3 ENTER: both byte answers belong to one display value.
    // The alias ignores A20; this bus split is a Machine-only model fact.
    for checked in [false, true] {
        let (mut vm, mut cpu) = setup_with_memory(
            Some(CpuModel::I80386),
            checked,
            b"\xc8\x00\x00\x03",
            0x200000,
        );
        cpu.cpu.ss = 0xffff;
        cpu.cpu.legacy.segments[SegmentId::Ss as usize].base = 0xffff0;
        cpu.cpu.legacy.a20_enabled = 0;
        cpu.cpu.gpr[5] = 0x11;
        trap(&mut vm, 0);
        vm.map(0xff000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        read(&vm, &mut cpu, 0xfffff, 1, 0x11);
        read(&vm, &mut cpu, 0, 1, 0x22);
        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(
            result.stop(),
            StepStop::Exit(Exit::MmioRead {
                addr: 0xffffd,
                size: 2,
                ..
            })
        ));
        vm.unmap(0xff000, 0x1000).unwrap();
        enter_read_fault(&vm, &mut cpu, &[(0xfffff, 0x11), (0, 0x22)]);
        vm.map(0xff000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
        vm.write_bytes(0xffffd, &0x3333u16.to_le_bytes()).unwrap();
        done(&vm, &mut cpu);
        assert_eq!((word(&vm, 0x7fec), word(&vm, 0x7fea)), (0x2211, 0x3333));
    }
}

#[test]
fn enter_fault_after_completed_frame_writes_keeps_device_evidence() {
    // SDM Vol 2A §3.3 ENTER; 80386 PRM §9.1: frame-write retries
    // retain both display answers and completed device writes.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\xc8\x00\x00\x03");
        cpu.cpu.gpr[4] = 0x1004;
        cpu.cpu.gpr[5] = 0x4004;
        trap(&mut vm, 0x4000);
        trap(&mut vm, 0x1000);
        trap(&mut vm, 0);
        read(&vm, &mut cpu, 0x4002, 2, 0x1111);
        read(&vm, &mut cpu, 0x4000, 2, 0x2222);
        for (at, value) in [(0x1002, 0x4004), (0x1000, 0x1111)] {
            let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(result.stop(), StepStop::Exit(Exit::MmioWrite { addr, value: v, size: 2, .. }) if *addr == at && *v == value)
            );
            assert_eq!((result.steps(), cpu.retired_instructions()), (0, 0));
            cpu.complete_mmio_write();
        }
        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(
            result.stop(),
            StepStop::Exit(Exit::MmioWrite {
                addr: 0xffe,
                size: 2,
                ..
            })
        ));
        vm.unmap(0, 0x1000).unwrap();
        enter_read_fault(
            &vm,
            &mut cpu,
            &[
                (0x4002, 0x1111),
                (0x4000, 0x2222),
                (0x1002, 0x4004),
                (0x1000, 0x1111),
            ],
        );
    }
}

#[test]
fn all_register_stack_frame_limit_precedes_mmio() {
    // SDM Vol 2B §4.3 POPA/POPAD/PUSHA/PUSHAD; 80386 PRM §17 POPA:
    // the complete 16/32-byte frame must fit before the first stack transfer.
    for (code, sp, limit) in [
        (b"\x61".as_slice(), 0x3000, 0x3007),
        (b"\x66\x61".as_slice(), 0x3000, 0x300f),
        (b"\x60".as_slice(), 0x3010, 0x3007),
        (b"\x66\x60".as_slice(), 0x3020, 0x300f),
    ] {
        let (mut vm, mut cpu) = setup(Some(CpuModel::I80386), true, code);
        trap(&mut vm, 0x3000);
        cpu.cpu.legacy.cr0 = 1;
        cpu.cpu.legacy.segments[SegmentId::Ss as usize].limit = limit;
        cpu.cpu.gpr[4] = sp;
        let before = cpu.cpu.gpr;
        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(
                result.stop(),
                StepStop::Exit(Exit::Exception { vector: 12, .. })
            ),
            "{:?}",
            result.stop()
        );
        assert_eq!((result.steps(), cpu.retired_instructions()), (0, 0));
        assert_eq!(cpu.cpu.gpr, before);
        let StepStop::Exit(exit) = result.stop() else {
            unreachable!()
        };
        assert_eq!(
            exit.fault_metadata().unwrap().error_code(),
            x86jit_core::fault::ErrorCode::bits(0, 0xffff)
        );
    }
}

#[test]
fn alias_parent_67_mov_uses_low16_limit_gate_and_full_linear_address() {
    // Decision-23 §6 freezes this parent quirk. At 24a97d0 the gate tests
    // (EBX & FFFFh) + 2, but the load still uses DS.base + EBX.
    let (vm, mut cpu) = setup(None, false, b"\x67\x8b\x03");
    cpu.cpu.gpr[3] = 0x10000;
    vm.write_bytes(0x10000, &0x1234u16.to_le_bytes()).unwrap();
    done(&vm, &mut cpu);
    assert_eq!(cpu.cpu.gpr[0], 0x1234);
}

#[test]
fn multi_read_frames_resume_each_transfer_once() {
    // SDM Vol 2B §4.3 RET/POPA/POP; Vol 2A §3.3 IRET/LEAVE;
    // 80386 PRM §9.1: completed reads survive instruction restart.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        for (code, answers, final_sp, final_bp, final_ip) in [
            (
                b"\xcb".as_slice(),
                vec![(0x8000, 0x2200), (0x8002, 0)],
                0x8004,
                0x8000,
                0x2200,
            ),
            (
                b"\xcf".as_slice(),
                vec![(0x8000, 0x2200), (0x8002, 0), (0x8004, 2)],
                0x8006,
                0x8000,
                0x2200,
            ),
            (
                b"\x61".as_slice(),
                vec![
                    (0x8000, 7),
                    (0x8002, 6),
                    (0x8004, 5),
                    (0x8008, 3),
                    (0x800a, 2),
                    (0x800c, 1),
                    (0x800e, 0),
                ],
                0x8010,
                5,
                0x2001,
            ),
            (
                b"\xc9".as_slice(),
                vec![(0x8000, 0x1234)],
                0x8002,
                0x1234,
                0x2001,
            ),
        ] {
            let (mut vm, mut cpu) = setup(model, checked, code);
            trap(&mut vm, 0x8000);
            cpu.cpu.gpr[5] = 0x8000;
            for (at, value) in answers {
                read(&vm, &mut cpu, at, 2, value);
            }
            done(&vm, &mut cpu);
            assert_eq!(
                (cpu.cpu.gpr[4], cpu.cpu.gpr[5], cpu.cpu.rip),
                (final_sp, final_bp, final_ip)
            );
        }
    }
}

#[test]
fn pop_memory_retains_stack_read_across_device_write() {
    // SDM Vol 2B §4.3 POP; 80386 PRM §9.1.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\x8f\x06\x00\x80");
        trap(&mut vm, 0x8000);
        read(&vm, &mut cpu, 0x8000, 2, 0x1234);
        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(
                result.stop(),
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0x8000,
                    value: 0x1234,
                    ..
                })
            ),
            "{:?}",
            result.stop()
        );
        cpu.complete_mmio_write();
        done(&vm, &mut cpu);
        assert_eq!(cpu.cpu.gpr[4], 0x8002);
    }
}

#[test]
fn frozen_alias_parent_compatibility_table() {
    // READ: expectations were hand-evaluated from 24a97d0's lower_read,
    // with_segment, scalar ALU, exec_rep_string_real and stack handlers.
    // In particular the parent masks only the scalar limit gate, ignores
    // FS/GS for these ordinary MOVs, and uses ES:DI for string destinations.
    // These are compatibility assertions (decision-23 §6), not ISA claims.
    struct Case {
        code: &'static [u8],
        input: &'static [(usize, u64)],
        output: &'static [(usize, u64)],
        flags: u64,
        memory: Option<(u64, &'static [u8])>,
    }
    let cases = [
        Case {
            code: b"\x66\x01\xd0",
            input: &[],
            output: &[(0, 0x1234_89ab)],
            flags: 2,
            memory: None,
        },
        Case {
            code: b"\x67\x8b\x03",
            input: &[],
            output: &[(0, 0x1234_beef)],
            flags: 2,
            memory: None,
        },
        Case {
            // READ: parent effective_address_no_segment keeps disp32=10000h;
            // with_segment/SegLimitCheck masks only the gate to zero. The word
            // load reads linear 10000h and the AX write preserves high EAX.
            code: b"\x67\xa1\x00\x00\x01\x00",
            input: &[],
            output: &[(0, 0x1234_1234)],
            flags: 2,
            memory: Some((0x10000, b"\x34\x12")),
        },
        Case {
            code: b"\x66\x67\x8b\x03",
            input: &[],
            output: &[(0, 0xdead_beef)],
            flags: 2,
            memory: None,
        },
        Case {
            code: b"\x66\x67\x89\x03",
            input: &[],
            output: &[],
            flags: 2,
            memory: Some((0x10000, b"\x78\x56\x34\x12")),
        },
        Case {
            code: b"\x67\x83\x03\x01",
            input: &[],
            output: &[],
            flags: 0x96,
            memory: Some((0x10000, b"\xf0\xbe")),
        },
        Case {
            code: b"\x66\x67\x83\x03\x01",
            input: &[],
            output: &[],
            flags: 0x96,
            memory: Some((0x10000, b"\xf0\xbe\xad\xde")),
        },
        Case {
            code: b"\x26\x8b\x04",
            input: &[],
            output: &[(0, 0x1234_4321)],
            flags: 2,
            memory: None,
        },
        Case {
            code: b"\x64\x8b\x04",
            input: &[],
            output: &[(0, 0x1234_1234)],
            flags: 2,
            memory: None,
        },
        Case {
            code: b"\x65\x8b\x04",
            input: &[],
            output: &[(0, 0x1234_1234)],
            flags: 2,
            memory: None,
        },
        Case {
            code: b"\x66\xa5",
            input: &[],
            output: &[(6, 0x3004), (7, 0x5004)],
            flags: 2,
            memory: Some((0x6000, b"\x34\x12\xcd\xab")),
        },
        Case {
            code: b"\x67\xa5",
            input: &[(6, 0x10000)],
            output: &[(6, 0x10002), (7, 0x5002)],
            flags: 2,
            memory: Some((0x6000, b"\xef\xbe")),
        },
        Case {
            code: b"\x66\x67\xad",
            input: &[(6, 0x10000)],
            output: &[(0, 0xdead_beef), (6, 0x10004)],
            flags: 2,
            memory: None,
        },
        Case {
            code: b"\x26\xad",
            input: &[],
            output: &[(0, 0x1234_4321), (6, 0x3002)],
            flags: 2,
            memory: None,
        },
        Case {
            code: b"\x50",
            input: &[],
            output: &[(4, 0x7ffe)],
            flags: 2,
            memory: Some((0x7ffe, b"\x78\x56")),
        },
        Case {
            code: b"\x5b",
            input: &[],
            output: &[(3, 0x1_aabb), (4, 0x8002)],
            flags: 2,
            memory: None,
        },
        Case {
            code: b"\xc9",
            input: &[],
            output: &[(4, 0x8102), (5, 0xccdd)],
            flags: 2,
            memory: None,
        },
        Case {
            code: b"\x9c",
            input: &[],
            output: &[(4, 0x7ffe)],
            flags: 2,
            memory: Some((0x7ffe, b"\x02\x00")),
        },
    ];
    for case in cases {
        let (vm, mut cpu) = setup(None, false, case.code);
        cpu.cpu.gpr[..8].copy_from_slice(&[
            0x1234_5678,
            1,
            0x3333,
            0x10000,
            0x8000,
            0x8100,
            0x3000,
            0x5000,
        ]);
        cpu.cpu.es = 0x100;
        cpu.cpu.fs = 0x200;
        cpu.cpu.gs = 0x300;
        cpu.cpu.set_user_rflags(2, 2, false);
        for &(reg, value) in case.input {
            cpu.cpu.gpr[reg] = value;
        }
        let mut expected = cpu.cpu.gpr;
        for &(reg, value) in case.output {
            expected[reg] = value;
        }
        for (at, bytes) in [
            (0x3000, b"\x34\x12\xcd\xab".as_slice()),
            (0x4000, b"\x21\x43".as_slice()),
            (0x6000, b"\x21\x43".as_slice()),
            (0x10000, b"\xef\xbe\xad\xde".as_slice()),
            (0x8000, b"\xbb\xaa".as_slice()),
            (0x8100, b"\xdd\xcc".as_slice()),
        ] {
            vm.write_bytes(at, bytes).unwrap();
        }
        if case.code == b"\x67\xa1\x00\x00\x01\x00" {
            vm.write_bytes(0x10000, b"\x34\x12").unwrap();
        }
        done(&vm, &mut cpu);
        assert_eq!(cpu.cpu.gpr, expected, "{:02x?}", case.code);
        assert_eq!(cpu.cpu.rflags() & 0xffff, case.flags, "{:02x?}", case.code);
        assert_eq!(cpu.cpu.rip, 0x2000 + case.code.len() as u64);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs),
            (0, 0, 0x100, 0, 0x200, 0x300)
        );
        if let Some((at, bytes)) = case.memory {
            let mut actual = vec![0; bytes.len()];
            vm.read_bytes(at, &mut actual).unwrap();
            assert_eq!(actual, bytes, "{:02x?}", case.code);
        }
    }
}

fn alias_gp_handler(vm: &Vm) {
    vm.write_bytes(13 * 4, b"\x00\x28\x00\x00").unwrap();
    vm.write_bytes(0x2800, b"\xf4").unwrap();
}
fn gp_or_ss(vm: &Vm, cpu: &mut Vcpu, model: Option<CpuModel>) {
    if model.is_none() {
        alias_gp_handler(vm);
    }
    let result = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    if let Some(model) = model {
        let vector = if model == CpuModel::I80286 { 13 } else { 12 };
        assert!(
            matches!(result.stop(), StepStop::Exit(Exit::Exception { vector: v, .. }) if *v == vector),
            "{:?}",
            result.stop()
        );
        assert_eq!((result.steps(), cpu.retired_instructions()), (0, 0));
    } else {
        assert!(
            matches!(result.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            result.stop()
        );
        assert_eq!(cpu.cpu.rip, 0x2801);
        assert_eq!(word(vm, cpu.cpu.gpr[4] & 0xffff), 0x2000);
    }
}

#[test]
fn accepted_alias_xlat_uses_address_size_and_fs_gs() {
    // Accepted alias compatibility policy. SDM Vol 2B §4.3 XLAT: address-size
    // selects BX/EBX and a segment override selects the data segment.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (vm, mut cpu) = setup(model, checked, b"\x67\xd7");
        cpu.cpu.gpr[3] = 0x10000;
        cpu.cpu.gpr[0] = 1;
        vm.write_bytes(1, b"\x11").unwrap();
        if model.is_none() {
            alias_gp_handler(&vm);
        }
        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        if model.is_none() {
            assert!(matches!(result.stop(), StepStop::Exit(Exit::Hlt)));
            assert_eq!(cpu.cpu.rip, 0x2801);
            assert_eq!(word(&vm, 0x7ffa), 0x2000);
        } else {
            assert!(matches!(
                result.stop(),
                StepStop::Exit(Exit::Exception { vector: 13, .. })
            ));
            assert_eq!((result.steps(), cpu.retired_instructions()), (0, 0));
        }
        assert_eq!(cpu.cpu.gpr[0], 1);
        for (prefix, seg) in [(0x64, SegmentId::Fs), (0x65, SegmentId::Gs)] {
            let (vm, mut cpu) = setup(model, checked, &[prefix, 0xd7]);
            cpu.cpu.gpr[3] = 0;
            cpu.cpu.gpr[0] = 1;
            cpu.cpu.fs = 0x100;
            cpu.cpu.gs = 0x100;
            if model.is_some() {
                cpu.cpu.legacy.segments[seg as usize].base = 0x1000;
            }
            vm.write_bytes(1, b"\x11").unwrap();
            vm.write_bytes(0x1001, b"\x22").unwrap();
            done(&vm, &mut cpu);
            assert_eq!(cpu.cpu.gpr[0], 0x22);
        }
    }
}

#[test]
fn accepted_stack_wrap_and_whole_operand_limit() {
    // SDM Vol 2B §4.3 PUSH/POP; 80386 PRM §17.1.3: SP wraps
    // independently of operand width, but an individual word must fit.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80286), false),
        (Some(CpuModel::I80286), true),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        for (sp, next) in [(0, 0xfffe), (0xffff, 0xfffd)] {
            let (vm, mut cpu) = setup(model, checked, b"\x50");
            cpu.cpu.gpr[4] = sp;
            cpu.cpu.gpr[0] = 0x1234;
            done(&vm, &mut cpu);
            assert_eq!(cpu.cpu.gpr[4], next);
            assert_eq!(word(&vm, next), 0x1234);
        }
        let (vm, mut cpu) = setup(model, checked, b"\x5b");
        cpu.cpu.gpr[4] = 0xffff;
        cpu.cpu.gpr[3] = 0x1234;
        gp_or_ss(&vm, &mut cpu, model);
        assert_eq!(cpu.cpu.gpr[3], 0x1234);
    }
}

#[test]
fn accepted_pop_uses_sp_with_high_esp_bits() {
    // SDM Vol 2B §4.3 POP; 80386 PRM §17.1.3: SS.B=0 uses SP.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (vm, mut cpu) = setup(model, checked, b"\x5b");
        cpu.cpu.gpr[4] = 0x18000;
        vm.write_bytes(0x8000, &0x1111u16.to_le_bytes()).unwrap();
        vm.write_bytes(0x18000, &0x2222u16.to_le_bytes()).unwrap();
        done(&vm, &mut cpu);
        assert_eq!((cpu.cpu.gpr[3], cpu.cpu.gpr[4]), (0x1111, 0x18002));
    }
}

#[test]
fn accepted_far_call_and_iret_defer_commits() {
    // SDM Vol 2A §3.3 CALL/IRET; 80386 PRM §9.1 and §17 CALL/IRET.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\x9a\x00\x22\x00\x01");
        trap(&mut vm, 0x8000);
        cpu.cpu.gpr[4] = 0x8004;
        for (at, value) in [(0x8002, 0), (0x8000, 0x2005)] {
            let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(result.stop(), StepStop::Exit(Exit::MmioWrite { addr, value: v, size: 2, .. }) if *addr == at && *v == value),
                "{:?}",
                result.stop()
            );
            assert_eq!((result.steps(), cpu.retired_instructions()), (0, 0));
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
                (0, 0x2000, 0x8004)
            );
            cpu.complete_mmio_write();
        }
        done(&vm, &mut cpu);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (0x100, 0x2200, 0x8000)
        );
        let (mut vm, mut cpu) = setup(model, checked, b"\xcf");
        trap(&mut vm, 0x8000);
        for (at, value) in [(0x8000, 0x2200), (0x8002, 0x100), (0x8004, 0x202)] {
            read(&vm, &mut cpu, at, 2, value);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
                (0, 0x2000, 0x8000)
            );
            assert_eq!(cpu.cpu.rflags() & 0x200, 0);
        }
        done(&vm, &mut cpu);
        assert_eq!(
            (
                cpu.cpu.cs,
                cpu.cpu.rip,
                cpu.cpu.gpr[4],
                cpu.cpu.rflags() & 0x200
            ),
            (0x100, 0x2200, 0x8006, 0x200)
        );
    }
}

#[test]
fn accepted_far_frame_preflight_precedes_first_read() {
    // SDM Vol 2A §3.3 IRET and Vol 2B §4.3 RET: preflight a whole frame.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        for (code, sp) in [(b"\xcf".as_slice(), 0xfffc), (b"\xcb".as_slice(), 0xfffe)] {
            let (vm, mut cpu) = setup(model, checked, code);
            vm.write_bytes(sp, &0x1234u16.to_le_bytes()).unwrap();
            cpu.cpu.gpr[4] = sp;
            gp_or_ss(&vm, &mut cpu, model);
        }
    }
}

#[test]
fn accepted_popa_skips_discarded_sp_device_slot() {
    // SDM Vol 2B §4.3 POPA, Operation: increment SP over the discarded slot.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80286), false),
        (Some(CpuModel::I80286), true),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\x61");
        trap(&mut vm, 0x1000);
        cpu.cpu.gpr[4] = 0xffa;
        vm.write_bytes(0xffa, b"\x07\x00\x06\x00\x05\x00").unwrap();
        // The skipped SP slot is at 1000h; the first device read is BX at 1002h.
        for (at, value) in [(0x1002, 3), (0x1004, 2), (0x1006, 1), (0x1008, 0)] {
            read(&vm, &mut cpu, at, 2, value);
        }
        done(&vm, &mut cpu);
        assert_eq!(&cpu.cpu.gpr[..8], &[0, 1, 2, 3, 0x100a, 5, 6, 7]);
    }
}

#[test]
fn accepted_far_call_whole_frame_limit_preflight() {
    // SDM Vol 2A §3.3 CALL: check space for the whole far return frame.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\x9a\x00\x22\x00\x01");
        cpu.cpu.gpr[4] = 2;
        if model.is_some() {
            trap(&mut vm, 0);
        }
        gp_or_ss(&vm, &mut cpu, model);
        if model.is_none() {
            assert_eq!(word(&vm, 0), 2);
        }
    }
}

#[test]
fn all_register_frame_limit_uses_model_fault_vector() {
    // I286 PRM App. B POPA p. B-85; SDM Vol 3B §23.1.4 Table 23-1:
    // I286 real-mode stack limits report #GP; I386 reports #SS.
    for (model, checked) in [
        (Some(CpuModel::I80286), false),
        (Some(CpuModel::I80286), true),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        for (code, sp) in [(b"\x61".as_slice(), 0xfff8), (b"\x60".as_slice(), 8)] {
            let (mut vm, mut cpu) = setup(model, checked, code);
            trap(&mut vm, 0xf000);
            cpu.cpu.gpr[4] = sp;
            gp_or_ss(&vm, &mut cpu, model);
        }
    }
}

#[test]
fn enter_ram_snapshot_is_dropped_after_successful_completion() {
    // A completed ENTER must not reuse its display snapshot at the same IP.
    let (vm, mut cpu) = setup(None, false, b"\xc8\x00\x00\x02\xeb\xfa");
    for display in [0x1111u16, 0x2222] {
        cpu.cpu.rip = 0x2000;
        cpu.cpu.gpr[4] = 0x1006;
        cpu.cpu.gpr[5] = 0x1002;
        vm.write_bytes(0x1000, &display.to_le_bytes()).unwrap();
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        assert_eq!(word(&vm, 0x1002), display);
        assert_eq!(cpu.cpu.mmio_parts_len, 0);
    }
}

#[test]
fn enter_maximum_nesting_partial_display_fits_retry_ledger() {
    // SDM Vol 2A §3.3 ENTER: the nesting level is reduced modulo 32.
    // With this unaligned display each dword contains three fresh bytes and
    // one staged frame byte. Retain values without exhausting the fixed ledger.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (vm, mut cpu) = setup(model, checked, b"\x66\xc8\x00\x00\x1f");
        cpu.cpu.gpr[4] = 0x1008;
        cpu.cpu.gpr[5] = 0x1005;
        done(&vm, &mut cpu);
        assert_eq!((cpu.cpu.gpr[4], cpu.cpu.gpr[5]), (0xf88, 0x1004));
        assert_eq!(cpu.cpu.mmio_parts_len, 0);
    }
}

#[test]
fn enter_maximum_nesting_partial_device_display_and_frame_resume() {
    // SDM Vol 2A §3.3 ENTER; 80386 PRM §9.1: completing a display
    // coalesces its partial answers before later display/frame transfers.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\x66\xc8\x00\x00\x1f");
        trap(&mut vm, 0);
        cpu.cpu.gpr[4] = 0x1008;
        cpu.cpu.gpr[5] = 0x1005;
        for i in 2..31 {
            for byte in 0..3 {
                read(&vm, &mut cpu, 0x1005 - i * 4 + byte, 1, 0x11);
            }
        }
        for i in 3..=32 {
            let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(result.stop(), StepStop::Exit(Exit::MmioWrite { addr, size: 4, .. }) if *addr == 0x1008 - i * 4),
                "{:?}",
                result.stop()
            );
            assert_eq!((result.steps(), cpu.retired_instructions()), (0, 0));
            assert_eq!((cpu.cpu.gpr[4], cpu.cpu.gpr[5]), (0x1008, 0x1005));
            cpu.complete_mmio_write();
        }
        done(&vm, &mut cpu);
        assert_eq!((cpu.cpu.gpr[4], cpu.cpu.gpr[5]), (0xf88, 0x1004));
        assert_eq!(cpu.cpu.mmio_parts_len, 0);
    }
}

#[test]
fn enter_changed_bp_maximum_nesting_drops_abandoned_attempt() {
    // SDM Vol 2A §3.3 ENTER: level 31 copies thirty display words and
    // pushes the old BP and frame pointer, before committing BP and SP.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\xc8\x00\x00\x1f");
        cpu.cpu.gpr[5] = 0x103a;
        vm.unmap(0, 0x1000).unwrap();
        trap(&mut vm, 0x7000);
        enter_read_fault(&vm, &mut cpu, &[]);
        assert_eq!(cpu.cpu.mmio_parts_len, 29);
        cpu.set_reg(Reg::Rbp, 0x403c);
        let mut peak = 0;
        for i in 1..=32 {
            let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(result.steps(), 0);
            assert!(
                matches!(result.stop(), StepStop::Exit(Exit::MmioWrite { addr, size: 2, .. }) if *addr == 0x8000 - i * 2),
                "{model:?}, checked={checked}, write {i}: {:?}",
                result.stop()
            );
            peak = peak.max(cpu.cpu.mmio_parts_len);
            cpu.complete_mmio_write();
        }
        done(&vm, &mut cpu);
        assert_eq!((cpu.cpu.gpr[4], cpu.cpu.gpr[5]), (0x7fc0, 0x7ffe));
        assert!(peak <= 63);
        assert_eq!(cpu.cpu.mmio_parts_len, 0);
    }
}

#[test]
fn enter_changed_bp_does_not_absorb_abandoned_device_answer() {
    // SDM Vol 2A §3.3 ENTER; Vol 3A §11.7.13.4: the display dword
    // crosses the A20 bus gate. An input edit starts a fresh attempt;
    // the old exit owns the completed byte, even if RAM now covers it.
    for checked in [false, true] {
        let (mut vm, mut cpu) = setup_with_memory(
            Some(CpuModel::I80386),
            checked,
            b"\x66\xc8\x00\x00\x03",
            0x200000,
        );
        cpu.cpu.ss = 0xffff;
        cpu.cpu.legacy.segments[SegmentId::Ss as usize].base = 0xffff0;
        cpu.cpu.legacy.a20_enabled = 0;
        cpu.cpu.gpr[5] = 0x13;
        vm.map(0xfffff, 1, Prot::RW, RegionKind::Trap).unwrap();
        vm.unmap(0, 0x1000).unwrap();
        read(&vm, &mut cpu, 0xfffff, 1, 0x11);
        let old_exit = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(
            old_exit.stop(),
            StepStop::Exit(Exit::UnmappedMemory { addr: 0, .. })
        ));
        cpu.set_reg(Reg::Rbp, 0x12);
        vm.unmap(0xfffff, 1).unwrap();
        vm.map(0xffffe, 2, Prot::RW, RegionKind::Ram).unwrap();
        vm.write_bytes(0xffffe, &[0x33, 0x22]).unwrap();
        vm.map(0, 2, Prot::RW, RegionKind::Ram).unwrap();
        // Abandonment deliberately drops the old exit's device answers from
        // this attempt. In particular RAM's 22h must not replace that 11h.
        enter_read_fault(&vm, &mut cpu, &[]);
        let StepStop::Exit(exit) = old_exit.stop() else {
            unreachable!()
        };
        assert_eq!(
            exit.fault_metadata()
                .unwrap()
                .progress()
                .acknowledged_transfers(),
            &[(0xfffff, 0x11)]
        );
    }
}

#[test]
fn checked_popa_fault_reports_completed_device_read() {
    // SDM Vol 2B §4.3 POPA/POPAD: DI is the first popped word.
    // Engine evidence retains that device completion when the next read faults.
    let (mut vm, mut cpu) = setup(Some(CpuModel::I80386), true, b"\x61");
    cpu.cpu.gpr[4] = 0xffe;
    trap(&mut vm, 0);
    vm.unmap(0x1000, 0x1000).unwrap();
    read(&vm, &mut cpu, 0xffe, 2, 0x1234);
    enter_read_fault(&vm, &mut cpu, &[(0xffe, 0x1234)]);
}

#[test]
fn enter_changed_sp_abandons_incomplete_display_answers() {
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\x66\xc8\x00\x00\x02");
        cpu.cpu.gpr[4] = 0x1000;
        cpu.cpu.gpr[5] = 0x1003;
        vm.unmap(0x1000, 0x1000).unwrap();
        vm.map(0x1000, 2, Prot::RW, RegionKind::Trap).unwrap();
        read(&vm, &mut cpu, 0x1000, 1, 0x11);
        read(&vm, &mut cpu, 0x1001, 1, 0x22);
        enter_read_fault(&vm, &mut cpu, &[(0x1000, 0x11), (0x1001, 0x22)]);
        // The display address is unchanged; the staged push now covers 1000h.
        // SDM Vol 2A §3.3 ENTER: the current stack pointer selects that push.
        cpu.set_reg(Reg::Rsp, 0x1001);
        enter_read_fault(&vm, &mut cpu, &[]);
    }
}

#[test]
fn enter_without_display_changed_sp_drops_frame_acknowledgements() {
    // SDM Vol 2A §3.3 ENTER: level one pushes BP and the frame pointer,
    // without reading a display. Input edits must also abandon these attempts.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\xc8\x00\x00\x01");
        trap(&mut vm, 0x7000);
        for attempt in 0..65 {
            let sp = 0x8000 - attempt * 8;
            cpu.set_reg(Reg::Rsp, sp);
            for i in 1..=2 {
                let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(
                    matches!(result.stop(), StepStop::Exit(Exit::MmioWrite { addr, size: 2, .. }) if *addr == sp - i * 2)
                );
                if i == 1 {
                    cpu.complete_mmio_write();
                }
            }
            assert!(cpu.cpu.mmio_parts_len <= 3);
        }
        cpu.complete_mmio_write();
        done(&vm, &mut cpu);
        assert_eq!((cpu.cpu.gpr[4], cpu.cpu.gpr[5]), (0x7dfc, 0x7dfe));
        assert_eq!(cpu.cpu.mmio_parts_len, 0);
    }
}

#[test]
fn enter_changed_ss_preserving_display_addresses_abandons_frame_answers() {
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\xc8\x00\x00\x1f");
        trap(&mut vm, 0x17000);
        trap(&mut vm, 0x18000);
        cpu.set_reg(Reg::Ss, 0x1000);
        cpu.cpu.legacy.segments[SegmentId::Ss as usize].base = 0x10000;
        cpu.set_reg(Reg::Rbp, 0x4000);
        for i in 1..=5 {
            let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(result.stop(), StepStop::Exit(Exit::MmioWrite { addr, size: 2, .. }) if *addr == 0x18000 - i * 2)
            );
            if i < 5 {
                cpu.complete_mmio_write();
            }
        }
        // All display linear addresses stay the same, but every frame address
        // moves. SDM Vol 2A §3.3 ENTER: SS applies to both frame and display.
        cpu.set_reg(Reg::Ss, 0x1100);
        cpu.cpu.legacy.segments[SegmentId::Ss as usize].base = 0x11000;
        cpu.set_reg(Reg::Rbp, 0x3000);
        for i in 1..=32 {
            let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(result.stop(), StepStop::Exit(Exit::MmioWrite { addr, size: 2, .. }) if *addr == 0x19000 - i * 2)
            );
            assert!(cpu.cpu.mmio_parts_len <= 63);
            if i == 1 {
                assert_eq!(cpu.cpu.mmio_parts_len, 30);
            }
            cpu.complete_mmio_write();
        }
        done(&vm, &mut cpu);
        assert_eq!((cpu.cpu.gpr[4], cpu.cpu.gpr[5]), (0x7fc0, 0x7ffe));
    }
}

fn write_exit(vm: &Vm, cpu: &mut Vcpu, at: u64, size: u8, value: u64) {
    let result = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!((result.steps(), cpu.retired_instructions()), (0, 0));
    assert!(
        matches!(result.stop(), StepStop::Exit(Exit::MmioWrite { addr, size: n, value: v, .. })
            if (*addr, *n, *v) == (at, size, value)),
        "{:?}",
        result.stop()
    );
}

#[test]
fn enter_dword_ebp_edit_restarts_overlapping_display() {
    // SDM Vol 2A §3.3 ENTER: the initial push reads EBP at operand size,
    // independently of the SP-sized display offset (80386 PRM §17.1.3).
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\x66\xc8\x00\x00\x02");
        trap(&mut vm, 0);
        cpu.cpu.gpr[4] = 0x1008;
        cpu.cpu.gpr[5] = 0x100a;
        vm.write_bytes(0x1008, &0x1111u16.to_le_bytes()).unwrap();
        write_exit(&vm, &mut cpu, 0xffc, 4, 0x1004);
        cpu.cpu.gpr[5] = 0x2222_100a;
        write_exit(&vm, &mut cpu, 0xffc, 4, 0x1004);
        let mut bytes = [0; 4];
        vm.read_bytes(0x1000, &mut bytes).unwrap();
        assert_eq!(
            u32::from_le_bytes(bytes),
            0x1111_2222,
            "{model:?} checked={checked}"
        );
        cpu.complete_mmio_write();
        done(&vm, &mut cpu);
    }
}

#[test]
fn enter_word_unused_rbp_bits_keep_frame_acknowledgement() {
    // SDM Vol 2A §3.3 ENTER; 80386 PRM §17.1.3: word ENTER on
    // a word stack reads BP and SP; high RBP bits are not retry inputs.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        let (mut vm, mut cpu) = setup(model, checked, b"\xc8\x00\x00\x01");
        trap(&mut vm, 0x7000);
        cpu.cpu.gpr[5] = 0x4004;
        write_exit(&vm, &mut cpu, 0x7ffe, 2, 0x4004);
        cpu.complete_mmio_write();
        write_exit(&vm, &mut cpu, 0x7ffc, 2, 0x7ffe);
        cpu.cpu.gpr[5] = 0x1000_4004;
        write_exit(&vm, &mut cpu, 0x7ffc, 2, 0x7ffe);
        cpu.complete_mmio_write();
        done(&vm, &mut cpu);
    }
}

#[test]
fn ordinary_alias_vector_fault_keeps_full_ledger_address() {
    // Parent 24a97d0 compatibility: ordinary alias address formation keeps
    // the full linear sum; its limit gate alone masks the offset to 16 bits.
    let (mut vm, mut cpu) = setup_with_memory(None, false, b"\x67\xf3\x0f\x6f\x00", 0x1_0000_2000);
    vm.map(0x1_0000_0000, 8, Prot::RW, RegionKind::Trap)
        .unwrap();
    cpu.cpu.ds = 1;
    cpu.cpu.gpr[0] = 0xffff_fff0;
    read(&vm, &mut cpu, 0x1_0000_0000, 8, 0x1234);
    enter_read_fault(&vm, &mut cpu, &[(0x1_0000_0000, 0x1234)]);
}

#[test]
fn enter_unread_register_bits_preserve_each_identity_phase() {
    // SDM Vol 2A §3.3 ENTER: FrameTemp uses operand size, addressing uses
    // stack size, and display addressing reads BP/EBP only at levels > 1.
    for (model, checked, stack32) in [
        (None, false, false),
        (Some(CpuModel::I80386), false, false),
        (Some(CpuModel::I80386), true, false),
        (Some(CpuModel::I80386), true, true),
    ] {
        for size in [2u8, 4] {
            for level in [0u8, 1, 3] {
                let mut code = vec![];
                if size == 4 {
                    code.push(0x66);
                }
                code.extend([0xc8, 0, 0, level]);
                let (mut vm, mut cpu) = setup(model, checked, &code);
                if stack32 {
                    cpu.cpu.legacy.segments[SegmentId::Ss as usize].attributes |= 0x4000;
                }
                cpu.cpu.gpr[5] = 0x4004;
                trap(&mut vm, 0x7000);
                let frame = 0x8000 - u64::from(size);
                write_exit(&vm, &mut cpu, frame, size, 0x4004);
                cpu.complete_mmio_write();
                let bp_bits = if size == 4 || (stack32 && level > 1) {
                    32
                } else {
                    16
                };
                let sp_bits = if size == 4 || stack32 { 32 } else { 16 };
                if level != 0 {
                    write_exit(
                        &vm,
                        &mut cpu,
                        frame - u64::from(size),
                        size,
                        if level == 1 { frame } else { 0 },
                    );
                }
                cpu.cpu.gpr[5] |= 0xfeed_beef_0000_0000 | (1 << bp_bits);
                cpu.cpu.gpr[4] |= 0xcafe_f00d_0000_0000 | (1 << sp_bits);
                cpu.cpu.gpr[0] = 0x1234;
                cpu.cpu.flags.cf = true;
                if level != 0 {
                    let pushes = u64::from(level) + 1;
                    for i in 2..=pushes {
                        write_exit(
                            &vm,
                            &mut cpu,
                            0x8000 - i * u64::from(size),
                            size,
                            if i == pushes { frame } else { 0 },
                        );
                        cpu.complete_mmio_write();
                    }
                }
                done(&vm, &mut cpu);
                assert_eq!(cpu.cpu.gpr[5] & 0xffff, frame);
                assert_eq!(cpu.cpu.gpr[0], 0x1234);
                assert!(cpu.cpu.flags.cf);
            }
        }
    }
}

#[test]
fn enter_dword_esp_frame_temp_edit_restarts_even_on_word_stack() {
    // SDM Vol 2A §3.3 ENTER: FrameTemp := ESP for dword operands,
    // even when the stack-address attribute selects SP for frame addresses.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        for level in [0u8, 1, 3] {
            let (mut vm, mut cpu) = setup(model, checked, &[0x66, 0xc8, 0, 0, level]);
            cpu.cpu.gpr[5] = 0x4004;
            trap(&mut vm, 0x7000);
            write_exit(&vm, &mut cpu, 0x7ffc, 4, 0x4004);
            cpu.complete_mmio_write();
            cpu.cpu.gpr[4] = 0x2222_8000;
            write_exit(&vm, &mut cpu, 0x7ffc, 4, 0x4004);
            cpu.complete_mmio_write();
            if level != 0 {
                for i in 2..=u64::from(level) + 1 {
                    write_exit(
                        &vm,
                        &mut cpu,
                        0x8000 - i * 4,
                        4,
                        if i == u64::from(level) + 1 {
                            0x2222_7ffc
                        } else {
                            0
                        },
                    );
                    cpu.complete_mmio_write();
                }
            }
            done(&vm, &mut cpu);
            assert_eq!(cpu.cpu.gpr[5], 0x2222_7ffc);
        }
    }
}

#[test]
fn enter_word_on_dword_stack_reads_ebp_for_display() {
    // SDM Vol 2A §3.3 ENTER: a word operand on a dword stack uses
    // EBP for display addressing, while the initial push still reads BP.
    let (mut vm, mut cpu) = setup(Some(CpuModel::I80386), true, b"\xc8\x00\x00\x02");
    cpu.cpu.legacy.segments[SegmentId::Ss as usize].attributes |= 0x4000;
    cpu.cpu.legacy.segments[SegmentId::Ss as usize].limit = 0x3ffff;
    cpu.cpu.gpr[5] = 0x4004;
    vm.write_bytes(0x4002, &0x1111u16.to_le_bytes()).unwrap();
    vm.write_bytes(0x14002, &0x2222u16.to_le_bytes()).unwrap();
    trap(&mut vm, 0x7000);
    write_exit(&vm, &mut cpu, 0x7ffe, 2, 0x4004);
    cpu.complete_mmio_write();
    write_exit(&vm, &mut cpu, 0x7ffc, 2, 0x1111);
    cpu.cpu.gpr[5] = 0x14004;
    write_exit(&vm, &mut cpu, 0x7ffe, 2, 0x4004);
    cpu.complete_mmio_write();
    write_exit(&vm, &mut cpu, 0x7ffc, 2, 0x2222);
    cpu.complete_mmio_write();
    write_exit(&vm, &mut cpu, 0x7ffa, 2, 0x7ffe);
    cpu.complete_mmio_write();
    done(&vm, &mut cpu);
    assert_eq!(cpu.cpu.gpr[5], 0x17ffe);
}

#[test]
fn enter_effective_limit_and_allocation_edits_restart_attempt() {
    // SDM Vol 2A §3.3 ENTER; Vol 3A §3.4.5.1: ENTER reads the
    // allocation and the retained stack limit even if this frame fits both.
    for edit_alloc in [false, true] {
        let (mut vm, mut cpu) = setup(Some(CpuModel::I80386), true, b"\xc8\x00\x00\x01");
        cpu.cpu.gpr[5] = 0x4004;
        trap(&mut vm, 0x7000);
        write_exit(&vm, &mut cpu, 0x7ffe, 2, 0x4004);
        cpu.complete_mmio_write();
        write_exit(&vm, &mut cpu, 0x7ffc, 2, 0x7ffe);
        if edit_alloc {
            vm.unmap(0x2000, 0x1000).unwrap();
            vm.map(0x2000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x2000, b"\xc8\x02\x00\x01").unwrap();
            // The decoded instruction survives device retries; a DS-only
            // context change makes the dispatcher decode the replacement.
            cpu.cpu.legacy.segments[SegmentId::Ds as usize].attributes ^= 1;
        } else {
            cpu.cpu.legacy.segments[SegmentId::Ss as usize].limit = 0xfffe;
        }
        write_exit(&vm, &mut cpu, 0x7ffe, 2, 0x4004);
        cpu.complete_mmio_write();
        write_exit(&vm, &mut cpu, 0x7ffc, 2, 0x7ffe);
        cpu.complete_mmio_write();
        done(&vm, &mut cpu);
        assert_eq!(cpu.cpu.gpr[4], if edit_alloc { 0x7ffa } else { 0x7ffc });
    }
}

#[test]
fn helper_faults_project_completed_reads_in_each_walk() {
    // SDM Vol 2A §3.3 BOUND/CALL/IRET/LDS; Vol 2B §4.3 POPA/RET:
    // these helpers read multiple operands before committing registers.
    for (model, checked) in [
        (None, false),
        (Some(CpuModel::I80386), false),
        (Some(CpuModel::I80386), true),
    ] {
        for code in [
            &b"\xc5\x06\xfe\x0f"[..], // LDS
            &b"\xff\x1e\xfe\x0f"[..], // far CALL
            &b"\xcb"[..],             // far RET
            &b"\xcf"[..],             // IRET
            &b"\x61"[..],             // POPA
            &b"\x62\x06\xfe\x0f"[..], // BOUND
        ] {
            let (mut vm, mut cpu) = setup(model, checked, code);
            cpu.cpu.gpr[4] = 0xffe;
            trap(&mut vm, 0);
            vm.unmap(0x1000, 0x1000).unwrap();
            read(&vm, &mut cpu, 0xffe, 2, 0x1234);
            enter_read_fault(&vm, &mut cpu, &[(0xffe, 0x1234)]);
        }
    }
}
