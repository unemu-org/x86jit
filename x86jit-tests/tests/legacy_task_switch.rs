//! Hardware task-switch witnesses. Sources are pinned in oracles/intel:
//! 80286 PRM §§8.2–8.5, pp. 8-1–9, App. B pp. B-12–13/B-25/B-51/B-57–58;
//! 80386 PRM §§7.1–7.6, pp. 7-1–10, §17 pp. 17-41/17-76/17-86.
//! The cited PRM page images were checked, including the source conflicts below.
#[path = "support/descriptor_fixture.rs"]
mod fixture;
use fixture::*;
use x86jit_core::{CpuModel, ExceptionDelivery, Exit, StepOptions, StepStop, Vcpu, Vm};

const A: u16 = 24;
const B: u16 = 32;
const C: u16 = 40;
const GATE: u16 = 48;
const CODE: u16 = 8;
const DATA: u16 = 16;

fn put(vm: &Vm, at: u64, size: usize, value: u32) {
    vm.write_bytes(at, &value.to_le_bytes()[..size]).unwrap();
}
fn far(call: bool, selector: u16) -> Vec<u8> {
    let mut bytes = vec![if call { 0x9a } else { 0xea }, 0, 0];
    bytes.extend(selector.to_le_bytes());
    bytes
}
fn task(vm: &Vm, sel: u16, wide: bool, ip: u32, stack: u32) {
    let base = 0x4000 + u64::from(sel - A) * 32;
    descriptor(
        vm,
        0x2000 + u64::from(sel),
        base as u32,
        if wide { 0x67 } else { 0x2b },
        if wide { 0x89 } else { 0x81 },
    );
    if wide {
        for (offset, size, val) in [
            (0x20, 4, ip),
            (0x24, 4, 2),
            (0x38, 4, stack),
            (0x48, 2, DATA.into()),
            (0x4c, 2, CODE.into()),
            (0x50, 2, DATA.into()),
            (0x54, 2, DATA.into()),
            (0x66, 2, 0x68),
        ] {
            put(vm, base + offset, size, val);
        }
    } else {
        for (offset, val) in [
            (0x0e, ip),
            (0x10, 2),
            (0x1a, stack),
            (0x22, DATA.into()),
            (0x24, CODE.into()),
            (0x26, DATA.into()),
            (0x28, DATA.into()),
        ] {
            put(vm, base + offset, 2, val);
        }
    }
}
fn gate(vm: &Vm, at: u64, target: u16, attr: u8) {
    vm.write_bytes(
        at,
        &[0, 0, target as u8, (target >> 8) as u8, 0, attr, 0, 0],
    )
    .unwrap();
}
fn setup_tasks(model: CpuModel, wide: bool, instruction: &[u8]) -> (Vm, Vcpu) {
    setup_tasks_delivery(model, wide, instruction, ExceptionDelivery::Outward)
}
fn setup_tasks_delivery(
    model: CpuModel,
    wide: bool,
    instruction: &[u8],
    delivery: ExceptionDelivery,
) -> (Vm, Vcpu) {
    let mut code = vec![0xb8, A as u8, 0, 0x0f, 0x00, 0xd8]; // MOV AX,A; LTR AX.
    code.extend(instruction);
    let (vm, mut cpu) = if delivery == ExceptionDelivery::Outward {
        setup(model, &code)
    } else {
        setup_delivery(model, &code, delivery)
    };
    descriptor(&vm, 0x2008, 0, 0xffff, 0x9b);
    descriptor(&vm, 0x2010, 0, 0xffff, 0x93);
    cpu.cpu.cs = CODE;
    cpu.cpu.ss = DATA;
    cpu.cpu.ds = DATA;
    cpu.cpu.es = DATA;
    task(&vm, A, wide, 0, 0x8000);
    task(&vm, B, wide, 0x1800, 0x9000);
    task(&vm, C, wide, 0x1900, 0xa000);
    gate(&vm, 0x2030, B, 0x85);
    steps(&mut cpu, &vm, 2);
    (vm, cpu)
}
fn base(sel: u16) -> u64 {
    0x4000 + u64::from(sel - A) * 32
}
fn busy(vm: &Vm, sel: u16) -> bool {
    value(vm, 0x2005 + u64::from(sel), 1) & 2 != 0
}
fn formats() -> [(CpuModel, bool); 3] {
    [(MODELS[0], false), (MODELS[1], false), (MODELS[1], true)]
}
fn outward(cpu: &mut Vcpu, vm: &Vm, vector: u8, code: u32) {
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0, "{:?}", run.stop());
    let StepStop::Exit(exit @ Exit::Exception { vector: actual, .. }) = run.stop() else {
        panic!("{:?}", run.stop())
    };
    assert_eq!(*actual, vector);
    assert_eq!(
        exit.fault_metadata().unwrap().error_code(),
        x86jit_core::fault::ErrorCode::bits(code, 0xffff)
    );
    assert_eq!(exit.fault_metadata().unwrap().saved_rip(), cpu.cpu.rip);
}

fn replace_page(vm: &mut Vm, at: u64, trap: bool) {
    let mut image = vec![0; 0x240000];
    vm.read_bytes(0, &mut image).unwrap();
    vm.unmap(0, 0x240000).unwrap();
    vm.map(
        0,
        at as usize,
        x86jit_core::Prot::RWX,
        x86jit_core::RegionKind::Ram,
    )
    .unwrap();
    vm.write_bytes(0, &image[..at as usize]).unwrap();
    if trap {
        vm.map(
            at,
            0x1000,
            x86jit_core::Prot::RW,
            x86jit_core::RegionKind::Trap,
        )
        .unwrap();
    }
    vm.map(
        at + 0x1000,
        (0x240000 - at - 0x1000) as usize,
        x86jit_core::Prot::RWX,
        x86jit_core::RegionKind::Ram,
    )
    .unwrap();
    vm.write_bytes(at + 0x1000, &image[at as usize + 0x1000..])
        .unwrap();
}

#[test]
fn jmp_to_tss_and_back_saves_state_and_sets_ts() {
    for (model, wide) in formats() {
        let (vm, mut cpu) = setup_tasks(model, wide, &far(false, B));
        vm.write_bytes(0x1800, &far(false, A)).unwrap();
        cpu.cpu.gpr[0] = 0x1234;
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1800));
        assert!(!busy(&vm, A) && busy(&vm, B));
        assert_ne!(cpu.cpu.legacy.cr0 & 8, 0);
        cpu.cpu.gpr[0] = 0x5678;
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.legacy.tr.selector, cpu.cpu.rip, cpu.cpu.gpr[0]),
            (A, 0x100b, 0x1234)
        );
        assert!(busy(&vm, A) && !busy(&vm, B));
        assert_eq!(
            value(
                &vm,
                base(B) + if wide { 0x28 } else { 0x12 },
                if wide { 4 } else { 2 }
            ),
            0x5678
        );
    }
}

#[test]
fn call_and_nested_iret_preserve_backlink_and_busy_chain() {
    for (model, wide) in formats() {
        let (vm, mut cpu) = setup_tasks(model, wide, &far(true, B));
        vm.write_bytes(0x1800, b"\xb8\x78\x56\xcf").unwrap();
        cpu.cpu.gpr[0] = 0x1234;
        steps(&mut cpu, &vm, 1);
        assert_eq!(value(&vm, base(B), 2), u64::from(A));
        assert!(busy(&vm, A) && busy(&vm, B));
        assert_ne!(cpu.cpu.rflags_system & 0x4000, 0);
        steps(&mut cpu, &vm, 2);
        assert_eq!(
            (cpu.cpu.legacy.tr.selector, cpu.cpu.rip, cpu.cpu.gpr[0]),
            (A, 0x100b, 0x1234)
        );
        assert!(busy(&vm, A) && !busy(&vm, B));
        assert_eq!(
            value(
                &vm,
                base(B) + if wide { 0x24 } else { 0x10 },
                if wide { 4 } else { 2 }
            ) & 0x4000,
            0
        );
        assert_eq!(
            value(
                &vm,
                base(B) + if wide { 0x20 } else { 0x0e },
                if wide { 4 } else { 2 }
            ),
            0x1804
        );
    }
}

#[test]
fn call_task_gate_in_gdt_and_ldt() {
    for (model, wide) in formats() {
        for ldt in [false, true] {
            let (vm, mut cpu) = setup_tasks(model, wide, &far(true, if ldt { 4 } else { GATE }));
            if ldt {
                descriptor(&vm, 0x2038, 0x3000, 7, 0x82);
                cpu.cpu.legacy.ldtr = x86jit_core::legacy::SystemSegment {
                    selector: 56,
                    cache: x86jit_core::legacy::SegmentCache {
                        base: 0x3000,
                        limit: 7,
                        attributes: 0x82,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                gate(&vm, 0x3000, B | 3, 0x85);
            }
            steps(&mut cpu, &vm, 1);
            assert_eq!((cpu.cpu.legacy.tr.selector & !3, cpu.cpu.rip), (B, 0x1800));
            assert_eq!(value(&vm, base(B), 2), u64::from(A));
        }
    }
}

#[test]
fn pre_switch_faults_keep_the_outgoing_task() {
    for (model, wide) in formats() {
        for (call, attr, vector) in [
            (false, if wide { 0x8b } else { 0x83 }, 13),
            (true, if wide { 9 } else { 1 }, 11),
        ] {
            let (vm, mut cpu) = setup_tasks(model, wide, &far(call, B));
            put(&vm, 0x2025, 1, attr);
            let before = format!("{:?}", cpu.cpu);
            fault(&mut cpu, &vm, vector, Some(B.into()));
            assert_eq!(before, format!("{:?}", cpu.cpu));
            assert_eq!(
                value(
                    &vm,
                    base(A) + if wide { 0x20 } else { 0x0e },
                    if wide { 4 } else { 2 }
                ),
                0
            );
        }
    }
}

#[test]
fn iret_backlink_must_be_busy() {
    for (model, wide) in formats() {
        let (vm, mut cpu) = setup_tasks(model, wide, &far(true, B));
        vm.write_bytes(0x1800, b"\xcf").unwrap();
        steps(&mut cpu, &vm, 1);
        put(&vm, 0x201d, 1, if wide { 0x89 } else { 0x81 });
        let before = format!("{:?}", cpu.cpu);
        outward(&mut cpu, &vm, 10, A.into());
        assert_eq!(before, format!("{:?}", cpu.cpu));
        assert!(busy(&vm, B));
    }
}

#[test]
fn short_tss_fault_placement_is_model_specific() {
    // 286 App. B SWITCH_TASKS p. B-12 commits TR before checking 2Bh.
    // 386 §7.5 p. 7-8 checks 2Bh/67h before saving the old task.
    for (model, wide) in formats() {
        let (vm, mut cpu) = setup_tasks(model, wide, &far(true, B));
        put(&vm, 0x2020, 2, if wide { 0x66 } else { 0x2a });
        let before = cpu.cpu.clone();
        outward(&mut cpu, &vm, 10, B.into());
        if model == CpuModel::I80286 {
            assert_eq!(cpu.cpu.legacy.tr.selector, B);
            assert!(busy(&vm, A) && busy(&vm, B));
            assert_eq!(value(&vm, base(A) + 0x0e, 2), 0x100b);
            assert_eq!(value(&vm, base(B), 2), u64::from(A));
        } else {
            assert_eq!(format!("{:?}", before), format!("{:?}", cpu.cpu));
            assert!(!busy(&vm, B));
            assert_eq!(
                value(
                    &vm,
                    base(A) + if wide { 0x20 } else { 0x0e },
                    if wide { 4 } else { 2 }
                ),
                0
            );
        }
    }
}

#[test]
fn descriptor_privilege_presence_and_task_gate_target_checks() {
    for (model, wide) in formats() {
        for call in [false, true] {
            let invalid = if call && model == CpuModel::I80386 {
                10
            } else {
                13
            };
            for sel in [B | 3, GATE | 3] {
                let (vm, mut cpu) = setup_tasks(model, wide, &far(call, sel));
                outward(&mut cpu, &vm, 13, u32::from(sel & !3));
                assert_eq!(cpu.cpu.legacy.tr.selector, A);
                assert!(!busy(&vm, B));
            }
            for (target, attr, vector, error) in [
                (B, 5, 11, GATE),
                (B | 4, 0x85, invalid, B | 4),
                (0, 0x85, invalid, 0),
                (0x78, 0x85, invalid, 0x78),
            ] {
                let (vm, mut cpu) = setup_tasks(model, wide, &far(call, GATE));
                gate(&vm, 0x2030, target, attr);
                if target == 0x78 {
                    cpu.cpu.legacy.gdtr.limit = 0x77;
                }
                outward(&mut cpu, &vm, vector, error.into());
                assert_eq!(cpu.cpu.legacy.tr.selector, A);
            }
            // DPL=3 gate permits RPL=3 while the target TSS remains DPL=0.
            let (vm, mut cpu) = setup_tasks(model, wide, &far(call, GATE | 3));
            gate(&vm, 0x2030, B | 3, 0xe5);
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.legacy.tr.selector, B | 3);
        }
    }
}

#[test]
fn iret_rejects_null_local_wrong_type_absent_and_outside_backlinks() {
    for (model, wide) in formats() {
        for (link, attr, vector, error) in [
            (0, 0x83, 10, 0),
            (B | 4, 0x83, 10, B | 4),
            (C, 0x93, 10, C),
            (C, if wide { 11 } else { 3 }, 11, C),
            (0x78, 0x83, 10, 0x78),
        ] {
            let (vm, mut cpu) = setup_tasks(model, wide, &far(true, B));
            vm.write_bytes(0x1800, b"\xcf").unwrap();
            steps(&mut cpu, &vm, 1);
            put(&vm, base(B), 2, link.into());
            put(&vm, 0x202d, 1, attr);
            if link == 0x78 {
                cpu.cpu.legacy.gdtr.limit = 0x77;
            }
            let before = format!("{:?}", cpu.cpu);
            outward(&mut cpu, &vm, vector, error.into());
            assert_eq!(before, format!("{:?}", cpu.cpu));
        }
    }
}

#[test]
fn incoming_nt_on_jmp_comes_from_the_tss() {
    // 286 PRM App. B SWITCH_TASKS p. B-12; 386 PRM §7.5 step 5 p. 7-8;
    // SDM Vol 3A §10.3 step 12 p. 10-11 / §10.4 Table 10-2 p. 10-15:
    // JMP restores incoming NT; outgoing NT and both back links are unchanged.
    for (model, wide) in formats() {
        for incoming_nt in [0, 0x4000] {
            for outgoing_nt in [0u32, 0x4000] {
                for through_gate in [false, true] {
                    let target = if through_gate { GATE } else { B };
                    let (vm, mut cpu) = setup_tasks(model, wide, &far(false, target));
                    let flags_offset = if wide { 0x24 } else { 0x10 };
                    let size = if wide { 4 } else { 2 };
                    put(&vm, base(B) + flags_offset, size, 0x203 | incoming_nt);
                    put(&vm, base(A), 2, C.into());
                    put(&vm, base(B), 2, A.into());
                    cpu.cpu.rflags_system = outgoing_nt.into();
                    steps(&mut cpu, &vm, 1);
                    assert_eq!(cpu.cpu.rflags(), u64::from(0x203 | incoming_nt));
                    assert_eq!(
                        value(&vm, base(A) + flags_offset, size) & 0x4000,
                        u64::from(outgoing_nt)
                    );
                    assert_eq!(value(&vm, base(A), 2), u64::from(C));
                    assert_eq!(value(&vm, base(B), 2), u64::from(A));
                    assert!(!busy(&vm, A) && busy(&vm, B));
                    assert_eq!(cpu.cpu.legacy.tr.selector, B);
                }
            }
        }
    }
}

fn nested_jmp_ping_pong(model: CpuModel, wide: bool) {
    // 286 PRM App. B SWITCH_TASKS p. B-12; 386 PRM §7.5 step 5 p. 7-8;
    // SDM Vol 3A §10.3 step 12 p. 10-11 / §10.4 Table 10-2 p. 10-15.
    // TSS fields: 286 PRM §8.2 Fig. 8-1 pp. 8-2–3;
    // 386 PRM §7.1 Fig. 7-1 pp. 7-1–3. IRET clears only departing NT.
    use x86jit_core::legacy::{SegmentCache, SystemSegment};

    const ROUNDS: u32 = 4;
    fn set_image_field(image: &mut [u8], offset: usize, size: usize, value: u32) {
        image[offset..offset + size].copy_from_slice(&value.to_le_bytes()[..size]);
    }
    fn check(
        cpu: &Vcpu,
        vm: &Vm,
        wide: bool,
        current: u16,
        ip: u32,
        flags: u32,
        registers: [u32; 8],
    ) {
        let mut expected = [0; 16];
        expected[..8].copy_from_slice(&registers.map(u64::from));
        assert_eq!(cpu.cpu.gpr, expected);
        assert_eq!(cpu.cpu.rip, u64::from(ip));
        assert_eq!(cpu.cpu.rflags(), u64::from(flags));
        assert_eq!(
            [cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs],
            [CODE, DATA, DATA, DATA, 0, 0]
        );
        assert_eq!(cpu.cpu.legacy.ldtr.selector, 0);
        assert_eq!(cpu.cpu.legacy.cr0, 9);
        assert_eq!(
            cpu.cpu.legacy.tr,
            SystemSegment {
                selector: current,
                cache: SegmentCache {
                    base: base(current) as u32,
                    limit: if wide { 0x67 } else { 0x2b },
                    attributes: if wide { 0x8b } else { 0x83 },
                    ..Default::default()
                },
                ..Default::default()
            }
        );
        for (selector, expected_busy, backlink) in
            [(A, true, C), (B, current == B, A), (C, current == C, B)]
        {
            assert_eq!(busy(vm, selector), expected_busy, "selector {selector}");
            assert_eq!(value(vm, base(selector), 2), u64::from(backlink));
        }
    }

    // Also return to a caller whose saved NT is set: IRET restores that image.
    for caller_nt in [0, 0x4000] {
        let (vm, mut cpu) = setup_tasks(model, wide, &far(true, GATE));
        let size = if wide { 4 } else { 2 };
        let ip_offset = if wide { 0x20 } else { 0x0e };
        let flags_offset = ip_offset + size;
        let mut registers = [[0; 8]; 3];
        let initial_flags = [0x247 | caller_nt, 0x46, 0x803];
        let mut images = Vec::new();
        for (index, selector) in [A, B, C].into_iter().enumerate() {
            for (reg, value) in registers[index].iter_mut().enumerate() {
                *value = 0x1111 * (index as u32 + 1) + reg as u32 * 0x100;
                if wide {
                    *value |= (0x1234 + index as u32) << 16;
                }
                if reg == 4 {
                    *value = 0x8000 + index as u32 * 0x1000;
                }
                put(
                    &vm,
                    base(selector) + (ip_offset + 2 * size + reg * size) as u64,
                    size,
                    *value,
                );
            }
            put(
                &vm,
                base(selector) + flags_offset as u64,
                size,
                initial_flags[index],
            );
            put(&vm, base(selector), 2, [C, C, B][index].into());
            let mut image = vec![0; if wide { 0x68 } else { 0x2c }];
            vm.read_bytes(base(selector), &mut image).unwrap();
            images.push(image);
        }
        cpu.cpu.gpr[..8].copy_from_slice(&registers[0].map(u64::from));
        cpu.cpu.flags.set_flags16(initial_flags[0] as u16);
        cpu.cpu.rflags_system = caller_nt.into();
        vm.write_bytes(0x1800, &far(false, C).repeat(ROUNDS as usize))
            .unwrap();
        vm.write_bytes(0x1800 + u64::from(5 * ROUNDS), b"\xcf")
            .unwrap();
        vm.write_bytes(0x1900, &far(false, B).repeat(ROUNDS as usize))
            .unwrap();
        // A stack IRET over this fill cannot masquerade as a task return.
        vm.write_bytes(0x9000, &[0xa5; 6]).unwrap();
        let check_images = |images: &[Vec<u8>]| {
            for (selector, expected) in [A, B, C].into_iter().zip(images) {
                let mut actual = vec![0; expected.len()];
                vm.read_bytes(base(selector), &mut actual).unwrap();
                assert_eq!(&actual, expected, "TSS selector {selector}");
            }
        };

        steps(&mut cpu, &vm, 1); // CALL A -> B through the task gate.
        set_image_field(&mut images[0], ip_offset, size, 0x100b);
        set_image_field(&mut images[1], 0, 2, A.into());
        check(&cpu, &vm, wide, B, 0x1800, 0x4046, registers[1]);
        check_images(&images);
        for round in 1..=ROUNDS {
            steps(&mut cpu, &vm, 1); // JMP B -> C saves NT=1 in B.
            set_image_field(&mut images[1], ip_offset, size, 0x1800 + 5 * round);
            set_image_field(&mut images[1], flags_offset, size, 0x4046);
            check(
                &cpu,
                &vm,
                wide,
                C,
                0x1900 + 5 * (round - 1),
                0x803,
                registers[2],
            );
            check_images(&images);
            steps(&mut cpu, &vm, 1); // JMP C -> B restores NT=1.
            set_image_field(&mut images[2], ip_offset, size, 0x1900 + 5 * round);
            check(&cpu, &vm, wide, B, 0x1800 + 5 * round, 0x4046, registers[1]);
            check_images(&images);
        }
        steps(&mut cpu, &vm, 1); // IRET B -> A through B's unchanged back link.
        set_image_field(&mut images[1], ip_offset, size, 0x1800 + 5 * ROUNDS + 1);
        set_image_field(&mut images[1], flags_offset, size, 0x46);
        check(&cpu, &vm, wide, A, 0x100b, initial_flags[0], registers[0]);
        check_images(&images);
    }
}

#[test]
fn i286_nested_jmp_ping_pong_returns_through_backlink() {
    nested_jmp_ping_pong(CpuModel::I80286, false);
}

#[test]
fn i386_286_tss_nested_jmp_ping_pong_returns_through_backlink() {
    nested_jmp_ping_pong(CpuModel::I80386, false);
}

#[test]
fn i386_386_tss_nested_jmp_ping_pong_returns_through_backlink() {
    nested_jmp_ping_pong(CpuModel::I80386, true);
}

#[test]
fn incoming_ldtr_and_every_segment_cache_are_reloaded() {
    for (model, wide) in formats() {
        let (vm, mut cpu) = setup_tasks(model, wide, &far(false, B));
        descriptor(&vm, 0x2038, 0x6000, 0x3f, 0x82);
        descriptor(&vm, 0x6000, 0x10000, 0xffff, 0x9a);
        descriptor(&vm, 0x6008, 0x14000, 0xafff, 0x92);
        put(&vm, base(B) + if wide { 0x60 } else { 0x2a }, 2, 56);
        for (off, val) in if wide {
            vec![
                (0x48, 12),
                (0x4c, 4),
                (0x50, 12),
                (0x54, 12),
                (0x58, 12),
                (0x5c, 12),
            ]
        } else {
            vec![(0x22, 12), (0x24, 4), (0x26, 12), (0x28, 12)]
        } {
            put(&vm, base(B) + off, 2, val);
        }
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (
                cpu.cpu.legacy.tr.selector,
                cpu.cpu.legacy.tr.cache.base,
                cpu.cpu.legacy.tr.cache.limit
            ),
            (B, base(B) as u32, if wide { 0x67 } else { 0x2b })
        );
        assert_eq!(
            (cpu.cpu.legacy.ldtr.selector, cpu.cpu.legacy.ldtr.cache.base),
            (56, 0x6000)
        );
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.ss, cpu.cpu.ds, cpu.cpu.es),
            (4, 12, 12, 12)
        );
        assert_eq!(cpu.cpu.legacy.segments[0].base, 0x10000);
        for index in [1, 2, 3] {
            assert_eq!(
                (
                    cpu.cpu.legacy.segments[index].base,
                    cpu.cpu.legacy.segments[index].limit,
                    cpu.cpu.legacy.segments[index].attributes
                ),
                (0x14000, 0xafff, 0x93)
            );
        }
        if wide {
            assert_eq!((cpu.cpu.fs, cpu.cpu.gs), (12, 12));
            for index in [4, 5] {
                assert_eq!(cpu.cpu.legacy.segments[index].base, 0x14000);
            }
        } else {
            assert_eq!((cpu.cpu.fs, cpu.cpu.gs), (0, 0));
        }
        assert_eq!(value(&vm, 0x6005, 1), 0x9b);
        assert_eq!(value(&vm, 0x600d, 1), 0x93);
    }
}

#[test]
fn segment_fault_stays_in_the_new_task_outward_and_in_guest() {
    for (model, wide) in formats() {
        for delivery in [ExceptionDelivery::Outward, ExceptionDelivery::InGuest] {
            let (vm, mut cpu) = setup_tasks_delivery(model, wide, &far(true, B), delivery);
            descriptor(&vm, 0x2040, 0, 0xffff, 0x12); // readable data, P=0
            put(&vm, base(B) + if wide { 0x54 } else { 0x28 }, 2, 64);
            cpu.cpu.legacy.idtr.base = 0x3000;
            cpu.cpu.legacy.idtr.limit = 0x7ff;
            vm.write_bytes(0x3058, &[0, 0x1a, 8, 0, 0, 0x86, 0, 0])
                .unwrap();
            vm.write_bytes(0x1a00, b"\x90").unwrap();
            if delivery == ExceptionDelivery::Outward {
                outward(&mut cpu, &vm, 11, 64);
                assert_eq!(cpu.cpu.rip, 0x1800);
                assert_eq!(cpu.cpu.ds, 64);
                assert_ne!(cpu.cpu.legacy.segments[1].unusable, 0);
            } else {
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.rip, 0x1a01);
                assert_eq!(cpu.cpu.gpr[4], 0x8ff8);
                assert_eq!(value(&vm, 0x8ff8, 2), 64);
                assert_eq!(value(&vm, 0x8ffa, 2), 0x1800);
                assert_eq!(value(&vm, 0x8ffc, 2), u64::from(CODE));
            }
            assert_eq!(cpu.cpu.legacy.tr.selector, B);
            assert_eq!(
                value(
                    &vm,
                    base(A) + if wide { 0x20 } else { 0x0e },
                    if wide { 4 } else { 2 }
                ),
                0x100b
            );
            assert!(busy(&vm, A) && busy(&vm, B));
        }
    }
}

#[test]
fn exception_task_gate_pushes_error_on_new_stack_and_returns_by_link() {
    // 286 PRM §9.4 p. 9-4; 386 PRM §9.6 p. 9-8: task gates nest,
    // save fault IP in old TSS, and push only the error code on the new stack.
    for (model, wide) in formats() {
        let (vm, mut cpu) = setup_tasks_delivery(
            model,
            wide,
            b"\xb8\x78\x00\x8e\xd8",
            ExceptionDelivery::InGuest,
        );
        cpu.cpu.legacy.idtr.base = 0x3000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        gate(&vm, 0x3000 + 13 * 8, B | 3, 0x85);
        vm.write_bytes(0x1800, b"\x58\xcf").unwrap();
        steps(&mut cpu, &vm, 1); // MOV AX,invalid DS selector
        steps(&mut cpu, &vm, 1); // #GP -> task B, POP AX (16-bit code)
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B | 3, 0x1801));
        assert_eq!(cpu.cpu.gpr[0], 0x78);
        assert_eq!(
            value(
                &vm,
                0x9000 - if wide { 4 } else { 2 },
                if wide { 4 } else { 2 }
            ),
            0x78
        );
        assert_eq!(
            value(
                &vm,
                base(A) + if wide { 0x20 } else { 0x0e },
                if wide { 4 } else { 2 }
            ),
            0x1009
        );
        assert_eq!(value(&vm, base(B), 2), u64::from(A));
        assert_ne!(cpu.cpu.rflags_system & 0x4000, 0);
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (A, 0x1009));
        assert!(!busy(&vm, B));
    }
}

#[test]
fn software_and_external_idt_task_gates_nest() {
    for (model, wide) in formats() {
        for external in [false, true] {
            let (vm, mut cpu) =
                setup_tasks(model, wide, if external { b"\x90" } else { b"\xcd\x30" });
            cpu.cpu.legacy.idtr.base = 0x3000;
            cpu.cpu.legacy.idtr.limit = 0x7ff;
            gate(&vm, 0x3180, B, 0x85);
            vm.write_bytes(0x1800, b"\x90\xcf").unwrap();
            if external {
                cpu.cpu.flags.if_ = true;
                cpu.inject_irq(0x30);
            }
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.legacy.tr.selector, B);
            assert_eq!(cpu.cpu.rip, if external { 0x1801 } else { 0x1800 });
            assert_eq!(
                value(
                    &vm,
                    base(A) + if wide { 0x20 } else { 0x0e },
                    if wide { 4 } else { 2 }
                ),
                if external { 0x1006 } else { 0x1008 }
            );
            assert_ne!(cpu.cpu.rflags_system & 0x4000, 0);
            assert_eq!(cpu.cpu.gpr[4], 0x9000);
        }
    }
}

#[test]
fn full_386_registers_and_static_tss_fields_survive() {
    let (vm, mut cpu) = setup_tasks(CpuModel::I80386, true, &far(false, B));
    vm.write_bytes(0x1800, &far(false, A)).unwrap();
    let old = [
        0xa1234567, 0xb2345678, 0xc3456789, 0xd456789a, 0x8000, 0xe56789ab, 0xf6789abc, 0x9789abcd,
    ];
    let new = [
        0x1234abcd, 0x2345bcde, 0x3456cdef, 0x4567def0, 0x9000, 0x5678ef01, 0x6789f012, 0x789a0123,
    ];
    cpu.cpu.gpr[..8].copy_from_slice(&old);
    cpu.cpu.legacy.cr3 = 0x12345000;
    cpu.cpu.fs = DATA;
    cpu.cpu.gs = DATA;
    for (i, v) in new.iter().enumerate() {
        put(&vm, base(B) + 0x28 + i as u64 * 4, 4, *v as u32);
    }
    put(&vm, base(A) + 0x1c, 4, 0xabcd0000);
    put(&vm, base(B) + 0x1c, 4, 0x98765000);
    put(&vm, base(A) + 0x60, 4, 0xaabb0000);
    put(&vm, base(A) + 0x64, 4, 0x77660000);
    for offset in [4, 8, 12, 16, 20, 24] {
        put(&vm, base(A) + offset, 4, 0xaabbccdd);
    }
    for offset in [0x4a, 0x4e, 0x52, 0x56, 0x5a, 0x5e] {
        put(&vm, base(A) + offset, 2, 0xcafe);
    }
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[..8], new);
    assert_eq!(cpu.cpu.legacy.cr3, 0x12345000);
    assert_eq!(cpu.cpu.task_io_map_base, 0x68);
    for (i, v) in old.iter().enumerate() {
        assert_eq!(value(&vm, base(A) + 0x28 + i as u64 * 4, 4), *v);
    }
    for offset in [4, 8, 12, 16, 20, 24] {
        assert_eq!(value(&vm, base(A) + offset, 4), 0xaabbccdd);
    }
    for offset in [0x4a, 0x4e, 0x52, 0x56, 0x5a, 0x5e] {
        assert_eq!(value(&vm, base(A) + offset, 2), 0xcafe);
    }
    assert_eq!(value(&vm, base(A) + 0x1c, 4), 0xabcd0000);
    assert_eq!(value(&vm, base(A) + 0x60, 4), 0xaabb0000);
    assert_eq!(value(&vm, base(A) + 0x64, 4), 0x77660000);
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[..8], old);
    assert_eq!((cpu.cpu.fs, cpu.cpu.gs), (DATA, DATA));
}

#[test]
fn typed_refusals_for_v86_privilege_and_task_debug_trap() {
    for (model, wide) in formats() {
        for privilege in [false, true] {
            if !wide && !privilege {
                continue;
            }
            let (vm, mut cpu) = setup_tasks(model, wide, &far(false, B));
            if privilege {
                put(&vm, base(B) + if wide { 0x4c } else { 0x24 }, 2, 11);
            } else {
                put(&vm, base(B) + 0x24, 4, 0x20002);
            }
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            let StepStop::Exit(Exit::UnsupportedSystemContext { operation, .. }) = run.stop()
            else {
                panic!("{:?}", run.stop())
            };
            assert_eq!(
                *operation,
                if privilege {
                    x86jit_core::UnsupportedSystemOperation::LegacyPrivilegeTransition
                } else {
                    x86jit_core::UnsupportedSystemOperation::InvalidLegacyContext(
                        x86jit_core::legacy::MachineContextError::Virtual8086Unsupported,
                    )
                }
            );
            // Incoming state is read only after acquiring B and saving A.
            // Unsupported incoming execution is reported at that committed boundary.
            assert_eq!(
                (cpu.cpu.legacy.tr.selector, cpu.cpu.rip, cpu.cpu.legacy.cr0),
                (B, 0x1800, 9)
            );
            assert!(!busy(&vm, A) && busy(&vm, B));
            assert_eq!(
                value(
                    &vm,
                    base(A) + if wide { 0x20 } else { 0x0e },
                    if wide { 4 } else { 2 }
                ),
                0x100b
            );
        }
    }
    let (vm, mut cpu) = setup_tasks(CpuModel::I80386, true, &far(true, B));
    put(&vm, base(B) + 0x64, 2, 1);
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 1);
    assert!(matches!(
        run.stop(),
        StepStop::Exit(Exit::UnsupportedSystemContext {
            operation: x86jit_core::UnsupportedSystemOperation::LegacyTaskDebugTrap,
            ..
        })
    ));
    assert_eq!(
        (
            cpu.cpu.legacy.tr.selector,
            cpu.cpu.rip,
            cpu.cpu.task_debug_trap
        ),
        (B, 0x1800, 1)
    );
    assert_ne!(cpu.cpu.legacy.cr0 & 8, 0);
}

#[test]
fn a20_gates_tss_images_and_both_busy_updates() {
    for model in MODELS {
        let (vm, mut cpu) = setup_tasks(model, false, &far(false, B));
        // LTR used the ordinary GDT. The task switches use its high alias.
        cpu.cpu.legacy.gdtr.base = 0x102000;
        cpu.cpu.legacy.tr.cache.base = 0x104000;
        descriptor(&vm, 0x2020, 0x104100, 0x2b, 0x81);
        let mut high = [0; 0x80];
        vm.read_bytes(0x2000, &mut high).unwrap();
        vm.write_bytes(0x102000, &high).unwrap();
        vm.write_bytes(0x1800, &far(false, A)).unwrap();
        cpu.set_a20_enabled(false);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rip, 0x1800);
        assert!(!busy(&vm, A) && busy(&vm, B));
        assert_eq!(value(&vm, 0x10201d, 1), 0x83);
        assert_eq!(value(&vm, 0x102025, 1), 0x81);
        assert_eq!(value(&vm, base(A) + 0x0e, 2), 0x100b);
        assert_eq!(value(&vm, 0x10400e, 2), 0);
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (A, 0x100b));
    }
}

#[test]
fn three_286_tasks_gate_call_register_checks_iret_then_two_jumps() {
    for model in MODELS {
        let (vm, mut cpu) = setup_tasks(model, false, &far(true, GATE));
        let mut program = far(true, GATE);
        let mut failures = vec![];
        for (at, expected) in [(base(B) + 0x12, 0x2468u16), (base(B) + 0x0e, 0x1804u16)] {
            program.extend([0x81, 0x3e]);
            program.extend((at as u16).to_le_bytes());
            program.extend(expected.to_le_bytes());
            program.push(0x75);
            failures.push(program.len());
            program.push(0);
        }
        program.extend(far(false, C));
        program.push(0x43); // INC BX after the second JMP, restored task A.
        program.push(0xf4);
        let bad = program.len();
        program.extend([0x0f, 0xff]);
        for at in failures {
            program[at] = (bad - at - 1) as u8;
        }
        vm.write_bytes(0x1006, &program).unwrap();
        vm.write_bytes(0x1800, b"\xb8\x68\x24\xcf").unwrap();
        let mut third = vec![0xba, 0x57, 0x13];
        third.extend(far(false, A));
        vm.write_bytes(0x1900, &third).unwrap();
        cpu.cpu.gpr[0] = 0x1234;
        steps(&mut cpu, &vm, 11);
        assert_eq!(
            (cpu.cpu.legacy.tr.selector, cpu.cpu.gpr[0], cpu.cpu.gpr[3]),
            (A, 0x1234, 1)
        );
        assert_eq!(value(&vm, base(C) + 0x16, 2), 0x1357);
        assert!(busy(&vm, A) && !busy(&vm, B) && !busy(&vm, C));
    }
}

#[test]
fn post_commit_unmapped_segment_load_resumes_without_restarting_transfer() {
    for (model, wide) in formats() {
        let (mut vm, mut cpu) = setup_tasks(model, wide, &far(true, B));
        descriptor(&vm, 0x2038, 0x6000, 7, 0x82);
        put(&vm, base(B) + if wide { 0x60 } else { 0x2a }, 2, 56);
        put(&vm, base(B) + if wide { 0x54 } else { 0x28 }, 2, 4);
        replace_page(&mut vm, 0x6000, false);
        cpu.cpu.gpr[0] = 0x1234;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::UnmappedMemory { addr: 0x6000, .. })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1800));
        vm.map(
            0x6000,
            0x1000,
            x86jit_core::Prot::RWX,
            x86jit_core::RegionKind::Ram,
        )
        .unwrap();
        descriptor(&vm, 0x6000, 0x12000, 0xffff, 0x92);
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.legacy.tr.selector, cpu.cpu.rip, cpu.cpu.ds),
            (B, 0x1800, 4)
        );
        assert_eq!(cpu.cpu.legacy.segments[1].base, 0x12000);
        assert_eq!(
            value(
                &vm,
                base(A) + if wide { 0x28 } else { 0x12 },
                if wide { 4 } else { 2 }
            ),
            0x1234
        );
        assert!(busy(&vm, A) && busy(&vm, B));
    }
}

#[test]
fn task_save_mmio_handoffs_preserve_the_acquired_task_and_old_state() {
    for (model, wide) in formats() {
        let (mut vm, mut cpu) = setup_tasks(model, wide, &far(false, B));
        cpu.cpu.legacy.tr.cache.base = 0x7000;
        replace_page(&mut vm, 0x7000, true);
        cpu.cpu.gpr[0] = 0x1234;
        let old = cpu.cpu.clone();
        let mut writes = vec![];
        loop {
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            match run.stop() {
                StepStop::Exit(Exit::MmioWrite {
                    addr, size, value, ..
                }) => {
                    assert_eq!(run.steps(), 0);
                    assert_eq!(cpu.cpu.gpr, old.gpr);
                    assert_eq!(cpu.cpu.legacy.tr, old.legacy.tr);
                    assert_eq!(cpu.cpu.rip, old.rip);
                    assert!(busy(&vm, B));
                    writes.push((*addr, *size, *value));
                    cpu.complete_mmio_write();
                }
                StepStop::Exit(Exit::BudgetExhausted) => {
                    assert_eq!(run.steps(), 1);
                    break;
                }
                stop => panic!("{stop:?}"),
            }
            assert!(writes.len() <= 16);
        }
        assert_eq!(writes.len(), if wide { 16 } else { 14 });
        assert_eq!(
            writes[0],
            (
                0x7000 + if wide { 0x20 } else { 0x0e },
                if wide { 4 } else { 2 },
                0x100b
            )
        );
        assert!(writes
            .iter()
            .any(|&(at, _, v)| at == 0x7000 + if wide { 0x28 } else { 0x12 } && v == 0x1234));
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1800));
        assert!(!busy(&vm, A) && busy(&vm, B));
    }
}

#[test]
fn idt_task_gate_error_push_mmio_resumes_in_new_task_once() {
    for (model, wide) in formats() {
        let (mut vm, mut cpu) =
            setup_tasks_delivery(model, wide, b"\x8e\xd8", ExceptionDelivery::InGuest);
        cpu.cpu.gpr[0] = 0x78;
        cpu.cpu.legacy.idtr.base = 0x3000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        gate(&vm, 0x3068, B, 0x85);
        vm.write_bytes(0x1800, b"\x90").unwrap();
        replace_page(&mut vm, 0x8000, true);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(run.stop(),StepStop::Exit(Exit::MmioWrite{addr,size,value:0x78,..}) if *addr==0x9000-if wide {4}else{2} && *size==if wide {4}else{2}),
            "{:?}",
            run.stop()
        );
        assert_eq!(
            (cpu.cpu.legacy.tr.selector, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (B, 0x1800, 0x9000)
        );
        cpu.complete_mmio_write();
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.legacy.tr.selector, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (B, 0x1801, 0x9000 - if wide { 4 } else { 2 })
        );
    }
}

#[test]
fn task_segment_validation_vectors_codes_and_order_follow_each_prm() {
    // 286 SWITCH_TASKS pp. B-12–13; 386 §9.8.10 Table 9-5 p. 9-14
    // agree with SDM Table 10-1: #TS(selector) for validity/privilege.
    // The conflicting summary tables and model ordering are in decision-24.
    for (model, wide) in formats() {
        for (segment, selector, attr, vector) in [
            (0, 64, 0x92, 10), // CS is data
            (1, 64, 0x92, 10), // DS system? descriptor below overrides
            (3, 64, 0xb2, 10), // SS DPL=1
            (3, 64, 0x12, 12), // SS not present
        ] {
            let (vm, mut cpu) = setup_tasks(model, wide, &far(false, B));
            descriptor(
                &vm,
                0x2040,
                0,
                0xffff,
                if segment == 1 { 0x82 } else { attr },
            );
            let offset = if wide {
                [0x4c, 0x54, 0x48, 0x50][segment]
            } else {
                [0x24, 0x28, 0x22, 0x26][segment]
            };
            put(&vm, base(B) + offset, 2, selector);
            outward(&mut cpu, &vm, vector, selector);
            assert_eq!(cpu.cpu.legacy.tr.selector, B);
            assert_eq!(cpu.cpu.rip, 0x1800);
        }
        for (ldt, attr) in [(56, 0x02), (56, 0x92), (60, 0x82)] {
            let (vm, mut cpu) = setup_tasks(model, wide, &far(false, B));
            descriptor(&vm, 0x2038, 0x6000, 0x3f, attr);
            put(&vm, base(B) + if wide { 0x60 } else { 0x2a }, 2, ldt);
            outward(&mut cpu, &vm, 10, ldt);
            assert_eq!(cpu.cpu.legacy.tr.selector, B);
        }
        // With two invalid selectors, 286 validates SS before CS, 386 CS before SS.
        let (vm, mut cpu) = setup_tasks(model, wide, &far(false, B));
        put(&vm, base(B) + if wide { 0x4c } else { 0x24 }, 2, 64);
        put(&vm, base(B) + if wide { 0x50 } else { 0x26 }, 2, 72);
        outward(
            &mut cpu,
            &vm,
            10,
            if model == CpuModel::I80286 { 72 } else { 64 },
        );
    }
}

#[test]
fn target_ip_limit_fault_is_in_new_task_with_gp() {
    // 286 App. B CALL/JMP pp. B-25/B-57–58: #GP(0).
    // 386 Appendix E Table 8-1 note r p. E-15 agrees with SDM CALL:
    // #GP(0); §17 CALL p. 17-41's #TS(0) is the recorded conflicting text.
    for (model, wide) in formats() {
        for call in [false, true] {
            let (vm, mut cpu) = setup_tasks(model, wide, &far(call, B));
            descriptor(&vm, 0x2008, 0, 0x17ff, 0x9b);
            outward(&mut cpu, &vm, 13, 0);
            assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1800));
        }
    }
}

#[test]
fn i286_rejects_386_tss_descriptors_and_old_tr_cache_owns_saves() {
    let (vm, mut cpu) = setup_tasks(CpuModel::I80286, false, &far(false, B));
    descriptor(&vm, 0x2020, base(B) as u32, 0x67, 0x89);
    fault(&mut cpu, &vm, 13, Some(B.into()));
    for (model, wide) in formats() {
        let (vm, mut cpu) = setup_tasks(model, wide, &far(false, B));
        // Change memory without reloading TR. Saving uses its retained base;
        // clearing B updates the live descriptor's access byte only.
        descriptor(
            &vm,
            0x2018,
            0x7000,
            if wide { 0x67 } else { 0x2b },
            if wide { 0xeb } else { 0xe3 },
        );
        cpu.cpu.gpr[0] = 0x3141;
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            value(
                &vm,
                base(A) + if wide { 0x28 } else { 0x12 },
                if wide { 4 } else { 2 }
            ),
            0x3141
        );
        assert_eq!(
            value(
                &vm,
                0x7000 + if wide { 0x28 } else { 0x12 },
                if wide { 4 } else { 2 }
            ),
            0
        );
        assert_eq!(value(&vm, 0x201d, 1), if wide { 0xe9 } else { 0xe1 });
    }
}

#[test]
fn fault_in_interrupt_task_load_uses_new_context_without_old_double_fault_pair() {
    for (model, wide) in formats() {
        let (vm, mut cpu) =
            setup_tasks_delivery(model, wide, b"\x8e\xd8", ExceptionDelivery::InGuest);
        cpu.cpu.gpr[0] = 0x78;
        cpu.cpu.legacy.idtr.base = 0x3000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        gate(&vm, 0x3068, B, 0x85); // original #GP -> B
        gate(&vm, 0x3058, C, 0x85); // #NP loading B.DS -> C
        descriptor(&vm, 0x2040, 0, 0xffff, 0x12);
        put(&vm, base(B) + if wide { 0x54 } else { 0x28 }, 2, 64);
        vm.write_bytes(0x1900, b"\x90").unwrap();
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (C, 0x1901));
        assert_eq!(value(&vm, base(B), 2), u64::from(A));
        assert_eq!(value(&vm, base(C), 2), u64::from(B));
        assert_eq!(
            value(
                &vm,
                base(B) + if wide { 0x20 } else { 0x0e },
                if wide { 4 } else { 2 }
            ),
            0x1800
        );
        assert_eq!(
            value(
                &vm,
                0xa000 - if wide { 4 } else { 2 },
                if wide { 4 } else { 2 }
            ),
            64
        );
        assert!(busy(&vm, A) && busy(&vm, B) && busy(&vm, C));
    }
}

#[test]
fn task_flags_preserve_rf_and_loaded_tf_traps_after_first_new_instruction() {
    for (model, wide) in formats() {
        let (vm, mut cpu) = setup_tasks(model, wide, &far(false, B));
        vm.write_bytes(0x1800, b"\x90\x90").unwrap();
        put(
            &vm,
            base(B) + if wide { 0x24 } else { 0x10 },
            if wide { 4 } else { 2 },
            // SDM Vol3B §§25.16–25.16.1 pp.25-6–7; PRM386
            // §2.3.4 Fig.2-8 p.2-11: absent AC/ID cannot load from a TSS.
            if wide { 0x250102 } else { 0x102 },
        );
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rflags_system & ((1 << 18) | (1 << 21)), 0);
        assert_eq!(cpu.cpu.rflags_system & 0x100, 0x100);
        if wide {
            assert_ne!(cpu.cpu.rflags_system & 0x10000, 0);
        }
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::Exception { vector: 1, .. })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!(cpu.cpu.rip, 0x1801);
        assert_eq!(cpu.cpu.rflags_system & 0x10000, 0);
    }
}

#[test]
fn i286_ldt_task_gate_call_iret_and_gate_jumps_preserve_complete_images() {
    // 286 PRM §8.2 Fig. 8-1 pp. 8-2–3: dynamic words and static LDT field;
    // §§8.3–8.5 pp. 8-5–9, App. B SWITCH_TASKS pp. B-12–13:
    // load LDTR before resolving task selectors, preserve task links on JMP.
    // App. B IRET p. B-51 saves the successor IP. MSW uses decision-21's
    // reserved-bit policy (PRM §§10.2.2, 10.4 pp. 10-4/10-6; SMSW p. B-103).
    use x86jit_core::legacy::{SegmentCache, SystemSegment, TableRegister};

    const LDT_FIRST: u16 = 64;
    const LDT_SECOND: u16 = 72;
    const SECOND_GATE: u16 = 56;
    const LOCAL_CODE: u16 = 4;
    const LOCAL_DATA: u16 = 12;
    const TSS_BASES: [u64; 3] = [0x4000, 0x4400, 0x4800];
    const TSS_LIMIT: u32 = 0xff;
    const CALLER: [u16; 8] = [
        0x1111, 0x2222, 0x3333, 0x4444, 0x1f00, 0x1f80, 0x6666, 0x7777,
    ];
    const FIRST: [u16; 8] = [
        0x1010, 0x2020, 0x3030, 0x4040, 0x1000, 0x1080, 0x6060, 0x7070,
    ];
    const SECOND: [u16; 8] = [
        0x1112, 0x2223, 0x3334, 0x4445, 0x1200, 0x1280, 0x6667, 0x7778,
    ];

    fn save_image(image: &mut [u8; 256], ip: u16, flags: u16, regs: [u16; 8], local: bool) {
        let words = [
            ip,
            flags,
            regs[0],
            regs[1],
            regs[2],
            regs[3],
            regs[4],
            regs[5],
            regs[6],
            regs[7],
            DATA,
            if local { LOCAL_CODE } else { CODE },
            DATA,
            if local { LOCAL_DATA } else { DATA },
        ];
        for (offset, word) in (0x0e..0x2a).step_by(2).zip(words) {
            image[offset..offset + 2].copy_from_slice(&word.to_le_bytes());
        }
    }
    fn assert_task(cpu: &Vcpu, task: u16, ip: u16, regs: [u16; 8], flags: u16, ldt: u16, msw: u16) {
        let mut full_regs = [0; 16];
        full_regs[..8].copy_from_slice(&regs.map(u64::from));
        assert_eq!(cpu.cpu.gpr, full_regs);
        assert_eq!(cpu.cpu.rip, u64::from(ip));
        assert_eq!(cpu.cpu.flags.to_flags16(), flags & 0x0fff);
        assert_eq!(cpu.cpu.rflags_system, u64::from(flags & 0xf000));
        assert!(!cpu.cpu.flags.if_);
        assert_eq!((cpu.cpu.legacy.cpl, cpu.cpu.legacy.a20_enabled), (0, 1));
        assert_eq!(cpu.cpu.legacy.cr0, u32::from(msw & 0xf));
        assert_eq!((cpu.cpu.legacy.cr0 as u16 & 0xf) | 0xfff0, msw);
        assert_eq!(
            cpu.cpu.legacy.gdtr,
            TableRegister {
                base: 0x2000,
                limit: 0x7f,
                reserved: 0
            }
        );
        assert_eq!(
            cpu.cpu.legacy.idtr,
            TableRegister {
                base: 0x3000,
                limit: 0x7ff,
                reserved: 0
            }
        );
        let index = usize::from((task - A) / 8);
        assert_eq!(
            cpu.cpu.legacy.tr,
            SystemSegment {
                selector: task,
                reserved: 0,
                cache: SegmentCache {
                    base: TSS_BASES[index] as u32,
                    limit: TSS_LIMIT,
                    attributes: 0x83,
                    unusable: 0,
                    reserved: 0
                },
            }
        );
        let null = SegmentCache {
            unusable: 1,
            ..Default::default()
        };
        let data = SegmentCache {
            base: 0x8000,
            limit: 0x1fff,
            attributes: 0x93,
            unusable: 0,
            reserved: 0,
        };
        let (cs, ds, code, local_data) = match task {
            A => (
                CODE,
                DATA,
                SegmentCache {
                    base: 0,
                    limit: 0xffff,
                    attributes: 0x9b,
                    unusable: 0,
                    reserved: 0,
                },
                data,
            ),
            B | C => (
                LOCAL_CODE,
                LOCAL_DATA,
                SegmentCache {
                    base: if task == B { 0x6000 } else { 0x6100 },
                    limit: 31,
                    attributes: 0x9b,
                    unusable: 0,
                    reserved: 0,
                },
                SegmentCache {
                    base: if task == B { 0x7000 } else { 0x7100 },
                    limit: 3,
                    attributes: 0x93,
                    unusable: 0,
                    reserved: 0,
                },
            ),
            _ => unreachable!(),
        };
        assert_eq!(
            [cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs],
            [cs, ds, DATA, DATA, 0, 0]
        );
        assert_eq!(
            cpu.cpu.legacy.segments,
            [code, local_data, data, data, null, null]
        );
        assert_eq!(
            cpu.cpu.legacy.ldtr,
            SystemSegment {
                selector: ldt,
                reserved: 0,
                cache: if ldt == 0 {
                    null
                } else {
                    SegmentCache {
                        base: if ldt == LDT_FIRST { 0x5000 } else { 0x5100 },
                        limit: 15,
                        attributes: 0x82,
                        unusable: 0,
                        reserved: 0,
                    }
                },
            }
        );
    }
    fn assert_memory(
        vm: &Vm,
        images: &[[u8; 256]; 3],
        busy_bytes: [u8; 3],
        local_access: [[u8; 2]; 2],
    ) {
        for (i, expected) in images.iter().enumerate() {
            let mut actual = [0; 256];
            vm.read_bytes(TSS_BASES[i], &mut actual).unwrap();
            assert_eq!(&actual, expected, "TSS selector {}", A + i as u16 * 8);
        }
        for (selector, expected) in [
            (CODE, 0x9b),
            (DATA, 0x93),
            (A, busy_bytes[0]),
            (B, busy_bytes[1]),
            (C, busy_bytes[2]),
            (GATE, 0x85),
            (SECOND_GATE, 0x85),
            (LDT_FIRST, 0x82),
            (LDT_SECOND, 0x82),
        ] {
            assert_eq!(
                value(vm, 0x2005 + u64::from(selector), 1),
                u64::from(expected),
                "GDT selector {selector}"
            );
        }
        for (base, expected) in [0x5000, 0x5100].into_iter().zip(local_access) {
            assert_eq!(value(vm, base + 5, 1), u64::from(expected[0]));
            assert_eq!(value(vm, base + 13, 1), u64::from(expected[1]));
        }
    }

    let mut program = vec![
        0xb8,
        A as u8,
        0,
        0x0f,
        0x00,
        0xd8,
        0xb8,
        LDT_FIRST as u8,
        0,
        0x0f,
        0x00,
        0xd0,
    ];
    let smsw = b"\x0f\x01\x26\x02\x00";
    program.extend(smsw);
    let mut call = far(true, GATE);
    call[1..3].copy_from_slice(&0xbeefu16.to_le_bytes());
    program.extend(call);
    program.extend(smsw);
    program.extend(far(false, SECOND_GATE));
    let (vm, mut cpu) = setup(CpuModel::I80286, &program);
    descriptor(&vm, 0x2008, 0, 0xffff, 0x9b);
    descriptor(&vm, 0x2010, 0x8000, 0x1fff, 0x93);
    descriptor(&vm, 0x2040, 0x5000, 15, 0x82);
    descriptor(&vm, 0x2048, 0x5100, 15, 0x82);
    for (ldt, code_base, data_base) in [(0x5000, 0x6000, 0x7000), (0x5100, 0x6100, 0x7100)] {
        descriptor(&vm, ldt, code_base, 31, 0x9a);
        descriptor(&vm, ldt + 8, data_base, 3, 0x92);
    }
    gate(&vm, 0x2030, B, 0x85);
    gate(&vm, 0x2038, C, 0x85);
    vm.write_bytes(0x2030, &0xabcdu16.to_le_bytes()).unwrap();
    vm.write_bytes(0x2038, &0xcafeu16.to_le_bytes()).unwrap();
    let mut first_code = smsw.to_vec();
    first_code.extend(b"\xa1\x00\x00\xbb\x68\x24\xba\x57\x13\xcf");
    assert_eq!(first_code.len(), 15);
    first_code.extend(smsw);
    first_code.extend(b"\xbe\xef\xbe\xf4");
    vm.write_bytes(0x6000, &first_code).unwrap();
    let mut second_code = smsw.to_vec();
    second_code.extend(b"\xa1\x00\x00\xb9\x76\x98");
    let mut jump = far(false, GATE);
    jump[1..3].copy_from_slice(&0xfaceu16.to_le_bytes());
    second_code.extend(jump);
    vm.write_bytes(0x6100, &second_code).unwrap();
    vm.write_bytes(0x7000, b"\x34\x12\xa5\x5a").unwrap();
    vm.write_bytes(0x7100, b"\x78\x56\x5a\xa5").unwrap();
    let mut images = [[0xa5; 256], [0x5a; 256], [0x3c; 256]];
    for (i, (registers, ldt)) in [(CALLER, 0), (FIRST, LDT_FIRST), (SECOND, LDT_SECOND)]
        .into_iter()
        .enumerate()
    {
        images[i][..2].copy_from_slice(&0u16.to_le_bytes());
        save_image(&mut images[i], 0, 0, registers, i != 0);
        images[i][0x2a..0x2c].copy_from_slice(&ldt.to_le_bytes());
        vm.write_bytes(TSS_BASES[i], &images[i]).unwrap();
        descriptor(
            &vm,
            0x2018 + i as u64 * 8,
            TSS_BASES[i] as u32,
            TSS_LIMIT,
            0x81,
        );
    }
    cpu.cpu.legacy.idtr = TableRegister {
        base: 0x3000,
        limit: 0x7ff,
        reserved: 0,
    };
    cpu.cpu.cs = CODE;
    cpu.cpu.ds = DATA;
    cpu.cpu.es = DATA;
    cpu.cpu.ss = DATA;
    cpu.cpu.legacy.segments = [
        SegmentCache {
            base: 0,
            limit: 0xffff,
            attributes: 0x9b,
            unusable: 0,
            reserved: 0,
        },
        SegmentCache {
            base: 0x8000,
            limit: 0x1fff,
            attributes: 0x93,
            unusable: 0,
            reserved: 0,
        },
        SegmentCache {
            base: 0x8000,
            limit: 0x1fff,
            attributes: 0x93,
            unusable: 0,
            reserved: 0,
        },
        SegmentCache {
            base: 0x8000,
            limit: 0x1fff,
            attributes: 0x93,
            unusable: 0,
            reserved: 0,
        },
        SegmentCache {
            unusable: 1,
            ..Default::default()
        },
        SegmentCache {
            unusable: 1,
            ..Default::default()
        },
    ];
    steps(&mut cpu, &vm, 4); // LTR and LLDT install the initial system caches.
    cpu.cpu.gpr[..8].copy_from_slice(&CALLER.map(u64::from));
    cpu.cpu.flags.set_flags16(0x47);
    cpu.cpu.rflags_system = 0;
    assert!(cpu.a20_enabled());
    steps(&mut cpu, &vm, 1);
    assert_eq!(value(&vm, 0x8002, 2), 0xfff1);
    assert_task(&cpu, A, 0x1011, CALLER, 0x47, LDT_FIRST, 0xfff1);
    assert_memory(
        &vm,
        &images,
        [0x83, 0x81, 0x81],
        [[0x9a, 0x92], [0x9a, 0x92]],
    );

    steps(&mut cpu, &vm, 1); // CALL through the first task gate.
    save_image(&mut images[0], 0x1016, 0x47, CALLER, false);
    images[1][..2].copy_from_slice(&A.to_le_bytes());
    assert_task(&cpu, B, 0, FIRST, 0x4002, LDT_FIRST, 0xfff9);
    assert_memory(
        &vm,
        &images,
        [0x83, 0x83, 0x81],
        [[0x9b, 0x93], [0x9a, 0x92]],
    );
    steps(&mut cpu, &vm, 1);
    assert_eq!(value(&vm, 0x7002, 2), 0xfff9);
    let mut first_saved = FIRST;
    first_saved[0] = 0x1234;
    first_saved[3] = 0x2468;
    first_saved[2] = 0x1357;
    steps(&mut cpu, &vm, 3);
    assert_task(&cpu, B, 14, first_saved, 0x4002, LDT_FIRST, 0xfff9);
    steps(&mut cpu, &vm, 1); // IRET through B's back link to the caller.
    save_image(&mut images[1], 15, 2, first_saved, true);
    assert_task(&cpu, A, 0x1016, CALLER, 0x47, 0, 0xfff9);
    assert_memory(
        &vm,
        &images,
        [0x83, 0x81, 0x81],
        [[0x9b, 0x93], [0x9a, 0x92]],
    );
    steps(&mut cpu, &vm, 1);
    assert_eq!(value(&vm, 0x8002, 2), 0xfff9);

    steps(&mut cpu, &vm, 1); // JMP through the second task gate.
    save_image(&mut images[0], 0x1020, 0x47, CALLER, false);
    assert_task(&cpu, C, 0, SECOND, 2, LDT_SECOND, 0xfff9);
    assert_memory(
        &vm,
        &images,
        [0x81, 0x81, 0x83],
        [[0x9b, 0x93], [0x9b, 0x93]],
    );
    steps(&mut cpu, &vm, 1);
    assert_eq!(value(&vm, 0x7102, 2), 0xfff9);
    let mut second_saved = SECOND;
    second_saved[0] = 0x5678;
    second_saved[1] = 0x9876;
    steps(&mut cpu, &vm, 2);
    assert_task(&cpu, C, 11, second_saved, 2, LDT_SECOND, 0xfff9);
    steps(&mut cpu, &vm, 1); // JMP back through the first gate resumes B after IRET.
    save_image(&mut images[2], 16, 2, second_saved, true);
    assert_task(&cpu, B, 15, first_saved, 2, LDT_FIRST, 0xfff9);
    assert_memory(
        &vm,
        &images,
        [0x81, 0x83, 0x81],
        [[0x9b, 0x93], [0x9b, 0x93]],
    );
    steps(&mut cpu, &vm, 1);
    assert_eq!(value(&vm, 0x7002, 2), 0xfff9);
    steps(&mut cpu, &vm, 1);
    first_saved[6] = 0xbeef;
    assert_task(&cpu, B, 23, first_saved, 2, LDT_FIRST, 0xfff9);
    assert_memory(
        &vm,
        &images,
        [0x81, 0x83, 0x81],
        [[0x9b, 0x93], [0x9b, 0x93]],
    );
}

#[test]
fn review_switch_continuation_retires_call_before_accepting_irq() {
    for (model, wide) in formats() {
        let (mut vm, mut cpu) = setup_tasks(model, wide, &far(true, B));
        cpu.cpu.legacy.gdtr.limit = 0x1007;
        cpu.cpu.legacy.idtr.base = 0x7000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        put(&vm, base(B) + if wide { 0x60 } else { 0x2a }, 2, 0x1000);
        put(
            &vm,
            base(B) + if wide { 0x24 } else { 0x10 },
            if wide { 4 } else { 2 },
            0x202,
        );
        vm.write_bytes(0x7180, &[0, 0x1a, 8, 0, 0, 0x86, 0, 0])
            .unwrap();
        vm.write_bytes(0x1800, b"\x43").unwrap();
        vm.write_bytes(0x1a00, b"\x90").unwrap();
        replace_page(&mut vm, 0x3000, false);
        let retired = cpu.retired_instructions();
        let executed = cpu.executed_instructions();
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::UnmappedMemory { addr: 0x3000, .. })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!(cpu.cpu.legacy.tr.selector, B);
        assert!(!cpu.cpu.mmio_retry_active);
        vm.map(
            0x3000,
            0x1000,
            x86jit_core::Prot::RWX,
            x86jit_core::RegionKind::Ram,
        )
        .unwrap();
        descriptor(&vm, 0x3000, 0x5000, 7, 0x82);
        cpu.inject_irq(0x30);
        assert!(!cpu.irq_ready());
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[3]), (0x1800, 0));
        assert!(cpu.has_pending_irq());
        assert_eq!(
            (cpu.retired_instructions(), cpu.executed_instructions()),
            (retired + 1, executed + 1)
        );
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rip, 0x1a01);
        assert!(!cpu.has_pending_irq());
        assert_eq!(value(&vm, 0x8ffa, 2), 0x1800);
        assert_eq!(
            (cpu.retired_instructions(), cpu.executed_instructions()),
            (retired + 2, executed + 2)
        );
    }
}

#[test]
fn review_aliasing_tss_loads_the_state_just_saved_including_a20_aliases() {
    // 286 App. B pp. B-12–13; 386 §7.5 p. 7-8: save precedes load.
    for (model, wide) in formats() {
        for alias_a20 in [false, true] {
            let (vm, mut cpu) = setup_tasks(model, wide, &far(false, B));
            descriptor(
                &vm,
                0x2020,
                base(A) as u32 | if alias_a20 { 0x100000 } else { 0 },
                if wide { 0x67 } else { 0x2b },
                if wide { 0x89 } else { 0x81 },
            );
            let expected = [
                0x1122, 0x3344, 0x5566, 0x7788, 0x8000, 0x99aa, 0xbbcc, 0xddee,
            ];
            cpu.cpu.gpr[..8].copy_from_slice(&expected);
            cpu.cpu.flags.set_flags16(0x47);
            if alias_a20 {
                cpu.set_a20_enabled(false);
            }
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.rip, 0x100b);
            assert_eq!(cpu.cpu.gpr[..8], expected);
            assert_eq!(cpu.cpu.flags.to_flags16(), 0x47);
        }
    }
}

#[test]
fn review_trapped_outgoing_busy_byte_refuses_before_any_task_write() {
    for (model, wide) in formats() {
        for alias_a20 in [false, true] {
            let (mut vm, mut cpu) = setup_tasks(model, wide, &far(false, B));
            let mut table = [0; 0x80];
            vm.read_bytes(0x2000, &mut table).unwrap();
            vm.write_bytes(0x2fe0, &table).unwrap();
            cpu.cpu.legacy.gdtr.base = 0x2fe0 | if alias_a20 { 0x100000 } else { 0 };
            if alias_a20 {
                cpu.set_a20_enabled(false);
            }
            let mut old = [0; 0x68];
            vm.read_bytes(base(A), &mut old).unwrap();
            replace_page(&mut vm, 0x2000, true);
            let before = format!("{:?}", cpu.cpu);
            let mut run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            if matches!(run.stop(), StepStop::Exit(Exit::MmioRead { size: 1, .. })) {
                // Complete the access-byte read to expose a lost continuation
                // if a late atomic refusal follows an already-acquired B.
                cpu.complete_mmio_read(if wide { 0x8b } else { 0x83 });
                run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            }
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnsupportedSystemContext {
                        operation: x86jit_core::UnsupportedSystemOperation::LegacyInstruction,
                        ..
                    })
                ),
                "{:?}",
                run.stop()
            );
            assert_eq!(run.steps(), 0);
            assert_eq!(value(&vm, 0x3005, 1), if wide { 0x89 } else { 0x81 });
            assert_eq!(format!("{:?}", cpu.cpu), before);
            let mut after = [0; 0x68];
            vm.read_bytes(base(A), &mut after).unwrap();
            assert_eq!(old, after);
        }
    }
}

#[test]
fn review_software_task_gate_preserves_incoming_rf_with_or_without_continuation() {
    for continuation in [false, true] {
        let (mut vm, mut cpu) = setup_tasks(CpuModel::I80386, true, b"\xcd\x30");
        cpu.cpu.legacy.idtr.base = 0x7000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        gate(&vm, 0x7180, B, 0x85);
        put(&vm, base(B) + 0x24, 4, 0x10002);
        if continuation {
            cpu.cpu.legacy.gdtr.limit = 0x1007;
            put(&vm, base(B) + 0x60, 2, 0x1000);
            replace_page(&mut vm, 0x3000, false);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            assert!(matches!(
                run.stop(),
                StepStop::Exit(Exit::UnmappedMemory { addr: 0x3000, .. })
            ));
            vm.map(
                0x3000,
                0x1000,
                x86jit_core::Prot::RWX,
                x86jit_core::RegionKind::Ram,
            )
            .unwrap();
            descriptor(&vm, 0x3000, 0x5000, 7, 0x82);
        }
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (
                cpu.cpu.legacy.tr.selector,
                cpu.cpu.rip,
                cpu.cpu.rflags_system
            ),
            (B, 0x1800, 0x14000)
        );
    }
}

#[test]
fn review_code_presence_and_privilege_combined_failure_uses_model_order() {
    for (model, wide) in formats() {
        let (vm, mut cpu) = setup_tasks(model, wide, &far(false, B));
        descriptor(&vm, 0x2040, 0, 0xffff, 0x7a);
        put(&vm, base(B) + if wide { 0x4c } else { 0x24 }, 2, 64);
        outward(
            &mut cpu,
            &vm,
            if model == CpuModel::I80286 { 10 } else { 11 },
            64,
        );
        assert_eq!(cpu.cpu.legacy.tr.selector, B);
    }
}

#[test]
fn review_short_incoming_i286_tss_does_not_set_ts() {
    let (vm, mut cpu) = setup_tasks(CpuModel::I80286, false, &far(false, B));
    put(&vm, 0x2020, 2, 0x2a);
    outward(&mut cpu, &vm, 10, B.into());
    assert_eq!(cpu.cpu.legacy.tr.selector, B);
    assert_eq!(cpu.cpu.legacy.cr0, 1);
    assert!(!busy(&vm, A) && busy(&vm, B));
}

#[test]
fn review_short_current_i286_tss_is_checked_after_busy_acquisition() {
    // LTR p. B-72 has no limit check; SWITCH_TASKS p. B-12 requires 29h.
    let (vm, mut cpu) = setup_tasks(CpuModel::I80286, false, &far(false, B));
    descriptor(&vm, 0x2018, base(A) as u32, 0x28, 0x81);
    cpu.cpu.rip = 0x1000;
    steps(&mut cpu, &vm, 2);
    assert_eq!(cpu.cpu.legacy.tr.cache.limit, 0x28);
    let before = cpu.cpu.legacy;
    outward(&mut cpu, &vm, 10, B.into());
    assert!(busy(&vm, B));
    assert_eq!(cpu.cpu.legacy, before);
    assert_eq!(value(&vm, base(A) + 0x0e, 2), 0);
}

#[test]
fn review_i386_exception_table_resolves_ldt_stack_and_data_conflicts() {
    for wide in [false, true] {
        for ldt in [56, 60] {
            let (vm, mut cpu) = setup_tasks(CpuModel::I80386, wide, &far(false, B));
            descriptor(&vm, 0x2038, 0x6000, 15, 0x02);
            put(&vm, base(B) + if wide { 0x60 } else { 0x2a }, 2, ldt);
            outward(&mut cpu, &vm, 10, ldt);
        }
        for (offset, attr) in [
            (if wide { 0x50 } else { 0x26 }, 0xb2),
            (if wide { 0x54 } else { 0x28 }, 0x82),
        ] {
            let (vm, mut cpu) = setup_tasks(CpuModel::I80386, wide, &far(false, B));
            descriptor(&vm, 0x2040, 0, 0xffff, attr);
            put(&vm, base(B) + offset, 2, 64);
            outward(&mut cpu, &vm, 10, 64);
        }
    }
}

#[test]
fn review_complete_i286_tss_alias_image_and_system_caches() {
    use x86jit_core::legacy::{SegmentCache, SystemSegment};
    let (vm, mut cpu) = setup_tasks(CpuModel::I80286, false, &far(false, B));
    descriptor(&vm, 0x2020, base(A) as u32, 0x2b, 0x81);
    descriptor(&vm, 0x2038, 0x6000, 15, 0x82);
    let words: [u16; 22] = [
        0, 0x1110, 16, 0x2220, 16, 0x3330, 16, 0x1800, 2, 0xaaaa, 0xbbbb, 0xcccc, 0xdddd, 0x9000,
        0x9990, 0xeeee, 0xffff, 16, 8, 16, 16, 56,
    ];
    let bytes: Vec<_> = words.into_iter().flat_map(u16::to_le_bytes).collect();
    vm.write_bytes(base(A), &bytes).unwrap();
    let regs = [
        0x1122, 0x3344, 0x5566, 0x7788, 0x8000, 0x99aa, 0xbbcc, 0xddee,
    ];
    cpu.cpu.gpr[..8].copy_from_slice(&regs);
    cpu.cpu.flags.set_flags16(0x47);
    steps(&mut cpu, &vm, 1);
    assert_eq!(
        (cpu.cpu.rip, cpu.cpu.gpr[..8].to_vec()),
        (0x100b, regs.to_vec())
    );
    let expected: [u16; 22] = [
        0, 0x1110, 16, 0x2220, 16, 0x3330, 16, 0x100b, 0x47, 0x1122, 0x3344, 0x5566, 0x7788,
        0x8000, 0x99aa, 0xbbcc, 0xddee, 16, 8, 16, 16, 56,
    ];
    for (i, word) in expected.into_iter().enumerate() {
        assert_eq!(
            value(&vm, base(A) + i as u64 * 2, 2),
            u64::from(word),
            "TSS word {i}"
        );
    }
    let mut descriptors = [0; 16];
    vm.read_bytes(0x2018, &mut descriptors).unwrap();
    assert_eq!(
        descriptors,
        [0x2b, 0, 0, 0x40, 0, 0x81, 0, 0, 0x2b, 0, 0, 0x40, 0, 0x83, 0, 0]
    );
    assert_eq!(
        cpu.cpu.legacy.tr,
        SystemSegment {
            selector: B,
            reserved: 0,
            cache: SegmentCache {
                base: 0x4000,
                limit: 0x2b,
                attributes: 0x83,
                unusable: 0,
                reserved: 0
            }
        }
    );
    assert_eq!(
        cpu.cpu.legacy.ldtr,
        SystemSegment {
            selector: 56,
            reserved: 0,
            cache: SegmentCache {
                base: 0x6000,
                limit: 15,
                attributes: 0x82,
                unusable: 0,
                reserved: 0
            }
        }
    );
    assert_eq!(cpu.cpu.legacy.cr0, 9);
    vm.write_bytes(0x100b, b"\x0f\x01\x26\x00\x70").unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(value(&vm, 0x7000, 2), 0xfff9);
}

#[test]
fn review_i386_call_privilege_ip_and_disputed_busy_checks() {
    for wide in [false, true] {
        for selector in [B | 3, GATE | 3] {
            let (vm, mut cpu) = setup_tasks(CpuModel::I80386, wide, &far(true, selector));
            outward(&mut cpu, &vm, 13, u32::from(selector & !3));
        }
        let (vm, mut cpu) = setup_tasks(CpuModel::I80386, wide, &far(true, B));
        put(&vm, 0x2025, 1, if wide { 0x8b } else { 0x83 });
        outward(&mut cpu, &vm, 10, B.into());
        let (vm, mut cpu) = setup_tasks(CpuModel::I80386, wide, &far(true, B));
        descriptor(&vm, 0x2008, 0, 0x17ff, 0x9b);
        outward(&mut cpu, &vm, 13, 0);
        assert_eq!(cpu.cpu.legacy.tr.selector, B);
    }
}

#[test]
fn i386_paging_off_does_not_read_cr3_from_the_incoming_tss() {
    // 386 PRM §7.1 p. 7-3 reads PDBR only with paging enabled. The
    // unconditional read in SDM Vol 3A §10.3 p. 10-11 is a recorded conflict.
    let (mut vm, mut cpu) = setup_tasks(CpuModel::I80386, true, &far(false, B));
    let mut table = [0; 0x80];
    vm.read_bytes(0x2000, &mut table).unwrap();
    vm.write_bytes(0x8000, &table).unwrap();
    cpu.cpu.legacy.gdtr.base = 0x8000;
    let mut image = [0; 0x68];
    vm.read_bytes(base(B), &mut image).unwrap();
    vm.write_bytes(0x2fe0, &image).unwrap();
    descriptor(&vm, 0x8020, 0x2fe0, 0x67, 0x89);
    replace_page(&mut vm, 0x2000, false);
    cpu.cpu.legacy.cr3 = 0x12345000;
    steps(&mut cpu, &vm, 1);
    assert_eq!(
        (cpu.cpu.legacy.tr.selector, cpu.cpu.rip, cpu.cpu.legacy.cr3),
        (B, 0x1800, 0x12345000)
    );
}

// Flat backing survives a tag change; split only the pages remapped by these witnesses.
fn split_task_pages(vm: &mut Vm) {
    vm.unmap(0, 0x240000).unwrap();
    let boundaries = [0, 0x2000, 0x3000, 0x6000, 0x7000, 0x8000, 0x9000, 0x240000];
    for pair in boundaries.windows(2) {
        vm.map(
            pair[0],
            (pair[1] - pair[0]) as usize,
            x86jit_core::Prot::RWX,
            x86jit_core::RegionKind::Ram,
        )
        .unwrap();
    }
}
fn retag_task_page(vm: &mut Vm, at: u64, kind: x86jit_core::RegionKind) {
    vm.unmap(at, 0x1000).unwrap();
    vm.map(at, 0x1000, x86jit_core::Prot::RWX, kind).unwrap();
}
fn expect_task_refusal(
    cpu: &mut Vcpu,
    vm: &Vm,
    operation: x86jit_core::UnsupportedSystemOperation,
) {
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0, "{:?}", run.stop());
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::UnsupportedSystemContext { operation: actual, .. }) if *actual == operation),
        "{:?}",
        run.stop()
    );
}

#[test]
fn review4_accessed_refusals_keep_each_incoming_segment_continuation() {
    // PRM 286 SWITCH_TASKS pp. B-12–13; 386 §7.5 pp. 7-8–9:
    // segment loading belongs to the committed incoming task.
    for (model, wide) in formats() {
        for index in 0..if wide { 6 } else { 4 } {
            let (mut vm, mut cpu) = setup_tasks(model, wide, &far(true, B));
            split_task_pages(&mut vm);
            descriptor(&vm, 0x2038, 0x6000, 7, 0x82);
            descriptor(&vm, 0x6000, 0, 0xffff, if index == 0 { 0x9a } else { 0x92 });
            put(&vm, base(B) + if wide { 0x60 } else { 0x2a }, 2, 56);
            let offset = if wide {
                [0x4c, 0x50, 0x54, 0x48, 0x58, 0x5c][index]
            } else {
                [0x24, 0x26, 0x28, 0x22][index]
            };
            put(&vm, base(B) + offset, 2, 4);
            retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Trap);
            for at in [0x6000, 0x6004] {
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(
                    matches!(run.stop(), StepStop::Exit(Exit::MmioRead { addr, size:4, .. }) if *addr == at),
                    "{:?}",
                    run.stop()
                );
                cpu.complete_mmio_read(value(&vm, at, 4));
            }
            expect_task_refusal(
                &mut cpu,
                &vm,
                x86jit_core::UnsupportedSystemOperation::LegacyInstruction,
            );
            assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1800));
            let retired = cpu.retired_instructions();
            retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Ram);
            steps(&mut cpu, &vm, 1);
            assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1800));
            assert_eq!(cpu.retired_instructions(), retired + 1);
            assert_eq!(value(&vm, 0x6005, 1), if index == 0 { 0x9b } else { 0x93 });
            assert!(busy(&vm, A) && busy(&vm, B));
        }
    }
}

#[test]
fn review4_ldt_facility_refusal_keeps_committed_continuation() {
    for wide in [false, true] {
        let (vm, mut cpu) = setup_tasks(CpuModel::I80386, wide, &far(false, B));
        descriptor(&vm, 0x2038, 0x6000, 7, 0x2082);
        put(&vm, base(B) + if wide { 0x60 } else { 0x2a }, 2, 56);
        expect_task_refusal(
            &mut cpu,
            &vm,
            x86jit_core::UnsupportedSystemOperation::LegacyEntryFacility,
        );
        descriptor(&vm, 0x2038, 0x6000, 7, 0x82);
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1800));
        assert_eq!(cpu.cpu.legacy.ldtr.cache.base, 0x6000);
        assert_eq!(cpu.cpu.legacy.ldtr.cache.unusable, 0);
    }
}

#[test]
fn review5_outgoing_busy_clear_discards_rom_write_after_save_handoff() {
    for (model, wide) in formats() {
        for returning in [false, true] {
            let (mut vm, mut cpu) = setup_tasks(model, wide, &far(returning, B));
            if returning {
                vm.write_bytes(0x1800, b"\xcf").unwrap();
                steps(&mut cpu, &vm, 1);
            }
            let (old, new, ip) = if returning {
                (B, A, 0x100b)
            } else {
                (A, B, 0x1800)
            };
            let save_base = if wide { 0x7fdc } else { 0x7ff0 };
            cpu.cpu.legacy.tr.cache.base = save_base;
            put(&vm, u64::from(save_base), 2, A.into());
            split_task_pages(&mut vm);
            retag_task_page(&mut vm, 0x8000, x86jit_core::RegionKind::Trap);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::MmioWrite { addr: 0x8000, .. })
                ),
                "{:?}",
                run.stop()
            );
            retag_task_page(&mut vm, 0x2000, x86jit_core::RegionKind::Rom);
            cpu.complete_mmio_write();
            loop {
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                if matches!(run.stop(), StepStop::Exit(Exit::MmioWrite { .. })) {
                    cpu.complete_mmio_write();
                } else {
                    assert!(
                        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                        "{:?}",
                        run.stop()
                    );
                    assert_eq!(run.steps(), 1);
                    break;
                }
            }
            assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (new, ip));
            assert!(busy(&vm, old) && busy(&vm, new));
            assert_ne!(cpu.cpu.legacy.tr.cache.attributes & 2, 0);
        }
    }
}

#[test]
fn review5_rom_descriptor_updates_complete_ltr_and_segment_loads() {
    for (model, wide) in formats() {
        for (code, attr, bit, local) in [
            (
                &b"\x0f\x00\xd8"[..],
                if wide { 0x89 } else { 0x81 },
                2,
                false,
            ),
            (&b"\x8e\xd8"[..], 0x92, 1, false),             // DS
            (&b"\x8e\xd0"[..], 0x92, 1, false),             // SS
            (&b"\xea\x00\x00\x10\x00"[..], 0x9a, 1, false), // CS
            (&b"\x8e\xd8"[..], 0x92, 1, true),              // DS through LDTR
        ] {
            let (mut vm, mut cpu) = setup(model, code);
            descriptor(&vm, 0x2010, 0x4000, 0xffff, attr);
            cpu.cpu.gpr[0] = if local { 20 } else { 16 };
            if local {
                cpu.cpu.legacy.ldtr = x86jit_core::legacy::SystemSegment {
                    selector: 8,
                    cache: x86jit_core::legacy::SegmentCache {
                        base: 0x2000,
                        limit: 0xffff,
                        attributes: 0x82,
                        ..Default::default()
                    },
                    ..Default::default()
                };
            }
            split_task_pages(&mut vm);
            retag_task_page(&mut vm, 0x2000, x86jit_core::RegionKind::Rom);
            steps(&mut cpu, &vm, 1);
            assert_eq!(value(&vm, 0x2015, 1), u64::from(attr));
            if bit == 2 {
                assert_eq!(cpu.cpu.legacy.tr.selector, 16);
                assert_eq!(cpu.cpu.legacy.tr.cache.attributes, attr | bit);
            } else {
                let segment = if attr == 0x9a {
                    0
                } else if code == b"\x8e\xd0" {
                    3
                } else {
                    1
                };
                assert_eq!(cpu.cpu.legacy.segments[segment].attributes, attr | bit);
                assert_eq!(cpu.cpu.legacy.segments[segment].unusable, 0);
            }
        }
    }
}

#[test]
fn review4_irq_task_debug_refusal_consumes_delivery_once() {
    for handoff in [false, true] {
        let (mut vm, mut cpu) = setup_tasks(CpuModel::I80386, true, b"\x90");
        cpu.cpu.legacy.idtr.base = 0x7000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        gate(&vm, 0x7180, B, 0x85);
        put(&vm, base(B) + 0x24, 4, 0x202);
        put(&vm, base(B) + 0x64, 2, 1);
        vm.write_bytes(0x1800, b"\x90").unwrap();
        cpu.cpu.flags.if_ = true;
        if handoff {
            cpu.cpu.legacy.gdtr.limit = 0x1007;
            put(&vm, base(B) + 0x60, 2, 0x1000);
            replace_page(&mut vm, 0x3000, false);
        }
        cpu.inject_irq(0x30);
        if handoff {
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnmappedMemory { addr: 0x3000, .. })
                ),
                "{:?}",
                run.stop()
            );
            vm.map(
                0x3000,
                0x1000,
                x86jit_core::Prot::RWX,
                x86jit_core::RegionKind::Ram,
            )
            .unwrap();
            descriptor(&vm, 0x3000, 0x6000, 7, 0x82);
        }
        expect_task_refusal(
            &mut cpu,
            &vm,
            x86jit_core::UnsupportedSystemOperation::LegacyTaskDebugTrap,
        );
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1800));
        assert!(!cpu.has_pending_irq(), "completed IRQ was left queued");
        let retired = cpu.retired_instructions();
        cpu.cpu.task_debug_trap = 0;
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rip, 0x1801);
        assert_eq!(cpu.retired_instructions(), retired + 1);
    }
}

fn return_alias_witness(model: CpuModel, wide: bool) {
    // 286 IRET p. B-51 / 386 IRET p. 17-76 mark the departed task
    // available after SWITCH_TASKS. Its live access byte aliases incoming IP.
    let (vm, mut cpu) = setup_tasks(model, wide, &far(true, B));
    vm.write_bytes(0x1800, b"\xcf").unwrap();
    steps(&mut cpu, &vm, 1);
    cpu.cpu.legacy.gdtr.base = 0x6000;
    let incoming_base = if wide { 0x6005 } else { 0x6017 };
    let attr = if wide { 0x8b } else { 0x83 };
    descriptor(&vm, 0x6008, 0, 0xffff, 0x9b);
    descriptor(&vm, 0x6010, 0, 0xffff, 0x93);
    descriptor(
        &vm,
        0x6018,
        incoming_base,
        if wide { 0x67 } else { 0x2b },
        attr,
    );
    descriptor(
        &vm,
        0x6020,
        base(B) as u32,
        if wide { 0x67 } else { 0x2b },
        attr,
    );
    let fields = if wide {
        vec![
            (0x24, 4, 2),
            (0x38, 4, 0x8000),
            (0x48, 2, 16),
            (0x4c, 2, 8),
            (0x50, 2, 16),
            (0x54, 2, 16),
            (0x58, 2, 0),
            (0x5c, 2, 0),
            (0x60, 2, 0),
            (0x64, 2, 0),
            (0x66, 2, 0x68),
        ]
    } else {
        vec![
            (0x10, 2, 2),
            (0x1a, 2, 0x8000),
            (0x22, 2, 16),
            (0x24, 2, 8),
            (0x26, 2, 16),
            (0x28, 2, 16),
            (0x2a, 2, 0),
        ]
    };
    for (off, size, val) in fields {
        put(&vm, u64::from(incoming_base) + off, size, val);
    }
    steps(&mut cpu, &vm, 1);
    assert_eq!(
        (cpu.cpu.legacy.tr.selector, cpu.cpu.rip),
        (A, u64::from(attr))
    );
    assert_eq!(value(&vm, 0x6025, 1), u64::from(attr & !2));
    assert_eq!(
        value(
            &vm,
            base(B) + if wide { 0x20 } else { 0x0e },
            if wide { 4 } else { 2 }
        ),
        0x1801
    );
}
#[test]
fn review4_i386_iret_reads_incoming_image_before_departing_busy_clear() {
    for wide in [false, true] {
        return_alias_witness(CpuModel::I80386, wide);
    }
}
#[test]
fn review4_i286_iret_reads_incoming_image_before_departing_busy_clear() {
    return_alias_witness(CpuModel::I80286, false);
}

#[test]
fn review4_image_refusals_keep_committed_state_until_input_is_corrected() {
    for wide in [false, true] {
        for virtual_mode in [false, true] {
            if virtual_mode && !wide {
                continue;
            }
            let (vm, mut cpu) = setup_tasks(CpuModel::I80386, wide, &far(true, B));
            let (offset, size, bad, good, operation) = if virtual_mode {
                (
                    0x24,
                    4,
                    0x20002,
                    2,
                    x86jit_core::UnsupportedSystemOperation::InvalidLegacyContext(
                        x86jit_core::legacy::MachineContextError::Virtual8086Unsupported,
                    ),
                )
            } else {
                (
                    if wide { 0x4c } else { 0x24 },
                    2,
                    11,
                    8,
                    x86jit_core::UnsupportedSystemOperation::LegacyPrivilegeTransition,
                )
            };
            put(&vm, base(B) + offset, size, bad);
            expect_task_refusal(&mut cpu, &vm, operation);
            assert_eq!(cpu.cpu.legacy.tr.selector, B);
            put(&vm, base(B) + offset, size, good);
            steps(&mut cpu, &vm, 1);
            assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1800));
            assert_eq!(
                value(
                    &vm,
                    base(A) + if wide { 0x20 } else { 0x0e },
                    if wide { 4 } else { 2 }
                ),
                0x100b
            );
            assert_eq!(cpu.cpu.rflags_system & 0x4000, 0x4000);
        }
    }
}

#[test]
fn review4_irq_segment_refusal_retains_event_and_does_not_consume_a_new_irq() {
    for (model, wide) in formats() {
        for delivery in [ExceptionDelivery::Outward, ExceptionDelivery::InGuest] {
            let (mut vm, mut cpu) = setup_tasks_delivery(model, wide, b"\x90", delivery);
            cpu.cpu.legacy.idtr.base = 0x7000;
            cpu.cpu.legacy.idtr.limit = 0x7ff;
            gate(&vm, 0x7180, B, 0x85);
            descriptor(&vm, 0x2038, 0x6000, 7, 0x82);
            descriptor(&vm, 0x6000, 0, 0xffff, 0x9a);
            put(&vm, base(B) + if wide { 0x60 } else { 0x2a }, 2, 56);
            put(&vm, base(B) + if wide { 0x4c } else { 0x24 }, 2, 4);
            vm.write_bytes(0x1800, b"\x90").unwrap();
            cpu.cpu.flags.if_ = true;
            split_task_pages(&mut vm);
            retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Trap);
            cpu.inject_irq(0x30);
            for at in [0x6000, 0x6004] {
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(
                    matches!(run.stop(), StepStop::Exit(Exit::MmioRead {addr, size:4, ..}) if *addr == at),
                    "{:?}",
                    run.stop()
                );
                cpu.complete_mmio_read(value(&vm, at, 4));
            }
            expect_task_refusal(
                &mut cpu,
                &vm,
                x86jit_core::UnsupportedSystemOperation::LegacyInstruction,
            );
            assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1800));
            assert!(!cpu.has_pending_irq());
            // A separately queued instance of the same vector belongs to the
            // next delivery. Incoming IF=0 keeps it pending after completion.
            cpu.inject_irq(0x30);
            retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Ram);
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.rip, 0x1801);
            assert!(cpu.has_pending_irq());
            assert_eq!(value(&vm, base(B), 2), A.into());
        }
    }
}

#[test]
fn review4_irq_new_task_segment_fault_consumes_irq_outward_and_in_guest() {
    for (model, wide) in formats() {
        for delivery in [ExceptionDelivery::Outward, ExceptionDelivery::InGuest] {
            let (vm, mut cpu) = setup_tasks_delivery(model, wide, b"\x90", delivery);
            cpu.cpu.legacy.idtr.base = 0x7000;
            cpu.cpu.legacy.idtr.limit = 0x7ff;
            gate(&vm, 0x7180, B, 0x85);
            vm.write_bytes(0x7050, &[0, 0x1a, 8, 0, 0, 0x86, 0, 0])
                .unwrap();
            vm.write_bytes(0x1a00, b"\x90").unwrap();
            descriptor(&vm, 0x2040, 0, 0xffff, 0x82);
            put(&vm, base(B) + if wide { 0x54 } else { 0x28 }, 2, 64);
            cpu.cpu.flags.if_ = true;
            cpu.inject_irq(0x30);
            if delivery == ExceptionDelivery::Outward {
                outward(&mut cpu, &vm, 10, 65);
            } else {
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.rip, 0x1a01);
                assert_eq!(value(&vm, 0x8ff8, 2), 65);
            }
            assert_eq!(cpu.cpu.legacy.tr.selector, B);
            assert!(!cpu.has_pending_irq());
        }
    }
}

#[test]
fn review4_iret_busy_clear_follows_each_models_segment_loads() {
    for (model, wide) in formats() {
        let (vm, mut cpu) = setup_tasks(model, wide, &far(true, B));
        vm.write_bytes(0x1800, b"\xcf").unwrap();
        steps(&mut cpu, &vm, 1);
        descriptor(&vm, 0x2040, 0, 0xffff, 0x93);
        put(&vm, base(A) + if wide { 0x4c } else { 0x24 }, 2, 64);
        outward(&mut cpu, &vm, 10, 64);
        assert_eq!(cpu.cpu.legacy.tr.selector, A);
        assert!(busy(&vm, A));
        // Each model's IRET follows its SWITCH_TASKS descriptor loads
        // (286 B-51/B-13; 386 17-76/7-8). SDM §10.3 step 10 differs
        // in placing this clear before the descriptor loads (step 14).
        assert!(busy(&vm, B));
    }
}

#[test]
fn review4_task_debug_completion_retires_instruction_only_once() {
    let (vm, mut cpu) = setup_tasks(CpuModel::I80386, true, &far(true, B));
    put(&vm, base(B) + 0x64, 2, 1);
    vm.write_bytes(0x1800, b"\x90").unwrap();
    let before = cpu.retired_instructions();
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 1);
    assert!(
        matches!(
            run.stop(),
            StepStop::Exit(Exit::UnsupportedSystemContext {
                operation: x86jit_core::UnsupportedSystemOperation::LegacyTaskDebugTrap,
                ..
            })
        ),
        "{:?}",
        run.stop()
    );
    assert_eq!(cpu.retired_instructions(), before + 1);
    cpu.cpu.task_debug_trap = 0;
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.rip, 0x1801);
    assert_eq!(cpu.retired_instructions(), before + 2);
}

#[test]
fn review4_exception_task_debug_completion_finishes_error_push_once() {
    for handoff in [false, true] {
        let (mut vm, mut cpu) = setup_tasks_delivery(
            CpuModel::I80386,
            true,
            b"\x8e\xd8",
            ExceptionDelivery::InGuest,
        );
        cpu.cpu.legacy.idtr.base = 0x7000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        gate(&vm, 0x7068, B, 0x85);
        cpu.cpu.gpr[0] = 0x78;
        put(&vm, base(B) + 0x64, 2, 1);
        vm.write_bytes(0x1800, b"\x90").unwrap();
        // This unrelated IRQ remains masked in both tasks. Exception delivery
        // owns its error-code push, not the separately queued IRQ.
        cpu.inject_irq(0x30);
        if handoff {
            split_task_pages(&mut vm);
            retag_task_page(&mut vm, 0x8000, x86jit_core::RegionKind::Trap);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 0);
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::MmioWrite {
                        addr: 0x8ffc,
                        size: 4,
                        value: 0x78,
                        ..
                    })
                ),
                "{:?}",
                run.stop()
            );
            assert_eq!(cpu.cpu.gpr[4], 0x9000);
            cpu.complete_mmio_write();
        }
        expect_task_refusal(
            &mut cpu,
            &vm,
            x86jit_core::UnsupportedSystemOperation::LegacyTaskDebugTrap,
        );
        assert_eq!(
            (cpu.cpu.legacy.tr.selector, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (B, 0x1800, 0x8ffc)
        );
        if !handoff {
            assert_eq!(value(&vm, 0x8ffc, 4), 0x78);
        }
        assert!(cpu.has_pending_irq());
        let retired = cpu.retired_instructions();
        cpu.cpu.task_debug_trap = 0;
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (0x1801, 0x8ffc));
        assert_eq!(cpu.retired_instructions(), retired + 1);
        assert!(cpu.has_pending_irq());
    }
}

#[test]
fn review5_task_busy_and_accessed_updates_complete_on_rom_from_entry() {
    // SDM Vol 3A §3.4.5.1 p. 3-13: A is set assuming descriptor
    // memory supports writes. The mapped ROM discards the locked write.
    for (model, wide) in formats() {
        let (mut vm, mut cpu) = setup_tasks(model, wide, &far(false, B));
        put(&vm, 0x200d, 1, 0x9a);
        put(&vm, 0x2015, 1, 0x92);
        vm.write_bytes(0x1800, &far(false, A)).unwrap();
        split_task_pages(&mut vm);
        retag_task_page(&mut vm, 0x2000, x86jit_core::RegionKind::Rom);
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1800));
        assert_eq!(
            cpu.cpu.legacy.tr.cache.attributes,
            if wide { 0x8b } else { 0x83 }
        );
        assert!(busy(&vm, A) && !busy(&vm, B));
        assert_eq!((value(&vm, 0x200d, 1), value(&vm, 0x2015, 1)), (0x9a, 0x92));
        for i in [0, 1, 2, 3] {
            assert_eq!(
                cpu.cpu.legacy.segments[i].attributes,
                if i == 0 { 0x9b } else { 0x93 }
            );
        }
        outward(&mut cpu, &vm, 13, A.into());
        assert_eq!(cpu.cpu.legacy.tr.selector, B);
    }
}

fn pending_device_task_segment(
    model: CpuModel,
    wide: bool,
    delivery: ExceptionDelivery,
    call: bool,
    code: bool,
) -> (Vm, Vcpu) {
    let instruction = if call { far(true, B) } else { vec![0x90] };
    let (mut vm, mut cpu) = setup_tasks_delivery(model, wide, &instruction, delivery);
    cpu.cpu.legacy.idtr.base = 0x7000;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    gate(&vm, 0x7180, B, 0x85);
    descriptor(&vm, 0x2038, 0x6000, 7, 0x82);
    descriptor(&vm, 0x6000, 0, 0xffff, if code { 0x9a } else { 0x92 });
    put(&vm, base(B) + if wide { 0x60 } else { 0x2a }, 2, 56);
    put(
        &vm,
        base(B)
            + if code {
                if wide {
                    0x4c
                } else {
                    0x24
                }
            } else {
                if wide {
                    0x54
                } else {
                    0x28
                }
            },
        2,
        4,
    );
    put(
        &vm,
        base(B) + if wide { 0x24 } else { 0x10 },
        if wide { 4 } else { 2 },
        0x202,
    );
    vm.write_bytes(0x1800, b"\x90\x90").unwrap();
    cpu.cpu.flags.if_ = true;
    split_task_pages(&mut vm);
    retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Trap);
    if !call {
        cpu.inject_irq(0x30);
    }
    for at in [0x6000, 0x6004] {
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::MmioRead {addr, size:4, ..}) if *addr == at),
            "{:?}",
            run.stop()
        );
        cpu.complete_mmio_read(value(&vm, at, 4));
    }
    expect_task_refusal(
        &mut cpu,
        &vm,
        x86jit_core::UnsupportedSystemOperation::LegacyInstruction,
    );
    assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1800));
    assert!(!cpu.has_pending_irq());
    (vm, cpu)
}

#[test]
fn review5_state_edits_abandon_task_and_consumed_irq_together() {
    for (model, wide) in formats() {
        for delivery in [ExceptionDelivery::Outward, ExceptionDelivery::InGuest] {
            for edit_gpr in [false, true] {
                let (mut vm, mut cpu) =
                    pending_device_task_segment(model, wide, delivery, false, false);
                assert_eq!(cpu.cpu.legacy.segments[0].unusable, 0);
                assert_eq!(cpu.cpu.legacy.segments[3].unusable, 0);
                assert_eq!(cpu.cpu.legacy.segments[1].unusable, 1);
                let retired = cpu.retired_instructions();
                if edit_gpr {
                    cpu.cpu.gpr[0] = 0xbeef;
                } else {
                    cpu.cpu.legacy.cr2 = 0x12345000;
                }
                retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Ram);
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.rip, 0x1801);
                assert_eq!(cpu.retired_instructions(), retired + 1);
                assert_eq!(
                    cpu.cpu.legacy.segments[1].unusable, 1,
                    "abandoned DS load was resumed"
                );
                assert_eq!(value(&vm, 0x6005, 1), 0x92);
                assert!(!cpu.has_pending_irq());
                if edit_gpr {
                    assert_eq!(cpu.cpu.gpr[0], 0xbeef);
                } else {
                    assert_eq!(cpu.cpu.legacy.cr2, 0x12345000);
                }
            }
        }
    }
}

#[test]
fn review5_gpr_edit_abandons_call_without_retiring_it() {
    for (model, wide) in formats() {
        let (mut vm, mut cpu) =
            pending_device_task_segment(model, wide, ExceptionDelivery::Outward, true, false);
        let retired = cpu.retired_instructions();
        cpu.cpu.gpr[0] = 0xbeef;
        retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Ram);
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            cpu.cpu.rip, 0x1801,
            "abandoned CALL was retired instead of executing NOP"
        );
        assert_eq!(cpu.retired_instructions(), retired + 1);
        assert_eq!(cpu.cpu.gpr[0], 0xbeef);
        assert_eq!(cpu.cpu.legacy.segments[1].unusable, 1);
        assert_eq!(value(&vm, 0x6005, 1), 0x92);
    }
}

#[test]
fn review5_abandonment_precedes_unusable_cs_validation_and_releases_a20() {
    for (model, wide) in formats() {
        for delivery in [ExceptionDelivery::Outward, ExceptionDelivery::InGuest] {
            let (mut vm, mut cpu) = pending_device_task_segment(model, wide, delivery, false, true);
            retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Ram);
            cpu.set_a20_enabled(false);
            assert_eq!(cpu.pending_a20_enabled(), Some(false));
            cpu.cpu.legacy.cr2 = 0x12345000;
            let retired = cpu.retired_instructions();
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnsupportedSystemContext {
                        operation: x86jit_core::UnsupportedSystemOperation::InvalidLegacyContext(
                            x86jit_core::legacy::MachineContextError::UnusableCodeOrStack
                        ),
                        ..
                    })
                ),
                "{:?}",
                run.stop()
            );
            assert_eq!(run.steps(), 0);
            assert_eq!(cpu.retired_instructions(), retired);
            assert_eq!(cpu.cpu.legacy.cr2, 0x12345000);
            assert_eq!(
                cpu.pending_a20_enabled(),
                None,
                "abandoned switch still holds A20"
            );
            assert!(!cpu.a20_enabled());
            assert_eq!(value(&vm, 0x6005, 1), 0x9a);
            assert!(!cpu.has_pending_irq());
            // The edited, unfinished caches stand. Once the embedder installs
            // valid caches, event readiness must have no abandoned owner left.
            for i in [0, 3] {
                cpu.cpu.legacy.segments[i] = x86jit_core::legacy::SegmentCache {
                    base: 0,
                    limit: 0xffff,
                    attributes: if i == 0 { 0x9b } else { 0x93 },
                    ..Default::default()
                };
            }
            assert!(cpu.irq_ready());
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.rip, 0x1801);
        }
    }
}

#[test]
fn review5_zero_budget_preserves_owner_until_the_next_resume() {
    for (model, wide) in formats() {
        let (mut vm, mut cpu) =
            pending_device_task_segment(model, wide, ExceptionDelivery::Outward, true, false);
        cpu.set_a20_enabled(false);
        cpu.cpu.gpr[0] = 0xbeef;
        let retired = cpu.retired_instructions();
        let run = cpu.run_steps(&vm, 0, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
        assert_eq!(cpu.retired_instructions(), retired);
        assert_eq!(cpu.pending_a20_enabled(), Some(false));
        assert!(cpu.a20_enabled());
        retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Ram);
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[0]), (0x1801, 0xbeef));
        assert_eq!(cpu.pending_a20_enabled(), None);
        assert!(!cpu.a20_enabled());
        assert_eq!(cpu.cpu.legacy.segments[1].unusable, 1);
    }
}

#[test]
fn review5_abandonment_discards_accepted_irq_before_task_commit() {
    for (model, wide) in formats() {
        for delivery in [ExceptionDelivery::Outward, ExceptionDelivery::InGuest] {
            let (mut vm, mut cpu) = setup_tasks_delivery(model, wide, b"\x90", delivery);
            cpu.cpu.legacy.idtr.base = 0x6000;
            cpu.cpu.legacy.idtr.limit = 0x7ff;
            gate(&vm, 0x6180, B, 0x85);
            cpu.cpu.flags.if_ = true;
            cpu.cpu.legacy.tr.cache.base = if wide { 0x7fdc } else { 0x7ff0 };
            split_task_pages(&mut vm);
            retag_task_page(&mut vm, 0x8000, x86jit_core::RegionKind::Trap);
            cpu.inject_irq(0x30);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::MmioWrite { addr: 0x8000, .. })
                ),
                "{:?}",
                run.stop()
            );
            assert_eq!(cpu.cpu.legacy.tr.selector, A);
            assert!(cpu.has_pending_irq());
            let retired = cpu.retired_instructions();
            cpu.complete_mmio_write();
            cpu.cpu.legacy.cr2 = 0x12345000;
            steps(&mut cpu, &vm, 1);
            assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (A, 0x1007));
            assert_eq!(cpu.retired_instructions(), retired + 1);
            assert_eq!(cpu.cpu.legacy.cr2, 0x12345000);
            assert!(!cpu.has_pending_irq());
            assert!(!cpu.cpu.mmio_retry_active);
            assert!(!cpu.cpu.pending_mmio_write);
        }
    }
}

#[test]
fn review5_abandonment_preserves_a_new_irq_after_the_old_one_was_consumed() {
    for (model, wide) in formats() {
        let (mut vm, mut cpu) =
            pending_device_task_segment(model, wide, ExceptionDelivery::Outward, false, false);
        cpu.cpu.flags.if_ = false;
        cpu.cpu.gpr[0] = 0xbeef;
        cpu.inject_irq(0x30);
        retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Ram);
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[0]), (0x1801, 0xbeef));
        assert!(cpu.has_pending_irq());
        assert_eq!(cpu.cpu.legacy.segments[1].unusable, 1);
    }
}

fn pending_sti_debug_task(model: CpuModel, wide: bool) -> (Vm, Vcpu) {
    // 286 §9.7.1 p. 9-14; 386 §12.3.1.4 p. 12-8: STI can
    // single-step into a task before accepting a maskable IRQ.
    let (mut vm, mut cpu) = setup_tasks_delivery(model, wide, b"\xfb", ExceptionDelivery::InGuest);
    cpu.cpu.legacy.idtr.base = 0x7000;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    gate(&vm, 0x7008, B, 0x85);
    vm.write_bytes(0x7180, &[0, 0x1a, 8, 0, 0, 0x86, 0, 0])
        .unwrap();
    vm.write_bytes(0x1a00, b"\x90").unwrap();
    vm.write_bytes(0x3000, b"\x90\x90").unwrap();
    descriptor(&vm, 0x2038, 0x6000, 7, 0x82);
    descriptor(&vm, 0x6000, 0, 0xffff, 0x92);
    for (offset, size, val) in if wide {
        [
            (0x20, 4, 0x3000),
            (0x24, 4, 0x202),
            (0x54, 2, 4),
            (0x60, 2, 56),
        ]
    } else {
        [
            (0x0e, 2, 0x3000),
            (0x10, 2, 0x202),
            (0x28, 2, 4),
            (0x2a, 2, 56),
        ]
    } {
        put(&vm, base(B) + offset, size, val);
    }
    cpu.cpu.flags.if_ = false;
    cpu.cpu.rflags_system = 0x100;
    split_task_pages(&mut vm);
    retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Trap);
    let retired = cpu.retired_instructions();
    for (index, at) in [0x6000, 0x6004].into_iter().enumerate() {
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), u64::from(index == 0));
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::MmioRead {addr, size:4, ..}) if *addr == at),
            "{:?}",
            run.stop()
        );
        cpu.complete_mmio_read(value(&vm, at, 4));
    }
    expect_task_refusal(
        &mut cpu,
        &vm,
        x86jit_core::UnsupportedSystemOperation::LegacyInstruction,
    );
    assert_eq!(cpu.retired_instructions(), retired + 1);
    assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x3000));
    assert!(cpu.cpu.flags.if_);
    assert_eq!(cpu.cpu.rflags_system & 0x100, 0);
    assert_eq!(cpu.cpu.legacy.segments[0].unusable, 0);
    assert_eq!(cpu.cpu.legacy.segments[3].unusable, 0);
    assert_eq!(cpu.cpu.legacy.segments[1].unusable, 1);
    (vm, cpu)
}

fn check_abandoned_sti_debug_task_irq(unmapped: bool) {
    for (model, wide) in formats() {
        let (mut vm, mut cpu) = pending_sti_debug_task(model, wide);
        retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Ram);
        if unmapped {
            vm.unmap(0x3000, 0x3000).unwrap();
            vm.map(
                0x4000,
                0x2000,
                x86jit_core::Prot::RWX,
                x86jit_core::RegionKind::Ram,
            )
            .unwrap();
        }
        cpu.cpu.gpr[0] = 0xbeef;
        cpu.inject_irq(0x30);
        let retired = cpu.retired_instructions();
        let executed = cpu.executed_instructions();
        let mut run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        if matches!(
            run.stop(),
            StepStop::Exit(Exit::UnmappedMemory { addr: 0x3000, .. })
        ) {
            // Expose the persistent inhibition across repeated fetch failures.
            for _ in 0..2 {
                assert!(cpu.has_pending_irq());
                assert_eq!(cpu.retired_instructions(), retired);
                run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            }
        }
        assert_eq!(run.steps(), 1, "{:?}", run.stop());
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
            "{:?}",
            run.stop()
        );
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1a01));
        assert_eq!(cpu.cpu.gpr[0], 0xbeef);
        assert_eq!(cpu.cpu.legacy.segments[1].unusable, 1);
        assert_eq!(value(&vm, 0x6005, 1), 0x92);
        assert_eq!(
            value(&vm, 0x8ffa, 2),
            0x3000,
            "IRQ must precede B's first instruction"
        );
        assert_eq!(cpu.cpu.gpr[4], 0x8ffa);
        assert_eq!(cpu.retired_instructions(), retired + 1);
        assert_eq!(cpu.executed_instructions(), executed + 1);
        assert!(!cpu.has_pending_irq());
    }
}

#[test]
fn review6_sti_debug_task_abandonment_accepts_irq_before_first_instruction() {
    check_abandoned_sti_debug_task_irq(false);
}

#[test]
fn review6_sti_debug_task_abandonment_accepts_irq_before_unmapped_fetch() {
    check_abandoned_sti_debug_task_irq(true);
}

#[test]
fn review6_redirected_debug_event_releases_sti_shadow_before_fetch() {
    for (model, wide) in formats() {
        let (mut vm, mut cpu) =
            setup_tasks_delivery(model, wide, b"\xfb", ExceptionDelivery::InGuest);
        cpu.cpu.legacy.idtr.base = 0x6000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        vm.write_bytes(0x6008, &[0, 0x1a, 8, 0, 0, 0x86, 0, 0])
            .unwrap();
        vm.write_bytes(0x6180, &[0, 0x1b, 8, 0, 0, 0x86, 0, 0])
            .unwrap();
        vm.write_bytes(0x1b00, b"\x90").unwrap();
        cpu.cpu.flags.if_ = false;
        cpu.cpu.rflags_system = 0x100;
        split_task_pages(&mut vm);
        retag_task_page(&mut vm, 0x7000, x86jit_core::RegionKind::Trap);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1);
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::MmioWrite { .. })),
            "{:?}",
            run.stop()
        );
        retag_task_page(&mut vm, 0x7000, x86jit_core::RegionKind::Ram);
        vm.unmap(0x3000, 0x3000).unwrap();
        vm.map(
            0x4000,
            0x2000,
            x86jit_core::Prot::RWX,
            x86jit_core::RegionKind::Ram,
        )
        .unwrap();
        cpu.cpu.rip = 0x3000;
        cpu.cpu.rflags_system &= !0x100;
        cpu.inject_irq(0x30);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rip, 0x1b01);
        assert_eq!(value(&vm, 0x7ffa, 2), 0x3000);
        assert!(!cpu.has_pending_irq());
    }
}

#[test]
fn review6_outward_debug_event_exit_releases_sti_shadow() {
    for (model, wide) in formats() {
        let (mut vm, mut cpu) =
            setup_tasks_delivery(model, wide, b"\xfb", ExceptionDelivery::Outward);
        cpu.cpu.legacy.idtr.base = 0x7000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        vm.write_bytes(0x7180, &[0, 0x1b, 8, 0, 0, 0x86, 0, 0])
            .unwrap();
        vm.write_bytes(0x1b00, b"\x90").unwrap();
        cpu.cpu.flags.if_ = false;
        cpu.cpu.rflags_system = 0x100;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::Exception { vector: 1, .. })
            ),
            "{:?}",
            run.stop()
        );
        cpu.cpu.rflags_system &= !0x100;
        cpu.cpu.rip = 0x3000;
        split_task_pages(&mut vm);
        vm.unmap(0x3000, 0x3000).unwrap();
        vm.map(
            0x4000,
            0x2000,
            x86jit_core::Prot::RWX,
            x86jit_core::RegionKind::Ram,
        )
        .unwrap();
        cpu.inject_irq(0x30);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rip, 0x1b01);
        assert_eq!(value(&vm, 0x7ffa, 2), 0x3000);
        assert!(!cpu.has_pending_irq());
    }
}

#[test]
fn review6_nonresumable_debug_delivery_refusal_releases_sti_shadow() {
    for wide in [false, true] {
        let (mut vm, mut cpu) =
            setup_tasks_delivery(CpuModel::I80386, wide, b"\xfb", ExceptionDelivery::InGuest);
        cpu.cpu.legacy.idtr.base = 0x7000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        descriptor(&vm, 0x2040, 0, 0xffff, 0x209b);
        vm.write_bytes(0x7008, &[0, 0x1a, 64, 0, 0, 0x86, 0, 0])
            .unwrap();
        vm.write_bytes(0x7180, &[0, 0x1b, 8, 0, 0, 0x86, 0, 0])
            .unwrap();
        vm.write_bytes(0x1b00, b"\x90").unwrap();
        cpu.cpu.flags.if_ = false;
        cpu.cpu.rflags_system = 0x100;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::UnsupportedSystemContext {
                    operation: x86jit_core::UnsupportedSystemOperation::LegacyEntryFacility,
                    ..
                })
            ),
            "{:?}",
            run.stop()
        );
        cpu.cpu.rflags_system &= !0x100;
        cpu.cpu.rip = 0x3000;
        split_task_pages(&mut vm);
        vm.unmap(0x3000, 0x3000).unwrap();
        vm.map(
            0x4000,
            0x2000,
            x86jit_core::Prot::RWX,
            x86jit_core::RegionKind::Ram,
        )
        .unwrap();
        cpu.inject_irq(0x30);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rip, 0x1b01);
        assert_eq!(value(&vm, 0x7ffa, 2), 0x3000);
        assert!(!cpu.has_pending_irq());
    }
}

// ---------------------------------------------------------------------------
// The NMI input through an IDT task gate (xj-50w.19; spec.md §17.10):
// I286 PRM §9.2 p. 9-3 ("the first IRET", which may be a task return), §9.6
// p. 9-9; 80386 PRM §9.2.1 p. 9-3.

// Vector 2 through a task gate to B; B starts at 1800h with the TSS FLAGS
// image `flags`.
fn nmi_task(model: CpuModel, wide: bool, flags: u32, b_code: &[u8]) -> (Vm, Vcpu) {
    let (vm, mut cpu) = setup_tasks_delivery(model, wide, b"\x90\x90", ExceptionDelivery::InGuest);
    cpu.cpu.legacy.idtr.base = 0x3000;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    gate(&vm, 0x3010, B, 0x85);
    put(
        &vm,
        base(B) + if wide { 0x24 } else { 0x10 },
        if wide { 4 } else { 2 },
        flags,
    );
    vm.write_bytes(0x1800, b_code).unwrap();
    (vm, cpu)
}

#[test]
fn nmi_task_gate_with_if_set_takes_a_queued_irq_first() {
    // I286 PRM §9.6 p. 9-9: "If other interrupts remain enabled, they are
    // processed before the first instruction of the current interrupt
    // handler"; §9.2 p. 9-3 recommends IF = 0 in the NMI task for this reason.
    for (model, wide) in formats() {
        let (vm, mut cpu) = nmi_task(model, wide, 0x202, b"\xf4");
        // Vector 30h: a 16-bit interrupt gate to 8:1A00 (HLT).
        vm.write_bytes(0x3180, &[0x00, 0x1a, 8, 0, 0, 0x86, 0, 0])
            .unwrap();
        vm.write_bytes(0x1a00, b"\xf4").unwrap();
        cpu.cpu.flags.if_ = true;
        cpu.inject_irq(0x30);
        cpu.raise_nmi().unwrap();
        let run = cpu.run_steps(&vm, 4, StepOptions::default()).unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            run.stop()
        );
        assert_eq!(run.steps(), 1);
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1a01));
        assert_eq!(
            value(&vm, 0x8ffa, 2),
            0x1800,
            "IRQ frame: the NMI task's entry"
        );
        assert!(!cpu.has_pending_irq() && !cpu.has_pending_nmi());
        assert!(!cpu.nmi_ready(), "blocked in the NMI task");
    }
}

#[test]
fn a_task_return_iret_ends_nmi_blocking() {
    // I286 PRM §9.2 p. 9-3: the NMI task's IRET (NT = 1, a task return) is
    // the first IRET; the I80286 then services the remembered edge.
    for (model, wide) in formats() {
        let (vm, mut cpu) = nmi_task(model, wide, 2, b"\x90\xcf");
        cpu.raise_nmi().unwrap();
        steps(&mut cpu, &vm, 1); // the switch, then B's NOP
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1801));
        assert!(!cpu.nmi_ready());
        cpu.raise_nmi().unwrap();
        steps(&mut cpu, &vm, 1); // B's IRET returns to A
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (A, 0x1006));
        assert!(cpu.nmi_ready(), "the task return ended blocking");
        assert_eq!(cpu.has_pending_nmi(), model == CpuModel::I80286);
    }
}

#[test]
fn a_task_return_iret_through_a_device_ends_nmi_blocking_at_completion() {
    // spec.md §17.10: a bus handoff inside the task-return IRET does not end
    // blocking; its completion does (I286 PRM §9.2 p. 9-3, "the first
    // IRET"). The TSS page is a device here. An edge raised during the first
    // handoff leaves the word as a blocked edge, remembered (80286) or
    // ignored (80386); the 80286 latches it again at completion.
    for (model, wide) in formats() {
        let (mut vm, mut cpu) = nmi_task(model, wide, 2, b"\x90\xcf");
        cpu.raise_nmi().unwrap();
        steps(&mut cpu, &vm, 1); // the switch, then B's NOP
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (B, 0x1801));
        vm.unmap(0, 0x240000).unwrap();
        for (at, size, kind) in [
            (0, 0x4000, x86jit_core::RegionKind::Ram),
            (0x4000, 0x1000, x86jit_core::RegionKind::Trap),
            (0x5000, 0x23b000, x86jit_core::RegionKind::Ram),
        ] {
            vm.map(at, size, x86jit_core::Prot::RWX, kind).unwrap();
        }
        let line = cpu.nmi_line().unwrap();
        let remember = model == CpuModel::I80286;
        let mut handoffs = 0;
        loop {
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            match *run.stop() {
                StepStop::Exit(Exit::MmioRead { addr, size, .. }) => {
                    cpu.complete_mmio_read(value(&vm, addr, size.into()));
                }
                StepStop::Exit(Exit::MmioWrite {
                    addr, size, value, ..
                }) => {
                    put(&vm, addr, size.into(), value as u32);
                    cpu.complete_mmio_write();
                }
                StepStop::Exit(Exit::BudgetExhausted) => break,
                ref other => panic!("{other:?}"),
            }
            assert_eq!(run.steps(), 0);
            assert!(!cpu.nmi_ready());
            if handoffs == 1 {
                assert!(!line.is_raised(), "a blocked edge during the handoff");
            }
            if handoffs != 0 {
                assert_eq!(cpu.has_pending_nmi(), remember, "still blocked");
            } else {
                cpu.raise_nmi().unwrap();
            }
            handoffs += 1;
        }
        assert!(handoffs > 2, "{handoffs}");
        assert_eq!((cpu.cpu.legacy.tr.selector, cpu.cpu.rip), (A, 0x1006));
        assert!(cpu.nmi_ready(), "the completed task return ended blocking");
        assert_eq!(
            (cpu.has_pending_nmi(), line.is_raised()),
            (remember, remember)
        );
    }
}

#[test]
fn an_edited_state_abandons_an_nmi_task_switch_and_its_blocking() {
    // decision-24 and spec.md §17.10: an accepted NMI whose task switch is
    // abandoned by a host edit is consumed, and leaves no blocking behind.
    for (model, wide) in formats() {
        let (mut vm, mut cpu) =
            setup_tasks_delivery(model, wide, b"\x90", ExceptionDelivery::InGuest);
        cpu.cpu.legacy.idtr.base = 0x7000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        gate(&vm, 0x7010, B, 0x85);
        descriptor(&vm, 0x2038, 0x6000, 7, 0x82);
        descriptor(&vm, 0x6000, 0, 0xffff, 0x92);
        put(&vm, base(B) + if wide { 0x60 } else { 0x2a }, 2, 56);
        put(&vm, base(B) + if wide { 0x54 } else { 0x28 }, 2, 4);
        vm.write_bytes(0x1800, b"\x90\x90").unwrap();
        split_task_pages(&mut vm);
        retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Trap);
        cpu.raise_nmi().unwrap();
        for at in [0x6000, 0x6004] {
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(run.stop(), StepStop::Exit(Exit::MmioRead {addr, size:4, ..}) if *addr == at),
                "{:?}",
                run.stop()
            );
            cpu.complete_mmio_read(value(&vm, at, 4));
        }
        expect_task_refusal(
            &mut cpu,
            &vm,
            x86jit_core::UnsupportedSystemOperation::LegacyInstruction,
        );
        assert!(
            !cpu.has_pending_nmi() && !cpu.nmi_ready(),
            "accepted, blocked"
        );
        cpu.cpu.gpr[0] = 0xbeef;
        retag_task_page(&mut vm, 0x6000, x86jit_core::RegionKind::Ram);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rip, 0x1801);
        assert!(!cpu.has_pending_nmi(), "consumed");
        assert!(cpu.nmi_ready(), "no blocking left");
    }
}
