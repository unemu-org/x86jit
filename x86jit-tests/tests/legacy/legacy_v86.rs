//! V86 execution, transfer and boundary witnesses (v86 design §§2, 7 steps 1–5).
//! Architectural expectations use SDM Vol 3B §§23.2–23.3
//! pp. 23-5–17, 386 PRM Ch. 15 pp. 15-1–13, and 486 PRM Ch. 23 pp. 23-1–15.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use x86jit_core::fault::{ErrorCode, FaultAccess, Knowledge};
use x86jit_core::legacy::SystemSegment;
use x86jit_core::{
    CpuModel, Exit, FpuModel, MachineContextError, PortDir, SegmentCache, SegmentId, StepOptions,
    StepStop, UnsupportedSystemOperation, Vcpu, Vm,
};

const MODELS: [CpuModel; 3] = [
    CpuModel::I80386,
    CpuModel::I80486Dx,
    CpuModel::I80486DxCpuid,
];
const VM: u64 = 1 << 17;
const TSS: u64 = 0x4000;
const MAP: u64 = 0x80;

fn canonical(cpu: &mut Vcpu, segment: SegmentId, selector: u16) {
    match segment {
        SegmentId::Cs => cpu.cpu.cs = selector,
        SegmentId::Ds => cpu.cpu.ds = selector,
        SegmentId::Es => cpu.cpu.es = selector,
        SegmentId::Ss => cpu.cpu.ss = selector,
        SegmentId::Fs => cpu.cpu.fs = selector,
        SegmentId::Gs => cpu.cpu.gs = selector,
    }
    cpu.cpu.legacy.segments[segment as usize] = SegmentCache {
        base: u32::from(selector) << 4,
        limit: 0xffff,
        // ASSUMED internal labels, not guest descriptor contents (design §4.1).
        attributes: if segment == SegmentId::Cs { 0x9b } else { 0x93 },
        ..Default::default()
    };
}
fn world(model: CpuModel, code: &[u8], iopl: u8) -> (Vm, Vcpu) {
    let (vm, mut cpu) = fixture::setup(model, code);
    cpu.cpu.legacy.cpl = 3;
    cpu.cpu.legacy.reset_fetch = 0;
    cpu.cpu.rflags_system = VM | u64::from(iopl) << 12;
    for segment in [
        SegmentId::Cs,
        SegmentId::Ds,
        SegmentId::Es,
        SegmentId::Ss,
        SegmentId::Fs,
        SegmentId::Gs,
    ] {
        canonical(&mut cpu, segment, 0);
    }
    // No descriptor storage is addressable: ordinary execution needs no GDT/TSS.
    cpu.cpu.legacy.gdtr.base = 0xdead_0000;
    cpu.cpu.legacy.gdtr.limit = 0;
    cpu.cpu.gpr[4] = 0xabcd_8000;
    (vm, cpu)
}
fn run(cpu: &mut Vcpu, vm: &Vm) -> Result<(), Exit> {
    cpu.interpret_virtual8086_for_test(vm)
}
fn exception(cpu: &mut Vcpu, vm: &Vm, vector: u8, code: Option<u32>) -> Exit {
    let ip = cpu.cpu.rip;
    let regs = cpu.cpu.gpr;
    let segments = cpu.cpu.legacy.segments;
    let exit = run(cpu, vm).unwrap_err();
    assert!(
        matches!(exit, Exit::Exception { vector: v, .. } if v == vector),
        "{exit:?}"
    );
    let metadata = exit.fault_metadata().unwrap();
    assert_eq!(
        metadata.error_code(),
        code.map_or(ErrorCode::NoCode, |v| ErrorCode::bits(
            v,
            if vector == 14 { 7 } else { 0xffff }
        ))
    );
    assert_eq!(metadata.saved_rip(), ip);
    assert_eq!(cpu.cpu.rip, ip);
    assert_eq!(cpu.cpu.gpr, regs);
    assert_eq!(cpu.cpu.legacy.segments, segments);
    exit
}
fn bitmap(vm: &Vm, cpu: &mut Vcpu) {
    cpu.cpu.legacy.tr = SystemSegment {
        selector: 0x28,
        cache: SegmentCache {
            base: TSS as u32,
            limit: (MAP + 0x2000) as u32,
            attributes: 0x8b,
            ..Default::default()
        },
        ..Default::default()
    };
    vm.write_bytes(TSS + 0x66, &(MAP as u16).to_le_bytes())
        .unwrap();
    cpu.cpu.task_io_map_base = 0xffff; // Deliberately stale task-switch snapshot.
}
fn deny(vm: &Vm, port: u32) {
    vm.write_bytes(TSS + MAP + u64::from(port >> 3), &[1 << (port & 7)])
        .unwrap();
}
fn port(cpu: &mut Vcpu, vm: &Vm, number: u16, size: u8, dir: PortDir) {
    let exit = run(cpu, vm).unwrap_err();
    assert!(
        matches!(exit, Exit::PortIo { port, size: s, dir: d, .. } if port == number && s == size && d == dir),
        "{exit:?}"
    );
}

// 386 PRM §15.1.2 pp. 15-2–3: all segment loads use paragraphs, zero included;
// no selector privilege, presence or accessed-bit lookup is involved.
#[test]
fn segment_loads_are_canonical_and_descriptor_free() {
    for model in MODELS {
        for selector in [0u16, 1, 0x1234, 0xffff] {
            for (segment, mov, pop) in [
                (SegmentId::Ds, vec![0x8e, 0xd8], vec![0x1f]),
                (SegmentId::Es, vec![0x8e, 0xc0], vec![0x07]),
                (SegmentId::Ss, vec![0x8e, 0xd0], vec![0x17]),
                (SegmentId::Fs, vec![0x8e, 0xe0], vec![0x0f, 0xa1]),
                (SegmentId::Gs, vec![0x8e, 0xe8], vec![0x0f, 0xa9]),
            ] {
                for code in [mov, pop] {
                    let (vm, mut cpu) = world(model, &code, 0);
                    cpu.cpu.gpr[0] = u64::from(selector);
                    vm.write_bytes(0x8000, &selector.to_le_bytes()).unwrap();
                    run(&mut cpu, &vm).unwrap();
                    let expected = SegmentCache {
                        base: u32::from(selector) << 4,
                        limit: 0xffff,
                        attributes: 0x93,
                        ..Default::default()
                    };
                    assert_eq!(cpu.cpu.legacy.segments[segment as usize], expected);
                    assert_eq!(cpu.cpu.legacy.cpl, 3);
                }
            }
            for (segment, opcode) in [
                (SegmentId::Ds, vec![0xc5]),
                (SegmentId::Es, vec![0xc4]),
                (SegmentId::Ss, vec![0x0f, 0xb2]),
                (SegmentId::Fs, vec![0x0f, 0xb4]),
                (SegmentId::Gs, vec![0x0f, 0xb5]),
            ] {
                let mut code = opcode;
                code.extend([0x06, 0x00, 0x02]); // AX, far pointer [0200h].
                let (vm, mut cpu) = world(model, &code, 0);
                vm.write_bytes(0x200, &[0x78, 0x56, selector as u8, (selector >> 8) as u8])
                    .unwrap();
                run(&mut cpu, &vm).unwrap();
                assert_eq!(cpu.cpu.gpr[0], 0x5678);
                assert_eq!(
                    cpu.cpu.legacy.segments[segment as usize].base,
                    u32::from(selector) << 4
                );
            }
        }
    }
}

// SDM Vol 3B §23.2 p. 23-5; 386 PRM §§15.1.1–15.1.2 pp. 15-1–3.
#[test]
fn defaults_overrides_segment_overrides_and_high_linear_addresses() {
    for model in MODELS {
        for (code, expected) in [
            (&[0xb8, 0x34, 0x12][..], 0xface_1234),
            (&[0x66, 0xb8, 0x78, 0x56, 0x34, 0x12][..], 0x1234_5678),
        ] {
            let (vm, mut cpu) = world(model, code, 0);
            cpu.cpu.gpr[0] = 0xface_0000;
            run(&mut cpu, &vm).unwrap();
            assert_eq!(cpu.cpu.gpr[0], expected);
            assert_eq!(cpu.cpu.rip, 0x1000 + code.len() as u64);
        }
        for (segment, prefix) in [
            (SegmentId::Cs, 0x2e),
            (SegmentId::Ds, 0x3e),
            (SegmentId::Es, 0x26),
            (SegmentId::Ss, 0x36),
            (SegmentId::Fs, 0x64),
            (SegmentId::Gs, 0x65),
        ] {
            let (vm, mut cpu) = world(model, &[prefix, 0xa3, 0x00, 0x02], 0);
            canonical(&mut cpu, segment, 0x1234);
            // Moving CS also moves fetch; selectors with RPL != 3 remain CPL 3.
            if segment == SegmentId::Cs {
                vm.write_bytes(0x13340, &[prefix, 0xa3, 0x00, 0x02])
                    .unwrap();
            }
            cpu.cpu.gpr[0] = 0xabcd;
            run(&mut cpu, &vm).unwrap();
            assert_eq!(fixture::value(&vm, 0x12540, 2), 0xabcd);
        }
        let (vm, mut cpu) = world(model, &[0x67, 0xa1, 0x34, 0x12, 0, 0], 0);
        canonical(&mut cpu, SegmentId::Ds, 0x2222);
        vm.write_bytes(0x23454, &[0xcd, 0xab]).unwrap();
        run(&mut cpu, &vm).unwrap();
        assert_eq!(cpu.cpu.gpr[0], 0xabcd);
        let (vm, mut cpu) = world(model, &[0xa0, 0xff, 0xff], 0);
        canonical(&mut cpu, SegmentId::Ds, 0xffff);
        vm.write_bytes(0x10ffef, &[0x5a]).unwrap();
        vm.write_bytes(0x0ffef, &[0xa5]).unwrap();
        run(&mut cpu, &vm).unwrap();
        assert_eq!(cpu.cpu.gpr[0], 0x5a);
        cpu.cpu.rip = 0x1000;
        cpu.set_a20_enabled(false);
        run(&mut cpu, &vm).unwrap();
        assert_eq!(cpu.cpu.gpr[0], 0xa5);
    }
}

// SDM Vol 2A BOUND p. 3-106; Vol 3B Table 23-1 p. 23-6: #SS(0)/#GP(0).
#[test]
fn whole_operand_fetch_and_stack_limits_do_not_wrap() {
    for model in MODELS {
        for (code, vector) in [
            (&[0xa1, 0xff, 0xff][..], 13),
            (&[0x36, 0xa1, 0xff, 0xff][..], 12),
            (&[0x67, 0xa0, 0, 0, 1, 0][..], 13),
            (&[0x36, 0x67, 0xa0, 0, 0, 1, 0][..], 12),
            (&[0x66, 0xa1, 0xfd, 0xff][..], 13),
        ] {
            let (vm, mut cpu) = world(model, code, 0);
            exception(&mut cpu, &vm, vector, Some(0));
        }
        let (vm, mut cpu) = world(model, &[0x50], 0);
        cpu.cpu.gpr[4] = 0xabcd_0001;
        exception(&mut cpu, &vm, 12, Some(0));
        let (vm, mut cpu) = world(model, &[0x58], 0);
        cpu.cpu.gpr[4] = 0xabcd_ffff;
        exception(&mut cpu, &vm, 12, Some(0));
        for ip in [0xffff, 0x10000] {
            let (vm, mut cpu) = world(model, &[0x90], 0);
            cpu.cpu.rip = ip;
            vm.write_bytes(ip, &[0xb8, 0x34, 0x12]).unwrap();
            exception(&mut cpu, &vm, 13, Some(0));
        }
        let (vm, mut cpu) = world(model, &[0x50, 0x66, 0x50], 0);
        cpu.cpu.gpr[0] = 0x1234_5678;
        run(&mut cpu, &vm).unwrap();
        assert_eq!(cpu.cpu.gpr[4], 0xabcd_7ffe);
        run(&mut cpu, &vm).unwrap();
        assert_eq!(cpu.cpu.gpr[4], 0xabcd_7ffa);
        assert_eq!(fixture::value(&vm, 0x7ffa, 6), 0x5678_1234_5678);
    }
}

// SDM Vol 3B §23.2.7 p. 23-10; Vol 2A INT p. 3-469, IRET pp. 3-491–493;
// Vol 2B PUSHF pp. 4-528–529, POPF pp. 4-407–410: six sensitive families.
// Privilege before operands is the engine's ASSUMED policy (design §2.3).
#[test]
fn all_six_sensitive_families_guard_live_iopl_before_stack_or_transfer() {
    for model in MODELS {
        for iopl in 0..=3 {
            for code in [
                vec![0xfa],
                vec![0xfb],
                vec![0x9c],
                vec![0x66, 0x9c],
                vec![0x9d],
                vec![0x66, 0x9d],
                vec![0xcd, 3],
                vec![0xcd, 4],
                vec![0x66, 0xcd, 3],
                vec![0x66, 0xcd, 4],
                vec![0xcf],
                vec![0x66, 0xcf],
            ] {
                let (vm, mut cpu) = world(model, &code, iopl);
                cpu.cpu.gpr[4] = 0xabcd_ffff;
                if iopl < 3 {
                    exception(&mut cpu, &vm, 13, Some(0));
                } else if code.contains(&0xcd) {
                    exception(
                        &mut cpu,
                        &vm,
                        13,
                        Some(u32::from(*code.last().unwrap()) * 8 + 2),
                    );
                } else if code.contains(&0xcf) || code.contains(&0x9d) {
                    exception(&mut cpu, &vm, 12, Some(0));
                } else {
                    run(&mut cpu, &vm).unwrap();
                }
            }
        }
        // IOPL is read at execution, not captured in the retained IR.
        let (vm, mut cpu) = world(model, &[0xfa], 0);
        exception(&mut cpu, &vm, 13, Some(0));
        let lifts = cpu.paging_cache_for_test().2;
        cpu.cpu.rflags_system |= 0x3000;
        run(&mut cpu, &vm).unwrap();
        assert!(cpu.paging_cache_for_test().2 >= lifts);
        assert!(!cpu.cpu.flags.if_);
    }
}

// SDM Vol 2B PUSHF pp. 4-528–529; POPF Table 4-20 pp. 4-407–410.
#[test]
fn flags_images_and_loads_preserve_vm_iopl_and_model_bits() {
    for model in MODELS {
        for wide in [false, true] {
            let code = if wide {
                vec![0x66, 0x9c, 0x66, 0x9d]
            } else {
                vec![0x9c, 0x9d]
            };
            let (vm, mut cpu) = world(model, &code, 3);
            cpu.cpu.rflags_system |= 1 << 16;
            run(&mut cpu, &vm).unwrap();
            let sp = cpu.cpu.gpr[4] & 0xffff;
            assert_eq!(
                fixture::value(&vm, sp, if wide { 4 } else { 2 }) & 0x30000,
                0
            );
            let image = 0x0027_0203u32; // Attempt IOPL=0, VM/RF/AC/ID=1; IF/CF=1.
            vm.write_bytes(sp, &image.to_le_bytes()[..if wide { 4 } else { 2 }])
                .unwrap();
            run(&mut cpu, &vm).unwrap();
            assert_eq!(cpu.cpu.rflags_system & 0x33000, VM | 0x3000);
            assert!(cpu.cpu.flags.if_ && cpu.cpu.flags.cf);
            let expected = if !wide || model == CpuModel::I80386 {
                0
            } else if model == CpuModel::I80486Dx {
                1 << 18
            } else {
                (1 << 18) | (1 << 21)
            };
            assert_eq!(cpu.cpu.rflags_system & 0x240000, expected);
            assert_eq!(cpu.cpu.gpr[4], 0xabcd_8000);
        }
    }
}

// Per-instruction references: design §2.3. #UD is before a simultaneously
// invalid operand, including LLDT/LTR; LMSW and valid CR reads instead #GP(0).
#[test]
fn protected_only_and_privileged_instruction_matrix() {
    let ud = [
        vec![0x63, 0xc0],
        vec![0x0f, 0x02, 0x06, 0xff, 0xff],
        vec![0x0f, 0x03, 0xc0],
        vec![0x0f, 0x00, 0xe0],
        vec![0x0f, 0x00, 0xe8],
        vec![0x0f, 0x00, 0xd0],
        vec![0x0f, 0x00, 0xc0],
        vec![0x0f, 0x00, 0xd8],
        vec![0x0f, 0x00, 0xc8],
    ];
    let gp = [
        vec![0x0f, 0x01, 0xf0],
        vec![0x0f, 0x20, 0xc0],
        vec![0x0f, 0x20, 0xd0],
        vec![0x0f, 0x20, 0xd8],
        vec![0x0f, 0x22, 0xc0],
        vec![0x0f, 0x22, 0xd0],
        vec![0x0f, 0x22, 0xd8],
        vec![0x0f, 0x21, 0xc0],
        vec![0x0f, 0x23, 0xc0],
        vec![0x0f, 0x24, 0xf0],
        vec![0x0f, 0x26, 0xf0],
        vec![0x0f, 0x01, 0x16, 0xff, 0xff],
        vec![0x0f, 0x01, 0x1e, 0xff, 0xff],
        vec![0x0f, 0x06],
        vec![0xf4],
    ];
    for model in MODELS {
        for iopl in 0..=3 {
            for code in &ud {
                let (vm, mut cpu) = world(model, code, iopl);
                exception(&mut cpu, &vm, 6, None);
            }
            for code in &gp {
                let (vm, mut cpu) = world(model, code, iopl);
                exception(&mut cpu, &vm, 13, Some(0));
            }
            for code in [
                &[0x0f, 0x08][..],
                &[0x0f, 0x09],
                &[0x0f, 0x01, 0x3e, 0xff, 0xff],
            ] {
                let (vm, mut cpu) = world(model, code, iopl);
                exception(
                    &mut cpu,
                    &vm,
                    if model == CpuModel::I80386 { 6 } else { 13 },
                    if model == CpuModel::I80386 {
                        None
                    } else {
                        Some(0)
                    },
                );
            }
        }
        for code in [&[0x0f, 0x20, 0xc8][..], &[0xf0, 0x90], &[0xf0, 0xfa]] {
            let (vm, mut cpu) = world(model, code, 0);
            exception(&mut cpu, &vm, 6, None);
        }
        // 386 PRM LOCK pp. 17-100–101; SDM Vol 3B §23.2.7 p. 23-10.
        let (vm, mut cpu) = world(model, &[0xf0, 0x01, 0x06, 0, 2], 0);
        cpu.cpu.gpr[0] = 3;
        vm.write_bytes(0x200, &4u16.to_le_bytes()).unwrap();
        run(&mut cpu, &vm).unwrap();
        assert_eq!(fixture::value(&vm, 0x200, 2), 7);
        for code in [
            &[0x0f, 0x01, 0x06, 0, 2][..],
            &[0x0f, 0x01, 0x0e, 0, 2],
            &[0x0f, 0x01, 0xe0],
        ] {
            let (vm, mut cpu) = world(model, code, 0);
            run(&mut cpu, &vm).unwrap();
        }
    }
}

// SDM Vol 2A INT pp. 3-467–471: CC and taken CE bypass INT n's IOPL
// check. Invalid IDT entries fault; OF=0 completes normally.
#[test]
fn traps_and_ordinary_ret_do_not_gain_an_iopl_guard() {
    for model in MODELS {
        for code in [&[0xcc][..], &[0xce]] {
            let (vm, mut cpu) = world(model, code, 0);
            cpu.cpu.flags.of = true;
            exception(
                &mut cpu,
                &vm,
                13,
                Some(if code[0] == 0xcc { 26 } else { 34 }),
            );
        }
        let (vm, mut cpu) = world(model, &[0xce], 0);
        run(&mut cpu, &vm).unwrap();
        assert_eq!(cpu.cpu.rip, 0x1001);
        let (vm, mut cpu) = world(model, &[0xc3], 0);
        vm.write_bytes(0x8000, &0x1234u16.to_le_bytes()).unwrap();
        run(&mut cpu, &vm).unwrap();
        assert_eq!(cpu.cpu.rip, 0x1234);
        assert_eq!(cpu.cpu.gpr[4], 0xabcd_8002);
        let (vm, mut cpu) = world(model, &[0x62, 0x06, 0, 2], 0);
        cpu.cpu.gpr[0] = 3;
        vm.write_bytes(0x200, &[0, 0, 2, 0]).unwrap();
        exception(&mut cpu, &vm, 5, None);
    }
}

// SDM Vol 3B §23.2.8.1 p. 23-11; Vol 1 §20.5.2 pp. 20-4–5:
// bitmap at every IOPL, every port byte, no wrap at FFFFh, live TSS+66h.
#[test]
fn scalar_and_string_io_force_the_live_bitmap_at_every_iopl() {
    for model in MODELS {
        for iopl in 0..=3 {
            for size in [1u8, 2, 4] {
                for (opcode, dir, string, immediate) in [
                    (0xec, PortDir::In, false, false),
                    (0xee, PortDir::Out, false, false),
                    (0xe4, PortDir::In, false, true),
                    (0xe6, PortDir::Out, false, true),
                    (0x6c, PortDir::In, true, false),
                    (0x6e, PortDir::Out, true, false),
                ] {
                    let mut code = if size == 4 { vec![0x66] } else { vec![] };
                    code.push(opcode + u8::from(size != 1));
                    if immediate {
                        code.push(0x27);
                    }
                    for denied in [false, true] {
                        let (vm, mut cpu) = world(model, &code, iopl);
                        bitmap(&vm, &mut cpu);
                        cpu.cpu.gpr[2] = 0x27;
                        cpu.cpu.gpr[6] = 0x200;
                        cpu.cpu.gpr[7] = 0x300;
                        if denied {
                            deny(&vm, 0x27 + u32::from(size - 1));
                            exception(&mut cpu, &vm, 13, Some(0));
                        } else {
                            port(&mut cpu, &vm, 0x27, size, dir);
                            if string {
                                if dir == PortDir::Out {
                                    assert_eq!(cpu.cpu.gpr[6], 0x200 + u64::from(size));
                                } else {
                                    cpu.complete_port_in(0x1234_5678);
                                    run(&mut cpu, &vm).unwrap();
                                    assert_eq!(
                                        fixture::value(&vm, 0x300, usize::from(size)),
                                        0x1234_5678 & ((1u64 << (size * 8)) - 1)
                                    );
                                    assert_eq!(cpu.cpu.gpr[7], 0x300 + u64::from(size));
                                }
                            }
                        }
                    }
                }
            }
            for bad_tss in [0, 1, 2] {
                let (vm, mut cpu) = world(model, &[0xec], iopl);
                bitmap(&vm, &mut cpu);
                match bad_tss {
                    0 => cpu.cpu.legacy.tr.cache.attributes = 0x83, // 16-bit TSS has no bitmap.
                    1 => cpu.cpu.legacy.tr.cache.limit = 0x66,
                    _ => vm
                        .write_bytes(TSS + 0x66, &0xffffu16.to_le_bytes())
                        .unwrap(),
                }
                exception(&mut cpu, &vm, 13, Some(0));
            }
            let (vm, mut cpu) = world(model, &[0xed], iopl);
            bitmap(&vm, &mut cpu);
            cpu.cpu.gpr[2] = 0xffff;
            deny(&vm, 0x10000);
            exception(&mut cpu, &vm, 13, Some(0));
            for op in [0x6c, 0x6d, 0x6e, 0x6f] {
                let (vm, mut cpu) = world(model, &[0xf3, op], iopl);
                cpu.cpu.gpr[1] = 0x1234_0000;
                run(&mut cpu, &vm).unwrap(); // CX=0: no TSS and no port access.
                assert_eq!(cpu.cpu.rip, 0x1002);
            }
        }
    }
}

// 386 PRM §15.1.2 p. 15-3, SDM Vol 3B §§23.2.3–23.2.4 pp. 23-7–8:
// all V86 explicit accesses, including fetch and stack, are user accesses.
#[test]
fn paged_fetch_data_and_stack_are_user_accesses_even_at_iopl_three() {
    for model in MODELS {
        for (code, at, write, fetch, stack) in [
            (&[0xa1, 0, 0x60][..], 0x6000, false, false, false),
            (&[0xa3, 0, 0x60][..], 0x6000, true, false, false),
            (&[0x90][..], 0x1000, false, true, false),
            (&[0x50][..], 0x7000, true, false, true),
        ] {
            for present in [false, true] {
                let (vm, mut cpu) = world(model, code, 3);
                cpu.cpu.legacy.cr0 |= 1 << 31;
                cpu.cpu.legacy.cr3 = 0x2000;
                vm.write_bytes(0x2000, &0x3007u32.to_le_bytes()).unwrap();
                for page in 0..16u32 {
                    vm.write_bytes(
                        0x3000 + u64::from(page * 4),
                        &((page * 0x1000) | 7).to_le_bytes(),
                    )
                    .unwrap();
                }
                vm.write_bytes(
                    0x3000 + (at >> 12) * 4,
                    &(at as u32 | if present { 3 } else { 6 }).to_le_bytes(),
                )
                .unwrap();
                if stack {
                    cpu.cpu.gpr[4] = 0xabcd_7002;
                }
                let exit = exception(
                    &mut cpu,
                    &vm,
                    14,
                    Some(u32::from(present) | u32::from(write) << 1 | 4),
                );
                assert_eq!(cpu.cpu.legacy.cr2, at as u32);
                let m = exit.fault_metadata().unwrap();
                assert_eq!(
                    m.access(),
                    Knowledge::Known(if fetch {
                        FaultAccess::Execute
                    } else if write {
                        FaultAccess::Write
                    } else {
                        FaultAccess::Read
                    })
                );
            }
        }
    }
}

// SDM Vol 1 §8.1.10 Figs. 8-9–8-12 pp. 8-11–12;
// 486 PRM §15.1.6 pp. 15-8–10: real-format pointers under PE=VM=1.
#[test]
fn x87_environment_and_save_images_use_real_formats_at_both_sizes() {
    for model in MODELS {
        for wide in [false, true] {
            for full in [false, true] {
                let mut code = if wide { vec![0x66] } else { vec![] };
                code.extend([if full { 0xdd } else { 0xd9 }, 0x36, 0, 2]);
                let (vm, mut cpu) = world(model, &code, 0);
                cpu.set_fpu_model(FpuModel::Present);
                cpu.cpu.fpu_fip = 0x2345;
                cpu.cpu.fpu_fcs = 0x1234;
                cpu.cpu.fpu_fdp = 0x6789;
                cpu.cpu.fpu_fds = 0x4321;
                cpu.cpu.fpu_fop = 0x345;
                let size = match (wide, full) {
                    (false, false) => 14,
                    (true, false) => 28,
                    (false, true) => 94,
                    (true, true) => 108,
                };
                vm.write_bytes(0x200, &vec![0xa5; size + 1]).unwrap();
                run(&mut cpu, &vm).unwrap();
                if wide {
                    assert_eq!(fixture::value(&vm, 0x20c, 4), 0x4685);
                    assert_eq!(fixture::value(&vm, 0x210, 4), 0x1345);
                    assert_eq!(fixture::value(&vm, 0x214, 4), 0x9999);
                    assert_eq!(fixture::value(&vm, 0x218, 4), 0x4000);
                } else {
                    assert_eq!(fixture::value(&vm, 0x206, 2), 0x4685);
                    assert_eq!(fixture::value(&vm, 0x208, 2), 0x1345);
                    assert_eq!(fixture::value(&vm, 0x20a, 2), 0x9999);
                    assert_eq!(fixture::value(&vm, 0x20c, 2), 0x4000);
                }
                assert_eq!(fixture::value(&vm, 0x200 + size as u64, 1), 0xa5);
            }
        }
    }
}

#[test]
fn public_entry_validates_embedder_supplied_canonical_images() {
    for model in MODELS {
        let (vm, mut cpu) = world(model, &[0x90; 8], 3);
        fixture::steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rip, 0x1001);
        for segment in [
            SegmentId::Cs,
            SegmentId::Ds,
            SegmentId::Es,
            SegmentId::Ss,
            SegmentId::Fs,
            SegmentId::Gs,
        ] {
            let before = cpu.cpu.legacy.segments[segment as usize];
            for edit in 0..5 {
                let cache = &mut cpu.cpu.legacy.segments[segment as usize];
                match edit {
                    0 => cache.limit = u32::MAX,
                    1 => cache.base += 16,
                    2 => cache.attributes |= 0x4000,
                    3 => cache.attributes ^= 1,
                    _ => cache.unusable = 1,
                }
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(r.steps(), 0);
                assert!(matches!(
                    r.stop(),
                    StepStop::Exit(Exit::UnsupportedSystemContext { .. })
                ));
                cpu.cpu.legacy.segments[segment as usize] = before;
            }
        }
        for (segment, selector) in [(SegmentId::Cs, 0x10), (SegmentId::Ss, 0x20)] {
            // V86 paragraph values need no RPL/CPL pairing, including through
            // the public coherent cache setter (SDM Vol 3B §23.2 p. 23-5).
            let cache = SegmentCache {
                base: u32::from(selector) << 4,
                limit: 0xffff,
                attributes: if segment == SegmentId::Cs { 0x9b } else { 0x93 },
                ..Default::default()
            };
            cpu.cpu
                .set_legacy_segment(vm.legacy_cpu_model().unwrap(), segment, selector, cache)
                .unwrap();
        }
        for (pe, cpl, reset, expected) in [
            (1, 0, 0, MachineContextError::InvalidPrivilege),
            (0, 0, 0, MachineContextError::Virtual8086Unsupported),
            (1, 3, 1, MachineContextError::InvalidStorage),
        ] {
            cpu.cpu.legacy.cr0 = pe;
            cpu.cpu.legacy.cpl = cpl;
            cpu.cpu.legacy.reset_fetch = reset;
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 0);
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::UnsupportedSystemContext {
                operation: UnsupportedSystemOperation::InvalidLegacyContext(e), ..
            }) if *e == expected),
                "{:?}",
                r.stop()
            );
        }
    }
    let (vm, mut cpu) = world(CpuModel::I80286, &[0x90], 3);
    let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(r.steps(), 0);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::UnsupportedSystemContext {
            operation: UnsupportedSystemOperation::InvalidLegacyContext(
                MachineContextError::Virtual8086Unsupported
            ),
            ..
        })
    ));
}

#[test]
fn i486_alignment_boundary_is_retained_in_v86() {
    // Intel486 PRM §4.1.3 p. 4-7: AM+AC at CPL=3 enables #AC.
    // V86 §9/Q5 deliberately retains the typed entry refusal instead.
    for model in [CpuModel::I80486Dx, CpuModel::I80486DxCpuid] {
        for paged in [false, true] {
            for (am, ac) in [(true, false), (false, true), (true, true)] {
                let (vm, mut cpu) = world(model, &[0x90], 3);
                if paged {
                    identity_paging(&vm, &mut cpu);
                }
                cpu.cpu.legacy.cr0 |= u32::from(am) << 18;
                cpu.cpu.rflags_system |= u64::from(ac) << 18;
                let before = format!("{:?}", cpu.cpu);
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                if am && ac {
                    assert_eq!(r.steps(), 0);
                    assert!(matches!(
                        r.stop(),
                        StepStop::Exit(Exit::UnsupportedSystemContext {
                            operation: UnsupportedSystemOperation::LegacyAlignmentCheck,
                            ..
                        })
                    ));
                    assert_eq!(format!("{:?}", cpu.cpu), before);
                    assert_eq!(cpu.retired_instructions(), 0);
                } else {
                    assert_eq!(r.steps(), 1, "{:?}", r.stop());
                    assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
                }
            }
        }
    }
}

// Intel486 PRM §§4.1.3, 19.2.4 pp. 4-7, 19-4: NE selects internal #MF
// versus FERR#/IGNNE#. V86 §9/Q5 keeps internal routing even with NE=0
// until that independent facility is implemented; this is an engine boundary.
#[test]
fn public_i486_v86_keeps_internal_numeric_routing_even_with_ne_clear() {
    use x86jit_core::{NumericErrorConfigError, NumericErrorRoute};
    for model in [CpuModel::I80486Dx, CpuModel::I80486DxCpuid] {
        for paged in [false, true] {
            for ne in [false, true] {
                let (vm, mut cpu) = world(model, &[0x9b], 3); // WAIT
                if paged {
                    identity_paging(&vm, &mut cpu);
                }
                cpu.cpu.legacy.cr0 |= u32::from(ne) << 5;
                cpu.set_fpu_model(FpuModel::Present);
                assert_eq!(
                    cpu.set_numeric_error_route(NumericErrorRoute::External),
                    Err(NumericErrorConfigError::UnsupportedModel)
                );
                assert_eq!(cpu.numeric_error_route(), NumericErrorRoute::Internal);
                cpu.cpu.fpu_cw = 0x37b; // Unmask divide-by-zero.
                cpu.cpu.fpu_sw = 0x84; // ES + ZE pending.
                fixture::fault(&mut cpu, &vm, 16, None);
                assert_eq!(cpu.cpu.legacy.cr0 & (1 << 5), u32::from(ne) << 5);
                assert_eq!(cpu.retired_instructions(), 0);
            }
        }
    }
}

// V86 §4.1/Q3 and paging §3.7: a checkpoint owns CpuState and RAM,
// not derived translations. Restore at a completed run boundary and invalidate
// even when CR0/CR3 are unchanged. SDM Vol 3B §23.2.2 p. 23-7 and
// 386 PRM §15.1.2 pp. 15-2–3 supply paragraph addressing and the 64-KiB limit.
#[test]
fn public_v86_boundary_checkpoint_restores_cpu_ram_and_paging() {
    for model in MODELS {
        for paged in [false, true] {
            // MOV AX,[6000h]; INC word [6000h]; MOV BX,[6000h].
            let (vm, mut cpu) = world(
                model,
                &[0xa1, 0, 0x60, 0xff, 0x06, 0, 0x60, 0x8b, 0x1e, 0, 0x60],
                3,
            );
            if paged {
                identity_paging(&vm, &mut cpu);
                dwords(&vm, 0x11000 + 6 * 4, &[0x6007]);
                dwords(&vm, 0x11000 + 7 * 4, &[0x7007]);
            }
            vm.write_bytes(0x6000, &41u16.to_le_bytes()).unwrap();
            vm.write_bytes(0x7000, &99u16.to_le_bytes()).unwrap();
            fixture::steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[0] as u16, 41);
            let checkpoint = cpu.cpu.clone();
            let mut ram = vec![0; 0x20000];
            vm.read_bytes(0, &mut ram).unwrap();
            fixture::steps(&mut cpu, &vm, 2);
            let expected = format!("{:?}", cpu.cpu);
            assert_eq!(cpu.cpu.gpr[3] as u16, 42);

            cpu.cpu = checkpoint.clone();
            if paged {
                // Same CR3, different PTE and physical operand backing.
                dwords(&vm, 0x11000 + 6 * 4, &[0x7007]);
            } else {
                canonical(&mut cpu, SegmentId::Ds, 0x100);
            }
            cpu.invalidate_paging();
            fixture::steps(&mut cpu, &vm, 2);
            assert_eq!(cpu.cpu.gpr[3] as u16, 100);
            assert_eq!(cpu.cpu.legacy.cr0, checkpoint.legacy.cr0);
            assert_eq!(cpu.cpu.legacy.cr3, checkpoint.legacy.cr3);

            vm.write_bytes(0, &ram).unwrap();
            cpu.cpu = checkpoint;
            let generation = cpu.paging_cache_for_test().0;
            cpu.invalidate_paging();
            if paged {
                assert!(cpu.paging_cache_for_test().0 > generation);
                assert_eq!(cpu.paging_cache_for_test().1, 0);
            }
            fixture::steps(&mut cpu, &vm, 2);
            assert_eq!(format!("{:?}", cpu.cpu), expected);
            assert_eq!(fixture::value(&vm, 0x6000, 2), 42);
            assert_eq!(fixture::value(&vm, 0x7000, 2), 99);
            // Execution diagnostics are deliberately not rewound by CpuState.
            assert_eq!(cpu.retired_instructions(), 7);
        }
    }
}

#[test]
fn public_paged_v86_run_boundaries_retain_fetch_proofs() {
    for model in MODELS {
        let (vm, mut cpu) = world(model, &[0xeb, 0xfe], 3);
        identity_paging(&vm, &mut cpu);
        fixture::steps(&mut cpu, &vm, 1);
        let warm = cpu.paging_cache_for_test();
        assert_eq!(warm.1, 1);
        assert_eq!(warm.2, 1);
        for budget in [1, 2, 17] {
            fixture::steps(&mut cpu, &vm, budget);
            assert_eq!(cpu.paging_cache_for_test(), warm);
        }
        let checkpoint = cpu.cpu.clone();
        cpu.cpu = checkpoint;
        cpu.invalidate_paging();
        assert_eq!(cpu.paging_cache_for_test().1, 0);
        fixture::steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.paging_cache_for_test().2, warm.2 + 1);
    }
}

// 386 PRM §§15.1.2, 15.3 pp. 15-2–6: ordinary far transfers use
// paragraphs; operand selectors never name protected call gates or tasks.
#[test]
fn far_jump_call_and_return_keep_paragraph_segments_and_cpl_three() {
    for model in MODELS {
        for wide in [false, true] {
            for call in [false, true] {
                let mut code = if wide { vec![0x66] } else { vec![] };
                code.push(if call { 0x9a } else { 0xea });
                code.extend([0x00, 0x02]);
                if wide {
                    code.extend([0, 0]);
                }
                code.extend([0x34, 0x12]);
                let (vm, mut cpu) = world(model, &code, 0);
                vm.write_bytes(0x12540, if wide { &[0x66, 0xcb] } else { &[0xcb] })
                    .unwrap();
                run(&mut cpu, &vm).unwrap();
                assert_eq!(cpu.cpu.rip, 0x200);
                assert_eq!(cpu.cpu.cs, 0x1234);
                assert_eq!(
                    cpu.cpu.legacy.segments[0],
                    SegmentCache {
                        base: 0x12340,
                        limit: 0xffff,
                        attributes: 0x9b,
                        ..Default::default()
                    }
                );
                assert_eq!(cpu.cpu.legacy.cpl, 3);
                if call {
                    run(&mut cpu, &vm).unwrap();
                    assert_eq!(cpu.cpu.cs, 0);
                    assert_eq!(cpu.cpu.rip, 0x1000 + code.len() as u64);
                    assert_eq!(cpu.cpu.gpr[4], 0xabcd_8000);
                }
            }
        }
    }
}

// 386 PRM §§15.1.1–15.1.2 pp. 15-1–3: REP/address overrides retain
// real-style bases with full-operand limits and 16-bit stack arithmetic.
#[test]
fn strings_and_enter_use_canonical_checked_caches() {
    for model in MODELS {
        for address32 in [false, true] {
            let mut code = if address32 { vec![0x67] } else { vec![] };
            code.extend([0xf3, 0xa5]); // REP MOVSW
            let (vm, mut cpu) = world(model, &code, 0);
            canonical(&mut cpu, SegmentId::Ds, 0x100);
            canonical(&mut cpu, SegmentId::Es, 0x200);
            cpu.cpu.gpr[1] = 1;
            cpu.cpu.gpr[6] = 0x200;
            cpu.cpu.gpr[7] = 0x300;
            vm.write_bytes(0x1200, &[0x34, 0x12]).unwrap();
            run(&mut cpu, &vm).unwrap();
            assert_eq!(fixture::value(&vm, 0x2300, 2), 0x1234);
            assert_eq!(cpu.cpu.gpr[1], 0);
            let (vm, mut cpu) = world(model, &code, 0);
            cpu.cpu.gpr[1] = 1;
            cpu.cpu.gpr[6] = 0xffff;
            cpu.cpu.gpr[7] = 0x300;
            exception(&mut cpu, &vm, 13, Some(0));
        }
        let (vm, mut cpu) = world(model, &[0xc8, 8, 0, 0], 0);
        cpu.cpu.gpr[5] = 0x1234_5678;
        run(&mut cpu, &vm).unwrap();
        assert_eq!(cpu.cpu.gpr[4], 0xabcd_7ff6);
        assert_eq!(cpu.cpu.gpr[5], 0x1234_7ffe);
        assert_eq!(fixture::value(&vm, 0x7ffe, 2), 0x5678);
    }
}

// SDM Vol 3B §23.2.3 p. 23-7; 386 PRM §15.1.2 p. 15-3:
// linear addresses above 1 MiB are paged before the physical A20 gate.
#[test]
fn high_linear_paging_precedes_the_physical_a20_gate() {
    for model in MODELS {
        let (vm, mut cpu) = world(model, &[0xa0, 0xff, 0xff], 3);
        canonical(&mut cpu, SegmentId::Ds, 0xffff);
        cpu.cpu.legacy.cr0 |= 1 << 31;
        cpu.cpu.legacy.cr3 = 0x2000;
        cpu.set_a20_enabled(false);
        vm.write_bytes(0x2000, &0x3007u32.to_le_bytes()).unwrap();
        vm.write_bytes(0x3004, &0x1007u32.to_le_bytes()).unwrap();
        vm.write_bytes(0x343c, &0x9007u32.to_le_bytes()).unwrap(); // linear page 10Fh
        vm.write_bytes(0x9fef, &[0x5a]).unwrap();
        run(&mut cpu, &vm).unwrap();
        assert_eq!(cpu.cpu.gpr[0], 0x5a);
    }
}

const HANDLER: u64 = 0x1800;
const IDT: u64 = 0x3000;
const ENTRY_IMAGE: [u32; 9] = [
    0x1000,
    0x20,
    0x33203,
    0xabcd_7000,
    0x30,
    0x40,
    0x50,
    0x60,
    0x70,
];

fn dwords(vm: &Vm, address: u64, words: &[u32]) {
    let bytes: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
    vm.write_bytes(address, &bytes).unwrap();
}
fn gate(vm: &Vm, vector: u8, kind: u8) {
    dwords(
        vm,
        IDT + u64::from(vector) * 8,
        &[(8 << 16) | HANDLER as u32, u32::from(kind) << 8],
    );
}
fn transfers(
    model: CpuModel,
    delivery: x86jit_core::ExceptionDelivery,
    code: &[u8],
    big_stack: bool,
) -> (Vm, Vcpu) {
    let (vm, mut cpu) = fixture::setup_delivery(model, &[0xcf], delivery);
    fixture::descriptor(&vm, 0x2008, 0, 0xffff, 0x409b);
    fixture::descriptor(&vm, 0x2010, 0, 0xffff, 0x4093);
    cpu.cpu.cs = 8;
    cpu.cpu.ss = 0x10;
    cpu.cpu.legacy.segments[SegmentId::Cs as usize] = SegmentCache {
        limit: 0xffff,
        attributes: 0x409b,
        ..Default::default()
    };
    cpu.cpu.legacy.segments[SegmentId::Ss as usize] = SegmentCache {
        limit: 0xffff,
        attributes: if big_stack { 0x4093 } else { 0x93 },
        ..Default::default()
    };
    cpu.cpu.legacy.idtr.base = IDT as u32;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    bitmap(&vm, &mut cpu);
    dwords(&vm, TSS + 4, &[0x9000, 0x10]);
    dwords(&vm, 0x8000, &ENTRY_IMAGE);
    vm.write_bytes(0x1200, code).unwrap();
    vm.write_bytes(HANDLER, &[0x90, 0xcf]).unwrap();
    cpu.enable_delivery_log_with(
        16,
        x86jit_core::DeliveryLogOptions::default().registers(true),
    )
    .unwrap();
    (vm, cpu)
}
fn transfer_run(cpu: &mut Vcpu, vm: &Vm, steps: u64) -> x86jit_core::StepRun {
    cpu.run_steps(vm, steps, StepOptions::default()).unwrap()
}
fn transfer_steps(cpu: &mut Vcpu, vm: &Vm, count: u64) {
    let result = transfer_run(cpu, vm, count);
    assert_eq!(result.steps(), count, "{:?}", result.stop());
    assert!(
        matches!(result.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{:?}",
        result.stop()
    );
}
fn assert_v86(cpu: &Vcpu, ip: u64, flags: u64) {
    assert_eq!(cpu.cpu.legacy.cpl, 3);
    assert_eq!(cpu.cpu.rip, ip);
    assert_eq!(cpu.cpu.gpr[4], 0xabcd_7000);
    assert_eq!(cpu.cpu.rflags_system & 0x37000, flags & 0x37000);
    assert!(cpu.cpu.flags.if_);
    for (id, selector) in [
        (SegmentId::Cs, 0x20u16),
        (SegmentId::Ss, 0x30),
        (SegmentId::Es, 0x40),
        (SegmentId::Ds, 0x50),
        (SegmentId::Fs, 0x60),
        (SegmentId::Gs, 0x70),
    ] {
        let cache = cpu.cpu.legacy.segments[id as usize];
        assert_eq!(
            cache,
            SegmentCache {
                base: u32::from(selector) << 4,
                limit: 0xffff,
                attributes: if id == SegmentId::Cs { 0x9b } else { 0x93 },
                ..Default::default()
            }
        );
    }
}
fn assert_handler(
    cpu: &Vcpu,
    vm: &Vm,
    saved_ip: u32,
    saved_flags: u32,
    error: Option<u32>,
    interrupt: bool,
) {
    assert_eq!(cpu.cpu.legacy.cpl, 0);
    assert_eq!(cpu.cpu.cs, 8);
    assert_eq!(cpu.cpu.ss, 0x10);
    assert_eq!(cpu.cpu.rflags_system & 0x34100, 0);
    assert_eq!(cpu.cpu.flags.if_, !interrupt);
    let mut words = vec![
        saved_ip,
        0x20,
        saved_flags,
        0xabcd_7000,
        0x30,
        0x40,
        0x50,
        0x60,
        0x70,
    ];
    if let Some(error) = error {
        words.insert(0, error);
    }
    let sp = 0x9000 - words.len() as u64 * 4;
    assert_eq!(cpu.cpu.gpr[4], sp);
    for (i, value) in words.into_iter().enumerate() {
        assert_eq!(fixture::value(vm, sp + i as u64 * 4, 4), u64::from(value));
    }
    for (id, selector) in [
        (SegmentId::Ds, cpu.cpu.ds),
        (SegmentId::Es, cpu.cpu.es),
        (SegmentId::Fs, cpu.cpu.fs),
        (SegmentId::Gs, cpu.cpu.gs),
    ] {
        assert_eq!(selector, 0);
        assert_eq!(
            cpu.cpu.legacy.segments[id as usize],
            SegmentCache {
                unusable: 1,
                ..Default::default()
            }
        );
    }
}

// SDM Vol 2A §3.3 IRET pp. 3-492–493; 486 PRM §26 IRET p. 26-174.
#[test]
fn iretd_enters_v86_from_either_stack_width_without_descriptor_lookups() {
    for model in MODELS {
        for big_stack in [false, true] {
            let (vm, mut cpu) = transfers(
                model,
                x86jit_core::ExceptionDelivery::Outward,
                &[0x90],
                big_stack,
            );
            cpu.cpu.legacy.gdtr.base = 0xdead0000;
            cpu.cpu.legacy.gdtr.limit = 0;
            // Model-specific AC/ID, plus TF/RF/IF, all loaded by IRETD.
            let image = ENTRY_IMAGE[2] | 0x240100;
            dwords(&vm, 0x8008, &[image]);
            transfer_steps(&mut cpu, &vm, 1);
            assert_v86(&cpu, 0x1000, u64::from(image));
            let model_bits = match model {
                CpuModel::I80486Dx => 0x40000,
                CpuModel::I80486DxCpuid => 0x240000,
                _ => 0,
            };
            assert_eq!(cpu.cpu.rflags_system & 0x240100, model_bits | 0x100);
            assert!(cpu.delivery_log().unwrap().is_empty());
            let stopped = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(stopped.steps(), 1);
            assert!(matches!(
                stopped.stop(),
                StepStop::Exit(Exit::Exception { vector: 1, .. })
            ));
        }
    }
}

// SDM Vol 3B §23.3.1.1 pp. 23-12–13, Fig. 23-4; 386 PRM §15.3.2 pp. 15-6–7.
#[test]
fn software_and_hardware_round_trips_save_nine_slots_and_log_v86_origins() {
    use x86jit_core::{DeliveryKind, ExceptionDelivery};
    for model in MODELS {
        for policy in [ExceptionDelivery::Outward, ExceptionDelivery::InGuest] {
            for hardware in [false, true] {
                for trap in [false, true] {
                    let (vm, mut cpu) = transfers(model, policy, &[0xcd, 0x40, 0x90], true);
                    gate(&vm, 0x40, if trap { 0xef } else { 0xee });
                    transfer_steps(&mut cpu, &vm, 1);
                    if hardware {
                        cpu.inject_irq(0x40);
                    }
                    transfer_steps(&mut cpu, &vm, 1);
                    let ip = if hardware { 0x1000 } else { 0x1002 };
                    assert_handler(&cpu, &vm, ip, ENTRY_IMAGE[2], None, !trap);
                    let log = cpu.delivery_log().unwrap();
                    assert_eq!(log.len(), 1);
                    let record = log.iter().next().unwrap();
                    assert!(record.virtual_8086_before);
                    assert!(!record.virtual_8086_after);
                    assert_eq!(
                        (record.saved_cs, record.saved_ip, record.saved_linear),
                        (0x20, ip, ip + 0x200)
                    );
                    assert_eq!((record.cpl_before, record.cpl_after), (3, 0));
                    assert_eq!(
                        record.kind,
                        if hardware {
                            DeliveryKind::Irq
                        } else {
                            DeliveryKind::Software
                        }
                    );
                    let regs = record.regs.as_ref().unwrap();
                    assert_eq!(regs.esp, 0xabcd_7000);
                    assert_eq!(regs.flags & 0x30000, 0);
                    if !hardware {
                        transfer_steps(&mut cpu, &vm, 1);
                    } // Handler NOP.
                    transfer_steps(&mut cpu, &vm, 1);
                    assert_v86(&cpu, u64::from(ip), u64::from(ENTRY_IMAGE[2]));
                    assert_eq!(cpu.delivery_log().unwrap().len(), 1); // IRET is not a delivery.
                }
            }
        }
    }
}

// SDM Vol 2A §3.3 IRET pp. 3-491–492: three slots, NT does not select a task return.
#[test]
fn v86_iret_uses_three_slots_preserves_iopl_vm_and_restores_rf() {
    for model in MODELS {
        for wide in [false, true] {
            let code = if wide { &[0x66, 0xcf][..] } else { &[0xcf][..] };
            let (vm, mut cpu) = world(model, code, 3);
            cpu.cpu.rflags_system |= 0x4000 | 0x10000;
            if wide {
                dwords(&vm, 0x8000, &[0x1234, 0x5678, 0x10203]);
            } else {
                vm.write_bytes(0x8000, &[0x34, 0x12, 0x78, 0x56, 3, 2])
                    .unwrap();
            }
            run(&mut cpu, &vm).unwrap();
            assert_eq!(cpu.cpu.rip, 0x1234);
            assert_eq!(cpu.cpu.cs, 0x5678);
            assert_eq!(cpu.cpu.legacy.segments[0].base, 0x56780);
            assert_eq!(cpu.cpu.gpr[4], 0xabcd_8000 + if wide { 12 } else { 6 });
            assert_eq!(cpu.cpu.rflags_system & 0x37000, VM | 0x3000 | 0x10000);
            assert!(cpu.cpu.flags.if_);
        }
    }
}

// SDM Vol 3B §23.3.1.1 pp. 23-12–13; Vol 2A INT p. 3-469 (#GP(0));
// I386 PRM §12.3.1.1 p. 12-7: a fault saves RF=1 and the faulting IP.
#[test]
fn exceptions_round_trip_with_error_codes_under_both_delivery_policies() {
    use x86jit_core::ExceptionDelivery;
    for model in MODELS {
        for policy in [ExceptionDelivery::Outward, ExceptionDelivery::InGuest] {
            for (code, vector, error) in [
                (&[0xcd, 0x40][..], 13, Some(0)),
                (&[0x0f, 0x0b][..], 6, None),
                (&[0xf7, 0xf1][..], 0, None),
            ] {
                let (vm, mut cpu) = transfers(model, policy, code, true);
                gate(&vm, vector, 0x8e); // Exceptions ignore gate DPL.
                transfer_steps(&mut cpu, &vm, 1);
                cpu.cpu.rflags_system &= !0x13000; // Intercept INT n with live RF clear.
                let flags = ENTRY_IMAGE[2] & !0x13000;
                let retired = cpu.retired_instructions();
                if policy == ExceptionDelivery::Outward {
                    let result = transfer_run(&mut cpu, &vm, 1);
                    assert_eq!(result.steps(), 0);
                    let StepStop::Exit(exit @ Exit::Exception { vector: actual, .. }) =
                        result.stop()
                    else {
                        panic!("{:?}", result.stop());
                    };
                    assert_eq!(*actual, vector);
                    assert_eq!(cpu.retired_instructions(), retired);
                    assert!(cpu.delivery_log().unwrap().is_empty());
                    // The test-only bridge supplies the outward exception to the same
                    // delivery seam; normal Outward execution leaves it with the caller.
                    cpu.deliver_virtual8086_fault_for_test(&vm, exit).unwrap();
                } else {
                    transfer_steps(&mut cpu, &vm, 1); // Handler NOP, fault takes no step.
                    assert_eq!(cpu.cpu.rip, HANDLER + 1);
                }
                assert_handler(&cpu, &vm, 0x1000, flags | 0x10000, error, true);
                let record = cpu.delivery_log().unwrap().iter().next().unwrap();
                assert_eq!(record.error_code, error);
                assert_eq!(record.kind, x86jit_core::DeliveryKind::Fault);
                assert!(record.virtual_8086_before && !record.virtual_8086_after);
                assert_eq!(record.saved_linear, 0x1200);
                if error.is_some() {
                    cpu.cpu.gpr[4] += 4;
                }
                if policy == ExceptionDelivery::Outward {
                    transfer_steps(&mut cpu, &vm, 1);
                }
                transfer_steps(&mut cpu, &vm, 1);
                assert_v86(&cpu, 0x1000, u64::from(flags | 0x10000));
            }
        }
    }
}

fn identity_paging(vm: &Vm, cpu: &mut Vcpu) {
    cpu.cpu.legacy.cr0 |= 1 << 31;
    cpu.cpu.legacy.cr3 = 0x10000;
    dwords(vm, 0x10000, &[0x11007]);
    for page in 0..32u32 {
        // Only the V86 instruction page is user-accessible. GDT, IDT, TSS
        // and both protected stacks stay supervisor-only.
        dwords(
            vm,
            0x11000 + u64::from(page * 4),
            &[(page * 0x1000) | if page == 1 { 7 } else { 3 }],
        );
    }
}

// SDM Vol 2A INT pp. 3-475–477; design v86 §4.3: supervisor preflight
// of the complete frame precedes its first store and any cache installation.
#[test]
fn delivery_stack_page_fault_and_limit_leave_no_partial_frame() {
    for model in MODELS {
        for page_fault in [false, true] {
            let (vm, mut cpu) = transfers(
                model,
                x86jit_core::ExceptionDelivery::Outward,
                &[0xcd, 0x40],
                true,
            );
            gate(&vm, 0x40, 0xee);
            if page_fault {
                identity_paging(&vm, &mut cpu);
                dwords(&vm, TSS + 4, &[0x8010]); // Frame straddles two pages.
                dwords(&vm, 0x11000 + 7 * 4, &[0x7002]);
            } else {
                fixture::descriptor(&vm, 0x2010, 0, 0x8ff0, 0x4093);
            }
            transfer_steps(&mut cpu, &vm, 1);
            vm.write_bytes(0x7f00, &[0xa5; 0x200]).unwrap();
            let before = (
                cpu.cpu.gpr,
                cpu.cpu.legacy.segments,
                cpu.cpu.rflags_system,
                cpu.cpu.cs,
                cpu.cpu.ss,
            );
            let result = transfer_run(&mut cpu, &vm, 1);
            assert_eq!(result.steps(), 0);
            let StepStop::Exit(exit @ Exit::Exception { vector, .. }) = result.stop() else {
                panic!("{:?}", result.stop());
            };
            assert_eq!(*vector, if page_fault { 14 } else { 12 });
            assert_eq!(
                exit.fault_metadata().unwrap().error_code(),
                ErrorCode::bits(
                    if page_fault { 2 } else { 0x10 },
                    if page_fault { 7 } else { 0xffff }
                )
            );
            if page_fault {
                assert_eq!(cpu.cpu.legacy.cr2, 0x7ffc);
            }
            assert_eq!(
                (
                    cpu.cpu.gpr,
                    cpu.cpu.legacy.segments,
                    cpu.cpu.rflags_system,
                    cpu.cpu.cs,
                    cpu.cpu.ss
                ),
                before
            );
            let mut bytes = [0; 0x200];
            vm.read_bytes(0x7f00, &mut bytes).unwrap();
            assert_eq!(bytes, [0xa5; 0x200]);
            assert!(cpu.delivery_log().unwrap().is_empty());
        }
    }
}

// SDM Vol 3B §23.3.1 p. 23-12 and design v86 §9 Q2: a 16-bit gate
// is a typed boundary, while conforming/nonzero-DPL targets are #GP(CS).
#[test]
fn v86_gate_width_and_target_privilege_are_checked_before_mutation() {
    for model in MODELS {
        for (gate_attr, code_attr, refusal) in [
            (0xe6, 0x409b, true),
            (0xe7, 0x409b, true),
            (0xee, 0x409f, false),
            (0xee, 0x40bb, false),
            (0xee, 0x40fb, false),
        ] {
            let (vm, mut cpu) = transfers(
                model,
                x86jit_core::ExceptionDelivery::Outward,
                &[0xcd, 0x40],
                true,
            );
            gate(&vm, 0x40, gate_attr);
            transfer_steps(&mut cpu, &vm, 1);
            fixture::descriptor(&vm, 0x2008, 0, 0xffff, code_attr);
            let before = (cpu.cpu.gpr, cpu.cpu.legacy.segments, cpu.cpu.rflags_system);
            let result = transfer_run(&mut cpu, &vm, 1);
            assert_eq!(result.steps(), 0);
            if refusal {
                assert!(matches!(
                    result.stop(),
                    StepStop::Exit(Exit::UnsupportedSystemContext {
                        operation: UnsupportedSystemOperation::LegacyGateOrTaskTransfer,
                        ..
                    })
                ));
            } else {
                let StepStop::Exit(exit @ Exit::Exception { vector: 13, .. }) = result.stop()
                else {
                    panic!("{:?}", result.stop());
                };
                assert_eq!(
                    exit.fault_metadata().unwrap().error_code(),
                    ErrorCode::bits(8, 0xffff)
                );
            }
            assert_eq!(
                (cpu.cpu.gpr, cpu.cpu.legacy.segments, cpu.cpu.rflags_system),
                before
            );
            assert!(cpu.delivery_log().unwrap().is_empty());
        }
    }
}

// SDM Vol 2A INT p. 3-476: current TSS width and new SS.B are independent
// of gate width; a 16-bit current TSS supplies zero-extended SP0.
#[test]
fn paged_v86_round_trip_uses_supervisor_tables_and_accepts_a_16_bit_tss() {
    for model in MODELS {
        for narrow_tss in [false, true] {
            let (vm, mut cpu) = transfers(
                model,
                x86jit_core::ExceptionDelivery::InGuest,
                &[0xcd, 0x40],
                false,
            );
            gate(&vm, 0x40, 0xee);
            identity_paging(&vm, &mut cpu);
            if narrow_tss {
                cpu.cpu.legacy.tr.cache.attributes = 0x83;
                vm.write_bytes(TSS + 2, &[0, 0x90, 0x10, 0]).unwrap();
            }
            fixture::descriptor(&vm, 0x2010, 0, 0xffff, 0x93);
            transfer_steps(&mut cpu, &vm, 2);
            assert_handler(&cpu, &vm, 0x1002, ENTRY_IMAGE[2], None, true);
            transfer_steps(&mut cpu, &vm, 2);
            assert_v86(&cpu, 0x1002, u64::from(ENTRY_IMAGE[2]));
        }
    }
}

fn trap_range(vm: &mut Vm, at: u64, size: usize) {
    use x86jit_core::{Prot, RegionKind};
    let mut bytes = vec![0; 0x240000];
    vm.read_bytes(0, &mut bytes).unwrap();
    vm.unmap(0, bytes.len()).unwrap();
    vm.map(0, at as usize, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(at, size, Prot::RW, RegionKind::Trap).unwrap();
    vm.map(
        at + size as u64,
        bytes.len() - at as usize - size,
        Prot::RWX,
        RegionKind::Ram,
    )
    .unwrap();
    vm.write_bytes(0, &bytes[..at as usize]).unwrap();
    vm.write_bytes(at + size as u64, &bytes[at as usize + size..])
        .unwrap();
}
fn zero_probe_preserves_answer(cpu: &mut Vcpu, vm: &Vm) {
    let before = format!("{:?}", cpu.cpu);
    let retired = cpu.retired_instructions();
    let result = transfer_run(cpu, vm, 0);
    assert_eq!(result.steps(), 0);
    assert!(matches!(
        result.stop(),
        StepStop::Exit(Exit::BudgetExhausted)
    ));
    assert_eq!(format!("{:?}", cpu.cpu), before);
    assert_eq!(cpu.retired_instructions(), retired);
}

// SDM Vol 2A IRET pp. 3-492–493; design v86 §4.2: every frame slot
// belongs to one resumable instruction, with no early register commit.
#[test]
fn nine_slot_entry_and_return_resume_each_mmio_read_and_write_exactly_once() {
    use x86jit_core::ExceptionDelivery;
    for model in MODELS {
        for paged in [false, true] {
            let (mut vm, mut cpu) =
                transfers(model, ExceptionDelivery::Outward, &[0xcd, 0x40], true);
            gate(&vm, 0x40, 0xee);
            if paged {
                identity_paging(&vm, &mut cpu);
            }
            trap_range(&mut vm, 0x8000, 36);
            let before = (cpu.cpu.gpr, cpu.cpu.legacy.segments, cpu.cpu.rflags_system);
            for (index, value) in ENTRY_IMAGE.into_iter().enumerate() {
                let result = transfer_run(&mut cpu, &vm, 1);
                assert_eq!(result.steps(), 0);
                assert!(
                    matches!(result.stop(), StepStop::Exit(Exit::MmioRead { addr, size: 4, .. }) if *addr == 0x8000 + index as u64 * 4),
                    "{:?}",
                    result.stop()
                );
                assert_eq!(
                    (cpu.cpu.gpr, cpu.cpu.legacy.segments, cpu.cpu.rflags_system),
                    before
                );
                assert_eq!(cpu.retired_instructions(), 0);
                cpu.cpu.pending_mmio = Some(u64::from(value));
                zero_probe_preserves_answer(&mut cpu, &vm);
            }
            transfer_steps(&mut cpu, &vm, 1);
            assert_v86(&cpu, 0x1000, u64::from(ENTRY_IMAGE[2]));
            assert_eq!(cpu.retired_instructions(), 1);
            // The protected handler's frame is a device too. A write answer
            // must survive replay of every later slot and every zero probe.
            let (mut vm, mut cpu) =
                transfers(model, ExceptionDelivery::Outward, &[0xcd, 0x40], true);
            gate(&vm, 0x40, 0xee);
            if paged {
                identity_paging(&vm, &mut cpu);
            }
            transfer_steps(&mut cpu, &vm, 1);
            trap_range(&mut vm, 0x8fdc, 36);
            let before = (cpu.cpu.gpr, cpu.cpu.legacy.segments, cpu.cpu.rflags_system);
            let pushed = [
                0x70,
                0x60,
                0x50,
                0x40,
                0x30,
                0xabcd_7000,
                ENTRY_IMAGE[2],
                0x20,
                0x1002,
            ];
            for (index, value) in pushed.into_iter().enumerate() {
                let result = transfer_run(&mut cpu, &vm, 1);
                assert_eq!(result.steps(), 0);
                assert!(
                    matches!(result.stop(), StepStop::Exit(Exit::MmioWrite { addr, size: 4, value: actual, .. }) if *addr == 0x8ffc - index as u64 * 4 && *actual == u64::from(value)),
                    "{:?}",
                    result.stop()
                );
                assert_eq!(
                    (cpu.cpu.gpr, cpu.cpu.legacy.segments, cpu.cpu.rflags_system),
                    before
                );
                assert!(cpu.delivery_log().unwrap().is_empty());
                assert_eq!(cpu.retired_instructions(), 1);
                cpu.cpu.pending_mmio_write = true;
                zero_probe_preserves_answer(&mut cpu, &vm);
            }
            transfer_steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.rip, HANDLER);
            assert_eq!(cpu.retired_instructions(), 2);
            let record = cpu.delivery_log().unwrap().iter().next().unwrap();
            assert!(record.virtual_8086_before && !record.virtual_8086_after);
            assert_eq!(record.saved_linear, 0x1202);
            transfer_steps(&mut cpu, &vm, 1); // Handler NOP.
            for (index, value) in pushed.into_iter().rev().enumerate() {
                let result = transfer_run(&mut cpu, &vm, 1);
                assert_eq!(result.steps(), 0);
                assert!(
                    matches!(result.stop(), StepStop::Exit(Exit::MmioRead { addr, size: 4, .. }) if *addr == 0x8fdc + index as u64 * 4),
                    "{:?}",
                    result.stop()
                );
                assert_eq!(cpu.cpu.legacy.cpl, 0);
                cpu.cpu.pending_mmio = Some(u64::from(value));
                zero_probe_preserves_answer(&mut cpu, &vm);
            }
            transfer_steps(&mut cpu, &vm, 1);
            assert_v86(&cpu, 0x1002, u64::from(ENTRY_IMAGE[2]));
            assert_eq!(cpu.retired_instructions(), 4);
            assert_eq!(cpu.delivery_log().unwrap().len(), 1);
        }
    }
}

// SDM Vol 2A IRET pp. 3-492–493; engine commit-last policy (design v86 §4.2).
#[test]
fn nine_slot_entry_preflights_full_stack_and_reads_before_committing() {
    for model in MODELS {
        for slot in 0..9usize {
            let (vm, mut cpu) = transfers(
                model,
                x86jit_core::ExceptionDelivery::Outward,
                &[0x90],
                true,
            );
            let pointer = 0x9000 - slot as u64 * 4;
            dwords(&vm, pointer, &ENTRY_IMAGE);
            cpu.cpu.gpr[4] = pointer;
            identity_paging(&vm, &mut cpu);
            dwords(&vm, 0x11000 + 9 * 4, &[0x9002]);
            let before = (cpu.cpu.gpr, cpu.cpu.legacy.segments, cpu.cpu.rflags_system);
            let result = transfer_run(&mut cpu, &vm, 1);
            assert_eq!(result.steps(), 0);
            let StepStop::Exit(exit @ Exit::Exception { vector: 14, .. }) = result.stop() else {
                panic!("{:?}", result.stop());
            };
            assert_eq!(
                exit.fault_metadata().unwrap().error_code(),
                ErrorCode::bits(0, 7)
            );
            assert_eq!(cpu.cpu.legacy.cr2, 0x9000);
            assert_eq!(
                (cpu.cpu.gpr, cpu.cpu.legacy.segments, cpu.cpu.rflags_system),
                before
            );
            assert_eq!(cpu.retired_instructions(), 0);
        }
        for limit in [0x800a, 0x801f] {
            let (vm, mut cpu) = transfers(
                model,
                x86jit_core::ExceptionDelivery::Outward,
                &[0x90],
                false,
            );
            cpu.cpu.legacy.segments[SegmentId::Ss as usize].limit = limit;
            let before = (cpu.cpu.gpr, cpu.cpu.legacy.segments, cpu.cpu.rflags_system);
            let result = transfer_run(&mut cpu, &vm, 1);
            assert!(matches!(
                result.stop(),
                StepStop::Exit(Exit::Exception { vector: 12, .. })
            ));
            assert_eq!(
                (cpu.cpu.gpr, cpu.cpu.legacy.segments, cpu.cpu.rflags_system),
                before
            );
        }
        let (vm, mut cpu) = transfers(
            model,
            x86jit_core::ExceptionDelivery::Outward,
            &[0x90],
            true,
        );
        dwords(&vm, 0x8000, &[0x10000]);
        let result = transfer_run(&mut cpu, &vm, 1);
        assert!(matches!(
            result.stop(),
            StepStop::Exit(Exit::Exception { vector: 13, .. })
        ));
        assert_eq!(cpu.cpu.legacy.cpl, 0);
        assert_eq!(cpu.cpu.gpr[4], 0x8000);
    }
}

// SDM Vol 3B §§23.2.1, 23.2.5 pp. 23-6, 23-8–9: the entry loads
// low selector words as paragraphs, including zero; CPL does not follow CS.RPL.
#[test]
fn entry_accepts_zero_paragraphs_and_discards_high_selector_words() {
    for model in MODELS {
        for cs in [0u32, 1, 2, 3] {
            let (vm, mut cpu) = transfers(
                model,
                x86jit_core::ExceptionDelivery::Outward,
                &[0x90],
                true,
            );
            dwords(
                &vm,
                0x8000,
                &[
                    0x1234,
                    0xface_0000 | cs,
                    ENTRY_IMAGE[2],
                    0xdead_fffe,
                    0xbeef_0000,
                    0xabcd_0000,
                    0xaaaa_0001,
                    0x1234_0002,
                    0xffff_0003,
                ],
            );
            transfer_steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.legacy.cpl, 3);
            assert_eq!(cpu.cpu.cs, cs as u16);
            assert_eq!(cpu.cpu.ss, 0);
            assert_eq!(cpu.cpu.es, 0);
            assert_eq!((cpu.cpu.ds, cpu.cpu.fs, cpu.cpu.gs), (1, 2, 3));
            assert_eq!(cpu.cpu.legacy.segments[0].base, cs << 4);
            assert_eq!(cpu.cpu.legacy.segments[SegmentId::Ss as usize].base, 0);
            assert_eq!(cpu.cpu.legacy.segments[SegmentId::Es as usize].unusable, 0);
            assert_eq!(cpu.cpu.gpr[4], 0xdead_fffe);
        }
    }
}

// SDM Vol 2A IRET pp. 3-492–493: only a CPL-0 dword frame can set VM.
#[test]
fn protected_iret_above_ring_zero_ignores_vm_and_word_iret_cannot_enter() {
    for model in MODELS {
        for cpl in 1..=3u16 {
            let (vm, mut cpu) = transfers(
                model,
                x86jit_core::ExceptionDelivery::Outward,
                &[0x90],
                true,
            );
            let attr = 0x409b | cpl << 5;
            fixture::descriptor(&vm, 0x2008, 0, 0xffff, attr);
            cpu.cpu.legacy.cpl = cpl as u8;
            cpu.cpu.cs = 8 | cpl;
            cpu.cpu.legacy.segments[0].attributes = attr;
            dwords(&vm, 0x8000, &[0x1234, u32::from(8 | cpl), ENTRY_IMAGE[2]]);
            transfer_steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.legacy.cpl, cpl as u8);
            assert_eq!(cpu.cpu.rflags_system & VM, 0);
            assert_eq!(cpu.cpu.gpr[4], 0x800c);
        }
        let (vm, mut cpu) = transfers(
            model,
            x86jit_core::ExceptionDelivery::Outward,
            &[0x90],
            true,
        );
        vm.write_bytes(0x1000, &[0x66, 0xcf]).unwrap();
        vm.write_bytes(0x8000, &[0x34, 0x12, 8, 0, 3, 0x32])
            .unwrap();
        transfer_steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rflags_system & VM, 0);
        assert_eq!(cpu.cpu.gpr[4], 0x8006);
        assert_eq!(cpu.cpu.rip, 0x1234);
    }
}

// Design v86 §5 and Q4: an accepted IRQ owns its source across device
// handoffs, including another queued IRQ; only completed deliveries are logged.
#[test]
fn hardware_delivery_continuations_keep_the_source_and_do_not_consume_the_next_irq() {
    for model in MODELS {
        for paged in [false, true] {
            let (mut vm, mut cpu) = transfers(
                model,
                x86jit_core::ExceptionDelivery::Outward,
                &[0x90],
                true,
            );
            gate(&vm, 0x40, 0x8e);
            gate(&vm, 0x41, 0x8e);
            if paged {
                identity_paging(&vm, &mut cpu);
            }
            transfer_steps(&mut cpu, &vm, 1);
            trap_range(&mut vm, 0x8fdc, 36);
            cpu.inject_irq(0x40);
            let pushed = [
                0x70,
                0x60,
                0x50,
                0x40,
                0x30,
                0xabcd_7000,
                ENTRY_IMAGE[2],
                0x20,
                0x1000,
            ];
            for (index, value) in pushed.into_iter().enumerate() {
                let result = transfer_run(&mut cpu, &vm, 1);
                assert_eq!(result.steps(), 0);
                assert!(
                    matches!(result.stop(), StepStop::Exit(Exit::MmioWrite { addr, value: actual, .. }) if *addr == 0x8ffc - index as u64 * 4 && *actual == u64::from(value)),
                    "{:?}",
                    result.stop()
                );
                if index == 0 {
                    cpu.inject_irq(0x41);
                }
                cpu.cpu.pending_mmio_write = true;
                zero_probe_preserves_answer(&mut cpu, &vm);
                assert!(cpu.delivery_log().unwrap().is_empty());
            }
            transfer_steps(&mut cpu, &vm, 1); // IRQ plus handler NOP.
            assert_eq!(cpu.cpu.rip, HANDLER + 1);
            let record = cpu.delivery_log().unwrap().iter().next().unwrap();
            assert_eq!(record.vector, 0x40);
            assert_eq!(record.kind, x86jit_core::DeliveryKind::Irq);
            assert_eq!(
                (record.saved_cs, record.saved_ip, record.saved_linear),
                (0x20, 0x1000, 0x1200)
            );
            assert!(record.virtual_8086_before && !record.virtual_8086_after);
            for value in pushed.into_iter().rev() {
                let result = transfer_run(&mut cpu, &vm, 1);
                assert!(
                    matches!(result.stop(), StepStop::Exit(Exit::MmioRead { .. })),
                    "{:?}",
                    result.stop()
                );
                cpu.cpu.pending_mmio = Some(u64::from(value));
            }
            transfer_steps(&mut cpu, &vm, 1); // IRETD to the interrupted V86 IP.
                                              // The queued successor enters on the next boundary, with its own frame.
            let result = transfer_run(&mut cpu, &vm, 1);
            assert!(
                matches!(
                    result.stop(),
                    StepStop::Exit(Exit::MmioWrite { value: 0x70, .. })
                ),
                "{:?}",
                result.stop()
            );
            assert_eq!(cpu.delivery_log().unwrap().len(), 1);
        }
    }
}

// SDM Vol 3B §23.3.1.1 p. 23-13: save old VM/TF/RF/NT, clear live
// VM/TF/RF/NT, then allow IRETD to restore the complete V86 flag image.
#[test]
fn v86_delivery_saves_and_clears_live_tf_rf_nt_before_iret_restores_them() {
    for model in MODELS {
        for trap in [false, true] {
            let (vm, mut cpu) = transfers(
                model,
                x86jit_core::ExceptionDelivery::Outward,
                &[0xcd, 0x40],
                true,
            );
            gate(&vm, 0x40, if trap { 0xef } else { 0xee });
            transfer_steps(&mut cpu, &vm, 1);
            cpu.cpu.rflags_system |= 0x4100;
            let saved = ENTRY_IMAGE[2] | 0x4100;
            transfer_steps(&mut cpu, &vm, 1);
            assert_handler(&cpu, &vm, 0x1002, saved, None, !trap);
            transfer_steps(&mut cpu, &vm, 2);
            assert_v86(&cpu, 0x1002, u64::from(saved));
            assert_eq!(cpu.cpu.rflags_system & 0x14100, 0x14100);
        }
    }
}

// SDM Vol 2A §3.3 IRET pp. 3-492–493; Vol 3B §23.2.5 pp. 23-8–9.
#[test]
fn public_test386_shaped_iretd_enters_the_first_v86_test() {
    for model in MODELS {
        let (vm, mut cpu) = transfers(
            model,
            x86jit_core::ExceptionDelivery::Outward,
            &[0x90],
            true,
        );
        cpu.cpu.cs = 0xd0;
        cpu.cpu.legacy.segments[0].base = 0xf0000;
        cpu.cpu.rip = 0x2ca0;
        vm.write_bytes(0xf2ca0, &[0xcf]).unwrap();
        dwords(
            &vm,
            0x8000,
            &[
                0x5630, 0xf000, 0x20246, 0x7fff, 0x1000, 0x3040, 0x2010, 0x5060, 0x7080,
            ],
        );
        let before = cpu.retired_instructions();
        fixture::steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.retired_instructions() - before, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl),
            (0xf000, 0x5630, 3)
        );
        assert_eq!((cpu.cpu.ss, cpu.cpu.gpr[4]), (0x1000, 0x7fff));
        assert_eq!(
            (cpu.cpu.es, cpu.cpu.ds, cpu.cpu.fs, cpu.cpu.gs),
            (0x3040, 0x2010, 0x5060, 0x7080)
        );
        assert_eq!(
            u64::from(cpu.cpu.flags.to_flags16()) | cpu.cpu.rflags_system,
            0x20246
        );
        let selectors = [
            cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs,
        ];
        for (i, selector) in selectors.into_iter().enumerate() {
            assert_eq!(
                cpu.cpu.legacy.segments[i],
                SegmentCache {
                    base: u32::from(selector) << 4,
                    limit: 0xffff,
                    attributes: if i == 0 { 0x9b } else { 0x93 },
                    ..Default::default()
                }
            );
        }
    }
}

const TASK_A: u16 = 0x28;
const TASK_B: u16 = 0x30;
const TASK_B_BASE: u64 = 0x5000;

fn task_image(vm: &Vm, selector: u16, base: u64, v86: bool, wide: bool) {
    // 386 PRM §7.1 Figs. 7-1/7-2 pp. 7-2–4; SDM Vol 3A §10.2.1.
    fixture::descriptor(
        vm,
        0x2000 + u64::from(selector),
        base as u32,
        if wide { 0x67 } else { 0x2b },
        if wide { 0x89 } else { 0x81 },
    );
    if wide {
        dwords(vm, base + 0x20, &[0x1000, if v86 { 0x23202 } else { 2 }]);
        dwords(vm, base + 0x38, &[if v86 { 0xabcd_7000 } else { 0x9000 }]);
        dwords(
            vm,
            base + 0x48,
            &if v86 {
                [0x40, 0x20, 0x30, 0x50, 0x60, 0x70]
            } else {
                [0x10, 8, 0x10, 0x10, 0, 0]
            },
        );
        vm.write_bytes(base + 0x66, &0x68u16.to_le_bytes()).unwrap();
    } else {
        for (at, value) in [
            (0x0e, 0x1800u16),
            (0x10, 2),
            (0x1a, 0x9000),
            (0x22, 0x10),
            (0x24, 8),
            (0x26, 0x10),
            (0x28, 0x10),
        ] {
            vm.write_bytes(base + at, &value.to_le_bytes()).unwrap();
        }
    }
}
fn task_gate(vm: &Vm, vector: u8, selector: u16) {
    dwords(
        vm,
        IDT + u64::from(vector) * 8,
        &[u32::from(selector) << 16, 0x8500],
    );
}
fn task_world(model: CpuModel, call: bool, via_gate: bool) -> (Vm, Vcpu) {
    task_world_delivery(
        model,
        call,
        via_gate,
        x86jit_core::ExceptionDelivery::Outward,
    )
}
fn task_world_delivery(
    model: CpuModel,
    call: bool,
    via_gate: bool,
    delivery: x86jit_core::ExceptionDelivery,
) -> (Vm, Vcpu) {
    let (vm, mut cpu) = transfers(model, delivery, &[0x90; 8], true);
    task_image(&vm, TASK_A, TSS, false, true);
    fixture::descriptor(&vm, 0x2000 + u64::from(TASK_A), TSS as u32, 0x67, 0x8b);
    cpu.cpu.legacy.tr = SystemSegment {
        selector: TASK_A,
        cache: SegmentCache {
            base: TSS as u32,
            limit: 0x67,
            attributes: 0x8b,
            ..Default::default()
        },
        ..Default::default()
    };
    task_image(&vm, TASK_B, TASK_B_BASE, true, true);
    let target = if via_gate {
        dwords(&vm, 0x2038, &[u32::from(TASK_B) << 16, 0x8500]);
        0x38u16
    } else {
        TASK_B
    };
    let mut code = vec![if call { 0x9a } else { 0xea }, 0, 0, 0, 0];
    code.extend(target.to_le_bytes());
    vm.write_bytes(0x1000, &code).unwrap();
    (vm, cpu)
}

// SDM Vol 3B §§23.2.5, 23.3.1.3 pp. 23-8–9, 23-14;
// 386 PRM §15.3.1 p. 15-6; 486 PRM §23.3.1 p. 23-6.
#[test]
fn public_task_jump_call_and_task_gate_install_v86_without_descriptor_reads() {
    for model in MODELS {
        for call in [false, true] {
            for via_gate in [false, true] {
                let (vm, mut cpu) = task_world(model, call, via_gate);
                fixture::steps(&mut cpu, &vm, 1);
                assert_v86(&cpu, 0x1000, if call { 0x27202 } else { 0x23202 });
                assert_eq!(cpu.cpu.legacy.tr.selector, TASK_B);
                assert_eq!(cpu.cpu.legacy.cr0 & 8, 8);
                assert_eq!(fixture::value(&vm, 0x2035, 1) & 2, 2);
                assert_eq!(fixture::value(&vm, 0x202d, 1) & 2, if call { 2 } else { 0 });
                assert_eq!(fixture::value(&vm, TSS + 0x20, 4), 0x1007);
                assert_eq!(
                    fixture::value(&vm, TASK_B_BASE, 2),
                    if call { u64::from(TASK_A) } else { 0 }
                );
                fixture::steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.rip, 0x1001);
            }
        }
    }
}

#[test]
fn public_v86_irq_task_gate_and_nested_iret_restore_v86_once() {
    for model in MODELS {
        for wide in [false, true] {
            let (vm, mut cpu) = task_world(model, false, false);
            fixture::steps(&mut cpu, &vm, 1);
            task_image(&vm, TASK_A, TSS, false, wide);
            vm.write_bytes(if wide { 0x1000 } else { 0x1800 }, &[0xcf])
                .unwrap();
            task_gate(&vm, 0x40, TASK_A);
            cpu.inject_irq(0x40);
            let retired = cpu.retired_instructions();
            // Delivery costs no step; IRET is the one retired instruction.
            fixture::steps(&mut cpu, &vm, 1);
            assert_v86(&cpu, 0x1000, 0x23202);
            assert_eq!(cpu.retired_instructions() - retired, 1);
            assert_eq!(fixture::value(&vm, TASK_B_BASE + 0x24, 4) & VM, VM);
            assert_eq!(fixture::value(&vm, TSS, 2), u64::from(TASK_B));
            assert_eq!(fixture::value(&vm, 0x202d, 1) & 2, 0);
            assert!(!cpu.has_pending_irq());
            let log = cpu.delivery_log().unwrap();
            assert_eq!(log.total(), 1);
            let record = log.iter().next().unwrap();
            assert!(record.virtual_8086_before);
            assert!(!record.virtual_8086_after);
        }
    }
}

#[test]
fn public_v86_task_faults_stay_in_the_reached_task() {
    for model in MODELS {
        for phase in 0..5 {
            let (vm, mut cpu) = task_world(model, false, false);
            let (vector, code, committed) = match phase {
                0 => {
                    fixture::descriptor(&vm, 0x2030, TASK_B_BASE as u32, 0x67, 0x09);
                    (11, 0x30, false)
                }
                1 => {
                    fixture::descriptor(&vm, 0x2030, TASK_B_BASE as u32, 0x20, 0x89);
                    (10, 0x30, false)
                }
                2 => {
                    dwords(&vm, TASK_B_BASE + 0x60, &[0x78]);
                    (10, 0x78, true)
                }
                3 => {
                    dwords(&vm, TASK_B_BASE + 0x20, &[0x10000]);
                    (13, 0, true)
                }
                _ => {
                    fixture::descriptor(&vm, 0x2078, 0, 0xffff, 0x82);
                    dwords(&vm, TASK_B_BASE + 0x60, &[0x78]);
                    // A valid LDT is loaded despite V86 paragraphs bypassing it.
                    fixture::steps(&mut cpu, &vm, 1);
                    assert_eq!(cpu.cpu.legacy.ldtr.selector, 0x78);
                    assert_eq!(cpu.cpu.legacy.ldtr.cache.unusable, 0);
                    continue;
                }
            };
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 0);
            let StepStop::Exit(e @ Exit::Exception { vector: actual, .. }) = r.stop() else {
                panic!("{:?}", r.stop());
            };
            assert_eq!(*actual, vector);
            assert_eq!(
                e.fault_metadata().unwrap().error_code(),
                ErrorCode::bits(code, 0xffff)
            );
            assert_eq!(
                cpu.cpu.legacy.tr.selector,
                if committed { TASK_B } else { TASK_A }
            );
            assert_eq!(cpu.retired_instructions(), 0);
            if phase >= 2 {
                assert_eq!(cpu.cpu.legacy.cpl, 3);
                assert_ne!(cpu.cpu.rflags_system & VM, 0);
            }
        }
    }
}

// 386 PRM §§11.1.2, 11.1.5.3 pp. 11-1, 11-4; §15.6 p. 15-12.
// The assertion/wait/ack handshake is the board contract of fpu-error-routing.md.
#[test]
fn public_v86_external_numeric_assertion_wait_and_irq_round_trip() {
    use x86jit_core::{NumericErrorEvent as E, NumericErrorRoute as R};
    let (vm, mut cpu) = transfers(
        CpuModel::I80386,
        x86jit_core::ExceptionDelivery::InGuest,
        &[0xd8, 0xf1, 0x9b, 0x90],
        true,
    );
    fixture::steps(&mut cpu, &vm, 1);
    cpu.set_fpu_model(FpuModel::Present);
    cpu.set_numeric_error_route(R::External).unwrap();
    cpu.cpu.fpu_top = 0;
    cpu.cpu.fpu_empty = !3;
    cpu.cpu.fpr[0] = [0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f];
    cpu.cpu.fpr[1] = [0; 10];
    cpu.cpu.fpu_cw = 0x37b;
    cpu.cpu.fpu_sw = 0;
    let retired = cpu.retired_instructions();
    for (event, steps, budget) in [(E::Asserted, 1, 1), (E::Waiting, 0, 8), (E::Waiting, 0, 1)] {
        let r = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
        assert_eq!(r.steps(), steps, "{:?}", r.stop());
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::ExternalNumericError { addr: 0x1002, event: e }) if *e == event),
            "{:?}",
            r.stop()
        );
        assert_eq!(cpu.cpu.rip, 0x1002);
        assert_eq!(cpu.retired_instructions() - retired, 1);
        assert!(cpu.numeric_error_pending());
        zero_probe_preserves_answer(&mut cpu, &vm);
    }
    gate(&vm, 0x40, 0x8e);
    vm.write_bytes(HANDLER, &[0xdb, 0xe2, 0xcf]).unwrap(); // FNCLEX; IRETD.
    cpu.inject_irq(0x40);
    fixture::steps(&mut cpu, &vm, 1);
    assert!(!cpu.numeric_error_pending());
    cpu.ack_external_numeric_error().unwrap();
    fixture::steps(&mut cpu, &vm, 1);
    assert_v86(&cpu, 0x1002, u64::from(ENTRY_IMAGE[2] & !0x10000));
    fixture::steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.rip, 0x1003);
    assert_eq!(cpu.retired_instructions() - retired, 4);
    assert_eq!(cpu.delivery_log().unwrap().total(), 1);
}

// Decision-24: accepted event ownership includes VM and every hidden cache;
// zero-budget probes retain the completion, edited inputs abandon the event.
#[test]
fn held_v86_event_vm_and_segment_edits_abandon_the_accepted_irq() {
    for model in MODELS {
        for paged in [false, true] {
            for edit in 0..7 {
                let (mut vm, mut cpu) = transfers(
                    model,
                    x86jit_core::ExceptionDelivery::InGuest,
                    &[0x90; 8],
                    true,
                );
                gate(&vm, 0x40, 0x8e);
                if paged {
                    identity_paging(&vm, &mut cpu);
                }
                fixture::steps(&mut cpu, &vm, 1);
                trap_range(&mut vm, 0x8ffc, 4);
                cpu.inject_irq(0x40);
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(r.steps(), 0);
                assert!(
                    matches!(
                        r.stop(),
                        StepStop::Exit(Exit::MmioWrite { addr: 0x8ffc, .. })
                    ),
                    "{:?}",
                    r.stop()
                );
                cpu.complete_mmio_write();
                zero_probe_preserves_answer(&mut cpu, &vm);
                if edit == 0 {
                    cpu.cpu.rflags_system &= !VM;
                    cpu.cpu.legacy.cpl = 0;
                } else {
                    cpu.cpu.legacy.segments[edit - 1].limit = 0xfffe;
                }
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(!cpu.has_pending_irq());
                assert_eq!(cpu.delivery_log().unwrap().total(), 0);
                assert_eq!(cpu.cpu.rip, if edit == 0 { 0x1001 } else { 0x1000 });
                if edit == 0 {
                    assert_eq!(r.steps(), 1, "{:?}", r.stop());
                } else {
                    assert_eq!(r.steps(), 0);
                    assert!(matches!(
                        r.stop(),
                        StepStop::Exit(Exit::UnsupportedSystemContext { .. })
                    ));
                }
                assert_eq!(cpu.cpu.rflags_system & VM, if edit == 0 { 0 } else { VM });
            }
        }
    }
}

#[test]
fn v86_task_image_and_outgoing_saves_resume_every_bus_transfer_once() {
    for model in MODELS {
        for paged in [false, true] {
            for incoming in [false, true] {
                let (mut vm, mut cpu) = task_world(model, true, false);
                if paged {
                    identity_paging(&vm, &mut cpu);
                }
                if paged {
                    dwords(&vm, TASK_B_BASE + 0x1c, &[0x10000]);
                }
                let at = if incoming { TASK_B_BASE } else { TSS };
                let mut bytes = [0; 0x68];
                vm.read_bytes(at, &mut bytes).unwrap();
                trap_range(&mut vm, at, bytes.len());
                let retired = cpu.retired_instructions();
                let mut seen = std::collections::BTreeMap::new();
                for _ in 0..100 {
                    let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    match r.stop() {
                        StepStop::Exit(Exit::MmioRead { addr, size, .. }) => {
                            assert_eq!(r.steps(), 0);
                            let visits = seen.entry((*addr, *size, false)).or_insert(0);
                            *visits += 1;
                            assert_eq!(*visits, 1, "repeated task-image read {addr:x}");
                            let start = (*addr - at) as usize;
                            let mut value = [0; 8];
                            value[..*size as usize]
                                .copy_from_slice(&bytes[start..start + *size as usize]);
                            cpu.complete_mmio_read(u64::from_le_bytes(value));
                        }
                        StepStop::Exit(Exit::MmioWrite {
                            addr, size, value, ..
                        }) => {
                            assert_eq!(r.steps(), 0);
                            assert!(
                                seen.insert((*addr, *size, true), 1).is_none(),
                                "repeated write {addr:x}"
                            );
                            let start = (*addr - at) as usize;
                            bytes[start..start + *size as usize]
                                .copy_from_slice(&value.to_le_bytes()[..*size as usize]);
                            cpu.complete_mmio_write();
                        }
                        StepStop::Exit(Exit::BudgetExhausted) => {
                            assert_eq!(r.steps(), 1);
                            break;
                        }
                        stop => panic!("{model:?} paged={paged} incoming={incoming}: {stop:?}"),
                    }
                    zero_probe_preserves_answer(&mut cpu, &vm);
                }
                assert_eq!(cpu.retired_instructions() - retired, 1);
                assert_v86(&cpu, 0x1000, 0x27202);
                assert!(!seen.is_empty());
            }
        }
    }
}

#[test]
fn public_task_switch_loads_cr3_before_v86_ldt_checks_and_fetch() {
    for model in MODELS {
        for bad_ldt in [false, true] {
            let (vm, mut cpu) = task_world(model, false, false);
            identity_paging(&vm, &mut cpu);
            dwords(&vm, TASK_B_BASE + 0x1c, &[0x12000]);
            dwords(&vm, 0x12000, &[0x13007]);
            for page in 0..32u32 {
                dwords(&vm, 0x13000 + u64::from(page * 4), &[(page * 0x1000) | 7]);
            }
            // In the new address space GDT is physically elsewhere; old GDT
            // has no valid LDT. Both tables/TSS are supervisor in the old map.
            dwords(&vm, 0x13008, &[0x14003]);
            fixture::descriptor(&vm, 0x14078, 0x16000, 7, if bad_ldt { 0x92 } else { 0x82 });
            dwords(&vm, TASK_B_BASE + 0x60, &[0x78]);
            dwords(&vm, 0x13004, &[0x15007]);
            vm.write_bytes(0x15200, &[0x43]).unwrap(); // INC BX, in V86.
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(cpu.cpu.legacy.cr3, 0x12000);
            assert_eq!(cpu.cpu.legacy.tr.selector, TASK_B);
            assert_eq!(cpu.cpu.legacy.cpl, 3);
            if bad_ldt {
                assert_eq!(r.steps(), 0);
                assert!(
                    matches!(r.stop(), StepStop::Exit(Exit::Exception { vector: 10, .. })),
                    "{:?}",
                    r.stop()
                );
            } else {
                assert_eq!(r.steps(), 1, "{:?}", r.stop());
                assert_eq!(cpu.cpu.legacy.ldtr.cache.base, 0x16000);
                fixture::steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.gpr[3], 1);
                assert_eq!(cpu.cpu.rip, 0x1001);
            }
        }
    }
}

#[test]
fn public_task_gate_from_protected_code_and_v86_exception_error_push() {
    for model in MODELS {
        for source_v86 in [false, true] {
            let (vm, mut cpu) =
                task_world_delivery(model, false, false, x86jit_core::ExceptionDelivery::InGuest);
            if source_v86 {
                fixture::steps(&mut cpu, &vm, 1);
                task_image(&vm, TASK_A, TSS, false, true);
                vm.write_bytes(0x1000, &[0x90]).unwrap();
                vm.write_bytes(0x1200, &[0xfa]).unwrap(); // CLI at IOPL=0: #GP(0).
                cpu.cpu.rflags_system &= !0x3000;
                task_gate(&vm, 13, TASK_A);
                let retired = cpu.retired_instructions();
                fixture::steps(&mut cpu, &vm, 1); // fault costs zero, handler NOP costs one.
                assert_eq!(cpu.retired_instructions() - retired, 1);
                assert_eq!(cpu.cpu.legacy.tr.selector, TASK_A);
                assert_eq!(cpu.cpu.rflags_system & VM, 0);
                assert_eq!(cpu.cpu.rflags_system & 0x4000, 0x4000);
                assert_eq!(cpu.cpu.gpr[4], 0x8ffc);
                assert_eq!(fixture::value(&vm, 0x8ffc, 4), 0);
                assert_eq!(fixture::value(&vm, TASK_B_BASE + 0x20, 4), 0x1000);
                assert_ne!(fixture::value(&vm, TASK_B_BASE + 0x24, 4) & VM, 0);
                let record = cpu.delivery_log().unwrap().iter().next().unwrap();
                assert_eq!(record.vector, 13);
                assert!(record.virtual_8086_before);
                assert!(!record.virtual_8086_after);
            } else {
                vm.write_bytes(0x1000, &[0xcd, 0x40]).unwrap();
                task_gate(&vm, 0x40, TASK_B);
                fixture::steps(&mut cpu, &vm, 1);
                assert_v86(&cpu, 0x1000, 0x27202);
                assert_eq!(fixture::value(&vm, TSS + 0x20, 4), 0x1002);
                let record = cpu.delivery_log().unwrap().iter().next().unwrap();
                assert!(!record.virtual_8086_before);
                assert!(record.virtual_8086_after);
            }
        }
    }
}

#[test]
fn faults_leaving_v86_through_a_task_gate_use_old_or_committed_context() {
    for model in MODELS {
        for phase in 0..4 {
            let (vm, mut cpu) = task_world(model, false, false);
            fixture::steps(&mut cpu, &vm, 1);
            task_image(&vm, TASK_A, TSS, false, true);
            task_gate(&vm, 0x40, TASK_A);
            let (vector, code, committed) = match phase {
                0 => {
                    fixture::descriptor(&vm, 0x2028, TSS as u32, 0x67, 0x09);
                    (11, 0x29, false)
                }
                1 => {
                    dwords(&vm, TSS + 0x60, &[0x78]);
                    (10, 0x79, true)
                }
                2 => {
                    dwords(&vm, TSS + 0x4c, &[0]);
                    (10, 1, true)
                }
                _ => {
                    dwords(&vm, TSS + 0x20, &[0x10000]);
                    (13, 1, true)
                }
            };
            cpu.inject_irq(0x40);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 0);
            let StepStop::Exit(exit @ Exit::Exception { vector: actual, .. }) = r.stop() else {
                panic!("{:?}", r.stop());
            };
            assert_eq!(*actual, vector);
            assert_eq!(
                exit.fault_metadata().unwrap().error_code(),
                ErrorCode::bits(code, 0xffff)
            );
            assert_eq!(
                cpu.cpu.legacy.tr.selector,
                if committed { TASK_A } else { TASK_B }
            );
            assert_eq!(cpu.cpu.rflags_system & VM, if committed { 0 } else { VM });
            assert!(!cpu.has_pending_irq());
            assert_eq!(cpu.retired_instructions(), 1);
            assert_eq!(cpu.delivery_log().unwrap().total(), 0);
        }
    }
}

#[test]
fn held_committed_v86_task_vm_edit_discards_switch_and_consumed_event_together() {
    for model in MODELS {
        for paged in [false, true] {
            let (mut vm, mut cpu) = task_world(model, false, false);
            task_gate(&vm, 0x40, TASK_B);
            dwords(&vm, TASK_B_BASE + 0x60, &[0x1078]);
            cpu.cpu.legacy.gdtr.limit = 0x107f;
            fixture::descriptor(&vm, 0x3078, 0x6000, 7, 0x82);
            if paged {
                identity_paging(&vm, &mut cpu);
                dwords(&vm, TASK_B_BASE + 0x1c, &[0x10000]);
            }
            trap_range(&mut vm, 0x3078, 8);
            cpu.cpu.flags.if_ = true;
            cpu.inject_irq(0x40);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 0);
            assert!(
                matches!(
                    r.stop(),
                    StepStop::Exit(Exit::MmioRead { addr: 0x3078, .. })
                ),
                "{:?}",
                r.stop()
            );
            cpu.complete_mmio_read(0x60000007);
            assert_eq!(cpu.cpu.legacy.tr.selector, TASK_B);
            assert_eq!(cpu.cpu.rflags_system & VM, VM);
            assert!(!cpu.has_pending_irq());
            zero_probe_preserves_answer(&mut cpu, &vm);
            cpu.cpu.rflags_system &= !VM;
            cpu.cpu.legacy.cpl = 0;
            fixture::steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.rip, 0x1001);
            assert_eq!(cpu.retired_instructions(), 1);
            assert_eq!(cpu.cpu.legacy.tr.selector, TASK_B);
            assert_eq!(cpu.delivery_log().unwrap().total(), 0);
            assert!(!cpu.has_pending_irq());
        }
    }
}

#[test]
fn incoming_v86_task_error_push_uses_its_stack_and_t_bit_completes_once() {
    for model in MODELS {
        for task_debug in [false, true] {
            let (mut vm, mut cpu) =
                task_world_delivery(model, false, false, x86jit_core::ExceptionDelivery::InGuest);
            vm.write_bytes(0x1000, &[0x8e, 0xd0]).unwrap(); // MOV SS,AX: #GP(0).
            task_gate(&vm, 13, TASK_B);
            if task_debug {
                vm.write_bytes(TASK_B_BASE + 0x64, &[1, 0]).unwrap();
            }
            trap_range(&mut vm, 0x72fc, 4);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 0);
            assert!(
                matches!(
                    r.stop(),
                    StepStop::Exit(Exit::MmioWrite {
                        addr: 0x72fc,
                        size: 4,
                        value: 0,
                        ..
                    })
                ),
                "{:?}",
                r.stop()
            );
            assert_eq!(cpu.cpu.legacy.tr.selector, TASK_B);
            assert_eq!(cpu.cpu.rflags_system & VM, VM);
            cpu.complete_mmio_write();
            zero_probe_preserves_answer(&mut cpu, &vm);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(cpu.cpu.gpr[4], 0xabcd_6ffc);
            assert_eq!(cpu.delivery_log().unwrap().total(), 1);
            assert!(
                cpu.delivery_log()
                    .unwrap()
                    .iter()
                    .next()
                    .unwrap()
                    .virtual_8086_after
            );
            if task_debug {
                assert_eq!(r.steps(), 0);
                assert!(
                    matches!(
                        r.stop(),
                        StepStop::Exit(Exit::UnsupportedSystemContext {
                            operation: UnsupportedSystemOperation::LegacyTaskDebugTrap,
                            ..
                        })
                    ),
                    "{:?}",
                    r.stop()
                );
                let again = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(again.steps(), 0);
                assert_eq!(cpu.cpu.gpr[4], 0xabcd_6ffc);
                assert_eq!(cpu.retired_instructions(), 0);
            } else {
                assert_eq!(r.steps(), 1, "{:?}", r.stop());
                assert_eq!(cpu.retired_instructions(), 1);
                assert_eq!(cpu.cpu.rip, 0x1001);
            }
        }
    }
}

// SDM Vol 3A §10.3 pp. 10-10–13; 386 PRM §9.8.14.1 p. 9-18:
// outgoing saves and incoming TSS residency faults precede task commit.
#[test]
fn v86_task_gate_precommit_page_faults_keep_the_outgoing_v86_context() {
    for model in MODELS {
        for (page, linear, code) in [
            (2, 0x2028, 0),
            (4, 0x4000, 2),
            (5, 0x5020, 2),
            (7, 0x7000, 0),
        ] {
            let (vm, mut cpu) = task_world(model, false, false);
            fixture::steps(&mut cpu, &vm, 1);
            task_image(&vm, TASK_A, TSS, false, true);
            if page == 7 {
                task_image(&vm, TASK_A, 0x6fa0, false, true);
            }
            task_gate(&vm, 0x40, TASK_A);
            identity_paging(&vm, &mut cpu);
            dwords(&vm, 0x11000 + page * 4, &[0]);
            cpu.cpu.rip = 0x1003;
            cpu.inject_irq(0x40);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 0, "{model:?} page={page}: {:?}", r.stop());
            let StepStop::Exit(exit @ Exit::Exception { vector: 14, .. }) = r.stop() else {
                panic!("{:?}", r.stop());
            };
            assert_eq!(
                exit.fault_metadata().unwrap().error_code(),
                ErrorCode::bits(code, 7)
            );
            assert_eq!(cpu.cpu.legacy.cr2, linear);
            assert_eq!(cpu.cpu.legacy.tr.selector, TASK_B);
            assert_eq!(cpu.cpu.legacy.cpl, 3);
            assert_eq!(cpu.cpu.rflags_system & VM, VM);
            assert_eq!(cpu.cpu.rip, 0x1003);
            assert_eq!(fixture::value(&vm, TASK_B_BASE + 0x20, 4), 0x1000);
            assert_eq!(cpu.delivery_log().unwrap().total(), 0);
            assert!(!cpu.has_pending_irq());
        }
    }
}

#[test]
fn incoming_v86_task_error_push_fault_delivers_ss_in_the_committed_task() {
    for model in MODELS {
        let (vm, mut cpu) =
            task_world_delivery(model, false, false, x86jit_core::ExceptionDelivery::InGuest);
        fixture::steps(&mut cpu, &vm, 1);
        task_image(&vm, TASK_A, TSS, true, true);
        dwords(&vm, TSS + 4, &[0x9000, 0x10]);
        dwords(&vm, TSS + 0x38, &[0xabcd_0001]);
        task_gate(&vm, 13, TASK_A);
        gate(&vm, 12, 0x8e);
        vm.write_bytes(0x1200, &[0xfa]).unwrap(); // Source V86 CLI: #GP(0).
        cpu.cpu.rflags_system &= !0x3000;
        let before = cpu.retired_instructions();
        // The incoming V86 task cannot push a dword at SP=1. Its #SS(0)
        // enters the ring-0 handler; the original task-gate event never completes.
        fixture::steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.retired_instructions() - before, 1);
        assert_eq!(cpu.cpu.legacy.tr.selector, TASK_A);
        assert_eq!(cpu.cpu.legacy.cpl, 0);
        assert_eq!(cpu.cpu.rflags_system & VM, 0);
        assert_eq!(cpu.cpu.rip, HANDLER + 1);
        let sp = 0x9000 - 40;
        assert_eq!(cpu.cpu.gpr[4], sp);
        assert_eq!(fixture::value(&vm, sp, 4), 0);
        assert_eq!(fixture::value(&vm, sp + 4, 4), 0x1000);
        assert_eq!(fixture::value(&vm, sp + 16, 4), 0xabcd_0001);
        assert_eq!(fixture::value(&vm, sp + 12, 4) & VM, VM);
        assert_eq!(fixture::value(&vm, TASK_B_BASE + 0x24, 4) & VM, VM);
        assert_eq!(cpu.delivery_log().unwrap().total(), 1);
        let record = cpu.delivery_log().unwrap().iter().next().unwrap();
        assert_eq!(record.vector, 12);
        assert!(record.virtual_8086_before);
        assert!(!record.virtual_8086_after);
    }
}

// Decision-16 §3 / decision-24: a completion belongs to the suspended
// instruction's segment context; zero-budget entry cannot abandon it.
fn instruction_retry_with_edited_context(edit: u8) {
    for model in MODELS {
        for changed in [false, true] {
            let code = [0xa1, 0x00, 0x02]; // MOV AX,[0200h].
            let (mut vm, mut cpu) = if edit == 2 {
                fixture::setup(model, &code)
            } else {
                world(model, &code, 0)
            };
            vm.unmap(0, 0x240000).unwrap();
            for (at, size, kind) in [
                (0, 0x200, x86jit_core::RegionKind::Ram),
                (0x200, 2, x86jit_core::RegionKind::Trap),
                (0x202, 0x2ffe, x86jit_core::RegionKind::Ram),
                (0x3200, 2, x86jit_core::RegionKind::Trap),
                (0x3202, 0x240000 - 0x3202, x86jit_core::RegionKind::Ram),
            ] {
                vm.map(at, size, x86jit_core::Prot::RWX, kind).unwrap();
            }
            vm.write_bytes(0x1000, &code).unwrap();
            let retired = cpu.retired_instructions();
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 0);
            assert!(
                matches!(
                    r.stop(),
                    StepStop::Exit(Exit::MmioRead {
                        addr: 0x200,
                        size: 2,
                        ..
                    })
                ),
                "{:?}",
                r.stop()
            );
            cpu.complete_mmio_read(0x1234);
            if changed {
                match edit {
                    0 => canonical(&mut cpu, SegmentId::Ds, 0x300),
                    1 => {
                        cpu.cpu.rflags_system &= !VM;
                        cpu.cpu.legacy.cpl = 0;
                    }
                    2 => cpu.cpu.legacy.segments[SegmentId::Ds as usize].base = 0x3000,
                    _ => unreachable!(),
                }
            }
            zero_probe_preserves_answer(&mut cpu, &vm);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            if changed {
                assert_eq!(r.steps(), 0);
                let expected = if edit == 1 { 0x200 } else { 0x3200 };
                assert!(
                    matches!(
                        r.stop(),
                        StepStop::Exit(Exit::MmioRead {
                            addr,
                            size: 2,
                            ..
                        }) if *addr == expected
                    ),
                    "{:?}",
                    r.stop()
                );
                assert_eq!(cpu.cpu.pending_mmio, None);
                assert_eq!(cpu.cpu.mmio_parts_len, 0);
                assert_eq!(cpu.cpu.gpr[0], 0);
                assert_eq!(cpu.retired_instructions(), retired);
                cpu.complete_mmio_read(0x5678);
                fixture::steps(&mut cpu, &vm, 1);
            } else {
                assert_eq!(r.steps(), 1, "{:?}", r.stop());
            }
            assert_eq!(cpu.cpu.gpr[0], if changed { 0x5678 } else { 0x1234 });
            assert_eq!(cpu.cpu.rip, 0x1003);
            assert_eq!(cpu.retired_instructions(), retired + 1);
            assert!(!cpu.cpu.mmio_retry_active);
        }
    }
}

#[test]
fn held_v86_instruction_ds_edit_abandons_mmio_answer() {
    instruction_retry_with_edited_context(0);
}

#[test]
fn held_v86_instruction_vm_edit_abandons_mmio_answer() {
    instruction_retry_with_edited_context(1);
}

#[test]
fn held_protected_instruction_cache_edit_abandons_mmio_answer() {
    instruction_retry_with_edited_context(2);
}

#[test]
fn paged_v86_segment_loads_retain_translations_and_prior_context_blocks() {
    // SDM Vol 3B §23.2; paging design §3.2.
    for model in MODELS {
        let (vm, mut cpu) = world(model, b"\x8e\xd8\x8e\xc0", 3);
        identity_paging(&vm, &mut cpu);
        let mut warm = None;
        for round in 0..4 {
            cpu.cpu.rip = 0x1000;
            canonical(&mut cpu, SegmentId::Ds, 0);
            canonical(&mut cpu, SegmentId::Es, 0);
            cpu.cpu.gpr[0] = 0x200;
            transfer_steps(&mut cpu, &vm, 2);
            assert_eq!(cpu.cpu.legacy.segments[1].base, 0x2000);
            assert_eq!(cpu.cpu.legacy.segments[2].base, 0x2000);
            let proof = cpu.paging_cache_for_test();
            // Both loads share one ordinary V86 block (spec.md §17.10).
            assert_eq!(proof.1, 1);
            if round == 0 {
                warm = Some(proof);
            } else {
                assert_eq!(Some(proof), warm);
            }
        }
    }
}
