//! Internal V86 execution witnesses (v86 design §§2, 7 steps 1–2).
//! Transfers stay gated. Architectural expectations use SDM Vol 3B §§23.2–23.3
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
                } else if code.contains(&0xcf) || code.contains(&0xcd) {
                    let exit = run(&mut cpu, &vm).unwrap_err();
                    assert!(
                        matches!(
                            exit,
                            Exit::UnsupportedSystemContext {
                                operation: UnsupportedSystemOperation::InvalidLegacyContext(
                                    MachineContextError::Virtual8086Unsupported
                                ),
                                ..
                            }
                        ),
                        "{exit:?}"
                    );
                } else if code.contains(&0x9d) {
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
// check. Gate transfer is deliberately deferred; OF=0 completes normally.
#[test]
fn traps_and_ordinary_ret_do_not_gain_an_iopl_guard() {
    for model in MODELS {
        for code in [&[0xcc][..], &[0xce]] {
            let (vm, mut cpu) = world(model, code, 0);
            cpu.cpu.flags.of = true;
            assert!(matches!(
                run(&mut cpu, &vm),
                Err(Exit::UnsupportedSystemContext { .. })
            ));
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
fn public_entry_remains_refused_and_private_entry_validates_canonical_images() {
    for model in MODELS {
        let (vm, mut cpu) = world(model, &[0x90], 3);
        let before = format!("{:?}", cpu.cpu);
        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(
            result.stop(),
            StepStop::Exit(Exit::UnsupportedSystemContext {
                operation: UnsupportedSystemOperation::InvalidLegacyContext(
                    MachineContextError::Virtual8086Unsupported
                ),
                ..
            })
        ));
        assert_eq!(format!("{:?}", cpu.cpu), before);
        assert_eq!(cpu.retired_instructions(), 0);
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].limit = u32::MAX;
        assert!(matches!(
            run(&mut cpu, &vm),
            Err(Exit::UnsupportedSystemContext {
                operation: UnsupportedSystemOperation::InvalidLegacyContext(
                    MachineContextError::InvalidStorage
                ),
                ..
            })
        ));
        cpu.cpu.legacy.segments[SegmentId::Ds as usize].limit = 0xffff;
        cpu.cpu.legacy.cpl = 0;
        assert!(matches!(
            run(&mut cpu, &vm),
            Err(Exit::UnsupportedSystemContext {
                operation: UnsupportedSystemOperation::InvalidLegacyContext(
                    MachineContextError::InvalidPrivilege
                ),
                ..
            })
        ));
    }
}

#[test]
fn i486_alignment_boundary_is_retained_in_v86() {
    for model in [CpuModel::I80486Dx, CpuModel::I80486DxCpuid] {
        let (vm, mut cpu) = world(model, &[0x90], 3);
        cpu.cpu.legacy.cr0 |= 1 << 18;
        cpu.cpu.rflags_system |= 1 << 18;
        assert!(matches!(
            run(&mut cpu, &vm),
            Err(Exit::UnsupportedSystemContext {
                operation: UnsupportedSystemOperation::LegacyAlignmentCheck,
                ..
            })
        ));
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
