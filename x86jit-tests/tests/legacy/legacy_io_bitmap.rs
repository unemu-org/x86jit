//! Machine I/O protection (spec.md §17.10). Printed source pages:
//! 386 PRM §§8.3.1–8.3.2 pp. 8-4–6, §17.2.2.5 p. 17-15,
//! §17 IN p. 17-65, INS pp. 17-67–68, OUT p. 17-122, OUTS pp. 17-123–124;
//! 486 PRM §8.3.2 pp. 8-7–8, §26 IN pp. 26-161–162, INS pp. 26-164–165,
//! OUT pp. 26-225–226, OUTS pp. 26-227–229;
//! SDM Vol 3A §§10.2.1–10.2.2 pp. 10-4–6,
//! Vol 1 §20.5.2 pp. 20-4–5, Vol 2A IN p. 3-455, INS pp. 3-464–465,
//! Vol 2B OUT p. 4-171, OUTS pp. 4-173–175, REP p. 4-564.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::{fault, setup, steps, value};
use x86jit_core::legacy::SystemSegment;
use x86jit_core::{CpuModel, Exit, PortDir, SegmentCache, StepOptions, StepStop, Vcpu, Vm};

const MODELS: [CpuModel; 3] = [
    CpuModel::I80386,
    CpuModel::I80486Dx,
    CpuModel::I80486DxCpuid,
];
const TSS: u64 = 0x3000;
const MAP: u32 = 0x80;

fn world(model: CpuModel, wide: bool, code: &[u8], port: u16) -> (Vm, Vcpu) {
    let (vm, mut cpu) = setup(model, code);
    cpu.cpu.legacy.cpl = 3;
    cpu.cpu.cs = 0x1b;
    cpu.cpu.ss = 0x23;
    cpu.cpu.ds = 0x23;
    cpu.cpu.es = 0x23;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        *cache = SegmentCache {
            base: 0,
            limit: 0xffff,
            attributes: if i == 0 { 0xfb } else { 0xf3 } | if wide { 0x4000 } else { 0 },
            ..Default::default()
        };
    }
    cpu.cpu.legacy.tr = SystemSegment {
        selector: 0x28,
        cache: SegmentCache {
            base: TSS as u32,
            limit: MAP + 0x2000, // Includes the required trailing byte.
            attributes: 0x8b,
            ..Default::default()
        },
        ..Default::default()
    };
    vm.write_bytes(TSS + 0x66, &(MAP as u16).to_le_bytes())
        .unwrap();
    // The task-switch snapshot deliberately disagrees with the live TSS.
    cpu.cpu.task_io_map_base = 0xffff;
    cpu.cpu.gpr[0] = 0x7654_3210;
    cpu.cpu.gpr[1] = 2;
    cpu.cpu.gpr[2] = 0xdead_0000 | u64::from(port);
    cpu.cpu.gpr[6] = 0x6000;
    cpu.cpu.gpr[7] = 0x7000;
    vm.write_bytes(0x6000, &[0xa5; 16]).unwrap();
    vm.write_bytes(0x7000, &[0x5a; 16]).unwrap();
    (vm, cpu)
}

fn instruction(wide: bool, size: u8, opcode: &[u8]) -> Vec<u8> {
    let mut code = vec![];
    if size != 1 && (size == 4) != wide {
        code.push(0x66);
    }
    code.extend_from_slice(opcode);
    code
}

fn deny(vm: &Vm, port: u32) {
    vm.write_bytes(TSS + u64::from(MAP + (port >> 3)), &[1 << (port & 7)])
        .unwrap();
}

fn port_exit(cpu: &mut Vcpu, vm: &Vm, port: u16, size: u8, out: bool) {
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::PortIo { port: p, size: s, dir, .. })
            if *p == port && *s == size && *dir == if out { PortDir::Out } else { PortDir::In }),
        "{run:?}"
    );
}

#[test]
fn scalar_io_checks_every_covered_bit_for_immediate_and_dx_ports() {
    for model in MODELS {
        for wide in [false, true] {
            for size in [1u8, 2, 4] {
                for out in [false, true] {
                    for immediate in [false, true] {
                        // Bit 7 crosses a bitmap byte for both wider transfers.
                        let port = if immediate { 0x27 } else { 0x127 };
                        let opcode = match (out, immediate) {
                            (false, false) => 0xec,
                            (false, true) => 0xe4,
                            (true, false) => 0xee,
                            (true, true) => 0xe6,
                        } + u8::from(size != 1);
                        let mut op = vec![opcode];
                        if immediate {
                            op.push(port as u8);
                        }
                        let code = instruction(wide, size, &op);
                        let (vm, mut cpu) = world(model, wide, &code, port);
                        // Adjacent denied bits must not affect this access.
                        deny(&vm, u32::from(port) - 1);
                        deny(&vm, u32::from(port) + u32::from(size));
                        if immediate {
                            cpu.cpu.gpr[2] = 0x501;
                            deny(&vm, 0x501);
                        }
                        port_exit(&mut cpu, &vm, port, size, out);
                        assert_eq!(cpu.cpu.task_io_map_base, 0xffff);
                        for byte in 0..size {
                            let (vm, mut cpu) = world(model, wide, &code, port);
                            deny(&vm, u32::from(port) + u32::from(byte));
                            fault(&mut cpu, &vm, 13, Some(0));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn tss_limit_bounds_the_pointer_and_both_bitmap_bytes() {
    for model in MODELS {
        for wide in [false, true] {
            for size in [1, 2, 4] {
                let code = instruction(wide, size, &[if size == 1 { 0xec } else { 0xed }]);
                for limit in [0x65, 0x66, MAP - 1, MAP, MAP + 4] {
                    let (vm, mut cpu) = world(model, wide, &code, 0x27);
                    cpu.cpu.legacy.tr.cache.limit = limit;
                    fault(&mut cpu, &vm, 13, Some(0));
                }
                let (vm, mut cpu) = world(model, wide, &code, 0x27);
                cpu.cpu.legacy.tr.cache.limit = MAP + 5;
                port_exit(&mut cpu, &vm, 0x27, size, false);
            }
        }
    }
}

#[test]
fn high_ports_consult_the_trailing_bits_without_wrapping_to_port_zero() {
    // The mask in 386 PRM §17.2.2.5 p. 17-15 spans adjacent bitmap bits.
    for model in MODELS {
        for wide in [false, true] {
            for size in [2, 4] {
                let code = instruction(wide, size, b"\xef");
                let (vm, mut cpu) = world(model, wide, &code, 0xffff);
                deny(&vm, 0);
                port_exit(&mut cpu, &vm, 0xffff, size, true);
                for byte in 1..size {
                    let (vm, mut cpu) = world(model, wide, &code, 0xffff);
                    deny(&vm, 0xffff + u32::from(byte));
                    fault(&mut cpu, &vm, 13, Some(0));
                }
            }
        }
    }
}

#[test]
fn ltr_uses_the_live_map_pointer_and_cached_descriptor() {
    // 386 PRM §17 LTR p. 17-108 loads the descriptor, not the TSS image;
    // §17.2.2.5 p. 17-15 fetches the permission pointer on the I/O access.
    for model in MODELS {
        for wide in [false, true] {
            let (vm, mut cpu) = world(model, wide, b"\x0f\x00\xd8\xec\xec", 0x127);
            fixture::descriptor(&vm, 0x2028, TSS as u32, MAP + 0x2000, 0x89);
            cpu.cpu.legacy.cpl = 0;
            cpu.cpu.gpr[0] = 0x28;
            cpu.cpu.legacy.tr = SystemSegment::default();
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.task_io_map_base, 0xffff);
            // Changing the descriptor in memory must not change the TR cache.
            fixture::descriptor(&vm, 0x2028, 0x9000, 0x67, 0x89);
            cpu.cpu.legacy.cpl = 3;
            port_exit(&mut cpu, &vm, 0x127, 1, false);
            cpu.complete_port_in(0xa5);
            vm.write_bytes(TSS + 0x66, &0xffffu16.to_le_bytes())
                .unwrap();
            fault(&mut cpu, &vm, 13, Some(0));
        }
    }
}

#[test]
fn rep_outs_rechecks_permissions_after_a_port_handoff() {
    for model in MODELS {
        for wide in [false, true] {
            let (vm, mut cpu) = world(model, wide, b"\xf3\x6e", 0x127);
            port_exit(&mut cpu, &vm, 0x127, 1, true);
            assert_eq!(
                (cpu.cpu.rip, cpu.cpu.gpr[1], cpu.cpu.gpr[6]),
                (0x1000, 1, 0x6001)
            );
            deny(&vm, 0x127);
            fault(&mut cpu, &vm, 13, Some(0));
        }
    }
}

#[test]
fn string_io_and_rep_check_before_memory_or_port_effects() {
    for model in MODELS {
        for wide in [false, true] {
            for size in [1, 2, 4] {
                for out in [false, true] {
                    for prefix in [None, Some(0xf3), Some(0xf2)] {
                        let mut op: Vec<u8> = prefix.into_iter().collect();
                        op.push(if out { 0x6e } else { 0x6c } + u8::from(size != 1));
                        let code = instruction(wide, size, &op);
                        for byte in 0..size {
                            let (vm, mut cpu) = world(model, wide, &code, 0x127);
                            deny(&vm, 0x127 + u32::from(byte));
                            // The invalid source/destination must not take priority.
                            cpu.cpu.legacy.segments[if out { 1 } else { 2 }].limit = 0;
                            fault(&mut cpu, &vm, 13, Some(0));
                            assert_eq!(value(&vm, 0x7000, 8), 0x5a5a_5a5a_5a5a_5a5a);
                        }
                        let (vm, mut cpu) = world(model, wide, &code, 0x127);
                        // Finishing the answered INS store charges no step;
                        // the next run also executes this NOP.
                        vm.write_bytes(0x1000 + code.len() as u64, &[0x90]).unwrap();
                        let count = if prefix.is_some() { 2 } else { 1 };
                        for _ in 0..count {
                            port_exit(&mut cpu, &vm, 0x127, size, out);
                            if !out {
                                cpu.complete_port_in(0x1234_5678);
                            }
                        }
                        if !out {
                            steps(&mut cpu, &vm, 1);
                        }
                        let index = if out { 6 } else { 7 };
                        assert_eq!(
                            cpu.cpu.gpr[index],
                            (if out { 0x6000 } else { 0x7000 }) + count * u64::from(size)
                        );
                        if prefix.is_some() {
                            assert_eq!(cpu.cpu.gpr[1], 0);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn privilege_and_zero_rep_count_bypass_an_unavailable_bitmap() {
    for model in MODELS {
        for wide in [false, true] {
            for out in [false, true] {
                let code = [if out { 0xee } else { 0xec }];
                for tss386 in [false, true] {
                    for (cpl, iopl) in [(0, 0), (2, 3), (3, 3), (3, 0)] {
                        let (vm, mut cpu) = world(model, wide, &code, 0x127);
                        cpu.cpu.legacy.cpl = cpl;
                        cpu.cpu.rflags_system = iopl << 12;
                        cpu.cpu.legacy.tr.cache.limit = 0x67;
                        if !tss386 {
                            cpu.cpu.legacy.tr.cache.attributes = 0x83;
                            cpu.cpu.legacy.tr.cache.limit = 0x2b;
                        }
                        if cpl > iopl as u8 {
                            fault(&mut cpu, &vm, 13, Some(0));
                        } else {
                            port_exit(&mut cpu, &vm, 0x127, 1, out);
                        }
                    }
                }
                for address_override in [false, true] {
                    let mut code = vec![];
                    if address_override {
                        code.push(0x67);
                    }
                    code.extend([0xf3, if out { 0x6e } else { 0x6c }]);
                    let (vm, mut cpu) = world(model, wide, &code, 0x127);
                    cpu.cpu.legacy.tr.cache.unusable = 1;
                    cpu.cpu.legacy.tr.cache.base = 0x8000_0000;
                    cpu.cpu.gpr[1] = if wide != address_override {
                        0
                    } else {
                        0x1_0000
                    };
                    steps(&mut cpu, &vm, 1);
                    assert_eq!(cpu.cpu.rip, 0x1000 + code.len() as u64);
                    assert_eq!((cpu.cpu.gpr[6], cpu.cpu.gpr[7]), (0x6000, 0x7000));
                }
            }
        }
    }
}
