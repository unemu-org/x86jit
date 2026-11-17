//! SDM Vol 2B §4.3, REP Description/Operation (pp. 4-563–4-564),
//! REPE/REPNE (pp. 4-565–4-568), MOVS/STOS/LODS/CMPS/SCAS:
//! each completed element advances the address-size indices and repeat count;
//! suspension preserves the next element and the string instruction's IP.
use x86jit_core::{
    Backend, CpuMode, Exit, InterpreterBackend, Prot, Reg, RegionKind, StepOptions, StepStop, Vm,
    VmConfig,
};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;
const DEVICE: u64 = 0x4000;
const RAM: u64 = 0x8000;
const VALUE: u64 = 0x8877_6655_4433_2222;

fn backend(jit: bool) -> Box<dyn Backend> {
    if jit {
        Box::new(JitBackend::new())
    } else {
        Box::new(InterpreterBackend)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Movs,
    Stos,
    Lods,
    Cmps,
    Scas,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    mode: CpuMode,
    op: Op,
    elem: u8,
    repeat: u8,
    backwards: bool,
    address_override: bool,
    source_trap: bool,
    dest_trap: bool,
    alias: bool,
}

fn exercise(jit: bool, c: Case) {
    let real = c.mode == CpuMode::Real16;
    let mut vm = Vm::with_backend(VmConfig::flat(0x20000), backend(jit));
    vm.set_cpu_mode(c.mode);
    vm.map(0, DEVICE as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(DEVICE, 0x2000, Prot::RW, RegionKind::Trap).unwrap();
    vm.map(0x6000, 0x1a000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut code = Vec::new();
    if c.repeat != 0 {
        code.push(c.repeat);
    }
    if c.address_override {
        code.push(0x67);
    }
    if c.elem == 2 && !real || c.elem == 4 && real {
        code.push(0x66);
    }
    if c.elem == 8 {
        code.push(0x48);
    }
    code.push(
        match c.op {
            Op::Movs => 0xa4,
            Op::Stos => 0xaa,
            Op::Lods => 0xac,
            Op::Cmps => 0xa6,
            Op::Scas => 0xae,
        } + u8::from(c.elem != 1),
    );
    let instruction_len = code.len() as u64;
    code.push(0xf4);
    vm.write_bytes(CODE, &code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    // Distinct nonzero segment bases, with low pointer offsets in real mode.
    let src_base = if real { 0x1000 } else { 0 };
    let dst_base = if real { 0x2000 } else { 0 };
    if real {
        cpu.set_reg(Reg::Ds, src_base >> 4);
        cpu.set_reg(Reg::Es, dst_base >> 4);
    }
    let source = if c.source_trap { DEVICE } else { RAM } + 0x80;
    let dest = if c.alias {
        source
    } else {
        (if c.dest_trap {
            DEVICE + 0x1000
        } else {
            RAM + 0x1000
        }) + 0x80
    };
    let addr_bits = if real {
        if c.address_override {
            32
        } else {
            16
        }
    } else if c.mode == CpuMode::Compat32 {
        if c.address_override {
            16
        } else {
            32
        }
    } else if c.address_override {
        32
    } else {
        64
    };
    let high = if addr_bits != 64 {
        0x1234_0000_0000_0000
    } else {
        0
    };
    let initial_si = high | (source - src_base);
    let initial_di = high | (dest - dst_base);
    cpu.set_reg(Reg::Rsi, initial_si);
    cpu.set_reg(Reg::Rdi, initial_di);
    cpu.set_reg(Reg::Rcx, high | 3);
    cpu.set_reg(Reg::Rax, VALUE);
    cpu.cpu.flags.df = c.backwards;
    let count = if c.repeat == 0 { 1 } else { 3 };
    let mask = u64::MAX >> (64 - c.elem as u32 * 8);
    let delta = if c.backwards {
        -(c.elem as i64)
    } else {
        c.elem as i64
    };
    let index = |reg: u64, done: u64| {
        let value = reg.wrapping_add((delta * done as i64) as u64);
        match addr_bits {
            16 => (reg & !0xffff) | (value & 0xffff),
            32 => value & 0xffff_ffff,
            _ => value,
        }
    };
    let address = |at: u64, i: u64| at.wrapping_add((delta * i as i64) as u64);
    let value = |i: u64| (VALUE + i) & mask;
    let accumulator = |done: u64| {
        if c.op != Op::Lods || done == 0 {
            VALUE
        } else if c.elem < 4 {
            (VALUE & !mask) | value(done - 1)
        } else {
            value(done - 1)
        }
    };
    let entry_flags = cpu.cpu.flags.to_rflags();
    for i in 0..count {
        if !c.source_trap {
            vm.write_bytes(
                address(source, i),
                &value(i).to_le_bytes()[..c.elem as usize],
            )
            .unwrap();
        }
        if !c.dest_trap && matches!(c.op, Op::Cmps | Op::Scas) {
            let v = if c.op == Op::Scas {
                VALUE & mask
            } else {
                value(i)
            };
            let v = if c.repeat == 0xf2 { v ^ 1 } else { v };
            vm.write_bytes(address(dest, i), &v.to_le_bytes()[..c.elem as usize])
                .unwrap();
        }
    }
    let reads_source = matches!(c.op, Op::Movs | Op::Lods | Op::Cmps);
    let uses_dest = c.op != Op::Lods;
    let source_exits = reads_source && c.source_trap;
    let dest_exits = uses_dest && c.dest_trap;
    let per_element = u64::from(source_exits) + u64::from(dest_exits);
    let mut seen = 0;
    for round in 0..16 {
        let exit = cpu.run(&vm, None);
        match exit {
            Exit::MmioRead { addr, size, .. } | Exit::MmioWrite { addr, size, .. } => {
                let i = seen / per_element;
                assert!(
                    i < count,
                    "repeated transfer: {c:?}, {exit:?}, round {round}"
                );
                let dest_transfer = !source_exits || seen % per_element == 1;
                assert_eq!(
                    matches!(exit, Exit::MmioWrite { .. }),
                    dest_transfer && matches!(c.op, Op::Movs | Op::Stos),
                    "{c:?}"
                );
                assert_eq!(cpu.reg(Reg::Rax), accumulator(i), "{c:?}");
                if matches!(c.op, Op::Cmps | Op::Scas) {
                    assert_eq!(cpu.cpu.flags.zf, i != 0 && c.repeat != 0xf2, "{c:?}");
                } else {
                    assert_eq!(cpu.cpu.flags.to_rflags(), entry_flags, "{c:?}");
                }
                assert_eq!(
                    addr,
                    address(if dest_transfer { dest } else { source }, i),
                    "{c:?}"
                );
                assert_eq!(size, c.elem, "{c:?}");
                assert_eq!(cpu.reg(Reg::Rip), CODE, "{c:?}");
                assert_eq!(
                    cpu.reg(Reg::Rcx),
                    if c.repeat == 0 || i == 0 {
                        high | 3
                    } else {
                        (if addr_bits == 16 { high } else { 0 }) | (3 - i)
                    },
                    "{c:?}"
                );
                assert_eq!(
                    cpu.reg(Reg::Rsi),
                    if reads_source && i != 0 {
                        index(initial_si, i)
                    } else {
                        initial_si
                    },
                    "{c:?}"
                );
                assert_eq!(
                    cpu.reg(Reg::Rdi),
                    if uses_dest && i != 0 {
                        index(initial_di, i)
                    } else {
                        initial_di
                    },
                    "{c:?}"
                );
                if let Exit::MmioWrite { value: v, .. } = exit {
                    assert_eq!(
                        v,
                        if c.op == Op::Stos {
                            VALUE & mask
                        } else {
                            value(i)
                        },
                        "{c:?}"
                    );
                    cpu.complete_mmio_write();
                } else {
                    let v = if c.op == Op::Scas {
                        VALUE & mask
                    } else {
                        value(i)
                    };
                    let v = if dest_transfer && c.repeat == 0xf2 {
                        v ^ 1
                    } else {
                        v
                    };
                    cpu.complete_mmio_read(v);
                }
                seen += 1;
            }
            Exit::Hlt => {
                assert_eq!(seen, count * per_element, "missing MMIO: {c:?}");
                assert_eq!(cpu.reg(Reg::Rip), CODE + instruction_len + 1);
                assert_eq!(
                    cpu.reg(Reg::Rsi),
                    if reads_source {
                        index(initial_si, count)
                    } else {
                        initial_si
                    }
                );
                assert_eq!(
                    cpu.reg(Reg::Rdi),
                    if uses_dest {
                        index(initial_di, count)
                    } else {
                        initial_di
                    }
                );
                assert_eq!(
                    cpu.reg(Reg::Rcx),
                    if c.repeat == 0 {
                        high | 3
                    } else if addr_bits == 16 {
                        high
                    } else {
                        0
                    }
                );
                assert_eq!(cpu.reg(Reg::Rax), accumulator(count), "{c:?}");
                if matches!(c.op, Op::Cmps | Op::Scas) {
                    assert_eq!(cpu.cpu.flags.zf, c.repeat != 0xf2);
                }
                if c.op == Op::Movs && !c.dest_trap {
                    for i in 0..count {
                        assert_eq!(vm.mem.read(address(dest, i), c.elem).unwrap(), value(i));
                    }
                }
                return;
            }
            other => panic!("{c:?}: {other:?}"),
        }
    }
    panic!("string MMIO failed to converge: {c:?}");
}

fn matrix(jit: bool, mode: CpuMode) {
    for op in [Op::Movs, Op::Stos, Op::Lods, Op::Cmps, Op::Scas] {
        for elem in [1, 2, 4, 8] {
            if (mode == CpuMode::Real16 && elem > 2) || (mode == CpuMode::Compat32 && elem == 8) {
                continue;
            }
            for repeat in [0, 0xf3, 0xf2] {
                if repeat == 0xf2 && !matches!(op, Op::Cmps | Op::Scas) {
                    continue;
                }
                for backwards in [false, true] {
                    for address_override in [false, true] {
                        for (source_trap, dest_trap, alias) in [
                            (true, false, false),
                            (false, true, false),
                            (true, true, false),
                            (true, true, true),
                        ] {
                            if matches!(op, Op::Stos | Op::Scas) && source_trap {
                                continue;
                            }
                            if op == Op::Lods && dest_trap {
                                continue;
                            }
                            exercise(
                                jit,
                                Case {
                                    mode,
                                    op,
                                    elem,
                                    repeat,
                                    backwards,
                                    address_override,
                                    source_trap,
                                    dest_trap,
                                    alias,
                                },
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn long64_interp() {
    matrix(false, CpuMode::Long64);
}
#[test]
fn long64_jit() {
    matrix(true, CpuMode::Long64);
}
#[test]
fn real16_interp() {
    matrix(false, CpuMode::Real16);
}
#[test]
fn real16_jit_backend() {
    matrix(true, CpuMode::Real16);
}

fn boundary(jit: bool, request: bool, mode: CpuMode) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(jit));
    vm.set_cpu_mode(mode);
    vm.map(0, DEVICE as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(DEVICE, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.write_bytes(
        CODE,
        if mode == CpuMode::Real16 {
            &[0xf3, 0xab, 0xf4]
        } else {
            &[0xf3, 0x66, 0xab, 0xf4]
        },
    )
    .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rdi, DEVICE);
    cpu.set_reg(Reg::Rcx, 2);
    cpu.set_reg(Reg::Rax, 0x1234);
    assert!(matches!(
        cpu.run(&vm, None),
        Exit::MmioWrite {
            addr: DEVICE,
            size: 2,
            value: 0x1234,
            ..
        }
    ));
    cpu.complete_mmio_write();
    if request {
        cpu.exit_request().request_exit();
        assert!(matches!(cpu.run(&vm, None), Exit::Interrupted));
    } else {
        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(result.steps(), 1);
        assert!(matches!(
            result.stop(),
            StepStop::Exit(Exit::BudgetExhausted)
        ));
    }
    assert_eq!(cpu.reg(Reg::Rcx), 1);
    assert_eq!(cpu.reg(Reg::Rdi), DEVICE + 2);
    assert_eq!(cpu.reg(Reg::Rip), CODE);
    assert!(matches!(cpu.run(&vm, None), Exit::MmioWrite { addr, .. } if addr == DEVICE + 2));
    cpu.complete_mmio_write();
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
}
#[test]
fn budget_interp() {
    boundary(false, false, CpuMode::Real16);
}
#[test]
fn budget_jit_backend() {
    boundary(true, false, CpuMode::Real16);
}
#[test]
fn request_interp() {
    boundary(false, true, CpuMode::Real16);
}
#[test]
fn request_jit_backend() {
    boundary(true, true, CpuMode::Real16);
}

#[test]
fn request_long64_interp() {
    boundary(false, true, CpuMode::Long64);
}
#[test]
fn request_long64_jit() {
    boundary(true, true, CpuMode::Long64);
}

fn request_between_transfers(jit: bool) {
    for mode in [CpuMode::Real16, CpuMode::Long64] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(jit));
        vm.set_cpu_mode(mode);
        vm.map(0, DEVICE as usize, Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.map(DEVICE, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.write_bytes(CODE, &[0xf3, 0xa4, 0xf4]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rsi, DEVICE);
        cpu.set_reg(Reg::Rdi, DEVICE);
        cpu.set_reg(Reg::Rcx, 2);
        assert!(matches!(
            cpu.run(&vm, None),
            Exit::MmioRead { addr: DEVICE, .. }
        ));
        cpu.complete_mmio_read(0x5a);
        cpu.exit_request().request_exit();
        assert!(matches!(
            cpu.run(&vm, None),
            Exit::MmioWrite {
                addr: DEVICE,
                value: 0x5a,
                ..
            }
        ));
        assert_eq!(cpu.reg(Reg::Rcx), 2);
        cpu.complete_mmio_write();
        assert!(matches!(cpu.run(&vm, None), Exit::Interrupted));
        assert_eq!(cpu.reg(Reg::Rcx), 1);
        assert_eq!(cpu.reg(Reg::Rsi), DEVICE + 1);
        assert_eq!(cpu.reg(Reg::Rdi), DEVICE + 1);
        assert!(matches!(cpu.run(&vm, None), Exit::MmioRead { addr, .. } if addr == DEVICE + 1));
        cpu.complete_mmio_read(0xa5);
        assert!(
            matches!(cpu.run(&vm, None), Exit::MmioWrite { addr, value: 0xa5, .. } if addr == DEVICE + 1)
        );
        cpu.complete_mmio_write();
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    }
}

#[test]
fn request_between_transfers_interp() {
    request_between_transfers(false);
}
#[test]
fn request_between_transfers_jit() {
    request_between_transfers(true);
}

fn ram_to_trap(jit: bool) {
    for mode in [CpuMode::Real16, CpuMode::Long64] {
        for backwards in [false, true] {
            for stos in [false, true] {
                let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(jit));
                vm.set_cpu_mode(mode);
                vm.map(0, DEVICE as usize, Prot::RWX, RegionKind::Ram)
                    .unwrap();
                vm.map(DEVICE, 0x2000, Prot::RW, RegionKind::Trap).unwrap();
                vm.map(0x6000, 0xa000, Prot::RW, RegionKind::Ram).unwrap();
                let mut code = vec![0xf3];
                if mode == CpuMode::Long64 {
                    code.push(0x66);
                }
                code.extend_from_slice(&[if stos { 0xab } else { 0xa5 }, 0xf4]);
                vm.write_bytes(CODE, &code).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, CODE);
                let first = if backwards {
                    DEVICE + 0x2000
                } else {
                    DEVICE - 2
                };
                let delta = if backwards { u64::MAX - 1 } else { 2 };
                cpu.set_reg(Reg::Rsi, first);
                cpu.set_reg(Reg::Rdi, if stos { first } else { RAM });
                cpu.set_reg(Reg::Rcx, 3);
                cpu.set_reg(Reg::Rax, 0x1234);
                cpu.cpu.flags.df = backwards;
                vm.write_bytes(first, &0x1234u16.to_le_bytes()).unwrap();
                for i in 1..3 {
                    let expected = first.wrapping_add(delta.wrapping_mul(i));
                    let exit = cpu.run(&vm, None);
                    assert_eq!(cpu.reg(Reg::Rcx), 3 - i);
                    assert_eq!(cpu.reg(if stos { Reg::Rdi } else { Reg::Rsi }), expected);
                    assert_eq!(cpu.reg(Reg::Rip), CODE);
                    if stos {
                        assert!(
                            matches!(exit, Exit::MmioWrite { addr, size: 2, value: 0x1234, .. } if addr == expected)
                        );
                        cpu.complete_mmio_write();
                    } else {
                        assert!(
                            matches!(exit, Exit::MmioRead { addr, size: 2, .. } if addr == expected)
                        );
                        cpu.complete_mmio_read(0x1234 + i);
                    }
                }
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                assert_eq!(cpu.reg(Reg::Rcx), 0);
                if !stos {
                    for i in 0..3 {
                        assert_eq!(
                            vm.mem
                                .read(RAM.wrapping_add(delta.wrapping_mul(i)), 2)
                                .unwrap(),
                            0x1234 + i
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn ram_to_trap_interp() {
    ram_to_trap(false);
}
#[test]
fn ram_to_trap_jit() {
    ram_to_trap(true);
}

#[test]
fn ram_between_devices_stays_compiled_and_device_writes_exit_per_element() {
    let mut vm = Vm::with_backend(VmConfig::flat(0x20000), backend(true));
    for (start, len, kind) in [
        (0, 0x4000, RegionKind::Ram),
        (0x4000, 0x1000, RegionKind::Trap),
        (0x5000, 0x13000, RegionKind::Ram),
        (0x18000, 0x1000, RegionKind::Trap),
        (0x19000, 0x7000, RegionKind::Ram),
    ] {
        vm.map(start, len, Prot::RWX, kind).unwrap();
    }
    vm.set_time_accounting(true);
    vm.write_bytes(0x8000, &[0x11, 0x22, 0x33]).unwrap();
    let mut expected_calls = 0;
    for addr32 in [false, true] {
        let code = if addr32 {
            &[0x67, 0xf3, 0xa4, 0xf4][..]
        } else {
            &[0xf3, 0xa4, 0xf4][..]
        };
        vm.write_bytes(CODE, code).unwrap();
        for df in [false, true] {
            for count in [0, 3] {
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, CODE);
                let high = if addr32 { 0x1234_0000_0000_0000 } else { 0 };
                let offset = if df { 2 } else { 0 };
                cpu.set_reg(Reg::Rcx, high | count);
                // A zero-count repeat may point at a device without touching it.
                cpu.set_reg(
                    Reg::Rsi,
                    high | if count == 0 { DEVICE } else { 0x8000 + offset },
                );
                cpu.set_reg(
                    Reg::Rdi,
                    high | if count == 0 { DEVICE } else { 0x10000 + offset },
                );
                cpu.cpu.flags.df = df;
                let before = vm.time_stats().helpers.calls;
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                // The generated helper counter proves execution of the raw string
                // helper. The run counter would include a second call for slow replay.
                expected_calls += 1;
                assert_eq!(
                    vm.backend.helper_calls(),
                    vec![("string_helper::<true>", expected_calls)]
                );
                assert_eq!(
                    vm.time_stats().helpers.calls - before,
                    1,
                    "interpreter replay occurred"
                );
                if count != 0 {
                    let mut bytes = [0; 3];
                    vm.read_bytes(0x10000, &mut bytes).unwrap();
                    assert_eq!(bytes, [0x11, 0x22, 0x33]);
                }
            }
        }
    }
    vm.write_bytes(CODE, &[0xf3, 0xa4, 0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rsi, 0x8000);
    cpu.set_reg(Reg::Rdi, DEVICE);
    cpu.set_reg(Reg::Rcx, 3);
    for (i, value) in [0x11, 0x22, 0x33].into_iter().enumerate() {
        assert!(
            matches!(cpu.run(&vm, None), Exit::MmioWrite { addr, size: 1, value: v, .. } if addr == DEVICE + i as u64 && v == value)
        );
        assert_eq!(cpu.reg(Reg::Rcx), 3 - i as u64);
        cpu.complete_mmio_write();
    }
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
}

#[test]
fn compat32_interp() {
    matrix(false, CpuMode::Compat32);
}
#[test]
fn compat32_jit() {
    matrix(true, CpuMode::Compat32);
}

fn warmed_memory_remap(jit: bool) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(jit));
    vm.map(0, DEVICE as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(DEVICE, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
    vm.write_bytes(CODE, &[0xf3, 0xaa, 0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rdi, DEVICE);
    cpu.set_reg(Reg::Rcx, 2);
    cpu.set_reg(Reg::Rax, 0x5a);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    if jit {
        assert_eq!(
            vm.backend
                .helper_calls()
                .iter()
                .map(|(_, n)| n)
                .sum::<u64>(),
            1
        );
    }
    vm.write_bytes(DEVICE, &[0xcc; 2]).unwrap();
    let misses = vm.cache.misses();
    vm.mem.unmap(DEVICE, 0x1000).unwrap();
    vm.mem
        .map(DEVICE, 0x1000, Prot::RW, RegionKind::Trap)
        .unwrap();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rdi, DEVICE);
    cpu.set_reg(Reg::Rcx, 2);
    for i in 0..2 {
        assert!(matches!(cpu.run(&vm, None), Exit::MmioWrite {
            addr, size: 1, value: 0x5a, ..
        } if addr == DEVICE + i));
        assert_eq!(cpu.reg(Reg::Rcx), 2 - i);
        assert_eq!(vm.cache.misses(), misses, "warmed translation was replaced");
        cpu.complete_mmio_write();
    }
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let mut backing = [0; 2];
    vm.read_bytes(DEVICE, &mut backing).unwrap();
    assert_eq!(backing, [0xcc; 2]);
}
#[test]
fn warmed_memory_remap_interp() {
    warmed_memory_remap(false);
}
#[test]
fn warmed_memory_remap_jit() {
    warmed_memory_remap(true);
}

// SDM Vol 2B §4.3, REP Description/Operation (pp. 4-563–4-564),
// STOS Description/Operation (pp. 4-676–4-677), MOVS Description/Operation
// (pp. 4-104–4-105): address size selects ECX/ESI/EDI under Real16's 67h.
fn real16_addr32(jit: bool) {
    for stepped in [false, true] {
        for movs in [false, true] {
            let mut vm = Vm::with_backend(VmConfig::flat(0x30000), backend(jit));
            vm.set_cpu_mode(CpuMode::Real16);
            vm.map(0, 0x14000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x15000, 0x1b000, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.mem
                .map(0x14000, 0x1000, Prot::RW, RegionKind::Trap)
                .unwrap();
            vm.write_bytes(CODE, &[0xf3, 0x67, if movs { 0xa4 } else { 0xaa }, 0xf4])
                .unwrap();
            vm.write_bytes(0x18000, &[0x5a]).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Ds, 0x1000);
            cpu.set_reg(Reg::Es, 0x1000);
            cpu.set_reg(Reg::Rsi, 0x8000);
            cpu.set_reg(Reg::Rdi, 0x4000);
            cpu.set_reg(Reg::Rcx, 0x10000);
            cpu.set_reg(Reg::Rax, 0x5a);
            let exit = if stepped {
                let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                match result.stop() {
                    StepStop::Exit(e) => e.clone(),
                    other => panic!("{other:?}"),
                }
            } else {
                cpu.run(&vm, None)
            };
            assert!(matches!(
                exit,
                Exit::MmioWrite {
                    addr: 0x14000,
                    size: 1,
                    value: 0x5a,
                    ..
                }
            ));
            cpu.complete_mmio_write();
            let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(matches!(
                result.stop(),
                StepStop::Exit(Exit::BudgetExhausted)
            ));
            assert_eq!(cpu.reg(Reg::Rcx), 0xffff);
            assert_eq!(cpu.reg(Reg::Rdi), 0x4001);
            assert_eq!(cpu.reg(Reg::Rsi), if movs { 0x8001 } else { 0x8000 });
        }
    }
    for movs in [false, true] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(jit));
        vm.set_cpu_mode(CpuMode::Real16);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(CODE, &[0xf3, 0x67, if movs { 0xa4 } else { 0xaa }, 0xf4])
            .unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rcx, 1);
        cpu.cpu.flags.df = true;
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rdi), 0xffff_ffff);
        assert_eq!(cpu.reg(Reg::Rsi), if movs { 0xffff_ffff } else { 0 });
    }
}
#[test]
fn real16_addr32_interp() {
    real16_addr32(false);
}
#[test]
fn real16_addr32_jit_backend() {
    real16_addr32(true);
}

// SDM Vol 2B §4.3, MOVS Description (pp. 4-104–4-105), SDM Vol 2A §3.3, LODS Description
// (p. 3-567), CMPS Description (pp. 3-181–3-182): only the source segment
// can be overridden; Long64 retains FS/GS bases.
fn segment_sources(jit: bool) {
    for (prefix, segment) in [(0x64, Reg::FsBase), (0x65, Reg::GsBase)] {
        for opcode in [0xa4, 0xac, 0xa6] {
            for addr32 in [false, true] {
                let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(jit));
                vm.map(0, DEVICE as usize, Prot::RWX, RegionKind::Ram)
                    .unwrap();
                vm.map(DEVICE, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                vm.map(RAM, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
                let mut code = vec![0xf3, prefix];
                if addr32 {
                    code.push(0x67);
                }
                code.extend([opcode, 0xf4]);
                vm.write_bytes(CODE, &code).unwrap();
                vm.write_bytes(RAM, &[0x31, 0x72]).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, CODE);
                cpu.set_reg(segment, DEVICE);
                let high = if addr32 { 1 << 40 } else { 0 };
                cpu.set_reg(Reg::Rsi, high | 0x80);
                cpu.set_reg(Reg::Rdi, high | RAM);
                cpu.set_reg(Reg::Rcx, high | 2);
                for (i, value) in [0x31, 0x72].into_iter().enumerate() {
                    assert!(
                        matches!(cpu.run(&vm, None), Exit::MmioRead { addr, size: 1, .. }
                        if addr == DEVICE + 0x80 + i as u64)
                    );
                    cpu.complete_mmio_read(value);
                }
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                assert_eq!(cpu.reg(Reg::Rsi), 0x82);
                assert_eq!(
                    cpu.reg(Reg::Rdi),
                    if opcode == 0xac { high | RAM } else { RAM + 2 }
                );
                if opcode == 0xac {
                    assert_eq!(cpu.reg(Reg::Rax), 0x72);
                }
                if opcode == 0xa6 {
                    assert!(cpu.cpu.flags.zf);
                }
                let mut bytes = [0; 2];
                vm.read_bytes(RAM, &mut bytes).unwrap();
                assert_eq!(bytes, [0x31, 0x72]);
            }
        }
    }
}
#[test]
fn segment_sources_interp() {
    segment_sources(false);
}
#[test]
fn segment_sources_jit() {
    segment_sources(true);
}

// SDM Vol 2A §3.3, CMPS Flags Affected (p. 3-183); Vol 2B §4.3,
// SCAS Flags Affected (p. 4-617), REPE/REPNE Description/Operation (pp. 4-565–4-568).
fn compare_flags_and_termination(jit: bool) {
    const ARITH: u64 = 0x8d5;
    // Explicit subtraction witnesses: borrow, signed overflow, sign, zero,
    // auxiliary borrow and low-byte parity. No engine ALU computes the oracle.
    for mode in [CpuMode::Real16, CpuMode::Compat32, CpuMode::Long64] {
        for opcode in [0xa6, 0xae] {
            for (prefix, comparisons) in [
                (0xf3, vec![(0x80, 0x80, 0x44), (0x80, 1, 0x810)]),
                (0xf3, vec![(0, 1, 0x95)]),
                (0xf3, vec![(0x7f, 0xff, 0x881)]),
                (0xf2, vec![(0, 1, 0x95), (0, 0, 0x44)]),
            ] {
                let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(jit));
                vm.set_cpu_mode(mode);
                vm.map(0, DEVICE as usize, Prot::RWX, RegionKind::Ram)
                    .unwrap();
                vm.map(DEVICE, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                vm.map(RAM, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
                vm.write_bytes(CODE, &[prefix, opcode, 0xf4]).unwrap();
                for (i, (left, _, _)) in comparisons.iter().enumerate() {
                    vm.write_bytes(RAM + i as u64, &[*left]).unwrap();
                }
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, CODE);
                cpu.set_reg(Reg::Rsi, RAM);
                cpu.set_reg(Reg::Rdi, DEVICE);
                cpu.set_reg(Reg::Rcx, 4);
                cpu.set_reg(Reg::Rax, u64::from(comparisons[0].0));
                let mut expected = cpu.cpu.flags.to_rflags() & ARITH;
                for (i, (_, right, flags)) in comparisons.iter().enumerate() {
                    assert!(
                        matches!(cpu.run(&vm, None), Exit::MmioRead { addr, size: 1, .. }
                        if addr == DEVICE + i as u64)
                    );
                    assert_eq!(
                        cpu.cpu.flags.to_rflags() & ARITH,
                        expected,
                        "flags committed before answer"
                    );
                    assert_eq!(cpu.reg(Reg::Rcx), 4 - i as u64);
                    cpu.complete_mmio_read(*right as u64);
                    expected = *flags;
                }
                assert!(
                    matches!(cpu.run(&vm, None), Exit::Hlt),
                    "repeat failed to stop early"
                );
                let done = comparisons.len() as u64;
                assert_eq!(
                    cpu.cpu.flags.to_rflags() & ARITH,
                    expected,
                    "{mode:?} {opcode:x} {prefix:x}"
                );
                assert_eq!(cpu.reg(Reg::Rcx), 4 - done);
                assert_eq!(cpu.reg(Reg::Rdi), DEVICE + done);
                assert_eq!(
                    cpu.reg(Reg::Rsi),
                    RAM + if opcode == 0xa6 { done } else { 0 }
                );
            }
        }
    }
}
#[test]
fn compare_flags_and_termination_interp() {
    compare_flags_and_termination(false);
}
#[test]
fn compare_flags_and_termination_jit() {
    compare_flags_and_termination(true);
}

fn completion_ownership(jit: bool) {
    for mode in [CpuMode::Real16, CpuMode::Compat32, CpuMode::Long64] {
        for write in [false, true] {
            let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(jit));
            vm.set_cpu_mode(mode);
            vm.map(0, DEVICE as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.map(DEVICE, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
            // REP LODSB / MOV AL,[BX/EBX/RBX], or REP STOSB / MOV [BX/EBX/RBX],DL.
            let modrm = if mode == CpuMode::Real16 { 0x07 } else { 0x03 };
            vm.write_bytes(
                CODE,
                &[
                    0xf3,
                    if write { 0xaa } else { 0xac },
                    if write { 0x88 } else { 0x8a },
                    modrm | if write { 0x10 } else { 0 },
                    0xf4,
                ],
            )
            .unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rsi, DEVICE);
            cpu.set_reg(Reg::Rdi, DEVICE);
            cpu.set_reg(Reg::Rcx, 2);
            cpu.set_reg(Reg::Rbx, DEVICE + 1);
            cpu.set_reg(Reg::Rax, 0x5a);
            cpu.set_reg(Reg::Rdx, 0xa5);
            for (i, value) in [0x31, 0x72].into_iter().enumerate() {
                let exit = cpu.run(&vm, None);
                if write {
                    assert!(
                        matches!(exit, Exit::MmioWrite { addr, size: 1, value: 0x5a, .. }
                        if addr == DEVICE + i as u64)
                    );
                    cpu.complete_mmio_write();
                } else {
                    assert!(matches!(exit, Exit::MmioRead { addr, size: 1, .. }
                        if addr == DEVICE + i as u64));
                    if i == 1 {
                        assert_eq!(cpu.reg(Reg::Rax), 0x31);
                    }
                    cpu.complete_mmio_read(value);
                }
            }
            let exit = cpu.run(&vm, None);
            assert_eq!(cpu.reg(Reg::Rip), CODE + 2);
            assert_eq!(cpu.reg(Reg::Rcx), 0);
            if write {
                assert!(
                    matches!(exit, Exit::MmioWrite { addr, size: 1, value: 0xa5, .. } if addr == DEVICE + 1)
                );
                cpu.complete_mmio_write();
            } else {
                assert!(matches!(exit, Exit::MmioRead { addr, size: 1, .. } if addr == DEVICE + 1));
                assert_eq!(cpu.reg(Reg::Rax), 0x72);
                cpu.complete_mmio_read(0xa5);
            }
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), if write { 0x5a } else { 0xa5 });
            let mut backing = [0; 2];
            vm.read_bytes(DEVICE, &mut backing).unwrap();
            assert_eq!(backing, [0; 2]);
        }
    }
}
#[test]
fn completion_ownership_interp() {
    completion_ownership(false);
}
#[test]
fn completion_ownership_jit() {
    completion_ownership(true);
}

#[test]
fn child_trap_intervals_differ_from_parent() {
    use x86jit_core::{
        child, ChildArena, ChildRefusal, ChildRegion, ChildStep, ChildViewMetadata, FetchVerdict,
    };
    for jit in [false, true] {
        for opcode in [0xa4, 0xaa, 0xac, 0xa6, 0xae] {
            let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(jit));
            vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(CODE, &[0xf3, opcode, 0xf4]).unwrap();
            vm.write_bytes(DEVICE, &[0xcc; 4]).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(
                Reg::Rsi,
                if matches!(opcode, 0xa4 | 0xac | 0xa6) {
                    DEVICE
                } else {
                    RAM
                },
            );
            cpu.set_reg(Reg::Rdi, if opcode == 0xac { RAM } else { DEVICE });
            cpu.set_reg(Reg::Rcx, 2);
            cpu.set_reg(Reg::Rax, 0xcc);
            let initial = cpu.cpu.clone();
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            vm.write_bytes(DEVICE, &[0xcc; 4]).unwrap();
            let view = vm
                .child_view_with(
                    ChildViewMetadata {
                        regions: vec![
                            ChildRegion::Ram {
                                range: 0..DEVICE,
                                backing_offset: 0,
                                fetch: FetchVerdict::Allow,
                            },
                            ChildRegion::Trap {
                                range: DEVICE..DEVICE + 0x1000,
                            },
                            ChildRegion::Ram {
                                range: DEVICE + 0x1000..0x10000,
                                backing_offset: (DEVICE + 0x1000) as usize,
                                fetch: FetchVerdict::Allow,
                            },
                        ],
                        code_store_refusals: vec![],
                    },
                    CpuMode::Long64,
                )
                .unwrap();
            let mut child_cpu = initial.clone();
            let mut arena = ChildArena::new(16);
            assert!(matches!(
                child::step(&view, &mut child_cpu, &mut arena),
                ChildStep::Refused(ChildRefusal::Mmio)
            ));
            assert_eq!(format!("{child_cpu:?}"), format!("{initial:?}"));
            let mut backing = [0; 4];
            vm.read_bytes(DEVICE, &mut backing).unwrap();
            assert_eq!(backing, [0xcc; 4]);
        }
    }
}

/// What the driver does at a device stop of a string element, around its
/// completion, before the retry runs.
#[derive(Clone, Copy, Debug)]
enum Between {
    /// Complete it, then `run`.
    Run,
    /// Request a host stop and `run` before the completion: Long64 and
    /// Compat32 take the stop at the boundary before the retry. Then complete
    /// it and `run`.
    HostFirst,
    /// As `HostFirst`, then `run` once more before the completion and answer
    /// whatever that run asks for.
    HostRepeat,
    /// `run(Some(0))` before the completion.
    ZeroFirst,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bus {
    Read(u64, u8),
    Write(u64, u8, u64),
}

/// A device whose answer names the byte it answers: its address's low byte
/// plus one in each byte of the transfer.
fn device_answer(addr: u64, size: u8) -> u64 {
    (0..u64::from(size)).fold(0, |v, i| v | (((addr + i + 1) & 0xff) << (8 * i)))
}

/// Drive `Vcpu::run` to a stop other than a device, budget or host stop,
/// applying `schedule[i % len]` at the i-th device stop. `None` after twelve
/// transfers: the instruction does not converge.
fn drive_between(
    vm: &Vm,
    cpu: &mut x86jit_core::Vcpu,
    schedule: &[Between],
) -> (Vec<Bus>, Option<Exit>) {
    let mut bus = Vec::new();
    let mut stops = 0;
    let mut exit = cpu.run(vm, Some(64));
    loop {
        match exit {
            Exit::BudgetExhausted | Exit::Interrupted => {
                exit = cpu.run(vm, Some(64));
                continue;
            }
            Exit::MmioRead { .. } | Exit::MmioWrite { .. } => {}
            other => return (bus, Some(other)),
        }
        if bus.len() >= 12 {
            return (bus, None);
        }
        let how = schedule[stops % schedule.len()];
        stops += 1;
        match how {
            Between::HostFirst | Between::HostRepeat => {
                cpu.exit_request().request_exit();
                let stop = cpu.run(vm, Some(64));
                if !matches!(stop, Exit::Interrupted) {
                    // Not taken at the boundary: a stale completion was pending.
                    return (bus, Some(stop));
                }
                if matches!(how, Between::HostRepeat) {
                    exit = cpu.run(vm, Some(64));
                    if !matches!(exit, Exit::MmioRead { .. } | Exit::MmioWrite { .. }) {
                        return (bus, Some(exit));
                    }
                }
            }
            Between::ZeroFirst => {
                let stop = cpu.run(vm, Some(0));
                assert!(matches!(stop, Exit::BudgetExhausted), "{stop:?}");
            }
            Between::Run => {}
        }
        match exit {
            Exit::MmioRead { addr, size, .. } => {
                bus.push(Bus::Read(addr, size));
                cpu.complete_mmio_read(device_answer(addr, size));
            }
            Exit::MmioWrite {
                addr, size, value, ..
            } => {
                bus.push(Bus::Write(addr, size, value));
                cpu.complete_mmio_write();
            }
            _ => unreachable!(),
        }
        exit = cpu.run(vm, Some(64));
    }
}

/// A string element between two device operands is two transfers: MOVS reads
/// the source and writes the destination (SDM Vol 2B §4.3, MOVS/MOVSB/MOVSW/
/// MOVSD/MOVSQ, Operation, p. 4-105), CMPS reads both (SDM Vol 2A §3.3,
/// CMPS/CMPSB/CMPSW/CMPSD/CMPSQ, Operation, p. 3-182). A budget stop or a host
/// stop between them is taken at the boundary before the retry: the element has
/// neither completed nor aborted, so the first transfer keeps its answer and
/// the completion given after the stop answers the second (spec.md §5.2). Each
/// transfer is performed once, and the instruction ends as it does on RAM
/// holding the device's values. Long64 and Compat32, interpreter and compiled
/// tiers. Measured at 5b04d7a2: a host stop before the second completion
/// dropped the first answer, so CMPSB compared the destination's answer with
/// itself and MOVSB read its source twice.
#[test]
fn boundary_stops_between_string_transfers() {
    use Between::*;
    const SRC: u64 = DEVICE + 0x10;
    const DST: u64 = DEVICE + 0x120;
    let schedules: &[&[Between]] = &[
        &[Run],
        &[HostFirst],
        &[Run, HostFirst],
        &[HostRepeat],
        &[Run, HostRepeat],
        &[ZeroFirst],
    ];
    // (name, bytes, count, expected transfers)
    let cases: &[(&str, &[u8], u64, &[Bus])] = &[
        (
            "movsb",
            &[0xa4],
            0,
            &[Bus::Read(SRC, 1), Bus::Write(DST, 1, 0x11)],
        ),
        (
            "movsd",
            &[0xa5],
            0,
            &[Bus::Read(SRC, 4), Bus::Write(DST, 4, 0x1413_1211)],
        ),
        (
            "rep movsb",
            &[0xf3, 0xa4],
            2,
            &[
                Bus::Read(SRC, 1),
                Bus::Write(DST, 1, 0x11),
                Bus::Read(SRC + 1, 1),
                Bus::Write(DST + 1, 1, 0x12),
            ],
        ),
        ("cmpsb", &[0xa6], 0, &[Bus::Read(SRC, 1), Bus::Read(DST, 1)]),
        (
            "repe cmpsd",
            &[0xf3, 0xa7],
            2,
            &[Bus::Read(SRC, 4), Bus::Read(DST, 4)],
        ),
    ];
    let setup = |mode: CpuMode, jit: bool, trap: bool, code: &[u8], count: u64| {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(jit));
        vm.set_cpu_mode(mode);
        vm.map(0, DEVICE as usize, Prot::RWX, RegionKind::Ram)
            .unwrap();
        let kind = if trap {
            RegionKind::Trap
        } else {
            RegionKind::Ram
        };
        vm.map(DEVICE, 0x1000, Prot::RW, kind).unwrap();
        if !trap {
            for at in DEVICE..DEVICE + 0x1000 {
                vm.mem.write(at, device_answer(at, 1), 1).unwrap();
            }
        }
        // inc ebx ; <case> ; inc ebp ; hlt
        let mut bytes = vec![0xff, 0xc3];
        bytes.extend_from_slice(code);
        bytes.extend_from_slice(&[0xff, 0xc5, 0xf4]);
        vm.write_bytes(CODE, &bytes).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rsi, SRC);
        cpu.set_reg(Reg::Rdi, DST);
        cpu.set_reg(Reg::Rcx, count);
        (vm, cpu)
    };
    let observe = |cpu: &x86jit_core::Vcpu| {
        (
            [Reg::Rbx, Reg::Rcx, Reg::Rsi, Reg::Rdi, Reg::Rbp, Reg::Rip].map(|r| cpu.reg(r)),
            cpu.flags(),
        )
    };
    let mut bad = Vec::new();
    let mut checked = 0;
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for &(name, code, count, expected) in cases {
            let (vm, mut cpu) = setup(mode, false, false, code, count);
            assert!(
                matches!(cpu.run(&vm, Some(64)), Exit::Hlt),
                "{mode:?} {name}"
            );
            let reference = observe(&cpu);
            for jit in [false, true] {
                for schedule in schedules {
                    let (vm, mut cpu) = setup(mode, jit, true, code, count);
                    let (bus, stop) = drive_between(&vm, &mut cpu, schedule);
                    checked += 1;
                    let observed = observe(&cpu);
                    if bus != expected || !matches!(stop, Some(Exit::Hlt)) || observed != reference
                    {
                        bad.push(format!(
                            "{mode:?} {name} jit {jit} {schedule:?}: {bus:x?} then {stop:?}, \
                             expected {expected:x?}; state {observed:x?}, RAM run {reference:x?}"
                        ));
                    }
                }
            }
        }
    }
    assert_eq!(checked, 2 * 5 * 2 * 6);
    assert!(
        bad.is_empty(),
        "{} failures:\n{}",
        bad.len(),
        bad.join("\n")
    );
}
