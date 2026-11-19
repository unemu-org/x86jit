//! Bounded MMIO retries, including two executions at the same guest RIP.

use iced_x86::code_asm::*;
use x86jit_core::{Backend, Exit, InterpreterBackend, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;
const MMIO: u64 = 0x4000;

// SDM Vol 2B, MOVUPS, Description and Operation (VEX.128): unaligned 128-bit
// moves. The exit contract (§5.2) exposes whole operands in 8-byte slots.
fn page_crossing_movups(backend: Box<dyn Backend>, unaligned: bool, store: bool) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(MMIO, 0x2000, Prot::RW, RegionKind::Trap).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    if store {
        a.vmovups(xmmword_ptr(rax), xmm0).unwrap();
    } else {
        a.vmovups(xmm0, xmmword_ptr(rax)).unwrap();
    }
    a.hlt().unwrap();
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
    let at = MMIO + 0x1000 - if unaligned { 4 } else { 8 };
    let value = 0x1122334455667788_99aabbccddeeff00u128;
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, at);
    cpu.set_xmm(0, value);
    cpu.set_ymm_hi(0, u128::MAX);
    // Repeat at the same RIP to check that completed answers are discarded.
    for _ in 0..2 {
        cpu.set_reg(Reg::Rip, CODE);
        let initial = cpu.xmm(0);
        for part in 0..2 {
            let exit = cpu.run(&vm, None);
            let expected = (at + part * 8, 8, 16, part as u8);
            match exit {
                Exit::MmioRead {
                    addr,
                    size,
                    total_size,
                    part,
                    ..
                } if !store => {
                    assert_eq!((addr, size, total_size, part), expected);
                    assert_eq!(cpu.xmm(0), initial, "no partial load destination");
                    cpu.complete_mmio_read((value >> (part * 64)) as u64);
                }
                Exit::MmioWrite {
                    addr,
                    size,
                    total_size,
                    part,
                    value: written,
                    ..
                } if store => {
                    assert_eq!((addr, size, total_size, part), expected);
                    assert_eq!(written, (value >> (part * 64)) as u64);
                    cpu.complete_mmio_write();
                }
                other => panic!("unexpected exit: {other:?}"),
            }
            assert_eq!(cpu.reg(Reg::Rip), CODE);
        }
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.xmm(0), value);
        if !store {
            assert_eq!(cpu.ymm_hi(0), 0);
        }
    }
}

macro_rules! page_crossing_movups_tests {
    ($($name:ident: ($backend:expr, $unaligned:expr, $store:expr)),* $(,)?) => {$(
        #[test]
        fn $name() {
            page_crossing_movups(Box::new($backend), $unaligned, $store);
        }
    )*};
}

page_crossing_movups_tests! {
    page_crossing_movups_aligned_load_interp: (InterpreterBackend, false, false),
    page_crossing_movups_aligned_load_jit: (JitBackend::new(), false, false),
    page_crossing_movups_unaligned_load_interp: (InterpreterBackend, true, false),
    page_crossing_movups_unaligned_load_jit: (JitBackend::new(), true, false),
    page_crossing_movups_aligned_store_interp: (InterpreterBackend, false, true),
    page_crossing_movups_aligned_store_jit: (JitBackend::new(), false, true),
    page_crossing_movups_unaligned_store_interp: (InterpreterBackend, true, true),
    page_crossing_movups_unaligned_store_jit: (JitBackend::new(), true, true),
}

#[derive(Clone, Copy, Debug)]
enum Case {
    MaskedLoad,
    MaskedStore,
    NarrowStore,
    Fma,
    BroadcastLane,
    Permute2,
    Permute1,
    Gf2p8,
    BroadcastScalar,
    ByteLoad(u64),
    ByteStore(u64),
}

fn lanes(bytes: &[u8; 64]) -> [u128; 4] {
    std::array::from_fn(|i| u128::from_le_bytes(bytes[i * 16..i * 16 + 16].try_into().unwrap()))
}

fn vector(cpu: &x86jit_core::Vcpu) -> [u128; 4] {
    [
        cpu.xmm(0),
        cpu.ymm_hi(0),
        cpu.zmm_hi(0, 0),
        cpu.zmm_hi(0, 1),
    ]
}

fn run(case: Case, backend: Box<dyn Backend>) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    a.mov(ecx, 2).unwrap();
    a.set_label(&mut top).unwrap();
    match case {
        Case::MaskedLoad => a.vmovdqu32(xmm0.k1(), xmmword_ptr(rax)),
        Case::MaskedStore => a.vmovdqu32(xmmword_ptr(rax).k1(), xmm0),
        Case::NarrowStore => a.vpmovqd(qword_ptr(rax), xmm0),
        Case::Fma => a.vfmadd213ps(xmm0, xmm1, xmmword_ptr(rax)),
        Case::BroadcastLane => a.vbroadcasti32x4(ymm0, xmmword_ptr(rax)),
        Case::Permute2 => a.vpermt2d(xmm0, xmm1, xmmword_ptr(rax)),
        Case::Permute1 => a.evex().vpermd(ymm0.k1(), ymm1, ymmword_ptr(rax)),
        Case::Gf2p8 => a.vgf2p8mulb(ymm0, ymm1, ymmword_ptr(rax)),
        Case::BroadcastScalar => a.vbroadcastss(xmm0, dword_ptr(rax)),
        Case::ByteLoad(_) => a.vmovdqu8(zmm0.k1(), zmmword_ptr(rax)),
        Case::ByteStore(_) => a.vmovdqu8(zmmword_ptr(rax).k1(), zmm0),
    }
    .unwrap();
    a.dec(ecx).unwrap();
    a.jnz(top).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();

    let mut data: [u8; 64] = std::array::from_fn(|i| i as u8 + 1);
    let mut initial = [0xcc; 64];
    let mut source = [0; 64];
    let mut expected = [0; 64];
    let mut transfers: Vec<(u64, u8)> = Vec::new();
    let mut write_values = Vec::new();
    let mut writemask = 0b0101;
    let mut intermediate = None;
    match case {
        // SDM Vol 2B, MOVDQU/VMOVDQU8/16/32/64, Operation (EVEX load/store).
        Case::MaskedLoad | Case::MaskedStore => {
            expected[..16].copy_from_slice(&initial[..16]);
            for i in [0, 2] {
                transfers.push((MMIO + i * 4, 4));
                expected[i as usize * 4..i as usize * 4 + 4]
                    .copy_from_slice(&data[i as usize * 4..i as usize * 4 + 4]);
                write_values.push(0xcccccccc);
            }
        }
        // SDM Vol 2C, VPMOVQD/VPMOVSQD/VPMOVUSQD, Operation (VPMOVQD).
        Case::NarrowStore => {
            initial[..16]
                .copy_from_slice(&0x99aa_bbcc_ddee_ff00_1122_3344_5566_7788u128.to_le_bytes());
            transfers = vec![(MMIO, 4), (MMIO + 4, 4)];
            write_values = vec![0x55667788, 0xddeeff00];
        }
        // SDM Vol 2C, VFMADD132PS/VFMADD213PS/VFMADD231PS, Operation (VFMADD213PS).
        Case::Fma => {
            for i in 0..4 {
                initial[i * 4..i * 4 + 4].copy_from_slice(&1f32.to_le_bytes());
                source[i * 4..i * 4 + 4].copy_from_slice(&2f32.to_le_bytes());
                data[i * 4..i * 4 + 4].copy_from_slice(&2f32.to_le_bytes());
                expected[i * 4..i * 4 + 4].copy_from_slice(&10f32.to_le_bytes());
            }
            let mut first = [0; 64];
            for i in 0..4 {
                first[i * 4..i * 4 + 4].copy_from_slice(&4f32.to_le_bytes());
            }
            intermediate = Some(lanes(&first));
            transfers = vec![(MMIO, 8), (MMIO + 8, 8)];
        }
        // SDM Vol 2C, VPBROADCAST, Operation (VBROADCASTI32X4).
        Case::BroadcastLane => {
            expected[..16].copy_from_slice(&data[..16]);
            expected[16..32].copy_from_slice(&data[..16]);
            transfers = vec![(MMIO, 8), (MMIO + 8, 8)];
        }
        // SDM Vol 2C, VPERMT2W/D/Q/PS/PD, Operation (VPERMT2D).
        Case::Permute2 => {
            for i in 0..4 {
                source[i * 4..i * 4 + 4].copy_from_slice(&(4 + i as u32).to_le_bytes());
            }
            expected[..16].copy_from_slice(&data[..16]);
            transfers = vec![(MMIO, 8), (MMIO + 8, 8)];
        }
        // SDM Vol 2C, VPERMD/VPERMW, Operation (EVEX VPERMD).
        Case::Permute1 => {
            writemask = 0xff;
            for i in 0..8 {
                source[i * 4..i * 4 + 4].copy_from_slice(&(7 - i as u32).to_le_bytes());
                expected[i * 4..i * 4 + 4].copy_from_slice(&data[(7 - i) * 4..(8 - i) * 4]);
            }
            transfers = (0..4).map(|i| (MMIO + i * 8, 8)).collect();
        }
        // SDM Vol 2A, GF2P8MULB, Operation: multiplication by polynomial 1 is identity.
        Case::Gf2p8 => {
            data.fill(1);
            source.fill(0xa5);
            expected[..32].fill(0xa5);
            transfers = (0..4).map(|i| (MMIO + i * 8, 8)).collect();
        }
        // SDM Vol 2C, VBROADCAST, Operation (VBROADCASTSS).
        Case::BroadcastScalar => {
            initial[32..].fill(0);
            for i in 0..4 {
                expected[i * 4..i * 4 + 4].copy_from_slice(&data[..4]);
            }
            transfers.push((MMIO, 4));
        }
        // SDM Vol 2B, MOVDQU/VMOVDQU8/16/32/64, Operation (EVEX VMOVDQU8).
        Case::ByteLoad(k) | Case::ByteStore(k) => {
            writemask = k;
            expected = initial;
            for i in 0..64 {
                if k & (1 << i) != 0 {
                    transfers.push((MMIO + i as u64, 1));
                    expected[i] = data[i];
                    write_values.push(0xcc);
                }
            }
        }
    }
    let store = matches!(
        case,
        Case::MaskedStore | Case::NarrowStore | Case::ByteStore(_)
    );
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, MMIO);
    cpu.set_kmask(1, writemask);
    for (i, value) in lanes(&initial).into_iter().enumerate() {
        match i {
            0 => cpu.set_xmm(0, value),
            1 => cpu.set_ymm_hi(0, value),
            _ => cpu.set_zmm_hi(0, i - 2, value),
        }
    }
    let source = lanes(&source);
    cpu.set_xmm(1, source[0]);
    cpu.set_ymm_hi(1, source[1]);
    let total_width = match case {
        Case::MaskedLoad | Case::MaskedStore | Case::Fma | Case::BroadcastLane | Case::Permute2 => {
            16
        }
        Case::NarrowStore => 8,
        Case::Permute1 | Case::Gf2p8 => 32,
        Case::BroadcastScalar => 4,
        Case::ByteLoad(_) | Case::ByteStore(_) => 64,
    };
    let mut seen = Vec::new();
    for _ in 0..(transfers.len() * 2 + 4) {
        match cpu.run(&vm, None) {
            Exit::MmioRead {
                addr,
                size,
                total_size,
                part,
                ..
            } if !store => {
                assert_eq!(total_size, total_width, "{case:?}: total width");
                assert_eq!(
                    u64::from(part),
                    (addr - MMIO) / u64::from(size),
                    "{case:?}: part index"
                );
                let previous = if cpu.reg(Reg::Rcx) == 2 {
                    lanes(&initial)
                } else {
                    intermediate.unwrap_or_else(|| lanes(&expected))
                };
                assert_eq!(
                    vector(&cpu),
                    previous,
                    "{case:?}: destination changed during a read"
                );
                seen.push((addr, size));
                let off = (addr - MMIO) as usize;
                let mut answer = [0; 8];
                answer[..size as usize].copy_from_slice(&data[off..off + size as usize]);
                cpu.complete_mmio_read(u64::from_le_bytes(answer));
            }
            Exit::MmioWrite {
                addr,
                size,
                value,
                total_size,
                part,
                ..
            } if store => {
                assert_eq!(total_size, total_width, "{case:?}: total width");
                assert_eq!(
                    u64::from(part),
                    (addr - MMIO) / u64::from(size),
                    "{case:?}: part index"
                );
                let part = transfers
                    .iter()
                    .position(|t| *t == (addr, size))
                    .expect("unexpected write transfer");
                let value_mask = u64::MAX >> (64 - u32::from(size) * 8);
                assert_eq!(
                    value & value_mask,
                    write_values[part],
                    "{case:?}: transfer value"
                );
                seen.push((addr, size));
                cpu.complete_mmio_write();
            }
            Exit::Hlt => {
                assert_eq!(
                    seen,
                    transfers.repeat(2),
                    "{case:?}: each execution must ask again"
                );
                assert_eq!(
                    vector(&cpu),
                    if store {
                        lanes(&initial)
                    } else {
                        lanes(&expected)
                    }
                );
                return;
            }
            other => panic!("{case:?}: unexpected exit {other:?}"),
        }
    }
    panic!("{case:?}: did not converge within the retry bound: {seen:?}");
}

macro_rules! cases {
    ($($name:ident: $case:expr),* $(,)?) => {$(
        mod $name {
            use super::*;
            #[test]
            fn interp() { run($case, Box::new(InterpreterBackend)); }
            #[test]
            fn jit() { run($case, Box::new(JitBackend::new())); }
        }
    )*};
}

cases! {
    masked_load: Case::MaskedLoad,
    masked_store: Case::MaskedStore,
    narrow_store: Case::NarrowStore,
    fma: Case::Fma,
    broadcast_lane: Case::BroadcastLane,
    permute2: Case::Permute2,
    permute1: Case::Permute1,
    gf2p8: Case::Gf2p8,
    broadcast_scalar: Case::BroadcastScalar,
    byte_load_full: Case::ByteLoad(u64::MAX),
    byte_store_full: Case::ByteStore(u64::MAX),
    byte_load_sparse: Case::ByteLoad(0x8000_0000_0000_0105),
    byte_store_sparse: Case::ByteStore(0x8000_0000_0000_0105),
}

// SDM Vol 2B, MOVDQU/VMOVDQU8/16/32/64, Operation (EVEX VMOVDQU8 load).
fn crossing_masked_load(backend: Box<dyn Backend>, active: u64) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.write_bytes(MMIO - 32, &[0x11; 32]).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    a.vmovdqu8(zmm0.k1().z(), zmmword_ptr(rax)).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, MMIO - 32);
    cpu.set_kmask(1, active);
    cpu.set_xmm(0, u128::MAX);
    cpu.set_ymm_hi(0, u128::MAX);
    cpu.set_zmm_hi(0, 0, u128::MAX);
    cpu.set_zmm_hi(0, 1, u128::MAX);
    let mut seen = Vec::new();
    for _ in 0..34 {
        match cpu.run(&vm, None) {
            Exit::MmioRead { addr, size, .. } => {
                assert_eq!(vector(&cpu), [u128::MAX; 4], "no partial destination");
                assert_eq!(size, 1);
                seen.push(addr);
                cpu.complete_mmio_read(0x22);
            }
            Exit::Hlt => {
                let expected: Vec<_> = (0..32)
                    .filter(|i| active & (1 << (i + 32)) != 0)
                    .map(|i| MMIO + i)
                    .collect();
                assert_eq!(seen, expected);
                let mut bytes = [0; 64];
                for (i, byte) in bytes.iter_mut().enumerate() {
                    if active & (1 << i) != 0 {
                        *byte = if i < 32 { 0x11 } else { 0x22 };
                    }
                }
                assert_eq!(vector(&cpu), lanes(&bytes));
                return;
            }
            other => panic!("unexpected exit {other:?}"),
        }
    }
    panic!("crossing access did not converge: {seen:?}");
}

#[test]
fn crossing_masked_load_interp() {
    crossing_masked_load(Box::new(InterpreterBackend), u64::MAX);
}

#[test]
fn crossing_masked_load_jit() {
    crossing_masked_load(Box::new(JitBackend::new()), u64::MAX);
}

#[test]
fn masked_off_trap_half_interp() {
    crossing_masked_load(Box::new(InterpreterBackend), 0xffff_ffff);
}

#[test]
fn masked_off_trap_half_jit() {
    crossing_masked_load(Box::new(JitBackend::new()), 0xffff_ffff);
}

#[derive(Clone, Copy, Debug)]
enum MaskedArithmetic {
    Fma,
    Broadcast,
    GfMultiply,
}

// SDM Vol 1 §15.6.1 and Vol 2A §2.8: masked-off memory elements suppress faults.
// SDM Vol 2C, VFMADD132PS/VFMADD213PS/VFMADD231PS, Operation (EVEX VFMADD213PS);
// VPBROADCAST, Operation (VBROADCASTI32X4): source element = destination element mod 4.
// SDM Vol 2A, GF2P8MULB, Description and Operation: byte-granular fault suppression.
fn masked_arithmetic(backend: Box<dyn Backend>, case: MaskedArithmetic, sparse: bool, trap: bool) {
    let (elem, source_index, output_index, width) = match case {
        MaskedArithmetic::Fma => (4, 1, 1, 16),
        MaskedArithmetic::Broadcast => (4, 1, 5, 32),
        MaskedArithmetic::GfMultiply => (1, 9, 9, 16),
    };
    // In the RAM case only the required source element ends before the unmapped page.
    let base = if trap {
        MMIO
    } else {
        MMIO - (source_index + 1) * elem
    };
    let mut vm = Vm::with_backend(VmConfig::flat(if trap { 0x10000 } else { MMIO }), backend);
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    if trap {
        vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    }
    let mut a = CodeAssembler::new(64).unwrap();
    match case {
        MaskedArithmetic::Fma => a.evex().vfmadd213ps(xmm0.k1(), xmm1, xmmword_ptr(rax)),
        MaskedArithmetic::Broadcast => a.vbroadcasti32x4(ymm0.k1(), xmmword_ptr(rax)),
        MaskedArithmetic::GfMultiply => a.evex().vgf2p8mulb(xmm0.k1(), xmm1, xmmword_ptr(rax)),
    }
    .unwrap();
    a.hlt().unwrap();
    let code = a.assemble(CODE).unwrap();
    let insn = iced_x86::Decoder::with_ip(64, &code, CODE, iced_x86::DecoderOptions::NONE).decode();
    assert_eq!(insn.op_mask(), iced_x86::Register::K1);
    vm.write_bytes(CODE, &code).unwrap();
    let answer: u64 = match case {
        MaskedArithmetic::Fma => 3f32.to_bits() as u64,
        MaskedArithmetic::Broadcast => 0x11223344,
        MaskedArithmetic::GfMultiply => 1,
    };
    if !trap {
        vm.write_bytes(MMIO - elem, &answer.to_le_bytes()[..elem as usize])
            .unwrap();
    }
    let mut initial = [0xcc; 64];
    if matches!(case, MaskedArithmetic::Fma) {
        initial[..16].copy_from_slice(&[1f32.to_le_bytes(); 4].concat());
    }
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, base);
    cpu.set_xmm(0, lanes(&initial)[0]);
    cpu.set_ymm_hi(0, lanes(&initial)[1]);
    cpu.set_zmm_hi(0, 0, lanes(&initial)[2]);
    cpu.set_zmm_hi(0, 1, lanes(&initial)[3]);
    cpu.set_xmm(
        1,
        if matches!(case, MaskedArithmetic::Fma) {
            u128::from_le_bytes([2f32.to_le_bytes(); 4].concat().try_into().unwrap())
        } else {
            u128::from_le_bytes([0xa5; 16])
        },
    );
    cpu.set_kmask(1, if sparse { 1 << output_index } else { 0 });
    if trap && sparse {
        let exit = cpu.run(&vm, None);
        assert!(
            matches!(exit, Exit::MmioRead { addr, size, .. }
            if addr == base + source_index * elem && u64::from(size) == elem),
            "{case:?}: expected only the required element, got {exit:?}"
        );
        assert_eq!(
            vector(&cpu),
            lanes(&initial),
            "destination must remain atomic"
        );
        cpu.complete_mmio_read(answer);
    }
    assert!(
        matches!(cpu.run(&vm, None), Exit::Hlt),
        "{case:?}: inactive memory was accessed"
    );
    let mut expected = initial;
    expected[width..].fill(0);
    if sparse {
        let result = match case {
            MaskedArithmetic::Fma => 5f32.to_bits() as u64,
            MaskedArithmetic::Broadcast => answer,
            MaskedArithmetic::GfMultiply => 0xa5,
        };
        let off = output_index as usize * elem as usize;
        expected[off..off + elem as usize].copy_from_slice(&result.to_le_bytes()[..elem as usize]);
    }
    assert_eq!(vector(&cpu), lanes(&expected));
}

macro_rules! masked_arithmetic_cases {
    ($($name:ident: $case:ident, $sparse:expr, $trap:expr;)*) => {$(
        mod $name {
            use super::*;
            #[test]
            fn interp() { masked_arithmetic(Box::new(InterpreterBackend), MaskedArithmetic::$case, $sparse, $trap); }
            #[test]
            fn jit() { masked_arithmetic(Box::new(JitBackend::new()), MaskedArithmetic::$case, $sparse, $trap); }
        }
    )*};
}

masked_arithmetic_cases! {
    fma_empty_trap: Fma, false, true;
    fma_sparse_trap: Fma, true, true;
    fma_empty_unmapped: Fma, false, false;
    fma_sparse_unmapped: Fma, true, false;
    broadcast_empty_trap: Broadcast, false, true;
    broadcast_sparse_trap: Broadcast, true, true;
    broadcast_empty_unmapped: Broadcast, false, false;
    broadcast_sparse_unmapped: Broadcast, true, false;
    gf_multiply_empty_trap: GfMultiply, false, true;
    gf_multiply_sparse_trap: GfMultiply, true, true;
    gf_multiply_empty_unmapped: GfMultiply, false, false;
    gf_multiply_sparse_unmapped: GfMultiply, true, false;
}

fn retry_after_ram_fault(backend: Box<dyn Backend>, redirect: bool) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    let base = MMIO + 0xff8;
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    a.set_label(&mut top).unwrap();
    // SDM Vol 2B, MOVDQU/VMOVDQU8/16/32/64, Operation (EVEX VMOVDQU64).
    a.vmovdqu64(xmm0.k1(), xmmword_ptr(rax)).unwrap();
    a.dec(ecx).unwrap();
    a.jnz(top).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, base);
    cpu.set_reg(Reg::Rcx, 2);
    cpu.set_kmask(1, 3);
    assert!(matches!(cpu.run(&vm, None), Exit::MmioRead { addr, size: 8, .. } if addr == base));
    cpu.complete_mmio_read(0x1111);
    assert!(
        matches!(cpu.run(&vm, None), Exit::UnmappedMemory { addr, .. } if addr == MMIO + 0x1000)
    );
    assert_eq!(cpu.reg(Reg::Rip), CODE);
    if redirect {
        cpu.set_reg(Reg::Rip, CODE + 0x100);
        cpu.set_reg(Reg::Rip, CODE);
    } else {
        vm.map(MMIO + 0x1000, 0x1000, Prot::RW, RegionKind::Ram)
            .unwrap();
    }
    let exit = cpu.run(&vm, None);
    assert!(
        matches!(exit, Exit::MmioRead { addr, size: 8, .. } if addr == base),
        "a new execution must request a fresh answer: {exit:?}"
    );
    if !redirect {
        cpu.complete_mmio_read(0x2222);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.xmm(0), 0x2222);
    }
}

#[test]
fn retry_after_ram_fault_interp() {
    retry_after_ram_fault(Box::new(InterpreterBackend), false);
}
#[test]
fn retry_after_ram_fault_jit() {
    retry_after_ram_fault(Box::new(JitBackend::new()), false);
}
#[test]
fn rip_redirection_abandons_retry_interp() {
    retry_after_ram_fault(Box::new(InterpreterBackend), true);
}
#[test]
fn rip_redirection_abandons_retry_jit() {
    retry_after_ram_fault(Box::new(JitBackend::new()), true);
}

// SDM Vol 2B, MOVDQU/VMOVDQU8/16/32/64, Operation (EVEX VMOVDQU8): only bit 32 is active.
fn wrapped_masked_access(backend: Box<dyn Backend>, store: bool) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(CODE, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    if store {
        a.vmovdqu8(zmmword_ptr(rax).k1(), zmm0).unwrap();
    } else {
        a.vmovdqu8(zmm0.k1().z(), zmmword_ptr(rax)).unwrap();
    }
    a.hlt().unwrap();
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, u64::MAX - 31);
    cpu.set_kmask(1, 1 << 32);
    cpu.set_zmm_hi(0, 0, 0x5a);
    let exit = cpu.run(&vm, None);
    if store {
        assert!(
            matches!(
                exit,
                Exit::MmioWrite {
                    addr: 0,
                    size: 1,
                    value: 0x5a,
                    total_size: 64,
                    part: 32,
                    ..
                }
            ),
            "{exit:?}"
        );
        cpu.complete_mmio_write();
    } else {
        assert!(
            matches!(
                exit,
                Exit::MmioRead {
                    addr: 0,
                    size: 1,
                    total_size: 64,
                    part: 32,
                    ..
                }
            ),
            "{exit:?}"
        );
        cpu.complete_mmio_read(0x5a);
    }
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(vector(&cpu), [0, 0, 0x5a, 0]);
}

#[test]
fn wrapped_masked_load_interp() {
    wrapped_masked_access(Box::new(InterpreterBackend), false);
}
#[test]
fn wrapped_masked_load_jit() {
    wrapped_masked_access(Box::new(JitBackend::new()), false);
}
#[test]
fn wrapped_masked_store_interp() {
    wrapped_masked_access(Box::new(InterpreterBackend), true);
}
#[test]
fn wrapped_masked_store_jit() {
    wrapped_masked_access(Box::new(JitBackend::new()), true);
}

fn rip_redirection_discards_completion(backend: Box<dyn Backend>, store: bool) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    if store {
        a.vmovdqu64(xmmword_ptr(rax).k1(), xmm0).unwrap();
    } else {
        a.vmovdqu64(xmm0.k1(), xmmword_ptr(rax)).unwrap();
    }
    a.hlt().unwrap();
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, MMIO);
    cpu.set_kmask(1, 1);
    for attempt in 0..2 {
        let exit = cpu.run(&vm, None);
        if store {
            assert!(
                matches!(
                    exit,
                    Exit::MmioWrite {
                        addr: MMIO,
                        size: 8,
                        ..
                    }
                ),
                "{exit:?}"
            );
            cpu.complete_mmio_write();
        } else {
            assert!(
                matches!(
                    exit,
                    Exit::MmioRead {
                        addr: MMIO,
                        size: 8,
                        ..
                    }
                ),
                "{exit:?}"
            );
            cpu.complete_mmio_read(0x1234);
        }
        if attempt == 0 {
            cpu.set_reg(Reg::Rip, CODE + 0x100);
            cpu.set_reg(Reg::Rip, CODE);
        }
    }
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
}

#[test]
fn rip_redirection_discards_read_completion_interp() {
    rip_redirection_discards_completion(Box::new(InterpreterBackend), false);
}
#[test]
fn rip_redirection_discards_read_completion_jit() {
    rip_redirection_discards_completion(Box::new(JitBackend::new()), false);
}
#[test]
fn rip_redirection_discards_write_completion_interp() {
    rip_redirection_discards_completion(Box::new(InterpreterBackend), true);
}
#[test]
fn rip_redirection_discards_write_completion_jit() {
    rip_redirection_discards_completion(Box::new(JitBackend::new()), true);
}

// SDM Vol 2A, GF2P8AFFINEQB and GF2P8AFFINEINVQB, Description:
// EVEX affine operations do not support memory fault suppression, even with k1 = 0.
fn affine_mask_keeps_memory_access(backend: Box<dyn Backend>) {
    let mut vm = Vm::with_backend(VmConfig::flat(MMIO + 0x1000), backend);
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    for inverse in [false, true] {
        let mut a = CodeAssembler::new(64).unwrap();
        if inverse {
            a.evex()
                .vgf2p8affineinvqb(xmm0.k1(), xmm1, xmmword_ptr(rax), 0)
                .unwrap();
        } else {
            a.evex()
                .vgf2p8affineqb(xmm0.k1(), xmm1, xmmword_ptr(rax), 0)
                .unwrap();
        }
        a.hlt().unwrap();
        vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rax, MMIO);
        cpu.set_kmask(1, 0);
        cpu.set_xmm(0, u128::MAX);
        for offset in [0, 8] {
            assert!(
                matches!(cpu.run(&vm, None), Exit::MmioRead { addr, size: 8, .. } if addr == MMIO + offset)
            );
            cpu.complete_mmio_read(0);
        }
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.xmm(0), u128::MAX);
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rax, MMIO + 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::UnmappedMemory { .. }));
    }
}

#[test]
fn affine_mask_keeps_memory_access_interp() {
    affine_mask_keeps_memory_access(Box::new(InterpreterBackend));
}
#[test]
fn affine_mask_keeps_memory_access_jit() {
    affine_mask_keeps_memory_access(Box::new(JitBackend::new()));
}

// These tests exercise the embedder's retry contract, rather than new ISA semantics.
fn bulk_redirection_discards_pending(backend: Box<dyn Backend>, store: bool, clone: bool) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    if store {
        a.vmovdqu64(xmmword_ptr(rax).k1(), xmm0).unwrap();
    } else {
        a.vmovdqu64(xmm0.k1(), xmmword_ptr(rax)).unwrap();
    }
    a.hlt().unwrap();
    let code = a.assemble(CODE).unwrap();
    vm.write_bytes(CODE, &code).unwrap();
    vm.write_bytes(CODE + 0x100, &code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, MMIO);
    cpu.set_kmask(1, 1);
    let first = cpu.run(&vm, None);
    if store {
        assert!(
            matches!(
                first,
                Exit::MmioWrite {
                    addr: MMIO,
                    size: 8,
                    ..
                }
            ),
            "{first:?}"
        );
        cpu.complete_mmio_write();
    } else {
        assert!(
            matches!(
                first,
                Exit::MmioRead {
                    addr: MMIO,
                    size: 8,
                    ..
                }
            ),
            "{first:?}"
        );
        cpu.complete_mmio_read(0x1234);
    }
    assert_eq!(cpu.cpu.mmio_parts_len, 0, "pending-only completion");
    if clone {
        let mut state = cpu.cpu.clone();
        state.rip = CODE + 0x100;
        cpu.cpu = state;
    } else {
        cpu.cpu.rip = CODE + 0x100;
    }
    let redirected = cpu.run(&vm, None);
    assert_eq!(
        format!("{redirected:?}"),
        format!("{first:?}"),
        "a different instruction must ask again"
    );
}

fn unsupported_step_abandons_retry(backend: Box<dyn Backend>, direct: bool) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    a.vmovdqu64(xmm0.k1(), xmmword_ptr(rax)).unwrap();
    a.hlt().unwrap();
    let code = a.assemble(CODE).unwrap();
    vm.write_bytes(CODE, &code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, MMIO);
    cpu.set_kmask(1, 3);
    assert!(matches!(
        cpu.run(&vm, None),
        Exit::MmioRead {
            addr: MMIO,
            size: 8,
            ..
        }
    ));
    cpu.complete_mmio_read(0x1111);
    assert!(matches!(cpu.run(&vm, None), Exit::MmioRead { addr, size: 8, .. } if addr == MMIO + 8));
    cpu.complete_mmio_read(0x2222);
    assert_eq!(cpu.cpu.mmio_parts_len, 1);
    // VMCALL is deliberately unsupported by the guest-agnostic engine.
    vm.write_bytes(CODE, &[0x0f, 0x01, 0xc1]).unwrap();
    let result = if direct {
        x86jit_core::interp::step_one(
            &vm.mem,
            &mut cpu.cpu,
            x86jit_core::CpuMode::Long64,
            &mut Vec::new(),
            &mut x86jit_core::interp::RetireInfo::default(),
        )
    } else {
        cpu.step_instruction(&vm)
    };
    assert!(matches!(
        result,
        x86jit_core::StepResult::Exit(Exit::UnknownInstruction { .. })
    ));
    vm.write_bytes(CODE, &code).unwrap();
    let restored = cpu.run(&vm, None);
    assert!(
        matches!(
            restored,
            Exit::MmioRead {
                addr: MMIO,
                size: 8,
                ..
            }
        ),
        "restored instruction must not reuse an abandoned answer: {restored:?}"
    );
}

macro_rules! retry_lifecycle_tests {
    ($($name:ident, $backend:expr, $store:expr, $clone:expr;)*) => {$ (
        #[test]
        fn $name() { bulk_redirection_discards_pending(Box::new($backend), $store, $clone); }
    )*};
}
retry_lifecycle_tests! {
    bulk_redirect_read_interp, InterpreterBackend, false, false;
    bulk_redirect_read_jit, JitBackend::new(), false, false;
    bulk_redirect_write_interp, InterpreterBackend, true, false;
    bulk_redirect_write_jit, JitBackend::new(), true, false;
    clone_redirect_read_interp, InterpreterBackend, false, true;
    clone_redirect_read_jit, JitBackend::new(), false, true;
    clone_redirect_write_interp, InterpreterBackend, true, true;
    clone_redirect_write_jit, JitBackend::new(), true, true;
}

#[test]
fn unsupported_step_abandons_retry_interp() {
    unsupported_step_abandons_retry(Box::new(InterpreterBackend), false);
}
#[test]
fn unsupported_step_abandons_retry_jit() {
    unsupported_step_abandons_retry(Box::new(JitBackend::new()), false);
}
#[test]
fn unsupported_direct_step_abandons_retry() {
    unsupported_step_abandons_retry(Box::new(InterpreterBackend), true);
}

// SDM Vol 2B, MOVUPS, Operation (VEX/EVEX load/store), and
// MOVDQU/VMOVDQU8/16/32/64, Operation (EVEX VMOVDQU32 load/store):
// vector moves copy the operand bytes; the writemask selects dword elements.
// SDM Vol 2B, MOV, Operation: the scalar forms copy the addressed operand.
fn access_width(backend: Box<dyn Backend>, width: u8, store: bool, masked: bool, ram_prefix: bool) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    match (width, store, masked) {
        (1, false, _) => a.mov(dl, byte_ptr(rax)),
        (2, false, _) => a.mov(dx, word_ptr(rax)),
        (4, false, _) => a.mov(edx, dword_ptr(rax)),
        (8, false, _) => a.mov(rdx, qword_ptr(rax)),
        (1, true, _) => a.mov(byte_ptr(rax), dl),
        (2, true, _) => a.mov(word_ptr(rax), dx),
        (4, true, _) => a.mov(dword_ptr(rax), edx),
        (8, true, _) => a.mov(qword_ptr(rax), rdx),
        (16, false, false) => a.vmovups(xmm0, xmmword_ptr(rax)),
        (32, false, false) => a.vmovups(ymm0, ymmword_ptr(rax)),
        (64, false, false) => a.vmovups(zmm0, zmmword_ptr(rax)),
        (16, true, false) => a.vmovups(xmmword_ptr(rax), xmm0),
        (32, true, false) => a.vmovups(ymmword_ptr(rax), ymm0),
        (64, true, false) => a.vmovups(zmmword_ptr(rax), zmm0),
        (16, false, true) => a.vmovdqu32(xmm0.k1().z(), xmmword_ptr(rax)),
        (32, false, true) => a.vmovdqu32(ymm0.k1().z(), ymmword_ptr(rax)),
        (64, false, true) => a.vmovdqu32(zmm0.k1().z(), zmmword_ptr(rax)),
        (16, true, true) => a.vmovdqu32(xmmword_ptr(rax).k1(), xmm0),
        (32, true, true) => a.vmovdqu32(ymmword_ptr(rax).k1(), ymm0),
        (64, true, true) => a.vmovdqu32(zmmword_ptr(rax).k1(), zmm0),
        _ => unreachable!(),
    }
    .unwrap();
    a.hlt().unwrap();
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
    let data: [u8; 64] = std::array::from_fn(|i| (i * 3 + 1) as u8);
    let base = MMIO - if ram_prefix { 8 } else { 0 };
    if ram_prefix {
        vm.write_bytes(base, &data[..8]).unwrap();
    }
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, base);
    cpu.set_reg(
        Reg::Rdx,
        if store {
            u64::from_le_bytes(data[..8].try_into().unwrap())
        } else {
            0
        },
    );
    if store {
        let v = lanes(&data);
        cpu.set_xmm(0, v[0]);
        cpu.set_ymm_hi(0, v[1]);
        cpu.set_zmm_hi(0, 0, v[2]);
        cpu.set_zmm_hi(0, 1, v[3]);
    }
    let active = if masked {
        1 | (1 << 2) | (1 << (width / 4 - 1))
    } else {
        0
    };
    cpu.set_kmask(1, active);
    let size = if masked { 4 } else { width.min(8) };
    let mut assembled = [0; 64];
    if ram_prefix {
        assembled[..8].copy_from_slice(&data[..8]);
    }
    for part in 0..width / size {
        if masked && active & (1 << part) == 0 {
            continue;
        }
        let off = part as usize * size as usize;
        if ram_prefix && off < 8 {
            continue;
        }
        let exit = cpu.run(&vm, None);
        assert_eq!(cpu.reg(Reg::Rip), CODE, "incomplete access advanced RIP");
        let (addr, actual_size, total_size, actual_part, value) = match exit {
            Exit::MmioRead {
                addr,
                size,
                total_size,
                part,
                ..
            } if !store => (addr, size, total_size, part, 0),
            Exit::MmioWrite {
                addr,
                size,
                total_size,
                part,
                value,
                ..
            } if store => (addr, size, total_size, part, value),
            other => panic!("unexpected exit: {other:?}"),
        };
        assert_eq!(
            (addr, actual_size, total_size, actual_part),
            (base + off as u64, size, u16::from(width), part)
        );
        let mut answer = [0; 8];
        answer[..size as usize].copy_from_slice(&data[off..off + size as usize]);
        if store {
            assert_eq!(
                &value.to_le_bytes()[..size as usize],
                &answer[..size as usize]
            );
            assembled[off..off + size as usize]
                .copy_from_slice(&value.to_le_bytes()[..size as usize]);
            cpu.complete_mmio_write();
        } else {
            assembled[off..off + size as usize].copy_from_slice(&answer[..size as usize]);
            cpu.complete_mmio_read(u64::from_le_bytes(answer));
        }
    }
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let mut expected = [0; 64];
    for part in 0..width / size {
        if !masked || active & (1 << part) != 0 {
            let off = part as usize * size as usize;
            expected[off..off + size as usize].copy_from_slice(&data[off..off + size as usize]);
        }
    }
    assert_eq!(assembled, expected);
    if !store {
        if width <= 8 {
            assert_eq!(
                cpu.reg(Reg::Rdx),
                u64::from_le_bytes(expected[..8].try_into().unwrap())
            );
        } else {
            assert_eq!(vector(&cpu), lanes(&expected));
        }
    } else if ram_prefix {
        let mut bytes = [0; 8];
        vm.read_bytes(base, &mut bytes).unwrap();
        assert_eq!(bytes, data[..8]);
    }
}

#[test]
fn access_width_interp() {
    for width in [1, 2, 4, 8, 16, 32, 64] {
        for store in [false, true] {
            access_width(Box::new(InterpreterBackend), width, store, false, false);
            if width > 8 {
                access_width(Box::new(InterpreterBackend), width, store, true, false);
                access_width(Box::new(InterpreterBackend), width, store, false, true);
            }
        }
    }
}

#[test]
fn access_width_jit() {
    for width in [1, 2, 4, 8, 16, 32, 64] {
        for store in [false, true] {
            access_width(Box::new(JitBackend::new()), width, store, false, false);
            if width > 8 {
                access_width(Box::new(JitBackend::new()), width, store, true, false);
                access_width(Box::new(JitBackend::new()), width, store, false, true);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Boundary stops between the transfers of one vector access (xj-ir6.35).
// ---------------------------------------------------------------------------

/// What the driver does at a device stop of the vector access, around its
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
    /// Complete it, request a host stop, then `run`.
    HostAfter,
    /// `run(Some(0))` before the completion.
    ZeroFirst,
    /// `run(Some(0))` after the completion.
    ZeroAfter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bus {
    Read(u64, u8),
    Write(u64, u8, u64),
}

/// The device window of the boundary-stop witnesses, and its operand.
const WINDOW: u64 = 0x3000;
const OPERAND: u64 = 0x3010;

/// A device whose answer names the transfer it answers: 1111h for the first
/// quadword of the operand, 2222h for the second, and so on.
fn device_answer(addr: u64) -> u64 {
    0x1111 * ((addr - OPERAND) / 8 + 1)
}

/// Drive `Vcpu::run` to a stop other than a device, budget or host stop,
/// applying `schedule[i % len]` at the i-th device stop. `None` after twelve
/// transfers: the access does not converge.
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
            Between::Run | Between::HostAfter | Between::ZeroAfter => {}
        }
        match exit {
            Exit::MmioRead { addr, size, .. } => {
                bus.push(Bus::Read(addr, size));
                cpu.complete_mmio_read(device_answer(addr));
            }
            Exit::MmioWrite {
                addr, size, value, ..
            } => {
                bus.push(Bus::Write(addr, size, value));
                cpu.complete_mmio_write();
            }
            _ => unreachable!(),
        }
        match how {
            Between::HostAfter => cpu.exit_request().request_exit(),
            Between::ZeroAfter => {
                let stop = cpu.run(vm, Some(0));
                assert!(matches!(stop, Exit::BudgetExhausted), "{stop:?}");
            }
            _ => {}
        }
        exit = cpu.run(vm, Some(64));
    }
}

/// A 16- or 32-byte vector load or store on a Trap region is one 8-byte
/// `MmioRead` or `MmioWrite` per quadword (spec.md §5.2). A budget stop or a
/// host stop between those exits is taken at the boundary before the retry:
/// the instruction has neither completed nor aborted, so each completed
/// transfer keeps its own answer or acknowledgement, and the completion given
/// after the stop answers the transfer that asked for it. Every transfer is
/// performed once, and the instruction ends as it does on RAM holding the
/// device's values (SDM Vol 2B §4.3, MOVDQU,VMOVDQU8/16/32/64, Operation,
/// pp. 4-66–4-69, where the VEX.256 form moves 32 bytes; MOVUPS, Operation,
/// p. 4-127). Long64 and Compat32, at the interpreter, block and region tiers,
/// with every tier-up setting: compiled code defers the access to the
/// interpreter. Measured at 5b04d7a2: a host stop requested before the second
/// completion dropped the first answer, and the first transfer took the second
/// answer, so MOVDQU loaded 2222h into the low quadword; with a host stop at
/// every device stop the access never completed.
#[test]
fn boundary_stops_between_vector_transfers() {
    use Between::*;
    let schedules: &[&[Between]] = &[
        &[Run],
        &[HostFirst],
        &[Run, HostFirst],
        &[HostRepeat],
        &[Run, HostRepeat],
        &[HostAfter],
        &[ZeroFirst],
        &[ZeroAfter],
    ];
    // (name, bytes, quadwords, store); the operand is [rdi] / [edi].
    let cases: &[(&str, &[u8], u64, bool)] = &[
        ("movdqu xmm0,[rdi]", &[0xf3, 0x0f, 0x6f, 0x07], 2, false),
        ("movups xmm0,[rdi]", &[0x0f, 0x10, 0x07], 2, false),
        ("vmovdqu xmm0,[rdi]", &[0xc5, 0xfa, 0x6f, 0x07], 2, false),
        ("vmovdqu ymm0,[rdi]", &[0xc5, 0xfe, 0x6f, 0x07], 4, false),
        ("movdqu [rdi],xmm0", &[0xf3, 0x0f, 0x7f, 0x07], 2, true),
        ("movups [rdi],xmm0", &[0x0f, 0x11, 0x07], 2, true),
        ("vmovdqu [rdi],xmm0", &[0xc5, 0xfa, 0x7f, 0x07], 2, true),
        ("vmovdqu [rdi],ymm0", &[0xc5, 0xfe, 0x7f, 0x07], 4, true),
    ];
    let setup =
        |mode: x86jit_core::CpuMode, tier: u8, tier_up: Option<u32>, trap: bool, code: &[u8]| {
            let backend: Box<dyn Backend> = match tier {
                0 => Box::new(InterpreterBackend),
                1 => Box::new(JitBackend::new()),
                _ => Box::new(JitBackend::with_superblocks(x86jit_core::RegionCaps {
                    max_blocks: 8,
                    max_icount: 64,
                })),
            };
            let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
            vm.set_cpu_mode(mode);
            vm.set_tier_up_after(tier_up);
            vm.map(CODE, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
            let kind = if trap {
                RegionKind::Trap
            } else {
                RegionKind::Ram
            };
            vm.map(WINDOW, 0x1000, Prot::RW, kind).unwrap();
            if !trap {
                for q in 0..4 {
                    let at = OPERAND + q * 8;
                    vm.mem.write(at, device_answer(at), 8).unwrap();
                }
            }
            // inc ebx ; <case> ; inc esi ; hlt (FF /0 keeps one encoding in both modes)
            let mut bytes = vec![0xff, 0xc3];
            bytes.extend_from_slice(code);
            bytes.extend_from_slice(&[0xff, 0xc6, 0xf4]);
            vm.write_bytes(CODE, &bytes).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rdi, OPERAND);
            cpu.set_xmm(0, 0x8888_7777_6666_5555_4444_3333_2222_1111_u128);
            cpu.set_ymm_hi(0, 0xffff_eeee_dddd_cccc_bbbb_aaaa_9999_8888_u128);
            (vm, cpu)
        };
    let observe = |cpu: &x86jit_core::Vcpu| {
        (
            cpu.reg(Reg::Rbx),
            cpu.reg(Reg::Rsi),
            cpu.reg(Reg::Rip),
            cpu.xmm(0),
            cpu.ymm_hi(0),
        )
    };
    let mut bad = Vec::new();
    let mut checked = 0;
    for mode in [x86jit_core::CpuMode::Long64, x86jit_core::CpuMode::Compat32] {
        for &(name, code, quads, store) in cases {
            // The reference: the same instruction on RAM holding the device's values.
            let (vm, mut cpu) = setup(mode, 0, None, false, code);
            assert!(
                matches!(cpu.run(&vm, Some(64)), Exit::Hlt),
                "{mode:?} {name}"
            );
            let reference = observe(&cpu);
            let expected: Vec<Bus> = (0..quads)
                .map(|q| {
                    let at = OPERAND + q * 8;
                    if store {
                        Bus::Write(at, 8, vm.mem.read(at, 8).unwrap())
                    } else {
                        Bus::Read(at, 8)
                    }
                })
                .collect();
            for tier in 0..3 {
                for tier_up in [None, Some(1), Some(2), Some(3)] {
                    if tier == 0 && tier_up.is_some() {
                        continue;
                    }
                    for schedule in schedules {
                        let (vm, mut cpu) = setup(mode, tier, tier_up, true, code);
                        let (bus, stop) = drive_between(&vm, &mut cpu, schedule);
                        checked += 1;
                        let observed = observe(&cpu);
                        if bus != expected
                            || !matches!(stop, Some(Exit::Hlt))
                            || observed != reference
                        {
                            bad.push(format!(
                                "{mode:?} {name} tier {tier} tier-up {tier_up:?} {schedule:?}: \
                                 {bus:x?} then {stop:?}, expected {expected:x?}; \
                                 state {observed:x?}, RAM run {reference:x?}"
                            ));
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checked, 2 * 8 * 9 * 8);
    assert!(
        bad.is_empty(),
        "{} failures:\n{}",
        bad.len(),
        bad.join("\n")
    );
}
