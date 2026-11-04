//! Bounded MMIO retries, including two executions at the same guest RIP.

use iced_x86::code_asm::*;
use x86jit_core::{Backend, Exit, InterpreterBackend, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;
const MMIO: u64 = 0x4000;

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
    let mut seen = Vec::new();
    for _ in 0..(transfers.len() * 2 + 4) {
        match cpu.run(&vm, None) {
            Exit::MmioRead { addr, size, .. } if !store => {
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
                addr, size, value, ..
            } if store => {
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
