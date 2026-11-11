//! Engine policy (§5.2): atomic RMW/CAS on Trap regions requires embedder emulation.
//! SDM Vol 2A §3.3 CMPXCHG, Operation; Vol 2D §6.1 XCHG, Description;
//! Vol 3A §§11.1.2.1–11.1.2.2: automatic and software-controlled locking.

use iced_x86::code_asm::*;
use x86jit_core::{
    Backend, Exit, InterpreterBackend, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;
const RAM: u64 = 0x2000;
const MMIO: u64 = 0x3000;
const LOOPS: u64 = 32;

fn rejects_mmio(tier: usize, instruction: &[u8]) {
    rejects_at(tier, instruction, MMIO, false);
}

fn rejects_at(tier: usize, instruction: &[u8], target: u64, reverse: bool) {
    let backend: Box<dyn Backend> = match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        2 => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 8,
            max_icount: 128,
        })),
        _ => unreachable!(),
    };
    let mut vm = Vm::with_backend(VmConfig::flat(0x5000), backend);
    vm.set_tier_up_after(None);
    vm.map(CODE, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
    vm.map(RAM, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
    vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    if reverse {
        vm.map(0x4000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
    }
    vm.write_bytes(RAM, &7u64.to_le_bytes()).unwrap();
    vm.write_bytes(MMIO, &7u64.to_le_bytes()).unwrap();
    let mut operand_before = [0; 4];
    for (i, byte) in operand_before.iter_mut().enumerate() {
        vm.read_bytes(target + i as u64, std::slice::from_mut(byte))
            .unwrap();
    }
    vm.watch_range(MMIO, 8);

    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    let mut done = a.create_label();
    a.jmp(top).unwrap();
    a.set_label(&mut top).unwrap();
    a.inc(r9).unwrap();
    a.mov(rax, 0xaaaa_bbbb_0000_0005u64).unwrap();
    a.mov(rcx, 0xcccc_dddd_0000_0009u64).unwrap();
    // SDM Vol 2A §3.3 CMP, Operation/Flags Affected: 0 - 1.
    a.cmp(r10, 1).unwrap();
    a.db(instruction).unwrap();
    a.inc(r11).unwrap();
    a.dec(r8).unwrap();
    a.jnz(top).unwrap();
    a.cmp(rdi, target as i32).unwrap();
    a.je(done).unwrap();
    // The same instruction executes on RAM before reaching Trap through an
    // internal region edge, so the region witness exercises compiled code.
    a.mov(rdi, target).unwrap();
    a.mov(r8, 1u64).unwrap();
    a.jmp(top).unwrap();
    a.set_label(&mut done).unwrap();
    a.hlt().unwrap();
    let code = a.assemble(CODE).unwrap();
    let offset = code
        .windows(instruction.len())
        .position(|b| b == instruction)
        .unwrap();
    let rip = CODE + offset as u64;
    vm.write_bytes(CODE, &code).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rdi, RAM);
    cpu.set_reg(Reg::R8, LOOPS);
    cpu.cpu.xmm.fill(0x1234);
    cpu.cpu.ymm_hi.fill(0x5678);
    cpu.cpu.zmm_hi.fill([0x9abc; 2]);
    let before = cpu.cpu.clone();
    let mut expected_gpr = before.gpr;
    expected_gpr[0] = 0xaaaa_bbbb_0000_0005;
    expected_gpr[1] = 0xcccc_dddd_0000_0009;
    expected_gpr[7] = target;
    expected_gpr[8] = 1;
    expected_gpr[9] = LOOPS + 1;
    expected_gpr[11] = LOOPS;
    let mut bytes = [0; 15];
    bytes[..instruction.len()].copy_from_slice(instruction);
    for _ in 0..2 {
        let exit = cpu.run(&vm, Some(4 * LOOPS));
        assert!(
            matches!(exit, Exit::UnknownInstruction { addr, bytes: actual, len, .. }
                if addr == rip && actual == bytes && usize::from(len) == instruction.len()),
            "tier={tier}, instruction={instruction:02x?}: {exit:?}"
        );
        assert_eq!(cpu.cpu.rip, rip);
        assert_eq!(cpu.cpu.gpr, expected_gpr);
        assert_eq!(cpu.cpu.flags.to_rflags() & 0x8d5, 0x95);
        assert_eq!(cpu.cpu.xmm, before.xmm);
        assert_eq!(cpu.cpu.ymm_hi, before.ymm_hi);
        assert_eq!(cpu.cpu.zmm_hi, before.zmm_hi);
        assert_eq!(cpu.cpu.pending_mmio, before.pending_mmio);
        assert_eq!(cpu.cpu.pending_mmio_write, before.pending_mmio_write);
        assert_eq!(cpu.cpu.mmio_parts_len, before.mmio_parts_len);
        assert_eq!(cpu.cpu.mmio_retry_active, before.mmio_retry_active);
        let mut operand = [0; 4];
        for (i, byte) in operand.iter_mut().enumerate() {
            vm.read_bytes(target + i as u64, std::slice::from_mut(byte))
                .unwrap();
        }
        assert_eq!(operand, operand_before);
        let mut device = [0; 8];
        vm.read_bytes(MMIO, &mut device).unwrap();
        assert_eq!(device, 7u64.to_le_bytes());
        let mut ram = [0; 8];
        vm.read_bytes(RAM, &mut ram).unwrap();
        assert_eq!(u64::from_le_bytes(ram), expected_ram(instruction));
        assert!(vm.take_dirty_ranges().is_empty());
    }
    if tier == 2 {
        assert!(vm.cache.regions() > 0, "fixture must compile a region");
    }
    // UnknownInstruction delegates emulation to the embedder. It advances RIP
    // after emulating; there is no MMIO transfer to acknowledge or replay.
    cpu.set_reg(Reg::Rip, rip + instruction.len() as u64);
    assert!(matches!(cpu.run(&vm, Some(16)), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::R9), LOOPS + 1);
    assert_eq!(cpu.reg(Reg::R11), LOOPS + 1);
    assert!(!cpu.cpu.pending_mmio_write);
    assert!(cpu.cpu.pending_mmio.is_none());
}

macro_rules! tiers {
    ($interp:ident, $jit:ident, $region:ident, $instructions:expr) => {
        #[test]
        fn $interp() {
            for instruction in $instructions {
                rejects_mmio(0, instruction);
            }
        }
        #[test]
        fn $jit() {
            for instruction in $instructions {
                rejects_mmio(1, instruction);
            }
        }
        #[test]
        fn $region() {
            for instruction in $instructions {
                rejects_mmio(2, instruction);
            }
        }
    };
}

// Memory CMPXCHG/XADD already lower to atomic IR with or without LOCK.
tiers!(
    cmpxchg_mmio_interp,
    cmpxchg_mmio_jit,
    cmpxchg_mmio_region,
    [
        &[0x0f, 0xb1, 0x0f][..],
        &[0xf0, 0x0f, 0xb1, 0x0f][..],
        &[0xf0, 0x0f, 0xb0, 0x0f][..],
        &[0xf0, 0x66, 0x0f, 0xb1, 0x0f][..],
        &[0xf0, 0x48, 0x0f, 0xb1, 0x0f][..],
    ]
);
tiers!(
    xadd_mmio_interp,
    xadd_mmio_jit,
    xadd_mmio_region,
    [
        &[0x0f, 0xc1, 0x0f][..],
        &[0xf0, 0x0f, 0xc1, 0x0f][..],
        &[0xf0, 0x0f, 0xc0, 0x0f][..],
        &[0xf0, 0x66, 0x0f, 0xc1, 0x0f][..],
        &[0xf0, 0x48, 0x0f, 0xc1, 0x0f][..],
    ]
);
tiers!(
    lock_alu_mmio_interp,
    lock_alu_mmio_jit,
    lock_alu_mmio_region,
    [
        &[0xf0, 0x01, 0x0f][..], // add [rdi], ecx
        &[0xf0, 0x00, 0x0f][..],
        &[0xf0, 0x66, 0x01, 0x0f][..],
        &[0xf0, 0x48, 0x01, 0x0f][..],
        &[0xf0, 0x09, 0x0f][..],       // or
        &[0xf0, 0x21, 0x0f][..],       // and
        &[0xf0, 0x29, 0x0f][..],       // sub
        &[0xf0, 0x31, 0x0f][..],       // xor
        &[0xf0, 0xff, 0x07][..],       // inc
        &[0xf0, 0xff, 0x0f][..],       // dec
        &[0xf0, 0xf7, 0x17][..],       // not
        &[0xf0, 0xf7, 0x1f][..],       // neg
        &[0xf0, 0x0f, 0xab, 0x0f][..], // bts
        &[0xf0, 0x0f, 0xb3, 0x0f][..], // btr
        &[0xf0, 0x0f, 0xbb, 0x0f][..], // btc
    ]
);
tiers!(
    xchg_mmio_interp,
    xchg_mmio_jit,
    xchg_mmio_region,
    [
        &[0x87, 0x0f][..],
        &[0x86, 0x0f][..],
        &[0x66, 0x87, 0x0f][..],
        &[0x48, 0x87, 0x0f][..],
        &[0xf0, 0x87, 0x0f][..],
    ]
);

#[test]
fn atomic_mmio_exact_bytes_across_code_mappings() {
    for mode in [x86jit_core::CpuMode::Long64, x86jit_core::CpuMode::Compat32] {
        for jit in [false, true] {
            for instruction in [&[0xf0, 0x0f, 0xb1, 0x0f][..], &[0xf0, 0x87, 0x0f][..]] {
                let backend: Box<dyn Backend> = if jit {
                    Box::new(JitBackend::new())
                } else {
                    Box::new(InterpreterBackend)
                };
                let mut vm = Vm::with_backend(VmConfig::flat(0x4000), backend);
                vm.set_cpu_mode(mode);
                // Each instruction byte has its own mapping; bytes past the
                // instruction are deliberately unmapped.
                for (i, byte) in instruction.iter().enumerate() {
                    let at = CODE + i as u64;
                    vm.map(at, 1, Prot::RX, RegionKind::Ram).unwrap();
                    vm.write_bytes(at, &[*byte]).unwrap();
                }
                vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, CODE);
                cpu.set_reg(Reg::Rdi, MMIO);
                cpu.set_reg(Reg::Rax, 5);
                cpu.set_reg(Reg::Rcx, 9);
                let before = cpu.cpu.clone();
                let mut bytes = [0; 15];
                bytes[..instruction.len()].copy_from_slice(instruction);
                for _ in 0..2 {
                    let exit = cpu.run(&vm, Some(4));
                    assert!(
                        matches!(exit, Exit::UnknownInstruction { addr: CODE, bytes: actual, len, .. }
                            if actual == bytes && usize::from(len) == instruction.len()),
                        "{mode:?}, jit={jit}: {exit:?}"
                    );
                    assert_eq!(cpu.cpu.rip, CODE);
                    assert_eq!(cpu.cpu.gpr, before.gpr);
                    assert_eq!(cpu.cpu.flags, before.flags);
                    assert_eq!(cpu.retired_instructions(), 0);
                }
            }
        }
    }
}

fn expected_ram(bytes: &[u8]) -> u64 {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};
    let insn = Decoder::new(64, bytes, DecoderOptions::NONE).decode();
    let size = insn.memory_size().size();
    let mask = u64::MAX >> (64 - size * 8);
    let src = 0xcccc_dddd_0000_0009 & mask;
    let mut value = 7u64;
    for _ in 0..LOOPS {
        value = match insn.mnemonic() {
            Mnemonic::Cmpxchg => {
                if value == 5 {
                    src
                } else {
                    value
                }
            }
            Mnemonic::Xadd | Mnemonic::Add => value.wrapping_add(src),
            Mnemonic::Xchg => src,
            Mnemonic::Or => value | src,
            Mnemonic::And => value & src,
            Mnemonic::Sub => value.wrapping_sub(src),
            Mnemonic::Xor => value ^ src,
            Mnemonic::Inc => value.wrapping_add(1),
            Mnemonic::Dec => value.wrapping_sub(1),
            Mnemonic::Not => !value,
            Mnemonic::Neg => value.wrapping_neg(),
            Mnemonic::Bts => value | (1 << 9),
            Mnemonic::Btr => value & !(1 << 9),
            Mnemonic::Btc => value ^ (1 << 9),
            other => panic!("unexpected mnemonic: {other:?}"),
        } & mask;
    }
    value
}

macro_rules! boundary_tier {
    ($name:ident, $tier:expr) => {
        #[test]
        fn $name() {
            for instruction in [&[0xf0, 0x01, 0x0f][..], &[0xf0, 0x0f, 0xb1, 0x0f][..]] {
                rejects_at($tier, instruction, 0x2ffe, false);
                rejects_at($tier, instruction, 0x3ffe, true);
            }
        }
    };
}
boundary_tier!(atomic_boundary_interp, 0);
boundary_tier!(atomic_boundary_jit, 1);
boundary_tier!(atomic_boundary_region, 2);
