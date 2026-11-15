//! Checked instruction restartability (decision-23 §5, spec.md §16).
//! SDM Vol 3A §§7.5–7.6: fault restart preserves the uncompleted instruction's
//! architectural state. Decision-16 §3 also requires that state across MMIO
//! handoffs; completed transfer evidence is separate engine-owned state.
use iced_x86::{Decoder, DecoderOptions, Mnemonic};
use x86jit_core::{
    CpuModel, ExceptionDelivery, Exit, Prot, RegionKind, SegmentId, StepOptions, StepStop, Vm,
    VmConfig,
};

// Memory kind 1 maps RAM at 0x4000 next to the RAM at 0x5000, so an operand at
// 0x4fff covers two adjacent RAM regions and completes instead of faulting.
#[test]
fn checked_fault_atomicity_sweep() {
    assert_eq!(fault_atomicity_sweep(false), (19_194, 6_293));
}

#[test]
fn protected_code32_fault_atomicity_sweep() {
    assert_eq!(fault_atomicity_sweep(true), (19_062, 6_306));
}

fn fault_atomicity_sweep(protected: bool) -> (usize, usize) {
    let mut faulted = 0;
    let mut cases = 0;
    for prefix in [
        &b""[..],
        &b"\x66"[..],
        &b"\x67"[..],
        &b"\x66\x67"[..],
        &b"\xf0"[..],
    ] {
        for extended in [false, true] {
            for opcode in 0u8..=255 {
                for reg in 0..8 {
                    let mut code = prefix.to_vec();
                    if extended {
                        code.push(0x0f);
                    }
                    code.push(opcode);
                    let address32 = protected != prefix.contains(&0x67);
                    code.push(if address32 { 0x05 } else { 0x06 } | (reg << 3));
                    code.extend_from_slice(if address32 {
                        &b"\xff\x4f\x00\x00"[..]
                    } else {
                        &b"\xff\x4f"[..]
                    });
                    code.extend_from_slice(&[1, 0, 0, 0, 0, 0]);
                    let insn = Decoder::with_ip(
                        if protected { 32 } else { 16 },
                        &code,
                        0x2000,
                        DecoderOptions::MOV_TR,
                    )
                    .decode();
                    use Mnemonic::*;
                    if insn.is_invalid()
                        || !matches!(
                            insn.mnemonic(),
                            Add | Adc
                                | Sub
                                | Sbb
                                | And
                                | Or
                                | Xor
                                | Cmp
                                | Test
                                | Inc
                                | Dec
                                | Neg
                                | Not
                                | Xchg
                                | Xadd
                                | Cmpxchg
                                | Cmpxchg8b
                                | Shl
                                | Shr
                                | Sar
                                | Rol
                                | Ror
                                | Rcl
                                | Rcr
                                | Shld
                                | Shrd
                                | Bt
                                | Bts
                                | Btr
                                | Btc
                                | Mul
                                | Imul
                                | Div
                                | Idiv
                                | Mov
                                | Movzx
                                | Movsx
                                | Push
                                | Pop
                                | Enter
                                | Leave
                                | Les
                                | Lds
                                | Lss
                                | Lfs
                                | Lgs
                                | Bound
                                | Bsf
                                | Bsr
                                | Fst
                                | Fstp
                                | Fist
                                | Fistp
                                | Fbstp
                                | Fsave
                                | Fnsave
                                | Fstenv
                                | Fnstenv
                                | Fstcw
                                | Fnstcw
                                | Fstsw
                                | Fnstsw
                                | Fld
                                | Fild
                                | Fldcw
                                | Fldenv
                                | Frstor
                        )
                    {
                        continue;
                    }
                    code.truncate(insn.len());
                    for kind in 0..3 {
                        let mut vm = Vm::new(VmConfig::flat(0x10000));
                        vm.set_cpu_model(CpuModel::I80386);
                        vm.set_exception_delivery(ExceptionDelivery::Outward);
                        vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
                        vm.map(0x5000, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
                        if kind != 0 {
                            vm.map(
                                0x4000,
                                0x1000,
                                if kind == 1 { Prot::R } else { Prot::RW },
                                if kind == 1 {
                                    RegionKind::Ram
                                } else {
                                    RegionKind::Trap
                                },
                            )
                            .unwrap();
                        }
                        vm.write_bytes(0x2000, &code).unwrap();
                        let mut cpu = vm.new_vcpu();
                        cpu.cpu.rip = 0x2000;
                        if protected {
                            cpu.cpu.legacy.cr0 = 1;
                            cpu.cpu.legacy.segments[0].attributes |= 0x4000;
                            cpu.cpu.legacy.segments[3].attributes |= 0x4000;
                        }
                        cpu.cpu.legacy.segments[SegmentId::Es as usize].attributes = 0x92;
                        cpu.cpu.gpr[..8].copy_from_slice(&[
                            0x83f1, 3, 0x87, 0x10, 0x8000, 0x4fff, 0x4fff, 0x4fff,
                        ]);
                        cpu.cpu.flags.cf = true;
                        cpu.cpu.flags.of = true;
                        cpu.cpu.flags.sf = true;
                        let before = (
                            cpu.cpu.gpr,
                            cpu.cpu.flags,
                            cpu.cpu.rflags_system,
                            cpu.cpu.legacy,
                            [
                                cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs,
                                cpu.cpu.gs,
                            ],
                        );
                        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                        cases += 1;
                        if r.steps() == 0
                            && matches!(
                                r.stop(),
                                StepStop::Exit(
                                    Exit::UnmappedMemory { .. }
                                        | Exit::MmioRead { .. }
                                        | Exit::MmioWrite { .. }
                                        | Exit::Exception { .. }
                                )
                            )
                        {
                            faulted += 1;
                            let after = (
                                cpu.cpu.gpr,
                                cpu.cpu.flags,
                                cpu.cpu.rflags_system,
                                cpu.cpu.legacy,
                                [
                                    cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs,
                                    cpu.cpu.gs,
                                ],
                            );
                            assert_eq!(
                                after, before,
                                "instruction bytes={code:02x?} {:?}, memory kind {kind}, stop={:?}; partial state (GPR/flags/system/hidden-cache/selectors)",
                                insn.mnemonic(), r.stop()
                            );
                        }
                    }
                }
            }
        }
    }
    (cases, faulted)
}
