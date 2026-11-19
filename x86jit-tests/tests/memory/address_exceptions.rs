//! SDM Vol 1 §3.3.7.1; Vol 3A §7.15, Events 12/13: precise #SS(0)/#GP(0).
//! Alignment: Vol 1 §§14.9, 15.7, Tables 14-21/22/23 and 15-6/7;
//! Vol 2A §2.5 and FXSAVE/FXRSTOR, Protected Mode Exceptions.
use iced_x86::code_asm::*;
use x86jit_core::{
    Backend, Exit, InterpreterBackend, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;
const DATA: u64 = 0x4000;
const BAD: u64 = 0x0100_0000_0000_0000; // Non-canonical with either 48 or 57 address bits.

fn backend(tier: usize) -> Box<dyn Backend> {
    match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        2 => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 16,
            max_icount: 256,
        })),
        _ => unreachable!(),
    }
}

fn check(tier: usize, code: &[u8], base: Reg, address: u64, vector: Option<u8>) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x8000), backend(tier));
    vm.set_tier_up_after(None);
    vm.map(0, 0x8000, Prot::RWX, RegionKind::Ram).unwrap();
    // A back-edge ensures the region tier actually compiles a region; RCX is
    // separate from every operand under test. Account for DEC flags below.
    let mut bytes = if tier == 2 {
        vec![0xb9, 2, 0, 0, 0, 0xeb, 0, 0xff, 0xc9, 0x75, 0xfc]
    } else {
        vec![]
    };
    let fault = CODE + bytes.len() as u64;
    bytes.extend_from_slice(code);
    bytes.push(0xf4);
    vm.write_bytes(CODE, &bytes).unwrap();
    vm.write_bytes(DATA, &[0x35; 1024]).unwrap();
    let mut cpu = vm.new_vcpu();
    for (i, r) in [
        Reg::Rax,
        Reg::Rcx,
        Reg::Rdx,
        Reg::Rbx,
        Reg::Rsp,
        Reg::Rbp,
        Reg::Rsi,
        Reg::Rdi,
    ]
    .into_iter()
    .enumerate()
    {
        cpu.set_reg(r, DATA + 0x100 + i as u64 * 8);
    }
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(base, address);
    cpu.cpu.flags.cf = true;
    cpu.cpu.flags.of = true;
    cpu.cpu.flags.set_pf(true);
    for i in 0..32 {
        cpu.set_xmm(i, 0x1234 + i as u128);
        cpu.set_ymm_hi(i, 0x5678 + i as u128);
        cpu.set_zmm_hi(i, 0, 0x9abc + i as u128);
        cpu.set_zmm_hi(i, 1, 0xdef0 + i as u128);
    }
    let mut gpr = cpu.cpu.gpr;
    if tier == 2 {
        gpr[1] = 0;
    }
    let mut entry_flags = cpu.cpu.flags;
    if tier == 2 {
        entry_flags.zf = true;
        entry_flags.sf = false;
        entry_flags.of = false;
        entry_flags.set_af(false);
        entry_flags.set_pf(true);
    }
    let flags = entry_flags.to_rflags();
    let xmm = cpu.cpu.xmm;
    let ymm_hi = cpu.cpu.ymm_hi;
    let zmm_hi = cpu.cpu.zmm_hi;
    let kmask = cpu.cpu.kmask;
    let fpr = cpu.cpu.fpr;
    let fp = (
        cpu.cpu.fpu_cw,
        cpu.cpu.fpu_sw,
        cpu.cpu.fpu_top,
        cpu.cpu.fpu_empty,
    );
    let exit = cpu.run(&vm, Some(100));
    if let Some(expected) = vector {
        assert!(
            matches!(exit, Exit::Exception { addr, vector, .. } if addr == fault && vector == expected),
            "tier={tier}, bytes={code:02x?}, address={address:#x}: {exit:?}"
        );
        assert_eq!(cpu.reg(Reg::Rip), fault);
        assert_eq!(cpu.cpu.gpr, gpr, "GPRs committed on a fault");
        assert_eq!(
            cpu.cpu.flags.to_rflags(),
            flags,
            "flags committed on a fault"
        );
        assert_eq!(cpu.cpu.xmm, xmm, "XMM committed on a fault");
        assert_eq!(
            cpu.cpu.ymm_hi, ymm_hi,
            "YMM upper bits committed on a fault"
        );
        assert_eq!(
            cpu.cpu.zmm_hi, zmm_hi,
            "ZMM upper bits committed on a fault"
        );
        assert_eq!(cpu.cpu.kmask, kmask);
        assert_eq!(cpu.cpu.fpr, fpr);
        assert_eq!(
            (
                cpu.cpu.fpu_cw,
                cpu.cpu.fpu_sw,
                cpu.cpu.fpu_top,
                cpu.cpu.fpu_empty
            ),
            fp
        );
        let mut memory = [0; 1024];
        vm.read_bytes(DATA, &mut memory).unwrap();
        assert_eq!(memory, [0x35; 1024], "memory committed on a fault");
    } else {
        assert!(
            matches!(exit, Exit::Hlt),
            "tier={tier}, bytes={code:02x?}: {exit:?}"
        );
    }
    if tier == 2 {
        assert!(vm.cache.regions() > 0, "must exercise compiled region");
    }
}

fn noncanonical(tier: usize) {
    for bad in [BAD, 0x0000_8000_0000_0000, 0xffff_7fff_ffff_ffff] {
        check(tier, &[0x48, 0x8b, 0x03], Reg::Rbx, bad, Some(13));
        check(tier, &[0x48, 0x89, 0x03], Reg::Rbx, bad, Some(13));
        check(tier, &[0x48, 0x83, 0x03, 1], Reg::Rbx, bad, Some(13));
    }
    for code in [&[0x50][..], &[0x58], &[0xc3], &[0xe8, 0, 0, 0, 0]] {
        check(tier, code, Reg::Rsp, BAD + 0x100, Some(12));
    }
    check(tier, &[0xc9], Reg::Rbp, BAD, Some(12)); // LEAVE.
    check(tier, &[0x48, 0x8b, 0x45, 0], Reg::Rbp, BAD, Some(12));
    check(tier, &[0x48, 0x8b, 0x04, 0x24], Reg::Rsp, BAD, Some(12));
    check(tier, &[0x36, 0x48, 0x8b, 0x03], Reg::Rbx, BAD, Some(13)); // ignored SS.
    check(tier, &[0x3e, 0x48, 0x8b, 0x45, 0], Reg::Rbp, BAD, Some(12)); // ignored DS.
    check(tier, &[0x64, 0x48, 0x8b, 0x45, 0], Reg::Rbp, BAD, Some(13)); // FS.
    check(
        tier,
        &[0x65, 0x48, 0x89, 0x04, 0x24],
        Reg::Rsp,
        BAD,
        Some(13),
    ); // GS.
    check(
        tier,
        &[0x48, 0x89, 0x03, 0x31, 0xc0],
        Reg::Rbx,
        BAD,
        Some(13),
    );
    check(tier, &[0xff, 0x33], Reg::Rbx, BAD, Some(13)); // PUSH reads DS before SS.
    check(tier, &[0x8f, 0x03], Reg::Rbx, BAD, Some(13)); // POP writes DS after reading SS.
    for code in [
        &[0x0f, 0x10, 0x03][..],
        &[0x0f, 0x11, 0x03],
        &[0xc5, 0xfc, 0x58, 0x03],
        &[0x0f, 0xae, 0x03],
        &[0x0f, 0xae, 0x0b],
    ] {
        check(tier, code, Reg::Rbx, BAD, Some(13));
    }
    check(tier, &[0x0f, 0x10, 0x45, 0], Reg::Rbp, BAD, Some(12));
    // The upper canonical half is unmapped, rather than a canonicality exception.
    let vm = Vm::with_backend(VmConfig::flat(0x8000), backend(tier));
    assert_eq!(
        vm.mem.read(0xffff_8000_0000_0000, 8),
        Err(x86jit_core::MemTrap::Unmapped)
    );
}

fn alignment(tier: usize) {
    let mut a = CodeAssembler::new(64).unwrap();
    a.movaps(xmm0, xmmword_ptr(rbx)).unwrap();
    let movaps = a.assemble(CODE).unwrap();
    let required: &[&[u8]] = &[
        &movaps,
        &[0x0f, 0x29, 0x03], // MOVAPS store.
        &[0x66, 0x0f, 0x6f, 0x03],
        &[0x66, 0x0f, 0x7f, 0x03], // MOVDQA.
        &[0x0f, 0x58, 0x03],       // ADDPS.
        &[0x66, 0x0f, 0x58, 0x03], // ADDPD.
        &[0x66, 0x0f, 0xfe, 0x03], // PADDD.
        &[0xc5, 0xf8, 0x28, 0x03], // VMOVAPS xmm.
        &[0xc5, 0xfc, 0x28, 0x03],
        &[0xc5, 0xfc, 0x29, 0x03], // VMOVAPS ymm.
        &[0x0f, 0xae, 0x03],
        &[0x0f, 0xae, 0x0b], // FXSAVE/FXRSTOR.
    ];
    for code in required {
        check(tier, code, Reg::Rbx, DATA + 1, Some(13));
    }
    // All explicitly aligned move families and vector widths share the rule.
    let mut moves = Vec::new();
    macro_rules! aligned {
        ($method:ident($dst:expr, $src:expr)) => {{
            let mut a = CodeAssembler::new(64).unwrap();
            a.$method($dst, $src).unwrap();
            moves.push(a.assemble(CODE).unwrap());
        }};
    }
    aligned!(movapd(xmm0, xmmword_ptr(rbx)));
    aligned!(movntps(xmmword_ptr(rbx), xmm0));
    aligned!(movntpd(xmmword_ptr(rbx), xmm0));
    aligned!(movntdq(xmmword_ptr(rbx), xmm0));
    aligned!(movntdqa(xmm0, xmmword_ptr(rbx)));
    aligned!(vmovapd(ymm0, ymmword_ptr(rbx)));
    aligned!(vmovdqa(ymm0, ymmword_ptr(rbx)));
    aligned!(vmovntps(ymmword_ptr(rbx), ymm0));
    aligned!(vmovntpd(ymmword_ptr(rbx), ymm0));
    aligned!(vmovntdq(ymmword_ptr(rbx), ymm0));
    aligned!(vmovntdqa(ymm0, ymmword_ptr(rbx)));
    aligned!(vmovaps(zmm0, zmmword_ptr(rbx)));
    aligned!(vmovapd(zmmword_ptr(rbx), zmm0));
    aligned!(vmovdqa32(zmm0, zmmword_ptr(rbx)));
    aligned!(vmovdqa64(zmmword_ptr(rbx), zmm0));
    for code in moves {
        check(tier, &code, Reg::Rbx, DATA + 1, Some(13));
        check(tier, &code, Reg::Rbx, DATA, None);
    }
    let mut zmm = CodeAssembler::new(64).unwrap();
    zmm.vmovaps(zmm0, zmmword_ptr(rbx)).unwrap();
    check(
        tier,
        &zmm.assemble(CODE).unwrap(),
        Reg::Rbx,
        DATA + 32,
        Some(13),
    );
    for code in [&[0x48, 0x0f, 0xae, 0x03][..], &[0x48, 0x0f, 0xae, 0x0b]] {
        check(tier, code, Reg::Rbx, DATA + 1, Some(13));
    }
    for opcode in [0x60, 0x61, 0x62, 0x63] {
        check(
            tier,
            &[0x66, 0x0f, 0x3a, opcode, 0x03, 0],
            Reg::Rbx,
            DATA + 1,
            None,
        );
    }

    // 16-byte alignment is insufficient for a 256-bit aligned move.
    check(
        tier,
        &[0xc5, 0xfc, 0x28, 0x03],
        Reg::Rbx,
        DATA + 16,
        Some(13),
    );
    for code in [
        &[0x0f, 0x10, 0x03][..],
        &[0x0f, 0x11, 0x03], // MOVUPS.
        &[0xf3, 0x0f, 0x6f, 0x03],
        &[0xf3, 0x0f, 0x7f, 0x03], // MOVDQU.
        &[0xf2, 0x0f, 0xf0, 0x03], // LDDQU.
        &[0xc5, 0xf8, 0x58, 0x03],
        &[0xc5, 0xfc, 0x58, 0x03], // VADDPS.
        &[0xf3, 0x0f, 0x58, 0x03], // scalar ADDSS.
    ] {
        check(tier, code, Reg::Rbx, DATA + 1, None);
    }
    // A later flag writer must not hide instruction-entry flags on the fault.
    check(
        tier,
        &[0x0f, 0x28, 0x03, 0x31, 0xc0],
        Reg::Rbx,
        DATA + 1,
        Some(13),
    );
    // Alignment #GP is independent of the operand's default segment.
    check(tier, &[0x0f, 0x28, 0x45, 0], Reg::Rbp, DATA + 1, Some(13));
}

#[test]
fn noncanonical_interp() {
    noncanonical(0);
}
#[test]
fn noncanonical_jit() {
    noncanonical(1);
}
#[test]
fn noncanonical_region() {
    noncanonical(2);
}
#[test]
fn alignment_interp() {
    alignment(0);
}
#[test]
fn alignment_jit() {
    alignment(1);
}
#[test]
fn alignment_region() {
    alignment(2);
}

#[derive(Clone, Copy)]
enum Expected {
    Exception(u8),
    Unmapped(u64),
    Hlt,
}

#[allow(clippy::too_many_arguments)]
fn whole_operand(
    tier: usize,
    mode: x86jit_core::CpuMode,
    code: &[u8],
    base: Reg,
    address: u64,
    expected: Expected,
    repeat: bool,
    routed: bool,
) {
    configured_operand(
        tier,
        mode,
        code,
        base,
        address,
        expected,
        repeat,
        routed,
        Setup::default(),
    );
}

#[derive(Default, Clone, Copy)]
struct Setup {
    k1: Option<u64>,
    vector_mask: Option<[u128; 4]>,
    add_before: bool,
    retry_fault: bool,
}

#[allow(clippy::too_many_arguments)]
fn configured_operand(
    tier: usize,
    mode: x86jit_core::CpuMode,
    code: &[u8],
    base: Reg,
    address: u64,
    expected: Expected,
    repeat: bool,
    routed: bool,
    setup: Setup,
) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x8000), backend(tier));
    vm.set_tier_up_after(None);
    vm.set_cpu_mode(mode);
    vm.mem.set_host_fault_recovery(routed);
    vm.map(0, 0x8000, Prot::RWX, RegionKind::Ram).unwrap();
    // R8 is independent of string pointers, counts, and paired-CAS operands.
    let mut bytes = if tier == 2 && mode == x86jit_core::CpuMode::Compat32 {
        // Compat32 has no R8: use ECX, which this control does not address through.
        vec![0xb9, 2, 0, 0, 0, 0xeb, 0, 0xff, 0xc9, 0x75, 0xfc]
    } else if tier == 2 {
        vec![
            0x41, 0xb8, 2, 0, 0, 0, 0xeb, 0, 0x41, 0xff, 0xc8, 0x75, 0xfb,
        ]
    } else {
        vec![]
    };
    if repeat {
        bytes.extend_from_slice(&[0xb9, 4, 0, 0, 0]);
    }
    if setup.add_before {
        bytes.extend_from_slice(&[0x48, 0x83, 0xc0, 1]);
    }
    let rip = CODE + bytes.len() as u64;
    bytes.extend_from_slice(code);
    bytes.push(0xf4);
    vm.write_bytes(CODE, &bytes).unwrap();
    let mut image = vec![0x35; 1024];
    // Valid FXRSTOR image: SDM Vol 2A, FXRSTOR, Description and #GP.
    // Reserved MXCSR bits stay zero, including in the aligned completion control.
    if code.ends_with(&[0x0f, 0xae, 0x0b]) {
        image.fill(0);
        let start = address.checked_sub(DATA).filter(|n| *n <= 512).unwrap_or(0) as usize;
        image[start..start + 2].copy_from_slice(&0x037fu16.to_le_bytes());
        image[start + 24..start + 28].copy_from_slice(&0x1f80u32.to_le_bytes());
    }
    vm.write_bytes(DATA, &image).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rax, 0x1234);
    cpu.set_reg(
        Reg::Rcx,
        if repeat {
            4
        } else if tier == 2 && mode == x86jit_core::CpuMode::Compat32 {
            0
        } else {
            0x5678
        },
    );
    cpu.set_reg(Reg::Rdx, 0x9abc);
    for reg in [Reg::Rbx, Reg::Rbp, Reg::Rsp, Reg::Rsi, Reg::Rdi] {
        cpu.set_reg(reg, DATA);
    }
    cpu.set_reg(base, address);
    cpu.set_reg(Reg::Rip, CODE);
    if mode == x86jit_core::CpuMode::Compat32 {
        cpu.cpu.fs_base = address;
    }
    cpu.cpu.flags.cf = true;
    cpu.cpu.flags.zf = true;
    cpu.cpu.flags.set_pf(true);
    for i in 0..32 {
        cpu.set_xmm(i, 0x1234 + i as u128);
        cpu.set_ymm_hi(i, 0x5678 + i as u128);
    }
    if let Some(k) = setup.k1 {
        cpu.cpu.kmask[1] = k;
    }
    if let Some(mask) = setup.vector_mask {
        cpu.cpu.set_vec(1, mask, 64);
    }
    let mut before = cpu.cpu.clone();
    if setup.add_before {
        // SDM Vol 2A, ADD, Operation/Flags Affected: 0x1234 + 1 is 0x1235.
        before.gpr[0] += 1;
        before.flags.cf = false;
        before.flags.zf = false;
    }
    let exit = cpu.run(&vm, Some(100));
    match expected {
        Expected::Exception(vector) => assert!(
            matches!(exit, Exit::Exception { addr, vector: got, .. } if addr == rip && got == vector),
            "tier={tier} routed={routed} code={code:02x?} address={address:#x}: {exit:?}"
        ),
        Expected::Unmapped(at) => assert!(
            matches!(exit, Exit::UnmappedMemory { addr, .. } if addr == at),
            "tier={tier} routed={routed} code={code:02x?}: {exit:?}"
        ),
        Expected::Hlt => {
            assert!(matches!(exit, Exit::Hlt), "{exit:?}");
            return;
        }
    }
    assert_eq!(cpu.reg(Reg::Rip), rip);
    if setup.retry_fault {
        let again = cpu.run(&vm, Some(100));
        assert!(matches!(again, Exit::Exception { addr, vector, .. }
            if addr == rip && matches!(expected, Expected::Exception(v) if v == vector)));
        assert_eq!(cpu.reg(Reg::Rip), rip);
    }
    assert_eq!(cpu.cpu.gpr, before.gpr, "GPRs committed: {code:02x?}");
    assert_eq!(cpu.cpu.flags.to_rflags(), before.flags.to_rflags());
    assert_eq!(cpu.cpu.xmm, before.xmm);
    assert_eq!(cpu.cpu.ymm_hi, before.ymm_hi);
    assert_eq!(cpu.cpu.zmm_hi, before.zmm_hi);
    assert_eq!(cpu.cpu.kmask, before.kmask);
    assert_eq!(cpu.cpu.fpr, before.fpr);
    assert_eq!(
        (
            cpu.cpu.fpu_cw,
            cpu.cpu.fpu_sw,
            cpu.cpu.fpu_top,
            cpu.cpu.fpu_empty
        ),
        (
            before.fpu_cw,
            before.fpu_sw,
            before.fpu_top,
            before.fpu_empty
        )
    );
    let mut after = vec![0; image.len()];
    vm.read_bytes(DATA, &mut after).unwrap();
    assert_eq!(after, image);
    if tier == 2 {
        assert!(vm.cache.regions() > 0);
    }
}

fn strings(tier: usize) {
    use x86jit_core::CpuMode::Long64;
    // SDM Vol 2A, LODS and CMPS; Vol 2B, STOS, MOVS and SCAS,
    // 64-Bit Mode Exceptions;
    // AMD APM Vol 2 §5.3.1: the whole element must be canonical.
    for routed in [false, true] {
        for (code, reg) in [
            (&[0x48, 0xab][..], Reg::Rdi),
            (&[0x48, 0xad], Reg::Rsi),
            (&[0x48, 0xa5], Reg::Rsi),
            (&[0x48, 0xa5], Reg::Rdi),
            (&[0x48, 0xaf], Reg::Rdi),
            (&[0x48, 0xa7], Reg::Rsi),
            (&[0x48, 0xa7], Reg::Rdi),
            (&[0x36, 0x48, 0xab], Reg::Rdi),
            (&[0x64, 0x48, 0xad], Reg::Rsi),
        ] {
            for repeat in [false, true] {
                let mut bytes = if repeat { vec![0xf3] } else { vec![] };
                bytes.extend_from_slice(code);
                for address in [BAD, 0x0000_7fff_ffff_fffc] {
                    whole_operand(
                        tier,
                        Long64,
                        &bytes,
                        reg,
                        address,
                        Expected::Exception(13),
                        repeat,
                        routed,
                    );
                }
            }
        }
    }
}

fn pairs(tier: usize) {
    use x86jit_core::CpuMode::Long64;
    // SDM Vol 2A, CMPXCHG8B/CMPXCHG16B, 64-Bit Mode Exceptions.
    for routed in [false, true] {
        for (operand, reg, vector) in [
            (&[0x0b][..], Reg::Rbx, 13),
            (&[0x4d, 0], Reg::Rbp, 12),
            (&[0x0c, 0x24], Reg::Rsp, 12),
        ] {
            for wide in [false, true] {
                let mut bytes = vec![0xf0];
                if wide {
                    bytes.push(0x48);
                }
                bytes.extend_from_slice(&[0x0f, 0xc7]);
                bytes.extend_from_slice(operand);
                whole_operand(
                    tier,
                    Long64,
                    &bytes,
                    reg,
                    BAD,
                    Expected::Exception(vector),
                    false,
                    routed,
                );
                if !wide {
                    whole_operand(
                        tier,
                        Long64,
                        &bytes,
                        reg,
                        0x0000_7fff_ffff_fffc,
                        Expected::Exception(vector),
                        false,
                        routed,
                    );
                }
            }
        }
    }
}

fn straddles(tier: usize) {
    use x86jit_core::CpuMode::Long64;
    // AMD APM Vol 2 §5.3.1: every byte, including later vector pieces.
    for routed in [false, true] {
        for (code, address) in [
            (&[0x48, 0x8b, 0x03][..], 0x0000_7fff_ffff_fffc),
            (&[0x48, 0x89, 0x03], 0x0000_7fff_ffff_fffc),
            (&[0x0f, 0x10, 0x03], 0x0000_7fff_ffff_fff8),
            (&[0x0f, 0x11, 0x03], 0x0000_7fff_ffff_fff8),
            (&[0xc5, 0xfc, 0x10, 0x03], 0x0000_7fff_ffff_fff0),
            (&[0xc5, 0xfc, 0x11, 0x03], 0x0000_7fff_ffff_fff0),
            (&[0xdb, 0x2b], 0x0000_7fff_ffff_fffc), // FLD m80.
            (&[0xdd, 0x03], 0x0000_7fff_ffff_fffc), // FLD m64.
            (&[0xd9, 0x03], 0x0000_7fff_ffff_fffe), // FLD m32.
            (&[0xd9, 0x2b], 0x0000_7fff_ffff_ffff), // FLDCW m16.
            (&[0xdb, 0x3b], 0x0000_7fff_ffff_fffc), // FSTP m80.
            (&[0xd9, 0x33], 0x0000_7fff_ffff_fff0), // FNSTENV m28.
            (&[0x0f, 0xae, 0x03], 0x0000_7fff_ffff_ff00), // FXSAVE m512.
            (&[0x0f, 0xae, 0x0b], 0x0000_7fff_ffff_ff00),
            (&[0xc4, 0xe2, 0x7d, 0x1a, 0x03], 0x0000_7fff_ffff_fff8), // VBROADCASTF128.
        ] {
            whole_operand(
                tier,
                Long64,
                code,
                Reg::Rbx,
                address,
                Expected::Exception(13),
                false,
                routed,
            );
        }
        whole_operand(
            tier,
            Long64,
            &[0x0f, 0x10, 0x45, 0],
            Reg::Rbp,
            0x0000_7fff_ffff_fff8,
            Expected::Exception(12),
            false,
            routed,
        );
    }
}

fn upper_controls(tier: usize) {
    use x86jit_core::CpuMode::Long64;
    // SDM Vol 1 §3.3.7.1: both halves of canonical space remain canonical.
    for routed in [false, true] {
        for code in [
            &[0x48, 0x8b, 0x03][..],
            &[0x48, 0x89, 0x03],
            &[0xdd, 0x03],
            &[0x0f, 0xae, 0x03],
            &[0xc4, 0xe2, 0x7d, 0x1a, 0x03],
        ] {
            whole_operand(
                tier,
                Long64,
                code,
                Reg::Rbx,
                0xffff_8000_0000_0000,
                Expected::Unmapped(0xffff_8000_0000_0000),
                false,
                routed,
            );
        }
        for address in [
            0xffff_8000_0000_0000,
            0xffff_ffff_ffff_fff8,
            0x0000_7fff_ffff_fff8,
        ] {
            whole_operand(
                tier,
                Long64,
                &[0x48, 0x8b, 0x03],
                Reg::Rbx,
                address,
                Expected::Unmapped(address),
                false,
                routed,
            );
            whole_operand(
                tier,
                Long64,
                &[0x50],
                Reg::Rsp,
                address.wrapping_add(8),
                Expected::Unmapped(address),
                false,
                routed,
            );
        }
    }
}

fn compat_controls(tier: usize) {
    // Retain the current flat Compat32 FS-base behavior; base truncation is separate.
    for address in [BAD, 0x0000_8000_0000_0000, 0x0000_7fff_ffff_f000] {
        whole_operand(
            tier,
            x86jit_core::CpuMode::Compat32,
            &[0x64, 0xa1, 0, 0x40, 0, 0],
            Reg::Rbx,
            address,
            Expected::Unmapped(address + DATA),
            false,
            false,
        );
    }
}

fn aligned_fxstate(tier: usize) {
    for address in [DATA, DATA + 16] {
        for rex in [false, true] {
            for op in [0x03, 0x0b] {
                let mut bytes = if rex { vec![0x48] } else { vec![] };
                bytes.extend_from_slice(&[0x0f, 0xae, op]);
                whole_operand(
                    tier,
                    x86jit_core::CpuMode::Long64,
                    &bytes,
                    Reg::Rbx,
                    address,
                    Expected::Hlt,
                    false,
                    false,
                );
            }
        }
    }
}

fn pcmp64(tier: usize) {
    // SDM Vol 2A §2.5, Table 2-15 note (***); Vol 2B, PCMPESTRI/M,
    // Other Exceptions: REX.W changes the length registers, not alignment.
    for opcode in [0x60, 0x61] {
        check(
            tier,
            &[0x66, 0x48, 0x0f, 0x3a, opcode, 0x03, 0],
            Reg::Rbx,
            DATA + 1,
            None,
        );
    }
}

macro_rules! tier_witnesses {
    ($module:ident, $body:ident) => {
        mod $module {
            use super::*;
            #[test]
            fn interp() {
                $body(0);
            }
            #[test]
            fn jit() {
                $body(1);
            }
            #[test]
            fn region() {
                $body(2);
            }
        }
    };
}
tier_witnesses!(string_addresses, strings);
tier_witnesses!(paired_cas_addresses, pairs);
tier_witnesses!(whole_operand_straddles, straddles);
tier_witnesses!(canonical_upper_half, upper_controls);
tier_witnesses!(compat32_canonicality, compat_controls);
tier_witnesses!(aligned_fxstate_controls, aligned_fxstate);
tier_witnesses!(pcmpestr_rex_w, pcmp64);

// Each entry reaches a distinct fault-recording vector helper or width formula.
// AMD APM Vol 2 §5.3.1: every selected byte of the operand must be canonical.
fn helper_forms(base: AsmRegister64) -> Vec<(&'static str, Vec<u8>, u64)> {
    let mut cases = vec![];
    macro_rules! form {
        ($name:literal, $width:expr, $method:ident($($arg:expr),*)) => {{
            let mut a = CodeAssembler::new(64).unwrap();
            a.$method($($arg),*).unwrap();
            cases.push(($name, a.assemble(CODE).unwrap(), $width));
        }};
    }
    form!(
        "vmaskmov-load",
        32,
        vmaskmovps(ymm0, ymm1, ymmword_ptr(base))
    );
    form!(
        "vmaskmov-store",
        32,
        vmaskmovps(ymmword_ptr(base), ymm1, ymm0)
    );
    form!("k-load", 64, vmovups(zmm0.k1(), zmmword_ptr(base)));
    form!("k-store", 64, vmovups(zmmword_ptr(base).k1(), zmm0));
    form!(
        "gfni-ymm",
        32,
        vgf2p8affineqb(ymm0, ymm1, ymmword_ptr(base), 0)
    );
    form!("gfni-zmm", 64, vgf2p8mulb(zmm0, zmm1, zmmword_ptr(base)));
    form!(
        "gfni-affine-zmm",
        64,
        vgf2p8affineqb(zmm0, zmm1, zmmword_ptr(base), 0)
    );
    form!("permute-two", 64, vpermt2d(zmm0, zmm1, zmmword_ptr(base)));
    form!("permute-one", 64, vpermd(zmm0, zmm1, zmmword_ptr(base)));
    form!("narrow-m256", 32, vpmovqd(ymmword_ptr(base), zmm1));
    form!("narrow-m64", 8, vpmovqb(qword_ptr(base), zmm1));
    form!(
        "broadcast-m128",
        16,
        vbroadcastf128(ymm0, xmmword_ptr(base))
    );
    form!(
        "broadcast-m256",
        32,
        vbroadcastf32x8(zmm0, ymmword_ptr(base))
    );
    form!("fma-m128", 16, vfmadd231ps(xmm0, xmm1, xmmword_ptr(base)));
    form!("fma-m256", 32, vfmadd231ps(ymm0, ymm1, ymmword_ptr(base)));
    form!("fma-m512", 64, vfmadd231ps(zmm0, zmm1, zmmword_ptr(base)));
    form!("fma-m32", 4, vfmadd231ss(xmm0, xmm1, dword_ptr(base)));
    form!("fma-m64", 8, vfmadd231sd(xmm0, xmm1, qword_ptr(base)));
    cases
}

fn helper_widths(tier: usize) {
    for routed in [false, true] {
        for (name, code, width) in helper_forms(rbx) {
            // Adjacent controls reject both undersized and oversized carriers.
            for (address, expected) in [
                (
                    (1u64 << 47) - width,
                    Expected::Unmapped((1u64 << 47) - width),
                ),
                ((1u64 << 47) - width / 2, Expected::Exception(13)),
            ] {
                configured_operand(
                    tier,
                    x86jit_core::CpuMode::Long64,
                    &code,
                    Reg::Rbx,
                    address,
                    expected,
                    false,
                    routed,
                    Setup {
                        k1: Some(u64::MAX),
                        vector_mask: Some([u128::MAX; 4]),
                        add_before: true,
                        ..Default::default()
                    },
                );
            }
            std::hint::black_box(name);
        }
    }
}

fn helper_stack(tier: usize) {
    // SDM Vol 1 §3.3.7.1: implicit SS for RSP/RBP applies to helpers too.
    for routed in [false, true] {
        for (code, reg) in [
            (&[0xdd, 0x45, 0][..], Reg::Rbp), // FLD m64.
            (&[0xdb, 0x2c, 0x24], Reg::Rsp),  // FLD m80.
            (&[0x0f, 0xae, 0x45, 0], Reg::Rbp),
            (&[0x0f, 0xae, 0x0c, 0x24], Reg::Rsp),
        ] {
            configured_operand(
                tier,
                x86jit_core::CpuMode::Long64,
                code,
                reg,
                BAD,
                Expected::Exception(12),
                false,
                routed,
                Setup {
                    add_before: true,
                    ..Default::default()
                },
            );
        }
        // Every vector memory helper is also checked through an RBP base.
        for (_, code, _) in helper_forms(rbp) {
            configured_operand(
                tier,
                x86jit_core::CpuMode::Long64,
                &code,
                Reg::Rbp,
                BAD,
                Expected::Exception(12),
                false,
                routed,
                Setup {
                    k1: Some(u64::MAX),
                    vector_mask: Some([u128::MAX; 4]),
                    add_before: true,
                    ..Default::default()
                },
            );
        }
    }
}

fn gfni_instruction_rip(tier: usize) {
    // SDM Vol 3A §6.5: a restarted fault resumes at the owning instruction.
    for routed in [false, true] {
        for (base, reg, vector) in [
            (rbx, Reg::Rbx, 13),
            (rbp, Reg::Rbp, 12),
            (rsp, Reg::Rsp, 12),
        ] {
            for (_, code, _) in helper_forms(base)
                .into_iter()
                .filter(|c| c.0.starts_with("gfni"))
            {
                configured_operand(
                    tier,
                    x86jit_core::CpuMode::Long64,
                    &code,
                    reg,
                    BAD,
                    Expected::Exception(vector),
                    false,
                    routed,
                    Setup {
                        add_before: true,
                        retry_fault: true,
                        ..Default::default()
                    },
                );
            }
        }
    }
}
tier_witnesses!(gfni_fault_instruction_rip, gfni_instruction_rip);

fn mask_suppression(tier: usize) {
    // SDM Vol 2C, VMASKMOV/VPMASKMOV; Vol 1 §15.6.1; Vol 2A §§2.8.1,
    // 2.8.2 and 2.8.4: masked-off memory elements do not cause faults.
    let mut cases = vec![];
    macro_rules! form {
        ($vec:expr, $elem:expr, $width:expr, $method:ident($($arg:expr),*)) => {{
            let mut a = CodeAssembler::new(64).unwrap(); a.$method($($arg),*).unwrap();
            cases.push((a.assemble(CODE).unwrap(), $vec, $elem, $width));
        }};
    }
    form!(true, 4, 32, vmaskmovps(ymm0, ymm1, ymmword_ptr(rbx)));
    form!(true, 4, 32, vmaskmovps(ymmword_ptr(rbx), ymm1, ymm0));
    form!(true, 4, 32, vpmaskmovd(ymm0, ymm1, ymmword_ptr(rbx)));
    form!(true, 8, 32, vpmaskmovq(ymmword_ptr(rbx), ymm1, ymm0));
    form!(false, 1, 64, vmovdqu8(zmm0.k1().z(), zmmword_ptr(rbx)));
    form!(false, 1, 64, vmovdqu8(zmmword_ptr(rbx).k1(), zmm0));
    form!(false, 4, 64, vmovdqu32(zmmword_ptr(rbx).k1(), zmm0));
    form!(false, 4, 64, vmovups(zmm0.k1(), zmmword_ptr(rbx)));
    form!(false, 4, 64, vfmadd231ps(zmm0.k1(), zmm1, zmmword_ptr(rbx)));
    form!(false, 1, 64, vgf2p8mulb(zmm0.k1(), zmm1, zmmword_ptr(rbx)));
    form!(false, 4, 16, vbroadcastf32x4(zmm0.k1(), xmmword_ptr(rbx)));
    for routed in [false, true] {
        for (code, vector, elem, width) in &cases {
            let address = (1u64 << 47) - width / 2;
            let last = width / elem - 1;
            for (mask, expected) in [
                (1u64, Expected::Unmapped(address)),
                (u64::MAX, Expected::Exception(13)),
                (1 << last, Expected::Exception(13)),
                ((1 << last) | 1, Expected::Exception(13)),
                (0, Expected::Hlt),
            ] {
                let mut vm = [0u128; 4];
                for i in 0..width / elem {
                    if mask & (1 << i) != 0 {
                        let byte = (i * elem) as usize;
                        vm[byte / 16] |= 1u128 << ((byte % 16 + *elem as usize) * 8 - 1);
                    }
                }
                let setup = Setup {
                    k1: Some(mask),
                    vector_mask: if *vector { Some(vm) } else { None },
                    add_before: true,
                    ..Default::default()
                };
                configured_operand(
                    tier,
                    x86jit_core::CpuMode::Long64,
                    code,
                    Reg::Rbx,
                    address,
                    expected,
                    false,
                    routed,
                    setup,
                );
                if mask == 0 || mask == 1 {
                    configured_operand(
                        tier,
                        x86jit_core::CpuMode::Long64,
                        code,
                        Reg::Rbx,
                        if mask == 0 { BAD } else { DATA },
                        Expected::Hlt,
                        false,
                        routed,
                        setup,
                    );
                }
            }
        }
    }
}

fn repeated_broadcast_mask(tier: usize) {
    // SDM Vol 2C, VBROADCASTF32X4, Operation: output j selects source j modulo 4.
    let mut a = CodeAssembler::new(64).unwrap();
    a.vbroadcastf32x4(zmm0.k1(), xmmword_ptr(rbx)).unwrap();
    for routed in [false, true] {
        for (address, expected) in [
            ((1u64 << 47) - 8, Expected::Unmapped((1u64 << 47) - 8)),
            (BAD, Expected::Exception(13)),
        ] {
            configured_operand(
                tier,
                x86jit_core::CpuMode::Long64,
                &a.assemble(CODE).unwrap(),
                Reg::Rbx,
                address,
                expected,
                false,
                routed,
                Setup {
                    k1: Some(1 << 8),
                    add_before: true,
                    ..Default::default()
                },
            );
        }
    }
}
tier_witnesses!(repeated_broadcast_mask_source, repeated_broadcast_mask);

fn embedded_broadcast(tier: usize) {
    // SDM Vol 2A §2.7.7: {1toN} carries one decoded element.
    let mut cases = vec![];
    macro_rules! form { ($method:ident($($arg:expr),*)) => {{
        let mut a = CodeAssembler::new(64).unwrap(); a.$method($($arg),*).unwrap();
        cases.push(a.assemble(CODE).unwrap());
    }}; }
    form!(vpaddd(zmm0, zmm1, dword_bcst(rbx)));
    form!(vpaddd(xmm0, xmm1, dword_bcst(rbx)));
    form!(vpxord(zmm0, zmm1, dword_bcst(rbx)));
    form!(vfmadd231ps(zmm0, zmm1, dword_bcst(rbx)));
    form!(vfmadd231ps(ymm0, ymm1, dword_bcst(rbx)));
    form!(vaddps(xmm0, xmm1, dword_bcst(rbx)));
    form!(vpcmpeqd(k1, zmm1, dword_bcst(rbx)));
    form!(vpermilps(xmm0, dword_bcst(rbx), 0x1b));
    for routed in [false, true] {
        for code in &cases {
            for (address, expected) in [
                ((1u64 << 47) - 4, Expected::Unmapped((1u64 << 47) - 4)),
                ((1u64 << 47) - 2, Expected::Exception(13)),
            ] {
                configured_operand(
                    tier,
                    x86jit_core::CpuMode::Long64,
                    code,
                    Reg::Rbx,
                    address,
                    expected,
                    false,
                    routed,
                    Setup {
                        add_before: true,
                        ..Default::default()
                    },
                );
            }
        }
    }
}

fn legacy_m128_types(tier: usize) {
    // SDM Vol 2A §2.5, Tables 2-15/2-18/2-19/2-21: Types 1, 2 and 4.
    for code in [
        &[0x0f, 0x28, 0x03][..],            // Type 1: MOVAPS.
        &[0x0f, 0x51, 0x03],                // Type 2: SQRTPS.
        &[0x66, 0x0f, 0x38, 0x17, 0x03],    // Type 4: PTEST.
        &[0x66, 0x0f, 0x3a, 0x0e, 0x03, 0], // PBLENDW.
        &[0x66, 0x0f, 0x3a, 0x42, 0x03, 0], // MPSADBW.
        &[0xf3, 0x0f, 0x70, 0x03, 0],       // PSHUFHW.
        &[0x66, 0x0f, 0xf3, 0x03],          // PSLLQ.
        &[0x66, 0x0f, 0xf5, 0x03],          // PMADDWD.
        &[0x66, 0x0f, 0x67, 0x03],          // PACKUSWB.
    ] {
        check(tier, code, Reg::Rbx, DATA + 1, Some(13));
    }
}

tier_witnesses!(helper_operand_widths, helper_widths);
tier_witnesses!(helper_stack_segments, helper_stack);
tier_witnesses!(masked_address_suppression, mask_suppression);
tier_witnesses!(embedded_broadcast_carriers, embedded_broadcast);
tier_witnesses!(legacy_m128_exception_types, legacy_m128_types);

// SDM Vol 3A §7.15, Event 13: prior instructions complete, but the faulting
// instruction retains its entry state. Exercise different flag recipes/RIP owners
// entering the same JIT alignment tail, with a later flag writer in the block.
#[test]
fn alignment_tail_preserves_each_instruction_entry() {
    for tier in 0..3 {
        for failing in 0..3 {
            let mut vm = Vm::with_backend(VmConfig::flat(0x8000), backend(tier));
            vm.set_tier_up_after(None);
            vm.map(0, 0x8000, Prot::RWX, RegionKind::Ram).unwrap();
            let mut bytes = if tier == 2 {
                vec![
                    0x41, 0xb8, 2, 0, 0, 0, 0xeb, 0, 0x41, 0xff, 0xc8, 0x75, 0xfb,
                ]
            } else {
                vec![]
            };
            let mut rips = vec![];
            for movaps in [
                &[0x0f, 0x28, 0x03][..],
                &[0x0f, 0x28, 0x45, 0],
                &[0x0f, 0x28, 0x04, 0x24],
            ] {
                bytes.extend_from_slice(&[0x48, 0x83, 0xc0, 1]); // ADD RAX,1.
                rips.push(CODE + bytes.len() as u64);
                bytes.extend_from_slice(movaps);
            }
            bytes.extend_from_slice(&[0x31, 0xc0, 0xf4]);
            vm.write_bytes(CODE, &bytes).unwrap();
            vm.write_bytes(DATA, &[0x35; 16]).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            for (i, reg) in [Reg::Rbx, Reg::Rbp, Reg::Rsp].into_iter().enumerate() {
                cpu.set_reg(reg, DATA + u64::from(i == failing));
            }
            let exit = cpu.run(&vm, Some(100));
            assert!(
                matches!(exit, Exit::Exception { addr, vector: 13, .. } if addr == rips[failing])
            );
            assert_eq!(cpu.reg(Reg::Rip), rips[failing]);
            assert_eq!(cpu.reg(Reg::Rax), failing as u64 + 1);
            assert_eq!(
                cpu.cpu.flags.to_rflags() & 0x8d5,
                if failing == 2 { 4 } else { 0 }
            );
            assert_eq!(
                cpu.xmm(0),
                if failing == 0 {
                    0
                } else {
                    u128::from_le_bytes([0x35; 16])
                }
            );
            if tier == 2 {
                assert!(vm.cache.regions() > 0);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod native {
    use super::*;
    use x86jit_tests::compare::compare;
    use x86jit_tests::native::{run_native, run_native_fault, NativeFault};
    use x86jit_tests::oracle::{run_with_backend, VectorInput};
    use x86jit_tests::vector::{CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec};

    fn witness(code: &[u8], base: usize, address: u64, vector: Option<u8>) {
        const ENTRY: u64 = 0x21_0000;
        const MEMORY: u64 = 0x22_0000;
        let mut bytes = code.to_vec();
        bytes.push(0xf4);
        let mut cpu_init = CpuSnapshot::default();
        cpu_init.gpr[4] = MEMORY + 0x800;
        cpu_init.gpr[base] = address;
        cpu_init.xmm[0] = 0x1234;
        let mut memory = vec![0x35; 4096];
        if code.ends_with(&[0x0f, 0xae, 0x0b]) {
            // SDM Vol 2A, FXRSTOR, Description and #GP: a valid image keeps
            // reserved MXCSR bits clear. Put it at the actual operand address.
            let start = (address - MEMORY) as usize;
            memory[start..start + 512].fill(0);
            memory[start..start + 2].copy_from_slice(&0x037fu16.to_le_bytes());
            memory[start + 24..start + 28].copy_from_slice(&0x1f80u32.to_le_bytes());
        }
        let input = VectorInput {
            cpu_init,
            mem_init: vec![
                MemChunk {
                    addr: ENTRY,
                    bytes,
                    kind: MemKind::Ram,
                },
                MemChunk {
                    addr: MEMORY,
                    bytes: memory,
                    kind: MemKind::Ram,
                },
            ],
            entry: ENTRY,
            run: RunSpec::UntilExit,
        };
        // ASSUMED: Linux maps #GP to SIGSEGV/SI_KERNEL and #SS to
        // SIGBUS/SI_KERNEL (BUS_ADRERR is also accepted). The pinned Linux uapi
        // siginfo.h defines the codes, not this mapping. REG_TRAPNO and ERR=0
        // are authoritative for the captured exception.
        let native = match vector {
            Some(12) => run_native_fault(&input, NativeFault::StackFault),
            Some(13) => run_native_fault(&input, NativeFault::GeneralProtection),
            None => run_native(&input),
            _ => unreachable!(),
        }
        .expect("native CPU must capture the designated fault or complete");
        if let Some(vector) = vector {
            assert!(
                matches!(native.exit, ExitKind::Exception { addr: ENTRY, vector: actual } if actual == vector)
            );
            assert_eq!(native.cpu.gpr, input.cpu_init.gpr);
            assert_eq!(native.cpu.xmm, input.cpu_init.xmm);
            assert_eq!(native.cpu.flags, input.cpu_init.flags);
            assert_eq!(native.mem[1].bytes, input.mem_init[1].bytes);
        }
        for tier in 0..3 {
            let mut actual = run_with_backend(&input, backend(tier));
            let mut expected = native.clone();
            if vector.is_none() && code.ends_with(&[0x0f, 0xae, 0x03]) {
                // Aligned save controls witness completion, not the host-specific
                // MXCSR_MASK or reserved bytes of the FXSAVE image (SDM Vol 1 §10.5.1).
                assert!(matches!(expected.exit, ExitKind::Hlt));
                assert!(matches!(actual.exit, ExitKind::Hlt));
                expected.mem.clear();
                actual.mem.clear();
            }
            let diff = compare(&expected, &actual, &[]);
            assert!(diff.is_none(), "tier={tier}, code={code:02x?}: {diff:?}");
        }
    }

    #[test]
    fn noncanonical_cpu_witness() {
        // Use an address non-canonical on LA57 hosts too. The separate tier
        // witnesses above pin the engine's 48-bit boundary.
        for code in [
            &[0x48, 0x8b, 0x03][..],
            &[0x48, 0x89, 0x03],
            &[0x36, 0x48, 0x8b, 0x03],
        ] {
            witness(code, 3, BAD, Some(13));
        }
        for code in [
            &[0x50][..],
            &[0x58],
            &[0xc3],
            &[0xe8, 0, 0, 0, 0],
            &[0x48, 0x8b, 0x04, 0x24],
        ] {
            witness(code, 4, BAD + 0x100, Some(12));
        }
        witness(&[0x48, 0x8b, 0x45, 0], 5, BAD, Some(12));
    }

    #[test]
    fn alignment_cpu_witness() {
        for code in [
            &[0x0f, 0x28, 0x03][..],
            &[0x0f, 0x29, 0x03],
            &[0x66, 0x0f, 0x6f, 0x03],
            &[0x0f, 0x58, 0x03],
            &[0x0f, 0xae, 0x03],
            &[0x0f, 0xae, 0x0b],
        ] {
            witness(code, 3, 0x22_0001, Some(13));
        }
        for code in [&[0x0f, 0x10, 0x03][..], &[0xf3, 0x0f, 0x6f, 0x03]] {
            witness(code, 3, 0x22_0001, None);
        }
        // Aligned controls distinguish alignment from an invalid FXRSTOR image.
        for code in [
            &[0x0f, 0xae, 0x03][..],
            &[0x0f, 0xae, 0x0b],
            &[0x48, 0x0f, 0xae, 0x03],
            &[0x48, 0x0f, 0xae, 0x0b],
        ] {
            witness(code, 3, 0x22_0000, None);
        }
        if std::is_x86_feature_detected!("sse4.2") {
            for opcode in [0x60, 0x61] {
                witness(
                    &[0x66, 0x48, 0x0f, 0x3a, opcode, 0x03, 0],
                    3,
                    0x22_0001,
                    None,
                );
            }
        }
        if std::is_x86_feature_detected!("sse3") {
            witness(&[0xf2, 0x0f, 0xf0, 0x03], 3, 0x22_0001, None);
        }
        if std::is_x86_feature_detected!("avx") {
            witness(&[0xc5, 0xf8, 0x58, 0x03], 3, 0x22_0001, None);
            witness(&[0xc5, 0xfc, 0x28, 0x03], 3, 0x22_0010, Some(13));
        }
        if std::is_x86_feature_detected!("avx512f") {
            let mut a = CodeAssembler::new(64).unwrap();
            a.vmovaps(zmm0, zmmword_ptr(rbx)).unwrap();
            witness(&a.assemble(CODE).unwrap(), 3, 0x22_0020, Some(13));
        }
    }
}
