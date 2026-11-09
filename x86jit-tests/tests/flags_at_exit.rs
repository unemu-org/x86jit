//! SDM Vol 3A §7.5 (Exception Classifications), AMD APM Vol 2 §8.1.3:
//! a fault exposes the state preceding the faulting instruction.
//! SDM Vol 1 §3.4.3.1 and Vol 2A, ADD, Operation / Flags Affected define
//! the arithmetic flag witnesses below, independently of either execution tier.

use iced_x86::code_asm::*;
use x86jit_core::{
    Backend, CpuMode, Exit, InterpreterBackend, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;
const SPAN: u64 = 0x10000;
const ARITH: u64 = 0x8d5;

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

fn vm(code: &[u8], mode: CpuMode, tier: usize) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(SPAN), backend(tier));
    vm.set_cpu_mode(mode);
    vm.map(0, 0x8000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x9000, 0x7000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(CODE, code).unwrap();
    vm
}

fn region_entry(a: &mut CodeAssembler, tier: usize) -> Option<CodeLabel> {
    if tier != 2 {
        return None;
    }
    let mut top = a.create_label();
    let mut body = a.create_label();
    a.set_label(&mut top).unwrap();
    a.jmp(body).unwrap();
    a.nop().unwrap();
    a.set_label(&mut body).unwrap();
    Some(top)
}

fn region_backedge(a: &mut CodeAssembler, top: &mut Option<CodeLabel>) {
    if let Some(top) = top {
        a.jne(*top).unwrap();
    }
}

fn fault_code(kind: &str, region: bool) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    a.set_label(&mut top).unwrap();
    if region {
        let mut body = a.create_label();
        a.jmp(body).unwrap();
        a.nop().unwrap();
        a.set_label(&mut body).unwrap();
    }
    if region && kind == "xrstor" {
        a.test(r11, r11).unwrap();
        a.jne(top).unwrap();
    }
    if kind == "mf" {
        // SDM Vol 1 §8.6: unmasked #Z is delivered on the next waiting x87 op.
        a.fldcw(word_ptr(0x3000)).unwrap();
        a.fld1().unwrap();
        a.fldz().unwrap();
        a.fdivp(st1, st0).unwrap();
    }
    a.add(rax, rbx).unwrap();
    match kind {
        "load" | "mmio_load" => a.mov(rcx, qword_ptr(rsi)).unwrap(),
        "store" | "mmio_store" => a.mov(qword_ptr(rsi), rcx).unwrap(),
        "div" => a.div(rcx).unwrap(),
        "string" => a.movsb().unwrap(),
        "x87" => a.fld(qword_ptr(rsi)).unwrap(),
        "fxsave" => a.fxsave(ptr(rsi)).unwrap(),
        "xsave" => a.xsave(ptr(rsi)).unwrap(),
        "xrstor" => a.xrstor(ptr(rsi)).unwrap(),
        "xgetbv" => {
            a.mov(ecx, 1).unwrap();
            a.xgetbv().unwrap();
        }
        "mf" => a.fld1().unwrap(),
        _ => unreachable!(),
    }
    a.cmp(rcx, 0).unwrap();
    if region {
        a.jne(top).unwrap();
    }
    a.hlt().unwrap();
    a.assemble(CODE).unwrap()
}

fn fault_flags(kind: &str, tier: usize) {
    for (lhs, expected) in [(i64::MAX as u64, 0x894), (u64::MAX, 0x55)] {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        let expected = if matches!(kind, "load" | "store" | "div") {
            native_fault_witness(kind, lhs, expected)
        } else {
            expected
        };
        let code = fault_code(kind, tier == 2);
        let mut vm = vm(&code, CpuMode::Long64, tier);
        let mmio = kind.starts_with("mmio");
        if mmio {
            vm.map(0x8000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        }
        if kind == "mf" {
            // SDM Vol 1 §8.1.5: FNINIT's 037FH with ZM (bit 2) cleared = 037BH.
            // Vol 3A §7.15, Event 16: #MF is reported at the next waiting instruction.
            vm.write_bytes(0x3000, &0x037bu16.to_le_bytes()).unwrap();
        }
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rax, lhs);
        cpu.set_reg(Reg::Rbx, 1);
        cpu.set_reg(Reg::Rsi, if mmio { 0x8000 } else { SPAN });
        cpu.set_reg(Reg::Rdi, 0x9000);
        let exit = cpu.run(&vm, Some(1000));
        match kind {
            "xgetbv" => assert!(
                matches!(exit, Exit::Exception { vector: 13, .. }),
                "{exit:?}"
            ),
            "mf" => assert!(
                matches!(exit, Exit::Exception { vector: 16, .. }),
                "{exit:?}"
            ),
            "div" => assert!(
                matches!(exit, Exit::Exception { vector: 0, .. }),
                "{exit:?}"
            ),
            "mmio_load" => assert!(matches!(exit, Exit::MmioRead { .. }), "{exit:?}"),
            "mmio_store" => assert!(matches!(exit, Exit::MmioWrite { .. }), "{exit:?}"),
            _ => assert!(matches!(exit, Exit::UnmappedMemory { .. }), "{exit:?}"),
        }
        if tier == 2 {
            assert!(vm.cache.regions() > 0, "must exercise a compiled region");
        }
        assert_eq!(
            cpu.flags().to_rflags() & ARITH,
            expected,
            "{kind}, tier {tier}, lhs {lhs:#x}"
        );
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn native_fault_witness(kind: &str, lhs: u64, expected: u64) -> u64 {
    use x86jit_tests::native::{run_native, run_native_fault, NativeFault};
    use x86jit_tests::oracle::VectorInput;
    use x86jit_tests::vector::{Access, CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec};

    const NATIVE_CODE: u64 = 0x210000;
    const UNMAPPED: u64 = 0x4000000000;
    let mut initial = CpuSnapshot {
        gpr: std::array::from_fn(|i| 0x100 + i as u64),
        ..CpuSnapshot::default()
    };
    initial.gpr[0] = lhs;
    initial.gpr[1] = 0;
    initial.gpr[2] = 0;
    initial.gpr[3] = 1;
    initial.gpr[6] = UNMAPPED;
    let input = VectorInput {
        cpu_init: initial.clone(),
        mem_init: vec![MemChunk {
            addr: NATIVE_CODE,
            bytes: fault_code(kind, false),
            kind: MemKind::Ram,
        }],
        entry: NATIVE_CODE,
        run: RunSpec::UntilExit,
    };
    let fault = if kind == "div" {
        NativeFault::DivideError
    } else {
        NativeFault::PageFault { addr: UNMAPPED }
    };
    let out = run_native_fault(&input, fault).expect("native designated fault capture");
    // Assert the independently derived witness before using it as an oracle.
    assert_eq!(out.cpu.flags.to_rflags() & ARITH, expected);
    let mut gpr = initial.gpr;
    gpr[0] = lhs.wrapping_add(1);
    assert_eq!(out.cpu.gpr, gpr, "all GPRs captured at the fault");
    assert_eq!(out.cpu.rip, NATIVE_CODE + 3);
    let exit = if kind == "div" {
        ExitKind::Exception {
            addr: NATIVE_CODE + 3,
            vector: 0,
        }
    } else {
        ExitKind::UnmappedMemory {
            addr: UNMAPPED,
            access: if kind == "load" {
                Access::Read
            } else {
                Access::Write
            },
        }
    };
    assert_eq!(out.exit, exit);
    assert!(
        run_native(&input).is_none(),
        "ordinary oracle still rejects guest faults"
    );
    out.cpu.flags.to_rflags() & ARITH
}

macro_rules! faults {
    ($($name:ident: $kind:literal, $tier:literal;)*) => {$(
        #[test]
        fn $name() { fault_flags($kind, $tier); }
    )*};
}

faults! {
    flags_fault_load_interp: "load", 0;
    flags_fault_load_jit: "load", 1;
    flags_fault_load_region: "load", 2;
    flags_fault_store_interp: "store", 0;
    flags_fault_store_jit: "store", 1;
    flags_fault_store_region: "store", 2;
    flags_fault_div_interp: "div", 0;
    flags_fault_div_jit: "div", 1;
    flags_fault_div_region: "div", 2;
    flags_fault_mmio_load_interp: "mmio_load", 0;
    flags_fault_mmio_load_jit: "mmio_load", 1;
    flags_fault_mmio_load_region: "mmio_load", 2;
    flags_fault_mmio_store_interp: "mmio_store", 0;
    flags_fault_mmio_store_jit: "mmio_store", 1;
    flags_fault_mmio_store_region: "mmio_store", 2;
    flags_fault_string_interp: "string", 0;
    flags_fault_string_jit: "string", 1;
    flags_fault_string_region: "string", 2;
    flags_fault_x87_interp: "x87", 0;
    flags_fault_x87_jit: "x87", 1;
    flags_fault_x87_region: "x87", 2;
    flags_fault_fxsave_interp: "fxsave", 0;
    flags_fault_fxsave_jit: "fxsave", 1;
    flags_fault_fxsave_region: "fxsave", 2;
    flags_fault_xsave_interp: "xsave", 0;
    flags_fault_xsave_jit: "xsave", 1;
    flags_fault_xsave_region: "xsave", 2;
    flags_fault_xrstor_interp: "xrstor", 0;
    flags_fault_xrstor_jit: "xrstor", 1;
    flags_fault_xrstor_region: "xrstor", 2;
    flags_fault_xgetbv_interp: "xgetbv", 0;
    flags_fault_xgetbv_jit: "xgetbv", 1;
    flags_fault_xgetbv_region: "xgetbv", 2;
    flags_fault_mf_interp: "mf", 0;
    flags_fault_mf_jit: "mf", 1;
    flags_fault_mf_region: "mf", 2;
}

fn lahf(mode: CpuMode, tier: usize) {
    // SDM Vol 2A, LAHF, Operation: AH = SF:ZF:0:AF:0:PF:1:CF.
    let mut a = CodeAssembler::new(mode.bits()).unwrap();
    a.add(eax, ebx).unwrap();
    a.lahf().unwrap();
    a.cmp(ecx, 0).unwrap();
    a.hlt().unwrap();
    let code = a.assemble(CODE).unwrap();
    let expected = 0x96;
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    let expected = if mode == CpuMode::Long64 {
        use x86jit_tests::native::run_native;
        use x86jit_tests::oracle::VectorInput;
        use x86jit_tests::vector::{CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec};
        const NATIVE_CODE: u64 = 0x210000;
        let mut initial = CpuSnapshot::default();
        initial.gpr[0] = 0x7fff_ffff;
        initial.gpr[3] = 1;
        let out = run_native(&VectorInput {
            cpu_init: initial,
            mem_init: vec![MemChunk {
                addr: NATIVE_CODE,
                bytes: a.assemble(NATIVE_CODE).unwrap(),
                kind: MemKind::Ram,
            }],
            entry: NATIVE_CODE,
            run: RunSpec::UntilExit,
        })
        .expect("native LAHF witness");
        assert!(matches!(out.exit, ExitKind::Hlt));
        let captured_ah = (out.cpu.gpr[0] >> 8) & 0xff;
        assert_eq!(captured_ah, expected);
        captured_ah
    } else {
        expected
    };
    let vm = vm(&code, mode, tier);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, 0x7fff_ffff);
    cpu.set_reg(Reg::Rbx, 1);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(
        (cpu.reg(Reg::Rax) >> 8) & 0xff,
        expected,
        "{mode:?}, {tier}"
    );
}

#[test]
fn flags_read_lahf_long64_interp() {
    lahf(CpuMode::Long64, 0);
}
#[test]
fn flags_read_lahf_long64_jit() {
    lahf(CpuMode::Long64, 1);
}
#[test]
fn flags_read_lahf_compat32_interp() {
    lahf(CpuMode::Compat32, 0);
}
#[test]
fn flags_read_lahf_compat32_jit() {
    lahf(CpuMode::Compat32, 1);
}

// Real16 is interpreter-only: these witnesses do not exercise JIT op_reads arms.
#[test]
fn flags_read_real16_daa() {
    // SDM Vol 2A, DAA, Operation: 0x0f + 1 sets AF; adjustment makes AL=0x16.
    real16_adjust(0x27, 0x16);
}

#[test]
fn flags_read_real16_das() {
    // SDM Vol 2A, DAS, Operation: AF=1 causes AL=0x10-6.
    real16_adjust(0x2f, 0x0a);
}

#[test]
fn flags_read_real16_aaa() {
    // SDM Vol 2A, AAA, Operation: AF=1 causes AX+=0x106, then AL&=0xf.
    real16_adjust(0x37, 0x106);
}

#[test]
fn flags_read_real16_aas() {
    // SDM Vol 2A, AAS, Operation: AF=1 causes AX-=6; AH-=1; AL&=0xf.
    real16_adjust(0x3f, 0xff0a);
}

fn real16_adjust(opcode: u8, expected: u64) {
    let vm = vm(&[0x00, 0xd8, opcode, 0x39, 0xc9, 0xf4], CpuMode::Real16, 0);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, 0xf);
    cpu.set_reg(Reg::Rbx, 1);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rax) & 0xffff, expected);
}

#[test]
fn flags_read_real16_pushf() {
    // SDM Vol 2B, PUSHF/PUSHFD/PUSHFQ, Operation; ADD 0xffff+1 sets CF/PF/AF/ZF.
    let vm = vm(&[0x01, 0xd8, 0x9c, 0x39, 0xc9, 0xf4], CpuMode::Real16, 0);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, 0xffff);
    cpu.set_reg(Reg::Rbx, 1);
    cpu.set_reg(Reg::Rsp, 0x8000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(vm.mem.read(0x7ffe, 2).unwrap() & ARITH, 0x55);
}

#[test]
fn flags_read_real16_cmc() {
    // SDM Vol 2A, CMC, Operation and LAHF, Operation: complement ADD's CF=1;
    // LAHF then captures ZF/PF/AF with CF=0 and reserved bit 1 set.
    let vm = vm(
        &[0x01, 0xd8, 0xf5, 0x9f, 0x39, 0xc9, 0xf4],
        CpuMode::Real16,
        0,
    );
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, 0xffff);
    cpu.set_reg(Reg::Rbx, 1);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!((cpu.reg(Reg::Rax) >> 8) & 0xff, 0x56);
}

#[test]
fn flags_fault_partial_writers() {
    // SDM Vol 2A, INC, Flags Affected: INC preserves ADD's CF=1 while
    // setting OF/SF/AF/PF for 0x7fff_ffff_ffff_ffff+1.
    for tier in 0..3 {
        let mut a = CodeAssembler::new(64).unwrap();
        let mut top = region_entry(&mut a, tier);
        a.add(rax, rbx).unwrap();
        a.inc(rdi).unwrap();
        a.mov(rcx, qword_ptr(rsi)).unwrap();
        a.cmp(rcx, 0).unwrap();
        region_backedge(&mut a, &mut top);
        a.hlt().unwrap();
        let vm = vm(&a.assemble(CODE).unwrap(), CpuMode::Long64, tier);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rax, u64::MAX);
        cpu.set_reg(Reg::Rbx, 1);
        cpu.set_reg(Reg::Rdi, i64::MAX as u64);
        cpu.set_reg(Reg::Rsi, SPAN);
        assert!(matches!(cpu.run(&vm, None), Exit::UnmappedMemory { .. }));
        if tier == 2 {
            assert!(vm.cache.regions() > 0);
        }
        assert_eq!(cpu.flags().to_rflags() & ARITH, 0x895, "tier {tier}");
    }
}

fn writer_fault(kind: &str) {
    // SDM Vol 2A, ADC / AND and Vol 2B, MUL / SBB, Operation / Flags Affected.
    // AND's AF and MUL's PF/AF/ZF/SF are undefined and excluded from comparison.
    let (lhs, operand, compared, expected) = match kind {
        "adc" => (u64::MAX, i64::MAX as u64, ARITH, 0x894),
        "sbb" => (u64::MAX, 0, ARITH, 0x95),
        "logic" => (u64::MAX, 1 << 63, ARITH & !0x10, 0x84),
        "mul" => (i64::MAX as u64, 3, 0x801, 0x801),
        _ => unreachable!(),
    };
    for tier in 0..3 {
        let mut a = CodeAssembler::new(64).unwrap();
        let mut top = region_entry(&mut a, tier);
        a.add(rax, rbx).unwrap();
        match kind {
            "adc" => a.adc(rdi, 0).unwrap(),
            "sbb" => a.sbb(rdi, 0).unwrap(),
            "logic" => a.and(rdi, rdi).unwrap(),
            "mul" => a.mul(rdi).unwrap(),
            _ => unreachable!(),
        }
        a.mov(rcx, qword_ptr(rsi)).unwrap();
        a.cmp(rcx, 0).unwrap();
        region_backedge(&mut a, &mut top);
        a.hlt().unwrap();
        let vm = vm(&a.assemble(CODE).unwrap(), CpuMode::Long64, tier);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rax, lhs);
        cpu.set_reg(Reg::Rbx, 1);
        cpu.set_reg(Reg::Rdi, operand);
        cpu.set_reg(Reg::Rsi, SPAN);
        assert!(matches!(cpu.run(&vm, None), Exit::UnmappedMemory { .. }));
        if tier == 2 {
            assert!(vm.cache.regions() > 0);
        }
        assert_eq!(
            cpu.flags().to_rflags() & compared,
            expected,
            "{kind}, tier {tier}"
        );
    }
}

#[test]
fn flags_fault_adc_recipe() {
    writer_fault("adc");
}

#[test]
fn flags_fault_sbb_recipe() {
    writer_fault("sbb");
}

#[test]
fn flags_fault_logic_recipe() {
    writer_fault("logic");
}

#[test]
fn flags_fault_mul_recipe() {
    writer_fault("mul");
}

fn helper_flags(kind: &str) {
    // SDM Vol 2A, FCOMI/FCOMIP/FUCOMI/FUCOMIP, Operation: equal sets ZF,
    // clears CF/PF/OF/SF/AF. Vol 2A, CMPS, and Vol 2B, SCAS, Operation:
    // subtracting equal bytes sets PF/ZF and clears CF/AF/SF/OF.
    for tier in 0..3 {
        let mut a = CodeAssembler::new(64).unwrap();
        let mut top = region_entry(&mut a, tier);
        if matches!(kind, "fcomi" | "fucomi") {
            a.fld1().unwrap();
            a.fld1().unwrap();
        }
        a.add(rax, rbx).unwrap();
        match kind {
            "fcomi" => a.fcomi(st0, st1).unwrap(),
            "fucomi" => a.fucomi(st0, st1).unwrap(),
            "cmps" => a.cmpsb().unwrap(),
            "scas" => a.scasb().unwrap(),
            "partial_rep_cmps" => a.repe().cmpsb().unwrap(),
            _ => unreachable!(),
        }
        a.mov(rdx, qword_ptr(r8)).unwrap();
        a.cmp(rdx, 0).unwrap();
        region_backedge(&mut a, &mut top);
        a.hlt().unwrap();
        let vm = vm(&a.assemble(CODE).unwrap(), CpuMode::Long64, tier);
        vm.write_bytes(0x3000, &[0, 0]).unwrap();
        vm.write_bytes(0x4000, &[0, 0]).unwrap();
        vm.write_bytes(SPAN - 1, &[0]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rax, u64::MAX);
        cpu.set_reg(Reg::Rbx, 1);
        cpu.set_reg(Reg::Rcx, 2);
        let partial = kind == "partial_rep_cmps";
        cpu.set_reg(Reg::Rsi, if partial { SPAN - 1 } else { 0x3000 });
        cpu.set_reg(Reg::Rdi, 0x4000);
        cpu.set_reg(Reg::R8, SPAN);
        assert!(matches!(cpu.run(&vm, None), Exit::UnmappedMemory { .. }));
        if tier == 2 {
            assert!(vm.cache.regions() > 0);
        }
        if partial {
            assert_eq!(cpu.reg(Reg::Rcx), 1, "one comparison retired");
            assert_eq!(cpu.reg(Reg::Rsi), SPAN);
            assert_eq!(cpu.reg(Reg::Rdi), 0x4001);
        }
        let expected = if matches!(kind, "fcomi" | "fucomi") {
            0x40
        } else {
            0x44
        };
        assert_eq!(
            cpu.flags().to_rflags() & ARITH,
            expected,
            "{kind}, tier {tier}"
        );
    }
}

#[test]
fn flags_helper_fcomi_barrier() {
    helper_flags("fcomi");
}
#[test]
fn flags_helper_fucomi_barrier() {
    helper_flags("fucomi");
}
#[test]
fn flags_helper_cmps_barrier() {
    helper_flags("cmps");
}
#[test]
fn flags_helper_scas_barrier() {
    helper_flags("scas");
}
#[test]
fn flags_helper_partial_rep_cmps_barrier() {
    helper_flags("partial_rep_cmps");
}

fn shift(a: &mut CodeAssembler, kind: &str, count: Option<u32>) {
    match (kind, count) {
        ("shl", Some(n)) => a.shl(rdi, n).unwrap(),
        ("shr", Some(n)) => a.shr(rdi, n).unwrap(),
        ("sar", Some(n)) => a.sar(rdi, n).unwrap(),
        ("rol", Some(n)) => a.rol(rdi, n).unwrap(),
        ("ror", Some(n)) => a.ror(rdi, n).unwrap(),
        ("rcl", Some(n)) => a.rcl(rdi, n).unwrap(),
        ("rcr", Some(n)) => a.rcr(rdi, n).unwrap(),
        ("shld", Some(n)) => a.shld(rdi, rbx, n).unwrap(),
        ("shrd", Some(n)) => a.shrd(rdi, rbx, n).unwrap(),
        ("shl", None) => a.shl(rdi, cl).unwrap(),
        ("shr", None) => a.shr(rdi, cl).unwrap(),
        ("sar", None) => a.sar(rdi, cl).unwrap(),
        ("rol", None) => a.rol(rdi, cl).unwrap(),
        ("ror", None) => a.ror(rdi, cl).unwrap(),
        ("rcl", None) => a.rcl(rdi, cl).unwrap(),
        ("rcr", None) => a.rcr(rdi, cl).unwrap(),
        ("shld", None) => a.shld(rdi, rbx, cl).unwrap(),
        ("shrd", None) => a.shrd(rdi, rbx, cl).unwrap(),
        _ => unreachable!(),
    }
}

fn shift_fault(count: Option<u32>, runtime_count: u64) {
    // SDM Vol 2B, SAL/SAR/SHL/SHR, SHLD, SHRD and RCL/RCR/ROL/ROR,
    // Operation / Flags Affected: count 0 preserves flags; count 1 defines OF.
    for kind in [
        "shl", "shr", "sar", "rol", "ror", "rcl", "rcr", "shld", "shrd",
    ] {
        let (compared, shifted) = match kind {
            "shl" | "shld" => (ARITH & !0x10, 0x801),
            "shr" => (ARITH & !0x10, 0x805),
            "sar" => (ARITH & !0x10, 0x85),
            "rol" | "rcl" => (ARITH, 0x895),
            "ror" => (ARITH, 0x95),
            "rcr" => (ARITH, 0x895),
            "shrd" => (ARITH & !0x10, 0x85),
            _ => unreachable!(),
        };
        for tier in 0..3 {
            let mut a = CodeAssembler::new(64).unwrap();
            let mut top = region_entry(&mut a, tier);
            a.add(rax, rbx).unwrap();
            shift(&mut a, kind, count);
            a.mov(rdx, qword_ptr(rsi)).unwrap();
            a.cmp(rdx, 0).unwrap();
            region_backedge(&mut a, &mut top);
            a.hlt().unwrap();
            let vm = vm(&a.assemble(CODE).unwrap(), CpuMode::Long64, tier);
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rax, i64::MAX as u64);
            cpu.set_reg(Reg::Rbx, 1);
            cpu.set_reg(Reg::Rdi, 0x8000_0000_0000_0001);
            cpu.set_reg(Reg::Rcx, runtime_count);
            cpu.set_reg(Reg::Rsi, SPAN);
            assert!(matches!(cpu.run(&vm, None), Exit::UnmappedMemory { .. }));
            if tier == 2 {
                assert!(vm.cache.regions() > 0);
            }
            let expected = if count == Some(1) || (count.is_none() && runtime_count == 1) {
                shifted
            } else {
                0x894
            };
            assert_eq!(
                cpu.flags().to_rflags() & compared,
                expected & compared,
                "{kind}, count {count:?}, tier {tier}"
            );
        }
    }
}

#[test]
fn flags_shift_immediate_fault() {
    shift_fault(Some(1), 0);
}
#[test]
fn flags_shift_masked_zero_fault() {
    shift_fault(Some(64), 0);
}
#[test]
fn flags_shift_cl_zero_fault() {
    shift_fault(None, 0);
}

#[test]
fn flags_shift_cl_nonzero_fault() {
    shift_fault(None, 1);
}

#[test]
fn flags_shift_immediate_narrows() {
    use x86jit_core::ir::{FlagMask, IrOp};
    use x86jit_core::lift::{elide_dead_flags, lift_block, FetchAddr};
    for kind in [
        "shl", "shr", "sar", "rol", "ror", "rcl", "rcr", "shld", "shrd",
    ] {
        for count in [Some(1), Some(64), None] {
            let mut a = CodeAssembler::new(64).unwrap();
            a.add(rax, rbx).unwrap();
            shift(&mut a, kind, count);
            a.cmp(rdx, 0).unwrap();
            a.hlt().unwrap();
            let vm = vm(&a.assemble(CODE).unwrap(), CpuMode::Long64, 0);
            let mut ir = lift_block(&vm.mem, FetchAddr::flat(CODE), CpuMode::Long64).unwrap();
            let original = elide_dead_flags(&mut ir.ops);
            for (op, full) in ir.ops.iter_mut().zip(original) {
                let mask = match op {
                    IrOp::Shl { set_flags, .. }
                    | IrOp::Shr { set_flags, .. }
                    | IrOp::Sar { set_flags, .. }
                    | IrOp::Rol { set_flags, .. }
                    | IrOp::Ror { set_flags, .. }
                    | IrOp::Rcl { set_flags, .. }
                    | IrOp::Rcr { set_flags, .. }
                    | IrOp::DoubleShift { set_flags, .. } => *set_flags,
                    _ => continue,
                };
                assert_eq!(
                    mask,
                    if count == Some(1) {
                        FlagMask::NONE
                    } else {
                        full
                    },
                    "{kind}, {count:?}"
                );
            }
        }
    }
}

#[test]
fn flags_recovery_multiple_fault_sites() {
    // SDM Vol 2A, ADD, Flags Affected: all exits observe the same retired ADD.
    // Each load has a distinct effective address so its bounds check is retained.
    for tier in 0..3 {
        for fault in 0..=8 {
            let mut a = CodeAssembler::new(64).unwrap();
            let mut top = region_entry(&mut a, tier);
            a.add(rax, rbx).unwrap();
            for i in 0..8 {
                a.mov(rdx, qword_ptr(rsi + i * 8)).unwrap();
                a.mov(r12, i as u64 + 1).unwrap();
            }
            a.div(r9).unwrap();
            a.cmp(rdx, 0).unwrap();
            region_backedge(&mut a, &mut top);
            a.hlt().unwrap();
            let code = a.assemble(CODE).unwrap();
            let fault_rip =
                iced_x86::Decoder::with_ip(64, &code, CODE, iced_x86::DecoderOptions::NONE)
                    .into_iter()
                    .filter(|i| {
                        i.mnemonic() == iced_x86::Mnemonic::Div
                            || (i.mnemonic() == iced_x86::Mnemonic::Mov
                                && i.op1_kind() == iced_x86::OpKind::Memory)
                    })
                    .nth(fault as usize)
                    .unwrap()
                    .ip();
            let vm = vm(&code, CpuMode::Long64, tier);
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rax, i64::MAX as u64);
            cpu.set_reg(Reg::Rbx, 1);
            cpu.set_reg(Reg::Rsi, SPAN - fault * 8);
            cpu.set_reg(Reg::R12, 0x1234);
            let exit = cpu.run(&vm, None);
            if fault == 8 {
                // SDM Vol 2A, DIV, Exceptions: a zero divisor raises #DE.
                assert!(matches!(exit, Exit::Exception { vector: 0, .. }));
            } else {
                assert!(matches!(exit, Exit::UnmappedMemory { addr: SPAN, .. }));
            }
            if tier == 2 {
                assert!(vm.cache.regions() > 0);
            }
            assert_eq!(cpu.reg(Reg::Rax), 1 << 63);
            assert_eq!(cpu.reg(Reg::R12), if fault == 0 { 0x1234 } else { fault });
            assert_eq!(cpu.reg(Reg::Rip), fault_rip);
            assert_eq!(
                cpu.flags().to_rflags() & ARITH,
                0x894,
                "tier {tier}, fault {fault}"
            );
        }
    }
}

#[test]
fn flags_recovery_distinct_epochs() {
    // SDM Vol 2A, ADD and Vol 2B, SUB, Operation / Flags Affected:
    // the first fault observes 0x7fff_ffff_ffff_ffff+1; the second observes 0-1.
    for tier in 0..3 {
        for second in [false, true] {
            let mut a = CodeAssembler::new(64).unwrap();
            let mut top = region_entry(&mut a, tier);
            a.add(rax, rbx).unwrap();
            a.mov(rdx, qword_ptr(rsi)).unwrap();
            a.sub(rdi, rbx).unwrap();
            a.mov(rdx, qword_ptr(r8)).unwrap();
            a.cmp(rdx, 0).unwrap();
            region_backedge(&mut a, &mut top);
            a.hlt().unwrap();
            let vm = vm(&a.assemble(CODE).unwrap(), CpuMode::Long64, tier);
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rax, i64::MAX as u64);
            cpu.set_reg(Reg::Rbx, 1);
            cpu.set_reg(Reg::Rsi, if second { 0x3000 } else { SPAN });
            cpu.set_reg(Reg::R8, SPAN);
            assert!(matches!(cpu.run(&vm, None), Exit::UnmappedMemory { .. }));
            if tier == 2 {
                assert!(vm.cache.regions() > 0);
            }
            assert_eq!(
                cpu.flags().to_rflags() & ARITH,
                if second { 0x95 } else { 0x894 },
                "tier {tier}, second {second}"
            );
            assert_eq!(cpu.reg(Reg::Rax), 1 << 63);
            assert_eq!(cpu.reg(Reg::Rdi), if second { u64::MAX } else { 0 });
        }
    }
}

#[test]
fn flags_recovery_helper_and_explicit_tails() {
    // SDM Vol 2A, ADD, Flags Affected; Vol 2A, FXSAVE, Operation:
    // both the load fault and the FXSAVE helper fault preserve the retired ADD.
    for tier in 0..3 {
        for helper_fault in [false, true] {
            let mut a = CodeAssembler::new(64).unwrap();
            let mut top = region_entry(&mut a, tier);
            a.add(rax, rbx).unwrap();
            a.mov(rdx, qword_ptr(rsi)).unwrap();
            a.fxsave(ptr(r8)).unwrap();
            a.cmp(rdx, 0).unwrap();
            region_backedge(&mut a, &mut top);
            a.hlt().unwrap();
            let vm = vm(&a.assemble(CODE).unwrap(), CpuMode::Long64, tier);
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rax, i64::MAX as u64);
            cpu.set_reg(Reg::Rbx, 1);
            cpu.set_reg(Reg::Rsi, if helper_fault { 0x3000 } else { SPAN });
            cpu.set_reg(Reg::R8, SPAN);
            assert!(matches!(
                cpu.run(&vm, None),
                Exit::UnmappedMemory { addr: SPAN, .. }
            ));
            if tier == 2 {
                assert!(vm.cache.regions() > 0);
            }
            assert_eq!(cpu.reg(Reg::Rax), 1 << 63);
            assert_eq!(
                cpu.flags().to_rflags() & ARITH,
                0x894,
                "tier {tier}, helper fault {helper_fault}"
            );
        }
    }
}
