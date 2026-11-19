//! Application Real16 versus legacy real execution, using the coverage probe corpus.
#[path = "../support/model_api.rs"]
mod model_api;
use iced_x86::{Code, CpuidFeature as F};
use x86jit_core::{
    Backend, CachedBlock, CpuGeneration, CpuMode, CpuState, Exit, IrBlock, LegacyCpuModel,
    MemConsistency, Prot, RegionKind, StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

struct Interpreter;
impl Backend for Interpreter {
    fn materialize(
        &self,
        _: &IrBlock,
        _: MemConsistency,
        _: Option<(u64, u64)>,
        _: u64,
    ) -> CachedBlock {
        panic!("Real16 must use the interpreter")
    }
}
fn available(code: Code, generation: CpuGeneration, bytes: &[u8]) -> bool {
    if bytes.len() > 1 && bytes[0] == 0x9b {
        return available(code, generation, &bytes[1..]);
    }
    if generation == CpuGeneration::I286
        && bytes
            .iter()
            .take_while(|b| {
                matches!(
                    b,
                    0x26 | 0x2e | 0x36 | 0x3e | 0x64..=0x67 | 0xf0 | 0xf2 | 0xf3
                )
            })
            .any(|b| matches!(b, 0x64..=0x67))
    {
        return false;
    }
    code.cpuid_features().iter().all(|f| {
        matches!(
            f,
            F::INTEL8086
                | F::INTEL8086_ONLY
                | F::INTEL186
                | F::INTEL286
                | F::INTEL286_ONLY
                | F::FPU
                | F::FPU287
        ) || generation == CpuGeneration::I386
            && matches!(
                f,
                F::INTEL386 | F::INTEL386_ONLY | F::INTEL386_A0_ONLY | F::FPU387
            )
    })
}
fn random(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}
fn setup(bytes: &[u8], generation: Option<CpuGeneration>, seed: u64) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(Interpreter));
    for page in (0..0x20000).step_by(0x1000) {
        vm.map(page, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    }
    vm.set_cpu_mode(CpuMode::Real16);
    if let Some(generation) = generation {
        model_api::select(
            &mut vm,
            LegacyCpuModel {
                generation,
                model_semantics_id: generation as u64,
            },
        );
    }
    let mut rng = seed;
    let mut memory = vec![0; 0x20000];
    for b in &mut memory {
        *b = random(&mut rng) as u8;
    }
    // Operands and implicit stack frames cannot overwrite the probe at 1000h.
    memory[0x1000..0x1000 + bytes.len()].copy_from_slice(bytes);
    // Fault/INT delivery is compared through the same IVT handler below.
    for vector in 0..256 {
        let handler = 0x1800 + vector * 4;
        memory[vector * 4..vector * 4 + 2].copy_from_slice(&(handler as u16).to_le_bytes());
        memory[vector * 4 + 2..vector * 4 + 4].fill(0);
        memory[handler] = 0xf4;
    }
    for (i, page) in memory.chunks(0x1000).enumerate() {
        vm.write_bytes((i * 0x1000) as u64, page).unwrap();
    }
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    for reg in &mut cpu.cpu.gpr {
        *reg = (random(&mut rng) & 0xffff_ffff_0000_0000) | (0x4000 + (random(&mut rng) & 0x7ff));
    }
    let decoded = iced_x86::Decoder::with_ip(
        16,
        bytes,
        0x1000,
        iced_x86::DecoderOptions::NO_INVALID_CHECK,
    )
    .decode();
    if decoded.op_kinds().any(|kind| {
        matches!(
            kind,
            iced_x86::OpKind::MemorySegSI
                | iced_x86::OpKind::MemorySegESI
                | iced_x86::OpKind::MemoryESDI
                | iced_x86::OpKind::MemoryESEDI
        )
    }) {
        cpu.cpu.gpr[1] = (cpu.cpu.gpr[1] & !0xffff_ffff) | 2;
    }
    cpu.cpu.gpr[4] = 0x8000;
    cpu.cpu
        .flags
        .set_flags16((random(&mut rng) as u16 & 0x0cd5) | 2);
    for (i, reg) in cpu.cpu.fpr.iter_mut().enumerate() {
        *reg = x86jit_core::f80::F80::from_f64(((i + 1) as f64).to_bits()).to_bytes();
    }
    cpu.cpu.fpu_empty = 0;
    cpu.cpu.fpu_top = 0;
    (vm, cpu)
}
fn image(cpu: &Vcpu) -> String {
    state_image(&cpu.cpu)
}
fn state_image(c: &CpuState) -> String {
    format!(
        "{:x?} {:?} {:x} {:?} {:?}",
        c.gpr,
        c.flags,
        c.rflags_system,
        [c.cs, c.ds, c.es, c.ss, c.fs, c.gs],
        (
            c.rip,
            c.fpr,
            c.fpu_cw,
            c.fpu_sw,
            c.fpu_top,
            c.fpu_empty,
            c.fpu_fip,
            c.fpu_fdp,
            c.fpu_fop,
            c.fpu_fcs,
            c.fpu_fds
        )
    )
}
fn run_form(vm: &Vm, cpu: &mut Vcpu, bytes: &[u8]) -> x86jit_core::StepRun {
    let first = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    // Iced's waiting x87 aliases encode WAIT plus an ESC instruction. Exercise
    // the complete encoding; lifting only WAIT must not count as FPU parity.
    if bytes.len() > 1
        && bytes[0] == 0x9b
        && matches!(first.stop(), StepStop::Exit(Exit::BudgetExhausted))
    {
        cpu.run_steps(vm, 1, StepOptions::default()).unwrap()
    } else {
        first
    }
}

fn memory_image(vm: &Vm) -> Vec<u8> {
    let mut bytes = vec![0; 0x20000];
    for (i, page) in bytes.chunks_mut(0x1000).enumerate() {
        vm.read_bytes((i * 0x1000) as u64, page).unwrap();
    }
    bytes
}
fn exit_image(stop: &StepStop) -> String {
    match stop {
        // Exit::MmioWrite specifies only the low `size` bytes of value.
        StepStop::Exit(Exit::MmioWrite {
            addr,
            size,
            value,
            total_size,
            part,
            ..
        }) => format!(
            "mmio write {addr:x} {size} {:x} {total_size} {part}",
            value & (u64::MAX >> (64 - u32::from(*size) * 8))
        ),
        StepStop::Exit(Exit::UnmappedMemory { addr, access, .. }) => {
            format!("unmapped {addr:x} {access:?}")
        }
        StepStop::Exit(Exit::Exception { addr, vector, .. }) => {
            format!("exception {addr:x} {vector}")
        }
        _ => format!("{stop:?}"),
    }
}

// U3 has no operand-size conflicts. Model-dependent flag images are asserted below.
fn application_conflict(_code: Code, _mmio: bool) -> Option<&'static str> {
    None
}

fn operand_size_witness(code: Code, mmio: bool) -> bool {
    use Code::*;
    // SDM Vol 2B §4.3 PUSH/POP/POPF/RET; Vol 2A §3.3 CALL/JMP,
    // ENTER/LEAVE, LDS/LES and LOOP. 80386 PRM §§17.1.2–17.1.3,
    // pp. 17-1–17-2: operand and stack/address sizes are independent.
    matches!(
        code,
        Pushd_ES
            | Popd_ES
            | Pushd_CS
            | Pushd_SS
            | Popd_SS
            | Pushd_DS
            | Popd_DS
            | Pushd_imm32
            | Popfw
            | Iretw
            | Retnd_imm16
            | Retnd
            | Retfd_imm16
            | Retfd
            | Les_r32_m1632
            | Lds_r32_m1632
            | Enterd_imm16_imm8
            | Leaved
            | Loopne_rel8_16_ECX
            | Loopne_rel8_32_ECX
            | Loope_rel8_16_ECX
            | Loope_rel8_32_ECX
            | Loop_rel8_16_ECX
            | Loop_rel8_32_ECX
            | Call_rel32_32
            | Call_rm32
            | Jmp_rm32
    ) || (code == Enterw_imm16_imm8 && mmio)
}

fn i286_empty_output(code: Code) -> bool {
    // Decision-25; I286/287 PRM Part II Appendix A, Table A-2 pp. A-2–A-4.
    // Exact forms with independent expectations for both execution profiles.
    use Code::*;
    matches!(
        code,
        Fnstcw_m2byte
            | Fstcw_m2byte
            | Fnstsw_m2byte
            | Fstsw_m2byte
            | Fnstsw_AX
            | Fstsw_AX
            | Fst_m32fp
            | Fst_m64fp
            | Fstp_m32fp
            | Fstp_m64fp
            | Fstp_m80fp
            | Fist_m16int
            | Fist_m32int
            | Fistp_m16int
            | Fistp_m32int
            | Fistp_m64int
            | Fbstp_m80bcd
            | Fnstenv_m14byte
            | Fstenv_m14byte
            | Fnsave_m94byte
            | Fsave_m94byte
    )
}

// Reference the instruction directly from the manuals, starting from the seeded
// input. No legacy output or execution helper supplies an expected value. The
// caller still compares the complete state image, memory, exit and step count.
fn conflict_expectation(
    vm: &Vm,
    input: &Vcpu,
    insn: &iced_x86::Instruction,
    generation: CpuGeneration,
    mmio: bool,
) -> (CpuState, Vec<u8>, StepStop, u64) {
    use Code::*;
    let initial = &input.cpu;
    let original = memory_image(vm);
    let mut memory = original.clone();
    let mut cpu = initial.clone();
    let read = |memory: &[u8], addr: u64, size: usize| -> Result<u64, Exit> {
        if mmio && (0x4000..0x5000).contains(&addr) {
            return Err(Exit::mmio_read(addr, size as u8));
        }
        let mut value = [0; 8];
        value[..size].copy_from_slice(&memory[addr as usize..addr as usize + size]);
        Ok(u64::from_le_bytes(value))
    };
    let store = |memory: &mut [u8], addr: u64, value: u64, size: usize| {
        memory[addr as usize..addr as usize + size].copy_from_slice(&value.to_le_bytes()[..size]);
    };
    let address = || {
        let base = match insn.memory_base() {
            iced_x86::Register::BX => initial.gpr[3] & 0xffff,
            iced_x86::Register::EBX => initial.gpr[3] & 0xffff_ffff,
            other => panic!("unexpected conflict address base: {other:?}"),
        };
        let mask = if insn.memory_base() == iced_x86::Register::BX {
            0xffff
        } else {
            0xffff_ffff
        };
        (base + insn.memory_displacement64()) & mask
    };
    let sp = initial.gpr[4] as u16 as u64;
    let bp = initial.gpr[5] as u16 as u64;
    let mut next_sp = sp;
    cpu.rip = insn.next_ip();
    let result = (|| -> Result<(), Exit> {
        match insn.code() {
            // SDM Vol 2B §4.3 PUSH/POP: 66h selects a dword slot.
            Pushd_ES | Pushd_CS | Pushd_SS | Pushd_DS | Pushd_imm32 => {
                let value = match insn.code() {
                    Pushd_ES => u64::from(initial.es),
                    Pushd_CS => u64::from(initial.cs),
                    Pushd_SS => u64::from(initial.ss),
                    Pushd_DS => u64::from(initial.ds),
                    _ => u64::from(insn.immediate32()),
                };
                next_sp = sp - 4;
                store(&mut memory, next_sp, value, 4);
            }
            Popd_ES | Popd_SS | Popd_DS => {
                let value = read(&memory, sp, 4)? as u16;
                match insn.code() {
                    Popd_ES => cpu.es = value,
                    Popd_SS => cpu.ss = value,
                    _ => cpu.ds = value,
                }
                next_sp = sp + 4;
            }
            // 80286 PRM Appendix B POPF p. B-86 (real mode leaves NT/IOPL
            // unchanged); §5.3 Table 5-3 p. 5-7 resets FLAGS to 0002h.
            // The compatibility personality keeps this 286 real FLAGS image.
            // 80386 PRM §17 POPF p. 17-130; SDM Vol 3B §25.17.2:
            // I386 instead restores bits 14:12 from the supplied image.
            Popfw | Iretw => {
                let iret = insn.code() == Iretw;
                let flags = read(&memory, sp + if iret { 4 } else { 0 }, 2)?;
                assert_eq!(flags & 0x100, 0, "reference inputs do not request TF");
                cpu.flags.set_flags16((flags & 0x0ed7) as u16);
                cpu.rflags_system = if generation == CpuGeneration::I386 {
                    flags & 0x7000
                } else {
                    0
                };
                if iret {
                    cpu.rip = read(&memory, sp, 2)?;
                    cpu.cs = read(&memory, sp + 2, 2)? as u16;
                }
                next_sp = sp + if iret { 6 } else { 2 };
            }
            // SDM Vol 2B §4.3 RET: a dword return target is checked
            // against the retained CS limit before stack/selector commit.
            Retnd_imm16 | Retnd | Retfd_imm16 | Retfd => {
                let ip = read(&memory, sp, 4)?;
                if ip > 0xffff {
                    return Err(Exit::exception(initial.rip, 13));
                }
                cpu.rip = ip;
                let far = matches!(insn.code(), Retfd_imm16 | Retfd);
                if far {
                    cpu.cs = read(&memory, sp + 4, 4)? as u16;
                }
                let extra = if matches!(insn.code(), Retnd_imm16 | Retfd_imm16) {
                    u64::from(insn.immediate16())
                } else {
                    0
                };
                next_sp = sp + if far { 8 } else { 4 } + extra;
            }
            // SDM Vol 2A §3.3 LDS/LES: m16:32 has its selector at EA+4.
            Les_r32_m1632 | Lds_r32_m1632 => {
                let a = address();
                let offset = read(&memory, a, 4)?;
                let selector = read(&memory, a + 4, 2)? as u16;
                cpu.gpr[insn.op0_register().number()] = offset;
                if insn.code() == Les_r32_m1632 {
                    cpu.es = selector;
                } else {
                    cpu.ds = selector;
                }
            }
            // SDM Vol 2A §3.3 ENTER/LEAVE; Vol 1 §6.2.3: operand
            // width selects frame values; SS.B selects SP arithmetic.
            Enterw_imm16_imm8 | Enterd_imm16_imm8 => {
                let size = if insn.code() == Enterd_imm16_imm8 {
                    4
                } else {
                    2
                };
                let level = insn.immediate8_2nd() & 31;
                let frame = sp - size as u64;
                store(&mut memory, frame, initial.gpr[5], size);
                next_sp = frame;
                for i in 1..level {
                    let value = read(&memory, bp - u64::from(i) * size as u64, size)?;
                    next_sp -= size as u64;
                    store(&mut memory, next_sp, value, size);
                }
                if level != 0 {
                    next_sp -= size as u64;
                    store(&mut memory, next_sp, frame, size);
                }
                next_sp -= u64::from(insn.immediate16());
                cpu.gpr[5] = if size == 4 {
                    frame
                } else {
                    (initial.gpr[5] & !0xffff) | frame
                };
            }
            Leaved => {
                cpu.gpr[5] = read(&memory, bp, 4)?;
                next_sp = bp + 4;
            }
            // SDM Vol 2A §3.3 LOOP/LOOPcc: 67h selects ECX.
            Loopne_rel8_16_ECX | Loopne_rel8_32_ECX | Loope_rel8_16_ECX | Loope_rel8_32_ECX
            | Loop_rel8_16_ECX | Loop_rel8_32_ECX => {
                let count = (initial.gpr[1] as u32).wrapping_sub(1);
                cpu.gpr[1] = u64::from(count);
                let condition = match insn.mnemonic() {
                    iced_x86::Mnemonic::Loope => initial.flags.zf,
                    iced_x86::Mnemonic::Loopne => !initial.flags.zf,
                    _ => true,
                };
                if count != 0 && condition {
                    cpu.rip = insn.near_branch_target();
                }
            }
            // SDM Vol 2A §3.3 CALL/JMP: a 32-bit target and return slot.
            Call_rel32_32 | Call_rm32 | Jmp_rm32 => {
                let ip = if insn.code() == Call_rel32_32 {
                    insn.near_branch_target()
                } else if insn.op0_kind() == iced_x86::OpKind::Memory {
                    read(&memory, address(), 4)?
                } else {
                    initial.gpr[insn.op0_register().number()] & 0xffff_ffff
                };
                if ip > 0xffff {
                    return Err(Exit::exception(initial.rip, 13));
                }
                cpu.rip = ip;
                if insn.mnemonic() == iced_x86::Mnemonic::Call {
                    next_sp = sp - 4;
                    store(&mut memory, next_sp, insn.next_ip(), 4);
                }
            }
            other => panic!("missing architectural conflict expectation: {other:?}"),
        }
        cpu.gpr[4] = (initial.gpr[4] & !0xffff) | (next_sp & 0xffff);
        Ok(())
    })();
    match result {
        Ok(()) => (cpu, memory, StepStop::Exit(Exit::BudgetExhausted), 1),
        // Faults and MMIO leave the full architectural input unchanged.
        // ENTER's staged frame is discarded (SDM Vol 3A §§7.5–7.6).
        Err(exit) => (initial.clone(), original, StepStop::Exit(exit), 0),
    }
}

#[test]
fn application_real16_forms_have_legacy_parity_cpu_model() {
    model_api::with_cpu_model(application_real16_forms_have_legacy_parity);
}

#[test]
fn application_real16_forms_have_legacy_parity() {
    let forms = x86jit_tests::compat::real16_forms();
    let mut gaps = Vec::new();
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let mut checked = 0;
        let mut architectural = std::collections::BTreeSet::new();
        let mut categories = std::collections::BTreeMap::new();
        for (code, bytes) in &forms {
            let insn = iced_x86::Decoder::with_ip(
                16,
                bytes,
                0x1000,
                iced_x86::DecoderOptions::NO_INVALID_CHECK,
            )
            .decode();
            if !available(*code, generation, bytes)
                || !available(insn.code(), generation, bytes)
                // POP CS predates I286; LOADALL is system-specific and its
                // bytes are decoded as SYSCALL by the Application decoder.
                || matches!(code, Code::Popw_CS | Code::Loadall286)
            {
                continue;
            }
            checked += 1;
            for (seed, absent, emulation, mmio) in [
                (0x123456789abcdef, false, false, false),
                (0x987654321abcdef, false, false, false),
                (0x123456789abcdef, true, false, false),
                (0x987654321abcdef, false, true, false),
                (0x123456789abcdef, false, false, true),
            ] {
                let (mut a, mut ac) = setup(bytes, None, seed);
                let (mut b, mut bc) = setup(bytes, Some(generation), seed);
                if absent {
                    ac.set_fpu_model(x86jit_core::FpuModel::None);
                    bc.set_fpu_model(x86jit_core::FpuModel::None);
                }
                if emulation {
                    ac.set_fpu_emulation(true);
                    bc.cpu.legacy.cr0 |= 4;
                }
                if mmio {
                    for vm in [&mut a, &mut b] {
                        vm.unmap(0x4000, 0x1000).unwrap();
                        vm.map(0x4000, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
                    }
                }
                assert_eq!(application_conflict(*code, mmio), None);
                let expected = operand_size_witness(*code, mmio).then(|| {
                    architectural.insert(format!("{code:?} {bytes:02x?}"));
                    let alias = conflict_expectation(&a, &ac, &insn, CpuGeneration::I286, mmio);
                    let machine = conflict_expectation(&b, &bc, &insn, generation, mmio);
                    (alias, machine)
                });
                let empty_output =
                    (generation == CpuGeneration::I286 && absent && i286_empty_output(*code))
                        .then(|| (ac.cpu.clone(), memory_image(&a)));
                let ar = run_form(&a, &mut ac, bytes);
                let br = run_form(&b, &mut bc, bytes);
                if let Some((initial, memory)) = empty_output {
                    // Application still refuses at ESC, after a separate WAIT
                    // if present. Machine retires; only the control/status
                    // word transfers write memory (decision-25).
                    let mut application = initial.clone();
                    application.rip += u64::from(bytes[0] == 0x9b);
                    let mut machine = initial;
                    machine.rip += bytes.len() as u64;
                    if matches!(code, Code::Fnstsw_AX | Code::Fstsw_AX) {
                        // PRM Part I Appendix A p. A-4: all-high bus assumption.
                        machine.gpr[0] = (machine.gpr[0] & !0xffff) | 0xffff;
                    }
                    let mut machine_memory = memory.clone();
                    if matches!(
                        code,
                        Code::Fnstcw_m2byte
                            | Code::Fstcw_m2byte
                            | Code::Fnstsw_m2byte
                            | Code::Fstsw_m2byte
                    ) {
                        // Part II Appendix B item 10 p. B-2: the memory status
                        // store carries the high bus value even without an NPX.
                        let prefix = usize::from(bytes[0] == 0x9b);
                        let esc = iced_x86::Decoder::with_ip(
                            16,
                            &bytes[prefix..],
                            0x1000 + prefix as u64,
                            iced_x86::DecoderOptions::NO_INVALID_CHECK,
                        )
                        .decode();
                        assert_eq!(esc.memory_base(), iced_x86::Register::BX);
                        assert_eq!(esc.memory_index(), iced_x86::Register::None);
                        let at = ((machine.gpr[3] + esc.memory_displacement64()) & 0xffff) as usize;
                        machine_memory[at..at + 2].fill(0xff);
                    }
                    assert_eq!(image(&ac), state_image(&application), "{code:?}");
                    assert_eq!(image(&bc), state_image(&machine), "{code:?}");
                    assert_eq!(memory_image(&a), memory, "{code:?}");
                    assert_eq!(memory_image(&b), machine_memory, "{code:?}");
                    assert_eq!(ar.steps(), 0, "{code:?}");
                    assert!(
                        matches!(ar.stop(), StepStop::Exit(Exit::UnknownInstruction { addr, .. }) if *addr == application.rip),
                        "{code:?}: {ar:?}"
                    );
                    assert_eq!(br.steps(), 1, "{code:?}");
                    assert!(
                        matches!(br.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                        "{code:?}: {br:?}"
                    );
                    continue;
                }
                if let Some((alias, (state, memory, stop, steps))) = expected {
                    let (mut alias_state, mut alias_memory, mut alias_stop, mut alias_steps) =
                        alias;
                    if let StepStop::Exit(Exit::Exception { vector, .. }) = &alias_stop {
                        // Decision-16: delivery costs no step; the handler's HLT does.
                        let sp = (alias_state.gpr[4] as u16).wrapping_sub(6);
                        for (offset, word) in [
                            (0, alias_state.rip as u16),
                            (2, alias_state.cs),
                            (4, alias_state.flags.to_flags16()),
                        ] {
                            alias_memory[usize::from(sp) + offset..usize::from(sp) + offset + 2]
                                .copy_from_slice(&word.to_le_bytes());
                        }
                        alias_state.gpr[4] = (alias_state.gpr[4] & !0xffff) | u64::from(sp);
                        alias_state.flags.if_ = false;
                        alias_state.cs = 0;
                        alias_state.rip = 0x1801 + u64::from(*vector) * 4;
                        alias_stop = StepStop::Exit(Exit::Hlt);
                        alias_steps = 1;
                    }
                    if state_image(&alias_state) != image(&ac)
                        || alias_memory != memory_image(&a)
                        || exit_image(&alias_stop) != exit_image(ar.stop())
                        || alias_steps != ar.steps()
                    {
                        gaps.push(format!("{generation:?} {code:?} {bytes:02x?} seed={seed:x} mmio={mmio}: alias architectural expectation {alias_steps} {alias_stop:?}, got {ar:?}\nEXPECTED {}\nAPP {}", state_image(&alias_state), image(&ac)));
                    }
                    if state_image(&state) != image(&bc)
                        || memory != memory_image(&b)
                        || exit_image(&stop) != exit_image(br.stop())
                        || steps != br.steps()
                    {
                        gaps.push(format!("{generation:?} {code:?} {bytes:02x?} seed={seed:x} absent={absent} em={emulation} mmio={mmio}: architectural expectation {steps} {stop:?}, got {br:?}\nEXPECTED {}\nLEG {}", state_image(&state), image(&bc)));
                    }
                    continue;
                }
                // Application faults enter the IVT; legacy faults are outward.
                if let StepStop::Exit(Exit::Exception { vector, .. }) = br.stop() {
                    let sp = (bc.cpu.gpr[4] as u16).wrapping_sub(6);
                    for (offset, word) in [
                        (0, bc.cpu.rip as u16),
                        (2, bc.cpu.cs),
                        (4, bc.cpu.flags.to_flags16()),
                    ] {
                        b.write_bytes(u64::from(sp) + offset, &word.to_le_bytes())
                            .unwrap();
                    }
                    bc.cpu.gpr[4] = (bc.cpu.gpr[4] & !0xffff) | u64::from(sp);
                    bc.cpu.flags.if_ = false;
                    bc.cpu.cs = 0;
                    bc.cpu.rip = 0x1801 + u64::from(*vector) * 4;
                }
                let am = memory_image(&a);
                let bm = memory_image(&b);
                let states = image(&ac) == image(&bc) && am == bm;
                let exits = exit_image(ar.stop()) == exit_image(br.stop())
                    || matches!(br.stop(), StepStop::Exit(Exit::Exception { vector, .. }) if ac.cpu.rip == 0x1801 + u64::from(*vector) * 4);
                if !states || !exits {
                    let cat = if matches!(
                        br.stop(),
                        StepStop::Exit(
                            Exit::UnsupportedSystemContext { .. } | Exit::UnknownInstruction { .. }
                        )
                    ) {
                        "refusal"
                    } else {
                        "result"
                    };
                    *categories.entry(cat).or_insert(0) += 1;
                    gaps.push(format!("{generation:?} {code:?} {bytes:02x?} absent={absent} em={emulation} mmio={mmio} {cat}: {:?} / {:?}\nAPP {}\nLEG {}",ar.stop(),br.stop(),image(&ac),image(&bc)));
                }
            }
        }
        eprintln!(
            "{generation:?}: {checked} forms; {} independently checked forms; {categories:?}\n{}",
            architectural.len(),
            architectural.into_iter().collect::<Vec<_>>().join("\n")
        );
    }
    assert!(
        gaps.is_empty(),
        "{} mismatches:\n{}",
        gaps.len(),
        gaps.join("\n")
    );
}

fn legacy_fixture(code: &[u8], generation: CpuGeneration) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x220000), Box::new(Interpreter));
    model_api::select(
        &mut vm,
        LegacyCpuModel {
            generation,
            model_semantics_id: generation as u64,
        },
    );
    for page in (0..0x220000).step_by(0x1000) {
        vm.map(page, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    }
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}
fn step(vm: &Vm, cpu: &mut Vcpu) -> x86jit_core::StepRun {
    cpu.run_steps(vm, 1, StepOptions::default()).unwrap()
}
fn assert_exception(run: &x86jit_core::StepRun, vector: u8) {
    assert_eq!(run.steps(), 0);
    assert!(
        matches!(run.stop(),StepStop::Exit(Exit::Exception {vector:v,..}) if *v==vector),
        "{run:?}"
    );
}
#[test]
fn i286_rejects_every_386_probe_before_operand_access_cpu_model() {
    model_api::with_cpu_model(i286_rejects_every_386_probe_before_operand_access);
}

#[test]
fn i286_rejects_every_386_probe_before_operand_access() {
    let mut checked = 0;
    for (code, bytes) in x86jit_tests::compat::real16_forms() {
        if !available(code, CpuGeneration::I386, &bytes)
            || available(code, CpuGeneration::I286, &bytes)
        {
            continue;
        }
        for model in [x86jit_core::FpuModel::Present, x86jit_core::FpuModel::None] {
            let (vm, mut cpu) = legacy_fixture(&bytes, CpuGeneration::I286);
            cpu.set_fpu_model(model);
            // Waiting x87 aliases encode a separate WAIT before the ESC opcode.
            if bytes[0] == 0x9b {
                assert_eq!(step(&vm, &mut cpu).steps(), 1);
            }
            let before = image(&cpu);
            let run = step(&vm, &mut cpu);
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::Exception { vector: 6, .. })
                ),
                "{code:?} {bytes:02x?}: {run:?}"
            );
            assert_eq!(run.steps(), 0);
            assert_eq!(image(&cpu), before, "{code:?} {bytes:02x?}");
        }
        checked += 1;
    }
    assert!(checked > 200, "{checked}");
    eprintln!("I286: {checked} unavailable forms raise #UD without effects");
}
#[test]
fn esc_and_wait_gate_before_operands_under_both_fpu_models_cpu_model() {
    model_api::with_cpu_model(esc_and_wait_gate_before_operands_under_both_fpu_models);
}

#[test]
fn esc_and_wait_gate_before_operands_under_both_fpu_models() {
    // SDM Vol 3A §2.5 Table 2-2; 80386 PRM §§11.1.2–11.1.4:
    // ESC tests EM/TS; WAIT tests MP && TS and ignores EM.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        for model in [x86jit_core::FpuModel::Present, x86jit_core::FpuModel::None] {
            for msw in 0..16u32 {
                for code in [
                    &b"\xdb\xe3"[..],
                    &b"\xd9\x07"[..],
                    &b"\xd9\xd1"[..],
                    &b"\x9b"[..],
                ] {
                    let (vm, mut cpu) = legacy_fixture(code, generation);
                    cpu.set_fpu_model(model);
                    cpu.cpu.legacy.cr0 = msw & !1;
                    cpu.cpu.gpr[3] = 0xffff; // A memory operand would overrun the limit.
                    cpu.cpu.fpu_sw = 0;
                    cpu.cpu.fpu_cw = 0x127f;
                    let before = image(&cpu);
                    let run = step(&vm, &mut cpu);
                    let wait = code[0] == 0x9b;
                    if if wait {
                        msw & 0xa == 0xa
                    } else {
                        msw & 0xc != 0
                    } {
                        assert_exception(&run, 7);
                        assert_eq!(image(&cpu), before);
                    } else if code == b"\xd9\x07" {
                        // I286 PRM App. B #MP p. B-9: with a coprocessor
                        // attached, a dword past the limit raises 9.
                        assert_exception(
                            &run,
                            if generation == CpuGeneration::I286
                                && model == x86jit_core::FpuModel::Present
                            {
                                9
                            } else {
                                13
                            },
                        );
                    } else if code == b"\xdb\xe3" && model == x86jit_core::FpuModel::Present {
                        assert_eq!(cpu.cpu.fpu_cw, 0x37f);
                        assert_eq!(cpu.cpu.fpu_empty, 0xff);
                        assert_eq!(cpu.cpu.rip, 0x1002);
                    } else if wait {
                        assert_eq!(cpu.cpu.rip, 0x1001);
                    }
                }
            }
        }
    }
}
#[test]
fn delegated_x87_uses_cached_limits_and_splits_at_a20_cpu_model() {
    model_api::with_cpu_model(delegated_x87_uses_cached_limits_and_splits_at_a20);
}

#[test]
fn delegated_x87_uses_cached_limits_and_splits_at_a20() {
    // SDM Vol 1 §8.2.2: the 80-bit image of +1.0. Addressing and A20:
    // Vol 3B §23.1.1; Vol 3A §11.7.13.4; decisions 20 and 22.
    let one = [0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f];
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) = legacy_fixture(b"\xdb\x2f\xdb\x3f", generation); // FLD/FSTP tbyte [BX]
        cpu.cpu.ds = 0xffff;
        cpu.cpu.legacy.segments[1].base = 0xffff0;
        cpu.cpu.gpr[3] = 0xf;
        cpu.set_a20_enabled(false);
        vm.write_bytes(0xfffff, &one[..1]).unwrap();
        vm.write_bytes(0, &one[1..]).unwrap();
        let run = step(&vm, &mut cpu);
        assert_eq!(run.steps(), 1);
        assert_eq!(cpu.cpu.fpr[7], one);
        cpu.cpu.gpr[3] = 0x1f;
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        let mut stored = [0; 10];
        vm.read_bytes(0xf, &mut stored).unwrap();
        assert_eq!(stored, one);
        let (vm, mut cpu) = legacy_fixture(b"\xdb\x2f", generation);
        cpu.cpu.gpr[3] = 0xfff8;
        // I286 PRM App. B #MP p. B-9: the tbyte's later words pass FFFFh.
        assert_exception(
            &step(&vm, &mut cpu),
            if generation == CpuGeneration::I286 {
                9
            } else {
                13
            },
        );
        if generation == CpuGeneration::I386 {
            cpu.cpu.legacy.segments[1].limit = 0x1ffff;
            cpu.cpu.gpr[3] = 0x10008;
            // 67h addressing retains the full offset above FFFFh.
            vm.write_bytes(0x1000, b"\x67\xdb\x2b").unwrap(); // FLD tbyte [EBX]
            vm.write_bytes(0x10008, &one).unwrap();
            assert_eq!(step(&vm, &mut cpu).steps(), 1);
            assert_eq!(cpu.cpu.fpr[7], one);
        }
    }
}
#[test]
fn delegated_bit_strings_check_the_adjusted_offset_and_a20_cpu_model() {
    model_api::with_cpu_model(delegated_bit_strings_check_the_adjusted_offset_and_a20);
}

#[test]
fn delegated_bit_strings_check_the_adjusted_offset_and_a20() {
    // SDM Vol 2A §3.3 BT/BTS/BTR/BTC: signed register index selects
    // the byte in the string before the segment-limit check.
    let (vm, mut cpu) = legacy_fixture(b"\x0f\xab\x0f", CpuGeneration::I386); // BTS [BX],CX
    cpu.cpu.ds = 0xffff;
    cpu.cpu.legacy.segments[1].base = 0xffff0;
    cpu.cpu.gpr[3] = 0xf;
    cpu.cpu.gpr[1] = 8;
    cpu.set_a20_enabled(false);
    assert_eq!(step(&vm, &mut cpu).steps(), 1);
    let mut byte = [0];
    vm.read_bytes(0, &mut byte).unwrap();
    assert_eq!(byte, [1]);
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[3] = 0x20;
    cpu.cpu.gpr[1] = 16;
    cpu.cpu.legacy.segments[1].limit = 0x20;
    assert_exception(&step(&vm, &mut cpu), 13);
}
#[test]
fn rep_port_strings_match_application_through_input_and_mmio_resumes_cpu_model() {
    model_api::with_cpu_model(rep_port_strings_match_application_through_input_and_mmio_resumes);
}

#[test]
fn rep_port_strings_match_application_through_input_and_mmio_resumes() {
    // SDM Vol 2A §3.3 INS; Vol 2B §4.3 OUTS and REP: one
    // successful element updates indices/count; IN completion is buffered.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        for prefix in [&[][..], &[0x66][..], &[0x67][..], &[0x66, 0x67][..]] {
            if generation == CpuGeneration::I286 && !prefix.is_empty() {
                continue;
            }
            for opcode in [0x6c, 0x6d, 0x6e, 0x6f] {
                for df in [false, true] {
                    for count in [0, 1, 3] {
                        let mut bytes = vec![0xf3];
                        bytes.extend(prefix);
                        bytes.extend([opcode, 0xf4]);
                        let (mut a, mut ac) = setup(&bytes, None, 0x123456789abcdef);
                        // As in the corpus, compare only forms Application
                        // accepts. Its Real16 port strings require word-sized
                        // indices and byte/word elements. Legacy's dword and
                        // address-override forms have architectural witnesses below.
                        if x86jit_core::lift::lift_one(
                            &a.mem,
                            x86jit_core::lift::FetchAddr::flat(0x1000),
                            CpuMode::Real16,
                        )
                        .is_err()
                        {
                            continue;
                        }
                        let (mut b, mut bc) = setup(&bytes, Some(generation), 0x123456789abcdef);
                        for cpu in [&mut ac, &mut bc] {
                            cpu.cpu.gpr[1] = 0xdead_beef_0000_0000 | count;
                            cpu.cpu.gpr[6] = 0xdead_beef_0000_4080;
                            cpu.cpu.gpr[7] = 0xdead_beef_0000_4180;
                            cpu.cpu.flags.df = df;
                        }
                        for vm in [&mut a, &mut b] {
                            vm.unmap(0x4000, 0x1000).unwrap();
                            vm.map(0x4000, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
                        }
                        let mut ports = 0;
                        let mut mmios = 0;
                        for _ in 0..20 {
                            let ar = step(&a, &mut ac);
                            let br = step(&b, &mut bc);
                            assert_eq!(ar.steps(), br.steps(), "opcode={opcode:x} prefix={prefix:02x?} df={df} count={count}: {ar:?} / {br:?}");
                            assert_eq!(exit_image(ar.stop()), exit_image(br.stop()));
                            assert_eq!(
                                image(&ac),
                                image(&bc),
                                "opcode={opcode:x} prefix={prefix:02x?} df={df} count={count}"
                            );
                            match ar.stop() {
                                StepStop::Exit(Exit::PortIo { dir, .. }) => {
                                    ports += 1;
                                    if *dir == x86jit_core::PortDir::In {
                                        ac.complete_port_in(0x89ab_cdef);
                                        bc.complete_port_in(0x89ab_cdef);
                                    }
                                }
                                StepStop::Exit(Exit::MmioRead { .. }) => {
                                    mmios += 1;
                                    ac.complete_mmio_read(0x1234_5678);
                                    bc.complete_mmio_read(0x1234_5678);
                                }
                                StepStop::Exit(Exit::MmioWrite { .. }) => {
                                    mmios += 1;
                                    ac.complete_mmio_write();
                                    bc.complete_mmio_write();
                                }
                                StepStop::Exit(Exit::BudgetExhausted) => {}
                                StepStop::Exit(Exit::Hlt) => break,
                                other => panic!("{other:?}"),
                            }
                        }
                        assert_eq!((ports, mmios), (count, count));
                        assert_eq!(ac.cpu.rip, 0x1000 + bytes.len() as u64);
                        assert_eq!(bc.cpu.gpr[1] & 0xffff_ffff, 0);
                        assert_eq!(ac.retired_instructions(), bc.retired_instructions());
                    }
                }
            }
        }
    }
}
#[test]
fn input_completion_keeps_a20_until_store_then_changes_at_element_boundary_cpu_model() {
    model_api::with_cpu_model(
        input_completion_keeps_a20_until_store_then_changes_at_element_boundary,
    );
}

#[test]
fn input_completion_keeps_a20_until_store_then_changes_at_element_boundary() {
    // Decision-22's boundary contract; REP interruption windows: SDM Vol 2B §4.3 REP.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (mut vm, mut cpu) = legacy_fixture(b"\xf3\x6c\xf4", generation);
        cpu.cpu.es = 0xffff;
        cpu.cpu.legacy.segments[2].base = 0xffff0;
        cpu.cpu.gpr[7] = 0x20;
        cpu.cpu.gpr[1] = 2;
        vm.unmap(0x100000, 0x1000).unwrap();
        vm.map(0x100000, 0x1000, Prot::RWX, RegionKind::Trap)
            .unwrap();
        assert!(matches!(
            step(&vm, &mut cpu).stop(),
            StepStop::Exit(Exit::PortIo { .. })
        ));
        cpu.set_a20_enabled(false);
        assert!(cpu.a20_enabled());
        assert_eq!(cpu.pending_a20_enabled(), Some(false));
        cpu.complete_port_in(0x31);
        assert!(matches!(
            step(&vm, &mut cpu).stop(),
            StepStop::Exit(Exit::MmioWrite {
                addr: 0x100010,
                value: 0x31,
                ..
            })
        ));
        assert_eq!(cpu.cpu.gpr[1], 2);
        assert!(cpu.a20_enabled());
        cpu.complete_mmio_write();
        assert!(matches!(
            step(&vm, &mut cpu).stop(),
            StepStop::Exit(Exit::PortIo { .. })
        ));
        assert!(!cpu.a20_enabled());
        assert_eq!(cpu.pending_a20_enabled(), None);
        assert_eq!(cpu.cpu.gpr[1], 1);
        cpu.complete_port_in(0x32);
        assert!(matches!(
            step(&vm, &mut cpu).stop(),
            StepStop::Exit(Exit::Hlt)
        ));
        let mut byte = [0];
        vm.read_bytes(0x11, &mut byte).unwrap();
        assert_eq!(byte, [0x32]);
        assert_eq!(cpu.cpu.gpr[1], 0);
    }
}

#[test]
fn enter_preserves_widths_aliases_and_fault_restart_cpu_model() {
    model_api::with_cpu_model(enter_preserves_widths_aliases_and_fault_restart);
}

#[test]
fn enter_preserves_widths_aliases_and_fault_restart() {
    // SDM Vol 2A §3.3 ENTER, Operation: FrameTemp comes from ESP for
    // operand size 32 even with a 16-bit stack. Display reads see prior pushes.
    let (vm, mut cpu) = legacy_fixture(b"\x66\xc8\x00\x00\x00", CpuGeneration::I386);
    cpu.cpu.gpr[4] = 0xdead_beef_1234_8000;
    cpu.cpu.gpr[5] = 0x8888;
    assert_eq!(step(&vm, &mut cpu).steps(), 1);
    assert_eq!(cpu.cpu.gpr[4], 0xdead_beef_1234_7ffc);
    assert_eq!(cpu.cpu.gpr[5], 0x1234_7ffc);
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) = legacy_fixture(b"\xc8\x00\x00\x02", generation);
        cpu.cpu.gpr[5] = 0x8000;
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        let mut words = [0; 6];
        vm.read_bytes(0x7ffa, &mut words).unwrap();
        assert_eq!(words, [0xfe, 0x7f, 0, 0x80, 0, 0x80]);
        // SDM Vol 2A §3.3 ENTER, Real-Address Mode Exceptions: #SS
        // when the new SP is outside the cached stack limit, before any commit.
        let (vm, mut cpu) = legacy_fixture(b"\xc8\x00\x20\x00", generation);
        cpu.cpu.gpr[4] = 0x1000;
        cpu.cpu.legacy.segments[3].limit = 0x1fff;
        let before = image(&cpu);
        assert_exception(&step(&vm, &mut cpu), 12);
        assert_eq!(image(&cpu), before);
        let mut word = [0; 2];
        vm.read_bytes(0xffe, &mut word).unwrap();
        assert_eq!(word, [0, 0]);
    }
}
#[test]
fn x87_wide_store_preflights_both_a20_destinations_cpu_model() {
    model_api::with_cpu_model(x87_wide_store_preflights_both_a20_destinations);
}

#[test]
fn x87_wide_store_preflights_both_a20_destinations() {
    // SDM Vol 1 §8.1.9 and Vol 2B §4.3 FSAVE/FNSAVE: the 16-bit
    // image is 94 bytes; FNINIT follows a successful save. Decision-22 splits
    // bus transfers at A20 without permitting a partial RAM store on failure.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) = legacy_fixture(b"\xdd\x37", generation);
        cpu.cpu.gpr[3] = 0x4000;
        cpu.cpu.fpu_cw = 0x127f;
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        let mut expected = [0; 94];
        vm.read_bytes(0x4000, &mut expected).unwrap();
        let (mut vm, mut cpu) = legacy_fixture(b"\xdd\x37", generation);
        cpu.cpu.ds = 0xffff;
        cpu.cpu.legacy.segments[1].base = 0xffff0;
        cpu.cpu.gpr[3] = 0xf;
        cpu.cpu.fpu_cw = 0x127f;
        cpu.set_a20_enabled(false);
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        let mut actual = [0; 94];
        vm.read_bytes(0xfffff, &mut actual[..1]).unwrap();
        vm.read_bytes(0, &mut actual[1..]).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(cpu.cpu.fpu_cw, 0x37f);
        cpu.cpu.rip = 0x1000;
        cpu.cpu.fpu_cw = 0x127f;
        vm.write_bytes(0xfffff, &[0xaa]).unwrap();
        vm.unmap(0, 0x1000).unwrap();
        vm.map(0, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
        let before = image(&cpu);
        let run = step(&vm, &mut cpu);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::UnmappedMemory { addr: 0, .. })
            ),
            "{run:?}"
        );
        assert_eq!(image(&cpu), before);
        let mut byte = [0];
        vm.read_bytes(0xfffff, &mut byte).unwrap();
        assert_eq!(byte, [0xaa]);
    }
}
#[test]
fn aad_decimal_base_and_absent_fpu_operand_checks_cpu_model() {
    model_api::with_cpu_model(aad_decimal_base_and_absent_fpu_operand_checks);
}

#[test]
fn aad_decimal_base_and_absent_fpu_operand_checks() {
    // SDM Vol 2A §3.3 AAD: AL := AH * imm8 + AL; AH := 0.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (vm, mut cpu) = legacy_fixture(b"\xd5\x0a", generation);
        cpu.cpu.gpr[0] = 0x1234_0304;
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        assert_eq!(cpu.cpu.gpr[0], 0x1234_0022);
        // §17.9's absent-input contract retains CPU limit checks and does not
        // access an unattached coprocessor's memory or change its register image.
        let (mut vm, mut cpu) = legacy_fixture(b"\xd9\x07", generation);
        cpu.set_fpu_model(x86jit_core::FpuModel::None);
        cpu.cpu.gpr[3] = 0x4000;
        vm.unmap(0x4000, 0x1000).unwrap();
        vm.map(0x4000, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
        cpu.cpu.fpu_cw = 0x127f;
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        assert_eq!(cpu.cpu.fpu_cw, 0x127f);
        cpu.cpu.rip = 0x1000;
        cpu.cpu.legacy.segments[1].limit = 0x4001;
        assert_exception(&step(&vm, &mut cpu), 13);
    }
}

#[test]
fn port_input_completion_uses_the_checked_wrapped_linear_address_cpu_model() {
    model_api::with_cpu_model(port_input_completion_uses_the_checked_wrapped_linear_address);
}

#[test]
fn port_input_completion_uses_the_checked_wrapped_linear_address() {
    // SDM Vol 3A §3.4.3: 32-bit segment base plus effective offset forms
    // the 32-bit linear address. The completion retains that checked address.
    let (vm, mut cpu) = legacy_fixture(b"\xf3\x67\x6c\xf4", CpuGeneration::I386);
    cpu.cpu.legacy.segments[2].base = 0xffff_fff0;
    cpu.cpu.gpr[7] = 0xdead_beef_0000_0020;
    cpu.cpu.gpr[1] = 1;
    assert!(matches!(
        step(&vm, &mut cpu).stop(),
        StepStop::Exit(Exit::PortIo { .. })
    ));
    cpu.complete_port_in(0xa5);
    assert!(matches!(
        step(&vm, &mut cpu).stop(),
        StepStop::Exit(Exit::Hlt)
    ));
    let mut byte = [0];
    vm.read_bytes(0x10, &mut byte).unwrap();
    assert_eq!(byte, [0xa5]);
    assert_eq!(cpu.cpu.gpr[7], 0x21);
    assert_eq!(cpu.cpu.gpr[1], 0);
}

#[test]
fn i386_dword_port_strings_resume_each_element_with_both_address_sizes_cpu_model() {
    model_api::with_cpu_model(i386_dword_port_strings_resume_each_element_with_both_address_sizes);
}

#[test]
fn i386_dword_port_strings_resume_each_element_with_both_address_sizes() {
    // SDM Vol 2A §3.3 INS; Vol 2B §4.3 OUTS and REP: operand size
    // determines four-byte transfers; address size selects indices and count.
    for addr32 in [false, true] {
        for opcode in [0x6d, 0x6f] {
            for df in [false, true] {
                let mut code = vec![0xf3, 0x66];
                if addr32 {
                    code.push(0x67);
                }
                code.extend([opcode, 0xf4]);
                let (mut vm, mut cpu) = legacy_fixture(&code, CpuGeneration::I386);
                cpu.cpu.gpr[1] = 0xdead_beef_0000_0003;
                cpu.cpu.gpr[6] = 0xdead_beef_0000_4080;
                cpu.cpu.gpr[7] = 0xdead_beef_0000_4180;
                cpu.cpu.flags.df = df;
                vm.unmap(0x4000, 0x1000).unwrap();
                vm.map(0x4000, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
                let mut ports = 0;
                let mut mmios = 0;
                for _ in 0..20 {
                    let run = step(&vm, &mut cpu);
                    match run.stop() {
                        StepStop::Exit(Exit::PortIo {
                            size: 4,
                            dir,
                            value,
                            ..
                        }) => {
                            assert_eq!(run.steps(), 1);
                            ports += 1;
                            if *dir == x86jit_core::PortDir::In {
                                cpu.complete_port_in(0x89ab_cdef);
                            } else {
                                assert_eq!(*value, 0x1234_5678);
                            }
                        }
                        StepStop::Exit(Exit::MmioRead { size: 4, .. }) => {
                            assert_eq!(run.steps(), 0);
                            mmios += 1;
                            cpu.complete_mmio_read(0x1234_5678);
                        }
                        StepStop::Exit(Exit::MmioWrite {
                            size: 4,
                            value: 0x89ab_cdef,
                            ..
                        }) => {
                            assert_eq!(run.steps(), 0);
                            mmios += 1;
                            cpu.complete_mmio_write();
                        }
                        StepStop::Exit(Exit::Hlt) => break,
                        other => panic!("{other:?}"),
                    }
                }
                assert_eq!((ports, mmios), (3, 3));
                assert_eq!(cpu.cpu.rip, 0x1000 + code.len() as u64);
                let index = if opcode == 0x6d { 7 } else { 6 };
                let start = if index == 7 { 0x4180u64 } else { 0x4080 };
                let high = if addr32 { 0 } else { 0xdead_beef_0000_0000 };
                assert_eq!(
                    cpu.cpu.gpr[index],
                    high | if df { start - 12 } else { start + 12 }
                );
                assert_eq!(cpu.cpu.gpr[1], high);
                assert_eq!(cpu.retired_instructions(), 2);
            }
        }
    }
}

#[test]
fn split_xchg_uses_gated_ram_and_keeps_the_locked_device_protocol_cpu_model() {
    model_api::with_cpu_model(split_xchg_uses_gated_ram_and_keeps_the_locked_device_protocol);
}

#[test]
fn split_xchg_uses_gated_ram_and_keeps_the_locked_device_protocol() {
    // SDM Vol 2B §4.3 XCHG; decision-22: the complete operand is
    // checked before physical splitting. A locked operand with a RAM piece and a
    // device piece is performed by pieces (spec.md §5.2, xj-ir6.34): the RAM
    // piece is read, the device piece is read and written once, and the RAM
    // piece is written after the device write is acknowledged.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let (mut vm, mut cpu) = legacy_fixture(b"\x87\x07", generation);
        cpu.cpu.ds = 0xffff;
        cpu.cpu.legacy.segments[1].base = 0xffff0;
        cpu.cpu.gpr[3] = 0xf;
        cpu.cpu.gpr[0] = 0x1234_aa55;
        cpu.set_a20_enabled(false);
        vm.write_bytes(0xfffff, &[0x11]).unwrap();
        vm.write_bytes(0, &[0x22]).unwrap();
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        assert_eq!(cpu.cpu.gpr[0], 0x1234_2211);
        let mut byte = [0];
        vm.read_bytes(0xfffff, &mut byte).unwrap();
        assert_eq!(byte, [0x55]);
        vm.read_bytes(0, &mut byte).unwrap();
        assert_eq!(byte, [0xaa]);
        cpu.cpu.rip = 0x1000;
        vm.unmap(0, 0x1000).unwrap();
        vm.map(0, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
        let run = step(&vm, &mut cpu);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioRead {
                    addr: 0,
                    size: 1,
                    ..
                })
            ),
            "{run:?}"
        );
        vm.read_bytes(0xfffff, &mut byte).unwrap();
        assert_eq!(byte, [0x55]);
        cpu.complete_mmio_read(0x77);
        let run = step(&vm, &mut cpu);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0,
                    size: 1,
                    value: 0x22,
                    ..
                })
            ),
            "{run:?}"
        );
        vm.read_bytes(0xfffff, &mut byte).unwrap();
        assert_eq!(byte, [0x55]);
        cpu.complete_mmio_write();
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        assert_eq!(cpu.cpu.gpr[0], 0x1234_7755);
        vm.read_bytes(0xfffff, &mut byte).unwrap();
        assert_eq!(byte, [0x11]);
    }
}

#[test]
fn unavailable_x87_output_keeps_ud_precedence_after_the_msw_gate_cpu_model() {
    model_api::with_cpu_model(unavailable_x87_output_keeps_ud_precedence_after_the_msw_gate);
}

#[test]
fn unavailable_x87_output_keeps_ud_precedence_after_the_msw_gate() {
    // SDM Vol 2A §3.3 FISTTP: the form requires SSE3. Decision-21's
    // generation contract and SDM Vol 3A §2.5 Table 2-2 apply:
    // EM raises #NM before #UD. An absent output does not grant a later ISA.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        for model in [x86jit_core::FpuModel::Present, x86jit_core::FpuModel::None] {
            for em in [false, true] {
                let (vm, mut cpu) = legacy_fixture(b"\xdf\x0f", generation); // FISTTP m16int requires SSE3.
                cpu.set_fpu_model(model);
                cpu.cpu.legacy.cr0 = u32::from(em) * 4;
                let before = image(&cpu);
                assert_exception(&step(&vm, &mut cpu), if em { 7 } else { 6 });
                assert_eq!(image(&cpu), before);
            }
        }
    }
}

#[test]
fn retained_cs_default_size_does_not_change_the_real_x87_environment_layout_cpu_model() {
    model_api::with_cpu_model(
        retained_cs_default_size_does_not_change_the_real_x87_environment_layout,
    );
}

#[test]
fn retained_cs_default_size_does_not_change_the_real_x87_environment_layout() {
    // SDM Vol 1 §8.1.10, Figs. 8-10/8-12: PE selects the real image;
    // the operand-size attribute selects its width independently of CS.D.
    for (normal, big, bytes) in [
        (&b"\xd9\x37"[..], &b"\x66\xd9\x33"[..], 14),
        (&b"\xdd\x37"[..], &b"\x66\xdd\x33"[..], 94),
    ] {
        let mut images = Vec::new();
        for (default32, code) in [(false, normal), (true, big)] {
            let (vm, mut cpu) = legacy_fixture(code, CpuGeneration::I386);
            if default32 {
                cpu.cpu.legacy.segments[0].attributes |= 0x4000;
            }
            cpu.cpu.gpr[3] = 0x4000;
            cpu.cpu.fpu_fip = 0x12345;
            cpu.cpu.fpu_fdp = 0xabcde;
            cpu.cpu.fpu_fop = 0x155;
            assert_eq!(step(&vm, &mut cpu).steps(), 1);
            let mut image = vec![0; bytes];
            vm.read_bytes(0x4000, &mut image).unwrap();
            images.push(image);
        }
        assert_eq!(images[0], images[1]);
    }
    // SDM Vol 2A §3.3 AAD also applies in real mode with retained CS.D=1.
    let (vm, mut cpu) = legacy_fixture(b"\xd5\x0a", CpuGeneration::I386);
    cpu.cpu.legacy.segments[0].attributes |= 0x4000;
    cpu.cpu.gpr[0] = 0x0304;
    assert_eq!(step(&vm, &mut cpu).steps(), 1);
    assert_eq!(cpu.cpu.gpr[0], 34);
}

#[cfg(unix)]
fn linear_edge_fixture(code: &[u8], a20: bool) -> (Vm, Vcpu) {
    // Reserve the linear space plus a poison page without committing 4 GiB of RAM.
    let len = 0x1_0000_4000;
    // SAFETY: the anonymous mapping is owned by HostRam, whose destructor
    // releases exactly this span after the VM's last access.
    let ram = unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        assert_ne!(ptr, libc::MAP_FAILED);
        x86jit_core::HostRam {
            ptr: ptr.cast(),
            len,
            guest_base: 0,
            dtor: Box::new(|ptr, len| {
                libc::munmap(ptr.cast(), len);
            }),
            protect: None,
        }
    };
    let mut vm =
        Vm::with_backend_host_ram(VmConfig::reserved(len as u64), Box::new(Interpreter), ram)
            .unwrap();
    model_api::select(
        &mut vm,
        LegacyCpuModel {
            generation: CpuGeneration::I386,
            model_semantics_id: 386,
        },
    );
    for page in [
        0,
        0x1000,
        0x7000,
        0x8000,
        0xffef_f000,
        0xffff_f000,
        0x1_0000_0000,
    ] {
        vm.map(page, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    }
    // Poison the unwrapped address so a missing wrap cannot pass by reading zero.
    vm.write_bytes(0x1_0000_0000, &[0xff; 128]).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    cpu.set_a20_enabled(a20);
    (vm, cpu)
}
#[cfg(unix)]
fn edge_physical(linear: u32, a20: bool) -> u64 {
    u64::from(if a20 { linear } else { linear & !(1 << 20) })
}
#[cfg(unix)]
fn edge_read(vm: &Vm, linear: u32, len: usize, a20: bool) -> Vec<u8> {
    (0..len)
        .map(|i| {
            let mut byte = [0];
            vm.read_bytes(edge_physical(linear.wrapping_add(i as u32), a20), &mut byte)
                .unwrap();
            byte[0]
        })
        .collect()
}

#[cfg(unix)]
#[test]
fn bound_wraps_the_upper_linear_address_before_a20_cpu_model() {
    model_api::with_cpu_model(bound_wraps_the_upper_linear_address_before_a20);
}

#[cfg(unix)]
#[test]
fn bound_wraps_the_upper_linear_address_before_a20() {
    // SDM Vol 3A §3.4: linear addresses range from 0 to FFFFFFFFh.
    // Vol 2A §3.3 BOUND: signed lower/upper words are consecutive.
    for a20 in [true, false] {
        let (mut vm, mut cpu) = linear_edge_fixture(b"\x62\x07", a20);
        cpu.cpu.legacy.segments[1].base = 0xffff_fffc;
        cpu.cpu.gpr[3] = 2;
        cpu.cpu.gpr[0] = 1;
        vm.write_bytes(edge_physical(0xffff_fffe, a20), &0i16.to_le_bytes())
            .unwrap();
        vm.write_bytes(0, &2i16.to_le_bytes()).unwrap();
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        assert_eq!(cpu.cpu.rip, 0x1002);
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[0] = 3;
        let before = image(&cpu);
        assert_exception(&step(&vm, &mut cpu), 5);
        assert_eq!(image(&cpu), before);
        cpu.cpu.gpr[0] = 1;
        vm.unmap(0, 0x1000).unwrap();
        let before = image(&cpu);
        let run = step(&vm, &mut cpu);
        let StepStop::Exit(exit @ Exit::UnmappedMemory { addr: 0, .. }) = run.stop() else {
            panic!("{run:?}");
        };
        let metadata = exit.fault_metadata().unwrap();
        let x86jit_core::Knowledge::Known(linear) = metadata.linear_address() else {
            panic!("{metadata:?}");
        };
        assert_eq!(linear.value(), 0);
        assert_eq!(
            metadata.physical_address(),
            x86jit_core::Knowledge::Known(0)
        );
        assert_eq!(image(&cpu), before);
        vm.map(0, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
        let before = image(&cpu);
        assert!(matches!(
            step(&vm, &mut cpu).stop(),
            StepStop::Exit(Exit::MmioRead {
                addr: 0,
                size: 2,
                ..
            })
        ));
        assert_eq!(image(&cpu), before);
    }
}

#[cfg(unix)]
#[test]
fn far_pointer_second_halves_wrap_linear_addresses_cpu_model() {
    model_api::with_cpu_model(far_pointer_second_halves_wrap_linear_addresses);
}

#[cfg(unix)]
#[test]
fn far_pointer_second_halves_wrap_linear_addresses() {
    // SDM Vol 3A §3.4; Vol 2A §3.3 LDS/LES/LFS/LGS/LSS:
    // the selector follows the operand-sized offset, including at 4 GiB.
    for a20 in [true, false] {
        for opcode in [
            &b"\xc4\x07"[..],
            &b"\xc5\x07"[..],
            &b"\x0f\xb2\x07"[..],
            &b"\x0f\xb4\x07"[..],
            &b"\x0f\xb5\x07"[..],
        ] {
            for dword in [false, true] {
                let mut code = Vec::new();
                if dword {
                    code.push(0x66);
                }
                code.extend_from_slice(opcode);
                let (vm, mut cpu) = linear_edge_fixture(&code, a20);
                let size = if dword { 4 } else { 2 };
                let start = 0u32.wrapping_sub(size);
                cpu.cpu.legacy.segments[1].base = start - 2;
                cpu.cpu.gpr[3] = 2;
                cpu.cpu.gpr[0] = 0xdead_beef_0000_0000;
                vm.write_bytes(
                    edge_physical(start, a20),
                    &0x1234_5678u32.to_le_bytes()[..size as usize],
                )
                .unwrap();
                vm.write_bytes(0, &0x2345u16.to_le_bytes()).unwrap();
                assert_eq!(step(&vm, &mut cpu).steps(), 1, "{code:02x?}");
                let selector = match opcode {
                    [0xc4, ..] => cpu.cpu.es,
                    [0xc5, ..] => cpu.cpu.ds,
                    [0x0f, 0xb2, ..] => cpu.cpu.ss,
                    [0x0f, 0xb4, ..] => cpu.cpu.fs,
                    _ => cpu.cpu.gs,
                };
                assert_eq!(selector, 0x2345, "{code:02x?}");
                assert_eq!(
                    cpu.cpu.gpr[0],
                    if dword {
                        0x1234_5678
                    } else {
                        0xdead_beef_0000_5678
                    }
                );
            }
        }
    }
}

#[cfg(unix)]
#[test]
fn x87_images_wrap_linear_parts_and_preflight_the_wrapped_destination_cpu_model() {
    model_api::with_cpu_model(x87_images_wrap_linear_parts_and_preflight_the_wrapped_destination);
}

#[cfg(unix)]
#[test]
fn x87_images_wrap_linear_parts_and_preflight_the_wrapped_destination() {
    // SDM Vol 3A §3.4; Vol 1 §8.1.10, Fig. 8-12; Vol 2A §3.3
    // FSAVE/FNSAVE, FRSTOR, FLDENV: 14-byte environment, 94-byte save.
    for a20 in [true, false] {
        let (mut vm, mut cpu) = linear_edge_fixture(b"\xdd\x37", a20);
        cpu.cpu.legacy.segments[1].base = 0xffff_fff0;
        cpu.cpu.gpr[3] = 0xf;
        cpu.cpu.fpu_cw = 0x127f;
        let mut expected = vec![0; 94];
        expected[..2].copy_from_slice(&0x127fu16.to_le_bytes());
        expected[4..6].copy_from_slice(&0xffffu16.to_le_bytes());
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        assert_eq!(edge_read(&vm, 0xffff_ffff, 94, a20), expected);
        assert_eq!(cpu.cpu.fpu_cw, 0x37f);
        for code in [b"\xdd\x27", b"\xd9\x27"] {
            // FRSTOR, FLDENV
            vm.write_bytes(0x1000, code).unwrap();
            cpu.cpu.rip = 0x1000;
            assert_eq!(step(&vm, &mut cpu).steps(), 1);
            assert_eq!(cpu.cpu.fpu_cw, 0x127f);
            assert_eq!(cpu.cpu.fpu_empty, 0xff);
        }
        vm.write_bytes(0x1000, b"\xdd\x37").unwrap();
        cpu.cpu.rip = 0x1000;
        vm.write_bytes(edge_physical(0xffff_ffff, a20), &[0xaa])
            .unwrap();
        vm.unmap(0, 0x1000).unwrap();
        vm.map(0, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
        let before = image(&cpu);
        assert!(matches!(
            step(&vm, &mut cpu).stop(),
            StepStop::Exit(Exit::UnmappedMemory { addr: 0, .. })
        ));
        assert_eq!(image(&cpu), before);
        assert_eq!(edge_read(&vm, 0xffff_ffff, 1, a20), [0xaa]);
    }
}

#[cfg(unix)]
#[test]
fn enter_and_all_register_frames_wrap_each_linear_slot_cpu_model() {
    model_api::with_cpu_model(enter_and_all_register_frames_wrap_each_linear_slot);
}

#[cfg(unix)]
#[test]
fn enter_and_all_register_frames_wrap_each_linear_slot() {
    // SDM Vol 3A §3.4; Vol 2A §3.3 ENTER; Vol 2B §4.3 PUSHA/POPA.
    for a20 in [true, false] {
        let (vm, mut cpu) = linear_edge_fixture(b"\xc8\x00\x00\x02", a20);
        cpu.cpu.legacy.segments[3].base = 0xffff_fff8;
        cpu.cpu.gpr[4] = 12;
        cpu.cpu.gpr[5] = 8;
        vm.write_bytes(edge_physical(0xffff_fffe, a20), &0x1234u16.to_le_bytes())
            .unwrap();
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        assert_eq!(
            edge_read(&vm, 0xffff_fffe, 6, a20),
            [10, 0, 0x34, 0x12, 8, 0]
        );
        assert_eq!((cpu.cpu.gpr[4], cpu.cpu.gpr[5]), (6, 10));
        let (vm, mut cpu) = linear_edge_fixture(b"\x60\x61", a20);
        cpu.cpu.legacy.segments[3].base = 0xffff_fff8;
        cpu.cpu.gpr[..8].copy_from_slice(&[0x11, 0x22, 0x33, 0x44, 16, 0x55, 0x66, 0x77]);
        let original = cpu.cpu.gpr;
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        assert_eq!(
            edge_read(&vm, 0xffff_fff8, 16, a20),
            [0x77, 0, 0x66, 0, 0x55, 0, 16, 0, 0x44, 0, 0x33, 0, 0x22, 0, 0x11, 0]
        );
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        assert_eq!(cpu.cpu.gpr, original);
    }
}

#[cfg(unix)]
#[test]
fn string_elements_and_xlat_wrap_linear_addresses_cpu_model() {
    model_api::with_cpu_model(string_elements_and_xlat_wrap_linear_addresses);
}

#[cfg(unix)]
#[test]
fn string_elements_and_xlat_wrap_linear_addresses() {
    // SDM Vol 3A §3.4; Vol 2B §4.3 MOVS/REP/XLAT:
    // each element uses its segmented linear address, then the A20 gate.
    for a20 in [true, false] {
        let (vm, mut cpu) = linear_edge_fixture(b"\xf3\xa5", a20);
        cpu.cpu.legacy.segments[1].base = 0xffff_fffc;
        cpu.cpu.gpr[6] = 3;
        cpu.cpu.gpr[7] = 0x8000;
        cpu.cpu.gpr[1] = 2;
        vm.write_bytes(edge_physical(0xffff_ffff, a20), &[0x11])
            .unwrap();
        vm.write_bytes(0, &[0x22, 0x33, 0x44]).unwrap();
        for _ in 0..2 {
            assert_eq!(step(&vm, &mut cpu).steps(), 1);
        }
        let mut copied = [0; 4];
        vm.read_bytes(0x8000, &mut copied).unwrap();
        assert_eq!(copied, [0x11, 0x22, 0x33, 0x44]);
        assert_eq!(
            (cpu.cpu.gpr[1], cpu.cpu.gpr[6], cpu.cpu.gpr[7]),
            (0, 7, 0x8004)
        );
        let (vm, mut cpu) = linear_edge_fixture(b"\xd7", a20);
        cpu.cpu.legacy.segments[1].base = 0xffff_fffc;
        cpu.cpu.gpr[3] = 3;
        cpu.cpu.gpr[0] = 0xab01;
        vm.write_bytes(0, &[0x5a]).unwrap();
        assert_eq!(step(&vm, &mut cpu).steps(), 1);
        assert_eq!(cpu.cpu.gpr[0], 0xab5a);
    }
}

#[test]
fn real16_flag_frames_and_retirement_agree_when_tf_is_clear_cpu_model() {
    model_api::with_cpu_model(real16_flag_frames_and_retirement_agree_when_tf_is_clear);
}

#[test]
fn real16_flag_frames_and_retirement_agree_when_tf_is_clear() {
    // I286 PRM §5.2.2 p. 5-5; 80386 PRM §14.3 p. 14-3:
    // both paths use FLAGS:CS:IP frames and IRET; delivery itself retires nothing.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let code = b"\x9c\x9d\xcd\x01\x90";
        let (a, mut ac) = setup(code, None, 17);
        let (b, mut bc) = setup(code, Some(generation), 17);
        for vm in [&a, &b] {
            vm.write_bytes(0x1804, b"\xcf").unwrap();
        }
        // Keep flag images in the common subset (no IOPL/NT/TF).
        ac.cpu.flags.set_flags16(0x202);
        bc.cpu.flags.set_flags16(0x202);
        for _ in 0..5 {
            let ar = step(&a, &mut ac);
            let br = step(&b, &mut bc);
            assert_eq!(ar.steps(), br.steps());
            assert_eq!(exit_image(ar.stop()), exit_image(br.stop()));
            assert_eq!(image(&ac), image(&bc));
            assert_eq!(memory_image(&a), memory_image(&b));
            assert_eq!(ac.retired_instructions(), bc.retired_instructions());
        }
        assert_eq!(ac.retired_instructions(), 5);
    }
}

#[test]
fn application_ignores_tf_while_legacy_delays_until_the_successor_cpu_model() {
    model_api::with_cpu_model(application_ignores_tf_while_legacy_delays_until_the_successor);
}

#[test]
fn application_ignores_tf_while_legacy_delays_until_the_successor() {
    // Application Real16's exec_popf/exec_iret use Flags::set_flags16, which
    // ignores TF. Legacy follows I286 PRM §9.7.1 p. 9-14 and 80386 PRM
    // §12.3.1.4 p. 12-8. Pin the difference without changing Application.
    for generation in [CpuGeneration::I286, CpuGeneration::I386] {
        let code = b"\x9d\x90";
        let (a, mut ac) = setup(code, None, 17);
        let (b, mut bc) = setup(code, Some(generation), 17);
        for vm in [&a, &b] {
            vm.write_bytes(0x8000, b"\x02\x01").unwrap();
        }
        assert_eq!(step(&a, &mut ac).steps(), 1);
        assert_eq!(step(&b, &mut bc).steps(), 1);
        assert_eq!((ac.cpu.rip, bc.cpu.rip), (0x1001, 0x1001));
        assert_eq!((ac.cpu.rflags_system, bc.cpu.rflags_system), (0, 0x100));
        assert_eq!(step(&a, &mut ac).steps(), 1);
        assert_eq!(step(&b, &mut bc).steps(), 1);
        assert_eq!((ac.cpu.rip, bc.cpu.rip), (0x1002, 0x1804));
        assert_eq!(
            (ac.retired_instructions(), bc.retired_instructions()),
            (2, 2)
        );
        assert_eq!(bc.cpu.rflags_system & 0x100, 0);
        assert_eq!(&memory_image(&b)[0x7ffc..0x8002], &[2, 0x10, 0, 0, 2, 1]);
    }
}

#[test]
fn alias_enter_device_display_restarts_before_committing_frame() {
    // SDM Vol 2A §3.3 ENTER; Vol 1 §6.2.3; Vol 3A §§7.5–7.6.
    // 80386 PRM §17 ENTER pp. 17-58–17-59: word/dword frame values
    // are independent of the stack-address attribute.
    for (code, size) in [
        (&b"\xc8\x08\x00\x02"[..], 2u8),
        (&b"\x66\xc8\x08\x00\x02"[..], 4),
    ] {
        let (mut vm, mut cpu) = setup(code, None, 0x123456789abcdef);
        vm.unmap(0x4000, 0x1000).unwrap();
        vm.map(0x4000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        cpu.cpu.gpr[4] = 0x8000;
        cpu.cpu.gpr[5] = 0x4000 + u64::from(size);
        let initial = cpu.cpu.gpr;
        let mut before = [0; 16];
        vm.read_bytes(0x7ff0, &mut before).unwrap();
        let first = run_form(&vm, &mut cpu, code);
        assert!(
            matches!(first.stop(), StepStop::Exit(Exit::MmioRead { addr: 0x4000, size: actual, .. }) if *actual == size)
        );
        assert_eq!((first.steps(), cpu.retired_instructions()), (0, 0));
        assert_eq!(cpu.cpu.gpr, initial);
        let mut after = [0; 16];
        vm.read_bytes(0x7ff0, &mut after).unwrap();
        assert_eq!(after, before);
        cpu.complete_mmio_read(0x1234_5678);
        let done = run_form(&vm, &mut cpu, code);
        assert!(matches!(done.stop(), StepStop::Exit(Exit::BudgetExhausted)));
        assert_eq!((done.steps(), cpu.retired_instructions()), (1, 1));
        assert_eq!(cpu.cpu.gpr[4], 0x8000 - 3 * u64::from(size) - 8);
        assert_eq!(cpu.cpu.gpr[5], 0x8000 - u64::from(size));
    }
}

#[test]
fn alias_flags_image_uses_286_and_machine_images_follow_the_model() {
    // 80286 PRM Appendix B POPF p. B-86; §5.3 Table 5-3 p. 5-7:
    // real mode retains IOPL/NT from the reset FLAGS image 0002h.
    // SDM Vol 3B §25.17.2; Vol 2A §3.3 IRET, Vol 2B §4.3 POPF.
    for (code, size, iret) in [(&b"\x9d"[..], 2u8, false), (&b"\xcf"[..], 2, true)] {
        for generation in [None, Some(CpuGeneration::I286), Some(CpuGeneration::I386)] {
            let (vm, mut cpu) = setup(code, generation, 0x123456789abcdef);
            for (i, value) in if iret {
                vec![0x2000u32, 0, 0x7002]
            } else {
                vec![0x7002]
            }
            .into_iter()
            .enumerate()
            {
                vm.write_bytes(
                    0x8000 + i as u64 * u64::from(size),
                    &value.to_le_bytes()[..usize::from(size)],
                )
                .unwrap();
            }
            let done = run_form(&vm, &mut cpu, code);
            assert!(
                matches!(done.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                "{generation:?} {code:02x?}: {done:?}"
            );
            assert_eq!(
                cpu.cpu.rflags_system & 0x7000,
                if generation == Some(CpuGeneration::I386) {
                    0x7000
                } else {
                    0
                }
            );
            assert_eq!(
                cpu.cpu.gpr[4],
                0x8000 + u64::from(size) * if iret { 3 } else { 1 }
            );
        }
    }
}

#[test]
fn alias_far_frame_uses_selector_base_without_a20_or_retained_caches() {
    // 80286 PRM §5.1 pp. 5-1–5-2: a real segment base is selector << 4.
    // Decision-23 §6: the compatibility alias ignores A20 and retained caches.
    let mut vm = Vm::with_backend(VmConfig::flat(0x110000), Box::new(Interpreter));
    vm.set_cpu_mode(CpuMode::Real16);
    vm.map(0, 0x110000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, b"\x9a\x00\x20\x00\x10").unwrap();
    vm.write_bytes(0x2c, &[0xaa; 4]).unwrap();
    vm.write_bytes(0x10002c, &[0xcc; 4]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.ss = 0xffff;
    cpu.cpu.gpr[4] = 0x40;
    let mut hidden = cpu.cpu.legacy;
    assert_eq!(hidden.a20_enabled, 0);
    let done = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert!(matches!(done.stop(), StepStop::Exit(Exit::BudgetExhausted)));
    assert_eq!((done.steps(), cpu.retired_instructions()), (1, 1));
    assert_eq!(
        (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
        (0x1000, 0x2000, 0x3c)
    );
    // M4 publishes selector-derived caches; other retained state stays inert.
    hidden.segments = CpuState::legacy_real_entry([
        cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs,
    ])
    .legacy
    .segments;
    assert_eq!(cpu.cpu.legacy, hidden);
    let mut low = [0; 4];
    let mut high = [0; 4];
    vm.read_bytes(0x2c, &mut low).unwrap();
    vm.read_bytes(0x10002c, &mut high).unwrap();
    assert_eq!(low, [0xaa; 4]);
    assert_eq!(high, [0x05, 0x10, 0, 0]);
}

#[test]
fn alias_flag_stack_write_invalidates_decoded_successors() {
    // SDM Vol 2B §4.3 PUSHF: the image is 0002h with all writable flags clear.
    // Preserve the existing VM SMC guard when the stack overwrites later code.
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(Interpreter));
    vm.set_cpu_mode(CpuMode::Real16);
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, b"\x9c\x40\x40\xf4").unwrap();
    vm.write_bytes(0x3000, &[5]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[3] = 0x3000;
    cpu.cpu.gpr[4] = 0x1003;
    cpu.cpu.flags.set_flags16(2);
    let done = cpu.run_steps(&vm, 8, StepOptions::default()).unwrap();
    assert!(matches!(done.stop(), StepStop::Exit(Exit::Hlt)));
    assert_eq!(
        (done.steps(), cpu.retired_instructions(), cpu.cpu.gpr[0]),
        (3, 3, 5)
    );
}
