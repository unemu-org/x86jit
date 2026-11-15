//! NEC V20/V30 witnesses (decision-23 C4a, spec.md §17.10). Every page
//! reference is NEC uPD70108/uPD70116 (V20/V30) User's Manual, Stock No.
//! 500350, October 1986 ("UM"); the pages cited were checked as images.
//! The 8080 side of the mode-switch witnesses is a test fixture, not an 8080
//! implementation of this library (decision-23 §3).
use x86jit_core::{
    CpuModel, Exit, FpuModel, I8080Entry, I8080Handback, I8080HandbackError, I8080View,
    InterpreterBackend, ModelIsa, ModelSystem, PortDir, Prot, RegionKind, StepOptions, StepRun,
    StepStop, UnsupportedSystemOperation, Vcpu, Vm, VmConfig,
};

const MODELS: [CpuModel; 2] = [CpuModel::NecV20, CpuModel::NecV30];
const IP: u64 = 0x1200;

fn setup(model: CpuModel, code: &[u8]) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
    vm.set_cpu_model(model);
    vm.map(0, 0x120000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(IP, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = IP;
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}
fn run(vm: &Vm, cpu: &mut Vcpu, steps: u64) -> StepRun {
    cpu.run_steps(vm, steps, StepOptions::default()).unwrap()
}
fn step(vm: &Vm, cpu: &mut Vcpu, count: u64) {
    let r = run(vm, cpu, count);
    assert_eq!(r.steps(), count, "{:?}", r.stop());
    assert!(
        matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{:?}",
        r.stop()
    );
}
fn word(vm: &Vm, addr: u64) -> u64 {
    vm.mem.read(addr, 2).unwrap()
}
fn vector(vm: &Vm, n: u8, ip: u16, cs: u16) {
    vm.mem
        .write(u64::from(n) * 4, u64::from(ip) | (u64::from(cs) << 16), 4)
        .unwrap();
}
fn refused(stop: &StepStop) -> bool {
    matches!(stop, StepStop::Exit(Exit::UnknownInstruction { .. }))
}
fn unsupported(stop: &StepStop) -> Option<UnsupportedSystemOperation> {
    match stop {
        StepStop::Exit(Exit::UnsupportedSystemContext { operation, .. }) => Some(*operation),
        _ => None,
    }
}
// The 16-bit PSW image of a native-mode vCPU (UM §3 p. 3-3).
fn psw(cpu: &Vcpu) -> u16 {
    cpu.cpu.flags.to_flags16() | (cpu.cpu.rflags_system as u16 & 0x100) | 0xf000
}

#[test]
fn catalogue_entries_and_bus_siblings() {
    // UM §1 p. 1-1: the two differ in external bus width; §12 gives one
    // instruction set; Table 6-1 and §12 give per-processor clocks only.
    let a = CpuModel::NecV20.profile();
    let b = CpuModel::NecV30.profile();
    assert_eq!(a.isa, ModelIsa::NecV20);
    assert_eq!((a.isa, a.quirks), (b.isa, b.quirks));
    assert_ne!(a.semantics_id, b.semantics_id);
    for model in MODELS {
        let p = model.profile();
        assert_eq!(p.system, ModelSystem::None);
        assert_eq!(p.physical_bits, 20);
        assert_eq!(p.max_insn_len, None);
        assert!(p.timing.is_none());
        assert_eq!(p.fpu, &[FpuModel::Present, FpuModel::None]);
        let q = p.quirks;
        assert_eq!(q.real_flags_set_mask, 0xf000);
        assert_eq!(q.max_processor_interrupt, 5);
        assert_eq!(q.shift_count_mask, None);
        assert_eq!(q.rep_restart_prefixes, Some(3));
        assert!(q.divide_saves_next_ip && q.idiv_rejects_signed_minimum);
        assert!(q.wrap_segment_offsets && q.wrap_sequential_ip);
        assert!(!q.iret_delays_interrupts);
        assert!(q.segment_store_inhibits_interrupts && q.step_irq_conflict_refused);
        assert!(q.ei_delay_unconditional);
        for other in [
            CpuModel::I8088,
            CpuModel::I8086,
            CpuModel::I80286,
            CpuModel::I80386,
        ] {
            assert_ne!(p.semantics_id, other.profile().semantics_id);
        }
    }
    // The fields added for these entries keep the earlier models' values.
    for model in [CpuModel::I8088, CpuModel::I8086] {
        let q = model.profile().quirks;
        assert_eq!(q.rep_restart_prefixes, Some(1));
        assert!(q.iret_delays_interrupts);
        assert!(!q.segment_store_inhibits_interrupts && !q.step_irq_conflict_refused);
        assert!(!q.ei_delay_unconditional);
    }
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let q = model.profile().quirks;
        assert_eq!(q.rep_restart_prefixes, None);
        assert!(!q.iret_delays_interrupts);
        assert!(!q.segment_store_inhibits_interrupts && !q.step_irq_conflict_refused);
        assert!(!q.ei_delay_unconditional);
    }
}

#[test]
fn push_immediate_forms_store_a_word() {
    // UM p. 12-149: 6Ah sign-extends imm8; 68h stores imm16.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x6a\xfe\x68\x34\x12");
        step(&vm, &mut cpu, 2);
        assert_eq!(cpu.cpu.gpr[4], 0x7ffc);
        assert_eq!(word(&vm, 0x7ffe), 0xfffe);
        assert_eq!(word(&vm, 0x7ffc), 0x1234);
    }
}

#[test]
fn push_and_pop_register_sets() {
    // UM p. 12-148 PUSH R stores the SP of entry; p. 12-152 POP R skips it.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x60\x61");
        for (i, v) in [
            0x1111, 0x2222, 0x3333, 0x4444, 0x8000, 0x6666, 0x7777, 0x8888,
        ]
        .into_iter()
        .enumerate()
        {
            cpu.cpu.gpr[i] = v;
        }
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[4], 0x7ff0);
        let stored: Vec<u64> = (0..8).map(|i| word(&vm, 0x7ffe - 2 * i)).collect();
        // AW CW DW BW temp BP IX IY, from SP-2 downwards.
        assert_eq!(
            stored,
            [0x1111, 0x2222, 0x3333, 0x4444, 0x8000, 0x6666, 0x7777, 0x8888]
        );
        // Change every slot; POP R reloads all but the SP slot.
        for i in 0..8u64 {
            vm.mem.write(0x7ff0 + 2 * i, 0xa000 + i, 2).unwrap();
        }
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[4], 0x8000);
        // IY IX BP (SP skipped) BW DW CW AW
        assert_eq!(cpu.cpu.gpr[7], 0xa000);
        assert_eq!(cpu.cpu.gpr[6], 0xa001);
        assert_eq!(cpu.cpu.gpr[5], 0xa002);
        assert_eq!(cpu.cpu.gpr[3], 0xa004);
        assert_eq!(cpu.cpu.gpr[2], 0xa005);
        assert_eq!(cpu.cpu.gpr[1], 0xa006);
        assert_eq!(cpu.cpu.gpr[0], 0xa007);
    }
}

#[test]
fn push_register_set_wraps_inside_the_stack_segment() {
    // §10 Fig. 10-1 p. 10-1: sixteen-bit offsets (decision-23 §3 wrap row).
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x60");
        cpu.cpu.ss = 0x3000;
        cpu.cpu.gpr[4] = 0x0004;
        cpu.cpu.gpr[0] = 0xabcd;
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[4], 0xfff4);
        assert_eq!(word(&vm, 0x30002), 0xabcd);
        assert_eq!(vm.mem.read(0x30000, 2).unwrap(), cpu.cpu.gpr[1] & 0xffff);
        // AW CW at 0002h/0000h, then DW BW temp... from FFFEh down.
        assert_eq!(word(&vm, 0x3fffa), 0x0004);
        assert_eq!(vm.mem.read(0x40000, 1).unwrap(), 0);
    }
}

#[test]
fn signed_multiply_by_immediate() {
    // UM pp. 12-54/55: CY = V = 1 when the product exceeds 16 bits.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x6b\xc3\xf6\x69\xcb\x00\x01");
        cpu.cpu.gpr[3] = 3;
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[0] & 0xffff, 0xffe2);
        assert!(!cpu.cpu.flags.cf && !cpu.cpu.flags.of);
        cpu.cpu.gpr[3] = 0x100;
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[1] & 0xffff, 0);
        assert!(cpu.cpu.flags.cf && cpu.cpu.flags.of);
    }
}

#[test]
fn immediate_shift_counts_are_not_masked() {
    // UM p. 12-104 SHL reg,imm8: "temp <- imm8, while temp != 0" (no mask).
    for model in MODELS {
        for (code, ax, expected) in [
            (&b"\xc1\xe0\x20"[..], 1, 0),           // SHL AX,32
            (&b"\xc1\xe8\x11"[..], 0x8000, 0),      // SHR AX,17
            (&b"\xc0\xc0\x09"[..], 0x0081, 0x0003), // ROL AL,9
        ] {
            let (vm, mut cpu) = setup(model, code);
            cpu.cpu.gpr[0] = ax;
            step(&vm, &mut cpu, 1);
            assert_eq!(cpu.cpu.gpr[0] & 0xffff, expected, "{code:x?}");
        }
    }
    // The same encoding on the 80286 masks the count to five bits.
    let mut vm = Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
    vm.set_cpu_model(CpuModel::I80286);
    vm.map(0, 0x120000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(IP, b"\xc1\xe0\x20").unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = IP;
    cpu.cpu.gpr[0] = 1;
    step(&vm, &mut cpu, 1);
    assert_eq!(cpu.cpu.gpr[0] & 0xffff, 1);
}

#[test]
fn port_strings_transfer_one_element_per_handoff() {
    // UM pp. 12-28/29: INM stores at DS1:IY, OUTM reads DS0:IX; REP repeats.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\xf3\x6f\x6c\xf4");
        cpu.cpu.ds = 0x100;
        cpu.cpu.es = 0x200;
        cpu.cpu.gpr[1] = 2;
        cpu.cpu.gpr[2] = 0x3f8;
        cpu.cpu.gpr[6] = 0x10;
        cpu.cpu.gpr[7] = 0xffff;
        vm.mem.write(0x1010, 0x55aa_4433, 4).unwrap();
        for value in [0x4433, 0x55aa] {
            let r = run(&vm, &mut cpu, 10);
            let StepStop::Exit(Exit::PortIo {
                port,
                size,
                dir,
                value: out,
                ..
            }) = r.stop()
            else {
                panic!("{:?}", r.stop())
            };
            assert_eq!((*port, *size, *dir, *out), (0x3f8, 2, PortDir::Out, value));
        }
        assert_eq!((cpu.cpu.gpr[1], cpu.cpu.gpr[6]), (0, 0x14));
        let r = run(&vm, &mut cpu, 10);
        assert!(
            matches!(
                r.stop(),
                StepStop::Exit(Exit::PortIo {
                    dir: PortDir::In,
                    size: 1,
                    ..
                })
            ),
            "{:?}",
            r.stop()
        );
        cpu.complete_port_in(0xab);
        let r = run(&vm, &mut cpu, 10);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        assert_eq!(vm.mem.read(0x2000 + 0xffff, 1).unwrap(), 0xab);
        assert_eq!(cpu.cpu.gpr[7] & 0xffff, 0);
    }
}

#[test]
fn input_word_wraps_inside_its_segment() {
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x6d\xf4");
        cpu.cpu.es = 0x200;
        cpu.cpu.gpr[7] = 0xffff;
        let r = run(&vm, &mut cpu, 10);
        assert!(matches!(
            r.stop(),
            StepStop::Exit(Exit::PortIo {
                dir: PortDir::In,
                size: 2,
                ..
            })
        ));
        cpu.complete_port_in(0x1234);
        let r = run(&vm, &mut cpu, 10);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(vm.mem.read(0x2000 + 0xffff, 1).unwrap(), 0x34);
        assert_eq!(vm.mem.read(0x2000, 1).unwrap(), 0x12);
    }
}

#[test]
fn check_index_raises_break_5_with_the_next_pc() {
    // UM p. 12-170 CHKIND; §6 Table 6-1 p. 6-1 vector 5; §3 p. 3-1 PC.
    for model in MODELS {
        for (index, out) in [
            (0x10, false),
            (0x5, false),
            (0x25, false),
            (0x30, true),
            (4, true),
        ] {
            let (vm, mut cpu) = setup(model, b"\x62\x06\x00\x20\xf4");
            vector(&vm, 5, 0x400, 0);
            vm.write_bytes(0x400, b"\xf4").unwrap();
            vm.mem.write(0x2000, 0x0025_0005, 4).unwrap();
            cpu.cpu.gpr[0] = index;
            let r = run(&vm, &mut cpu, 2);
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{:?}",
                r.stop()
            );
            if out {
                assert_eq!(cpu.cpu.rip, 0x401);
                assert_eq!(word(&vm, 0x7ffa), IP + 4);
                assert_eq!(word(&vm, 0x7ffe), 0xf002);
            } else {
                assert_eq!(cpu.cpu.rip, IP + 5);
            }
        }
    }
}

#[test]
fn check_index_refuses_a_sign_dependent_outcome() {
    // UM p. 12-170 does not say whether the comparison is signed.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x62\x06\x00\x20");
        vm.mem.write(0x2000, 0x0010_fff0, 4).unwrap();
        cpu.cpu.gpr[0] = 5;
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 0);
        assert_eq!(
            unsupported(r.stop()),
            Some(UnsupportedSystemOperation::NecIndexComparison)
        );
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[4]), (IP, 0x8000));
    }
}

#[test]
fn prepare_and_dispose_follow_the_listed_operations() {
    // UM pp. 12-152/153; the level is not reduced modulo 32.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\xc8\x08\x00\x00\xc9");
        cpu.cpu.gpr[5] = 0x9000;
        step(&vm, &mut cpu, 1);
        assert_eq!((cpu.cpu.gpr[5], cpu.cpu.gpr[4]), (0x7ffe, 0x7ff6));
        assert_eq!(word(&vm, 0x7ffe), 0x9000);
        step(&vm, &mut cpu, 1);
        assert_eq!((cpu.cpu.gpr[5], cpu.cpu.gpr[4]), (0x9000, 0x8000));

        let (vm, mut cpu) = setup(model, b"\xc8\x02\x00\x03");
        cpu.cpu.gpr[5] = 0x9000;
        vm.mem.write(0x8ffe, 0x1111, 2).unwrap();
        vm.mem.write(0x8ffc, 0x2222, 2).unwrap();
        step(&vm, &mut cpu, 1);
        // BP, two copied frame pointers, then the new frame pointer.
        assert_eq!(word(&vm, 0x7ffe), 0x9000);
        assert_eq!(word(&vm, 0x7ffc), 0x1111);
        assert_eq!(word(&vm, 0x7ffa), 0x2222);
        assert_eq!(word(&vm, 0x7ff8), 0x7ffe);
        assert_eq!((cpu.cpu.gpr[5], cpu.cpu.gpr[4]), (0x7ffe, 0x7ff6));

        let (vm, mut cpu) = setup(model, b"\xc8\x00\x00\x21");
        cpu.cpu.gpr[5] = 0x9000;
        step(&vm, &mut cpu, 1);
        // 2 + 32 copies + 1 frame pointer words.
        assert_eq!(cpu.cpu.gpr[4], 0x8000 - 2 * 34);
        assert_eq!(word(&vm, 0x8000 - 2 * 34), 0x7ffe);
    }
}

#[test]
fn compatibility_aliases_execute_and_other_forms_are_refused() {
    // Decision-23 compatibility amendment: 82h, F6h/F7h /1, 83h /1,/4,/6.
    for model in MODELS {
        for (code, ax) in [
            (&b"\x82\xc0\x01"[..], 0x0101),
            (&b"\x83\xc8\xf0"[..], 0xfff0),
            (&b"\x83\xe0\x0f"[..], 0x0000),
            (&b"\x83\xf0\xff"[..], 0xfeff),
        ] {
            let (vm, mut cpu) = setup(model, code);
            cpu.cpu.gpr[0] = 0x0100;
            step(&vm, &mut cpu, 1);
            assert_eq!(cpu.cpu.gpr[0] & 0xffff, ax, "{code:x?}");
        }
        let (vm, mut cpu) = setup(model, b"\xf6\xc8\x01");
        cpu.cpu.gpr[0] = 1;
        step(&vm, &mut cpu, 1);
        assert!(!cpu.cpu.flags.zf);
        // C4b groups, undefined bytes, not-stated forms and later ISAs.
        for code in [
            &b"\x63\xc0"[..],
            b"\x64\xa6",
            b"\x65\xa6",
            b"\x66\x90",
            b"\x67\x90",
            b"\xd6",
            b"\xf1\x90",
            b"\x0f\x10\xc0",
            b"\x0f\x20",
            b"\x0f\x31\xc0",
            b"\x0f\x01\xc0",
            b"\xc0\xf0\x01",
            b"\xd1\xf0",
            b"\xff\xf8",
            b"\x8f\xc8",
            b"\xc6\xc8\x00",
            b"\xfe\xd0",
            b"\x62\xc0",
            b"\x54",
            b"\x5c",
            b"\xff\xf4",
            b"\x8f\xc4",
            b"\xd4\x08",
            b"\xd5\x10",
            b"\x2e\x0f\xff\x80",
            b"\xf2\x6c",
            b"\x26\x6c",
            b"\xf3\x90",
            b"\x66\x40",
            b"\x0f\xa2",
            b"\xf3\x26\x2e\x3e\x6e",
        ] {
            let (vm, mut cpu) = setup(model, code);
            let r = run(&vm, &mut cpu, 1);
            assert!(refused(r.stop()), "{code:x?}: {:?}", r.stop());
            assert_eq!((r.steps(), cpu.cpu.rip), (0, IP), "{code:x?}");
        }
    }
}

#[test]
fn psw_image_divide_and_wrap_rows() {
    // UM §3 p. 3-3 image; pp. 12-56–59 divide; §3 p. 3-1 PC counter.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x9c");
        cpu.cpu.flags.if_ = true;
        step(&vm, &mut cpu, 1);
        assert_eq!(word(&vm, 0x7ffe), 0xf202);
        for (code, ax, dx, divisor) in [
            (&b"\xf6\xfb"[..], 0xff80, 0, 1),
            (&b"\xf7\xfb"[..], 0x8000, 0xffff, 1),
            (&b"\xf7\xf3"[..], 1, 0, 0),
        ] {
            let (vm, mut cpu) = setup(model, code);
            vector(&vm, 0, 0x400, 0);
            vm.write_bytes(0x400, b"\xf4").unwrap();
            cpu.cpu.gpr[0] = ax;
            cpu.cpu.gpr[2] = dx;
            cpu.cpu.gpr[3] = divisor;
            let r = run(&vm, &mut cpu, 1);
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{:?}",
                r.stop()
            );
            assert_eq!(word(&vm, 0x7ffa), IP + code.len() as u64);
        }
        // IP wraps at FFFFh; a word operand at FFFFh wraps to offset 0.
        let (vm, mut cpu) = setup(model, b"");
        cpu.cpu.cs = 0x1000;
        cpu.cpu.rip = 0xffff;
        vm.write_bytes(0x1ffff, b"\x90").unwrap();
        vm.write_bytes(0x10000, b"\xa1\xff\xff").unwrap();
        vm.mem.write(0xffff + 0x2000 * 16, 0x34, 1).unwrap();
        vm.mem.write(0x2000 * 16, 0x12, 1).unwrap();
        cpu.cpu.ds = 0x2000;
        step(&vm, &mut cpu, 2);
        assert_eq!(cpu.cpu.gpr[0] & 0xffff, 0x1234);
        assert_eq!(cpu.cpu.rip, 3);
    }
}

#[test]
fn segment_store_holds_int_and_reti_does_not() {
    // UM §6 p. 6-4: INT is held after MOV r/m16,sreg; RETI is not listed.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x8c\xd8\x40\x40");
        vector(&vm, 0x20, 0x400, 0);
        vm.write_bytes(0x400, b"\xf4").unwrap();
        cpu.cpu.flags.if_ = true;
        step(&vm, &mut cpu, 1);
        cpu.inject_irq(0x20);
        assert!(!cpu.irq_ready());
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[0] & 0xffff, 1);
        assert!(cpu.irq_ready());

        // RETI that sets IE: the interrupt is taken before the next instruction.
        let (vm, mut cpu) = setup(model, b"\xcf");
        vector(&vm, 0x20, 0x400, 0);
        vm.write_bytes(0x400, b"\xf4").unwrap();
        vm.write_bytes(0x500, b"\x40\xf4").unwrap();
        vm.mem.write(0x8000, 0x0500, 2).unwrap();
        vm.mem.write(0x8002, 0, 2).unwrap();
        vm.mem.write(0x8004, 0xf202, 2).unwrap();
        step(&vm, &mut cpu, 1);
        assert!(cpu.irq_ready());
        cpu.inject_irq(0x20);
        let r = run(&vm, &mut cpu, 5);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[0] & 0xffff), (0x401, 0));
    }
    // The 8086 keeps its IRET delay (C2) and has no segment-store hold.
    let mut vm = Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
    vm.set_cpu_model(CpuModel::I8088);
    vm.map(0, 0x120000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(IP, b"\x8c\xd8\x40").unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = IP;
    cpu.cpu.gpr[4] = 0x8000;
    cpu.cpu.flags.if_ = true;
    step(&vm, &mut cpu, 1);
    cpu.inject_irq(0x20);
    assert!(cpu.irq_ready());
}

#[test]
fn interrupted_block_transfer_keeps_up_to_three_prefixes() {
    // UM §6 p. 6-4: one return address back per prefix, at most three.
    for model in MODELS {
        for (code, restart) in [
            (&b"\x26\x2e\xf3\xa4"[..], IP),
            (&b"\x26\x2e\x3e\xf3\xa4"[..], IP + 1),
        ] {
            let (vm, mut cpu) = setup(model, code);
            vector(&vm, 0x20, 0x400, 0);
            vm.write_bytes(0x400, b"\xf4").unwrap();
            cpu.cpu.flags.if_ = true;
            cpu.cpu.gpr[1] = 3;
            step(&vm, &mut cpu, 1);
            cpu.inject_irq(0x20);
            let r = run(&vm, &mut cpu, 1);
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{:?}",
                r.stop()
            );
            assert_eq!(word(&vm, 0x7ffa), restart, "{code:x?}");
            assert_eq!(cpu.cpu.gpr[1], 2);
        }
    }
}

#[test]
fn single_step_with_an_acceptable_int_is_refused() {
    // UM Table 6-1 p. 6-1 orders INT before the BRK flag but does not say
    // whether the trap is then taken at the handler: a typed refusal.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x8e\xd0\x40\x40");
        vector(&vm, 0x20, 0x400, 0);
        vector(&vm, 1, 0x500, 0);
        step(&vm, &mut cpu, 1);
        // MOV SS holds both events through INC AX; after it both are due.
        cpu.cpu.flags.if_ = true;
        cpu.cpu.rflags_system = 0x100;
        cpu.inject_irq(0x20);
        let r = run(&vm, &mut cpu, 3);
        assert_eq!(
            unsupported(r.stop()),
            Some(UnsupportedSystemOperation::NecStepWithInterrupt),
            "{:?}",
            r.stop()
        );
        assert_eq!((r.steps(), cpu.cpu.gpr[0] & 0xffff), (1, 1));
        assert!(cpu.has_pending_irq());
        // Without a queued INT the trap follows the instruction.
        let (vm, mut cpu) = setup(model, b"\x40\x40");
        vector(&vm, 1, 0x500, 0);
        cpu.cpu.rflags_system = 0x100;
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 1, "{:?}", r.stop());
        assert_eq!((cpu.cpu.rip, word(&vm, 0x7ffa)), (0x500, IP + 1));
    }
}

// ---------------------------------------------------------------------------
// The mode switch, native side (UM §8 pp. 8-1–8-5; §12 pp. 12-168–12-177).

const BRKEM_AT: u64 = 0x1200;
fn emulation_setup(model: CpuModel) -> (Vm, Vcpu) {
    // BRKEM 80h; vector 80h -> 2000:0100 (8080 code), CALLN 40h -> 6000:0200.
    let (vm, mut cpu) = setup(model, b"\x0f\xff\x80\xf4");
    vector(&vm, 0x80, 0x0100, 0x2000);
    vector(&vm, 0x40, 0x0200, 0x6000);
    vm.write_bytes(0x60200, b"\x40\xcf").unwrap(); // INC AX; RETI
    cpu.cpu.flags.if_ = true;
    cpu.cpu.flags.cf = true;
    cpu.cpu.ds = 0x4000;
    cpu.cpu.gpr[0] = 0xa111;
    cpu.cpu.gpr[1] = 0xc445;
    cpu.cpu.gpr[2] = 0xd667;
    cpu.cpu.gpr[3] = 0xb223;
    cpu.cpu.gpr[5] = 0x9000;
    (vm, cpu)
}
fn enter(vm: &Vm, cpu: &mut Vcpu) -> I8080View {
    let r = run(vm, cpu, 10);
    assert_eq!(r.steps(), 1);
    let StepStop::Exit(Exit::I8080Mode { view, entry, .. }) = r.stop() else {
        panic!("{:?}", r.stop())
    };
    assert_eq!(*entry, I8080Entry::Brkem { vector: 0x80 });
    *view
}
fn resume(vm: &Vm, cpu: &mut Vcpu) -> I8080View {
    let r = run(vm, cpu, 10);
    assert_eq!(r.steps(), 0, "{:?}", r.stop());
    let StepStop::Exit(Exit::I8080Mode { view, .. }) = r.stop() else {
        panic!("{:?}", r.stop())
    };
    *view
}

#[test]
fn brkem_saves_the_native_frame_and_maps_the_registers() {
    // UM p. 12-169 BRKEM; Fig. 8-1/8-2 p. 8-1 register map; rule R2.
    for model in MODELS {
        let (vm, mut cpu) = emulation_setup(model);
        let view = enter(&vm, &mut cpu);
        assert_eq!(word(&vm, 0x7ffa), BRKEM_AT + 3);
        assert_eq!(word(&vm, 0x7ffc), 0);
        assert_eq!(word(&vm, 0x7ffe), 0xf203);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (0x2000, 0x100, 0x7ffa)
        );
        assert_eq!(cpu.cpu.legacy.reserved & 3, 3);
        assert!(!cpu.cpu.flags.if_);
        assert_eq!(
            (view.a, view.bc, view.de, view.hl, view.sp, view.pc),
            (0x11, 0xc445, 0xd667, 0xb223, 0x9000, 0x100)
        );
        assert_eq!(view.flags, 0x03);
        assert!(!view.interrupt_enable && !view.ei_delay && !view.halted);
        assert_eq!((view.code_segment, view.data_segment), (0x2000, 0x4000));
        assert_eq!(view.code_address(0x100), 0x20100);
        assert_eq!(view.data_address(0xffff), 0x4ffff);
        // The boundary repeats with zero steps until the state is handed back.
        for _ in 0..3 {
            assert_eq!(resume(&vm, &mut cpu), view);
        }
        let r = run(&vm, &mut cpu, 0);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
    }
}

#[test]
fn budget_around_the_mode_switch() {
    // Decision-16 §2 order: budget, host request, delivery, readiness stop,
    // then the 8080-mode boundary; the exit itself costs no step.
    for model in MODELS {
        // BRKEM uses the last unit of the budget: exhaustion is reported first.
        let (vm, mut cpu) = emulation_setup(model);
        step(&vm, &mut cpu, 1);
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 0);
        assert!(matches!(
            r.stop(),
            StepStop::Exit(Exit::I8080Mode {
                entry: I8080Entry::Brkem { vector: 0x80 },
                ..
            })
        ));
        // Two steps: BRKEM, then the zero-step exit.
        let (vm, mut cpu) = emulation_setup(model);
        vm.write_bytes(BRKEM_AT - 1, b"\x90\x0f\xff\x80").unwrap();
        cpu.cpu.rip = BRKEM_AT - 1;
        let r = run(&vm, &mut cpu, 2);
        assert_eq!(r.steps(), 2);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
        let (vm, mut cpu) = emulation_setup(model);
        vm.write_bytes(BRKEM_AT - 1, b"\x90\x0f\xff\x80").unwrap();
        cpu.cpu.rip = BRKEM_AT - 1;
        let r = run(&vm, &mut cpu, 3);
        assert_eq!(r.steps(), 2);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::I8080Mode { .. })));
        assert_eq!(cpu.retired_instructions(), 2);
        // RETEM is one unit; with a budget of N the native code continues.
        for (budget, steps, rip) in [(1, 1, BRKEM_AT + 3), (2, 2, BRKEM_AT + 4)] {
            let (vm, mut cpu) = emulation_setup(model);
            let view = enter(&vm, &mut cpu);
            vm.write_bytes(0x20100, b"\xed\xfd").unwrap();
            cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
            let before = cpu.retired_instructions();
            let r = run(&vm, &mut cpu, budget);
            assert_eq!(r.steps(), steps, "{:?}", r.stop());
            if budget == 2 {
                assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
            } else {
                assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
            }
            assert_eq!(cpu.retired_instructions() - before, steps);
            assert_eq!(cpu.cpu.rip, rip);
        }
    }
}

#[test]
fn retem_restores_the_brkem_frame_and_write_disables_md() {
    // UM p. 12-177: PC, PS, PSW restored; the 8080 flags do not carry back,
    // the mapped registers do (Fig. 8-7 p. 8-5).
    for model in MODELS {
        let (vm, mut cpu) = emulation_setup(model);
        let mut view = enter(&vm, &mut cpu);
        vm.write_bytes(0x20180, b"\xed\xfd").unwrap();
        view.a = 0x5a;
        view.hl = 0x1234;
        view.sp = 0x8ffe;
        view.pc = 0x180;
        view.flags = 0xd7;
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        step(&vm, &mut cpu, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (0, BRKEM_AT + 3, 0x8000)
        );
        assert_eq!(psw(&cpu), 0xf203);
        assert_eq!(cpu.cpu.legacy.reserved, 0);
        assert_eq!(
            (cpu.cpu.gpr[0], cpu.cpu.gpr[3], cpu.cpu.gpr[5]),
            (0xa15a, 0x1234, 0x8ffe)
        );
        assert!(matches!(
            cpu.hand_back_i8080(&view, I8080Handback::Boundary),
            Err(I8080HandbackError::NotInI8080Mode)
        ));
        // MD is write-disabled again: RETI with MD = 0 stays native.
        vm.write_bytes(BRKEM_AT + 3, b"\xcf").unwrap();
        vm.mem.write(0x8000, 0x1300, 2).unwrap();
        vm.mem.write(0x8002, 0, 2).unwrap();
        vm.mem.write(0x8004, 0x0002, 2).unwrap();
        vm.write_bytes(0x1300, b"\xf4").unwrap();
        let r = run(&vm, &mut cpu, 2);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        assert_eq!(cpu.cpu.legacy.reserved, 0);
    }
}

#[test]
fn calln_and_reti_round_trip_through_native_code() {
    // UM p. 12-176 CALLN: PSW (MD = 0 saved), PS, PC; MD <- 1; rule R2.
    // p. 12-168 RETI restores MD while MD is write-enabled.
    for model in MODELS {
        let (vm, mut cpu) = emulation_setup(model);
        let mut view = enter(&vm, &mut cpu);
        vm.write_bytes(0x20110, b"\xed\xed\x40").unwrap();
        view.pc = 0x110;
        view.interrupt_enable = true;
        view.ei_delay = true;
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        assert!(!cpu.irq_ready());
        step(&vm, &mut cpu, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (0x6000, 0x200, 0x7ff4)
        );
        assert_eq!(word(&vm, 0x7ff4), 0x113);
        assert_eq!(word(&vm, 0x7ff6), 0x2000);
        assert_eq!(word(&vm, 0x7ff8), 0x7203);
        assert_eq!(psw(&cpu), 0xf003);
        assert_eq!(cpu.cpu.legacy.reserved & 3, 2);
        // CALLN itself consumes the 8080 EI delay: with IE forced on right
        // after it, an INT would be acceptable at this boundary.
        cpu.cpu.flags.if_ = true;
        assert!(cpu.irq_ready());
        cpu.cpu.flags.if_ = false;
        let r = run(&vm, &mut cpu, 10);
        assert_eq!(r.steps(), 2);
        let StepStop::Exit(Exit::I8080Mode { view, entry, .. }) = r.stop() else {
            panic!("{:?}", r.stop())
        };
        assert_eq!(*entry, I8080Entry::Reti);
        assert_eq!(view.pc, 0x113);
        assert!(view.interrupt_enable);
        // CALLN completed after the EI, so the delay is spent (review case b).
        assert!(!view.ei_delay);
        assert_eq!(view.a, 0x12);
        assert_eq!(cpu.cpu.legacy.reserved & 3, 3);
    }
}

#[test]
fn interrupt_in_8080_mode_is_serviced_natively() {
    // UM §8 p. 8-4: "the interrupt process is performed the same as in
    // native mode"; §6 p. 6-1 entry; RETI returns to emulation; §9 p. 9-1.
    for model in MODELS {
        let (vm, mut cpu) = emulation_setup(model);
        vector(&vm, 0x41, 0x0300, 0x6100);
        vm.write_bytes(0x61300, b"\xcf").unwrap();
        let mut view = enter(&vm, &mut cpu);
        view.interrupt_enable = true;
        view.halted = true;
        view.pc = 0x121;
        cpu.hand_back_i8080(&view, I8080Handback::Boundary).unwrap();
        assert!(cpu.irq_ready());
        cpu.inject_irq(0x41);
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 1);
        assert_eq!(word(&vm, 0x7ff4), 0x121);
        assert_eq!(word(&vm, 0x7ff8), 0x7203);
        let r = run(&vm, &mut cpu, 10);
        assert_eq!(r.steps(), 0);
        let StepStop::Exit(Exit::I8080Mode { view, entry, .. }) = r.stop() else {
            panic!("{:?}", r.stop())
        };
        assert_eq!(*entry, I8080Entry::Reti);
        assert!(!view.halted && view.interrupt_enable && !view.ei_delay);
        assert_eq!(view.pc, 0x121);
    }
}

#[test]
fn delay_halt_and_readiness_are_exchanged_verbatim() {
    for model in MODELS {
        let (vm, mut cpu) = emulation_setup(model);
        let mut view = enter(&vm, &mut cpu);
        view.interrupt_enable = true;
        cpu.hand_back_i8080(&view, I8080Handback::Boundary).unwrap();
        // The readiness stop precedes the 8080-mode boundary.
        let r = cpu
            .run_steps(
                &vm,
                10,
                StepOptions::default().with_stop_when_irq_ready(true),
            )
            .unwrap();
        assert!(matches!(r.stop(), StepStop::IrqReady), "{:?}", r.stop());
        // A host request precedes it too, and is consumed by that call.
        cpu.exit_request().request_exit();
        let r = run(&vm, &mut cpu, 10);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Interrupted)));
        assert_eq!(r.steps(), 0);
        assert_eq!(cpu.cpu.legacy.reserved & 1, 1);
        // An 8080 EI delay holds a queued INT; the exit never clears it.
        view.ei_delay = true;
        view.halted = true;
        cpu.hand_back_i8080(&view, I8080Handback::Boundary).unwrap();
        assert!(!cpu.irq_ready());
        cpu.inject_irq(0x41);
        for _ in 0..2 {
            let back = resume(&vm, &mut cpu);
            assert!(back.ei_delay && back.halted && back.interrupt_enable);
        }
        assert!(cpu.has_pending_irq());
    }
}

#[test]
fn hand_back_rejections_change_nothing() {
    for model in MODELS {
        let (vm, mut cpu) = emulation_setup(model);
        let view = enter(&vm, &mut cpu);
        let before = cpu.cpu.clone();
        let mut moved = view;
        moved.code_segment = 0x2001;
        assert_eq!(
            cpu.hand_back_i8080(&moved, I8080Handback::Boundary),
            Err(I8080HandbackError::SegmentChanged)
        );
        let mut moved = view;
        moved.data_segment = 0;
        assert_eq!(
            cpu.hand_back_i8080(&moved, I8080Handback::Boundary),
            Err(I8080HandbackError::SegmentChanged)
        );
        let mut halted = view;
        halted.halted = true;
        halted.a = 0;
        assert_eq!(
            cpu.hand_back_i8080(&halted, I8080Handback::EdEscape),
            Err(I8080HandbackError::HaltedStop)
        );
        // EDh is the escape: a stop on it is EdEscape, never Refused.
        let mut edited = view;
        edited.a = 0;
        assert_eq!(
            cpu.hand_back_i8080(&edited, I8080Handback::Refused { opcode: 0xed }),
            Err(I8080HandbackError::RefusedEscape)
        );
        assert_eq!(cpu.cpu.gpr, before.gpr);
        assert_eq!(cpu.cpu.legacy, before.legacy);
    }
    // A vCPU in native mode, of any model, rejects the hand-back.
    let (vm, mut other) = emulation_setup(CpuModel::NecV30);
    let view = enter(&vm, &mut other);
    for model in [CpuModel::NecV20, CpuModel::I8088, CpuModel::I80286] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
        vm.set_cpu_model(model);
        let mut cpu = vm.new_vcpu();
        assert_eq!(
            cpu.hand_back_i8080(&view, I8080Handback::Boundary),
            Err(I8080HandbackError::NotInI8080Mode)
        );
    }
}

#[test]
fn refusals_at_the_mode_switch() {
    for model in MODELS {
        // An 8080-core refusal stays at its PS:PC with MD = 0.
        let (vm, mut cpu) = emulation_setup(model);
        let view = enter(&vm, &mut cpu);
        cpu.hand_back_i8080(&view, I8080Handback::Refused { opcode: 0xdd })
            .unwrap();
        for _ in 0..2 {
            let r = run(&vm, &mut cpu, 10);
            assert_eq!(r.steps(), 0);
            let StepStop::Exit(Exit::UnknownInstruction {
                addr, bytes, len, ..
            }) = r.stop()
            else {
                panic!("{:?}", r.stop())
            };
            assert_eq!((*addr, bytes[0], *len), (0x100, 0xdd, 1));
        }
        cpu.hand_back_i8080(&view, I8080Handback::Boundary).unwrap();
        resume(&vm, &mut cpu);
        // ED with a byte other than FD or ED (UM pp. 12-176/177).
        vm.write_bytes(0x20100, b"\xed\x00").unwrap();
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        let r = run(&vm, &mut cpu, 10);
        let StepStop::Exit(Exit::UnknownInstruction { bytes, len, .. }) = r.stop() else {
            panic!("{:?}", r.stop())
        };
        assert_eq!((bytes[0], bytes[1], *len, r.steps()), (0xed, 0x00, 2, 0));
        assert_eq!(cpu.cpu.legacy.reserved & 1, 1);

        // BRK = 1 at an 8080-mode boundary is not described.
        let (vm, mut cpu) = emulation_setup(model);
        let view = enter(&vm, &mut cpu);
        cpu.cpu.rflags_system = 0x100;
        cpu.hand_back_i8080(&view, I8080Handback::Boundary).unwrap();
        let r = run(&vm, &mut cpu, 10);
        assert_eq!(
            unsupported(r.stop()),
            Some(UnsupportedSystemOperation::NecEmulationSingleStep)
        );

        // RETEM whose frame has MD = 0 (UM p. 12-177 describes only BRKEM's).
        let (vm, mut cpu) = emulation_setup(model);
        let view = enter(&vm, &mut cpu);
        vm.write_bytes(0x20100, b"\xed\xfd").unwrap();
        vm.mem.write(0x7ffe, 0x7002, 2).unwrap();
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        let r = run(&vm, &mut cpu, 10);
        assert_eq!(
            unsupported(r.stop()),
            Some(UnsupportedSystemOperation::NecRetemToEmulation)
        );
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip, r.steps()), (0x2000, 0x100, 0));

        // Nested BRKEM in a native call from emulation (UM p. 8-5), and POP
        // PSW that would clear MD there (p. 12-151).
        for (code, operation) in [
            (
                &b"\x0f\xff\x80"[..],
                UnsupportedSystemOperation::NecNestedEmulation,
            ),
            (
                &b"\x68\x02\x00\x9d"[..],
                UnsupportedSystemOperation::NecPswModeWrite,
            ),
        ] {
            let (vm, mut cpu) = emulation_setup(model);
            vm.write_bytes(0x60200, code).unwrap();
            let view = enter(&vm, &mut cpu);
            vm.write_bytes(0x20100, b"\xed\xed\x40").unwrap();
            cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
            let r = run(&vm, &mut cpu, 10);
            assert_eq!(unsupported(r.stop()), Some(operation), "{:?}", r.stop());
            assert_eq!(cpu.cpu.cs, 0x6000);
            assert_eq!(cpu.cpu.legacy.reserved & 3, 2);
        }
    }
}

#[test]
fn psw_restores_keep_md_outside_emulation() {
    // UM pp. 12-151/168: outside BRKEM..RETEM MD stays 1 whatever the stack holds.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x68\x02\x00\x9d\x9c\xcf");
        vm.write_bytes(0x500, b"\xf4").unwrap();
        step(&vm, &mut cpu, 3);
        assert_eq!(word(&vm, 0x7ffe), 0xf002);
        vm.mem.write(0x7ffe, 0x0500, 2).unwrap();
        vm.mem.write(0x8000, 0, 2).unwrap();
        vm.mem.write(0x8002, 0x0002, 2).unwrap();
        let r = run(&vm, &mut cpu, 2);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        assert_eq!(cpu.cpu.legacy.reserved, 0);
    }
}

#[test]
fn escape_fetch_failure_leaves_the_escape_pending() {
    for model in MODELS {
        let mut vm = Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
        vm.set_cpu_model(model);
        vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(BRKEM_AT, b"\x0f\xff\x80").unwrap();
        vector(&vm, 0x80, 0x0100, 0x2000);
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = BRKEM_AT;
        cpu.cpu.gpr[4] = 0x8000;
        let view = enter(&vm, &mut cpu);
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        let r = run(&vm, &mut cpu, 10);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::UnmappedMemory { .. })),
            "{:?}",
            r.stop()
        );
        assert_eq!(r.steps(), 0);
        vm.map(0x20000, 0x10000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.write_bytes(0x20100, b"\xed\xfd").unwrap();
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.rip, BRKEM_AT + 3);
    }
}

#[test]
fn stores_from_the_8080_side_invalidate_native_translations() {
    // An adapter writes through Vm::write_bytes; native blocks lifted before
    // BRKEM are re-fetched (spec.md §10 self-modifying code).
    for model in MODELS {
        let (vm, mut cpu) = emulation_setup(model);
        vm.write_bytes(BRKEM_AT + 3, b"\x40\xf4").unwrap();
        cpu.cpu.rip = BRKEM_AT + 3;
        let r = run(&vm, &mut cpu, 10);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(cpu.cpu.gpr[0] & 0xffff, 0xa112);
        cpu.cpu.rip = BRKEM_AT;
        let view = enter(&vm, &mut cpu);
        // The 8080 program stores 48h (DEC AX) over the native INC AX.
        vm.write_bytes(BRKEM_AT + 3, &[0x48]).unwrap();
        vm.write_bytes(0x20100, b"\xed\xfd").unwrap();
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        let r = run(&vm, &mut cpu, 10);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        assert_eq!(cpu.cpu.gpr[0] & 0xffff, 0xa111);
    }
}

#[test]
fn reset_image_returns_to_native_mode() {
    // UM §8 p. 8-4: RESET aborts emulation; §7 p. 7-1 reset values. The
    // write-enable latch is cleared (ASSUMED; the page names only MD).
    for model in MODELS {
        let (vm, mut cpu) = emulation_setup(model);
        enter(&vm, &mut cpu);
        cpu.reset_execution_state(Default::default());
        cpu.cpu.legacy.reserved = 0;
        cpu.cpu.cs = 0xffff;
        cpu.cpu.rip = 0;
        cpu.cpu.ss = 0;
        cpu.cpu.ds = 0;
        cpu.cpu.es = 0;
        cpu.cpu.flags.if_ = false;
        cpu.cpu.flags.df = false;
        cpu.cpu.rflags_system = 0;
        vm.write_bytes(0xffff0, b"\xf4").unwrap();
        let r = run(&vm, &mut cpu, 1);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
    }
}

#[test]
fn other_models_never_produce_the_8080_mode_exit() {
    // BRKEM is 0F FF imm8 only on the NEC entries; elsewhere the bytes keep
    // their earlier meaning (8086: refused; 286/386: UD0 #UD).
    for model in [CpuModel::I8088, CpuModel::I8086] {
        let (vm, mut cpu) = {
            let mut vm = Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
            vm.set_cpu_model(model);
            vm.map(0, 0x120000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(IP, b"\x0f\xff\x80").unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = IP;
            (vm, cpu)
        };
        let r = run(&vm, &mut cpu, 1);
        assert!(refused(r.stop()), "{:?}", r.stop());
        cpu.cpu.legacy.reserved = 1;
        cpu.cpu.rip = IP;
        vm.write_bytes(IP, b"\x90").unwrap();
        let r = run(&vm, &mut cpu, 1);
        assert!(
            matches!(
                r.stop(),
                StepStop::Exit(Exit::UnsupportedSystemContext { .. })
            ),
            "{:?}",
            r.stop()
        );
    }
}

// ---------------------------------------------------------------------------
// Witnesses added with the C4a review: three rules the first version got
// wrong, the hand-back rejections, the host stop seen from the 8080 core,
// and rules the suite did not pin.

#[test]
fn ei_with_ie_already_set_holds_int() {
    // UM §6 p. 6-4: INT is not acknowledged "between the EI instruction and
    // the next instruction"; p. 12-172: "The system does not enter the
    // interrupt-enable state until executing the instruction immediately
    // after EI." Neither page conditions this on IE having been 0.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\xfb\x40\xf4");
        vector(&vm, 0x20, 0x400, 0);
        vm.write_bytes(0x400, b"\xf4").unwrap();
        cpu.cpu.flags.if_ = true;
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 1);
        cpu.inject_irq(0x20);
        assert!(
            !cpu.irq_ready(),
            "{model:?}: INT acceptable right after EI with IE = 1"
        );
        step(&vm, &mut cpu, 1);
        assert!(cpu.irq_ready());
    }
    // The 8086 keeps the SDM Vol 2B §4.3 STI rule: no hold when IF was 1.
    let mut vm = Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
    vm.set_cpu_model(CpuModel::I8088);
    vm.map(0, 0x120000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(IP, b"\xfb\x40").unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = IP;
    cpu.cpu.gpr[4] = 0x8000;
    cpu.cpu.flags.if_ = true;
    step(&vm, &mut cpu, 1);
    cpu.inject_irq(0x20);
    assert!(cpu.irq_ready());
}

#[test]
fn reti_into_emulation_with_brk_is_refused_before_commit() {
    // A RETI that would re-enter 8080 mode with BRK = 1 at its start: the
    // break after it is not described for emulation mode. The refusal has
    // no guest effect and repeats.
    for model in MODELS {
        // BRKEM 80h -> 2000:0100 (8080), CALLN 40h -> 6000:0200 (native).
        let (vm, mut cpu) = setup(model, b"\x0f\xff\x80\xf4");
        vector(&vm, 0x80, 0x0100, 0x2000);
        vector(&vm, 0x40, 0x0200, 0x6000);
        vector(&vm, 1, 0x0500, 0);
        vm.write_bytes(0x500, b"\xf4").unwrap();
        // PUSH F102h (MD = 1, BRK = 1); POP PSW; RETI.
        vm.write_bytes(0x60200, b"\x68\x02\xf1\x9d\xcf").unwrap();
        vm.write_bytes(0x20100, b"\xed\xed\x40").unwrap();
        let r = run(&vm, &mut cpu, 10);
        let StepStop::Exit(Exit::I8080Mode { view, .. }) = r.stop() else {
            panic!("{:?}", r.stop())
        };
        let view = *view;
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        // CALLN, PUSH, POP PSW, then RETI (BRK = 1 at its start).
        let r = run(&vm, &mut cpu, 10);
        assert_eq!(
            unsupported(r.stop()),
            Some(UnsupportedSystemOperation::NecEmulationSingleStep),
            "{:?}",
            r.stop()
        );
        assert_eq!(r.steps(), 3);
        let md0 = cpu.cpu.legacy.reserved & 1 != 0;
        let (cs, ip) = (cpu.cpu.cs, cpu.cpu.rip);
        // A typed refusal has no guest effect: RETI did not commit.
        let committed = md0 && (cs, ip) == (0x2000, 0x103);
        assert_eq!((cs, ip, cpu.cpu.gpr[4]), (0x6000, 0x204, 0x7ff4));
        assert_eq!(cpu.cpu.legacy.reserved & 3, 2);
        // And it repeats on the next call.
        let r = run(&vm, &mut cpu, 10);
        let repeated = matches!(
            r.stop(),
            StepStop::Exit(Exit::UnsupportedSystemContext {
                operation: UnsupportedSystemOperation::NecEmulationSingleStep,
                ..
            })
        );
        let lost = matches!(
            r.stop(),
            StepStop::Exit(Exit::I8080Mode {
                entry: I8080Entry::Reti,
                ..
            })
        );
        assert!(
            !committed && repeated && r.steps() == 0,
            "{model:?}: committed {committed}, repeated {repeated}, trap lost {lost}"
        );
    }
}

#[test]
fn brk_traps_and_brkem_with_brk_is_refused() {
    // BRK = 1 at the start of BRK imm8 and of BRKEM imm8. Both clear BRK
    // (p. 12-167; rule R2), and §6 p. 6-3 takes the break after the current
    // instruction. BRK imm8 takes it; for BRKEM, which also leaves native
    // mode, no page says, so it is a typed refusal before commit
    // (decision-23 §3).
    for model in MODELS {
        // BRK 20h with BRK = 1.
        let (vm, mut cpu) = setup(model, b"\xcd\x20");
        vector(&vm, 0x20, 0x0400, 0);
        vector(&vm, 1, 0x0500, 0);
        vm.write_bytes(0x400, b"\x90\xf4").unwrap();
        vm.write_bytes(0x500, b"\xf4").unwrap();
        cpu.cpu.rflags_system = 0x100;
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 1, "{:?}", r.stop());
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x500), "{model:?} BRK 20h");
        // BRKEM 80h with BRK = 1.
        let (vm, mut cpu) = setup(model, b"\x0f\xff\x80");
        vector(&vm, 0x80, 0x0100, 0x2000);
        vector(&vm, 1, 0x0500, 0);
        vm.write_bytes(0x500, b"\xf4").unwrap();
        cpu.cpu.rflags_system = 0x100;
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(
            unsupported(r.stop()),
            Some(UnsupportedSystemOperation::NecEmulationSingleStep),
            "{model:?} BRKEM 80h: {:?}",
            r.stop()
        );
        assert_eq!(r.steps(), 0);
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]), (0, IP, 0x8000));
        assert_eq!(cpu.cpu.legacy.reserved, 0);
        assert_eq!(cpu.cpu.rflags_system, 0x100);
        assert_eq!(word(&vm, 0x7ffa) | word(&vm, 0x7ffc) | word(&vm, 0x7ffe), 0);
        assert_eq!(cpu.retired_instructions(), 0);
    }
}

#[test]
fn frames_and_vectors_are_read_at_escape_time() {
    // x80 review: frames and vectors are read when RETEM/CALLN execute.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x0f\xff\x80\xf4");
        vector(&vm, 0x80, 0x0100, 0x2000);
        vector(&vm, 0x40, 0x0200, 0x6000);
        vm.write_bytes(0x20100, b"\xed\xfd").unwrap();
        vm.write_bytes(0x1300, b"\xf4").unwrap();
        let r = run(&vm, &mut cpu, 10);
        let StepStop::Exit(Exit::I8080Mode { view, .. }) = r.stop() else {
            panic!("{:?}", r.stop())
        };
        let view = *view;
        // The 8080 side rewrites the saved PC of the BRKEM frame.
        vm.write_bytes(0x7ffa, &[0x00, 0x13]).unwrap();
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 1);
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x1300));

        let (vm, mut cpu) = setup(model, b"\x0f\xff\x80\xf4");
        vector(&vm, 0x80, 0x0100, 0x2000);
        vector(&vm, 0x40, 0x0200, 0x6000);
        vm.write_bytes(0x20100, b"\xed\xed\x40").unwrap();
        let r = run(&vm, &mut cpu, 10);
        let StepStop::Exit(Exit::I8080Mode { view, .. }) = r.stop() else {
            panic!("{:?}", r.stop())
        };
        let view = *view;
        // The 8080 side rewrites the CALLN vector.
        vector(&vm, 0x40, 0x0280, 0x6100);
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 1);
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0x6100, 0x280));
    }
}

fn enter_at(model: CpuModel, emu: &[u8], at: u16) -> (Vm, Vcpu, I8080View) {
    let (vm, mut cpu) = setup(model, b"\x0f\xff\x80\xf4");
    vector(&vm, 0x80, at, 0x2000);
    vector(&vm, 0x40, 0x0200, 0x6000);
    vector(&vm, 0x41, 0x0300, 0x6100);
    vm.write_bytes(0x61300, b"\xcf").unwrap();
    for (i, b) in emu.iter().enumerate() {
        let off = at.wrapping_add(i as u16);
        vm.write_bytes(0x20000 + u64::from(off), &[*b]).unwrap();
    }
    cpu.cpu.flags.df = true;
    cpu.cpu.flags.of = true;
    let r = run(&vm, &mut cpu, 10);
    let StepStop::Exit(Exit::I8080Mode { view, .. }) = r.stop() else {
        panic!("{:?}", r.stop())
    };
    let view = *view;
    (vm, cpu, view)
}

#[test]
fn hand_back_keeps_v_and_dir() {
    // The view carries the five 8080 flags only; V and DIR stay native.
    for model in MODELS {
        let (_vm, mut cpu, view) = enter_at(model, b"\xed\xfd", 0x100);
        cpu.hand_back_i8080(&view, I8080Handback::Boundary).unwrap();
        assert!(cpu.cpu.flags.df && cpu.cpu.flags.of, "{model:?} V/DIR lost");
    }
}

#[test]
fn escape_fetch_wraps_inside_ps() {
    // ED at PS:FFFFh, FD at PS:0000h: the escape wraps inside PS (S10).
    for model in MODELS {
        let (vm, mut cpu, view) = enter_at(model, b"\xed\xfd", 0xffff);
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 1, "{model:?} {:?}", r.stop());
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, IP + 3));
    }
}

#[test]
fn delivery_drops_a_pending_escape() {
    // An INT delivered at the boundary of a pending EdEscape drops the
    // escape; RETI returns to the ED (contract draft §2.4).
    for model in MODELS {
        let (vm, mut cpu, mut view) = enter_at(model, b"\xed\xfd", 0x100);
        view.interrupt_enable = true;
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        cpu.inject_irq(0x41);
        let r = run(&vm, &mut cpu, 10);
        let StepStop::Exit(Exit::I8080Mode { view, entry, .. }) = r.stop() else {
            panic!("{:?}", r.stop())
        };
        assert_eq!((*entry, view.pc, r.steps()), (I8080Entry::Reti, 0x100, 1));
    }
}

#[test]
fn host_request_is_visible_while_the_8080_core_runs() {
    // An adapter ORs ExitRequest::is_requested into its 8080 stop test. A
    // request made outside run_steps is visible without being consumed, and
    // the next call still reports it.
    for model in MODELS {
        let (vm, mut cpu) = emulation_setup(model);
        let view = enter(&vm, &mut cpu);
        let request = cpu.exit_request();
        assert!(!request.is_requested());
        let other = request.clone();
        std::thread::spawn(move || other.request_exit())
            .join()
            .unwrap();
        for _ in 0..2 {
            assert!(request.is_requested());
        }
        cpu.hand_back_i8080(&view, I8080Handback::Boundary).unwrap();
        let r = run(&vm, &mut cpu, 10);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Interrupted)),
            "{:?}",
            r.stop()
        );
        assert_eq!(r.steps(), 0);
        assert!(!request.is_requested());
        assert_eq!(resume(&vm, &mut cpu), view);
    }
}

#[test]
fn hand_back_is_rejected_while_mmio_is_in_flight() {
    // A RETEM frame read that stopped on MMIO is unfinished: a hand-back
    // then would drop the escape and leave the retry latched.
    for model in MODELS {
        let mut vm = Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
        vm.set_cpu_model(model);
        vm.map(0, 0x70000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x70000, 0x10000, Prot::RW, RegionKind::Trap)
            .unwrap();
        vm.map(0x80000, 0xa0000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.write_bytes(BRKEM_AT, b"\x0f\xff\x80").unwrap();
        vector(&vm, 0x80, 0x0100, 0x2000);
        vm.write_bytes(0x20100, b"\xed\xfd").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = BRKEM_AT;
        cpu.cpu.gpr[4] = 0x8000;
        let view = enter(&vm, &mut cpu);
        // The native stack now lies in the trapped region.
        cpu.cpu.ss = 0x7000;
        cpu.cpu.gpr[4] = 0;
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        let r = run(&vm, &mut cpu, 10);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::MmioRead { .. })),
            "{:?}",
            r.stop()
        );
        let before = cpu.cpu.clone();
        for how in [
            I8080Handback::Boundary,
            I8080Handback::EdEscape,
            I8080Handback::Refused { opcode: 0xdd },
        ] {
            assert_eq!(
                cpu.hand_back_i8080(&view, how),
                Err(I8080HandbackError::InFlight)
            );
        }
        cpu.complete_mmio_read(0x34);
        assert_eq!(
            cpu.hand_back_i8080(&view, I8080Handback::Boundary),
            Err(I8080HandbackError::InFlight)
        );
        assert_eq!(cpu.cpu.gpr, before.gpr);
        assert_eq!(cpu.cpu.legacy, before.legacy);
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (before.cs, before.rip));
    }
}

// ---------------------------------------------------------------------------
// Contract vectors of the sibling 8080-family project x80, format
// x80-v20-contract-2, copied verbatim from x80 commit 2d5a535
// (x80-v20-contract/vectors/mode-switch.txt). Here x86jit is the native side
// in place of that project's mock; the 8080 side is the small fixture below,
// which implements only the opcodes the vectors use (Intel 8080 semantics,
// as the vectors' README cites them). Differences from the mock, each named
// where it applies:
// - The service routines are native code: HLT, then edits equal to the mock's
//   (AL C7h, CW ABCDh, IX DEADh, PSW saved XOR 0ED5h with MD = 1, BRK kept),
//   then RETI. The routine's PUSH leaves one word below the service frame
//   (3000:7FF2), which the memory comparison adds.
// - The NMI cases are not ported: x86jit has no NMI input (xj-50w.19).
const VECTORS: &str = r#"# x80-v20-contract-2; hexadecimal numbers; native word order:
# AW BW CW DW BP SP IX IY PS SS DS0 DS1 PC PSW
# Option B (decision-9 amendment, 2026-11-15): the native side decodes the
# bytes after ED, so `access` counts only x80's bus and never an escape read.
format x80-v20-contract-2
# The initial PSW F292 has BRK = 0: the native core refuses BRKEM with BRK = 1
# (decision-9 amendment, 2026-11-15). Format 1 started from F392.
initial A111 B223 C445 D667 9000 8000 1357 2468 1000 3000 4000 5000 2345 F292
brkem 80
# After BRKEM, IE = 0 (ASSUMPTION R2, decision-9 amendment) and BRK = 0
# (precondition), so the emulation PSW starts as 7092. R2's BRK clear has no
# observable effect here. Cases that need IE = 1 execute an 8080 EI;
# where an INT must be acceptable, a NOP or HLT then expires the EI delay.
# Vector table: NMI=02, CALLN=40, INT=41, BRKEM=80.
memory 0000 0008 00 04 00 62
memory 0000 0100 00 02 00 60
memory 0000 0104 00 03 00 61
memory 0000 0200 00 01 00 20
# Common 8080 prefix, 9 instructions, ends at 2000:0111:
# MVI A,5A; LXI B,1234; LXI D,5678; LXI H,4000; MVI M,A6;
# MOV A,M; PUSH B; POP H; STC. A ends A6, NOT 5A.
memory 2000 0100 3E 5A 01 34 12 11 78 56 21 00 40 36 A6 7E C5 E1 37
# Expected writes common to every scenario (data and emulated stack).
changed 4000 4000 A6
changed 4000 8FFE 34 12

case brkem_retem
memory 2000 0111 ED FD
run 0009
# After BRKEM: IE = 0 (ASSUMPTION R2), BRK = 0 (precondition), MD = 0;
# the prefix set CY.
emulation 7093
until retem
final A1A6 1234 1234 5678 9000 8000 1357 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
access 2000 4000 000A 0008 0001 0001 0002 0002
end

case calln_reti_retem
# EI makes the saved IE = 1, so CALLN's IE clear (ASSUMPTION R2) is visible.
memory 2000 0111 FB ED ED 40 ED FD
until serviced
# NEC p. 12-168: RETI restores the full saved PSW, not live FC46.
emulation 7293
until retem
final A1C7 1234 ABCD 5678 9000 8000 DEAD 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
frame calln 3000 7FF4 0115 2000 7293
service A1A6 1234 1234 5678 9000 7FF4 1357 2468 6000 3000 4000 5000 0200 F093
routine FC46
access 2000 4000 000C 0008 0001 0001 0002 0002
end

case int_enabled_running
memory 2000 0111 FB 00 00 ED FD
run 000B
signal 1 0 0 0 41 serviced
emulation 7293
until retem
final A1C7 1234 ABCD 5678 9000 8000 DEAD 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
frame int 3000 7FF4 0113 2000 7293
service A1A6 1234 1234 5678 9000 7FF4 1357 2468 6100 3000 4000 5000 0300 F093
routine FC46
access 2000 4000 000D 0008 0001 0001 0002 0002
end

case int_disabled_running
memory 2000 0111 F3 00 ED FD
run 000A
signal 1 0 0 0 41 budget
until retem
final A1A6 1234 1234 5678 9000 8000 1357 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
access 2000 4000 000C 0008 0001 0001 0002 0002
end

case nmi_running
memory 2000 0111 F3 00 ED FD
run 000A
signal 0 1 0 0 00 serviced
emulation 7093
until retem
final A1C7 1234 ABCD 5678 9000 8000 DEAD 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
frame nmi 3000 7FF4 0112 2000 7093
service A1A6 1234 1234 5678 9000 7FF4 1357 2468 6200 3000 4000 5000 0400 F093
routine FE46
access 2000 4000 000C 0008 0001 0001 0002 0002
end

case hlt_int_disabled
memory 2000 0111 F3 76 00 ED FD
until halt
signal 1 0 0 0 41 wake
until retem
final A1A6 1234 1234 5678 9000 8000 1357 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
access 2000 4000 000D 0008 0001 0001 0002 0002
end

case hlt_int_enabled
# HLT completes after EI, so the EI delay has expired in standby.
memory 2000 0111 FB 76 00 ED FD
until halt
signal 1 0 0 0 41 serviced
emulation 7293
until retem
final A1C7 1234 ABCD 5678 9000 8000 DEAD 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
frame int 3000 7FF4 0113 2000 7293
service A1A6 1234 1234 5678 9000 7FF4 1357 2468 6100 3000 4000 5000 0300 F093
routine FC46
access 2000 4000 000D 0008 0001 0001 0002 0002
end

case hlt_nmi
memory 2000 0111 F3 76 00 ED FD
until halt
signal 0 1 0 0 00 serviced
emulation 7093
until retem
final A1C7 1234 ABCD 5678 9000 8000 DEAD 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
frame nmi 3000 7FF4 0113 2000 7093
service A1A6 1234 1234 5678 9000 7FF4 1357 2468 6200 3000 4000 5000 0400 F093
routine FE46
access 2000 4000 000D 0008 0001 0001 0002 0002
end

case reset_running
# EI makes IE = 1 at the boundary, so RESET's IE clear (NEC p. 7-1) is visible.
memory 2000 0111 FB ED FD
run 000A
signal 0 0 0 1 00 reset
# Undefined reset fields retained solely by fixture policy (NEC p. 7-1).
final A1A6 1234 1234 5678 9000 7FFA 1357 2468 FFFF 0000 0000 0000 0000 F093
frame brkem 3000 7FFA 2345 1000 F292
access 2000 4000 000A 0008 0001 0001 0002 0002
end

case int_low_running
memory 2000 0111 FB 00 00 ED FD
run 000B
signal 0 0 0 0 41 budget
until retem
final A1A6 1234 1234 5678 9000 8000 1357 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
access 2000 4000 000D 0008 0001 0001 0002 0002
end

case int_held_running
memory 2000 0111 FB 00 00 ED FD
run 000B
signal 1 0 1 0 41 budget
until retem
final A1A6 1234 1234 5678 9000 8000 1357 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
access 2000 4000 000D 0008 0001 0001 0002 0002
end

case nmi_over_int_and_hold
memory 2000 0111 FB 00 00 ED FD
run 000B
signal 1 1 1 0 41 serviced
emulation 7293
until retem
final A1C7 1234 ABCD 5678 9000 8000 DEAD 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
frame nmi 3000 7FF4 0113 2000 7293
service A1A6 1234 1234 5678 9000 7FF4 1357 2468 6200 3000 4000 5000 0400 F093
routine FC46
access 2000 4000 000D 0008 0001 0001 0002 0002
end

# NEC p. 9-1 does not restrict the instruction after a masked HLT wake.
case hlt_int_disabled_retem
memory 2000 0111 F3 76 ED FD
until halt
signal 1 0 0 0 41 retem
final A1A6 1234 1234 5678 9000 8000 1357 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
access 2000 4000 000C 0008 0001 0001 0002 0002
end

case hlt_int_disabled_calln
memory 2000 0111 F3 76 ED ED 40 ED FD
until halt
signal 1 0 0 0 41 serviced
emulation 7093
until retem
final A1C7 1234 ABCD 5678 9000 8000 DEAD 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
frame calln 3000 7FF4 0116 2000 7093
service A1A6 1234 1234 5678 9000 7FF4 1357 2468 6000 3000 4000 5000 0200 F093
routine FE46
access 2000 4000 000D 0008 0001 0001 0002 0002
end

# decision-9 amendment, 2026-11-15, ASSUMED (CALLN_CONSUMES_EI_DELAY): no EI
# delay survives a CALLN excursion, and CALLN itself consumes it. On a native
# core the routine's own RETI would also end it. This case witnesses that
# constant and that the adapter carries no 8080 latch across the excursion:
# after RETI the INT is serviced at the first 8080 boundary, before the NOP
# at 0115.
case calln_consumes_ei_delay
memory 2000 0111 FB ED ED 40 00 ED FD
until serviced
emulation 7293
signal 1 0 0 0 41 serviced
emulation 7293
until retem
final A1C7 1234 ABCD 5678 9000 8000 DEAD 2468 1000 3000 4000 5000 2345 F292
frame brkem 3000 7FFA 2345 1000 F292
frame calln 3000 7FF4 0115 2000 7293
frame int 3000 7FF4 0115 2000 7293
service A1A6 1234 1234 5678 9000 7FF4 1357 2468 6000 3000 4000 5000 0200 F093
service A1C7 1234 ABCD 5678 9000 7FF4 DEAD 2468 6100 3000 4000 5000 0300 F093
routine FC46
routine FC46
access 2000 4000 000D 0008 0001 0001 0002 0002
end
"#;
const NOT_PORTED: [&str; 3] = ["nmi_running", "hlt_nmi", "nmi_over_int_and_hold"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Budget,
    Halt,
    Retem,
    Serviced,
    Wake,
    Reset,
}
#[derive(Clone, Copy, Default)]
struct Lines {
    int: bool,
    hold: bool,
    reset: bool,
    vector: u8,
}
enum Core {
    Budget,
    Halt,
    Stop,
    Unsupported(u8),
}

// Intel 8080 subset: NOP, MVI A, LXI B/D/H, MVI M, MOV A,M, PUSH B, POP H,
// STC, EI, DI, HLT. EI arms a one-instruction delay that any other completed
// instruction clears (x80 fixture ASSUMPTION EMULATION_EI_DELAY).
#[derive(Clone, Copy, Debug, Default)]
struct I8080 {
    a: u8,
    f: u8,
    b: u8,
    c: u8,
    d: u8,
    e: u8,
    h: u8,
    l: u8,
    sp: u16,
    pc: u16,
    ie: bool,
    ei_delay: bool,
    halted: bool,
}
impl I8080 {
    fn load(view: &I8080View) -> Self {
        Self {
            a: view.a,
            f: view.flags,
            b: (view.bc >> 8) as u8,
            c: view.bc as u8,
            d: (view.de >> 8) as u8,
            e: view.de as u8,
            h: (view.hl >> 8) as u8,
            l: view.hl as u8,
            sp: view.sp,
            pc: view.pc,
            ie: view.interrupt_enable,
            ei_delay: view.ei_delay,
            halted: view.halted,
        }
    }
    fn store(&self, view: &mut I8080View) {
        view.a = self.a;
        view.flags = self.f;
        view.bc = u16::from_be_bytes([self.b, self.c]);
        view.de = u16::from_be_bytes([self.d, self.e]);
        view.hl = u16::from_be_bytes([self.h, self.l]);
        view.sp = self.sp;
        view.pc = self.pc;
        view.interrupt_enable = self.ie;
        view.ei_delay = self.ei_delay;
        view.halted = self.halted;
    }
}

struct Port {
    vm: Vm,
    cpu: Vcpu,
    view: I8080View,
    i: I8080,
    // opcode fetch, operand fetch, data read/write, stack read/write
    counts: [u64; 6],
    frames: Vec<(&'static str, u16, u16, [u16; 3])>,
    services: Vec<[u16; 14]>,
    routines: Vec<u16>,
    ended: bool,
}

fn physical(segment: u16, offset: u16) -> u64 {
    ((u64::from(segment) << 4) + u64::from(offset)) & 0xf_ffff
}
// Native mode payload: HLT; MOV SI,SP; SS: PUSH [SI+4]; SS: XOR [SI-2],0ED5h;
// SS: OR [SI-2],8000h; POPF; MOV AL,C7h; MOV CX,ABCDh; MOV SI,DEADh; RETI.
const ROUTINE: &[u8] = b"\xf4\x89\xe6\x36\xff\x74\x04\x36\x81\x74\xfe\xd5\x0e\x36\x81\x4c\xfe\x00\x80\x9d\xb0\xc7\xb9\xcd\xab\xbe\xad\xde\xcf";

impl Port {
    fn native(&self) -> [u16; 14] {
        let c = &self.cpu.cpu;
        let md = if c.legacy.reserved & 1 == 0 {
            0x8000
        } else {
            0
        };
        let psw = c.flags.to_flags16() | (c.rflags_system as u16 & 0x100) | 0x7000 | md;
        [
            c.gpr[0] as u16,
            c.gpr[3] as u16,
            c.gpr[1] as u16,
            c.gpr[2] as u16,
            c.gpr[5] as u16,
            c.gpr[4] as u16,
            c.gpr[6] as u16,
            c.gpr[7] as u16,
            c.cs,
            c.ss,
            c.ds,
            c.es,
            c.rip as u16,
            psw,
        ]
    }
    fn read(&mut self, segment: u16, offset: u16, slot: usize) -> u8 {
        self.counts[slot] += 1;
        self.vm.mem.read(physical(segment, offset), 1).unwrap() as u8
    }
    fn write(&mut self, offset: u16, value: u8, slot: usize) {
        self.counts[slot] += 1;
        let ds = self.view.data_segment;
        self.vm.write_bytes(physical(ds, offset), &[value]).unwrap();
    }
    fn fetch(&mut self, slot: usize) -> u8 {
        let pc = self.i.pc;
        self.i.pc = pc.wrapping_add(1);
        let ps = self.view.code_segment;
        self.read(ps, pc, slot)
    }
    // The 8080 core polls `stop` once per boundary, before halt or fetch.
    fn core(&mut self, steps: u64, stop: bool) -> Core {
        for _ in 0..steps {
            if stop {
                return Core::Stop;
            }
            if self.i.halted {
                return Core::Halt;
            }
            let at = self.i.pc;
            let ds = self.view.data_segment;
            let opcode = self.fetch(0);
            match opcode {
                0x00 => {}
                0x3e => self.i.a = self.fetch(1),
                0x01 | 0x11 | 0x21 => {
                    let lo = self.fetch(1);
                    let hi = self.fetch(1);
                    match opcode {
                        0x01 => (self.i.b, self.i.c) = (hi, lo),
                        0x11 => (self.i.d, self.i.e) = (hi, lo),
                        _ => (self.i.h, self.i.l) = (hi, lo),
                    }
                }
                0x36 => {
                    let v = self.fetch(1);
                    let hl = u16::from_be_bytes([self.i.h, self.i.l]);
                    self.write(hl, v, 3);
                }
                0x7e => {
                    let hl = u16::from_be_bytes([self.i.h, self.i.l]);
                    self.i.a = self.read(ds, hl, 2);
                }
                0xc5 => {
                    let (b, c) = (self.i.b, self.i.c);
                    self.write(self.i.sp.wrapping_sub(1), b, 5);
                    self.write(self.i.sp.wrapping_sub(2), c, 5);
                    self.i.sp = self.i.sp.wrapping_sub(2);
                }
                0xe1 => {
                    self.i.l = self.read(ds, self.i.sp, 4);
                    self.i.h = self.read(ds, self.i.sp.wrapping_add(1), 4);
                    self.i.sp = self.i.sp.wrapping_add(2);
                }
                0x37 => self.i.f |= 1,
                0xfb => self.i.ie = true,
                0xf3 => self.i.ie = false,
                0x76 => self.i.halted = true,
                other => {
                    self.i.pc = at;
                    return Core::Unsupported(other);
                }
            }
            self.i.ei_delay = opcode == 0xfb;
        }
        Core::Budget
    }
    fn hand_back(&mut self, how: I8080Handback) {
        self.i.store(&mut self.view);
        self.cpu.hand_back_i8080(&self.view, how).unwrap();
    }
    // Every return to the native core goes through the exchange: the vCPU is
    // the only home of the state between 8080 runs.
    fn reenter(&mut self) {
        let r = run(&self.vm, &mut self.cpu, 64);
        assert_eq!(r.steps(), 0, "{:?}", r.stop());
        let StepStop::Exit(Exit::I8080Mode { view, entry, .. }) = r.stop() else {
            panic!("{:?}", r.stop())
        };
        assert_eq!(*entry, I8080Entry::Resume);
        self.view = *view;
        self.i = I8080::load(view);
    }
    fn frame(&mut self, source: &'static str) {
        let c = &self.cpu.cpu;
        let (ss, sp) = (c.ss, c.gpr[4] as u16);
        let words = [0, 2, 4].map(|d| {
            self.vm
                .mem
                .read(physical(ss, sp.wrapping_add(d)), 2)
                .unwrap() as u16
        });
        self.frames.push((source, ss, sp, words));
    }
    // From the routine's entry to RETI back into 8080 mode.
    fn routine(&mut self, source: &'static str, entry: [u16; 14]) {
        self.frame(source);
        self.services.push(entry);
        let r = run(&self.vm, &mut self.cpu, 8);
        assert_eq!(r.steps(), 8, "{:?}", r.stop());
        self.routines.push(self.native()[13]);
        let r = run(&self.vm, &mut self.cpu, 64);
        assert_eq!(r.steps(), 1, "{:?}", r.stop());
        let StepStop::Exit(Exit::I8080Mode { view, entry, .. }) = r.stop() else {
            panic!("{:?}", r.stop())
        };
        assert_eq!(*entry, I8080Entry::Reti);
        self.view = *view;
        self.i = I8080::load(view);
    }
    fn run(&mut self, steps: u64, lines: Lines) -> Outcome {
        assert!(!self.ended);
        // UM §6 p. 6-2: INT at a boundary with IE = 1 and no NMI or hold;
        // the adapter decides acceptance from the 8080 latches, and x86jit's
        // irq_ready must agree after the hand-back.
        let accepted = lines.int && self.i.ie && !self.i.ei_delay && !lines.hold;
        // UM §8 p. 8-2, §9 p. 9-1: a masked INT wakes the 8080 standby.
        let wake = self.i.halted && lines.int && !self.i.ie;
        match self.core(steps, lines.reset || accepted || wake) {
            Core::Budget => {
                self.hand_back(I8080Handback::Boundary);
                self.reenter();
                Outcome::Budget
            }
            Core::Halt => {
                self.hand_back(I8080Handback::Boundary);
                self.reenter();
                Outcome::Halt
            }
            Core::Stop if lines.reset => {
                // UM §8 p. 8-4, §7 p. 7-1: copy out, then a native reset.
                self.hand_back(I8080Handback::Boundary);
                let c = &mut self.cpu;
                c.reset_execution_state(Default::default());
                c.cpu.legacy.reserved = 0;
                (c.cpu.cs, c.cpu.rip, c.cpu.ss, c.cpu.ds, c.cpu.es) = (0xffff, 0, 0, 0, 0);
                c.cpu.flags.if_ = false;
                c.cpu.flags.df = false;
                c.cpu.rflags_system = 0;
                self.ended = true;
                Outcome::Reset
            }
            Core::Stop if accepted => {
                self.hand_back(I8080Handback::Boundary);
                assert!(self.cpu.irq_ready());
                self.cpu.inject_irq(lines.vector);
                // Delivery (zero steps), then the routine's HLT.
                let r = run(&self.vm, &mut self.cpu, 1);
                assert!(
                    matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                    "{:?}",
                    r.stop()
                );
                let mut entry = self.native();
                entry[12] -= 1;
                self.routine("int", entry);
                Outcome::Serviced
            }
            Core::Stop => {
                self.i.halted = false;
                match self.run(1, Lines { ..lines }) {
                    Outcome::Budget => Outcome::Wake,
                    other => other,
                }
            }
            Core::Unsupported(0xed) => {
                self.hand_back(I8080Handback::EdEscape);
                let r = run(&self.vm, &mut self.cpu, 1);
                assert_eq!(r.steps(), 1, "{:?}", r.stop());
                if self.cpu.cpu.legacy.reserved & 2 == 0 {
                    self.ended = true;
                    return Outcome::Retem;
                }
                let entry = self.native();
                let r = run(&self.vm, &mut self.cpu, 1);
                assert!(
                    matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                    "{:?}",
                    r.stop()
                );
                self.routine("calln", entry);
                Outcome::Serviced
            }
            Core::Unsupported(opcode) => panic!("8080 fixture refused {opcode:02x}"),
        }
    }
    fn until(&mut self, expected: Outcome) {
        for _ in 0..64 {
            let outcome = self.run(1, Lines::default());
            if outcome == expected {
                return;
            }
            assert!(
                matches!(outcome, Outcome::Budget | Outcome::Serviced),
                "{outcome:?}"
            );
        }
        panic!("vector exceeded the fixture cap");
    }
}

fn hex(token: &str) -> u16 {
    u16::from_str_radix(token, 16).unwrap()
}
fn patch(image: &mut [u8], tokens: &[&str]) {
    let segment = hex(tokens[0]);
    let offset = hex(tokens[1]);
    for (i, byte) in tokens[2..].iter().enumerate() {
        image[physical(segment, offset.wrapping_add(i as u16)) as usize] = hex(byte) as u8;
    }
}
fn sections(name: &str) -> Vec<Vec<&'static str>> {
    let mut common = Vec::new();
    let mut selected = Vec::new();
    let mut current: Option<&str> = None;
    for line in VECTORS.lines() {
        let tokens: Vec<_> = line.split('#').next().unwrap().split_whitespace().collect();
        match tokens.as_slice() {
            [] => {}
            ["case", id] => current = Some(id),
            ["end"] => current = Some(""),
            _ if current.is_none() => common.push(tokens),
            _ if current == Some(name) => selected.push(tokens),
            _ => {}
        }
    }
    assert!(!selected.is_empty(), "missing vector {name}");
    common.into_iter().chain(selected).collect()
}

// The native machine of one vector before BRKEM: memory image, registers and
// the initial PSW, which `psw` replaces when given.
fn vector_machine(
    model: CpuModel,
    records: &[Vec<&str>],
    psw: Option<u16>,
) -> (Vm, Vcpu, Vec<u8>, u8) {
    let mut image = vec![0xcc_u8; 1 << 20];
    let mut initial = None;
    let mut vector = None;
    for tokens in records {
        match tokens.as_slice() {
            ["format", "x80-v20-contract-2"] => {}
            ["initial", w @ ..] => initial = Some(w.iter().map(|t| hex(t)).collect::<Vec<_>>()),
            ["brkem", n] => vector = Some(hex(n) as u8),
            ["memory", data @ ..] => patch(&mut image, data),
            _ => {}
        }
    }
    let w = initial.unwrap();
    // Native code of this port: BRKEM before the continuation PC, and the
    // routines at the CALLN (40h) and INT (41h) vectors' targets.
    let at = physical(w[8], w[12].wrapping_sub(3)) as usize;
    image[at..at + 3].copy_from_slice(&[0x0f, 0xff, vector.unwrap()]);
    for (segment, offset) in [(0x6000, 0x200), (0x6100, 0x300)] {
        let at = physical(segment, offset) as usize;
        image[at..at + ROUTINE.len()].copy_from_slice(ROUTINE);
    }
    let mut vm = Vm::with_backend(VmConfig::flat(1 << 20), Box::new(InterpreterBackend));
    vm.set_cpu_model(model);
    vm.map(0, 1 << 20, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0, &image).unwrap();
    let mut cpu = vm.new_vcpu();
    {
        let c = &mut cpu.cpu;
        for (index, word) in [0, 3, 1, 2, 5, 4, 6, 7].into_iter().zip(&w) {
            c.gpr[index] = u64::from(*word);
        }
        (c.cs, c.ss, c.ds, c.es) = (w[8], w[9], w[10], w[11]);
        c.rip = u64::from(w[12].wrapping_sub(3));
        let psw = psw.unwrap_or(w[13]);
        c.flags.set_flags16(psw);
        c.rflags_system = u64::from(psw & 0x100);
    }
    (vm, cpu, image, vector.unwrap())
}

fn run_vector(model: CpuModel, name: &str) {
    let records = sections(name);
    let (vm, mut cpu, image, vector) = vector_machine(model, &records, None);
    let mut expected = image.clone();
    let r = run(&vm, &mut cpu, 64);
    assert_eq!(r.steps(), 1, "{:?}", r.stop());
    let StepStop::Exit(Exit::I8080Mode { view, entry, .. }) = r.stop() else {
        panic!("{:?}", r.stop())
    };
    assert_eq!(*entry, I8080Entry::Brkem { vector });
    let view = *view;
    let mut port = Port {
        vm,
        cpu,
        view,
        i: I8080::load(&view),
        counts: [0; 6],
        frames: Vec::new(),
        services: Vec::new(),
        routines: Vec::new(),
        ended: false,
    };
    port.frame("brkem");
    let mut frames = Vec::new();
    let mut services = Vec::new();
    let mut routines = Vec::new();
    let mut seen = (false, false);
    for tokens in &records {
        match tokens.as_slice() {
            ["format", "x80-v20-contract-2"] | ["initial", ..] | ["brkem", _] | ["memory", ..] => {}
            ["run", n] => assert_eq!(
                port.run(u64::from(hex(n)), Lines::default()),
                Outcome::Budget
            ),
            ["until", what] => port.until(match *what {
                "halt" => Outcome::Halt,
                "retem" => Outcome::Retem,
                "serviced" => Outcome::Serviced,
                other => panic!("{other}"),
            }),
            ["emulation", psw] => {
                assert!(!port.ended);
                assert_eq!(port.native()[13], hex(psw), "{name}");
                assert_eq!(u16::from(port.i.f), hex(psw) & 0xff, "{name}");
            }
            ["signal", int, nmi, hold, reset, vector, outcome] => {
                assert_eq!(*nmi, "0", "NMI is not ported");
                let lines = Lines {
                    int: *int == "1",
                    hold: *hold == "1",
                    reset: *reset == "1",
                    vector: hex(vector) as u8,
                };
                let expected = match *outcome {
                    "budget" => Outcome::Budget,
                    "serviced" => Outcome::Serviced,
                    "wake" => Outcome::Wake,
                    "reset" => Outcome::Reset,
                    "retem" => Outcome::Retem,
                    other => panic!("{other}"),
                };
                assert_eq!(port.run(1, lines), expected, "{name}");
            }
            ["final", w @ ..] => {
                seen.0 = true;
                assert!(port.ended);
                let w: Vec<u16> = w.iter().map(|t| hex(t)).collect();
                assert_eq!(port.native().as_slice(), w.as_slice(), "{name} final");
            }
            ["changed", data @ ..] => patch(&mut expected, data),
            ["frame", source, ss, sp, pc, ps, psw] => {
                let words = [hex(pc), hex(ps), hex(psw)];
                let (ss, sp) = (hex(ss), hex(sp));
                for (i, word) in words.iter().enumerate() {
                    let [lo, hi] = word.to_le_bytes();
                    expected[physical(ss, sp.wrapping_add(2 * i as u16)) as usize] = lo;
                    expected[physical(ss, sp.wrapping_add(2 * i as u16 + 1)) as usize] = hi;
                }
                frames.push((*source, ss, sp, words));
            }
            ["service", w @ ..] => services.push(w.iter().map(|t| hex(t)).collect::<Vec<_>>()),
            ["routine", psw] => routines.push(hex(psw)),
            ["access", ps, ds0, counts @ ..] => {
                seen.1 = true;
                assert_eq!(
                    (port.view.code_segment, port.view.data_segment),
                    (hex(ps), hex(ds0))
                );
                let want: Vec<u64> = counts.iter().map(|t| u64::from(hex(t))).collect();
                assert_eq!(port.counts.as_slice(), want.as_slice(), "{name} access");
            }
            other => panic!("unknown record {other:?}"),
        }
    }
    assert!(seen.0 && seen.1);
    let observed: Vec<_> = port.frames.iter().map(|f| (f.0, f.1, f.2, f.3)).collect();
    assert_eq!(observed, frames, "{name} frames");
    let entries: Vec<Vec<u16>> = port.services.iter().map(|s| s.to_vec()).collect();
    assert_eq!(entries, services, "{name} services");
    assert_eq!(port.routines, routines, "{name} routines");
    if let Some(last) = port.routines.last() {
        // The routine's PUSH of the edited PSW, one word below its frame.
        let [lo, hi] = last.to_le_bytes();
        expected[physical(0x3000, 0x7ff2) as usize] = lo;
        expected[physical(0x3000, 0x7ff3) as usize] = hi;
    }
    let mut actual = vec![0u8; 1 << 20];
    port.vm.read_bytes(0, &mut actual).unwrap();
    assert_eq!(
        actual.iter().zip(&expected).position(|(a, b)| a != b),
        None,
        "{name}: first differing physical address"
    );
}

#[test]
fn x80_contract_vectors_on_the_native_side() {
    let mut names = Vec::new();
    for line in VECTORS.lines() {
        if let Some(name) = line.strip_prefix("case ") {
            names.push(name.trim());
        }
    }
    assert_eq!(names.len(), 15);
    for model in MODELS {
        let mut ported = 0;
        for name in &names {
            if !NOT_PORTED.contains(name) {
                run_vector(model, name);
                ported += 1;
            }
        }
        assert_eq!(ported, 12);
    }
}

#[test]
fn x80_initial_psw_with_brk_refuses_brkem() {
    // Format 1 of the source started from PSW F392 (BRK = 1); format 2 starts
    // from F292. BRKEM from F392 is a typed refusal (decision-23 §3) with no
    // guest effect.
    for model in MODELS {
        let records = sections("brkem_retem");
        let (vm, mut cpu, image, _) = vector_machine(model, &records, Some(0xf392));
        let before = cpu.cpu.clone();
        let r = run(&vm, &mut cpu, 64);
        assert_eq!(
            unsupported(r.stop()),
            Some(UnsupportedSystemOperation::NecEmulationSingleStep),
            "{:?}",
            r.stop()
        );
        assert_eq!(r.steps(), 0);
        assert_eq!(cpu.cpu.gpr, before.gpr);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.ss, cpu.cpu.ds, cpu.cpu.es),
            (before.cs, before.rip, before.ss, before.ds, before.es)
        );
        assert_eq!(psw(&cpu), 0xf392);
        assert_eq!(cpu.cpu.legacy.reserved, 0);
        let mut actual = vec![0u8; 1 << 20];
        vm.read_bytes(0, &mut actual).unwrap();
        assert!(actual == image, "memory changed");
    }
}
