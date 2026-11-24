//! NEC V20/V30 native-mode extension witnesses (decision-23 C4b, spec.md
//! §17.10). Every page reference is NEC uPD70108/uPD70116 (V20/V30) User's
//! Manual, Stock No. 500350, October 1986 ("UM"), §12; the pages cited were
//! checked as images. Flags the manual marks undefined (U) are kept by the
//! engine, as C2 and C4a keep them: an engine choice, not a hardware claim.
use x86jit_core::{
    CpuModel, Exit, FpuModel, InterpreterBackend, Prot, RegionKind, StepOptions, StepRun, StepStop,
    UnsupportedSystemOperation, Vcpu, Vm, VmConfig,
};

const MODELS: [CpuModel; 2] = [CpuModel::NecV20, CpuModel::NecV30];
const IP: u64 = 0x1200;
// FLAGS image bits: CF, PF, AF, ZF, SF, OF.
const STATUS: u16 = 0x08d5;
const CF: u16 = 0x0001;
const PF: u16 = 0x0004;
const AF: u16 = 0x0010;
const ZF: u16 = 0x0040;
const SF: u16 = 0x0080;
const OF: u16 = 0x0800;

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
// RAM below 0x70000 and from 0x80000; a trapped (device) region between.
fn setup_with_device(model: CpuModel, code: &[u8]) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
    vm.set_cpu_model(model);
    vm.map(0, 0x70000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x70000, 0x10000, Prot::RW, RegionKind::Trap)
        .unwrap();
    vm.map(0x80000, 0xa0000, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.write_bytes(IP, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = IP;
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}
fn run(vm: &Vm, cpu: &mut Vcpu, steps: u64) -> StepRun {
    cpu.run_steps(vm, steps, StepOptions::default()).unwrap()
}
// Run exactly `count` steps (instructions, or iterations of a repeat).
fn step(vm: &Vm, cpu: &mut Vcpu, count: u64) {
    let r = run(vm, cpu, count);
    assert_eq!(r.steps(), count, "{:?}", r.stop());
    assert!(
        matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{:?}",
        r.stop()
    );
}
fn byte(vm: &Vm, addr: u64) -> u64 {
    vm.mem.read(addr, 1).unwrap()
}
fn word(vm: &Vm, addr: u64) -> u64 {
    vm.mem.read(addr, 2).unwrap()
}
fn flags(cpu: &Vcpu) -> u16 {
    cpu.cpu.flags.to_flags16() & STATUS
}
fn set_flags(cpu: &mut Vcpu, image: u16) {
    let rest = cpu.cpu.flags.to_flags16() & !STATUS;
    cpu.cpu.flags.set_flags16(rest | image);
}
fn vector(vm: &Vm, n: u8, ip: u16, cs: u16) {
    vm.mem
        .write(u64::from(n) * 4, u64::from(ip) | (u64::from(cs) << 16), 4)
        .unwrap();
}
fn refused(r: &StepRun) -> bool {
    matches!(r.stop(), StepStop::Exit(Exit::UnknownInstruction { .. }))
}
fn unsupported(r: &StepRun) -> Option<UnsupportedSystemOperation> {
    match r.stop() {
        StepStop::Exit(Exit::UnsupportedSystemContext { operation, .. }) => Some(*operation),
        _ => None,
    }
}
// A typed refusal at zero steps, with IP on the instruction.
fn assert_refused(model: CpuModel, code: &[u8]) {
    let (vm, mut cpu) = setup(model, code);
    let r = run(&vm, &mut cpu, 1);
    assert!(refused(&r), "{model:?} {code:x?}: {:?}", r.stop());
    assert_eq!((r.steps(), cpu.cpu.rip), (0, IP), "{model:?} {code:x?}");
}

#[test]
fn catalogue_identities_include_native_prefix_policy() {
    // Prefix decoder options join the existing NEC extension revision and
    // bus-width fields in the identity (spec.md §17.10).
    for (model, id) in [
        (CpuModel::I8088, 0x0b71_3477_b270_a8b9),
        (CpuModel::I8086, 0xe283_0b47_9d3b_74e5),
        (CpuModel::I80286, 0xa812_1b75_802e_461b),
        (CpuModel::I80386, 0x11f31a4fa911fbc2),
        (CpuModel::NecV20, 0xd70e_6f86_6f32_e09c),
        (CpuModel::NecV30, 0x9868_73c6_78e7_9ab2),
    ] {
        assert_eq!(model.profile().semantics_id, id, "{model:?}");
    }
    for (model, c4a) in [
        (CpuModel::NecV20, 0xcf97_0e55_8978_5659u64),
        (CpuModel::NecV30, 0xcf97_0e55_8978_5661),
    ] {
        assert_ne!(model.profile().semantics_id, c4a);
    }
}

#[test]
fn bit_tests_set_z_and_clear_v_and_cy() {
    // pp. 12-81–85 TEST1: Z <- 1 when the bit is 0; V 0, CY 0; S, AC, P U.
    // Only bits 2:0 (byte) or 3:0 (word) of CL or the immediate are used.
    // The corpus metadata disagrees (oracles/ssv20/v1_native/metadata.json:
    // 0F10/0F11 "flags": "...szap.", "flags-mask": 65387, Z undefined; 0F18/
    // 0F19 no flag undefined). It holds masks, not captured bytes; the manual
    // is followed (spec.md §17.10).
    for model in MODELS {
        // (code, AX, BX, CL, memory at DS:BX+5 word, expected Z)
        for (code, ax, bx, cl, mem, z) in [
            // TEST1 AL,CL (0F 10 /0, reg)
            (&b"\x0f\x10\xc0"[..], 0x0008, 0, 0x03, 0, false),
            (b"\x0f\x10\xc0", 0x0008, 0, 0xfb, 0, false),
            (b"\x0f\x10\xc0", 0x0008, 0, 0x02, 0, true),
            // TEST1 BH,CL (high-byte register)
            (b"\x0f\x10\xc7", 0, 0x8000, 0x07, 0, false),
            // TEST1 BYTE [BX+5],CL (0F 10 /0, mem8)
            (b"\x0f\x10\x47\x05", 0, 0x100, 0x0e, 0x40, false),
            (b"\x0f\x10\x47\x05", 0, 0x100, 0x05, 0x40, true),
            // TEST1 AW,CL (0F 11, reg16): bit 15 from CL = 1Fh
            (b"\x0f\x11\xc0", 0x8000, 0, 0x1f, 0, false),
            (b"\x0f\x11\xc0", 0x7fff, 0, 0x0f, 0, true),
            // TEST1 WORD [BX+5],CL (0F 11, mem16)
            (b"\x0f\x11\x47\x05", 0, 0x100, 0x09, 0x0200, false),
            // TEST1 BH,1 (0F 18, reg8, imm3); imm 0F9h uses bit 1
            (b"\x0f\x18\xc7\xf9", 0, 0x0200, 0, 0, false),
            (b"\x0f\x18\xc7\x01", 0, 0x0100, 0, 0, true),
            // TEST1 BYTE [BX+5],3 (0F 18, mem8, disp8, imm3)
            (b"\x0f\x18\x47\x05\x0b", 0, 0x100, 0, 0x08, false),
            // TEST1 AW,15 (0F 19, reg16, imm4); imm 2Fh uses bit 15
            (b"\x0f\x19\xc0\x2f", 0x8000, 0, 0, 0, false),
            // TEST1 WORD [BX+5],8 (0F 19, mem16, disp8, imm4)
            (b"\x0f\x19\x47\x05\x08", 0, 0x100, 0, 0x0100, false),
            (b"\x0f\x19\x47\x05\x08", 0, 0x100, 0, 0xfeff, true),
        ] {
            for start in [STATUS, 0] {
                let (vm, mut cpu) = setup(model, code);
                cpu.cpu.gpr[0] = ax;
                cpu.cpu.gpr[3] = bx;
                cpu.cpu.gpr[1] = cl;
                vm.mem.write(0x105, mem, 2).unwrap();
                set_flags(&mut cpu, start);
                step(&vm, &mut cpu, 1);
                assert_eq!(cpu.cpu.rip, IP + code.len() as u64, "{code:x?}");
                let expected = (start & (SF | AF | PF)) | if z { ZF } else { 0 };
                assert_eq!(flags(&cpu), expected, "{model:?} {code:x?} {start:#x}");
                assert_eq!(cpu.cpu.gpr[0], ax);
                assert_eq!(cpu.cpu.gpr[3], bx);
                assert_eq!(word(&vm, 0x105), mem);
            }
        }
    }
}

#[test]
fn bit_clear_set_and_invert_change_one_bit_and_no_flags() {
    // pp. 12-85–98 NOT1, CLR1, SET1: flag operation "None".
    for model in MODELS {
        // (code, register index, before, after) for register forms.
        for (code, reg, before, after) in [
            // SET1 BL,CL (0F 14 /0): CL = 0Bh uses bit 3
            (&b"\x0f\x14\xc3"[..], 3, 0x1234, 0x123c),
            // CLR1 AW,CL (0F 13): CL = 0Bh, bit 11
            (b"\x0f\x13\xc0", 0, 0xffff, 0xf7ff),
            // NOT1 BH,CL (0F 16): bit 3 of BH
            (b"\x0f\x16\xc7", 3, 0x1234, 0x1a34),
            // NOT1 AH,3 (0F 1E, imm3)
            (b"\x0f\x1e\xc4\x03", 0, 0x0000, 0x0800),
            // SET1 CW,0 (0F 1D, imm4)
            (b"\x0f\x1d\xc1\x00", 1, 0x000b, 0x000b),
            // CLR1 DL,7 (0F 1A, imm3), imm 0FFh uses bit 7
            (b"\x0f\x1a\xc2\xff", 2, 0x00ff, 0x007f),
            // NOT1 BW,15 (0F 1F, imm4)
            (b"\x0f\x1f\xc3\x0f", 3, 0x0001, 0x8001),
            // SET1 SP,CL (0F 15): bit 11
            (b"\x0f\x15\xc4", 4, 0x0000, 0x0800),
            // CLR1 AL,CL (0F 12): bit 3
            (b"\x0f\x12\xc0", 0, 0xffff, 0xfff7),
            // CLR1 BW,14 (0F 1B, imm4)
            (b"\x0f\x1b\xc3\x0e", 3, 0xffff, 0xbfff),
            // SET1 AL,4 (0F 1C, imm3)
            (b"\x0f\x1c\xc0\x04", 0, 0x1200, 0x1210),
            // NOT1 DW,CL (0F 17): bit 11
            (b"\x0f\x17\xc2", 2, 0x0800, 0x0000),
        ] {
            let (vm, mut cpu) = setup(model, code);
            cpu.cpu.gpr[1] = 0x0b;
            cpu.cpu.gpr[reg] = before;
            set_flags(&mut cpu, STATUS & !SF);
            step(&vm, &mut cpu, 1);
            assert_eq!(cpu.cpu.gpr[reg], after, "{model:?} {code:x?}");
            assert_eq!(flags(&cpu), STATUS & !SF);
        }
        // Memory forms: (code, word before at 0x2105, after); BX = 100h, BP = 104h, SS = DS = 200h.
        for (code, before, after) in [
            // SET1 BYTE [BX+5],CL: bit 3
            (&b"\x0f\x14\x47\x05"[..], 0x0000, 0x0008),
            // CLR1 WORD [BX+5],CL: bit 11
            (b"\x0f\x13\x47\x05", 0xffff, 0xf7ff),
            // NOT1 BYTE [BX+5],CL
            (b"\x0f\x16\x47\x05", 0x00ff, 0x00f7),
            // NOT1 WORD [BX+5],CL
            (b"\x0f\x17\x47\x05", 0x0000, 0x0800),
            // SET1 BYTE [BX+5],5 (imm3)
            (b"\x0f\x1c\x47\x05\x05", 0x0000, 0x0020),
            // CLR1 WORD [BP+1],0 (imm4, SS default for BP)
            (b"\x0f\x1b\x46\x01\x00", 0xffff, 0xfffe),
            // NOT1 BYTE [BX+5],4 (imm3)
            (b"\x0f\x1e\x47\x05\x04", 0x0010, 0x0000),
            // SET1 WORD [BX+5],15 (imm4)
            (b"\x0f\x1d\x47\x05\x0f", 0x0000, 0x8000),
            // CLR1 BYTE [BX+5],CL with a disp16
            (b"\x0f\x12\x87\x05\x00", 0x00ff, 0x00f7),
            // SET1 WORD ES:[BX+5],CL: the override selects ES (= DS here)
            (b"\x26\x0f\x15\x47\x05", 0x0000, 0x0800),
            // CLR1 BYTE [BX+5],2 (imm3)
            (b"\x0f\x1a\x47\x05\x02", 0x00ff, 0x00fb),
            // NOT1 WORD [BX+5],9 (imm4)
            (b"\x0f\x1f\x47\x05\x09", 0x0200, 0x0000),
        ] {
            let (vm, mut cpu) = setup(model, code);
            cpu.cpu.ds = 0x200;
            cpu.cpu.es = 0x200;
            cpu.cpu.ss = 0x200;
            cpu.cpu.gpr[3] = 0x100;
            cpu.cpu.gpr[5] = 0x104;
            cpu.cpu.gpr[1] = 0x0b;
            vm.mem.write(0x2105, before, 2).unwrap();
            set_flags(&mut cpu, PF | CF);
            step(&vm, &mut cpu, 1);
            assert_eq!(word(&vm, 0x2105), after, "{model:?} {code:x?}");
            assert_eq!(flags(&cpu), PF | CF);
            assert_eq!(cpu.cpu.rip, IP + code.len() as u64);
        }
    }
}

#[test]
fn bit_word_operands_wrap_inside_the_segment() {
    // §10 Fig. 10-1: a word at offset FFFFh takes its high byte from offset
    // 0000h of the same segment.
    for model in MODELS {
        // SET1 WORD [BX],15; then TEST1 WORD [BX],CL with CL = 15.
        let (vm, mut cpu) = setup(model, b"\x0f\x1d\x07\x0f\x0f\x11\x07");
        cpu.cpu.ds = 0x5000;
        cpu.cpu.gpr[3] = 0xffff;
        cpu.cpu.gpr[1] = 15;
        step(&vm, &mut cpu, 1);
        assert_eq!(byte(&vm, 0x50000), 0x80);
        assert_eq!(byte(&vm, 0x5ffff), 0x00);
        assert_eq!(byte(&vm, 0x60000), 0x00);
        set_flags(&mut cpu, ZF);
        step(&vm, &mut cpu, 1);
        assert_eq!(flags(&cpu) & ZF, 0);
    }
}

#[test]
fn nibble_rotates_follow_the_manual_examples() {
    // pp. 12-45–46: ROL4/ROR4 rotate the byte's two digits one digit through
    // AL bits 3:0. AL bits 7:4 are "not assured": the engine keeps them.
    for model in MODELS {
        // ROL4 BL: BL = 95h, AL = 03h -> BL = 53h, AL = X9h.
        let (vm, mut cpu) = setup(model, b"\x0f\x28\xc3");
        cpu.cpu.gpr[3] = 0x95;
        cpu.cpu.gpr[0] = 0xa3;
        set_flags(&mut cpu, STATUS);
        step(&vm, &mut cpu, 1);
        assert_eq!((cpu.cpu.gpr[3], cpu.cpu.gpr[0]), (0x53, 0xa9));
        assert_eq!(flags(&cpu), STATUS);
        // ROR4 CL: CL = 95h, AL = 03h -> CL = 39h, AL = X5h.
        let (vm, mut cpu) = setup(model, b"\x0f\x2a\xc1");
        cpu.cpu.gpr[1] = 0x95;
        cpu.cpu.gpr[0] = 0x03;
        step(&vm, &mut cpu, 1);
        assert_eq!((cpu.cpu.gpr[1], cpu.cpu.gpr[0]), (0x39, 0x05));
        // ROL4 [IX]: 12h, AL = 03h -> 23h, AL = X1h; ROR4 [IX]: -> 31h, X2h.
        for (code, stored, al) in [
            (&b"\x0f\x28\x04"[..], 0x23, 0xf1),
            (b"\x0f\x2a\x04", 0x31, 0xf2),
        ] {
            let (vm, mut cpu) = setup(model, code);
            cpu.cpu.ds = 0x300;
            cpu.cpu.gpr[6] = 0x40;
            cpu.cpu.gpr[0] = 0xf3;
            vm.mem.write(0x3040, 0x12, 1).unwrap();
            set_flags(&mut cpu, 0);
            step(&vm, &mut cpu, 1);
            assert_eq!(byte(&vm, 0x3040), stored);
            assert_eq!(byte(&vm, 0x3041), 0);
            assert_eq!(cpu.cpu.gpr[0], al);
            assert_eq!(flags(&cpu), 0);
        }
        // ROL4 AH: AH is the operand, AL bits 3:0 the digit.
        let (vm, mut cpu) = setup(model, b"\x0f\x28\xc4");
        cpu.cpu.gpr[0] = 0x9503;
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x5309);
    }
}

#[test]
fn bit_fields_follow_the_worked_examples() {
    // pp. 12-21–22 INS: CL = 3, DL = 4, AW = 5555h; INS CL,DL then INS CL,12.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x0f\x31\xd1\x0f\x39\xc1\x0c");
        cpu.cpu.es = 0x2000;
        cpu.cpu.gpr[7] = 0x0100;
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[2] = 4;
        cpu.cpu.gpr[0] = 0x5555;
        set_flags(&mut cpu, STATUS & !ZF);
        step(&vm, &mut cpu, 1);
        // (B): 5 bits at bit 3; CL = 8, IY = base.
        assert_eq!(word(&vm, 0x20100), 0x00a8);
        assert_eq!((cpu.cpu.gpr[1], cpu.cpu.gpr[7]), (8, 0x100));
        step(&vm, &mut cpu, 1);
        // (C): 13 bits at bit 8 into the next word; CL = 5, IY = base + 2.
        assert_eq!(word(&vm, 0x20100), 0x55a8);
        assert_eq!(word(&vm, 0x20102), 0x0015);
        assert_eq!((cpu.cpu.gpr[1], cpu.cpu.gpr[7]), (5, 0x102));
        assert_eq!(cpu.cpu.gpr[0], 0x5555);
        assert_eq!(flags(&cpu), STATUS & !ZF);
        assert_eq!(cpu.cpu.rip, IP + 7);

        // pp. 12-23–24 EXT: [IX] = 5555h, [IX+2] = 3333h; EXT CL,DL, EXT CL,12.
        let (vm, mut cpu) = setup(model, b"\x0f\x33\xd1\x0f\x3b\xc1\x0c");
        cpu.cpu.ds = 0x3000;
        cpu.cpu.gpr[6] = 0x0200;
        vm.mem.write(0x30200, 0x3333_5555, 4).unwrap();
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[2] = 4;
        cpu.cpu.gpr[0] = 0xffff;
        set_flags(&mut cpu, 0);
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x000a);
        assert_eq!((cpu.cpu.gpr[1], cpu.cpu.gpr[6]), (8, 0x200));
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x1355);
        assert_eq!((cpu.cpu.gpr[1], cpu.cpu.gpr[6]), (5, 0x202));
        assert_eq!(flags(&cpu), 0);

        // A 16-bit field (length 15) at offset 0 ends exactly at the word end:
        // the pointer advances and the offset returns to 0.
        let (vm, mut cpu) = setup(model, b"\x0f\x3b\xc2\x0f");
        cpu.cpu.ds = 0x3000;
        cpu.cpu.gpr[6] = 0x0200;
        vm.mem.write(0x30200, 0xbeef, 2).unwrap();
        cpu.cpu.gpr[2] = 0;
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[0], 0xbeef);
        assert_eq!((cpu.cpu.gpr[2], cpu.cpu.gpr[6]), (0, 0x202));

        // INS AL,CL: AW is read before AL takes the next offset.
        let (vm, mut cpu) = setup(model, b"\x0f\x31\xc8");
        cpu.cpu.gpr[0] = 0x0002;
        cpu.cpu.gpr[1] = 0x01;
        cpu.cpu.gpr[7] = 0x0300;
        step(&vm, &mut cpu, 1);
        // Two bits of AW (10b) at bit 2; AL <- 4.
        assert_eq!(word(&vm, 0x300), 0x0008);
        assert_eq!(cpu.cpu.gpr[0], 0x0004);
    }
}

#[test]
fn bit_fields_touch_the_second_word_only_when_reached() {
    // "Transfers: 2 or 4" (INS) and "1 or 2" (EXT): a field inside the first
    // word does not touch the next one. Here the next word is a device.
    for model in MODELS {
        for (code, offset, reaches) in [
            // INS CL,7 (8 bits) at offset 8: ends at bit 16.
            (&b"\x0f\x39\xc1\x07"[..], 8, false),
            // EXT CL,7 at offset 8.
            (b"\x0f\x3b\xc1\x07", 8, false),
            // At offset 9 both reach the device word.
            (b"\x0f\x39\xc1\x07", 9, true),
            (b"\x0f\x3b\xc1\x07", 9, true),
        ] {
            let (vm, mut cpu) = setup_with_device(model, code);
            cpu.cpu.ds = 0x6fff;
            cpu.cpu.es = 0x6fff;
            cpu.cpu.gpr[6] = 0x000e;
            cpu.cpu.gpr[7] = 0x000e;
            cpu.cpu.gpr[1] = offset;
            let r = run(&vm, &mut cpu, 1);
            if reaches {
                let StepStop::Exit(Exit::MmioRead { addr, .. }) = r.stop() else {
                    panic!("{code:x?}: {:?}", r.stop())
                };
                assert_eq!(*addr, 0x70000);
                assert_eq!(r.steps(), 0);
            } else {
                assert_eq!(r.steps(), 1, "{code:x?}: {:?}", r.stop());
                assert_eq!(cpu.cpu.gpr[1], 0);
            }
        }
    }
}

#[test]
fn bit_field_words_wrap_inside_the_segment() {
    // §10 Fig. 10-1: offsets are 16 bits. IY = FFFEh: the second word is at
    // DS1:0000, and IY wraps to 0000.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x0f\x39\xc1\x0f");
        cpu.cpu.es = 0x4000;
        cpu.cpu.gpr[7] = 0xfffe;
        cpu.cpu.gpr[1] = 8;
        cpu.cpu.gpr[0] = 0xabcd;
        step(&vm, &mut cpu, 1);
        assert_eq!(word(&vm, 0x4fffe), 0xcd00);
        assert_eq!(word(&vm, 0x40000), 0x00ab);
        assert_eq!(byte(&vm, 0x50000), 0);
        assert_eq!((cpu.cpu.gpr[1], cpu.cpu.gpr[7]), (8, 0));
        // The word at IY = FFFFh itself wraps byte by byte.
        let (vm, mut cpu) = setup(model, b"\x0f\x3b\xc1\x0f");
        cpu.cpu.ds = 0x4000;
        cpu.cpu.gpr[6] = 0xffff;
        cpu.cpu.gpr[1] = 0;
        vm.mem.write(0x4ffff, 0x34, 1).unwrap();
        vm.mem.write(0x40000, 0x12, 1).unwrap();
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x1234);
        assert_eq!(cpu.cpu.gpr[6], 0x0001);
    }
}

#[test]
fn bit_field_refusals() {
    for model in MODELS {
        // pp. 12-21–24 notes: upper four bits of the offset and length
        // registers must be 0 "for correct operation" -> refused, unchanged.
        for (code, cl, dl) in [
            (&b"\x0f\x31\xd1"[..], 0x13, 0x04),
            (b"\x0f\x31\xd1", 0x03, 0x14),
            (b"\x0f\x33\xd1", 0x83, 0x04),
            (b"\x0f\x33\xd1", 0x03, 0x40),
            (b"\x0f\x39\xc1\x04", 0x23, 0),
            (b"\x0f\x3b\xc1\x04", 0x10, 0),
        ] {
            let (vm, mut cpu) = setup(model, code);
            cpu.cpu.gpr[1] = cl;
            cpu.cpu.gpr[2] = dl;
            cpu.cpu.gpr[0] = 0xffff;
            let before = cpu.cpu.clone();
            let r = run(&vm, &mut cpu, 1);
            assert_eq!(
                unsupported(&r),
                Some(UnsupportedSystemOperation::NecBitFieldOperand),
                "{code:x?}"
            );
            assert_eq!(r.steps(), 0);
            assert_eq!(cpu.cpu.gpr, before.gpr);
            assert_eq!(cpu.cpu.rip, IP);
            assert_eq!(word(&vm, 0), 0);
        }
        // Encodings the pages do not give: immediate above 15, a memory
        // operand, ModRM bits 5:3 nonzero in the immediate forms, EXT into
        // AL or AH (AW and the offset register would both be written), and
        // any prefix.
        for code in [
            &b"\x0f\x39\xc1\x10"[..],
            b"\x0f\x3b\xc1\xff",
            b"\x0f\x31\x07",
            b"\x0f\x33\x47\x05",
            b"\x0f\x39\xc9\x04",
            b"\x0f\x3b\xd1\x04",
            b"\x0f\x33\xc8",
            b"\x0f\x33\xcc",
            b"\x0f\x3b\xc0\x04",
            b"\x0f\x3b\xc4\x04",
            b"\x26\x0f\x31\xd1",
            b"\x3e\x0f\x3b\xc1\x04",
            b"\xf3\x0f\x31\xd1",
            b"\x65\x0f\x33\xd1",
        ] {
            assert_refused(model, code);
        }
    }
}

// The manual's BCD example (p. 12-44): STR0 and STR1 are 07654321, CL = 8.
fn bcd_setup(model: CpuModel, code: &[u8]) -> (Vm, Vcpu) {
    let (vm, mut cpu) = setup(model, code);
    cpu.cpu.ds = 0x400;
    cpu.cpu.es = 0x400;
    cpu.cpu.gpr[6] = 0x10;
    cpu.cpu.gpr[7] = 0x20;
    cpu.cpu.gpr[1] = 8;
    vm.mem.write(0x4010, 0x0765_4321, 4).unwrap();
    vm.mem.write(0x4020, 0x0765_4321, 4).unwrap();
    (vm, cpu)
}

#[test]
fn bcd_strings_follow_the_manual_example() {
    // p. 12-44: CMP4S; ADD4S; CMP4S; SUB4S; SUB4S; SUB4S, with STR1 and the
    // Z and CY values the example lists at (B)-(G). V, S, AC and P are
    // undefined: the engine keeps them.
    let code = b"\x0f\x26\x0f\x20\x0f\x26\x0f\x22\x0f\x22\x0f\x22";
    for model in MODELS {
        let (vm, mut cpu) = bcd_setup(model, code);
        set_flags(&mut cpu, OF | SF | AF | PF);
        for (str1, z, cy) in [
            (0x0765_4321, true, false),
            (0x1530_8642, false, false),
            (0x1530_8642, false, false),
            (0x0765_4321, false, false),
            (0x0000_0000, true, false),
            (0x9234_5679, false, true),
        ] {
            step(&vm, &mut cpu, 1);
            assert_eq!(vm.mem.read(0x4020, 4).unwrap(), str1);
            assert_eq!(vm.mem.read(0x4010, 4).unwrap(), 0x0765_4321);
            let f = flags(&cpu);
            assert_eq!((f & ZF != 0, f & CF != 0), (z, cy), "{str1:#x}");
            assert_eq!(f & (OF | SF | AF | PF), OF | SF | AF | PF);
            assert_eq!(
                (cpu.cpu.gpr[1], cpu.cpu.gpr[6], cpu.cpu.gpr[7]),
                (8, 0x10, 0x20)
            );
        }
        assert_eq!(cpu.cpu.rip, IP + code.len() as u64);
        assert_eq!(vm.mem.read(0x4024, 1).unwrap(), 0);
    }
}

#[test]
fn bcd_strings_carry_override_and_wrap() {
    for model in MODELS {
        // 99 + 01 = (1)00: Z = 1, CY = 1 (two digits).
        let (vm, mut cpu) = bcd_setup(model, b"\x0f\x20");
        cpu.cpu.gpr[1] = 2;
        vm.mem.write(0x4010, 0x01, 1).unwrap();
        vm.mem.write(0x4020, 0x99, 1).unwrap();
        step(&vm, &mut cpu, 1);
        assert_eq!(byte(&vm, 0x4020), 0x00);
        assert_eq!(byte(&vm, 0x4021), 0x43);
        assert_eq!(flags(&cpu) & (ZF | CF), ZF | CF);
        // p. 12-42: the source segment may be overridden (CS here); the
        // destination stays DS1.
        let (vm, mut cpu) = bcd_setup(model, b"\x2e\x0f\x20");
        cpu.cpu.cs = 0;
        cpu.cpu.gpr[6] = 0x3000;
        cpu.cpu.gpr[1] = 2;
        vm.mem.write(0x3000, 0x11, 1).unwrap();
        step(&vm, &mut cpu, 1);
        assert_eq!(byte(&vm, 0x4020), 0x32);
        // §10 Fig. 10-1: IY = FFFFh, four digits -> DS1:FFFF and DS1:0000.
        let (vm, mut cpu) = bcd_setup(model, b"\x0f\x22");
        cpu.cpu.gpr[7] = 0xffff;
        cpu.cpu.gpr[1] = 4;
        vm.mem.write(0x4010, 0x0001, 2).unwrap();
        vm.mem.write(0x13fff, 0x00, 1).unwrap();
        vm.mem.write(0x4000, 0x10, 1).unwrap();
        step(&vm, &mut cpu, 1);
        // 1000 - 0001 = 0999.
        assert_eq!(byte(&vm, 0x13fff), 0x99);
        assert_eq!(byte(&vm, 0x4000), 0x09);
        assert_eq!(byte(&vm, 0x14000), 0);
        assert_eq!(flags(&cpu) & (ZF | CF), 0);
        // In place (IX = IY): STR1 + STR1.
        let (vm, mut cpu) = bcd_setup(model, b"\x0f\x20");
        cpu.cpu.gpr[6] = 0x20;
        step(&vm, &mut cpu, 1);
        assert_eq!(vm.mem.read(0x4020, 4).unwrap(), 0x1530_8642);
    }
}

#[test]
fn bcd_string_refusals_change_nothing() {
    use UnsupportedSystemOperation as U;
    for model in MODELS {
        // pp. 12-42–44: 1 to 254 digits; an odd count leaves the upper digit
        // and the Z and CY flags undefined.
        for (cl, op) in [
            (0, U::NecBcdStringLength),
            (255, U::NecBcdStringLength),
            (3, U::NecBcdStringLength),
            (7, U::NecBcdStringLength),
        ] {
            for code in [&b"\x0f\x20"[..], b"\x0f\x22", b"\x0f\x26"] {
                let (vm, mut cpu) = bcd_setup(model, code);
                cpu.cpu.gpr[1] = cl;
                set_flags(&mut cpu, ZF);
                let r = run(&vm, &mut cpu, 1);
                assert_eq!(unsupported(&r), Some(op), "{code:x?} {cl}");
                assert_eq!((r.steps(), cpu.cpu.rip), (0, IP));
                assert_eq!(vm.mem.read(0x4020, 4).unwrap(), 0x0765_4321);
                assert_eq!(flags(&cpu), ZF);
            }
        }
        // A digit above 9 in either string, found before any store.
        for (at, value) in [(0x4013, 0x0a), (0x4021, 0xb0)] {
            let (vm, mut cpu) = bcd_setup(model, b"\x0f\x20");
            vm.mem.write(at, value, 1).unwrap();
            let r = run(&vm, &mut cpu, 1);
            assert_eq!(unsupported(&r), Some(U::NecBcdDigit));
            assert_eq!(byte(&vm, 0x4020), 0x21);
        }
        // An unmapped destination byte faults before the first store.
        let mut vm = Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
        vm.set_cpu_model(model);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(IP, b"\x0f\x20").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = IP;
        cpu.cpu.es = 0x0fff;
        cpu.cpu.gpr[7] = 0x000e;
        cpu.cpu.gpr[1] = 8;
        vm.mem.write(0xfffe, 0x0101, 2).unwrap();
        let r = run(&vm, &mut cpu, 1);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::UnmappedMemory { .. })),
            "{:?}",
            r.stop()
        );
        assert_eq!(r.steps(), 0);
        assert_eq!(word(&vm, 0xfffe), 0x0101);
        // A byte in a device region: refused before any transfer.
        let (vm, mut cpu) = setup_with_device(model, b"\x0f\x20");
        cpu.cpu.ds = 0x6fff;
        cpu.cpu.gpr[6] = 0x000e;
        cpu.cpu.gpr[1] = 8;
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(unsupported(&r), Some(U::NecBcdStringDevice));
        // No repeat prefix is described for these instructions.
        for code in [&b"\xf3\x0f\x20"[..], b"\xf2\x0f\x22", b"\x65\x0f\x26"] {
            assert_refused(model, code);
        }
    }
}

#[test]
fn carry_repeats_compare_while_cy_holds() {
    // p. 12-14: REPC (65h) repeats CMPBK/CMPM while CY = 1, REPNC (64h)
    // while CY = 0; CW is tested before each iteration and decremented.
    for model in MODELS {
        // (code, source bytes at DS:10h, destination bytes at ES:20h, AL,
        //  iterations, CY after)
        for (code, src, dst, al, n, cy) in [
            // REPC CMPBKB: 1-2 and 1-2 borrow, 5-2 does not.
            (
                &b"\x65\xa6"[..],
                [1u8, 1, 5, 1],
                [2u8, 2, 2, 2],
                0,
                3,
                false,
            ),
            // REPNC CMPBKB: 5-2, 5-2 no borrow, 1-2 borrows.
            (b"\x64\xa6", [5, 5, 1, 5], [2, 2, 2, 2], 0, 3, true),
            // REPC CMPMB: AL - (IY) borrows while (IY) > AL.
            (b"\x65\xae", [0; 4], [9, 9, 1, 9], 5, 3, false),
            // REPNC CMPMB.
            (b"\x64\xae", [0; 4], [1, 9, 9, 9], 5, 2, true),
            // Ends on CW = 0 while the condition still holds.
            (b"\x65\xa6", [0, 0, 0, 0], [1, 1, 1, 1], 0, 4, true),
        ] {
            for start_cy in [false, true] {
                let (vm, mut cpu) = setup(model, code);
                cpu.cpu.ds = 0x500;
                cpu.cpu.es = 0x600;
                cpu.cpu.gpr[6] = 0x10;
                cpu.cpu.gpr[7] = 0x20;
                cpu.cpu.gpr[1] = 4;
                cpu.cpu.gpr[0] = al;
                vm.write_bytes(0x5010, &src).unwrap();
                vm.write_bytes(0x6020, &dst).unwrap();
                // "The contents of CY immediately before ... are don't care."
                set_flags(&mut cpu, if start_cy { CF } else { 0 });
                step(&vm, &mut cpu, n);
                assert_eq!(cpu.cpu.rip, IP + 2, "{code:x?}");
                assert_eq!(cpu.cpu.gpr[1], 4 - n);
                assert_eq!(cpu.cpu.gpr[7], 0x20 + n);
                assert_eq!(cpu.cpu.flags.cf, cy);
            }
        }
        // CW = 0: nothing is compared and no flag changes.
        let (vm, mut cpu) = setup(model, b"\x65\xa7");
        set_flags(&mut cpu, ZF | PF);
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.rip, IP + 2);
        assert_eq!(flags(&cpu), ZF | PF);
        // Word forms: REPNC CMPBKW and REPC CMPMW with DIR = 1.
        let (vm, mut cpu) = setup(model, b"\x64\xa7");
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[6] = 0x104;
        cpu.cpu.gpr[7] = 0x204;
        cpu.cpu.flags.df = true;
        vm.mem.write(0x100, 0x0005_0005_0001, 6).unwrap();
        vm.mem.write(0x200, 0x0002_0002_0002, 6).unwrap();
        step(&vm, &mut cpu, 3);
        assert_eq!((cpu.cpu.gpr[1], cpu.cpu.gpr[6]), (0, 0xfe));
        assert!(cpu.cpu.flags.cf);
        let (vm, mut cpu) = setup(model, b"\x65\xaf");
        cpu.cpu.gpr[0] = 0x1000;
        cpu.cpu.gpr[1] = 5;
        cpu.cpu.gpr[7] = 0x200;
        vm.mem.write(0x200, 0x0fff_2000, 4).unwrap();
        step(&vm, &mut cpu, 2);
        assert_eq!((cpu.cpu.gpr[1], cpu.cpu.gpr[7]), (3, 0x204));
        assert!(!cpu.cpu.flags.cf);
    }
}

#[test]
fn carry_repeats_budget_and_interrupt_restart() {
    for model in MODELS {
        // Budgets 1, 2 and N: one step per iteration, IP stays on the
        // instruction until it completes.
        for budget in [1, 2, 4] {
            let (vm, mut cpu) = setup(model, b"\x65\xa6\xf4");
            cpu.cpu.gpr[1] = 4;
            cpu.cpu.gpr[6] = 0x10;
            cpu.cpu.gpr[7] = 0x20;
            vm.write_bytes(0x10, &[0, 0, 0, 0]).unwrap();
            vm.write_bytes(0x20, &[1, 1, 1, 1]).unwrap();
            step(&vm, &mut cpu, budget);
            assert_eq!(cpu.cpu.gpr[1], 4 - budget);
            let expected = if budget == 4 { IP + 2 } else { IP };
            assert_eq!(cpu.cpu.rip, expected);
            let r = run(&vm, &mut cpu, 10);
            assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
            assert_eq!(cpu.cpu.gpr[1], 0);
        }
        // §6 p. 6-4: an interrupted block comparison keeps up to three
        // prefixes, REPC and REPNC among them.
        for (code, restart) in [
            (&b"\x26\x2e\x65\xa6"[..], IP),
            (&b"\x26\x2e\x3e\x64\xa6"[..], IP + 1),
        ] {
            let (vm, mut cpu) = setup(model, code);
            vector(&vm, 0x20, 0x400, 0);
            vm.write_bytes(0x400, b"\xf4").unwrap();
            cpu.cpu.flags.if_ = true;
            cpu.cpu.gpr[1] = 3;
            cpu.cpu.gpr[6] = 0x10;
            cpu.cpu.gpr[7] = 0x20;
            let (src, dst) = if code.contains(&0x65) {
                ([0u8, 0, 0], [1u8, 1, 1])
            } else {
                ([1, 1, 1], [0, 0, 0])
            };
            vm.write_bytes(0x10, &src).unwrap();
            vm.write_bytes(0x20, &dst).unwrap();
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
fn carry_repeat_refusals() {
    for model in MODELS {
        // p. 12-14 names the block comparisons only; two different repeat
        // prefixes are not described. The corpus README says "REPC/REPNC
        // prefixes on INS and OUTS act as regular REP prefixes" (6C/6D); the
        // page does not describe that, so 65 6C and 64 6F stay refused.
        for code in [
            &b"\x65\xa4"[..],
            b"\x64\xa5",
            b"\x65\xaa",
            b"\x64\xac",
            b"\x65\x6c",
            b"\x64\x6f",
            b"\x65\x90",
            b"\xf3\x65\xa6",
            b"\x65\xf3\xa6",
            b"\x64\x65\xa6",
            b"\xf2\x64\xae",
            // REPC before ADD (the byte after the prefix is 00h).
            b"\x65",
        ] {
            let (vm, mut cpu) = setup(model, code);
            cpu.cpu.gpr[1] = 2;
            let r = run(&vm, &mut cpu, 1);
            assert!(refused(&r), "{code:x?}: {:?}", r.stop());
            assert_eq!((r.steps(), cpu.cpu.rip, cpu.cpu.gpr[1]), (0, IP, 2));
        }
        // A repeated REPC folds, as other repeated prefixes do.
        let (vm, mut cpu) = setup(model, b"\x65\x65\xa6");
        cpu.cpu.gpr[1] = 1;
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.rip, IP + 3);
    }
}

#[test]
fn fpo2_does_the_cpu_part_with_an_empty_socket() {
    // pp. 12-174–175 FPO2 (66h/67h): the CPU forms the address and starts one
    // memory read; flags "None". The device region shows the read.
    for model in MODELS {
        for (code, len, addr) in [
            // FPO2 fp-op,[BX] (66h, mod 00)
            (&b"\x66\x07"[..], 2, 0x70010),
            // FPO2 fp-op,[BX+disp16] (67h, mod 10)
            (b"\x67\x87\x00\x01", 4, 0x70110),
            // ES override: ES:[BX]
            (b"\x26\x66\x07", 3, 0x71010),
        ] {
            let (vm, mut cpu) = setup_with_device(model, code);
            cpu.set_fpu_model(FpuModel::None);
            cpu.cpu.ds = 0x7000;
            cpu.cpu.es = 0x7100;
            cpu.cpu.gpr[3] = 0x10;
            // Exactly one word, then the step completes (p. 12-175 clocks
            // 15/15/11, the odd/even split of a word): two device bytes on
            // the V20's 8-bit bus, one word transfer at these even addresses
            // on the V30's 16-bit bus (UM §4 pp. 4-2–4-3).
            let transfers: &[(u64, u8)] = if model == CpuModel::NecV30 {
                &[(0, 2)]
            } else {
                &[(0, 1), (1, 1)]
            };
            for &(byte, width) in transfers {
                let r = run(&vm, &mut cpu, 1);
                let StepStop::Exit(Exit::MmioRead { addr: at, size, .. }) = r.stop() else {
                    panic!("{code:x?}: {:?}", r.stop())
                };
                assert_eq!(
                    (*at, *size, r.steps()),
                    (addr + byte, width, 0),
                    "{code:x?}"
                );
                cpu.complete_mmio_read(0x5a);
            }
            let r = run(&vm, &mut cpu, 1);
            assert_eq!(r.steps(), 1, "{:?}", r.stop());
            assert_eq!(cpu.cpu.rip, IP + len);
        }
        // Register forms do nothing in the CPU.
        for code in [&b"\x66\xc0"[..], b"\x67\xff", b"\x2e\x66\xd9"] {
            let (vm, mut cpu) = setup(model, code);
            cpu.set_fpu_model(FpuModel::None);
            set_flags(&mut cpu, STATUS);
            let before = cpu.cpu.gpr;
            step(&vm, &mut cpu, 1);
            assert_eq!(cpu.cpu.rip, IP + code.len() as u64);
            assert_eq!(cpu.cpu.gpr, before);
            assert_eq!(flags(&cpu), STATUS);
        }
    }
}

#[test]
fn fpo2_refusals() {
    for model in MODELS {
        // An attached generation-neutral coprocessor defines no 66h/67h
        // instruction (the result is not stated), and the EM input asks for
        // a handoff: both refuse before the read.
        for (fpu, em) in [
            (FpuModel::Present, false),
            (FpuModel::None, true),
            (FpuModel::Present, true),
        ] {
            let (vm, mut cpu) = setup_with_device(model, b"\x66\x07");
            cpu.set_fpu_model(fpu);
            cpu.set_fpu_emulation(em);
            cpu.cpu.ds = 0x7000;
            let r = run(&vm, &mut cpu, 1);
            assert!(refused(&r), "{fpu:?} {em}: {:?}", r.stop());
            assert_eq!((r.steps(), cpu.cpu.rip), (0, IP));
        }
        // No repeat prefix is described (as for ESC).
        for code in [&b"\xf3\x66\x07"[..], b"\xf2\x67\xc0", b"\x65\x66\x07"] {
            let (vm, mut cpu) = setup(model, code);
            cpu.set_fpu_model(FpuModel::None);
            let r = run(&vm, &mut cpu, 1);
            assert!(refused(&r), "{code:x?}: {:?}", r.stop());
        }
    }
}

#[test]
fn extension_encodings_the_pages_do_not_give_are_refused() {
    for model in MODELS {
        for code in [
            // 0Fh second bytes outside the listed set.
            &b"\x0f\x00\xc0"[..],
            b"\x0f\x21",
            b"\x0f\x24",
            b"\x0f\x29\xc0",
            b"\x0f\x30\xc0",
            b"\x0f\x3a\xc0\x01",
            b"\x0f\xa2",
            b"\x0f\xfe",
            // ModRM bits 5:3 must be 000.
            b"\x0f\x10\xc8",
            b"\x0f\x1f\x7f\x00\x01",
            b"\x0f\x28\xd3",
            // ROL4/ROR4 AL: AL is the operand and the digit source.
            b"\x0f\x28\xc0",
            b"\x0f\x2a\xc0",
            // Repeat prefixes on the bit and nibble forms.
            b"\xf3\x0f\x10\xc0",
            b"\x64\x0f\x14\xc3",
            b"\xf2\x0f\x28\xc3",
            // BRKEM keeps its C4a refusal of prefixes.
            b"\x26\x0f\xff\x80",
        ] {
            assert_refused(model, code);
        }
    }
}

#[test]
fn other_models_keep_refusing_the_extension_bytes() {
    // The 8086 has no 0Fh map and no 64h-67h (8086 Family UM Table 4-13);
    // C2 refuses them, and C4b leaves that decode unchanged.
    for model in [CpuModel::I8088, CpuModel::I8086] {
        for code in [
            &b"\x0f\x10\xc0"[..],
            b"\x0f\x20",
            b"\x0f\x28\xc3",
            b"\x0f\x31\xd1",
            b"\x0f\x39\xc1\x04",
            b"\x0f\xff\x80",
            b"\x64\xa6",
            b"\x65\xa6",
            b"\x66\x07",
            b"\x67\xc0",
        ] {
            let (vm, mut cpu) = setup(model, code);
            cpu.cpu.gpr[1] = 2;
            let r = run(&vm, &mut cpu, 1);
            assert!(refused(&r), "{model:?} {code:x?}: {:?}", r.stop());
            assert_eq!((r.steps(), cpu.cpu.rip), (0, IP));
        }
    }
}

#[test]
fn one_step_and_one_retirement_per_instruction() {
    // Decision-16 §2 as for C4a: each extension instruction is one step and
    // one retired instruction; a repeat is one step per iteration and one
    // retired instruction.
    for model in MODELS {
        // TEST1 AL,CL; SET1 BL,5; ROL4 BL; CMP4S; ADD4S; EXT CL,DL;
        // INS CL,DL; FPO2 (register); REPC CMPBKB; HALT.
        let code = b"\x0f\x10\xc0\x0f\x1c\xc3\x05\x0f\x28\xc3\x0f\x26\x0f\x20\x0f\x33\xd1\x0f\x31\xd1\x66\xc0\x65\xa6\xf4";
        let (vm, mut cpu) = setup(model, code);
        cpu.set_fpu_model(FpuModel::None);
        cpu.cpu.ds = 0x900;
        cpu.cpu.es = 0x900;
        cpu.cpu.gpr[1] = 2;
        cpu.cpu.gpr[2] = 1;
        cpu.cpu.gpr[6] = 0x10;
        cpu.cpu.gpr[7] = 0x20;
        for count in [1u64, 1, 1, 1, 1, 1, 1, 1] {
            let before = cpu.retired_instructions();
            step(&vm, &mut cpu, count);
            assert_eq!(cpu.retired_instructions() - before, count);
        }
        // REPC CMPBKB with CW = 6: (9010h) - (9020h) = 0 clears CY, so one
        // iteration ends it; then HALT.
        let before = cpu.retired_instructions();
        let r = run(&vm, &mut cpu, 10);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        assert_eq!(cpu.retired_instructions() - before, 2);
        assert_eq!(r.steps(), 2);
    }
}

// Review witnesses for C4b (xj-50w.14): blocks, TF, SMC, mid-block refusals,
// INS ending at bit 16, the FPO2 word, BCD Z over every byte and overlap.
// FPO2 refused at run time (Present) after an earlier instruction of the
// same block: the earlier instruction stays retired, IP names the FPO2.
#[test]
fn fpo2_refusal_after_prior_instruction() {
    for model in MODELS {
        for (fpu, em) in [(FpuModel::Present, false), (FpuModel::None, true)] {
            let (vm, mut cpu) = setup(model, b"\x40\x66\x07\xf4");
            cpu.set_fpu_model(fpu);
            cpu.set_fpu_emulation(em);
            let r = run(&vm, &mut cpu, 5);
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::UnknownInstruction { .. })),
                "{:?}",
                r.stop()
            );
            assert_eq!(
                (r.steps(), cpu.cpu.rip, cpu.cpu.gpr[0]),
                (1, IP + 1, 1),
                "{model:?} {fpu:?} {em}"
            );
        }
    }
}

// A whole block of extension forms run in one call equals stepping.
#[test]
fn multi_instruction_block_equals_stepping() {
    // TEST1 AL,CL; CMP4S; EXT CL,DL; FPO2 reg; TEST1 WORD [BX],3; ROL4 BL;
    // SET1 BYTE [BX],1; INS CL,DL; HLT
    let code = b"\x0f\x10\xc0\x0f\x26\x0f\x33\xd1\x66\xc0\x0f\x19\x07\x03\x0f\x28\xc3\x0f\x1c\x07\x01\x0f\x31\xd1\xf4";
    for model in MODELS {
        let init = |cpu: &mut Vcpu, vm: &Vm| {
            cpu.set_fpu_model(FpuModel::None);
            cpu.cpu.ds = 0x900;
            cpu.cpu.es = 0x900;
            cpu.cpu.gpr[0] = 0x1234;
            cpu.cpu.gpr[1] = 4;
            cpu.cpu.gpr[2] = 6;
            cpu.cpu.gpr[3] = 0x40;
            cpu.cpu.gpr[6] = 0x10;
            cpu.cpu.gpr[7] = 0x20;
            vm.mem.write(0x9010, 0x1234, 2).unwrap();
            vm.mem.write(0x9020, 0x5678, 2).unwrap();
            vm.mem.write(0x9040, 0x0008, 2).unwrap();
        };
        let (vm_a, mut a) = setup(model, code);
        init(&mut a, &vm_a);
        let r = run(&vm_a, &mut a, 100);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        let (vm_b, mut b) = setup(model, code);
        init(&mut b, &vm_b);
        for _ in 0..20 {
            let r = run(&vm_b, &mut b, 1);
            if matches!(r.stop(), StepStop::Exit(Exit::Hlt)) {
                break;
            }
        }
        assert_eq!(a.cpu.gpr, b.cpu.gpr, "{model:?}");
        assert_eq!(a.cpu.rip, b.cpu.rip);
        assert_eq!(
            a.cpu.flags.to_flags16(),
            b.cpu.flags.to_flags16(),
            "{model:?}"
        );
        for at in (0x9000u64..0x9100).step_by(2) {
            assert_eq!(word(&vm_a, at), word(&vm_b, at), "{at:#x}");
        }
        assert_eq!(a.retired_instructions(), b.retired_instructions());
    }
}

// TF set: the single-step trap follows each extension instruction, with the
// next instruction's IP saved.
#[test]
fn single_step_after_extension_instructions() {
    for model in MODELS {
        for code in [
            &b"\x0f\x10\xc0"[..],
            b"\x0f\x31\xd1",
            b"\x0f\x33\xd1",
            b"\x0f\x20",
            b"\x0f\x26",
            b"\x0f\x28\xc3",
            b"\x0f\x1c\x07\x01",
            b"\x66\xc0",
            b"\x66\x07",
        ] {
            let mut full = code.to_vec();
            full.push(0x90);
            let (vm, mut cpu) = setup(model, &full);
            cpu.set_fpu_model(FpuModel::None);
            vector(&vm, 1, 0x500, 0);
            vm.write_bytes(0x500, b"\xf4").unwrap();
            cpu.cpu.ds = 0x900;
            cpu.cpu.es = 0x900;
            cpu.cpu.gpr[1] = 2;
            cpu.cpu.gpr[2] = 1;
            cpu.cpu.gpr[3] = 0x40;
            cpu.cpu.gpr[6] = 0x10;
            cpu.cpu.gpr[7] = 0x20;
            cpu.cpu.rflags_system = 0x100;
            let r = run(&vm, &mut cpu, 5);
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{code:x?}: {:?}",
                r.stop()
            );
            assert_eq!(cpu.cpu.rip, 0x501, "{code:x?}");
            assert_eq!(word(&vm, 0x7ffa), IP + code.len() as u64, "{code:x?}");
        }
    }
}

// REPC with TF behaves as REPE with TF when every iteration continues.
#[test]
fn single_step_repc_matches_repe() {
    for model in MODELS {
        let mut seen = Vec::new();
        for (code, src, dst) in [
            (&b"\xf3\xa6\x90"[..], [1u8, 1, 1], [1u8, 1, 1]),
            (b"\x65\xa6\x90", [0u8, 0, 0], [1u8, 1, 1]),
        ] {
            let (vm, mut cpu) = setup(model, code);
            vector(&vm, 1, 0x500, 0);
            vm.write_bytes(0x500, b"\xf4").unwrap();
            cpu.cpu.gpr[1] = 3;
            cpu.cpu.gpr[6] = 0x10;
            cpu.cpu.gpr[7] = 0x20;
            vm.write_bytes(0x10, &src).unwrap();
            vm.write_bytes(0x20, &dst).unwrap();
            cpu.cpu.rflags_system = 0x100;
            let r = run(&vm, &mut cpu, 10);
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{:?}",
                r.stop()
            );
            seen.push((word(&vm, 0x7ffa), cpu.cpu.gpr[1], r.steps()));
        }
        assert_eq!(seen[0], seen[1], "{model:?}");
    }
}

// The INS end-of-word rule for INS (only EXT checks the pointer).
#[test]
fn ins_ending_at_bit_16_advances_iy() {
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x0f\x39\xc1\x07");
        cpu.cpu.es = 0x2000;
        cpu.cpu.gpr[7] = 0x100;
        cpu.cpu.gpr[1] = 8;
        cpu.cpu.gpr[0] = 0xab;
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 1);
        assert_eq!(word(&vm, 0x20100), 0xab00);
        assert_eq!((cpu.cpu.gpr[1], cpu.cpu.gpr[7]), (0, 0x102));
    }
}

// SMC: INS, ADD4S (block terminators) and CLR1 (lifted store) rewrite an
// instruction that has already been executed and cached.
#[test]
fn extension_stores_invalidate_cached_code() {
    for model in MODELS {
        // INS CL,7 at ES:DI = the NOP at IP+4.
        let (vm, mut cpu) = setup(model, b"\x0f\x39\xc1\x07\x90\xf4");
        let pass = |cpu: &mut Vcpu, aw: u64| {
            cpu.cpu.rip = IP;
            cpu.cpu.gpr[1] = 0;
            cpu.cpu.gpr[7] = IP + 4;
            cpu.cpu.gpr[0] = aw;
            run(&vm, cpu, 10)
        };
        let r = pass(&mut cpu, 0x90);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(cpu.cpu.rip, IP + 6);
        let r = pass(&mut cpu, 0xf4);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(cpu.cpu.rip, IP + 5, "{model:?} INS");

        // ADD4S, one byte at ES:DI = IP+2 (90h; 90h + 50h = 40h, INC AW).
        let (vm, mut cpu) = setup(model, b"\x0f\x20\x90\xf4");
        vm.write_bytes(0x10, &[0x00]).unwrap();
        let pass = |cpu: &mut Vcpu| {
            cpu.cpu.rip = IP;
            cpu.cpu.gpr[0] = 0;
            cpu.cpu.gpr[1] = 2;
            cpu.cpu.gpr[6] = 0x10;
            cpu.cpu.gpr[7] = IP + 2;
            run(&vm, cpu, 10)
        };
        let r = pass(&mut cpu);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[0]), (IP + 4, 0));
        vm.write_bytes(0x10, &[0x50]).unwrap();
        let r = pass(&mut cpu);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(
            (cpu.cpu.rip, cpu.cpu.gpr[0]),
            (IP + 4, 1),
            "{model:?} ADD4S"
        );

        // CLR1 BYTE [BX],3: DEC AW (48h) at IP+4 becomes INC AW (40h).
        let (vm, mut cpu) = setup(model, b"\x0f\x1a\x07\x03\x48\xf4");
        let pass = |cpu: &mut Vcpu, bx: u64| {
            cpu.cpu.rip = IP;
            cpu.cpu.gpr[0] = 0;
            cpu.cpu.gpr[3] = bx;
            run(&vm, cpu, 10)
        };
        let r = pass(&mut cpu, 0x30);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(cpu.cpu.gpr[0], 0xffff);
        let r = pass(&mut cpu, IP + 4);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(cpu.cpu.gpr[0], 1, "{model:?} CLR1");
    }
}

// Overlapping BCD strings (IY = IX + 1): the result depends on the transfer
// order, which pp. 12-42–43 do not give ("Transfers: 3n"). Byte by byte,
// [12h] = 33h + (new [11h] = 33h) = 66h; with all reads first it would be
// 33h + 22h = 55h. Refused before any transfer (decision-23 §3), as C4a
// refuses PUSH SP. An overlap whose shared bytes are read no later than they
// are stored has one result and runs, and CMP4S stores nothing.
#[test]
fn bcd_overlap_order_is_refused_where_observable() {
    use x86jit_core::UnsupportedSystemOperation as U;
    for model in MODELS {
        for code in [&b"\x0f\x20"[..], b"\x0f\x22"] {
            let (vm, mut cpu) = setup(model, code);
            cpu.cpu.gpr[1] = 4;
            cpu.cpu.gpr[6] = 0x10;
            cpu.cpu.gpr[7] = 0x11;
            vm.write_bytes(0x10, &[0x11, 0x22, 0x33]).unwrap();
            let r = run(&vm, &mut cpu, 1);
            assert!(
                matches!(
                    r.stop(),
                    StepStop::Exit(Exit::UnsupportedSystemContext {
                        operation: U::NecBcdStringOverlap,
                        ..
                    })
                ),
                "{code:x?}: {:?}",
                r.stop()
            );
            assert_eq!((r.steps(), cpu.cpu.rip), (0, IP));
            assert_eq!(vm.mem.read(0x10, 3).unwrap(), 0x33_2211);
        }
        // The same overlap through segments: DS1:0001 is DS0:0011 when
        // DS1 = DS0 + 1 (IY = 1, IX = 10h).
        let (vm, mut cpu) = setup(model, b"\x0f\x20");
        cpu.cpu.es = 1;
        cpu.cpu.gpr[1] = 4;
        cpu.cpu.gpr[6] = 0x10;
        cpu.cpu.gpr[7] = 0x01;
        let r = run(&vm, &mut cpu, 1);
        assert!(
            matches!(
                r.stop(),
                StepStop::Exit(Exit::UnsupportedSystemContext {
                    operation: U::NecBcdStringOverlap,
                    ..
                })
            ),
            "{:?}",
            r.stop()
        );
        // IY = IX - 1: every shared byte is read (step j) before it is
        // stored (step j + 1); either order gives [10h] = 11h + 22h = 33h,
        // [11h] = 22h + 33h = 55h.
        let (vm, mut cpu) = setup(model, b"\x0f\x20");
        cpu.cpu.gpr[1] = 4;
        cpu.cpu.gpr[6] = 0x11;
        cpu.cpu.gpr[7] = 0x10;
        vm.write_bytes(0x10, &[0x11, 0x22, 0x33]).unwrap();
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 1, "{:?}", r.stop());
        assert_eq!(vm.mem.read(0x10, 3).unwrap(), 0x33_5533);
        // CMP4S with IY = IX + 1 stores nothing and runs: 3322h - 2211h.
        let (vm, mut cpu) = setup(model, b"\x0f\x26");
        cpu.cpu.gpr[1] = 4;
        cpu.cpu.gpr[6] = 0x10;
        cpu.cpu.gpr[7] = 0x11;
        vm.write_bytes(0x10, &[0x11, 0x22, 0x33]).unwrap();
        let r = run(&vm, &mut cpu, 1);
        assert_eq!(r.steps(), 1, "{:?}", r.stop());
        assert!(!cpu.cpu.flags.zf && !cpu.cpu.flags.cf);
        assert_eq!(vm.mem.read(0x10, 3).unwrap(), 0x33_2211);
    }
}

// Run-time refusals after an earlier instruction of the same block.
#[test]
fn runtime_refusals_after_prior_instruction() {
    use x86jit_core::UnsupportedSystemOperation as U;
    for model in MODELS {
        for (code, cl, expected) in [
            (&b"\x40\x0f\x31\xd1\xf4"[..], 0x13u64, U::NecBitFieldOperand),
            (b"\x40\x0f\x20\xf4", 3, U::NecBcdStringLength),
        ] {
            let (vm, mut cpu) = setup(model, code);
            cpu.cpu.gpr[1] = cl;
            cpu.cpu.gpr[2] = 1;
            let r = run(&vm, &mut cpu, 5);
            let StepStop::Exit(Exit::UnsupportedSystemContext {
                operation, addr, ..
            }) = r.stop()
            else {
                panic!("{code:x?}: {:?}", r.stop())
            };
            assert_eq!((*operation, *addr), (expected, IP + 1));
            assert_eq!((r.steps(), cpu.cpu.rip, cpu.cpu.gpr[0]), (1, IP + 1, 1));
        }
        // A digit above 9 found after the first instruction.
        let (vm, mut cpu) = setup(model, b"\x40\x0f\x20\xf4");
        cpu.cpu.gpr[1] = 2;
        vm.write_bytes(0, &[0x0a]).unwrap();
        let r = run(&vm, &mut cpu, 5);
        assert!(
            matches!(
                r.stop(),
                StepStop::Exit(Exit::UnsupportedSystemContext {
                    operation: U::NecBcdDigit,
                    ..
                })
            ),
            "{:?}",
            r.stop()
        );
        assert_eq!((r.steps(), cpu.cpu.rip), (1, IP + 1));
    }
}

// FPO2 width: how many device bytes the memory form reads.
#[test]
fn fpo2_reads_a_word() {
    for model in MODELS {
        let mut vm = Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
        vm.set_cpu_model(model);
        vm.map(0, 0x70000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x70000, 0x10000, Prot::RW, RegionKind::Trap)
            .unwrap();
        vm.write_bytes(IP, b"\x66\x07\xf4").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = IP;
        cpu.cpu.gpr[4] = 0x8000;
        cpu.set_fpu_model(FpuModel::None);
        cpu.cpu.ds = 0x7000;
        let (mut reads, mut bytes) = (0, 0);
        loop {
            let r = run(&vm, &mut cpu, 5);
            match r.stop() {
                StepStop::Exit(Exit::MmioRead { size, .. }) => {
                    reads += 1;
                    bytes += *size;
                    cpu.complete_mmio_read(0);
                }
                StepStop::Exit(Exit::Hlt) => break,
                other => panic!("{other:?}"),
            }
        }
        // Two bytes: two transfers on the V20, one even word on the V30.
        let transfers = if model == CpuModel::NecV30 { 1 } else { 2 };
        assert_eq!((reads, bytes), (transfers, 2), "{model:?}");
    }
}

// Z covers every result byte (pp. 12-42–44, "set according to the result").
// The shipped witnesses never have a zero byte next to a nonzero one, so a
// Z taken from one byte only would also pass them.
#[test]
fn bcd_zero_flag_covers_every_byte() {
    for model in MODELS {
        for (dst, src, expected) in [
            (0x0000_0001u64, 0u64, 0x0000_0001u64),
            (0x0100_0000, 0, 0x0100_0000),
            (0x0000_0100, 0, 0x0000_0100),
        ] {
            let (vm, mut cpu) = setup(model, b"\x0f\x22");
            cpu.cpu.gpr[1] = 8;
            cpu.cpu.gpr[6] = 0x10;
            cpu.cpu.gpr[7] = 0x20;
            vm.mem.write(0x10, src, 4).unwrap();
            vm.mem.write(0x20, dst, 4).unwrap();
            let r = run(&vm, &mut cpu, 1);
            assert_eq!(r.steps(), 1);
            assert_eq!(vm.mem.read(0x20, 4).unwrap(), expected);
            assert!(!cpu.cpu.flags.zf, "{model:?} {dst:#x}");
        }
    }
}

// xj-50w.15 on the shared twenty-bit dispatcher. §6 p. 6-4: an interrupted
// block instruction keeps up to three prefixes, so a REPC CMPBK with four
// resumes at its second byte. That restart is keyed on the inputs the
// comparison reads: a host edit of CW, IX, IY, DS0 (the source segment
// here) or DS1 after a yield starts a new operation, and an interrupt taken
// before its first element names the first prefix. An edit of AW, which
// CMPBK does not read, keeps it.
#[test]
fn host_edit_of_a_block_comparison_input_abandons_its_restart() {
    type Edit = fn(&mut x86jit_core::CpuState);
    let cases: [(&str, Edit, u64); 7] = [
        ("none", |_| {}, IP + 1),
        ("cw", |c| c.gpr[1] = 3, IP),
        ("ix", |c| c.gpr[6] = 0x30, IP),
        ("iy", |c| c.gpr[7] = 0x40, IP),
        ("ds0", |c| c.ds = 0x10, IP),
        ("ds1", |c| c.es = 0x10, IP),
        ("aw", |c| c.gpr[0] = 0x55, IP + 1),
    ];
    for model in MODELS {
        for (name, edit, frame) in cases {
            let (vm, mut cpu) = setup(model, b"\x26\x2e\x3e\x65\xa6\xf4");
            vector(&vm, 0x20, 0x400, 0);
            vm.write_bytes(0x400, b"\xf4").unwrap();
            cpu.cpu.flags.if_ = true;
            cpu.cpu.gpr[1] = 3;
            cpu.cpu.gpr[6] = 0x10;
            cpu.cpu.gpr[7] = 0x20;
            vm.write_bytes(0x10, &[0, 0, 0, 0]).unwrap();
            vm.write_bytes(0x20, &[1, 1, 1, 1]).unwrap();
            step(&vm, &mut cpu, 1);
            assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[1]), (IP, 2));
            edit(&mut cpu.cpu);
            cpu.inject_irq(0x20);
            let r = run(&vm, &mut cpu, 1);
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{model:?} {name}: {:?}",
                r.stop()
            );
            assert_eq!(word(&vm, 0x7ffa), frame, "{model:?} {name}");
        }
    }
}

// xj-50w.12: BUSLOCK on the forms only the V20/V30 have. P. 12-173: BUSLOCK
// "Outputs the buslock signal (BUSLOCK) while the instruction immediately
// after the BUSLOCK instruction is being executed", and through the whole
// block operation of a repeated one; "Flag operation: None". The shared rule
// (legacy_i8086.rs, LOCK witnesses for all four twenty-bit models) applies:
// a form that transfers no data runs as the unprefixed one, and every form
// that transfers data without the atomic read-modify-write is a typed
// refusal, as is every repeated block instruction, REPC and REPNC included.
#[test]
fn buslock_on_the_nec_forms() {
    let state = |model: CpuModel, code: &[u8]| {
        let (vm, mut cpu) = setup(model, code);
        // FPO2 has a CPU part only with an empty socket (pp. 12-174–175).
        cpu.set_fpu_model(FpuModel::None);
        cpu.cpu.ds = 0x100;
        cpu.cpu.gpr[0] = 0x8387;
        cpu.cpu.gpr[1] = 0x0003;
        cpu.cpu.gpr[2] = 0x0010;
        cpu.cpu.gpr[3] = 0x0010;
        vm.write_bytes(0x1010, &[0x5a, 0xa5, 0x11, 0x22]).unwrap();
        let r = run(&vm, &mut cpu, 1);
        (
            r.steps(),
            format!("{:?}", r.stop()),
            cpu.cpu.rip,
            cpu.cpu.gpr,
            flags(&cpu),
            vm.mem.read(0x1010, 4).unwrap(),
            vm.mem.read(0x7ff0, 8).unwrap() | vm.mem.read(0x7ff8, 8).unwrap(),
        )
    };
    let executes: &[&[u8]] = &[
        b"\x0f\x10\xc0",
        b"\x0f\x1c\xc3\x05",
        b"\x0f\x16\xc3",
        b"\x0f\x28\xc3",
        b"\x66\xc0",
        b"\xc1\xe0\x03",
        b"\x6b\xc3\x05",
        // CLR1, SET1, NOT1 on memory, CL and imm forms, byte and word: the
        // atomic path (one AND, OR or XOR of the bit mask; pp. 12-85–98).
        b"\x0f\x12\x07",
        b"\x0f\x13\x07",
        b"\x0f\x14\x07",
        b"\x0f\x15\x07",
        b"\x0f\x16\x07",
        b"\x0f\x17\x07",
        b"\x0f\x1a\x07\x06",
        b"\x0f\x1b\x07\x0e",
        b"\x0f\x1c\x07\x05",
        b"\x0f\x1d\x07\x09",
        b"\x0f\x1e\x07\x00",
        b"\x0f\x1f\x07\x0f",
        b"\x26\x0f\x16\x07",
    ];
    let refused_forms: &[&[u8]] = &[
        b"\x65\xa6",
        b"\x64\xa7",
        b"\xf3\x6c",
        b"\xf3\x6e",
        b"\x6c",
        b"\x6f",
        b"\x68\x34\x12",
        b"\x6a\x05",
        b"\x60",
        b"\x61",
        b"\xc8\x04\x00\x00",
        b"\xc9",
        b"\x62\x07",
        b"\xc0\x27\x03",
        b"\x69\x07\x34\x12",
        b"\x0f\x18\x07\x03",
        b"\x0f\x28\x07",
        b"\x0f\x20",
        b"\x0f\x22",
        b"\x0f\x26",
        b"\x0f\x31\xc0",
        b"\x0f\xff\x10",
        b"\x66\x07",
    ];
    for model in MODELS {
        for code in executes {
            let plain = state(model, code);
            let mut locked = state(model, &[&[0xf0][..], code].concat());
            assert_eq!(locked.0, 1, "{model:?} {code:02x?}: {}", locked.1);
            locked.2 -= 1;
            assert_eq!(locked, plain, "{model:?} {code:02x?}");
        }
        for code in refused_forms {
            for prefixes in [&b"\xf0"[..], b"\xf0\x26", b"\x26\xf0"] {
                let locked = [prefixes, code].concat();
                let (steps, stop, rip, ..) = state(model, &locked);
                assert!(
                    stop.contains("UnknownInstruction") && (steps, rip) == (0, IP),
                    "{model:?} {locked:02x?}: {stop}"
                );
                let r = state(model, &locked);
                assert_eq!((r.5, r.6), (0x2211_a55a, 0), "{model:?} {locked:02x?}");
            }
        }
        // REPC after BUSLOCK, in either order.
        for code in [&b"\x65\xf0\xa6"[..], b"\xf0\x65\x26\xa6"] {
            assert_refused(model, code);
        }
        // At an odd address inside one aligned 16-byte granule a word runs on
        // its atomic path, a swap of the containing word (xj-0w3.8), as a
        // byte does.
        for (code, word_form) in [
            (&b"\xf0\x0f\x17\x07"[..], true),
            (b"\xf0\x0f\x16\x07", false),
        ] {
            let (vm, mut cpu) = setup(model, code);
            cpu.cpu.gpr[1] = 9;
            cpu.cpu.gpr[3] = 0x11;
            vm.write_bytes(0x11, &[0x00, 0x00]).unwrap();
            let r = run(&vm, &mut cpu, 1);
            if word_form {
                assert_eq!((r.steps(), word(&vm, 0x11)), (1, 0x0200), "{model:?}");
            } else {
                assert_eq!((r.steps(), byte(&vm, 0x11)), (1, 0x02), "{model:?}");
            }
        }
    }
}

// xj-50w.12 (re-review): BUSLOCK NOT1 byte [BX] (p. 12-173; NOT1 pp. 12-95–98)
// from two vCPUs on different bits, an odd number of times each. The locked
// form is one atomic XOR, so no update is lost and both bits end set. The
// unlocked form is a control only: its read-modify-write may lose an update,
// so its result is not asserted.
#[test]
fn buslock_not1_from_two_vcpus_loses_no_update() {
    const N: u64 = 20_001;
    for model in MODELS {
        for locked in [true, false] {
            for _ in 0..3 {
                let (vm, _) = setup(model, b"\xf4");
                for (at, bit) in [(0x1300u64, 0u8), (0x1400, 1)] {
                    let mut code = vec![];
                    if locked {
                        code.push(0xf0);
                    }
                    // NOT1 byte [BX], imm3; LOOP back; HLT.
                    code.extend_from_slice(&[0x0f, 0x1e, 0x07, bit]);
                    let back = -(code.len() as i8 + 2);
                    code.extend_from_slice(&[0xe2, back as u8, 0xf4]);
                    vm.write_bytes(at, &code).unwrap();
                }
                vm.mem.write(0x2000, 0, 1).unwrap();
                std::thread::scope(|scope| {
                    for at in [0x1300u64, 0x1400] {
                        let vm = &vm;
                        scope.spawn(move || {
                            let mut cpu = vm.new_vcpu();
                            cpu.cpu.rip = at;
                            cpu.cpu.gpr[4] = 0x8000;
                            cpu.cpu.gpr[3] = 0x2000;
                            cpu.cpu.gpr[1] = N;
                            let r = cpu.run_steps(vm, 3 * N, StepOptions::default()).unwrap();
                            assert!(
                                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                                "{model:?}: {:?}",
                                r.stop()
                            );
                        });
                    }
                });
                if locked {
                    assert_eq!(byte(&vm, 0x2000), 0x03, "{model:?}");
                }
            }
        }
    }
}
