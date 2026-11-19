//! Real-address-mode stack frames that wrap SP inside the stack segment, on
//! the I80286 and I80386 Machine models.
//!
//! Both PRMs name the only SP values at which a real-address interrupt or
//! PUSHA shuts down: SP = 1, 3 or 5 (I286 PRM App. B INT p. B-50, PUSHA
//! p. B-88; 80386 PRM §17 INT p. 17-73, PUSHA p. 17-134). At those values one
//! frame word itself crosses offset FFFFh (80386 PRM §14.7 item 7 p. 14-6).
//! Any other SP wraps from 0000h to FFFEh and the frame completes. The pops
//! and far CALL follow the same rule: interrupt 13 only for a word at offset
//! FFFFh (I286 PRM App. B CALL p. B-26, IRET p. B-53, POPA p. B-85; 80386 PRM
//! §17 CALL p. 17-42, IRET p. 17-79, POPA p. 17-129, RET p. 17-145, Table
//! 14-2 p. 14-8). SDM Vol 2A §3.3 INT, REAL-ADDRESS-MODE p. 3-470 checks room
//! for the whole six-byte frame instead; decision-24 keeps the PRM rule.
//!
//! The frozen Real16 alias keeps the whole-frame checks (decision-23 §6;
//! `legacy_u3_review`). The 8086 family has no shutdown and no limit: its
//! offsets wrap modulo 64K (8086 Family UM §2.3 p. 2-12).
use x86jit_core::{
    CpuModel, CpuState, ExceptionDelivery, Exit, Knowledge, Prot, RegionKind, SegmentId,
    StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

const MODELS: [CpuModel; 2] = [CpuModel::I80286, CpuModel::I80386];
const MEMORY: u64 = 0x200000;
const REAL_CS: u16 = 0x1000;
const REAL_SS: u16 = 0x2000;
const SS_BASE: u64 = 0x20000;
const HANDLER: u64 = 0x3000;
/// 0F FF is unassigned on both models: #UD (I286 PRM §9.6.1 p. 9-10;
/// 80386 PRM §14.6 Table 14-1 p. 14-5).
const UD: &[u8] = b"\x0f\xff";
const INT21: &[u8] = b"\xcd\x21";
const NOP: &[u8] = b"\x90";
const IRQ: u8 = 0x30;

/// What raises the event under test.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Event {
    Software,
    Exception,
    Irq,
}

impl Event {
    const ALL: [Self; 3] = [Self::Software, Self::Exception, Self::Irq];

    fn code(self) -> &'static [u8] {
        match self {
            Self::Software => INT21,
            Self::Exception => UD,
            Self::Irq => NOP,
        }
    }

    /// The saved IP: past INT n, at the faulting instruction (I286 PRM
    /// App. C item 3 p. C-1), at the boundary the IRQ is taken on.
    fn saved_ip(self) -> u16 {
        match self {
            Self::Software => 2,
            Self::Exception | Self::Irq => 0,
        }
    }
}

/// `code` at 1000:0000, SS = 2000h, every IVT entry -> 0000:3000 (HLT),
/// in-guest delivery, IF = 1 and CF = 1 so the FLAGS image is not zero.
fn real(model: CpuModel, code: &[u8], sp: u64) -> (Vm, Vcpu) {
    real_with(model, code, sp, ExceptionDelivery::InGuest, None)
}

/// `real`, with the given delivery and an optional device (`Trap`) page.
fn real_with(
    model: CpuModel,
    code: &[u8],
    sp: u64,
    delivery: ExceptionDelivery,
    device: Option<u64>,
) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(MEMORY));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(delivery);
    match device {
        None => vm
            .map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
            .unwrap(),
        Some(page) => {
            vm.map(0, page as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.map(page, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
            vm.map(
                page + 0x1000,
                (MEMORY - page - 0x1000) as usize,
                Prot::RWX,
                RegionKind::Ram,
            )
            .unwrap();
        }
    }
    vm.write_bytes(u64::from(REAL_CS) << 4, code).unwrap();
    for vector in 0..256u64 {
        vm.write_bytes(vector * 4, &[0x00, 0x30, 0x00, 0x00])
            .unwrap();
    }
    vm.write_bytes(HANDLER, b"\xf4").unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([REAL_CS, 0, 0, REAL_SS, 0, 0]);
    cpu.cpu.rip = 0;
    cpu.cpu.gpr[4] = sp;
    cpu.cpu.flags.if_ = true;
    cpu.cpu.flags.cf = true;
    (vm, cpu)
}

fn event(model: CpuModel, event: Event, sp: u64) -> (Vm, Vcpu) {
    let (vm, mut cpu) = real(model, event.code(), sp);
    if event == Event::Irq {
        assert!(cpu.irq_ready());
        cpu.inject_irq(IRQ);
    }
    (vm, cpu)
}

fn word(vm: &Vm, offset: u64) -> u16 {
    let mut bytes = [0; 2];
    vm.read_bytes(SS_BASE + (offset & 0xffff), &mut bytes)
        .unwrap();
    u16::from_le_bytes(bytes)
}

fn stack(vm: &Vm) -> Vec<u8> {
    let mut bytes = vec![0; 0x10000];
    vm.read_bytes(SS_BASE, &mut bytes).unwrap();
    bytes
}

/// Run to the handler's HLT.
fn halts(label: &str, vm: &Vm, cpu: &mut Vcpu) {
    let result = cpu
        .run_steps(vm, 16, x86jit_core::StepOptions::default())
        .unwrap();
    assert!(
        matches!(result.stop(), StepStop::Exit(Exit::Hlt)),
        "{label}: {:?}",
        result.stop()
    );
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, HANDLER + 1), "{label}");
    assert!(!cpu.cpu.flags.if_, "{label}: IF");
}

/// Shutdown, with nothing retired and nothing stored on the stack.
fn shuts_down(label: &str, vm: &Vm, cpu: &mut Vcpu) {
    let image = stack(vm);
    let result = cpu
        .run_steps(vm, 16, x86jit_core::StepOptions::default())
        .unwrap();
    assert!(
        matches!(result.stop(), StepStop::Exit(Exit::Shutdown { .. })),
        "{label}: {:?}",
        result.stop()
    );
    assert_eq!(cpu.retired_instructions(), 0, "{label}");
    assert!(stack(vm) == image, "{label}: a failed frame left stores");
}

#[test]
fn interrupt_frame_at_an_even_sp_wraps_and_completes() {
    // I286 PRM App. B INT p. B-50; 80386 PRM §17 INT p. 17-73: SP = 2 and 4
    // are not shutdown values. FLAGS, CS and IP go to SP-2, SP-4 and SP-6,
    // each wrapped to 16 bits.
    for model in MODELS {
        for event_kind in Event::ALL {
            for (sp, slots) in [
                (2u64, [0x0000, 0xfffe, 0xfffc]),
                (4, [0x0002, 0x0000, 0xfffe]),
            ] {
                let label = format!("{model:?} {event_kind:?} SP={sp}");
                let (vm, mut cpu) = event(model, event_kind, sp);
                let flags = cpu.cpu.rflags() as u16 & 0x0fd7;
                halts(&label, &vm, &mut cpu);
                assert_eq!(cpu.cpu.gpr[4], slots[2], "{label}: SP");
                assert_eq!(word(&vm, slots[0]) & 0x0fd7, flags, "{label}: FLAGS");
                assert_eq!(word(&vm, slots[1]), REAL_CS, "{label}: CS");
                assert_eq!(word(&vm, slots[2]), event_kind.saved_ip(), "{label}: IP");
            }
        }
    }
}

#[test]
fn interrupt_frame_word_across_ffffh_shuts_down() {
    // SP = 1, 3, 5: FLAGS, CS or IP straddles FFFFh/0000h; its fault is
    // delivered on the same SP, crosses again, and so does #DF's: shutdown
    // (I286 PRM App. B INT p. B-50, §9.6.2 p. 9-10; 80386 PRM §17 INT
    // p. 17-73).
    for model in MODELS {
        for event_kind in Event::ALL {
            for sp in [1u64, 3, 5] {
                let label = format!("{model:?} {event_kind:?} SP={sp}");
                let (vm, mut cpu) = event(model, event_kind, sp);
                shuts_down(&label, &vm, &mut cpu);
            }
        }
    }
}

#[test]
fn pusha_at_an_even_sp_wraps_and_completes() {
    // I286 PRM App. B PUSHA p. B-88; 80386 PRM §17 PUSHA p. 17-134: only
    // SP = 1, 3, 5 (shutdown) and 7..15 odd (exception 13) are exceptions.
    // AX, CX, DX, BX, the original SP, BP, SI, DI go to SP-2 .. SP-16.
    for model in MODELS {
        for sp in [2u64, 4, 6, 8, 10, 12, 14] {
            let label = format!("{model:?} PUSHA SP={sp}");
            let (vm, mut cpu) = real(model, b"\x60\xf4", sp);
            for i in [0, 1, 2, 3, 5, 6, 7] {
                cpu.cpu.gpr[i] = 0x1110 + i as u64;
            }
            let result = cpu
                .run_steps(&vm, 16, x86jit_core::StepOptions::default())
                .unwrap();
            assert!(
                matches!(result.stop(), StepStop::Exit(Exit::Hlt)),
                "{label}: {:?}",
                result.stop()
            );
            assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (REAL_CS, 2), "{label}");
            assert_eq!(cpu.cpu.gpr[4], sp.wrapping_sub(16) & 0xffff, "{label}: SP");
            for (slot, reg) in (1u64..).zip([0usize, 1, 2, 3, 4, 5, 6, 7]) {
                let value = if reg == 4 { sp } else { 0x1110 + reg as u64 };
                assert_eq!(
                    u64::from(word(&vm, sp.wrapping_sub(2 * slot))),
                    value,
                    "{label}: slot {slot}"
                );
            }
        }
    }
}

#[test]
fn pushad_at_a_dword_aligned_sp_wraps_and_completes() {
    // 80386 PRM §17 PUSHA p. 17-134 names no exception at SP = 8; no dword
    // crosses FFFFh (§14.7 item 7 p. 14-6). EAX .. EDI go to SP-4 .. SP-32.
    let (vm, mut cpu) = real(CpuModel::I80386, b"\x66\x60\xf4", 8);
    for i in [0, 1, 2, 3, 5, 6, 7] {
        cpu.cpu.gpr[i] = 0x1111_0000 + i as u64;
    }
    let result = cpu
        .run_steps(&vm, 16, x86jit_core::StepOptions::default())
        .unwrap();
    assert!(
        matches!(result.stop(), StepStop::Exit(Exit::Hlt)),
        "{:?}",
        result.stop()
    );
    assert_eq!(cpu.cpu.gpr[4], 0xffe8);
    for (slot, reg) in (1u64..).zip([0usize, 1, 2, 3, 4, 5, 6, 7]) {
        let at = 8u64.wrapping_sub(4 * slot);
        let value = u64::from(word(&vm, at)) | u64::from(word(&vm, at + 2)) << 16;
        let expected = if reg == 4 {
            8
        } else {
            0x1111_0000 + reg as u64
        };
        assert_eq!(value, expected, "slot {slot}");
    }
}

#[test]
fn pusha_boundary_cases_follow_the_prm() {
    // I286 PRM App. B PUSHA p. B-88; 80386 PRM §17 PUSHA p. 17-134:
    // SP = 1, 3, 5 shut down; SP = 7, 9, 11, 13, 15 raise exception 13 on
    // both models, delivered on the unchanged SP with IP at the PUSHA.
    for model in MODELS {
        for sp in [1u64, 3, 5] {
            let label = format!("{model:?} PUSHA SP={sp}");
            let (vm, mut cpu) = real(model, b"\x60", sp);
            shuts_down(&label, &vm, &mut cpu);
        }
        for sp in [7u64, 9, 11, 13, 15] {
            let label = format!("{model:?} PUSHA SP={sp}");
            let (vm, mut cpu) = real(model, b"\x60", sp);
            // Vector 13 alone leads to the HLT handler.
            for vector in 0..256u64 {
                let target: &[u8] = if vector == 13 {
                    &[0x00, 0x30, 0x00, 0x00]
                } else {
                    &[0x00, 0x40, 0x00, 0x00]
                };
                vm.write_bytes(vector * 4, target).unwrap();
            }
            halts(&label, &vm, &mut cpu);
            assert_eq!(cpu.cpu.gpr[4], sp - 6, "{label}: SP");
            assert_eq!(word(&vm, sp - 4), REAL_CS, "{label}: CS");
            assert_eq!(word(&vm, sp - 6), 0, "{label}: IP");
        }
    }
}

#[test]
fn twenty_bit_models_wrap_without_shutdown() {
    // 8086 Family UM §2.3 p. 2-12: offsets wrap modulo 64K; the family has
    // no shutdown (SDM Vol 3B §25.33.1 p. 25-34). An INT at SP = 1 stores
    // its FLAGS word across FFFFh/0000h and completes.
    for model in [CpuModel::I8086, CpuModel::I8088] {
        let label = format!("{model:?}");
        let (vm, mut cpu) = real(model, INT21, 1);
        halts(&label, &vm, &mut cpu);
        assert_eq!(cpu.cpu.gpr[4], 0xfffb, "{label}: SP");
        assert_eq!(word(&vm, 0xfffd), REAL_CS, "{label}: CS");
        assert_eq!(word(&vm, 0xfffb), 2, "{label}: IP");
    }
}

/// Run until HLT and return where it stopped.
fn run_to_hlt(label: &str, vm: &Vm, cpu: &mut Vcpu) {
    let result = cpu.run_steps(vm, 64, StepOptions::default()).unwrap();
    assert!(
        matches!(result.stop(), StepStop::Exit(Exit::Hlt)),
        "{label}: {:?}",
        result.stop()
    );
}

#[test]
fn interrupt_round_trip_at_an_even_sp() {
    // The handler is IRET. INT 21h and an IRQ at SP = 2 and 4 enter it on a
    // wrapped frame and return through it: IRET pops IP, CS and FLAGS slot by
    // slot (I286 PRM App. B IRET p. B-53; 80386 PRM §17 IRET p. 17-79), and
    // the program reaches its HLT with SP and FLAGS restored.
    for model in MODELS {
        for (event_kind, code, hlt_ip) in [
            (Event::Software, &b"\xcd\x21\xf4"[..], 3),
            (Event::Irq, &b"\x90\xf4"[..], 2),
        ] {
            for sp in [2u64, 4] {
                let label = format!("{model:?} {event_kind:?} SP={sp}");
                let (vm, mut cpu) = real(model, code, sp);
                vm.write_bytes(HANDLER, b"\xcf").unwrap();
                if event_kind == Event::Irq {
                    cpu.inject_irq(IRQ);
                }
                run_to_hlt(&label, &vm, &mut cpu);
                assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (REAL_CS, hlt_ip), "{label}");
                assert_eq!(cpu.cpu.gpr[4], sp, "{label}: SP");
                assert!(cpu.cpu.flags.if_ && cpu.cpu.flags.cf, "{label}: FLAGS");
                assert!(!cpu.has_pending_irq(), "{label}");
            }
        }
    }
}

/// MOV r16, 0 for AX, CX, DX, BX, BP, SI and DI.
const CLOBBER: &[u8] =
    b"\xb8\x00\x00\xb9\x00\x00\xba\x00\x00\xbb\x00\x00\xbd\x00\x00\xbe\x00\x00\xbf\x00\x00";

#[test]
fn pusha_popa_round_trip_at_an_even_sp() {
    // PUSHA wraps (p. B-88; p. 17-134); POPA pops slot by slot and faults
    // only for a word at FFFFh (I286 PRM App. B POPA p. B-85; 80386 PRM §17
    // POPA p. 17-129), so it restores every register the PUSHA saved.
    for model in MODELS {
        for sp in [2u64, 4, 6, 8, 10, 12, 14] {
            let label = format!("{model:?} PUSHA/POPA SP={sp}");
            let code = [b"\x60".as_slice(), CLOBBER, b"\x61\xf4"].concat();
            let (vm, mut cpu) = real(model, &code, sp);
            for i in [0, 1, 2, 3, 5, 6, 7] {
                cpu.cpu.gpr[i] = 0x1110 + i as u64;
            }
            run_to_hlt(&label, &vm, &mut cpu);
            assert_eq!(cpu.cpu.cs, REAL_CS, "{label}");
            for i in [0, 1, 2, 3, 5, 6, 7] {
                assert_eq!(cpu.cpu.gpr[i], 0x1110 + i as u64, "{label}: reg {i}");
            }
            assert_eq!(cpu.cpu.gpr[4], sp, "{label}: SP");
        }
    }
    // POPAD at SP = FFE8h after the PUSHAD at SP = 8 (80386 PRM p. 17-129).
    let code = [b"\x66\x60".as_slice(), CLOBBER, b"\x66\x61\xf4"].concat();
    let (vm, mut cpu) = real(CpuModel::I80386, &code, 8);
    for i in [0, 1, 2, 3, 5, 6, 7] {
        cpu.cpu.gpr[i] = 0x1111_0000 + i as u64;
    }
    run_to_hlt("POPAD", &vm, &mut cpu);
    for i in [0, 1, 2, 3, 5, 6, 7] {
        assert_eq!(cpu.cpu.gpr[i], 0x1111_0000 + i as u64, "POPAD: reg {i}");
    }
    assert_eq!(cpu.cpu.gpr[4], 8);
}

#[test]
fn far_call_and_retf_round_trip_at_an_even_sp() {
    // CALL FAR 1000:0500 at SP = 2 and 4 pushes CS and IP on a wrapped frame
    // (I286 PRM App. B CALL p. B-26; 80386 PRM §17 CALL p. 17-42, Table 14-2
    // p. 14-8); RETF and RETF 4 pop it slot by slot (80386 PRM §17 RET
    // p. 17-145; I286 PRM App. B RET p. B-95 read as a word popped at FFFFh,
    // decision-24).
    for model in MODELS {
        for (ret, release) in [(&b"\xcb"[..], 0u64), (&b"\xca\x04\x00"[..], 4)] {
            for (sp, cs_at, ip_at) in [(2u64, 0x0000, 0xfffe), (4, 0x0002, 0x0000)] {
                let label = format!("{model:?} {ret:02x?} SP={sp}");
                let (vm, mut cpu) = real(model, b"\x9a\x00\x05\x00\x10\xf4", sp);
                vm.write_bytes((u64::from(REAL_CS) << 4) + 0x500, ret)
                    .unwrap();
                run_to_hlt(&label, &vm, &mut cpu);
                assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (REAL_CS, 6), "{label}");
                assert_eq!(cpu.cpu.gpr[4], (sp + release) & 0xffff, "{label}: SP");
                assert_eq!(word(&vm, cs_at), REAL_CS, "{label}: CS slot");
                assert_eq!(word(&vm, ip_at), 5, "{label}: IP slot");
            }
        }
    }
}

#[test]
fn far_call_word_across_ffffh_shuts_down() {
    // SP = 1 and 3: the CS or IP word crosses FFFFh; the fault's own frame
    // crosses on the same SP, then #DF's: shutdown.
    for model in MODELS {
        for sp in [1u64, 3] {
            let label = format!("{model:?} CALL FAR SP={sp}");
            let (vm, mut cpu) = real(model, b"\x9a\x00\x05\x00\x10", sp);
            shuts_down(&label, &vm, &mut cpu);
        }
    }
}

#[test]
fn crossing_word_fault_names_that_word() {
    // Outward delivery of an INT 21h at SP = 1 returns the fault of the
    // FLAGS word at FFFFh, the slot that crosses (80386 PRM §14.7 item 7 p. 14-6):
    // #GP on the 286, #SS on the 386 (I286 PRM App. B MOV p. B-74; 80386 PRM
    // §14.6 Table 14-1 p. 14-5), with that operand as its evidence.
    for (model, vector) in [(CpuModel::I80286, 13), (CpuModel::I80386, 12)] {
        let (vm, mut cpu) = real_with(model, INT21, 1, ExceptionDelivery::Outward, None);
        let result = cpu.run_steps(&vm, 16, StepOptions::default()).unwrap();
        let StepStop::Exit(exit @ Exit::Exception { vector: actual, .. }) = result.stop() else {
            panic!("{model:?}: {:?}", result.stop());
        };
        assert_eq!(*actual, vector, "{model:?}");
        let metadata = exit.fault_metadata().unwrap();
        assert_eq!(
            metadata.operand_base(),
            Knowledge::Known(0xffff),
            "{model:?}"
        );
        assert_eq!(metadata.operand_size(), Knowledge::Known(2), "{model:?}");
        assert_eq!(cpu.cpu.gpr[4], 1, "{model:?}");
    }
}

#[test]
fn pushad_dword_across_ffffh_delivers_ss() {
    // I80386 PUSHAD at SP = 2 and 6: a dword crosses FFFFh, which the PUSHAD
    // list on p. 17-134 does not name; §14.7 item 7 p. 14-6 gives exception
    // 12. Its six-byte frame fits on the same SP and is delivered.
    for sp in [2u64, 6] {
        let label = format!("PUSHAD SP={sp}");
        let (vm, mut cpu) = real(CpuModel::I80386, b"\x66\x60", sp);
        for vector in 0..256u64 {
            let target: &[u8] = if vector == 12 {
                &[0x00, 0x30, 0x00, 0x00]
            } else {
                &[0x00, 0x40, 0x00, 0x00]
            };
            vm.write_bytes(vector * 4, target).unwrap();
        }
        halts(&label, &vm, &mut cpu);
        assert_eq!(cpu.cpu.gpr[4], sp.wrapping_sub(6) & 0xffff, "{label}: SP");
        assert_eq!(word(&vm, sp.wrapping_sub(4)), REAL_CS, "{label}: CS");
        assert_eq!(word(&vm, sp.wrapping_sub(6)), 0, "{label}: IP");
    }
}

#[test]
fn wrapped_frame_follows_the_a20_gate() {
    // SS = FFFFh, SP = 2: the FLAGS word is at linear FFFF0h and the CS and
    // IP words wrap to offsets FFFEh and FFFCh, linear 10FFEEh and 10FFECh
    // with A20 enabled, FFEEh and FFECh with it masked.
    for model in MODELS {
        for (a20, cs_at, ip_at) in [(1u8, 0x10ffee, 0x10ffec), (0, 0xffee, 0xffec)] {
            let label = format!("{model:?} A20={a20}");
            let (vm, mut cpu) = real(model, INT21, 2);
            cpu.cpu.ss = 0xffff;
            cpu.cpu.legacy.segments[SegmentId::Ss as usize].base = 0xffff0;
            cpu.cpu.legacy.a20_enabled = a20;
            halts(&label, &vm, &mut cpu);
            let mut bytes = [0; 2];
            vm.read_bytes(cs_at, &mut bytes).unwrap();
            assert_eq!(u16::from_le_bytes(bytes), REAL_CS, "{label}: CS");
            vm.read_bytes(ip_at, &mut bytes).unwrap();
            assert_eq!(u16::from_le_bytes(bytes), 2, "{label}: IP");
        }
    }
}

#[test]
fn wrapped_frame_on_a_device_stack_page() {
    // SS:0000 is a device page: the FLAGS word at SP = 2 goes to the device
    // as one MMIO write; after its acknowledgment CS and IP go to RAM at
    // FFFEh and FFFCh and the handler is entered.
    for model in MODELS {
        let label = format!("{model:?}");
        let (vm, mut cpu) = real_with(model, INT21, 2, ExceptionDelivery::InGuest, Some(SS_BASE));
        let result = cpu.run_steps(&vm, 16, StepOptions::default()).unwrap();
        assert!(
            matches!(
                result.stop(),
                StepStop::Exit(Exit::MmioWrite { addr, size: 2, .. }) if *addr == SS_BASE
            ),
            "{label}: {:?}",
            result.stop()
        );
        cpu.complete_mmio_write();
        halts(&label, &vm, &mut cpu);
        assert_eq!(word(&vm, 0xfffe), REAL_CS, "{label}: CS");
        assert_eq!(word(&vm, 0xfffc), 2, "{label}: IP");
    }
}

#[test]
fn big_real_stack_wraps_at_two_to_the_32() {
    // I80386 with SS.B = 1 and a 4G limit: the stack pointer is ESP, and an
    // INT at ESP = 2 or 4 wraps through 2^32 (80386 PRM §17 INT p. 17-73
    // names only "SP or ESP = 1, 3, or 5"): CS and IP land below the SS base.
    // The handler's IRET returns through the same wrap to 1000:0002.
    for esp in [2u64, 4] {
        let label = format!("ESP={esp:x}");
        let (vm, mut cpu) = real(CpuModel::I80386, b"\xcd\x21\xf4", esp);
        let ss = &mut cpu.cpu.legacy.segments[SegmentId::Ss as usize];
        ss.attributes |= 0x4000;
        ss.limit = 0xffff_ffff;
        vm.write_bytes(HANDLER, b"\xcf").unwrap();
        run_to_hlt(&label, &vm, &mut cpu);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (REAL_CS, 3, esp),
            "{label}"
        );
        let mut bytes = [0; 2];
        vm.read_bytes(SS_BASE + esp - 4, &mut bytes).unwrap();
        assert_eq!(u16::from_le_bytes(bytes), REAL_CS, "{label}: CS");
        vm.read_bytes(SS_BASE + esp - 6, &mut bytes).unwrap();
        assert_eq!(u16::from_le_bytes(bytes), 2, "{label}: IP");
    }
}
