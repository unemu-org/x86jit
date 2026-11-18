//! 8086/8088 manual witnesses (decision-23 C2, spec.md §17.10).
//! Intel 9800722-03 §2.3 p. 2-12, §2.6 pp. 2-24–28,
//! §2.7 pp. 2-33–48, §4.2 Tables 4-12/4-13 pp. 4-22–36.
use x86jit_core::{
    CpuGeneration, CpuModel, ExceptionDelivery, Exit, FpuModel, InterpreterBackend, LegacyCpuModel,
    ModelSystem, Prot, RegionKind, ResetOptions, StepOptions, StepStop, UnsupportedSystemOperation,
    Vcpu, Vm, VmConfig,
};
#[path = "support/port_block.rs"]
mod port_block;
const MODELS: [CpuModel; 2] = [CpuModel::I8088, CpuModel::I8086];
const IP: u64 = 0x1200;
fn setup(model: CpuModel, code: &[u8]) -> (Vm, Vcpu) {
    setup_delivery(model, code, ExceptionDelivery::InGuest)
}
fn setup_delivery(model: CpuModel, code: &[u8], delivery: ExceptionDelivery) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(delivery);
    vm.map(0, 0x120000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(IP, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = IP;
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}
fn step(vm: &Vm, cpu: &mut Vcpu, count: u64) {
    let r = cpu.run_steps(vm, count, StepOptions::default()).unwrap();
    assert_eq!(r.steps(), count, "{:?}", r.stop());
    assert!(
        matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{:?}",
        r.stop()
    );
}
fn vector(vm: &Vm, n: u8, ip: u16, cs: u16) {
    vm.mem
        .write(u64::from(n) * 4, u64::from(ip) | (u64::from(cs) << 16), 4)
        .unwrap();
    vm.write_bytes((u64::from(cs) << 4) + u64::from(ip), b"\xf4")
        .unwrap();
}
#[test]
fn catalogue_and_bus_siblings() {
    let a = CpuModel::I8088.profile();
    let b = CpuModel::I8086.profile();
    assert_eq!(a.isa, b.isa);
    assert_eq!(a.quirks, b.quirks);
    assert_ne!(a.semantics_id, b.semantics_id);
    for model in MODELS {
        let p = model.profile();
        assert_eq!(p.system, ModelSystem::None);
        assert_eq!(p.physical_bits, 20);
        assert_eq!(p.max_insn_len, None);
        assert!(p.timing.is_none());
        assert_eq!(p.fpu, &[FpuModel::Present, FpuModel::None]);
        assert!(p.quirks.push_sp_after_decrement);
        assert_eq!(p.quirks.shift_count_mask, None);
        assert_eq!(p.quirks.real_flags_set_mask, 0xf000);
        assert_eq!(p.quirks.max_processor_interrupt, 4);
    }
}
#[test]
fn push_sp_stores_decremented_value() {
    // SDM Vol 3B §25.17.1; 286 PRM App. C item 8 p. C-2.
    for model in MODELS {
        for code in [&b"\x54"[..], &b"\xff\xf4"[..]] {
            let (vm, mut cpu) = setup(model, code);
            step(&vm, &mut cpu, 1);
            assert_eq!(cpu.cpu.gpr[4], 0x7ffe);
            assert_eq!(vm.mem.read(0x7ffe, 2).unwrap(), 0x7ffe);
        }
    }
}
#[test]
fn stored_flags_are_ones_and_popf_cannot_change_them() {
    // SDM Vol 3B §25.17.2; UM §2.7 p. 2-33 defines only nine flag bits.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x9d\x9c");
        vm.mem.write(0x8000, 0xffff, 2).unwrap();
        // TF is tested separately; keep this witness outside single-step.
        vm.mem.write(0x8000, 0xfeff, 2).unwrap();
        step(&vm, &mut cpu, 2);
        assert_eq!(vm.mem.read(0x8000, 2).unwrap(), 0xfed7);
    }
}
#[test]
fn shifts_and_rotates_use_all_eight_count_bits() {
    // UM §2.7 p. 2-39: up to 255, CF is the last bit shifted/rotated out.
    for model in MODELS {
        for (opcode, count, input, expected, carry) in [
            (0xe0, 32, 1, 0, false),
            (0xe8, 32, 0x8000, 0, false),
            (0xf8, 255, 0x8000, 0xffff, true),
            (0xc0, 32, 0x8001, 0x8001, true),
            (0xc8, 32, 0x8001, 0x8001, true),
            (0xd0, 32, 1, 0x8000, false),
        ] {
            let (vm, mut cpu) = setup(model, &[0xd3, opcode]);
            cpu.cpu.gpr[0] = input;
            cpu.cpu.gpr[1] = count;
            step(&vm, &mut cpu, 1);
            assert_eq!(cpu.cpu.gpr[0], expected, "{opcode:x} {count}");
            assert_eq!(cpu.cpu.flags.cf, carry, "{opcode:x} {count}");
        }
    }
}
#[test]
fn divide_error_saves_next_ip_and_signed_minimum_is_outside_range() {
    // UM §2.6 p. 2-26, §2.7 p. 2-37 (page images checked).
    for model in MODELS {
        for (code, ax, dx, divisor) in [
            (&b"\xf6\xfb"[..], 0xff80, 0, 1),
            (&b"\xf7\xfb"[..], 0x8000, 0xffff, 1),
            (&b"\x3e\xf7\xf3"[..], 1, 0, 0),
        ] {
            let (vm, mut cpu) = setup(model, code);
            vector(&vm, 0, 0x400, 0);
            cpu.cpu.gpr[0] = ax;
            cpu.cpu.gpr[2] = dx;
            cpu.cpu.gpr[3] = divisor;
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{:?}",
                r.stop()
            );
            assert_eq!(vm.mem.read(0x7ffa, 2).unwrap(), IP + code.len() as u64);
            assert_eq!(vm.mem.read(0x7ffe, 2).unwrap() & 0xf000, 0xf000);
            assert_eq!(cpu.cpu.gpr[0], ax);
            assert_eq!(cpu.cpu.gpr[2], dx);
        }
    }
}
#[test]
fn divide_outward_metadata_names_next_ip() {
    for model in MODELS {
        let (vm, mut cpu) = setup_delivery(model, b"\xf7\xf3", ExceptionDelivery::Outward);
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        let StepStop::Exit(exit @ Exit::Exception { vector: 0, .. }) = r.stop() else {
            panic!("{:?}", r.stop())
        };
        assert_eq!(exit.fault_metadata().unwrap().saved_rip(), IP + 2);
        assert_eq!(r.steps(), 0);
    }
}
#[test]
fn no_system_state_is_consulted_and_a20_cannot_change_the_bus() {
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\xa1\x10\x00");
        cpu.cpu.ds = 0xffff;
        cpu.cpu.legacy.cr0 = u32::MAX;
        cpu.cpu.legacy.cpl = 255;
        cpu.cpu.legacy.idtr.base = u32::MAX;
        cpu.cpu.legacy.idtr.limit = 0;
        cpu.cpu.legacy.segments[0].limit = 0;
        cpu.cpu.legacy.segments[3].attributes = 0;
        vm.mem.write(0, 0x1234, 2).unwrap();
        vm.mem.write(0x100000, 0x5678, 2).unwrap();
        cpu.set_a20_enabled(true);
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x1234);
        cpu.cpu.rip = IP;
        cpu.set_a20_enabled(false);
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x1234);
        assert_eq!(cpu.cpu.legacy.cr0, u32::MAX);
    }
}
#[test]
fn word_and_far_pointer_bytes_wrap_inside_segment() {
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\xa1\xff\xff\xa3\xff\xff\xc5\x1e\xfe\xff");
        cpu.cpu.ds = 0x1000;
        for (at, v) in [(0x1ffff, 0x34), (0x10000, 0x12), (0x20000, 0x99)] {
            vm.mem.write(at, v, 1).unwrap();
        }
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x1234);
        cpu.cpu.gpr[0] = 0xabcd;
        step(&vm, &mut cpu, 1);
        assert_eq!(vm.mem.read(0x1ffff, 1).unwrap(), 0xcd);
        assert_eq!(vm.mem.read(0x10000, 1).unwrap(), 0xab);
        assert_eq!(vm.mem.read(0x20000, 1).unwrap(), 0x99);
        vm.mem.write(0x1fffe, 0x4321, 2).unwrap();
        vm.mem.write(0x10000, 0x3456, 2).unwrap();
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[3], 0x4321);
        assert_eq!(cpu.cpu.ds, 0x3456);
    }
}
#[test]
fn stack_word_wraps_and_ip_wraps_at_ffff() {
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x50\x5b");
        cpu.cpu.ss = 0x3000;
        cpu.cpu.gpr[4] = 1;
        cpu.cpu.gpr[0] = 0x1234;
        step(&vm, &mut cpu, 2);
        assert_eq!(cpu.cpu.gpr[3], 0x1234);
        assert_eq!(cpu.cpu.gpr[4], 1);
        assert_eq!(vm.mem.read(0x3ffff, 1).unwrap(), 0x34);
        assert_eq!(vm.mem.read(0x30000, 1).unwrap(), 0x12);
        cpu.cpu.cs = 0x1000;
        cpu.cpu.rip = 0xffff;
        vm.write_bytes(0x1ffff, b"\xb0").unwrap();
        vm.write_bytes(0x10000, b"\x77\x40").unwrap();
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.rip, 1);
        assert_eq!(cpu.cpu.gpr[0] & 255, 0x77);
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.rip, 2);
    }
}
#[test]
fn prefix_runs_longer_than_fifteen_and_255_have_no_length_fault() {
    for model in MODELS {
        for n in [14, 15, 16, 31, 255, 256, 300] {
            let mut code = vec![0x3e; n];
            code.extend_from_slice(b"\xb0\x73");
            let (vm, mut cpu) = setup(model, &code);
            step(&vm, &mut cpu, 1);
            assert_eq!(cpu.cpu.gpr[0], 0x73);
            assert_eq!(cpu.cpu.rip, IP + n as u64 + 2);
        }
    }
}
#[test]
fn later_and_undefined_encodings_are_typed_refusals() {
    // UM §4.2 Table 4-13 pp. 4-27–35; §2.7 p. 2-42;
    // 286 PRM App. C item 6 p. C-2.
    let forms: &[&[u8]] = &[
        b"\x0f",
        b"\x60",
        b"\x61",
        b"\x62\x07",
        b"\x68\x12\x34",
        b"\x6a\x12",
        b"\x69\xc0\x12\x34",
        b"\x6b\xc0\x12",
        b"\xc0\xe0\x03",
        b"\xc1\xe0\x03",
        b"\x6c",
        b"\x6d",
        b"\x6e",
        b"\x6f",
        b"\xc8\x00\x00\x00",
        b"\xc9",
        b"\x0f\x01\xe0",
        b"\x0f\x01\xf0",
        b"\x0f\x22\xc0",
        b"\x8e\xc8",
        b"\x66\x90",
        b"\x67\x90",
        b"\xd6",
        b"\xf1",
        b"\xd0\xf0",
        b"\xd1\xf0",
        b"\xd2\xf0",
        b"\xd3\xf0",
        b"\x8c\xe0",
        b"\x8f\xc8",
        b"\xc6\xc8\x01",
        b"\xc7\xc8\x01\x00",
        b"\x8d\xc0",
        b"\xc4\xc0",
        b"\xc5\xc0",
        b"\xfe\xd0",
        b"\xff\xf8",
        b"\xff\xd8",
        b"\xff\xe8",
        b"\xf0\xf3\xa5",
        b"\xf3\xd9\xe8",
        b"\xf2\xa4",
        b"\xf2\xab",
        b"\xf3\x90",
        b"\xd4\x09",
        b"\xd5\x0b",
    ];
    for model in MODELS {
        for code in forms {
            let (vm, mut cpu) = setup(model, code);
            let before = cpu.cpu.gpr;
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::UnknownInstruction { .. })),
                "{code:x?}: {:?}",
                r.stop()
            );
            assert_eq!(r.steps(), 0);
            assert_eq!(cpu.cpu.rip, IP);
            assert_eq!(cpu.cpu.gpr, before);
        }
    }
}
#[test]
fn aam_aad_documented_second_byte_and_undefined_flag_policy() {
    // UM §4.2 Table 4-12 p. 4-24 fixes byte 2 = 0Ah.
    // §2.7 pp. 2-37–38 leaves AF/CF/OF undefined; engine preserves them.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\xd4\x0a\xd5\x0a");
        cpu.cpu.gpr[0] = 42;
        cpu.cpu.flags.set_flags16(0x811);
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x0402);
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[0], 42);
        assert!(cpu.cpu.flags.cf && cpu.cpu.flags.af() && cpu.cpu.flags.of);
    }
}
#[test]
fn repeated_word_moves_wrap_both_segments() {
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\xf3\xa5");
        cpu.cpu.ds = 0x1000;
        cpu.cpu.es = 0x3000;
        cpu.cpu.gpr[6] = 0xffff;
        cpu.cpu.gpr[7] = 0xffff;
        cpu.cpu.gpr[1] = 2;
        vm.mem.write(0x1ffff, 0x12, 1).unwrap();
        vm.mem.write(0x10000, 0x34, 1).unwrap();
        vm.mem.write(0x10001, 0x5678, 2).unwrap();
        step(&vm, &mut cpu, 2);
        assert_eq!(vm.mem.read(0x3ffff, 1).unwrap(), 0x12);
        assert_eq!(vm.mem.read(0x30000, 1).unwrap(), 0x34);
        assert_eq!(vm.mem.read(0x30001, 2).unwrap(), 0x5678);
        assert_eq!(cpu.cpu.gpr[1], 0);
        assert_eq!(cpu.cpu.gpr[6], 3);
        assert_eq!(cpu.cpu.gpr[7], 3);
        assert_eq!(cpu.retired_instructions(), 1);
    }
}
#[test]
fn lods_can_repeat_and_retains_the_final_element() {
    // UM §2.7 p. 2-43 explicitly discusses repeated LODS and its final value.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\xf3\xad");
        cpu.cpu.gpr[1] = 2;
        cpu.cpu.gpr[6] = 0x3000;
        vm.mem.write(0x3000, 0x12345678, 4).unwrap();
        step(&vm, &mut cpu, 2);
        assert_eq!(cpu.cpu.gpr[0], 0x1234);
        assert_eq!(cpu.cpu.gpr[1], 0);
        assert_eq!(cpu.cpu.gpr[6], 0x3004);
        assert_eq!(cpu.retired_instructions(), 1);
    }
}
#[test]
fn external_interrupt_precedes_single_step_and_handler_is_steppable() {
    // UM §2.6 Table 2-3 p. 2-26, Single-Step p. 2-28.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x90");
        vector(&vm, 0x20, 0x400, 0x1000);
        vector(&vm, 1, 0x500, 0);
        cpu.cpu.flags.if_ = true;
        cpu.cpu.rflags_system = 0x100;
        cpu.inject_irq(0x20);
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        assert_eq!(cpu.cpu.cs, 0);
        assert_eq!(cpu.cpu.rip, 0x501);
        assert_eq!(vm.mem.read(0x7ff4, 2).unwrap(), 0x400);
        assert_eq!(vm.mem.read(0x7ff6, 2).unwrap(), 0x1000);
        assert_eq!(vm.mem.read(0x7ffa, 2).unwrap(), IP);
        assert_eq!(vm.mem.read(0x7ffe, 2).unwrap() & 0xf000, 0xf000);
        assert_eq!(vm.mem.read(0x7ff8, 2).unwrap() & 0xf000, 0xf000);
        assert!(!cpu.has_pending_irq());
    }
}
#[test]
fn interrupted_rep_resumes_at_last_prefix_in_both_orders() {
    // UM §2.7 p. 2-42; §2.10 p. 2-112 (last segment override loses REP).
    for model in MODELS {
        for (code, saved) in [(b"\x26\xf3\xa4", IP + 1), (b"\xf3\x26\xa4", IP + 1)] {
            let (vm, mut cpu) = setup(model, code);
            vector(&vm, 0x20, 0x400, 0);
            cpu.cpu.flags.if_ = true;
            cpu.cpu.es = 0x1000;
            cpu.cpu.gpr[1] = 3;
            cpu.cpu.gpr[6] = 0x100;
            cpu.cpu.gpr[7] = 0x200;
            vm.mem.write(0x10100, 0x332211, 4).unwrap();
            step(&vm, &mut cpu, 1);
            assert_eq!(cpu.cpu.gpr[1], 2);
            assert_eq!(cpu.cpu.gpr[6], 0x101);
            assert_eq!(cpu.cpu.gpr[7], 0x201);
            cpu.inject_irq(0x20);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{:?}",
                r.stop()
            );
            assert_eq!(vm.mem.read(0x7ffa, 2).unwrap(), saved);
            assert_eq!(cpu.cpu.gpr[1], 2);
        }
    }
}
#[test]
fn empty_socket_and_emulation_are_per_vcpu_engine_policies() {
    // UM §2.7 p. 2-48; iAPX PRM §6.3 p. 6-8: even output ESCs read a word.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\xd9\x3e\x00\x10\xdf\xe0\x9b\xd9\xe8");
        cpu.set_fpu_model(FpuModel::None);
        cpu.cpu.gpr[0] = 0x1234;
        vm.mem.write(0x1000, 0x5678, 2).unwrap();
        let cw = cpu.cpu.fpu_cw;
        step(&vm, &mut cpu, 3);
        assert_eq!(vm.mem.read(0x1000, 2).unwrap(), 0x5678);
        assert_eq!(cpu.cpu.gpr[0], 0x1234);
        assert_eq!(cpu.cpu.fpu_cw, cw);
        cpu.set_fpu_model(FpuModel::Present);
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.fpu_top, 7);
        cpu.cpu.rip = IP + 7;
        cpu.set_fpu_emulation(true);
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::UnknownInstruction { .. })),
            "{:?}",
            r.stop()
        );
        cpu.set_fpu_emulation(false);
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.fpu_top, 6);
        cpu.cpu.rip = IP + 7;
        cpu.cpu.fpu_sw |= 0x80;
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(
            r.stop(),
            StepStop::Exit(Exit::UnknownInstruction { .. })
        )); // pending numeric delivery has no CPU interrupt 16

        // CPU ESC /1 still consumes its complete address when that extension
        // has no x87 meaning (UM §4.2 Table 4-12 p. 4-27).
        let (vm, mut cpu) = setup(model, b"\xd9\x0e\x00\x30");
        cpu.set_fpu_model(FpuModel::None);
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.rip, IP + 4);
    }
}
#[test]
fn memory_shift_uses_unmasked_count_before_the_store() {
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\xd3\x26\x00\x10");
        cpu.cpu.gpr[1] = 32;
        vm.mem.write(0x1000, 0x1234, 2).unwrap();
        step(&vm, &mut cpu, 1);
        assert_eq!(vm.mem.read(0x1000, 2).unwrap(), 0);
    }
}
#[test]
fn undefined_status_flags_have_a_deterministic_engine_value() {
    // UM §2.7 pp. 2-35–40: U is not a value a guest may rely on.
    for model in MODELS {
        for code in [
            &b"\xf7\xe3"[..],
            &b"\xf7\xeb"[..],
            &b"\xf7\xf3"[..],
            &b"\xf7\xfb"[..],
        ] {
            let (vm, mut cpu) = setup(model, code);
            cpu.cpu.gpr[0] = 12;
            cpu.cpu.gpr[3] = 3;
            cpu.cpu.flags.set_flags16(0x8d5);
            step(&vm, &mut cpu, 1);
            let flags = cpu.cpu.flags.to_flags16();
            assert_eq!(flags & 0xd4, 0xd4, "{code:x?}");
            if code[1] >= 0xf0 {
                assert_eq!(flags & 0x801, 0x801);
            }
        }
        let (vm, mut cpu) = setup(model, b"\x25\xff\xff\xd1\xe0\x37\x27");
        cpu.cpu.flags.set_flags16(0x8d5);
        cpu.cpu.gpr[0] = 1;
        step(&vm, &mut cpu, 1);
        assert!(!cpu.cpu.flags.af()); // logical AF = zero policy
        cpu.cpu.flags.set_af(true);
        step(&vm, &mut cpu, 1);
        assert!(cpu.cpu.flags.af());
        cpu.cpu.flags.set_flags16(0x8c4);
        step(&vm, &mut cpu, 1); // AAA preserves OF/SF/ZF/PF
        assert_eq!(cpu.cpu.flags.to_flags16() & 0x8c4, 0x8c4);
        cpu.cpu.flags.of = true;
        step(&vm, &mut cpu, 1);
        assert!(!cpu.cpu.flags.of);
    }
}
#[test]
fn fetch_wraps_twenty_bits_and_smc_refetches_the_following_instruction() {
    for model in MODELS {
        // UM §2.2 p. 2-7: IP is sixteen bits. The physical cache key must
        // use that same offset when the embedder supplies upper storage bits.
        let (vm, mut cpu) = setup(model, b"\x90");
        vm.write_bytes(0x11200, b"\xf4").unwrap();
        cpu.cpu.rip = 0x11200;
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.rip, IP + 1);
        let mut other = vm.new_vcpu();
        other.cpu.cs = 0x1000;
        other.cpu.rip = IP;
        let r = other.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        assert_eq!(other.cpu.rip, IP + 1);

        let (vm, mut cpu) = setup(model, b"\x90");
        cpu.cpu.cs = 0xffff;
        cpu.cpu.ds = 0xffff;
        cpu.cpu.rip = 0x200;
        vm.write_bytes(0x1f0, b"\xc6\x06\x06\x02\x42\xb0\x00\xf4")
            .unwrap();
        vm.write_bytes(0x1001f0, b"\xb0\x11\xf4").unwrap();
        step(&vm, &mut cpu, 2);
        assert_eq!(cpu.cpu.gpr[0] & 255, 0x42);
        assert_eq!(cpu.cpu.rip, 0x207);

        // The shared-cache SMC contract also holds across the physical wrap;
        // decision-23 excludes modelling a hardware prefetch queue.
        let (vm, mut cpu) = setup(model, b"");
        cpu.cpu.cs = 0xffff;
        cpu.cpu.rip = 0xb;
        vm.write_bytes(0xffffb, b"\xc6\x06\x00\x00\x43").unwrap();
        vm.write_bytes(0, b"\x40\xf4").unwrap();
        step(&vm, &mut cpu, 2);
        assert_eq!(cpu.cpu.gpr[0], 0);
        assert_eq!(cpu.cpu.gpr[3], 1);
        assert_eq!(cpu.cpu.rip, 0x11);
    }
}
#[test]
fn software_interrupts_keep_all_eight_vector_bits() {
    // UM §2.6 p. 2-24: software INT supports 0–255 independently of internal vectors.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\xcd\x80");
        vector(&vm, 0x80, 0x600, 0x1000);
        step(&vm, &mut cpu, 1);
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0x1000, 0x600));
        assert_eq!(vm.mem.read(0x7ffa, 2).unwrap(), IP + 2);
        assert_eq!(vm.mem.read(0x7ffe, 2).unwrap() & 0xf000, 0xf000);
    }
}
#[test]
fn nop_and_halt_at_ffff_wrap_the_sequential_ip() {
    for model in MODELS {
        for opcode in [0x90, 0xf4] {
            let (vm, mut cpu) = setup(model, b"\x90");
            cpu.cpu.cs = 0x1000;
            cpu.cpu.rip = 0xffff;
            vm.write_bytes(0x1ffff, &[opcode]).unwrap();
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 1);
            assert_eq!(cpu.cpu.rip, 0);
            if opcode == 0xf4 {
                assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
            }
        }
    }
}
#[test]
fn resumed_multi_prefix_string_loses_the_other_prefix() {
    for model in MODELS {
        for repeat_last in [false, true] {
            let code: &[u8] = if repeat_last {
                b"\x26\xf3\xa4"
            } else {
                b"\xf3\x26\xa4"
            };
            let (vm, mut cpu) = setup(model, code);
            vector(&vm, 0x20, 0x400, 0);
            vm.write_bytes(0x400, b"\xcf").unwrap();
            cpu.cpu.flags.if_ = true;
            cpu.cpu.es = 0x1000;
            cpu.cpu.gpr[1] = 3;
            cpu.cpu.gpr[6] = 0x100;
            cpu.cpu.gpr[7] = 0x200;
            vm.mem.write(0x10100, 0x332211, 4).unwrap();
            vm.mem.write(0x101, 0xbbaa, 2).unwrap();
            step(&vm, &mut cpu, 1);
            cpu.inject_irq(0x20);
            step(&vm, &mut cpu, 1); // IRET
            assert_eq!(cpu.cpu.rip, IP + 1);
            step(&vm, &mut cpu, if repeat_last { 2 } else { 1 });
            assert_eq!(vm.mem.read(0x10200, 1).unwrap(), 0x11);
            assert_eq!(
                vm.mem.read(0x10201, 1).unwrap(),
                if repeat_last { 0xaa } else { 0x22 }
            );
            assert_eq!(cpu.cpu.gpr[1], if repeat_last { 0 } else { 2 });
        }
    }
}
#[test]
fn rep_restart_distinguishes_a_budget_before_the_instruction_from_an_iteration() {
    for model in MODELS {
        for budget in [1, 2] {
            let (vm, mut cpu) = setup(model, b"\x90\x26\xf3\xa4");
            vector(&vm, 0x20, 0x400, 0);
            cpu.cpu.flags.if_ = true;
            cpu.cpu.gpr[1] = 3;
            cpu.cpu.gpr[6] = 0x100;
            cpu.cpu.gpr[7] = 0x300;
            step(&vm, &mut cpu, budget);
            cpu.inject_irq(0x20);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
            assert_eq!(
                vm.mem.read(0x7ffa, 2).unwrap(),
                IP + if budget == 1 { 1 } else { 2 }
            );
            assert_eq!(cpu.cpu.gpr[1], if budget == 1 { 3 } else { 2 });
        }
    }
}

#[test]
fn execution_reset_abandons_the_repeat_prefix_owner() {
    // The engine reset contract abandons pending execution while preserving
    // architectural state; an IRQ at the new entry names its current IP.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x26\xf3\xa4");
        vector(&vm, 0x20, 0x400, 0);
        cpu.cpu.flags.if_ = true;
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[6] = 0x100;
        cpu.cpu.gpr[7] = 0x300;
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[1], 2);
        cpu.reset_execution_state(ResetOptions::default());
        cpu.inject_irq(0x20);
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(vm.mem.read(0x7ffa, 2).unwrap(), IP);
    }
}

#[test]
fn a_full_ip_wrap_inside_one_instruction_ends_the_shared_block() {
    // 286 PRM App. C item 10 p. C-2: 8086 has no length limit;
    // 8086 Family UM §2.2 p. 2-7 and §2.3 p. 2-12: IP/offset wrap.
    // The common engine SMC contract refetches a changed following instruction.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"");
        let mut code = vec![0x26; 0x10000];
        code[0xefff] = 0xa2; // ES: MOV moffs16,AL; operands reuse two prefix bytes.
        vm.write_bytes(0x10000, &code).unwrap();
        cpu.cpu.cs = 0x1000;
        cpu.cpu.es = 0x1000;
        cpu.cpu.rip = 0xf000;
        cpu.cpu.gpr[0] = 0x90;
        step(&vm, &mut cpu, 2);
        assert_eq!(vm.mem.read(0x12626, 1).unwrap(), 0x90);
        assert_eq!(cpu.cpu.rip, 0x2627);
    }
}

#[test]
fn compatibility_aliases_follow_the_accepted_model_policy() {
    // Decision-23 amendment; SDM Vol 3B §25.15 pp. 25-5–6;
    // oracles/ss8088/v2/metadata.json: 82h and F6h/F7h /1 status "alias".
    for model in MODELS {
        let mut pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..8)
            .map(|extension| {
                let modrm = 0xc0 | (extension << 3);
                (vec![0x82, modrm, 0x18], vec![0x80, modrm, 0x18])
            })
            .collect();
        pairs.extend([
            (vec![0xf6, 0xc8, 0x81], vec![0xf6, 0xc0, 0x81]),
            (vec![0xf7, 0xc8, 0x34, 0x12], vec![0xf7, 0xc0, 0x34, 0x12]),
            (
                vec![0xf6, 0x0e, 0x00, 0x40, 0x81],
                vec![0xf6, 0x06, 0x00, 0x40, 0x81],
            ),
            (
                vec![0xf7, 0x0e, 0x00, 0x40, 0x34, 0x12],
                vec![0xf7, 0x06, 0x00, 0x40, 0x34, 0x12],
            ),
        ]);
        for (alias, canonical) in pairs {
            let mut results = Vec::new();
            for bytes in [&alias, &canonical] {
                let (vm, mut cpu) = setup(model, bytes);
                cpu.cpu.gpr[0] = 0x9235;
                cpu.cpu.flags.set_flags16(0x8d5);
                vm.mem.write(0x4000, 0xa55a, 2).unwrap();
                step(&vm, &mut cpu, 1);
                assert_eq!(cpu.cpu.rip, IP + bytes.len() as u64);
                results.push((
                    cpu.cpu.gpr,
                    cpu.cpu.flags.to_flags16(),
                    vm.mem.read(0x4000, 2).unwrap(),
                ));
            }
            assert_eq!(results[0], results[1], "{alias:x?}");
        }
    }
}

// Adversarial review regressions; the same transfers run on both bus siblings.
fn setup_review(model: CpuModel, code: &[u8], device: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(InterpreterBackend));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::InGuest);
    vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(
        0x4000,
        0x1000,
        Prot::RWX,
        if device {
            RegionKind::Trap
        } else {
            RegionKind::Ram
        },
    )
    .unwrap();
    vm.map(0x5000, 0x1b000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(IP, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = IP;
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}

#[test]
fn iret_defers_irq_through_following_instruction() {
    // 8086 Family UM §2.6 p. 2-24: STI and IRET reenable interrupts
    // only after the end of the following instruction (page image checked).
    for model in [CpuModel::I8088, CpuModel::I8086] {
        let (vm, mut cpu) = setup_review(model, b"\xcf", false);
        vm.mem.write(0x8000, 0x2000, 2).unwrap();
        vm.mem.write(0x8002, 0, 2).unwrap();
        vm.mem.write(0x8004, 0x202, 2).unwrap();
        vm.write_bytes(0x2000, b"\x40\x90").unwrap(); // INC AX; NOP
        vm.mem.write(0x80, 0x3000, 4).unwrap();
        vm.write_bytes(0x3000, b"\xf4").unwrap();
        cpu.inject_irq(0x20);
        step(&vm, &mut cpu, 1); // IRET, IF was clear on entry.
        let r = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(cpu.cpu.gpr[0], 1, "IRQ must wait for INC AX");
        assert_eq!(vm.mem.read(0x8000, 2).unwrap(), 0x2001);
    }
}

#[test]
fn wrapped_scalar_and_string_mmio_roundtrips() {
    for model in MODELS {
        for code in [&b"\xa1\x00\x40"[..], &b"\xad"[..], &b"\xf3\xad"[..]] {
            let (vm, mut cpu) = setup_review(model, code, true);
            cpu.cpu.gpr[6] = 0x4000;
            cpu.cpu.gpr[1] = 1;
            // The 8086 moves the even word in one bus cycle (UM §2.3 p. 2-15).
            let reads: &[(u64, u8, u64)] = if model == CpuModel::I8086 {
                &[(0x4000, 2, 0x1234)]
            } else {
                &[(0x4000, 1, 0x34), (0x4001, 1, 0x12)]
            };
            for &(addr, size, value) in reads {
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(
                    matches!(r.stop(), StepStop::Exit(Exit::MmioRead { addr: a, size: n, .. }) if *a == addr && *n == size),
                    "{code:x?} expected {addr:x}: {:?}",
                    r.stop()
                );
                assert_eq!(cpu.cpu.gpr[0], 0);
                cpu.complete_mmio_read(value);
            }
            step(&vm, &mut cpu, 1);
            assert_eq!(cpu.cpu.gpr[0], 0x1234);
            assert_eq!(cpu.retired_instructions(), 1);
        }
        for code in [&b"\xa3\x00\x40"[..], &b"\xab"[..], &b"\xf3\xab"[..]] {
            let (vm, mut cpu) = setup_review(model, code, true);
            cpu.cpu.gpr[7] = 0x4000;
            cpu.cpu.gpr[1] = 1;
            cpu.cpu.gpr[0] = 0x1234;
            let writes: &[(u64, u8, u64)] = if model == CpuModel::I8086 {
                &[(0x4000, 2, 0x1234)]
            } else {
                &[(0x4000, 1, 0x34), (0x4001, 1, 0x12)]
            };
            for &(addr, size, value) in writes {
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(
                    matches!(r.stop(), StepStop::Exit(Exit::MmioWrite { addr: a, size: n, value: v, .. }) if *a == addr && *n == size && *v == value),
                    "{:?}",
                    r.stop()
                );
                cpu.complete_mmio_write();
            }
            step(&vm, &mut cpu, 1);
            assert_eq!(cpu.retired_instructions(), 1);
        }
    }
}

#[test]
fn present_fpu_memory_operand_mmio_is_resumable() {
    for model in MODELS {
        let (vm, mut cpu) = setup_review(model, b"\xd9\x06\x00\x40", true); // FLD m32
        cpu.set_fpu_model(FpuModel::Present);
        // CPU ESC dummy word read, followed by the coprocessor's operand read.
        for value in [0, 0] {
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::MmioRead { .. })),
                "{:?}",
                r.stop()
            );
            cpu.complete_mmio_read(value);
        }
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::MmioRead { .. })),
            "coprocessor read must retain MMIO identity: {:?}",
            r.stop()
        );
    }
}

#[test]
fn string_write_reports_second_mmio_byte() {
    // An odd word is two byte cycles on both models (UM §2.3 p. 2-15).
    for model in MODELS {
        let (vm, mut cpu) = setup_review(model, b"\xab", true);
        cpu.cpu.gpr[7] = 0x4001;
        cpu.cpu.gpr[0] = 0x1234;
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(
            r.stop(),
            StepStop::Exit(Exit::MmioWrite {
                addr: 0x4001,
                size: 1,
                value: 0x34,
                ..
            })
        ));
        cpu.complete_mmio_write();
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(
                r.stop(),
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0x4002,
                    size: 1,
                    value: 0x12,
                    ..
                })
            ),
            "{:?}",
            r.stop()
        );
    }
}

#[test]
fn wrapped_string_fault_evidence_names_bus_transfers() {
    for model in MODELS {
        let (mut vm, mut cpu) = setup_review(model, b"\xa4", true); // MOVSB
        cpu.cpu.gpr[6] = 0x4000;
        cpu.cpu.gpr[7] = 0x6000;
        vm.unmap(0x5000, 0x1b000).unwrap();
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(
            r.stop(),
            StepStop::Exit(Exit::MmioRead { addr: 0x4000, .. })
        ));
        cpu.complete_mmio_read(0x5a);
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        let StepStop::Exit(exit @ Exit::UnmappedMemory { addr: 0x6000, .. }) = r.stop() else {
            panic!("{:?}", r.stop())
        };
        // The architectural evidence must not expose interpreter ledger slots.
        assert_eq!(
            exit.fault_metadata()
                .unwrap()
                .progress()
                .acknowledged_transfers(),
            &[(0x4000, 0x5a)]
        );
    }
}

#[test]
fn completed_rep_iterations_are_in_fault_progress() {
    for model in MODELS {
        let (mut vm, mut cpu) = setup_review(model, b"\xf3\xa4", false);
        vm.unmap(0x5000, 0x1b000).unwrap();
        vm.map(0x6000, 1, Prot::RW, RegionKind::Ram).unwrap();
        vm.write_bytes(0x2000, &[0x11, 0x22]).unwrap();
        cpu.cpu.gpr[6] = 0x2000;
        cpu.cpu.gpr[7] = 0x6000;
        cpu.cpu.gpr[1] = 2;
        let r = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
        let StepStop::Exit(exit @ Exit::UnmappedMemory { addr: 0x6001, .. }) = r.stop() else {
            panic!("{:?}", r.stop())
        };
        assert_eq!(r.steps(), 1);
        assert_eq!(cpu.retired_instructions(), 0);
        assert_eq!(vm.mem.read(0x6000, 1).unwrap(), 0x11);
        assert_eq!(
            (cpu.cpu.gpr[6], cpu.cpu.gpr[7], cpu.cpu.gpr[1]),
            (0x2001, 0x6001, 1)
        );
        assert!(
            exit.fault_metadata().unwrap().progress().rep().is_some(),
            "completed REP element must have restart evidence: {:?}",
            exit.fault_metadata().unwrap()
        );
        let progress = exit.fault_metadata().unwrap().progress();
        assert_eq!(
            progress.registers(),
            x86jit_core::RegisterProgress::RepRestart
        );
        assert_eq!(progress.ram(), x86jit_core::MemoryEffects::PossiblyPartial);
        assert!(progress.acknowledged_transfers().is_empty());
    }
}

#[test]
fn logical_83_forms_sign_extend_the_immediate() {
    // Decision-23 compatibility amendment: SDM Vol 2A §3.3 AND p. 3-60,
    // Vol 2B §4.3 OR p. 4-163, Vol 2D §6.1 XOR p. 6-40; oracles/ss8088/v2/metadata.json: "normal".
    for model in MODELS {
        for (extension, expected) in [(1, 0xffb4), (4, 0x1200), (6, 0xedb4)] {
            for memory in [false, true] {
                let modrm = if memory { 6 } else { 0xc0 } | (extension << 3);
                let mut short = vec![0x83, modrm];
                let mut wide = vec![0x81, modrm];
                if memory {
                    short.extend([0, 0x40]);
                    wide.extend([0, 0x40]);
                }
                short.push(0x80);
                wide.extend([0x80, 0xff]);
                let mut results = Vec::new();
                for code in [short, wide] {
                    let (vm, mut cpu) = setup(model, &code);
                    cpu.cpu.gpr[0] = 0x1234;
                    vm.mem.write(0x4000, 0x1234, 2).unwrap();
                    step(&vm, &mut cpu, 1);
                    assert_eq!(cpu.cpu.rip, IP + code.len() as u64);
                    let value = if memory {
                        vm.mem.read(0x4000, 2).unwrap()
                    } else {
                        cpu.cpu.gpr[0]
                    };
                    assert_eq!(value, expected);
                    results.push((value, cpu.cpu.flags.to_flags16()));
                }
                assert_eq!(results[0], results[1]);
            }
        }
    }
}

#[test]
fn compatibility_amendment_keeps_other_encodings_absent() {
    // Decision-23 amendment: SDM Vol 3B §25.15 pp. 25-5–6 and
    // oracles/ss8088/v2/metadata.json distinguish aliases/normal forms
    // from undocumented behavior. UM §4.2 Tables 4-12/13 pp. 4-22–35.
    let forms = (0x60..=0x6f)
        .map(|opcode| vec![opcode, 0, 0, 0])
        .chain([
            vec![0xc0, 0xf0, 1],
            vec![0xc1, 0xf0, 1],
            vec![0xc8, 0, 0, 0],
            vec![0xc9],
            vec![0xff, 0xf8],
            vec![0xd6],
        ])
        .chain((0xd0..=0xd3).map(|opcode| vec![opcode, 0xf0]));
    for code in forms {
        for model in MODELS {
            let (vm, mut cpu) = setup(model, &code);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::UnknownInstruction { .. })),
                "{model:?} {code:x?}: {:?}",
                r.stop()
            );
            assert_eq!(cpu.cpu.rip, IP);
            assert_eq!(r.steps(), 0);
        }
    }
}

#[test]
fn coprocessor_mmio_read_and_write_complete_once_per_bus_byte() {
    // Engine retry contract for FpuModel::Present (8087 fidelity is C6).
    // CPU ESC dummy word: UM §2.7 p. 2-48; 186 PRM §6.3 p. 6-8.
    for model in MODELS {
        let (vm, mut cpu) = setup_review(model, b"\xd9\x06\x00\x40\xd9\x1e\x08\x40", true);
        cpu.set_fpu_model(FpuModel::Present);
        // The CPU's dummy word is one bus cycle on the 8086 (UM §2.3
        // p. 2-15); the coprocessor's own transfers stay byte-wide (ASSUMED,
        // spec.md §5.2). This pins today's known-divergent behaviour: the
        // load reads 4000h-4001h again after the dummy read, where the 8087
        // captures that word from the dummy read and reads only the rest
        // (iAPX 86/88, 186/188 PRM §6.3 p. 6-8); owned by xj-50w.26.
        let dummy: &[(u64, u8)] = if model == CpuModel::I8086 {
            &[(0x4000, 2)]
        } else {
            &[(0x4000, 1), (0x4001, 1)]
        };
        let coprocessor = [(0x4000, 1), (0x4001, 1), (0x4002, 1), (0x4003, 1)];
        let values = [0, 0, 0x80, 0x3f];
        let reads = dummy
            .iter()
            .map(|&t| (t, 0))
            .chain(coprocessor.iter().zip(values).map(|(&t, v)| (t, v)));
        for ((addr, size), value) in reads {
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::MmioRead { addr: a, size: n, .. }) if *a == addr && *n == size),
                "{:?}",
                r.stop()
            );
            assert_eq!(cpu.retired_instructions(), 0);
            cpu.complete_mmio_read(value);
        }
        step(&vm, &mut cpu, 1);
        let dummy: &[(u64, u8)] = if model == CpuModel::I8086 {
            &[(0x4008, 2)]
        } else {
            &[(0x4008, 1), (0x4009, 1)]
        };
        for &(addr, size) in dummy {
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::MmioRead { addr: a, size: n, .. }) if *a == addr && *n == size),
                "{:?}",
                r.stop()
            );
            cpu.complete_mmio_read(0);
        }
        for (addr, byte) in [(0x4008, 0), (0x4009, 0), (0x400a, 0x80), (0x400b, 0x3f)] {
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::MmioWrite { addr: a, size: 1, value, .. }) if *a == addr && *value == byte),
                "{:?}",
                r.stop()
            );
            assert_eq!(cpu.retired_instructions(), 1);
            cpu.complete_mmio_write();
        }
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.retired_instructions(), 2);
    }
}

#[test]
fn coprocessor_large_image_mmio_fits_the_fixed_retry_ledger() {
    // Engine transport witness: the existing Present x87 environment image can
    // exceed CpuState's 64 ledger entries. No 8087 image fidelity is asserted.
    for model in MODELS {
        let (vm, mut cpu) = setup_review(model, b"\xdd\x36\x00\x40", true); // FNSAVE m94
        cpu.set_fpu_model(FpuModel::Present);
        // The CPU's dummy word: one bus cycle on the 8086 (UM §2.3 p. 2-15).
        let dummy: &[(u64, u8)] = if model == CpuModel::I8086 {
            &[(0x4000, 2)]
        } else {
            &[(0x4000, 1), (0x4001, 1)]
        };
        for &(addr, size) in dummy {
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::MmioRead { addr: a, size: n, .. }) if *a == addr && *n == size)
            );
            cpu.complete_mmio_read(0);
        }
        for addr in 0x4000..0x405e {
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::MmioWrite { addr: a, size: 1, .. }) if *a == addr),
                "{:?}",
                r.stop()
            );
            assert_eq!(cpu.retired_instructions(), 0);
            cpu.complete_mmio_write();
        }
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.retired_instructions(), 1);
    }
}

// Round-3 adversarial review regressions.

#[test]
fn budget_yield_with_pending_irq_keeps_the_prefix_owner() {
    // UM §2.7 p. 2-42 names the last prefix for an interrupt that is taken.
    // A budget yield is an engine boundary and keeps the complete owner
    // (spec.md §17.10). STI's shadow holds the IRQ through one element.
    for model in MODELS {
        let (vm, mut cpu) = setup_review(model, b"\xfb\x26\xf3\xa4\xf4", false);
        vector(&vm, 0x20, 0x3000, 0);
        cpu.cpu.es = 0x1000;
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[6] = 0x100;
        cpu.cpu.gpr[7] = 0x200;
        vm.write_bytes(0x10100, &[0x11, 0x22, 0x33]).unwrap();
        vm.write_bytes(0x100, &[0xaa, 0xbb, 0xcc]).unwrap();
        cpu.inject_irq(0x20);
        step(&vm, &mut cpu, 2); // STI, first element
        assert_eq!(cpu.cpu.gpr[1], 2);
        let rip_at_yield = cpu.cpu.rip;
        // The embedder abandons the pending request (documented reset contract).
        cpu.reset_execution_state(ResetOptions::default());
        let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        let copied = [
            vm.mem.read(0x10200, 1).unwrap(),
            vm.mem.read(0x10201, 1).unwrap(),
            vm.mem.read(0x10202, 1).unwrap(),
        ];
        assert_eq!(
            (rip_at_yield, copied),
            (IP + 1, [0x11, 0x22, 0x33]),
            "{model:?}: yield IP and copied bytes"
        );
    }
}

#[test]
fn budget_yield_with_pending_irq_rep_first_order() {
    // F3 26 A4: the last prefix is the segment override. Resuming there
    // after the abandoned request would execute a single MOVSB.
    for model in MODELS {
        let (vm, mut cpu) = setup_review(model, b"\xfb\xf3\x26\xa4\xf4", false);
        vector(&vm, 0x20, 0x3000, 0);
        cpu.cpu.es = 0x1000;
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[6] = 0x100;
        cpu.cpu.gpr[7] = 0x200;
        vm.write_bytes(0x10100, &[0x11, 0x22, 0x33]).unwrap();
        cpu.inject_irq(0x20);
        step(&vm, &mut cpu, 2);
        let rip_at_yield = cpu.cpu.rip;
        cpu.reset_execution_state(ResetOptions::default());
        let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        assert_eq!(
            (
                rip_at_yield,
                cpu.cpu.gpr[1],
                vm.mem.read(0x10202, 1).unwrap()
            ),
            (IP + 1, 0, 0x33),
            "{model:?}: yield IP, final CX, third byte"
        );
    }
}

#[test]
fn budget_yield_with_pending_irq_then_delivery_names_the_last_prefix() {
    // The same yield followed by delivery: the frame names the last prefix
    // (UM §2.7 p. 2-42), for both prefix orders.
    for model in MODELS {
        for code in [&b"\xfb\x26\xf3\xa4\xf4"[..], &b"\xfb\xf3\x26\xa4\xf4"[..]] {
            let (vm, mut cpu) = setup_review(model, code, false);
            vector(&vm, 0x20, 0x3000, 0);
            cpu.cpu.es = 0x1000;
            cpu.cpu.gpr[1] = 3;
            cpu.cpu.gpr[6] = 0x100;
            cpu.cpu.gpr[7] = 0x200;
            cpu.inject_irq(0x20);
            step(&vm, &mut cpu, 2);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{:?}",
                r.stop()
            );
            assert_eq!(
                vm.mem.read(0x7ffa, 2).unwrap(),
                IP + 2,
                "{model:?} {code:x?}"
            );
            assert_eq!(cpu.cpu.gpr[1], 2);
        }
    }
}

#[test]
fn unmapped_string_fault_does_not_open_an_mmio_retry() {
    // irq_ready (vm.rs) is blocked only by device completions and retries;
    // an unmapped destination answered nothing. Compare the 286 entry.
    let mut seen = Vec::new();
    for model in [CpuModel::I80286, CpuModel::I8088, CpuModel::I8086] {
        for code in [&b"\xa4"[..], &b"\xf3\xa4"[..], &b"\xa2\x00\x60"[..]] {
            // Only 6000h..7000h is absent; the stack and IVT stay mapped.
            let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(InterpreterBackend));
            vm.set_cpu_model(model);
            vm.set_exception_delivery(ExceptionDelivery::InGuest);
            vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x7000, 0x19000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(IP, code).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = IP;
            cpu.cpu.gpr[4] = 0x8000;
            cpu.cpu.gpr[1] = 2;
            cpu.cpu.gpr[6] = 0x100;
            cpu.cpu.gpr[7] = 0x6000;
            cpu.cpu.flags.if_ = true;
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::UnmappedMemory { .. })),
                "{:?}",
                r.stop()
            );
            // The embedder maps the page and queues an IRQ before resuming.
            vm.map(0x6000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
            vector(&vm, 0x20, 0x3000, 0);
            cpu.inject_irq(0x20);
            let ready = cpu.irq_ready();
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            seen.push((
                model,
                code.to_vec(),
                ready,
                format!("{:?}", r.stop()),
                cpu.cpu.rip,
            ));
        }
    }
    let reference: Vec<_> = seen
        .iter()
        .filter(|s| s.0 == CpuModel::I80286)
        .map(|s| (s.2, s.3.clone(), s.4))
        .collect();
    for model in MODELS {
        let got: Vec<_> = seen
            .iter()
            .filter(|s| s.0 == model)
            .map(|s| (s.2, s.3.clone(), s.4))
            .collect();
        assert_eq!(got, reference, "{model:?}");
    }
}

#[test]
fn sti_and_iret_delay_only_when_they_reenable_interrupts() {
    // UM §2.6 p. 2-24 (page image checked): "the STI and IRET instructions
    // will reenable interrupts only after the end of the following
    // instruction". With IF already set neither one reenables interrupts,
    // so a queued IRQ is recognized before the following instruction.
    for model in MODELS {
        for code in [&b"\xcf"[..], &b"\xfb\x40\x43\xf4"[..]] {
            let (vm, mut cpu) = setup_review(model, code, false);
            vm.mem.write(0x8000, 0x2000, 2).unwrap();
            vm.mem.write(0x8002, 0, 2).unwrap();
            vm.mem.write(0x8004, 0x202, 2).unwrap();
            vm.write_bytes(0x2000, b"\x40\x43\x90\x90\xf4").unwrap();
            vector(&vm, 0x20, 0x3000, 0);
            cpu.cpu.flags.if_ = true;
            step(&vm, &mut cpu, 1);
            cpu.inject_irq(0x20);
            let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{:?}",
                r.stop()
            );
            assert_eq!(cpu.cpu.rip, 0x3001);
            assert_eq!(
                cpu.cpu.gpr[0], 0,
                "{model:?} {code:x?}: the IRQ precedes the following instruction"
            );
        }
    }
}

#[test]
#[allow(deprecated)]
fn deprecated_selector_keeps_refusing_the_8086_generation() {
    // Decision-23 Migration: the deprecated selector keeps its C1 behavior
    // until removal; I8088/I8086 are reachable through set_cpu_model only.
    let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(InterpreterBackend));
    let selected = LegacyCpuModel {
        generation: CpuGeneration::I8086,
        model_semantics_id: 0x1234,
    };
    vm.set_legacy_cpu_model(selected);
    vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(IP, b"\x54\xf4").unwrap(); // PUSH SP; HLT
    assert_eq!(vm.cpu_model(), None);
    assert_eq!(vm.legacy_cpu_model(), Some(selected));
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = IP;
    cpu.cpu.gpr[4] = 0x8000;
    let before = format!("{:?}", cpu.cpu);
    let refused = |exit: &Exit| {
        matches!(
            exit,
            Exit::UnsupportedSystemContext {
                operation: UnsupportedSystemOperation::LegacyEntryFacility,
                ..
            }
        )
    };
    for budget in [Some(0), Some(1), None] {
        let exit = cpu.run(&vm, budget);
        assert!(refused(&exit), "{budget:?}: {exit:?}");
    }
    for budget in [0, 2] {
        let r = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
        assert_eq!(r.steps(), 0);
        assert!(
            matches!(r.stop(), StepStop::Exit(exit) if refused(exit)),
            "{:?}",
            r.stop()
        );
    }
    assert_eq!(format!("{:?}", cpu.cpu), before);
    assert_eq!(vm.mem.read(0x7ffe, 2).unwrap(), 0);
}

#[test]
fn flag_restores_loading_tf_trap_after_the_following_instruction() {
    // UM §2.6 p. 2-28: the single-step trap follows an instruction that began
    // with TF set, so the instruction after IRET or POPF loading TF is trapped.
    // IF stays clear, so neither restore arms an interrupt delay.
    for model in MODELS {
        for iret in [false, true] {
            let code: &[u8] = if iret { b"\xcf" } else { b"\x9d\x40\x43\xf4" };
            let (vm, mut cpu) = setup_review(model, code, false);
            if iret {
                vm.mem.write(0x8000, 0x2000, 2).unwrap();
                vm.mem.write(0x8002, 0, 2).unwrap();
                vm.mem.write(0x8004, 0x102, 2).unwrap();
                vm.write_bytes(0x2000, b"\x40\x43\x90\x90\xf4").unwrap();
            } else {
                vm.mem.write(0x8000, 0x102, 2).unwrap();
            }
            vector(&vm, 1, 0x500, 0);
            let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{:?}",
                r.stop()
            );
            assert_eq!(
                (cpu.cpu.rip, cpu.cpu.gpr[0], cpu.cpu.gpr[3]),
                (0x501, 1, 0),
                "{model:?} iret={iret}"
            );
            // Single-step frame: IP after INC AX.
            let (frame, ip) = if iret {
                (0x8000, 0x2001)
            } else {
                (0x7ffc, IP + 2)
            };
            assert_eq!(vm.mem.read(frame, 2).unwrap(), ip, "{model:?} iret={iret}");
        }
    }
}

#[test]
fn irq_after_a_string_fault_names_the_last_prefix_only_after_an_element() {
    // UM §2.7 p. 2-42: an interrupt taken inside a repeated string saves the
    // last prefix. Before its first element the string has not begun, so the
    // frame names the first prefix. ES: REP MOVSB with 6000h..7000h absent.
    for model in MODELS {
        for case in 0..3 {
            let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(InterpreterBackend));
            vm.set_cpu_model(model);
            vm.set_exception_delivery(ExceptionDelivery::InGuest);
            vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x7000, 0x19000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(IP, b"\x26\xf3\xa4\xf4").unwrap();
            vector(&vm, 0x20, 0x3000, 0);
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = IP;
            cpu.cpu.gpr[4] = 0x8000;
            cpu.cpu.gpr[1] = 2;
            cpu.cpu.gpr[6] = 0x100;
            // Case 0 faults on the first element; cases 1 and 2 complete one.
            cpu.cpu.gpr[7] = if case == 0 { 0x6000 } else { 0x5fff };
            cpu.cpu.flags.if_ = true;
            if case == 2 {
                step(&vm, &mut cpu, 1); // one element, then a budget yield
            }
            let r = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
            assert!(
                matches!(
                    r.stop(),
                    StepStop::Exit(Exit::UnmappedMemory { addr: 0x6000, .. })
                ),
                "{:?}",
                r.stop()
            );
            assert_eq!(cpu.cpu.gpr[1], if case == 0 { 2 } else { 1 });
            vm.map(0x6000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
            cpu.inject_irq(0x20);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{:?}",
                r.stop()
            );
            assert_eq!(
                vm.mem.read(0x7ffa, 2).unwrap(),
                if case == 0 { IP } else { IP + 1 },
                "{model:?} case {case}"
            );
        }
    }
}

// xj-50w.15. UM §2.7 p. 2-42: an interrupt taken inside a repeated string
// saves the last prefix only; spec.md §17.10: before its first element the
// frame names the first prefix. The owner/restart pair is keyed on the
// inputs the string reads (CX, SI and its segment, DI and ES, DF; AL/AX for
// STOS and SCAS), as the U3 ENTER identity contract keys an attempt. A host
// edit of one of them between runs starts a new operation at the same
// address; an edit of a register the string does not read keeps the pair.
#[test]
fn host_edit_of_a_string_input_abandons_the_restart_pair() {
    type Edit = fn(&mut x86jit_core::CpuState);
    let cases: [(&str, Edit, u64); 8] = [
        ("none", |_| {}, IP + 1),
        ("cx", |c| c.gpr[1] = 3, IP),
        ("si", |c| c.gpr[6] = 0x180, IP),
        ("di", |c| c.gpr[7] = 0x300, IP),
        ("es", |c| c.es = 0x1100, IP),
        ("df", |c| c.flags.df = true, IP),
        // ES: REP MOVSB reads neither DS nor AX.
        ("ds", |c| c.ds = 0x2000, IP + 1),
        ("ax", |c| c.gpr[0] = 0x55, IP + 1),
    ];
    for model in MODELS {
        for (name, edit, frame) in cases {
            let (vm, mut cpu) = setup(model, b"\x26\xf3\xa4\xf4");
            vector(&vm, 0x20, 0x3000, 0);
            cpu.cpu.flags.if_ = true;
            cpu.cpu.es = 0x1000;
            cpu.cpu.gpr[1] = 3;
            cpu.cpu.gpr[6] = 0x100;
            cpu.cpu.gpr[7] = 0x200;
            step(&vm, &mut cpu, 1); // one element, then a budget yield
            assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[1]), (IP, 2));
            edit(&mut cpu.cpu);
            let cx = cpu.cpu.gpr[1];
            cpu.inject_irq(0x20);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{model:?} {name}: {:?}",
                r.stop()
            );
            assert_eq!(
                (vm.mem.read(0x7ffa, 2).unwrap(), cpu.cpu.gpr[1]),
                (frame, cx),
                "{model:?} {name}"
            );
        }
    }
    // The other inputs, each against a string that reads it and has two
    // prefixes (so the pair's restart differs from IP): DS of a DS source in
    // both prefix orders, SS of an SS source (the frame goes to the new
    // SS:SP), AH of REP STOSW (the value is read at word width; REP STOSB
    // reads AL only), AL of REPNE SCASB, SI of REP LODSB, and DS of a LODSB
    // whose source is CS.
    type Case = (&'static str, &'static [u8], Edit, u64);
    let cases: [Case; 8] = [
        ("ds of F3 3E A4", b"\xf3\x3e\xa4\xf4", |c| c.ds = 0x2000, IP),
        ("ds of 3E F3 A4", b"\x3e\xf3\xa4\xf4", |c| c.ds = 0x2000, IP),
        ("ss of 36 F3 A4", b"\x36\xf3\xa4\xf4", |c| c.ss = 0x10, IP),
        (
            "ah of 2E F3 AB",
            b"\x2e\xf3\xab\xf4",
            |c| c.gpr[0] = 0x5534,
            IP,
        ),
        (
            "ah of 2E F3 AA",
            b"\x2e\xf3\xaa\xf4",
            |c| c.gpr[0] = 0x5534,
            IP + 1,
        ),
        (
            "al of 2E F2 AE",
            b"\x2e\xf2\xae\xf4",
            |c| c.gpr[0] = 0x1235,
            IP,
        ),
        (
            "si of 2E F3 AC",
            b"\x2e\xf3\xac\xf4",
            |c| c.gpr[6] = 0x180,
            IP,
        ),
        (
            "ds of 2E F3 AC",
            b"\x2e\xf3\xac\xf4",
            |c| c.ds = 0x2000,
            IP + 1,
        ),
    ];
    for model in MODELS {
        for (name, code, edit, frame) in cases {
            let (vm, mut cpu) = setup(model, code);
            vector(&vm, 0x20, 0x3000, 0);
            cpu.cpu.flags.if_ = true;
            cpu.cpu.es = 0x1000;
            cpu.cpu.gpr[0] = 0x1234;
            cpu.cpu.gpr[1] = 3;
            cpu.cpu.gpr[6] = 0x100;
            cpu.cpu.gpr[7] = 0x200;
            step(&vm, &mut cpu, 1);
            assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[1]), (IP, 2), "{model:?} {name}");
            edit(&mut cpu.cpu);
            cpu.inject_irq(0x20);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{model:?} {name}: {:?}",
                r.stop()
            );
            let sp = (u64::from(cpu.cpu.ss) << 4) + 0x7ffa;
            assert_eq!(vm.mem.read(sp, 2).unwrap(), frame, "{model:?} {name}");
        }
    }
}

// xj-50w.15, the C2 round-3 review case: a yield after one element, a host
// edit that re-points the string at a fresh operation, a fault on that
// operation's first element, then an IRQ. No element of the edited
// instruction completed, so the frame names the first prefix.
#[test]
fn a_fault_after_a_host_edit_does_not_keep_the_old_restart_pair() {
    for model in MODELS {
        let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(InterpreterBackend));
        vm.set_cpu_model(model);
        vm.set_exception_delivery(ExceptionDelivery::InGuest);
        vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x7000, 0x19000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(IP, b"\x26\xf3\xa4\xf4").unwrap();
        vector(&vm, 0x20, 0x3000, 0);
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = IP;
        cpu.cpu.gpr[4] = 0x8000;
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[6] = 0x100;
        cpu.cpu.gpr[7] = 0x200;
        cpu.cpu.flags.if_ = true;
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.rip, IP);
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[7] = 0x6000;
        let r = cpu.run_steps(&vm, 3, StepOptions::default()).unwrap();
        assert!(
            matches!(
                r.stop(),
                StepStop::Exit(Exit::UnmappedMemory { addr: 0x6000, .. })
            ),
            "{:?}",
            r.stop()
        );
        assert_eq!(r.steps(), 0);
        vm.map(0x6000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        cpu.inject_irq(0x20);
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        assert_eq!(
            (vm.mem.read(0x7ffa, 2).unwrap(), cpu.cpu.gpr[1]),
            (IP, 3),
            "{model:?}"
        );
    }
}

// The same keying for a string that reads AL/AX: an edit of AX abandons a
// REP STOSB pair, while an edit of SI, which STOS does not read, keeps it.
#[test]
fn stos_restart_pair_is_keyed_on_its_value_register() {
    for model in MODELS {
        for (edit_ax, frame) in [(false, IP + 1), (true, IP)] {
            let (vm, mut cpu) = setup(model, b"\x2e\xf3\xaa\xf4");
            vector(&vm, 0x20, 0x3000, 0);
            cpu.cpu.flags.if_ = true;
            cpu.cpu.gpr[0] = 0x5a;
            cpu.cpu.gpr[1] = 3;
            cpu.cpu.gpr[7] = 0x200;
            step(&vm, &mut cpu, 1);
            if edit_ax {
                cpu.cpu.gpr[0] = 0x5b;
            } else {
                cpu.cpu.gpr[6] = 0x700;
            }
            cpu.inject_irq(0x20);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
            assert_eq!(
                vm.mem.read(0x7ffa, 2).unwrap(),
                frame,
                "{model:?} edit AX {edit_ax}"
            );
        }
    }
}

// ---- xj-50w.12: LOCK on the twenty-bit models ----
//
// 8086 Family UM §2.5 p. 2-17: the LOCK signal "remains active throughout
// execution of the instruction that follows the LOCK prefix", and "in the
// case of a locked repeated string instruction, LOCK does remain active for
// the duration of the block operation"; p. 2-18: "The LOCK signal provides
// information only"; §2.7 p. 2-48: LOCK makes the processor "assert its bus
// LOCK signal while the following instruction executes. LOCK does not affect
// any flags"; §4.2 p. 4-14: "Greater latencies occur when a 'long'
// instruction is locked." No page restricts the instructions LOCK may
// precede. NEC UM p. 12-173 BUSLOCK says the same for the V20/V30. The
// engine executes a locked read-modify-write through its atomic path and a
// locked form that transfers no data as the unlocked form; every other
// locked form, and every repeated string, is a typed refusal.
const LOCK_MODELS: [CpuModel; 4] = [
    CpuModel::I8088,
    CpuModel::I8086,
    CpuModel::NecV20,
    CpuModel::NecV30,
];

// The operand is DS:BX = 0100:0010 (physical 1010h); BP points at a second
// word for the r/m forms that name it.
#[derive(Debug, PartialEq)]
struct LockState {
    stop: String,
    steps: u64,
    retired: u64,
    end: u64,
    gpr: [u64; 8],
    flags: u16,
    segments: [u16; 4],
    operand: [u64; 5],
}
fn lock_case(model: CpuModel, code: &[u8], prepare: impl Fn(&Vm, &mut Vcpu)) -> LockState {
    let (vm, mut cpu) = setup(model, code);
    vector(&vm, 0x21, 0x3000, 0);
    cpu.cpu.ds = 0x100;
    cpu.cpu.es = 0x200;
    cpu.cpu.gpr[0] = 0x8387;
    cpu.cpu.gpr[1] = 0x0102;
    cpu.cpu.gpr[2] = 0x0010;
    cpu.cpu.gpr[3] = 0x0010;
    cpu.cpu.gpr[5] = 0x2020;
    cpu.cpu.flags.cf = true;
    vm.write_bytes(0x1010, &[0x5a, 0xa5, 0x11, 0x22]).unwrap();
    vm.write_bytes(0x2010, &[0x96, 0x69, 0x55, 0x66]).unwrap();
    vm.write_bytes(0x2020, &[0xc3, 0x3c, 0x33, 0x44]).unwrap();
    prepare(&vm, &mut cpu);
    let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    let mut gpr = [0; 8];
    gpr.copy_from_slice(&cpu.cpu.gpr[..8]);
    LockState {
        stop: format!("{:?}", r.stop()),
        steps: r.steps(),
        retired: cpu.retired_instructions(),
        end: cpu.cpu.rip,
        gpr,
        flags: cpu.cpu.flags.to_flags16(),
        segments: [cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss],
        operand: [
            vm.mem.read(0x1010, 4).unwrap(),
            vm.mem.read(0x2010, 4).unwrap(),
            vm.mem.read(0x2020, 4).unwrap(),
            vm.mem.read(0x7ff0, 8).unwrap(),
            vm.mem.read(0x7ff8, 8).unwrap(),
        ],
    }
}
// The locked form runs as the unlocked one: one step, the same registers,
// flags and memory, IP one byte further (the prefix).
fn assert_lock_matches_unlocked(model: CpuModel, code: &[u8], prepare: impl Fn(&Vm, &mut Vcpu)) {
    let unlocked = lock_case(model, code, &prepare);
    let mut locked_code = vec![0xf0];
    locked_code.extend_from_slice(code);
    let mut locked = lock_case(model, &locked_code, &prepare);
    assert_eq!(locked.steps, 1, "{model:?} {code:02x?}: {}", locked.stop);
    locked.end -= 1;
    assert_eq!(locked, unlocked, "{model:?} {code:02x?}");
}

#[test]
fn locked_read_modify_write_forms_match_their_unlocked_results() {
    let forms: &[&[u8]] = &[
        b"\x00\x07",
        b"\x01\x07",
        b"\x08\x07",
        b"\x09\x07",
        b"\x20\x07",
        b"\x21\x07",
        b"\x28\x07",
        b"\x29\x07",
        b"\x30\x07",
        b"\x31\x07",
        b"\x80\x07\x5a",
        b"\x80\x0f\x5a",
        b"\x80\x27\x5a",
        b"\x80\x2f\x5a",
        b"\x80\x37\x5a",
        b"\x81\x07\x34\x12",
        b"\x81\x0f\x34\x12",
        b"\x81\x27\x34\x12",
        b"\x81\x2f\x34\x12",
        b"\x81\x37\x34\x12",
        b"\x82\x07\xa6",
        b"\x82\x2f\xa6",
        b"\x83\x07\xff",
        b"\x83\x0f\x80",
        b"\x83\x27\x7f",
        b"\x83\x2f\x01",
        b"\x83\x37\xfe",
        b"\xfe\x07",
        b"\xfe\x0f",
        b"\xff\x07",
        b"\xff\x0f",
        b"\xf6\x17",
        b"\xf6\x1f",
        b"\xf7\x17",
        b"\xf7\x1f",
        b"\x86\x07",
        b"\x87\x07",
        // Other addressing: displacement, BP (SS) and segment overrides.
        b"\x01\x46\x00",
        b"\x26\x01\x47\x10",
        b"\x2e\xff\x87\x00\x10",
        b"\x87\x8f\x00\x00",
    ];
    for model in LOCK_MODELS {
        for code in forms {
            assert_lock_matches_unlocked(model, code, |_, _| {});
            // An operand at an odd address inside one aligned 16-byte granule
            // (1011h, 2021h) runs as above, a word too: its atomic path
            // is a swap of the containing word (xj-0w3.8). The [BP] form keeps
            // its even address.
            assert_lock_matches_unlocked(model, code, |_, cpu| cpu.cpu.gpr[3] = 0x11);
        }
        // Flag inputs: an operand at its boundary values.
        for value in [0u16, 0x7fff, 0x8000, 0xffff, 0x00ff] {
            for code in [
                &b"\xff\x07"[..],
                b"\xff\x0f",
                b"\xf7\x1f",
                b"\x01\x07",
                b"\x29\x07",
            ] {
                assert_lock_matches_unlocked(model, code, |vm, _| {
                    vm.mem.write(0x1010, u64::from(value), 2).unwrap()
                });
            }
        }
        // Prefix order, repeated LOCK and a long prefix run.
        for prefixes in [
            &b"\x26"[..],
            b"\xf0\x26",
            b"\x26\xf0\x26",
            &[0xf0u8; 20][..],
        ] {
            let mut code = prefixes.to_vec();
            code.extend_from_slice(b"\x01\x07");
            let mut plain = code.clone();
            plain.retain(|b| *b != 0xf0);
            let locked = lock_case(model, &[&[0xf0][..], &code].concat(), |_, _| {});
            let unlocked = lock_case(model, &plain, |_, _| {});
            assert_eq!(locked.steps, 1, "{model:?} {code:02x?}");
            assert_eq!(
                (locked.gpr, locked.flags, locked.operand),
                (unlocked.gpr, unlocked.flags, unlocked.operand),
                "{model:?} {code:02x?}"
            );
            assert_eq!(locked.end, IP + code.len() as u64 + 1, "{model:?}");
        }
    }
}

// A locked form whose lowering transfers no data runs as the unlocked one:
// the bus lock "provides information only" (UM p. 2-18) and LOCK "does not
// affect any flags" (p. 2-48). NEC UM p. 2-2 describes BUSLOCK before HALT.
#[test]
fn locked_forms_without_a_data_transfer_execute() {
    let forms: &[&[u8]] = &[
        b"\x90",
        b"\x40",
        b"\x4b",
        b"\x01\xd8",
        b"\x31\xc0",
        b"\x80\xc3\x7f",
        b"\x87\xd8",
        b"\x93",
        b"\xb0\x05",
        b"\x8b\xc3",
        b"\x8d\x07",
        b"\xf7\xd3",
        b"\xf7\xdb",
        b"\xf6\xe3",
        b"\xf7\xf1",
        b"\xd1\xe0",
        b"\xd3\xc8",
        b"\x98",
        b"\x99",
        b"\x9e",
        b"\x9f",
        b"\x27",
        b"\x37",
        b"\xd4\x0a",
        b"\xf8",
        b"\xf9",
        b"\xf5",
        b"\xfc",
        b"\xfd",
        b"\xfa",
        b"\xfb",
        b"\x8e\xc0",
        b"\x8c\xd8",
        b"\x9b",
        // A register ESC: no CPU transfer (UM §2.7 p. 2-48).
        b"\xd9\xc0",
        b"\xd9\xe8",
        b"\xeb\x00",
        b"\x74\x00",
        b"\x75\x00",
        b"\xe3\x00",
    ];
    for model in LOCK_MODELS {
        for code in forms {
            assert_lock_matches_unlocked(model, code, |_, _| {});
        }
        // HLT: the locked halt stops after the whole instruction.
        let r = lock_case(model, b"\xf0\xf4", |_, _| {});
        assert!(r.stop.contains("Hlt"), "{model:?}: {}", r.stop);
        assert_eq!(r.end, IP + 2);
    }
}

// Every other locked form is a typed refusal before it commits anything:
// no step, IP on the first prefix, registers and memory unchanged.
#[test]
fn locked_forms_without_an_atomic_path_are_typed_refusals() {
    let forms: &[&[u8]] = &[
        // Repeated strings (decision: they stay typed handoffs).
        b"\xf3\xa4",
        b"\xf3\xa5",
        b"\xf3\xaa",
        b"\xf3\xac",
        b"\xf3\xa6",
        b"\xf2\xa6",
        b"\xf3\xae",
        b"\xf2\xaf",
        b"\x26\xf3\xa4",
        b"\xf3\x26\xa5",
        // Single strings.
        b"\xa4",
        b"\xa5",
        b"\xaa",
        b"\xac",
        b"\xa6",
        b"\xae",
        // A read-modify-write without an atomic form.
        b"\x10\x07",
        b"\x11\x07",
        b"\x18\x07",
        b"\x19\x07",
        b"\x80\x17\x01",
        b"\x80\x1f\x01",
        b"\x81\x17\x01\x00",
        b"\x83\x1f\x01",
        b"\xd0\x27",
        b"\xd1\x07",
        b"\xd3\x2f",
        b"\xd2\x1f",
        // Only a read or only a write of memory.
        b"\x02\x07",
        b"\x03\x07",
        b"\x2b\x07",
        b"\x38\x07",
        b"\x39\x07",
        b"\x3a\x07",
        b"\x80\x3f\x01",
        b"\x83\x3f\x01",
        b"\x84\x07",
        b"\xf6\x07\x01",
        b"\xf7\x07\x01\x00",
        b"\x88\x07",
        b"\x89\x07",
        b"\x8a\x07",
        b"\x8b\x07",
        b"\xa0\x10\x00",
        b"\xa3\x10\x00",
        b"\xc6\x07\x01",
        b"\xc7\x07\x01\x00",
        b"\x8c\x07",
        b"\x8e\x07",
        b"\xc4\x07",
        b"\xc5\x07",
        b"\xd7",
        b"\xf6\x27",
        b"\xf7\x37",
        b"\xd9\x07",
        b"\xdd\x07",
        // Stack and control transfers through memory.
        b"\x50",
        b"\x5b",
        b"\x06",
        b"\x1f",
        b"\x9c",
        b"\x9d",
        b"\xff\x37",
        b"\x8f\x07",
        b"\xe8\x00\x00",
        b"\xff\xd0",
        b"\xff\x17",
        b"\xff\x1f",
        b"\xff\x27",
        b"\xff\x2f",
        b"\x9a\x00\x00\x00\x00",
        b"\xc3",
        b"\xc2\x02\x00",
        b"\xcb",
        b"\xcf",
        b"\xcd\x21",
        b"\xcc",
        b"\xce",
        // Port transfers.
        b"\xe4\x10",
        b"\xe5\x10",
        b"\xe6\x10",
        b"\xe7\x10",
        b"\xec",
        b"\xed",
        b"\xee",
        b"\xef",
    ];
    for model in LOCK_MODELS {
        for code in forms {
            let mut locked = vec![0xf0];
            locked.extend_from_slice(code);
            let (vm, mut cpu) = setup(model, &locked);
            cpu.cpu.flags.of = true;
            cpu.cpu.ds = 0x100;
            cpu.cpu.gpr[3] = 0x10;
            cpu.cpu.gpr[1] = 2;
            vm.write_bytes(0x1010, &[0x5a, 0xa5, 0, 0x10]).unwrap();
            let before = (cpu.cpu.gpr, cpu.cpu.flags.to_flags16());
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::UnknownInstruction { addr, .. }) if *addr == IP),
                "{model:?} {locked:02x?}: {:?}",
                r.stop()
            );
            assert_eq!(r.steps(), 0, "{model:?} {locked:02x?}");
            assert_eq!(cpu.cpu.rip, IP);
            assert_eq!((cpu.cpu.gpr, cpu.cpu.flags.to_flags16()), before);
            assert_eq!(vm.mem.read(0x1010, 4).unwrap(), 0x1000_a55a);
            assert_eq!(
                (
                    vm.mem.read(0x7ff0, 8).unwrap(),
                    vm.mem.read(0x7ff8, 8).unwrap()
                ),
                (0, 0),
                "{model:?} {locked:02x?}"
            );
        }
    }
}

// A refused locked form after an earlier instruction of the same block: the
// earlier instruction retires, IP names the locked one.
#[test]
fn a_refused_locked_form_ends_the_block_before_it() {
    for model in LOCK_MODELS {
        let (vm, mut cpu) = setup(model, b"\x40\xf0\x89\x07\xf4");
        let r = cpu.run_steps(&vm, 5, StepOptions::default()).unwrap();
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::UnknownInstruction { addr, .. }) if *addr == IP + 1),
            "{model:?}: {:?}",
            r.stop()
        );
        assert_eq!((r.steps(), cpu.cpu.rip, cpu.cpu.gpr[0]), (1, IP + 1, 1));
    }
}

// The atomic path makes one host atomic of the operand (Memory::atomic_rmw):
// two vCPUs incrementing one aligned word lose no update. A word at an odd
// address is refused (a_locked_word_at_an_odd_address_has_no_atomic_path).
#[test]
fn locked_increments_from_two_vcpus_lose_no_update() {
    const N: u64 = 20_000;
    for model in LOCK_MODELS {
        // LOCK INC word [2000h]; LOOP back; HLT.
        let (vm, _) = setup(model, b"\xf0\xff\x06\x00\x20\xe2\xf9\xf4");
        vm.mem.write(0x2000, 0, 2).unwrap();
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    let mut cpu = vm.new_vcpu();
                    cpu.cpu.rip = IP;
                    cpu.cpu.gpr[4] = 0x8000;
                    cpu.cpu.gpr[1] = N;
                    let r = cpu.run_steps(&vm, 3 * N, StepOptions::default()).unwrap();
                    assert!(
                        matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                        "{:?}",
                        r.stop()
                    );
                });
            }
        });
        assert_eq!(
            vm.mem.read(0x2000, 2).unwrap(),
            2 * N % 0x10000,
            "{model:?}"
        );
    }
}

// A locked device operand inside one Trap region is one read, then one write,
// of the whole operand (spec.md §5.2, the maintainer's MMIO rule): the bus lock
// holds through the instruction (UM §2.5 p. 2-18; SDM Vol 3A §11.1.2.2), and
// the vCPU is stopped across the exits. Each transfer follows the model's data
// bus, as every device transfer of the twenty-bit walk does (xj-50w.20): a word
// at an even address is one transfer on the 8086 and V30 (UM §2.3 p. 2-15; NEC
// UM §4 pp. 4-2–4-3), a byte at a time on the 8088 and V20. The instruction
// retires once, after the last write is acknowledged. The unlocked form reaches
// the device the same way.
#[test]
fn a_locked_device_operand_reads_then_writes_once() {
    // One device transfer: (write, address, size, value).
    type Transfer = (bool, u64, u8, u64);
    // Answer each device stop of `run_steps(1)` until the instruction retires.
    fn drive(vm: &Vm, cpu: &mut Vcpu) -> Vec<Transfer> {
        let mut bus = Vec::new();
        for _ in 0..8 {
            let r = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
            match r.stop() {
                StepStop::Exit(Exit::MmioRead { addr, size, .. }) => {
                    assert_eq!(r.steps(), 0);
                    // Device bytes 4000h + i = 30h + i.
                    let value = (0..u64::from(*size))
                        .map(|i| (0x30 + ((*addr + i) & 0xff)) << (8 * i))
                        .sum();
                    bus.push((false, *addr, *size, value));
                    cpu.complete_mmio_read(value);
                }
                StepStop::Exit(Exit::MmioWrite {
                    addr, size, value, ..
                }) => {
                    assert_eq!(r.steps(), 0);
                    bus.push((true, *addr, *size, *value));
                    cpu.complete_mmio_write();
                }
                _ if r.steps() == 1 => return bus,
                other => panic!("{other:?}"),
            }
        }
        panic!("{bus:x?}")
    }
    // Device bytes 4000h = 30h, 4001h = 31h; AX = 1234h. Each case gives the
    // eight-bit bus transfers, then the sixteen-bit bus transfers.
    type Case<'a> = (&'a [u8], &'a [Transfer], &'a [Transfer], u64);
    let cases: [Case; 3] = [
        // LOCK INC BYTE [BX]
        (
            b"\xf0\xfe\x07",
            &[(false, 0x4000, 1, 0x30), (true, 0x4000, 1, 0x31)],
            &[(false, 0x4000, 1, 0x30), (true, 0x4000, 1, 0x31)],
            0x1234,
        ),
        // LOCK XCHG [BX],AX
        (
            b"\xf0\x87\x07",
            &[
                (false, 0x4000, 1, 0x30),
                (false, 0x4001, 1, 0x31),
                (true, 0x4000, 1, 0x34),
                (true, 0x4001, 1, 0x12),
            ],
            &[(false, 0x4000, 2, 0x3130), (true, 0x4000, 2, 0x1234)],
            0x3130,
        ),
        // LOCK ADD ES:[BX],AX: 3130h + 1234h = 4364h
        (
            b"\xf0\x26\x01\x07",
            &[
                (false, 0x4000, 1, 0x30),
                (false, 0x4001, 1, 0x31),
                (true, 0x4000, 1, 0x64),
                (true, 0x4001, 1, 0x43),
            ],
            &[(false, 0x4000, 2, 0x3130), (true, 0x4000, 2, 0x4364)],
            0x1234,
        ),
    ];
    for model in LOCK_MODELS {
        let word_bus = matches!(model, CpuModel::I8086 | CpuModel::NecV30);
        for (code, bytes, words, ax) in cases {
            let transfers = if word_bus { words } else { bytes };
            let (vm, mut cpu) = setup_review(model, code, true);
            cpu.cpu.gpr[0] = 0x1234;
            cpu.cpu.gpr[3] = 0x4000;
            let bus = drive(&vm, &mut cpu);
            assert_eq!(bus, transfers, "{model:?} {code:02x?}");
            assert_eq!(cpu.cpu.rip, IP + code.len() as u64, "{model:?} {code:02x?}");
            assert_eq!(cpu.cpu.gpr[0] & 0xffff, ax, "{model:?} {code:02x?}");
            assert_eq!(cpu.retired_instructions(), 1, "{model:?} {code:02x?}");
            // After an earlier instruction of its block: that one retires
            // before the first device stop.
            let (vm, mut cpu) = setup_review(model, &[&b"\x40"[..], code].concat(), true);
            cpu.cpu.gpr[3] = 0x4000;
            let r = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
            assert!(
                matches!(
                    r.stop(),
                    StepStop::Exit(Exit::MmioRead { addr: 0x4000, .. })
                ),
                "{model:?} {code:02x?}: {:?}",
                r.stop()
            );
            assert_eq!((r.steps(), cpu.cpu.rip, cpu.cpu.gpr[0]), (1, IP + 1, 1));
            assert_eq!(cpu.retired_instructions(), 1);
            // Without LOCK the operand is read from the device.
            let (vm, mut cpu) = setup_review(model, &code[1..], true);
            cpu.cpu.gpr[3] = 0x4000;
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            let size = transfers[0].2;
            assert!(
                matches!(
                    r.stop(),
                    StepStop::Exit(Exit::MmioRead {
                        addr: 0x4000,
                        size: s,
                        ..
                    }) if *s == size
                ),
                "{model:?} {code:02x?}: {:?}",
                r.stop()
            );
        }
    }
}

// A locked word over two adjacent RAM regions keeps the containment miss, as
// every LOCK operand does on the other paths (spec.md "Data accesses across
// adjacent regions"); the I80286 real-mode run of the same program agrees.
// The exit names the word's first byte, which is mapped. Three placements: an
// even word over a boundary at an odd address; an odd word inside one 16-byte
// granule over a boundary at an even address; and an odd word at the usual
// page-aligned boundary, at 15 mod 16. The odd words had a typed refusal
// before xj-0w3.8 admitted them; an odd word now keeps the same containment
// miss as an even one. The unlocked word is performed one byte at a time.
#[test]
fn a_locked_word_over_two_regions_keeps_the_single_region_rule() {
    for (boundary, word) in [(0x4001usize, 0x4000u64), (0x4002, 0x4001), (0x4000, 0x3fff)] {
        let run = |model: CpuModel, code: &[u8]| {
            let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(InterpreterBackend));
            vm.set_cpu_model(model);
            vm.map(0, boundary, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(
                boundary as u64,
                0x20000 - boundary,
                Prot::RWX,
                RegionKind::Ram,
            )
            .unwrap();
            vm.write_bytes(IP, code).unwrap();
            vm.write_bytes(word, &[0xff]).unwrap();
            vm.write_bytes(word + 1, &[0x12]).unwrap();
            let mut cpu = vm.new_vcpu();
            if model == CpuModel::I80286 {
                cpu.cpu = x86jit_core::CpuState::legacy_real_entry([0; 6]);
            }
            cpu.cpu.rip = IP;
            cpu.cpu.gpr[4] = 0x8000;
            cpu.cpu.gpr[3] = word;
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            (
                matches!(
                    r.stop(),
                    StepStop::Exit(Exit::UnmappedMemory {
                        addr,
                        access: x86jit_core::AccessKind::Write,
                        ..
                    }) if *addr == word
                ),
                r.steps(),
                vm.mem.read(word, 1).unwrap() | vm.mem.read(word + 1, 1).unwrap() << 8,
                format!("{word:#x}: {:?}", r.stop()),
            )
        };
        let reference = run(CpuModel::I80286, b"\xf0\xff\x07");
        assert_eq!(
            (reference.0, reference.1, reference.2),
            (true, 0, 0x12ff),
            "{reference:?}"
        );
        for model in LOCK_MODELS {
            let locked = run(model, b"\xf0\xff\x07");
            assert_eq!(
                (locked.0, locked.1, locked.2),
                (true, 0, 0x12ff),
                "{model:?}: {locked:?}"
            );
            let unlocked = run(model, b"\xff\x07");
            assert_eq!(
                (unlocked.1, unlocked.2),
                (1, 0x1300),
                "{model:?}: {unlocked:?}"
            );
        }
    }
}

// A locked word that wraps its segment (DS:FFFFh, UM §2.3 p. 2-12) with one
// byte in a Trap region and the other in RAM has neither an atomic path nor
// the device path of a word wholly in Trap regions: a typed refusal before any
// transfer, the RAM byte unchanged (`locked_split`; spec.md §17.10).
#[test]
fn a_locked_word_that_wraps_half_into_a_device_is_refused() {
    // DS = 1000h, BX = FFFFh: the low byte at 1FFFFh, the high byte at 10000h.
    // The Trap region holds the high byte, then the low byte.
    for (trap, ram) in [(0x10000usize, 0x1ffffu64), (0x1fff0, 0x10000)] {
        for model in LOCK_MODELS {
            let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(InterpreterBackend));
            vm.set_cpu_model(model);
            vm.map(0, trap, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(trap as u64, 0x10, Prot::RWX, RegionKind::Trap)
                .unwrap();
            if trap + 0x10 < 0x20000 {
                vm.map(
                    (trap + 0x10) as u64,
                    0x20000 - trap - 0x10,
                    Prot::RWX,
                    RegionKind::Ram,
                )
                .unwrap();
            }
            vm.write_bytes(IP, b"\xf0\xff\x07").unwrap();
            vm.write_bytes(ram, &[0x5a]).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = IP;
            cpu.cpu.gpr[4] = 0x8000;
            cpu.cpu.ds = 0x1000;
            cpu.cpu.gpr[3] = 0xffff;
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(
                    r.stop(),
                    StepStop::Exit(Exit::UnknownInstruction {
                        addr: IP,
                        len: 3,
                        ..
                    })
                ),
                "{model:?} {trap:#x}: {:?}",
                r.stop()
            );
            assert_eq!((r.steps(), cpu.cpu.rip), (0, IP), "{model:?} {trap:#x}");
            assert_eq!(vm.mem.read(ram, 1).unwrap(), 0x5a, "{model:?} {trap:#x}");
        }
    }
}

// A locked word whose bytes are not adjacent on the bus (offset FFFFh wraps
// in its segment, UM §2.3 p. 2-12; physical FFFFFh wraps to 0, §2.8
// p. 2-68) has no single host address. Since xj-0w3.8 it takes the policy of
// an operand that crosses an aligned 16-byte granule (spec.md §8.2.3): the
// read, then the write, of its byte transfers, not atomic against another
// vCPU. On one vCPU it wraps exactly as the unlocked form does.
#[test]
fn a_locked_word_that_wraps_matches_the_unlocked_form() {
    for model in LOCK_MODELS {
        for (ds, bx, low, high) in [
            (0x1000u16, 0xffffu64, 0x1ffffu64, 0x10000u64),
            (0xffff, 0x000f, 0xfffff, 0),
            (0xf000, 0xffff, 0xfffff, 0xf0000),
        ] {
            for code in [&b"\xff\x07"[..], b"\x87\x07", b"\x81\x37\x01\x80"] {
                let run = |locked: bool| {
                    let mut bytes = if locked { vec![0xf0] } else { vec![] };
                    bytes.extend_from_slice(code);
                    let (vm, mut cpu) = setup(model, &bytes);
                    cpu.cpu.ds = ds;
                    cpu.cpu.gpr[3] = bx;
                    cpu.cpu.gpr[0] = 0x5566;
                    vm.mem.write(low, 0xff, 1).unwrap();
                    vm.mem.write(high, 0x12, 1).unwrap();
                    let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    (
                        format!("{:?}", r.stop()),
                        r.steps(),
                        cpu.cpu.rip,
                        cpu.cpu.gpr[0],
                        vm.mem.read(low, 1).unwrap(),
                        vm.mem.read(high, 1).unwrap(),
                    )
                };
                let what = format!("{model:?} {ds:x}:{bx:x} {code:02x?}");
                let mut locked = run(true);
                let unlocked = run(false);
                assert_eq!(unlocked.1, 1, "{what}: {unlocked:?}");
                assert_ne!((unlocked.4, unlocked.5), (0xff, 0x12), "{what}");
                // The LOCK prefix is the one byte the locked form adds.
                locked.2 -= 1;
                assert_eq!(locked, unlocked, "{what}");
            }
        }
    }
}

// A locked operand in ROM: the read half returns the byte and the store is
// dropped, as an unlocked store to ROM is (spec.md "Data accesses across
// adjacent regions"); flags follow the read value.
#[test]
fn a_locked_rom_operand_matches_the_unlocked_form() {
    for model in LOCK_MODELS {
        for code in [&b"\xff\x07"[..], b"\x87\x07", b"\x00\x27"] {
            let state = |locked: bool| {
                let mut bytes = if locked { vec![0xf0] } else { vec![] };
                bytes.extend_from_slice(code);
                let mut vm =
                    Vm::with_backend(VmConfig::flat(0x20000), Box::new(InterpreterBackend));
                vm.set_cpu_model(model);
                vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.map(0x4000, 0x1000, Prot::RWX, RegionKind::Rom).unwrap();
                vm.map(0x5000, 0x1b000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.write_bytes(IP, &bytes).unwrap();
                vm.write_bytes(0x4000, &[0xff, 0x7f]).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.cpu.rip = IP;
                cpu.cpu.gpr[3] = 0x4000;
                cpu.cpu.gpr[0] = 0x0102;
                step(&vm, &mut cpu, 1);
                (
                    cpu.cpu.gpr,
                    cpu.cpu.flags.to_flags16(),
                    vm.mem.read(0x4000, 2).unwrap(),
                )
            };
            let locked = state(true);
            assert_eq!(locked.2, 0x7fff, "{model:?} {code:02x?}");
            assert_eq!(locked, state(false), "{model:?} {code:02x?}");
        }
    }
}

// UM §2.6 p. 2-24: "Repeat, LOCK and segment override prefixes are
// considered 'part of' the instructions they prefix; no interrupt is
// recognized between execution of a prefix and an instruction"; p. 2-17:
// "Interrupts are not affected by the LOCK prefix". An IRQ held by STI's
// shadow is taken after the whole locked instruction, and a single-step trap
// follows the whole instruction.
#[test]
fn a_locked_instruction_is_one_instruction_to_interrupts_and_traps() {
    for model in LOCK_MODELS {
        // STI from IF = 0 holds a pending IRQ through the following
        // instruction (UM §2.6 p. 2-24; NEC UM §6 p. 6-4). Were the LOCK
        // prefix an instruction of its own, the IRQ would be taken after it,
        // with a frame naming the 26h byte and no increment.
        let (vm, mut cpu) = setup(model, b"\xfb\xf0\x26\xff\x07\xf4");
        vector(&vm, 0x20, 0x3000, 0);
        cpu.cpu.es = 0x100;
        cpu.cpu.gpr[3] = 0x10;
        cpu.inject_irq(0x20);
        let r = cpu.run_steps(&vm, 5, StepOptions::default()).unwrap();
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        assert_eq!(vm.mem.read(0x7ffa, 2).unwrap(), IP + 5, "{model:?}");
        assert_eq!(vm.mem.read(0x1010, 2).unwrap(), 1, "{model:?}");

        let (vm, mut cpu) = setup(model, b"\xf0\x26\xff\x07\xf4");
        vector(&vm, 1, 0x3000, 0);
        cpu.cpu.rflags_system |= 0x100;
        cpu.cpu.es = 0x100;
        cpu.cpu.gpr[3] = 0x10;
        let r = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{:?}",
            r.stop()
        );
        assert_eq!(vm.mem.read(0x7ffa, 2).unwrap(), IP + 4, "{model:?}");
        assert_eq!(vm.mem.read(0x1010, 2).unwrap(), 1, "{model:?}");
    }
}

// xj-50w.12 changes the twenty-bit entries' decode and lowering (LOCK), so
// their prefix revision word, and with it their identities, change. Of the
// system models, the I80286 identity changes with its own quirk fields
// (xj-50w.11) and the I80386 identity does not (values of b2ac76e4 on the
// left). xj-ir6.33 changes no identity: its device transfers are interpreter
// behaviour, and no lifting rule or catalogue field changes. xj-50w.19 then
// raises the NEC extension revision word to 2 (the segment-register store
// holds NMI as well as INT): the NEC "after" values below are that revision's
// (2a386bfd83ff8f1f and 2a386bfd83ff8f27 after xj-50w.12); its new
// `nmi_nesting` field keeps its canonical value everywhere and moves nothing.
// xj-50w.20's `data_bus_bits` departs from its canonical value only on the
// I8086 and NecV30: their "after" values below are that change's
// (8815844fbd9d4557 and cd4816e26331d184 before it).
#[test]
fn lock_admission_changes_only_the_twenty_bit_identities() {
    for (model, before, after) in [
        (
            CpuModel::I8088,
            0x22fb_677c_f449_d2fe,
            0x8815_844f_bd9d_4559,
        ),
        (
            CpuModel::I8086,
            0x22fb_677c_f449_d2f0,
            0xde9d_608e_bea8_4545,
        ),
        (
            CpuModel::NecV20,
            0x909a_c76c_b38b_71d8,
            0xcd48_16e2_6331_d1bc,
        ),
        (
            CpuModel::NecV30,
            0x909a_c76c_b38b_71e0,
            0x196e_97d4_9eba_3cd2,
        ),
        (
            CpuModel::I80286,
            0x3699_e02a_8ef1_3e63,
            // xj-50w.11 gives the 80286 its own quirk fields; xj-50w.12 does not touch it.
            // xj-oog adds LOADALL to its decode set (0x24d3_4f81_af8d_a20b before).
            0x7357_723a_3f16_2910,
        ),
        (
            CpuModel::I80386,
            0xe8d0_0c17_f97e_b71f,
            0xe8d0_0c17_f97e_b71f,
        ),
    ] {
        assert_eq!(model.profile().semantics_id, after, "{model:?}");
        assert_eq!(before == after, model == CpuModel::I80386, "{model:?}");
    }
}

// ---- Review witnesses (xj-50w.12/xj-50w.15 review), the scratch tests of
// the review adopted with their substance; their assertions name the fixed
// behaviour.

// One element of a REP string, then a budget yield: the pair is pending.
fn pending(model: CpuModel, code: &[u8]) -> (Vm, Vcpu) {
    let (vm, mut cpu) = setup(model, code);
    vector(&vm, 0x20, 0x3000, 0);
    cpu.cpu.flags.if_ = true;
    cpu.cpu.es = 0x1000;
    cpu.cpu.gpr[1] = 3;
    cpu.cpu.gpr[6] = 0x100;
    cpu.cpu.gpr[7] = 0x200;
    step(&vm, &mut cpu, 1);
    assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[1]), (IP, 2));
    (vm, cpu)
}
fn irq_frame(vm: &Vm, cpu: &mut Vcpu) -> u64 {
    cpu.inject_irq(0x20);
    let r = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert!(
        matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
        "{:?}",
        r.stop()
    );
    vm.mem.read(0x7ffa, 2).unwrap()
}

// xj-50w.15: the pair is keyed on the instruction too. A host write that
// replaces the interrupted string, with no register changed, starts new
// code at CS:IP; no element of it completed, so the frame names IP (UM §2.7
// p. 2-42). NecV20/V30 keep three prefixes, so four give a restart of IP+1.
#[test]
fn a_code_write_at_the_string_drops_the_restart_pair() {
    for model in LOCK_MODELS {
        let nec = matches!(model, CpuModel::NecV20 | CpuModel::NecV30);
        let code: &[u8] = if nec {
            b"\x26\x2e\x26\xf3\xa4\xf4"
        } else {
            b"\x26\xf3\xa4\xf4"
        };
        // INC AX; INC AX; HLT, and a plain REP MOVSB.
        for new in [
            &b"\x40\x40\xf4\xf4\xf4\xf4"[..],
            b"\xf3\xa4\xf4\xf4\xf4\xf4",
        ] {
            let (vm, mut cpu) = pending(model, code);
            vm.write_bytes(IP, &new[..code.len()]).unwrap();
            assert_eq!(irq_frame(&vm, &mut cpu), IP, "{model:?} {new:02x?}");
        }
    }
}

// xj-50w.15: a zero-budget call is no attempt boundary, so an edit undone
// around it (ABA) keeps the pair as an edit undone with no call between.
#[test]
fn a_zero_budget_call_between_an_edit_and_its_undo_keeps_the_pair() {
    for model in LOCK_MODELS {
        let code: &[u8] = if matches!(model, CpuModel::NecV20 | CpuModel::NecV30) {
            b"\x26\x2e\x26\xf3\xa4\xf4"
        } else {
            b"\x26\xf3\xa4\xf4"
        };
        let (vm, mut cpu) = pending(model, code);
        cpu.cpu.gpr[1] = 7;
        cpu.cpu.gpr[1] = 2;
        assert_eq!(irq_frame(&vm, &mut cpu), IP + 1, "{model:?} plain");

        let (vm, mut cpu) = pending(model, code);
        cpu.cpu.gpr[1] = 7;
        let r = cpu.run_steps(&vm, 0, StepOptions::default()).unwrap();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
        cpu.cpu.gpr[1] = 2;
        assert_eq!(irq_frame(&vm, &mut cpu), IP + 1, "{model:?} run_steps(0)");

        let (vm, mut cpu) = pending(model, code);
        cpu.cpu.gpr[1] = 7;
        assert!(matches!(cpu.run(&vm, Some(0)), Exit::BudgetExhausted));
        cpu.cpu.gpr[1] = 2;
        assert_eq!(irq_frame(&vm, &mut cpu), IP + 1, "{model:?} run(Some(0))");
    }
}

// xj-50w.15: the run entry compares the string's bytes with guest memory at
// CS:IP, not with the translation cache, which still holds the old block
// until the dispatcher applies the write. A write of the same bytes keeps the
// pair, and an edit of AX, which MOVSB does not read, is irrelevant; other
// bytes drop it.
#[test]
fn a_rewrite_of_the_string_is_compared_by_its_bytes() {
    for (bytes, frame) in [
        (&b"\x26\xf3\xa4\xf4"[..], IP + 1),
        (b"\x40\x40\xf4\xf4", IP),
    ] {
        let (vm, mut cpu) = pending(CpuModel::I8088, b"\x26\xf3\xa4\xf4");
        let key = x86jit_core::BlockKey::machine_direct(IP, cpu.cpu.mxcsr, false);
        assert!(vm.cache.get(key).is_some());
        vm.write_bytes(IP, bytes).unwrap();
        assert!(vm.cache.get(key).is_some(), "the write is not applied yet");
        cpu.cpu.gpr[0] = 0x55;
        assert_eq!(irq_frame(&vm, &mut cpu), frame, "{bytes:02x?}");
    }
}

// xj-50w.15 (re-review): the host rewrites the interrupted `26 F3 A4` as
// `26 F3 A5`, with the same prefixes, and a second vCPU executes the new
// string first, so the cache holds a REP string with the same restart IP
// again. The bytes differ, so the paused vCPU's pair is dropped and the frame
// names IP (UM §2.7 p. 2-42).
#[test]
fn a_string_rewritten_and_relifted_by_another_vcpu_drops_the_pair() {
    for model in LOCK_MODELS {
        let (code, new): (&[u8], &[u8]) = if matches!(model, CpuModel::NecV20 | CpuModel::NecV30) {
            (b"\x26\x2e\x26\xf3\xa4\xf4", b"\x26\x2e\x26\xf3\xa5\xf4")
        } else {
            (b"\x26\xf3\xa4\xf4", b"\x26\xf3\xa5\xf4")
        };
        let (vm, mut a) = pending(model, code);
        vm.write_bytes(IP, new).unwrap();
        let mut b = vm.new_vcpu();
        b.cpu.rip = IP;
        b.cpu.gpr[4] = 0x9000;
        b.cpu.es = 0x2000;
        b.cpu.gpr[1] = 5;
        b.cpu.gpr[6] = 0x400;
        b.cpu.gpr[7] = 0x400;
        let r = b.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(
            (r.steps(), b.cpu.rip, b.cpu.gpr[1]),
            (1, IP, 4),
            "{model:?}"
        );
        let key = x86jit_core::BlockKey::machine_direct(IP, a.cpu.mxcsr, false);
        assert!(
            vm.cache.get(key).is_some(),
            "{model:?}: the new string is cached"
        );
        assert_eq!(irq_frame(&vm, &mut a), IP, "{model:?}");
    }
}

// xj-50w.15 (re-review): with no code write and no register edit, mapping a
// device region elsewhere flushes the whole translation cache (`Vm::map`).
// The pair is kept: the frame names the kept prefix, as with no map.
#[test]
fn a_cache_flush_with_nothing_changed_keeps_the_pair() {
    for model in LOCK_MODELS {
        let code: &[u8] = if matches!(model, CpuModel::NecV20 | CpuModel::NecV30) {
            b"\x26\x2e\x26\xf3\xa4\xf4"
        } else {
            b"\x26\xf3\xa4\xf4"
        };
        let mut vm = Vm::with_backend(VmConfig::flat(0x120000), Box::new(InterpreterBackend));
        vm.set_cpu_model(model);
        vm.set_exception_delivery(ExceptionDelivery::InGuest);
        vm.map(0, 0x60000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x70000, 0xb0000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.write_bytes(IP, code).unwrap();
        vector(&vm, 0x20, 0x3000, 0);
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = IP;
        cpu.cpu.gpr[4] = 0x8000;
        cpu.cpu.flags.if_ = true;
        cpu.cpu.es = 0x1000;
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[6] = 0x100;
        cpu.cpu.gpr[7] = 0x200;
        step(&vm, &mut cpu, 1);
        assert_eq!((cpu.cpu.rip, cpu.cpu.gpr[1]), (IP, 2));
        let key = x86jit_core::BlockKey::machine_direct(IP, cpu.cpu.mxcsr, false);
        assert!(vm.cache.get(key).is_some());
        vm.map(0x60000, 0x1000, Prot::RWX, RegionKind::Trap)
            .unwrap();
        assert!(
            vm.cache.get(key).is_none(),
            "{model:?}: the map flushed the cache"
        );
        assert_eq!(irq_frame(&vm, &mut cpu), IP + 1, "{model:?}");
    }
}

// xj-50w.15 (re-review 2): the pair's bytes are the ones its block was lifted
// from, not a read of guest memory when the run returns. An external writer
// stores new string bytes and reports the write only later (`note_write`), so
// the next run still walks the cached old string and returns with its pair
// pending while memory already holds the new bytes. After the late notice the
// entry finds memory different from the old string's bytes and drops the
// pair: the frame names IP (UM §2.7 p. 2-42).
#[test]
fn a_write_reported_after_the_run_returns_drops_the_pair() {
    for model in LOCK_MODELS {
        let (code, new): (&[u8], &[u8]) = if matches!(model, CpuModel::NecV20 | CpuModel::NecV30) {
            (b"\x26\x2e\x26\xf3\xa4\xf4", b"\x26\x2e\x26\xf3\xa5\xf4")
        } else {
            (b"\x26\xf3\xa4\xf4", b"\x26\xf3\xa5\xf4")
        };
        let (vm, mut cpu) = setup(model, code);
        vector(&vm, 0x20, 0x3000, 0);
        cpu.cpu.flags.if_ = true;
        cpu.cpu.es = 0x1000;
        cpu.cpu.gpr[1] = 1;
        // One element completes the string: its block is cached, no pair.
        step(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.rip, IP + code.len() as u64 - 1, "{model:?}");
        cpu.cpu.rip = IP;
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[6] = 0x100;
        cpu.cpu.gpr[7] = 0x200;
        // SAFETY: the range is inside this VM's RAM; no vCPU is running.
        unsafe {
            std::ptr::copy_nonoverlapping(
                new.as_ptr(),
                vm.mem.host_base().add(IP as usize).cast_mut(),
                new.len(),
            );
        }
        let key = x86jit_core::BlockKey::machine_direct(IP, cpu.cpu.mxcsr, false);
        assert!(
            vm.cache.get(key).is_some(),
            "{model:?}: the write is not reported yet"
        );
        step(&vm, &mut cpu, 1);
        assert_eq!(
            (cpu.cpu.rip, cpu.cpu.gpr[1]),
            (IP, 2),
            "{model:?}: the old MOVSB ran"
        );
        vm.mem.note_write(IP, new.len());
        assert_eq!(irq_frame(&vm, &mut cpu), IP, "{model:?}");
    }
}

// xj-0w3.8 (replacing xj-50w.12's refusal): a locked word at an odd address
// inside one aligned 16-byte granule takes the atomic path and loses no
// update across two vCPUs, at an offset inside the containing 8-byte word
// (2001h) and one that crosses it (2007h). A word whose first byte is at 15
// mod 16 crosses the granule (200Fh), as does one that wraps its segment
// (DS:FFFFh) or the twenty-bit bus (FFFF:000Fh; UM §2.3 p. 2-12, §2.8
// p. 2-68). No host atomic covers it, so it is a plain read, then write
// (spec.md §8.2.3): one vCPU, which must get the single-processor result.
#[test]
fn a_locked_odd_word_inside_a_granule_is_atomic() {
    const N: u64 = 20_000;
    for model in [
        CpuModel::I8088,
        CpuModel::I8086,
        CpuModel::NecV20,
        CpuModel::NecV30,
        CpuModel::I80286,
    ] {
        // (DS, offset, the word's two physical bytes, vCPUs)
        let mut rows = vec![
            (0u16, 0x2000u16, [0x2000u64, 0x2001], 2),
            (0, 0x2001, [0x2001, 0x2002], 2),
            (0, 0x2007, [0x2007, 0x2008], 2),
            (0, 0x200f, [0x200f, 0x2010], 1),
        ];
        if model != CpuModel::I80286 {
            rows.push((0, 0xffff, [0xffff, 0], 1));
            rows.push((0xffff, 0x000f, [0xfffff, 0], 1));
        }
        for (ds, addr, bytes, vcpus) in rows {
            let [lo, hi] = addr.to_le_bytes();
            let code = [0xf0, 0xff, 0x06, lo, hi, 0xe2, 0xf9, 0xf4];
            let (vm, _) = setup(model, &code);
            for byte in bytes {
                vm.mem.write(byte, 0, 1).unwrap();
            }
            std::thread::scope(|scope| {
                for _ in 0..vcpus {
                    scope.spawn(|| {
                        let mut cpu = vm.new_vcpu();
                        if model == CpuModel::I80286 {
                            cpu.cpu = x86jit_core::CpuState::legacy_real_entry([0; 6]);
                        }
                        cpu.cpu.ds = ds;
                        cpu.cpu.rip = IP;
                        cpu.cpu.gpr[4] = 0x8000;
                        cpu.cpu.gpr[1] = N;
                        let r = cpu.run_steps(&vm, 3 * N, StepOptions::default()).unwrap();
                        assert!(
                            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                            "{model:?} {ds:#x}:{addr:#x}: {:?}",
                            r.stop()
                        );
                    });
                }
            });
            let got = vm.mem.read(bytes[0], 1).unwrap() | vm.mem.read(bytes[1], 1).unwrap() << 8;
            assert_eq!(got, vcpus * N, "{model:?} {ds:#x}:{addr:#x}");
        }
    }
}

// xj-50w.12, a systematic scan. Data, stack and string memory lie in a
// device region, so every data transfer of an unlocked form shows as an
// MMIO or port exit. A locked form whose unlocked form transfers data is
// refused, or it is a read-modify-write with the atomic path (it runs on
// RAM), whose device operand is then one read and one write of its bus
// transfers (spec.md §5.2, xj-ir6.33): its first transfer is the unlocked
// form's. Every locked form that runs must end as its unlocked form does. The one locked form refused although its unlocked form runs
// here without a transfer is INTO with OF clear: INTO with OF set reads the
// interrupt pointer and pushes as INT does (UM §2.7 pp. 2-46–47), the lift
// decides per form, and so the form has no atomic path. An unlocked form
// that stops with a device refusal of its own (an NEC BCD string over the
// device region) is a transfer, not a run.
#[derive(Debug, Clone, PartialEq)]
struct ScanOutcome {
    kind: String,
    steps: u64,
    gpr: [u64; 8],
    flags: u16,
}
fn scan_run(model: CpuModel, code: &[u8]) -> ScanOutcome {
    scan_run_in(model, code, RegionKind::Trap)
}
fn scan_run_in(model: CpuModel, code: &[u8], data: RegionKind) -> ScanOutcome {
    let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(InterpreterBackend));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::InGuest);
    vm.map(0, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x1000, 0x2000, Prot::RWX, data).unwrap();
    vm.map(0x3000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x500, code).unwrap();
    for v in 0..256u64 {
        vm.mem.write(v * 4, 0x0300_0000, 4).unwrap();
    }
    vm.write_bytes(0x3000, b"\xf4").unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.cs = 0;
    cpu.cpu.rip = 0x500;
    cpu.cpu.ds = 0x100;
    cpu.cpu.es = 0x100;
    cpu.cpu.ss = 0x200;
    cpu.cpu.gpr = [0; 16];
    cpu.cpu.gpr[0] = 0x0087;
    cpu.cpu.gpr[1] = 0x0002;
    cpu.cpu.gpr[3] = 0x10;
    cpu.cpu.gpr[4] = 0x100;
    cpu.cpu.gpr[5] = 0x40;
    cpu.cpu.gpr[6] = 0x20;
    cpu.cpu.gpr[7] = 0x30;
    let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    let kind = match r.stop() {
        StepStop::Exit(Exit::MmioRead { .. } | Exit::MmioWrite { .. } | Exit::PortIo { .. }) => {
            "transfer".to_string()
        }
        StepStop::Exit(Exit::UnknownInstruction { .. }) => "unknown".to_string(),
        StepStop::Exit(Exit::BudgetExhausted) => "ran".to_string(),
        StepStop::Exit(Exit::Hlt) => "hlt".to_string(),
        other => format!("{other:?}"),
    };
    let mut gpr = [0; 8];
    gpr.copy_from_slice(&cpu.cpu.gpr[..8]);
    ScanOutcome {
        kind,
        steps: r.steps(),
        gpr,
        flags: cpu.cpu.flags.to_flags16(),
    }
}
#[test]
fn every_locked_form_runs_as_its_unlocked_form_or_is_refused() {
    for model in [CpuModel::I8088, CpuModel::NecV20] {
        let nec = model == CpuModel::NecV20;
        let modrms: Vec<u8> = (0..8u8)
            .flat_map(|r| [0xc3 | (r << 3), 0x07 | (r << 3)])
            .collect();
        let mut cases: Vec<Vec<u8>> = Vec::new();
        for op in 0..=255u8 {
            if matches!(op, 0x26 | 0x2e | 0x36 | 0x3e | 0xf0 | 0xf2 | 0xf3)
                || (nec && matches!(op, 0x64 | 0x65))
            {
                continue;
            }
            if nec && op == 0x0f {
                for op2 in 0..=255u8 {
                    for m in &modrms {
                        cases.push(vec![0x0f, op2, *m, 1, 1, 1, 1]);
                    }
                }
                continue;
            }
            for m in &modrms {
                cases.push(vec![op, *m, 1, 1, 1, 1]);
            }
        }
        let mut wrong = Vec::new();
        for code in &cases {
            let unlocked = scan_run(model, code);
            let locked_code = [&[0xf0][..], code].concat();
            let locked = scan_run(model, &locked_code);
            let refused = locked.kind == "unknown";
            if unlocked.kind == "transfer" && !refused {
                let on_ram = scan_run_in(model, &locked_code, RegionKind::Ram);
                if on_ram.kind != "ran" || locked != unlocked {
                    wrong.push(format!(
                        "{code:02x?}: transfers locked without the atomic path: \
                         {locked:?}, unlocked {unlocked:?}, on RAM {on_ram:?}"
                    ));
                }
            } else if refused && matches!(unlocked.kind.as_str(), "ran" | "hlt") {
                if code[0] != 0xce {
                    wrong.push(format!(
                        "{code:02x?}: refused, runs unlocked without a transfer"
                    ));
                }
            } else if !refused && locked != unlocked {
                wrong.push(format!("{code:02x?}: {locked:?} != {unlocked:?}"));
            }
        }
        assert!(wrong.is_empty(), "{model:?}:\n{}", wrong.join("\n"));
    }
}

// xj-50w.15: the pair is not part of CpuState. Restoring a mid-string
// CpuState after the string has finished names the first prefix: the
// conservative frame, as for an abandoned request.
#[test]
fn a_restored_mid_string_state_names_the_first_prefix() {
    let (vm, mut cpu) = pending(CpuModel::I8088, b"\x26\xf3\xa4\xf4");
    let saved = cpu.cpu.clone();
    let r = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
    assert_eq!(r.steps(), 2, "{:?}", r.stop());
    let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert!(
        matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
        "{:?}",
        r.stop()
    );
    cpu.cpu = saved;
    assert_eq!(irq_frame(&vm, &mut cpu), IP);
}

// ---------------------------------------------------------------------------
// The NMI input (xj-50w.19; spec.md §17.10). 8086 Family UM §2.6
// pp. 2-24–28, §2.7 pp. 2-42 and 2-48, App. A AP-67 pp. A-27–29, App. B
// p. B-6. Each witness compares the whole CpuState, the frame words, the
// steps and the retired count.

use x86jit_core::CpuState;

const NMI_CS: u16 = 0x1000;
const NMI_IP: u16 = 0x400;

fn state(c: &CpuState) -> String {
    format!("{c:?}")
}
// The FLAGS image an 8086 frame stores: bits 15:12 set (UM p. 4-17).
fn image(c: &CpuState) -> u64 {
    (c.rflags() & 0xffff) | 0xf000
}
// The CPU after one interrupt entry: FLAGS, CS, IP pushed, IF and TF clear,
// CS:IP from the vector (UM §2.6 p. 2-26).
fn entered(c: &CpuState, cs: u16, ip: u16) -> CpuState {
    let mut e = c.clone();
    e.gpr[4] = c.gpr[4].wrapping_sub(6) & 0xffff;
    e.flags.if_ = false;
    e.rflags_system &= !0x100;
    e.cs = cs;
    e.rip = u64::from(ip);
    e
}
fn frame_at(vm: &Vm, sp: u64) -> [u64; 3] {
    [0, 2, 4].map(|d| vm.mem.read(sp + d, 2).unwrap())
}
fn to_hlt(vm: &Vm, cpu: &mut Vcpu, budget: u64) -> u64 {
    let r = cpu.run_steps(vm, budget, StepOptions::default()).unwrap();
    assert!(
        matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
        "{:?}",
        r.stop()
    );
    r.steps()
}
fn nmi_setup(model: CpuModel, code: &[u8]) -> (Vm, Vcpu) {
    let (vm, cpu) = setup(model, code);
    vector(&vm, 2, NMI_IP, NMI_CS);
    (vm, cpu)
}

#[test]
fn nmi_ignores_if_and_coalesces_its_edges() {
    // UM §2.6 p. 2-24 (IF does not mask NMI; no INTA cycles), p. 2-26
    // (frame, IF and TF cleared); App. B p. B-6 (an edge is latched and
    // serviced at the end of the current instruction).
    for model in MODELS {
        let (vm, mut cpu) = nmi_setup(model, b"\xfa\x90");
        cpu.cpu.flags.if_ = true;
        step(&vm, &mut cpu, 1); // CLI
        cpu.raise_nmi().unwrap();
        let line = cpu.nmi_line().unwrap();
        line.raise();
        assert!(line.is_raised());
        assert!(cpu.has_pending_nmi() && cpu.nmi_ready());
        let before = cpu.cpu.clone();
        assert_eq!(to_hlt(&vm, &mut cpu, 4), 1);
        assert_eq!(
            state(&cpu.cpu),
            state(&entered(&before, NMI_CS, NMI_IP + 1))
        );
        assert_eq!(frame_at(&vm, 0x7ffa), [IP + 1, 0, image(&before)]);
        assert_eq!(image(&before) & 0x200, 0, "delivered with IF = 0");
        assert_eq!(cpu.retired_instructions(), 2);
        assert!(!cpu.has_pending_nmi() && !line.is_raised(), "one delivery");
    }
}

#[test]
fn an_edge_after_acceptance_nests_on_the_8086() {
    // UM §2.6 p. 2-26 ("always may be interrupted by a request arriving on
    // NMI"); App. B p. B-6 ("Another high-going edge triggers another
    // response if it occurs after the start of the NMI procedure").
    for model in MODELS {
        let (vm, mut cpu) = nmi_setup(model, b"\x90");
        vm.write_bytes(0x10400, b"\x90\xf4").unwrap();
        cpu.raise_nmi().unwrap();
        step(&vm, &mut cpu, 1); // delivery, then the handler's NOP
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (NMI_CS, 0x401));
        cpu.raise_nmi().unwrap();
        assert!(cpu.has_pending_nmi() && cpu.nmi_ready());
        let before = cpu.cpu.clone();
        assert_eq!(to_hlt(&vm, &mut cpu, 4), 2);
        assert_eq!(
            state(&cpu.cpu),
            state(&entered(&before, NMI_CS, NMI_IP + 2))
        );
        assert_eq!(
            frame_at(&vm, 0x7ff4),
            [0x401, u64::from(NMI_CS), image(&before)]
        );
        assert_eq!(frame_at(&vm, 0x7ffa)[0], IP);
        assert_eq!(cpu.retired_instructions(), 3);
    }
}

#[test]
fn nmi_frame_precedes_the_step_frame_and_a_queued_irq_waits() {
    // UM §2.6 p. 2-28 ("either internal or external"), AP-67 Fig. 3E3a
    // p. A-29: an NMI met by the single-step trap is entered first, then the
    // step frame, so the step routine returns into the unstepped NMI
    // handler. UM p. 2-24: NMI before INTR; the IRQ waits behind IF = 0.
    // Both the post-instruction trap and the port-I/O trap (deferred until
    // the handoff returns) take this order.
    for model in MODELS {
        for (code, port) in [(&b"\xa0\x00\x40\x90"[..], false), (&b"\xee\x90"[..], true)] {
            for irq in [false, true] {
                let (vm, mut cpu) = setup_review(model, code, true);
                vector(&vm, 2, NMI_IP, NMI_CS);
                vector(&vm, 1, 0x500, 0);
                vector(&vm, 0x20, 0x600, 0);
                vm.write_bytes(0x500, b"\xf4\xcf").unwrap(); // step routine: HLT; IRET
                cpu.cpu.flags.if_ = irq;
                cpu.cpu.rflags_system = 0x100;
                let start = cpu.cpu.clone();
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                if port {
                    assert!(matches!(r.stop(), StepStop::Exit(Exit::PortIo { .. })));
                    assert_eq!(r.steps(), 1);
                } else {
                    assert!(matches!(r.stop(), StepStop::Exit(Exit::MmioRead { .. })));
                    assert_eq!(r.steps(), 0);
                    // Answered but unfinished: not yet an NMI boundary.
                    cpu.complete_mmio_read(0x5a);
                }
                cpu.raise_nmi().unwrap();
                if irq {
                    cpu.inject_irq(0x20);
                }
                assert_eq!(cpu.nmi_ready(), port);
                // The port handoff names the next instruction (xj-50w.22),
                // and the NMI frame saves it.
                let next = if port { IP + 1 } else { IP + 3 };
                if port {
                    assert_eq!(cpu.cpu.rip, next);
                }
                let mut before = start.clone();
                before.rip = next;
                if !port {
                    before.gpr[0] = 0x5a;
                }
                assert_eq!(to_hlt(&vm, &mut cpu, 2), if port { 1 } else { 2 });
                let inner = entered(&before, NMI_CS, NMI_IP);
                let mut expected = entered(&inner, 0, 0x501);
                // The finished device transfer's retry bookkeeping.
                expected.mmio_parts = cpu.cpu.mmio_parts;
                assert_eq!(state(&cpu.cpu), state(&expected));
                assert_eq!(frame_at(&vm, 0x7ffa), [next, 0, image(&before)]);
                assert_eq!(
                    frame_at(&vm, 0x7ff4),
                    [u64::from(NMI_IP), u64::from(NMI_CS), image(&inner)]
                );
                assert_eq!(cpu.has_pending_irq(), irq);
                assert!(!cpu.has_pending_nmi());
                step(&vm, &mut cpu, 1); // the step routine's IRET
                assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (NMI_CS, u64::from(NMI_IP)));
                // The NMI handler runs unstepped: no further step frame.
                assert_eq!(to_hlt(&vm, &mut cpu, 4), 1);
                assert_eq!(cpu.cpu.gpr[4], 0x7ffa);
                assert_eq!(cpu.has_pending_irq(), irq);
            }
        }
    }
}

#[test]
fn ap67_internal_interrupt_nmi_and_tf_take_no_step_frame() {
    // AP-67 p. A-28 and Fig. 3E3b p. A-29: an internal interrupt, an NMI and
    // TF together run both service routines without single stepping; with
    // TF = 0 the same two frames are taken. The segment load before the
    // interrupting instruction holds the latched NMI one instruction
    // (UM p. 2-24 footnote), so the NMI meets that instruction's trap.
    for model in MODELS {
        for (code, n, saved, of) in [
            (&b"\x8e\xd8\xcd\x21"[..], 0x21u8, IP + 4, false),
            (&b"\x8e\xd8\xcc"[..], 3, IP + 3, false),
            (&b"\x8e\xd8\xce"[..], 4, IP + 3, true),
            (&b"\x8e\xd8\xf7\xf3"[..], 0, IP + 4, false),
        ] {
            for tf in [true, false] {
                let (vm, mut cpu) = nmi_setup(model, code);
                vector(&vm, n, 0x300, 0x2000);
                vector(&vm, 1, 0x500, 0);
                cpu.cpu.flags.of = of;
                cpu.cpu.rflags_system = if tf { 0x100 } else { 0 };
                step(&vm, &mut cpu, 1); // MOV DS,AX: no step trap after it
                assert_eq!(cpu.cpu.rip, IP + 2);
                cpu.raise_nmi().unwrap();
                assert!(cpu.has_pending_nmi() && !cpu.nmi_ready());
                let before = cpu.cpu.clone();
                let r = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
                assert!(
                    matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                    "{n} {tf}: {:?}",
                    r.stop()
                );
                // The divide error is a trap of the DIV, which retires no
                // unit; INT n, INT 3 and INTO complete one.
                assert_eq!(r.steps(), if n == 0 { 1 } else { 2 }, "{n}");
                let internal = entered(&before, 0x2000, 0x300);
                let mut expected = entered(&internal, NMI_CS, NMI_IP + 1);
                expected.flags.of = of;
                assert_eq!(state(&cpu.cpu), state(&expected), "{n} {tf}");
                assert_eq!(frame_at(&vm, 0x7ffa), [saved, 0, image(&before)]);
                assert_eq!(
                    frame_at(&vm, 0x7ff4),
                    [0x300, 0x2000, image(&internal)],
                    "{n} {tf}"
                );
                assert_eq!(cpu.cpu.gpr[4], 0x7ff4, "no step frame");
            }
        }
    }
}

#[test]
fn an_edge_after_the_step_frame_is_stacked_on_top() {
    // UM §2.6 p. 2-26: NMI is not masked inside the step routine; the edge
    // raised between calls meets an ordinary boundary.
    for model in MODELS {
        let (vm, mut cpu) = nmi_setup(model, b"\x90");
        vector(&vm, 1, 0x500, 0);
        vm.write_bytes(0x500, b"\x90").unwrap();
        cpu.cpu.rflags_system = 0x100;
        let first = cpu.cpu.clone();
        step(&vm, &mut cpu, 1); // NOP, then the step frame
        let mut stepped = first.clone();
        stepped.rip = IP + 1;
        assert_eq!(state(&cpu.cpu), state(&entered(&stepped, 0, 0x500)));
        cpu.raise_nmi().unwrap();
        let before = cpu.cpu.clone();
        assert_eq!(to_hlt(&vm, &mut cpu, 4), 1);
        assert_eq!(
            state(&cpu.cpu),
            state(&entered(&before, NMI_CS, NMI_IP + 1))
        );
        assert_eq!(frame_at(&vm, 0x7ffa), [IP + 1, 0, image(&stepped)]);
        assert_eq!(frame_at(&vm, 0x7ff4), [0x500, 0, image(&before)]);
    }
}

#[test]
fn sti_and_iret_delays_do_not_hold_nmi() {
    // UM p. 2-24 and STI p. 2-48: the one-instruction delay concerns INTR;
    // NMI is not masked by IF (p. 2-24). CLI leaves NMI deliverable.
    for model in MODELS {
        // STI from IF = 0, then IRET that sets IF.
        for (code, frame) in [
            (&b"\xfb\x90"[..], None),
            (&b"\xcf\x90"[..], Some(0xf202u64)),
        ] {
            let (vm, mut cpu) = nmi_setup(model, code);
            if let Some(flags) = frame {
                vm.mem.write(0x8000, IP + 1, 2).unwrap();
                vm.mem.write(0x8002, 0, 2).unwrap();
                vm.mem.write(0x8004, flags, 2).unwrap();
            }
            step(&vm, &mut cpu, 1);
            assert!(cpu.cpu.flags.if_);
            // IF = 1, yet INTR is not ready: the delay is in force at this
            // boundary, and the NMI is accepted there.
            assert!(!cpu.irq_ready(), "the INTR delay is in force");
            cpu.raise_nmi().unwrap();
            assert!(cpu.nmi_ready());
            let before = cpu.cpu.clone();
            assert_eq!(to_hlt(&vm, &mut cpu, 4), 1);
            assert_eq!(
                state(&cpu.cpu),
                state(&entered(&before, NMI_CS, NMI_IP + 1))
            );
            assert_eq!(frame_at(&vm, before.gpr[4] - 6)[0], IP + 1);
        }
    }
}

#[test]
fn every_segment_load_holds_nmi_for_one_instruction() {
    // UM §2.6 p. 2-24 footnote: after MOV or POP to a segment register "no
    // interrupt is recognized until after the following instruction"
    // (spec.md §17.10: held for NMI as for INTR).
    for model in MODELS {
        for load in [
            &b"\x8e\xc0"[..],
            b"\x8e\xd0",
            b"\x8e\xd8",
            b"\x07",
            b"\x17",
            b"\x1f",
        ] {
            let mut code = load.to_vec();
            code.extend_from_slice(b"\x90\x90");
            let (vm, mut cpu) = nmi_setup(model, &code);
            step(&vm, &mut cpu, 1);
            cpu.raise_nmi().unwrap();
            assert!(cpu.has_pending_nmi() && !cpu.nmi_ready(), "{load:x?}");
            step(&vm, &mut cpu, 1); // the held instruction
            assert!(cpu.nmi_ready(), "{load:x?}");
            let before = cpu.cpu.clone();
            assert_eq!(to_hlt(&vm, &mut cpu, 4), 1);
            assert_eq!(
                state(&cpu.cpu),
                state(&entered(&before, NMI_CS, NMI_IP + 1))
            );
            let next = IP + load.len() as u64 + 1;
            assert_eq!(frame_at(&vm, before.gpr[4] - 6)[0], next, "{load:x?}");
        }
    }
}

#[test]
fn consecutive_segment_loads_hold_nmi_at_each_boundary() {
    // spec.md §17.10: each load holds the following boundary for NMI;
    // INTR keeps decision-16 §4 (only the first load holds it).
    for model in MODELS {
        let (vm, mut cpu) = nmi_setup(model, b"\x8e\xd8\x8e\xc0\x90\x90");
        cpu.cpu.flags.if_ = true;
        step(&vm, &mut cpu, 1); // MOV DS,AX
        cpu.raise_nmi().unwrap();
        assert!(!cpu.nmi_ready());
        step(&vm, &mut cpu, 1); // MOV ES,AX: held
        assert!(!cpu.nmi_ready(), "the second load holds NMI again");
        assert!(cpu.irq_ready(), "INTR is not held again (decision-16 §4)");
        step(&vm, &mut cpu, 1); // NOP: held
        assert!(cpu.nmi_ready());
        assert!(!cpu.irq_ready(), "NMI has priority (UM p. 2-24)");
        let before = cpu.cpu.clone();
        assert_eq!(to_hlt(&vm, &mut cpu, 4), 1);
        assert_eq!(frame_at(&vm, 0x7ffa), [IP + 5, 0, image(&before)]);
    }
}

#[test]
fn hlt_wakes_on_nmi_and_saves_the_next_ip() {
    // UM HLT p. 2-48 (NMI resumes); p. 2-26 (the next instruction's
    // address is saved).
    for model in MODELS {
        let (vm, mut cpu) = nmi_setup(model, b"\xf4");
        assert_eq!(to_hlt(&vm, &mut cpu, 4), 1);
        assert_eq!(cpu.cpu.rip, IP + 1);
        assert!(!cpu.has_pending_nmi());
        cpu.raise_nmi().unwrap();
        assert!(
            cpu.has_pending_nmi() && cpu.nmi_ready(),
            "the wake condition"
        );
        let before = cpu.cpu.clone();
        assert_eq!(to_hlt(&vm, &mut cpu, 4), 1);
        assert_eq!(
            state(&cpu.cpu),
            state(&entered(&before, NMI_CS, NMI_IP + 1))
        );
        assert_eq!(frame_at(&vm, 0x7ffa), [IP + 1, 0, image(&before)]);
    }
}

#[test]
fn nmi_inside_rep_follows_the_last_prefix_rule() {
    // UM §2.7 p. 2-42; App. B p. B-6 ("between whole moves"): after an
    // element the saved IP names the last prefix; a budget cut before the
    // first element keeps the whole instruction.
    for model in MODELS {
        for (code, budget, saved, count) in [
            (&b"\x26\xf3\xa4"[..], 1, IP + 1, 2),
            (&b"\xf3\x26\xa4"[..], 1, IP + 1, 2),
            (&b"\x90\x26\xf3\xa4"[..], 1, IP + 1, 3),
            (&b"\x90\x26\xf3\xa4"[..], 2, IP + 2, 2),
        ] {
            let (vm, mut cpu) = nmi_setup(model, code);
            cpu.cpu.es = 0x1000;
            cpu.cpu.gpr[1] = 3;
            cpu.cpu.gpr[6] = 0x100;
            cpu.cpu.gpr[7] = 0x200;
            step(&vm, &mut cpu, budget);
            cpu.raise_nmi().unwrap();
            let before = cpu.cpu.clone();
            assert_eq!(to_hlt(&vm, &mut cpu, 4), 1);
            assert_eq!(cpu.cpu.gpr[1], count);
            assert_eq!(frame_at(&vm, 0x7ffa), [saved, 0, image(&before)]);
            assert_eq!(
                state(&cpu.cpu),
                state(&entered(&before, NMI_CS, NMI_IP + 1))
            );
        }
    }
}

#[test]
fn nmi_delivery_costs_no_step_and_irq_ready_follows_the_handler() {
    // decision-16 §2 order extended (spec.md §17.10): delivery costs zero
    // steps; no stop reason is added. UM p. 2-24 (IRET delays INTR).
    for model in MODELS {
        for budget in [1, 2, 3] {
            let (vm, mut cpu) = nmi_setup(model, b"\x90\x90");
            vm.write_bytes(0x10400, b"\x90\x90\xcf").unwrap();
            cpu.raise_nmi().unwrap();
            let before = cpu.cpu.clone();
            step(&vm, &mut cpu, budget);
            let mut expected = if budget < 3 {
                entered(&before, NMI_CS, NMI_IP + budget as u16)
            } else {
                before.clone()
            };
            if budget == 3 {
                expected.rip = IP;
            }
            assert_eq!(state(&cpu.cpu), state(&expected), "{budget}");
            assert_eq!(cpu.retired_instructions(), budget);
        }
        let (vm, mut cpu) = nmi_setup(model, b"\x90\x90");
        vm.write_bytes(0x10400, b"\x90\x90\xcf").unwrap();
        cpu.raise_nmi().unwrap();
        assert!(matches!(
            cpu.step_instruction(&vm),
            x86jit_core::StepResult::Continue
        ));
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (NMI_CS, u64::from(NMI_IP) + 1));
        // The readiness stop: IF = 0 inside the handler; after its IRET the
        // INTR delay holds one more instruction.
        let (vm, mut cpu) = nmi_setup(model, b"\x90\x90");
        vm.write_bytes(0x10400, b"\x90\x90\xcf").unwrap();
        cpu.cpu.flags.if_ = true;
        cpu.raise_nmi().unwrap();
        assert!(!cpu.irq_ready());
        let r = cpu
            .run_steps(
                &vm,
                16,
                StepOptions::default().with_stop_when_irq_ready(true),
            )
            .unwrap();
        assert!(matches!(r.stop(), StepStop::IrqReady), "{:?}", r.stop());
        assert_eq!(r.steps(), 4);
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, IP + 1));
    }
}

#[test]
fn zero_budget_host_request_and_pending_input_keep_the_latch() {
    // decision-16 §2 extended (spec.md §17.10): N = 0 changes nothing; a host
    // request returns first with the latch kept; an IN completion holds it.
    for model in MODELS {
        let (vm, mut cpu) = nmi_setup(model, b"\x90\xec\x90");
        vector(&vm, 0x20, 0x600, 0);
        cpu.cpu.flags.if_ = true;
        cpu.raise_nmi().unwrap();
        cpu.inject_irq(0x20);
        cpu.exit_request().request_exit();
        let before = cpu.cpu.clone();
        let r = cpu.run_steps(&vm, 0, StepOptions::default()).unwrap();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
        assert_eq!(r.steps(), 0);
        let r = cpu.run_steps(&vm, 5, StepOptions::default()).unwrap();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Interrupted)));
        assert_eq!(r.steps(), 0);
        assert_eq!(state(&cpu.cpu), state(&before));
        assert!(cpu.has_pending_nmi() && cpu.has_pending_irq());
        assert_eq!(to_hlt(&vm, &mut cpu, 4), 1);
        assert_eq!(
            state(&cpu.cpu),
            state(&entered(&before, NMI_CS, NMI_IP + 1))
        );
        assert!(cpu.has_pending_irq(), "INTR waits behind IF = 0");
        assert_eq!(cpu.retired_instructions(), 1);
        // A pending IN: zero budget, then the refusal, then the completion.
        let (vm, mut cpu) = nmi_setup(model, b"\xec\x90");
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::PortIo { .. })));
        // The handoff names the next instruction (xj-50w.22), and the NMI
        // frame saves it.
        assert_eq!(cpu.cpu.rip, IP + 1);
        cpu.raise_nmi().unwrap();
        assert!(!cpu.nmi_ready());
        let held = cpu.cpu.clone();
        let r = cpu.run_steps(&vm, 0, StepOptions::default()).unwrap();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
        assert!(cpu.run_steps(&vm, 1, StepOptions::default()).is_err());
        assert_eq!(state(&cpu.cpu), state(&held));
        cpu.complete_port_in(0x5a);
        assert!(cpu.nmi_ready());
        let before = cpu.cpu.clone();
        assert_eq!(to_hlt(&vm, &mut cpu, 4), 1);
        assert_eq!(
            state(&cpu.cpu),
            state(&entered(&before, NMI_CS, NMI_IP + 1))
        );
        assert_eq!(frame_at(&vm, 0x7ffa)[0], IP + 1);
    }
}

#[test]
fn execution_reset_clears_the_nmi_latch() {
    // spec.md §17.10: reset_execution_state discards the latch.
    for model in MODELS {
        let (vm, mut cpu) = nmi_setup(model, b"\x90\xf4");
        cpu.raise_nmi().unwrap();
        cpu.reset_execution_state(ResetOptions::default());
        assert!(!cpu.has_pending_nmi());
        assert!(!cpu.nmi_line().unwrap().is_raised());
        assert_eq!(to_hlt(&vm, &mut cpu, 4), 2);
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, IP + 2));
    }
}

// xj-50w.22: IN and OUT end their twenty-bit block. The exit names the next
// instruction, and the rest of the block runs after the port completes.
#[test]
fn in_and_out_end_their_block() {
    for model in MODELS {
        port_block::check(|code| setup(model, code), IP, true);
    }
}
