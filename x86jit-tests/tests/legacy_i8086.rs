//! 8086/8088 manual witnesses (decision-23 C2, spec.md §17.10).
//! Intel 9800722-03 §2.3 p. 2-12, §2.6 pp. 2-24–28,
//! §2.7 pp. 2-33–48, §4.2 Tables 4-12/4-13 pp. 4-22–36.
use x86jit_core::{
    CpuGeneration, CpuModel, ExceptionDelivery, Exit, FpuModel, InterpreterBackend, LegacyCpuModel,
    ModelSystem, Prot, RegionKind, ResetOptions, StepOptions, StepStop, UnsupportedSystemOperation,
    Vcpu, Vm, VmConfig,
};
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
        b"\xf0\x90",
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
            for (addr, value) in [(0x4000, 0x34), (0x4001, 0x12)] {
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(
                    matches!(r.stop(), StepStop::Exit(Exit::MmioRead { addr: a, size: 1, .. }) if *a == addr),
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
            for (addr, value) in [(0x4000, 0x34), (0x4001, 0x12)] {
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert!(
                    matches!(r.stop(), StepStop::Exit(Exit::MmioWrite { addr: a, size: 1, value: v, .. }) if *a == addr && *v == value),
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
    for model in MODELS {
        let (vm, mut cpu) = setup_review(model, b"\xab", true);
        cpu.cpu.gpr[7] = 0x4000;
        cpu.cpu.gpr[0] = 0x1234;
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(matches!(
            r.stop(),
            StepStop::Exit(Exit::MmioWrite {
                addr: 0x4000,
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
                    addr: 0x4001,
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
        for (addr, value) in [
            (0x4000, 0),
            (0x4001, 0),
            (0x4000, 0),
            (0x4001, 0),
            (0x4002, 0x80),
            (0x4003, 0x3f),
        ] {
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::MmioRead { addr: a, size: 1, .. }) if *a == addr),
                "{:?}",
                r.stop()
            );
            assert_eq!(cpu.retired_instructions(), 0);
            cpu.complete_mmio_read(value);
        }
        step(&vm, &mut cpu, 1);
        for addr in [0x4008, 0x4009] {
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::MmioRead { addr: a, size: 1, .. }) if *a == addr),
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
        for addr in [0x4000, 0x4001] {
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::MmioRead { addr: a, size: 1, .. }) if *a == addr)
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
