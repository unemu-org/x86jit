//! Model identity and decode witnesses (decision-23 §§2–3, spec.md §17.10).
//! SDM Vol 3B §25.13.1/Table 25-1 pp.25-4–5, §§25.15–25.16.1 pp.25-5–7;
//! 80386 PRM §2.3.4 Fig.2-8 p.2-11, §14.6 Table14-1 p.14-5;
//! 80286 PRM §2.3.4 Fig.2-10 p.2-15, App.B #UD p.B-12 (scan checked).
use x86jit_core::{
    CpuMode, CpuModel, CpuState, ExceptionDelivery, Exit, InterpreterBackend, Prot, RegionKind,
    StepOptions, StepStop, Vcpu, Vm, VmConfig,
};
const IDENTITY: u64 = (1 << 18) | (1 << 21);
const IP: u64 = 0x234;
fn setup(model: Option<CpuModel>, code: &[u8]) -> (Vm, Vcpu) {
    setup_delivery(model, code, false)
}
fn setup_delivery(model: Option<CpuModel>, code: &[u8], guest: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(InterpreterBackend));
    if let Some(model) = model {
        vm.set_cpu_model(model);
    } else {
        vm.set_cpu_mode(CpuMode::Real16);
    }
    vm.set_exception_delivery(if guest {
        ExceptionDelivery::InGuest
    } else {
        ExceptionDelivery::Outward
    });
    vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000 + IP, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([0x100, 0, 0, 0, 0, 0]);
    cpu.cpu.rip = IP;
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}
fn step(vm: &Vm, cpu: &mut Vcpu, units: u64) {
    let result = cpu.run_steps(vm, units, StepOptions::default()).unwrap();
    assert_eq!(result.steps(), units, "{:?}", result.stop());
    assert!(matches!(
        result.stop(),
        StepStop::Exit(Exit::BudgetExhausted)
    ));
}
#[test]
fn popf_pushf_identity_toggle_probes() {
    for model in [Some(CpuModel::I80286), Some(CpuModel::I80386), None] {
        for attempted in [0u64, 1 << 18, 1 << 21, IDENTITY] {
            let wide = model == Some(CpuModel::I80386);
            let code: &[u8] = if wide {
                b"\x66\x9d\x66\x9c"
            } else {
                b"\x9d\x9c"
            };
            let (vm, mut cpu) = setup(model, code);
            // POPF/POPFD: PRM386 §17 p.17-130; 286 App.B pp.B-85–86.
            // Start with dirty embedder storage as well as attempted stack bits.
            cpu.cpu.rflags_system = IDENTITY;
            vm.mem
                .write(0x8000, attempted | 2, if wide { 4 } else { 2 })
                .unwrap();
            step(&vm, &mut cpu, 2);
            assert_eq!(cpu.cpu.rflags_system & IDENTITY, 0, "{model:?}");
            assert_eq!(
                vm.mem.read(0x8000, if wide { 4 } else { 2 }).unwrap() & IDENTITY,
                0
            );
            assert_eq!(vm.mem.read(0x8000, 2).unwrap(), 2);
        }
    }
}
#[test]
fn iret_identity_toggle_probes() {
    // IRET: PRM386 §17 pp.17-79–83; 286 App.B pp.B-51–52.
    for model in [Some(CpuModel::I80286), Some(CpuModel::I80386), None] {
        for attempted in [0u64, 1 << 18, 1 << 21, IDENTITY] {
            let wide = model == Some(CpuModel::I80386);
            let (vm, mut cpu) = setup(model, if wide { b"\x66\xcf" } else { b"\xcf" });
            cpu.cpu.rflags_system = IDENTITY;
            let size = if wide { 4 } else { 2 };
            for (i, value) in [0x456, 0x100, attempted | 2].into_iter().enumerate() {
                vm.mem
                    .write(0x8000 + i as u64 * u64::from(size), value, size)
                    .unwrap();
            }
            step(&vm, &mut cpu, 1);
            assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0x100, 0x456));
            assert_eq!(cpu.cpu.rflags_system & IDENTITY, 0, "{model:?}");
        }
    }
}
const LATER: &[(&str, &[u8])] = &[
    ("CPUID", b"\x0f\xa2"),
    ("RDTSC", b"\x0f\x31"),
    ("RDMSR", b"\x0f\x32"),
    ("WRMSR", b"\x0f\x30"),
    ("INVD", b"\x0f\x08"),
    ("WBINVD", b"\x0f\x09"),
    ("INVLPG", b"\x0f\x01\x38"),
    ("BSWAP", b"\x0f\xc8"),
    ("CMPXCHG byte", b"\x0f\xb0\xc1"),
    ("CMPXCHG word", b"\x0f\xb1\xc1"),
    ("XADD byte", b"\x0f\xc0\xc1"),
    ("XADD word", b"\x0f\xc1\xc1"),
];
fn ud(model: CpuModel, name: &str, code: &[u8], guest: bool) {
    let (vm, mut cpu) = setup_delivery(Some(model), code, guest);
    vm.write_bytes(6 * 4, b"\x00\x40\x00\x00").unwrap();
    vm.write_bytes(0x4000, b"\xf4").unwrap();
    let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    if guest {
        assert!(
            matches!(result.stop(), StepStop::Exit(Exit::Hlt)),
            "{model:?} {name}: {:?}",
            result.stop()
        );
        assert_eq!(
            (
                vm.mem.read(0x7ffa, 2).unwrap(),
                vm.mem.read(0x7ffc, 2).unwrap()
            ),
            (IP, 0x100)
        );
    } else {
        let StepStop::Exit(exit @ Exit::Exception { vector: 6, .. }) = result.stop() else {
            panic!("{model:?} {name}: {:?}", result.stop());
        };
        assert_eq!(result.steps(), 0);
        assert_eq!(exit.fault_metadata().unwrap().saved_rip(), IP);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (0x100, IP, 0x8000)
        );
    }
}
#[test]
fn later_instructions_ud_at_prefixed_cs_ip_on_both_models() {
    // SDM Vol2A §3.3: BSWAP p.3-111, CMPXCHG p.3-194, CPUID p.3-203,
    // INVD p.3-483, INVLPG p.3-485; Vol2B §4.3 RDMSR p.4-544,
    // RDTSC p.4-559; Vol2D §6 WBINVD p.6-2, WRMSR p.6-8, XADD p.6-27.
    // Introductions: Vol3B §25.13.1/Table25-1 pp.25-4–5.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for &(name, code) in LATER {
            for guest in [false, true] {
                let mut prefixed = vec![0x2e];
                prefixed.extend_from_slice(code);
                ud(model, name, &prefixed, guest);
            }
        }
    }
}
#[test]
fn absent_ud1_is_ud_instead_of_a_refusal() {
    // 386 PRM App.A two-byte map p.A-7 leaves 0F B9 undefined;
    // §14.6 Table14-1 p.14-5 requires exception 6 at the instruction.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        ud(model, "UD1", b"\x0f\xb9\xc0", false);
    }
}
#[test]
fn reserved_386_map_slots_raise_ud() {
    // 80386 PRM Appendix A two-byte map pp.A-6–A-7: 0F 05/07/A6/A7
    // are reserved. Table14-1 p.14-5 specifies #UD at the faulting instruction.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for code in [
            &b"\x0f\x05"[..],
            &b"\x0f\x07"[..],
            &b"\x0f\xa6\xc1"[..],
            &b"\x0f\xa7\xc1"[..],
            &b"\x66\x0f\xa6\xc1"[..],
            &b"\x66\x0f\xa7\xc1"[..],
        ] {
            ud(model, "reserved slot", code, false);
        }
    }
}
#[test]
fn documented_aliases_keep_their_results() {
    // SDM Vol3B §25.15 p.25-6: 82h ADD, group2 /6 SAL and group3 /1 TEST.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for (code, expected) in [
            (&b"\x82\xc0\x01"[..], 2),
            (&b"\xd0\xf0"[..], 2),
            (&b"\xf6\xc8\x01"[..], 1),
        ] {
            let (vm, mut cpu) = setup(Some(model), code);
            cpu.cpu.gpr[0] = 1;
            step(&vm, &mut cpu, 1);
            assert_eq!(cpu.cpu.gpr[0], expected);
        }
    }
}
#[test]
fn application_cpuid_and_alias_superset_remain_available() {
    for mode in [CpuMode::Long64, CpuMode::Compat32, CpuMode::Real16] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(InterpreterBackend));
        vm.set_cpu_mode(mode);
        vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\x0f\xa2\xf4").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt), "{mode:?}");
        assert_ne!(cpu.cpu.gpr[0], 0);
    }
}

fn protected(model: CpuModel, code: &[u8]) -> (Vm, Vcpu) {
    // 286 PRM §6.3 Fig.6-3 p.6-5; 386 PRM §6.3.1 Fig.6-3 p.6-3:
    // present ring-zero code/data descriptors; CS.D/SS.B select 32-bit on 386.
    let (vm, mut cpu) = setup(Some(model), code);
    let high = if model == CpuModel::I80386 { 0x40 } else { 0 };
    for (selector, base, rights) in [(8u64, 0x1000u32, 0x9bu8), (16, 0, 0x93)] {
        vm.write_bytes(
            0x6000 + selector,
            &[
                0xff,
                0xff,
                base as u8,
                (base >> 8) as u8,
                (base >> 16) as u8,
                rights,
                high,
                0,
            ],
        )
        .unwrap();
    }
    cpu.cpu.cs = 8;
    cpu.cpu.ss = 16;
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.gdtr.base = 0x6000;
    cpu.cpu.legacy.gdtr.limit = 23;
    cpu.cpu.legacy.segments[0].attributes |= u16::from(high) << 8;
    cpu.cpu.legacy.segments[3].attributes |= u16::from(high) << 8;
    (vm, cpu)
}

#[test]
fn protected_default_width_flag_loads_clear_identity() {
    // POPF/IRET keep the model's absent flags in protected operation too:
    // SDM Vol3B §§25.16–25.16.1 pp.25-6–7; 386 PRM §2.3.4 p.2-11.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let size = if model == CpuModel::I80386 { 4 } else { 2 };
        let (vm, mut cpu) = protected(model, b"\x9d\x9c");
        cpu.cpu.rflags_system = IDENTITY;
        vm.mem.write(0x8000, IDENTITY | 2, size).unwrap();
        step(&vm, &mut cpu, 2);
        assert_eq!(cpu.cpu.rflags_system & IDENTITY, 0);
        assert_eq!(vm.mem.read(0x8000, size).unwrap(), 2);
        let (vm, mut cpu) = protected(model, b"\xcf");
        cpu.cpu.rflags_system = IDENTITY;
        for (i, value) in [0x456, 8, IDENTITY | 2].into_iter().enumerate() {
            vm.mem
                .write(0x8000 + i as u64 * u64::from(size), value, size)
                .unwrap();
        }
        step(&vm, &mut cpu, 1);
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (8, 0x456));
        assert_eq!(cpu.cpu.rflags_system & IDENTITY, 0);
    }
}

#[test]
fn protected_default_width_does_not_grant_later_instructions() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for &(name, code) in LATER {
            let (vm, mut cpu) = protected(model, code);
            let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            let StepStop::Exit(exit @ Exit::Exception { vector: 6, .. }) = result.stop() else {
                panic!("{model:?} {name}: {:?}", result.stop());
            };
            assert_eq!(exit.fault_metadata().unwrap().saved_rip(), IP);
            assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (8, IP));
        }
    }
}
