//! 80386 PRM §10.1 pp. 10-1–10-2 and §10.2.3 p. 10-3;
//! SDM Vol 3A §12.1.4 p. 12-5 (the pinned edition's reset-fetch section).

use x86jit_core::{CpuState, FpuModel};

#[test]
fn reset_register_image() {
    for fpu in [FpuModel::Present, FpuModel::None] {
        let cpu = CpuState::i386_reset(fpu);
        assert_eq!(cpu.rflags(), 2);
        assert_eq!(cpu.rip, 0xfff0);
        assert_eq!(
            [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs],
            [0xf000, 0, 0, 0, 0, 0]
        );
        assert_eq!((cpu.legacy.idtr.base, cpu.legacy.idtr.limit), (0, 0x3ff));
        assert_eq!(
            cpu.legacy.cr0,
            if fpu == FpuModel::Present { 0x10 } else { 0 }
        );
        assert_eq!(cpu.fpu_model, fpu);
        assert_eq!(cpu.legacy.cpl, 0);
        assert_eq!(cpu.legacy.reset_fetch, 1);
        // PRM Fig. 10-1: DH=03h; ASSUMED stepping 03h and zero undefined GPRs.
        assert_eq!(
            cpu.gpr,
            [0, 0, 0x303, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        for (i, cache) in cpu.legacy.segments.iter().enumerate() {
            assert_eq!(cache.base, if i == 0 { 0xffff_0000 } else { 0 });
            // DERIVED ordinary real limits; ASSUMED engine attribute labels.
            assert_eq!(cache.limit, 0xffff);
            assert_eq!(cache.attributes, if i == 0 { 0x9b } else { 0x93 });
        }
        // ASSUMED deterministic defaults for state left undefined by PRM §10.1.
        assert_eq!((cpu.legacy.cr2, cpu.legacy.cr3), (0, 0));
        assert_eq!(cpu.legacy.gdtr, Default::default());
        assert_eq!(cpu.legacy.ldtr, Default::default());
        assert_eq!(cpu.legacy.tr, Default::default());
        assert_eq!(cpu.legacy.a20_enabled, 1);
    }
}

#[cfg(unix)]
mod fetch {
    use super::*;
    use x86jit_core::{
        AccessKind, Backend, CachedBlock, CpuModel, Exit, HostRam, IrBlock, Knowledge,
        MemConsistency, Prot, RegionKind, StepOptions, StepStop, Vcpu, Vm, VmConfig,
    };

    const HIGH: u64 = 0xffff_0000;
    // DERIVED from physical bit-20 masking (SDM Vol 3A §11.7.13.4 p. 11-33).
    // ASSUMED board policy: the same gate applies at reset (decision-22).
    const GATED: u64 = 0xffef_0000;

    struct NoBackend;
    impl Backend for NoBackend {
        fn materialize(
            &self,
            _: &IrBlock,
            _: MemConsistency,
            _: Option<(u64, u64)>,
            _: u64,
        ) -> CachedBlock {
            panic!("Machine reset reached application backend")
        }
    }

    fn machine() -> (Vm, Vcpu) {
        let len = 0x1_0000_0000;
        // SAFETY: reserve anonymous virtual space, with physical backing allocated
        // only on demand. HostRam owns it and unmaps it after the last VM access.
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
            HostRam {
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
            Vm::with_backend_host_ram(VmConfig::reserved(len as u64), Box::new(NoBackend), ram)
                .unwrap();
        vm.set_cpu_model(CpuModel::I80386);
        // Only 192 KiB of mapped guest RAM: IVT/stack, low firmware and high copy.
        for addr in [0, 0xf0000, HIGH] {
            vm.map(addr, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        }
        let mut cpu = vm.new_vcpu();
        cpu.cpu = CpuState::i386_reset(FpuModel::Present);
        (vm, cpu)
    }

    fn steps(vm: &Vm, cpu: &mut Vcpu, n: u64) {
        let run = cpu.run_steps(vm, n, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), n, "{:?}", run.stop());
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
            "{:?}",
            run.stop()
        );
    }

    #[test]
    fn near_then_far_jump_uses_high_then_low_firmware_with_either_a20_line() {
        let (mut vm, mut cpu) = machine();
        vm.map(GATED, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        for (base, value) in [(HIGH, 0x1234u16), (GATED, 0x5678)] {
            vm.write_bytes(base + 0xfff0, b"\xe9\x0d\xff").unwrap(); // near JMP FF00h
            let [lo, hi] = value.to_le_bytes();
            vm.write_bytes(base + 0xff00, &[0xbb, lo, hi, 0xea, 0x00, 0x02, 0x00, 0xf0])
                .unwrap();
        }
        vm.write_bytes(0xf0200, b"\xb8\xbc\x9a\xf4").unwrap();
        for a20 in [true, false, true] {
            cpu.cpu = CpuState::i386_reset(FpuModel::Present);
            cpu.set_a20_enabled(a20);
            let cache = cpu.cpu.legacy.segments[0];
            steps(&vm, &mut cpu, 1);
            assert_eq!(cpu.cpu.rip, 0xff00);
            assert_eq!(cpu.cpu.legacy.segments[0], cache);
            assert_eq!(cpu.cpu.legacy.reset_fetch, 1);
            steps(&vm, &mut cpu, 1);
            assert_eq!(cpu.cpu.gpr[3], if a20 { 0x1234 } else { 0x5678 });
            assert_eq!(cpu.cpu.legacy.segments[0], cache);
            steps(&vm, &mut cpu, 1);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.reset_fetch),
                (0xf000, 0x200, 0)
            );
            assert_eq!(cpu.cpu.legacy.segments[0].base, 0xf0000);
            assert_eq!(cpu.cpu.legacy.segments[0].limit, 0xffff);
            steps(&vm, &mut cpu, 1);
            assert_eq!(cpu.cpu.gpr[0], 0x9abc);
        }
    }

    #[test]
    fn a20_off_requires_the_physical_alias_mapping() {
        let (mut vm, mut cpu) = machine();
        vm.write_bytes(HIGH + 0xfff0, b"\x43").unwrap(); // INC BX
        cpu.set_a20_enabled(false);
        let before = cpu.cpu.clone();
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        let StepStop::Exit(
            exit @ Exit::UnmappedMemory {
                addr: 0xffef_fff0,
                access: AccessKind::Execute,
                ..
            },
        ) = run.stop()
        else {
            panic!("{:?}", run.stop())
        };
        let metadata = exit.fault_metadata().unwrap();
        assert_eq!(metadata.physical_address(), Knowledge::Known(0xffef_fff0));
        let Knowledge::Known(linear) = metadata.linear_address() else {
            panic!("{metadata:?}")
        };
        assert_eq!(linear.value(), 0xffff_fff0);
        assert_eq!(cpu.cpu.rip, before.rip);
        assert_eq!(cpu.cpu.gpr, before.gpr);
        assert_eq!(cpu.cpu.legacy, before.legacy);
        vm.map(GATED, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(GATED + 0xfff0, b"\x43").unwrap();
        steps(&vm, &mut cpu, 1);
        assert_eq!(cpu.cpu.gpr[3], 1);
    }

    #[test]
    fn each_cs_reload_ends_reset_fetch_and_retains_big_real_limit() {
        // SDM Vol 3A §12.1.4 p. 12-5: loading CS ends the exceptional base.
        // DERIVED: exercise every real CS-reload path, including RETF and IRET.
        // ASSUMED existing engine big-real contract: a reload retains the limit.
        for code in [
            &b"\xea\x00\x02\x00\xf0"[..],
            b"\x9a\x00\x02\x00\xf0",
            b"\xcb",
            b"\xcf",
            b"\xcd\x20",
        ] {
            for limit in [0xffff, 0x1ffff] {
                let (vm, mut cpu) = machine();
                vm.write_bytes(HIGH + 0xfff0, code).unwrap();
                vm.write_bytes(0xf0200, b"\x43").unwrap();
                cpu.cpu.legacy.segments[0].limit = limit;
                cpu.cpu.gpr[4] = 0x8000;
                // RETF/IRET frame, and INT 20h's IVT target: F000:0200.
                vm.write_bytes(0x8000, b"\x00\x02\x00\xf0\x02\x00").unwrap();
                vm.write_bytes(0x80, b"\x00\x02\x00\xf0").unwrap();
                steps(&vm, &mut cpu, 1);
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.reset_fetch),
                    (0xf000, 0x200, 0),
                    "{code:02x?}"
                );
                assert_eq!(cpu.cpu.legacy.segments[0].base, 0xf0000);
                assert_eq!(cpu.cpu.legacy.segments[0].limit, limit);
                steps(&vm, &mut cpu, 1);
                assert_eq!(cpu.cpu.gpr[3], 1);
            }
        }
    }
}
