//! XSAVE witnesses: SDM Vol 1 §§13.4–13.10; Vol 2D §6.3 XSAVE/XRSTOR.
use iced_x86::code_asm::*;
use x86jit_core::{
    Backend, CpuMode, Exit, GuestCpuFeatures, InterpreterBackend, Prot, Reg, RegionCaps,
    RegionKind, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x10000;
const AREA: u64 = 0x18000;

fn backend(tier: usize) -> Box<dyn Backend> {
    match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        _ => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 16,
            max_icount: 256,
        })),
    }
}
fn vm(tier: usize, mode: CpuMode, build: impl FnOnce(&mut CodeAssembler)) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x20000), backend(tier));
    vm.set_cpu_mode(mode);
    vm.set_guest_cpu_features(GuestCpuFeatures::v4());
    vm.map(CODE, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut a = CodeAssembler::new(if mode == CpuMode::Long64 { 64 } else { 32 }).unwrap();
    region_entry(&mut a);
    build(&mut a);
    a.hlt().unwrap();
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
    vm
}
fn region_entry(a: &mut CodeAssembler) {
    let mut top = a.create_label();
    let mut body = a.create_label();
    a.set_label(&mut top).unwrap();
    a.jmp(body).unwrap();
    a.nop().unwrap();
    a.set_label(&mut body).unwrap();
    // EBP starts at zero. The static backedge makes this eligible for a region,
    // including when XRSTOR terminates its block before the following instruction.
    a.test(ebp, ebp).unwrap();
    a.js(top).unwrap();
}

fn seed(cpu: &mut Vcpu, mask: u64) {
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, mask as u32 as u64 | 0xface_0000_0000_0000);
    cpu.set_reg(Reg::Rdx, mask >> 32 | 0xbeef_0000_0000_0000);
    cpu.cpu.mxcsr = 0x5fc1;
    cpu.cpu.fpu_cw = 0x077f;
    cpu.cpu.fpu_top = 3;
    cpu.cpu.fpu_empty = 0x5a;
    cpu.cpu.fpu_fop = 0x321;
    cpu.cpu.fpu_fip = 0x1234_5678_abcd_ef00;
    cpu.cpu.fpu_fdp = 0x2345_6789_abcd_ef00;
    for i in 0..8 {
        cpu.cpu.fpr[i] = [i as u8 + 1; 10];
        cpu.cpu.kmask[i] = 0x1234_5678_9000 + i as u64;
    }
    for i in 0..32 {
        cpu.cpu.xmm[i] = 0x1000 + i as u128;
        cpu.cpu.ymm_hi[i] = 0x2000 + i as u128;
        cpu.cpu.zmm_hi[i] = [0x3000 + i as u128, 0x4000 + i as u128];
    }
}
fn bytes(vm: &Vm, addr: u64, len: usize) -> Vec<u8> {
    let mut buf = vec![0; len];
    vm.read_bytes(addr, &mut buf).unwrap();
    buf
}
fn u64_at(vm: &Vm, off: u64) -> u64 {
    u64::from_le_bytes(bytes(vm, AREA + off, 8).try_into().unwrap())
}

#[test]
fn xsave_all_forms_round_trip_all_components_and_tiers() {
    for tier in 0..3 {
        for kind in 0..6 {
            let vm = vm(tier, CpuMode::Long64, |a| match kind {
                0 => a.xsave(ptr(AREA)).unwrap(),
                1 => a.xsave64(ptr(AREA)).unwrap(),
                2 => a.xsaveopt(ptr(AREA)).unwrap(),
                3 => a.xsaveopt64(ptr(AREA)).unwrap(),
                4 => a.xsavec(ptr(AREA)).unwrap(),
                _ => a.xsavec64(ptr(AREA)).unwrap(),
            });
            let mut cpu = vm.new_vcpu();
            seed(&mut cpu, 0xe7);
            let before = cpu.cpu.clone();
            assert!(
                matches!(cpu.run(&vm, None), Exit::Hlt),
                "tier={tier}, kind={kind}"
            );
            for i in 0..16 {
                assert_eq!(
                    bytes(&vm, AREA + 160 + i * 16, 16),
                    (0x1000u128 + i as u128).to_le_bytes()
                );
                assert_eq!(
                    bytes(&vm, AREA + 576 + i * 16, 16),
                    (0x2000u128 + i as u128).to_le_bytes()
                );
                assert_eq!(
                    bytes(&vm, AREA + 896 + i * 32, 16),
                    (0x3000u128 + i as u128).to_le_bytes()
                );
                for (j, value) in [0x1010u128, 0x2010, 0x3010, 0x4010].iter().enumerate() {
                    assert_eq!(
                        bytes(&vm, AREA + 1408 + i * 64 + j as u64 * 16, 16),
                        (value + i as u128).to_le_bytes()
                    );
                }
            }
            for i in 0..8 {
                assert_eq!(u64_at(&vm, 832 + i * 8), 0x1234_5678_9000 + i);
                assert_eq!(
                    bytes(&vm, AREA + 32 + i * 16, 10),
                    before.fpr[(i as usize + 3) & 7]
                );
            }
            if tier == 2 {
                assert!(vm.cache.regions() > 0, "must execute a compiled region");
            }
            assert_eq!(u64_at(&vm, 512), 0xe7);
            assert_eq!(
                u64_at(&vm, 520),
                if kind >= 4 { (1 << 63) | 0xe7 } else { 0 }
            );
            let mut a = CodeAssembler::new(64).unwrap();
            region_entry(&mut a);
            if kind & 1 == 1 {
                a.xrstor64(ptr(AREA)).unwrap();
            } else {
                a.xrstor(ptr(AREA)).unwrap();
            }
            a.hlt().unwrap();
            vm.write_bytes(CODE + 0x100, &a.assemble(CODE + 0x100).unwrap())
                .unwrap();
            let mut restored = vm.new_vcpu();
            restored.set_reg(Reg::Rip, CODE + 0x100);
            restored.set_reg(Reg::Rax, 0xe7);
            assert!(matches!(restored.run(&vm, None), Exit::Hlt));
            if tier == 2 {
                assert!(
                    vm.cache.regions() >= 2,
                    "save and restore must each compile a region"
                );
            }
            assert_eq!(restored.cpu.xmm, before.xmm);
            assert_eq!(restored.cpu.ymm_hi, before.ymm_hi);
            assert_eq!(restored.cpu.zmm_hi, before.zmm_hi);
            assert_eq!(restored.cpu.kmask, before.kmask);
            assert_eq!(restored.cpu.fpr, before.fpr);
            assert_eq!(restored.cpu.fpu_top, 3);
            assert_eq!(restored.cpu.fpu_empty, 0x5a);
            assert_eq!(restored.cpu.fpu_cw, 0x077f);
            assert_eq!(restored.cpu.mxcsr, 0x5fc1);
            assert_eq!(restored.cpu.fpu_fop, before.fpu_fop);
            let wide = kind & 1 == 1;
            // SDM Vol 3A §4.5.3 p. 4-6: XRSTOR64 loads the seeded 0x1234_5678_abcd_ef00
            // canonical at the engine's 48 bits. FDP is not made canonical.
            assert_eq!(
                restored.cpu.fpu_fip,
                if wide {
                    0x0000_5678_abcd_ef00
                } else {
                    before.fpu_fip as u32 as u64
                }
            );
            assert_eq!(
                restored.cpu.fpu_fdp,
                if wide {
                    before.fpu_fdp
                } else {
                    before.fpu_fdp as u32 as u64
                }
            );
        }
    }
}

#[test]
fn xsave_selective_layout_and_header_witness() {
    // SDM Vol 1 §§13.4.1, 13.5.3, 13.7: AVX-only also writes MXCSR,
    // preserves SSE/x87, reserved bytes, XCOMP_BV and unrequested XSTATE_BV.
    for tier in 0..3 {
        let vm = vm(tier, CpuMode::Long64, |a| {
            a.xsave64(ptr(AREA)).unwrap();
        });
        vm.write_bytes(AREA, &[0xa5; 4096]).unwrap();
        let mut cpu = vm.new_vcpu();
        seed(&mut cpu, u64::MAX);
        cpu.cpu.cpuid_overrides.xcr0 = Some(7);
        cpu.set_reg(Reg::Rax, 4);
        cpu.set_reg(Reg::Rdx, 0);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(bytes(&vm, AREA, 24), vec![0xa5; 24]);
        assert_eq!(bytes(&vm, AREA + 32, 480), vec![0xa5; 480]);
        assert_eq!(bytes(&vm, AREA + 24, 4), 0x5fc1u32.to_le_bytes());
        assert_eq!(u64_at(&vm, 512), 0xa5a5_a5a5_a5a5_a5a5 | 4);
        assert_eq!(bytes(&vm, AREA + 520, 56), vec![0xa5; 56]);
        assert_eq!(bytes(&vm, AREA + 576, 16), 0x2000u128.to_le_bytes());
        assert_eq!(bytes(&vm, AREA + 816, 16), 0x200fu128.to_le_bytes());
        assert_eq!(bytes(&vm, AREA + 832, 512), vec![0xa5; 512]);
    }
}

#[test]
fn xsave_xcr0_masks_disabled_components_on_all_tiers() {
    // SDM Vol 1 §§13.7–13.10: RFBM = XCR0 & EDX:EAX. Disabled components
    // are neither transferred nor initialized; compacted headers also mask them.
    for tier in 0..3 {
        for kind in 0..4 {
            for mapped_extended in [false, true] {
                let mut vm = Vm::with_backend(VmConfig::flat(0x20000), backend(tier));
                vm.set_guest_cpu_features(GuestCpuFeatures::v4());
                vm.map(CODE, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.map(AREA, 576, Prot::RW, RegionKind::Ram).unwrap();
                if mapped_extended {
                    vm.map(AREA + 576, 1856, Prot::RW, RegionKind::Ram).unwrap();
                    vm.write_bytes(AREA + 576, &[0xa5; 1856]).unwrap();
                }
                vm.write_bytes(AREA + 24, &0x1f80u32.to_le_bytes()).unwrap();
                if kind == 3 {
                    vm.write_bytes(AREA + 512, &3u64.to_le_bytes()).unwrap();
                }
                let mut a = CodeAssembler::new(64).unwrap();
                region_entry(&mut a);
                match kind {
                    0 => a.xsave64(ptr(AREA)).unwrap(),
                    1 => a.xsaveopt64(ptr(AREA)).unwrap(),
                    2 => a.xsavec64(ptr(AREA)).unwrap(),
                    _ => a.xrstor64(ptr(AREA)).unwrap(),
                }
                a.hlt().unwrap();
                vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
                let mut cpu = vm.new_vcpu();
                seed(&mut cpu, 0xe7);
                cpu.cpu.cpuid_overrides.xcr0 = Some(3);
                let before = cpu.cpu.clone();
                let exit = cpu.run(&vm, None);
                assert!(
                    matches!(exit, Exit::Hlt),
                    "tier={tier}, kind={kind}, mapped_extended={mapped_extended}: {exit:?}"
                );
                if tier == 2 {
                    assert!(vm.cache.regions() > 0, "must execute a compiled region");
                }
                assert_eq!(cpu.cpu.ymm_hi, before.ymm_hi);
                assert_eq!(cpu.cpu.zmm_hi, before.zmm_hi);
                assert_eq!(cpu.cpu.kmask, before.kmask);
                assert_eq!(cpu.cpu.xmm[16..], before.xmm[16..]);
                if mapped_extended {
                    assert_eq!(bytes(&vm, AREA + 576, 1856), vec![0xa5; 1856]);
                }
                assert_eq!(u64_at(&vm, 512), 3);
                assert_eq!(u64_at(&vm, 520), if kind == 2 { (1 << 63) | 3 } else { 0 });
                if kind == 3 {
                    assert_eq!(cpu.cpu.xmm[..16], [0; 16]);
                    assert_eq!(cpu.cpu.mxcsr, 0x1f80);
                } else {
                    assert_eq!(bytes(&vm, AREA + 160, 16), before.xmm[0].to_le_bytes());
                }
            }
        }
    }
}

#[test]
fn xsave_exception_metadata_matches_on_all_tiers() {
    use x86jit_core::{ErrorCode, FaultCause, FaultClass, FaultOrigin, Feature, Knowledge};
    // SDM Vol 1 §§13.7–13.10; Vol 2D §6.3 XGETBV: misalignment and
    // reserved MXCSR bits raise #GP(0); OSXSAVE=0 raises #UD (no error code).
    for case in 0..8 {
        let (vector, cause) = match case {
            0..=2 => (13, FaultCause::RequiredAlignment),
            3 => (13, FaultCause::InvalidMxcsr),
            _ => (6, FaultCause::InvalidOpcode),
        };
        let mut interp = None;
        for tier in 0..3 {
            let vm = vm(tier, CpuMode::Long64, |a| match case {
                0 => a.xsave64(ptr(AREA + 1)).unwrap(),
                1 => a.xsaveopt64(ptr(AREA + 1)).unwrap(),
                2 => a.xsavec64(ptr(AREA + 1)).unwrap(),
                3 => a.xrstor64(ptr(AREA)).unwrap(),
                4 => a.xsave64(ptr(AREA)).unwrap(),
                5 => a.xsaveopt64(ptr(AREA)).unwrap(),
                6 => a.xsavec64(ptr(AREA)).unwrap(),
                _ => a.xgetbv().unwrap(),
            });
            let mut cpu = vm.new_vcpu();
            seed(&mut cpu, 7);
            if case == 3 {
                vm.write_bytes(AREA + 24, &0x1_1f80u32.to_le_bytes())
                    .unwrap();
            }
            if case >= 4 {
                cpu.cpu.features = cpu.cpu.features.without(Feature::Osxsave);
            }
            let exit = cpu.run(&vm, None);
            assert!(matches!(exit, Exit::Exception { vector: v, .. } if v == vector));
            if tier == 2 {
                assert!(vm.cache.regions() > 0, "must execute a compiled region");
            }
            let m = exit.fault_metadata().unwrap();
            let detail = (m.origin(), m.class(), m.error_code(), m.vector());
            assert_eq!(
                detail,
                (
                    FaultOrigin::GuestException(cause),
                    FaultClass::Fault,
                    if vector == 13 {
                        ErrorCode::bits(0, u32::MAX)
                    } else {
                        ErrorCode::NoCode
                    },
                    Knowledge::Known(vector),
                ),
                "tier={tier}, case={case}: {exit:?}"
            );
            if let Some(expected) = interp {
                assert_eq!(detail, expected, "tier={tier}, case={case}");
            } else {
                interp = Some(detail);
            }
        }
    }
}

#[test]
fn xrstor_initializes_only_requested_components_and_loads_mxcsr() {
    // SDM Vol 1 §§13.6, 13.8.1: clear BV initializes, RFBM=0 preserves.
    for tier in 0..3 {
        let vm = vm(tier, CpuMode::Long64, |a| {
            a.xrstor64(ptr(AREA)).unwrap();
        });
        vm.write_bytes(AREA + 24, &0x3f80u32.to_le_bytes()).unwrap();
        let mut cpu = vm.new_vcpu();
        seed(&mut cpu, 7);
        let upper = cpu.cpu.zmm_hi;
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.cpu.fpu_cw, 0x037f);
        assert_eq!(cpu.cpu.fpu_top, 0);
        assert_eq!(cpu.cpu.fpu_empty, 0xff);
        assert_eq!(cpu.cpu.fpr, [[0; 10]; 8]);
        assert_eq!(cpu.cpu.fpu_fop, 0);
        assert_eq!(cpu.cpu.fpu_fip, 0);
        assert_eq!(cpu.cpu.fpu_fdp, 0);
        assert_eq!(cpu.cpu.xmm[..16], [0; 16]);
        assert_eq!(cpu.cpu.ymm_hi[..16], [0; 16]);
        assert_eq!(cpu.cpu.zmm_hi, upper);
        assert_eq!(cpu.cpu.mxcsr, 0x3f80);
    }
}

#[test]
fn xrstor_invalid_header_and_mxcsr_are_atomic() {
    for tier in 0..3 {
        for off in [512, 520, 528, 24] {
            let vm = vm(tier, CpuMode::Long64, |a| {
                a.xrstor64(ptr(AREA)).unwrap();
            });
            vm.write_bytes(AREA + 24, &0x1f80u32.to_le_bytes()).unwrap();
            vm.write_bytes(
                AREA + off,
                &(if off == 24 { 1u64 << 16 } else { 1u64 << 9 }).to_le_bytes(),
            )
            .unwrap();
            let mut cpu = vm.new_vcpu();
            seed(&mut cpu, 7);
            let before = cpu.cpu.clone();
            assert!(matches!(
                cpu.run(&vm, None),
                Exit::Exception { vector: 13, .. }
            ));
            assert_eq!(cpu.cpu.xmm, before.xmm);
            assert_eq!(cpu.cpu.fpr, before.fpr);
            assert_eq!(cpu.cpu.fpu_cw, before.fpu_cw);
            assert_eq!(cpu.cpu.mxcsr, before.mxcsr);
        }
    }
}

#[test]
fn xsave_fault_reports_exact_byte_without_partial_commit() {
    // SDM Vol 1 §13.5.3: the AVX component is exactly 256 bytes in long mode.
    // Truncate each component and the header independently; save must probe first.
    for tier in 0..3 {
        for restore in [false, true] {
            for end in [23, 27, 153, 415, 519, 575, 831] {
                let mut vm = Vm::with_backend(VmConfig::flat(0x20000), backend(tier));
                vm.set_guest_cpu_features(GuestCpuFeatures::v3());
                vm.map(CODE, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.map(AREA, end, Prot::RW, RegionKind::Ram).unwrap();
                if end < 512 {
                    vm.map(AREA + 512, 64, Prot::RW, RegionKind::Ram).unwrap();
                    vm.map(AREA + 576, 256, Prot::RW, RegionKind::Ram).unwrap();
                    vm.write_bytes(AREA + 512, &7u64.to_le_bytes()).unwrap();
                }
                let mut a = CodeAssembler::new(64).unwrap();
                region_entry(&mut a);
                if restore {
                    a.xrstor64(ptr(AREA)).unwrap();
                } else {
                    a.xsave64(ptr(AREA)).unwrap();
                }
                a.hlt().unwrap();
                vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
                let mut initial = vec![0x5a; end];
                if end >= 28 {
                    initial[24..28].copy_from_slice(&0x1f80u32.to_le_bytes());
                }
                if end >= 520 {
                    initial[512..520].copy_from_slice(&7u64.to_le_bytes());
                }
                if end > 520 {
                    initial[520..].fill(0);
                }
                vm.write_bytes(AREA, &initial).unwrap();
                let mut cpu = vm.new_vcpu();
                seed(
                    &mut cpu,
                    match end {
                        23 | 153 => 1,
                        27 | 415 => 2,
                        _ => 7,
                    },
                );
                let before = cpu.cpu.clone();
                let exit = cpu.run(&vm, None);
                // Header is checked first. A save reads just BV; a restore reads all 64 bytes.
                let expected = if !restore && end == 575 {
                    AREA + 576
                } else {
                    AREA + end as u64
                };
                assert!(
                    matches!(exit, Exit::UnmappedMemory { addr, .. } if addr == expected),
                    "tier={tier}, restore={restore}, end={end}: {exit:?}"
                );
                let code_bytes = a.assemble(CODE).unwrap();
                let mut decoded = iced_x86::Decoder::with_ip(
                    64,
                    &code_bytes,
                    CODE,
                    iced_x86::DecoderOptions::NONE,
                );
                let fault_ip = loop {
                    let insn = decoded.decode();
                    if matches!(
                        insn.mnemonic(),
                        iced_x86::Mnemonic::Xsave64 | iced_x86::Mnemonic::Xrstor64
                    ) {
                        break insn.ip();
                    }
                };
                assert_eq!(cpu.reg(Reg::Rip), fault_ip);
                assert_eq!(cpu.cpu.xmm, before.xmm);
                assert_eq!(cpu.cpu.fpr, before.fpr);
                assert_eq!(cpu.cpu.mxcsr, before.mxcsr);
                assert_eq!(bytes(&vm, AREA, end), initial);
            }
        }
    }
}

#[test]
fn xsave_inactive_components_do_not_access_memory() {
    for tier in 0..3 {
        for restore in [false, true] {
            let mut vm = Vm::with_backend(VmConfig::flat(0x20000), backend(tier));
            vm.map(CODE, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
            // Map only the mandatory header and the selected MXCSR/AVX fields.
            vm.map(AREA + 512, 64, Prot::RW, RegionKind::Ram).unwrap();
            vm.map(AREA + 24, 8, Prot::RW, RegionKind::Ram).unwrap();
            vm.map(AREA + 576, 256, Prot::RW, RegionKind::Ram).unwrap();
            let mut a = CodeAssembler::new(64).unwrap();
            region_entry(&mut a);
            if restore {
                a.xrstor64(ptr(AREA)).unwrap();
            } else {
                a.xsave64(ptr(AREA)).unwrap();
            }
            a.hlt().unwrap();
            vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
            let mut cpu = vm.new_vcpu();
            seed(&mut cpu, 4);
            let old_xmm = cpu.cpu.xmm;
            vm.write_bytes(AREA + 24, &0x1f80u32.to_le_bytes()).unwrap();
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.cpu.xmm, old_xmm);
        }
    }
}

#[test]
fn xsave_compacted_sparse_layout_and_mxcsr_initialization() {
    // SDM Vol 1 §§13.4.3, 13.8.2: a compacted opmask-only image starts at 576,
    // and a requested SSE component absent from BV initializes MXCSR.
    for tier in 0..3 {
        let vm = vm(tier, CpuMode::Long64, |a| {
            a.xsavec64(ptr(AREA)).unwrap();
        });
        let mut cpu = vm.new_vcpu();
        seed(&mut cpu, 1 << 5);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(u64_at(&vm, 512), 1 << 5);
        assert_eq!(u64_at(&vm, 520), (1 << 63) | (1 << 5));
        for i in 0..8 {
            assert_eq!(u64_at(&vm, 576 + i * 8), 0x1234_5678_9000 + i);
        }
        let mut a = CodeAssembler::new(64).unwrap();
        region_entry(&mut a);
        a.xrstor64(ptr(AREA)).unwrap();
        a.hlt().unwrap();
        vm.write_bytes(CODE + 0x100, &a.assemble(CODE + 0x100).unwrap())
            .unwrap();
        let mut cpu = vm.new_vcpu();
        seed(&mut cpu, 0x22);
        cpu.cpu.kmask.fill(0);
        cpu.set_reg(Reg::Rip, CODE + 0x100);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.cpu.mxcsr, 0x1f80);
        assert_eq!(cpu.cpu.xmm[..16], [0; 16]);
        assert_eq!(cpu.cpu.kmask[7], 0x1234_5678_9007);
        for (offset, value) in [(512, 1u64 << 6), (528, 1)] {
            vm.write_bytes(AREA + offset, &value.to_le_bytes()).unwrap();
            cpu.set_reg(Reg::Rip, CODE + 0x100);
            assert!(matches!(
                cpu.run(&vm, None),
                Exit::Exception { vector: 13, .. }
            ));
            vm.write_bytes(AREA + offset, &0u64.to_le_bytes()).unwrap();
        }
    }
}

#[test]
fn xsave_compat_preserves_inaccessible_vector_registers() {
    // SDM Vol 1 §§13.5.2–13.5.5: compat mode accesses only XMM/YMM/ZMM0–7
    // and never the Hi16_ZMM component, even when requested.
    for tier in 0..3 {
        let vm = vm(tier, CpuMode::Compat32, |a| {
            a.xsave(ptr(AREA as u32)).unwrap();
        });
        vm.write_bytes(AREA, &[0x5a; 4096]).unwrap();
        vm.write_bytes(AREA + 512, &[0; 64]).unwrap();
        let mut cpu = vm.new_vcpu();
        seed(&mut cpu, 0xe7);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        for (off, len) in [(288, 128), (704, 128), (1152, 256), (1408, 1024)] {
            assert_eq!(bytes(&vm, AREA + off, len), vec![0x5a; len]);
        }
        let mut a = CodeAssembler::new(32).unwrap();
        region_entry(&mut a);
        a.xrstor(ptr(AREA as u32)).unwrap();
        a.hlt().unwrap();
        vm.write_bytes(CODE + 0x100, &a.assemble(CODE + 0x100).unwrap())
            .unwrap();
        let mut cpu = vm.new_vcpu();
        seed(&mut cpu, 0xe7);
        cpu.cpu.xmm[..8].fill(0);
        cpu.set_reg(Reg::Rip, CODE + 0x100);
        let before = cpu.cpu.clone();
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.cpu.xmm[0], 0x1000);
        assert_eq!(cpu.cpu.xmm[8..], before.xmm[8..]);
        assert_eq!(cpu.cpu.ymm_hi[8..], before.ymm_hi[8..]);
        assert_eq!(cpu.cpu.zmm_hi[8..], before.zmm_hi[8..]);
    }
}

#[test]
fn xsave_cpuid_layout_and_xgetbv_consistency() {
    // SDM Vol 1 §13.2: mask, enabled size, maximum size, per-component size/offset.
    for tier in 0..3 {
        let vm = vm(tier, CpuMode::Long64, |a| {
            a.xgetbv().unwrap();
        });
        let mut cpu = vm.new_vcpu();
        for (mask, size) in [(3, 576), (7, 832), (0xe7, 2432)] {
            cpu.cpu.cpuid_overrides.xcr0 = Some(mask);
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rcx, 0);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), mask);
            assert_eq!(cpu.reg(Reg::Rdx), 0);
            let leaf = cpu.cpu.cpuid(0xd, 0);
            assert_eq!(
                (leaf.eax, leaf.ebx, leaf.ecx, leaf.edx),
                (0xe7, size, 2432, 0)
            );
            assert_eq!(cpu.cpu.cpuid(0xd, 1).ebx, size);
        }
        for (bit, size, off) in [(2, 256, 576), (5, 64, 832), (6, 512, 896), (7, 1024, 1408)] {
            let leaf = cpu.cpu.cpuid(0xd, bit);
            assert_eq!((leaf.eax, leaf.ebx, leaf.ecx, leaf.edx), (size, off, 0, 0));
        }
        assert_eq!(cpu.cpu.cpuid(0xd, 1).eax, 3);
        assert_eq!(cpu.cpu.cpuid(0, 0).eax, 0xd);
        for index in [1, 2, u32::MAX] {
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rcx, index as u64);
            cpu.set_reg(Reg::Rax, 0x1234);
            assert!(matches!(
                cpu.run(&vm, None),
                Exit::Exception { vector: 13, .. }
            ));
            assert_eq!(cpu.reg(Reg::Rax), 0x1234);
        }
    }
}

#[test]
fn xsave_feature_gating_and_alignment_faults() {
    use x86jit_core::{CpuidOverride, CpuidRegisters, Feature};
    for tier in 0..3 {
        for kind in 0..5 {
            let vm = vm(tier, CpuMode::Long64, |a| match kind {
                0 => a.xsave(ptr(AREA + 1)).unwrap(),
                1 => a.xrstor(ptr(AREA + 1)).unwrap(),
                2 => a.xsaveopt(ptr(AREA + 1)).unwrap(),
                3 => a.xsavec(ptr(AREA + 1)).unwrap(),
                _ => a.xgetbv().unwrap(),
            });
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            if kind != 4 {
                assert!(matches!(
                    cpu.run(&vm, None),
                    Exit::Exception { vector: 13, .. }
                ));
            }
            cpu.cpu.features = cpu.cpu.features.without(Feature::Osxsave);
            cpu.set_reg(Reg::Rip, CODE);
            assert!(matches!(
                cpu.run(&vm, None),
                Exit::Exception { vector: 6, .. }
            ));
            cpu.cpu.features = GuestCpuFeatures::v4();
            if matches!(kind, 2 | 3) {
                cpu.cpu.cpuid_overrides.leaves.insert(
                    (0xd, 1),
                    CpuidOverride {
                        mask: CpuidRegisters::new(3, 0, 0, 0),
                        value: CpuidRegisters::ZERO,
                    },
                );
                cpu.set_reg(Reg::Rip, CODE);
                assert!(matches!(
                    cpu.run(&vm, None),
                    Exit::Exception { vector: 6, .. }
                ));
            }
        }
    }
}

fn oracle_input(kind: usize, mask: u32) -> x86jit_tests::oracle::VectorInput {
    use x86jit_tests::oracle::VectorInput;
    use x86jit_tests::vector::{CpuSnapshot, MemChunk, MemKind, RunSpec};
    let mut cpu = CpuSnapshot {
        rip: CODE,
        fpu_cw: 0x077f,
        mxcsr: 0x5fc0,
        ..Default::default()
    };
    cpu.gpr[0] = mask as u64;
    for i in 0..if mask & 0xe0 != 0 { 32 } else { 16 } {
        cpu.xmm[i] = 0x1000 + i as u128;
        cpu.ymm_hi[i] = 0x2000 + i as u128;
        if mask & 0xe0 != 0 {
            cpu.zmm_hi[i] = [0x3000 + i as u128, 0x4000 + i as u128];
        }
    }
    if mask & 0xe0 != 0 {
        for i in 0..8 {
            cpu.kmask[i] = 0x8765 + i as u64;
        }
    }
    // SDM Vol 1 §4.2.2: explicit integer bit and exponent bias encode 1.0.
    cpu.st.fill([0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f]);
    let zmms = [
        zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13,
        zmm14, zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24, zmm25, zmm26,
        zmm27, zmm28, zmm29, zmm30, zmm31,
    ];
    let masks = [k0, k1, k2, k3, k4, k5, k6, k7];
    let mut state = Vec::new();
    let mut a = CodeAssembler::new(64).unwrap();
    region_entry(&mut a);
    if mask & 0xe0 != 0 {
        // SDM Vol 2, VMOVDQU64 and KMOVQ: establish extended state in the
        // guest; the native stub can seed only XMM0–15 and YMM0–15.
        for (i, z) in zmms.into_iter().enumerate() {
            a.vmovdqu64(z, zmmword_ptr(AREA + 4096 + state.len() as u64))
                .unwrap();
            for value in [
                cpu.xmm[i],
                cpu.ymm_hi[i],
                cpu.zmm_hi[i][0],
                cpu.zmm_hi[i][1],
            ] {
                state.extend_from_slice(&value.to_le_bytes());
            }
        }
        for (i, k) in masks.into_iter().enumerate() {
            a.kmovq(k, qword_ptr(AREA + 4096 + state.len() as u64))
                .unwrap();
            state.extend_from_slice(&cpu.kmask[i].to_le_bytes());
        }
    }
    cpu.zmm_hi.fill([0; 2]);
    cpu.kmask.fill(0);
    cpu.xmm[16..].fill(0);
    cpu.ymm_hi[16..].fill(0);
    match kind {
        0 => a.xsave64(ptr(AREA)).unwrap(),
        1 => a.xsaveopt64(ptr(AREA)).unwrap(),
        _ => a.xsavec64(ptr(AREA)).unwrap(),
    }
    a.fninit().unwrap();
    a.vzeroall().unwrap();
    if mask & 0xe0 != 0 {
        // SDM Vol 2, VPXORQ/KXORQ: clear saved registers before restoring them.
        for z in zmms {
            a.vpxorq(z, z, z).unwrap();
        }
        for k in masks {
            a.kxorq(k, k, k).unwrap();
        }
    }
    a.xrstor64(ptr(AREA)).unwrap();
    a.hlt().unwrap();
    VectorInput {
        cpu_init: cpu,
        entry: CODE,
        run: RunSpec::UntilExit,
        mem_init: vec![
            MemChunk {
                addr: CODE,
                bytes: a.assemble(CODE).unwrap(),
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: AREA,
                bytes: vec![0; 4096],
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: AREA + 4096,
                bytes: if state.is_empty() { vec![0; 64] } else { state },
                kind: MemKind::Ram,
            },
        ],
    }
}

fn check_round_trip_state(cpu: &x86jit_tests::vector::CpuSnapshot, mask: u32) {
    for i in 0..32 {
        let active = i < 16 || mask & 0xe0 != 0;
        assert_eq!(
            cpu.xmm[i],
            if active { 0x1000 + i as u128 } else { 0 },
            "xmm{i}"
        );
        assert_eq!(
            cpu.ymm_hi[i],
            if active { 0x2000 + i as u128 } else { 0 },
            "ymm{i} high"
        );
        assert_eq!(
            cpu.zmm_hi[i],
            if mask & 0xe0 != 0 {
                [0x3000 + i as u128, 0x4000 + i as u128]
            } else {
                [0; 2]
            },
            "zmm{i} high"
        );
    }
    for i in 0..8 {
        assert_eq!(
            cpu.kmask[i],
            if mask & 0xe0 != 0 {
                0x8765 + i as u64
            } else {
                0
            },
            "k{i}"
        );
    }
}

#[test]
fn xsave_oracle_guest_prologue_round_trip() {
    use x86jit_tests::{oracle::run_with_backend_features, vector::ExitKind};
    for mask in [7, 0xe7] {
        for kind in 0..3 {
            let input = oracle_input(kind, mask);
            assert!(input.cpu_init.zmm_hi.iter().flatten().all(|&v| v == 0));
            assert_eq!(input.cpu_init.kmask, [0; 8]);
            assert_eq!(input.cpu_init.xmm[16..], [0; 16]);
            assert_eq!(input.cpu_init.ymm_hi[16..], [0; 16]);
            for tier in 0..3 {
                let result =
                    run_with_backend_features(&input, backend(tier), GuestCpuFeatures::v4());
                assert_eq!(
                    result.exit,
                    ExitKind::Hlt,
                    "tier={tier} kind={kind} mask={mask:#x}"
                );
                check_round_trip_state(&result.cpu, mask);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn xsave_native_all_components_round_trip() {
    use x86jit_tests::oracle::run_with_backend_features;
    use x86jit_tests::vector::ExitKind;
    if !std::is_x86_feature_detected!("avx") {
        return;
    }
    // SDM Vol 1 §13.2 defines these availability bits.
    let extras = std::arch::x86_64::__cpuid_count(0xd, 1).eax;
    let wide = std::is_x86_feature_detected!("avx512f");
    let mask = if wide { 0xe7 } else { 7 };
    // SDM Vol 1 §13.3: requested components must be enabled in XCR0.
    // SAFETY: AVX detection above establishes OSXSAVE support.
    let xcr0 = unsafe { std::arch::x86_64::_xgetbv(0) };
    assert!(
        xcr0 & mask as u64 == mask as u64,
        "XCR0={xcr0:#x} missing requested components mask={mask:#x}"
    );
    for kind in 0..3 {
        if kind != 0 && extras & (1 << (kind - 1)) == 0 {
            continue;
        }
        let input = oracle_input(kind, mask);
        let native = x86jit_tests::native::run_native(&input).unwrap_or_else(|| {
            panic!("native XSAVE unavailable: kind={kind} mask={mask:#x} XCR0={xcr0:#x}")
        });
        assert_eq!(native.exit, ExitKind::Hlt);
        check_round_trip_state(&native.cpu, mask);
        assert_eq!(native.cpu.mxcsr, input.cpu_init.mxcsr);
        for tier in 0..3 {
            let result = run_with_backend_features(&input, backend(tier), GuestCpuFeatures::v4());
            assert_eq!(result.exit, ExitKind::Hlt);
            for (i, (actual, expected)) in result.cpu.xmm.iter().zip(&native.cpu.xmm).enumerate() {
                assert!(
                    actual == expected,
                    "xmm[{i}] engine={actual:x?} native={expected:x?}; tier={tier} kind={kind}"
                );
            }
            for (i, (actual, expected)) in
                result.cpu.ymm_hi.iter().zip(&native.cpu.ymm_hi).enumerate()
            {
                assert!(
                    actual == expected,
                    "ymm_hi[{i}] engine={actual:x?} native={expected:x?}; tier={tier} kind={kind}"
                );
            }
            assert_eq!(result.cpu.fpu_cw, native.cpu.fpu_cw);
            for (i, (actual, expected)) in result.cpu.st.iter().zip(&native.cpu.st).enumerate() {
                assert!(
                    actual == expected,
                    "st[{i}] engine={actual:x?} native={expected:x?}; tier={tier} kind={kind}"
                );
            }
            assert_eq!(result.cpu.mxcsr, native.cpu.mxcsr);
            if wide {
                for (i, (actual, expected)) in
                    result.cpu.zmm_hi.iter().zip(&native.cpu.zmm_hi).enumerate()
                {
                    assert!(actual == expected, "zmm_hi[{i}] engine={actual:x?} native={expected:x?}; tier={tier} kind={kind}");
                }
                for (i, (actual, expected)) in
                    result.cpu.kmask.iter().zip(&native.cpu.kmask).enumerate()
                {
                    assert!(actual == expected, "kmask[{i}] engine={actual:x?} native={expected:x?}; tier={tier} kind={kind}");
                }
            }
            // Compare only defined fields, mapping standard offsets through host CPUID.
            let host = &native.mem.iter().find(|m| m.addr == AREA).unwrap().bytes;
            let engine = &result.mem.iter().find(|m| m.addr == AREA).unwrap().bytes;
            for i in (24..28).chain(160..416) {
                assert!(
                    engine[i] == host[i],
                    "legacy byte={i} engine={:#x} native={:#x}; tier={tier} kind={kind}",
                    engine[i],
                    host[i]
                );
            }
            let mut compact_offset = 576;
            for (bit, size, engine_off) in
                [(2, 256, 576), (5, 64, 832), (6, 512, 896), (7, 1024, 1408)]
            {
                if mask & (1 << bit) == 0 {
                    continue;
                }
                // SDM Vol 1 §13.2: query this supported extended-state component.
                let leaf = std::arch::x86_64::__cpuid_count(0xd, bit);
                if kind == 2 && leaf.ecx & 2 != 0 {
                    compact_offset = (compact_offset + 63) & !63;
                }
                let host_off = if kind == 2 {
                    compact_offset
                } else {
                    leaf.ebx as usize
                };
                for i in 0..size {
                    assert!(engine[engine_off + i] == host[host_off + i],
                        "component={bit} byte={i} engine={:#x} native={:#x}; tier={tier} kind={kind}",
                        engine[engine_off + i], host[host_off + i]);
                }
                compact_offset += size;
            }
        }
    }
}

#[cfg(feature = "unicorn")]
// MEASURED with Unicorn 2.1.4 on macOS/aarch64: this oracle run aborts with
// SIGILL even with only aligned XSAVE64 (48 0f ae 24 25 00 80 01 00) in the body.
// The unchanged MXCSR oracle tests also abort. SDM Vol 2D §6.3 XSAVE permits
// the encoding; retain this comparison for explicit execution on a working runtime.
#[test]
#[ignore = "Requires a working Unicorn XSAVE runtime; local oracle runs abort with SIGILL"]
fn xsave_unicorn_supported_components_round_trip() {
    use x86jit_tests::oracle::{run_with_backend, Oracle};
    use x86jit_tests::vector::ExitKind;
    let input = oracle_input(0, 7);
    let reference = x86jit_tests::unicorn::UnicornOracle.run(&input);
    assert_eq!(reference.exit, ExitKind::Hlt);
    for tier in 0..3 {
        let result = run_with_backend(&input, backend(tier));
        assert_eq!(result.exit, ExitKind::Hlt);
        assert_eq!(result.cpu.xmm[..16], reference.cpu.xmm[..16]);
        assert_eq!(result.cpu.ymm_hi[..16], reference.cpu.ymm_hi[..16]);
        assert_eq!(result.cpu.fpu_cw, reference.cpu.fpu_cw);
        assert_eq!(result.cpu.mxcsr, reference.cpu.mxcsr);
    }
}

#[test]
fn xrstor_mxcsr_changes_successor_arithmetic() {
    // SDM Vol 1 §§10.2.3.2, 13.8.1: an AVX-only standard restore loads RC.
    for tier in 0..3 {
        let vm = vm(tier, CpuMode::Long64, |a| {
            a.xrstor64(ptr(AREA)).unwrap();
            a.addss(xmm0, xmm1).unwrap();
        });
        vm.write_bytes(AREA + 24, &0x5f80u32.to_le_bytes()).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rax, 4);
        cpu.set_xmm(0, 0x3f80_0000);
        cpu.set_xmm(1, 0x3380_0000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.cpu.xmm[0] as u32, 0x3f80_0001);
    }
}

#[test]
fn xsave_uses_live_cpuid_component_offsets() {
    // SDM Vol 1 §§13.2, 13.4.3: standard layout follows CPUID.0DH.i:EBX.
    use x86jit_core::{CpuidOverride, CpuidRegisters};
    for tier in 0..3 {
        let vm = vm(tier, CpuMode::Long64, |a| {
            a.xsave64(ptr(AREA)).unwrap();
        });
        let mut cpu = vm.new_vcpu();
        seed(&mut cpu, 4);
        cpu.cpu.cpuid_overrides.leaves.insert(
            (0xd, 2),
            CpuidOverride {
                mask: CpuidRegisters::new(0, u32::MAX, 0, 0),
                value: CpuidRegisters::new(0, 2048, 0, 0),
            },
        );
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(bytes(&vm, AREA + 576, 256), vec![0; 256]);
        assert_eq!(bytes(&vm, AREA + 2048, 16), 0x2000u128.to_le_bytes());
        assert_eq!(bytes(&vm, AREA + 2288, 16), 0x200fu128.to_le_bytes());
    }
}

#[test]
fn xsave_x87_pointer_model_matches_legacy_image_policy() {
    // SDM Vol 1 §§8.1.8, 13.5.1; AMD64 APM Vol 4 §2 XSAVE: in the AMD model,
    // error pointers/opcode are zero when no unmasked floating-point error exists.
    use x86jit_core::state::X87PointerModel;
    for tier in 0..3 {
        for model in [X87PointerModel::Sdm, X87PointerModel::Amd] {
            for summary in [false, true] {
                let vm = vm(tier, CpuMode::Long64, |a| {
                    a.xsave64(ptr(AREA)).unwrap();
                });
                let mut cpu = vm.new_vcpu();
                seed(&mut cpu, 1);
                cpu.cpu.x87_pointer_model = model;
                cpu.cpu.fpu_sw = if summary { 1 << 7 } else { 0 };
                let expected = if model == X87PointerModel::Amd && !summary {
                    0
                } else {
                    cpu.cpu.fpu_fip
                };
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                assert_eq!(u64_at(&vm, 8), expected);
                assert_eq!(
                    u64_at(&vm, 16),
                    if expected == 0 { 0 } else { cpu.cpu.fpu_fdp }
                );
                assert_eq!(
                    u16::from_le_bytes(bytes(&vm, AREA + 6, 2).try_into().unwrap()),
                    if expected == 0 { 0 } else { 0x321 }
                );
            }
        }
    }
}

#[test]
fn xsave_noncanonical_components_fault_in_the_selected_segment() {
    // SDM Vol 1 §3.3.7.1; Vol 2D §6.3 XSAVE/XRSTOR, 64-bit-mode exceptions.
    for tier in 0..3 {
        for stack in [false, true] {
            for restore in [false, true] {
                let vm = vm(tier, CpuMode::Long64, |a| match (stack, restore) {
                    (false, false) => a.xsave64(ptr(rdi)).unwrap(),
                    (false, true) => a.xrstor64(ptr(rdi)).unwrap(),
                    (true, false) => a.xsave64(ptr(rsp)).unwrap(),
                    (true, true) => a.xrstor64(ptr(rsp)).unwrap(),
                });
                let mut cpu = vm.new_vcpu();
                seed(&mut cpu, 7);
                cpu.set_reg(
                    if stack { Reg::Rsp } else { Reg::Rdi },
                    0x0000_8000_0000_0000,
                );
                let before = cpu.cpu.clone();
                let exit = cpu.run(&vm, None);
                assert!(
                    matches!(exit, Exit::Exception { vector, .. } if vector == if stack { 12 } else { 13 }),
                    "{exit:?}"
                );
                assert_eq!(cpu.cpu.xmm, before.xmm);
                assert_eq!(cpu.cpu.fpr, before.fpr);
                assert_eq!(cpu.cpu.mxcsr, before.mxcsr);
            }
        }
    }
}

#[test]
fn xstate_transfers_preserve_cpuid_override_storage() {
    use x86jit_core::{CpuidOverride, CpuidRegisters};
    for tier in 0..3 {
        let vm = vm(tier, CpuMode::Long64, |a| {
            a.xsave64(ptr(AREA)).unwrap();
            a.xrstor64(ptr(AREA)).unwrap();
        });
        let mut cpu = vm.new_vcpu();
        seed(&mut cpu, 7);
        cpu.set_reg(Reg::Rbp, 0x12345678);
        for leaf in 0x4000_0000..0x4000_0020 {
            cpu.cpu.cpuid_overrides.leaves.insert(
                (leaf, 0),
                CpuidOverride {
                    mask: CpuidRegisters::new(u32::MAX, 0, 0, 0),
                    value: CpuidRegisters::new(leaf, 0, 0, 0),
                },
            );
        }
        cpu.cpu.cpuid_overrides.xcr0 = Some(7);
        let address = cpu.cpu.cpuid_overrides.leaves.first_key_value().unwrap().1 as *const _;
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rbp), 0x12345678);
        assert_eq!(cpu.cpu.cpuid_overrides.xcr0, Some(7));
        assert_eq!(cpu.cpu.cpuid_overrides.leaves.len(), 32);
        assert_eq!(
            cpu.cpu.cpuid_overrides.leaves.first_key_value().unwrap().1 as *const _,
            address,
            "extended-state transfers must retain the original CPUID map"
        );
        assert_eq!(cpu.cpu.cpuid(0x4000_0010, 0).eax, 0x4000_0010);
    }
}
