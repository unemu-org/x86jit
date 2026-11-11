//! Same-CPL I386 protected segmentation (decision-18 §§1–4).
//! Authority: Intel SDM 325462-092US; 80386 PRM 230985-001 (scan checked).
#[path = "support/model_api.rs"]
mod model_api;
use x86jit_core::fault::{ErrorCode, FaultClass};
use x86jit_core::{
    Backend, CachedBlock, CpuGeneration, Exit, IrBlock, LegacyCpuModel, MemConsistency, Prot,
    RegionKind, SegmentCache, StepOptions, StepStop, UnsupportedSystemOperation, Vcpu, Vm,
    VmConfig,
};

const MODEL: LegacyCpuModel = LegacyCpuModel {
    generation: CpuGeneration::I386,
    model_semantics_id: 0x386,
};
struct NoBackend;
impl Backend for NoBackend {
    fn materialize(
        &self,
        _: &IrBlock,
        _: MemConsistency,
        _: Option<(u64, u64)>,
        _: u64,
    ) -> CachedBlock {
        panic!("legacy execution reached application backend")
    }
}
fn setup(code: &[u8], code32: bool, stack32: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x40000), Box::new(NoBackend));
    vm.map(0, 0x40000, Prot::RWX, RegionKind::Ram).unwrap();
    crate::model_api::select(&mut vm, MODEL);
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.gdtr.base = 0x2000;
    cpu.cpu.legacy.gdtr.limit = 0x3f;
    cpu.cpu.legacy.segments[0].attributes |= u16::from(code32) << 14;
    cpu.cpu.legacy.segments[3].attributes |= u16::from(stack32) << 14;
    (vm, cpu)
}
fn descriptor(vm: &Vm, selector: u16, base: u32, limit: u32, attr: u16) {
    // SDM Vol 3A §3.4.5 Fig. 3-8; 80386 PRM §5.1.1 Fig. 5-3:
    // inclusive limit, split base, access byte and flags nibble.
    let raw = if attr & 0x8000 != 0 {
        limit >> 12
    } else {
        limit
    };
    let low = (raw & 0xffff) | (base << 16);
    let high = ((base >> 16) & 0xff)
        | (base & 0xff000000)
        | (u32::from(attr & 0xf0ff) << 8)
        | (raw & 0xf0000);
    let address = 0x2000 + u64::from(selector & !7);
    vm.write_bytes(address, &low.to_le_bytes()).unwrap();
    vm.write_bytes(address + 4, &high.to_le_bytes()).unwrap();
}
fn value(vm: &Vm, addr: u64, size: usize) -> u64 {
    let mut b = [0; 8];
    vm.read_bytes(addr, &mut b[..size]).unwrap();
    u64::from_le_bytes(b)
}
fn steps(cpu: &mut Vcpu, vm: &Vm, count: u64) {
    let run = cpu.run_steps(vm, count, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), count, "{:?}", run.stop());
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{:?}",
        run.stop()
    );
}
fn fault(cpu: &mut Vcpu, vm: &Vm, vector: u8, code: u32) {
    let before = format!("{:?}", cpu.cpu);
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    let StepStop::Exit(exit @ Exit::Exception { vector: actual, .. }) = run.stop() else {
        panic!("{:?}", run.stop())
    };
    assert_eq!(*actual, vector);
    let details = exit.fault_metadata().expect("protected fault evidence");
    assert_eq!(details.class(), FaultClass::Fault);
    assert_eq!(details.saved_rip(), cpu.cpu.rip);
    assert_eq!(
        details.instruction_rip(),
        x86jit_core::fault::Knowledge::Known(cpu.cpu.rip)
    );
    // 80386 PRM §9.7 Fig. 9-7: bits 31:16 are undefined, including for #GP(0).
    assert_eq!(details.error_code(), ErrorCode::bits(code, 0xffff));
    assert_eq!(details.error_code().bit(16), None);
    assert_eq!(format!("{:?}", cpu.cpu), before);
}
fn refused(cpu: &mut Vcpu, vm: &Vm, operation: UnsupportedSystemOperation) {
    let before = format!("{:?}", cpu.cpu);
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::UnsupportedSystemContext { operation: op, .. }) if *op == operation),
        "{:?}",
        run.stop()
    );
    assert_eq!(format!("{:?}", cpu.cpu), before);
}

#[test]
fn enter_use32_reload_all_caches_execute_and_return_to_real_in_step_slices_cpu_model() {
    crate::model_api::with_cpu_model(
        enter_use32_reload_all_caches_execute_and_return_to_real_in_step_slices,
    );
}

#[test]
fn enter_use32_reload_all_caches_execute_and_return_to_real_in_step_slices() {
    // SDM Vol 3A §§12.9.1–12.9.2; 80386 PRM §§14.4.1–14.5 p. 14-4.
    let (vm, mut cpu) = setup(
        b"\x0f\x01\x16\x00\x21\x66\xb8\x01\x00\x00\x00\x0f\x22\xc0\x66\xea\x00\x12\x00\x00\x08\x00",
        false,
        false,
    );
    cpu.cpu.legacy.cr0 = 0;
    cpu.cpu.legacy.gdtr = Default::default();
    vm.write_bytes(0x2100, b"\x27\x00\x00\x20\x00\x00").unwrap();
    descriptor(&vm, 8, 0, 0xffff, 0x409a);
    descriptor(&vm, 16, 0, 0xffff, 0x4092);
    descriptor(&vm, 24, 0, 0xffff, 0x009a);
    descriptor(&vm, 32, 0, 0xffff, 0x0092);
    vm.write_bytes(0x1200, b"\x66\xb8\x10\x00\x8e\xd8\x8e\xc0\x8e\xe0\x8e\xe8\x8e\xd0\xb8\x78\x56\x34\x12\xa3\x00\x30\x00\x00\xea\x00\x14\x00\x00\x18\x00").unwrap();
    vm.write_bytes(0x1400, b"\xb8\x20\x00\x8e\xd8\x8e\xc0\x8e\xe0\x8e\xe8\x8e\xd0\x0f\x20\xc0\x66\x83\xe0\xfe\x0f\x22\xc0\xea\x00\x15\x00\x00").unwrap();
    vm.write_bytes(0x1500, b"\xbb\xfe\xca\xf4").unwrap();
    steps(&mut cpu, &vm, 2);
    let caches = cpu.cpu.legacy.segments;
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.legacy.cr0, 1);
    assert_eq!(cpu.cpu.legacy.segments, caches);
    assert_eq!(cpu.cpu.rip, 0x100e);
    steps(&mut cpu, &vm, 1);
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (8, 0x1200));
    assert_eq!(cpu.cpu.legacy.segments[0].attributes, 0x409b);
    steps(&mut cpu, &vm, 6);
    assert_eq!(
        [cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs],
        [16; 5]
    );
    for cache in &cpu.cpu.legacy.segments[1..] {
        assert_eq!(cache.attributes, 0x4093);
    }
    steps(&mut cpu, &vm, 3);
    assert_eq!(value(&vm, 0x3000, 4), 0x12345678);
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (24, 0x1400));
    steps(&mut cpu, &vm, 8);
    let caches = cpu.cpu.legacy.segments;
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.legacy.cr0, 0);
    assert_eq!(cpu.cpu.legacy.segments, caches);
    steps(&mut cpu, &vm, 1);
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x1500));
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(cpu.cpu.gpr[3], 0xcafe);
    assert_eq!(cpu.retired_instructions(), 25);
}

#[test]
fn pe_changes_retain_all_selectors_bases_limits_and_size_bits_cpu_model() {
    crate::model_api::with_cpu_model(pe_changes_retain_all_selectors_bases_limits_and_size_bits);
}

#[test]
fn pe_changes_retain_all_selectors_bases_limits_and_size_bits() {
    // SDM Vol 3A §12.9.1 step 9, §12.9.2 step 4; PRM §§14.4.1–14.5.
    for (old, new) in [(0, 1), (1, 0)] {
        for wide in [false, true] {
            let (vm, mut cpu) = setup(b"\x0f\x22\xc0\x43", wide, !wide);
            cpu.cpu.legacy.cr0 = old;
            cpu.cpu.cs = 1;
            cpu.cpu.ss = 3;
            cpu.cpu.legacy.segments[1].limit = 0x12345;
            cpu.cpu.legacy.segments[4].base = 0x3000;
            let before = cpu.cpu.legacy.segments;
            cpu.cpu.gpr[0] = new;
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.legacy.cr0, new as u32);
            assert_eq!((cpu.cpu.cs, cpu.cpu.ss), (1, 3));
            assert_eq!(cpu.cpu.legacy.segments, before);
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[3], 1);
        }
    }
    for code in [&b"\x0f\x01\xf0"[..], &b"\x0f\x01\x36\x00\x30"[..]] {
        let (vm, mut cpu) = setup(code, false, false);
        cpu.cpu.legacy.cr0 = 0;
        cpu.cpu.gpr[0] = 1;
        vm.write_bytes(0x3000, &1u16.to_le_bytes()).unwrap();
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.legacy.cr0, 1);
    }
}

#[test]
fn mov_pop_and_far_pointer_loads_validate_and_update_descriptor_accessed_cpu_model() {
    crate::model_api::with_cpu_model(
        mov_pop_and_far_pointer_loads_validate_and_update_descriptor_accessed,
    );
}

#[test]
fn mov_pop_and_far_pointer_loads_validate_and_update_descriptor_accessed() {
    // SDM Vol 2A §3.3 LDS/LES/LFS/LGS/LSS; Vol 2B §4.3 MOV/POP;
    // Vol 3A §3.4.5.1; PRM §5.1.1 p. 5-4, §17 MOV pp. 17-109–110.
    for (modrm, segment) in [(0xc0, 2), (0xd0, 3), (0xd8, 1), (0xe0, 4), (0xe8, 5)] {
        let (vm, mut cpu) = setup(&[0x8e, modrm], false, false);
        descriptor(&vm, 16, 0x12345000, 0xabcdefff, 0xc092);
        cpu.cpu.gpr[0] = 16;
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            cpu.cpu.legacy.segments[segment],
            SegmentCache {
                base: 0x12345000,
                limit: 0xabcdefff,
                attributes: 0xc093,
                unusable: 0,
                reserved: 0
            }
        );
        assert_eq!(value(&vm, 0x2015, 1), 0x93);
    }
    for code32 in [false, true] {
        for (opcode, segment) in [
            (&b"\x07"[..], 2),
            (&b"\x17"[..], 3),
            (&b"\x1f"[..], 1),
            (&b"\x0f\xa1"[..], 4),
            (&b"\x0f\xa9"[..], 5),
        ] {
            let (vm, mut cpu) = setup(opcode, code32, !code32);
            descriptor(&vm, 16, 0x3000, 0xffff, 0x92);
            vm.write_bytes(0x8000, &16u32.to_le_bytes()).unwrap();
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.legacy.segments[segment].base, 0x3000);
            assert_eq!(cpu.cpu.gpr[4], if code32 { 0x8004 } else { 0x8002 });
        }
        for opcode in [
            &b"\xc4"[..],
            &b"\xc5"[..],
            &b"\x0f\xb2"[..],
            &b"\x0f\xb4"[..],
            &b"\x0f\xb5"[..],
        ] {
            let mut code = opcode.to_vec();
            code.extend(if code32 {
                &b"\x05\x00\x30\x00\x00"[..]
            } else {
                &b"\x06\x00\x30"[..]
            });
            let (vm, mut cpu) = setup(&code, code32, false);
            descriptor(&vm, 16, 0x12345000, 0xffff, 0x92);
            vm.write_bytes(
                0x3000,
                if code32 {
                    &b"\x78\x56\x34\x12\x10\x00"[..]
                } else {
                    &b"\x78\x56\x10\x00"[..]
                },
            )
            .unwrap();
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[0], if code32 { 0x12345678 } else { 0x5678 });
        }
    }
}

#[test]
fn segment_load_fault_codes_null_rules_and_check_order_cpu_model() {
    crate::model_api::with_cpu_model(segment_load_fault_codes_null_rules_and_check_order);
}

#[test]
fn segment_load_fault_codes_null_rules_and_check_order() {
    // SDM Vol 3A §§6.3, 6.6–6.7; Vol 2B §4.3 MOV;
    // PRM §17 MOV pp. 17-109–110: type/privilege precede presence.
    for (segment, modrm) in [(1, 0xd8), (2, 0xc0), (3, 0xd0), (4, 0xe0), (5, 0xe8)] {
        for (sel, attr, limit, expected) in [
            (
                0,
                0x92,
                0x3f,
                if segment == 3 { Some((13, 0)) } else { None },
            ),
            (
                3,
                0x92,
                0x3f,
                if segment == 3 { Some((13, 0)) } else { None },
            ),
            (16, 0x92, 0x16, Some((13, 16))),
            (16, 0x82, 0x3f, Some((13, 16))),
            (
                16,
                0x12,
                0x3f,
                Some((if segment == 3 { 12 } else { 11 }, 16)),
            ),
            (16, 0x98, 0x3f, Some((13, 16))),
            (19, 0x92, 0x3f, Some((13, 16))),
            (
                16,
                0xf2,
                0x3f,
                if segment == 3 { Some((13, 16)) } else { None },
            ),
            (
                16,
                0x9e,
                0x3f,
                if segment == 3 { Some((13, 16)) } else { None },
            ),
            (
                19,
                0x9e,
                0x3f,
                if segment == 3 { Some((13, 16)) } else { None },
            ),
        ] {
            let (vm, mut cpu) = setup(&[0x8e, modrm], false, false);
            descriptor(&vm, 16, 0x3000, 0xffff, attr);
            cpu.cpu.legacy.gdtr.limit = limit;
            cpu.cpu.gpr[0] = sel;
            if let Some((vector, code)) = expected {
                fault(&mut cpu, &vm, vector, code);
                assert_eq!(value(&vm, 0x2015, 1), u64::from(attr as u8));
            } else {
                steps(&mut cpu, &vm, 1);
                assert_eq!(
                    cpu.cpu.legacy.segments[segment].unusable,
                    u8::from(sel <= 3)
                );
            }
        }
    }
    let (vm, mut cpu) = setup(b"\x8e\xd8\xa1\x00\x30", false, false);
    cpu.cpu.gpr[0] = 3;
    steps(&mut cpu, &vm, 1);
    fault(&mut cpu, &vm, 13, 0);
    for code in [&b"\x1f"[..], &b"\xc5\x06\x00\x30"[..]] {
        let (vm, mut cpu) = setup(code, false, false);
        descriptor(&vm, 16, 0x3000, 0xffff, 0x12);
        vm.write_bytes(0x8000, &16u16.to_le_bytes()).unwrap();
        vm.write_bytes(0x3000, b"\x78\x56\x10\x00").unwrap();
        fault(&mut cpu, &vm, 11, 16);
    }
}

#[test]
fn same_selector_reload_refreshes_but_gdt_edits_do_not_refresh_hidden_cache_cpu_model() {
    crate::model_api::with_cpu_model(
        same_selector_reload_refreshes_but_gdt_edits_do_not_refresh_hidden_cache,
    );
}

#[test]
fn same_selector_reload_refreshes_but_gdt_edits_do_not_refresh_hidden_cache() {
    // SDM Vol 3A §3.4.3; PRM §5.1.1: hidden descriptor survives table edits.
    let (vm, mut cpu) = setup(
        b"\x8e\xd8\x8b\x1e\x00\x01\x8e\xd8\x8b\x1e\x00\x01",
        false,
        false,
    );
    descriptor(&vm, 16, 0x3000, 0xffff, 0x92);
    cpu.cpu.gpr[0] = 16;
    vm.write_bytes(0x3100, &0x1234u16.to_le_bytes()).unwrap();
    vm.write_bytes(0x4100, &0x5678u16.to_le_bytes()).unwrap();
    steps(&mut cpu, &vm, 1);
    descriptor(&vm, 16, 0x4000, 0xffff, 0x92);
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[3], 0x1234);
    steps(&mut cpu, &vm, 2);
    assert_eq!(cpu.cpu.gpr[3], 0x5678);
    assert_eq!(value(&vm, 0x2015, 1), 0x93);
}

#[test]
fn every_access_checks_type_full_limit_and_expand_down_bounds_cpu_model() {
    crate::model_api::with_cpu_model(every_access_checks_type_full_limit_and_expand_down_bounds);
}

#[test]
fn every_access_checks_type_full_limit_and_expand_down_bounds() {
    // SDM Vol 3A §§6.3–6.4; PRM §§6.3.1.1–6.3.1.2 Table 6-2 pp. 6-4–6-6.
    for (code, attr, offset, vector) in [
        (&b"\x89\x07"[..], 0x91, 0x3000, 13), // read-only destination
        (&b"\x83\x07\x01"[..], 0x91, 0x3000, 13), // RMW
        (&b"\x8b\x07"[..], 0x99, 0x3000, 13), // execute-only source
        (&b"\x8b\x07"[..], 0x93, 0xffff, 13),
        (&b"\x36\x8b\x07"[..], 0x93, 0xffff, 12),
    ] {
        let (vm, mut cpu) = setup(code, false, false);
        let segment = if vector == 12 { 3 } else { 1 };
        cpu.cpu.legacy.segments[segment].attributes = attr;
        cpu.cpu.gpr[3] = offset;
        fault(&mut cpu, &vm, vector, 0);
    }
    for big in [false, true] {
        for offset in [0x7fff, 0x8000, 0xfffe, 0xffff, 0x10000] {
            let (vm, mut cpu) = setup(b"\x67\x8b\x03", false, false);
            cpu.cpu.legacy.segments[1].attributes = if big { 0x4097 } else { 0x97 };
            cpu.cpu.legacy.segments[1].limit = 0x7fff;
            cpu.cpu.gpr[3] = offset;
            if offset >= 0x8000 && (big || offset <= 0xfffe) {
                steps(&mut cpu, &vm, 1);
            } else {
                fault(&mut cpu, &vm, 13, 0);
            }
        }
    }
    for (opcode, attr) in [(&b"\x50"[..], 0x97), (&b"\x58"[..], 0x97)] {
        let (vm, mut cpu) = setup(opcode, false, false);
        cpu.cpu.legacy.segments[3].attributes = attr;
        cpu.cpu.legacy.segments[3].limit = 0x7fff;
        cpu.cpu.gpr[4] = if opcode[0] == 0x50 { 0x8001 } else { 0x7fff };
        fault(&mut cpu, &vm, 12, 0);
    }
    let (vm, mut cpu) = setup(b"\xb8\x34\x12", false, false);
    cpu.cpu.legacy.segments[0].limit = 0x1001;
    fault(&mut cpu, &vm, 13, 0);
}

#[test]
fn far_call_return_and_iret_support_both_transfer_and_stack_sizes_cpu_model() {
    crate::model_api::with_cpu_model(
        far_call_return_and_iret_support_both_transfer_and_stack_sizes,
    );
}

#[test]
fn far_call_return_and_iret_support_both_transfer_and_stack_sizes() {
    // SDM Vol 3A §6.8; Vol 2A §3.3 CALL/IRET/JMP; Vol 2B §4.3 RET;
    // PRM §17 pp. 17-39, 17-76–78, 17-85, 17-142–143.
    for code32 in [false, true] {
        for stack32 in [false, true] {
            for operand32 in [false, true] {
                let mut code = if code32 != operand32 {
                    vec![0x66]
                } else {
                    vec![]
                };
                code.push(0x9a);
                if operand32 {
                    code.extend(0x12000u32.to_le_bytes());
                } else {
                    code.extend(0x1200u16.to_le_bytes());
                }
                code.extend(8u16.to_le_bytes());
                let (vm, mut cpu) = setup(&code, code32, stack32);
                cpu.cpu.cs = 16;
                descriptor(&vm, 16, 0, 0xffff, if code32 { 0x409a } else { 0x9a });
                descriptor(&vm, 8, 0, 0x3ffff, if code32 { 0x409a } else { 0x9a });
                let ret = if code32 != operand32 {
                    &b"\x66\xcb"[..]
                } else {
                    &b"\xcb"[..]
                };
                vm.write_bytes(if operand32 { 0x12000 } else { 0x1200 }, ret)
                    .unwrap();
                cpu.cpu.gpr[4] = 0x18000;
                cpu.cpu.legacy.segments[3].limit = 0x3ffff;
                steps(&mut cpu, &vm, 1);
                let width = if operand32 { 4 } else { 2 };
                let stack = if stack32 { 0x18000 } else { 0x8000 };
                assert_eq!(cpu.cpu.gpr[4], 0x18000 - 2 * width);
                assert_eq!(
                    value(&vm, stack - 2 * width, width as usize),
                    0x1000 + code.len() as u64
                );
                assert_eq!(value(&vm, stack - width, width as usize), 16);
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.rip, 0x1000 + code.len() as u64);
                assert_eq!(cpu.cpu.gpr[4], 0x18000);
                let mut code = if code32 != operand32 {
                    vec![0x66]
                } else {
                    vec![]
                };
                code.push(0xcf);
                vm.write_bytes(0x1000, &code).unwrap();
                cpu.cpu.rip = 0x1000;
                for (i, val) in [0x1400u32, 8, 0x10203].into_iter().enumerate() {
                    vm.write_bytes(
                        stack + i as u64 * width,
                        &val.to_le_bytes()[..width as usize],
                    )
                    .unwrap();
                }
                steps(&mut cpu, &vm, 1);
                assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (8, 0x1400));
                assert!(cpu.cpu.flags.cf && cpu.cpu.flags.if_);
                assert_eq!(cpu.cpu.gpr[4], 0x18000 + 3 * width);
                assert_eq!(
                    cpu.cpu.rflags_system & 0x10000,
                    if operand32 { 0x10000 } else { 0 }
                );
            }
        }
    }
}

#[test]
fn far_transfer_faults_leave_frame_and_context_unchanged_cpu_model() {
    crate::model_api::with_cpu_model(far_transfer_faults_leave_frame_and_context_unchanged);
}

#[test]
fn far_transfer_faults_leave_frame_and_context_unchanged() {
    // SDM Vol 3A §§6.8.1, 6.8.3–6.8.6; PRM §17 CALL/JMP/RET/IRET.
    for opcode in [0xea, 0x9a, 0xcb, 0xcf] {
        for (sel, attr, limit, ip, vector, code) in [
            (0, 0x9a, 0xffff, 0x1200, 13, 0),
            (8, 0x92, 0xffff, 0x1200, 13, 8),
            (8, 0x1a, 0xffff, 0x1200, 11, 8),
            (8, 0xfa, 0xffff, 0x1200, 13, 8),
            (8, 0x9a, 0x11ff, 0x1200, 13, 0),
        ] {
            let mut bytes = vec![opcode];
            let sel: u16 = sel;
            if matches!(opcode, 0xea | 0x9a) {
                bytes.extend((ip as u16).to_le_bytes());
                bytes.extend(sel.to_le_bytes());
            }
            let (vm, mut cpu) = setup(&bytes, false, false);
            descriptor(&vm, 8, 0, limit, attr);
            vm.write_bytes(0x8000, &(ip as u16).to_le_bytes()).unwrap();
            vm.write_bytes(0x8002, &sel.to_le_bytes()).unwrap();
            vm.write_bytes(0x8004, &0x202u16.to_le_bytes()).unwrap();
            let before = value(&vm, 0x7ffc, 8);
            fault(&mut cpu, &vm, vector, code);
            assert_eq!(value(&vm, 0x7ffc, 8), before);
            assert_eq!(value(&vm, 0x200d, 1), u64::from(attr as u8));
        }
    }
    let (vm, mut cpu) = setup(b"\x9a\x00\x12\x08\x00", false, false);
    descriptor(&vm, 8, 0, 0xffff, 0x9a);
    cpu.cpu.legacy.segments[3].limit = 0x7ffc;
    fault(&mut cpu, &vm, 12, 0);
    assert_eq!(value(&vm, 0x200d, 1), 0x9a);
}

#[test]
fn gates_outer_returns_and_nonzero_cpl_are_typed_refusals_cpu_model() {
    crate::model_api::with_cpu_model(gates_outer_returns_and_nonzero_cpl_are_typed_refusals);
}

#[test]
fn gates_outer_returns_and_nonzero_cpl_are_typed_refusals() {
    // Decision-18 §4: unsupported delivery/transitions never construct an IVT frame.
    for (bytes, operation) in [
        (
            &b"\xea\x00\x12\x08\x00"[..],
            UnsupportedSystemOperation::LegacyGateOrTaskTransfer,
        ),
        (
            &b"\xcb"[..],
            UnsupportedSystemOperation::LegacyPrivilegeTransition,
        ),
        (
            &b"\xcf"[..],
            UnsupportedSystemOperation::LegacyPrivilegeTransition,
        ),
    ] {
        let (vm, mut cpu) = setup(bytes, false, false);
        cpu.cpu.gpr[0] = 12;
        descriptor(&vm, 8, 0, 0xffff, 0x84);
        vm.write_bytes(0x8000, b"\x00\x12\x0b\x00\x02\x02").unwrap();
        refused(&mut cpu, &vm, operation);
    }
    let (vm, mut cpu) = setup(b"\x90", false, false);
    cpu.cpu.flags.if_ = true;
    cpu.inject_irq(0x20);
    assert!(cpu.irq_ready());
    let stack = value(&vm, 0x7ffa, 6);
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    let StepStop::Exit(exit @ Exit::Exception { vector: 13, .. }) = run.stop() else {
        panic!("{:?}", run.stop())
    };
    assert_eq!(
        exit.fault_metadata().unwrap().error_code(),
        ErrorCode::bits(0x103, 0xffff)
    );
    assert!(!cpu.has_pending_irq());
    assert_eq!(value(&vm, 0x7ffa, 6), stack);
    cpu.inject_irq(0x20);
    cpu.cpu.legacy.cr0 = 0;
    vm.write_bytes(0x80, b"\x00\x12\x00\x00").unwrap();
    vm.write_bytes(0x1200, b"\x90").unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.rip, 0x1201); // explicitly requeued vector uses real delivery
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.cpl = 1;
    refused(
        &mut cpu,
        &vm,
        UnsupportedSystemOperation::LegacyEntryFacility,
    );
}

#[test]
fn accessed_updates_are_atomic_and_device_updates_require_emulation_cpu_model() {
    crate::model_api::with_cpu_model(
        accessed_updates_are_atomic_and_device_updates_require_emulation,
    );
}

#[test]
fn accessed_updates_are_atomic_and_device_updates_require_emulation() {
    // SDM Vol 3A §§3.4.5.1, 11.1.2.1; PRM §5.1.1 p. 5-4:
    // A write-back follows LOCK semantics; separate MMIO acknowledgments cannot
    // implement it. Unsupported device updates preserve the destination cache.
    for accessed in [false, true] {
        let (vm, mut cpu) = setup(b"\x8e\xd8", false, false);
        descriptor(&vm, 16, 0x3000, 0xffff, if accessed { 0x93 } else { 0x92 });
        cpu.cpu.gpr[0] = 16;
        vm.watch_range(0x2015, 1);
        steps(&mut cpu, &vm, 1);
        assert_eq!(value(&vm, 0x2015, 1), 0x93);
        assert_eq!(vm.take_dirty_ranges().is_empty(), accessed);
    }

    let (mut vm, mut cpu) = setup(b"\x8e\xd8", false, false);
    vm.unmap(0, 0x40000).unwrap();
    vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x2000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.write_bytes(0x1000, b"\x8e\xd8").unwrap();
    cpu.cpu.gpr[0] = 16;
    let caches = cpu.cpu.legacy.segments;
    for (address, answer) in [(0x2010, 0x3000ffff), (0x2014, 0x00009200)] {
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::MmioRead { addr, size: 4, .. }) if *addr == address),
            "{:?}",
            run.stop()
        );
        assert_eq!(cpu.cpu.legacy.segments, caches);
        cpu.complete_mmio_read(answer);
    }
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(
        matches!(
            run.stop(),
            StepStop::Exit(Exit::UnsupportedSystemContext {
                operation: UnsupportedSystemOperation::LegacyInstruction,
                ..
            })
        ),
        "{:?}",
        run.stop()
    );
    assert_eq!(cpu.cpu.legacy.segments, caches);
    assert_eq!(cpu.cpu.ds, 0);
}

#[test]
fn protected_strings_stack_and_far_pointers_use_the_same_access_boundary_cpu_model() {
    crate::model_api::with_cpu_model(
        protected_strings_stack_and_far_pointers_use_the_same_access_boundary,
    );
}

#[test]
fn protected_strings_stack_and_far_pointers_use_the_same_access_boundary() {
    // SDM Vol 3A §§6.3–6.4; Vol 2B §4.3 MOVS/STOS/CMPS/LODS/SCAS;
    // PRM §§6.3.1.1–6.3.1.2: implicit memory operands also check cached rights.
    for (opcode, segment, attr) in [
        (0xa4, 1, 0x99),
        (0xaa, 2, 0x91),
        (0xa6, 2, 0x99),
        (0xac, 1, 0x99),
        (0xae, 2, 0x99),
    ] {
        let (vm, mut cpu) = setup(&[0xf3, opcode], false, false);
        cpu.cpu.legacy.segments[segment].attributes = attr;
        cpu.cpu.gpr[1] = 2;
        cpu.cpu.gpr[6] = 0x3000;
        cpu.cpu.gpr[7] = 0x4000;
        fault(&mut cpu, &vm, 13, 0);
    }
    for wide in [false, true] {
        let code = if wide {
            &b"\x66\xff\x2e\x00\x30"[..]
        } else {
            &b"\xff\x2e\x00\x30"[..]
        };
        let (vm, mut cpu) = setup(code, false, false);
        descriptor(&vm, 8, 0x10000, 0xffff, 0x9e); // conforming ignores selector RPL
        vm.write_bytes(
            0x3000,
            if wide {
                &b"\x00\x12\x00\x00\x0b\x00"[..]
            } else {
                &b"\x00\x12\x0b\x00"[..]
            },
        )
        .unwrap();
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (8, 0x1200));
        assert_eq!(cpu.cpu.legacy.segments[0].base, 0x10000);
        cpu.cpu.rip = 0x1000;
        cpu.cpu.legacy.segments[0].base = 0;
        cpu.cpu.legacy.segments[1].limit = if wide { 0x3004 } else { 0x3002 };
        fault(&mut cpu, &vm, 13, 0);
    }
    let (vm, mut cpu) = setup(b"\xca\x0c\x00", false, false);
    descriptor(&vm, 8, 0, 0xffff, 0x9a);
    vm.write_bytes(0x8000, b"\x00\x12\x08\x00").unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[4], 0x8010);
}

#[test]
fn iret_refuses_task_and_v86_before_changing_the_frame_cpu_model() {
    crate::model_api::with_cpu_model(iret_refuses_task_and_v86_before_changing_the_frame);
}

#[test]
fn iret_refuses_task_and_v86_before_changing_the_frame() {
    // Decision-18 §2; SDM Vol 2A §3.3 IRET; PRM §17 pp. 17-74, 17-78.
    let (vm, mut cpu) = setup(b"\xcf", true, false);
    descriptor(&vm, 8, 0, 0xffff, 0x9a);
    for (current, image, operation) in [
        (0x4000, 2, UnsupportedSystemOperation::LegacyTaskReturn),
        (
            0,
            0x20002,
            UnsupportedSystemOperation::InvalidLegacyContext(
                x86jit_core::MachineContextError::Virtual8086Unsupported,
            ),
        ),
    ] {
        let image: u32 = image;
        cpu.cpu.rflags_system = current;
        vm.write_bytes(0x8000, b"\x00\x12\x00\x00\x08\x00\x00\x00")
            .unwrap();
        vm.write_bytes(0x8008, &image.to_le_bytes()).unwrap();
        refused(&mut cpu, &vm, operation);
        assert_eq!(value(&vm, 0x200d, 1), 0x9a);
    }
}

#[test]
fn far_call_mmio_frame_retry_keeps_overwritten_descriptor_and_far_pointer_cpu_model() {
    crate::model_api::with_cpu_model(
        far_call_mmio_frame_retry_keeps_overwritten_descriptor_and_far_pointer,
    );
}

#[test]
fn far_call_mmio_frame_retry_keeps_overwritten_descriptor_and_far_pointer() {
    // SDM Vol 2A §3.3 CALL; PRM §17 CALL p. 17-39: read the target
    // before pushing the return frame. Decision-16 §3: a retry preserves inputs
    // already consumed by an instruction whose earlier transfers took effect.
    for indirect in [false, true] {
        let code = if indirect {
            &b"\xff\x1d\x08\x20\x00\x00"[..]
        } else {
            &b"\x9a\x00\x12\x00\x00\x08\x00"[..]
        };
        let (mut vm, mut cpu) = setup(code, true, true);
        vm.unmap(0, 0x40000).unwrap();
        vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x2000, 8, Prot::RW, RegionKind::Trap).unwrap();
        vm.map(0x2008, 0x3dff8, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, code).unwrap();
        cpu.cpu.cs = 16;
        cpu.cpu.gpr[4] = 0x200c;
        if indirect {
            cpu.cpu.legacy.gdtr.base = 0x3000;
            vm.write_bytes(0x3008, b"\xff\xff\x00\x00\x00\x9b\x40\x00")
                .unwrap();
            vm.write_bytes(0x2008, b"\x00\x12\x00\x00\x08\x00").unwrap();
        } else {
            descriptor(&vm, 8, 0, 0xffff, 0x409b);
        }
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::MmioWrite { addr: 0x2004, size: 4, value, .. }) if *value == 0x1000 + code.len() as u64),
            "{:?}",
            run.stop()
        );
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (16, 0x1000, 0x200c)
        );
        assert_eq!(value(&vm, 0x2008, 4), 16);
        cpu.complete_mmio_write();
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (8, 0x1200, 0x2004)
        );
        assert_eq!(cpu.retired_instructions(), 1);
    }
}

#[test]
fn far_call_mmio_frame_retry_finishes_the_consumed_instruction_after_self_modification_cpu_model() {
    crate::model_api::with_cpu_model(
        far_call_mmio_frame_retry_finishes_the_consumed_instruction_after_self_modification,
    );
}

#[test]
fn far_call_mmio_frame_retry_finishes_the_consumed_instruction_after_self_modification() {
    // SDM Vol 2A §3.3 CALL; decision-16 §3: a partial frame must resume
    // the decoded CALL, even after its first push replaced that instruction.
    for toggle in [false, true] {
        let code = b"\x9a\x00\x12\x00\x00\x08\x00";
        let (mut vm, mut cpu) = setup(code, true, true);
        vm.unmap(0, 0x40000).unwrap();
        vm.map(0xffc, 4, Prot::RW, RegionKind::Trap).unwrap();
        vm.map(0x1000, 0x3f000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, code).unwrap();
        descriptor(&vm, 8, 0, 0xffff, 0x409b);
        cpu.cpu.cs = 16;
        cpu.cpu.gpr[4] = 0x1004;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0xffc,
                    value: 0x1007,
                    ..
                })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!(value(&vm, 0x1000, 4), 16);
        if toggle {
            cpu.set_a20_enabled(false);
            assert!(cpu.a20_enabled());
            assert_eq!(cpu.pending_a20_enabled(), Some(false));
        }
        cpu.complete_mmio_write();
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (8, 0x1200, 0xffc)
        );
        assert_eq!(cpu.a20_enabled(), !toggle);
        assert_eq!(cpu.pending_a20_enabled(), None);
        assert_eq!(cpu.retired_instructions(), 1);
    }
}

#[test]
fn far_call_mmio_frame_retry_survives_context_eviction_after_self_modification_cpu_model() {
    crate::model_api::with_cpu_model(
        far_call_mmio_frame_retry_survives_context_eviction_after_self_modification,
    );
}

#[test]
fn far_call_mmio_frame_retry_survives_context_eviction_after_self_modification() {
    // SDM Vol 2A §3.3 CALL; decision-16 §3: the consumed instruction and
    // completed transfers survive context churn while its frame write is pending.
    let code = b"\x9a\x00\x12\x00\x00\x08\x00";
    let (mut vm, mut cpu) = setup(code, true, true);
    vm.unmap(0, 0x40000).unwrap();
    vm.map(0xffc, 4, Prot::RW, RegionKind::Trap).unwrap();
    vm.map(0x1000, 0x3f000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    descriptor(&vm, 8, 0, 0xffff, 0x409b);
    cpu.cpu.cs = 16;
    cpu.cpu.gpr[4] = 0x1004;
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(matches!(
        run.stop(),
        StepStop::Exit(Exit::MmioWrite {
            addr: 0xffc,
            size: 4,
            value: 0x1007,
            ..
        })
    ));
    assert_eq!(value(&vm, 0x1000, 4), 16);
    let parts = cpu.cpu.mmio_parts;
    let parts_len = cpu.cpu.mmio_parts_len;
    assert!(parts_len > 0);
    vm.set_fetch_policy(Box::new(|_| x86jit_core::FetchVerdict::DenyNotPresent));
    let original_base = cpu.cpu.legacy.segments[1].base;
    for base in 1..=16 {
        cpu.cpu.legacy.segments[1].base = original_base + base;
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::UnmappedMemory { .. })),
            "{:?}",
            run.stop()
        );
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
            (16, 0x1000, 0x1004)
        );
        assert!(cpu.cpu.mmio_retry_active);
        assert_eq!(
            (cpu.cpu.mmio_retry_cs, cpu.cpu.mmio_parts_rip),
            (16, 0x1000)
        );
        assert_eq!(cpu.cpu.mmio_parts_len, parts_len);
        assert_eq!(cpu.cpu.mmio_parts, parts);
        assert!(!cpu.cpu.pending_mmio_write);
    }
    cpu.cpu.legacy.segments[1].base = original_base;
    vm.set_fetch_policy(Box::new(|_| x86jit_core::FetchVerdict::Allow));
    cpu.complete_mmio_write();
    steps(&mut cpu, &vm, 1);
    assert_eq!(
        (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
        (8, 0x1200, 0xffc)
    );
    assert_eq!(cpu.retired_instructions(), 1);
    assert!(!cpu.cpu.mmio_retry_active);
    assert_eq!(cpu.cpu.mmio_parts_len, 0);
    assert!(!cpu.cpu.pending_mmio_write);
}

#[test]
fn big_real_reload_preserves_descriptor_limits_attributes_and_full_effective_offsets_cpu_model() {
    crate::model_api::with_cpu_model(
        big_real_reload_preserves_descriptor_limits_attributes_and_full_effective_offsets,
    );
}

#[test]
fn big_real_reload_preserves_descriptor_limits_attributes_and_full_effective_offsets() {
    // SDM 325462-092US Vol 3A §§2.2, 12.9.2 and Vol 3B §23.1.1;
    // 80386 PRM §§14.1, 14.5 pp. 14-1, 14-4 (scan checked): ordinary
    // real limits are 64 KiB. Neither explicitly specifies nonstandard limits
    // after a later real reload. Witness the accepted big-real contract;
    // AMD64 APM Vol 2 §4.1 explicitly preserves limit/attributes on reload.
    for (modrm, segment) in [(0xd8, 1), (0xc0, 2), (0xd0, 3), (0xe0, 4), (0xe8, 5)] {
        // Protected MOV Sreg,AX; XOR AX,AX; MOV CR0,EAX; MOV AX,123h;
        // real MOV Sreg,AX. The inherited B bit is independent of CS.D.
        let code = [
            0x8e, modrm, 0x31, 0xc0, 0x0f, 0x22, 0xc0, 0xb8, 0x23, 0x01, 0x8e, modrm,
        ];
        let (vm, mut cpu) = setup(&code, false, false);
        descriptor(&vm, 16, 0x3000, 0xffff_ffff, 0xc092);
        cpu.cpu.gpr[0] = 16;
        steps(&mut cpu, &vm, 1);
        let protected = cpu.cpu.legacy.segments[segment];
        assert_eq!(protected.limit, 0xffff_ffff);
        assert_eq!(protected.attributes, 0xc093);
        steps(&mut cpu, &vm, 2);
        assert_eq!(cpu.cpu.legacy.cr0, 0);
        assert_eq!(cpu.cpu.legacy.segments[segment], protected);
        steps(&mut cpu, &vm, 2);
        assert_eq!(
            cpu.cpu.legacy.segments[segment],
            SegmentCache {
                base: 0x1230,
                ..protected
            }
        );
    }
    // 67 MOV AX,[EBX]; 67 MOV [EBX],AX; REP MOVSB; PUSH AX; POP DX.
    let (vm, mut cpu) = setup(
        b"\x8e\xd8\x8e\xc0\x8e\xd0\x31\xc0\x0f\x22\xc0\xb8\x23\x01\x8e\xd8\x8e\xc0\x8e\xd0\x67\x8b\x03\x67\x89\x03\x67\xf3\xa4\x50\x5a",
        false, false);
    descriptor(&vm, 16, 0x3000, 0xffff_ffff, 0xc092);
    cpu.cpu.gpr[0] = 16;
    steps(&mut cpu, &vm, 9);
    assert_eq!(cpu.cpu.legacy.cr0, 0);
    for segment in [1, 2, 3] {
        assert_eq!(cpu.cpu.legacy.segments[segment].base, 0x1230);
        assert_eq!(cpu.cpu.legacy.segments[segment].limit, 0xffff_ffff);
        assert_eq!(cpu.cpu.legacy.segments[segment].attributes, 0xc093);
    }
    cpu.cpu.gpr[3] = 0x10020;
    cpu.cpu.gpr[6] = 0x10020;
    cpu.cpu.gpr[7] = 0x20020;
    cpu.cpu.gpr[1] = 2;
    cpu.cpu.gpr[4] = 0x18000;
    vm.write_bytes(0x11250, b"\x78\x56").unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[0], 0x5678);
    cpu.cpu.gpr[0] = 0xbeef;
    steps(&mut cpu, &vm, 1);
    assert_eq!(value(&vm, 0x11250, 2), 0xbeef);
    steps(&mut cpu, &vm, 2);
    assert_eq!(value(&vm, 0x21250, 2), 0xbeef);
    steps(&mut cpu, &vm, 2);
    assert_eq!(cpu.cpu.gpr[2], 0xbeef);
    assert_eq!(cpu.cpu.gpr[4], 0x18000);
    assert_eq!(value(&vm, 0x1922e, 2), 0xbeef);
}

#[test]
fn big_real_retained_limit_faults_at_the_whole_operand_without_a_fixed_64k_check_cpu_model() {
    crate::model_api::with_cpu_model(
        big_real_retained_limit_faults_at_the_whole_operand_without_a_fixed_64k_check,
    );
}

#[test]
fn big_real_retained_limit_faults_at_the_whole_operand_without_a_fixed_64k_check() {
    // Same narrow retained-limit contract as the witness above. Limit is
    // inclusive (SDM Vol 3A §3.4.5; PRM §6.3.1.2), and real faults have
    // no error code (PRM §14.6 Table 14-1).
    let (vm, mut cpu) = setup(
        b"\x8e\xd8\x31\xc0\x0f\x22\xc0\x8e\xd8\x67\x8b\x03\x67\x8b\x03",
        false,
        false,
    );
    descriptor(&vm, 16, 0x3000, 0x10001, 0x0092);
    cpu.cpu.gpr[0] = 16;
    steps(&mut cpu, &vm, 4);
    assert_eq!(cpu.cpu.legacy.segments[1].base, 0);
    assert_eq!(cpu.cpu.legacy.segments[1].limit, 0x10001);
    cpu.cpu.gpr[3] = 0x10000;
    vm.write_bytes(0x10000, b"\x78\x56").unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[0], 0x5678);
    cpu.cpu.gpr[3] = 0x10001;
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    let StepStop::Exit(exit @ Exit::Exception { vector: 13, .. }) = run.stop() else {
        panic!("{:?}", run.stop())
    };
    assert_eq!(
        exit.fault_metadata().unwrap().error_code(),
        ErrorCode::NoCode
    );
    assert_eq!(cpu.cpu.gpr[0], 0x5678);
}

#[test]
fn big_real_far_reload_preserves_cs_descriptor_limit_and_attributes_cpu_model() {
    crate::model_api::with_cpu_model(
        big_real_far_reload_preserves_cs_descriptor_limit_and_attributes,
    );
}

#[test]
fn big_real_far_reload_preserves_cs_descriptor_limit_and_attributes() {
    // SDM Vol 3A §12.9.2; PRM §14.5 p. 14-4 prescribe a far return
    // sequence but do not specify nonstandard inherited CS attributes on a later
    // real far reload. Accepted retained-cache contract; APM Vol 2 §4.1.
    let (vm, mut cpu) = setup(b"\xea\x00\x12\x08\x00", false, false);
    descriptor(&vm, 8, 0, 0xffff_ffff, 0x809a);
    vm.write_bytes(0x1200, b"\x31\xc0\x0f\x22\xc0\xea\x00\x10\x23\x01")
        .unwrap();
    vm.write_bytes(0x2230, b"\xbb\xfe\xca").unwrap();
    steps(&mut cpu, &vm, 3);
    assert_eq!(cpu.cpu.legacy.cr0, 0);
    assert_eq!(cpu.cpu.legacy.segments[0].limit, 0xffff_ffff);
    assert_eq!(cpu.cpu.legacy.segments[0].attributes, 0x809b);
    steps(&mut cpu, &vm, 1);
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0x123, 0x1000));
    assert_eq!(
        cpu.cpu.legacy.segments[0],
        SegmentCache {
            base: 0x1230,
            limit: 0xffff_ffff,
            attributes: 0x809b,
            unusable: 0,
            reserved: 0,
        }
    );
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[3], 0xcafe);
}
