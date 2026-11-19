//! Same-CPL I386 protected segmentation (decision-18 §§1–4).
//! Authority: Intel SDM 325462-092US; 80386 PRM 230985-001 (scan checked).
#[path = "../support/model_api.rs"]
mod model_api;
#[path = "../support/port_block.rs"]
mod port_block;
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
    model_api::select(&mut vm, MODEL);
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
    model_api::with_cpu_model(
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
    model_api::with_cpu_model(pe_changes_retain_all_selectors_bases_limits_and_size_bits);
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
    model_api::with_cpu_model(
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
    model_api::with_cpu_model(segment_load_fault_codes_null_rules_and_check_order);
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
    model_api::with_cpu_model(
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
    model_api::with_cpu_model(every_access_checks_type_full_limit_and_expand_down_bounds);
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
    model_api::with_cpu_model(far_call_return_and_iret_support_both_transfer_and_stack_sizes);
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
    model_api::with_cpu_model(far_transfer_faults_leave_frame_and_context_unchanged);
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
fn call_gates_and_outer_returns_transfer_cpu_model() {
    model_api::with_cpu_model(call_gates_and_outer_returns_transfer);
}

#[test]
fn call_gates_and_outer_returns_transfer() {
    // A call gate is a transfer (xj-srh.9): this one names a null code
    // selector, #GP(0) (SDM Vol 2A CALL p. 3-131), and constructs no frame.
    // A RETF or IRET whose CS.RPL exceeds CPL returns to the outer ring; here
    // the frame's SS slot is null, #GP(0) (SDM Vol 2B RET p. 4-573), with
    // nothing changed.
    let (vm, mut cpu) = setup(b"\xea\x00\x12\x08\x00", false, false);
    descriptor(&vm, 8, 0, 0xffff, 0x84);
    fault(&mut cpu, &vm, 13, 0);
    for bytes in [&b"\xcb"[..], b"\xcf"] {
        let (vm, mut cpu) = setup(bytes, false, false);
        descriptor(&vm, 8, 0, 0xffff, 0xfa);
        vm.write_bytes(0x8000, b"\x00\x12\x0b\x00\x02\x02\x00\x00\x00\x00")
            .unwrap();
        fault(&mut cpu, &vm, 13, 0);
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
    vm.write_bytes(0x1200, b"\x90\x90").unwrap();
    steps(&mut cpu, &vm, 1);
    // An explicitly requeued vector uses real delivery.
    assert_eq!(cpu.cpu.rip, 0x1201);
    // An embedder state at CPL 1 executes (xj-srh.9).
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.cpl = 1;
    cpu.cpu.cs = 9;
    cpu.cpu.ss = 17;
    cpu.cpu.legacy.segments[0].attributes = 0xbb;
    cpu.cpu.legacy.segments[3].attributes = 0xb3;
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.rip, 0x1202);
}

#[test]
fn accessed_updates_are_atomic_and_device_updates_require_emulation_cpu_model() {
    model_api::with_cpu_model(accessed_updates_are_atomic_and_device_updates_require_emulation);
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
    model_api::with_cpu_model(
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
fn iret_checks_task_return_and_refuses_v86_before_changing_the_frame_cpu_model() {
    model_api::with_cpu_model(iret_checks_task_return_and_refuses_v86_before_changing_the_frame);
}

#[test]
fn iret_checks_task_return_and_refuses_v86_before_changing_the_frame() {
    // Decision-18 §2; SDM Vol 2A §3.3 IRET; PRM §17 pp. 17-74, 17-78.
    let (vm, mut cpu) = setup(b"\xcf", true, false);
    descriptor(&vm, 8, 0, 0xffff, 0x9a);
    cpu.cpu.rflags_system = 0x4000;
    fault(&mut cpu, &vm, 10, 0); // IRET requires a current TSS/back link.
    cpu.cpu.rflags_system = 0;
    vm.write_bytes(0x8000, b"\x00\x12\x00\x00\x08\x00\x00\x00")
        .unwrap();
    vm.write_bytes(0x8008, &0x20002u32.to_le_bytes()).unwrap();
    refused(
        &mut cpu,
        &vm,
        UnsupportedSystemOperation::InvalidLegacyContext(
            x86jit_core::MachineContextError::Virtual8086Unsupported,
        ),
    );
    assert_eq!(value(&vm, 0x200d, 1), 0x9a);
}

#[test]
fn far_call_mmio_frame_retry_keeps_overwritten_descriptor_and_far_pointer_cpu_model() {
    model_api::with_cpu_model(
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
    model_api::with_cpu_model(
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
    model_api::with_cpu_model(
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
    model_api::with_cpu_model(
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
    model_api::with_cpu_model(
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
    model_api::with_cpu_model(big_real_far_reload_preserves_cs_descriptor_limit_and_attributes);
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

mod stepping_contracts {
    use super::*;
    use x86jit_core::{PortDir, StepRunError};

    fn ready() -> StepOptions {
        StepOptions::default().with_stop_when_irq_ready(true)
    }

    fn string_code(addr16: bool, word: bool, segment: u8, opcode: u8) -> Vec<u8> {
        let mut code = Vec::new();
        if segment != 0 {
            code.push(segment);
        }
        code.push(0xf3);
        if addr16 {
            code.push(0x67);
        }
        if word {
            code.push(0x66);
        }
        code.extend([opcode, 0x90]);
        code
    }

    #[test]
    fn code32_rep_empty_counts_and_iteration_slices() {
        // SDM Vol 2B §4.3 REP/MOVS/CMPS/STOS/LODS/SCAS, pp. 4-563–564:
        // address size selects count/indices; empty REP touches no operand.
        // 80386 PRM §17 REP pp. 17-139–141; decision-16 §§1–3.
        for segment in [0, 0x64] {
            for addr16 in [false, true] {
                for word in [false, true] {
                    for opcode in [0xa5, 0xa7, 0xab, 0xad, 0xaf, 0x6d, 0x6f] {
                        for budget in [1, 2, 8] {
                            for count in [0, 3] {
                                if count != 0 && matches!(opcode, 0x6d | 0x6f) {
                                    continue;
                                }
                                for df in [false, true] {
                                    let code = string_code(addr16, word, segment, opcode);
                                    let (vm, mut cpu) = setup(&code, true, true);
                                    let source = if segment == 0 { 1 } else { 4 };
                                    cpu.cpu.legacy.segments[source].base = 0x10000;
                                    let high = if addr16 { 0x1234_0000 } else { 0 };
                                    cpu.cpu.gpr[1] = high | count;
                                    cpu.cpu.gpr[6] = high | 0x4000;
                                    cpu.cpu.gpr[7] = high | 0x5000;
                                    cpu.cpu.gpr[0] = 0x1111_1111;
                                    cpu.cpu.flags.df = df;
                                    vm.write_bytes(0x13ff0, &[0x11; 32]).unwrap();
                                    vm.write_bytes(0x4ff0, &[0x11; 32]).unwrap();
                                    if count == 0 {
                                        cpu.cpu.legacy.segments[source].unusable = 1;
                                        cpu.cpu.legacy.segments[2].unusable = 1;
                                    }
                                    let units = count.max(1);
                                    let mut completed = 0;
                                    while completed < units {
                                        let n = budget.min(units - completed);
                                        steps(&mut cpu, &vm, n);
                                        completed += n;
                                        assert_eq!(
                                            cpu.cpu.rip,
                                            if completed == units {
                                                0x1000 + code.len() as u64 - 1
                                            } else {
                                                0x1000
                                            }
                                        );
                                        assert_eq!(
                                            cpu.retired_instructions(),
                                            u64::from(completed == units)
                                        );
                                        assert_eq!(
                                            cpu.cpu.gpr[1],
                                            high | count.saturating_sub(completed)
                                        );
                                    }
                                    let delta = count * if word { 2 } else { 4 };
                                    let index = |base: u64| {
                                        high | if df { base - delta } else { base + delta }
                                    };
                                    assert_eq!(
                                        cpu.cpu.gpr[6],
                                        if matches!(opcode, 0xa5 | 0xa7 | 0xad) {
                                            index(0x4000)
                                        } else {
                                            high | 0x4000
                                        }
                                    );
                                    assert_eq!(
                                        cpu.cpu.gpr[7],
                                        if opcode != 0xad {
                                            index(0x5000)
                                        } else {
                                            high | 0x5000
                                        }
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn code32_rep_fault_prefix_and_mmio_retry_in_step_slices() {
        // SDM Vol 2B §4.3 REP pp. 4-563–564 and MOVS protected exceptions:
        // completed elements survive; EIP/count/indices identify the next one.
        // Vol 3A §§7.5, 7.6 (fault restart); 80386 PRM §17 REP p. 17-140.
        // Segment violations use #SS for SS and #GP for other segments.
        for addr16 in [false, true] {
            for word in [false, true] {
                for failure in 0..6 {
                    for budget in [1, 2, 8] {
                        let segment = if failure == 5 { 0x36 } else { 0x64 };
                        let code = string_code(addr16, word, segment, 0xa5);
                        let (mut vm, mut cpu) = setup(&code, true, true);
                        vm.unmap(0, 0x40000).unwrap();
                        for page in (0..0x40000).step_by(0x1000) {
                            vm.map(page, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
                        }
                        vm.write_bytes(0x1000, &code).unwrap();
                        let elem = if word { 2 } else { 4 };
                        let source = if failure == 5 { 3 } else { 4 };
                        let base = if source == 4 { 0x10000 } else { 0 };
                        cpu.cpu.legacy.segments[source].base = base;
                        let high = if addr16 { 0xabcd_0000 } else { 0 };
                        let write_fault = failure == 1 || failure == 3;
                        let si = if write_fault {
                            0x4000
                        } else {
                            0x9000 - 2 * elem
                        };
                        let di = if write_fault {
                            0xb000 - 2 * elem
                        } else {
                            0x5000
                        };
                        cpu.cpu.gpr[1] = high | 3;
                        cpu.cpu.gpr[6] = high | si;
                        cpu.cpu.gpr[7] = high | di;
                        for offset in 0..12 {
                            vm.write_bytes(u64::from(base) + si + offset, &[0x5a])
                                .unwrap();
                        }
                        let at = if write_fault {
                            0xb000
                        } else {
                            u64::from(base) + 0x9000
                        };
                        if failure < 4 {
                            vm.unmap(at, 0x1000).unwrap();
                            if failure >= 2 {
                                vm.map(at, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                            }
                        } else {
                            cpu.cpu.legacy.segments[source].limit = 0x8fff;
                        }
                        let mut completed = 0;
                        loop {
                            let run = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                            completed += run.steps();
                            if matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)) {
                                assert!(completed <= 2);
                                continue;
                            }
                            assert_eq!(completed, 2, "{failure} {run:?}");
                            match run.stop() {
                                StepStop::Exit(Exit::UnmappedMemory { addr, .. })
                                    if failure < 2 =>
                                {
                                    assert_eq!(*addr, at)
                                }
                                StepStop::Exit(Exit::MmioRead { addr, size, .. })
                                    if failure == 2 =>
                                {
                                    assert_eq!((*addr, *size), (at, elem as u8));
                                    cpu.complete_mmio_read(0x5a5a_5a5a);
                                }
                                StepStop::Exit(Exit::MmioWrite {
                                    addr, size, value, ..
                                }) if failure == 3 => {
                                    assert_eq!((*addr, *size), (at, elem as u8));
                                    assert_eq!(*value, if word { 0x5a5a } else { 0x5a5a_5a5a });
                                    cpu.complete_mmio_write();
                                }
                                StepStop::Exit(exit @ Exit::Exception { vector, .. })
                                    if failure >= 4 =>
                                {
                                    let metadata = exit.fault_metadata().unwrap();
                                    assert_eq!(metadata.saved_rip(), 0x1000);
                                    assert_eq!(metadata.class(), FaultClass::Fault);
                                    assert_eq!(*vector, if source == 3 { 12 } else { 13 })
                                }
                                other => panic!("failure={failure}, {other:?}"),
                            }
                            break;
                        }
                        assert_eq!(cpu.cpu.rip, 0x1000);
                        assert_eq!(cpu.cpu.gpr[1], high | 1);
                        assert_eq!(cpu.cpu.gpr[6], high | (si + 2 * elem));
                        assert_eq!(cpu.cpu.gpr[7], high | (di + 2 * elem));
                        assert_eq!(cpu.retired_instructions(), 0);
                        for index in 0..2 {
                            assert_eq!(
                                value(&vm, di + index * elem, elem as usize),
                                if word { 0x5a5a } else { 0x5a5a_5a5a }
                            );
                        }
                        if !write_fault {
                            assert_eq!(value(&vm, di + 2 * elem, elem as usize), 0);
                        }
                        if failure < 2 {
                            vm.map(at, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
                            if !write_fault {
                                vm.write_bytes(at, &[0x5a; 4]).unwrap();
                            }
                        } else if failure >= 4 {
                            cpu.cpu.legacy.segments[source].limit = 0xffff;
                        }
                        steps(&mut cpu, &vm, 1);
                        assert_eq!(cpu.cpu.rip, 0x1000 + code.len() as u64 - 1);
                        assert_eq!(cpu.cpu.gpr[1], high);
                        assert_eq!(cpu.retired_instructions(), 1);
                        assert!(!cpu.cpu.mmio_retry_active);
                    }
                }
            }
        }
    }

    #[test]
    fn inhibition_survives_pe_and_cs_transfers() {
        // SDM Vol 2B §4.3 STI/MOV/POP, Vol 3A §§7.8.1, 7.8.3:
        // inhibition covers the next instruction, including control transfers.
        // Transfers: Vol 2A §3.3 CALL/IRET/JMP, Vol 2B §4.3 RET.
        // PE retains caches: Vol 3A §§12.9.1–12.9.2; 80386 PRM §9.2.4
        // pp. 9-3–4, §17 STI p. 17-163, §§14.4.1–14.5 p. 14-4.
        for protected in [false, true] {
            for wide in [false, true] {
                for shadow in 0..3 {
                    for transfer in 0..5 {
                        for slicing in 0..3 {
                            let mut code = match shadow {
                                0 => vec![0xfb],
                                1 => vec![0x8e, 0xd3],
                                _ => vec![0x17],
                            };
                            let target_cs: u16 = if protected { 24 } else { 0x20 };
                            let target_ip: u32 = 0x1400;
                            match transfer {
                                0 => code.extend([0x0f, 0x22, 0xc0]),
                                1 | 2 => {
                                    code.push(if transfer == 1 { 0xea } else { 0x9a });
                                    code.extend(
                                        &target_ip.to_le_bytes()[..if wide { 4 } else { 2 }],
                                    );
                                    code.extend(target_cs.to_le_bytes());
                                }
                                3 => code.push(0xcb),
                                _ => code.push(0xcf),
                            }
                            let (vm, mut cpu) = setup(&code, wide, wide);
                            cpu.cpu.legacy.cr0 = u32::from(protected);
                            cpu.cpu.flags.if_ = shadow != 0;
                            cpu.cpu.gpr[0] = u64::from(!protected);
                            cpu.cpu.gpr[3] = if protected { 16 } else { 0 };
                            descriptor(&vm, 16, 0, 0xffff, if wide { 0x4092 } else { 0x92 });
                            descriptor(&vm, 24, 0, 0xffff, 0x9a);
                            let size = if wide { 4 } else { 2 };
                            let mut frame = Vec::new();
                            if shadow == 2 {
                                frame.extend(&cpu.cpu.gpr[3].to_le_bytes()[..size]);
                            }
                            if transfer >= 3 {
                                frame.extend(&target_ip.to_le_bytes()[..size]);
                                frame.extend(&u32::from(target_cs).to_le_bytes()[..size]);
                                if transfer == 4 {
                                    frame.extend(&0x202u32.to_le_bytes()[..size]);
                                }
                            }
                            vm.write_bytes(0x8000, &frame).unwrap();
                            if slicing == 2 {
                                steps(&mut cpu, &vm, 2);
                            } else {
                                steps(&mut cpu, &vm, 1);
                                assert!(!cpu.irq_ready());

                                let n = if slicing == 0 { 1 } else { 8 };
                                let run = cpu.run_steps(&vm, n, ready()).unwrap();
                                assert_eq!(run.steps(), 1, "shadow={shadow}, transfer={transfer}, protected={protected}, wide={wide}: {run:?}");
                                assert!(
                                    if n == 1 {
                                        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted))
                                    } else {
                                        matches!(run.stop(), StepStop::IrqReady)
                                    },
                                    "{run:?}"
                                );
                            }
                            assert!(cpu.irq_ready());
                            assert_eq!(cpu.retired_instructions(), 2);
                            if transfer == 0 {
                                assert_eq!(cpu.cpu.legacy.cr0 & 1, u32::from(!protected));
                                assert_eq!(cpu.cpu.rip, 0x1000 + code.len() as u64);
                            } else {
                                assert_eq!(
                                    (cpu.cpu.cs, cpu.cpu.rip),
                                    (target_cs, u64::from(target_ip))
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn protected_port_completions_precede_irq_readiness_once() {
        // SDM Vol 2A §3.3 IN/INS and Vol 2B §4.3 OUT/OUTS/REP:
        // operand size selects transfer width; address size selects count/indices.
        // Vol 3A §§7.8.1, 7.8.3; decision-16 §3: completion adds no unit.
        for addr16 in [false, true] {
            for size in [1, 2, 4] {
                for kind in 0..4 {
                    for shadow in [false, true] {
                        for budget in [1, 8] {
                            let mut code = if shadow { vec![0x8e, 0xd3] } else { vec![0xfb] };
                            let port_ip = 0x1000 + code.len() as u64;
                            let string = kind >= 2;
                            let input = kind % 2 == 0;
                            if string {
                                code.extend([0x64, 0xf3]);
                                if addr16 {
                                    code.push(0x67);
                                }
                            }
                            if size == 2 {
                                code.push(0x66);
                            }
                            code.push(match (kind, size == 1) {
                                (0, true) => 0xec,
                                (0, false) => 0xed,
                                (1, true) => 0xee,
                                (1, false) => 0xef,
                                (2, true) => 0x6c,
                                (2, false) => 0x6d,
                                (_, true) => 0x6e,
                                _ => 0x6f,
                            });
                            let end = 0x1000 + code.len() as u64;
                            code.push(0x90);
                            let (vm, mut cpu) = setup(&code, true, true);
                            descriptor(&vm, 16, 0, 0xffff, 0x4092);
                            cpu.cpu.flags.if_ = shadow;
                            cpu.cpu.gpr[3] = 16;
                            cpu.cpu.gpr[2] = 0x80;
                            cpu.cpu.gpr[0] = 0x8877_6655;
                            cpu.cpu.legacy.segments[4].base = 0x10000;
                            let high = if addr16 { 0xabcd_0000 } else { 0 };
                            cpu.cpu.gpr[1] = high | 2;
                            cpu.cpu.gpr[6] = high | 0x4000;
                            cpu.cpu.gpr[7] = high | 0x5000;
                            vm.write_bytes(0x14000, &[0x55; 8]).unwrap();
                            steps(&mut cpu, &vm, 1);
                            assert!(!cpu.irq_ready());
                            for iteration in 0..if string { 2 } else { 1 } {
                                let run =
                                    cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                                assert_eq!(run.steps(), 1, "{run:?}");
                                let StepStop::Exit(Exit::PortIo {
                                    port,
                                    size: actual,
                                    dir,
                                    value: port_value,
                                    ..
                                }) = run.stop()
                                else {
                                    panic!("{run:?}")
                                };
                                assert_eq!(
                                    (*port, *actual, *dir),
                                    (0x80, size, if input { PortDir::In } else { PortDir::Out })
                                );
                                if !input {
                                    assert_eq!(
                                        *port_value,
                                        if string { 0x5555_5555 } else { 0x8877_6655 }
                                            & (u64::MAX >> (64 - size * 8))
                                    );
                                }
                                if input {
                                    assert!(!cpu.irq_ready());
                                    assert!(matches!(
                                        cpu.run_steps(&vm, 1, ready()),
                                        Err(StepRunError::PendingPortIn)
                                    ));
                                    cpu.complete_port_in(0x4433_2211);
                                }
                                let rip = cpu.cpu.rip;
                                steps(&mut cpu, &vm, 0);
                                assert_eq!(cpu.cpu.rip, rip);
                                let completion = cpu.run_steps(&vm, budget, ready()).unwrap();
                                assert_eq!(completion.steps(), 0, "{completion:?}");
                                assert!(
                                    matches!(completion.stop(), StepStop::IrqReady),
                                    "{completion:?}"
                                );
                                assert_eq!(
                                    cpu.cpu.rip,
                                    if string && iteration == 0 {
                                        port_ip
                                    } else {
                                        end
                                    }
                                );
                                if string {
                                    assert_eq!(cpu.cpu.gpr[1], high | (1 - iteration));
                                    assert_eq!(
                                        cpu.cpu.gpr[if input { 7 } else { 6 }],
                                        high | (if input { 0x5000 } else { 0x4000 }
                                            + (iteration + 1) * u64::from(size))
                                    );
                                    if input {
                                        assert_eq!(
                                            value(
                                                &vm,
                                                0x5000 + iteration * u64::from(size),
                                                size as usize
                                            ),
                                            0x4433_2211 & (u64::MAX >> (64 - size * 8))
                                        );
                                    }
                                } else if input {
                                    let mask = u64::MAX >> (64 - size * 8);
                                    assert_eq!(
                                        cpu.cpu.gpr[0],
                                        (0x8877_6655 & !mask) | (0x4433_2211 & mask)
                                    );
                                }
                                let repeated = cpu.run_steps(&vm, budget, ready()).unwrap();
                                assert_eq!(repeated.steps(), 0);
                                assert!(matches!(repeated.stop(), StepStop::IrqReady));
                            }
                            assert_eq!(cpu.retired_instructions(), 2);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn code32_rep_shadow_and_device_retry_boundaries() {
        // SDM Vol 2B §4.3 REP/STI/MOV; Vol 3A §§7.8.1, 7.8.3;
        // 80386 PRM §17 REP p. 17-140: each completed REP element
        // provides an interrupt window, including in a caller's step slice.
        for shadow in [false, true] {
            for addr16 in [false, true] {
                for word in [false, true] {
                    for trap in [false, true] {
                        for count in [0, 3] {
                            for budget in [1, 2, 8] {
                                let mut code = if shadow { vec![0x8e, 0xd3] } else { vec![0xfb] };
                                let ip = 0x1000 + code.len() as u64;
                                code.extend(string_code(addr16, word, 0x64, 0xa5));
                                let end = 0x1000 + code.len() as u64 - 1;
                                let (mut vm, mut cpu) = setup(&code, true, true);
                                descriptor(&vm, 16, 0, 0xffff, 0x4092);
                                cpu.cpu.gpr[3] = 16;
                                cpu.cpu.flags.if_ = shadow;
                                cpu.cpu.gpr[1] = count;
                                cpu.cpu.gpr[6] = 0x4000;
                                cpu.cpu.gpr[7] = 0x5000;
                                if trap {
                                    vm.unmap(0, 0x40000).unwrap();
                                    vm.map(0, 0x30000, Prot::RWX, RegionKind::Ram).unwrap();
                                    vm.map(0x30000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                                    vm.map(0x31000, 0xf000, Prot::RWX, RegionKind::Ram).unwrap();
                                    vm.write_bytes(0x1000, &code).unwrap();
                                    descriptor(&vm, 16, 0, 0xffff, 0x4092);
                                    cpu.cpu.legacy.segments[4].base = 0x2c000;
                                }
                                steps(&mut cpu, &vm, 1);
                                assert!(!cpu.irq_ready());
                                let mut run = cpu.run_steps(&vm, budget, ready()).unwrap();
                                if trap && count != 0 {
                                    assert_eq!(run.steps(), 0);
                                    assert!(matches!(
                                        run.stop(),
                                        StepStop::Exit(Exit::MmioRead { .. })
                                    ));
                                    assert!(!cpu.irq_ready());
                                    cpu.complete_mmio_read(0x1234);
                                    assert!(!cpu.irq_ready());
                                    run = cpu.run_steps(&vm, budget, ready()).unwrap();
                                }
                                assert_eq!(run.steps(), 1, "{run:?}");
                                assert!(
                                    if budget == 1 {
                                        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted))
                                    } else {
                                        matches!(run.stop(), StepStop::IrqReady)
                                    },
                                    "{run:?}"
                                );
                                assert!(cpu.irq_ready());
                                assert_eq!(cpu.cpu.gpr[1], count.saturating_sub(1));
                                assert_eq!(cpu.cpu.rip, if count == 0 { end } else { ip });
                                assert_eq!(
                                    cpu.retired_instructions(),
                                    if count == 0 { 2 } else { 1 }
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn queued_irq_enters_after_shadowed_pe_or_cs_change() {
        // SDM Vol 2B §4.3 STI/MOV/POP; Vol 3A §§7.8.1, 7.8.3,
        // 7.12.1 (interrupt gates), 12.9.1–12.9.2 (PE transitions).
        // 80386 PRM §9.2.4 pp. 9-3–4: inhibit the SS-load boundary.
        for protected in [false, true] {
            for shadow in 0..3 {
                for change_pe in [false, true] {
                    let mut code = match shadow {
                        0 => vec![0xfb],
                        1 => vec![0x8e, 0xd3],
                        _ => vec![0x17],
                    };
                    if change_pe {
                        code.extend([0x0f, 0x22, 0xc0]);
                    } else {
                        code.extend([0xea, 0x00, 0x14, if protected { 24 } else { 0x20 }, 0]);
                    }
                    let expected_ip = if change_pe {
                        0x1000 + code.len() as u64
                    } else {
                        0x1400
                    };
                    let (vm, mut cpu) = setup(&code, false, false);
                    cpu.cpu.legacy.cr0 = u32::from(protected);
                    cpu.cpu.flags.if_ = shadow != 0;
                    cpu.cpu.gpr[0] = u64::from(!protected);
                    cpu.cpu.gpr[3] = if protected { 16 } else { 0 };
                    descriptor(&vm, 16, 0, 0xffff, 0x92);
                    descriptor(&vm, 24, 0, 0xffff, 0x9a);
                    vm.write_bytes(0x8000, &cpu.cpu.gpr[3].to_le_bytes()[..2])
                        .unwrap();
                    vm.write_bytes(0x1800, b"\xf4").unwrap();
                    // Both IDT formats share base zero but use disjoint entries.
                    cpu.cpu.legacy.idtr.base = 0;
                    cpu.cpu.legacy.idtr.limit = 0x3ff;
                    vm.write_bytes(0xc0, b"\x00\x18\x00\x00").unwrap();
                    vm.write_bytes(0x180, b"\x00\x18\x18\x00\x00\x86\x00\x00")
                        .unwrap();
                    steps(&mut cpu, &vm, 1);
                    assert!(!cpu.irq_ready());
                    cpu.inject_irq(0x30);
                    let run = cpu.run_steps(&vm, 8, StepOptions::default()).unwrap();
                    assert_eq!(run.steps(), 2, "{run:?}");
                    assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)), "{run:?}");
                    assert!(!cpu.has_pending_irq());
                    assert_eq!(cpu.cpu.rip, 0x1801);
                    assert_eq!(value(&vm, cpu.cpu.gpr[4], 2), expected_ip);
                    assert_eq!(
                        value(&vm, cpu.cpu.gpr[4] + 2, 2),
                        if change_pe {
                            0
                        } else if protected {
                            24
                        } else {
                            0x20
                        }
                    );
                    assert_eq!(cpu.retired_instructions(), 3);
                }
            }
        }
    }

    #[test]
    fn protected_ins_answer_survives_destination_mmio_before_irq() {
        // SDM Vol 2A §3.3 INS, Vol 2B §4.3 REP, Vol 3A §7.8.3;
        // decision-16 §3: one port handoff, retry its answered destination
        // without another IN or charged unit, then allow IRQ readiness.
        for addr16 in [false, true] {
            for word in [false, true] {
                for budget in [1, 8] {
                    let mut code = vec![0x8e, 0xd3];
                    code.extend(string_code(addr16, word, 0x64, 0x6d));
                    let (mut vm, mut cpu) = setup(&code, true, true);
                    descriptor(&vm, 16, 0, 0xffff, 0x4092);
                    vm.unmap(0, 0x40000).unwrap();
                    vm.map(0, 0x30000, Prot::RWX, RegionKind::Ram).unwrap();
                    vm.map(0x30000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                    vm.map(0x31000, 0xf000, Prot::RWX, RegionKind::Ram).unwrap();
                    vm.write_bytes(0x1000, &code).unwrap();
                    descriptor(&vm, 16, 0, 0xffff, 0x4092);
                    cpu.cpu.legacy.segments[2].base = 0x2b000;
                    cpu.cpu.flags.if_ = true;
                    cpu.cpu.gpr[3] = 16;
                    cpu.cpu.gpr[1] = 2;
                    cpu.cpu.gpr[7] = 0x5000;
                    steps(&mut cpu, &vm, 1);
                    let port = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                    assert_eq!(port.steps(), 1);
                    assert!(matches!(
                        port.stop(),
                        StepStop::Exit(Exit::PortIo {
                            dir: PortDir::In,
                            ..
                        })
                    ));
                    cpu.complete_port_in(0x4433_2211);
                    let store = cpu.run_steps(&vm, budget, ready()).unwrap();
                    assert_eq!(store.steps(), 0);
                    assert!(
                        matches!(
                            store.stop(),
                            StepStop::Exit(Exit::MmioWrite { addr: 0x30000, .. })
                        ),
                        "{store:?}"
                    );
                    assert!(!cpu.irq_ready());
                    assert_eq!((cpu.cpu.gpr[1], cpu.cpu.gpr[7]), (2, 0x5000));
                    cpu.complete_mmio_write();
                    let complete = cpu.run_steps(&vm, budget, ready()).unwrap();
                    assert_eq!(complete.steps(), 0);
                    assert!(
                        matches!(complete.stop(), StepStop::IrqReady),
                        "{complete:?}"
                    );
                    assert_eq!(
                        (cpu.cpu.gpr[1], cpu.cpu.gpr[7]),
                        (1, 0x5000 + if word { 2 } else { 4 })
                    );
                    assert_eq!(cpu.retired_instructions(), 1);
                    assert!(!cpu.cpu.mmio_retry_active);
                }
            }
        }
    }

    #[test]
    fn code32_rep_and_shadow_share_one_call() {
        // SDM Vol 2B §4.3 REP/STI/MOV; Vol 3A §§7.6, 7.8.1, 7.8.3:
        // completed REP iterations are interrupt windows. Decision-16 §§1–3:
        // caller slicing does not change instruction retirement or the window.
        for shadow in [false, true] {
            for addr16 in [false, true] {
                for word in [false, true] {
                    for count in [0, 3] {
                        for budget in [1, 2, 8] {
                            let mut code = if shadow { vec![0x8e, 0xd3] } else { vec![0xfb] };
                            let rep_ip = 0x1000 + code.len() as u64;
                            code.extend(string_code(addr16, word, 0x64, 0xa5));
                            *code.last_mut().unwrap() = 0xf4;
                            let (vm, mut cpu) = setup(&code, true, true);
                            descriptor(&vm, 16, 0, 0xffff, 0x4092);
                            cpu.cpu.gpr[3] = 16;
                            cpu.cpu.flags.if_ = shadow;
                            cpu.cpu.gpr[1] = count;
                            cpu.cpu.gpr[6] = 0x4000;
                            cpu.cpu.gpr[7] = 0x5000;
                            let mut units = 0;
                            loop {
                                let run =
                                    cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                                units += run.steps();
                                if matches!(run.stop(), StepStop::Exit(Exit::Hlt)) {
                                    break;
                                }
                                assert!(
                                    matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                                    "{run:?}"
                                );
                                if units == 1 {
                                    assert!(!cpu.irq_ready());
                                }
                                if count != 0 && units == 2 {
                                    assert!(cpu.irq_ready());
                                    assert_eq!(cpu.cpu.rip, rep_ip);
                                    assert_eq!(cpu.cpu.gpr[1], 2);
                                    assert_eq!(cpu.retired_instructions(), 1);
                                }
                            }
                            assert_eq!(units, 2 + count.max(1));
                            assert_eq!(cpu.retired_instructions(), 3);
                            assert_eq!(cpu.cpu.gpr[1], 0);
                            assert_eq!(cpu.cpu.rip, 0x1000 + code.len() as u64);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn code32_repe_repne_fault_restores_original_flags_with_committed_prefix() {
        // SDM Vol 2B §4.3 REPE/REPZ p. 4-565, REPNE/REPNZ p. 4-567:
        // a CMPS/SCAS fault restores pre-instruction EFLAGS, while completed
        // indices/count remain committed. Host step slices are not interrupts.
        for addr16 in [false, true] {
            for word in [false, true] {
                for opcode in [0xa7, 0xaf] {
                    for repe in [false, true] {
                        for budget in [1, 2, 8] {
                            let mut code = string_code(addr16, word, 0x64, opcode);
                            if !repe {
                                *code.iter_mut().find(|b| **b == 0xf3).unwrap() = 0xf2;
                            }
                            let (mut vm, mut cpu) = setup(&code, true, true);
                            let elem = if word { 2 } else { 4 };
                            let at = if opcode == 0xa7 { 0x19000 } else { 0x9000 };
                            vm.unmap(0, 0x40000).unwrap();
                            vm.map(0, at as usize, Prot::RWX, RegionKind::Ram).unwrap();
                            vm.map(
                                at + 0x1000,
                                0x40000 - at as usize - 0x1000,
                                Prot::RWX,
                                RegionKind::Ram,
                            )
                            .unwrap();
                            vm.write_bytes(0x1000, &code).unwrap();
                            cpu.cpu.legacy.segments[4].base = 0x10000;
                            cpu.cpu.gpr[0] = 0x1111_1111;
                            cpu.cpu.gpr[1] = 3;
                            cpu.cpu.gpr[6] = 0x9000 - 2 * elem;
                            cpu.cpu.gpr[7] = if opcode == 0xa7 {
                                0x5000
                            } else {
                                0x9000 - 2 * elem
                            };
                            if opcode == 0xa7 {
                                vm.write_bytes(
                                    0x10000 + cpu.cpu.gpr[6],
                                    &[0x11; 8][..(2 * elem) as usize],
                                )
                                .unwrap();
                            }
                            vm.write_bytes(
                                cpu.cpu.gpr[7],
                                &[if repe { 0x11 } else { 0x10 }; 8][..(2 * elem) as usize],
                            )
                            .unwrap();
                            cpu.cpu.flags.cf = true;
                            cpu.cpu.flags.of = true;
                            cpu.cpu.flags.zf = !repe;
                            let original = format!("{:?}", cpu.cpu.flags);
                            let mut prefix = 0;
                            loop {
                                let run =
                                    cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                                prefix += run.steps();
                                if matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)) {
                                    continue;
                                }
                                assert!(
                                    matches!(run.stop(), StepStop::Exit(Exit::UnmappedMemory { addr, .. }) if *addr == at),
                                    "{run:?}"
                                );
                                break;
                            }
                            assert_eq!(prefix, 2);
                            assert_eq!(cpu.cpu.gpr[1], 1);
                            assert_eq!(cpu.cpu.rip, 0x1000);
                            assert_eq!(cpu.retired_instructions(), 0);
                            assert_eq!(format!("{:?}", cpu.cpu.flags), original, "addr16={addr16}, word={word}, opcode={opcode:x}, repe={repe}, budget={budget}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn comparison_mmio_preserves_flag_origin_until_a_later_fault() {
        // SDM Vol 2B §4.3 REPE/REPZ p. 4-565: preserve the original
        // EFLAGS through a completed comparison and a device handoff. The
        // completed device element is committed before the following fault.
        for addr16 in [false, true] {
            for word in [false, true] {
                for opcode in [0xa7, 0xaf] {
                    for budget in [1, 8] {
                        let code = string_code(addr16, word, 0x64, opcode);
                        let (mut vm, mut cpu) = setup(&code, true, true);
                        let elem = if word { 2 } else { 4 };
                        vm.unmap(0, 0x40000).unwrap();
                        vm.map(0, 0x9000 - elem, Prot::RWX, RegionKind::Ram)
                            .unwrap();
                        vm.map((0x9000 - elem) as u64, elem, Prot::RW, RegionKind::Trap)
                            .unwrap();
                        vm.map(0xa000, 0x36000, Prot::RWX, RegionKind::Ram).unwrap();
                        vm.write_bytes(0x1000, &code).unwrap();
                        vm.write_bytes((0x9000 - 2 * elem) as u64, &[0x11; 4][..elem])
                            .unwrap();
                        vm.write_bytes(0x14000, &[0x11; 12]).unwrap();
                        cpu.cpu.legacy.segments[4].base = 0x10000;
                        cpu.cpu.gpr[0] = 0x1111_1111;
                        cpu.cpu.gpr[1] = 3;
                        cpu.cpu.gpr[6] = 0x4000;
                        cpu.cpu.gpr[7] = (0x9000 - 2 * elem) as u64;
                        cpu.cpu.flags.cf = true;
                        cpu.cpu.flags.of = true;
                        let original = format!("{:?}", cpu.cpu.flags);
                        let mut units = 0;
                        loop {
                            let run = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                            units += run.steps();
                            if matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)) {
                                continue;
                            }
                            assert!(
                                matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })),
                                "{run:?}"
                            );
                            break;
                        }
                        assert_eq!(units, 1);
                        assert_ne!(format!("{:?}", cpu.cpu.flags), original);
                        cpu.complete_mmio_read(0x1111_1111);
                        let fault = cpu.run_steps(&vm, 8, StepOptions::default()).unwrap();
                        assert_eq!(fault.steps(), 1);
                        assert!(
                            matches!(
                                fault.stop(),
                                StepStop::Exit(Exit::UnmappedMemory { addr: 0x9000, .. })
                            ),
                            "{fault:?}"
                        );
                        assert_eq!(cpu.cpu.gpr[1], 1);
                        assert_eq!(cpu.cpu.rip, 0x1000);
                        assert_eq!(cpu.retired_instructions(), 0);
                        assert_eq!(format!("{:?}", cpu.cpu.flags), original);
                        assert_eq!(cpu.cpu.mmio_parts_len, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn comparison_flag_origin_is_abandoned_for_edited_inputs() {
        // Decision-16 §3; SDM Vol 2B §4.3 REPE/REPZ p. 4-565:
        // edited inputs start a new instruction attempt, whose current flags
        // become its fault-restoration origin rather than a stale prior slice.
        for edit in 0..3 {
            let code = string_code(false, false, 0x64, 0xaf);
            let (mut vm, mut cpu) = setup(&code, true, true);
            vm.unmap(0, 0x40000).unwrap();
            vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x7000, 0x39000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, &code).unwrap();
            vm.write_bytes(0x5ffc, &[0x11; 4]).unwrap();
            cpu.cpu.gpr[0] = 0x1111_1111;
            cpu.cpu.gpr[1] = 3;
            cpu.cpu.gpr[7] = 0x5ffc;
            cpu.cpu.flags.cf = true;
            steps(&mut cpu, &vm, 1);
            match edit {
                0 => cpu.cpu.flags.of = true,
                1 => cpu.cpu.gpr[1] = 2 + 0x10000,
                _ => cpu.cpu.legacy.segments[2].limit = 0x5fff,
            }
            let current = format!("{:?}", cpu.cpu.flags);
            let fault = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(fault.steps(), 0);
            assert!(
                matches!(
                    fault.stop(),
                    StepStop::Exit(
                        Exit::UnmappedMemory { .. } | Exit::Exception { vector: 13, .. }
                    )
                ),
                "{fault:?}"
            );
            assert_eq!(format!("{:?}", cpu.cpu.flags), current);
        }
    }

    #[test]
    fn comparison_irq_restart_uses_interrupted_flags_as_new_origin() {
        // SDM Vol 3A §§7.6, 7.12.1 (trap gates preserve IF), and
        // Vol 2B §4.3 REPE p. 4-565: an interrupt starts a new instruction
        // execution. A trap gate targeting the same REP must not reuse its
        // old origin even when code, count, indices and flags still match.
        let code = string_code(false, false, 0x64, 0xaf);
        let (mut vm, mut cpu) = setup(&code, true, true);
        vm.unmap(0, 0x40000).unwrap();
        vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x7000, 0x39000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &code).unwrap();
        vm.write_bytes(0x5ffc, &[0x11; 4]).unwrap();
        descriptor(&vm, 8, 0, 0xffff, 0x409b);
        descriptor(&vm, 16, 0, 0xffff, 0x4093);
        cpu.cpu.cs = 8;
        cpu.cpu.ss = 16;
        cpu.cpu.legacy.idtr.base = 0x3000;
        cpu.cpu.legacy.idtr.limit = 0x187;
        vm.write_bytes(0x3180, b"\x00\x10\x08\x00\x00\x8f\x00\x00")
            .unwrap();
        cpu.cpu.gpr[0] = 0x1111_1111;
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[7] = 0x5ffc;
        cpu.cpu.flags.if_ = true;
        cpu.cpu.flags.cf = true;
        steps(&mut cpu, &vm, 1);
        let interrupted = format!("{:?}", cpu.cpu.flags);
        cpu.inject_irq(0x30);
        let fault = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(fault.steps(), 0);
        assert!(
            matches!(
                fault.stop(),
                StepStop::Exit(Exit::UnmappedMemory { addr: 0x6000, .. })
            ),
            "{fault:?}"
        );
        assert!(!cpu.has_pending_irq());
        assert_eq!(cpu.cpu.gpr[4], 0x7ff4);
        assert_eq!(value(&vm, 0x7ff4, 4), 0x1000);
        assert_eq!(value(&vm, 0x7ff8, 4), 8);
        assert_eq!(format!("{:?}", cpu.cpu.flags), interrupted);
        assert_eq!(cpu.retired_instructions(), 0);
    }

    #[test]
    fn catalogue_model_repeats_the_protected_stepping_contracts() {
        super::model_api::with_cpu_model(|| {
            code32_rep_empty_counts_and_iteration_slices();
            code32_rep_fault_prefix_and_mmio_retry_in_step_slices();
            inhibition_survives_pe_and_cs_transfers();
            protected_port_completions_precede_irq_readiness_once();
            code32_rep_shadow_and_device_retry_boundaries();
            code32_rep_and_shadow_share_one_call();
            code32_repe_repne_fault_restores_original_flags_with_committed_prefix();
            comparison_mmio_preserves_flag_origin_until_a_later_fault();
            comparison_flag_origin_is_abandoned_for_edited_inputs();
            comparison_irq_restart_uses_interrupted_flags_as_new_origin();
            queued_irq_enters_after_shadowed_pe_or_cs_change();
            protected_ins_answer_survives_destination_mmio_before_irq();
        });
    }
}

mod dispatch_guards {
    use super::*;

    fn protected32(code: &[u8]) -> (Vm, Vcpu) {
        let mut vm = Vm::with_backend(VmConfig::flat(0x40000), Box::new(NoBackend));
        vm.set_cpu_model(x86jit_core::CpuModel::I80386);
        vm.set_exception_delivery(x86jit_core::ExceptionDelivery::Outward);
        for page in (0..0x40000).step_by(0x1000) {
            vm.map(page, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        }
        vm.write_bytes(0x2000, code).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x2000;
        cpu.cpu.gpr[4] = 0x8000;
        cpu.cpu.legacy.cr0 = 1;
        for segment in &mut cpu.cpu.legacy.segments {
            segment.limit = 0x3ffff;
            segment.attributes |= 0x4000;
        }
        (vm, cpu)
    }

    #[test]
    fn code32_rep_one_unit_and_partial_fault_preserve_full_indices() {
        // SDM Vol 2B §4.3 REP pp. 4-563–564; decision-16: full-sized
        // count/indices retain the completed prefix on a fault.
        for stepped in [false, true] {
            let (mut vm, mut c) = protected32(b"\xf3\xa5\xf4");
            vm.write_bytes(0x15ffc, &[1, 2, 3, 4]).unwrap();
            vm.write_bytes(0x16000, &[5, 6, 7, 8]).unwrap();
            vm.unmap(0x26000, 0x1000).unwrap();
            c.cpu.gpr[1] = 0x10002;
            c.cpu.gpr[6] = 0x15ffc;
            c.cpu.gpr[7] = 0x25ffc;
            if stepped {
                let r = c.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(r.steps(), 1);
                assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
            } else {
                assert!(matches!(c.run(&vm, Some(1)), Exit::BudgetExhausted));
            }
            assert_eq!(
                (
                    c.cpu.gpr[1],
                    c.cpu.gpr[6],
                    c.cpu.gpr[7],
                    c.cpu.rip,
                    c.retired_instructions()
                ),
                (0x10001, 0x16000, 0x26000, 0x2000, 0)
            );
            let before = c.cpu.clone();
            let r = c.run_steps(&vm, 64, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 0);
            assert!(
                matches!(
                    r.stop(),
                    StepStop::Exit(Exit::UnmappedMemory { addr: 0x26000, .. })
                ),
                "{r:?}"
            );
            assert_eq!(c.cpu.gpr, before.gpr);
            assert_eq!(c.cpu.rip, before.rip);
        }
    }

    #[test]
    fn code32_rep_retry_queues_a20_and_budget_preserves_irq() {
        // SDM Vol 3A §11.7.13.4; decisions 16 and 22: owned MMIO
        // completes before applying A20 or accepting IRQ; budget has priority.
        let (mut vm, mut c) = protected32(b"\xf3\xa5\xf4");
        vm.unmap(0x26000, 0x1000).unwrap();
        vm.map(0x26000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.write_bytes(0x16000, &[1, 2, 3, 4]).unwrap();
        c.cpu.gpr[1] = 2;
        c.cpu.gpr[6] = 0x16000;
        c.cpu.gpr[7] = 0x26000;
        c.cpu.flags.if_ = true;
        let r = c.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(r.steps(), 0);
        assert!(
            matches!(
                r.stop(),
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0x26000,
                    size: 4,
                    ..
                })
            ),
            "{r:?}"
        );
        c.set_a20_enabled(false);
        c.inject_irq(0x20);
        assert_eq!(c.pending_a20_enabled(), Some(false));
        assert!(!c.irq_ready());
        c.complete_mmio_write();
        let r = c.run_steps(&vm, 0, StepOptions::default()).unwrap();
        assert_eq!(r.steps(), 0);
        assert_eq!(c.pending_a20_enabled(), Some(false));
        let r = c.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(r.steps(), 1);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)),
            "{r:?}"
        );
        assert_eq!(
            (
                c.cpu.gpr[1],
                c.cpu.gpr[6],
                c.cpu.gpr[7],
                c.cpu.rip,
                c.retired_instructions()
            ),
            (1, 0x16004, 0x26004, 0x2000, 0)
        );
        assert_eq!(c.pending_a20_enabled(), None);
        assert!(!c.a20_enabled());
        assert!(c.has_pending_irq());
        assert!(c.irq_ready());
        assert!(!c.cpu.mmio_retry_active);
    }

    #[test]
    fn code32_popfd_tf_is_seen_in_the_same_run_and_after_public_edit() {
        // SDM Vol 2B §4.3 POPF/POPFD; Vol 3B §20.3.1.4:
        // the newly loaded TF controls the following instruction boundary.
        for budget in [1, 64] {
            let (vm, mut c) = protected32(b"\x9d\x90\xf4");
            vm.write_bytes(0x8000, &0x102u32.to_le_bytes()).unwrap();
            let r = c.run_steps(&vm, budget, StepOptions::default()).unwrap();
            if budget == 1 {
                assert_eq!(r.steps(), 1);
                assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
                let r = c.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(r.steps(), 1);
                assert!(
                    matches!(r.stop(), StepStop::Exit(Exit::Exception { vector: 1, .. })),
                    "{r:?}"
                );
            } else {
                assert_eq!(r.steps(), 2);
                assert!(
                    matches!(r.stop(), StepStop::Exit(Exit::Exception { vector: 1, .. })),
                    "{r:?}"
                );
            }
            c.cpu.rflags_system &= !0x100;
            let r = c.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{r:?}");
            assert_eq!(c.retired_instructions(), 3);
        }
    }
}

mod comparison_attempts {
    use x86jit_core::{
        CpuModel, ExceptionDelivery, Exit, Prot, RegionKind, ResetOptions, StepOptions, StepStop,
        Vcpu, Vm, VmConfig,
    };

    fn setup(code: &[u8]) -> (Vm, Vcpu) {
        setup_delivery(code, ExceptionDelivery::Outward)
    }
    fn setup_delivery(code: &[u8], delivery: ExceptionDelivery) -> (Vm, Vcpu) {
        let mut vm = Vm::new(VmConfig::flat(0x40000));
        vm.set_cpu_model(CpuModel::I80386);
        vm.set_exception_delivery(delivery);
        vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x7000, 0x39000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, code).unwrap();
        vm.write_bytes(0x4000, &[0x11; 32]).unwrap();
        vm.write_bytes(0x5ffc, &[0x11; 4]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.legacy.cr0 = 1;
        cpu.cpu.legacy.segments[0].attributes |= 0x4000;
        cpu.cpu.legacy.segments[3].attributes |= 0x4000;
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[0] = 0x1111_1111;
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[6] = 0x4000;
        cpu.cpu.gpr[7] = 0x5ffc;
        cpu.cpu.gpr[4] = 0x8000;
        cpu.cpu.flags.cf = true;
        cpu.cpu.flags.of = true;
        (vm, cpu)
    }
    fn slice(cpu: &mut Vcpu, vm: &Vm, n: u64) {
        let r = cpu.run_steps(vm, n, StepOptions::default()).unwrap();
        assert_eq!(r.steps(), n, "{r:?}");
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)),
            "{r:?}"
        );
    }
    fn map_fault(cpu: &mut Vcpu, vm: &Vm) {
        let r = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
        assert_eq!(r.steps(), 0, "{r:?}");
        assert!(
            matches!(
                r.stop(),
                StepStop::Exit(Exit::UnmappedMemory { addr: 0x6000, .. })
            ),
            "{r:?}"
        );
    }

    #[test]
    fn origin_is_abandoned_after_an_observed_different_instruction_attempt() {
        // Decision-16 §3: a redirected entry abandons the old instruction attempt.
        // SDM Vol 2B §4.3 REPE p. 4-565: restore this attempt's entry flags.
        let (vm, mut cpu) = setup(b"\xf3\xaf");
        vm.write_bytes(0x1800, b"\x0f\x0b").unwrap(); // UD2, outward #UD
        slice(&mut cpu, &vm, 1);
        let restart_flags = cpu.cpu.flags;
        cpu.cpu.rip = 0x1800;
        let other = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(other.steps(), 0);
        assert!(
            matches!(
                other.stop(),
                StepStop::Exit(Exit::UnknownInstruction { .. } | Exit::Exception { vector: 6, .. })
            ),
            "{other:?}"
        );
        cpu.cpu.rip = 0x1000;
        map_fault(&mut cpu, &vm);
        assert_eq!(cpu.cpu.flags, restart_flags);
    }

    #[test]
    fn comparison_origin_is_not_reused_after_cr0_round_trip() {
        // Decision-23 §5: a public CR0 edit selects a fresh execution context.
        // SDM Vol 2B §4.3 REPE p. 4-565: a fresh attempt uses current flags.
        let mut stale = Vec::new();
        for addr16 in [false, true] {
            for word in [false, true] {
                for opcode in [0xa7, 0xaf] {
                    for repe in [false, true] {
                        for budget in [1, 2, 8] {
                            let mut code = Vec::new();
                            if addr16 {
                                code.push(0x67);
                            }
                            if word {
                                code.push(0x66);
                            }
                            code.extend([if repe { 0xf3 } else { 0xf2 }, opcode]);
                            let (vm, mut cpu) = setup(&code);
                            let elem = if word { 2 } else { 4 };
                            cpu.cpu.gpr[7] = 0x6000 - 2 * elem;
                            vm.write_bytes(0x5ff8, &[if repe { 0x11 } else { 0x10 }; 8])
                                .unwrap();
                            cpu.cpu.flags.zf = !repe;
                            slice(&mut cpu, &vm, 2);
                            let restart_flags = cpu.cpu.flags;
                            cpu.cpu.legacy.cr0 = 0;
                            let mut units = 0;
                            loop {
                                let run =
                                    cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                                units += run.steps();
                                if matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)) {
                                    continue;
                                }
                                assert!(
                                    matches!(
                                        run.stop(),
                                        StepStop::Exit(Exit::UnmappedMemory { addr: 0x6000, .. })
                                    ),
                                    "{run:?}"
                                );
                                break;
                            }
                            assert_eq!(units, 0);
                            assert_eq!(cpu.cpu.flags, restart_flags);
                            cpu.cpu.legacy.cr0 = 1;
                            map_fault(&mut cpu, &vm);
                            if cpu.cpu.flags != restart_flags {
                                stale.push((
                                    addr16,
                                    word,
                                    opcode,
                                    repe,
                                    budget,
                                    restart_flags.to_flags16(),
                                    cpu.cpu.flags.to_flags16(),
                                ));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            stale.is_empty(),
            "{} stale flag origins; first {:?}",
            stale.len(),
            stale.first()
        );
    }

    #[test]
    fn zero_probe_keeps_origin_but_reset_drops_it() {
        // Decision-16 §3: a zero budget neither executes nor consumes ownership.
        for reset in [false, true] {
            let (vm, mut cpu) = setup(b"\xf3\xaf");
            let original = cpu.cpu.flags;
            slice(&mut cpu, &vm, 1);
            let current = cpu.cpu.flags;
            slice(&mut cpu, &vm, 0);
            if reset {
                cpu.reset_execution_state(ResetOptions::default());
            }
            map_fault(&mut cpu, &vm);
            assert_eq!(cpu.cpu.flags, if reset { current } else { original });
        }
    }

    fn descriptor(vm: &Vm, selector: u16, attr: u16) {
        // SDM Vol 3A §3.4.5 Fig. 3-8: flat 64-KiB descriptor image.
        let raw = 0xffffu64 | (u64::from(attr) << 40);
        vm.write_bytes(0x2000 + u64::from(selector), &raw.to_le_bytes())
            .unwrap();
    }
    fn read32(vm: &Vm, at: u64) -> u32 {
        let mut b = [0; 4];
        vm.read_bytes(at, &mut b).unwrap();
        u32::from_le_bytes(b)
    }
    #[test]
    fn comparison_architectural_fault_frame_contains_original_flags() {
        // SDM Vol 2B §4.3 REPE pp. 4-565–566, REPNE pp. 4-567–568;
        // Vol 3A §§7.6, 7.12.1: the fault frame saves instruction-entry EFLAGS.
        for addr16 in [false, true] {
            for word in [false, true] {
                for opcode in [0xa7, 0xaf] {
                    for repe in [false, true] {
                        for budget in [1, 2, 8] {
                            let mut code = vec![];
                            if addr16 {
                                code.push(0x67);
                            }
                            if word {
                                code.push(0x66);
                            }
                            code.extend([if repe { 0xf3 } else { 0xf2 }, opcode]);
                            let (vm, mut cpu) = setup_delivery(&code, ExceptionDelivery::InGuest);
                            let elem = if word { 2 } else { 4 };
                            vm.write_bytes(0x5ff8, &[if repe { 0x11 } else { 0x10 }; 8])
                                .unwrap();
                            vm.write_bytes(0x1800, b"\xf4").unwrap();
                            descriptor(&vm, 8, 0x9b);
                            descriptor(&vm, 16, 0x93);
                            vm.write_bytes(0x3068, b"\x00\x18\x08\x00\x00\x8e\x00\x00")
                                .unwrap();
                            cpu.cpu.legacy.gdtr.base = 0x2000;
                            cpu.cpu.legacy.gdtr.limit = 0x3f;
                            cpu.cpu.legacy.idtr.base = 0x3000;
                            cpu.cpu.legacy.idtr.limit = 0x7ff;
                            cpu.cpu.cs = 8;
                            cpu.cpu.ss = 16;
                            cpu.cpu.legacy.segments[2].limit = 0x5fff;
                            cpu.cpu.gpr[7] = 0x6000 - 2 * elem;
                            cpu.cpu.flags.zf = !repe;
                            let original = cpu.cpu.flags.to_flags16();
                            let mut units = 0;
                            for _ in 0..5 {
                                let r = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                                units += r.steps();
                                if matches!(r.stop(), StepStop::Exit(Exit::Hlt)) {
                                    break;
                                }
                                assert!(
                                    matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                                    "{r:?}"
                                );
                            }
                            assert_eq!(units, 3);
                            assert_eq!(cpu.cpu.gpr[1], 1);
                            assert_eq!(cpu.cpu.gpr[4], 0x7ff0);
                            assert_eq!(read32(&vm, 0x7ff4), 0x1000);
                            assert_eq!(read32(&vm, 0x7ffc) as u16, original,
                "addr16={addr16}, word={word}, opcode={opcode:x}, repe={repe}, budget={budget}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mmio_answer_that_ends_comparison_retires_and_discards_origin() {
        // SDM Vol 2B §4.3 REPE/REPNE pp. 4-565–568: answered comparison
        // commits its flags; a stopping comparison retires the REP instruction.
        for addr16 in [false, true] {
            for word in [false, true] {
                for opcode in [0xa7, 0xaf] {
                    for repe in [false, true] {
                        for budget in [1, 2, 8] {
                            let mut code = vec![];
                            if addr16 {
                                code.push(0x67);
                            }
                            if word {
                                code.push(0x66);
                            }
                            code.extend([if repe { 0xf3 } else { 0xf2 }, opcode, 0xf4]);
                            let (mut vm, mut cpu) = setup(&code);
                            vm.map(0x6000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                            cpu.cpu.flags.zf = !repe;
                            let elem = if word { 2 } else { 4 };
                            cpu.cpu.gpr[7] = 0x6000 - elem;
                            vm.write_bytes(0x5ffc, &[if repe { 0x11 } else { 0x10 }; 4])
                                .unwrap();
                            let mut units = 0;
                            loop {
                                let r = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                                units += r.steps();
                                if matches!(r.stop(), StepStop::Exit(Exit::MmioRead { .. })) {
                                    break;
                                }
                                assert!(
                                    matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                                    "{r:?}"
                                );
                            }
                            assert_eq!(units, 1);
                            cpu.complete_mmio_read(if repe { 0x1010_1010 } else { 0x1111_1111 });
                            slice(&mut cpu, &vm, 0);
                            slice(&mut cpu, &vm, 1);
                            assert_eq!(cpu.retired_instructions(), 1);
                            assert_eq!(cpu.cpu.gpr[1], 1);
                            assert_eq!(cpu.cpu.flags.zf, !repe);
                            let current = cpu.cpu.flags;
                            cpu.cpu.rip = 0x1000;
                            cpu.cpu.gpr[7] = 0x7000;
                            cpu.cpu.legacy.segments[2].limit = 0x6fff;
                            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                            assert!(
                                matches!(
                                    r.stop(),
                                    StepStop::Exit(Exit::Exception { vector: 13, .. })
                                ),
                                "{r:?}"
                            );
                            assert_eq!(cpu.cpu.flags, current);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn origin_is_abandoned_after_invalid_smc_at_the_same_address() {
        // Decision-16 §3: changed source bytes start another instruction attempt.
        // SDM Vol 2B §4.3 SCAS: LOCK is invalid; REPE p. 4-565 restores
        // the flags at entry to the later, restored REP instruction.
        let (vm, mut cpu) = setup(b"\xf3\xaf");
        slice(&mut cpu, &vm, 1);
        let restart_flags = cpu.cpu.flags;
        vm.write_bytes(0x1000, b"\xf0\xaf").unwrap();
        let changed = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(changed.steps(), 0);
        assert!(
            matches!(
                changed.stop(),
                StepStop::Exit(Exit::UnknownInstruction { .. } | Exit::Exception { vector: 6, .. })
            ),
            "{changed:?}"
        );
        vm.write_bytes(0x1000, b"\xf3\xaf").unwrap();
        map_fault(&mut cpu, &vm);
        assert_eq!(cpu.cpu.flags, restart_flags);
    }

    #[test]
    fn protected_selection_survives_context_churn_and_pe_round_trips() {
        // Decision-20 context eviction and decision-23 §5 executor selection.
        // SDM Vol 2B §4.3 REPE p. 4-565 requires original flags in PE;
        // the legacy real-mode last-comparison behavior is deliberately frozen.
        let (vm, mut cpu) = setup(b"\xf3\xaf");
        vm.write_bytes(0x5ff8, &[0x11; 8]).unwrap();
        for salt in (0..24).chain((0..24).rev()) {
            for protected in [true, false, true] {
                cpu.cpu.legacy.cr0 = u32::from(protected);
                cpu.cpu.legacy.gdtr.base = 0x2000 + salt * 8;
                cpu.cpu.rip = 0x1000;
                cpu.cpu.gpr[1] = 3;
                cpu.cpu.gpr[7] = 0x5ff8;
                cpu.cpu.flags.cf = true;
                cpu.cpu.flags.of = true;
                cpu.cpu.flags.zf = false;
                let original = cpu.cpu.flags;
                slice(&mut cpu, &vm, 2);
                let compared = cpu.cpu.flags;
                map_fault(&mut cpu, &vm);
                assert_eq!(
                    cpu.cpu.flags,
                    if protected { original } else { compared },
                    "salt={salt}, protected={protected}"
                );
            }
        }
    }

    #[test]
    fn comparison_origin_is_not_reused_after_hidden_cs_fetch_fault() {
        // Decision-23 §5: edited hidden caches invalidate an instruction attempt.
        // SDM Vol 3A §7.6 and Vol 2B §4.3 REPE p. 4-565: a subsequent
        // attempt must not resurrect flags from before the abandoned execution.
        let (vm, mut cpu) = setup(b"\xf3\xaf");
        slice(&mut cpu, &vm, 1);
        let restart_flags = cpu.cpu.flags;
        let old_limit = cpu.cpu.legacy.segments[0].limit;
        cpu.cpu.legacy.segments[0].limit = 0x0fff;
        let fetch = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(fetch.steps(), 0);
        assert!(
            matches!(
                fetch.stop(),
                StepStop::Exit(Exit::Exception { vector: 13, .. })
            ),
            "{fetch:?}"
        );
        assert_eq!(cpu.cpu.flags, restart_flags);
        cpu.cpu.legacy.segments[0].limit = old_limit;
        map_fault(&mut cpu, &vm);
        assert_eq!(cpu.cpu.flags, restart_flags);
    }
    #[test]
    fn comparison_origin_is_not_reused_after_fetch_mapping_round_trip() {
        // Decision-16 §3 and SDM Vol 2B §4.3 REPE p. 4-565:
        // a fetch fault abandons this attempt before the mapping is restored.
        for use_run in [false, true] {
            for budget in [1, 2, 8] {
                let (mut vm, mut cpu) = setup(b"\xf3\xaf");
                slice(&mut cpu, &vm, 1);
                let restart_flags = cpu.cpu.flags;
                vm.unmap(0, 0x6000).unwrap();
                if use_run {
                    assert!(matches!(
                        cpu.run(&vm, Some(budget)),
                        Exit::UnmappedMemory { addr: 0x1000, .. }
                    ));
                } else {
                    let run = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
                    assert_eq!(run.steps(), 0);
                    assert!(matches!(
                        run.stop(),
                        StepStop::Exit(Exit::UnmappedMemory { addr: 0x1000, .. })
                    ));
                }
                assert_eq!(cpu.cpu.flags, restart_flags);
                assert_eq!(cpu.retired_instructions(), 0);
                vm.map(0, 0x6000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.write_bytes(0x1000, b"\xf3\xaf").unwrap();
                map_fault(&mut cpu, &vm);
                assert_eq!(
                    cpu.cpu.flags, restart_flags,
                    "use_run={use_run}, budget={budget}"
                );
            }
        }
    }

    #[test]
    fn comparison_origin_is_not_reused_after_entry_rejection() {
        // Decision-16 §3: an entry refusal cannot leave an old attempt live.
        // SDM Vol 2B §4.3 REPE p. 4-565 defines the later attempt's flags.
        for pending_input in [false, true] {
            let (vm, mut cpu) = setup(b"\xf3\xaf");
            slice(&mut cpu, &vm, 1);
            let restart_flags = cpu.cpu.flags;
            if pending_input {
                cpu.cpu.pending_port_in = Some(1);
                assert!(matches!(
                    cpu.run_steps(&vm, 1, StepOptions::default()),
                    Err(x86jit_core::StepRunError::PendingPortIn)
                ));
                cpu.cpu.pending_port_in = None;
            } else {
                cpu.cpu.legacy.cr0 |= 1 << 31;
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 0);
                assert!(matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnsupportedSystemContext { .. })
                ));
                cpu.cpu.legacy.cr0 &= !(1 << 31);
            }
            assert_eq!(cpu.cpu.flags, restart_flags);
            assert_eq!(cpu.retired_instructions(), 0);
            map_fault(&mut cpu, &vm);
            assert_eq!(
                cpu.cpu.flags, restart_flags,
                "pending_input={pending_input}"
            );
        }
    }

    #[test]
    fn host_stop_preserves_only_the_unchanged_comparison_attempt() {
        // Decision-16 §3 permits preemption of the same instruction attempt.
        // Observed edits abandon it; SDM Vol 2B §4.3 REPE p. 4-565 then
        // restores the new attempt's entry flags on its subsequent fault.
        for edit in 0..5 {
            for use_run in [false, true] {
                let (vm, mut cpu) = setup(b"\xf3\xaf");
                let original = cpu.cpu.flags;
                slice(&mut cpu, &vm, 1);
                let current = cpu.cpu.flags;
                let cr0 = cpu.cpu.legacy.cr0;
                let limit = cpu.cpu.legacy.segments[0].limit;
                let count = cpu.cpu.gpr[1];
                match edit {
                    0 => {}
                    1 => cpu.cpu.legacy.cr0 = 0,
                    2 => cpu.cpu.legacy.segments[0].limit = 0x0fff,
                    3 => cpu.cpu.flags.cf = !current.cf,
                    4 => cpu.cpu.gpr[1] += 1,
                    _ => unreachable!(),
                }
                cpu.exit_request().request_exit();
                if use_run {
                    assert!(matches!(cpu.run(&vm, Some(1)), Exit::Interrupted));
                } else {
                    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    assert_eq!(run.steps(), 0);
                    assert!(matches!(run.stop(), StepStop::Exit(Exit::Interrupted)));
                }
                cpu.cpu.legacy.cr0 = cr0;
                cpu.cpu.legacy.segments[0].limit = limit;
                cpu.cpu.gpr[1] = count;
                cpu.cpu.flags = current;
                map_fault(&mut cpu, &vm);
                assert_eq!(
                    cpu.cpu.flags,
                    if edit == 0 { original } else { current },
                    "edit={edit}, use_run={use_run}"
                );
            }
        }
    }
}

#[test]
fn in_and_out_end_their_block_cpu_model() {
    model_api::with_cpu_model(in_and_out_end_their_block);
}

// xj-50w.22: protected-mode IN and OUT at CPL 0 in a 16-bit code segment.
// The exit names the next instruction, and the rest of the block runs after
// the port completes.
#[test]
fn in_and_out_end_their_block() {
    port_block::check(|code| setup(code, false, false), 0x1000, false);
}
