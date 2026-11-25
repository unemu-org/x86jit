//! Segmented-class CS.D=1 code: 32-bit code in non-flat segments against the
//! Checked reference (decision-27, spec.md §17.10). The class is the one of
//! `legacy_seg16`; these witnesses cover what CS.D=1 adds: 32-bit default
//! operands and addresses beside 67h-prefixed 16-bit addressing, offsets past
//! 64 KiB, SS.B selecting ESP or SP, and expand-down B=1 upper bounds
//! (SDM Vol 3A §3.4.5 pp. 3-11–3-12; 80386 PRM §§16.1–16.2 pp. 16-1–16-3).
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::descriptor;
use x86jit_core::fault::ErrorCode;
use x86jit_core::{
    derive_legacy_context, CpuModel, ExceptionDelivery, Exit, FpuModel, Prot, RegionKind,
    SegmentClass, StepOptions, StepRun, StepStop, Vcpu, Vm, VmConfig,
};

const MODELS: [CpuModel; 2] = [CpuModel::I80386, CpuModel::I80486Dx];

fn class(cpu: &Vcpu) -> SegmentClass {
    derive_legacy_context(cpu.legacy_cpu_model().unwrap(), &cpu.cpu)
        .unwrap()
        .segment_class
}

/// CS.D=1 at `cpl`: CS 08h (base 1000h, limit FFFFh), DS 5000h limit FFh,
/// ES 6000h, SS 18h (7000h, B=1), FS 8000h and GS 9000h with limit 1FFFh;
/// ESP 1000h, code at CS:100h. `checked` clears the unused GS cache's P bit,
/// which keeps the same program on the Checked class (P is checked on a
/// descriptor load, not on a cached access: SDM Vol 3A §3.4.5).
fn setup(model: CpuModel, code: &[u8], cpl: u8, paged: bool, checked: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x20000));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1100, code).unwrap();
    let dpl = u16::from(cpl) << 5;
    let mut cpu = vm.new_vcpu();
    cpu.set_a20_enabled(true);
    cpu.cpu.rip = 0x100;
    cpu.cpu.gpr[4] = 0x1000;
    cpu.cpu.legacy.cr0 = if paged { 0x80000001 } else { 1 };
    cpu.cpu.legacy.cr3 = 0x1e000;
    cpu.cpu.legacy.cpl = cpl;
    cpu.cpu.legacy.gdtr.base = 0x3000;
    cpu.cpu.legacy.gdtr.limit = 0x7f;
    cpu.cpu.cs = 8 | u16::from(cpl);
    cpu.cpu.ds = 16 | u16::from(cpl);
    cpu.cpu.es = cpu.cpu.ds;
    cpu.cpu.ss = 24 | u16::from(cpl);
    cpu.cpu.fs = cpu.cpu.ds;
    cpu.cpu.gs = cpu.cpu.ds;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        cache.base = [0x1000, 0x5000, 0x6000, 0x7000, 0x8000, 0x9000][i];
        cache.limit = match i {
            0 => 0xffff,
            1 => 0xff,
            _ => 0x1fff,
        };
        cache.attributes = (if i == 0 { 0x9b } else { 0x93 }) | 0x4000 | dpl;
        cache.unusable = 0;
    }
    descriptor(&vm, 0x3008, 0x1000, 0xffff, 0x409b | dpl);
    descriptor(&vm, 0x3010, 0xa000, 0xff, 0x4093 | dpl);
    descriptor(&vm, 0x3018, 0x7000, 0x1fff, 0x4093 | dpl);
    // G=1 data: limit field 1 scales to 1FFFh (SDM Vol 3A §3.4.5 p. 3-11).
    descriptor(&vm, 0x3028, 0xa000, 0x1fff, 0xc093 | dpl);
    // Expand-down data, limit FFFh: B=1 (G=1, field 0) and B=0 (G=0).
    descriptor(&vm, 0x3030, 0x4000, 0xfff, 0xc097 | dpl);
    descriptor(&vm, 0x3038, 0x4000, 0xfff, 0x0097 | dpl);
    // A second 32-bit code segment for far transfers.
    descriptor(&vm, 0x3040, 0x4000, 0xfff, 0x409b | dpl);
    if paged {
        vm.write_bytes(0x1e000, &0x1f007u32.to_le_bytes()).unwrap();
        for page in 0..32u32 {
            vm.write_bytes(
                0x1f000 + u64::from(page * 4),
                &((page * 4096) | 7).to_le_bytes(),
            )
            .unwrap();
        }
    }
    if checked {
        cpu.cpu.legacy.segments[5].attributes &= !0x80;
    }
    let context = derive_legacy_context(cpu.legacy_cpu_model().unwrap(), &cpu.cpu).unwrap();
    assert_eq!(context.default_operand_bits, 32);
    assert_eq!(
        context.segment_class,
        if checked {
            SegmentClass::Checked
        } else {
            SegmentClass::Segmented
        }
    );
    (vm, cpu)
}

/// The pair of fixtures, the second on the Checked reference.
fn pair(model: CpuModel, code: &[u8], cpl: u8, paged: bool) -> [(Vm, Vcpu); 2] {
    [
        setup(model, code, cpl, paged, false),
        setup(model, code, cpl, paged, true),
    ]
}

/// Run both for `budget` steps and require identical runs, state and RAM.
fn compare(pair: &mut [(Vm, Vcpu); 2], budget: u64, what: &str) -> StepRun {
    let [(avm, a), (bvm, b)] = pair;
    let ar = a.run_steps(avm, budget, StepOptions::default()).unwrap();
    let br = b.run_steps(bvm, budget, StepOptions::default()).unwrap();
    assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what}");
    b.cpu.legacy.segments[5].attributes |= 0x80;
    assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu), "{what}");
    b.cpu.legacy.segments[5].attributes &= !0x80;
    assert_eq!(a.retired_instructions(), b.retired_instructions(), "{what}");
    let mut am = vec![0; 0x20000];
    let mut bm = vec![0; 0x20000];
    avm.read_bytes(0, &mut am).unwrap();
    bvm.read_bytes(0, &mut bm).unwrap();
    assert!(am == bm, "{what}: RAM differs");
    ar
}

fn vector(run: &StepRun) -> Option<u8> {
    match run.stop() {
        StepStop::Exit(Exit::Exception { vector, .. }) => Some(*vector),
        _ => None,
    }
}

fn error_code(run: &StepRun) -> ErrorCode {
    let StepStop::Exit(exit) = run.stop() else {
        panic!("{run:?}")
    };
    exit.fault_metadata().unwrap().error_code()
}

#[test]
fn budgets_fault_ordinals_and_segment_loads_match_checked() {
    let bodies: &[&[u8]] = &[
        b"\x40\x01\xc3\x31\xda",
        b"\xa1\x00\x00\x00\x00\x40\xa3\x04\x00\x00\x00",
        b"\x50\x5b\x66\x50\x66\x59",
        b"\x64\xa1\x00\x00\x00\x00\x65\xa3\x04\x00\x00\x00",
        b"\x67\x8b\x07\x66\x01\xc3",
        b"\x8e\xdb\xa1\x00\x00\x00\x00",
        b"\x0f\xa1\x0f\xa0\x26\x8b\x46\x08",
        b"\xf3\xa4",
        b"\xe2\xfe",
        b"\xe8\x01\x00\x00\x00\x90\xc3",
    ];
    let faults: &[&[u8]] = &[
        b"\xf4",
        b"\x0f\x0b",
        b"\xa1\xfd\x00\x00\x00",
        b"\x67\xa1\xfe\x00",
        b"\x2e\xa3\x00\x00\x00\x00",
        b"\x36\xa1\xfd\x1f\x00\x00",
        b"\x8e\xd8\xa1\x00\x00\x00\x00", // a null DS load, then an access
        b"\xfa",                         // privileged/IOPL-sensitive singleton
    ];
    for model in MODELS {
        for cpl in [0, 3] {
            for paged in [false, true] {
                for body in bodies {
                    for fault in faults {
                        for ordinal in 0..3 {
                            let mut code = body.to_vec();
                            code.extend(vec![0x42; ordinal]);
                            code.extend_from_slice(fault);
                            for budget in [0, 1, 2, 3, 5, 9] {
                                let mut fixtures = pair(model, &code, cpl, paged);
                                for (_, cpu) in &mut fixtures {
                                    cpu.cpu.gpr[1] = 2;
                                    cpu.cpu.gpr[3] = 16 | u64::from(cpl);
                                    // SS.B=0 on odd ordinals: SP, not ESP
                                    // (SDM Vol 3A §3.4.5 p. 3-11).
                                    if ordinal % 2 == 1 {
                                        cpu.cpu.legacy.segments[3].attributes &= !0x4000;
                                        cpu.cpu.gpr[4] |= 0x5678_0000;
                                    }
                                }
                                compare(
                                    &mut fixtures,
                                    budget,
                                    &format!(
                                        "{model:?} CPL{cpl} PG={paged} {code:02x?} budget={budget}"
                                    ),
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
fn whole_operand_edges_with_both_address_sizes_and_expand_down_b() {
    // SDM Vol 3A §§3.4.5 p. 3-11, 3.4.5.1, 6.3–6.4: an expand-up segment
    // admits offsets up to the inclusive limit; an expand-down one those
    // above the limit, up to FFFFh (B=0) or FFFFFFFFh (B=1). Every byte of
    // the operand is checked; #SS through SS, #GP otherwise.
    let limit = 0x1ffu32;
    for (model, paged) in [
        (CpuModel::I80386, false),
        (CpuModel::I80386, true),
        (CpuModel::I80486Dx, false),
        (CpuModel::I80486Dx, true),
    ] {
        for cpl in [0, 3] {
            for (segment, prefix) in [0x2e, 0x3e, 0x26, 0x36, 0x64, 0x65].into_iter().enumerate() {
                for (down, big) in [(false, true), (true, false), (true, true)] {
                    if segment == 0 && down {
                        continue;
                    }
                    for address32 in [false, true] {
                        let mut offsets = vec![
                            0,
                            limit - 3,
                            limit - 2,
                            limit - 1,
                            limit,
                            limit + 1,
                            0xfffc,
                            0xfffd,
                            0xfffe,
                            0xffff,
                        ];
                        if address32 {
                            offsets.extend([
                                0x1_0000,
                                0xffff_fffc,
                                0xffff_fffd,
                                0xffff_fffe,
                                0xffff_ffff,
                            ]);
                        }
                        for bytes in [1u8, 2, 4] {
                            for &offset in &offsets {
                                let mut code = vec![0x43, prefix];
                                if bytes == 2 {
                                    code.push(0x66);
                                }
                                if !address32 {
                                    code.push(0x67);
                                }
                                code.push(if bytes == 1 { 0xa0 } else { 0xa1 });
                                if address32 {
                                    code.extend_from_slice(&offset.to_le_bytes());
                                } else {
                                    code.extend_from_slice(&(offset as u16).to_le_bytes());
                                }
                                let mut fixtures = pair(model, &code, cpl, paged);
                                for (_, cpu) in &mut fixtures {
                                    let cache = &mut cpu.cpu.legacy.segments[segment];
                                    cache.limit = limit;
                                    if down {
                                        cache.attributes |= 4;
                                    }
                                    if !big {
                                        cache.attributes &= !0x4000;
                                    }
                                }
                                let what = format!(
                                    "{model:?} PG={paged} CPL{cpl} segment={segment} down={down} B={big} a32={address32} offset={offset:x} bytes={bytes}"
                                );
                                let run = compare(&mut fixtures, 2, &what);
                                let last = u64::from(offset) + u64::from(bytes) - 1;
                                let allowed = if down {
                                    offset > limit && last <= if big { 0xffff_ffff } else { 0xffff }
                                } else {
                                    last <= u64::from(limit)
                                };
                                assert_eq!(run.steps(), if allowed { 2 } else { 1 }, "{what}");
                                if !allowed {
                                    assert_eq!(
                                        vector(&run),
                                        Some(if segment == 3 { 12 } else { 13 }),
                                        "{what}"
                                    );
                                    assert_eq!(error_code(&run), ErrorCode::bits(0, 0xffff));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn effective_addresses_wrap_by_address_size_before_the_limit_check() {
    // SDM Vol 1 §3.6 p. 3-18; 80386 PRM §16.2 p. 16-2: the address-size
    // attribute (D, reversed by 67h) makes offsets 16 or 32 bits wide. An
    // operand crossing offset FFFFh is not wrapped but faults (80386 PRM
    // §14.7 item 7 p. 14-6; SDM Vol 3A §6.3).
    // (code, EBX, DS limit, allowed)
    let cases: &[(&[u8], u64, u32, bool)] = &[
        // MOV EAX,[BX+2] with BX=FFFEh: offset 0.
        (b"\x43\x67\x8b\x47\x02", 0x1233_fffd, 0xff, true),
        // MOV EAX,[EBX+2] with EBX=FFFEh: offset 10000h, past the limit.
        (b"\x43\x8b\x43\x02", 0xfffd, 0xff, false),
        (b"\x43\x8b\x43\x02", 0xfffd, 0x1_0003, true),
        // MOV EAX,[EBX+2] with EBX=FFFFFFFEh: offset 0.
        (b"\x43\x8b\x43\x02", 0xffff_fffd, 0xff, true),
        // A dword at 16-bit offset FFFEh: its last byte is at 10001h.
        (b"\x43\x67\x8b\x47\x02", 0xfffb, 0xffff, false),
        (b"\x43\x67\x8b\x47\x02", 0xfffb, 0x1_0001, true),
    ];
    for model in MODELS {
        for cpl in [0, 3] {
            for paged in [false, true] {
                for &(code, ebx, limit, allowed) in cases {
                    let mut fixtures = pair(model, code, cpl, paged);
                    for (_, cpu) in &mut fixtures {
                        cpu.cpu.gpr[3] = ebx;
                        cpu.cpu.legacy.segments[1].limit = limit;
                    }
                    let what = format!("{model:?} PG={paged} CPL{cpl} {code:02x?} ebx={ebx:x}");
                    let run = compare(&mut fixtures, 2, &what);
                    assert_eq!(run.steps(), if allowed { 2 } else { 1 }, "{what}");
                    if !allowed {
                        assert_eq!(vector(&run), Some(13), "{what}");
                    }
                }
            }
        }
    }
}

#[test]
fn stack_width_follows_ss_b_and_expand_down_stacks_fault_with_ss() {
    // SDM Vol 3A §3.4.5 p. 3-11: SS.B selects ESP or SP for implicit stack
    // accesses and, for an expand-down stack, the upper bound. PUSH and CALL
    // faults are #SS(0) and leave ESP unchanged (SDM Vol 2B PUSH, Protected
    // Mode Exceptions).
    // SDM Vol 1 §6.2.3 p. 6-3: SP stack accesses reach at most FFFFh.
    // (B, expand-down, SS limit, ESP before, final ESP, or None for #SS)
    let cases: &[(bool, bool, u32, u64, Option<u64>)] = &[
        (false, false, 0xffff, 0x1234_0004, Some(0x1234_0000)),
        (false, false, 0x1fff, 0x1234_0004, None),
        (true, false, 0xffff, 0x0000_0004, None),
        (true, false, 0x1fff, 0x0000_1008, Some(0x1004)),
        (true, true, 0xfff, 0x0000_1008, Some(0x1004)),
        (true, true, 0xfff, 0x0000_1004, None),
        (true, true, 0xfff, 0x0000_0004, None),
        (true, true, 0xfff, 0x0000_0000, Some(0xffff_fffc)),
        (false, true, 0xfff, 0x5678_0000, Some(0x5678_fffc)),
        (false, true, 0xfff, 0x5678_1004, None),
    ];
    for model in MODELS {
        for cpl in [0, 3] {
            for paged in [false, true] {
                for &(big, down, limit, before, after) in cases {
                    // INC EBX; PUSH EAX; CALL $+5 (a push); POP ECX; INC EDX.
                    let code = b"\x43\x50\xe8\x00\x00\x00\x00\x59\x42";
                    let mut fixtures = pair(model, code, cpl, paged);
                    for (_, cpu) in &mut fixtures {
                        let ss = &mut cpu.cpu.legacy.segments[3];
                        ss.limit = limit;
                        ss.base = 0x4000;
                        if !big {
                            ss.attributes &= !0x4000;
                        }
                        if down {
                            ss.attributes |= 4;
                        }
                        cpu.cpu.gpr[4] = before;
                    }
                    let what = format!(
                        "{model:?} PG={paged} CPL{cpl} B={big} down={down} limit={limit:x} esp={before:x}"
                    );
                    let run = compare(&mut fixtures, 5, &what);
                    let a = &fixtures[0].1;
                    match after {
                        Some(esp) => {
                            assert_eq!(run.steps(), 5, "{what}");
                            assert_eq!(a.cpu.gpr[4], esp, "{what}");
                            assert_eq!(a.cpu.gpr[1], 0x107, "{what}");
                        }
                        None => {
                            assert!(run.steps() < 3, "{what}");
                            assert_eq!(vector(&run), Some(12), "{what}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn granular_and_expand_down_loads_inside_a_block_check_the_new_limit() {
    // A MOV DS inside the block installs the scaled limit (SDM Vol 3A
    // §3.4.5 p. 3-11: G=1 counts 4-KByte units and does not test the low
    // twelve bits) and later operands check it, not the entry stamp.
    // (selector, offset, operand bytes, allowed)
    let cases: &[(u16, u32, u8, bool)] = &[
        (0x28, 0x1ffc, 4, true),
        (0x28, 0x1ffd, 4, false),
        (0x28, 0x1fff, 1, true),
        (0x28, 0x2000, 1, false),
        (0x30, 0x0fff, 1, false),
        (0x30, 0x1000, 4, true),
        (0x30, 0xffff_fffc, 4, true),
        (0x30, 0xffff_fffd, 4, false),
        (0x38, 0x1000, 4, true),
        (0x38, 0xfffc, 4, true),
        (0x38, 0xfffd, 4, false),
        (0x38, 0x1_0000, 1, false),
    ];
    for model in MODELS {
        for cpl in [0, 3] {
            for paged in [false, true] {
                for &(selector, offset, bytes, allowed) in cases {
                    // INC EBX; MOV DS,AX; MOV EAX/AL,[offset]; INC EDX.
                    let mut code = vec![0x43, 0x8e, 0xd8, if bytes == 1 { 0xa0 } else { 0xa1 }];
                    code.extend_from_slice(&offset.to_le_bytes());
                    code.push(0x42);
                    let mut fixtures = pair(model, &code, cpl, paged);
                    for (_, cpu) in &mut fixtures {
                        cpu.cpu.gpr[0] = u64::from(selector | u16::from(cpl));
                    }
                    let what = format!("{model:?} PG={paged} CPL{cpl} {selector:x}:{offset:x}");
                    let run = compare(&mut fixtures, 4, &what);
                    let a = &fixtures[0].1;
                    let ds = a.cpu.legacy.segments[1];
                    assert_eq!(
                        ds.limit,
                        if selector == 0x28 { 0x1fff } else { 0xfff },
                        "{what}"
                    );
                    assert_eq!(run.steps(), if allowed { 4 } else { 2 }, "{what}");
                    if !allowed {
                        assert_eq!(vector(&run), Some(13), "{what}");
                    }
                }
            }
        }
    }
}

#[test]
fn null_caches_admit_the_class_and_fault_on_access() {
    // SDM Vol 3A §3.4.2: a null selector loads DS, ES, FS or GS; a memory
    // access through it raises #GP(0). The null cache is unusable but keeps
    // the Segmented class, entered with one or loaded inside the block.
    for model in MODELS {
        for cpl in [0, 3] {
            for paged in [false, true] {
                for (code, entry_null, steps) in [
                    (&b"\x43\x64\xa1\x00\x00\x00\x00"[..], true, 1),
                    (&b"\x43\x8e\xe0\x64\xa1\x00\x00\x00\x00"[..], false, 2),
                    (&b"\x43\x8e\xe0\x65\xa1\x00\x00\x00\x00\x42"[..], false, 4),
                ] {
                    let mut fixtures = pair(model, code, cpl, paged);
                    for (_, cpu) in &mut fixtures {
                        cpu.cpu.gpr[0] = 0;
                        if entry_null {
                            cpu.cpu.fs = 0;
                            cpu.cpu.legacy.segments[4].unusable = 1;
                        }
                    }
                    assert_eq!(
                        (class(&fixtures[0].1), class(&fixtures[1].1)),
                        (SegmentClass::Segmented, SegmentClass::Checked)
                    );
                    let what = format!("{model:?} PG={paged} CPL{cpl} {code:02x?}");
                    let run = compare(&mut fixtures, 4, &what);
                    assert_eq!(run.steps(), steps, "{what}");
                    assert_eq!(fixtures[0].1.cpu.legacy.segments[4].unusable, 1);
                    if steps < 4 {
                        assert_eq!(vector(&run), Some(13), "{what}");
                        assert_eq!(error_code(&run), ErrorCode::bits(0, 0xffff));
                    }
                }
            }
        }
    }
}

#[test]
fn segment_faults_precede_page_faults_and_crossing_operands_set_cr2() {
    // SDM Vol 3A §§6.3–6.4, 5.6.1, 7.15 (#PF): segmentation precedes
    // paging; an operand crossing into a missing page faults with CR2 at
    // that page's first byte and the U/S and W/R error bits.
    for model in MODELS {
        for cpl in [0, 3] {
            for store in [false, true] {
                let code = [
                    0x43,
                    if store { 0xa3 } else { 0xa1 },
                    0xff,
                    0x0f,
                    0,
                    0,
                    0x42,
                ];
                for limit in [0xff, 0x1fff] {
                    let mut fixtures = pair(model, &code, cpl, true);
                    for (vm, cpu) in &mut fixtures {
                        cpu.cpu.legacy.segments[1].limit = limit;
                        cpu.cpu.legacy.cr2 = 0x1234;
                        // The page after DS.base + FFFh.
                        vm.write_bytes(0x1f018, &0u32.to_le_bytes()).unwrap();
                    }
                    let run = compare(&mut fixtures, 10, &format!("{model:?} {code:02x?}"));
                    assert_eq!(run.steps(), 1);
                    let a = &fixtures[0].1;
                    assert_eq!(
                        a.cpu.legacy.cr2,
                        if limit == 0xff { 0x1234 } else { 0x6000 }
                    );
                    assert_eq!(vector(&run), Some(if limit == 0xff { 13 } else { 14 }));
                    if limit != 0xff {
                        assert_eq!(
                            error_code(&run),
                            ErrorCode::bits(u32::from(store) * 2 + u32::from(cpl == 3) * 4, 7)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn near_targets_past_the_cs_limit_fault_before_any_effect() {
    // SDM Vol 3A §6.3; Vol 2A CALL/Jcc, Vol 2B RET/LOOP: a near target past
    // the CS limit raises #GP(0); RET, CALL and LOOP keep ESP and ECX.
    for model in MODELS {
        for cpl in [0, 3] {
            for paged in [false, true] {
                for big in [false, true] {
                    for code in [
                        &b"\x43\xc3"[..],
                        &b"\x43\xe8\xfa\xff\x00\x00"[..],
                        &b"\x43\xe2\x10"[..],
                        &b"\x43\x75\x10"[..],
                        &b"\x43\xe9\x00\x00\x01\x00"[..],
                    ] {
                        let mut fixtures = pair(model, code, cpl, paged);
                        for (vm, cpu) in &mut fixtures {
                            if !big {
                                cpu.cpu.legacy.segments[3].attributes &= !0x4000;
                                cpu.cpu.gpr[4] |= 0x1234_0000;
                            }
                            if matches!(code[1], 0xe2 | 0x75) {
                                cpu.cpu.legacy.segments[0].limit = 0x108;
                            }
                            cpu.cpu.gpr[1] = 2;
                            // The RET target: past 64 KiB.
                            vm.write_bytes(0x8000, &0x1_0000u32.to_le_bytes()).unwrap();
                        }
                        let what = format!("{model:?} PG={paged} CPL{cpl} B={big} {code:02x?}");
                        let run = compare(&mut fixtures, 10, &what);
                        let a = &fixtures[0].1;
                        assert_eq!(run.steps(), 1, "{what}");
                        assert_eq!(a.cpu.rip, 0x101, "{what}");
                        assert_eq!(
                            a.cpu.gpr[4],
                            if big { 0x1000 } else { 0x1234_1000 },
                            "{what}"
                        );
                        assert_eq!(a.cpu.gpr[1], 2, "{what}");
                        assert_eq!(vector(&run), Some(13), "{what}");
                    }
                }
            }
        }
    }
}

#[test]
fn smc_and_device_retries_stop_before_successors() {
    for model in MODELS {
        for paged in [false, true] {
            // INC EAX; MOV BYTE [109h],F4h rewrites the INC two bytes on.
            let code = b"\x40\xc6\x05\x09\x01\x00\x00\xf4\x40\x40\xf4";
            let mut fixtures = pair(model, code, 0, paged);
            for (_, cpu) in &mut fixtures {
                cpu.cpu.legacy.segments[1].base = 0x1000;
                cpu.cpu.legacy.segments[1].limit = 0xffff;
            }
            // INC EAX, the store, INC EAX at 108h, then the stored HLT.
            let run = compare(&mut fixtures, 20, "smc");
            assert_eq!(run.steps(), 4);
            assert_eq!(fixtures[0].1.cpu.gpr[0], 2);

            // A DS page on a device: the read and the write retry in place.
            let code = b"\x43\xa1\x00\x00\x00\x00\x40\xa3\x04\x00\x00\x00\x42\xf4";
            let mut fixtures = pair(model, code, 3, paged);
            for (vm, _) in &mut fixtures {
                let mut contents = vec![0; 0x20000];
                vm.read_bytes(0, &mut contents).unwrap();
                vm.unmap(0, 0x20000).unwrap();
                vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.map(0x6000, 0x1a000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.write_bytes(0, &contents[..0x5000]).unwrap();
                vm.write_bytes(0x6000, &contents[0x6000..]).unwrap();
                vm.map(0x5000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
            }
            for (vm, cpu) in &mut fixtures {
                let run = cpu.run_steps(vm, 20, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1);
                assert!(matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })));
                cpu.complete_mmio_read(41);
                let run = cpu.run_steps(vm, 20, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 2);
                assert!(matches!(
                    run.stop(),
                    StepStop::Exit(Exit::MmioWrite { value: 42, .. })
                ));
                cpu.complete_mmio_write();
                let run = cpu.run_steps(vm, 20, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 2);
                assert_eq!(vector(&run), Some(13));
                assert_eq!(cpu.retired_instructions(), 5);
            }
            let [(_, a), (_, b)] = &mut fixtures;
            b.cpu.legacy.segments[5].attributes |= 0x80;
            assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu));
        }
    }
}

#[test]
fn far_calls_and_call_gates_with_32_bit_frames_match_checked() {
    // SDM Vol 3A §6.8.5; Vol 2A CALL, Vol 2B RET: a direct far CALL and a
    // 32-bit call gate push a two-dword frame at the same privilege level.
    for model in MODELS {
        for cpl in [0u8, 3] {
            for paged in [false, true] {
                for gate in [false, true] {
                    for budget in 0..9 {
                        let selector = if gate { 0x48 } else { 0x40 } | cpl;
                        let code = [0x43, 0x9a, 0x80, 0, 0, 0, selector, 0, 0x42, 0xf4];
                        let mut fixtures = pair(model, &code, cpl, paged);
                        for (vm, _) in &mut fixtures {
                            // INC EAX; RETF in the second code segment.
                            vm.write_bytes(0x4080, b"\x40\xcb").unwrap();
                            // A 32-bit call gate to 40h:80h (SDM Vol 3A §6.8.3).
                            vm.write_bytes(
                                0x3048,
                                &[0x80, 0, 0x40 | cpl, 0, 0, 0x8c | (cpl << 5), 0, 0],
                            )
                            .unwrap();
                        }
                        let run = compare(
                            &mut fixtures,
                            budget,
                            &format!("{model:?} PG={paged} CPL{cpl} gate={gate} budget={budget}"),
                        );
                        let a = &fixtures[0].1;
                        assert_eq!(
                            a.cpu.gpr[4],
                            if (2..4).contains(&budget) {
                                0xff8
                            } else {
                                0x1000
                            },
                            "{run:?}"
                        );
                        assert_eq!(class(a), SegmentClass::Segmented);
                    }
                }
            }
        }
    }
}

#[test]
fn paged_segment_reloads_and_far_returns_retain_context_blocks() {
    // SDM Vol 2A CALL/MOV, Vol 2B RET; spec.md §17.10: reloads and
    // same-level far transfers reuse retained context identities and blocks.
    for model in MODELS {
        for cpl in [0, 3] {
            // MOV DS,AX; MOV ES,AX; CALL FAR 40h:20h (RETF there).
            let code = [0x8e, 0xd8, 0x8e, 0xc0, 0x9a, 0x20, 0, 0, 0, 0x40 | cpl, 0];
            let (vm, mut cpu) = setup(model, &code, cpl, true, false);
            vm.write_bytes(0x4020, b"\xcb").unwrap();
            let mut warm = None;
            for round in 0..4 {
                cpu.cpu.rip = 0x100;
                cpu.cpu.gpr[0] = 0x28 | u64::from(cpl);
                fixture::steps(&mut cpu, &vm, 4);
                assert_eq!(cpu.cpu.rip, 0x10b);
                assert_eq!(cpu.cpu.legacy.segments[1].limit, 0x1fff);
                assert_eq!(cpu.cpu.legacy.segments[2].base, 0xa000);
                assert_eq!(cpu.cpu.gpr[4], 0x1000);
                let proof = cpu.paging_cache_for_test();
                assert!(proof.1 > 0, "{proof:?}");
                if round == 1 {
                    warm = Some(proof);
                } else if round > 1 {
                    assert_eq!(Some(proof), warm);
                }
            }
        }
    }
}

#[test]
fn operand_size_16_near_transfers_truncate_eip_under_both_cs_limits() {
    // 80386 PRM §17.2 CALL p. 17-37, Jcc p. 17-81, JMP p. 17-83, RET
    // p. 17-142; SDM Vol 2A CALL p. 3-125, Jcc p. 3-503, JMP p. 3-506,
    // Vol 2B RET p. 4-570: with a 16-bit operand size (66h in CS.D=1 code)
    // the new EIP is ANDed with 0000FFFFh, and RET pops only IP.
    // INC EBX first, so each transfer sits inside a block.
    let cases: &[(&[u8], &str)] = &[
        // JMP rel16: 105h + FF00h = 10005h -> 0005h.
        (b"\x43\x66\xe9\x00\xff", "jmp"),
        // CALL rel16 to the same target; pushes IP 0105h.
        (b"\x43\x66\xe8\x00\xff", "call"),
        // JNZ rel16 (ZF=0 after INC EBX): 106h + FEFFh = 10005h -> 0005h.
        (b"\x43\x66\x0f\x85\xff\xfe", "jcc"),
        // JMP AX with EAX = 10005h: DEST AND 0000FFFFh.
        (b"\x43\x66\xff\xe0", "jmp r16"),
        // RET pops the word 0005h of the dword 00010005h.
        (b"\x43\x66\xc3", "ret"),
    ];
    for model in MODELS {
        for cpl in [0, 3] {
            for paged in [false, true] {
                for four_gib in [false, true] {
                    for &(code, what) in cases {
                        let mut fixtures = pair(model, code, cpl, paged);
                        for (vm, cpu) in &mut fixtures {
                            if four_gib {
                                let cs = &mut cpu.cpu.legacy.segments[0];
                                cs.limit = u32::MAX;
                                cs.attributes |= 0x8000;
                            }
                            cpu.cpu.gpr[0] = 0x1_0005;
                            // CS:0005h is INC EDX; CS:10005h would be INC ECX.
                            vm.write_bytes(0x1005, b"\x42\xf4").unwrap();
                            vm.write_bytes(0x11005, b"\x41\xf4").unwrap();
                            vm.write_bytes(0x8000, &0x1_0005u32.to_le_bytes()).unwrap();
                        }
                        assert_eq!(
                            (class(&fixtures[0].1), class(&fixtures[1].1)),
                            (SegmentClass::Segmented, SegmentClass::Checked)
                        );
                        let what = format!("{model:?} PG={paged} CPL{cpl} 4G={four_gib} {what}");
                        let run = compare(&mut fixtures, 3, &what);
                        let a = &fixtures[0].1;
                        assert_eq!(run.steps(), 3, "{what}");
                        assert_eq!(a.cpu.rip, 6, "{what}");
                        assert_eq!((a.cpu.gpr[2], a.cpu.gpr[1]), (1, 0), "{what}");
                        let esp = match code[2] {
                            0xe8 => 0xffe,
                            0xc3 => 0x1002,
                            _ => 0x1000,
                        };
                        assert_eq!(a.cpu.gpr[4], esp, "{what}");
                        if code[2] == 0xe8 {
                            let mut pushed = [0; 4];
                            fixtures[0].0.read_bytes(0x7ffe, &mut pushed).unwrap();
                            assert_eq!(pushed, [0x05, 0x01, 0x05, 0x00], "{what}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn far_pointer_loads_inside_a_block_check_the_whole_pointer() {
    // SDM Vol 2A LDS/LES/LFS/LGS/LSS p. 3-541, Protected Mode Exceptions:
    // #GP(0) if the memory operand is outside the DS limit. The m16:32
    // pointer is six bytes and the m16:16 one four; DS limit FFh admits the
    // last byte at FFh only (SDM Vol 3A §3.4.5 p. 3-11).
    // (opcode bytes, loaded segment)
    let loads: &[(&[u8], usize)] = &[
        (b"\xc5", 1),
        (b"\xc4", 2),
        (b"\x0f\xb4", 4),
        (b"\x0f\xb5", 5),
    ];
    for model in MODELS {
        for cpl in [0u8, 3] {
            for paged in [false, true] {
                for &(opcode, segment) in loads {
                    for operand32 in [false, true] {
                        for address32 in [false, true] {
                            let size = if operand32 { 6u32 } else { 4 };
                            for offset in [0xff - size, 0x100 - size, 0x101 - size, 0x100] {
                                let allowed = offset + size - 1 <= 0xff;
                                // INC EBX; LxS EAX,[offset]; INC EDX.
                                let mut code = vec![0x43];
                                if !operand32 {
                                    code.push(0x66);
                                }
                                if !address32 {
                                    code.push(0x67);
                                }
                                code.extend_from_slice(opcode);
                                if address32 {
                                    code.push(0x05);
                                    code.extend_from_slice(&offset.to_le_bytes());
                                } else {
                                    code.push(0x06);
                                    code.extend_from_slice(&(offset as u16).to_le_bytes());
                                }
                                code.push(0x42);
                                let mut fixtures = pair(model, &code, cpl, paged);
                                let selector = 0x10 | u16::from(cpl);
                                for (vm, _) in &mut fixtures {
                                    let at = 0x5000 + u64::from(offset);
                                    let pointer = 0x1234_5678u32;
                                    vm.write_bytes(at, &pointer.to_le_bytes()[..size as usize - 2])
                                        .unwrap();
                                    vm.write_bytes(
                                        at + u64::from(size) - 2,
                                        &selector.to_le_bytes(),
                                    )
                                    .unwrap();
                                }
                                let what = format!(
                                    "{model:?} PG={paged} CPL{cpl} {code:02x?} o32={operand32} a32={address32}"
                                );
                                let run = compare(&mut fixtures, 3, &what);
                                let a = &fixtures[0].1;
                                if allowed {
                                    assert_eq!(run.steps(), 3, "{what}");
                                    assert_eq!(
                                        a.cpu.gpr[0],
                                        if operand32 { 0x1234_5678 } else { 0x5678 },
                                        "{what}"
                                    );
                                    assert_eq!(a.cpu.legacy.segments[segment].base, 0xa000);
                                    assert_eq!(a.cpu.gpr[2], 1, "{what}");
                                } else {
                                    assert_eq!(run.steps(), 1, "{what}");
                                    assert_eq!(vector(&run), Some(13), "{what}");
                                    assert_eq!(error_code(&run), ErrorCode::bits(0, 0xffff));
                                    assert_ne!(a.cpu.legacy.segments[segment].base, 0xa000);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn x87_ten_byte_operands_check_the_whole_operand_against_the_limit() {
    // SDM Vol 2A FLD p. 3-353 and FST/FSTP p. 3-387, Protected Mode
    // Exceptions: #GP(0) if the memory operand is outside the DS limit. An
    // m80 at DS limit FFh fits at offset F6h and not at F7h.
    // (FLD m80 /5, FSTP m80 /7)
    for model in MODELS {
        for cpl in [0, 3] {
            for paged in [false, true] {
                for reg in [5u8, 7] {
                    for address32 in [false, true] {
                        for offset in [0xf6u32, 0xf7, 0xff, 0x100] {
                            let allowed = offset <= 0xf6;
                            let mut code = vec![0x43];
                            if !address32 {
                                code.push(0x67);
                            }
                            code.push(0xdb);
                            if address32 {
                                code.push(0x05 | reg << 3);
                                code.extend_from_slice(&offset.to_le_bytes());
                            } else {
                                code.push(0x06 | reg << 3);
                                code.extend_from_slice(&(offset as u16).to_le_bytes());
                            }
                            code.push(0x42);
                            let mut fixtures = pair(model, &code, cpl, paged);
                            for (vm, cpu) in &mut fixtures {
                                cpu.set_fpu_model(FpuModel::Present);
                                // 1.0 as an m80 (SDM Vol 1 §4.2.2 Table 4-3).
                                let mut one = [0u8; 10];
                                one[7] = 0x80;
                                one[8] = 0xff;
                                one[9] = 0x3f;
                                vm.write_bytes(0x5000 + u64::from(offset), &one).unwrap();
                            }
                            let what = format!(
                                "{model:?} PG={paged} CPL{cpl} {code:02x?} a32={address32}"
                            );
                            let run = compare(&mut fixtures, 3, &what);
                            if allowed {
                                assert_eq!(run.steps(), 3, "{what}");
                                assert_eq!(fixtures[0].1.cpu.gpr[2], 1, "{what}");
                            } else {
                                assert_eq!(run.steps(), 1, "{what}");
                                assert_eq!(vector(&run), Some(13), "{what}");
                                assert_eq!(error_code(&run), ErrorCode::bits(0, 0xffff));
                            }
                        }
                    }
                }
            }
        }
    }
}
