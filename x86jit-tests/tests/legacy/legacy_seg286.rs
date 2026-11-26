//! The 80286 in the Segmented class (xj-to9.36; spec.md §17.10): every run is
//! compared with the Checked lowering of the same state. Authority: Intel
//! 80286 and 80287 PRM 210498-005 ("286 PRM"), cited by printed page.
//! The Checked reference clears the P bit of the GS cache, which the 80286
//! never addresses: admission then fails and nothing else differs.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::descriptor;
use iced_x86::{Decoder, DecoderOptions};
use x86jit_core::fault::ErrorCode;
use x86jit_core::{
    derive_legacy_context, CpuModel, ExceptionDelivery, Exit, Prot, RegionKind, SegmentClass,
    StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

const MODEL: CpuModel = CpuModel::I80286;

fn class(cpu: &Vcpu) -> SegmentClass {
    derive_legacy_context(cpu.legacy_cpu_model().unwrap(), &cpu.cpu)
        .unwrap()
        .segment_class
}

/// CS base 1000h (code at physical 1100h, IP 100h), DS base A000h limit FFh,
/// SS base 7000h; the GDT at 3000h holds the same descriptors at 08h, 10h
/// and 18h. 286 PRM §6.3 Fig. 6-3 p. 6-5: 24-bit bases, 16-bit limits,
/// the access byte and a zero last word.
fn setup(code: &[u8], cpl: u8, checked: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x20000));
    vm.set_cpu_model(MODEL);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1100, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_a20_enabled(true);
    cpu.cpu.rip = 0x100;
    cpu.cpu.gpr[4] = 0x1000;
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.cpl = cpl;
    cpu.cpu.legacy.gdtr.base = 0x3000;
    cpu.cpu.legacy.gdtr.limit = 0x7f;
    cpu.cpu.cs = 8 | u16::from(cpl);
    cpu.cpu.ds = 16 | u16::from(cpl);
    cpu.cpu.es = cpu.cpu.ds;
    cpu.cpu.ss = 24 | u16::from(cpl);
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        cache.base = [0x1000, 0xa000, 0x6000, 0x7000, 0x8000, 0x9000][i];
        cache.limit = if i == 1 { 0xff } else { 0x1fff };
        cache.attributes = (if i == 0 { 0x9b } else { 0x93 }) | (u16::from(cpl) << 5);
        cache.unusable = 0;
    }
    descriptor(&vm, 0x3008, 0x1000, 0x1fff, 0x9b | (u16::from(cpl) << 5));
    descriptor(&vm, 0x3010, 0xa000, 0xff, 0x93 | (u16::from(cpl) << 5));
    descriptor(&vm, 0x3018, 0x7000, 0x1fff, 0x93 | (u16::from(cpl) << 5));
    if checked {
        cpu.cpu.legacy.segments[5].attributes &= !0x80;
    }
    assert_eq!(
        class(&cpu),
        if checked {
            SegmentClass::Checked
        } else {
            SegmentClass::Segmented
        }
    );
    (vm, cpu)
}

fn same(a: &Vcpu, b: &mut Vcpu, avm: &Vm, bvm: &Vm, what: &str) {
    b.cpu.legacy.segments[5].attributes |= 0x80;
    assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu), "{what}");
    b.cpu.legacy.segments[5].attributes &= !0x80;
    assert_eq!(a.retired_instructions(), b.retired_instructions(), "{what}");
    let mut am = vec![0; 0x20000];
    let mut bm = vec![0; 0x20000];
    avm.read_bytes(0, &mut am).unwrap();
    bvm.read_bytes(0, &mut bm).unwrap();
    assert!(am == bm, "memory differs: {what}");
}

/// Runs both and asserts equal results, state, retirement and memory.
fn compare(
    (avm, a): (&Vm, &mut Vcpu),
    (bvm, b): (&Vm, &mut Vcpu),
    budget: u64,
    what: &str,
) -> String {
    let ar = a.run_steps(avm, budget, StepOptions::default());
    let br = b.run_steps(bvm, budget, StepOptions::default());
    let (ar, br) = (format!("{ar:?}"), format!("{br:?}"));
    assert_eq!(ar, br, "{what} budget={budget}");
    same(a, b, avm, bvm, &format!("{what} budget={budget}"));
    ar
}

fn exception(cpu: &mut Vcpu, vm: &Vm, budget: u64) -> (u64, u8, ErrorCode) {
    let run = cpu.run_steps(vm, budget, StepOptions::default()).unwrap();
    let StepStop::Exit(exit @ Exit::Exception { vector, .. }) = run.stop() else {
        panic!("{run:?}")
    };
    (
        run.steps(),
        *vector,
        exit.fault_metadata().unwrap().error_code(),
    )
}

#[test]
fn protected_80286_states_select_segmented_and_others_keep_checked() {
    // 286 PRM §7.4.1 p. 7-12: DS/ES/SS hold data or readable code; §7.4.2
    // p. 7-13: CS holds an executable segment. A non-present or system cache
    // cannot be proved from the stamp and keeps Checked.
    for cpl in 0..=3 {
        let (_, cpu) = setup(b"\xf4", cpl, false);
        let model = cpu.legacy_cpu_model().unwrap();
        for (segment, attributes) in [(0, 0x93), (0, 0x1b), (1, 0x13), (3, 0x03), (1, 0x82)] {
            let mut changed = cpu.cpu.clone();
            changed.legacy.segments[segment].attributes = attributes | (u16::from(cpl) << 5);
            assert_eq!(
                derive_legacy_context(model, &changed)
                    .unwrap()
                    .segment_class,
                SegmentClass::Checked,
                "CPL{cpl} segment {segment} {attributes:02x}"
            );
        }
        // A null data cache stays admissible and faults on use.
        let mut null = cpu.cpu.clone();
        null.legacy.segments[1].unusable = 1;
        null.ds = 0;
        assert_eq!(
            derive_legacy_context(model, &null).unwrap().segment_class,
            SegmentClass::Segmented
        );
    }
    // LMSW loads only the MSW (286 PRM App. B LMSW p. B-67), so the real
    // caches remain: present, readable code and writable data — Segmented.
    let model = setup(b"\xf4", 0, false).1.legacy_cpu_model().unwrap();
    let mut real = x86jit_core::CpuState::legacy_real_entry([0x100, 0x200, 0x300, 0x400, 0, 0]);
    real.legacy.cr0 = 1;
    assert_eq!(
        derive_legacy_context(model, &real).unwrap().segment_class,
        SegmentClass::Segmented
    );
}

fn census_probes(cpl: u8) -> Vec<Vec<u8>> {
    // Every one-byte opcode and every 0F xx, defined on the 80286 or not
    // (both lowerings lift the same bytes), with register and memory ModRM
    // forms; immediates are 1 unless the operand needs a usable value.
    let modrm = |two: bool, op: u8| {
        if two {
            op <= 0x03 || (0x20..=0x27).contains(&op) || op >= 0xa3 && op != 0xc8
        } else {
            (op < 0x40 && op & 7 <= 3)
                || matches!(op, 0x62..=0x63 | 0x69 | 0x6b | 0x80..=0x8f | 0xc0..=0xc1 | 0xc4..=0xc7 | 0xd0..=0xd3 | 0xd8..=0xdf | 0xf6..=0xf7 | 0xfe..=0xff)
        }
    };
    let mut out = Vec::new();
    for two in [false, true] {
        for op in 0..=255u8 {
            if !two && op == 0x0f {
                continue;
            }
            if !two && matches!(op, 0x26 | 0x2e | 0x36 | 0x3e | 0xf0 | 0xf2 | 0xf3) {
                for next in [
                    &[0xa1, 0x10, 0][..],
                    b"\xa4",
                    b"\xa5",
                    b"\xa7",
                    b"\xab",
                    b"\xad",
                ] {
                    out.push([&[op][..], next].concat());
                }
                continue;
            }
            let forms: Vec<Vec<u8>> = if modrm(two, op) {
                (0..8)
                    .flat_map(|reg| [vec![0xc3 | reg << 3], vec![0x06 | reg << 3, 0x10, 0]])
                    .collect()
            } else {
                vec![vec![]]
            };
            for form in forms {
                let mut b = if two { vec![0x0f, op] } else { vec![op] };
                b.extend(form);
                let head = b.len();
                b.resize(15, 1);
                let insn = Decoder::with_ip(16, &b, 0x100, DecoderOptions::NONE).decode();
                b.truncate(if insn.is_invalid() { head } else { insn.len() });
                match (two, op) {
                    // CALL/JMP far 08h:0700h.
                    (false, 0x9a | 0xea) => {
                        b[1..].copy_from_slice(&[0, 7, 8 | cpl, 0]);
                    }
                    (false, 0xc8) => b[1..].copy_from_slice(&[0x10, 0, 2]),
                    (false, 0xcd) => b[1] = 0x30,
                    (false, 0xd4 | 0xd5) => b[1] = 10,
                    _ => {}
                }
                out.push(b);
            }
        }
    }
    out
}

fn census(cpl: u8) {
    let probes = census_probes(cpl);
    // Runs that retired the probe and the trailer's first two instructions.
    let mut through = 0;
    for probe in &probes {
        let mut code = probe.clone();
        code.extend_from_slice(b"\x43\x31\xda\xf4");
        for budget in [1, 3, 5] {
            let (avm, mut a) = setup(&code, cpl, false);
            let (bvm, mut b) = setup(&code, cpl, true);
            for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                // Wide segments so that the probes' operands are in range.
                for cache in &mut cpu.cpu.legacy.segments {
                    cache.limit = 0xffff;
                }
                cpu.cpu.gpr[..8].copy_from_slice(&[
                    7,
                    1,
                    0,
                    u64::from(16 | cpl),
                    0x1000,
                    0x100,
                    0x10,
                    0x20,
                ]);
                // DS:0010h holds a data selector twice (MOV/LDS/LES/POP
                // forms load DS or ES inside the block); SS:1000h a far
                // frame 08h:0700h and FLAGS; CS:0700h HLT.
                for at in [0xa010, 0xa012] {
                    vm.write_bytes(at, &u16::from(16 | cpl).to_le_bytes())
                        .unwrap();
                }
                vm.write_bytes(0x8000, &[0, 7, 8 | cpl, 0, 2, 0]).unwrap();
                vm.write_bytes(0x1700, b"\xf4").unwrap();
                cpu.cpu.legacy.segments[3].base = 0x7000;
            }
            compare(
                (&avm, &mut a),
                (&bvm, &mut b),
                budget,
                &format!("CPL{cpl} {code:02x?}"),
            );
            through += usize::from(a.retired_instructions() >= 3);
        }
    }
    eprintln!(
        "80286 Segmented census CPL{cpl}: {} forms, {through} of {} runs retired three",
        probes.len(),
        probes.len() * 3
    );
    assert!(probes.len() > 3000 && through > 2000);
}

#[test]
fn census_against_checked_cpl0() {
    census(0);
}

#[test]
fn census_against_checked_cpl3() {
    census(3);
}

#[test]
fn whole_operand_limit_edges_and_expand_down_upper_bound() {
    // 286 PRM §7.4.1 pp. 7-12–13: expand-up offsets 0..limit, expand-down
    // limit+1..FFFFh, checked for the whole operand before any reference;
    // §11.2.2 p. 11-2. App. B MOV p. B-74: #SS(0) through SS, #GP(0) else.
    for cpl in [0, 3] {
        for (segment, prefix) in [0x2e, 0x3e, 0x26, 0x36].into_iter().enumerate() {
            for down in [false, true] {
                if segment == 0 && down {
                    continue;
                }
                let limit = 0x1ff;
                for (bytes, opcode) in [(1u32, 0xa0), (2, 0xa1), (1, 0xa2), (2, 0xa3)] {
                    for offset in [0, limit - 1, limit, limit + 1, 0xfffe, 0xffff] {
                        let mut code = vec![0x43, prefix, opcode];
                        code.extend_from_slice(&(offset as u16).to_le_bytes());
                        code.push(0x42);
                        let (avm, mut a) = setup(&code, cpl, false);
                        let (bvm, mut b) = setup(&code, cpl, true);
                        for cpu in [&mut a, &mut b] {
                            for_limit(cpu, segment, limit, down);
                        }
                        let what = format!(
                            "CPL{cpl} segment={segment} down={down} {opcode:02x} offset={offset:x}"
                        );
                        compare((&avm, &mut a), (&bvm, &mut b), 3, &what);
                        let store = opcode >= 0xa2;
                        let allowed = !(segment == 0 && store)
                            && if down {
                                offset > limit && offset + bytes - 1 <= 0xffff
                            } else {
                                offset + bytes - 1 <= limit
                            };
                        assert_eq!(
                            a.retired_instructions(),
                            if allowed { 3 } else { 1 },
                            "{what}"
                        );
                        if !allowed {
                            let (avm, mut a) = setup(&code, cpl, false);
                            for_limit(&mut a, segment, limit, down);
                            let (steps, vector, error) = exception(&mut a, &avm, 3);
                            assert_eq!(steps, 1, "{what}");
                            assert_eq!(vector, if segment == 3 { 12 } else { 13 }, "{what}");
                            assert_eq!(error, ErrorCode::bits(0, 0xffff), "{what}");
                        }
                    }
                }
            }
        }
    }
    // §7.4.2 p. 7-13: an execute-only code segment cannot be read through
    // CS; a code segment is never writable; a read-only data segment
    // refuses stores. All three raise #GP(0) before the reference.
    for (segment, attributes, code) in [
        (0, 0x99, &b"\x43\x2e\xa1\x00\x00"[..]),
        (0, 0x9b, b"\x43\x2e\xa3\x00\x00"),
        (1, 0x91, b"\x43\xa3\x00\x00"),
        (2, 0x91, b"\x43\x26\x88\x07"),
    ] {
        for cpl in [0, 3] {
            let (avm, mut a) = setup(code, cpl, false);
            let (bvm, mut b) = setup(code, cpl, true);
            for cpu in [&mut a, &mut b] {
                cpu.cpu.legacy.segments[segment].attributes = attributes | (u16::from(cpl) << 5);
            }
            assert_eq!(class(&a), SegmentClass::Segmented);
            compare((&avm, &mut a), (&bvm, &mut b), 3, &format!("{code:02x?}"));
            assert_eq!(a.retired_instructions(), 1);
        }
    }
}

fn for_limit(cpu: &mut Vcpu, segment: usize, limit: u32, down: bool) {
    cpu.cpu.legacy.segments[segment].limit = limit;
    if down {
        cpu.cpu.legacy.segments[segment].attributes |= 4;
    }
}

#[test]
fn in_block_segment_loads_and_their_faults_match_checked() {
    // 286 PRM App. B MOV pp. B-73–74 (POP, LDS/LES alike): a null selector
    // loads DS/ES and faults on use with #GP(0); an index past the table
    // limit, a system or execute-only descriptor and a DPL below CPL raise
    // #GP(selector); a non-present one #NP(selector); null SS #GP(0); SS
    // with DPL != CPL or read-only #GP(selector); non-present SS #SS(selector).
    for cpl in [0, 3] {
        let rpl = u16::from(cpl);
        let dpl = u16::from(cpl) << 5;
        // (MOV Sreg,AX ModRM, selector, descriptor 28h access byte, the
        // fault and the instructions retired before it, or None.)
        for (reg, selector, attributes, outcome) in [
            (0xd8, 0, 0, Some((13, 0, 2))), // null DS, then its use
            (0xc0, 0, 0, None),             // null ES, never used
            (0xd8, 0x80 | rpl, 0, Some((13, 0x80, 1))),
            (0xd8, 0x28 | rpl, 0x82 | dpl, Some((13, 0x28, 1))),
            (0xd8, 0x28 | rpl, 0x99 | dpl, Some((13, 0x28, 1))),
            (0xd8, 0x28 | rpl, 0x13 | dpl, Some((11, 0x28, 1))),
            (0xc0, 0x28 | rpl, 0x13 | dpl, Some((11, 0x28, 1))),
            // Expand-down limit 1FFh: offset 0 is outside (§7.4.1 p. 7-13).
            (0xd8, 0x28 | rpl, 0x97 | dpl, Some((13, 0, 2))),
            (0xd8, 0x28 | rpl, 0x9b | dpl, None), // readable code
            (0xd0, 0, 0, Some((13, 0, 1))),
            (0xd0, 0x28 | rpl, 0x91 | dpl, Some((13, 0x28, 1))),
            (0xd0, 0x28 | rpl, 0x13 | dpl, Some((12, 0x28, 1))),
        ] {
            let code = [0x43, 0x8e, reg, 0xa1, 0x00, 0x00, 0x42, 0xf4];
            let load = |checked| {
                let (vm, mut cpu) = setup(&code, cpl, checked);
                descriptor(&vm, 0x3028, 0x5000, 0x1ff, attributes);
                cpu.cpu.gpr[0] = u64::from(selector);
                (vm, cpu)
            };
            let what = format!("CPL{cpl} {reg:02x} {selector:04x} {attributes:02x}");
            for budget in [1, 2, 3, 10] {
                let ((avm, mut a), (bvm, mut b)) = (load(false), load(true));
                compare((&avm, &mut a), (&bvm, &mut b), budget, &what);
            }
            let (vm, mut cpu) = load(false);
            if let Some((vector, error, retired)) = outcome {
                let (steps, actual, code) = exception(&mut cpu, &vm, 10);
                assert_eq!(steps, retired, "{what}");
                assert_eq!(
                    (actual, code),
                    (vector, ErrorCode::bits(error, 0xffff)),
                    "{what}"
                );
            } else {
                let run = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
                // HLT retires at CPL 0 and raises #GP(0) at CPL 3.
                assert_eq!(run.steps(), 4 + u64::from(cpl == 0), "{what}: {run:?}");
            }
        }
        // DPL < CPL for data at CPL 3: #GP(selector).
        if cpl == 3 {
            let (vm, mut cpu) = setup(&[0x43, 0x8e, 0xd8, 0x42], 3, false);
            descriptor(&vm, 0x3028, 0x5000, 0x1ff, 0x93);
            cpu.cpu.gpr[0] = 0x2b;
            let (steps, vector, error) = exception(&mut cpu, &vm, 10);
            assert_eq!(
                (steps, vector, error),
                (1, 13, ErrorCode::bits(0x28, 0xffff))
            );
        }
    }
}

#[test]
fn far_transfers_same_level_and_through_call_gates_match_checked() {
    // 286 PRM App. B CALL pp. B-23–25 and RET pp. B-94–95: a same-level
    // far CALL and a 286 call gate (type 4) to a same-level code segment
    // push a word CS:IP frame; RETF pops it.
    for cpl in [0, 3] {
        for gate in [false, true] {
            for jump in [false, true] {
                let selector = if gate { 0x20 } else { 8 } | cpl;
                let code = [
                    0x43,
                    if jump { 0xea } else { 0x9a },
                    0x80,
                    1,
                    selector,
                    0,
                    0x42,
                    0xf4,
                ];
                for budget in 0..9 {
                    let (avm, mut a) = setup(&code, cpl, false);
                    let (bvm, mut b) = setup(&code, cpl, true);
                    for vm in [&avm, &bvm] {
                        vm.write_bytes(0x1180, if jump { b"\x40\xf4" } else { b"\x40\xcb" })
                            .unwrap();
                        vm.write_bytes(0x3020, &[0x80, 1, 8 | cpl, 0, 0, 0x84 | (cpl << 5), 0, 0])
                            .unwrap();
                    }
                    compare(
                        (&avm, &mut a),
                        (&bvm, &mut b),
                        budget,
                        &format!("CPL{cpl} gate={gate} jump={jump}"),
                    );
                    if budget >= 4 {
                        assert_eq!(a.cpu.gpr[0], 1);
                        assert_eq!(a.cpu.gpr[4], 0x1000);
                    }
                    assert_eq!(class(&a), SegmentClass::Segmented);
                    assert_eq!(class(&b), SegmentClass::Checked);
                }
            }
        }
    }
}

#[test]
fn task_switch_faults_report_the_new_tss_selector() {
    // 286 PRM App. B SWITCH_TASKS p. B-11: a new TSS whose limit is
    // below 43 raises #TS(new TSS); §8.2 p. 8-3. The JMP ends the block
    // and the INC before it retires in both lowerings.
    let tss = |checked: bool| {
        let (vm, mut cpu) = setup(&[0x43, 0xea, 0, 0, 0x38, 0, 0x42], 0, checked);
        descriptor(&vm, 0x3030, 0xc000, 0x2b, 0x83);
        descriptor(&vm, 0x3038, 0xd000, 0x20, 0x81);
        cpu.cpu.legacy.tr.selector = 0x30;
        cpu.cpu.legacy.tr.cache = x86jit_core::SegmentCache {
            base: 0xc000,
            limit: 0x2b,
            attributes: 0x83,
            ..Default::default()
        };
        (vm, cpu)
    };
    for budget in [1, 2, 5] {
        let ((avm, mut a), (bvm, mut b)) = (tss(false), tss(true));
        compare((&avm, &mut a), (&bvm, &mut b), budget, "task switch");
        assert_eq!(a.retired_instructions(), 1);
    }
    let (vm, mut cpu) = tss(false);
    let (steps, vector, error) = exception(&mut cpu, &vm, 5);
    assert_eq!((steps, vector), (1, 10));
    assert_eq!(error, ErrorCode::bits(0x38, 0xffff));
}

#[test]
fn budgets_and_fault_ordinals_match_checked() {
    let bodies: &[&[u8]] = &[
        b"\x40\x01\xc3\x31\xda",
        b"\xa1\x00\x00\x40\xa3\x02\x00",
        b"\x50\x5b\x53\x59",
        b"\x8e\xdb\xa1\x00\x00\x26\xa1\x04\x00",
        b"\xf3\xa4",
        b"\xe2\xfe",
        b"\xe8\x01\x00\x90\xc3",
        b"\xc8\x04\x00\x01\xc9",
    ];
    let faults: &[&[u8]] = &[
        b"\xf4",
        b"\x0f\x0b",
        b"\xf7\xf1",
        b"\xa1\xff\x00",
        b"\x2e\xa3\x00\x00",
        b"\x36\xa1\xff\x1f",
        b"\x8e\xd8\xa1\x00\x00",
        b"\xfa",
        b"\x66\x40",
        // 286 PRM App. B #GP p. B-9: an instruction longer than ten bytes.
        b"\x2e\x2e\x2e\x2e\x2e\x2e\x2e\x2e\x2e\x2e\x40",
    ];
    for cpl in [0, 3] {
        for body in bodies {
            for fault in faults {
                for ordinal in 0..3 {
                    let mut code = body.to_vec();
                    code.extend(vec![0x42; ordinal]);
                    code.extend_from_slice(fault);
                    for budget in [0, 1, 2, 3, 5, 9] {
                        let (avm, mut a) = setup(&code, cpl, false);
                        let (bvm, mut b) = setup(&code, cpl, true);
                        for cpu in [&mut a, &mut b] {
                            cpu.cpu.gpr[1] = 2;
                            cpu.cpu.gpr[3] = 16 | u64::from(cpl);
                        }
                        compare(
                            (&avm, &mut a),
                            (&bvm, &mut b),
                            budget,
                            &format!("CPL{cpl} {code:02x?}"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn single_step_and_interrupt_boundaries_match_checked() {
    // 286 PRM §9.7.1 p. 9-14: TF traps after each instruction. §9.2 p. 9-2:
    // IF=1 admits INTR, taken between instructions; the IRQ arrives after
    // 0..3 retired, inside what the Segmented lowering runs as one block.
    let code = b"\x43\xa1\x00\x00\x40\x31\xda\x42\xf4";
    for cpl in [0, 3] {
        for budget in [1, 2, 4, 9] {
            let (avm, mut a) = setup(code, cpl, false);
            let (bvm, mut b) = setup(code, cpl, true);
            for cpu in [&mut a, &mut b] {
                cpu.cpu.rflags_system |= 0x100;
            }
            let r = compare((&avm, &mut a), (&bvm, &mut b), budget, "TF");
            assert!(r.contains("vector: 1"), "{r}");
            assert_eq!(a.retired_instructions(), 1);
        }
        for after in [0, 1, 2, 3] {
            let (avm, mut a) = setup(code, cpl, false);
            let (bvm, mut b) = setup(code, cpl, true);
            for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                cpu.cpu.rflags_system |= 0x200;
                cpu.cpu.legacy.idtr.base = 0x4000;
                cpu.cpu.legacy.idtr.limit = 0x7ff;
                // 286 interrupt gate (type 6) to 08h:0700h, DPL 3.
                vm.write_bytes(0x4000 + 0x20 * 8, &[0, 7, 8, 0, 0, 0xe6, 0, 0])
                    .unwrap();
                vm.write_bytes(0x1700, b"\x41\xf4").unwrap();
            }
            compare((&avm, &mut a), (&bvm, &mut b), after, "before IRQ");
            a.inject_irq(0x20);
            b.inject_irq(0x20);
            compare((&avm, &mut a), (&bvm, &mut b), 3, "IRQ");
        }
    }
}

#[test]
fn smc_stops_before_stale_successors_including_past_the_24_bit_wrap() {
    // A store into a later instruction of the running block: both lowerings
    // execute the new bytes.
    let code = b"\x40\xc6\x06\x07\x01\xf4\x40\x40\xf4";
    let (avm, mut a) = setup(code, 0, false);
    let (bvm, mut b) = setup(code, 0, true);
    for cpu in [&mut a, &mut b] {
        cpu.cpu.legacy.segments[1].base = 0x1000;
        cpu.cpu.legacy.segments[1].limit = 0x1fff;
    }
    compare((&avm, &mut a), (&bvm, &mut b), 20, "SMC");
    assert_eq!(a.retired_instructions(), 4);
    assert_eq!(a.cpu.gpr[0], 2);
    // 286 PRM §6.4 p. 6-6: CS.base FFFF00h + IP 0100h forms 24-bit address
    // 000000h. The store through DS (base 0) to physical 5 rewrites the
    // block's INC AX to INC CX; the block was registered at the wrapped
    // physical bytes, so the write stops it before the stale successor.
    for warm in [false, true] {
        let (avm, mut a) = setup(b"", 0, false);
        let (bvm, mut b) = setup(b"", 0, true);
        for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
            vm.write_bytes(0, b"\xc6\x06\x05\x00\x41\x40\xf4").unwrap();
            cpu.cpu.legacy.segments[0].base = 0xff_ff00;
            cpu.cpu.legacy.segments[0].limit = 0xffff;
            cpu.cpu.legacy.segments[1].base = 0;
            cpu.cpu.legacy.segments[1].limit = 0xffff;
            if warm {
                // Lift and publish the block once, then restore its bytes.
                cpu.cpu.rip = 0x105;
                let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
                assert_eq!(run.steps(), 1);
                cpu.cpu.gpr[0] = 0;
            }
            cpu.cpu.rip = 0x100;
        }
        let r = compare((&avm, &mut a), (&bvm, &mut b), 20, "wrapped SMC");
        assert!(r.contains("Hlt"), "{r}");
        assert_eq!((a.cpu.gpr[0], a.cpu.gpr[1]), (0, 1), "warm={warm}");
    }
}

#[test]
fn loadall_ends_the_block_and_reselects_from_the_loaded_caches() {
    // Intel, "Undocumented iAPX 286 Test Instruction" p. 2 (table at
    // 000800h) and p. 1 (CPL 0 in protected mode): the loaded caches are the
    // only authority; the next block is proved from them, an expand-down ES
    // included.
    let code = b"\x43\x0f\x05";
    let mut table = [0u8; 102];
    let mut word = |at: usize, v: u16| table[at - 0x800..][..2].copy_from_slice(&v.to_le_bytes());
    word(0x806, 0xfff1); // MSW: PE.
    word(0x816, 0x30); // TR.
    word(0x818, 0x0002); // FLAGS.
    word(0x81a, 0x0120); // IP.
    word(0x81c, 0x28); // LDT.
    word(0x81e, 0x10); // DS.
    word(0x820, 0x18); // SS.
    word(0x822, 0x08); // CS.
    word(0x824, 0x10); // ES.
    word(0x82c, 0x1000); // SP.
    for (at, base, access, limit) in [
        (0x836, 0x6000u32, 0x97u8, 0x0fffu16), // ES: expand-down, 1000h..FFFFh.
        (0x83c, 0x1000, 0x9b, 0x1fff),         // CS.
        (0x842, 0x7000, 0x93, 0x1fff),         // SS.
        (0x848, 0x5000, 0x93, 0x0fff),         // DS.
        (0x84e, 0x3000, 0, 0x7f),              // GDTR.
        (0x854, 0x4000, 0x82, 0x37),           // LDT.
        (0x85a, 0, 0, 0),                      // IDTR.
        (0x860, 0xc000, 0x83, 0x2b),           // TSS.
    ] {
        let bytes = &mut table[at - 0x800..][..6];
        bytes[..3].copy_from_slice(&base.to_le_bytes()[..3]);
        bytes[3] = access;
        bytes[4..].copy_from_slice(&limit.to_le_bytes());
    }
    // MOV AX,[0FFEh]; MOV AX,ES:[1000h]; MOV ES:[0FFFh],AL (#GP(0)).
    let after = b"\xa1\xfe\x0f\x26\xa1\x00\x10\x26\xa2\xff\x0f";
    for budget in 1..7 {
        let (avm, mut a) = setup(code, 0, false);
        let (bvm, mut b) = setup(code, 0, true);
        for vm in [&avm, &bvm] {
            vm.write_bytes(0x800, &table).unwrap();
            vm.write_bytes(0x1120, after).unwrap();
        }
        let r = compare((&avm, &mut a), (&bvm, &mut b), budget, "LOADALL");
        if budget >= 2 {
            assert_eq!(a.cpu.legacy.segments[2].attributes, 0x97);
            assert_eq!(class(&a), SegmentClass::Segmented);
            assert_eq!(class(&b), SegmentClass::Checked);
        }
        if budget >= 5 {
            assert!(r.contains("vector: 13"), "{r}");
            assert_eq!(a.retired_instructions(), 4);
        }
    }
}
