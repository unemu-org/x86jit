//! LDT and descriptor-query witnesses on both Machine models (interpreter only).
//! Intel 286 PRM 210498-005 and 386 PRM 230985-001: cited scans checked.
#[path = "support/descriptor_fixture.rs"]
mod fixture;
use fixture::*;
use x86jit_core::{CpuModel, SegmentCache, SegmentId, SystemSegment};

#[test]
fn every_instruction_completes_on_both_models() {
    // Instruction pages cited by the individual witnesses below and in
    // legacy_task_registers.rs. Collect all failures to expose both models.
    let mut failures = Vec::new();
    for model in MODELS {
        for (name, code) in [
            ("LLDT", &b"\x0f\x00\xd0"[..]),
            ("SLDT", &b"\x0f\x00\xc0"[..]),
            ("LTR", &b"\x0f\x00\xd8"[..]),
            ("STR", &b"\x0f\x00\xc8"[..]),
            ("LAR", &b"\x0f\x02\xc1"[..]),
            ("LSL", &b"\x0f\x03\xc1"[..]),
            ("VERR", &b"\x0f\x00\xe1"[..]),
            ("VERW", &b"\x0f\x00\xe9"[..]),
            ("ARPL", &b"\x63\xc8"[..]),
        ] {
            let (vm, mut cpu) = setup(model, code);
            if name == "LTR" {
                descriptor(&vm, 0x2008, 0x4000, 0x2b, 0x81);
            }
            cpu.cpu.gpr[0] = 8;
            cpu.cpu.gpr[1] = 8;
            let run = cpu
                .run_steps(&vm, 1, x86jit_core::StepOptions::default())
                .unwrap();
            if run.steps() != 1 {
                failures.push(format!("{model:?} {name}: {:?}", run.stop()));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

macro_rules! query_witness {
    ($name:ident, $code:expr, $expected:expr) => {
        #[test]
        fn $name() {
            // 286 PRM App. B pp. B-60/B-71/B-111;
            // 386 PRM §17 pp. 17-89/17-106/17-169.
            for model in MODELS {
                let (vm, mut cpu) = setup(model, $code);
                descriptor(&vm, 0x2010, 0, 0x2345, 0x92);
                cpu.cpu.gpr[1] = 16;
                steps(&mut cpu, &vm, 1);
                assert!(cpu.cpu.flags.zf);
                assert_eq!(cpu.cpu.gpr[0], $expected);
            }
        }
    };
}
query_witness!(lar_returns_rights, b"\x0f\x02\xc1", 0x9200);
query_witness!(lsl_returns_limit, b"\x0f\x03\xc1", 0x2345);
query_witness!(verr_accepts_readable_data, b"\x0f\x00\xe1", 0);
query_witness!(verw_accepts_writable_data, b"\x0f\x00\xe9", 0);

#[test]
fn sldt_stores_visible_selector() {
    // 286 PRM App. B p. B-102; 386 PRM §17 p. 17-159.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x0f\x00\xc0");
        cpu.cpu.legacy.ldtr.selector = 11;
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[0], 11);
    }
}

#[test]
fn lldt_sldt_cache_and_null_selector() {
    // 286 PRM §§6.6.2 pp. 6-12–13, App. B pp. B-66/B-102;
    // 386 PRM §17 pp. 17-98/17-159; SDM Vol 3A §3.5.1.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x0f\x00\xd0\x0f\x00\xc1\x0f\x00\xd0\x0f\x00\xc2");
        cpu.cpu.gpr[0] = 11; // RPL does not restrict LLDT at CPL 0.
        let segments = cpu.cpu.legacy.segments;
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.legacy.segments, segments);
        assert_eq!(cpu.cpu.legacy.ldtr.selector, 11);
        assert_eq!(cpu.cpu.legacy.ldtr.cache.base, 0x3000);
        assert_eq!(cpu.cpu.legacy.ldtr.cache.limit, 0x3f);
        assert_eq!(cpu.cpu.legacy.ldtr.cache.attributes, 0x82);
        let cached = cpu.cpu.legacy.ldtr;
        descriptor(&vm, 0x2008, 0x4000, 7, 0x82);
        cpu.cpu.gpr[1] = 0x1234_5678;
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[1], 0x1234_000b);
        assert_eq!(cpu.cpu.legacy.ldtr, cached);
        cpu.cpu.gpr[0] = 3;
        steps(&mut cpu, &vm, 2);
        assert_eq!(cpu.cpu.gpr[2], 3);
        assert_eq!(cpu.cpu.legacy.ldtr.selector, 3);
        assert_eq!(cpu.cpu.legacy.ldtr.cache.unusable, 1);
        assert_eq!(value(&vm, 0x200d, 1), 0x82);
    }
}

#[test]
fn lldt_faults_do_not_install_a_cache() {
    // 286 PRM App. B p. B-66; 386 PRM §17 p. 17-98:
    // GDT only, descriptor type before presence, complete table bounds.
    for model in MODELS {
        for (selector, attr, limit, vector, code) in [
            (12, 0x82, 0x7f, 13, 12),
            (8, 0x92, 0x7f, 13, 8),
            (8, 0x02, 0x7f, 11, 8),
            (11, 0x82, 14, 13, 8),
        ] {
            let (vm, mut cpu) = setup(model, b"\x0f\x00\xd0");
            descriptor(&vm, 0x2008, 0x3000, 0x3f, attr);
            cpu.cpu.gpr[0] = selector;
            cpu.cpu.legacy.gdtr.limit = limit;
            fault(&mut cpu, &vm, vector, Some(code));
        }
    }
}

#[test]
fn ldt_data_stack_and_code_loads_set_accessed_and_use_cached_limits() {
    // 286 PRM §§7.4–7.5 pp. 7-11–15; 386 PRM §§5.1.1, 6.3.1–6.3.2;
    // SDM Vol 3A §§3.4.2, 3.4.5, 3.5.1: TI and cached table bounds.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x0f\x00\xd0\x8e\xd9\x8e\xd2\xea\x00\x12\x14\x00");
        descriptor(&vm, 0x3000, 0x4000, 0x1f, 0x92); // LDT index zero is valid.
        descriptor(&vm, 0x3008, 0x5000, 0x8fff, 0x92);
        descriptor(&vm, 0x3010, 0, 0xffff, 0x9a);
        cpu.cpu.gpr[0] = 8;
        cpu.cpu.gpr[1] = 4;
        cpu.cpu.gpr[2] = 12;
        steps(&mut cpu, &vm, 1);
        // Editing the GDT descriptor cannot refresh the cached LDTR.
        descriptor(&vm, 0x2008, 0x6000, 0, 0x82);
        steps(&mut cpu, &vm, 3);
        assert_eq!((cpu.cpu.ds, cpu.cpu.ss, cpu.cpu.cs), (4, 12, 20));
        assert_eq!(cpu.cpu.legacy.segments[1].base, 0x4000);
        assert_eq!(cpu.cpu.legacy.segments[3].base, 0x5000);
        assert_eq!(cpu.cpu.rip, 0x1200);
        for address in [0x3005, 0x300d, 0x3015] {
            assert_eq!(value(&vm, address, 1) & 1, 1);
        }
        vm.write_bytes(0x1200, b"\xa1\x1f\x00").unwrap();
        fault(&mut cpu, &vm, 13, Some(0));
    }
}

#[test]
fn lar_lsl_verr_verw_descriptor_types_and_privileges() {
    // SDM Vol 2A §3.3 LAR Table 3-55, LSL Table 3-59, Vol 2C §5.2 VERR/VERW pp. 5-162–163;
    // 286 PRM §11.3 p. 11-3, App. B pp. B-60/B-71/B-111;
    // 386 PRM §6.3.6 pp. 6-18–19, §17 pp. 17-89/17-106/17-169.
    // SDM tables are primary: the 386 PRM LAR table includes interrupt/trap
    // gates, and its LSL table marks reserved type 8 valid; both disagree.
    for model in MODELS {
        for (code, allowed) in [
            (
                &b"\x0f\x02\xc1"[..],
                if model == CpuModel::I80286 {
                    0x003e
                } else {
                    0x1a3e
                },
            ),
            (
                &b"\x0f\x03\xc1"[..],
                if model == CpuModel::I80286 {
                    0x000e
                } else {
                    0x0a0e
                },
            ),
            (&b"\x0f\x00\xe1"[..], 0),
            (&b"\x0f\x00\xe9"[..], 0),
        ] {
            for ty in 0..32 {
                let (vm, mut cpu) = setup(model, code);
                let attr = 0x80 | ty;
                descriptor(&vm, 0x2010, 0x4000, 0x2345, attr);
                cpu.cpu.gpr[0] = 0x1234_5678_9abc_def0;
                cpu.cpu.gpr[1] = 16;
                cpu.cpu.flags.cf = true;
                cpu.cpu.flags.of = true;
                let conforming = ty & 0x1c == 0x1c;
                let expected = match code {
                    b"\x0f\x00\xe1" => ty & 16 != 0 && (ty & 8 == 0 || ty & 2 != 0),
                    b"\x0f\x00\xe9" => ty & 16 != 0 && ty & 8 == 0 && ty & 2 != 0,
                    b"\x0f\x03\xc1" if model == CpuModel::I80286 && conforming => false,
                    _ => ty & 16 != 0 || allowed & (1 << ty) != 0,
                };
                steps(&mut cpu, &vm, 1);
                assert_eq!(
                    cpu.cpu.flags.zf, expected,
                    "{model:?} {code:x?} type={ty:x}"
                );
                assert!(cpu.cpu.flags.cf && cpu.cpu.flags.of);
                let result = if expected && code[1] == 2 {
                    u64::from(attr) << 8
                } else if expected && code[1] == 3 {
                    0x2345
                } else {
                    0xdef0
                };
                assert_eq!(cpu.cpu.gpr[0], 0x1234_5678_9abc_0000 | result);
                assert_eq!(value(&vm, 0x2015, 1), u64::from(attr));
            }
        }
    }
}

#[test]
fn arpl_adjusts_only_rpl_and_zf() {
    // 286 PRM App. B p. B-21; 386 PRM §17 p. 17-25.
    for model in MODELS {
        for (left, right, expected, zf) in [(0x1230, 3, 0x1233, true), (0x1233, 1, 0x1233, false)] {
            for code in [&b"\x63\xc8"[..], &b"\x63\x0e\x00\x40"[..]] {
                let (vm, mut cpu) = setup(model, code);
                cpu.cpu.gpr[0] = 0x5678_0000 | left;
                cpu.cpu.gpr[1] = right;
                cpu.cpu.flags.cf = true;
                vm.write_bytes(0x4000, &(left as u16).to_le_bytes())
                    .unwrap();
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.flags.zf, zf);
                assert!(cpu.cpu.flags.cf);
                assert_eq!(
                    if code.len() == 2 {
                        cpu.cpu.gpr[0] & 0xffff
                    } else {
                        value(&vm, 0x4000, 2)
                    },
                    expected
                );
            }
        }
    }
}

#[test]
fn protected_only_instructions_ud_before_real_operand_access() {
    // 286 PRM App. B pp. B-21/60/66/71/72/102/108/111;
    // 386 PRM §17 pp. 17-25/89/98/106/108/159/166/169: real #UD.
    for model in MODELS {
        for code in [
            &b"\x0f\x00\x16\xff\xff"[..],
            &b"\x0f\x00\x06\xff\xff"[..],
            &b"\x0f\x00\x1e\xff\xff"[..],
            &b"\x0f\x00\x0e\xff\xff"[..],
            &b"\x0f\x02\x06\xff\xff"[..],
            &b"\x0f\x03\x06\xff\xff"[..],
            &b"\x0f\x00\x26\xff\xff"[..],
            &b"\x0f\x00\x2e\xff\xff"[..],
            &b"\x63\x0e\xff\xff"[..],
        ] {
            let (vm, mut cpu) = setup(model, code);
            cpu.cpu.legacy.cr0 = 0;
            fault(&mut cpu, &vm, 6, None);
        }
    }
}

fn install_ldt(cpu: &mut x86jit_core::Vcpu, base: u32, limit: u32) {
    cpu.cpu.legacy.ldtr = SystemSegment {
        selector: 8,
        cache: SegmentCache {
            base,
            limit,
            attributes: 0x82,
            ..Default::default()
        },
        ..Default::default()
    };
}

#[test]
fn ldt_selector_fault_codes_preserve_ti_and_clear_rpl() {
    // 286 PRM §7.4 pp. 7-11–13, App. B MOV pp. B-73–74;
    // 386 PRM §6.3.1.2, §17 MOV pp. 17-109–110; SDM Vol 2B §4.3 MOV.
    for model in MODELS {
        for (code, selector, attr, limit, vector, error) in [
            (&b"\x8e\xd8"[..], 12, 0x92, 14, 13, 12),
            (&b"\x8e\xd8"[..], 15, 0x72, 15, 11, 12),
            (&b"\x8e\xd0"[..], 12, 0x12, 15, 12, 12),
            (&b"\x8e\xd8"[..], 12, 0x82, 15, 13, 12),
            (&b"\x8e\xd0"[..], 15, 0x92, 15, 13, 12),
            (&b"\xea\x00\x12\x0c\x00"[..], 12, 0x1a, 15, 11, 12),
            (&b"\xea\x00\x12\x0c\x00"[..], 12, 0x9a, 14, 13, 12),
        ] {
            let (vm, mut cpu) = setup(model, code);
            install_ldt(&mut cpu, 0x3000, limit);
            descriptor(&vm, 0x3008, 0, 0xffff, attr);
            cpu.cpu.gpr[0] = selector;
            fault(&mut cpu, &vm, vector, Some(error));
            assert_eq!(value(&vm, 0x300d, 1), u64::from(attr));
        }
        for unusable in [false, true] {
            let (vm, mut cpu) = setup(model, b"\x8e\xd8");
            install_ldt(&mut cpu, 0x3000, 15);
            if unusable {
                cpu.cpu.legacy.ldtr.cache.unusable = 1;
            } else {
                cpu.cpu.legacy.ldtr.selector = 0;
            }
            cpu.cpu.gpr[0] = 12;
            fault(&mut cpu, &vm, 13, Some(12));
        }
    }
}

#[test]
fn queries_fail_without_fault_or_destination_change_for_invalid_selectors() {
    // 286 PRM App. B pp. B-60/B-71/B-111; 386 PRM §6.3.6 pp. 6-18–19:
    // selector faults become ZF=0, including an unavailable LDT.
    for model in MODELS {
        for code in [
            b"\x0f\x02\xc1",
            b"\x0f\x03\xc1",
            b"\x0f\x00\xe1",
            b"\x0f\x00\xe9",
        ] {
            for selector in [0, 3, 4, 12, 0xfffb] {
                let (vm, mut cpu) = setup(model, code);
                cpu.cpu.gpr[0] = 0x1234_5678_9abc_def0;
                cpu.cpu.gpr[1] = selector;
                cpu.cpu.flags.zf = true;
                steps(&mut cpu, &vm, 1);
                assert!(!cpu.cpu.flags.zf);
                assert_eq!(cpu.cpu.gpr[0], 0x1234_5678_9abc_def0);
            }
        }
    }
}

#[test]
fn queries_check_rpl_but_ignore_presence_and_conforming_privilege() {
    // 286 PRM §11.3 pp. 11-3–4/App. B p. B-71 (nonconforming LSL);
    // 386 PRM §6.3.6 pp. 6-18–19; SDM Vol 2A §3.3 LAR/LSL,
    // Vol 2C §5.2 VERR/VERW pp. 5-162–163: P and A do not affect query results.
    for model in MODELS {
        for (code, writable) in [
            (b"\x0f\x02\xc1", false),
            (b"\x0f\x03\xc1", false),
            (b"\x0f\x00\xe1", false),
            (b"\x0f\x00\xe9", true),
        ] {
            for (attr, selector, accepted) in [
                (0x12, 12, true),
                (0x92, 15, false),
                (0xf2, 15, true),
                (
                    0x9e,
                    15,
                    !writable && !(model == CpuModel::I80286 && code[1] == 3),
                ),
            ] {
                let (vm, mut cpu) = setup(model, code);
                install_ldt(&mut cpu, 0x3000, 15);
                descriptor(&vm, 0x3008, 0, 0x2345, attr);
                cpu.cpu.gpr[1] = selector;
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.flags.zf, accepted, "{model:?} {code:x?} {attr:x}");
                assert_eq!(value(&vm, 0x300d, 1), u64::from(attr));
            }
        }
    }
}

#[test]
fn descriptor_query_memory_sources_are_words_and_operand_faults_escape() {
    // SDM Vol 2A §3.3 LAR/LSL; 386 PRM §17 pp. 17-89/17-106:
    // memory source remains a selector word with a 32-bit destination.
    for model in MODELS {
        for code in [
            &b"\x0f\x02\x06\xfe\xff"[..],
            &b"\x0f\x03\x06\xfe\xff"[..],
            &b"\x0f\x00\x26\xfe\xff"[..],
            &b"\x0f\x00\x2e\xfe\xff"[..],
        ] {
            let (vm, mut cpu) = setup(model, code);
            descriptor(&vm, 0x2010, 0, 0x2345, 0x92);
            vm.write_bytes(0xfffe, &16u16.to_le_bytes()).unwrap();
            steps(&mut cpu, &vm, 1);
            assert!(cpu.cpu.flags.zf);
            cpu.cpu.rip = 0x1000;
            cpu.cpu.legacy.segments[SegmentId::Ds as usize].limit = 0xfffe;
            fault(&mut cpu, &vm, 13, Some(0));
        }
    }
    for opcode in [2, 3] {
        let (vm, mut cpu) = setup(CpuModel::I80386, &[0x66, 0x0f, opcode, 0x06, 0xfe, 0xff]);
        descriptor(&vm, 0x2010, 0, 0x1234_5fff, 0xc092);
        vm.write_bytes(0xfffe, &16u16.to_le_bytes()).unwrap();
        cpu.cpu.gpr[0] = u64::MAX;
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            cpu.cpu.gpr[0],
            if opcode == 2 { 0xc09200 } else { 0x1234_5fff }
        );
    }
}

#[test]
fn i386_queries_preserve_full_destination_on_failure_and_truncate_word_results() {
    // SDM Vol 2A §3.3 LAR/LSL: failed queries do not write the destination;
    // 16-bit results preserve the upper bits, including for page-granular limits.
    for opcode in [2, 3] {
        let (vm, mut cpu) = setup(CpuModel::I80386, &[0x66, 0x0f, opcode, 0xc1]);
        cpu.cpu.gpr[0] = 0x1234_5678_9abc_def0;
        cpu.cpu.gpr[1] = 0;
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[0], 0x1234_5678_9abc_def0);
        assert!(!cpu.cpu.flags.zf);
    }
    let (vm, mut cpu) = setup(CpuModel::I80386, b"\x0f\x03\xc1");
    descriptor(&vm, 0x2010, 0, 0x1234_5fff, 0xc092);
    cpu.cpu.gpr[0] = 0x1234_5678_9abc_def0;
    cpu.cpu.gpr[1] = 16;
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[0], 0x1234_5678_9abc_5fff);
}

#[test]
fn a20_gates_gdt_ldt_reads_and_ldt_accessed_writeback() {
    // Decision-22 bus contract; SDM Vol 3A §§3.5.1, 11.7.13.4;
    // 286 PRM §6.6.2 pp. 6-12–13; 386 PRM §5.1.2.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x0f\x00\xd0\x8e\xd9");
        cpu.cpu.legacy.gdtr.base = 0x102000;
        descriptor(&vm, 0x2008, 0x103000, 7, 0x82);
        descriptor(&vm, 0x102008, 0x4000, 7, 0x82);
        descriptor(&vm, 0x3000, 0x5000, 0xffff, 0x92);
        descriptor(&vm, 0x103000, 0x6000, 0xffff, 0x92);
        cpu.cpu.gpr[0] = 8;
        cpu.cpu.gpr[1] = 4;
        cpu.set_a20_enabled(false);
        steps(&mut cpu, &vm, 2);
        assert_eq!(cpu.cpu.legacy.ldtr.cache.base, 0x103000);
        assert_eq!(cpu.cpu.legacy.segments[1].base, 0x5000);
        assert_eq!(value(&vm, 0x3005, 1), 0x93);
        assert_eq!(value(&vm, 0x103005, 1), 0x92);
    }
}

#[test]
fn i286_ignores_reserved_descriptor_bytes_while_i386_expands_them() {
    // 286 PRM §6.3 Fig. 6-3 p. 6-5/decision-21 reserved-byte policy;
    // 386 PRM §5.1.1 Fig. 5-3: high base, flags, 20-bit limit and G.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x0f\x00\xd0");
        descriptor(&vm, 0x2008, 0x1230_3000, 0x1234_5fff, 0x8082);
        cpu.cpu.gpr[0] = 8;
        steps(&mut cpu, &vm, 1);
        let cache = cpu.cpu.legacy.ldtr.cache;
        assert_eq!(
            cache.base,
            if model == CpuModel::I80286 {
                0x303000
            } else {
                0x1230_3000
            }
        );
        assert_eq!(
            cache.limit,
            if model == CpuModel::I80286 {
                0x2345
            } else {
                0x1234_5fff
            }
        );
    }
}

#[test]
fn system_selector_memory_stores_are_words() {
    // 286 PRM App. B pp. B-102/B-108;
    // SDM Vol 2B §4.3 SLDT/STR pp. 4-656/4-679: memory stores are words.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x0f\x00\x06\x00\x40\x0f\x00\x0e\x02\x40");
        cpu.cpu.legacy.ldtr.selector = 11;
        cpu.cpu.legacy.tr.selector = 19;
        vm.write_bytes(0x4000, &[0xaa; 6]).unwrap();
        steps(&mut cpu, &vm, 2);
        assert_eq!(value(&vm, 0x4000, 6), 0xaaaa_0013_000b);
    }
}

#[test]
fn i386_sldt_preserves_undefined_upper_bits_and_str_clears_them() {
    // SDM 325462-092US Vol 2B §4.3 SLDT p. 4-656: high-order 16 bits
    // "are undefined for Pentium, Intel486, and Intel386 processors."
    // Preserve them as deterministic engine policy, consistent with 386 PRM
    // §17 SLDT p. 17-159: "The operand-size attribute has no effect on the
    // operation of the instruction."
    // SDM Vol 2B §4.3 STR p. 4-679: "the upper 16 bits of the register are cleared."
    // Conflict: 386 PRM §17 STR p. 17-166 says "The operand-size attribute has
    // no effect on this instruction." PROVENANCE.md §1 gives the SDM priority.
    // All four passages were checked against the pinned page images.
    for (code, expected_sldt, expected_str) in [
        (
            &b"\x0f\x00\xc0\x0f\x00\xc9"[..],
            0x1234_5678_9abc_000b,
            0xfedc_ba98_7654_0013,
        ),
        (
            &b"\x66\x0f\x00\xc0\x66\x0f\x00\xc9"[..],
            0x1234_5678_9abc_000b,
            19,
        ),
    ] {
        let (vm, mut cpu) = setup(CpuModel::I80386, code);
        cpu.cpu.legacy.ldtr.selector = 11;
        cpu.cpu.legacy.tr.selector = 19;
        cpu.cpu.gpr[0] = 0x1234_5678_9abc_def0;
        cpu.cpu.gpr[1] = 0xfedc_ba98_7654_3210;
        steps(&mut cpu, &vm, 2);
        assert_eq!(cpu.cpu.gpr[0], expected_sldt);
        assert_eq!(cpu.cpu.gpr[1], expected_str);
    }
}

#[test]
fn ldt_accessed_device_update_requires_atomic_emulation() {
    // SDM Vol 3A §11.1.2.1: accessed-bit write-back follows LOCK semantics.
    // Separate MMIO read/write acknowledgments cannot implement that operation.
    for model in MODELS {
        let (mut vm, mut cpu) = setup(model, b"\x8e\xd8");
        install_ldt(&mut cpu, 0x3000, 7);
        cpu.cpu.gpr[0] = 4;
        vm.unmap(0, 0x240000).unwrap();
        vm.map(
            0,
            0x3000,
            x86jit_core::Prot::RWX,
            x86jit_core::RegionKind::Ram,
        )
        .unwrap();
        vm.map(
            0x4000,
            0x23c000,
            x86jit_core::Prot::RWX,
            x86jit_core::RegionKind::Ram,
        )
        .unwrap();
        vm.map(
            0x3000,
            0x1000,
            x86jit_core::Prot::RW,
            x86jit_core::RegionKind::Trap,
        )
        .unwrap();
        let before = cpu.cpu.legacy;
        for (address, response) in [(0x3000, 0x4000_ffff), (0x3004, 0x0000_9200)] {
            let run = cpu
                .run_steps(&vm, 1, x86jit_core::StepOptions::default())
                .unwrap();
            assert_eq!(run.steps(), 0);
            assert!(
                matches!(run.stop(), x86jit_core::StepStop::Exit(x86jit_core::Exit::MmioRead { addr, size: 4, .. }) if *addr == address),
                "{:?}",
                run.stop()
            );
            assert_eq!(cpu.cpu.legacy, before);
            cpu.complete_mmio_read(response);
        }
        let run = cpu
            .run_steps(&vm, 1, x86jit_core::StepOptions::default())
            .unwrap();
        assert_eq!(run.steps(), 0);
        assert!(matches!(
            run.stop(),
            x86jit_core::StepStop::Exit(x86jit_core::Exit::UnsupportedSystemContext {
                operation: x86jit_core::UnsupportedSystemOperation::LegacyInstruction,
                ..
            })
        ));
        assert_eq!(cpu.cpu.legacy, before);
        assert_eq!(cpu.cpu.ds, 0);
    }
}

#[test]
fn arpl_memory_fault_keeps_flags_and_unchanged_rpl_requires_writable_segment() {
    // 286 PRM App. B p. B-21; 386 PRM §17 p. 17-25:
    // memory destination must be writable even if its RPL needs no increase.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x63\x0e\xff\xff");
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.flags.zf = false;
        fault(&mut cpu, &vm, 13, Some(0));
        let (vm, mut cpu) = setup(model, b"\x63\x0e\x00\x40");
        cpu.cpu.legacy.segments[1].attributes = 0x91;
        vm.write_bytes(0x4000, &3u16.to_le_bytes()).unwrap();
        cpu.cpu.gpr[1] = 1;
        fault(&mut cpu, &vm, 13, Some(0));
    }
}

#[test]
fn query_table_limit_failure_does_not_read_an_incomplete_ldt_descriptor() {
    // SDM Vol 3A §3.5.1/Vol 2A §3.3 LAR/LSL:
    // the cached LDTR limit must include all eight descriptor bytes.
    for model in MODELS {
        for code in [
            b"\x0f\x02\xc1",
            b"\x0f\x03\xc1",
            b"\x0f\x00\xe1",
            b"\x0f\x00\xe9",
        ] {
            let (vm, mut cpu) = setup(model, code);
            install_ldt(&mut cpu, 0x3000, 14);
            descriptor(&vm, 0x3008, 0, 0x2345, 0x92);
            cpu.cpu.gpr[0] = 0x1234;
            cpu.cpu.gpr[1] = 12;
            steps(&mut cpu, &vm, 1);
            assert!(!cpu.cpu.flags.zf);
            assert_eq!(cpu.cpu.gpr[0], 0x1234);
            cpu.cpu.rip = 0x1000;
            cpu.cpu.legacy.ldtr.cache.limit = 15;
            steps(&mut cpu, &vm, 1);
            assert!(cpu.cpu.flags.zf);
        }
    }
}

#[test]
fn lldt_memory_source_is_a_word_and_invalid_operand_faults_before_loading() {
    // 286 PRM App. B p. B-66; 386 PRM §17 p. 17-98.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x0f\x00\x16\xfe\xff");
        vm.write_bytes(0xfffe, &8u16.to_le_bytes()).unwrap();
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.legacy.ldtr.selector, 8);
        cpu.cpu.rip = 0x1000;
        cpu.cpu.legacy.segments[1].limit = 0xfffe;
        fault(&mut cpu, &vm, 13, Some(0));
    }
}
