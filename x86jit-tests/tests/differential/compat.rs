//! ISA compatibility map enforcement: the checked-in map cannot rot. Adding a lift arm
//! without refreshing `backlog/docs/compat/coverage.json` fails this test — the map is
//! always the truth about what the lifter handles.

use x86jit_tests::compat::{compute_coverage, cpuid_lift_gaps, cpuid_waivers, Coverage};

#[test]
fn optional_extensions_have_complete_coverage() {
    let coverage = compute_coverage();
    for (mode, buckets, forms) in [
        ("long64", &coverage.generations, 26),
        ("compat32", &coverage.compat32, 18),
    ] {
        let optional = buckets.get("optional").expect("optional bucket present");
        assert_eq!(
            optional.lifted, forms,
            "{mode}: RDTSCP, RDRAND/RDSEED, SSE4A, ADX and user XSAVE forms"
        );
        assert_eq!(optional.unsupported, 0, "{mode}");
        assert_eq!(optional.unencodable, 0, "{mode}");
        assert_eq!(optional.reg_only, 0, "{mode}");
    }
}

#[test]
fn compat_map_is_current() {
    let fresh = compute_coverage().to_json();
    let checked_in = std::fs::read_to_string(
        x86jit_tests::compat::artifact_dir().join("coverage.json"),
    )
    .expect("backlog/docs/compat/coverage.json missing — run: cargo run -p x86jit-tests --bin compat -- --write");

    assert_eq!(
        fresh, checked_in,
        "ISA compat map is stale. The lifter's coverage changed but \
         backlog/docs/compat/coverage.json wasn't regenerated. Run:\n  \
         cargo run -p x86jit-tests --bin compat -- --write\nand commit the result."
    );
}

fn presets() -> [(&'static str, x86jit_core::GuestCpuFeatures); 4] {
    use x86jit_core::GuestCpuFeatures as F;
    [
        ("default", F::default()),
        ("v2", F::v2()),
        ("v3", F::v3()),
        ("v4", F::v4()),
    ]
}

/// Every preset's advertised instruction-feature bits must lift or carry an
/// explicit per-preset waiver. This includes unknown (unencodable) probe coverage.
#[test]
fn cpuid_advertises_only_what_lifts() {
    let waivers = cpuid_waivers();
    let mut failures = Vec::new();
    for (preset, features) in presets() {
        for gap in cpuid_lift_gaps(|l, s| features.cpuid(l, s)) {
            let name = format!("{:?}", gap.advertised.feature);
            if !waivers
                .iter()
                .any(|(p, f, reason)| p == preset && f == &name && !reason.trim().is_empty())
            {
                failures.push(format!(
                    "{preset}/{name}: {} missing, {} unencodable, e.g. {:?} / {:?}",
                    gap.missing.len(),
                    gap.unencodable.len(),
                    &gap.missing[..gap.missing.len().min(5)],
                    &gap.unencodable[..gap.unencodable.len().min(5)]
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "Unwaived CPUID lift gaps:\n{}",
        failures.join("\n")
    );
}

#[test]
fn cpuid_waivers_are_not_stale() {
    let waivers = cpuid_waivers();
    let mut seen = std::collections::HashSet::new();
    for (preset, feature, reason) in &waivers {
        assert!(
            seen.insert((preset, feature)),
            "duplicate waiver: {preset}/{feature}"
        );
        assert!(
            !reason.trim().is_empty(),
            "empty waiver: {preset}/{feature}"
        );
        let (_, features) = presets()
            .into_iter()
            .find(|(p, _)| p == preset)
            .unwrap_or_else(|| panic!("unknown preset {preset}"));
        assert!(
            cpuid_lift_gaps(|l, s| features.cpuid(l, s))
                .iter()
                .any(|g| format!("{:?}", g.advertised.feature) == *feature),
            "stale waiver: {preset}/{feature}"
        );
    }
}

/// Regression for gaps the old generation buckets and leaf-1-only check hid.
#[test]
fn cpuid_checker_accepts_scalar_and_xsave_lifts() {
    use iced_x86::{Code, CpuidFeature};
    use x86jit_tests::compat::{probe_code, Probe};
    for (name, features) in presets() {
        let gaps = cpuid_lift_gaps(|l, s| features.cpuid(l, s));
        assert!(
            gaps.iter().all(|g| !matches!(
                g.advertised.feature,
                CpuidFeature::CX8 | CpuidFeature::CMPXCHG16B
            )),
            "{name}/paired CAS"
        );
        assert_eq!(probe_code(Code::Cmpxchg8b_m64), Some(Probe::Lifted));
        assert_eq!(probe_code(Code::Cmpxchg16b_m128), Some(Probe::Lifted));
        assert!(
            gaps.iter()
                .all(|g| g.advertised.feature != CpuidFeature::XSAVE),
            "{name}/XSAVE"
        );
        for code in [
            Code::Xsave_mem,
            Code::Xsave64_mem,
            Code::Xrstor_mem,
            Code::Xrstor64_mem,
            Code::Xsaveopt_mem,
            Code::Xsaveopt64_mem,
            Code::Xsavec_mem,
            Code::Xsavec64_mem,
        ] {
            assert_eq!(probe_code(code), Some(Probe::Lifted));
        }
    }
}

#[test]
fn cpuid_checker_accepts_tables_and_sees_extended_and_leaf7_bits() {
    use iced_x86::CpuidFeature::*;
    use x86jit_core::{CpuidLeaf, CpuidOutOfRange, CpuidRegisters as R, CpuidTable};
    let mut table = CpuidTable::new(CpuidOutOfRange::HighestBasicLeaf);
    table
        .leaves
        .insert(0, CpuidLeaf::Constant(R::new(7, 0, 0, 0)));
    table.leaves.insert(
        0x8000_0000,
        CpuidLeaf::Constant(R::new(0x8000_0001, 0, 0, 0)),
    );
    table.leaves.insert(
        1,
        CpuidLeaf::Constant(R::new(0, 0, (1 << 30) | (1 << 28), 1 << 8)),
    );
    table.leaves.insert(
        7,
        CpuidLeaf::Indexed {
            subleaves: [
                (0, R::new(1, (1 << 5) | (1 << 18) | (1 << 19), 0, 0)),
                (1, R::new(1, 0, 0, 0)),
            ]
            .into(),
            invalid_subleaf: R::ZERO,
        },
    );
    table.leaves.insert(
        0x8000_0001,
        CpuidLeaf::Constant(R::new(0, 0, 1 << 6, 1 << 27)),
    );
    let gaps = cpuid_lift_gaps(|l, s| table.query(l, s));
    assert!(gaps.iter().any(|g| g.advertised.feature == SHA512));
    for feature in [CX8, RDRAND, RDSEED, RDTSCP, ADX, SSE4A] {
        assert!(
            !gaps.iter().any(|g| g.advertised.feature == feature),
            "{feature:?}"
        );
    }
    // The highest-basic-leaf reply must not be mistaken for feature bits in 0xD.
    assert!(!gaps.iter().any(|g| g.advertised.leaf == 0xd));
}

#[test]
fn cpuid_checker_requires_all_known_instruction_features() {
    use iced_x86::CpuidFeature;
    use x86jit_core::{Feature, GuestCpuFeatures};
    let vl_only = GuestCpuFeatures::empty().with(Feature::Avx512vl);
    let gaps = cpuid_lift_gaps(|l, s| vl_only.cpuid(l, s));
    assert!(
        !gaps
            .iter()
            .any(|g| g.advertised.feature == CpuidFeature::AVX512VL),
        "AVX512VL instructions also require unadvertised foundation features"
    );
    let with_foundation = vl_only.with(Feature::Avx512f);
    assert!(cpuid_lift_gaps(|l, s| with_foundation.cpuid(l, s))
        .iter()
        .any(|g| g.advertised.feature == CpuidFeature::AVX512VL));
}

#[test]
fn cpuid_checker_preserves_disjunctive_requirements() {
    use iced_x86::{Code, CpuidFeature::*};
    use x86jit_core::CpuidRegisters as R;
    use x86jit_tests::compat::advertised_instruction_features;
    for (leaf, register, bits, alternative) in [
        (7, "ebx", [4, 11], HLE_or_RTM),
        (0x8000_0001, "ecx", [2, 12], SKINIT_or_SVM),
    ] {
        for enabled in [0, 1 << bits[0], 1 << bits[1]] {
            let source = |l, _| match l {
                0 => R::new(7, 0, 0, 0),
                0x8000_0000 => R::new(0x8000_0001, 0, 0, 0),
                l if l == leaf && register == "ebx" => R::new(0, enabled, 0, 0),
                l if l == leaf => R::new(0, 0, enabled, 0),
                _ => R::ZERO,
            };
            assert_eq!(
                advertised_instruction_features(source)
                    .iter()
                    .any(|f| f.feature == alternative),
                enabled != 0
            );
            if alternative == HLE_or_RTM {
                assert_eq!(
                    cpuid_lift_gaps(source)
                        .iter()
                        .any(|g| g.advertised.feature == alternative
                            && g.missing.contains(&Code::Xtest)),
                    enabled != 0
                );
            }
        }
    }
}

/// Sanity: the probe actually measures something (guards against a silently-broken
/// probe reporting everything as unencodable, which would make the map meaningless).
#[test]
fn probe_measures_real_coverage() {
    let cov: Coverage = Coverage::load_checked_in()
        .expect("coverage.json missing — run the compat bin with --write");
    let v1 = cov.generations.get("x86-64-v1").expect("v1 row present");
    assert!(
        v1.lifted > 100,
        "v1 baseline should have many lifted instructions, got {}",
        v1.lifted
    );
    // The 32-bit compat map is measured too, and it sees the forms long
    // mode can't encode: legacy-only + 16-bit operand-size codes appear in its gap
    // list until they're lifted (trap-and-fix).
    let c32 = cov
        .compat32
        .get("x86-64-v1")
        .expect("compat32 v1 row present");
    assert!(
        c32.lifted > 100,
        "compat32 v1 should have many lifted instructions, got {}",
        c32.lifted
    );
    assert!(
        c32.missing.iter().any(|m| m == "Pushad"),
        "compat32 map should list the legacy-only forms (e.g. Pushad) as missing"
    );
}

#[test]
fn cpu_source_instructions_are_in_compat_scope_and_lift() {
    use iced_x86::Code::*;
    use x86jit_tests::compat::{probe_code, Probe};
    for code in [
        Rdtscp, Rdrand_r16, Rdrand_r32, Rdrand_r64, Rdseed_r16, Rdseed_r32, Rdseed_r64,
    ] {
        assert_eq!(probe_code(code), Some(Probe::Lifted), "{code:?}");
    }
}

/// Witness rows from SDM Vol 1 Tables 21-9/10/19/20/21/22/23/25,
/// 21-35/47/51/62; AMD APM Vol 3 §E.4.2 and §E.4.7. Pin positions
/// independently of the reader so a swapped bit cannot silently change the audit.
#[test]
fn cpuid_checker_feature_positions_match_manual() {
    use iced_x86::CpuidFeature::*;
    use x86jit_core::CpuidRegisters as R;
    use x86jit_tests::compat::advertised_instruction_features;
    #[rustfmt::skip]
    let witnesses = [
        (1, 0, "ecx", 0, SSE3),
        (1, 0, "ecx", 1, PCLMULQDQ),
        (1, 0, "ecx", 3, MONITOR),
        (1, 0, "ecx", 5, VMX),
        (1, 0, "ecx", 6, SMX),
        (1, 0, "ecx", 9, SSSE3),
        (1, 0, "ecx", 12, FMA),
        (1, 0, "ecx", 13, CMPXCHG16B),
        (1, 0, "ecx", 19, SSE4_1),
        (1, 0, "ecx", 20, SSE4_2),
        (1, 0, "ecx", 22, MOVBE),
        (1, 0, "ecx", 23, POPCNT),
        (1, 0, "ecx", 25, AES),
        (1, 0, "ecx", 26, XSAVE),
        (1, 0, "ecx", 28, AVX),
        (1, 0, "ecx", 29, F16C),
        (1, 0, "ecx", 30, RDRAND),
        (1, 0, "edx", 0, FPU),
        (1, 0, "edx", 4, TSC),
        (1, 0, "edx", 5, MSR),
        (1, 0, "edx", 8, CX8),
        (1, 0, "edx", 11, SEP),
        (1, 0, "edx", 15, CMOV),
        (1, 0, "edx", 19, CLFSH),
        (1, 0, "edx", 23, MMX),
        (1, 0, "edx", 24, FXSR),
        (1, 0, "edx", 25, SSE),
        (1, 0, "edx", 26, SSE2),
        (7, 0, "ebx", 0, FSGSBASE),
        (7, 0, "ebx", 3, BMI1),
        (7, 0, "ebx", 4, HLE),
        (7, 0, "ebx", 5, AVX2),
        (7, 0, "ebx", 8, BMI2),
        (7, 0, "ebx", 10, INVPCID),
        (7, 0, "ebx", 11, RTM),
        (7, 0, "ebx", 14, MPX),
        (7, 0, "ebx", 16, AVX512F),
        (7, 0, "ebx", 17, AVX512DQ),
        (7, 0, "ebx", 18, RDSEED),
        (7, 0, "ebx", 19, ADX),
        (7, 0, "ebx", 20, SMAP),
        (7, 0, "ebx", 21, AVX512_IFMA),
        (7, 0, "ebx", 23, CLFLUSHOPT),
        (7, 0, "ebx", 24, CLWB),
        (7, 0, "ebx", 26, AVX512PF),
        (7, 0, "ebx", 27, AVX512ER),
        (7, 0, "ebx", 28, AVX512CD),
        (7, 0, "ebx", 29, SHA),
        (7, 0, "ebx", 30, AVX512BW),
        (7, 0, "ebx", 31, AVX512VL),
        (7, 0, "ecx", 0, PREFETCHWT1),
        (7, 0, "ecx", 1, AVX512_VBMI),
        (7, 0, "ecx", 3, PKU),
        (7, 0, "ecx", 5, WAITPKG),
        (7, 0, "ecx", 6, AVX512_VBMI2),
        (7, 0, "ecx", 7, CET_SS),
        (7, 0, "ecx", 8, GFNI),
        (7, 0, "ecx", 9, VAES),
        (7, 0, "ecx", 10, VPCLMULQDQ),
        (7, 0, "ecx", 11, AVX512_VNNI),
        (7, 0, "ecx", 12, AVX512_BITALG),
        (7, 0, "ecx", 14, AVX512_VPOPCNTDQ),
        (7, 0, "ecx", 22, RDPID),
        (7, 0, "ecx", 23, KL),
        (7, 0, "ecx", 25, CLDEMOTE),
        (7, 0, "ecx", 27, MOVDIRI),
        (7, 0, "ecx", 28, MOVDIR64B),
        (7, 0, "ecx", 29, ENQCMD),
        (7, 0, "edx", 2, AVX512_4VNNIW),
        (7, 0, "edx", 3, AVX512_4FMAPS),
        (7, 0, "edx", 5, UINTR),
        (7, 0, "edx", 8, AVX512_VP2INTERSECT),
        (7, 0, "edx", 14, SERIALIZE),
        (7, 0, "edx", 16, TSXLDTRK),
        (7, 0, "edx", 18, PCONFIG),
        (7, 0, "edx", 20, CET_IBT),
        (7, 0, "edx", 22, AMX_BF16),
        (7, 0, "edx", 23, AVX512_FP16),
        (7, 0, "edx", 24, AMX_TILE),
        (7, 0, "edx", 25, AMX_INT8),
        (7, 1, "eax", 0, SHA512),
        (7, 1, "eax", 1, SM3),
        (7, 1, "eax", 2, SM4),
        (7, 1, "eax", 4, AVX_VNNI),
        (7, 1, "eax", 5, AVX512_BF16),
        (7, 1, "eax", 7, CMPCCXADD),
        (7, 1, "eax", 17, FRED),
        (7, 1, "eax", 18, LKGS),
        (7, 1, "eax", 19, WRMSRNS),
        (7, 1, "eax", 21, AMX_FP16),
        (7, 1, "eax", 22, HRESET),
        (7, 1, "eax", 23, AVX_IFMA),
        (7, 1, "eax", 27, MSRLIST),
        (7, 1, "ebx", 1, TSE),
        (7, 1, "edx", 4, AVX_VNNI_INT8),
        (7, 1, "edx", 5, AVX_NE_CONVERT),
        (7, 1, "edx", 8, AMX_COMPLEX),
        (7, 1, "edx", 10, AVX_VNNI_INT16),
        (7, 1, "edx", 14, PREFETCHITI),
        (0xd, 1, "eax", 0, XSAVEOPT),
        (0xd, 1, "eax", 1, XSAVEC),
        (0xd, 1, "eax", 3, XSAVES),
        (0x12, 0, "eax", 0, SGX1),
        (0x14, 0, "ebx", 4, PTWRITE),
        (0x19, 0, "ebx", 0, AESKLE),
        (0x19, 0, "ebx", 2, WIDE_KL),
        (0x8000_0001, 0, "ecx", 2, SVM),
        (0x8000_0001, 0, "ecx", 5, LZCNT),
        (0x8000_0001, 0, "ecx", 6, SSE4A),
        (0x8000_0001, 0, "ecx", 8, PREFETCHW),
        (0x8000_0001, 0, "ecx", 11, XOP),
        (0x8000_0001, 0, "ecx", 12, SKINIT),
        (0x8000_0001, 0, "ecx", 15, LWP),
        (0x8000_0001, 0, "ecx", 16, FMA4),
        (0x8000_0001, 0, "ecx", 21, TBM),
        (0x8000_0001, 0, "ecx", 29, MONITORX),
        (0x8000_0001, 0, "edx", 0, FPU),
        (0x8000_0001, 0, "edx", 4, TSC),
        (0x8000_0001, 0, "edx", 5, MSR),
        (0x8000_0001, 0, "edx", 8, CX8),
        (0x8000_0001, 0, "edx", 11, SYSCALL),
        (0x8000_0001, 0, "edx", 15, CMOV),
        (0x8000_0001, 0, "edx", 23, MMX),
        (0x8000_0001, 0, "edx", 24, FXSR),
        (0x8000_0001, 0, "edx", 27, RDTSCP),
        (0x8000_0001, 0, "edx", 29, X64),
        (0x8000_0001, 0, "edx", 30, D3NOWEXT),
        (0x8000_0001, 0, "edx", 31, D3NOW),
        (0x8000_0008, 0, "ebx", 0, CLZERO),
        (0x8000_0008, 0, "ebx", 3, INVLPGB),
        (0x8000_0008, 0, "ebx", 4, RDPRU),
        (0x8000_0008, 0, "ebx", 8, MCOMMIT),
        (0x8000_0008, 0, "ebx", 9, WBNOINVD),
    ];
    for (leaf, subleaf, register, bit, feature) in witnesses {
        let found = advertised_instruction_features(|l, s| {
            if l == 0 {
                return R::new(0x1f, 0, 0, 0);
            }
            if l == 0x8000_0000 {
                return R::new(0x8000_0008, 0, 0, 0);
            }
            let mut regs = if l == 7 && s == 0 {
                R::new(2, 0, 0, 0)
            } else {
                R::ZERO
            };
            if (l, s) == (leaf, subleaf) {
                match register {
                    "eax" => regs.eax |= 1 << bit,
                    "ebx" => regs.ebx |= 1 << bit,
                    "ecx" => regs.ecx |= 1 << bit,
                    "edx" => regs.edx |= 1 << bit,
                    _ => unreachable!(),
                }
            }
            regs
        });
        assert!(
            found
                .iter()
                .any(|f| (f.leaf, f.subleaf, f.register, f.bit, f.feature)
                    == (leaf, subleaf, register, bit, feature)),
            "{feature:?} at {leaf:#x}:{subleaf} {register}[{bit}]"
        );
    }
}

#[test]
fn user_flag_and_selector_forms_are_measured() {
    use iced_x86::Code;
    use x86jit_core::CpuMode;
    use x86jit_tests::compat::{probe_code_in, Probe};
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for code in [
            Code::Clc,
            Code::Stc,
            Code::Cmc,
            Code::Pushfw,
            Code::Popfw,
            Code::Mov_rm16_Sreg,
            Code::Mov_r32m16_Sreg,
        ] {
            assert_eq!(
                probe_code_in(code, mode),
                Some(Probe::Lifted),
                "{mode:?}/{code:?}"
            );
        }
        assert_eq!(
            probe_code_in(Code::Mov_Sreg_rm16, mode),
            Some(Probe::Unsupported)
        );
    }
    for code in [
        Code::Pushfq,
        Code::Popfq,
        Code::Iretq,
        Code::Mov_r64m16_Sreg,
    ] {
        assert_eq!(
            probe_code_in(code, CpuMode::Long64),
            Some(Probe::Lifted),
            "{code:?}"
        );
    }
    for code in [Code::Pushfd, Code::Popfd, Code::Iretd] {
        assert_eq!(
            probe_code_in(code, CpuMode::Compat32),
            Some(Probe::Lifted),
            "{code:?}"
        );
    }
}

#[test]
fn cpuid_sse4a_adx_opt_in_has_complete_lift_coverage() {
    use x86jit_core::{Feature, GuestCpuFeatures};
    // AMD APM Vol 3 §E.4.2: SSE4A is Fn8000_0001_ECX[6].
    // SDM Vol 2A §3.3 ADCX/ADOX: ADX is CPUID.07H.00H:EBX[19].
    for preset in [
        GuestCpuFeatures::baseline(),
        GuestCpuFeatures::v2(),
        GuestCpuFeatures::v3(),
        GuestCpuFeatures::v4(),
        GuestCpuFeatures::default(),
    ] {
        assert!(!preset.has(Feature::Sse4a));
        assert!(!preset.has(Feature::Adx));
    }
    let profile = GuestCpuFeatures::empty()
        .with(Feature::Sse4a)
        .with(Feature::Adx);
    assert_eq!(profile.cpuid(0x8000_0001, 0).ecx & (1 << 6), 1 << 6);
    assert_eq!(profile.cpuid(7, 0).ebx & (1 << 19), 1 << 19);
    let gaps = cpuid_lift_gaps(|leaf, subleaf| profile.cpuid(leaf, subleaf));
    assert!(!gaps.iter().any(|gap| matches!(
        gap.advertised.feature,
        iced_x86::CpuidFeature::SSE4A | iced_x86::CpuidFeature::ADX
    )));
    // Isolate the opt-in bits from the preset's existing scalar baseline waivers.
    assert!(cpuid_lift_gaps(|leaf, subleaf| match leaf {
        7 | 0x8000_0001 => profile.cpuid(leaf, subleaf),
        _ => x86jit_core::CpuidRegisters::ZERO,
    })
    .is_empty());
    for mode in [x86jit_core::CpuMode::Long64, x86jit_core::CpuMode::Compat32] {
        for code in iced_x86::Code::values().filter(|code| {
            let valid = if mode == x86jit_core::CpuMode::Long64 {
                code.op_code().mode64()
            } else {
                code.op_code().mode32()
            };
            valid
                && code.cpuid_features().iter().any(|f| {
                    matches!(
                        f,
                        iced_x86::CpuidFeature::SSE4A | iced_x86::CpuidFeature::ADX
                    )
                })
        }) {
            assert_eq!(
                x86jit_tests::compat::probe_code_in(code, mode),
                Some(x86jit_tests::compat::Probe::Lifted),
                "{mode:?}/{code:?}"
            );
        }
    }
}

#[cfg(debug_assertions)]
#[test]
fn every_probed_block_defines_temps_before_use() {
    let mut failures = Vec::new();
    for mode in [
        x86jit_core::CpuMode::Real16,
        x86jit_core::CpuMode::Compat32,
        x86jit_core::CpuMode::Long64,
    ] {
        match x86jit_tests::compat::verify_probe_temp_defs_in(mode) {
            Ok(checked) => {
                assert!(checked > 1000, "{mode:?}: only {checked} blocks checked");
                eprintln!("{mode:?}: verified {checked} probe blocks");
            }
            Err(errors) => failures.extend(errors),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
