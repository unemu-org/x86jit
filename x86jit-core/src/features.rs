//! Guest CPU descriptions: feature presets or complete CPUID tables. The embedder
//! selects the ISA and identity. [`CpuidOverrides`] carries logical-processor and
//! runtime fields, including an optional XCR0 value for XGETBV(0).
//!
//! Advertising does not guarantee lift coverage. The compat checker probes
//! advertised instruction features for every preset, with reasoned per-preset
//! waivers for existing gaps (including CX8 and XSAVE). Embedders can run the same
//! checker against a table; unsupported guest instructions still return an `Exit`.

/// Registers returned by CPUID, in architectural EAX/EBX/ECX/EDX order
/// (SDM Vol 2A, CPUID, "Description").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CpuidRegisters {
    pub eax: u32,
    pub ebx: u32,
    pub ecx: u32,
    pub edx: u32,
}

impl CpuidRegisters {
    pub const ZERO: Self = Self::new(0, 0, 0, 0);

    pub const fn new(eax: u32, ebx: u32, ecx: u32, edx: u32) -> Self {
        Self { eax, ebx, ecx, edx }
    }
}

/// Whether input ECX selects a subleaf. Leaf 1 ignores it; leaf 7 uses it
/// (SDM Vol 1 Tables 21-5 and 21-17). The table supplies the policy for each leaf.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CpuidLeaf {
    Constant(CpuidRegisters),
    Indexed {
        subleaves: std::collections::BTreeMap<u32, CpuidRegisters>,
        /// Reply for an unlisted subleaf; not inferred from subleaf zero's EAX.
        invalid_subleaf: CpuidRegisters,
    },
}

impl CpuidLeaf {
    fn query(&self, subleaf: u32) -> (u32, CpuidRegisters) {
        match self {
            Self::Constant(regs) => (0, *regs),
            Self::Indexed {
                subleaves,
                invalid_subleaf,
            } => (
                subleaf,
                subleaves.get(&subleaf).copied().unwrap_or(*invalid_subleaf),
            ),
        }
    }
}

/// Reply outside the ranges enumerated by leaf 0 EAX and 0x8000_0000 EAX.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CpuidOutOfRange {
    /// An embedder-selected reply, including zero for a profile that requires it.
    Fixed(CpuidRegisters),
    /// SDM Vol 1 §21.1.3 specifies the highest-basic-leaf redirect. Retaining
    /// input ECX is an engine choice; Vol 2A CPUID, Operation calls outputs
    /// for invalid leaves "Reserved".
    HighestBasicLeaf,
}

/// A complete, comparable CPU description. It replaces the feature projection;
/// unlisted leaves never fall through to [`GuestCpuFeatures`]. The embedder owns
/// consistency between feature bits, maxima, subleaf lists, and instruction support.
/// All fields are data so an embedder can serialize them or include them in a cache key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CpuidTable {
    pub leaves: std::collections::BTreeMap<u32, CpuidLeaf>,
    /// Reply for an unlisted leaf inside the enumerated ranges.
    pub missing_leaf: CpuidRegisters,
    pub out_of_range: CpuidOutOfRange,
}

impl CpuidTable {
    pub fn new(out_of_range: CpuidOutOfRange) -> Self {
        Self {
            leaves: std::collections::BTreeMap::new(),
            missing_leaf: CpuidRegisters::ZERO,
            out_of_range,
        }
    }

    pub fn query(&self, leaf: u32, subleaf: u32) -> CpuidRegisters {
        self.resolve(leaf, subleaf).2
    }

    pub(crate) fn resolve(&self, leaf: u32, subleaf: u32) -> (u32, u32, CpuidRegisters) {
        let max = |header| {
            self.leaves
                .get(&header)
                .map(|l| l.query(0).1.eax)
                .unwrap_or(0)
        };
        let max_basic = max(0);
        let max_extended = max(0x8000_0000);
        // Software-defined leaves occupy a separate range (SDM Vol 1 §21.1.3).
        let software_leaf =
            (0x4000_0000..=0x4fff_ffff).contains(&leaf) && self.leaves.contains_key(&leaf);
        let leaf = if software_leaf
            || leaf <= max_basic
            || (leaf >= 0x8000_0000 && leaf <= max_extended)
        {
            leaf
        } else {
            match self.out_of_range {
                CpuidOutOfRange::Fixed(regs) => return (leaf, subleaf, regs),
                CpuidOutOfRange::HighestBasicLeaf => max_basic,
            }
        };
        match self.leaves.get(&leaf) {
            Some(entry) => {
                let (subleaf, regs) = entry.query(subleaf);
                (leaf, subleaf, regs)
            }
            None => (leaf, subleaf, self.missing_leaf),
        }
    }
}

/// Replace only masked bits: `(base & !mask) | (value & mask)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuidOverride {
    pub mask: CpuidRegisters,
    pub value: CpuidRegisters,
}

/// Per-vCPU data applied after CPUID lookup. Keys use the resolved leaf and subleaf
/// (zero for a leaf that ignores ECX), including highest-basic-leaf redirection.
/// Initial APIC ID (leaf 1 EBX[31:24]) and x2APIC ID (0xB/0x1F EDX) belong here:
/// SDM Vol 1 §21.1.4, Tables 21-8, 21-31 and 21-70.
/// The embedder must maintain runtime fields through overrides: OSXSAVE, leaf
/// 0xD sizes, 07H ECX[4] OSPKE and 19H EBX[0] AESKLE (SDM Vol 1 §21.1.5,
/// Table 21-1), plus 80000001H EDX[11] SYSCALL (Table 21-90), when applicable
/// to its modeled mode and control state. XCR0 can be overridden alongside them.
/// The embedder updates these values when its modeled runtime state changes;
/// the core does not emulate CR4/XSETBV transitions.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CpuidOverrides {
    pub leaves: std::collections::BTreeMap<(u32, u32), CpuidOverride>,
    /// Value returned by XGETBV(0); `None` keeps the feature preset's projection.
    pub xcr0: Option<u64>,
}

impl CpuidOverrides {
    pub(crate) fn apply(&self, leaf: u32, subleaf: u32, base: CpuidRegisters) -> CpuidRegisters {
        let Some(over) = self.leaves.get(&(leaf, subleaf)) else {
            return base;
        };
        let merge = |base: u32, mask: u32, value: u32| (base & !mask) | (value & mask);
        CpuidRegisters::new(
            merge(base.eax, over.mask.eax, over.value.eax),
            merge(base.ebx, over.mask.ebx, over.value.ebx),
            merge(base.ecx, over.mask.ecx, over.value.ecx),
            merge(base.edx, over.mask.edx, over.value.edx),
        )
    }
}

/// A single guest CPU feature bit. The discriminant is the internal bit index within
/// [`GuestCpuFeatures`]; the CPUID leaf position is assigned by the projection methods.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Feature {
    // x86-64-v1 baseline SIMD.
    Sse,
    Sse2,
    Mmx,
    // x86-64-v2.
    Sse3,
    Ssse3,
    Sse41,
    Sse42,
    Popcnt,
    Cx16,
    // x86-64-v3.
    Movbe,
    Avx,
    Avx2,
    Bmi1,
    Bmi2,
    Fma,
    F16c,
    Lzcnt,
    // AVX enable plumbing (XSAVE/OSXSAVE gate XCR0 + the AVX/AVX-512 state).
    Xsave,
    Osxsave,
    // x86-64-v4 (AVX-512).
    Avx512f,
    Avx512bw,
    Avx512dq,
    Avx512vl,
    Avx512cd,
    // Crypto ISA extensions. Orthogonal to the v-levels but ubiquitous on
    // real v2+ (AES/PCLMUL) and v4-era (SHA/GFNI) hardware. Only the 128-bit forms are
    // lifted — the wide VAES/VPCLMULQDQ (leaf7 ECX bits 9/10) stay unadvertised so guests
    // pick the AES-NI/PCLMULQDQ path.
    Aes,
    Pclmul,
    Sha,
    Gfni,
    Sse4a,
    Adx,
    Rdrand,
    Rdseed,
    Rdtscp,
    Cx8,
}

impl Feature {
    #[inline]
    const fn bit(self) -> u64 {
        1u64 << (self as u8)
    }
}

/// A guest CPU feature set: a bitset over [`Feature`]. Build from a preset and refine
/// with [`GuestCpuFeatures::with`] / [`GuestCpuFeatures::without`].
///
/// ```ignore
/// let f = GuestCpuFeatures::v3().with(Feature::Avx512f); // v3 + a single v4 bit
/// vm.set_guest_cpu_features(GuestCpuFeatures::v4());
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GuestCpuFeatures(u64);

impl GuestCpuFeatures {
    /// Empty set (only the always-on scalar baseline that CPUID reports unconditionally).
    pub const fn empty() -> Self {
        GuestCpuFeatures(0)
    }

    const fn from_slice(fs: &[Feature]) -> Self {
        Self::empty().with_all(fs)
    }

    /// Union a slice of features into the set (const — used to layer each tier's delta
    /// onto the tier below it).
    const fn with_all(self, fs: &[Feature]) -> Self {
        let mut bits = self.0;
        let mut i = 0;
        while i < fs.len() {
            bits |= fs[i].bit();
            i += 1;
        }
        GuestCpuFeatures(bits)
    }

    /// x86-64-v1 baseline: MMX + SSE + SSE2 (+ scalar, always on). MMX is present on
    /// every x86-64 CPU and is load-bearing for glibc's cpu-features init (the level
    /// derivation mis-fires without it), so every preset
    /// carries it even though no MMX instruction is lifted.
    pub const fn baseline() -> Self {
        Self::from_slice(&[Feature::Mmx, Feature::Sse, Feature::Sse2, Feature::Cx8])
    }

    /// x86-64-v2: baseline + SSE3/SSSE3/SSE4.1/SSE4.2/POPCNT/CMPXCHG16B/MOVBE, plus the
    /// near-universal AES-NI + PCLMULQDQ crypto (present on essentially every
    /// v2-era CPU and load-bearing for openssl/ssh taking the hardware crypto path).
    pub const fn v2() -> Self {
        Self::baseline().with_all(&[
            Feature::Sse3,
            Feature::Ssse3,
            Feature::Sse41,
            Feature::Sse42,
            Feature::Popcnt,
            Feature::Cx16,
            Feature::Movbe,
            Feature::Aes,
            Feature::Pclmul,
        ])
    }

    /// x86-64-v3: v2 + AVX/AVX2/BMI1/BMI2/FMA/F16C/LZCNT (+ XSAVE/OSXSAVE).
    pub const fn v3() -> Self {
        Self::v2().with_all(&[
            Feature::Avx,
            Feature::Avx2,
            Feature::Bmi1,
            Feature::Bmi2,
            Feature::Fma,
            Feature::F16c,
            Feature::Lzcnt,
            Feature::Xsave,
            Feature::Osxsave,
        ])
    }

    /// x86-64-v4: v3 + AVX-512 F/BW/DQ/VL/CD, plus SHA-NI + GFNI (standard on
    /// the v4-era cores this preset models — Ice Lake / Zen 4 — and lets openssl/ssh
    /// dgst-sha256 and GFNI-accelerated codecs exercise our lifts).
    pub const fn v4() -> Self {
        Self::v3().with_all(&[
            Feature::Avx512f,
            Feature::Avx512bw,
            Feature::Avx512dq,
            Feature::Avx512vl,
            Feature::Avx512cd,
            Feature::Sha,
            Feature::Gfni,
        ])
    }

    /// The set x86jit advertises by default: SSE, SSE2, SSE3, SSSE3, POPCNT, MMX, XSAVE,
    /// OSXSAVE, AVX, AVX2 and CX8. Several advertised instruction families remain
    /// incomplete; see the per-preset compat waivers.
    ///
    /// SSE4, BMI and AVX-512 stay off here even though much of each now lifts: this preset
    /// is frozen at what was hardcoded before the feature set became embedder-selected, so
    /// an embedder that never calls `set_guest_cpu_features` sees the same guest CPU it
    /// always did. Raising the level is the caller's explicit choice — [`Self::v2`],
    /// [`Self::v3`], [`Self::v4`].
    ///
    /// MMX is a detection-only bit glibc's cpu-features init needs (waived in the compat
    /// map).
    pub const fn stable() -> Self {
        Self::from_slice(&[
            Feature::Sse,
            Feature::Sse2,
            Feature::Mmx,
            Feature::Sse3,
            Feature::Ssse3,
            Feature::Popcnt,
            Feature::Xsave,
            Feature::Osxsave,
            Feature::Avx,
            Feature::Avx2,
            Feature::Cx8,
        ])
    }

    /// Add a feature.
    pub const fn with(self, f: Feature) -> Self {
        GuestCpuFeatures(self.0 | f.bit())
    }

    /// Remove a feature.
    pub const fn without(self, f: Feature) -> Self {
        GuestCpuFeatures(self.0 & !f.bit())
    }

    /// Is a feature present?
    #[inline]
    pub const fn has(self, f: Feature) -> bool {
        self.0 & f.bit() != 0
    }

    #[inline]
    fn if_has(self, f: Feature, bit: u32) -> u32 {
        if self.has(f) {
            1 << bit
        } else {
            0
        }
    }

    // --- CPUID projections. The single place feature → leaf-bit mapping lives. ---

    /// CPUID leaf 1 ECX. Every bit is a feature (no always-on scalar bits here).
    pub fn leaf1_ecx(self) -> u32 {
        self.if_has(Feature::Sse3, 0)
            | self.if_has(Feature::Pclmul, 1)
            | self.if_has(Feature::Ssse3, 9)
            | self.if_has(Feature::Fma, 12)
            | self.if_has(Feature::Cx16, 13)
            | self.if_has(Feature::Sse41, 19)
            | self.if_has(Feature::Sse42, 20)
            | self.if_has(Feature::Movbe, 22)
            | self.if_has(Feature::Popcnt, 23)
            | self.if_has(Feature::Aes, 25)
            | self.if_has(Feature::Xsave, 26)
            | self.if_has(Feature::Osxsave, 27)
            | self.if_has(Feature::Avx, 28)
            | self.if_has(Feature::F16c, 29)
            | self.if_has(Feature::Rdrand, 30)
    }

    /// CPUID leaf 1 EDX. The always-on scalar baseline (FPU/TSC/CMOV/FXSR) plus the
    /// SSE/SSE2/MMX/CX8 feature bits.
    pub fn leaf1_edx(self) -> u32 {
        const BASELINE: u32 = (1 << 0)   // FPU
            | (1 << 4)   // TSC
            | (1 << 15)  // CMOV
            | (1 << 24); // FXSR
        BASELINE
            | self.if_has(Feature::Cx8, 8)
            | self.if_has(Feature::Mmx, 23)
            | self.if_has(Feature::Sse, 25)
            | self.if_has(Feature::Sse2, 26)
    }

    /// CPUID leaf 7 subleaf 0 EBX (AVX2, BMI1/2, AVX-512 family, SHA-NI).
    pub fn leaf7_ebx(self) -> u32 {
        self.if_has(Feature::Bmi1, 3)
            | self.if_has(Feature::Avx2, 5)
            | self.if_has(Feature::Bmi2, 8)
            | self.if_has(Feature::Avx512f, 16)
            | self.if_has(Feature::Avx512dq, 17)
            | self.if_has(Feature::Rdseed, 18)
            | self.if_has(Feature::Adx, 19)
            | self.if_has(Feature::Sha, 29)
            | self.if_has(Feature::Avx512cd, 28)
            | self.if_has(Feature::Avx512bw, 30)
            | self.if_has(Feature::Avx512vl, 31)
    }

    /// CPUID leaf 7 subleaf 0 ECX (GFNI at bit 8; the wide VAES/VPCLMULQDQ bits 9/10 stay
    /// off — we lift only the 128-bit forms, so a guest picks the AES-NI/PCLMULQDQ path).
    pub fn leaf7_ecx(self) -> u32 {
        self.if_has(Feature::Gfni, 8)
    }

    /// CPUID extended leaf 0x8000_0001 ECX (LZCNT/ABM). LAHF (bit 0) always on.
    pub fn ext_leaf1_ecx(self) -> u32 {
        (1 << 0) | self.if_has(Feature::Lzcnt, 5) | self.if_has(Feature::Sse4a, 6)
    }

    /// Extended feature flags: SDM Vol 1 Table 21-90 (RDTSCP at EDX[27],
    /// SYSCALL at EDX[11], Intel 64 at EDX[29]). The frozen feature projection
    /// sets SYSCALL in every mode, deviating from Table 21-90: Intel reports
    /// zero outside 64-bit mode. Embedders model that transition with overrides.
    pub fn ext_leaf1_edx(self) -> u32 {
        (1 << 11) | (1 << 29) | self.if_has(Feature::Rdtscp, 27)
    }

    /// Legacy feature projection. SDM Vol 1 Tables 21-4/6, 21-9/10, 21-17/19/20;
    /// AMD APM Vol 3 §E.4.1–E.4.2 for extended leaves.
    /// Unlisted leaves return zero, deliberately deviating from the highest-basic-
    /// leaf rule in SDM Vol 1 §21.1.3 to preserve the frozen default profile.
    pub fn cpuid(self, leaf: u32, subleaf: u32) -> CpuidRegisters {
        match leaf {
            0 => CpuidRegisters::new(7, 0x756e_6547, 0x6c65_746e, 0x4965_6e69),
            1 => CpuidRegisters::new(0x0003_06c3, 0, self.leaf1_ecx(), self.leaf1_edx()),
            7 if subleaf == 0 => CpuidRegisters::new(0, self.leaf7_ebx(), self.leaf7_ecx(), 0),
            0x8000_0000 => CpuidRegisters::new(0x8000_0001, 0, 0, 0),
            0x8000_0001 => CpuidRegisters::new(0, 0, self.ext_leaf1_ecx(), self.ext_leaf1_edx()),
            _ => CpuidRegisters::ZERO,
        }
    }

    /// XCR0 value returned by `xgetbv` (ECX=0). x87|SSE always; +AVX (bit 2) when AVX is
    /// enabled; +opmask|ZMM_hi|hi16_ZMM (bits 5..7) when AVX-512 is enabled.
    pub fn xcr0(self) -> u64 {
        let mut x = 0b011; // x87 | SSE
        if self.has(Feature::Avx) {
            x |= 0b100; // AVX state
        }
        if self.has(Feature::Avx512f) {
            x |= 0b1110_0000; // opmask | ZMM_hi256 | hi16_ZMM
        }
        x
    }
}

impl Default for GuestCpuFeatures {
    /// The advertised set for every embedder that doesn't call
    /// `set_guest_cpu_features` — see [`GuestCpuFeatures::stable`].
    fn default() -> Self {
        Self::stable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Witness test** for every CPUID feature-bit position this file encodes.
    ///
    /// Source: Intel SDM Vol 1 Tables 21-9 (leaf 1 ECX), 21-10 (leaf 1 EDX),
    /// 21-19/21-20 (leaf 7 subleaf 0 EBX/ECX); and AMD64 APM Vol 3 Appendix E
    /// §E.4.2 (extended leaf `8000_0001` ECX/EDX), which is where `LZCNT`/`ABM` is defined —
    /// Intel documents the same bit as `LZCNT` under leaf `8000_0001` ECX bit 5.
    /// `oracles/intel/sdm.pdf`, `oracles/intel/amd64-apm.pdf`.
    ///
    /// The point of stating the table again here, rather than trusting the `if_has(f, n)`
    /// calls above, is that a citation nobody can check is not a citation. Each row asserts
    /// that a set containing *exactly* one feature projects to *exactly* the documented bit,
    /// so a transposed digit fails instead of silently advertising the wrong thing to a
    /// guest's IFUNC resolver — which is the kind of bug that surfaces as "glibc picked a
    /// string routine we never lifted", a very long way from its cause.
    ///
    /// This runs with no stash and no network: the values are pinned here by value.
    #[test]
    fn cpuid_feature_bits_match_the_documented_positions() {
        // (feature, leaf projection, documented bit)
        let leaf1_ecx: &[(Feature, u32)] = &[
            (Feature::Sse3, 0),
            (Feature::Pclmul, 1),
            (Feature::Ssse3, 9),
            (Feature::Fma, 12),
            (Feature::Cx16, 13),
            (Feature::Sse41, 19),
            (Feature::Sse42, 20),
            (Feature::Movbe, 22),
            (Feature::Popcnt, 23),
            (Feature::Aes, 25),
            (Feature::Xsave, 26),
            (Feature::Osxsave, 27),
            (Feature::Avx, 28),
            (Feature::F16c, 29),
            (Feature::Rdrand, 30),
        ];
        for &(f, bit) in leaf1_ecx {
            let got = GuestCpuFeatures::empty().with(f).leaf1_ecx();
            assert_eq!(got, 1 << bit, "leaf 1 ECX bit for {f:?}: got {got:#x}");
        }

        // Leaf 1 EDX carries an always-on scalar baseline as well as feature bits, so the
        // assertion is against the baseline rather than against zero.
        const EDX_BASELINE: u32 = (1 << 0)    // FPU
            | (1 << 4)                        // TSC
            | (1 << 15)                       // CMOV
            | (1 << 24); // FXSR
        assert_eq!(
            GuestCpuFeatures::empty().leaf1_edx(),
            EDX_BASELINE,
            "leaf 1 EDX baseline (FPU/TSC/CMOV/FXSR) — SDM Vol 1 Table 21-10"
        );
        for &(f, bit) in &[
            (Feature::Cx8, 8u32),
            (Feature::Mmx, 23),
            (Feature::Sse, 25),
            (Feature::Sse2, 26),
        ] {
            let got = GuestCpuFeatures::empty().with(f).leaf1_edx();
            assert_eq!(
                got,
                EDX_BASELINE | (1 << bit),
                "leaf 1 EDX bit for {f:?}: got {got:#x}"
            );
        }

        for &(f, bit) in &[
            (Feature::Bmi1, 3u32),
            (Feature::Avx2, 5),
            (Feature::Bmi2, 8),
            (Feature::Avx512f, 16),
            (Feature::Avx512dq, 17),
            (Feature::Rdseed, 18),
            (Feature::Adx, 19),
            (Feature::Avx512cd, 28),
            (Feature::Sha, 29),
            (Feature::Avx512bw, 30),
            (Feature::Avx512vl, 31),
        ] {
            let got = GuestCpuFeatures::empty().with(f).leaf7_ebx();
            assert_eq!(got, 1 << bit, "leaf 7 EBX bit for {f:?}: got {got:#x}");
        }

        // AMD APM Vol 3 §E.4.2, Fn8000_0001_ECX/EDX.
        for (f, bit) in [(Feature::Lzcnt, 5), (Feature::Sse4a, 6)] {
            assert_eq!(
                GuestCpuFeatures::empty().with(f).ext_leaf1_ecx(),
                1 | (1 << bit)
            );
        }
        assert_eq!(
            GuestCpuFeatures::empty()
                .with(Feature::Rdtscp)
                .ext_leaf1_edx(),
            (1 << 11) | (1 << 27) | (1 << 29)
        );

        // GFNI is bit 8. Bits 9 and 10 (VAES, VPCLMULQDQ — the *wide* forms) must stay
        // clear: only the 128-bit forms are lifted, so advertising the wide ones would
        // steer a guest onto a path that traps. Deliberate, and load-bearing.
        assert_eq!(
            GuestCpuFeatures::empty().with(Feature::Gfni).leaf7_ecx(),
            1 << 8,
            "leaf 7 ECX GFNI bit"
        );
        for f in [Feature::Aes, Feature::Pclmul] {
            let ecx = GuestCpuFeatures::empty().with(f).leaf7_ecx();
            assert_eq!(
                ecx & ((1 << 9) | (1 << 10)),
                0,
                "wide VAES/VPCLMULQDQ must stay off (only the 128-bit forms lift)"
            );
        }
    }

    #[test]
    fn the_default_preset_projects_the_exact_frozen_cpuid_bits() {
        let f = GuestCpuFeatures::default();
        assert_eq!(
            f.leaf1_ecx(),
            (1 << 0) | (1 << 9) | (1 << 23) | (1 << 26) | (1 << 27) | (1 << 28)
        );
        assert_eq!(
            f.leaf1_edx(),
            (1 << 0)
                | (1 << 4)
                | (1 << 8)
                | (1 << 15)
                | (1 << 23)
                | (1 << 24)
                | (1 << 25)
                | (1 << 26)
        );
        assert_eq!(f.leaf7_ebx(), 1 << 5); // AVX2 only
        assert_eq!(f.ext_leaf1_ecx(), 1 << 0); // LAHF, no LZCNT
        assert_eq!(f.xcr0(), 0x7); // x87|SSE|AVX
    }

    #[test]
    fn v4_advertises_avx512_and_wide_xcr0() {
        let f = GuestCpuFeatures::v4();
        // leaf7 EBX: AVX2(5) + AVX512 F(16)/DQ(17)/CD(28)/BW(30)/VL(31) + BMI1(3)/BMI2(8).
        assert!(f.leaf7_ebx() & (1 << 16) != 0, "AVX512F");
        assert!(f.leaf7_ebx() & (1 << 30) != 0, "AVX512BW");
        assert!(f.leaf7_ebx() & (1 << 31) != 0, "AVX512VL");
        assert_eq!(f.xcr0(), 0xE7); // + opmask|ZMM_hi|hi16_ZMM
                                    // leaf1 ECX gains SSE4.1/4.2/FMA/F16C/MOVBE.
        assert!(f.leaf1_ecx() & (1 << 20) != 0, "SSE4.2");
    }

    #[test]
    fn baseline_has_no_avx() {
        let f = GuestCpuFeatures::baseline();
        assert_eq!(f.leaf1_ecx() & (1 << 28), 0); // no AVX
        assert_eq!(f.leaf7_ebx(), 0); // no AVX2/AVX-512
        assert_eq!(f.xcr0(), 0x3); // x87|SSE only
    }

    #[test]
    fn with_without_toggle() {
        let f = GuestCpuFeatures::v3().with(Feature::Avx512f);
        assert!(f.has(Feature::Avx512f));
        assert!(f.has(Feature::Avx2));
        assert!(!f.without(Feature::Avx2).has(Feature::Avx2));
    }
}
