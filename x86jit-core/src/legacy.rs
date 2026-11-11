//! Legacy execution-context and inline-state contract (decision-18 §§1–2;
//! spec.md §§3, 17). Execution support is supplied separately.
use crate::{CpuMode, CpuState};

/// Fixed application contexts and explicitly opted-in system execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionProfile {
    Application(CpuMode),
    LegacySystem,
}

/// Generation is independent of decoder width and CPUID enumeration.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CpuGeneration {
    I8086,
    I286,
    I386,
}

/// Shared model-profile boundary. The embedder supplies a stable identity for
/// all model policies; the CPU-model catalogue owns presets, not this module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegacyCpuModel {
    pub generation: CpuGeneration,
    pub model_semantics_id: u64,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LegacyRegime {
    Real,
    Protected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegacyDecodeContext {
    /// Instruction-generation policy, independent of decoder width.
    pub generation: CpuGeneration,
    /// Real or protected execution regime derived from the retained CPU state.
    pub regime: LegacyRegime,
    /// Default operand width (16 or 32 bits) before instruction prefixes.
    pub default_operand_bits: u8,
    /// Default effective-address width (16 or 32 bits) before instruction prefixes.
    pub default_address_bits: u8,
    /// Stack-pointer width (16 or 32 bits), selected independently of code defaults.
    pub stack_address_bits: u8,
    /// Current privilege level (0–3), validated against the execution regime.
    pub cpl: u8,
    /// Whether the retained control state enables paging; no translation is implied.
    pub paging_enabled: bool,
}

/// Invalid storage or a context outside the selected generation/engine scope.
/// These are configuration refusals, not synthesized guest exceptions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LegacyContextError {
    ProtectedModeRequires286,
    PagingRequires386,
    PagingRequiresProtectedMode,
    SizeRequires386,
    InvalidPrivilege,
    SelectorPrivilegeMismatch,
    InvalidStorage,
    UnusableCodeOrStack,
    LongModeUnsupported,
    Virtual8086Unsupported,
    DebugTrapUnsupported,
}

/// Engine layout, not the packed guest table-register operand image.
/// SDM Vol 3A §2.4: table registers retain base and limit.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TableRegister {
    pub base: u32,
    pub limit: u16,
    pub reserved: u16,
}

/// SDM Vol 3A §§3.4.3–3.4.5: hidden base, expanded inclusive byte limit,
/// descriptor access byte (7:0), and AVL/L/D-or-B/G flags (12:15).
/// Bits 11:8 and `reserved` must be zero; `unusable` is an engine marker.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentCache {
    pub base: u32,
    pub limit: u32,
    pub attributes: u16,
    pub unusable: u8,
    pub reserved: u8,
}

/// SDM Vol 3A §§2.4, 3.5.1, 10.2.4: LDTR/TR retain selector and cache.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SystemSegment {
    pub selector: u16,
    pub reserved: u16,
    pub cache: SegmentCache,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentId {
    Cs,
    Ds,
    Es,
    Ss,
    Fs,
    Gs,
}

/// Reserved inline state; Application execution never consumes these values.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LegacyCpuState {
    pub cr0: u32,
    pub cr2: u32,
    pub cr3: u32,
    pub gdtr: TableRegister,
    pub idtr: TableRegister,
    pub ldtr: SystemSegment,
    pub tr: SystemSegment,
    pub segments: [SegmentCache; 6],
    pub cpl: u8,
    pub a20_enabled: u8,
    pub reset_fetch: u8,
    pub reserved: u8,
}

/// SDM Vol 3A §2.5: PE is bit 0, PG is bit 31; PG requires PE.
/// SDM Vol 3A §3.4.5: CS.D and SS.B independently select default sizes.
/// Retained attributes follow §§12.9.1–12.9.2 and AMD64 APM Vol 2 §§4.1, 14.6.
/// No caches are rebuilt or model-specific CR0 masks inferred here.
pub fn derive_legacy_context(
    model: LegacyCpuModel,
    cpu: &CpuState,
) -> Result<LegacyDecodeContext, LegacyContextError> {
    use LegacyContextError::*;
    let state = &cpu.legacy;
    let protected = state.cr0 & 1 != 0;
    let paging = state.cr0 & (1 << 31) != 0;
    let cs = state.segments[SegmentId::Cs as usize];
    let ss = state.segments[SegmentId::Ss as usize];
    if paging && !protected {
        return Err(PagingRequiresProtectedMode);
    }
    // SDM Vol 1 §§2.1.1–2.1.3: generation availability is separate from size.
    if protected && model.generation == CpuGeneration::I8086 {
        return Err(ProtectedModeRequires286);
    }
    if paging && model.generation != CpuGeneration::I386 {
        return Err(PagingRequires386);
    }
    if state.cpl > 3 || (!protected && state.cpl != 0) {
        return Err(InvalidPrivilege);
    }
    if state.a20_enabled > 1
        || state.reset_fetch > 1
        || state.reserved != 0
        || state.gdtr.reserved != 0
        || state.idtr.reserved != 0
        || state.ldtr.reserved != 0
        || state.tr.reserved != 0
    {
        return Err(InvalidStorage);
    }
    for cache in state
        .segments
        .iter()
        .chain([&state.ldtr.cache, &state.tr.cache])
    {
        if cache.attributes & 0x0f00 != 0 || cache.unusable > 1 || cache.reserved != 0 {
            return Err(InvalidStorage);
        }
        if cache.attributes & (1 << 13) != 0 {
            return Err(LongModeUnsupported);
        }
        if model.generation != CpuGeneration::I386 && cache.attributes & (1 << 14) != 0 {
            return Err(SizeRequires386);
        }
    }
    if cs.unusable != 0 || ss.unusable != 0 {
        return Err(UnusableCodeOrStack);
    }
    // SDM Vol 3A §6.5: CPL is distinct from descriptor DPL.
    if protected && (cpu.cs & 3 != u16::from(state.cpl) || cpu.ss & 3 != u16::from(state.cpl)) {
        return Err(SelectorPrivilegeMismatch);
    }
    // SDM Vol 1 §3.4.3.3: VM and TF are stored separately from arithmetic flags.
    if cpu.rflags_system & (1 << 17) != 0 {
        return Err(Virtual8086Unsupported);
    }
    if cpu.rflags_system & (1 << 8) != 0 {
        return Err(DebugTrapUnsupported);
    }
    let bits = if cs.attributes & (1 << 14) != 0 {
        32
    } else {
        16
    };
    Ok(LegacyDecodeContext {
        generation: model.generation,
        regime: if protected {
            LegacyRegime::Protected
        } else {
            LegacyRegime::Real
        },
        default_operand_bits: bits,
        default_address_bits: bits,
        stack_address_bits: if ss.attributes & (1 << 14) != 0 {
            32
        } else {
            16
        },
        cpl: state.cpl,
        paging_enabled: paging,
    })
}

impl CpuState {
    /// Engine ordinary real entry, not hardware reset. The array uses SegmentId
    /// order. SDM Vol 3A §§3.4.3, 12.9.2: selector-derived normal real caches;
    /// SDM Vol 3B §23.1.4: the ordinary IVT has 256 four-byte entries.
    pub fn legacy_real_entry(selectors: [u16; 6]) -> Self {
        let [cs, ds, es, ss, fs, gs] = selectors;
        Self {
            cs,
            ds,
            es,
            ss,
            fs,
            gs,
            legacy: LegacyCpuState {
                idtr: TableRegister {
                    base: 0,
                    limit: 0x03ff,
                    reserved: 0,
                },
                segments: std::array::from_fn(|i| SegmentCache {
                    base: u32::from(selectors[i]) << 4,
                    limit: 0xffff,
                    attributes: if i == SegmentId::Cs as usize {
                        0x9b
                    } else {
                        0x93
                    },
                    ..SegmentCache::default()
                }),
                a20_enabled: 1,
                ..LegacyCpuState::default()
            },
            ..Self::default()
        }
    }

    /// Replace one visible selector and its actual hidden cache atomically.
    /// Validates context storage, generation/size and CS/SS privilege coherence;
    /// this is not a guest descriptor load and never reads descriptor tables.
    /// Real caches may retain bases/limits/attributes unrelated to the selector
    /// (SDM Vol 3A §§3.4.3, 12.9.1–12.9.2).
    pub fn set_legacy_segment(
        &mut self,
        model: LegacyCpuModel,
        segment: SegmentId,
        selector: u16,
        cache: SegmentCache,
    ) -> Result<(), LegacyContextError> {
        let mut proposed = self.clone();
        *match segment {
            SegmentId::Cs => &mut proposed.cs,
            SegmentId::Ds => &mut proposed.ds,
            SegmentId::Es => &mut proposed.es,
            SegmentId::Ss => &mut proposed.ss,
            SegmentId::Fs => &mut proposed.fs,
            SegmentId::Gs => &mut proposed.gs,
        } = selector;
        proposed.legacy.segments[segment as usize] = cache;
        derive_legacy_context(model, &proposed)?;
        *self = proposed;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(generation: CpuGeneration) -> LegacyCpuModel {
        LegacyCpuModel {
            generation,
            model_semantics_id: 37,
        }
    }

    #[test]
    fn generation_regime_code_stack_privilege_paging_matrix() {
        // SDM Vol 3A §§2.5, 3.4.5, 6.5, 12.9.1–12.9.2;
        // Vol 1 §§2.1.1–2.1.3. Retained cache sizes: AMD64 APM Vol 2 §14.6.
        for generation in [
            CpuGeneration::I8086,
            CpuGeneration::I286,
            CpuGeneration::I386,
        ] {
            for pe in [false, true] {
                for cs_d in [false, true] {
                    for ss_b in [false, true] {
                        for cpl in 0..=3 {
                            for pg in [false, true] {
                                let mut cpu = CpuState::legacy_real_entry([
                                    0x100, 0x200, 0x300, 0x400, 0x500, 0x600,
                                ]);
                                cpu.legacy.cr0 = u32::from(pe) | (u32::from(pg) << 31);
                                cpu.legacy.cpl = cpl;
                                if pe {
                                    cpu.cs |= u16::from(cpl);
                                    cpu.ss |= u16::from(cpl);
                                }
                                // Give retained caches bases/limits that a selector rebuild would lose.
                                cpu.legacy.segments[0].base = 0x1234_5678;
                                cpu.legacy.segments[0].limit = 0x1234_ffff;
                                cpu.legacy.segments[0].attributes |= u16::from(cs_d) << 14;
                                cpu.legacy.segments[3].base = 0x7654_3210;
                                cpu.legacy.segments[3].attributes |= u16::from(ss_b) << 14;
                                let retained = cpu.legacy;
                                let result = derive_legacy_context(model(generation), &cpu);
                                let valid = (!pg || pe)
                                    && (!pe || generation != CpuGeneration::I8086)
                                    && (!pg || generation == CpuGeneration::I386)
                                    && (!(cs_d || ss_b) || generation == CpuGeneration::I386)
                                    && (pe || cpl == 0);
                                assert_eq!(result.is_ok(), valid, "{generation:?} PE={pe} CS.D={cs_d} SS.B={ss_b} CPL={cpl} PG={pg}: {result:?}");
                                assert_eq!(cpu.legacy, retained);
                                if let Ok(context) = result {
                                    assert_eq!(
                                        context,
                                        LegacyDecodeContext {
                                            generation,
                                            regime: if pe {
                                                LegacyRegime::Protected
                                            } else {
                                                LegacyRegime::Real
                                            },
                                            default_operand_bits: if cs_d { 32 } else { 16 },
                                            default_address_bits: if cs_d { 32 } else { 16 },
                                            stack_address_bits: if ss_b { 32 } else { 16 },
                                            cpl,
                                            paging_enabled: pg,
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
    fn ordinary_entry_and_selector_cache_setter() {
        // SDM Vol 3A §§3.4.3–3.4.5, 12.9.2; Vol 3B §23.1.4.
        let mut cpu = CpuState::legacy_real_entry([0x100, 2, 3, 0x400, 5, 6]);
        assert_eq!(
            [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs],
            [0x100, 2, 3, 0x400, 5, 6]
        );
        for (i, selector) in [0x100u32, 2, 3, 0x400, 5, 6].into_iter().enumerate() {
            assert_eq!(
                cpu.legacy.segments[i],
                SegmentCache {
                    base: selector << 4,
                    limit: 0xffff,
                    attributes: if i == 0 { 0x9b } else { 0x93 },
                    unusable: 0,
                    reserved: 0,
                }
            );
        }
        assert_eq!(
            cpu.legacy.idtr,
            TableRegister {
                base: 0,
                limit: 0x03ff,
                reserved: 0
            }
        );
        assert_eq!(cpu.legacy.a20_enabled, 1);
        assert_eq!(cpu.legacy.reset_fetch, 0);
        assert_eq!(cpu.legacy.cr0, 0);
        let cache = SegmentCache {
            base: 0xdead_0000,
            limit: 0xffff_ffff,
            attributes: 0xc093,
            unusable: 0,
            reserved: 0,
        };
        let model = model(CpuGeneration::I386);
        cpu.set_legacy_segment(model, SegmentId::Ss, 0x1234, cache)
            .unwrap();
        assert_eq!(cpu.ss, 0x1234);
        assert_eq!(cpu.legacy.segments[3], cache);
        assert_eq!(
            derive_legacy_context(model, &cpu)
                .unwrap()
                .stack_address_bits,
            32
        );
        let before = cpu.legacy;
        assert_eq!(
            cpu.set_legacy_segment(
                model,
                SegmentId::Ss,
                0x4444,
                SegmentCache {
                    reserved: 1,
                    ..cache
                }
            ),
            Err(LegacyContextError::InvalidStorage)
        );
        assert_eq!(cpu.ss, 0x1234);
        assert_eq!(cpu.legacy, before);
        cpu.legacy.cr0 = 1;
        cpu.legacy.cpl = 0;
        assert_eq!(
            cpu.set_legacy_segment(model, SegmentId::Ss, 3, cache),
            Err(LegacyContextError::SelectorPrivilegeMismatch)
        );
        assert_eq!(cpu.ss, 0x1234);
    }

    #[test]
    fn invalid_storage_and_unsupported_facilities_are_explicit() {
        let model = model(CpuGeneration::I386);
        type StateEdit = fn(&mut CpuState);
        let cases: &[(StateEdit, LegacyContextError)] = &[
            (|c| c.legacy.cpl = 4, LegacyContextError::InvalidPrivilege),
            (|c| c.legacy.cpl = 1, LegacyContextError::InvalidPrivilege),
            (
                |c| c.legacy.a20_enabled = 2,
                LegacyContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.reset_fetch = 2,
                LegacyContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.reserved = 1,
                LegacyContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.gdtr.reserved = 1,
                LegacyContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.idtr.reserved = 1,
                LegacyContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.ldtr.reserved = 1,
                LegacyContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.tr.reserved = 1,
                LegacyContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.segments[0].attributes |= 0x100,
                LegacyContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.segments[0].attributes |= 0x2000,
                LegacyContextError::LongModeUnsupported,
            ),
            (
                |c| c.legacy.segments[0].unusable = 1,
                LegacyContextError::UnusableCodeOrStack,
            ),
            (
                |c| c.legacy.segments[3].unusable = 1,
                LegacyContextError::UnusableCodeOrStack,
            ),
            (
                |c| c.legacy.segments[5].unusable = 2,
                LegacyContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.ldtr.cache.reserved = 1,
                LegacyContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.tr.cache.attributes = 0x100,
                LegacyContextError::InvalidStorage,
            ),
            (
                |c| c.rflags_system = 1 << 17,
                LegacyContextError::Virtual8086Unsupported,
            ),
            (
                |c| c.rflags_system = 1 << 8,
                LegacyContextError::DebugTrapUnsupported,
            ),
        ];
        for (edit, error) in cases {
            let mut cpu = CpuState::legacy_real_entry([0; 6]);
            edit(&mut cpu);
            assert_eq!(derive_legacy_context(model, &cpu), Err(*error));
        }
    }

    #[test]
    fn reserved_records_have_fixed_layout() {
        use std::mem::{align_of, offset_of, size_of};
        assert_eq!(
            (size_of::<TableRegister>(), align_of::<TableRegister>()),
            (8, 4)
        );
        assert_eq!(
            (size_of::<SegmentCache>(), align_of::<SegmentCache>()),
            (12, 4)
        );
        assert_eq!(
            (size_of::<SystemSegment>(), align_of::<SystemSegment>()),
            (16, 4)
        );
        assert_eq!(
            (size_of::<LegacyCpuState>(), align_of::<LegacyCpuState>()),
            (136, 4)
        );
        assert_eq!(offset_of!(LegacyCpuState, segments), 60);
        assert_eq!(offset_of!(LegacyCpuState, cpl), 132);
    }
}
