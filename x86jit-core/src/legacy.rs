//! Legacy execution-context and inline-state contract (decision-18 §§1–2;
//! spec.md §§3, 17). Real execution is interpreted separately from Application.
use crate::{CpuMode, CpuState};

/// Fixed Application contexts and model-selected Machine execution (decision-23 §1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionProfile {
    Application(CpuMode),
    /// Processor execution selected by a CPU model (decision-23 §1).
    Machine,
}

impl ExecutionProfile {
    /// Compatibility name for [`Self::Machine`] (decision-23, Migration M1).
    #[allow(non_upper_case_globals)]
    #[deprecated(note = "use ExecutionProfile::Machine (decision-23 §1)")]
    pub const LegacySystem: Self = Self::Machine;
}

/// Per-VM routing of processor exceptions (decision-23 §5). Software INT
/// instructions always execute guest delivery, independently of this option.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExceptionDelivery {
    InGuest,
    #[default]
    Outward,
}

/// Inputs retained across an accepted delivery's device handoffs.
#[derive(Clone, Copy)]
pub(crate) struct LegacyEvent {
    pub vector: u8,
    pub saved_ip: u32,
    pub software: bool,
    pub external: bool,
    pub error_code: Option<u32>,
    pub fault: bool,
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

/// Compatibility model boundary retained by decision-23, Migration M1.
/// The deprecated selector preserves this embedder-supplied identity; new callers
/// select [`crate::CpuModel`] and its catalogue-owned identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LegacyCpuModel {
    pub generation: CpuGeneration,
    pub model_semantics_id: u64,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MachineRegime {
    Real,
    Protected,
}

/// Lift-time segment-access proof (decision-23 §5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentClass {
    Direct,
    Checked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MachineContext {
    pub segment_class: SegmentClass,
    /// Instruction-generation policy, independent of decoder width.
    pub generation: CpuGeneration,
    /// Real or protected execution regime derived from the retained CPU state.
    pub regime: MachineRegime,
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

// Frozen compatibility catalogue entry (decision-23 §6). Its decoder and
// low-word data-limit policy remain those of the Real16 lowering.
pub(crate) struct AliasProfile {
    pub context: MachineContext,
    pub exception_delivery: ExceptionDelivery,
}
pub(crate) const REAL16_ALIAS: AliasProfile = AliasProfile {
    context: MachineContext {
        segment_class: SegmentClass::Direct,
        generation: CpuGeneration::I286,
        regime: MachineRegime::Real,
        default_operand_bits: 16,
        default_address_bits: 16,
        stack_address_bits: 16,
        cpl: 0,
        paging_enabled: false,
    },
    exception_delivery: ExceptionDelivery::InGuest,
};

impl AliasProfile {
    // Decision-23 §6: visible selectors are the authority between alias runs.
    // Cache image: SDM Vol 3A §§3.4.3, 12.9.2.
    pub(crate) fn derive_caches(&self, cpu: &mut CpuState) {
        let selectors = [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs];
        for (i, cache) in cpu.legacy.segments.iter_mut().enumerate() {
            *cache = SegmentCache {
                base: u32::from(selectors[i]) << 4,
                limit: 0xffff,
                attributes: if i == SegmentId::Cs as usize {
                    0x9b
                } else {
                    0x93
                },
                ..SegmentCache::default()
            };
        }
    }
}

/// Invalid storage or a context outside the selected generation/engine scope.
/// These are configuration refusals, not synthesized guest exceptions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MachineContextError {
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

/// Compatibility name for [`MachineContext`] (decision-23, Migration M1).
#[deprecated(note = "use MachineContext (decision-23 §5)")]
pub type LegacyDecodeContext = MachineContext;

/// Compatibility name for [`MachineRegime`] (decision-23, Migration M1).
#[deprecated(note = "use MachineRegime (decision-23, Migration M1)")]
pub type LegacyRegime = MachineRegime;

/// Compatibility name for [`MachineContextError`] (decision-23, Migration M1).
#[deprecated(note = "use MachineContextError (decision-23, Migration M1)")]
pub type LegacyContextError = MachineContextError;

/// Engine layout, not the packed guest table-register operand image.
/// SDM Vol 3A §2.4: table registers retain base and limit.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TableRegister {
    pub base: u32,
    pub limit: u16,
    pub reserved: u16,
}

/// SDM Vol 3A §§3.4.3–3.4.5: hidden base, expanded inclusive byte limit,
/// descriptor access byte (7:0), and AVL/L/D-or-B/G flags (12:15).
/// Bits 11:8 and `reserved` must be zero; `unusable` is an engine marker.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SegmentCache {
    pub base: u32,
    pub limit: u32,
    pub attributes: u16,
    pub unusable: u8,
    pub reserved: u8,
}

/// SDM Vol 3A §§2.4, 3.5.1, 10.2.4: LDTR/TR retain selector and cache.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SystemSegment {
    pub selector: u16,
    pub reserved: u16,
    pub cache: SegmentCache,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SegmentId {
    Cs,
    Ds,
    Es,
    Ss,
    Fs,
    Gs,
}

/// Inline legacy state; Application execution never consumes these values.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
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
    /// Effective board line: 0 masks physical bit 20, 1 enables it. Application
    /// ignores it. Use `Vcpu::set_a20_enabled` between runs so an in-flight
    /// instruction or string element keeps its line until an architectural
    /// interruption point (decision-22): completion, a completed REP iteration,
    /// or abandonment by an architectural fault.
    pub a20_enabled: u8,
    pub reset_fetch: u8,
    /// Zero on every model except `NecV20`/`NecV30`, where it holds their mode
    /// state (decision-23 C4a): bit 0 is set while PSW.MD = 0 (8080 emulation
    /// mode), bit 1 is the hidden MD write-enable latch that BRKEM sets and
    /// RETEM clears (NEC V20/V30 UM pp. 12-169, 12-177), and bit 2 is the
    /// 8080-mode standby latch exchanged as `I8080View::halted`. A V20/V30
    /// reset image writes zero (MD = 1, NEC UM §7 p. 7-1). The 286 and 386
    /// refuse a nonzero value.
    pub reserved: u8,
}

/// `LegacyCpuState::reserved` bits of the NEC V20/V30 (decision-23 C4a).
pub(crate) const NEC_EMULATION: u8 = 1;
pub(crate) const NEC_MD_WRITE: u8 = 2;
pub(crate) const NEC_STANDBY: u8 = 4;

/// SDM Vol 3A §2.5: PE is bit 0, PG is bit 31; PG requires PE.
/// SDM Vol 3A §3.4.5: CS.D and SS.B independently select default sizes.
/// Retained attributes follow §§12.9.1–12.9.2 and AMD64 APM Vol 2 §§4.1, 14.6.
/// No caches are rebuilt or model-specific CR0 masks inferred here.
pub fn derive_legacy_context(
    model: LegacyCpuModel,
    cpu: &CpuState,
) -> Result<MachineContext, MachineContextError> {
    use MachineContextError::*;
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
        // I286 PRM §6.3 Fig. 6-3 p. 6-5: hidden descriptors have a
        // 24-bit base, 16-bit limit and only the access byte as attributes.
        if model.generation == CpuGeneration::I286
            && (cache.base > 0x00ff_ffff || cache.limit > 0xffff || cache.attributes & 0xf000 != 0)
        {
            return Err(InvalidStorage);
        }
    }
    if cs.unusable != 0 || ss.unusable != 0 {
        return Err(UnusableCodeOrStack);
    }
    // SDM Vol 3A §12.9.1: PE retains the real selectors and enters at CPL 0.
    // Their low bits need not describe CPL until a protected reload. CPL remains
    // authoritative; guest reloads and the coherent setter validate proposed pairs.
    // SDM Vol 1 §3.4.3.3: VM and TF are stored separately from arithmetic flags.
    if cpu.rflags_system & (1 << 17) != 0 {
        return Err(Virtual8086Unsupported);
    }
    let bits = if cs.attributes & (1 << 14) != 0 {
        32
    } else {
        16
    };
    Ok(MachineContext {
        segment_class: if !protected
            && state.reset_fetch == 0
            && state
                .segments
                .iter()
                .zip([cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs])
                .enumerate()
                .all(|(i, (cache, selector))| {
                    *cache
                        == SegmentCache {
                            base: u32::from(selector) << 4,
                            limit: 0xffff,
                            attributes: if i == SegmentId::Cs as usize {
                                0x9b
                            } else {
                                0x93
                            },
                            ..SegmentCache::default()
                        }
                }) {
            SegmentClass::Direct
        } else {
            SegmentClass::Checked
        },
        generation: model.generation,
        regime: if protected {
            MachineRegime::Protected
        } else {
            MachineRegime::Real
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
    ) -> Result<(), MachineContextError> {
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
        if proposed.legacy.cr0 & 1 != 0
            && matches!(segment, SegmentId::Cs | SegmentId::Ss)
            && selector & 3 != u16::from(proposed.legacy.cpl)
        {
            return Err(MachineContextError::SelectorPrivilegeMismatch);
        }
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
                                cpu.legacy.segments[0].base = if generation == CpuGeneration::I286 {
                                    0x12_5678
                                } else {
                                    0x1234_5678
                                };
                                cpu.legacy.segments[0].limit = if generation == CpuGeneration::I286
                                {
                                    0xabcd
                                } else {
                                    0x1234_ffff
                                };
                                cpu.legacy.segments[0].attributes |= u16::from(cs_d) << 14;
                                cpu.legacy.segments[3].base = if generation == CpuGeneration::I286 {
                                    0x54_3210
                                } else {
                                    0x7654_3210
                                };
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
                                        MachineContext {
                                            segment_class: SegmentClass::Checked,
                                            generation,
                                            regime: if pe {
                                                MachineRegime::Protected
                                            } else {
                                                MachineRegime::Real
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
            Err(MachineContextError::InvalidStorage)
        );
        assert_eq!(cpu.ss, 0x1234);
        assert_eq!(cpu.legacy, before);
        cpu.legacy.cr0 = 1;
        cpu.legacy.cpl = 0;
        assert_eq!(
            cpu.set_legacy_segment(model, SegmentId::Ss, 3, cache),
            Err(MachineContextError::SelectorPrivilegeMismatch)
        );
        assert_eq!(cpu.ss, 0x1234);
    }

    #[test]
    fn invalid_storage_and_unsupported_facilities_are_explicit() {
        let model = model(CpuGeneration::I386);
        type StateEdit = fn(&mut CpuState);
        let cases: &[(StateEdit, MachineContextError)] = &[
            (|c| c.legacy.cpl = 4, MachineContextError::InvalidPrivilege),
            (|c| c.legacy.cpl = 1, MachineContextError::InvalidPrivilege),
            (
                |c| c.legacy.a20_enabled = 2,
                MachineContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.reset_fetch = 2,
                MachineContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.reserved = 1,
                MachineContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.gdtr.reserved = 1,
                MachineContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.idtr.reserved = 1,
                MachineContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.ldtr.reserved = 1,
                MachineContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.tr.reserved = 1,
                MachineContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.segments[0].attributes |= 0x100,
                MachineContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.segments[0].attributes |= 0x2000,
                MachineContextError::LongModeUnsupported,
            ),
            (
                |c| c.legacy.segments[0].unusable = 1,
                MachineContextError::UnusableCodeOrStack,
            ),
            (
                |c| c.legacy.segments[3].unusable = 1,
                MachineContextError::UnusableCodeOrStack,
            ),
            (
                |c| c.legacy.segments[5].unusable = 2,
                MachineContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.ldtr.cache.reserved = 1,
                MachineContextError::InvalidStorage,
            ),
            (
                |c| c.legacy.tr.cache.attributes = 0x100,
                MachineContextError::InvalidStorage,
            ),
            (
                |c| c.rflags_system = 1 << 17,
                MachineContextError::Virtual8086Unsupported,
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

/// NEC V20/V30 BCD string operation (NEC UM pp. 12-42–44).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BcdStringOp {
    /// ADD4S: (IY) <- (IY) + (IX).
    Add,
    /// SUB4S: (IY) <- (IY) - (IX).
    Sub,
    /// CMP4S: (IY) - (IX), flags only.
    Cmp,
}

/// Interpreter-only operations (decision-18 §§1, 3–4). Widths are bytes;
/// address widths are bits. The Real16 compatibility alias shares the width-aware forms.
#[derive(Clone, Debug)]
pub enum LegacyOp {
    /// A Direct block using the shared Real16 walk.
    SharedWalk,
    /// Architectural rollback is required by this instruction's helper lowering.
    Checkpoint,
    Address {
        dst: crate::ir::Temp,
        segment: SegmentId,
        offset: crate::ir::Val,
        bytes: u8,
        access: crate::AccessKind,
    },
    ReadSelector {
        dst: crate::ir::Temp,
        segment: SegmentId,
    },
    LoadSegment {
        segment: SegmentId,
        value: crate::ir::Val,
        /// SDM Vol 3A §7.8.3: MOV/POP SS inhibit events; LSS does not.
        inhibit: bool,
    },
    Push {
        value: crate::ir::Val,
        size: u8,
    },
    Pop {
        dst: crate::ir::Temp,
        size: u8,
    },
    AdjustStack {
        bytes: u32,
    },
    Flags {
        size: u8,
        pop: bool,
    },
    All {
        size: u8,
        pop: bool,
    },
    Transfer {
        cs: Option<crate::ir::Val>,
        ip: crate::ir::Val,
        size: u8,
        return_ip: Option<u32>,
    },
    Return {
        size: u8,
        far: bool,
        pop_extra: u16,
    },
    Loop {
        kind: crate::ir::LoopKind,
        count_bits: u8,
        /// Direct constant targets within FFFFh need no runtime CS check.
        check_target: bool,
        taken: u32,
        next: u32,
    },
    String {
        kind: crate::ir::StrOp,
        elem: u8,
        rep: crate::ir::RepKind,
        address_bits: u8,
        source: SegmentId,
    },
    Enter {
        size: u8,
        alloc: u16,
        level: u8,
    },
    Coprocessor {
        bytes: [u8; 15],
        len: u8,
        wait: bool,
    },
    Leave {
        size: u8,
    },
    Table {
        addr: crate::ir::Val,
        idtr: bool,
        load: bool,
        size: u8,
    },
    ReadControl {
        dst: crate::ir::Temp,
        register: ControlRegister,
    },
    WriteControl {
        register: ControlRegister,
        value: crate::ir::Val,
    },
    LoadMsw {
        value: crate::ir::Val,
    },
    ClearTs,
    ReadSystemSegment {
        dst: crate::ir::Temp,
        register: SystemRegister,
    },
    LoadSystemSegment {
        register: SystemRegister,
        value: crate::ir::Val,
        allowed_types: u16,
    },
    QueryDescriptor {
        selector: crate::ir::Val,
        query: DescriptorQuery,
        allowed_system_types: u16,
        allow_conforming: bool,
    },
    AdjustRpl {
        selector: crate::ir::Val,
        rpl: crate::ir::Val,
        destination: SelectorDestination,
    },
    Interrupt {
        vector: u8,
        saved_ip: u32,
        overflow_only: bool,
    },
    Bound {
        index: crate::ir::Val,
        lower: crate::ir::Val,
        upper: crate::ir::Val,
        size: u8,
    },
    CheckTaskReturn,
    Iret {
        size: u8,
    },
    Refuse {
        operation: crate::UnsupportedSystemOperation,
    },
    /// A coprocessor operand of `bytes` (more than one word) whose first byte
    /// passes the segment check and whose end does not raises `vector` when a
    /// coprocessor is attached (the model's `coprocessor_overrun_vector`).
    /// Otherwise the following CPU operand check applies. A pending numeric
    /// error that `kind` tests at its start raises interrupt 16 first.
    CoprocessorOverrun {
        segment: SegmentId,
        offset: crate::ir::Val,
        bytes: u8,
        access: crate::AccessKind,
        vector: u8,
        kind: Option<crate::x87::FpuKind>,
    },
    /// Ordinary real-mode SS:SP address with a model-constant whole-operand gate.
    DirectStackAddress {
        dst: crate::ir::Temp,
        offset: crate::ir::Val,
        bytes: u8,
        vector: u8,
    },

    /// Bytewise transfers through a segment-wrapping twenty-bit address.
    WrappedLoad {
        dst: crate::ir::Temp,
        addr: crate::ir::Val,
        size: u8,
    },
    WrappedStore {
        addr: crate::ir::Val,
        value: crate::ir::Val,
        size: u8,
    },
    /// CPU ESC/WAIT policy with a generation-neutral optional coprocessor.
    Coprocessor20 {
        kind: Option<crate::x87::FpuKind>,
        addr: crate::ir::Val,
        sti: u8,
        memory: bool,
        wait: bool,
        bytes: [u8; 15],
        len: u8,
    },
    /// IVT delivery without an IDTR or a system-register image.
    Interrupt20 {
        vector: u8,
        saved_ip: u16,
        overflow_only: bool,
        flags_set: u16,
    },
    /// Lift-resolved FLAGS image constants (decision-23 §3).
    FlagsImage {
        dst: crate::ir::Temp,
        clear: u16,
        set: u16,
    },
    /// Restore the implemented real-mode flag bits, without system registers.
    LoadFlagsWord {
        value: crate::ir::Val,
        mask: u16,
        delay_irq: bool,
    },
    /// Divide with a lift-resolved return IP and signed quotient bound.
    DivideTrap {
        quot: crate::ir::Temp,
        rem: crate::ir::Temp,
        hi: crate::ir::Temp,
        lo: crate::ir::Val,
        divisor: crate::ir::Val,
        size: u8,
        signed: bool,
        reject_minimum: bool,
        saved_ip: u16,
    },
    /// String bytes wrap inside their segments and on a twenty-bit bus.
    /// `carry` is NEC V20/V30 REPC/REPNC: `Repe`/`Repne` test CY in place
    /// of Z (NEC UM p. 12-14). `code` holds the instruction's bytes, its
    /// prefixes and opcode, as they were lifted; `code_len` is 0 when there
    /// are more than fifteen (the REP restart pair's key, xj-50w.15).
    WrappedString {
        kind: crate::ir::StrOp,
        elem: u8,
        rep: crate::ir::RepKind,
        source: SegmentId,
        restart_ip: u16,
        carry: bool,
        code: [u8; 15],
        code_len: u8,
    },
    /// NEC V20/V30 INS (`insert`) and EXT (NEC UM pp. 12-21–24). `offset`
    /// is a Table 12-6 reg8 code; `length` is a reg8 code when
    /// `length_register`, else the immediate (the field length minus one).
    BitField20 {
        insert: bool,
        offset: u8,
        length: u8,
        length_register: bool,
    },
    /// NEC V20/V30 ADD4S, SUB4S and CMP4S (NEC UM pp. 12-42–44); `source`
    /// is the segment of the IX string.
    BcdString20 {
        op: BcdStringOp,
        source: SegmentId,
    },
    /// NEC V20/V30 FPO2 (NEC UM pp. 12-174–175): the CPU's part, with the
    /// instruction bytes for a typed refusal.
    Fpo2Escape20 {
        addr: crate::ir::Val,
        memory: bool,
        bytes: [u8; 15],
        len: u8,
    },
    /// NEC V20/V30 CHKIND: BRK 5 when the index lies outside the bounds
    /// (NEC UM p. 12-170). The page does not say whether the comparison is
    /// signed; a word pair on which the two readings disagree is refused.
    CheckIndex20 {
        index: crate::ir::Temp,
        lower: crate::ir::Temp,
        upper: crate::ir::Temp,
        saved_ip: u16,
        flags_set: u16,
    },
    /// NEC V20/V30 BRKEM imm8: frame, MD <- 0, MD write-enable, vector
    /// (NEC UM p. 12-169; IE and BRK cleared under decision-23 rule R2).
    Brkem20 {
        vector: u8,
        saved_ip: u16,
        flags_set: u16,
    },
    /// NEC V20/V30 RETI: PC, PS, PSW from the popped words; MD only while
    /// the MD write-enable latch is set (NEC UM p. 12-168).
    Reti20 {
        ip: crate::ir::Temp,
        cs: crate::ir::Temp,
        flags: crate::ir::Temp,
        sp: crate::ir::Temp,
    },
    /// NEC V20/V30 POP PSW flag restore with the MD rule (NEC UM p. 12-151).
    PopPsw20 {
        value: crate::ir::Temp,
    },
    /// Hold INT through the following instruction (NEC UM §6 p. 6-4).
    InhibitIrq20,
    /// Hold INT and NMI through the following instruction: MOV reg16/mem16,
    /// sreg (NEC UM §6 p. 6-4).
    HoldExternal20,
    /// A LOCK-prefixed read-modify-write through a twenty-bit address
    /// (8086 Family UM §2.5 pp. 2-17–18, §2.7 p. 2-48; NEC V20/V30 UM
    /// p. 12-173 BUSLOCK), with the instruction bytes for the typed refusal
    /// of a device operand.
    LockedRmw20 {
        old: crate::ir::Temp,
        addr: crate::ir::Val,
        src: crate::ir::Val,
        size: u8,
        op: crate::ir::RmwOp,
        bytes: [u8; 15],
        len: u8,
    },
    /// 80286 LOADALL: the whole register image from the table at physical
    /// 000800H (Intel, "Undocumented iAPX 286 Test Instruction",
    /// `oracles/intel/i286-loadall.pdf`, pp. 1-3).
    Loadall,
    /// The run-time CPL or IOPL test of a protected-mode instruction, before
    /// any operand transfer, port exit or other effect. The lifter emits it
    /// only in the protected regime.
    Privileged {
        kind: PrivilegeCheck,
    },
    /// A locked MOVS, INS or OUTS on the 80286 in the protected regime,
    /// after its `Privileged` LOCK test: the engine cannot hold the bus
    /// across the transfers, so the instruction is handed off
    /// (`UnknownInstruction`) before any effect (80286 PRM App. C item 11
    /// p. C-2, App. B LOCK p. B-68).
    LockedTransfer,
}

/// What a [`LegacyOp::Privileged`] test requires of the running program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrivilegeCheck {
    /// CPL 0, else #GP(0): HLT, LGDT, LIDT, LMSW, CLTS and MOV to or from a
    /// control, debug or test register (286 PRM App. B pp. B-31, B-42,
    /// B-65, B-67; 386 PRM §6.3.5.1 p. 6-17; SDM Vol 3A §6.9 p. 6-23).
    Cpl0,
    /// CPL <= IOPL, else #GP(0): CLI and STI (286 PRM §3.11.2 p. 3-28,
    /// App. B pp. B-30, B-106; 386 PRM §8.3.1 pp. 8-4–5), and on the 80286
    /// an instruction with a LOCK prefix (286 PRM §3.11.2 p. 3-29, App. B
    /// LOCK p. B-68).
    Iopl,
    /// CPL <= IOPL for IN, OUT, INS and OUTS. Otherwise the 80286 raises
    /// #GP(0) (286 PRM App. B pp. B-45, B-47, B-81, B-82). The 80386
    /// consults the I/O permission bitmap of a 386 TSS (386 PRM §8.3.2
    /// p. 8-5), which the engine refuses with `LegacyIoPermissionBitmap`;
    /// with any other TR its I/O permission function returns FALSE and the
    /// result is #GP(0) (386 PRM §17.2.2.5 p. 17-15).
    IoPort,
    /// REP INS or REP OUTS: the `IoPort` test, made only when the count
    /// register (CX, or ECX when `count_bytes` is 4) is not zero. A zero
    /// count executes nothing (SDM Vol 2B REP Operation p. 4-564).
    RepIoPort { count_bytes: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlRegister {
    Cr0,
    Cr2,
    Cr3,
}

#[derive(Clone, Copy, Debug)]
pub enum SystemRegister {
    Ldtr,
    Tr,
}

#[derive(Clone, Copy, Debug)]
pub enum DescriptorQuery {
    Rights { dst: crate::state::Reg, size: u8 },
    Limit { dst: crate::state::Reg, size: u8 },
    Read,
    Write,
}

#[derive(Clone, Copy, Debug)]
pub enum SelectorDestination {
    Register(crate::state::Reg),
    Memory(crate::ir::Val),
}

// SDM Vol 2A §3.3 LMSW; I286 PRM §10.2.2 Table 10-1 p. 10-4,
// Appendix B LMSW p. B-67 (scan checked): only PE/MP/EM/TS load; PE is sticky.
pub(crate) fn lmsw_value(cr0: u32, source: u16) -> u32 {
    (cr0 & !0xe) | u32::from(source & 0xf)
}

/// Intel, "Undocumented iAPX 286 Test Instruction", p. 2: LOADALL reads 102
/// bytes of physical memory from 000800H in 51 bus cycles.
pub(crate) const LOADALL_TABLE: u32 = 0x800;
pub(crate) const LOADALL_TABLE_BYTES: usize = 102;

/// The register image in an 80286 LOADALL table, as engine records (Intel,
/// "Undocumented iAPX 286 Test Instruction", p. 2 "LOADALL Memory Area
/// Format", p. 3 entry formats).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LoadallImage {
    pub msw: u16,
    pub flags: u16,
    pub ip: u16,
    /// CS, DS, ES and SS, in `SegmentId` order.
    pub selectors: [u16; 4],
    pub caches: [SegmentCache; 4],
    /// AX, CX, DX, BX, SP, BP, SI and DI, in the engine's register order.
    pub gpr: [u16; 8],
    pub gdtr: TableRegister,
    pub idtr: TableRegister,
    pub ldtr: SystemSegment,
    pub tr: SystemSegment,
}

impl LoadallImage {
    pub(crate) fn parse(table: &[u8; LOADALL_TABLE_BYTES]) -> Self {
        // p. 2 gives each register's physical address. 800H-805H and
        // 808H-815H are read but load no register.
        let at = |address: usize| &table[address - LOADALL_TABLE as usize..];
        let word = |address: usize| u16::from_le_bytes([at(address)[0], at(address)[1]]);
        // p. 3: a descriptor cache entry is the 24-bit base, least significant
        // byte first, the access rights byte, then the 16-bit limit. Its
        // present bit "becomes a valid bit": zero marks the cache invalid.
        let cache = |address: usize| {
            let entry = at(address);
            SegmentCache {
                base: u32::from_le_bytes([entry[0], entry[1], entry[2], 0]),
                limit: u32::from(u16::from_le_bytes([entry[4], entry[5]])),
                attributes: u16::from(entry[3]),
                unusable: u8::from(entry[3] & 0x80 == 0),
                reserved: 0,
            }
        };
        // p. 3: GDTR and IDTR use the same base and limit bytes. Their byte 3
        // "should be zeroes" and has no field to load: it is ignored, as LGDT
        // and LIDT ignore the last byte of their operand (I286 PRM App. B
        // LGDT/LIDT p. B-65, scan checked). DERIVED.
        let table_register = |address: usize| {
            let entry = cache(address);
            TableRegister {
                base: entry.base,
                limit: entry.limit as u16,
                reserved: 0,
            }
        };
        let system = |selector: usize, address: usize| SystemSegment {
            selector: word(selector),
            reserved: 0,
            cache: cache(address),
        };
        Self {
            msw: word(0x806),
            flags: word(0x818),
            ip: word(0x81a),
            selectors: [word(0x822), word(0x81e), word(0x824), word(0x820)],
            caches: [cache(0x83c), cache(0x848), cache(0x836), cache(0x842)],
            gpr: [0x834, 0x832, 0x830, 0x82e, 0x82c, 0x82a, 0x828, 0x826].map(word),
            gdtr: table_register(0x84e),
            idtr: table_register(0x85a),
            ldtr: system(0x81c, 0x854),
            tr: system(0x816, 0x860),
        }
    }

    /// Whether the engine loads this image as the 80286 would, in the regime
    /// the loaded MSW selects; otherwise LOADALL is the typed refusal
    /// `I286LoadallTable`, with nothing loaded. Intel, "Undocumented iAPX 286
    /// Test Instruction" (`oracles/intel/i286-loadall.pdf`): LOADALL itself
    /// checks nothing (pp. 1, 3), but p. 4 requires of the caches:
    /// - item 1: SS is a writable, valid data segment;
    /// - item 2: CS is execute-only code, execute/read code, or a writable
    ///   expand-up data segment;
    /// - item 3, "for proper protected mode operation": the CS and SS DPLs
    ///   are equal; they are the CPL.
    ///
    /// The policy for the document's wording. A table that breaks one of
    /// these requirements is refused: the document gives no result for it.
    /// Its advice ("should": a DS and ES DPL of 3, p. 3 and p. 4 item 4;
    /// GDTR and IDTR byte 3, p. 3) is not checked, and the values load as
    /// given. p. 2's protected-mode MSW "must have a one in bit position 0"
    /// states its own outcome (LOADALL cannot leave protected mode), so it is
    /// not checked either. Beyond the document, the engine refuses what it
    /// cannot execute as the 80286 would:
    /// - in protected mode, a data CS (the engine fetches only through code)
    ///   and a CS DPL other than 0 (kept refused at CPL 1–3, decision O12);
    /// - in real mode, an expand-down SS, DS or ES (real-mode access checks
    ///   every limit as expand-up);
    /// - a valid DS or ES with S = 0, a system type, in either mode.
    ///
    /// A conforming CS has one of item 2's two code capabilities and loads.
    /// The engine's real-mode access checks the limit and the valid bit but
    /// no rights, where the 80286 checks them "in either operating mode"
    /// (p. 1): a real-mode write through a loaded code or read-only data
    /// cache, or a read through execute-only code, faults on the 80286 and
    /// succeeds in the engine. Access byte bits: I286 PRM §6.5 Fig. 6-7
    /// p. 6-9 (scan checked).
    pub(crate) fn supported(&self, protected: bool) -> bool {
        const PRESENT: u16 = 0x80;
        const DPL: u16 = 0x60;
        // Present, S = 1, executable: code of any kind.
        let code = |rights: u16| rights & 0x98 == 0x98;
        // Present, S = 1, not executable, writable.
        let writable = |rights: u16| rights & 0x9a == 0x92;
        // S = 1, not executable, expand-down.
        let expand_down = |rights: u16| rights & 0x1c == 0x14;
        let rights = |id: SegmentId| self.caches[id as usize].attributes;
        let (cs, ss) = (rights(SegmentId::Cs), rights(SegmentId::Ss));
        let data = [rights(SegmentId::Ds), rights(SegmentId::Es)];
        writable(ss)
            && data
                .iter()
                .all(|&r| r & PRESENT == 0 || (r & 0x10 != 0 && (protected || !expand_down(r))))
            && if protected {
                code(cs) && (cs | ss) & DPL == 0
            } else {
                (code(cs) || (writable(cs) && !expand_down(cs))) && !expand_down(ss)
            }
    }
}

/// SDM Vol 3B §23.1.1: check the complete effective offset before forming
/// the linear address, including 67h offsets above FFFFh. Retained cache limits
/// follow decision-18 §1 (SDM Vol 3A §§3.4.3, 12.9.2; AMD64 APM Vol 2 §4.1).
/// SDM Vol 3B §23.1.1 and 80386 PRM §14.1 describe ordinary 64-KiB limits;
/// they do not specify nonstandard inherited limits after a real-mode reload.
/// The accepted big-real contract checks the retained inclusive byte limit.
/// Returns the linear address; bus transfers apply `physical_span` afterwards.
pub(crate) fn access(
    cpu: &CpuState,
    generation: CpuGeneration,
    segment: SegmentId,
    offset: u64,
    bytes: u8,
    access: crate::AccessKind,
) -> Result<u64, crate::Exit> {
    // I286 PRM §§7.4.1–7.4.2 pp. 7-12–13: whole-operand limits,
    // access rights and a 16-bit expand-down bound use this same boundary.
    let cache = cpu.legacy.segments[segment as usize];
    let last = offset.checked_add(u64::from(bytes.saturating_sub(1)));
    let protected = cpu.legacy.cr0 & 1 != 0;
    // SDM Vol 3A §§3.4.5.1, 6.3–6.4; 80386 PRM §§6.3.1.1–6.3.1.2,
    // pp. 6-4–6-6 (scan checked): type and whole-operand limit checks.
    let code = cache.attributes & 8 != 0;
    let expand_down = protected && !code && cache.attributes & 4 != 0;
    let bound = if cache.attributes & 0x4000 != 0 {
        u32::MAX
    } else {
        0xffff
    };
    let within_limit = last.is_some_and(|last| {
        if expand_down {
            offset > u64::from(cache.limit) && last <= u64::from(bound)
        } else {
            last <= u64::from(cache.limit)
        }
    });
    let permitted = !protected
        || match access {
            crate::AccessKind::Execute => code,
            crate::AccessKind::Read => !code || cache.attributes & 2 != 0,
            crate::AccessKind::Write => !code && cache.attributes & 2 != 0,
        };
    if cache.unusable != 0 || !within_limit || !permitted {
        let vector = if generation == CpuGeneration::I286 && !protected {
            // I286 PRM Appendix B MOV p. B-74: real-mode word operands
            // at FFFFh raise interrupt 13, including accesses through SS.
            13
        } else if segment == SegmentId::Ss {
            // I286 PRM Appendix B MOV p. B-74 (protected): SS violations
            // raise #SS(0). I386 real/protected: SDM Vol 3B §23.1.4
            // Table 23-1; Vol 3A §§6.3–6.4; 80386 PRM §14.6 Table 14-1 p. 14-5.
            12
        } else {
            // I286 PRM Appendix B MOV p. B-74 (protected): CS/DS/ES
            // violations raise #GP(0). I386: the same SDM/80386 PRM tables.
            13
        };
        return Err(segment_fault(cpu, vector, offset, bytes, access));
    }
    Ok(u64::from(cache.base.wrapping_add(offset as u32)))
}

/// Preserve the operand evidence for cached and proven ordinary segments alike.
#[cold]
pub(crate) fn segment_fault(
    cpu: &CpuState,
    vector: u8,
    offset: u64,
    bytes: u8,
    access: crate::AccessKind,
) -> crate::Exit {
    // Real-address pseudo-protection faults carry no error code (80386
    // PRM §14.1 p. 14-1; §14.6 Table 14-1 p. 14-5). Protected codes
    // define only bits 15:0 on I386 (PRM §9.7 Fig. 9-7, scan checked).
    operand_fault(
        cpu,
        vector,
        offset,
        bytes,
        access,
        if cpu.legacy.cr0 & 1 != 0 {
            crate::fault::ErrorCode::bits(0, 0xffff)
        } else {
            crate::fault::ErrorCode::NoCode
        },
    )
}

/// The 80286 coprocessor operand overrun, with the operand evidence and no
/// error code in either mode (I286 PRM Table 9-3 p. 9-9; App. B #MP p. B-9).
#[cold]
pub(crate) fn coprocessor_overrun_fault(
    cpu: &CpuState,
    vector: u8,
    offset: u64,
    bytes: u8,
    access: crate::AccessKind,
) -> crate::Exit {
    operand_fault(
        cpu,
        vector,
        offset,
        bytes,
        access,
        crate::fault::ErrorCode::NoCode,
    )
}

fn operand_fault(
    cpu: &CpuState,
    vector: u8,
    offset: u64,
    bytes: u8,
    access: crate::AccessKind,
    error_code: crate::fault::ErrorCode,
) -> crate::Exit {
    let mut details = crate::fault::FaultMetadata::architectural_exception(
        cpu.rip,
        cpu.rip,
        vector,
        crate::fault::FaultClass::Fault,
        error_code,
    );
    details.path = crate::fault::FaultPath::Interpreter;
    details.operand_base = crate::fault::Knowledge::Known(offset);
    details.operand_size = crate::fault::Knowledge::Known(u16::from(bytes));
    details.access = crate::fault::Knowledge::Known(access.into());
    crate::Exit::exception(cpu.rip, vector).with_fault_metadata(details)
}

/// The catalogue entry whose bus and fetch values a generation's context
/// uses. The twenty-bit generation takes the 8086-family values (I8088 and
/// I8086 share them, 8086 Family UM App. B pp. B-31–32).
pub(crate) const fn bus_model(generation: CpuGeneration) -> crate::CpuModel {
    match generation {
        CpuGeneration::I8086 => crate::CpuModel::I8088,
        CpuGeneration::I286 => crate::CpuModel::I80286,
        CpuGeneration::I386 => crate::CpuModel::I80386,
    }
}

/// The catalogue's formed physical-address width for a context, as a mask
/// (decision-23 §2): 24 bits on the 80286, which "sums the 24-bit segment
/// base and the specified 16-bit offset to generate the resulting 24-bit
/// physical address" (I286 PRM §6.4 p. 6-6; App. C hardware item 2 p. C-3),
/// 32 on the 80386 (80386 PRM §14.7 item 18 p. 14-7) and 20 on the 8086
/// family. Const blocks read the catalogue once, at compile time.
#[inline(always)]
pub(crate) const fn physical_mask(generation: CpuGeneration) -> u32 {
    const fn mask(model: crate::CpuModel) -> u32 {
        (((1u64 << model.profile().physical_bits) - 1) & 0xffff_ffff) as u32
    }
    match generation {
        CpuGeneration::I8086 => const { mask(bus_model(CpuGeneration::I8086)) },
        CpuGeneration::I286 => const { mask(bus_model(CpuGeneration::I286)) },
        CpuGeneration::I386 => const { mask(bus_model(CpuGeneration::I386)) },
    }
}

/// Final physical-bus gate, also for future page-walk transfers (decision-22).
/// SDM Vol 3A §11.7.13.4: asserted A20M# forces physical bit 20 to zero
/// on all external memory accesses. Vol 3B §§23.1.1, 25.33.1 describe wrapping.
/// The engine exposes this board gate for I286/I386 too; the SDM dates the
/// CPU pin to Intel486. I286 PRM Appendix C, software item 16 and hardware
/// item 2 (p. C-3), prescribe external gating of A20 for real-mode wrapping;
/// 80386 PRM §14.7 item 18 describes the unmasked address above 1 MiB.
/// `mask` is the model's [`physical_mask`].
pub(crate) fn physical_span(cpu: &CpuState, mask: u32, address: u64, bytes: u8) -> (u64, u8) {
    physical_span_with_gate(cpu.legacy.a20_enabled != 0, mask, address, bytes)
}

pub(crate) fn physical_span_with_gate(
    enabled: bool,
    mask: u32,
    address: u64,
    bytes: u8,
) -> (u64, u8) {
    // SDM Vol 3A §3.4: wrap each linear part before applying the bus gate,
    // and split transfers that cross the end of the 32-bit linear space.
    // The 80286 forms 24-bit addresses: a part past FFFFFFh wraps to the
    // first megabyte (I286 PRM §6.4 p. 6-6; App. D item 2 p. D-1), so its
    // transfers split at the end of the 24-bit space.
    let mask = u64::from(mask);
    let address = address & mask;
    let bytes = u64::from(bytes).min(mask - address + 1) as u8;
    if enabled {
        (address, bytes)
    } else {
        let remaining = (1 << 20) - (address & ((1 << 20) - 1));
        (address & !(1 << 20), u64::from(bytes).min(remaining) as u8)
    }
}

/// SDM Vol 1 §21.1.2: 8086 high FLAGS bits are one, 286 real high
/// bits zero, and 386 IOPL/NT writable with bit 15 zero; AC/ID are absent.
/// 80386 PRM §2.3.4, Fig. 2-8, p. 2-11: reserved EFLAGS bits are zero/one
/// as pictured (verified against the scan); §§4.1.1, 17 POPF, p. 17-130:
/// POPFD leaves RF/VM unchanged and real mode has privilege zero.
pub(crate) fn flags_image(cpu: &CpuState, generation: CpuGeneration) -> u32 {
    let low = (u32::from(cpu.flags.to_flags16()) & 0x0fd7) | (cpu.rflags_system as u32 & 0x100);
    match generation {
        CpuGeneration::I8086 => low | 0xf000,
        // I286 PRM §10.1 p. 10-1; Appendix B IRET
        // pp. B-51–52: protected FLAGS includes IOPL and NT.
        CpuGeneration::I286 if cpu.legacy.cr0 & 1 != 0 => low | (cpu.rflags_system as u32 & 0x7000),
        CpuGeneration::I286 => low,
        CpuGeneration::I386 => low | (cpu.rflags_system as u32 & 0x0003_7000),
    }
}

pub(crate) fn restore_flags(
    cpu: &mut CpuState,
    generation: CpuGeneration,
    image: u32,
) -> Result<(), crate::Exit> {
    let image = image & !crate::model::EARLY_FLAGS_FIXED_ZERO;
    cpu.flags.set_flags16((image & 0x0ed7) as u16);
    // I286 PRM Appendix B IRET pp. B-51–52; §10.1 p. 10-1.
    // This is the CPL-0 load, and a task switch's; POPF and IRET at CPL 1–3
    // mask IF and IOPL in `interp::legacy::privilege::load_flags`.
    let system = if generation == CpuGeneration::I386
        || (generation == CpuGeneration::I286 && cpu.legacy.cr0 & 1 != 0)
    {
        image & 0x7000
    } else {
        0
    };
    cpu.rflags_system = (cpu.rflags_system & 0x30000) | u64::from(system | (image & 0x100));
    Ok(())
}

#[cfg(test)]
mod real_flags_tests {
    use super::*;
    #[test]
    fn lmsw_preserves_pe_and_every_bit_above_the_low_four() {
        // SDM Vol 2A §3.3 LMSW: PE cannot be cleared; only bits 3:0 load.
        for (cr0, source, expected) in [
            (0x8000_001f, 0, 0x8000_0011),
            (0x10, 0xfffe, 0x1e),
            (0x10, 0xffff, 0x1f),
            (0x8000_0011, 0x8000, 0x8000_0011),
        ] {
            assert_eq!(lmsw_value(cr0, source), expected);
        }
    }
    #[test]
    fn identification_images_and_pop_masks_are_generation_values() {
        // SDM Vol 1 §21.1.2; 80386 PRM §2.3.4 Fig. 2-8 p. 2-11
        // and §17 POPF p. 17-130 (verified page images).
        for (generation, expected) in [
            (CpuGeneration::I8086, 0xfed7),
            (CpuGeneration::I286, 0x0ed7),
            (CpuGeneration::I386, 0x7ed7),
        ] {
            let mut cpu = CpuState::legacy_real_entry([0; 6]);
            restore_flags(&mut cpu, generation, 0xffff_feff).unwrap();
            assert_eq!(flags_image(&cpu, generation), expected);
            assert_eq!(
                cpu.rflags_system,
                if generation == CpuGeneration::I386 {
                    0x7000
                } else {
                    0
                }
            );
        }
        let mut cpu = CpuState::legacy_real_entry([0; 6]);
        cpu.rflags_system = 0x30000;
        restore_flags(&mut cpu, CpuGeneration::I386, 0).unwrap();
        assert_eq!(cpu.rflags_system, 0x30000);
        assert_eq!(flags_image(&cpu, CpuGeneration::I386), 0x30002);
    }
}

#[cfg(test)]
mod loadall_tests {
    use super::*;

    #[test]
    fn loadall_table_follows_the_documented_layout() {
        // Intel, "Undocumented iAPX 286 Test Instruction", p. 2 "LOADALL
        // Memory Area Format", restated row by row: each word register is
        // written with its own address. The rows that load no register hold
        // A5h and must not reach any field.
        let mut table = [0xa5; LOADALL_TABLE_BYTES];
        let mut put = |address: usize, bytes: &[u8]| {
            table[address - 0x800..][..bytes.len()].copy_from_slice(bytes);
        };
        for address in [
            0x806, 0x816, 0x818, 0x81a, 0x81c, 0x81e, 0x820, 0x822, 0x824, 0x826, 0x828, 0x82a,
            0x82c, 0x82e, 0x830, 0x832, 0x834,
        ] {
            put(address, &(address as u16).to_le_bytes());
        }
        // p. 3: base bytes 0-2 least significant first, access byte, limit.
        for (address, access) in [
            (0x836, 0x93), // ES descriptor cache
            (0x83c, 0x9b), // CS descriptor cache
            (0x842, 0x92), // SS descriptor cache
            (0x848, 0x13), // DS descriptor cache, invalid (present bit zero)
            (0x84e, 0x00), // GDTR
            (0x854, 0x82), // LDT descriptor cache
            (0x85a, 0x01), // IDTR, byte 3 nonzero and ignored
            (0x860, 0x03), // TSS descriptor cache, invalid
        ] {
            put(
                address,
                &[
                    address as u8,
                    0x34,
                    0x12,
                    access,
                    address as u8,
                    0xf0 | (address >> 8) as u8,
                ],
            );
        }
        let cache = |address: u32, attributes: u16| SegmentCache {
            base: 0x12_3400 | (address & 0xff),
            limit: 0xf000 | address,
            attributes,
            unusable: u8::from(attributes & 0x80 == 0),
            reserved: 0,
        };
        let table_register = |address: u32| TableRegister {
            base: 0x12_3400 | (address & 0xff),
            limit: 0xf000 | address as u16,
            reserved: 0,
        };
        assert_eq!(
            LoadallImage::parse(&table),
            LoadallImage {
                msw: 0x806,
                flags: 0x818,
                ip: 0x81a,
                // CS, DS, ES, SS.
                selectors: [0x822, 0x81e, 0x824, 0x820],
                caches: [
                    cache(0x83c, 0x9b),
                    cache(0x848, 0x13),
                    cache(0x836, 0x93),
                    cache(0x842, 0x92),
                ],
                // AX, CX, DX, BX, SP, BP, SI, DI.
                gpr: [0x834, 0x832, 0x830, 0x82e, 0x82c, 0x82a, 0x828, 0x826],
                gdtr: table_register(0x84e),
                idtr: table_register(0x85a),
                ldtr: SystemSegment {
                    selector: 0x81c,
                    reserved: 0,
                    cache: cache(0x854, 0x82),
                },
                tr: SystemSegment {
                    selector: 0x816,
                    reserved: 0,
                    cache: cache(0x860, 0x03),
                },
            }
        );
    }

    #[test]
    fn loadall_tables_the_engine_loads_exactly() {
        // Intel, "Undocumented iAPX 286 Test Instruction", pp. 1-4, and the
        // policy in `LoadallImage::supported`; access byte: I286 PRM §6.5
        // Fig. 6-7 p. 6-9.
        let image = |cs: u8, ss: u8, ds: u8, es: u8| {
            let mut table = [0; LOADALL_TABLE_BYTES];
            for (address, access) in [(0x83c, cs), (0x842, ss), (0x848, ds), (0x836, es)] {
                table[address - 0x800 + 3] = access;
            }
            LoadallImage::parse(&table)
        };
        // Real mode: SS writable expand-up data; CS any code or writable
        // expand-up data; DS and ES invalid, code or expand-up data. DPLs
        // are free; rights faults are not raised in real mode.
        for (cs, ss, ds, es, expected) in [
            (0x93, 0x93, 0x93, 0x93, true),
            (0x92, 0xf2, 0xb3, 0xd3, true),
            (0x93, 0x93, 0x13, 0x00, true),
            (0x9b, 0x93, 0x93, 0x93, true),
            (0x98, 0x93, 0x93, 0x93, true),
            (0x9f, 0x93, 0x93, 0x93, true),
            (0xfb, 0x93, 0x93, 0x93, true),
            (0x93, 0x93, 0x91, 0x93, true),
            (0x93, 0x93, 0x93, 0x9a, true),
            (0x93, 0x93, 0x98, 0x9e, true),
            (0x91, 0x93, 0x93, 0x93, false),
            (0x97, 0x93, 0x93, 0x93, false),
            (0x93, 0x97, 0x93, 0x93, false),
            (0x93, 0x91, 0x93, 0x93, false),
            (0x93, 0x9b, 0x93, 0x93, false),
            (0x13, 0x93, 0x93, 0x93, false),
            (0x1b, 0x93, 0x93, 0x93, false),
            (0x93, 0x13, 0x93, 0x93, false),
            (0x93, 0x93, 0x97, 0x93, false),
            (0x93, 0x93, 0x93, 0x95, false),
            (0x93, 0x93, 0x93, 0x82, false),
        ] {
            assert_eq!(
                image(cs, ss, ds, es).supported(false),
                expected,
                "real {cs:02x} {ss:02x} {ds:02x} {es:02x}"
            );
        }
        // Protected mode: SS valid writable data; CS valid code, conforming
        // or not; both at DPL 0; DS and ES invalid or any S = 1 type.
        for (cs, ss, ds, es, expected) in [
            (0x9b, 0x93, 0x93, 0x93, true),
            (0x99, 0x97, 0x9a, 0x00, true),
            (0x9f, 0x93, 0xf1, 0xf9, true),
            (0x9c, 0x92, 0x95, 0x13, true),
            (0x93, 0x93, 0x93, 0x93, false),
            (0x1b, 0x93, 0x93, 0x93, false),
            (0xfb, 0xf3, 0x93, 0x93, false),
            (0xbb, 0xb3, 0x93, 0x93, false),
            (0x9b, 0xd3, 0x93, 0x93, false),
            (0x9b, 0x91, 0x93, 0x93, false),
            (0x9b, 0x9b, 0x93, 0x93, false),
            (0x9b, 0x93, 0x81, 0x93, false),
            (0x9b, 0x93, 0x93, 0x83, false),
        ] {
            assert_eq!(
                image(cs, ss, ds, es).supported(true),
                expected,
                "protected {cs:02x} {ss:02x} {ds:02x} {es:02x}"
            );
        }
    }
}
