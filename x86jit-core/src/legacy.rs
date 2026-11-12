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
    pub reserved: u8,
}

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

/// Interpreter-only operations (decision-18 §§1, 3–4). Widths are bytes;
/// address widths are bits. Application lifting never emits these operations.
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
    /// Ordinary real-mode SS:SP address with a model-constant whole-operand gate.
    DirectStackAddress {
        dst: crate::ir::Temp,
        offset: crate::ir::Val,
        bytes: u8,
        vector: u8,
    },
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
    let mut details = crate::fault::FaultMetadata::architectural_exception(
        cpu.rip,
        cpu.rip,
        vector,
        crate::fault::FaultClass::Fault,
        if cpu.legacy.cr0 & 1 != 0 {
            crate::fault::ErrorCode::bits(0, 0xffff)
        } else {
            crate::fault::ErrorCode::NoCode
        },
    );
    details.path = crate::fault::FaultPath::Interpreter;
    details.operand_base = crate::fault::Knowledge::Known(offset);
    details.operand_size = crate::fault::Knowledge::Known(u16::from(bytes));
    details.access = crate::fault::Knowledge::Known(access.into());
    crate::Exit::exception(cpu.rip, vector).with_fault_metadata(details)
}

/// Final physical-bus gate, also for future page-walk transfers (decision-22).
/// SDM Vol 3A §11.7.13.4: asserted A20M# forces physical bit 20 to zero
/// on all external memory accesses. Vol 3B §§23.1.1, 25.33.1 describe wrapping.
/// The engine exposes this board gate for I286/I386 too; the SDM dates the
/// CPU pin to Intel486. I286 PRM Appendix C, software item 16 and hardware
/// item 2 (p. C-3), prescribe external gating of A20 for real-mode wrapping;
/// 80386 PRM §14.7 item 18 describes the unmasked address above 1 MiB.
pub(crate) fn physical_span(cpu: &CpuState, address: u64, bytes: u8) -> (u64, u8) {
    physical_span_with_gate(cpu.legacy.a20_enabled != 0, address, bytes)
}

pub(crate) fn physical_span_with_gate(enabled: bool, address: u64, bytes: u8) -> (u64, u8) {
    // SDM Vol 3A §3.4: wrap each linear part before applying the bus gate,
    // and split transfers that cross the end of the 32-bit linear space.
    let address = u64::from(address as u32);
    let bytes = u64::from(bytes).min((1u64 << 32) - address) as u8;
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
    cpu.flags.set_flags16((image & 0x0ed7) as u16);
    // I286 PRM Appendix B IRET pp. B-51–52; §10.1 p. 10-1.
    // Execution is currently limited to CPL 0, where IOPL is writable.
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
