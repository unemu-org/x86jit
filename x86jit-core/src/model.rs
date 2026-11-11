//! CPU-model catalogue (decision-23 §§2–4; spec.md §17.10).
//!
//! Entries describe the existing Machine path. Decode filters and semantic
//! consumers retain their generation checks until the catalogue migration.
use crate::{CpuGeneration, FpuModel, LegacyCpuModel};

/// Processor selected before vCPU creation (decision-23 §2).
/// Later processor families extend this enum without changing existing entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CpuModel {
    /// Intel 80286 (decision-23 §2).
    I80286,
    /// Intel 80386DX (decision-23 §2).
    I80386,
}

/// Decode-set identity, independent of instruction width (decision-23 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelIsa {
    /// The current 286 real/protected decode policy (decision-23 §2).
    I80286,
    /// The current 386 real/protected decode policy (decision-23 §2).
    I80386,
}

impl ModelIsa {
    /// Compatibility projection for existing generation checks (decision-23 §2).
    pub const fn generation(self) -> CpuGeneration {
        match self {
            Self::I80286 => CpuGeneration::I286,
            Self::I80386 => CpuGeneration::I386,
        }
    }
}

/// Architectural system-register family (decision-23 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelSystem {
    /// No protected-mode system registers (decision-23 §2).
    None,
    /// 286 MSW and 16-bit protected-mode state (decision-23 §2).
    Msw286,
    /// 386 control registers and protected-mode state (decision-23 §2).
    /// This describes the processor; it does not imply engine paging support.
    Cr386,
}

/// Named policies already used by the Machine path (decision-23 §§2–3).
/// Additional quirk fields are introduced with their catalogue consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelQuirks {
    /// Whether PUSH SP stores the decremented value (decision-23 §3;
    /// SDM Vol 3B §25.17.1).
    pub push_sp_after_decrement: bool,
    /// Shift/rotate count mask, or no mask (decision-23 §3;
    /// SDM Vol 2B §4.2, SAL/SAR/SHL/SHR, IA-32 Architecture Compatibility).
    pub shift_count_mask: Option<u8>,
    /// Real-mode FLAGS bits forced clear in the stored image (decision-23 §3;
    /// SDM Vol 3B §25.17.2, Vol 1 §21.1.2).
    pub real_flags_clear_mask: u16,
}

/// Reserved catalogue handle for a cycle table (decision-23, Timing model).
/// No entry has a table until cycle timing is implemented. Its representation
/// is private so adding tables does not change the catalogue API.
#[derive(Debug)]
pub struct ModelTiming {
    _private: (),
}

/// Static processor attributes, in decision-23 §2 catalogue order.
///
/// `physical_bits` and `system` describe the processor, including facilities
/// that execution may still refuse. Timing is absent until cycle tables land.
#[derive(Debug)]
#[non_exhaustive]
pub struct ModelProfile {
    /// Existing decode-set policy (decision-23 §2).
    pub isa: ModelIsa,
    /// Named semantic policies (decision-23 §§2–3).
    pub quirks: ModelQuirks,
    /// System-register family (decision-23 §2).
    pub system: ModelSystem,
    /// Formed physical-address width (decision-23 §2).
    /// I286 PRM §6.3 Fig. 6-3; 80386 PRM §14.7 item 18.
    pub physical_bits: u8,
    /// Maximum instruction bytes, or no architectural limit (decision-23 §2).
    /// I286 PRM Appendix B, General Protection; SDM Vol 2 §2.1.
    pub max_insn_len: Option<u8>,
    /// Accepted available FPU policies (decision-23 §4). `Present` preserves
    /// the generation-neutral engine; generation-specific variants arrive later.
    pub fpu: &'static [FpuModel],
    /// Cycle table, currently absent (decision-23, Timing model).
    pub timing: Option<&'static ModelTiming>,
    /// Stable lifting-policy identity owned by the catalogue (decision-23 §2).
    pub semantics_id: u64,
}

static I80286: ModelProfile = ModelProfile {
    isa: ModelIsa::I80286,
    quirks: ModelQuirks {
        push_sp_after_decrement: false,
        shift_count_mask: Some(31),
        real_flags_clear_mask: 0xf000,
    },
    system: ModelSystem::Msw286,
    physical_bits: 24,
    max_insn_len: Some(10),
    fpu: &[FpuModel::Present, FpuModel::None],
    timing: None,
    semantics_id: 0x286,
};

static I80386: ModelProfile = ModelProfile {
    isa: ModelIsa::I80386,
    quirks: ModelQuirks {
        push_sp_after_decrement: false,
        shift_count_mask: Some(31),
        real_flags_clear_mask: 0x8000,
    },
    system: ModelSystem::Cr386,
    physical_bits: 32,
    max_insn_len: Some(15),
    fpu: &[FpuModel::Present, FpuModel::None],
    timing: None,
    semantics_id: 0x386,
};

impl CpuModel {
    /// The immutable catalogue entry for this model (decision-23 §2).
    pub const fn profile(self) -> &'static ModelProfile {
        match self {
            Self::I80286 => &I80286,
            Self::I80386 => &I80386,
        }
    }

    pub(crate) const fn legacy_model(self) -> LegacyCpuModel {
        let profile = self.profile();
        LegacyCpuModel {
            generation: profile.isa.generation(),
            model_semantics_id: profile.semantics_id,
        }
    }
}
