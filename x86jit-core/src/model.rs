//! CPU-model catalogue (decision-23 §§2–4; spec.md §17.10).
//!
//! Decode policies and lifting constants are catalogue data. Remaining system
//! helper generation checks are compatibility projections (decision-23 §2).
use crate::{CpuGeneration, FpuModel, LegacyCpuModel};
pub(crate) mod decode;

// SDM Vol 3B §§25.16–25.16.1 pp.25-6–7; 80386 PRM §2.3.4,
// Fig.2-8 p.2-11; 80286 PRM §2.3.4 Fig.2-10 p.2-15.
pub(crate) const EARLY_FLAGS_FIXED_ZERO: u32 = (1 << 18) | (1 << 21);

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
    /// Absent identity flags (SDM Vol 3B §§25.16–25.16.1 pp.25-6–7).
    pub flags_fixed_zero: u32,
}

/// Reserved catalogue handle for a cycle table (decision-23, Timing model).
/// No entry has a table until cycle timing is implemented. Its representation
/// is private so adding tables does not change the catalogue API.
#[derive(Debug)]
pub struct ModelTiming {
    _private: (),
}

// Existing lift-time system and fault policies; each value is copied into IR.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelLiftPolicy {
    pub direct_stack_fault: u8,
    pub available_tss_types: u16,
    pub lar_system_types: u16,
    pub lsl_system_types: u16,
    pub lsl_conforming: bool,
    pub protected_coprocessor_outputs: bool,
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
    pub(crate) lift: ModelLiftPolicy,
}

impl ModelProfile {
    pub(crate) fn flags_helper_supported(&self) -> bool {
        self.quirks.flags_fixed_zero == EARLY_FLAGS_FIXED_ZERO
            && self.quirks.real_flags_clear_mask
                == match self.system {
                    ModelSystem::Msw286 => 0xf000,
                    _ => 0x8000,
                }
    }
}

static I80286: ModelProfile = identified(ModelProfile {
    isa: ModelIsa::I80286,
    quirks: ModelQuirks {
        push_sp_after_decrement: false,
        shift_count_mask: Some(31),
        real_flags_clear_mask: 0xf000,
        flags_fixed_zero: EARLY_FLAGS_FIXED_ZERO,
    },
    system: ModelSystem::Msw286,
    physical_bits: 24,
    max_insn_len: Some(10),
    fpu: &[FpuModel::Present, FpuModel::None],
    timing: None,
    semantics_id: 0,
    // 286 PRM App.B MOV p.B-74; LTR p.B-72; LAR/LSL pp.B-60,B-71;
    // decision-25: protected coprocessor outputs retain their typed refusal.
    lift: ModelLiftPolicy {
        direct_stack_fault: 13,
        available_tss_types: 1 << 1,
        lar_system_types: 0x003e,
        lsl_system_types: 0x000e,
        lsl_conforming: false,
        protected_coprocessor_outputs: false,
    },
});

static I80386: ModelProfile = identified(ModelProfile {
    isa: ModelIsa::I80386,
    quirks: ModelQuirks {
        push_sp_after_decrement: false,
        shift_count_mask: Some(31),
        real_flags_clear_mask: 0x8000,
        flags_fixed_zero: EARLY_FLAGS_FIXED_ZERO,
    },
    system: ModelSystem::Cr386,
    physical_bits: 32,
    max_insn_len: Some(15),
    fpu: &[FpuModel::Present, FpuModel::None],
    timing: None,
    semantics_id: 0,
    // 386 PRM §14.6 Table14-1 p.14-5; §17 LTR p.17-108;
    // SDM Vol2A §3.3 LAR Table3-55 / LSL Table3-59.
    lift: ModelLiftPolicy {
        direct_stack_fault: 12,
        available_tss_types: (1 << 1) | (1 << 9),
        lar_system_types: 0x1a3e,
        lsl_system_types: 0x0a0e,
        lsl_conforming: true,
        protected_coprocessor_outputs: true,
    },
});

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

/// Compatibility lookup used only during lifting. I8086 execution is still
/// refused at run entry; its catalogue is introduced separately (decision-23 C2).
pub(crate) const fn profile_for_generation(generation: CpuGeneration) -> &'static ModelProfile {
    match generation {
        CpuGeneration::I286 | CpuGeneration::I8086 => &I80286,
        CpuGeneration::I386 => &I80386,
    }
}

// FNV-1a over explicit encoding membership and all current lift attributes.
// Bump this revision when lifting/operand-validation rules change independently
// of the data. No persisted translation with a different ID may be reused.
const LIFT_POLICY_REVISION: u64 = 1;
const fn identified(mut p: ModelProfile) -> ModelProfile {
    const fn word(mut hash: u64, value: u64) -> u64 {
        let mut byte = 0;
        while byte < 8 {
            hash = (hash ^ ((value >> (byte * 8)) & 255)).wrapping_mul(0x100000001b3);
            byte += 1;
        }
        hash
    }
    let mut hash = word(0xcbf29ce484222325, LIFT_POLICY_REVISION);
    let mask = p.isa.model_mask();
    let mut i = 0;
    while i < decode::ENCODINGS.len() {
        let (code, models) = decode::ENCODINGS[i];
        if models & mask != 0 {
            hash = word(hash, code as u64);
            if let Some(alias) = decode::defined_alias(code) {
                hash = word(hash, alias as u64);
            }
        }
        i += 1;
    }
    let values = [
        mask as u64,
        p.isa.decoder_options() as u64,
        match p.isa.absent_encoding() {
            Some(v) => v as u64,
            None => 256,
        },
        p.quirks.push_sp_after_decrement as u64,
        match p.quirks.shift_count_mask {
            Some(v) => v as u64,
            None => 256,
        },
        p.quirks.real_flags_clear_mask as u64,
        p.quirks.flags_fixed_zero as u64,
        p.system as u64,
        p.physical_bits as u64,
        match p.max_insn_len {
            Some(v) => v as u64,
            None => 256,
        },
        p.lift.direct_stack_fault as u64,
        p.lift.available_tss_types as u64,
        p.lift.lar_system_types as u64,
        p.lift.lsl_system_types as u64,
        p.lift.lsl_conforming as u64,
        p.lift.protected_coprocessor_outputs as u64,
        p.isa.legacy_segments_only() as u64,
    ];
    i = 0;
    while i < values.len() {
        hash = word(hash, values[i]);
        i += 1;
    }
    i = 0;
    while i < p.fpu.len() {
        hash = word(hash, p.fpu[i] as u64);
        i += 1;
    }
    hash = word(hash, p.timing.is_some() as u64);
    p.semantics_id = hash;
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    fn copy(p: &ModelProfile) -> ModelProfile {
        ModelProfile {
            isa: p.isa,
            quirks: p.quirks,
            system: p.system,
            physical_bits: p.physical_bits,
            max_insn_len: p.max_insn_len,
            fpu: p.fpu,
            timing: p.timing,
            semantics_id: 0,
            lift: p.lift,
        }
    }
    #[test]
    fn semantics_identity_tracks_catalogue_content() {
        for model in [CpuModel::I80286, CpuModel::I80386] {
            let p = model.profile();
            assert_eq!(identified(copy(p)).semantics_id, p.semantics_id);
            for change in 0..9 {
                let mut changed = copy(p);
                match change {
                    0 => changed.quirks.push_sp_after_decrement = true,
                    1 => changed.quirks.shift_count_mask = None,
                    2 => changed.quirks.real_flags_clear_mask ^= 0x1000,
                    3 => changed.quirks.flags_fixed_zero ^= 1 << 18,
                    4 => changed.physical_bits -= 1,
                    5 => changed.max_insn_len = None,
                    6 => changed.lift.direct_stack_fault ^= 1,
                    7 => changed.fpu = &[FpuModel::None],
                    _ => changed.lift.available_tss_types ^= 1,
                }
                assert_ne!(identified(changed).semantics_id, p.semantics_id);
            }
        }
        assert_ne!(I80286.semantics_id, I80386.semantics_id);
    }
}
