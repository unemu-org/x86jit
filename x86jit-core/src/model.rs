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
    /// Intel 8088, with an 8-bit external bus (8086 Family UM App. B pp. B-31–32).
    I8088,
    /// Intel 8086, with a 16-bit external bus (8086 Family UM App. B pp. B-31–32).
    I8086,
    /// NEC uPD70108 (V20), with an 8-bit external bus (NEC V20/V30 UM §1 p. 1-1).
    NecV20,
    /// NEC uPD70116 (V30), with a 16-bit external bus (NEC V20/V30 UM §1 p. 1-1).
    NecV30,
    /// Early Intel486 DX without CPUID (486 PRM §10.1 pp. 10-1–3).
    I80486Dx,
    /// Later Intel486 DX with CPUID (Embedded 486 DM App. D pp. D-1–3).
    I80486DxCpuid,
}

/// Decode-set identity, independent of instruction width (decision-23 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelIsa {
    /// The current 286 real/protected decode policy (decision-23 §2).
    I80286,
    /// The current 386 real/protected decode policy (decision-23 §2).
    I80386,
    /// 8086/8088 encodings (8086 Family UM §4.2 Tables 4-12/4-13 pp. 4-22–36).
    I8086,
    /// NEC V20/V30 native mode: the 8086 set, the 80186-class forms NEC UM
    /// §12 lists and BRKEM (decision-23 C4a), and the native-mode extensions
    /// (bit, BCD-string, nibble-rotate and bit-field groups, REPC/REPNC and
    /// FPO2; decision-23 C4b).
    NecV20,
    /// Intel486 PRM App. A pp. A-4–7, without CPUID.
    I80486Dx,
    /// The 486 set plus Embedded 486 DM App. D CPUID.
    I80486DxCpuid,
}

impl ModelIsa {
    /// Compatibility projection for existing generation checks (decision-23 §2).
    pub const fn generation(self) -> CpuGeneration {
        match self {
            Self::I8086 | Self::NecV20 => CpuGeneration::I8086,
            Self::I80286 => CpuGeneration::I286,
            Self::I80386 | Self::I80486Dx | Self::I80486DxCpuid => CpuGeneration::I386,
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
    /// Highest processor-generated interrupt (8086 Family UM §2.6 pp. 2-24–28;
    /// 286 PRM App. C Table C-1 p. C-1; 386 PRM §14.6 Table 14-1 p. 14-5).
    pub max_processor_interrupt: u8,
    /// Ones in the stored FLAGS image (SDM Vol 3B §25.17.2).
    pub real_flags_set_mask: u16,
    /// Divide interrupt returns after the instruction (8086 Family UM §2.6 p. 2-26).
    pub divide_saves_next_ip: bool,
    /// Signed minimum quotient is outside the divide range (UM §2.7 p. 2-37).
    pub idiv_rejects_signed_minimum: bool,
    /// Each byte offset wraps inside its segment (UM §2.3 p. 2-12).
    pub wrap_segment_offsets: bool,
    /// Sequential IP wraps to zero (UM §2.2 p. 2-7; §2.3 p. 2-12).
    pub wrap_sequential_ip: bool,
    /// Prefix bytes an interrupted repeated string keeps in its return
    /// address, counted back from the opcode; `None` returns to the
    /// instruction's first byte (8086 Family UM §2.7 p. 2-42: the last one;
    /// NEC V20/V30 UM §6 p. 6-4: up to three).
    pub rep_restart_prefixes: Option<u8>,
    /// The single-step trap follows external interrupt entry (UM §2.6 p. 2-28).
    pub step_interrupt_handler: bool,
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
    /// IRET delays maskable interrupts through its successor when it sets
    /// IF (8086 Family UM §2.6 p. 2-24). NEC V20/V30 UM §6 p. 6-4 does not
    /// list RETI among the instructions after which INT is held.
    pub iret_delays_interrupts: bool,
    /// A store of a segment register (MOV r/m16, sreg) holds NMI and INT
    /// through the following instruction (NEC V20/V30 UM §6 p. 6-4).
    pub segment_store_inhibits_interrupts: bool,
    /// The manual gives no order for a single-step trap and an acceptable
    /// external interrupt, INT or NMI, at one boundary; the engine refuses
    /// that boundary (NEC V20/V30 UM §6 Table 6-1 p. 6-1 lists priorities
    /// only).
    pub step_irq_conflict_refused: bool,
    /// STI holds maskable interrupts through the next instruction whatever IF
    /// was: NEC V20/V30 UM §6 p. 6-4 ("between the EI instruction and the next
    /// instruction") and p. 12-172 EI state no condition. Otherwise only an
    /// STI that begins with IF = 0 does (SDM Vol 2B §4.3 STI).
    pub ei_delay_unconditional: bool,
    /// LOCK may prefix MOV, MOVS, INS and OUTS without #UD (80286 PRM App. C
    /// item 11 p. C-2). A locked MOV takes the engine's atomic path; a locked
    /// MOVS, INS or OUTS, whose transfers the engine cannot hold the bus
    /// across, ends in a typed handoff (`Exit::UnknownInstruction`). Other
    /// LOCK forms follow the decode set: #UD outside the lockable forms, a
    /// manual value on the 80386 (80386 PRM §14.7 item 9 p. 14-6) and the
    /// engine's rule on the 80286, whose App. B LOCK p. B-68 and #UD p. B-12
    /// give no LOCK #UD.
    pub lock_data_movement: bool,
    /// Vector raised when a coprocessor memory operand larger than one word
    /// starts inside its segment and ends past it (80286 PRM App. B #MP
    /// p. B-9; §5.2 p. 5-7; App. C Table C-1 p. C-1). Only an attached
    /// coprocessor raises it; `None` keeps the CPU's whole-operand check.
    pub coprocessor_overrun_vector: Option<u8>,
    /// What an NMI edge does while an earlier NMI is being serviced
    /// (spec.md §17.10).
    pub nmi_nesting: NmiNesting,
    /// ASSUMED 386 compatibility option: ignore REP/REPNE on non-strings.
    /// Default false follows 386 PRM §17 REP p. 17-141 (#UD); SDM Vol 2A
    /// §2.1.1 p. 2-2 instead reserves this use with unpredictable behavior.
    pub i386_ignore_rep_non_string: bool,
}

/// What a rising edge on the NMI input does while an earlier NMI is being
/// serviced (spec.md §17.10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NmiNesting {
    /// No blocking: an edge after acceptance is serviced again (8086 Family
    /// UM §2.6 p. 2-26 and App. B p. B-6; ASSUMED for the NEC V20/V30, whose
    /// manual is silent).
    Nested,
    /// Blocked from acceptance until the next IRET; one further edge is
    /// remembered and serviced after that IRET (80286 PRM §9.2 p. 9-3).
    BlockedRemember,
    /// Blocked from acceptance until the next IRET; edges meanwhile are
    /// ignored (80386 PRM §9.2.1 p. 9-3).
    BlockedIgnore,
}

impl NmiNesting {
    const fn word(self) -> u64 {
        match self {
            Self::Nested => 0,
            Self::BlockedRemember => 1,
            Self::BlockedIgnore => 2,
        }
    }
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
    /// External data-bus width in bits. On the twenty-bit models it sets the
    /// width of a device (Trap region) transfer: 8 moves a word as two byte
    /// cycles, 16 moves an even-addressed word in one cycle (8086 Family UM
    /// §2.3 p. 2-15; NEC V20/V30 UM §4 pp. 4-2–4-3; spec.md §5.2). The 80286
    /// and 80386 entries record their buses (80286 PRM §2.5.1 p. 2-23; 80386
    /// PRM §8.1.1 p. 8-1, which states 32-bit port alignment "so that the 32
    /// bits can be transferred in a single bus access"); their Machine paths
    /// do not read it. Only a value that departs from the canonical one feeds
    /// the identity, so a path that starts reading it on the 80286 or 80386
    /// must move that identity itself (a revision word).
    pub data_bus_bits: u8,
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
        (self.quirks.flags_fixed_zero == EARLY_FLAGS_FIXED_ZERO
            || matches!(self.isa, ModelIsa::I80486Dx | ModelIsa::I80486DxCpuid))
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
        max_processor_interrupt: 16,
        real_flags_set_mask: 0,
        divide_saves_next_ip: false,
        idiv_rejects_signed_minimum: false,
        wrap_segment_offsets: false,
        wrap_sequential_ip: false,
        rep_restart_prefixes: None,
        step_interrupt_handler: false,
        push_sp_after_decrement: false,
        shift_count_mask: Some(31),
        real_flags_clear_mask: 0xf000,
        flags_fixed_zero: EARLY_FLAGS_FIXED_ZERO,
        iret_delays_interrupts: false,
        segment_store_inhibits_interrupts: false,
        step_irq_conflict_refused: false,
        ei_delay_unconditional: false,
        lock_data_movement: true,
        coprocessor_overrun_vector: Some(9),
        i386_ignore_rep_non_string: false,
        nmi_nesting: NmiNesting::BlockedRemember,
    },
    system: ModelSystem::Msw286,
    physical_bits: 24,
    data_bus_bits: 16,
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
        max_processor_interrupt: 16,
        real_flags_set_mask: 0,
        divide_saves_next_ip: false,
        idiv_rejects_signed_minimum: false,
        wrap_segment_offsets: false,
        wrap_sequential_ip: false,
        rep_restart_prefixes: None,
        step_interrupt_handler: false,
        push_sp_after_decrement: false,
        shift_count_mask: Some(31),
        real_flags_clear_mask: 0x8000,
        flags_fixed_zero: EARLY_FLAGS_FIXED_ZERO,
        iret_delays_interrupts: false,
        segment_store_inhibits_interrupts: false,
        step_irq_conflict_refused: false,
        ei_delay_unconditional: false,
        lock_data_movement: false,
        coprocessor_overrun_vector: None,
        i386_ignore_rep_non_string: false,
        nmi_nesting: NmiNesting::BlockedIgnore,
    },
    system: ModelSystem::Cr386,
    physical_bits: 32,
    data_bus_bits: 32,
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

// Intel486 PRM §4.1.1 Fig. 4-1 p. 4-2: AC is implemented; Embedded
// Intel486 DM §D.1 p. D-1: ID is writable exactly when CPUID is present.
// DERIVED: inherit the 386 system/lift policies; these entries have no timing
// model. ASSUMED: retain its NMI policy pending a 486-specific event audit.
const fn i486_model(cpuid: bool) -> ModelProfile {
    let mut p = ModelProfile {
        isa: I80386.isa,
        quirks: I80386.quirks,
        system: I80386.system,
        physical_bits: I80386.physical_bits,
        data_bus_bits: I80386.data_bus_bits,
        max_insn_len: I80386.max_insn_len,
        fpu: I80386.fpu,
        timing: I80386.timing,
        semantics_id: 0,
        lift: I80386.lift,
    };
    p.isa = if cpuid {
        ModelIsa::I80486DxCpuid
    } else {
        ModelIsa::I80486Dx
    };
    p.quirks.flags_fixed_zero = if cpuid { 0 } else { 1 << 21 };
    // PRM §9.9.16 p. 9-24: alignment-check exception vector.
    p.quirks.max_processor_interrupt = 17;
    // PRM §4.1.3 p. 4-7: ET=1, 387-compatible integrated FPU.
    p.fpu = &[FpuModel::Present];
    identified(p)
}
pub(crate) static I80386_IGNORE_REP_NON_STRING: ModelProfile = identified(ModelProfile {
    quirks: ModelQuirks {
        i386_ignore_rep_non_string: true,
        ..I80386.quirks
    },
    ..I80386
});

/// The REP compatibility option requires the public I80386 catalogue entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I386RepQuirkError {
    /// Selected catalogue model; `None` for Application or the legacy selector.
    pub model: Option<CpuModel>,
}

impl std::fmt::Display for I386RepQuirkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "386 REP quirk requires CpuModel::I80386, selected {:?}",
            self.model
        )
    }
}

impl std::error::Error for I386RepQuirkError {}

static I80486_DX: ModelProfile = i486_model(false);
static I80486_DX_CPUID: ModelProfile = i486_model(true);
// ASSUMED model-1 stepping 0; Embedded 486 DM §D.2/Table D-2 pp. D-2–3.
pub(crate) const I486_DX_CPUID_SIGNATURE: u32 = 0x0410;
// ASSUMED: FPU only; DM p. D-2's EDX[3:1] field is ambiguous (decision-23).
pub(crate) const I486_DX_CPUID_FEATURES: u32 = 1;

// 8086 Family UM §§2.2–2.3 pp. 2-7–12, §2.6 pp. 2-24–28,
// §2.7 pp. 2-37–42; 286 PRM App. C pp. C-1–3; SDM Vol 3B §25.17.
// Present retains the generation-neutral engine; I8087 pairing arrives in C6.
// UM §2.3 p. 2-15: the 8088 "always accesses memory in bytes"; the 8086
// accesses an even-addressed word "in one bus cycle".
const fn early_model(identity: u64, data_bus_bits: u8) -> ModelProfile {
    let mut p = identified(ModelProfile {
        isa: ModelIsa::I8086,
        quirks: ModelQuirks {
            max_processor_interrupt: 4,
            real_flags_set_mask: 0xf000,
            divide_saves_next_ip: true,
            idiv_rejects_signed_minimum: true,
            wrap_segment_offsets: true,
            wrap_sequential_ip: true,
            rep_restart_prefixes: Some(1),
            step_interrupt_handler: true,
            push_sp_after_decrement: true,
            shift_count_mask: None,
            real_flags_clear_mask: 0,
            flags_fixed_zero: EARLY_FLAGS_FIXED_ZERO,
            iret_delays_interrupts: true,
            segment_store_inhibits_interrupts: false,
            step_irq_conflict_refused: false,
            ei_delay_unconditional: false,
            lock_data_movement: false,
            coprocessor_overrun_vector: None,
            i386_ignore_rep_non_string: false,
            nmi_nesting: NmiNesting::Nested,
        },
        system: ModelSystem::None,
        physical_bits: 20,
        data_bus_bits,
        max_insn_len: None,
        fpu: &[FpuModel::Present, FpuModel::None],
        timing: None,
        semantics_id: 0,
        lift: ModelLiftPolicy {
            direct_stack_fault: 0,
            available_tss_types: 0,
            lar_system_types: 0,
            lsl_system_types: 0,
            lsl_conforming: false,
            protected_coprocessor_outputs: false,
        },
    });
    p.semantics_id ^= identity;
    p
}
static I8088: ModelProfile = early_model(8088, 8);
static I8086: ModelProfile = early_model(8086, 16);

// NEC V20/V30 native mode (decision-23 C4a). Values are read from NEC
// uPD70108/70116 User's Manual, Stock No. 500350 (page images checked):
// - PSW image MD,1,1,1 above V (§3 p. 3-3); MD is 1 whenever a native
//   instruction executes (§8 p. 8-1), so lifted images store F000h;
// - vectors 0-5 are processor-assigned (§6 p. 6-1, Fig. 6-1, Table 6-1);
// - DIV/DIVU push PC, the offset of the next instruction (§12 pp. 12-56–60;
//   §3 p. 3-1), and the signed quotient stops at -127/-32767 (pp. 12-58–59).
//   Those pages' operation lines give the bound as "0-7FH-1"/"0-7FFFH-1"
//   (-128/-32768) with garbled comparison signs; the prose minimum, stated
//   with its hexadecimal value (81H/8001H), is taken;
// - offsets are 16-bit values added to segment x 16 on a 20-bit bus
//   (§10 Fig. 10-1 p. 10-1; §4 p. 4-1); PC is a 16-bit counter (§3 p. 3-1);
// - shift and rotate counts are used whole (pp. 12-102, 12-104);
// - an interrupted block instruction keeps up to three prefixes (§6 p. 6-4);
// - INT is held after a segment-register store, not after RETI (§6 p. 6-4),
//   and after every EI, whatever IE was (§6 p. 6-4; p. 12-172);
// - no instruction-length limit is stated.
// PUSH SP/POP SP and the single-step/INT order are not stated: the first is a
// typed refusal of the decode set, the second the refused-conflict field.
// - the V20 has an 8-bit data bus and the V30 a 16-bit one, which moves an
//   even-addressed word in one bus cycle (§4 pp. 4-2–4-3).
const fn nec_model(identity: u64, data_bus_bits: u8) -> ModelProfile {
    let mut p = identified(ModelProfile {
        isa: ModelIsa::NecV20,
        quirks: ModelQuirks {
            max_processor_interrupt: 5,
            real_flags_set_mask: 0xf000,
            divide_saves_next_ip: true,
            idiv_rejects_signed_minimum: true,
            wrap_segment_offsets: true,
            wrap_sequential_ip: true,
            rep_restart_prefixes: Some(3),
            step_interrupt_handler: false,
            push_sp_after_decrement: false,
            shift_count_mask: None,
            real_flags_clear_mask: 0,
            flags_fixed_zero: EARLY_FLAGS_FIXED_ZERO,
            iret_delays_interrupts: false,
            segment_store_inhibits_interrupts: true,
            step_irq_conflict_refused: true,
            ei_delay_unconditional: true,
            lock_data_movement: false,
            coprocessor_overrun_vector: None,
            i386_ignore_rep_non_string: false,
            nmi_nesting: NmiNesting::Nested,
        },
        system: ModelSystem::None,
        physical_bits: 20,
        data_bus_bits,
        max_insn_len: None,
        fpu: &[FpuModel::Present, FpuModel::None],
        timing: None,
        semantics_id: 0,
        lift: ModelLiftPolicy {
            direct_stack_fault: 0,
            available_tss_types: 0,
            lar_system_types: 0,
            lsl_system_types: 0,
            lsl_conforming: false,
            protected_coprocessor_outputs: false,
        },
    });
    p.semantics_id ^= identity;
    p
}
static NEC_V20: ModelProfile = nec_model(70108, 8);
static NEC_V30: ModelProfile = nec_model(70116, 16);

impl CpuModel {
    /// The immutable catalogue entry for this model (decision-23 §2).
    pub const fn profile(self) -> &'static ModelProfile {
        match self {
            Self::I8088 => &I8088,
            Self::I8086 => &I8086,
            Self::NecV20 => &NEC_V20,
            Self::NecV30 => &NEC_V30,
            Self::I80286 => &I80286,
            Self::I80386 => &I80386,
            Self::I80486Dx => &I80486_DX,
            Self::I80486DxCpuid => &I80486_DX_CPUID,
        }
    }

    /// Reset EDX and CPUID.1:EAX. ASSUMED representative stepping choices:
    /// 0403h for early DX (486 PRM §10.1 Fig. 10-1 pp. 10-1–2), 0410h
    /// for CPUID DX model 1 (Embedded 486 DM §D.2/Table D-2 pp. D-2–3).
    /// Neither manual assigns an exact stepping to CPUID availability.
    pub const fn i486_signature(self) -> Option<u32> {
        match self {
            Self::I80486Dx => Some(0x0403),
            Self::I80486DxCpuid => Some(I486_DX_CPUID_SIGNATURE),
            _ => None,
        }
    }

    pub(crate) const fn machine_flags_mask(self) -> u32 {
        match self {
            Self::I80486Dx | Self::I80486DxCpuid => {
                EARLY_FLAGS_FIXED_ZERO & !self.profile().quirks.flags_fixed_zero
            }
            _ => 0,
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

/// Compatibility lookup used only while lifting for the system models.
/// I8088/I8086 lowering and delivery read the selected entry instead
/// (decision-23 §2). The deprecated selector's I8086 generation is still
/// refused at run entry and keeps its C1 projection here.
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
    // Existing entries retain their canonical implicit defaults in the digest.
    // Any departure is explicit data, including a future change to an old model.
    let early_entry = p.quirks.max_processor_interrupt != 16
        || p.quirks.real_flags_set_mask != 0
        || p.quirks.divide_saves_next_ip
        || p.quirks.idiv_rejects_signed_minimum
        || p.quirks.wrap_segment_offsets
        || p.quirks.wrap_sequential_ip
        || p.quirks.rep_restart_prefixes.is_some()
        || p.quirks.step_interrupt_handler;
    if early_entry {
        let early = [
            p.quirks.max_processor_interrupt as u64,
            p.quirks.real_flags_set_mask as u64,
            p.quirks.divide_saves_next_ip as u64,
            p.quirks.idiv_rejects_signed_minimum as u64,
            p.quirks.wrap_segment_offsets as u64,
            p.quirks.wrap_sequential_ip as u64,
            match p.quirks.rep_restart_prefixes {
                Some(v) => v as u64,
                None => 0,
            },
            p.quirks.step_interrupt_handler as u64,
            // Byte-extension disposition, including the decision-23 alias
            // amendment, and prefix folding revision; 4 admits LOCK
            // (xj-50w.12).
            4,
        ];
        i = 0;
        while i < early.len() {
            hash = word(hash, early[i]);
            i += 1;
        }
    }
    // Interrupt-timing fields added with the NEC entries (decision-23 C4a).
    // Their canonical values are the 8086's inside the early block and the
    // system models' outside it, so existing identities are unchanged.
    if p.quirks.iret_delays_interrupts != early_entry
        || p.quirks.segment_store_inhibits_interrupts
        || p.quirks.step_irq_conflict_refused
        || p.quirks.ei_delay_unconditional
    {
        let timing = [
            p.quirks.iret_delays_interrupts as u64,
            p.quirks.segment_store_inhibits_interrupts as u64,
            p.quirks.step_irq_conflict_refused as u64,
            p.quirks.ei_delay_unconditional as u64,
        ];
        i = 0;
        while i < timing.len() {
            hash = word(hash, timing[i]);
            i += 1;
        }
    }
    // The NEC native-mode extension forms (decision-23 C4b), which the
    // twenty-bit fetch decodes outside the encoding table. Only the NEC
    // entries hash this word, so no other identity changes.
    if matches!(p.isa, ModelIsa::NecV20) {
        hash = word(hash, NEC_EXTENSION_REVISION);
    }
    // The 80286 LOCK and coprocessor-overrun fields (xj-50w.11). Their
    // canonical values are false and None, so only a departing entry hashes
    // them and no other identity changes.
    if p.quirks.lock_data_movement || p.quirks.coprocessor_overrun_vector.is_some() {
        hash = word(hash, p.quirks.lock_data_movement as u64);
        hash = word(
            hash,
            match p.quirks.coprocessor_overrun_vector {
                Some(v) => v as u64,
                None => 256,
            },
        );
    }
    // NMI nesting (xj-50w.19). Its canonical value is Nested for the early
    // entries, BlockedRemember for the 286 and BlockedIgnore for the 386, so
    // only a departing entry hashes it and no identity changes.
    let canonical_nmi = if early_entry {
        NmiNesting::Nested
    } else if matches!(p.system, ModelSystem::Msw286) {
        NmiNesting::BlockedRemember
    } else {
        NmiNesting::BlockedIgnore
    };
    if p.quirks.nmi_nesting.word() != canonical_nmi.word() {
        hash = word(hash, 0x4e4d49);
        hash = word(hash, p.quirks.nmi_nesting.word());
    }
    // Data-bus width (xj-50w.20). Its canonical value is 8 for the early
    // entries, 16 for the 286 and 32 for the 386, so only a departing entry
    // (the I8086 and NEC V30) hashes it and no other identity changes.
    let canonical_bus = if early_entry {
        8
    } else if matches!(p.system, ModelSystem::Msw286) {
        16
    } else {
        32
    };
    if p.data_bus_bits != canonical_bus {
        hash = word(hash, 0x627573);
        hash = word(hash, p.data_bus_bits as u64);
    }
    if matches!(p.isa, ModelIsa::I80486Dx | ModelIsa::I80486DxCpuid) {
        // 486-specific lowering/helper revision; existing model IDs stay fixed.
        hash = word(hash, 1);
        if matches!(p.isa, ModelIsa::I80486DxCpuid) {
            hash = word(hash, I486_DX_CPUID_SIGNATURE as u64);
            hash = word(hash, I486_DX_CPUID_FEATURES as u64);
        }
    }
    // Preserve canonical catalogue identities; only the selectable 386 variant
    // adds this domain word and its quirk value (decision-23 §3).
    if p.quirks.i386_ignore_rep_non_string {
        hash = word(hash, 0x726570333836);
        hash = word(hash, p.quirks.i386_ignore_rep_non_string as u64);
    }
    p.semantics_id = hash;
    p
}

// Bump when the NEC extension decode or lowering changes (decision-23 C4b).
// 2: the segment-register store holds NMI as well as INT (xj-50w.19).
const NEC_EXTENSION_REVISION: u64 = 2;

#[cfg(test)]
mod tests {
    use super::*;
    fn copy(p: &ModelProfile) -> ModelProfile {
        ModelProfile {
            isa: p.isa,
            quirks: p.quirks,
            system: p.system,
            physical_bits: p.physical_bits,
            data_bus_bits: p.data_bus_bits,
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
            for change in 0..26 {
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
                    8 => changed.lift.available_tss_types ^= 1,
                    9 => changed.quirks.max_processor_interrupt = 4,
                    10 => changed.quirks.real_flags_set_mask = 0xf000,
                    11 => changed.quirks.divide_saves_next_ip = true,
                    12 => changed.quirks.idiv_rejects_signed_minimum = true,
                    13 => changed.quirks.wrap_segment_offsets = true,
                    14 => changed.quirks.wrap_sequential_ip = true,
                    15 => changed.quirks.rep_restart_prefixes = Some(1),
                    16 => changed.quirks.iret_delays_interrupts = true,
                    17 => changed.quirks.segment_store_inhibits_interrupts = true,
                    18 => changed.quirks.step_irq_conflict_refused = true,
                    19 => changed.quirks.ei_delay_unconditional = true,
                    20 => changed.quirks.lock_data_movement ^= true,
                    21 => {
                        changed.quirks.coprocessor_overrun_vector =
                            match changed.quirks.coprocessor_overrun_vector {
                                Some(_) => None,
                                None => Some(9),
                            }
                    }
                    22 => changed.quirks.nmi_nesting = NmiNesting::Nested,
                    23 => changed.data_bus_bits = 8,
                    24 => changed.quirks.i386_ignore_rep_non_string = true,
                    _ => changed.quirks.step_interrupt_handler = true,
                }
                assert_ne!(identified(changed).semantics_id, p.semantics_id);
            }
        }
        assert_ne!(I80286.semantics_id, I80386.semantics_id);
    }

    // Historical prefix decoder options participate in every Machine identity.
    // Pin the catalogue after selecting NOP/BSF/BSR/WBINVD instead of their
    // later mandatory-prefix interpretations (spec.md §17.10).
    #[test]
    fn machine_identities_include_historical_prefix_policy() {
        for (model, identity) in [
            (CpuModel::I80386, 0xc82b_1bb8_08a6_d958_u64),
            (CpuModel::I8088, 0x0b71_3477_b270_a8b9),
            (CpuModel::I8086, 0xe283_0b47_9d3b_74e5),
            (CpuModel::NecV20, 0xd70e_6f86_6f32_e09c),
            (CpuModel::NecV30, 0x9868_73c6_78e7_9ab2),
        ] {
            assert_eq!(model.profile().semantics_id, identity, "{model:?}");
            assert!(!model.profile().quirks.lock_data_movement, "{model:?}");
            assert_eq!(model.profile().quirks.coprocessor_overrun_vector, None);
        }
        let q = CpuModel::I80286.profile().quirks;
        assert!(q.lock_data_movement);
        assert_eq!(q.coprocessor_overrun_vector, Some(9));
        assert_eq!(I80286.semantics_id, 0xa812_1b75_802e_461b);
        // The two quirk fields feed the identity.
        let mut canonical = copy(&I80286);
        canonical.quirks.lock_data_movement = false;
        canonical.quirks.coprocessor_overrun_vector = None;
        assert_ne!(identified(canonical).semantics_id, I80286.semantics_id);
    }

    // Decision-23 C4a: the NEC entries share content and differ in identity;
    // their content digest differs from the 8086's in every new field.
    #[test]
    fn nec_identity_tracks_its_fields() {
        assert_eq!(
            identified(copy(&NEC_V20)).semantics_id,
            NEC_V20.semantics_id ^ 70108
        );
        assert_eq!(
            identified(copy(&NEC_V30)).semantics_id,
            NEC_V30.semantics_id ^ 70116
        );
        assert_eq!(
            identified(copy(&I8088)).semantics_id,
            I8088.semantics_id ^ 8088
        );
        let base = identified(copy(&NEC_V20)).semantics_id;
        for change in 0..7 {
            let mut changed = copy(&NEC_V20);
            match change {
                0 => changed.quirks.rep_restart_prefixes = Some(1),
                4 => changed.quirks.nmi_nesting = NmiNesting::BlockedIgnore,
                5 => changed.data_bus_bits = 16,
                1 => changed.quirks.iret_delays_interrupts = true,
                2 => changed.quirks.segment_store_inhibits_interrupts = false,
                3 => changed.quirks.ei_delay_unconditional = false,
                _ => changed.quirks.step_irq_conflict_refused = false,
            }
            assert_ne!(identified(changed).semantics_id, base, "{change}");
        }
        // xj-50w.20: the bus siblings differ in content only by the data bus.
        let mut v30 = copy(&NEC_V30);
        v30.data_bus_bits = 8;
        assert_eq!(identified(v30).semantics_id, base);
        let mut i8086 = copy(&I8086);
        i8086.data_bus_bits = 8;
        assert_eq!(
            identified(i8086).semantics_id,
            identified(copy(&I8088)).semantics_id
        );
        let mut as_8086 = copy(&NEC_V20);
        as_8086.isa = ModelIsa::I8086;
        assert_ne!(identified(as_8086).semantics_id, base);
    }
}
