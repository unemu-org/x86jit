//! Return-based exit reasons and the backend execution result (§5.2, §8).

/// Owned execution identity for a non-fault exit (§5.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionEvidence {
    path: crate::FaultPath,
    rip: u64,
}

impl ExecutionEvidence {
    /// Execution unit that produced the exit. Compiled paths use the function
    /// entry as their code-map key; a site is supplied only for an exact match.
    pub const fn path(&self) -> crate::FaultPath {
        self.path
    }

    /// Guest instruction RIP for a syscall; resume RIP for an interruption.
    pub const fn rip(&self) -> u64 {
        self.rip
    }

    pub(crate) const fn new(path: crate::FaultPath, rip: u64) -> Self {
        Self { path, rip }
    }
}

/// Direction of a memory access that could not complete inline.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
    Execute,
}

/// Direction of a port-I/O access (§5.2).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PortDir {
    /// `in al/ax/eax, …` — the guest reads a port; the embedder must supply the value.
    In,
    /// `out …, al/ax/eax` — the guest writes `value` to a port; the embedder acts on it.
    Out,
}

/// Engine operations whose system-context implementation is unavailable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnsupportedSystemOperation {
    LegacyExecution,
    LegacyStepping,
    LegacyEventDelivery,
    LegacyInstruction,
    LegacyTaskReturn,
    LegacyEntryFacility,
    ApplicationIrqDelivery,
    InvalidLegacyContext(crate::legacy::MachineContextError),
    LegacyLdtSelector,
    LegacyGateOrTaskTransfer,
    LegacyPrivilegeTransition,
    LegacyTaskPaging,
    LegacyTaskDebugTrap,
    /// NEC V20/V30: BRKEM while the MD write-enable latch is set. Emulation
    /// mode cannot be nested and "normal operation cannot be expected"
    /// (NEC V20/V30 UM §8 p. 8-5).
    NecNestedEmulation,
    /// NEC V20/V30: POP PSW would clear MD between BRKEM and RETEM. NEC UM
    /// p. 12-151 says not to do this and describes no result.
    NecPswModeWrite,
    /// NEC V20/V30: RETEM popped a PSW with MD = 0. NEC UM p. 12-177
    /// describes only the frame BRKEM saved.
    NecRetemToEmulation,
    /// NEC V20/V30: BRK = 1 where the single-step break would meet the mode
    /// switch: at an 8080-mode boundary, at the start of a BRKEM, or at the
    /// start of a RETI that would re-enter 8080 mode. NEC UM §6 p. 6-3 gives the
    /// general break rule; how it combines with the mode switch is not
    /// described.
    NecEmulationSingleStep,
    /// NEC V20/V30: a single-step trap and an acceptable INT at one boundary.
    /// NEC UM Table 6-1 p. 6-1 gives priorities, not the sequence.
    NecStepWithInterrupt,
    /// NEC V20/V30: CHKIND bounds on which a signed and an unsigned
    /// comparison disagree. NEC UM p. 12-170 does not say which applies.
    NecIndexComparison,
    /// NEC V20/V30: INS or EXT with a nonzero upper four bits in its
    /// bit-offset register or bit-length register. NEC UM pp. 12-21–24 require
    /// them to be zero "for correct operation" and describe no other result.
    NecBitFieldOperand,
    /// NEC V20/V30: ADD4S, SUB4S or CMP4S with CL = 0, 255 or odd. NEC UM
    /// pp. 12-42–44 give 1 to 254 digits, and leave the upper digit and the
    /// Z and CY flags undefined for an odd count.
    NecBcdStringLength,
    /// NEC V20/V30: a digit above 9 in an ADD4S, SUB4S or CMP4S operand.
    /// NEC UM pp. 12-42–44 describe packed BCD strings only.
    NecBcdDigit,
    /// NEC V20/V30: an ADD4S, SUB4S or CMP4S operand byte in a trapped
    /// (device) region. An engine limit, not a gap in the page: the
    /// instruction is one unit of up to 381 transfers, and a replay after an
    /// MMIO stop would read RAM bytes it had already stored.
    NecBcdStringDevice,
    /// NEC V20/V30: ADD4S or SUB4S whose destination byte i is, at its
    /// physical address, the source byte j of a later step (j > i). NEC UM
    /// pp. 12-42–43 give "Transfers: 3n" but no order of the reads and
    /// stores, and this overlap is where the order shows.
    NecBcdStringOverlap,
}

/// Reason `run()` returned control to the user (§5.2).
///
/// MMIO exits describe one transfer of at most eight bytes. A wider memory
/// operand produces several exits at the same RIP; `total_size` and `part`
/// identify the operand width and transfer slot without decoding the instruction.
///
/// This enum and its struct-like variants are non-exhaustive so new exit reasons
/// and fields can be added. Outside this crate, matches need a wildcard arm and
/// struct-like patterns need `..`:
///
/// ```
/// use x86jit_core::Exit;
///
/// fn handle(exit: Exit) {
///     match exit {
///         Exit::MmioRead { addr, size, .. } => { /* service the read */ }
///         Exit::MmioWrite { addr, size, value, .. } => { /* service the write */ }
///         _ => { /* handle other or future exits */ }
///     }
/// }
/// ```
///
/// To synthesise exits, use the stable constructors for the struct-like variants:
///
/// ```
/// use x86jit_core::{AccessKind, Exit, PortDir};
///
/// const UNMAPPED: Exit = Exit::unmapped_memory(0x1000, AccessKind::Read);
/// const MMIO_READ: Exit = Exit::mmio_read(0x3000, 4);
/// const MMIO_WRITE: Exit = Exit::mmio_write(0x4000, 8, 0x1234);
/// const UNKNOWN: Exit = Exit::unknown_instruction(0x5000, [0xab; 15], 3);
/// const EXCEPTION: Exit = Exit::exception(0x2000, 42);
/// const PORT_IO: Exit = Exit::port_io(0x1234, 2, PortDir::Out, 0x5678);
/// const CODE_SPACE: Exit = Exit::code_space_exhausted(0x6000);
///
/// assert!(matches!(UNMAPPED, Exit::UnmappedMemory {
///     addr: 0x1000, access: AccessKind::Read, ..
/// }));
/// assert!(matches!(MMIO_READ, Exit::MmioRead {
///     addr: 0x3000, size: 4, total_size: 4, part: 0, ..
/// }));
/// assert!(matches!(MMIO_WRITE, Exit::MmioWrite {
///     addr: 0x4000, size: 8, value: 0x1234, total_size: 8, part: 0, ..
/// }));
/// assert!(matches!(UNKNOWN, Exit::UnknownInstruction {
///     addr: 0x5000, bytes, len: 3, ..
/// } if bytes == [0xab; 15]));
/// assert!(matches!(EXCEPTION, Exit::Exception {
///     addr: 0x2000, vector: 42, ..
/// }));
/// assert!(matches!(PORT_IO, Exit::PortIo {
///     port: 0x1234, size: 2, dir: PortDir::Out, value: 0x5678, ..
/// }));
/// assert!(matches!(CODE_SPACE, Exit::CodeSpaceExhausted { addr: 0x6000, .. }));
/// ```
///
/// Listing every current variant still requires a wildcard arm:
///
/// ```compile_fail,E0004
/// use x86jit_core::Exit;
///
/// fn handle(exit: Exit) {
///     match exit {
///         Exit::Syscall | Exit::Hlt | Exit::BudgetExhausted => {}
///         Exit::UnmappedMemory { .. } | Exit::MmioRead { .. }
///         | Exit::MmioWrite { .. } | Exit::UnknownInstruction { .. }
///         | Exit::Exception { .. } | Exit::PortIo { .. }
///         | Exit::CodeSpaceExhausted { .. } => {}
///     }
/// }
/// ```
///
/// Struct literals for these variants are unavailable outside this crate:
///
/// ```compile_fail,E0639
/// use x86jit_core::Exit;
///
/// let exit = Exit::MmioRead { addr: 0x1000, size: 4 };
/// ```
///
/// ```compile_fail,E0639
/// use x86jit_core::Exit;
///
/// let exit = Exit::MmioWrite { addr: 0x1000, size: 4, value: 0 };
/// ```
///
/// ```compile_fail,E0639
/// use x86jit_core::Exit;
///
/// let exit = Exit::CodeSpaceExhausted { addr: 0x1000 };
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Exit {
    /// Guest executed syscall/sysenter/int 0x80. Args are in guest registers.
    /// Execution identity and instruction RIP are available through
    /// [`crate::Vcpu::syscall_evidence`] until the next `run`.
    Syscall,
    /// Guest executed `hlt`.
    Hlt,
    /// Access to an unmapped address, or instruction fetch from a `RegionKind::Trap`
    /// region (`access: AccessKind::Execute`). Trap backing is never decoded.
    /// For a non-atomic scalar integer write rejected by a mapping boundary, `addr`
    /// identifies the first unavailable byte; the write has not partially committed.
    /// Scalar floating-point stores report the failed preflight transfer's base
    /// (the operand base for an unsplit operand), rather than the first missing byte.
    /// For an execute fault, `addr` identifies the first unavailable fetch byte
    /// (including a Trap boundary); RIP stays on the attempted instruction's start.
    /// Fetch continues across adjacent mapped RAM/ROM regions.
    /// Earlier instructions have completed; the incomplete instruction has no effects.
    #[non_exhaustive]
    UnmappedMemory {
        addr: u64,
        access: AccessKind,
        /// Optional owned details; legacy constructors leave this absent.
        metadata: Option<Box<crate::fault::FaultMetadata>>,
    },
    /// READ from a Trap region (MMIO). Guest waits; user must call
    /// `complete_mmio_read` before the next `run()`.
    #[non_exhaustive]
    MmioRead {
        /// Address of this transfer, rather than the operand's base address.
        addr: u64,
        /// Bytes requested by this transfer (1–8); the answer occupies the low
        /// `size` bytes passed to `complete_mmio_read`.
        size: u8,
        /// Full memory operand width in bytes, including masked-off elements
        /// and any bytes served by RAM. This is the source width for broadcasts
        /// and the destination width for narrowing stores, rather than the
        /// vector register width or the count of active bytes.
        total_size: u16,
        /// Zero-based transfer slot within the memory operand:
        /// `addr.wrapping_sub(operand_base) / size`. Ordinary wide accesses use
        /// 8-byte slots. Per-element masked memory transfers and narrowing stores
        /// use element-sized slots, retaining gaps for inactive elements.
        /// Whole-table reads, including masked VPERMD and GFNI affine memory
        /// forms, use 8-byte chunks: a 32-byte table reports parts 0, 1, 2 and 3,
        /// regardless of the element width. RAM-served slots also retain
        /// their indices. Scalar accesses have `total_size == size`, `part == 0`.
        part: u8,
    },
    /// WRITE to a Trap region (MMIO). User handles the side effect, then resumes.
    #[non_exhaustive]
    MmioWrite {
        /// Address of this transfer, rather than the operand's base address.
        addr: u64,
        /// Bytes written by this transfer (1–8).
        size: u8,
        /// This transfer's bytes, in the low `size` bytes of the value.
        value: u64,
        /// Full memory operand width in bytes, with the same meaning as
        /// the `total_size` field on [`Exit::MmioRead`]. Masked-off bytes still count.
        total_size: u16,
        /// Zero-based transfer slot, with the same meaning as
        /// the `part` field on [`Exit::MmioRead`]. Masked-off and RAM-served slots
        /// keep their indices. Scalar accesses have `total_size == size`, `part == 0`.
        part: u8,
    },
    /// A decoded instruction the lift does not yet support. `addr` and RIP both
    /// identify that instruction; earlier instructions have completed.
    #[non_exhaustive]
    UnknownInstruction { addr: u64, bytes: [u8; 15], len: u8 },
    /// A guest CPU exception, distinct from a lift failure. `vector` is the x86
    /// exception vector; the embedder decides how to deliver it.
    /// `addr` is the guest **saved RIP**: a fault leaves it on the faulting
    /// instruction, a trap (`#BP`/`#DB`) resumes past it. It always equals the
    /// vcpu's RIP at exit. Earlier instructions have completed.
    ///
    /// Non-canonical Long64 data operands raise #GP (13), or #SS (12) for SS
    /// references; required operand alignment raises #GP (13). Error code zero
    /// is implied for these exits. Registers, flags and memory retain their
    /// instruction-entry values (SDM Vol 1 §3.3.7.1; Vol 2A §2.5;
    /// Vol 3A §7.15, Events 12/13; AMD APM Vol 2 §5.3.1).
    /// Invalid opcodes produce #UD (6); in Long64/Compat32, instructions exceeding
    /// 15 bytes produce #GP (13). Divide errors produce #DE (0)
    /// (SDM Vol 3A §7.15, Events 0/6/13; Vol 2A §2.3.11).
    #[non_exhaustive]
    Exception {
        addr: u64,
        vector: u8,
        /// Optional owned details; legacy constructors leave this absent.
        metadata: Option<Box<crate::fault::FaultMetadata>>,
    },
    /// Guest handed one port access to the embedder (§5.2).
    /// IN/OUT advance RIP. INS/OUTS retain the instruction RIP while a REP
    /// has more elements; INS also retains it until its pending store succeeds.
    /// OUT/OUTS carry the low `size` bytes in `value`. For IN/INS, `value` is
    /// zero and the embedder must call `complete_port_in` before resuming.
    /// `port` is DX's low 16 bits (or IN/OUT's immediate), `size` is 1/2/4.
    #[non_exhaustive]
    PortIo {
        port: u16,
        size: u8,
        dir: PortDir,
        value: u64,
    },
    /// `budget` blocks executed — cooperative yield.
    BudgetExhausted,
    /// A host thread requested a yield at a checked boundary. Guest state is
    /// resumable; REP may have completed only some iterations.
    /// Like mid-block faults, this may over-charge the executed-instruction counter:
    /// compiled code charges the whole block at entry, including its unexecuted tail.
    /// Evidence for this yield is available from [`crate::Vcpu::interruption_evidence`]
    /// until the next call to [`crate::Vcpu::run`].
    Interrupted,
    /// JIT code storage could not grow or its background compiler could not start.
    /// Returned once per vcpu, at a block
    /// boundary; `addr` is its current RIP and no instruction at that boundary
    /// has executed yet. Resume with `run()` to continue. The backend permanently
    /// stops compiling new code and interprets blocks it cannot compile; existing
    /// translations remain usable. This does not report global allocator failure.
    /// On a weak host, interpreter fallback provides only the interpreter's
    /// ordering for ordinary loads/stores; it does not honor the configured
    /// [`MemConsistency::AcqRel`](crate::vm::MemConsistency::AcqRel) or
    /// [`MemConsistency::FullTso`](crate::vm::MemConsistency::FullTso). An embedder
    /// relying on either tier for a multithreaded guest should treat this exit
    /// as fatal.
    #[non_exhaustive]
    CodeSpaceExhausted { addr: u64 },
    /// Unsupported engine scope; no guest effect and no architectural fault.
    #[non_exhaustive]
    UnsupportedSystemContext {
        addr: u64,
        operation: UnsupportedSystemOperation,
    },
    /// The processor faulted while invoking #DF. Execution remains stopped until
    /// `Vcpu::reset_execution_state`; the embedder supplies the reset image.
    /// I286 PRM §9.6.2 p. 9-10; I386 PRM §9.8.8 p. 9-13.
    #[non_exhaustive]
    Shutdown { addr: u64 },
    /// NEC V20/V30 only: an instruction boundary with PSW.MD = 0 (8080
    /// emulation mode, NEC V20/V30 UM §8 p. 8-1) at which the engine has
    /// nothing to execute. Zero steps; nothing is in flight. The embedder runs
    /// an 8080 core on `view` and returns through
    /// [`crate::Vcpu::hand_back_i8080`] (decision-23 C4a; spec.md §17.10). The
    /// core's stop test ORs in [`crate::ExitRequest::is_requested`].
    #[non_exhaustive]
    I8080Mode { view: I8080View, entry: I8080Entry },
}

/// How a NEC V20/V30 vCPU reached the boundary of an [`Exit::I8080Mode`].
/// Diagnostic only; the hand-back does not depend on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum I8080Entry {
    /// BRKEM imm8 completed since the last I8080Mode exit (NEC V20/V30 UM
    /// p. 12-169).
    Brkem { vector: u8 },
    /// RETI restored MD = 0 since the last I8080Mode exit (NEC V20/V30 UM
    /// p. 12-168).
    Reti,
    /// The vCPU was already in 8080 mode, for example after a hand-back.
    Resume,
}

/// The 8080-visible state of a NEC V20/V30 in emulation mode, by value
/// (NEC V20/V30 UM §8 p. 8-1, Figs. 8-1 and 8-2; register encodings §12
/// Tables 12-6/12-7 p. 12-3). The fields an 8080 instruction can change are
/// written back by [`crate::Vcpu::hand_back_i8080`]; the segments are not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct I8080View {
    /// A: AL. AH is not addressable in emulation mode and is not exchanged.
    pub a: u8,
    /// 8080 flags: PSW bits 7:0, S Z 0 AC 0 P 1 CY.
    pub flags: u8,
    /// BC: CW (CX); B is CH, C is CL.
    pub bc: u16,
    /// DE: DW (DX); D is DH, E is DL.
    pub de: u16,
    /// HL: BW (BX); H is BH, L is BL.
    pub hl: u16,
    /// The 8080 stack pointer: BP. Its stack is in the data segment.
    pub sp: u16,
    /// The 8080 PC: PC (IP), an offset in the code segment.
    pub pc: u16,
    /// PSW.IE (IF). That the 8080 EI and DI set and clear it is ASSUMED
    /// (contract draft S4); NEC V20/V30 UM §8 does not say.
    pub interrupt_enable: bool,
    /// The 8080 core's EI delay: INT is not accepted at this boundary.
    /// Stored and returned verbatim; the zero-step exit never clears it.
    pub ei_delay: bool,
    /// The 8080 core is in standby after HLT (NEC V20/V30 UM §9 p. 9-1).
    pub halted: bool,
    /// PS (CS): instruction and operand fetches.
    pub code_segment: u16,
    /// DS0 (DS): data operands and the 8080 stack.
    pub data_segment: u16,
}

impl I8080View {
    /// Twenty-bit address of a code-segment offset (NEC UM §10 Fig. 10-1).
    pub const fn code_address(&self, offset: u16) -> u32 {
        ((self.code_segment as u32) * 16 + offset as u32) & 0xf_ffff
    }

    /// Twenty-bit address of a data-segment offset (NEC UM §10 Fig. 10-1).
    pub const fn data_address(&self, offset: u16) -> u32 {
        ((self.data_segment as u32) * 16 + offset as u32) & 0xf_ffff
    }
}

/// How the 8080 core stopped, for [`crate::Vcpu::hand_back_i8080`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum I8080Handback {
    /// Stopped on opcode EDh at `pc` without consuming it. The engine decodes
    /// that byte again at PS:PC as its next unit: RETEM (ED FD) or CALLN
    /// (ED ED imm8), each one step; any other ED form is
    /// [`Exit::UnknownInstruction`].
    EdEscape,
    /// Stopped at an instruction boundary (budget, host stop or interrupt).
    Boundary,
    /// Refused the opcode at `pc`, which is not EDh. MD stays 0; until the next
    /// hand-back, each run at this PS:PC returns [`Exit::UnknownInstruction`]
    /// with that byte. An interrupt taken at this boundary also ends it.
    Refused { opcode: u8 },
}

/// A hand-back the engine rejected; the vCPU is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum I8080HandbackError {
    /// The vCPU is not a NEC V20/V30 in 8080 mode.
    NotInI8080Mode,
    /// The view names other segments; emulation mode cannot change them
    /// (NEC V20/V30 UM §8 p. 8-1).
    SegmentChanged,
    /// A halted 8080 core cannot stop on an opcode.
    HaltedStop,
    /// `Refused` named EDh, the escape byte; an ED stop is `EdEscape`.
    RefusedEscape,
    /// An MMIO transfer is in flight: the last call stopped a frame access of
    /// RETEM, CALLN or an interrupt taken in 8080 mode with an MMIO exit.
    /// Complete it and run again first; a pending escape stays pending.
    InFlight,
}

impl Exit {
    /// Details of this owned fault event, if the producer supplied them.
    pub fn fault_metadata(&self) -> Option<&crate::fault::FaultMetadata> {
        match self {
            Self::UnmappedMemory { metadata, .. } | Self::Exception { metadata, .. } => {
                metadata.as_deref()
            }
            _ => None,
        }
    }

    /// Attach immutable metadata. Only memory and exception exits accept it.
    /// Allocation occurs here, on the cold fault path, outside signal context.
    #[must_use]
    pub fn with_fault_metadata(mut self, details: crate::fault::FaultMetadata) -> Self {
        if let Self::UnmappedMemory { metadata, .. } | Self::Exception { metadata, .. } = &mut self
        {
            *metadata = Some(Box::new(details));
        }
        self
    }

    /// Construct an [`Exit::CodeSpaceExhausted`] at a resumable block boundary.
    #[must_use]
    pub const fn code_space_exhausted(addr: u64) -> Self {
        Self::CodeSpaceExhausted { addr }
    }

    /// Construct an [`Exit::UnmappedMemory`] for an embedder-reported memory fault.
    ///
    /// This is the stable construction API for this non-exhaustive variant:
    /// fields added later receive defaults through this constructor.
    #[must_use]
    pub const fn unmapped_memory(addr: u64, access: AccessKind) -> Self {
        Self::UnmappedMemory {
            addr,
            access,
            metadata: None,
        }
    }

    /// Construct an [`Exit::MmioRead`].
    ///
    /// Defaults to a single transfer: `total_size == size`, `part == 0`.
    ///
    /// This is the stable construction API for this non-exhaustive variant:
    /// fields added later receive defaults through this constructor.
    #[must_use]
    pub const fn mmio_read(addr: u64, size: u8) -> Self {
        Self::MmioRead {
            addr,
            size,
            total_size: size as u16,
            part: 0,
        }
    }

    /// Construct an [`Exit::MmioWrite`].
    ///
    /// Defaults to a single transfer: `total_size == size`, `part == 0`.
    ///
    /// This is the stable construction API for this non-exhaustive variant:
    /// fields added later receive defaults through this constructor.
    #[must_use]
    pub const fn mmio_write(addr: u64, size: u8, value: u64) -> Self {
        Self::MmioWrite {
            addr,
            size,
            value,
            total_size: size as u16,
            part: 0,
        }
    }

    /// Construct an [`Exit::UnknownInstruction`].
    ///
    /// This is the stable construction API for this non-exhaustive variant:
    /// fields added later receive defaults through this constructor.
    #[must_use]
    pub const fn unknown_instruction(addr: u64, bytes: [u8; 15], len: u8) -> Self {
        Self::UnknownInstruction { addr, bytes, len }
    }

    /// Construct an [`Exit::Exception`] for an embedder-reported guest exception.
    ///
    /// This is the stable construction API for this non-exhaustive variant:
    /// fields added later receive defaults through this constructor.
    /// The arguments follow the [`Exit::Exception`] field contract.
    #[must_use]
    pub const fn exception(addr: u64, vector: u8) -> Self {
        Self::Exception {
            addr,
            vector,
            metadata: None,
        }
    }

    /// Construct an [`Exit::PortIo`].
    ///
    /// This is the stable construction API for this non-exhaustive variant:
    /// fields added later receive defaults through this constructor.
    #[must_use]
    pub const fn port_io(port: u16, size: u8, dir: PortDir, value: u64) -> Self {
        Self::PortIo {
            port,
            size,
            dir,
            value,
        }
    }
}

/// Result of executing one materialized block. Distinguishes "keep going" from
/// "trap out" so the dispatcher knows whether to continue or return (§8).
pub enum StepResult {
    /// Block finished; new RIP already written to `CpuState`. Continue.
    Continue,
    /// Block trapped out to the user. Execution stops.
    Exit(Exit),
}
