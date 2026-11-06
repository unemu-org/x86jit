//! Return-based exit reasons and the backend execution result (§5.2, §8).

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
    Syscall,
    /// Guest executed `hlt`.
    Hlt,
    /// Access to an unmapped address, or instruction fetch from a `RegionKind::Trap`
    /// region (`access: AccessKind::Execute`). Trap backing is never decoded.
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
    /// Guest executed a port-I/O instruction (`in`/`out`, §5.2) — the machine
    /// counterpart of MMIO. RIP is already advanced past the instruction (like
    /// `Syscall`), so the embedder services it and re-enters `run()`. For
    /// `PortDir::Out`, `value` holds the low `size` bytes the guest wrote. For
    /// `PortDir::In`, the embedder must call `complete_port_in` with the port's
    /// value before the next `run()`; `value` is 0. `port` is the 16-bit port
    /// number, `size` the access width (1/2/4 bytes).
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
