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
///
/// assert!(matches!(UNMAPPED, Exit::UnmappedMemory {
///     addr: 0x1000, access: AccessKind::Read, ..
/// }));
/// assert!(matches!(MMIO_READ, Exit::MmioRead {
///     addr: 0x3000, size: 4, ..
/// }));
/// assert!(matches!(MMIO_WRITE, Exit::MmioWrite {
///     addr: 0x4000, size: 8, value: 0x1234, ..
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
///         | Exit::Exception { .. } | Exit::PortIo { .. } => {}
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
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Exit {
    /// Guest executed syscall/sysenter/int 0x80. Args are in guest registers.
    Syscall,
    /// Guest executed `hlt`.
    Hlt,
    /// Access to an unmapped address, or instruction fetch from a `RegionKind::Trap`
    /// region (`access: AccessKind::Execute`). Trap backing is never decoded.
    /// For a Trap fetch, `addr` and RIP identify the attempted instruction's start
    /// (the decode IP in Real16). An instruction straddling a region boundary also
    /// exits with `Execute` at its start, without reading bytes from the next region.
    /// Earlier instructions have completed; the incomplete instruction has no effects.
    #[non_exhaustive]
    UnmappedMemory { addr: u64, access: AccessKind },
    /// READ from a Trap region (MMIO). Guest waits; user must call
    /// `complete_mmio_read` before the next `run()`.
    #[non_exhaustive]
    MmioRead { addr: u64, size: u8 },
    /// WRITE to a Trap region (MMIO). User handles the side effect, then resumes.
    #[non_exhaustive]
    MmioWrite { addr: u64, size: u8, value: u64 },
    /// An instruction the lift does not yet support — tells you what to add next.
    #[non_exhaustive]
    UnknownInstruction { addr: u64, bytes: [u8; 15], len: u8 },
    /// A guest CPU exception, NOT a lift failure: `#DE` divide-by-zero, `#UD`
    /// (`ud2`), `#BP` (`int3`), `#DB` (`int1`), etc. HLE maps these to
    /// SIGFPE/SIGILL/SIGTRAP. `vector` = x86 exception vector (§14 open decision).
    /// `addr` is the guest **saved RIP** — the x86 fault/trap convention: a fault
    /// (`#DE`/`#UD`) leaves it on the faulting instruction, a trap (`#BP`/`#DB`)
    /// resumes past it. It always equals the vcpu's RIP at exit.
    #[non_exhaustive]
    Exception { addr: u64, vector: u8 },
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
}

impl Exit {
    /// Construct an [`Exit::UnmappedMemory`] for an embedder-reported memory fault.
    ///
    /// This is the stable construction API for this non-exhaustive variant:
    /// fields added later receive defaults through this constructor.
    #[must_use]
    pub const fn unmapped_memory(addr: u64, access: AccessKind) -> Self {
        Self::UnmappedMemory { addr, access }
    }

    /// Construct an [`Exit::MmioRead`].
    ///
    /// This is the stable construction API for this non-exhaustive variant:
    /// fields added later receive defaults through this constructor.
    #[must_use]
    pub const fn mmio_read(addr: u64, size: u8) -> Self {
        Self::MmioRead { addr, size }
    }

    /// Construct an [`Exit::MmioWrite`].
    ///
    /// This is the stable construction API for this non-exhaustive variant:
    /// fields added later receive defaults through this constructor.
    #[must_use]
    pub const fn mmio_write(addr: u64, size: u8, value: u64) -> Self {
        Self::MmioWrite { addr, size, value }
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
        Self::Exception { addr, vector }
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
