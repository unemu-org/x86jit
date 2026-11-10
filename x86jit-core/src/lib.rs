//! `x86jit-core` — a guest-agnostic x86-64 recompiler engine.
//!
//! The core knows nothing about ELF, syscalls of any concrete OS, or GPUs.
//! It is fed a memory map plus an entry point and executes guest instructions,
//! yielding control through [`Exit`] whenever it hits something it does not
//! handle itself. See `backlog/docs/design/spec.md` for the full design.
//!
//! Fault exits optionally own [`FaultMetadata`], queried with
//! [`Exit::fault_metadata`]. Engine faults retain reporting/saved RIP, available
//! transfer widths, restart state and cause-specific error-code evidence. Legacy
//! const constructors leave metadata absent; matches using `..` remain valid.
//! [`ErrorCode::NoCode`] differs from a known zero code and [`Knowledge::Unknown`]
//! never means zero. A [`PageFaultProjection`] describes only flat-map facts;
//! [`FaultMetadata::architectural_error_code`] never accepts it as a complete #PF.
//! Host protection failures do not establish a guest exception or paging bits.
//! [`RawHostFault`] retains host-labelled signal/context values independently.
//! Embedders copy them in their signal adapter and call
//! [`jit_abi::fault::classify_with_raw_fault`]; compiled landings and transfer
//! fixups carry them to [`FaultMetadata::raw_host_fault`] on vCPU fault returns,
//! including helper, interpreter and fetch faults and instruction stepping.
//! The handler copies only fixed-size records and never allocates.
//! Its host fault address may differ from the guest [`FaultMetadata::operand_base`]
//! and [`FaultMetadata::linear_address`] for split or aligned host accesses.
//! The returned box survives retry, CPU mutation and unmap. It is allocated only
//! on a fault, after signal recovery; successful execution has no metadata reset
//! or allocation. REP exposes exact current restart registers, with unknown
//! per-attempt iteration deltas; split stores may have partial memory effects.
//! [`Vm::set_precise_split_stores`] opts into compiled crossing-store preflight
//! (off by default); revocation after preflight can still leave partial effects.
//!
//! Host yield requests return [`Exit::Interrupted`]. After it returns,
//! [`Vcpu::interruption_evidence`] reports whether engine work deferred the
//! request, with the section kinds and count. The evidence is cleared when
//! [`Vcpu::run`] is next entered; copy it to retain it across resumption.
//!
//! Module map mirrors the spec's dependency order:
//! `state` + `memory` -> `ir` -> `lift` -> `interp` -> `cache`/`vm`.
//!
//! # Shared-address-space children
//!
//! Before forwarding a fork that shares guest RAM, call [`Vm::child_view`] and
//! [`ChildArena::new`] in the parent, and copy the child's [`CpuState`]. Give each
//! child its own CPU, arena, host stack and TLS. In the child call [`child::step`]
//! until exec, exit or a refusal. Integer/SSE/AVX inline IR and string operations
//! reuse the normal interpreter; gathers, x87/state-save, source callbacks, locked
//! memory operations and other unsupported helpers return [`ChildStep::Refused`]. A
//! refusal changes neither CPU nor guest bytes. Stores to snapshotted code pages
//! are refused; other writes enter the bounded arena page log. REP faults retain
//! completed iterations; host split stores can leave a partial memory effect.
//!
//! The child never consults parent memory metadata, caches, code maps, policies
//! or shared engine atomics. It does not allocate or run parent-owned destructors.
//! Keep the inherited view, arena and CPU alive and exit/exec without dropping
//! them in the child. After the child has exited or exec'd, call
//! [`Vm::invalidate_written_pages`] in the parent before releasing the view and
//! backing. This also reconciles watched data pages. Each arena belongs to one
//! child and one VM; never submit its log to a different VM.
//!
//! [`ChildView<'vm>`] owns copied metadata with only a `PhantomData` lifetime
//! borrow of the VM. Safe Rust therefore keeps the VM and its backing alive and
//! prevents unmap/remap while the view survives. An embedder crossing an FFI
//! boundary must preserve that contract externally: keep the VM alive and its
//! span mapped with the same routing, without unmap/remap, until the child has
//! exited or exec'd **and** parent invalidation has completed. Guest bytes remain
//! shared, so synchronize conflicting accesses as for normal VM execution.
//!
//! If host protections can fault inside the otherwise mapped span, install a
//! handler on each child thread calling [`child::classify`]. Copy the interrupted
//! PC and x9/r11 scratch register into [`jit_abi::fault::HostRegisters`], supply
//! [`RawHostFault`], and copy the returned PC back on `true`. Call only for
//! verified synchronous access faults. The classifier uses a static transfer PC
//! range and the arena record carried in scratch; it needs no parent fault handle,
//! global code map or TLS lookup. A `false` result belongs to the prior handler.
//! Snapshot creation may allocate and call the fetch policy once per covered
//! page; stepping and refusal/fault construction are allocation-free. Convert
//! [`ChildFault::into_metadata`] only after returning to the parent.
//!
//! ```
//! use x86jit_core::{child, ChildArena, ChildStep, CpuMode, CpuState,
//!     Prot, RegionKind, Vm, VmConfig};
//! let mut vm = Vm::new(VmConfig::flat(0x4000));
//! vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
//! vm.write_bytes(0x1000, &[0x0f, 0x05]).unwrap();
//! let view = vm.child_view(CpuMode::Long64);
//! let mut arena = ChildArena::new(64);
//! let mut cpu = CpuState::default();
//! cpu.rip = 0x1000;
//! // Child entry, on the embedder's stack and TLS:
//! assert!(matches!(child::step(&view, &mut cpu, &mut arena), ChildStep::Syscall { .. }));
//! // Parent, after the child has exited or exec'd:
//! vm.invalidate_written_pages(&arena);
//! drop(view);
//! ```
//!
//! ```compile_fail
//! use x86jit_core::{ChildArena, CpuMode, Vm, VmConfig};
//! let mut vm = Vm::new(VmConfig::flat(0x4000));
//! let view = vm.child_view(CpuMode::Long64);
//! vm.unmap(0x1000, 0x1000); // the live view prevents changing the backing map
//! drop(view);
//! ```
//!
//! # Example
//!
//! Map a flat address space, drop in a few hand-assembled bytes, and run them on
//! the default interpreter backend:
//!
//! ```
//! use x86jit_core::{Exit, MemConsistency, MemoryModel, Prot, Reg, RegionKind, Vm, VmConfig};
//!
//! let mut vm = Vm::new(VmConfig {
//!     memory_model: MemoryModel::Flat { size: 0x1_0000 },
//!     consistency: MemConsistency::Fast,
//! });
//! vm.map(0, 0x1_0000, Prot::RWX, RegionKind::Ram).unwrap();
//! vm.write_bytes(0x1000, &[0xB8, 0x05, 0x00, 0x00, 0x00, 0xF4]).unwrap(); // mov eax,5 ; hlt
//!
//! let mut cpu = vm.new_vcpu();
//! cpu.set_reg(Reg::Rip, 0x1000);
//! assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
//! assert_eq!(cpu.reg(Reg::Rax) as u32, 5);
//! ```
//!
//! Inject a [`Backend`] (e.g. the `x86jit-cranelift` JIT) via [`Vm::with_backend`]
//! for native-speed execution with identical guest state. See the crate's
//! `examples/` for MMIO devices and the JIT.

pub mod aes;
pub mod cache;
pub mod child;
pub mod codemap;
pub mod disasm;
#[cfg(feature = "environment")]
mod environment;
pub mod exit;
pub mod f80;
pub mod fault;
pub mod features;
pub mod gfni;
mod host_write;
pub mod interp;
pub mod interruption;
pub mod ir;
pub mod jit_abi;
pub mod lift;
pub mod lockstep;
pub mod memory;
mod memory_stats;
pub mod pclmul;
pub mod sha;
pub mod state;
#[doc(hidden)]
pub mod time_stats;
mod usercopy;
pub mod vm;
pub mod x87;

pub use cache::{BlockKey, CachedBlock, CompiledPtr, TranslationCache};
pub use child::{ChildArena, ChildFault, ChildRefusal, ChildStep, ChildView};
pub use codemap::{GuestAccessKind, GuestTrapSite};
pub use disasm::{disassemble, print_disassembly, DecodedInsn};
pub use exit::{AccessKind, ExecutionEvidence, Exit, PortDir, StepResult};
pub use fault::{
    AddressPrecision, AddressSpace, ErrorCode, FaultAccess, FaultAddress, FaultCause, FaultClass,
    FaultHelper, FaultMetadata, FaultOrigin, FaultPath, FaultProgress, HelperCaller,
    HostFaultPlatform, Knowledge, MemoryEffects, PageFaultProjection, RawHostFault,
    RegisterProgress, RepRestart, TransferUnit,
};
pub use features::{
    CpuidLeaf, CpuidOutOfRange, CpuidOverride, CpuidOverrides, CpuidRegisters, CpuidTable, Feature,
    GuestCpuFeatures,
};
pub use interruption::{CriticalSectionKind, InterruptionEvidence};
pub use ir::{
    AesOp, BitScanOp, BtOp, Cond, FPrec, FenceKind, FlagMask, FloatBinOp, FloatUnOp, GfniOp,
    HFloatOp, HIntOp, IrBlock, IrOp, IrRegion, MemOrder, PackedBinOp, PackedCvtKind, RegionCaps,
    RepKind, RmwOp, ShaOp, StrOp, Temp, TempGen, VKLogicOp, VLogicOp, Val, VpUnaryOp,
};
pub use lift::CpuMode;
pub use memory::{
    FetchPermit, FetchVerdict, HostRam, HostRamError, MapError, MemError, MemTrap, Memory,
    MemoryModel, Prot, RegionKind,
};
pub use memory_stats::{BackendMemoryStats, CacheMemoryStats, MemoryStats, VcpuMemoryStats};
pub use state::{
    CpuSource, CpuState, Flags, RandomKind, Reg, TscSample, X87PointerModel, X87Precision,
    AF_SRC_ONE, AF_SRC_ZERO, PF_SRC_ONE, PF_SRC_ZERO,
};
pub use time_stats::{
    BackendTimeStats, ExitTimeStats, ExitsTimeStats, SampledTimeStats, TimeStats,
};
pub use vm::{
    Backend, CompileFailureReason, ExitRequest, ForceRegionError, InterpreterBackend,
    MemConsistency, ShutdownError, ShutdownReport, SourceIdError, StepOptions, StepRun,
    StepRunError, StepStop, TierUpFinished, TierUpRequest, TierUpSubmit, TierUpUnit, Vcpu, Vm,
    VmConfig,
};
pub use x87::FpuKind;
