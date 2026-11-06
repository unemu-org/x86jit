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
//!
//! Module map mirrors the spec's dependency order:
//! `state` + `memory` -> `ir` -> `lift` -> `interp` -> `cache`/`vm`.
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
pub mod codemap;
pub mod disasm;
#[cfg(feature = "environment")]
mod environment;
pub mod exit;
pub mod f80;
pub mod fault;
pub mod features;
pub mod gfni;
pub mod interp;
pub mod ir;
pub mod jit_abi;
pub mod lift;
pub mod lockstep;
pub mod memory;
pub mod pclmul;
pub mod sha;
pub mod state;
mod usercopy;
pub mod vm;
pub mod x87;

pub use cache::{BlockKey, CachedBlock, CompiledPtr, TranslationCache};
pub use codemap::{GuestAccessKind, GuestTrapSite};
pub use disasm::{disassemble, print_disassembly, DecodedInsn};
pub use exit::{AccessKind, Exit, PortDir, StepResult};
pub use fault::{
    AddressPrecision, AddressSpace, ErrorCode, FaultAccess, FaultAddress, FaultCause, FaultClass,
    FaultMetadata, FaultOrigin, FaultProgress, HostFaultPlatform, Knowledge, MemoryEffects,
    PageFaultProjection, RawHostFault, RegisterProgress, RepRestart, TransferUnit,
};
pub use features::{
    CpuidLeaf, CpuidOutOfRange, CpuidOverride, CpuidOverrides, CpuidRegisters, CpuidTable, Feature,
    GuestCpuFeatures,
};
pub use ir::{
    AesOp, BitScanOp, BtOp, Cond, FPrec, FenceKind, FlagMask, FloatBinOp, FloatUnOp, GfniOp,
    HFloatOp, HIntOp, IrBlock, IrOp, IrRegion, MemOrder, PackedBinOp, PackedCvtKind, RegionCaps,
    RepKind, RmwOp, ShaOp, StrOp, Temp, TempGen, VKLogicOp, VLogicOp, Val, VpUnaryOp,
};
pub use lift::CpuMode;
pub use memory::{
    HostRam, HostRamError, MapError, MemError, MemTrap, Memory, MemoryModel, Prot, RegionKind,
};
pub use state::{
    CpuSource, CpuState, Flags, RandomKind, Reg, TscSample, X87Precision, AF_SRC_ONE, AF_SRC_ZERO,
    PF_SRC_ONE, PF_SRC_ZERO,
};
pub use vm::{
    Backend, ExitRequest, InterpreterBackend, MemConsistency, TierUpFinished, TierUpRequest,
    TierUpSubmit, TierUpUnit, Vcpu, Vm, VmConfig,
};
pub use x87::FpuKind;
