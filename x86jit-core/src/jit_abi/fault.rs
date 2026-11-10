//! Signal-context recovery for inline compiled guest accesses (§8.2).
//!
//! The embedder owns signal installation and its per-thread handle slot. The
//! classifier reads immutable transfer fixups, codemap metadata and run records.
//! Compiled calls remain direct calls, with no recovery work on chain edges.

use super::{CpuState, MemCtx};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};

#[path = "landing.rs"]
mod landing;
#[cfg(all(
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
use landing::landing_stub;

/// Embedder classification of the synchronous signal's reason. User-sent and
/// unsupported signals must use `Unsupported`, regardless of interrupted PC.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HostFaultKind {
    #[default]
    Unsupported,
    Access,
    Alignment,
}

/// Registers copied from/to the interrupted signal context. `callee_saved` is
/// [rbx, r12, r13, r14, r15] on x86-64, x19..x28 on AArch64. `simd` contains
/// the low 64 bits of v8..v15 on AArch64. `scratch` is r11 / x9. Copy back every
/// field on OURS, including LR, scratch and SIMD; unused fields are ignored.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostRegisters {
    pub pc: u64,
    pub sp: u64,
    pub fp: u64,
    pub lr: u64,
    pub callee_saved: [u64; 10],
    pub simd: [u64; 8],
    pub scratch: u64,
}

/// Immutable one-frame recipe, extracted from the compiler's unwind description.
/// All offsets are signed byte distances from the interrupted frame pointer.
/// A missing register slot means that register retains its interrupted value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FrameRecipe {
    pub prologue_end: u32,
    /// AArch64 return-address authentication: 0 = unsigned, 1 = A key, 2 = B key.
    pub return_auth: u8,
    pub caller_sp: i32,
    pub saved_fp: i32,
    pub saved_return: i32,
    pub gpr: [Option<i32>; 10],
    pub simd: [Option<i32>; 8],
}

/// Stable, non-owning handle. Moving a vCPU preserves it; dropping it invalidates
/// it. A dangling handle must never be passed to `classify`.
#[derive(Clone, Copy)]
pub struct FaultHandle(usize);

impl FaultHandle {
    /// Integer for an embedder-owned per-thread signal-handler slot.
    pub fn as_usize(self) -> usize {
        self.0
    }
}

#[repr(C)]
#[derive(Default)]
struct Snapshot {
    // First field: landing assembly loads it through the scratch register.
    fp_control: u64,
    cpu: *mut CpuState,
    ctx: *mut MemCtx,
    kind: HostFaultKind,
    raw_fault: Option<crate::RawHostFault>,
}

// Independent of compiled activation: fetches and Rust helpers can fault while
// cpu/ctx are unpublished. The outer cold exit path consumes this once.
#[derive(Default)]
struct TransferFault {
    pending: bool,
    host_address: u64,
    raw: Option<crate::RawHostFault>,
}

/// Per-run storage, boxed by the vCPU. Standalone compiled callers may use a
/// fixed-address record and explicitly activate/deactivate it around execution.
#[derive(Default)]
pub struct EntryRecord {
    active: AtomicBool,
    snapshot: UnsafeCell<Snapshot>,
    transfer: UnsafeCell<TransferFault>,
}

// SAFETY: execution and synchronous signal access are confined to the owning
// thread; sharing a handle does not authorize another thread to classify with it.
unsafe impl Send for EntryRecord {}
// SAFETY: mutation requires exclusive access, and classification is restricted
// to the thread executing that record's run.
unsafe impl Sync for EntryRecord {}

impl EntryRecord {
    pub(crate) fn prepare_run(&mut self) {
        let saved = self.snapshot.get_mut();
        #[cfg(target_arch = "aarch64")]
        // SAFETY: reading the current thread's host FP control has no memory effects.
        unsafe {
            core::arch::asm!("mrs {}, fpcr", out(reg) saved.fp_control, options(nomem, nostack));
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: STMXCSR writes four bytes of this exclusive snapshot
        // (Intel SDM Vol 1 §10.2.3).
        unsafe {
            core::arch::asm!("stmxcsr [{}]", in(reg) &mut saved.fp_control, options(nostack));
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = saved;
    }

    /// Capture host FP control and publish a standalone run's live ABI pointers.
    ///
    /// # Safety
    /// Keep this record, `cpu`, and `ctx` at fixed addresses until `deactivate`.
    /// They must belong exclusively to this thread, and compiled code must obey
    /// the ABI and precise guest trap-state contract. Do not nest activation.
    pub unsafe fn activate(&mut self, cpu: *mut CpuState, ctx: *mut MemCtx) {
        self.prepare_run();
        self.publish(cpu, ctx);
    }

    pub(crate) fn publish(&mut self, cpu: *mut CpuState, ctx: *mut MemCtx) {
        let saved = self.snapshot.get_mut();
        saved.cpu = cpu;
        saved.ctx = ctx;
        self.active.store(true, Ordering::Release);
    }

    /// End a standalone run before its ABI pointers or record become invalid.
    pub fn deactivate(&mut self) {
        self.active.store(false, Ordering::Release);
        let saved = self.snapshot.get_mut();
        saved.cpu = std::ptr::null_mut();
        saved.ctx = std::ptr::null_mut();
        saved.kind = HostFaultKind::Unsupported;
    }

    pub(crate) fn fault_kind(&self) -> HostFaultKind {
        // SAFETY: read by the owning dispatcher only after the compiled return.
        unsafe { (*self.snapshot.get()).kind }
    }

    /// Evidence from the latest compiled landing. Read only after a
    /// `RET_HOST_FAULT_DETAIL` return, before another compiled call. Legacy
    /// `classify` callers leave it absent. No allocation occurs here.
    pub fn raw_fault(&mut self) -> Option<crate::RawHostFault> {
        self.snapshot.get_mut().raw_fault
    }

    /// Consume only on the cold fault-return route, including interpreted steps
    /// and decode faults. Successful transfers never touch the validity tag.
    #[cold]
    #[inline(never)]
    pub(crate) fn finish_transfer_fault(
        &mut self,
        mut exit: crate::Exit,
        mem: &crate::Memory,
    ) -> crate::Exit {
        let transfer = std::mem::take(self.transfer.get_mut());
        if transfer.pending {
            if let crate::Exit::UnmappedMemory {
                metadata: Some(m), ..
            } = &mut exit
            {
                let host_address = match m.linear_address() {
                    crate::Knowledge::Known(address) => address
                        .value()
                        .checked_sub(mem.guest_base())
                        .and_then(|offset| (mem.host_base() as u64).checked_add(offset)),
                    _ => None,
                };
                if m.origin() == crate::FaultOrigin::HostAccess
                    && host_address == Some(transfer.host_address)
                    // A compiled landing can carry newer evidence than an accepted probe.
                    && m.raw_host_fault().is_none()
                {
                    m.raw_host_fault = transfer.raw;
                }
            }
        }
        exit
    }

    pub fn handle(&self) -> FaultHandle {
        FaultHandle(self as *const Self as usize)
    }
}

/// Classify a synchronous memory fault. On OURS (`true`), copy all host register
/// fields back into the ucontext and return from the handler. On NOT OURS
/// (`false`), state and registers remain untouched; chain the prior disposition.
///
/// Access faults inside a transfer fixup range recover with no active run:
/// only PC and scratch change, and the routine returns normally to Rust. An
/// Alignment fault in such a routine is NOT OURS. Other PCs require an active
/// run, an exact guest trap site with a supported unwind recipe, an outermost
/// compiled frame, and a data address inside the run's guest span. FreeBSD
/// AArch64 Alignment recovery ignores `data_addr` and replays from guest RIP:
/// that signal's `si_addr` names the host PC. Unsupported
/// hosts answer NOT OURS. There is no allocation, lock, TLS or libc call: a fixed
/// table lookup, a code-map reader-count increment and decrement (atomic RMWs,
/// which may retry under contention where the host lowers them to LL/SC loops),
/// bounded AVL lookups, exact-site binary search and fixed-size stack reads. The landing stub restores host
/// FP control before control returns to call_block's caller (MXCSR: Intel SDM
/// Vol 1 §10.2.3; x86-64 psABI §3.2.1). The signal handler and classifier must
/// not depend on the thread's interrupted FP mode.
///
/// # Safety
/// `handle` is zero or a live handle for this thread's currently executing vCPU
/// (or an inactive live record). Never use another thread's active handle. Keep
/// the record and CPU alive and exclusively borrowed throughout a compiled run;
/// keep the shared guest mapping live throughout every transfer. Fixup recovery
/// itself needs no record or CPU pointer.
/// Supply the actual interrupted PC, SP, FP, preserved registers and `si_addr`.
/// Derive `kind` from a verified OS fault cause: Linux/FreeBSD signal/code or
/// macOS AArch64's saved data-abort ESR (decision-13). Ambiguous and user-sent
/// signals are Unsupported.
/// Alignment recovery replays the guest instruction; its locked-operation
/// atomicity limitation is tracked in xj-0w3.5.
/// Do not fabricate frames or handles, reenter execution, or modify guest state
/// in the handler. Compiled accesses must preserve precise instruction-entry
/// state. The stack and compiler-emitted frame save slots must remain readable.
pub unsafe fn classify(
    handle: usize,
    host_pc: usize,
    data_addr: usize,
    kind: HostFaultKind,
    regs: &mut HostRegisters,
) -> bool {
    classify_inner(handle, host_pc, data_addr, kind, regs, None)
}

/// Like [`classify`], additionally retaining an opaque raw signal record on a
/// compiled landing or transfer fixup. Copy the context fields before calling:
/// recovery rewrites the interrupted registers. Transfer evidence is attached
/// when the vCPU returns the helper, interpreter or fetch fault.
///
/// # Safety
/// All [`classify`]'s requirements apply. `raw` must describe this signal's
/// actual host context, with unavailable fields left `None`.
pub unsafe fn classify_with_raw_fault(
    handle: usize,
    host_pc: usize,
    data_addr: usize,
    kind: HostFaultKind,
    regs: &mut HostRegisters,
    raw: crate::RawHostFault,
) -> bool {
    classify_inner(handle, host_pc, data_addr, kind, regs, Some(raw))
}

unsafe fn classify_inner(
    handle: usize,
    host_pc: usize,
    data_addr: usize,
    kind: HostFaultKind,
    regs: &mut HostRegisters,
    raw: Option<crate::RawHostFault>,
) -> bool {
    #[cfg(all(
        any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    {
        if kind == HostFaultKind::Unsupported || regs.pc != host_pc as u64 {
            return false;
        }
        if let Some(fixup) = crate::usercopy::lookup(host_pc) {
            if kind != HostFaultKind::Access {
                return false;
            }
            if handle != 0 {
                // SAFETY: even an inactive handle names a live record on this
                // thread. No CPU/context pointer is read on the fixup route.
                let record = &*(handle as *const EntryRecord);
                *record.transfer.get() = TransferFault {
                    pending: true,
                    host_address: data_addr as u64,
                    raw,
                };
            }
            regs.pc = fixup as u64;
            regs.scratch = data_addr as u64;
            return true;
        }
        if handle == 0 {
            return false;
        }
        // SAFETY: the live record belongs to this thread under the caller's contract.
        let record = &*(handle as *const EntryRecord);
        if !record.active.load(Ordering::Acquire) {
            return false;
        }
        let Some((site, recipe)) = crate::codemap::lookup_recovery(host_pc) else {
            return false;
        };
        let saved = &mut *record.snapshot.get();
        let ctx = &*saved.ctx;
        // FreeBSD 14.3 sys/arm64/arm64/trap.c, align_abort/call_trapsignal:
        // alignment si_addr is tf_elr, not FAR (source unpinned in oracles/).
        // Replay recomputes the operand; every other cause/host needs its address.
        let addr = if cfg!(all(target_os = "freebsd", target_arch = "aarch64"))
            && kind == HostFaultKind::Alignment
        {
            None
        } else {
            let Some(offset) = (data_addr as u64).checked_sub(ctx.base) else {
                return false;
            };
            let Some(addr) = ctx.guest_base.checked_add(offset).filter(|a| *a < ctx.size) else {
                return false;
            };
            Some(addr)
        };
        let Some(caller_sp) = regs.fp.checked_add_signed(i64::from(recipe.caller_sp)) else {
            return false;
        };
        if regs.fp == 0 || regs.fp & 7 != 0 || regs.sp > regs.fp {
            return false;
        }
        // Validate every slot before reading or changing any state. A supported
        // recipe names only eight-byte slots inside this compiled frame.
        let slot = |off: i32| {
            regs.fp.checked_add_signed(i64::from(off)).filter(|p| {
                *p >= regs.sp && p.checked_add(8).is_some_and(|end| end <= caller_sp) && p & 7 == 0
            })
        };
        let Some(fp_slot) = slot(recipe.saved_fp) else {
            return false;
        };
        let Some(ra_slot) = slot(recipe.saved_return) else {
            return false;
        };
        for off in recipe.gpr.iter().chain(&recipe.simd).flatten() {
            if slot(*off).is_none() {
                return false;
            }
        }
        // SAFETY: these compiler-described frame slots have been validated above;
        // actual signal context and a readable compiled stack are caller obligations.
        #[allow(unused_mut)]
        let mut return_address = (ra_slot as *const u64).read();
        #[cfg(target_arch = "aarch64")]
        match recipe.return_auth {
            // Arm ARM DDI0487J.a §§C6.2.22–C6.2.23: authenticate using the signing
            // modifier. Cranelift's ASP/BSP prologues use the caller's SP.
            1 => core::arch::asm!(".arch_extension pauth", "autia {ra}, {modifier}",
                ra = inout(reg) return_address, modifier = in(reg) caller_sp,
                options(nomem, nostack)),
            2 => core::arch::asm!(".arch_extension pauth", "autib {ra}, {modifier}",
                ra = inout(reg) return_address, modifier = in(reg) caller_sp,
                options(nomem, nostack)),
            0 => {}
            _ => return false,
        }
        #[cfg(not(target_arch = "aarch64"))]
        if recipe.return_auth != 0 {
            return false;
        }
        if crate::codemap::contains(return_address as usize) {
            return false;
        }
        let mut recovered = *regs;
        recovered.fp = (fp_slot as *const u64).read();
        for (value, off) in recovered.callee_saved.iter_mut().zip(recipe.gpr) {
            if let Some(off) = off {
                *value = (slot(off).unwrap() as *const u64).read();
            }
        }
        for (value, off) in recovered.simd.iter_mut().zip(recipe.simd) {
            if let Some(off) = off {
                *value = (slot(off).unwrap() as *const u64).read();
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            recovered.sp = caller_sp;
            recovered.lr = return_address;
        }
        #[cfg(target_arch = "x86_64")]
        {
            recovered.sp = ra_slot;
        }
        recovered.scratch = saved as *const Snapshot as u64;
        recovered.pc = landing_stub as *const () as u64;
        saved.kind = kind;
        saved.raw_fault = raw;
        (*saved.cpu).rip = site.guest_rip;
        if let Some(addr) = addr {
            (*saved.ctx).fault_addr = addr;
        }
        (*saved.ctx).next_entry = host_pc as u64;
        (*saved.ctx).link_slot = (match site.access {
            crate::GuestAccessKind::Read => 0,
            crate::GuestAccessKind::Write => 1,
            crate::GuestAccessKind::ReadWrite => 2,
        }) | if kind == HostFaultKind::Alignment {
            0x100
        } else {
            0
        };
        (*saved.ctx).fault_size = u64::from(site.width);
        (*saved.ctx).fault_access = u64::from(site.access != crate::GuestAccessKind::Read);
        *regs = recovered;
        true
    }
    #[cfg(not(all(
        any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
        any(target_arch = "x86_64", target_arch = "aarch64")
    )))]
    {
        let _ = (handle, host_pc, data_addr, kind, regs, raw);
        false
    }
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod tests {
    use super::*;

    #[test]
    fn transfer_evidence_is_consumed_once_and_requires_matching_host_address() {
        let vm = crate::Vm::new(crate::VmConfig::flat(0x4000));
        let mut record = EntryRecord::default();
        let raw = crate::RawHostFault::new(crate::HostFaultPlatform::LinuxX86_64);
        let fault =
            || crate::fault::memory_exit(0x1000, 0x2000, 8, crate::AccessKind::Read, true, true);
        *record.transfer.get_mut() = TransferFault {
            pending: true,
            host_address: vm.mem.host_base() as u64 + 0x2000,
            raw: Some(raw),
        };
        let owned = record.finish_transfer_fault(fault(), &vm.mem);
        assert_eq!(owned.fault_metadata().unwrap().raw_host_fault(), Some(&raw));
        assert_eq!(
            record
                .finish_transfer_fault(fault(), &vm.mem)
                .fault_metadata()
                .unwrap()
                .raw_host_fault(),
            None
        );
        *record.transfer.get_mut() = TransferFault {
            pending: true,
            host_address: vm.mem.host_base() as u64 + 0x2001,
            raw: Some(raw),
        };
        assert_eq!(
            record
                .finish_transfer_fault(fault(), &vm.mem)
                .fault_metadata()
                .unwrap()
                .raw_host_fault(),
            None
        );
        assert!(!record.transfer.get_mut().pending);
        assert_eq!(owned.fault_metadata().unwrap().raw_host_fault(), Some(&raw));
    }

    #[test]
    fn alignment_with_pc_as_address_replays_only_on_freebsd_arm64() {
        static CODE: [u8; 16] = [0; 16];
        static NO_RECIPE: [u8; 16] = [0; 16];
        let pc = CODE.as_ptr() as usize;
        let site = crate::GuestTrapSite {
            host_offset: 4,
            guest_rip: 0x1000,
            access: crate::GuestAccessKind::Read,
            width: 4,
        };
        crate::codemap::register_with_recovery(
            pc,
            16,
            Box::new([]),
            Box::new([]),
            vec![site],
            Some(FrameRecipe {
                prologue_end: 4,
                return_auth: 0,
                caller_sp: 16,
                saved_fp: 0,
                saved_return: 8,
                gpr: [None; 10],
                simd: [None; 8],
            }),
        )
        .unwrap();
        let missing = NO_RECIPE.as_ptr() as usize;
        crate::codemap::register_with_traps(missing, 16, Box::new([]), Box::new([]), vec![site])
            .unwrap();
        let mut vm = crate::Vm::new(crate::VmConfig::flat(0x4000));
        vm.map(0x1000, 0x2000, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        // MOV EAX, [RBX] (SDM Vol 2B, MOV, Operation): replay must recompute RBX.
        vm.write_bytes(0x1000, &[0x8b, 0x03]).unwrap();
        vm.write_bytes(0x2000, &0x1234_5678u32.to_le_bytes())
            .unwrap();
        let mut cpu = CpuState::new();
        cpu.gpr[crate::Reg::Rbx.gpr_index().unwrap()] = 0x2000;
        let mut ctx = MemCtx::for_memory(&vm.mem);
        ctx.fault_addr = u64::MAX;
        assert!(!((ctx.base)..(ctx.base + ctx.size)).contains(&((pc + 4) as u64)));
        let mut frame = [0u64, landing_stub as *const () as u64];
        let original = HostRegisters {
            pc: (pc + 4) as u64,
            sp: frame.as_ptr() as u64,
            fp: frame.as_ptr() as u64,
            ..Default::default()
        };
        let mut record = EntryRecord::default();
        let handle = record.handle().as_usize();
        // SAFETY: all run pointers and synthetic frame slots are live in this thread.
        unsafe {
            record.activate(&mut cpu, &mut ctx);
            for (candidate, live_handle, kind) in [
                (pc + 3, handle, HostFaultKind::Alignment),
                (missing + 4, handle, HostFaultKind::Alignment),
                (pc + 4, 0, HostFaultKind::Alignment),
                (pc + 4, handle, HostFaultKind::Access),
                (pc + 4, handle, HostFaultKind::Unsupported),
            ] {
                let mut regs = HostRegisters {
                    pc: candidate as u64,
                    ..original
                };
                let before = regs;
                assert!(!classify(
                    live_handle,
                    candidate,
                    candidate,
                    kind,
                    &mut regs
                ));
                assert_eq!(regs, before);
            }
            frame[1] = (pc + 8) as u64;
            let mut regs = original;
            assert!(!classify(
                handle,
                pc + 4,
                pc + 4,
                HostFaultKind::Alignment,
                &mut regs
            ));
            assert_eq!(regs, original);
            frame[1] = landing_stub as *const () as u64;
            std::hint::black_box(&frame);
            let recovered = classify(handle, pc + 4, pc + 4, HostFaultKind::Alignment, &mut regs);
            assert_eq!(
                recovered,
                cfg!(all(target_os = "freebsd", target_arch = "aarch64"))
            );
            // No fabricated data address is recorded, even for an accepted signal.
            assert_eq!(ctx.fault_addr, u64::MAX);
            if recovered {
                assert_eq!(record.fault_kind(), HostFaultKind::Alignment);
                assert_eq!(regs.pc, landing_stub as *const () as u64);
                assert_eq!(cpu.rip, site.guest_rip);
                // The dispatcher's Alignment path single-steps from this guest RIP.
                let result = crate::interp::step_one(
                    &vm.mem,
                    &mut cpu,
                    crate::CpuMode::Long64,
                    &mut Vec::new(),
                    &mut Default::default(),
                );
                assert!(matches!(result, crate::StepResult::Continue));
                assert_eq!(cpu.gpr[crate::Reg::Rax.gpr_index().unwrap()], 0x1234_5678);
                assert_eq!(cpu.rip, 0x1002);
            } else {
                assert_eq!(regs, original);
                assert_eq!(cpu.rip, 0);
            }
            record.deactivate();
            let mut regs = original;
            assert!(!classify(
                handle,
                pc + 4,
                pc + 4,
                HostFaultKind::Alignment,
                &mut regs
            ));
            assert_eq!(regs, original);
        }
    }

    #[test]
    fn classifier_requires_recipe_exact_site_outer_frame_and_live_run() {
        static CODE: [u8; 16] = [0; 16];
        static NO_RECIPE: [u8; 16] = [0; 16];
        static INNER_CALLER: [u8; 16] = [0; 16];
        let pc = CODE.as_ptr() as usize;
        let missing = NO_RECIPE.as_ptr() as usize;
        let site = crate::GuestTrapSite {
            host_offset: 4,
            guest_rip: 0x1_1234_5678,
            access: crate::GuestAccessKind::ReadWrite,
            width: 16,
        };
        // Synthetic readable frame: eight saved SIMD words, ten GPR words,
        // then caller FP and RA. All save locations differ from live registers.
        let mut frame: [u64; 20] = std::array::from_fn(|i| 0x5500 + i as u64);
        frame[19] = 0x3300;
        let recipe = FrameRecipe {
            prologue_end: 4,
            return_auth: 0,
            caller_sp: 16,
            saved_fp: 0,
            saved_return: 8,
            gpr: std::array::from_fn(|i| Some(-80 + i as i32 * 8)),
            simd: std::array::from_fn(|i| Some(-144 + i as i32 * 8)),
        };
        crate::codemap::register_with_recovery(
            pc,
            16,
            Box::new([]),
            Box::new([]),
            vec![
                crate::GuestTrapSite {
                    host_offset: 0,
                    ..site
                },
                site,
            ],
            Some(recipe),
        )
        .unwrap();
        crate::codemap::register_with_traps(missing, 16, Box::new([]), Box::new([]), vec![site])
            .unwrap();
        let inner = INNER_CALLER.as_ptr() as usize;
        crate::codemap::register(inner, 16, Box::new([]), Box::new([])).unwrap();
        let mem = crate::Memory::new(crate::MemoryModel::Flat { size: 64 });
        let mut cpu = CpuState::new();
        let mut ctx = MemCtx::for_memory(&mem);
        ctx.guest_base = 0x1_0000_0000;
        ctx.size += ctx.guest_base;
        let mut record = EntryRecord::default();
        let fp = frame.as_ptr().wrapping_add(18) as u64;
        let original = HostRegisters {
            pc: (pc + 4) as u64,
            sp: frame.as_ptr() as u64,
            fp,
            callee_saved: [0xdead; 10],
            simd: [0xbeef; 8],
            ..Default::default()
        };
        let addr = ctx.base as usize + 4;
        let handle = record.handle().as_usize();
        // SAFETY: the synthetic frame and exclusive run's pointers stay live;
        // no signal or concurrent execution accesses them in this test.
        unsafe {
            let mut regs = original;
            assert!(!classify(
                handle,
                pc + 4,
                addr,
                HostFaultKind::Access,
                &mut regs
            ));
            record.activate(&mut cpu, &mut ctx);
            assert!(!classify(
                handle,
                pc + 4,
                addr,
                HostFaultKind::Unsupported,
                &mut regs
            ));
            assert_eq!(regs, original);
            for rejected in [
                pc,
                pc + 3,
                pc + 5,
                missing + 4,
                0,
                classify as *const () as usize,
                landing_stub as *const () as usize,
            ] {
                let mut regs = HostRegisters {
                    pc: rejected as u64,
                    ..original
                };
                let before = regs;
                assert!(!classify(
                    handle,
                    rejected,
                    addr,
                    HostFaultKind::Access,
                    &mut regs
                ));
                assert_eq!(regs, before);
            }
            for bad_addr in [ctx.base as usize - 1, ctx.base as usize + 64] {
                assert!(!classify(
                    handle,
                    pc + 4,
                    bad_addr,
                    HostFaultKind::Access,
                    &mut regs
                ));
                assert_eq!(regs, original);
            }
            frame[19] = inner as u64 + 1;
            assert!(!classify(
                handle,
                pc + 4,
                addr,
                HostFaultKind::Access,
                &mut regs
            ));
            assert_eq!(regs, original);
            assert_eq!(cpu.rip, 0);
            assert_eq!(ctx.fault_size, 0);
            frame[19] = 0x3300;
            let raw = crate::RawHostFault {
                signal: Some(11),
                fault_address: Some(addr as u64),
                ..crate::RawHostFault::new(crate::HostFaultPlatform::LinuxX86_64)
            };
            assert!(classify_with_raw_fault(
                handle,
                pc + 4,
                addr,
                HostFaultKind::Access,
                &mut regs,
                raw
            ));
            assert_eq!(record.raw_fault(), Some(raw));
            assert_eq!(regs.fp, frame[18]);
            assert_eq!(regs.callee_saved, frame[8..18]);
            assert_eq!(regs.simd, frame[..8]);
            #[cfg(target_arch = "aarch64")]
            {
                assert_eq!(regs.sp, fp + 16);
                assert_eq!(regs.lr, 0x3300);
            }
            #[cfg(target_arch = "x86_64")]
            assert_eq!(regs.sp, fp + 8);
            assert_eq!(regs.pc, landing_stub as *const () as u64);
            let snapshot = &*record.snapshot.get();
            assert_eq!(regs.scratch, snapshot as *const Snapshot as u64);
            assert_eq!(std::ptr::addr_of!(snapshot.fp_control) as u64, regs.scratch);
            assert_eq!(cpu.rip, 0x1_1234_5678);
            assert_eq!(ctx.fault_addr, 0x1_0000_0004);
            assert_eq!(ctx.fault_size, 16);
            assert_eq!(ctx.fault_access, 1);
            let mut legacy = original;
            assert!(classify(
                handle,
                pc + 4,
                addr,
                HostFaultKind::Access,
                &mut legacy
            ));
            assert_eq!(
                record.raw_fault(),
                None,
                "legacy classification must not reuse evidence"
            );
            record.deactivate();
            let mut regs = original;
            assert!(!classify(
                handle,
                pc + 4,
                addr,
                HostFaultKind::Access,
                &mut regs
            ));
            assert_eq!(regs, original);
        }
    }
    #[test]
    fn panicking_run_deactivates_and_clears_its_record() {
        struct Panics;
        impl crate::Backend for Panics {
            fn materialize(
                &self,
                _: &crate::IrBlock,
                _: crate::MemConsistency,
                _: Option<(u64, u64)>,
                _: u64,
            ) -> crate::CachedBlock {
                panic!("forced compilation panic");
            }
        }
        let mut vm = crate::Vm::with_backend(crate::VmConfig::flat(0x10000), Box::new(Panics));
        vm.map(0x1000, 0x1000, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        vm.write_bytes(0x1000, &[0xf4]).unwrap();
        let mut vcpu = vm.new_vcpu();
        vcpu.set_reg(crate::Reg::Rip, 0x1000);
        let handle = vcpu.fault_handle().as_usize();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| vcpu.run(&vm, None))).is_err()
        );
        // SAFETY: the boxed record belongs to the still-live vCPU in this thread.
        unsafe {
            let record = &*(handle as *const EntryRecord);
            assert!(!record.active.load(Ordering::Acquire));
            let snapshot = &*record.snapshot.get();
            assert!(snapshot.cpu.is_null());
            assert!(snapshot.ctx.is_null());
            assert!(!classify(
                handle,
                0,
                0,
                HostFaultKind::Access,
                &mut HostRegisters::default()
            ));
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn authentication_is_verified_only_when_pac_is_active() {
        if !std::arch::is_aarch64_feature_detected!("paca") {
            eprintln!("PAuth is unavailable; runtime authentication unverified");
            return;
        }
        let plain = landing_stub as *const () as u64;
        let mut frame = [0x1234u64, plain];
        let fp = frame.as_ptr() as u64;
        let caller_sp = fp + 16;
        let mut signed = plain;
        // Arm ARM DDI0487J.a §C6.2 (PACIB), §C6.2.23 (AUTIB): identical key and modifier
        // authenticate a signed pointer; SP-vs-zero encoding is tested separately.
        // SAFETY: feature check precedes these register-only instructions.
        unsafe {
            core::arch::asm!(".arch_extension pauth", "pacib {p}, {m}",
                p = inout(reg) signed, m = in(reg) caller_sp, options(nomem, nostack));
        }
        if signed == plain {
            eprintln!(
                "PAC instructions are identity in this process; runtime authentication unverified"
            );
            return;
        }
        frame[1] = signed;
        static CODE: [u8; 8] = [0; 8];
        let pc = CODE.as_ptr() as usize;
        crate::codemap::register_with_recovery(
            pc,
            8,
            Box::new([]),
            Box::new([]),
            vec![crate::GuestTrapSite {
                host_offset: 4,
                guest_rip: 0x1000,
                access: crate::GuestAccessKind::Read,
                width: 8,
            }],
            Some(FrameRecipe {
                prologue_end: 4,
                return_auth: 2,
                caller_sp: 16,
                saved_fp: 0,
                saved_return: 8,
                gpr: [None; 10],
                simd: [None; 8],
            }),
        )
        .unwrap();
        let mem = crate::Memory::new(crate::MemoryModel::Flat { size: 64 });
        let mut cpu = CpuState::new();
        let mut ctx = MemCtx::for_memory(&mem);
        let mut record = EntryRecord::default();
        let mut regs = HostRegisters {
            pc: (pc + 4) as u64,
            sp: fp,
            fp,
            ..Default::default()
        };
        // SAFETY: the synthetic readable frame contains a genuinely signed return
        // address with the default ABI's SP modifier; all run pointers stay live.
        unsafe {
            record.activate(&mut cpu, &mut ctx);
            assert!(classify(
                record.handle().as_usize(),
                pc + 4,
                ctx.base as usize,
                HostFaultKind::Access,
                &mut regs
            ));
            record.deactivate();
        }
        assert_eq!(regs.lr, plain);
        assert_eq!(regs.sp, caller_sp);
        assert_eq!(regs.fp, frame[0]);
    }

    #[test]
    fn landing_symbols_link_in_two_crate_instances() {
        use std::process::Command;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("x86jit-landing-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let fixture = format!("pub mod jit_abi {{ pub const RET_HOST_FAULT_DETAIL: u64 = 15; pub mod fault {{ mod landing {{ {} }} pub fn address() -> usize {{ std::hint::black_box(landing::landing_stub as *const () as usize) }} }} }} pub fn address() -> usize {{ jit_abi::fault::address() }}", include_str!("landing.rs"));
        let source = dir.join("lib.rs");
        std::fs::write(&source, fixture).unwrap();
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        for version in ["one", "two"] {
            let result = Command::new(&rustc)
                .args([
                    "--edition=2021",
                    "--crate-name",
                    "landing_fixture",
                    "--crate-type=rlib",
                    "-C",
                ])
                .arg(format!("metadata={version}"))
                .arg(&source)
                .arg("-o")
                .arg(dir.join(format!("lib{version}.rlib")))
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
        let main = dir.join("main.rs");
        std::fs::write(
            &main,
            "fn main() { assert_ne!(one::address(), two::address()); }",
        )
        .unwrap();
        let binary = dir.join("probe");
        let result = Command::new(&rustc)
            .arg("--edition=2021")
            .arg(&main)
            .arg("--extern")
            .arg(format!("one={}", dir.join("libone.rlib").display()))
            .arg("--extern")
            .arg(format!("two={}", dir.join("libtwo.rlib").display()))
            .arg("-o")
            .arg(&binary)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(Command::new(&binary).status().unwrap().success());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
