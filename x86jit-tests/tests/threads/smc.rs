//! Self-modifying code (M6, spec §10, testing.md §6): a write onto a page that
//! backs a translated block must invalidate the cache so the next execution
//! re-lifts the changed bytes. Two write sources are covered:
//!
//! - the guest patching its own `.text`, and
//! - an embedder overwriting code between runs (`write_bytes` — loader / syscall
//!   passthrough).
//!
//! **Every guest-self-patch case runs on BOTH backends, from one body.** A JIT-compiled
//! store writes host RAM directly and can miss the SMC hook entirely, leaving a guest
//! that patches another block and calls it running the stale translation — while the
//! embedder-side tests here stay green, because `write_bytes` routes through a path the
//! guest never takes. A one-backend assertion on a two-backend property is how that
//! hides; hence `both_backends!`.
//!
//! Compiled same-block modification remains deferred (§10). Interpreter witnesses
//! for refetching later instructions in the current block live in same_block_smc.rs.

use iced_x86::code_asm::*;
use x86jit_core::{Backend, Exit, InterpreterBackend, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

const FLAT: u64 = 0x1_0000;
const MAIN: u64 = 0x1000;
const TARGET: u64 = 0x2000;
const STACK_TOP: u64 = 0x8000;

fn assemble(origin: u64, build: impl FnOnce(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    build(&mut a);
    a.assemble(origin).unwrap()
}

fn new_vm(backend: Box<dyn Backend>) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(FLAT), backend);
    vm.map(0, FLAT as usize, Prot::RW, RegionKind::Ram).unwrap();
    vm
}

/// Declare one test body and run it on both backends, so a property that holds on the
/// interpreter can never be recorded as the engine's property while the JIT breaks it.
macro_rules! both_backends {
    ($body:ident, $interp:ident, $jit:ident) => {
        #[test]
        fn $interp() {
            $body(Box::new(InterpreterBackend));
        }
        #[test]
        fn $jit() {
            $body(Box::new(JitBackend::new()));
        }
    };
}

fn run_to_hlt(vm: &Vm, cpu: &mut x86jit_core::Vcpu) {
    match cpu.run(vm, None) {
        Exit::Hlt => {}
        other => panic!("unexpected exit: {other:?}"),
    }
}

fn vm_at_47bit_top(backend: Box<dyn Backend>) -> Vm {
    // SDM Vol 1 §3.3.7.1: all bytes below 2^47 are in the lower canonical half.
    const TOP: u64 = 1 << 47;
    const LEN: usize = 2 << 20;
    let base = TOP - LEN as u64;
    let backing = vec![0u128; LEN / 16].into_boxed_slice();
    let ram = x86jit_core::HostRam {
        ptr: Box::into_raw(backing) as *mut u8,
        len: LEN,
        guest_base: base,
        dtor: Box::new(|ptr, len| {
            // SAFETY: the pointer and length describe the boxed u128 slice above.
            drop(unsafe {
                Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                    ptr.cast::<u128>(),
                    len / 16,
                ))
            });
        }),
        protect: None,
    };
    let mut vm = Vm::with_backend_host_ram(VmConfig::reserved(TOP), backend, ram).unwrap();
    vm.map(base, LEN, Prot::RWX, RegionKind::Ram).unwrap();
    vm
}

fn smc_at_47bit_top(backend: Box<dyn Backend>) {
    const TOP: u64 = 1 << 47;
    let vm = vm_at_47bit_top(backend);
    let base = vm.mem.guest_base();
    let main_addr = TOP - 0x20000;
    let target = TOP - 0x1000;
    let stack = TOP - 0x4000;
    let data = base + 0x10000;
    vm.mem.watch_range(data, 0x1000);
    vm.write_bytes(
        target,
        &assemble(target, |a| {
            a.mov(eax, 1i32).unwrap();
            a.ret().unwrap();
        }),
    )
    .unwrap();
    vm.write_bytes(
        main_addr,
        &assemble(main_addr, |a| {
            a.mov(r15, target).unwrap();
            a.call(r15).unwrap();
            a.mov(r14, target + 1).unwrap();
            a.mov(dword_ptr(r14), 2i32).unwrap();
            // End the patching block before calling the target again (§10).
            let mut next = a.create_label();
            a.jmp(next).unwrap();
            a.set_label(&mut next).unwrap();
            a.call(r15).unwrap();
            // A watched store below the code extent exercises the cold bitmap probe.
            a.mov(r14, data).unwrap();
            a.mov(dword_ptr(r14), eax).unwrap();
            a.hlt().unwrap();
        }),
    )
    .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, main_addr);
    cpu.set_reg(Reg::Rsp, stack);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(
        cpu.reg(Reg::Rax),
        2,
        "guest patch must invalidate high code"
    );
    assert_eq!(vm.mem.take_dirty_ranges(), vec![(data, 0x1000)]);
    vm.write_bytes(target + 1, &42u32.to_le_bytes()).unwrap();
    let call_only = assemble(main_addr, |a| {
        a.mov(r15, target).unwrap();
        a.call(r15).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(main_addr, &call_only).unwrap();
    cpu.set_reg(Reg::Rip, main_addr);
    cpu.set_reg(Reg::Rsp, stack);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(
        cpu.reg(Reg::Rax),
        42,
        "embedder patch must invalidate high code"
    );
}

both_backends!(
    smc_at_47bit_top,
    smc_at_47bit_top_interp,
    smc_at_47bit_top_jit
);

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn smc_47bit_top_trap_mapping_stays_sparse() {
    fn peak_rss() -> u64 {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: getrusage initializes the rusage buffer on success.
        let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        assert_eq!(result, 0);
        // SAFETY: the successful call initialized usage.
        let rss = unsafe { usage.assume_init() }.ru_maxrss as u64;
        if cfg!(target_os = "macos") {
            rss
        } else {
            rss * 1024
        }
    }
    let before = peak_rss();
    let mut vm = vm_at_47bit_top(Box::new(InterpreterBackend));
    let fresh = peak_rss();
    let code = (1 << 47) - 0x1000;
    vm.unmap(vm.mem.guest_base(), 2 << 20).unwrap();
    vm.mem.mark_code(code, 1);
    vm.map(code - 0x10000, 0x1000, Prot::RW, RegionKind::Trap)
        .unwrap();
    let after = peak_rss();
    assert!(!vm.mem.is_code_page(code >> 12));
    println!("47-bit span peak RSS bytes: before={before}, fresh={fresh}, after Trap={after}");
    assert!(fresh.saturating_sub(before) < 16 << 20);
    assert!(after.saturating_sub(fresh) < 4 << 20);
}

#[test]
fn smc_47bit_concurrent_marks_reach_compiled_store_gate() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use x86jit_core::jit_abi::run_compiled;
    use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
    use x86jit_core::{CachedBlock, StepResult};

    const TOP: u64 = 1 << 47;
    let vm = vm_at_47bit_top(Box::new(JitBackend::new()));
    let entry_addr = TOP - 0x1000;
    vm.write_bytes(entry_addr, &[0x89, 0x07, 0xf4]).unwrap(); // mov [rdi],eax; hlt
    let ir = lift_block(&vm.mem, FetchAddr::flat(entry_addr), CpuMode::Long64).unwrap();
    let CachedBlock::Compiled { entry } = vm.backend.materialize(
        &ir,
        vm.consistency,
        vm.mem.trap_window(),
        vm.mem.guest_base(),
    ) else {
        panic!("JIT backend must compile the store");
    };
    let published = AtomicU64::new(0);
    let barrier = std::sync::Barrier::new(3);
    std::thread::scope(|scope| {
        for parity in 0..2 {
            let (mem, published, barrier) = (&vm.mem, &published, &barrier);
            scope.spawn(move || {
                barrier.wait();
                for i in (parity..32).step_by(2) {
                    let addr = TOP - ((i + 1) << 16);
                    mem.mark_code(addr, 1);
                    published.fetch_or(1 << i, Ordering::Release);
                    std::thread::yield_now();
                }
            });
        }
        let mut cpu = vm.new_vcpu();
        barrier.wait();
        loop {
            let marked = published.load(Ordering::Acquire);
            for i in 0..32 {
                if marked & (1 << i) != 0 {
                    let addr = TOP - ((i + 1) << 16);
                    cpu.set_reg(Reg::Rdi, addr);
                    cpu.set_reg(Reg::Rip, entry_addr);
                    // SAFETY: the entry remains owned by vm's backend; its backing
                    // and context match the freshly materialized block.
                    let result = unsafe {
                        run_compiled(
                            entry,
                            &mut cpu.cpu,
                            &vm.mem,
                            CpuMode::Long64,
                            vm.consistency,
                        )
                    };
                    assert!(matches!(result, StepResult::Exit(Exit::Hlt)));
                    assert_eq!(vm.mem.take_dirty_code(), vec![addr >> 12]);
                }
            }
            if marked == u32::MAX as u64 {
                break;
            }
        }
    });
}

#[test]
fn smc_47bit_store_outside_watermark_follows_source_protocol() {
    use x86jit_core::jit_abi::run_compiled;
    use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
    use x86jit_core::{CachedBlock, StepResult};

    const TOP: u64 = 1 << 47;
    let vm = vm_at_47bit_top(Box::new(JitBackend::new()));
    let entry_addr = TOP - 0x1000;
    let code = TOP - 0x10000;
    vm.write_bytes(entry_addr, &[0x89, 0x07, 0xf4]).unwrap(); // mov [rdi],eax; hlt
    let ir = lift_block(&vm.mem, FetchAddr::flat(entry_addr), CpuMode::Long64).unwrap();
    let CachedBlock::Compiled { entry } = vm.backend.materialize(
        &ir,
        vm.consistency,
        vm.mem.trap_window(),
        vm.mem.guest_base(),
    ) else {
        panic!("JIT backend must compile the store");
    };
    vm.mem.mark_code(code, 1);
    let mut cpu = vm.new_vcpu();
    let degraded =
        vm.memory_stats().source_barrier_mode == x86jit_core::SourceBarrierMode::Degraded;
    for (index, (addr, calls, dirty)) in [
        (code - 0x10004, 0, false),
        (code + 0x1000, 1, false),
        (code, 2, true),
    ]
    .into_iter()
    .enumerate()
    {
        let calls = if degraded { index as u64 + 1 } else { calls };
        cpu.set_reg(Reg::Rdi, addr);
        cpu.set_reg(Reg::Rip, entry_addr);
        // SAFETY: the entry's owning backend, backing and context remain live.
        let result = unsafe {
            run_compiled(
                entry,
                &mut cpu.cpu,
                &vm.mem,
                CpuMode::Long64,
                vm.consistency,
            )
        };
        assert!(matches!(result, StepResult::Exit(Exit::Hlt)));
        let count: u64 = vm
            .backend
            .helper_calls()
            .iter()
            .map(|(_, count)| count)
            .sum();
        assert_eq!(count, calls, "helper count at {addr:#x}");
        assert_eq!(vm.mem.has_dirty_code(), dirty);
    }
    assert_eq!(vm.mem.take_dirty_code(), vec![code >> 12]);
}

/// The guest overwrites its own code: it calls `target` (caching that block),
/// patches `target`'s first instruction from `mov eax, 1` to `mov eax, 2`, then
/// calls `target` again. Without SMC invalidation the second call would replay
/// the stale cached `eax = 1`; with it, the engine re-lifts and yields `eax = 2`.
fn guest_self_modification(backend: Box<dyn Backend>) {
    let vm = new_vm(backend);

    // target: `mov eax, 1; ret`  ->  B8 01 00 00 00 C3
    let target = assemble(TARGET, |a| {
        a.mov(eax, 1i32).unwrap();
        a.ret().unwrap();
    });
    vm.write_bytes(TARGET, &target).unwrap();

    let main = assemble(MAIN, |a| {
        a.mov(r15, TARGET).unwrap();
        a.call(r15).unwrap(); // run target v1 (eax = 1), caches its block
                              // patch target's first 5 bytes to `mov eax, 2` (B8 02 00 00 00)
        a.mov(byte_ptr(TARGET), 0xB8i32).unwrap();
        a.mov(dword_ptr(TARGET + 1), 2i32).unwrap();
        a.call(r15).unwrap(); // run target v2 — must observe eax = 2
        a.hlt().unwrap();
    });
    vm.write_bytes(MAIN, &main).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN);
    cpu.set_reg(Reg::Rsp, STACK_TOP);
    run_to_hlt(&vm, &mut cpu);

    assert_eq!(
        cpu.reg(Reg::Rax) as u32,
        2,
        "second call must run the patched code"
    );
    assert!(
        vm.cache.misses() >= 2,
        "target must have been lifted twice (initial + re-lift)"
    );
}

both_backends!(
    guest_self_modification,
    guest_self_modification_interp,
    guest_self_modification_jit
);

/// An embedder overwrites a cached block between runs via `write_bytes` (the
/// loader / syscall-passthrough path). This works on both backends — the write
/// routes through the SMC hook regardless of who executes the code.
fn embedder_rewrite_reexecutes(backend: Box<dyn Backend>) {
    let vm = new_vm(backend);

    let v1 = assemble(TARGET, |a| {
        a.mov(eax, 1i32).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(TARGET, &v1).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, TARGET);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(cpu.reg(Reg::Rax) as u32, 1, "first run");

    // Overwrite the block with `mov eax, 42; hlt` and re-run from the same entry.
    let v2 = assemble(TARGET, |a| {
        a.mov(eax, 42i32).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(TARGET, &v2).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, TARGET);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(
        cpu.reg(Reg::Rax) as u32,
        42,
        "re-run must see the rewritten code"
    );
}

#[test]
fn embedder_rewrite_reexecutes_interp() {
    embedder_rewrite_reexecutes(Box::new(InterpreterBackend));
}

#[test]
fn embedder_rewrite_reexecutes_jit() {
    embedder_rewrite_reexecutes(Box::new(JitBackend::new()));
}

/// A *chained* edge must survive SMC: block MAIN ends in a direct `jmp TARGET`,
/// so after the first run its link slot points at TARGET's compiled entry. When
/// the embedder rewrites TARGET (a different code page — MAIN's block is NOT
/// invalidated), the next run of MAIN must NOT chain into TARGET's stale compiled
/// code. Requires `handle_smc` to clear the backend's link slots on invalidation
/// (otherwise the filled slot returns `RET_CHAIN` into the dropped block). JIT
/// only — the interpreter has no link slots.
#[test]
fn stale_link_slot_cleared_on_invalidation() {
    let vm = new_vm(Box::new(JitBackend::new()));

    // MAIN (page 0x1000): jump straight to TARGET (a direct, chainable edge).
    let main = assemble(MAIN, |a| {
        a.jmp(TARGET).unwrap();
    });
    vm.write_bytes(MAIN, &main).unwrap();

    // TARGET (page 0x2000): mov eax, 1; hlt.
    let v1 = assemble(TARGET, |a| {
        a.mov(eax, 1i32).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(TARGET, &v1).unwrap();

    // First run: MAIN links to TARGET (slot filled), TARGET yields eax = 1.
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(cpu.reg(Reg::Rax) as u32, 1, "first run");

    // Embedder rewrites ONLY TARGET (mov eax, 42; hlt). MAIN's page is untouched,
    // so MAIN's compiled block — and its filled link slot — survive.
    let v2 = assemble(TARGET, |a| {
        a.mov(eax, 42i32).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(TARGET, &v2).unwrap();
    let misses_before = vm.cache.misses();

    // Second run from MAIN: the stale slot must not be followed — SMC clears it,
    // MAIN re-links, TARGET is re-lifted → eax = 42.
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(
        cpu.reg(Reg::Rax) as u32,
        42,
        "chained edge must re-resolve the rewritten TARGET, not run stale code"
    );
    assert!(
        vm.cache.misses() > misses_before,
        "TARGET must have been re-lifted after invalidation"
    );
}

/// The per-vcpu fast-resolve cache (fast-dispatch R3) must not outlive an invalidation:
/// the SAME vcpu runs a block, the embedder rewrites it, and the vcpu runs it
/// again. Without the invalidation-epoch flush the vcpu's fast cache would serve
/// the stale compiled entry; with it, the cache flushes and the block re-lifts.
#[test]
fn fast_resolve_cache_flushes_on_invalidation() {
    let vm = new_vm(Box::new(JitBackend::new()));

    let v1 = assemble(TARGET, |a| {
        a.mov(eax, 1i32).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(TARGET, &v1).unwrap();

    // One vcpu, reused across both runs, so its fast-resolve cache persists.
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, TARGET);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(cpu.reg(Reg::Rax) as u32, 1, "first run");

    let v2 = assemble(TARGET, |a| {
        a.mov(eax, 42i32).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(TARGET, &v2).unwrap();
    let misses_before = vm.cache.misses();

    cpu.set_reg(Reg::Rip, TARGET);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(
        cpu.reg(Reg::Rax) as u32,
        42,
        "same vcpu must re-lift the rewritten block, not serve its stale fast entry"
    );
    assert!(
        vm.cache.misses() > misses_before,
        "the rewritten block must have been re-lifted"
    );
}

/// IBTC slots (fast-dispatch R4) inherit the same SMC coherence as link slots: an
/// indirect `jmp reg` fills a per-site descriptor pointing at TARGET's compiled
/// entry; when the embedder rewrites TARGET, the next run must not chain through
/// the stale descriptor. `invalidate_links` zeroes the IBTC slot (in the same
/// arena as link slots), so the site re-resolves. JIT only.
#[test]
fn stale_ibtc_descriptor_cleared_on_invalidation() {
    let vm = new_vm(Box::new(JitBackend::new()));

    // MAIN (page 0x1000): mov rdx, TARGET; jmp rdx  — a monomorphic indirect jump.
    let main = assemble(MAIN, |a| {
        a.mov(rdx, TARGET).unwrap();
        a.jmp(rdx).unwrap();
    });
    vm.write_bytes(MAIN, &main).unwrap();

    let v1 = assemble(TARGET, |a| {
        a.mov(eax, 1i32).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(TARGET, &v1).unwrap();

    // First run: the jmp reg fills its IBTC slot with {TARGET, v1 entry}, eax = 1.
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(cpu.reg(Reg::Rax) as u32, 1, "first run");
    assert!(vm.cache.ibtc_filled() >= 1, "IBTC must have fired");

    // Rewrite ONLY TARGET; MAIN (and its filled IBTC slot) survive.
    let v2 = assemble(TARGET, |a| {
        a.mov(eax, 42i32).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(TARGET, &v2).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(
        cpu.reg(Reg::Rax) as u32,
        42,
        "indirect edge must re-resolve the rewritten TARGET, not run stale code"
    );
}

/// A guest `rep stosb` that overwrites its own cached code must invalidate it,
/// exactly like a scalar store. The interpreter's string ops must not write guest RAM
/// through a raw pointer that bypasses `Memory::write`'s SMC `note_write`, or a
/// self-modifying `rep stos` leaves the stale block cached and replays it. `target` is
/// `mov al, 1; ret`; the guest patches its immediate byte to 42 with a one-element
/// `rep stosb`, then re-calls it.
fn self_modification_via_rep_stos(backend: Box<dyn Backend>) {
    let vm = new_vm(backend);

    // target: `mov al, 1; ret`  ->  B0 01 C3  (imm at TARGET+1)
    let target = assemble(TARGET, |a| {
        a.mov(al, 1i32).unwrap();
        a.ret().unwrap();
    });
    vm.write_bytes(TARGET, &target).unwrap();

    let main = assemble(MAIN, |a| {
        a.mov(r15, TARGET).unwrap();
        a.call(r15).unwrap(); // run target v1 (al = 1), caches + marks its page
                              // patch target's immediate byte to 42 via `rep stosb` (AL=42, one element)
        a.mov(al, 42i32).unwrap();
        a.mov(edi, (TARGET + 1) as u32).unwrap();
        a.mov(ecx, 1u32).unwrap();
        a.cld().unwrap();
        a.rep().stosb().unwrap();
        a.call(r15).unwrap(); // run target v2 — must observe al = 42
        a.hlt().unwrap();
    });
    vm.write_bytes(MAIN, &main).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN);
    cpu.set_reg(Reg::Rsp, STACK_TOP);
    run_to_hlt(&vm, &mut cpu);

    assert_eq!(
        cpu.reg(Reg::Rax) as u8,
        42,
        "second call must run the rep-stos-patched code"
    );
    assert!(
        vm.cache.misses() >= 2,
        "target must have been lifted twice (initial + re-lift after rep stos)"
    );
}

both_backends!(
    self_modification_via_rep_stos,
    self_modification_via_rep_stos_interp,
    self_modification_via_rep_stos_jit
);

/// A guest x87 store that overwrites its own cached code must invalidate it, like
/// a scalar store — the x87 helper must not write guest RAM through a raw pointer that
/// skips `Memory`'s SMC `note_write`. `target` is `mov eax, 1; ret`; the guest rewrites
/// its 32-bit immediate to 2 with `fild`/`fistp dword`, then re-calls it.
fn self_modification_via_x87_store(backend: Box<dyn Backend>) {
    const SCRATCH: u64 = 0x3000; // holds the integer to store (a non-code page)
    let vm = new_vm(backend);

    // target: `mov eax, 1; ret`  ->  B8 01 00 00 00 C3  (imm32 at TARGET+1)
    let target = assemble(TARGET, |a| {
        a.mov(eax, 1i32).unwrap();
        a.ret().unwrap();
    });
    vm.write_bytes(TARGET, &target).unwrap();
    vm.write_bytes(SCRATCH, &2u32.to_le_bytes()).unwrap();

    let main = assemble(MAIN, |a| {
        a.mov(r15, TARGET).unwrap();
        a.call(r15).unwrap(); // run target v1 (eax = 1), caches + marks its page
                              // patch target's imm32 to 2 via an x87 int store
        a.fild(dword_ptr(SCRATCH)).unwrap(); // ST0 = 2.0
        a.fistp(dword_ptr(TARGET + 1)).unwrap(); // write i32 2 into the immediate
        a.call(r15).unwrap(); // run target v2 — must observe eax = 2
        a.hlt().unwrap();
    });
    vm.write_bytes(MAIN, &main).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN);
    cpu.set_reg(Reg::Rsp, STACK_TOP);
    run_to_hlt(&vm, &mut cpu);

    assert_eq!(
        cpu.reg(Reg::Rax) as u32,
        2,
        "second call must run the x87-patched code"
    );
    assert!(
        vm.cache.misses() >= 2,
        "target must have been lifted twice (initial + re-lift after x87 store)"
    );
}

both_backends!(
    self_modification_via_x87_store,
    self_modification_via_x87_store_interp,
    self_modification_via_x87_store_jit
);

/// An MMIO read yields `Exit::MmioRead`, and after `complete_mmio_read` the guest
/// resumes and the retried load gets the supplied value (§5.2).
#[test]
fn mmio_read_resumes_with_the_supplied_value() {
    let mut vm = Vm::with_backend(VmConfig::flat(FLAT), Box::new(InterpreterBackend));
    vm.map(0x1000, 0x2000, Prot::RW, RegionKind::Ram).unwrap();
    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();

    // mov eax, [0x3000]  (an MMIO read);  hlt
    let code = assemble(MAIN, |a| {
        a.mov(eax, dword_ptr(0x3000u64)).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(MAIN, &code).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN);
    match cpu.run(&vm, None) {
        Exit::MmioRead { addr, size, .. } => {
            assert_eq!(addr, 0x3000);
            assert_eq!(size, 4);
        }
        other => panic!("expected MmioRead, got {other:?}"),
    }
    // Deliver the value; the guest re-executes the load and gets it, then hlts.
    cpu.complete_mmio_read(0xDEAD_BEEF);
    match cpu.run(&vm, None) {
        Exit::Hlt => {}
        other => panic!("expected Hlt after resume, got {other:?}"),
    }
    assert_eq!(
        cpu.reg(Reg::Rax),
        0xDEAD_BEEF,
        "load consumed the MMIO value"
    );
}

/// A `rep stos` into a `Trap` (MMIO) region must yield `Exit::MmioWrite`, not
/// silently scribble the backing buffer. A raw string path bypasses the region check
/// `Memory::write` performs; routing through it traps on the first element with RIP
/// left on the `rep` instruction (restartable).
#[test]
fn rep_stos_into_mmio_region_traps() {
    let mut vm = Vm::with_backend(VmConfig::flat(FLAT), Box::new(InterpreterBackend));
    // Code/data RAM in [0x1000, 0x3000); an MMIO trap region in [0x3000, 0x4000).
    vm.map(0x1000, 0x2000, Prot::RW, RegionKind::Ram).unwrap();
    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();

    let code = assemble(MAIN, |a| {
        a.mov(al, 0xABi32).unwrap();
        a.mov(edi, 0x3000u32).unwrap();
        a.mov(ecx, 4u32).unwrap();
        a.cld().unwrap();
        a.rep().stosb().unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(MAIN, &code).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN);
    match cpu.run(&vm, None) {
        Exit::MmioWrite {
            addr, size, value, ..
        } => {
            assert_eq!(addr, 0x3000, "trap at the first MMIO byte");
            assert_eq!(size, 1, "stosb element width");
            assert_eq!(value, 0xAB, "the AL byte being stored");
        }
        other => panic!("expected MmioWrite, got {other:?}"),
    }
    assert_eq!(
        cpu.reg(Reg::Rcx),
        4,
        "no element committed before the trap (restartable)"
    );
}

/// After an `Exit::MmioWrite`, `complete_mmio_write` lets the guest resume: the
/// retried store skips re-trapping (side effect already done by the embedder) and
/// execution continues. A following RAM write proves progress past the store.
#[test]
fn mmio_write_resumes_and_continues() {
    let mut vm = Vm::with_backend(VmConfig::flat(FLAT), Box::new(InterpreterBackend));
    vm.map(0x1000, 0x2000, Prot::RW, RegionKind::Ram).unwrap();
    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();

    // mov eax, 0x12345678 ; mov [0x3004], eax (MMIO) ; mov [0x1500], eax (RAM) ; hlt
    let code = assemble(MAIN, |a| {
        a.mov(eax, 0x1234_5678u32).unwrap();
        a.mov(dword_ptr(0x3004u64), eax).unwrap();
        a.mov(dword_ptr(0x1500u64), eax).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(MAIN, &code).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN);
    match cpu.run(&vm, None) {
        Exit::MmioWrite {
            addr, size, value, ..
        } => {
            assert_eq!(addr, 0x3004);
            assert_eq!(size, 4);
            assert_eq!(value, 0x1234_5678);
        }
        other => panic!("expected MmioWrite, got {other:?}"),
    }
    // Acknowledge the side effect; the store must not re-trap on resume.
    cpu.complete_mmio_write();
    match cpu.run(&vm, None) {
        Exit::Hlt => {}
        other => panic!("expected Hlt after resume, got {other:?}"),
    }
    let mut buf = [0u8; 4];
    vm.read_bytes(0x1500, &mut buf).unwrap();
    assert_eq!(
        u32::from_le_bytes(buf),
        0x1234_5678,
        "the RAM store after the MMIO write executed — resume made progress"
    );
}

/// JIT-side MMIO (§5.2): an inlined load into a `Trap` region is deferred
/// to the interpreter, which yields `MmioRead`; on resume the deferred load returns
/// the supplied value — identical to the interpreter backend.
#[test]
fn mmio_read_resumes_on_jit() {
    let mut vm = Vm::with_backend(VmConfig::flat(FLAT), Box::new(JitBackend::new()));
    vm.map(0x1000, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();

    let code = assemble(MAIN, |a| {
        a.mov(eax, dword_ptr(0x3000u64)).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(MAIN, &code).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN);
    match cpu.run(&vm, None) {
        Exit::MmioRead { addr, size, .. } => {
            assert_eq!(addr, 0x3000);
            assert_eq!(size, 4);
        }
        other => panic!("expected MmioRead under JIT, got {other:?}"),
    }
    cpu.complete_mmio_read(0xDEAD_BEEF);
    match cpu.run(&vm, None) {
        Exit::Hlt => {}
        other => panic!("expected Hlt after JIT resume, got {other:?}"),
    }
    assert_eq!(
        cpu.reg(Reg::Rax),
        0xDEAD_BEEF,
        "JIT load consumed the MMIO value"
    );
}

/// JIT-side MMIO write: the inlined store defers, yields `MmioWrite`, and
/// after `complete_mmio_write` the guest resumes and a following RAM store lands.
#[test]
fn mmio_write_resumes_on_jit() {
    let mut vm = Vm::with_backend(VmConfig::flat(FLAT), Box::new(JitBackend::new()));
    vm.map(0x1000, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();

    let code = assemble(MAIN, |a| {
        a.mov(eax, 0x1234_5678u32).unwrap();
        a.mov(dword_ptr(0x3004u64), eax).unwrap();
        a.mov(dword_ptr(0x1500u64), eax).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(MAIN, &code).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN);
    match cpu.run(&vm, None) {
        Exit::MmioWrite {
            addr, size, value, ..
        } => {
            assert_eq!(addr, 0x3004);
            assert_eq!(size, 4);
            assert_eq!(value, 0x1234_5678);
        }
        other => panic!("expected MmioWrite under JIT, got {other:?}"),
    }
    cpu.complete_mmio_write();
    match cpu.run(&vm, None) {
        Exit::Hlt => {}
        other => panic!("expected Hlt after JIT resume, got {other:?}"),
    }
    let mut buf = [0u8; 4];
    vm.read_bytes(0x1500, &mut buf).unwrap();
    assert_eq!(
        u32::from_le_bytes(buf),
        0x1234_5678,
        "JIT resume made progress"
    );
}

/// `Vm::unmap` must invalidate blocks cached from the unmapped range, so a
/// later execution faults instead of running the stale translation. The block is
/// cached by a first run, the region is unmapped, and a re-run must not return `Hlt`.
#[test]
fn unmap_invalidates_cached_blocks() {
    let mut vm = Vm::with_backend(VmConfig::flat(FLAT), Box::new(InterpreterBackend));
    vm.map(TARGET, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
    let code = assemble(TARGET, |a| {
        a.mov(eax, 1i32).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(TARGET, &code).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, TARGET);
    run_to_hlt(&vm, &mut cpu); // caches TARGET's block

    vm.unmap(TARGET, 0x1000).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, TARGET);
    let exit = cpu.run(&vm, None);
    assert!(
        !matches!(exit, Exit::Hlt),
        "must not run the stale cached block after unmap: {exit:?}"
    );
}

/// A write to a NON-code page must not perturb the cache (no false invalidation).
#[test]
fn write_to_data_page_does_not_invalidate() {
    let vm = new_vm(Box::new(InterpreterBackend));
    let code = assemble(TARGET, |a| {
        a.mov(eax, 7i32).unwrap();
        a.mov(dword_ptr(0x4000u64), eax).unwrap(); // store to a far data page
        a.hlt().unwrap();
    });
    vm.write_bytes(TARGET, &code).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, TARGET);
    run_to_hlt(&vm, &mut cpu);
    let misses_after_first = vm.cache.misses();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, TARGET);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(
        vm.cache.misses(),
        misses_after_first,
        "data-page write must not re-lift code"
    );
    assert!(vm.cache.hits() >= 1, "second run should hit the cache");
}

/// The two page-boundary cases the JIT's inline SMC gate can get wrong.
///
/// That gate is a watermark — `(page - lo) < len` on the page of the store's FIRST
/// byte — so two shapes have to be pinned by execution rather than by reading it:
///
/// - **A store that starts one page BELOW the code range and spills into it.** Its
///   first byte's page is not code, so a naive range holding exactly the code extent
///   rejects it and the patch to the range's lowest page is lost. `widen_code_range`
///   keeps `lo` one page low precisely for this; a store is at most 64 bytes against a
///   4096-byte page, so one page of skew is exactly enough and no more.
/// - **A store to the HIGHEST code page**, the other end of the same compare.
///
/// The layout puts `main` BETWEEN the two targets so the low target really is the
/// bottom of the range and the page below it really is not code.
fn smc_at_the_edges_of_the_code_range(backend: Box<dyn Backend>) {
    const TARGET_LO: u64 = 0x3000; // lowest code page; page 0x2 below it is data
    const MAIN_MID: u64 = 0x5000;
    const TARGET_HI: u64 = 0x6000; // highest code page

    let vm = new_vm(backend);
    for at in [TARGET_LO, TARGET_HI] {
        let code = assemble(at, |a| {
            a.mov(eax, 1i32).unwrap(); // B8 01 00 00 00 — imm32 at at+1
            a.ret().unwrap();
        });
        vm.write_bytes(at, &code).unwrap();
    }

    let main = assemble(MAIN_MID, |a| {
        // Cache and mark both targets, so the code range spans pages 3..=6.
        a.mov(r15, TARGET_LO).unwrap();
        a.call(r15).unwrap();
        a.mov(r14, TARGET_HI).unwrap();
        a.call(r14).unwrap();

        // Straddle: a dword at TARGET_LO-1 writes 0x2fff..0x3002, so its FIRST byte
        // is on the data page below the code range. Little-endian 0x0002_b800 lands
        // B8 02 00 over TARGET_LO..+2, turning `mov eax,1` into `mov eax,2`.
        a.mov(dword_ptr(TARGET_LO - 1), 0x0002_b800u32 as i32)
            .unwrap();
        // Wholly inside the range's top page.
        a.mov(dword_ptr(TARGET_HI + 1), 3i32).unwrap();

        a.call(r15).unwrap();
        a.mov(ebx, eax).unwrap(); // ebx = low target's result
        a.call(r14).unwrap(); // eax = high target's result
        a.hlt().unwrap();
    });
    vm.write_bytes(MAIN_MID, &main).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN_MID);
    cpu.set_reg(Reg::Rsp, STACK_TOP);
    run_to_hlt(&vm, &mut cpu);

    assert_eq!(
        cpu.reg(Reg::Rbx) as u32,
        2,
        "a store straddling into the LOWEST code page must invalidate it"
    );
    assert_eq!(
        cpu.reg(Reg::Rax) as u32,
        3,
        "a store to the HIGHEST code page must invalidate it"
    );
}

both_backends!(
    smc_at_the_edges_of_the_code_range,
    smc_at_the_edges_of_the_code_range_interp,
    smc_at_the_edges_of_the_code_range_jit
);

/// Guest code ABOVE the old 4 GiB `CODE_WINDOW` must be SMC-tracked.
///
/// The code-page table was once one bool per page capped at 4 GiB, so `mark_code` and
/// `note_write` past the cap silently no-opped: a guest whose `.text` sat high patched
/// itself and kept running the stale translation, on both backends, with nothing
/// reporting anything. "Guest code always lives low" holds for the fixtures and is not a
/// guarantee — a dynamic image mapped high breaks it outright — so the table is a packed
/// bitset over the whole span.
fn smc_tracks_code_above_the_old_window(backend: Box<dyn Backend>) {
    const HIGH: u64 = 8 << 30; // 8 GiB — twice the old cap
    const MAIN_HI: u64 = HIGH + 0x1000;
    const TARGET_HI: u64 = HIGH + 0x2000;

    // `Reserved`, not `Flat`: this needs a span past 4 GiB, and `Reserved` goes through
    // `alloc_zeroed`, so the pages the test never touches are never committed.
    // Just past the code: a larger reservation can exceed a CI host's overcommit
    // limit (xj-jge.18).
    let mut vm = Vm::with_backend(VmConfig::reserved(HIGH + (1 << 20)), backend);
    vm.map(HIGH, 0x8000, Prot::RWX, RegionKind::Ram).unwrap();

    let target = assemble(TARGET_HI, |a| {
        a.mov(eax, 1i32).unwrap();
        a.ret().unwrap();
    });
    vm.write_bytes(TARGET_HI, &target).unwrap();

    let main = assemble(MAIN_HI, |a| {
        a.mov(r15, TARGET_HI).unwrap();
        a.call(r15).unwrap();
        // Through a register: an absolute displacement is an i32, so a high address
        // cannot be encoded directly.
        a.mov(r14, TARGET_HI).unwrap();
        a.mov(byte_ptr(r14), 0xB8i32).unwrap();
        a.mov(dword_ptr(r14 + 1), 2i32).unwrap();
        a.call(r15).unwrap();
        a.hlt().unwrap();
    });
    vm.write_bytes(MAIN_HI, &main).unwrap();

    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, MAIN_HI);
    cpu.set_reg(Reg::Rsp, HIGH + 0x7000);
    run_to_hlt(&vm, &mut cpu);

    assert_eq!(
        cpu.reg(Reg::Rax) as u32,
        2,
        "code above 4 GiB must be invalidated like any other"
    );
}

both_backends!(
    smc_tracks_code_above_the_old_window,
    smc_tracks_code_above_the_old_window_interp,
    smc_tracks_code_above_the_old_window_jit
);

fn unmap_shared_code_page(backend: Box<dyn Backend>) {
    let mut vm = Vm::with_backend(VmConfig::flat(FLAT), backend);
    vm.map(0x1000, 0x400, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x1400, 0x400, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1400, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1400);
    run_to_hlt(&vm, &mut cpu);
    vm.unmap(0x1000, 0x400).unwrap();
    vm.write_bytes(0x1401, &2u32.to_le_bytes()).unwrap();
    cpu.set_reg(Reg::Rip, 0x1400);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(cpu.reg(Reg::Rax), 2);
}
both_backends!(
    unmap_shared_code_page,
    unmap_shared_code_page_interp,
    unmap_shared_code_page_jit
);

#[cfg(unix)]
fn alias_vm(backend: Box<dyn Backend>) -> Vm {
    use std::os::fd::AsRawFd;
    const LEN: usize = 0x20_000;
    let file = tempfile_shared_page();
    // SAFETY: reserve a fresh writable anonymous range; MAP_FIXED below replaces
    // only pages inside this owned reservation with two views of the same file.
    let ptr = unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            LEN,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        assert_ne!(ptr, libc::MAP_FAILED);
        for offset in [0x4000, 0x18000] {
            assert_eq!(
                libc::mmap(
                    ptr.cast::<u8>().add(offset).cast(),
                    0x4000,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    file.as_raw_fd(),
                    0
                ),
                ptr.cast::<u8>().add(offset).cast()
            );
        }
        ptr.cast::<u8>()
    };
    let ram = x86jit_core::HostRam {
        ptr,
        len: LEN,
        guest_base: 0,
        dtor: Box::new(|ptr, len| {
            // SAFETY: this is the reservation allocated above, including its views.
            assert_eq!(unsafe { libc::munmap(ptr.cast(), len) }, 0);
        }),
        protect: None,
    };
    let mut vm = Vm::with_backend_host_ram(VmConfig::reserved(LEN as u64), backend, ram).unwrap();
    vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x4000, 0x4000, Prot::RX, RegionKind::Ram).unwrap();
    vm.map(0x18000, 0x4000, Prot::RW, RegionKind::Ram).unwrap();
    vm.map(0x8000, 0x10000, Prot::RW, RegionKind::Ram).unwrap();
    vm.map(0x1c000, 0x4000, Prot::RW, RegionKind::Ram).unwrap();
    vm
}

#[cfg(unix)]
#[test]
fn child_store_invalidates_an_alias_removed_before_dispatch() {
    use x86jit_core::{child, ChildArena, ChildStep, CpuMode, CpuState};

    for jit in [false, true] {
        let backend: Box<dyn Backend> = if jit {
            Box::new(JitBackend::new())
        } else {
            Box::new(InterpreterBackend)
        };
        let vm = alias_vm(backend);
        vm.write_bytes(0x1000, &[0x89, 0x18]).unwrap();
        vm.write_bytes(0x4000, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
        let view = vm.child_view(CpuMode::Long64);
        vm.mem.declare_code_alias(0x4000, 0x18000, 0x4000).unwrap();
        let mut parent = vm.new_vcpu();
        parent.cpu.rip = 0x4000;
        assert!(matches!(parent.run(&vm, Some(100)), Exit::Hlt));
        assert_eq!(parent.cpu.gpr[0], 1);
        let mut cpu = CpuState {
            rip: 0x1000,
            ..CpuState::default()
        };
        cpu.gpr[0] = 0x18001;
        cpu.gpr[3] = 2;
        let mut arena = ChildArena::new(2);
        assert!(matches!(
            child::step(&view, &mut cpu, &mut arena),
            ChildStep::Continue
        ));
        assert!(vm.mem.has_dirty_code());
        assert!(vm.mem.remove_code_alias(0x4000, 0x18000, 0x4000));
        assert!(vm.mem.is_code_page(0x18));
        parent.cpu.rip = 0x4000;
        assert!(matches!(parent.run(&vm, Some(100)), Exit::Hlt));
        assert_eq!(parent.cpu.gpr[0], 2, "jit={jit}");
        assert!(!vm.mem.has_dirty_code());
    }
}

#[cfg(unix)]
fn tempfile_shared_page() -> std::fs::File {
    use std::os::fd::FromRawFd;
    let mut name = std::env::temp_dir()
        .join("x86jit-smc-XXXXXX")
        .as_os_str()
        .as_encoded_bytes()
        .to_vec();
    name.push(0);
    // SAFETY: mkstemp receives a writable NUL-terminated template; on success
    // ownership of its descriptor moves to File, and unlink removes only its name.
    unsafe {
        let fd = libc::mkstemp(name.as_mut_ptr().cast());
        assert!(fd >= 0);
        assert_eq!(libc::unlink(name.as_ptr().cast()), 0);
        let file = std::fs::File::from_raw_fd(fd);
        file.set_len(0x4000).unwrap();
        file
    }
}

#[cfg(unix)]
fn declared_alias_patches(backend: Box<dyn Backend>) {
    let vm = alias_vm(backend);
    vm.mem.declare_code_alias(0x4000, 0x18000, 0x4000).unwrap();
    vm.write_bytes(0x4000, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x4000);
    run_to_hlt(&vm, &mut cpu);
    for patch in 0u32..4 {
        let value = patch + 2;
        vm.write_bytes(0x3000, &value.to_le_bytes()).unwrap();
        let code = assemble(MAIN, |a| {
            match patch {
                0 => a.mov(dword_ptr(0x18001u64), value as i32).unwrap(),
                1 => {
                    a.mov(esi, 0x3000).unwrap();
                    a.mov(edi, 0x18001).unwrap();
                    a.mov(ecx, 4).unwrap();
                    a.rep().movsb().unwrap();
                }
                2 => {
                    a.lock().add(byte_ptr(0x18001u64), 1).unwrap();
                }
                _ => {}
            }
            // SDM Vol 3A §§11.1.3, 11.3: serialize after modifying code.
            a.xor(eax, eax).unwrap();
            a.cpuid().unwrap();
            a.hlt().unwrap();
        });
        vm.write_bytes(MAIN, &code).unwrap();
        if patch == 3 {
            // An external writer must explicitly report its completed write.
            // SAFETY: the shared view is inside this VM's live host reservation;
            // the vCPU is stopped and these four bytes have no concurrent readers.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    value.to_le_bytes().as_ptr(),
                    vm.mem.host_base().add(0x18001).cast_mut(),
                    4,
                );
            }
            vm.mem.note_write(0x18001, 4);
        }
        cpu.set_reg(Reg::Rip, MAIN);
        run_to_hlt(&vm, &mut cpu);
        cpu.set_reg(Reg::Rip, 0x4000);
        run_to_hlt(&vm, &mut cpu);
        assert_eq!(cpu.reg(Reg::Rax), value as u64, "patch path {patch}");
        assert!(
            vm.mem.is_code_page(0x18),
            "alias must be re-tagged after invalidation"
        );
    }
}
#[cfg(unix)]
both_backends!(
    declared_alias_patches,
    declared_alias_patches_interp,
    declared_alias_patches_jit
);

#[cfg(unix)]
#[test]
fn declared_alias_invalidation_every_tier() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use x86jit_core::{FetchVerdict, RegionCaps};
    let caps = RegionCaps {
        max_blocks: 16,
        max_icount: 256,
    };
    for tier in 0..6 {
        for precise in [false, true] {
            let jit = if tier == 2 || tier == 5 {
                JitBackend::with_superblocks(caps)
            } else {
                JitBackend::new()
            };
            let handle = jit.tier_up_handle();
            let backend: Box<dyn Backend> = if tier == 0 {
                Box::new(InterpreterBackend)
            } else {
                Box::new(jit)
            };
            let mut vm = alias_vm(backend);
            vm.set_precise_split_stores(precise);
            if tier >= 3 {
                vm.set_tier_up_after(Some(1));
            }
            if tier >= 4 {
                vm.set_tier_up_background(true);
            }
            let approvals = Arc::new(AtomicUsize::new(0));
            vm.set_fetch_policy(Box::new({
                let approvals = approvals.clone();
                move |page| {
                    if page == 0x4000 {
                        approvals.fetch_add(1, Ordering::Relaxed);
                    }
                    FetchVerdict::Allow
                }
            }));
            // A small loop forms an actual multi-block region when requested.
            vm.write_bytes(
                0x4000,
                &[
                    0xb8, 1, 0, 0, 0, 0xb9, 3, 0, 0, 0, 0xeb, 0, 0xff, 0xc9, 0x75, 0xfc, 0xf4,
                ],
            )
            .unwrap();
            let mut cpu = vm.new_vcpu();
            for _ in 0..6 {
                cpu.set_reg(Reg::Rip, 0x4000);
                run_to_hlt(&vm, &mut cpu);
                handle.wait_idle();
            }
            if tier == 2 {
                vm.test_support_force_region(0x4000, caps, 0x1f80).unwrap();
            }
            if tier >= 4 {
                assert!(vm.cache.tier_bg_published() > 0);
            }
            if tier == 2 || tier == 5 {
                assert!(vm.cache.regions() > 0);
            }
            // Registration after translation must tag the existing code's other view.
            vm.mem.declare_code_alias(0x18000, 0x4000, 0x4000).unwrap();
            let approved = approvals.load(Ordering::Relaxed);
            vm.write_bytes(
                MAIN,
                &assemble(MAIN, |a| {
                    a.mov(dword_ptr(0x18001u64), 9i32).unwrap();
                    a.xor(eax, eax).unwrap();
                    a.cpuid().unwrap();
                    a.hlt().unwrap();
                }),
            )
            .unwrap();
            cpu.set_reg(Reg::Rip, MAIN);
            run_to_hlt(&vm, &mut cpu);
            cpu.set_reg(Reg::Rip, 0x4000);
            run_to_hlt(&vm, &mut cpu);
            assert_eq!(cpu.reg(Reg::Rax), 9, "tier={tier}, precise={precise}");
            assert!(
                approvals.load(Ordering::Relaxed) > approved,
                "alias invalidation must forget the RX fetch approval"
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn declared_alias_rejects_pending_background_translation() {
    use x86jit_core::RegionCaps;
    for regions in [false, true] {
        let jit = if regions {
            JitBackend::with_superblocks(RegionCaps {
                max_blocks: 16,
                max_icount: 256,
            })
        } else {
            JitBackend::new()
        };
        let handle = jit.tier_up_handle();
        let pause = handle.pause_compiler();
        let mut vm = alias_vm(Box::new(jit));
        vm.set_tier_up_after(Some(1));
        vm.set_tier_up_background(true);
        vm.mem.declare_code_alias(0x4000, 0x18000, 0x4000).unwrap();
        vm.write_bytes(
            0x4000,
            &[
                0xb8, 1, 0, 0, 0, 0xb9, 3, 0, 0, 0, 0xeb, 0, 0xff, 0xc9, 0x75, 0xfc, 0xf4,
            ],
        )
        .unwrap();
        let mut cpu = vm.new_vcpu();
        for _ in 0..3 {
            cpu.set_reg(Reg::Rip, 0x4000);
            run_to_hlt(&vm, &mut cpu);
        }
        vm.write_bytes(0x18001, &7u32.to_le_bytes()).unwrap();
        vm.write_bytes(MAIN, &[0xf4]).unwrap();
        cpu.set_reg(Reg::Rip, MAIN);
        run_to_hlt(&vm, &mut cpu);
        drop(pause);
        handle.wait_idle();
        cpu.set_reg(Reg::Rip, MAIN);
        run_to_hlt(&vm, &mut cpu);
        assert!(vm.cache.tier_bg_rejected() > 0);
        cpu.set_reg(Reg::Rip, 0x4000);
        run_to_hlt(&vm, &mut cpu);
        assert_eq!(cpu.reg(Reg::Rax), 7);
    }
}

#[test]
fn code_alias_lifecycle_preserves_other_views() {
    let mut vm = new_vm(Box::new(InterpreterBackend));
    let mem = &vm.mem;
    mem.mark_code(0x4000, 1);
    mem.mark_code(0x9000, 1);
    mem.declare_code_alias(0x8000, 0x4000, 0x1000).unwrap();
    mem.declare_code_alias(0x9000, 0x8000, 0x1000).unwrap();
    mem.declare_code_alias(0x4000, 0x8000, 0x1000).unwrap(); // idempotent
    mem.note_write(0x8000, 1);
    assert_eq!(mem.take_dirty_code(), vec![4, 9]);
    mem.clear_code_page(4);
    assert!(
        mem.is_code_page(8),
        "the other translation still needs the alias tag"
    );
    mem.note_write(0x4000, 1);
    assert_eq!(mem.take_dirty_code(), vec![9]);
    assert!(mem.remove_code_alias(0x8000, 0x4000, 0x1000));
    assert!(!mem.is_code_page(4));
    assert!(mem.is_code_page(8));
    assert!(!mem.remove_code_alias(0x8000, 0x4000, 0x1000));
    mem.clear_code_page(9);
    assert!(!mem.is_code_page(8));
    mem.mark_code(0x9000, 1);
    assert!(mem.is_code_page(8));
    mem.clear_all_code_pages();
    assert!(!mem.is_code_page(8));
    mem.mark_code(0x9000, 1);
    assert!(mem.remove_code_alias(0x9000, 0x8000, 0x1000));
    assert!(!mem.is_code_page(8));
    assert!(mem.is_code_page(9));
    assert!(mem.declare_code_alias(0, 0x4000, 0).is_err());
    assert!(mem.declare_code_alias(0x4000, u64::MAX, 1).is_err());
    // Unmap removes declarations, including pairs whose other view survives.
    mem.declare_code_alias(0x8000, 0x4000, 0x1000).unwrap();
    vm.unmap(0, FLAT as usize).unwrap();
    assert!(!vm.mem.remove_code_alias(0x8000, 0x4000, 0x1000));
}

#[test]
fn code_alias_subpage_overlap_and_split_write() {
    let vm = new_vm(Box::new(InterpreterBackend));
    let mem = &vm.mem;
    mem.declare_code_alias(0x4001, 0x8fff, 4).unwrap();
    mem.declare_code_alias(0x4002, 0x8fff, 2).unwrap();
    mem.mark_code(0x4000, 1);
    assert!(mem.is_code_page(8));
    assert!(mem.is_code_page(9));
    assert!(mem.remove_code_alias(0x4001, 0x8fff, 4));
    assert!(mem.is_code_page(8));
    assert!(mem.is_code_page(9));
    mem.note_write(0x8fff, 2);
    let mut dirty = mem.take_dirty_code();
    dirty.sort_unstable();
    dirty.dedup();
    assert_eq!(dirty, vec![4]);
    mem.remove_code_alias(0x4002, 0x8fff, 2);
    assert!(!mem.is_code_page(8));
    assert!(!mem.is_code_page(9));
    assert!(mem.is_code_page(4));
}

#[cfg(unix)]
#[test]
fn code_alias_unmap_preserves_unrelated_subpage_declaration() {
    let mut vm = Vm::with_backend(VmConfig::flat(FLAT), Box::new(InterpreterBackend));
    vm.map(0x1000, 0x400, Prot::RW, RegionKind::Ram).unwrap();
    vm.map(0x1400, 0x400, Prot::RW, RegionKind::Ram).unwrap();
    vm.mem.declare_code_alias(0x1400, 0x4000, 0x400).unwrap();
    vm.mem.declare_code_alias(0x1000, 0x8000, 0x400).unwrap();
    vm.unmap(0x1000, 0x400).unwrap();
    assert!(!vm.mem.remove_code_alias(0x1000, 0x8000, 0x400));
    assert!(vm.mem.remove_code_alias(0x1400, 0x4000, 0x400));
}

#[cfg(unix)]
#[test]
fn declared_alias_concurrent_registration_reaches_next_compiled_store() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use x86jit_core::jit_abi::run_compiled;
    use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
    use x86jit_core::{CachedBlock, StepResult};
    let vm = alias_vm(Box::new(JitBackend::new()));
    vm.mem.mark_code(0x4000, 1);
    vm.write_bytes(MAIN, &[0x89, 0x07, 0xf4]).unwrap(); // mov [rdi],eax; hlt
    let ir = lift_block(&vm.mem, FetchAddr::flat(MAIN), CpuMode::Long64).unwrap();
    let CachedBlock::Compiled { entry } = vm.backend.materialize(
        &ir,
        vm.consistency,
        vm.mem.trap_window(),
        vm.mem.guest_base(),
    ) else {
        panic!("compiled store required")
    };
    let before = alias_watermark(&vm);
    let published = AtomicBool::new(false);
    let started = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !started.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            vm.mem.declare_code_alias(0x4000, 0x18000, 0x4000).unwrap();
            published.store(true, Ordering::Release);
        });
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rdi, 0xc000);
        started.store(true, Ordering::Release);
        while !published.load(Ordering::Acquire) {
            // SAFETY: vm owns the compiled entry and its matching memory/context.
            assert!(matches!(
                unsafe {
                    run_compiled(
                        entry,
                        &mut cpu.cpu,
                        &vm.mem,
                        CpuMode::Long64,
                        vm.consistency,
                    )
                },
                StepResult::Exit(Exit::Hlt)
            ));
        }
        let after = alias_watermark(&vm);
        if vm.memory_stats().source_barrier_mode == x86jit_core::SourceBarrierMode::Process {
            assert_ne!(before, after, "registration must widen the gate");
        } else {
            assert_eq!(
                before,
                1 << 31,
                "degraded gate covers all RAM from construction"
            );
        }
        assert!((0x18000u64 >> 16).wrapping_sub(after >> 32) < (after & 0xffff_ffff));
        cpu.set_reg(Reg::Rdi, 0x18000);
        // SAFETY: vm still owns the entry and backing; registration has completed.
        assert!(matches!(
            unsafe {
                run_compiled(
                    entry,
                    &mut cpu.cpu,
                    &vm.mem,
                    CpuMode::Long64,
                    vm.consistency,
                )
            },
            StepResult::Exit(Exit::Hlt)
        ));
        assert_eq!(vm.mem.take_dirty_code(), vec![4]);
    });
}

#[cfg(unix)]
fn declared_alias_split_store(backend: Box<dyn Backend>) {
    let mut vm = alias_vm(backend);
    vm.set_precise_split_stores(true);
    vm.mem.declare_code_alias(0x4000, 0x18000, 0x4000).unwrap();
    // The immediate crosses a guest code-page boundary in both views.
    vm.write_bytes(0x4ffc, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
    vm.write_bytes(
        MAIN,
        &assemble(MAIN, |a| {
            a.mov(dword_ptr(0x18ffdu64), 2i32).unwrap();
            // SDM Vol 3A §§11.1.3, 11.3: serialize before executing patched code.
            a.xor(eax, eax).unwrap();
            a.cpuid().unwrap();
            a.hlt().unwrap();
        }),
    )
    .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x4ffc);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(cpu.reg(Reg::Rax), 1);
    cpu.set_reg(Reg::Rip, MAIN);
    run_to_hlt(&vm, &mut cpu);
    cpu.set_reg(Reg::Rip, 0x4ffc);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(cpu.reg(Reg::Rax), 2);
}
#[cfg(unix)]
both_backends!(
    declared_alias_split_store,
    declared_alias_split_store_interp,
    declared_alias_split_store_jit
);

#[cfg(unix)]
fn alias_watermark(vm: &Vm) -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    // SAFETY: Memory owns the live, aligned atomic exposed by this ABI pointer.
    unsafe { (&*(vm.mem.code_range_ptr() as *const AtomicU64)).load(Ordering::Acquire) }
}

#[cfg(unix)]
#[test]
fn declared_alias_early_registration_widens_precompiled_store_gate() {
    alias_registration_widens_precompiled_store_gate(true);
}

#[cfg(unix)]
#[test]
fn declared_alias_late_registration_widens_precompiled_store_gate() {
    alias_registration_widens_precompiled_store_gate(false);
}

#[cfg(unix)]
fn alias_registration_widens_precompiled_store_gate(early: bool) {
    use x86jit_core::jit_abi::run_compiled;
    use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
    use x86jit_core::{CachedBlock, StepResult};
    {
        let vm = alias_vm(Box::new(JitBackend::new()));
        vm.write_bytes(0x4000, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
        vm.write_bytes(MAIN, &[0x89, 0x07, 0xf4]).unwrap();
        let ir = lift_block(&vm.mem, FetchAddr::flat(MAIN), CpuMode::Long64).unwrap();
        let CachedBlock::Compiled { entry } = vm.backend.materialize(
            &ir,
            vm.consistency,
            vm.mem.trap_window(),
            vm.mem.guest_base(),
        ) else {
            panic!("compiled store required")
        };
        vm.mem.mark_code(MAIN, 1);
        let before = alias_watermark(&vm);
        assert_eq!(
            (0x18000u64 >> 16).wrapping_sub(before >> 32) < (before & 0xffff_ffff),
            vm.memory_stats().source_barrier_mode == x86jit_core::SourceBarrierMode::Degraded,
            "only degraded mode starts with an always-open store gate"
        );
        if early {
            vm.mem.declare_code_alias(0x4000, 0x18000, 0x4000).unwrap();
        }
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x4000);
        run_to_hlt(&vm, &mut cpu);
        if !early {
            assert_eq!(before, alias_watermark(&vm));
            vm.mem.declare_code_alias(0x4000, 0x18000, 0x4000).unwrap();
        }
        let after = alias_watermark(&vm);
        if vm.memory_stats().source_barrier_mode == x86jit_core::SourceBarrierMode::Process {
            assert_ne!(before, after, "registration/marking must widen the gate");
        } else {
            assert_eq!(before, after, "degraded gate remains open");
        }
        assert!((0x18000u64 >> 16).wrapping_sub(after >> 32) < (after & 0xffff_ffff));
        cpu.set_reg(Reg::Rdi, 0x18001);
        cpu.set_reg(Reg::Rax, 2);
        // SAFETY: vm owns the compiled entry and its matching memory/context.
        assert!(matches!(
            unsafe {
                run_compiled(
                    entry,
                    &mut cpu.cpu,
                    &vm.mem,
                    CpuMode::Long64,
                    vm.consistency,
                )
            },
            StepResult::Exit(Exit::Hlt)
        ));
        assert!(vm.mem.has_dirty_code());
        cpu.set_reg(Reg::Rip, 0x4000);
        run_to_hlt(&vm, &mut cpu);
        assert_eq!(cpu.reg(Reg::Rax), 2, "early={early}");
    }
}

#[cfg(unix)]
#[path = "../../../x86jit-cranelift/src/unwind_test_signal.rs"]
mod smc_signal;

#[cfg(unix)]
#[test]
fn faulting_split_store_invalidates_code_every_tier() {
    use x86jit_core::RegionCaps;
    // Fork only in a fresh process: other tests can hold global JIT locks.
    if std::env::var_os("X86JIT_SMC_FAULT_CHILD").is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "smc::faulting_split_store_invalidates_code_every_tier",
                "--nocapture",
            ])
            .env("X86JIT_SMC_FAULT_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let caps = RegionCaps {
        max_blocks: 16,
        max_icount: 256,
    };
    let mut failures = Vec::new();
    for tier in 0..6 {
        for alias in [false, true] {
            for replay in [false, true] {
                for precise in [false, true] {
                    let status = smc_signal::child(|| {
                        let jit = if tier == 2 || tier == 5 {
                            JitBackend::with_superblocks(caps)
                        } else {
                            JitBackend::new()
                        };
                        let handle = jit.tier_up_handle();
                        let backend: Box<dyn Backend> = if tier == 0 {
                            Box::new(InterpreterBackend)
                        } else {
                            Box::new(jit)
                        };
                        let mut vm = alias_vm(backend);
                        vm.set_precise_split_stores(precise);
                        vm.mem.set_host_fault_recovery(true);
                        if tier >= 3 {
                            vm.set_tier_up_after(Some(1));
                        }
                        if tier >= 4 {
                            vm.set_tier_up_background(true);
                        }
                        if alias {
                            vm.mem.declare_code_alias(0x4000, 0x18000, 0x4000).unwrap();
                        }
                        let target = if alias { 0x7ff0 } else { 0x1bff0 };
                        vm.write_bytes(target, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
                        vm.write_bytes(0x2000, &[0x31, 0xc0, 0x0f, 0xa2, 0xf4])
                            .unwrap();
                        vm.write_bytes(
                            MAIN,
                            &assemble(MAIN, |a| {
                                let mut top = a.create_label();
                                a.mov(ecx, 3).unwrap();
                                a.set_label(&mut top).unwrap();
                                a.dec(ecx).unwrap();
                                a.jnz(top).unwrap();
                                a.vmovdqu(ymmword_ptr(rdi), ymm0).unwrap();
                                a.hlt().unwrap();
                            }),
                        )
                        .unwrap();
                        let mut cpu = vm.new_vcpu();
                        // Warm the writer on RAM so tier-up happens before its fault.
                        cpu.set_reg(Reg::Rdi, 0xc000);
                        for _ in 0..6 {
                            cpu.set_reg(Reg::Rip, MAIN);
                            run_to_hlt(&vm, &mut cpu);
                            cpu.set_reg(Reg::Rip, target);
                            run_to_hlt(&vm, &mut cpu);
                            handle.wait_idle();
                        }
                        if tier == 2 {
                            vm.test_support_force_region(MAIN, caps, 0x1f80).unwrap();
                            // Forcing a region discards earlier translations.
                            cpu.set_reg(Reg::Rip, target);
                            run_to_hlt(&vm, &mut cpu);
                        }
                        if tier >= 4 {
                            assert!(vm.cache.tier_bg_published() > 0);
                        }
                        if tier == 2 || tier == 5 {
                            assert!(vm.cache.regions() > 0);
                        }
                        let mut payload = [0u8; 16];
                        payload[..6].copy_from_slice(&[0xb8, 2, 0, 0, 0, 0xf4]);
                        cpu.set_xmm(0, u128::from_le_bytes(payload));
                        cpu.set_ymm_hi(0, 0);
                        cpu.set_reg(Reg::Rdi, 0x1bff0);
                        cpu.set_reg(Reg::Rip, MAIN);
                        if replay {
                            vm.unmap(0x1c000, 0x4000).unwrap();
                        }
                        smc_signal::HANDLE.store(
                            cpu.fault_handle().as_usize(),
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        // SAFETY: the child owns this aligned mapping and recovery handle.
                        unsafe {
                            smc_signal::install();
                            assert_eq!(
                                libc::mprotect(
                                    vm.mem.host_base().add(0x1c000).cast_mut().cast(),
                                    0x4000,
                                    libc::PROT_NONE
                                ),
                                0
                            );
                        }
                        let exit = cpu.run(&vm, None);
                        assert!(matches!(exit, Exit::UnmappedMemory { .. }), "{exit:?}");
                        let mut actual = [0u8; 6];
                        vm.read_bytes(target, &mut actual).unwrap();
                        let expected = u32::from_le_bytes(actual[1..5].try_into().unwrap()) as u64;
                        if tier != 0 && !precise {
                            assert_eq!(expected, 2, "compiled low half must have landed");
                        }
                        // SDM Vol 3A §§11.1.3, 11.3: serialize before executing patched code.
                        cpu.set_reg(Reg::Rip, 0x2000);
                        run_to_hlt(&vm, &mut cpu);
                        cpu.set_reg(Reg::Rip, target);
                        run_to_hlt(&vm, &mut cpu);
                        assert_eq!(
                            cpu.reg(Reg::Rax),
                            expected,
                            "tier={tier}, alias={alias}, replay={replay}, precise={precise}"
                        );
                    });
                    if status != 0 {
                        failures.push((tier, alias, replay, precise, status));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:?}");
}

fn byte_precise_code_page(backend: Box<dyn Backend>) {
    use x86jit_core::{BlockKey, CpuMode};
    let vm = new_vm(backend);
    for pc in [TARGET, TARGET + 0x100] {
        vm.write_bytes(pc, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
    }
    // A different page runs the store, so the instruction stream is untouched.
    vm.write_bytes(
        MAIN,
        &assemble(MAIN, |a| {
            a.mov(byte_ptr(rdi), 2).unwrap();
            let mut done = a.create_label();
            a.jmp(done).unwrap();
            a.set_label(&mut done).unwrap();
            a.hlt().unwrap();
        }),
    )
    .unwrap();
    let mut cpu = vm.new_vcpu();
    for pc in [TARGET, TARGET + 0x100, MAIN] {
        cpu.set_reg(Reg::Rip, pc);
        cpu.set_reg(Reg::Rdi, TARGET + 0xf00);
        run_to_hlt(&vm, &mut cpu);
    }
    let before = vm.cache.misses();
    let epoch = vm.cache.epoch();
    for offset in [6, 0xf00] {
        cpu.set_reg(Reg::Rdi, TARGET + offset);
        for _ in 0..10 {
            for pc in [MAIN, TARGET, TARGET + 0x100] {
                cpu.set_reg(Reg::Rip, pc);
                run_to_hlt(&vm, &mut cpu);
            }
        }
    }
    assert_eq!(
        vm.cache.misses(),
        before,
        "data writes must not re-lift code"
    );
    assert_eq!(vm.cache.epoch(), epoch);
    assert!(vm.mem.is_code_page(TARGET >> 12));
    cpu.set_reg(Reg::Rdi, TARGET + 1);
    cpu.set_reg(Reg::Rip, MAIN);
    run_to_hlt(&vm, &mut cpu);
    assert!(vm
        .cache
        .get(BlockKey::new(TARGET, CpuMode::Long64))
        .is_none());
    assert!(vm
        .cache
        .get(BlockKey::new(TARGET + 0x100, CpuMode::Long64))
        .is_some());
    assert!(vm.mem.is_code_page(TARGET >> 12));
    let before = vm.cache.misses();
    cpu.set_reg(Reg::Rip, TARGET + 0x100);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(vm.cache.misses(), before);
    cpu.set_reg(Reg::Rip, TARGET);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(cpu.reg(Reg::Rax), 2);
    assert_eq!(vm.cache.misses(), before + 1);
}
both_backends!(
    byte_precise_code_page,
    data_bytes_preserve_cached_code_interp,
    data_bytes_preserve_cached_code_jit
);

#[test]
fn concurrent_data_store_and_same_page_publication_preserve_code() {
    use std::sync::Barrier;
    use x86jit_core::{BlockKey, CpuMode};
    let vm = new_vm(Box::new(InterpreterBackend));
    for i in 0..100u64 {
        vm.write_bytes(TARGET + i * 16, &[0xb8, i as u8, 0, 0, 0, 0xf4])
            .unwrap();
    }
    vm.write_bytes(
        MAIN,
        &assemble(MAIN, |a| {
            let mut top = a.create_label();
            a.mov(ecx, 10000).unwrap();
            a.set_label(&mut top).unwrap();
            a.mov(byte_ptr(TARGET + 0xf00), 42).unwrap();
            a.dec(ecx).unwrap();
            a.jnz(top).unwrap();
            a.hlt().unwrap();
        }),
    )
    .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, TARGET);
    run_to_hlt(&vm, &mut cpu);
    let start = Barrier::new(2);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut writer = vm.new_vcpu();
            writer.set_reg(Reg::Rip, MAIN);
            start.wait();
            run_to_hlt(&vm, &mut writer);
        });
        start.wait();
        for i in 1..100u64 {
            cpu.set_reg(Reg::Rip, TARGET + i * 16);
            run_to_hlt(&vm, &mut cpu);
            assert_eq!(cpu.reg(Reg::Rax), i);
        }
    });
    assert!(vm.mem.is_code_page(TARGET >> 12));
    for i in 0..100u64 {
        assert!(vm
            .cache
            .get(BlockKey::new(TARGET + i * 16, CpuMode::Long64))
            .is_some());
    }
    let before = vm.cache.misses();
    for i in 0..100u64 {
        cpu.set_reg(Reg::Rip, TARGET + i * 16);
        run_to_hlt(&vm, &mut cpu);
        assert_eq!(cpu.reg(Reg::Rax), i);
    }
    assert_eq!(vm.cache.misses(), before);
}

#[cfg(unix)]
#[test]
fn alias_data_offsets_preserve_neighboring_translations() {
    use x86jit_core::{BlockKey, CpuMode};
    for jit in [false, true] {
        let backend: Box<dyn Backend> = if jit {
            Box::new(JitBackend::new())
        } else {
            Box::new(InterpreterBackend)
        };
        let vm = alias_vm(backend);
        vm.mem.declare_code_alias(0x4000, 0x18000, 0x4000).unwrap();
        for pc in [0x4000, 0x4100] {
            vm.write_bytes(pc, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
        }
        vm.write_bytes(MAIN, &[0xf4]).unwrap();
        let mut cpu = vm.new_vcpu();
        for pc in [0x4000, 0x4100, MAIN] {
            cpu.set_reg(Reg::Rip, pc);
            run_to_hlt(&vm, &mut cpu);
        }
        let before = vm.cache.misses();
        vm.write_bytes(0x18f00, &[42]).unwrap();
        for pc in [MAIN, 0x4000, 0x4100] {
            cpu.set_reg(Reg::Rip, pc);
            run_to_hlt(&vm, &mut cpu);
        }
        assert_eq!(vm.cache.misses(), before);
        vm.write_bytes(0x18001, &[2]).unwrap();
        cpu.set_reg(Reg::Rip, MAIN);
        run_to_hlt(&vm, &mut cpu);
        assert!(vm
            .cache
            .get(BlockKey::new(0x4000, CpuMode::Long64))
            .is_none());
        assert!(vm
            .cache
            .get(BlockKey::new(0x4100, CpuMode::Long64))
            .is_some());
        assert!(vm.mem.is_code_page(4) && vm.mem.is_code_page(0x18));
        cpu.set_reg(Reg::Rip, 0x4000);
        run_to_hlt(&vm, &mut cpu);
        assert_eq!(cpu.reg(Reg::Rax), 2);
    }
}

fn crossing_block_byte_precision(backend: Box<dyn Backend>) {
    use x86jit_core::{BlockKey, CpuMode};
    let vm = new_vm(backend);
    vm.write_bytes(0x2ffd, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
    vm.write_bytes(0x3100, &[0xf4]).unwrap();
    vm.write_bytes(MAIN, &[0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    for pc in [0x2ffd, 0x3100, MAIN] {
        cpu.set_reg(Reg::Rip, pc);
        run_to_hlt(&vm, &mut cpu);
    }
    vm.write_bytes(0x2fff, &[2, 0]).unwrap();
    cpu.set_reg(Reg::Rip, MAIN);
    run_to_hlt(&vm, &mut cpu);
    assert!(vm
        .cache
        .get(BlockKey::new(0x2ffd, CpuMode::Long64))
        .is_none());
    assert!(vm
        .cache
        .get(BlockKey::new(0x3100, CpuMode::Long64))
        .is_some());
    assert!(vm.mem.is_code_page(3));
    cpu.set_reg(Reg::Rip, 0x2ffd);
    run_to_hlt(&vm, &mut cpu);
    assert_eq!(cpu.reg(Reg::Rax), 0x201);
}
both_backends!(
    crossing_block_byte_precision,
    crossing_block_bytes_interp,
    crossing_block_bytes_jit
);

// Pause after decoding, before the cache owns any span of the new unit.
struct PausedPublicationBackend {
    jit: JitBackend,
    regions: bool,
    pause: std::sync::Mutex<
        Option<(
            std::sync::mpsc::SyncSender<()>,
            std::sync::mpsc::Receiver<()>,
        )>,
    >,
}
impl PausedPublicationBackend {
    fn pause(&self, pc: u64) {
        if pc == TARGET + 0x100 {
            if let Some((entered, resume)) = self.pause.lock().unwrap().take() {
                entered.send(()).unwrap();
                resume.recv().unwrap();
            }
        }
    }
}
impl Backend for PausedPublicationBackend {
    fn code_lifetime(&self) -> Option<std::sync::Arc<x86jit_core::code_lifetime::CodeLifetime>> {
        self.jit.code_lifetime()
    }

    fn materialize(
        &self,
        ir: &x86jit_core::IrBlock,
        consistency: x86jit_core::MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> x86jit_core::CachedBlock {
        self.pause(ir.guest_start);
        self.jit.materialize(ir, consistency, mmio, guest_base)
    }
    fn region_caps(&self) -> Option<x86jit_core::RegionCaps> {
        self.regions.then_some(x86jit_core::RegionCaps {
            max_blocks: 8,
            max_icount: 128,
        })
    }
    fn materialize_region(
        &self,
        region: &x86jit_core::IrRegion,
        consistency: x86jit_core::MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> x86jit_core::CachedBlock {
        self.pause(region.entry);
        self.jit
            .materialize_region(region, consistency, mmio, guest_base)
    }
    fn invalidate_links(&self) {
        self.jit.invalidate_links();
    }
}

fn victim_free_drain_rejects_first_publication(regions: bool) {
    use std::sync::{mpsc, Arc};
    use std::time::Duration;
    use x86jit_core::{BlockKey, CpuMode};
    let (entered_tx, entered_rx) = mpsc::sync_channel(0);
    let (resume_tx, resume_rx) = mpsc::sync_channel(0);
    let vm = new_vm(Box::new(PausedPublicationBackend {
        jit: JitBackend::new(),
        regions,
        pause: std::sync::Mutex::new(Some((entered_tx, resume_rx))),
    }));
    vm.write_bytes(TARGET, &[0xf4]).unwrap();
    vm.write_bytes(MAIN, &[0xf4]).unwrap();
    let entry = TARGET + 0x100;
    let successor = TARGET + 0x180;
    if regions {
        vm.write_bytes(
            entry,
            &assemble(entry, |a| {
                a.jmp(successor).unwrap();
            }),
        )
        .unwrap();
        vm.write_bytes(
            successor,
            &assemble(successor, |a| {
                a.mov(eax, 1).unwrap();
                a.dec(ecx).unwrap();
                a.jnz(entry).unwrap();
                a.hlt().unwrap();
            }),
        )
        .unwrap();
    } else {
        vm.write_bytes(entry, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
    }
    let mut drain = vm.new_vcpu();
    drain.set_reg(Reg::Rip, TARGET);
    run_to_hlt(&vm, &mut drain);
    let vm = Arc::new(vm);
    let runner_vm = vm.clone();
    let worker = std::thread::spawn(move || {
        let mut cpu = runner_vm.new_vcpu();
        cpu.set_reg(Reg::Rip, entry);
        cpu.set_reg(Reg::Rcx, 2);
        let exit = cpu.run(&runner_vm, Some(100));
        (exit, cpu.reg(Reg::Rax))
    });
    entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(vm
        .cache
        .get(BlockKey::new(entry, CpuMode::Long64))
        .is_none());
    assert!(vm
        .cache
        .get(BlockKey::new(successor, CpuMode::Long64))
        .is_none());
    let epoch = vm.cache.epoch();
    vm.write_bytes(if regions { successor + 1 } else { entry + 1 }, &[2])
        .unwrap();
    drain.set_reg(Reg::Rip, MAIN);
    run_to_hlt(&vm, &mut drain);
    assert!(!vm.mem.has_dirty_code());
    assert_eq!(
        vm.cache.epoch(),
        epoch,
        "surviving code keeps dispatch predictors"
    );
    assert!(vm
        .cache
        .get(BlockKey::new(TARGET, CpuMode::Long64))
        .is_some());
    resume_tx.send(()).unwrap();
    let (exit, value) = worker.join().unwrap();
    assert!(matches!(exit, Exit::Hlt), "{exit:?}");
    assert_eq!(value, 2, "stale first publication must be retried");
    if regions {
        assert!(vm.cache.regions() > 0, "exercise region publication");
    }
}
#[test]
fn victim_free_drain_rejects_first_block_publication() {
    victim_free_drain_rejects_first_publication(false);
}
#[test]
fn victim_free_drain_rejects_first_region_publication() {
    victim_free_drain_rejects_first_publication(true);
}

fn accumulated_code_writes(backend: Box<dyn Backend>) {
    use x86jit_core::{BlockKey, CpuMode};
    let vm = new_vm(backend);
    let entries = [TARGET, TARGET + 0x100, TARGET + 0x200];
    vm.write_bytes(MAIN, &[0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    for pc in entries {
        vm.write_bytes(pc, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
        cpu.set_reg(Reg::Rip, pc);
        run_to_hlt(&vm, &mut cpu);
    }
    // Overlapping writes and disjoint spans accumulate before a single drain.
    vm.write_bytes(TARGET + 1, &[2, 0]).unwrap();
    vm.write_bytes(TARGET + 2, &[0]).unwrap();
    vm.write_bytes(TARGET + 0x101, &[3]).unwrap();
    vm.write_bytes(TARGET + 0xf00, &[9]).unwrap();
    cpu.set_reg(Reg::Rip, MAIN);
    run_to_hlt(&vm, &mut cpu);
    for pc in &entries[..2] {
        assert!(vm.cache.get(BlockKey::new(*pc, CpuMode::Long64)).is_none());
    }
    assert!(vm
        .cache
        .get(BlockKey::new(entries[2], CpuMode::Long64))
        .is_some());
    assert!(vm.mem.is_code_page(TARGET >> 12));
    for (pc, expected) in entries.into_iter().zip([2, 3, 1]) {
        cpu.set_reg(Reg::Rip, pc);
        run_to_hlt(&vm, &mut cpu);
        assert_eq!(cpu.reg(Reg::Rax), expected);
    }
}
both_backends!(
    accumulated_code_writes,
    accumulated_code_writes_interp,
    accumulated_code_writes_jit
);

/// Race the generated store's watermark gate against first translation of each
/// target page. The writer stays compiled; every target starts out untagged.
#[test]
fn compiled_writer_races_first_source_tag() {
    const PAGES: u64 = 128;
    let size = 0x10_0000;
    let mut vm = Vm::with_backend(VmConfig::flat(size), Box::new(JitBackend::new()));
    vm.map(0, size as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.write_bytes(MAIN, &[0xc6, 0x07, 2, 0xf4]).unwrap(); // mov byte [rdi],2; hlt
    let mut writer = vm.new_vcpu();
    writer.set_reg(Reg::Rdi, 0xf0000);
    writer.set_reg(Reg::Rip, MAIN);
    run_to_hlt(&vm, &mut writer);
    let mut reader = vm.new_vcpu();
    for page in 0..PAGES {
        let target = 0x2_0000 + page * 0x1000;
        vm.write_bytes(target, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
        assert!(!vm.mem.is_code_page(target >> 12));
        writer.set_reg(Reg::Rdi, target + 1);
        writer.set_reg(Reg::Rip, MAIN);
        std::thread::scope(|scope| {
            let write = scope.spawn(|| run_to_hlt(&vm, &mut writer));
            reader.set_reg(Reg::Rip, target);
            run_to_hlt(&vm, &mut reader);
            write.join().unwrap();
        });
        // Whether the initial read won or lost the race, the completed write
        // must prevent permanent reuse of a stale source snapshot.
        reader.set_reg(Reg::Rip, target);
        run_to_hlt(&vm, &mut reader);
        assert_eq!(reader.reg(Reg::Rax), 2, "target page {page}");
    }
}
