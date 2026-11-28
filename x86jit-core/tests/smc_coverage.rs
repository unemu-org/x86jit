//! Allocation witness for stale code tags after abandoning a tier-up (§10).
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use x86jit_core::{BlockKey, CpuMode, Memory, MemoryModel, TranslationCache};

struct CountingAllocator;
thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

// SAFETY: every operation forwards its allocator contract unchanged to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| {
            if let Some(n) = count.get() {
                count.set(Some(n + 1));
            }
        });
        // SAFETY: forward the allocator contract unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forward the allocator contract unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.with(|count| {
            if let Some(n) = count.get() {
                count.set(Some(n + 1));
            }
        });
        // SAFETY: forward the allocator contract unchanged.
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn clearing_abandoned_unaliased_code_tags_allocates_nothing() {
    let mem = Memory::new(MemoryModel::Flat { size: 0x4000 });
    let cache = TranslationCache::new();
    let key = BlockKey::new(0x1000, CpuMode::Long64);
    assert!(cache.try_begin_tier_up(key));
    mem.mark_code(0x3000, 4);
    cache.end_tier_up(key);
    assert!(mem.is_code_page(3));
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let victims = cache.invalidate_overlapping(0x3000, 0x3001, || mem.clear_code_page(3));
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    assert!(victims.is_empty());
    assert!(!mem.is_code_page(3));
    assert_eq!(allocations, 0);
}

/// The Real16 alias's self-modifying loops of `real16_loops` (`smc-next-*`,
/// `smc-rmw-64`): a store patches the next instruction's immediate, so every
/// iteration drains the write, drops both overlapping blocks and relifts
/// them. Once warm, an iteration reuses the dropped blocks' allocations and
/// the drain's buffers, and allocates nothing (xj-to9.71).
#[test]
fn steady_real16_alias_smc_windows_allocate_nothing() {
    use x86jit_core::{Prot, Reg, RegionKind, StepOptions, Vm, VmConfig};
    // mov byte [0x7c06],0x42 / add byte [0x7c06],1; mov al,0; jmp 0x7c00
    let next = [0xc6, 6, 6, 0x7c, 0x42, 0xb0, 0, 0xeb, 0xf7];
    let rmw = [0x80, 6, 6, 0x7c, 1, 0xb0, 0, 0xeb, 0xf7];
    for (code, slice) in [(next, 1), (next, 64), (rmw, 64)] {
        let mut vm = Vm::new(VmConfig::flat(0x10_0000));
        vm.set_cpu_mode(CpuMode::Real16);
        vm.map(0, 0xa_0000, Prot::RWX, RegionKind::Ram).unwrap();
        let mut cpu = vm.new_vcpu();
        for (reg, value) in [(Reg::Cs, 0), (Reg::Rip, 0x7c00), (Reg::Ds, 0), (Reg::Ss, 0)] {
            cpu.set_reg(reg, value);
        }
        vm.write_bytes(0x7c00, &code).unwrap();
        // Three steps per iteration; run whole iterations per phase.
        let run = |cpu: &mut x86jit_core::Vcpu, iterations: u64| {
            let mut done = 0;
            while done < 3 * iterations {
                let r = cpu.run_steps(&vm, slice, StepOptions::default()).unwrap();
                assert_eq!(r.steps(), slice);
                done += slice;
            }
        };
        run(&mut cpu, 64);
        let misses = vm.cache.misses();
        ALLOCATIONS.with(|count| count.set(Some(0)));
        run(&mut cpu, 256);
        let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
        // Each iteration relifts the block at 0x7c00 and the one at the patch.
        assert!(
            vm.cache.misses() >= misses + 2 * 256,
            "every window relifts"
        );
        assert_eq!(allocations, 0, "code {code:02x?}, {slice} steps per call");
    }
}
