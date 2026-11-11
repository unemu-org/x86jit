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
