//! Measure retained source-index allocations for 31,000 block/region keys.
//! Run this same example against both revisions; materialized-cache storage is
//! subtracted using an independently allocated map with identical keys/values.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashMap;
use x86jit_core::{BlockKey, CachedBlock, CompiledPtr, CpuMode, TranslationCache};

struct CountingAllocator;
thread_local! {
    static LIVE_BYTES: Cell<Option<isize>> = const { Cell::new(None) };
}

fn account(bytes: isize) {
    LIVE_BYTES.with(|live| {
        if let Some(n) = live.get() {
            live.set(Some(n + bytes));
        }
    });
}

// SAFETY: every operation forwards its allocator contract unchanged to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forward the allocator contract unchanged.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            account(layout.size() as isize);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        account(-(layout.size() as isize));
        // SAFETY: forward the allocator contract unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: forward the allocator contract unchanged.
        let next = unsafe { System.realloc(ptr, layout, size) };
        if !next.is_null() {
            account(size as isize - layout.size() as isize);
        }
        next
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn key(i: u64) -> BlockKey {
    BlockKey::new(
        (i / 32 + 1) * 4096 + (i % 16) * 128 + u64::from(i % 32 >= 16) * 4,
        CpuMode::Long64,
    )
}

fn block() -> CachedBlock {
    CachedBlock::Compiled {
        entry: CompiledPtr(std::ptr::null()),
    }
}

fn main() {
    LIVE_BYTES.with(|live| live.set(Some(0)));
    let cache = TranslationCache::new();
    for i in 0..31_000 {
        let page = (i / 32 + 1) * 4096;
        let slot = i % 32;
        let spans = if slot < 16 {
            vec![(page + slot * 128, 64)]
        } else {
            (0..4)
                .map(|j| {
                    (
                        page + ((slot - 16 + j) % 16) * 128 + u64::from(j == 0) * 4,
                        if j == 0 { 60 } else { 64 },
                    )
                })
                .collect()
        };
        cache.insert(key(i), block(), spans, |_| {});
    }
    let cache_bytes = LIVE_BYTES.with(|live| live.replace(None).unwrap());

    LIVE_BYTES.with(|live| live.set(Some(0)));
    let mut materialized = HashMap::new();
    for i in 0..31_000 {
        materialized.insert(key(i), block());
    }
    let materialized_bytes = LIVE_BYTES.with(|live| live.replace(None).unwrap());
    println!("keys,keys_per_page,cache_bytes,materialized_bytes,index_bytes");
    println!(
        "31000,32,{cache_bytes},{materialized_bytes},{}",
        cache_bytes - materialized_bytes
    );
    std::hint::black_box((&cache, &materialized));
}
