//! Controlled shared-hit component costs, not additive production timings.
//! Arguments: vcpus iterations thunks (defaults 1, 1000000, 1024). Thread count is configurable.
//! Subtract the loop baseline; compare lookup/locked lookup to isolate reader
//! contention and compiled/Arc clones to isolate refcount traffic. Production
//! compiled hits have no Arc clone. No compilation occurs inside measurements.
use std::collections::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault};
#[path = "../../x86jit-core/src/cache_hash.rs"]
mod cache_hash;
use cache_hash::BlockHasher;
use x86jit_core::time_stats;
#[allow(dead_code)]
#[path = "../../x86jit-core/src/cache_shards.rs"]
mod cache_shards;
use std::hint::black_box;
use std::sync::{Arc, Barrier, RwLock};
use std::time::Instant;
use x86jit_core::{BlockKey, CachedBlock, CompiledPtr, CpuMode, IrBlock, TranslationCache};
#[path = "dispatch_support/usage.rs"]
mod usage;

fn measure(
    n: usize,
    iterations: u64,
    name: &str,
    op: impl Fn(u64, &mut [(u64, usize); 1024]) + Sync,
) {
    let ready = Barrier::new(n + 1);
    let go = Barrier::new(n + 1);
    let done = Barrier::new(n + 1);
    std::thread::scope(|scope| {
        for _ in 0..n {
            let (ready, go, done, op) = (&ready, &go, &done, &op);
            scope.spawn(move || {
                let mut slots = [(0, 0); 1024];
                ready.wait();
                go.wait();
                for i in 0..iterations {
                    op(black_box(i), &mut slots);
                }
                done.wait();
            });
        }
        ready.wait();
        let before = usage::retired();
        let start = Instant::now();
        go.wait();
        done.wait();
        let ns = start.elapsed().as_nanos();
        let instructions = usage::retired() - before;
        let count = iterations * n as u64;
        println!(
            "{name},{n},{count},{instructions},{:.3},{:.3}",
            instructions as f64 / count as f64,
            ns as f64 / count as f64
        );
    });
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1);
    let iterations: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1_000_000);
    assert!((1..=1024).contains(&n));
    assert!((1..=100_000_000).contains(&iterations));
    let thunks: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1024);
    assert!(matches!(thunks, 1024 | 4096));
    let mask = thunks - 1;
    let keys: Vec<_> = (0..thunks)
        .map(|i| BlockKey::new(0x10000 + 32 * i as u64, CpuMode::Long64))
        .collect();
    let entry = CompiledPtr::new(std::ptr::dangling()); // Never executed.
    let compiled = CachedBlock::Compiled { entry };
    let interpreted = CachedBlock::Interpreted(Arc::new(IrBlock {
        legacy_context: None,
        mode: CpuMode::Long64,
        guest_start: 0,
        ops: vec![],
        temp_count: 0,
        guest_len: 1,
        icount: 0,
        link_exit: false,
    }));
    let map: HashMap<_, _, BuildHasherDefault<BlockHasher>> =
        keys.iter().map(|&key| (key, compiled.clone())).collect();
    let locked = RwLock::new(map.clone());
    let sharded = cache_shards::DispatchMap::new();
    {
        let mut write = sharded.write();
        for &key in &keys {
            write.insert(key, compiled.clone());
        }
    }
    let lock = RwLock::new(());
    let cache = TranslationCache::new();
    for &key in &keys {
        cache.insert(key, compiled.clone(), vec![(key.guest_addr, 1)], |_| {});
    }
    println!("component,vcpus,operations,host_instructions,instructions_per_op,ns_per_op");
    measure(n, iterations, "baseline", |i, _| {
        black_box(keys[(i as usize) & mask]);
    });
    measure(n, iterations, "reader_lock", |_, _| {
        drop(black_box(lock.read().unwrap()));
    });
    measure(n, iterations, "hash_key", |i, _| {
        black_box(map.hasher().hash_one(keys[(i as usize) & mask]));
    });
    measure(n, iterations, "lookup_compiled", |i, _| {
        black_box(map.get(&keys[(i as usize) & mask]).cloned());
    });
    measure(n, iterations, "locked_lookup_compiled", |i, _| {
        black_box(
            locked
                .read()
                .unwrap()
                .get(&keys[(i as usize) & mask])
                .cloned(),
        );
    });
    measure(n, iterations, "sharded_locked_lookup_compiled", |i, _| {
        black_box(sharded.get(&keys[(i as usize) & mask]));
    });
    measure(n, iterations, "clone_compiled", |_, _| {
        black_box(black_box(&compiled).clone());
    });
    measure(n, iterations, "clone_interpreted_arc", |_, _| {
        black_box(black_box(&interpreted).clone());
    });
    // Public get also updates a shared hit atomic; dispatcher batches that update
    // once per run, so this deliberately reports its additional traffic.
    measure(n, iterations, "public_cache_get", |i, _| {
        black_box(cache.get(keys[(i as usize) & mask]));
    });
    let entry_address = entry.0 as usize;
    measure(n, iterations, "private_refill", |i, slots| {
        let rip = keys[(i as usize) & mask].guest_addr;
        let index = (rip.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 54) as usize;
        black_box(slots)[index] = (rip, entry_address);
    });
}
