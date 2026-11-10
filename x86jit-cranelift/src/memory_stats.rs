//! Exact provider allocation diagnostics (spec §8.2; decision-17).
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub(crate) struct RegionCounters {
    pub regions: AtomicU64,
    pub views: AtomicU64,
    pub reserved: AtomicU64,
    pub committed: AtomicU64,
    pub used: AtomicU64,
    pub free: AtomicU64,
    pub largest: AtomicU64,
    pub failures: AtomicU64,
    pub allocation_metadata: AtomicU64,
}

#[derive(Default)]
pub(crate) struct CodeCounters {
    units: AtomicU64,
    emitted: AtomicU64,
    providers: Mutex<Vec<Arc<RegionCounters>>>,
}

impl CodeCounters {
    pub(crate) fn register(&self) -> Arc<RegionCounters> {
        let counters = Arc::new(RegionCounters::default());
        self.providers.lock().unwrap().push(counters.clone());
        counters
    }

    pub(crate) fn snapshot(&self) -> x86jit_core::BackendMemoryStats {
        let mut stats = x86jit_core::BackendMemoryStats::default();
        stats.compiled_units_live = self.units.load(Ordering::Relaxed);
        stats.compiled_units_total = stats.compiled_units_live;
        stats.code_bytes_emitted = self.emitted.load(Ordering::Relaxed);
        for c in self.providers.lock().unwrap().iter() {
            stats.code_regions += c.regions.load(Ordering::Relaxed);
            stats.code_mapping_views += c.views.load(Ordering::Relaxed);
            stats.code_bytes_reserved += c.reserved.load(Ordering::Relaxed);
            stats.code_bytes_committed += c.committed.load(Ordering::Relaxed);
            stats.code_bytes_used += c.used.load(Ordering::Relaxed);
            stats.code_bytes_free += c.free.load(Ordering::Relaxed);
            stats.code_largest_free_extent = stats
                .code_largest_free_extent
                .max(c.largest.load(Ordering::Relaxed));
            stats.code_allocation_failures += c.failures.load(Ordering::Relaxed);
            stats.code_allocation_metadata_bytes += c.allocation_metadata.load(Ordering::Relaxed);
        }
        // All logical macOS regions refer to one process-wide mapping. This is
        // the number of distinct code views touched by this backend, not a sum
        // over its MXCSR modules. The global reservation also serves other VMs.
        if cfg!(target_os = "macos") && stats.code_regions != 0 {
            stats.code_mapping_views = 1;
            stats.code_process_reservation_bytes = 1 << 40;
        }
        stats
    }
}

#[derive(Default)]
pub(crate) struct PendingAllocations {
    emitted: u64,
    units: u64,
}
impl PendingAllocations {
    pub(crate) fn define(&mut self, size: usize) {
        self.emitted += size as u64;
        self.units += 1;
    }
    pub(crate) fn finalize(&mut self, counters: &CodeCounters, success: bool) {
        if success {
            counters.emitted.fetch_add(self.emitted, Ordering::Relaxed);
            counters.units.fetch_add(self.units, Ordering::Relaxed);
        }
        *self = Self::default();
    }
}
