//! Allocation diagnostics for the pinned cranelift-jit 0.115.1 (§8.2).

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub(crate) struct CodeCounters {
    units: AtomicU64,
    emitted: AtomicU64,
    reserved: AtomicU64,
}

impl CodeCounters {
    pub(crate) fn snapshot(&self) -> x86jit_core::BackendMemoryStats {
        let mut stats = x86jit_core::BackendMemoryStats::default();
        stats.compiled_units_live = self.units.load(Ordering::Relaxed);
        stats.compiled_units_total = stats.compiled_units_live;
        stats.code_bytes_emitted = self.emitted.load(Ordering::Relaxed);
        stats.code_bytes_reserved = self.reserved.load(Ordering::Relaxed);
        stats
    }
}

/// Mirrors memory.rs::Memory::allocate/finish_current in cranelift-jit 0.115.1.
/// That private allocator has no observation hook. Keep this model pinned to it:
/// code allocations are page-rounded, aligned definitions share the current
/// tail, and each finalize closes that tail. This backend declares no data and
/// disables PIC/hotswap, so all executable allocations are function definitions.
#[derive(Default)]
pub(crate) struct PendingAllocations {
    len: usize,
    position: usize,
    reserved: u64,
    emitted: u64,
    units: u64,
}

impl PendingAllocations {
    pub(crate) fn define(&mut self, size: usize, align: usize) {
        self.position = self.position.div_ceil(align) * align;
        if size > self.len.saturating_sub(self.position) {
            let page = region::page::size();
            self.len = size.div_ceil(page) * page;
            self.position = 0;
            self.reserved += self.len as u64;
        }
        self.position += size;
        self.emitted += size as u64;
        self.units += 1;
    }

    pub(crate) fn finalize(&mut self, counters: &CodeCounters, success: bool) {
        counters
            .reserved
            .fetch_add(self.reserved, Ordering::Relaxed);
        if success {
            counters.emitted.fetch_add(self.emitted, Ordering::Relaxed);
            counters.units.fetch_add(self.units, Ordering::Relaxed);
        }
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tails_alignment_and_finalize_boundaries() {
        let counters = CodeCounters::default();
        let mut pending = PendingAllocations::default();
        let page = region::page::size();
        pending.define(40, 16);
        pending.define(40, 16);
        assert_eq!(counters.snapshot().code_bytes_reserved, 0);
        pending.finalize(&counters, true);
        assert_eq!(counters.snapshot().code_bytes_reserved, page as u64);
        assert_eq!(counters.snapshot().code_bytes_emitted, 80);
        pending.define(page + 1, 16);
        pending.finalize(&counters, true);
        assert_eq!(counters.snapshot().code_bytes_reserved, 3 * page as u64);
        assert_eq!(counters.snapshot().compiled_units_live, 3);
        pending.define(1, 16);
        pending.finalize(&counters, false);
        assert_eq!(counters.snapshot().code_bytes_reserved, 4 * page as u64);
        assert_eq!(counters.snapshot().compiled_units_total, 3);
    }
}
