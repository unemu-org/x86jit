//! Compiled-unit retirement and execution grace (spec §9.1; decision-17).
//!
//! This is a cold ownership registry, separate from the signal-reader protocol.
//! Retirement does not remove code-map ranges or authorize storage reuse.

use crate::cache::{CachedBlock, CompiledPtr};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Identity of one allocated unit. Generations never wrap or repeat, even
/// across backends. Addresses alone cannot identify a unit after reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeUnit {
    pub generation: u64,
    pub host_start: usize,
    pub host_len: u32,
}

impl CodeUnit {
    /// Reserve an identity before an allocation can be published. Failed and
    /// unpublished allocations also consume identities; generations never repeat.
    pub fn new(host_start: usize, host_len: u32) -> Self {
        let generation = NEXT_GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                old.checked_add(1)
            })
            .expect("code allocation generations exhausted");
        Self {
            generation,
            host_start,
            host_len,
        }
    }

    pub fn entry(self) -> CompiledPtr {
        CompiledPtr::with_generation(self.host_start as *const u8, self.generation)
    }
}

/// An unlinked unit awaiting execution grace, then code-map reader grace.
#[derive(Clone, Copy, Debug)]
pub struct RetiredCodeUnit {
    unit: CodeUnit,
    epoch: u64,
}

impl RetiredCodeUnit {
    pub fn unit(&self) -> CodeUnit {
        self.unit
    }
}

struct PublishedUnit {
    unit: CodeUnit,
    publications: usize,
}

#[derive(Default)]
#[allow(clippy::vec_box)] // Descriptor addresses are loaded by generated code.
struct State {
    live: HashMap<usize, PublishedUnit>,
    retired: Vec<RetiredCodeUnit>,
    retired_by_generation: HashMap<u64, usize>,
    retirement_cursor: usize,
    slots: Vec<Weak<AtomicU64>>,
    participants: Vec<Weak<AtomicU64>>,
    descriptors: Vec<Box<[u64; 2]>>,
    retired_descriptors: Vec<(u64, Box<[u64; 2]>)>,
}

/// Shared by a backend and every cache publishing its units. No registry lock is
/// held while executing code, invoking an observer or entering a signal handler.
pub struct CodeLifetime {
    epoch: AtomicU64,
    require_generation: bool,
    closed: AtomicBool,
    maintenance: Condvar,
    reclamation_workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    state: Mutex<State>,
}

impl Default for CodeLifetime {
    fn default() -> Self {
        Self {
            epoch: AtomicU64::new(1),
            require_generation: false,
            closed: AtomicBool::new(false),
            maintenance: Condvar::new(),
            reclamation_workers: Mutex::default(),
            state: Mutex::default(),
        }
    }
}

impl CodeLifetime {
    /// A domain whose cache rejects address-only entries, including entries
    /// returned by a wrapper that drops the underlying backend's generation.
    pub fn managed() -> Self {
        Self {
            require_generation: true,
            ..Self::default()
        }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }

    pub(crate) fn advance(&self) -> u64 {
        self.epoch
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |old| old.checked_add(1))
            .expect("execution epochs exhausted")
    }

    pub(crate) fn flush_predictions(&self) {
        let mut state = self.state.lock().unwrap();
        self.clear_slots(&mut state);
        self.advance();
    }

    /// Called after finalization and code-map registration, before exposing the
    /// pointer. Physical extents and compiler records remain provider-owned.
    pub fn register(&self, entry: CompiledPtr, host_len: u32) -> CodeUnit {
        assert_eq!(
            entry.generation(),
            0,
            "entry already has an allocation identity"
        );
        let unit = CodeUnit::new(entry.0 as usize, host_len);
        // SAFETY: new reserves a process-wide identity which has never been used.
        unsafe { self.register_unit(unit) };
        unit
    }

    /// Bind a provider allocation identity after finalization and registration.
    ///
    /// # Safety
    /// The identity must be freshly reserved, never previously registered, and
    /// owned by this backend. Its extent must remain allocated through execution
    /// and signal-reader grace and removal of compiler references.
    pub unsafe fn register_unit(&self, unit: CodeUnit) {
        assert_ne!(unit.generation, 0);
        let previous = self.state.lock().unwrap().live.insert(
            unit.host_start,
            PublishedUnit {
                unit,
                publications: 0,
            },
        );
        assert!(
            previous.is_none(),
            "compiled allocation is already registered"
        );
    }

    fn matches(&self, state: &State, entry: CompiledPtr) -> bool {
        if !self.is_open() {
            return false;
        }
        match state.live.get(&(entry.0 as usize)) {
            Some(live) => entry.generation() == live.unit.generation,
            None => !self.require_generation && entry.generation() == 0,
        }
    }

    /// Register the stable slot before generated code can load its address.
    pub fn register_slot(&self, slot: &Arc<AtomicU64>) {
        self.state.lock().unwrap().slots.push(Arc::downgrade(slot));
    }

    fn clear_slots(&self, state: &mut State) {
        let epoch = self.epoch();
        state
            .retired_descriptors
            .extend(state.descriptors.drain(..).map(|d| (epoch, d)));
        if !state.retired_descriptors.is_empty() {
            self.maintenance.notify_all();
        }
        state.slots.retain(|weak| {
            if let Some(slot) = weak.upgrade() {
                slot.store(0, Ordering::SeqCst);
                true
            } else {
                false
            }
        });
    }

    pub(crate) fn publish_if_current(&self, epoch: u64, publish: impl FnOnce()) -> bool {
        let state = self.state.lock().unwrap();
        if self.epoch() != epoch {
            return false;
        }
        // A caller's publication panic must not poison the lifetime registry:
        // cache/backend destruction still needs it to retire owned entries.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(publish));
        drop(state);
        match result {
            Ok(()) => true,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    pub(crate) fn publish_entry_if_current(
        &self,
        epoch: u64,
        entry: CompiledPtr,
        publish: impl FnOnce(),
    ) -> bool {
        let state = self.state.lock().unwrap();
        if self.epoch() != epoch || !self.matches(&state, entry) {
            return false;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(publish));
        drop(state);
        match result {
            Ok(()) => true,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    fn retire(&self, state: &mut State, start: usize) {
        self.retire_many(state, [start]);
    }

    fn retire_many(&self, state: &mut State, starts: impl IntoIterator<Item = usize>) {
        let units: Vec<_> = starts
            .into_iter()
            .filter_map(|start| state.live.remove(&start))
            .collect();
        if units.is_empty() {
            return;
        }
        // Cache publication locks exclude pointer resolution/publication while
        // shared links are cleared. Old execution participants may still follow
        // a previously loaded pointer; their epoch pins this generation.
        self.clear_slots(state);
        let epoch = self.advance();
        self.maintenance.notify_all();
        for unit in units {
            let index = state.retired.len();
            state
                .retired_by_generation
                .insert(unit.unit.generation, index);
            state.retired.push(RetiredCodeUnit {
                unit: unit.unit,
                epoch,
            });
        }
    }

    pub(crate) fn remove_publications<'a>(&self, blocks: impl Iterator<Item = &'a CachedBlock>) {
        let mut state = self.state.lock().unwrap();
        let mut retiring = Vec::new();
        for block in blocks {
            if let CachedBlock::Compiled { entry } = block {
                let start = entry.0 as usize;
                if let Some(unit) = state
                    .live
                    .get_mut(&start)
                    .filter(|u| u.unit.generation == entry.generation())
                {
                    assert_ne!(unit.publications, 0);
                    unit.publications -= 1;
                    if unit.publications == 0 {
                        retiring.push(start);
                    }
                }
            }
        }
        self.retire_many(&mut state, retiring);
    }

    pub(crate) fn replace(&self, old: Option<&CachedBlock>, new: Option<&CachedBlock>) -> bool {
        fn entry(block: Option<&CachedBlock>) -> Option<CompiledPtr> {
            match block {
                Some(CachedBlock::Compiled { entry }) => Some(*entry),
                _ => None,
            }
        }
        let old = entry(old);
        let new = entry(new);
        let mut state = self.state.lock().unwrap();
        if new.is_some_and(|entry| !self.matches(&state, entry)) {
            return false;
        }
        if old.map(|e| (e.0, e.generation())) == new.map(|e| (e.0, e.generation())) {
            return true;
        }
        if let Some(unit) = new.and_then(|entry| state.live.get_mut(&(entry.0 as usize))) {
            unit.publications += 1;
        }
        if let Some(entry) = old {
            let start = entry.0 as usize;
            if let Some(unit) = state
                .live
                .get_mut(&start)
                .filter(|u| u.unit.generation == entry.generation())
            {
                assert_ne!(unit.publications, 0);
                unit.publications -= 1;
                if unit.publications == 0 {
                    self.retire(&mut state, start);
                }
            }
        }
        true
    }

    /// Discard an unpublished completion. A cloned entry already published under
    /// another cache key retains its publication ownership.
    pub fn discard(&self, block: &CachedBlock) {
        if let CachedBlock::Compiled { entry } = block {
            let mut state = self.state.lock().unwrap();
            let start = entry.0 as usize;
            if state
                .live
                .get(&start)
                .is_some_and(|u| u.unit.generation == entry.generation() && u.publications == 0)
            {
                self.retire(&mut state, start);
            }
        }
    }

    /// [`Self::discard`] for several unpublished units, in one retirement: the
    /// compiled Machine units a vCPU owns (decision-29), which are never
    /// published to a cache.
    pub(crate) fn discard_many(&self, blocks: &[CachedBlock]) {
        if blocks.is_empty() {
            return;
        }
        let mut state = self.state.lock().unwrap();
        let starts: Vec<_> = blocks
            .iter()
            .filter_map(|block| match block {
                CachedBlock::Compiled { entry } => Some(*entry),
                CachedBlock::Interpreted(_) => None,
            })
            .filter(|entry| {
                state
                    .live
                    .get(&(entry.0 as usize))
                    .is_some_and(|u| u.unit.generation == entry.generation() && u.publications == 0)
            })
            .map(|entry| entry.0 as usize)
            .collect();
        self.retire_many(&mut state, starts);
    }

    /// Backend teardown, after compilation has stopped and workers have joined.
    ///
    /// # Safety
    /// Direct ABI callers must stop using the backend at teardown. This closes
    /// the domain before retirement: caches reject future compiled resolution
    /// and publication, and the epoch change clears private predictions.
    pub unsafe fn retire_all(&self) {
        let mut state = self.state.lock().unwrap();
        if !self.closed.swap(true, Ordering::SeqCst) {
            self.clear_slots(&mut state);
            self.advance();
        }
        let starts: Vec<_> = state.live.keys().copied().collect();
        self.retire_many(&mut state, starts);
    }

    /// One predictable cache-resolution check after backend teardown.
    pub fn is_open(&self) -> bool {
        !self.closed.load(Ordering::Acquire)
    }

    pub fn has_pending_reclamation(&self) -> bool {
        let state = self.state.lock().unwrap();
        !state.retired.is_empty() || !state.retired_descriptors.is_empty()
    }

    /// Park without polling while idle; pending grace periods are retried at
    /// `interval`. Retirement and shutdown notify under the same state mutex.
    pub fn wait_for_reclamation(&self, stop: &AtomicBool, interval: std::time::Duration) -> bool {
        let mut state = self.state.lock().unwrap();
        while state.retired.is_empty() && state.retired_descriptors.is_empty() {
            if stop.load(Ordering::Acquire) {
                return false;
            }
            state = self.maintenance.wait(state).unwrap();
        }
        if !stop.load(Ordering::Acquire) {
            let _wait = self.maintenance.wait_timeout(state, interval).unwrap();
        }
        !stop.load(Ordering::Acquire)
    }

    /// Wake a parked collector after setting its stop flag.
    pub fn wake_maintenance(&self) {
        let _state = self.state.lock().unwrap();
        self.maintenance.notify_all();
    }

    /// Retain a deferred backend's worker in the cache domain, including after
    /// public backend replacement. VM shutdown must join it before a host fork.
    pub fn retain_reclamation_worker(&self, worker: std::thread::JoinHandle<()>) {
        self.reclamation_workers.lock().unwrap().push(worker);
    }

    /// Join deferred teardown after grace completes. Pending grace is reported
    /// without blocking on execution participants or signal readers.
    pub fn join_reclamation_workers(&self) -> Result<(), crate::ShutdownError> {
        if self.has_pending_reclamation() {
            return Err(crate::ShutdownError::LiveResources {
                resource: "code reclamation",
                report: crate::ShutdownReport {
                    outstanding_handles: 1,
                    ..Default::default()
                },
            });
        }
        let mut workers = self.reclamation_workers.lock().unwrap();
        let mut panicked = false;
        for worker in workers.drain(..) {
            if let Err(payload) = worker.join() {
                panicked = true;
                std::mem::forget(payload);
            }
        }
        if panicked {
            Err(crate::ShutdownError::WorkerPanicked {
                report: crate::ShutdownReport::default(),
            })
        } else {
            Ok(())
        }
    }

    /// Cold diagnostic/maintenance snapshot. A ticket is not permission to reuse
    /// its extent: signal-reader grace and module-record retirement are separate.
    pub fn retired_units(&self) -> Vec<RetiredCodeUnit> {
        self.state.lock().unwrap().retired.clone()
    }

    /// Bounded, rotating maintenance snapshot. Rotation lets a collector revisit
    /// pending grace periods without pinning its scan to the oldest batch.
    pub fn retirement_candidates(&self, limit: usize) -> Vec<RetiredCodeUnit> {
        let mut state = self.state.lock().unwrap();
        let len = state.retired.len();
        if len == 0 {
            return Vec::new();
        }
        let start = state.retirement_cursor % len;
        let count = limit.min(len);
        let tickets = (0..count)
            .map(|i| state.retired[(start + i) % len])
            .collect();
        state.retirement_cursor = (start + count) % len;
        tickets
    }

    pub fn has_retired_units(&self) -> bool {
        !self.state.lock().unwrap().retired.is_empty()
    }

    /// Consume a completed retirement without retaining a per-unit tombstone.
    ///
    /// # Safety
    /// Execution grace, range removal and its signal-reader grace must be
    /// complete. The caller must have disposed of the compiler references and
    /// either retained the extent or returned it to its owning provider.
    pub unsafe fn finish_retirement(&self, ticket: RetiredCodeUnit) {
        let mut state = self.state.lock().unwrap();
        let index = state
            .retired_by_generation
            .remove(&ticket.unit.generation)
            .expect("retirement belongs to this domain and is not already consumed");
        assert_eq!(state.retired[index].epoch, ticket.epoch);
        assert_eq!(state.retired[index].unit, ticket.unit);
        state.retired.swap_remove(index);
        if let Some(moved) = state.retired.get(index) {
            let generation = moved.unit.generation;
            state.retired_by_generation.insert(generation, index);
        }
    }

    pub fn live_units(&self) -> Vec<CodeUnit> {
        self.state
            .lock()
            .unwrap()
            .live
            .values()
            .map(|u| u.unit)
            .collect()
    }

    /// Has this domain passed the ticket's execution grace period? New entrants
    /// announce a later epoch and cannot prolong it. Idle participants announce 0.
    pub fn execution_complete(&self, ticket: RetiredCodeUnit) -> bool {
        let mut state = self.state.lock().unwrap();
        assert!(state
            .retired_by_generation
            .contains_key(&ticket.unit.generation));
        let mut complete = true;
        state.participants.retain(|weak| {
            if let Some(participant) = weak.upgrade() {
                let epoch = participant.load(Ordering::SeqCst);
                complete &= epoch == 0 || epoch > ticket.epoch;
                true
            } else {
                false
            }
        });
        complete
    }

    /// Cold descriptor allocation; ownership follows the execution domain even
    /// when its cache is destroyed while an older participant is still active.
    pub fn alloc_ibtc_descriptor(&self, target: u64, entry: CompiledPtr) -> u64 {
        let descriptor = Box::new([target, entry.0 as u64]);
        let address = &*descriptor as *const [u64; 2] as u64;
        self.state.lock().unwrap().descriptors.push(descriptor);
        address
    }

    /// Collect immutable descriptors only after slots were cleared and every
    /// participant which could have loaded their pointers has left execution.
    pub fn collect_ibtc_descriptors(&self) {
        let mut state = self.state.lock().unwrap();
        if state.retired_descriptors.is_empty() {
            return;
        }
        state.slots.retain(|slot| slot.strong_count() != 0);
        let oldest = state
            .participants
            .iter()
            .filter_map(Weak::upgrade)
            .map(|p| p.load(Ordering::SeqCst))
            .filter(|e| *e != 0)
            .min();
        state
            .retired_descriptors
            .retain(|(epoch, _)| oldest.is_some_and(|old| old <= *epoch));
    }

    pub fn ibtc_descriptor_count(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.descriptors.len() + state.retired_descriptors.len()
    }

    pub fn unit_at(&self, address: usize) -> Option<CodeUnit> {
        self.state
            .lock()
            .unwrap()
            .live
            .get(&address)
            .map(|u| u.unit)
    }

    /// Gauges of registered units, excluding cumulative compilation statistics.
    pub fn unit_stats(&self) -> (u64, u64, u64, u64) {
        let state = self.state.lock().unwrap();
        (
            state
                .live
                .values()
                .map(|u| u64::from(u.unit.host_len))
                .sum(),
            state
                .retired
                .iter()
                .map(|u| u64::from(u.unit.host_len))
                .sum(),
            state.live.len() as u64,
            state.retired.len() as u64,
        )
    }

    pub fn participant(self: &Arc<Self>) -> ExecutionParticipant {
        let epoch = Arc::new(AtomicU64::new(0));
        let mut state = self.state.lock().unwrap();
        state.participants.retain(|p| p.strong_count() != 0);
        state.participants.push(Arc::downgrade(&epoch));
        ExecutionParticipant {
            domain: self.clone(),
            epoch,
        }
    }
}

/// A thread's execution announcement. Enter before resolving any pointer and
/// retain the guard through chains, helper calls and cold fault decoding. Drop
/// predictions from earlier epochs before resolving again. Do not use in signals.
pub struct ExecutionParticipant {
    domain: Arc<CodeLifetime>,
    epoch: Arc<AtomicU64>,
}

impl ExecutionParticipant {
    // The dispatcher holds exclusive Vcpu access. The stored participant is
    // never entered while this local copy exists, including during unwinding.
    pub(crate) fn for_dispatch(&mut self) -> Self {
        Self {
            domain: self.domain.clone(),
            epoch: self.epoch.clone(),
        }
    }

    pub fn enter(&mut self) -> ExecutionGuard<'_> {
        self.enter_probed(|| {})
    }

    fn enter_probed(&mut self, mut probe: impl FnMut()) -> ExecutionGuard<'_> {
        loop {
            let epoch = self.domain.epoch();
            probe();
            self.epoch.store(epoch, Ordering::SeqCst);
            probe();
            if self.domain.epoch() == epoch {
                return ExecutionGuard {
                    participant: self,
                    epoch,
                };
            }
            self.epoch.store(0, Ordering::SeqCst);
        }
    }
}

#[must_use = "retain participation through compiled calls and cold fault decoding"]
pub struct ExecutionGuard<'a> {
    participant: &'a mut ExecutionParticipant,
    epoch: u64,
}

impl ExecutionGuard<'_> {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Validate a direct ABI entry before calling it. Keep this guard through
    /// execution and cold fault decoding. A stale identity cannot resolve to a
    /// later unit at the same address. Discard predictions on an epoch change.
    pub fn admits(&self, entry: CompiledPtr) -> bool {
        let domain = &self.participant.domain;
        let state = domain.state.lock().unwrap();
        domain.epoch() == self.epoch && domain.matches(&state, entry)
    }
}

impl Drop for ExecutionGuard<'_> {
    fn drop(&mut self) {
        self.participant.epoch.store(0, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{BlockKey, TranslationCache};
    use crate::CpuMode;

    fn compiled(domain: &CodeLifetime, start: usize) -> CachedBlock {
        let entry = CompiledPtr::new(start as *const u8);
        let entry = domain.register(entry, 16).entry();
        CachedBlock::Compiled { entry }
    }

    fn key(start: u64) -> BlockKey {
        BlockKey::new(start, CpuMode::Long64)
    }

    #[test]
    fn current_epoch_cannot_publish_retired_or_reused_target() {
        let domain = Arc::new(CodeLifetime::managed());
        let cache = TranslationCache::with_code_lifetime(Some(domain.clone()));
        let block = compiled(&domain, 0x1000);
        let CachedBlock::Compiled { entry } = block.clone() else {
            unreachable!()
        };
        cache.insert(key(0x2000), block, vec![(0x2000, 16)], |_| {});
        let mut participant = domain.participant();
        let held = participant.enter();
        cache.invalidate_overlapping(0x2000, 0x2010, || {});
        assert_ne!(held.epoch(), cache.epoch());
        let slot = AtomicU64::new(0);
        assert!(!cache.publish_entry_if_current(cache.epoch(), entry, || {
            slot.store(entry.0 as u64, Ordering::Release);
        }));
        let replacement = compiled(&domain, 0x1000);
        assert!(!cache.publish_entry_if_current(cache.epoch(), entry, || {
            slot.store(entry.0 as u64, Ordering::Release);
        }));
        let CachedBlock::Compiled { entry: current } = replacement else {
            unreachable!()
        };
        assert!(cache.publish_entry_if_current(cache.epoch(), current, || {
            slot.store(current.0 as u64, Ordering::Release);
        }));
        assert_eq!(slot.load(Ordering::Acquire), 0x1000);
    }

    #[test]
    fn completed_retirement_backlog_requires_multiple_candidate_batches() {
        let domain = CodeLifetime::managed();
        for i in 0..129 {
            let block = compiled(&domain, 0x1000 + i * 16);
            domain.discard(&block);
        }
        let mut batches = 0;
        let mut counts = Vec::new();
        while domain.has_retired_units() {
            let tickets = domain.retirement_candidates(64);
            counts.push(tickets.len());
            for ticket in tickets {
                assert!(domain.execution_complete(ticket));
                // SAFETY: this registry-only fixture owns no executable storage,
                // code-map range or compiler records; execution grace is complete.
                unsafe { domain.finish_retirement(ticket) };
            }
            batches += 1;
        }
        assert_eq!(counts, [64, 64, 1]);
        assert_eq!(batches, 3, "a poll interval cannot bound backlog latency");
    }

    #[test]
    fn maintenance_parks_until_retirement_prediction_retirement_or_stop() {
        use std::sync::mpsc;
        use std::time::Duration;
        let domain = Arc::new(CodeLifetime::managed());
        let block = compiled(&domain, 0x1000);
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let worker_domain = domain.clone();
        let worker_stop = stop.clone();
        let worker = std::thread::spawn(move || {
            while worker_domain.wait_for_reclamation(&worker_stop, Duration::from_millis(1)) {
                tx.send(()).unwrap();
                worker_domain.collect_ibtc_descriptors();
            }
        });
        assert!(rx.recv_timeout(Duration::from_millis(40)).is_err());
        let CachedBlock::Compiled { entry } = block.clone() else {
            unreachable!()
        };
        domain.alloc_ibtc_descriptor(0x2000, entry);
        domain.flush_predictions();
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
        // A descriptor-only retirement must wake maintenance as well as a unit.
        domain.discard(&block);
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
        stop.store(true, Ordering::Release);
        domain.wake_maintenance();
        worker.join().unwrap();
        // Stop also wakes a completely idle domain.
        let idle = CodeLifetime::managed();
        assert!(!idle.wait_for_reclamation(&stop, Duration::from_secs(1)));
    }

    #[test]
    fn cache_drop_and_prediction_flush_keep_descriptors_until_execution_grace() {
        let domain = Arc::new(CodeLifetime::managed());
        let cache = TranslationCache::with_code_lifetime(Some(domain.clone()));
        let block = compiled(&domain, 0x1000);
        let CachedBlock::Compiled { entry } = block.clone() else {
            unreachable!()
        };
        cache.insert(key(0x2000), block, vec![(0x2000, 16)], |_| {});
        let slot = Arc::new(AtomicU64::new(0));
        domain.register_slot(&slot);
        let mut participant = domain.participant();
        let held = participant.enter();
        let address = cache.alloc_ibtc_descriptor(0x2000, entry);
        slot.store(address, Ordering::Release);
        drop(cache);
        assert_eq!(slot.load(Ordering::SeqCst), 0);
        domain.collect_ibtc_descriptors();
        assert_eq!(domain.ibtc_descriptor_count(), 1);
        // SAFETY: the active participant still owns the loaded descriptor.
        assert_eq!(unsafe { *(address as *const [u64; 2]) }, [0x2000, 0x1000]);
        drop(held);
        domain.collect_ibtc_descriptors();
        assert_eq!(domain.ibtc_descriptor_count(), 0);
        let capacity = domain.state.lock().unwrap().retired_descriptors.capacity();
        for _ in 0..64 {
            domain.alloc_ibtc_descriptor(0x2000, entry);
            domain.flush_predictions();
            domain.collect_ibtc_descriptors();
            assert_eq!(domain.ibtc_descriptor_count(), 0);
            assert_eq!(
                domain.state.lock().unwrap().retired_descriptors.capacity(),
                capacity
            );
        }
    }

    #[test]
    fn stale_completion_cannot_publish_or_discard_a_later_address_generation() {
        let domain = Arc::new(CodeLifetime::managed());
        let cache = TranslationCache::with_code_lifetime(Some(domain.clone()));
        let stale = compiled(&domain, 0x1000);
        cache.insert(key(0x2000), stale.clone(), vec![(0x2000, 16)], |_| {});
        cache.invalidate_overlapping(0x2000, 0x2010, || {});
        assert!(domain.execution_complete(domain.retired_units()[0]));
        // Registry-only fixture: model a later provider generation at the same
        // address. No executable storage or code-map range is reused here.
        let current = compiled(&domain, 0x1000);
        let generation = domain.live_units()[0].generation;
        domain.discard(&stale);
        assert_eq!(domain.live_units()[0].generation, generation);
        assert!(!cache.upgrade(
            key(0x2000),
            stale.clone(),
            (0x2000, 16),
            cache.publication_generation()
        ));
        assert!(!cache.upgrade_region(
            key(0x2000),
            stale.clone(),
            vec![(0x2000, 16)],
            cache.publication_generation(),
            |_| {}
        ));
        assert!(!cache.insert_checked(key(0x2000), stale.clone(), vec![(0x2000, 16)], |_| true));
        assert!(cache.get(key(0x2000)).is_none());
        assert_eq!(cache.memory_stats().compiled_entries, 0);

        cache.insert(
            key(0x2000),
            CachedBlock::Interpreted(Arc::new(crate::IrBlock {
                mode: CpuMode::Long64,
                legacy_context: None,
                guest_start: 0x2000,
                guest_len: 16,
                icount: 1,
                temp_count: 0,
                ops: Vec::new(),
                link_exit: false,
            })),
            vec![(0x2000, 16)],
            |_| {},
        );
        assert!(cache.try_begin_tier_up(key(0x2000)));
        let request = cache
            .prepare_tier_up(
                key(0x2000),
                vec![(0x2000, 16)],
                cache.publication_generation(),
                |_| {},
            )
            .unwrap();
        assert!(!cache.publish_tier_up(key(0x2000), stale, None, request, |_| {}));
        assert_eq!(cache.memory_stats().interpreted_entries, 1);
        assert!(cache.upgrade(
            key(0x2000),
            current,
            (0x2000, 16),
            cache.publication_generation()
        ));
        assert_eq!(cache.memory_stats().compiled_entries, 1);
        assert_eq!(domain.retired_units().len(), 1);
    }

    #[test]
    fn managed_admission_rejects_untracked_and_foreign_entries() {
        let domain = Arc::new(CodeLifetime::managed());
        let foreign = CodeLifetime::managed();
        let cache = TranslationCache::with_code_lifetime(Some(domain.clone()));
        let block = compiled(&domain, 0x1000);
        let raw = CachedBlock::Compiled {
            entry: CompiledPtr::new(0x1000 as *const u8),
        };
        assert!(!cache.upgrade(key(0x2000), raw, (0x2000, 16), 0));
        let wrong_domain = compiled(&foreign, 0x1000);
        assert!(!cache.upgrade(key(0x2000), wrong_domain.clone(), (0x2000, 16), 0));
        domain.discard(&wrong_domain);
        assert_eq!(domain.live_units().len(), 1);
        assert!(cache.upgrade(key(0x2000), block, (0x2000, 16), 0));
        assert_eq!(cache.memory_stats().compiled_entries, 1);
    }

    #[test]
    fn direct_admission_requires_current_guard_and_matching_generation() {
        let domain = Arc::new(CodeLifetime::managed());
        let block = compiled(&domain, 0x1000);
        let CachedBlock::Compiled { entry } = block.clone() else {
            unreachable!()
        };
        let mut participant = domain.participant();
        let guard = participant.enter();
        assert!(guard.admits(entry));
        assert!(!guard.admits(CompiledPtr::new(entry.0)));
        domain.discard(&block);
        assert!(!guard.admits(entry));
        assert!(!domain.execution_complete(domain.retired_units()[0]));
        drop(guard);
        let current = compiled(&domain, 0x1000);
        let CachedBlock::Compiled { entry: new_entry } = current else {
            unreachable!()
        };
        let guard = participant.enter();
        assert!(!guard.admits(entry));
        assert!(guard.admits(new_entry));
    }

    #[test]
    fn retirement_during_source_validation_is_checked_before_cache_admission() {
        let domain = Arc::new(CodeLifetime::managed());
        let cache = TranslationCache::with_code_lifetime(Some(domain.clone()));
        let block = compiled(&domain, 0x1000);
        assert!(
            !cache.insert_checked(key(0x2000), block.clone(), vec![(0x2000, 16)], |_| {
                domain.discard(&block);
                true
            })
        );
        assert!(cache.get(key(0x2000)).is_none());
        assert_eq!(domain.retired_units().len(), 1);
    }

    #[test]
    fn retired_generation_waits_for_execution_but_not_idle_or_new_participants() {
        let domain = Arc::new(CodeLifetime::default());
        let block = compiled(&domain, 0x1000);
        let mut old = domain.participant();
        let old_guard = old.enter();
        let _idle = domain.participant();
        domain.discard(&block);
        let ticket = domain.retired_units()[0];
        let mut new = domain.participant();
        let _new_guard = new.enter();
        assert!(!domain.execution_complete(ticket));
        drop(old_guard);
        assert!(domain.execution_complete(ticket));
        assert!(domain.live_units().is_empty());
    }

    #[test]
    fn retirement_at_each_entry_boundary_requires_revalidation_before_resolution() {
        for boundary in 0..2 {
            let domain = Arc::new(CodeLifetime::default());
            let block = compiled(&domain, 0x1000);
            let mut participant = domain.participant();
            let mut calls = 0;
            let guard = participant.enter_probed(|| {
                if calls == boundary {
                    domain.discard(&block);
                    let ticket = domain.retired_units()[0];
                    assert_eq!(domain.execution_complete(ticket), boundary == 0);
                }
                calls += 1;
            });
            assert!(calls >= 4, "entry retried after retirement");
            assert_eq!(guard.epoch(), domain.epoch());
            assert!(domain.execution_complete(domain.retired_units()[0]));
        }
    }

    #[test]
    fn unwinding_and_dropping_participants_release_execution_grace() {
        let domain = Arc::new(CodeLifetime::default());
        let block = compiled(&domain, 0x1000);
        let mut participant = domain.participant();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = participant.enter();
            domain.discard(&block);
            panic!("leave a compiled helper");
        }));
        assert!(result.is_err());
        assert!(domain.execution_complete(domain.retired_units()[0]));
        drop(participant);
        assert!(domain.execution_complete(domain.retired_units()[0]));
    }

    #[test]
    fn last_cache_publication_retires_unit_and_clears_links_before_epoch_change() {
        let domain = Arc::new(CodeLifetime::default());
        let cache = TranslationCache::with_code_lifetime(Some(domain.clone()));
        let block = compiled(&domain, 0x1000);
        let slot = Arc::new(AtomicU64::new(0x1000));
        domain.register_slot(&slot);
        cache.insert(key(0x2000), block.clone(), vec![(0x2000, 16)], |_| {});
        cache.insert(key(0x3000), block.clone(), vec![(0x3000, 16)], |_| {});
        domain.discard(&block);
        cache.invalidate_overlapping(0x2000, 0x2010, || {});
        assert!(domain.retired_units().is_empty());
        assert_eq!(
            slot.load(Ordering::SeqCst),
            0,
            "invalidation conservatively flushes shared predictions"
        );
        let epoch = domain.epoch();
        assert!(cache.publish_if_current(epoch, || slot.store(0x1000, Ordering::SeqCst)));
        let mut participant = domain.participant();
        let guard = participant.enter();
        cache.invalidate_overlapping(0x3000, 0x3010, || {});
        assert!(domain.epoch() > epoch);
        assert_eq!(slot.load(Ordering::SeqCst), 0);
        let ticket = domain.retired_units()[0];
        assert!(!domain.execution_complete(ticket));
        assert!(!cache.publish_if_current(epoch, || panic!("stale slot fill")));
        drop(guard);
        assert!(domain.execution_complete(ticket));
    }

    #[test]
    fn replacement_rejected_completions_and_cache_drop_transfer_unit_ownership() {
        let domain = Arc::new(CodeLifetime::default());
        let cache = TranslationCache::with_code_lifetime(Some(domain.clone()));
        let old = compiled(&domain, 0x1000);
        let old_id = domain.live_units()[0].generation;
        cache.insert(key(0x2000), old, vec![(0x2000, 16)], |_| {});
        let new = compiled(&domain, 0x1100);
        assert!(cache.upgrade(key(0x2000), new, (0x2000, 16), 0));
        assert_eq!(domain.retired_units()[0].unit().generation, old_id);
        let rejected = compiled(&domain, 0x1200);
        assert!(!cache.upgrade(key(0x2000), rejected, (0x2000, 16), u64::MAX));
        let rejected = compiled(&domain, 0x1300);
        assert!(!cache.publish_tier_up(key(0x4000), rejected, None, 1, |_| {}));
        assert_eq!(domain.retired_units().len(), 3);
        drop(cache);
        assert_eq!(domain.retired_units().len(), 4);
        assert!(domain.live_units().is_empty());
    }

    #[test]
    fn publication_panic_still_allows_teardown_to_retire_owned_units() {
        let domain = Arc::new(CodeLifetime::default());
        let cache = TranslationCache::with_code_lifetime(Some(domain.clone()));
        cache.insert(
            key(0x2000),
            compiled(&domain, 0x1000),
            vec![(0x2000, 16)],
            |_| {},
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cache.publish_if_current(cache.epoch(), || panic!("slot fill failed"))
        }));
        assert!(result.is_err());
        drop(cache);
        assert!(domain.live_units().is_empty());
        assert_eq!(domain.retired_units().len(), 1);
    }
}
