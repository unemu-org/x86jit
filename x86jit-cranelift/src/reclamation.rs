//! Cold collection and deferred teardown (spec §9.1; decision-17).
use super::*;
use std::ops::{Deref, DerefMut};
use std::time::Duration;
use x86jit_core::code_lifetime::CodeUnit;

/// Allocation lifecycle. Release is delivered before the extent can be reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeLifecycleKind {
    Publish,
    Retire,
    Release,
}

/// Identity includes a process-wide generation, even when the host range repeats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeLifecycleEvent {
    pub unit: CodeUnit,
    /// Publication metadata, preserved unchanged through Retire and Release.
    pub code: CompiledCode,
    pub kind: CodeLifecycleKind,
}

pub(super) struct Definition<'a> {
    jit: &'a mut Jit,
    slots_start: usize,
    pub id: Option<FuncId>,
    pub keep: bool,
}
impl<'a> Definition<'a> {
    pub fn new(jit: &'a mut Jit) -> Self {
        Self {
            slots_start: jit.slots.len(),
            jit,
            id: None,
            keep: false,
        }
    }
}
impl Deref for Definition<'_> {
    type Target = Jit;
    fn deref(&self) -> &Jit {
        self.jit
    }
}
impl DerefMut for Definition<'_> {
    fn deref_mut(&mut self) -> &mut Jit {
        self.jit
    }
}
impl Drop for Definition<'_> {
    fn drop(&mut self) {
        if !self.keep {
            self.jit.slots.truncate(self.slots_start);
            if let Some(id) = self.id {
                discard_definition(self.jit, id);
            }
        }
    }
}

pub(super) fn discard_definition(jit: &mut Jit, id: FuncId) {
    jit.pending_slots.remove(&id);
    jit.functions.retain(|_, record| record.id != id);
    // SAFETY: only unpublished failure paths call this; they never register a
    // code-map range or expose the entry. Units have no cross-unit relocations.
    unsafe { jit.module.discard_unpublished_function(id) };
    jit.reusable_functions.push(id);
}

impl JitBackend {
    /// Observe allocation publication, retirement and release, including teardown.
    /// Install before compiling. Callbacks may reenter compilation and explicit
    /// collection; no compiler, registry or observer lock is held. Publish runs
    /// on the compiler thread; retire/release run on a collector (normally the
    /// maintenance thread). Retire delivery also depends on the candidate backlog.
    /// Release completes before address reuse; a consumer must stop using the
    /// range before returning from Release. System storage can remain reserved.
    /// Notifications for one generation are ordered. Callback panics disable
    /// notifications with the same containment rules as the legacy observer.
    /// Consumers must discard their references on panic. The legacy observer's
    /// backend-lifetime retention promise still applies if both are installed.
    /// Callbacks must not wait for maintenance or invoke backend shutdown.
    pub fn set_code_lifecycle_observer(
        &self,
        observer: impl Fn(CodeLifecycleEvent) + Send + Sync + 'static,
    ) {
        assert_eq!(
            self.shared.compile_attempts.load(Ordering::Relaxed),
            0,
            "install the lifecycle observer before compiling"
        );
        let previous = self
            .shared
            .lifecycle_observer
            .lock()
            .unwrap()
            .replace(Arc::new(observer));
        drop_observer_value(previous);
    }

    pub(super) fn ensure_maintenance(&self) {
        if check_compile_allowed().is_err() || self.shared.maintenance_stop.load(Ordering::Acquire)
        {
            return;
        }
        let mut worker = self.maintenance.lock().unwrap();
        if worker.is_some() {
            return;
        }
        let shared = self.shared.clone();
        match (self.worker_spawner)(Box::new(move || {
            while shared
                .lifetime
                .wait_for_reclamation(&shared.maintenance_stop, Duration::from_millis(10))
            {
                shared.collect_retired_metadata();
            }
        })) {
            Ok(handle) => *worker = Some(handle),
            Err(_) => {
                self.shared.exhaust_code_space::<()>();
            }
        }
    }

    pub(super) fn defer_reclamation(&self) {
        if self.shared.owner != x86jit_core::codemap::ProcessIdentity::current() {
            return;
        }
        self.shared.collect_retired_metadata();
        if !self.shared.lifetime.has_pending_reclamation() {
            return;
        }
        let shared = self.shared.clone();
        match (self.worker_spawner)(Box::new(move || {
            while shared.lifetime.has_pending_reclamation() {
                shared.collect_retired_metadata();
                if shared.lifetime.has_pending_reclamation() {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        })) {
            Ok(worker) => self.shared.lifetime.retain_reclamation_worker(worker),
            Err(_) => {
                // Retain helpers, slots, compiler records and provider storage
                // if the policy cannot start the deferred owner.
                std::mem::forget(self.shared.clone());
            }
        }
    }
}

struct Collecting<'a>(&'a AtomicBool);
impl Drop for Collecting<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl Shared {
    pub(super) fn discard_unpublished_entry(&self, entry: CompiledPtr) {
        let _serial = self
            .compile_serial
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for mode in [0, 1] {
            if let Some(jit) = self.lock_module(mode).as_mut() {
                if let Some(record) = jit.functions.get(&entry.generation()) {
                    let id = record.id;
                    discard_definition(jit, id);
                    break;
                }
            }
        }
    }

    pub(super) fn collect_retired_metadata(&self) -> usize {
        if self.owner != x86jit_core::codemap::ProcessIdentity::current()
            || self.collecting.swap(true, Ordering::AcqRel)
        {
            return 0;
        }
        let _collecting = Collecting(&self.collecting);
        self.lifetime.collect_ibtc_descriptors();
        let mut completed = 0;
        for ticket in self.lifetime.retirement_candidates(64) {
            let unit = ticket.unit();
            let code = [0, 1]
                .into_iter()
                .find_map(|mode| {
                    self.lock_module(mode).as_ref().and_then(|jit| {
                        jit.functions
                            .get(&unit.generation)
                            .map(|record| record.code)
                    })
                })
                .expect("registered unit owns publication metadata");
            let first = self
                .retirement_announced
                .lock()
                .unwrap()
                .insert(unit.generation);
            if first {
                Self::notify_observer(
                    &self.lifecycle_observer,
                    CodeLifecycleEvent {
                        unit,
                        code,
                        kind: CodeLifecycleKind::Retire,
                    },
                );
            }
            if !self.lifetime.execution_complete(ticket) {
                continue;
            }
            {
                let _serial = self.lock_telemetry.lock(
                    &self.compile_serial,
                    LockKind::CompileSerial,
                    "JIT metadata retirement",
                );
                let mut removals = self.retired_metadata.lock().unwrap();
                let removal = match removals.entry(unit.generation) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let Some(removal) = x86jit_core::codemap::unregister_with_completion(
                            unit.host_start,
                            unit.host_len,
                        ) else {
                            continue;
                        };
                        entry.insert(removal)
                    }
                };
                if !removal.is_complete() {
                    continue;
                }
            }
            // Keep the provider extent owned while consumer references drain.
            // Reentrant collection returns immediately through `collecting`.
            Self::notify_observer(
                &self.lifecycle_observer,
                CodeLifecycleEvent {
                    unit,
                    code,
                    kind: CodeLifecycleKind::Release,
                },
            );
            let _serial = self.lock_telemetry.lock(
                &self.compile_serial,
                LockKind::CompileSerial,
                "JIT allocation release",
            );
            let mut found = false;
            for mode in [0, 1] {
                if let Some(jit) = self.lock_module(mode).as_mut() {
                    if let Some(record) = jit.functions.remove(&unit.generation) {
                        // SAFETY: unlinking, execution grace and exact range's
                        // signal-reader grace completed. No cross-unit relocation
                        // targets this ID; helpers and slots use indirect calls.
                        unsafe { jit.module.retire_function_definition(record.id) };
                        jit.reusable_functions.push(record.id);
                        drop(record);
                        if !self.retain_observed_code.load(Ordering::Acquire) {
                            // SAFETY: compiler references and observer references
                            // are gone, and both grace periods preceded release.
                            unsafe {
                                jit.module.release_function_allocation(
                                    unit.host_start as *const u8,
                                    unit.generation,
                                )
                            };
                        }
                        found = true;
                        break;
                    }
                }
            }
            assert!(found, "registered unit owns a compiler definition");
            // SAFETY: metadata, compiler records and provider ownership settled.
            unsafe { self.lifetime.finish_retirement(ticket) };
            self.retired_metadata
                .lock()
                .unwrap()
                .remove(&unit.generation);
            self.retirement_announced
                .lock()
                .unwrap()
                .remove(&unit.generation);
            completed += 1;
        }
        completed
    }
}
