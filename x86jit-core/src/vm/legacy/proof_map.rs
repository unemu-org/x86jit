//! Per-vCPU interpreted proof maps indexed by their registered source pages
//! (xj-to9.45; spec.md §17.10), shared by the Flat32/Segmented and paged maps.
use super::*;

// A proof, its registered source byte ranges `[lo, hi)`, whether a lookup
// returned it since the last eviction sweep, and its weight (`insert`).
type Entry<V> = (V, Box<[(u64, u64)]>, std::cell::Cell<bool>, usize);

/// A per-vCPU proof map with a page index of each proof's registered source
/// bytes (xj-to9.45). A source-generation advance drops only the proofs whose
/// bytes the writes recorded since the map's generation overlap
/// ([`crate::cache::TranslationCache::written_since`]); an unrecorded
/// advance drops every proof, as any advance did before.
pub(super) struct ProofMap<K, V> {
    proofs: EngineMap<K, Entry<V>>,
    generation: u64,
    // Advances whenever a proof, and with it its registration, may be
    // dropped or replaced; successor links (`links.rs`) name it.
    version: u64,
    // The sum of the proofs' weights (`insert`).
    weight: usize,
    // Off the proof hit path, out of line.
    index: Box<SourceIndex<K>>,
}

struct SourceIndex<K> {
    pages: EngineMap<u64, Vec<K>>,
    written: Vec<(u64, u64)>,
    // Why each dropped proof went, for the relift diagnostics (xj-to9.53).
    #[cfg(feature = "test-hooks")]
    history: EngineMap<K, Dropped>,
    // The largest `weight` since the last `take_peak_weight`.
    #[cfg(feature = "test-hooks")]
    peak: usize,
}

/// Why a proof left the map (`ProofMap::dropped`), with the store that
/// dropped it and its registered source bytes.
#[cfg(feature = "test-hooks")]
pub(super) struct Dropped {
    pub(super) cause: super::diag::ReliftCause,
    pub(super) write: Option<(u64, u64)>,
    pub(super) sources: Box<[(u64, u64)]>,
}

/// The cause a proof map records for a drop: a `diag::ReliftCause` with
/// `test-hooks`, nothing otherwise (`why!`).
#[cfg(feature = "test-hooks")]
pub(super) type Why = super::diag::ReliftCause;
#[cfg(not(feature = "test-hooks"))]
pub(super) type Why = ();

// Keys whose drop the diagnostics remember per map.
#[cfg(feature = "test-hooks")]
const HISTORY: usize = 1 << 16;

impl<K, V> Default for ProofMap<K, V> {
    fn default() -> Self {
        Self {
            proofs: EngineMap::default(),
            generation: 0,
            version: 0,
            weight: 0,
            index: Box::new(SourceIndex {
                pages: EngineMap::default(),
                written: Vec::new(),
                #[cfg(feature = "test-hooks")]
                history: EngineMap::default(),
                #[cfg(feature = "test-hooks")]
                peak: 0,
            }),
        }
    }
}

fn source_pages(sources: &[(u64, u64)]) -> impl Iterator<Item = u64> + '_ {
    sources
        .iter()
        .filter(|(lo, hi)| lo < hi)
        .flat_map(|&(lo, hi)| {
            (lo >> crate::memory::CODE_PAGE_BITS)..=((hi - 1) >> crate::memory::CODE_PAGE_BITS)
        })
}

impl<K: Copy + Eq + Hash, V> ProofMap<K, V> {
    #[inline(always)]
    pub(super) fn get(&self, key: &K) -> Option<&V> {
        self.proofs.get(key).map(|(proof, _, used, _)| {
            used.set(true);
            proof
        })
    }

    pub(super) fn contains_key(&self, key: &K) -> bool {
        self.proofs.contains_key(key)
    }

    #[cfg(any(test, feature = "test-hooks"))]
    pub(super) fn len(&self) -> usize {
        self.proofs.len()
    }

    /// The proof generation the map last reconciled to: its proofs hold
    /// every byte as of it (decision-27 §2.4).
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn version(&self) -> u64 {
        self.version
    }

    /// The sum of the proofs' weights.
    #[cfg(any(test, feature = "test-hooks"))]
    pub(super) fn weight(&self) -> usize {
        self.weight
    }

    /// The largest sum of weights since the last call. Diagnostics only.
    #[cfg(feature = "test-hooks")]
    pub(super) fn take_peak_weight(&mut self) -> usize {
        std::mem::replace(&mut self.index.peak, self.weight)
    }

    pub(super) fn clear(&mut self, _why: Why) {
        self.version = self.version.wrapping_add(1);
        #[cfg(feature = "test-hooks")]
        for (key, (_, sources, ..)) in self.proofs.drain() {
            self.index.remember(key, _why, None, sources);
        }
        self.proofs.clear();
        self.index.pages.clear();
        self.weight = 0;
    }

    /// Drop every proof; later advances are compared from `generation`.
    pub(super) fn reset(&mut self, generation: u64, why: Why) {
        self.clear(why);
        self.generation = generation;
    }

    /// Drop the proof for `key`, if any, after a lift that could not be
    /// published: its next lift is counted as `Unpublished`.
    pub(super) fn remove_unpublished(&mut self, key: &K) {
        self.remove_because(key, why!(Unpublished), None);
        // A key with no proof still names why it lifts again.
        #[cfg(feature = "test-hooks")]
        if let Some(entry) = self.index.history.get_mut(key) {
            entry.cause = why!(Unpublished);
        } else {
            self.index
                .remember(*key, why!(Unpublished), None, Box::default());
        }
    }

    /// Drop every proof whose key `drop` names, recording why.
    pub(super) fn remove_where(&mut self, drop: impl Fn(&K) -> bool, _why: Why) {
        let keys: Vec<K> = self.proofs.keys().filter(|k| drop(k)).copied().collect();
        if keys.is_empty() {
            return;
        }
        // As `remove_because` of each key, with one pass over each source
        // page: a page's list holds only keys of held proofs, so `drop`
        // names exactly the dropped ones there, and `retain` keeps the
        // survivors in their order.
        let mut pages = Vec::new();
        for key in &keys {
            self.version = self.version.wrapping_add(1);
            let Some((_, sources, _, weight)) = self.proofs.remove(key) else {
                continue;
            };
            self.weight -= weight;
            pages.extend(source_pages(&sources));
            #[cfg(feature = "test-hooks")]
            self.index.remember(*key, _why, None, sources);
        }
        pages.sort_unstable();
        pages.dedup();
        for page in pages {
            if let Some(list) = self.index.pages.get_mut(&page) {
                list.retain(|k| !drop(k));
                if list.is_empty() {
                    self.index.pages.remove(&page);
                }
            }
        }
    }

    /// Drop the proof for `key`, if any, recording why.
    pub(super) fn remove(&mut self, key: &K, why: Why) {
        self.remove_because(key, why, None);
    }

    fn remove_because(&mut self, key: &K, _why: Why, _write: Option<(u64, u64)>) {
        self.version = self.version.wrapping_add(1);
        let Some((_, sources, _, weight)) = self.proofs.remove(key) else {
            return;
        };
        self.weight -= weight;
        for page in source_pages(&sources) {
            if let Some(keys) = self.index.pages.get_mut(&page) {
                keys.retain(|k| k != key);
                if keys.is_empty() {
                    self.index.pages.remove(&page);
                }
            }
        }
        #[cfg(feature = "test-hooks")]
        self.index.remember(*key, _why, _write, sources);
    }

    /// Why the proof for `key` was last dropped; `None` for a key this map
    /// never held (or has forgotten). Relift diagnostics only.
    #[cfg(feature = "test-hooks")]
    pub(super) fn dropped(&self, key: &K) -> Option<&Dropped> {
        self.index.history.get(key)
    }

    /// Insert a proof registered over `sources` (address, length) that
    /// weighs `weight` (its IR bytes, `make_room`'s budget).
    pub(super) fn insert(&mut self, key: K, proof: V, sources: &[(u64, u32)], weight: usize) {
        if self.proofs.contains_key(&key) {
            self.remove_because(&key, why!(First), None);
        }
        let sources: Box<[(u64, u64)]> = sources
            .iter()
            .map(|&(start, len)| (start, start.saturating_add(u64::from(len))))
            .collect();
        #[cfg(feature = "test-hooks")]
        self.index.history.remove(&key);
        for page in source_pages(&sources) {
            let keys = self.index.pages.entry(page).or_default();
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        self.proofs
            .insert(key, (proof, sources, std::cell::Cell::new(false), weight));
        self.weight += weight;
        #[cfg(feature = "test-hooks")]
        {
            self.index.peak = self.index.peak.max(self.weight);
        }
    }

    #[cfg(test)]
    pub(super) fn keys(&self) -> impl Iterator<Item = &K> {
        self.proofs.keys()
    }

    #[cfg(test)]
    pub(super) fn values(&self) -> impl Iterator<Item = &V> {
        self.proofs.values().map(|(proof, ..)| proof)
    }

    #[cfg(test)]
    pub(super) fn entries(&self) -> impl Iterator<Item = (&K, &V)> {
        self.proofs.iter().map(|(key, (proof, ..))| (key, proof))
    }

    /// Make room for one more proof, of weight `incoming`, in a map of at
    /// most `bound` proofs weighing at most `budget` in all: once either is
    /// reached, drop proofs until both are three quarters or less, rather
    /// than all (xj-to9.53, xj-to9.56), by second chance: first the proofs
    /// no lookup returned since the last sweep and that `hot` (a successor
    /// link or a running walk holding the block) does not name, then,
    /// every flag cleared, the rest in map order. A hot loop under pressure
    /// from blocks entered once (an interrupt path through many contexts)
    /// keeps its proofs. Among equally cold proofs the order is the map's
    /// bucket order, which the fixed hasher makes the same keys every time
    /// rather than the oldest: a loop over more blocks than the bound keeps
    /// a stable three quarters proved, where evicting by age would evict
    /// each block just before its reuse. Returns whether it swept.
    pub(super) fn make_room(
        &mut self,
        bound: usize,
        budget: usize,
        incoming: usize,
        why: Why,
        hot: impl Fn(&V) -> bool,
    ) -> bool {
        if self.proofs.len() < bound && self.weight + incoming <= budget {
            return false;
        }
        let (count, weight) = (bound - bound / 4, budget - budget / 4);
        let (mut left, mut heavy) = (self.proofs.len(), self.weight + incoming);
        // One pass: cold proofs in map order until both targets are met,
        // and the others in map order for when too few are cold.
        let mut gone: Vec<K> = Vec::new();
        let mut rest: Vec<(K, usize)> = Vec::new();
        for (key, (proof, _, used, w)) in &self.proofs {
            if (left > count || heavy > weight) && !used.get() && !hot(proof) {
                gone.push(*key);
                left -= 1;
                heavy -= w;
            } else {
                rest.push((*key, *w));
            }
            used.set(false);
        }
        for (key, w) in rest {
            if left <= count && heavy <= weight {
                break;
            }
            gone.push(key);
            left -= 1;
            heavy -= w;
        }
        for key in gone {
            self.remove_because(&key, why, None);
        }
        true
    }

    /// Drop the proofs that source writes since the map's generation
    /// overlapped, and return the current source generation.
    #[inline(always)]
    pub(super) fn reconcile(&mut self, cache: &crate::cache::TranslationCache) -> u64 {
        let generation = cache.proof_generation();
        if generation == self.generation {
            return generation;
        }
        self.drop_written(cache)
    }

    #[cold]
    #[inline(never)]
    fn drop_written(&mut self, cache: &crate::cache::TranslationCache) -> u64 {
        let mut written = std::mem::take(&mut self.index.written);
        #[cfg(feature = "test-hooks")]
        let since = self.generation;
        let (generation, complete) = cache.written_since(self.generation, &mut written);
        self.generation = generation;
        #[cfg(feature = "test-hooks")]
        let kinds = cache.advances_for_diag(since, generation);
        if !complete {
            #[cfg(feature = "test-hooks")]
            let why = {
                use super::diag::GapKind;
                use crate::cache::advance_kind as k;
                // The first unrecorded advance names the gap; none (or none
                // retained) means the log overflowed.
                let gap = kinds.iter().find_map(|&(.., kind)| match kind {
                    k::INVALIDATE => Some(GapKind::Invalidate),
                    k::UNRECORDED_DRAIN => Some(GapKind::UnrecordedDrain),
                    k::CHILD => Some(GapKind::ChildArena),
                    k::RESET => Some(GapKind::Reset),
                    _ => None,
                });
                super::diag::ReliftCause::Gap(gap.unwrap_or(GapKind::Overflow))
            };
            #[cfg(not(feature = "test-hooks"))]
            let why = ();
            self.clear(why);
        } else if !self.proofs.is_empty() {
            let mut victims = Vec::new();
            // Recorded writes are drained notices, each within one code page
            // (`Memory::note_code_write_range`, or a whole page).
            for &(lo, hi) in &written {
                for page in source_pages(&[(lo, hi)]) {
                    for key in self.index.pages.get(&page).into_iter().flatten() {
                        let (_, sources, ..) = &self.proofs[key];
                        if sources.iter().any(|&(a, b)| a < hi && lo < b) {
                            victims.push((*key, (lo, hi)));
                        }
                    }
                }
            }
            for (key, _write) in &victims {
                #[cfg(feature = "test-hooks")]
                let why = if kinds.iter().any(|&(l, h, kind)| {
                    (l, h) == *_write && kind == crate::cache::advance_kind::PROMOTION
                }) {
                    super::diag::ReliftCause::Promotion
                } else {
                    // Classified when the key relifts (`diag::classify`).
                    super::diag::ReliftCause::Store(super::diag::StoreClass::Outside)
                };
                #[cfg(not(feature = "test-hooks"))]
                let why = ();
                self.remove_because(key, why, Some(*_write));
            }
        }
        written.clear();
        self.index.written = written;
        generation
    }
}

#[cfg(feature = "test-hooks")]
impl<K: Copy + Eq + Hash> SourceIndex<K> {
    fn remember(
        &mut self,
        key: K,
        cause: Why,
        write: Option<(u64, u64)>,
        sources: Box<[(u64, u64)]>,
    ) {
        if self.history.len() >= HISTORY && !self.history.contains_key(&key) {
            self.history.clear();
        }
        self.history.insert(
            key,
            Dropped {
                cause,
                write,
                sources,
            },
        );
    }
}

#[cfg(feature = "test-hooks")]
impl<K: Copy + Eq + Hash + super::diag::DiagKey, V> ProofMap<K, V> {
    pub(super) fn record_lift(
        &self,
        vm: &Vm,
        class: super::diag::ProofClass,
        key: &K,
        retry: Option<super::diag::RetryKind>,
        ip: u32,
        context: MachineContext,
    ) {
        use super::diag::{ReliftCause, ReliftRecord};
        let diag = &vm.live_fields.diag;
        let dropped = self.dropped(key);
        let mut cause = if let Some(retry) = retry {
            ReliftCause::Retry(retry)
        } else if self.proofs.contains_key(key) {
            ReliftCause::Permit
        } else {
            dropped.map_or(ReliftCause::First, |d| d.cause)
        };
        let mut instruction = None;
        let sources = dropped.map(|d| &d.sources[..]);
        let write = dropped.and_then(|d| d.write);
        if let (ReliftCause::Store(_) | ReliftCause::Promotion, Some(sources), Some(write)) =
            (cause, sources, write)
        {
            if let Some((class, found)) = super::diag::classify(vm, sources, write, ip, context) {
                if let ReliftCause::Store(_) = cause {
                    cause = ReliftCause::Store(class);
                }
                instruction = found;
            }
        }
        diag.lift(class, cause);
        let (key, key_diff) = diag.key(key);
        diag.record(ReliftRecord {
            class,
            cause,
            block_ip: ip,
            block_bus: sources.and_then(|s| s.first()).map(|&(lo, _)| lo),
            write: write.map(|(lo, hi)| (lo, (hi - lo) as u32)),
            instruction,
            key,
            key_diff,
        });
    }
}
