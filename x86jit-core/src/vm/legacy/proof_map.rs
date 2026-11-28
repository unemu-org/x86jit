//! Per-vCPU interpreted proof maps indexed by their registered source bytes
//! (xj-to9.45, xj-to9.85; spec.md §17.10), shared by the Flat32/Segmented
//! and paged maps.
use super::*;
use std::cell::Cell;

/// A proof map's key. `group` names the keys [`ProofMap::remove_group`]
/// drops together; a map whose keys name none keeps no group index.
pub(super) trait ProofKey: Copy + Eq + Hash {
    fn group(&self) -> Option<u64> {
        None
    }
}

impl ProofKey for LegacyBlockKey {}

// Plain integer keys, for the maps the unit tests build.
#[cfg(test)]
impl ProofKey for u64 {}

struct Entry<V> {
    proof: V,
    // Whether a lookup returned it since the last eviction sweep.
    used: Cell<bool>,
    // Its weight (`insert`).
    weight: usize,
    // Its key's slot in `SourceIndex::keys`, which its nodes name.
    slot: u32,
    // Its node in its group's chain, or `NONE` (`ProofKey::group`).
    group: u32,
    // Its first granule node; the others follow through `Node::sibling`.
    spans: u32,
    // Its registered source byte ranges `[lo, hi)`, for the diagnostics.
    #[cfg(feature = "test-hooks")]
    sources: Box<[(u64, u64)]>,
}

/// A per-vCPU proof map with an index of each proof's registered source
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

// No node: a chain's end, an empty free list.
const NONE: u32 = u32::MAX;

// The index's granule: 64 bytes (xj-to9.85). A recorded write visits the
// granules it covers and, in each, the proofs whose registered bytes lie
// there, not every proof on its page: a store beside a page of code that
// holds a thousand proofs meets the few whose bytes share its 64.
const GRANULE_BITS: u32 = 6;

// A chain member: one proof's registered bytes within one granule, or a
// proof in its group's chain. Chains are doubly linked through the node
// slab, so a proof leaves each of its chains in constant time. 32 bytes: the
// proof's key is named by its slot.
struct Node {
    // The granule number (address >> GRANULE_BITS) or the group whose
    // chain holds the node.
    chain: u64,
    prev: u32,
    next: u32,
    // The proof's next granule node, `NONE` after its last.
    sibling: u32,
    // The proof's slot in `SourceIndex::keys`.
    slot: u32,
    // A granule node's bytes `[lo, hi)`, as offsets into its granule.
    lo: u8,
    hi: u8,
}

/// The bytes a proof of `key` over `sources` (address, length) adds to the
/// index (xj-to9.85): its key's slot and one node per granule its merged
/// ranges cover, and one for its group. Part of the proof's weight, which
/// `ProofMap::make_room` bounds.
pub(super) fn index_weight<K: ProofKey>(key: &K, sources: &[(u64, u32)]) -> usize {
    let nodes = merged(ranges(sources))
        .map(|(lo, hi)| granules(lo, hi).count())
        .sum::<usize>()
        + usize::from(key.group().is_some());
    nodes * std::mem::size_of::<Node>() + std::mem::size_of::<K>()
}

// `[start, start + len)` of each source.
fn ranges(sources: &[(u64, u32)]) -> impl Iterator<Item = (u64, u64)> + Clone + '_ {
    sources
        .iter()
        .map(|&(start, len)| (start, start.saturating_add(u64::from(len))))
}

// `ranges` without its empty ones, each run of touching or overlapping
// successive ranges joined into one: the same bytes. A paged lift registers
// one range per instruction, contiguous within a page.
fn merged(ranges: impl Iterator<Item = (u64, u64)>) -> impl Iterator<Item = (u64, u64)> {
    let mut ranges = ranges.filter(|(lo, hi)| lo < hi).peekable();
    std::iter::from_fn(move || {
        let (mut lo, mut hi) = ranges.next()?;
        while let Some(&(a, b)) = ranges.peek() {
            if a > hi || b < lo {
                break;
            }
            (lo, hi) = (lo.min(a), hi.max(b));
            ranges.next();
        }
        Some((lo, hi))
    })
}

struct SourceIndex<K> {
    // Each granule's chain of the nodes that register its bytes.
    granules: EngineMap<u64, u32>,
    // Each group's chain (`ProofKey::group`).
    groups: EngineMap<u64, u32>,
    nodes: Vec<Node>,
    // The first free node of `nodes`, chained through `next`.
    free: u32,
    // Each held proof's key, by its slot (`Entry::slot`), and the free slots.
    keys: Vec<K>,
    free_keys: Vec<u32>,
    written: Vec<(u64, u64)>,
    // The proofs a recorded write or a group drop names, and the write.
    victims: Vec<(K, (u64, u64))>,
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
                granules: EngineMap::default(),
                groups: EngineMap::default(),
                nodes: Vec::new(),
                free: NONE,
                keys: Vec::new(),
                free_keys: Vec::new(),
                written: Vec::new(),
                victims: Vec::new(),
                #[cfg(feature = "test-hooks")]
                history: EngineMap::default(),
                #[cfg(feature = "test-hooks")]
                peak: 0,
            }),
        }
    }
}

// The granules `[lo, hi)` covers, each with the offsets of its bytes there.
fn granules(lo: u64, hi: u64) -> impl Iterator<Item = (u64, u8, u8)> {
    let (first, last) = if lo < hi {
        (lo >> GRANULE_BITS, (hi - 1) >> GRANULE_BITS)
    } else {
        (1, 0)
    };
    (first..=last).map(move |granule| {
        let base = granule << GRANULE_BITS;
        let end = base.saturating_add(1 << GRANULE_BITS);
        (
            granule,
            (lo.max(base) - base) as u8,
            (hi.min(end) - base) as u8,
        )
    })
}

impl<K: ProofKey> SourceIndex<K> {
    fn chains(&mut self, group: bool) -> &mut EngineMap<u64, u32> {
        if group {
            &mut self.groups
        } else {
            &mut self.granules
        }
    }

    // A new node at the head of `chain`'s chain.
    fn add(&mut self, slot: u32, chain: u64, (lo, hi): (u8, u8), sibling: u32, group: bool) -> u32 {
        let next = self.chains(group).get(&chain).copied().unwrap_or(NONE);
        let node = Node {
            chain,
            prev: NONE,
            next,
            sibling,
            slot,
            lo,
            hi,
        };
        let n = if self.free == NONE {
            let n = u32::try_from(self.nodes.len()).expect("proof index nodes");
            assert_ne!(n, NONE, "proof index nodes");
            self.nodes.push(node);
            n
        } else {
            let n = self.free;
            self.free = self.nodes[n as usize].next;
            self.nodes[n as usize] = node;
            n
        };
        if next != NONE {
            self.nodes[next as usize].prev = n;
        }
        self.chains(group).insert(chain, n);
        n
    }

    // Unlink node `n` from its chain and free it.
    fn unlink(&mut self, n: u32, group: bool) {
        let Node {
            chain, prev, next, ..
        } = self.nodes[n as usize];
        if prev == NONE {
            if next == NONE {
                self.chains(group).remove(&chain);
            } else {
                self.chains(group).insert(chain, next);
            }
        } else {
            self.nodes[prev as usize].next = next;
        }
        if next != NONE {
            self.nodes[next as usize].prev = prev;
        }
        self.nodes[n as usize].next = self.free;
        self.free = n;
    }

    // Register a proof of `key` over `sources`: its key's slot, its group
    // node and its granule nodes, one per granule of each merged range
    // (`merged`). An empty range registers no byte.
    fn register(&mut self, key: K, sources: impl Iterator<Item = (u64, u64)>) -> (u32, u32, u32) {
        let slot = if let Some(slot) = self.free_keys.pop() {
            self.keys[slot as usize] = key;
            slot
        } else {
            let slot = u32::try_from(self.keys.len()).expect("proof index slots");
            self.keys.push(key);
            slot
        };
        let group = key
            .group()
            .map_or(NONE, |group| self.add(slot, group, (0, 0), NONE, true));
        let mut spans = NONE;
        for (lo, hi) in merged(sources) {
            for (granule, lo, hi) in granules(lo, hi) {
                spans = self.add(slot, granule, (lo, hi), spans, false);
            }
        }
        (slot, group, spans)
    }

    // Unlink every node of a proof `register` returned and free its slot.
    fn release(&mut self, slot: u32, group: u32, spans: u32) {
        self.free_keys.push(slot);
        if group != NONE {
            self.unlink(group, true);
        }
        let mut n = spans;
        while n != NONE {
            let sibling = self.nodes[n as usize].sibling;
            self.unlink(n, false);
            n = sibling;
        }
    }

    fn clear(&mut self) {
        self.granules.clear();
        self.groups.clear();
        self.nodes.clear();
        self.free = NONE;
        self.keys.clear();
        self.free_keys.clear();
    }

    // Append to `victims` each proof whose registered bytes a write of
    // `written` overlaps, with the write, in the order of `written`.
    fn overlapping(&self, written: &[(u64, u64)], victims: &mut Vec<(K, (u64, u64))>) {
        for &(lo, hi) in written {
            for (granule, a, b) in granules(lo, hi) {
                let mut n = self.granules.get(&granule).copied().unwrap_or(NONE);
                while n != NONE {
                    let node = &self.nodes[n as usize];
                    if node.lo < b && a < node.hi {
                        victims.push((self.keys[node.slot as usize], (lo, hi)));
                    }
                    n = node.next;
                }
            }
        }
    }
}

impl<K: ProofKey, V> ProofMap<K, V> {
    #[inline(always)]
    pub(super) fn get(&self, key: &K) -> Option<&V> {
        self.proofs.get(key).map(|entry| {
            entry.used.set(true);
            &entry.proof
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

    /// The bytes the index holds now: its node and key slabs and its chain
    /// heads, at their capacity. Diagnostics only.
    #[cfg(feature = "test-hooks")]
    pub(super) fn index_bytes(&self) -> usize {
        let index = &self.index;
        let head = std::mem::size_of::<(u64, u32)>() + 1;
        index.nodes.capacity() * std::mem::size_of::<Node>()
            + index.keys.capacity() * std::mem::size_of::<K>()
            + index.free_keys.capacity() * 4
            + (index.granules.capacity() + index.groups.capacity()) * head
    }

    /// The largest sum of weights since the last call. Diagnostics only.
    #[cfg(feature = "test-hooks")]
    pub(super) fn take_peak_weight(&mut self) -> usize {
        std::mem::replace(&mut self.index.peak, self.weight)
    }

    pub(super) fn clear(&mut self, _why: Why) {
        self.version = self.version.wrapping_add(1);
        #[cfg(feature = "test-hooks")]
        for (key, entry) in self.proofs.drain() {
            self.index.remember(key, _why, None, entry.sources);
        }
        self.proofs.clear();
        self.index.clear();
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

    /// Drop every proof whose key names `group` (`ProofKey::group`),
    /// recording why: a walk of the group's chain, not of every proof.
    pub(super) fn remove_group(&mut self, group: u64, why: Why) {
        let mut victims = std::mem::take(&mut self.index.victims);
        let mut n = self.index.groups.get(&group).copied().unwrap_or(NONE);
        while n != NONE {
            let node = &self.index.nodes[n as usize];
            victims.push((self.index.keys[node.slot as usize], (0, 0)));
            n = node.next;
        }
        for (key, _) in &victims {
            self.remove_because(key, why, None);
        }
        victims.clear();
        self.index.victims = victims;
    }

    /// Drop the proof for `key`, if any, recording why.
    pub(super) fn remove(&mut self, key: &K, why: Why) {
        self.remove_because(key, why, None);
    }

    fn remove_because(&mut self, key: &K, _why: Why, _write: Option<(u64, u64)>) {
        self.version = self.version.wrapping_add(1);
        let Some(entry) = self.proofs.remove(key) else {
            return;
        };
        self.weight -= entry.weight;
        self.index.release(entry.slot, entry.group, entry.spans);
        #[cfg(feature = "test-hooks")]
        self.index.remember(*key, _why, _write, entry.sources);
    }

    /// Why the proof for `key` was last dropped; `None` for a key this map
    /// never held (or has forgotten). Relift diagnostics only.
    #[cfg(feature = "test-hooks")]
    pub(super) fn dropped(&self, key: &K) -> Option<&Dropped> {
        self.index.history.get(key)
    }

    /// Insert a proof registered over `sources` (address, length) that
    /// weighs `weight` (its IR bytes and `index_weight`, `make_room`'s
    /// budget).
    pub(super) fn insert(&mut self, key: K, proof: V, sources: &[(u64, u32)], weight: usize) {
        if self.proofs.contains_key(&key) {
            self.remove_because(&key, why!(First), None);
        }
        let ranges = ranges(sources);
        #[cfg(feature = "test-hooks")]
        self.index.history.remove(&key);
        let (slot, group, spans) = self.index.register(key, ranges.clone());
        self.proofs.insert(
            key,
            Entry {
                proof,
                used: Cell::new(false),
                weight,
                slot,
                group,
                spans,
                #[cfg(feature = "test-hooks")]
                sources: ranges.collect(),
            },
        );
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
        self.proofs.values().map(|entry| &entry.proof)
    }

    #[cfg(test)]
    pub(super) fn entries(&self) -> impl Iterator<Item = (&K, &V)> {
        self.proofs.iter().map(|(key, entry)| (key, &entry.proof))
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
    /// each block just before its reuse. Returns whether it swept. A sweep
    /// visits every proof and drops a quarter of the bound or the budget,
    /// each in time proportional to its registration: a constant cost per
    /// insert, amortized.
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
        for (key, entry) in &self.proofs {
            if (left > count || heavy > weight) && !entry.used.get() && !hot(&entry.proof) {
                gone.push(*key);
                left -= 1;
                heavy -= entry.weight;
            } else {
                rest.push((*key, entry.weight));
            }
            entry.used.set(false);
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
            // Recorded writes are drained notices, each within one code page
            // (`Memory::note_code_write_range`, or a whole page).
            let mut victims = std::mem::take(&mut self.index.victims);
            self.index.overlapping(&written, &mut victims);
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
            victims.clear();
            self.index.victims = victims;
        }
        written.clear();
        self.index.written = written;
        generation
    }
}

#[cfg(feature = "test-hooks")]
impl<K: ProofKey> SourceIndex<K> {
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
impl<K: ProofKey + super::diag::DiagKey, V> ProofMap<K, V> {
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

#[cfg(test)]
mod tests {
    use super::*;

    // A key and its group, as the paged map's context id.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    struct Grouped(u64, u64);

    impl ProofKey for Grouped {
        fn group(&self) -> Option<u64> {
            Some(self.1)
        }
    }

    // Every chain links only held proofs' nodes, both ways, from its head;
    // each proof has exactly its group node and one node per granule each
    // source range covers.
    fn check<K: ProofKey + std::fmt::Debug>(map: &ProofMap<K, Vec<(u64, u64)>>) {
        let index = &map.index;
        let mut seen = 0;
        for (group, chains) in [(false, &index.granules), (true, &index.groups)] {
            for (&chain, &first) in chains {
                let (mut n, mut prev) = (first, NONE);
                assert_ne!(n, NONE);
                while n != NONE {
                    let node = &index.nodes[n as usize];
                    assert_eq!((node.chain, node.prev), (chain, prev));
                    let key = index.keys[node.slot as usize];
                    let entry = &map.proofs[&key];
                    assert_eq!(entry.slot, node.slot);
                    if group {
                        assert_eq!(entry.group, n);
                        assert_eq!(key.group(), Some(chain));
                    }
                    seen += 1;
                    (prev, n) = (n, node.next);
                }
            }
        }
        let mut expected = 0;
        for (key, entry) in &map.proofs {
            let ranges = &entry.proof;
            let mut granules: Vec<(u64, u8, u8)> = merged(ranges.iter().copied())
                .flat_map(|(lo, hi)| granules(lo, hi))
                .collect();
            let mut n = entry.spans;
            let mut found = Vec::new();
            while n != NONE {
                let node = &index.nodes[n as usize];
                assert_eq!(node.slot, entry.slot);
                found.push((node.chain, node.lo, node.hi));
                n = node.sibling;
            }
            granules.sort_unstable();
            found.sort_unstable();
            assert_eq!(found, granules, "{key:?}");
            expected += granules.len() + usize::from(key.group().is_some());
        }
        assert_eq!(seen, expected);
        // The free list holds the rest.
        let (mut free, mut n) = (0, index.free);
        while n != NONE {
            free += 1;
            n = index.nodes[n as usize].next;
        }
        assert_eq!(seen + free, index.nodes.len());
        assert_eq!(map.proofs.len() + index.free_keys.len(), index.keys.len());
    }

    // A paged lift's per-instruction ranges, contiguous within a page,
    // register one node per granule they cover, and `index_weight` counts
    // the nodes and the key's slot.
    #[test]
    fn contiguous_instruction_ranges_register_one_node_per_granule() {
        let mut map: ProofMap<Grouped, Vec<(u64, u64)>> = ProofMap::default();
        // 20 instructions of 1-6 bytes from 1030h to 1072h.
        let mut sources = Vec::new();
        let mut at = 0x1030u64;
        for i in 0..20u32 {
            let len = 1 + i % 6;
            sources.push((at, len));
            at += u64::from(len);
        }
        let key = Grouped(1, 2);
        let node = std::mem::size_of::<Node>();
        assert_eq!(node, 32);
        // Granules 40h and 41h, and the group node.
        assert_eq!(
            index_weight(&key, &sources),
            3 * node + std::mem::size_of::<Grouped>()
        );
        map.insert(key, vec![(0x1030, at)], &sources, 1);
        check(&map);
        assert_eq!(map.index.nodes.len(), 3);
    }

    // The index finds exactly the proofs a brute-force overlap test names,
    // across inserts, replacements, removals, group drops and write drops
    // of proofs over one to three ranges, many sharing granules and pages.
    #[test]
    fn the_granule_index_names_exactly_the_overlapped_proofs() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let mut map: ProofMap<Grouped, Vec<(u64, u64)>> = ProofMap::default();
        for step in 0..20_000 {
            match next(10) {
                0..=4 => {
                    let key = Grouped(next(600), next(8));
                    let ranges: Vec<(u64, u32)> = if next(2) == 0 {
                        // A paged lift's: one range per instruction.
                        let mut at = next(0x3000);
                        (0..1 + next(8))
                            .map(|_| {
                                let len = next(15) as u32;
                                at += u64::from(len);
                                (at - u64::from(len), len)
                            })
                            .collect()
                    } else {
                        (0..1 + next(3))
                            .map(|_| (next(0x3000), next(200) as u32))
                            .collect()
                    };
                    let proof = ranges
                        .iter()
                        .map(|&(lo, len)| (lo, lo + u64::from(len)))
                        .collect();
                    map.insert(key, proof, &ranges, 1);
                }
                5 => map.remove(&Grouped(next(600), next(8)), why!(First)),
                6 if step % 7 == 0 => {
                    let group = next(8);
                    map.remove_group(group, why!(ContextEvicted));
                    assert!(map.keys().all(|key| key.1 != group));
                }
                _ => {
                    let written: Vec<(u64, u64)> = (0..1 + next(3))
                        .map(|_| {
                            let lo = next(0x3000);
                            (lo, lo + next(8))
                        })
                        .collect();
                    // An empty write or source range holds no byte.
                    let overlaps = |ranges: &[(u64, u64)]| {
                        ranges.iter().any(|&(a, b)| {
                            written
                                .iter()
                                .any(|&(lo, hi)| lo < hi && a < b && a < hi && lo < b)
                        })
                    };
                    let mut expected: Vec<Grouped> = map
                        .entries()
                        .filter(|(_, ranges)| overlaps(ranges))
                        .map(|(key, _)| *key)
                        .collect();
                    let mut victims = Vec::new();
                    map.index.overlapping(&written, &mut victims);
                    let mut found: Vec<Grouped> = victims.iter().map(|&(key, _)| key).collect();
                    for (key, write) in &victims {
                        assert!(written.contains(write));
                        map.remove_because(key, why!(First), Some(*write));
                    }
                    let order = |a: &Grouped, b: &Grouped| (a.0, a.1).cmp(&(b.0, b.1));
                    expected.sort_unstable_by(order);
                    found.sort_unstable_by(order);
                    found.dedup();
                    assert_eq!(found, expected, "step {step}: {written:?}");
                }
            }
            if step % 97 == 0 {
                check(&map);
            }
        }
        check(&map);
        map.clear(why!(First));
        check(&map);
        assert!(map.index.nodes.is_empty());
    }
}
