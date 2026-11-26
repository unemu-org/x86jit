//! Per-vCPU interpreted proof maps indexed by their registered source pages
//! (xj-to9.45; spec.md §17.10), shared by the Flat32/Segmented and paged maps.
use super::*;

// A proof and its registered source byte ranges `[lo, hi)`.
type Entry<V> = (V, Box<[(u64, u64)]>);

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
    // Off the proof hit path, out of line.
    index: Box<SourceIndex<K>>,
}

struct SourceIndex<K> {
    pages: EngineMap<u64, Vec<K>>,
    written: Vec<(u64, u64)>,
}

impl<K, V> Default for ProofMap<K, V> {
    fn default() -> Self {
        Self {
            proofs: EngineMap::default(),
            generation: 0,
            version: 0,
            index: Box::new(SourceIndex {
                pages: EngineMap::default(),
                written: Vec::new(),
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
        self.proofs.get(key).map(|(proof, _)| proof)
    }

    pub(super) fn contains_key(&self, key: &K) -> bool {
        self.proofs.contains_key(key)
    }

    pub(super) fn len(&self) -> usize {
        self.proofs.len()
    }

    pub(super) fn version(&self) -> u64 {
        self.version
    }

    pub(super) fn clear(&mut self) {
        self.version = self.version.wrapping_add(1);
        self.proofs.clear();
        self.index.pages.clear();
    }

    /// Drop every proof; later advances are compared from `generation`.
    pub(super) fn reset(&mut self, generation: u64) {
        self.clear();
        self.generation = generation;
    }

    pub(super) fn remove(&mut self, key: &K) {
        self.version = self.version.wrapping_add(1);
        let Some((_, sources)) = self.proofs.remove(key) else {
            return;
        };
        for page in source_pages(&sources) {
            if let Some(keys) = self.index.pages.get_mut(&page) {
                keys.retain(|k| k != key);
                if keys.is_empty() {
                    self.index.pages.remove(&page);
                }
            }
        }
    }

    /// Insert a proof registered over `sources` (address, length).
    pub(super) fn insert(&mut self, key: K, proof: V, sources: &[(u64, u32)]) {
        if self.proofs.contains_key(&key) {
            self.remove(&key);
        }
        let sources: Box<[(u64, u64)]> = sources
            .iter()
            .map(|&(start, len)| (start, start.saturating_add(u64::from(len))))
            .collect();
        for page in source_pages(&sources) {
            let keys = self.index.pages.entry(page).or_default();
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        self.proofs.insert(key, (proof, sources));
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &V> {
        self.proofs.values().map(|(proof, _)| proof)
    }

    pub(super) fn retain(&mut self, mut keep: impl FnMut(&K) -> bool) {
        self.version = self.version.wrapping_add(1);
        let gone: Vec<K> = self.proofs.keys().filter(|k| !keep(k)).copied().collect();
        for key in gone {
            self.remove(&key);
        }
    }

    /// Drop the proofs that source writes since the map's generation
    /// overlapped, and return the current source generation.
    #[inline(always)]
    pub(super) fn reconcile(&mut self, cache: &crate::cache::TranslationCache) -> u64 {
        let generation = cache.publication_generation();
        if generation == self.generation {
            return generation;
        }
        self.drop_written(cache)
    }

    #[cold]
    #[inline(never)]
    fn drop_written(&mut self, cache: &crate::cache::TranslationCache) -> u64 {
        let mut written = std::mem::take(&mut self.index.written);
        let (generation, complete) = cache.written_since(self.generation, &mut written);
        self.generation = generation;
        if !complete {
            self.clear();
        } else if !self.proofs.is_empty() {
            let mut victims = Vec::new();
            // Recorded writes are drained notices, each within one code page
            // (`Memory::note_code_write_range`, or a whole page).
            for &(lo, hi) in &written {
                for page in source_pages(&[(lo, hi)]) {
                    for key in self.index.pages.get(&page).into_iter().flatten() {
                        let (_, sources) = &self.proofs[key];
                        if sources.iter().any(|&(a, b)| a < hi && lo < b) {
                            victims.push(*key);
                        }
                    }
                }
            }
            for key in &victims {
                self.remove(key);
            }
        }
        written.clear();
        self.index.written = written;
        generation
    }
}
