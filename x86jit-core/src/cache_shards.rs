//! Shared dispatch reads with independent reader locks (§9.1, §10).
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::{RwLock, RwLockWriteGuard};

use super::cache_hash::{dispatch_hash, BlockHasher};
use crate::{BlockKey, CachedBlock};

type Map = HashMap<BlockKey, CachedBlock, BuildHasherDefault<BlockHasher>>;
const SHARDS: usize = 16;

// Separate reader lock words on hosts with 64- or 128-byte cache lines.
#[repr(align(128))]
struct Shard(RwLock<Map>);

pub(super) struct DispatchMap {
    publication: RwLock<()>,
    shards: [Shard; SHARDS],
}

impl DispatchMap {
    pub(super) fn new() -> Self {
        Self {
            publication: RwLock::new(()),
            shards: std::array::from_fn(|_| Shard(RwLock::new(Map::default()))),
        }
    }

    #[inline]
    fn index(key: &BlockKey) -> usize {
        dispatch_hash(key.guest_addr) as usize & (SHARDS - 1)
    }

    #[inline]
    pub(super) fn get(&self, key: &BlockKey) -> Option<CachedBlock> {
        self.shards[Self::index(key)]
            .0
            .read()
            .unwrap()
            .get(key)
            .cloned()
    }

    pub(super) fn get_accounted(&self, key: &BlockKey) -> Option<CachedBlock> {
        let shard = &self.shards[Self::index(key)].0;
        let guard = match shard.try_read() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::WouldBlock) => {
                let timer = crate::time_stats::ShardReadTimer::contended();
                let guard = shard.read().unwrap();
                drop(timer);
                guard
            }
            Err(std::sync::TryLockError::Poisoned(err)) => panic!("{err}"),
        };
        guard.get(key).cloned()
    }

    // Cache lock order: spans → publication → shards (ascending index) →
    // CodeLifetime registry. Slot fills omit shards; readers take one shard
    // alone. Registry callbacks must never acquire cache locks.
    // Slot publication shares the mutation gate without blocking map readers.
    pub(super) fn publication(&self) -> RwLockWriteGuard<'_, ()> {
        self.publication.write().unwrap()
    }

    // Keep the existing publication write lock and acquire every shard in a
    // fixed order. No reader can observe partial invalidation or publication;
    // all shards stay locked through the caller's epoch update. Readers clone
    // under their shard lock, preserving Arc and JIT-arena lifetime guarantees.
    pub(super) fn write(&self) -> DispatchMapWrite<'_> {
        let publication = self.publication.write().unwrap();
        let shards = std::array::from_fn(|i| self.shards[i].0.write().unwrap());
        DispatchMapWrite {
            shards,
            _publication: publication,
        }
    }

    // Cache destruction has exclusive access and must also work after a
    // publication panic poisoned the gate or any reader shard.
    pub(super) fn exclusive_values(&mut self) -> impl Iterator<Item = &CachedBlock> {
        self.shards.iter_mut().flat_map(|shard| {
            shard
                .0
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .values()
        })
    }

    #[cfg(test)]
    pub(super) fn is_poisoned(&self) -> bool {
        self.publication.is_poisoned()
    }
}

pub(super) struct DispatchMapWrite<'a> {
    shards: [RwLockWriteGuard<'a, Map>; SHARDS],
    _publication: RwLockWriteGuard<'a, ()>,
}

impl DispatchMapWrite<'_> {
    pub(super) fn iter(&self) -> impl Iterator<Item = (&BlockKey, &CachedBlock)> {
        self.shards.iter().flat_map(|shard| shard.iter())
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &CachedBlock> {
        self.shards.iter().flat_map(|shard| shard.values())
    }

    pub(super) fn clear(&mut self) {
        for shard in &mut self.shards {
            shard.clear();
        }
    }

    pub(super) fn get(&self, key: &BlockKey) -> Option<&CachedBlock> {
        self.shards[DispatchMap::index(key)].get(key)
    }

    pub(super) fn insert(&mut self, key: BlockKey, block: CachedBlock) -> Option<CachedBlock> {
        self.shards[DispatchMap::index(&key)].insert(key, block)
    }

    pub(super) fn remove(&mut self, key: &BlockKey) -> Option<CachedBlock> {
        self.shards[DispatchMap::index(key)].remove(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_publication_does_not_lock_reader_shards() {
        let map = DispatchMap::new();
        let _publication = map.publication();
        for shard in &map.shards {
            assert!(shard.0.try_read().is_ok());
        }
    }

    #[test]
    fn strides_spread_across_shards() {
        for base in [0x10000_u64, 0xffff_8000_0001_0000] {
            for stride in [16, 4096, 65536, 1 << 20, 4 << 20, 16 << 20] {
                let spread: std::collections::HashSet<_> = (0..64)
                    .map(|i| {
                        DispatchMap::index(&BlockKey::new(
                            base + i * stride,
                            crate::CpuMode::Long64,
                        ))
                    })
                    .collect();
                assert!(
                    spread.len() >= 12,
                    "base={base:x} stride={stride} spread={}",
                    spread.len()
                );
            }
        }
    }

    #[test]
    fn publication_excludes_every_reader_shard() {
        let map = DispatchMap::new();
        let mut write = map.write();
        for shard in &map.shards {
            assert!(shard.0.try_read().is_err());
        }
        for i in 0..SHARDS {
            let key = BlockKey::new(i as u64 * 32, crate::CpuMode::Long64);
            write.insert(
                key,
                CachedBlock::Compiled {
                    entry: crate::CompiledPtr::new(std::ptr::dangling()),
                },
            );
        }
        drop(write);
        for i in 0..SHARDS {
            let key = BlockKey::new(i as u64 * 32, crate::CpuMode::Long64);
            assert!(map.get(&key).is_some());
            assert!(map.write().remove(&key).is_some());
            assert!(map.get(&key).is_none());
        }
    }
}
