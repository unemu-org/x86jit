//! Process-global host-PC → guest-RIP side table for guard-page fault recovery.
//! A JIT registers, per compiled function, its host code range `[start, end)` plus
//! a sorted `(host_off, rip_index)` srcloc table and an array of full-width guest
//! RIPs (indexed by `set_srcloc` at each guest instruction; zero machine-code
//! cost). On a SIGSEGV in JIT'd code, the embedder's guarded-run wrapper looks the
//! faulting host PC up here to recover the precise guest RIP — the instruction
//! the interpreter would have stopped on.
//!
//! **Async-signal-safe reads.** The map is append-only: entries and their srcloc
//! tables and RIP arrays are allocated once and never moved or freed for the
//! process's life — cranelift-jit never frees compiled code, and an SMC-dropped
//! block's bytes are
//! unchanged, so this matches the real lifetime. Storage is a fixed array of heap
//! chunks (a chunk, once installed, never moves); a release/acquire `AtomicUsize`
//! length publishes each append. A reader only loads the length and dereferences
//! stable, never-freed pointers — no allocation, no lock — so it is safe to call
//! from a signal-handler context. Writes are serialized by a plain mutex on the
//! (cold) compile path.
//!
//! Pure data — no OS dependencies, so the core stays `{iced-x86}`.

use std::cell::UnsafeCell;
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Entries per chunk. A boxed chunk never moves once installed, so a growing map
/// never invalidates a pointer a concurrent reader already holds.
const CHUNK_CAP: usize = 1024;
/// Chunk-pointer slots. `CHUNK_CAP * MAX_CHUNKS` compiled functions before the
/// map saturates — 4M, far beyond any real run.
const MAX_CHUNKS: usize = 4096;

/// One compiled function's host range and its sorted srcloc table.
#[derive(Clone, Copy)]
struct Entry {
    /// Host code range `[start, end)` (absolute addresses).
    start: usize,
    end: usize,
    /// Leaked, sorted-by-`host_off` `(host_off, rip_index)` pairs. A raw pointer
    /// (not `Box`) keeps `Entry` `Copy` and makes the handler-side read obviously
    /// a plain load of never-freed memory.
    table: *const (u32, u32),
    table_len: usize,
    /// Leaked full-width RIPs. Every table index is validated before publication.
    guest_rips: *const u64,
}

impl Entry {
    const EMPTY: Entry = Entry {
        start: 0,
        end: 0,
        table: ptr::null(),
        table_len: 0,
        guest_rips: ptr::null(),
    };
}

struct Chunk {
    entries: [UnsafeCell<Entry>; CHUNK_CAP],
}

/// Append-only host-PC → guest-RIP map (see module docs for the AS-safety model).
pub struct CodeMap {
    chunks: [AtomicPtr<Chunk>; MAX_CHUNKS],
    /// Number of published entries; release/acquire publishes each append.
    len: AtomicUsize,
    /// Serializes appends across JIT instances. Never taken by a reader.
    write_lock: Mutex<()>,
}

// SAFETY: readers only load `len` (Acquire) and then read slots `< len`, whose
// full initialization happens-before the Release store of `len`; writers hold
// `write_lock`, so no two threads write the same slot. Raw pointers are to
// never-freed heap.
unsafe impl Sync for CodeMap {}

impl CodeMap {
    const fn new() -> Self {
        CodeMap {
            chunks: [const { AtomicPtr::new(ptr::null_mut()) }; MAX_CHUNKS],
            len: AtomicUsize::new(0),
            write_lock: Mutex::new(()),
        }
    }

    /// Register a compiled function's host range and srcloc table. `table` must be
    /// sorted ascending by `host_off`, with each `rip_index` indexing `guest_rips`.
    /// Cold path (compile-time), so it may lock and allocate.
    pub fn register(
        &self,
        start: usize,
        code_len: u32,
        table: Box<[(u32, u32)]>,
        guest_rips: Box<[u64]>,
    ) {
        assert!(
            table.iter().all(|&(_, i)| (i as usize) < guest_rips.len()),
            "CodeMap RIP index out of bounds"
        );
        let _g = self.write_lock.lock().unwrap();
        let i = self.len.load(Ordering::Relaxed);
        let ci = i / CHUNK_CAP;
        assert!(ci < MAX_CHUNKS, "CodeMap capacity exhausted");
        let mut chunk = self.chunks[ci].load(Ordering::Acquire);
        if chunk.is_null() {
            chunk = Box::into_raw(Box::new(Chunk {
                entries: [const { UnsafeCell::new(Entry::EMPTY) }; CHUNK_CAP],
            }));
            self.chunks[ci].store(chunk, Ordering::Release);
        }
        let table_len = table.len();
        let table_ptr = Box::into_raw(table) as *const (u32, u32);
        let guest_rips = Box::into_raw(guest_rips) as *const u64;
        // SAFETY: slot `i` is exclusive (write_lock held) and not yet published;
        // it is fully written before the Release store of `len` below.
        unsafe {
            *(*chunk).entries[i % CHUNK_CAP].get() = Entry {
                start,
                end: start + code_len as usize,
                table: table_ptr,
                table_len,
                guest_rips,
            };
        }
        self.len.store(i + 1, Ordering::Release);
    }

    /// The guest RIP whose emitted code contains host `pc`, or `None` if `pc` is
    /// in no registered range. Async-signal-safe: atomic loads plus reads of
    /// never-freed memory, no lock, no allocation.
    pub fn lookup(&self, pc: usize) -> Option<u64> {
        let n = self.len.load(Ordering::Acquire);
        for i in 0..n {
            let chunk = self.chunks[i / CHUNK_CAP].load(Ordering::Acquire);
            if chunk.is_null() {
                continue;
            }
            // SAFETY: slot `i < n` was published (its write happens-before the
            // Acquire load of `len`); the chunk never moves or frees.
            let e = unsafe { *(*chunk).entries[i % CHUNK_CAP].get() };
            if pc >= e.start && pc < e.end {
                let off = (pc - e.start) as u32;
                // SAFETY: `table` is a leaked, never-freed boxed slice of `table_len`.
                let table = unsafe { slice::from_raw_parts(e.table, e.table_len) };
                // Largest `host_off <= off` → the guest RIP covering this PC.
                let idx = table.partition_point(|&(h, _)| h <= off);
                return (idx > 0).then(|| {
                    // SAFETY: register validated every index into the leaked RIP
                    // array, which was published together with this table.
                    unsafe { *e.guest_rips.add(table[idx - 1].1 as usize) }
                });
            }
        }
        None
    }
}

static CODE_MAP: CodeMap = CodeMap::new();

/// Register a compiled function in the process-global map (see [`CodeMap::register`]).
pub fn register(start: usize, code_len: u32, table: Box<[(u32, u32)]>, guest_rips: Box<[u64]>) {
    CODE_MAP.register(start, code_len, table, guest_rips);
}

/// Look a host PC up in the process-global map (see [`CodeMap::lookup`]).
pub fn lookup(pc: usize) -> Option<u64> {
    CODE_MAP.lookup(pc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_resolves_within_a_range_and_picks_the_covering_rip() {
        let map = CodeMap::new();
        // Host range [0x1000, 0x1040); two guest instructions at host offsets 0 and 0x10.
        map.register(
            0x1000,
            0x40,
            vec![(0, 0), (0x10, 1)].into_boxed_slice(),
            vec![0x400100, 0x400105].into_boxed_slice(),
        );
        assert_eq!(map.lookup(0x1000), Some(0x400100)); // first insn
        assert_eq!(map.lookup(0x100f), Some(0x400100)); // still first insn
        assert_eq!(map.lookup(0x1010), Some(0x400105)); // second insn
        assert_eq!(map.lookup(0x103f), Some(0x400105)); // tail of second insn
        assert_eq!(map.lookup(0x0fff), None); // below the range
        assert_eq!(map.lookup(0x1040), None); // one past the range
    }

    #[test]
    fn lookup_resolves_full_width_rips_by_index() {
        let map = CodeMap::new();
        map.register(
            0x1000,
            0x40,
            vec![(4, 1), (0x10, 0), (0x20, 1)].into_boxed_slice(),
            vec![u64::MAX, 0x1_ffff_ffff].into_boxed_slice(),
        );
        assert_eq!(map.lookup(0x1000), None);
        assert_eq!(map.lookup(0x1004), Some(0x1_ffff_ffff));
        assert_eq!(map.lookup(0x100f), Some(0x1_ffff_ffff));
        assert_eq!(map.lookup(0x1010), Some(u64::MAX));
        assert_eq!(map.lookup(0x1020), Some(0x1_ffff_ffff));
        assert_eq!(map.lookup(0x103f), Some(0x1_ffff_ffff));
        assert_eq!(map.lookup(0x1040), None);
    }

    #[test]
    #[should_panic(expected = "CodeMap RIP index out of bounds")]
    fn register_rejects_invalid_rip_indices() {
        CodeMap::new().register(
            0x1000,
            0x40,
            vec![(0, 1)].into_boxed_slice(),
            Box::new([42]),
        );
    }

    #[test]
    fn many_entries_span_multiple_chunks() {
        let map = CodeMap::new();
        for i in 0..(CHUNK_CAP + 5) {
            let base = 0x10_0000 + i * 0x10;
            map.register(
                base,
                0x10,
                vec![(0, 0)].into_boxed_slice(),
                vec![(0x40_0000 + i) as u64].into_boxed_slice(),
            );
        }
        // An entry in the first chunk and one in the second both resolve.
        assert_eq!(map.lookup(0x10_0000), Some(0x40_0000));
        let last = CHUNK_CAP + 4;
        assert_eq!(
            map.lookup(0x10_0000 + last * 0x10),
            Some((0x40_0000 + last) as u64)
        );
    }
}
