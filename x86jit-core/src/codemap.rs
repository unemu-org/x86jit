//! Process-global host-PC → guest-RIP side table for guard-page fault recovery.
//! Full-width guest RIPs are stored separately from compact `(host_off, rip_index)`
//! source locations. Lookup uses an immutable AVL tree of non-overlapping host
//! ranges, then binary-searches the function's source locations.
//!
//! Writers serialize under a mutex, copy only the insertion path, and publish a
//! fully initialized root with Release. Readers acquire ONE root and traverse only
//! immutable nodes: no allocation, lock, reference counting, or retry. Old paths,
//! source locations and RIP arrays are never freed, even after an SMC invalidation
//! or backend drop; a handler interrupted before publication can finish on its old
//! root. This matches the unreclaimed executable arena's lifetime. There is no
//! unregister operation. Lookup is async-signal-safe and O(log n + log srclocs).
//! Retained index storage is O(n log n), bounded by the registration capacity;
//! reclaiming translations and their metadata is separate work.
//!
//! Pure data — no OS dependencies, so the core stays `{iced-x86}` (§8.2).

use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Mutex;

const MAX_ENTRIES: usize = 1024 * 4096;

/// The process-global code map has reached its registration capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeMapFull;

#[derive(Clone, Copy)]
struct Entry {
    start: usize,
    end: usize,
    table: *const (u32, u32),
    table_len: usize,
    guest_rips: *const u64,
}

struct Node {
    entry: Entry,
    left: Option<&'static Node>,
    right: Option<&'static Node>,
    height: u8,
}

impl Node {
    fn height(node: Option<&Self>) -> u8 {
        node.map_or(0, |n| n.height)
    }

    fn new(
        entry: Entry,
        left: Option<&'static Self>,
        right: Option<&'static Self>,
    ) -> &'static Self {
        Box::leak(Box::new(Self {
            entry,
            left,
            right,
            height: 1 + Self::height(left).max(Self::height(right)),
        }))
    }

    fn balanced(
        entry: Entry,
        left: Option<&'static Self>,
        right: Option<&'static Self>,
    ) -> &'static Self {
        if Self::height(left) > Self::height(right) + 1 {
            let l = left.unwrap();
            if Self::height(l.left) >= Self::height(l.right) {
                return Self::new(l.entry, l.left, Some(Self::new(entry, l.right, right)));
            }
            let pivot = l.right.unwrap();
            return Self::new(
                pivot.entry,
                Some(Self::new(l.entry, l.left, pivot.left)),
                Some(Self::new(entry, pivot.right, right)),
            );
        }
        if Self::height(right) > Self::height(left) + 1 {
            let r = right.unwrap();
            if Self::height(r.right) >= Self::height(r.left) {
                return Self::new(r.entry, Some(Self::new(entry, left, r.left)), r.right);
            }
            let pivot = r.left.unwrap();
            return Self::new(
                pivot.entry,
                Some(Self::new(entry, left, pivot.left)),
                Some(Self::new(r.entry, pivot.right, r.right)),
            );
        }
        Self::new(entry, left, right)
    }

    fn insert(node: Option<&'static Self>, entry: Entry) -> &'static Self {
        let Some(n) = node else {
            return Self::new(entry, None, None);
        };
        if entry.end <= n.entry.start {
            Self::balanced(n.entry, Some(Self::insert(n.left, entry)), n.right)
        } else {
            assert!(entry.start >= n.entry.end, "CodeMap host ranges overlap");
            Self::balanced(n.entry, n.left, Some(Self::insert(n.right, entry)))
        }
    }
}

/// Append-only host-PC → guest-RIP map (see module docs for signal safety).
pub struct CodeMap {
    root: AtomicPtr<Node>,
    /// Only writers read or update the count; readers never take this mutex.
    len: Mutex<usize>,
    capacity: usize,
}

impl CodeMap {
    const fn new() -> Self {
        Self::with_capacity(MAX_ENTRIES)
    }

    const fn with_capacity(capacity: usize) -> Self {
        Self {
            root: AtomicPtr::new(ptr::null_mut()),
            len: Mutex::new(0),
            capacity,
        }
    }

    fn root(&self) -> Option<&'static Node> {
        // SAFETY: Release publication initializes the whole immutable tree before
        // this Acquire. Every reachable node is leaked; even old roots stay valid.
        unsafe { self.root.load(Ordering::Acquire).as_ref() }
    }

    /// Register a function. Host ranges must not overlap; `table` must be sorted
    /// ascending by host offset, with each index indexing `guest_rips`.
    /// Cold path: locks and allocates. Capacity exhaustion leaves the map unchanged.
    pub fn register(
        &self,
        start: usize,
        code_len: u32,
        table: Box<[(u32, u32)]>,
        guest_rips: Box<[u64]>,
    ) -> Result<(), CodeMapFull> {
        assert!(
            table.iter().all(|&(_, i)| (i as usize) < guest_rips.len()),
            "CodeMap RIP index out of bounds"
        );
        let mut len = self.len.lock().unwrap();
        if *len >= self.capacity {
            return Err(CodeMapFull);
        }
        let entry = Entry {
            start,
            end: start
                .checked_add(code_len as usize)
                .expect("CodeMap host range overflow"),
            table_len: table.len(),
            table: Box::into_raw(table) as *const (u32, u32),
            guest_rips: Box::into_raw(guest_rips) as *const u64,
        };
        let root = Node::insert(self.root(), entry);
        self.root
            .store(root as *const Node as *mut Node, Ordering::Release);
        *len += 1;
        Ok(())
    }

    /// Resolve a host PC. Allocation-free, lock-free and async-signal-safe.
    pub fn lookup(&self, pc: usize) -> Option<u64> {
        self.lookup_probed(pc, || {})
    }

    fn lookup_probed(&self, pc: usize, mut visit: impl FnMut()) -> Option<u64> {
        let mut node = self.root();
        while let Some(n) = node {
            visit();
            let e = n.entry;
            if pc < e.start {
                node = n.left;
            } else if pc >= e.end {
                node = n.right;
            } else {
                let off = (pc - e.start) as u32;
                // SAFETY: the leaked slice was published with the acquired root.
                let table = unsafe { slice::from_raw_parts(e.table, e.table_len) };
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
pub fn register(
    start: usize,
    code_len: u32,
    table: Box<[(u32, u32)]>,
    guest_rips: Box<[u64]>,
) -> Result<(), CodeMapFull> {
    CODE_MAP.register(start, code_len, table, guest_rips)
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
        )
        .unwrap();
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
        )
        .unwrap();
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
        CodeMap::new()
            .register(
                0x1000,
                0x40,
                vec![(0, 1)].into_boxed_slice(),
                Box::new([42]),
            )
            .unwrap();
    }

    #[test]
    fn many_entries_resolve() {
        let map = CodeMap::new();
        for i in 0..(1024 + 5) {
            let base = 0x10_0000 + i * 0x10;
            map.register(
                base,
                0x10,
                vec![(0, 0)].into_boxed_slice(),
                vec![(0x40_0000 + i) as u64].into_boxed_slice(),
            )
            .unwrap();
        }
        // Early and late registrations both resolve.
        assert_eq!(map.lookup(0x10_0000), Some(0x40_0000));
        let last = 1024 + 4;
        assert_eq!(
            map.lookup(0x10_0000 + last * 0x10),
            Some((0x40_0000 + last) as u64)
        );
    }
    #[test]
    fn lookup_cost_is_logarithmic() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut costs = Vec::new();
        for n in [128, 8192] {
            let map = CodeMap::new();
            for i in 0..n {
                map.register(0x1000 + i * 16, 8, Box::new([(0, 0)]), Box::new([i as u64]))
                    .unwrap();
            }
            let pc = 0x1000 + (n - 1) * 16;
            let mut visits = 0;
            assert_eq!(map.lookup_probed(pc, || visits += 1), Some((n - 1) as u64));
            let start = Instant::now();
            for _ in 0..20_000 {
                black_box(map.lookup(black_box(pc)));
            }
            eprintln!(
                "MEASURED: {n} functions: {visits} index visits, {:.1} ns/lookup",
                start.elapsed().as_nanos() as f64 / 20_000.0
            );
            costs.push(visits);
        }
        assert!(
            costs[1] <= costs[0] * 4,
            "64x entries must not cost 64x visits: {costs:?}"
        );
    }

    #[test]
    fn capacity_exhaustion_preserves_published_entries() {
        let map = CodeMap::with_capacity(2);
        for i in 0..2 {
            map.register(
                0x1000 + i * 16,
                8,
                Box::new([(0, 0)]),
                Box::new([u64::MAX - i as u64]),
            )
            .unwrap();
        }
        assert_eq!(
            map.register(0x2000, 8, Box::new([(0, 0)]), Box::new([42])),
            Err(CodeMapFull)
        );
        // A handler interrupted while a writer holds the mutex must still resolve.
        let _writer = map.len.lock().unwrap();
        assert_eq!(map.lookup(0x1000), Some(u64::MAX));
        assert_eq!(map.lookup(0x1010), Some(u64::MAX - 1));
        assert_eq!(map.lookup(0x2000), None);
    }

    #[test]
    fn indexed_lookup_matches_linear_reference_in_arbitrary_registration_order() {
        let map = CodeMap::new();
        let mut linear = Vec::new();
        // Odd multiplication permutes this power-of-two range, exercising both
        // insertion directions and all AVL rotation shapes.
        for i in 0..2048 {
            let start = 0x1000 + ((i * 137) % 2048) * 32;
            let rips = [u64::MAX - i as u64, 0x1_0000_0000 + i as u64];
            map.register(start, 24, Box::new([(4, 1), (12, 0)]), Box::new(rips))
                .unwrap();
            linear.push((start, rips));
        }
        for pc in (0x0ffc..0x1000 + 2048 * 32 + 4).step_by(4) {
            let expected = linear
                .iter()
                .find(|(start, _)| pc >= *start && pc < *start + 24)
                .and_then(|(start, rips)| match pc - start {
                    0..4 => None,
                    4..12 => Some(rips[1]),
                    _ => Some(rips[0]),
                });
            assert_eq!(map.lookup(pc), expected, "host PC {pc:#x}");
            let mut visits = 0;
            map.lookup_probed(pc, || visits += 1);
            assert!(visits <= 16, "unbalanced index: {visits} visits");
        }
    }

    #[test]
    fn lookup_during_concurrent_registration_sees_complete_entries() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Barrier;
        let map = CodeMap::new();
        let done = AtomicBool::new(false);
        let barrier = Barrier::new(2);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                barrier.wait();
                loop {
                    // Read done BEFORE scanning: the final scan must see all ranges.
                    // During earlier scans, even unregistered ranges are probed;
                    // no per-entry handshake can hide an early root publication.
                    let finished = done.load(Ordering::Acquire);
                    for i in 0..2048 {
                        for off in [0, 4, 11, 12, 23, 24, 31] {
                            let expected = match off {
                                4 | 11 => Some(u64::MAX - i as u64),
                                12 | 23 => Some(0x1_0000_0000 + i as u64),
                                _ => None,
                            };
                            let actual = map.lookup(0x1000 + i * 32 + off);
                            if finished || actual.is_some() {
                                assert_eq!(actual, expected, "range {i}, offset {off}");
                            }
                        }
                    }
                    if finished {
                        break;
                    }
                }
            });
            barrier.wait();
            for i in 0..2048 {
                map.register(
                    0x1000 + i * 32,
                    24,
                    Box::new([(4, 0), (12, 1)]),
                    Box::new([u64::MAX - i as u64, 0x1_0000_0000 + i as u64]),
                )
                .unwrap();
            }
            done.store(true, Ordering::Release);
        });
    }
}
