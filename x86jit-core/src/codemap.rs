//! Process-global host-PC → guest-RIP side table for guard-page fault recovery.
//! Full-width guest RIPs are stored separately from compact `(host_off, rip_index)`
//! source locations. Lookup uses an immutable AVL tree of non-overlapping host
//! ranges, then binary-searches the function's source locations or exact trap sites.
//!
//! Writers serialize under a mutex, copy only the insertion path, and publish a
//! fully initialized root with Release. Readers acquire ONE root and traverse only
//! immutable nodes: no allocation, lock, reference counting, or retry. Old paths,
//! source locations, RIP arrays and trap sites are never freed, even after an SMC invalidation
//! or backend drop; a handler interrupted before publication can finish on its old
//! root. This matches the unreclaimed executable arena's lifetime. There is no
//! unregister operation. Lookup is async-signal-safe and O(log n + log table size).
//! Retained index storage is O(n log n), bounded by the registration capacity;
//! reclaiming translations and their metadata is separate work.
//!
//! Pure data — no OS dependencies, so the core stays `{iced-x86}` (§8.2).

use crate::jit_abi::fault::FrameRecipe;
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Mutex;

const MAX_ENTRIES: usize = 1024 * 4096;
const MAX_TRAP_SITES: usize = 64 * 1024 * 1024;

/// Direction of a faulting transfer in compiled guest code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuestAccessKind {
    Read,
    Write,
    ReadWrite,
}

/// One potentially faulting host instruction accessing guest memory.
/// Split operands have a record for each host transfer, with that transfer's width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuestTrapSite {
    pub host_offset: u32,
    /// RIP of the access owner, independent of a merged consumer's source location.
    pub guest_rip: u64,
    pub access: GuestAccessKind,
    /// Transfer width in bytes.
    pub width: u8,
}

/// The process-global code map has reached its function or trap-site capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeMapFull;

/// Identity of the compiled unit published with its immutable code-map entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub enum FaultUnit {
    Block,
    Region { entry_rip: u64 },
}

#[derive(Clone, Copy)]
struct Entry {
    start: usize,
    end: usize,
    table: *const (u32, u32),
    table_len: usize,
    guest_rips: *const u64,
    traps: *const GuestTrapSite,
    traps_len: usize,
    recipe: Option<&'static FrameRecipe>,
    instructions: &'static [crate::fault::InstructionFaultInfo],
    unit: Option<FaultUnit>,
}

struct Counts {
    entries: usize,
    traps: usize,
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
    len: Mutex<Counts>,
    capacity: usize,
    trap_capacity: usize,
}

impl CodeMap {
    const fn new() -> Self {
        Self::with_capacity(MAX_ENTRIES)
    }

    const fn with_capacity(capacity: usize) -> Self {
        Self {
            root: AtomicPtr::new(ptr::null_mut()),
            len: Mutex::new(Counts {
                entries: 0,
                traps: 0,
            }),
            capacity,
            trap_capacity: MAX_TRAP_SITES,
        }
    }

    fn root(&self) -> Option<&'static Node> {
        #[cfg(test)]
        assert!(
            !CHILD_TEST_POISON.with(std::cell::Cell::get),
            "child consulted parent code map"
        );
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
        self.register_with_traps(start, code_len, table, guest_rips, Vec::new())
    }

    /// Register source locations and sorted, unique guest trap sites together.
    /// The caller allocates `traps` fallibly; registration retains its allocation
    /// without shrinking or copying it, charging its capacity (including spare slots).
    /// Either capacity limit leaves the map intact.
    pub fn register_with_traps(
        &self,
        start: usize,
        code_len: u32,
        table: Box<[(u32, u32)]>,
        guest_rips: Box<[u64]>,
        traps: Vec<GuestTrapSite>,
    ) -> Result<(), CodeMapFull> {
        self.register_with_recovery(start, code_len, table, guest_rips, traps, None)
    }

    /// Publish a function's unwind recipe and exact transfer records atomically.
    #[allow(clippy::too_many_arguments)]
    pub fn register_with_recovery(
        &self,
        start: usize,
        code_len: u32,
        table: Box<[(u32, u32)]>,
        guest_rips: Box<[u64]>,
        traps: Vec<GuestTrapSite>,
        recipe: Option<FrameRecipe>,
    ) -> Result<(), CodeMapFull> {
        self.register_with_fault_info(
            start,
            code_len,
            table,
            guest_rips,
            traps,
            recipe,
            Vec::new(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn register_with_fault_info(
        &self,
        start: usize,
        code_len: u32,
        table: Box<[(u32, u32)]>,
        guest_rips: Box<[u64]>,
        traps: Vec<GuestTrapSite>,
        recipe: Option<FrameRecipe>,
        instructions: Vec<crate::fault::InstructionFaultInfo>,
    ) -> Result<(), CodeMapFull> {
        self.register_with_fault_path(
            start,
            code_len,
            table,
            guest_rips,
            traps,
            recipe,
            instructions,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn register_with_fault_path(
        &self,
        start: usize,
        code_len: u32,
        table: Box<[(u32, u32)]>,
        guest_rips: Box<[u64]>,
        traps: Vec<GuestTrapSite>,
        recipe: Option<FrameRecipe>,
        instructions: Vec<crate::fault::InstructionFaultInfo>,
        unit: Option<FaultUnit>,
    ) -> Result<(), CodeMapFull> {
        assert!(
            table.iter().all(|&(_, i)| (i as usize) < guest_rips.len()),
            "CodeMap RIP index out of bounds"
        );
        assert!(traps
            .windows(2)
            .all(|s| s[0].host_offset < s[1].host_offset));
        assert!(traps.iter().all(|s| s.host_offset < code_len));
        let end = start
            .checked_add(code_len as usize)
            .expect("CodeMap host range overflow");
        let retained_traps = traps.capacity();
        let mut len = self.len.lock().unwrap();
        if len.entries >= self.capacity || retained_traps > self.trap_capacity - len.traps {
            return Err(CodeMapFull);
        }
        let entry = Entry {
            start,
            end,
            unit,
            table_len: table.len(),
            table: Box::into_raw(table) as *const (u32, u32),
            guest_rips: Box::into_raw(guest_rips) as *const u64,
            traps: traps.as_ptr(),
            traps_len: traps.len(),
            recipe: recipe.map(|r| &*Box::leak(Box::new(r))),
            instructions: Box::leak(instructions.into_boxed_slice()),
        };
        std::mem::forget(traps);
        let root = Node::insert(self.root(), entry);
        self.root
            .store(root as *const Node as *mut Node, Ordering::Release);
        len.entries += 1;
        len.traps += retained_traps;
        Ok(())
    }

    /// Resolve a host PC. Allocation-free, lock-free and async-signal-safe.
    pub fn lookup(&self, pc: usize) -> Option<u64> {
        self.lookup_probed(pc, || {})
    }

    fn lookup_probed(&self, pc: usize, mut visit: impl FnMut()) -> Option<u64> {
        let e = self.find_entry(pc, &mut visit)?;
        let off = (pc - e.start) as u32;
        // SAFETY: the leaked slice was published with the acquired root.
        let table = unsafe { slice::from_raw_parts(e.table, e.table_len) };
        let idx = table.partition_point(|&(h, _)| h <= off);
        (idx > 0).then(|| {
            // SAFETY: register validated every index into the leaked RIP
            // array, which was published together with this table.
            unsafe { *e.guest_rips.add(table[idx - 1].1 as usize) }
        })
    }

    /// Resolve an exact faulting host PC to its guest transfer. Nearby instructions
    /// do not inherit a trap site. Allocation-free, lock-free, async-signal-safe;
    /// O(log functions + log sites in the function).
    pub fn lookup_trap(&self, pc: usize) -> Option<GuestTrapSite> {
        let e = self.find_entry(pc, &mut || {})?;
        Self::trap_in_entry(e, pc)
    }

    fn trap_in_entry(e: Entry, pc: usize) -> Option<GuestTrapSite> {
        // SAFETY: registration leaks this immutable Vec allocation before Release
        // publication; find_entry acquired the root and the length travels with it.
        let traps = unsafe { slice::from_raw_parts(e.traps, e.traps_len) };
        let off = (pc - e.start) as u32;
        traps
            .binary_search_by_key(&off, |s| s.host_offset)
            .ok()
            .map(|i| traps[i])
    }

    fn lookup_recovery(&self, pc: usize) -> Option<(GuestTrapSite, &'static FrameRecipe)> {
        let entry = self.find_entry(pc, &mut || {})?;
        let recipe = entry.recipe?;
        if pc - entry.start < recipe.prologue_end as usize {
            return None;
        }
        Some((Self::trap_in_entry(entry, pc)?, recipe))
    }

    fn find_entry(&self, pc: usize, visit: &mut impl FnMut()) -> Option<Entry> {
        let mut node = self.root();
        while let Some(n) = node {
            visit();
            let e = n.entry;
            if pc < e.start {
                node = n.left;
            } else if pc >= e.end {
                node = n.right;
            } else {
                return Some(e);
            }
        }
        None
    }
}

#[cfg(test)]
std::thread_local! { static CHILD_TEST_POISON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
#[cfg(test)]
pub(crate) fn with_child_test_poison<T>(f: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            CHILD_TEST_POISON.with(|v| v.set(false));
        }
    }
    CHILD_TEST_POISON.with(|v| v.set(true));
    let _reset = Reset;
    f()
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

/// Register a function and its guest trap sites in the process-global map.
pub fn register_with_traps(
    start: usize,
    code_len: u32,
    table: Box<[(u32, u32)]>,
    guest_rips: Box<[u64]>,
    traps: Vec<GuestTrapSite>,
) -> Result<(), CodeMapFull> {
    CODE_MAP.register_with_traps(start, code_len, table, guest_rips, traps)
}

/// Resolve an exact faulting PC in the process-global map.
pub fn lookup_trap(pc: usize) -> Option<GuestTrapSite> {
    CODE_MAP.lookup_trap(pc)
}

/// Register immutable recovery metadata alongside a compiled function.
pub fn register_with_recovery(
    start: usize,
    code_len: u32,
    table: Box<[(u32, u32)]>,
    guest_rips: Box<[u64]>,
    traps: Vec<GuestTrapSite>,
    recipe: Option<FrameRecipe>,
) -> Result<(), CodeMapFull> {
    CODE_MAP.register_with_recovery(start, code_len, table, guest_rips, traps, recipe)
}

/// Publish cold per-instruction descriptors with the function and transfer table.
/// Existing registration APIs publish no descriptors and retain compatibility.
#[allow(clippy::too_many_arguments)]
pub fn register_with_fault_info(
    start: usize,
    code_len: u32,
    table: Box<[(u32, u32)]>,
    guest_rips: Box<[u64]>,
    traps: Vec<GuestTrapSite>,
    recipe: Option<FrameRecipe>,
    instructions: Vec<crate::fault::InstructionFaultInfo>,
) -> Result<(), CodeMapFull> {
    CODE_MAP.register_with_fault_info(
        start,
        code_len,
        table,
        guest_rips,
        traps,
        recipe,
        instructions,
    )
}

/// Publish execution identity alongside the existing fault descriptors and sites.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn register_with_fault_path(
    start: usize,
    code_len: u32,
    table: Box<[(u32, u32)]>,
    guest_rips: Box<[u64]>,
    traps: Vec<GuestTrapSite>,
    recipe: Option<FrameRecipe>,
    instructions: Vec<crate::fault::InstructionFaultInfo>,
    unit: FaultUnit,
) -> Result<(), CodeMapFull> {
    CODE_MAP.register_with_fault_path(
        start,
        code_len,
        table,
        guest_rips,
        traps,
        recipe,
        instructions,
        Some(unit),
    )
}

#[cold]
pub(crate) fn fault_path(pc: usize, rip: crate::Knowledge<u64>, helper: bool) -> crate::FaultPath {
    use crate::{FaultPath, HelperCaller};
    let Some(entry) = CODE_MAP.find_entry(pc, &mut || {}) else {
        return FaultPath::Unknown;
    };
    let caller = match entry.unit {
        Some(FaultUnit::Block) => HelperCaller::CompiledBlock { host_pc: pc },
        Some(FaultUnit::Region { entry_rip }) => HelperCaller::Region {
            entry_rip,
            host_pc: pc,
        },
        None => return FaultPath::Unknown,
    };
    if helper {
        let family = entry
            .instructions
            .iter()
            .find(|i| crate::Knowledge::Known(i.rip) == rip)
            .and_then(|i| i.helper);
        return family.map_or(FaultPath::Unknown, |family| FaultPath::Helper {
            family,
            caller,
        });
    }
    let site = CODE_MAP.lookup_trap(pc);
    match caller {
        HelperCaller::CompiledBlock { host_pc } => FaultPath::CompiledBlock { host_pc, site },
        HelperCaller::Region { entry_rip, host_pc } => FaultPath::Region {
            entry_rip,
            host_pc,
            site,
        },
        _ => unreachable!(),
    }
}

/// Cold lookup by function/site PC and full-width instruction identity.
#[doc(hidden)]
pub fn lookup_fault_info(
    pc: usize,
    rip: u64,
) -> Option<&'static crate::fault::InstructionFaultInfo> {
    CODE_MAP
        .find_entry(pc, &mut || {})?
        .instructions
        .iter()
        .find(|i| i.rip == rip)
}

/// Exact transfer and one-frame unwind recipe, acquired from the immutable map.
pub fn lookup_recovery(pc: usize) -> Option<(GuestTrapSite, &'static FrameRecipe)> {
    CODE_MAP.lookup_recovery(pc)
}

/// Whether an address belongs to any registered compiled function, even one with
/// no source locations, transfer records, or supported unwind recipe.
pub fn contains(pc: usize) -> bool {
    CODE_MAP.find_entry(pc, &mut || {}).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trap_lookup_is_exact_and_keeps_split_transfer_records() {
        let map = CodeMap::new();
        let rip = 0x1234_5678_ffff_ffff;
        let sites = vec![
            GuestTrapSite {
                host_offset: 4,
                guest_rip: rip,
                access: GuestAccessKind::Read,
                width: 8,
            },
            GuestTrapSite {
                host_offset: 8,
                guest_rip: rip,
                access: GuestAccessKind::Read,
                width: 8,
            },
            GuestTrapSite {
                host_offset: 16,
                guest_rip: rip + (1 << 32),
                access: GuestAccessKind::ReadWrite,
                width: 4,
            },
        ];
        map.register_with_traps(
            0x1000,
            24,
            Box::new([(0, 0), (12, 1)]),
            Box::new([rip, rip + (1 << 32)]),
            sites.clone(),
        )
        .unwrap();
        // Interrupting the writer must not prevent either lookup from completing.
        let _writer = map.len.lock().unwrap();
        for off in 0..32 {
            let expected = sites
                .iter()
                .find(|s| s.host_offset as usize == off)
                .copied();
            assert_eq!(map.lookup_trap(0x1000 + off), expected);
            if let Some(s) = expected {
                assert_eq!(map.lookup(0x1000 + off), Some(s.guest_rip));
            }
        }
        assert_eq!(map.lookup_trap(0xfff), None);
    }

    #[test]
    fn trap_capacity_exhaustion_preserves_the_published_tree() {
        let mut map = CodeMap::with_capacity(3);
        map.trap_capacity = 1;
        let site = GuestTrapSite {
            host_offset: 4,
            guest_rip: u64::MAX,
            access: GuestAccessKind::Write,
            width: 2,
        };
        map.register_with_traps(
            0x1000,
            8,
            Box::new([(0, 0)]),
            Box::new([u64::MAX]),
            vec![site],
        )
        .unwrap();
        let root = map.root.load(Ordering::Relaxed);
        assert_eq!(
            map.register_with_traps(
                0x2000,
                8,
                Box::new([(0, 0)]),
                Box::new([u64::MAX]),
                vec![site]
            ),
            Err(CodeMapFull)
        );
        assert_eq!(map.root.load(Ordering::Relaxed), root);
        assert_eq!(map.lookup_trap(0x1004), Some(site));
        assert_eq!(map.lookup_trap(0x2004), None);
        let counts = map.len.lock().unwrap();
        assert_eq!(counts.entries, 1);
        assert_eq!(counts.traps, 1);
    }

    #[test]
    fn trap_capacity_charges_retained_allocation() {
        let mut map = CodeMap::with_capacity(2);
        map.trap_capacity = 2;
        let mut traps = Vec::with_capacity(3);
        traps.push(GuestTrapSite {
            host_offset: 0,
            guest_rip: 1,
            access: GuestAccessKind::Read,
            width: 1,
        });
        assert_eq!(
            map.register_with_traps(0x1000, 8, Box::new([]), Box::new([]), traps),
            Err(CodeMapFull)
        );
        assert!(map.root().is_none());

        map.trap_capacity = 4;
        let site = GuestTrapSite {
            host_offset: 0,
            guest_rip: 1,
            access: GuestAccessKind::Read,
            width: 1,
        };
        let mut traps = Vec::with_capacity(3);
        traps.push(site);
        map.register_with_traps(0x1000, 8, Box::new([]), Box::new([]), traps)
            .unwrap();
        assert_eq!(map.len.lock().unwrap().traps, 3);
        let mut traps = Vec::with_capacity(2);
        traps.push(site);
        assert_eq!(
            map.register_with_traps(0x2000, 8, Box::new([]), Box::new([]), traps),
            Err(CodeMapFull)
        );
        assert_eq!(map.lookup_trap(0x1000), Some(site));
        assert_eq!(map.lookup_trap(0x2000), None);
    }

    #[test]
    fn invalid_trap_registration_does_not_poison_writer() {
        let map = CodeMap::new();
        let invalid = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            map.register_with_traps(
                0x1000,
                8,
                Box::new([]),
                Box::new([]),
                vec![GuestTrapSite {
                    host_offset: 8,
                    guest_rip: 1,
                    access: GuestAccessKind::Read,
                    width: 1,
                }],
            )
        }));
        assert!(invalid.is_err());
        assert_eq!(map.register(0x1000, 8, Box::new([]), Box::new([])), Ok(()));
    }

    #[test]
    fn trap_lookup_uses_the_logarithmic_function_index_and_sorted_sites() {
        let map = CodeMap::new();
        for i in 0..2048 {
            let start = 0x1000 + ((i * 137) % 2048) * 4096;
            let sites = (0..256)
                .map(|j| GuestTrapSite {
                    host_offset: j * 4,
                    guest_rip: u64::MAX - i as u64,
                    access: GuestAccessKind::Read,
                    width: 1 << (j % 4),
                })
                .collect();
            map.register_with_traps(
                start,
                2048,
                Box::new([(0, 0)]),
                Box::new([u64::MAX - i as u64]),
                sites,
            )
            .unwrap();
        }
        for i in 0..2048 {
            let start = 0x1000 + ((i * 137) % 2048) * 4096;
            let mut visits = 0;
            assert!(map.find_entry(start, &mut || visits += 1).is_some());
            assert!(visits <= 16, "unbalanced index: {visits} visits");
            for j in 0..256 {
                assert_eq!(
                    map.lookup_trap(start + j * 4),
                    Some(GuestTrapSite {
                        host_offset: (j * 4) as u32,
                        guest_rip: u64::MAX - i as u64,
                        access: GuestAccessKind::Read,
                        width: 1 << (j % 4),
                    })
                );
                assert_eq!(map.lookup_trap(start + j * 4 + 1), None);
            }
            assert_eq!(map.lookup_trap(start + 2048), None);
        }
    }

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
                            let trap = map.lookup_trap(0x1000 + i * 32 + off);
                            if finished || trap.is_some() {
                                let expected = (off == 4).then_some(GuestTrapSite {
                                    host_offset: 4,
                                    guest_rip: u64::MAX - i as u64,
                                    access: GuestAccessKind::Read,
                                    width: 8,
                                });
                                assert_eq!(trap, expected);
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
                map.register_with_traps(
                    0x1000 + i * 32,
                    24,
                    Box::new([(4, 0), (12, 1)]),
                    Box::new([u64::MAX - i as u64, 0x1_0000_0000 + i as u64]),
                    vec![GuestTrapSite {
                        host_offset: 4,
                        guest_rip: u64::MAX - i as u64,
                        access: GuestAccessKind::Read,
                        width: 8,
                    }],
                )
                .unwrap();
            }
            done.store(true, Ordering::Release);
        });
    }
}
