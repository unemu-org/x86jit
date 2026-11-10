//! Process-global host-PC → guest-RIP side table for guard-page fault recovery.
//! Writers publish immutable AVL roots (§8.2; decision-13). Only writers own
//! reference counts. A reader enters a sequentially consistent counter before
//! loading the root and leaves after copying its result. Writers retain old roots
//! until a zero-reader observation after publication. Thus a reader that can load
//! an old root is counted before reclamation; a later reader can only load the
//! new root. All three operations must be SeqCst, including on weakly ordered hosts.
//!
//! A signal interrupting publication can finish on either root, even when the
//! interrupted thread holds the writer mutex or is freeing an older tree. Readers
//! never lock, allocate, free, touch reference counts, or wait for a writer. Nested
//! handlers are counted independently. Only writer/cold maintenance paths reclaim.
//! Returned values own their data; no reference escapes a reader critical section.
//! Normal signal return/unwinding must run the guard's destructor; abandoning a
//! handler with a nonlocal jump would pin the grace period (and is unsupported).
//!
//! Live storage is O(n); unreclaimed paths depend on writes during a reader grace
//! period. Invalidation does not remove ranges: executable code is still retained.
//! Removal is reserved for callers that have established execution quiescence.

use crate::jit_abi::fault::FrameRecipe;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Default)]
struct Accounting {
    nodes: AtomicU64,
    entries: AtomicU64,
    sources: AtomicU64,
    traps: AtomicU64,
    instructions: AtomicU64,
}

/// Retained allocation sizes, excluding allocator bookkeeping. Shared payloads
/// are charged once, including payloads still held by deferred roots.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryStats {
    pub nodes: u64,
    pub entries: u64,
    pub sources: u64,
    pub traps: u64,
    pub recipes: u64,
    pub instructions: u64,
    pub writer_storage: u64,
}
impl MemoryStats {
    pub fn total(self) -> u64 {
        self.nodes
            + self.entries
            + self.sources
            + self.traps
            + self.recipes
            + self.instructions
            + self.writer_storage
    }
}

const ARC_HEADER: usize = 2 * std::mem::size_of::<usize>();
const fn arc_bytes<T>() -> usize {
    let alignment = if std::mem::align_of::<T>() > std::mem::align_of::<usize>() {
        std::mem::align_of::<T>()
    } else {
        std::mem::align_of::<usize>()
    };
    (std::mem::size_of::<T>() + ARC_HEADER + alignment - 1) & !(alignment - 1)
}

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

// Fixed-width little-endian columns preserve binary search while choosing the
// smallest width needed by each unit. RIP deltas wrap so arbitrary full-width
// instruction identities (including around u64::MAX) remain lossless.
struct Metadata {
    bytes: Box<[u8]>,
    rips: u32,
    sources: u32,
    traps: u32,
    host_width: u8,
    index_width: u8,
    rip_width: u8,
    base_rip: u64,
}
impl Metadata {
    fn width(n: u64) -> u8 {
        ((64 - n.leading_zeros()).max(1).div_ceil(8)) as u8
    }
    fn push(bytes: &mut Vec<u8>, n: u64, width: u8) {
        bytes.extend_from_slice(&n.to_le_bytes()[..width as usize]);
    }
    fn get(&self, offset: usize, width: u8) -> u64 {
        let mut n = [0; 8];
        n[..width as usize].copy_from_slice(&self.bytes[offset..offset + width as usize]);
        u64::from_le_bytes(n)
    }
    fn new(table: &[(u32, u32)], rips: &[u64], traps: &[GuestTrapSite]) -> Self {
        let mut all_rips = rips.to_vec();
        let mut indexes: std::collections::HashMap<u64, u32> = rips
            .iter()
            .enumerate()
            .map(|(i, &rip)| (rip, i as u32))
            .collect();
        let trap_indexes: Vec<_> = traps
            .iter()
            .map(|s| {
                *indexes.entry(s.guest_rip).or_insert_with(|| {
                    let i = u32::try_from(all_rips.len()).expect("CodeMap RIP table too large");
                    all_rips.push(s.guest_rip);
                    i
                })
            })
            .collect();
        let base_rip = all_rips.first().copied().unwrap_or(0);
        let rip_width = Self::width(
            all_rips
                .iter()
                .map(|r| r.wrapping_sub(base_rip))
                .max()
                .unwrap_or(0),
        );
        let host_width = Self::width(
            table
                .iter()
                .map(|t| t.0)
                .chain(traps.iter().map(|s| s.host_offset))
                .max()
                .unwrap_or(0) as u64,
        );
        let index_width = Self::width(all_rips.len().saturating_sub(1) as u64);
        let mut bytes = Vec::new();
        for rip in &all_rips {
            Self::push(&mut bytes, rip.wrapping_sub(base_rip), rip_width);
        }
        for &(offset, index) in table {
            Self::push(&mut bytes, offset as u64, host_width);
            Self::push(&mut bytes, index as u64, index_width);
        }
        for (s, index) in traps.iter().zip(trap_indexes) {
            Self::push(&mut bytes, s.host_offset as u64, host_width);
            Self::push(&mut bytes, index as u64, index_width);
            bytes.push(s.access as u8);
            bytes.push(s.width);
        }
        Self {
            bytes: bytes.into_boxed_slice(),
            rips: u32::try_from(all_rips.len()).unwrap(),
            sources: u32::try_from(table.len()).unwrap(),
            traps: u32::try_from(traps.len()).unwrap(),
            host_width,
            index_width,
            rip_width,
            base_rip,
        }
    }
    fn source_bytes(&self) -> usize {
        self.rips as usize * self.rip_width as usize
            + self.sources as usize * (self.host_width + self.index_width) as usize
    }
    fn rip(&self, offset: usize) -> u64 {
        let index = self.get(offset, self.index_width) as usize;
        self.base_rip
            .wrapping_add(self.get(index * self.rip_width as usize, self.rip_width))
    }
    fn source(&self, off: u32) -> Option<u64> {
        let base = self.rips as usize * self.rip_width as usize;
        let stride = (self.host_width + self.index_width) as usize;
        let mut lo = 0;
        let mut hi = self.sources as usize;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.get(base + mid * stride, self.host_width) <= off as u64 {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo > 0).then(|| self.rip(base + (lo - 1) * stride + self.host_width as usize))
    }
    fn trap(&self, off: u32) -> Option<GuestTrapSite> {
        let base = self.source_bytes();
        let stride = (self.host_width + self.index_width + 2) as usize;
        let mut lo = 0;
        let mut hi = self.traps as usize;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let at = base + mid * stride;
            match self.get(at, self.host_width).cmp(&(off as u64)) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let tail = at + (self.host_width + self.index_width) as usize;
                    return Some(GuestTrapSite {
                        host_offset: off,
                        guest_rip: self.rip(at + self.host_width as usize),
                        access: match self.bytes[tail] {
                            0 => GuestAccessKind::Read,
                            1 => GuestAccessKind::Write,
                            _ => GuestAccessKind::ReadWrite,
                        },
                        width: self.bytes[tail + 1],
                    });
                }
            }
        }
        None
    }
}

struct Instruction {
    info: crate::fault::InstructionFaultInfo,
    accounting: Arc<Accounting>,
}
impl Instruction {
    fn bytes(&self) -> usize {
        arc_bytes::<Self>() + self.info.retained_address_bytes()
    }
    fn new(info: crate::fault::InstructionFaultInfo, accounting: Arc<Accounting>) -> Arc<Self> {
        let value = Self { info, accounting };
        value
            .accounting
            .instructions
            .fetch_add(value.bytes() as u64, Ordering::Relaxed);
        Arc::new(value)
    }
}
impl Drop for Instruction {
    fn drop(&mut self) {
        self.accounting
            .instructions
            .fetch_sub(self.bytes() as u64, Ordering::Relaxed);
    }
}
struct UnitInstruction {
    rip: u64,
    shared: Arc<Instruction>,
}

struct Entry {
    start: usize,
    end: usize,
    metadata: Metadata,
    recipe: Option<Arc<FrameRecipe>>,
    instructions: Box<[UnitInstruction]>,
    unit: Option<FaultUnit>,
    accounting: Arc<Accounting>,
}
impl Entry {
    fn account(&self, add: bool) {
        let a = &self.accounting;
        let sources = self.metadata.source_bytes();
        for (counter, bytes) in [
            (&a.entries, arc_bytes::<Self>()),
            (&a.sources, sources),
            (&a.traps, self.metadata.bytes.len() - sources),
            (&a.instructions, std::mem::size_of_val(&*self.instructions)),
        ] {
            if add {
                counter.fetch_add(bytes as u64, Ordering::Relaxed);
            } else {
                counter.fetch_sub(bytes as u64, Ordering::Relaxed);
            }
        }
    }
}
impl Drop for Entry {
    fn drop(&mut self) {
        self.account(false);
    }
}

struct Counts {
    entries: usize,
    traps: usize,
    root: Option<Arc<Node>>,
    retired: Vec<Arc<Node>>,
    recipes: Vec<Arc<FrameRecipe>>,
    instructions: Vec<Arc<Instruction>>,
}

struct Node {
    entry: Arc<Entry>,
    left: Option<Arc<Node>>,
    right: Option<Arc<Node>>,
    height: u8,
}
impl Drop for Node {
    fn drop(&mut self) {
        self.entry
            .accounting
            .nodes
            .fetch_sub(arc_bytes::<Self>() as u64, Ordering::Relaxed);
    }
}
impl Node {
    fn height(node: &Option<Arc<Self>>) -> u8 {
        node.as_ref().map_or(0, |n| n.height)
    }
    fn new(entry: Arc<Entry>, left: Option<Arc<Self>>, right: Option<Arc<Self>>) -> Arc<Self> {
        entry
            .accounting
            .nodes
            .fetch_add(arc_bytes::<Self>() as u64, Ordering::Relaxed);
        let height = 1 + Self::height(&left).max(Self::height(&right));
        Arc::new(Self {
            entry,
            left,
            right,
            height,
        })
    }
    fn balanced(entry: Arc<Entry>, left: Option<Arc<Self>>, right: Option<Arc<Self>>) -> Arc<Self> {
        if Self::height(&left) > Self::height(&right) + 1 {
            let l = left.as_ref().unwrap();
            if Self::height(&l.left) >= Self::height(&l.right) {
                return Self::new(
                    l.entry.clone(),
                    l.left.clone(),
                    Some(Self::new(entry, l.right.clone(), right)),
                );
            }
            let p = l.right.as_ref().unwrap();
            return Self::new(
                p.entry.clone(),
                Some(Self::new(l.entry.clone(), l.left.clone(), p.left.clone())),
                Some(Self::new(entry, p.right.clone(), right)),
            );
        }
        if Self::height(&right) > Self::height(&left) + 1 {
            let r = right.as_ref().unwrap();
            if Self::height(&r.right) >= Self::height(&r.left) {
                return Self::new(
                    r.entry.clone(),
                    Some(Self::new(entry, left, r.left.clone())),
                    r.right.clone(),
                );
            }
            let p = r.left.as_ref().unwrap();
            return Self::new(
                p.entry.clone(),
                Some(Self::new(entry, left, p.left.clone())),
                Some(Self::new(r.entry.clone(), p.right.clone(), r.right.clone())),
            );
        }
        Self::new(entry, left, right)
    }
    fn insert(node: &Option<Arc<Self>>, entry: Arc<Entry>) -> Arc<Self> {
        let Some(n) = node else {
            return Self::new(entry, None, None);
        };
        if entry.end <= n.entry.start {
            Self::balanced(
                n.entry.clone(),
                Some(Self::insert(&n.left, entry)),
                n.right.clone(),
            )
        } else {
            assert!(entry.start >= n.entry.end, "CodeMap host ranges overlap");
            Self::balanced(
                n.entry.clone(),
                n.left.clone(),
                Some(Self::insert(&n.right, entry)),
            )
        }
    }
    fn remove(node: &Arc<Self>, start: usize) -> Option<Arc<Self>> {
        if start < node.entry.start {
            return Some(Self::balanced(
                node.entry.clone(),
                Self::remove(node.left.as_ref().unwrap(), start),
                node.right.clone(),
            ));
        }
        if start > node.entry.start {
            return Some(Self::balanced(
                node.entry.clone(),
                node.left.clone(),
                Self::remove(node.right.as_ref().unwrap(), start),
            ));
        }
        match (&node.left, &node.right) {
            (None, right) => right.clone(),
            (left, None) => left.clone(),
            (left, Some(right)) => {
                let mut successor = right;
                while let Some(l) = &successor.left {
                    successor = l;
                }
                Some(Self::balanced(
                    successor.entry.clone(),
                    left.clone(),
                    Self::remove(right, successor.entry.start),
                ))
            }
        }
    }
}

struct ReadGuard<'a>(&'a CodeMap);
impl ReadGuard<'_> {
    fn root(&self) -> Option<&Node> {
        #[cfg(test)]
        assert!(
            !CHILD_TEST_POISON.with(std::cell::Cell::get),
            "child consulted parent code map"
        );
        // SAFETY: this reference cannot outlive the guard. The current root
        // and every root retired since reader entry still have writer owners.
        unsafe { self.0.root.load(Ordering::SeqCst).as_ref() }
    }

    fn find_entry(&self, pc: usize, visit: &mut impl FnMut()) -> Option<&Entry> {
        let mut node = self.root();
        while let Some(n) = node {
            visit();
            let e = &*n.entry;
            if pc < e.start {
                node = n.left.as_deref();
            } else if pc >= e.end {
                node = n.right.as_deref();
            } else {
                return Some(e);
            }
        }
        None
    }
}
impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        self.0.readers.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Host-PC → guest-RIP map (see module docs for signal safety).
pub struct CodeMap {
    root: AtomicPtr<Node>,
    readers: AtomicUsize,
    accounting: OnceLock<Arc<Accounting>>,
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
            readers: AtomicUsize::new(0),
            accounting: OnceLock::new(),
            len: Mutex::new(Counts {
                entries: 0,
                traps: 0,
                root: None,
                retired: Vec::new(),
                recipes: Vec::new(),
                instructions: Vec::new(),
            }),
            capacity,
            trap_capacity: MAX_TRAP_SITES,
        }
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
    /// Spare trap capacity is discarded; tables use dense per-unit widths.
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
        // The compact per-unit tables store counts as u32: reject larger inputs here,
        // before the writer lock, so a bad registration cannot poison the map.
        assert!(
            u32::try_from(guest_rips.len()).is_ok()
                && u32::try_from(table.len()).is_ok()
                && u32::try_from(traps.len()).is_ok(),
            "CodeMap per-unit table count exceeds u32"
        );
        let end = start
            .checked_add(code_len as usize)
            .expect("CodeMap host range overflow");
        let retained_traps = traps.len();
        let mut len = self.len.lock().unwrap();
        if len.entries >= self.capacity || retained_traps > self.trap_capacity - len.traps {
            return Err(CodeMapFull);
        }
        let accounting = self
            .accounting
            .get_or_init(|| Arc::new(Accounting::default()))
            .clone();
        let recipe =
            recipe.map(
                |r| match len.recipes.binary_search_by(|prior| (**prior).cmp(&r)) {
                    Ok(i) => len.recipes[i].clone(),
                    Err(i) => {
                        let recipe = Arc::new(r);
                        len.recipes.insert(i, recipe.clone());
                        recipe
                    }
                },
            );
        let instructions = instructions
            .into_iter()
            .map(|info| {
                let rip = info.rip;
                let shared = if let Some(key) = info.shared_key() {
                    match len
                        .instructions
                        .binary_search_by_key(&key, |prior| prior.info.shared_key().unwrap())
                    {
                        Ok(i) => len.instructions[i].clone(),
                        Err(i) => {
                            let shared = Instruction::new(info, accounting.clone());
                            len.instructions.insert(i, shared.clone());
                            shared
                        }
                    }
                } else {
                    Instruction::new(info, accounting.clone())
                };
                UnitInstruction { rip, shared }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let entry = Arc::new(Entry {
            start,
            end,
            unit,
            metadata: Metadata::new(&table, &guest_rips, &traps),
            recipe,
            instructions,
            accounting,
        });
        entry.account(true);
        let root = Some(Node::insert(&len.root, entry));
        self.publish(&mut len, root);
        len.entries += 1;
        len.traps += retained_traps;
        Ok(())
    }

    fn read(&self) -> ReadGuard<'_> {
        self.readers.fetch_add(1, Ordering::SeqCst);
        ReadGuard(self)
    }

    fn publish(&self, len: &mut Counts, root: Option<Arc<Node>>) {
        self.publish_probed(len, root, || {});
    }

    fn publish_probed(&self, len: &mut Counts, root: Option<Arc<Node>>, mut probe: impl FnMut()) {
        let raw = root
            .as_ref()
            .map_or(ptr::null_mut(), |n| Arc::as_ptr(n).cast_mut());
        // Keep ownership of both roots across publication, even if interrupted.
        probe();
        self.root.store(raw, Ordering::SeqCst);
        probe();
        if let Some(old) = std::mem::replace(&mut len.root, root) {
            len.retired.push(old);
        }
        probe();
        self.reclaim(len);
        probe();
    }

    fn reclaim(&self, len: &mut Counts) {
        if self.readers.load(Ordering::SeqCst) == 0 {
            len.retired.clear();
            len.retired.shrink_to(4);
        }
    }

    /// Remove exactly one unit range. The caller must first prevent new execution
    /// and wait for existing executions (including their cold fault paths) to end.
    /// Removed metadata stays deferred until a subsequent write or memory_stats
    /// observes signal-reader quiescence. This does not establish execution
    /// quiescence and is deliberately unused by invalidation. Returns false for
    /// a missing range or a length mismatch.
    pub fn unregister(&self, start: usize, code_len: u32) -> bool {
        let Some(end) = start.checked_add(code_len as usize) else {
            return false;
        };
        let mut len = self.len.lock().unwrap();
        let guard = self.read();
        let Some(entry) = guard.find_entry(start, &mut || {}) else {
            return false;
        };
        if entry.start != start || entry.end != end {
            return false;
        }
        let traps = entry.metadata.traps as usize;
        drop(guard);
        let root = Node::remove(len.root.as_ref().unwrap(), start);
        self.publish(&mut len, root);
        len.entries -= 1;
        len.traps -= traps;
        Self::prune_pools(&mut len);
        true
    }

    fn prune_pools(len: &mut Counts) {
        len.recipes.retain(|r| Arc::strong_count(r) > 1);
        len.instructions.retain(|i| Arc::strong_count(i) > 1);
        len.recipes.shrink_to_fit();
        len.instructions.shrink_to_fit();
    }

    /// Reclaim obsolete versions after a signal-reader grace period and sample
    /// retained bytes. This cold operation takes the writer mutex.
    pub fn memory_stats(&self) -> MemoryStats {
        let mut len = self.len.lock().unwrap();
        self.reclaim(&mut len);
        Self::prune_pools(&mut len);
        let Some(a) = self.accounting.get() else {
            return MemoryStats::default();
        };
        MemoryStats {
            nodes: a.nodes.load(Ordering::Relaxed),
            entries: a.entries.load(Ordering::Relaxed),
            sources: a.sources.load(Ordering::Relaxed),
            traps: a.traps.load(Ordering::Relaxed),
            recipes: (len.recipes.len() * arc_bytes::<FrameRecipe>()) as u64,
            instructions: a.instructions.load(Ordering::Relaxed),
            writer_storage: ((len.retired.capacity()
                + len.recipes.capacity()
                + len.instructions.capacity())
                * std::mem::size_of::<Arc<Node>>()
                + arc_bytes::<Accounting>()) as u64,
        }
    }

    /// Resolve a host PC. Allocation-free, lock-free and async-signal-safe.
    pub fn lookup(&self, pc: usize) -> Option<u64> {
        self.lookup_probed(pc, || {})
    }

    fn lookup_probed(&self, pc: usize, mut visit: impl FnMut()) -> Option<u64> {
        let guard = self.read();
        let e = guard.find_entry(pc, &mut visit)?;
        let off = (pc - e.start) as u32;
        e.metadata.source(off)
    }

    /// Resolve an exact transfer. Lock-free and allocation-free, including when
    /// this interrupts a writer on the same thread.
    pub fn lookup_trap(&self, pc: usize) -> Option<GuestTrapSite> {
        let guard = self.read();
        Self::trap_in_entry(guard.find_entry(pc, &mut || {})?, pc)
    }

    fn trap_in_entry(e: &Entry, pc: usize) -> Option<GuestTrapSite> {
        let off = (pc - e.start) as u32;
        e.metadata.trap(off)
    }

    fn lookup_recovery(&self, pc: usize) -> Option<(GuestTrapSite, FrameRecipe)> {
        let guard = self.read();
        let entry = guard.find_entry(pc, &mut || {})?;
        let recipe = **entry.recipe.as_ref()?;
        if pc - entry.start < recipe.prologue_end as usize {
            return None;
        }
        Some((Self::trap_in_entry(entry, pc)?, recipe))
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

/// Process-wide retained allocations, including deferred reader versions.
pub fn memory_bytes() -> u64 {
    CODE_MAP.memory_stats().total()
}

/// Process-wide retained allocation breakdown.
pub fn memory_stats() -> MemoryStats {
    CODE_MAP.memory_stats()
}

/// Remove a retired unit after execution quiescence (see [`CodeMap::unregister`]).
pub fn unregister(start: usize, code_len: u32) -> bool {
    CODE_MAP.unregister(start, code_len)
}

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
    let guard = CODE_MAP.read();
    let Some(entry) = guard.find_entry(pc, &mut || {}) else {
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
            .and_then(|i| i.shared.info.helper);
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
/// Returns an owned descriptor so removal cannot invalidate the result. This
/// clone can allocate; unlike recovery lookup, this is not a signal-handler API.
#[doc(hidden)]
pub fn lookup_fault_info(pc: usize, rip: u64) -> Option<crate::fault::InstructionFaultInfo> {
    CODE_MAP
        .read()
        .find_entry(pc, &mut || {})?
        .instructions
        .iter()
        .find(|i| i.rip == rip)
        .map(|i| {
            let mut info = i.shared.info.clone();
            info.rip = i.rip;
            info
        })
}

/// Exact transfer and one-frame unwind recipe, acquired from the immutable map.
pub fn lookup_recovery(pc: usize) -> Option<(GuestTrapSite, FrameRecipe)> {
    CODE_MAP.lookup_recovery(pc)
}

/// Whether an address belongs to any registered compiled function, even one with
/// no source locations, transfer records, or supported unwind recipe.
pub fn contains(pc: usize) -> bool {
    CODE_MAP.read().find_entry(pc, &mut || {}).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe() -> FrameRecipe {
        FrameRecipe {
            prologue_end: 0,
            return_auth: 0,
            caller_sp: 16,
            saved_fp: 0,
            saved_return: 8,
            gpr: [None; 10],
            simd: [None; 8],
        }
    }

    #[test]
    fn signal_like_lookup_at_each_publication_boundary() {
        let map = CodeMap::new();
        map.register_with_recovery(
            0x1000,
            16,
            Box::new([(0, 0)]),
            Box::new([7]),
            vec![GuestTrapSite {
                host_offset: 4,
                guest_rip: 7,
                access: GuestAccessKind::Write,
                width: 8,
            }],
            Some(recipe()),
        )
        .unwrap();
        let mut writer = map.len.lock().unwrap();
        let old_root = writer.root.as_ref().unwrap().clone();
        let new_root = Node::new(old_root.entry.clone(), None, None);
        let mut calls = 0;
        // These callbacks run with the writer mutex held, before and after the
        // atomic swap, retirement, and reclamation, just like an interrupt.
        map.publish_probed(&mut writer, Some(new_root), || {
            calls += 1;
            assert_eq!(map.lookup(0x1004), Some(7));
            let nested = map.read();
            assert!(nested.find_entry(0x1004, &mut || {}).is_some());
            let (site, recovered) = map.lookup_recovery(0x1004).unwrap();
            assert_eq!(site.width, 8);
            assert_eq!(recovered, recipe());
        });
        assert_eq!(calls, 4);
    }

    #[test]
    fn paused_reader_keeps_removed_payload_until_grace_and_allows_address_reuse() {
        use std::sync::Barrier;
        let map = CodeMap::new();
        map.register(0x1000, 16, Box::new([(0, 0)]), Box::new([7]))
            .unwrap();
        let arrived = Barrier::new(2);
        let resume = Barrier::new(2);
        std::thread::scope(|scope| {
            let reader = scope.spawn(|| {
                let mut paused = false;
                assert_eq!(
                    map.lookup_probed(0x1000, || {
                        if !paused {
                            paused = true;
                            arrived.wait();
                            resume.wait();
                        }
                    }),
                    Some(7)
                );
            });
            arrived.wait();
            assert!(map.unregister(0x1000, 16));
            assert_eq!(map.lookup(0x1000), None);
            map.register(0x1000, 16, Box::new([(0, 0)]), Box::new([99]))
                .unwrap();
            assert_eq!(map.lookup(0x1000), Some(99));
            assert!(map.memory_stats().nodes > arc_bytes::<Node>() as u64);
            resume.wait();
            reader.join().unwrap();
        });
        assert_eq!(map.memory_stats().nodes, arc_bytes::<Node>() as u64);
        assert!(map.unregister(0x1000, 16));
        let stats = map.memory_stats();
        assert_eq!(
            stats.nodes
                + stats.entries
                + stats.sources
                + stats.traps
                + stats.recipes
                + stats.instructions,
            0
        );
    }

    #[test]
    fn unregister_releases_shared_recipes_and_instruction_descriptors() {
        let map = CodeMap::new();
        for i in 0..2 {
            let rip = 7 + i as u64;
            let descriptor = crate::fault::InstructionFaultInfo::from_ops(
                rip,
                Some(rip + 1),
                &[],
                crate::CpuMode::Long64,
            );
            map.register_with_fault_info(
                0x1000 + i * 32,
                16,
                Box::new([(0, 0)]),
                Box::new([rip]),
                vec![GuestTrapSite {
                    host_offset: 4,
                    guest_rip: rip,
                    access: GuestAccessKind::Read,
                    width: 8,
                }],
                Some(recipe()),
                vec![descriptor],
            )
            .unwrap();
        }
        let recovered = map.lookup_recovery(0x1004).unwrap();
        assert_eq!(
            map.memory_stats().recipes,
            arc_bytes::<FrameRecipe>() as u64
        );
        assert!(map.unregister(0x1000, 16));
        assert_eq!(
            map.memory_stats().recipes,
            arc_bytes::<FrameRecipe>() as u64
        );
        assert!(map.unregister(0x1020, 16));
        let stats = map.memory_stats();
        assert_eq!(stats.recipes + stats.instructions, 0);
        // A result obtained before removal is independent of both pooled objects.
        assert_eq!(recovered.0.guest_rip, 7);
        assert_eq!(recovered.1, recipe());
    }

    #[test]
    fn unregister_balances_tree_restores_capacity_and_preserves_neighbors() {
        fn balanced(n: &Option<Arc<Node>>) -> u8 {
            let Some(n) = n else {
                return 0;
            };
            let left = balanced(&n.left);
            let right = balanced(&n.right);
            assert!(left.abs_diff(right) <= 1);
            assert_eq!(n.height, 1 + left.max(right));
            n.height
        }
        let map = CodeMap::with_capacity(1024);
        for i in 0..1024 {
            map.register(
                0x1000 + i * 32,
                24,
                Box::new([(0, 0)]),
                Box::new([i as u64]),
            )
            .unwrap();
        }
        assert!(!map.unregister(0x1001, 23));
        assert!(!map.unregister(0x1000, 23));
        assert!(!map.unregister(usize::MAX, 24));
        for i in 0..1024 {
            let index = (i * 137) % 1024;
            let pc = 0x1000 + index * 32;
            assert!(map.unregister(pc, 24));
            assert!(!map.unregister(pc, 24));
            assert_eq!(map.lookup(pc), None);
            balanced(&map.len.lock().unwrap().root);
        }
        map.register(0x1000, 24, Box::new([(0, 0)]), Box::new([55]))
            .unwrap();
        assert_eq!(map.lookup(0x1000), Some(55));
    }

    #[test]
    fn compact_metadata_preserves_large_offsets_indices_and_full_width_rips() {
        for max_offset in [255, 256, 65535, 65536, u32::MAX - 1] {
            let rips: Vec<_> = (0..300)
                .map(|i| u64::MAX.wrapping_sub(i * 0x100000000))
                .collect();
            let traps = vec![GuestTrapSite {
                host_offset: max_offset,
                guest_rip: 0x123456789,
                access: GuestAccessKind::ReadWrite,
                width: 64,
            }];
            let metadata = Metadata::new(&[(0, 299), (max_offset, 256)], &rips, &traps);
            assert_eq!(metadata.source(0), Some(rips[299]));
            assert_eq!(metadata.source(max_offset), Some(rips[256]));
            assert_eq!(metadata.trap(max_offset), Some(traps[0]));
            assert_eq!(metadata.trap(max_offset - 1), None);
        }
        // A trap RIP need not appear in the source-location RIP dictionary.
        let metadata = Metadata::new(
            &[],
            &[],
            &[GuestTrapSite {
                host_offset: 0,
                guest_rip: u64::MAX,
                access: GuestAccessKind::Read,
                width: 1,
            }],
        );
        assert_eq!(metadata.trap(0).unwrap().guest_rip, u64::MAX);
    }

    #[test]
    fn metadata_is_bounded_at_one_hundred_thousand_units_and_freed_on_drop() {
        let map = CodeMap::new();
        let mut descriptor =
            crate::fault::InstructionFaultInfo::from_ops(1, Some(2), &[], crate::CpuMode::Long64);
        for i in 0..100_000 {
            let rip = 0x400000 + i as u64 * 16;
            let instructions = (0..3)
                .map(|j| {
                    descriptor.rip = rip + j;
                    descriptor.clone()
                })
                .collect();
            map.register_with_fault_path(
                0x1000 + i * 32,
                24,
                Box::new([(0, 0), (8, 1), (16, 2)]),
                Box::new([rip, rip + 1, rip + 2]),
                vec![GuestTrapSite {
                    host_offset: 4,
                    guest_rip: rip,
                    access: GuestAccessKind::Read,
                    width: 8,
                }],
                Some(recipe()),
                instructions,
                Some(FaultUnit::Block),
            )
            .unwrap();
            if i == 9999 || i == 99999 {
                let stats = map.memory_stats();
                assert!(stats.total() <= 256 * (i as u64 + 1), "{stats:?}");
                assert_eq!(stats.nodes, (i as u64 + 1) * arc_bytes::<Node>() as u64);
                assert_eq!(stats.recipes, arc_bytes::<FrameRecipe>() as u64);
                eprintln!("MEASURED: {} units: {stats:?}", i + 1);
            }
        }
        let accounting = map.accounting.get().unwrap().clone();
        drop(map);
        for counter in [
            &accounting.nodes,
            &accounting.entries,
            &accounting.sources,
            &accounting.traps,
            &accounting.instructions,
        ] {
            assert_eq!(counter.load(Ordering::Relaxed), 0);
        }
    }

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
    fn trap_capacity_discards_spare_allocation() {
        let mut map = CodeMap::with_capacity(2);
        map.trap_capacity = 2;
        let mut traps = Vec::with_capacity(3);
        traps.push(GuestTrapSite {
            host_offset: 0,
            guest_rip: 1,
            access: GuestAccessKind::Read,
            width: 1,
        });
        map.register_with_traps(0x1000, 8, Box::new([]), Box::new([]), traps)
            .unwrap();
        assert_eq!(map.len.lock().unwrap().traps, 1);
        assert_eq!(map.lookup_trap(0x1000).unwrap().guest_rip, 1);
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
            assert!(map.read().find_entry(start, &mut || visits += 1).is_some());
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
                            let recovery = map.lookup_recovery(0x1000 + i * 32 + off);
                            if finished || recovery.is_some() {
                                let expected = (off == 4).then_some((
                                    GuestTrapSite {
                                        host_offset: 4,
                                        guest_rip: u64::MAX - i as u64,
                                        access: GuestAccessKind::Read,
                                        width: 8,
                                    },
                                    recipe(),
                                ));
                                assert_eq!(recovery, expected);
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
                map.register_with_recovery(
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
                    Some(recipe()),
                )
                .unwrap();
            }
            done.store(true, Ordering::Release);
        });
    }
}
