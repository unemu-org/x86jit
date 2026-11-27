//! Internal Machine translation boundary (paging design §§3.1–3.4, spec §17.10).
//! A functional TLB shares its proof generation with paged instruction sources.
use super::*;
use crate::fault::{
    AddressPrecision, AddressSpace, ErrorCode, FaultAddress, FaultCause, FaultClass, FaultMetadata,
    FaultOrigin, FaultPath, Knowledge, MemoryEffects, PagingWalk, PagingWalkLevel,
    RegisterProgress,
};

const PRESENT: u32 = 1;
const WRITE: u32 = 2;
const USER: u32 = 4;
const ACCESSED: u32 = 0x20;
const DIRTY: u32 = 0x40;
const FRAME: u32 = 0xffff_f000;
const IDENTITY: u32 = FRAME | PRESENT | WRITE | USER;

/// Storage exists only in the paged adapter; PG=0 still carries `()`.
#[derive(Default)]
pub(crate) struct AdapterState<'a> {
    pub(super) transaction: Option<&'a std::cell::RefCell<Transaction>>,
    pub(super) tlb: Option<&'a std::cell::RefCell<Tlb>>,
    pub(super) fast_operand: Option<FastOperand>,
    /// A strict lane's access found no proved window (`FastOperand::strict`).
    pub(super) fast_missed: Cell<bool>,
}

/// The pages a `tlb_lane` has translated. A later access to one reuses its
/// translation, as the transaction's plans reuse theirs (`plan`): a store
/// never depends on a TLB entry that a later walk of the lane replaced, and
/// the frame's slots, the descriptors of one table page and the TSS fields
/// cost one translation each.
#[derive(Default)]
struct LaneWindows {
    pages: [Option<LanePage>; 8],
    len: usize,
    stored: bool,
    /// The TLB slots the lane's translations may have changed, as they were,
    /// and the generation: a lane that replays puts them back, so that the
    /// transaction path meets the TLB, stale entries included, that it would
    /// have met (SDM Vol 3A §5.10.2: a stale entry may still be used).
    saved: [Option<(usize, Option<Entry>)>; 8],
    generation: u64,
}

/// One translated page of a `tlb_lane`, with the rights it was proved for.
#[derive(Clone, Copy)]
struct LanePage {
    page: u32,
    frame: u32,
    /// Proved for writes: write permission and PTE.D (`lane_proof`).
    write: bool,
    supervisor: bool,
}

/// What a `tlb_lane` access does with its window.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LaneAccess {
    Read,
    Probe,
    Store,
}

#[cfg(debug_assertions)]
thread_local! {
    /// A strict lane committed its descriptor's accessed bit (`strict_lane`).
    static LANE_COMMITTED: Cell<bool> = const { Cell::new(false) };
}

/// RAM proved before execution from existing TLB entries, for an access of
/// one privilege and kind. A write window also admits reads of its bytes.
#[derive(Clone, Copy)]
struct Window {
    linear: u64,
    bus: u64,
    size: u8,
    write: bool,
    supervisor: bool,
}

/// The proved operand windows of one instruction: the scalar operand, or a
/// segment load's descriptor and a far transfer's stack frame. A strict
/// (system-lane) instruction stops at the first access outside its windows,
/// before any window write, and replays on the transaction path. Its only
/// earlier commit is the descriptor's accessed-bit CAS (`strict_lane`).
///
/// A non-strict operand is installed only by a `Fast` instruction's first
/// access (`access_operand`), whose class (`classify`) admits operations that
/// make no system (`Supervisor`) or task-stage (`AddressKeyed`) access; those
/// views assert that they never meet one.
#[derive(Clone, Copy)]
pub(crate) struct FastOperand {
    windows: [Option<Window>; 2],
    strict: bool,
    /// A strict lane that proves each access when it is made, through the
    /// TLB and the pages it already translated (`lane_window`), instead of
    /// from the two windows (`tlb_lane`). It makes no descriptor CAS.
    tlb: bool,
}

impl FastOperand {
    /// A system lane's descriptor and frame windows (`strict_lane`).
    pub(super) fn is_strict(&self) -> bool {
        self.strict
    }

    /// The strict lane of `tlb_lane`, proved access by access.
    pub(super) fn proves_each_access(&self) -> bool {
        self.tlb
    }

    /// Every access proved from the TLB when it is made (`tlb_lane`).
    const fn each_access() -> Self {
        Self {
            windows: [None, None],
            strict: true,
            tlb: true,
        }
    }

    fn single(linear: u64, bus: u64, size: u8, access: AccessKind, cpu: &CpuState) -> Self {
        Self {
            windows: [
                Some(Window {
                    linear,
                    bus,
                    size,
                    write: access == AccessKind::Write,
                    supervisor: cpu.legacy.cpl != 3,
                }),
                None,
            ],
            strict: false,
            tlb: false,
        }
    }

    #[inline(always)]
    fn window<const ROUTED: bool, const ORDER: u8>(
        &self,
        cpu: &CpuState,
        mem: &impl LegacyMemory<ROUTED, ORDER>,
        linear: u64,
        size: u8,
        write: bool,
    ) -> Option<Window> {
        let supervisor = cpu.legacy.cpl != 3 || mem.supervisor();
        let fits = |w: &Window| {
            w.supervisor == supervisor
                && (w.write || !write)
                && linear >= w.linear
                && linear + u64::from(size) <= w.linear + u64::from(w.size)
        };
        if let Some(w) = self.windows[0].filter(fits) {
            return Some(w);
        }
        self.second_window(mem, fits)
    }

    #[cold]
    #[inline(never)]
    fn second_window<const ROUTED: bool, const ORDER: u8>(
        &self,
        mem: &impl LegacyMemory<ROUTED, ORDER>,
        fits: impl Fn(&Window) -> bool,
    ) -> Option<Window> {
        let found = self.windows[1].filter(fits);
        if found.is_none() {
            // A non-strict operand covers the whole access set of the
            // instructions of the `Fast` class (`classify`); no system access.
            debug_assert!(self.strict, "an access outside the proved operand");
            // A lane proved access by access has no windows: its caller
            // proves the access (`lane_access`).
            if self.strict && !self.tlb {
                mem.fast_miss();
            }
        }
        found
    }
}

impl Window {
    fn bus(self, linear: u64) -> u64 {
        self.bus + (linear - self.linear)
    }

    fn failure(self, cpu: &mut CpuState, trap: MemTrap, access: AccessKind, value: u64) -> Exit {
        let StepResult::Exit(exit) =
            trap_out(cpu, cpu.rip, trap, self.bus, self.size, access, value).into_step()
        else {
            unreachable!()
        };
        crate::fault::qualify_legacy_bus(exit, self.linear, self.bus)
    }
}

/// The access of a lane proved access by access (`tlb_lane`), which its
/// operand's windows (none) did not cover; for any other lane, the miss that
/// `FastOperand::window` has already reported.
#[cold]
#[inline(never)]
fn lane_access<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u64,
    size: u8,
    access: LaneAccess,
) -> Result<Window, Exit> {
    if mem
        .fast_operand()
        .is_some_and(FastOperand::proves_each_access)
    {
        if let Some(window) = lane_window(cpu, mem, linear, size, access) {
            return Ok(window);
        }
    }
    Err(fast_miss_exit(cpu))
}

/// `read` outside the operand's windows (`lane_access`).
#[cold]
#[inline(never)]
fn lane_read<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
) -> Result<u64, Exit> {
    let window = lane_access(cpu, mem, addr, size, LaneAccess::Read)?;
    mem.raw()
        .read(window.bus(addr), size)
        .map_err(|trap| window.failure(cpu, trap, AccessKind::Read, 0))
}

/// `write` outside the operand's windows (`lane_access`).
#[cold]
#[inline(never)]
fn lane_write<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
    value: u64,
) -> Result<(), Exit> {
    let window = lane_access(cpu, mem, addr, size, LaneAccess::Store)?;
    let value = value & mask(size);
    let bus = window.bus(addr);
    mem.raw()
        .write(bus, value, size)
        .map_err(|trap| window.failure(cpu, trap, AccessKind::Write, value))?;
    mem.note_write(bus, size);
    Ok(())
}

/// The window of one `tlb_lane` access: through a page the lane translated
/// already, or one it translates as `plan` does, through `translate_linear`
/// (a TLB hit, or a walk that sets A and fills the entry, as the transaction
/// preview's walk would). The access must lie in one page and in RAM, and a
/// write needs D already set and no registered code. Anything else, a fault
/// included, is a miss: the lane replays on the transaction path, which
/// repeats the same walks. A store needs no new translation: every lane
/// probes its frame first. Host fault recovery is off (`tlb_lane_admits`),
/// so RAM needs no write probe (`Memory::probe_write`).
#[cold]
#[inline(never)]
fn lane_window<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u64,
    size: u8,
    access: LaneAccess,
) -> Option<Window> {
    let found = lane_proof(cpu, mem, linear, size, access);
    if found.is_none() {
        mem.fast_miss();
    }
    found
}

fn lane_proof<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u64,
    size: u8,
    access: LaneAccess,
) -> Option<Window> {
    let cell = mem.tlb()?;
    if size == 0 || linear > u64::from(u32::MAX) || u64::from(size) > 0x1000 - (linear & 0xfff) {
        return None;
    }
    // SDM Vol 3A §5.6.1 p. 5-33; 80386 PRM §6.5 p. 6-22: implicit table,
    // TSS and inner-stack accesses are supervisor at any CPL (`plan`).
    let privilege = if cpu.legacy.cpl == 3 && !mem.supervisor() {
        Privilege::User
    } else {
        Privilege::Supervisor
    };
    let supervisor = privilege == Privilege::Supervisor;
    let write = access != LaneAccess::Read;
    let page = (linear >> 12) as u32;
    let known = {
        let tlb = cell.borrow();
        let lane = &tlb.lane;
        // Before a store, a page whose entry a later walk of the lane
        // replaced is translated again, as the preview's next access to it
        // would be; after one, the committing phase reuses its plans.
        let present = tlb.entries[Tlb::index(linear as u32)].is_some_and(|e| e.page == page);
        lane.pages[..lane.len]
            .iter()
            .flatten()
            .copied()
            .find(|p| p.page == page && p.supervisor == supervisor && (p.write || !write))
            .filter(|_| present || lane.stored)
    };
    let frame = if let Some(known) = known {
        known.frame
    } else {
        {
            let lane = &cell.borrow().lane;
            debug_assert!(!lane.stored, "a TLB lane proved an access after a store");
            if lane.stored || lane.len == lane.pages.len() {
                return None;
            }
        }
        let kind = if write {
            AccessKind::Write
        } else {
            AccessKind::Read
        };
        {
            let tlb = &mut *cell.borrow_mut();
            let index = Tlb::index(linear as u32);
            let lane = &mut tlb.lane;
            if !lane.saved.iter().flatten().any(|(at, _)| *at == index) {
                let at = lane.len;
                lane.saved[at] = Some((index, tlb.entries[index]));
            }
        }
        let t = translate_linear(cpu, mem, linear as u32, size, kind, privilege, true).ok()?;
        // 80386 PRM §5.2.4.3 p. 5-10: D is set by the first write, which
        // the transaction path performs after its preview (`finish_plan`).
        if write && t.pte & DIRTY == 0 {
            return None;
        }
        let lane = &mut cell.borrow_mut().lane;
        let at = lane.len;
        lane.pages[at] = Some(LanePage {
            page,
            frame: t.physical & FRAME,
            write,
            supervisor,
        });
        lane.len += 1;
        t.physical & FRAME
    };
    let physical = u64::from(frame | (linear as u32 & 0xfff));
    let (bus, part) = legacy::physical_span(cpu, mem.physical_mask(), physical, size);
    let memory = &mem.raw().0;
    if part != size
        || memory.region_kind(bus, size) != Ok(crate::RegionKind::Ram)
        || (write && memory.is_code_page(bus >> 12))
    {
        return None;
    }
    if access == LaneAccess::Store {
        cell.borrow_mut().lane.stored = true;
    }
    Some(Window {
        linear,
        bus,
        size,
        write,
        supervisor,
    })
}

/// The stop of a strict lane's unproved access; never leaves the lane.
#[cold]
fn fast_miss_exit(cpu: &CpuState) -> Exit {
    refusal(cpu, crate::UnsupportedSystemOperation::LegacyExecution)
}

#[derive(Clone, Copy)]
struct Entry {
    page: u32,
    translation: Translation,
    rights: u32,
    /// An instruction fetch used this translation, so retained source proofs
    /// may depend on it.
    fetched: bool,
}

/// The per-vCPU TLB's entries (decision-32 Q3). The architecture fixes no
/// size: software "should not rely on the existence of TLBs or on the
/// retention of TLB entries" (SDM Vol 3A §5.10.2.2 p. 5-43), so any size, and
/// any replacement, is a correct TLB. 256 entries cover 1 MiB of 4 KiB pages.
pub(crate) const TLB_ENTRIES: usize = 256;
const _: () = assert!(TLB_ENTRIES.is_power_of_two());

/// Paging design §3.2 / decision-27: tokens are issued by the VM namespace,
/// never per-vCPU counters. Removing or replacing an entry that an instruction
/// fetch used advances the generation: a source proof whose code pages were
/// translated at an older generation translates them again before it is
/// used (`vm/legacy/paging.rs`, xj-to9.70). A data or stack entry backs no
/// proof: fetch translates through its own entries, and a page whose fetch
/// entry was replaced walks again. A flush advances the generation too: a
/// CR3 load, equal values included, or a change of another walk input
/// empties the TLB, and every proof walks its code pages again.
pub(crate) struct Tlb {
    entries: [Option<Entry>; TLB_ENTRIES],
    pub(crate) generation: u64,
    context: Option<(u32, u32, u8, u32)>,
    /// The pages of the running `tlb_lane`, reset when one starts.
    lane: LaneWindows,
    /// Generation advances by site since the relift diagnostics last read
    /// them (`crate::vm::TlbAdvance` order).
    #[cfg(feature = "test-hooks")]
    pub(crate) advances: [u64; crate::vm::TLB_ADVANCES],
}
impl Default for Tlb {
    fn default() -> Self {
        Self {
            entries: [None; TLB_ENTRIES],
            generation: 0,
            context: None,
            lane: LaneWindows::default(),
            #[cfg(feature = "test-hooks")]
            advances: [0; crate::vm::TLB_ADVANCES],
        }
    }
}

/// Why a TLB generation advances, in `crate::vm::TlbAdvance` order.
#[derive(Clone, Copy)]
pub(crate) enum Advance {
    ContextFlush,
    ExplicitFlush,
    Replaced,
    FaultRemoved,
    Rewalked,
    DirtyFailed,
    InvalidatePage,
    Cr3Load,
    // Counted only: a replayed lane restores its saved generation.
    #[cfg(feature = "test-hooks")]
    LaneUndone,
}
impl Tlb {
    #[inline]
    pub(crate) fn index(linear: u32) -> usize {
        let page = linear >> 12;
        // Fold the whole 20-bit page number over 8 bits: descriptor tables
        // and data separated by 1 MiB must not repeatedly expire each other's
        // instruction proofs. The fold leaves bit 0 alone, so adjacent pages
        // never share a slot: a block straddling two pages keeps both
        // entries (xj-to9.70); bits 8-14 and 15-19 enter at bits 1-7.
        ((page ^ (((page >> 8) ^ (page >> 15)) << 1)) as usize) & (TLB_ENTRIES - 1)
    }

    /// Advance the generation, counting why.
    #[inline(always)]
    fn advance(&mut self, _why: Advance) {
        self.generation = crate::vm::paging_generation();
        #[cfg(feature = "test-hooks")]
        {
            self.advances[_why as usize] += 1;
        }
    }

    /// Flush every entry: a reset or an explicit invalidation.
    pub(crate) fn invalidate(&mut self) {
        self.flush(Advance::ExplicitFlush);
    }

    /// Flush every entry for a guest CR3 load, equal values included
    /// (80386 PRM §5.2.5 p. 5-11; Intel486 PRM §5.3.5 p. 5-22).
    pub(crate) fn load_cr3(&mut self) {
        self.flush(Advance::Cr3Load);
    }

    fn flush(&mut self, why: Advance) {
        self.entries.fill(None);
        self.context = None;
        self.advance(why);
    }

    /// INVLPG: remove the entry of `linear`'s page (paging design §3.2;
    /// Intel486 PRM §26 INVLPG p. 26-172; SDM Vol 3A §5.10.4.1 pp. 5-48-49).
    /// As any removal, one a fetch used advances the generation, so every
    /// retained proof translates its code pages again before it runs; INVLPG
    /// is a singleton block, so no block runs on past it. A data or stack
    /// entry backs no proof.
    pub(crate) fn invalidate_page(&mut self, linear: u32) {
        let slot = &mut self.entries[Self::index(linear)];
        if slot.is_some_and(|e| e.page == linear >> 12) {
            self.remove(linear, Advance::InvalidatePage);
        }
    }

    /// The physical address of an instruction fetch of `linear` that this
    /// TLB holds a permitted entry for, marking the entry fetched; `None`
    /// when a fetch would walk (`translate_linear`'s hit path, without its
    /// fault or rewalk arms). The context must be reconciled.
    pub(crate) fn fetch_hit(&mut self, cpu: &CpuState, linear: u32) -> Option<u32> {
        let privilege = if cpu.legacy.cpl == 3 {
            Privilege::User
        } else {
            Privilege::Supervisor
        };
        let e = self.entries[Self::index(linear)]
            .as_mut()
            .filter(|e| e.page == linear >> 12)?;
        if !permitted(cpu, AccessKind::Execute, privilege, e.rights) {
            return None;
        }
        e.fetched = true;
        Some((e.translation.physical & FRAME) | (linear & 0xfff))
    }

    pub(crate) fn reconcile(&mut self, cpu: &CpuState) {
        let context = (
            cpu.legacy.cr3,
            // Paging design §3.2: segment/CPL/flags and CR0's execution
            // controls do not identify translations. Retain PE/CD/NW as
            // conservative walk inputs alongside PG/WP. No model has CR4.
            cpu.legacy.cr0 & 0xe001_0001,
            cpu.legacy.a20_enabled,
            cpu.machine_flags_mask,
        );
        if self.context != Some(context) {
            // A flush leaves no context and no entries: naming the new one
            // needs no second flush.
            if self.context.is_some() {
                self.flush(Advance::ContextFlush);
            }
            self.context = Some(context);
        }
    }

    /// A `tlb_lane` starts: the translation context is current, as its
    /// first walk would make it, and nothing is saved yet.
    fn begin_lane(&mut self, cpu: &CpuState) {
        self.reconcile(cpu);
        self.lane = LaneWindows {
            generation: self.generation,
            ..LaneWindows::default()
        };
    }

    /// A `tlb_lane` replays: restore the slots its translations changed.
    fn undo_lane(&mut self) {
        let saved = std::mem::take(&mut self.lane.saved);
        for (index, entry) in saved.into_iter().flatten().rev() {
            self.entries[index] = entry;
        }
        #[cfg(feature = "test-hooks")]
        if self.generation != self.lane.generation {
            self.advances[Advance::LaneUndone as usize] += 1;
        }
        self.generation = self.lane.generation;
    }

    /// The valid entries, for the witnesses: slot, page, frame, PTE, rights
    /// and whether a fetch used it. The cached physical address keeps the
    /// offset of the access that filled the entry, which no lookup uses.
    #[cfg(feature = "test-hooks")]
    pub(crate) fn entries_for_test(&self) -> Vec<(usize, u32, u32, u32, u32, bool)> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(at, e)| {
                e.map(|e| {
                    (
                        at,
                        e.page,
                        e.translation.physical & FRAME,
                        e.translation.pte,
                        e.rights,
                        e.fetched,
                    )
                })
            })
            .collect()
    }

    fn remove(&mut self, linear: u32, why: Advance) {
        let slot = &mut self.entries[Self::index(linear)];
        if slot.take().is_some_and(|e| e.fetched) {
            self.advance(why);
        }
    }

    fn insert(&mut self, linear: u32, translation: Translation, rights: u32, fetched: bool) {
        let index = Self::index(linear);
        if self.entries[index].is_some_and(|e| e.fetched && e.page != linear >> 12) {
            self.advance(Advance::Replaced);
        }
        self.entries[index] = Some(Entry {
            page: linear >> 12,
            translation,
            rights,
            fetched,
        });
    }

    fn dirty(&mut self, linear: u32, t: Translation) {
        if let Some(e) = &mut self.entries[Self::index(linear)] {
            if e.page == linear >> 12
                && e.translation.pte_address == t.pte_address
                && (e.translation.pte ^ t.pte) & IDENTITY == 0
            {
                e.translation.pte |= DIRTY;
            }
        }
    }
}

fn permitted(cpu: &CpuState, access: AccessKind, privilege: Privilege, rights: u32) -> bool {
    // 80386 PRM §§6.4.1–6.4.2 Table 6-5; SDM Vol 3A §5.6.1.
    let wp = cpu.machine_flags_mask != 0 && cpu.legacy.cr0 & (1 << 16) != 0;
    (!privilege.user() || rights & USER != 0)
        && (access != AccessKind::Write || (!privilege.user() && !wp) || rights & WRITE != 0)
}

#[cfg(any(test, feature = "test-hooks"))]
thread_local! {
    /// The lanes that skip the transaction (`tlb_lane`, and the
    /// read-modify-write operand of the `Fast` class) are off.
    static FAST_LANES_OFF: Cell<bool> = const { Cell::new(false) };
}

#[inline(always)]
fn fast_lanes_off() -> bool {
    #[cfg(any(test, feature = "test-hooks"))]
    return FAST_LANES_OFF.with(Cell::get);
    #[cfg(not(any(test, feature = "test-hooks")))]
    false
}

#[cfg(any(test, feature = "test-hooks"))]
thread_local! {
    static FAST_LANE_HITS: Cell<u64> = const { Cell::new(0) };
}

/// One delivery, IRET or refused instruction completed in a `tlb_lane`,
/// for the witnesses.
#[inline(always)]
fn count_fast_lane() {
    #[cfg(any(test, feature = "test-hooks"))]
    FAST_LANE_HITS.with(|hits| hits.set(hits.get() + 1));
}

/// The lanes of `without_fast_lanes_for_test` completed on this thread.
#[cfg(feature = "test-hooks")]
pub fn fast_lane_hits_for_test() -> u64 {
    FAST_LANE_HITS.with(Cell::get)
}

/// Run `run` on this thread with every delivery, IRET, refused instruction
/// and read-modify-write on the transaction path, for parity witnesses.
#[cfg(feature = "test-hooks")]
pub fn without_fast_lanes_for_test<R>(run: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            FAST_LANES_OFF.with(|off| off.set(self.0));
        }
    }
    let _reset = Reset(FAST_LANES_OFF.with(|off| off.replace(true)));
    run()
}

#[cfg(feature = "test-hooks")]
type CasHook = Box<dyn FnMut(&Memory, u64, u32, u32, Option<u32>)>;
#[cfg(feature = "test-hooks")]
thread_local! {
    static CAS_HOOK: std::cell::RefCell<Option<CasHook>> = const { std::cell::RefCell::new(None) };
}

/// Observe/inject table CAS races on this thread only. `None` precedes the CAS;
/// `Some(observed)` follows it, including failed comparisons.
#[cfg(feature = "test-hooks")]
pub fn with_cas_hook_for_test<R>(
    hook: impl FnMut(&Memory, u64, u32, u32, Option<u32>) + 'static,
    run: impl FnOnce() -> R,
) -> R {
    struct Reset(Option<CasHook>);
    impl Drop for Reset {
        fn drop(&mut self) {
            CAS_HOOK.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let _reset = Reset(CAS_HOOK.with(|slot| slot.replace(Some(Box::new(hook)))));
    run()
}

#[cfg(feature = "test-hooks")]
fn cas_hook(memory: &Memory, bus: u64, expected: u32, bits: u32, observed: Option<u32>) {
    CAS_HOOK.with(|slot| {
        if let Some(hook) = &mut *slot.borrow_mut() {
            hook(memory, bus, expected, bits, observed);
        }
    });
}

// The paged adapter retains the same control-write validation as PG=0.
#[cold]
#[inline(never)]
pub(super) fn write_cr0(cpu: &mut CpuState, value: u32) -> Result<bool, Exit> {
    // SDM Vol 3A §5.1.1 p. 5-2; Intel486 PRM §12.2.1 p. 12-3:
    // PG requires PE and NW requires CD.
    if value & 1 == 0 || (cpu.machine_flags_mask != 0 && value & 0x6000_0000 == 0x2000_0000) {
        return Err(exception(cpu, 13));
    }
    let changed = cpu.legacy.cr0 != value;
    cpu.legacy.cr0 = value;
    Ok(changed)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Privilege {
    Supervisor,
    User,
}
impl Privilege {
    fn user(self) -> bool {
        matches!(self, Self::User)
    }
}

struct PageFault {
    linear: u32,
    access: AccessKind,
    privilege: Privilege,
    protection: bool,
}
impl PageFault {
    fn exit(self, cpu: &CpuState) -> Exit {
        // 80386 PRM §9.8.14, Fig. 9-8 pp. 9-17–18; Intel486 PRM
        // §9.9.14 pp. 9-21–23: CR2 is linear, only P/W/U are defined.
        let code = u32::from(self.protection)
            | (u32::from(self.access == AccessKind::Write) << 1)
            | (u32::from(self.privilege.user()) << 2);
        let mut m = FaultMetadata::architectural_exception(
            cpu.rip,
            cpu.rip,
            14,
            FaultClass::Fault,
            ErrorCode::bits(code, 7),
        );
        m.origin = FaultOrigin::GuestException(FaultCause::PageTranslation);
        m.path = FaultPath::Interpreter;
        m.address = Knowledge::Known(FaultAddress::new(
            u64::from(self.linear),
            AddressPrecision::ExactByte,
            AddressSpace::Linear,
        ));
        m.access = Knowledge::Known(self.access.into());
        m.transfer_access = m.access;
        m.progress.registers = RegisterProgress::InstructionEntry;
        // A updates can precede this failure even with unchanged operand RAM.
        m.progress.ram = MemoryEffects::PossiblyPartial;
        m.progress.device = MemoryEffects::None;
        Exit::exception(cpu.rip, 14).with_fault_metadata(m)
    }
}

pub(crate) fn publish_cr2(cpu: &mut CpuState, exit: &Exit) {
    if let Some(m) = exit.fault_metadata() {
        if m.origin == FaultOrigin::GuestException(FaultCause::PageTranslation) {
            if let Knowledge::Known(address) = m.address {
                cpu.legacy.cr2 = address.value() as u32;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Translation {
    physical: u32,
    bytes: u8,
    pte_address: u32,
    pte: u32,
}

#[allow(clippy::too_many_arguments)]
fn bus_failure(
    cpu: &mut CpuState,
    trap: MemTrap,
    linear: u32,
    physical: u32,
    bus: u64,
    level: PagingWalkLevel,
    access: AccessKind,
    value: u64,
) -> Exit {
    let StepResult::Exit(mut exit) =
        trap_out(cpu, cpu.rip, trap, bus, 4, access, value).into_step()
    else {
        unreachable!()
    };
    if let Exit::UnmappedMemory {
        metadata: Some(m), ..
    } = &mut exit
    {
        // A table transfer's byte offset is physical evidence, not an offset
        // within the original operand's linear address.
        m.physical_address = match m.address {
            Knowledge::Known(address) => Knowledge::Known(address.value()),
            _ => Knowledge::Known(bus),
        };
        m.address = Knowledge::Known(FaultAddress::new(
            u64::from(linear),
            AddressPrecision::ExactByte,
            AddressSpace::Linear,
        ));
        m.paging_walk = Some(PagingWalk {
            level,
            physical_address: u64::from(physical),
        });
        m.progress.ram = MemoryEffects::PossiblyPartial;
    }
    exit
}

fn table_bus<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u32,
    physical: u32,
    level: PagingWalkLevel,
) -> Result<u64, Exit> {
    // 80386 PRM §§5.2.3–5.2.4 pp. 5-8–10: table references are physical.
    // Apply the board gate only after selecting the entry (paging design §3.6).
    let (bus, bytes) = legacy::physical_span(cpu, mem.physical_mask(), u64::from(physical), 4);
    debug_assert_eq!(bytes, 4); // aligned entries cannot cross the bus/A20 boundary
    match mem.raw().region_kind(bus, 4) {
        Ok(crate::RegionKind::Ram) => Ok(bus),
        Ok(_) => Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::PagingTableStorage,
        )),
        Err(_)
            if (0..4).any(|byte| {
                matches!(
                    mem.raw().region_kind(bus + byte, 1),
                    Ok(crate::RegionKind::Trap | crate::RegionKind::Rom)
                )
            }) =>
        {
            Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::PagingTableStorage,
            ))
        }
        Err(trap) => Err(bus_failure(
            cpu,
            trap,
            linear,
            physical,
            bus,
            level,
            AccessKind::Read,
            0,
        )),
    }
}

fn table_read<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u32,
    physical: u32,
    level: PagingWalkLevel,
) -> Result<u32, Exit> {
    let bus = table_bus(cpu, mem, linear, physical, level)?;
    mem.raw()
        .read(bus, 4)
        .map(|v| v as u32)
        .map_err(|trap| bus_failure(cpu, trap, linear, physical, bus, level, AccessKind::Read, 0))
}

// None means that frame/presence/permissions changed: restart the translation.
fn table_or<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u32,
    physical: u32,
    level: PagingWalkLevel,
    mut expected: u32,
    bits: u32,
) -> Result<Option<u32>, Exit> {
    let bus = table_bus(cpu, mem, linear, physical, level)?;
    while expected & bits != bits {
        // SDM Vol 3A §11.1.2.1 p. 11-3; 80386 PRM §11.2.2 p. 11-5:
        // individual A/D updates are locked and preserve competing edits.
        let value = expected | bits;
        #[cfg(feature = "test-hooks")]
        cas_hook(mem.raw(), bus, expected, bits, None);
        let observed = mem
            .raw()
            .atomic_cas(bus, u64::from(expected), u64::from(value), 4)
            .map_err(|trap| {
                bus_failure(
                    cpu,
                    trap,
                    linear,
                    physical,
                    bus,
                    level,
                    AccessKind::Write,
                    u64::from(value),
                )
            })? as u32;
        #[cfg(feature = "test-hooks")]
        cas_hook(mem.raw(), bus, expected, bits, Some(observed));
        if observed == expected {
            mem.note_write(bus, 4);
            return Ok(Some(value));
        }
        if (observed ^ expected) & IDENTITY != 0 {
            return Ok(None);
        }
        expected = observed;
    }
    Ok(Some(expected))
}

fn translate_linear<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u32,
    bytes: u8,
    access: AccessKind,
    privilege: Privilege,
    set_accessed: bool,
) -> Result<Translation, Exit> {
    if let Some(tlb) = mem.tlb() {
        let mut tlb = tlb.borrow_mut();
        tlb.reconcile(cpu);
        let index = Tlb::index(linear);
        if let Some(e) = tlb.entries[index].filter(|e| e.page == linear >> 12) {
            if !permitted(cpu, access, privilege, e.rights)
                && (cpu.machine_flags_mask == 0 || !set_accessed)
            {
                // SDM Vol 3A §5.10.4.1 p. 5-49: #PF invalidates the
                // faulting translation, including its instruction-source proof.
                tlb.remove(linear, Advance::FaultRemoved);
                return Err(PageFault {
                    linear,
                    access,
                    privilege,
                    protection: true,
                }
                .exit(cpu));
            }
            // Paging design §§3.2–3.3: re-walk a clean read fill before D-setting
            // and a 486 protection failure to perform both A updates.
            // SDM Vol 3A §5.10.2 permits eviction and observing edited tables.
            if permitted(cpu, access, privilege, e.rights)
                && (access != AccessKind::Write || e.translation.pte & DIRTY != 0)
            {
                if access == AccessKind::Execute {
                    tlb.entries[index].as_mut().unwrap().fetched = true;
                }
                return Ok(Translation {
                    physical: (e.translation.physical & FRAME) | (linear & 0xfff),
                    bytes: u32::from(bytes).min(0x1000 - (linear & 0xfff)) as u8,
                    ..e.translation
                });
            }
            tlb.remove(linear, Advance::Rewalked);
        }
    }
    loop {
        // 80386 PRM §5.2, Figs. 5-8–10 pp. 5-7–9; SDM Vol 3A §5.3
        // pp. 5-9–13: two 1024-entry levels and 4 KiB frames, no PSE.
        let pde_address = (cpu.legacy.cr3 & FRAME) + 4 * (linear >> 22);
        let pde = table_read(cpu, mem, linear, pde_address, PagingWalkLevel::Directory)?;
        let fault = |protection| PageFault {
            linear,
            access,
            privilege,
            protection,
        };
        if pde & PRESENT == 0 {
            return Err(fault(false).exit(cpu));
        }
        let pte_address = (pde & FRAME) + 4 * ((linear >> 12) & 0x3ff);
        let pte = table_read(cpu, mem, linear, pte_address, PagingWalkLevel::Table)?;
        if pte & PRESENT == 0 {
            return Err(fault(false).exit(cpu));
        }
        // ASSUMED (paging design §3.3): 80386 PRM §5.2.4.3 p. 5-10
        // sets A before a page read/write but leaves failed walks unspecified.
        // A public CPU test suite falsified setting PDE.A with PTE.P=0.
        // ASSUMED for I80386 (paging design §3.3): a public CPU test suite
        // for the 386 falsified A updates on protection faults. Intel486 PRM
        // §9.9.14 p. 9-21 requires both A bits on those faults on the 486.
        if cpu.machine_flags_mask == 0 && !permitted(cpu, access, privilege, pde & pte) {
            return Err(fault(true).exit(cpu));
        }
        let pde = if set_accessed {
            let Some(pde) = table_or(
                cpu,
                mem,
                linear,
                pde_address,
                PagingWalkLevel::Directory,
                pde,
                ACCESSED,
            )?
            else {
                continue;
            };
            pde
        } else {
            pde
        };
        let pte = if set_accessed {
            let Some(pte) = table_or(
                cpu,
                mem,
                linear,
                pte_address,
                PagingWalkLevel::Table,
                pte,
                ACCESSED,
            )?
            else {
                continue;
            };
            pte
        } else {
            pte
        };
        // 80386 PRM §§6.4.1–6.4.2 pp. 6-20–22, Table 6-5;
        // Intel486 PRM §4.1.3 p. 4-7 describes WP for user pages.
        // SDM Vol 3A §§5.1.3, 5.6.1 pp. 5-5, 5-33–35 applies WP to
        // both mapping classes (paging design Q3).
        let rights = pde & pte;
        if !permitted(cpu, access, privilege, rights) {
            return Err(fault(true).exit(cpu));
        }
        let translation = Translation {
            physical: (pte & FRAME) | (linear & 0xfff),
            bytes: u32::from(bytes).min(0x1000 - (linear & 0xfff)) as u8,
            pte_address,
            pte,
        };
        if set_accessed {
            if let Some(tlb) = mem.tlb() {
                tlb.borrow_mut()
                    .insert(linear, translation, rights, access == AccessKind::Execute);
            }
        }
        return Ok(translation);
    }
}

#[derive(Clone)]
struct Fragment {
    linear: u32,
    bus: u64,
    bytes: u8,
    shift: u8,
    translation: Translation,
    dirty_complete: bool,
}

fn complete_dirty<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    fragments: &mut [Fragment],
) -> Result<bool, Exit> {
    // 80386 PRM §5.2.4.3 p. 5-10; Intel486 PRM §5.3.4.3 pp. 5-21–22:
    // a permission probe does not complete the D update required by a write.
    for f in fragments {
        if f.dirty_complete {
            continue;
        }
        let t = f.translation;
        if table_or(
            cpu,
            mem,
            f.linear,
            t.pte_address,
            PagingWalkLevel::Table,
            t.pte,
            DIRTY,
        )?
        .is_none()
        {
            if let Some(tlb) = mem.tlb() {
                tlb.borrow_mut().remove(f.linear, Advance::DirtyFailed);
            }
            return Ok(false);
        }
        if let Some(tlb) = mem.tlb() {
            tlb.borrow_mut().dirty(f.linear, t);
        }
        f.dirty_complete = true;
    }
    Ok(true)
}

fn plan<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
    access: AccessKind,
    dirty: bool,
) -> Result<Vec<Fragment>, Exit> {
    // SDM Vol 3A §5.6.1 p. 5-33; 80386 PRM §6.5 p. 6-22:
    // implicit table/TSS and inner-stack accesses are supervisor at any CPL.
    let privilege = if cpu.legacy.cpl == 3 && !mem.supervisor() {
        Privilege::User
    } else {
        Privilege::Supervisor
    };
    let key = PlanKey {
        addr: addr as u32,
        size,
        access,
        privilege,
        cr3: cpu.legacy.cr3,
        cr0: cpu.legacy.cr0,
    };
    if let Some(tx) = mem.transaction() {
        let cached = tx
            .borrow()
            .plans
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, fragments)| fragments.clone());
        if let Some(mut fragments) = cached {
            if !dirty || preparing(mem) {
                return Ok(fragments);
            }
            if complete_dirty(cpu, mem, &mut fragments)? {
                let mut tx = tx.borrow_mut();
                tx.plans.iter_mut().find(|(k, _)| *k == key).unwrap().1 = fragments.clone();
                return Ok(fragments);
            }
            tx.borrow_mut().plans.retain(|(k, _)| *k != key);
        }
    }
    'restart: loop {
        let mut fragments = Vec::new();
        let mut consumed = 0;
        while consumed < size {
            let linear = (addr as u32).wrapping_add(u32::from(consumed));
            let translation =
                translate_linear(cpu, mem, linear, size - consumed, access, privilege, true)?;
            let (bus, bytes) = legacy::physical_span(
                cpu,
                mem.physical_mask(),
                u64::from(translation.physical),
                translation.bytes,
            );
            let bytes = mem
                .raw()
                .region_end(bus)
                .map_or(bytes, |end| u64::from(bytes).min(end - bus) as u8);
            fragments.push(Fragment {
                linear,
                bus,
                bytes,
                shift: consumed * 8,
                translation,
                dirty_complete: translation.pte & DIRTY != 0,
            });
            consumed += bytes;
        }
        if access == AccessKind::Write {
            for fragment in &fragments {
                probe_physical(
                    cpu,
                    mem,
                    u64::from(fragment.linear),
                    fragment.bus,
                    fragment.bytes,
                )?;
            }
        }
        if dirty && !preparing(mem) {
            // 80386 PRM §5.2.4.3 p. 5-10; Intel486 PRM §5.3.4.3
            // pp. 5-21–22: PTE.D precedes the operand write; PDE.D is preserved.
            // Preflight the entire span before D, retaining even aliased frames.
            if !complete_dirty(cpu, mem, &mut fragments)? {
                continue 'restart;
            }
        }
        if let Some(tx) = mem.transaction() {
            tx.borrow_mut().plans.push((key, fragments.clone()));
        }
        return Ok(fragments);
    }
}

fn commit_ram<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
) -> Result<(), Exit> {
    let mut tx = mem.transaction().expect("paged transaction").borrow_mut();
    let linears: Vec<_> = tx
        .plans
        .iter()
        .flat_map(|(_, spans)| spans)
        .map(|f| (f.bus, f.linear))
        .collect();
    for (bus, bytes, value, committed) in &mut tx.writes {
        if !*committed {
            if let Err(trap) = mem.raw().write(*bus, *value, *bytes) {
                let linear = linears.iter().find(|f| f.0 == *bus).map_or(0, |f| f.1);
                let StepResult::Exit(exit) =
                    trap_out(cpu, cpu.rip, trap, *bus, *bytes, AccessKind::Write, *value)
                        .into_step()
                else {
                    unreachable!()
                };
                let mut exit = crate::fault::qualify_legacy_bus(exit, u64::from(linear), *bus);
                if let Exit::UnmappedMemory {
                    metadata: Some(m), ..
                } = &mut exit
                {
                    m.progress.ram = MemoryEffects::PossiblyPartial;
                    m.progress.acknowledged = mmio_evidence(cpu);
                    if !m.progress.acknowledged.is_empty() {
                        m.progress.device = MemoryEffects::PossiblyPartial;
                    }
                }
                return Err(exit);
            }
            mem.note_write(*bus, *bytes);
            *committed = true;
        }
    }
    Ok(())
}

// Transfer visits belong to one retained instruction/iteration. Separate read
// and write cursors distinguish two accesses even when their bus bytes alias.
fn transfer_read<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    f: &Fragment,
) -> Result<u64, Exit> {
    let Some(cell) = mem.transaction() else {
        return read_physical(cpu, mem, u64::from(f.linear), f.bus, f.bytes);
    };
    let index = {
        let mut tx = cell.borrow_mut();
        let i = tx.read_cursor;
        tx.read_cursor += 1;
        i
    };
    if preparing(mem) {
        let result = read_physical(cpu, mem, u64::from(f.linear), f.bus, f.bytes);
        cell.borrow_mut()
            .prepared_reads
            .push((f.bus, f.bytes, result.as_ref().ok().copied()));
        return result;
    }
    if let Some(&(bus, bytes, value)) = cell.borrow().reads.get(index) {
        if (bus, bytes) != (f.bus, f.bytes) {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyEntryFacility,
            ));
        }
        return Ok(value);
    }
    let prepared = cell.borrow().prepared_reads.get(index).copied();
    let mut value = if let Some((bus, bytes, Some(value))) = prepared {
        if (bus, bytes) != (f.bus, f.bytes) {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyEntryFacility,
            ));
        }
        value
    } else {
        match mem.raw().read(f.bus, f.bytes) {
            Ok(value) => value,
            Err(MemTrap::Mmio) if cpu.pending_mmio.is_some() => {
                let value = cpu.pending_mmio.take().unwrap() & mask(f.bytes);
                cpu.record_mmio_part(cpu.rip, f.bus | ((index as u64) << 32), value);
                value
            }
            Err(trap) => return Err(transfer_failure(cpu, mem, trap, f, AccessKind::Read, 0)),
        }
    } & mask(f.bytes);
    // A later read in ENTER or a task save sees this attempt's staged RAM.
    let mut tx = cell.borrow_mut();
    for byte in 0..f.bytes {
        for &(bus, bytes, stored, committed) in tx.writes.iter().rev() {
            let at = f.bus + u64::from(byte);
            if !committed && (bus..bus + u64::from(bytes)).contains(&at) {
                let shift = byte * 8;
                value =
                    (value & !(0xff << shift)) | (((stored >> ((at - bus) * 8)) & 0xff) << shift);
                break;
            }
        }
    }
    tx.reads.push((f.bus, f.bytes, value));
    Ok(value)
}

fn transfer_failure<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    trap: MemTrap,
    f: &Fragment,
    access: AccessKind,
    value: u64,
) -> Exit {
    if trap == MemTrap::Mmio {
        if let Some(tx) = mem.transaction() {
            tx.borrow_mut().pending_mmio_linear = Some(f.linear);
        }
    }
    let StepResult::Exit(exit) =
        trap_out(cpu, cpu.rip, trap, f.bus, f.bytes, access, value).into_step()
    else {
        unreachable!()
    };
    crate::fault::qualify_legacy_bus(exit, u64::from(f.linear), f.bus)
}

fn transfer_write<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    f: &Fragment,
    value: u64,
) -> Result<(), Exit> {
    let Some(cell) = mem.transaction() else {
        return write_physical(cpu, mem, u64::from(f.linear), f.bus, f.bytes, value);
    };
    let value = value & mask(f.bytes);
    let index = {
        let mut tx = cell.borrow_mut();
        let i = tx.write_cursor;
        tx.write_cursor += 1;
        i
    };
    if let Some(&(bus, bytes, stored, _)) = cell.borrow().writes.get(index) {
        if (bus, bytes, stored) != (f.bus, f.bytes, value) {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyEntryFacility,
            ));
        }
        return Ok(());
    }
    let device = mem.raw().region_kind(f.bus, f.bytes) == Ok(crate::RegionKind::Trap);
    if device {
        if cpu.pending_mmio_write {
            cpu.pending_mmio_write = false;
            cpu.record_mmio_part(cpu.rip, f.bus | ((index as u64) << 32) | (1 << 63), value);
            mem.raw().note_write(f.bus, usize::from(f.bytes));
            mem.note_write(f.bus, f.bytes);
        } else {
            return Err(transfer_failure(
                cpu,
                mem,
                MemTrap::Mmio,
                f,
                AccessKind::Write,
                value,
            ));
        }
    }
    cell.borrow_mut()
        .writes
        .push((f.bus, f.bytes, value, device));
    Ok(())
}

pub(super) fn read<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
    access: AccessKind,
) -> Result<u64, Exit> {
    if let Some(operand) = mem.fast_operand() {
        let Some(window) = operand.window(cpu, mem, addr, size, false) else {
            return lane_read(cpu, mem, addr, size);
        };
        return mem
            .raw()
            .read(window.bus(addr), size)
            .map_err(|trap| window.failure(cpu, trap, AccessKind::Read, 0));
    }
    read_slow(cpu, mem, addr, size, access)
}

#[cold]
#[inline(never)]
fn read_slow<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
    access: AccessKind,
) -> Result<u64, Exit> {
    let fragments = plan(cpu, mem, addr, size, access, false)?;
    let mut value = 0;
    for f in fragments {
        let part = match transfer_read(cpu, mem, &f) {
            Err(Exit::MmioRead { .. }) if preparing(mem) => {
                if mem.supervisor() || !mem.transaction().unwrap().borrow().device_dummy {
                    return Err(refusal(
                        cpu,
                        crate::UnsupportedSystemOperation::LegacyInstruction,
                    ));
                }
                0
            }
            other => other?,
        };
        let part = if preparing(mem) {
            let mut tx = mem.transaction().unwrap().borrow_mut();
            let mut part = part;
            for byte in 0..f.bytes {
                if let Some((_, value)) = tx
                    .overlay
                    .iter()
                    .rev()
                    .find(|(at, _)| *at == f.bus + u64::from(byte))
                {
                    let shift = byte * 8;
                    part = (part & !(0xff << shift)) | (u64::from(*value) << shift);
                }
            }
            if let Some((_, _, snapshot)) = tx.prepared_reads.last_mut() {
                if snapshot.is_some() {
                    *snapshot = Some(part);
                }
            }
            part
        } else {
            part
        };
        value |= part << f.shift;
    }
    Ok(value)
}

// Control-transfer frame inputs cannot use dummy values: EFLAGS.VM selects
// both the frame extent and the segment-loading rules (design v86 §4.2).
#[cold]
#[inline(never)]
pub(super) fn read_frame<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
) -> Result<u64, Exit> {
    if mem.fast_operand().is_some() {
        return read(cpu, mem, addr, size, AccessKind::Read);
    }
    if let Some(tx) = mem.transaction() {
        tx.borrow_mut().frame_reads = true;
    }
    let fragments = plan(cpu, mem, addr, size, AccessKind::Read, false)?;
    let mut value = 0;
    for f in fragments {
        let part = transfer_read(cpu, mem, &f).inspect_err(|exit| {
            if matches!(exit, Exit::MmioRead { .. }) {
                if let Some(tx) = mem.transaction() {
                    tx.borrow_mut().pending_mmio_linear = Some(f.linear);
                }
            }
        })?;
        value |= part << f.shift;
    }
    Ok(value)
}

// Only bus completion outputs cross from a preparation preview. Architectural
// destinations remain in the original image until the complete return succeeds.
#[cold]
#[inline(never)]
fn retain_frame_answers(cpu: &mut CpuState, preview: &CpuState) {
    cpu.pending_mmio = preview.pending_mmio;
    cpu.mmio_parts = preview.mmio_parts;
    cpu.mmio_parts_len = preview.mmio_parts_len;
    cpu.mmio_parts_rip = preview.mmio_parts_rip;
    cpu.mmio_retry_active = preview.mmio_retry_active;
    cpu.mmio_retry_cs = preview.mmio_retry_cs;
}

pub(super) fn write<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
    value: u64,
) -> Result<(), Exit> {
    if let Some(operand) = mem.fast_operand() {
        let Some(window) = operand.window(cpu, mem, addr, size, true) else {
            return lane_write(cpu, mem, addr, size, value);
        };
        let value = value & mask(size);
        let bus = window.bus(addr);
        mem.raw()
            .write(bus, value, size)
            .map_err(|trap| window.failure(cpu, trap, AccessKind::Write, value))?;
        mem.note_write(bus, size);
        return Ok(());
    }
    write_slow(cpu, mem, addr, size, value)
}

#[cold]
#[inline(never)]
fn write_slow<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
    value: u64,
) -> Result<(), Exit> {
    let fragments = plan(cpu, mem, addr, size, AccessKind::Write, true)?;
    for f in fragments {
        if preparing(mem) {
            let mut tx = mem.transaction().unwrap().borrow_mut();
            for byte in 0..f.bytes {
                tx.overlay.push((
                    f.bus + u64::from(byte),
                    (value >> (f.shift + byte * 8)) as u8,
                ));
            }
            continue;
        }
        transfer_write(cpu, mem, &f, value >> f.shift)?;
    }
    Ok(())
}

#[cold]
#[inline(never)]
pub(super) fn probe<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
    size: u8,
) -> Result<(), Exit> {
    if let Some(operand) = mem.fast_operand() {
        return match operand.window(cpu, mem, addr, size, true) {
            Some(_) => Ok(()),
            None => lane_access(cpu, mem, addr, size, LaneAccess::Probe).map(|_| ()),
        };
    }
    plan(cpu, mem, addr, size, AccessKind::Write, false).map(|_| ())
}

#[cold]
#[inline(never)]
pub(super) fn check_write_permission<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    addr: u64,
) -> Result<(), Exit> {
    if mem.fast_operand().is_some() {
        // The single-push ENTER lane checked the allocation address too.
        return Ok(());
    }
    // SDM Vol 2A §3.3 ENTER pp. 3-295–297 checks whether a write would
    // #PF without transferring an operand. ASSUMED: no A/D updates for
    // this check; neither ENTER nor Vol 3A §5.8 specifies them without
    // an access. Do not retain an operand plan or check physical backing.
    let privilege = if cpu.legacy.cpl == 3 && !mem.supervisor() {
        Privilege::User
    } else {
        Privilege::Supervisor
    };
    translate_linear(
        cpu,
        mem,
        addr as u32,
        1,
        AccessKind::Write,
        privilege,
        false,
    )
    .map(|_| ())
}

// Fetch performs the same read-permission walk but retains execute metadata.
// 80386 PRM §9.8.14 Fig. 9-8 pp. 9-17–18; Intel486 PRM §9.9.14
// Fig. 9-7 pp. 9-21–22: no I/D bit exists on these models.
pub(crate) fn translate_fetch(
    cpu: &mut CpuState,
    memory: &Memory,
    linear: u32,
    wrote: &Cell<bool>,
    tlb: Option<&std::cell::RefCell<Tlb>>,
) -> Result<u64, Exit> {
    if memory.host_fault_recovery() {
        fetch_bus::<true>(cpu, memory, linear, wrote, tlb)
    } else {
        fetch_bus::<false>(cpu, memory, linear, wrote, tlb)
    }
}

fn fetch_bus<const ROUTED: bool>(
    cpu: &mut CpuState,
    memory: &Memory,
    linear: u32,
    wrote: &Cell<bool>,
    tlb: Option<&std::cell::RefCell<Tlb>>,
) -> Result<u64, Exit> {
    let privilege = if cpu.legacy.cpl == 3 {
        Privilege::User
    } else {
        Privilege::Supervisor
    };
    let observe = |_, _| wrote.set(true);
    let mut mem = LegacyAccess::<ROUTED, 0, true>::new(memory, u32::MAX);
    mem.writes = Some(&observe);
    mem.paging.tlb = tlb;
    let t = translate_linear(cpu, &mem, linear, 1, AccessKind::Execute, privilege, true)?;
    Ok(legacy::physical_span(cpu, u32::MAX, u64::from(t.physical), 1).0)
}

#[cfg(feature = "test-hooks")]
pub(crate) fn scalar_for_test(
    code: &[u8],
    model: crate::CpuModel,
    cpu: &mut CpuState,
    memory: &Memory,
) -> Result<(), Exit> {
    if !matches!(
        model,
        crate::CpuModel::I80386 | crate::CpuModel::I80486Dx | crate::CpuModel::I80486DxCpuid
    ) || cpu.legacy.cr0 & 0x8000_0001 != 0x8000_0001
    {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyEntryFacility,
        ));
    }
    cpu.machine_flags_mask = model.machine_flags_mask();
    let context = legacy::derive_legacy_context(model.legacy_model(), cpu).map_err(|error| {
        refusal(
            cpu,
            crate::UnsupportedSystemOperation::InvalidLegacyContext(error),
        )
    })?;
    let ir = crate::lift::legacy::instruction_for_profile(
        code,
        cpu.rip as u32,
        context,
        model.profile(),
        crate::lift::legacy::alignment_mask(cpu),
    )
    .map_err(|_| refusal(cpu, crate::UnsupportedSystemOperation::LegacyInstruction))?;
    execute_for_test::<true, false>(
        &ir,
        cpu,
        memory,
        context,
        &mut tasks::SwitchState::default(),
    )
    .map(|_| ())
}

#[cfg(feature = "test-hooks")]
pub(crate) fn execute_for_test<const PAGED: bool, const V86: bool>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
) -> Result<(Outcome, Option<PortPlan>), Exit>
where
    PagingMode<PAGED>: PagingStorage,
{
    let mut scratch = vec![0; ir.temp_count as usize];
    let mem = LegacyAccess::<false, 0, PAGED>::new(memory, u32::MAX);
    let view = InstructionView {
        guest_start: ir.guest_start,
        guest_len: ir.guest_len,
        ops: &ir.ops,
        proven: false,
    };
    if PAGED {
        return execute::<false, 0, PAGED, V86>(
            ir,
            &view,
            cpu,
            &mem,
            &mut scratch,
            context,
            task_switch,
        );
    }
    execute_instruction::<false, 0, PAGED, V86>(
        ir,
        &InstructionView {
            guest_start: ir.guest_start,
            guest_len: ir.guest_len,
            ops: &ir.ops,
            proven: false,
        },
        cpu,
        &LegacyAccess::<false, 0, PAGED>::new(memory, u32::MAX),
        &mut scratch,
        context,
        task_switch,
    )
    .map(|outcome| (outcome, None))
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PlanKey {
    addr: u32,
    size: u8,
    access: AccessKind,
    privilege: Privilege,
    cr3: u32,
    cr0: u32,
}

// One instruction/REP iteration owns these spans. No page-table lookup after
// the first operand store: operand bytes may themselves alias a later PTE.
pub(crate) struct PortPlan {
    inputs: Box<CpuState>,
    mapping_identity: std::sync::Arc<()>,
    spans: Vec<Fragment>,
    completed: usize,
    pending_mmio_linear: Option<u32>,
}

impl PortPlan {
    pub(crate) fn pending_mmio_linear_address(&self) -> Option<u32> {
        self.pending_mmio_linear
    }
    pub(crate) fn matches_owner(&self, cpu: &CpuState, memory: &Memory) -> bool {
        let Some(p) = cpu.port_string_in else {
            return false;
        };
        matches_cpu_inputs(&self.inputs, cpu)
            && std::sync::Arc::ptr_eq(&self.mapping_identity, &memory.mapping_identity())
            && self.inputs.pending_mmio == cpu.pending_mmio
            && (self.pending_mmio_linear.is_some()
                || self.inputs.pending_mmio_write == cpu.pending_mmio_write)
            && self.inputs.port_string_in.is_some_and(|owner| {
                // Only the requested port value may change the retained handoff.
                owner.rip == p.rip
                    && owner.next_ip == p.next_ip
                    && owner.instruction_end == p.instruction_end
                    && owner.base == p.base
                    && owner.offset == p.offset
                    && owner.next_index == p.next_index
                    && owner.next_count == p.next_count
                    && owner.addr_bits == p.addr_bits
                    && owner.elem == p.elem
                    && owner.rep == p.rep
                    && match (owner.value, p.value, cpu.pending_port_in) {
                        (None, None, Some(size)) => size == p.elem,
                        (None, Some(_), None) => true,
                        (Some(before), Some(after), None) => before == after,
                        _ => false,
                    }
            })
    }
}

#[derive(Default)]
pub(crate) struct Transaction {
    mapping_identity: Option<std::sync::Arc<()>>,
    plans: Vec<(PlanKey, Vec<Fragment>)>,
    overlay: Vec<(u64, u8)>,
    preparing: bool,
    device_dummy: bool,
    frame_reads: bool,
    reads: Vec<(u64, u8, u64)>,
    prepared_reads: Vec<(u64, u8, Option<u64>)>,
    writes: Vec<(u64, u8, u64, bool)>,
    read_cursor: usize,
    write_cursor: usize,
    inputs: Option<Box<CpuState>>,
    task_phase: Option<(tasks::Stage, usize, usize)>,
    pending_mmio_linear: Option<u32>,
}

impl Transaction {
    pub(crate) fn matches_memory(&self, memory: &Memory) -> bool {
        same_arc(&self.mapping_identity, &Some(memory.mapping_identity()))
    }

    pub(crate) fn pending_mmio_linear_address(&self) -> Option<u32> {
        self.pending_mmio_linear
    }

    pub(crate) fn matches_inputs(&self, cpu: &CpuState) -> bool {
        let Some(inputs) = &self.inputs else {
            return true;
        };
        matches_cpu_inputs(inputs, cpu)
    }
}

fn matches_cpu_inputs(inputs: &CpuState, cpu: &CpuState) -> bool {
    macro_rules! same {
            ($($field:ident),* $(,)?) => {{
                let CpuState {
                    $($field: _,)*
                    flags: _, fpu_pad: _, pending_port_in: _,
                    pending_mmio: _, pending_mmio_write: _, port_string_in: _,
                    mmio_parts: _, mmio_parts_len: _, mmio_parts_rip: _,
                    mmio_retry_active: _, mmio_retry_cs: _,
                    cpuid_table: _, cpu_source: _,
                } = cpu;
                true $(&& inputs.$field == cpu.$field)*
            }};
        }
    // Completion slots and the transfer ledger are outputs of the bus handoff.
    // Numeric and system inputs own the retained image just as GPRs do.
    same!(
        gpr,
        rip,
        rflags_system,
        legacy,
        cs,
        ds,
        es,
        ss,
        fs,
        gs,
        fs_base,
        gs_base,
        xmm,
        ymm_hi,
        zmm_hi,
        kmask,
        mxcsr,
        fpr,
        fpu_top,
        fpu_cw,
        fpu_sw,
        fpu_empty,
        fpu_fip,
        fpu_fdp,
        fpu_fop,
        fpu_fcs,
        fpu_fds,
        fpu_model,
        fpu_emulation,
        x87_precision,
        x87_pointer_model,
        features,
        machine_flags_mask,
        debug_test_registers,
        numeric_error_route,
        external_numeric_error,
        task_io_map_base,
        task_debug_trap,
        cpuid_overrides,
        source_id,
        random_state,
    ) && inputs.flags == cpu.flags
        && same_arc(&inputs.cpuid_table, &cpu.cpuid_table)
        && same_arc(&inputs.cpu_source, &cpu.cpu_source)
}

fn same_arc<T: ?Sized>(a: &Option<std::sync::Arc<T>>, b: &Option<std::sync::Arc<T>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

fn preparing<const ROUTED: bool, const ORDER: u8>(mem: &impl LegacyMemory<ROUTED, ORDER>) -> bool {
    mem.transaction().is_some_and(|tx| tx.borrow().preparing)
}

#[cold]
#[inline(never)]
fn finish_plan<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    tx: &std::cell::RefCell<Transaction>,
) -> Result<bool, Exit> {
    let mut tx = tx.borrow_mut();
    let mut dirtied = false;
    for (key, spans) in &tx.plans {
        if key.access != AccessKind::Write {
            continue;
        }
        for f in spans {
            if f.dirty_complete {
                continue;
            }
            dirtied |= f.translation.pte & DIRTY == 0;
            if table_or(
                cpu,
                mem,
                f.linear,
                f.translation.pte_address,
                PagingWalkLevel::Table,
                f.translation.pte,
                DIRTY,
            )?
            .is_none()
            {
                if let Some(tlb) = mem.tlb() {
                    tlb.borrow_mut().remove(f.linear, Advance::DirtyFailed);
                }
                return Ok(false);
            }
            if let Some(tlb) = mem.tlb() {
                tlb.borrow_mut().dirty(f.linear, f.translation);
            }
        }
    }
    // A source can alias a PTE whose D bit was just set. Re-evaluate the
    // prepared instruction with those physical walk writes already visible.
    if !dirtied {
        for (key, spans) in &mut tx.plans {
            if key.access == AccessKind::Write {
                for f in spans {
                    f.dirty_complete = true;
                }
            }
        }
    }
    Ok(!dirtied)
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute<const ROUTED: bool, const ORDER: u8, const PAGED: bool, const V86: bool>(
    block: &IrBlock,
    ir: &InstructionView<'_>,
    cpu: &mut CpuState,
    original: &LegacyAccess<'_, ROUTED, ORDER, PAGED>,
    temps: &mut [u64],
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
) -> Result<(Outcome, Option<PortPlan>), Exit>
where
    PagingMode<PAGED>: PagingStorage,
{
    execute_with_retry::<ROUTED, ORDER, PAGED, V86>(
        block,
        ir,
        cpu,
        original,
        temps,
        context,
        task_switch,
        &mut None,
    )
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_with_retry<
    const ROUTED: bool,
    const ORDER: u8,
    const PAGED: bool,
    const V86: bool,
>(
    block: &IrBlock,
    ir: &InstructionView<'_>,
    cpu: &mut CpuState,
    original: &LegacyAccess<'_, ROUTED, ORDER, PAGED>,
    temps: &mut [u64],
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
    retry: &mut Option<Transaction>,
) -> Result<(Outcome, Option<PortPlan>), Exit>
where
    PagingMode<PAGED>: PagingStorage,
{
    loop {
        if retry
            .as_ref()
            .is_some_and(|tx| !tx.matches_memory(original.raw()))
        {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyEntryFacility,
            ));
        }
        let retained = retry.take();
        let resuming = retained.is_some();
        let tx = std::cell::RefCell::new(retained.unwrap_or_else(|| Transaction {
            mapping_identity: Some(original.raw().mapping_identity()),
            preparing: true,
            device_dummy: ir.ops.iter().all(|op| match op {
                IrOp::Legacy(op) => matches!(
                    op,
                    L::Checkpoint
                        | L::I486Gate
                        | L::Address { .. }
                        | L::Flat32Address { .. }
                        | L::SegmentedAddress { .. }
                        | L::ReadSelector { .. }
                        | L::LiveField { .. }
                        | L::Push { .. }
                        | L::Pop { .. }
                        | L::AdjustStack { .. }
                        | L::String { .. }
                        | L::Enter { .. }
                        | L::Leave { .. }
                        | L::All { .. }
                        | L::Flags { .. }
                        | L::Table { .. }
                        | L::Privileged { .. }
                        | L::LoadMsw { .. }
                        | L::Coprocessor { .. }
                        | L::CoprocessorOverrun { .. }
                ),
                IrOp::Jump { .. } | IrOp::Branch { .. } => false,
                _ => true,
            }),
            ..Transaction::default()
        }));
        {
            let mut tx = tx.borrow_mut();
            // Task continuations resume their saved phase; ordinary instructions
            // replay from their checkpoint and consume their earlier visits.
            let task_resume = resuming && task_switch.context().is_some();
            let (read, write) = if task_resume {
                tx.task_phase
                    .map_or((tx.reads.len(), tx.writes.len()), |(_, read, write)| {
                        (read, write)
                    })
            } else {
                (0, 0)
            };
            tx.read_cursor = read;
            tx.write_cursor = write;
        }
        let mut mem =
            LegacyAccess::<ROUTED, ORDER, true>::new(&original.access.0, original.physical_mask);
        mem.writes = original.writes;
        mem.paging.transaction = Some(&tx);
        mem.paging.tlb = original.tlb();
        if !resuming || tx.borrow().preparing {
            if resuming {
                tx.borrow_mut().prepared_reads.clear();
                tx.borrow_mut().overlay.clear();
            }
            let mut preview_cpu = cpu.clone();
            let mut preview_tasks = task_switch.clone();
            let mut preview_temps = temps.to_vec();
            let preview = execute_instruction::<ROUTED, ORDER, true, V86>(
                block,
                ir,
                &mut preview_cpu,
                &mem,
                &mut preview_temps,
                context,
                &mut preview_tasks,
            );
            if let Err(exit) = preview {
                if tx.borrow().frame_reads && matches!(exit, Exit::MmioRead { .. }) {
                    retain_frame_answers(cpu, &preview_cpu);
                    tx.borrow_mut().inputs = Some(Box::new(cpu.clone()));
                    *retry = Some(tx.into_inner());
                    return Err(exit);
                }
                if !preview_tasks.committed {
                    publish_cr2(cpu, &exit);
                    return Err(exit);
                }
                // Hardware task switches have their own commit boundary. Replay the
                // outgoing saves and incoming commit even when the new task faults
                // (80386 PRM §7.5 pp. 7-8–9; design §3.7).
            }
            if !finish_plan(cpu, &mem, &tx)? {
                continue;
            }
            let mut tx = tx.borrow_mut();
            tx.preparing = false;
            tx.read_cursor = 0;
        }
        let before_commit = cpu.clone();
        let result = execute_instruction::<ROUTED, ORDER, true, V86>(
            block,
            ir,
            cpu,
            &mem,
            temps,
            context,
            task_switch,
        );
        let outcome = match result {
            Ok(outcome) => outcome,
            Err(exit) => {
                if matches!(
                    exit,
                    Exit::MmioRead { .. } | Exit::MmioWrite { .. } | Exit::UnmappedMemory { .. }
                ) {
                    tx.borrow_mut().inputs = Some(Box::new(cpu.clone()));
                    *retry = Some(tx.into_inner());
                }
                return Err(exit);
            }
        };
        // RAM stores commit only after every irreversible device transfer has
        // completed. Reads below see the staged stores in instruction order.
        let next_ip = cpu.rip;
        if !task_switch.committed {
            cpu.rip = before_commit.rip;
        }
        if let Err(exit) = commit_ram(cpu, &mem) {
            if !task_switch.committed {
                restore_destinations(cpu, before_commit);
            }
            tx.borrow_mut().inputs = Some(Box::new(cpu.clone()));
            *retry = Some(tx.into_inner());
            return Err(exit);
        }
        cpu.rip = next_ip;
        let input = cpu.port_string_in.map(|p| {
            let linear = (p.base as u32).wrapping_add(p.offset as u32);
            let spans = tx
                .borrow()
                .plans
                .iter()
                .find(|(key, _)| {
                    key.addr == linear && key.size == p.elem && key.access == AccessKind::Write
                })
                .expect("preflighted INS destination")
                .1
                .clone();
            PortPlan {
                inputs: Box::new(cpu.clone()),
                mapping_identity: original.raw().mapping_identity(),
                spans,
                completed: 0,
                pending_mmio_linear: None,
            }
        });
        return Ok((outcome, input));
    }
}

#[cold]
#[inline(never)]
pub(super) fn byte_bus<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u64,
    access: AccessKind,
) -> Result<u64, Exit> {
    Ok(plan(cpu, mem, linear, 1, access, false)?[0].bus)
}

#[cold]
#[inline(never)]
pub(super) fn probe_descriptor_cas<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    address: u32,
) -> Result<(), Exit> {
    // A lane proved access by access (`tlb_lane`) never walks: a descriptor
    // whose A bit is clear replays on the transaction path before any write.
    if mem
        .fast_operand()
        .is_some_and(FastOperand::proves_each_access)
    {
        mem.fast_miss();
        return Err(fast_miss_exit(cpu));
    }
    let spans = plan(cpu, mem, u64::from(address), 1, AccessKind::Write, false)?;
    if mem.raw().region_kind(spans[0].bus, 1) == Ok(crate::RegionKind::Trap) {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyInstruction,
        ));
    }
    Ok(())
}

#[cold]
#[inline(never)]
pub(super) fn descriptor_cas<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    address: u32,
    expected: u64,
    value: u64,
) -> Result<u64, Exit> {
    probe_descriptor_cas(cpu, mem, address)?;
    if preparing(mem) {
        let observed = read(cpu, mem, u64::from(address), 1, AccessKind::Write)?;
        // The committing phase uses a physical CAS rather than an operand
        // read visit. Keep the ordinary operand snapshot cursor aligned.
        if let Some(tx) = mem.transaction() {
            let mut tx = tx.borrow_mut();
            tx.prepared_reads.pop();
            tx.read_cursor -= 1;
        }
        if observed == expected {
            write(cpu, mem, u64::from(address), 1, value)?;
        }
        return Ok(observed);
    }
    let f = &plan(cpu, mem, u64::from(address), 1, AccessKind::Write, true)?[0];
    // Inside a strict lane this plan commits the page's A/D and the CAS the
    // descriptor's A before the frame stores; no later access may miss.
    #[cfg(debug_assertions)]
    if mem.fast_operand().is_some() {
        LANE_COMMITTED.with(|committed| committed.set(true));
    }
    let observed = mem
        .raw()
        .atomic_cas(f.bus, expected, value, 1)
        .map_err(|trap| {
            let StepResult::Exit(exit) =
                trap_out(cpu, cpu.rip, trap, f.bus, 1, AccessKind::Write, value).into_step()
            else {
                unreachable!()
            };
            crate::fault::qualify_legacy_bus(exit, u64::from(address), f.bus)
        })?;
    if observed == expected {
        mem.note_write(f.bus, 1);
    }
    Ok(observed)
}

#[cold]
#[inline(never)]
pub(super) fn locked<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    linear: u64,
    src: u64,
    size: u8,
    update: super::super::integer::LockedUpdate,
) -> Result<u64, Exit> {
    let spans = plan(cpu, mem, linear, size, AccessKind::Write, true)?;
    if !preparing(mem) && spans.len() == 1 {
        let f = &spans[0];
        let result = match update {
            super::super::integer::LockedUpdate::Cas(expected) => {
                mem.raw().atomic_cas(f.bus, expected, src, size)
            }
            super::super::integer::LockedUpdate::Rmw(op) => {
                mem.raw().atomic_rmw(f.bus, src, size, op)
            }
        };
        match result {
            Ok(value) => {
                mem.note_write(f.bus, size);
                return Ok(value);
            }
            Err(MemTrap::Mmio) => {}
            Err(trap) => {
                let StepResult::Exit(exit) =
                    trap_out(cpu, cpu.rip, trap, f.bus, size, AccessKind::Write, src).into_step()
                else {
                    unreachable!()
                };
                return Err(crate::fault::qualify_legacy_bus(exit, linear, f.bus));
            }
        }
    }
    // SDM Vol 3A §11.1.2.2 pp. 11-4–5; design §3.4: one retained
    // snapshot for the scatter operand, with the existing single-vCPU contract.
    // The transaction retains reads and completed writes across bus handoffs.
    let old = read(cpu, mem, linear, size, AccessKind::Write)?;
    write(cpu, mem, linear, size, update.apply(old, src, size))?;
    Ok(old)
}

// Architectural rollback must not resurrect an already consumed device answer.
// The translated transaction and its transfer visits live outside CpuState.
fn restore_destinations(cpu: &mut CpuState, mut before: CpuState) {
    std::mem::swap(&mut before.pending_mmio, &mut cpu.pending_mmio);
    std::mem::swap(&mut before.pending_mmio_write, &mut cpu.pending_mmio_write);
    std::mem::swap(&mut before.mmio_parts, &mut cpu.mmio_parts);
    std::mem::swap(&mut before.mmio_parts_len, &mut cpu.mmio_parts_len);
    std::mem::swap(&mut before.mmio_parts_rip, &mut cpu.mmio_parts_rip);
    std::mem::swap(&mut before.mmio_retry_active, &mut cpu.mmio_retry_active);
    std::mem::swap(&mut before.mmio_retry_cs, &mut cpu.mmio_retry_cs);
    *cpu = before;
}

struct FpBuffer {
    base: u64,
    bytes: std::cell::RefCell<Vec<u8>>,
    stored: Cell<bool>,
}
impl crate::x87::FpMem for FpBuffer {
    fn load(&self, addr: u64, out: &mut [u8]) -> bool {
        let offset = (addr as u32).wrapping_sub(self.base as u32) as usize;
        out.copy_from_slice(&self.bytes.borrow()[offset..offset + out.len()]);
        true
    }
    fn store(&self, addr: u64, bytes: &[u8]) -> bool {
        let offset = (addr as u32).wrapping_sub(self.base as u32) as usize;
        self.bytes.borrow_mut()[offset..offset + bytes.len()].copy_from_slice(bytes);
        self.stored.set(true);
        true
    }
}

#[cold]
#[inline(never)]
pub(super) fn x87<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    kind: crate::x87::FpuKind,
    address: u64,
    sti: u8,
) -> Result<(), Exit> {
    let bytes = crate::x87::mem_operand_bytes_format(kind, sti) as usize;
    let writing = crate::x87::mem_write_bytes(kind).is_some();
    // Strong image restart is engine policy (design §3.4). SDM Vol 3A
    // §11.1.1 pp. 11-2–3 permits partial stores for some x87 instructions.
    let mut data = vec![0; bytes];
    for offset in (0..bytes).step_by(8) {
        let linear = u64::from((address as u32).wrapping_add(offset as u32));
        let size = (bytes - offset).min(8) as u8;
        if !writing {
            plan(cpu, mem, linear, size, AccessKind::Read, false)?;
        }
    }
    if !writing {
        for offset in (0..bytes).step_by(8) {
            let linear = u64::from((address as u32).wrapping_add(offset as u32));
            let size = (bytes - offset).min(8) as u8;
            let value = read(cpu, mem, linear, size, AccessKind::Read)?;
            data[offset..offset + usize::from(size)]
                .copy_from_slice(&value.to_le_bytes()[..usize::from(size)]);
        }
    }
    let fp = FpBuffer {
        base: address,
        bytes: std::cell::RefCell::new(data),
        stored: Cell::new(false),
    };
    let before = cpu.clone();
    let fault = crate::x87::exec_x87(cpu, &fp, kind, address, sti);
    debug_assert!(fault.is_none());
    if fp.stored.get() {
        for offset in (0..bytes).step_by(8) {
            let linear = u64::from((address as u32).wrapping_add(offset as u32));
            if let Err(exit) = probe(cpu, mem, linear, (bytes - offset).min(8) as u8) {
                restore_destinations(cpu, before);
                return Err(exit);
            }
        }
        let data = fp.bytes.borrow();
        for offset in (0..bytes).step_by(8) {
            let size = (bytes - offset).min(8);
            let mut value = [0; 8];
            value[..size].copy_from_slice(&data[offset..offset + size]);
            let linear = u64::from((address as u32).wrapping_add(offset as u32));
            if let Err(exit) = write(cpu, mem, linear, size as u8, u64::from_le_bytes(value)) {
                restore_destinations(cpu, before);
                return Err(exit);
            }
        }
    }
    Ok(())
}

#[cold]
#[inline(never)]
pub(crate) fn deliver_interrupt<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    memory: &Memory,
    context: MachineContext,
    event: legacy::LegacyEvent,
    task_switch: &mut tasks::SwitchState,
    retry: &mut Option<Transaction>,
    tlb: &std::cell::RefCell<Tlb>,
) -> Result<(), Exit> {
    // A preflight fault belongs to this delivery, even after a previous task
    // completed. Match the unpaged delivery's per-attempt completion boundary.
    task_switch.committed = false;
    task_switch.completed = false;
    if retry.is_none() && task_switch.context().is_none() {
        if let Some(result) =
            deliver_fast::<ROUTED, ORDER>(cpu, memory, context, event, task_switch, tlb)
        {
            return result;
        }
    }
    loop {
        if retry.as_ref().is_some_and(|tx| !tx.matches_memory(memory)) {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyEntryFacility,
            ));
        }
        let retained = retry.take();
        let resuming = retained.is_some();
        let tx = std::cell::RefCell::new(retained.unwrap_or_else(|| Transaction {
            mapping_identity: Some(memory.mapping_identity()),
            preparing: true,
            ..Transaction::default()
        }));
        {
            let mut tx = tx.borrow_mut();
            // Task continuations resume their saved phase; ordinary instructions
            // replay from their checkpoint and consume their earlier visits.
            let task_resume = resuming && task_switch.context().is_some();
            let (read, write) = if task_resume {
                tx.task_phase
                    .map_or((tx.reads.len(), tx.writes.len()), |(_, read, write)| {
                        (read, write)
                    })
            } else {
                (0, 0)
            };
            tx.read_cursor = read;
            tx.write_cursor = write;
        }
        let mut mem = LegacyAccess::<ROUTED, ORDER, true>::new(memory, u32::MAX);
        mem.paging.transaction = Some(&tx);
        mem.paging.tlb = Some(tlb);
        if !resuming {
            let mut preview_cpu = cpu.clone();
            let mut preview_tasks = task_switch.clone();
            let preview = if preview_tasks.context().is_some() {
                tasks::resume(&mut preview_cpu, &mem.address_keyed(), &mut preview_tasks)
            } else {
                deliver(&mut preview_cpu, &mem, context, event, &mut preview_tasks)
            };
            if let Err(exit) = preview {
                if !preview_tasks.committed {
                    publish_cr2(cpu, &exit);
                    return Err(exit);
                }
            }
            if !finish_plan(cpu, &mem, &tx)? {
                continue;
            }
            let mut tx = tx.borrow_mut();
            tx.preparing = false;
            tx.read_cursor = 0;
        }
        let before_commit = cpu.clone();
        let mut result = if task_switch.context().is_some() {
            tasks::resume(cpu, &mem.address_keyed(), task_switch)
        } else {
            deliver(cpu, &mem, context, event, task_switch)
        };
        if result.is_ok() {
            let next_ip = cpu.rip;
            if !task_switch.committed {
                cpu.rip = before_commit.rip;
            }
            if let Err(exit) = commit_ram(cpu, &mem) {
                if !task_switch.committed {
                    restore_destinations(cpu, before_commit);
                }
                result = Err(exit);
            } else {
                cpu.rip = next_ip;
            }
        }
        if let Err(exit) = &result {
            publish_cr2(cpu, exit);
            if matches!(
                exit,
                Exit::MmioRead { .. } | Exit::MmioWrite { .. } | Exit::UnmappedMemory { .. }
            ) {
                tx.borrow_mut().inputs = Some(Box::new(cpu.clone()));
                *retry = Some(tx.into_inner());
            }
        }
        return result;
    }
}

#[cold]
#[inline(never)]
pub(crate) fn finish_port_input<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    memory: &Memory,
    plan: &mut Option<PortPlan>,
) -> Result<(), Exit> {
    let p = cpu.port_string_in.expect("pending input iteration");
    let saved = plan
        .as_mut()
        .filter(|saved| saved.matches_owner(cpu, memory))
        .ok_or_else(|| refusal(cpu, crate::UnsupportedSystemOperation::LegacyEntryFacility))?;
    let Some(value) = p.value else {
        return Err(Exit::port_io(cpu.gpr[RDX] as u16, p.elem, PortDir::In, 0));
    };
    saved.inputs.port_string_in.as_mut().unwrap().value = Some(value);
    let mem = LegacyAccess::<ROUTED, ORDER, true>::new(memory, u32::MAX);
    // The port read is irreversible. Retain this iteration's translated spans
    // over the handoff, even if the embedder edits its PTEs before completion.
    while saved.completed < saved.spans.len() {
        let f = &saved.spans[saved.completed];
        let value = (value >> f.shift) & mask(f.bytes);
        if mem.raw().region_kind(f.bus, f.bytes) == Ok(crate::RegionKind::Trap) {
            if !cpu.pending_mmio_write {
                saved.pending_mmio_linear = Some(f.linear);
                return Err(transfer_failure(
                    cpu,
                    &mem,
                    MemTrap::Mmio,
                    f,
                    AccessKind::Write,
                    value,
                ));
            }
            cpu.pending_mmio_write = false;
            cpu.record_mmio_part(
                cpu.rip,
                f.bus | ((saved.completed as u64) << 32) | (1 << 63),
                value,
            );
            mem.raw().note_write(f.bus, usize::from(f.bytes));
        } else if let Err(trap) = mem.raw().write(f.bus, value, f.bytes) {
            return Err(transfer_failure(
                cpu,
                &mem,
                trap,
                f,
                AccessKind::Write,
                value,
            ));
        }
        mem.note_write(f.bus, f.bytes);
        saved.completed += 1;
        saved.pending_mmio_linear = None;
    }
    *plan = None;
    cpu.write_gpr(RDI, p.next_index, p.addr_bits / 8);
    if p.rep {
        cpu.write_gpr(RCX, p.next_count, p.addr_bits / 8);
    }
    cpu.rip = p.next_ip;
    cpu.port_string_in = None;
    cpu.clear_mmio_parts();
    Ok(())
}

// A task phase can contain several reads (the incoming image or a segment
// descriptor). Retry that phase's visits without replaying earlier phases.
#[cold]
#[inline(never)]
pub(super) fn task_phase<const ROUTED: bool, const ORDER: u8>(
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    stage: tasks::Stage,
) {
    if preparing(mem) {
        return;
    }
    let mut tx = mem
        .transaction()
        .expect("paged task transaction")
        .borrow_mut();
    if tx.task_phase.is_none_or(|(old, _, _)| old != stage) {
        tx.task_phase = Some((stage, tx.read_cursor, tx.write_cursor));
    }
}

// SDM Vol 3A §10.3 steps 6, 10 pp. 10-10–11: outgoing saves precede
// clearing the outgoing busy bit, even when their physical storage overlaps.
#[cold]
#[inline(never)]
pub(super) fn task_saves_complete<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
) -> Result<(), Exit> {
    if preparing(mem) {
        return Ok(());
    }
    commit_ram(cpu, mem)
}

// Task switches have an architectural commit seam; ordinary instruction
// preflight must stop there rather than roll an incoming-task fault back.
#[cold]
#[inline(never)]
pub(super) fn task_commit<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
) -> Result<(), Exit> {
    if preparing(mem) {
        // Internal preflight stop, consumed only after SwitchState::committed.
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyTaskPaging,
        ));
    }
    commit_ram(cpu, mem)?;
    // Keep the outgoing plans through the incoming image read. The saves may
    // alias its paging structures; that image was preflighted before commit.
    Ok(())
}

#[cold]
#[inline(never)]
pub(super) fn task_cr3_loaded<const ROUTED: bool, const ORDER: u8>(
    mem: &impl LegacyMemory<ROUTED, ORDER>,
) {
    if let Some(tlb) = mem.tlb() {
        tlb.borrow_mut().load_cr3();
    }
    // Even an equal CR3 load starts fresh translations. Outgoing TSS saves
    // can alias the page tables used by incoming descriptor loads (§3.7).
    mem.transaction()
        .expect("paged task transaction")
        .borrow_mut()
        .plans
        .clear();
}

#[cold]
#[inline(never)]
pub(super) fn task_resident<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    tss: legacy::SystemSegment,
) -> Result<(), Exit> {
    // SDM Vol 3A §10.3 step 5 p. 10-10: the entire incoming TSS image
    // must be resident before saves/TR. Cover static fields as well as the
    // register fields below; do not read unused fields from devices.
    let bytes = if tasks::wide(tss) { 0x68 } else { 0x2c };
    for offset in (0..bytes).step_by(8) {
        plan(
            cpu,
            &mem.system(),
            u64::from(tss.cache.base.wrapping_add(offset)),
            (bytes - offset).min(8) as u8,
            AccessKind::Read,
            false,
        )?;
    }
    // Plans are keyed by exact span. Retain each image read's translation
    // across outgoing saves that can overwrite the incoming TSS's PTE.
    let mut field = |offset, size| {
        plan(
            cpu,
            &mem.system(),
            u64::from(tss.cache.base.wrapping_add(offset)),
            size,
            AccessKind::Read,
            false,
        )
        .map(|_| ())
    };
    if tasks::wide(tss) {
        // 386 PRM §7.1 Fig. 7-1 pp. 7-2–3: CR3, EIP, EFLAGS,
        // eight GPRs, six selectors, LDTR, T and I/O-map base.
        for offset in (0x1c..0x48).step_by(4) {
            field(offset, 4)?;
        }
        for offset in (0x48..=0x60).step_by(4).chain([0x64, 0x66]) {
            field(offset, 2)?;
        }
    } else {
        // 286 PRM §8.2 Fig. 8-1 pp. 8-2–3: IP through LDTR.
        for offset in (0x0e..0x2c).step_by(2) {
            field(offset, 2)?;
        }
    }
    Ok(())
}

#[cold]
#[inline(never)]
pub(super) fn task_image<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &impl LegacyMemory<ROUTED, ORDER>,
    tss: legacy::SystemSegment,
) -> Result<(tasks::Image, Option<u32>), Exit> {
    task_resident(cpu, mem, tss)?;
    // 80386 PRM §7.1 Fig. 7-1 pp. 7-2–3; §7.5 p. 7-8:
    // CR3 is a static incoming field, read through the outgoing address space.
    let cr3 = if tasks::wide(tss) {
        Some(read(
            cpu,
            &mem.system(),
            u64::from(tss.cache.base.wrapping_add(0x1c)),
            4,
            AccessKind::Read,
        )? as u32)
    } else {
        None
    };
    Ok((tasks::load_image(cpu, mem, tss)?, cr3))
}

#[cold]
#[inline(never)]
pub(crate) fn execute_retained<const ROUTED: bool, const ORDER: u8, const V86: bool>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
    retry: &mut Option<Transaction>,
    tlb: &std::cell::RefCell<Tlb>,
) -> Result<(Outcome, Option<PortPlan>), Exit> {
    // Two shapes, told apart by one op each: a privilege check first
    // (`lift::legacy` inserts it after `InsnStart`), or an IRET last. A V86
    // IRET has both: refused below IOPL 3, otherwise an IRET.
    let iret = matches!(ir.ops.last(), Some(IrOp::Legacy(L::Iret { .. })));
    let lane = if matches!(ir.ops.get(1), Some(IrOp::Legacy(L::Privileged { .. }))) {
        execute_denied_fast::<ROUTED, ORDER, V86>(ir, cpu, memory, context, task_switch, retry, tlb)
    } else {
        None
    };
    let lane = match lane {
        None if iret => execute_iret_fast::<ROUTED, ORDER, V86>(
            ir,
            cpu,
            memory,
            context,
            task_switch,
            retry,
            tlb,
        ),
        lane => lane,
    };
    if let Some(result) = lane {
        return result.map(|outcome| (outcome, None));
    }
    let mut scratch = vec![0; ir.temp_count as usize];
    let mut mem = LegacyAccess::<ROUTED, ORDER, true>::new(memory, u32::MAX);
    mem.paging.tlb = Some(tlb);
    let view = InstructionView {
        guest_start: ir.guest_start,
        guest_len: ir.guest_len,
        ops: &ir.ops,
        proven: false,
    };
    execute_with_retry::<ROUTED, ORDER, true, V86>(
        ir,
        &view,
        cpu,
        &mem,
        &mut scratch,
        context,
        task_switch,
        retry,
    )
}

/// Effect proof for the RAM-free lane (decision-27 interpreter checkpoint).
/// Every accepted operation is register/flags computation or near control flow:
/// no operand memory or handoff. Segmented near targets may raise a segment fault;
/// the instruction walk retains their checks and rollback before retirement.
/// The caller excludes the 486 alignment-refusal regime before this lane.
/// New IR operations default to the existing transaction path.
fn register_effects(ops: &[IrOp]) -> bool {
    ops.iter().all(register_effect)
}

#[inline(always)]
fn register_effect(op: &IrOp) -> bool {
    matches!(
        op,
        IrOp::InsnStart { .. }
            | IrOp::ReadReg { .. }
            | IrOp::WriteReg { .. }
            | IrOp::ReadSelector { .. }
            | IrOp::Add { .. }
            | IrOp::Adc { .. }
            | IrOp::Sub { .. }
            | IrOp::Sbb { .. }
            | IrOp::And { .. }
            | IrOp::Or { .. }
            | IrOp::Xor { .. }
            | IrOp::Shl { .. }
            | IrOp::Shr { .. }
            | IrOp::Sar { .. }
            | IrOp::Rol { .. }
            | IrOp::Ror { .. }
            | IrOp::Rcl { .. }
            | IrOp::Rcr { .. }
            | IrOp::DoubleShift { .. }
            | IrOp::Sext { .. }
            | IrOp::Mul { .. }
            | IrOp::GetCond { .. }
            | IrOp::Jump { .. }
            | IrOp::Branch { .. }
            | IrOp::BitScan { .. }
            | IrOp::Bt { .. }
            | IrOp::Bswap { .. }
            | IrOp::SetCf { .. }
            | IrOp::SetDf { .. }
            | IrOp::Lahf
            | IrOp::Sahf
            | IrOp::Legacy(
                L::Checkpoint | L::ReadSelector { .. } | L::I486Gate | L::LiveField { .. }
            )
    )
}

/// The register lane: `register_effects` plus the CS check of a near JMP
/// target (`check_target`). That check reads only the retained CS cache and
/// a temporary, and raises #GP(0) for a target beyond the CS limit before
/// the `Jump` changes EIP (SDM Vol 2A JMP, Operation p. 3-507 and Protected
/// Mode Exceptions p. 3-510; 80386 PRM §17 JMP pp. 17-83, 17-87). No
/// instruction of this lane accesses memory, so a fault leaves RAM as it was;
/// a register write before a faulting op carries the lift's checkpoint, which
/// `execute_instruction` restores, as for the Jcc and JMP target checks of
/// `Branch` and `Jump`. The check stays out of `register_effects`, whose
/// other caller, `classify`, admits a CS check only after a load and
/// never after a store.
fn register_lane(ops: &[IrOp]) -> bool {
    ops.iter().all(|op| {
        register_effect(op)
            || matches!(
                op,
                IrOp::Legacy(
                    L::Flat32Address {
                        segment: SegmentId::Cs,
                        bytes: 1,
                        access: AccessKind::Execute,
                        ..
                    } | L::SegmentedAddress {
                        segment: SegmentId::Cs,
                        bytes: 1,
                        access: AccessKind::Execute,
                        ..
                    } | L::Address {
                        segment: SegmentId::Cs,
                        bytes: 1,
                        access: AccessKind::Execute,
                        ..
                    }
                )
            )
    })
}

// Paging design §§3.2–3.6: use only an existing translation; a miss must not
// walk, update A/D, consume a device answer or change fault precedence.
fn fast_operand(
    cpu: &CpuState,
    memory: &Memory,
    tlb: &Tlb,
    linear: u64,
    size: u8,
    access: AccessKind,
) -> Option<FastOperand> {
    let privilege = if cpu.legacy.cpl == 3 {
        Privilege::User
    } else {
        Privilege::Supervisor
    };
    let window = fast_window(cpu, memory, tlb, linear, size, access, privilege)?;
    Some(FastOperand::single(linear, window.bus, size, access, cpu))
}

fn fast_window(
    cpu: &CpuState,
    memory: &Memory,
    tlb: &Tlb,
    linear: u64,
    size: u8,
    access: AccessKind,
    privilege: Privilege,
) -> Option<Window> {
    if size == 0
        || size > 8
        || linear > u64::from(u32::MAX)
        || u64::from(size) > 0x1000 - (linear & 0xfff)
    {
        return None;
    }
    let entry = tlb.entries[Tlb::index(linear as u32)]
        .filter(|entry| u64::from(entry.page) == linear >> 12)?;
    // 80386 PRM §5.2.4.3; SDM Vol 3A §5.8: a cached translation needs
    // no A update, but the first write to a clean fill must set D slowly.
    if !permitted(cpu, access, privilege, entry.rights)
        || (access == AccessKind::Write && entry.translation.pte & DIRTY == 0)
    {
        return None;
    }
    let physical = (entry.translation.physical & FRAME) | (linear as u32 & 0xfff);
    let (bus, part) = legacy::physical_span(cpu, u32::MAX, u64::from(physical), size);
    // A store into a code page takes this window too (xj-to9.48): it is
    // performed and noted through `Memory::write`, as the transaction's
    // commit performs it (`commit_ram`), and its notice ends the block at the
    // instruction's boundary (`BlockPass::commit`).
    if part != size || memory.region_kind(bus, size) != Ok(crate::RegionKind::Ram) {
        return None;
    }
    Some(Window {
        linear,
        bus,
        size,
        write: access == AccessKind::Write,
        supervisor: privilege == Privilege::Supervisor,
    })
}

/// The descriptor window of a protected segment or CS load: the eight bytes
/// `descriptor_address` names, read with supervisor privilege (80386 PRM
/// §6.5 p. 6-22; SDM Vol 3A §5.6.1). None if no descriptor is read (a null
/// selector, a V86 paragraph, or a selector beyond its table: #GP first).
fn descriptor_window(
    cpu: &CpuState,
    memory: &Memory,
    tlb: &Tlb,
    selector: u16,
    context: MachineContext,
) -> Result<Option<Window>, ()> {
    if context.regime == legacy::MachineRegime::Virtual8086 || selector & !3 == 0 {
        return Ok(None);
    }
    let Some(address) = super::descriptor_address(cpu, selector, context.generation) else {
        return Ok(None);
    };
    fast_window(
        cpu,
        memory,
        tlb,
        u64::from(address),
        8,
        AccessKind::Read,
        Privilege::Supervisor,
    )
    .map(Some)
    .ok_or(())
}

/// A far CALL or JMP with an immediate pointer, or a far RET, that stays at
/// the current privilege level: prove its descriptor and its stack frame as
/// RAM windows from existing TLB entries. A gate, TSS or outward return, an
/// unproved access or a wrapping frame keeps the transaction path.
fn fast_far(
    ops: &[IrOp],
    cpu: &CpuState,
    memory: &Memory,
    tlb: &Tlb,
    context: MachineContext,
) -> Option<FastOperand> {
    let mut far = None;
    for op in ops {
        match op {
            IrOp::InsnStart { .. } | IrOp::Legacy(L::Checkpoint | L::I486Gate) => {}
            IrOp::Legacy(
                op @ (L::Transfer {
                    cs: Some(Val::Imm(_)),
                    ip: Val::Imm(_),
                    ..
                }
                | L::Return { far: true, .. }),
            ) if far.is_none() => far = Some(op),
            _ => return None,
        }
    }
    let v86 = context.regime == legacy::MachineRegime::Virtual8086;
    let user = if cpu.legacy.cpl == 3 {
        Privilege::User
    } else {
        Privilege::Supervisor
    };
    let sp = cpu.gpr[RSP] & mask(context.stack_address_bits / 8);
    let stack = |offset: u64, bytes: u8, access: AccessKind| {
        let linear = if v86 {
            legacy::virtual8086_access(cpu, SegmentId::Ss, offset, bytes, access)
        } else {
            legacy::access(
                cpu,
                context.generation,
                SegmentId::Ss,
                offset,
                bytes,
                access,
            )
        }
        .ok()?;
        fast_window(cpu, memory, tlb, linear, bytes, access, user)
    };
    // A code-segment descriptor: no call gate, TSS or task gate (SDM Vol 3A
    // §6.8 p. 6-10 lists the four targets of a far CALL or JMP).
    let code = |window: Option<Window>| {
        window.is_none_or(|w| {
            memory
                .read(w.bus + 5, 1)
                .is_ok_and(|attributes| attributes & 0x18 == 0x18)
        })
    };
    let (stack, descriptor) = match far? {
        L::Transfer {
            cs: Some(Val::Imm(cs)),
            size,
            return_ip,
            ..
        } => {
            let frame = 2 * *size;
            let stack = if return_ip.is_some() {
                // The pushed CS and IP slots do not wrap the stack pointer.
                Some(stack(
                    sp.checked_sub(u64::from(frame))?,
                    frame,
                    AccessKind::Write,
                )?)
            } else {
                None
            };
            (
                stack,
                descriptor_window(cpu, memory, tlb, *cs as u16, context).ok()?,
            )
        }
        L::Return { size, .. } => {
            let frame = 2 * *size;
            let stack = stack(sp, frame, AccessKind::Read)?;
            let cs = memory.read(stack.bus + u64::from(*size), 2).ok()? as u16;
            // An outward return loads SS:SP too (`privilege::outward`).
            if !v86 && (cs & 3) as u8 != cpu.legacy.cpl {
                return None;
            }
            (
                Some(stack),
                descriptor_window(cpu, memory, tlb, cs, context).ok()?,
            )
        }
        _ => unreachable!(),
    };
    if !code(descriptor) {
        return None;
    }
    Some(FastOperand {
        windows: [stack, descriptor],
        strict: true,
        tlb: false,
    })
}

/// Paged far transfers at the current privilege level, without the operand
/// transaction when `fast_far` proves every access (spec.md §17.10). Faults
/// are those of the same helper and order; an access outside the windows
/// restores the instruction's inputs and returns None for the transaction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_far_fast<const ROUTED: bool, const ORDER: u8, const V86: bool>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
    retry: &Option<Transaction>,
    tlb: &std::cell::RefCell<Tlb>,
) -> Option<Result<Outcome, Exit>> {
    if retry.is_some()
        || task_switch.context().is_some()
        || memory.host_fault_recovery()
        || cpu.mmio_retry_active
        || (cpu.legacy.cpl == 3
            && cpu.legacy.cr0 & (1 << 18) != 0
            && cpu.rflags_system & (1 << 18) != 0)
    {
        return None;
    }
    let windows = fast_far(&ir.ops, cpu, memory, &tlb.borrow(), context)?;
    let mut temps = [0u64; 16];
    let temps = temps.get_mut(..ir.temp_count as usize)?;
    let mut mem = LegacyAccess::<ROUTED, ORDER, true>::new(memory, u32::MAX);
    mem.paging.tlb = Some(tlb);
    mem.paging.fast_operand = Some(windows);
    let view = InstructionView {
        guest_start: ir.guest_start,
        guest_len: ir.guest_len,
        ops: &ir.ops,
        proven: false,
    };
    strict_lane(cpu, &mem, task_switch, |cpu, task_switch| {
        execute_instruction::<ROUTED, ORDER, true, V86>(
            ir,
            &view,
            cpu,
            &mem,
            temps,
            context,
            task_switch,
        )
    })
}

/// Whether a lane may skip the two-phase transaction at all: no retained
/// bus handoff or task continuation, no host fault recovery, and not the
/// 486 alignment-check regime the ordinary lanes also exclude.
fn tlb_lane_admits(
    cpu: &CpuState,
    memory: &Memory,
    task_switch: &tasks::SwitchState,
    retry: &Option<Transaction>,
) -> bool {
    !fast_lanes_off()
        && retry.is_none()
        && task_switch.context().is_none()
        && !memory.host_fault_recovery()
        && !cpu.mmio_retry_active
        && !(cpu.legacy.cpl == 3
            && cpu.legacy.cr0 & (1 << 18) != 0
            && cpu.rflags_system & (1 << 18) != 0)
}

/// Run `run` once, without the two-phase transaction, with every access
/// proved when it is made (`FastOperand::each_access`, `lane_window`). Its
/// callers run only helpers whose reads and probes all precede their first
/// memory write, and whose only other write, a descriptor's accessed bit,
/// misses before it is made (`probe_descriptor_cas`); so a miss leaves
/// memory as it was, but for the walks' A bits, which the transaction
/// path's preview sets in the same order. After a miss, and after an exit
/// unless `keep_exits`, the lane restores what the helpers change
/// (`strict_lane`) and returns None: the transaction path then produces the
/// exact result, faults, CR2 and replay state included. A lane that succeeds
/// made the transfers of that path's committing phase, which needs no
/// further walk, D update or device answer when every access hits RAM with
/// D already set (paging design §§3.2–3.4).
fn tlb_lane<const ROUTED: bool, const ORDER: u8, T>(
    cpu: &mut CpuState,
    memory: &Memory,
    tlb: &std::cell::RefCell<Tlb>,
    task_switch: &mut tasks::SwitchState,
    keep_exits: bool,
    run: impl FnOnce(
        &mut CpuState,
        &LegacyAccess<'_, ROUTED, ORDER, true>,
        &mut tasks::SwitchState,
    ) -> Result<T, Exit>,
) -> Option<Result<T, Exit>> {
    tlb.borrow_mut().begin_lane(cpu);
    let mut mem = LegacyAccess::<ROUTED, ORDER, true>::new(memory, u32::MAX);
    mem.paging.tlb = Some(tlb);
    mem.paging.fast_operand = Some(FastOperand::each_access());
    let saved = (
        cpu.gpr,
        cpu.flags,
        cpu.rflags_system,
        cpu.legacy,
        [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs],
        cpu.rip,
    );
    let saved_tasks = task_switch.clone();
    let result = run(cpu, &mem, task_switch);
    let missed = mem.paging.fast_missed.get();
    if !missed && result.is_ok() {
        return Some(result);
    }
    (
        cpu.gpr,
        cpu.flags,
        cpu.rflags_system,
        cpu.legacy,
        [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs],
        cpu.rip,
    ) = saved;
    *task_switch = saved_tasks;
    cpu.clear_mmio_parts();
    if missed || !keep_exits {
        tlb.borrow_mut().undo_lane();
        return None;
    }
    // The transaction path's preview fails on a copy and leaves the inputs
    // as they were, CR2 apart, and its walks' TLB fills (`execute_with_retry`).
    let exit = result.err()?;
    publish_cr2(cpu, &exit);
    Some(Err(exit))
}

/// A paged delivery through an interrupt or trap gate, once and without the
/// transaction, when `tlb_lane` proves every access: the gate, descriptors
/// and TSS stack fields it reads and the frame it pushes, V86 exits to ring
/// 0 included (`privilege::virtual8086_inward`). Every delivery helper reads
/// and probes its whole frame before the commit point. A task gate (type 5)
/// writes the outgoing TSS before it reads the incoming one, so it keeps the
/// transaction path (80386 PRM §9.6.1 Fig. 9-4 p. 9-9, §7.5 pp. 7-8–9).
fn deliver_fast<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    memory: &Memory,
    context: MachineContext,
    event: legacy::LegacyEvent,
    task_switch: &mut tasks::SwitchState,
    tlb: &std::cell::RefCell<Tlb>,
) -> Option<Result<(), Exit>> {
    if !tlb_lane_admits(cpu, memory, task_switch, &None)
        || context.regime == legacy::MachineRegime::Real
    {
        return None;
    }
    let offset = u32::from(event.vector) * 8;
    if offset + 7 > u32::from(cpu.legacy.idtr.limit) {
        return None;
    }
    let high = cpu.legacy.idtr.base.wrapping_add(offset).wrapping_add(4);
    let result = tlb_lane::<ROUTED, ORDER, ()>(
        cpu,
        memory,
        tlb,
        task_switch,
        false,
        |cpu, mem, task_switch| {
            // The gate type, bits 12:8 of its high doubleword (80386 PRM
            // §9.5 Fig. 9-3 p. 9-7): 6/7 and 14/15 are interrupt and trap
            // gates. The gate's page is the one `deliver` reads it from.
            let image = read(cpu, &mem.system(), u64::from(high), 4, AccessKind::Read)?;
            if !matches!((image >> 8) & 0x17, 6 | 7) {
                return Err(fast_miss_exit(cpu));
            }
            deliver(cpu, mem, context, event, task_switch)
        },
    )?;
    count_fast_lane();
    Some(result)
}

/// IRET without a task return, once and without the transaction, when
/// `tlb_lane` proves its frame and descriptor reads. Every IRET path short
/// of a task return reads and writes no memory but descriptor accessed bits:
/// the same-level, outward and V86 returns and the IRETD that enters V86
/// (SDM Vol 2A §3.3 IRET pp. 3-491–497; 386 PRM §17 IRET pp. 17-74–78).
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn execute_iret_fast<const ROUTED: bool, const ORDER: u8, const V86: bool>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
    retry: &Option<Transaction>,
    tlb: &std::cell::RefCell<Tlb>,
) -> Option<Result<Outcome, Exit>> {
    // NT selects the task return in protected mode (`L::CheckTaskReturn`;
    // 386 PRM §17 IRET p. 17-74); V86 IRET ignores NT.
    if !tlb_lane_admits(cpu, memory, task_switch, retry)
        || (!V86 && cpu.rflags_system & 0x4000 != 0)
        || !ir.ops.iter().all(|op| {
            matches!(
                op,
                IrOp::InsnStart { .. }
                    | IrOp::Fence { .. }
                    | IrOp::Legacy(
                        L::Checkpoint
                            | L::I486Gate
                            | L::Privileged { .. }
                            | L::CheckTaskReturn
                            | L::Iret { .. }
                    )
            )
        })
    {
        return None;
    }
    let mut temps = [0u64; 16];
    let temps = temps.get_mut(..ir.temp_count as usize)?;
    let view = InstructionView {
        guest_start: ir.guest_start,
        guest_len: ir.guest_len,
        ops: &ir.ops,
        proven: false,
    };
    let result = tlb_lane::<ROUTED, ORDER, Outcome>(
        cpu,
        memory,
        tlb,
        task_switch,
        false,
        |cpu, mem, task_switch| {
            execute_instruction::<ROUTED, ORDER, true, V86>(
                ir,
                &view,
                cpu,
                mem,
                temps,
                context,
                task_switch,
            )
        },
    )?;
    count_fast_lane();
    Some(result)
}

/// An instruction whose privilege check (`L::Privileged`, its first effect)
/// refuses it: CPL-0 and IOPL-sensitive instructions, and IN, OUT, INS and
/// OUTS that the I/O permission bitmap denies, V86 included. Each is #GP(0)
/// before any operand or port access (spec.md §17.10; SDM Vol 3B §23.2.8.1
/// p. 23-11; 386 PRM §8.3 pp. 8-4–5). The check reads at most the TSS's I/O
/// map base and two bitmap bytes, proved by `tlb_lane`; when it refuses, the
/// instruction runs there to its fault, which is returned with the inputs
/// restored, as the transaction path's preview returns it. A permitted
/// instruction keeps that path, which owns its port handoff.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn execute_denied_fast<const ROUTED: bool, const ORDER: u8, const V86: bool>(
    ir: &IrBlock,
    cpu: &mut CpuState,
    memory: &Memory,
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
    retry: &Option<Transaction>,
    tlb: &std::cell::RefCell<Tlb>,
) -> Option<Result<Outcome, Exit>> {
    let Some(IrOp::Legacy(L::Privileged { kind })) = ir.ops.get(1) else {
        return None;
    };
    if !tlb_lane_admits(cpu, memory, task_switch, retry)
        || cpu.pending_port_in.is_some()
        || cpu.port_string_in.is_some()
    {
        return None;
    }
    let refused =
        tlb_lane::<ROUTED, ORDER, bool>(cpu, memory, tlb, task_switch, false, |cpu, mem, _| {
            match privilege::check::<ROUTED, ORDER, V86>(cpu, mem, context.generation, *kind) {
                Ok(()) => Ok(false),
                Err(Exit::Exception { vector: 13, .. }) => Ok(true),
                Err(exit) => Err(exit),
            }
        })?
        .ok()?;
    if !refused {
        return None;
    }
    let mut temps = [0u64; 16];
    let temps = temps.get_mut(..ir.temp_count as usize)?;
    let view = InstructionView {
        guest_start: ir.guest_start,
        guest_len: ir.guest_len,
        ops: &ir.ops,
        proven: false,
    };
    let result = tlb_lane::<ROUTED, ORDER, Outcome>(
        cpu,
        memory,
        tlb,
        task_switch,
        true,
        |cpu, mem, task_switch| {
            execute_instruction::<ROUTED, ORDER, true, V86>(
                ir,
                &view,
                cpu,
                mem,
                temps,
                context,
                task_switch,
            )
        },
    )?;
    debug_assert!(result.is_err(), "a refused instruction retired");
    count_fast_lane();
    Some(result)
}

/// A strict (system) lane inside an ordinary walk, or its transaction replay.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(super) fn strict_instruction<const ROUTED: bool, const ORDER: u8, const V86: bool>(
    ir: &IrBlock,
    view: &InstructionView<'_>,
    cpu: &mut CpuState,
    mem: &mut LegacyAccess<'_, ROUTED, ORDER, true>,
    temps: &mut [u64],
    context: MachineContext,
    task_switch: &mut tasks::SwitchState,
    retry: &mut Option<Transaction>,
) -> Result<Outcome, Exit> {
    let lane = strict_lane(cpu, mem, task_switch, |cpu, task_switch| {
        execute_instruction::<ROUTED, ORDER, true, V86>(
            ir,
            view,
            cpu,
            mem,
            temps,
            context,
            task_switch,
        )
    });
    if let Some(result) = lane {
        return result;
    }
    mem.paging.fast_operand = None;
    execute_with_retry::<ROUTED, ORDER, true, V86>(
        ir,
        view,
        cpu,
        mem,
        temps,
        context,
        task_switch,
        retry,
    )
    .map(|(outcome, _)| outcome)
}

/// Run a strict lane; after an access outside its windows restore the inputs
/// it may have changed and return None. Every read and probe precedes the
/// frame stores, so no lane writes a window before its last possible miss.
/// One write does precede them: `accessed` commits the descriptor's A bit
/// (and the descriptor page's A/D) through `descriptor_cas`, which is not
/// undone. That is safe only because every access after `accessed` is inside
/// a proved window and cannot miss; a debug build asserts it.
fn strict_lane<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &LegacyAccess<'_, ROUTED, ORDER, true>,
    task_switch: &mut tasks::SwitchState,
    run: impl FnOnce(&mut CpuState, &mut tasks::SwitchState) -> Result<Outcome, Exit>,
) -> Option<Result<Outcome, Exit>> {
    let saved = (
        cpu.gpr,
        cpu.flags,
        cpu.rflags_system,
        cpu.legacy,
        [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs],
        cpu.rip,
    );
    let saved_tasks = task_switch.clone();
    mem.paging.fast_missed.set(false);
    #[cfg(debug_assertions)]
    LANE_COMMITTED.with(|committed| committed.set(false));
    let result = run(cpu, task_switch);
    if !mem.paging.fast_missed.get() {
        return Some(result);
    }
    #[cfg(debug_assertions)]
    debug_assert!(
        !LANE_COMMITTED.with(Cell::get),
        "a strict lane missed after its descriptor CAS committed"
    );
    (
        cpu.gpr,
        cpu.flags,
        cpu.rflags_system,
        cpu.legacy,
        [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs],
        cpu.rip,
    ) = saved;
    *task_switch = saved_tasks;
    cpu.clear_mmio_parts();
    None
}

/// How an instruction of a paged ordinary block takes its lane: the part of
/// `ordinary_lane` that depends only on its ops, decided once per block when
/// its proof is made (`lanes`). A class is a promise about the ops alone; the
/// run-time proof (`ordinary_lane`, and a `Fast` instruction's first access)
/// can still send the instruction to the operand transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    /// `register_lane`: register effects and a near JMP's CS target check.
    Registers,
    /// No proved lane admits these ops: the operand transaction.
    Transaction,
    /// MOV Sreg from a register (`data_segment_load`) after ops that write
    /// only temporaries, then register effects only: the descriptor window
    /// is proved before the instruction runs (`segment_operand`) and the
    /// strict lane runs it.
    Segment,
    /// One RAM operand after ops that write only temporaries, then register
    /// effects only: the walk runs the instruction and proves the operand's
    /// window from the TLB at its first access (`access_operand`). The
    /// operand is a Store's, a Load's, or a read-modify-write's (ADD, INC,
    /// shifts and the like to memory): a Load and one Store to the same
    /// address temporary and width. `write` is the access the window is
    /// proved for, a Store's for a read-modify-write, whose write window also
    /// admits the load before it; both transfers then hit RAM through one
    /// translation, as the transaction's would (SDM Vol 3A §5.6.1; 80386 PRM
    /// §6.4.1 p. 6-20).
    Fast { write: bool, rmw: bool },
}

/// One instruction of a paged ordinary block, at its `InsnStart` index in the
/// block's table (`lanes`): the op index where it ends (the next marker, or
/// the block's end), its guest length and its lane class.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Insn {
    pub(super) end: u32,
    pub(super) len: u32,
    pub(super) class: Class,
}

/// The instruction table of a paged ordinary block: one entry per op,
/// meaningful at each `InsnStart` (`Insn`). Built once per proof
/// (`vm::legacy::paging`) and for a block that runs without one.
pub(crate) fn lanes_into(ir: &IrBlock, insns: &mut Vec<Insn>) {
    let ops = &ir.ops;
    insns.clear();
    insns.resize(
        ops.len(),
        Insn {
            end: 0,
            len: 0,
            class: Class::Transaction,
        },
    );
    let mut at = 0;
    while at < ops.len() {
        let IrOp::InsnStart { guest_addr } = ops[at] else {
            unreachable!("instruction marker")
        };
        let end = ops[at + 1..]
            .iter()
            .position(|op| matches!(op, IrOp::InsnStart { .. }))
            .map_or(ops.len(), |offset| at + 1 + offset);
        let next = match ops.get(end) {
            Some(IrOp::InsnStart { guest_addr }) => *guest_addr,
            _ => ir.guest_start + u64::from(ir.guest_len),
        };
        insns[at] = Insn {
            end: end as u32,
            len: (next - guest_addr) as u32,
            class: classify(&ops[at..end]),
        };
        at = end;
    }
}

/// [`lanes_into`] for a proof to keep.
pub(crate) fn lanes(ir: &IrBlock) -> std::sync::Arc<[Insn]> {
    let mut insns = Vec::new();
    lanes_into(ir, &mut insns);
    insns.into()
}

/// The lane class of one instruction's ops (`Class`).
pub(super) fn classify(ops: &[IrOp]) -> Class {
    if register_lane(ops) {
        return Class::Registers;
    }
    let Some(at) = ops.iter().position(|op| !prepares(op)) else {
        return Class::Transaction;
    };
    let rest = &ops[at + 1..];
    match &ops[at] {
        IrOp::Legacy(op @ L::LoadSegment { .. })
            if super::data_segment_load(op).is_some() && register_effects(rest) =>
        {
            Class::Segment
        }
        IrOp::Store { .. } if register_effects(rest) => Class::Fast {
            write: true,
            rmw: false,
        },
        IrOp::Load {
            dst, addr, size, ..
        } => {
            // A near RET checks its target immediately after loading a 16/32-bit
            // offset. Flat32 CS admits every such offset (decision-27). A Segmented
            // or V86 target check still runs, and can fault, in the execution: it
            // follows only the load, so a fault leaves RAM and registers as
            // they were (SDM Vol 2B RET, Protected Mode Exceptions).
            let checked = *size <= 4
                && rest.first().is_some_and(|op| {
                    matches!(op,
                        IrOp::Legacy(L::Flat32Address { segment: SegmentId::Cs,
                            offset, bytes: 1, access: AccessKind::Execute, .. }
                        | L::SegmentedAddress { segment: SegmentId::Cs,
                            offset, bytes: 1, access: AccessKind::Execute, .. }
                        | L::Address { segment: SegmentId::Cs,
                            offset, bytes: 1, access: AccessKind::Execute, .. })
                            if *offset == Val::Temp(*dst)
                    )
                });
            let rest = &rest[usize::from(checked)..];
            let Some(at) = rest.iter().position(|op| !register_effect(op)) else {
                return Class::Fast {
                    write: false,
                    rmw: false,
                };
            };
            match &rest[at] {
                IrOp::Store {
                    addr: a, size: s, ..
                } if a == addr && s == size && register_effects(&rest[at + 1..]) => Class::Fast {
                    write: true,
                    rmw: true,
                },
                _ => Class::Transaction,
            }
        }
        _ => Class::Transaction,
    }
}

/// The ops a proved lane may run before its operand: they write only
/// temporaries (`prepare`). The alignment-refusal regime is excluded before
/// any proved lane (`ordinary_lane`), so the 486 gate admits.
fn prepares(op: &IrOp) -> bool {
    match op {
        IrOp::InsnStart { .. }
        | IrOp::Legacy(
            L::Checkpoint
            | L::I486Gate
            | L::LiveField { .. }
            | L::SegmentedAddress { .. }
            | L::Flat32Address { .. }
            | L::Address { .. },
        )
        | IrOp::ReadReg { .. } => true,
        IrOp::Add { set_flags, .. }
        | IrOp::Sub { set_flags, .. }
        | IrOp::And { set_flags, .. }
        | IrOp::Shl { set_flags, .. } => *set_flags == FlagMask::NONE,
        _ => false,
    }
}

/// One op `prepares` admits, as the instruction's execution would run it,
/// into temporaries only; None where it would fault.
fn prepare(
    op: &IrOp,
    cpu: &mut CpuState,
    memory: &Memory,
    temps: &mut [u64],
    context: MachineContext,
) -> Option<()> {
    match op {
        IrOp::InsnStart { .. } | IrOp::Legacy(L::Checkpoint | L::I486Gate) => {}
        IrOp::ReadReg { dst, reg } => temps[*dst as usize] = crate::interp::read_reg(cpu, *reg),
        IrOp::Legacy(L::LiveField {
            dst,
            bus,
            bytes,
            extend,
        }) => temps[*dst as usize] = super::live_field(memory, *bus, *bytes, *extend).ok()?,
        IrOp::Add {
            dst,
            a,
            b,
            size,
            set_flags,
        } => {
            crate::interp::integer::exec_add(cpu, temps, dst, a, b, size, set_flags);
        }
        IrOp::Sub {
            dst,
            a,
            b,
            size,
            set_flags,
        } => {
            crate::interp::integer::exec_sub(cpu, temps, dst, a, b, size, set_flags);
        }
        IrOp::And {
            dst,
            a,
            b,
            size,
            set_flags,
        } => {
            crate::interp::integer::exec_and(cpu, temps, dst, a, b, size, set_flags);
        }
        IrOp::Shl {
            dst,
            a,
            b,
            size,
            set_flags,
        } => {
            crate::interp::integer::exec_shl(cpu, temps, dst, a, b, size, set_flags);
        }
        IrOp::Legacy(L::SegmentedAddress {
            dst,
            offset,
            bytes,
            proof,
            ..
        }) => {
            temps[*dst as usize] = proof.address(read_val(*offset, temps), *bytes)?;
        }
        IrOp::Legacy(L::Flat32Address {
            dst,
            segment,
            offset,
            bytes,
            access,
        }) => {
            temps[*dst as usize] =
                flat32_address(cpu, *segment, read_val(*offset, temps), *bytes, *access).ok()?;
        }
        // The live Checked access: Flat32 FS/GS, Segmented operands after an
        // in-block reload of their segment, and V86 (`execute_op`).
        IrOp::Legacy(L::Address {
            dst,
            segment,
            offset,
            bytes,
            access,
        }) => {
            let offset = read_val(*offset, temps);
            temps[*dst as usize] = if context.regime == legacy::MachineRegime::Virtual8086 {
                legacy::virtual8086_access(cpu, *segment, offset, *bytes, *access)
            } else {
                legacy::access(cpu, context.generation, *segment, offset, *bytes, *access)
            }
            .ok()?;
        }
        _ => unreachable!("an op `prepares` refuses: {op:?}"),
    }
    debug_assert!(prepares(op));
    Some(())
}

/// The descriptor window of a `Segment` instruction, proved before it runs:
/// the selector is known before the load (`descriptor_window`). None if an
/// op before the load would fault: the transaction lane runs it then.
fn segment_operand(
    ops: &[IrOp],
    cpu: &mut CpuState,
    memory: &Memory,
    tlb: &Tlb,
    temps: &mut [u64],
    context: MachineContext,
) -> Option<FastOperand> {
    for op in ops {
        if let IrOp::Legacy(L::LoadSegment { value, .. }) = op {
            let selector = read_val(*value, temps) as u16;
            return Some(FastOperand {
                windows: [
                    descriptor_window(cpu, memory, tlb, selector, context).ok()?,
                    None,
                ],
                strict: true,
                tlb: false,
            });
        }
        prepare(op, cpu, memory, temps, context)?;
    }
    unreachable!("a Segment instruction loads a segment")
}

/// The operand window of a `Fast` instruction, proved at its first access
/// (`write` as its class names it). The ops before that access wrote only
/// temporaries, so this is the window the TLB grants before the instruction
/// for the same linear address. None sends the instruction, which has
/// changed nothing else, to the transaction lane.
#[inline(always)]
pub(super) fn access_operand(
    cpu: &CpuState,
    memory: &Memory,
    tlb: &std::cell::RefCell<Tlb>,
    linear: u64,
    size: u8,
    write: bool,
) -> Option<FastOperand> {
    fast_operand(
        cpu,
        memory,
        &tlb.borrow(),
        linear,
        size,
        if write {
            AccessKind::Write
        } else {
            AccessKind::Read
        },
    )
}

/// Paged Flat32, Segmented and V86 blocks execute effect-proved registers and
/// preflighted RAM once; all other instructions retain the full operand
/// transaction ledger. V86 takes the same walk with V86 address checks.
/// `insns` is the block's instruction table (`lanes`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_flat32<const ROUTED: bool, const ORDER: u8>(
    ir: &IrBlock,
    insns: &[Insn],
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
    limit: u64,
    exit_word: &std::sync::atomic::AtomicU64,
    task_switch: &mut tasks::SwitchState,
    retry: &mut Option<Transaction>,
    tlb: &std::cell::RefCell<Tlb>,
) -> Progress {
    if ir
        .legacy_context
        .is_some_and(|c| c.regime == legacy::MachineRegime::Virtual8086)
    {
        return execute_ordinary::<ROUTED, ORDER, true>(
            ir,
            insns,
            cpu,
            memory,
            scratch,
            limit,
            exit_word,
            task_switch,
            retry,
            tlb,
        );
    }
    execute_ordinary::<ROUTED, ORDER, false>(
        ir,
        insns,
        cpu,
        memory,
        scratch,
        limit,
        exit_word,
        task_switch,
        retry,
        tlb,
    )
}

/// The lane of one ordinary paged instruction of class `class`: register
/// effects alone, a RAM operand proved from existing translations at its
/// access (`Lane::Lazy`), a descriptor window proved now for the strict lane,
/// or the full operand transaction.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn ordinary_lane(
    class: Class,
    view: &InstructionView<'_>,
    cpu: &mut CpuState,
    memory: &Memory,
    tlb: &std::cell::RefCell<Tlb>,
    temps: &mut [u64],
    context: MachineContext,
    retry_none: bool,
) -> (super::Lane, Option<FastOperand>) {
    use super::Lane;
    let once = retry_none
        && !(cpu.legacy.cpl == 3
            && cpu.legacy.cr0 & (1 << 18) != 0
            && cpu.rflags_system & (1 << 18) != 0);
    if !once {
        return (Lane::Transaction, None);
    }
    match class {
        Class::Registers => (Lane::Registers, None),
        Class::Transaction => (Lane::Transaction, None),
        _ if memory.host_fault_recovery() || cpu.mmio_retry_active => (Lane::Transaction, None),
        Class::Fast { rmw: true, .. } if fast_lanes_off() => (Lane::Transaction, None),
        Class::Fast { write, .. } => (Lane::Lazy { write }, None),
        Class::Segment => {
            match segment_operand(view.ops, cpu, memory, &tlb.borrow(), temps, context) {
                Some(operand) => (Lane::Strict, Some(operand)),
                None => (Lane::Transaction, None),
            }
        }
    }
}

/// The ordinary paged walk of Flat32, Segmented and V86 blocks, in one pass
/// (`super::BlockPass`): register and preflighted instructions run in the
/// walk, strict and transaction instructions on their own.
/// The table has an entry per op of `ir` (`lanes_into`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn execute_ordinary<const ROUTED: bool, const ORDER: u8, const V86: bool>(
    ir: &IrBlock,
    insns: &[Insn],
    cpu: &mut CpuState,
    memory: &Memory,
    scratch: &mut impl ScratchBuffer,
    limit: u64,
    exit_word: &std::sync::atomic::AtomicU64,
    task_switch: &mut tasks::SwitchState,
    retry: &mut Option<Transaction>,
    tlb: &std::cell::RefCell<Tlb>,
) -> Progress {
    let context = ir.legacy_context.expect("protected block context");
    let temps = scratch.reused(ir.temp_count as usize);
    let mut mem = LegacyAccess::<ROUTED, ORDER, true>::new(memory, u32::MAX);
    mem.paging.tlb = Some(tlb);
    let mut pass = super::BlockPass::<ROUTED, ORDER, true, V86>::new(
        ir,
        insns,
        context,
        memory,
        LegacyAccess::<ROUTED, ORDER>::new(memory, u32::MAX),
        Some(mem),
        task_switch,
        exit_word,
        limit,
    );
    let outcome = pass.run(cpu, temps, retry);
    Progress {
        sti_shadow: false,
        ss_shadow: 0,
        steps: pass.steps,
        retired: pass.retired,
        outcome: Some(outcome),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lane classes of one instruction's ops (`classify`), for the
    /// shapes the lifter gives register, load, store, read-modify-write,
    /// near RET and MOV Sreg instructions.
    #[test]
    fn classify_names_each_lane_shape() {
        use crate::ir::{MemOrder, MemoryAccess, Temp};
        use crate::state::Reg;
        let start = || IrOp::InsnStart { guest_addr: 0 };
        let read = |dst: Temp, reg| IrOp::ReadReg { dst, reg };
        let address = |dst: Temp, offset, segment, bytes, access| {
            IrOp::Legacy(L::Flat32Address {
                dst,
                segment,
                offset,
                bytes,
                access,
            })
        };
        let load = |dst: Temp, addr| IrOp::Load {
            fault: MemoryAccess::new(4, false, false),
            dst,
            addr,
            size: 4,
        };
        let store = |addr, src, size| IrOp::Store {
            fault: MemoryAccess::new(4, false, false),
            addr,
            src,
            size,
            order: MemOrder::None,
        };
        let add = |dst: Temp, a, b, set_flags| IrOp::Add {
            dst,
            a,
            b,
            size: 4,
            set_flags,
        };
        let write = |src| IrOp::WriteReg {
            reg: Reg::Rax,
            src,
            size: 4,
        };
        let (t0, t1, t2) = (Val::Temp(0), Val::Temp(1), Val::Temp(2));
        let ds = |access| address(1, t0, SegmentId::Ds, 4, access);
        // ADD EAX, ECX
        assert_eq!(
            classify(&[
                start(),
                read(0, Reg::Rax),
                read(1, Reg::Rcx),
                add(2, t0, t1, FlagMask::ALL),
                write(t2),
            ]),
            Class::Registers
        );
        // MOV EAX, [ECX]
        let ops = [
            start(),
            read(0, Reg::Rcx),
            ds(AccessKind::Read),
            load(2, t1),
            write(t2),
        ];
        assert_eq!(
            classify(&ops),
            Class::Fast {
                write: false,
                rmw: false
            }
        );
        // MOV [ECX], EAX
        let ops = [
            start(),
            read(0, Reg::Rcx),
            ds(AccessKind::Write),
            read(2, Reg::Rax),
            store(t1, t2, 4),
        ];
        assert_eq!(
            classify(&ops),
            Class::Fast {
                write: true,
                rmw: false
            }
        );
        // ADD [ECX], EAX: one Store to the operand the Load read.
        let rmw = |addr, size| {
            [
                start(),
                read(0, Reg::Rcx),
                ds(AccessKind::Write),
                load(2, t1),
                IrOp::ReadReg {
                    dst: 3,
                    reg: Reg::Rax,
                },
                add(4, t2, Val::Temp(3), FlagMask::ALL),
                store(addr, Val::Temp(4), size),
            ]
        };
        assert_eq!(
            classify(&rmw(t1, 4)),
            Class::Fast {
                write: true,
                rmw: true
            }
        );
        // A Store elsewhere, or of another width, is a second operand.
        assert_eq!(classify(&rmw(t0, 4)), Class::Transaction);
        assert_eq!(classify(&rmw(t1, 2)), Class::Transaction);
        // A flag-setting op before the operand changes more than temporaries.
        assert_eq!(
            classify(&[
                start(),
                read(0, Reg::Rcx),
                add(1, t0, t0, FlagMask::ALL),
                load(2, t1),
                write(t2),
            ]),
            Class::Transaction
        );
        // Two loads.
        assert_eq!(
            classify(&[
                start(),
                read(0, Reg::Rcx),
                load(1, t0),
                load(2, t0),
                write(t2),
            ]),
            Class::Transaction
        );
        // A near RET: the CS check of the loaded target follows the load.
        let ret = |target| {
            [
                start(),
                read(0, Reg::Rsp),
                address(1, t0, SegmentId::Ss, 4, AccessKind::Read),
                load(2, t1),
                address(3, target, SegmentId::Cs, 1, AccessKind::Execute),
                IrOp::Jump { target: t2 },
            ]
        };
        assert_eq!(
            classify(&ret(t2)),
            Class::Fast {
                write: false,
                rmw: false
            }
        );
        // A CS check of another value after the load is not the RET's.
        assert_eq!(classify(&ret(t0)), Class::Transaction);
        // MOV DS, AX, and MOV SS, AX, which ends an ordinary block.
        let mov = |segment, inhibit| {
            [
                start(),
                read(0, Reg::Rax),
                IrOp::Legacy(L::LoadSegment {
                    segment,
                    value: t0,
                    inhibit,
                }),
            ]
        };
        assert_eq!(classify(&mov(SegmentId::Ds, false)), Class::Segment);
        assert_eq!(classify(&mov(SegmentId::Ss, true)), Class::Transaction);
    }

    #[test]
    fn enter_and_leave_keep_blocks_off_the_ordinary_walk() {
        // The `Fast` class has no ENTER/LEAVE lane: their frames are
        // SS-relative, so they must stay out of ordinary blocks.
        for op in [
            L::Enter {
                size: 4,
                alloc: 8,
                level: 0,
            },
            L::Leave { size: 4 },
        ] {
            let ir = IrBlock {
                ops: vec![IrOp::InsnStart { guest_addr: 0 }, IrOp::Legacy(op)],
                guest_start: 0,
                temp_count: 0,
                guest_len: 1,
                icount: 1,
                mode: crate::CpuMode::Compat32,
                legacy_context: None,
                link_exit: false,
            };
            assert!(!super::super::flat32_ordinary(&ir), "{ir:?}");
        }
    }

    #[test]
    fn a_strict_lane_miss_restores_the_instruction_inputs() {
        // A system lane stops at an access outside its proved windows and
        // replays on the transaction path from the original inputs.
        let vm = crate::Vm::new(crate::VmConfig::flat(0x1000));
        let mut cpu = vm.new_vcpu().cpu;
        cpu.rip = 0x10;
        let before = cpu.clone();
        let mut mem = LegacyAccess::<false, 0, true>::new(&vm.mem, u32::MAX);
        mem.paging.fast_operand = Some(FastOperand {
            windows: [None, None],
            strict: true,
            tlb: false,
        });
        let mut tasks = tasks::SwitchState::default();
        let result = strict_lane(&mut cpu, &mem, &mut tasks, |cpu, _| {
            cpu.gpr[0] = 7;
            cpu.ds = 0x20;
            cpu.legacy.segments[1].base = 0x1234;
            cpu.record_mmio_part(cpu.rip, 1, 2);
            cpu.rip = 0x20;
            read(cpu, &mem, 0x100, 2, AccessKind::Read).map(|_| Outcome::default())
        });
        assert!(result.is_none());
        assert_eq!(
            (cpu.gpr, cpu.ds, cpu.legacy, cpu.rip),
            (before.gpr, before.ds, before.legacy, before.rip)
        );
        assert!(!cpu.mmio_retry_active && cpu.mmio_parts_len == 0);
        // A lane without a miss keeps its result.
        let result = strict_lane(&mut cpu, &mem, &mut tasks, |cpu, _| {
            cpu.gpr[0] = 7;
            Ok(Outcome::default())
        });
        assert!(result.is_some_and(|r| r.is_ok()));
        assert_eq!(cpu.gpr[0], 7);
    }

    #[test]
    fn fast_data_requires_dirty_permitted_ram_hits_and_preserves_cached_ad() {
        // 80386 PRM §§5.2.4.3, 5.2.5, 6.4.1–6.4.2; SDM Vol 3A
        // §§5.6.1, 5.8: cached rights/D, supervisor WP and no repeated A update.
        for model in [
            crate::CpuModel::I80386,
            crate::CpuModel::I80486Dx,
            crate::CpuModel::I80486DxCpuid,
        ] {
            for cpl in [0, 3] {
                for wp in [false, true] {
                    for rights in [1, 3, 5, 7] {
                        let mut vm = crate::Vm::new(crate::VmConfig::flat(0x8000));
                        vm.set_cpu_model(model);
                        vm.map(0, 0x6000, crate::Prot::RWX, crate::RegionKind::Ram)
                            .unwrap();
                        vm.map(0x6000, 0x1000, crate::Prot::RW, crate::RegionKind::Trap)
                            .unwrap();
                        vm.map(0x7000, 0x1000, crate::Prot::RX, crate::RegionKind::Rom)
                            .unwrap();
                        vm.write_bytes(0x2000, &0x3007u32.to_le_bytes()).unwrap();
                        vm.write_bytes(0x3004, &(0x4000u32 | rights).to_le_bytes())
                            .unwrap();
                        let mut cpu = vm.new_vcpu().cpu;
                        cpu.legacy.cr0 = 0x80000001 | if wp { 1 << 16 } else { 0 };
                        cpu.legacy.cr3 = 0x2000;
                        cpu.legacy.cpl = cpl;
                        cpu.legacy.a20_enabled = 1;
                        let tlb = std::cell::RefCell::new(Tlb::default());
                        let mut mem = LegacyAccess::<false, 0, true>::new(&vm.mem, u32::MAX);
                        mem.paging.tlb = Some(&tlb);
                        assert!(fast_operand(
                            &cpu,
                            &vm.mem,
                            &tlb.borrow(),
                            0x1000,
                            4,
                            AccessKind::Read
                        )
                        .is_none());
                        // An implicit supervisor read fills even supervisor-only pages.
                        translate_linear(
                            &mut cpu,
                            &mem,
                            0x1000,
                            4,
                            AccessKind::Read,
                            Privilege::Supervisor,
                            true,
                        )
                        .unwrap();
                        let user_allowed = cpl == 0 || rights & 4 != 0;
                        assert_eq!(
                            fast_operand(&cpu, &vm.mem, &tlb.borrow(), 0x1000, 4, AccessKind::Read)
                                .is_some(),
                            user_allowed
                        );
                        assert!(
                            fast_operand(
                                &cpu,
                                &vm.mem,
                                &tlb.borrow(),
                                0x1000,
                                4,
                                AccessKind::Write
                            )
                            .is_none(),
                            "clean read fill"
                        );
                        let write_allowed = user_allowed
                            && (rights & 2 != 0
                                || (cpl == 0 && (model == crate::CpuModel::I80386 || !wp)));
                        let result = write(&mut cpu, &mem, 0x1000, 4, 41);
                        assert_eq!(result.is_ok(), write_allowed);
                        if !write_allowed {
                            continue;
                        }
                        // Clear stored A/D without invalidating the dirty cached entry.
                        vm.write_bytes(0x2000, &0x3007u32.to_le_bytes()).unwrap();
                        vm.write_bytes(0x3004, &(0x4000u32 | rights).to_le_bytes())
                            .unwrap();
                        mem.paging.fast_operand = fast_operand(
                            &cpu,
                            &vm.mem,
                            &tlb.borrow(),
                            0x1000,
                            4,
                            AccessKind::Write,
                        );
                        assert!(mem.paging.fast_operand.is_some());
                        write(&mut cpu, &mem, 0x1000, 4, 42).unwrap();
                        assert_eq!(
                            read(&mut cpu, &mem, 0x1000, 4, AccessKind::Read).unwrap(),
                            42
                        );
                        let mut bytes = [0; 4];
                        for at in [0x2000, 0x3004] {
                            vm.read_bytes(at, &mut bytes).unwrap();
                            assert_eq!(u32::from_le_bytes(bytes) & 0x60, 0);
                        }
                        mem.paging.fast_operand = None;
                        assert!(fast_operand(
                            &cpu,
                            &vm.mem,
                            &tlb.borrow(),
                            0x1fff,
                            4,
                            AccessKind::Read
                        )
                        .is_none());
                        // A store into a code page takes the window too, and
                        // its write is noted as the transaction's commit notes
                        // it (xj-to9.48).
                        vm.mem.mark_code(0x4000, 1);
                        mem.paging.fast_operand = fast_operand(
                            &cpu,
                            &vm.mem,
                            &tlb.borrow(),
                            0x1000,
                            4,
                            AccessKind::Write,
                        );
                        assert!(mem.paging.fast_operand.is_some());
                        assert!(!vm.mem.has_dirty_code());
                        write(&mut cpu, &mem, 0x1000, 4, 43).unwrap();
                        assert!(vm.mem.has_dirty_code());
                        mem.paging.fast_operand = None;
                        assert!(fast_operand(
                            &cpu,
                            &vm.mem,
                            &tlb.borrow(),
                            0x1000,
                            4,
                            AccessKind::Read
                        )
                        .is_some());
                        for physical in [0x6000, 0x7000, 0x8000] {
                            tlb.borrow_mut().entries[1]
                                .as_mut()
                                .unwrap()
                                .translation
                                .physical = physical;
                            for access in [AccessKind::Read, AccessKind::Write] {
                                assert!(fast_operand(
                                    &cpu,
                                    &vm.mem,
                                    &tlb.borrow(),
                                    0x1000,
                                    4,
                                    access
                                )
                                .is_none());
                            }
                        }
                        tlb.borrow_mut().invalidate();
                        assert!(fast_operand(
                            &cpu,
                            &vm.mem,
                            &tlb.borrow(),
                            0x1000,
                            4,
                            AccessKind::Read
                        )
                        .is_none());
                    }
                }
            }
        }
    }

    #[test]
    fn adjacent_pages_never_share_a_tlb_slot() {
        // A block straddling two pages keeps both fetch entries (xj-to9.70),
        // and every page-number bit reaches the slot: pages that differ in
        // one bit never share it.
        for page in 0..0xf_ffffu32 {
            assert_ne!(
                Tlb::index(page << 12),
                Tlb::index((page + 1) << 12),
                "{page:#x}"
            );
        }
        for bit in 0..20 {
            for page in (0..0x10_0000u32).step_by(0x101) {
                assert_ne!(
                    Tlb::index(page << 12),
                    Tlb::index((page ^ (1 << bit)) << 12),
                    "{page:#x} bit {bit}"
                );
            }
        }
    }

    #[test]
    fn only_fetched_entries_expire_source_proofs_on_replacement() {
        // Paging design §3.2: replacing a data or stack translation leaves
        // every fetch translation, and so every retained source proof, intact.
        // Pages 6 and 104h share a set (`Tlb::index`); 104h maps frame 47h.
        assert_eq!(Tlb::index(0x6000), Tlb::index(0x104000));
        let mut vm = crate::Vm::new(crate::VmConfig::flat(0x50000));
        vm.set_cpu_model(crate::CpuModel::I80386);
        vm.map(0, 0x50000, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        vm.write_bytes(0x1000, &0x2007u32.to_le_bytes()).unwrap();
        for page in 0..0x50u32 {
            vm.write_bytes(
                0x2000 + u64::from(page * 4),
                &(page << 12 | 7).to_le_bytes(),
            )
            .unwrap();
        }
        vm.write_bytes(0x2000 + 0x104 * 4, &0x47007u32.to_le_bytes())
            .unwrap();
        let mut cpu = vm.new_vcpu().cpu;
        cpu.legacy.cr0 = 0x80000001;
        cpu.legacy.cr3 = 0x1000;
        let tlb = std::cell::RefCell::new(Tlb::default());
        let mut mem = LegacyAccess::<false, 0, true>::new(&vm.mem, u32::MAX);
        mem.paging.tlb = Some(&tlb);
        let mut translate = |linear, access| {
            translate_linear(
                &mut cpu,
                &mem,
                linear,
                1,
                access,
                Privilege::Supervisor,
                true,
            )
            .unwrap()
        };
        translate(0x6000, AccessKind::Write);
        let token = tlb.borrow().generation;
        for _ in 0..3 {
            translate(0x104000, AccessKind::Read);
            translate(0x6000, AccessKind::Write);
        }
        assert_eq!(tlb.borrow().generation, token);
        // A fetch through the set makes its replacement expire the token,
        // also when the fetch only hit an entry a data access filled.
        for fill in [AccessKind::Execute, AccessKind::Read] {
            let phys = translate(0x104000, fill).physical;
            if fill == AccessKind::Read {
                assert_eq!(translate(0x104000, AccessKind::Execute).physical, phys);
            }
            let token = tlb.borrow().generation;
            translate(0x6000, AccessKind::Read);
            assert_ne!(tlb.borrow().generation, token);
            let token = tlb.borrow().generation;
            translate(0x104000, AccessKind::Read);
            assert_eq!(tlb.borrow().generation, token);
        }
    }

    #[test]
    fn tlb_keeps_256_distinct_pages_and_the_257th_replaces_one() {
        // Decision-32 Q3; SDM Vol 3A §5.10.2.2 p. 5-43 fixes no TLB size.
        // Linear pages 100h-1FFh fold to the 256 distinct sets, all mapping
        // frame 3; the PTEs then move to frame 4 without a flush, so a hit
        // still returns frame 3 and only a walk sees frame 4. Page 200h
        // shares page 106h's set.
        assert_eq!(Tlb::index(0x200000), Tlb::index(0x106000));
        let mut vm = crate::Vm::new(crate::VmConfig::flat(0x5000));
        vm.set_cpu_model(crate::CpuModel::I80386);
        vm.map(0, 0x5000, crate::Prot::RWX, crate::RegionKind::Ram)
            .unwrap();
        vm.write_bytes(0x1000, &0x2007u32.to_le_bytes()).unwrap();
        let set_frames = |vm: &crate::Vm, frame: u32| {
            for page in 0x100..=0x200u64 {
                vm.write_bytes(0x2000 + page * 4, &(frame << 12 | 7).to_le_bytes())
                    .unwrap();
            }
        };
        set_frames(&vm, 3);
        let mut cpu = vm.new_vcpu().cpu;
        cpu.legacy.cr0 = 0x80000001;
        cpu.legacy.cr3 = 0x1000;
        let tlb = std::cell::RefCell::new(Tlb::default());
        let mut mem = LegacyAccess::<false, 0, true>::new(&vm.mem, u32::MAX);
        mem.paging.tlb = Some(&tlb);
        let mut frame = |page: u32| {
            translate_linear(
                &mut cpu,
                &mem,
                page << 12,
                1,
                AccessKind::Read,
                Privilege::Supervisor,
                true,
            )
            .unwrap()
            .physical
                >> 12
        };
        for page in 0x100..0x200 {
            assert_eq!(frame(page), 3);
        }
        let resident = |tlb: &Tlb| tlb.entries.iter().flatten().count();
        assert_eq!(resident(&tlb.borrow()), TLB_ENTRIES);
        set_frames(&vm, 4);
        for page in 0x100..0x200 {
            assert_eq!(frame(page), 3, "page {page:#x} was walked again");
        }
        assert_eq!(frame(0x200), 4);
        assert_eq!(resident(&tlb.borrow()), TLB_ENTRIES);
        for page in 0x100..0x200 {
            assert_eq!(frame(page), if page == 0x106 { 4 } else { 3 });
        }
    }

    #[test]
    fn user_access_rechecks_supervisor_tlb_fills_and_model_fault_ad() {
        // SDM Vol 3A §5.6.1; paging design §3.3: implicit supervisor
        // translations never grant user access, including on a warm hit.
        for model in [crate::CpuModel::I80386, crate::CpuModel::I80486Dx] {
            for access in [AccessKind::Read, AccessKind::Write, AccessKind::Execute] {
                for warm in [false, true] {
                    let mut vm = crate::Vm::new(crate::VmConfig::flat(0x8000));
                    vm.set_cpu_model(model);
                    vm.map(0, 0x8000, crate::Prot::RWX, crate::RegionKind::Ram)
                        .unwrap();
                    vm.write_bytes(0x2000, &0x3007u32.to_le_bytes()).unwrap();
                    vm.write_bytes(0x3004, &0x4003u32.to_le_bytes()).unwrap();
                    let mut cpu = vm.new_vcpu().cpu;
                    cpu.legacy.cr0 = 0x80000001;
                    cpu.legacy.cr3 = 0x2000;
                    cpu.legacy.cpl = 3;
                    let tlb = std::cell::RefCell::new(Tlb::default());
                    let mut mem = LegacyAccess::<false, 0, true>::new(&vm.mem, u32::MAX);
                    mem.paging.tlb = Some(&tlb);
                    if warm {
                        // A supervisor fetch fill, which can back a source proof.
                        assert!(translate_linear(
                            &mut cpu,
                            &mem,
                            0x1000,
                            1,
                            AccessKind::Execute,
                            Privilege::Supervisor,
                            true
                        )
                        .is_ok());
                        assert!(tlb.borrow().entries[1].is_some_and(|e| e.fetched));
                        // Leave A clear in RAM while retaining the supervisor fill.
                        // A 486 denial must rewalk to perform its A updates.
                        vm.write_bytes(0x2000, &0x3007u32.to_le_bytes()).unwrap();
                        vm.write_bytes(0x3004, &0x4003u32.to_le_bytes()).unwrap();
                    }
                    let generation = tlb.borrow().generation;
                    let exit = match translate_linear(
                        &mut cpu,
                        &mem,
                        0x1000,
                        1,
                        access,
                        Privilege::User,
                        true,
                    ) {
                        Err(exit) => exit,
                        Ok(_) => panic!("supervisor fill authorized a user access"),
                    };
                    if warm {
                        assert!(tlb.borrow().entries[1].is_none());
                        assert_ne!(tlb.borrow().generation, generation);
                    }
                    assert_eq!(
                        exit.fault_metadata().unwrap().error_code(),
                        ErrorCode::bits(if access == AccessKind::Write { 7 } else { 5 }, 7)
                    );
                    let mut bytes = [0; 4];
                    let ad = if model == crate::CpuModel::I80386 {
                        0
                    } else {
                        ACCESSED
                    };
                    for addr in [0x2000, 0x3004] {
                        vm.read_bytes(addr, &mut bytes).unwrap();
                        assert_eq!(u32::from_le_bytes(bytes) & (ACCESSED | DIRTY), ad);
                    }
                }
            }
        }
    }
}
