//! Live instruction fields for Machine proofs (xj-to9.46).
//!
//! A guest that patches the immediate or displacement of an instruction in
//! its running code page would otherwise drop and relift every private proof
//! (`flat32`, `paging`) on each patch. Here the engine counts small code
//! writes by byte range. When a range has been written `THRESHOLD` times and
//! lies inside the immediate or displacement of an instruction a private
//! proof lift decodes, that lift lowers the field as `LegacyOp::LiveField`,
//! read from the bus address the fetch translated it to each time the
//! instruction executes, and records the field as live. A later store lying
//! entirely inside a live field is then absorbed: it advances the shared
//! publication generation (cancelling in-flight shared lifts) but not the
//! proof generation (`TranslationCache::proof_generation`), so no proof map
//! drops a proof and no successor link (`links.rs`) ends.
//!
//! Invariants:
//! * Every private proof whose source bytes include a live field's bytes
//!   reads that field live. A promotion is a recorded advance over its
//!   instruction's bytes, which drops every proof over them and fails every
//!   racing lift's registration (`LiveFields::commit`). A lift that meets a
//!   live field it cannot read live (another decoding of those bytes, or an
//!   ineligible form) demotes the field and lifts again, and one that cannot
//!   read a patched field live marks its count refused, so none promotes it.
//! * A store is absorbed only when no shared translation or pending shared
//!   source covers its bytes (`TranslationCache::absorb_written`): shared
//!   units, compiled code and the Real16 Direct path never read fields live.
//! * A store that touches any other byte of a live field's instruction, or
//!   crosses the field, is not absorbed and invalidates as before; the field
//!   returns to normal and needs `THRESHOLD` further patches to become live.
//!   Live proofs stay exact after a demotion, so a demotion drops nothing.
//! * A store made during a Machine run (`StoreSite`) that would be absorbed
//!   is absorbed where it is made (`absorb_at_store`, xj-to9.48), under the
//!   same locks and with the same effect, so it queues no notice for any
//!   vCPU's dispatcher and needs no drain. A store into a page with a
//!   declared code alias, or made outside such a run, is queued and drained
//!   as before.
//!
//! A live field is read once per execution of its instruction, before that
//! instruction's own stores (the eligible MOV, LEA and ALU forms read their
//! operands first), so it sees the bytes a relift after every earlier store
//! would decode. A store by an earlier instruction of the same block still
//! ends an unpaged walk (SDM Vol 3A §14.6 p. 14-18; `RunningMemory::overlap`);
//! a paged walk ends at the notice of a queued store, and goes on after one
//! absorbed at its store site, whose field the later instruction reads. A
//! store by another vCPU is seen at the patched instruction's next execution
//! rather than at this vCPU's next block boundary. SDM Vol 3A §11.1.3
//! pp. 11-5–11-6 makes cross-modifying code without its synchronization
//! algorithm model-specific; with it, both paths execute the new bytes.
use crate::cache::TranslationCache;
use crate::legacy::MachineContext;
use crate::lift::LiftError;
use crate::IrBlock;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Writes of one byte range before a lift reads a field holding it live.
const THRESHOLD: u32 = 2;
/// A counted write is at most a dword: the widest immediate or displacement
/// field (80386 PRM §17.2 Fig. 17-1 and text, pp. 17-2–17-3).
const MAX_WRITE: u64 = 4;
/// Retention bounds. Clearing counts only delays a promotion; a live table
/// full of fields whose code some proof still holds promotes nothing more,
/// so those fields keep the ordinary relift.
const MAX_WRITES: usize = 1024;
const MAX_LIVE: usize = 256;
/// A write count whose field could not be read live; it is not retried.
const REFUSED: u32 = u32::MAX;

#[derive(Default)]
pub(crate) struct LiveFields {
    table: Mutex<Table>,
    /// Set by the first private proof lift: until then no write is counted
    /// (Real16, twenty-bit and Application code never lift such proofs).
    armed: AtomicBool,
    /// Set by the first counted write; until then lifts take the plain path.
    active: AtomicBool,
    /// Whether a field is live: only then can a store be absorbed. Kept with
    /// the memory's `Memory::live_store_site` under the table lock
    /// (`LiveFields::sync`), and cleared when the last field leaves.
    promoted: AtomicBool,
    /// The reference for the witnesses: every write relifts, as before.
    #[cfg(test)]
    disabled: AtomicBool,
    /// The reference for the store-site witnesses: absorbed writes are
    /// queued and drained, as before xj-to9.48.
    #[cfg(test)]
    drained: AtomicBool,
}

// The VM of this thread's Machine run that may absorb a store at its store
// site (`StoreSite`). A small code write into a memory whose VM holds no live
// field (`Memory::live_store_site`) returns before looking it up.
std::thread_local! {
    static STORE_SITE: std::cell::Cell<*const crate::Vm> =
        const { std::cell::Cell::new(std::ptr::null()) };
}

/// A Machine run's scope in which a store wholly inside a live field is
/// absorbed where it is made (`absorb_at_store`, xj-to9.48), instead of being
/// queued for the dispatcher's drain. It borrows the VM for the scope, so the
/// pointer it publishes stays valid until it drops.
pub(crate) struct StoreSite<'a> {
    previous: *const crate::Vm,
    _vm: std::marker::PhantomData<&'a crate::Vm>,
}

impl<'a> StoreSite<'a> {
    /// While a field is live: otherwise no store can be absorbed, and the
    /// run keeps the queued path.
    #[inline(always)]
    pub(crate) fn enter(vm: &'a crate::Vm) -> Option<Self> {
        vm.live_fields
            .promoted
            .load(Ordering::Relaxed)
            .then(|| Self::publish(vm))
    }

    #[cold]
    #[inline(never)]
    fn publish(vm: &'a crate::Vm) -> Self {
        Self {
            previous: STORE_SITE.with(|site| site.replace(vm)),
            _vm: std::marker::PhantomData,
        }
    }
}

impl Drop for StoreSite<'_> {
    fn drop(&mut self) {
        STORE_SITE.with(|site| site.set(self.previous));
    }
}

/// A code write `[lo, hi)` of at most four bytes into one page of `memory`
/// that no declared code alias shares, from `Memory::note_code_write`.
/// Returns whether this thread's `StoreSite` run absorbed it: it lies wholly
/// inside a live field and no shared translation covers it
/// (`LiveFields::absorb_store`), so the write needs no notice, drain or
/// relift, and every vCPU reads the new bytes when it next executes the
/// field's instruction. Otherwise the caller queues it as before, and the
/// drain decides as it always did (`LiveFields::absorb`).
#[cold]
#[inline(never)]
pub(crate) fn absorb_at_store(memory: &crate::Memory, lo: u64, hi: u64) -> bool {
    let vm = STORE_SITE.with(std::cell::Cell::get);
    if vm.is_null() {
        return false;
    }
    // SAFETY: a `StoreSite` publishes the pointer while it borrows the VM
    // and restores the previous one when it drops, on this thread only.
    let vm = unsafe { &*vm };
    // Fetch approvals are page-wide: a fetch policy never absorbs (`handle_smc`).
    std::ptr::eq(&vm.mem, memory)
        && vm.mem.fetch_policy.is_none()
        && vm.live_fields.absorb_store(&vm.cache, &vm.mem, lo, hi)
}

#[derive(Default)]
struct Table {
    /// Small code writes by first bus byte: (end, count).
    writes: BTreeMap<u64, (u64, u32)>,
    /// Live fields by first bus byte.
    live: BTreeMap<u64, Field>,
    /// The longest instruction a live field has belonged to: a field whose
    /// instruction a write changes starts less than this before or after it.
    reach: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Field {
    end: u64,
    /// The bus range of the field's instruction.
    instruction: (u64, u64),
}

/// The live-field changes one proof lift needs (`LiveFields::commit`).
#[derive(Default)]
pub(crate) struct Plan {
    promote: Vec<(u64, Field)>,
    demote: Vec<u64>,
    refuse: Vec<(u64, u64)>,
    /// The table entries near the bus page this lift fetches from, read once
    /// per page. A change after the read is safe: a promotion advances the
    /// proof generation, so this lift cannot register; a demotion leaves a
    /// field this lift reads live exact; a new count only delays promotion.
    snapshot: Snapshot,
}

#[derive(Default)]
struct Snapshot {
    /// The bus range `[lo, hi)` the entries below cover, by first byte.
    range: (u64, u64),
    /// Written ranges counted `THRESHOLD` times and not refused.
    patched: Vec<(u64, u64)>,
    live: Vec<(u64, Field)>,
}

impl Snapshot {
    fn covers(&self, lo: u64, hi: u64) -> bool {
        self.range.0 <= lo.saturating_sub(MAX_WRITE) && hi <= self.range.1
    }
}

impl LiveFields {
    /// Whether a Machine private proof lift has run: only then are writes
    /// counted (Real16, twenty-bit and Application code never arm it).
    #[inline(always)]
    pub(crate) fn armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed)
    }

    /// Handle a drained code write `[lo, hi)` (one page, one store) once
    /// `armed`. Returns whether it is absorbed: it lies inside a live field
    /// and no shared translation covers it, so the caller performs no
    /// invalidation.
    #[cold]
    #[inline(never)]
    pub(crate) fn absorb(
        &self,
        cache: &TranslationCache,
        memory: &crate::Memory,
        lo: u64,
        hi: u64,
    ) -> bool {
        #[cfg(test)]
        if self.disabled.load(Ordering::Relaxed) {
            return false;
        }
        if hi <= lo || hi - lo > MAX_WRITE {
            if self.active.load(Ordering::Relaxed) {
                self.demote_overlapping(memory, lo, hi);
            }
            return false;
        }
        self.active.store(true, Ordering::Relaxed);
        let mut table = self.table.lock().unwrap();
        let settled = table.settle(lo, hi);
        self.sync(&table, memory);
        if settled.is_some() {
            if cache.absorb_written(lo, hi) {
                return true;
            }
            // A shared translation covers the field: invalidate as before.
            return false;
        }
        if table.writes.len() == MAX_WRITES && !table.writes.contains_key(&lo) {
            table.writes.clear();
        }
        let count = table.writes.entry(lo).or_insert((hi, 0));
        if count.0 != hi {
            *count = (hi, 0);
        }
        if count.1 != REFUSED {
            count.1 += 1;
        }
        false
    }

    /// `absorb` at the store site (`absorb_at_store`, xj-to9.48): a write
    /// `[lo, hi)` of at most four bytes that lies wholly inside a live field
    /// and that no shared translation covers is absorbed, under the same
    /// locks and with the same effect as the drain's. Any other write is
    /// left to the drain, which counts it and demotes what it changes; a
    /// refused write may still have demoted the fields of other instructions
    /// it changes, which the drain would do first in any case.
    #[cold]
    #[inline(never)]
    pub(crate) fn absorb_store(
        &self,
        cache: &TranslationCache,
        memory: &crate::Memory,
        lo: u64,
        hi: u64,
    ) -> bool {
        #[cfg(test)]
        if self.disabled.load(Ordering::Relaxed) || self.drained.load(Ordering::Relaxed) {
            return false;
        }
        if hi <= lo || hi - lo > MAX_WRITE || !self.promoted.load(Ordering::Relaxed) {
            return false;
        }
        // Under the table lock, as in `absorb`: a field is absorbed only once
        // its promotion is committed (`commit`).
        let mut table = self.table.lock().unwrap();
        let settled = table.settle(lo, hi);
        self.sync(&table, memory);
        settled.is_some() && cache.absorb_written(lo, hi)
    }

    #[cold]
    fn demote_overlapping(&self, memory: &crate::Memory, lo: u64, hi: u64) {
        let mut table = self.table.lock().unwrap();
        table
            .live
            .retain(|_, field| !(lo < field.instruction.1 && hi.max(lo + 1) > field.instruction.0));
        self.sync(&table, memory);
    }

    /// Under the table lock: whether a field is live, for `StoreSite` and
    /// for the memory's store path (`Memory::live_store_site`), which then
    /// skip the store-site absorb altogether once the last field has left.
    #[inline(always)]
    fn sync(&self, table: &Table, memory: &crate::Memory) {
        let live = !table.live.is_empty();
        if self.promoted.load(Ordering::Relaxed) != live {
            self.promoted.store(live, Ordering::Relaxed);
        }
        if memory.live_store_site.load(Ordering::Relaxed) != live {
            memory.live_store_site.store(live, Ordering::Relaxed);
        }
    }

    /// Lift the instruction `code` at `ip`, reading its immediate or
    /// displacement live when that field is live or has been patched
    /// `THRESHOLD` times. `bus(i)` is the bus address the fetch translated
    /// byte `i` to. Changes the lift needs go to `plan`.
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn instruction(
        &self,
        code: &[u8],
        ip: u32,
        context: MachineContext,
        profile: &'static crate::ModelProfile,
        alignment_mask: bool,
        bus: impl Fn(u32) -> u64,
        plan: &mut Plan,
    ) -> Result<IrBlock, LiftError> {
        let plain = || {
            crate::lift::legacy::instruction_for_profile(code, ip, context, profile, alignment_mask)
        };
        if !self.armed.load(Ordering::Relaxed) {
            self.armed.store(true, Ordering::Relaxed);
        }
        if !self.active.load(Ordering::Relaxed) || code.is_empty() {
            return plain();
        }
        let first = bus(0);
        let end = first + code.len() as u64;
        let instruction = (first, end);
        // An instruction spans at most two pages, each mapped linearly: its
        // bytes are one bus range if its first and last bytes are.
        let contiguous = bus(code.len() as u32 - 1) == end - 1;
        let (lo, hi) = if contiguous {
            (first, end)
        } else {
            (0..code.len() as u32)
                .map(&bus)
                .fold((u64::MAX, 0), |(lo, hi), b| (lo.min(b), hi.max(b + 1)))
        };
        if !plan.snapshot.covers(lo, hi) {
            let page = lo & !0xfff;
            let range = (
                page.min(lo).saturating_sub(MAX_WRITE),
                (page + 0x1000).max(hi),
            );
            let table = self.table.lock().unwrap();
            plan.snapshot = Snapshot {
                range,
                patched: table
                    .writes
                    .range(range.0..range.1)
                    .filter(|(_, &(_, count))| (THRESHOLD..REFUSED).contains(&count))
                    .map(|(&lo, &(hi, _))| (lo, hi))
                    .collect(),
                live: table
                    .live
                    .range(range.0..range.1)
                    .map(|(&start, &field)| (start, field))
                    .collect(),
            };
        }
        let snapshot = &plan.snapshot;
        // Live fields among this instruction's bytes, and whether a patched
        // range lies among them: otherwise there is no field to read live,
        // and nothing to decode.
        let overlapping: Vec<u64> = snapshot
            .live
            .iter()
            .filter(|(start, field)| {
                if contiguous {
                    *start < end && field.end > first
                } else {
                    (0..code.len() as u32).any(|i| (*start..field.end).contains(&bus(i)))
                }
            })
            .map(|&(start, _)| start)
            .collect();
        let patched_in = |lo: u64, hi: u64| {
            snapshot
                .patched
                .iter()
                .any(|&(start, end)| lo <= start && end <= hi)
        };
        let patched_here = if contiguous {
            patched_in(first, end)
        } else {
            snapshot
                .patched
                .iter()
                .any(|&(lo, hi)| (0..code.len() as u32).any(|i| (lo..hi).contains(&bus(i))))
        };
        if overlapping.is_empty() && !patched_here {
            return plain();
        }
        let fields = contiguous
            .then(|| crate::lift::legacy::live_fields(code, ip, context))
            .flatten();
        let mut requested = [None; 2];
        let mut fresh = [false; 2];
        if let Some(fields) = fields {
            for (kind, field) in fields.fields.iter().enumerate() {
                let Some((offset, bytes)) = *field else {
                    continue;
                };
                let start = first + u64::from(offset);
                let live = Field {
                    end: start + u64::from(bytes),
                    instruction,
                };
                if snapshot.live.contains(&(start, live)) {
                    requested[kind] = Some(start);
                } else if patched_in(start, live.end) {
                    requested[kind] = Some(start);
                    fresh[kind] = true;
                }
            }
        }
        // A patched range outside every field read live (an opcode, ModRM
        // or SIB byte, a form that never reads fields live, or a field this
        // lift cannot read live because its bytes are not one bus range) is
        // not examined again, so no later lift promotes a field this proof
        // bakes.
        let field_spans = fields.map_or([None; 2], |f| f.fields);
        for &(lo, hi) in &snapshot.patched {
            let read_live = requested.iter().zip(field_spans).any(|(start, field)| {
                start.zip(field).is_some_and(|(start, (_, bytes))| {
                    start <= lo && hi <= start + u64::from(bytes)
                })
            });
            let inside = if contiguous {
                first <= lo && hi <= end
            } else {
                (0..code.len() as u32).any(|i| (lo..hi).contains(&bus(i)))
            };
            if inside && !read_live {
                plan.refuse.push((lo, hi));
            }
        }
        // A live field this lift would bake must return to normal.
        for start in overlapping {
            if !requested.contains(&Some(start)) && !plan.demote.contains(&start) {
                plan.demote.push(start);
            }
        }
        let Some(fields) = fields.filter(|_| requested.iter().any(Option::is_some)) else {
            return plain();
        };
        if let Some(ir) = crate::lift::legacy::live_instruction(
            code,
            ip,
            context,
            profile,
            alignment_mask,
            fields,
            requested,
        ) {
            for (kind, start) in requested.iter().enumerate() {
                if let (Some(start), true, Some((_, bytes))) =
                    (*start, fresh[kind], fields.fields[kind])
                {
                    plan.promote.push((
                        start,
                        Field {
                            end: start + u64::from(bytes),
                            instruction,
                        },
                    ));
                }
            }
            return Ok(ir);
        }
        for (kind, start) in requested.iter().enumerate() {
            if let (Some(start), Some((_, bytes))) = (*start, fields.fields[kind]) {
                if fresh[kind] {
                    let end = start + u64::from(bytes);
                    let inside = plan
                        .snapshot
                        .patched
                        .iter()
                        .filter(|&&(lo, hi)| start <= lo && hi <= end);
                    plan.refuse.extend(inside);
                } else {
                    plan.demote.push(start);
                }
            }
        }
        plain()
    }

    /// Apply `plan`. Returns whether the lift may register its proof:
    /// `false` when it baked a field that was live, which may have been
    /// patched since the lift read it (proofs that read it live stay exact,
    /// so only this lift is repeated), and when it promoted a field.
    ///
    /// A promotion is a recorded advance over its instruction's bus bytes,
    /// made under the table lock before any store into the field can be
    /// absorbed: every proof over any of those bytes drops, whatever it
    /// decoded there (a proof over part of the field, or another decoding),
    /// every link naming one ends, and a lift racing this one, which may have
    /// baked the field, fails its registration and lifts again, reading it
    /// live. The promoting lift is repeated too: one extra lift.
    #[must_use]
    #[cold]
    #[inline(never)]
    pub(crate) fn commit(
        &self,
        cache: &TranslationCache,
        memory: &crate::memory::Memory,
        plan: Plan,
    ) -> bool {
        if plan.promote.is_empty() && plan.demote.is_empty() && plan.refuse.is_empty() {
            return true;
        }
        let mut table = self.table.lock().unwrap();
        for (start, end) in plan.refuse {
            table.writes.insert(start, (end, REFUSED));
        }
        for start in &plan.demote {
            if let Some(field) = table.live.remove(start) {
                table.clear_writes(*start, field.end);
            }
        }
        let promoted = !plan.promote.is_empty();
        for (start, field) in plan.promote {
            if table.live.len() >= MAX_LIVE && !table.live.contains_key(&start) {
                // Fields of code no proof holds any more (no page tag: its
                // proofs were dropped, or the page unmapped) leave first.
                table.live.retain(|_, field| {
                    memory.is_code_page(field.instruction.0 >> crate::memory::CODE_PAGE_BITS)
                });
            }
            if table.live.len() < MAX_LIVE || table.live.contains_key(&start) {
                table.clear_writes(start, field.end);
                table.reach = table.reach.max(field.instruction.1 - field.instruction.0);
                table.live.insert(start, field);
            } else {
                // A table of live code promotes nothing more: those fields
                // keep the ordinary relift, and are not examined again.
                for (_, count) in table.writes.range_mut(start..field.end) {
                    count.1 = REFUSED;
                }
            }
            cache.advance_for_promotion(field.instruction.0, field.instruction.1);
        }
        self.sync(&table, memory);
        drop(table);
        plan.demote.is_empty() && !promoted
    }
}

impl Table {
    /// The instruction of the live field that holds all of `[lo, hi)`, if
    /// any; the fields of other instructions the write changes return to
    /// normal.
    fn settle(&mut self, lo: u64, hi: u64) -> Option<(u64, u64)> {
        let containing = self
            .live
            .range(..hi)
            .next_back()
            .filter(|(&start, field)| start <= lo && hi <= field.end)
            .map(|(_, field)| field.instruction);
        // A field lies inside its instruction, at most `reach` bytes long.
        let changed: Vec<u64> = self
            .live
            .range(lo.saturating_sub(self.reach)..hi.saturating_add(self.reach))
            .filter(|(_, field)| {
                Some(field.instruction) != containing
                    && lo < field.instruction.1
                    && hi > field.instruction.0
            })
            .map(|(&start, _)| start)
            .collect();
        for start in changed {
            self.live.remove(&start);
        }
        containing
    }

    fn clear_writes(&mut self, start: u64, end: u64) {
        let inside: Vec<_> = self.writes.range(start..end).map(|(&k, _)| k).collect();
        for key in inside {
            self.writes.remove(&key);
        }
    }
}

/// This thread's published `StoreSite` VM, for the witnesses.
#[cfg(test)]
fn store_site_for_test() -> *const crate::Vm {
    STORE_SITE.with(std::cell::Cell::get)
}

#[cfg(test)]
impl LiveFields {
    fn live_for_test(&self) -> usize {
        self.table.lock().unwrap().live.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        CpuMode, CpuModel, ExceptionDelivery, Exit, Prot, RegionKind, StepOptions, StepStop, Vcpu,
        Vm, VmConfig,
    };
    use std::sync::atomic::Ordering;

    const MEMORY: u64 = 0x20000;
    const CODE: u64 = 0x10000;
    const STACK: u64 = 0x9000;
    const DATA: u32 = 0xa000;
    const TABLE: u64 = 0xc000;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Class {
        Flat32,
        Segmented,
        Paged,
    }

    fn machine(model: CpuModel, class: Class, live: bool, memory: u64) -> (Vm, Vcpu) {
        let mut vm = Vm::new(VmConfig::flat(memory));
        vm.set_cpu_mode(CpuMode::Compat32);
        vm.set_cpu_model(model);
        vm.set_exception_delivery(ExceptionDelivery::Outward);
        vm.map(0, memory as usize, Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.live_fields.disabled.store(!live, Ordering::Relaxed);
        if class == Class::Paged {
            identity_tables(&vm, 0x1000, 0x2000);
        }
        let mut cpu = vm.new_vcpu();
        cpu.cpu.legacy.cr0 = if class == Class::Paged {
            0x8000_0001
        } else {
            1
        };
        cpu.cpu.legacy.cr3 = 0x1000;
        cpu.cpu.legacy.cpl = 0;
        for (i, segment) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
            segment.base = 0;
            // A byte-granular limit below 4 GiB keeps CS.D=1: Segmented.
            segment.limit = if class == Class::Segmented {
                0xf_ffff
            } else {
                u32::MAX
            };
            segment.attributes = (if i == 0 { 0x9b } else { 0x93 }) | 0x4000;
            segment.unusable = 0;
        }
        cpu.set_a20_enabled(true);
        cpu.cpu.gpr[4] = STACK;
        (vm, cpu)
    }

    /// A page directory at `pd` whose one table at `pt` maps the first
    /// 4 MiB to itself.
    fn identity_tables(vm: &Vm, pd: u64, pt: u64) {
        vm.write_bytes(pd, &(pt as u32 | 3).to_le_bytes()).unwrap();
        for page in 0..1024u32 {
            vm.write_bytes(pt + u64::from(4 * page), &((page << 12) | 3).to_le_bytes())
                .unwrap();
        }
    }

    /// One patched form: the instruction, its field's offset and width, and
    /// the field values the guest stores in turn.
    struct Form {
        name: &'static str,
        insn: &'static [u8],
        offset: usize,
        width: usize,
        values: fn(u32) -> u32,
    }

    const IMM: [u32; 16] = [
        0,
        1,
        0x7f,
        0x80,
        0xff,
        0x7fff,
        0x8000,
        0xffff,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_ffff,
        0x1234_5678,
        3,
        0xfe,
        0xa5a5_a5a5,
        0x0001_0000,
    ];

    fn forms() -> Vec<Form> {
        let any = |i| IMM[i as usize % IMM.len()];
        vec![
            // MOV EAX, imm32.
            Form {
                name: "mov imm32",
                insn: b"\xb8\x11\x11\x11\x11",
                offset: 1,
                width: 4,
                values: any,
            },
            // ADD EAX, imm32 (81 /0).
            Form {
                name: "add imm32",
                insn: b"\x81\xc0\x22\x22\x22\x22",
                offset: 2,
                width: 4,
                values: any,
            },
            // SUB EAX, imm8 sign-extended (83 /5).
            Form {
                name: "sub imm8sx",
                insn: b"\x83\xe8\x05",
                offset: 2,
                width: 1,
                values: any,
            },
            // ADC EAX, imm8 sign-extended (83 /2).
            Form {
                name: "adc imm8sx",
                insn: b"\x83\xd0\x05",
                offset: 2,
                width: 1,
                values: any,
            },
            // MOV AX, imm16.
            Form {
                name: "mov imm16",
                insn: b"\x66\xb8\x33\x33",
                offset: 2,
                width: 2,
                values: any,
            },
            // MOV AL, imm8.
            Form {
                name: "mov imm8",
                insn: b"\xb0\x44",
                offset: 1,
                width: 1,
                values: any,
            },
            // CMP EAX, imm32; TEST EAX, imm32; XOR EAX, imm32.
            Form {
                name: "cmp imm32",
                insn: b"\x3d\x55\x55\x55\x55",
                offset: 1,
                width: 4,
                values: any,
            },
            Form {
                name: "test imm32",
                insn: b"\xa9\x66\x66\x66\x66",
                offset: 1,
                width: 4,
                values: any,
            },
            Form {
                name: "xor imm32",
                insn: b"\x35\x77\x77\x77\x77",
                offset: 1,
                width: 4,
                values: any,
            },
            // MOV EAX, [EBP+disp8]: a signed displacement around EBP.
            Form {
                name: "load disp8",
                insn: b"\x8b\x45\x10",
                offset: 2,
                width: 1,
                values: any,
            },
            // MOV EAX, [disp32]: absolute.
            Form {
                name: "load disp32",
                insn: b"\x8b\x05\x00\xa0\x00\x00",
                offset: 2,
                width: 4,
                values: |i| DATA + (IMM[i as usize % IMM.len()] & 0xffc),
            },
            // LEA EAX, [EBP+disp32].
            Form {
                name: "lea disp32",
                insn: b"\x8d\x85\x00\x01\x00\x00",
                offset: 2,
                width: 4,
                values: any,
            },
            // MOV DWORD [EBP+disp32], imm32: its displacement, then its immediate.
            Form {
                name: "store disp32",
                insn: b"\xc7\x85\x00\x02\x00\x00\x88\x88\x88\x88",
                offset: 2,
                width: 4,
                values: |i| IMM[i as usize % IMM.len()] & 0xffc,
            },
            Form {
                name: "store imm32",
                insn: b"\xc7\x85\x00\x02\x00\x00\x88\x88\x88\x88",
                offset: 6,
                width: 4,
                values: any,
            },
            // ADD EAX, [BP+disp8] with 16-bit addressing (67h).
            Form {
                name: "add disp8 a16",
                insn: b"\x67\x03\x46\x10",
                offset: 3,
                width: 1,
                values: any,
            },
        ]
    }

    /// The census loop: per iteration, the patched instruction, a fold of
    /// EAX, the flags and EBX into ESI, and the store of the next field value
    /// from the table at EDI. `before` places the store immediately before
    /// the patched instruction, in the same block; otherwise it follows the
    /// fold. Returns the code and the patched field's address.
    fn census(form: &Form, before: bool) -> (Vec<u8>, u64) {
        let store: &[u8] = match form.width {
            4 => b"\x89\x15",
            2 => b"\x66\x89\x15",
            _ => b"\x88\x15",
        };
        let load = b"\x8b\x17"; // MOV EDX, [EDI]
                                // PUSHFD; POP EBX; AND EBX, 8D5h; ADD ESI, EBX; ADD ESI, EAX; ROL ESI, 3.
        let fold = b"\x9c\x5b\x81\xe3\xd5\x08\x00\x00\x01\xde\x01\xc6\xc1\xc6\x03";
        let store_len = (load.len() + store.len() + 4) as u64;
        let insn_at = CODE + if before { store_len } else { 0 };
        let field = insn_at + form.offset as u64;
        let mut patch = load.to_vec();
        patch.extend_from_slice(store);
        patch.extend_from_slice(&(field as u32).to_le_bytes());
        let mut code = Vec::new();
        if before {
            code.extend_from_slice(&patch);
        }
        code.extend_from_slice(form.insn);
        code.extend_from_slice(fold);
        if !before {
            code.extend_from_slice(&patch);
        }
        // ADD EDI, 4; DEC ECX; JNZ top; HLT.
        code.extend_from_slice(b"\x83\xc7\x04\x49\x0f\x85");
        let rel = -(code.len() as i32 + 4);
        code.extend_from_slice(&rel.to_le_bytes());
        code.push(0xf4);
        (code, field)
    }

    fn load_census(vm: &Vm, cpu: &mut Vcpu, form: &Form, before: bool, n: u32) {
        let (code, _) = census(form, before);
        vm.write_bytes(CODE, &code).unwrap();
        for i in 0..n {
            vm.write_bytes(TABLE + u64::from(4 * i), &(form.values)(i).to_le_bytes())
                .unwrap();
        }
        for i in 0..0x400u32 {
            vm.write_bytes(
                u64::from(DATA) - 0x100 + u64::from(4 * i),
                &i.wrapping_mul(0x9e37_79b9).to_le_bytes(),
            )
            .unwrap();
        }
        cpu.cpu.rip = CODE;
        cpu.cpu.gpr[0] = 0x0102_0304;
        cpu.cpu.gpr[1] = u64::from(n);
        cpu.cpu.gpr[5] = u64::from(DATA);
        cpu.cpu.gpr[6] = 0;
        cpu.cpu.gpr[7] = TABLE;
    }

    fn memory(vm: &Vm) -> Vec<u8> {
        let mut bytes = vec![0; MEMORY as usize];
        vm.read_bytes(0, &mut bytes).unwrap();
        bytes
    }

    /// Run both machines in the same slices until HLT, comparing each stop
    /// and the complete state.
    fn lockstep(
        a: (&Vm, &mut Vcpu),
        b: (&Vm, &mut Vcpu),
        slice: Option<u64>,
        end: u64,
        what: &str,
    ) {
        let (avm, a) = a;
        let (bvm, b) = b;
        for round in 0.. {
            assert!(round < 100_000, "{what}: no HLT");
            let budget = slice.map_or(1_000_000, |k| 1 + (round % k));
            let ar = a.run_steps(avm, budget, StepOptions::default()).unwrap();
            let br = b.run_steps(bvm, budget, StepOptions::default()).unwrap();
            assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what} round {round}");
            assert_eq!(
                format!("{:?}", a.cpu),
                format!("{:?}", b.cpu),
                "{what} round {round}"
            );
            assert_eq!(a.retired_instructions(), b.retired_instructions(), "{what}");
            match ar.stop() {
                StepStop::Exit(Exit::Hlt) => break,
                // Under TF, stop at the HLT the loop reaches.
                StepStop::Exit(Exit::Exception { vector: 1, .. }) if a.cpu.rip == end => break,
                StepStop::Exit(Exit::BudgetExhausted | Exit::Exception { vector: 1, .. }) => {}
                stop => panic!("{what}: {stop:?}"),
            }
        }
        assert!(memory(avm) == memory(bvm), "{what}: memory");
    }

    #[test]
    fn live_fields_match_relifting_on_every_patched_form() {
        const N: u32 = 32;
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            for class in [Class::Flat32, Class::Segmented, Class::Paged] {
                for form in forms() {
                    for before in [false, true] {
                        for (slice, tf) in [(None, false), (Some(5), false), (None, true)] {
                            let what = format!(
                                "{model:?} {class:?} {} before={before} slice={slice:?} tf={tf}",
                                form.name
                            );
                            let (vm, mut cpu) = machine(model, class, true, MEMORY);
                            let (rvm, mut reference) = machine(model, class, false, MEMORY);
                            for (vm, cpu) in [(&vm, &mut cpu), (&rvm, &mut reference)] {
                                load_census(vm, cpu, &form, before, N);
                                if tf {
                                    cpu.cpu.rflags_system |= 1 << 8;
                                }
                            }
                            let end = CODE + census(&form, before).0.len() as u64 - 1;
                            lockstep((&vm, &mut cpu), (&rvm, &mut reference), slice, end, &what);
                            // The reference relifts after every patch; the live
                            // run promotes the field once and absorbs the rest.
                            let (live, relift) = (lifts(&cpu), lifts(&reference));
                            assert!(relift >= u64::from(N), "{what}: {relift}");
                            if !tf {
                                assert_eq!(vm.live_fields.live_for_test(), 1, "{what}");
                                // Budget cuts lift suffix blocks under both.
                                let factor = if slice.is_some() { 1 } else { 3 };
                                assert!(factor * live < relift, "{what}: {live} vs {relift} lifts");
                            }
                        }
                    }
                }
            }
        }
    }

    /// The proof lifts of `cpu`'s unpaged and paged maps.
    fn lifts(cpu: &Vcpu) -> u64 {
        let execution = cpu.legacy_execution.as_ref().unwrap();
        execution.flat32.lifts + execution.paging.as_ref().map_or(0, |p| p.lifts)
    }

    /// Run from `rip` to HLT; returns EAX.
    fn run(vm: &Vm, cpu: &mut Vcpu, rip: u64) -> u32 {
        cpu.cpu.rip = rip;
        let exit = cpu.run(vm, None);
        assert!(matches!(exit, Exit::Hlt), "{exit:?}");
        cpu.cpu.gpr[0] as u32
    }

    /// `MOV [field], ECX; HLT` at `at`: a 4-byte guest store.
    fn patcher(vm: &Vm, at: u64, field: u64) {
        let mut code = b"\x89\x0d".to_vec();
        code.extend_from_slice(&(field as u32).to_le_bytes());
        code.push(0xf4);
        vm.write_bytes(at, &code).unwrap();
    }

    fn patch(vm: &Vm, cpu: &mut Vcpu, at: u64, value: u32) {
        cpu.cpu.gpr[1] = u64::from(value);
        run(vm, cpu, at);
    }

    /// `MOV EAX, imm32; HLT`.
    fn target(vm: &Vm, at: u64, imm: u32) {
        let mut code = vec![0xb8];
        code.extend_from_slice(&imm.to_le_bytes());
        code.push(0xf4);
        vm.write_bytes(at, &code).unwrap();
    }

    #[test]
    fn a_patch_crossing_into_the_opcode_resets_the_field() {
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            let (vm, mut cpu) = machine(model, Class::Flat32, true, MEMORY);
            target(&vm, CODE, 0x1111_1111);
            patcher(&vm, CODE + 0x100, CODE + 1);
            patcher(&vm, CODE + 0x200, CODE);
            assert_eq!(run(&vm, &mut cpu, CODE), 0x1111_1111);
            for value in 1..=4 {
                patch(&vm, &mut cpu, CODE + 0x100, value);
                assert_eq!(run(&vm, &mut cpu, CODE), value);
            }
            // The second patch promoted the field; later ones are absorbed.
            assert_eq!(vm.live_fields.live_for_test(), 1);
            let lifted = lifts(&cpu);
            patch(&vm, &mut cpu, CODE + 0x100, 5);
            assert_eq!(run(&vm, &mut cpu, CODE), 5);
            assert_eq!(lifts(&cpu), lifted);
            // B8 55 66 77 over the opcode and three immediate bytes: the same
            // opcode, but the store crosses the field. It relifts and the
            // field returns to normal.
            patch(&vm, &mut cpu, CODE + 0x200, 0x7766_55b8);
            assert_eq!(run(&vm, &mut cpu, CODE), 0x0077_6655);
            assert_eq!(vm.live_fields.live_for_test(), 0);
            assert!(lifts(&cpu) > lifted);
            // Two more patches inside the field make it live again.
            for value in [6, 7] {
                let lifted = lifts(&cpu);
                patch(&vm, &mut cpu, CODE + 0x100, value);
                assert_eq!(run(&vm, &mut cpu, CODE), value);
                assert!(lifts(&cpu) > lifted);
            }
            assert_eq!(vm.live_fields.live_for_test(), 1);
            // The promotion's gap dropped the patcher's proofs too.
            patch(&vm, &mut cpu, CODE + 0x100, 8);
            assert_eq!(run(&vm, &mut cpu, CODE), 8);
            let lifted = lifts(&cpu);
            patch(&vm, &mut cpu, CODE + 0x100, 9);
            assert_eq!(run(&vm, &mut cpu, CODE), 9);
            assert_eq!(lifts(&cpu), lifted);
            // A one-byte opcode store (MOV EAX -> MOV ECX) demotes and relifts.
            vm.write_bytes(CODE, b"\xb9").unwrap();
            cpu.cpu.gpr[0] = 0;
            assert_eq!(run(&vm, &mut cpu, CODE), 0);
            assert_eq!(cpu.cpu.gpr[1], 9);
            assert_eq!(vm.live_fields.live_for_test(), 0);
        }
    }

    #[test]
    fn another_vcpus_patch_is_seen_at_the_next_execution() {
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            for class in [Class::Flat32, Class::Segmented, Class::Paged] {
                let (vm, mut a) = machine(model, class, true, MEMORY);
                let mut b = vm.new_vcpu();
                b.cpu = a.cpu.clone();
                target(&vm, CODE, 0);
                patcher(&vm, CODE + 0x100, CODE + 1);
                for value in 1..=8u32 {
                    let lifted = lifts(&a);
                    patch(&vm, &mut b, CODE + 0x100, value.wrapping_mul(0x0101_0101));
                    assert_eq!(run(&vm, &mut a, CODE), value.wrapping_mul(0x0101_0101));
                    if value > 3 {
                        // Absorbed: A's proof survives B's store.
                        assert_eq!(lifts(&a), lifted, "{class:?}");
                    }
                }
                assert_eq!(vm.live_fields.live_for_test(), 1);
            }
        }
    }

    #[test]
    fn a_remapped_code_page_reads_the_field_through_its_own_translation() {
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            let (vm, mut cpu) = machine(model, Class::Paged, true, MEMORY);
            // A second directory maps linear page 10h to frame 11h.
            identity_tables(&vm, 0x3000, 0x4000);
            vm.write_bytes(0x4000 + 4 * 0x10, &0x11003u32.to_le_bytes())
                .unwrap();
            target(&vm, CODE, 0x1000_0000);
            target(&vm, CODE + 0x1000, 0x2000_0000);
            patcher(&vm, CODE + 0x2000, CODE + 1);
            for value in 1..=3 {
                patch(&vm, &mut cpu, CODE + 0x2000, value);
                assert_eq!(run(&vm, &mut cpu, CODE), value);
            }
            assert_eq!(vm.live_fields.live_for_test(), 1);
            cpu.cpu.legacy.cr3 = 0x3000;
            assert_eq!(run(&vm, &mut cpu, CODE), 0x2000_0000);
            for value in 100..=103 {
                patch(&vm, &mut cpu, CODE + 0x2000, value);
                assert_eq!(run(&vm, &mut cpu, CODE), value);
            }
            assert_eq!(vm.live_fields.live_for_test(), 2);
            cpu.cpu.legacy.cr3 = 0x1000;
            assert_eq!(run(&vm, &mut cpu, CODE), 3);
            patch(&vm, &mut cpu, CODE + 0x2000, 4);
            assert_eq!(run(&vm, &mut cpu, CODE), 4);
            let mut field = [0; 4];
            vm.read_bytes(CODE + 0x1001, &mut field).unwrap();
            assert_eq!(u32::from_le_bytes(field), 103);
        }
    }

    #[test]
    fn a20_selects_the_field_the_fetch_reads() {
        const HIGH: u64 = 0x10_1000;
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            let (vm, mut cpu) = machine(model, Class::Flat32, true, 0x12_0000);
            // With A20 masked, linear 101000h fetches physical 1000h.
            target(&vm, HIGH, 0x1000_0000);
            target(&vm, HIGH & 0xf_ffff, 0x2000_0000);
            patcher(&vm, 0x5000, HIGH + 1);
            for value in 1..=3 {
                patch(&vm, &mut cpu, 0x5000, value);
                assert_eq!(run(&vm, &mut cpu, HIGH), value);
            }
            assert_eq!(vm.live_fields.live_for_test(), 1);
            cpu.set_a20_enabled(false);
            assert_eq!(run(&vm, &mut cpu, HIGH), 0x2000_0000);
            for value in 10..=13 {
                patch(&vm, &mut cpu, 0x5000, value);
                assert_eq!(run(&vm, &mut cpu, HIGH), value);
            }
            assert_eq!(vm.live_fields.live_for_test(), 2);
            cpu.set_a20_enabled(true);
            assert_eq!(run(&vm, &mut cpu, HIGH), 3);
            let mut field = [0; 4];
            vm.read_bytes(0x1001, &mut field).unwrap();
            assert_eq!(u32::from_le_bytes(field), 13);
        }
    }

    /// One linked run of the patching loop below from a fresh machine:
    /// (EAX, EBX, links made, links followed, lifts).
    fn linked_patch_loop(model: CpuModel, class: Class, live: bool, links: bool) -> [u64; 5] {
        // A: MOV EAX, imm32; ADD EBX, EAX; JMP B. B: INC ECX;
        // MOV [A+1], ECX; DEC EDI; JNZ A; HLT. B's store patches A's
        // immediate on every iteration; both blocks end in a direct exit.
        let mut code = b"\xb8\x00\x00\x00\x00\x01\xc3\xeb\x00\x41\x89\x0d".to_vec();
        code.extend_from_slice(&(CODE as u32 + 1).to_le_bytes());
        code.extend_from_slice(b"\x4f\x75\xed\xf4");
        let (vm, mut cpu) = machine(model, class, live, MEMORY);
        vm.write_bytes(CODE, &code).unwrap();
        cpu.cpu.rip = CODE;
        cpu.cpu.gpr[7] = 300;
        crate::vm::legacy::links::OFF.with(|off| off.set(!links));
        let exit = cpu.run(&vm, None);
        crate::vm::legacy::links::OFF.with(|off| off.set(false));
        assert!(matches!(exit, Exit::Hlt), "{exit:?}");
        let links = &cpu.legacy_execution.as_ref().unwrap().links;
        [
            cpu.cpu.gpr[0],
            cpu.cpu.gpr[3],
            links.made,
            links.followed,
            lifts(&cpu),
        ]
    }

    #[test]
    fn an_absorbed_patch_keeps_the_links_of_blocks_that_read_it_live() {
        // A store absorbed by a live field leaves the link generation: the
        // linked block reads the field when it executes, so the links made
        // once stay followed. Without live fields every patch relifts A and
        // ends its links. All three agree with the run that refuses links.
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            for class in [Class::Flat32, Class::Segmented, Class::Paged] {
                let reference = linked_patch_loop(model, class, true, false);
                let linked = linked_patch_loop(model, class, true, true);
                let relifted = linked_patch_loop(model, class, false, true);
                let what = format!("{model:?} {class:?}");
                assert_eq!(reference[..2], linked[..2], "{what}");
                assert_eq!(reference[..2], relifted[..2], "{what}");
                // EBX sums the immediates 0 and 1..=299.
                assert_eq!(linked[1], 299 * 300 / 2, "{what}");
                assert!(linked[2] <= 6, "{what}: {} links made", linked[2]);
                // A -> B on every iteration; B drains its store, so B -> A
                // takes the full lookup (`links.rs`).
                assert!(linked[3] >= 290, "{what}: {} followed", linked[3]);
                assert!(linked[4] <= 12, "{what}: {} lifts", linked[4]);
                assert!(relifted[3] < 50, "{what}: {:?}", relifted);
                assert!(relifted[4] >= 300, "{what}: {} lifts", relifted[4]);
            }
        }
    }

    /// `MOV [field], CL; HLT` at `at`: a 1-byte guest store.
    fn byte_patcher(vm: &Vm, at: u64, field: u64) {
        let mut code = b"\x88\x0d".to_vec();
        code.extend_from_slice(&(field as u32).to_le_bytes());
        code.push(0xf4);
        vm.write_bytes(at, &code).unwrap();
    }

    #[test]
    fn a_lift_that_bakes_a_live_field_returns_it_to_normal() {
        // MOV EAX, imm32 at CODE, its immediate patched live. Entered at
        // CODE+1 the immediate's bytes B0 07 F4 decode as MOV AL,7; HLT:
        // that lift bakes the field's bytes and so must demote it, or the
        // byte store below would be absorbed and the block keep AL = 7.
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            for class in [Class::Flat32, Class::Paged] {
                let (vm, mut cpu) = machine(model, class, true, MEMORY);
                target(&vm, CODE, 0);
                patcher(&vm, CODE + 0x100, CODE + 1);
                byte_patcher(&vm, CODE + 0x200, CODE + 2);
                for value in [0x90f4_07b1, 0x90f4_07b0] {
                    patch(&vm, &mut cpu, CODE + 0x100, value);
                    assert_eq!(run(&vm, &mut cpu, CODE), value);
                }
                assert_eq!(vm.live_fields.live_for_test(), 1);
                assert_eq!(run(&vm, &mut cpu, CODE + 1) & 0xff, 7);
                assert_eq!(vm.live_fields.live_for_test(), 0, "{model:?} {class:?}");
                patch(&vm, &mut cpu, CODE + 0x200, 9);
                assert_eq!(
                    run(&vm, &mut cpu, CODE + 1) & 0xff,
                    9,
                    "{model:?} {class:?}"
                );
                assert_eq!(run(&vm, &mut cpu, CODE), 0x90f4_09b0);
            }
        }
    }

    #[test]
    fn a_proof_that_baked_the_field_before_its_promotion_is_dropped() {
        // vCPU A lifts the target after its first patch, baking the field.
        // B's second patch promotes it with B's own lift, and B's third is
        // absorbed: A, which never saw a patch drained, still sees the
        // third value, because the second patch dropped A's proof
        // (`LiveFields::commit`, a promotion needs no advance).
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            for class in [Class::Flat32, Class::Segmented, Class::Paged] {
                let (vm, mut a) = machine(model, class, true, MEMORY);
                let mut b = vm.new_vcpu();
                b.cpu = a.cpu.clone();
                target(&vm, CODE, 0);
                patcher(&vm, CODE + 0x100, CODE + 1);
                patch(&vm, &mut b, CODE + 0x100, 1);
                assert_eq!(run(&vm, &mut a, CODE), 1);
                patch(&vm, &mut b, CODE + 0x100, 2);
                assert_eq!(run(&vm, &mut b, CODE), 2);
                assert_eq!(vm.live_fields.live_for_test(), 1);
                let lifted = lifts(&b);
                patch(&vm, &mut b, CODE + 0x100, 3);
                assert_eq!(run(&vm, &mut b, CODE), 3);
                assert_eq!(lifts(&b), lifted, "{class:?}: absorbed");
                assert_eq!(run(&vm, &mut a, CODE), 3, "{model:?} {class:?}");
            }
        }
    }

    #[test]
    fn a_field_a_shared_block_covers_is_not_absorbed() {
        // vCPU A (Flat32) reads the imm8 of MOV AL, imm8 live; vCPU B runs
        // the same bytes in real mode, from the shared cache, which bakes
        // them. A store into the field must still invalidate B's block.
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            let (vm, mut a) = machine(model, Class::Flat32, true, MEMORY);
            vm.write_bytes(CODE, b"\xb0\x00\xf4").unwrap();
            byte_patcher(&vm, CODE + 0x100, CODE + 1);
            for value in [1, 2] {
                patch(&vm, &mut a, CODE + 0x100, value);
                assert_eq!(run(&vm, &mut a, CODE) & 0xff, value);
            }
            assert_eq!(vm.live_fields.live_for_test(), 1);
            let mut b = vm.new_vcpu();
            b.cpu = crate::CpuState::legacy_real_entry([0x1000, 0x1000, 0x1000, 0x1000, 0, 0]);
            b.cpu.gpr[4] = 0x8000;
            for value in [3, 4, 5] {
                patch(&vm, &mut a, CODE + 0x100, value);
                assert_eq!(run(&vm, &mut b, 0) & 0xff, value, "{model:?}");
                assert_eq!(run(&vm, &mut a, CODE) & 0xff, value, "{model:?}");
            }
        }
    }

    /// The 16-bit loop: MOV AX, imm16; ADD BX, AX; INC CX;
    /// MOV [0001h], CX (the immediate); DEC DI; JNZ 0; JMP $. Entered at
    /// offset 0 of a segment based at CODE, it runs 100 iterations.
    const LOOP16: &[u8] = b"\xb8\x00\x00\x01\xc3\x41\x89\x0e\x01\x00\x4f\x75\xf3\xeb\xfe";

    fn run16(vm: &Vm, cpu: &mut Vcpu) {
        vm.write_bytes(CODE, LOOP16).unwrap();
        cpu.cpu.rip = 0;
        cpu.cpu.gpr[7] = 100;
        let run = cpu.run_steps(vm, 600, StepOptions::default()).unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
            "{run:?}"
        );
    }

    fn census16(machine: impl Fn(bool) -> (Vm, Vcpu), what: &str) {
        let (vm, mut cpu) = machine(true);
        let (rvm, mut reference) = machine(false);
        run16(&vm, &mut cpu);
        run16(&rvm, &mut reference);
        assert_eq!(
            format!("{:?}", cpu.cpu),
            format!("{:?}", reference.cpu),
            "{what}"
        );
        assert_eq!(cpu.retired_instructions(), reference.retired_instructions());
        // BX sums the immediates 0..=99.
        assert_eq!(cpu.cpu.gpr[3] & 0xffff, 4950, "{what}");
        assert_eq!(vm.live_fields.live_for_test(), 1, "{what}");
        let (live, relift) = (lifts(&cpu), lifts(&reference));
        assert!(3 * live < relift, "{what}: {live} vs {relift} lifts");
    }

    #[test]
    fn an_80286_protected_segmented_loop_reads_its_immediate_live() {
        census16(
            |live| {
                let (vm, mut cpu) = machine(CpuModel::I80286, Class::Segmented, live, MEMORY);
                for (i, segment) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
                    *segment = crate::SegmentCache {
                        base: CODE as u32,
                        limit: 0xffff,
                        attributes: if i == 0 { 0x9b } else { 0x93 },
                        ..Default::default()
                    };
                }
                cpu.cpu.legacy.cr0 = 1;
                cpu.cpu.gpr[4] = 0x8000;
                (vm, cpu)
            },
            "80286",
        );
    }

    #[test]
    fn a_paged_virtual_8086_loop_reads_its_immediate_live() {
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            census16(
                |live| {
                    let (vm, _) = machine(model, Class::Paged, live, MEMORY);
                    // User pages: the identity tables at 1000h with U set.
                    vm.write_bytes(0x1000, &0x2007u32.to_le_bytes()).unwrap();
                    for page in 0..1024u32 {
                        vm.write_bytes(
                            0x2000 + u64::from(4 * page),
                            &((page << 12) | 7).to_le_bytes(),
                        )
                        .unwrap();
                    }
                    let mut cpu = vm.new_vcpu();
                    cpu.cpu =
                        crate::CpuState::legacy_real_entry([0x1000, 0x1000, 0x1000, 0x1000, 0, 0]);
                    cpu.cpu.legacy.cr0 = 0x8000_0001;
                    cpu.cpu.legacy.cr3 = 0x1000;
                    cpu.cpu.legacy.cpl = 3;
                    cpu.cpu.rflags_system = 0x20000;
                    cpu.cpu.gpr[4] = 0x8000;
                    cpu.set_a20_enabled(true);
                    (vm, cpu)
                },
                &format!("{model:?} V86"),
            );
        }
    }

    /// `B8 00 00 B0 07 F4`: MOV EAX, imm32; HLT at CODE, and from inside
    /// its immediate, at CODE+3, MOV AL,7; HLT: a proof over the field's
    /// last two bytes and the HLT after it.
    fn overlapped(vm: &Vm) {
        vm.write_bytes(CODE, b"\xb8\x00\x00\xb0\x07\xf4").unwrap();
        byte_patcher(vm, CODE + 0x100, CODE + 1);
        byte_patcher(vm, CODE + 0x200, CODE + 4);
    }

    #[test]
    fn a_promotion_drops_a_proof_over_part_of_its_field() {
        // CODE+3's proof covers the field's bytes 2..4, none of those the
        // counted writes (CODE+1) changed. The promotion itself must drop
        // it, or the absorbed store to CODE+4 leaves it AL = 7.
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            for class in [Class::Flat32, Class::Segmented, Class::Paged] {
                let (vm, mut cpu) = machine(model, class, true, MEMORY);
                overlapped(&vm);
                assert_eq!(run(&vm, &mut cpu, CODE + 3) & 0xff, 7);
                for value in [1, 2] {
                    patch(&vm, &mut cpu, CODE + 0x100, value);
                    assert_eq!(run(&vm, &mut cpu, CODE), 0x07b0_0000 | value);
                }
                assert_eq!(vm.live_fields.live_for_test(), 1);
                patch(&vm, &mut cpu, CODE + 0x200, 9);
                assert_eq!(
                    run(&vm, &mut cpu, CODE + 3) & 0xff,
                    9,
                    "{model:?} {class:?}"
                );
            }
        }
    }

    thread_local! {
        static RACER: std::cell::RefCell<Option<Vcpu>> = const { std::cell::RefCell::new(None) };
    }

    // Between the source read and the registration of another vCPU's lift:
    // the racer runs the promoting lift.
    fn promote_in_between(vm: &Vm) {
        RACER.with(|racer| {
            let mut a = racer.borrow_mut().take().unwrap();
            assert_eq!(run(vm, &mut a, CODE), 0x07b0_0002);
            assert_eq!(vm.live_fields.live_for_test(), 1, "the racer promoted");
            *racer.borrow_mut() = Some(a);
        });
    }

    #[test]
    fn a_promotion_fails_a_racing_lift_that_baked_its_field() {
        // B lifts CODE+3 (baking the field's bytes 2..4) while A's lift
        // promotes the field: B's registration must fail, so that its relift
        // meets the live field and demotes it; otherwise B keeps a proof
        // that bakes a live field and the store to CODE+4 is absorbed.
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            for class in [Class::Flat32, Class::Segmented, Class::Paged] {
                let (vm, mut a) = machine(model, class, true, MEMORY);
                let mut b = vm.new_vcpu();
                b.cpu = a.cpu.clone();
                overlapped(&vm);
                for value in [1, 2] {
                    patch(&vm, &mut a, CODE + 0x100, value);
                }
                // Drain the second patch first: only the promotion may
                // advance the generation during B's lift.
                run(&vm, &mut b, CODE + 5);
                RACER.with(|racer| *racer.borrow_mut() = Some(a));
                *vm.source_read_hook.lock().unwrap() = Some(promote_in_between);
                assert_eq!(run(&vm, &mut b, CODE + 3) & 0xff, 7);
                assert!(
                    vm.source_read_hook.lock().unwrap().is_none(),
                    "the race ran"
                );
                let mut a = RACER.with(|racer| racer.borrow_mut().take()).unwrap();
                patch(&vm, &mut a, CODE + 0x200, 9);
                assert_eq!(run(&vm, &mut b, CODE + 3) & 0xff, 9, "{model:?} {class:?}");
                assert_eq!(run(&vm, &mut a, CODE), 0x09b0_0002);
            }
        }
    }

    #[test]
    fn a_store_site_absorb_queues_no_notice() {
        // xj-to9.48: in a Machine run's `StoreSite` scope, a store wholly
        // inside a live field is absorbed where it is made: nothing is queued
        // for a drain, no vCPU relifts, and both read the new bytes at their
        // next execution of the instruction. Queued (`drained`), the same
        // store leaves its notice and has the same effect.
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            for class in [Class::Flat32, Class::Segmented, Class::Paged] {
                for drained in [false, true] {
                    let what = format!("{model:?} {class:?} drained {drained}");
                    let (vm, mut cpu) = machine(model, class, true, MEMORY);
                    vm.live_fields.drained.store(drained, Ordering::Relaxed);
                    target(&vm, CODE, 0);
                    patcher(&vm, CODE + 0x100, CODE + 1);
                    for value in 1..=3 {
                        patch(&vm, &mut cpu, CODE + 0x100, value);
                        assert_eq!(run(&vm, &mut cpu, CODE), value, "{what}");
                    }
                    let mut other = vm.new_vcpu();
                    other.cpu = cpu.cpu.clone();
                    assert_eq!(run(&vm, &mut other, CODE), 3, "{what}");
                    for value in 4..=6 {
                        let lifted = (lifts(&cpu), lifts(&other));
                        {
                            let _site = super::StoreSite::enter(&vm).unwrap();
                            vm.mem.write(CODE + 1, u64::from(value), 4).unwrap();
                        }
                        assert_eq!(vm.mem.has_dirty_code(), drained, "{what}");
                        assert_eq!(run(&vm, &mut cpu, CODE), value, "{what}");
                        assert_eq!(run(&vm, &mut other, CODE), value, "{what}");
                        assert_eq!((lifts(&cpu), lifts(&other)), lifted, "{what}");
                    }
                }
            }
        }
    }

    /// A real-mode vCPU of `vm` at `CS:0`; runs to HLT and returns AX.
    fn real_ax(vm: &Vm, real: &mut Vcpu) -> u32 {
        run(vm, real, 0) & 0xffff
    }

    fn real_vcpu(vm: &Vm, cs: u16) -> Vcpu {
        let mut real = vm.new_vcpu();
        let entry = crate::CpuState::legacy_real_entry([cs; 6]);
        real.cpu.legacy = entry.legacy;
        (real.cpu.cs, real.cpu.ds, real.cpu.es) = (cs, cs, cs);
        (real.cpu.ss, real.cpu.fs, real.cpu.gs) = (cs, cs, cs);
        real
    }

    #[test]
    fn a_store_site_patch_under_a_shared_translation_invalidates_it() {
        // xj-to9.48: a real-mode vCPU runs the field's bytes through a shared
        // translation (MOV AX, imm16; NOP; NOP; HLT), which bakes them. A
        // patch inside the live field is not absorbed at its store site
        // (`TranslationCache::absorb_written`), so the shared copy drops.
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            let (vm, mut cpu) = machine(model, Class::Flat32, true, MEMORY);
            target(&vm, CODE, 0x9090_0000);
            patcher(&vm, CODE + 0x100, CODE + 1);
            for value in 1..=3 {
                patch(&vm, &mut cpu, CODE + 0x100, 0x9090_0000 | value);
                assert_eq!(run(&vm, &mut cpu, CODE), 0x9090_0000 | value);
            }
            assert_eq!(vm.live_fields.live_for_test(), 1);
            let mut real = real_vcpu(&vm, (CODE >> 4) as u16);
            assert_eq!(real_ax(&vm, &mut real), 3);
            for value in 4..=6 {
                patch(&vm, &mut cpu, CODE + 0x100, 0x9090_0000 | value);
                assert_eq!(real_ax(&vm, &mut real), value, "{model:?}");
                assert_eq!(run(&vm, &mut cpu, CODE), 0x9090_0000 | value);
            }
        }
    }

    #[test]
    fn a_store_site_patch_reaches_a_declared_code_alias() {
        // xj-to9.48: with a code alias declared on the field's page, a patch
        // is queued with its alias range, which drops the proof another vCPU
        // made of the alias view. Its lift baked the field: no write had
        // reached the alias before the declaration.
        const ALIAS: u64 = CODE + 0x2000;
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            let (vm, mut cpu) = machine(model, Class::Flat32, true, MEMORY);
            target(&vm, CODE, 0);
            target(&vm, ALIAS, 0);
            patcher(&vm, CODE + 0x3000, CODE + 1);
            for value in 1..=3 {
                patch(&vm, &mut cpu, CODE + 0x3000, value);
                assert_eq!(run(&vm, &mut cpu, CODE), value);
            }
            assert_eq!(vm.live_fields.live_for_test(), 1);
            let mut view = vm.new_vcpu();
            view.cpu = cpu.cpu.clone();
            assert_eq!(run(&vm, &mut view, ALIAS), 0);
            vm.mem.declare_code_alias(CODE, ALIAS, 0x1000).unwrap();
            // After two writes reach it, the alias view's field reads live too.
            for value in 4..=5 {
                let lifted = lifts(&view);
                patch(&vm, &mut cpu, CODE + 0x3000, value);
                assert_eq!(run(&vm, &mut cpu, CODE), value);
                // The backing does not alias: the view keeps its bytes.
                assert_eq!(run(&vm, &mut view, ALIAS), 0);
                assert!(lifts(&view) > lifted, "{model:?} {value}");
            }
        }
    }

    #[test]
    fn a_store_site_patch_revokes_a_fetch_approval() {
        // xj-to9.48: a fetch policy's approvals are page-wide, so no patch is
        // absorbed at its store site once one is set: the next fetch from the
        // written page asks the policy again.
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            let (mut vm, mut cpu) = machine(model, Class::Flat32, true, MEMORY);
            target(&vm, CODE, 0);
            patcher(&vm, CODE + 0x3000, CODE + 1);
            for value in 1..=3 {
                patch(&vm, &mut cpu, CODE + 0x3000, value);
                assert_eq!(run(&vm, &mut cpu, CODE), value);
            }
            assert_eq!(vm.live_fields.live_for_test(), 1);
            let asked = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            let count = asked.clone();
            vm.set_fetch_policy(Box::new(move |_| {
                count.fetch_add(1, Ordering::Relaxed);
                crate::FetchVerdict::Allow
            }));
            assert_eq!(run(&vm, &mut cpu, CODE), 3);
            for value in 4..=6 {
                patch(&vm, &mut cpu, CODE + 0x3000, value);
                let before = asked.load(Ordering::Relaxed);
                assert_eq!(run(&vm, &mut cpu, CODE), value);
                assert!(asked.load(Ordering::Relaxed) > before, "{model:?} {value}");
            }
        }
    }

    /// A machine whose field at `CODE + 1` (`MOV EAX, imm32; HLT`) is live,
    /// patched by `MOV [CODE+1], ECX; HLT` at `CODE + 0x100`.
    fn live_machine(model: CpuModel, class: Class, drained: bool) -> (Vm, Vcpu) {
        let (vm, mut cpu) = machine(model, class, true, MEMORY);
        vm.live_fields.drained.store(drained, Ordering::Relaxed);
        target(&vm, CODE, 0x9090_0000);
        patcher(&vm, CODE + 0x100, CODE + 1);
        for value in 1..=3 {
            patch(&vm, &mut cpu, CODE + 0x100, 0x9090_0000 | value);
            assert_eq!(run(&vm, &mut cpu, CODE), 0x9090_0000 | value);
        }
        assert_eq!(vm.live_fields.live_for_test(), 1);
        (vm, cpu)
    }

    #[test]
    fn an_absorbed_patch_is_reported_by_take_dirty_code() {
        // xj-to9.48: a store absorbed at its store site queues no range, but
        // `Memory::take_dirty_code` still reports its page, as it reports a
        // queued store's (`drained`); once.
        for class in [Class::Flat32, Class::Segmented, Class::Paged] {
            for drained in [false, true] {
                // The vCPU holds the proofs whose registrations tag the page.
                let (vm, _cpu) = live_machine(CpuModel::I80486Dx, class, drained);
                vm.mem.take_dirty_code();
                {
                    let _site = super::StoreSite::enter(&vm).unwrap();
                    vm.mem.write(CODE + 1, 0x9090_0004, 4).unwrap();
                }
                let what = format!("{class:?} drained {drained}");
                assert_eq!(vm.mem.has_dirty_code(), drained, "{what}");
                assert_eq!(vm.mem.take_dirty_code(), vec![CODE >> 12], "{what}");
                assert!(vm.mem.take_dirty_code().is_empty(), "{what}");
            }
        }
    }

    #[test]
    fn the_last_demoted_field_ends_the_store_site() {
        // xj-to9.48: once no field is live, small code stores take the
        // ordinary path again: no `StoreSite`, no store-site absorb.
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            let (vm, mut cpu) = live_machine(model, Class::Flat32, false);
            assert!(vm.mem.live_store_site.load(Ordering::Relaxed));
            assert!(super::StoreSite::enter(&vm).is_some());
            // MOV EAX -> MOV ECX: the opcode store demotes the field.
            vm.write_bytes(CODE, b"\xb9").unwrap();
            run(&vm, &mut cpu, CODE);
            assert_eq!(vm.live_fields.live_for_test(), 0);
            assert!(!vm.mem.live_store_site.load(Ordering::Relaxed), "{model:?}");
            assert!(super::StoreSite::enter(&vm).is_none(), "{model:?}");
        }
    }

    #[test]
    fn a_store_into_another_vms_memory_is_not_absorbed_by_this_one() {
        // xj-to9.48: two VMs with the same live field; the second also runs
        // it through a shared real-mode translation. A store into the
        // second's memory made in the first's `StoreSite` scope (as a device
        // callback during its run would) is the second's to settle: it is
        // queued, and the shared copy drops.
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            let (vm1, _cpu1) = live_machine(model, Class::Flat32, false);
            let (vm2, mut cpu2) = live_machine(model, Class::Flat32, false);
            let mut real = real_vcpu(&vm2, (CODE >> 4) as u16);
            assert_eq!(real_ax(&vm2, &mut real), 3);
            for value in 4..=6 {
                {
                    let _site = super::StoreSite::enter(&vm1).unwrap();
                    vm2.mem
                        .write(CODE + 1, u64::from(0x9090_0000 | value), 4)
                        .unwrap();
                }
                assert!(vm2.mem.has_dirty_code(), "{model:?}");
                assert_eq!(real_ax(&vm2, &mut real), value, "{model:?}");
                assert_eq!(run(&vm2, &mut cpu2, CODE), 0x9090_0000 | value);
            }
        }
    }

    #[test]
    fn a_run_restores_the_store_site_it_found() {
        // xj-to9.48: a Machine run publishes its VM for the store-site absorb
        // only while it runs, after a HLT, a budget or a fault alike, and
        // restores the scope around it.
        let (vm, mut cpu) = live_machine(CpuModel::I80486Dx, Class::Flat32, false);
        assert!(super::store_site_for_test().is_null());
        run(&vm, &mut cpu, CODE);
        assert!(super::store_site_for_test().is_null());
        cpu.cpu.rip = CODE;
        let exit = cpu.run(&vm, Some(1));
        assert!(matches!(exit, Exit::BudgetExhausted), "{exit:?}");
        assert!(super::store_site_for_test().is_null());
        // UD2 at the patcher's HLT.
        vm.write_bytes(CODE + 0x106, b"\x0f\x0b").unwrap();
        cpu.cpu.rip = CODE + 0x100;
        let exit = cpu.run(&vm, None);
        assert!(!matches!(exit, Exit::Hlt), "{exit:?}");
        assert!(super::store_site_for_test().is_null());
        let other = Vm::new(VmConfig::flat(0x1000));
        {
            let _outer = super::StoreSite::publish(&other);
            run(&vm, &mut cpu, CODE);
            assert!(std::ptr::eq(super::store_site_for_test(), &other));
        }
        assert!(super::store_site_for_test().is_null());
    }

    #[test]
    fn a_paged_code_store_that_is_not_absorbed_ends_its_block() {
        // xj-to9.48: a paged block's store into its own next instruction's
        // opcode takes the fast lane once the TLB holds the page dirty; it
        // is queued (no live field), its notice ends the block, and the
        // patched instruction executes as rewritten (INC EAX -> INC ECX).
        for model in [CpuModel::I80386, CpuModel::I80486Dx] {
            let (vm, mut cpu) = live_machine(model, Class::Paged, false);
            // MOV byte [CODE+0x207], 41h; INC EAX; HLT at CODE + 0x200.
            let at = CODE + 0x200;
            let mut code = b"\xc6\x05".to_vec();
            code.extend_from_slice(&(at as u32 + 7).to_le_bytes());
            code.extend_from_slice(b"\x41\x40\xf4");
            for round in 0..3 {
                vm.write_bytes(at, &code).unwrap();
                cpu.cpu.gpr[0] = 0;
                cpu.cpu.gpr[1] = 0;
                run(&vm, &mut cpu, at);
                assert_eq!(
                    (cpu.cpu.gpr[0], cpu.cpu.gpr[1]),
                    (0, 1),
                    "{model:?} round {round}"
                );
            }
        }
    }
}
