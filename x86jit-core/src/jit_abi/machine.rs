//! Compiled Machine blocks (decision-29, J1): the helpers their loads and stores
//! call, and which instructions they complete themselves.
//!
//! A compiled Machine block is a faster producer of the interpreted walk's
//! result. It completes a prefix of its block and returns `RET_CONTINUE`,
//! [`RET_MACHINE_DEFER`](super::RET_MACHINE_DEFER) or
//! [`RET_MACHINE_STOP`](super::RET_MACHINE_STOP) with the completed count in
//! [`MemCtx::machine_ordinal`](super::MemCtx::machine_ordinal); the dispatcher
//! charges it and, after a defer, walks the rest of the block on the interpreter.
use super::MemCtx;
use crate::ir::IrOp;
use crate::state::CpuState;
use std::sync::atomic::{AtomicU64, Ordering};

/// Marks a run-time defer (an access helper's or an inline check's) in
/// `MemCtx::machine_ordinal`; an unconditional defer leaves it clear.
pub const HELPER_DEFER: u64 = 1 << 63;

/// Helper status: the access completed.
pub const ACCESS_DONE: u64 = 0;
/// Helper status: nothing happened; the block defers the instruction.
pub const ACCESS_DEFER: u64 = 1;
/// Store status: the store completed and the vCPU's dispatch word is set, so
/// the block stops after this instruction, as the interpreted walk does
/// (`BlockPass::commit`).
pub const ACCESS_STOP: u64 = 2;

/// One slot of a vCPU's compiled TLB view (decision-30 §4, decision-32 §3),
/// `repr(C)` because compiled loads and stores read it inline. A tag is the
/// linear page number plus one, so zero is invalid. The supervisor pair
/// keeps J2's offsets; the user pair is appended (decision-32 Q4): a unit's
/// key names its CPL, so its lowering picks the pair at compile time.
#[repr(C, align(64))]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct FastEntry {
    /// Supervisor loads may go to this page inline (RAM or ROM).
    pub read_tag: u32,
    /// Supervisor stores may go to this page inline (RAM with a code-tag
    /// word; with paging, a write right for CPL 0 and PTE.D set).
    pub write_tag: u32,
    /// Host address of the page's first byte minus its linear address.
    pub addend: u64,
    /// The live code-tag word of the page's bus frame (`Memory::code_tag_slot`).
    pub tag_word: u64,
    /// The bus frame number, for the store barrier's physical address.
    pub frame: u32,
    /// The frame's bit in `tag_word`.
    pub tag_bit: u32,
    /// CPL-3 loads may go to this page inline: with paging, U set in both
    /// levels (80386 PRM §§6.4.1-6.4.2 Table 6-5; SDM Vol 3A §5.6.1).
    pub user_read_tag: u32,
    /// CPL-3 stores may go to this page inline: with paging, U and W set in
    /// both levels and PTE.D set.
    pub user_write_tag: u32,
}

/// Slots of the view: direct-mapped on the linear page's low bits.
pub const TLB_SLOTS: usize = 256;
/// Byte offsets inside a [`FastEntry`], for generated code.
pub const ENTRY_READ_TAG: i32 = 0;
pub const ENTRY_WRITE_TAG: i32 = 4;
pub const ENTRY_ADDEND: i32 = 8;
pub const ENTRY_TAG_WORD: i32 = 16;
pub const ENTRY_FRAME: i32 = 24;
pub const ENTRY_TAG_BIT: i32 = 28;
pub const ENTRY_USER_READ_TAG: i32 = 32;
pub const ENTRY_USER_WRITE_TAG: i32 = 36;
/// `log2(size_of::<FastEntry>())`.
pub const ENTRY_SHIFT: i64 = 6;

/// The read and write tag offsets a unit compiled at `cpl` probes.
pub fn tag_offsets(cpl: u8) -> (i32, i32) {
    if cpl == 3 {
        (ENTRY_USER_READ_TAG, ENTRY_USER_WRITE_TAG)
    } else {
        (ENTRY_READ_TAG, ENTRY_WRITE_TAG)
    }
}

/// The compiled view of a vCPU's TLB (decision-30 §4, decision-32 §3).
///
/// With paging off its entries are identity fills from the bus: the linear
/// page's frame after the model's physical mask and A20. A fill reads the
/// region map and the code-tag table, so the view is emptied whenever A20,
/// the mapping or the physical mask it was filled under changes.
///
/// With paging on it is a subset of the vCPU's interpreter TLB
/// (`interp::legacy::paging::Tlb`): every valid entry equals a current entry
/// there, with that entry's frame, rights and D bit. Only the miss helper
/// fills it, and only from a TLB hit; compiled code never walks. The TLB
/// logs the pages each of its mutations touches, and every compiled entry
/// clears those slots before the call (`sync`), so the view never holds a
/// translation the TLB has dropped while compiled code can read it.
///
/// Code-tag words are read live, so a page tagged after the fill is seen by
/// the next store.
///
/// The vCPU's shadow return ring for compiled Machine CALL and RET
/// (decision-31 §2.1) follows the entries, at [`RING_OFFSET`]: Machine code
/// reaches both through `MemCtx::ret_stack`, so `MemCtx` keeps its layout.
#[repr(C)]
pub struct MachineTlb {
    pub entries: [FastEntry; TLB_SLOTS],
    pub ring: super::RetStack,
    a20: bool,
    paged: bool,
    mask: u32,
    /// With paging on, the TLB (`RefCell<Tlb>`) the view was last synced
    /// with, for the miss helper; zero otherwise. An address, so the view
    /// stays `Send` with its vCPU.
    source: usize,
}

/// Byte offset of [`MachineTlb::ring`], for generated code.
pub const RING_OFFSET: i64 = std::mem::offset_of!(MachineTlb, ring) as i64;

impl Default for MachineTlb {
    fn default() -> Self {
        Self {
            entries: [FastEntry::default(); TLB_SLOTS],
            ring: super::RetStack::new(),
            a20: false,
            paged: false,
            mask: 0,
            source: 0,
        }
    }
}

type PagedTlb = std::cell::RefCell<crate::interp::legacy::paging::Tlb>;

impl MachineTlb {
    /// Empty the view unless it was filled under this A20 line, mask and
    /// paging state: a PG change in either direction empties it.
    /// A mapping change empties it at run entry (`compiled::State::begin_run`).
    #[inline(always)]
    pub(crate) fn reconcile(&mut self, a20: bool, mask: u32, paged: bool) {
        if self.a20 != a20 || self.mask != mask || self.paged != paged {
            self.flush();
            self.a20 = a20;
            self.mask = mask;
            self.paged = paged;
            self.source = 0;
        }
    }

    #[cold]
    pub(crate) fn flush(&mut self) {
        self.entries = [FastEntry::default(); TLB_SLOTS];
    }

    /// With paging on, before a compiled entry: clear the slots of every
    /// page the TLB changed since the last sync (decision-32 §3 rule 2), and
    /// name the TLB the miss helper fills from. Clearing is always safe:
    /// the view is a subset of the TLB, not a copy of it. The TLB is first
    /// reconciled with `cpu` (one compare of its context), so a changed CR3,
    /// CR0.WP or A20 empties both before the view is read: write tags granted
    /// under WP = 0 cannot outlive a WP change, whatever path reached here.
    #[inline(always)]
    pub(crate) fn sync(&mut self, cpu: &CpuState, tlb: &PagedTlb) {
        debug_assert!(self.paged);
        self.source = std::ptr::from_ref(tlb) as usize;
        let mut tlb = tlb.borrow_mut();
        tlb.reconcile(cpu);
        if !tlb.view_log_empty() {
            self.clear_logged(&mut tlb);
        }
    }

    #[cold]
    #[inline(never)]
    fn clear_logged(&mut self, tlb: &mut crate::interp::legacy::paging::Tlb) {
        match tlb.take_view_log() {
            None => self.flush(),
            Some(pages) => {
                for page in pages {
                    self.entries[page as usize % TLB_SLOTS] = FastEntry::default();
                }
            }
        }
    }

    /// Fill the slot of `linear`'s page after a completed helper access: an
    /// identity entry when its whole bus frame is RAM or ROM in the backing.
    #[cold]
    #[inline(never)]
    fn fill(&mut self, cpu: &CpuState, memory: &crate::Memory, mask: u32, linear: u32) {
        if self.paged || self.mask != mask || self.a20 != (cpu.legacy.a20_enabled != 0) {
            return;
        }
        let page = linear >> 12;
        let base = u64::from(page) << 12;
        let (frame, _) = crate::legacy::physical_span(cpu, mask, base, 1);
        let slot = &mut self.entries[page as usize % TLB_SLOTS];
        let Some((host, writable)) = memory.fast_frame(frame) else {
            return;
        };
        let tag = memory.code_tag_slot(frame >> 12);
        let write = if writable && tag.is_some() {
            page + 1
        } else {
            0
        };
        // No page protection exists with PG = 0: both privileges alike.
        *slot = FastEntry {
            read_tag: page + 1,
            write_tag: write,
            addend: host.wrapping_sub(base),
            tag_word: tag.map_or(0, |(word, _)| word as u64),
            frame: (frame >> 12) as u32,
            tag_bit: tag.map_or(0, |(_, bit)| bit),
            user_read_tag: page + 1,
            user_write_tag: write,
        };
    }

    /// Paged (decision-32 §3): fill the slot of `linear`'s page from the
    /// TLB entry the helper's access just hit. The tags are the
    /// interpreter's own rights for the CPU state now (`permitted`, WP and
    /// the model included), a write tag needs PTE.D, and the frame must be
    /// one RAM or ROM region of the backing, RAM with a code-tag word for a
    /// write tag.
    #[cold]
    #[inline(never)]
    fn fill_paged(&mut self, cpu: &CpuState, memory: &crate::Memory, linear: u32) {
        if !self.paged || self.source == 0 {
            return;
        }
        // SAFETY: `sync` set it at this compiled entry from the vCPU's live
        // paging state, which outlives the compiled call.
        let cell = unsafe { &*(self.source as *const PagedTlb) };
        let tlb = cell.borrow();
        let Some(rights) = crate::interp::legacy::paging::view_rights(cpu, &tlb, linear) else {
            return;
        };
        let page = linear >> 12;
        let base = u64::from(page) << 12;
        let (frame, _) = crate::legacy::physical_span(cpu, u32::MAX, u64::from(rights.frame), 1);
        let Some((host, writable)) = memory.fast_frame(frame) else {
            return;
        };
        let tag = memory.code_tag_slot(frame >> 12);
        let writes = writable && tag.is_some();
        let tag_of = |ok: bool| if ok { page + 1 } else { 0 };
        self.entries[page as usize % TLB_SLOTS] = FastEntry {
            read_tag: page + 1,
            write_tag: tag_of(writes && rights.supervisor_write),
            addend: host.wrapping_sub(base),
            tag_word: tag.map_or(0, |(word, _)| word as u64),
            frame: (frame >> 12) as u32,
            tag_bit: tag.map_or(0, |(_, bit)| bit),
            user_read_tag: tag_of(rights.user_read),
            user_write_tag: tag_of(writes && rights.user_write),
        };
        #[cfg(debug_assertions)]
        self.assert_subset(cpu, &tlb);
    }

    /// Decision-32 §3: every valid entry equals a current TLB entry, with
    /// its frame and no right or D bit the TLB does not hold.
    #[cfg(debug_assertions)]
    fn assert_subset(&self, cpu: &CpuState, tlb: &crate::interp::legacy::paging::Tlb) {
        for (index, entry) in self.entries.iter().enumerate() {
            let tag = entry.read_tag | entry.write_tag | entry.user_read_tag | entry.user_write_tag;
            if tag == 0 {
                continue;
            }
            let page = entry.read_tag.wrapping_sub(1);
            assert!(
                entry.read_tag != 0 && page as usize % TLB_SLOTS == index,
                "a view slot {index} without its page: {entry:?}"
            );
            let rights = crate::interp::legacy::paging::view_rights(cpu, tlb, page << 12)
                .unwrap_or_else(|| panic!("view page {page:#x} has no TLB entry"));
            let (frame, _) =
                crate::legacy::physical_span(cpu, u32::MAX, u64::from(rights.frame), 1);
            assert_eq!(u64::from(entry.frame), frame >> 12, "view page {page:#x}");
            assert!(entry.write_tag == 0 || rights.supervisor_write, "{page:#x}");
            assert!(entry.user_read_tag == 0 || rights.user_read, "{page:#x}");
            assert!(entry.user_write_tag == 0 || rights.user_write, "{page:#x}");
        }
    }
}

/// Whether a J1 compiled block lowers this op itself.
pub fn lowered_op(op: &IrOp) -> bool {
    crate::interp::legacy::machine_jit::lowered_op(op)
}

/// Whether a J1 compiled block completes this instruction itself, given its ops
/// after the `InsnStart` marker. Every other instruction is an unconditional
/// defer at its start.
pub fn lowered_instruction(ops: &[IrOp]) -> bool {
    crate::interp::legacy::machine_jit::lowered_instruction(ops)
}

/// How a Machine block's last instruction transfers (decision-31 §3), for
/// its lowering: a near CALL pushes the shadow return ring, a near RET
/// predicts through it. The ring only gates a chain, so this bears on cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MachineExit {
    /// Any other last instruction.
    Plain,
    /// A near CALL that pushes this return address.
    Call(u64),
    /// A near RET, with or without imm16: the last `Jump` is to a temp and
    /// the block's `IrBlock::link_exit` flag holds. The flag, recorded when
    /// the block is published, holds for a direct exit or a near RET, so with
    /// an indirect jump it names a near RET.
    Return,
}

/// The last instruction's transfer kind of a Machine block.
pub fn machine_exit(ir: &crate::ir::IrBlock) -> MachineExit {
    use crate::ir::Val;
    let start = ir
        .ops
        .iter()
        .rposition(|op| matches!(op, IrOp::InsnStart { .. }))
        .unwrap_or(0);
    let last = &ir.ops[start..];
    let Some(IrOp::Jump { target }) = last
        .iter()
        .rev()
        .find(|op| !matches!(op, IrOp::Legacy(crate::legacy::LegacyOp::SharedWalk)))
    else {
        return MachineExit::Plain;
    };
    // A near CALL (`lift::legacy`, `push_value` then `Jump`): the return
    // address is the next instruction's offset in the operand size, and
    // ESP is written.
    let end = ir.guest_start.wrapping_add(u64::from(ir.guest_len));
    let sp = last.iter().any(|op| {
        matches!(
            op,
            IrOp::WriteReg {
                reg: crate::Reg::Rsp,
                ..
            }
        )
    });
    let pushed = last.iter().find_map(|op| match op {
        IrOp::Store {
            src: Val::Imm(value),
            ..
        } if *value == end & 0xffff_ffff || *value == end & 0xffff => Some(*value),
        _ => None,
    });
    match (target, pushed) {
        (_, Some(value)) if sp => MachineExit::Call(value),
        (Val::Temp(_), _) if ir.link_exit => MachineExit::Return,
        _ => MachineExit::Plain,
    }
}

/// The helpers' baked access configuration: the model's physical mask, the
/// run's host-fault routing and its memory-consistency tier.
pub fn access_config(physical_mask: u32, routed: bool, order: u8) -> u64 {
    u64::from(physical_mask) | (u64::from(routed) << 32) | (u64::from(order) << 33)
}

/// The configuration a background Machine request compiles with
/// (decision-30 §2): the context's physical mask, no host-fault routing
/// (production requests come only from VMs without it) and `order`.
pub fn request_config(context: &crate::legacy::MachineContext, order: u8) -> u64 {
    access_config(
        crate::legacy::physical_mask(context.generation),
        false,
        order,
    )
}

fn split(config: u64) -> (u32, bool, u8) {
    (
        config as u32,
        config & (1 << 32) != 0,
        (config >> 33) as u8 & 3,
    )
}

/// Machine load helper: [`ACCESS_DONE`] with the value in `MemCtx::machine_value`,
/// or [`ACCESS_DEFER`].
///
/// # Safety
/// `cpu` and `mem` are the live `CpuState` and `MemCtx` of a Machine compiled
/// call, whose `mem_self` names the run's `Memory`.
pub unsafe extern "C" fn machine_load(
    cpu: *mut u8,
    mem: *mut u8,
    addr: u64,
    size: u64,
    config: u64,
    _time_enabled: u64,
) -> u64 {
    // SAFETY: the compiled-block ABI supplies these live, per-vCPU pointers.
    let (cpu, ctx) = unsafe { (&mut *(cpu as *mut CpuState), &mut *(mem as *mut MemCtx)) };
    // SAFETY: `for_memory` retains the Memory for the whole compiled call.
    let memory = unsafe { &*(ctx.mem_self as *const crate::Memory) };
    let (mask, routed, order) = split(config);
    // In: the instruction's IP, which the compiled block stored there.
    let ip = ctx.machine_value;
    match checked_access(ctx, cpu, memory, mask, routed, order, addr, size, None, ip) {
        crate::interp::legacy::machine_jit::Access::Done(value) => {
            ctx.machine_value = value;
            fill(ctx, cpu, memory, mask, routed, addr, size);
            ACCESS_DONE
        }
        crate::interp::legacy::machine_jit::Access::Defer => ACCESS_DEFER,
    }
}

/// Machine store helper: [`ACCESS_DONE`], [`ACCESS_STOP`] or [`ACCESS_DEFER`].
///
/// # Safety
/// As [`machine_load`].
pub unsafe extern "C" fn machine_store(
    cpu: *mut u8,
    mem: *mut u8,
    addr: u64,
    size: u64,
    value: u64,
    config: u64,
    _time_enabled: u64,
) -> u64 {
    // SAFETY: the compiled-block ABI supplies these live, per-vCPU pointers.
    let (cpu, ctx) = unsafe { (&mut *(cpu as *mut CpuState), &mut *(mem as *mut MemCtx)) };
    // SAFETY: `for_memory` retains the Memory for the whole compiled call.
    let memory = unsafe { &*(ctx.mem_self as *const crate::Memory) };
    let (mask, routed, order) = split(config);
    // In: the instruction's IP, which the compiled block stored there.
    let ip = ctx.machine_value;
    match checked_access(
        ctx,
        cpu,
        memory,
        mask,
        routed,
        order,
        addr,
        size,
        Some(value),
        ip,
    ) {
        crate::interp::legacy::machine_jit::Access::Done(_) => {
            fill(ctx, cpu, memory, mask, routed, addr, size);
            // SAFETY: never null; a Machine run points it at its exit word.
            let word = unsafe { &*(ctx.dispatch_word_ptr as *const AtomicU64) };
            if word.load(Ordering::Relaxed) != 0 {
                ACCESS_STOP
            } else {
                ACCESS_DONE
            }
        }
        crate::interp::legacy::machine_jit::Access::Defer => ACCESS_DEFER,
    }
}

/// The helpers' checked access: unpaged, J1's; paged, the decision-32 §3
/// miss helper, which completes an access only from an existing TLB
/// translation and otherwise defers without walking.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn checked_access(
    ctx: &MemCtx,
    cpu: &mut CpuState,
    memory: &crate::Memory,
    mask: u32,
    routed: bool,
    order: u8,
    addr: u64,
    size: u64,
    store: Option<u64>,
    ip: u64,
) -> crate::interp::legacy::machine_jit::Access {
    if cpu.legacy.cr0 & PAGING == 0 {
        return crate::interp::legacy::machine_jit::access(
            cpu, memory, mask, routed, order, addr, size as u8, store, ip, None,
        );
    }
    if ctx.ret_stack == 0 {
        return crate::interp::legacy::machine_jit::Access::Defer;
    }
    // SAFETY: a Machine run points `ret_stack` at its vCPU's view, whose
    // `sync` named the live TLB at this compiled entry.
    let source = unsafe { (*(ctx.ret_stack as *const MachineTlb)).source };
    if source == 0 {
        return crate::interp::legacy::machine_jit::Access::Defer;
    }
    // SAFETY: as above; the paging state outlives the compiled call.
    let tlb = unsafe { &*(source as *const PagedTlb) };
    crate::interp::legacy::machine_jit::paged_access(
        cpu, memory, tlb, routed, order, addr, size as u8, store, ip,
    )
}

/// CR0.PG (SDM Vol 3A §2.5).
const PAGING: u32 = 1 << 31;

/// After a helper access inside one page, fill that page's slot of the view,
/// when the run has one and its accesses are inlined (no host-fault routing).
#[inline(always)]
fn fill(
    ctx: &MemCtx,
    cpu: &CpuState,
    memory: &crate::Memory,
    mask: u32,
    routed: bool,
    addr: u64,
    size: u64,
) {
    if ctx.ret_stack == 0 || routed || (addr & 0xfff) + size > 0x1000 {
        return;
    }
    // SAFETY: a Machine run points `ret_stack` at its vCPU's view, which
    // nothing else touches during the compiled call.
    let tlb = unsafe { &mut *(ctx.ret_stack as *mut MachineTlb) };
    if cpu.legacy.cr0 & PAGING == 0 {
        tlb.fill(cpu, memory, mask, addr as u32);
    } else {
        tlb.fill_paged(cpu, memory, addr as u32);
    }
}

/// Machine store barrier (decision-30 §4): an inline store wrote a page whose
/// code tag is set, or some page is watched. Reports the bytes as the
/// interpreter's write does (`Memory::note_write`: alias expansion, the
/// store-site absorb of live fields, watched pages), then [`ACCESS_STOP`] when
/// the vCPU's dispatch word is set, else [`ACCESS_DONE`].
///
/// # Safety
/// As [`machine_load`].
pub unsafe extern "C" fn machine_note(
    _cpu: *mut u8,
    mem: *mut u8,
    physical: u64,
    size: u64,
    _time_enabled: u64,
) -> u64 {
    // SAFETY: the compiled-block ABI supplies these live, per-vCPU pointers.
    let ctx = unsafe { &*(mem as *const MemCtx) };
    // SAFETY: `for_memory` retains the Memory for the whole compiled call.
    let memory = unsafe { &*(ctx.mem_self as *const crate::Memory) };
    memory.note_write(physical, size as usize);
    // SAFETY: never null; a Machine run points it at its exit word.
    let word = unsafe { &*(ctx.dispatch_word_ptr as *const AtomicU64) };
    if word.load(Ordering::Relaxed) != 0 {
        ACCESS_STOP
    } else {
        ACCESS_DONE
    }
}

/// The access configuration's routing bit: set, compiled accesses always
/// call the helpers (no inline path).
pub const CONFIG_ROUTED: u64 = 1 << 32;

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    /// Generated code reads `FastEntry` fields at these offsets and indexes the
    /// view with `ENTRY_SHIFT` (decision-30 §4).
    #[test]
    fn fast_entry_layout_matches_generated_code() {
        assert_eq!(size_of::<FastEntry>(), 1 << ENTRY_SHIFT);
        assert_eq!(align_of::<FastEntry>(), 64);
        assert_eq!(offset_of!(FastEntry, read_tag) as i32, ENTRY_READ_TAG);
        assert_eq!(offset_of!(FastEntry, write_tag) as i32, ENTRY_WRITE_TAG);
        assert_eq!(offset_of!(FastEntry, addend) as i32, ENTRY_ADDEND);
        assert_eq!(offset_of!(FastEntry, tag_word) as i32, ENTRY_TAG_WORD);
        assert_eq!(offset_of!(FastEntry, frame) as i32, ENTRY_FRAME);
        assert_eq!(offset_of!(FastEntry, tag_bit) as i32, ENTRY_TAG_BIT);
        assert_eq!(
            offset_of!(FastEntry, user_read_tag) as i32,
            ENTRY_USER_READ_TAG
        );
        assert_eq!(
            offset_of!(FastEntry, user_write_tag) as i32,
            ENTRY_USER_WRITE_TAG
        );
        assert_eq!(tag_offsets(0), (ENTRY_READ_TAG, ENTRY_WRITE_TAG));
        assert_eq!(tag_offsets(3), (ENTRY_USER_READ_TAG, ENTRY_USER_WRITE_TAG));
        assert_eq!(offset_of!(MachineTlb, entries), 0);
        assert_eq!(offset_of!(MachineTlb, ring) as i64, RING_OFFSET);
        assert_eq!(RING_OFFSET % 8, 0);
        assert!(TLB_SLOTS.is_power_of_two());
        assert_eq!(
            size_of::<[FastEntry; TLB_SLOTS]>(),
            TLB_SLOTS << ENTRY_SHIFT
        );
    }
}
