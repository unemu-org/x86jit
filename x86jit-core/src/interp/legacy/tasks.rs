//! Hardware tasks on the Machine interpreter (spec.md §17; decisions 18, 23).
//! 286 PRM §§8.2–8.5 pp. 8-1–9, App. B SWITCH_TASKS pp. B-12–13;
//! 386 PRM §§7.1–7.6 pp. 7-1–10. Page images checked at the pinned editions.
use super::*;
use crate::legacy::{CpuGeneration, SystemSegment};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cause {
    Jump,
    Call,
    Return,
    Event,
}

#[derive(Default)]
pub(crate) struct SwitchState {
    pub committed: bool,
    pub completed: bool,
    pending: Option<Pending>,
}
impl SwitchState {
    pub fn context(&self) -> Option<MachineContext> {
        self.pending.as_ref().map(|p| p.context)
    }
    pub fn context_for(&self, cpu: &CpuState) -> Option<MachineContext> {
        self.pending
            .as_ref()
            .filter(|p| p.owner.matches(cpu))
            .map(|p| p.context)
    }
    pub fn abandon_if_changed(&mut self, cpu: &CpuState) -> bool {
        let abandon = self.pending.as_ref().is_some_and(|p| !p.owner.matches(cpu));
        if abandon {
            self.reset();
        }
        abandon
    }
    pub fn seal_exit(&mut self, cpu: &CpuState, exit: &Exit) {
        if let Some(pending) = &mut self.pending {
            pending.owner = Owner::new(cpu);
            pending.owner.completion = match exit {
                Exit::MmioRead { .. } => Completion::Read,
                Exit::MmioWrite { .. } => Completion::Write,
                _ => Completion::None,
            };
        }
    }
    pub fn suppress_debug(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|p| p.cause == Cause::Event)
    }
    pub fn single_step(&self) -> Option<bool> {
        self.pending.as_ref().map(|p| p.single_step)
    }
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy)]
enum Completion {
    None,
    Read,
    Write,
}
struct Owner {
    cpu: Box<CpuState>,
    completion: Completion,
}
impl Owner {
    fn new(cpu: &CpuState) -> Self {
        Self {
            cpu: Box::new(cpu.clone()),
            completion: Completion::None,
        }
    }
    fn matches(&self, cpu: &CpuState) -> bool {
        // Exhaustive destructuring makes appended CPU state part of the owner
        // contract. Only the completion requested by this exit may differ.
        macro_rules! same_fields {
            ($($field:ident),* $(,)?) => {{
                let CpuState {
                    $($field: _,)*
                    flags: _, pending_mmio: _, pending_mmio_write: _,
                    cpu_source: _, cpuid_table: _, port_string_in: _,
                } = cpu;
                true $(&& self.cpu.$field == cpu.$field)*
            }};
        }
        fn same_arc<T: ?Sized>(
            a: &Option<std::sync::Arc<T>>,
            b: &Option<std::sync::Arc<T>>,
        ) -> bool {
            match (a, b) {
                (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
                (None, None) => true,
                _ => false,
            }
        }
        fn same_port(
            a: Option<crate::state::PortStringIn>,
            b: Option<crate::state::PortStringIn>,
        ) -> bool {
            match (a, b) {
                (Some(a), Some(b)) => {
                    macro_rules! compare {
                        ($($field:ident),* $(,)?) => {{
                            let crate::state::PortStringIn { $($field: _,)* } = a;
                            true $(&& a.$field == b.$field)*
                        }};
                    }
                    compare!(
                        rip,
                        next_ip,
                        instruction_end,
                        base,
                        offset,
                        next_index,
                        next_count,
                        addr_bits,
                        elem,
                        rep,
                        value
                    )
                }
                (None, None) => true,
                _ => false,
            }
        }
        same_fields!(
            gpr,
            rip,
            fs_base,
            gs_base,
            xmm,
            ymm_hi,
            zmm_hi,
            kmask,
            fpr,
            fpu_top,
            fpu_cw,
            fpu_pad,
            pending_port_in,
            features,
            x87_precision,
            cs,
            ds,
            es,
            ss,
            fpu_empty,
            fpu_sw,
            fpu_fip,
            fpu_fdp,
            fpu_fop,
            fpu_fcs,
            fpu_fds,
            mmio_parts,
            mmio_parts_len,
            mmio_parts_rip,
            mmio_retry_active,
            mmio_retry_cs,
            cpuid_overrides,
            source_id,
            random_state,
            rflags_system,
            fs,
            gs,
            mxcsr,
            x87_pointer_model,
            legacy,
            fpu_model,
            fpu_emulation,
            task_io_map_base,
            task_debug_trap,
        ) && self.cpu.flags == cpu.flags
            && self.cpu.flags.pf_src == cpu.flags.pf_src
            && self.cpu.flags.af_src == cpu.flags.af_src
            && same_arc(&self.cpu.cpu_source, &cpu.cpu_source)
            && same_arc(&self.cpu.cpuid_table, &cpu.cpuid_table)
            && same_port(self.cpu.port_string_in, cpu.port_string_in)
            && (matches!(self.completion, Completion::Read)
                || self.cpu.pending_mmio == cpu.pending_mmio)
            && (matches!(self.completion, Completion::Write)
                || self.cpu.pending_mmio_write == cpu.pending_mmio_write)
    }
}

struct Image {
    ip: u32,
    flags: u32,
    gpr: [u32; 8],
    segments: [u16; 6],
    ldt: u16,
    io_map_base: u16,
    debug_trap: u8,
}
#[derive(Clone, Copy)]
enum Stage {
    Save(usize),
    ClearBusy,
    ReturnBusy,
    Commit,
    ReadImage,
    Ldt,
    Segment(usize),
    Ip,
    ErrorCode,
    Complete,
}
struct Pending {
    owner: Owner,
    context: MachineContext,
    single_step: bool,
    cause: Cause,
    incoming: SystemSegment,
    stores: Vec<(u32, u8, u64)>,
    old_access: u32,
    error_code: Option<u32>,
    ext: u16,
    stage: Stage,
    committed: bool,
}

fn selector_fault(cpu: &CpuState, vector: u8, value: u16, ext: u16) -> Exit {
    fault(cpu, vector, (value & !3) | ext)
}

// SDM Vol 3A Tables 7-6/10-1 conflict on invalid/nonavailable TSSs.
// Keep 386 PRM §17 CALL/INT pp. 17-41/17-72's specific #TS rule there;
// JMP p. 17-86 uses #GP and IRET p. 17-76 uses #TS (decision-24).
fn transfer_vector(generation: CpuGeneration, cause: Cause) -> u8 {
    if cause == Cause::Return
        || (generation == CpuGeneration::I386 && matches!(cause, Cause::Call | Cause::Event))
    {
        10
    } else {
        13
    }
}

fn task_descriptor<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    value: u16,
    generation: CpuGeneration,
    vector: u8,
    ext: u16,
) -> Result<Descriptor, Exit> {
    if value & 4 != 0 {
        return Err(selector_fault(cpu, vector, value, ext));
    }
    task_segment_descriptor(cpu, mem, value, generation, vector, ext)
}
fn task_segment_descriptor<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    value: u16,
    generation: CpuGeneration,
    vector: u8,
    ext: u16,
) -> Result<Descriptor, Exit> {
    let address =
        descriptor_address(cpu, value).ok_or_else(|| selector_fault(cpu, vector, value, ext))?;
    descriptor_at(cpu, mem, address, generation)
}

fn is_tss(attr: u16, generation: CpuGeneration) -> bool {
    attr & 0x10 == 0
        && (matches!(attr & 15, 1 | 3)
            || (generation == CpuGeneration::I386 && matches!(attr & 15, 9 | 11)))
}
fn wide(tss: SystemSegment) -> bool {
    tss.cache.attributes & 8 != 0
}
fn minimum(tss: SystemSegment) -> u32 {
    if wide(tss) {
        0x67
    } else {
        0x2b
    }
}

/// None leaves a normal code transfer to the existing helper.
pub(super) fn transfer<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    context: MachineContext,
    value: u16,
    cause: Cause,
    saved_ip: u32,
    state: &mut SwitchState,
) -> Result<bool, Exit> {
    if state.pending.is_some() {
        resume(cpu, mem, state)?;
        return Ok(true);
    }
    let d = descriptor(cpu, mem, value, context.generation)?;
    let attr = d.cache.attributes;
    if context.generation == CpuGeneration::I286 && attr & 0x10 == 0 && matches!(attr & 15, 9 | 11)
    {
        return Err(selector_fault(cpu, 13, value, 0));
    }
    if attr & 0x10 != 0 || (!is_tss(attr, context.generation) && attr & 31 != 5) {
        return Ok(false);
    }
    if ((attr >> 5) & 3) < u16::from(context.cpl).max(value & 3) {
        // 386 PRM §9.8.13 p. 9-17 and SDM Vol 2A CALL p. 3-134:
        // task/gate DPL failures use #GP; CALL p. 17-41 says #TS.
        return Err(selector_fault(cpu, 13, value, 0));
    }
    if attr & 31 == 5 {
        if attr & 0x80 == 0 {
            return Err(selector_fault(cpu, 11, value, 0));
        }
        // 286 PRM §8.5 Fig. 8-3 pp. 8-8–9; 386 PRM §7.4 pp. 7-5–6:
        // selector is bytes 2:3; the target TSS DPL/RPL is not checked.
        let target = d.cache.base as u16;
        start(
            cpu, mem, context, target, cause, saved_ip, None, 0, false, state,
        )?;
    } else {
        start(
            cpu, mem, context, value, cause, saved_ip, None, 0, true, state,
        )?;
    }
    Ok(true)
}

pub(super) fn task_return<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    context: MachineContext,
    next: u32,
    state: &mut SwitchState,
) -> Result<(), Exit> {
    if state.pending.is_some() {
        return resume(cpu, mem, state);
    }
    // 286 PRM App. B IRET p. B-51; 386 PRM §17 IRET p. 17-76.
    let tr = cpu.legacy.tr;
    if tr.cache.unusable != 0 || tr.cache.limit < 1 {
        return Err(selector_fault(cpu, 10, tr.selector, 0));
    }
    let value = read(cpu, mem, u64::from(tr.cache.base), 2)? as u16;
    start(
        cpu,
        mem,
        context,
        value,
        Cause::Return,
        next,
        None,
        0,
        false,
        state,
    )
}

pub(super) fn event<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    context: MachineContext,
    value: u16,
    event: legacy::LegacyEvent,
    state: &mut SwitchState,
) -> Result<(), Exit> {
    if state.pending.is_some() {
        return resume(cpu, mem, state);
    }
    // 286 PRM App. B INT p. B-50; 386 PRM §9.7 p. 9-10:
    // processor instruction faults are internal to the program; INTR and
    // asynchronous coprocessor/debug events carry EXT.
    let external = event.external && (!event.fault || matches!(event.vector, 7 | 9 | 16));
    start(
        cpu,
        mem,
        context,
        value,
        Cause::Event,
        event.saved_ip,
        event.error_code,
        u16::from(external),
        false,
        state,
    )
}

#[allow(clippy::too_many_arguments)]
fn start<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    context: MachineContext,
    value: u16,
    cause: Cause,
    saved_ip: u32,
    error_code: Option<u32>,
    ext: u16,
    direct: bool,
    state: &mut SwitchState,
) -> Result<(), Exit> {
    state.committed = false;
    let vector = transfer_vector(context.generation, cause);
    let mut d = task_descriptor(cpu, mem, value, context.generation, vector, ext)?;
    loop {
        let attr = d.cache.attributes;
        if direct && ((attr >> 5) & 3) < u16::from(context.cpl).max(value & 3) {
            return Err(selector_fault(cpu, 13, value, ext));
        }
        if !is_tss(attr, context.generation) || (attr & 2 != 0) != (cause == Cause::Return) {
            return Err(selector_fault(cpu, vector, value, ext));
        }
        if attr & 0x80 == 0 {
            return Err(selector_fault(cpu, 11, value, ext));
        }
        let incoming = SystemSegment {
            selector: value,
            cache: d.cache,
            ..Default::default()
        };
        // 386 PRM §7.5 p. 7-8: the limit check belongs to the old task.
        // 286 PRM App. B p. B-12 checks it after changing TR instead.
        if context.generation == CpuGeneration::I386 && d.cache.limit < minimum(incoming) {
            return Err(selector_fault(cpu, 10, value, ext));
        }
        if context.paging_enabled {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyTaskPaging,
            ));
        }
        let old = cpu.legacy.tr;
        let old_minimum = if wide(old) { 0x5f } else { 0x29 };
        let invalid_old = old.selector & !3 == 0
            || old.cache.unusable != 0
            || !is_tss(old.cache.attributes, context.generation)
            || old.cache.limit < old_minimum;
        if invalid_old && context.generation != CpuGeneration::I286 {
            return Err(selector_fault(cpu, 10, old.selector, ext));
        }
        let old_access = cpu
            .legacy
            .gdtr
            .base
            .wrapping_add(u32::from(old.selector & !7))
            .wrapping_add(5);
        // Device descriptors require embedder emulation. Preflight before any
        // task mutation; ROM still accepts a locked update and discards its write.
        if cause != Cause::Return {
            probe_descriptor_cas(cpu, mem, d.access_address.unwrap())?;
        }
        if !invalid_old && matches!(cause, Cause::Jump | Cause::Return) {
            probe_descriptor_cas(cpu, mem, old_access)?;
        }
        let mut stores = save_image(cpu, old, saved_ip, cause, context.generation);
        if matches!(cause, Cause::Call | Cause::Event) {
            stores.push((incoming.cache.base, 2, u64::from(old.selector)));
        }
        if !invalid_old {
            for &(at, size, _) in &stores {
                probe(cpu, mem, u64::from(at), size)?;
            }
        }
        // Acquire with the same locked, A20-gated byte CAS as LTR. Retry the
        // whole descriptor validation if another CPU changes the access byte.
        if cause != Cause::Return {
            let address = d.access_address.unwrap();
            let expected = u64::from(attr as u8);
            if descriptor_cas(cpu, mem, address, expected, expected | 2)? != expected {
                d = task_descriptor(cpu, mem, value, context.generation, vector, ext)?;
                continue;
            }
        }
        // 286 PRM App. B p. B-12: the locked incoming B update precedes
        // validation of the current cache/29h limit. LTR p. B-72 has no limit check.
        if invalid_old {
            return Err(selector_fault(cpu, 10, value, ext));
        }
        let mut incoming = incoming;
        incoming.cache.attributes |= 2;
        state.pending = Some(Pending {
            owner: Owner::new(cpu),
            context,
            single_step: cpu.rflags_system & 0x100 != 0,
            cause,
            incoming,
            stores,
            old_access,
            error_code,
            ext,
            stage: Stage::Save(0),
            committed: false,
        });
        return resume(cpu, mem, state);
    }
}

fn probe_descriptor_cas<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    address: u32,
) -> Result<(), Exit> {
    let linear = u64::from(address);
    let (physical, _) = legacy::physical_span(cpu, linear, 1);
    match mem.0.region_kind(physical, 1) {
        Ok(crate::RegionKind::Ram | crate::RegionKind::Rom) => mem.probe_write(physical, 1),
        Ok(_) => {
            return Err(refusal(
                cpu,
                crate::UnsupportedSystemOperation::LegacyInstruction,
            ))
        }
        Err(trap) => Err(trap),
    }
    .map_err(|trap| {
        match trap_out(cpu, cpu.rip, trap, physical, 1, AccessKind::Write, 0).into_step() {
            StepResult::Exit(exit) => crate::fault::qualify_legacy_bus(exit, linear, physical),
            _ => unreachable!(),
        }
    })
}

fn load_image<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    tss: SystemSegment,
) -> Result<Image, Exit> {
    let base = tss.cache.base;
    let is_wide = wide(tss);
    let mut field = |offset: u32, size| {
        read(cpu, mem, u64::from(base.wrapping_add(offset)), size).map(|v| v as u32)
    };
    let size = if is_wide { 4 } else { 2 };
    let ip = field(if is_wide { 0x20 } else { 0x0e }, size)?;
    let flags = field(if is_wide { 0x24 } else { 0x10 }, size)?;
    let mut gpr = [0; 8];
    for (i, reg) in gpr.iter_mut().enumerate() {
        *reg = field(
            if is_wide { 0x28 } else { 0x12 } + i as u32 * u32::from(size),
            size,
        )?;
    }
    let mut segments = [0; 6];
    for (i, index) in [2, 0, 3, 1, 4, 5]
        .into_iter()
        .enumerate()
        .take(if is_wide { 6 } else { 4 })
    {
        segments[index] = field(
            if is_wide {
                0x48 + i as u32 * 4
            } else {
                0x22 + i as u32 * 2
            },
            2,
        )? as u16;
    }
    let ldt = field(if is_wide { 0x60 } else { 0x2a }, 2)? as u16;
    let (debug_trap, io_map_base) = if is_wide {
        // 386 PRM §7.1 Fig. 7-1 pp. 7-2–3; CR3 is read only with PG=1.
        ((field(0x64, 2)? & 1) as u8, field(0x66, 2)? as u16)
    } else {
        (0, 0)
    };
    Ok(Image {
        ip,
        flags,
        gpr,
        segments,
        ldt,
        debug_trap,
        io_map_base,
    })
}

fn save_image(
    cpu: &CpuState,
    tss: SystemSegment,
    ip: u32,
    cause: Cause,
    generation: CpuGeneration,
) -> Vec<(u32, u8, u64)> {
    // 286 PRM §8.2 Fig. 8-1 pp. 8-2–3; 386 PRM §7.1 pp. 7-1–3:
    // LDTR, CR3, privilege stacks, T and I/O-map base are static, never saved.
    let is_wide = wide(tss);
    let size = if is_wide { 4 } else { 2 };
    let mut flags = legacy::flags_image(cpu, generation);
    if cause == Cause::Return {
        flags &= !0x4000;
    }
    let mut fields = vec![
        (if is_wide { 0x20 } else { 0x0e }, size, u64::from(ip)),
        (if is_wide { 0x24 } else { 0x10 }, size, u64::from(flags)),
    ];
    for i in 0..8 {
        fields.push((
            if is_wide { 0x28 } else { 0x12 } + i as u32 * u32::from(size),
            size,
            cpu.gpr[i],
        ));
    }
    for (i, segment) in [
        SegmentId::Es,
        SegmentId::Cs,
        SegmentId::Ss,
        SegmentId::Ds,
        SegmentId::Fs,
        SegmentId::Gs,
    ]
    .into_iter()
    .enumerate()
    .take(if is_wide { 6 } else { 4 })
    {
        fields.push((
            if is_wide {
                0x48 + i as u32 * 4
            } else {
                0x22 + i as u32 * 2
            },
            2,
            u64::from(selector(cpu, segment)),
        ));
    }
    fields
        .into_iter()
        .map(|(offset, size, value)| (tss.cache.base.wrapping_add(offset), size, value))
        .collect()
}

/// A descriptor-load fault needs the new task's sizes even while some caches are
/// invalid. Normal entry validation deliberately rejects such intermediate state.
pub(crate) fn fault_context(cpu: &CpuState, generation: CpuGeneration) -> MachineContext {
    MachineContext {
        // A task switch loads protected-mode state.
        segment_class: crate::SegmentClass::Checked,
        generation,
        regime: legacy::MachineRegime::Protected,
        default_operand_bits: if cpu.legacy.segments[0].attributes & 0x4000 != 0 {
            32
        } else {
            16
        },
        default_address_bits: if cpu.legacy.segments[0].attributes & 0x4000 != 0 {
            32
        } else {
            16
        },
        stack_address_bits: if cpu.legacy.segments[3].attributes & 0x4000 != 0 {
            32
        } else {
            16
        },
        cpl: cpu.legacy.cpl,
        paging_enabled: cpu.legacy.cr0 & 0x8000_0000 != 0,
    }
}

pub(crate) fn resume<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    state: &mut SwitchState,
) -> Result<(), Exit> {
    let mut pending = state.pending.take().expect("task switch continuation");
    let result = pending.run(cpu, mem);
    state.committed = pending.committed;
    state.completed = result.is_ok();
    if result.as_ref().is_err_and(|e| {
        matches!(
            e,
            Exit::MmioRead { .. } | Exit::MmioWrite { .. } | Exit::UnmappedMemory { .. }
        ) || matches!(e, Exit::UnsupportedSystemContext { .. })
    }) {
        pending.owner = Owner::new(cpu);
        state.pending = Some(pending);
    }
    // A T-bit trap belongs after a completed switch (386 PRM §12.3.1.5
    // p. 12-8). There is no unfinished switch to replay or retire again.
    result?;
    if cpu.task_debug_trap != 0 {
        return Err(refusal(
            cpu,
            crate::UnsupportedSystemOperation::LegacyTaskDebugTrap,
        ));
    }
    Ok(())
}
impl Pending {
    fn clear_busy<const ROUTED: bool, const ORDER: u8>(
        &self,
        cpu: &mut CpuState,
        mem: &Access<ROUTED, ORDER>,
    ) -> Result<(), Exit> {
        let mut expected = read(cpu, mem, u64::from(self.old_access), 1)?;
        loop {
            let observed = descriptor_cas(cpu, mem, self.old_access, expected, expected & !2)?;
            if observed == expected {
                return Ok(());
            }
            expected = observed;
        }
    }
    fn run<const ROUTED: bool, const ORDER: u8>(
        &mut self,
        cpu: &mut CpuState,
        mem: &Access<ROUTED, ORDER>,
    ) -> Result<(), Exit> {
        loop {
            match self.stage {
                Stage::Save(i) => {
                    if let Some(&(at, size, value)) = self.stores.get(i) {
                        write(cpu, mem, u64::from(at), size, value)?;
                        self.stage = Stage::Save(i + 1);
                    } else {
                        self.stage = Stage::ClearBusy;
                    }
                }
                Stage::ClearBusy => {
                    if self.cause == Cause::Jump {
                        self.clear_busy(cpu, mem)?;
                    }
                    self.stage = Stage::Commit;
                }
                Stage::ReturnBusy => {
                    self.clear_busy(cpu, mem)?;
                    self.stage = Stage::Ip;
                }
                Stage::Commit => {
                    cpu.legacy.tr = self.incoming;
                    self.committed = true;
                    if self.context.generation == CpuGeneration::I386 {
                        cpu.legacy.cr0 |= 8;
                    }
                    // 286 PRM App. B p. B-12: TR changes before this limit
                    // check; TS is set only after the incoming registers load.
                    if self.incoming.cache.limit < minimum(self.incoming) {
                        return Err(selector_fault(cpu, 10, self.incoming.selector, self.ext));
                    }
                    self.stage = Stage::ReadImage;
                }
                Stage::ReadImage => {
                    // 286 PRM App. B pp. B-12–13; 386 PRM §7.5 steps 3–5
                    // p. 7-8: acquire/save first, then read the incoming state.
                    let image = load_image(cpu, mem, self.incoming)?;
                    cpu.clear_mmio_parts();
                    cpu.rip = u64::from(image.ip);
                    for i in 0..8 {
                        cpu.write_gpr(i, u64::from(image.gpr[i]), 4);
                    }
                    legacy::restore_flags(cpu, self.context.generation, image.flags)?;
                    cpu.rflags_system =
                        (cpu.rflags_system & !0x30000) | u64::from(image.flags & 0x30000);
                    // 286 PRM App. B SWITCH_TASKS p. B-12; 386 PRM §7.5
                    // step 5 p. 7-8; SDM Vol 3A §10.3 step 12 p. 10-11 /
                    // §10.4 Table 10-2 p. 10-15: JMP/IRET load incoming NT.
                    match self.cause {
                        Cause::Call | Cause::Event => cpu.rflags_system |= 0x4000,
                        Cause::Jump | Cause::Return => {}
                    }
                    [cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs] = image.segments;
                    for cache in &mut cpu.legacy.segments {
                        cache.unusable = 1;
                    }
                    cpu.legacy.cpl = (cpu.cs & 3) as u8;
                    cpu.legacy.ldtr = SystemSegment {
                        selector: image.ldt,
                        cache: legacy::SegmentCache {
                            unusable: 1,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    cpu.task_io_map_base = image.io_map_base;
                    cpu.task_debug_trap = image.debug_trap;
                    cpu.legacy.cr0 |= 8;
                    if image.flags & (1 << 17) != 0 {
                        return Err(refusal(
                            cpu,
                            crate::UnsupportedSystemOperation::InvalidLegacyContext(
                                legacy::MachineContextError::Virtual8086Unsupported,
                            ),
                        ));
                    }
                    if cpu.legacy.cpl != 0 {
                        return Err(refusal(
                            cpu,
                            crate::UnsupportedSystemOperation::LegacyPrivilegeTransition,
                        ));
                    }
                    self.stage = Stage::Ldt;
                }
                Stage::Ldt => {
                    let value = cpu.legacy.ldtr.selector;
                    if value & !3 != 0 {
                        // 386 PRM §9.8.10 Table 9-5 p. 9-14 and SDM Vol 3A
                        // Table 10-1 p. 10-14 name LDT, unlike PRM Table 7-1.
                        let mut d = task_descriptor(
                            cpu,
                            mem,
                            value,
                            self.context.generation,
                            10,
                            self.ext,
                        )?;
                        if d.cache.attributes & 0x9f != 0x82 {
                            return Err(selector_fault(cpu, 10, value, self.ext));
                        }
                        supported_descriptor(cpu, d.cache.attributes)?;
                        d.cache.unusable = 0;
                        cpu.legacy.ldtr.cache = d.cache;
                    }
                    self.stage = Stage::Segment(0);
                }
                Stage::Segment(i) => {
                    // 286 PRM App. B p. B-13 explicitly loads SS before CS.
                    // 386 PRM Table 7-1 p. 7-9 lists CS before SS.
                    let order = if self.context.generation == CpuGeneration::I286 {
                        [
                            SegmentId::Ss,
                            SegmentId::Cs,
                            SegmentId::Ds,
                            SegmentId::Es,
                            SegmentId::Fs,
                            SegmentId::Gs,
                        ]
                    } else {
                        [
                            SegmentId::Cs,
                            SegmentId::Ss,
                            SegmentId::Ds,
                            SegmentId::Es,
                            SegmentId::Fs,
                            SegmentId::Gs,
                        ]
                    };
                    if i == 6 {
                        self.stage = if self.cause == Cause::Event {
                            Stage::ErrorCode
                        } else if self.cause == Cause::Return {
                            // 286 IRET p. B-51 / 386 IRET p. 17-76 mark the
                            // abandoned task after SWITCH_TASKS. This includes
                            // descriptor loads (286 B-13 / 386 §7.5 p. 7-8).
                            // SDM Vol 3A §10.3 steps 9–10 p. 10-11 also reads
                            // the incoming image before clearing old B.
                            Stage::ReturnBusy
                        } else {
                            Stage::Ip
                        };
                    } else {
                        load_segment(cpu, mem, self.context.generation, order[i], self.ext)?;
                        self.stage = Stage::Segment(i + 1);
                    }
                }
                Stage::Ip => {
                    if cpu.rip > u64::from(cpu.legacy.segments[0].limit) {
                        return Err(fault(cpu, 13, self.ext));
                    }
                    self.stage = Stage::Complete;
                }
                Stage::ErrorCode => {
                    if let Some(code) = self.error_code {
                        // 286 PRM App. B INT p. B-50; 386 PRM §17 INT
                        // p. 17-72: push the error code before checking new IP.
                        // Width follows the destination task (SDM Vol 3A §7.12.2).
                        let context = fault_context(cpu, self.context.generation);
                        push(
                            cpu,
                            mem,
                            context,
                            &[u64::from(code)],
                            if wide(self.incoming) { 4 } else { 2 },
                        )?;
                    }
                    self.stage = Stage::Ip;
                }
                Stage::Complete => return Ok(()),
            }
        }
    }
}

fn load_segment<const ROUTED: bool, const ORDER: u8>(
    cpu: &mut CpuState,
    mem: &Access<ROUTED, ORDER>,
    generation: CpuGeneration,
    segment: SegmentId,
    ext: u16,
) -> Result<(), Exit> {
    let value = selector(cpu, segment);
    let data = !matches!(segment, SegmentId::Cs | SegmentId::Ss);
    // 286 PRM App. B p. B-13; 386 PRM §9.8.10 Table 9-5 p. 9-14:
    // selector/type/privilege failures use #TS, matching SDM Table 10-1.
    let invalid = 10;
    if value & !3 == 0 {
        if !data {
            return Err(selector_fault(cpu, invalid, value, ext));
        }
        cpu.legacy.segments[segment as usize] = legacy::SegmentCache {
            unusable: 1,
            ..Default::default()
        };
        return Ok(());
    }
    let mut d = task_segment_descriptor(cpu, mem, value, generation, invalid, ext)?;
    let attr = d.cache.attributes;
    let code = attr & 8 != 0;
    let dpl = ((attr >> 5) & 3) as u8;
    let cpl = cpu.legacy.cpl;
    let valid_type = attr & 0x10 != 0
        && match segment {
            SegmentId::Cs => code,
            SegmentId::Ss => !code && attr & 2 != 0,
            _ => !code || attr & 2 != 0,
        };
    if !valid_type {
        return Err(selector_fault(cpu, 10, value, ext));
    }
    let presence = |cpu: &CpuState| {
        if attr & 0x80 == 0 {
            Err(selector_fault(
                cpu,
                if segment == SegmentId::Ss { 12 } else { 11 },
                value,
                ext,
            ))
        } else {
            Ok(())
        }
    };
    // 386 Table 7-1 tests 7/10/15 precede the privilege tests 8/11–12/16.
    // 286 SWITCH_TASKS p. B-13 checks privilege before presence.
    if generation == CpuGeneration::I386 {
        presence(cpu)?;
    }
    let valid_privilege = match segment {
        SegmentId::Cs => {
            if attr & 4 != 0 {
                dpl <= cpl
            } else {
                dpl == cpl
            }
        }
        SegmentId::Ss => dpl == cpl && value & 3 == u16::from(cpl),
        _ => (code && attr & 4 != 0) || dpl >= cpl.max((value & 3) as u8),
    };
    if !valid_privilege {
        return Err(selector_fault(cpu, 10, value, ext));
    }
    presence(cpu)?;
    supported_descriptor(cpu, attr)?;
    accessed(cpu, mem, &mut d)?;
    install(cpu, segment, value, d.cache);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Prot, RegionKind, Vm, VmConfig};

    fn setup() -> (Vm, CpuState, MachineContext) {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        let mut cpu = CpuState::legacy_real_entry([0; 6]);
        cpu.legacy.cr0 = 1;
        cpu.legacy.gdtr = legacy::TableRegister {
            base: 0x2000,
            limit: 0x7f,
            ..Default::default()
        };
        cpu.legacy.tr = SystemSegment {
            selector: 24,
            cache: legacy::SegmentCache {
                base: 0x4000,
                limit: 0x2b,
                attributes: 0x83,
                ..Default::default()
            },
            ..Default::default()
        };
        vm.mem.write(0x2018, 0x83004000002b, 8).unwrap();
        vm.mem.write(0x2020, 0xe1004100002b, 8).unwrap();
        vm.mem.write(0x410e, 0x1800, 2).unwrap();
        let context = fault_context(&cpu, CpuGeneration::I286);
        (vm, cpu, context)
    }

    #[test]
    fn task_owner_rejects_edits_outside_the_requested_completion() {
        let cpu = CpuState::default();
        let owner = Owner::new(&cpu);
        macro_rules! rejects {
            ($($edit:expr),* $(,)?) => {$({
                let mut changed = cpu.clone();
                let edit: fn(&mut CpuState) = $edit;
                edit(&mut changed);
                assert!(!owner.matches(&changed), "{}", stringify!($edit));
            })*};
        }
        rejects!(
            |c| c.gpr[15] = 1,
            |c| c.rip = 1,
            |c| c.fs_base = 1,
            |c| c.gs_base = 1,
            |c| c.flags.cf = !c.flags.cf,
            |c| c.flags.pf_src ^= 3, // same parity still counts as a public edit
            |c| c.flags.af_src ^= 1,
            |c| c.flags.if_ = !c.flags.if_,
            |c| c.xmm[31] = 1,
            |c| c.ymm_hi[31] = 1,
            |c| c.zmm_hi[31][1] = 1,
            |c| c.kmask[7] = 1,
            |c| c.fpr[7][9] = 1,
            |c| c.fpu_top ^= 1,
            |c| c.fpu_cw ^= 1,
            |c| c.fpu_pad ^= 1,
            |c| c.fpu_empty ^= 1,
            |c| c.fpu_sw ^= 1,
            |c| c.fpu_fip ^= 1,
            |c| c.fpu_fdp ^= 1,
            |c| c.fpu_fop ^= 1,
            |c| c.fpu_fcs ^= 1,
            |c| c.fpu_fds ^= 1,
            |c| c.cs ^= 1,
            |c| c.ds ^= 1,
            |c| c.es ^= 1,
            |c| c.ss ^= 1,
            |c| c.fs ^= 1,
            |c| c.gs ^= 1,
            |c| c.source_id ^= 1,
            |c| c.random_state ^= 1,
            |c| c.rflags_system ^= 0x10000,
            |c| c.mxcsr ^= 1,
            |c| c.legacy.cr2 ^= 1,
            |c| c.task_io_map_base ^= 1,
            |c| c.task_debug_trap ^= 1,
            |c| c.pending_port_in = Some(1),
            |c| c.mmio_parts[63] = (1, 2),
            |c| c.mmio_parts_len = 1,
            |c| c.mmio_parts_rip = 1,
            |c| c.mmio_retry_active = true,
            |c| c.mmio_retry_cs = 1,
            |c| c.pending_mmio = Some(1),
            |c| c.pending_mmio_write = true,
            |c| c.features = crate::features::GuestCpuFeatures::empty(),
            |c| c.cpuid_overrides.xcr0 = Some(1),
            |c| c.fpu_emulation = !c.fpu_emulation,
            |c| c.fpu_model = crate::state::FpuModel::None,
            |c| c.x87_precision = crate::state::X87Precision::Extended,
            |c| c.x87_pointer_model = crate::state::X87PointerModel::Amd,
        );
        let mut read_owner = Owner::new(&cpu);
        read_owner.completion = Completion::Read;
        let mut supplied = cpu.clone();
        supplied.pending_mmio = Some(0x1234);
        assert!(read_owner.matches(&supplied));
        supplied.pending_mmio_write = true;
        assert!(!read_owner.matches(&supplied));
        let mut write_owner = Owner::new(&cpu);
        write_owner.completion = Completion::Write;
        supplied.pending_mmio = None;
        assert!(write_owner.matches(&supplied));
        supplied.pending_mmio = Some(0x1234);
        assert!(!write_owner.matches(&supplied));
    }

    #[test]
    fn task_acquire_cas_revalidates_competing_privilege_and_busy_edits() {
        // 286 PRM §8.4 p. 8-7; 386 PRM §7.6.1 p. 7-10: locked B
        // acquisition, with the same byte CAS and revalidation as LTR.
        for attr in [0x81, 0xe3] {
            let (vm, mut cpu, context) = setup();
            let before = cpu.legacy;
            super::super::DESCRIPTOR_UPDATE.with(|update| {
                update.set(Some((0x2020, (u64::from(attr as u8) << 40) | 0x4100002b)))
            });
            let exit = transfer(
                &mut cpu,
                Access::<false, 0>::new(&vm.mem),
                context,
                35,
                Cause::Jump,
                0x1005,
                &mut SwitchState::default(),
            )
            .unwrap_err();
            assert!(matches!(exit, Exit::Exception { vector: 13, .. }));
            assert_eq!(
                exit.fault_metadata().unwrap().error_code(),
                crate::fault::ErrorCode::bits(32, 0xffff)
            );
            assert_eq!(cpu.legacy, before);
            assert_eq!(vm.mem.read(0x2025, 1).unwrap(), u64::from(attr as u8));
            assert_eq!(vm.mem.read(0x400e, 2).unwrap(), 0);
        }
    }

    #[test]
    fn task_paging_reload_is_refused_before_saves_or_busy_updates() {
        let (vm, mut cpu, mut context) = setup();
        context.generation = CpuGeneration::I386;
        context.paging_enabled = true;
        cpu.legacy.cr0 |= 0x8000_0000;
        let exit = transfer(
            &mut cpu,
            Access::<false, 0>::new(&vm.mem),
            context,
            35,
            Cause::Jump,
            0x1005,
            &mut SwitchState::default(),
        )
        .unwrap_err();
        assert!(matches!(
            exit,
            Exit::UnsupportedSystemContext {
                operation: crate::UnsupportedSystemOperation::LegacyTaskPaging,
                ..
            }
        ));
        assert_eq!(vm.mem.read(0x2025, 1).unwrap(), 0xe1);
        assert_eq!(vm.mem.read(0x400e, 2).unwrap(), 0);
    }
    #[test]
    fn interrupt_task_pushes_error_before_checking_new_ip_limit() {
        for generation in [CpuGeneration::I286, CpuGeneration::I386] {
            let (vm, mut cpu, mut context) = setup();
            context.generation = generation;
            vm.mem.write(0x2008, 0x9b00000017ff, 8).unwrap();
            vm.mem.write(0x2010, 0x93000000ffff, 8).unwrap();
            for (offset, value) in [
                (0x10, 2),
                (0x1a, 0x9000),
                (0x22, 16),
                (0x24, 8),
                (0x26, 16),
                (0x28, 16),
            ] {
                vm.mem.write(0x4100 + offset, value, 2).unwrap();
            }
            let mut state = SwitchState::default();
            let exit = event(
                &mut cpu,
                Access::<false, 0>::new(&vm.mem),
                context,
                32,
                legacy::LegacyEvent {
                    vector: 13,
                    saved_ip: 0x1000,
                    software: false,
                    external: false,
                    error_code: Some(64),
                    fault: true,
                },
                &mut state,
            )
            .unwrap_err();
            assert!(matches!(exit, Exit::Exception { vector: 13, .. }));
            assert!(state.committed);
            assert_eq!(
                (cpu.legacy.tr.selector, cpu.rip, cpu.gpr[4]),
                (32, 0x1800, 0x8ffe)
            );
            assert_eq!(vm.mem.read(0x8ffe, 2).unwrap(), 64);
        }
    }
    #[test]
    fn review_incoming_image_is_read_after_successful_busy_cas() {
        for (generation, wide) in [
            (CpuGeneration::I286, false),
            (CpuGeneration::I386, false),
            (CpuGeneration::I386, true),
        ] {
            let (vm, mut cpu, mut context) = setup();
            context.generation = generation;
            vm.mem.write(0x2008, 0x9b000000ffff, 8).unwrap();
            vm.mem.write(0x2010, 0x93000000ffff, 8).unwrap();
            vm.mem
                .write(
                    0x2020,
                    if wide { 0xe90041000067 } else { 0xe1004100002b },
                    8,
                )
                .unwrap();
            for (offset, value) in if wide {
                vec![
                    (0x20, 0x1800),
                    (0x24, 2),
                    (0x38, 0x9000),
                    (0x48, 16),
                    (0x4c, 8),
                    (0x50, 16),
                    (0x54, 16),
                ]
            } else {
                vec![
                    (0x0e, 0x1800),
                    (0x10, 2),
                    (0x1a, 0x9000),
                    (0x22, 16),
                    (0x24, 8),
                    (0x26, 16),
                    (0x28, 16),
                ]
            } {
                vm.mem
                    .write(
                        0x4100 + offset,
                        value,
                        if wide && offset < 0x48 { 4 } else { 2 },
                    )
                    .unwrap();
            }
            super::super::DESCRIPTOR_UPDATE
                .with(|u| u.set(Some((0x4100 + if wide { 0x28 } else { 0x12 }, 0x7654))));
            transfer(
                &mut cpu,
                Access::<false, 0>::new(&vm.mem),
                context,
                35,
                Cause::Jump,
                0x1005,
                &mut SwitchState::default(),
            )
            .unwrap();
            assert_eq!(cpu.gpr[0], 0x7654);
        }
    }
}
