//! String instructions with a 16-bit address size in protected mode
//! (xj-srh.45): MOVS, STOS, LODS, CMPS, SCAS, INS and OUTS, with and
//! without REP/REPE/REPNE, at 8/16/32-bit operand sizes, in 16-bit code and
//! under 67h in 32-bit code, at CPL 0 and 3, paged and unpaged, through the
//! Segmented, Checked and Flat32 execution classes. Every case runs twice
//! through public Machine dispatch, once a step at a time and once in a
//! single run, and is compared with a reference model of the manual:
//!
//! - SDM Vol 2B §4.3 REP/REPE/REPZ/REPNE/REPNZ, Description and Operation
//!   pp. 4-563–4-565: the address-size attribute selects CX or ECX as the
//!   count; a zero count performs no iteration and no memory access; a
//!   repeated string is interruptible with its registers at the boundary of
//!   the element it stopped at, and resumes from there.
//! - MOVS (Vol 2B §4.3 pp. 4-104–4-105), STOS (pp. 4-676–4-677), SCAS
//!   (pp. 4-615–4-617), LODS (Vol 2A §3.3 pp. 3-567–3-569), CMPS (Vol 2A
//!   §3.3 pp. 3-181–3-183), INS (Vol 2A §3.3 pp. 3-464–3-465) and OUTS
//!   (Vol 2B §4.3 pp. 4-173–4-175), Description and Operation: with a 16-bit
//!   address size the operands are DS:SI (segment-overridable) and ES:DI, SI
//!   and DI step by the operand size under DF, and only the low words of
//!   ESI, EDI and ECX change. Protected Mode Exceptions: #GP(0) for an
//!   effective address outside the DS, ES, FS or GS limit, #SS(0) outside
//!   the SS limit.
//! - SDM Vol 3A §6.3 Limit Checking pp. 6-4–6-5: whole-operand checks
//!   against the effective limit (G scales it); expand-down segments are
//!   valid from limit+1 to FFFFh (B clear) or FFFFFFFFh (B set).
//!
//! The 16-bit effective offset wraps at 64 KiB between elements (SI and DI
//! are 16-bit registers); one element is checked and transferred at
//! `base + offset` over its whole width.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::descriptor;
use std::collections::HashMap;
use x86jit_core::{
    derive_legacy_context, CpuModel, ExceptionDelivery, Exit, PortDir, Prot, RegionKind,
    SegmentCache, SegmentClass, StepOptions, StepStop, SystemSegment, Vcpu, Vm, VmConfig,
};

const MEMORY: u64 = 0x10_0000;
const PD: u32 = 0xf_0000;
const PT: u32 = 0xf_1000;
const GDT: u32 = 0xf_2000;
const LDT: u32 = 0xe_0000;
/// The reported instruction pointer.
const IP: u64 = 0x798e;
const CS_BASE: u32 = 0x1_0000;
// An odd DS base, and an ES base that places word and doubleword elements
// at offsets 347Fh+ across 4-KiB page boundaries.
const DS_BASE: u32 = 0x4_0335;
const ES_BASE: u32 = 0x6_0b70;
const SS_BASE: u32 = 0x8_0ffe;
const PORT: u16 = 0x3ce;

const DS: usize = 1;
const ES: usize = 2;
const SS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Movs,
    Stos,
    Lods,
    Cmps,
    Scas,
    Ins,
    Outs,
}
impl Op {
    fn reads_source(self) -> bool {
        matches!(self, Op::Movs | Op::Lods | Op::Cmps | Op::Outs)
    }
    fn uses_destination(self) -> bool {
        matches!(self, Op::Movs | Op::Stos | Op::Cmps | Op::Scas | Op::Ins)
    }
    fn compares(self) -> bool {
        matches!(self, Op::Cmps | Op::Scas)
    }
    fn port(self) -> bool {
        matches!(self, Op::Ins | Op::Outs)
    }
    fn opcode(self, elem: u8) -> u8 {
        let byte = match self {
            Op::Movs => 0xa4,
            Op::Stos => 0xaa,
            Op::Lods => 0xac,
            Op::Cmps => 0xa6,
            Op::Scas => 0xae,
            Op::Ins => 0x6c,
            Op::Outs => 0x6e,
        };
        byte + u8::from(elem != 1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Rep {
    None,
    /// F3: REP, or REPE for CMPS/SCAS.
    F3,
    /// F2: REPNE (CMPS/SCAS only).
    F2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Class {
    Segmented,
    Checked,
    Flat32,
}

#[derive(Clone, Copy, Debug)]
struct Context {
    model: CpuModel,
    cpl: u8,
    paged: bool,
    class: Class,
    code32: bool,
}

/// A segment cache: base, effective (byte) limit and the descriptor's
/// attribute word (type, S, DPL, P; B at bit 14, G at bit 15).
#[derive(Clone, Copy, Debug)]
struct Seg {
    base: u32,
    limit: u32,
    attr: u16,
}
impl Seg {
    /// SDM Vol 3A §6.3 pp. 6-4–6-5 (whole-operand limit) and §6.4 (type).
    fn admits(self, offset: u16, bytes: u8, write: bool) -> bool {
        let first = u64::from(offset);
        let last = first + u64::from(bytes) - 1;
        let code = self.attr & 8 != 0;
        let rights = if write {
            !code && self.attr & 2 != 0
        } else {
            !code || self.attr & 2 != 0
        };
        let within = if !code && self.attr & 4 != 0 {
            let upper = if self.attr & 0x4000 != 0 {
                u64::from(u32::MAX)
            } else {
                0xffff
            };
            first > u64::from(self.limit) && last <= upper
        } else {
            last <= u64::from(self.limit)
        };
        rights && within
    }
}

#[derive(Clone, Debug)]
struct Case {
    op: Op,
    elem: u8,
    rep: Rep,
    df: bool,
    ss_source: bool,
    /// A NOP before the string, so that the string is not its block's first
    /// instruction in the single run.
    lead: bool,
    gpr: [u64; 8],
    segs: [Seg; 6],
    name: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
struct State {
    gpr: [u64; 8],
    rip: u64,
}

#[derive(Debug, PartialEq)]
enum End {
    Done,
    Fault(u8),
}

struct Model {
    /// `states[k]`: the registers after k completed elements.
    states: Vec<State>,
    end: End,
    /// The completed elements' linear byte writes.
    writes: HashMap<u32, u8>,
    /// Port values: OUTS outputs or INS inputs, in element order.
    ports: Vec<u64>,
    zf: bool,
    cf: bool,
}

fn mask(bytes: u8) -> u64 {
    u64::MAX >> (64 - 8 * u32::from(bytes))
}

fn set_word(gpr: &mut [u64; 8], index: usize, value: u64) {
    gpr[index] = (gpr[index] & !0xffff) | (value & 0xffff);
}

fn set_low(gpr: &mut [u64; 8], index: usize, value: u64, bytes: u8) {
    let m = mask(bytes);
    gpr[index] = (gpr[index] & !m) | (value & m);
}

/// INS input of element `k`.
fn input(k: usize, elem: u8) -> u64 {
    (0x1357_9bdf_u64.wrapping_mul(k as u64 + 3)) & mask(elem)
}

fn run_model(case: &Case, start: u64, next: u64, read: impl Fn(u32) -> u8) -> Model {
    let mut gpr = case.gpr;
    let mut writes = HashMap::new();
    let mut ports = Vec::new();
    let mut states = vec![State { gpr, rip: start }];
    let (mut zf, mut cf) = (false, false);
    let source = if case.ss_source { SS } else { DS };
    let source_vector = if case.ss_source { 12 } else { 13 };
    let elem = case.elem;
    let repeated = case.rep != Rep::None;
    let delta = if case.df {
        0u64.wrapping_sub(u64::from(elem))
    } else {
        u64::from(elem)
    };
    let load = |writes: &HashMap<u32, u8>, linear: u32| -> u64 {
        (0..elem).fold(0, |v, i| {
            let at = linear.wrapping_add(u32::from(i));
            v | u64::from(writes.get(&at).copied().unwrap_or_else(|| read(at))) << (8 * i)
        })
    };
    let store = |writes: &mut HashMap<u32, u8>, linear: u32, value: u64| {
        for i in 0..elem {
            writes.insert(linear.wrapping_add(u32::from(i)), (value >> (8 * i)) as u8);
        }
    };
    if repeated && gpr[1] & 0xffff == 0 {
        states.push(State { gpr, rip: next });
        return Model {
            states,
            end: End::Done,
            writes,
            ports,
            zf,
            cf,
        };
    }
    let mut k = 0;
    let end = loop {
        let si = gpr[6] as u16;
        let di = gpr[7] as u16;
        let src = if case.op.reads_source() {
            let seg = case.segs[source];
            if !seg.admits(si, elem, false) {
                break End::Fault(source_vector);
            }
            Some(load(&writes, seg.base.wrapping_add(u32::from(si))))
        } else {
            None
        };
        let dst = if case.op.uses_destination() {
            let seg = case.segs[ES];
            let write = matches!(case.op, Op::Movs | Op::Stos | Op::Ins);
            if !seg.admits(di, elem, write) {
                break End::Fault(13);
            }
            Some(seg.base.wrapping_add(u32::from(di)))
        } else {
            None
        };
        match case.op {
            Op::Movs => store(&mut writes, dst.unwrap(), src.unwrap()),
            Op::Stos => store(&mut writes, dst.unwrap(), gpr[0] & mask(elem)),
            Op::Ins => {
                let value = input(k, elem);
                ports.push(value);
                store(&mut writes, dst.unwrap(), value);
            }
            Op::Lods => set_low(&mut gpr, 0, src.unwrap(), elem),
            Op::Outs => ports.push(src.unwrap()),
            Op::Cmps | Op::Scas => {
                let lhs = src.unwrap_or(gpr[0] & mask(elem));
                let rhs = load(&writes, dst.unwrap());
                zf = lhs == rhs;
                cf = lhs < rhs;
            }
        }
        if src.is_some() {
            set_word(&mut gpr, 6, u64::from(si).wrapping_add(delta));
        }
        if dst.is_some() {
            set_word(&mut gpr, 7, u64::from(di).wrapping_add(delta));
        }
        k += 1;
        if !repeated {
            states.push(State { gpr, rip: next });
            break End::Done;
        }
        let count = gpr[1].wrapping_sub(1);
        set_word(&mut gpr, 1, count);
        let more = gpr[1] & 0xffff != 0
            && match (case.op.compares(), case.rep) {
                (true, Rep::F3) => zf,
                (true, Rep::F2) => !zf,
                _ => true,
            };
        states.push(State {
            gpr,
            rip: if more { start } else { next },
        });
        if !more {
            break End::Done;
        }
    };
    Model {
        states,
        end,
        writes,
        ports,
        zf,
        cf,
    }
}

fn flat(attr: u16) -> Seg {
    Seg {
        base: 0,
        limit: u32::MAX,
        attr: attr | 0xc000,
    }
}

fn segments(context: Context) -> [Seg; 6] {
    let dpl = u16::from(context.cpl) << 5;
    if context.class == Class::Flat32 {
        let data = flat(0x93 | dpl);
        return [flat(0x9b | dpl), data, data, data, data, data];
    }
    let seg = |base, attr: u16| Seg {
        base,
        limit: 0xffff,
        attr: attr | dpl,
    };
    let data = seg(DS_BASE, 0x93);
    [
        seg(CS_BASE, 0x9b | if context.code32 { 0x4000 } else { 0 }),
        data,
        seg(ES_BASE, 0x93),
        seg(SS_BASE, 0x93),
        data,
        data,
    ]
}

struct Machine {
    vm: Vm,
    context: Context,
    /// The DS and ES windows are a device (MMIO) region; `device` holds its
    /// bytes, which the embedder side of `drive` serves.
    devices: bool,
    device: std::cell::RefCell<HashMap<u32, u8>>,
}

/// Whether `linear` lies in a device page when the windows are devices.
fn device_page(linear: u32) -> bool {
    let page = linear & !0xfff;
    [DS_BASE, ES_BASE]
        .iter()
        .any(|&base| page >= base & !0xfff && page <= (base + 0x1_0004) & !0xfff)
}

impl Machine {
    fn new(context: Context) -> Self {
        Self::with_devices(context, false)
    }

    fn with_devices(context: Context, devices: bool) -> Self {
        let mut vm = Vm::new(VmConfig::flat(MEMORY));
        vm.set_cpu_model(context.model);
        vm.set_exception_delivery(ExceptionDelivery::Outward);
        if devices {
            // One device region per window: a transfer that straddles two
            // pages stays within one region.
            let mut start = 0u32;
            while u64::from(start) < MEMORY {
                let device = device_page(start);
                let mut end = start + 0x1000;
                while u64::from(end) < MEMORY && device_page(end) == device {
                    end += 0x1000;
                }
                let (prot, kind) = if device {
                    (Prot::RW, RegionKind::Trap)
                } else {
                    (Prot::RWX, RegionKind::Ram)
                };
                vm.map(u64::from(start), (end - start) as usize, prot, kind)
                    .unwrap();
                start = end;
            }
        } else {
            vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
        }
        // Identity paging of the whole memory, user and writable.
        vm.write_bytes(u64::from(PD), &(PT | 7).to_le_bytes())
            .unwrap();
        for page in 0..(MEMORY >> 12) as u32 {
            vm.write_bytes(u64::from(PT + 4 * page), &((page << 12) | 7).to_le_bytes())
                .unwrap();
        }
        Self {
            vm,
            context,
            devices,
            device: Default::default(),
        }
    }

    fn is_device(&self, linear: u32) -> bool {
        self.devices && device_page(linear)
    }

    fn write(&self, linear: u32, bytes: &[u8]) {
        for (i, &b) in bytes.iter().enumerate() {
            let at = linear.wrapping_add(i as u32);
            if self.is_device(at) {
                self.device.borrow_mut().insert(at, b);
            } else {
                self.vm.write_bytes(u64::from(at), &[b]).unwrap();
            }
        }
    }

    fn code_base(&self) -> u64 {
        if self.context.class == Class::Flat32 {
            0
        } else {
            u64::from(CS_BASE)
        }
    }

    /// A vCPU at `ip` with `case`'s registers and segment caches.
    fn vcpu(&self, case: &Case, ip: u64) -> Vcpu {
        let context = self.context;
        let rpl = u16::from(context.cpl);
        let mut cpu = self.vm.new_vcpu();
        cpu.set_a20_enabled(true);
        cpu.cpu.gpr[..8].copy_from_slice(&case.gpr);
        cpu.cpu.rip = ip;
        cpu.cpu.flags.df = case.df;
        cpu.cpu.flags.if_ = true;
        // IOPL 3: the port strings pass the I/O privilege test at every CPL.
        cpu.cpu.rflags_system |= 0x3000;
        cpu.cpu.legacy.cr0 = if context.paged { 0x8000_0001 } else { 1 };
        cpu.cpu.legacy.cr3 = PD;
        cpu.cpu.legacy.cpl = context.cpl;
        cpu.cpu.legacy.gdtr.base = GDT;
        cpu.cpu.legacy.gdtr.limit = 0x7f;
        let selectors: [u16; 6] = [0x08, 0x18, 0x10, 0x20, 0x28, 0x30];
        [
            cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs,
        ] = selectors.map(|s| s | rpl);
        for (i, seg) in case.segs.iter().enumerate() {
            cpu.cpu.legacy.segments[i] = SegmentCache {
                base: seg.base,
                limit: seg.limit,
                attributes: seg.attr,
                ..Default::default()
            };
            descriptor(
                &self.vm,
                u64::from(GDT + u32::from(selectors[i])),
                seg.base,
                seg.limit,
                seg.attr,
            );
        }
        if context.class == Class::Checked {
            // A non-present hidden GS cache that no operand uses keeps the
            // context out of Segmented admission (as in legacy_seg16).
            cpu.cpu.legacy.segments[5].attributes &= !0x80;
        }
        let class = derive_legacy_context(cpu.legacy_cpu_model().unwrap(), &cpu.cpu)
            .unwrap()
            .segment_class;
        assert_eq!(
            class,
            match context.class {
                Class::Segmented => SegmentClass::Segmented,
                Class::Checked => SegmentClass::Checked,
                Class::Flat32 => SegmentClass::Flat32,
            },
            "{context:?} {case:?}"
        );
        cpu
    }

    fn code(&self, case: &Case) -> Vec<u8> {
        let mut code = Vec::new();
        if case.lead {
            code.push(0x90);
        }
        if case.ss_source {
            code.push(0x36);
        }
        if self.context.code32 {
            code.push(0x67);
        }
        if case.elem != 1 && (case.elem == 4) != self.context.code32 {
            code.push(0x66);
        }
        match case.rep {
            Rep::None => {}
            Rep::F3 => code.push(0xf3),
            Rep::F2 => code.push(0xf2),
        }
        code.push(case.op.opcode(case.elem));
        // JMP $: the instruction after the string stays at its address.
        code.extend([0xeb, 0xfe]);
        code
    }

    fn linear(case: &Case, segment: usize, offset: u64) -> u32 {
        case.segs[segment].base.wrapping_add(offset as u32 & 0xffff)
    }

    /// The element offsets (SI, DI) a case visits, at most 32.
    fn visits(case: &Case) -> Vec<(u64, u64)> {
        let count = if case.rep == Rep::None {
            1
        } else {
            (case.gpr[1] & 0xffff).min(32)
        };
        let step = u64::from(case.elem);
        (0..count)
            .map(|k| {
                let d = if case.df {
                    0u64.wrapping_sub(k * step)
                } else {
                    k * step
                };
                (
                    case.gpr[6].wrapping_add(d) & 0xffff,
                    case.gpr[7].wrapping_add(d) & 0xffff,
                )
            })
            .collect()
    }

    /// Writes the operands so that a repeated comparison stops at element 2
    /// (REPE: element 2 differs; REPNE: element 2 is equal), and returns the
    /// linear bytes to compare afterwards.
    fn prepare(&self, case: &Case) -> Vec<u32> {
        let visits = Self::visits(case);
        let source = if case.ss_source { SS } else { DS };
        let elem = usize::from(case.elem);
        let mut watch = Vec::new();
        for (si, di) in &visits {
            for (segment, offset) in [(source, *si), (ES, *di)] {
                let at = Self::linear(case, segment, offset);
                for i in 0..elem + 2 {
                    let byte = at.wrapping_add(i as u32).wrapping_sub(1);
                    if u64::from(byte) < MEMORY {
                        self.write(byte, &[0xee]);
                        watch.push(byte);
                    }
                }
            }
        }
        for (k, (si, di)) in visits.iter().enumerate() {
            let value = (0x0102_0304_u64 * (k as u64 + 0x11)) & mask(case.elem);
            if case.op.reads_source() {
                let src = Self::linear(case, source, *si);
                self.write(src, &value.to_le_bytes()[..elem]);
            }
            if case.op.compares() {
                let lhs = if case.op == Op::Cmps {
                    value
                } else {
                    case.gpr[0] & mask(case.elem)
                };
                let equal = (k == 2) == (case.rep == Rep::F2);
                let rhs = if equal { lhs } else { lhs ^ 1 };
                let dst = Self::linear(case, ES, *di);
                self.write(dst, &rhs.to_le_bytes()[..elem]);
            }
        }
        watch.sort_unstable();
        watch.dedup();
        watch
    }

    fn byte(&self, linear: u32) -> u8 {
        if self.is_device(linear) {
            return self.device.borrow().get(&linear).copied().unwrap_or(0);
        }
        let mut b = [0];
        self.vm.read_bytes(u64::from(linear), &mut b).unwrap();
        b[0]
    }

    fn check_case(&self, case: &Case) {
        let code = self.code(case);
        let lead = u64::from(case.lead);
        let start = IP + lead;
        let next = IP + code.len() as u64 - 2;
        self.vm.write_bytes(self.code_base() + IP, &code).unwrap();
        let what = format!("{:?} {}: {case:?}", self.context, case.name);
        for stepped in [true, false] {
            let watch = self.prepare(case);
            let before: Vec<u8> = watch.iter().map(|&b| self.byte(b)).collect();
            let model = run_model(case, start, next, |at| self.byte(at));
            let mut cpu = self.vcpu(case, if stepped { start } else { IP });
            let initial_flags = (cpu.cpu.flags.zf, cpu.cpu.flags.cf);
            let (state, fault) = self.drive(&mut cpu, case, &model, stepped, next, &what);
            let completed = model.states.len() - 1;
            assert_eq!(
                state, model.states[completed],
                "final state, stepped={stepped}: {what}"
            );
            match model.end {
                End::Done => {
                    assert_eq!(fault, None, "stepped={stepped}: {what}");
                    let flags = (cpu.cpu.flags.zf, cpu.cpu.flags.cf);
                    let compared = case.rep == Rep::None || case.gpr[1] & 0xffff != 0;
                    if case.op.compares() && compared {
                        assert_eq!(flags, (model.zf, model.cf), "flags: {what}");
                    } else {
                        assert_eq!(flags, initial_flags, "flags: {what}");
                    }
                }
                End::Fault(vector) => {
                    assert_eq!(fault, Some(vector), "stepped={stepped}: {what}");
                }
            }
            for (&at, &old) in watch.iter().zip(&before) {
                let want = model.writes.get(&at).copied().unwrap_or(old);
                assert_eq!(
                    self.byte(at),
                    want,
                    "byte {at:#x}, stepped={stepped}: {what}"
                );
            }
        }
    }

    /// Runs `case` until it reaches `next` or faults: stepped, one step per call with every element
    /// boundary compared with the model (memory strings); otherwise in one
    /// run. Returns the final registers and the fault vector, if any.
    fn drive(
        &self,
        cpu: &mut Vcpu,
        case: &Case,
        model: &Model,
        stepped: bool,
        next: u64,
        what: &str,
    ) -> (State, Option<u8>) {
        let mut done = 0usize;
        let mut ports = model.ports.iter();
        let budget = if stepped {
            1
        } else {
            u64::from(case.lead) + model.states.len() as u64 + 1
        };
        for _ in 0..400 {
            let run = cpu
                .run_steps(&self.vm, budget, StepOptions::default())
                .unwrap();
            let state = State {
                gpr: cpu.cpu.gpr[..8].try_into().unwrap(),
                rip: cpu.cpu.rip,
            };
            match run.stop() {
                StepStop::Exit(Exit::BudgetExhausted) => {}
                StepStop::Exit(Exit::PortIo {
                    port,
                    size,
                    dir,
                    value,
                    ..
                }) => {
                    assert_eq!((*port, *size), (PORT, case.elem), "{what}");
                    let expected = *ports.next().expect("an extra port transfer");
                    match dir {
                        PortDir::Out => assert_eq!(*value, expected, "{what}"),
                        PortDir::In => cpu.complete_port_in(expected),
                    }
                    continue;
                }
                StepStop::Exit(Exit::MmioRead { addr, size, .. }) if self.devices => {
                    let value = (0..*size).fold(0u64, |v, i| {
                        v | u64::from(self.byte(*addr as u32 + u32::from(i))) << (8 * i)
                    });
                    cpu.complete_mmio_read(value);
                    continue;
                }
                StepStop::Exit(Exit::MmioWrite {
                    addr, size, value, ..
                }) if self.devices => {
                    self.write(*addr as u32, &value.to_le_bytes()[..usize::from(*size)]);
                    cpu.complete_mmio_write();
                    continue;
                }
                StepStop::Exit(Exit::Exception { vector, .. }) => {
                    return (state, Some(*vector));
                }
                other => panic!("{other:?}: {what}"),
            }
            if stepped && !case.op.port() && !self.devices {
                done += 1;
                assert_eq!(
                    Some(&state),
                    model.states.get(done),
                    "boundary {done}: {what}"
                );
            }
            if state.rip == next && cpu.cpu.port_string_in.is_none() {
                assert!(ports.next().is_none(), "a missing port transfer: {what}");
                return (state, None);
            }
        }
        panic!("no end: {what}");
    }
}

/// The scenario's registers and segment caches; `None` where it does not
/// apply to the context or the case.
fn scenario(context: Context, index: usize, op: Op, elem: u8, rep: Rep, df: bool) -> Option<Case> {
    let e = u64::from(elem);
    let mut segs = segments(context);
    let flat = context.class == Class::Flat32;
    let i286 = context.model == CpuModel::I80286;
    // Wrap: elements at 10000h-2e, 10000h-e, then 0000h (DF=0), or at e,
    // 0000h, then 10000h-e (DF=1).
    let wrap = if df { e } else { 0x1_0000 - 2 * e };
    // Straddle: element 1 of a word or doubleword string covers FFFFh and
    // 10000h.
    let straddle = if df { e - 1 } else { 0x1_0000 - e - 1 };
    let (name, si, di, cx, upper): (_, u64, u64, u64, [u64; 3]) = match index {
        // The reported image: REP MOVSW, CX=13h, DS:1CCAh to ES:347Fh, and
        // the reported upper halves of ECX, ESI and EDI.
        0 => ("reported", 0x1cca, 0x347f, 0x13, [0x2056, 0x0002, 0x0002]),
        1 => ("source wraps", wrap, 0x347f, 4, [0xdead, 0xbeef, 0x7fff]),
        2 => (
            "destination wraps",
            0x1cca,
            wrap,
            4,
            [0xffff, 0x8000, 0x0001],
        ),
        3 => ("source straddles", straddle, 0x347f, 3, [0x1234, 0xffff, 0]),
        4 => (
            "destination straddles",
            0x1cca,
            straddle,
            3,
            [0, 0x5555, 0xaaaa],
        ),
        5 if !flat => {
            // Element 2 crosses the DS limit (DF=0).
            segs[DS].limit = (0x1cca + 3 * e - 2) as u32;
            ("source limit", 0x1cca, 0x347f, 5, [0x00ff, 0xff00, 0x0f0f])
        }
        6 if !flat => {
            segs[ES].limit = (0x347f + 3 * e - 2) as u32;
            ("destination limit", 0x1cca, 0x347f, 5, [0x7f7f, 1, 2])
        }
        7 if !flat => {
            // Expand-down, B clear: offsets 1001h-FFFFh. DF=1 reaches 1000h
            // at element 2; DF=0 wraps from FFFFh to 0000h.
            segs[DS].attr |= 4;
            segs[DS].limit = 0x1000;
            let si = if df { 0x1000 + 2 * e } else { wrap };
            ("source expand-down", si, 0x347f, 5, [3, 4, 5])
        }
        8 if !flat && !i286 => {
            // Expand-down, B set: offsets 1001h-FFFFFFFFh, so an element at
            // FFFFh is within; the 16-bit wrap to the low offsets is not.
            segs[DS].attr |= 0x4004;
            segs[DS].limit = 0x1000;
            let si = if df { 0x1000 + 2 * e } else { straddle };
            ("source expand-down big", si, 0x347f, 5, [6, 7, 8])
        }
        9 if !flat => {
            segs[ES].attr |= 4;
            segs[ES].limit = 0x2000;
            let di = if df { 0x2000 + 2 * e } else { wrap };
            ("destination expand-down", 0x1cca, di, 5, [9, 10, 11])
        }
        10 if !flat && !i286 => {
            // Page granularity: limit FFFFFh pages, effective FFFFFFFFh.
            for s in [DS, ES] {
                segs[s].attr |= 0x8000;
                segs[s].limit = u32::MAX;
            }
            ("granular", straddle, straddle, 3, [12, 13, 14])
        }
        11 if !flat && op.reads_source() => {
            // SS override on the source: its limit faults #SS at element 2.
            segs[SS].limit = (0x1cca + 3 * e - 2) as u32;
            ("stack source limit", 0x1cca, 0x347f, 5, [15, 16, 17])
        }
        12 if !flat && rep != Rep::None => {
            // A zero count makes no access: limits that would fault any.
            for s in [DS, ES, SS] {
                segs[s].limit = 0;
            }
            ("zero count", 0x1cca, 0x347f, 0, [0xffff, 0xffff, 0xffff])
        }
        _ => return None,
    };
    let [ecx, esi, edi] = upper.map(|u| u << 16);
    Some(Case {
        op,
        elem,
        rep,
        df,
        ss_source: index == 11,
        lead: (index + usize::from(df) + usize::from(elem)) % 2 == 1,
        gpr: [
            0x5a5a_a5a5,
            ecx | cx,
            u64::from(PORT),
            0x0001_f811,
            0x7958,
            0x7b90,
            esi | si,
            edi | di,
        ],
        segs,
        name,
    })
}

const SCENARIOS: usize = 13;

fn check_context(context: Context, ops: &[Op]) {
    check_machine(Machine::new(context), ops);
}

fn check_machine(machine: Machine, ops: &[Op]) {
    let context = machine.context;
    let elems: &[u8] = if context.model == CpuModel::I80286 {
        &[1, 2]
    } else {
        &[1, 2, 4]
    };
    for &op in ops {
        for &elem in elems {
            let reps: &[Rep] = if op.compares() {
                &[Rep::None, Rep::F3, Rep::F2]
            } else {
                &[Rep::None, Rep::F3]
            };
            for &rep in reps {
                for df in [false, true] {
                    for index in 0..SCENARIOS {
                        if let Some(case) = scenario(context, index, op, elem, rep, df) {
                            machine.check_case(&case);
                        }
                    }
                }
            }
        }
    }
}

const MEMORY_OPS: [Op; 5] = [Op::Movs, Op::Stos, Op::Lods, Op::Cmps, Op::Scas];
const MODELS: [CpuModel; 2] = [CpuModel::I80386, CpuModel::I80486Dx];

fn sweep(paged: bool, class: Class, ops: &[Op]) {
    sweep_devices(paged, class, ops, false);
}

fn sweep_devices(paged: bool, class: Class, ops: &[Op], devices: bool) {
    for model in MODELS {
        for cpl in [0, 3] {
            for code32 in [false, true] {
                if class == Class::Flat32 && !code32 {
                    continue;
                }
                let context = Context {
                    model,
                    cpl,
                    paged,
                    class,
                    code32,
                };
                check_machine(Machine::with_devices(context, devices), ops);
            }
        }
    }
}

#[test]
fn memory_strings_segmented_unpaged() {
    sweep(false, Class::Segmented, &MEMORY_OPS);
}

#[test]
fn memory_strings_segmented_paged() {
    sweep(true, Class::Segmented, &MEMORY_OPS);
}

#[test]
fn memory_strings_checked_unpaged() {
    sweep(false, Class::Checked, &MEMORY_OPS);
}

#[test]
fn memory_strings_checked_paged() {
    sweep(true, Class::Checked, &MEMORY_OPS);
}

#[test]
fn memory_strings_flat32_under_67h() {
    // 67h in Flat32 code: the 16-bit offsets address the low 64 KiB.
    sweep(false, Class::Flat32, &MEMORY_OPS);
    sweep(true, Class::Flat32, &MEMORY_OPS);
}

#[test]
fn memory_strings_through_device_regions() {
    // Every element a device transfer: the handoff and its re-execution
    // keep the 16-bit indices (decision-16).
    for paged in [false, true] {
        for class in [Class::Segmented, Class::Checked] {
            sweep_devices(paged, class, &MEMORY_OPS, true);
        }
    }
}

#[test]
fn port_strings() {
    for paged in [false, true] {
        for class in [Class::Segmented, Class::Checked, Class::Flat32] {
            sweep(paged, class, &[Op::Ins, Op::Outs]);
        }
    }
}

#[test]
fn i286_strings() {
    // The 80286 has the 16-bit address size only (I286 PRM App. B MOVS,
    // STOS, LODS, CMPS, SCAS, INS, OUTS).
    for cpl in [0, 3] {
        check_context(
            Context {
                model: CpuModel::I80286,
                cpl,
                paged: false,
                class: Class::Segmented,
                code32: false,
            },
            &[
                Op::Movs,
                Op::Stos,
                Op::Lods,
                Op::Cmps,
                Op::Scas,
                Op::Ins,
                Op::Outs,
            ],
        );
    }
}

/// The reported register image (xj-srh.45): REP MOVSW at CS:798Eh in a
/// 16-bit ring-3 code segment named by an LDT selector; DS, ES, SS and FS
/// LDT selectors, a null GS. Returns the machine, the vCPU and the source
/// words.
fn reported(model: CpuModel, paged: bool) -> (Machine, Vcpu, Vec<u8>) {
    let machine = Machine::new(Context {
        model,
        cpl: 3,
        paged,
        class: Class::Segmented,
        code32: false,
    });
    let vm = &machine.vm;
    let mut cpu = vm.new_vcpu();
    cpu.set_a20_enabled(true);
    vm.write_bytes(u64::from(CS_BASE) + IP, &[0xf3, 0xa5, 0xeb, 0xfe])
        .unwrap();
    cpu.cpu.gpr[..8].copy_from_slice(&[
        0x02ef_0003,
        0x2056_0013,
        0x0000_03ce,
        0x0000_f811,
        0x0000_7958,
        0x0000_7b90,
        0x0002_1cca,
        0x0002_347f,
    ]);
    cpu.cpu.rip = IP;
    cpu.cpu.flags.if_ = true;
    cpu.cpu.legacy.cr0 = if paged { 0x8000_0001 } else { 1 };
    cpu.cpu.legacy.cr3 = PD;
    cpu.cpu.legacy.cpl = 3;
    cpu.cpu.legacy.gdtr.base = GDT;
    cpu.cpu.legacy.gdtr.limit = 0x7f;
    descriptor(vm, u64::from(GDT) + 0x28, LDT, 0x20ff, 0x82);
    cpu.cpu.legacy.ldtr = SystemSegment {
        selector: 0x28,
        cache: SegmentCache {
            base: LDT,
            limit: 0x20ff,
            attributes: 0x82,
            ..Default::default()
        },
        ..Default::default()
    };
    let selectors = [0x204f, 0x1027, 0x0c47, 0x18c7, 0x056f];
    let bases = [CS_BASE, DS_BASE, ES_BASE, SS_BASE, 0x3_0000];
    for i in 0..5 {
        let attr = if i == 0 { 0xfb } else { 0xf3 };
        cpu.cpu.legacy.segments[i] = SegmentCache {
            base: bases[i],
            limit: 0xffff,
            attributes: attr,
            ..Default::default()
        };
        descriptor(
            vm,
            u64::from(LDT) + u64::from(selectors[i] & !7),
            bases[i],
            0xffff,
            attr,
        );
    }
    [cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs] = selectors;
    cpu.cpu.gs = 0;
    cpu.cpu.legacy.segments[5] = SegmentCache {
        unusable: 1,
        ..Default::default()
    };
    let source: Vec<u8> = (0..0x26u8).map(|i| i.wrapping_mul(7)).collect();
    vm.write_bytes(u64::from(DS_BASE) + 0x1cca, &source)
        .unwrap();
    (machine, cpu, source)
}

fn assert_reported_done(machine: &Machine, cpu: &Vcpu, source: &[u8], what: &str) {
    assert_eq!(cpu.cpu.rip, IP + 2, "{what}");
    assert_eq!(cpu.cpu.cs, 0x204f, "{what}");
    assert_eq!(cpu.cpu.gpr[1], 0x2056_0000, "{what}");
    assert_eq!(cpu.cpu.gpr[6], 0x0002_1cf0, "{what}");
    assert_eq!(cpu.cpu.gpr[7], 0x0002_34a5, "{what}");
    let copied: Vec<u8> = (0..0x26)
        .map(|i| machine.byte(ES_BASE + 0x347f + i))
        .collect();
    assert_eq!(copied, source, "{what}");
}

const REPORTED_MODELS: [CpuModel; 3] = [
    CpuModel::I80386,
    CpuModel::I80486Dx,
    CpuModel::I80486DxCpuid,
];

#[test]
fn reported_rep_movsw_completes_without_fault() {
    for model in REPORTED_MODELS {
        for paged in [false, true] {
            for budget in [1, 7, 1000] {
                let (machine, mut cpu, source) = reported(model, paged);
                let what = format!("{model:?} paged={paged} budget={budget}");
                let mut runs = 0;
                while cpu.cpu.rip == IP && runs < 100 {
                    let run = cpu
                        .run_steps(&machine.vm, budget, StepOptions::default())
                        .unwrap();
                    assert!(
                        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                        "{what}: {:?}",
                        run.stop()
                    );
                    runs += 1;
                }
                assert_reported_done(&machine, &cpu, &source, &what);
            }
        }
    }
}

/// The reported REP MOVSW interrupted at element `k` by an external
/// interrupt through a 32-bit interrupt gate to CPL 0 and resumed by IRETD.
/// SDM Vol 2B REP p. 4-564: the interrupted string's registers are its
/// state at that element, and the saved EIP names the string. Vol 3A
/// §7.12.1 (interrupt to a more privileged level): SS, ESP, EFLAGS, CS and
/// EIP on the TSS's level-0 stack.
#[test]
fn reported_rep_movsw_resumes_after_an_interrupt() {
    const HANDLER: u32 = 0x500;
    const TSS: u32 = 0x5000;
    const IDT: u32 = 0x6000;
    const STACK0: u32 = 0x9000;
    for model in REPORTED_MODELS {
        for paged in [false, true] {
            for k in [0u64, 1, 5, 0x12] {
                let (machine, mut cpu, source) = reported(model, paged);
                let vm = &machine.vm;
                let what = format!("{model:?} paged={paged} k={k}");
                // Flat 32-bit CPL 0 code and stack; a 32-bit TSS naming the
                // level-0 stack; a 32-bit interrupt gate for vector 20h.
                descriptor(vm, u64::from(GDT) + 0x38, 0, 0xffff_ffff, 0xc09b);
                descriptor(vm, u64::from(GDT) + 0x40, 0, 0xffff_ffff, 0xc093);
                descriptor(vm, u64::from(GDT) + 0x48, TSS, 0x67, 0x8b);
                vm.write_bytes(u64::from(TSS) + 4, &STACK0.to_le_bytes())
                    .unwrap();
                vm.write_bytes(u64::from(TSS) + 8, &0x40u32.to_le_bytes())
                    .unwrap();
                vm.write_bytes(u64::from(TSS) + 0x66, &0x68u16.to_le_bytes())
                    .unwrap();
                let gate = (u64::from(HANDLER) & 0xffff) | (0x38 << 16) | (0x8e00 << 32);
                vm.write_bytes(u64::from(IDT) + 0x20 * 8, &gate.to_le_bytes())
                    .unwrap();
                // NOP; IRETD
                vm.write_bytes(u64::from(HANDLER), &[0x90, 0xcf]).unwrap();
                cpu.cpu.legacy.tr = SystemSegment {
                    selector: 0x48,
                    cache: SegmentCache {
                        base: TSS,
                        limit: 0x67,
                        attributes: 0x8b,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                cpu.cpu.legacy.idtr.base = IDT;
                cpu.cpu.legacy.idtr.limit = 0x7ff;
                for _ in 0..k {
                    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
                    assert!(
                        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                        "{what}: {:?}",
                        run.stop()
                    );
                }
                cpu.inject_irq(0x20);
                // The delivery, then the handler's NOP.
                let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
                assert!(
                    matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                    "{what}: {:?}",
                    run.stop()
                );
                assert_eq!(cpu.cpu.cs, 0x38, "{what}");
                assert_eq!(cpu.cpu.rip, u64::from(HANDLER) + 1, "{what}");
                assert_eq!(cpu.cpu.gpr[1], 0x2056_0013 - k, "{what}");
                assert_eq!(cpu.cpu.gpr[6], 0x0002_1cca + 2 * k, "{what}");
                assert_eq!(cpu.cpu.gpr[7], 0x0002_347f + 2 * k, "{what}");
                let frame: Vec<u32> = (0..5)
                    .map(|i| {
                        let mut b = [0; 4];
                        vm.read_bytes(u64::from(STACK0 - 20 + 4 * i), &mut b)
                            .unwrap();
                        u32::from_le_bytes(b)
                    })
                    .collect();
                assert_eq!(
                    [frame[0], frame[1], frame[3], frame[4]],
                    [IP as u32, 0x204f, 0x7958, 0x18c7],
                    "{what}"
                );
                let mut runs = 0;
                while cpu.cpu.rip != IP + 2 && runs < 100 {
                    cpu.run_steps(vm, 3, StepOptions::default()).unwrap();
                    runs += 1;
                }
                assert_reported_done(&machine, &cpu, &source, &what);
            }
        }
    }
}

/// The string after the instructions that set its operands, in one block:
/// MOV CX/SI/DI, LES DI and LDS SI from memory, CLD/STD, PUSH DS/POP ES.
#[test]
fn string_after_its_setup_in_one_block() {
    for model in MODELS {
        for paged in [false, true] {
            for class in [Class::Segmented, Class::Checked] {
                for cpl in [0, 3] {
                    for setup in 0..4 {
                        let context = Context {
                            model,
                            cpl,
                            paged,
                            class,
                            code32: false,
                        };
                        let machine = Machine::new(context);
                        let case = scenario(context, 0, Op::Movs, 2, Rep::F3, false).unwrap();
                        let rpl = u16::from(cpl);
                        // Far pointers at DS:0100h (ES:347Fh) and DS:0104h
                        // (DS:1CCAh), selectors as `vcpu` loads them.
                        let pointer = |offset: u16, selector: u16| {
                            let mut p = offset.to_le_bytes().to_vec();
                            p.extend(selector.to_le_bytes());
                            p
                        };
                        machine.write(DS_BASE + 0x100, &pointer(0x347f, 0x10 | rpl));
                        machine.write(DS_BASE + 0x104, &pointer(0x1cca, 0x18 | rpl));
                        let mut code = Vec::new();
                        match setup {
                            0 => code.extend([
                                0xb9, 0x13, 0x00, 0xbe, 0xca, 0x1c, 0xbf, 0x7f, 0x34, 0xfc,
                            ]),
                            // LES DI,[0100h]; LDS SI,[0104h]
                            1 => code.extend([0xc4, 0x3e, 0x00, 0x01, 0xc5, 0x36, 0x04, 0x01]),
                            // PUSH DS; POP ES; STD; CLD
                            2 => code.extend([0x1e, 0x07, 0xfd, 0xfc]),
                            // MOV AX,ES; MOV ES,AX; MOV CX,13h
                            _ => code.extend([0x8c, 0xc0, 0x8e, 0xc0, 0xb9, 0x13, 0x00]),
                        }
                        let string = IP + code.len() as u64;
                        code.extend([0xf3, 0xa5, 0xeb, 0xfe]);
                        machine.write(CS_BASE + IP as u32, &code);
                        let mut cpu = machine.vcpu(&case, IP);
                        let source: Vec<u8> = (0..0x26u8).map(|i| i ^ 0x5c).collect();
                        machine.write(DS_BASE + 0x1cca, &source);
                        let es = if setup == 2 { DS_BASE } else { ES_BASE };
                        if setup == 2 {
                            cpu.cpu.gpr[7] = 0x0002_747f;
                        }
                        let what = format!("{context:?} setup {setup}");
                        let mut steps = 0;
                        while cpu.cpu.rip != string + 2 && steps < 200 {
                            let run = cpu
                                .run_steps(&machine.vm, 7, StepOptions::default())
                                .unwrap();
                            assert!(
                                matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                                "{what}: {:?}",
                                run.stop()
                            );
                            steps += 1;
                        }
                        assert_eq!(cpu.cpu.gpr[1], 0x2056_0000, "{what}");
                        assert_eq!(cpu.cpu.gpr[6], 0x0002_1cf0, "{what}");
                        let di = if setup == 2 { 0x0002_74a5 } else { 0x0002_34a5 };
                        assert_eq!(cpu.cpu.gpr[7], di, "{what}");
                        let destination = if setup == 2 { 0x747f } else { 0x347f };
                        let copied: Vec<u8> = (0..0x26)
                            .map(|i| machine.byte(es + destination + i))
                            .collect();
                        assert_eq!(copied, source, "{what}");
                    }
                }
            }
        }
    }
}

/// The reported REP MOVSW after DS and ES are loaded by MOV, LDS/LES and
/// POP from LDT descriptors of every shape that admits its operands (SDM
/// Vol 3A §3.4.5 Fig. 3-8 and §6.3 pp. 6-4–6-5): a byte-granular 64-KiB
/// limit, a 20-bit byte limit, page granularity (limits Fh and FFFFFh
/// pages), and expand-down with B clear and set. The caches are the loads'.
#[test]
fn reported_rep_movsw_after_descriptor_loads() {
    // (raw limit, attributes)
    let shapes: [(u32, u16); 6] = [
        (0xffff, 0xf3),
        (0x1_ffff, 0xf3),
        (0xf, 0x80f3),
        (0xf_ffff, 0x80f3),
        (0, 0xf7),
        (0, 0x40f7),
    ];
    for model in REPORTED_MODELS {
        for paged in [false, true] {
            for (raw, attr) in shapes {
                for load in 0..3 {
                    let (machine, mut cpu, source) = reported(model, paged);
                    let vm = &machine.vm;
                    let what = format!("{model:?} paged={paged} {raw:#x}/{attr:#x} load {load}");
                    let effective = if attr & 0x8000 != 0 {
                        (raw << 12) | 0xfff
                    } else {
                        raw
                    };
                    // The fixture takes the effective limit and encodes G.
                    for (selector, base) in [(0x1027u16, DS_BASE), (0x0c47, ES_BASE)] {
                        let at = u64::from(LDT) + u64::from(selector & !7);
                        descriptor(vm, at, base, effective, attr);
                    }
                    // Stale caches that would fault: the loads replace them.
                    cpu.cpu.legacy.segments[1].limit = 0;
                    cpu.cpu.legacy.segments[2].limit = 0;
                    // Far pointers at SS:0100h.
                    let mut pointers = 0x347fu16.to_le_bytes().to_vec();
                    pointers.extend(0x0c47u16.to_le_bytes());
                    pointers.extend(0x1ccau16.to_le_bytes());
                    pointers.extend(0x1027u16.to_le_bytes());
                    machine.write(SS_BASE + 0x100, &pointers);
                    let code: Vec<u8> = match load {
                        // MOV AX,0C47h; MOV ES,AX; MOV AX,1027h; MOV DS,AX
                        0 => vec![0xb8, 0x47, 0x0c, 0x8e, 0xc0, 0xb8, 0x27, 0x10, 0x8e, 0xd8],
                        // SS: LES DI,[0100h]; SS: LDS SI,[0104h]
                        1 => vec![0x36, 0xc4, 0x3e, 0x00, 0x01, 0x36, 0xc5, 0x36, 0x04, 0x01],
                        // PUSH 0C47h; POP ES; PUSH 1027h; POP DS
                        _ => vec![0x68, 0x47, 0x0c, 0x07, 0x68, 0x27, 0x10, 0x1f],
                    };
                    let at = IP - code.len() as u64;
                    machine.write(CS_BASE + at as u32, &code);
                    cpu.cpu.rip = at;
                    let mut runs = 0;
                    while cpu.cpu.rip != IP + 2 && runs < 100 {
                        let run = cpu.run_steps(vm, 5, StepOptions::default()).unwrap();
                        assert!(
                            matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                            "{what}: {:?}",
                            run.stop()
                        );
                        runs += 1;
                    }
                    assert_eq!(cpu.cpu.legacy.segments[1].limit, effective, "{what}");
                    assert_eq!(cpu.cpu.legacy.segments[2].limit, effective, "{what}");
                    assert_reported_done(&machine, &cpu, &source, &what);
                }
            }
        }
    }
}
