//! Read-modify-write and locked operands on MMIO (spec.md §5.2).
//!
//! A memory-destination arithmetic, logical, shift or rotate instruction reads its
//! operand once and writes it once: the 8086 Family User's Manual (9800722-03)
//! Table 2-21 (pp. 2-51–2-67) counts two transfers for each `memory, register`,
//! `memory, immediate` and one-operand `memory` form, and one for a
//! `register, memory` form and for CMP (p. 2-53). XCHG with memory and a
//! LOCK-prefixed form hold the bus across that same read and write: SDM Vol 3A
//! §§11.1.2.1–11.1.2.2, Vol 2A §3.3 CMPXCHG ("the processor never produces a
//! locked read without also producing a locked write"); 80286 PRM Appendix B,
//! XCHG p. B-114. On Real16 and every Machine model a Trap operand
//! therefore performs one MmioRead and then one MmioWrite of its bus transfers,
//! with no replayed read; the twenty-bit models move every data byte as its own
//! transfer (spec.md §17.10).
//!
//! Each witness runs the instruction twice on the same path: once with the
//! operand window as RAM, once as a Trap region served by a byte-addressed
//! device. The two runs must agree on every register, flag, step and retirement
//! count and on the final operand bytes, and the Trap run must show exactly the
//! expected transfers, at budgets 1, 2, 3 and 64.
use std::collections::BTreeMap;
use x86jit_core::{
    Backend, CpuMode, CpuModel, Exit, InterpreterBackend, Prot, RegionCaps, RegionKind, SegmentId,
    StepOptions, StepStop, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x7c00;
const HANDLER: u64 = 0x0500;
const RAM_END: u64 = 0xa_0000;
/// The window and pointer of the Real16 report: ES = DS = B800h, DI = 10h.
const WINDOW: (u64, u64) = (0xb_8000, 0x8000);
const SEG: u16 = 0xb800;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    Alias,
    AliasJit,
    I8088,
    I8086,
    NecV20,
    NecV30,
    I286,
    I286Checked,
    I386,
    I386Checked,
}

const PATHS: [Path; 10] = [
    Path::Alias,
    Path::AliasJit,
    Path::I8088,
    Path::I8086,
    Path::NecV20,
    Path::NecV30,
    Path::I286,
    Path::I286Checked,
    Path::I386,
    Path::I386Checked,
];

impl Path {
    fn twenty_bit(self) -> bool {
        matches!(
            self,
            Self::I8088 | Self::I8086 | Self::NecV20 | Self::NecV30
        )
    }
    fn model(self) -> Option<CpuModel> {
        match self {
            Self::Alias | Self::AliasJit => None,
            Self::I8088 => Some(CpuModel::I8088),
            Self::I8086 => Some(CpuModel::I8086),
            Self::NecV20 => Some(CpuModel::NecV20),
            Self::NecV30 => Some(CpuModel::NecV30),
            Self::I286 | Self::I286Checked => Some(CpuModel::I80286),
            Self::I386 | Self::I386Checked => Some(CpuModel::I80386),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// One read, then one write of the operand.
    Rmw,
    /// One read only (CMP, TEST, BT, `reg, mem`).
    Read,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    bytes: &'static [u8],
    size: u8,
    kind: Kind,
}

const fn rmw(name: &'static str, bytes: &'static [u8], size: u8) -> Case {
    Case {
        name,
        bytes,
        size,
        kind: Kind::Rmw,
    }
}

const fn read(name: &'static str, bytes: &'static [u8], size: u8) -> Case {
    Case {
        name,
        bytes,
        size,
        kind: Kind::Read,
    }
}

/// ModRM 05h is `[DI]` (mod 00, r/m 101); 2Dh selects register 5 (CH/BP).
const XCHG: &[Case] = &[
    rmw("xchg es:[di],al", &[0x26, 0x86, 0x05], 1),
    rmw("xchg es:[di],ax", &[0x26, 0x87, 0x05], 2),
    rmw("xchg [di],al", &[0x86, 0x05], 1),
    rmw("xchg [di],ax", &[0x87, 0x05], 2),
    rmw("xchg es:[di],ch", &[0x26, 0x86, 0x2d], 1),
    rmw("xchg es:[di],bp", &[0x26, 0x87, 0x2d], 2),
    rmw("xchg [di],ch", &[0x86, 0x2d], 1),
    rmw("xchg [di],bp", &[0x87, 0x2d], 2),
];

const ALU: &[Case] = &[
    rmw("add es:[di],al", &[0x26, 0x00, 0x05], 1),
    rmw("add es:[di],ax", &[0x26, 0x01, 0x05], 2),
    rmw("or es:[di],al", &[0x26, 0x08, 0x05], 1),
    rmw("or es:[di],ax", &[0x26, 0x09, 0x05], 2),
    rmw("adc es:[di],al", &[0x26, 0x10, 0x05], 1),
    rmw("adc es:[di],ax", &[0x26, 0x11, 0x05], 2),
    rmw("sbb es:[di],al", &[0x26, 0x18, 0x05], 1),
    rmw("sbb es:[di],ax", &[0x26, 0x19, 0x05], 2),
    rmw("and es:[di],al", &[0x26, 0x20, 0x05], 1),
    rmw("and es:[di],ax", &[0x26, 0x21, 0x05], 2),
    rmw("sub es:[di],al", &[0x26, 0x28, 0x05], 1),
    rmw("sub es:[di],ax", &[0x26, 0x29, 0x05], 2),
    rmw("xor es:[di],al", &[0x26, 0x30, 0x05], 1),
    rmw("xor es:[di],ax", &[0x26, 0x31, 0x05], 2),
    read("cmp es:[di],al", &[0x26, 0x38, 0x05], 1),
    read("cmp es:[di],ax", &[0x26, 0x39, 0x05], 2),
    read("cmp al,es:[di]", &[0x26, 0x3a, 0x05], 1),
    read("add ax,es:[di]", &[0x26, 0x03, 0x05], 2),
    read("test es:[di],al", &[0x26, 0x84, 0x05], 1),
    read("test es:[di],ax", &[0x26, 0x85, 0x05], 2),
    rmw("add [di],al", &[0x00, 0x05], 1),
    rmw("add [di],ax", &[0x01, 0x05], 2),
    rmw("add byte [di],96h", &[0x80, 0x05, 0x96], 1),
    rmw("or byte [di],96h", &[0x80, 0x0d, 0x96], 1),
    rmw("adc byte [di],96h", &[0x80, 0x15, 0x96], 1),
    rmw("sbb byte [di],96h", &[0x80, 0x1d, 0x96], 1),
    rmw("and byte [di],96h", &[0x80, 0x25, 0x96], 1),
    rmw("sub byte [di],96h", &[0x80, 0x2d, 0x96], 1),
    rmw("xor byte [di],96h", &[0x80, 0x35, 0x96], 1),
    read("cmp byte [di],96h", &[0x80, 0x3d, 0x96], 1),
    rmw("add word [di],9234h", &[0x81, 0x05, 0x34, 0x92], 2),
    rmw("sbb word [di],9234h", &[0x81, 0x1d, 0x34, 0x92], 2),
    rmw("xor word [di],9234h", &[0x81, 0x35, 0x34, 0x92], 2),
    read("cmp word [di],9234h", &[0x81, 0x3d, 0x34, 0x92], 2),
    rmw("add word [di],-0fh", &[0x83, 0x05, 0xf1], 2),
    rmw("or word [di],-0fh", &[0x83, 0x0d, 0xf1], 2),
    rmw("sub word [di],-0fh", &[0x83, 0x2d, 0xf1], 2),
    read("test byte [di],96h", &[0xf6, 0x05, 0x96], 1),
    read("test word [di],9234h", &[0xf7, 0x05, 0x34, 0x92], 2),
];

const UNARY: &[Case] = &[
    rmw("inc byte es:[di]", &[0x26, 0xfe, 0x05], 1),
    rmw("dec byte es:[di]", &[0x26, 0xfe, 0x0d], 1),
    rmw("inc word es:[di]", &[0x26, 0xff, 0x05], 2),
    rmw("dec word es:[di]", &[0x26, 0xff, 0x0d], 2),
    rmw("not byte es:[di]", &[0x26, 0xf6, 0x15], 1),
    rmw("neg byte es:[di]", &[0x26, 0xf6, 0x1d], 1),
    rmw("not word es:[di]", &[0x26, 0xf7, 0x15], 2),
    rmw("neg word es:[di]", &[0x26, 0xf7, 0x1d], 2),
    rmw("inc byte [di]", &[0xfe, 0x05], 1),
    rmw("neg word [di]", &[0xf7, 0x1d], 2),
];

const SHIFT: &[Case] = &[
    rmw("rol byte [di],1", &[0xd0, 0x05], 1),
    rmw("ror byte [di],1", &[0xd0, 0x0d], 1),
    rmw("rcl byte [di],1", &[0xd0, 0x15], 1),
    rmw("rcr byte [di],1", &[0xd0, 0x1d], 1),
    rmw("shl byte [di],1", &[0xd0, 0x25], 1),
    rmw("shr byte [di],1", &[0xd0, 0x2d], 1),
    rmw("sar byte [di],1", &[0xd0, 0x3d], 1),
    rmw("rol word [di],1", &[0xd1, 0x05], 2),
    rmw("rcr word [di],1", &[0xd1, 0x1d], 2),
    rmw("shl word [di],1", &[0xd1, 0x25], 2),
    rmw("sar word [di],1", &[0xd1, 0x3d], 2),
    rmw("rol byte [di],cl", &[0xd2, 0x05], 1),
    rmw("rcl byte [di],cl", &[0xd2, 0x15], 1),
    rmw("shr byte [di],cl", &[0xd2, 0x2d], 1),
    rmw("ror word [di],cl", &[0xd3, 0x0d], 2),
    rmw("rcr word [di],cl", &[0xd3, 0x1d], 2),
    rmw("shl word [di],cl", &[0xd3, 0x25], 2),
    rmw("sar word es:[di],cl", &[0x26, 0xd3, 0x3d], 2),
    // 80186-class immediate counts: absent on the 8086 entries (typed refusal).
    rmw("shl byte [di],2", &[0xc0, 0x25, 0x02], 1),
    rmw("rcr word [di],5", &[0xc1, 0x1d, 0x05], 2),
];

/// LOCK forms: admitted on the alias and the 286/386, a typed refusal on the
/// twenty-bit entries (spec.md §17.10). LOCK ADC/SBB is refused at lift on
/// every path (`lift_binop`). Both prefix orders reach the same operand.
const LOCK: &[Case] = &[
    rmw("lock add es:[di],al", &[0xf0, 0x26, 0x00, 0x05], 1),
    rmw("es lock add [di],al", &[0x26, 0xf0, 0x00, 0x05], 1),
    rmw("lock add es:[di],ax", &[0xf0, 0x26, 0x01, 0x05], 2),
    rmw("lock or es:[di],al", &[0xf0, 0x26, 0x08, 0x05], 1),
    rmw("lock or [di],ax", &[0xf0, 0x09, 0x05], 2),
    rmw("lock and [di],al", &[0xf0, 0x20, 0x05], 1),
    rmw("lock sub [di],ax", &[0xf0, 0x29, 0x05], 2),
    rmw("lock xor [di],al", &[0xf0, 0x30, 0x05], 1),
    rmw("lock adc [di],al", &[0xf0, 0x10, 0x05], 1),
    rmw("lock add byte [di],96h", &[0xf0, 0x80, 0x05, 0x96], 1),
    rmw(
        "lock sub word [di],9234h",
        &[0xf0, 0x81, 0x2d, 0x34, 0x92],
        2,
    ),
    rmw("lock or word [di],-0fh", &[0xf0, 0x83, 0x0d, 0xf1], 2),
    rmw("lock inc byte es:[di]", &[0xf0, 0x26, 0xfe, 0x05], 1),
    rmw("lock dec word [di]", &[0xf0, 0xff, 0x0d], 2),
    rmw("lock not byte [di]", &[0xf0, 0xf6, 0x15], 1),
    rmw("lock neg word [di]", &[0xf0, 0xf7, 0x1d], 2),
    rmw("lock xchg es:[di],al", &[0xf0, 0x26, 0x86, 0x05], 1),
    rmw("es lock xchg [di],ax", &[0x26, 0xf0, 0x87, 0x05], 2),
];

/// 80386 forms (operand-size prefix, bit tests) and the alias superset's
/// 80486 forms. Absent encodings take each path's own refusal or #UD.
const WIDE: &[Case] = &[
    rmw("xchg es:[di],eax", &[0x66, 0x26, 0x87, 0x05], 4),
    rmw("add es:[di],eax", &[0x66, 0x26, 0x01, 0x05], 4),
    rmw("lock add es:[di],eax", &[0xf0, 0x66, 0x26, 0x01, 0x05], 4),
    rmw("inc dword [di]", &[0x66, 0xff, 0x05], 4),
    rmw("shl dword [di],cl", &[0x66, 0xd3, 0x25], 4),
    // SI = 0 selects bit 0 of the operand. A register bit offset addresses the
    // bit string through the byte that holds the bit (`lift_bt`); an immediate
    // offset accesses the operand width.
    rmw("bts [di],si", &[0x0f, 0xab, 0x35], 1),
    rmw("lock bts [di],si", &[0xf0, 0x0f, 0xab, 0x35], 1),
    rmw("lock btr [di],si", &[0xf0, 0x0f, 0xb3, 0x35], 1),
    rmw("lock btc word [di],3", &[0xf0, 0x0f, 0xba, 0x3d, 0x03], 2),
    read("bt [di],si", &[0x0f, 0xa3, 0x35], 1),
    rmw("cmpxchg [di],cl", &[0x0f, 0xb0, 0x0d], 1),
    rmw("lock cmpxchg [di],cx", &[0xf0, 0x0f, 0xb1, 0x0d], 2),
    rmw("xadd [di],al", &[0x0f, 0xc0, 0x05], 1),
    rmw("lock xadd es:[di],ax", &[0xf0, 0x26, 0x0f, 0xc1, 0x05], 2),
];

/// A word operand at an odd address, and at the end of the segment (8086 wraps
/// the offset; the 286, 386 and the alias raise #GP before any transfer).
const EDGE: &[Case] = &[
    rmw("xchg es:[di],ax", &[0x26, 0x87, 0x05], 2),
    rmw("add es:[di],ax", &[0x26, 0x01, 0x05], 2),
    rmw("inc word es:[di]", &[0x26, 0xff, 0x05], 2),
    rmw("shl word [di],1", &[0xd1, 0x25], 2),
    rmw("lock add es:[di],ax", &[0xf0, 0x26, 0x01, 0x05], 2),
    read("cmp es:[di],ax", &[0x26, 0x39, 0x05], 2),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bus {
    Read(u64, u8),
    Write(u64, u8, u64),
}

/// A byte-addressed device: reads return its bytes, writes replace them.
struct Device(BTreeMap<u64, u8>);
impl Device {
    fn new(window: (u64, u64)) -> Self {
        Self(
            (window.0..window.0 + window.1)
                .map(|a| (a, seed(a)))
                .collect(),
        )
    }
    fn read(&self, addr: u64, size: u8) -> u64 {
        (0..size).fold(0, |v, i| {
            v | u64::from(self.0[&(addr + u64::from(i))]) << (8 * i)
        })
    }
    fn write(&mut self, addr: u64, size: u8, value: u64) {
        for i in 0..size {
            self.0.insert(addr + u64::from(i), (value >> (8 * i)) as u8);
        }
    }
}

fn seed(addr: u64) -> u8 {
    (addr.wrapping_mul(37) as u8) ^ 0x81
}

struct Fixture {
    path: Path,
    window: (u64, u64),
    segment: u16,
    offset: u16,
}

impl Fixture {
    fn setup(&self, trap: bool, code: &[u8]) -> (Vm, Vcpu) {
        let backend: Box<dyn Backend> = match self.path {
            Path::AliasJit => Box::new(JitBackend::with_superblocks(RegionCaps {
                max_blocks: 8,
                max_icount: 64,
            })),
            _ => Box::new(InterpreterBackend),
        };
        let mut vm = Vm::with_backend(VmConfig::flat(0x10_0000), backend);
        match self.path.model() {
            Some(model) => vm.set_cpu_model(model),
            None => {
                vm.set_cpu_mode(CpuMode::Real16);
                vm.set_tier_up_after(None);
                vm.set_tier_up_region_after(Some(1));
            }
        }
        vm.map(0, RAM_END as usize, Prot::RWX, RegionKind::Ram)
            .unwrap();
        let kind = if trap {
            RegionKind::Trap
        } else {
            RegionKind::Ram
        };
        vm.map(self.window.0, self.window.1 as usize, Prot::RW, kind)
            .unwrap();
        if !trap {
            let bytes: Vec<u8> = (self.window.0..self.window.0 + self.window.1)
                .map(seed)
                .collect();
            vm.write_bytes(self.window.0, &bytes).unwrap();
        }
        // Every processor exception enters 0000:0500, which halts.
        for vector in 0..32u64 {
            vm.mem.write(vector * 4, HANDLER, 4).unwrap();
        }
        vm.write_bytes(HANDLER, &[0xf4]).unwrap();
        vm.write_bytes(CODE, code).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = CODE;
        for (id, selector) in [
            (SegmentId::Cs, 0),
            (SegmentId::Ss, 0),
            (SegmentId::Ds, self.segment),
            (SegmentId::Es, self.segment),
        ] {
            set_segment(&mut cpu, id, selector);
        }
        if matches!(self.path, Path::I286Checked | Path::I386Checked) {
            // A non-canonical SS cache withholds the Direct proof; no case
            // reloads SS or touches the stack.
            cpu.cpu.legacy.segments[SegmentId::Ss as usize].attributes = 0x92;
        }
        cpu.cpu.gpr[0] = 0xc35a; // AX
        cpu.cpu.gpr[1] = 0x0503; // CX: CL = 3, CH = 5
        cpu.cpu.gpr[2] = 0x1111; // DX
        cpu.cpu.gpr[3] = 0; // BX
        cpu.cpu.gpr[4] = 0x7000; // SP
        cpu.cpu.gpr[5] = 0x1234; // BP
        cpu.cpu.gpr[6] = 0; // SI
        cpu.cpu.gpr[7] = u64::from(self.offset); // DI
        cpu.cpu.flags.cf = true;
        (vm, cpu)
    }
}

fn set_segment(cpu: &mut Vcpu, id: SegmentId, selector: u16) {
    match id {
        SegmentId::Cs => cpu.cpu.cs = selector,
        SegmentId::Ds => cpu.cpu.ds = selector,
        SegmentId::Es => cpu.cpu.es = selector,
        SegmentId::Ss => cpu.cpu.ss = selector,
        _ => unreachable!(),
    }
    let cache = &mut cpu.cpu.legacy.segments[id as usize];
    cache.base = u32::from(selector) << 4;
    cache.limit = 0xffff;
    cache.attributes = if id == SegmentId::Cs { 0x9b } else { 0x93 };
}

#[derive(Debug, PartialEq, Eq)]
struct Observation {
    stop: String,
    steps: u64,
    retired: u64,
    gpr: [u64; 8],
    rip: u64,
    cs: u16,
    flags: u64,
}

struct Run {
    observation: Observation,
    bus: Vec<Bus>,
    /// Retired-instruction count at each MMIO stop.
    retired_at_bus: Vec<u64>,
}

fn observe(cpu: &Vcpu, stop: String, steps: u64) -> Observation {
    let mut gpr = [0; 8];
    gpr.copy_from_slice(&cpu.cpu.gpr[..8]);
    Observation {
        stop,
        steps,
        retired: cpu.retired_instructions(),
        gpr,
        rip: cpu.cpu.rip,
        cs: cpu.cpu.cs,
        flags: cpu.cpu.flags.to_rflags() & 0x8d5,
    }
}

/// Drive `run_steps` with `budget` until a stop other than budget or MMIO.
fn drive(vm: &Vm, cpu: &mut Vcpu, budget: u64, mut device: Option<&mut Device>) -> Run {
    let mut bus = Vec::new();
    let mut retired_at_bus = Vec::new();
    let mut steps = 0;
    for _ in 0..64 {
        let run = cpu.run_steps(vm, budget, StepOptions::default()).unwrap();
        assert!(run.steps() <= budget, "{:?}", run.stop());
        steps += run.steps();
        match run.stop() {
            StepStop::Exit(Exit::BudgetExhausted) => {}
            StepStop::Exit(Exit::MmioRead { addr, size, .. }) => {
                let device = device.as_deref_mut().expect("MMIO read from RAM");
                bus.push(Bus::Read(*addr, *size));
                retired_at_bus.push(cpu.retired_instructions());
                cpu.complete_mmio_read(device.read(*addr, *size));
            }
            StepStop::Exit(Exit::MmioWrite {
                addr, size, value, ..
            }) => {
                let device = device.as_deref_mut().expect("MMIO write to RAM");
                bus.push(Bus::Write(*addr, *size, *value));
                retired_at_bus.push(cpu.retired_instructions());
                device.write(*addr, *size, *value);
                cpu.complete_mmio_write();
            }
            stop => {
                let stop = match stop {
                    StepStop::Exit(Exit::Hlt) => "Hlt".to_string(),
                    StepStop::Exit(Exit::UnknownInstruction { addr, len, .. }) => {
                        format!("UnknownInstruction@{addr:#x}+{len}")
                    }
                    other => format!("{other:?}"),
                };
                return Run {
                    observation: observe(cpu, stop, steps),
                    bus,
                    retired_at_bus,
                };
            }
        }
    }
    panic!("no final stop: {bus:x?}");
}

/// The bus transfers the Trap run must show for an instruction that completes.
fn expected_bus(f: &Fixture, case: &Case, ram_window: &dyn Fn(u64) -> u8) -> Vec<Bus> {
    let base = u64::from(f.segment) << 4;
    let pieces: Vec<(u64, u8)> = if f.path.twenty_bit() {
        // UM §2.3 p. 2-12: the offset wraps inside the segment, one byte per transfer.
        (0..case.size)
            .map(|i| (base + u64::from(f.offset.wrapping_add(u16::from(i))), 1))
            .collect()
    } else {
        vec![(base + u64::from(f.offset), case.size)]
    };
    let mut bus: Vec<Bus> = pieces.iter().map(|&(a, n)| Bus::Read(a, n)).collect();
    if case.kind == Kind::Rmw {
        bus.extend(pieces.iter().map(|&(a, n)| {
            let value = (0..n).fold(0, |v, i| {
                v | u64::from(ram_window(a + u64::from(i))) << (8 * i)
            });
            Bus::Write(a, n, value)
        }));
    }
    bus
}

/// Outcome of one case on one path: whether the instruction completed. Every
/// disagreement is appended to `failures`, so one run reports the whole matrix.
fn witness(f: &Fixture, case: &Case, failures: &mut Vec<String>) -> bool {
    let what = format!(
        "{:?} {} at {:04x}:{:04x}",
        f.path, case.name, f.segment, f.offset
    );
    // INC BX ; <case> ; INC SI ; HLT
    let mut code = vec![0x43];
    code.extend_from_slice(case.bytes);
    code.extend_from_slice(&[0x46, 0xf4]);
    let end = CODE + code.len() as u64;

    let (vm, mut cpu) = f.setup(false, &code);
    let reference = drive(&vm, &mut cpu, 64, None);
    assert!(reference.bus.is_empty(), "{what}");
    let ram: Vec<u8> = (f.window.0..f.window.0 + f.window.1)
        .map(|a| vm.mem.read(a, 1).unwrap() as u8)
        .collect();
    let ram_at = |a: u64| ram[(a - f.window.0) as usize];
    let completed = reference.observation.stop == "Hlt" && reference.observation.rip == end;
    let expected = if completed {
        expected_bus(f, case, &ram_at)
    } else {
        Vec::new()
    };

    for budget in [1, 2, 3, 64] {
        let what = format!("{what}, budget {budget}");
        let (vm, mut cpu) = f.setup(true, &code);
        let mut device = Device::new(f.window);
        let run = drive(&vm, &mut cpu, budget, Some(&mut device));
        if run.bus != expected {
            failures.push(format!(
                "{what}: transfers {:x?}, expected {:x?}",
                run.bus, expected
            ));
            break;
        }
        if run.observation != reference.observation {
            failures.push(format!(
                "{what}: {:x?}, RAM run {:x?}",
                run.observation, reference.observation
            ));
            break;
        }
        // Only INC BX has retired while the case's transfers are outstanding.
        if !run.retired_at_bus.iter().all(|&r| r == 1) {
            failures.push(format!("{what}: retired {:?}", run.retired_at_bus));
            break;
        }
        let device_bytes: Vec<u8> = (f.window.0..f.window.0 + f.window.1)
            .map(|a| device.0[&a])
            .collect();
        if device_bytes != ram {
            failures.push(format!("{what}: device and RAM differ"));
            break;
        }
    }
    completed
}

/// The cases that completed on `path`; mismatches go to `failures`.
fn sweep(
    path: Path,
    cases: &[Case],
    window: (u64, u64),
    segment: u16,
    offset: u16,
    failures: &mut Vec<String>,
) -> Vec<&'static str> {
    let f = Fixture {
        path,
        window,
        segment,
        offset,
    };
    cases
        .iter()
        .filter(|case| witness(&f, case, failures))
        .map(|case| case.name)
        .collect()
}

/// Run every path; `check` sees the completed case names of each path.
fn matrix(
    cases: &[Case],
    window: (u64, u64),
    segment: u16,
    offset: u16,
    check: impl Fn(Path, &[&'static str]) -> bool,
) {
    let mut failures = Vec::new();
    for path in PATHS {
        let done = sweep(path, cases, window, segment, offset, &mut failures);
        println!(
            "{path:?}: {} of {} completed: {done:?}",
            done.len(),
            cases.len()
        );
        if !check(path, &done) {
            failures.push(format!("{path:?}: unexpected completed set {done:?}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn xchg_reads_then_writes_once() {
    matrix(XCHG, WINDOW, SEG, 0x10, |_, done| done.len() == XCHG.len());
}

#[test]
fn alu_reads_then_writes_once() {
    matrix(ALU, WINDOW, SEG, 0x10, |_, done| done.len() == ALU.len());
}

#[test]
fn inc_dec_not_neg_read_then_write_once() {
    matrix(UNARY, WINDOW, SEG, 0x10, |_, done| {
        done.len() == UNARY.len()
    });
}

#[test]
fn shifts_and_rotates_read_then_write_once() {
    matrix(SHIFT, WINDOW, SEG, 0x10, |path, done| {
        // 80186-class immediate counts are absent on the 8086 entries only.
        let absent = matches!(path, Path::I8088 | Path::I8086);
        done.len() == SHIFT.len() - if absent { 2 } else { 0 }
    });
}

#[test]
fn lock_forms_read_then_write_once_where_admitted() {
    matrix(LOCK, WINDOW, SEG, 0x10, |path, done| {
        !done.contains(&"lock adc [di],al") && (path.twenty_bit() || done.len() == LOCK.len() - 1)
    });
}

#[test]
fn wide_forms_read_then_write_once_where_present() {
    matrix(WIDE, WINDOW, SEG, 0x10, |path, done| match path {
        Path::Alias | Path::AliasJit => done.len() == WIDE.len(),
        Path::I386 | Path::I386Checked => done.len() == 10,
        _ => true,
    });
}

#[test]
fn word_at_odd_address_reads_then_writes_once() {
    matrix(EDGE, WINDOW, SEG, 0x11, |path, done| {
        done.len() == EDGE.len() - usize::from(path.twenty_bit())
    });
}

#[test]
fn word_at_segment_edge_wraps_or_faults_before_any_transfer() {
    // B000:FFFF and its 8086 wrap B000:0000 both lie in the device window.
    matrix(EDGE, (0xb_0000, 0x1_0000), 0xb000, 0xffff, |path, done| {
        if path.twenty_bit() {
            done.len() == EDGE.len() - 1
        } else {
            done.is_empty()
        }
    });
}

/// A word operand over the last RAM byte and the first device byte. A plain
/// operand keeps the containment miss of spec.md §5.2 (splitting device accesses
/// is xj-ir6.30). A locked one keeps embedder emulation, as before:
/// `UnknownInstruction` with the instruction's bytes, before any transfer and
/// with nothing committed; the processor performs it in as many bus cycles as
/// necessary (SDM Vol 3A §11.1.2.2), which a follow-up will express. The
/// twenty-bit models move each byte separately (`twenty_bit_rmw_over_ram_and_device`).
#[test]
fn operand_across_ram_and_device() {
    const BASE: u64 = 0xb_7fff;
    for path in PATHS.into_iter().filter(|p| !p.twenty_bit()) {
        for (name, bytes, plain) in [
            ("add es:[di],ax", &[0x26, 0x01, 0x05][..], true),
            ("xchg es:[di],ax", &[0x26, 0x87, 0x05][..], false),
            ("lock add es:[di],ax", &[0xf0, 0x26, 0x01, 0x05][..], false),
        ] {
            let what = format!("{path:?} {name}");
            let f = Fixture {
                path,
                window: WINDOW,
                segment: 0xb000,
                offset: 0x7fff,
            };
            let mut code = vec![0x43];
            code.extend_from_slice(bytes);
            code.extend_from_slice(&[0x46, 0xf4]);
            let (mut vm, mut cpu) = f.setup(true, &code);
            vm.map(
                RAM_END,
                (WINDOW.0 - RAM_END) as usize,
                Prot::RW,
                RegionKind::Ram,
            )
            .unwrap();
            vm.write_bytes(BASE, &[0x77]).unwrap();
            let before = cpu.cpu.gpr;
            let mut device = Device::new(WINDOW);
            let run = drive(&vm, &mut cpu, 64, Some(&mut device));
            assert!(run.bus.is_empty(), "{what}: {:x?}", run.bus);
            if plain {
                assert!(
                    run.observation
                        .stop
                        .contains(&format!("UnmappedMemory {{ addr: {BASE}, access: Read")),
                    "{what}: {}",
                    run.observation.stop
                );
            } else {
                assert_eq!(
                    run.observation.stop,
                    format!("UnknownInstruction@{:#x}+{}", CODE + 1, bytes.len()),
                    "{what}"
                );
            }
            assert_eq!(run.observation.rip, CODE + 1, "{what}");
            assert_eq!(run.observation.retired, 1, "{what}");
            let mut expected = before;
            expected[3] += 1;
            assert_eq!(cpu.cpu.gpr[..8], expected[..8], "{what}");
            assert_eq!(vm.mem.read(BASE, 1).unwrap(), 0x77, "{what}");
        }
    }
}

/// Twenty-bit word XCHG and ADD over one RAM byte and one device byte, in both
/// byte orders. The device byte is read once and written once; the store writes
/// it before the RAM byte, so the RAM byte commits only after the device write is
/// acknowledged and the retry computes from the operand it read first. The
/// result equals the one computed from the operand as first read.
#[test]
fn twenty_bit_rmw_over_ram_and_device() {
    for path in PATHS.into_iter().filter(|p| p.twenty_bit()) {
        // (segment, offset, RAM byte address, device byte address)
        for (segment, offset, ram_at, device_at) in [
            (0xb000u16, 0x7fffu16, 0xb_7fffu64, 0xb_8000u64),
            (0xb800, 0x7fff, 0xc_0000, 0xb_ffff),
        ] {
            for (name, bytes) in [
                ("xchg", &[0x26, 0x87, 0x05][..]),
                ("add", &[0x26, 0x01, 0x05][..]),
            ] {
                let what = format!("{path:?} {name} RAM {ram_at:#x} device {device_at:#x}");
                let f = Fixture {
                    path,
                    window: WINDOW,
                    segment,
                    offset,
                };
                let mut code = bytes.to_vec();
                code.push(0xf4);
                let (mut vm, mut cpu) = f.setup(true, &code);
                vm.map(
                    RAM_END,
                    (WINDOW.0 - RAM_END) as usize,
                    Prot::RW,
                    RegionKind::Ram,
                )
                .unwrap();
                vm.map(WINDOW.0 + WINDOW.1, 0x1000, Prot::RW, RegionKind::Ram)
                    .unwrap();
                vm.write_bytes(ram_at, &[0x77]).unwrap();
                let mut device = Device(BTreeMap::from([(device_at, 0x12)]));
                let run = drive(&vm, &mut cpu, 64, Some(&mut device));
                let low_is_ram = ram_at < device_at;
                let old: u64 = if low_is_ram { 0x1277 } else { 0x7712 };
                let (ax, result) = if name == "xchg" {
                    (old, 0xc35a)
                } else {
                    (0xc35a, (old + 0xc35a) & 0xffff)
                };
                let (ram_value, device_value) = if low_is_ram {
                    (result & 0xff, result >> 8)
                } else {
                    (result >> 8, result & 0xff)
                };
                assert_eq!(
                    run.bus,
                    [
                        Bus::Read(device_at, 1),
                        Bus::Write(device_at, 1, device_value)
                    ],
                    "{what}"
                );
                assert_eq!(run.observation.stop, "Hlt", "{what}");
                assert_eq!(cpu.cpu.gpr[0] & 0xffff, ax, "{what}");
                assert_eq!(vm.mem.read(ram_at, 1).unwrap(), ram_value, "{what}");
                assert_eq!(device.0[&device_at], device_value as u8, "{what}");
                assert_eq!(run.observation.retired, 2, "{what}");
            }
        }
    }
}

/// With A20 off, a word at FFFF:000F is split at the bus into FFFFFh and 0
/// (spec.md §17.10; decision-22). Each piece is a transfer of its own. Both
/// pieces in devices: a plain or locked operand reads both pieces, then writes
/// both. One RAM piece: the plain RMW writes the device piece first (an engine
/// choice for restartability, spec.md §5.2), in either piece order; the locked
/// forms keep embedder emulation (`UnknownInstruction`), as for an operand partly
/// in a Trap region.
#[test]
fn a20_split_operand_transfers_each_bus_piece() {
    for path in [Path::I286, Path::I286Checked, Path::I386, Path::I386Checked] {
        for (name, bytes) in [
            ("xchg [di],ax", &[0x87, 0x05][..]),
            ("lock add [di],ax", &[0xf0, 0x01, 0x05][..]),
            ("add [di],ax", &[0x01, 0x05][..]),
        ] {
            // Which pieces are devices: (FFFFFh, 0).
            for (high_device, low_device) in [(true, true), (false, true), (true, false)] {
                let what =
                    format!("{path:?} {name}, FFFFFh device {high_device}, 0 device {low_device}");
                let mut vm =
                    Vm::with_backend(VmConfig::flat(0x20_0000), Box::new(InterpreterBackend));
                vm.set_cpu_model(path.model().unwrap());
                let kind = |device| {
                    if device {
                        RegionKind::Trap
                    } else {
                        RegionKind::Ram
                    }
                };
                vm.map(0, 0x1000, Prot::RW, kind(low_device)).unwrap();
                vm.map(0x1000, 0xf_efff, Prot::RWX, RegionKind::Ram)
                    .unwrap();
                vm.map(0xf_ffff, 1, Prot::RW, kind(high_device)).unwrap();
                if !high_device {
                    vm.write_bytes(0xf_ffff, &[0x12]).unwrap();
                }
                if !low_device {
                    vm.write_bytes(0, &[0x34]).unwrap();
                }
                let mut code = bytes.to_vec();
                code.push(0xf4);
                vm.write_bytes(CODE, &code).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.cpu.rip = CODE;
                set_segment(&mut cpu, SegmentId::Ds, 0xffff);
                if matches!(path, Path::I286Checked | Path::I386Checked) {
                    cpu.cpu.legacy.segments[SegmentId::Ss as usize].attributes = 0x92;
                }
                cpu.cpu.gpr[0] = 0xc35a;
                cpu.cpu.gpr[7] = 0x000f;
                cpu.set_a20_enabled(false);
                let mut device = Device(BTreeMap::from([(0, 0x34), (0xf_ffff, 0x12)]));
                let run = drive(&vm, &mut cpu, 64, Some(&mut device));
                let locked = !name.starts_with("add");
                if locked && high_device != low_device {
                    assert!(run.bus.is_empty(), "{what}: {:x?}", run.bus);
                    assert_eq!(
                        run.observation.stop,
                        format!("UnknownInstruction@{CODE:#x}+{}", bytes.len()),
                        "{what}"
                    );
                    assert_eq!(cpu.cpu.gpr[0], 0xc35a, "{what}");
                    continue;
                }
                let old = 0x3412;
                let new = if name.starts_with("xchg") {
                    0xc35a
                } else {
                    (old + 0xc35a) & 0xffff
                };
                let mut expected = Vec::new();
                for (at, is_device) in [(0xf_ffff, high_device), (0, low_device)] {
                    if is_device {
                        expected.push(Bus::Read(at, 1));
                    }
                }
                for (at, is_device, byte) in [
                    (0xf_ffff, high_device, new & 0xff),
                    (0, low_device, new >> 8),
                ] {
                    if is_device {
                        expected.push(Bus::Write(at, 1, byte));
                    }
                }
                assert_eq!(run.bus, expected, "{what}");
                assert_eq!(run.observation.stop, "Hlt", "{what}");
                if !high_device {
                    assert_eq!(vm.mem.read(0xf_ffff, 1).unwrap(), new & 0xff, "{what}");
                }
                if !low_device {
                    assert_eq!(vm.mem.read(0, 1).unwrap(), new >> 8, "{what}");
                }
                let ax = if name.starts_with("xchg") {
                    old
                } else {
                    0xc35a
                };
                assert_eq!(cpu.cpu.gpr[0] & 0xffff, ax, "{what}");
            }
        }
    }
}

/// `Vcpu::run` with a block budget drives the alias through the same two exits.
#[test]
fn alias_run_reads_then_writes_once() {
    for path in [Path::Alias, Path::AliasJit] {
        for case in XCHG.iter().chain(LOCK).chain(&ALU[..4]).chain(&UNARY[..2]) {
            let f = Fixture {
                path,
                window: WINDOW,
                segment: SEG,
                offset: 0x10,
            };
            let what = format!("{path:?} {}", case.name);
            let mut code = vec![0x43];
            code.extend_from_slice(case.bytes);
            code.extend_from_slice(&[0x46, 0xf4]);
            let (vm, mut cpu) = f.setup(false, &code);
            let reference = drive(&vm, &mut cpu, 64, None);
            let ram: Vec<u8> = (WINDOW.0..WINDOW.0 + WINDOW.1)
                .map(|a| vm.mem.read(a, 1).unwrap() as u8)
                .collect();
            let completed = reference.observation.stop == "Hlt";
            let expected = if completed {
                expected_bus(&f, case, &|a| ram[(a - WINDOW.0) as usize])
            } else {
                Vec::new()
            };
            for budget in [Some(1), Some(2), None] {
                let (vm, mut cpu) = f.setup(true, &code);
                let mut device = Device::new(WINDOW);
                let mut bus = Vec::new();
                let stop = loop {
                    match cpu.run(&vm, budget) {
                        Exit::BudgetExhausted => {}
                        Exit::MmioRead { addr, size, .. } => {
                            bus.push(Bus::Read(addr, size));
                            cpu.complete_mmio_read(device.read(addr, size));
                        }
                        Exit::MmioWrite {
                            addr, size, value, ..
                        } => {
                            bus.push(Bus::Write(addr, size, value));
                            device.write(addr, size, value);
                            cpu.complete_mmio_write();
                        }
                        other => break other,
                    }
                    assert!(bus.len() <= 8, "{what}: {bus:x?}");
                };
                assert_eq!(bus, expected, "{what}, budget {budget:?}");
                assert_eq!(matches!(stop, Exit::Hlt), completed, "{what}: {stop:?}");
                let mut gpr = [0; 8];
                gpr.copy_from_slice(&cpu.cpu.gpr[..8]);
                assert_eq!(gpr, reference.observation.gpr, "{what}");
                assert_eq!(
                    cpu.retired_instructions(),
                    reference.observation.retired,
                    "{what}"
                );
            }
        }
    }
}

/// The alias superset's CMPXCHG8B (an 8-byte paired compare-exchange) keeps
/// embedder emulation on a device operand: `Exit::UnknownInstruction` with the
/// instruction's bytes, no transfer and no state change (spec.md §5.2).
#[test]
fn alias_cmpxchg8b_keeps_embedder_emulation() {
    for path in [Path::Alias, Path::AliasJit] {
        for code in [&[0x0f, 0xc7, 0x0d][..], &[0xf0, 0x26, 0x0f, 0xc7, 0x0d][..]] {
            let f = Fixture {
                path,
                window: WINDOW,
                segment: SEG,
                offset: 0x10,
            };
            let what = format!("{path:?} {code:02x?}");
            let (vm, mut cpu) = f.setup(true, &[code, &[0xf4]].concat());
            let before = cpu.cpu.gpr;
            let mut device = Device::new(WINDOW);
            let run = drive(&vm, &mut cpu, 64, Some(&mut device));
            assert!(run.bus.is_empty(), "{what}: {:x?}", run.bus);
            assert_eq!(
                run.observation.stop,
                format!("UnknownInstruction@{CODE:#x}+{}", code.len()),
                "{what}"
            );
            assert_eq!(cpu.cpu.gpr, before, "{what}");
            assert_eq!(cpu.cpu.rip, CODE, "{what}");
        }
    }
}

/// Long64 and Compat32 share the retry path: an ordinary read-modify-write on a
/// Trap operand reads once and writes once at every tier, because compiled code
/// defers the MMIO instruction to the interpreter (spec.md §5.2). Locked and
/// implicitly locked operands keep the `UnknownInstruction` policy there
/// (`locked_mmio.rs`).
#[test]
fn application_modes_read_then_write_once_at_every_tier() {
    const AT: u64 = 0x3010;
    let cases: &[(CpuMode, &[u8], u8, Kind)] = &[
        (CpuMode::Long64, &[0x00, 0x07], 1, Kind::Rmw),
        (CpuMode::Long64, &[0x01, 0x07], 4, Kind::Rmw),
        (CpuMode::Long64, &[0x66, 0x01, 0x07], 2, Kind::Rmw),
        (CpuMode::Long64, &[0x48, 0x01, 0x07], 8, Kind::Rmw),
        (CpuMode::Long64, &[0x08, 0x07], 1, Kind::Rmw),
        (CpuMode::Long64, &[0x11, 0x07], 4, Kind::Rmw),
        (CpuMode::Long64, &[0x80, 0x37, 0x96], 1, Kind::Rmw),
        (CpuMode::Long64, &[0xfe, 0x07], 1, Kind::Rmw),
        (CpuMode::Long64, &[0xf7, 0x1f], 4, Kind::Rmw),
        (CpuMode::Long64, &[0xd3, 0x27], 4, Kind::Rmw),
        (CpuMode::Long64, &[0x0f, 0xab, 0x37], 1, Kind::Rmw),
        (CpuMode::Long64, &[0x0f, 0xba, 0x2f, 0x03], 4, Kind::Rmw),
        (CpuMode::Long64, &[0x39, 0x07], 4, Kind::Read),
        (CpuMode::Compat32, &[0x00, 0x07], 1, Kind::Rmw),
        (CpuMode::Compat32, &[0x66, 0xff, 0x07], 2, Kind::Rmw),
        (CpuMode::Compat32, &[0xd1, 0x17], 4, Kind::Rmw),
        (CpuMode::Compat32, &[0x83, 0x2f, 0xf1], 4, Kind::Rmw),
    ];
    let setup = |tier: usize, mode: CpuMode, trap: bool, code: &[u8]| {
        let backend: Box<dyn Backend> = match tier {
            0 => Box::new(InterpreterBackend),
            1 => Box::new(JitBackend::new()),
            _ => Box::new(JitBackend::with_superblocks(RegionCaps {
                max_blocks: 8,
                max_icount: 64,
            })),
        };
        let mut vm = Vm::with_backend(VmConfig::flat(0x5000), backend);
        vm.set_cpu_mode(mode);
        vm.set_tier_up_after(None);
        vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        let kind = if trap {
            RegionKind::Trap
        } else {
            RegionKind::Ram
        };
        vm.map(0x3000, 0x1000, Prot::RW, kind).unwrap();
        if !trap {
            let bytes: Vec<u8> = (0x3000..0x4000).map(seed).collect();
            vm.write_bytes(0x3000, &bytes).unwrap();
        }
        // INC EBX ; <case> ; INC ESI ; HLT (FF /0 keeps one encoding in both modes)
        let mut bytes = vec![0xff, 0xc3];
        bytes.extend_from_slice(code);
        bytes.extend_from_slice(&[0xff, 0xc6, 0xf4]);
        vm.write_bytes(0x1000, &bytes).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[0] = 0x8899_aabb_c35a;
        cpu.cpu.gpr[1] = 3;
        cpu.cpu.gpr[7] = AT;
        cpu.cpu.flags.cf = true;
        (vm, cpu)
    };
    let mut failures = Vec::new();
    for tier in 0..3 {
        for &(mode, code, size, kind) in cases {
            let what = format!("tier {tier} {mode:?} {code:02x?}");
            let (vm, mut cpu) = setup(tier, mode, false, code);
            assert!(matches!(cpu.run(&vm, Some(64)), Exit::Hlt), "{what}");
            // Retirement counts are the Real16/Machine contract (§17.6); compiled
            // Long64 code does not count them, so they are not compared here.
            let reference = (cpu.cpu.gpr, cpu.cpu.flags, cpu.cpu.rip);
            let value = vm.mem.read(AT, size).unwrap();
            let mut expected = vec![Bus::Read(AT, size)];
            if kind == Kind::Rmw {
                expected.push(Bus::Write(AT, size, value));
            }
            for budget in [Some(1), Some(2), Some(64), None] {
                let (vm, mut cpu) = setup(tier, mode, true, code);
                let mut device = Device::new((0x3000, 0x1000));
                let mut bus = Vec::new();
                let stop = loop {
                    match cpu.run(&vm, budget) {
                        Exit::BudgetExhausted => {}
                        Exit::MmioRead { addr, size, .. } => {
                            bus.push(Bus::Read(addr, size));
                            cpu.complete_mmio_read(device.read(addr, size));
                        }
                        Exit::MmioWrite {
                            addr, size, value, ..
                        } => {
                            bus.push(Bus::Write(addr, size, value));
                            device.write(addr, size, value);
                            cpu.complete_mmio_write();
                        }
                        other => break other,
                    }
                    assert!(bus.len() <= 8, "{what}: {bus:x?}");
                };
                let observed = (cpu.cpu.gpr, cpu.cpu.flags, cpu.cpu.rip);
                if bus != expected || !matches!(stop, Exit::Hlt) || observed != reference {
                    failures.push(format!(
                        "{what} budget {budget:?}: {bus:x?} then {stop:?}, expected {expected:x?}; \
                         state {observed:x?}, RAM run {reference:x?}"
                    ));
                    break;
                }
            }
        }
        // XCHG with memory keeps embedder emulation in these modes.
        let (vm, mut cpu) = setup(tier, CpuMode::Long64, true, &[0x87, 0x07]);
        assert!(
            matches!(
                cpu.run(&vm, Some(64)),
                Exit::UnknownInstruction { addr: 0x1002, .. }
            ),
            "tier {tier}"
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ---------------------------------------------------------------------------
// Review witnesses (xj-ir6.33 review): the CMPXCHG success path, interrupt and
// trace boundaries across the two exits, the answer binding and its lifetime.
// ---------------------------------------------------------------------------

mod review {
    use super::*;
    use x86jit_core::{StepOptions, StepStop};

    fn real16(model: Option<CpuModel>, trap: (u64, usize), ram_extra: Option<(u64, usize)>) -> Vm {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10_0000), Box::new(InterpreterBackend));
        match model {
            Some(m) => vm.set_cpu_model(m),
            None => vm.set_cpu_mode(CpuMode::Real16),
        }
        vm.map(0, 0xa_0000, Prot::RWX, RegionKind::Ram).unwrap();
        if let Some((a, n)) = ram_extra {
            vm.map(a, n, Prot::RW, RegionKind::Ram).unwrap();
        }
        vm.map(trap.0, trap.1, Prot::RW, RegionKind::Trap).unwrap();
        for v in 0..32u64 {
            vm.mem.write(v * 4, 0x500, 4).unwrap();
        }
        vm.write_bytes(0x500, &[0xf4]).unwrap();
        vm
    }

    fn vcpu(vm: &Vm, es: u16, di: u64) -> Vcpu {
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = CODE;
        set_segment(&mut cpu, SegmentId::Cs, 0);
        set_segment(&mut cpu, SegmentId::Ss, 0);
        set_segment(&mut cpu, SegmentId::Ds, es);
        set_segment(&mut cpu, SegmentId::Es, es);
        cpu.cpu.gpr[4] = 0x7000;
        cpu.cpu.gpr[7] = di;
        cpu
    }

    /// Drive `Vcpu::run`, a device answering `answer(addr, size)`.
    fn run_device(
        vm: &Vm,
        cpu: &mut Vcpu,
        mut answer: impl FnMut(u64, u8) -> u64,
    ) -> (Vec<Bus>, Exit) {
        let mut bus = Vec::new();
        loop {
            match cpu.run(vm, Some(64)) {
                Exit::BudgetExhausted => {}
                Exit::MmioRead { addr, size, .. } => {
                    bus.push(Bus::Read(addr, size));
                    cpu.complete_mmio_read(answer(addr, size));
                }
                Exit::MmioWrite {
                    addr, size, value, ..
                } => {
                    bus.push(Bus::Write(addr, size, value));
                    cpu.complete_mmio_write();
                }
                other => return (bus, other),
            }
            assert!(bus.len() < 16, "{bus:x?}");
        }
    }

    /// CMPXCHG on a device operand whose comparison succeeds writes the source,
    /// with ZF set and the accumulator kept (SDM Vol 2A §3.3 CMPXCHG).
    #[test]
    fn alias_cmpxchg_success_writes_source() {
        for (code, size, acc, src) in [
            (&[0x26, 0x0f, 0xb0, 0x0d, 0xf4][..], 1u8, 0x5au64, 0x03u64),
            (&[0xf0, 0x26, 0x0f, 0xb1, 0x0d, 0xf4][..], 2, 0xc35a, 0x0503),
        ] {
            let vm = real16(None, (0xb_8000, 0x8000), None);
            vm.write_bytes(CODE, code).unwrap();
            let mut cpu = vcpu(&vm, 0xb800, 0x10);
            cpu.cpu.gpr[0] = acc;
            cpu.cpu.gpr[1] = src;
            let (bus, stop) = run_device(&vm, &mut cpu, |_, _| acc);
            assert!(matches!(stop, Exit::Hlt), "{stop:?}");
            assert_eq!(
                bus,
                [Bus::Read(0xb_8010, size), Bus::Write(0xb_8010, size, src)],
                "{code:02x?}"
            );
            assert!(cpu.cpu.flags.zf);
            assert_eq!(cpu.cpu.gpr[0], acc);
        }
    }

    /// An IRQ injected after the read is answered is not accepted before the
    /// write; it is taken once the instruction completes.
    #[test]
    fn alias_irq_waits_for_the_write() {
        let vm = real16(None, (0xb_8000, 0x8000), None);
        vm.write_bytes(CODE, &[0xf0, 0x26, 0x00, 0x05, 0xf4])
            .unwrap();
        vm.mem.write(0x20 * 4, 0x600, 4).unwrap();
        vm.write_bytes(0x600, &[0xf4]).unwrap();
        let mut cpu = vcpu(&vm, 0xb800, 0x10);
        cpu.cpu.flags.if_ = true;
        assert!(matches!(cpu.run(&vm, Some(64)), Exit::MmioRead { .. }));
        cpu.complete_mmio_read(0x11);
        cpu.inject_irq(0x20);
        assert!(!cpu.irq_ready());
        let e = cpu.run(&vm, Some(64));
        assert!(matches!(e, Exit::MmioWrite { value: 0x11, .. }), "{e:?}");
        assert!(!cpu.irq_ready());
        cpu.complete_mmio_write();
        let e = cpu.run(&vm, Some(64));
        assert!(matches!(e, Exit::Hlt), "{e:?}");
        assert_eq!(cpu.cpu.rip, 0x601, "the handler's HLT");
    }

    /// Long64: a device read answered, then a RAM fault of the same instruction.
    /// The fault ends the answer, as a fault aborts the instruction on the
    /// processor: the evidence reports no acknowledged transfer, and the
    /// re-execution after the embedder maps the stack reads the device again.
    #[test]
    fn long64_fault_ends_the_answer() {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(InterpreterBackend));
        vm.set_cpu_mode(CpuMode::Long64);
        vm.set_tier_up_after(None);
        vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        // push qword [rdi] ; hlt
        vm.write_bytes(0x1000, &[0xff, 0x37, 0xf4]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[7] = 0x3010;
        cpu.cpu.gpr[4] = 0x6000; // the stack page is unmapped
        let mut log = Vec::new();
        for _ in 0..6 {
            let e = cpu.run(&vm, Some(64));
            match &e {
                Exit::MmioRead { addr, .. } => {
                    log.push(format!("R {addr:#x}"));
                    cpu.complete_mmio_read(0xdead_beef);
                }
                Exit::UnmappedMemory { addr, .. } => {
                    let acknowledged = e
                        .fault_metadata()
                        .map(|m| m.progress().acknowledged_transfers().to_vec());
                    log.push(format!("U {addr:#x} {acknowledged:x?}"));
                    vm.map(0x5000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
                }
                other => {
                    log.push(format!("{other:?}"));
                    break;
                }
            }
        }
        assert_eq!(
            log,
            ["R 0x3010", "U 0x5ff8 Some([])", "R 0x3010", "Hlt"],
            "{log:?}"
        );
        assert_eq!(vm.mem.read(0x5ff8, 8).unwrap(), 0xdead_beef);
    }

    /// Real16: a device divisor of zero raises #DE after the read is answered.
    /// The exception ends the answer: it is not reported as acknowledged, and the
    /// DIV run again reads the device again.
    #[test]
    fn real16_exception_ends_the_answer() {
        let vm = real16(None, (0xb_8000, 0x8000), None);
        // div byte es:[di] ; hlt
        vm.write_bytes(CODE, &[0x26, 0xf6, 0x35, 0xf4]).unwrap();
        let mut cpu = vcpu(&vm, 0xb800, 0x10);
        cpu.cpu.gpr[0] = 0x0010;
        let (bus, stop) = run_device(&vm, &mut cpu, |_, _| 0);
        assert_eq!(bus, [Bus::Read(0xb_8010, 1)]);
        let Exit::Exception { vector: 0, .. } = &stop else {
            panic!("{stop:?}")
        };
        assert_eq!(
            stop.fault_metadata()
                .unwrap()
                .progress()
                .acknowledged_transfers(),
            &[]
        );
        assert_eq!(cpu.cpu.rip, CODE);
        let (bus, stop) = run_device(&vm, &mut cpu, |_, _| 2);
        assert!(matches!(stop, Exit::Hlt), "{stop:?}");
        assert_eq!(bus, [Bus::Read(0xb_8010, 1)]);
        assert_eq!(cpu.cpu.gpr[0] & 0xffff, 0x0008, "AL = 10h / 2, AH = 0");
    }

    /// The code at RIP changes between the write exit and its acknowledgement.
    /// The new instruction at the same RIP must not take the old instruction's
    /// recorded read: it reads the device itself.
    #[test]
    fn alias_code_write_between_exits() {
        let vm = real16(None, (0xb_8000, 0x8000), None);
        // add es:[di],al ; hlt
        vm.write_bytes(CODE, &[0x26, 0x00, 0x05, 0xf4]).unwrap();
        let mut cpu = vcpu(&vm, 0xb800, 0x10);
        cpu.cpu.gpr[0] = 0x01;
        let mut log = Vec::new();
        let mut rewritten = false;
        for _ in 0..8 {
            let e = cpu.run(&vm, Some(64));
            match e {
                Exit::MmioRead { addr, .. } => {
                    log.push(format!("R {addr:#x}"));
                    cpu.complete_mmio_read(if rewritten { 0x99 } else { 0x40 });
                }
                Exit::MmioWrite { addr, value, .. } => {
                    log.push(format!("W {addr:#x} {value:#x}"));
                    if !rewritten {
                        // mov al,es:[di]
                        vm.write_bytes(CODE, &[0x26, 0x8a, 0x05]).unwrap();
                        rewritten = true;
                    }
                    cpu.complete_mmio_write();
                }
                other => {
                    log.push(format!("{other:?} AL={:#x}", cpu.cpu.gpr[0] & 0xff));
                    break;
                }
            }
        }
        assert_eq!(
            log,
            ["R 0xb8010", "W 0xb8010 0x41", "R 0xb8010", "Hlt AL=0x99"],
            "{log:?}"
        );
    }

    fn shadow_vm(model: Option<CpuModel>, trap: bool) -> Vm {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10_0000), Box::new(InterpreterBackend));
        match model {
            Some(m) => vm.set_cpu_model(m),
            None => vm.set_cpu_mode(CpuMode::Real16),
        }
        vm.map(0, 0xa_0000, Prot::RWX, RegionKind::Ram).unwrap();
        let kind = if trap {
            RegionKind::Trap
        } else {
            RegionKind::Ram
        };
        vm.map(0xb_8000, 0x8000, Prot::RW, kind).unwrap();
        if !trap {
            vm.write_bytes(0xb_8010, &[0x40, 0x41]).unwrap();
        }
        for v in 0..0x21u64 {
            vm.mem.write(v * 4, 0x500 + v, 4).unwrap();
            vm.write_bytes(0x500 + v, &[0xf4]).unwrap();
        }
        vm
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Outcome {
        stop: String,
        rip: u64,
        sp: u64,
        frame: u64,
        ax: u64,
        retired: u64,
    }

    fn outcome(
        model: Option<CpuModel>,
        trap: bool,
        code: &[u8],
        prepare: &dyn Fn(&mut Vcpu),
        stepped: bool,
    ) -> (Vec<Bus>, Outcome) {
        let vm = shadow_vm(model, trap);
        vm.write_bytes(CODE, code).unwrap();
        let mut cpu = vcpu(&vm, 0xb800, 0x10);
        cpu.cpu.gpr[0] = 0x5a;
        prepare(&mut cpu);
        let mut bus = Vec::new();
        let mut dev = [0x40u8, 0x41];
        let stop = loop {
            let e = if stepped {
                match cpu
                    .run_steps(&vm, 1, StepOptions::default())
                    .unwrap()
                    .stop()
                {
                    StepStop::Exit(e) => e.clone(),
                    other => panic!("{other:?}"),
                }
            } else {
                cpu.run(&vm, Some(64))
            };
            match e {
                Exit::BudgetExhausted => {}
                Exit::MmioRead { addr, size, .. } => {
                    bus.push(Bus::Read(addr, size));
                    let i = (addr - 0xb_8010) as usize;
                    let v =
                        (0..size as usize).fold(0u64, |v, k| v | u64::from(dev[i + k]) << (8 * k));
                    cpu.complete_mmio_read(v);
                }
                Exit::MmioWrite {
                    addr, size, value, ..
                } => {
                    bus.push(Bus::Write(addr, size, value));
                    let i = (addr - 0xb_8010) as usize;
                    for k in 0..size as usize {
                        dev[i + k] = (value >> (8 * k)) as u8;
                    }
                    cpu.complete_mmio_write();
                }
                other => break format!("{other:?}"),
            }
            assert!(bus.len() < 16);
        };
        let sp = cpu.cpu.gpr[4] & 0xffff;
        let frame = vm.mem.read(sp, 6).unwrap_or(0);
        (
            bus,
            Outcome {
                stop,
                rip: cpu.cpu.rip,
                sp,
                frame,
                ax: cpu.cpu.gpr[0] & 0xffff,
                retired: cpu.retired_instructions(),
            },
        )
    }

    /// TF single-step, the STI shadow and the MOV SS shadow around a device RMW:
    /// the Trap run must match the RAM run apart from the transfers, with
    /// `Vcpu::run` and with `run_steps(1)`.
    #[test]
    fn trace_and_shadows_around_device_rmw() {
        let paths: [(&str, Option<CpuModel>); 4] = [
            ("alias", None),
            ("i8086", Some(CpuModel::I8086)),
            ("i286", Some(CpuModel::I80286)),
            ("i386", Some(CpuModel::I80386)),
        ];
        let cases: [(&str, &[u8]); 3] = [
            ("xchg", &[0x26, 0x86, 0x05]),
            ("add", &[0x26, 0x00, 0x05]),
            ("lock add", &[0xf0, 0x26, 0x00, 0x05]),
        ];
        let mut bad = Vec::new();
        let mut checked = 0;
        for (pname, model) in paths {
            for (cname, case) in cases {
                for (setup, stepped) in [
                    ("tf", false),
                    ("tf", true),
                    ("sti", false),
                    ("sti", true),
                    ("movss", false),
                    ("movss", true),
                ] {
                    let mut code = match setup {
                        "sti" => vec![0xfb],
                        "movss" => vec![0x8e, 0xd3], // mov ss,bx
                        _ => vec![],
                    };
                    code.extend_from_slice(case);
                    code.extend_from_slice(&[0x90, 0xf4]);
                    let prepare = |cpu: &mut Vcpu| match setup {
                        "tf" => {
                            cpu.cpu.rflags_system |= 0x100;
                        }
                        "sti" => {
                            cpu.cpu.flags.if_ = false;
                            cpu.inject_irq(0x20);
                        }
                        _ => {
                            cpu.cpu.flags.if_ = true;
                            cpu.cpu.gpr[3] = 0;
                            cpu.inject_irq(0x20);
                        }
                    };
                    let (rb, ro) = outcome(model, false, &code, &prepare, stepped);
                    let (tb, to) = outcome(model, true, &code, &prepare, stepped);
                    let line = format!(
                        "{pname} {cname} {setup} stepped={stepped}: RAM {ro:x?} | Trap {to:x?} bus {tb:x?}"
                    );
                    println!("{line}");
                    // Known, not caused by the device protocol: `Vcpu::run` on the
                    // alias accepts interrupts at block boundaries. The RAM run's one
                    // block ends at its HLT with the interrupt still pending; the MMIO
                    // stop ends the Trap run's block, so its interrupt is taken after
                    // the device RMW, once the STI shadow has ended.
                    let block_granular = pname == "alias" && !stepped && setup == "sti";
                    if !rb.is_empty() || (ro != to && !block_granular) {
                        bad.push(line);
                    }
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 72);
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    /// What the driver does after a device stop, before the retry runs.
    #[derive(Clone, Copy, Debug)]
    enum Call {
        Run,
        /// `run(Some(0))` after the completion, then `run`.
        ZeroBudget,
        /// `run_steps(0)` after the completion, then `run`.
        ZeroSteps,
        /// A host stop requested after the completion, then `run`.
        HostStop,
        /// A host stop requested and `run` called before the completion.
        HostStopFirst,
    }

    std::thread_local! {
        static HOST_STOPS_BEFORE_RETRY: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    /// Drive `Vcpu::run` to a stop other than a device, budget or host stop.
    /// `schedule[i % len]` is applied after the i-th device stop. Answers count up
    /// from 40h, so a replayed read also shows in the written value.
    fn drive_schedule(vm: &Vm, cpu: &mut Vcpu, schedule: &[Call]) -> (Vec<Bus>, Exit) {
        let mut bus = Vec::new();
        let mut answer = 0x40;
        let mut stops = 0;
        let mut exit = cpu.run(vm, Some(64));
        loop {
            let how = schedule[stops % schedule.len()];
            match exit {
                Exit::BudgetExhausted | Exit::Interrupted => {
                    exit = cpu.run(vm, Some(64));
                    continue;
                }
                Exit::MmioRead { .. } | Exit::MmioWrite { .. } => {
                    if matches!(how, Call::HostStopFirst) {
                        // The stop is taken at the boundary before the retry, or
                        // the unanswered request is repeated and the stop stays
                        // requested for a later boundary.
                        cpu.exit_request().request_exit();
                        let stop = cpu.run(vm, Some(64));
                        if matches!(stop, Exit::Interrupted) {
                            HOST_STOPS_BEFORE_RETRY.with(|n| n.set(n.get() + 1));
                        } else {
                            assert_eq!(format!("{stop:?}"), format!("{exit:?}"));
                        }
                    }
                    match exit {
                        Exit::MmioRead { addr, size, .. } => {
                            bus.push(Bus::Read(addr, size));
                            cpu.complete_mmio_read(answer);
                            answer += 0x10;
                        }
                        Exit::MmioWrite {
                            addr, size, value, ..
                        } => {
                            bus.push(Bus::Write(addr, size, value));
                            cpu.complete_mmio_write();
                        }
                        _ => unreachable!(),
                    }
                }
                other => return (bus, other),
            }
            stops += 1;
            assert!(bus.len() < 8, "{bus:x?}");
            match how {
                Call::ZeroBudget => {
                    let stop = cpu.run(vm, Some(0));
                    assert!(matches!(stop, Exit::BudgetExhausted), "{stop:?}");
                }
                Call::ZeroSteps => {
                    let run = cpu.run_steps(vm, 0, StepOptions::default()).unwrap();
                    assert!(
                        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                        "{:?}",
                        run.stop()
                    );
                    assert_eq!(run.steps(), 0);
                }
                Call::HostStop => cpu.exit_request().request_exit(),
                Call::Run | Call::HostStopFirst => {}
            }
            exit = cpu.run(vm, Some(64));
        }
    }

    /// A budget stop or a host stop between the two exits is taken at the
    /// boundary before the retry: the instruction has neither completed nor
    /// aborted, so its recorded answer stays and the device is read once
    /// (spec.md §5.2). Zero budgets are the established way to collect deferred
    /// exits. Long64 has no `run_steps` (UnsupportedMode), so its schedules use
    /// `run` only. Without the answer kept across the host stop, Long64 reads
    /// the device again after the write (R, W, R; measured with that path
    /// disabled).
    #[test]
    fn boundary_stops_between_the_exits_keep_the_answer() {
        use Call::*;
        let schedules: &[&[Call]] = &[
            &[ZeroBudget, Run],
            &[Run, ZeroBudget],
            &[ZeroBudget],
            &[ZeroSteps, Run],
            &[Run, ZeroSteps],
            &[ZeroSteps],
            &[HostStop],
            &[HostStopFirst, Run],
            &[Run, HostStopFirst],
            &[HostStopFirst],
        ];
        let mut bad = Vec::new();
        let mut checked = 0;
        // inc bx ; <op> es:[di],al ; inc si ; hlt
        for (name, code, written, al) in [
            ("add", &[0x43, 0x26, 0x00, 0x05, 0x46, 0xf4][..], 0x41, 0x01),
            (
                "lock add",
                &[0x43, 0xf0, 0x26, 0x00, 0x05, 0x46, 0xf4][..],
                0x41,
                0x01,
            ),
            (
                "xchg",
                &[0x43, 0x26, 0x86, 0x05, 0x46, 0xf4][..],
                0x01,
                0x40,
            ),
        ] {
            for schedule in schedules {
                let vm = real16(None, (0xb_8000, 0x8000), None);
                vm.write_bytes(CODE, code).unwrap();
                let mut cpu = vcpu(&vm, 0xb800, 0x10);
                cpu.cpu.gpr[0] = 0x01;
                let (bus, stop) = drive_schedule(&vm, &mut cpu, schedule);
                checked += 1;
                let al_now = cpu.cpu.gpr[0] & 0xff;
                if bus != [Bus::Read(0xb_8010, 1), Bus::Write(0xb_8010, 1, written)]
                    || !matches!(stop, Exit::Hlt)
                    || al_now != al
                {
                    bad.push(format!(
                        "alias {name} {schedule:?}: {bus:x?} {stop:?} AL={al_now:#x}"
                    ));
                }
            }
        }
        // The alias walk keeps a requested host stop while the retry is open: the
        // unanswered request is repeated instead.
        assert_eq!(HOST_STOPS_BEFORE_RETRY.with(|n| n.get()), 0);
        // inc ebx ; add [rdi],al ; inc esi ; hlt, at the interpreter, block and
        // region tiers, with every tier-up setting.
        for tier in 0..3 {
            for tier_up in [None, Some(1), Some(2), Some(3)] {
                for schedule in schedules
                    .iter()
                    .filter(|s| !s.iter().any(|c| matches!(c, ZeroSteps)))
                {
                    let backend: Box<dyn Backend> = match tier {
                        0 => Box::new(InterpreterBackend),
                        1 => Box::new(JitBackend::new()),
                        _ => Box::new(JitBackend::with_superblocks(RegionCaps {
                            max_blocks: 8,
                            max_icount: 64,
                        })),
                    };
                    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
                    vm.set_cpu_mode(CpuMode::Long64);
                    vm.set_tier_up_after(tier_up);
                    vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
                    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                    vm.write_bytes(0x1000, &[0xff, 0xc3, 0x00, 0x07, 0xff, 0xc6, 0xf4])
                        .unwrap();
                    let mut cpu = vm.new_vcpu();
                    cpu.cpu.rip = 0x1000;
                    cpu.cpu.gpr[0] = 0x01;
                    cpu.cpu.gpr[7] = 0x3010;
                    let (bus, stop) = drive_schedule(&vm, &mut cpu, schedule);
                    checked += 1;
                    if bus != [Bus::Read(0x3010, 1), Bus::Write(0x3010, 1, 0x41)]
                        || !matches!(stop, Exit::Hlt)
                    {
                        bad.push(format!(
                            "long64 tier {tier} tier-up {tier_up:?} {schedule:?}: {bus:x?} {stop:?}"
                        ));
                    }
                }
            }
        }
        assert_eq!(checked, 3 * 10 + 3 * 4 * 7);
        // Long64 takes the host stop at the dispatch boundary before the retry:
        // once in each of the two one-sided schedules and twice in the other,
        // in all twelve tier settings.
        assert_eq!(HOST_STOPS_BEFORE_RETRY.with(|n| n.get()), 12 * 4);
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    /// Run until a stop other than a budget or host stop.
    fn run_on(vm: &Vm, cpu: &mut Vcpu) -> Exit {
        for _ in 0..8 {
            match cpu.run(vm, Some(64)) {
                Exit::BudgetExhausted | Exit::Interrupted => {}
                other => return other,
            }
        }
        panic!("no stop")
    }

    /// A Long64 vCPU at `tier` (interpreter, block, region) with tier-up
    /// `tier_up`: code at 1000h, a Trap window at 3000h, RDI = 3010h, AL = 1.
    fn long64_vm(tier: u8, tier_up: Option<u32>, code: &[u8]) -> (Vm, Vcpu) {
        let backend: Box<dyn Backend> = match tier {
            0 => Box::new(InterpreterBackend),
            1 => Box::new(JitBackend::new()),
            _ => Box::new(JitBackend::with_superblocks(RegionCaps {
                max_blocks: 8,
                max_icount: 64,
            })),
        };
        let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
        vm.set_cpu_mode(CpuMode::Long64);
        vm.set_tier_up_after(tier_up);
        vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.write_bytes(0x1000, code).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[0] = 0x01;
        cpu.cpu.gpr[7] = 0x3010;
        (vm, cpu)
    }

    /// Answer every device stop until another stop; the transfers in order.
    fn drain(vm: &Vm, cpu: &mut Vcpu, bus: &mut Vec<Bus>) -> Exit {
        loop {
            match run_on(vm, cpu) {
                Exit::MmioRead { addr, size, .. } => {
                    bus.push(Bus::Read(addr, size));
                    cpu.complete_mmio_read(0x99);
                }
                Exit::MmioWrite {
                    addr, size, value, ..
                } => {
                    bus.push(Bus::Write(addr, size, value));
                    cpu.complete_mmio_write();
                }
                other => return other,
            }
            assert!(bus.len() < 6, "{bus:x?}");
        }
    }

    /// The device window is remapped as RAM holding 77h after the write exit of
    /// ADD [mem],AL, before the retry. The recorded device answer no longer
    /// stands: the retry computes from RAM (77h + 1 = 78h), as at b2ac76e4.
    /// With and without a host stop before the retry.
    #[test]
    fn device_remapped_as_ram_between_the_exits() {
        let mut bad = Vec::new();
        for host_stop in [false, true] {
            // Long64: inc ebx ; add [rdi],al ; inc esi ; hlt
            for tier in 0..3 {
                for tier_up in [None, Some(1), Some(2), Some(3)] {
                    let code = [0xff, 0xc3, 0x00, 0x07, 0xff, 0xc6, 0xf4];
                    let (mut vm, mut cpu) = long64_vm(tier, tier_up, &code);
                    let stop = run_on(&vm, &mut cpu);
                    assert!(
                        matches!(stop, Exit::MmioRead { addr: 0x3010, .. }),
                        "{stop:?}"
                    );
                    cpu.complete_mmio_read(0x40);
                    let stop = run_on(&vm, &mut cpu);
                    assert!(
                        matches!(stop, Exit::MmioWrite { value: 0x41, .. }),
                        "{stop:?}"
                    );
                    if host_stop {
                        cpu.exit_request().request_exit();
                        let stop = cpu.run(&vm, Some(64));
                        assert!(matches!(stop, Exit::Interrupted), "{stop:?}");
                    }
                    cpu.complete_mmio_write();
                    vm.mem.unmap(0x3000, 0x1000).unwrap();
                    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
                    vm.mem.write(0x3010, 0x77, 1).unwrap();
                    let mut bus = Vec::new();
                    let stop = drain(&vm, &mut cpu, &mut bus);
                    let byte = vm.mem.read(0x3010, 1).unwrap();
                    if !bus.is_empty() || !matches!(stop, Exit::Hlt) || byte != 0x78 {
                        bad.push(format!(
                            "long64 tier {tier} {tier_up:?} host stop {host_stop}: \
                             {bus:x?} {stop:?} {byte:#x}"
                        ));
                    }
                }
            }
            // Alias: inc bx ; add es:[di],al ; inc si ; hlt
            let mut vm = real16(None, (0xb_8000, 0x8000), None);
            vm.write_bytes(CODE, &[0x43, 0x26, 0x00, 0x05, 0x46, 0xf4])
                .unwrap();
            let mut cpu = vcpu(&vm, 0xb800, 0x10);
            cpu.cpu.gpr[0] = 0x01;
            let stop = run_on(&vm, &mut cpu);
            assert!(
                matches!(stop, Exit::MmioRead { addr: 0xb_8010, .. }),
                "{stop:?}"
            );
            cpu.complete_mmio_read(0x40);
            let stop = run_on(&vm, &mut cpu);
            assert!(
                matches!(stop, Exit::MmioWrite { value: 0x41, .. }),
                "{stop:?}"
            );
            if host_stop {
                // The alias repeats the unanswered request instead.
                cpu.exit_request().request_exit();
                let stop = cpu.run(&vm, Some(64));
                assert!(
                    matches!(stop, Exit::MmioWrite { value: 0x41, .. }),
                    "{stop:?}"
                );
            }
            cpu.complete_mmio_write();
            vm.mem.unmap(0xb_8000, 0x8000).unwrap();
            vm.map(0xb_8000, 0x8000, Prot::RW, RegionKind::Ram).unwrap();
            vm.mem.write(0xb_8010, 0x77, 1).unwrap();
            let mut bus = Vec::new();
            let stop = drain(&vm, &mut cpu, &mut bus);
            let byte = vm.mem.read(0xb_8010, 1).unwrap();
            if !bus.is_empty() || !matches!(stop, Exit::Hlt) || byte != 0x78 {
                bad.push(format!(
                    "alias host stop {host_stop}: {bus:x?} {stop:?} {byte:#x}"
                ));
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    /// The reverse: the stack, RAM at the read exit of PUSH [mem], is remapped as
    /// Trap before the retry. The retry stops at the stack write, and the retry
    /// after its acknowledgement reuses the device answer, whose access still
    /// stops at Trap: one read, then one write of the value read.
    #[test]
    fn ram_remapped_as_device_between_the_exits() {
        let mut bad = Vec::new();
        // Long64: push qword [rdi] ; hlt, stack page 5000h.
        for tier in 0..3 {
            for tier_up in [None, Some(1), Some(2), Some(3)] {
                let (mut vm, mut cpu) = long64_vm(tier, tier_up, &[0xff, 0x37, 0xf4]);
                vm.map(0x5000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
                cpu.cpu.gpr[4] = 0x6000;
                let mut bus = Vec::new();
                let Exit::MmioRead { addr, size, .. } = run_on(&vm, &mut cpu) else {
                    panic!("no read exit")
                };
                bus.push(Bus::Read(addr, size));
                cpu.complete_mmio_read(0x1122_3344_5566_7788);
                vm.mem.unmap(0x5000, 0x1000).unwrap();
                vm.map(0x5000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                let stop = drain(&vm, &mut cpu, &mut bus);
                if bus
                    != [
                        Bus::Read(0x3010, 8),
                        Bus::Write(0x5ff8, 8, 0x1122_3344_5566_7788),
                    ]
                    || !matches!(stop, Exit::Hlt)
                    || cpu.cpu.gpr[4] != 0x5ff8
                {
                    bad.push(format!(
                        "long64 tier {tier} {tier_up:?}: {bus:x?} {stop:?} rsp {:#x}",
                        cpu.cpu.gpr[4]
                    ));
                }
            }
        }
        // Alias: push word es:[di] ; hlt, stack C000:0800 in a RAM page at C0000h.
        let mut vm = real16(None, (0xb_8000, 0x8000), Some((0xc_0000, 0x1000)));
        vm.write_bytes(CODE, &[0x26, 0xff, 0x35, 0xf4]).unwrap();
        let mut cpu = vcpu(&vm, 0xb800, 0x10);
        set_segment(&mut cpu, SegmentId::Ss, 0xc000);
        cpu.cpu.gpr[4] = 0x0800;
        let mut bus = Vec::new();
        let Exit::MmioRead { addr, size, .. } = run_on(&vm, &mut cpu) else {
            panic!("no read exit")
        };
        bus.push(Bus::Read(addr, size));
        cpu.complete_mmio_read(0x1234);
        vm.mem.unmap(0xc_0000, 0x1000).unwrap();
        vm.map(0xc_0000, 0x1000, Prot::RW, RegionKind::Trap)
            .unwrap();
        let stop = drain(&vm, &mut cpu, &mut bus);
        if bus != [Bus::Read(0xb_8010, 2), Bus::Write(0xc_07fe, 2, 0x1234)]
            || !matches!(stop, Exit::Hlt)
            || cpu.cpu.gpr[4] & 0xffff != 0x07fe
        {
            bad.push(format!("alias: {bus:x?} {stop:?} sp {:#x}", cpu.cpu.gpr[4]));
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }
}
