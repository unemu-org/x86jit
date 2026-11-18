//! A locked or implicitly locked operand partly in RAM and partly in a Trap
//! region (xj-ir6.34; spec.md §5.2, Locked operands on Trap regions).
//!
//! The processor performs such an operand: the lock "is not affected by the
//! alignment of the memory field" and is held "for as many bus cycles as
//! necessary to update the entire operand" (SDM Vol 3A §11.1.2.2 p. 11-4;
//! 80386 PRM §11.2.1 p. 11-5). On the Real16 alias and every Machine model the
//! engine reads the pieces in operand byte order (the RAM piece directly, the
//! device piece through one `MmioRead`), then writes the device piece through
//! one `MmioWrite` and commits the RAM piece once that write is acknowledged.
//! The low-byte-first order of the twenty-bit models is the bus order of a word
//! at an odd address (8086 Family UM §4.2 p. 4-15); for the 80286, the 80386
//! and the alias it is ASSUMED, as is the device-first write.
//!
//! Each witness runs one instruction twice on each path: once with the device
//! window as RAM, once as a Trap region served by a byte-addressed device. The
//! Trap run must show exactly one read and one write of the device piece, both
//! while the instruction has not retired, and must end in the RAM run's
//! registers, flags, retirement count and bytes. The boundary drive adds a
//! budget stop and a host stop after every completion, and stores other bytes
//! into the RAM piece at each device stop once the instruction has read it: the
//! retry computes from the RAM bytes it read first, and writes the RAM once.
use std::collections::BTreeMap;
use x86jit_core::{
    Backend, CpuMode, CpuModel, Exit, InterpreterBackend, Prot, RegionCaps, RegionKind, SegmentId,
    StepOptions, StepStop, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;
#[cfg(all(
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[path = "../../x86jit-cranelift/src/unwind_test_signal.rs"]
#[allow(dead_code)]
mod signal;

const CODE: u64 = 0x7c00;
const HANDLER: u64 = 0x0500;
/// The device window.
const WINDOW: (u64, u64) = (0xb_8000, 0x8000);
const MEMORY: u64 = 0x20_0000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Generation {
    TwentyBit,
    I286,
    I386,
    /// The alias superset (80486 forms and later).
    Alias,
}

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
    fn generation(self) -> Generation {
        match self {
            Self::Alias | Self::AliasJit => Generation::Alias,
            Self::I8088 | Self::I8086 | Self::NecV20 | Self::NecV30 => Generation::TwentyBit,
            Self::I286 | Self::I286Checked => Generation::I286,
            Self::I386 | Self::I386Checked => Generation::I386,
        }
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
    fn a20_gate(self) -> bool {
        matches!(
            self,
            Self::I286 | Self::I286Checked | Self::I386 | Self::I386Checked
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct Form {
    name: &'static str,
    bytes: &'static [u8],
    size: u8,
    /// The oldest generation that has the form.
    since: Generation,
    /// The newest generation that has the form.
    until: Generation,
    /// CMPXCHG whose comparison succeeds: AX/EAX holds the operand.
    matching: bool,
    /// The form reads its operand: all but the 80286's locked MOV store.
    reads: bool,
}

const fn form(name: &'static str, bytes: &'static [u8], size: u8, since: Generation) -> Form {
    Form {
        name,
        bytes,
        size,
        since,
        until: Generation::Alias,
        matching: false,
        reads: true,
    }
}

impl Form {
    fn on(&self, path: Path) -> bool {
        (self.since..=self.until).contains(&path.generation())
    }
}

/// ModRM 05h is `[DI]`; 0Dh is `[DI]` with CX; 2Dh is `[DI]` with /5.
const WORD_FORMS: &[Form] = &[
    form(
        "lock xchg [di],ax",
        &[0xf0, 0x87, 0x05],
        2,
        Generation::TwentyBit,
    ),
    // XCHG asserts LOCK without the prefix from the 80286 on (80286 PRM
    // Appendix B XCHG p. B-114); on the twenty-bit models it is a plain
    // read-modify-write (`mmio_rmw.rs`, `twenty_bit_rmw_over_ram_and_device`).
    form("xchg [di],ax", &[0x87, 0x05], 2, Generation::I286),
    form(
        "lock add [di],ax",
        &[0xf0, 0x01, 0x05],
        2,
        Generation::TwentyBit,
    ),
    form(
        "lock inc word [di]",
        &[0xf0, 0xff, 0x05],
        2,
        Generation::TwentyBit,
    ),
    form(
        "lock bts word [di],9",
        &[0xf0, 0x0f, 0xba, 0x2d, 0x09],
        2,
        Generation::I386,
    ),
    form(
        "lock xadd [di],ax",
        &[0xf0, 0x0f, 0xc1, 0x05],
        2,
        Generation::Alias,
    ),
    form(
        "lock cmpxchg [di],cx",
        &[0xf0, 0x0f, 0xb1, 0x0d],
        2,
        Generation::Alias,
    ),
    Form {
        matching: true,
        ..form(
            "lock cmpxchg [di],cx",
            &[0xf0, 0x0f, 0xb1, 0x0d],
            2,
            Generation::Alias,
        )
    },
    // The 80286's locked MOV store is one write and no read (spec.md §17.10).
    Form {
        until: Generation::I286,
        reads: false,
        ..form("lock mov [di],ax", &[0xf0, 0x89, 0x05], 2, Generation::I286)
    },
];

const DWORD_FORMS: &[Form] = &[
    form("xchg [di],eax", &[0x66, 0x87, 0x05], 4, Generation::I386),
    form(
        "lock add [di],eax",
        &[0xf0, 0x66, 0x01, 0x05],
        4,
        Generation::I386,
    ),
    form(
        "lock inc dword [di]",
        &[0xf0, 0x66, 0xff, 0x05],
        4,
        Generation::I386,
    ),
    form(
        "lock bts dword [di],27",
        &[0xf0, 0x66, 0x0f, 0xba, 0x2d, 0x1b],
        4,
        Generation::I386,
    ),
    form(
        "lock xadd [di],eax",
        &[0xf0, 0x66, 0x0f, 0xc1, 0x05],
        4,
        Generation::Alias,
    ),
    form(
        "lock cmpxchg [di],ecx",
        &[0xf0, 0x66, 0x0f, 0xb1, 0x0d],
        4,
        Generation::Alias,
    ),
    Form {
        matching: true,
        ..form(
            "lock cmpxchg [di],ecx",
            &[0xf0, 0x66, 0x0f, 0xb1, 0x0d],
            4,
            Generation::Alias,
        )
    },
];

fn seed(addr: u64) -> u8 {
    (addr.wrapping_mul(37) as u8) ^ 0x81
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bus {
    Read(u64, u8),
    Write(u64, u8, u64),
}

/// A byte-addressed device: reads return its bytes, writes replace them.
struct Device(BTreeMap<u64, u8>);
impl Device {
    fn new(regions: &[(u64, u64)]) -> Self {
        Self(
            regions
                .iter()
                .flat_map(|&(start, len)| start..start + len)
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

/// One machine: the device regions (`traps`), what else is not RAM
/// (`others`), the operand's segment and offset, and the A20 gate.
#[derive(Clone, Debug)]
struct Layout {
    traps: Vec<(u64, u64)>,
    others: Vec<(u64, u64, Option<RegionKind>)>,
    segment: u16,
    offset: u16,
    a20: bool,
}

impl Layout {
    fn straddle(start: u64) -> Self {
        // The operand's physical address is segment * 16 + offset, with the
        // offset at 0x7000 or above so that INC BX and INC SI touch no byte of
        // it, and the segment chosen so that the offset fits.
        let segment = ((start - 0x7000) >> 4) as u16;
        Self {
            traps: vec![WINDOW],
            others: Vec::new(),
            segment,
            offset: (start - (u64::from(segment) << 4)) as u16,
            a20: true,
        }
    }
    /// The region holding `addr`: its kind (`None` for no region) and an
    /// index that separates adjacent regions of one kind.
    fn region_at(&self, addr: u64, trap: bool) -> (Option<RegionKind>, usize) {
        if let Some(i) = self
            .traps
            .iter()
            .position(|&(s, l)| (s..s + l).contains(&addr))
        {
            // The RAM run's window is part of the RAM around it: a locked
            // operand there lies in one region.
            return if trap {
                (Some(RegionKind::Trap), 1 + i)
            } else {
                (Some(RegionKind::Ram), 0)
            };
        }
        for (i, &(s, l, kind)) in self.others.iter().enumerate() {
            if (s..s + l).contains(&addr) {
                return (kind, 1 + self.traps.len() + i);
            }
        }
        (Some(RegionKind::Ram), 0)
    }
    fn kind_at(&self, addr: u64, trap: bool) -> Option<RegionKind> {
        self.region_at(addr, trap).0
    }
    /// The regions as `(start, end, kind)`, adjacent ranges of one region
    /// joined, covering `0..MEMORY`.
    fn regions(&self, trap: bool) -> Vec<(u64, u64, Option<RegionKind>)> {
        let mut edges = vec![0, MEMORY];
        for &(s, l) in &self.traps {
            edges.extend([s, s + l]);
        }
        for &(s, l, _) in &self.others {
            edges.extend([s, s + l]);
        }
        edges.sort();
        edges.dedup();
        let mut out: Vec<(u64, u64, (Option<RegionKind>, usize))> = Vec::new();
        for pair in edges.windows(2) {
            let region = self.region_at(pair[0], trap);
            match out.last_mut() {
                Some(last) if last.2 == region && last.1 == pair[0] => last.1 = pair[1],
                _ => out.push((pair[0], pair[1], region)),
            }
        }
        out.into_iter().map(|(s, e, r)| (s, e, r.0)).collect()
    }
}

fn backend(path: Path) -> Box<dyn Backend> {
    match path {
        Path::AliasJit => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 8,
            max_icount: 64,
        })),
        _ => Box::new(InterpreterBackend),
    }
}

fn setup(path: Path, layout: &Layout, form: &Form, trap: bool, code: &[u8]) -> (Vm, Vcpu) {
    let vm = Vm::with_backend(VmConfig::flat(MEMORY), backend(path));
    setup_on(vm, path, layout, form, trap, code)
}

/// [`setup`] on a VM made by the caller, with no region mapped yet.
fn setup_on(
    mut vm: Vm,
    path: Path,
    layout: &Layout,
    form: &Form,
    trap: bool,
    code: &[u8],
) -> (Vm, Vcpu) {
    match path.model() {
        Some(model) => vm.set_cpu_model(model),
        None => {
            vm.set_cpu_mode(CpuMode::Real16);
            vm.set_tier_up_after(None);
            vm.set_tier_up_region_after(Some(1));
        }
    }
    // Map each region, and the RAM between them in maximal runs.
    let regions = layout.regions(trap);
    for &(at, end, kind) in &regions {
        if let Some(kind) = kind {
            let prot = if kind == RegionKind::Ram {
                Prot::RWX
            } else {
                Prot::RW
            };
            vm.map(at, (end - at) as usize, prot, kind).unwrap();
        }
    }
    // Every byte of interest holds its seed: RAM, ROM and the RAM run's
    // window, from B0000h to the end of the first megabyte and its wrap.
    for (from, to) in [(0u64, 4u64), (0xb_0000, 0x10_0010)] {
        for &(at, end, kind) in &regions {
            let (at, end) = (at.max(from), end.min(to));
            if at < end && matches!(kind, Some(RegionKind::Ram | RegionKind::Rom)) {
                let bytes: Vec<u8> = (at..end).map(seed).collect();
                vm.write_bytes(at, &bytes).unwrap();
            }
        }
    }
    // Every processor exception the forms can raise enters 0000:0500, which
    // halts. Vectors 0 and 1 keep their seeds: a layout's operand may hold
    // them, and no form divides or single-steps.
    for vector in 2..32u64 {
        if (0..4).all(|i| layout.kind_at(vector * 4 + i, trap) == Some(RegionKind::Ram)) {
            vm.mem.write(vector * 4, HANDLER, 4).unwrap();
        }
    }
    vm.write_bytes(HANDLER, &[0xf4]).unwrap();
    vm.write_bytes(CODE, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = CODE;
    for (id, selector) in [
        (SegmentId::Cs, 0),
        (SegmentId::Ss, 0),
        (SegmentId::Ds, layout.segment),
        (SegmentId::Es, layout.segment),
    ] {
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
    if matches!(path, Path::I286Checked | Path::I386Checked) {
        // A non-canonical SS cache withholds the Direct proof.
        cpu.cpu.legacy.segments[SegmentId::Ss as usize].attributes = 0x92;
    }
    cpu.cpu.gpr[0] = 0x9e37_c35a; // EAX
    cpu.cpu.gpr[1] = 0x1d0b_0503; // ECX
    cpu.cpu.gpr[3] = 0; // BX
    cpu.cpu.gpr[4] = 0x7000; // SP
    cpu.cpu.gpr[6] = 0; // SI
    cpu.cpu.gpr[7] = u64::from(layout.offset); // DI
    if form.matching {
        // The operand as seeded, so that the comparison succeeds.
        let base = (u64::from(layout.segment) << 4) + u64::from(layout.offset);
        let old = (0..form.size).fold(0u64, |v, i| {
            v | u64::from(seed(operand_byte(path, layout, base, i))) << (8 * i)
        });
        cpu.cpu.gpr[0] = old;
    }
    cpu.cpu.flags.cf = true;
    if path.a20_gate() {
        cpu.set_a20_enabled(layout.a20);
    }
    (vm, cpu)
}

/// The bus address of byte `i` of the operand at linear `base`.
fn operand_byte(path: Path, layout: &Layout, base: u64, i: u8) -> u64 {
    let linear = base + u64::from(i);
    match path.generation() {
        Generation::TwentyBit => {
            ((u64::from(layout.segment) << 4) + u64::from(layout.offset.wrapping_add(u16::from(i))))
                & 0xf_ffff
        }
        _ if path.a20_gate() && !layout.a20 => linear & !(1 << 20),
        _ => linear,
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Observation {
    stop: String,
    retired: u64,
    gpr: [u64; 8],
    rip: u64,
    flags: u64,
}

struct Run {
    observation: Observation,
    bus: Vec<Bus>,
    retired_at_bus: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Drive {
    /// `run_steps` with this budget until a final stop.
    Budget(u64),
    /// After each completion a zero-step `run_steps` (a budget stop) and a
    /// requested host stop; at each device stop, other bytes are stored into
    /// the RAM pieces the instruction has already read.
    Boundaries,
}

fn drive(
    vm: &Vm,
    cpu: &mut Vcpu,
    how: Drive,
    device: &mut Device,
    poke: &dyn Fn(&Vm, &Bus),
) -> Run {
    let budget = match how {
        Drive::Budget(budget) => budget,
        Drive::Boundaries => 64,
    };
    let mut bus = Vec::new();
    let mut retired_at_bus = Vec::new();
    for _ in 0..64 {
        let run = cpu.run_steps(vm, budget, StepOptions::default()).unwrap();
        match run.stop() {
            StepStop::Exit(Exit::BudgetExhausted | Exit::Interrupted) => continue,
            StepStop::Exit(Exit::MmioRead { addr, size, .. }) => {
                let transfer = Bus::Read(*addr, *size);
                if how == Drive::Boundaries {
                    poke(vm, &transfer);
                }
                bus.push(transfer);
                retired_at_bus.push(cpu.retired_instructions());
                cpu.complete_mmio_read(device.read(*addr, *size));
            }
            StepStop::Exit(Exit::MmioWrite {
                addr, size, value, ..
            }) => {
                let transfer = Bus::Write(*addr, *size, *value);
                if how == Drive::Boundaries {
                    poke(vm, &transfer);
                }
                bus.push(transfer);
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
                    StepStop::Exit(Exit::UnmappedMemory { addr, access, .. }) => {
                        format!("UnmappedMemory@{addr:#x} {access:?}")
                    }
                    other => format!("{other:?}"),
                };
                let mut gpr = [0; 8];
                gpr.copy_from_slice(&cpu.cpu.gpr[..8]);
                return Run {
                    observation: Observation {
                        stop,
                        retired: cpu.retired_instructions(),
                        gpr,
                        rip: cpu.cpu.rip,
                        flags: cpu.cpu.flags.to_rflags() & 0x8d5,
                    },
                    bus,
                    retired_at_bus,
                };
            }
        }
        if how == Drive::Boundaries {
            let run = cpu.run_steps(vm, 0, StepOptions::default()).unwrap();
            assert!(
                matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                "{:?}",
                run.stop()
            );
            cpu.exit_request().request_exit();
        }
    }
    panic!("no final stop: {bus:x?}");
}

// INC BX ; <form> ; INC SI ; HLT
fn program(form: &Form) -> Vec<u8> {
    let mut code = vec![0x43];
    code.extend_from_slice(form.bytes);
    code.extend_from_slice(&[0x46, 0xf4]);
    code
}

/// Run `form` on `path` over `layout` and compare the Trap run with the RAM
/// run. Returns whether the instruction completed. Disagreements go to
/// `failures`.
fn witness(path: Path, layout: &Layout, form: &Form, failures: &mut Vec<String>) -> bool {
    let what = format!(
        "{path:?} {}{} at {:04x}:{:04x} (A20 {})",
        form.name,
        if form.matching { " (match)" } else { "" },
        layout.segment,
        layout.offset,
        if layout.a20 { "on" } else { "off" }
    );
    let code = program(form);
    let end = CODE + code.len() as u64;
    let base = (u64::from(layout.segment) << 4) + u64::from(layout.offset);
    let bytes: Vec<u64> = (0..form.size)
        .map(|i| operand_byte(path, layout, base, i))
        .collect();
    let is_device = |a: u64| layout.traps.iter().any(|&(s, l)| (s..s + l).contains(&a));

    let (vm, mut cpu) = setup(path, layout, form, false, &code);
    let mut unused = Device(BTreeMap::new());
    let reference = drive(&vm, &mut cpu, Drive::Budget(64), &mut unused, &|_, _| {});
    assert!(reference.bus.is_empty(), "{what}");
    let after: Vec<u8> = bytes
        .iter()
        .map(|&a| vm.mem.read(a, 1).unwrap() as u8)
        .collect();
    let completed = reference.observation.stop == "Hlt" && reference.observation.rip == end;
    if !completed {
        failures.push(format!(
            "{what}: the RAM run stops {:x?}",
            reference.observation
        ));
    }
    // The device piece: the operand's bytes in the window, one run.
    let device: Vec<(u8, u64)> = bytes
        .iter()
        .enumerate()
        .filter(|&(_, &a)| is_device(a))
        .map(|(i, &a)| (i as u8, a))
        .collect();
    assert!(
        !device.is_empty() && device.len() < bytes.len(),
        "{what}: not a split operand"
    );
    let expected = if completed {
        let (first, addr) = device[0];
        let len = device.len() as u8;
        let value = (0..len).fold(0, |v, i| {
            v | u64::from(after[usize::from(first + i)]) << (8 * i)
        });
        let write = Bus::Write(addr, len, value);
        if form.reads {
            vec![Bus::Read(addr, len), write]
        } else {
            vec![write]
        }
    } else {
        Vec::new()
    };
    // The RAM bytes, and those read before the device piece.
    let ram_bytes: Vec<u64> = bytes.iter().copied().filter(|&a| !is_device(a)).collect();
    let ram_before: Vec<u64> = bytes
        .iter()
        .copied()
        .take_while(|&a| !is_device(a))
        .collect();

    for how in [
        Drive::Budget(1),
        Drive::Budget(2),
        Drive::Budget(3),
        Drive::Budget(64),
        Drive::Boundaries,
    ] {
        let what = format!("{what}, {how:?}");
        let (vm, mut cpu) = setup(path, layout, form, true, &code);
        let mut device_model = Device::new(&layout.traps);
        // Every RAM piece has been read at the device write; at the device
        // read, those before the device piece have.
        let poke = |vm: &Vm, transfer: &Bus| {
            let read = if matches!(transfer, Bus::Write(..)) {
                &ram_bytes
            } else {
                &ram_before
            };
            for &a in read {
                vm.write_bytes(a, &[0xee]).unwrap();
            }
        };
        let run = drive(&vm, &mut cpu, how, &mut device_model, &poke);
        if run.bus != expected {
            failures.push(format!(
                "{what}: transfers {:x?}, expected {:x?}",
                run.bus, expected
            ));
            continue;
        }
        if run.observation != reference.observation {
            failures.push(format!(
                "{what}: {:x?}, RAM run {:x?}",
                run.observation, reference.observation
            ));
            continue;
        }
        // Only INC BX has retired while the transfers are outstanding.
        if !run.retired_at_bus.iter().all(|&r| r == 1) {
            failures.push(format!("{what}: retired {:?}", run.retired_at_bus));
            continue;
        }
        let final_bytes: Vec<u8> = bytes
            .iter()
            .map(|&a| {
                if is_device(a) {
                    device_model.0[&a]
                } else {
                    vm.mem.read(a, 1).unwrap() as u8
                }
            })
            .collect();
        if final_bytes != after {
            failures.push(format!(
                "{what}: operand {final_bytes:02x?}, RAM run {after:02x?}"
            ));
        }
    }
    completed
}

fn check(failures: Vec<String>) {
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A word over the last RAM byte before the window and its first device byte,
/// and over the window's last device byte and the RAM byte after it, on every
/// path, with every locked word form the path has.
#[test]
fn locked_word_over_ram_and_device_is_performed() {
    let mut failures = Vec::new();
    for path in PATHS {
        let mut done = 0;
        let mut forms = 0;
        for start in [WINDOW.0 - 1, WINDOW.0 + WINDOW.1 - 1] {
            let layout = Layout::straddle(start);
            for form in WORD_FORMS.iter().filter(|f| f.on(path)) {
                forms += 1;
                done += usize::from(witness(path, &layout, form, &mut failures));
            }
        }
        println!("{path:?}: {done} of {forms} completed");
        if done != forms {
            failures.push(format!("{path:?}: {done} of {forms} completed"));
        }
    }
    check(failures);
}

/// A doubleword with one, two or three RAM bytes on either side of the
/// window's edges, on the 80386 and the alias: the device piece is one
/// transfer of the remaining bytes.
#[test]
fn locked_dword_over_ram_and_device_is_performed() {
    let mut failures = Vec::new();
    for path in PATHS
        .into_iter()
        .filter(|p| p.generation() >= Generation::I386)
    {
        let mut done = 0;
        let mut forms = 0;
        for ram in 1..4 {
            for start in [WINDOW.0 - ram, WINDOW.0 + WINDOW.1 - (4 - ram)] {
                let layout = Layout::straddle(start);
                for form in DWORD_FORMS.iter().filter(|f| f.on(path)) {
                    forms += 1;
                    done += usize::from(witness(path, &layout, form, &mut failures));
                }
            }
        }
        println!("{path:?}: {done} of {forms} completed");
        if done != forms {
            failures.push(format!("{path:?}: {done} of {forms} completed"));
        }
    }
    check(failures);
}

/// The 80286 and 80386 with the A20 gate off: a word at FFFF:000F is split at
/// the bus into FFFFFh and 0, and a doubleword at FFFF:000E into FFFFEh–FFFFFh
/// and 0–1 (spec.md §17.10). One device piece and RAM pieces, in either
/// order. With the gate on, the same word lies at FFFFFh–100000h, one range of
/// bus addresses, and a device byte at either end is the same split.
#[test]
fn a20_split_locked_operand_over_ram_and_device_is_performed() {
    let mut failures = Vec::new();
    let paths = [Path::I286, Path::I286Checked, Path::I386, Path::I386Checked];
    for path in paths {
        let mut done = 0;
        let mut forms = 0;
        let mut layouts = Vec::new();
        for a20 in [false, true] {
            // (device region) for a word at FFFF:000F.
            let high = if a20 { 0x10_0000 } else { 0 };
            for trap in [(0xf_ffffu64, 1u64), (high, 0x10)] {
                layouts.push((
                    2u8,
                    Layout {
                        traps: vec![trap],
                        others: Vec::new(),
                        segment: 0xffff,
                        offset: 0x000f,
                        a20,
                    },
                ));
            }
        }
        if path.generation() >= Generation::I386 {
            // A doubleword at FFFF:000E with A20 off: one device byte at
            // FFFFFh, or both at 0–1, or FFFFEh–FFFFFh.
            for trap in [(0xf_ffffu64, 1u64), (0, 2), (0xf_fffe, 2)] {
                layouts.push((
                    4,
                    Layout {
                        traps: vec![trap],
                        others: Vec::new(),
                        segment: 0xffff,
                        offset: 0x000e,
                        a20: false,
                    },
                ));
            }
        }
        for (size, layout) in &layouts {
            let forms_of = if *size == 2 { WORD_FORMS } else { DWORD_FORMS };
            for form in forms_of.iter().filter(|f| f.on(path)) {
                forms += 1;
                done += usize::from(witness(path, layout, form, &mut failures));
            }
        }
        println!("{path:?}: {done} of {forms} completed");
        if done != forms {
            failures.push(format!("{path:?}: {done} of {forms} completed"));
        }
    }
    check(failures);
}

/// What the device saw and how the instruction stopped, for an operand that
/// does not complete.
fn stopped(path: Path, layout: &Layout, form: &Form) -> (Run, Vm, Vcpu) {
    let code = program(form);
    let (vm, mut cpu) = setup(path, layout, form, true, &code);
    let mut device = Device::new(&layout.traps);
    let run = drive(&vm, &mut cpu, Drive::Boundaries, &mut device, &|_, _| {});
    for (&a, &b) in &device.0 {
        assert_eq!(b, seed(a), "{path:?} {}: device byte {a:#x}", form.name);
    }
    (run, vm, cpu)
}

/// A RAM piece in ROM or in no region is the containment miss of a locked
/// operand over two regions: `UnmappedMemory` for a write at that piece, with
/// no device transfer issued, the RAM and the device unchanged and the
/// instruction not retired.
#[test]
fn ram_piece_in_rom_or_unmapped_faults_before_any_transfer() {
    for path in PATHS {
        for (kind, name) in [(Some(RegionKind::Rom), "ROM"), (None, "unmapped")] {
            for (start, other) in [
                (WINDOW.0 - 1, WINDOW.0 - 1),
                (WINDOW.0 + WINDOW.1 - 1, WINDOW.0 + WINDOW.1),
            ] {
                let mut layout = Layout::straddle(start);
                layout.others.push((other & !0xf, 0x10, kind));
                let form = &WORD_FORMS[2];
                let what = format!("{path:?} {} with a {name} byte at {other:#x}", form.name);
                let (run, vm, cpu) = stopped(path, &layout, form);
                assert!(run.bus.is_empty(), "{what}: {:x?}", run.bus);
                assert_eq!(
                    run.observation.stop,
                    format!("UnmappedMemory@{other:#x} Write"),
                    "{what}"
                );
                assert_eq!(run.observation.retired, 1, "{what}");
                assert_eq!(cpu.cpu.gpr[0] & 0xffff_ffff, 0x9e37_c35a, "{what}");
                assert_eq!(cpu.cpu.rip & 0xffff, CODE + 1, "{what}");
                if kind.is_some() {
                    assert_eq!(vm.mem.read(other, 1).unwrap() as u8, seed(other), "{what}");
                }
            }
        }
    }
}

/// A RAM piece in a host page that refuses writes faults at that piece with
/// host-fault recovery on, before any transfer: the piece is probed for a
/// write with every other piece, before the device is read. The RAM keeps its
/// bytes and the instruction does not retire.
#[cfg(all(
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[test]
fn ram_piece_the_host_refuses_to_write_faults_before_any_transfer() {
    use x86jit_core::memory::HostRam;
    // SAFETY: sysconf has no pointer argument.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 };
    // A device window that starts, or ends, at a host page boundary.
    const EDGE: u64 = 0xc_0000;
    for path in [
        Path::Alias,
        Path::I8088,
        Path::I8086,
        Path::I286,
        Path::I386,
    ] {
        for (window, start, guard) in [
            ((EDGE, 0x1000), EDGE - 1, EDGE - page),
            ((EDGE - 0x1000, 0x1000), EDGE - 1, EDGE),
        ] {
            let status = signal::child(|| {
                // SAFETY: the forked child owns its handlers and mapping.
                unsafe {
                    signal::install();
                }
                // SAFETY: the mapping stays owned by HostRam and is freed exactly once.
                let vm = unsafe {
                    let ptr = libc::mmap(
                        std::ptr::null_mut(),
                        MEMORY as usize,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANON,
                        -1,
                        0,
                    );
                    assert_ne!(ptr, libc::MAP_FAILED);
                    Vm::with_backend_host_ram(
                        VmConfig::flat(MEMORY),
                        backend(path),
                        HostRam {
                            ptr: ptr.cast(),
                            len: MEMORY as usize,
                            guest_base: 0,
                            protect: None,
                            dtor: Box::new(|p, len| {
                                libc::munmap(p.cast(), len);
                            }),
                        },
                    )
                    .unwrap()
                };
                let mut layout = Layout::straddle(start);
                layout.traps = vec![window];
                let form = &WORD_FORMS[2];
                let (mut vm, mut cpu) = setup_on(vm, path, &layout, form, true, &program(form));
                vm.mem.set_host_fault_recovery(true);
                signal::HANDLE.store(
                    cpu.fault_handle().as_usize(),
                    std::sync::atomic::Ordering::Relaxed,
                );
                let ram = if window.0 == EDGE { start } else { start + 1 };
                let before = vm.mem.read(ram, 1).unwrap();
                // SAFETY: the page is aligned and belongs to this VM's live mapping.
                unsafe {
                    let host = vm.mem.host_base().add(guard as usize).cast_mut();
                    assert_eq!(
                        libc::mprotect(host.cast(), page as usize, libc::PROT_READ),
                        0
                    );
                }
                let mut device = Device::new(&layout.traps);
                let run = drive(&vm, &mut cpu, Drive::Boundaries, &mut device, &|_, _| {});
                // SAFETY: as above.
                unsafe {
                    let host = vm.mem.host_base().add(guard as usize).cast_mut();
                    let prot = libc::PROT_READ | libc::PROT_WRITE;
                    assert_eq!(libc::mprotect(host.cast(), page as usize, prot), 0);
                }
                let what = format!("{path:?} {} with the RAM byte at {ram:#x}", form.name);
                assert!(run.bus.is_empty(), "{what}: {:x?}", run.bus);
                assert_eq!(
                    run.observation.stop,
                    format!("UnmappedMemory@{ram:#x} Write"),
                    "{what}"
                );
                assert_eq!(run.observation.retired, 1, "{what}");
                assert_eq!(vm.mem.read(ram, 1).unwrap(), before, "{what}");
                assert_eq!(cpu.cpu.gpr[0] & 0xffff_ffff, 0x9e37_c35a, "{what}");
            });
            assert_eq!(status, 0, "{path:?} window {window:x?}");
        }
    }
}

/// A word over two adjacent device regions keeps `Exit::UnknownInstruction`
/// with the instruction's bytes, before any transfer: one exit names one
/// region. So does a doubleword with a RAM byte between two device regions.
#[test]
fn operand_over_two_device_regions_keeps_embedder_emulation() {
    for path in PATHS {
        let mut layout = Layout::straddle(WINDOW.0 + WINDOW.1 - 1);
        layout.traps.push((WINDOW.0 + WINDOW.1, 0x10));
        let form = &WORD_FORMS[2];
        let (run, _, cpu) = stopped(path, &layout, form);
        let what = format!("{path:?} {}", form.name);
        assert!(run.bus.is_empty(), "{what}: {:x?}", run.bus);
        assert_eq!(
            run.observation.stop,
            format!("UnknownInstruction@{:#x}+{}", CODE + 1, form.bytes.len()),
            "{what}"
        );
        assert_eq!(cpu.cpu.gpr[0] & 0xffff_ffff, 0x9e37_c35a, "{what}");
        if path.generation() >= Generation::I386 {
            let mut layout = Layout::straddle(WINDOW.0 + WINDOW.1 - 1);
            layout.traps.push((WINDOW.0 + WINDOW.1 + 1, 0x10));
            let form = &DWORD_FORMS[1];
            let (run, _, _) = stopped(path, &layout, form);
            let what = format!("{path:?} {}", form.name);
            assert!(run.bus.is_empty(), "{what}: {:x?}", run.bus);
            assert_eq!(
                run.observation.stop,
                format!("UnknownInstruction@{:#x}+{}", CODE + 1, form.bytes.len()),
                "{what}"
            );
        }
    }
}

/// The alias's CMPXCHG8B keeps embedder emulation over RAM and a device, as
/// on a device alone (spec.md §5.2).
#[test]
fn alias_cmpxchg8b_over_ram_and_device_keeps_embedder_emulation() {
    for path in [Path::Alias, Path::AliasJit] {
        for ram in [1u64, 4, 7] {
            let layout = Layout::straddle(WINDOW.0 - ram);
            let form = form(
                "lock cmpxchg8b [di]",
                &[0xf0, 0x0f, 0xc7, 0x0d],
                8,
                Generation::Alias,
            );
            let (run, _, _) = stopped(path, &layout, &form);
            assert!(run.bus.is_empty(), "{path:?} {ram}: {:x?}", run.bus);
            assert_eq!(
                run.observation.stop,
                format!("UnknownInstruction@{:#x}+4", CODE + 1),
                "{path:?} {ram}"
            );
        }
    }
}

/// Compat32 keeps `Exit::UnknownInstruction` for a locked operand over RAM
/// and a device, as Long64 does (`locked_mmio.rs`, `atomic_boundary_*`).
#[test]
fn compat32_locked_operand_over_ram_and_device_keeps_embedder_emulation() {
    for jit in [false, true] {
        let backend: Box<dyn Backend> = if jit {
            Box::new(JitBackend::new())
        } else {
            Box::new(InterpreterBackend)
        };
        let mut vm = Vm::with_backend(VmConfig::flat(0x5000), backend);
        vm.set_cpu_mode(CpuMode::Compat32);
        vm.set_tier_up_after(Some(0));
        vm.map(0x1000, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.write_bytes(0x2ffe, &[0x11, 0x22]).unwrap();
        // lock add [edi],eax ; hlt
        let code = [0xf0, 0x01, 0x07, 0xf4];
        vm.write_bytes(0x1000, &code).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[0] = 0x1234_5678;
        cpu.cpu.gpr[7] = 0x2ffe;
        let exit = cpu.run(&vm, Some(16));
        assert!(
            matches!(
                exit,
                Exit::UnknownInstruction {
                    addr: 0x1000,
                    len: 3,
                    ..
                }
            ),
            "jit {jit}: {exit:?}"
        );
        let mut ram = [0; 2];
        vm.read_bytes(0x2ffe, &mut ram).unwrap();
        assert_eq!(ram, [0x11, 0x22]);
    }
}

/// A host stop requested after a device read is answered is taken after the
/// retry, not at the fetch of the retry's block: the Machine models finish the
/// open retry first (spec.md §5.2). On the twenty-bit models the retry of an
/// instruction that did not start its block is fetched again, and a stop there
/// dropped the answer and read the device a second time. A plain read-modify-
/// write inside the window shows it on every path.
#[test]
fn host_stop_before_a_retry_keeps_the_answer() {
    for path in PATHS {
        let layout = Layout::straddle(WINDOW.0 + 0x10);
        let form = form("add [di],ax", &[0x01, 0x05], 2, Generation::TwentyBit);
        let code = program(&form);
        let (vm, mut cpu) = setup(path, &layout, &form, true, &code);
        let mut device = Device::new(&layout.traps);
        let run = drive(&vm, &mut cpu, Drive::Boundaries, &mut device, &|_, _| {});
        let at = WINDOW.0 + 0x10;
        let sum = (device.read(at, 2) as u16).wrapping_sub(0xc35a);
        let expected = if matches!(path, Path::I8088 | Path::NecV20) {
            vec![
                Bus::Read(at, 1),
                Bus::Read(at + 1, 1),
                Bus::Write(at, 1, u64::from(sum.wrapping_add(0xc35a) & 0xff)),
                Bus::Write(at + 1, 1, u64::from(sum.wrapping_add(0xc35a) >> 8)),
            ]
        } else {
            vec![
                Bus::Read(at, 2),
                Bus::Write(at, 2, u64::from(sum.wrapping_add(0xc35a))),
            ]
        };
        assert_eq!(run.bus, expected, "{path:?}");
        assert_eq!(
            sum,
            u16::from_le_bytes([seed(at), seed(at + 1)]),
            "{path:?}"
        );
        assert_eq!(run.observation.stop, "Hlt", "{path:?}");
    }
}
