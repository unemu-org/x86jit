//! Device transfers of the twenty-bit models follow the catalogue's data-bus
//! width (xj-50w.20; spec.md §5.2, §17.10).
//!
//! The 8088 "always accesses memory in bytes"; the 8086 accesses a word at an
//! even address "in one bus cycle" and a word at an odd address "one byte at a
//! time in two consecutive bus cycles" (8086 Family User's Manual 9800722-03
//! §2.3 p. 2-15). The odd word's first cycle moves its low-order byte at the
//! odd address, then the address is incremented for the high-order byte (§4.2
//! p. 4-15). The NEC V20/V30 manual says the same of the uPD70108 (8-bit bus)
//! and the uPD70116 (16-bit bus) (NEC V20/V30 UM §4 pp. 4-2–4-3).
//!
//! Each witness runs one instruction twice on each model: once with the device
//! windows as RAM, once as Trap regions served by a byte-addressed device. The
//! Trap run must show exactly the expected transfers, in order, with sizes and
//! addresses, every one while the instruction has not retired, and must end in
//! the RAM run's registers, flags and memory.
use std::collections::BTreeMap;
use x86jit_core::{
    CpuModel, Exit, FpuModel, I8080Handback, InterpreterBackend, PortDir, Prot, RegionKind,
    StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

const MODELS: [CpuModel; 4] = [
    CpuModel::I8088,
    CpuModel::I8086,
    CpuModel::NecV20,
    CpuModel::NecV30,
];
const CODE: u64 = 0x7c00;
const HANDLER: u64 = 0x0500;
const PORT: u16 = 0x3f8;
const PORT_ANSWER: u64 = 0x5a3c;
/// The default device window: B800:0000-00FF.
const WINDOW: (u64, u64) = (0xb_8000, 0x100);
const SEG: u16 = 0xb800;

/// The 8086 and V30 move an even-addressed word in one bus cycle.
fn word_bus(model: CpuModel) -> bool {
    matches!(model, CpuModel::I8086 | CpuModel::NecV30)
}
fn nec(model: CpuModel) -> bool {
    matches!(model, CpuModel::NecV20 | CpuModel::NecV30)
}
fn phys(segment: u16, offset: u16) -> u64 {
    (u64::from(segment) << 4) + u64::from(offset)
}
fn seed(addr: u64) -> u8 {
    (addr.wrapping_mul(37) as u8) ^ 0x81
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bus {
    Read(u64, u8),
    Write(u64, u8, u64),
    PortIn(u16, u8),
    PortOut(u16, u8, u64),
}

/// An architectural transfer of the instruction, in the engine's order.
#[derive(Clone, Copy, Debug)]
enum Op {
    /// A byte, or a word at `addr` and `addr + 1`.
    Read(u64, u8),
    Write(u64, u8),
    PortIn(u8),
    PortOut(u8),
}

struct Case {
    name: String,
    nec_only: bool,
    no_fpu: bool,
    code: Vec<u8>,
    /// (index, value) of the low 16 bits of a GPR; segments are separate.
    gpr: Vec<(usize, u64)>,
    ds: u16,
    es: u16,
    ss: u16,
    traps: Vec<(u64, u64)>,
    preload: Vec<(u64, Vec<u8>)>,
    ops: Vec<Op>,
    /// Applied to the vCPU after setup (flags, a pending interrupt).
    prepare: Option<fn(&mut Vcpu)>,
    /// V20/V30: at the first `I8080Mode` exit, hand back an EDh escape at
    /// this 8080 PC (CS is the handler's, 0000h).
    escape: Option<u16>,
}

fn case(name: &str, code: &[u8]) -> Case {
    Case {
        name: name.to_string(),
        nec_only: false,
        no_fpu: false,
        code: code.to_vec(),
        gpr: Vec::new(),
        ds: SEG,
        es: SEG,
        ss: 0,
        traps: vec![WINDOW],
        preload: Vec::new(),
        ops: Vec::new(),
        prepare: None,
        escape: None,
    }
}
impl Case {
    fn gpr(mut self, index: usize, value: u64) -> Self {
        self.gpr.push((index, value));
        self
    }
    fn segs(mut self, ds: u16, es: u16, ss: u16) -> Self {
        (self.ds, self.es, self.ss) = (ds, es, ss);
        self
    }
    fn traps(mut self, traps: &[(u64, u64)]) -> Self {
        self.traps = traps.to_vec();
        self
    }
    fn preload(mut self, addr: u64, bytes: &[u8]) -> Self {
        self.preload.push((addr, bytes.to_vec()));
        self
    }
    fn ops(mut self, ops: &[Op]) -> Self {
        self.ops = ops.to_vec();
        self
    }
    fn nec_only(mut self) -> Self {
        self.nec_only = true;
        self
    }
    fn no_fpu(mut self) -> Self {
        self.no_fpu = true;
        self
    }
    fn prepare(mut self, prepare: fn(&mut Vcpu)) -> Self {
        self.prepare = Some(prepare);
        self
    }
    fn escape(mut self, pc: u16) -> Self {
        self.escape = Some(pc);
        self
    }
    fn trap_region(&self, addr: u64) -> Option<usize> {
        self.traps
            .iter()
            .position(|&(start, len)| start <= addr && addr < start + len)
    }
}

/// A byte-addressed device: reads return its bytes, writes replace them.
struct Device(BTreeMap<u64, u8>);
impl Device {
    fn new(c: &Case) -> Self {
        let mut bytes = BTreeMap::new();
        for &(start, len) in &c.traps {
            for a in start..start + len {
                bytes.insert(a, seed(a));
            }
        }
        for (addr, data) in &c.preload {
            for (i, b) in data.iter().enumerate() {
                if let Some(slot) = bytes.get_mut(&(addr + i as u64)) {
                    *slot = *b;
                }
            }
        }
        Self(bytes)
    }
    fn read(&self, addr: u64, size: u8) -> u64 {
        (0..size).fold(0, |v, i| {
            v | u64::from(self.0[&(addr + u64::from(i))]) << (8 * i)
        })
    }
    fn write(&mut self, addr: u64, size: u8, value: u64) {
        for i in 0..size {
            let slot = self.0.get_mut(&(addr + u64::from(i))).expect("device byte");
            *slot = (value >> (8 * i)) as u8;
        }
    }
}

fn setup(model: CpuModel, c: &Case, trap: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10_0000), Box::new(InterpreterBackend));
    vm.set_cpu_model(model);
    let mut traps = c.traps.clone();
    traps.sort();
    let mut at = 0;
    for &(start, len) in &traps {
        if start > at {
            vm.map(at, (start - at) as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
        }
        let kind = if trap {
            RegionKind::Trap
        } else {
            RegionKind::Ram
        };
        vm.map(start, len as usize, Prot::RW, kind).unwrap();
        at = start + len;
    }
    if at < 0x10_0000 {
        vm.map(at, (0x10_0000 - at) as usize, Prot::RWX, RegionKind::Ram)
            .unwrap();
    }
    // `device`: the bytes also go to the device windows of the RAM run (a
    // preload, which the device holds too); otherwise they go to RAM only.
    let write = |vm: &Vm, addr: u64, bytes: &[u8], device: bool| {
        for (i, b) in bytes.iter().enumerate() {
            let a = addr + i as u64;
            if c.trap_region(a).is_none() || (device && !trap) {
                vm.write_bytes(a, &[*b]).unwrap();
            }
        }
    };
    if !trap {
        for &(start, len) in &c.traps {
            let bytes: Vec<u8> = (start..start + len).map(seed).collect();
            vm.write_bytes(start, &bytes).unwrap();
        }
    }
    // Every interrupt vector the cases name enters 0000:0500, which halts.
    for vector in 0..0x21u64 {
        write(&vm, vector * 4, &(HANDLER as u32).to_le_bytes(), false);
    }
    write(&vm, HANDLER, &[0xf4], false);
    for (addr, bytes) in &c.preload {
        write(&vm, *addr, bytes, true);
    }
    let mut code = c.code.clone();
    code.push(0xf4);
    vm.write_bytes(CODE, &code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = CODE;
    cpu.cpu.cs = 0;
    cpu.cpu.ds = c.ds;
    cpu.cpu.es = c.es;
    cpu.cpu.ss = c.ss;
    cpu.cpu.gpr[0] = 0xc35a;
    cpu.cpu.gpr[1] = 0x0503;
    cpu.cpu.gpr[2] = u64::from(PORT);
    cpu.cpu.gpr[3] = 0x2211;
    cpu.cpu.gpr[4] = 0x7000;
    cpu.cpu.gpr[5] = 0x1234;
    cpu.cpu.gpr[6] = 0x10;
    cpu.cpu.gpr[7] = 0x10;
    for &(index, value) in &c.gpr {
        cpu.cpu.gpr[index] = value;
    }
    if c.no_fpu {
        cpu.set_fpu_model(FpuModel::None);
    }
    if let Some(prepare) = c.prepare {
        prepare(&mut cpu);
    }
    (vm, cpu)
}

#[derive(Debug, PartialEq, Eq)]
struct Observation {
    stop: String,
    retired: u64,
    gpr: [u64; 8],
    rip: u64,
    segments: [u16; 4],
    flags: u64,
}

struct Run {
    observation: Observation,
    bus: Vec<Bus>,
    /// Retired-instruction count at each device stop, counted from the
    /// 8080-mode hand-back once there was one.
    retired_at_bus: Vec<u64>,
}

/// How the driver resumes after each device completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Resume {
    /// `run_steps` again.
    Run,
    /// A zero-step `run_steps` (a budget stop), a requested host stop, then
    /// `run_steps`: boundary stops before the retry (spec.md §5.2).
    Boundaries,
}

fn drive(
    vm: &Vm,
    cpu: &mut Vcpu,
    budget: u64,
    mut device: Option<&mut Device>,
    resume: Resume,
    escape: Option<u16>,
) -> Run {
    let mut bus = Vec::new();
    let mut retired_at_bus = Vec::new();
    // RETEM and CALLN are units of their own after BRKEM retired.
    let mut escaped = false;
    let mut base = 0;
    // INM's input handoff retires the instruction; its store completes it
    // after (spec.md §5.2), so its device stop is not checked.
    let mut handed_off = false;
    for _ in 0..256 {
        let run = cpu.run_steps(vm, budget, StepOptions::default()).unwrap();
        let mut completed = false;
        match run.stop() {
            StepStop::Exit(Exit::BudgetExhausted | Exit::Interrupted) => continue,
            StepStop::Exit(Exit::MmioRead { addr, size, .. }) => {
                let device = device.as_deref_mut().expect("MMIO read from RAM");
                bus.push(Bus::Read(*addr, *size));
                if !handed_off {
                    retired_at_bus.push(cpu.retired_instructions() - base);
                }
                cpu.complete_mmio_read(device.read(*addr, *size));
                completed = true;
            }
            StepStop::Exit(Exit::MmioWrite {
                addr, size, value, ..
            }) => {
                let device = device.as_deref_mut().expect("MMIO write to RAM");
                bus.push(Bus::Write(*addr, *size, *value));
                if !handed_off {
                    retired_at_bus.push(cpu.retired_instructions() - base);
                }
                device.write(*addr, *size, *value);
                cpu.complete_mmio_write();
                completed = true;
            }
            StepStop::Exit(Exit::I8080Mode { view, .. }) if escape.is_some() && !escaped => {
                let mut view = *view;
                view.pc = escape.unwrap();
                cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
                escaped = true;
                base = cpu.retired_instructions();
            }
            StepStop::Exit(Exit::PortIo {
                port,
                size,
                dir,
                value,
                ..
            }) => {
                handed_off = true;
                if *dir == PortDir::In {
                    bus.push(Bus::PortIn(*port, *size));
                    cpu.complete_port_in(PORT_ANSWER);
                } else {
                    bus.push(Bus::PortOut(*port, *size, *value));
                }
                // A port handoff may complete the instruction (OUTM's last
                // element); only device stops are checked for retirement.
            }
            stop => {
                let stop = match stop {
                    StepStop::Exit(Exit::Hlt) => "Hlt".to_string(),
                    StepStop::Exit(Exit::UnknownInstruction { addr, len, .. }) => {
                        format!("UnknownInstruction@{addr:#x}+{len}")
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
                        segments: [cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss],
                        flags: cpu.cpu.flags.to_rflags() & 0xfd5,
                    },
                    bus,
                    retired_at_bus,
                };
            }
        }
        if completed && resume == Resume::Boundaries {
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

/// The transfers the Trap run must show: each operand word inside one Trap
/// region at an even address is one transfer on a sixteen-bit bus; every other
/// device byte is a transfer of its own; RAM bytes are none.
fn expected(model: CpuModel, c: &Case, ram: &dyn Fn(u64) -> u8, outs: &[u64]) -> Vec<Bus> {
    let mut bus = Vec::new();
    let mut outs = outs.iter();
    let value =
        |a: u64, n: u8| (0..n).fold(0, |v, i| v | u64::from(ram(a + u64::from(i))) << (8 * i));
    for op in &c.ops {
        let (addr, size, write) = match *op {
            Op::Read(a, n) => (a, n, false),
            Op::Write(a, n) => (a, n, true),
            Op::PortIn(n) => {
                bus.push(Bus::PortIn(PORT, n));
                continue;
            }
            Op::PortOut(n) => {
                bus.push(Bus::PortOut(PORT, n, *outs.next().expect("port output")));
                continue;
            }
        };
        let pieces: Vec<(u64, u8)> = if size == 2
            && word_bus(model)
            && addr & 1 == 0
            && c.trap_region(addr).is_some()
            && c.trap_region(addr) == c.trap_region(addr + 1)
        {
            vec![(addr, 2)]
        } else {
            (0..u64::from(size))
                .map(|i| (addr + i, 1))
                .filter(|&(a, _)| c.trap_region(a).is_some())
                .collect()
        };
        for (a, n) in pieces {
            bus.push(if write {
                Bus::Write(a, n, value(a, n))
            } else {
                Bus::Read(a, n)
            });
        }
    }
    bus
}

fn device_addresses(c: &Case) -> Vec<u64> {
    c.traps
        .iter()
        .flat_map(|&(start, len)| start..start + len)
        .collect()
}

/// Run `cases` on every model; every disagreement is collected.
fn witness(cases: &[Case]) {
    let mut failures = Vec::new();
    let mut checked = 0;
    for model in MODELS {
        for c in cases {
            if c.nec_only && !nec(model) {
                continue;
            }
            let what = format!("{model:?} {}", c.name);
            let (vm, mut cpu) = setup(model, c, false);
            let reference = drive(&vm, &mut cpu, 64, None, Resume::Run, c.escape);
            let outs: Vec<u64> = reference
                .bus
                .iter()
                .filter_map(|b| match b {
                    Bus::PortOut(_, _, v) => Some(*v),
                    _ => None,
                })
                .collect();
            let addresses = device_addresses(c);
            let ram: BTreeMap<u64, u8> = addresses
                .iter()
                .map(|&a| (a, vm.mem.read(a, 1).unwrap() as u8))
                .collect();
            let stack: Vec<u8> = (0..0x100)
                .map(|a| vm.mem.read(0x6f00 + a, 1).unwrap() as u8)
                .collect();
            let strings: Vec<u8> = (0..0x100)
                .map(|a| vm.mem.read(0x1_0000 + a, 1).unwrap() as u8)
                .collect();
            let want = expected(model, c, &|a| ram[&a], &outs);
            let before = failures.len();
            for budget in [1, 64] {
                for resume in [Resume::Run, Resume::Boundaries] {
                    let what = format!("{what}, budget {budget}, {resume:?}");
                    checked += 1;
                    let (vm, mut cpu) = setup(model, c, true);
                    let mut device = Device::new(c);
                    let run = drive(&vm, &mut cpu, budget, Some(&mut device), resume, c.escape);
                    if run.bus != want {
                        failures.push(format!(
                            "{what}: transfers {:x?}, expected {want:x?}",
                            run.bus
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
                    if !run.retired_at_bus.iter().all(|&r| r == 0) {
                        failures.push(format!("{what}: retired {:?}", run.retired_at_bus));
                        continue;
                    }
                    if addresses.iter().any(|a| device.0[a] != ram[a]) {
                        failures.push(format!("{what}: device and RAM run differ"));
                        continue;
                    }
                    let stack_now: Vec<u8> = (0..0x100)
                        .map(|a| vm.mem.read(0x6f00 + a, 1).unwrap() as u8)
                        .collect();
                    let strings_now: Vec<u8> = (0..0x100)
                        .map(|a| vm.mem.read(0x1_0000 + a, 1).unwrap() as u8)
                        .collect();
                    if stack_now != stack || strings_now != strings {
                        failures.push(format!("{what}: RAM differs from the RAM run"));
                    }
                }
            }
            // One line per row, so a run names each row's outcome.
            let word = want
                .iter()
                .any(|b| matches!(b, Bus::Read(_, 2) | Bus::Write(_, 2, _)));
            println!(
                "row {what} [{}]: {}",
                if word { "word" } else { "bytes" },
                if failures.len() == before {
                    "ok"
                } else {
                    "FAIL"
                }
            );
        }
    }
    assert!(checked > 0);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

const R: fn(u64, u8) -> Op = Op::Read;
const W: fn(u64, u8) -> Op = Op::Write;

/// Rules 1 and 2: a word at an even address in one device region is one
/// transfer on the 8086 and V30; at an odd address it is two byte transfers,
/// the low byte first at the address, then the high byte at the address plus
/// one (UM §4.2 p. 4-15; NEC UM §4 p. 4-3). The 8088 and V20 move bytes.
#[test]
fn scalar_words_follow_the_bus_width() {
    let mut cases = Vec::new();
    for di in [0x10u64, 0x11] {
        let at = phys(SEG, di as u16);
        cases.push(
            case(&format!("mov ax,[{di:#x}]"), &[0x8b, 0x05])
                .gpr(7, di)
                .ops(&[R(at, 2)]),
        );
        cases.push(
            case(&format!("mov [{di:#x}],ax"), &[0x89, 0x05])
                .gpr(7, di)
                .ops(&[W(at, 2)]),
        );
        cases.push(
            case(&format!("mov al,[{di:#x}]"), &[0x8a, 0x05])
                .gpr(7, di)
                .ops(&[R(at, 1)]),
        );
        cases.push(
            case(&format!("add [{di:#x}],ax"), &[0x01, 0x05])
                .gpr(7, di)
                .ops(&[R(at, 2), W(at, 2)]),
        );
        cases.push(
            case(&format!("inc word [{di:#x}]"), &[0xff, 0x05])
                .gpr(7, di)
                .ops(&[R(at, 2), W(at, 2)]),
        );
        // An unlocked XCHG is a read, then a write (UM §2.5 p. 2-18).
        cases.push(
            case(&format!("xchg [{di:#x}],ax"), &[0x87, 0x05])
                .gpr(7, di)
                .ops(&[R(at, 2), W(at, 2)]),
        );
    }
    witness(&cases);
}

/// Rule 3: a word that wraps its segment (offset FFFFh, UM §2.3 p. 2-12) or the
/// twenty-bit bus (FFFFFh, §2.8 p. 2-68) starts at an odd address, so it is two
/// byte transfers on every model, the second at the wrapped address.
#[test]
fn wrapped_words_stay_byte_transfers() {
    let segment = [(0xb_0000u64, 0x10u64), (0xb_fff0, 0x10)];
    let bus = [(0x0u64, 0x10u64), (0xf_fff0, 0x10)];
    let mut cases = Vec::new();
    for (name, code, rmw) in [
        ("mov ax,[di]", &[0x8b, 0x05][..], false),
        ("mov [di],ax", &[0x89, 0x05][..], false),
        ("add [di],ax", &[0x01, 0x05][..], true),
        ("lock add [di],ax", &[0xf0, 0x01, 0x05][..], true),
    ] {
        for (edge, traps, ds, di, low, high) in [
            (
                "segment",
                &segment,
                0xb000u16,
                0xffffu64,
                0xb_ffffu64,
                0xb_0000u64,
            ),
            ("bus", &bus, 0xffff, 0x000f, 0xf_ffff, 0x0),
        ] {
            let ops: Vec<Op> = if rmw {
                vec![R(low, 1), R(high, 1), W(low, 1), W(high, 1)]
            } else if name.starts_with("mov ax") {
                vec![R(low, 1), R(high, 1)]
            } else {
                vec![W(low, 1), W(high, 1)]
            };
            cases.push(
                case(&format!("{name} {edge} wrap"), code)
                    .segs(ds, ds, 0)
                    .gpr(7, di)
                    .traps(traps)
                    .ops(&ops),
            );
        }
    }
    witness(&cases);
}

/// Rule 4: far pointers, stack words and frames. Each word of the operand
/// follows rules 1 and 2. The frame rows pin the engine's store order: an
/// interrupt frame from SP-6 upward (IP, CS, FLAGS), a far call frame from SP-4
/// upward (IP, CS). That order is a known divergence from the sources, which
/// push FLAGS, then CS, then IP (8086 Family UM §2.6 p. 2-26; AP-67, UM
/// p. A-28) and, for an intersegment CALL, CS, then IP (§2.7 p. 2-44); owned by
/// xj-50w.25 (spec.md §5.2).
#[test]
fn multi_word_operands_follow_the_bus_width() {
    let mut cases = Vec::new();
    for di in [0x10u64, 0x11] {
        let at = phys(SEG, di as u16);
        for (name, code) in [("les bx", [0xc4, 0x1d]), ("lds bx", [0xc5, 0x1d])] {
            cases.push(
                case(&format!("{name},[{di:#x}]"), &code)
                    .gpr(7, di)
                    .ops(&[R(at, 2), R(at + 2, 2)]),
            );
        }
    }
    for sp in [0x20u64, 0x21] {
        let stack = |delta: i64| phys(SEG, (sp as i64 + delta) as u16);
        let on_device = |c: Case| c.segs(SEG, SEG, SEG).gpr(4, sp);
        cases
            .push(on_device(case(&format!("push ax sp={sp:#x}"), &[0x50])).ops(&[W(stack(-2), 2)]));
        cases.push(on_device(case(&format!("pop bx sp={sp:#x}"), &[0x5b])).ops(&[R(stack(0), 2)]));
        cases.push(on_device(case(&format!("pushf sp={sp:#x}"), &[0x9c])).ops(&[W(stack(-2), 2)]));
        cases.push(
            on_device(case(&format!("int 20h sp={sp:#x}"), &[0xcd, 0x20])).ops(&[
                W(stack(-6), 2),
                W(stack(-4), 2),
                W(stack(-2), 2),
            ]),
        );
        cases.push(
            on_device(case(
                &format!("call far 0:500 sp={sp:#x}"),
                &[0x9a, 0x00, 0x05, 0x00, 0x00],
            ))
            .ops(&[W(stack(-4), 2), W(stack(-2), 2)]),
        );
        // A frame that returns to the HLT at 0000:0500.
        let frame = [0x00, 0x05, 0x00, 0x00, 0x02, 0x00];
        cases.push(
            on_device(case(&format!("iret sp={sp:#x}"), &[0xcf]))
                .preload(stack(0), &frame)
                .ops(&[R(stack(0), 2), R(stack(2), 2), R(stack(4), 2)]),
        );
        cases.push(
            on_device(case(&format!("retf sp={sp:#x}"), &[0xcb]))
                .preload(stack(0), &frame[..4])
                .ops(&[R(stack(0), 2), R(stack(2), 2)]),
        );
    }
    // The interrupt vector table on a device: two words at an even address
    // (NEC UM §4 p. 4-3), then the frame in RAM.
    cases.push(
        case("int 20h, IVT on a device", &[0xcd, 0x20])
            .traps(&[(0x80, 4)])
            .preload(0x80, &[0x00, 0x05, 0x00, 0x00])
            .ops(&[R(0x80, 2), R(0x82, 2)]),
    );
    witness(&cases);
}

/// Rule 4: string elements. A word element follows rules 1 and 2; a REP string
/// repeats them per element.
#[test]
fn string_words_follow_the_bus_width() {
    let ram = 0x1000u16;
    let mut cases = Vec::new();
    for offset in [0x10u64, 0x11] {
        let dev = phys(SEG, offset as u16);
        let dev2 = phys(SEG, offset as u16 + 0x20);
        let o = |c: Case| c.gpr(6, offset).gpr(7, offset);
        cases.push(
            o(case(&format!("movsw dev->ram {offset:#x}"), &[0xa5]).segs(SEG, ram, 0))
                .ops(&[R(dev, 2)]),
        );
        cases.push(
            o(case(&format!("movsw ram->dev {offset:#x}"), &[0xa5]).segs(ram, SEG, 0))
                .ops(&[W(dev, 2)]),
        );
        cases.push(
            case(&format!("movsw dev->dev {offset:#x}"), &[0xa5])
                .gpr(6, offset)
                .gpr(7, offset + 0x20)
                .ops(&[R(dev, 2), W(dev2, 2)]),
        );
        cases.push(o(case(&format!("stosw {offset:#x}"), &[0xab])).ops(&[W(dev, 2)]));
        cases.push(o(case(&format!("lodsw {offset:#x}"), &[0xad])).ops(&[R(dev, 2)]));
        cases.push(
            case(&format!("cmpsw {offset:#x}"), &[0xa7])
                .gpr(6, offset)
                .gpr(7, offset + 0x20)
                .ops(&[R(dev, 2), R(dev2, 2)]),
        );
        cases.push(o(case(&format!("scasw {offset:#x}"), &[0xaf])).ops(&[R(dev, 2)]));
        cases.push(
            o(case(&format!("movsb {offset:#x}"), &[0xa4]).segs(SEG, ram, 0)).ops(&[R(dev, 1)]),
        );
        cases.push(
            case(&format!("rep movsw dev->dev {offset:#x}"), &[0xf3, 0xa5])
                .gpr(1, 2)
                .gpr(6, offset)
                .gpr(7, offset + 0x20)
                .ops(&[R(dev, 2), W(dev2, 2), R(dev + 2, 2), W(dev2 + 2, 2)]),
        );
    }
    witness(&cases);
}

/// Rule 4: the V20/V30 block forms. OUTM reads its word from memory, then
/// outputs it; INM inputs a word, then stores it (NEC UM pp. 12-28–29); the
/// bit-field forms move their words (pp. 12-21–24). PUSH R
/// stores eight words from SP-2 downward; POP R reads seven, skipping SP+6
/// (pp. 12-148, 12-152).
#[test]
fn nec_block_and_stack_forms_follow_the_bus_width() {
    let mut cases = Vec::new();
    for offset in [0x10u64, 0x11] {
        let dev = phys(SEG, offset as u16);
        cases.push(
            case(&format!("outm {offset:#x}"), &[0x6f])
                .nec_only()
                .gpr(6, offset)
                .ops(&[R(dev, 2), Op::PortOut(2)]),
        );
        cases.push(
            case(&format!("inm {offset:#x}"), &[0x6d])
                .nec_only()
                .gpr(7, offset)
                .ops(&[Op::PortIn(2), W(dev, 2)]),
        );
        cases.push(
            case(&format!("rep outm {offset:#x}"), &[0xf3, 0x6f])
                .nec_only()
                .gpr(1, 2)
                .gpr(6, offset)
                .ops(&[R(dev, 2), Op::PortOut(2), R(dev + 2, 2), Op::PortOut(2)]),
        );
    }
    // EXT CL,DL and INS CL,DL (0F 33/31 D1): the field's word at DS0:IX
    // (EXT) or DS1:IY (INS), and the next word when the field reaches it
    // ("Transfers: 1 or 2" and "2 or 4", NEC UM pp. 12-21–24).
    for offset in [0x10u64, 0x11] {
        let dev = phys(SEG, offset as u16);
        cases.push(
            case(&format!("ext cl,dl {offset:#x}"), &[0x0f, 0x33, 0xd1])
                .nec_only()
                .gpr(1, 0x04)
                .gpr(2, 0x07)
                .gpr(6, offset)
                .ops(&[R(dev, 2)]),
        );
        cases.push(
            case(&format!("ins cl,dl {offset:#x}"), &[0x0f, 0x31, 0xd1])
                .nec_only()
                .gpr(1, 0x0c)
                .gpr(2, 0x07)
                .gpr(7, offset)
                .ops(&[R(dev, 2), R(dev + 2, 2), W(dev, 2), W(dev + 2, 2)]),
        );
    }
    for sp in [0x20u64, 0x21] {
        let stack = |delta: i64| phys(SEG, (sp as i64 + delta) as u16);
        let pushes: Vec<Op> = (1..=8).map(|i| W(stack(-2 * i), 2)).collect();
        cases.push(
            case(&format!("push r sp={sp:#x}"), &[0x60])
                .nec_only()
                .segs(SEG, SEG, SEG)
                .gpr(4, sp)
                .ops(&pushes),
        );
        let pops: Vec<Op> = [0, 2, 4, 8, 10, 12, 14]
            .iter()
            .map(|&d| R(stack(d), 2))
            .collect();
        cases.push(
            case(&format!("pop r sp={sp:#x}"), &[0x61])
                .nec_only()
                .segs(SEG, SEG, SEG)
                .gpr(4, sp)
                .ops(&pops),
        );
    }
    witness(&cases);
}

/// Rule 5: a locked word inside one device region is one read, then one write,
/// of its bus transfers: one word each on the 8086 and V30 at an even address,
/// bytes otherwise. The vCPU stays on the instruction across both exits: no
/// transfer is seen after it retires.
#[test]
fn locked_words_on_a_device_follow_the_bus_width() {
    let mut cases = Vec::new();
    for di in [0x10u64, 0x11] {
        let at = phys(SEG, di as u16);
        for (name, code) in [
            ("lock add", &[0xf0, 0x01, 0x05][..]),
            ("lock xchg", &[0xf0, 0x87, 0x05][..]),
            ("lock not", &[0xf0, 0xf7, 0x15][..]),
        ] {
            cases.push(
                case(&format!("{name} [{di:#x}]"), code)
                    .gpr(7, di)
                    .ops(&[R(at, 2), W(at, 2)]),
            );
        }
    }
    witness(&cases);
}

/// Rule 5: a locked word over two device regions keeps the typed refusal on
/// every model, with no transfer and nothing committed: one exit names one
/// region. A locked word over a device byte and a RAM byte is performed since
/// xj-ir6.34 (`locked_split_device.rs`).
#[test]
fn locked_word_over_two_device_regions_is_refused() {
    for model in MODELS {
        // B800:0010 is the first region's last device byte; 0011 is the first
        // byte of a second device region.
        let traps = &[(0xb_8000u64, 0x11u64), (0xb_8011, 0xef)][..];
        let c = case("lock add [di],ax", &[0xf0, 0x01, 0x05]).traps(traps);
        let (vm, mut cpu) = setup(model, &c, true);
        let mut device = Device::new(&c);
        let run = drive(&vm, &mut cpu, 64, Some(&mut device), Resume::Run, None);
        assert!(run.bus.is_empty(), "{model:?}: {:x?}", run.bus);
        assert_eq!(
            run.observation.stop,
            format!("UnknownInstruction@{CODE:#x}+3"),
            "{model:?}"
        );
        assert_eq!(run.observation.retired, 0, "{model:?}");
        assert_eq!(cpu.cpu.gpr[0], 0xc35a, "{model:?}");
    }
}

/// Rule 7: an even word over a device byte and a RAM byte, or over two device
/// regions, keeps its byte transfers on every model: one exit names one region.
#[test]
fn even_words_over_two_regions_keep_byte_transfers() {
    let mut cases = Vec::new();
    let at = phys(SEG, 0x10);
    for (name, traps) in [
        ("device then RAM", &[(0xb_8000u64, 0x11u64)][..]),
        ("RAM then device", &[(0xb_8011, 0xef)][..]),
        ("two devices", &[(0xb_8000, 0x11), (0xb_8011, 0xef)][..]),
    ] {
        cases.push(
            case(&format!("mov ax,[di] {name}"), &[0x8b, 0x05])
                .traps(traps)
                .ops(&[R(at, 2)]),
        );
        cases.push(
            case(&format!("mov [di],ax {name}"), &[0x89, 0x05])
                .traps(traps)
                .ops(&[W(at, 2)]),
        );
        cases.push(
            case(&format!("add [di],ax {name}"), &[0x01, 0x05])
                .traps(traps)
                .ops(&[R(at, 2), W(at, 2)]),
        );
        cases.push(
            case(&format!("stosw {name}"), &[0xab])
                .traps(traps)
                .ops(&[W(at, 2)]),
        );
        cases.push(
            case(&format!("lodsw {name}"), &[0xad])
                .traps(traps)
                .ops(&[R(at, 2)]),
        );
    }
    // A locked word over two device regions keeps its typed refusal:
    // `locked_word_over_device_and_ram_is_refused`.
    witness(&cases);
}

/// A device stop in the middle of an access of several transfers, then a
/// budget stop and a host stop before the retry, on the 8086: the retry ledger
/// keeps each acknowledged word, and the retry asks only for the next one
/// (spec.md §5.2). The frame of INT 20h is three word writes, PUSH of a
/// far-call frame two, and REP MOVSW two words per element.
#[test]
fn device_stops_inside_a_multi_word_access_resume_on_the_8086() {
    let sp = 0x20u64;
    let stack = |delta: i64| phys(SEG, (sp as i64 + delta) as u16);
    let c = case("int 20h", &[0xcd, 0x20])
        .segs(SEG, SEG, SEG)
        .gpr(4, sp)
        .ops(&[W(stack(-6), 2), W(stack(-4), 2), W(stack(-2), 2)]);
    let (vm, mut cpu) = setup(CpuModel::I8086, &c, true);
    let mut device = Device::new(&c);
    let run = drive(
        &vm,
        &mut cpu,
        64,
        Some(&mut device),
        Resume::Boundaries,
        None,
    );
    assert_eq!(
        run.bus
            .iter()
            .map(|b| match b {
                Bus::Write(a, n, _) => (*a, *n),
                other => panic!("{other:?}"),
            })
            .collect::<Vec<_>>(),
        [(0xb_801a, 2), (0xb_801c, 2), (0xb_801e, 2)]
    );
    // IP after INT 20h, then CS 0.
    assert_eq!(run.bus[0], Bus::Write(0xb_801a, 2, CODE + 2));
    assert_eq!(run.bus[1], Bus::Write(0xb_801c, 2, 0));
    assert_eq!(run.observation.stop, "Hlt");
    assert_eq!(run.observation.rip, HANDLER + 1);
    assert_eq!(cpu.cpu.gpr[4], sp - 6);
    assert!(run.retired_at_bus.iter().all(|&r| r == 0));
    // Two elements of two word transfers each; every witness above also runs
    // this schedule (`Resume::Boundaries`).
    let c = case("rep movsw", &[0xf3, 0xa5]).gpr(1, 2).gpr(7, 0x30);
    let (vm, mut cpu) = setup(CpuModel::I8086, &c, true);
    let mut device = Device::new(&c);
    let run = drive(
        &vm,
        &mut cpu,
        64,
        Some(&mut device),
        Resume::Boundaries,
        None,
    );
    assert_eq!(
        run.bus,
        [
            Bus::Read(0xb_8010, 2),
            Bus::Write(0xb_8030, 2, device.read(0xb_8030, 2)),
            Bus::Read(0xb_8012, 2),
            Bus::Write(0xb_8032, 2, device.read(0xb_8032, 2)),
        ]
    );
    assert_eq!(
        device.read(0xb_8030, 4),
        u64::from(seed(0xb_8010))
            | u64::from(seed(0xb_8011)) << 8
            | u64::from(seed(0xb_8012)) << 16
            | u64::from(seed(0xb_8013)) << 24
    );
    assert_eq!(
        (cpu.cpu.gpr[1], cpu.cpu.gpr[6], cpu.cpu.gpr[7]),
        (0, 0x14, 0x34)
    );
}

/// Rule 4: with no coprocessor, a memory ESC (8086 Family UM §2.7 p. 2-48) and
/// the V20/V30 FPO2 (NEC UM p. 12-175: "Transfers: 1") read their word operand
/// and discard it: one word transfer at an even address on a sixteen-bit bus.
#[test]
fn coprocessor_escape_reads_follow_the_bus_width() {
    let mut cases = Vec::new();
    for di in [0x10u64, 0x11] {
        let at = phys(SEG, di as u16);
        cases.push(
            case(&format!("esc [{di:#x}]"), &[0xd8, 0x05])
                .no_fpu()
                .gpr(7, di)
                .ops(&[R(at, 2)]),
        );
        cases.push(
            case(&format!("fpo2 [{di:#x}]"), &[0x66, 0x05])
                .nec_only()
                .no_fpu()
                .gpr(7, di)
                .ops(&[R(at, 2)]),
        );
    }
    witness(&cases);
}

/// Rule 4: the frames of the run-time services that read the bus width from
/// the vCPU's catalogue entry, on a device stack: external interrupt delivery,
/// exception delivery (divide error), INT3, INTO, the V20/V30 CHKIND (its
/// bounds, then its BRK 5 frame, NEC UM p. 12-170), BRKEM (p. 12-169), and the
/// 8080-mode escapes RETEM (p. 12-177) and CALLN (p. 12-176) after BRKEM. The
/// rows pin the engine's frame store order, a known divergence (xj-50w.25,
/// spec.md §5.2).
#[test]
fn service_frames_follow_the_bus_width() {
    let mut cases = Vec::new();
    for sp in [0x20u64, 0x21] {
        let stack = |delta: i64| phys(SEG, (sp as i64 + delta) as u16);
        let frame = [W(stack(-6), 2), W(stack(-4), 2), W(stack(-2), 2)];
        let on_device = |c: Case| c.segs(SEG, SEG, SEG).gpr(4, sp);
        cases.push(
            on_device(case(&format!("irq 20h sp={sp:#x}"), &[0x90]))
                .prepare(|cpu| {
                    cpu.cpu.flags.if_ = true;
                    cpu.inject_irq(0x20);
                })
                .ops(&frame),
        );
        cases.push(
            on_device(case(&format!("div bl=0 sp={sp:#x}"), &[0xf6, 0xf3]))
                .gpr(3, 0)
                .ops(&frame),
        );
        cases.push(on_device(case(&format!("int3 sp={sp:#x}"), &[0xcc])).ops(&frame));
        cases.push(
            on_device(case(&format!("into sp={sp:#x}"), &[0xce]))
                .prepare(|cpu| cpu.cpu.flags.of = true)
                .ops(&frame),
        );
        // CHKIND AW,[IY]: AX = C35Ah lies outside 1000h..2000h whether the
        // bounds are read signed or unsigned. The bounds share the parity of SP.
        let bounds = phys(SEG, 0x40 + sp as u16 - 0x20);
        let mut ops = vec![R(bounds, 2), R(bounds + 2, 2)];
        ops.extend(frame);
        cases.push(
            on_device(case(&format!("chkind sp={sp:#x}"), &[0x62, 0x05]))
                .nec_only()
                .gpr(7, 0x40 + sp - 0x20)
                .preload(bounds, &[0x00, 0x10, 0x00, 0x20])
                .ops(&ops),
        );
        // BRKEM 20h enters 8080 mode at 0000:0500; the hand-back escapes at
        // 0600h (RETEM: the BRKEM frame read back, then the HLT after BRKEM)
        // or at 0610h (CALLN 20h: a second frame below it, then the HLT at
        // 0000:0500 in native mode).
        let mut retem = frame.to_vec();
        retem.extend([R(stack(-6), 2), R(stack(-4), 2), R(stack(-2), 2)]);
        cases.push(
            on_device(case(
                &format!("brkem 20h, retem sp={sp:#x}"),
                &[0x0f, 0xff, 0x20],
            ))
            .nec_only()
            .preload(0x600, &[0xed, 0xfd])
            .escape(0x600)
            .ops(&retem),
        );
        let mut calln = frame.to_vec();
        calln.extend([W(stack(-12), 2), W(stack(-10), 2), W(stack(-8), 2)]);
        cases.push(
            on_device(case(
                &format!("brkem 20h, calln 20h sp={sp:#x}"),
                &[0x0f, 0xff, 0x20],
            ))
            .nec_only()
            .preload(0x610, &[0xed, 0xed, 0x20])
            .escape(0x610)
            .ops(&calln),
        );
    }
    witness(&cases);
}

/// A fault after INT, INT3, INTO or BRKEM has read the interrupt vector table
/// from a device reports the transfers it completed as plain addresses, a word
/// entry at an even address on the 8086 and V30 (spec.md §5.2): here the frame
/// store faults on an unmapped stack after both vector words were answered.
#[test]
fn fault_after_a_device_ivt_read_reports_plain_addresses() {
    let mut failures = Vec::new();
    for model in MODELS {
        for (name, code, vector, nec_only, of) in [
            ("int 20h", &[0xcd, 0x20][..], 0x20u64, false, false),
            ("int3", &[0xcc][..], 3, false, false),
            ("into", &[0xce][..], 4, false, true),
            ("brkem 20h", &[0x0f, 0xff, 0x20][..], 0x20, true, false),
        ] {
            if nec_only && !nec(model) {
                continue;
            }
            let ivt = vector * 4;
            let mut c = case(name, code)
                .traps(&[(ivt, 4)])
                .segs(SEG, SEG, 0x9000)
                .gpr(4, 0x100)
                .preload(ivt, &[0x00, 0x05, 0x00, 0x00]);
            if of {
                c = c.prepare(|cpu| cpu.cpu.flags.of = true);
            }
            let (mut vm, mut cpu) = setup(model, &c, true);
            // Leave 9000:0000-0FFF unmapped: remap the RAM above the vector
            // around it, then rewrite the code.
            vm.unmap(ivt + 4, (0x10_0000 - ivt - 4) as usize).unwrap();
            vm.map(
                ivt + 4,
                (0x9_0000 - ivt - 4) as usize,
                Prot::RWX,
                RegionKind::Ram,
            )
            .unwrap();
            vm.map(0x9_1000, 0x6_f000, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.write_bytes(CODE, &[code, &[0xf4][..]].concat()).unwrap();
            let device = Device::new(&c);
            let mut reads = Vec::new();
            let acknowledged = loop {
                let run = cpu.run_steps(&vm, 64, StepOptions::default()).unwrap();
                match run.stop() {
                    StepStop::Exit(Exit::MmioRead { addr, size, .. }) => {
                        reads.push((*addr, *size));
                        cpu.complete_mmio_read(device.read(*addr, *size));
                    }
                    StepStop::Exit(exit @ Exit::UnmappedMemory { addr, .. })
                        if (0x9_00fa..0x9_0100).contains(addr) =>
                    {
                        break exit
                            .fault_metadata()
                            .expect("fault metadata")
                            .progress()
                            .acknowledged_transfers()
                            .to_vec();
                    }
                    other => panic!("{model:?} {name}: {other:?}"),
                }
            };
            let want: Vec<(u64, u64)> = if word_bus(model) {
                vec![(ivt, 0x0500), (ivt + 2, 0)]
            } else {
                (0..4).map(|i| (ivt + i, device.read(ivt + i, 1))).collect()
            };
            let ok = acknowledged == want;
            println!(
                "row {model:?} {name} [{}]: {}",
                if word_bus(model) { "word" } else { "bytes" },
                if ok { "ok" } else { "FAIL" }
            );
            if !ok {
                failures.push(format!(
                    "{model:?} {name}: reads {reads:x?}, acknowledged {acknowledged:x?}, expected {want:x?}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
