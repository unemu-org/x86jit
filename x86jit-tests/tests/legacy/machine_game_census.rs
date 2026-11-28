//! Compiled Machine coverage on a flat protected-mode game loop, and the
//! census of why its blocks ran interpreted (`MachineJitCounters::refusals`,
//! `no_unit`).
//!
//! The guest is a CPL-0 Flat32 program without paging, as a game written for
//! a 32-bit DOS extender runs: a tight arithmetic loop, a span renderer
//! called sixteen times a frame, REP STOSD and REP MOVSD over a frame buffer,
//! a keyboard read and a palette upload through ports served in process, a
//! timer interrupt through a 32-bit interrupt gate whose handler writes the
//! controller's end-of-interrupt port, and, every eighth frame, a call into
//! 16-bit real-address code by the switch of SDM Vol 3A §10.9.2 and back by
//! §10.9.1. The embedder loop runs fixed slices, raises INTR at a fixed
//! step interval and injects the vector at `IntrReady`.
//!
//! A second game calls 512 small routines once a frame, a working set of
//! about 1,500 blocks, wider than the per-vCPU memo, with a timer interrupt
//! every 3,000 steps; its cold census counts the turns by block key.
//!
//! The subject uses the production background tier-up; an interpreted
//! reference runs the same calls, and every call's stop, state, retired
//! count and RAM must agree.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::descriptor;
use iced_x86::code_asm::*;
use x86jit_core::{
    CpuModel, DeviceContext, DeviceDirection, DeviceIo, DeviceRange, DeviceRequest, DeviceResult,
    DeviceStop, ExceptionDelivery, Exit, MachineJitCounters, MachineJitMode, Prot, RegionKind,
    StepOptions, StepStop, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const SIZE: u64 = 0x40000;
const MAIN: u64 = 0x1000;
const RENDER: u64 = 0x1600;
const HANDLER: u64 = 0x1800;
const PM16: u64 = 0x3000;
const REAL16: u64 = 0x3400;
const BACK32: u64 = 0x3800;
const FRAMES: u32 = 0x5000;
const TICKS: u32 = 0x5004;
const SAVED_ESP: u32 = 0x5008;
const KEYS: u32 = 0x500c;
const REAL_RESULT: u16 = 0x5010;
const FRAME_COUNT: u32 = 120;
/// Steps per embedder call, and between timer INTR assertions.
const SLICE: u64 = 5_000;
const TICK: u64 = 20_000;

fn asm(bits: u32, at: u64, build: impl FnOnce(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(bits).unwrap();
    build(&mut a);
    a.assemble(at).unwrap()
}

/// The game's code pieces, each at its address.
fn pieces() -> Vec<(u64, Vec<u8>)> {
    let main = asm(32, MAIN, |a| {
        let mut frame = a.create_label();
        let mut inner = a.create_label();
        let mut render = a.create_label();
        let mut palette = a.create_label();
        let mut skip = a.create_label();
        let mut real_call = a.create_label();
        a.sti().unwrap();
        a.set_label(&mut frame).unwrap();
        a.mov(ecx, 2000u32).unwrap();
        a.set_label(&mut inner).unwrap();
        a.add(eax, ecx).unwrap();
        a.xor(ebx, eax).unwrap();
        a.dec(ecx).unwrap();
        a.jne(inner).unwrap();
        a.mov(edi, 0x20000u32).unwrap();
        a.mov(edx, 16u32).unwrap();
        a.set_label(&mut render).unwrap();
        a.call(RENDER).unwrap();
        a.dec(edx).unwrap();
        a.jne(render).unwrap();
        a.cld().unwrap();
        a.mov(edi, 0x20000u32).unwrap();
        a.mov(ecx, 1000u32).unwrap();
        a.xor(eax, eax).unwrap();
        a.rep().stosd().unwrap();
        a.mov(esi, 0x20000u32).unwrap();
        a.mov(edi, 0x30000u32).unwrap();
        a.mov(ecx, 1000u32).unwrap();
        a.rep().movsd().unwrap();
        a.in_(al, 0x60).unwrap();
        a.movzx(edx, al).unwrap();
        a.add(dword_ptr(KEYS), edx).unwrap();
        a.mov(dx, 0x3c8).unwrap();
        a.xor(al, al).unwrap();
        a.out(dx, al).unwrap();
        a.inc(dx).unwrap();
        a.mov(ecx, 48u32).unwrap();
        a.set_label(&mut palette).unwrap();
        a.mov(al, cl).unwrap();
        a.out(dx, al).unwrap();
        a.dec(ecx).unwrap();
        a.jne(palette).unwrap();
        a.inc(dword_ptr(FRAMES)).unwrap();
        a.test(byte_ptr(FRAMES), 7).unwrap();
        a.jne(skip).unwrap();
        a.call(real_call).unwrap();
        a.set_label(&mut skip).unwrap();
        a.cmp(dword_ptr(FRAMES), FRAME_COUNT).unwrap();
        a.jb(frame).unwrap();
        a.cli().unwrap();
        a.hlt().unwrap();
        // SDM Vol 3A §10.9.2: interrupts off, then a far jump to a 16-bit
        // code segment.
        a.set_label(&mut real_call).unwrap();
        a.cli().unwrap();
        a.mov(dword_ptr(SAVED_ESP), esp).unwrap();
        a.jmp_far(0x18, PM16 as u32).unwrap();
    });
    let render = asm(32, RENDER, |a| {
        let mut span = a.create_label();
        a.mov(ecx, 64u32).unwrap();
        a.set_label(&mut span).unwrap();
        a.mov(dword_ptr(edi), eax).unwrap();
        a.add(edi, 4).unwrap();
        a.add(eax, 3).unwrap();
        a.dec(ecx).unwrap();
        a.jne(span).unwrap();
        a.ret().unwrap();
    });
    let handler = asm(32, HANDLER, |a| {
        a.push(eax).unwrap();
        a.inc(dword_ptr(TICKS)).unwrap();
        a.mov(al, 0x20).unwrap();
        a.out(0x20, al).unwrap();
        a.pop(eax).unwrap();
        a.iretd().unwrap();
    });
    // 16-bit protected mode: real-mode-compatible data selectors, PE clear,
    // and a far jump that loads a real-mode CS.
    let pm16 = asm(16, PM16, |a| {
        a.mov(ax, 0x20).unwrap();
        a.mov(ds, ax).unwrap();
        a.mov(es, ax).unwrap();
        a.mov(ss, ax).unwrap();
        a.mov(eax, cr0).unwrap();
        a.and(al, 0xfe).unwrap();
        a.mov(cr0, eax).unwrap();
        a.jmp_far(0, REAL16 as u32).unwrap();
    });
    // Real mode: some work, then PE set (§10.9.1) and a far jump to 32-bit code.
    let real16 = asm(16, REAL16, |a| {
        let mut work = a.create_label();
        a.xor(ax, ax).unwrap();
        a.mov(ds, ax).unwrap();
        a.mov(es, ax).unwrap();
        a.mov(ss, ax).unwrap();
        a.mov(sp, 0x7000).unwrap();
        a.mov(cx, 200).unwrap();
        a.set_label(&mut work).unwrap();
        a.add(bx, cx).unwrap();
        a.rol(bx, 1).unwrap();
        a.loop_(work).unwrap();
        a.mov(word_ptr(REAL_RESULT), bx).unwrap();
        a.mov(eax, cr0).unwrap();
        a.or(al, 1).unwrap();
        a.mov(cr0, eax).unwrap();
        a.jmp_far(0x08, BACK32 as u32).unwrap();
    });
    let back32 = asm(32, BACK32, |a| {
        a.mov(ax, 0x10).unwrap();
        a.mov(ds, ax).unwrap();
        a.mov(es, ax).unwrap();
        a.mov(ss, ax).unwrap();
        a.mov(esp, dword_ptr(SAVED_ESP)).unwrap();
        a.sti().unwrap();
        a.ret().unwrap();
    });
    vec![
        (MAIN, main),
        (RENDER, render),
        (HANDLER, handler),
        (PM16, pm16),
        (REAL16, real16),
        (BACK32, back32),
    ]
}

/// The keyboard, palette and interrupt-controller ports, served in process.
#[derive(Default)]
struct Ports {
    requests: Vec<(u64, u64)>,
}

impl DeviceIo for Ports {
    fn access(&mut self, request: &DeviceRequest, _: &mut DeviceContext<'_>) -> DeviceResult {
        self.requests.push((request.address, request.value));
        match request.direction {
            DeviceDirection::Read => DeviceResult::Handled(self.requests.len() as u64 & 0x7f),
            _ => DeviceResult::Handled(0),
        }
    }
}

fn machine(background: bool) -> (Vm, Vcpu) {
    machine_with(pieces(), background)
}

fn machine_with(pieces: Vec<(u64, Vec<u8>)>, background: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(SIZE), Box::new(JitBackend::new()));
    vm.map(0, SIZE as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.set_cpu_model(CpuModel::I80386);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    for ports in [0x20..0x22, 0x60..0x61, 0x3c8..0x3ca] {
        vm.register_device_range(DeviceRange::Ports(ports)).unwrap();
    }
    vm.set_machine_jit_for_test(MachineJitMode::Off);
    if background {
        vm.set_tier_up_after(Some(50));
        vm.set_tier_up_background(true);
    }
    for (at, code) in pieces {
        vm.write_bytes(at, &code).unwrap();
    }
    descriptor(&vm, 0x2008, 0, u32::MAX, 0xc09b);
    descriptor(&vm, 0x2010, 0, u32::MAX, 0xc093);
    descriptor(&vm, 0x2018, 0, 0xffff, 0x9b);
    descriptor(&vm, 0x2020, 0, 0xffff, 0x93);
    // 80386 PRM §9.5: a 32-bit interrupt gate, vector 20h.
    let gate = [HANDLER as u8, (HANDLER >> 8) as u8, 8, 0, 0, 0x8e, 0, 0];
    vm.write_bytes(0x2800 + 0x20 * 8, &gate).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = MAIN;
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.gdtr.base = 0x2000;
    cpu.cpu.legacy.gdtr.limit = 0x2f;
    cpu.cpu.legacy.idtr.base = 0x2800;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        cache.base = 0;
        cache.limit = u32::MAX;
        cache.attributes = if i == 0 { 0x409b } else { 0x4093 };
    }
    cpu.cpu.cs = 8;
    cpu.cpu.ds = 16;
    cpu.cpu.es = 16;
    cpu.cpu.ss = 16;
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}

fn print_census(c: &MachineJitCounters, retired: u64) {
    println!(
        "retired {retired}, compiled_retired {} ({:.1} %), units {}, entries {}, refused {}",
        c.compiled_retired,
        100.0 * c.compiled_retired as f64 / retired as f64,
        c.compiled,
        c.entries,
        c.refused
    );
    for ((name, n), r) in MachineJitCounters::REFUSAL_NAMES
        .iter()
        .zip(c.refusals)
        .zip(c.refusals_retired)
    {
        if n != 0 {
            println!("  refusal {name}: {n} turns, {r} retired");
        }
    }
    for ((name, n), r) in MachineJitCounters::NO_UNIT_NAMES
        .iter()
        .zip(c.no_unit)
        .zip(c.no_unit_retired)
    {
        if n != 0 {
            println!("  no unit {name}: {n} turns, {r} retired");
        }
    }
    for (name, n) in MachineJitCounters::CHAIN_END_NAMES.iter().zip(c.chain_ends) {
        if n != 0 {
            println!("  chain end {name}: {n}");
        }
    }
    for (name, n) in MachineJitCounters::DEFER_CAUSE_NAMES
        .iter()
        .zip(c.defer_causes)
    {
        if n != 0 {
            println!("  defer {name}: {n}");
        }
    }
    println!(
        "  unconditional defers {}, helper defers {}",
        c.unconditional_defers, c.helper_defers
    );
}

/// The class census (xj-to9.79): which instruction classes kept blocks off
/// the ordinary walk, and which boundary terms forced one step.
fn print_classes(cpu: &Vcpu) {
    let Some(census) = cpu.class_census_for_test() else {
        return;
    };
    let mut rows: Vec<_> = census.classes.into_iter().collect();
    rows.sort_by_key(|row| std::cmp::Reverse(row.1 .0));
    for ((row, class), (turns, retired)) in rows {
        println!("  class {row} / {class}: {turns} turns, {retired} retired");
    }
}

/// Run the subject `a` and the interpreted reference `b` in lockstep to HLT,
/// asserting INTR every `tick` steps, and compare every call, the device
/// transcripts and RAM.
/// Returns the host instructions the subject's calls took (macOS only, 0
/// elsewhere): the process's count, background compiles included.
fn lockstep(avm: &Vm, a: &mut Vcpu, bvm: &Vm, b: &mut Vcpu, tick: u64) -> u64 {
    let (mut ad, mut bd) = (Ports::default(), Ports::default());
    let mut host = 0;
    let mut steps = 0;
    let mut next_tick = tick;
    for call in 0.. {
        assert!(call < 100_000, "the game did not end");
        let start = host_instructions();
        let ar = a
            .run_steps_with_devices(avm, SLICE, StepOptions::default(), Some(&mut ad))
            .unwrap();
        host += host_instructions() - start;
        let br = b
            .run_steps_with_devices(bvm, SLICE, StepOptions::default(), Some(&mut bd))
            .unwrap();
        let what = format!("call {call}");
        assert_eq!(
            format!("{:?}", ar.stop()),
            format!("{:?}", br.stop()),
            "{what}"
        );
        assert_eq!(ar.steps(), br.steps(), "{what}");
        assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu), "{what}");
        assert_eq!(a.retired_instructions(), b.retired_instructions(), "{what}");
        steps += ar.steps();
        match ar.into_stop() {
            StepStop::Exit(Exit::BudgetExhausted) => {}
            StepStop::Exit(Exit::Device(DeviceStop::IntrReady)) => {
                for cpu in [&mut *a, &mut *b] {
                    cpu.inject_irq(0x20);
                    cpu.set_intr(false);
                }
            }
            StepStop::Exit(Exit::Hlt) => break,
            other => panic!("{what}: {other:?}"),
        }
        if steps >= next_tick {
            a.set_intr(true);
            b.set_intr(true);
            next_tick += tick;
        }
    }
    assert_eq!(ad.requests, bd.requests, "device transcripts differ");
    let mut am = vec![0; SIZE as usize];
    let mut bm = vec![0; SIZE as usize];
    avm.read_bytes(0, &mut am).unwrap();
    bvm.read_bytes(0, &mut bm).unwrap();
    assert!(am == bm, "RAM differs");
    host
}

/// The process's retired host instructions.
#[cfg(target_os = "macos")]
fn host_instructions() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::uninit();
    // SAFETY: V4 writes the matching structure, read only after success.
    unsafe {
        if libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            usage.as_mut_ptr().cast(),
        ) != 0
        {
            return 0;
        }
        usage.assume_init().ri_instructions
    }
}
#[cfg(not(target_os = "macos"))]
fn host_instructions() -> u64 {
    0
}

#[test]
fn a_flat_game_loop_runs_compiled_and_its_refusals_are_counted() {
    let (avm, mut a) = machine(true);
    let (bvm, mut b) = machine(false);
    a.enable_class_census_for_test();
    let host = lockstep(&avm, &mut a, &bvm, &mut b, TICK);
    println!(
        "host instructions per step {:.1}",
        host as f64 / a.retired_instructions() as f64
    );
    let frames = fixture::value(&avm, u64::from(FRAMES), 4);
    let ticks = fixture::value(&avm, u64::from(TICKS), 4);
    assert_eq!(frames, u64::from(FRAME_COUNT));
    assert!(ticks > 10, "{ticks} timer interrupts");
    let c = a.machine_jit_counters_for_test();
    print_census(&c, a.retired_instructions());
    print_classes(&a);
    assert!(c.compiled > 0, "{c:?}");
    assert!(c.compiled_retired > 0, "{c:?}");
    assert_eq!(
        c.refusals[MachineJitCounters::FIRST_ENTRY_REFUSAL..]
            .iter()
            .sum::<u64>(),
        c.refused,
        "{c:?}"
    );
}

/// Bead xj-to9.72: the production background tier-up counts the units it
/// publishes, and after publication every turn of a loop block runs
/// compiled: the compiled-retired count grows by exactly the instructions
/// retired, and the state matches an interpreted reference.
#[test]
fn background_units_are_counted_and_retire_compiled() {
    // `mov ecx, -1; L: add eax, ecx; xor ebx, eax; dec ecx; jnz L`.
    let code = asm(32, MAIN, |a| {
        let mut top = a.create_label();
        a.mov(ecx, u32::MAX).unwrap();
        a.set_label(&mut top).unwrap();
        a.add(eax, ecx).unwrap();
        a.xor(ebx, eax).unwrap();
        a.dec(ecx).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
    });
    let setup = |background| {
        let (vm, cpu) = machine(background);
        if background {
            let mut vm = vm;
            vm.set_tier_up_after(Some(2));
            vm.write_bytes(MAIN, &code).unwrap();
            return (vm, cpu);
        }
        vm.write_bytes(MAIN, &code).unwrap();
        (vm, cpu)
    };
    let (avm, mut a) = setup(true);
    let (bvm, mut b) = setup(false);
    let run = |cpu: &mut Vcpu, vm: &Vm, n: u64| {
        let r = cpu.run_steps(vm, n, StepOptions::default()).unwrap();
        assert_eq!(r.steps(), n);
    };
    // The MOV, then whole iterations until the worker's unit is published.
    run(&mut a, &avm, 1);
    run(&mut b, &bvm, 1);
    let start = std::time::Instant::now();
    while a.machine_jit_counters_for_test().compiled == 0 {
        assert!(start.elapsed().as_secs() < 30, "no background unit");
        run(&mut a, &avm, 4);
        run(&mut b, &bvm, 4);
        std::thread::yield_now();
    }
    // The next turn enters the published unit.
    let before = a.machine_jit_counters_for_test();
    run(&mut a, &avm, 4000);
    run(&mut b, &bvm, 4000);
    let after = a.machine_jit_counters_for_test();
    assert_eq!(before.compiled, 1, "{before:?}");
    assert_eq!(
        after.compiled_retired - before.compiled_retired,
        4000,
        "{after:?}"
    );
    assert_eq!(after.refused, before.refused, "{after:?}");
    assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu));
    assert_eq!(a.retired_instructions(), b.retired_instructions());
}

const ROUTINES: u64 = 0x10000;
const TABLE: u64 = 0x18000;
/// Routines the second game calls once a frame, each of three blocks.
const ROUTINE_COUNT: u32 = 512;
const ROUTINE_FRAMES: u32 = 1500;
/// Steps between timer INTR assertions in the second game.
const ROUTINE_TICK: u64 = 3_000;

/// The second game: per frame, a short tight loop, then each of
/// `ROUTINE_COUNT` small routines called once through a table (as a game's
/// entity, sound and input handlers are), a keyboard read and a short
/// palette upload; the timer interrupt every `ROUTINE_TICK` steps. Each
/// routine is three blocks, so the frame's working set is about 1,500
/// blocks, every one of them run `ROUTINE_FRAMES` times.
fn routine_pieces() -> Vec<(u64, Vec<u8>)> {
    let main = asm(32, MAIN, |a| {
        let mut frame = a.create_label();
        let mut inner = a.create_label();
        let mut calls = a.create_label();
        let mut palette = a.create_label();
        a.sti().unwrap();
        a.set_label(&mut frame).unwrap();
        a.mov(ecx, 100u32).unwrap();
        a.set_label(&mut inner).unwrap();
        a.add(eax, ecx).unwrap();
        a.dec(ecx).unwrap();
        a.jne(inner).unwrap();
        a.mov(esi, TABLE as u32).unwrap();
        a.mov(ebp, ROUTINE_COUNT).unwrap();
        a.set_label(&mut calls).unwrap();
        a.call(dword_ptr(esi)).unwrap();
        a.add(esi, 4).unwrap();
        a.dec(ebp).unwrap();
        a.jne(calls).unwrap();
        a.in_(al, 0x60).unwrap();
        a.movzx(edx, al).unwrap();
        a.add(dword_ptr(KEYS), edx).unwrap();
        a.mov(dx, 0x3c8).unwrap();
        a.xor(al, al).unwrap();
        a.out(dx, al).unwrap();
        a.inc(dx).unwrap();
        a.mov(ecx, 6u32).unwrap();
        a.set_label(&mut palette).unwrap();
        a.mov(al, cl).unwrap();
        a.out(dx, al).unwrap();
        a.dec(ecx).unwrap();
        a.jne(palette).unwrap();
        a.inc(dword_ptr(FRAMES)).unwrap();
        a.cmp(dword_ptr(FRAMES), ROUTINE_FRAMES).unwrap();
        a.jb(frame).unwrap();
        a.cli().unwrap();
        a.hlt().unwrap();
    });
    let mut pieces = vec![(MAIN, main)];
    let mut table = Vec::new();
    for i in 0..ROUTINE_COUNT {
        let at = ROUTINES + u64::from(i) * 32;
        // `add eax, i; test al, 1; jz skip; xor ebx, eax; rol ebx, 1; skip: ret`.
        pieces.push((
            at,
            asm(32, at, |a| {
                let mut skip = a.create_label();
                a.add(eax, i * 7 + 1).unwrap();
                a.test(al, 1).unwrap();
                a.je(skip).unwrap();
                a.xor(ebx, eax).unwrap();
                a.rol(ebx, 1).unwrap();
                a.set_label(&mut skip).unwrap();
                a.ret().unwrap();
            }),
        ));
        table.extend((at as u32).to_le_bytes());
    }
    pieces.push((TABLE, table));
    pieces.push((
        HANDLER,
        asm(32, HANDLER, |a| {
            a.push(eax).unwrap();
            a.inc(dword_ptr(TICKS)).unwrap();
            a.mov(al, 0x20).unwrap();
            a.out(0x20, al).unwrap();
            a.pop(eax).unwrap();
            a.iretd().unwrap();
        }),
    ));
    pieces
}

/// Bead xj-to9.78: the cold census of a working set larger than the
/// per-vCPU memo. Prints the census, the distinct cold block keys against
/// the cold turns, and how many keys reached the threshold.
#[test]
fn a_wide_working_set_reaches_the_threshold() {
    let (avm, mut a) = machine_with(routine_pieces(), true);
    let (bvm, mut b) = machine_with(routine_pieces(), false);
    a.enable_cold_census_for_test();
    let host = lockstep(&avm, &mut a, &bvm, &mut b, ROUTINE_TICK);
    let frames = fixture::value(&avm, u64::from(FRAMES), 4);
    assert_eq!(frames, u64::from(ROUTINE_FRAMES));
    let c = a.machine_jit_counters_for_test();
    let retired = a.retired_instructions();
    print_census(&c, retired);
    println!(
        "host instructions per step {:.1}",
        host as f64 / retired as f64
    );
    let census = a.cold_census_for_test().unwrap();
    let cold: Vec<_> = census.keys.values().filter(|k| k.cold_turns != 0).collect();
    let turns: u64 = cold.iter().map(|k| k.cold_turns).sum();
    let reached = census.keys.values().filter(|k| k.reached).count();
    let never = cold.iter().filter(|k| !k.reached).count();
    let slots: u64 = census.keys.values().map(|k| k.slots).sum();
    println!(
        "cold keys {} ({turns} cold turns, {} retired), keys {}, reached T1 {reached}, \
         cold keys that never did {never}",
        cold.len(),
        cold.iter().map(|k| k.cold_retired).sum::<u64>(),
        census.keys.len(),
    );
    println!(
        "memo slots {slots}: over empty {}, conflict {} ({} evicted a cold count), \
         dropped block {}, counts restored {}",
        census.slots_empty,
        census.slots_conflict,
        census.conflicts_cold,
        census.slots_dropped,
        census.victims_restored
    );
    let mut by_hot = std::collections::BTreeMap::<u32, (usize, u64)>::new();
    for k in cold.iter().filter(|k| !k.reached) {
        let bucket = match k.max_hot {
            0..=4 => 4,
            5..=9 => 9,
            10..=24 => 24,
            _ => 49,
        };
        let e = by_hot.entry(bucket).or_default();
        e.0 += 1;
        e.1 += k.cold_turns;
    }
    for (bucket, (keys, turns)) in by_hot {
        println!("  never reached, max hot <= {bucket}: {keys} keys, {turns} cold turns");
    }
    let coverage = c.compiled_retired as f64 / retired as f64;
    println!("coverage {:.1} %", 100.0 * coverage);
    let most = cold.iter().map(|k| k.cold_turns).max().unwrap_or(0);
    println!("most cold turns of one key {most}");
    // Every routine's entry block and its RET block run once a frame, so at
    // least 2 * ROUTINE_COUNT keys pass T1 = 50. Counts that memo conflicts
    // restarted would leave them cold for thousands of turns each.
    assert!(
        reached >= 2 * ROUTINE_COUNT as usize,
        "{reached} keys reached T1"
    );
    // A Busy answer from the background queue makes a block count T1 again
    // (xj-to9.82), so a loaded host can add a recount or two; restarted
    // counts left keys cold for thousands of turns.
    assert!(most < 5 * 50, "a key stayed cold for {most} turns");
}
