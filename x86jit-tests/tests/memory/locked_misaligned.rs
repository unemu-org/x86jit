//! Misaligned locked read-modify-writes across vCPUs (xj-0w3.8). "The integrity
//! of a bus lock is not affected by the alignment of the memory field" and
//! "Locked operations are atomic with respect to all other memory operations"
//! (SDM Vol 3A §11.1.2.2 p. 11-4; Vol 2A LOCK p. 3-565; XCHG is locked without
//! the prefix, §11.1.2.1 p. 11-3). The twenty-bit models hold the bus through
//! a locked instruction (8086 Family UM §2.5 pp. 2-17–18; NEC V20/V30 UM
//! p. 12-173 BUSLOCK). An operand inside one aligned 16-byte granule loses no
//! update on any path. One that crosses the granule is a documented limitation
//! (spec.md §8.2.3): its row runs on one vCPU and asserts only the single-vCPU
//! result.
//!
//! A row is a witness only if its vCPUs ran at the same time. Every vCPU
//! waits at a guest start barrier, stores its loop counter to its progress
//! word every iteration, and reads the next vCPU's progress word just before
//! and just after its own loop. A run counts only if one of those pairs
//! differs: the other vCPU ran iterations while this one was in its loop. A
//! row that never overlaps in `ATTEMPTS` runs fails. A compiled vCPU finishes
//! N iterations within one host time slice, so it keeps running after its
//! quota until every vCPU has met its own, up to `CAP` iterations; its blocks
//! are compiled by a one-iteration warm-up first, so compile time does not
//! keep the vCPUs apart.
//!
//! Compiled rows run in a forked child with the decision-13 alignment-fault
//! handler installed, as an embedder must. Whether an in-granule misaligned
//! host atomic completes or takes an alignment fault is a host property (the
//! Arm ARM is not pinned in oracles/), so no row depends on it: a host that
//! faults replays the instruction through `Memory::locked_unaligned`, and one
//! that completes the access must still lose no update. Each compiled row
//! reports how many accesses the handler replayed.

use iced_x86::code_asm::*;
use std::sync::atomic::Ordering;
use x86jit_core::{
    Backend, BlockKey, CachedBlock, CpuMode, CpuModel, CpuState, Exit, InterpreterBackend, Prot,
    Reg, RegionCaps, RegionKind, StepOptions, StepStop, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

#[cfg(all(
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
#[path = "../../../x86jit-cranelift/src/unwind_test_signal.rs"]
mod signal;

/// vCPU `i` runs the code at `CODE + i * 0x400`.
const CODE: u64 = 0x1000;
/// The operand area: 64-byte aligned, so its 16-byte granules are the host's.
const DATA: u64 = 0x8000;
/// A byte in another granule of the operand area's page: the number of
/// fixed-count vCPUs that have finished their loop.
const FINISHED: i32 = 0x40;
/// The next byte: the number of vCPUs that have reached the start barrier.
const ARRIVED: i32 = 0x41;
/// vCPU `i`'s result at `SLOTS + SLOT * i`; above it its progress word
/// (`PROGRESS`) and its samples of the next vCPU's progress word before
/// (`BEFORE`) and after (`AFTER`) its loop.
const SLOTS: u64 = 0x9000;
const SLOT: u64 = 64;
const PROGRESS: i32 = 8;
const BEFORE: i32 = 16;
const AFTER: i32 = 24;
const N: u64 = 20_000;
/// The most iterations an open vCPU runs: a compiled CMPXCHG loop that never
/// stops would starve an interpreted one.
const CAP: u64 = 50 * N;
const VCPUS: usize = 3;
/// Runs of a multi-vCPU row before it fails for want of an overlap.
const ATTEMPTS: usize = 5;
const FILL: u8 = 0xa5;
const CAPS: RegionCaps = RegionCaps {
    max_blocks: 8,
    max_icount: 64,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Path {
    Interp64,
    /// Every vCPU runs compiled blocks.
    Jit64,
    /// vCPU 1 runs a forced compiled region; every other vCPU is interpreted
    /// (tier-up never fires). `run` checks both tiers in the cache.
    Mixed64,
    Interp32,
    /// The Application Real16 alias, always interpreted.
    Real16,
    /// A Machine model in real mode, always interpreted.
    Machine(CpuModel),
}

impl Path {
    fn bits(self) -> u32 {
        match self {
            Self::Interp32 => 32,
            Self::Real16 | Self::Machine(_) => 16,
            _ => 64,
        }
    }
    /// An unlocked XCHG on the 8086 and V20 is two bus cycles that another
    /// processor can come between (8086 Family UM §2.5 p. 2-18); LOCK makes it
    /// atomic. The 286 and later lock it without the prefix.
    fn twenty_bit(self) -> bool {
        matches!(self, Self::Machine(CpuModel::I8086 | CpuModel::NecV20))
    }
    /// Whether vCPU `i` runs compiled code.
    fn compiled(self, i: usize) -> bool {
        self == Self::Jit64 || (self == Self::Mixed64 && i == 1)
    }
    /// Whether vCPU `i` counts its iterations up and runs until the others
    /// have finished, or for `CAP` iterations: every compiled vCPU, so that it
    /// contends for the whole run however fast it is, or however slowly the
    /// host schedules the others.
    fn open(self, i: usize) -> bool {
        self.compiled(i)
    }
    /// Whether an open vCPU `i` also has a quota of N iterations, which the
    /// others wait for: on the block path. The compiled vCPU of the mixed
    /// path only follows the interpreted ones.
    fn quota(self, i: usize) -> bool {
        self == Self::Jit64 && self.open(i)
    }
    /// The forms and operand sizes the model's ISA has: the 8086, V20 and 286
    /// lock word operands only; the 386 adds doublewords and BTS/BTR; XADD and
    /// CMPXCHG arrived with the 486 and CMPXCHG8B with the Pentium.
    fn has(self, form: Form, size: u8) -> bool {
        match self {
            Self::Machine(CpuModel::I80386) => {
                size <= 4 && matches!(form, Form::Add | Form::Inc | Form::Xchg | Form::BtsBtr)
            }
            Self::Machine(_) => size == 2 && matches!(form, Form::Add | Form::Inc | Form::Xchg),
            Self::Interp64 | Self::Jit64 | Self::Mixed64 => true,
            _ => size <= 4 || form == Form::Cmpxchg8bLoop,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Form {
    Add,
    Inc,
    Xadd,
    Xchg,
    CmpxchgLoop,
    BtsBtr,
    Cmpxchg8bLoop,
}

fn base(bits: u32, offset: i32) -> AsmMemoryOperand {
    match bits {
        64 => rsi + offset,
        32 => esi + offset,
        _ => si + offset,
    }
}

fn sized(size: u8, m: AsmMemoryOperand) -> AsmMemoryOperand {
    match size {
        2 => word_ptr(m),
        4 => dword_ptr(m),
        _ => qword_ptr(m),
    }
}

/// Vcpu `i` of `n`, and how its loop ends.
#[derive(Clone, Copy)]
struct Seat {
    i: usize,
    n: usize,
    open: Option<Open>,
}

/// An open vCPU's loop: it adds itself to `FINISHED` after N iterations if it
/// has a `quota`, and runs until `FINISHED` reaches `until` or for `CAP`
/// iterations.
#[derive(Clone, Copy)]
struct Open {
    quota: bool,
    until: u8,
}

/// `[DI + disp]` at the vCPU's width: its slot area.
fn slot_word(bits: u32, disp: i32) -> AsmMemoryOperand {
    match bits {
        64 => qword_ptr(rdi + disp),
        32 => dword_ptr(edi + disp),
        _ => word_ptr(di + disp),
    }
}

/// Copy the next vCPU's progress word to this vCPU's slot word `to`, through
/// DX, which no loop keeps live across the frame.
fn sample(a: &mut CodeAssembler, bits: u32, seat: Seat, to: i32) {
    let next = (seat.i + 1) % seat.n;
    let from = (next as i32 - seat.i as i32) * SLOT as i32 + PROGRESS;
    match bits {
        64 => a.mov(rdx, slot_word(bits, from)).unwrap(),
        32 => a.mov(edx, slot_word(bits, from)).unwrap(),
        _ => a.mov(dx, slot_word(bits, from)).unwrap(),
    }
    match bits {
        64 => a.mov(slot_word(bits, to), rdx).unwrap(),
        32 => a.mov(slot_word(bits, to), edx).unwrap(),
        _ => a.mov(slot_word(bits, to), dx).unwrap(),
    }
}

/// A vCPU's loop around `body`, which may branch back to the label it is
/// given. Before the loop: the start barrier on `ARRIVED`, then a sample of
/// the next vCPU's progress word. The loop counter is BP/EBP/RBP, stored to
/// the progress word every iteration. A fixed-count vCPU counts down from N;
/// an open vCPU (64-bit only) counts up from 0 (`Open`). After the loop: the
/// second sample, then a fixed-count vCPU adds itself to `FINISHED`.
fn frame(
    a: &mut CodeAssembler,
    bits: u32,
    seat: Seat,
    body: impl FnOnce(&mut CodeAssembler, CodeLabel),
) {
    a.lock().inc(byte_ptr(base(bits, ARRIVED))).unwrap();
    let mut wait = a.create_label();
    a.set_label(&mut wait).unwrap();
    a.cmp(byte_ptr(base(bits, ARRIVED)), seat.n as i32).unwrap();
    a.jb(wait).unwrap();
    sample(a, bits, seat, BEFORE);
    let mut top = a.create_label();
    a.set_label(&mut top).unwrap();
    body(a, top);
    match (bits, seat.open) {
        (64, Some(open)) => {
            a.inc(rbp).unwrap();
            a.mov(slot_word(bits, PROGRESS), rbp).unwrap();
            if open.quota {
                let mut counted = a.create_label();
                a.cmp(rbp, N as i32).unwrap();
                a.jne(counted).unwrap();
                a.lock().inc(byte_ptr(rsi + FINISHED)).unwrap();
                a.set_label(&mut counted).unwrap();
            }
            let mut capped = a.create_label();
            a.cmp(rbp, CAP as i32).unwrap();
            a.jae(capped).unwrap();
            a.cmp(byte_ptr(rsi + FINISHED), i32::from(open.until))
                .unwrap();
            a.jb(top).unwrap();
            a.set_label(&mut capped).unwrap();
        }
        (_, Some(_)) => unreachable!("only the 64-bit mixed path has an open vCPU"),
        (64, None) => {
            a.dec(rbp).unwrap();
            a.mov(slot_word(bits, PROGRESS), rbp).unwrap();
            a.jnz(top).unwrap();
        }
        (32, None) => {
            a.dec(ebp).unwrap();
            a.mov(slot_word(bits, PROGRESS), ebp).unwrap();
            a.jnz(top).unwrap();
        }
        _ => {
            a.dec(bp).unwrap();
            a.mov(slot_word(bits, PROGRESS), bp).unwrap();
            a.jnz(top).unwrap();
        }
    }
    sample(a, bits, seat, AFTER);
    if seat.open.is_none() {
        a.lock().inc(byte_ptr(base(bits, FINISHED))).unwrap();
    }
}

fn count_error(a: &mut CodeAssembler, bits: u32) {
    match bits {
        64 => a.inc(rcx).unwrap(),
        32 => a.inc(ecx).unwrap(),
        _ => a.inc(cx).unwrap(),
    }
}

/// One vCPU's program: `form` on the operand at `DATA + offset` in its
/// `frame`, then its result to its slot (DI/EDI/RDI), then HLT. `lock_xchg`
/// adds the prefix to XCHG.
fn program(
    bits: u32,
    form: Form,
    (offset, size): (u64, u8),
    bit: u32,
    lock_xchg: bool,
    seat: Seat,
) -> Vec<u8> {
    let mut a = CodeAssembler::new(bits).unwrap();
    let offset = offset as i32;
    let m = sized(size, base(bits, offset));
    let slot = match bits {
        64 => rdi + 0,
        32 => edi + 0,
        _ => di + 0,
    };
    frame(&mut a, bits, seat, |a, top| match form {
        Form::Add => a.lock().add(m, 1).unwrap(),
        Form::Inc => a.lock().inc(m).unwrap(),
        Form::Xadd => match size {
            2 => {
                a.mov(dx, 1).unwrap();
                a.lock().xadd(m, dx).unwrap();
            }
            4 => {
                a.mov(edx, 1).unwrap();
                a.lock().xadd(m, edx).unwrap();
            }
            _ => {
                a.mov(rdx, 1u64).unwrap();
                a.lock().xadd(m, rdx).unwrap();
            }
        },
        Form::Xchg => match size {
            2 if lock_xchg => a.lock().xchg(m, ax).unwrap(),
            2 => a.xchg(m, ax).unwrap(),
            4 => a.xchg(m, eax).unwrap(),
            _ => a.xchg(m, rax).unwrap(),
        },
        Form::CmpxchgLoop => {
            // A failed compare retries from `top`, the start of the body.
            match size {
                2 => {
                    a.mov(ax, m).unwrap();
                    a.mov(dx, ax).unwrap();
                    a.inc(dx).unwrap();
                    a.lock().cmpxchg(m, dx).unwrap();
                }
                4 => {
                    a.mov(eax, m).unwrap();
                    a.mov(edx, eax).unwrap();
                    a.inc(edx).unwrap();
                    a.lock().cmpxchg(m, edx).unwrap();
                }
                _ => {
                    a.mov(rax, m).unwrap();
                    a.mov(rdx, rax).unwrap();
                    a.inc(rdx).unwrap();
                    a.lock().cmpxchg(m, rdx).unwrap();
                }
            }
            a.jnz(top).unwrap();
        }
        Form::BtsBtr => {
            // SDM Vol 2A BTS/BTR: CF receives the selected bit's prior value.
            let mut set = a.create_label();
            let mut cleared = a.create_label();
            a.lock().bts(m, bit).unwrap();
            a.jnc(set).unwrap();
            count_error(a, bits);
            a.set_label(&mut set).unwrap();
            a.lock().btr(m, bit).unwrap();
            a.jc(cleared).unwrap();
            count_error(a, bits);
            a.set_label(&mut cleared).unwrap();
        }
        Form::Cmpxchg8bLoop => {
            // SDM Vol 2A CMPXCHG8B: EDX:EAX is compared, ECX:EBX stored.
            a.mov(eax, dword_ptr(base(bits, offset))).unwrap();
            a.mov(edx, dword_ptr(base(bits, offset + 4))).unwrap();
            a.mov(ebx, eax).unwrap();
            a.mov(ecx, edx).unwrap();
            a.add(ebx, 1).unwrap();
            a.adc(ecx, 0).unwrap();
            a.lock().cmpxchg8b(m).unwrap();
            a.jnz(top).unwrap();
        }
    });
    match form {
        Form::Xchg => match size {
            2 => a.mov(word_ptr(slot), ax).unwrap(),
            4 => a.mov(dword_ptr(slot), eax).unwrap(),
            _ => a.mov(qword_ptr(slot), rax).unwrap(),
        },
        Form::BtsBtr => a.mov(word_ptr(slot), cx).unwrap(),
        _ => {}
    }
    a.hlt().unwrap();
    a.assemble(entry(seat.i)).unwrap()
}

/// Where vCPU `i`'s program goes.
fn entry(i: usize) -> u64 {
    CODE + i as u64 * 0x400
}

/// vCPU `i` of `n` on `path`.
fn seat(path: Path, i: usize, n: usize) -> Seat {
    Seat {
        i,
        n,
        open: path.open(i).then_some(Open {
            quota: path.quota(i),
            until: until(path, n),
        }),
    }
}

/// How many vCPUs add themselves to `FINISHED`: the fixed-count ones and the
/// open ones with a quota.
fn until(path: Path, n: usize) -> u8 {
    (0..n).filter(|&j| !path.open(j) || path.quota(j)).count() as u8
}

fn new_vm(path: Path) -> Vm {
    let backend: Box<dyn Backend> = match path {
        Path::Jit64 => Box::new(JitBackend::new()),
        Path::Mixed64 => Box::new(JitBackend::with_superblocks(CAPS)),
        _ => Box::new(InterpreterBackend),
    };
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    match path {
        Path::Machine(model) => vm.set_cpu_model(model),
        Path::Interp32 => vm.set_cpu_mode(CpuMode::Compat32),
        Path::Real16 => vm.set_cpu_mode(CpuMode::Real16),
        _ => vm.set_cpu_mode(CpuMode::Long64),
    }
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    if path == Path::Mixed64 {
        // Background tier-up keeps a region-forming backend from compiling on
        // first sight; the threshold is never reached.
        vm.set_tier_up_after(Some(u32::MAX));
        vm.set_tier_up_background(true);
    }
    vm
}

/// One run of a row.
struct Outcome {
    vm: Vm,
    after: [u8; 32],
    /// Iterations each vCPU completed.
    done: Vec<u64>,
    /// Some vCPU's sample of the next vCPU's progress word changed across
    /// its own loop.
    overlapped: bool,
}

/// A vCPU at `entry(i)` with the registers the programs use; `count` is the
/// loop counter's start for a fixed-count vCPU.
fn vcpu(vm: &Vm, path: Path, i: usize, count: u64, seed: u64) -> x86jit_core::Vcpu {
    let mut cpu = vm.new_vcpu();
    match path {
        Path::Real16 => {
            for seg in [Reg::Cs, Reg::Ds, Reg::Es, Reg::Ss] {
                cpu.set_reg(seg, 0);
            }
        }
        Path::Machine(CpuModel::I80286 | CpuModel::I80386) => {
            cpu.cpu = CpuState::legacy_real_entry([0; 6]);
        }
        _ => {}
    }
    cpu.set_reg(Reg::Rip, entry(i));
    cpu.set_reg(Reg::Rsi, DATA);
    cpu.set_reg(Reg::Rdi, SLOTS + SLOT * i as u64);
    cpu.set_reg(Reg::Rbp, if path.open(i) { 0 } else { count });
    cpu.set_reg(Reg::Rcx, 0);
    cpu.set_reg(Reg::Rax, seed);
    cpu.set_reg(Reg::Rsp, 0xf000);
    cpu
}

/// Run to HLT, with the decision-13 handle of a compiled vCPU registered for
/// this thread meanwhile. A Machine model runs in step budgets, which a vCPU
/// waiting at the start barrier may use up.
fn run_to_hlt(vm: &Vm, path: Path, i: usize, cpu: &mut x86jit_core::Vcpu) {
    #[cfg(all(
        any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    if path.compiled(i) {
        signal::THREAD_HANDLE.with(|h| h.set(cpu.fault_handle().as_usize()));
    }
    if let Path::Machine(_) = path {
        loop {
            let r = cpu.run_steps(vm, 1 << 20, StepOptions::default()).unwrap();
            match r.stop() {
                StepStop::Exit(Exit::Hlt) => break,
                StepStop::Exit(Exit::BudgetExhausted) => {}
                other => panic!("{path:?} vCPU {i}: {other:?}"),
            }
        }
    } else {
        let exit = cpu.run(vm, None);
        assert!(matches!(exit, Exit::Hlt), "{path:?} vCPU {i}: {exit:?}");
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    signal::THREAD_HANDLE.with(|h| h.set(0));
}

/// Write each vCPU's program and the operand area, then compile on the
/// compiled paths: force vCPU 1's region on the mixed path; on the block path
/// run every program once, one iteration on this thread, and write the data
/// again. Then run every vCPU on its own host thread to HLT. vCPU `i` starts
/// with `seeds[i]` in RAX.
fn run(path: Path, programs: &[Vec<u8>], image: &[u8; 32], seeds: &[u64]) -> Outcome {
    let n = programs.len();
    let mut vm = new_vm(path);
    for (i, code) in programs.iter().enumerate() {
        vm.write_bytes(entry(i), code).unwrap();
    }
    let data = |vm: &Vm, finished: u8, arrived: u8| {
        vm.write_bytes(DATA, image).unwrap();
        vm.write_bytes(DATA + FINISHED as u64, &[finished, arrived])
            .unwrap();
        vm.write_bytes(SLOTS, &vec![0; n * SLOT as usize]).unwrap();
    };
    let compiled: Vec<bool> = (0..n).map(|i| path.compiled(i)).collect();
    if path == Path::Mixed64 && n > 1 {
        // One call: forcing a region discards every earlier translation.
        vm.test_support_force_region(entry(1), CAPS, 0x1f80)
            .unwrap();
    }
    if path == Path::Jit64 {
        for (i, &seed) in seeds.iter().enumerate() {
            // Every other vCPU counts as arrived at the barrier, and every
            // quota as met, so the loop runs once.
            data(&vm, until(path, n), n as u8 - 1);
            run_to_hlt(&vm, path, i, &mut vcpu(&vm, path, i, 1, seed));
        }
    }
    data(&vm, 0, 0);
    let vm_ref = &vm;
    std::thread::scope(|s| {
        for (i, &seed) in seeds.iter().enumerate() {
            s.spawn(move || run_to_hlt(vm_ref, path, i, &mut vcpu(vm_ref, path, i, N, seed)));
        }
    });
    if path == Path::Mixed64 {
        // The tiers: vCPU 1's entry is the forced region, every other vCPU's
        // entry an interpreted block, and nothing else was compiled.
        for (i, &compiled) in compiled.iter().enumerate() {
            let entry = vm.cache.get(BlockKey::new(entry(i), CpuMode::Long64));
            assert_eq!(
                matches!(entry, Some(CachedBlock::Compiled { .. })),
                compiled,
                "vCPU {i}"
            );
            assert_eq!(
                matches!(entry, Some(CachedBlock::Interpreted(_))),
                !compiled,
                "vCPU {i}"
            );
        }
        let cache = vm.memory_stats().cache;
        let regions = compiled.iter().filter(|&&c| c).count() as u64;
        assert_eq!((cache.region_entries, cache.compiled_entries), (regions, 0));
    }
    let width = (path.bits() / 8) as u8;
    let word = |i: usize, at: i32| {
        vm.mem
            .read(SLOTS + SLOT * i as u64 + at as u64, width)
            .unwrap()
    };
    let overlapped = (0..n).any(|i| word(i, BEFORE) != word(i, AFTER));
    let done = (0..n)
        .map(|i| if path.open(i) { word(i, PROGRESS) } else { N })
        .collect();
    let mut after = [0u8; 32];
    vm.read_bytes(DATA, &mut after).unwrap();
    Outcome {
        vm,
        after,
        done,
        overlapped,
    }
}

/// Run `attempt` until one run's vCPUs overlapped; `attempt` asserts each
/// run's result. Returns a line for the log.
fn until_overlap(tag: &str, vcpus: usize, mut attempt: impl FnMut() -> Outcome) -> String {
    let mut iterations = 0;
    for tries in 1..=ATTEMPTS {
        let out = attempt();
        iterations += out.done.iter().sum::<u64>();
        if vcpus == 1 || out.overlapped {
            return format!("{tries} run(s), {iterations} iterations");
        }
        eprintln!("{tag}: run {tries} did not overlap");
    }
    panic!("{tag}: no run of {ATTEMPTS} had its vCPUs overlap; the row proves nothing here");
}

/// Run `body` in-process, or, when a vCPU is compiled, in a forked child with
/// the decision-13 handler installed (`fault_landing_pad`'s pattern), and log
/// its line with the number of alignment faults the handler replayed.
fn isolate(tag: &str, compiled: bool, body: impl FnOnce() -> String) {
    if !compiled {
        let line = body();
        eprintln!("{tag}: {line}");
        return;
    }
    #[cfg(all(
        any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    {
        let report = Report::new();
        let status = signal::child(|| {
            // SAFETY: the child owns its signal disposition and its timer.
            unsafe {
                libc::alarm(150);
                signal::install();
            }
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
                Ok(line) => report.put(&format!(
                    "{line}, {} alignment fault(s) replayed",
                    signal::ALIGNMENTS.load(Ordering::Relaxed)
                )),
                Err(payload) => {
                    let message = payload
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| payload.downcast_ref::<&str>().copied())
                        .unwrap_or("non-string panic");
                    report.put(message);
                    std::panic::resume_unwind(payload);
                }
            }
        });
        let line = report.get();
        assert_eq!(status, 0, "{tag}: child status {status}: {line}");
        eprintln!("{tag}: {line}");
    }
    #[cfg(not(all(
        any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
        any(target_arch = "aarch64", target_arch = "x86_64")
    )))]
    {
        let _ = body;
        eprintln!("{tag}: skipped, no decision-13 handler on this host");
    }
}

/// A page shared with the forked child, for its one line of report.
#[cfg(all(
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
struct Report(*mut u8);

#[cfg(all(
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
impl Report {
    const LEN: usize = 4096;
    fn new() -> Self {
        // SAFETY: a fresh anonymous shared mapping, owned by this value.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                Self::LEN,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED);
        Self(ptr.cast())
    }
    fn put(&self, line: &str) {
        let bytes = &line.as_bytes()[..line.len().min(Self::LEN - 2)];
        // SAFETY: inside the mapping; only the child writes, before it exits.
        unsafe {
            self.0
                .cast::<u16>()
                .write(u16::try_from(bytes.len()).unwrap());
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.0.add(2), bytes.len());
        }
    }
    fn get(&self) -> String {
        // SAFETY: inside the mapping; the child has exited.
        unsafe {
            let len = usize::from(self.0.cast::<u16>().read());
            String::from_utf8_lossy(std::slice::from_raw_parts(self.0.add(2), len)).into_owned()
        }
    }
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
impl Drop for Report {
    fn drop(&mut self) {
        // SAFETY: the mapping made in `new`.
        unsafe { libc::munmap(self.0.cast(), Self::LEN) };
    }
}

fn mask(size: u8) -> u64 {
    if size == 8 {
        u64::MAX
    } else {
        (1 << (8 * size)) - 1
    }
}

fn field(bytes: &[u8; 32], offset: u64, size: u8) -> u64 {
    let mut v = [0u8; 8];
    v[..size as usize].copy_from_slice(&bytes[offset as usize..][..size as usize]);
    u64::from_le_bytes(v)
}

/// A distinct exchange token per vCPU, and one more for the initial operand.
fn token(i: usize, size: u8) -> u64 {
    0x0102_0304_0506_0708u64.wrapping_mul(i as u64 + 1) & mask(size)
}

/// Bits that land in different bytes of the operand, one per vCPU.
fn own_bit(size: u8, i: usize) -> u32 {
    let bits: [u32; 4] = match size {
        2 => [1, 9, 14, 4],
        4 => [2, 13, 30, 21],
        _ => [5, 33, 62, 20],
    };
    bits[i]
}

/// Whether an operand crosses an aligned 16-byte granule.
fn crossing(offset: u64, size: u8) -> bool {
    (offset & 15) + u64::from(size) > 16
}

/// One row: `form` at `DATA + offset` on `vcpus` vCPUs. The other bytes of the
/// operand area hold `FILL` throughout and must still hold it.
fn row(path: Path, form: Form, offset: u64, size: u8, vcpus: usize) {
    let tag = format!("{path:?} {form:?} size {size} at +{offset}, {vcpus} vCPU(s)");
    if !path.has(form, size) {
        return;
    }
    let compiled = (0..vcpus).any(|i| path.compiled(i));
    // On x86-64 a compiled crossing operand can span a host cache line (the
    // owned backing is only 16-byte aligned): a split lock, which the host
    // may slow down or turn into #AC (SDM Vol 3A §11.1.2.3 p. 11-4). That row
    // runs single-vCPU in `fault_landing_pad` instead.
    if compiled && crossing(offset, size) && cfg!(target_arch = "x86_64") {
        eprintln!("{tag}: skipped, a split lock on this host");
        return;
    }
    let bits = path.bits();
    let programs: Vec<_> = (0..vcpus)
        .map(|i| {
            program(
                bits,
                form,
                (offset, size),
                own_bit(size, i),
                path.twenty_bit(),
                seat(path, i, vcpus),
            )
        })
        .collect();
    let mut image = [FILL; 32];
    let initial = if form == Form::Xchg {
        token(vcpus, size)
    } else {
        0
    };
    image[offset as usize..][..size as usize]
        .copy_from_slice(&initial.to_le_bytes()[..size as usize]);
    let seeds: Vec<_> = (0..vcpus).map(|i| token(i, size)).collect();
    isolate(&tag, compiled, || {
        until_overlap(&tag, vcpus, || {
            let out = run(path, &programs, &image, &seeds);
            check_row(&tag, &out, form, (offset, size), vcpus);
            out
        })
    });
}

fn check_row(tag: &str, out: &Outcome, form: Form, (offset, size): (u64, u8), vcpus: usize) {
    for (i, &byte) in out.after.iter().enumerate() {
        let i = i as u64;
        if i < offset || i >= offset + u64::from(size) {
            assert_eq!(byte, FILL, "{tag}: neighbour byte +{i}");
        }
    }
    let value = field(&out.after, offset, size);
    let slot = |i: usize| out.vm.mem.read(SLOTS + SLOT * i as u64, size).unwrap();
    match form {
        Form::Xchg => {
            // A torn exchange duplicates or drops a token.
            let mut before: Vec<_> = (0..=vcpus).map(|i| token(i, size)).collect();
            let mut now: Vec<_> = (0..vcpus).map(slot).collect();
            now.push(value);
            before.sort_unstable();
            now.sort_unstable();
            assert_eq!(now, before, "{tag}: tokens");
        }
        Form::BtsBtr => {
            assert_eq!(value, 0, "{tag}: operand");
            for i in 0..vcpus {
                assert_eq!(
                    slot(i) & 0xffff,
                    0,
                    "{tag}: vCPU {i} saw a foreign bit state"
                );
            }
        }
        _ => {
            let total: u64 = out.done.iter().sum();
            assert_eq!(
                value,
                total & mask(size),
                "{tag}: lost {} update(s)",
                total.wrapping_sub(value) & mask(size)
            );
        }
    }
}

/// The placements: an aligned control, an operand inside the containing 8-byte
/// word (word +1), three inside the 16-byte granule only (word +7, dword +6,
/// qword +4), and the crossing word +15 on one vCPU.
/// On the mixed path the one vCPU of a crossing row is the interpreted one.
fn rows(path: Path, form: Form) {
    let vcpus = if path == Path::Mixed64 { 2 } else { VCPUS };
    if form == Form::Cmpxchg8bLoop {
        for offset in [0, 4] {
            row(path, form, offset, 8, vcpus);
        }
        row(path, form, 12, 8, 1);
        return;
    }
    for size in [2u8, 4, 8] {
        row(path, form, 0, size, vcpus);
    }
    row(path, form, 1, 2, vcpus);
    row(path, form, 7, 2, vcpus);
    row(path, form, 6, 4, vcpus);
    row(path, form, 4, 8, vcpus);
    row(path, form, 15, 2, 1);
}

/// vCPU A increments the misaligned word at `a`; vCPU B adds 1 to the byte at
/// `a` with a locked add to its `b_size` operand at `b`; vCPU C stores its
/// loop counter's low byte to the neighbour `c` of the containing word with a
/// plain MOV. Every update adds exactly 1 to byte `a`, whatever the order and
/// the carries, so that byte is the update count mod 256, and C's last store
/// survives every swap of the containing word. When B's operand also covers
/// byte `a + 1` and the count stays below 2^16, no carry leaves the word and
/// the whole word is the count; otherwise the bytes above `a` depend on the
/// order of the carries and are not checked.
fn overlap(path: Path, a: u64, b: u64, b_size: u8, c: u64) {
    let tag = format!("{path:?} overlap word +{a}, size {b_size} at +{b}, byte +{c}");
    if !path.has(Form::Add, b_size) {
        return;
    }
    let bits = path.bits();
    let assemble = |i: usize, body: &dyn Fn(&mut CodeAssembler)| {
        let mut asm = CodeAssembler::new(bits).unwrap();
        frame(&mut asm, bits, seat(path, i, 3), |asm, _| body(asm));
        asm.hlt().unwrap();
        asm.assemble(entry(i)).unwrap()
    };
    let programs = [
        assemble(0, &|asm| {
            asm.lock().inc(word_ptr(base(bits, a as i32))).unwrap()
        }),
        assemble(1, &|asm| {
            let addend = 1 << (8 * (a - b));
            asm.lock()
                .add(sized(b_size, base(bits, b as i32)), addend)
                .unwrap()
        }),
        assemble(2, &|asm| {
            asm.mov(ax, bp).unwrap();
            asm.mov(byte_ptr(base(bits, c as i32)), al).unwrap();
        }),
    ];
    let mut image = [FILL; 32];
    image[b as usize..][..b_size as usize].fill(0);
    image[a as usize..][..2].fill(0);
    let compiled = (0..3).any(|i| path.compiled(i));
    isolate(&tag, compiled, || {
        until_overlap(&tag, 3, || {
            let out = run(path, &programs, &image, &[0; 3]);
            let total = out.done[0] + out.done[1];
            let mut expected = image;
            // C stores its counter before stepping it: 1 last when it counts
            // down, its last iteration's index when it counts up.
            expected[c as usize] = if path.open(2) {
                (out.done[2] - 1) as u8
            } else {
                1
            };
            let mut after = out.after;
            if b + u64::from(b_size) >= a + 2 && total < 0x1_0000 {
                expected[a as usize..][..2].copy_from_slice(&(total as u16).to_le_bytes());
            } else {
                expected[a as usize] = total as u8;
                let end = (a + 2).max(b + u64::from(b_size)) as usize;
                after[a as usize + 1..end].copy_from_slice(&expected[a as usize + 1..end]);
            }
            assert_eq!(after, expected, "{tag}: {total} updates");
            out
        })
    });
}

macro_rules! witnesses {
    ($($name:ident: $path:expr;)*) => {
        $(
            mod $name {
                use super::*;
                #[test]
                fn lock_add() {
                    rows($path, Form::Add);
                }
                #[test]
                fn lock_inc() {
                    rows($path, Form::Inc);
                }
                #[test]
                fn lock_xadd() {
                    rows($path, Form::Xadd);
                }
                #[test]
                fn xchg() {
                    rows($path, Form::Xchg);
                }
                #[test]
                fn lock_cmpxchg_loop() {
                    rows($path, Form::CmpxchgLoop);
                }
                #[test]
                fn lock_bts_btr() {
                    rows($path, Form::BtsBtr);
                }
                #[test]
                fn lock_cmpxchg8b_loop() {
                    rows($path, Form::Cmpxchg8bLoop);
                }
                #[test]
                fn overlapping_locked_operands_and_a_plain_byte() {
                    // Inside the containing 8-byte word, then the 16-byte
                    // granule. An aligned word partner shares only byte a, so
                    // A's swap of the containing word races a host atomic of
                    // another size and place.
                    overlap($path, 1, 0, 2, 5);
                    overlap($path, 7, 6, 2, 12);
                    if $path.has(Form::Add, 4) {
                        // A doubleword partner: aligned and covering the word;
                        // misaligned and covering it; aligned and sharing only
                        // byte a.
                        overlap($path, 1, 0, 4, 5);
                        overlap($path, 7, 6, 4, 12);
                        overlap($path, 7, 4, 4, 12);
                    } else {
                        // The same misaligned word on both vCPUs.
                        overlap($path, 1, 1, 2, 5);
                        overlap($path, 7, 7, 2, 12);
                    }
                }
            }
        )*
    };
}

witnesses! {
    long64_interpreter: Path::Interp64;
    long64_jit_block: Path::Jit64;
    long64_mixed_tiers: Path::Mixed64;
    compat32_interpreter: Path::Interp32;
    real16_alias: Path::Real16;
    machine_i8086: Path::Machine(CpuModel::I8086);
    machine_nec_v20: Path::Machine(CpuModel::NecV20);
    machine_i80286: Path::Machine(CpuModel::I80286);
    machine_i80386: Path::Machine(CpuModel::I80386);
}
