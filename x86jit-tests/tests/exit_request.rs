//! Host-requested yields at dispatch and REP iteration boundaries (§9.2).

use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use x86jit_core::{
    Backend, CpuState, Exit, ExitRequest, FaultHelper, FaultPath, HelperCaller, InterpreterBackend,
    Prot, Reg, RegionCaps, RegionKind, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;
const DATA: u64 = 0x8000;
const SIGNAL: u64 = 0x7000;
// mov r11,256; xchg qword ptr [SIGNAL],r11. SDM Vol 3A §11.1.2.1:
// memory XCHG uses automatic locking. Both backends lower it to an atomic RMW,
// matching the host's aligned 8-byte poll. SDM Vol 2D §6, XCHG, Flags Affected:
// XCHG preserves flags (witnessed by the seeded REP flag assertions below).
// REP's RawStrMem output is never read concurrently: its
// stores are non-atomic, so even a same-width atomic poll would be invalid.
const REP_SIGNAL: [u8; 15] = [
    0x49, 0xc7, 0xc3, 0x00, 0x01, 0x00, 0x00, 0x4c, 0x87, 0x1c, 0x25, 0x00, 0x70, 0x00, 0x00,
];
const TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Copy)]
enum Tier {
    Interp,
    Chained,
    Region,
}

fn backend(tier: Tier) -> Box<dyn Backend> {
    match tier {
        Tier::Interp => Box::new(InterpreterBackend),
        Tier::Chained => Box::new(JitBackend::new()),
        Tier::Region => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 8,
            max_icount: 128,
        })),
    }
}

fn vm(tier: Tier, code: &[u8], size: u64) -> Arc<Vm> {
    let mut vm = Vm::with_backend(VmConfig::flat(size), backend(tier));
    vm.map(0, size as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.write_bytes(CODE, code).unwrap();
    Arc::new(vm)
}

fn cpu(vm: &Vm) -> Vcpu {
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu
}

fn same_state(actual: &CpuState, expected: &CpuState) {
    assert_eq!(actual.rip, expected.rip, "RIP");
    assert_eq!(actual.gpr, expected.gpr, "GPRs");
    assert_eq!(
        actual.flags.to_rflags(),
        expected.flags.to_rflags(),
        "RFLAGS"
    );
}

// Hlt means the worker finished before the requesting thread was scheduled; the
// caller retries with a larger workload. A progress read makes startup timing
// irrelevant, and the timeout catches a missing poll even for infinite guests.
fn interrupt(vm: Arc<Vm>, mut cpu: Vcpu, progress: u64) -> Option<Vcpu> {
    let request = cpu.exit_request();
    let observed = Arc::clone(&vm);
    let (done_tx, done_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let exit = cpu.run(&vm, None);
        done_tx.send((exit, cpu)).unwrap();
    });
    let deadline = Instant::now() + TIMEOUT;
    loop {
        if observed.mem.read(progress, 8).unwrap() >= 256 {
            break;
        }
        if let Ok((exit, _)) = done_rx.try_recv() {
            worker.join().unwrap();
            assert!(matches!(exit, Exit::Hlt), "{exit:?}");
            return None;
        }
        assert!(Instant::now() < deadline, "guest made no progress");
        thread::yield_now();
    }
    request.request_exit();
    let (exit, cpu) = done_rx
        .recv_timeout(TIMEOUT)
        .expect("request_exit must bound an unbudgeted run");
    worker.join().unwrap();
    match exit {
        Exit::Interrupted => Some(cpu),
        Exit::Hlt => None,
        other => panic!("{other:?}"),
    }
}

// Only the last iteration is needed to reconstruct this loop's state: earlier
// iterations contribute RAX=n-1 and RBX=3*(n-1); their final CMP leaves CF set.
// Replay at most two blocks, even if the requesting thread was descheduled.
fn loop_reference(code: &[u8], seed: &CpuState, interrupted: &Vcpu) {
    let reference_vm = vm_for_reference(code, 0x10000);
    let mut reference = cpu(&reference_vm);
    reference.cpu = seed.clone();
    let prior = interrupted.reg(Reg::Rax) - 1;
    reference.set_reg(Reg::Rax, prior);
    reference.set_reg(Reg::Rbx, 3 * prior);
    reference.cpu.flags.cf = seed.flags.cf;
    let blocks = 2 - u64::from(interrupted.cpu.rip == CODE + 13);
    assert!(matches!(
        reference.run(&reference_vm, Some(blocks)),
        Exit::BudgetExhausted
    ));
    same_state(&interrupted.cpu, &reference.cpu);
}

fn loop_interrupt(tier: Tier) {
    // inc rax; mov [DATA],rax; jmp second;
    // second: add rbx,3; cmp rax,rcx; jb first; hlt
    let code = [
        0x48, 0xff, 0xc0, 0x48, 0x89, 0x04, 0x25, 0x00, 0x80, 0x00, 0x00, 0xeb, 0x00, 0x48, 0x83,
        0xc3, 0x03, 0x48, 0x39, 0xc8, 0x72, 0xea, 0xf4,
    ];
    let vm = vm(tier, &code, 0x10000);
    let mut count = 16_384;
    let mut initial = cpu(&vm);
    initial.set_reg(Reg::Rcx, count);
    initial.cpu.flags.cf = true;
    assert!(matches!(initial.run(&vm, Some(64)), Exit::BudgetExhausted));
    let (mut interrupted, seed) = loop {
        let mut initial = cpu(&vm);
        initial.set_reg(Reg::Rcx, count);
        initial.cpu.flags.cf = true;
        let seed = initial.cpu.clone();
        vm.mem.write(DATA, 0, 8).unwrap();
        if let Some(interrupted) = interrupt(Arc::clone(&vm), initial, DATA) {
            // A request may win immediately before the final HLT, after the
            // finite loop has reached its limit. Retry that terminal boundary too.
            if interrupted.reg(Reg::Rax) < count {
                break (interrupted, seed);
            }
        }
        count *= 2;
    };
    assert!(
        interrupted.reg(Reg::Rax) > 0,
        "request must arrive after execution starts"
    );
    assert!(interrupted.reg(Reg::Rax) < count);
    assert!(matches!(interrupted.cpu.rip, CODE | 0x100d));
    match tier {
        Tier::Region => assert!(vm.cache.regions() > 0, "must execute a compiled region"),
        Tier::Chained => {
            assert_eq!(vm.cache.regions(), 0);
            assert!(vm.cache.chained() > 0, "must follow compiled chain edges");
        }
        Tier::Interp => {}
    }

    loop_reference(&code, &seed, &interrupted);

    // An uninterrupted execution of the same input must end identically.
    let mut uninterrupted = cpu(&vm);
    uninterrupted.cpu = seed;
    assert!(matches!(uninterrupted.run(&vm, None), Exit::Hlt));
    assert!(matches!(interrupted.run(&vm, None), Exit::Hlt));
    same_state(&interrupted.cpu, &uninterrupted.cpu);
}

fn vm_for_reference(code: &[u8], size: u64) -> Arc<Vm> {
    vm(Tier::Interp, code, size)
}

#[test]
fn exit_request_unbudgeted_interpreter() {
    loop_interrupt(Tier::Interp);
}

#[test]
fn exit_request_unbudgeted_chained() {
    loop_interrupt(Tier::Chained);
}

#[test]
fn exit_request_unbudgeted_region() {
    loop_interrupt(Tier::Region);
}

fn infinite_loop_interrupt(tier: Tier) {
    // Two blocks with an unconditional back-edge: no guest exit or block budget
    // can rescue a missing exit-request poll. The channel timeout bounds failure.
    let code = [
        0x48, 0xff, 0xc0, 0x48, 0x89, 0x04, 0x25, 0x00, 0x80, 0x00, 0x00, 0xeb, 0x00, 0x48, 0x83,
        0xc3, 0x03, 0xeb, 0xed,
    ];
    let vm = vm(tier, &code, 0x10000);
    let mut initial = cpu(&vm);
    assert!(matches!(initial.run(&vm, Some(64)), Exit::BudgetExhausted));
    initial.cpu = CpuState::new();
    initial.cpu.rip = CODE;
    let seed = initial.cpu.clone();
    vm.mem.write(DATA, 0, 8).unwrap();
    let interrupted = interrupt(Arc::clone(&vm), initial, DATA).unwrap();
    assert!(interrupted.reg(Reg::Rax) > 0);
    loop_reference(&code, &seed, &interrupted);
    match tier {
        Tier::Region => assert!(vm.cache.regions() > 0),
        Tier::Chained => assert!(vm.cache.chained() > 0),
        Tier::Interp => {}
    }
}

#[test]
fn exit_request_infinite_interpreter() {
    infinite_loop_interrupt(Tier::Interp);
}

#[test]
fn exit_request_infinite_chained() {
    infinite_loop_interrupt(Tier::Chained);
}

#[test]
fn exit_request_infinite_region() {
    infinite_loop_interrupt(Tier::Region);
}

fn rep_interrupt(tier: Tier, movs: bool) {
    let mut count = 65_536;
    loop {
        if rep_interrupt_attempt(tier, movs, count) {
            return;
        }
        count *= 2;
    }
}

fn rep_interrupt_attempt(tier: Tier, movs: bool, count: u64) -> bool {
    let source = DATA + count;
    let opcode = if movs { 0xa4 } else { 0xaa };
    let code = if matches!(tier, Tier::Region) {
        // A statically cyclic region containing REP; RBX=0 takes the exit at
        // runtime. This exercises helper exits with carried GPRs and live fuel.
        vec![
            0xf3, opcode, 0xeb, 0x00, 0x48, 0x83, 0xfb, 0x00, 0x75, 0xf6, 0xf4,
        ]
    } else {
        vec![0xf3, opcode, 0xf4]
    };
    let code = [REP_SIGNAL.as_slice(), &code].concat();
    let vm = vm(tier, &code, source + count);
    // Warm the code/helper with one iteration, then reset the architectural input.
    let mut warm = cpu(&vm);
    warm.set_reg(Reg::Rax, 0x5a);
    warm.set_reg(Reg::Rdi, DATA);
    warm.set_reg(Reg::Rsi, source);
    warm.set_reg(Reg::Rcx, 1);
    assert!(matches!(warm.run(&vm, None), Exit::Hlt));
    vm.mem.write(SIGNAL, 0, 8).unwrap();
    if movs {
        vm.mem
            .write_bytes(source, &vec![0x5a; count as usize])
            .unwrap();
    }
    let mut initial = cpu(&vm);
    initial.set_reg(Reg::Rax, 0x5a);
    initial.set_reg(Reg::Rdi, DATA);
    initial.set_reg(Reg::Rsi, source);
    initial.set_reg(Reg::Rcx, count);
    initial.cpu.flags.cf = true;
    initial.cpu.flags.zf = true;
    let seed = initial.cpu.clone();
    let Some(mut interrupted) = interrupt(Arc::clone(&vm), initial, SIGNAL) else {
        return false;
    };
    if matches!(tier, Tier::Region) {
        assert!(vm.cache.regions() > 0);
    }
    let remaining = interrupted.reg(Reg::Rcx);
    if remaining == 0 || remaining == count {
        return false; // The request must arrive between REP iterations.
    }
    assert!(
        remaining > 0 && remaining < count,
        "must stop part-way: {remaining}"
    );
    let evidence = interrupted.interruption_evidence().unwrap();
    assert_eq!(evidence.rip(), interrupted.cpu.rip);
    assert!(!evidence.chain_link());
    match (tier, evidence.path()) {
        (
            Tier::Interp,
            FaultPath::Helper {
                family: FaultHelper::String,
                caller: HelperCaller::Interpreter,
            },
        ) => {}
        (
            Tier::Chained,
            FaultPath::Helper {
                family: FaultHelper::String,
                caller: HelperCaller::CompiledBlock { host_pc },
            },
        ) => assert!(x86jit_core::codemap::contains(host_pc)),
        (
            Tier::Region,
            FaultPath::Helper {
                family: FaultHelper::String,
                caller: HelperCaller::Region { entry_rip, host_pc },
            },
        ) => {
            assert_eq!(entry_rip, CODE);
            assert!(x86jit_core::codemap::contains(host_pc));
        }
        (_, path) => panic!("unexpected REP interruption boundary: {path:?}"),
    }
    let completed = count - remaining;
    // SDM Vol 2B, REP/REPE/REPZ/REPNE/REPNZ—Repeat String Operation
    // Prefix, Description and Operation (REP pp. 4-563–4-564): interrupted repeats save the instruction RIP, next
    // element pointers, and count after the last successful iteration. REP's
    // Flags Affected section specifies no flag changes.
    assert_eq!(interrupted.cpu.rip, CODE + REP_SIGNAL.len() as u64);
    assert_eq!(interrupted.reg(Reg::Rdi), DATA + completed);
    assert_eq!(
        interrupted.reg(Reg::Rsi),
        source + if movs { completed } else { 0 }
    );
    assert_eq!(interrupted.cpu.flags.to_rflags(), seed.flags.to_rflags());
    assert_eq!(vm.mem.read(DATA + completed - 1, 1).unwrap(), 0x5a);
    assert_eq!(vm.mem.read(DATA + completed, 1).unwrap(), 0);
    assert!(matches!(interrupted.run(&vm, None), Exit::Hlt));
    assert_eq!(interrupted.reg(Reg::Rcx), 0);
    assert_eq!(interrupted.reg(Reg::Rdi), DATA + count);
    assert_eq!(vm.mem.read(DATA + count - 1, 1).unwrap(), 0x5a);

    let reference_vm = vm_for_reference(&code, source + count);
    if movs {
        reference_vm
            .mem
            .write_bytes(source, &vec![0x5a; count as usize])
            .unwrap();
    }
    let mut reference = cpu(&reference_vm);
    reference.cpu = seed;
    assert!(matches!(reference.run(&reference_vm, None), Exit::Hlt));
    same_state(&interrupted.cpu, &reference.cpu);
    let mut actual_bytes = vec![0; count as usize];
    let mut expected_bytes = vec![0; count as usize];
    vm.mem.read_bytes(DATA, &mut actual_bytes).unwrap();
    reference_vm
        .mem
        .read_bytes(DATA, &mut expected_bytes)
        .unwrap();
    assert_eq!(actual_bytes, expected_bytes);
    true
}

#[test]
fn exit_request_rep_stos_interpreter() {
    rep_interrupt(Tier::Interp, false);
}

#[test]
fn exit_request_rep_stos_jit() {
    rep_interrupt(Tier::Chained, false);
}

#[test]
fn exit_request_rep_movs_interpreter() {
    rep_interrupt(Tier::Interp, true);
}

#[test]
fn exit_request_rep_movs_jit() {
    rep_interrupt(Tier::Chained, true);
}

#[test]
fn exit_request_rep_stos_region() {
    rep_interrupt(Tier::Region, false);
}

#[test]
fn exit_request_rep_movs_region() {
    rep_interrupt(Tier::Region, true);
}

#[test]
fn exit_request_pending_coalesces_and_is_per_vcpu() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<ExitRequest>();
    for tier in [Tier::Interp, Tier::Chained, Tier::Region] {
        let vm = vm(tier, &[0x48, 0xff, 0xc0, 0xf4], 0x10000);
        let mut target = cpu(&vm);
        let mut other = cpu(&vm);
        let request = target.exit_request();
        let cloned = request.clone();
        thread::spawn(move || {
            cloned.request_exit();
            cloned.request_exit();
        })
        .join()
        .unwrap();
        assert!(matches!(other.run(&vm, None), Exit::Hlt));
        assert!(matches!(target.run(&vm, Some(0)), Exit::Interrupted));
        assert_eq!(target.cpu.rip, CODE);
        assert_eq!(target.reg(Reg::Rax), 0);
        assert!(matches!(target.run(&vm, None), Exit::Hlt));
        assert_eq!(target.reg(Reg::Rax), 1);
        request.request_exit();
        assert!(matches!(target.run(&vm, None), Exit::Interrupted));
    }
}

fn survives_smc_drain(tier: Tier) {
    for before_write in [false, true] {
        let vm = vm(tier, &[0x48, 0xff, 0xc0, 0xf4], 0x10000);
        let mut target = cpu(&vm);
        assert!(matches!(target.run(&vm, None), Exit::Hlt));
        let request = target.exit_request();
        if before_write {
            request.request_exit();
        }
        vm.mem.write_bytes(CODE, &[0x48, 0xff, 0xc0, 0xf4]).unwrap();
        if !before_write {
            request.request_exit();
        }
        assert!(!vm.mem.take_dirty_code().is_empty());
        target.set_reg(Reg::Rip, CODE);
        assert!(matches!(target.run(&vm, None), Exit::Interrupted));
        assert_eq!(target.reg(Reg::Rax), 1);
        assert!(matches!(target.run(&vm, None), Exit::Hlt));
        assert_eq!(target.reg(Reg::Rax), 2);
    }
}

#[test]
fn exit_request_survives_smc_drain_interpreter() {
    survives_smc_drain(Tier::Interp);
}

#[test]
fn exit_request_survives_smc_drain_jit() {
    survives_smc_drain(Tier::Chained);
}

#[test]
fn exit_request_survives_smc_drain_region() {
    survives_smc_drain(Tier::Region);
}

fn mid_block_rep(tier: Tier, later_flags: bool) {
    let mut count = 65_536;
    loop {
        // add edx,1; rep stosb; [add ebx,1]; hlt. In region mode the
        // statically cyclic suffix exits at runtime because R8 is zero.
        let mut code = REP_SIGNAL.to_vec();
        code.extend_from_slice(&[0x83, 0xc2, 0x01, 0xf3, 0xaa]);
        if later_flags {
            code.extend_from_slice(&[0x83, 0xc3, 0x01]);
        }
        if matches!(tier, Tier::Region) {
            code.extend_from_slice(&[0xeb, 0x00, 0x49, 0x83, 0xf8, 0x00, 0x75]);
            code.push((-(code.len() as i8) - 1) as u8);
        }
        code.push(0xf4);
        let vm = vm(tier, &code, DATA + count);
        let mut warm = cpu(&vm);
        warm.set_reg(Reg::Rdi, DATA);
        warm.set_reg(Reg::Rcx, 1);
        assert!(matches!(warm.run(&vm, None), Exit::Hlt));
        vm.mem.write(SIGNAL, 0, 8).unwrap();
        let mut initial = cpu(&vm);
        initial.set_reg(Reg::Rax, 0x5a);
        initial.set_reg(Reg::Rdi, DATA);
        initial.set_reg(Reg::Rcx, count);
        initial.cpu.flags.cf = true;
        initial.cpu.flags.zf = true;
        let seed = initial.cpu.clone();
        let Some(mut interrupted) = interrupt(Arc::clone(&vm), initial, SIGNAL) else {
            count *= 2;
            continue;
        };
        if interrupted.reg(Reg::Rcx) == 0 || interrupted.reg(Reg::Rcx) == count {
            count *= 2;
            continue;
        }
        let mut prefix = seed.clone();
        prefix.rip += REP_SIGNAL.len() as u64;
        assert!(matches!(
            x86jit_core::interp::step_one(
                &vm.mem,
                &mut prefix,
                x86jit_core::CpuMode::Long64,
                &mut Vec::new(),
                &mut Default::default(),
            ),
            x86jit_core::StepResult::Continue
        ));
        // SDM Vol 2B, REP—Repeat String Operation (Prefix), Description and
        // Operation, pp. 4-563–4-564: restart at REP with completed prefix state.
        assert_eq!(interrupted.cpu.rip, CODE + REP_SIGNAL.len() as u64 + 3);
        assert_eq!(interrupted.reg(Reg::Rdx), 1);
        assert_eq!(interrupted.cpu.flags.to_rflags(), prefix.flags.to_rflags());
        let completed = count - interrupted.reg(Reg::Rcx);
        assert!(completed > 0 && completed < count);
        assert_eq!(interrupted.reg(Reg::Rdi), DATA + completed);
        if matches!(tier, Tier::Region) {
            assert!(vm.cache.regions() > 0);
        }
        assert!(matches!(interrupted.run(&vm, None), Exit::Hlt));
        assert_eq!(
            interrupted.reg(Reg::Rdx),
            1,
            "prefix must not execute twice"
        );
        assert_eq!(interrupted.reg(Reg::Rcx), 0);
        assert_eq!(interrupted.reg(Reg::Rdi), DATA + count);
        assert_eq!(interrupted.reg(Reg::Rbx), u64::from(later_flags));
        return;
    }
}

#[test]
fn exit_request_mid_block_rep_interpreter() {
    mid_block_rep(Tier::Interp, false);
}
#[test]
fn exit_request_mid_block_rep_jit() {
    mid_block_rep(Tier::Chained, false);
}
#[test]
fn exit_request_mid_block_rep_region() {
    mid_block_rep(Tier::Region, false);
}
#[test]
fn exit_request_rep_flag_barrier_interpreter() {
    mid_block_rep(Tier::Interp, true);
}
#[test]
fn exit_request_rep_flag_barrier_jit() {
    mid_block_rep(Tier::Chained, true);
}
#[test]
fn exit_request_rep_flag_barrier_region() {
    mid_block_rep(Tier::Region, true);
}

fn finishes_pending_mmio(tier: Tier) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(tier));
    vm.map(CODE, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(DATA, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    // mov eax,[DATA]; hlt
    vm.write_bytes(CODE, &[0x8b, 0x04, 0x25, 0x00, 0x80, 0x00, 0x00, 0xf4])
        .unwrap();
    let mut target = cpu(&vm);
    assert!(matches!(
        target.run(&vm, None),
        Exit::MmioRead {
            addr: DATA,
            size: 4,
            ..
        }
    ));
    target.complete_mmio_read(0xdead_beef);
    target.exit_request().request_exit();
    assert!(matches!(target.run(&vm, None), Exit::Interrupted));
    assert_eq!(target.reg(Reg::Rax), 0xdead_beef);
    assert_eq!(target.cpu.rip, CODE + 7);
    assert!(target.cpu.pending_mmio.is_none());
    let evidence = target.interruption_evidence().unwrap();
    assert_eq!(evidence.path(), FaultPath::Interpreter);
    assert_eq!(evidence.rip(), CODE + 7);
    assert!(!evidence.chain_link());
    assert!(matches!(target.run(&vm, None), Exit::Hlt));
}

#[test]
fn exit_request_finishes_pending_mmio_interpreter() {
    finishes_pending_mmio(Tier::Interp);
}

#[test]
fn exit_request_finishes_pending_mmio_jit() {
    finishes_pending_mmio(Tier::Chained);
}

#[test]
fn exit_request_finishes_pending_mmio_region() {
    finishes_pending_mmio(Tier::Region);
}

// Test-only execution hook: raise a request inside the compiled call, immediately
// before it returns a different exit. No scheduler race or dispatch poll can
// consume the request first. The interpreter hook requests during materialization.
thread_local! {
    static DURING_RUN: std::cell::RefCell<Option<ExitRequest>> = const { std::cell::RefCell::new(None) };
}

unsafe extern "C" fn request_then_return(cpu: *mut u8, mem: *mut u8) -> u64 {
    DURING_RUN.with(|slot| slot.borrow().as_ref().unwrap().request_exit());
    // SAFETY: the dispatcher supplies exclusive CPU and MemCtx pointers using
    // the compiled-block ABI; this function remains alive for the entire test.
    let (cpu, ctx) = unsafe {
        (
            &mut *cpu.cast::<CpuState>(),
            &mut *mem.cast::<x86jit_core::jit_abi::MemCtx>(),
        )
    };
    cpu.rip += 2;
    ctx.next_entry = request_then_return as *const () as u64;
    cpu.gpr[0]
}

struct RequestOnExitBackend {
    interpreted: bool,
}
impl Backend for RequestOnExitBackend {
    fn materialize(
        &self,
        ir: &x86jit_core::IrBlock,
        consistency: x86jit_core::MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> x86jit_core::CachedBlock {
        if self.interpreted {
            DURING_RUN.with(|slot| slot.borrow().as_ref().unwrap().request_exit());
            InterpreterBackend.materialize(ir, consistency, mmio, guest_base)
        } else {
            x86jit_core::CachedBlock::Compiled {
                entry: x86jit_core::CompiledPtr(request_then_return as *const () as *const u8),
            }
        }
    }
}

fn survives_other_exit(interpreted: bool, code: u64) {
    use x86jit_core::jit_abi::{RET_CHAIN, RET_HLT, RET_SYSCALL};
    let mut vm = Vm::with_backend(
        VmConfig::flat(0x10000),
        Box::new(RequestOnExitBackend { interpreted }),
    );
    vm.map(CODE, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(
        CODE,
        if code == RET_SYSCALL {
            &[0x0f, 0x05]
        } else {
            &[0xf4]
        },
    )
    .unwrap();
    let mut target = cpu(&vm);
    target.set_reg(Reg::Rax, code);
    DURING_RUN.with(|slot| *slot.borrow_mut() = Some(target.exit_request()));
    let exit = target.run(&vm, Some(1));
    match code {
        RET_HLT => assert!(matches!(exit, Exit::Hlt)),
        RET_SYSCALL => assert!(matches!(exit, Exit::Syscall)),
        RET_CHAIN => assert!(matches!(exit, Exit::BudgetExhausted)),
        _ => unreachable!(),
    }
    let rip = target.cpu.rip;
    assert!(matches!(target.run(&vm, Some(0)), Exit::Interrupted));
    assert_eq!(target.cpu.rip, rip);
    assert!(matches!(target.run(&vm, Some(0)), Exit::BudgetExhausted));
    DURING_RUN.with(|slot| *slot.borrow_mut() = None);
}

#[test]
fn exit_request_survives_other_exit_hlt_interpreter() {
    survives_other_exit(true, x86jit_core::jit_abi::RET_HLT);
}

#[test]
fn exit_request_survives_other_exit_hlt_jit() {
    survives_other_exit(false, x86jit_core::jit_abi::RET_HLT);
}

#[test]
fn exit_request_survives_other_exit_syscall_interpreter() {
    survives_other_exit(true, x86jit_core::jit_abi::RET_SYSCALL);
}

#[test]
fn exit_request_survives_other_exit_syscall_jit() {
    survives_other_exit(false, x86jit_core::jit_abi::RET_SYSCALL);
}

#[test]
fn exit_request_survives_other_exit_budget_jit() {
    survives_other_exit(false, x86jit_core::jit_abi::RET_CHAIN);
}

#[test]
fn exit_request_legacy_string_helpers_remain_public() {
    use x86jit_core::interp::{string_run, string_run_impl, StrFault};
    use x86jit_core::{RepKind, StrOp};

    let vm = vm_for_reference(&[0xf4], 0x10000);
    let mut state = CpuState::new();
    state.gpr[0] = 0x5a;
    state.gpr[7] = DATA;
    let result: Option<StrFault> = string_run(
        &mut state,
        &vm.mem,
        StrOp::Stos,
        1,
        RepKind::None,
        CODE,
        64,
        0,
    );
    assert!(result.is_none());
    assert_eq!(vm.mem.read(DATA, 1).unwrap(), 0x5a);

    state.gpr[7] = 0;
    let result: Option<StrFault> = string_run_impl(
        &mut state,
        &vm.mem,
        StrOp::Stos,
        1,
        RepKind::None,
        CODE,
        64,
        0,
        DATA + 1,
    );
    assert!(result.is_none());
    assert_eq!(vm.mem.read(DATA + 1, 1).unwrap(), 0x5a);

    state.gpr[7] = 0x10000;
    let fault = string_run(
        &mut state,
        &vm.mem,
        StrOp::Stos,
        1,
        RepKind::None,
        CODE,
        64,
        0,
    )
    .expect("legacy helper returns a memory fault");
    assert_eq!(fault.addr, 0x10000);
    assert!(fault.write);
    assert_eq!(state.rip, CODE);
}

fn interruption_boundary(tier: Tier, x87: bool, self_link: bool) {
    // Atomic XCHG publishes progress without racing the requester's read.
    let mut code = REP_SIGNAL.to_vec();
    if x87 {
        // SDM Vol 2A §3.2, FNOP: no FPU operation is performed.
        code.extend_from_slice(&[0xd9, 0xd0]); // fnop
    }
    if !self_link {
        code.extend_from_slice(&[0xeb, 0x00]);
    }
    let displacement = -(code.len() as i8 + 2);
    code.extend_from_slice(&[0xeb, displacement as u8]);
    let mut vm = vm(tier, &code, 0x10000);
    let mut initial = cpu(&vm);
    assert!(matches!(initial.run(&vm, Some(64)), Exit::BudgetExhausted));
    if matches!(tier, Tier::Region) {
        Arc::get_mut(&mut vm)
            .unwrap()
            .test_support_force_region(
                CODE,
                RegionCaps {
                    max_blocks: 8,
                    max_icount: 128,
                },
                initial.cpu.mxcsr,
            )
            .unwrap();
    }
    initial.cpu = CpuState::new();
    initial.cpu.rip = CODE;
    vm.mem.write(SIGNAL, 0, 8).unwrap();
    let mut interrupted = interrupt(Arc::clone(&vm), initial, SIGNAL).unwrap();
    let evidence = interrupted.interruption_evidence().unwrap();
    assert_eq!(evidence.rip(), interrupted.cpu.rip);
    assert!(
        evidence.rip() == CODE || (!self_link && evidence.rip() == CODE + code.len() as u64 - 2)
    );
    assert_eq!(evidence.chain_link(), matches!(tier, Tier::Chained));
    match (tier, evidence.path()) {
        (Tier::Interp, FaultPath::Interpreter) => {}
        (Tier::Chained, FaultPath::CompiledBlock { host_pc, site }) => {
            assert!(x86jit_core::codemap::contains(host_pc));
            assert_eq!(site, x86jit_core::codemap::lookup_trap(host_pc));
        }
        (
            Tier::Region,
            FaultPath::Region {
                entry_rip,
                host_pc,
                site,
            },
        ) => {
            assert_eq!(entry_rip, CODE);
            assert!(x86jit_core::codemap::contains(host_pc));
            assert_eq!(site, x86jit_core::codemap::lookup_trap(host_pc));
        }
        (_, path) => panic!("unexpected interruption boundary: {path:?}"),
    }
    let saved = evidence;
    assert!(matches!(
        interrupted.run(&vm, Some(0)),
        Exit::BudgetExhausted
    ));
    assert!(interrupted.interruption_evidence().is_none());
    assert_ne!(saved.path(), FaultPath::Unknown);
    assert_eq!(saved.rip(), interrupted.cpu.rip);
}

macro_rules! interruption_boundary_test {
    ($name:ident, $tier:expr, $x87:expr, $self_link:expr) => {
        #[test]
        fn $name() {
            interruption_boundary($tier, $x87, $self_link);
        }
    };
}

interruption_boundary_test!(
    interruption_path_loop_interpreter,
    Tier::Interp,
    false,
    false
);
interruption_boundary_test!(interruption_path_loop_block, Tier::Chained, false, false);
interruption_boundary_test!(interruption_path_loop_region, Tier::Region, false, false);
interruption_boundary_test!(
    interruption_path_self_link_interpreter,
    Tier::Interp,
    false,
    true
);
interruption_boundary_test!(
    interruption_path_self_link_block,
    Tier::Chained,
    false,
    true
);
interruption_boundary_test!(
    interruption_path_self_link_region,
    Tier::Region,
    false,
    true
);
interruption_boundary_test!(
    interruption_path_returned_x87_interpreter,
    Tier::Interp,
    true,
    false
);
interruption_boundary_test!(
    interruption_path_returned_x87_block,
    Tier::Chained,
    true,
    false
);
interruption_boundary_test!(
    interruption_path_returned_x87_region,
    Tier::Region,
    true,
    false
);

#[test]
fn interruption_path_before_execution_and_owned_lifetime() {
    for tier in [Tier::Interp, Tier::Chained, Tier::Region] {
        let vm = vm(tier, &[0xeb, 0xfe], 0x10000);
        let mut cpu = cpu(&vm);
        cpu.exit_request().request_exit();
        assert!(matches!(cpu.run(&vm, Some(0)), Exit::Interrupted));
        let evidence = cpu.interruption_evidence().unwrap();
        assert_eq!(evidence.rip(), CODE);
        assert_eq!(evidence.path(), FaultPath::Unknown);
        assert!(!evidence.chain_link());
        assert!(matches!(cpu.run(&vm, Some(0)), Exit::BudgetExhausted));
        assert!(cpu.interruption_evidence().is_none());
        assert_eq!(evidence.rip(), CODE);
    }
}
