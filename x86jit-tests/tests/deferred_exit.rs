//! Host-request deferral through engine work (§9.2).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use x86jit_core::{
    Backend, CachedBlock, CriticalSectionKind, Exit, FetchVerdict, InterpreterBackend, IrBlock,
    MemConsistency, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const ENTRY: u64 = 0x1000;
const TIMEOUT: Duration = Duration::from_secs(5);

fn fixture(backend: Box<dyn Backend>) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(ENTRY, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    // Two unconditional edges form a region without a terminal guest exit.
    vm.write_bytes(ENTRY, &[0xeb, 0x00, 0xeb, 0xfc]).unwrap();
    vm
}

fn held_compile(backend: Box<dyn Backend>, tier_up: bool, region: bool) {
    let mut vm = fixture(backend);
    if tier_up {
        vm.set_tier_up_after(Some(1));
    }
    let held = vm.test_support_hold_compile(Duration::from_millis(250));
    let vm = Arc::new(vm);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, ENTRY);
    assert!(cpu.interruption_evidence().is_none());
    let request = cpu.exit_request();
    let run_vm = vm.clone();
    let (tx, rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let exit = cpu.run(&run_vm, None);
        tx.send((exit, cpu)).unwrap();
    });
    let deadline = Instant::now() + TIMEOUT;
    while !held.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "compile hold was never entered");
        thread::yield_now();
    }
    // Repeated requests coalesce. No returned exit can precede completion of the
    // bounded hold and backend callback.
    for _ in 0..32 {
        request.request_exit();
    }
    let (exit, mut cpu) = rx
        .recv_timeout(TIMEOUT)
        .expect("deferred yield must complete");
    worker.join().unwrap();
    assert!(!held.load(Ordering::Acquire));
    assert!(matches!(exit, Exit::Interrupted), "{exit:?}");
    let evidence = cpu.interruption_evidence().unwrap();
    assert!(evidence.deferred(), "{exit:?}");
    assert!(evidence.contains(CriticalSectionKind::Compile), "{exit:?}");
    assert!(evidence.contains(CriticalSectionKind::Publish), "{exit:?}");
    assert!(evidence.deferral_count() >= 1);
    if region {
        assert!(vm.cache.regions() > 0, "must materialize a compiled region");
    }
    // These acquire cache locks after run returns; no engine lock may escape.
    vm.cache.invalidate_overlapping(ENTRY, 4, || {});
    request.request_exit();
    let immediate = cpu.run(&vm, Some(0));
    assert!(matches!(immediate, Exit::Interrupted));
    assert!(!cpu.interruption_evidence().unwrap().deferred());
    assert!(matches!(cpu.run(&vm, Some(0)), Exit::BudgetExhausted));
    assert!(cpu.interruption_evidence().is_none());
    assert!(
        evidence.deferred(),
        "owned evidence survives resume and invalidation"
    );
}

#[test]
fn deferred_exit_compile_interpreter() {
    held_compile(Box::new(InterpreterBackend), false, false);
}

#[test]
fn deferred_exit_compile_jit() {
    held_compile(Box::new(JitBackend::new()), false, false);
}

#[test]
fn deferred_exit_compile_region() {
    held_compile(
        Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 8,
            max_icount: 128,
        })),
        false,
        true,
    );
}

#[test]
fn deferred_exit_inline_tier_up() {
    held_compile(Box::new(JitBackend::new()), true, false);
}

#[test]
fn deferred_exit_lift_and_region_build() {
    for tier in 0..3 {
        let backend: Box<dyn Backend> = match tier {
            0 => Box::new(InterpreterBackend),
            1 => Box::new(JitBackend::new()),
            _ => Box::new(JitBackend::with_superblocks(RegionCaps {
                max_blocks: 8,
                max_icount: 128,
            })),
        };
        let mut vm = fixture(backend);
        let request_slot = Arc::new(Mutex::new(None::<x86jit_core::ExitRequest>));
        let slot = request_slot.clone();
        vm.set_fetch_policy(Box::new(move |_| {
            slot.lock().unwrap().as_ref().unwrap().request_exit();
            FetchVerdict::Allow
        }));
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, ENTRY);
        *request_slot.lock().unwrap() = Some(cpu.exit_request());
        let exit = cpu.run(&vm, None);
        assert!(matches!(exit, Exit::Interrupted), "{exit:?}");
        let evidence = cpu.interruption_evidence().expect("requested yield");
        let kind = if tier == 2 {
            CriticalSectionKind::RegionBuild
        } else {
            CriticalSectionKind::Lift
        };
        assert!(evidence.contains(kind), "{exit:?}");
        assert!(evidence.deferred());
        assert!(
            request_slot.try_lock().is_ok(),
            "callback lock released before yielding"
        );
    }
}

struct LockedBackend {
    lock: Arc<Mutex<()>>,
    entered: Arc<AtomicBool>,
    release: mpsc::Receiver<()>,
}

// Receiver is protected by the backend's outer mutex below.
struct BlockingBackend(Mutex<LockedBackend>);
impl Backend for BlockingBackend {
    fn materialize(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> CachedBlock {
        let backend = self.0.lock().unwrap();
        let _locked = backend.lock.lock().unwrap();
        backend.entered.store(true, Ordering::Release);
        backend
            .release
            .recv_timeout(TIMEOUT)
            .expect("release compile fixture");
        InterpreterBackend.materialize(ir, consistency, mmio, guest_base)
    }
}

#[test]
fn deferred_exit_never_returns_while_backend_lock_is_held() {
    let lock = Arc::new(Mutex::new(()));
    let entered = Arc::new(AtomicBool::new(false));
    let (release_tx, release) = mpsc::channel();
    let vm = fixture(Box::new(BlockingBackend(Mutex::new(LockedBackend {
        lock: lock.clone(),
        entered: entered.clone(),
        release,
    }))));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, ENTRY);
    let request = cpu.exit_request();
    let (done_tx, done_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let exit = cpu.run(&vm, None);
        done_tx.send((exit, cpu.interruption_evidence())).unwrap();
    });
    let deadline = Instant::now() + TIMEOUT;
    while !entered.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline);
        thread::yield_now();
    }
    assert!(lock.try_lock().is_err());
    request.request_exit();
    assert!(matches!(done_rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
    release_tx.send(()).unwrap();
    let (exit, evidence) = done_rx.recv_timeout(TIMEOUT).unwrap();
    worker.join().unwrap();
    assert!(lock.try_lock().is_ok());
    assert!(matches!(exit, Exit::Interrupted), "{exit:?}");
    assert!(evidence.unwrap().contains(CriticalSectionKind::Compile));
}

#[test]
fn deferred_exit_survives_another_exit_without_stale_metadata() {
    let mut vm = fixture(Box::new(InterpreterBackend));
    vm.write_bytes(ENTRY, &[0xf4]).unwrap();
    let slot = Arc::new(Mutex::new(None::<x86jit_core::ExitRequest>));
    vm.set_fetch_policy(Box::new({
        let slot = slot.clone();
        move |_| {
            slot.lock().unwrap().as_ref().unwrap().request_exit();
            FetchVerdict::Allow
        }
    }));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, ENTRY);
    *slot.lock().unwrap() = Some(cpu.exit_request());
    let hlt = cpu.run(&vm, None);
    assert!(matches!(hlt, Exit::Hlt));
    assert!(cpu.interruption_evidence().is_none());
    let interrupted = cpu.run(&vm, Some(0));
    assert!(matches!(interrupted, Exit::Interrupted));
    let evidence = cpu.interruption_evidence().unwrap();
    assert!(evidence.contains(CriticalSectionKind::Lift));
    assert!(matches!(cpu.run(&vm, Some(0)), Exit::BudgetExhausted));
    assert!(cpu.interruption_evidence().is_none());
    slot.lock().unwrap().as_ref().unwrap().request_exit();
    let immediate = cpu.run(&vm, Some(0));
    assert_eq!(cpu.interruption_evidence().unwrap().deferral_count(), 0);
    assert!(matches!(immediate, Exit::Interrupted));
    assert!(evidence.deferred());
}

struct SubmitBackend(Arc<Mutex<Option<x86jit_core::ExitRequest>>>);

impl Backend for SubmitBackend {
    fn materialize(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> CachedBlock {
        InterpreterBackend.materialize(ir, consistency, mmio, guest_base)
    }

    fn tier_up_async(&self, _: x86jit_core::TierUpRequest) -> x86jit_core::TierUpSubmit {
        self.0.lock().unwrap().as_ref().unwrap().request_exit();
        x86jit_core::TierUpSubmit::Busy
    }
}

#[test]
fn deferred_exit_background_tier_up_submission() {
    let slot = Arc::new(Mutex::new(None));
    let mut vm = fixture(Box::new(SubmitBackend(slot.clone())));
    vm.set_tier_up_after(Some(1));
    vm.set_tier_up_background(true);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, ENTRY);
    *slot.lock().unwrap() = Some(cpu.exit_request());
    let exit = cpu.run(&vm, None);
    assert!(matches!(exit, Exit::Interrupted), "{exit:?}");
    let evidence = cpu.interruption_evidence().expect("tier-up yield");
    assert!(
        evidence.contains(CriticalSectionKind::TierUpWait),
        "{exit:?}"
    );
    assert_eq!(evidence.deferral_count(), 1);
    assert!(slot.try_lock().is_ok());
}

#[test]
fn deferred_exit_evidence_clears_on_unwinding_run_entry() {
    let mut vm = fixture(Box::new(InterpreterBackend));
    vm.set_fetch_policy(Box::new(|_| panic!("fetch fixture")));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, ENTRY);
    cpu.exit_request().request_exit();
    assert!(matches!(cpu.run(&vm, None), Exit::Interrupted));
    let evidence = cpu.interruption_evidence().unwrap();
    assert!(!evidence.deferred());
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cpu.run(&vm, None)));
    assert!(unwind.is_err());
    assert!(cpu.interruption_evidence().is_none());
    assert!(
        !evidence.deferred(),
        "the accessor returns an independent copy"
    );
}
