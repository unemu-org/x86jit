//! Signal-context requester contract (§9.2).
#![cfg(unix)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use x86jit_core::{Exit, ExitRequest, Reg, Vm, VmConfig};

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
}
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static REQUEST: AtomicPtr<ExitRequest> = AtomicPtr::new(ptr::null_mut());
static DELIVERED: AtomicUsize = AtomicUsize::new(0);

struct TrackingAllocator;
// SAFETY: allocation and deallocation are forwarded with the original layout.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.try_with(Cell::get).unwrap_or(false) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller provides a valid allocation layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller supplies the pointer and layout from the allocation.
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

extern "C" fn request_from_signal(_: libc::c_int) {
    let request = REQUEST.load(Ordering::Relaxed);
    if !request.is_null() {
        // SAFETY: the test keeps the handle alive until delivery has completed
        // and restores the previous handler before dropping it.
        unsafe { &*request }.request_exit();
        DELIVERED.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn exit_request_signal_handler_does_not_allocate() {
    let vm = Vm::new(VmConfig::flat(0x10000));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    let request = cpu.exit_request();
    REQUEST.store(ptr::from_ref(&request).cast_mut(), Ordering::Relaxed);
    // SAFETY: zero is a valid initial sigaction; the mask is initialized below.
    let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
    // SAFETY: sigaction writes the previous action before it is used.
    let mut previous: libc::sigaction = unsafe { std::mem::zeroed() };
    action.sa_sigaction = request_from_signal as *const () as usize;
    // SAFETY: the action and previous storage are valid, and this test owns SIGUSR2
    // in this dedicated integration-test process.
    unsafe {
        assert_eq!(libc::sigemptyset(&mut action.sa_mask), 0);
        assert_eq!(libc::sigaction(libc::SIGUSR2, &action, &mut previous), 0);
    }
    TRACK.with(|track| track.set(true));
    // SAFETY: pthread_self identifies the calling thread. Delivery runs the
    // installed handler while the borrowed request is still live.
    let result = unsafe { libc::pthread_kill(libc::pthread_self(), libc::SIGUSR2) };
    TRACK.with(|track| track.set(false));
    // SAFETY: previous holds the initialized action saved above.
    let restore = unsafe { libc::sigaction(libc::SIGUSR2, &previous, ptr::null_mut()) };
    REQUEST.store(ptr::null_mut(), Ordering::Relaxed);
    assert_eq!(result, 0);
    assert_eq!(restore, 0);
    assert_eq!(DELIVERED.load(Ordering::Relaxed), 1);
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert!(matches!(cpu.run(&vm, Some(0)), Exit::Interrupted));
}

static NMI: AtomicPtr<x86jit_core::NmiLine> = AtomicPtr::new(ptr::null_mut());
static NMI_DELIVERED: AtomicUsize = AtomicUsize::new(0);

extern "C" fn nmi_from_signal(_: libc::c_int) {
    let line = NMI.load(Ordering::Relaxed);
    if !line.is_null() {
        // SAFETY: the test keeps the handle alive until delivery has completed
        // and restores the previous handler before dropping it.
        unsafe { &*line }.raise();
        NMI_DELIVERED.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn nmi_line_signal_handler_does_not_allocate() {
    // spec.md §17.10: `NmiLine::raise` is one lock-free RMW, like
    // `ExitRequest::request_exit`. A Machine vCPU takes the edge at its next
    // boundary (vector 2 through the IVT; 0:1800 halts).
    let mut vm = Vm::new(VmConfig::flat(0x10000));
    vm.set_cpu_model(x86jit_core::CpuModel::I8086);
    vm.map(
        0,
        0x10000,
        x86jit_core::Prot::RWX,
        x86jit_core::RegionKind::Ram,
    )
    .unwrap();
    vm.write_bytes(0x1000, b"\x90\xf4").unwrap();
    vm.write_bytes(0x1800, b"\xf4").unwrap();
    vm.mem.write(8, 0x1800, 4).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    let line = cpu.nmi_line().unwrap();
    NMI.store(ptr::from_ref(&line).cast_mut(), Ordering::Relaxed);
    // SAFETY: zero is a valid initial sigaction; the mask is initialized below.
    let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
    // SAFETY: sigaction writes the previous action before it is used.
    let mut previous: libc::sigaction = unsafe { std::mem::zeroed() };
    action.sa_sigaction = nmi_from_signal as *const () as usize;
    // SAFETY: the action and previous storage are valid, and this test owns
    // SIGUSR1 in this dedicated integration-test process.
    unsafe {
        assert_eq!(libc::sigemptyset(&mut action.sa_mask), 0);
        assert_eq!(libc::sigaction(libc::SIGUSR1, &action, &mut previous), 0);
    }
    TRACK.with(|track| track.set(true));
    // SAFETY: pthread_self identifies the calling thread. Delivery runs the
    // installed handler while the borrowed line is still live.
    let result = unsafe { libc::pthread_kill(libc::pthread_self(), libc::SIGUSR1) };
    TRACK.with(|track| track.set(false));
    // SAFETY: previous holds the initialized action saved above.
    let restore = unsafe { libc::sigaction(libc::SIGUSR1, &previous, ptr::null_mut()) };
    NMI.store(ptr::null_mut(), Ordering::Relaxed);
    assert_eq!(result, 0);
    assert_eq!(restore, 0);
    assert_eq!(NMI_DELIVERED.load(Ordering::Relaxed), 1);
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert!(line.is_raised() && cpu.has_pending_nmi());
    assert!(matches!(cpu.run(&vm, Some(4)), Exit::Hlt));
    assert_eq!(
        (cpu.cpu.rip, vm.mem.read(0x7ffa, 2).unwrap()),
        (0x1801, 0x1000)
    );
}
