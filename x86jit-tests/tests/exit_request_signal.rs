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
