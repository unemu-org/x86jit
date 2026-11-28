//! The paged transaction lane allocates nothing per instruction once warm
//! (xj-to9.83): neither the two-phase transaction of stack, string and frame
//! instructions, nor a page fault at an instruction's first or a later
//! access, its delivery, the handler's IRETD and the instruction's second
//! run (`x86jit_tests::paged_retry`). The one allocation of a fault is its
//! exit's boxed metadata (`Exit::with_fault_metadata`), which the delivery
//! reads.
use std::cell::Cell;
use x86jit_core::Exit;
use x86jit_tests::paged_retry::{restart, setup, Workload};

thread_local! {
    // `None` while not counting; this thread's allocations while counting.
    static ALLOCATIONS: Cell<Option<u64>> = const { Cell::new(None) };
}

struct Counter;
fn count() {
    let _ = ALLOCATIONS.try_with(|n| {
        if let Some(count) = n.get() {
            n.set(Some(count + 1));
        }
    });
}
// SAFETY: forwards every call to the system allocator unchanged.
unsafe impl std::alloc::GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        count();
        // SAFETY: the caller supplies a valid allocator layout.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: forwards the original allocation pointer and layout.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        count();
        // SAFETY: forwards the original allocation and requested size.
        unsafe { std::alloc::System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;

/// Allocations of a warm run of `iterations`.
fn warm_run(workload: Workload, iterations: u32) -> u64 {
    let (vm, mut cpu) = setup(workload, 8);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    restart(&mut cpu, workload, iterations);
    ALLOCATIONS.with(|n| n.set(Some(0)));
    let exit = cpu.run(&vm, None);
    let count = ALLOCATIONS.with(|n| n.replace(None).unwrap());
    assert!(matches!(exit, Exit::Hlt), "{exit:?}");
    count
}

#[test]
fn warm_paged_transactions_allocate_nothing() {
    assert_eq!(warm_run(Workload::Stack, 200), 0);
}

#[test]
fn warm_paged_fault_retries_allocate_only_the_fault_exit() {
    assert_eq!(
        warm_run(Workload::Fault, 200),
        200 * u64::from(x86jit_tests::paged_retry::FAULTS)
    );
}
