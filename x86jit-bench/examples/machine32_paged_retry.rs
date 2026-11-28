//! The paged transaction lane and its fault retries on the Machine
//! interpreter (`x86jit_tests::paged_retry`, xj-to9.83): `stack`, the
//! two-phase transaction of stack, string and frame instructions, and
//! `fault`, page faults at the first and later accesses of an instruction,
//! each handled in the guest and the instruction run again.
//! Arguments: iterations (default 200k), rounds (default 3), workload
//! (default both). Prints host instructions and heap allocations per
//! retired guest instruction.
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use x86jit_core::Exit;
use x86jit_tests::paged_retry::{setup, Workload};

struct Counting;
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
// SAFETY: forwards every call to the system allocator unchanged.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller supplies a valid allocator layout.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: forwards the original allocation pointer and layout.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwards the original allocation and requested size.
        unsafe { std::alloc::System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

#[cfg(target_os = "macos")]
fn instructions() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::uninit();
    // SAFETY: V4 writes the matching structure, read only after success.
    unsafe {
        assert_eq!(
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                usage.as_mut_ptr().cast()
            ),
            0
        );
        let count = usage.assume_init().ri_instructions;
        assert_ne!(count, 0, "host instruction counter unavailable");
        count
    }
}
#[cfg(not(target_os = "macos"))]
fn instructions() -> u64 {
    0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: u32 = args.get(1).map_or(200_000, |s| s.parse().unwrap());
    let rounds: usize = args.get(2).map_or(3, |s| s.parse().unwrap());
    let workloads = match args.get(3).map(String::as_str) {
        Some("stack") => vec![Workload::Stack],
        Some("fault") => vec![Workload::Fault],
        None => vec![Workload::Stack, Workload::Fault],
        Some(other) => panic!("unknown workload {other}"),
    };
    println!("workload,round,iterations,retired,seconds,host_per_step,allocations_per_k");
    for workload in workloads {
        for round in 0..rounds {
            let (vm, mut cpu) = setup(workload, n);
            let host_before = instructions();
            let allocations_before = ALLOCATIONS.load(Ordering::Relaxed);
            let start = Instant::now();
            let exit = cpu.run(&vm, None);
            let seconds = start.elapsed().as_secs_f64();
            let allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocations_before;
            let host = instructions() - host_before;
            assert!(matches!(exit, Exit::Hlt), "{exit:?}");
            let retired = cpu.retired_instructions();
            println!(
                "{},{round},{n},{retired},{seconds:.6},{:.1},{:.3}",
                format!("{workload:?}").to_lowercase(),
                host as f64 / retired as f64,
                allocations as f64 * 1000.0 / retired as f64
            );
        }
    }
}
