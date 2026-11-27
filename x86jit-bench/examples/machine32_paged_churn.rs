//! The paged TLB-churn loop of `x86jit_tests::paged_churn` (xj-to9.70): a
//! ring-0 kernel loop over more data pages than the TLB holds, with INVLPG,
//! CR0.TS toggles, a V86 round trip and a second data selector per iteration.
//! Arguments: iterations (default 200k), rounds (default 3). Prints host
//! instructions per retired guest instruction and, with `--features
//! test-hooks`, the relift diagnostics and lifts per thousand retired.
use std::time::Instant;
use x86jit_core::{CpuModel, Exit};
use x86jit_tests::paged_churn;

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
    println!("workload,round,iterations,retired,seconds,mips,host_per_step,lifts_per_k");
    for round in 0..rounds {
        let (vm, mut cpu) = paged_churn::setup(CpuModel::I80486Dx, n);
        #[cfg(feature = "test-hooks")]
        {
            vm.relift_diagnostics(true);
            x86jit_core::Vcpu::take_machine_counters_for_test();
        }
        let host_before = instructions();
        let start = Instant::now();
        let exit = cpu.run(&vm, None);
        let seconds = start.elapsed().as_secs_f64();
        let host = instructions() - host_before;
        assert!(matches!(exit, Exit::Hlt), "{exit:?}");
        let retired = cpu.retired_instructions();
        assert_eq!(retired, paged_churn::expected_retired(n));
        assert_eq!(cpu.cpu.gpr[5] as u32, paged_churn::expected_sum(n));
        #[allow(unused_mut, unused_assignments)]
        let mut lifts = 0;
        #[cfg(feature = "test-hooks")]
        {
            lifts = x86jit_core::Vcpu::take_machine_counters_for_test()[2];
            if round == 0 {
                eprintln!("{}", vm.relift_diagnostics(true));
            }
        }
        println!(
            "paged-churn,{round},{n},{retired},{seconds:.6},{:.3},{:.1},{:.3}",
            retired as f64 / seconds / 1e6,
            host as f64 / retired as f64,
            lifts as f64 * 1000.0 / retired as f64
        );
    }
}
