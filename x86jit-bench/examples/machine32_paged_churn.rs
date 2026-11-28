//! The paged TLB-churn loop of `x86jit_tests::paged_churn` (xj-to9.70): a
//! ring-0 kernel loop over more data pages than the TLB holds, with INVLPG,
//! CR0.TS toggles, a V86 round trip and a second data selector per iteration.
//! Arguments: iterations (default 200k), rounds (default 3), and `rotate` or
//! `rotate128` for the variants that reload DS, ES, FS and GS from a table of
//! 32 or 128 flat and non-flat selector sets each iteration (xj-to9.75; the
//! 128 sets outrun the 64 retained paged identities), `wide:B:K` (B blocks
//! under K CR3 values), `stores:B:S` and `datastores:B:S` (B blocks, a store
//! every S to a data word beside the code or on a page without code,
//! xj-to9.85) and `contexts:B:N` (B resident blocks, then N Segmented
//! stamps in turn, more than the retained and retired ones). Prints host
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

// A variant named by the third argument.
#[derive(Clone, Copy)]
enum Variant {
    Churn(u32),
    // `wide:B:K`: B blocks under K CR3 values.
    Wide(u32, u32),
    // `stores:B:S` / `datastores:B:S`: B blocks, a store every S.
    Stores(u32, u32, paged_churn::Stores),
    // `contexts:B:N`: B resident blocks, N Segmented stamps in turn.
    Contexts(u32, u32),
}

fn variant(arg: Option<&str>) -> Variant {
    let Some(arg) = arg else {
        return Variant::Churn(paged_churn::FIXED);
    };
    let numbers = |rest: &str| -> (u32, u32) {
        let mut parts = rest.split(':').map(|p| p.parse().unwrap());
        (parts.next().unwrap(), parts.next().unwrap())
    };
    if let Some(rest) = arg.strip_prefix("wide:") {
        let (b, k) = numbers(rest);
        return Variant::Wide(b, k);
    }
    if let Some(rest) = arg.strip_prefix("stores:") {
        let (b, s) = numbers(rest);
        return Variant::Stores(b, s, paged_churn::Stores::CodePage);
    }
    if let Some(rest) = arg.strip_prefix("datastores:") {
        let (b, s) = numbers(rest);
        return Variant::Stores(b, s, paged_churn::Stores::Data);
    }
    if let Some(rest) = arg.strip_prefix("contexts:") {
        let (b, n) = numbers(rest);
        return Variant::Contexts(b, n);
    }
    match arg {
        "rotate" => Variant::Churn(paged_churn::ROTATE),
        "rotate128" => Variant::Churn(paged_churn::ROTATE_MANY),
        _ => Variant::Churn(paged_churn::FIXED),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: u32 = args.get(1).map_or(200_000, |s| s.parse().unwrap());
    let rounds: usize = args.get(2).map_or(3, |s| s.parse().unwrap());
    let variant = variant(args.get(3).map(String::as_str));
    let workload = match (variant, args.get(3)) {
        (Variant::Churn(paged_churn::ROTATE), _) => "paged-churn-rotate",
        (Variant::Churn(paged_churn::ROTATE_MANY), _) => "paged-churn-rotate128",
        (Variant::Churn(_), _) => "paged-churn",
        (_, Some(arg)) => arg.as_str(),
        (_, None) => unreachable!(),
    };
    println!("workload,round,iterations,retired,seconds,mips,host_per_step,lifts_per_k");
    for round in 0..rounds {
        let model = CpuModel::I80486Dx;
        let (vm, mut cpu) = match variant {
            Variant::Wide(blocks, directories) => {
                paged_churn::setup_wide(model, n, blocks, directories)
            }
            Variant::Stores(blocks, stride, stores) => {
                paged_churn::setup_stores(model, n, blocks, stride, stores)
            }
            Variant::Contexts(blocks, stamps) => {
                paged_churn::setup_contexts(model, n, blocks, stamps)
            }
            Variant::Churn(rotate) => paged_churn::setup_with(model, n, rotate),
        };
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
        match variant {
            Variant::Wide(blocks, _) => {
                assert_eq!(retired, paged_churn::expected_retired_wide(n, blocks));
            }
            Variant::Stores(blocks, stride, _) => {
                assert_eq!(
                    retired,
                    paged_churn::expected_retired_stores(n, blocks, stride)
                );
            }
            Variant::Contexts(blocks, _) => {
                assert_eq!(retired, paged_churn::expected_retired_contexts(n, blocks));
            }
            Variant::Churn(rotate) => {
                assert_eq!(retired, paged_churn::expected_retired_with(n, rotate));
                assert_eq!(cpu.cpu.gpr[5] as u32, paged_churn::expected_sum(n));
            }
        }
        #[allow(unused_mut, unused_assignments)]
        let mut lifts = 0;
        #[cfg(feature = "test-hooks")]
        {
            lifts = x86jit_core::Vcpu::take_machine_counters_for_test()[2];
            if round == 0 {
                eprintln!("{}", vm.relift_diagnostics(true));
                let (now, peak, count) = cpu.take_paged_proof_bytes_for_test();
                let index = cpu.paged_proof_index_bytes_for_test();
                eprintln!(
                    "paged proofs: {count}, {now} bytes now ({} per proof), peak {peak}; \
                     source index {index} bytes ({} per proof)",
                    now / count.max(1),
                    index / count.max(1)
                );
            }
        }
        println!(
            "{workload},{round},{n},{retired},{seconds:.6},{:.3},{:.1},{:.3}",
            retired as f64 / seconds / 1e6,
            host as f64 / retired as f64,
            lifts as f64 * 1000.0 / retired as f64
        );
    }
}
