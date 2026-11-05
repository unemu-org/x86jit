//! One workload/tier per process, suitable for `/usr/bin/time -l` counters.
//! Measure A/B with identical copies of this driver: three interleaved rounds,
//! subtract the one-run retired count from the eleven-run count, then divide by ten.
//! On systems where `time -l` is restricted, `proc_pid_rusage(RUSAGE_INFO_V4)`
//! supplies the same process retired-instruction counter before the child is reaped.
//! Region mode compiles synchronously at Speed to avoid background scheduling noise.
//! Example: `cargo run --release -p x86jit-bench --example access_probe -- movaps jit 11`.
#[path = "access_probe/probes.rs"]
mod probes;
#[allow(dead_code)]
#[path = "../src/workloads.rs"]
mod workloads;

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        3,
        "usage: access_probe WORKLOAD interp|jit|region RUNS"
    );
    let workload = workloads::all()
        .into_iter()
        .chain(probes::all())
        .find(|w| w.name == args[0])
        .expect("workload");
    let runs: usize = args[2].parse().expect("runs");
    for _ in 0..runs {
        let (backend, tier): (Box<dyn x86jit_core::Backend>, _) = match args[1].as_str() {
            "interp" => (workloads::interp(), workloads::TierCfg::EAGER),
            "jit" => (
                workloads::jit(workloads::TierCfg::EAGER),
                workloads::TierCfg::EAGER,
            ),
            "region" => (
                Box::new(x86jit_cranelift::JitBackend::with_options(
                    Some(x86jit_core::RegionCaps {
                        max_blocks: 16,
                        max_icount: 256,
                    }),
                    x86jit_cranelift::HostTarget::Native,
                    x86jit_cranelift::OptLevel::Speed,
                )),
                workloads::TierCfg::EAGER,
            ),
            _ => panic!("tier must be interp, jit or region"),
        };
        let (output, _) = (workload.guest)(backend, tier);
        assert_eq!(output, workload.expect);
        std::hint::black_box(output);
    }
}
