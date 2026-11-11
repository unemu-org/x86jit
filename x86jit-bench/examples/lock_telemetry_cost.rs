//! Bounded disabled-telemetry measurement, without updating benchmark records.
//! Build and run this example in release mode against both revisions using the
//! same toolchain. Each case warms up twice and reports nine samples.

#[allow(dead_code)]
#[path = "../src/workloads.rs"]
mod workloads;

use std::time::Instant;
use x86jit_core::{Backend, CachedBlock, IrBlock, IrOp, MemConsistency};
use x86jit_cranelift::JitBackend;

fn main() {
    println!("workload,median_ns,min_ns,mad_ns,samples");
    for wl in workloads::all()
        .into_iter()
        .filter(|w| ["fib32", "hotloop"].contains(&w.name))
    {
        measure(wl.name, || {
            let (out, _) = (wl.guest)(
                workloads::jit(workloads::TierCfg::EAGER),
                workloads::TierCfg::EAGER,
            );
            assert_eq!(out, wl.expect);
        });
    }
    measure("compile1000", || {
        let jit = JitBackend::new();
        let mut ir = IrBlock {
            legacy_context: None,
            mode: x86jit_core::CpuMode::Long64,
            guest_start: 0x1000,
            ops: vec![IrOp::Hlt],
            temp_count: 0,
            guest_len: 1,
            icount: 1,
        };
        for i in 0..1000 {
            ir.guest_start = 0x1000 + i;
            assert!(matches!(
                jit.materialize(&ir, MemConsistency::Fast, None, 0),
                CachedBlock::Compiled { .. }
            ));
        }
    });
}

fn measure(name: &str, mut run: impl FnMut()) {
    for _ in 0..2 {
        run();
    }
    let mut samples = Vec::with_capacity(9);
    for _ in 0..9 {
        let start = Instant::now();
        run();
        samples.push(start.elapsed().as_nanos());
    }
    samples.sort_unstable();
    let median = samples[4];
    let mut deviations: Vec<_> = samples.iter().map(|ns| ns.abs_diff(median)).collect();
    deviations.sort_unstable();
    println!("{name},{median},{},{},9", samples[0], deviations[4]);
}
