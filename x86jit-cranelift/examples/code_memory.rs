//! Direct small-unit probe (decision-17), separate from the background-tier example.
#[path = "../tests/support/code_memory.rs"]
mod probe;
use std::sync::{Arc, Mutex};
use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
use x86jit_core::{Backend, CachedBlock, MemConsistency, Memory, MemoryModel, Prot, RegionKind};
use x86jit_cranelift::{JitBackend, OptLevel};
fn main() {
    let n: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let level = match std::env::args().nth(2).as_deref() {
        Some("speed") => OptLevel::Speed,
        _ => OptLevel::None,
    };
    let mut mem = Memory::new(MemoryModel::Flat { size: 65536 });
    mem.map(0, 65536, Prot::RWX, RegionKind::Ram).unwrap();
    mem.write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xf4]).unwrap();
    let mut ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let jit = JitBackend::with_opt_level(level);
    let ranges = Arc::new(Mutex::new(Vec::with_capacity(n)));
    let observed = ranges.clone();
    jit.set_compiled_code_observer(move |code| {
        observed
            .lock()
            .unwrap()
            .push((code.host_start, code.host_len));
    });
    probe::snapshot("baseline", &[]);
    for i in 0..n {
        ir.guest_start = 0x1000 + i as u64 * 16;
        assert!(matches!(
            jit.materialize(&ir, MemConsistency::Fast, None, 0),
            CachedBlock::Compiled { .. }
        ));
    }
    let ranges = ranges.lock().unwrap();
    println!(
        "units={} code_bytes={} opt={level:?}",
        ranges.len(),
        ranges.iter().map(|r| u64::from(r.1)).sum::<u64>()
    );
    let measured = probe::snapshot("compiled", &ranges);
    println!("measure={measured:?} counters={:#?}", jit.memory_stats());
    let retained_ranges = ranges.clone();
    drop(ranges);
    drop(jit);
    probe::snapshot("dropped", &retained_ranges);
}
