//! Synthetic allocation/RSS probe: cargo run --release -p x86jit-cranelift
//! --example memory_stats -- 10000 102
#[path = "../tests/support/code_memory.rs"]
mod probe;
use std::sync::{Arc, Mutex};
use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn rss_bytes() -> Option<u64> {
    let mut info = std::mem::MaybeUninit::<libc::mach_task_basic_info>::uninit();
    let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
    // SAFETY: task_info writes the count-bounded structure for this process.
    let result = unsafe {
        libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            info.as_mut_ptr().cast(),
            &mut count,
        )
    };
    if result == libc::KERN_SUCCESS {
        // SAFETY: successful task_info initialized the structure.
        Some(unsafe { info.assume_init() }.resident_size)
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(resident * region::page::size() as u64)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn rss_bytes() -> Option<u64> {
    None
}

fn sample(phase: &str, vm: &Vm) {
    println!(
        "{phase}: rss_bytes={:?}\n{:#?}",
        rss_bytes(),
        vm.memory_stats()
    );
    println!("codemap: {:#?}", x86jit_core::codemap::memory_stats());
}

fn main() {
    let units: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(10000);
    let vcpus: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(102);
    assert!(units > 0 && vcpus > 0);
    println!(
        "host={}-{} page_bytes={} units={units} vcpus={vcpus} tier=32 background=true",
        std::env::consts::ARCH,
        std::env::consts::OS,
        region::page::size()
    );
    let jit = JitBackend::new();
    let ranges = Arc::new(Mutex::new(Vec::with_capacity(units)));
    let observed = ranges.clone();
    jit.set_compiled_code_observer(move |code| {
        observed
            .lock()
            .unwrap()
            .push((code.host_start, code.host_len))
    });
    let handle = jit.tier_up_handle();
    let size = 0x1000 + units * 16;
    let mut vm = Vm::with_backend(VmConfig::flat(size as u64), Box::new(jit));
    vm.set_tier_up_after(Some(32));
    vm.set_tier_up_background(true);
    vm.map(0, size, Prot::RWX, RegionKind::Ram).unwrap();
    for i in 0..units {
        vm.write_bytes(
            (0x1000 + i * 16) as u64,
            &[0xb8, 7, 0, 0, 0, 0x48, 0xff, 0xc0, 0xf4],
        )
        .unwrap();
    }
    let mut cpus: Vec<_> = (0..vcpus).map(|_| vm.new_vcpu()).collect();
    sample("baseline", &vm);
    let pass = |cpu: &mut x86jit_core::Vcpu| {
        for i in 0..units {
            cpu.set_reg(Reg::Rip, (0x1000 + i * 16) as u64);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax), 8);
        }
    };
    pass(&mut cpus[0]);
    sample("interpreted", &vm);
    for _ in 0..32 {
        pass(&mut cpus[0]);
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        handle.wait_idle();
        pass(&mut cpus[0]); // drain completions and retry queue backpressure
        if vm.memory_stats().cache.compiled_entries == units as u64 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "tier-up did not finish"
        );
    }
    handle.wait_idle();
    sample("compiled", &vm);
    probe::snapshot("compiled", &ranges.lock().unwrap());
    vm.cache.invalidate_overlapping(0, u64::MAX, || {});
    sample("invalidated", &vm);
    drop(cpus);
    sample("vcpus_dropped", &vm);
}
