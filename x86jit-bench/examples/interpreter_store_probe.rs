//! Store-heavy interpreter timing. Arguments: WIDTH single|crossing off|on ITERATIONS.
//! Build identical copies against both revisions and interleave process runs.
//! Reports ns per guest store, including interpreter dispatch and loop overhead,
//! and, on macOS, proc_pid_rusage host instructions per guest store around the
//! measured run (zero elsewhere). Output:
//! width,placement,recovery,stores,ns_per_store,instructions_per_store
//! `scripts/perf-campaign.sh` runs it as the counted store probe.
use iced_x86::code_asm::*;
use std::time::Instant;
use x86jit_core::{Exit, InterpreterBackend, Prot, Reg, RegionKind, Vm, VmConfig};

#[cfg(target_os = "macos")]
fn instructions() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::uninit();
    // SAFETY: flavor V4 writes exactly rusage_info_v4; the buffer is only read
    // after success. libproc takes the buffer address through rusage_info_t*.
    unsafe {
        assert_eq!(
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                usage.as_mut_ptr().cast()
            ),
            0,
            "proc_pid_rusage: {}",
            std::io::Error::last_os_error()
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

#[cfg(unix)]
fn host_page_size() -> usize {
    unsafe extern "C" {
        fn getpagesize() -> std::ffi::c_int;
    }
    // SAFETY: getpagesize has no pointer arguments.
    unsafe { getpagesize() as usize }
}
#[cfg(not(unix))]
fn host_page_size() -> usize {
    4096
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 4, "WIDTH single|crossing off|on ITERATIONS");
    let width: usize = args[0].parse().unwrap();
    let crossing = match args[1].as_str() {
        "single" => false,
        "crossing" => true,
        _ => panic!("placement"),
    };
    let routed = match args[2].as_str() {
        "off" => false,
        "on" => true,
        _ => panic!("recovery"),
    };
    let iterations: u32 = args[3].parse().unwrap();
    assert!(iterations > 0);
    let mut vm = Vm::with_backend(VmConfig::flat(0x40000), Box::new(InterpreterBackend));
    vm.set_tier_up_after(None);
    vm.mem.set_host_fault_recovery(routed);
    vm.map(0, 0x40000, Prot::RWX, RegionKind::Ram).unwrap();
    let host = vm.mem.host_base() as usize;
    let page = host_page_size();
    let edge = (host + 0x20000).next_multiple_of(page) - host;
    let address = edge - if crossing { width / 2 } else { 64 };
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    a.mov(ecx, iterations).unwrap();
    a.set_label(&mut top).unwrap();
    for _ in 0..16 {
        match width {
            8 => a.mov(qword_ptr(rdi), rax).unwrap(),
            16 => a.movdqu(xmmword_ptr(rdi), xmm0).unwrap(),
            32 => a.vmovdqu(ymmword_ptr(rdi), ymm0).unwrap(),
            64 => a.vmovdqu32(zmmword_ptr(rdi), zmm0).unwrap(),
            _ => panic!("width must be 8, 16, 32 or 64"),
        }
    }
    a.dec(ecx).unwrap();
    a.jnz(top).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
        .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rdi, address as u64);
    cpu.set_reg(Reg::Rax, 0x3535_3535_3535_3535);
    cpu.cpu.set_vec(0, [u128::from_le_bytes([0x35; 16]); 4], 64);
    // Populate decoded-block caches before timing; warm and measured runs both store.
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    cpu.set_reg(Reg::Rip, 0x1000);
    let before = instructions();
    let start = Instant::now();
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let elapsed = start.elapsed();
    let host_instructions = instructions() - before;
    assert_eq!(
        vm.mem.read(address as u64, 8).unwrap(),
        0x3535_3535_3535_3535
    );
    assert_eq!(
        vm.mem.read((address + width - 8) as u64, 8).unwrap(),
        0x3535_3535_3535_3535
    );
    let stores = u64::from(iterations) * 16;
    println!(
        "{width},{},{},{stores},{:.3},{:.3}",
        args[1],
        args[2],
        elapsed.as_nanos() as f64 / stores as f64,
        host_instructions as f64 / stores as f64
    );
}
