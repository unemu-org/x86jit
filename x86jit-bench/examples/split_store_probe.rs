//! One store workload per process for interleaved reference/candidate measurements.
//! Arguments: WORKLOAD jit|region default|routed. Output includes compilation.
#[allow(dead_code)]
#[path = "../src/workloads.rs"]
mod workloads;

use iced_x86::code_asm::*;
use std::time::Instant;
use x86jit_core::{Backend, Exit, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::{HostTarget, JitBackend, OptLevel};

#[cfg(target_os = "macos")]
fn retired() -> u64 {
    // macOS SDK sys/resource.h, RUSAGE_INFO_V4 through ri_runnable_time.
    #[repr(C)]
    #[derive(Default)]
    struct Usage {
        uuid: [u8; 16],
        before: [u64; 29],
        instructions: u64,
        after: [u64; 5],
    }
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut Usage) -> i32;
    }
    let mut usage = Usage::default();
    // SAFETY: usage has the SDK's V4 layout and remains live for the call.
    assert_eq!(
        unsafe { proc_pid_rusage(std::process::id() as i32, 4, &mut usage) },
        0
    );
    assert!(usage.instructions > 0, "retired counter unavailable");
    usage.instructions
}
#[cfg(not(target_os = "macos"))]
fn retired() -> u64 {
    0
}

fn backend(region: bool) -> Box<dyn Backend> {
    Box::new(JitBackend::with_options(
        region.then_some(x86jit_core::RegionCaps {
            max_blocks: 16,
            max_icount: 256,
        }),
        HostTarget::Native,
        OptLevel::Speed,
    ))
}

fn stores(width: usize, sliding: bool, region: bool, routed: bool) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x100000), backend(region));
    vm.set_tier_up_after(None);
    vm.mem.set_host_fault_recovery(routed);
    vm.map(0, 0x100000, Prot::RWX, RegionKind::Ram).unwrap();
    let data = 0x40000 + (vm.mem.host_base() as usize).wrapping_neg() % 64;
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    a.mov(ecx, 100_000).unwrap();
    a.mov(edi, data as u32).unwrap();
    a.mov(eax, 0x35353535).unwrap();
    a.set_label(&mut top).unwrap();
    for _ in 0..8 {
        match width {
            2 => a.mov(word_ptr(rdi), ax).unwrap(),
            4 => a.mov(dword_ptr(rdi), eax).unwrap(),
            8 => a.mov(qword_ptr(rdi), rax).unwrap(),
            16 => a.movdqu(xmmword_ptr(rdi), xmm0).unwrap(),
            32 => a.vmovdqu(ymmword_ptr(rdi), ymm0).unwrap(),
            64 => a.vmovdqu32(zmmword_ptr(rdi), zmm0).unwrap(),
            _ => unreachable!(),
        }
        if sliding {
            a.inc(edi).unwrap();
            a.and(edi, 0x41fff).unwrap();
        }
    }
    a.dec(ecx).unwrap();
    a.jnz(top).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
        .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    cpu.set_xmm(0, u128::from_le_bytes([0x35; 16]));
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert!(vm.mem.read(data as u64, 2).is_ok());
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 3, "WORKLOAD jit|region default|routed");
    let region = match args[1].as_str() {
        "jit" => false,
        "region" => true,
        _ => panic!("tier"),
    };
    let routed = match args[2].as_str() {
        "default" => false,
        "routed" => true,
        _ => panic!("mode"),
    };
    let before = retired();
    let start = Instant::now();
    if let Some((pattern, width)) = args[0].split_once('-') {
        stores(width.parse().unwrap(), pattern == "sliding", region, routed);
    } else {
        assert!(!routed, "existing bench workloads use the default policy");
        let workload = workloads::all()
            .into_iter()
            .find(|w| w.name == args[0])
            .unwrap();
        let (output, _) = (workload.guest)(backend(region), workloads::TierCfg::EAGER);
        assert_eq!(output, workload.expect);
    }
    println!(
        "{},{},{},ns={},retired={}",
        args[0],
        args[1],
        args[2],
        start.elapsed().as_nanos(),
        retired() - before
    );
}
