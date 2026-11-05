//! Conversion cost probe: X86JIT_ICOUNT=1 enables the guest counter.
//! On macOS, `host` counts retired process instructions, including JIT compilation.

#[allow(dead_code)]
#[path = "../src/workloads.rs"]
mod workloads;

use iced_x86::code_asm::*;
use std::time::Instant;
use x86jit_core::{Backend, Exit, InterpreterBackend, Prot, Reg, RegionKind, Vm, VmConfig};

#[cfg(target_os = "macos")]
fn host_instructions() -> u64 {
    // macOS SDK sys/resource.h: RUSAGE_INFO_V4, through ri_runnable_time.
    #[repr(C)]
    #[derive(Default)]
    struct RusageInfoV4 {
        uuid: [u8; 16],
        before_instructions: [u64; 29],
        instructions: u64,
        after_instructions: [u64; 5],
    }
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfoV4) -> i32;
    }
    let mut info = RusageInfoV4::default();
    // SAFETY: info has the SDK's V4 layout and lives through the synchronous call.
    let result = unsafe { proc_pid_rusage(std::process::id() as i32, 4, &mut info) };
    assert_eq!(result, 0, "proc_pid_rusage failed");
    assert!(
        info.instructions > 0,
        "retired-instruction counter unavailable"
    );
    info.instructions
}

#[cfg(not(target_os = "macos"))]
fn host_instructions() -> u64 {
    0
}

fn main() {
    let host_start = host_instructions();
    let args: Vec<_> = std::env::args().collect();
    let workload = args.get(1).expect("simd or convert");
    let tier = args.get(2).expect("interp or jit");
    let backend: Box<dyn Backend> = match tier.as_str() {
        "interp" => Box::new(InterpreterBackend),
        "jit" => workloads::jit(workloads::TierCfg::EAGER),
        _ => panic!("interp or jit"),
    };
    if workload == "simd" {
        let wl = workloads::all()
            .into_iter()
            .find(|w| w.name == "simd")
            .unwrap();
        let start = Instant::now();
        let (out, counts) = (wl.guest)(backend, workloads::TierCfg::EAGER);
        assert_eq!(out, wl.expect);
        assert!(counts.executed > 0, "run with X86JIT_ICOUNT=1");
        println!(
            "simd,{tier},ns={},guest={},helpers={},host={}",
            start.elapsed().as_nanos(),
            counts.executed,
            counts.helper_calls,
            host_instructions() - host_start
        );
    } else {
        assert_eq!(workload, "convert");
        let mut a = CodeAssembler::new(64).unwrap();
        let mut top = a.create_label();
        a.mov(ecx, 100_000).unwrap();
        a.set_label(&mut top).unwrap();
        for _ in 0..4 {
            a.cvtps2dq(xmm0, xmm1).unwrap();
            a.cvttps2dq(xmm2, xmm1).unwrap();
            a.cvtpd2dq(xmm3, xmm4).unwrap();
            a.cvttpd2dq(xmm5, xmm4).unwrap();
            a.cvtsd2si(rax, xmm4).unwrap();
            a.cvttsd2si(edx, xmm4).unwrap();
        }
        a.dec(ecx).unwrap();
        a.jnz(top).unwrap();
        a.hlt().unwrap();
        let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
            .unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.set_xmm(1, 0x3fc00000_3fc00000_3fc00000_3fc00000);
        cpu.set_xmm(4, 0x3ff8000000000000_3ff8000000000000);
        let start = Instant::now();
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        let ns = start.elapsed().as_nanos();
        assert!(cpu.executed_instructions() > 0, "run with X86JIT_ICOUNT=1");
        assert_eq!(cpu.reg(Reg::Rax), 2);
        assert_eq!(cpu.reg(Reg::Rdx), 1);
        assert_eq!(cpu.xmm(0), 0x00000002_00000002_00000002_00000002);
        assert_eq!(cpu.xmm(2), 0x00000001_00000001_00000001_00000001);
        let helpers: u64 = vm.backend.helper_calls().iter().map(|(_, n)| n).sum();
        println!(
            "convert,{tier},ns={ns},guest={},helpers={helpers},host={}",
            cpu.executed_instructions(),
            host_instructions() - host_start
        );
    }
}
