//! One workload per process; run under /usr/bin/time -l. Also report a scoped
//! proc_pid_rusage counter on macOS when sandbox policy blocks time -l statistics.
//! Arguments: real16-cold|jit-stores [degraded]. Build with test-hooks to force fallback.
use std::time::Instant;
use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};

#[cfg(target_os = "macos")]
fn retired() -> u64 {
    // Apple SDK sys/resource.h, rusage_info_v4: ri_instructions follows 29 u64s.
    #[repr(C)]
    #[derive(Default)]
    struct Usage {
        uuid: [u8; 16],
        before: [u64; 29],
        instructions: u64,
        after: [u64; 5],
    }
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut Usage) -> i32;
    }
    let mut usage = Usage::default();
    // SAFETY: V4 storage has the SDK's exact layout and lives through the call.
    assert_eq!(
        unsafe { proc_pid_rusage(std::process::id() as i32, 4, &mut usage) },
        0
    );
    assert_ne!(
        usage.instructions, 0,
        "retired instruction counter unavailable"
    );
    usage.instructions
}

#[cfg(not(target_os = "macos"))]
fn retired() -> u64 {
    0
}

fn construct(build: impl FnOnce() -> Vm) -> Vm {
    #[cfg(feature = "test-hooks")]
    if std::env::args().nth(2).as_deref() == Some("degraded") {
        return x86jit_core::with_barrier_unavailable(build);
    }
    build()
}

fn real16_cold() {
    use x86jit_core::{CpuMode, StepOptions, StepStop};
    let mut vm = construct(|| Vm::new(VmConfig::flat(0x10_0000)));
    vm.set_cpu_mode(CpuMode::Real16);
    vm.map(0, 0xa_0000, Prot::RWX, RegionKind::Ram).unwrap();
    // The real16_loops straight-line cold pass, without unrelated warm workloads.
    let pattern = [0x01, 0xd8, 0x89, 0x40, 0x10, 0x46];
    let mut code = Vec::new();
    while code.len() + pattern.len() <= 0xef00 {
        code.extend_from_slice(&pattern);
    }
    let displacement = (-(code.len() as i32 + 3)) as i16 as u16;
    let count = (code.len() / pattern.len() * 3 + 1) as u64;
    code.extend_from_slice(&[0xe9, displacement as u8, (displacement >> 8) as u8]);
    vm.write_bytes(0x10100, &code).unwrap();
    let mut cpu = vm.new_vcpu();
    for (reg, value) in [
        (Reg::Cs, 0x1000),
        (Reg::Rip, 0x100),
        (Reg::Ds, 0x3000),
        (Reg::Es, 0x3000),
        (Reg::Ss, 0),
        (Reg::Rsp, 0x7000),
    ] {
        cpu.set_reg(reg, value);
    }
    let start = Instant::now();
    let mut done = 0;
    while done < count {
        let result = cpu
            .run_steps(&vm, count - done, StepOptions::default())
            .unwrap();
        assert!(matches!(
            result.stop(),
            StepStop::Exit(Exit::BudgetExhausted)
        ));
        assert_ne!(result.steps(), 0);
        done += result.steps();
    }
    println!(
        "real16-cold,steps={count},ns={},lifts={}",
        start.elapsed().as_nanos(),
        vm.cache.misses()
    );
    print_mode(&vm);
}

fn jit_stores() {
    use iced_x86::code_asm::*;
    let mut vm = construct(|| {
        Vm::with_backend(
            VmConfig::flat(0x80000),
            Box::new(x86jit_cranelift::JitBackend::new()),
        )
    });
    vm.map(0, 0x80000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut asm = CodeAssembler::new(64).unwrap();
    let mut top = asm.create_label();
    asm.mov(ecx, 1_000_000).unwrap();
    asm.mov(edi, 0x40000).unwrap();
    asm.mov(eax, 42).unwrap();
    asm.set_label(&mut top).unwrap();
    for offset in (0..64).step_by(8) {
        asm.mov(qword_ptr(rdi + offset), rax).unwrap();
    }
    asm.dec(ecx).unwrap();
    asm.jnz(top).unwrap();
    asm.hlt().unwrap();
    vm.write_bytes(0x1000, &asm.assemble(0x1000).unwrap())
        .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    let start = Instant::now();
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    for offset in (0..64).step_by(8) {
        assert_eq!(vm.mem.read(0x40000 + offset, 8).unwrap(), 42);
    }
    assert!(vm.memory_stats().backend.compiled_units_total > 0);
    println!(
        "jit-stores,stores=8000000,ns={}",
        start.elapsed().as_nanos()
    );
    print_mode(&vm);
}

fn print_mode(vm: &Vm) {
    #[cfg(feature = "test-hooks")]
    {
        let stats = vm.memory_stats();
        println!(
            "mode={:?},failures={},errno={:?}",
            stats.source_barrier_mode,
            stats.source_barrier_failures,
            stats.source_barrier_last_errno
        );
    }
    let _ = vm;
}

fn main() {
    let before = retired();
    match std::env::args().nth(1).as_deref() {
        Some("real16-cold") => real16_cold(),
        Some("jit-stores") => jit_stores(),
        _ => panic!("expected real16-cold|jit-stores [degraded]"),
    }
    println!("scoped_retired={}", retired() - before);
}
