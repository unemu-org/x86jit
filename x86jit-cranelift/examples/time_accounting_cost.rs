//! Short transition-accounting benchmark; never updates performance records.
//! `--enabled` enables timing; `--hotloop` uses the existing benchmark's multi-block
//! hotloop shape (x86jit-bench/src/workloads.rs), otherwise a two-instruction loop.
//! `--helpers` uses two x87 helper calls per iteration.
//! Two warmups and fifteen measured samples; snapshots are never read.
use std::time::Instant;
use x86jit_core::{Backend, Exit, InterpreterBackend, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

fn main() {
    println!("backend,median_ns,min_ns,mad_ns");
    for jit in [false, true] {
        let backend: Box<dyn Backend> = if jit {
            Box::new(JitBackend::new())
        } else {
            Box::new(InterpreterBackend)
        };
        let mut vm = Vm::with_backend(VmConfig::flat(0x4000), backend);
        vm.set_time_accounting(std::env::args().any(|a| a == "--enabled"));
        vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        let code: &[u8] = if std::env::args().any(|a| a == "--helpers") {
            // fld1; fstp st(0); dec ecx; jnz -8; hlt
            // Both x87 instructions call x87_helper in the compiled tier.
            &[0xd9, 0xe8, 0xdd, 0xd8, 0xff, 0xc9, 0x75, 0xf8, 0xf4]
        } else if std::env::args().any(|a| a == "--hotloop") {
            // xor eax,eax; top: test ecx,3; jz quad; add eax,ecx; xor edx,edx;
            // jmp cont; quad: imul eax,eax; add eax,7; cont: dec ecx; jnz top; hlt
            &[
                0x31, 0xc0, 0xf7, 0xc1, 3, 0, 0, 0, 0x74, 6, 0x01, 0xc8, 0x31, 0xd2, 0xeb, 6, 0x0f,
                0xaf, 0xc0, 0x83, 0xc0, 7, 0xff, 0xc9, 0x75, 0xe8, 0xf4,
            ]
        } else {
            // dec ecx; jnz -4; hlt
            &[0xff, 0xc9, 0x75, 0xfc, 0xf4]
        };
        vm.write_bytes(0x1000, code).unwrap();
        let mut cpu = vm.new_vcpu();
        let mut samples = Vec::new();
        for sample in 0..17 {
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rcx, if jit { 2_000_000 } else { 20_000 });
            let start = Instant::now();
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            if sample >= 2 {
                samples.push(start.elapsed().as_nanos());
            }
        }
        samples.sort_unstable();
        let median = samples[7];
        let mut deviations: Vec<_> = samples.iter().map(|n| n.abs_diff(median)).collect();
        deviations.sort_unstable();
        println!(
            "{},{median},{},{}",
            if jit { "jit" } else { "interp" },
            samples[0],
            deviations[7]
        );
    }
}
