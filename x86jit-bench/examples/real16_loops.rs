//! Real16 embedder loop benchmarks using only the core API. Release mode;
//! warmup excluded. Optional argument scales step counts (default 1).
use std::time::Instant;
use x86jit_core::{
    BlockKey, CachedBlock, CpuMode, Exit, IrBlock, IrOp, Prot, Reg, RegionKind, StepOptions,
    StepStop, Vcpu, Vm, VmConfig,
};

fn setup(cs: u64, ip: u64, ds: u64) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x10_0000));
    vm.set_cpu_mode(CpuMode::Real16);
    vm.map(0, 0xa_0000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut cpu = vm.new_vcpu();
    for (reg, val) in [
        (Reg::Cs, cs),
        (Reg::Rip, ip),
        (Reg::Ds, ds),
        (Reg::Es, ds),
        (Reg::Ss, 0),
        (Reg::Rsp, 0x7000),
    ] {
        cpu.set_reg(reg, val);
    }
    (vm, cpu)
}
fn steps(vm: &Vm, cpu: &mut Vcpu, steps: u64) {
    let mut done = 0;
    while done < steps {
        let r = cpu
            .run_steps(vm, steps - done, StepOptions::default())
            .unwrap();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
        assert!(r.steps() != 0);
        done += r.steps();
    }
}
fn bench(name: &str, code: &[u8], count: u64, ds: u64) {
    bench_cached(name, code, count, ds, 0);
}
fn bench_cached(name: &str, code: &[u8], count: u64, ds: u64, fillers: u64) {
    let (vm, mut cpu) = setup(0, 0x7c00, ds);
    vm.write_bytes(0x7c00, code).unwrap();
    for i in 0..fillers {
        let pc = 0x7000 + i;
        vm.cache.insert(
            BlockKey::new(pc, CpuMode::Real16),
            CachedBlock::Interpreted(std::sync::Arc::new(IrBlock {
                mode: CpuMode::Real16,
                guest_start: pc,
                guest_len: 1,
                icount: 1,
                temp_count: 0,
                ops: vec![IrOp::Hlt],
                legacy_context: None,
            })),
            vec![(pc, 1)],
            |_| vm.mem.mark_code(pc, 1),
        );
    }
    steps(&vm, &mut cpu, 1_000_000);
    let misses = vm.cache.misses();
    let start = Instant::now();
    steps(&vm, &mut cpu, count);
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{name},{count},{:.3},{},{:.1}",
        secs * 1e9 / count as f64,
        vm.cache.misses() - misses,
        (vm.cache.misses() - misses) as f64 / secs
    );
}
fn looped(body: &[u8]) -> Vec<u8> {
    let mut v = vec![0xb9, 0, 0];
    v.extend_from_slice(body);
    v.extend_from_slice(&[
        0xe2,
        (-(body.len() as i32 + 2)) as u8,
        0xeb,
        (-(body.len() as i32 + 7)) as u8,
    ]);
    v
}
fn main() {
    let scale: u64 = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(1);
    println!("name,steps,ns_per_step,lifts,lifts_per_second");
    for (name, body, count) in [
        ("alu", vec![0x01, 0xd8, 0x46, 0x31, 0xc2], 50_000_000),
        ("load", vec![0x03, 0x06, 0x00, 0x10, 0x01, 0xd8], 20_000_000),
        (
            "load-odd",
            vec![0x03, 0x06, 0x01, 0x10, 0x01, 0xd8],
            20_000_000,
        ),
        (
            "store",
            vec![0x89, 0x06, 0x00, 0x10, 0x01, 0xd8],
            20_000_000,
        ),
        ("push-pop", vec![0x50, 0x5b, 0x01, 0xd8], 20_000_000),
    ] {
        bench(name, &looped(&body), count / scale, 0x1000);
    }
    bench(
        "data-next-to-code",
        &looped(&[0x89, 0x06, 0x00, 0x78, 0x01, 0xd8]),
        20_000_000 / scale,
        0,
    );
    bench_cached(
        "data-next-to-dense-code",
        &looped(&[0x89, 0x06, 0x00, 0x78, 0x01, 0xd8]),
        20_000_000 / scale,
        0,
        1024,
    );
    bench(
        "call-ret",
        &[0xb9, 0, 0, 0xe8, 4, 0, 0xe2, 0xfb, 0xeb, 0xf6, 0xc3],
        20_000_000 / scale,
        0x1000,
    );
    bench(
        "rep-movsw",
        &[
            0x31, 0xf6, 0x31, 0xff, 0xb9, 0, 0x40, 0xf3, 0xa5, 0xeb, 0xf5,
        ],
        100_000_000 / scale,
        0x1000,
    );
    let pat = [0x01, 0xd8, 0x89, 0x40, 0x10, 0x46];
    let mut code = Vec::new();
    while code.len() + pat.len() <= 0xef00 {
        code.extend_from_slice(&pat);
    }
    let disp = (-(code.len() as i32 + 3)) as i16 as u16;
    let per_pass = (code.len() / pat.len() * 3 + 1) as u64;
    code.extend_from_slice(&[0xe9, disp as u8, (disp >> 8) as u8]);
    let (vm, mut cpu) = setup(0x1000, 0x100, 0x3000);
    vm.write_bytes(0x10100, &code).unwrap();
    for pass in 0..4 {
        let start = Instant::now();
        steps(&vm, &mut cpu, per_pass);
        println!(
            "straight-line-{pass},{per_pass},{:.3}",
            start.elapsed().as_secs_f64() * 1e9 / per_pass as f64
        );
    }
    for (name, cs, ip, writes) in [
        (
            "ip-wrap",
            0x1000,
            0xfffe,
            vec![
                (0x1fffe, vec![0x90, 0x90]),
                (0x10000, vec![0x42, 0xf4]),
                (0x20000, vec![0x43, 0xf4]),
            ],
        ),
        (
            "smc-next",
            0,
            0x7c00,
            vec![(0x7c00, vec![0xc6, 6, 6, 0x7c, 0x42, 0xb0, 0, 0xf4])],
        ),
    ] {
        for limit in [1, 64] {
            let (vm, mut cpu) = setup(cs, ip, cs);
            for (addr, bytes) in &writes {
                vm.write_bytes(*addr, bytes).unwrap();
            }
            let mut halted = false;
            for _ in 0..64 {
                let r = cpu.run_steps(&vm, limit, StepOptions::default()).unwrap();
                if matches!(r.stop(), StepStop::Exit(Exit::Hlt)) {
                    halted = true;
                    break;
                }
            }
            assert!(halted);
            assert_eq!(
                cpu.reg(if name == "ip-wrap" {
                    Reg::Rdx
                } else {
                    Reg::Rax
                }),
                if name == "ip-wrap" { 1 } else { 0x42 }
            );
            println!(
                "{name},limit={limit},ax={},bx={},dx={}",
                cpu.reg(Reg::Rax),
                cpu.reg(Reg::Rbx),
                cpu.reg(Reg::Rdx)
            );
        }
    }
}
