//! Real16 embedder loop benchmarks using only the core API. Release mode;
//! warmup excluded. Optional argument divides step counts (default 1).
//! macOS reports proc_pid_rusage host instructions around each measured region.
use std::time::Instant;
use x86jit_core::{
    BlockKey, CachedBlock, CpuMode, Exit, IrBlock, IrOp, Prot, Reg, RegionKind, StepOptions,
    StepStop, Vcpu, Vm, VmConfig,
};

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
    let before = instructions();
    steps(&vm, &mut cpu, count);
    let host_instructions = instructions() - before;
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{name},{count},{:.3},{},{:.1},{host_instructions},{:.3}",
        secs * 1e9 / count as f64,
        vm.cache.misses() - misses,
        (vm.cache.misses() - misses) as f64 / secs,
        host_instructions as f64 / count as f64
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

fn mixed(count: u64) {
    let (vm, mut cpu) = setup(0x800, 0x100, 0x1000);
    cpu.set_reg(Reg::Es, 0x2000);
    cpu.set_reg(Reg::Ss, 0x4000);
    let mut code = vec![
        0xfc, // cld
        0xbe, 0, 0x10, // mov si,0x1000
        0xbf, 0, 0x20, // mov di,0x2000
        0xb9, 32, 0, // mov cx,32
        0xf3, 0xa5, // rep movsw
        0xbb, 0, 0x10, // mov bx,0x1000
        0xb9, 16, 0, // mov cx,16
    ];
    let body = code.len();
    code.extend_from_slice(&[
        0xad, // lodsw
        0x03, 0x07, // add ax,[bx]
        0x30, 0xc4, // xor ah,al
        0xd1, 0xe0, // shl ax,1
        0xd1, 0xea, // shr dx,1
        0x11, 0xc2, // adc dx,ax
        0x25, 0xff, 0x7f, // and ax,0x7fff
        0xab, // stosw
        0x43, // inc bx
        0x50, // push ax
        0xe8, 0, 0,    // call subroutine
        0x58, // pop ax
        0xa8, 1, // test al,1
        0x74, 1,    // jz past inc
        0x42, // inc dx
    ]);
    let call = code.len() - 9;
    code.extend_from_slice(&[0xe2, (body as isize - code.len() as isize - 2) as u8]);
    let back = (-(code.len() as i16 + 3)).to_le_bytes();
    code.extend_from_slice(&[0xe9, back[0], back[1]]);
    let target = (code.len() as i16 - (call + 3) as i16).to_le_bytes();
    assert_eq!(code[call], 0xe8);
    code[call + 1..call + 3].copy_from_slice(&target);
    code.extend_from_slice(&[
        0x53, // push bx
        0x01, 0xd3, // add bx,dx
        0xd1, 0xcb, // ror bx,1
        0x30, 0xf8, // xor al,bh
        0x5b, // pop bx
        0xc3, // ret
    ]);
    vm.write_bytes(0x8100, &code).unwrap();
    vm.write_bytes(0x11000, &[0x5a; 256]).unwrap();
    let run = |cpu: &mut Vcpu, count: u64| {
        let mut done = 0;
        let mut slice = 0;
        while done < count {
            let budget = [1, 7, 64, 256, 1024][slice % 5].min(count - done);
            let result = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
            assert!(matches!(
                result.stop(),
                StepStop::Exit(Exit::BudgetExhausted)
            ));
            assert_eq!(result.steps(), budget);
            done += result.steps();
            slice += 1;
        }
    };
    run(&mut cpu, 1_000_000);
    let misses = vm.cache.misses();
    let start = Instant::now();
    let before = instructions();
    run(&mut cpu, count);
    let host_instructions = instructions() - before;
    let secs = start.elapsed().as_secs_f64();
    println!(
        "dos-mixed,{count},{:.3},{},{:.1},{host_instructions},{:.3}",
        secs * 1e9 / count as f64,
        vm.cache.misses() - misses,
        (vm.cache.misses() - misses) as f64 / secs,
        host_instructions as f64 / count as f64
    );
    println!("dos-mixed-state,{:?}", cpu.cpu);
}
fn main() {
    let scale: u64 = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(1);
    assert_ne!(scale, 0);
    println!(
        "name,steps,ns_per_step,lifts,lifts_per_second,host_instructions,instructions_per_step"
    );
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
    mixed(8_000_000 / scale);
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
        let before = instructions();
        steps(&vm, &mut cpu, per_pass);
        let host_instructions = instructions() - before;
        println!(
            "straight-line-{pass},{per_pass},{:.3},0,0,{host_instructions},{:.3}",
            start.elapsed().as_secs_f64() * 1e9 / per_pass as f64,
            host_instructions as f64 / per_pass as f64
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
            let mut count = 0;
            let before = instructions();
            for _ in 0..64 {
                let r = cpu.run_steps(&vm, limit, StepOptions::default()).unwrap();
                count += r.steps();
                if matches!(r.stop(), StepStop::Exit(Exit::Hlt)) {
                    halted = true;
                    break;
                }
            }
            let host_instructions = instructions() - before;
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
            println!(
                "{name}-{limit},{count},0,0,0,{host_instructions},{:.3}",
                host_instructions as f64 / count as f64
            );
        }
    }
}
