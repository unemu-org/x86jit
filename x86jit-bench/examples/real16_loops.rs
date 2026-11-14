//! Real16 embedder loop benchmarks using only the core API. Release mode;
//! warmup excluded. Arguments: step-count divisor (default 1), rounds (default 5).
//! Each round rotates the seven paths; every sample starts from a fresh VM.
//! macOS reports proc_pid_rusage host instructions around each measured region.
use std::time::Instant;
use x86jit_core::{
    BlockKey, CachedBlock, CpuMode, CpuModel, CpuState, ExecutionProfile, Exit, IrBlock, IrOp,
    Prot, Reg, RegionKind, StepOptions, StepStop, Vcpu, Vm, VmConfig,
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

#[derive(Clone, Copy, Debug)]
enum Path {
    Application,
    I8088,
    I8086,
    I286,
    I386,
    I286A20Off,
    I386A20Off,
}

const PATHS: [Path; 7] = [
    Path::Application,
    Path::I8088,
    Path::I8086,
    Path::I286,
    Path::I386,
    Path::I286A20Off,
    Path::I386A20Off,
];

impl Path {
    fn name(self) -> &'static str {
        match self {
            Self::Application => "application-real16",
            Self::I8088 => "machine-i8088",
            Self::I8086 => "machine-i8086",
            Self::I286 => "legacy-i286",
            Self::I386 => "legacy-i386",
            Self::I286A20Off => "legacy-i286-a20-off",
            Self::I386A20Off => "legacy-i386-a20-off",
        }
    }
}

fn setup(path: Path, cs: u64, ip: u64, ds: u64) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x10_0000));
    vm.set_cpu_mode(CpuMode::Real16);
    if !matches!(path, Path::Application) {
        let model = if matches!(path, Path::I8088) {
            CpuModel::I8088
        } else if matches!(path, Path::I8086) {
            CpuModel::I8086
        } else if matches!(path, Path::I386 | Path::I386A20Off) {
            CpuModel::I80386
        } else {
            CpuModel::I80286
        };
        vm.set_cpu_model(model);
        assert_eq!(vm.execution_profile(), ExecutionProfile::Machine);
    } else {
        assert_eq!(
            vm.execution_profile(),
            ExecutionProfile::Application(CpuMode::Real16)
        );
    }
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
fn steps(vm: &Vm, cpu: &mut Vcpu, steps: u64, slices: &[u64]) -> u64 {
    let mut done = 0;
    let mut slice = 0;
    while done < steps {
        let budget = if slices.is_empty() {
            steps - done
        } else {
            slices[slice % slices.len()].min(steps - done)
        };
        let r = cpu.run_steps(vm, budget, StepOptions::default()).unwrap();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
        assert_eq!(r.steps(), budget);
        done += r.steps();
        slice += 1;
    }
    done
}

struct Workload {
    name: &'static str,
    cs: u64,
    ip: u64,
    ds: u64,
    regs: Vec<(Reg, u64)>,
    writes: Vec<(u64, Vec<u8>)>,
    count: u64,
    warmup: u64,
    fillers: u64,
    slices: &'static [u64],
}

impl Workload {
    fn new(name: &'static str, code: Vec<u8>, count: u64, ds: u64) -> Self {
        Self {
            name,
            cs: 0,
            ip: 0x7c00,
            ds,
            regs: vec![],
            writes: vec![(0x7c00, code)],
            count,
            warmup: 1_000_000,
            fillers: 0,
            slices: &[],
        }
    }

    fn setup(&self, path: Path) -> (Vm, Vcpu) {
        let (vm, mut cpu) = setup(path, self.cs, self.ip, self.ds);
        for &(reg, value) in &self.regs {
            cpu.set_reg(reg, value);
        }
        if !matches!(path, Path::Application) {
            // set_reg changes visible selectors only; ordinary legacy entry
            // must also seed their hidden caches through the core helper.
            let c = &cpu.cpu;
            cpu.cpu.legacy =
                CpuState::legacy_real_entry([c.cs, c.ds, c.es, c.ss, c.fs, c.gs]).legacy;
        }
        cpu.set_a20_enabled(!matches!(path, Path::I286A20Off | Path::I386A20Off));
        for (address, bytes) in &self.writes {
            assert!(*address + bytes.len() as u64 <= 0x10_0000);
            vm.write_bytes(*address, bytes).unwrap();
        }
        // This intentionally fills the Application cache, as in the original
        // dense-code probe. Checked Machine contexts use a per-vCPU block cache.
        for i in 0..self.fillers {
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
        (vm, cpu)
    }
}

fn assert_registers_equal(expected: &CpuState, actual: &CpuState) {
    assert_eq!(actual.gpr, expected.gpr);
    assert_eq!(actual.rip, expected.rip);
    assert_eq!(actual.flags, expected.flags);
    assert_eq!(actual.rflags_system, expected.rflags_system);
    assert_eq!(
        [actual.cs, actual.ds, actual.es, actual.ss, actual.fs, actual.gs],
        [
            expected.cs,
            expected.ds,
            expected.es,
            expected.ss,
            expected.fs,
            expected.gs
        ]
    );
    assert_eq!(
        (actual.fs_base, actual.gs_base),
        (expected.fs_base, expected.gs_base)
    );
    assert_eq!(actual.xmm, expected.xmm);
    assert_eq!(actual.ymm_hi, expected.ymm_hi);
    assert_eq!(actual.zmm_hi, expected.zmm_hi);
    assert_eq!(actual.kmask, expected.kmask);
    assert_eq!(actual.mxcsr, expected.mxcsr);
    assert_eq!(actual.fpr, expected.fpr);
    assert_eq!(
        (
            actual.fpu_cw,
            actual.fpu_sw,
            actual.fpu_top,
            actual.fpu_empty,
            actual.fpu_fip,
            actual.fpu_fdp,
            actual.fpu_fop,
            actual.fpu_fcs,
            actual.fpu_fds
        ),
        (
            expected.fpu_cw,
            expected.fpu_sw,
            expected.fpu_top,
            expected.fpu_empty,
            expected.fpu_fip,
            expected.fpu_fdp,
            expected.fpu_fop,
            expected.fpu_fcs,
            expected.fpu_fds
        )
    );
}

fn summary(mut values: Vec<f64>) -> (f64, f64, f64) {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    let median = if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    };
    (median, values[0], values[values.len() - 1])
}

fn bench(work: &Workload, scale: u64, rounds: usize) {
    let count = (work.count / scale).max(1);
    let warmup = (work.warmup / scale).max(1);
    let mut samples: [Vec<(f64, f64)>; 7] = std::array::from_fn(|_| Vec::new());
    let mut expected: Option<CpuState> = None;
    let mut expected_counts = None;
    for round in 0..rounds {
        #[cfg(target_os = "macos")]
        {
            let mut load = [0.0; 3];
            // SAFETY: getloadavg writes at most three doubles into this array.
            assert_eq!(unsafe { libc::getloadavg(load.as_mut_ptr(), 3) }, 3);
            eprintln!("load,{},{round},{load:.2?}", work.name);
        }
        for offset in 0..PATHS.len() {
            let index = (round + offset) % PATHS.len();
            let path = PATHS[index];
            let (vm, mut cpu) = work.setup(path);
            let warmed = steps(&vm, &mut cpu, warmup, work.slices);
            let warmup_retired = cpu.retired_instructions();
            let before = instructions();
            let start = Instant::now();
            let guest_steps = steps(&vm, &mut cpu, count, work.slices);
            let elapsed = start.elapsed();
            let host_instructions = instructions() - before;
            let retired = cpu.retired_instructions() - warmup_retired;
            let ns = elapsed.as_secs_f64() * 1e9 / guest_steps as f64;
            let inst = host_instructions as f64 / guest_steps as f64;
            eprintln!(
                "sample,{},{},{round},{guest_steps},{retired},{warmup},{ns:.3},{inst:.3}",
                work.name,
                path.name()
            );
            let counts = (warmed, guest_steps, warmup_retired, retired);
            if let Some(state) = &expected {
                assert_registers_equal(state, &cpu.cpu);
                assert_eq!(Some(counts), expected_counts);
            } else {
                expected = Some(cpu.cpu.clone());
                expected_counts = Some(counts);
            }
            samples[index].push((ns, inst));
        }
    }
    for (path, samples) in PATHS.into_iter().zip(samples) {
        let (ns, ns_min, ns_max) = summary(samples.iter().map(|s| s.0).collect());
        let (inst, inst_min, inst_max) = summary(samples.iter().map(|s| s.1).collect());
        let retired = expected_counts.unwrap().3;
        println!(
            "{},{},{count},{retired},{warmup},{rounds},{inst:.3},{inst_min:.3},{inst_max:.3},{ns:.3},{ns_min:.3},{ns_max:.3}",
            work.name, path.name()
        );
    }
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

fn mixed() -> Workload {
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
    Workload {
        cs: 0x800,
        ip: 0x100,
        regs: vec![(Reg::Es, 0x2000), (Reg::Ss, 0x4000)],
        writes: vec![(0x8100, code), (0x11000, vec![0x5a; 256])],
        slices: &[1, 7, 64, 256, 1024],
        ..Workload::new("dos-mixed", vec![], 8_000_000, 0x1000)
    }
}
fn main() {
    if cfg!(debug_assertions) {
        eprintln!("run this benchmark with --release");
        std::process::exit(1);
    }
    let mut args = std::env::args().skip(1);
    let scale: u64 = args.next().map(|s| s.parse().unwrap()).unwrap_or(1);
    let rounds: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(5);
    let workload = args.next();
    assert!(
        args.next().is_none(),
        "usage: real16_loops [divisor] [rounds] [workload]"
    );
    assert_ne!(scale, 0);
    assert!(rounds >= 3, "at least three serial rounds are required");
    if !cfg!(target_os = "macos") {
        eprintln!("host instruction counter unavailable: instruction columns are zero");
    }
    println!(
        "name,path,steps,retired_instructions,warmup_steps,rounds,instructions_per_step,instructions_min,instructions_max,ns_per_step,ns_min,ns_max"
    );
    let mut workloads = Vec::new();
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
        workloads.push(Workload::new(name, looped(&body), count, 0x1000));
    }
    for (name, fillers) in [("data-next-to-code", 0), ("data-next-to-dense-code", 1024)] {
        let mut work = Workload::new(
            name,
            looped(&[0x89, 0x06, 0x00, 0x78, 0x01, 0xd8]),
            20_000_000,
            0,
        );
        work.fillers = fillers;
        workloads.push(work);
    }
    workloads.push(Workload::new(
        "call-ret",
        vec![0xb9, 0, 0, 0xe8, 4, 0, 0xe2, 0xfb, 0xeb, 0xf6, 0xc3],
        20_000_000,
        0x1000,
    ));
    workloads.push(mixed());
    workloads.push(Workload::new(
        "rep-movsw",
        vec![
            0x31, 0xf6, 0x31, 0xff, 0xb9, 0, 0x40, 0xf3, 0xa5, 0xeb, 0xf5,
        ],
        100_000_000,
        0x1000,
    ));
    let pat = [0x01, 0xd8, 0x89, 0x40, 0x10, 0x46];
    let mut code = Vec::new();
    while code.len() + pat.len() <= 0xef00 {
        code.extend_from_slice(&pat);
    }
    let disp = (-(code.len() as i32 + 3)) as i16 as u16;
    let per_pass = (code.len() / pat.len() * 3 + 1) as u64;
    code.extend_from_slice(&[0xe9, disp as u8, (disp >> 8) as u8]);
    // Keep the original four passes: one warm and three measured, independent
    // of the divisor. Longer runs let SI reach a segment-crossing word access.
    let straight_count = per_pass.checked_mul(scale).unwrap();
    workloads.push(Workload {
        cs: 0x1000,
        ip: 0x100,
        writes: vec![(0x10100, code)],
        warmup: straight_count,
        ..Workload::new(
            "straight-line",
            vec![],
            straight_count.checked_mul(3).unwrap(),
            0x3000,
        )
    });
    for (name, slices) in [("ip-boundary-1", &[1][..]), ("ip-boundary-64", &[64][..])] {
        // Branch before the segment end instead of relying on sequential IP
        // wrap, which is not common behaviour across the execution paths.
        workloads.push(Workload {
            cs: 0x1000,
            ip: 0xfff8,
            writes: vec![
                (0x1fff8, vec![0x90, 0x90, 0x42, 0xe9, 0x02, 0x01]),
                (0x10100, vec![0xe9, 0xf5, 0xfe]),
            ],
            slices,
            ..Workload::new(name, vec![], 1_000_000, 0x1000)
        });
    }
    for (name, slices) in [("smc-next-1", &[1][..]), ("smc-next-64", &[64][..])] {
        let mut work = Workload::new(
            name,
            vec![0xc6, 6, 6, 0x7c, 0x42, 0xb0, 0, 0xeb, 0xf7],
            1_000_000,
            0,
        );
        work.slices = slices;
        workloads.push(work);
    }
    workloads.push(Workload {
        slices: &[64],
        ..Workload::new(
            "smc-rmw-64",
            vec![0x80, 6, 6, 0x7c, 1, 0xb0, 0, 0xeb, 0xf7],
            1_000_000,
            0,
        )
    });
    assert!(
        workload
            .as_ref()
            .is_none_or(|name| workloads.iter().any(|work| work.name == name)),
        "unknown workload"
    );
    for work in workloads {
        if workload.as_ref().is_none_or(|name| work.name == name) {
            bench(&work, scale, rounds);
        }
    }
}
