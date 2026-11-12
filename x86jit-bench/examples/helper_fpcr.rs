//! Warmed Long64 JIT helper costs. Arguments: loop iterations, rounds (>=5).
//! Rotate MXCSR modes each round; exclude compilation and verify helper counts.
use iced_x86::code_asm::*;
use std::time::Instant;
use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

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

fn setup(kind: &str, mxcsr: u32) -> (Vm, x86jit_core::Vcpu, u64, &'static str) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(JitBackend::new()));
    if matches!(kind, "xsave" | "masked") {
        vm.set_guest_cpu_features(x86jit_core::GuestCpuFeatures::v4());
    }
    vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    a.set_label(&mut top).unwrap();
    let (steps, helper) = match kind {
        "string" => {
            a.mov(ecx, 16).unwrap();
            a.mov(esi, 0x8000).unwrap();
            a.mov(edi, 0x10000).unwrap();
            a.rep().movsb().unwrap();
            (6, "string_helper::<true>")
        }
        "notify" => {
            // A watched page outside the executing block forces write notification.
            a.mov(qword_ptr(0x18000), rax).unwrap();
            (3, "note_write_helper")
        }
        "pack" => {
            a.packsswb(xmm0, xmm1).unwrap();
            (3, "vpack_helper")
        }
        "masked" => {
            a.vmovdqu32(zmmword_ptr(0x10000).k1(), zmm1).unwrap();
            (3, "vmaskmov_mem_helper")
        }
        "fp" => {
            a.haddps(xmm0, xmm1).unwrap();
            (3, "vhfloat_helper")
        }
        "mixed" => {
            a.mov(ecx, 16).unwrap();
            a.mov(esi, 0x8000).unwrap();
            a.mov(edi, 0x10000).unwrap();
            a.rep().movsb().unwrap();
            a.mov(qword_ptr(0x18000), rax).unwrap();
            a.packsswb(xmm0, xmm1).unwrap();
            a.haddps(xmm0, xmm1).unwrap();
            (9, "string_helper::<true>")
        }
        "xsave" => {
            a.xsave64(ptr(0x10000)).unwrap();
            (3, "xstate_helper")
        }
        _ => unreachable!(),
    };
    a.dec(ebx).unwrap();
    a.jnz(top).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
        .unwrap();
    vm.write_bytes(0x8000, &[0x5a; 16]).unwrap();
    vm.mem.mark_code(0x18000, 8);
    let mut cpu = vm.new_vcpu_with_mxcsr(mxcsr);
    cpu.set_reg(Reg::Rax, if kind == "xsave" { 3 } else { 0x1234 });
    if kind == "xsave" {
        for leaf in 0x4000_0000..0x4000_0020 {
            cpu.cpu.cpuid_overrides.leaves.insert(
                (leaf, 0),
                x86jit_core::CpuidOverride {
                    mask: x86jit_core::CpuidRegisters::ZERO,
                    value: x86jit_core::CpuidRegisters::ZERO,
                },
            );
        }
    }
    cpu.set_xmm(1, 0x3f800000_3f800000_3f800000_3f800000);
    if kind == "masked" {
        cpu.cpu.kmask[1] = 0xa5a5;
        cpu.cpu.ymm_hi[1] = cpu.xmm(1);
        cpu.cpu.zmm_hi[1] = [cpu.xmm(1); 2];
    }
    (vm, cpu, steps, helper)
}
fn run(vm: &Vm, cpu: &mut x86jit_core::Vcpu, n: u64) {
    cpu.set_reg(Reg::Rip, 0x1000);
    cpu.set_reg(Reg::Rbx, n);
    cpu.set_xmm(0, 0);
    assert!(matches!(cpu.run(vm, None), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rbx), 0);
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn main() {
    if cfg!(debug_assertions) || !cfg!(target_arch = "aarch64") {
        eprintln!("run this benchmark with --release on aarch64");
        std::process::exit(1);
    }
    let mut args = std::env::args().skip(1);
    let n: u64 = args.next().map(|s| s.parse().unwrap()).unwrap_or(500_000);
    let rounds: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(7);
    assert!(n > 0 && rounds >= 5);
    println!("kind,mxcsr,guest_steps,ns_per_step,instructions_per_step");
    #[cfg(target_arch = "aarch64")]
    control_cost(rounds);
    for kind in ["string", "notify", "pack", "masked", "fp", "mixed", "xsave"] {
        let mut cases = [0x1f80, 0x9fc0].map(|m| setup(kind, m));
        for (vm, cpu, _, _) in &mut cases {
            run(vm, cpu, 1000);
        }
        let mut samples: [Vec<(f64, f64)>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..rounds {
            for offset in 0..2 {
                let i = (round + offset) % 2;
                let (vm, cpu, steps, helper) = &mut cases[i];
                let count = || {
                    vm.backend
                        .helper_calls()
                        .iter()
                        .find(|(name, _)| name == helper)
                        .map(|(_, n)| *n)
                        .unwrap_or(0)
                };
                let before_calls = count();
                let before = instructions();
                let start = Instant::now();
                run(vm, cpu, n);
                let ns = start.elapsed().as_nanos() as f64;
                let inst = instructions() - before;
                assert_eq!(
                    count() - before_calls,
                    n,
                    "{kind}: representative helper must execute every iteration"
                );
                if kind == "fp" && (2..=4_000_000).contains(&n) {
                    let low = (((n - 2) * 4) as f32).to_bits();
                    assert_eq!(
                        cpu.xmm(0),
                        0x40000000_40000000_40800000_00000000 | u128::from(low)
                    );
                }
                if kind == "string" || kind == "mixed" {
                    let mut dest = [0; 16];
                    vm.read_bytes(0x10000, &mut dest).unwrap();
                    assert_eq!(dest, [0x5a; 16]);
                }
                if kind == "masked" {
                    let mut dest = [0; 64];
                    vm.read_bytes(0x10000, &mut dest).unwrap();
                    for (lane, bytes) in dest.as_chunks::<4>().0.iter().enumerate() {
                        let expected = if 0xa5a5 & (1 << lane) != 0 {
                            0x3f800000
                        } else {
                            0
                        };
                        assert_eq!(u32::from_le_bytes(*bytes), expected);
                    }
                }
                samples[i].push((
                    ns / (n * *steps + 1) as f64,
                    inst as f64 / (n * *steps + 1) as f64,
                ));
            }
        }
        for (i, mode) in [0x1f80, 0x9fc0].into_iter().enumerate() {
            let ns = median(&mut samples[i].iter().map(|s| s.0).collect::<Vec<_>>());
            let inst = median(&mut samples[i].iter().map(|s| s.1).collect::<Vec<_>>());
            println!("{kind},{mode:04x},{},{ns:.3},{inst:.3}", n * cases[i].2 + 1);
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn control_cost(rounds: usize) {
    #[unsafe(naked)]
    unsafe extern "C" fn always_write(_value: u64) {
        core::arch::naked_asm!("msr fpcr, x0", "ret");
    }
    #[unsafe(naked)]
    unsafe extern "C" fn read_only(_value: u64) {
        core::arch::naked_asm!("mrs x9, fpcr", "ret");
    }
    #[unsafe(naked)]
    unsafe extern "C" fn empty(_value: u64) {
        core::arch::naked_asm!("ret");
    }
    use x86jit_core::jit_abi::fp::set_control;
    let saved: u64;
    // SAFETY: the benchmark masks all traps and restores FPCR before reporting.
    unsafe {
        core::arch::asm!("mrs {}, fpcr", out(reg) saved, options(nomem, nostack));
    }
    for (name, entry, toggle) in [
        ("empty", empty as unsafe extern "C" fn(u64), false),
        ("mrs", read_only as unsafe extern "C" fn(u64), false),
        ("msr", always_write as unsafe extern "C" fn(u64), false),
        (
            "msr-change",
            always_write as unsafe extern "C" fn(u64),
            true,
        ),
        ("set-same", set_control as unsafe extern "C" fn(u64), false),
        ("set-change", set_control as unsafe extern "C" fn(u64), true),
    ] {
        let mut ns = Vec::new();
        let mut inst = Vec::new();
        const N: u64 = 2_000_000;
        for _ in 0..rounds {
            unsafe {
                set_control(0);
            }
            let before = instructions();
            let start = Instant::now();
            for i in 0..N {
                // SAFETY: the loop uses only integer operations under guest control.
                unsafe {
                    std::hint::black_box(entry)(if toggle { (i & 1) << 22 } else { 0 });
                }
            }
            unsafe {
                set_control(0);
            }
            ns.push(start.elapsed().as_nanos() as f64 / N as f64);
            inst.push((instructions() - before) as f64 / N as f64);
        }
        unsafe {
            set_control(saved);
        }
        println!(
            "control-{name},host,{N},{:.3},{:.3}",
            median(&mut ns),
            median(&mut inst)
        );
    }
}
