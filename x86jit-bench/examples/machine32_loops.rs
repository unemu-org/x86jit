//! Warm, finite five-instruction loops for the compiled-Machine J0 decision.
//! Arguments: iterations (default 2M), rounds (default 5), optional path/workload.
//! MACHINE32_MODEL=486 selects I80486Dx; the default is I80386.
//! Application retirement is derived from the validated finite guest loop.
use std::sync::Arc;
use std::time::Instant;
use x86jit_core::{
    BlockKey, CachedBlock, CpuMode, CpuModel, CpuState, ExceptionDelivery, Exit, Prot, Reg,
    RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::{JitBackend, OptLevel};

#[cfg(target_os = "macos")]
fn instructions() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::uninit();
    // SAFETY: V4 writes the matching structure, read only after success.
    unsafe {
        assert_eq!(
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                usage.as_mut_ptr().cast()
            ),
            0
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

const PATHS: [&str; 5] = [
    "machine-real16",
    "machine-protected32",
    "machine-paged32",
    "application-interp32",
    "application-jit32",
];
const WORKS: [&str; 3] = ["register", "memory", "callret"];
const CODE: u64 = 0x10000;
const DATA: u64 = 0x4000;
const STACK: u64 = 0x8000;

fn code(work: &str, real: bool) -> Vec<u8> {
    match work {
        // INC A; ADD B,A; XOR D,B; DEC DI; JNZ start.
        "register" => vec![0x40, 0x01, 0xc3, 0x31, 0xda, 0x4f, 0x75, 0xf8, 0xf4],
        // Same effects, four redundant segment prefixes: isolates fetch-length cost.
        "register-pad" => vec![
            0x2e, 0x40, 0x2e, 0x01, 0xc3, 0x2e, 0x31, 0xda, 0x2e, 0x4f, 0x75, 0xf4, 0xf4,
        ],
        // MOV A,[SI]; ADD A,1; MOV [SI],A; DEC DI; JNZ start.
        "memory" => vec![
            0x8b,
            if real { 0x04 } else { 0x06 },
            0x83,
            0xc0,
            0x01,
            0x89,
            if real { 0x04 } else { 0x06 },
            0x4f,
            0x75,
            0xf6,
            0xf4,
        ],
        // CALL sub; DEC DI; JNZ start; HLT; sub: INC A; RET.
        "callret" if real => vec![0xe8, 4, 0, 0x4f, 0x75, 0xfa, 0xf4, 0x40, 0xc3],
        "callret" => vec![0xe8, 4, 0, 0, 0, 0x4f, 0x75, 0xf8, 0xf4, 0x40, 0xc3],
        _ => panic!("unknown workload"),
    }
}

fn setup(path: &str, work: &str) -> (Vm, x86jit_core::Vcpu) {
    let mut vm = if path == "application-jit32" {
        Vm::with_backend(
            VmConfig::flat(0x20000),
            Box::new(JitBackend::with_opt_level(OptLevel::Speed)),
        )
    } else {
        Vm::new(VmConfig::flat(0x20000))
    };
    vm.set_cpu_mode(CpuMode::Compat32);
    let machine = path.starts_with("machine");
    let real = path == "machine-real16";
    if machine {
        vm.set_cpu_model(match std::env::var("MACHINE32_MODEL").as_deref() {
            Ok("486") => CpuModel::I80486Dx,
            _ => CpuModel::I80386,
        });
    }
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(CODE, &code(work, real)).unwrap();
    if path == "application-single32" {
        let bytes = code(work, false);
        let mut decoder =
            iced_x86::Decoder::with_ip(32, &bytes, CODE, iced_x86::DecoderOptions::NONE);
        while decoder.can_decode() {
            let addr = decoder.decode().ip();
            let ir = x86jit_core::lift::lift_one(
                &vm.mem,
                x86jit_core::lift::FetchAddr { pa: addr, ip: addr },
                CpuMode::Compat32,
            )
            .unwrap();
            let len = ir.guest_len;
            vm.cache.insert(
                BlockKey::new(addr, CpuMode::Compat32),
                CachedBlock::Interpreted(Arc::new(ir)),
                vec![(addr, len)],
                |_| vm.mem.mark_code(addr, len),
            );
        }
    }
    if path == "machine-paged32" {
        vm.write_bytes(0x1000, &0x2003u32.to_le_bytes()).unwrap();
        for page in 0..32u32 {
            vm.write_bytes(
                0x2000 + u64::from(4 * page),
                &((page * 4096) | 3).to_le_bytes(),
            )
            .unwrap();
        }
    }
    let mut cpu = vm.new_vcpu();
    if real {
        cpu.cpu = CpuState::legacy_real_entry([0x1000, 0, 0, 0, 0, 0]);
    } else if machine {
        // Same retained-cache fixture as legacy_paging.rs, with flat bases.
        cpu.cpu.legacy.cr0 = if path == "machine-paged32" {
            0x80000001
        } else {
            1
        };
        cpu.cpu.legacy.cr3 = 0x1000;
        cpu.cpu.legacy.cpl = 0;
        for (i, segment) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
            segment.base = 0;
            segment.limit = u32::MAX;
            segment.attributes = (if i == 0 { 0x9b } else { 0x93 }) | 0x4000;
            segment.unusable = 0;
        }
    }
    cpu.set_a20_enabled(true);
    (vm, cpu)
}

fn execute(vm: &Vm, cpu: &mut x86jit_core::Vcpu, real: bool, n: u64) {
    // Real16 loops use DI, so split into runs without adding guest setup ops.
    let mut remaining = n;
    while remaining != 0 {
        let count = if real {
            remaining.min(60000)
        } else {
            remaining
        };
        cpu.set_reg(Reg::Rip, if real { 0 } else { CODE });
        cpu.set_reg(Reg::Rdi, count);
        cpu.set_reg(Reg::Rsi, DATA);
        cpu.set_reg(Reg::Rsp, STACK);
        assert!(matches!(cpu.run(vm, None), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rdi), 0);
        assert_eq!(cpu.reg(Reg::Rsp), STACK);
        remaining -= count;
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n = args.get(1).map_or(2_000_000, |s| s.parse::<u64>().unwrap());
    let rounds = args.get(2).map_or(5, |s| s.parse::<usize>().unwrap());
    assert!(n > 0 && n <= u32::MAX as u64);
    if args.get(3).is_some_and(|s| s == "components") {
        components(n, rounds);
        return;
    }
    println!(
        "workload,path,round,iterations,instructions,retired_counter,seconds,mips,host_per_step"
    );
    for round in 0..rounds {
        let works: Vec<&str> = args
            .get(4)
            .map_or_else(|| WORKS.to_vec(), |s| vec![s.as_str()]);
        for work in works {
            let paths: Vec<&str> = args.get(3).map_or_else(
                || {
                    (0..PATHS.len())
                        .map(|offset| PATHS[(round + offset) % PATHS.len()])
                        .collect()
                },
                |s| vec![s.as_str()],
            );
            for path in paths {
                let real = path == "machine-real16";
                let (vm, mut cpu) = setup(path, work);
                execute(&vm, &mut cpu, real, 10000);
                cpu.set_reg(Reg::Rax, 0);
                cpu.set_reg(Reg::Rbx, 0);
                cpu.set_reg(Reg::Rdx, 0);
                vm.write_bytes(DATA, &[0; 4]).unwrap();
                let retired_before = cpu.retired_instructions();
                let host_before = instructions();
                let start = Instant::now();
                execute(&vm, &mut cpu, real, n);
                let seconds = start.elapsed().as_secs_f64();
                let host = instructions() - host_before;
                let retired = cpu.retired_instructions() - retired_before;
                let count = 5 * n + if real { n.div_ceil(60000) } else { 1 };
                if path.starts_with("machine") {
                    assert_eq!(retired, count);
                }
                let mask = if real { 0xffff } else { 0xffff_ffff };
                assert_eq!(cpu.reg(Reg::Rax) & mask, n & mask);
                if work == "memory" {
                    let mut data = [0; 4];
                    vm.read_bytes(DATA, &mut data).unwrap();
                    assert_eq!(u64::from(u32::from_le_bytes(data)) & mask, n & mask);
                }
                println!(
                    "{work},{path},{round},{n},{count},{retired},{seconds:.9},{:.3},{:.3}",
                    count as f64 / seconds / 1e6,
                    host as f64 / count as f64
                );
            }
        }
    }
}

// Isolate the public decoder work repeated by Checked fetch. This excludes
// segment checks, bus reads, source proofs, dispatch and the IR interpreter.
fn components(n: u64, rounds: usize) {
    use iced_x86::{Decoder, DecoderOptions as D};
    let options = D::NO_PAUSE | D::NO_WBNOINVD | D::NO_MPFX_0FBC | D::NO_MPFX_0FBD | D::MOV_TR;
    println!(
        "workload,path,round,iterations,instructions,retired_counter,seconds,mips,host_per_step"
    );
    for round in 0..rounds {
        for work in ["register", "register-pad"] {
            let bytes = code(work, false);
            let mut decoder = Decoder::with_ip(32, &bytes, CODE, options);
            let lengths: Vec<_> = (0..5).map(|_| decoder.decode().len()).collect();
            let host = instructions();
            let start = Instant::now();
            for _ in 0..n {
                let mut offset = 0;
                for &len in &lengths {
                    for prefix in 1..=len {
                        let slice = std::hint::black_box(&bytes[offset..offset + prefix]);
                        let mut decoder =
                            Decoder::with_ip(32, slice, CODE + offset as u64, options);
                        std::hint::black_box(decoder.decode());
                    }
                    offset += len;
                }
            }
            let seconds = start.elapsed().as_secs_f64();
            let host = instructions() - host;
            let count = 5 * n;
            println!(
                "{work},component-decode,{round},{n},{count},0,{seconds:.9},{:.3},{:.3}",
                count as f64 / seconds / 1e6,
                host as f64 / count as f64
            );
        }
    }
}
