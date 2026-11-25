//! Exit/completion and borrowed-handler transfers (spec.md §5.4, decision-28).
//! Usage: machine_exits [iterations=20000] [rounds=3] [mode] [workload].
//! EXIT_MODEL=486 or 286 selects I80486Dx or I80286 (default I80386; 286 runs
//! the sixteen-bit modes only); EXIT_SPLITS=1 times
//! individual run/completion calls separately; EXIT_TRACE=1 emits exact traces.
//! EXIT_COMPLETION_BUDGET=10000000 demonstrates continuing with the existing
//! run_steps budget after MMIO completion (default 1, matching a one-step shim).
//! EXIT_LOG=1 enables the delivery-log epilogue (off by default).
//! EXIT_PATH=exit|handler|exit-noreg runs only that path (for host profiling).
//! Timings include guest execution: ns/exit is amortized round-trip cost.
//! Workloads: out/in/store/load at N=10/25/100; `planar`, a handled byte
//! MMIO store every N=3/4/5 instructions; `unserved-out`/`unserved-in`, port
//! accesses outside the one registered port range, so every one takes the Exit.
//! Unserved cells add an `exit-noreg` path, the same loop on a VM with no
//! registered range, to compare the registry's cost against none.
use std::time::{Duration, Instant};
use x86jit_core::{
    CpuModel, CpuState, DeviceContext, DeviceIo, DeviceRange, DeviceRequest, DeviceResult,
    ExceptionDelivery, Exit, PortDir, Prot, Reg, RegionKind, StepOptions, StepStop, Vcpu, Vm,
    VmConfig,
};

#[cfg(target_os = "macos")]
fn host_instructions() -> u64 {
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
        usage.assume_init().ri_instructions
    }
}
#[cfg(not(target_os = "macos"))]
fn host_instructions() -> u64 {
    0
}

const CODE: u64 = 0x10000;
const DEVICE: u64 = 0xa000;
const MODES: [&str; 4] = ["real16", "flat32", "paged32", "seg16"];
const WORKS: [&str; 7] = [
    "out",
    "in",
    "store",
    "load",
    "planar",
    "unserved-out",
    "unserved-in",
];

fn setup(mode: &str, work: &str, interval: usize, register: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x20000));
    vm.set_cpu_model(match std::env::var("EXIT_MODEL").as_deref() {
        Ok("486") => CpuModel::I80486Dx,
        Ok("286") => CpuModel::I80286,
        Ok("8088") => CpuModel::I8088,
        Ok("8086") => CpuModel::I8086,
        Ok("v20") => CpuModel::NecV20,
        Ok("v30") => CpuModel::NecV30,
        _ => CpuModel::I80386,
    });
    vm.set_exception_delivery(if work == "delivery" {
        ExceptionDelivery::InGuest
    } else {
        ExceptionDelivery::Outward
    });
    vm.map(0, DEVICE as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(DEVICE, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.map(DEVICE + 0x1000, 0x15000, Prot::RWX, RegionKind::Ram)
        .unwrap();
    let unserved = work.starts_with("unserved");
    if register {
        // An unserved loop's port 0x80 lies outside the one registered range.
        vm.register_device_range(DeviceRange::Ports(if unserved {
            0x90..0x91
        } else {
            0x80..0x81
        }))
        .unwrap();
        if !unserved {
            vm.register_device_range(DeviceRange::Mmio(DEVICE..DEVICE + 0x1000))
                .unwrap();
        }
    }
    let short = mode == "real16" || mode == "seg16";
    // INC BX/EBX fillers; one device instruction, DEC DI/EDI, JNZ start.
    let mut code = vec![0x43; interval - 3];
    match work {
        "out" | "delivery" | "unserved-out" => code.extend([0xe6, 0x80]),
        "in" | "unserved-in" => code.extend([0xe4, 0x80]),
        "store" | "planar" => code.extend([0x88, if short { 0x04 } else { 0x06 }]),
        "load" => code.extend([0x8a, if short { 0x04 } else { 0x06 }]),
        _ => unreachable!(),
    }
    code.extend([0x4f, 0x75]);
    code.push((-(code.len() as i16 + 1)) as u8);
    code.push(0xf4);
    vm.write_bytes(CODE, &code).unwrap();
    if mode == "paged32" {
        vm.write_bytes(0x1000, &0x2003u32.to_le_bytes()).unwrap();
        for page in 0..32u32 {
            vm.write_bytes(
                0x2000 + u64::from(page * 4),
                &((page * 4096) | 3).to_le_bytes(),
            )
            .unwrap();
        }
    }
    let mut cpu = vm.new_vcpu();
    if std::env::var_os("EXIT_LOG").is_some() {
        cpu.enable_delivery_log(64).unwrap();
    }
    if mode == "real16" {
        cpu.cpu = CpuState::legacy_real_entry([0x1000, 0, 0, 0, 0, 0]);
    } else {
        cpu.cpu.legacy.cr0 = if mode == "paged32" { 0x8000_0001 } else { 1 };
        cpu.cpu.legacy.cr3 = 0x1000;
        for (i, segment) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
            segment.base = if mode == "seg16" && i == 0 {
                CODE as u32
            } else {
                0
            };
            segment.limit = if short { 0xffff } else { u32::MAX };
            segment.attributes =
                (if i == 0 { 0x9b } else { 0x93 }) | if short { 0 } else { 0x4000 };
            segment.unusable = 0;
        }
    }
    cpu.set_reg(Reg::Rsi, DEVICE);
    cpu.set_reg(Reg::Rsp, 0x8000);
    cpu.set_reg(Reg::Rax, 0x5a);
    (vm, cpu)
}

#[derive(Default)]
struct Counts {
    exits: u64,
    calls: u64,
    completion_calls: u64,
    run_time: Duration,
    completion_time: Duration,
}

#[derive(Default)]
struct BenchDevice {
    transfers: u64,
}
impl DeviceIo for BenchDevice {
    fn access(&mut self, _: &DeviceRequest, _: &mut DeviceContext<'_>) -> DeviceResult {
        self.transfers += 1;
        DeviceResult::Handled(0xa5)
    }
}

#[allow(clippy::too_many_arguments)]
fn execute(
    vm: &Vm,
    cpu: &mut Vcpu,
    mode: &str,
    iterations: u64,
    splits: bool,
    trace: bool,
    completion_budget: u64,
    handler: bool,
) -> Counts {
    let mut counts = Counts::default();
    let mut device = BenchDevice::default();
    let mut left = iterations;
    while left != 0 {
        let batch = left.min(60000);
        left -= batch;
        cpu.cpu.rip = if mode == "real16" || mode == "seg16" {
            0
        } else {
            CODE
        };
        cpu.set_reg(Reg::Rdi, batch);
        let mut completing = false;
        loop {
            let start = splits.then(Instant::now);
            let budget = if completing {
                completion_budget
            } else {
                10_000_000
            };
            let result = if handler {
                cpu.run_steps_with_devices(vm, budget, StepOptions::default(), Some(&mut device))
                    .map(|run| run.into_run())
            } else {
                cpu.run_steps(vm, budget, StepOptions::default())
            }
            .unwrap();
            if let Some(start) = start {
                if completing {
                    counts.completion_time += start.elapsed();
                } else {
                    counts.run_time += start.elapsed();
                }
            }
            counts.calls += 1;
            counts.completion_calls += u64::from(completing);
            if trace {
                println!(
                    "{mode}: {result:?} retired={} linear={:?} cpu={:?}",
                    cpu.retired_instructions(),
                    cpu.pending_mmio_linear_address(),
                    cpu.cpu
                );
            }
            completing = false;
            match result.stop() {
                StepStop::Exit(Exit::PortIo {
                    dir: PortDir::In, ..
                }) => {
                    cpu.complete_port_in(0xa5);
                    counts.exits += 1;
                }
                StepStop::Exit(Exit::PortIo { .. }) => counts.exits += 1,
                StepStop::Exit(Exit::MmioRead { .. }) => {
                    cpu.complete_mmio_read(0xa5);
                    counts.exits += 1;
                    completing = true;
                }
                StepStop::Exit(Exit::MmioWrite { .. }) => {
                    cpu.complete_mmio_write();
                    counts.exits += 1;
                    completing = true;
                }
                StepStop::Exit(Exit::BudgetExhausted) => {}
                StepStop::Exit(Exit::Hlt) => break,
                stop => panic!("unexpected stop: {stop:?}"),
            }
        }
    }
    counts.exits += device.transfers;
    assert_eq!(counts.exits, iterations);
    counts
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let iterations = args.get(1).map_or(20000, |s| s.parse::<u64>().unwrap());
    let rounds = args.get(2).map_or(3, |s| s.parse::<usize>().unwrap());
    let splits = std::env::var_os("EXIT_SPLITS").is_some();
    let trace = std::env::var_os("EXIT_TRACE").is_some();
    let completion_budget =
        std::env::var("EXIT_COMPLETION_BUDGET").map_or(1, |s| s.parse().unwrap());
    assert!(completion_budget > 0);
    if std::env::var_os("EXIT_CENSUS").is_some() {
        census();
        return;
    }
    println!("path,mode,work,n,round,retired,exits,calls,completion_calls,mips,ns_exit,run_ns_exit,completion_ns_exit,host_instructions_exit");
    let i286 = std::env::var("EXIT_MODEL").as_deref() == Ok("286");
    for round in 0..rounds {
        for mode in MODES {
            if args.get(3).is_some_and(|s| s != mode) || (i286 && mode.contains("32")) {
                continue;
            }
            for work in WORKS {
                if args.get(4).is_some_and(|s| s != work) {
                    continue;
                }
                let unserved = work.starts_with("unserved");
                let intervals: &[usize] = if work == "planar" {
                    &[3, 4, 5]
                } else {
                    &[10, 25, 100]
                };
                for &n in intervals {
                    // (handler, registered); rotated by round.
                    let mut paths = vec![(false, true), (true, true)];
                    if unserved {
                        paths.push((false, false));
                    }
                    if round % 2 == 1 {
                        paths.reverse();
                    }
                    for (handler, register) in paths {
                        let path = match (handler, register) {
                            (true, _) => "handler",
                            (false, true) => "exit",
                            (false, false) => "exit-noreg",
                        };
                        if std::env::var("EXIT_PATH").is_ok_and(|p| p != path) {
                            continue;
                        }
                        let (vm, mut cpu) = setup(mode, work, n, register);
                        execute(
                            &vm,
                            &mut cpu,
                            mode,
                            100,
                            false,
                            false,
                            completion_budget,
                            handler,
                        );
                        let retired = cpu.retired_instructions();
                        let host = host_instructions();
                        let start = Instant::now();
                        let counts = execute(
                            &vm,
                            &mut cpu,
                            mode,
                            iterations,
                            splits,
                            trace,
                            completion_budget,
                            handler,
                        );
                        let seconds = start.elapsed().as_secs_f64();
                        let host = host_instructions() - host;
                        let retired = cpu.retired_instructions() - retired;
                        assert_eq!(retired, iterations * n as u64 + iterations.div_ceil(60000));
                        let exits = counts.exits as f64;
                        println!(
                        "{path},{mode},{work},{n},{round},{retired},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3}",
                        counts.exits,
                        counts.calls,
                        counts.completion_calls,
                        retired as f64 / seconds / 1e6,
                        seconds * 1e9 / exits,
                        counts.run_time.as_nanos() as f64 / exits,
                        counts.completion_time.as_nanos() as f64 / exits,
                        host as f64 / exits
                    );
                    }
                }
            }
        }
    }
}

// Deterministic bus/fault/delivery census for comparing two engine binaries.
fn census() {
    let early = matches!(
        std::env::var("EXIT_MODEL").as_deref(),
        Ok("8088" | "8086" | "v20" | "v30")
    );
    for mode in MODES {
        if early && mode != "real16" {
            continue;
        }
        if std::env::var("EXIT_MODEL").as_deref() == Ok("286") && mode.contains("32") {
            continue;
        }
        for scenario in ["scalar", "rmw", "string", "fault", "delivery"] {
            if scenario == "delivery" && mode != "real16" {
                continue;
            }
            let (vm, mut cpu) = setup(
                mode,
                if scenario == "delivery" {
                    "delivery"
                } else {
                    "out"
                },
                10,
                true,
            );
            cpu.enable_delivery_log_with(
                2,
                x86jit_core::DeliveryLogOptions::default().registers(true),
            )
            .unwrap();
            let short = mode == "real16" || mode == "seg16";
            let address = if short { 0x04 } else { 0x06 };
            let mut code = vec![0xe6, 0x80, 0xe4, 0x80, 0x88, address, 0x8a, address];
            match scenario {
                "scalar" => code.extend([0xe6, 0x80, 0xe4, 0x80, 0x88, address, 0x8a, address]),
                "rmw" => code.extend([0x00, address, 0x86, address]),
                "string" => {
                    cpu.set_reg(Reg::Rcx, 3);
                    cpu.set_reg(Reg::Rdi, DEVICE + 16);
                    code.extend([0xf3, 0xaa, 0xa4]);
                    if !early {
                        code.extend([0x6c, 0x6e]);
                    }
                }
                "fault" => code.extend([0x31, 0xc0, 0xf7, 0xf0]),
                "delivery" => {
                    vm.write_bytes(0x80, &[0, 1, 0, 0]).unwrap();
                    vm.write_bytes(0x100, &[0xcf]).unwrap();
                    code.extend([0xcd, 0x20, 0xcd, 0x20, 0xcd, 0x20]);
                }
                _ => unreachable!(),
            }
            code.push(0xf4);
            vm.write_bytes(CODE, &code).unwrap();
            cpu.cpu.rip = if short { 0 } else { CODE };
            for turn in 0..200 {
                let result = cpu
                    .run_steps(&vm, [0, 1, 2, 7, 100][turn % 5], StepOptions::default())
                    .unwrap();
                let mut ram = [0; 32];
                vm.read_bytes(0x7fe0, &mut ram).unwrap();
                println!("{mode}/{scenario}/{turn}: {result:?} retired={} executed={} linear={:?} ram={ram:?} log={:?} cpu={:?}", cpu.retired_instructions(), cpu.executed_instructions(), cpu.pending_mmio_linear_address(), cpu.delivery_log(), cpu.cpu);
                match result.stop() {
                    StepStop::Exit(Exit::PortIo {
                        dir: PortDir::In, ..
                    }) => cpu.complete_port_in(0xa5),
                    StepStop::Exit(Exit::PortIo { .. } | Exit::BudgetExhausted) => {}
                    StepStop::Exit(Exit::MmioRead { .. }) => cpu.complete_mmio_read(0xa5),
                    StepStop::Exit(Exit::MmioWrite { .. }) => cpu.complete_mmio_write(),
                    StepStop::Exit(
                        Exit::Hlt
                        | Exit::Exception { .. }
                        | Exit::UnmappedMemory { .. }
                        | Exit::UnknownInstruction { .. },
                    ) => break,
                    stop => panic!("unexpected census stop: {stop:?}"),
                }
                assert!(turn < 199, "census failed to finish");
            }
        }
    }
}
