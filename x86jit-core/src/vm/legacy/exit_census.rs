//! Exit/completion census: compare reusable entry state with full entry validation
//! on every call (spec.md §5, decision-28 stage 1).
use super::*;
use crate::{CpuModel, DeliveryLogOptions, ExceptionDelivery, PortDir, Reg};

fn trace(model: CpuModel, mode: u8, scenario: u8, cold: bool) -> Vec<String> {
    let mut vm = Vm::new(VmConfig::flat(0x20000));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(if scenario == 1 {
        ExceptionDelivery::InGuest
    } else {
        ExceptionDelivery::Outward
    });
    vm.map(0, 0xa000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0xa000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.map(0xb000, 0x15000, Prot::RWX, RegionKind::Ram).unwrap();
    let short = mode == 0 || mode == 3;
    let operand = if short { 0x04 } else { 0x06 };
    // Scalar port/MMIO, then a read-modify-write and an implicitly locked
    // exchange. Repetition revisits the same source block's exiting instruction.
    let mut code = vec![0x43, 0xe6, 0x80, 0xe4, 0x80, 0x88, operand, 0x8a, operand];
    code.extend([0x00, operand, 0x86, operand]);
    code.extend([0xf3, 0xaa, 0xa4]);
    if model.profile().isa.generation() != CpuGeneration::I8086 {
        code.extend([0x6c, 0x6e]);
    }
    if scenario == 1 {
        code.extend([0xcd, 0x20, 0xcd, 0x20, 0xcd, 0x20]);
    } else if scenario == 2 {
        // SDM Vol 2A §3.3 DIV: a zero divisor raises #DE.
        code.extend([0x31, 0xc0, 0xf7, 0xf0]);
    }
    code.push(0xf4);
    vm.write_bytes(0x10000, &code).unwrap();
    vm.write_bytes(0x80, &[0, 1, 0, 0]).unwrap();
    vm.write_bytes(0x100, &[0xcf]).unwrap();
    if mode == 2 {
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
    cpu.enable_delivery_log_with(2, DeliveryLogOptions::default().registers(true))
        .unwrap();
    if mode == 0 {
        cpu.cpu = CpuState::legacy_real_entry([0x1000, 0, 0, 0, 0, 0]);
    } else {
        cpu.cpu.legacy.cr0 = if mode == 2 { 0x8000_0001 } else { 1 };
        cpu.cpu.legacy.cr3 = if mode == 2 { 0x1000 } else { 0 };
        for (i, segment) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
            segment.base = if mode == 3 && i == 0 { 0x10000 } else { 0 };
            segment.limit = if short { 0xffff } else { u32::MAX };
            segment.attributes =
                (if i == 0 { 0x9b } else { 0x93 }) | if short { 0 } else { 0x4000 };
            segment.unusable = 0;
        }
    }
    cpu.cpu.rip = if short { 0 } else { 0x10000 };
    cpu.set_reg(Reg::Rsi, 0xa000);
    cpu.set_reg(Reg::Rdi, 0xa010);
    cpu.set_reg(Reg::Rcx, 3);
    cpu.set_reg(Reg::Rsp, 0x8000);
    let mut trace = Vec::new();
    let mut mmio = 0;
    let mut edited = false;
    let mut repeated = false;
    for turn in 0..200 {
        if cold {
            let execution = cpu.legacy_execution.as_mut().unwrap();
            execution.validated = None;
            execution.retry_slice = None;
        }
        let run = cpu
            .run_steps(&vm, [0, 1, 2, 7, 100][turn % 5], StepOptions::default())
            .unwrap();
        let mut ram = [0; 32];
        vm.read_bytes(0x7fe0, &mut ram).unwrap();
        trace.push(format!(
            "{run:?} retired={} executed={} linear={:?} cpu={:?} ram={ram:?} log={:?}",
            cpu.retired,
            cpu.executed,
            cpu.pending_mmio_linear_address(),
            cpu.cpu,
            cpu.delivery_log()
        ));
        if scenario == 3 && mmio == 1 && !edited {
            edited = true;
            // A public selector edit invalidates the decode stamp even though
            // its real-mode base stays zero; completion ownership is rechecked.
            cpu.cpu.fs ^= 1;
        }
        if scenario == 4 && mmio == 1 && !edited {
            edited = true;
            // A source write must not substitute a new decoded instruction for
            // an already owned bus transfer. Subsequent fetches validate SMC.
            vm.write_bytes(0x10000, &[0x90]).unwrap();
        }
        match run.stop() {
            StepStop::Exit(Exit::PortIo {
                dir: PortDir::In, ..
            }) => cpu.complete_port_in(0xa5),
            StepStop::Exit(Exit::PortIo { .. } | Exit::BudgetExhausted) => {}
            StepStop::Exit(Exit::MmioRead { .. }) => {
                cpu.complete_mmio_read(0xa5);
                mmio += 1;
            }
            StepStop::Exit(Exit::MmioWrite { .. }) => {
                cpu.complete_mmio_write();
                mmio += 1;
            }
            StepStop::Exit(Exit::Hlt | Exit::Exception { .. }) => {
                if scenario == 4 && !repeated {
                    repeated = true;
                    cpu.cpu.rip = if short { 0 } else { 0x10000 };
                    cpu.set_reg(Reg::Rcx, 3);
                    cpu.set_reg(Reg::Rdi, 0xa010);
                    continue;
                }
                assert!(mmio >= 9, "incomplete census: {model:?}/{mode}/{scenario}");
                if scenario == 1 {
                    let log = cpu.delivery_log().unwrap();
                    assert_eq!(log.total(), 3);
                    assert_eq!(log.len(), 2);
                    assert_eq!(log.dropped(), 1);
                }
                return trace;
            }
            stop => panic!("{model:?}/{mode}/{scenario}: {stop:?}"),
        }
    }
    panic!("census did not finish");
}

#[test]
fn exit_completion_census_matches_full_entry_validation() {
    for model in [
        CpuModel::I8088,
        CpuModel::I8086,
        CpuModel::NecV20,
        CpuModel::NecV30,
        CpuModel::I80286,
        CpuModel::I80386,
        CpuModel::I80486Dx,
    ] {
        for mode in 0..4 {
            if model.profile().isa.generation() == CpuGeneration::I8086 && mode != 0 {
                continue;
            }
            if model == CpuModel::I80286 && matches!(mode, 1 | 2) {
                continue;
            }
            for scenario in 0..5 {
                if scenario == 1 && mode != 0 {
                    continue;
                }
                assert_eq!(
                    trace(model, mode, scenario, false),
                    trace(model, mode, scenario, true),
                    "{model:?}/{mode}/{scenario}"
                );
            }
        }
    }
}
