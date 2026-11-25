//! Equivalence to the existing Exit/completion protocol (decision-28).
use super::*;
use crate::{
    CpuModel, CpuState, DeliveryLogOptions, ExceptionDelivery, Prot, Reg, ResetOptions, StepStop,
    VmConfig,
};

const CODE: u64 = 0x10000;
const MMIO: u64 = 0xa000;
fn setup(model: CpuModel, mode: u8, code: &[u8]) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x20000));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.map(0xb000, 0x15000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.register_device_range(DeviceRange::Ports(0..65536))
        .unwrap();
    vm.register_device_range(DeviceRange::Mmio(MMIO..MMIO + 0x1000))
        .unwrap();
    vm.write_bytes(CODE, code).unwrap();
    if mode == 2 || mode == 6 {
        vm.write_bytes(0x1000, &0x2007u32.to_le_bytes()).unwrap();
        for page in 0..32u32 {
            vm.write_bytes(
                0x2000 + u64::from(page * 4),
                &((page * 4096) | 7).to_le_bytes(),
            )
            .unwrap();
        }
    }
    let mut cpu = vm.new_vcpu();
    cpu.enable_delivery_log_with(2, DeliveryLogOptions::default().registers(true))
        .unwrap();
    if mode == 0 || mode >= 5 {
        cpu.cpu = CpuState::legacy_real_entry([0x1000, 0, 0, 0, 0, 0]);
        if mode >= 5 {
            cpu.cpu.legacy.cr0 = if mode == 6 { 0x8000_0001 } else { 1 };
            cpu.cpu.legacy.cr3 = 0x1000;
            cpu.cpu.legacy.cpl = 3;
            cpu.cpu.rflags_system = 0x23000;
            cpu.cpu.legacy.tr.selector = 0x28;
            cpu.cpu.legacy.tr.cache.base = 0x4000;
            cpu.cpu.legacy.tr.cache.limit = 0x2000;
            cpu.cpu.legacy.tr.cache.attributes = 0x89;
            cpu.cpu.legacy.tr.cache.unusable = 0;
            vm.write_bytes(0x4066, &0x80u16.to_le_bytes()).unwrap();
        }
    } else {
        cpu.cpu.legacy.cr0 = if mode == 2 { 0x8000_0001 } else { 1 };
        cpu.cpu.legacy.cr3 = if mode == 2 { 0x1000 } else { 0 };
        for (i, s) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
            s.base = if mode == 3 && i == 0 { CODE as u32 } else { 0 };
            s.limit = if mode == 3 {
                0xffff
            } else if mode == 4 {
                0x1ffff
            } else {
                u32::MAX
            };
            s.attributes = (if i == 0 { 0x9b } else { 0x93 }) | if mode == 3 { 0 } else { 0x4000 };
            s.unusable = 0;
        }
    }
    cpu.cpu.rip = if mode == 0 || mode == 3 || mode >= 5 {
        0
    } else {
        CODE
    };
    cpu.set_reg(Reg::Rsi, MMIO);
    cpu.set_reg(Reg::Rdi, MMIO + 16);
    cpu.set_reg(Reg::Rcx, 3);
    cpu.set_reg(Reg::Rdx, 0x80);
    cpu.set_reg(Reg::Rsp, 0x8000);
    cpu.set_reg(Reg::Rax, 0x5a);
    (vm, cpu)
}
#[derive(Default)]
struct Device {
    effects: Vec<(DeviceSpace, u64, DeviceDirection, u8, u64)>,
    requests: Vec<DeviceRequest>,
}
impl Device {
    fn transfer(
        &mut self,
        space: DeviceSpace,
        address: u64,
        dir: DeviceDirection,
        size: u8,
        value: u64,
    ) -> u64 {
        let answer = 0x1234_5678_9abc_dea5u64 >> (self.effects.len() % 8);
        self.effects.push((
            space,
            address,
            dir,
            size,
            if dir == DeviceDirection::Read {
                answer & (u64::MAX >> (64 - u32::from(size) * 8))
            } else {
                value
            },
        ));
        answer
    }
    fn exit(&mut self, cpu: &mut Vcpu, exit: &Exit) -> bool {
        match *exit {
            Exit::PortIo {
                port,
                size,
                dir,
                value,
            } => {
                let direction = if dir == PortDir::In {
                    DeviceDirection::Read
                } else {
                    DeviceDirection::Write
                };
                let value =
                    self.transfer(DeviceSpace::Port, u64::from(port), direction, size, value);
                if dir == PortDir::In {
                    cpu.complete_port_in(value);
                }
            }
            Exit::MmioRead { addr, size, .. } => {
                let v = self.transfer(DeviceSpace::Mmio, addr, DeviceDirection::Read, size, 0);
                cpu.complete_mmio_read(v);
            }
            Exit::MmioWrite {
                addr, size, value, ..
            } => {
                self.transfer(DeviceSpace::Mmio, addr, DeviceDirection::Write, size, value);
                cpu.complete_mmio_write();
            }
            _ => return false,
        }
        true
    }
}
impl DeviceIo for Device {
    fn access(&mut self, r: &DeviceRequest, _: &mut DeviceContext<'_>) -> DeviceResult {
        self.requests.push(r.clone());
        DeviceResult::Handled(self.transfer(r.space, r.address, r.direction, r.size, r.value))
    }
}
fn finish(vm: &Vm, cpu: &mut Vcpu, handler: bool, budget: u64) -> (Device, u64, String) {
    let mut device = Device::default();
    let mut count = 0;
    for _ in 0..200 {
        let before = device.effects.len();
        let zero = cpu
            .run_steps_with_devices(
                vm,
                0,
                StepOptions::default(),
                if handler { Some(&mut device) } else { None },
            )
            .unwrap();
        assert_eq!(zero.steps(), 0);
        assert_eq!(before, device.effects.len());
        let run = cpu
            .run_steps_with_devices(
                vm,
                budget,
                StepOptions::default(),
                if handler { Some(&mut device) } else { None },
            )
            .unwrap();
        count += run.steps();
        match run.into_stop() {
            StepStop::Exit(Exit::BudgetExhausted) => {}
            StepStop::Exit(exit) => {
                if !device.exit(cpu, &exit) {
                    return (device, count, format!("{exit:?}"));
                }
            }
            stop => panic!("{stop:?}"),
        }
    }
    panic!("did not finish")
}
fn architectural(cpu: &Vcpu) -> String {
    format!(
        "{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
        cpu.cpu.gpr,
        cpu.cpu.rip,
        cpu.cpu.flags,
        cpu.cpu.rflags_system,
        cpu.cpu.legacy,
        cpu.cpu.cs,
        cpu.cpu.ds,
        cpu.cpu.es,
        cpu.cpu.ss,
        cpu.retired_instructions()
    )
}
#[test]
fn handler_equals_exit_census() {
    for model in [
        CpuModel::I8088,
        CpuModel::I8086,
        CpuModel::NecV20,
        CpuModel::NecV30,
        CpuModel::I80286,
        CpuModel::I80386,
        CpuModel::I80486Dx,
        CpuModel::I80486DxCpuid,
    ] {
        for mode in 0..7 {
            let early = model.profile().isa.generation() == crate::CpuGeneration::I8086;
            if (early && mode != 0)
                || (model == CpuModel::I80286 && matches!(mode, 1 | 2 | 4 | 5 | 6))
            {
                continue;
            }
            let short = mode == 0 || mode == 3 || mode >= 5;
            let operand = if short { 0x04 } else { 0x06 };
            for scenario in 0..5 {
                let mut code = vec![0x43, 0xe6, 0x80, 0xe4, 0x80, 0x88, operand, 0x8a, operand];
                match scenario {
                    0 => code.extend([0x00, operand, 0x86, operand, 0xf0, 0x01, operand]),
                    1 => {
                        code.extend([0xf3, 0xa4]);
                    }
                    2 if !early => {
                        code.extend([0xf3, 0x6c, 0xb9, 3, 0]);
                        if !short {
                            code.extend([0, 0]);
                        }
                        code.extend([0xf3, 0x6e]);
                    }
                    3 => code.extend([0x8b, operand]),
                    4 => code.extend([0x31, 0xc0, 0xf7, 0xf0]),
                    _ => continue,
                }
                code.push(0xf4);
                for budget in [1, 7, 100] {
                    let (v1, mut c1) = setup(model, mode, &code);
                    let (v2, mut c2) = setup(model, mode, &code);
                    if scenario == 3 {
                        c1.set_reg(Reg::Rsi, MMIO - 1);
                        c2.set_reg(Reg::Rsi, MMIO - 1);
                    }
                    let (d1, n1, stop1) = finish(&v1, &mut c1, false, budget);
                    let (d2, n2, stop2) = finish(&v2, &mut c2, true, budget);
                    let tag = format!("{model:?}/{mode}/{scenario}/{budget}");
                    assert_eq!((n1, stop1), (n2, stop2), "{tag}");
                    assert_eq!(d1.effects, d2.effects, "{tag}");
                    assert_eq!(architectural(&c1), architectural(&c2), "{tag}");
                    for (addr, len) in [(0, MMIO as usize), (0xb000, 0x15000)] {
                        let mut a = vec![0; len];
                        let mut b = a.clone();
                        v1.read_bytes(addr, &mut a).unwrap();
                        v2.read_bytes(addr, &mut b).unwrap();
                        assert_eq!(a, b, "RAM {tag}");
                    }
                    assert_eq!(
                        format!("{:?}", c1.delivery_log()),
                        format!("{:?}", c2.delivery_log()),
                        "{tag}"
                    );
                    assert!(!d2.effects.is_empty());
                    for r in d2.requests {
                        assert!(r.unit_start.steps <= r.position.steps);
                        assert_eq!(r.cycles, None);
                    }
                }
            }
        }
    }
}
struct Callback<F>(F);
impl<F: FnMut(&DeviceRequest, &mut DeviceContext<'_>) -> DeviceResult> DeviceIo for Callback<F> {
    fn access(&mut self, r: &DeviceRequest, c: &mut DeviceContext<'_>) -> DeviceResult {
        (self.0)(r, c)
    }
}
fn run(cpu: &mut Vcpu, vm: &Vm, n: u64, h: &mut dyn DeviceIo) -> DeviceRun {
    cpu.run_steps_with_devices(vm, n, StepOptions::default(), Some(h))
        .unwrap()
}
#[test]
fn registration_quiescence_and_no_handler() {
    let (mut vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xe4, 0x80, 0xf4]);
    for r in [
        DeviceRange::Ports(1..1),
        DeviceRange::Ports(0..65537),
        DeviceRange::Mmio(std::ops::Range { start: 8, end: 4 }),
    ] {
        assert_eq!(
            vm.register_device_range(r),
            Err(DeviceRegistrationError::InvalidRange)
        );
    }
    assert_eq!(
        vm.register_device_range(DeviceRange::Mmio(0..1)),
        Err(DeviceRegistrationError::NotTrap)
    );
    assert_eq!(
        vm.register_device_range(DeviceRange::Ports(0x80..0x81)),
        Err(DeviceRegistrationError::Overlap)
    );
    let r = cpu
        .run_steps_with_devices(&vm, 1, StepOptions::default(), None)
        .unwrap();
    assert!(matches!(r.stop(), StepStop::Exit(Exit::PortIo { .. })));
    assert_eq!(
        vm.register_device_range(DeviceRange::Ports(0..1)),
        Err(DeviceRegistrationError::Busy)
    );
    assert!(matches!(
        vm.unmap(MMIO, 0x1000),
        Err(crate::MapError::DeviceBusy)
    ));
    cpu.complete_port_in(7);
    assert!(vm.devices().quiescent());
    cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    let id = vm.device_table.entries[0].0;
    vm.replace_device_range(id, DeviceRange::Ports(0x80..0x81))
        .unwrap();
    vm.unregister_device_range(id).unwrap();
    assert_eq!(
        vm.unregister_device_range(id),
        Err(DeviceRegistrationError::UnknownHandle)
    );
    let mut app = Vm::new(VmConfig::flat(1));
    assert_eq!(
        app.register_device_range(DeviceRange::Ports(0..1)),
        Err(DeviceRegistrationError::UnsupportedProfile)
    );
}
#[test]
fn fallback_fault_are_effect_free_and_fault_requires_explicit_recovery() {
    for fault in [false, true] {
        let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xe4, 0x80, 0xf4]);
        let mut calls = 0;
        let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
            calls += 1;
            c.write_ram(0x3000, &[0x55]).unwrap();
            c.set_intr(true);
            c.assert_nmi();
            c.request_host_stop();
            c.set_a20_enabled(false);
            c.invalidate_paging();
            if fault {
                DeviceResult::DeviceFault(DeviceError::new(42))
            } else {
                DeviceResult::Fallback
            }
        });
        let r = run(&mut cpu, &vm, 10, &mut h);
        assert_eq!(r.steps(), 1);
        assert!(if fault {
            matches!(r.stop(),StepStop::Exit(Exit::Device(crate::DeviceStop::Fault {error,..})) if error.code==42)
        } else {
            matches!(r.stop(), StepStop::Exit(Exit::PortIo { .. }))
        });
        if fault {
            assert!(matches!(
                run(&mut cpu, &vm, 10, &mut h).stop(),
                StepStop::Exit(Exit::Device(crate::DeviceStop::Fault { .. }))
            ));
        }

        assert_eq!(calls, 1);
        let mut byte = [0];
        vm.read_bytes(0x3000, &mut byte).unwrap();
        assert_eq!(byte, [0]);
        assert!(!cpu.device_cpu().intr);
        assert!(!cpu.has_pending_nmi());
        assert!(cpu.a20_enabled());
        cpu.complete_port_in(3);
        let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
    }
}
#[test]
fn handled_ins_finishes_last_step_and_retains_answer_through_destination_mmio() {
    for model in [CpuModel::I80286, CpuModel::I80386, CpuModel::I80486Dx] {
        for mode in [0, 3] {
            let (vm, mut cpu) = setup(model, mode, &[0xf3, 0x6c, 0xf4]);
            let mut h = Device::default();
            for remaining in [2, 1, 0] {
                let r = run(&mut cpu, &vm, 1, &mut h);
                assert_eq!(r.steps(), 1);
                assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
                assert_eq!(cpu.reg(Reg::Rcx), remaining);
            }
            assert_eq!(h.effects.len(), 6);
            assert!(h
                .requests
                .iter()
                .all(|r| r.unit_start.retired_instructions == 0));
            assert_eq!(
                h.effects
                    .iter()
                    .filter(|x| x.0 == DeviceSpace::Port)
                    .count(),
                3
            );
            assert_eq!(cpu.retired_instructions(), 1);
        }
    }
}
#[test]
fn callback_ram_smc_dirty_and_read_your_writes() {
    for mode in 0..5 {
        let operand = if mode == 0 || mode == 3 { 0x04 } else { 0x06 };
        let (vm, mut cpu) = setup(CpuModel::I80386, mode, &[0x8a, operand, 0x43, 0xf4]);
        vm.watch_range(CODE, 0x100);
        let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
            assert!(c.write_ram(MMIO, &[1]).is_err());
            assert!(c.read_ram(MMIO, &mut [0]).is_err());
            assert!(c.write_ram(u64::MAX, &[1, 2]).is_err());
            c.write_ram(CODE + 2, &[0x4b]).unwrap();
            let mut b = [0];
            c.read_ram(CODE + 2, &mut b).unwrap();
            assert_eq!(b, [0x4b]);
            assert_eq!(c.charge_extra_cycles(100), None);
            c.invalidate_paging();
            DeviceResult::Handled(0)
        });
        let r = run(&mut cpu, &vm, 10, &mut h);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{r:?}");
        assert_eq!(
            cpu.reg(Reg::Rbx) as u32,
            if mode == 0 || mode == 3 {
                0xffff
            } else {
                u32::MAX
            }
        );
        assert!(!vm.take_dirty_ranges().is_empty());
    }
}
#[test]
fn intr_level_if_shadow_and_deadlines() {
    for if_ in [false, true] {
        let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xe6, 0x80, 0xfb, 0x90, 0x43, 0xf4]);
        cpu.cpu.flags.if_ = if_;
        let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
            c.set_intr(true);
            DeviceResult::Handled(0)
        });
        let r = run(&mut cpu, &vm, 100, &mut h);
        assert!(
            matches!(
                r.stop(),
                StepStop::Exit(Exit::Device(crate::DeviceStop::IntrReady))
            ),
            "{r:?}"
        );
        assert_eq!(r.steps(), if if_ { 1 } else { 3 });
        assert_eq!(cpu.reg(Reg::Rbx), 0);
        let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(r.steps(), 0);
        assert!(matches!(
            r.stop(),
            StepStop::Exit(Exit::Device(crate::DeviceStop::IntrReady))
        ));
        cpu.device_cpu_mut().intr = false;
    }
    for deadline in [
        DeviceDeadline::Steps(3),
        DeviceDeadline::RetiredInstructions(3),
    ] {
        let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xe6, 0x80, 0x43, 0x43, 0x43, 0xf4]);
        let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
            c.set_deadline(deadline).unwrap();
            assert!(c.set_deadline(DeviceDeadline::Steps(100)).is_err());
            DeviceResult::Handled(0)
        });
        let r = run(&mut cpu, &vm, 100, &mut h);
        assert!(
            matches!(r.stop(),StepStop::Exit(Exit::Device(crate::DeviceStop::Deadline {position})) if position.steps==3),
            "{r:?}"
        );
        assert_eq!(r.steps(), 3);
        assert_eq!(cpu.reg(Reg::Rbx), 2);
    }
}
#[test]
fn panic_poison_is_vm_wide_and_reset_cannot_resume() {
    let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xe6, 0x80, 0xf4]);
    let mut other = vm.new_vcpu();
    let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
        c.write_ram(0x3000, &[9]).unwrap();
        panic!("device failed")
    });
    assert!(matches!(
        run(&mut cpu, &vm, 10, &mut h).stop(),
        StepStop::Exit(Exit::Device(crate::DeviceStop::HandlerFailed))
    ));
    assert!(vm.device_poisoned());
    cpu.reset_execution_state(ResetOptions::default());
    assert!(matches!(
        cpu.run(&vm, Some(1)),
        Exit::Device(crate::DeviceStop::Poisoned)
    ));
    assert!(matches!(
        other.run(&vm, Some(1)),
        Exit::Device(crate::DeviceStop::Poisoned)
    ));
    assert!(matches!(
        cpu.step_instruction(&vm),
        crate::StepResult::Exit(Exit::Device(crate::DeviceStop::Poisoned))
    ));
}

#[test]
fn fallback_mid_rep_and_mmio_fault_keep_acknowledged_transfers() {
    let code = [0xf3, 0x6e, 0xf4];
    let (base, mut a) = setup(CpuModel::I80386, 0, &code);
    let (vm, mut b) = setup(CpuModel::I80386, 0, &code);
    let (expected, n, stop) = finish(&base, &mut a, false, 100);
    let mut actual = Device::default();
    let mut count = 0;
    let mut declined = false;
    loop {
        let mut h = Callback(|r: &DeviceRequest, _: &mut DeviceContext<'_>| {
            if !declined && actual.effects.len() == 3 {
                declined = true;
                return DeviceResult::Fallback;
            }
            DeviceResult::Handled(actual.transfer(r.space, r.address, r.direction, r.size, r.value))
        });
        let r = run(&mut b, &vm, 100, &mut h);
        count += r.steps();
        match r.into_stop() {
            StepStop::Exit(exit) => {
                if !actual.exit(&mut b, &exit) {
                    assert_eq!(format!("{exit:?}"), stop);
                    break;
                }
            }
            s => panic!("{s:?}"),
        }
    }
    assert!(declined);
    assert_eq!(count, n);
    assert_eq!(actual.effects, expected.effects);
    assert_eq!(architectural(&a), architectural(&b));

    let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xf0, 0x01, 0x04, 0xf4]);
    let mut calls = Vec::new();
    let mut h = Callback(|r: &DeviceRequest, _: &mut DeviceContext<'_>| {
        calls.push(r.direction);
        if r.direction == DeviceDirection::Read {
            DeviceResult::Handled(7)
        } else {
            DeviceResult::DeviceFault(DeviceError::new(9))
        }
    });
    let r = run(&mut cpu, &vm, 1, &mut h);
    assert_eq!(r.steps(), 0);
    assert!(
        matches!(r.stop(),StepStop::Exit(Exit::Device(crate::DeviceStop::Fault {..})) if matches!(r.request().unwrap().handoff(),Exit::MmioWrite{..}))
    );

    assert_eq!(calls, [DeviceDirection::Read, DeviceDirection::Write]);
    cpu.complete_mmio_write();
    let mut h = Device::default();
    let r = run(&mut cpu, &vm, 1, &mut h);
    assert_eq!(r.steps(), 1);
    assert!(h.effects.is_empty());
}
#[test]
fn preflight_and_registration_boundaries_never_add_device_calls() {
    let (mut vm, mut cpu) = setup(CpuModel::I80386, 0, &[0x8b, 0x04, 0xf4]);
    let id = vm.device_table.entries[1].0;
    vm.unregister_device_range(id).unwrap();
    vm.register_device_range(DeviceRange::Mmio(MMIO..MMIO + 1))
        .unwrap();
    vm.register_device_range(DeviceRange::Mmio(MMIO + 1..MMIO + 2))
        .unwrap();
    let mut h = Device::default();
    let r = run(&mut cpu, &vm, 10, &mut h);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::MmioRead { size: 2, .. })
    ));
    assert!(h.effects.is_empty());
    let (vm, mut cpu) = setup(CpuModel::I80386, 4, &[0x8b, 0x06, 0xf4]);
    cpu.cpu.legacy.segments[crate::SegmentId::Ds as usize].limit = MMIO as u32 - 1;
    let r = run(&mut cpu, &vm, 10, &mut h);
    assert_eq!(r.steps(), 0);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::Exception { vector: 13, .. })
    ));
    assert!(h.effects.is_empty());
    let (vm, mut cpu) = setup(CpuModel::I80386, 6, &[0xe4, 0x80, 0xf4]);
    vm.write_bytes(0x4080 + 0x80 / 8, &[1]).unwrap();
    let r = run(&mut cpu, &vm, 10, &mut h);
    assert_eq!(r.steps(), 0);
    assert!(
        matches!(r.stop(), StepStop::Exit(Exit::Exception { vector: 13, .. })),
        "{r:?}"
    );
    assert!(h.effects.is_empty());
}
#[test]
fn device_nmi_and_host_stop_use_normal_delivery_and_completion_sites() {
    {
        let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xe6, 0x80, 0x43, 0xf4]);
        vm.write_bytes(8, &[0, 1, 0, 0]).unwrap();
        vm.write_bytes(0x100, &[0x43, 0xcf]).unwrap();
        let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
            c.assert_nmi();
            DeviceResult::Handled(0)
        });
        let r = run(&mut cpu, &vm, 100, &mut h);
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{r:?}");
        assert_eq!(r.steps(), 5);
        assert_eq!(cpu.reg(Reg::Rbx), 2);
        let log = cpu.delivery_log().unwrap();
        assert_eq!(log.total(), 1);
        assert_eq!(log.iter().next().unwrap().kind, crate::DeliveryKind::Nmi);
    }
    for code in [&[0x8a, 0x04, 0x43, 0xf4][..], &[0xf3, 0x6c, 0x43, 0xf4][..]] {
        let (vm, mut cpu) = setup(CpuModel::I80386, 0, code);
        let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
            c.request_host_stop();
            DeviceResult::Handled(0xa5)
        });
        let r = run(&mut cpu, &vm, 100, &mut h);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Interrupted)),
            "{r:?}"
        );
        assert_eq!(r.steps(), 1);
        assert_eq!(cpu.reg(Reg::Rbx), 0);
        assert!(cpu.cpu.port_string_in.is_none());
        assert!(cpu.cpu.pending_mmio.is_none());
    }
}
#[test]
fn rep_deadlines_and_deferred_a20_finish_the_owned_iteration() {
    let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xf3, 0x6c, 0xf4]);
    let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
        c.set_deadline(DeviceDeadline::Steps(2)).unwrap();
        c.set_a20_enabled(false);
        c.invalidate_paging();
        DeviceResult::Handled(0xa5)
    });
    let r = run(&mut cpu, &vm, 100, &mut h);
    assert_eq!(r.steps(), 2);
    assert!(
        matches!(
            r.stop(),
            StepStop::Exit(Exit::Device(crate::DeviceStop::Deadline { .. }))
        ),
        "{r:?}"
    );
    assert_eq!(cpu.reg(Reg::Rcx), 1);
    assert!(cpu.cpu.port_string_in.is_none());
    assert!(!cpu.a20_enabled());
    assert_eq!(cpu.pending_a20_enabled(), None);
}
#[test]
fn delivery_frame_mmio_and_log_overflow_equal_exit() {
    for model in [
        CpuModel::I8088,
        CpuModel::I8086,
        CpuModel::I80286,
        CpuModel::I80386,
        CpuModel::I80486Dx,
    ] {
        let code = [0xcd, 0x20, 0xcd, 0x20, 0xcd, 0x20, 0xf4];
        let mut outcomes = Vec::new();
        for handler in [false, true] {
            let (vm, mut cpu) = setup(model, 0, &code);
            cpu.exception_delivery = ExceptionDelivery::InGuest;
            vm.write_bytes(0x80, &[0, 1, 0, 0]).unwrap();
            vm.write_bytes(0x100, &[0xcf]).unwrap();
            let (d, n, s) = finish(&vm, &mut cpu, handler, 100);
            assert_eq!(cpu.delivery_log().unwrap().total(), 3);
            assert_eq!(cpu.delivery_log().unwrap().dropped(), 1);
            outcomes.push((
                d.effects,
                n,
                s,
                architectural(&cpu),
                format!("{:?}", cpu.delivery_log()),
            ));
        }
        assert_eq!(outcomes[0], outcomes[1]);
        let (vm, mut cpu) = setup(model, 0, &[0xcd, 0x20, 0xf4]);
        cpu.exception_delivery = ExceptionDelivery::InGuest;
        vm.write_bytes(0x80, &[0, 1, 0, 0]).unwrap();
        vm.write_bytes(0x100, &[0xf4]).unwrap();
        cpu.set_reg(Reg::Rsp, MMIO + 8);
        let mut h = Device::default();
        let r = run(&mut cpu, &vm, 10, &mut h);
        assert!(
            matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
            "{model:?}: {r:?}"
        );
        assert!(!h.effects.is_empty());
        assert_eq!(cpu.delivery_log().unwrap().total(), 1);
    }
}

#[test]
fn range_handles_are_vm_local_and_restore_rebuilds_registration() {
    let (mut vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xe4, 0x80, 0xf4]);
    let (mut other, _) = setup(CpuModel::I80386, 0, &[0xf4]);
    let foreign = other.device_table.entries[0].0;
    assert_eq!(
        vm.unregister_device_range(foreign),
        Err(DeviceRegistrationError::UnknownHandle)
    );
    cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert!(matches!(
        vm.try_fork_with_backend(Box::new(super::super::InterpreterBackend)),
        Err(crate::ForkError::DeviceBusy)
    ));
    cpu.complete_port_in(0xa5);
    cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    let mut child = vm
        .try_fork_with_backend(Box::new(super::super::InterpreterBackend))
        .unwrap();
    assert!(child.device_table.is_empty());
    child
        .register_device_range(DeviceRange::Ports(0x80..0x81))
        .unwrap();
    let mut restored = child.new_vcpu();
    restored.cpu = CpuState::legacy_real_entry([0x1000, 0, 0, 0, 0, 0]);
    restored.invalidate_paging();
    let mut h = Device::default();
    let r = run(&mut restored, &child, 10, &mut h);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
    assert_eq!(h.effects.len(), 1);
    other.unregister_device_range(foreign).unwrap();
}
#[test]
fn port_fault_can_be_acknowledged_and_intr_controller_can_choose_vector() {
    let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xe6, 0x80, 0x43, 0xf4]);
    let mut h = Callback(|_: &DeviceRequest, _: &mut DeviceContext<'_>| {
        DeviceResult::DeviceFault(DeviceError::new(7))
    });
    let r = run(&mut cpu, &vm, 10, &mut h);
    assert_eq!(r.steps(), 1);
    assert!(cpu.complete_device_fault(0));
    assert!(!cpu.complete_device_fault(0));
    cpu.set_intr(true);
    cpu.cpu.flags.if_ = true;
    assert!(matches!(
        cpu.run_steps(&vm, 10, StepOptions::default())
            .unwrap()
            .stop(),
        StepStop::Exit(Exit::Device(crate::DeviceStop::IntrReady))
    ));
    vm.write_bytes(0x80, &[0, 1, 0, 0]).unwrap();
    vm.write_bytes(0x100, &[0xf4]).unwrap();
    cpu.exception_delivery = ExceptionDelivery::InGuest;
    cpu.inject_irq(0x20);
    let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{r:?}");
    assert_eq!(cpu.delivery_log().unwrap().total(), 1);
    cpu.set_intr(false);
    assert!(!cpu.intr_asserted());
}

#[test]
fn request_identity_counts_and_deadline_moves_are_exact() {
    for mode in 0..7 {
        let (vm, mut cpu) = setup(CpuModel::I80386, mode, &[0x43, 0xe4, 0x80, 0xf4]);
        cpu.cpu.source_id = 999;
        let mut h = Device::default();
        run(&mut cpu, &vm, 100, &mut h);
        assert_eq!(h.requests.len(), 1);
        let r = &h.requests[0];
        assert_eq!(r.position.steps, 2);
        assert_eq!(r.position.retired_instructions, 2);
        assert_eq!(r.unit_start.steps, 1);
        assert_eq!(r.unit_start.retired_instructions, 1);
        assert_eq!(r.vcpu_id, 0);
        assert_eq!(
            r.instruction_ip,
            if mode == 0 || mode == 3 || mode >= 5 {
                1
            } else {
                CODE + 1
            }
        );
    }
    let (vm, mut cpu) = setup(
        CpuModel::I80386,
        0,
        &[0xe6, 0x80, 0xe6, 0x80, 0x43, 0x43, 0xf4],
    );
    let mut visits = 0;
    let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
        visits += 1;
        c.set_deadline(DeviceDeadline::Steps(if visits == 1 { 10 } else { 3 }))
            .unwrap();
        DeviceResult::Handled(0)
    });
    let r = run(&mut cpu, &vm, 100, &mut h);
    assert_eq!(r.steps(), 3);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::Device(crate::DeviceStop::Deadline { .. }))
    ));
    assert_eq!(cpu.reg(Reg::Rbx), 1);
}
#[test]
fn ss_shadow_blocks_intr_and_context_ram_writes_preflight_whole_span() {
    let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0x8e, 0xd0, 0xe6, 0x80, 0x43, 0xf4]);
    cpu.cpu.flags.if_ = true;
    cpu.set_reg(Reg::Rax, 0);
    // MOV SS holds its successor, whose OUT callback raises the level.
    let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
        assert!(c.write_ram(MMIO - 1, &[1, 2]).is_err());
        c.set_intr(true);
        DeviceResult::Handled(0)
    });
    let r = run(&mut cpu, &vm, 100, &mut h);
    assert_eq!(r.steps(), 2);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::Device(crate::DeviceStop::IntrReady))
    ));
    assert_eq!(cpu.reg(Reg::Rbx), 0);
    let mut b = [0];
    vm.read_bytes(MMIO - 1, &mut b).unwrap();
    assert_eq!(b, [0]);
}

#[test]
fn callback_a20_and_page_table_edits_revalidate_the_next_fetch() {
    // decision-22 §3: the final physical bus address follows the board gate.
    let mut vm = Vm::new(VmConfig::flat(0x120000));
    vm.set_cpu_model(CpuModel::I80386);
    vm.map(0, 0x120000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.register_device_range(DeviceRange::Ports(0x80..0x81))
        .unwrap();
    vm.write_bytes(0xffffe, &[0xe6, 0x80]).unwrap();
    vm.write_bytes(0, &[0xb8, 0x34, 0x12, 0xf4]).unwrap();
    vm.write_bytes(0x100000, &[0xb8, 0x78, 0x56, 0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([0xffff, 0, 0, 0, 0, 0]);
    cpu.cpu.rip = 0xe;
    let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
        c.set_a20_enabled(false);
        DeviceResult::Handled(0)
    });
    let r = run(&mut cpu, &vm, 10, &mut h);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{r:?}");
    assert_eq!(cpu.reg(Reg::Rax), 0x1234);

    let (vm, mut cpu) = setup(CpuModel::I80386, 2, &[0xe6, 0x80, 0x43, 0xf4]);
    vm.write_bytes(0x11000, &[0xe6, 0x80, 0x4b, 0xf4]).unwrap();
    let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
        c.write_ram(0x2040, &0x11003u32.to_le_bytes()).unwrap();
        c.invalidate_paging();
        DeviceResult::Handled(0)
    });
    let r = run(&mut cpu, &vm, 10, &mut h);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{r:?}");
    assert_eq!(cpu.reg(Reg::Rbx) as u32, u32::MAX);
}

#[test]
fn consistency_tiers_and_paging_fault_preflight_keep_the_exit_protocol() {
    for consistency in [
        crate::MemConsistency::Fast,
        crate::MemConsistency::AcqRel,
        crate::MemConsistency::FullTso,
    ] {
        for mode in 0..7 {
            let short = mode == 0 || mode == 3 || mode >= 5;
            let operand = if short { 0x04 } else { 0x06 };
            let code = [0xf0, 0x01, operand, 0xf3, 0xa4, 0xf4];
            let (mut v1, mut c1) = setup(CpuModel::I80386, mode, &code);
            let (mut v2, mut c2) = setup(CpuModel::I80386, mode, &code);
            v1.consistency = consistency;
            v2.consistency = consistency;
            let (d1, n1, s1) = finish(&v1, &mut c1, false, 7);
            let (d2, n2, s2) = finish(&v2, &mut c2, true, 7);
            assert_eq!((n1, s1), (n2, s2));
            assert_eq!(d1.effects, d2.effects);
            assert_eq!(architectural(&c1), architectural(&c2));
        }
    }
    let (vm, mut cpu) = setup(CpuModel::I80386, 2, &[0x8a, 0x06, 0xf4]);
    vm.write_bytes(0x2028, &0xa000u32.to_le_bytes()).unwrap();
    let mut h = Device::default();
    let r = run(&mut cpu, &vm, 10, &mut h);
    assert_eq!(r.steps(), 0);
    assert!(
        matches!(r.stop(), StepStop::Exit(Exit::Exception { vector: 14, .. })),
        "{r:?}"
    );
    assert!(h.effects.is_empty());
}
#[test]
fn context_signal_delivery_matches_signals_after_the_port_exit() {
    for nmi in [false, true] {
        let (v1, mut a) = setup(CpuModel::I80386, 0, &[0xe6, 0x80, 0x43, 0xf4]);
        let (v2, mut b) = setup(CpuModel::I80386, 0, &[0xe6, 0x80, 0x43, 0xf4]);
        for vm in [&v1, &v2] {
            vm.write_bytes(8, &[0, 1, 0, 0]).unwrap();
            vm.write_bytes(0x100, &[0x43, 0xcf]).unwrap();
        }
        let first = a.run_steps(&v1, 100, StepOptions::default()).unwrap();
        assert_eq!(first.steps(), 1);
        assert!(matches!(first.stop(), StepStop::Exit(Exit::PortIo { .. })));
        if nmi {
            a.raise_nmi().unwrap();
        } else {
            a.exit_request().request_exit();
        }
        let last = a.run_steps(&v1, 99, StepOptions::default()).unwrap();
        let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
            if nmi {
                c.assert_nmi();
            } else {
                c.request_host_stop();
            }
            DeviceResult::Handled(0)
        });
        let result = run(&mut b, &v2, 100, &mut h);
        assert_eq!(first.steps() + last.steps(), result.steps());
        assert_eq!(format!("{:?}", last.stop()), format!("{:?}", result.stop()));
        assert_eq!(architectural(&a), architectural(&b));
        assert_eq!(
            format!("{:?}", a.delivery_log()),
            format!("{:?}", b.delivery_log())
        );
        let mut ra = [0; 32];
        let mut rb = ra;
        v1.read_bytes(0x7fe0, &mut ra).unwrap();
        v2.read_bytes(0x7fe0, &mut rb).unwrap();
        assert_eq!(ra, rb);
    }
}

#[test]
fn ins_completion_entered_from_exit_has_current_call_position() {
    let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0x43, 0x6c, 0xf4]);
    let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
    assert_eq!(r.steps(), 2);
    assert_eq!(cpu.retired_instructions(), 2);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::PortIo { .. })));
    cpu.complete_port_in(0xa5);
    let mut h = Device::default();
    let r = run(&mut cpu, &vm, 1, &mut h);
    assert_eq!(r.steps(), 1);
    assert_eq!(h.requests.len(), 1);
    let request = &h.requests[0];
    assert_eq!(request.position.steps, 0);
    assert_eq!(request.unit_start.steps, 0);
    assert_eq!(request.position.retired_instructions, 2);
    assert_eq!(request.unit_start.retired_instructions, 1);
    assert_eq!(request.instruction_ip, 1);
}

#[test]
fn first_registration_after_an_unregistered_handoff_requires_completion() {
    let mut vm = Vm::new(VmConfig::flat(0x2000));
    vm.set_cpu_model(CpuModel::I80386);
    vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xe4, 0x80, 0xf4]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert!(matches!(r.stop(), StepStop::Exit(Exit::PortIo { .. })));
    assert_eq!(
        vm.register_device_range(DeviceRange::Ports(0x80..0x81)),
        Err(DeviceRegistrationError::Busy)
    );
    cpu.complete_port_in(9);
    vm.register_device_range(DeviceRange::Ports(0x80..0x81))
        .unwrap();
    cpu.cpu.rip = 0x1000;
    let mut h = Device::default();
    let r = run(&mut cpu, &vm, 10, &mut h);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
    assert_eq!(h.effects.len(), 1);
}

#[test]
fn intr_level_survives_execution_abandonment() {
    let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xf4]);
    cpu.set_intr(true);
    cpu.cpu.flags.if_ = true;
    cpu.reset_execution_state(ResetOptions::default());
    assert!(cpu.intr_asserted());
    let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert_eq!(r.steps(), 0);
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::Device(crate::DeviceStop::IntrReady))
    ));
    cpu.set_intr(false);
    assert!(matches!(
        cpu.run_steps(&vm, 1, StepOptions::default())
            .unwrap()
            .stop(),
        StepStop::Exit(Exit::Hlt)
    ));
}

#[test]
fn device_run_owns_request_evidence_after_abandonment() {
    let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xe4, 0x80, 0xf4]);
    let mut h = Callback(|_: &DeviceRequest, _: &mut DeviceContext<'_>| {
        DeviceResult::DeviceFault(DeviceError::new(17))
    });
    let result = run(&mut cpu, &vm, 10, &mut h);
    let request = result.request().unwrap().clone();
    assert_eq!(request.address, 0x80);
    assert_eq!(cpu.device_request(), Some(&request));
    cpu.reset_execution_state(ResetOptions::default());
    assert!(cpu.device_request().is_none());
    assert_eq!(result.request(), Some(&request));
    assert!(
        matches!(result.stop(),StepStop::Exit(Exit::Device(crate::DeviceStop::Fault {error})) if error.code==17)
    );
}

#[test]
fn snapshots_reject_registered_execution_without_a_handler() {
    let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xeb, 0xfe]);
    let stop = cpu.exit_request();
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| cpu.run_steps(&vm, u64::MAX, StepOptions::default()));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while vm.devices().quiescent() && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        let active = !vm.devices().quiescent();
        let refused = matches!(
            vm.try_fork_with_backend(Box::new(crate::InterpreterBackend)),
            Err(crate::ForkError::DeviceBusy)
        );
        stop.request_exit();
        let result = worker.join().unwrap().unwrap();
        assert!(active);
        assert!(refused);
        assert!(matches!(result.stop(), StepStop::Exit(Exit::Interrupted)));
    });
}

fn fork_ok(vm: &Vm) -> bool {
    vm.try_fork_with_backend(Box::new(crate::InterpreterBackend))
        .is_ok()
}
#[test]
fn completed_handoffs_leave_fork_and_registration_available() {
    // No ranges: an OUT needs no completion, so fork stays available.
    let mut vm = Vm::new(VmConfig::flat(0x20000));
    vm.set_cpu_model(CpuModel::I80386);
    vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(CODE, &[0xe6, 0x80, 0xe4, 0x80, 0xf4])
        .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([0x1000, 0, 0, 0, 0, 0]);
    let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::PortIo {
            dir: PortDir::Out,
            ..
        })
    ));
    assert!(fork_ok(&vm));
    // A genuinely pending IN still refuses until it is answered.
    let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::PortIo {
            dir: PortDir::In,
            ..
        })
    ));
    assert!(matches!(
        vm.try_fork_with_backend(Box::new(crate::InterpreterBackend)),
        Err(crate::ForkError::DeviceBusy)
    ));
    cpu.complete_port_in(1);
    assert!(fork_ok(&vm));

    // With ranges: an OUT taken as an Exit, by Fallback or with no handler.
    for handler in [false, true] {
        let (mut vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xe6, 0x80, 0xf4]);
        let mut h = Callback(|_: &DeviceRequest, _: &mut DeviceContext<'_>| DeviceResult::Fallback);
        let r = cpu
            .run_steps_with_devices(
                &vm,
                10,
                StepOptions::default(),
                if handler { Some(&mut h) } else { None },
            )
            .unwrap();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::PortIo { .. })));
        let id = vm.device_table.entries[0].0;
        vm.replace_device_range(id, DeviceRange::Ports(0..0x100))
            .unwrap();
        vm.register_device_range(DeviceRange::Ports(0x100..0x101))
            .unwrap();
        assert!(fork_ok(&vm));
    }

    // MMIO read and write: busy while issued, available once completed,
    // including a zero-budget call in between.
    for code in [[0x8a, 0x04, 0xf4], [0x88, 0x04, 0xf4]] {
        let (mut vm, mut cpu) = setup(CpuModel::I80386, 0, &code);
        let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
        let read = matches!(r.stop(), StepStop::Exit(Exit::MmioRead { .. }));
        assert!(read || matches!(r.stop(), StepStop::Exit(Exit::MmioWrite { .. })));
        cpu.run_steps(&vm, 0, StepOptions::default()).unwrap();
        assert!(!fork_ok(&vm));
        assert_eq!(
            vm.register_device_range(DeviceRange::Ports(0..1)),
            Err(DeviceRegistrationError::Busy)
        );
        if read {
            cpu.complete_mmio_read(0x11);
        } else {
            cpu.complete_mmio_write();
        }
        assert!(fork_ok(&vm));
        let id = vm.device_table.entries[0].0;
        vm.unregister_device_range(id).unwrap();
        let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)));
        assert!(fork_ok(&vm));
    }
}
#[test]
fn held_single_step_trap_precedes_intr_ready() {
    // SDM Vol 3A §6.9: the single-step trap of a completed OUT outranks INTR.
    let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xe6, 0x80, 0x43, 0xf4]);
    // #DB handler at 0000:0600: STI; NOP; HLT.
    vm.write_bytes(4, &[0x00, 0x06, 0x00, 0x00]).unwrap();
    vm.write_bytes(0x600, &[0xfb, 0x90, 0xf4]).unwrap();
    cpu.cpu.flags.if_ = true;
    cpu.cpu.rflags_system |= 0x100;
    let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
        c.set_intr(true);
        DeviceResult::Handled(0)
    });
    let r = run(&mut cpu, &vm, 100, &mut h);
    assert!(
        matches!(
            r.stop(),
            StepStop::Exit(Exit::Device(crate::DeviceStop::IntrReady))
        ),
        "{r:?}"
    );
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x602));
    assert_eq!(cpu.reg(Reg::Rbx), 0);
    let mut frame = [0; 4];
    vm.read_bytes(0x8000 - 6, &mut frame).unwrap();
    assert_eq!(frame, [0x02, 0x00, 0x00, 0x10]);
}
#[test]
fn intr_and_deadline_stops_are_exact_without_single_steps() {
    // INTR asserted across IF=0: STI inside a block, POPF ending one.
    for (code, steps, rbx) in [
        (
            &[0xe6, 0x80, 0x43, 0x43, 0xfb, 0x43, 0x43, 0x43, 0xf4][..],
            5,
            3,
        ),
        (&[0xe6, 0x80, 0x43, 0x9d, 0x43, 0x43, 0xf4][..], 3, 1),
    ] {
        let (vm, mut cpu) = setup(CpuModel::I80386, 0, code);
        cpu.cpu.flags.if_ = false;
        vm.write_bytes(0x8000, &0x0202u16.to_le_bytes()).unwrap();
        let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
            c.set_intr(true);
            DeviceResult::Handled(0)
        });
        let r = run(&mut cpu, &vm, 100, &mut h);
        assert!(
            matches!(
                r.stop(),
                StepStop::Exit(Exit::Device(crate::DeviceStop::IntrReady))
            ),
            "{r:?}"
        );
        assert_eq!((r.steps(), cpu.reg(Reg::Rbx)), (steps, rbx));
    }
    // The level persists into a plain run.
    let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0x43, 0x43, 0xfb, 0x43, 0x43, 0xf4]);
    cpu.cpu.flags.if_ = false;
    cpu.set_intr(true);
    let r = cpu.run_steps(&vm, 100, StepOptions::default()).unwrap();
    assert!(matches!(
        r.stop(),
        StepStop::Exit(Exit::Device(crate::DeviceStop::IntrReady))
    ));
    assert_eq!((r.steps(), cpu.reg(Reg::Rbx)), (4, 3));
    // A deadline inside a straight-line block.
    for deadline in [
        DeviceDeadline::Steps(5),
        DeviceDeadline::RetiredInstructions(5),
    ] {
        let code = [0xe6, 0x80, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0xf4];
        let (vm, mut cpu) = setup(CpuModel::I80386, 0, &code);
        let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
            c.set_deadline(deadline).unwrap();
            DeviceResult::Handled(0)
        });
        let r = run(&mut cpu, &vm, 100, &mut h);
        assert!(
            matches!(r.stop(),StepStop::Exit(Exit::Device(crate::DeviceStop::Deadline {position})) if position.steps==5),
            "{r:?}"
        );
        assert_eq!((r.steps(), cpu.reg(Reg::Rbx)), (5, 4));
    }
}
#[test]
fn handled_access_rewriting_the_running_block_executes_the_new_bytes() {
    // The handler rewrites the successor instruction, and the one after it,
    // of the cached, running block: real-address, Flat32 and paged blocks.
    for mode in 0..3 {
        let operand = if mode == 0 { 0x04 } else { 0x06 };
        let code = [0x8a, operand, 0x43, 0x43, 0x43, 0xf4];
        let (vm, mut cpu) = setup(CpuModel::I80386, mode, &code);
        let context =
            crate::legacy::derive_legacy_context(CpuModel::I80386.legacy_model(), &cpu.cpu)
                .unwrap();
        assert_eq!(context.paging_enabled, mode == 2);
        if mode == 1 {
            assert_eq!(context.segment_class, crate::SegmentClass::Flat32);
        }
        let entry = cpu.cpu.clone();
        let r = run(&mut cpu, &vm, 100, &mut Device::default());
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{r:?}");
        assert_eq!(cpu.reg(Reg::Rbx), 3);
        for (at, rbx) in [(2, 1), (3, 1)] {
            vm.write_bytes(CODE, &code).unwrap();
            cpu.cpu = entry.clone();
            let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
                c.write_ram(CODE + at, &[0x4b]).unwrap();
                DeviceResult::Handled(0)
            });
            let r = run(&mut cpu, &vm, 100, &mut h);
            assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{r:?}");
            assert_eq!(cpu.reg(Reg::Rbx), rbx, "mode {mode}, offset {at}");
            // The rewritten block, cached again, keeps the new bytes.
            cpu.cpu = entry.clone();
            let r = run(&mut cpu, &vm, 100, &mut Device::default());
            assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{r:?}");
            assert_eq!(cpu.reg(Reg::Rbx), rbx);
        }
    }
}
#[test]
fn registration_errors_display_and_unknown_handle_wins() {
    let (mut vm, _) = setup(CpuModel::I80386, 0, &[0xe6, 0x80, 0xf4]);
    let (other, _) = setup(CpuModel::I80386, 0, &[0xf4]);
    let foreign = other.device_table.entries[0].0;
    assert_eq!(
        vm.replace_device_range(foreign, DeviceRange::Ports(0x80..0x81)),
        Err(DeviceRegistrationError::UnknownHandle)
    );
    let error: &dyn std::error::Error = &DeviceRegistrationError::Overlap;
    assert_eq!(
        error.to_string(),
        "device range overlaps a registered range"
    );
    // With every range removed, a handler is never consulted.
    let ids: Vec<_> = vm.device_table.entries.iter().map(|e| e.0).collect();
    for id in ids {
        vm.unregister_device_range(id).unwrap();
    }
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([0x1000, 0, 0, 0, 0, 0]);
    let mut calls = 0;
    let mut h = Callback(|_: &DeviceRequest, _: &mut DeviceContext<'_>| {
        calls += 1;
        DeviceResult::Handled(0)
    });
    let r = run(&mut cpu, &vm, 10, &mut h);
    assert!(matches!(r.stop(), StepStop::Exit(Exit::PortIo { .. })));
    assert_eq!(calls, 0);
}
#[test]
fn event_continuation_and_held_trap_keep_fork_and_registration_busy() {
    // An external IRQ whose frame pushes hit MMIO: each completed write leaves
    // the accepted event's continuation, which CpuState does not carry.
    let (mut vm, mut cpu) = setup(CpuModel::I80386, 0, &[0x90, 0xf4]);
    vm.write_bytes(0x20 * 4, &[0x00, 0x01, 0x00, 0x00]).unwrap();
    vm.write_bytes(0x100, &[0xf4]).unwrap();
    cpu.cpu.flags.if_ = true;
    cpu.set_reg(Reg::Rsp, MMIO + 8);
    cpu.inject_irq(0x20);
    let mut writes = 0;
    loop {
        let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
        match r.stop() {
            StepStop::Exit(Exit::MmioWrite { .. }) => {
                writes += 1;
                cpu.complete_mmio_write();
                assert!(!fork_ok(&vm), "write {writes}");
                assert_eq!(
                    vm.register_device_range(DeviceRange::Ports(0..1)),
                    Err(DeviceRegistrationError::Busy)
                );
            }
            StepStop::Exit(Exit::Hlt) => break,
            stop => panic!("{stop:?}"),
        }
    }
    assert_eq!((writes, cpu.cpu.cs), (3, 0));
    assert!(fork_ok(&vm));
    let id = vm.device_table.entries[0].0;
    vm.replace_device_range(id, DeviceRange::Ports(0..0x100))
        .unwrap();

    // SDM Vol 3A §6.9: an OUT with TF=1 holds its single-step trap past the
    // exit, or past an in-process completion that ends the budget.
    for handler in [false, true] {
        let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xe6, 0x80, 0x43, 0xf4]);
        // #DB handler at 0000:0600: HLT.
        vm.write_bytes(4, &[0x00, 0x06, 0x00, 0x00]).unwrap();
        vm.write_bytes(0x600, &[0xf4]).unwrap();
        cpu.cpu.rflags_system |= 0x100;
        let r = if handler {
            run(&mut cpu, &vm, 1, &mut Device::default()).into_run()
        } else {
            cpu.run_steps(&vm, 10, StepOptions::default()).unwrap()
        };
        assert!(
            matches!(
                r.stop(),
                StepStop::Exit(Exit::BudgetExhausted | Exit::PortIo { .. })
            ),
            "{r:?}"
        );
        assert!(!fork_ok(&vm), "handler {handler}");
        let r = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{r:?}");
        assert_eq!((cpu.cpu.cs, cpu.reg(Reg::Rbx)), (0, 0));
        assert!(fork_ok(&vm));
    }
}
#[test]
fn near_transfers_and_a_last_flags_load_keep_the_block_whole() {
    use crate::ir::{IrBlock, IrOp, Val};
    use crate::legacy::LegacyOp as L;
    let block = |ops: Vec<IrOp>| IrBlock {
        ops,
        guest_start: 0,
        temp_count: 0,
        guest_len: 0,
        icount: 0,
        mode: crate::CpuMode::Real16,
        legacy_context: None,
    };
    let insn = |guest_addr| IrOp::InsnStart { guest_addr };
    let near_call = IrOp::Legacy(L::Transfer {
        cs: None,
        ip: Val::Imm(0x10),
        size: 2,
        return_ip: Some(3),
    });
    let far_call = IrOp::Legacy(L::Transfer {
        cs: Some(Val::Imm(0x10)),
        ip: Val::Imm(0x10),
        size: 2,
        return_ip: Some(5),
    });
    let near_ret = IrOp::Legacy(L::Return {
        size: 2,
        far: false,
        pop_extra: 0,
    });
    let far_ret = IrOp::Legacy(L::Return {
        size: 2,
        far: true,
        pop_extra: 0,
    });
    let sti = IrOp::SetIf { value: true };
    let popf = IrOp::Legacy(L::Flags { size: 2, pop: true });
    let (_vm, mut cpu) = setup(CpuModel::I80386, 0, &[0xf4]);
    cpu.cpu.flags.if_ = false;
    cpu.device_cpu_mut().intr = true;
    let devices = Run::default();
    let mut single = 0;
    for (ops, whole) in [
        (vec![insn(0), near_call.clone()], true),
        (vec![insn(0), near_ret.clone()], true),
        (
            vec![insn(0), near_call.clone(), insn(3), near_ret.clone()],
            true,
        ),
        (vec![insn(0), insn(1), sti.clone()], true),
        (vec![insn(0), insn(1), popf.clone()], true),
        (vec![insn(0), insn(1), far_call.clone()], true),
        (vec![insn(0), insn(1), far_ret.clone()], true),
        (vec![insn(0), sti, insn(1)], false),
        (vec![insn(0), popf, insn(1)], false),
        (vec![insn(0), far_call, insn(1)], false),
        (vec![insn(0), far_ret, insn(1)], false),
    ] {
        let cap = devices.cap(&cpu, &block(ops), 0, 100);
        assert_eq!(cap, if whole { 100 } else { 1 });
        single += u32::from(cap == 1);
    }
    assert_eq!(single, 4);
    // Stop positions: INTR held across IF=0 through a near CALL and RET.
    for (mode, code) in [
        (
            0,
            &[
                0xe6, 0x80, 0xe8, 0x04, 0x00, 0xfb, 0x43, 0x43, 0xf4, 0x43, 0xc3,
            ][..],
        ),
        (
            1,
            &[
                0xe6, 0x80, 0xe8, 0x04, 0x00, 0x00, 0x00, 0xfb, 0x43, 0x43, 0xf4, 0x43, 0xc3,
            ][..],
        ),
    ] {
        let (vm, mut cpu) = setup(CpuModel::I80386, mode, code);
        cpu.cpu.flags.if_ = false;
        let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
            c.set_intr(true);
            DeviceResult::Handled(0)
        });
        let r = run(&mut cpu, &vm, 100, &mut h);
        assert!(
            matches!(
                r.stop(),
                StepStop::Exit(Exit::Device(crate::DeviceStop::IntrReady))
            ),
            "{r:?}"
        );
        // OUT, CALL, INC, RET, STI, INC (the STI shadow).
        assert_eq!((r.steps(), cpu.reg(Reg::Rbx)), (6, 2), "mode {mode}");
    }
}
#[test]
fn blocked_nmi_edge_does_not_delay_intr_ready() {
    // spec.md §17.10: inside an NMI handler a new edge is discarded or
    // remembered, so INTR is accepted at the first boundary with IF=1.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let (vm, mut cpu) = setup(model, 0, &[0x90, 0xf4]);
        // NMI handler at 0000:0600: STI; OUT 80h,AL; INC BX; HLT.
        vm.write_bytes(8, &[0x00, 0x06, 0x00, 0x00]).unwrap();
        vm.write_bytes(0x600, &[0xfb, 0xe6, 0x80, 0x43, 0xf4])
            .unwrap();
        cpu.raise_nmi().unwrap();
        let mut h = Callback(|_: &DeviceRequest, c: &mut DeviceContext<'_>| {
            c.assert_nmi();
            c.set_intr(true);
            DeviceResult::Handled(0)
        });
        let r = run(&mut cpu, &vm, 100, &mut h);
        assert!(
            matches!(
                r.stop(),
                StepStop::Exit(Exit::Device(crate::DeviceStop::IntrReady))
            ),
            "{model:?}: {r:?}"
        );
        assert_eq!(
            (r.steps(), cpu.cpu.cs, cpu.cpu.rip, cpu.reg(Reg::Rbx)),
            (2, 0, 0x603, 0),
            "{model:?}"
        );
    }
}
#[test]
fn external_numeric_error_precedes_intr_ready() {
    // §17.9: the External route reports a pending unmasked x87 error at the
    // boundary's service step, which precedes accepting an IRQ.
    let (vm, mut cpu) = setup(CpuModel::I80386, 0, &[0x90, 0xf4]);
    cpu.set_numeric_error_route(crate::NumericErrorRoute::External)
        .unwrap();
    cpu.cpu.fpu_sw |= 0x81;
    cpu.cpu.flags.if_ = true;
    cpu.set_intr(true);
    let r = run(&mut cpu, &vm, 100, &mut Device::default());
    assert!(
        matches!(
            r.stop(),
            StepStop::Exit(Exit::ExternalNumericError {
                event: crate::NumericErrorEvent::Asserted,
                ..
            })
        ),
        "{r:?}"
    );
    assert_eq!(r.steps(), 0);
    let r = run(&mut cpu, &vm, 100, &mut Device::default());
    assert!(
        matches!(
            r.stop(),
            StepStop::Exit(Exit::Device(crate::DeviceStop::IntrReady))
        ),
        "{r:?}"
    );
    assert_eq!(r.steps(), 0);
}

/// Run `f` with the in-block path of decision-28 stage 3 disabled: every
/// handled transfer then takes the stage-2 handoff inside the dispatcher.
fn stage_two<T>(f: impl FnOnce() -> T) -> T {
    IN_BLOCK_TEST.with(|t| t.set((true, t.get().1, t.get().2)));
    let result = f();
    IN_BLOCK_TEST.with(|t| t.set((false, t.get().1, t.get().2)));
    result
}
/// Callbacks the in-block path made, and transfers it completed in block.
fn in_block_counts() -> (u64, u64) {
    IN_BLOCK_TEST.with(|t| (t.get().1, t.get().2))
}
fn in_block_completions() -> u64 {
    in_block_counts().1
}
fn assert_same_machine(tag: &str, (v1, c1): (&Vm, &Vcpu), (v2, c2): (&Vm, &Vcpu)) {
    assert_eq!(architectural(c1), architectural(c2), "{tag}");
    assert_eq!(
        c1.executed_instructions(),
        c2.executed_instructions(),
        "{tag}"
    );
    for (addr, len) in [(0, MMIO as usize), (0xb000, 0x15000)] {
        let mut a = vec![0; len];
        let mut b = a.clone();
        v1.read_bytes(addr, &mut a).unwrap();
        v2.read_bytes(addr, &mut b).unwrap();
        assert_eq!(a, b, "RAM {tag}");
    }
    assert_eq!(
        format!("{:?}", c1.delivery_log()),
        format!("{:?}", c2.delivery_log()),
        "{tag}"
    );
}

/// Decision-28 stage 3 inside a block: a handled scalar MMIO transfer of a
/// real-mode shared walk completes where the walk stands. The result equals the
/// Exit/completion protocol and the stage-2 handoff (in-block path disabled):
/// state, RAM, device effect order, counts, the stop and the delivery log, and
/// stage 2's exact requests (positions, unit start, IP).
#[test]
fn in_block_equals_exit_and_stage_two() {
    // mov [si],al; inc bx; mov bl,[si]; mov [si+2],ax; mov cx,[si+2];
    // mov [0xa020],al; mov ax,[0xa022]; inc bx
    let scalar = [
        0x88, 0x04, 0x43, 0x8a, 0x1c, 0x89, 0x44, 0x02, 0x8b, 0x4c, 0x02, 0xa2, 0x20, 0xa0, 0xa1,
        0x22, 0xa0, 0x43,
    ];
    let scenarios: [&[u8]; 6] = [
        // Scalar transfers between register instructions.
        &[],
        // A later instruction of the block faults after handled transfers:
        // a word at offset 0xffff (#GP on the 80286 and later).
        &[0x8b, 0x1e, 0xff, 0xff],
        // A fault on the instruction's later operand after its device read:
        // PUSH [SI] with the stack outside mapped memory.
        &[0x8e, 0xd2, 0xff, 0x34],
        // A divide of the device answer that faults (#DE).
        &[0xb8, 0xff, 0xff, 0xf6, 0x34],
        // A software interrupt delivered between transfers.
        &[0xcd, 0x20, 0x88, 0x04],
        // Read-modify-write and a locked form: two transfers, never in block.
        &[0x00, 0x04, 0xf0, 0x01, 0x04],
    ];
    for model in [
        CpuModel::I8086,
        CpuModel::I80286,
        CpuModel::I80386,
        CpuModel::I80486Dx,
    ] {
        for (scenario, tail) in scenarios.iter().enumerate() {
            let mut code = scalar.to_vec();
            code.extend_from_slice(tail);
            code.push(0xf4);
            for budget in [1, 2, 7, 100] {
                let tag = format!("{model:?}/{scenario}/{budget}");
                let machine = || {
                    let (vm, mut cpu) = setup(model, 0, &code);
                    // DX selects an unmapped stack segment for PUSH.
                    cpu.set_reg(Reg::Rdx, 0x3000);
                    vm.write_bytes(0x80, &[0, 1, 0, 0]).unwrap();
                    vm.write_bytes(0x100, &[0x43, 0xcf]).unwrap();
                    (vm, cpu)
                };
                let (v1, mut c1) = machine();
                let (v2, mut c2) = machine();
                let (v3, mut c3) = machine();
                let r1 = finish(&v1, &mut c1, false, budget);
                let r2 = stage_two(|| finish(&v2, &mut c2, true, budget));
                let before = in_block_completions();
                let r3 = finish(&v3, &mut c3, true, budget);
                let completed = in_block_completions() - before;
                assert_eq!((r1.1, &r1.2), (r3.1, &r3.2), "{tag}");
                assert_eq!((r2.1, &r2.2), (r3.1, &r3.2), "{tag}");
                assert_eq!(r1.0.effects, r3.0.effects, "{tag}");
                assert_eq!(r2.0.effects, r3.0.effects, "{tag}");
                assert_eq!(r2.0.requests, r3.0.requests, "{tag}");
                assert_same_machine(&tag, (&v1, &c1), (&v3, &c3));
                assert_same_machine(&tag, (&v2, &c2), (&v3, &c3));
                // The twenty-bit walk keeps the handoff; the others complete
                // the scalar transfers of shared-walk blocks inside them.
                if model == CpuModel::I8086 {
                    assert_eq!(completed, 0, "{tag}");
                } else if scenario <= 1 {
                    assert!(completed >= 6, "{tag}: {completed}");
                }
            }
        }
    }
}

/// A handler that rewrites the next instruction makes its RAM write an effect:
/// the walk issues the ordinary handoff, which settles the same callback, and
/// the rewritten instruction executes. Equal to an embedder that writes the
/// code at the Exit, and to stage 2.
#[test]
fn in_block_handler_code_write_matches_exit() {
    // mov [si],al; inc bx (rewritten to dec bx); mov [si],al; inc bx; hlt
    let code = [0x88, 0x04, 0x43, 0x88, 0x04, 0x43, 0xf4];
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for budget in [1, 2, 100] {
            let tag = format!("{model:?}/{budget}");
            let (v1, mut c1) = setup(model, 0, &code);
            let mut writes = 0;
            loop {
                let run = c1.run_steps(&v1, budget, StepOptions::default()).unwrap();
                match run.into_stop() {
                    StepStop::Exit(Exit::BudgetExhausted) => {}
                    StepStop::Exit(Exit::MmioWrite { .. }) => {
                        if writes == 0 {
                            v1.write_bytes(CODE + 2, &[0x4b]).unwrap();
                        }
                        writes += 1;
                        c1.complete_mmio_write();
                    }
                    StepStop::Exit(Exit::Hlt) => break,
                    stop => panic!("{tag}: {stop:?}"),
                }
            }
            let handler = |two: bool| {
                let (vm, mut cpu) = setup(model, 0, &code);
                let mut calls = Vec::new();
                let mut h = Callback(|r: &DeviceRequest, c: &mut DeviceContext<'_>| {
                    if calls.is_empty() {
                        c.write_ram(CODE + 2, &[0x4b]).unwrap();
                    }
                    calls.push(r.clone());
                    DeviceResult::Handled(0)
                });
                let mut go = || loop {
                    let r = run(&mut cpu, &vm, budget, &mut h);
                    match r.into_stop() {
                        StepStop::Exit(Exit::BudgetExhausted) => {}
                        StepStop::Exit(Exit::Hlt) => break,
                        stop => panic!("{tag}: {stop:?}"),
                    }
                };
                if two {
                    stage_two(go);
                } else {
                    go();
                }
                (vm, cpu, calls)
            };
            let (v2, c2, calls2) = handler(true);
            let before = in_block_counts();
            let (v3, c3, calls3) = handler(false);
            let after = in_block_counts();
            // The writing callback is made inside its block and settles at
            // the handoff. A page with written code then runs the checked
            // walk, which takes the stage-2 handoff.
            assert_eq!((after.0 - before.0, after.1 - before.1), (1, 0), "{tag}");
            assert_eq!(writes, 2, "{tag}");
            assert_eq!(calls2, calls3, "{tag}");
            assert_eq!(calls3.len(), 2, "{tag}");
            assert_eq!(c3.reg(Reg::Rbx), 0, "{tag}");
            assert_same_machine(&tag, (&v1, &c1), (&v3, &c3));
            assert_same_machine(&tag, (&v2, &c2), (&v3, &c3));
        }
    }
}

/// Fallback, DeviceFault and a panic inside a block call the handler once and
/// keep stage 2's results: the ordinary Exit, a latched fault with its
/// request, and terminal poisoning.
#[test]
fn in_block_declined_results_call_once() {
    for result in 0..3 {
        let (vm, mut cpu) = setup(CpuModel::I80286, 0, &[0x43, 0x88, 0x04, 0x43, 0xf4]);
        let mut calls = 0;
        let mut h = Callback(|_: &DeviceRequest, _: &mut DeviceContext<'_>| {
            calls += 1;
            match result {
                0 => DeviceResult::Fallback,
                1 => DeviceResult::DeviceFault(DeviceError::new(5)),
                _ => panic!("device failed"),
            }
        });
        let before = in_block_counts();
        let r = run(&mut cpu, &vm, 100, &mut h);
        assert_eq!(in_block_counts().0 - before.0, 1);
        assert_eq!(in_block_counts().1, before.1);
        assert_eq!(r.steps(), 1);
        match result {
            0 => assert!(matches!(r.stop(), StepStop::Exit(Exit::MmioWrite { .. }))),
            1 => {
                assert!(matches!(
                    r.stop(),
                    StepStop::Exit(Exit::Device(crate::DeviceStop::Fault { .. }))
                ));
                let request = r.request().unwrap();
                assert_eq!(request.instruction_ip, 1);
                assert_eq!(request.position.retired_instructions, 1);
            }
            _ => {
                assert!(matches!(
                    r.stop(),
                    StepStop::Exit(Exit::Device(crate::DeviceStop::HandlerFailed))
                ));
                assert!(vm.device_poisoned());
            }
        }
        assert_eq!(calls, 1);
        assert_eq!(cpu.cpu.rip, 1);
    }
}

/// A handler that runs another vCPU of its VM from inside a callback breaks
/// decision-28's contract, but safe code must stay sound: the hook armed for
/// the outer walk is disarmed while its handler runs, so the nested walk of the
/// same cached block takes the ordinary handoff instead of calling the running
/// handler again. Both vCPUs end as the Exit path leaves them.
#[test]
fn in_block_handler_running_another_vcpu_is_not_reentered() {
    // mov [si],al; inc bx; mov [si],al; inc bx; hlt
    let code = [0x88, 0x04, 0x43, 0x88, 0x04, 0x43, 0xf4];
    for model in [CpuModel::I80286, CpuModel::I80386, CpuModel::I80486Dx] {
        let (v1, mut c1) = setup(model, 0, &code);
        let exit = finish(&v1, &mut c1, false, 100);
        let (vm, mut cpu) = setup(model, 0, &code);
        let mut nested = vm.new_vcpu();
        nested.cpu = cpu.cpu.clone();
        let depth = std::cell::Cell::new(0);
        let mut deepest = 0;
        let mut calls = Vec::new();
        let mut nested_exits = Vec::new();
        let mut h = Callback(|r: &DeviceRequest, _: &mut DeviceContext<'_>| {
            depth.set(depth.get() + 1);
            deepest = deepest.max(depth.get());
            calls.push((r.vcpu_id, r.position.retired_instructions));
            if depth.get() == 1 && calls.len() == 1 {
                loop {
                    let r = nested.run_steps(&vm, 100, StepOptions::default()).unwrap();
                    match r.into_stop() {
                        StepStop::Exit(Exit::BudgetExhausted) => {}
                        StepStop::Exit(Exit::MmioWrite { addr, .. }) => {
                            nested_exits.push(addr);
                            nested.complete_mmio_write();
                        }
                        StepStop::Exit(Exit::Hlt) => break,
                        stop => panic!("{model:?}: {stop:?}"),
                    }
                }
            }
            depth.set(depth.get() - 1);
            DeviceResult::Handled(0)
        });
        let before = in_block_counts();
        let r = run(&mut cpu, &vm, 100, &mut h);
        let after = in_block_counts();
        assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{r:?}");
        let id = cpu.device_cpu().id;
        assert_eq!(deepest, 1, "{model:?}");
        assert_eq!(calls, [(id, 0), (id, 2)], "{model:?}");
        assert_eq!(nested_exits, [MMIO, MMIO], "{model:?}");
        assert_eq!(
            (after.0 - before.0, after.1 - before.1),
            (2, 2),
            "{model:?}"
        );
        assert_eq!(exit.1, 5, "{model:?}");
        assert_same_machine(&format!("{model:?}"), (&v1, &c1), (&vm, &cpu));
        assert_eq!(architectural(&nested), architectural(&c1), "{model:?}");
    }
}

/// A result kept for the handoff settles only the transfer it was made for;
/// any other handoff drops it and takes the ordinary path.
#[test]
fn decided_result_settles_only_its_own_transfer() {
    let (vm, mut cpu) = setup(CpuModel::I80286, 0, &[0x88, 0x04, 0xf4]);
    let mut seen = Vec::new();
    let mut h = Callback(|r: &DeviceRequest, _: &mut DeviceContext<'_>| {
        seen.push(r.address);
        DeviceResult::Fallback
    });
    let request = |address| DeviceRequest {
        space: DeviceSpace::Mmio,
        address,
        direction: DeviceDirection::Write,
        size: 1,
        value: 0x5a,
        total_size: 1,
        part: 0,
        linear_address: None,
        instruction_ip: 0,
        code_segment: 0x1000,
        event: None,
        vcpu_id: cpu.device_cpu().id,
        position: DevicePosition::default(),
        unit_start: DevicePosition::default(),
        cycles: None,
    };
    let mut devices = Run {
        handler: Some(&mut h),
        decided: Some(Box::new(Decided {
            request: request(MMIO),
            result: Ok(DeviceResult::Handled(0)),
            effects: Effects::new(None),
        })),
        ..Run::default()
    };
    let other = request(MMIO + 4).handoff();
    assert!(devices.access(&mut cpu, &vm, 0, &other).is_none());
    assert!(devices.decided.is_none());
    drop(devices);
    assert_eq!(seen, [MMIO + 4]);
}

/// A run without a handler still reports a VM poisoned during it at an issued
/// handoff, whatever the registration.
#[test]
fn poisoned_handoff_is_served_without_a_handler() {
    let (vm, _) = setup(CpuModel::I80286, 0, &[0xf4]);
    let devices = Run::default();
    let unregistered = Exit::port_io(0x80, 1, PortDir::Out, 0);
    let registered = Exit::MmioWrite {
        addr: MMIO,
        size: 1,
        value: 0,
        total_size: 1,
        part: 0,
    };
    assert!(!devices.serves(&vm, &registered));
    vm.devices().poison_execution();
    assert!(devices.serves(&vm, &registered));
    assert!(devices.serves(&vm, &unregistered));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Effect {
    None,
    Intr,
    Deadline,
    A20,
    HostStop,
    Nmi,
}
/// Run to HLT or IntrReady: with a handler applying `effect` at its first
/// callback, or as an embedder applying it at the first Exit. Returns the
/// stops other than budget and MMIO handoffs, the steps, and the requests.
fn effect_run(
    vm: &Vm,
    cpu: &mut Vcpu,
    budget: u64,
    effect: Effect,
    handler: bool,
) -> (Vec<String>, u64, Vec<DeviceRequest>) {
    let first = std::cell::Cell::new(true);
    let mut requests = Vec::new();
    let mut stops = Vec::new();
    let mut steps = 0;
    let mut h = Callback(|r: &DeviceRequest, c: &mut DeviceContext<'_>| {
        requests.push(r.clone());
        if first.replace(false) {
            match effect {
                Effect::None => {}
                Effect::Intr => c.set_intr(true),
                Effect::Deadline => c.set_deadline(DeviceDeadline::Steps(3)).unwrap(),
                Effect::A20 => c.set_a20_enabled(false),
                Effect::HostStop => c.request_host_stop(),
                Effect::Nmi => c.assert_nmi(),
            }
        }
        DeviceResult::Handled(0)
    });
    for _ in 0..200 {
        let r = if handler {
            run(cpu, vm, budget, &mut h)
        } else {
            cpu.run_steps_with_devices(vm, budget, StepOptions::default(), None)
                .unwrap()
        };
        steps += r.steps();
        match r.into_stop() {
            StepStop::Exit(Exit::BudgetExhausted) => {}
            StepStop::Exit(Exit::MmioWrite { .. }) if !handler => {
                cpu.complete_mmio_write();
                if first.replace(false) {
                    match effect {
                        Effect::None => {}
                        Effect::Intr => cpu.set_intr(true),
                        Effect::Deadline => unreachable!("a deadline is a handler's"),
                        Effect::A20 => cpu.set_a20_enabled(false),
                        Effect::HostStop => cpu.exit_request().request_exit(),
                        Effect::Nmi => cpu.raise_nmi().unwrap(),
                    }
                }
            }
            StepStop::Exit(stop @ (Exit::Hlt | Exit::Device(crate::DeviceStop::IntrReady))) => {
                stops.push(format!("{stop:?}"));
                return (stops, steps, requests);
            }
            StepStop::Exit(
                stop @ (Exit::Interrupted | Exit::Device(crate::DeviceStop::Deadline { .. })),
            ) => stops.push(format!("{stop:?}")),
            stop => panic!("{stop:?}"),
        }
    }
    panic!("did not finish")
}

/// Handled results with a context effect (INTR, a deadline, an A20 request, a
/// host stop, an NMI) are made inside the block and settled at the ordinary
/// handoff; a result with none completes in the block. Both equal the stage-2
/// handoff and, for the effects an embedder can apply, the Exit path: stops,
/// steps, state, RAM and delivery logs, with and without single-step traps
/// after the handled transfers.
#[test]
fn in_block_effects_and_single_step_take_the_handoff_and_equal_exit() {
    // inc bx; mov [si],al; inc bx; mov [si],al; inc bx; inc bx; jmp $+2;
    // then, in a block of its own, pushf; pop ax; and ah,0xfe; push ax; popf
    // (TF=0, so HLT stops); hlt
    let code = [
        0x43, 0x88, 0x04, 0x43, 0x88, 0x04, 0x43, 0x43, 0xeb, 0x00, 0x9c, 0x58, 0x80, 0xe4, 0xfe,
        0x50, 0x9d, 0xf4,
    ];
    let effects = [
        Effect::None,
        Effect::Intr,
        Effect::Deadline,
        Effect::A20,
        Effect::HostStop,
        Effect::Nmi,
    ];
    for model in [CpuModel::I80286, CpuModel::I80386, CpuModel::I80486Dx] {
        for tf in [false, true] {
            for effect in effects {
                for budget in [1, 2, 100] {
                    let tag = format!("{model:?}/tf={tf}/{effect:?}/{budget}");
                    let machine = || {
                        let (vm, mut cpu) = setup(model, 0, &code);
                        // #DB at 0000:0600 and NMI at 0000:0700: INC DX; IRET.
                        vm.write_bytes(4, &[0x00, 0x06, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00])
                            .unwrap();
                        vm.write_bytes(0x600, &[0x42, 0xcf]).unwrap();
                        vm.write_bytes(0x700, &[0x42, 0xcf]).unwrap();
                        cpu.cpu.flags.if_ = true;
                        if tf {
                            cpu.cpu.rflags_system |= 0x100;
                        }
                        (vm, cpu)
                    };
                    let (v2, mut c2) = machine();
                    let r2 = stage_two(|| effect_run(&v2, &mut c2, budget, effect, true));
                    let (v3, mut c3) = machine();
                    let before = in_block_counts();
                    let r3 = effect_run(&v3, &mut c3, budget, effect, true);
                    let after = in_block_counts();
                    let (calls, completed) = (after.0 - before.0, after.1 - before.1);
                    assert_eq!(r2, r3, "{tag}");
                    assert_same_machine(&tag, (&v2, &c2), (&v3, &c3));
                    assert!(calls >= 1, "{tag}");
                    let declined = u64::from(effect != Effect::None);
                    assert_eq!(calls - completed, declined, "{tag}");
                    if effect != Effect::Deadline {
                        let (v1, mut c1) = machine();
                        let r1 = effect_run(&v1, &mut c1, budget, effect, false);
                        assert_eq!((&r1.0, r1.1), (&r3.0, r3.1), "{tag}");
                        assert_same_machine(&tag, (&v1, &c1), (&v3, &c3));
                    }
                }
            }
        }
    }
}
