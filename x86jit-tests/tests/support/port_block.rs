//! IN and OUT inside a multi-instruction block (xj-50w.22). `IrOp::PortIo`
//! ends its block with IP past the instruction (spec.md §5.2): the exit names
//! the next instruction, and the instructions after it run once the embedder
//! has serviced the port. Each caller supplies the setup of its path.
use x86jit_core::{Exit, PortDir, StepOptions, StepStop, Vcpu, Vm};

/// The 16-bit-operand IN and OUT forms (SDM Vol 2A §3.3 IN, Vol 2B §4.3 OUT):
/// bytes, access size, direction and port (`None` reads DX).
const FORMS: [(&[u8], u8, PortDir, Option<u16>); 8] = [
    (&[0xe4, 0x10], 1, PortDir::In, Some(0x10)),
    (&[0xe5, 0x10], 2, PortDir::In, Some(0x10)),
    (&[0xec], 1, PortDir::In, None),
    (&[0xed], 2, PortDir::In, None),
    (&[0xe6, 0x10], 1, PortDir::Out, Some(0x10)),
    (&[0xe7, 0x10], 2, PortDir::Out, Some(0x10)),
    (&[0xee], 1, PortDir::Out, None),
    (&[0xef], 2, PortDir::Out, None),
];

#[derive(Clone, Copy, Debug)]
enum Drive {
    /// `run_steps` with a budget of one step per call.
    Step,
    /// One `run_steps` call with a budget past the HLT.
    Budget,
    /// `Vcpu::run` with a block budget.
    Run,
    /// TF set, a budget of one step: the trap after the IN or OUT names the
    /// next instruction.
    Trap,
}

fn exit(stop: &StepStop, what: &str) -> Exit {
    match stop {
        StepStop::Exit(exit) => exit.clone(),
        other => panic!("{what}: {other:?}"),
    }
}

/// `INC CX; <form>; INC BX; HLT` at `ip` for every form. `trap` adds the TF
/// leg, which needs a real-mode IVT at 0 and SS:SP = 0:8000h.
pub fn check(setup: impl Fn(&[u8]) -> (Vm, Vcpu), ip: u64, trap: bool) {
    let drives: &[Drive] = if trap {
        &[Drive::Step, Drive::Budget, Drive::Run, Drive::Trap]
    } else {
        &[Drive::Step, Drive::Budget, Drive::Run]
    };
    for form in FORMS {
        for &drive in drives {
            case(&setup, ip, form, drive);
        }
    }
}

fn case(
    setup: &impl Fn(&[u8]) -> (Vm, Vcpu),
    ip: u64,
    (form, size, dir, port): (&[u8], u8, PortDir, Option<u16>),
    drive: Drive,
) {
    let what = format!("{form:02x?} {drive:?}");
    let mut code = vec![0x41];
    code.extend_from_slice(form);
    code.extend_from_slice(&[0x43, 0xf4]);
    let next = ip + 1 + form.len() as u64;
    let (vm, mut cpu) = setup(&code);
    cpu.cpu.gpr[0] = 0x1234;
    cpu.cpu.gpr[1] = 0;
    cpu.cpu.gpr[2] = 0x3f8;
    cpu.cpu.gpr[3] = 0;
    let first = match drive {
        Drive::Step => {
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 1, "{what}");
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                "{what}: {:?}",
                r.stop()
            );
            assert_eq!(cpu.cpu.rip & 0xffff, ip + 1, "{what}");
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 1, "{what}");
            exit(r.stop(), &what)
        }
        Drive::Budget => {
            let r = cpu.run_steps(&vm, 64, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 2, "{what}");
            exit(r.stop(), &what)
        }
        Drive::Run => cpu.run(&vm, Some(16)),
        Drive::Trap => {
            // Vector 1 -> 0000:0500h, a HLT.
            vm.write_bytes(4, &[0x00, 0x05, 0x00, 0x00]).unwrap();
            vm.write_bytes(0x500, &[0xf4]).unwrap();
            cpu.cpu.rip = ip + 1;
            cpu.cpu.rflags_system |= 0x100;
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 1, "{what}");
            exit(r.stop(), &what)
        }
    };
    let Exit::PortIo {
        port: actual_port,
        size: actual_size,
        dir: actual_dir,
        value,
        ..
    } = first
    else {
        panic!("{what}: {first:?}")
    };
    assert_eq!(
        (actual_port, actual_size, actual_dir),
        (port.unwrap_or(0x3f8), size, dir),
        "{what}"
    );
    if dir == PortDir::Out {
        assert_eq!(value, if size == 1 { 0x34 } else { 0x1234 }, "{what}");
    }
    // The exit names the instruction after the IN or OUT.
    assert_eq!(cpu.cpu.rip & 0xffff, next, "{what}");
    if dir == PortDir::In {
        cpu.complete_port_in(0x5aa5);
    }
    let ax = match (dir, size) {
        (PortDir::Out, _) => 0x1234,
        (PortDir::In, 1) => 0x12a5,
        (PortDir::In, _) => 0x5aa5,
    };
    assert_eq!(cpu.cpu.gpr[0] & 0xffff, ax, "{what}");
    match drive {
        Drive::Step => {
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 1, "{what}");
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                "{what}: {:?}",
                r.stop()
            );
            assert_eq!(cpu.cpu.rip & 0xffff, next + 1, "{what}");
            assert_eq!(cpu.cpu.gpr[3] & 0xffff, 1, "{what}");
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 1, "{what}");
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{what}: {:?}",
                r.stop()
            );
        }
        Drive::Budget => {
            let r = cpu.run_steps(&vm, 64, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 2, "{what}");
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{what}: {:?}",
                r.stop()
            );
        }
        Drive::Run => {
            let r = cpu.run(&vm, Some(16));
            assert!(matches!(r, Exit::Hlt), "{what}: {r:?}");
        }
        Drive::Trap => {
            // Delivery takes no step; the handler's HLT takes one.
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(r.steps(), 1, "{what}");
            assert!(
                matches!(r.stop(), StepStop::Exit(Exit::Hlt)),
                "{what}: {:?}",
                r.stop()
            );
            assert_eq!((cpu.cpu.cs, cpu.cpu.rip & 0xffff), (0, 0x501), "{what}");
            assert_eq!(vm.mem.read(0x7ffa, 2).unwrap(), next, "{what}");
            assert_eq!(vm.mem.read(0x7ffc, 2).unwrap(), 0, "{what}");
            assert_eq!(cpu.cpu.gpr[3] & 0xffff, 0, "{what}");
            assert_eq!(cpu.retired_instructions(), 2, "{what}");
            return;
        }
    }
    assert_eq!(cpu.cpu.rip & 0xffff, next + 2, "{what}");
    assert_eq!(cpu.cpu.gpr[0] & 0xffff, ax, "{what}");
    assert_eq!(cpu.cpu.gpr[1] & 0xffff, 1, "{what}");
    assert_eq!(cpu.cpu.gpr[3] & 0xffff, 1, "{what}");
    assert_eq!(cpu.retired_instructions(), 4, "{what}");
}
