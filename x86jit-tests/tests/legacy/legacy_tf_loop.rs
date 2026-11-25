//! Single-step through a `DEC BX / JNZ` loop on the Machine profile, real mode.
//!
//! 80286 PRM §5.2.1 and Table 5-2 (vector 1, single step): the trap follows
//! each instruction that began with TF set, TF and IF are cleared on handler
//! entry after FLAGS is pushed, and IRET restores the pushed FLAGS. SDM Vol 3B
//! §20.3.1.4 p. 20-11: an instruction that sets TF (POPF here) is not itself followed
//! by a trap; the next instruction is. The pushed FLAGS image carries the
//! trapped instruction's arithmetic result, so ZF from DEC reaches JNZ.
//! 8086 UM §2.6 p. 2-28: the trap follows every instruction that began with
//! TF set, so the POPF that clears TF is trapped and the HLT after it is not.
use x86jit_core::{
    CpuModel, ExceptionDelivery, Exit, InterpreterBackend, Prot, RegionKind, StepOptions, StepStop,
    Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const MODELS: [CpuModel; 3] = [CpuModel::I8086, CpuModel::I80286, CpuModel::I80386];
const IP: u16 = 0x1000;
const HANDLER: u16 = 0x2000;
const COUNT: u64 = 0x600;
const LOG_IP: u64 = 0x610;
const LOG_FLAGS: u64 = 0x810;

// PUSHF; POP AX; OR AX,0100h; PUSH AX; POPF; MOV BX,3; L: DEC BX; JNZ L;
// PUSHF; POP AX; AND AX,0FEFFh; PUSH AX; POPF; HLT
const MAIN: &[u8] = b"\x9c\x58\x0d\x00\x01\x50\x9d\xbb\x03\x00\x4b\x75\xfd\
\x9c\x58\x25\xff\xfe\x50\x9d\xf4";
const MOV_BX: u16 = IP + 7;
const DEC: u16 = IP + 10;
const JNZ: u16 = IP + 11;
const TAIL: u16 = IP + 13;
const HLT: u16 = IP + 20;

#[derive(Clone, Copy, Debug)]
enum Handler {
    /// IRET only.
    Iret,
    /// INC WORD [600h]; IRET — touches no register; the INC's flags are
    /// discarded by IRET.
    Count,
    /// Saves SI/AX/BP, logs the frame's IP and FLAGS, restores, IRET.
    Log,
}

fn handler_code(handler: Handler) -> &'static [u8] {
    match handler {
        Handler::Iret => b"\xcf",
        Handler::Count => b"\xff\x06\x00\x06\xcf",
        Handler::Log => LOG_HANDLER,
    }
}

// PUSH SI; PUSH AX; PUSH BP; MOV BP,SP; MOV SI,[600h]; MOV AX,[BP+6];
// MOV [SI+610h],AX; MOV AX,[BP+10]; MOV [SI+810h],AX; ADD WORD [600h],2;
// POP BP; POP AX; POP SI; IRET
const LOG_HANDLER: &[u8] = b"\x56\x50\x55\x89\xe5\x8b\x36\x00\x06\x8b\x46\x06\x89\x84\x10\x06\
\x8b\x46\x0a\x89\x84\x10\x08\x83\x06\x00\x06\x02\x5d\x58\x5e\xcf";

/// Instructions the handler retires per trap.
fn handler_len(handler: Handler) -> u64 {
    match handler {
        Handler::Iret => 1,
        Handler::Count => 2,
        Handler::Log => 14,
    }
}

fn setup(model: CpuModel, backend: u8, handler: Handler) -> (Vm, Vcpu) {
    let config = VmConfig::flat(0x20000);
    // Interpreter; JIT compiling eagerly; JIT tiering up after two runs.
    let mut vm = match backend {
        0 => Vm::with_backend(config, Box::new(InterpreterBackend)),
        _ => Vm::with_backend(config, Box::new(JitBackend::new())),
    };
    if backend == 2 {
        vm.set_tier_up_after(Some(2));
    }
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::InGuest);
    vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(u64::from(IP), MAIN).unwrap();
    vm.write_bytes(u64::from(HANDLER), handler_code(handler))
        .unwrap();
    // IVT vector 1 -> 0000:2000.
    vm.write_bytes(4, &u32::from(HANDLER).to_le_bytes())
        .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = u64::from(IP);
    cpu.cpu.gpr[3] = 0x1111;
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}

/// Frame IP of each trap in order (the next instruction) and, inside the
/// loop, ZF as JNZ sees it: MOV BX,3, (DEC, JNZ) x3, then the five tail
/// instructions up to and including the POPF that clears TF.
fn expected_traps() -> Vec<(u16, Option<bool>)> {
    let mut v = vec![(DEC, Some(false))];
    for bx in [2u16, 1, 0] {
        v.push((JNZ, Some(bx == 0)));
        v.push((if bx == 0 { TAIL } else { DEC }, Some(bx == 0)));
    }
    for next in [TAIL + 1, TAIL + 2, TAIL + 5, TAIL + 6, HLT] {
        v.push((next, None));
    }
    v
}

fn tf(cpu: &Vcpu) -> bool {
    cpu.cpu.rflags_system & 0x100 != 0
}

fn label(model: CpuModel, backend: u8, handler: Handler) -> String {
    format!("{model:?} backend={backend} {handler:?}")
}

/// Main instructions before HLT: 5 setup + MOV + 6 loop + 5 tail.
const MAIN_BEFORE_HLT: u64 = 17;
const TRAPS: u64 = 12;

fn check_end(vm: &Vm, cpu: &Vcpu, handler: Handler, label: &str) {
    assert_eq!(cpu.cpu.gpr[3] & 0xffff, 0, "{label}: BX");
    assert_eq!(cpu.cpu.cs, 0, "{label}");
    assert_eq!(cpu.cpu.rip, u64::from(HLT) + 1, "{label}: halted after HLT");
    assert_eq!(cpu.cpu.gpr[4] & 0xffff, 0x8000, "{label}: SP balanced");
    assert!(!tf(cpu), "{label}: TF cleared by the tail POPF");
    match handler {
        Handler::Iret => {}
        Handler::Count => {
            assert_eq!(vm.mem.read(COUNT, 2).unwrap(), TRAPS, "{label}: traps")
        }
        Handler::Log => {
            assert_eq!(vm.mem.read(COUNT, 2).unwrap(), TRAPS * 2, "{label}");
            for (i, (ip, zf)) in expected_traps().into_iter().enumerate() {
                let at = 2 * i as u64;
                let frame_ip = vm.mem.read(LOG_IP + at, 2).unwrap();
                let frame_flags = vm.mem.read(LOG_FLAGS + at, 2).unwrap();
                assert_eq!(frame_ip, u64::from(ip), "{label}: trap {i} IP");
                let tf_after = ip != HLT;
                assert_eq!(frame_flags & 0x100 != 0, tf_after, "{label}: trap {i} TF");
                if let Some(zf) = zf {
                    assert_eq!(frame_flags & 0x40 != 0, zf, "{label}: trap {i} ZF");
                }
            }
        }
    }
    assert_eq!(
        cpu.retired_instructions(),
        MAIN_BEFORE_HLT + 1 + TRAPS * handler_len(handler),
        "{label}: retired"
    );
}

fn run_to_halt(vm: &Vm, cpu: &mut Vcpu, budget: u64, label: &str) {
    let mut total = 0;
    for _ in 0..10_000 {
        let r = cpu.run_steps(vm, budget, StepOptions::default()).unwrap();
        total += r.steps();
        assert!(r.steps() <= budget, "{label}");
        match r.stop() {
            StepStop::Exit(Exit::BudgetExhausted) => assert_eq!(r.steps(), budget, "{label}"),
            StepStop::Exit(Exit::Hlt) => {
                assert_eq!(total, cpu.retired_instructions(), "{label}: steps");
                return;
            }
            other => panic!("{label}: {other:?}"),
        }
        // BX never wraps past zero.
        let bx = cpu.cpu.gpr[3] & 0xffff;
        assert!(bx <= 3 || bx == 0x1111, "{label}: BX={bx:#x}");
    }
    panic!("{label}: loop did not exit");
}

#[test]
fn tf_dec_jnz_loop_exits_for_every_budget() {
    for model in MODELS {
        for backend in 0..3 {
            for handler in [Handler::Iret, Handler::Count, Handler::Log] {
                for budget in [1, 2, 3, 7, 1_000_000] {
                    let l = format!("{} budget={budget}", label(model, backend, handler));
                    let (vm, mut cpu) = setup(model, backend, handler);
                    run_to_halt(&vm, &mut cpu, budget, &l);
                    check_end(&vm, &cpu, handler, &l);
                }
            }
        }
    }
}

#[test]
fn tf_dec_jnz_loop_step_by_step_trap_delivery() {
    // Budget 1: each step retires one instruction; trap delivery retires none
    // and leaves CS:IP at the handler with TF and IF clear.
    for model in MODELS {
        for backend in 0..3 {
            let handler = Handler::Count;
            let l = label(model, backend, handler);
            let (vm, mut cpu) = setup(model, backend, handler);
            let step = |cpu: &mut Vcpu| {
                let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                assert_eq!(r.steps(), 1, "{l}: {:?}", r.stop());
                r
            };
            for _ in 0..5 {
                step(&mut cpu);
            }
            // POPF set TF; no trap after it.
            assert!(tf(&cpu), "{l}");
            assert_eq!(cpu.cpu.rip, u64::from(MOV_BX), "{l}: no trap after POPF");
            let mut expected = expected_traps().into_iter();
            let mut traps = 0;
            for _ in 0..TRAPS {
                // One main instruction, then the trap is delivered.
                step(&mut cpu);
                let (next, zf) = expected.next().unwrap();
                assert_eq!(cpu.cpu.cs, 0, "{l}");
                assert_eq!(cpu.cpu.rip, u64::from(HANDLER), "{l}: in handler");
                assert!(!tf(&cpu) && !cpu.cpu.flags.if_, "{l}: TF/IF cleared");
                let sp = cpu.cpu.gpr[4] & 0xffff;
                // The tail's PUSHF and PUSH AX leave one word on the stack.
                let pushed = if [TAIL + 1, TAIL + 6].contains(&next) {
                    2
                } else {
                    0
                };
                assert_eq!(sp, 0x8000 - 6 - pushed, "{l}");
                assert_eq!(vm.mem.read(sp, 2).unwrap(), u64::from(next), "{l}");
                let flags = vm.mem.read(sp + 4, 2).unwrap();
                assert_eq!(flags & 0x100 != 0, next != HLT, "{l}: pushed TF");
                if let Some(zf) = zf {
                    assert_eq!(flags & 0x40 != 0, zf, "{l}: pushed ZF");
                }
                // INC [600h]; IRET runs untrapped.
                step(&mut cpu);
                step(&mut cpu);
                traps += 1;
                assert_eq!(cpu.cpu.rip, u64::from(next), "{l}: back");
                // IRET restores the pushed TF; the trap after the clearing
                // POPF returns with it clear.
                assert_eq!(tf(&cpu), flags & 0x100 != 0, "{l}: IRET restores TF");
                assert_eq!(vm.mem.read(COUNT, 2).unwrap(), traps, "{l}");
            }
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(matches!(r.stop(), StepStop::Exit(Exit::Hlt)), "{l}");
            check_end(&vm, &cpu, handler, &l);
        }
    }
}
