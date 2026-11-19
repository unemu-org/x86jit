//! Faults raised by event delivery, and delivery that retires nothing, on the
//! I80286 and I80386 Machine models.
//!
//! Escalation: a contributory fault while delivering a contributory fault is
//! a double fault, #DF(0); a fault while delivering #DF is shutdown (I286 PRM
//! §9.6.2 p. 9-10; 80386 PRM §9.8.8 Tables 9-3/9-4 p. 9-13; SDM Vol 3A §7.15
//! Event 8, Tables 7-4/7-5 pp. 7-36–37). A real-address interrupt whose frame
//! word crosses FFFFh shuts down (I286 PRM App. B INT p. B-50; 80386 PRM §17
//! INT p. 17-73).
//!
//! Liveness: delivery costs no unit, so a handler whose first instruction
//! faults is entered again and again with nothing retired. On a stack that
//! never runs out, only the engine's delivery bound returns the run, so every
//! run here is watched by a timeout rather than allowed to hang the suite.
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;
use x86jit_core::{
    CpuModel, CpuState, ExceptionDelivery, Exit, Prot, RegionKind, SegmentCache, SegmentId,
    StepOptions, StepRun, StepStop, Vcpu, Vm, VmConfig,
};

const MODELS: [CpuModel; 2] = [CpuModel::I80286, CpuModel::I80386];
const DELIVERIES: [ExceptionDelivery; 2] = [ExceptionDelivery::InGuest, ExceptionDelivery::Outward];
/// The engine's bound on consecutive deliveries that complete no unit
/// (`vm/legacy.rs` `DELIVERY_STALL_LIMIT`, an engine choice).
const STALL: u64 = 1024;
const BUDGET: u64 = 100_000;
const WATCHDOG: Duration = Duration::from_secs(30);
/// 0F FF is unassigned on both models: #UD (I286 PRM §9.6.1 p. 9-10;
/// 80386 PRM §14.6 Table 14-1 p. 14-5).
const UD: &[u8] = b"\x0f\xff";
const MEMORY: u64 = 0x200000;
/// The region a failed delivery must leave untouched: tables, code, stacks.
const WATCHED: usize = 0x30000;

/// Run `f` on its own thread and fail the test if it does not return.
fn watched(label: &str, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    match rx.recv_timeout(WATCHDOG) {
        Ok(()) => {}
        Err(RecvTimeoutError::Timeout) => {
            panic!("{label}: run_steps did not return within {WATCHDOG:?} (delivery livelock)")
        }
        Err(RecvTimeoutError::Disconnected) => match handle.join() {
            Err(panic) => std::panic::resume_unwind(panic),
            Ok(()) => unreachable!("the run ended without reporting"),
        },
    }
}

fn run(vm: &Vm, cpu: &mut Vcpu, budget: u64) -> StepRun {
    cpu.run_steps(vm, budget, StepOptions::default()).unwrap()
}

fn word(vm: &Vm, address: u64) -> u16 {
    let mut bytes = [0; 2];
    vm.read_bytes(address, &mut bytes).unwrap();
    u16::from_le_bytes(bytes)
}

fn machine(model: CpuModel, delivery: ExceptionDelivery) -> Vm {
    let mut vm = Vm::new(VmConfig::flat(MEMORY));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(delivery);
    vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm
}

const REAL_CS: u16 = 0x1000;
const REAL_SS: u16 = 0x2000;
const SS_BASE: u64 = 0x20000;

/// Real-address mode: `code` at 1000:0000, SS = 2000h, every IVT entry ->
/// 0000:3000, which holds `handler`.
fn real(
    model: CpuModel,
    delivery: ExceptionDelivery,
    code: &[u8],
    handler: &[u8],
    sp: u64,
) -> (Vm, Vcpu) {
    let vm = machine(model, delivery);
    vm.write_bytes(u64::from(REAL_CS) << 4, code).unwrap();
    for vector in 0..256u64 {
        vm.write_bytes(vector * 4, &[0x00, 0x30, 0x00, 0x00])
            .unwrap();
    }
    vm.write_bytes(0x3000, handler).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([REAL_CS, 0, 0, REAL_SS, 0, 0]);
    cpu.cpu.rip = 0;
    cpu.cpu.gpr[4] = sp;
    (vm, cpu)
}

fn memory(vm: &Vm) -> Vec<u8> {
    let mut bytes = vec![0; WATCHED];
    vm.read_bytes(0, &mut bytes).unwrap();
    bytes
}

/// Shutdown, with nothing retired and every failed delivery atomic: the
/// faulting state and memory remain.
fn shutdown(label: &str, vm: &Vm, cpu: &mut Vcpu) {
    let before = format!("{:?}", cpu.cpu);
    let image = memory(vm);
    let result = run(vm, cpu, BUDGET);
    assert!(
        matches!(result.stop(), StepStop::Exit(Exit::Shutdown { .. })),
        "{label}: {:?}",
        result.stop()
    );
    assert_eq!(
        (result.steps(), cpu.retired_instructions()),
        (0, 0),
        "{label}"
    );
    assert_eq!(format!("{:?}", cpu.cpu), before, "{label}");
    assert!(memory(vm) == image, "{label}: a failed frame left stores");
}

fn outward(label: &str, vm: &Vm, cpu: &mut Vcpu, vector: u8) {
    let before = format!("{:?}", cpu.cpu);
    let result = run(vm, cpu, BUDGET);
    let StepStop::Exit(Exit::Exception { vector: actual, .. }) = result.stop() else {
        panic!("{label}: {:?}", result.stop());
    };
    assert_eq!(*actual, vector, "{label}");
    assert_eq!(result.steps(), 0, "{label}");
    assert_eq!(format!("{:?}", cpu.cpu), before, "{label}");
}

/// A bounded return: the budget is unspent, nothing retired, and the vCPU
/// sits at the handler's first instruction.
fn stalled(label: &str, result: &StepRun, cpu: &Vcpu, at: (u16, u64)) {
    assert!(
        matches!(result.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{label}: {:?}",
        result.stop()
    );
    assert_eq!(result.steps(), 0, "{label}");
    assert_eq!(cpu.retired_instructions(), 0, "{label}");
    assert_eq!((cpu.cpu.cs, cpu.cpu.rip), at, "{label}");
}

/// Run until something other than the delivery bound stops the run: each
/// bounded return is at the handler's first instruction, `STALL` frames
/// further down the 16-bit stack. A stack from `sp` holds `sp / 6` frames;
/// the next delivery faults and escalates to shutdown. In real-address mode
/// that needs an odd `sp`: an even one wraps at offset 0 instead.
fn bounded_then_shutdown(label: &str, vm: &Vm, cpu: &mut Vcpu, handler: (u16, u64), sp: u64) {
    let frames = sp / 6;
    let mut runs = 0;
    let stop = loop {
        let result = run(vm, cpu, BUDGET);
        runs += 1;
        assert_eq!(result.steps(), 0, "{label}");
        if !matches!(result.stop(), StepStop::Exit(Exit::BudgetExhausted)) {
            break format!("{:?}", result.stop());
        }
        stalled(label, &result, cpu, handler);
        assert_eq!(cpu.cpu.gpr[4] & 0xffff, sp - 6 * STALL * runs, "{label}");
    };
    assert!(stop.starts_with("Exit(Shutdown"), "{label}: {stop}");
    assert_eq!(runs, frames / STALL + 1, "{label}");
    assert_eq!(cpu.cpu.gpr[4] & 0xffff, sp - 6 * frames, "{label}");
    assert_eq!(cpu.retired_instructions(), 0, "{label}");
}

#[test]
fn real_mode_handler_that_faults_at_once_is_bounded_then_shuts_down() {
    // The #UD handler is itself 0F FF. Each run returns after STALL
    // deliveries; the frames step down from SP = 8001h to SP = 3, where a
    // frame word crosses FFFFh: #GP (286) or #SS (386), whose own frame
    // crosses too (#DF), and so does #DF's: shutdown (I286 PRM App. B INT
    // p. B-50; 80386 PRM §17 INT p. 17-73). From an even SP the frames wrap
    // and never stop (the next test).
    for model in MODELS {
        let label = format!("{model:?}");
        let (vm, mut cpu) = real(model, ExceptionDelivery::InGuest, UD, UD, 0x8001);
        watched(&label.clone(), move || {
            bounded_then_shutdown(&label, &vm, &mut cpu, (0, 0x3000), 0x8001);
            assert_eq!(cpu.cpu.gpr[4] & 0xffff, 3, "{label}: SP");
            // The first frame, for the fault at 1000:0000.
            assert_eq!(word(&vm, SS_BASE + 0x7ffb), 0x0000, "{label}: IP");
            assert_eq!(word(&vm, SS_BASE + 0x7ffd), REAL_CS, "{label}: CS");
        });
    }
}

#[test]
fn real_mode_handler_that_faults_at_once_wraps_from_an_even_sp() {
    // From SP = 10h the frames pass SP = 4 and wrap below 0000h: no frame
    // word crosses FFFFh (I286 PRM App. B INT p. B-50; 80386 PRM §17 INT
    // p. 17-73), so only the delivery bound returns each run.
    for model in MODELS {
        let label = format!("{model:?}");
        let (vm, mut cpu) = real(model, ExceptionDelivery::InGuest, UD, UD, 0x10);
        vm.write_bytes(SS_BASE, &[0xaa, 0xaa]).unwrap();
        watched(&label.clone(), move || {
            for runs in 1..=2u64 {
                let result = run(&vm, &mut cpu, BUDGET);
                stalled(&label, &result, &cpu, (0, 0x3000));
                assert_eq!(
                    cpu.cpu.gpr[4],
                    0x10u64.wrapping_sub(6 * STALL * runs) & 0xffff,
                    "{label}"
                );
            }
            // The frame pushed at SP = 4: FLAGS at 0002h, CS at 0000h, IP
            // at FFFEh, for the handler's fault at 0000:3000. A sentinel at
            // 0000h shows that the wrapped CS word was written.
            assert_eq!(word(&vm, SS_BASE), 0, "{label}: CS");
            assert_eq!(word(&vm, SS_BASE + 0xfffe), 0x3000, "{label}: IP");
        });
    }
}

#[test]
fn real_mode_frame_word_across_the_limit_escalates_to_shutdown() {
    // SP = 1, 3, 5: a frame word crosses FFFFh, the delivery of that fault
    // crosses it again (#DF), and so does #DF's: shutdown (I286 PRM App. B
    // INT p. B-50, §9.6.2 p. 9-10; 80386 PRM §17 INT p. 17-73, §9.8.8
    // p. 9-13). The same holds for a reset image whose SS keeps a limit
    // below the stack (a protected-mode or LOADALL cache): every frame
    // overruns it.
    for model in MODELS {
        for delivery in DELIVERIES {
            for code in [&b"\xcd\x21"[..], UD] {
                for (sp, limit) in [(1u64, 0xffff), (3, 0xffff), (5, 0xffff), (0x8000, 0xff)] {
                    let label =
                        format!("{model:?} {delivery:?} {code:02x?} SP={sp:x} limit={limit:x}");
                    let (vm, mut cpu) = real(model, delivery, code, b"\xf4", sp);
                    cpu.cpu.legacy.segments[SegmentId::Ss as usize].limit = limit;
                    let software = code[0] == 0xcd;
                    watched(&label.clone(), move || {
                        if delivery == ExceptionDelivery::InGuest {
                            shutdown(&label, &vm, &mut cpu);
                        } else if software {
                            // INT n keeps guest delivery under Outward; its
                            // stack fault is the exception returned: 13 on
                            // the 286 (I286 PRM App. B MOV p. B-74), 12 on
                            // the 386 (80386 PRM §14.6 Table 14-1 p. 14-5).
                            let vector = if model == CpuModel::I80286 { 13 } else { 12 };
                            outward(&label, &vm, &mut cpu, vector);
                        } else {
                            outward(&label, &vm, &mut cpu, 6);
                        }
                    });
                }
            }
        }
    }
}

const P_CODE: u16 = 0x08;
const P_STACK: u16 = 0x10;
const P_USER_CODE: u16 = 0x1b;
const P_USER_STACK: u16 = 0x23;
const P_TSS: u16 = 0x28;
const P_FLAT_STACK: u16 = 0x30;
const GDT: u64 = 0x2000;
const IDT: u64 = 0x3000;
const TSS: u64 = 0x4000;
const HANDLER: u32 = 0x1800;

fn descriptor(vm: &Vm, selector: u16, base: u32, limit: u32, attr: u16) {
    // 286 PRM §6.3 Fig. 6-3 p. 6-5; 80386 PRM §5.1.1 Fig. 5-3: the
    // eight-byte image, inclusive limit, split base, flags nibble.
    let raw = if attr & 0x8000 != 0 {
        limit >> 12
    } else {
        limit
    };
    let low = (raw & 0xffff) | (base << 16);
    let high = ((base >> 16) & 0xff)
        | (base & 0xff00_0000)
        | (u32::from(attr & 0xf0ff) << 8)
        | (raw & 0xf0000);
    let address = GDT + u64::from(selector & !7);
    vm.write_bytes(address, &low.to_le_bytes()).unwrap();
    vm.write_bytes(address + 4, &high.to_le_bytes()).unwrap();
}

fn cache(base: u32, limit: u32, attributes: u16) -> SegmentCache {
    SegmentCache {
        base,
        limit,
        attributes,
        ..SegmentCache::default()
    }
}

/// Protected mode at CPL 0, `code` at 0008:1000 on a 16-bit stack at
/// 0010:SP. Every IDT gate (`kind`, DPL 0) leads to 0008:1800, which holds
/// `handler`.
fn protected(
    model: CpuModel,
    delivery: ExceptionDelivery,
    code: &[u8],
    handler: &[u8],
    kind: u8,
    sp: u64,
) -> (Vm, Vcpu) {
    let vm = machine(model, delivery);
    vm.write_bytes(0x1000, code).unwrap();
    vm.write_bytes(u64::from(HANDLER), handler).unwrap();
    descriptor(&vm, P_CODE, 0, 0xffff, 0x9b);
    descriptor(&vm, P_STACK, 0, 0xffff, 0x93);
    descriptor(&vm, P_USER_CODE, 0, 0xffff, 0xfb);
    descriptor(&vm, P_USER_STACK, 0, 0xffff, 0xf3);
    for vector in 0..32u64 {
        // 286 PRM §9.4 Fig. 9-3 p. 9-4; 80386 PRM §9.5 Fig. 9-3 p. 9-7.
        let low = (u32::from(P_CODE) << 16) | (HANDLER & 0xffff);
        let high = (HANDLER & 0xffff_0000) | (u32::from(kind) << 8);
        vm.write_bytes(IDT + vector * 8, &low.to_le_bytes())
            .unwrap();
        vm.write_bytes(IDT + vector * 8 + 4, &high.to_le_bytes())
            .unwrap();
    }
    let mut cpu = vm.new_vcpu();
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.gdtr.base = GDT as u32;
    cpu.cpu.legacy.gdtr.limit = 0x3f;
    cpu.cpu.legacy.idtr.base = IDT as u32;
    cpu.cpu.legacy.idtr.limit = 32 * 8 - 1;
    cpu.cpu.legacy.cpl = 0;
    cpu.cpu.cs = P_CODE;
    cpu.cpu.ss = P_STACK;
    cpu.cpu.ds = P_STACK;
    cpu.cpu.es = P_STACK;
    cpu.cpu.legacy.segments[SegmentId::Cs as usize] = cache(0, 0xffff, 0x9b);
    for segment in [SegmentId::Ss, SegmentId::Ds, SegmentId::Es] {
        cpu.cpu.legacy.segments[segment as usize] = cache(0, 0xffff, 0x93);
    }
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = sp;
    (vm, cpu)
}

#[test]
fn protected_frame_with_no_room_escalates_to_shutdown() {
    // SP = 4 leaves no room for a 6-byte frame: #UD's delivery raises #SS(0),
    // handled serially after the benign #UD; #SS's own frame faults again,
    // a double fault, and #DF's frame faults: shutdown (I286 PRM App. B INT
    // p. B-49 "room for 6 bytes", §9.6.2 p. 9-10; 80386 PRM §9.8.8
    // Tables 9-3/9-4 p. 9-13).
    for model in MODELS {
        for delivery in DELIVERIES {
            let label = format!("{model:?} {delivery:?}");
            let (vm, mut cpu) = protected(model, delivery, UD, b"\xf4", 0x86, 4);
            watched(&label.clone(), move || {
                if delivery == ExceptionDelivery::InGuest {
                    shutdown(&label, &vm, &mut cpu);
                } else {
                    outward(&label, &vm, &mut cpu, 6);
                }
            });
        }
    }
}

#[test]
fn protected_handler_that_faults_at_once_is_bounded_then_shuts_down() {
    // As in real mode, on a 16-bit protected stack: its overrun is #SS(0),
    // handled serially after the benign #UD, then #DF(0), then shutdown.
    for model in MODELS {
        let label = format!("{model:?}");
        let (vm, mut cpu) = protected(model, ExceptionDelivery::InGuest, UD, UD, 0x86, 0x8000);
        // The stack at 20000h keeps the frames clear of the tables.
        descriptor(&vm, P_STACK, SS_BASE as u32, 0xffff, 0x93);
        cpu.cpu.legacy.segments[SegmentId::Ss as usize] = cache(SS_BASE as u32, 0xffff, 0x93);
        watched(&label.clone(), move || {
            bounded_then_shutdown(&label, &vm, &mut cpu, (P_CODE, u64::from(HANDLER)), 0x8000);
        });
    }
}

#[test]
fn i386_flat_stack_handler_that_faults_at_once_returns_after_the_delivery_bound() {
    // A 4 GiB expand-up stack (G = B = 1) never runs out of room: without
    // the bound the run delivers until its frames overwrite the tables below
    // them. The stack base places ESP = FFF80000h at linear 10000h (80386
    // PRM §5.1 p. 5-2: the linear address is the 32-bit sum of base and
    // offset).
    let label = "I80386 flat stack";
    let (vm, mut cpu) = protected(
        CpuModel::I80386,
        ExceptionDelivery::InGuest,
        UD,
        UD,
        0x8e,
        0xfff8_0000,
    );
    descriptor(&vm, P_FLAT_STACK, 0x0009_0000, 0xffff_ffff, 0xc093);
    cpu.cpu.ss = P_FLAT_STACK;
    cpu.cpu.legacy.segments[SegmentId::Ss as usize] = cache(0x0009_0000, 0xffff_ffff, 0xc093);
    watched(label, move || {
        for pass in 1..=2 {
            let result = run(&vm, &mut cpu, BUDGET);
            stalled(label, &result, &cpu, (P_CODE, u64::from(HANDLER)));
            // A 32-bit gate pushes EFLAGS, CS and EIP as doublewords.
            assert_eq!(cpu.cpu.gpr[4], 0xfff8_0000 - 12 * STALL * pass, "{label}");
        }
        // The first frame: EIP 1000h and CS 8, below linear 10000h.
        assert_eq!(word(&vm, 0x10000 - 12), 0x1000, "{label}");
        assert_eq!(word(&vm, 0x10000 - 8), P_CODE, "{label}");
    });
}

#[test]
fn inward_delivery_to_a_bad_inner_stack_escalates_to_shutdown() {
    // CPL 3 #UD through a DPL 0 gate to a ring-0 handler. The TSS ring-0
    // stack is not present (#SS(selector)) or has no room for the 10-byte
    // frame (#SS): the benign #UD is followed serially by #SS, whose own
    // inward delivery faults the same way (#DF(0)), and so does #DF's:
    // shutdown (I286 PRM App. B INT p. B-49, §9.6.2 p. 9-10; 80386 PRM
    // §9.8.8 Tables 9-3/9-4 p. 9-13).
    for model in MODELS {
        for (attr, sp0) in [(0x12u16, 0x8000u32), (0x92, 4)] {
            for delivery in DELIVERIES {
                let label = format!("{model:?} {delivery:?} SS0 attr={attr:x} SP0={sp0:x}");
                let (vm, mut cpu) = protected(model, delivery, UD, b"\xf4", 0x86, 0x8000);
                descriptor(&vm, P_STACK, 0, 0xffff, attr);
                // 286 PRM §8.2 Fig. 8-1 p. 8-2; 80386 PRM §7.1 Fig. 7-1 p. 7-2.
                let (tss_limit, tss_attr, sp_at, ss_at, size) = if model == CpuModel::I80386 {
                    (0x67, 0x8b, TSS + 4, TSS + 8, 4)
                } else {
                    (0x2b, 0x83, TSS + 2, TSS + 4, 2)
                };
                descriptor(&vm, P_TSS, TSS as u32, tss_limit, tss_attr);
                vm.write_bytes(sp_at, &sp0.to_le_bytes()[..size]).unwrap();
                vm.write_bytes(ss_at, &P_STACK.to_le_bytes()).unwrap();
                cpu.cpu.legacy.tr = x86jit_core::legacy::SystemSegment {
                    selector: P_TSS,
                    cache: cache(TSS as u32, tss_limit, tss_attr),
                    ..Default::default()
                };
                cpu.cpu.legacy.cpl = 3;
                cpu.cpu.cs = P_USER_CODE;
                cpu.cpu.ss = P_USER_STACK;
                cpu.cpu.ds = P_USER_STACK;
                cpu.cpu.es = P_USER_STACK;
                cpu.cpu.legacy.segments[SegmentId::Cs as usize] = cache(0, 0xffff, 0xfb);
                for segment in [SegmentId::Ss, SegmentId::Ds, SegmentId::Es] {
                    cpu.cpu.legacy.segments[segment as usize] = cache(0, 0xffff, 0xf3);
                }
                watched(&label.clone(), move || {
                    if delivery == ExceptionDelivery::InGuest {
                        shutdown(&label, &vm, &mut cpu);
                    } else {
                        outward(&label, &vm, &mut cpu, 6);
                    }
                });
            }
        }
    }
}

#[test]
fn a_guest_that_retires_between_faults_spends_its_budget() {
    // The #UD handler retires INC AX and a far JMP back to the fault before
    // the next delivery, so no two deliveries are consecutive and the run
    // spends its whole budget, as without the bound.
    //   0000:3000 inc ax; jmp far 1000:0000
    for model in MODELS {
        let label = format!("{model:?}");
        let (vm, mut cpu) = real(
            model,
            ExceptionDelivery::InGuest,
            UD,
            b"\x40\xea\x00\x00\x00\x10",
            0x8000,
        );
        watched(&label.clone(), move || {
            let budget = 2 * STALL + 1;
            let result = run(&vm, &mut cpu, budget);
            assert!(
                matches!(result.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                "{label}: {:?}",
                result.stop()
            );
            assert_eq!(result.steps(), budget, "{label}");
            assert_eq!(cpu.retired_instructions(), budget, "{label}");
            assert_eq!(cpu.cpu.gpr[0] & 0xffff, STALL + 1, "{label}");
        });
    }
}

#[test]
fn twenty_bit_divide_handler_that_divides_by_zero_returns_after_the_delivery_bound() {
    // Every 8086-family stack byte wraps within its segment (8086 Family UM
    // §2.3 p. 2-12), so the stack never runs out. A type-0 handler whose DIV
    // divides by zero again never retires; before the bound this run did not
    // return.
    //   1000:0000, 0000:3000  div bl   (BL = 0)
    for model in [CpuModel::I8088, CpuModel::NecV20] {
        let label = format!("{model:?}");
        let (vm, mut cpu) = real(
            model,
            ExceptionDelivery::InGuest,
            b"\xf6\xf3",
            b"\xf6\xf3",
            0x8000,
        );
        watched(&label.clone(), move || {
            for pass in 1..=2 {
                let result = run(&vm, &mut cpu, BUDGET);
                stalled(&label, &result, &cpu, (0, 0x3000));
                assert_eq!(
                    cpu.cpu.gpr[4] & 0xffff,
                    0x8000u64.wrapping_sub(6 * STALL * pass) & 0xffff,
                    "{label}"
                );
            }
        });
    }
}
