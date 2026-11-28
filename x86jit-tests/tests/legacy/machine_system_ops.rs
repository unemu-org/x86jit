//! xj-to9.79: compiled Flat32 Machine blocks complete CLI, STI, IN, OUT,
//! PUSHF and POPF, each a singleton block outside the ordinary walk. Each
//! program runs on two machines with the same recording port handler: the
//! reference interprets, the subject compiles (forced, forced with chains
//! off, or the production background configuration). After every call the
//! state, the RAM, the step result, the retired count and the device
//! transcript (every request, with its positions) must agree.
//!
//! The ports cover a handled access, one whose handler raises INTR (its
//! result settles at the re-execution's exit, the handler called once), a
//! Fallback and an unregistered port (both exits to the embedder), and a
//! DeviceFault. Interrupts come either from an INTR line the harness raises
//! at a fixed step interval, or from an embedder interrupt controller that
//! runs with `StepOptions::with_stop_when_irq_ready` until the vCPU can take
//! its request; both meet the STI shadow and the IF a POPF restores. At CPL 3
//! with IOPL 0, IN and CLI raise #GP(0) through the interpreter.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::descriptor;
use iced_x86::code_asm::*;
use x86jit_core::{
    CpuModel, DeviceContext, DeviceError, DeviceIo, DeviceRange, DeviceRequest, DeviceResult,
    DeviceStop, ExceptionDelivery, Exit, MachineJitCounters, MachineJitMode, PortDir, Prot,
    RegionKind, StepOptions, StepStop, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const SIZE: u64 = 0x20000;
const MAIN: u64 = 0x1000;
const HANDLER: u64 = 0x1800;
const IDT: u64 = 0x2800;
const TICKS: u64 = 0x5000;
const VECTOR: u8 = 0x20;

/// Answers reads from the port and the request's position and records every
/// request. Port 61h raises INTR on every fifth read and 64h on every
/// third; 62h falls back; 63h returns DeviceFault on every fourth read.
#[derive(Default)]
struct Ports {
    requests: Vec<DeviceRequest>,
}

impl DeviceIo for Ports {
    fn access(&mut self, request: &DeviceRequest, context: &mut DeviceContext<'_>) -> DeviceResult {
        self.requests.push(request.clone());
        let n = self
            .requests
            .iter()
            .filter(|r| r.address == request.address)
            .count();
        match request.address {
            0x61 if n % 5 == 2 => context.set_intr(true),
            0x64 if n % 3 == 1 => context.set_intr(true),
            0x62 => return DeviceResult::Fallback,
            0x63 if n % 4 == 1 => return DeviceResult::DeviceFault(DeviceError::new(3)),
            _ => {}
        }
        DeviceResult::Handled(request.address ^ request.position.retired_instructions)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Subject {
    Interpreted,
    Forced,
    ForcedChainOff,
    Background,
}

#[derive(Clone, Copy, Default)]
struct Ring {
    user: bool,
    iopl: u8,
}

fn asm(at: u64, build: impl FnOnce(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(32).unwrap();
    build(&mut a);
    a.assemble(at).unwrap()
}

/// The timer handler: count, end of interrupt at port 20h, IRETD.
fn handler() -> Vec<u8> {
    asm(HANDLER, |a| {
        a.push(eax).unwrap();
        a.inc(dword_ptr(TICKS as u32)).unwrap();
        a.mov(al, 0x20).unwrap();
        a.out(0x20, al).unwrap();
        a.pop(eax).unwrap();
        a.iretd().unwrap();
    })
}

/// Every port form and flag instruction in one loop: IN and OUT with imm8
/// and DX at 8, 16 and 32 bits; PUSHFD/CLI/POPFD restoring IF; CLI/STI with
/// the shadow and an STI with IF already set; a POPFD that toggles IOPL;
/// 16-bit PUSHF/POPF. Inside both IF = 0 windows a read of port 64h raises
/// INTR every third time, so the POPFD and the STI that set IF meet an
/// asserted INTR: it is taken after the POPFD and after the instruction
/// that follows the STI (SDM Vol 2B §4.3 STI).
fn program(iterations: u32) -> Vec<u8> {
    asm(MAIN, |a| {
        let mut top = a.create_label();
        a.sti().unwrap();
        a.mov(ebp, iterations).unwrap();
        a.set_label(&mut top).unwrap();
        a.in_(al, 0x60).unwrap();
        a.movzx(ebx, al).unwrap();
        a.add(esi, ebx).unwrap();
        a.mov(dx, 0x3c8).unwrap();
        a.out(dx, al).unwrap();
        a.inc(dx).unwrap();
        a.in_(ax, dx).unwrap();
        a.add(esi, eax).unwrap();
        a.out(dx, eax).unwrap();
        a.in_(eax, 0x61).unwrap();
        a.add(edi, eax).unwrap();
        a.in_(al, 0x62).unwrap();
        a.add(esi, eax).unwrap();
        a.out(0x62, al).unwrap();
        a.in_(al, 0x63).unwrap();
        a.add(edi, eax).unwrap();
        a.out(0x80, al).unwrap();
        a.in_(ax, 0x81).unwrap();
        a.add(esi, eax).unwrap();
        a.pushfd().unwrap();
        a.cli().unwrap();
        a.in_(al, 0x64).unwrap();
        a.add(ecx, esi).unwrap();
        a.xor(ecx, edi).unwrap();
        a.popfd().unwrap();
        a.add(esi, 1).unwrap();
        a.cli().unwrap();
        a.in_(al, 0x64).unwrap();
        a.add(eax, ecx).unwrap();
        a.sti().unwrap();
        a.inc(ebx).unwrap();
        a.sti().unwrap();
        a.pushfd().unwrap();
        a.pop(eax).unwrap();
        a.xor(eax, 0x3000).unwrap();
        a.push(eax).unwrap();
        a.popfd().unwrap();
        a.pushf().unwrap();
        a.add(word_ptr(esp), 1).unwrap();
        a.popf().unwrap();
        a.dec(ebp).unwrap();
        a.jne(top).unwrap();
        a.cli().unwrap();
        a.hlt().unwrap();
    })
}

fn machine(code: &[u8], subject: Subject, ring: Ring) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(SIZE), Box::new(JitBackend::new()));
    vm.map(0, SIZE as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.set_cpu_model(CpuModel::I80386);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    for ports in [0x20..0x22, 0x60..0x65, 0x3c8..0x3ca] {
        vm.register_device_range(DeviceRange::Ports(ports)).unwrap();
    }
    match subject {
        Subject::Interpreted => vm.set_machine_jit_for_test(MachineJitMode::Off),
        Subject::Forced => vm.set_machine_jit_for_test(MachineJitMode::Forced),
        Subject::ForcedChainOff => {
            vm.set_machine_jit_for_test(MachineJitMode::Forced);
            vm.set_machine_chains_for_test(false);
        }
        Subject::Background => {
            vm.set_machine_jit_for_test(MachineJitMode::Off);
            vm.set_tier_up_after(Some(2));
            vm.set_tier_up_background(true);
        }
    }
    vm.write_bytes(MAIN, code).unwrap();
    vm.write_bytes(HANDLER, &handler()).unwrap();
    descriptor(&vm, 0x2008, 0, u32::MAX, 0xc09b);
    descriptor(&vm, 0x2010, 0, u32::MAX, 0xc093);
    // 80386 PRM §9.5: a 32-bit interrupt gate.
    let gate = [HANDLER as u8, (HANDLER >> 8) as u8, 8, 0, 0, 0x8e, 0, 0];
    vm.write_bytes(IDT + u64::from(VECTOR) * 8, &gate).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = MAIN;
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.gdtr.base = 0x2000;
    cpu.cpu.legacy.gdtr.limit = 0x2f;
    cpu.cpu.legacy.idtr.base = IDT as u32;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        cache.base = 0;
        cache.limit = u32::MAX;
        cache.attributes = if i == 0 { 0x409b } else { 0x4093 };
    }
    cpu.cpu.cs = 8;
    cpu.cpu.ds = 16;
    cpu.cpu.es = 16;
    cpu.cpu.ss = 16;
    if ring.user {
        // DPL 3 caches and descriptors, RPL 3 selectors (SDM Vol 3A §5.5).
        cpu.cpu.legacy.cpl = 3;
        for cache in cpu.cpu.legacy.segments.iter_mut() {
            cache.attributes |= 0x60;
        }
        cpu.cpu.cs = 8 | 3;
        cpu.cpu.ds = 16 | 3;
        cpu.cpu.es = 16 | 3;
        cpu.cpu.ss = 16 | 3;
        descriptor(&vm, 0x2008, 0, u32::MAX, 0xc0fb);
        descriptor(&vm, 0x2010, 0, u32::MAX, 0xc0f3);
    }
    // EFLAGS.IOPL, bits 13:12 (SDM Vol 1 §3.4.3.3).
    cpu.cpu.rflags_system |= u64::from(ring.iopl) << 12;
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}

/// How the harness raises interrupts.
#[derive(Clone, Copy, Debug)]
enum Irq {
    /// The INTR line, asserted every this many steps; a run stops at
    /// `IntrReady` and the harness injects the vector.
    Line(u64),
    /// An interrupt controller with a request every this many steps,
    /// injected once `irq_ready`; until then each call runs with
    /// `with_stop_when_irq_ready`.
    Controller(u64),
}

/// Run reference and subject in lockstep with this budget until HLT or an
/// exception exit, which is returned with the subject's counters.
fn lockstep(
    code: &[u8],
    subject: Subject,
    ring: Ring,
    budget: u64,
    irq: Irq,
) -> (Exit, MachineJitCounters, Vcpu) {
    let (bvm, mut b) = machine(code, Subject::Interpreted, ring);
    let (avm, mut a) = machine(code, subject, ring);
    a.enable_class_census_for_test();
    let (mut ad, mut bd) = (Ports::default(), Ports::default());
    let period = match irq {
        Irq::Line(n) | Irq::Controller(n) => n,
    };
    let mut next_tick = period;
    let mut requested = false;
    let mut steps = 0;
    for call in 0..200_000 {
        let what = format!("{subject:?} {irq:?} budget {budget} call {call}");
        if matches!(irq, Irq::Controller(_)) {
            assert_eq!(a.irq_ready(), b.irq_ready(), "{what}");
            if requested && b.irq_ready() {
                a.inject_irq(VECTOR);
                b.inject_irq(VECTOR);
                requested = false;
            }
        }
        let options = StepOptions::default().with_stop_when_irq_ready(requested);
        let ar = a
            .run_steps_with_devices(&avm, budget, options, Some(&mut ad))
            .unwrap();
        let br = b
            .run_steps_with_devices(&bvm, budget, options, Some(&mut bd))
            .unwrap();
        assert_eq!(
            format!("{:?}", ar.stop()),
            format!("{:?}", br.stop()),
            "{what}"
        );
        assert_eq!(ar.steps(), br.steps(), "{what}");
        assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu), "{what}");
        assert_eq!(a.retired_instructions(), b.retired_instructions(), "{what}");
        assert_eq!(
            ad.requests, bd.requests,
            "{what}: device transcripts differ"
        );
        let mut am = vec![0; SIZE as usize];
        let mut bm = vec![0; SIZE as usize];
        avm.read_bytes(0, &mut am).unwrap();
        bvm.read_bytes(0, &mut bm).unwrap();
        assert!(am == bm, "{what}: RAM differs");
        steps += ar.steps();
        match ar.into_stop() {
            StepStop::Exit(Exit::BudgetExhausted) => {}
            StepStop::IrqReady => {}
            StepStop::Exit(Exit::Device(DeviceStop::IntrReady)) => {
                for cpu in [&mut a, &mut b] {
                    cpu.inject_irq(VECTOR);
                    cpu.set_intr(false);
                }
            }
            StepStop::Exit(Exit::Device(DeviceStop::Fault { .. })) => {
                let value = a.retired_instructions() ^ 0x5a;
                assert!(a.complete_device_fault(value), "{what}");
                assert!(b.complete_device_fault(value), "{what}");
            }
            StepStop::Exit(Exit::PortIo { port, dir, .. }) => {
                if dir == PortDir::In {
                    let value = u64::from(port) ^ a.retired_instructions();
                    a.complete_port_in(value);
                    b.complete_port_in(value);
                }
            }
            StepStop::Exit(exit @ (Exit::Hlt | Exit::Exception { .. })) => {
                let c = a.machine_jit_counters_for_test();
                return (exit, c, a);
            }
            other => panic!("{what}: {other:?}"),
        }
        while steps >= next_tick {
            match irq {
                Irq::Line(_) => {
                    a.set_intr(true);
                    b.set_intr(true);
                }
                Irq::Controller(_) => requested = true,
            }
            next_tick += period;
        }
    }
    panic!("{subject:?} {irq:?} budget {budget}: the program did not end");
}

/// The class census rows of the E2 walk refusals that name a lowered
/// instruction: none, for a subject that compiles at the first execution.
fn lowered_walk_refusals(cpu: &Vcpu) -> Vec<String> {
    cpu.class_census_for_test()
        .unwrap()
        .classes
        .into_iter()
        .filter(|((row, class), _)| {
            row.starts_with("E2 walk")
                && ["IN ", "OUT ", "CLI", "STI", "PUSHF", "POPF"]
                    .iter()
                    .any(|name| class.starts_with(name))
        })
        .map(|((row, class), (turns, _))| format!("{row} / {class}: {turns}"))
        .collect()
}

#[test]
fn port_and_flag_instructions_equal_the_interpreter() {
    let code = program(150);
    for subject in [
        Subject::Forced,
        Subject::ForcedChainOff,
        Subject::Background,
    ] {
        for irq in [Irq::Line(97), Irq::Controller(131)] {
            for budget in [1, 2, 3, 7, 13, 64, 100_000] {
                let (exit, c, a) = lockstep(&code, subject, Ring::default(), budget, irq);
                assert!(matches!(exit, Exit::Hlt), "{exit:?}");
                if subject != Subject::Background && budget >= 2 {
                    assert!(c.compiled_retired > 0, "{c:?}");
                    let refused = lowered_walk_refusals(&a);
                    assert!(
                        refused.is_empty(),
                        "{subject:?} {irq:?} {budget}: {refused:?}"
                    );
                }
            }
        }
    }
}

/// The lowered instructions complete in compiled code: their blocks chain,
/// a handled port access continues the block, and the STI and POPF that set
/// IF return to the dispatcher at the boundary after them.
#[test]
fn port_and_flag_instructions_run_compiled() {
    let (exit, c, _) = lockstep(
        &program(150),
        Subject::Forced,
        Ring::default(),
        100_000,
        Irq::Line(97),
    );
    assert!(matches!(exit, Exit::Hlt), "{exit:?}");
    let boundary = MachineJitCounters::CHAIN_END_NAMES
        .iter()
        .position(|name| *name == "boundary")
        .unwrap();
    // Per iteration: the POPFD that restores IF, the STI that sets it and
    // the POPFD that toggles IOPL return at a boundary.
    assert!(c.chain_ends[boundary] >= 3 * 150, "{c:?}");
    assert!(c.chained > 0, "{c:?}");
    // The deferred port accesses: Fallback (two), the unregistered ports
    // (two), the INTR and DeviceFault answers; the handled ones complete.
    assert!(c.helper_defers <= 150 * 6 + 150 / 2, "{c:?}");
    assert!(c.compiled_retired > 150 * 30, "{c:?}");
}

/// CPL 3: with IOPL 3 the instructions run compiled at CPL 3; with IOPL 0
/// the first IN raises #GP(0), and a CLI does, through the interpreter (SDM
/// Vol 2A CLI and IN, "Protected Mode Exceptions"; the 80386 consults the
/// I/O permission bitmap of a 386 TSS, which this TR does not name).
#[test]
fn iopl_denial_at_cpl_3_raises_gp_through_the_interpreter() {
    let user = asm(MAIN, |a| {
        let mut top = a.create_label();
        a.mov(ebp, 40u32).unwrap();
        a.set_label(&mut top).unwrap();
        a.in_(al, 0x60).unwrap();
        a.add(ebx, eax).unwrap();
        a.mov(dx, 0x3c8).unwrap();
        a.out(dx, al).unwrap();
        a.cli().unwrap();
        a.add(ecx, ebx).unwrap();
        a.sti().unwrap();
        a.dec(ebp).unwrap();
        a.jne(top).unwrap();
        // HLT at CPL 3 is #GP(0) (SDM Vol 2A HLT): the end of the program.
        a.hlt().unwrap();
    });
    let cli_first = asm(MAIN, |a| {
        a.cli().unwrap();
        a.hlt().unwrap();
    });
    for subject in [Subject::Forced, Subject::Background] {
        for budget in [1, 3, 64, 100_000] {
            let open = Ring {
                user: true,
                iopl: 3,
            };
            let (exit, c, _) = lockstep(&user, subject, open, budget, Irq::Line(1 << 40));
            assert!(
                matches!(exit, Exit::Exception { vector: 13, .. }),
                "{exit:?}"
            );
            if subject == Subject::Forced && budget >= 64 {
                assert!(c.compiled_retired_cpl3 > 40 * 5, "{c:?}");
            }
            let closed = Ring {
                user: true,
                iopl: 0,
            };
            for code in [&user, &cli_first] {
                let (exit, c, a) = lockstep(code, subject, closed, budget, Irq::Line(1 << 40));
                assert!(
                    matches!(exit, Exit::Exception { vector: 13, .. }),
                    "{exit:?}"
                );
                assert_eq!(a.retired_instructions(), 1 - u64::from(code == &cli_first));
                let _ = c;
            }
        }
    }
}
