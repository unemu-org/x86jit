//! Decision-30 §4, §10: compiled Flat32 blocks reach devices served in
//! process through the decision-28 in-block contract. Each program runs on two
//! machines with the same recording handler: the reference interprets, the
//! subject compiles (forced, or the production background configuration).
//! After every call the state, the RAM, the step result, the retired count and
//! the device transcript (every request, with its position) must agree; a
//! Fallback is answered identically on both, and the handler is called once
//! per transfer.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::descriptor;
use iced_x86::code_asm::*;
use x86jit_core::{
    CpuModel, DeviceContext, DeviceIo, DeviceRange, DeviceRequest, DeviceResult, ExceptionDelivery,
    Exit, MachineJitMode, Prot, RegionKind, StepOptions, StepStop, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const SIZE: u64 = 0x20000;
const MMIO: u64 = 0x9000;

/// Answers reads from the address and the request's position, accepts
/// writes, and falls back at `MMIO + 0xf0`, recording every request.
#[derive(Default)]
struct Recorder {
    requests: Vec<DeviceRequest>,
}

impl DeviceIo for Recorder {
    fn access(&mut self, request: &DeviceRequest, _: &mut DeviceContext<'_>) -> DeviceResult {
        self.requests.push(request.clone());
        if request.address == MMIO + 0xf0 {
            return DeviceResult::Fallback;
        }
        DeviceResult::Handled(request.address ^ request.position.retired_instructions)
    }
}

#[derive(Clone, Copy, Debug)]
enum Subject {
    Interpreted,
    Forced,
    Background,
}

fn machine(code: &[u8], subject: Subject) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(SIZE), Box::new(JitBackend::new()));
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.map(
        MMIO + 0x1000,
        (SIZE - MMIO - 0x1000) as usize,
        Prot::RWX,
        RegionKind::Ram,
    )
    .unwrap();
    vm.set_cpu_model(CpuModel::I80386);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.register_device_range(DeviceRange::Mmio(MMIO..MMIO + 0x100))
        .unwrap();
    match subject {
        Subject::Interpreted => vm.set_machine_jit_for_test(MachineJitMode::Off),
        Subject::Forced => vm.set_machine_jit_for_test(MachineJitMode::Forced),
        Subject::Background => {
            vm.set_machine_jit_for_test(MachineJitMode::Off);
            vm.set_tier_up_after(Some(2));
            vm.set_tier_up_background(true);
        }
    }
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.gdtr.base = 0x2000;
    cpu.cpu.legacy.gdtr.limit = 0x7f;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        cache.base = 0;
        cache.limit = u32::MAX;
        cache.attributes = if i == 0 { 0x409b } else { 0x4093 };
    }
    cpu.cpu.cs = 8;
    cpu.cpu.ds = 16;
    cpu.cpu.es = 16;
    cpu.cpu.ss = 16;
    cpu.cpu.gpr[4] = 0x8000;
    descriptor(&vm, 0x2008, 0, u32::MAX, 0xc09b);
    descriptor(&vm, 0x2010, 0, u32::MAX, 0xc093);
    (vm, cpu)
}

/// The registers without the retry ledger's dead entries.
fn live(cpu: &x86jit_core::CpuState) -> String {
    let mut cpu = cpu.clone();
    cpu.mmio_parts[usize::from(cpu.mmio_parts_len)..].fill((0, 0));
    format!("{cpu:?}")
}

/// Run reference and subject in lockstep with this budget until HLT.
fn lockstep(code: &[u8], subject: Subject, budget: u64) {
    let (bvm, mut b) = machine(code, Subject::Interpreted);
    let (avm, mut a) = machine(code, subject);
    let (mut subject_handler, mut reference_handler) = (Recorder::default(), Recorder::default());
    for call in 0..100_000 {
        let what = format!("{subject:?} budget {budget} call {call}");
        let ar = a
            .run_steps_with_devices(
                &avm,
                budget,
                StepOptions::default(),
                Some(&mut subject_handler),
            )
            .unwrap();
        let br = b
            .run_steps_with_devices(
                &bvm,
                budget,
                StepOptions::default(),
                Some(&mut reference_handler),
            )
            .unwrap();
        assert_eq!(
            format!("{:?}", ar.stop()),
            format!("{:?}", br.stop()),
            "{what}"
        );
        assert_eq!(ar.steps(), br.steps(), "{what}");
        assert_eq!(live(&a.cpu), live(&b.cpu), "{what}");
        assert_eq!(a.retired_instructions(), b.retired_instructions(), "{what}");
        assert_eq!(
            subject_handler.requests, reference_handler.requests,
            "{what}: device transcripts differ"
        );
        let mut am = vec![0; MMIO as usize];
        let mut bm = vec![0; MMIO as usize];
        avm.read_bytes(0, &mut am).unwrap();
        bvm.read_bytes(0, &mut bm).unwrap();
        assert!(am == bm, "{what}: RAM differs");
        match ar.into_stop() {
            StepStop::Exit(Exit::BudgetExhausted) => {}
            StepStop::Exit(Exit::MmioRead { addr, .. }) => {
                a.complete_mmio_read(addr ^ 0x77);
                b.complete_mmio_read(addr ^ 0x77);
            }
            StepStop::Exit(Exit::MmioWrite { .. }) => {
                a.complete_mmio_write();
                b.complete_mmio_write();
            }
            StepStop::Exit(Exit::Hlt) => {
                // A budget below the block length keeps it interpreted (E4).
                if matches!(subject, Subject::Forced) && budget >= 64 {
                    let c = a.machine_jit_counters_for_test();
                    assert!(c.compiled_retired > 0, "{what}: {c:?}");
                }
                return;
            }
            other => panic!("{what}: {other:?}"),
        }
    }
    panic!("{subject:?} budget {budget}: the program did not end");
}

/// `mov ebp, N; L: mov eax, [9000h]; add ebx, eax; mov [9004h], ebx;
/// inc ecx; mov [90F0h], ecx (Fallback); movzx edx, byte [9008h]; dec ebp;
/// jnz L; hlt`.
fn program(iterations: u32) -> Vec<u8> {
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    a.mov(ebp, iterations).unwrap();
    a.set_label(&mut top).unwrap();
    a.mov(eax, dword_ptr(0x9000)).unwrap();
    a.add(ebx, eax).unwrap();
    a.mov(dword_ptr(0x9004), ebx).unwrap();
    a.inc(ecx).unwrap();
    a.mov(dword_ptr(0x90f0), ecx).unwrap();
    a.movzx(edx, byte_ptr(0x9008)).unwrap();
    a.add(esi, edx).unwrap();
    a.dec(ebp).unwrap();
    a.jne(top).unwrap();
    a.hlt().unwrap();
    a.assemble(0x1000).unwrap()
}

#[test]
fn compiled_device_accesses_equal_the_interpreter() {
    for subject in [Subject::Forced, Subject::Background] {
        for budget in [1, 3, 7, 64, 100_000] {
            lockstep(&program(200), subject, budget);
        }
    }
}

/// A handled access inside a compiled block continues the block: the
/// program's reads and writes complete without leaving compiled code.
#[test]
fn handled_device_accesses_stay_in_compiled_code() {
    let (vm, mut cpu) = machine(&program(50), Subject::Forced);
    let mut handler = Recorder::default();
    loop {
        let run = cpu
            .run_steps_with_devices(&vm, 100_000, StepOptions::default(), Some(&mut handler))
            .unwrap();
        match run.into_stop() {
            StepStop::Exit(Exit::MmioWrite { .. }) => cpu.complete_mmio_write(),
            StepStop::Exit(Exit::Hlt) => break,
            other => panic!("{other:?}"),
        }
    }
    let c = cpu.machine_jit_counters_for_test();
    // The Fallback write defers once per iteration; the handled ones do not.
    assert!(c.compiled_retired >= 50 * 3, "{c:?}");
    assert_eq!(handler.requests.len(), 50 * 4, "one call per transfer");
}

/// `mov ebp, 100; L: inc eax; add ebx, eax; shld si, di, 3; mov eax, [9000h];
/// add ecx, eax; dec ebp; jnz L; hlt`. The 16-bit SHLD stays interpreted, so a
/// compiled unit of the loop block defers at ordinal 2, and the interpreted
/// rest of the block serves the handled read in its pass (`in_pass_access`).
fn deferred_then_device() -> Vec<u8> {
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    a.mov(ebp, 100u32).unwrap();
    a.set_label(&mut top).unwrap();
    a.inc(eax).unwrap();
    a.add(ebx, eax).unwrap();
    a.shld(si, di, 3).unwrap();
    a.mov(eax, dword_ptr(0x9000)).unwrap();
    a.add(ecx, eax).unwrap();
    a.dec(ebp).unwrap();
    a.jne(top).unwrap();
    a.hlt().unwrap();
    a.assemble(0x1000).unwrap()
}

/// Decision-30 §4: after a compiled block defers at ordinal c, a device
/// transfer its interpreted rest serves in place reports the position of the
/// interpreted walk of the whole block (`device::advance_armed`): transcripts
/// equal the interpreter's in every compile mode and budget.
#[test]
fn a_transfer_after_a_compiled_defer_reports_the_whole_block_position() {
    let code = deferred_then_device();
    for subject in [Subject::Forced, Subject::Background] {
        for budget in [1, 2, 7, 64] {
            lockstep(&code, subject, budget);
        }
    }
    // The compiled unit ran and deferred inside its block.
    let (vm, mut cpu) = machine(&code, Subject::Forced);
    let mut handler = Recorder::default();
    let run = cpu
        .run_steps_with_devices(&vm, 100_000, StepOptions::default(), Some(&mut handler))
        .unwrap();
    assert!(matches!(run.into_stop(), StepStop::Exit(Exit::Hlt)));
    let c = cpu.machine_jit_counters_for_test();
    assert!(
        c.unconditional_defers > 0 && c.compiled_retired > 0,
        "{c:?}"
    );
    assert!(c.defers_at_zero < c.unconditional_defers, "{c:?}");
    assert_eq!(handler.requests.len(), 100, "one call per transfer");
}
