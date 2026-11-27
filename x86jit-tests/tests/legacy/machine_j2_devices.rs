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
    CpuModel, DeviceContext, DeviceError, DeviceIo, DeviceRange, DeviceRequest, DeviceResult,
    DeviceStop, ExceptionDelivery, Exit, MachineJitMode, Prot, RegionKind, StepOptions, StepStop,
    Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const SIZE: u64 = 0x20000;
const MMIO: u64 = 0x9000;

/// Answers reads from the address and the request's position, accepts
/// writes, and falls back at `MMIO + 0xf0`, recording every request.
///
/// With `effects`, the handler also acts on the vCPU and RAM through its
/// context: a read at `MMIO + 0x20` raises INTR every fifth time, a write at
/// `MMIO + 0x30` writes RAM at `DATA` (and every seventh time the code bytes
/// at `effects`), and a read at `MMIO + 0x40` returns DeviceFault every
/// fourth time.
#[derive(Default)]
struct Recorder {
    requests: Vec<DeviceRequest>,
    effects: Option<u64>,
}

/// The RAM a handler with effects writes.
const DATA: u64 = 0x6000;

impl Recorder {
    fn with_effects(code_patch: u64) -> Self {
        Recorder {
            requests: Vec::new(),
            effects: Some(code_patch),
        }
    }
}

impl DeviceIo for Recorder {
    fn access(&mut self, request: &DeviceRequest, context: &mut DeviceContext<'_>) -> DeviceResult {
        self.requests.push(request.clone());
        if request.address == MMIO + 0xf0 {
            return DeviceResult::Fallback;
        }
        let value = request.address ^ request.position.retired_instructions;
        let Some(patch) = self.effects else {
            return DeviceResult::Handled(value);
        };
        let n = self
            .requests
            .iter()
            .filter(|r| r.address == request.address)
            .count() as u64;
        match request.address - MMIO {
            0x20 if n % 5 == 2 => context.set_intr(true),
            0x30 => {
                let word = (request.value ^ request.position.retired_instructions) as u32;
                context
                    .write_ram(DATA + (n % 4) * 4, &word.to_le_bytes())
                    .unwrap();
                if n % 7 == 3 {
                    context.write_ram(patch, &(n as u32).to_le_bytes()).unwrap();
                }
            }
            0x40 if n % 4 == 1 => return DeviceResult::DeviceFault(DeviceError::new(5)),
            _ => {}
        }
        DeviceResult::Handled(value)
    }
}

#[derive(Clone, Copy, Debug)]
enum Subject {
    Interpreted,
    Forced,
    /// Forced, with the test-only chain-off switch (decision-31 §8).
    ForcedChainOff,
    Background,
}

thread_local! {
    /// Decision-32 §9: [`machine`] builds paged machines.
    static PAGED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `run` with paged machines: every page identity-mapped through the
/// directory at 3000h and the table at 4000h, but for linear 1A000h, a
/// second mapping of the device page (80386 PRM §5.2 Figs. 5-8-10).
fn paged(run: impl FnOnce()) {
    PAGED.with(|p| p.set(true));
    run();
    PAGED.with(|p| p.set(false));
}

/// The linear page that [`paged`] machines map to the device frame.
const ALIAS: u64 = 0x1a000;

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
    if PAGED.with(std::cell::Cell::get) {
        vm.write_bytes(0x3000, &0x4007u32.to_le_bytes()).unwrap();
        for page in 0..SIZE >> 12 {
            let frame = if page << 12 == ALIAS {
                MMIO
            } else {
                page << 12
            };
            vm.write_bytes(0x4000 + 4 * page, &(frame as u32 | 7).to_le_bytes())
                .unwrap();
        }
        cpu.cpu.legacy.cr0 = 0x8000_0001;
        cpu.cpu.legacy.cr3 = 0x3000;
    }
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
    lockstep_with(code, subject, budget, Recorder::default);
}

/// [`lockstep`] with handlers made by `handler`. An INTR the handler raised
/// is lowered on both when a run stops for it, and a DeviceFault is
/// completed on both with the same value.
fn lockstep_with(code: &[u8], subject: Subject, budget: u64, handler: impl Fn() -> Recorder) {
    let (bvm, mut b) = machine(code, Subject::Interpreted);
    let (avm, mut a) = machine(code, subject);
    let (mut subject_handler, mut reference_handler) = (handler(), handler());
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
            StepStop::Exit(Exit::Device(DeviceStop::IntrReady)) => {
                a.set_intr(false);
                b.set_intr(false);
            }
            StepStop::Exit(Exit::Device(DeviceStop::Fault { .. })) => {
                let value = a.retired_instructions() ^ 0x5a5a;
                assert!(a.complete_device_fault(value), "{what}");
                assert!(b.complete_device_fault(value), "{what}");
            }
            StepStop::Exit(Exit::Hlt) => {
                // A budget below the block length keeps it interpreted (E4).
                if matches!(subject, Subject::Forced | Subject::ForcedChainOff) && budget >= 64 {
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

/// Twelve blocks per iteration, chained (decision-31): handled reads and
/// writes in the first, second and twelfth block, and a Fallback write in
/// the twelfth.
fn chained_devices(iterations: u32) -> Vec<u8> {
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    a.mov(ebp, iterations).unwrap();
    a.set_label(&mut top).unwrap();
    for block in 0..11u32 {
        let mut next = a.create_label();
        match block {
            0 => {
                a.mov(eax, dword_ptr(0x9000)).unwrap();
                a.add(ebx, eax).unwrap();
            }
            1 => a.mov(dword_ptr(0x9004), ebx).unwrap(),
            _ => a.add(ecx, block).unwrap(),
        }
        a.jmp(next).unwrap();
        a.set_label(&mut next).unwrap();
    }
    a.movzx(edx, byte_ptr(0x9008)).unwrap();
    a.add(esi, edx).unwrap();
    a.mov(dword_ptr(0x90f0), ecx).unwrap();
    a.dec(ebp).unwrap();
    a.jne(top).unwrap();
    a.hlt().unwrap();
    a.assemble(0x1000).unwrap()
}

/// Decision-31 §4.4, §8 Devices: handled transfers in the first, second and
/// twelfth block of a chain report the interpreter's positions, the handler
/// is called once per transfer, and the blocks chained.
#[test]
fn device_accesses_in_chained_blocks_equal_the_interpreter() {
    let code = chained_devices(60);
    for subject in [Subject::Forced, Subject::Background] {
        for budget in [1, 2, 7, 13, 64, 100_000] {
            lockstep(&code, subject, budget);
        }
    }
    let (vm, mut cpu) = machine(&code, Subject::Forced);
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
    assert!(c.chained > 0, "{c:?}");
    assert_eq!(handler.requests.len(), 60 * 4, "one call per transfer");
}

/// The imm32 of `add ecx, imm32` in [`chained_effects`]' sixth block, which a
/// handler with effects rewrites.
const PATCH_IMM: u32 = 0x1234_5678;

/// Twelve blocks per iteration, chained (decision-31), with the device
/// accesses past the chain's first block: block 2 reads `MMIO + 0x20` (INTR),
/// block 3 reads the RAM the handler writes, block 4 writes `MMIO + 0x30`
/// (RAM, and the code of block 5), block 5 adds the rewritten immediate, and
/// block 6 reads `MMIO + 0x40` (DeviceFault). Instructions follow each access
/// in its block. IF is set, so a raised INTR stops the run at a boundary.
fn chained_effects(iterations: u32) -> Vec<u8> {
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    a.sti().unwrap();
    a.mov(ebp, iterations).unwrap();
    a.set_label(&mut top).unwrap();
    for block in 0..11u32 {
        let mut next = a.create_label();
        match block {
            2 => {
                a.mov(eax, dword_ptr(MMIO + 0x20)).unwrap();
                a.add(ebx, eax).unwrap();
                a.inc(edi).unwrap();
            }
            3 => {
                a.add(edx, dword_ptr(DATA + 4)).unwrap();
                a.xor(edx, dword_ptr(DATA + 8)).unwrap();
            }
            4 => {
                a.mov(dword_ptr(MMIO + 0x30), ebx).unwrap();
                a.add(ebx, edx).unwrap();
            }
            5 => a.add(ecx, PATCH_IMM).unwrap(),
            6 => {
                a.mov(eax, dword_ptr(MMIO + 0x40)).unwrap();
                a.add(esi, eax).unwrap();
                a.rol(esi, 3).unwrap();
            }
            _ => a.add(ecx, block).unwrap(),
        }
        a.jmp(next).unwrap();
        a.set_label(&mut next).unwrap();
    }
    a.dec(ebp).unwrap();
    a.jne(top).unwrap();
    a.hlt().unwrap();
    a.assemble(0x1000).unwrap()
}

/// Where [`PATCH_IMM`] sits in `code`, assembled at 1000h.
fn patch_address(code: &[u8]) -> u64 {
    let imm = PATCH_IMM.to_le_bytes();
    0x1000 + code.windows(4).position(|w| w == imm).unwrap() as u64
}

/// Decision-31 §4.4, §8 Devices: a handler that raises INTR, writes RAM
/// (data, and code the chain runs through) or returns DeviceFault from
/// block 2 to 6 of a chain leaves the same transcript, retired count,
/// registers, RAM and exit as the interpreter, with chains on and off.
#[test]
fn device_effects_in_chained_blocks_equal_the_interpreter() {
    let code = chained_effects(60);
    let patch = patch_address(&code);
    for subject in [
        Subject::Forced,
        Subject::ForcedChainOff,
        Subject::Background,
    ] {
        for budget in [1, 2, 7, 13, 64, 100_000] {
            lockstep_with(&code, subject, budget, || Recorder::with_effects(patch));
        }
    }
    // The chain ran through the accesses, and every effect happened.
    let (vm, mut cpu) = machine(&code, Subject::Forced);
    let mut handler = Recorder::with_effects(patch);
    let (mut intr, mut faults) = (0, 0);
    loop {
        let run = cpu
            .run_steps_with_devices(&vm, 100_000, StepOptions::default(), Some(&mut handler))
            .unwrap();
        match run.into_stop() {
            StepStop::Exit(Exit::Device(DeviceStop::IntrReady)) => {
                intr += 1;
                cpu.set_intr(false);
            }
            StepStop::Exit(Exit::Device(DeviceStop::Fault { .. })) => {
                faults += 1;
                assert!(cpu.complete_device_fault(0));
            }
            StepStop::Exit(Exit::Hlt) => break,
            other => panic!("{other:?}"),
        }
    }
    let c = cpu.machine_jit_counters_for_test();
    assert!(c.chained > 0, "{c:?}");
    assert_eq!(intr, 12, "every fifth of 60 INTR reads");
    assert_eq!(faults, 15, "every fourth of 60 fault reads");
    let mut imm = [0; 4];
    vm.read_bytes(patch, &mut imm).unwrap();
    assert_ne!(u32::from_le_bytes(imm), PATCH_IMM, "the code was rewritten");
}

/// `mov ebp, N; L: mov eax, [ALIAS]; add ebx, eax; mov [ALIAS + 4], ebx;
/// movzx edx, byte [9008h]; add esi, edx; dec ebp; jnz L; hlt`: the device
/// page through its second linear mapping and its own.
fn alias_program(iterations: u32) -> Vec<u8> {
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    a.mov(ebp, iterations).unwrap();
    a.set_label(&mut top).unwrap();
    a.mov(eax, dword_ptr(ALIAS)).unwrap();
    a.add(ebx, eax).unwrap();
    a.mov(dword_ptr(ALIAS + 4), ebx).unwrap();
    a.movzx(edx, byte_ptr(0x9008)).unwrap();
    a.add(esi, edx).unwrap();
    a.dec(ebp).unwrap();
    a.jne(top).unwrap();
    a.hlt().unwrap();
    a.assemble(0x1000).unwrap()
}

/// Decision-32 Q9, §9: paged compiled code reaches in-process devices
/// through the in-block contract with the access's linear address, which
/// is what the paged exit path reports: transcripts (positions and linear
/// addresses included), state and RAM equal the interpreter's, with chains
/// on and off and in the background configuration, and a handled transfer
/// keeps its block compiled.
#[test]
fn paged_device_accesses_equal_the_interpreter() {
    paged(|| {
        let effects = chained_effects(30);
        let patch = patch_address(&effects);
        for subject in [
            Subject::Forced,
            Subject::ForcedChainOff,
            Subject::Background,
        ] {
            for budget in [1, 3, 7, 13, 64, 100_000] {
                lockstep(&program(60), subject, budget);
                lockstep(&chained_devices(20), subject, budget);
                lockstep(&alias_program(60), subject, budget);
                lockstep(&deferred_then_device(), subject, budget);
                lockstep_with(&effects, subject, budget, || Recorder::with_effects(patch));
            }
        }
        let (vm, mut cpu) = machine(&alias_program(50), Subject::Forced);
        let mut handler = Recorder::default();
        let run = cpu
            .run_steps_with_devices(&vm, 100_000, StepOptions::default(), Some(&mut handler))
            .unwrap();
        assert!(matches!(run.into_stop(), StepStop::Exit(Exit::Hlt)));
        let c = cpu.machine_jit_counters_for_test();
        assert!(c.paged_entries > 0 && c.compiled_retired >= 50 * 5, "{c:?}");
        assert_eq!(handler.requests.len(), 50 * 3, "one call per transfer");
        for request in &handler.requests {
            let linear = request.linear_address.expect("a paged linear address");
            let expected = if request.address == MMIO + 8 {
                MMIO + 8
            } else {
                ALIAS + (request.address - MMIO)
            };
            assert_eq!(u64::from(linear), expected, "{request:?}");
        }
    });
}
