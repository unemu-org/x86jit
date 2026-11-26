//! Events raised while a block is in flight (xj-to9.38): a host exit request,
//! an NMI edge and a code write, each made by another thread while the vCPU
//! is stopped at a device read in the middle of a block, and an interrupt
//! request queued there. The completion replays the device instruction in a
//! new walk of the same block; the event must be taken at the boundary after
//! it, the first one without an I/O completion outstanding (spec.md §5.2,
//! §17.10), and never later. The other thread is joined before the run
//! resumes, so each position is deterministic.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::descriptor;
use x86jit_core::{
    CpuModel, ExceptionDelivery, Exit, InterpreterBackend, Prot, RegionKind, StepOptions, StepStop,
    Vcpu, Vm, VmConfig,
};

const IP: u64 = 0x1200;
const DEVICE: u64 = 0x6000;
const HANDLER: u64 = 0x3000;
const PT: u64 = 0x1f000;

/// Where a world runs its code.
#[derive(Clone, Copy, Debug, PartialEq)]
enum World {
    /// Real mode on a twenty-bit model or on the Direct class.
    Real(CpuModel),
    /// Flat32 protected mode on an I80386, paged or not.
    Flat32 { paged: bool },
}
use World::*;

const WORLDS: [World; 5] = [
    Real(CpuModel::I8086),
    Real(CpuModel::I80286),
    Real(CpuModel::I80386),
    Flat32 { paged: false },
    Flat32 { paged: true },
];

impl World {
    /// INC AX; INC AX; MOV AX,[DEVICE]; INC CX; INC DX; INC BX; HLT, and the
    /// offset of INC CX and of INC DX.
    fn code(self) -> (Vec<u8>, u64, u64) {
        match self {
            Real(_) => (b"\x40\x40\xa1\x00\x60\x41\x42\x43\xf4".to_vec(), 5, 6),
            Flat32 { .. } => (
                b"\x40\x40\xa1\x00\x60\x00\x00\x41\x42\x43\xf4".to_vec(),
                7,
                8,
            ),
        }
    }
    fn real(self) -> bool {
        matches!(self, Real(_))
    }
}

/// RAM everywhere but one trap page at `DEVICE`; for real mode, vector 2
/// (NMI) and vector 20h each point at a HLT.
fn build(world: World) -> (Vm, Vcpu) {
    let (code, _, _) = world.code();
    let model = match world {
        Real(model) => model,
        Flat32 { .. } => CpuModel::I80386,
    };
    let mut vm = Vm::with_backend(VmConfig::flat(0x240000), Box::new(InterpreterBackend));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(if world.real() {
        ExceptionDelivery::InGuest
    } else {
        ExceptionDelivery::Outward
    });
    vm.map(0, DEVICE as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(DEVICE, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.map(
        DEVICE + 0x1000,
        0x240000 - DEVICE as usize - 0x1000,
        Prot::RWX,
        RegionKind::Ram,
    )
    .unwrap();
    vm.write_bytes(IP, &code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = IP;
    cpu.cpu.gpr[4] = 0x8000;
    cpu.set_a20_enabled(true);
    match world {
        Real(_) => {
            // 8086 manual 9800722-03 §2.5 p. 2-25: the vector table at 0.
            for vector in [2u64, 0x20] {
                vm.write_bytes(vector * 4, &(HANDLER as u32).to_le_bytes())
                    .unwrap();
            }
            vm.write_bytes(HANDLER, b"\xf4").unwrap();
        }
        Flat32 { paged } => {
            cpu.cpu.legacy.cr0 = 1;
            cpu.cpu.legacy.gdtr.base = 0x2000;
            cpu.cpu.legacy.gdtr.limit = 0x7f;
            descriptor(&vm, 0x2008, 0, u32::MAX, 0xc09b);
            descriptor(&vm, 0x2010, 0, u32::MAX, 0xc093);
            for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
                cache.base = 0;
                cache.limit = u32::MAX;
                cache.attributes = if i == 0 { 0x409b } else { 0x4093 };
            }
            cpu.cpu.cs = 8;
            cpu.cpu.ds = 16;
            cpu.cpu.es = 16;
            cpu.cpu.ss = 16;
            if paged {
                // 80386 PRM §5.2 pp. 5-7–11: one table of identity user pages.
                vm.write_bytes(0x1e000, &0x1f007u32.to_le_bytes()).unwrap();
                for page in 0..32u32 {
                    vm.write_bytes(PT + u64::from(page * 4), &((page * 4096) | 7).to_le_bytes())
                        .unwrap();
                }
                cpu.cpu.legacy.cr3 = 0x1e000;
                cpu.cpu.legacy.cr0 |= 1 << 31;
            }
        }
    }
    (vm, cpu)
}

/// Run the two INCs and stop at the device read with the block cached:
/// a first pass to the HLT warms the block, then the run starts over.
fn to_the_device(world: World) -> (Vm, Vcpu) {
    let (vm, mut cpu) = build(world);
    let initial = cpu.cpu.clone();
    let run = cpu.run_steps(&vm, 20, StepOptions::default()).unwrap();
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })),
        "{world:?}: {run:?}"
    );
    cpu.complete_mmio_read(0x40);
    let run = cpu.run_steps(&vm, 20, StepOptions::default()).unwrap();
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
        "{world:?}: {run:?}"
    );
    cpu.cpu = initial;
    let retired = cpu.retired_instructions();
    let run = cpu.run_steps(&vm, 20, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 2, "{world:?}: {run:?}");
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })),
        "{world:?}: {run:?}"
    );
    assert_eq!(cpu.retired_instructions() - retired, 2, "{world:?}");
    (vm, cpu)
}

/// The return IP of the interrupt frame the handler's HLT stopped after.
fn frame_ip(vm: &Vm, cpu: &Vcpu) -> u64 {
    let mut ip = [0; 2];
    vm.read_bytes(
        (u64::from(cpu.cpu.ss) << 4) + (cpu.cpu.gpr[4] & 0xffff),
        &mut ip,
    )
    .unwrap();
    u64::from(u16::from_le_bytes(ip))
}

#[test]
fn an_exit_request_from_another_thread_stops_after_the_completed_device_instruction() {
    for world in WORLDS {
        let (_, next, _) = world.code();
        let (vm, mut cpu) = to_the_device(world);
        let request = cpu.exit_request();
        std::thread::spawn(move || request.request_exit())
            .join()
            .unwrap();
        cpu.complete_mmio_read(41);
        let retired = cpu.retired_instructions();
        let run = cpu.run_steps(&vm, 20, StepOptions::default()).unwrap();
        // The completion's replay retires; the request is taken before INC CX.
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::Interrupted)),
            "{world:?}: {run:?}"
        );
        assert_eq!(run.steps(), 1, "{world:?}");
        assert_eq!(cpu.retired_instructions() - retired, 1, "{world:?}");
        assert_eq!(cpu.cpu.rip, IP + next, "{world:?}");
        assert_eq!(cpu.cpu.gpr[0] & 0xffff, 41, "{world:?}");
        assert_eq!(cpu.cpu.gpr[1], 0, "{world:?}: INC CX did not run");
        let run = cpu.run_steps(&vm, 20, StepOptions::default()).unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
            "{world:?}: {run:?}"
        );
        assert_eq!([cpu.cpu.gpr[1], cpu.cpu.gpr[2], cpu.cpu.gpr[3]], [1, 1, 1]);
    }
}

#[test]
fn a_code_write_from_another_thread_is_executed_by_the_resumed_block() {
    // SDM Vol 3A §11.6: a write to code that has not yet been fetched is
    // observed when it executes. HLT over INC DX while the block is stopped
    // at its device read.
    for world in WORLDS {
        let (_, _, written) = world.code();
        let (vm, mut cpu) = to_the_device(world);
        std::thread::scope(|scope| {
            scope.spawn(|| vm.write_bytes(IP + written, b"\xf4").unwrap());
        });
        cpu.complete_mmio_read(41);
        let run = cpu.run_steps(&vm, 20, StepOptions::default()).unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
            "{world:?}: {run:?}"
        );
        // The device read, INC CX and the written HLT, which stops after itself.
        assert_eq!(run.steps(), 3, "{world:?}");
        assert_eq!(cpu.cpu.rip, IP + written + 1, "{world:?}");
        assert_eq!(cpu.cpu.gpr[1], 1, "{world:?}: INC CX ran");
        assert_eq!(
            cpu.cpu.gpr[2], 0,
            "{world:?}: the overwritten INC DX never ran"
        );
    }
}

#[test]
fn an_nmi_from_another_thread_is_taken_after_the_completed_device_instruction() {
    // spec.md §17.10: a latched edge is accepted at the first boundary with
    // no I/O completion outstanding; 8086 manual 9800722-03 §2.5 p. 2-27,
    // 80286 PRM §5.4: NMI vectors through entry 2 and pushes the next IP.
    for world in WORLDS.into_iter().filter(|world| world.real()) {
        let (_, next, _) = world.code();
        let (vm, mut cpu) = to_the_device(world);
        let line = cpu.nmi_line().unwrap();
        std::thread::spawn(move || line.raise()).join().unwrap();
        cpu.complete_mmio_read(41);
        let run = cpu.run_steps(&vm, 20, StepOptions::default()).unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
            "{world:?}: {run:?}"
        );
        assert_eq!(cpu.cpu.rip, HANDLER + 1, "{world:?}: the handler's HLT");
        assert_eq!(frame_ip(&vm, &cpu), IP + next, "{world:?}");
        assert_eq!(cpu.cpu.gpr[0] & 0xffff, 41, "{world:?}");
        assert_eq!(cpu.cpu.gpr[1], 0, "{world:?}: INC CX did not run");
    }
}

#[test]
fn an_interrupt_queued_at_a_device_exit_is_taken_after_the_completed_instruction() {
    // 8086 manual 9800722-03 §2.5 p. 2-27: INTR is sampled at an instruction
    // boundary, here the first one after the device read's completion.
    for world in WORLDS.into_iter().filter(|world| world.real()) {
        let (_, next, _) = world.code();
        let (vm, mut cpu) = to_the_device(world);
        cpu.cpu.flags.if_ = true;
        cpu.inject_irq(0x20);
        cpu.complete_mmio_read(41);
        let run = cpu.run_steps(&vm, 20, StepOptions::default()).unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
            "{world:?}: {run:?}"
        );
        assert_eq!(cpu.cpu.rip, HANDLER + 1, "{world:?}: the handler's HLT");
        assert_eq!(frame_ip(&vm, &cpu), IP + next, "{world:?}");
        assert_eq!(cpu.cpu.gpr[0] & 0xffff, 41, "{world:?}");
        assert_eq!(cpu.cpu.gpr[1], 0, "{world:?}: INC CX did not run");
    }
}
