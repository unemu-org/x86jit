use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use x86jit_core::{interp, CpuMode, CpuState, Memory, MemoryModel, Prot, RegionKind};

struct CountingAllocator;
thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
    static ABORT_ON_ALLOCATION: Cell<bool> = const { Cell::new(false) };
}
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| {
            if let Some(n) = count.get() {
                count.set(Some(n + 1));
            }
        });
        ABORT_ON_ALLOCATION.with(|abort| {
            if abort.get() {
                std::process::abort();
            }
        });
        // SAFETY: forward the allocator contract unchanged.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ABORT_ON_ALLOCATION.with(|abort| {
            if abort.get() {
                std::process::abort();
            }
        });
        // SAFETY: forward the allocator contract unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.with(|count| {
            if let Some(n) = count.get() {
                count.set(Some(n + 1));
            }
        });
        ABORT_ON_ALLOCATION.with(|abort| {
            if abort.get() {
                std::process::abort();
            }
        });
        // SAFETY: forward the allocator contract unchanged.
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn count_allocations<T>(f: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = f();
    let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    (result, allocations)
}

#[test]
fn audit_step_one_allocations() {
    let mut mem = Memory::new(MemoryModel::Flat { size: 0x10000 });
    mem.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    mem.map(0x3000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
    let mut scratch = Vec::with_capacity(4096);
    // Warm iced's immutable tables before measuring engine work.
    mem.write_bytes(0x1000, &[0x90]).unwrap();
    let mut cpu = CpuState {
        rip: 0x1000,
        ..CpuState::default()
    };
    interp::step_one(
        &mem,
        &mut cpu,
        CpuMode::Long64,
        &mut scratch,
        &mut Default::default(),
    );
    for (name, code, addr) in [
        ("integer", &[0x48, 0x83, 0xc0, 1][..], 0x3000),
        ("sse", &[0x0f, 0x10, 0x00], 0x3000),
        ("avx", &[0xc5, 0xfc, 0x10, 0x00], 0x3000),
        ("rep_stos", &[0xf3, 0xaa], 0x3000),
        ("fault", &[0x48, 0x8b, 0x00], 0x8000),
        ("syscall", &[0x0f, 0x05], 0x3000),
        ("code_store", &[0x48, 0x89, 0x00], 0x1000),
    ] {
        mem.write_bytes(0x1000, code).unwrap();
        mem.mark_code(0x1000, 1);
        mem.watch_range(0x3000, 0x1000);
        cpu.rip = 0x1000;
        cpu.gpr[0] = addr;
        cpu.gpr[1] = 2;
        cpu.gpr[7] = 0x3000;
        let (_result, allocations) = count_allocations(|| {
            interp::step_one(
                &mem,
                &mut cpu,
                CpuMode::Long64,
                &mut scratch,
                &mut Default::default(),
            )
        });
        eprintln!("audit {name}: {allocations} allocations");
        assert!(
            allocations > 0,
            "baseline {name} unexpectedly allocation-free"
        );
    }
}

use x86jit_core::{
    child, ChildArena, ChildRefusal, ChildStep, FaultCause, FaultOrigin, FaultPath, Knowledge,
    RegisterProgress, Vm, VmConfig,
};

pub(crate) fn no_alloc<T>(f: impl FnOnce() -> T) -> T {
    ABORT_ON_ALLOCATION.with(|abort| abort.set(true));
    let result = f();
    ABORT_ON_ALLOCATION.with(|abort| abort.set(false));
    result
}
fn vm(code: &[u8]) -> Vm {
    let mut vm = Vm::new(VmConfig::flat(0x10000));
    vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x3000, 0x3000, Prot::RW, RegionKind::Ram).unwrap();
    vm.map(0x7000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    vm
}
pub(crate) fn initial_cpu() -> CpuState {
    let mut cpu = CpuState {
        rip: 0x1000,
        ..CpuState::default()
    };
    cpu.gpr[0] = 0x3100;
    cpu.gpr[1] = 5;
    cpu.gpr[2] = 0;
    cpu.gpr[3] = 7;
    cpu.gpr[4] = 0x5800;
    cpu.gpr[6] = 0x3100;
    cpu.gpr[7] = 0x4100;
    cpu.fs_base = 0x3000;
    cpu.gs_base = 0x4000;
    cpu.fpu_cw = 0x77f;
    cpu.fpu_sw = 0x120;
    cpu.fpu_empty = 0x5a;
    cpu.fpr[2] = [0x73; 10];
    for i in 0..32 {
        cpu.xmm[i] = 0x3f80_0000_4000_0000_4040_0000_4080_0000 + i as u128;
        cpu.ymm_hi[i] = 0x3f80_0000_4000_0000_4040_0000_4080_0000;
        cpu.zmm_hi[i] = [0xeded; 2];
    }
    cpu.cpuid_overrides.xcr0 = Some(0x1234);
    cpu
}

#[test]
fn child_inline_corpus_matches_step_one_without_allocation() {
    let cases: &[&[u8]] = &[
        &[0x90],
        &[0xb8, 0xff, 0xff, 0xff, 0xff],
        &[0x48, 0x83, 0xc0, 1],
        &[0x48, 0x11, 0xd8],
        &[0x48, 0x19, 0xd8],
        &[0x48, 0xc1, 0xe0, 3],
        &[0x48, 0xf7, 0xe3],
        &[0x48, 0xf7, 0xf3],
        &[0x48, 0x8b, 0x00],
        &[0x48, 0x89, 0x18],
        &[0x48, 0x83, 0x00, 1],
        &[0x53],
        &[0x5b],
        &[0xe8, 0x10, 0, 0, 0],
        &[0xc3],
        &[0x9c],
        &[0x9d],
        &[0x74, 2],
        &[0xeb, 2],
        &[0x0f, 0x10, 0x00],
        &[0x0f, 0x11, 0x00],
        &[0x0f, 0x58, 0xc1],
        &[0x66, 0x0f, 0xef, 0xc1],
        &[0x66, 0x0f, 0x3a, 0xcf, 0xc1, 0],
        &[0x66, 0x0f, 0x3a, 0xcf, 0x00, 0],
        &[0x66, 0x0f, 0xfc, 0x00],
        &[0xc5, 0xfc, 0x10, 0x00],
        &[0xc5, 0xfc, 0x11, 0x00],
        &[0x62, 0xf1, 0x7c, 0x48, 0x11, 0x00],
        &[0xc5, 0xfc, 0x58, 0xc1],
        &[0xc5, 0xf8, 0x77],
        &[0xf3, 0xa4],
        &[0xf3, 0xaa],
        &[0xf3, 0x48, 0xab],
        &[0xa4],
        &[0xaa],
        &[0xac],
        &[0xae],
        &[0xa6],
        &[0x0f, 0x05],
        &[0xcd, 0x80],
        &[0xf4],
        &[0xf8],
        &[0xf9],
        &[0xfc],
        &[0xfd],
        &[0x0f, 0xae, 0x18], // stmxcsr
        &[0xe6, 0x80],
        &[0xe4, 0x80],
    ];
    for (mode, tier) in [CpuMode::Long64, CpuMode::Compat32]
        .into_iter()
        .flat_map(|mode| {
            [
                x86jit_core::MemConsistency::Fast,
                x86jit_core::MemConsistency::AcqRel,
                x86jit_core::MemConsistency::FullTso,
            ]
            .into_iter()
            .map(move |tier| (mode, tier))
        })
    {
        for &code in cases {
            let mut vm = vm(code);
            vm.consistency = tier;
            let data = [0x12; 0x3000];
            vm.mem.write_bytes(0x3000, &data).unwrap();
            let mut normal = initial_cpu();
            let mut child_cpu = normal.clone();
            let normal_result = interp::step_one(
                &vm.mem,
                &mut normal,
                mode,
                &mut Vec::new(),
                &mut interp::RetireInfo {
                    consistency: tier,
                    ..Default::default()
                },
            );
            let mut normal_data = [0; 0x3000];
            vm.mem.read_bytes(0x3000, &mut normal_data).unwrap();
            vm.mem.write_bytes(0x3000, &data).unwrap();
            let view = vm.child_view(mode);
            assert_eq!(view.mem_consistency(), tier);
            let mut arena = ChildArena::new(16);
            let child_result = no_alloc(|| child::step(&view, &mut child_cpu, &mut arena));
            match (normal_result, &child_result) {
                (x86jit_core::StepResult::Continue, ChildStep::Continue)
                | (
                    x86jit_core::StepResult::Exit(x86jit_core::Exit::Syscall),
                    ChildStep::Syscall { .. },
                )
                | (x86jit_core::StepResult::Exit(x86jit_core::Exit::Hlt), ChildStep::Hlt)
                | (
                    x86jit_core::StepResult::Exit(x86jit_core::Exit::PortIo { .. }),
                    ChildStep::PortIo { .. },
                )
                | (
                    x86jit_core::StepResult::Exit(x86jit_core::Exit::Exception { .. }),
                    ChildStep::Fault(_),
                ) => {}
                _ => panic!("unexpected child result {mode:?} {code:x?}: {child_result:?}"),
            }
            if matches!(child_result, ChildStep::Fault(_)) {
                assert!(arena.written_pages().is_empty());
            }
            let mut child_data = [0; 0x3000];
            vm.mem.read_bytes(0x3000, &mut child_data).unwrap();
            assert_eq!(
                format!("{normal:?}"),
                format!("{child_cpu:?}"),
                "{mode:?} {code:x?}: {child_result:?}"
            );
            assert_eq!(normal_data, child_data, "{mode:?} {code:x?}");
        }
    }
}

#[test]
fn child_refusals_preserve_cpu_and_memory() {
    for (code, addr, capacity, why) in [
        (&[0xd9, 0xe8][..], 0x3100, 8, ChildRefusal::X87),
        (&[0x0f, 0xae, 0x00], 0x3100, 8, ChildRefusal::FxState),
        (&[0x0f, 0xae, 0x20], 0x3100, 8, ChildRefusal::FxState),
        (&[0x0f, 0xc7, 0x20], 0x3100, 8, ChildRefusal::FxState),
        (&[0x0f, 0x01, 0xd0], 0x3100, 8, ChildRefusal::Helper),
        (&[0x0f, 0xa2], 0x3100, 8, ChildRefusal::Helper),
        (
            &[0xf0, 0x48, 0x01, 0x18],
            0x313f,
            8,
            ChildRefusal::AtomicAlignment,
        ),
        (
            &[0xf0, 0x48, 0x01, 0x18],
            0x3fff,
            8,
            ChildRefusal::AtomicAlignment,
        ),
        (&[0xf0, 0x48, 0x01, 0x18], 0x7100, 8, ChildRefusal::Mmio),
        (
            &[0xf0, 0x48, 0x01, 0x18],
            0x1000,
            8,
            ChildRefusal::CodeStore,
        ),
        (
            &[0xf0, 0x48, 0x01, 0x18],
            0x3100,
            0,
            ChildRefusal::ArenaFull,
        ),
        (&[0x48, 0x8b, 0x00], 0x7100, 8, ChildRefusal::Mmio),
        (&[0x48, 0x89, 0x18], 0x1000, 8, ChildRefusal::CodeStore),
        (&[0x48, 0x89, 0x18], 0x3100, 0, ChildRefusal::ArenaFull),
        (
            &[0xc5, 0xfc, 0x11, 0x00],
            0x3ff0,
            1,
            ChildRefusal::ArenaFull,
        ),
        (&[0xf3, 0xaa], 0x3100, 1, ChildRefusal::CodeStore),
    ] {
        let vm = vm(code);
        vm.mem.mark_code(0x1000, 1);
        let mut cpu = initial_cpu();
        cpu.gpr[0] = addr;
        if code == [0xf3, 0xaa] {
            cpu.gpr[7] = 0x1fff;
            cpu.gpr[1] = 2;
        }
        let before = format!("{cpu:?}");
        let mut before_data = [0; 0x3000];
        vm.mem.read_bytes(0x3000, &mut before_data).unwrap();
        let view = vm.child_view(CpuMode::Long64);
        let mut arena = ChildArena::new(capacity);
        assert!(
            matches!(no_alloc(|| child::step(&view, &mut cpu, &mut arena)), ChildStep::Refused(r) if r == why),
            "{code:x?}"
        );
        let mut after_data = [0; 0x3000];
        vm.mem.read_bytes(0x3000, &mut after_data).unwrap();
        assert_eq!(before, format!("{cpu:?}"));
        assert_eq!(before_data, after_data);
        assert!(arena.written_pages().is_empty());
    }
}

#[test]
fn child_locked_unaligned_operand_within_one_cache_line() {
    let vm = vm(&[0xf0, 0x01, 0x18]);
    vm.mem.write_bytes(0x3100, &[0xa5; 16]).unwrap();
    let view = vm.child_view(CpuMode::Long64);
    let mut cpu = initial_cpu();
    cpu.gpr[0] = 0x3101;
    let before = cpu.clone();
    let mut arena = ChildArena::new(1);
    let result = no_alloc(|| child::step(&view, &mut cpu, &mut arena));
    let mut bytes = [0; 16];
    vm.mem.read_bytes(0x3100, &mut bytes).unwrap();
    #[cfg(target_arch = "aarch64")]
    {
        assert!(matches!(
            result,
            ChildStep::Refused(ChildRefusal::AtomicAlignment)
        ));
        assert_eq!(format!("{cpu:?}"), format!("{before:?}"));
        assert_eq!(bytes, [0xa5; 16]);
        assert!(arena.written_pages().is_empty());
    }
    #[cfg(target_arch = "x86_64")]
    {
        assert!(matches!(result, ChildStep::Continue));
        vm.mem.write_bytes(0x3100, &[0xa5; 16]).unwrap();
        let mut expected = before;
        assert!(matches!(
            interp::step_one(
                &vm.mem,
                &mut expected,
                CpuMode::Long64,
                &mut Vec::new(),
                &mut Default::default()
            ),
            x86jit_core::StepResult::Continue
        ));
        let mut expected_bytes = [0; 16];
        vm.mem.read_bytes(0x3100, &mut expected_bytes).unwrap();
        assert_eq!(format!("{cpu:?}"), format!("{expected:?}"));
        assert_eq!(bytes, expected_bytes);
        assert_eq!(&*arena.written_pages(), &[3]);
    }
}

#[test]
fn child_fault_before_store_commit_discards_reservations() {
    // A faulting vector store must not retain a destination reservation.
    let vm = vm(&[0xc5, 0xfc, 0x11, 0x00]);
    let view = vm.child_view(CpuMode::Long64);
    let mut cpu = initial_cpu();
    cpu.gpr[0] = 0x5ff0;
    let before = format!("{cpu:?}");
    let mut arena = ChildArena::new(1);
    assert!(matches!(
        no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
        ChildStep::Fault(_)
    ));
    assert_eq!(format!("{cpu:?}"), before);
    assert_eq!(vm.mem.read(0x5ff0, 8).unwrap(), 0);
    assert!(arena.written_pages().is_empty());
    cpu.gpr[0] = 0x3100;
    assert!(matches!(
        no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
        ChildStep::Continue
    ));
    assert_eq!(&*arena.written_pages(), &[3]);
}

#[test]
fn child_rep_source_fault_discards_unwritten_destination_pages() {
    let vm = vm(&[0xf3, 0xa4]);
    let view = vm.child_view(CpuMode::Long64);
    let mut cpu = initial_cpu();
    cpu.gpr[6] = 0x8000;
    cpu.gpr[7] = 0x3fff;
    cpu.gpr[1] = 3;
    let before = format!("{cpu:?}");
    let mut arena = ChildArena::new(2);
    assert!(matches!(
        no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
        ChildStep::Fault(_)
    ));
    assert_eq!(format!("{cpu:?}"), before);
    assert_eq!(vm.mem.read(0x3fff, 2).unwrap(), 0);
    assert!(arena.written_pages().is_empty());
    cpu.gpr[6] = 0x3100;
    cpu.gpr[7] = 0x5100;
    cpu.gpr[1] = 1;
    assert!(matches!(
        no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
        ChildStep::Continue
    ));
    assert_eq!(&*arena.written_pages(), &[5]);
}

#[test]
fn child_gathers_refuse_without_allocation_or_state_changes() {
    for float in [false, true] {
        for elem in [4, 8] {
            for index in [4, 8] {
                for wide in [false, true] {
                    let code = [
                        0xc4,
                        0xe2,
                        0x69 | if wide { 4 } else { 0 } | if elem == 8 { 0x80 } else { 0 },
                        0x90 + u8::from(index == 8) + 2 * u8::from(float),
                        0x04,
                        0x88,
                    ];
                    let vm = vm(&code);
                    let view = vm.child_view(CpuMode::Long64);
                    let mut arena = ChildArena::new(4);
                    for addr in [0x3100, 0x7100, 0x8000] {
                        for mask in [0, u128::MAX] {
                            let mut cpu = initial_cpu();
                            cpu.gpr[0] = addr;
                            cpu.set_vec(1, [0; 4], 64);
                            cpu.set_vec(2, [mask; 4], 64);
                            let before = format!("{cpu:?}");
                            let mut before_data = [0; 0x3000];
                            vm.mem.read_bytes(0x3000, &mut before_data).unwrap();
                            assert!(matches!(
                                no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
                                ChildStep::Refused(ChildRefusal::Gather)
                            ));
                            let mut after_data = [0; 0x3000];
                            vm.mem.read_bytes(0x3000, &mut after_data).unwrap();
                            assert_eq!(before, format!("{cpu:?}"));
                            assert_eq!(before_data, after_data);
                            assert!(arena.written_pages().is_empty());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn child_faults_are_unboxed_and_precise() {
    for (code, addr, cause, vector) in [
        (&[0x48, 0xf7, 0xf1][..], 0x3100, FaultCause::Divide, 0),
        (
            &[0x0f, 0x28, 0x00],
            0x3101,
            FaultCause::RequiredAlignment,
            13,
        ),
        (
            &[0x48, 0x8b, 0x00],
            0x0000_8000_0000_0000,
            FaultCause::NonCanonical,
            13,
        ),
        (&[0xcc], 0x3100, FaultCause::Breakpoint, 3),
        (&[0x0f, 0x0b], 0x3100, FaultCause::InvalidOpcode, 6),
    ] {
        let vm = vm(code);
        let view = vm.child_view(CpuMode::Long64);
        let mut cpu = initial_cpu();
        cpu.gpr[0] = addr;
        cpu.gpr[1] = 0;
        let mut arena = ChildArena::new(4);
        let ChildStep::Fault(f) = no_alloc(|| child::step(&view, &mut cpu, &mut arena)) else {
            panic!("{code:x?}")
        };
        assert!(arena.written_pages().is_empty());
        assert_eq!(f.origin, FaultOrigin::GuestException(cause));
        assert_eq!(f.vector, Knowledge::Known(vector));
        assert_eq!(f.instruction_rip, 0x1000);
        assert_eq!(f.instruction_len, Knowledge::Known(code.len() as u8));
        assert_eq!(cpu.rip, if vector == 3 { 0x1001 } else { 0x1000 });
    }
    let vm = vm(&[0xc5, 0xfc, 0x10, 0x00]);
    let view = vm.child_view(CpuMode::Long64);
    let mut cpu = initial_cpu();
    cpu.gpr[0] = 0x5ff0;
    let before = cpu.xmm;
    let mut arena = ChildArena::new(4);
    let ChildStep::Fault(f) = no_alloc(|| child::step(&view, &mut cpu, &mut arena)) else {
        panic!()
    };
    assert_eq!(
        f.address,
        Knowledge::Known(x86jit_core::FaultAddress::new(
            0x6000,
            x86jit_core::AddressPrecision::ExactByte,
            x86jit_core::AddressSpace::Linear
        ))
    );
    assert_eq!(f.operand_base, Knowledge::Known(0x5ff0));
    assert_eq!(f.operand_size, Knowledge::Known(32));
    assert_eq!(cpu.xmm, before);
    assert!(arena.written_pages().is_empty());
}

#[test]
fn child_rep_restart_overlap_and_late_refusal() {
    // SDM Vol 2B §4.3 REP: restart registers describe the next unfinished element.
    let vm = vm(&[0xf3, 0xa4]);
    let view = vm.child_view(CpuMode::Long64);
    vm.mem.write_bytes(0x3100, &[0x57, 1, 2, 3, 4, 5]).unwrap();
    let mut cpu = initial_cpu();
    cpu.gpr[7] = 0x3101;
    let mut arena = ChildArena::new(8);
    assert!(matches!(
        no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
        ChildStep::Continue
    ));
    let mut bytes = [0; 6];
    vm.mem.read_bytes(0x3100, &mut bytes).unwrap();
    assert_eq!(bytes, [0x57; 6]);
    cpu = initial_cpu();
    cpu.gpr[7] = 0x5ffe;
    cpu.gpr[1] = 5;
    let ChildStep::Fault(f) = no_alloc(|| child::step(&view, &mut cpu, &mut arena)) else {
        panic!()
    };
    assert_eq!(&*arena.written_pages(), &[3, 5]);
    assert_eq!(cpu.gpr[1], 3);
    assert_eq!(cpu.gpr[7], 0x6000);
    assert_eq!(cpu.rip, 0x1000);
    assert_eq!(f.registers, RegisterProgress::RepRestart);
    assert_eq!(f.rep.unwrap().remaining(), 3);

    vm.mem.mark_code(0x5000, 1);
    let view = vm.child_view(CpuMode::Long64);
    cpu = initial_cpu();
    cpu.gpr[7] = 0x4ffe;
    let before = format!("{cpu:?}");
    let mut before_data = [0; 5];
    vm.mem.read_bytes(0x4ffe, &mut before_data).unwrap();
    assert!(matches!(
        no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
        ChildStep::Refused(ChildRefusal::CodeStore)
    ));
    let mut after_data = [0; 5];
    vm.mem.read_bytes(0x4ffe, &mut after_data).unwrap();
    assert_eq!(before, format!("{cpu:?}"));
    assert_eq!(before_data, after_data);
    assert_eq!(&*arena.written_pages(), &[3, 5]);
}

#[test]
fn child_fetch_snapshot_and_syscall_evidence() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    let mut vm = vm(&[0x0f, 0x05]);
    let poison = Arc::new(AtomicBool::new(false));
    let observed = poison.clone();
    vm.set_fetch_policy(Box::new(move |page| {
        assert!(
            !observed.load(Ordering::Relaxed),
            "parent fetch policy reached in child"
        );
        if page == 0x1000 {
            x86jit_core::FetchVerdict::Allow
        } else {
            x86jit_core::FetchVerdict::DenyProtection
        }
    }));
    let view = vm.child_view(CpuMode::Long64);
    poison.store(true, Ordering::Relaxed);
    let mut cpu = initial_cpu();
    let mut arena = ChildArena::new(4);
    // SDM Vol 2B §4.3 SYSCALL: RCX is next RIP, R11 holds the flags image.
    let flags = cpu.rflags();
    let ChildStep::Syscall {
        evidence,
        instruction_len,
    } = no_alloc(|| child::step(&view, &mut cpu, &mut arena))
    else {
        panic!()
    };
    assert_eq!(evidence.path(), FaultPath::Interpreter);
    assert_eq!(evidence.rip(), 0x1000);
    assert_eq!(instruction_len, 2);
    assert_eq!(evidence.rip() + u64::from(instruction_len), cpu.rip);
    assert_eq!(cpu.gpr[1], cpu.rip);
    assert_eq!(cpu.gpr[11], flags & !(1 << 16));
    cpu.rip = 0x3000;
    assert!(matches!(
        no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
        ChildStep::Fault(_)
    ));
    assert!(arena.written_pages().is_empty());
}

#[test]
fn child_sparse_metadata_preserves_holes_traps_fetch_and_store_refusals() {
    use x86jit_core::{ChildRegion, ChildViewMetadata, FetchVerdict};
    let mut vm = Vm::new(VmConfig::flat(0x10000));
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0x48, 0x89, 0x18]).unwrap();
    // A load spanning adjacent metadata intervals must still access shared RAM.
    vm.write_bytes(0x1100, &[0x48, 0x8b, 0x18]).unwrap();
    vm.write_bytes(0x3ffc, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    vm.set_fetch_policy(Box::new(|_| panic!("caller froze child fetch permissions")));
    vm.mem.mark_code(0x6000, 1);
    let ram = |range: std::ops::Range<u64>, fetch| ChildRegion::Ram {
        backing_offset: range.start as usize,
        range,
        fetch,
    };
    let view = vm
        .child_view_with(
            ChildViewMetadata {
                regions: vec![
                    ram(0x1000..0x2000, FetchVerdict::Allow),
                    ram(0x3000..0x4000, FetchVerdict::DenyProtection),
                    ram(0x4000..0x5000, FetchVerdict::DenyNotPresent),
                    ChildRegion::Trap {
                        range: 0x5000..0x6000,
                    },
                    ram(0x6000..0x7000, FetchVerdict::Allow),
                    ram(0x7000..0x8000, FetchVerdict::Allow),
                ],
                code_store_refusals: vec![7],
            },
            CpuMode::Long64,
        )
        .unwrap();
    let mut cpu = initial_cpu();
    let mut arena = ChildArena::new(4);
    cpu.rip = 0x1100;
    cpu.gpr[0] = 0x3ffc;
    assert!(matches!(
        no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
        ChildStep::Continue
    ));
    assert_eq!(cpu.gpr[3], 0x0807_0605_0403_0201);
    for addr in [0x2000, 0x5000, 0x6000, 0x7000] {
        cpu.rip = 0x1000;
        cpu.gpr[0] = addr;
        let before = format!("{cpu:?}");
        let result = no_alloc(|| child::step(&view, &mut cpu, &mut arena));
        match addr {
            0x2000 => assert!(matches!(result, ChildStep::Fault(_))),
            0x5000 => assert!(matches!(result, ChildStep::Refused(ChildRefusal::Mmio))),
            _ => assert!(matches!(
                result,
                ChildStep::Refused(ChildRefusal::CodeStore)
            )),
        }
        assert_eq!(before, format!("{cpu:?}"));
        assert_eq!(vm.mem.read(addr, 8), Ok(0));
    }
    for (rip, present) in [(0x3000, 1), (0x4000, 0), (0x2000, 0)] {
        cpu.rip = rip;
        let ChildStep::Fault(fault) = no_alloc(|| child::step(&view, &mut cpu, &mut arena)) else {
            panic!()
        };
        let projection = fault.projection.unwrap();
        assert!(projection.is_fetch());
        assert_eq!(
            projection.error_code(),
            x86jit_core::ErrorCode::bits(present, 1)
        );
        assert_eq!(cpu.rip, rip);
    }
    cpu.rip = 0x5000;
    assert!(matches!(
        no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
        ChildStep::Refused(ChildRefusal::Mmio)
    ));
    assert!(arena.written_pages().is_empty());
    // The same frozen policy refuses only fetches: data stores still use RAM.
    cpu.rip = 0x1000;
    cpu.gpr[0] = 0x3000;
    cpu.gpr[3] = 42;
    assert!(matches!(
        no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
        ChildStep::Continue
    ));
    assert_eq!(vm.mem.read(0x3000, 8), Ok(42));
    vm.invalidate_written_pages(&arena);
}

#[test]
fn child_arenas_are_independent_and_parent_reconciles_watches() {
    let vm = vm(&[0x48, 0x89, 0x18]);
    vm.watch_range(0x3000, 0x3000);
    let view = vm.child_view(CpuMode::Long64);
    let arenas = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..3)
            .map(|i| {
                let view = &view;
                scope.spawn(move || {
                    let mut arena = ChildArena::new(4);
                    let mut cpu = initial_cpu();
                    cpu.gpr[0] = 0x3000 + i * 0x1000;
                    assert!(matches!(
                        no_alloc(|| child::step(view, &mut cpu, &mut arena)),
                        ChildStep::Continue
                    ));
                    arena
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(
        vm.take_dirty_ranges().is_empty(),
        "child must not touch parent watch state"
    );
    for arena in &arenas {
        vm.invalidate_written_pages(arena);
    }
    assert_eq!(vm.take_dirty_ranges(), vec![(0x3000, 0x3000)]);
}

#[test]
fn child_nondefault_mxcsr_and_saved_identity_survive() {
    // SDM Vol 1 §10.2.3: MXCSR modes are state even when the next instruction
    // does not modify them. The shared numerical implementation interprets RC.
    for mxcsr in [0x1f80, 0x3f80, 0x5f80, 0x9fc0] {
        let vm = vm(&[0xc5, 0xfc, 0x58, 0xc1]);
        let view = vm.child_view(CpuMode::Long64);
        let mut cpu = initial_cpu();
        cpu.mxcsr = mxcsr;
        cpu.cpuid_table = Some(std::sync::Arc::new(x86jit_core::CpuidTable::new(
            x86jit_core::CpuidOutOfRange::Fixed(x86jit_core::CpuidRegisters::ZERO),
        )));
        cpu.cpuid_overrides.leaves.insert(
            (1, 0),
            x86jit_core::CpuidOverride {
                mask: x86jit_core::CpuidRegisters::ZERO,
                value: x86jit_core::CpuidRegisters::ZERO,
            },
        );
        let mut expected = cpu.clone();
        interp::step_one(
            &vm.mem,
            &mut expected,
            CpuMode::Long64,
            &mut Vec::new(),
            &mut Default::default(),
        );
        let mut arena = ChildArena::new(4);
        assert!(matches!(
            no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
            ChildStep::Continue
        ));
        assert_eq!(format!("{cpu:?}"), format!("{expected:?}"));
        assert_eq!(
            std::sync::Arc::strong_count(cpu.cpuid_table.as_ref().unwrap()),
            2
        );
    }
    for (image, valid) in [(0x5f80u32, true), (0x10000, false)] {
        let vm = vm(&[0x0f, 0xae, 0x10]);
        vm.mem.write_bytes(0x3100, &image.to_le_bytes()).unwrap();
        let view = vm.child_view(CpuMode::Long64);
        let mut cpu = initial_cpu();
        let mut arena = ChildArena::new(4);
        let result = no_alloc(|| child::step(&view, &mut cpu, &mut arena));
        if valid {
            assert!(matches!(result, ChildStep::Continue));
            assert_eq!(cpu.mxcsr, image);
        } else {
            assert!(
                matches!(result, ChildStep::Fault(f) if f.origin == FaultOrigin::GuestException(FaultCause::InvalidMxcsr))
            );
            assert!(arena.written_pages().is_empty());
        }
    }
}

pub(crate) fn locked_code_forms(mode: CpuMode) -> Vec<Vec<u8>> {
    let forms: &[&[u8]] = &[
        &[0x01, 0x1e],
        &[0x29, 0x1e],
        &[0x21, 0x1e],
        &[0x09, 0x1e],
        &[0x31, 0x1e],
        &[0x87, 0x1e],
        &[0x0f, 0xc1, 0x1e],
        &[0xff, 0x06],
        &[0xff, 0x0e],
        &[0xf7, 0x1e],
        &[0xf7, 0x16],
        &[0x0f, 0xab, 0x1e],
        &[0x0f, 0xb3, 0x1e],
        &[0x0f, 0xbb, 0x1e],
        &[0x0f, 0xb1, 0x1e],
    ];
    let mut result = Vec::new();
    for size in [1, 2, 4, 8] {
        if size == 8 && mode == CpuMode::Compat32 {
            continue;
        }
        for (form, bytes) in forms.iter().enumerate() {
            if size == 1 && (11..14).contains(&form) {
                continue;
            }
            let mut code = vec![0xf0];
            if size == 2 {
                code.push(0x66);
            }
            if size == 8 {
                code.push(0x48);
            }
            let mut bytes = bytes.to_vec();
            if size == 1 {
                let i = bytes.len() - 2;
                bytes[i] -= 1;
            }
            code.extend(bytes);
            result.push(code);
        }
    }
    result
}

// SDM Vol 2A §3.2, ADD, INC, DEC, CMPXCHG and CMPXCHG8B/CMPXCHG16B;
// Vol 2B §4.2, NEG, NOT; Vol 2D §6.2, XADD, XCHG, XOR, Flags Affected.
#[test]
fn child_locked_forms_match_step_one_without_allocation() {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for code in locked_code_forms(mode) {
            for value in [0u64, 1, 0x7f, 0x80, 0x7fff_ffff, u64::MAX] {
                let vm = vm(&code);
                vm.mem.write(0x3100, value, 8).unwrap();
                let mut normal = initial_cpu();
                normal.gpr[0] = value;
                normal.gpr[3] = 1;
                normal.flags.cf = true;
                normal.flags.of = true;
                let mut actual = normal.clone();
                assert!(
                    matches!(
                        interp::step_one(
                            &vm.mem,
                            &mut normal,
                            mode,
                            &mut Vec::new(),
                            &mut Default::default()
                        ),
                        x86jit_core::StepResult::Continue
                    ),
                    "{mode:?} {code:x?}"
                );
                let expected = vm.mem.read(0x3100, 8).unwrap();
                vm.mem.write(0x3100, value, 8).unwrap();
                let view = vm.child_view(mode);
                let mut arena = ChildArena::new(4);
                let result = no_alloc(|| child::step(&view, &mut actual, &mut arena));
                assert!(
                    matches!(result, ChildStep::Continue),
                    "{mode:?} {code:x?}: {result:?}"
                );
                assert_eq!(format!("{normal:?}"), format!("{actual:?}"), "{code:x?}");
                assert_eq!(expected, vm.mem.read(0x3100, 8).unwrap(), "{code:x?}");
                assert_eq!(&*arena.written_pages(), &[3]);
            }
        }
        for size in [8, 16] {
            if size == 16 && mode == CpuMode::Compat32 {
                continue;
            }
            let code: &[u8] = if size == 16 {
                &[0xf0, 0x48, 0x0f, 0xc7, 0x0e]
            } else {
                &[0xf0, 0x0f, 0xc7, 0x0e]
            };
            for success in [false, true] {
                let vm = vm(code);
                vm.mem.write(0x3100, 0x1234, 8).unwrap();
                vm.mem.write(0x3108, 0x5678, 8).unwrap();
                let mut normal = initial_cpu();
                normal.gpr[0] = if success { 0x1234 } else { 0xabcd };
                normal.gpr[2] = if size == 16 { 0x5678 } else { 0 };
                normal.flags.cf = true;
                normal.flags.of = true;
                let mut actual = normal.clone();
                assert!(matches!(
                    interp::step_one(
                        &vm.mem,
                        &mut normal,
                        mode,
                        &mut Vec::new(),
                        &mut Default::default()
                    ),
                    x86jit_core::StepResult::Continue
                ));
                let mut expected = [0; 16];
                vm.mem.read_bytes(0x3100, &mut expected).unwrap();
                vm.mem.write(0x3100, 0x1234, 8).unwrap();
                vm.mem.write(0x3108, 0x5678, 8).unwrap();
                let view = vm.child_view(mode);
                let mut arena = ChildArena::new(4);
                let result = no_alloc(|| child::step(&view, &mut actual, &mut arena));
                assert!(
                    matches!(result, ChildStep::Continue),
                    "{code:x?}: {result:?}"
                );
                assert_eq!(format!("{normal:?}"), format!("{actual:?}"));
                let mut bytes = [0; 16];
                vm.mem.read_bytes(0x3100, &mut bytes).unwrap();
                assert_eq!(expected, bytes);
            }
        }
    }
}

#[test]
fn child_locked_counters_are_atomic_against_host_threads() {
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Barrier,
    };
    const THREADS: usize = 8;
    const ITERATIONS: u64 = 2000;
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for cas in [false, true] {
            // Both modes use a 32-bit operand; the counter cannot overflow.
            let vm = vm(if cas {
                &[0xf0, 0x0f, 0xb1, 0x1e]
            } else {
                &[0xf0, 0x0f, 0xc1, 0x1e]
            });
            let view = vm.child_view(mode);
            // SAFETY: aligned RAM retained by vm; all concurrent data accesses
            // use atomics (host u32, or the child's host atomic instructions).
            let word = unsafe {
                std::sync::atomic::AtomicU32::from_ptr(
                    vm.mem.host_base().add(0x3100).cast_mut().cast(),
                )
            };
            let barrier = Barrier::new(THREADS);
            let attempts = AtomicU64::new(0);
            std::thread::scope(|scope| {
                for thread in 0..THREADS {
                    let (view, barrier, attempts) = (&view, &barrier, &attempts);
                    scope.spawn(move || {
                        let mut arena = ChildArena::new(4);
                        let mut cpu = initial_cpu();
                        cpu.gpr[0] = 0;
                        barrier.wait();
                        if thread < THREADS / 2 {
                            for _ in 0..ITERATIONS {
                                loop {
                                    cpu.rip = 0x1000;
                                    cpu.gpr[3] = if cas { cpu.gpr[0] + 1 } else { 1 };
                                    let result =
                                        no_alloc(|| child::step(view, &mut cpu, &mut arena));
                                    assert!(matches!(result, ChildStep::Continue));
                                    attempts.fetch_add(1, Ordering::Relaxed);
                                    if !cas || cpu.flags.zf {
                                        break;
                                    }
                                }
                            }
                        } else {
                            for _ in 0..ITERATIONS {
                                word.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    });
                }
            });
            assert_eq!(
                word.load(Ordering::SeqCst),
                (THREADS as u64 * ITERATIONS) as u32
            );
            assert!(attempts.load(Ordering::Relaxed) >= THREADS as u64 / 2 * ITERATIONS);
        }
    }
}

// SDM Vol 3A §11.2.3.9: earlier stores and later loads cannot cross a
// locked instruction. The two threads must not both read the initial zero.
#[test]
fn child_locked_operations_are_full_barriers() {
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Barrier,
    };
    let vm = vm(&[0xf0, 0x48, 0x87, 0x1e]);
    let view = vm.child_view(CpuMode::Long64);
    // SAFETY: live, aligned RAM; only atomic accesses until both threads join.
    let words = unsafe {
        [0x3100, 0x3180].map(|at| AtomicU64::from_ptr(vm.mem.host_base().add(at).cast_mut().cast()))
    };
    let barrier = Barrier::new(2);
    let results = [AtomicU64::new(0), AtomicU64::new(0)];
    std::thread::scope(|scope| {
        for thread in 0..2 {
            let (view, barrier, results, words) = (&view, &barrier, &results, &words);
            scope.spawn(move || {
                let mut arena = ChildArena::new(4);
                let mut cpu = initial_cpu();
                cpu.gpr[6] = 0x3300 + thread as u64 * 128;
                for _ in 0..2000 {
                    words[thread].store(0, Ordering::Relaxed);
                    barrier.wait();
                    words[thread].store(1, Ordering::Relaxed);
                    cpu.rip = 0x1000;
                    assert!(matches!(
                        no_alloc(|| child::step(view, &mut cpu, &mut arena)),
                        ChildStep::Continue
                    ));
                    results[thread]
                        .store(words[1 - thread].load(Ordering::Relaxed), Ordering::Relaxed);
                    barrier.wait();
                    assert!(
                        results[0].load(Ordering::Relaxed) != 0
                            || results[1].load(Ordering::Relaxed) != 0
                    );
                    barrier.wait();
                }
            });
        }
    });
}

#[test]
fn child_locked_architectural_witnesses() {
    // SDM Vol 2A §3.2, INC/DEC, Flags Affected: CF is preserved;
    // the other arithmetic flags describe the increment/decrement result.
    for (code, input, output, sign) in [
        (&[0xf0, 0xfe, 0x06][..], 0x7f, 0x80, true),
        (&[0xf0, 0xfe, 0x0e], 0x80, 0x7f, false),
    ] {
        let vm = vm(code);
        vm.mem.write(0x3100, input, 1).unwrap();
        let view = vm.child_view(CpuMode::Long64);
        let mut arena = ChildArena::new(4);
        let mut cpu = initial_cpu();
        cpu.flags.cf = true;
        assert!(matches!(
            no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
            ChildStep::Continue
        ));
        assert_eq!(vm.mem.read(0x3100, 1).unwrap(), output);
        assert!(cpu.flags.cf);
        assert!(cpu.flags.of);
        assert!(cpu.flags.af());
        assert!(!cpu.flags.pf());
        assert!(!cpu.flags.zf);
        assert_eq!(cpu.flags.sf, sign);
    }
    // SDM Vol 2A §3.2, CMPXCHG, Operation; Vol 1 §3.4.1.1:
    // success preserves RAX, failure writes and zero-extends EAX.
    for success in [false, true] {
        let vm = vm(&[0xf0, 0x0f, 0xb1, 0x1e]);
        vm.mem.write(0x3100, 0x1122_3344, 4).unwrap();
        let view = vm.child_view(CpuMode::Long64);
        let mut arena = ChildArena::new(4);
        let mut cpu = initial_cpu();
        cpu.gpr[0] = if success {
            0xfeed_beef_1122_3344
        } else {
            0xfeed_beef_0000_0001
        };
        assert!(matches!(
            no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
            ChildStep::Continue
        ));
        assert_eq!(
            cpu.gpr[0],
            if success {
                0xfeed_beef_1122_3344
            } else {
                0x1122_3344
            }
        );
        assert_eq!(cpu.flags.zf, success);
        assert_eq!(
            vm.mem.read(0x3100, 4).unwrap(),
            if success { 7 } else { 0x1122_3344 }
        );
    }
    // SDM Vol 2A §3.2, CMPXCHG8B/CMPXCHG16B, 64-Bit Mode
    // Exceptions: CMPXCHG16B requires 16-byte alignment and raises #GP(0).
    let vm = vm(&[0xf0, 0x48, 0x0f, 0xc7, 0x0e]);
    let view = vm.child_view(CpuMode::Long64);
    let mut arena = ChildArena::new(4);
    let mut cpu = initial_cpu();
    cpu.gpr[6] = 0x3108;
    let before = format!("{cpu:?}");
    let ChildStep::Fault(f) = no_alloc(|| child::step(&view, &mut cpu, &mut arena)) else {
        panic!();
    };
    assert_eq!(f.vector, Knowledge::Known(13));
    assert_eq!(
        f.origin,
        FaultOrigin::GuestException(FaultCause::RequiredAlignment)
    );
    assert_eq!(f.error_code, x86jit_core::ErrorCode::bits(0, u32::MAX));
    assert_eq!(before, format!("{cpu:?}"));
    assert!(arena.written_pages().is_empty());
}
