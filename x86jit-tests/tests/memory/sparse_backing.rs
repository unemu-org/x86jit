//! Sparse host backing (§4.1): checked guest accesses must never touch a hole.
#![cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]

use iced_x86::code_asm::*;
use x86jit_core::{
    AccessKind, Backend, CpuMode, CpuModel, CpuState, Exit, ForkError, FpuModel,
    InterpreterBackend, Memory, MemoryModel, Prot, RegionCaps, RegionKind, ResetOptions,
    SparseError, StepOptions, StepStop, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const SPAN: u64 = 1 << 32;
const LOW: usize = 64 << 20;
const HIGH: u64 = 0xfffe_0000;
const CODE: u64 = 0x1000;
const CAPS: RegionCaps = RegionCaps {
    max_blocks: 16,
    max_icount: 256,
};

fn backend(tier: u8) -> Box<dyn Backend> {
    match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        _ => Box::new(JitBackend::with_superblocks(CAPS)),
    }
}

fn app(tier: u8, sparse: bool, code: &[u8]) -> Vm {
    let config = VmConfig::flat(0x10000);
    let mut vm = if sparse {
        Vm::with_backend_sparse(config, backend(tier)).unwrap()
    } else {
        Vm::with_backend(config, backend(tier))
    };
    vm.set_tier_up_after(None);
    vm.map(CODE, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
    vm.write_bytes(CODE, code).unwrap();
    if tier == 2 && !sparse {
        vm.test_support_force_region(CODE, CAPS, 0x1f80).unwrap();
    }
    vm
}

fn assert_execution_tier(vm: &Vm, tier: u8) {
    if vm.mem.is_sparse() {
        assert_eq!(vm.cache.memory_stats().compiled_entries, 0);
        assert_eq!(vm.cache.regions(), 0);
    } else if tier == 1 {
        assert!(vm.cache.memory_stats().compiled_entries > 0);
    } else if tier == 2 {
        assert!(vm.cache.regions() > 0);
    }
}

#[test]
fn reset_fetch_ram_rom_smc_device_and_reset_on_four_gib() {
    // Reset and CS reload: 80386 PRM §10.1–10.2.3; SDM Vol 3A §12.1.4.
    for tier in 0..2 {
        let mut vm = Vm::with_backend_sparse(VmConfig::reserved(SPAN), backend(tier)).unwrap();
        vm.set_cpu_model(CpuModel::I80386);
        vm.map(0, LOW, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(HIGH, 128 << 10, Prot::RX, RegionKind::Rom).unwrap();
        vm.write_bytes(0xffff_fff0, b"\xe9\x0d\xff").unwrap(); // near JMP FF00h
        vm.write_bytes(0xffff_ff00, b"\xea\x00\x02\x00\xf0")
            .unwrap();
        vm.write_bytes(
            0xf0200,
            b"\xb8\x34\x12\xa3\x00\x20\x8b\x1e\x00\x20\xe4\x80\xf4",
        )
        .unwrap();
        vm.watch_range(0x2000, 2);
        let mut cpu = vm.new_vcpu();
        for value in [0x1234, 0x5678] {
            cpu.reset_execution_state(ResetOptions::default());
            cpu.cpu = CpuState::i386_reset(FpuModel::Present);
            let run = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 2);
            assert_eq!(cpu.cpu.legacy.segments[0].base, 0xf0000);
            assert_eq!(cpu.cpu.rip, 0x200);
            assert!(matches!(
                cpu.run(&vm, Some(64)),
                Exit::PortIo { port: 0x80, .. }
            ));
            assert_eq!(cpu.cpu.gpr[3], value);
            let mut bytes = [0; 2];
            vm.read_bytes(0x2000, &mut bytes).unwrap();
            assert_eq!(u16::from_le_bytes(bytes), value as u16);
            assert_eq!(vm.take_dirty_ranges(), vec![(0x2000, 4096)]);
            cpu.complete_port_in(0x5a);
            assert!(matches!(cpu.run(&vm, Some(64)), Exit::Hlt));
            // Code was fetched and cached; a loader write must invalidate it.
            vm.write_bytes(0xf0201, &0x5678u16.to_le_bytes()).unwrap();
        }
        let mut byte = [0];
        vm.read_bytes(0xffff_fff0, &mut byte).unwrap();
        vm.mem.write(0xffff_fff0, 0, 1).unwrap();
        let mut after = [0];
        vm.read_bytes(0xffff_fff0, &mut after).unwrap();
        assert_eq!(after, byte, "guest ROM stores are discarded");
        assert!(vm.mem.deep_copy().is_none());
        assert!(matches!(
            vm.try_fork_with_backend(backend(0)),
            Err(ForkError::SparseBacking)
        ));
    }
}

#[test]
fn scalar_vector_string_atomic_and_fp_gaps_match_checked_flat_memory() {
    let mut snippets = Vec::new();
    for store in [false, true] {
        let mut a = CodeAssembler::new(64).unwrap();
        if store {
            a.mov(qword_ptr(rdi), rax).unwrap();
        } else {
            a.mov(rax, qword_ptr(rdi)).unwrap();
        }
        a.hlt().unwrap();
        snippets.push(a.assemble(CODE).unwrap());
    }
    for store in [false, true] {
        let mut a = CodeAssembler::new(64).unwrap();
        if store {
            a.movdqu(xmmword_ptr(rdi), xmm0).unwrap();
        } else {
            a.movdqu(xmm0, xmmword_ptr(rdi)).unwrap();
        }
        a.hlt().unwrap();
        snippets.push(a.assemble(CODE).unwrap());
    }
    let mut a = CodeAssembler::new(64).unwrap();
    a.lock().add(qword_ptr(rdi), 1).unwrap();
    a.hlt().unwrap();
    snippets.push(a.assemble(CODE).unwrap());
    let mut a = CodeAssembler::new(64).unwrap();
    a.rep().movsb().unwrap();
    a.hlt().unwrap();
    snippets.push(a.assemble(CODE).unwrap());
    let mut a = CodeAssembler::new(64).unwrap();
    a.fld(qword_ptr(rdi)).unwrap();
    a.hlt().unwrap();
    snippets.push(a.assemble(CODE).unwrap());
    let mut a = CodeAssembler::new(64).unwrap();
    a.fxsave(ptr(rdi)).unwrap();
    a.hlt().unwrap();
    snippets.push(a.assemble(CODE).unwrap());

    for code in snippets {
        for addr in [0x8000, 0x3ffc] {
            let baseline = app(0, false, &code);
            let mut reference = baseline.new_vcpu();
            reference.cpu.rip = CODE;
            reference.cpu.gpr[0] = 0x1122;
            reference.cpu.gpr[1] = 16;
            reference.cpu.gpr[6] = 0x3000;
            reference.cpu.gpr[7] = addr;
            let before = reference.cpu.clone();
            let expected = reference.run(&baseline, Some(64));
            assert!(
                matches!(
                    expected,
                    Exit::UnmappedMemory { .. } | Exit::Exception { .. }
                ),
                "{expected:?}"
            );
            for tier in 0..3 {
                let vm = app(tier, true, &code);
                let mut cpu = vm.new_vcpu();
                cpu.cpu = before.clone();
                let actual = cpu.run(&vm, Some(64));
                match (&actual, &expected) {
                    (Exit::UnmappedMemory { addr: a, access: aa, .. }, Exit::UnmappedMemory { addr: b, access: ba, .. }) => {
                        assert_eq!((a, aa), (b, ba));
                        let a = actual.fault_metadata().unwrap();
                        let b = expected.fault_metadata().unwrap();
                        assert_eq!(a.operand_size(), b.operand_size());
                        assert_eq!(a.transfer_size(), b.transfer_size());
                    }
                    (Exit::Exception { vector: a, .. }, Exit::Exception { vector: b, .. }) => assert_eq!(a, b),
                    _ => panic!("actual={actual:?}, expected={expected:?}, tier={tier}, addr={addr:#x}, code={code:02x?}"),
                }
                assert_eq!(cpu.cpu.gpr, reference.cpu.gpr);
                assert_eq!(cpu.cpu.xmm, reference.cpu.xmm);
                assert_eq!(cpu.cpu.rflags(), reference.cpu.rflags());
                assert_execution_tier(&vm, tier);
            }
        }
    }
}

#[test]
fn sparse_fault_preserves_prior_flags_and_instruction_state() {
    // SDM Vol 2A, ADD: FFFFFFFFh + 1 sets CF and ZF. The subsequent memory
    // instruction must fault without changing its destination or these flags.
    let code = b"\xb8\xff\xff\xff\xff\x83\xc0\x01\x48\x8b\x1f\xf4";
    for tier in 0..3 {
        let vm = app(tier, true, code);
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = CODE;
        cpu.cpu.gpr[3] = 0x1234;
        cpu.cpu.gpr[7] = 0x8000;
        assert!(matches!(
            cpu.run(&vm, Some(64)),
            Exit::UnmappedMemory { addr: 0x8000, .. }
        ));
        assert_eq!(cpu.cpu.rip, CODE + 8);
        assert_eq!(cpu.cpu.gpr[0], 0);
        assert_eq!(cpu.cpu.gpr[3], 0x1234);
        assert_eq!(cpu.cpu.rflags() & 0x41, 0x41);
        assert_execution_tier(&vm, tier);
    }
}

#[test]
fn child_view_copies_cpu_only_and_checks_sparse_gaps() {
    use x86jit_core::{child, ChildArena, ChildStep};
    let vm = app(0, true, b"\x48\x8b\x07\xf4");
    vm.write_bytes(0x3000, &0x1234u64.to_le_bytes()).unwrap();
    let view = vm.child_view(CpuMode::Long64);
    let mut arena = ChildArena::new(8);
    let mut cpu = CpuState::new();
    cpu.rip = CODE;
    cpu.gpr[7] = 0x3000;
    let checkpoint = cpu.clone();
    assert!(matches!(
        child::step(&view, &mut cpu, &mut arena),
        ChildStep::Continue
    ));
    assert_eq!(cpu.gpr[0], 0x1234);
    cpu = checkpoint;
    cpu.gpr[7] = 0x8000;
    assert!(matches!(
        child::step(&view, &mut cpu, &mut arena),
        ChildStep::Fault(_)
    ));
    assert_eq!(cpu.gpr[0], 0);
    assert_eq!(cpu.rip, CODE);
}

#[test]
fn mapped_helpers_smc_and_mmio_with_injected_backends() {
    for tier in 0..3 {
        let mut a = CodeAssembler::new(64).unwrap();
        a.rep().movsb().unwrap();
        a.lock().add(dword_ptr(0x3000), 1).unwrap();
        a.mov(eax, dword_ptr(0x8000)).unwrap();
        a.hlt().unwrap();
        let mut vm = app(tier, true, &a.assemble(CODE).unwrap());
        vm.map(0x8000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.write_bytes(0x3080, &[7, 0, 0, 0]).unwrap();
        vm.watch_range(0x3000, 4);
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = CODE;
        cpu.cpu.gpr[1] = 4;
        cpu.cpu.gpr[6] = 0x3080;
        cpu.cpu.gpr[7] = 0x3000;
        assert!(matches!(
            cpu.run(&vm, Some(64)),
            Exit::MmioRead {
                addr: 0x8000,
                size: 4,
                ..
            }
        ));
        cpu.complete_mmio_read(0x1234);
        assert!(matches!(cpu.run(&vm, Some(64)), Exit::Hlt));
        assert_eq!(cpu.cpu.gpr[0], 0x1234);
        assert_eq!(vm.mem.read(0x3000, 4).unwrap(), 8);
        assert_eq!(vm.take_dirty_ranges(), vec![(0x3000, 4096)]);
        assert_execution_tier(&vm, tier);
        vm.write_bytes(CODE, b"\xb8\x78\x56\x00\x00\xf4").unwrap();
        cpu.reset_execution_state(ResetOptions::default());
        cpu.cpu.rip = CODE;
        assert!(matches!(cpu.run(&vm, Some(64)), Exit::Hlt));
        assert_eq!(cpu.cpu.gpr[0], 0x5678);
        vm.unmap(CODE, 0x1000).unwrap();
        cpu.cpu.rip = CODE;
        assert!(matches!(
            cpu.run(&vm, Some(64)),
            Exit::UnmappedMemory {
                access: AccessKind::Execute,
                ..
            }
        ));
    }
}

#[test]
fn host_page_edges_adjacent_regions_and_zero_after_remap() {
    let page = x86jit_core::memory::host_page_size();
    let mut memory = Memory::sparse(MemoryModel::Reserved {
        span: (4 * page) as u64,
    })
    .unwrap();
    // Deliberately sub-page, including sharing on 16 KiB Darwin hosts.
    memory.map(1, 31, Prot::RW, RegionKind::Ram).unwrap();
    memory.map(32, 32, Prot::RX, RegionKind::Rom).unwrap();
    memory.write_bytes(1, &[0xaa; 31]).unwrap();
    memory.write_bytes(32, &[0xbb; 32]).unwrap();
    memory.unmap(1, 31).unwrap();
    assert_eq!(memory.read(32, 8).unwrap(), 0xbbbb_bbbb_bbbb_bbbb);
    assert!(memory.read_bytes(1, &mut [0]).is_err());
    memory.map(1, 31, Prot::RW, RegionKind::Ram).unwrap();
    assert_eq!(memory.read(1, 8).unwrap(), 0);
    memory.unmap(1, 31).unwrap();
    memory.unmap(32, 32).unwrap();
    memory.map(0, 2 * page, Prot::RW, RegionKind::Ram).unwrap();
    assert_eq!(memory.read((page - 4) as u64, 8).unwrap(), 0);
    memory.write_bytes(0, &[0xff; 64]).unwrap();
    memory.unmap(0, 2 * page).unwrap();
    memory.map(0, 2 * page, Prot::RW, RegionKind::Ram).unwrap();
    assert_eq!(memory.read(0, 8).unwrap(), 0);
}

#[test]
fn invalid_reservations_are_typed_errors() {
    assert!(matches!(
        Vm::sparse(VmConfig::reserved(u64::MAX)),
        Err(SparseError::InvalidSpan)
    ));
    assert!(matches!(
        Memory::sparse(MemoryModel::SoftMmu),
        Err(SparseError::UnsupportedModel)
    ));
    assert!(matches!(
        Vm::sparse(VmConfig::flat(0)),
        Err(SparseError::InvalidSpan)
    ));
}

#[test]
fn application_modes_fetch_high_firmware_and_fault_on_gap() {
    for mode in [CpuMode::Long64, CpuMode::Compat32, CpuMode::Real16] {
        for tier in 0..2 {
            let mut vm = Vm::with_backend_sparse(VmConfig::flat(SPAN), backend(tier)).unwrap();
            vm.set_cpu_mode(mode);
            vm.map(HIGH, 128 << 10, Prot::RX, RegionKind::Rom).unwrap();
            // Long64/Compat32 fetch high directly. Real16 uses a low CS:IP window.
            let entry = if mode == CpuMode::Real16 {
                vm.map(0x1000, 4096, Prot::RWX, RegionKind::Ram).unwrap();
                0x1000
            } else {
                HIGH
            };
            vm.write_bytes(entry, b"\xb8\x34\x12\x00\x00\xf4").unwrap();
            if mode == CpuMode::Real16 {
                vm.write_bytes(entry, b"\xb8\x34\x12\xf4").unwrap();
            }
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = entry;
            assert!(matches!(cpu.run(&vm, Some(64)), Exit::Hlt));
            assert_eq!(cpu.cpu.gpr[0], 0x1234);
            cpu.cpu.rip = if mode == CpuMode::Real16 {
                0x8000
            } else {
                0x8000_0000
            };
            assert!(matches!(
                cpu.run(&vm, Some(64)),
                Exit::UnmappedMemory {
                    access: AccessKind::Execute,
                    ..
                }
            ));
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn reservation_resident_growth_is_far_below_the_span() {
    fn rss() -> u64 {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: getrusage initializes the output structure on success.
        assert_eq!(
            unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
            0
        );
        let rss = unsafe { usage.assume_init() }.ru_maxrss as u64;
        if cfg!(target_os = "linux") {
            rss * 1024
        } else {
            rss
        }
    }
    let before = rss();
    let vm = Vm::sparse(VmConfig::reserved(SPAN)).unwrap();
    assert!(rss().saturating_sub(before) < 256 << 20);
    assert_eq!(vm.mem.size(), SPAN);
}

#[test]
fn replacing_public_memory_invalidates_both_backing_policies() {
    let code = b"\x48\x8b\x07\xeb\x00\xf4";
    for tier in 1..3 {
        let mut vm = app(tier, false, code);
        vm.map(0x8000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
        vm.write_bytes(0x8000, &0x1234u64.to_le_bytes()).unwrap();
        let mut cpus = [vm.new_vcpu(), vm.new_vcpu()];
        for cpu in &mut cpus {
            for _ in 0..2 {
                cpu.cpu.rip = CODE;
                cpu.cpu.gpr[7] = 0x8000;
                assert!(matches!(cpu.run(&vm, Some(64)), Exit::Hlt));
                assert_eq!(cpu.cpu.gpr[0], 0x1234);
            }
        }
        assert_execution_tier(&vm, tier);
        let mut sparse = Memory::sparse(MemoryModel::Flat { size: 0x10000 }).unwrap();
        sparse
            .map(CODE, 0x1000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        sparse.write_bytes(CODE, code).unwrap();
        vm.mem = sparse;
        let generation = vm.cache.publication_generation();
        for cpu in &mut cpus {
            cpu.cpu.rip = CODE;
            cpu.cpu.gpr[0] = 0x5678;
            assert!(matches!(
                cpu.run(&vm, Some(64)),
                Exit::UnmappedMemory {
                    addr: 0x8000,
                    access: AccessKind::Read,
                    ..
                }
            ));
            assert_eq!(cpu.cpu.rip, CODE);
            assert_eq!(cpu.cpu.gpr[0], 0x5678);
        }
        assert!(vm.cache.publication_generation() > generation);
        assert_eq!(vm.cache.memory_stats().compiled_entries, 0);

        let mut ordinary = Memory::new(MemoryModel::Flat { size: 0x10000 });
        ordinary
            .map(CODE, 0x1000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        ordinary
            .write_bytes(CODE, b"\xb8\xbc\x9a\x00\x00\xf4")
            .unwrap();
        vm.mem = ordinary;
        let generation = vm.cache.publication_generation();
        for cpu in &mut cpus {
            cpu.cpu.rip = CODE;
            assert!(matches!(cpu.run(&vm, Some(64)), Exit::Hlt));
            assert_eq!(cpu.cpu.gpr[0], 0x9abc);
        }
        assert!(vm.cache.publication_generation() > generation);
        assert!(vm.cache.memory_stats().compiled_entries > 0);
    }
}

#[test]
fn sparse_icount_is_exact_with_block_and_region_backends() {
    for regions in [false, true] {
        for hot in [false, true] {
            let jit = if regions {
                JitBackend::with_superblocks(CAPS)
            } else {
                JitBackend::new()
            };
            jit.enable_icount();
            let mut vm = Vm::with_backend_sparse(VmConfig::flat(0x10000), Box::new(jit)).unwrap();
            if hot {
                vm.set_tier_up_after(Some(1));
                vm.set_tier_up_background(true);
            }
            vm.map(CODE, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x8000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
            vm.write_bytes(CODE, b"\x48\x8b\x07\x48\x8b\x1f\xf4")
                .unwrap();
            vm.write_bytes(0x8000, &0x1234u64.to_le_bytes()).unwrap();
            assert!(matches!(
                vm.test_support_force_region(CODE, CAPS, 0x1f80),
                Err(x86jit_core::ForceRegionError::NotCompiled)
            ));
            let mut cpu = vm.new_vcpu();
            cpu.cpu.gpr[7] = 0x8000;
            for run in 1..=3 {
                cpu.cpu.rip = CODE;
                assert!(matches!(cpu.run(&vm, Some(64)), Exit::Hlt));
                assert_eq!(cpu.executed_instructions(), 3 * run);
                assert_eq!(cpu.retired_instructions(), 3 * run);
                assert_eq!(cpu.cpu.gpr[0], 0x1234);
                assert_eq!(cpu.cpu.gpr[3], 0x1234);
                assert_execution_tier(&vm, 1);
                assert_eq!(vm.cache.tier_pending_len(), 0);
            }
        }
    }
}

#[test]
fn sparse_run_steps_counts_each_load_and_halt_once() {
    for tier in 0..3 {
        let mut vm = Vm::with_backend_sparse(VmConfig::flat(0x10000), backend(tier)).unwrap();
        vm.set_cpu_mode(CpuMode::Real16);
        vm.map(CODE, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x8000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
        vm.write_bytes(CODE, b"\x8b\x07\x8b\x57\x02\xf4").unwrap();
        vm.write_bytes(0x8000, &[0x34, 0x12, 0x78, 0x56]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = CODE;
        cpu.cpu.gpr[3] = 0x8000;
        for (steps, rip) in [(1, CODE + 2), (2, CODE + 5)] {
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1);
            assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
            assert_eq!(cpu.cpu.rip, rip);
            assert_eq!(cpu.retired_instructions(), steps);
            assert_eq!(cpu.executed_instructions(), steps);
        }
        assert_eq!(cpu.cpu.gpr[0], 0x1234);
        assert_eq!(cpu.cpu.gpr[2], 0x5678);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1);
        assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)));
        assert_eq!(cpu.retired_instructions(), 3);
        assert_eq!(cpu.executed_instructions(), 3);
        assert_execution_tier(&vm, tier);
    }
}

#[test]
fn memory_stats_report_the_committed_bytes_of_a_sparse_backing() {
    let mut vm = Vm::sparse(VmConfig::reserved(SPAN)).unwrap();
    let stats = |vm: &Vm| vm.memory_stats().guest_ram_committed;
    assert_eq!(stats(&vm), Some(0));
    // Regions aligned to 64 KiB commit exactly their size on every supported host page size.
    vm.map(0, LOW, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(HIGH, 128 << 10, Prot::RX, RegionKind::Rom).unwrap();
    assert_eq!(stats(&vm), Some(LOW as u64 + (128 << 10)));
    assert_eq!(vm.memory_stats().guest_ram_reserved, SPAN);
    // Two small regions in one host page share it: it counts once, before and after one unmap.
    let page_base = 0x1000_0000;
    vm.map(page_base, 0x100, Prot::RW, RegionKind::Ram).unwrap();
    let one = stats(&vm).unwrap();
    assert!(one > LOW as u64 + (128 << 10));
    vm.map(page_base + 0x100, 0x100, Prot::RW, RegionKind::Ram)
        .unwrap();
    assert_eq!(stats(&vm), Some(one));
    vm.unmap(page_base, 0x100).unwrap();
    assert_eq!(stats(&vm), Some(one));
    vm.unmap(page_base + 0x100, 0x100).unwrap();
    assert_eq!(stats(&vm), Some(LOW as u64 + (128 << 10)));
    vm.unmap(HIGH, 128 << 10).unwrap();
    assert_eq!(stats(&vm), Some(LOW as u64));
}
