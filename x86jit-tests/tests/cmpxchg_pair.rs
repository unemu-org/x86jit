//! SDM Vol 2A, CMPXCHG8B/CMPXCHG16B, Operation, Flags Affected and
//! 64-Bit Mode Exceptions: paired compare-exchange and precise alignment faults.
use x86jit_core::{Backend, Exit, InterpreterBackend, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

fn backends() -> [Box<dyn Backend>; 3] {
    [
        Box::new(InterpreterBackend),
        Box::new(JitBackend::new()),
        Box::new(JitBackend::with_superblocks(x86jit_core::RegionCaps {
            max_blocks: 8,
            max_icount: 128,
        })),
    ]
}
fn vm(backend: Box<dyn Backend>, wide: bool, locked: bool) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x4000), backend);
    vm.map(0, 0x4000, Prot::RX, RegionKind::Ram).unwrap();
    let mut code = Vec::new();
    if locked {
        code.push(0xf0);
    }
    if wide {
        code.push(0x48);
    }
    code.extend_from_slice(&[0x0f, 0xc7, 0x0f, 0xf4]); // [rdi]; hlt
    vm.write_bytes(0x1000, &code).unwrap();
    vm
}

#[test]
fn cmpxchg_pair_semantics() {
    for wide in [false, true] {
        for locked in [false, true] {
            for case in 0..3 {
                let equal = case == 0;
                for backend in backends() {
                    let vm = vm(backend, wide, locked);
                    let addr = if wide { 0x2000 } else { 0x2001 };
                    let size = if wide { 16 } else { 8 };
                    let old: u128 = if wide { (7 << 64) | 5 } else { (7 << 32) | 5 };
                    vm.write_bytes(addr, &old.to_le_bytes()[..size]).unwrap();
                    vm.watch_range(addr, size as u64);
                    vm.mem.mark_code(addr, size as u32);
                    let mut cpu = vm.new_vcpu();
                    cpu.set_reg(Reg::Rip, 0x1000);
                    cpu.set_reg(Reg::Rdi, addr);
                    let upper = if wide { 0 } else { 0xdead_beef_0000_0000 };
                    cpu.set_reg(Reg::Rax, upper | if case == 1 { 6 } else { 5 });
                    cpu.set_reg(Reg::Rdx, upper | if case == 2 { 8 } else { 7 });
                    cpu.set_reg(Reg::Rbx, 9);
                    cpu.set_reg(Reg::Rcx, 11);
                    cpu.cpu.flags.cf = true;
                    cpu.cpu.flags.of = true;
                    cpu.cpu.flags.pf_src = if equal { 0 } else { 1 };
                    cpu.cpu.flags.af_src = if wide { 0x10 } else { 0 };
                    cpu.cpu.flags.sf = true;
                    let initial = cpu.cpu.flags;
                    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                    let mut expected_flags = initial;
                    expected_flags.zf = equal;
                    assert_eq!(cpu.cpu.flags, expected_flags);
                    assert_eq!(cpu.reg(Reg::Rax), if equal { upper | 5 } else { 5 });
                    assert_eq!(cpu.reg(Reg::Rdx), if equal { upper | 7 } else { 7 });
                    let mut got = [0u8; 16];
                    vm.read_bytes(addr, &mut got[..size]).unwrap();
                    let new: u128 = if wide { (11 << 64) | 9 } else { (11 << 32) | 9 };
                    assert_eq!(u128::from_le_bytes(got), if equal { new } else { old });
                    assert!(!vm.take_dirty_ranges().is_empty());
                    assert!(vm.mem.has_dirty_code());
                }
            }
        }
    }
}

#[test]
fn cmpxchg16b_alignment_fault_is_precise() {
    for locked in [false, true] {
        for backend in backends() {
            let vm = vm(backend, true, locked);
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rdi, 0x2001);
            cpu.set_reg(Reg::Rax, 17);
            cpu.set_reg(Reg::Rdx, 19);
            cpu.set_reg(Reg::Rbx, 23);
            cpu.set_reg(Reg::Rcx, 29);
            let before = cpu.cpu.gpr;
            let flags = cpu.cpu.flags;
            assert!(matches!(
                cpu.run(&vm, None),
                Exit::Exception {
                    vector: 13,
                    addr: 0x1000,
                    ..
                }
            ));
            assert_eq!(cpu.reg(Reg::Rax), 17);
            assert_eq!(cpu.reg(Reg::Rdx), 19);
            assert_eq!(cpu.cpu.gpr, before);
            assert_eq!(cpu.cpu.flags, flags);
            let mut got = [1u8; 16];
            vm.read_bytes(0x2001, &mut got).unwrap();
            assert_eq!(got, [0; 16]);
        }
    }
}

#[test]
fn cmpxchg_pair_register_operand_is_ud() {
    for wide in [false, true] {
        for locked in [false, true] {
            for backend in backends() {
                let vm = vm(backend, wide, locked);
                let mut code = Vec::new();
                if locked {
                    code.push(0xf0);
                }
                if wide {
                    code.push(0x48);
                }
                code.extend_from_slice(&[0x0f, 0xc7, 0xc8, 0xf4]);
                vm.write_bytes(0x1000, &code).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, 0x1000);
                let before = cpu.cpu.gpr;
                assert!(matches!(
                    cpu.run(&vm, None),
                    Exit::Exception {
                        vector: 6,
                        addr: 0x1000,
                        ..
                    }
                ));
                assert_eq!(cpu.cpu.gpr, before);
            }
        }
    }
}

#[test]
fn cmpxchg_pair_unmapped_operand_is_precise() {
    use x86jit_core::AccessKind;
    for wide in [false, true] {
        // In-span holes, an operand crossing the backing end, and out-of-span RAM.
        for addr in [0x2000, 0x3ff0, 0x4000] {
            for backend in backends() {
                let mut vm = vm(backend, wide, true);
                vm.mem.unmap(0, 0x4000).unwrap();
                vm.map(0x1000, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.set_reg(Reg::Rdi, addr);
                cpu.set_reg(Reg::Rax, 41);
                let before = cpu.cpu.gpr;
                let flags = cpu.cpu.flags;
                assert!(
                    matches!(cpu.run(&vm, None), Exit::UnmappedMemory { addr: a, access: AccessKind::Write, .. } if a == addr)
                );
                assert_eq!(cpu.reg(Reg::Rip), 0x1000);
                assert_eq!(cpu.cpu.gpr, before);
                assert_eq!(cpu.cpu.flags, flags);
            }
        }
    }
}

fn counter_code(iters: u64) -> Vec<u8> {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(64).unwrap();
    let mut retry = a.create_label();
    a.mov(rdi, 0x2000u64).unwrap();
    a.mov(r8, iters).unwrap();
    a.xor(eax, eax).unwrap();
    a.xor(edx, edx).unwrap();
    a.set_label(&mut retry).unwrap();
    a.mov(rbx, rax).unwrap();
    a.mov(rcx, rdx).unwrap();
    a.add(rbx, 1).unwrap();
    a.adc(rcx, 0).unwrap();
    a.lock().cmpxchg16b(xmmword_ptr(rdi)).unwrap();
    a.jnz(retry).unwrap();
    a.dec(r8).unwrap();
    a.jnz(retry).unwrap();
    a.hlt().unwrap();
    a.assemble(0x1000).unwrap()
}

#[test]
fn cmpxchg16b_mixed_tier_contention() {
    use std::sync::{Arc, Barrier};
    use x86jit_core::{CachedBlock, MemConsistency, StepResult};
    // SDM Vol 3A §11.1.2, Bus Locking: locked instructions
    // serialize the full update. Carry into the high half must not be lost.
    for consistency in [
        MemConsistency::Fast,
        MemConsistency::AcqRel,
        MemConsistency::FullTso,
    ] {
        const ITERS: u64 = 4000;
        let mut vm = vm(Box::new(JitBackend::new()), true, true);
        vm.consistency = consistency;
        vm.write_bytes(0x1000, &counter_code(ITERS)).unwrap();
        let initial = (3u128 << 64) | (u64::MAX - 100) as u128;
        vm.write_bytes(0x2000, &initial.to_le_bytes()).unwrap();
        let vm = Arc::new(vm);
        let barrier = Arc::new(Barrier::new(2));
        let handles: Vec<_> = [false, true]
            .into_iter()
            .map(|interpreted| {
                let vm = Arc::clone(&vm);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let mut cpu = vm.new_vcpu();
                    cpu.set_reg(Reg::Rip, 0x1000);
                    barrier.wait();
                    if interpreted {
                        loop {
                            match cpu.step_instruction(&vm) {
                                StepResult::Continue => {}
                                StepResult::Exit(Exit::Hlt) => break,
                                StepResult::Exit(other) => panic!("interpreted counter: {other:?}"),
                            }
                        }
                    } else {
                        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        let mut got = [0u8; 16];
        vm.read_bytes(0x2000, &mut got).unwrap();
        assert_eq!(u128::from_le_bytes(got), initial + 2 * ITERS as u128);
        assert!(matches!(
            vm.cache.get(x86jit_core::BlockKey::new(
                0x1000,
                x86jit_core::CpuMode::Long64
            )),
            Some(CachedBlock::Compiled { .. })
        ));
    }
}

#[test]
fn cmpxchg_pair_host_backing_alignment() {
    use x86jit_core::{HostRam, HostRamError, Memory, MemoryModel};
    // mmap gives a page-aligned allocation, using a low hint so an identity mapping
    // fits the engine's tracked address span on both test hosts.
    for kind in [0, 1, 2] {
        // SAFETY: anonymous writable mapping; the HostRam destructor unmaps it once.
        let ptr = unsafe {
            libc::mmap(
                0x2_0000_0000usize as *mut _,
                0x2000,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED);
        let guest_base = if kind == 1 { ptr as u64 } else { 0 };
        let mapping = ptr as usize;
        let ram = HostRam {
            ptr: if kind == 2 {
                ptr.cast::<u8>().wrapping_add(1)
            } else {
                ptr.cast()
            },
            len: 0x1000,
            guest_base,
            dtor: Box::new(move |_, _| {
                // SAFETY: this destructor owns the mapping above.
                assert_eq!(unsafe { libc::munmap(mapping as *mut _, 0x2000) }, 0);
            }),
            protect: None,
        };
        let result = Memory::from_host_ram(
            MemoryModel::Reserved {
                span: guest_base + 0x1000,
            },
            ram,
        );
        if kind == 2 {
            assert!(matches!(result, Err(HostRamError::Misaligned)));
        } else {
            let mut mem = result.unwrap();
            mem.map(guest_base, 0x1000, Prot::RW, RegionKind::Ram)
                .unwrap();
            assert_eq!(mem.atomic_cas128(guest_base, 0, 7).unwrap(), 0);
            assert_eq!(mem.read(guest_base, 8).unwrap(), 7);
        }
    }
}

#[cfg(any(feature = "unicorn", all(target_arch = "x86_64", target_os = "linux")))]
#[test]
fn cmpxchg_pair_matches_oracles() {
    use x86jit_tests::oracle::{run_with_backend, VectorInput};
    use x86jit_tests::vector::{CpuSnapshot, MemChunk, MemKind, RunSpec, SnapFlags};
    for wide in [false, true] {
        for locked in [false, true] {
            for case in 0..3 {
                let mut code = Vec::new();
                if locked {
                    code.push(0xf0);
                }
                if wide {
                    code.push(0x48);
                }
                code.extend_from_slice(&[0x0f, 0xc7, 0x0f, 0xf4]);
                let mut cpu = CpuSnapshot::default();
                let upper = if wide { 0 } else { 0xabcd_0000_0000_0000 };
                cpu.gpr[0] = upper | if case == 1 { 6 } else { 5 };
                cpu.gpr[2] = upper | if case == 2 { 8 } else { 7 };
                cpu.gpr[3] = 9;
                cpu.gpr[1] = 11;
                cpu.gpr[7] = 0x22_0000 + if wide { 0 } else { 1 };
                cpu.flags = SnapFlags::from_rflags(0x897);
                let old: u128 = if wide { (7 << 64) | 5 } else { (7 << 32) | 5 };
                let mut data = vec![0; 0x1000];
                let offset = if wide { 0 } else { 1 };
                let size = if wide { 16 } else { 8 };
                data[offset..offset + size].copy_from_slice(&old.to_le_bytes()[..size]);
                let input = VectorInput {
                    cpu_init: cpu,
                    mem_init: vec![
                        MemChunk {
                            addr: 0x21_0000,
                            bytes: code,
                            kind: MemKind::Ram,
                        },
                        MemChunk {
                            addr: 0x22_0000,
                            bytes: data,
                            kind: MemKind::Ram,
                        },
                    ],
                    entry: 0x21_0000,
                    run: RunSpec::UntilExit,
                };
                for backend in backends() {
                    let got = run_with_backend(&input, backend);
                    #[cfg(feature = "unicorn")]
                    {
                        use x86jit_tests::oracle::Oracle;
                        let witness = x86jit_tests::unicorn::UnicornOracle.run(&input);
                        assert_eq!(x86jit_tests::compare::compare(&witness, &got, &[]), None);
                    }
                    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                    {
                        let witness = x86jit_tests::native::run_native(&input)
                            .expect("native paired CAS witness");
                        assert_eq!(x86jit_tests::compare::compare(&witness, &got, &[]), None);
                    }
                }
            }
        }
    }
}

/// Targeted timing experiment; run explicitly with --release --ignored --nocapture.
/// Compares the complete guest helper loop against a direct inline host loop.
/// The difference includes guest dispatch, bounds checks and register synchronization.
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore]
fn cmpxchg16b_path_cost() {
    const ITERS: u64 = 1_000_000;
    #[repr(align(16))]
    struct Pair([u64; 2]);
    let mut pair = Pair([0, 0]);
    let ptr = pair.0.as_mut_ptr();
    let start = std::time::Instant::now();
    // Same LDAXP/STLXP sequence as the memory primitive, within one host loop.
    // SAFETY: ptr names the private aligned pair for this entire asm block.
    unsafe {
        core::arch::asm!(
            "mov {expected}, #0",
            "2:",
            "add {desired}, {expected}, #1",
            "dmb ish",
            "3:",
            "ldaxp {lo}, {hi}, [{ptr}]",
            "cmp {lo}, {expected}",
            "ccmp {hi}, #0, #0, eq",
            "csel {newlo}, {desired}, {lo}, eq",
            "csel {newhi}, xzr, {hi}, eq",
            "stlxp {status:w}, {newlo}, {newhi}, [{ptr}]",
            "cbnz {status:w}, 3b",
            "dmb ish",
            "mov {expected}, {desired}",
            "subs {remaining}, {remaining}, #1",
            "b.ne 2b",
            ptr = in(reg) ptr, remaining = inout(reg) ITERS => _,
            expected = out(reg) _, desired = out(reg) _, lo = out(reg) _, hi = out(reg) _,
            newlo = out(reg) _, newhi = out(reg) _, status = out(reg) _,
            options(nostack),
        )
    };
    let inline_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;
    assert_eq!(pair.0, [ITERS, 0]);

    let vm = vm(Box::new(JitBackend::new()), true, true);
    vm.write_bytes(0x1000, &counter_code(ITERS)).unwrap();
    // Compile and warm the blocks before timing.
    let run = || {
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    };
    run();
    vm.write_bytes(0x2000, &[0; 16]).unwrap();
    let start = std::time::Instant::now();
    run();
    let helper_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;
    let mut got = [0u8; 16];
    vm.read_bytes(0x2000, &mut got).unwrap();
    assert_eq!(u128::from_le_bytes(got), ITERS as u128);
    eprintln!("MEASURED cmpxchg16b: {ITERS} increments; inline host {inline_ns:.2} ns/op; JIT helper guest loop {helper_ns:.2} ns/op; ratio {:.2}", helper_ns / inline_ns);
}

#[test]
fn cmpxchg_pair_owned_backing_preserves_alignment() {
    use x86jit_core::{Memory, MemoryModel};
    for len in [0, 1, 8, 4097] {
        let mem = Memory::new(MemoryModel::Flat { size: len });
        assert_eq!(mem.host_base() as usize & 15, 0);
        assert_eq!(mem.deep_copy().unwrap().host_base() as usize & 15, 0);
    }
}

const LOOP_ITERS: u64 = 32;

fn cas_loop(body: &[u8]) -> Vec<u8> {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    let mut latch = a.create_label();
    a.jmp(top).unwrap();
    a.set_label(&mut top).unwrap();
    a.db(body).unwrap();
    a.jmp(latch).unwrap();
    a.set_label(&mut latch).unwrap();
    a.dec(r8).unwrap();
    a.jnz(top).unwrap();
    a.hlt().unwrap();
    a.assemble(0x1000).unwrap()
}

fn run_cas_loop(vm: &Vm, cpu: &mut x86jit_core::Vcpu, region: bool) {
    cpu.set_reg(Reg::R8, LOOP_ITERS);
    // Entry + 32 bodies + 31 latches. Stop before the final latch so the
    // returned flags are the CAS flags, before DEC overwrites them.
    assert!(matches!(
        cpu.run(vm, Some(2 * LOOP_ITERS)),
        Exit::BudgetExhausted
    ));
    assert_eq!(cpu.reg(Reg::R8), 1);
    if region {
        assert!(
            vm.cache.regions() > 0,
            "the CAS loop must compile as a region"
        );
    }
}

#[test]
fn cmpxchg_pair_semantics_superblock() {
    use iced_x86::code_asm::*;
    for wide in [false, true] {
        for locked in [false, true] {
            for case in 0..3 {
                for backend in backends() {
                    let region = backend.region_caps().is_some();
                    let vm = vm(backend, wide, locked);
                    let upper = if wide { 0 } else { 0xdead_beef_0000_0000 };
                    let mut a = CodeAssembler::new(64).unwrap();
                    // Reset the operand and accumulator so every iteration
                    // exercises the same success or low/high mismatch.
                    a.mov(rax, 5u64).unwrap();
                    a.mov(qword_ptr(rdi), rax).unwrap();
                    a.mov(rax, 7u64).unwrap();
                    if wide {
                        a.mov(qword_ptr(rdi + 8), rax).unwrap();
                    } else {
                        a.mov(dword_ptr(rdi + 4), eax).unwrap();
                    }
                    a.mov(rax, upper | if case == 1 { 6u64 } else { 5 })
                        .unwrap();
                    a.mov(rdx, upper | if case == 2 { 8u64 } else { 7 })
                        .unwrap();
                    a.mov(rbx, 9u64).unwrap();
                    a.mov(rcx, 11u64).unwrap();
                    let mut body = a.assemble(0x1000).unwrap();
                    if locked {
                        body.push(0xf0);
                    }
                    if wide {
                        body.push(0x48);
                    }
                    body.extend_from_slice(&[0x0f, 0xc7, 0x0f]);
                    vm.write_bytes(0x1000, &cas_loop(&body)).unwrap();
                    let mut cpu = vm.new_vcpu();
                    cpu.set_reg(Reg::Rip, 0x1000);
                    cpu.set_reg(Reg::Rdi, 0x2000);
                    cpu.cpu.flags.cf = true;
                    cpu.cpu.flags.of = true;
                    cpu.cpu.flags.pf_src = 0;
                    cpu.cpu.flags.af_src = 0x10;
                    cpu.cpu.flags.sf = true;
                    // DEC in earlier latches changes these flags. Capture the
                    // final iteration's input immediately before its CAS.
                    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
                    // Start the counter after the entry block has run.
                    cpu.set_reg(Reg::R8, LOOP_ITERS);
                    assert!(matches!(
                        cpu.run(&vm, Some(2 * (LOOP_ITERS - 1))),
                        Exit::BudgetExhausted
                    ));
                    let mut expected_flags = cpu.cpu.flags;
                    expected_flags.zf = case == 0;
                    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
                    assert_eq!(cpu.cpu.flags, expected_flags);
                    assert_eq!(cpu.reg(Reg::Rax), if case == 0 { upper | 5 } else { 5 });
                    assert_eq!(cpu.reg(Reg::Rdx), if case == 0 { upper | 7 } else { 7 });
                    let mut got = [0; 16];
                    let size = if wide { 16 } else { 8 };
                    vm.read_bytes(0x2000, &mut got[..size]).unwrap();
                    let bits = if wide { 64 } else { 32 };
                    let expected: u128 = if case == 0 {
                        (11 << bits) | 9
                    } else {
                        (7 << bits) | 5
                    };
                    assert_eq!(u128::from_le_bytes(got), expected);
                    if region {
                        assert!(
                            vm.cache.regions() > 0,
                            "the paired CAS loop must compile as a region"
                        );
                    }
                }
            }
        }
    }
}

fn cmpxchg_pair_mmio_is_unsupported(backend: fn() -> Box<dyn Backend>, wide: bool) {
    // Engine policy: device transfers cannot express an atomic paired CAS.
    use iced_x86::code_asm::*;
    for locked in [false, true] {
        let backend = backend();
        let region = backend.region_caps().is_some();
        let mut vm = vm(backend, wide, locked);
        vm.mem.unmap(0, 0x4000).unwrap();
        vm.map(0x1000, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
        vm.map(0x2000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
        vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.write_bytes(0x2000, &[0x55; 16]).unwrap();
        vm.write_bytes(0x3000, &[0x55; 16]).unwrap();
        let mut cas = Vec::new();
        if locked {
            cas.push(0xf0);
        }
        if wide {
            cas.push(0x48);
        }
        cas.extend_from_slice(&[0x0f, 0xc7, 0x0f]);
        let mut a = CodeAssembler::new(64).unwrap();
        let mut top = a.create_label();
        let mut latch = a.create_label();
        a.jmp(top).unwrap();
        a.set_label(&mut top).unwrap();
        a.add(r9, 1).unwrap();
        a.mov(rax, 17u64).unwrap();
        a.mov(rdx, 19u64).unwrap();
        a.mov(rbx, 23u64).unwrap();
        a.mov(rcx, 29u64).unwrap();
        // SDM Vol 2A, CMP, Operation and Flags Affected: 0 - 1.
        a.cmp(r10, 1).unwrap();
        a.db(&cas).unwrap();
        a.jmp(latch).unwrap();
        a.set_label(&mut latch).unwrap();
        a.dec(r8).unwrap();
        a.jnz(top).unwrap();
        // First execute the CAS repeatedly on RAM, then reach the same
        // instruction on MMIO through an internal region edge.
        a.mov(rdi, 0x3000u64).unwrap();
        a.mov(r8, 1u64).unwrap();
        a.jmp(top).unwrap();
        let code = a.assemble(0x1000).unwrap();
        let cas_offset = code.windows(cas.len()).position(|b| b == cas).unwrap();
        let cas_rip = 0x1000 + cas_offset as u64;
        vm.write_bytes(0x1000, &code).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.set_reg(Reg::Rdi, 0x2000);
        cpu.set_reg(Reg::R8, LOOP_ITERS);
        let before = cpu.cpu.clone();
        let mut expected_gpr = before.gpr;
        expected_gpr[0] = 17;
        expected_gpr[1] = 29;
        expected_gpr[2] = 19;
        expected_gpr[3] = 23;
        expected_gpr[7] = 0x3000;
        expected_gpr[8] = 1;
        expected_gpr[9] = LOOP_ITERS + 1;
        let len = cas.len();
        let mut expected_bytes = [0; 15];
        expected_bytes[..len].copy_from_slice(&cas);
        // Exact matching also rejects any MmioRead or MmioWrite exit.
        for _ in 0..2 {
            let exit = cpu.run(&vm, Some(4 * LOOP_ITERS));
            if region {
                assert!(
                    vm.cache.regions() > 0,
                    "the MMIO CAS must be inside a compiled region"
                );
            }
            assert!(
                matches!(exit, Exit::UnknownInstruction { addr, bytes, len: n, .. }
                    if addr == cas_rip && n as usize == len && bytes == expected_bytes),
                "{exit:?}"
            );
            assert_eq!(cpu.cpu.gpr, expected_gpr);
            assert_eq!(cpu.cpu.flags.to_rflags() & 0x8d5, 0x95);
            assert_eq!(cpu.cpu.rip, cas_rip);
            assert_eq!(cpu.cpu.xmm, before.xmm);
            assert_eq!(cpu.cpu.ymm_hi, before.ymm_hi);
            assert_eq!(cpu.cpu.zmm_hi, before.zmm_hi);
            assert_eq!(cpu.cpu.pending_mmio, before.pending_mmio);
            assert_eq!(cpu.cpu.pending_mmio_write, before.pending_mmio_write);
            assert_eq!(cpu.cpu.mmio_parts_len, before.mmio_parts_len);
            assert_eq!(cpu.cpu.mmio_retry_active, before.mmio_retry_active);
            let mut got = [0; 16];
            vm.read_bytes(0x3000, &mut got).unwrap();
            assert_eq!(got, [0x55; 16]);
            vm.read_bytes(0x2000, &mut got).unwrap();
            assert_eq!(got, [0x55; 16]);
            assert!(vm.take_dirty_ranges().is_empty());
        }
    }
}

#[test]
fn cmpxchg8b_mmio_interp() {
    cmpxchg_pair_mmio_is_unsupported(|| Box::new(InterpreterBackend), false);
}

#[test]
fn cmpxchg8b_mmio_jit() {
    cmpxchg_pair_mmio_is_unsupported(|| Box::new(JitBackend::new()), false);
}

#[test]
fn cmpxchg8b_mmio_superblock() {
    cmpxchg_pair_mmio_is_unsupported(|| backends().into_iter().nth(2).unwrap(), false);
}

#[test]
fn cmpxchg_pair_preserves_alu_flags_at_a_later_fault() {
    use iced_x86::code_asm::*;
    // SDM Vol 2A CMPXCHG8B/CMPXCHG16B, Flags Affected: only ZF changes.
    // A later flag writer makes the preceding CMP dead unless the helper is
    // modeled as a flag barrier. A fault between them must preserve CAS's ZF.
    for wide in [false, true] {
        for backend in backends() {
            let mut vm = vm(backend, wide, true);
            vm.mem.unmap(0, 0x4000).unwrap();
            vm.map(0x1000, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
            vm.map(0x2000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
            let mut a = CodeAssembler::new(64).unwrap();
            a.cmp(r9, 1).unwrap(); // 0 - 1: CF/PF/AF/SF=1, OF/ZF=0
            if wide {
                a.lock().cmpxchg16b(xmmword_ptr(rdi)).unwrap();
            } else {
                a.lock().cmpxchg8b(qword_ptr(rdi)).unwrap();
            }
            let fault = 0x1000 + a.assemble(0x1000).unwrap().len() as u64;
            a.mov(r8, qword_ptr(0x4000u64)).unwrap();
            a.cmp(r8, r8).unwrap();
            a.hlt().unwrap();
            vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
                .unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rdi, 0x2000);
            assert!(matches!(
                cpu.run(&vm, None),
                Exit::UnmappedMemory { addr: 0x4000, .. }
            ));
            assert_eq!(cpu.reg(Reg::Rip), fault);
            assert_eq!(cpu.cpu.flags.to_rflags() & 0x8d5, 0xd5);
        }
    }
}

// SDM Vol 2A, CMPXCHG, Operation: the accumulator is written only on failure;
// DEST is written on either outcome. SDM Vol 1 §3.4.1.1 specifies partial GPR
// writes: 32-bit writes zero-extend, while 8/16-bit writes preserve upper bits.
fn cmpxchg_accumulator(backend: fn() -> Box<dyn Backend>, memory: bool) {
    for size in [4, 1, 2, 8] {
        for equal in [true, false] {
            let backend = backend();
            let region = backend.region_caps().is_some();
            let vm = vm(backend, false, false);
            let mut code = Vec::new();
            match size {
                2 => code.push(0x66),
                8 => code.push(0x48),
                _ => {}
            }
            // cmpxchg [rdi]/rbx, rcx (CL/CX/ECX/RCX)
            code.extend_from_slice(&[
                0x0f,
                if size == 1 { 0xb0 } else { 0xb1 },
                if memory { 0x0f } else { 0xcb },
            ]);
            let old: u64 = 0x1234_5678_9abc_7d35;
            let src: u64 = 0xfedc_ba98_7654_3210;
            let mask = u64::MAX >> (64 - size * 8);
            let acc = (0xabcd_ef01_2345_6789 & !mask) | ((old ^ u64::from(!equal)) & mask);
            use iced_x86::code_asm::*;
            let mut a = CodeAssembler::new(64).unwrap();
            a.mov(rax, acc).unwrap();
            a.mov(rbx, old).unwrap();
            if memory {
                a.mov(qword_ptr(rdi), rbx).unwrap();
            }
            let mut body = a.assemble(0x1000).unwrap();
            body.extend_from_slice(&code);
            vm.write_bytes(0x1000, &cas_loop(&body)).unwrap();
            vm.write_bytes(0x2000, &old.to_le_bytes()).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rdi, 0x2000);
            cpu.set_reg(Reg::Rax, acc);
            cpu.set_reg(Reg::Rbx, old);
            cpu.set_reg(Reg::Rcx, src);
            let mut expected_gpr = cpu.cpu.gpr;
            expected_gpr[8] = 1;
            if !equal {
                expected_gpr[0] = if size < 4 {
                    (acc & !mask) | (old & mask)
                } else {
                    old & mask
                };
            }
            let result = if equal { src & mask } else { old & mask };
            if !memory {
                expected_gpr[3] = if size < 4 {
                    (old & !mask) | result
                } else {
                    result
                };
            }
            run_cas_loop(&vm, &mut cpu, region);
            assert_eq!(cpu.cpu.gpr, expected_gpr, "size={size} equal={equal}");
            assert_eq!(cpu.cpu.flags.zf, equal);
            let mut got = [0; 8];
            vm.read_bytes(0x2000, &mut got).unwrap();
            let expected_mem = if memory { (old & !mask) | result } else { old };
            assert_eq!(u64::from_le_bytes(got), expected_mem);
        }
    }
}

#[test]
fn cmpxchg_accumulator_memory_interp() {
    cmpxchg_accumulator(|| Box::new(InterpreterBackend), true);
}
#[test]
fn cmpxchg_accumulator_memory_jit() {
    cmpxchg_accumulator(|| Box::new(JitBackend::new()), true);
}
#[test]
fn cmpxchg_accumulator_memory_superblock() {
    cmpxchg_accumulator(|| backends().into_iter().nth(2).unwrap(), true);
}
#[test]
fn cmpxchg_accumulator_register_interp() {
    cmpxchg_accumulator(|| Box::new(InterpreterBackend), false);
}
#[test]
fn cmpxchg_accumulator_register_jit() {
    cmpxchg_accumulator(|| Box::new(JitBackend::new()), false);
}
#[test]
fn cmpxchg_accumulator_register_superblock() {
    cmpxchg_accumulator(|| backends().into_iter().nth(2).unwrap(), false);
}

#[test]
fn cmpxchg_matches_interp() {
    use x86jit_tests::oracle::{run_with_backend, VectorInput};
    use x86jit_tests::vector::{CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec};
    // SDM Vol 2A, CMPXCHG, Operation: DEST is written on both outcomes,
    // including when it aliases AL/AX/EAX/RAX or AH.
    for size in [1, 2, 4, 8] {
        for form in [0x0f, 0xcb, 0xc8, 0xcc] {
            if form == 0xcc && size != 1 {
                continue; // AH exists only in the byte form without REX.
            }
            for equal in [false, true] {
                let mut code = Vec::new();
                match size {
                    2 => code.push(0x66),
                    8 => code.push(0x48),
                    _ => {}
                }
                code.extend_from_slice(&[0x0f, if size == 1 { 0xb0 } else { 0xb1 }, form, 0xf4]);
                let mut cpu = CpuSnapshot::default();
                cpu.gpr[0] = 0xdead_beef_1234_3535 ^ u64::from(!equal);
                cpu.gpr[3] = 0xabcd_ef01_1234_3535;
                cpu.gpr[1] = 0x8765_4321_fedc_ba98;
                cpu.gpr[7] = 0x2000;
                if size == 8 {
                    cpu.gpr[3] = 0xdead_beef_1234_3535;
                }
                let mut data = vec![0; 0x1000];
                data[..8].copy_from_slice(&cpu.gpr[3].to_le_bytes());
                let input = VectorInput {
                    cpu_init: cpu,
                    mem_init: vec![
                        MemChunk {
                            addr: 0x1000,
                            bytes: code,
                            kind: MemKind::Ram,
                        },
                        MemChunk {
                            addr: 0x2000,
                            bytes: data,
                            kind: MemKind::Ram,
                        },
                    ],
                    entry: 0x1000,
                    run: RunSpec::UntilExit,
                };
                let interp = run_with_backend(&input, Box::new(InterpreterBackend));
                assert!(matches!(interp.exit, ExitKind::Hlt));
                // Accumulator aliases always compare equal; DEST then takes SRC.
                if form == 0xc8 {
                    let mask = u64::MAX >> (64 - size * 8);
                    let expected = if size < 4 {
                        (input.cpu_init.gpr[0] & !mask) | (input.cpu_init.gpr[1] & mask)
                    } else {
                        input.cpu_init.gpr[1] & mask
                    };
                    assert_eq!(interp.cpu.gpr[0], expected);
                }
                if form == 0xcc {
                    let initial = input.cpu_init.gpr[0];
                    let expected = if equal {
                        (initial & !0xff00) | ((input.cpu_init.gpr[1] & 0xff) << 8)
                    } else {
                        (initial & !0xff) | ((initial >> 8) & 0xff)
                    };
                    assert_eq!(interp.cpu.gpr[0], expected);
                }
                for backend in backends().into_iter().skip(1) {
                    let jit = run_with_backend(&input, backend);
                    assert_eq!(
                        x86jit_tests::compare::compare(&interp, &jit, &[]),
                        None,
                        "size={size} form={form:x} equal={equal}"
                    );
                }
            }
        }
    }
}

#[test]
fn cmpxchg16b_mmio_interp() {
    cmpxchg_pair_mmio_is_unsupported(|| Box::new(InterpreterBackend), true);
}
#[test]
fn cmpxchg16b_mmio_jit() {
    cmpxchg_pair_mmio_is_unsupported(|| Box::new(JitBackend::new()), true);
}
#[test]
fn cmpxchg16b_mmio_superblock() {
    cmpxchg_pair_mmio_is_unsupported(|| backends().into_iter().nth(2).unwrap(), true);
}
