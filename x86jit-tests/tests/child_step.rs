//! Child execution compared with both normal tiers and real host protections.

#[path = "../../x86jit-core/tests/child.rs"]
mod core_tests;

use x86jit_core::{
    child, Backend, ChildArena, ChildStep, CpuMode, Exit, InterpreterBackend, Prot, RegionKind, Vm,
    VmConfig,
};
#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
use x86jit_core::{HostFaultPlatform, RawHostFault};
use x86jit_cranelift::JitBackend;
#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
#[path = "../../x86jit-cranelift/src/unwind_test_fault_context.rs"]
mod fault_context;

#[test]
fn child_matches_interpreter_and_jit() {
    for jit in [false, true] {
        for &code in &[
            &[0x48, 0x83, 0xc0, 1][..],
            &[0x48, 0x11, 0xd8],
            &[0x48, 0x19, 0xd8],
            &[0x48, 0xf7, 0xe3],
            &[0x48, 0xf7, 0xf3],
            &[0x48, 0x8b, 0x00],
            &[0x48, 0x89, 0x18],
            &[0x48, 0x83, 0x00, 1],
            &[0x0f, 0x10, 0x00],
            &[0x0f, 0x11, 0x00],
            &[0x0f, 0x58, 0xc1],
            &[0x66, 0x0f, 0xef, 0xc1],
            &[0x66, 0x0f, 0x3a, 0xcf, 0xc1, 0],
            &[0x66, 0x0f, 0x3a, 0xcf, 0x00, 0],
            &[0xc5, 0xfc, 0x10, 0x00],
            &[0xc5, 0xfc, 0x11, 0x00],
            &[0xc5, 0xfc, 0x58, 0xc1],
            &[0xf3, 0xa4],
            &[0xf3, 0xaa],
            &[0xf3, 0x48, 0xab],
        ] {
            let backend: Box<dyn Backend> = if jit {
                Box::new(JitBackend::new())
            } else {
                Box::new(InterpreterBackend)
            };
            let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
            vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x3000, 0x3000, Prot::RW, RegionKind::Ram).unwrap();
            let mut text = code.to_vec();
            text.push(0xf4);
            vm.write_bytes(0x1000, &text).unwrap();
            let data = [0x12; 0x3000];
            vm.write_bytes(0x3000, &data).unwrap();
            let mut normal = vm.new_vcpu();
            normal.cpu = core_tests::initial_cpu();
            let mut cpu = normal.cpu.clone();
            assert!(matches!(normal.run(&vm, Some(100)), Exit::Hlt));
            let mut expected_data = [0; 0x3000];
            vm.read_bytes(0x3000, &mut expected_data).unwrap();
            vm.write_bytes(0x3000, &data).unwrap();
            let view = vm.child_view(CpuMode::Long64);
            let mut arena = ChildArena::new(16);
            assert!(matches!(
                core_tests::no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
                ChildStep::Continue
            ));
            assert!(matches!(
                core_tests::no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
                ChildStep::Hlt
            ));
            assert_eq!(
                format!("{:?}", normal.cpu),
                format!("{cpu:?}"),
                "jit={jit} {code:x?}"
            );
            let mut actual_data = [0; 0x3000];
            vm.read_bytes(0x3000, &mut actual_data).unwrap();
            assert_eq!(expected_data, actual_data, "jit={jit} {code:x?}");
        }
    }
}

#[test]
fn child_parent_invalidation_drops_later_translations_during_execution() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(JitBackend::new()));
    vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x3000, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0x89, 0x18]).unwrap();
    vm.write_bytes(0x3000, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
    let view = vm.child_view(CpuMode::Long64);
    // Compile after the snapshot so the written page is initially allowed.
    let mut normal = vm.new_vcpu();
    normal.cpu.rip = 0x3000;
    assert!(matches!(normal.run(&vm, Some(100)), Exit::Hlt));
    assert_eq!(normal.cpu.gpr[0], 1);
    let running = Arc::new(AtomicBool::new(true));
    std::thread::scope(|scope| {
        let child_running = running.clone();
        let vm = &vm;
        scope.spawn(move || {
            let mut cpu = vm.new_vcpu();
            while child_running.load(Ordering::Acquire) {
                cpu.cpu.rip = 0x3000;
                assert!(matches!(cpu.run(vm, Some(100)), Exit::Hlt));
                assert!([1, 2].contains(&cpu.cpu.gpr[0]));
            }
        });
        let mut cpu = core_tests::initial_cpu();
        cpu.gpr[0] = 0x3001;
        cpu.gpr[3] = 2;
        let mut arena = ChildArena::new(2);
        assert!(matches!(
            core_tests::no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
            ChildStep::Continue
        ));
        vm.invalidate_written_pages(&arena);
        normal.cpu.rip = 0x3000;
        assert!(matches!(normal.run(vm, Some(100)), Exit::Hlt));
        assert_eq!(normal.cpu.gpr[0], 2);
        running.store(false, Ordering::Release);
    });
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod protected {
    use super::*;
    use x86jit_core::jit_abi::fault::HostRegisters;
    use x86jit_core::{FaultOrigin, HostFaultPlatform, Knowledge, RawHostFault};

    use super::fault_context;

    extern "C" fn handler(signal: i32, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
        // SAFETY: SA_SIGINFO provides a live host context on the accessing thread.
        unsafe {
            if (*info).si_code <= 0 {
                libc::_exit(91);
            }
            let raw = fault_context::extract(signal, &*info, context);
            let context = &mut *context.cast::<libc::ucontext_t>();
            let mut regs = HostRegisters::default();
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                let m = &*context.uc_mcontext;
                // Arm ARM DDI0487I.a §D17.2.37: only data aborts from memory
                // access translation/access/permission failures enter this fixup.
                let esr = m.__es.__esr;
                if ![0x24, 0x25].contains(&(esr >> 26)) || !(4..=15).contains(&(esr & 63)) {
                    libc::_exit(92);
                }
                regs.pc = m.__ss.__pc;
                regs.scratch = m.__ss.__x[9];
            }
            #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
            {
                regs.pc = (*context.uc_mcontext).__ss.__rip;
                regs.scratch = (*context.uc_mcontext).__ss.__r11;
            }
            #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
            {
                regs.pc = context.uc_mcontext.pc;
                regs.scratch = context.uc_mcontext.regs[9];
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            {
                regs.pc = context.uc_mcontext.gregs[libc::REG_RIP as usize] as u64;
                regs.scratch = context.uc_mcontext.gregs[libc::REG_R11 as usize] as u64;
            }
            if !child::classify(regs.pc as usize, &mut regs, raw) {
                libc::_exit(93);
            }
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                (*context.uc_mcontext).__ss.__pc = regs.pc;
            }
            #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
            {
                (*context.uc_mcontext).__ss.__rip = regs.pc;
            }
            #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
            {
                context.uc_mcontext.pc = regs.pc;
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            {
                context.uc_mcontext.gregs[libc::REG_RIP as usize] = regs.pc as libc::greg_t;
            }
        }
    }
    #[test]
    fn child_identity_mapped_host_faults_recover_without_allocation() {
        // Deliberate host faults and signal dispositions stay in an isolated process.
        // SAFETY: parent waits for its child; child exits without engine destructors.
        unsafe {
            let pid = libc::fork();
            assert!(pid >= 0);
            if pid == 0 {
                libc::alarm(15);
                let result = std::panic::catch_unwind(|| {
                    let mut action: libc::sigaction = std::mem::zeroed();
                    action.sa_sigaction = handler as *const () as usize;
                    action.sa_flags = libc::SA_SIGINFO;
                    libc::sigemptyset(&mut action.sa_mask);
                    assert_eq!(
                        libc::sigaction(libc::SIGSEGV, &action, std::ptr::null_mut()),
                        0
                    );
                    assert_eq!(
                        libc::sigaction(libc::SIGBUS, &action, std::ptr::null_mut()),
                        0
                    );
                    let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
                    let len = 4 * page;
                    let ptr = libc::mmap(
                        std::ptr::null_mut(),
                        len,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANON,
                        -1,
                        0,
                    );
                    assert_ne!(ptr, libc::MAP_FAILED);
                    let base = ptr as u64;
                    let ram = x86jit_core::HostRam {
                        ptr: ptr.cast(),
                        len,
                        guest_base: base,
                        protect: None,
                        dtor: Box::new(|p, n| {
                            libc::munmap(p.cast(), n);
                        }),
                    };
                    let mut vm = Vm::with_backend_host_ram(
                        VmConfig::flat(base + len as u64),
                        Box::new(InterpreterBackend),
                        ram,
                    )
                    .unwrap();
                    vm.map(base, len, Prot::RWX, RegionKind::Ram).unwrap();
                    vm.mem.set_host_fault_recovery(true);
                    let view = vm.child_view(CpuMode::Long64);
                    let mut arena = ChildArena::new(16);
                    let hole = base + 2 * page as u64;
                    // SDM Vol 2A §3.2, CMPXCHG and CMPXCHG8B/CMPXCHG16B:
                    // write permission is required even when comparison fails.
                    #[cfg(target_arch = "aarch64")]
                    let paths: &[bool] = if std::arch::is_aarch64_feature_detected!("lse") {
                        &[false, true]
                    } else {
                        &[true]
                    };
                    #[cfg(not(target_arch = "aarch64"))]
                    let paths = &[false];
                    for &exclusive in paths {
                        for (code, size) in [
                            (&[0xf0, 0x00, 0x18][..], 1),
                            (&[0xf0, 0x66, 0x01, 0x18], 2),
                            (&[0xf0, 0x01, 0x18], 4),
                            (&[0xf0, 0x48, 0x0f, 0xc1, 0x18], 8),
                            (&[0xf0, 0x48, 0x0f, 0xb1, 0x18], 8),
                            (&[0xf0, 0x0f, 0xc7, 0x08], 8),
                            (&[0xf0, 0x48, 0x0f, 0xc7, 0x08], 16),
                        ] {
                            for protection in [libc::PROT_NONE, libc::PROT_READ, -1] {
                                let mut arena = ChildArena::new(1);
                                #[cfg(target_arch = "aarch64")]
                                if exclusive {
                                    arena.disable_lse_atomics();
                                }
                                #[cfg(not(target_arch = "aarch64"))]
                                let _ = exclusive;
                                std::ptr::copy_nonoverlapping(
                                    code.as_ptr(),
                                    ptr.cast::<u8>(),
                                    code.len(),
                                );
                                std::ptr::write_bytes(hole as *mut u8, 0xa5, 16);
                                if protection == -1 {
                                    assert_eq!(libc::munmap(hole as *mut _, page), 0);
                                } else {
                                    assert_eq!(libc::mprotect(hole as *mut _, page, protection), 0);
                                }
                                let mut cpu = core_tests::initial_cpu();
                                cpu.rip = base;
                                cpu.gpr[0] = hole;
                                let before = format!("{cpu:?}");
                                let ChildStep::Fault(f) = core_tests::no_alloc(|| {
                                    child::step(&view, &mut cpu, &mut arena)
                                }) else {
                                    panic!("{code:x?}");
                                };
                                assert!(
                                    arena.written_pages().is_empty(),
                                    "{code:x?} exclusive={exclusive} protection={protection}"
                                );
                                assert_eq!(f.origin, FaultOrigin::HostAccess);
                                assert!(f.raw_host_fault.is_some());
                                assert_eq!(
                                    f.address,
                                    Knowledge::Known(x86jit_core::FaultAddress::new(
                                        hole,
                                        x86jit_core::AddressPrecision::ExactByte,
                                        x86jit_core::AddressSpace::Linear
                                    ))
                                );
                                assert_eq!(f.operand_base, Knowledge::Known(hole));
                                assert_eq!(f.operand_size, Knowledge::Known(size));
                                assert_eq!(f.transfer_size, Knowledge::Known(size));
                                assert_eq!(
                                    f.access,
                                    Knowledge::Known(x86jit_core::FaultAccess::ReadWrite)
                                );
                                assert_eq!(
                                    f.transfer_access,
                                    Knowledge::Known(x86jit_core::FaultAccess::Write)
                                );
                                assert_eq!(f.ram, x86jit_core::MemoryEffects::None);
                                assert_eq!(before, format!("{cpu:?}"));
                                cpu.gpr[0] = hole + page as u64;
                                assert!(matches!(
                                    core_tests::no_alloc(|| child::step(
                                        &view, &mut cpu, &mut arena
                                    )),
                                    ChildStep::Continue
                                ));
                                assert_eq!(&*arena.written_pages(), &[(hole + page as u64) >> 12]);
                                // Fault again on an already logged page; preserve its earlier entry.
                                cpu.rip = base;
                                cpu.gpr[0] = hole + page as u64;
                                assert_eq!(
                                    libc::mprotect(
                                        (hole + page as u64) as *mut _,
                                        page,
                                        libc::PROT_READ
                                    ),
                                    0
                                );
                                let retry_before = format!("{cpu:?}");
                                let prior_memory =
                                    std::slice::from_raw_parts(cpu.gpr[0] as *const u8, 16)
                                        .to_vec();
                                let ChildStep::Fault(retry) = core_tests::no_alloc(|| {
                                    child::step(&view, &mut cpu, &mut arena)
                                }) else {
                                    panic!()
                                };
                                assert!(retry.raw_host_fault.is_some());
                                assert_eq!(retry.ram, x86jit_core::MemoryEffects::None);
                                assert_eq!(format!("{cpu:?}"), retry_before);
                                assert_eq!(
                                    std::slice::from_raw_parts(cpu.gpr[0] as *const u8, 16),
                                    prior_memory
                                );
                                assert_eq!(&*arena.written_pages(), &[(hole + page as u64) >> 12]);
                                assert_eq!(
                                    libc::mprotect(
                                        (hole + page as u64) as *mut _,
                                        page,
                                        libc::PROT_READ | libc::PROT_WRITE
                                    ),
                                    0
                                );
                                if protection == -1 {
                                    assert_eq!(
                                        libc::mmap(
                                            hole as *mut _,
                                            page,
                                            libc::PROT_READ | libc::PROT_WRITE,
                                            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
                                            -1,
                                            0
                                        ),
                                        hole as *mut _
                                    );
                                } else {
                                    assert_eq!(
                                        libc::mprotect(
                                            hole as *mut _,
                                            page,
                                            libc::PROT_READ | libc::PROT_WRITE
                                        ),
                                        0
                                    );
                                    assert_eq!(
                                        std::slice::from_raw_parts(hole as *const u8, 16),
                                        &[0xa5; 16]
                                    );
                                }
                            }
                        }
                    }
                    for (code, prefix) in [
                        (&[0x48, 0x8b, 0x00][..], 0),
                        (&[0x48, 0x89, 0x18][..], 0),
                        (&[0xc5, 0xfc, 0x10, 0x00][..], 0),
                        (&[0xc5, 0xfc, 0x11, 0x00][..], 0),
                        (&[0x48, 0x89, 0x18][..], 4),
                        (&[0xc5, 0xfc, 0x11, 0x00][..], 16),
                        (&[0xf3, 0xaa][..], 2),
                    ] {
                        let mut arena = ChildArena::new(if prefix == 0 { 1 } else { 2 });
                        // Install code directly: the parent's normal classifier is not installed here.
                        std::ptr::copy_nonoverlapping(code.as_ptr(), ptr.cast::<u8>(), code.len());
                        std::ptr::write_bytes((hole - 32) as *mut u8, 0xa5, 64);
                        assert_eq!(libc::mprotect(hole as *mut _, page, libc::PROT_NONE), 0);
                        let mut cpu = core_tests::initial_cpu();
                        cpu.rip = base;
                        cpu.gpr[0] = hole - prefix;
                        cpu.gpr[7] = hole - prefix;
                        cpu.gpr[1] = 5;
                        let before = format!("{cpu:?}");
                        let ChildStep::Fault(f) =
                            core_tests::no_alloc(|| child::step(&view, &mut cpu, &mut arena))
                        else {
                            panic!("{code:x?}")
                        };
                        assert_eq!(f.origin, FaultOrigin::HostAccess);
                        assert!(f.raw_host_fault.is_some());
                        assert_eq!(
                            f.address,
                            Knowledge::Known(x86jit_core::FaultAddress::new(
                                hole,
                                x86jit_core::AddressPrecision::ExactByte,
                                x86jit_core::AddressSpace::Linear
                            ))
                        );
                        if prefix == 0 {
                            assert!(arena.written_pages().is_empty(), "{code:x?}");
                        } else {
                            assert_eq!(&*arena.written_pages(), &[(hole - 1) >> 12]);
                        }
                        assert_eq!(cpu.rip, base);
                        if code == [0xf3, 0xaa] {
                            assert_eq!(cpu.gpr[1], 3);
                            assert_eq!(cpu.gpr[7], hole);
                        } else {
                            assert_eq!(format!("{cpu:?}"), before);
                        }
                        assert_eq!(
                            libc::mprotect(
                                hole as *mut _,
                                page,
                                libc::PROT_READ | libc::PROT_WRITE
                            ),
                            0
                        );
                        assert_eq!(
                            std::slice::from_raw_parts(hole as *const u8, 32),
                            &[0xa5; 32]
                        );
                        if prefix != 0 {
                            let expected = if code == [0xf3, 0xaa] {
                                vec![(hole - prefix) as u8; prefix as usize]
                            } else if prefix == 4 {
                                7u32.to_le_bytes().to_vec()
                            } else {
                                core_tests::initial_cpu().xmm[0].to_le_bytes().to_vec()
                            };
                            assert_eq!(
                                std::slice::from_raw_parts(
                                    (hole - prefix) as *const u8,
                                    prefix as usize
                                ),
                                expected
                            );
                        }
                        // A fault with no writes must leave room for a different page.
                        if prefix == 0 && code.last() == Some(&0x18) {
                            cpu.gpr[0] = hole + page as u64;
                            assert!(matches!(
                                core_tests::no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
                                ChildStep::Continue
                            ));
                            assert_eq!(&*arena.written_pages(), &[(hole + page as u64) >> 12]);
                        }
                    }

                    // Each signal recovers into the record belonging to its own thread.
                    std::ptr::copy_nonoverlapping([0x48, 0x8b, 0x00].as_ptr(), ptr.cast::<u8>(), 3);
                    assert_eq!(
                        libc::mprotect((hole as usize) as *mut _, page, libc::PROT_NONE),
                        0
                    );
                    std::thread::scope(|scope| {
                        for i in 0..3 {
                            let view = &view;
                            scope.spawn(move || {
                                let mut arena = ChildArena::new(4);
                                let mut cpu = core_tests::initial_cpu();
                                cpu.rip = base;
                                cpu.gpr[0] = hole + i * 8;
                                let ChildStep::Fault(f) = core_tests::no_alloc(|| {
                                    child::step(view, &mut cpu, &mut arena)
                                }) else {
                                    panic!()
                                };
                                assert_eq!(
                                    f.address,
                                    Knowledge::Known(x86jit_core::FaultAddress::new(
                                        hole + i * 8,
                                        x86jit_core::AddressPrecision::ExactByte,
                                        x86jit_core::AddressSpace::Linear
                                    ))
                                );
                                assert!(f.raw_host_fault.is_some());
                                assert!(arena.written_pages().is_empty());
                            });
                        }
                    });
                    assert_eq!(
                        libc::mprotect(
                            (hole as usize) as *mut _,
                            page,
                            libc::PROT_READ | libc::PROT_WRITE
                        ),
                        0
                    );
                    // A complete instruction at the boundary must not fetch the next page.
                    *((hole - 1) as *mut u8) = 0x90;
                    assert_eq!(
                        libc::mprotect((hole as usize) as *mut _, page, libc::PROT_NONE),
                        0
                    );
                    let mut cpu = core_tests::initial_cpu();
                    cpu.rip = hole - 1;
                    assert!(matches!(
                        core_tests::no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
                        ChildStep::Continue
                    ));
                    cpu.rip = hole;
                    let ChildStep::Fault(f) =
                        core_tests::no_alloc(|| child::step(&view, &mut cpu, &mut arena))
                    else {
                        panic!()
                    };
                    assert_eq!(
                        f.transfer_access,
                        Knowledge::Known(x86jit_core::FaultAccess::Execute)
                    );
                    assert!(f.raw_host_fault.is_some());
                    assert!(arena.written_pages().is_empty());
                    // The classifier is inert on an unrelated PC, even with no record.
                    let mut regs = HostRegisters::default();
                    let old = regs;
                    assert!(!child::classify(
                        0,
                        &mut regs,
                        RawHostFault::new(HostFaultPlatform::MacOsAarch64)
                    ));
                    assert_eq!(regs, old);
                    std::mem::forget(arena);
                    std::mem::forget(view);
                    std::mem::forget(vm);
                });
                libc::_exit(i32::from(result.is_err()));
            }
            let mut status = 0;
            assert_eq!(libc::waitpid(pid, &mut status, 0), pid);
            assert_eq!(status, 0);
        }
    }
}

#[test]
fn child_invalidation_rejects_first_publication_already_compiling() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Barrier,
    };
    struct PausedCompile {
        jit: JitBackend,
        first: AtomicBool,
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
    }
    impl Backend for PausedCompile {
        fn materialize(
            &self,
            ir: &x86jit_core::IrBlock,
            consistency: x86jit_core::MemConsistency,
            mmio: Option<(u64, u64)>,
            base: u64,
        ) -> x86jit_core::CachedBlock {
            if self.first.swap(false, Ordering::Relaxed) {
                self.entered.wait();
                self.release.wait();
            }
            self.jit.materialize(ir, consistency, mmio, base)
        }
    }
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let backend = PausedCompile {
        jit: JitBackend::new(),
        first: AtomicBool::new(true),
        entered: entered.clone(),
        release: release.clone(),
    };
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(backend));
    vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x3000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0x89, 0x18]).unwrap();
    vm.write_bytes(0x3000, &[0xb8, 1, 0, 0, 0, 0xf4]).unwrap();
    let view = vm.child_view(CpuMode::Long64);
    std::thread::scope(|scope| {
        let vm = &vm;
        let compile = scope.spawn(move || {
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = 0x3000;
            assert!(matches!(cpu.run(vm, Some(100)), Exit::Hlt));
            cpu.cpu.gpr[0]
        });
        entered.wait();
        let mut cpu = core_tests::initial_cpu();
        cpu.gpr[0] = 0x3001;
        cpu.gpr[3] = 2;
        let mut arena = ChildArena::new(2);
        assert!(matches!(
            core_tests::no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
            ChildStep::Continue
        ));
        vm.invalidate_written_pages(&arena);
        release.wait();
        assert_eq!(
            compile.join().unwrap(),
            2,
            "old lifted bytes must not publish after child invalidation"
        );
    });
}

#[test]
fn child_replays_stored_inline_corpus_against_both_backends() {
    use x86jit_tests::oracle::{run_with_backend, VectorInput};
    use x86jit_tests::vector::{SnapFlags, TestVector};
    fn collect(path: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                collect(&path, files);
            } else if path.extension().is_some_and(|ext| ext == "ron") {
                files.push(path);
            }
        }
    }
    let mut files = Vec::new();
    collect(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("vectors"),
        &mut files,
    );
    assert!(!files.is_empty());
    for file in files {
        let vector = TestVector::from_ron(&std::fs::read_to_string(file).unwrap()).unwrap();
        let input = VectorInput {
            cpu_init: vector.cpu_init.clone(),
            mem_init: vector.mem_init.clone(),
            entry: vector.entry,
            run: vector.run,
        };
        let end = input
            .mem_init
            .iter()
            .map(|c| c.addr + c.bytes.len() as u64)
            .max()
            .unwrap();
        let mut vm = Vm::new(VmConfig::flat((end + 0xfff) & !0xfff));
        for chunk in &input.mem_init {
            vm.map(chunk.addr, chunk.bytes.len(), Prot::RWX, chunk.kind.into())
                .unwrap();
            vm.write_bytes(chunk.addr, &chunk.bytes).unwrap();
        }
        let view = vm.child_view(CpuMode::Long64);
        let mut arena = ChildArena::new(64);
        let snap = &input.cpu_init;
        let mut cpu = x86jit_core::CpuState {
            gpr: snap.gpr,
            rip: input.entry,
            flags: snap.flags.into(),
            fs_base: snap.fs_base,
            gs_base: snap.gs_base,
            xmm: snap.xmm,
            ymm_hi: snap.ymm_hi,
            zmm_hi: snap.zmm_hi,
            kmask: snap.kmask,
            mxcsr: snap.mxcsr,
            fpu_cw: snap.fpu_cw,
            fpu_top: u32::from(snap.fpu_top),
            ..Default::default()
        };
        for (i, bytes) in snap.st.iter().enumerate() {
            cpu.fpr[(usize::from(snap.fpu_top) + i) & 7] = *bytes;
        }
        let mut halted = false;
        for _ in 0..1024 {
            match core_tests::no_alloc(|| child::step(&view, &mut cpu, &mut arena)) {
                ChildStep::Continue => {}
                ChildStep::Hlt => {
                    halted = true;
                    break;
                }
                other => panic!("{}: unexpected child exit {other:?}", vector.name),
            }
        }
        assert!(halted, "{}: exceeded child step budget", vector.name);
        for backend in [
            Box::new(InterpreterBackend) as Box<dyn Backend>,
            Box::new(JitBackend::new()),
        ] {
            let expected = run_with_backend(&input, backend);
            assert_eq!(cpu.gpr, expected.cpu.gpr, "{}", vector.name);
            assert_eq!(cpu.rip, expected.cpu.rip, "{}", vector.name);
            assert_eq!(
                SnapFlags::from(cpu.flags),
                expected.cpu.flags,
                "{}",
                vector.name
            );
            assert_eq!(cpu.xmm, expected.cpu.xmm);
            assert_eq!(cpu.ymm_hi, expected.cpu.ymm_hi);
            assert_eq!(cpu.zmm_hi, expected.cpu.zmm_hi);
            assert_eq!(cpu.kmask, expected.cpu.kmask);
            for chunk in expected.mem {
                let mut bytes = vec![0; chunk.bytes.len()];
                vm.mem.read_bytes(chunk.addr, &mut bytes).unwrap();
                assert_eq!(bytes, chunk.bytes, "{}", vector.name);
            }
        }
    }
}

#[test]
fn child_masked_vectors_suppress_inactive_noncanonical_lanes() {
    use iced_x86::code_asm::*;
    // SDM Vol 2C §5.2 VMASKMOV: inactive elements suppress memory accesses.
    // The backing ends at the canonical boundary; only low elements participate.
    let top = 1u64 << 47;
    let base = top - 4096;
    for jit in [false, true] {
        for form in 0..4 {
            let mut asm = CodeAssembler::new(64).unwrap();
            match form {
                0 => asm.vmaskmovps(ymm0, ymm1, ymmword_ptr(rax)).unwrap(),
                1 => asm.vmaskmovps(ymmword_ptr(rax), ymm1, ymm0).unwrap(),
                2 => asm.vmovups(zmm0.k1(), zmmword_ptr(rax)).unwrap(),
                _ => asm.vmovups(zmmword_ptr(rax).k1(), zmm0).unwrap(),
            }
            asm.hlt().unwrap();
            let code = asm.assemble(base).unwrap();
            let backing = vec![0u8; 4096].into_boxed_slice();
            let ptr = Box::into_raw(backing).cast::<u8>();
            let ram = x86jit_core::HostRam {
                ptr,
                len: 4096,
                guest_base: base,
                protect: None,
                dtor: Box::new(|p, n| {
                    // SAFETY: exactly the boxed slice transferred into HostRam above.
                    unsafe {
                        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(p, n)));
                    }
                }),
            };
            let backend: Box<dyn Backend> = if jit {
                Box::new(JitBackend::new())
            } else {
                Box::new(InterpreterBackend)
            };
            let mut vm = Vm::with_backend_host_ram(VmConfig::flat(top), backend, ram).unwrap();
            vm.map(base, 4096, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(base, &code).unwrap();
            vm.write_bytes(top - 16, &[0x12; 16]).unwrap();
            let view = vm.child_view(CpuMode::Long64);
            let mut normal = vm.new_vcpu();
            normal.cpu.rip = base;
            normal.cpu.gpr[0] = top - 16;
            normal.cpu.xmm[0] = 0x1234;
            normal.cpu.xmm[1] = u128::MAX;
            normal.cpu.ymm_hi[1] = 0;
            normal.cpu.kmask[1] = 0xf;
            let mut cpu = normal.cpu.clone();
            assert!(matches!(normal.run(&vm, Some(100)), Exit::Hlt));
            let mut expected = [0; 16];
            vm.read_bytes(top - 16, &mut expected).unwrap();
            vm.write_bytes(top - 16, &[0x12; 16]).unwrap();
            let mut arena = ChildArena::new(4);
            assert!(
                matches!(
                    core_tests::no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
                    ChildStep::Continue
                ),
                "form={form} jit={jit}"
            );
            assert!(matches!(
                core_tests::no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
                ChildStep::Hlt
            ));
            assert_eq!(format!("{:?}", normal.cpu), format!("{cpu:?}"));
            let mut actual = [0; 16];
            vm.read_bytes(top - 16, &mut actual).unwrap();
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn child_locked_forms_match_both_backends() {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        let mut forms = core_tests::locked_code_forms(mode);
        forms.push(vec![0xf0, 0x0f, 0xc7, 0x0e]);
        if mode == CpuMode::Long64 {
            forms.push(vec![0xf0, 0x48, 0x0f, 0xc7, 0x0e]);
        }
        for code in forms {
            for jit in [false, true] {
                for success in [false, true] {
                    let backend: Box<dyn Backend> = if jit {
                        Box::new(JitBackend::new())
                    } else {
                        Box::new(InterpreterBackend)
                    };
                    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
                    vm.set_cpu_mode(mode);
                    vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
                    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
                    let mut text = code.clone();
                    text.push(0xf4);
                    vm.write_bytes(0x1000, &text).unwrap();
                    let data = [0x12; 16];
                    vm.write_bytes(0x3100, &data).unwrap();
                    let mut normal = vm.new_vcpu();
                    normal.cpu = core_tests::initial_cpu();
                    normal.cpu.gpr[0] = if success {
                        0x1212_1212_1212_1212
                    } else {
                        0x1234
                    };
                    normal.cpu.gpr[2] = if success {
                        0x1212_1212_1212_1212
                    } else {
                        0x5678
                    };
                    normal.cpu.gpr[3] = 1;
                    normal.cpu.flags.cf = true;
                    let mut cpu = normal.cpu.clone();
                    assert!(
                        matches!(normal.run(&vm, Some(100)), Exit::Hlt),
                        "{mode:?} jit={jit} {code:x?}"
                    );
                    let mut expected = [0; 16];
                    vm.read_bytes(0x3100, &mut expected).unwrap();
                    vm.write_bytes(0x3100, &data).unwrap();
                    let view = vm.child_view(mode);
                    let mut arena = ChildArena::new(4);
                    assert!(matches!(
                        core_tests::no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
                        ChildStep::Continue
                    ));
                    assert!(matches!(
                        core_tests::no_alloc(|| child::step(&view, &mut cpu, &mut arena)),
                        ChildStep::Hlt
                    ));
                    assert_eq!(
                        format!("{:?}", normal.cpu),
                        format!("{cpu:?}"),
                        "{mode:?} jit={jit} {code:x?}"
                    );
                    let mut actual = [0; 16];
                    vm.read_bytes(0x3100, &mut actual).unwrap();
                    assert_eq!(expected, actual, "{mode:?} jit={jit} {code:x?}");
                }
            }
        }
    }
}
