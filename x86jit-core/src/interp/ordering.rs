//! Interpreter counterparts of JIT gload/gstore and call_memory_helper (§8.2.3).
use std::sync::atomic::{fence, Ordering};

#[derive(Clone, Copy)]
pub(crate) enum Direction {
    Read,
    Write,
    ReadWrite,
}

// 0 = Fast, 1 = AcqRel, 2 = FullTso. Const specialization erases all ordering
// machinery for Fast and for every production walk on an x86-64 host.
pub(crate) struct Barrier<const TIER: u8> {
    after: bool,
}
impl<const TIER: u8> Barrier<TIER> {
    #[inline(always)]
    pub(crate) fn new(direction: Option<Direction>, helper: bool) -> Self {
        let loads = matches!(direction, Some(Direction::Read | Direction::ReadWrite));
        let stores = matches!(direction, Some(Direction::Write | Direction::ReadWrite));
        let (before, after) = match TIER {
            1 => (stores, loads),
            2 => (helper && loads, loads || stores),
            _ => (false, false),
        };
        // SDM Vol 3A §11.2.2, §11.2.4.1: preserve guest memory ordering,
        // including ordinary accesses around string and state-image helpers.
        if before {
            full_fence();
        }
        Self { after }
    }
}
impl<const TIER: u8> Drop for Barrier<TIER> {
    #[inline(always)]
    fn drop(&mut self) {
        // Helpers may commit partial progress before returning a fault or exit.
        if self.after {
            full_fence();
        }
    }
}

#[inline(always)]
fn full_fence() {
    fence(Ordering::SeqCst);
    #[cfg(test)]
    EVENTS.with(|events| events.borrow_mut().push("fence"));
}

#[cfg(test)]
std::thread_local! {
    pub(crate) static EVENTS: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Exit, MemConsistency, Prot, Reg, RegionKind, StepResult, Vm, VmConfig};

    fn take_events() -> Vec<&'static str> {
        EVENTS.with(|events| std::mem::take(&mut *events.borrow_mut()))
    }

    #[test]
    fn inline_and_helper_fence_positions() {
        // SDM Vol 3A §11.2.2: preserve load/load, load/store and store/store
        // ordering. FullTso additionally closes the store/load window on ARM.
        fn check<const TIER: u8>(direction: Direction, helper: bool, expected: &[&str]) {
            take_events();
            {
                let _ordering = Barrier::<TIER>::new(Some(direction), helper);
                EVENTS.with(|events| events.borrow_mut().push("access"));
            }
            assert_eq!(take_events(), expected);
        }
        for helper in [false, true] {
            for direction in [Direction::Read, Direction::Write, Direction::ReadWrite] {
                check::<0>(direction, helper, &["access"]);
            }
            check::<1>(Direction::Read, helper, &["access", "fence"]);
            check::<1>(Direction::Write, helper, &["fence", "access"]);
            check::<1>(Direction::ReadWrite, helper, &["fence", "access", "fence"]);
            check::<2>(Direction::Write, helper, &["access", "fence"]);
        }
        check::<2>(Direction::Read, false, &["access", "fence"]);
        check::<2>(Direction::Read, true, &["fence", "access", "fence"]);
        check::<2>(Direction::ReadWrite, true, &["fence", "access", "fence"]);
    }

    fn fixture(tier: MemConsistency, code: &[u8]) -> Vm {
        let mut config = VmConfig::flat(0x10000);
        config.consistency = tier;
        let mut vm = Vm::new(config);
        for page in 0..16 {
            vm.map(page * 0x1000, 0x1000, Prot::RWX, RegionKind::Ram)
                .unwrap();
        }
        vm.write_bytes(0x1000, code).unwrap();
        vm
    }

    #[test]
    fn child_steps_use_configured_tier() {
        use crate::{child, ChildArena, ChildStep, CpuMode, CpuState};
        // SDM Vol 3A §11.2.2, §11.2.4.1: ordinary and string accesses
        // retain ordering with other threads sharing the guest bytes.
        let cases: &[(&[u8], &[&str], &[&str])] = &[
            (&[0x48, 0x8b, 0x03], &["read", "fence"], &["read", "fence"]),
            (
                &[0x48, 0x89, 0x03],
                &["fence", "write", "fence", "commit"],
                &["write", "fence", "commit", "fence"],
            ),
            (
                &[0x48, 0x83, 0x03, 1],
                &["read", "fence", "fence", "write", "fence", "commit"],
                &["read", "fence", "write", "fence", "commit", "fence"],
            ),
            (
                &[0xf3, 0xa4],
                &[
                    "fence", "read", "commit", "write", "read", "commit", "write", "fence",
                ],
                &[
                    "fence", "read", "commit", "write", "read", "commit", "write", "fence",
                ],
            ),
            (
                &[0xf3, 0xaa],
                &["fence", "commit", "write", "commit", "write"],
                &["commit", "write", "commit", "write", "fence"],
            ),
            (
                &[0x0f, 0x10, 0x03],
                &["read", "read", "fence"],
                &["read", "read", "fence"],
            ),
            (
                &[0x0f, 0x11, 0x03],
                &[
                    "fence", "write", "write", "fence", "commit", "fence", "commit",
                ],
                &[
                    "write", "write", "fence", "commit", "fence", "commit", "fence",
                ],
            ),
        ];
        for tier in [
            MemConsistency::Fast,
            MemConsistency::AcqRel,
            MemConsistency::FullTso,
        ] {
            for &(code, acqrel, fulltso) in cases {
                let vm = fixture(tier, code);
                let view = vm.child_view(CpuMode::Long64);
                let mut arena = ChildArena::new(4);
                let mut cpu = CpuState {
                    rip: 0x1000,
                    ..CpuState::default()
                };
                cpu.gpr[1] = 2;
                cpu.gpr[3] = 0x8000;
                cpu.gpr[6] = 0x8000;
                cpu.gpr[7] = 0x8100;
                let mut expected = if tier == MemConsistency::FullTso {
                    fulltso
                } else {
                    acqrel
                }
                .to_vec();
                if !cfg!(target_arch = "aarch64") || tier == MemConsistency::Fast {
                    expected.retain(|&event| event != "fence");
                }
                take_events();
                assert!(matches!(
                    child::step(&view, &mut cpu, &mut arena),
                    ChildStep::Continue
                ));
                assert_eq!(take_events(), expected, "{tier:?} {code:x?}");
            }
        }
    }

    fn check_child_wide_store_commits(tier: MemConsistency, code: &[u8], lanes: usize) {
        use crate::{child, ChildArena, ChildStep, CpuMode, CpuState};
        // The commit guard is per transfer, conservatively fencing both halves
        // of each vstore lane (the ordinary interpreter fences once per lane).
        let vm = fixture(tier, code);
        let mut cpu = CpuState {
            rip: 0x1000,
            ..CpuState::default()
        };
        cpu.gpr[3] = 0x8000;
        let mut normal = cpu.clone();
        take_events();
        assert!(matches!(
            crate::interp::step_one(
                &vm.mem,
                &mut normal,
                CpuMode::Long64,
                &mut Vec::new(),
                &mut crate::interp::RetireInfo {
                    consistency: tier,
                    ..Default::default()
                },
            ),
            StepResult::Continue
        ));
        let normal_events = take_events();
        let view = vm.child_view(CpuMode::Long64);
        let mut arena = ChildArena::new(4);
        take_events();
        assert!(matches!(
            child::step(&view, &mut cpu, &mut arena),
            ChildStep::Continue
        ));
        let events = take_events();
        let first_commit = events.iter().position(|&e| e == "commit").unwrap();
        let ordered = cfg!(target_arch = "aarch64") && tier != MemConsistency::Fast;
        let start = first_commit - usize::from(ordered && tier == MemConsistency::AcqRel);
        let commits = &events[start..];
        let transfer: &[&str] = match (ordered, tier) {
            (true, MemConsistency::AcqRel) => &["fence", "commit"],
            (true, MemConsistency::FullTso) => &["commit", "fence"],
            _ => &["commit"],
        };
        assert_eq!(commits, transfer.repeat(lanes * 2), "{tier:?} {code:x?}");
        // Remove only the extra intra-lane fences, then compare the
        // actual writes and lane boundaries with the normal interpreter.
        let mut lane_events = Vec::new();
        for lane in commits.chunks_exact(transfer.len() * 2) {
            let extra = match tier {
                MemConsistency::AcqRel => 2,
                MemConsistency::FullTso => 1,
                _ => usize::MAX,
            };
            for (i, &event) in lane.iter().enumerate() {
                if !ordered || i != extra {
                    lane_events.push(if event == "commit" { "write" } else { event });
                }
            }
        }
        assert_eq!(lane_events, normal_events, "{tier:?} {code:x?}");
    }

    #[test]
    fn child_256_store_commits_fast() {
        check_child_wide_store_commits(MemConsistency::Fast, &[0xc5, 0xfc, 0x11, 0x03], 2);
    }

    #[test]
    fn child_256_store_commits_acqrel() {
        check_child_wide_store_commits(MemConsistency::AcqRel, &[0xc5, 0xfc, 0x11, 0x03], 2);
    }

    #[test]
    fn child_256_store_commits_fulltso() {
        check_child_wide_store_commits(MemConsistency::FullTso, &[0xc5, 0xfc, 0x11, 0x03], 2);
    }

    #[test]
    fn child_512_store_commits_fast() {
        check_child_wide_store_commits(
            MemConsistency::Fast,
            &[0x62, 0xf1, 0x7c, 0x48, 0x11, 0x03],
            4,
        );
    }

    #[test]
    fn child_512_store_commits_acqrel() {
        check_child_wide_store_commits(
            MemConsistency::AcqRel,
            &[0x62, 0xf1, 0x7c, 0x48, 0x11, 0x03],
            4,
        );
    }

    #[test]
    fn child_512_store_commits_fulltso() {
        check_child_wide_store_commits(
            MemConsistency::FullTso,
            &[0x62, 0xf1, 0x7c, 0x48, 0x11, 0x03],
            4,
        );
    }

    #[test]
    fn child_faulting_store_keeps_access_barriers() {
        use crate::{child, ChildArena, ChildStep, CpuMode, CpuState};
        for tier in [MemConsistency::AcqRel, MemConsistency::FullTso] {
            let mut vm = fixture(tier, &[0x48, 0x89, 0x03]);
            vm.mem.unmap(0x9000, 0x1000).unwrap();
            let view = vm.child_view(CpuMode::Long64);
            let mut arena = ChildArena::new(4);
            let mut cpu = CpuState {
                rip: 0x1000,
                ..CpuState::default()
            };
            cpu.gpr[3] = 0x9000;
            take_events();
            assert!(matches!(
                child::step(&view, &mut cpu, &mut arena),
                ChildStep::Fault(_)
            ));
            let mut expected = if tier == MemConsistency::AcqRel {
                vec!["fence", "write"]
            } else {
                vec!["write", "fence"]
            };
            if !cfg!(target_arch = "aarch64") {
                expected.retain(|&event| event != "fence");
            }
            assert_eq!(take_events(), expected, "{tier:?}");
            assert_eq!(cpu.rip, 0x1000);
            assert!(arena.written_pages().is_empty());
        }
    }

    // Keep the signal adapter in tests: production core depends only on iced-x86.
    // Darwin layouts: sys/signal.h, sys/_types/_ucontext.h, arm/_mcontext.h
    // and mach/arm/_structs.h. The mcontext prefix stops before the NEON state.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    mod child_host_fault {
        use super::*;
        use crate::jit_abi::fault::HostRegisters;
        use crate::{
            child, ChildArena, ChildStep, CpuMode, CpuState, HostFaultPlatform, HostRam,
            RawHostFault,
        };
        use std::ffi::c_void;

        #[repr(C)]
        struct Action {
            handler: usize,
            mask: u32,
            flags: i32,
        }
        #[repr(C)]
        struct SignalInfo {
            signal: i32,
            errno: i32,
            code: i32,
        }
        #[repr(C)]
        struct Stack {
            ptr: *mut c_void,
            size: usize,
            flags: i32,
        }
        #[repr(C)]
        struct Context {
            on_stack: i32,
            mask: u32,
            stack: Stack,
            link: *mut c_void,
            size: usize,
            machine: *mut Machine,
        }
        #[repr(C)]
        struct Machine {
            far: u64,
            esr: u32,
            exception: u32,
            x: [u64; 29],
            fp: u64,
            lr: u64,
            sp: u64,
            pc: u64,
            cpsr: u32,
            pad: u32,
        }
        extern "C" {
            fn fork() -> i32;
            fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
            fn _exit(status: i32) -> !;
            fn alarm(seconds: u32) -> u32;
            fn getpagesize() -> i32;
            fn mmap(
                addr: *mut c_void,
                len: usize,
                prot: i32,
                flags: i32,
                fd: i32,
                offset: i64,
            ) -> *mut c_void;
            fn munmap(addr: *mut c_void, len: usize) -> i32;
            fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
            fn sigaction(signal: i32, action: *const Action, old: *mut Action) -> i32;
        }
        extern "C" fn handler(signal: i32, info: *mut SignalInfo, context: *mut Context) {
            // SAFETY: SA_SIGINFO supplies this accessing thread's Darwin context.
            unsafe {
                let machine = &mut *(*context).machine;
                // Arm ARM DDI0487I.a §D17.2.37: synchronous data-access abort.
                if (*info).code <= 0
                    || ![0x24, 0x25].contains(&(machine.esr >> 26))
                    || !(4..=15).contains(&(machine.esr & 63))
                {
                    _exit(91);
                }
                let mut raw = RawHostFault::new(HostFaultPlatform::MacOsAarch64);
                raw.signal = Some(signal);
                raw.signal_code = Some((*info).code);
                raw.fault_address = Some(machine.far);
                raw.esr = Some(u64::from(machine.esr));
                let mut regs = HostRegisters {
                    pc: machine.pc,
                    scratch: machine.x[9],
                    ..HostRegisters::default()
                };
                if !child::classify(regs.pc as usize, &mut regs, raw) {
                    _exit(92);
                }
                machine.pc = regs.pc;
            }
        }

        #[test]
        fn child_host_fault_store_keeps_post_fence() {
            // Mirror the protected child_step setup, with the unit fence recorder
            // enabled. SDM Vol 3A §11.2.2, §11.2.4.1: retain ordering of any
            // completed store prefix before reporting the fault to the embedder.
            // SAFETY: faults and signal handlers are isolated in a reaped child.
            unsafe {
                let pid = fork();
                assert!(pid >= 0);
                if pid == 0 {
                    alarm(10);
                    let result = std::panic::catch_unwind(|| {
                        let action = Action {
                            handler: handler as *const () as usize,
                            mask: 0,
                            flags: 0x40,
                        };
                        for signal in [10, 11] {
                            assert_eq!(sigaction(signal, &action, std::ptr::null_mut()), 0);
                        }
                        let page = getpagesize() as usize;
                        for tier in [MemConsistency::FullTso, MemConsistency::AcqRel] {
                            for (code, prefix, lanes) in [
                                (&[0x48, 0x89, 0x03][..], 0, 0),
                                (&[0x48, 0x89, 0x03][..], 4, 0),
                                (&[0x0f, 0x11, 0x03][..], 8, 1),
                                (&[0xc5, 0xfc, 0x11, 0x03][..], 24, 2),
                                (&[0xc5, 0xfc, 0x11, 0x03][..], 20, 2),
                                (&[0x62, 0xf1, 0x7c, 0x48, 0x11, 0x03][..], 56, 4),
                                (&[0x62, 0xf1, 0x7c, 0x48, 0x11, 0x03][..], 52, 4),
                                (&[0xf3, 0xaa][..], 2, 0),
                            ] {
                                let ptr = mmap(std::ptr::null_mut(), 2 * page, 3, 0x1002, -1, 0);
                                assert_ne!(ptr as usize, usize::MAX);
                                let base = ptr as u64;
                                let hole = base + page as u64;
                                let mut config = VmConfig::flat(base + 2 * page as u64);
                                config.consistency = tier;
                                let ram = HostRam {
                                    ptr: ptr.cast(),
                                    len: 2 * page,
                                    guest_base: base,
                                    protect: None,
                                    dtor: Box::new(|p, n| {
                                        munmap(p.cast(), n);
                                    }),
                                };
                                let mut vm = Vm::with_backend_host_ram(
                                    config,
                                    Box::new(crate::InterpreterBackend),
                                    ram,
                                )
                                .unwrap();
                                vm.map(base, 2 * page, Prot::RWX, RegionKind::Ram).unwrap();
                                vm.mem.set_host_fault_recovery(true);
                                vm.write_bytes(base, code).unwrap();
                                let view = vm.child_view(CpuMode::Long64);
                                let mut arena = ChildArena::new(2);
                                let mut cpu = CpuState {
                                    rip: base,
                                    ..CpuState::default()
                                };
                                cpu.gpr[1] = 3;
                                cpu.gpr[3] = hole - prefix;
                                cpu.gpr[7] = hole - prefix;
                                assert_eq!(mprotect(hole as *mut _, page, 0), 0);
                                take_events();
                                let ChildStep::Fault(fault) =
                                    child::step(&view, &mut cpu, &mut arena)
                                else {
                                    panic!("expected recovered host fault")
                                };
                                let events = take_events();
                                assert!(fault.raw_host_fault.is_some());
                                assert_eq!(fault.origin, crate::FaultOrigin::HostAccess);
                                let expected = if lanes != 0 {
                                    let staging = if tier == MemConsistency::AcqRel {
                                        ["fence", "write", "write"]
                                    } else {
                                        ["write", "write", "fence"]
                                    };
                                    let commit = if tier == MemConsistency::AcqRel {
                                        ["fence", "commit"]
                                    } else {
                                        ["commit", "fence"]
                                    };
                                    let mut expected = staging.repeat(lanes);
                                    // Include the faulting transfer, even when only
                                    // its bytewise prefix reaches the writable page.
                                    expected.extend(commit.repeat((prefix / 8 + 1) as usize));
                                    expected
                                } else {
                                    let expected: &[&str] = match (tier, code) {
                                        (MemConsistency::AcqRel, [0xf3, 0xaa]) => &[
                                            "fence", "commit", "write", "commit", "write",
                                            "commit", "write",
                                        ],
                                        (_, [0xf3, 0xaa]) => &[
                                            "commit", "write", "commit", "write", "commit",
                                            "write", "fence",
                                        ],
                                        (MemConsistency::AcqRel, _) => {
                                            &["fence", "write", "fence", "commit"]
                                        }
                                        _ => &["write", "fence", "commit", "fence"],
                                    };
                                    expected.to_vec()
                                };
                                assert_eq!(events, expected, "{tier:?} {code:x?}, prefix={prefix}");
                                assert_eq!(mprotect(hole as *mut _, page, 3), 0);
                            }
                        }
                    });
                    _exit(i32::from(result.is_err()));
                }
                let mut status = 0;
                assert_eq!(waitpid(pid, &mut status, 0), pid);
                assert_eq!(status, 0);
            }
        }
    }

    #[test]
    fn child_partial_string_fault_keeps_post_fence() {
        use crate::{child, ChildArena, ChildStep, CpuMode, CpuState};
        for tier in [MemConsistency::AcqRel, MemConsistency::FullTso] {
            let mut vm = fixture(tier, &[0xf3, 0xa4]);
            vm.mem.unmap(0x9000, 0x1000).unwrap();
            let view = vm.child_view(CpuMode::Long64);
            let mut arena = ChildArena::new(4);
            let mut cpu = CpuState {
                rip: 0x1000,
                ..CpuState::default()
            };
            cpu.gpr[1] = 2;
            cpu.gpr[6] = 0x8fff;
            cpu.gpr[7] = 0x8100;
            take_events();
            assert!(matches!(
                child::step(&view, &mut cpu, &mut arena),
                ChildStep::Fault(_)
            ));
            let mut expected = vec!["fence", "read", "commit", "write", "read", "fence"];
            if !cfg!(target_arch = "aarch64") {
                expected.retain(|&event| event != "fence");
            }
            assert_eq!(take_events(), expected, "{tier:?}");
            assert_eq!(cpu.gpr[1], 1);
        }
    }

    #[test]
    fn vm_entries_use_configured_tier() {
        // mov [rbx],rax; mov rax,[rbx]; hlt. InterpreterBackend never tiers up.
        for tier in [
            MemConsistency::Fast,
            MemConsistency::AcqRel,
            MemConsistency::FullTso,
        ] {
            let vm = fixture(tier, &[0x48, 0x89, 0x03, 0x48, 0x8b, 0x03, 0xf4]);
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rbx, 0x8000);
            cpu.set_reg(Reg::Rip, 0x1000);
            take_events();
            assert!(matches!(cpu.step_instruction(&vm), StepResult::Continue));
            let ordered = cfg!(target_arch = "aarch64") && tier != MemConsistency::Fast;
            assert_eq!(
                take_events().iter().filter(|&&e| e == "fence").count(),
                usize::from(ordered)
            );
            cpu.set_reg(Reg::Rip, 0x1000);
            assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
            assert_eq!(
                take_events().iter().filter(|&&e| e == "fence").count(),
                2 * usize::from(ordered)
            );
        }
    }

    #[test]
    fn inline_access_sequences_match_jit() {
        use crate::{Memory, MemoryModel};
        let mut mem = Memory::new(MemoryModel::Flat { size: 0x10000 });
        mem.map(0, 0x10000, Prot::RW, RegionKind::Ram).unwrap();
        fn check<const TIER: u8>(mem: &Memory, expected: &[&str]) {
            take_events();
            let access = crate::memory::Access::<false, TIER>::new(mem);
            access.write(0x8000, 1, 8).unwrap();
            assert_eq!(access.read(0x8000, 8).unwrap(), 1);
            assert_eq!(take_events(), expected);
        }
        check::<0>(&mem, &["write", "read"]);
        check::<1>(&mem, &["fence", "write", "read", "fence"]);
        check::<2>(&mem, &["write", "fence", "read", "fence"]);
    }

    #[test]
    fn helper_post_fence_survives_partial_string_fault() {
        use crate::interp::{interpret_block_ordered, RetireInfo};
        use crate::lift::{lift_one, CpuMode, FetchAddr};
        // MOVSB commits one element, then faults on its next source load.
        let mut vm = fixture(MemConsistency::Fast, &[0xf3, 0xa4]);
        vm.mem.unmap(0x9000, 0x1000).unwrap();
        let ir = lift_one(&vm.mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
        for tier in [1, 2] {
            let mut cpu = crate::CpuState::new();
            cpu.gpr[1] = 2;
            cpu.gpr[6] = 0x8fff;
            cpu.gpr[7] = 0x8100;
            take_events();
            let mut scratch = Vec::new();
            let mut info = RetireInfo::default();
            let result = if tier == 1 {
                interpret_block_ordered::<false, true, 1>(
                    &ir,
                    &mut cpu,
                    &vm.mem,
                    &mut scratch,
                    &mut info,
                )
            } else {
                interpret_block_ordered::<false, true, 2>(
                    &ir,
                    &mut cpu,
                    &vm.mem,
                    &mut scratch,
                    &mut info,
                )
            };
            assert!(matches!(result, crate::interp::BlockResult::Exit(_)));
            assert_eq!(cpu.gpr[1], 1);
            assert_eq!(take_events(), ["fence", "read", "write", "read", "fence"]);
        }
    }

    #[test]
    fn vm_mmio_replay_retains_ordering() {
        for tier in [MemConsistency::AcqRel, MemConsistency::FullTso] {
            let mut vm = fixture(tier, &[0x48, 0x8b, 0x03, 0x48, 0x89, 0x07, 0xf4]);
            vm.mem.unmap(0x8000, 0x1000).unwrap();
            vm.map(0x8000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rcx, 1);
            cpu.set_reg(Reg::Rbx, 0x8000);
            cpu.set_reg(Reg::Rdi, 0x7000);
            take_events();
            assert!(matches!(cpu.run(&vm, Some(8)), Exit::MmioRead { .. }));
            let ordered = cfg!(target_arch = "aarch64");
            assert_eq!(
                take_events().iter().filter(|&&e| e == "fence").count(),
                usize::from(ordered)
            );
            cpu.complete_mmio_read(0x42);
            assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
            assert_eq!(
                take_events().iter().filter(|&&e| e == "fence").count(),
                2 * usize::from(ordered)
            );
            assert_eq!(vm.mem.read(0x7000, 1).unwrap(), 0x42);
        }
    }

    fn interrupt_events(tier: u8, partial: bool) -> &'static [&'static str] {
        // SDM Vol 3B §23.1.4: the interrupt frame and two-word IVT entry.
        // SDM Vol 3A §11.2.2: preserve ordering of the frame and vector accesses.
        match (tier, partial) {
            (0, false) => &["write", "write", "write", "read", "read"],
            (1, false) => &[
                "fence", "write", "fence", "write", "fence", "write", "read", "fence", "read",
                "fence",
            ],
            (2, false) => &[
                "write", "fence", "write", "fence", "write", "fence", "read", "fence", "read",
                "fence",
            ],
            (0, true) => &["write", "write"],
            (1, true) => &["fence", "write", "fence", "write"],
            (2, true) => &["write", "fence", "write", "fence"],
            _ => unreachable!("unknown ordering tier"),
        }
    }

    fn real16_fixture(tier: MemConsistency, code: &[u8], partial: bool) -> Vm {
        let mut vm = fixture(tier, code);
        vm.set_cpu_mode(crate::CpuMode::Real16);
        for vector in [0, 0x30] {
            vm.mem.write(vector * 4, 0x20, 2).unwrap();
            vm.mem.write(vector * 4 + 2, 0x200, 2).unwrap();
        }
        if partial {
            // The first push succeeds at 0x8000, the second faults at 0x7ffe.
            vm.mem.unmap(0x7000, 0x1000).unwrap();
        }
        vm
    }

    fn real16_cpu() -> crate::CpuState {
        let mut cpu = crate::CpuState::new();
        cpu.cs = 0x100;
        cpu.gpr[4] = 0x8002;
        cpu.flags.if_ = true;
        cpu
    }

    fn check_interrupt_frame(vm: &Vm, cpu: &crate::CpuState, saved_ip: u64) {
        assert_eq!(cpu.cs, 0x200);
        assert_eq!(cpu.rip, 0x20);
        assert_eq!(cpu.gpr[4], 0x7ffc);
        assert!(!cpu.flags.if_);
        assert_eq!(vm.mem.read(0x7ffc, 2).unwrap(), saved_ip);
        assert_eq!(vm.mem.read(0x7ffe, 2).unwrap(), 0x100);
        assert_eq!(vm.mem.read(0x8000, 2).unwrap() & 0x200, 0x200);
    }

    #[test]
    fn real16_int_preserves_tier_and_partial_fault_fences() {
        use crate::interp::{interpret_block_ordered, BlockResult, RetireInfo};
        use crate::lift::{lift_one, CpuMode, FetchAddr};
        fn check<const ROUTED: bool, const TIER: u8>(partial: bool) {
            let vm = real16_fixture(MemConsistency::Fast, &[0xcd, 0x30], partial);
            let ir = lift_one(
                &vm.mem,
                FetchAddr::for_mode(CpuMode::Real16, 0, 0x100),
                CpuMode::Real16,
            )
            .unwrap();
            let mut cpu = real16_cpu();
            take_events();
            let result = interpret_block_ordered::<ROUTED, true, TIER>(
                &ir,
                &mut cpu,
                &vm.mem,
                &mut Vec::new(),
                &mut RetireInfo::default(),
            );
            assert_eq!(take_events(), interrupt_events(TIER, partial));
            if partial {
                assert!(matches!(
                    result.into_step(),
                    StepResult::Exit(Exit::UnmappedMemory { .. })
                ));
                assert_eq!(cpu.rip, 0);
                assert_eq!(vm.mem.read(0x8000, 2).unwrap() & 0x200, 0x200);
            } else {
                assert!(matches!(result, BlockResult::Continue));
                check_interrupt_frame(&vm, &cpu, 2);
            }
        }
        for partial in [false, true] {
            check::<false, 0>(partial);
            check::<false, 1>(partial);
            check::<false, 2>(partial);
            check::<true, 0>(partial);
            check::<true, 1>(partial);
            check::<true, 2>(partial);
        }
    }

    #[test]
    fn injected_irq_uses_vm_tier() {
        for tier in [MemConsistency::AcqRel, MemConsistency::FullTso] {
            for partial in [false, true] {
                for routed in [false, true] {
                    let mut vm = real16_fixture(tier, &[0xf4], partial);
                    vm.mem.set_host_fault_recovery(routed);
                    let mut vcpu = vm.new_vcpu();
                    vcpu.cpu = real16_cpu();
                    vcpu.inject_irq(0x30);
                    take_events();
                    let exit = vcpu.run(&vm, Some(1));
                    let order = if cfg!(target_arch = "aarch64") {
                        if tier == MemConsistency::AcqRel {
                            1
                        } else {
                            2
                        }
                    } else {
                        0
                    };
                    assert_eq!(take_events(), interrupt_events(order, partial));
                    assert!(!vcpu.has_pending_irq());
                    if partial {
                        assert!(matches!(exit, Exit::UnmappedMemory { .. }));
                    } else {
                        assert!(matches!(exit, Exit::BudgetExhausted));
                        check_interrupt_frame(&vm, &vcpu.cpu, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn vm_real16_interrupt_and_exception_entries_use_vm_tier() {
        for tier in [MemConsistency::AcqRel, MemConsistency::FullTso] {
            // INT 0x30 delivers in the walk; DIV CX with CX=0 redelivers via the VM.
            for (code, saved_ip) in [([0xcd, 0x30], 2), ([0xf7, 0xf1], 0)] {
                for step in [false, true] {
                    let vm = real16_fixture(tier, &code, false);
                    let mut vcpu = vm.new_vcpu();
                    vcpu.cpu = real16_cpu();
                    take_events();
                    if step {
                        assert!(matches!(vcpu.step_instruction(&vm), StepResult::Continue));
                    } else {
                        assert!(matches!(vcpu.run(&vm, Some(1)), Exit::BudgetExhausted));
                    }
                    let order = if cfg!(target_arch = "aarch64") {
                        if tier == MemConsistency::AcqRel {
                            1
                        } else {
                            2
                        }
                    } else {
                        0
                    };
                    assert_eq!(take_events(), interrupt_events(order, false));
                    check_interrupt_frame(&vm, &vcpu.cpu, saved_ip);
                }
            }
        }
    }

    #[test]
    fn run_compiled_mmio_defer_replay_preserves_tier() {
        use crate::jit_abi::{run_compiled, MemCtx, RET_HOST_FAULT_DETAIL, RET_MMIO_DEFER};
        // ABI stubs isolate replay so only interpreter fences are recorded.
        unsafe extern "C" fn defer(_cpu: *mut u8, _mem: *mut u8) -> u64 {
            RET_MMIO_DEFER
        }
        unsafe extern "C" fn alignment_replay(_cpu: *mut u8, mem: *mut u8) -> u64 {
            // SAFETY: run_compiled supplies a live MemCtx with the compiled ABI.
            unsafe {
                (*mem.cast::<MemCtx>()).link_slot = 0x100;
            }
            RET_HOST_FAULT_DETAIL
        }
        for tier in [MemConsistency::AcqRel, MemConsistency::FullTso] {
            for entry in [defer as *const u8, alignment_replay as *const u8] {
                for load in [false, true] {
                    let code = if load {
                        &[0x48, 0x8b, 0x03]
                    } else {
                        &[0x48, 0x89, 0x03]
                    };
                    let mut vm = fixture(tier, code);
                    for addr in [0x7000, 0x9000] {
                        vm.mem.unmap(addr, 0x1000).unwrap();
                        vm.map(addr, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                    }
                    vm.mem
                        .write(0x8000, if load { 0x42 } else { 0 }, 8)
                        .unwrap();
                    let mut cpu = crate::CpuState::new();
                    let entry = crate::CompiledPtr(entry);
                    let ordered = cfg!(target_arch = "aarch64");
                    let before = ordered && !load && tier == MemConsistency::AcqRel;
                    let after = ordered && (load || tier == MemConsistency::FullTso);
                    for trap in [false, true] {
                        cpu.rip = 0x1000;
                        cpu.gpr[0] = 0x42;
                        cpu.gpr[3] = if trap { 0x7000 } else { 0x8000 };
                        take_events();
                        // SAFETY: both stubs have the compiled ABI and remain live.
                        let result = unsafe {
                            run_compiled(
                                entry,
                                &mut cpu,
                                &vm.mem,
                                crate::CpuMode::Long64,
                                vm.consistency,
                            )
                        };
                        let mut expected = Vec::new();
                        if before {
                            expected.push("fence");
                        }
                        expected.push(if load { "read" } else { "write" });
                        if after {
                            expected.push("fence");
                        }
                        assert_eq!(take_events(), expected);
                        if trap {
                            assert!(matches!(
                                result,
                                StepResult::Exit(Exit::MmioRead { .. } | Exit::MmioWrite { .. })
                            ));
                            assert_eq!(cpu.rip, 0x1000);
                            cpu.pending_mmio = load.then_some(0x42);
                            cpu.pending_mmio_write = !load;
                            // SAFETY: same live ABI stub and memory as above.
                            let result = unsafe {
                                run_compiled(
                                    entry,
                                    &mut cpu,
                                    &vm.mem,
                                    crate::CpuMode::Long64,
                                    vm.consistency,
                                )
                            };
                            assert!(matches!(result, StepResult::Continue));
                            // A completed load consumes its answer without rereading memory.
                            if load {
                                expected.retain(|&event| event != "read");
                            }
                            assert_eq!(take_events(), expected);
                            assert!(cpu.pending_mmio.is_none());
                            assert!(!cpu.pending_mmio_write);
                        } else {
                            // The conservative window between Trap regions contains RAM.
                            assert!(matches!(result, StepResult::Continue));
                            assert_eq!(vm.mem.read(0x8000, 8).unwrap(), 0x42);
                        }
                        assert_eq!(cpu.rip, 0x1003);
                        if load {
                            assert_eq!(cpu.gpr[0], 0x42);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn walks_order_inline_vectors_rmw_and_helpers_on_every_host() {
        use crate::interp::{interpret_block_ordered, RetireInfo};
        use crate::lift::{lift_one, CpuMode, FetchAddr};
        use crate::{CpuState, Memory};

        // Counts pin helper direction independently of the implementation table.
        // SDM Vol 3A §11.2.2 and §11.2.4.1; the helper bracket matches the JIT.
        let cases: &[(&str, &[u8], usize, usize)] = &[
            ("load", &[0x48, 0x8b, 0x03], 1, 1),
            ("store", &[0x48, 0x89, 0x03], 1, 1),
            ("unlocked rmw", &[0x48, 0x01, 0x03], 2, 2),
            ("call", &[0xe8, 0, 0, 0, 0], 1, 1),
            ("ret", &[0xc3], 1, 1),
            ("push flags", &[0x9c], 1, 1),
            ("locked rmw", &[0xf0, 0x48, 0x01, 0x03], 0, 0),
            ("paired cas", &[0xf0, 0x48, 0x0f, 0xc7, 0x0b], 2, 2),
            ("locked cas", &[0xf0, 0x48, 0x0f, 0xb1, 0x03], 0, 0),
            ("vector load", &[0xf3, 0x0f, 0x6f, 0x03], 1, 1),
            ("vector store", &[0xf3, 0x0f, 0x7f, 0x03], 1, 1),
            ("wide load", &[0xc5, 0xfe, 0x6f, 0x03], 2, 2),
            ("wide store", &[0xc5, 0xfe, 0x7f, 0x03], 2, 2),
            ("string read", &[0xf3, 0xac], 1, 2),
            ("string write", &[0xf3, 0xaa], 1, 1),
            ("string read/write", &[0xf3, 0xa4], 2, 2),
            ("x87 register", &[0xd9, 0xe8], 0, 0),
            ("x87 read", &[0xdd, 0x03], 1, 2),
            ("x87 write", &[0xdd, 0x1b], 1, 1),
            ("fxsave", &[0x0f, 0xae, 0x03], 1, 1),
            ("fxrstor", &[0x0f, 0xae, 0x0b], 1, 2),
            ("xsavec", &[0x0f, 0xc7, 0x23], 1, 1),
            ("xsaveopt", &[0x0f, 0xae, 0x33], 2, 2),
            ("cvtps2ph store", &[0xc4, 0xe3, 0x79, 0x1d, 0x03, 0], 2, 2),
            ("xsave", &[0x0f, 0xae, 0x23], 2, 2),
            ("xrstor", &[0x0f, 0xae, 0x2b], 1, 2),
            ("masked read", &[0xc4, 0xe2, 0x79, 0x2c, 0x03], 1, 2),
            ("masked write", &[0xc4, 0xe2, 0x79, 0x2e, 0x03], 1, 1),
            ("gather", &[0xc4, 0xe2, 0x71, 0x90, 0x04, 0x13], 1, 2),
        ];
        fn check<const ROUTED: bool, const TIER: u8>(name: &str, code: &[u8], expected: usize) {
            let vm = fixture(MemConsistency::Fast, code);
            let mem: &Memory = &vm.mem;
            let ir = lift_one(mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
            let mut cpu = CpuState::new();
            cpu.gpr[3] = 0x8000;
            cpu.gpr[4] = 0x9000;
            cpu.gpr[6] = 0x8000;
            cpu.gpr[7] = 0x8100;
            cpu.gpr[1] = 2;
            cpu.xmm[0] = u128::MAX;
            cpu.xmm[1] = u128::MAX;
            take_events();
            let result = interpret_block_ordered::<ROUTED, true, TIER>(
                &ir,
                &mut cpu,
                mem,
                &mut Vec::new(),
                &mut RetireInfo::default(),
            );
            let result = result.into_step();
            assert!(
                matches!(result, StepResult::Continue),
                "{name}: unexpected execution exit"
            );
            let events = take_events();
            assert_eq!(
                events.iter().filter(|&&e| e == "fence").count(),
                expected,
                "{name}: {events:?}"
            );
        }
        for &(name, code, acqrel, full) in cases {
            check::<false, 0>(name, code, 0);
            check::<false, 1>(name, code, acqrel);
            check::<false, 2>(name, code, full);
            check::<true, 0>(name, code, 0);
            check::<true, 1>(name, code, acqrel);
            check::<true, 2>(name, code, full);
        }
    }
}
