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
