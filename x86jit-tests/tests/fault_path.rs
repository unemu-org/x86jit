//! Fault restart: SDM Vol 3A §7.5–7.6. REP restart: SDM Vol 2B, REP, Description/Operation.
//! Authored fixtures use real host protection, independently of advisory guest Prot.
#![cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
use iced_x86::code_asm::*;
use std::sync::atomic::Ordering;
use x86jit_core::{
    memory::HostRam, Backend, CpuState, Exit, FaultHelper, FaultMetadata, FaultPath, FetchVerdict,
    ForceRegionError, HelperCaller, InterpreterBackend, Knowledge::Known, MemoryEffects, Prot,
    RegionCaps, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

#[path = "../../x86jit-cranelift/src/unwind_test_signal.rs"]
mod signal;

const BASE: u64 = 0x4000_0000;
const CAPS: RegionCaps = RegionCaps {
    max_blocks: 16,
    max_icount: 256,
};

struct Fixture {
    vm: Vm,
    page: usize,
    host: *mut u8,
    base: u64,
    initial_data: Vec<u8>,
    initial_scratch: Vec<u8>,
}
impl Fixture {
    fn new(tier: usize, base: u64) -> Self {
        // SAFETY: the child owns the mapping until Vm's HostRam destructor unmaps it.
        unsafe {
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
            let backend: Box<dyn Backend> = if tier == 0 {
                Box::new(InterpreterBackend)
            } else if tier == 2 {
                Box::new(JitBackend::with_superblocks(CAPS))
            } else {
                Box::new(JitBackend::new())
            };
            let mut vm = Vm::with_backend_host_ram(
                VmConfig::flat(base + len as u64),
                backend,
                HostRam {
                    ptr: ptr.cast(),
                    len,
                    guest_base: base,
                    protect: None,
                    dtor: Box::new(|p, n| {
                        libc::munmap(p.cast(), n);
                    }),
                },
            )
            .unwrap();
            vm.map(base, len, Prot::RWX, RegionKind::Ram).unwrap();
            vm.mem.set_host_fault_recovery(true);
            Self {
                vm,
                page,
                host: ptr.cast(),
                base,
                initial_data: vec![0; page],
                initial_scratch: vec![0; page],
            }
        }
    }
    fn hole(&self) -> u64 {
        self.base + (2 * self.page) as u64
    }
    fn scratch(&self) -> u64 {
        self.base + (3 * self.page) as u64
    }
    fn protect(&self, protection: i32) {
        // SAFETY: the third page belongs to this fixture and is host-page aligned.
        unsafe {
            assert_eq!(
                libc::mprotect(self.host.add(2 * self.page).cast(), self.page, protection),
                0
            );
        }
    }
    fn prepare(&mut self, tier: usize, entry: u64) {
        self.vm
            .read_bytes(self.base + self.page as u64, &mut self.initial_data)
            .unwrap();
        self.vm
            .read_bytes(self.scratch(), &mut self.initial_scratch)
            .unwrap();
        if tier == 2 {
            // Deliberately contradict automatic tiering: the test hook must override it.
            self.vm.set_tier_up_after(Some(u32::MAX));
            self.vm.set_tier_up_background(true);
            self.vm
                .test_support_force_region(entry, CAPS, 0x1f80)
                .unwrap();
        }
    }
    fn run(&self, cpu: &mut x86jit_core::Vcpu) -> Exit {
        signal::HANDLE.store(cpu.fault_handle().as_usize(), Ordering::Relaxed);
        // SAFETY: isolated child with a live recovery handle throughout execution.
        unsafe {
            signal::install();
        }
        cpu.run(&self.vm, Some(100))
    }
}
fn seed(cpu: &mut x86jit_core::Vcpu, entry: u64, operand: u64) {
    cpu.cpu.gpr = std::array::from_fn(|i| 0x1010 + i as u64 * 0x100);
    cpu.cpu.gpr[6] = operand;
    cpu.cpu.rip = entry;
    cpu.cpu.xmm = std::array::from_fn(|i| 0x1234_5678_9abc_def0 + i as u128);
    cpu.cpu.ymm_hi = std::array::from_fn(|i| 0x2234_5678_9abc_def0 + i as u128);
    cpu.cpu.zmm_hi = std::array::from_fn(|i| [0x3344 + i as u128, 0x5566 + i as u128]);
    cpu.cpu.kmask = std::array::from_fn(|i| 0x7890 + i as u64);
    cpu.cpu.fpr = std::array::from_fn(|i| {
        let mut value = [0; 10];
        value[..8].copy_from_slice(&(0x8000_0000_0000_0000 + i as u64).to_le_bytes());
        value[8..].copy_from_slice(&0x3fffu16.to_le_bytes());
        value
    });
    cpu.cpu.flags.cf = true;
    cpu.cpu.flags.set_af(true);
    cpu.cpu.flags.set_pf(false);
}
fn state(actual: &CpuState, expected: &CpuState) {
    assert_eq!(actual.gpr, expected.gpr);
    assert_eq!(actual.rip, expected.rip);
    assert_eq!(actual.rflags(), expected.rflags());
    assert_eq!(actual.xmm, expected.xmm);
    assert_eq!(actual.ymm_hi, expected.ymm_hi);
    assert_eq!(actual.zmm_hi, expected.zmm_hi);
    assert_eq!(actual.kmask, expected.kmask);
    assert_eq!(actual.mxcsr, expected.mxcsr);
    assert_eq!(actual.fpr, expected.fpr);
    assert_eq!(actual.fpu_top, expected.fpu_top);
    assert_eq!(actual.fpu_cw, expected.fpu_cw);
    assert_eq!(actual.fpu_sw, expected.fpu_sw);
    assert_eq!(actual.fpu_empty, expected.fpu_empty);
    assert_eq!(actual.fpu_fop, expected.fpu_fop);
    assert_eq!(actual.fpu_fip, expected.fpu_fip);
    assert_eq!(actual.fpu_fdp, expected.fpu_fdp);
}
fn path(exit: &Exit, tier: usize, entry: u64, helper: Option<FaultHelper>, exact: bool) {
    let m = exit.fault_metadata().expect("fault evidence");
    match (tier, helper, m.path()) {
        (0, None, FaultPath::Interpreter) => {}
        (
            0,
            Some(want),
            FaultPath::Helper {
                family,
                caller: HelperCaller::Interpreter,
            },
        ) => assert_eq!(family, want),
        (1, None, FaultPath::CompiledBlock { host_pc, site }) => {
            assert!(
                matches!(m.instruction_rip(), Known(rip) if x86jit_core::codemap::lookup_fault_info(host_pc, rip).is_some())
            );
            if exact {
                assert_eq!(site, x86jit_core::codemap::lookup_trap(host_pc));
                assert_eq!(Known(site.unwrap().guest_rip), m.instruction_rip());
            }
        }
        (
            2,
            None,
            FaultPath::Region {
                entry_rip,
                host_pc,
                site,
            },
        ) => {
            assert_eq!(entry_rip, entry);
            assert!(
                matches!(m.instruction_rip(), Known(rip) if x86jit_core::codemap::lookup_fault_info(host_pc, rip).is_some())
            );
            if exact {
                assert_eq!(site, x86jit_core::codemap::lookup_trap(host_pc));
                assert_eq!(Known(site.unwrap().guest_rip), m.instruction_rip());
            }
        }
        (
            1,
            Some(want),
            FaultPath::Helper {
                family,
                caller: HelperCaller::CompiledBlock { host_pc },
            },
        ) => {
            assert_eq!(family, want);
            assert!(
                matches!(m.instruction_rip(), Known(rip) if x86jit_core::codemap::lookup_fault_info(host_pc, rip).is_some())
            );
        }
        (
            2,
            Some(want),
            FaultPath::Helper {
                family,
                caller: HelperCaller::Region { entry_rip, host_pc },
            },
        ) => {
            assert_eq!(family, want);
            assert_eq!(entry_rip, entry);
            assert!(
                matches!(m.instruction_rip(), Known(rip) if x86jit_core::codemap::lookup_fault_info(host_pc, rip).is_some())
            );
        }
        other => panic!("wrong execution path: {other:?}; {exit:?}"),
    }
}
fn child(f: impl FnOnce()) {
    assert_eq!(signal::child(f), 0);
}

#[test]
fn path_x87_helper_preserves_seeded_state() {
    for tier in 0..3 {
        child(|| {
            let mut f = Fixture::new(tier, BASE);
            // SDM Vol 2A, FLD, Description/64-Bit Mode Exceptions; Vol 3A §7.5: #PF abandons the push.
            let code = [0xdb, 0x2e, 0xc3]; // fld tbyte [rsi]; ret
            f.vm.write_bytes(BASE, &code).unwrap();
            f.prepare(tier, BASE);
            let mut cpu = f.vm.new_vcpu();
            seed(&mut cpu, BASE, f.hole());
            let expected = cpu.cpu.clone();
            f.protect(libc::PROT_NONE);
            let exit = f.run(&mut cpu);
            path(&exit, tier, BASE, Some(FaultHelper::X87), false);
            state(&cpu.cpu, &expected);
            assert_eq!(
                exit.fault_metadata().unwrap().instruction_rip(),
                Known(BASE)
            );
            native(&f, &code, &expected, &cpu.cpu, f.hole());
        });
    }
}

#[test]
fn path_rep_helper_keeps_partial_progress() {
    for tier in 0..3 {
        for source in [false, true] {
            child(|| {
                let mut f = Fixture::new(tier, BASE);
                let code = [0xf3, 0xa4, 0xc3]; // rep movsb; ret
                f.vm.write_bytes(BASE, &code).unwrap();
                f.vm.write_bytes(
                    f.hole() - 3,
                    if source {
                        &[0x51, 0x52, 0x53]
                    } else {
                        &[0x35; 3]
                    },
                )
                .unwrap();
                f.vm.write_bytes(
                    f.scratch(),
                    if source {
                        &[0x35; 5]
                    } else {
                        &[0x51, 0x52, 0x53, 0x54, 0x55]
                    },
                )
                .unwrap();
                f.prepare(tier, BASE);
                let mut cpu = f.vm.new_vcpu();
                seed(
                    &mut cpu,
                    BASE,
                    if source { f.hole() - 3 } else { f.scratch() },
                );
                cpu.cpu.gpr[7] = if source { f.scratch() } else { f.hole() - 3 };
                cpu.cpu.gpr[1] = 5;
                let initial = cpu.cpu.clone();
                let mut expected = initial.clone();
                // SDM Vol 2B, REP, Description: pointers/count describe the next uncompleted element.
                expected.gpr[6] += 3;
                expected.gpr[7] += 3;
                expected.gpr[1] = 2;
                f.protect(libc::PROT_NONE);
                let exit = f.run(&mut cpu);
                path(&exit, tier, BASE, Some(FaultHelper::String), false);
                state(&cpu.cpu, &expected);
                let progress = exit.fault_metadata().unwrap().progress().rep().unwrap();
                assert_eq!(progress.remaining(), 2);
                let mut bytes = [0; 3];
                f.vm.read_bytes(initial.gpr[7], &mut bytes).unwrap();
                assert_eq!(bytes, [0x51, 0x52, 0x53]);
                native(&f, &code, &initial, &cpu.cpu, f.hole());
            });
        }
    }
}

#[test]
fn path_fxstate_helper_and_preceding_alignment_check() {
    for tier in 0..3 {
        for misaligned in [false, true] {
            child(|| {
                let mut f = Fixture::new(tier, BASE);
                // SDM Vol 2A, FXRSTOR, 64-Bit Mode Exceptions: #GP on misalignment,
                // #PF on an unavailable save-area byte. Failed restore commits no state.
                f.vm.write_bytes(BASE, &[0x48, 0x0f, 0xae, 0x0e, 0xc3])
                    .unwrap();
                f.prepare(tier, BASE);
                let mut cpu = f.vm.new_vcpu();
                seed(&mut cpu, BASE, f.hole() + u64::from(misaligned));
                let expected = cpu.cpu.clone();
                f.protect(libc::PROT_NONE);
                let exit = f.run(&mut cpu);
                path(
                    &exit,
                    tier,
                    BASE,
                    if misaligned {
                        None
                    } else {
                        Some(FaultHelper::FxState)
                    },
                    false,
                );
                state(&cpu.cpu, &expected);
            });
        }
    }
}

#[test]
fn path_pair_helper_keeps_fault_and_alignment_witnesses() {
    for tier in 0..3 {
        for misaligned in [false, true] {
            child(|| {
                let mut f = Fixture::new(tier, BASE);
                // SDM Vol 2A, CMPXCHG8B/CMPXCHG16B, 64-Bit Mode Exceptions:
                // #GP for an unaligned m128; an inaccessible operand raises #PF.
                f.vm.write_bytes(BASE, &[0x48, 0x0f, 0xc7, 0x0e, 0xc3])
                    .unwrap();
                f.prepare(tier, BASE);
                let mut cpu = f.vm.new_vcpu();
                seed(&mut cpu, BASE, f.hole() + u64::from(misaligned));
                let expected = cpu.cpu.clone();
                f.protect(libc::PROT_NONE);
                let exit = f.run(&mut cpu);
                path(
                    &exit,
                    tier,
                    BASE,
                    Some(FaultHelper::CompareExchangePair),
                    false,
                );
                state(&cpu.cpu, &expected);
            });
        }
    }
}

#[test]
fn path_divide_distinguishes_operand_load_from_helper() {
    for tier in 0..3 {
        for operand_fault in [false, true] {
            child(|| {
                let mut f = Fixture::new(tier, BASE);
                // SDM Vol 2A, DIV, Operation/64-Bit Mode Exceptions: a source load may
                // fault before division; a zero divisor raises #DE without committing.
                f.vm.write_bytes(BASE, &[0x48, 0xf7, 0x36, 0xc3]).unwrap();
                f.prepare(tier, BASE);
                let mut cpu = f.vm.new_vcpu();
                seed(
                    &mut cpu,
                    BASE,
                    if operand_fault { f.hole() } else { f.scratch() },
                );
                let expected = cpu.cpu.clone();
                f.protect(libc::PROT_NONE);
                let exit = f.run(&mut cpu);
                path(
                    &exit,
                    tier,
                    BASE,
                    if operand_fault {
                        None
                    } else {
                        Some(FaultHelper::Divide)
                    },
                    operand_fault && tier != 0,
                );
                state(&cpu.cpu, &expected);
            });
        }
    }
}

#[test]
fn path_fetch_crossing_and_high_rip() {
    for tier in 0..3 {
        for base in [BASE, 0x1_4000_0000] {
            for host_fault in [false, true] {
                child(|| {
                    let mut f = Fixture::new(tier, base);
                    let denied = if host_fault {
                        f.hole()
                    } else {
                        base + f.page as u64
                    };
                    let entry = denied - 1;
                    f.vm.write_bytes(entry, &[0x48, 0x8b, 0x06]).unwrap();
                    if host_fault {
                        f.protect(libc::PROT_NONE);
                    } else {
                        f.vm.set_fetch_policy(Box::new(move |page| {
                            if page == denied {
                                FetchVerdict::DenyNotPresent
                            } else {
                                FetchVerdict::Allow
                            }
                        }));
                    }
                    let mut cpu = f.vm.new_vcpu();
                    seed(&mut cpu, entry, f.scratch());
                    let expected = cpu.cpu.clone();
                    signal::HANDLE.store(cpu.fault_handle().as_usize(), Ordering::Relaxed);
                    // SAFETY: isolated child; transfer fixups also recover outside a run.
                    unsafe {
                        signal::install();
                    }
                    let exit = if tier == 2 {
                        match f.vm.test_support_force_region(entry, CAPS, 0x1f80) {
                            Err(ForceRegionError::Lift(exit)) => exit,
                            other => panic!("fetch refusal must prevent compilation: {other:?}"),
                        }
                    } else {
                        f.run(&mut cpu)
                    };
                    // SDM Vol 3A §7.15, Event 14: RIP names the prefix, address the denied byte.
                    let m = exit.fault_metadata().unwrap();
                    assert_eq!(
                        m.path(),
                        FaultPath::LifterFetch {
                            first_unavailable: denied
                        }
                    );
                    assert_eq!(m.instruction_rip(), Known(entry));
                    assert!(matches!(m.linear_address(), Known(a) if a.value() == denied));
                    state(&cpu.cpu, &expected);
                });
            }
        }
    }
}

#[test]
fn path_dead_flags_load_and_rmw() {
    for tier in 0..3 {
        for rmw in [false, true] {
            child(|| {
                let mut f = Fixture::new(tier, BASE);
                let mut a = CodeAssembler::new(64).unwrap();
                a.mov(eax, 0x7fff_ffffu32).unwrap();
                a.add(eax, 1).unwrap();
                let offset = a.assemble(BASE).unwrap().len();
                if rmw {
                    a.add(qword_ptr(rsi), rbx).unwrap();
                } else {
                    a.mov(rdx, qword_ptr(rsi)).unwrap();
                }
                a.cmp(eax, 0).unwrap();
                a.ret().unwrap();
                let code = a.assemble(BASE).unwrap();
                f.vm.write_bytes(BASE, &code).unwrap();
                f.prepare(tier, BASE);
                let mut cpu = f.vm.new_vcpu();
                seed(&mut cpu, BASE, f.hole());
                let initial = cpu.cpu.clone();
                let mut expected = initial.clone();
                expected.rip += offset as u64;
                expected.gpr[0] = 0x8000_0000;
                // SDM Vol 2A, ADD, Flags Affected; Vol 1 §3.4.3.
                expected.flags.cf = false;
                expected.flags.of = true;
                expected.flags.sf = true;
                expected.flags.zf = false;
                expected.flags.set_af(true);
                expected.flags.set_pf(true);
                f.protect(libc::PROT_NONE);
                let exit = f.run(&mut cpu);
                path(&exit, tier, BASE, None, tier != 0);
                state(&cpu.cpu, &expected);
                native(&f, &code, &initial, &cpu.cpu, f.hole());
            });
        }
    }
}

#[test]
fn path_memory_units_and_split_store_policy() {
    for tier in 0..3 {
        for kind in ["load", "store", "rmw", "vector", "split"] {
            for precise in [false, true] {
                child(|| {
                    let mut f = Fixture::new(tier, BASE);
                    f.vm.set_precise_split_stores(precise);
                    let code: &[u8] = match kind {
                        "load" => &[0x48, 0x8b, 0x06, 0xc3],
                        "store" => &[0x48, 0x89, 0x06, 0xc3],
                        "rmw" => &[0x48, 0x01, 0x1e, 0xc3],
                        "vector" => &[0xf3, 0x0f, 0x6f, 0x06, 0xc3],
                        "split" => &[0xf3, 0x0f, 0x7f, 0x06, 0xc3],
                        _ => unreachable!(),
                    };
                    f.vm.write_bytes(BASE, code).unwrap();
                    f.vm.write_bytes(f.hole() - 8, &[0x35; 8]).unwrap();
                    f.prepare(tier, BASE);
                    let mut cpu = f.vm.new_vcpu();
                    seed(
                        &mut cpu,
                        BASE,
                        f.hole() - if kind == "split" { 8 } else { 0 },
                    );
                    let expected = cpu.cpu.clone();
                    f.protect(libc::PROT_NONE);
                    let exit = f.run(&mut cpu);
                    let checked_preflight = tier != 0 && kind == "split" && precise;
                    path(&exit, tier, BASE, None, tier != 0 && !checked_preflight);
                    state(&cpu.cpu, &expected);
                    if checked_preflight {
                        // Decision-14: the shared preflight returns a checked exit;
                        // its fixup PC is outside the compiled unit's transfer sites.
                        let m = exit.fault_metadata().unwrap();
                        assert!(matches!(
                            m.path(),
                            FaultPath::CompiledBlock { site: None, .. }
                                | FaultPath::Region { site: None, .. }
                        ));
                        assert_eq!(m.operand_size(), Known(16));
                        assert_eq!(m.transfer_size(), Known(1));
                        assert_eq!(m.transfer_access(), Known(x86jit_core::FaultAccess::Write));
                        assert_eq!(m.progress().ram(), MemoryEffects::None);
                    }
                    if kind == "split" {
                        let mut visible = [0; 8];
                        f.vm.read_bytes(f.hole() - 8, &mut visible).unwrap();
                        if tier != 0 && !precise {
                            // Engine split-store contract: completed pieces may remain visible.
                            assert_eq!(
                                exit.fault_metadata().unwrap().progress().ram(),
                                MemoryEffects::PossiblyPartial
                            );
                            assert!(
                                visible == [0x35; 8]
                                    || visible == cpu.cpu.xmm[0].to_le_bytes()[..8]
                            );
                        } else {
                            assert_eq!(visible, [0x35; 8]);
                            assert_eq!(
                                exit.fault_metadata().unwrap().progress().ram(),
                                if tier == 0 {
                                    MemoryEffects::PossiblyPartial
                                } else {
                                    MemoryEffects::None
                                }
                            );
                        }
                    } else {
                        native(&f, code, &expected, &cpu.cpu, f.hole());
                    }
                });
            }
        }
    }
}

#[test]
fn path_forced_region_loop_fault_iteration() {
    for tier in 0..3 {
        child(|| {
            let mut f = Fixture::new(tier, BASE);
            let mut a = CodeAssembler::new(64).unwrap();
            let mut top = a.create_label();
            a.set_label(&mut top).unwrap();
            a.add(rax, rbx).unwrap();
            a.xor(rdx, rax).unwrap();
            a.add(r8, 3).unwrap();
            let fault_offset = a.assemble(BASE).unwrap().len();
            a.mov(qword_ptr(rsi), rax).unwrap();
            a.add(rsi, 8).unwrap();
            a.dec(ecx).unwrap();
            a.jnz(top).unwrap();
            a.ret().unwrap();
            let code = a.assemble(BASE).unwrap();
            f.vm.write_bytes(BASE, &code).unwrap();
            f.prepare(tier, BASE);
            let mut cpu = f.vm.new_vcpu();
            seed(&mut cpu, BASE, f.hole() - 20 * 8);
            cpu.cpu.gpr[1] = 32;
            cpu.cpu.gpr[8] = 0x100;
            let initial = cpu.cpu.clone();
            let mut expected = initial.clone();
            for _ in 0..21 {
                expected.gpr[0] += expected.gpr[3];
                expected.gpr[2] ^= expected.gpr[0];
                expected.gpr[8] += 3;
            }
            expected.gpr[6] = f.hole();
            expected.gpr[1] = 12;
            expected.rip = BASE + fault_offset as u64;
            expected.flags.cf = false;
            expected.flags.zf = false;
            expected.flags.sf = false;
            expected.flags.of = false;
            expected.flags.set_af(false);
            expected.flags.set_pf(true); // 0x13f: six set bits in the low byte.
            f.protect(libc::PROT_NONE);
            let exit = f.run(&mut cpu);
            path(&exit, tier, BASE, None, tier != 0);
            state(&cpu.cpu, &expected);
            let mut completed = [0; 160];
            f.vm.read_bytes(initial.gpr[6], &mut completed).unwrap();
            for (i, bytes) in completed.as_chunks::<8>().0.iter().enumerate() {
                assert_eq!(
                    u64::from_le_bytes(*bytes),
                    initial.gpr[0] + (i as u64 + 1) * initial.gpr[3]
                );
            }
            native(&f, &code, &initial, &cpu.cpu, f.hole());
        });
    }
}

#[test]
fn path_force_region_replaces_cached_block_and_obeys_invalidation() {
    child(|| {
        let mut f = Fixture::new(2, BASE);
        f.vm.write_bytes(BASE, &[0x48, 0x8b, 0x06, 0xf4]).unwrap();
        let mut cpu = f.vm.new_vcpu_with_mxcsr(0x9f80);
        seed(&mut cpu, BASE, f.hole());
        f.protect(libc::PROT_NONE);
        let block = f.run(&mut cpu);
        // Automatic region selection rejects this single-block, non-loop fixture.
        path(&block, 1, BASE, None, true);
        f.vm.set_tier_up_after(Some(u32::MAX));
        f.vm.set_tier_up_background(true);
        f.vm.test_support_force_region(BASE, CAPS, cpu.cpu.mxcsr)
            .unwrap();
        let region = f.run(&mut cpu);
        path(&region, 2, BASE, None, true);
        let owned = region.fault_metadata().unwrap().clone();
        f.vm.write_bytes(BASE, &[0xf4]).unwrap();
        assert!(matches!(f.run(&mut cpu), Exit::Hlt));
        // Owned evidence survives retirement; the live code-map range may be gone.
        let retained = region.fault_metadata().unwrap();
        assert_eq!(retained.path(), owned.path());
        assert_eq!(retained.instruction_rip(), owned.instruction_rip());
    });
}

#[test]
fn path_owned_exit_survives_success_and_force_region_rejects_fallback() {
    assert_eq!(FaultMetadata::unknown(BASE).path(), FaultPath::Unknown);
    let mut vm = Vm::new(VmConfig::flat(0x2000));
    vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0x0f, 0x0b]).unwrap();
    assert!(matches!(
        vm.test_support_force_region(0x1000, CAPS, 0x1f80),
        Err(ForceRegionError::NotCompiled)
    ));
    for tier in 0..3 {
        child(|| {
            let mut f = Fixture::new(tier, BASE);
            f.vm.write_bytes(BASE, &[0x48, 0x8b, 0x06, 0xf4]).unwrap();
            f.prepare(tier, BASE);
            let mut cpu = f.vm.new_vcpu();
            seed(&mut cpu, BASE, f.hole());
            f.protect(libc::PROT_NONE);
            let exit = f.run(&mut cpu);
            path(&exit, tier, BASE, None, tier != 0);
            let owned = exit.clone();
            let witness = owned.fault_metadata().unwrap().path();
            f.protect(libc::PROT_READ | libc::PROT_WRITE);
            assert!(matches!(f.run(&mut cpu), Exit::Hlt));
            assert_eq!(owned.fault_metadata().unwrap().path(), witness);
        });
    }
}

// The existing native oracle captures GPRs, arithmetic flags, x87 data/TOP/CW,
// XMM and YMM. ZMM/opmask and x87 environment fields are asserted above on both
// tiers; they are outside that oracle's input/capture format.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn native(f: &Fixture, code: &[u8], initial: &CpuState, actual: &CpuState, fault: u64) {
    use x86jit_tests::{
        native::{run_native_fault, NativeFault},
        oracle::VectorInput,
        vector::{CpuSnapshot, MemChunk, MemKind, RunSpec, SnapFlags},
    };
    let mut init = CpuSnapshot {
        gpr: initial.gpr,
        rip: initial.rip,
        flags: initial.flags.into(),
        xmm: initial.xmm,
        ymm_hi: initial.ymm_hi,
        st: std::array::from_fn(|i| initial.fpr[(initial.fpu_top as usize + i) & 7]),
        fpu_cw: initial.fpu_cw,
        fpu_top: initial.fpu_top as u8,
        mxcsr: initial.mxcsr,
        ..CpuSnapshot::default()
    };
    let avx = std::is_x86_feature_detected!("avx");
    if !avx {
        init.ymm_hi.fill(0);
    }
    init.xmm[16..].fill(0);
    init.ymm_hi[16..].fill(0);
    // Native's mappings are independent of HostRam's backing addresses.
    let input = VectorInput {
        cpu_init: init,
        entry: initial.rip,
        run: RunSpec::UntilExit,
        mem_init: vec![
            MemChunk {
                addr: f.base,
                bytes: code.to_vec(),
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: f.base + f.page as u64,
                bytes: f.initial_data.clone(),
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: f.scratch(),
                bytes: f.initial_scratch.clone(),
                kind: MemKind::Ram,
            },
        ],
    };
    let out = run_native_fault(&input, NativeFault::PageFault { addr: fault })
        .expect("native fault capture");
    for chunk in &out.mem[1..] {
        let mut bytes = vec![0; chunk.bytes.len()];
        f.vm.read_bytes(chunk.addr, &mut bytes).unwrap();
        assert_eq!(bytes, chunk.bytes);
    }
    assert_eq!(out.cpu.gpr, actual.gpr);
    assert_eq!(out.cpu.rip, actual.rip);
    assert_eq!(out.cpu.flags, SnapFlags::from(actual.flags));
    assert_eq!(&out.cpu.xmm[..16], &actual.xmm[..16]);
    if avx {
        assert_eq!(&out.cpu.ymm_hi[..16], &actual.ymm_hi[..16]);
    }
    assert_eq!(
        out.cpu.st,
        std::array::from_fn(|i| actual.fpr[(actual.fpu_top as usize + i) & 7])
    );
    assert_eq!(u32::from(out.cpu.fpu_top), actual.fpu_top);
    assert_eq!(out.cpu.fpu_cw, actual.fpu_cw);
    assert_eq!(out.cpu.mxcsr, actual.mxcsr);
}
#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn native(_: &Fixture, _: &[u8], _: &CpuState, _: &CpuState, _: u64) {}

#[test]
fn live_fault_classification_survives_concurrent_retirement_and_address_reuse() {
    child(|| {
        use std::sync::{
            atomic::{AtomicBool, AtomicUsize},
            Arc, Barrier,
        };
        use std::time::{Duration, Instant};
        let mut f = Fixture::new(1, BASE);
        // SDM Vol 2B, MOV: a faulting source load leaves the destination unchanged.
        f.vm.write_bytes(BASE, &[0x48, 0x8b, 0x06, 0xf4]).unwrap();
        f.prepare(1, BASE);
        f.protect(libc::PROT_NONE);
        let mut cpu = f.vm.new_vcpu();
        seed(&mut cpu, BASE, f.hole());
        let expected = cpu.cpu.clone();
        let first = f.run(&mut cpu);
        path(&first, 1, BASE, None, true);

        let ready = Arc::new(Barrier::new(2));
        let done = Arc::new(AtomicBool::new(false));
        let retired = Arc::new(AtomicUsize::new(0));
        let worker = {
            let ready = ready.clone();
            let done = done.clone();
            let retired = retired.clone();
            std::thread::spawn(move || {
                let jit = JitBackend::new();
                let mut memory =
                    x86jit_core::Memory::new(x86jit_core::MemoryModel::Flat { size: 0x2000 });
                memory.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
                memory
                    .write_bytes(0x1000, &[0xb8, 7, 0, 0, 0, 0xf4])
                    .unwrap();
                let ir = x86jit_core::lift::lift_block(
                    &memory,
                    x86jit_core::lift::FetchAddr::flat(0x1000),
                    x86jit_core::CpuMode::Long64,
                )
                .unwrap();
                let domain = jit.code_lifetime().unwrap();
                ready.wait();
                for _ in 0..16 {
                    let block = jit.materialize(&ir, x86jit_core::MemConsistency::Fast, None, 0);
                    assert!(matches!(block, x86jit_core::CachedBlock::Compiled { .. }));
                    domain.discard(&block);
                    retired.fetch_add(1, Ordering::Release);
                    let deadline = Instant::now() + Duration::from_secs(5);
                    while !domain.retired_units().is_empty() {
                        assert!(Instant::now() < deadline);
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
                done.store(true, Ordering::Release);
            })
        };
        ready.wait();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut faults = 0;
        while !done.load(Ordering::Acquire) {
            assert!(Instant::now() < deadline);
            cpu.cpu = expected.clone();
            let exit = f.run(&mut cpu);
            path(&exit, 1, BASE, None, true);
            state(&cpu.cpu, &expected);
            faults += 1;
            std::thread::sleep(Duration::from_millis(1));
        }
        worker.join().unwrap();
        assert_eq!(retired.load(Ordering::Acquire), 16);
        assert!(faults > 0);
    });
}
