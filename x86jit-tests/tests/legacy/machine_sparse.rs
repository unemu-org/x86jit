//! Decision-33: compiled Machine blocks on sparse backing.
//!
//! A Machine VM whose backing is a 4 GiB sparse reservation compiles its
//! Flat32 blocks as an ordinary one does. The J1, J2 and J3+J4 programs run
//! on such a VM in lockstep with the interpreter on the same backing, and
//! accesses to holes (uncommitted host pages, frames that share a committed
//! host page with a region, operands that cross from a region into a hole, a
//! region unmapped between runs) give the interpreter's exit or fault, never
//! a host fault. Application VMs on sparse backing stay interpreted.
use super::machine_j1_lockstep::{
    add, answer, asm, families, machine, programs, same, same_memory, Setup, Subject, MMIO, MODELS,
    PAGE_TABLE, SIZE, SPANS, SPARSE_EDGE, SPARSE_ROM,
};
use super::machine_j34_paged::{lockstep, plain, PAGED, PAGED_USER};
use iced_x86::code_asm::*;
use x86jit_core::{
    Exit, MachineJitCounters, MachineJitMode, Prot, RegionKind, StepOptions, StepStop, Vcpu, Vm,
    VmConfig,
};
use x86jit_cranelift::JitBackend;

fn sparse(setup: Setup) -> Setup {
    Setup {
        sparse_compile: true,
        ..setup
    }
}

fn cause(name: &str) -> usize {
    MachineJitCounters::DEFER_CAUSE_NAMES
        .iter()
        .position(|n| *n == name)
        .unwrap()
}

/// The J1 programs and families, unpaged and paged at CPL 0 and CPL 3, on
/// sparse backing, in the test modes and the production configuration.
#[test]
fn compiled_programs_on_sparse_backing_equal_the_interpreter() {
    for model in MODELS {
        let mut total = MachineJitCounters::default();
        let mut paged = MachineJitCounters::default();
        for (index, (name, code, extra)) in programs(model).into_iter().enumerate() {
            let setups = [
                ("unpaged", sparse(extra)),
                (
                    "paged",
                    sparse(Setup {
                        a20_off: extra.a20_off,
                        watch: extra.watch,
                        ..PAGED
                    }),
                ),
                (
                    "paged-cpl3",
                    sparse(Setup {
                        a20_off: extra.a20_off,
                        watch: extra.watch,
                        ..PAGED_USER
                    }),
                ),
            ];
            for (kind, setup) in setups {
                for subject in [
                    Subject::Mode(MachineJitMode::Forced),
                    Subject::Mode(MachineJitMode::Delayed(1)),
                    Subject::Background(1),
                ] {
                    for budget in [1u64, 7, 64] {
                        let what = format!("{name} {kind} {model:?} {subject:?} budget={budget}");
                        let (c, _) = lockstep(
                            model,
                            &code,
                            subject,
                            setup,
                            index as u64 + 1,
                            &plain,
                            &mut |_| budget,
                            &what,
                        );
                        if setup.paged {
                            add(&mut paged, c);
                        } else {
                            add(&mut total, c);
                        }
                    }
                }
            }
        }
        for c in [total, paged] {
            assert!(c.entries > 0, "{model:?}: {c:?}");
            assert!(c.compiled_retired > 5_000, "{model:?}: {c:?}");
            assert!(c.helper_defers > 0, "{model:?}: {c:?}");
        }
        assert!(paged.paged_entries > 0, "{model:?}: {paged:?}");
        assert!(paged.compiled_retired_cpl3 > 0, "{model:?}: {paged:?}");
    }
}

/// Every op family compiles on sparse backing: only the final HLT stays
/// interpreted.
#[test]
fn every_lowered_family_runs_compiled_on_sparse_backing() {
    for model in MODELS {
        for (name, code) in families(model) {
            // 16-bit SHLD/SHRD stay deferred (decision-30 §3).
            if name == "double-shift-16" {
                continue;
            }
            let (c, _) = lockstep(
                model,
                &code,
                Subject::Mode(MachineJitMode::Forced),
                sparse(Setup::default()),
                3,
                &plain,
                &mut |_| 1000,
                name,
            );
            assert_eq!(c.unconditional_defers, 0, "{name} {model:?}: {c:?}");
            assert!(c.continues >= 5, "{name} {model:?}: {c:?}");
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Access {
    Load,
    Store,
    Rmw,
}

/// `mov edi, 40; mov esi, 6000h; L: <access [esi]>; add ebx, eax; cmp edi,
/// 2; jne S; mov esi, target; S: dec edi; jnz L; hlt`: the hot loop's last
/// iteration accesses `target` from the compiled loop block.
fn hot_then(access: Access, width: u8, target: u32) -> Vec<u8> {
    asm(|a| {
        let mut top = a.create_label();
        let mut skip = a.create_label();
        a.mov(edi, 40u32).unwrap();
        a.mov(esi, 0x6000u32).unwrap();
        a.set_label(&mut top).unwrap();
        match (access, width) {
            (Access::Load, 1) => a.movzx(eax, byte_ptr(esi)).unwrap(),
            (Access::Load, 2) => a.movzx(eax, word_ptr(esi)).unwrap(),
            (Access::Load, _) => a.mov(eax, dword_ptr(esi)).unwrap(),
            (Access::Store, 1) => a.mov(byte_ptr(esi), al).unwrap(),
            (Access::Store, 2) => a.mov(word_ptr(esi), ax).unwrap(),
            (Access::Store, _) => a.mov(dword_ptr(esi), eax).unwrap(),
            (Access::Rmw, 1) => a.add(byte_ptr(esi), al).unwrap(),
            (Access::Rmw, 2) => a.add(word_ptr(esi), ax).unwrap(),
            (Access::Rmw, _) => a.add(dword_ptr(esi), eax).unwrap(),
        }
        a.add(ebx, eax).unwrap();
        a.inc(eax).unwrap();
        a.cmp(edi, 2).unwrap();
        a.jne(skip).unwrap();
        a.mov(esi, target).unwrap();
        a.set_label(&mut skip).unwrap();
        a.dec(edi).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
    })
}

/// Linear pages of the paged hole witnesses, below the fixture's 4 MiB:
/// a frame in an uncommitted hole, a frame that shares the host page of
/// [`SPARSE_EDGE`] (committed on a 16 KiB-page host, unmapped), and
/// `SPARSE_EDGE` itself.
const LINEAR_HOLE: u64 = 0x3fe000;
const LINEAR_EDGE_HOLE: u64 = 0x3fd000;
const LINEAR_EDGE: u64 = 0x3fc000;
const HOLE: u64 = 0x30_0000;

fn hole_tables(vm: &Vm, _: &mut Vcpu) {
    for (linear, frame) in [
        (LINEAR_HOLE, HOLE),
        (LINEAR_EDGE_HOLE, SPARSE_EDGE + 0x1000),
        (LINEAR_EDGE, SPARSE_EDGE),
    ] {
        let entry = (frame | 7) as u32;
        vm.write_bytes(PAGE_TABLE + 4 * (linear >> 12), &entry.to_le_bytes())
            .unwrap();
    }
}

/// Holes of a sparse span, reached from compiled code by every access kind
/// and width, unpaged and paged: each gives the interpreter's exit or fault,
/// and a region alone in its host page, the top ROM page and a page past a
/// hole complete inline.
#[test]
fn hole_accesses_on_sparse_backing_equal_the_interpreter() {
    let unpaged_targets: [u32; 9] = [
        HOLE as u32,                   // Uncommitted.
        (SPARSE_EDGE + 0x1000) as u32, // Shares a committed host page.
        (SPARSE_EDGE + 0xffe) as u32,  // Crosses from a region into it.
        (SIZE - 2) as u32,             // Crosses from RAM into the hole above.
        0x8000_0000,                   // Mid-span.
        0xffff_f000 - 2,               // Crosses from a hole into the ROM.
        0xffff_fffe,                   // Reaches past 4 GiB.
        SPARSE_EDGE as u32 + 0x10,     // RAM alone in its host page: inline.
        (SPARSE_ROM + 0x10) as u32,    // ROM: loads inline, writes discarded.
    ];
    let paged_targets = [
        LINEAR_HOLE as u32,
        LINEAR_EDGE_HOLE as u32,
        LINEAR_EDGE as u32 + 0x20,
        (LINEAR_EDGE + 0xffe) as u32,
    ];
    let mut holes = 0;
    let mut halted = 0;
    for model in MODELS {
        for access in [Access::Load, Access::Store, Access::Rmw] {
            for width in [1u8, 2, 4] {
                let mut cells: Vec<(Setup, u32)> = Vec::new();
                for &target in &unpaged_targets {
                    cells.push((sparse(Setup::default()), target));
                    cells.push((
                        sparse(Setup {
                            user: true,
                            ..Setup::default()
                        }),
                        target,
                    ));
                }
                for &target in &paged_targets {
                    cells.push((sparse(PAGED), target));
                    cells.push((sparse(PAGED_USER), target));
                }
                for (setup, target) in cells {
                    let code = hot_then(access, width, target);
                    for subject in [
                        Subject::Mode(MachineJitMode::Forced),
                        Subject::Background(1),
                    ] {
                        let what = format!(
                            "{model:?} {access:?}{width} at {target:#x} paged={} user={} {subject:?}",
                            setup.paged, setup.user
                        );
                        let tweak: &dyn Fn(&Vm, &mut Vcpu) =
                            if setup.paged { &hole_tables } else { &plain };
                        let (c, last) =
                            lockstep(model, &code, subject, setup, 7, tweak, &mut |_| 64, &what);
                        if subject == Subject::Mode(MachineJitMode::Forced) {
                            assert!(c.entries > 0, "{what}: {c:?}");
                        }
                        holes += c.defer_causes[cause("device or hole")];
                        if matches!(last, StepStop::Exit(Exit::Hlt)) {
                            halted += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(holes > 0);
    assert!(halted > 0);
}

/// Run both machines to the end of the program; the subject's last stop.
fn drive(avm: &Vm, a: &mut Vcpu, bvm: &Vm, b: &mut Vcpu, what: &str) -> StepStop {
    for call in 0..2000 {
        let ar = a.run_steps(avm, 4096, StepOptions::default()).unwrap();
        let br = b.run_steps(bvm, 4096, StepOptions::default()).unwrap();
        let at = format!("{what} call {call}");
        assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{at}");
        same(a, b, &at);
        same_memory(avm, bvm, &[(0, MMIO)], &at);
        let go = answer(a, ar.stop());
        answer(b, br.stop());
        same(a, b, &at);
        if !go {
            return ar.into_stop();
        }
    }
    panic!("{what}: did not end");
}

/// A region the compiled view holds is unmapped between runs, and its host
/// pages are decommitted: the next run's accesses to it give the
/// interpreter's exit, never a host fault, and after a remap compiled code
/// reads the fresh zeroed region.
#[test]
fn a_region_unmapped_between_runs_is_never_touched_inline() {
    // mov edi, 100000; mov esi, SPARSE_EDGE; L: add [esi], eax; mov ebx,
    // [esi + 0ffch]; inc eax; dec edi; jnz L; hlt. Long enough for a
    // background compile to land.
    let code = asm(|a| {
        let mut top = a.create_label();
        a.mov(edi, 100_000u32).unwrap();
        a.mov(esi, SPARSE_EDGE as u32).unwrap();
        a.set_label(&mut top).unwrap();
        a.add(dword_ptr(esi), eax).unwrap();
        a.mov(ebx, dword_ptr(esi + 0xffc)).unwrap();
        a.inc(eax).unwrap();
        a.dec(edi).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
    });
    for model in MODELS {
        for setup in [sparse(Setup::default()), sparse(PAGED), sparse(PAGED_USER)] {
            for subject in [
                Subject::Mode(MachineJitMode::Forced),
                Subject::Background(1),
            ] {
                let what = format!("{model:?} paged={} {subject:?}", setup.paged);
                let (mut bvm, mut b) = machine(
                    JitBackend::new(),
                    model,
                    &code,
                    Subject::Mode(MachineJitMode::Off),
                    setup,
                    9,
                );
                let (mut avm, mut a) = machine(JitBackend::new(), model, &code, subject, setup, 9);
                if setup.paged {
                    // Identity for the region and the rest of its host page.
                    for vm in [&avm, &bvm] {
                        for page in 0..4u64 {
                            let linear = SPARSE_EDGE + (page << 12);
                            let entry = (linear | 7) as u32;
                            vm.write_bytes(PAGE_TABLE + 4 * (linear >> 12), &entry.to_le_bytes())
                                .unwrap();
                        }
                    }
                }
                // HLT, or at CPL 3 its #GP(0) (SDM Vol 2A HLT).
                let first = format!("{:?}", drive(&avm, &mut a, &bvm, &mut b, &what));
                assert!(
                    first.contains("Hlt") || first.contains("vector: 13"),
                    "{what}: {first}"
                );
                let entries = a.machine_jit_counters_for_test().entries;
                assert!(entries > 0, "{what}");
                let restart = |cpu: &mut Vcpu| {
                    cpu.cpu.rip = 0x1000;
                };
                for vm in [&mut avm, &mut bvm] {
                    vm.unmap(SPARSE_EDGE, 0x1000).unwrap();
                }
                restart(&mut a);
                restart(&mut b);
                let gone = drive(&avm, &mut a, &bvm, &mut b, &format!("{what} unmapped"));
                assert_ne!(format!("{gone:?}"), first, "{what}");
                for vm in [&mut avm, &mut bvm] {
                    vm.map(SPARSE_EDGE, 0x1000, Prot::RW, RegionKind::Ram)
                        .unwrap();
                }
                restart(&mut a);
                restart(&mut b);
                let back = drive(&avm, &mut a, &bvm, &mut b, &format!("{what} remapped"));
                assert_eq!(format!("{back:?}"), first, "{what}");
                same_memory(&avm, &bvm, &SPANS, &what);
                same_memory(&avm, &bvm, &[(SPARSE_EDGE, SPARSE_EDGE + 0x1000)], &what);
                assert!(
                    a.machine_jit_counters_for_test().entries > entries,
                    "{what}: no compiled entry after the remap"
                );
            }
        }
    }
}

/// Decision-33 Q3: an Application VM on sparse backing stays on the
/// checked interpreter (spec §4.1), with a JIT backend, eager or tiered;
/// the same program on owned backing compiles.
#[test]
fn application_vms_on_sparse_backing_stay_interpreted() {
    // mov ecx, 1000; L: inc rax; dec ecx; jnz L; hlt
    let code = {
        let mut a = CodeAssembler::new(64).unwrap();
        let mut top = a.create_label();
        a.mov(ecx, 1000u32).unwrap();
        a.set_label(&mut top).unwrap();
        a.inc(rax).unwrap();
        a.dec(ecx).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
        a.assemble(0x1000).unwrap()
    };
    for tiering in [None, Some(1)] {
        for sparse in [true, false] {
            let config = VmConfig::flat(0x10000);
            let backend = Box::new(JitBackend::new());
            let mut vm = if sparse {
                Vm::with_backend_sparse(config, backend).unwrap()
            } else {
                Vm::with_backend(config, backend)
            };
            vm.set_tier_up_after(tiering);
            vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, &code).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = 0x1000;
            assert!(matches!(cpu.run(&vm, Some(100_000)), Exit::Hlt));
            assert_eq!(cpu.cpu.gpr[0], 1000);
            let compiled = vm.cache.memory_stats().compiled_entries;
            assert_eq!(compiled == 0, sparse, "sparse={sparse} {tiering:?}");
        }
    }
}
