//! Decision-32 (J3+J4): compiled Machine blocks with paging on and at CPL 3,
//! against the interpreter in lockstep at every `run_steps` call. The paged
//! machines map every page through `machine_j1_lockstep::paging_fixture`;
//! after every call the full state (CR2 included), the written RAM with the
//! page tables, the step result, the retired count and the TLB entries must
//! agree, and all RAM at the end of a run.
//!
//! Compiled code only reads the TLB: a miss, a right the entry lacks and a
//! clean page's first write defer, and the interpreter walks, sets A and D,
//! raises the #PF and publishes CR2. The witnesses cover the J1 programs paged
//! at CPL 0 and CPL 3 and unpaged at CPL 3; #PF from every cause and access
//! kind at several ordinals, with the 80386/Intel486 difference on protection
//! faults; A and D; CR3 loads that reuse shared units; INVLPG and CR0
//! toggles; INT n and IRET between CPL 3 and CPL 0; privileged instructions
//! at CPL 3; the 486 alignment gate; FS and GS operands; stores into code
//! through a second linear mapping; and two vCPUs sharing page tables.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use super::machine_j1_lockstep::{
    add, answer, asm, families, live, looped, machine, programs, same, same_memory, Setup, Subject,
    CODE_ALIAS, MODELS, MODES, NOT_PRESENT, PAGE_DIRECTORY, SPANS, SUPERVISOR,
    SUPERVISOR_READ_ONLY, USER_READ_ONLY,
};
use iced_x86::code_asm::*;
use x86jit_core::{
    CpuModel, Exit, MachineJitCounters, MachineJitMode, SegmentCache, StepOptions, StepStop,
    SystemSegment, TableRegister, Vcpu, Vm,
};
use x86jit_cranelift::JitBackend;

const PAGED: Setup = Setup {
    a20_off: false,
    sparse: false,
    watch: false,
    mxcsr_mode: false,
    rf: false,
    tf: false,
    chain_off: false,
    paged: true,
    user: false,
    wp: false,
};
const PAGED_USER: Setup = Setup {
    user: true,
    ..PAGED
};
const UNPAGED_USER: Setup = Setup {
    paged: false,
    user: true,
    ..PAGED
};

fn cause(name: &str) -> usize {
    MachineJitCounters::DEFER_CAUSE_NAMES
        .iter()
        .position(|n| *n == name)
        .unwrap()
}

/// `machine` with `tweak` applied to the vCPU and VM of both machines.
fn pair(
    model: CpuModel,
    code: &[u8],
    subject: Subject,
    setup: Setup,
    seed: u64,
    tweak: &dyn Fn(&Vm, &mut Vcpu),
) -> ((Vm, Vcpu), (Vm, Vcpu)) {
    let (bvm, mut b) = machine(
        JitBackend::new(),
        model,
        code,
        Subject::Mode(MachineJitMode::Off),
        setup,
        seed,
    );
    tweak(&bvm, &mut b);
    let (avm, mut a) = machine(JitBackend::new(), model, code, subject, setup, seed);
    tweak(&avm, &mut a);
    ((avm, a), (bvm, b))
}

/// Run both machines with this budget schedule until the program ends; the
/// subject's counters and its last stop.
#[allow(clippy::too_many_arguments)]
fn lockstep(
    model: CpuModel,
    code: &[u8],
    subject: Subject,
    setup: Setup,
    seed: u64,
    tweak: &dyn Fn(&Vm, &mut Vcpu),
    budgets: &mut dyn FnMut(usize) -> u64,
    what: &str,
) -> (MachineJitCounters, StepStop) {
    let ((avm, mut a), (bvm, mut b)) = pair(model, code, subject, setup, seed, tweak);
    let mut last = StepStop::Exit(Exit::BudgetExhausted);
    for call in 0..6000 {
        let budget = budgets(call);
        let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
        let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
        let at = format!("{what} call {call} budget {budget}");
        assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{at}");
        same(&a, &b, &at);
        // The pages the programs write and the page tables, at each call.
        same_memory(&avm, &bvm, &[(0, 0x9000), (0x30000, 0x70000)], &at);
        let go = answer(&mut a, ar.stop());
        answer(&mut b, br.stop());
        same(&a, &b, &at);
        last = ar.into_stop();
        if !go {
            break;
        }
    }
    same_memory(&avm, &bvm, &SPANS, what);
    (a.machine_jit_counters_for_test(), last)
}

fn plain(_: &Vm, _: &mut Vcpu) {}

/// Decision-32 §9: the J1 programs and families, paged at CPL 0 and CPL 3
/// and unpaged at CPL 3, in every compiled mode, against the interpreter.
/// The Intel486 also runs paged at CPL 0 with WP set.
fn j1_programs_paged_and_at_cpl3(model: CpuModel) {
    let mut setups = vec![("paged", PAGED), ("paged-cpl3", PAGED_USER)];
    setups.push(("unpaged-cpl3", UNPAGED_USER));
    if model != CpuModel::I80386 {
        setups.push(("paged-wp", Setup { wp: true, ..PAGED }));
    }
    let mut forced = [MachineJitCounters::default(); 2];
    for (index, (name, code, extra)) in programs(model).into_iter().enumerate() {
        for &(kind, setup) in &setups {
            let setup = Setup {
                a20_off: extra.a20_off,
                watch: extra.watch,
                ..setup
            };
            for mode in MODES {
                for budget in [1u64, 7, 64] {
                    let what = format!("{name} {kind} {model:?} {mode:?} budget={budget}");
                    let (c, _) = lockstep(
                        model,
                        &code,
                        Subject::Mode(mode),
                        setup,
                        index as u64 + 1,
                        &plain,
                        &mut |_| budget,
                        &what,
                    );
                    if mode == MachineJitMode::Forced && setup.paged {
                        add(&mut forced[usize::from(setup.user)], c);
                    }
                }
                for first in [0u64, 2, 5] {
                    let what = format!("{name} {kind} {model:?} {mode:?} first={first}");
                    lockstep(
                        model,
                        &code,
                        Subject::Mode(mode),
                        setup,
                        3,
                        &plain,
                        &mut |call| if call == 0 { first } else { 7 },
                        &what,
                    );
                }
            }
        }
    }
    // Done-when 1: compiled code ran paged, at CPL 0 and at CPL 3.
    for (cpl, c) in [(0, forced[0]), (3, forced[1])] {
        assert!(c.paged_entries > 0, "CPL {cpl}: {c:?}");
        assert!(c.compiled_retired > 5_000, "CPL {cpl}: {c:?}");
        assert!(c.chained > 0, "CPL {cpl}: {c:?}");
        assert!(c.helper_defers > 0, "CPL {cpl}: {c:?}");
    }
    assert!(forced[1].compiled_retired_cpl3 > 5_000, "{:?}", forced[1]);
    assert_eq!(forced[0].compiled_retired_cpl3, 0, "{:?}", forced[0]);
    for name in ["tlb miss", "clean write", "page crossing", "device or hole"] {
        assert!(
            forced[0].defer_causes[cause(name)] > 0,
            "{name}: {:?}",
            forced[0]
        );
    }
}

#[test]
fn paged_and_cpl3_programs_equal_the_interpreter_386() {
    j1_programs_paged_and_at_cpl3(CpuModel::I80386);
}

#[test]
fn paged_and_cpl3_programs_equal_the_interpreter_486() {
    j1_programs_paged_and_at_cpl3(CpuModel::I80486Dx);
}

/// Every op family compiles paged at CPL 3 too: only the final HLT, a
/// singleton block, stays interpreted, and the first touch of each page
/// defers once.
#[test]
fn every_lowered_family_runs_compiled_paged_at_cpl3() {
    for model in MODELS {
        for (name, code) in families(model) {
            if name == "double-shift-16" {
                continue;
            }
            let (c, _) = lockstep(
                model,
                &code,
                Subject::Mode(MachineJitMode::Forced),
                PAGED_USER,
                3,
                &plain,
                &mut |_| 1000,
                name,
            );
            assert_eq!(c.unconditional_defers, 0, "{name} {model:?}: {c:?}");
            assert!(c.continues >= 5, "{name} {model:?}: {c:?}");
            assert!(c.compiled_retired_cpl3 > 0, "{name} {model:?}: {c:?}");
        }
    }
}

/// The production configuration, background compilation at T1 = 1 and 3,
/// paged at CPL 0 and CPL 3, whatever the worker's timing.
#[test]
fn paged_background_tier_up_equals_the_interpreter() {
    for model in MODELS {
        let mut entries = [0; 2];
        for (name, code, extra) in programs(model).into_iter().take(12) {
            for setup in [PAGED, PAGED_USER] {
                let setup = Setup {
                    a20_off: extra.a20_off,
                    ..setup
                };
                for t1 in [1, 3] {
                    for budget in [7u64, 64] {
                        let what = format!(
                            "{name} {model:?} user={} background t1={t1} budget={budget}",
                            setup.user
                        );
                        let (c, _) = lockstep(
                            model,
                            &code,
                            Subject::Background(t1),
                            setup,
                            3,
                            &plain,
                            &mut |_| budget,
                            &what,
                        );
                        entries[usize::from(setup.user)] += c.paged_entries;
                    }
                }
            }
        }
        // A loop long enough for the worker's publication to land mid-run.
        let long = asm(|a| {
            let mut top = a.create_label();
            a.mov(edi, 400_000u32).unwrap();
            a.mov(esi, 0x6000u32).unwrap();
            a.set_label(&mut top).unwrap();
            a.inc(eax).unwrap();
            a.add(dword_ptr(esi), eax).unwrap();
            a.dec(edi).unwrap();
            a.jne(top).unwrap();
            a.hlt().unwrap();
        });
        for setup in [PAGED, PAGED_USER] {
            let (c, _) = lockstep(
                model,
                &long,
                Subject::Background(50),
                setup,
                3,
                &plain,
                &mut |_| 50_000,
                "long background",
            );
            entries[usize::from(setup.user)] += c.paged_entries;
        }
        assert!(entries[0] > 0 && entries[1] > 0, "{model:?}: {entries:?}");
    }
}

#[derive(Clone, Copy, Debug)]
enum Access {
    Load,
    Store,
    Push,
    Pop,
    CallMemory,
    CallPush,
    Ret,
    /// A load that fills the TLB, then, in the next block, a store the entry
    /// refuses: the protection fault on a TLB hit (80386 PRM §5.2.5;
    /// Intel486 PRM §9.9.14 p. 9-21 sets both A bits first), or the first
    /// write to a clean cached page.
    ReadThenWrite,
}

const ACCESSES: [Access; 8] = [
    Access::Load,
    Access::Store,
    Access::Push,
    Access::Pop,
    Access::CallMemory,
    Access::CallPush,
    Access::Ret,
    Access::ReadThenWrite,
];

/// `mov esi, 1; inc eax` x `prefix`; the access to `page`; `inc ebx; hlt`,
/// in one block, so the access is at ordinal `prefix + 1` (+1 for the stack
/// forms, which set ESP first).
fn fault_program(prefix: usize, access: Access, page: u64) -> Vec<u8> {
    let at = page as u32 + 0x800;
    asm(|a| {
        a.mov(esi, 1u32).unwrap();
        for _ in 0..prefix {
            a.inc(eax).unwrap();
        }
        match access {
            Access::Load => a.mov(ecx, dword_ptr(at)).unwrap(),
            Access::Store => a.mov(dword_ptr(at), eax).unwrap(),
            Access::Push => {
                a.mov(esp, at).unwrap();
                a.push(eax).unwrap();
            }
            Access::Pop => {
                a.mov(esp, at).unwrap();
                a.pop(ecx).unwrap();
            }
            Access::CallMemory => a.call(dword_ptr(at)).unwrap(),
            Access::CallPush => {
                let mut target = a.create_label();
                a.mov(esp, at).unwrap();
                a.call(target).unwrap();
                a.set_label(&mut target).unwrap();
            }
            Access::Ret => {
                a.mov(esp, at).unwrap();
                a.ret().unwrap();
            }
            Access::ReadThenWrite => {
                // The store's block starts after the load's walk filled the
                // TLB, so the compiled store meets the cached entry.
                let mut next = a.create_label();
                a.mov(ecx, dword_ptr(at)).unwrap();
                a.jmp(next).unwrap();
                a.set_label(&mut next).unwrap();
                a.inc(ecx).unwrap();
                a.mov(dword_ptr(at), ecx).unwrap();
            }
        }
        a.inc(ebx).unwrap();
        a.hlt().unwrap();
    })
}

/// Decision-32 §9: #PF at several ordinals of a compiled block, for every
/// cause and access kind, with the error code and CR2 compared. The
/// compiled block defers each such access; the interpreter faults.
#[test]
fn page_faults_at_every_ordinal_equal_the_interpreter() {
    let not_present_directory = 0x40_0000u64;
    for model in MODELS {
        let wp = Setup { wp: true, ..PAGED };
        let mut cases: Vec<(&str, u64, Setup, bool)> = vec![
            ("pde-not-present", not_present_directory, PAGED, true),
            (
                "pde-not-present-cpl3",
                not_present_directory,
                PAGED_USER,
                true,
            ),
            ("pte-not-present", NOT_PRESENT, PAGED, true),
            ("pte-not-present-cpl3", NOT_PRESENT, PAGED_USER, true),
            ("user-on-supervisor", SUPERVISOR, PAGED_USER, true),
            ("user-write-read-only", USER_READ_ONLY, PAGED_USER, false),
            // 80386 PRM §§6.4.1-6.4.2 Table 6-5: a supervisor write ignores
            // R/W without WP, which the 80386 lacks (SDM Vol 3A §5.6.1).
            (
                "supervisor-write-read-only",
                SUPERVISOR_READ_ONLY,
                PAGED,
                false,
            ),
        ];
        if model != CpuModel::I80386 {
            cases.push((
                "supervisor-write-read-only-wp",
                SUPERVISOR_READ_ONLY,
                wp,
                false,
            ));
        }
        let mut faults = 0;
        let mut forced = MachineJitCounters::default();
        for (name, page, setup, reads_fault) in cases {
            for access in ACCESSES {
                let reads = matches!(
                    access,
                    Access::Load | Access::Pop | Access::CallMemory | Access::Ret
                );
                if reads && !reads_fault {
                    continue;
                }
                for prefix in [0, 1, 3] {
                    let code = fault_program(prefix, access, page);
                    for subject in MODES
                        .into_iter()
                        .map(Subject::Mode)
                        .chain([Subject::Background(1)])
                    {
                        let what = format!("{name} {access:?} @{prefix} {model:?} {subject:?}");
                        let (c, stop) =
                            lockstep(model, &code, subject, setup, 1, &plain, &mut |_| 64, &what);
                        if matches!(stop, StepStop::Exit(Exit::Exception { vector: 14, .. })) {
                            faults += 1;
                        }
                        if subject == Subject::Mode(MachineJitMode::Forced) {
                            add(&mut forced, c);
                        }
                    }
                }
            }
        }
        assert!(faults > 100, "{model:?}: {faults} faults");
        for name in ["tlb miss", "rights", "clean write"] {
            assert!(forced.defer_causes[cause(name)] > 0, "{name}: {forced:?}");
        }
    }
}

/// Decision-32 §9, A and D: the first read sets A on both levels and the
/// first write D, also for a clean page cached in the middle of a block; a
/// guest that clears A or D with no INVLPG keeps the stale entry in both
/// tiers; and page tables in a frame that also holds code, whose walks'
/// A/D updates and the guest's own entry stores land in a code page.
#[test]
fn accessed_and_dirty_bits_equal_the_interpreter() {
    let pages = asm(|a| {
        let mut top = a.create_label();
        a.mov(esi, 0x60000u32).unwrap();
        a.mov(edi, 6u32).unwrap();
        a.set_label(&mut top).unwrap();
        // Read, then write, a fresh page; then write the previous one again.
        a.mov(eax, dword_ptr(esi)).unwrap();
        a.add(eax, 3).unwrap();
        a.mov(dword_ptr(esi + 4), eax).unwrap();
        a.mov(dword_ptr(esi - 0x1000 + 8), edi).unwrap();
        a.add(esi, 0x1000).unwrap();
        a.dec(edi).unwrap();
        a.jne(top).unwrap();
        // Clear A and D of 60000h's entry with no INVLPG, write the page
        // again (the stale entry keeps D), and read the entry back.
        a.mov(ebx, (0x4000 + 4 * 0x60) as u32).unwrap();
        a.and(dword_ptr(ebx), !0x60i32).unwrap();
        a.mov(dword_ptr(0x60010), eax).unwrap();
        a.mov(ecx, dword_ptr(0x60014)).unwrap();
        a.mov(edx, dword_ptr(ebx)).unwrap();
        a.hlt().unwrap();
    });
    // Directory entry 1 names a table in the code page's frame: linear
    // 7F0000h is mapped by the entry at 1FC0h, the code page's end.
    let tables_in_code = asm(|a| {
        let mut top = a.create_label();
        a.mov(dword_ptr(PAGE_DIRECTORY as u32 + 4), 0x1007).unwrap();
        a.mov(dword_ptr(0x1fc0), 0x62007).unwrap();
        a.mov(edi, 5u32).unwrap();
        a.set_label(&mut top).unwrap();
        a.add(dword_ptr(0x7f_0000 + 0x20), edi).unwrap();
        a.mov(eax, dword_ptr(0x1fc0)).unwrap();
        a.inc(ebx).unwrap();
        a.dec(edi).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
    });
    for model in MODELS {
        for (name, code) in [("pages", &pages), ("tables-in-code", &tables_in_code)] {
            for setup in [PAGED, PAGED_USER] {
                for subject in MODES.into_iter().map(Subject::Mode) {
                    for budget in [1u64, 5, 64] {
                        let what =
                            format!("{name} {model:?} {subject:?} user={} {budget}", setup.user);
                        lockstep(
                            model,
                            code,
                            subject,
                            setup,
                            1,
                            &plain,
                            &mut |_| budget,
                            &what,
                        );
                    }
                }
            }
        }
    }
}

/// A second address space for the CR3 witnesses: directory 2A000h with
/// table 2B000h, the same entries as `paging_fixture` but for linear
/// 40000h, mapped to frame 41000h.
fn second_space(vm: &Vm) {
    vm.write_bytes(0x2a000, &0x2b007u32.to_le_bytes()).unwrap();
    for page in 0..1024u64 {
        let mut entry = [0; 4];
        vm.read_bytes(0x4000 + 4 * page, &mut entry).unwrap();
        let entry = if page == 0x40 {
            0x41007
        } else {
            u32::from_le_bytes(entry)
        };
        vm.write_bytes(0x2b000 + 4 * page, &entry.to_le_bytes())
            .unwrap();
    }
    // 40000h: inc eax; ret. Frame 41000h: inc ecx; ret.
    vm.write_bytes(0x40000, &[0x40, 0xc3]).unwrap();
    vm.write_bytes(0x41000, &[0x41, 0xc3]).unwrap();
}

/// `mov edi, n; L: call 40000h; inc ebx; dec ebp...`: every `interval`
/// iterations the other address space. EBP holds the directory.
fn cr3_program(iterations: u32, interval: u32) -> Vec<u8> {
    asm(|a| {
        let mut top = a.create_label();
        let mut stay = a.create_label();
        a.mov(ebp, PAGE_DIRECTORY as u32).unwrap();
        a.mov(edi, iterations).unwrap();
        a.mov(esi, interval).unwrap();
        a.set_label(&mut top).unwrap();
        a.call(0x40000u64).unwrap();
        a.inc(ebx).unwrap();
        a.add(dword_ptr(0x6000), ebx).unwrap();
        a.dec(esi).unwrap();
        a.jne(stay).unwrap();
        a.xor(ebp, (PAGE_DIRECTORY ^ 0x2a000) as u32).unwrap();
        a.mov(cr3, ebp).unwrap();
        a.mov(esi, interval).unwrap();
        a.set_label(&mut stay).unwrap();
        a.dec(edi).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
    })
}

/// Decision-32 §4 and its amendment: CR3 loads mid-run. Each load empties
/// the TLB and the compiled view; the paged proofs, and with them the memo
/// slots and units, survive and revalidate their code pages. The second
/// address space reuses every unit of the shared pages (no new compile or
/// lift once both spaces have run), and the page mapped differently in the
/// two spaces gets a proof and a unit of its own.
#[test]
fn cr3_loads_reuse_shared_units() {
    let switch = |vm: &Vm, _: &mut Vcpu| second_space(vm);
    for model in MODELS {
        for subject in MODES
            .into_iter()
            .map(Subject::Mode)
            .chain([Subject::Background(1), Subject::Background(3)])
        {
            for budget in [7u64, 64, 1000] {
                let what = format!("cr3 {model:?} {subject:?} {budget}");
                lockstep(
                    model,
                    &cr3_program(60, 4),
                    subject,
                    PAGED,
                    1,
                    &switch,
                    &mut |_| budget,
                    &what,
                );
            }
        }
        // The counters, in one run (a budget cut would start new blocks
        // mid-block): 40 CR3 loads flush the TLB, and the paged proofs
        // revalidate their code pages instead of relifting (decision-32's
        // amendment); each block compiles once, and 40000h once per address
        // space, as its bytes differ (frame 40000h or 41000h).
        let compiled = |iterations: u32| {
            let ((vm, mut cpu), _) = pair(
                model,
                &cr3_program(iterations, 10),
                Subject::Mode(MachineJitMode::Forced),
                PAGED,
                1,
                &switch,
            );
            let run = cpu
                .run_steps(&vm, 1_000_000, StepOptions::default())
                .unwrap();
            assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)), "{run:?}");
            (
                cpu.machine_jit_counters_for_test(),
                cpu.paging_cache_for_test().2,
            )
        };
        let (few, few_lifts) = compiled(30);
        let (many, many_lifts) = compiled(400);
        assert_eq!(few.compiled, many.compiled, "{model:?}: {few:?} {many:?}");
        assert!(many.compiled >= 6, "{model:?}: {many:?}");
        assert_eq!(many_lifts, few_lifts, "{model:?}: CR3 loads relifted");
        assert!(many.paged_entries > 500, "{model:?}: {many:?}");
    }
}

/// Decision-32 §9: INVLPG of a data page in a loop and of the running code
/// page (Intel486 PRM §26 INVLPG p. 26-172), and CR0 toggles of WP, CD and
/// NW, and of PG off and back on, at CPL 0 (SDM Vol 3A §5.10.4.1).
#[test]
fn invlpg_and_cr0_toggles_equal_the_interpreter() {
    let invlpg = asm(|a| {
        let mut top = a.create_label();
        a.mov(edi, 8u32).unwrap();
        a.set_label(&mut top).unwrap();
        a.add(dword_ptr(0x60000), edi).unwrap();
        a.invlpg(dword_ptr(0x60000)).unwrap();
        a.mov(eax, dword_ptr(0x60000)).unwrap();
        a.invlpg(dword_ptr(0x1000)).unwrap();
        a.inc(ebx).unwrap();
        a.dec(edi).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
    });
    let toggles = |bits: u32| {
        asm(move |a| {
            let mut top = a.create_label();
            a.mov(edi, 6u32).unwrap();
            a.set_label(&mut top).unwrap();
            a.add(dword_ptr(0x6000), edi).unwrap();
            a.mov(eax, cr0).unwrap();
            a.xor(eax, bits).unwrap();
            a.mov(cr0, eax).unwrap();
            a.mov(dword_ptr(0x60000), eax).unwrap();
            a.mov(ecx, dword_ptr(SUPERVISOR_READ_ONLY as u32)).unwrap();
            a.dec(edi).unwrap();
            a.jne(top).unwrap();
            a.hlt().unwrap();
        })
    };
    let mut programs = vec![("pg", toggles(0x8000_0000))];
    for model in MODELS {
        if model != CpuModel::I80386 {
            programs.push(("invlpg", invlpg.clone()));
            programs.push(("wp", toggles(1 << 16)));
            programs.push(("cd", toggles(1 << 30)));
            programs.push(("cd-nw", toggles(0x6000_0000)));
        }
        for (name, code) in &programs {
            for subject in MODES.into_iter().map(Subject::Mode) {
                for budget in [1u64, 4, 64] {
                    let what = format!("{name} {model:?} {subject:?} {budget}");
                    lockstep(
                        model,
                        code,
                        subject,
                        PAGED,
                        1,
                        &plain,
                        &mut |_| budget,
                        &what,
                    );
                }
            }
        }
    }
}

/// An IDT with a DPL-3 interrupt gate for vector 80h to `1800h: inc esi;
/// iretd`, ring-0 code and data descriptors (GDT indexes 3 and 4) and a TSS
/// whose ring-0 stack is at 7800h (SDM Vol 3A §§6.12.1, 7.2.1).
fn kernel(vm: &Vm, cpu: &mut Vcpu) {
    fixture::descriptor(vm, 0x2018, 0, u32::MAX, 0xc09b);
    fixture::descriptor(vm, 0x2020, 0, u32::MAX, 0xc093);
    fixture::descriptor(vm, 0x2028, 0x2e00, 0x67, 0x8b);
    let handler = 0x1800u32;
    let low = (0x18 << 16) | (handler & 0xffff);
    let high = (handler & 0xffff_0000) | 0xee00;
    vm.write_bytes(0x2800 + 8 * 0x80, &low.to_le_bytes())
        .unwrap();
    vm.write_bytes(0x2800 + 8 * 0x80 + 4, &high.to_le_bytes())
        .unwrap();
    vm.write_bytes(0x2e04, &0x7800u32.to_le_bytes()).unwrap();
    vm.write_bytes(0x2e08, &0x20u32.to_le_bytes()).unwrap();
    vm.write_bytes(0x1800, &[0x46, 0xcf]).unwrap();
    cpu.cpu.legacy.idtr = TableRegister {
        base: 0x2800,
        limit: 0x7ff,
        reserved: 0,
    };
    cpu.cpu.legacy.tr = SystemSegment {
        selector: 0x28,
        reserved: 0,
        cache: SegmentCache {
            base: 0x2e00,
            limit: 0x67,
            attributes: 0x8b,
            ..Default::default()
        },
    };
}

/// Decision-32 §5, §9: a CPL-3 loop that calls the kernel through INT 80h
/// and returns with IRET, and privileged and I/O instructions at CPL 3 with
/// IOPL 0, which fault (#GP) in the interpreter after the compiled prefix
/// (SDM Vol 3A §6.9; Vol 1 §20.5.1; 80386 PRM §§6.3.5.1-6.3.5.2).
#[test]
fn user_mode_kernel_calls_and_privileged_instructions_equal_the_interpreter() {
    let calls = asm(|a| {
        let mut top = a.create_label();
        a.mov(edi, 12u32).unwrap();
        a.set_label(&mut top).unwrap();
        a.inc(eax).unwrap();
        a.add(dword_ptr(0x6000), eax).unwrap();
        a.int(0x80).unwrap();
        a.add(ebx, esi).unwrap();
        a.dec(edi).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
    });
    let privileged: Vec<(&str, Vec<u8>)> = vec![
        ("cli", vec![0xfa]),
        ("sti", vec![0xfb]),
        ("in", vec![0xe4, 0x60]),
        ("out", vec![0xe6, 0x60]),
        ("hlt", vec![0xf4]),
        ("mov-from-cr0", vec![0x0f, 0x20, 0xc0]),
        ("mov-to-cr3", vec![0x0f, 0x22, 0xd8]),
        ("lgdt", vec![0x0f, 0x01, 0x15, 0x00, 0x20, 0x00, 0x00]),
        ("clts", vec![0x0f, 0x06]),
    ];
    for model in MODELS {
        let mut programs = vec![("int-iret".to_string(), calls.clone())];
        let mut list = privileged.clone();
        if model != CpuModel::I80386 {
            list.push(("invlpg", vec![0x0f, 0x01, 0x3d, 0x00, 0x60, 0x00, 0x00]));
            list.push(("wbinvd", vec![0x0f, 0x09]));
        }
        for (name, bytes) in list {
            let mut code = b"\x40\x01\xc3\xa3\x00\x60\x00\x00\x43".to_vec();
            code.extend(bytes);
            code.extend(b"\x40\xf4");
            programs.push((name.to_string(), code));
        }
        for (name, code) in &programs {
            for setup in [PAGED_USER, UNPAGED_USER] {
                for subject in MODES.into_iter().map(Subject::Mode) {
                    for budget in [1u64, 5, 64] {
                        let what = format!(
                            "{name} {model:?} {subject:?} paged={} {budget}",
                            setup.paged
                        );
                        let (c, stop) = lockstep(
                            model,
                            code,
                            subject,
                            setup,
                            1,
                            &kernel,
                            &mut |_| budget,
                            &what,
                        );
                        if name != "int-iret" {
                            assert!(
                                matches!(stop, StepStop::Exit(Exit::Exception { vector: 13, .. })),
                                "{what}: {stop:?}"
                            );
                        } else if subject == Subject::Mode(MachineJitMode::Forced) && budget == 64 {
                            assert!(c.compiled_retired_cpl3 > 0, "{what}: {c:?}");
                        }
                    }
                }
            }
        }
    }
}

/// Decision-32 §6: the Intel486 alignment gate at CPL 3 in all four AM and
/// AC combinations. #AC needs CPL 3, CR0.AM and EFLAGS.AC (SDM Vol 3A §7.15,
/// Event 17, p. 7-53); with both set the compiled block defers at the gate
/// and the interpreter decides the instruction.
#[test]
fn the_alignment_gate_equals_the_interpreter() {
    let code = looped(|a| {
        a.add(eax, dword_ptr(0x6001)).unwrap();
        a.mov(word_ptr(0x6002), ax).unwrap();
        a.add(dword_ptr(0x6010), ecx).unwrap();
        a.inc(ebx).unwrap();
    });
    let mut gated = 0;
    for am in [false, true] {
        for ac in [false, true] {
            let tweak = move |_: &Vm, cpu: &mut Vcpu| {
                if am {
                    cpu.cpu.legacy.cr0 |= 1 << 18;
                }
                if ac {
                    cpu.cpu.rflags_system |= 1 << 18;
                }
            };
            for setup in [PAGED_USER, UNPAGED_USER] {
                for subject in MODES.into_iter().map(Subject::Mode) {
                    for budget in [1u64, 64] {
                        let what =
                            format!("am={am} ac={ac} paged={} {subject:?} {budget}", setup.paged);
                        let (c, _) = lockstep(
                            CpuModel::I80486Dx,
                            &code,
                            subject,
                            setup,
                            1,
                            &tweak,
                            &mut |_| budget,
                            &what,
                        );
                        gated += c.defer_causes[cause("ac gate")];
                        if subject == Subject::Mode(MachineJitMode::Forced) && budget == 64 {
                            assert!(c.entries > 0, "{what}: {c:?}");
                            if !(am && ac) {
                                assert!(c.compiled_retired_cpl3 > 10, "{what}: {c:?}");
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(gated > 0, "the gate never deferred");
}

/// Decision-32 §6: FS and GS operands in Flat32 code, inline for a usable
/// expand-up data segment whose limit holds the operand, and deferred to the
/// interpreter's checks otherwise: null, a limit crossed mid-loop, an
/// expand-down segment, readable and execute-only code segments, a read-only
/// segment written, and a base that wraps the operand past 4 GiB.
#[test]
fn fs_and_gs_operands_equal_the_interpreter() {
    let code = asm(|a| {
        let mut top = a.create_label();
        a.mov(ebx, 0xfe8u32).unwrap();
        a.mov(edi, 8u32).unwrap();
        a.set_label(&mut top).unwrap();
        a.mov(eax, dword_ptr(ebx).fs()).unwrap();
        a.add(dword_ptr(ebx + 0x10).gs(), eax).unwrap();
        a.mov(dword_ptr(0x20).fs(), ecx).unwrap();
        a.add(ebx, 4).unwrap();
        a.dec(edi).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
    });
    let caches: Vec<(&str, SegmentCache)> = vec![
        (
            "flat",
            SegmentCache {
                base: 0x6000,
                limit: 0xffff,
                attributes: 0x4093,
                ..Default::default()
            },
        ),
        (
            "null",
            SegmentCache {
                unusable: 1,
                ..Default::default()
            },
        ),
        (
            "limit",
            SegmentCache {
                base: 0x6000,
                limit: 0xfff,
                attributes: 0x4093,
                ..Default::default()
            },
        ),
        (
            "expand-down",
            SegmentCache {
                base: 0x6000,
                limit: 0xf,
                attributes: 0x4097,
                ..Default::default()
            },
        ),
        (
            "readable-code",
            SegmentCache {
                base: 0x6000,
                limit: 0xffff,
                attributes: 0x409b,
                ..Default::default()
            },
        ),
        (
            "execute-only",
            SegmentCache {
                base: 0x6000,
                limit: 0xffff,
                attributes: 0x4099,
                ..Default::default()
            },
        ),
        (
            "read-only",
            SegmentCache {
                base: 0x6000,
                limit: 0xffff,
                attributes: 0x4091,
                ..Default::default()
            },
        ),
        (
            "wrapping",
            SegmentCache {
                base: 0xffff_f000,
                limit: u32::MAX,
                attributes: 0xc093,
                ..Default::default()
            },
        ),
    ];
    let mut inline = 0;
    let mut refused = 0;
    for model in MODELS {
        for (name, cache) in &caches {
            let cache = *cache;
            let tweak = move |_: &Vm, cpu: &mut Vcpu| {
                let dpl = u16::from(cpu.cpu.legacy.cpl) << 5;
                for index in [4, 5] {
                    let mut cache = cache;
                    if cache.unusable == 0 {
                        cache.attributes |= dpl;
                    }
                    cpu.cpu.legacy.segments[index] = cache;
                }
            };
            for setup in [PAGED, PAGED_USER, UNPAGED_USER] {
                for subject in MODES.into_iter().map(Subject::Mode) {
                    for budget in [1u64, 64] {
                        let what = format!(
                            "{name} {model:?} {subject:?} user={} paged={} {budget}",
                            setup.user, setup.paged
                        );
                        let (c, _) = lockstep(
                            model,
                            &code,
                            subject,
                            setup,
                            1,
                            &tweak,
                            &mut |_| budget,
                            &what,
                        );
                        refused += c.defer_causes[cause("segment")];
                        if *name == "flat" && subject == Subject::Mode(MachineJitMode::Forced) {
                            inline += c.compiled_retired;
                            assert_eq!(c.defer_causes[cause("segment")], 0, "{what}: {c:?}");
                        }
                    }
                }
            }
        }
    }
    assert!(inline > 100, "{inline}");
    assert!(refused > 0, "{refused}");
}

/// Decision-32 §4, §9: stores into code through a second linear mapping of
/// its frame (`CODE_ALIAS`): into another block and into the running
/// block's next instruction, with TF clear and set. The code-tag bit is the
/// frame's, so the alias store finds it. And a store through a write
/// translation the view already holds into a frame that becomes code later.
#[test]
fn stores_through_an_alias_mapping_equal_the_interpreter() {
    let alias = |address: u64| (address - 0x1000 + CODE_ALIAS) as u32;
    // inc eax; mov byte [alias(1010h)], F4h (INC EAX there becomes HLT);
    // inc ecx; ...; 1010h: inc eax; inc ebx; hlt.
    let next = {
        let mut code = asm(|a| {
            a.inc(eax).unwrap();
            a.mov(byte_ptr(alias(0x1010)), 0xf4).unwrap();
            a.inc(ecx).unwrap();
        });
        code.resize(0x10, 0x90);
        code.extend(b"\x40\x43\xf4");
        code
    };
    // mov edi, 6; L: mov byte [alias(1040h)], 41h (INC EBX becomes INC ECX);
    // call 1040h; dec edi; jnz L; hlt ... 1040h: inc ebx; ret.
    let other = {
        let mut code = asm(|a| {
            let mut top = a.create_label();
            a.mov(edi, 6u32).unwrap();
            a.set_label(&mut top).unwrap();
            a.xor(byte_ptr(alias(0x1040)), 2).unwrap();
            a.call(0x1040u64).unwrap();
            a.dec(edi).unwrap();
            a.jne(top).unwrap();
            a.hlt().unwrap();
        });
        code.resize(0x40, 0xf4);
        code.extend(b"\x43\xc3");
        code
    };
    // A data page written through the view becomes code: its bytes are
    // stored, called, and stored again through the same translation.
    let retagged = asm(|a| {
        let mut top = a.create_label();
        a.mov(edi, 8u32).unwrap();
        a.mov(edx, 0xc340u32).unwrap();
        a.set_label(&mut top).unwrap();
        a.mov(dword_ptr(0x60000), edx).unwrap();
        a.call(0x60000u64).unwrap();
        a.xor(edx, 1).unwrap();
        a.mov(dword_ptr(0x60004), edi).unwrap();
        a.dec(edi).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
    });
    let mut stops = 0;
    for model in MODELS {
        for (name, code) in [("next", &next), ("other", &other), ("retagged", &retagged)] {
            for tf in [false, true] {
                for setup in [PAGED, PAGED_USER] {
                    let setup = Setup { tf, ..setup };
                    for subject in MODES.into_iter().map(Subject::Mode) {
                        for budget in [1u64, 3, 64] {
                            let what = format!(
                                "{name} {model:?} {subject:?} tf={tf} user={} {budget}",
                                setup.user
                            );
                            let (c, _) = lockstep(
                                model,
                                code,
                                subject,
                                setup,
                                1,
                                &plain,
                                &mut |_| budget,
                                &what,
                            );
                            if !tf && subject == Subject::Mode(MachineJitMode::Forced) {
                                stops += c.stops;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(stops > 0, "no compiled alias store stopped its block");
}

/// Decision-32 §3: the view stays a subset of the TLB through more data
/// pages than the TLB holds and through conflicting view slots: each fill
/// asserts the invariant in debug builds, and both tiers keep equal TLBs.
#[test]
fn the_view_follows_tlb_turnover() {
    // 300 pages (above SIZE, aliases of 80000h-BFFFFh), then pages 60h and
    // 160h, which share a view slot but not a TLB slot, in turn.
    let code = asm(|a| {
        let mut outer = a.create_label();
        let mut inner = a.create_label();
        let mut pair = a.create_label();
        a.mov(edx, 2u32).unwrap();
        a.set_label(&mut outer).unwrap();
        a.mov(esi, 0x20_0000u32).unwrap();
        a.mov(ecx, 300u32).unwrap();
        a.set_label(&mut inner).unwrap();
        a.add(eax, dword_ptr(esi + 8)).unwrap();
        a.add(dword_ptr(esi + 12), eax).unwrap();
        a.add(esi, 0x1000).unwrap();
        a.dec(ecx).unwrap();
        a.jne(inner).unwrap();
        a.dec(edx).unwrap();
        a.jne(outer).unwrap();
        a.mov(ecx, 40u32).unwrap();
        a.set_label(&mut pair).unwrap();
        a.add(dword_ptr(0x60020), ecx).unwrap();
        a.add(dword_ptr(0x160020), ecx).unwrap();
        a.dec(ecx).unwrap();
        a.jne(pair).unwrap();
        a.hlt().unwrap();
    });
    for model in MODELS {
        for setup in [PAGED, PAGED_USER] {
            for subject in MODES.into_iter().map(Subject::Mode) {
                for budget in [7u64, 64, 5000] {
                    let what = format!(
                        "turnover {model:?} {subject:?} user={} {budget}",
                        setup.user
                    );
                    lockstep(
                        model,
                        &code,
                        subject,
                        setup,
                        1,
                        &plain,
                        &mut |_| budget,
                        &what,
                    );
                }
            }
        }
    }
}

/// Decision-32 §3, cross-vCPU: two vCPUs share page tables. One edits a PTE
/// (frame 62000h for linear 61000h, then back) and both invalidate it after
/// a flag handshake, as an operating system's shootdown does (SDM Vol 3A
/// §5.10.5); a stale entry stays usable until each vCPU's own INVLPG.
/// The reader's result is fixed by the handshake whatever the schedule.
#[test]
fn two_vcpus_share_page_tables() {
    // vCPU 0: for 50 rounds: wait for flag [6100h] == round, read [61000h]
    // into the sum, then INVLPG and acknowledge in [6104h].
    // vCPU 1: for 50 rounds: edit the PTE, INVLPG, set the flag, wait for
    // the acknowledgement, and patch the other's loop body immediate.
    let pte = (0x4000 + 4 * 0x61) as u32;
    let reader = asm(|a| {
        let mut top = a.create_label();
        a.mov(edi, 1u32).unwrap();
        a.set_label(&mut top).unwrap();
        a.cmp(dword_ptr(0x6100), edi).unwrap();
        a.jne(top).unwrap();
        a.invlpg(dword_ptr(0x61000)).unwrap();
        a.add(eax, dword_ptr(0x61000)).unwrap();
        a.mov(dword_ptr(0x6104), edi).unwrap();
        a.inc(edi).unwrap();
        a.cmp(edi, 51).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
    });
    let writer_at = 0x1800u64;
    let writer = {
        let mut a = CodeAssembler::new(32).unwrap();
        let mut top = a.create_label();
        let mut wait = a.create_label();
        a.mov(edi, 1u32).unwrap();
        a.set_label(&mut top).unwrap();
        a.mov(eax, edi).unwrap();
        a.and(eax, 1).unwrap();
        a.shl(eax, 13).unwrap();
        a.add(eax, 0x61007).unwrap();
        a.mov(dword_ptr(pte), eax).unwrap();
        a.invlpg(dword_ptr(0x61000)).unwrap();
        a.mov(dword_ptr(0x6100), edi).unwrap();
        a.set_label(&mut wait).unwrap();
        a.cmp(dword_ptr(0x6104), edi).unwrap();
        a.jne(wait).unwrap();
        a.inc(edi).unwrap();
        a.cmp(edi, 51).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
        a.assemble(writer_at).unwrap()
    };
    let mut code = reader;
    code.resize((writer_at - 0x1000) as usize, 0xf4);
    code.extend(&writer);
    for subject in [
        Subject::Mode(MachineJitMode::Forced),
        Subject::Mode(MachineJitMode::Delayed(1)),
        Subject::Background(3),
    ] {
        let (vm, mut first) = machine(
            JitBackend::new(),
            CpuModel::I80486Dx,
            &code,
            subject,
            PAGED,
            1,
        );
        vm.write_bytes(0x61000, &0x11u32.to_le_bytes()).unwrap();
        vm.write_bytes(0x63000, &0x2200u32.to_le_bytes()).unwrap();
        let mut second = vm.new_vcpu();
        second.cpu = first.cpu.clone();
        second.cpu.rip = writer_at;
        first.cpu.gpr[0] = 0;
        std::thread::scope(|scope| {
            scope.spawn(|| loop {
                let run = second
                    .run_steps(&vm, 10_000, StepOptions::default())
                    .unwrap();
                if !matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)) {
                    assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)), "{run:?}");
                    break;
                }
            });
            loop {
                let run = first
                    .run_steps(&vm, 10_000, StepOptions::default())
                    .unwrap();
                if !matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)) {
                    assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)), "{run:?}");
                    break;
                }
            }
        });
        // Round r reads frame 61000h + (r & 1) * 2000h: 25 of each.
        assert_eq!(
            first.cpu.gpr[0] & 0xffff_ffff,
            25 * 0x11 + 25 * 0x2200,
            "{subject:?}: {}",
            live(&first.cpu)
        );
        // The worker's timing decides whether a background unit lands in
        // this short run; the synchronous modes always enter one.
        if !matches!(subject, Subject::Background(_)) {
            assert!(
                first.machine_jit_counters_for_test().paged_entries > 0,
                "{subject:?}"
            );
        }
    }
}

/// Decision-32 §9: a store by one vCPU through the alias mapping into code
/// another vCPU runs compiled, paged, is seen at that vCPU's next turn.
#[test]
fn an_alias_store_from_another_vcpu_reaches_running_paged_code() {
    // 1000h: L: inc eax; jmp L. 7000h: mov byte [alias(1000h)], F4h; hlt.
    let mut code = vec![0x40, 0xeb, 0xfd];
    code.resize(0x6000, 0xf4);
    code.extend(b"\xc6\x05");
    code.extend((CODE_ALIAS as u32).to_le_bytes());
    code.extend(b"\xf4\xf4");
    for setup in [PAGED, PAGED_USER] {
        for subject in [
            Subject::Mode(MachineJitMode::Forced),
            Subject::Background(10),
        ] {
            let (vm, mut looping) = machine(
                JitBackend::new(),
                CpuModel::I80386,
                &code,
                subject,
                setup,
                1,
            );
            let mut patcher = vm.new_vcpu();
            patcher.cpu = looping.cpu.clone();
            patcher.cpu.rip = 0x7000;
            let started = std::sync::atomic::AtomicBool::new(false);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    while !started.load(std::sync::atomic::Ordering::Acquire) {
                        std::thread::yield_now();
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    let run = patcher.run_steps(&vm, 100, StepOptions::default()).unwrap();
                    assert!(
                        matches!(
                            run.stop(),
                            StepStop::Exit(Exit::Hlt | Exit::Exception { vector: 13, .. })
                        ),
                        "{run:?}"
                    );
                });
                let mut stopped = false;
                for _ in 0..100_000 {
                    let run = looping
                        .run_steps(&vm, 100_000, StepOptions::default())
                        .unwrap();
                    started.store(true, std::sync::atomic::Ordering::Release);
                    if !matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)) {
                        stopped = true;
                        break;
                    }
                }
                assert!(stopped, "{subject:?}: the patch was never seen");
            });
            assert!(
                looping.machine_jit_counters_for_test().paged_entries > 0,
                "{subject:?}"
            );
        }
    }
}

/// Decision-32 §9, D-bit races: two vCPUs write the same 32 clean pages at
/// once, each its own dword of each page. Every first write defers in
/// compiled code, and each vCPU's walk sets D by a locked update that keeps
/// the other's (SDM Vol 3A §11.1.2.1 p. 11-3; 80386 PRM §11.2.2 p. 11-5).
#[test]
fn two_vcpus_race_to_set_dirty_bits() {
    let code = |offset: i32| {
        asm(move |a| {
            let mut outer = a.create_label();
            let mut inner = a.create_label();
            a.mov(edx, 20u32).unwrap();
            a.set_label(&mut outer).unwrap();
            a.mov(esi, 0x60000u32).unwrap();
            a.mov(ecx, 32u32).unwrap();
            a.set_label(&mut inner).unwrap();
            a.add(dword_ptr(esi + offset), ecx).unwrap();
            a.add(esi, 0x1000).unwrap();
            a.dec(ecx).unwrap();
            a.jne(inner).unwrap();
            a.dec(edx).unwrap();
            a.jne(outer).unwrap();
            a.hlt().unwrap();
        })
    };
    let (first_code, second_code) = (code(0), code(4));
    let mut both = first_code.clone();
    both.resize(0x800, 0xf4);
    both.extend(&second_code);
    for subject in [
        Subject::Mode(MachineJitMode::Forced),
        Subject::Background(2),
    ] {
        let (vm, mut first) = machine(
            JitBackend::new(),
            CpuModel::I80486Dx,
            &both,
            subject,
            PAGED,
            1,
        );
        let mut second = vm.new_vcpu();
        second.cpu = first.cpu.clone();
        second.cpu.rip = 0x1800;
        std::thread::scope(|scope| {
            for cpu in [&mut first, &mut second] {
                let vm = &vm;
                scope.spawn(move || {
                    let run = cpu
                        .run_steps(vm, 1_000_000, StepOptions::default())
                        .unwrap();
                    assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)), "{run:?}");
                });
            }
        });
        for page in 0..32u64 {
            let mut bytes = [0; 8];
            vm.read_bytes(0x60000 + 0x1000 * page, &mut bytes).unwrap();
            // ECX is 32 - page there, 20 times, for each vCPU's dword.
            let expected = 20 * (32 - page as u32);
            assert_eq!(
                (
                    u32::from_le_bytes(bytes[..4].try_into().unwrap()),
                    u32::from_le_bytes(bytes[4..].try_into().unwrap())
                ),
                (expected, expected),
                "{subject:?} page {page}"
            );
            let mut entry = [0; 4];
            vm.read_bytes(0x4000 + 4 * (0x60 + page), &mut entry)
                .unwrap();
            assert_eq!(
                u32::from_le_bytes(entry) & 0x60,
                0x60,
                "{subject:?}: page {page} lost A or D"
            );
        }
        // The worker's timing decides whether a background unit lands in
        // this short run; forced units always do.
        if subject == Subject::Mode(MachineJitMode::Forced) {
            for cpu in [&first, &second] {
                let c = cpu.machine_jit_counters_for_test();
                assert!(c.paged_entries > 0, "{subject:?}: {c:?}");
            }
        }
    }
}

/// Decision-32 §3: a supervisor store to a read-only page completes with
/// WP clear (80386 PRM §§6.4.1-6.4.2 Table 6-5), so the compiled view may
/// hold a supervisor write tag for it. Setting CR0.WP (Intel486 PRM §4.1.3
/// p. 4-7; SDM Vol 3A §5.6.1 pp. 5-33-35) must take that tag away: the next
/// store to the page faults (#PF, P and W/R set), in both tiers.
#[test]
fn a_wp_toggle_revokes_supervisor_write_tags() {
    let page = SUPERVISOR_READ_ONLY as u32 + 0x800;
    let code = asm(move |a| {
        let mut top = a.create_label();
        a.mov(edi, 20u32).unwrap();
        a.set_label(&mut top).unwrap();
        a.add(dword_ptr(page), edi).unwrap();
        a.inc(ebx).unwrap();
        a.dec(edi).unwrap();
        a.jne(top).unwrap();
        a.mov(eax, cr0).unwrap();
        a.or(eax, 1 << 16).unwrap();
        a.mov(cr0, eax).unwrap();
        a.inc(ecx).unwrap();
        a.mov(dword_ptr(page), ebx).unwrap();
        a.hlt().unwrap();
    });
    for subject in MODES
        .into_iter()
        .map(Subject::Mode)
        .chain([Subject::Background(1)])
    {
        for budget in [1u64, 5, 64, 1000] {
            let what = format!("wp toggle {subject:?} {budget}");
            let (c, stop) = lockstep(
                CpuModel::I80486Dx,
                &code,
                subject,
                PAGED,
                1,
                &plain,
                &mut |_| budget,
                &what,
            );
            assert!(
                matches!(stop, StepStop::Exit(Exit::Exception { vector: 14, .. })),
                "{what}: {stop:?}"
            );
            if subject == Subject::Mode(MachineJitMode::Forced) && budget == 1000 {
                assert!(c.compiled_retired > 40, "{what}: {c:?}");
            }
        }
    }
}
