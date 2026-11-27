//! Decision-31 (J2b): compiled Machine blocks chained without a dispatcher
//! turn between them, against the interpreter in lockstep at every
//! `run_steps` call (`machine_j1_lockstep`'s comparison: the full state, the
//! written RAM, the step result and the retired count), in every compile mode
//! with chains on and with the test-only chain-off switch.
//!
//! The programs cover every chained exit of decision-31 §3: JMP, Jcc, the
//! LOOP family, direct CALL, a fall-through at the lifter's bound and into a
//! singleton, near RET and RET imm16 through the shadow ring (recursion deeper
//! than its 64 frames, a rewritten return address, a stack switch, a
//! `longjmp` past pending returns), cross-page edges and a twelve-block loop.
//! Budgets cut chains at every position. Further witnesses raise the dispatch
//! word and retire units at a chosen edge, run two vCPUs with A20 on and off
//! over the same units, and run vCPUs on threads over slots they share while
//! another thread rewrites the code they chain through.
use super::machine_j1_lockstep::{
    add, answer, asm, lockstep_subject, machine, same, same_memory, Setup, Subject, MMIO, MODELS,
};
use iced_x86::code_asm::*;
use x86jit_core::{
    ChainEdgeAction, CpuModel, Exit, MachineJitCounters, MachineJitMode, StepOptions, StepStop,
    Vcpu,
};
use x86jit_cranelift::JitBackend;

const NAMES: [&str; 9] = MachineJitCounters::CHAIN_END_NAMES;

fn reason(name: &str) -> usize {
    NAMES.iter().position(|n| *n == name).unwrap()
}

/// Bytes for code assembled at several addresses, from 1000h on: each piece
/// is `(address, code)`, and the gaps hold HLT.
fn pieces(parts: &[(u64, Vec<u8>)]) -> Vec<u8> {
    let end = parts
        .iter()
        .map(|(at, code)| at + code.len() as u64)
        .max()
        .unwrap();
    let mut bytes = vec![0xf4; (end - 0x1000) as usize];
    for (at, code) in parts {
        let at = (*at - 0x1000) as usize;
        bytes[at..at + code.len()].copy_from_slice(code);
    }
    bytes
}

fn asm_at(at: u64, build: impl Fn(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(32).unwrap();
    build(&mut a);
    a.assemble(at).unwrap()
}

/// Twelve blocks per iteration: eleven `add; xor; jmp next` blocks, then
/// `dec ebp; jnz top`.
fn twelve_blocks(iterations: u32) -> Vec<u8> {
    asm(|a| {
        let mut top = a.create_label();
        a.mov(ebp, iterations).unwrap();
        a.set_label(&mut top).unwrap();
        for i in 0..11u32 {
            let mut next = a.create_label();
            a.add(eax, 0x1111 * (i + 1)).unwrap();
            a.xor(ebx, eax).unwrap();
            a.jmp(next).unwrap();
            // Padding between blocks: never executed.
            a.int3().unwrap();
            a.set_label(&mut next).unwrap();
        }
        a.dec(ebp).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
    })
}

/// The programs, each exercising a chained exit of decision-31 §3.
fn programs() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("twelve-blocks", twelve_blocks(6)),
        (
            "self-loop",
            asm(|a| {
                let mut top = a.create_label();
                a.mov(ebp, 40u32).unwrap();
                a.set_label(&mut top).unwrap();
                a.inc(eax).unwrap();
                a.add(ebx, eax).unwrap();
                a.dec(ebp).unwrap();
                a.jne(top).unwrap();
                a.hlt().unwrap();
            }),
        ),
        // CALL / RET / JNZ: three blocks per iteration.
        (
            "triangle",
            asm(|a| {
                let mut top = a.create_label();
                let mut sub = a.create_label();
                a.mov(ebp, 20u32).unwrap();
                a.set_label(&mut top).unwrap();
                a.call(sub).unwrap();
                a.dec(ebp).unwrap();
                a.jne(top).unwrap();
                a.hlt().unwrap();
                a.set_label(&mut sub).unwrap();
                a.inc(eax).unwrap();
                a.add(ebx, eax).unwrap();
                a.ret().unwrap();
            }),
        ),
        // RET imm16 releases the argument the caller pushed.
        (
            "ret-imm16",
            asm(|a| {
                let mut top = a.create_label();
                let mut sub = a.create_label();
                a.mov(ebp, 12u32).unwrap();
                a.set_label(&mut top).unwrap();
                a.push(eax).unwrap();
                a.call(sub).unwrap();
                a.dec(ebp).unwrap();
                a.jne(top).unwrap();
                a.hlt().unwrap();
                a.set_label(&mut sub).unwrap();
                a.add(eax, dword_ptr(esp + 4)).unwrap();
                a.inc(eax).unwrap();
                a.ret_1(4).unwrap();
            }),
        ),
        // 16-bit operand-size CALL and RET in 32-bit code (66h): a CALL
        // rel16 pushes a 16-bit return IP, RET pops one, RET imm16 also
        // releases a 16-bit argument (SDM Vol 2A CALL, RET: the operand-size
        // attribute selects a 16-bit IP push and pop).
        ("call-ret-o16", {
            let main = vec![
                0xbd, 0x0f, 0x00, 0x00, 0x00, // 1000: mov ebp, 15
                0x66, 0x50, // 1005: push ax
                0x66, 0xe8, 0xf5, 0x00, // 1007: call 1100h (o16)
                0x66, 0xe8, 0x71, 0x01, // 100b: call 1180h (o16)
                0x4d, // 100f: dec ebp
                0x75, 0xf3, // 1010: jnz 1005h
                0xf4, // 1012: hlt
            ];
            let first = asm_at(0x1100, |a| {
                a.inc(eax).unwrap();
                a.add(ebx, eax).unwrap();
                a.db(&[0x66, 0xc2, 0x02, 0x00]).unwrap(); // ret 2 (o16)
            });
            let second = asm_at(0x1180, |a| {
                a.add(ecx, ebx).unwrap();
                a.xor(edx, 0x55u32).unwrap();
                a.db(&[0x66, 0xc3]).unwrap(); // ret (o16)
            });
            pieces(&[(0x1000, main), (0x1100, first), (0x1180, second)])
        }),
        // Recursion 80 deep: more frames than the ring's 64.
        (
            "deep-recursion",
            asm(|a| {
                let mut top = a.create_label();
                let mut rec = a.create_label();
                let mut base = a.create_label();
                a.mov(ebp, 3u32).unwrap();
                a.set_label(&mut top).unwrap();
                a.mov(ecx, 80u32).unwrap();
                a.call(rec).unwrap();
                a.dec(ebp).unwrap();
                a.jne(top).unwrap();
                a.hlt().unwrap();
                a.set_label(&mut rec).unwrap();
                a.add(eax, ecx).unwrap();
                a.dec(ecx).unwrap();
                a.je(base).unwrap();
                a.call(rec).unwrap();
                a.set_label(&mut base).unwrap();
                a.xor(ebx, eax).unwrap();
                a.ret().unwrap();
            }),
        ),
        // The callee moves its return address past the caller's next
        // instruction: every return mispredicts.
        (
            "rewritten-return",
            asm(|a| {
                let mut top = a.create_label();
                let mut sub = a.create_label();
                a.mov(ebp, 12u32).unwrap();
                a.set_label(&mut top).unwrap();
                a.call(sub).unwrap();
                a.inc(ebx).unwrap();
                a.dec(ebp).unwrap();
                a.jne(top).unwrap();
                a.hlt().unwrap();
                a.set_label(&mut sub).unwrap();
                a.inc(eax).unwrap();
                a.inc(dword_ptr(esp)).unwrap();
                a.ret().unwrap();
            }),
        ),
        // A callee that calls on another stack and switches back, then a
        // `longjmp` from a nested callee past two pending returns.
        (
            "stack-switch-longjmp",
            asm(|a| {
                let mut top = a.create_label();
                let mut after = a.create_label();
                let mut f = a.create_label();
                let mut g = a.create_label();
                let mut h = a.create_label();
                a.mov(ebp, 10u32).unwrap();
                a.set_label(&mut top).unwrap();
                a.mov(esi, esp).unwrap();
                a.call(f).unwrap();
                a.set_label(&mut after).unwrap();
                a.dec(ebp).unwrap();
                a.jne(top).unwrap();
                a.hlt().unwrap();
                a.set_label(&mut f).unwrap();
                a.mov(edi, esp).unwrap();
                a.mov(esp, 0x7800u32).unwrap();
                a.call(h).unwrap();
                a.mov(esp, edi).unwrap();
                a.call(g).unwrap();
                a.ret().unwrap();
                a.set_label(&mut g).unwrap();
                a.add(ebx, eax).unwrap();
                a.mov(esp, esi).unwrap();
                a.jmp(after).unwrap();
                a.set_label(&mut h).unwrap();
                a.inc(ecx).unwrap();
                a.ret().unwrap();
            }),
        ),
        // LOOP, LOOPE, LOOPNE and JECXZ, both arms.
        (
            "loop-family",
            asm(|a| {
                let mut top = a.create_label();
                let mut inner = a.create_label();
                let mut second = a.create_label();
                let mut skip = a.create_label();
                a.mov(ebp, 6u32).unwrap();
                a.set_label(&mut top).unwrap();
                a.mov(ecx, 5u32).unwrap();
                a.set_label(&mut inner).unwrap();
                a.add(eax, ecx).unwrap();
                a.loop_(inner).unwrap();
                a.mov(ecx, 4u32).unwrap();
                a.set_label(&mut second).unwrap();
                a.inc(edx).unwrap();
                a.cmp(edx, 1000u32).unwrap();
                a.loopne(second).unwrap();
                a.jecxz(skip).unwrap();
                a.inc(ebx).unwrap();
                a.set_label(&mut skip).unwrap();
                a.cmp(eax, eax).unwrap();
                a.mov(ecx, 2u32).unwrap();
                let mut third = a.create_label();
                a.set_label(&mut third).unwrap();
                a.loope(third).unwrap();
                a.dec(ebp).unwrap();
                a.jne(top).unwrap();
                a.hlt().unwrap();
            }),
        ),
        // Forty lowered instructions: the block is cut at the lifter's bound
        // and falls through into the rest.
        (
            "fall-through",
            asm(|a| {
                let mut top = a.create_label();
                a.mov(ebp, 6u32).unwrap();
                a.set_label(&mut top).unwrap();
                for i in 0..40u32 {
                    a.add(eax, i + 1).unwrap();
                }
                a.dec(ebp).unwrap();
                a.jne(top).unwrap();
                a.hlt().unwrap();
            }),
        ),
        // A block that falls through into a singleton (CLI), never compiled.
        (
            "fall-into-singleton",
            asm(|a| {
                let mut top = a.create_label();
                a.mov(ebp, 10u32).unwrap();
                a.set_label(&mut top).unwrap();
                a.inc(eax).unwrap();
                a.add(ebx, eax).unwrap();
                a.cli().unwrap();
                a.dec(ebp).unwrap();
                a.jne(top).unwrap();
                a.hlt().unwrap();
            }),
        ),
        // Blocks in three pages, one of them straddling a page boundary,
        // joined by direct jumps and a call.
        ("cross-page", {
            let start = asm_at(0x1000, |a| {
                a.mov(ebp, 10u32).unwrap();
                a.add(eax, 3).unwrap();
                a.jmp(0x3ff8u64).unwrap();
            });
            let straddle = asm_at(0x3ff8, |a| {
                a.add(ebx, eax).unwrap();
                a.add(ecx, ebx).unwrap();
                a.xor(edx, ecx).unwrap();
                a.call(0x5800u64).unwrap();
                a.dec(ebp).unwrap();
                a.jne(0x1005u64).unwrap();
                a.hlt().unwrap();
            });
            let callee = asm_at(0x5800, |a| {
                a.inc(esi).unwrap();
                a.ret().unwrap();
            });
            pieces(&[(0x1000, start), (0x3ff8, straddle), (0x5800, callee)])
        }),
        // An indirect JMP and CALL (unchained, Q9), and a block that defers
        // a 16-bit SHLD after chained blocks.
        ("indirect-and-defer", {
            let main = asm_at(0x1000, |a| {
                a.mov(ebp, 10u32).unwrap();
                a.mov(ecx, 0x1800u32).unwrap();
                a.mov(edx, 0x1900u32).unwrap();
                a.inc(eax).unwrap();
                a.jmp(ecx).unwrap();
            });
            let target = asm_at(0x1800, |a| {
                a.add(ebx, eax).unwrap();
                a.call(edx).unwrap();
                a.inc(esi).unwrap();
                a.shld(si, di, 3).unwrap();
                a.inc(edi).unwrap();
                a.dec(ebp).unwrap();
                a.jne(0x100fu64).unwrap();
                a.hlt().unwrap();
            });
            let callee = asm_at(0x1900, |a| {
                a.xor(ebx, 0x77u32).unwrap();
                a.ret().unwrap();
            });
            pieces(&[(0x1000, main), (0x1800, target), (0x1900, callee)])
        }),
        // A chained loop that stores into its own next block's immediate.
        (
            "own-code-store",
            asm(|a| {
                let mut top = a.create_label();
                let mut patched = a.create_label();
                a.mov(ebp, 12u32).unwrap();
                a.set_label(&mut top).unwrap();
                a.inc(eax).unwrap();
                a.jmp(patched).unwrap();
                a.set_label(&mut patched).unwrap();
                a.add(ebx, 0x1234_5678u32).unwrap();
                a.mov(byte_ptr(esi), al).unwrap();
                a.dec(ebp).unwrap();
                a.jne(top).unwrap();
                a.hlt().unwrap();
            }),
        ),
    ]
}

/// The `own-code-store` program's ESI: the immediate of its `add ebx`.
fn setup_registers(name: &str, cpu: &mut Vcpu, code: &[u8]) {
    if name == "own-code-store" {
        // mov ebp,imm32 (5), inc eax (1), jmp rel8 (2), then 81 C3 imm32.
        assert_eq!(&code[8..10], &[0x81, 0xc3]);
        cpu.cpu.gpr[6] = 0x1000 + 10;
    }
}

/// `lockstep_subject`, with the program's registers set on both machines.
fn run(
    model: CpuModel,
    name: &str,
    code: &[u8],
    subject: Subject,
    setup: Setup,
    budgets: &mut dyn FnMut(usize) -> u64,
    what: &str,
) -> MachineJitCounters {
    if name != "own-code-store" {
        return lockstep_subject(model, code, subject, setup, 3, budgets, what);
    }
    // ESI must point at the patched immediate on both machines.
    let (bvm, mut b) = machine(
        JitBackend::new(),
        model,
        code,
        Subject::Mode(MachineJitMode::Off),
        setup,
        3,
    );
    let (avm, mut a) = machine(JitBackend::new(), model, code, subject, setup, 3);
    setup_registers(name, &mut a, code);
    setup_registers(name, &mut b, code);
    for call in 0..4000 {
        let budget = budgets(call);
        let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
        let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
        let at = format!("{what} call {call} budget {budget}");
        assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{at}");
        same(&a, &b, &at);
        same_memory(&avm, &bvm, &[(0, MMIO)], &at);
        let go = answer(&mut a, ar.stop());
        answer(&mut b, br.stop());
        if !go {
            break;
        }
    }
    a.machine_jit_counters_for_test()
}

const SUBJECTS: [Subject; 7] = [
    Subject::Mode(MachineJitMode::Forced),
    Subject::Mode(MachineJitMode::Delayed(1)),
    Subject::Mode(MachineJitMode::Delayed(3)),
    Subject::Mode(MachineJitMode::Scripted(7)),
    Subject::Background(1),
    Subject::Background(3),
    Subject::Background(50),
];

/// Decision-31 §8 Lockstep: every program, in every mode, with chains on
/// and off, at budgets 1, 2, 7 and 64 and from every first budget up to two
/// iterations' instructions, equals the interpreter at every call. Chains
/// ran in the compiled modes, and their ends covered every reason the
/// programs can reach.
#[test]
fn chained_blocks_equal_the_interpreter_at_every_call() {
    let mut forced = MachineJitCounters::default();
    let mut scripted = MachineJitCounters::default();
    for model in MODELS {
        for (name, code) in programs() {
            for subject in SUBJECTS {
                for chain_off in [false, true] {
                    let setup = Setup {
                        chain_off,
                        ..Setup::default()
                    };
                    let mut total = MachineJitCounters::default();
                    for constant in [1u64, 2, 7, 64, 100_000] {
                        let what = format!(
                            "{name} {model:?} {subject:?} chain_off={chain_off} budget={constant}"
                        );
                        let c = run(model, name, &code, subject, setup, &mut |_| constant, &what);
                        add(&mut total, c);
                    }
                    // Every first budget, then budgets that cut chains anew.
                    if matches!(
                        subject,
                        Subject::Mode(MachineJitMode::Forced | MachineJitMode::Scripted(_))
                    ) {
                        for first in 0..=72u64 {
                            let what = format!(
                                "{name} {model:?} {subject:?} chain_off={chain_off} first={first}"
                            );
                            let c = run(
                                model,
                                name,
                                &code,
                                subject,
                                setup,
                                &mut |call| {
                                    if call == 0 {
                                        first
                                    } else {
                                        64 + call as u64 % 13
                                    }
                                },
                                &what,
                            );
                            add(&mut total, c);
                        }
                    }
                    if chain_off {
                        assert_eq!(total.chained, 0, "{name} {subject:?}: {total:?}");
                    }
                    match (subject, chain_off) {
                        (Subject::Mode(MachineJitMode::Forced), false) => add(&mut forced, total),
                        (Subject::Mode(MachineJitMode::Scripted(_)), false) => {
                            add(&mut scripted, total)
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    // Done-when 1: chains ran, and the reasons the programs reach occurred.
    assert!(forced.chained > 1000, "{forced:?}");
    assert!(scripted.chained > 1000, "{scripted:?}");
    for name in [
        "budget",
        "empty slot",
        "return miss",
        "unchained",
        "defer",
        "stop",
    ] {
        assert!(
            forced.chain_ends[reason(name)] > 0,
            "forced {name}: {forced:?}"
        );
    }
    for name in ["budget", "empty slot", "unchained", "defer", "stop"] {
        assert!(
            scripted.chain_ends[reason(name)] > 0,
            "scripted {name}: {scripted:?}"
        );
    }
}

/// Decision-31 §8, chains broken mid-run: an exit request, an NMI edge, a
/// foreign code write, a retirement or exhausted code space raised at the
/// k-th edge of a chain ends it there. The run equals the interpreter's run
/// with the same request raised at the retired count the chained run
/// observed, and an exit request stops it exactly there.
#[test]
fn a_request_at_a_chain_edge_ends_the_chain_there() {
    let code = twelve_blocks(300);
    let mut total = MachineJitCounters::default();
    for action in [
        ChainEdgeAction::ExitRequest,
        ChainEdgeAction::NmiEdge,
        ChainEdgeAction::CodeWrite(0x1010),
        ChainEdgeAction::Retire,
        ChainEdgeAction::CodeSpaceExhausted,
    ] {
        for edge in [0u64, 1, 5, 11, 40] {
            for subject in [
                Subject::Mode(MachineJitMode::Forced),
                Subject::Mode(MachineJitMode::Scripted(3)),
            ] {
                let what = format!("{action:?} edge {edge} {subject:?}");
                let model = CpuModel::I80386;
                let (bvm, mut b) = machine(
                    JitBackend::new(),
                    model,
                    &code,
                    Subject::Mode(MachineJitMode::Off),
                    Setup::default(),
                    3,
                );
                let (avm, mut a) = machine(
                    JitBackend::new(),
                    model,
                    &code,
                    subject,
                    Setup::default(),
                    3,
                );
                // Warm: the slots of the loop are filled.
                let warm_a = a.run_steps(&avm, 200, StepOptions::default()).unwrap();
                let warm_b = b.run_steps(&bvm, 200, StepOptions::default()).unwrap();
                assert_eq!(format!("{warm_a:?}"), format!("{warm_b:?}"), "{what}");
                let start = a.retired_instructions();
                Vcpu::raise_at_chain_edge_for_test(edge, action);
                let ar = a.run_steps(&avm, 100_000, StepOptions::default()).unwrap();
                let Some(raised) = Vcpu::chain_edge_raised_at_for_test() else {
                    panic!("{what}: never raised: {ar:?}");
                };
                assert!(raised >= start, "{what}");
                // The reference: the same request at the same retired count.
                let first = b
                    .run_steps(&bvm, raised - start, StepOptions::default())
                    .unwrap();
                assert!(
                    matches!(first.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                    "{what}: {first:?}"
                );
                match action {
                    ChainEdgeAction::ExitRequest => b.exit_request().request_exit(),
                    ChainEdgeAction::NmiEdge => b.nmi_line().unwrap().raise(),
                    // The bytes are unchanged: nothing the interpreter sees.
                    _ => {}
                }
                let second = b
                    .run_steps(&bvm, 100_000 - (raised - start), StepOptions::default())
                    .unwrap();
                assert_eq!(ar.steps(), first.steps() + second.steps(), "{what}");
                assert_eq!(
                    format!("{:?}", ar.stop()),
                    format!("{:?}", second.stop()),
                    "{what}"
                );
                same(&a, &b, &what);
                if action == ChainEdgeAction::ExitRequest {
                    assert!(
                        matches!(ar.stop(), StepStop::Exit(Exit::Interrupted)),
                        "{what}: {ar:?}"
                    );
                    assert_eq!(ar.steps(), raised - start, "{what}");
                }
                add(&mut total, a.machine_jit_counters_for_test());
            }
        }
    }
    for name in ["word", "epoch"] {
        assert!(total.chain_ends[reason(name)] > 0, "{name}: {total:?}");
    }
}

/// Decision-31 §2.5, §8: two vCPUs with A20 on and off share units and
/// slots; a slot one fills under its flags ends the other's chain (the flags
/// compare). A callee whose caller's A20 changes between runs returns
/// through a ring entry made under the other flags, which ends the chain too.
#[test]
fn the_flags_compare_ends_chains_across_an_a20_change() {
    let code = asm(|a| {
        let mut top = a.create_label();
        let mut sub = a.create_label();
        a.mov(ebp, 400u32).unwrap();
        a.set_label(&mut top).unwrap();
        a.inc(eax).unwrap();
        a.call(sub).unwrap();
        a.dec(ebp).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
        a.set_label(&mut sub).unwrap();
        a.add(ebx, eax).unwrap();
        a.add(ecx, ebx).unwrap();
        a.ret().unwrap();
    });
    let model = CpuModel::I80386;
    let off = Setup {
        a20_off: true,
        ..Setup::default()
    };
    let mut total = MachineJitCounters::default();
    for subject in [
        Subject::Mode(MachineJitMode::Forced),
        Subject::Mode(MachineJitMode::Scripted(5)),
    ] {
        // One VM, two vCPUs, alternating slices: A with A20 on, B off.
        let (avm, mut a) = machine(
            JitBackend::new(),
            model,
            &code,
            subject,
            Setup::default(),
            3,
        );
        let mut b = avm.new_vcpu();
        b.cpu = a.cpu.clone();
        b.set_a20_enabled(false);
        let (rvm, mut ra) = machine(
            JitBackend::new(),
            model,
            &code,
            Subject::Mode(MachineJitMode::Off),
            Setup::default(),
            3,
        );
        let (svm, mut rb) = machine(
            JitBackend::new(),
            model,
            &code,
            Subject::Mode(MachineJitMode::Off),
            off,
            3,
        );
        rb.cpu = ra.cpu.clone();
        rb.set_a20_enabled(false);
        let mut halted = [false; 2];
        for call in 0..2000 {
            let what = format!("{subject:?} call {call}");
            for (index, (cpu, reference, rvm)) in [(&mut a, &mut ra, &rvm), (&mut b, &mut rb, &svm)]
                .into_iter()
                .enumerate()
            {
                if halted[index] {
                    continue;
                }
                let x = cpu.run_steps(&avm, 37, StepOptions::default()).unwrap();
                let y = reference
                    .run_steps(rvm, 37, StepOptions::default())
                    .unwrap();
                assert_eq!(format!("{x:?}"), format!("{y:?}"), "{what}");
                same(cpu, reference, &what);
                halted[index] = matches!(x.stop(), StepStop::Exit(Exit::Hlt));
            }
            if halted == [true; 2] {
                break;
            }
            // Between runs, A's A20 line changes: its pending returns were
            // pushed under the other flags.
            if call % 3 == 2 {
                let on = a.cpu.legacy.a20_enabled == 0;
                a.set_a20_enabled(on);
                ra.set_a20_enabled(on);
            }
        }
        assert_eq!(halted, [true; 2], "{subject:?}");
        add(&mut total, a.machine_jit_counters_for_test());
        add(&mut total, b.machine_jit_counters_for_test());
    }
    assert!(total.chained > 0, "{total:?}");
    assert!(total.chain_ends[reason("flags")] > 0, "{total:?}");
}

/// Decision-31 §8 Threads: vCPUs on two threads chain through the same
/// shared slots while a third rewrites the code they chain through with its
/// own bytes, retiring their units again and again. Each vCPU's result equals
/// the interpreter's, and the second follows slots the first filled.
#[test]
fn vcpus_on_threads_share_slots_while_their_code_is_rewritten() {
    let code = twelve_blocks(20_000);
    let model = CpuModel::I80386;
    for subject in [
        Subject::Mode(MachineJitMode::Forced),
        Subject::Background(3),
    ] {
        let (vm, mut first) = machine(
            JitBackend::new(),
            model,
            &code,
            subject,
            Setup::default(),
            3,
        );
        let mut second = vm.new_vcpu();
        second.cpu = first.cpu.clone();
        let (rvm, mut reference) = machine(
            JitBackend::new(),
            model,
            &code,
            Subject::Mode(MachineJitMode::Off),
            Setup::default(),
            3,
        );
        let expected = loop {
            let run = reference
                .run_steps(&rvm, 1_000_000, StepOptions::default())
                .unwrap();
            if matches!(run.stop(), StepStop::Exit(Exit::Hlt)) {
                break reference.cpu.clone();
            }
        };
        // The first vCPU fills the loop's slots before the second starts.
        first.run_steps(&vm, 5_000, StepOptions::default()).unwrap();
        let done = std::sync::atomic::AtomicU32::new(0);
        std::thread::scope(|scope| {
            for cpu in [&mut first, &mut second] {
                let (vm, done, expected) = (&vm, &done, &expected);
                scope.spawn(move || {
                    for _ in 0..1_000_000 {
                        let run = cpu.run_steps(vm, 64, StepOptions::default()).unwrap();
                        if matches!(run.stop(), StepStop::Exit(Exit::Hlt)) {
                            break;
                        }
                    }
                    assert_eq!(cpu.cpu.gpr[..8], expected.gpr[..8], "{subject:?}");
                    assert_eq!(cpu.cpu.rip, expected.rip, "{subject:?}");
                    done.fetch_add(1, std::sync::atomic::Ordering::Release);
                });
            }
            scope.spawn(|| {
                let bytes = code[0x20..0x40].to_vec();
                while done.load(std::sync::atomic::Ordering::Acquire) < 2 {
                    vm.write_bytes(0x1020, &bytes).unwrap();
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
            });
        });
        let a = first.machine_jit_counters_for_test();
        let b = second.machine_jit_counters_for_test();
        if subject == Subject::Mode(MachineJitMode::Forced) {
            assert!(a.chained > 0, "{a:?}");
            assert!(b.chained > 0, "{b:?}");
        }
    }
}
