//! The interpreter's fused real-mode stack sequences (xj-to9.29) against the
//! unfused walk.
//!
//! A real-mode PUSH, POP, CALL or RET with a sixteen-bit SP lifts to one
//! header op in front of its unchanged sequence (`lift::fused`), and the walk
//! runs the header instead of the sequence. That is an equivalence, not a
//! hardware fact: each program here runs twice, with the headers and without
//! (`set_stack_fusion_for_test`), on the Real16 alias, the twenty-bit Machine
//! paths and the Direct Machine paths, and every stop, the registers, the
//! flags, the retired count and the stack and data memory must agree. The
//! census covers register, segment-register, memory and immediate pushes and
//! pops, PUSH SP and POP SP, near CALL (direct, register and memory), RET and
//! RET imm16, operand-size prefixed pushes, CALL and RET, POP SS and the
//! instruction it holds, a flag producer before a push whose flags a later
//! branch reads, and a LOOP around a push and pop; at SP values that wrap or
//! fault at the segment end, on a trapped (device) stack whose stops are
//! answered and replayed, an unmapped stack, a stack word across two RAM
//! regions, a watched stack and a stack inside the running code; cut into
//! budgets of one, two, three and many instructions; and with TF set. A
//! counter of the headers the walks ran shows that every path takes them,
//! and only when fused.
use x86jit_core::{
    BlockKey, CachedBlock, CpuMode, CpuModel, CpuState, ExceptionDelivery, Exit, IrOp, Prot, Reg,
    RegionKind, StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

const CODE: u64 = 0x1000;
const HALT: u64 = 0x0f00;
const IRET: u64 = 0x0f10;
const DATA: u64 = 0x3000;
const RAM_SS: u16 = 0x2000;
const TRAP_SS: u16 = 0x5000;
const UNMAPPED_SS: u16 = 0x7000;
/// The RAM below 28000h and above it are two regions: a stack word at
/// 27FFFh crosses them and takes the walk's retry after a containment miss.
const SPLIT: u64 = 0x2_8000;
/// SS for a stack inside the running code: the pushes overwrite the program.
const CODE_SS: u16 = 0;

const PROGRAMS: [(&str, &[u8]); 12] = [
    // PUSH AX; PUSH BX; POP CX; POP DX; ADD AX,BX; HLT
    ("registers", b"\x50\x53\x59\x5a\x01\xd8\xf4"),
    // CALL +2; HLT; NOP; callee: PUSH AX; POP AX; RET
    ("call-ret", b"\xe8\x02\x00\xf4\x90\x50\x58\xc3"),
    // PUSH AX; CALL +2; HLT; NOP; callee: RET 2
    ("ret-imm", b"\x50\xe8\x02\x00\xf4\x90\xc2\x02\x00"),
    // PUSH SP; POP SP; PUSH SS; PUSH DS; PUSH ES; POP DS; POP ES; PUSH CS; POP AX; HLT
    ("segments", b"\x54\x5c\x16\x1e\x06\x1f\x07\x0e\x58\xf4"),
    // PUSH [BX]; POP [BX+2]; HLT
    ("memory", b"\xff\x37\x8f\x47\x02\xf4"),
    // CMP AX,BX; PUSH AX; POP CX; JNZ +0; PUSHF; POP DX; HLT
    ("flags", b"\x39\xd8\x50\x59\x75\x00\x9c\x5a\xf4"),
    // PUSH 1234h; POP BX; PUSH -80h; POP CX; HLT
    ("immediate", b"\x68\x34\x12\x5b\x6a\x80\x59\xf4"),
    // PUSH EAX; POP EBX; HLT (an invalid prefix before the 80386)
    ("operand-size", b"\x66\x50\x66\x5b\xf4"),
    // MOV CX,5; L: PUSH AX; POP BX; LOOP L; HLT
    ("loop", b"\xb9\x05\x00\x50\x5b\xe2\xfc\xf4"),
    // MOV BX,1010h; CALL BX; MOV WORD [3010h],1010h; CALL [3010h]; HLT;
    // 1010h: RET
    (
        "call-indirect",
        b"\xbb\x10\x10\xff\xd3\xc7\x06\x10\x30\x10\x10\xff\x16\x10\x30\xf4\xc3",
    ),
    // CALL +2 with a 32-bit operand; HLT; NOP; callee: RET with a 32-bit operand
    ("call-ret-32", b"\x66\xe8\x02\x00\x00\x00\xf4\x90\x66\xc3"),
    // PUSH SS; POP SS (interrupts and the trap held for one instruction);
    // PUSH AX; POP AX; HLT
    ("pop-ss", b"\x16\x17\x50\x58\xf4"),
];

const PATHS: [Option<CpuModel>; 6] = [
    None,
    Some(CpuModel::I8088),
    Some(CpuModel::I8086),
    Some(CpuModel::NecV20),
    Some(CpuModel::I80286),
    Some(CpuModel::I80386),
];

#[derive(Debug, PartialEq)]
struct Outcome {
    stops: Vec<(u64, String)>,
    gpr: [u64; 16],
    rip: u64,
    flags: String,
    rflags_system: u64,
    segments: [u16; 6],
    retired: u64,
    stack: Vec<u8>,
    data: Vec<u8>,
    code: Vec<u8>,
    dirty: Vec<(u64, u64)>,
}

/// One run's stack and stepping.
#[derive(Clone, Copy, Debug)]
struct Case {
    ss: u16,
    sp: u64,
    slice: u64,
    tf: bool,
    /// Watch the stack segment for host-visible writes.
    watch: bool,
}

fn setup(path: Option<CpuModel>, code: &[u8], ss: u16, sp: u64, tf: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x6_0000));
    match path {
        Some(model) => vm.set_cpu_model(model),
        None => vm.set_cpu_mode(CpuMode::Real16),
    }
    vm.set_exception_delivery(ExceptionDelivery::InGuest);
    vm.map(0, SPLIT as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(SPLIT, 0x4_0000 - SPLIT as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(0x5_0000, 0x1_0000, Prot::RW, RegionKind::Trap)
        .unwrap();
    // A trap after HLT returns past it: more HLTs follow the program.
    vm.write_bytes(CODE, &[0xf4; 0x40]).unwrap();
    vm.write_bytes(CODE, code).unwrap();
    vm.write_bytes(HALT, b"\xf4").unwrap();
    vm.write_bytes(IRET, b"\xcf").unwrap();
    let ivt: Vec<u8> = (0..256u64)
        .flat_map(|vector| (if vector == 1 { IRET } else { HALT } as u32).to_le_bytes())
        .collect();
    vm.write_bytes(0, &ivt).unwrap();
    vm.write_bytes(DATA, &[0x5a, 0xa5, 0x11, 0x22]).unwrap();
    let mut cpu = vm.new_vcpu();
    for (reg, value) in [
        (Reg::Cs, 0),
        (Reg::Rip, CODE),
        (Reg::Ds, 0),
        (Reg::Es, 0),
        (Reg::Ss, u64::from(ss)),
        (Reg::Rsp, sp),
        (Reg::Rax, 0x1234_8001),
        (Reg::Rbx, DATA),
        (Reg::Rcx, 0x0007_0000),
        (Reg::Rdx, 0x00ff_00ff),
    ] {
        cpu.set_reg(reg, value);
    }
    if path.is_some() {
        // set_reg changes visible selectors only; seed the hidden caches.
        let c = &cpu.cpu;
        cpu.cpu.legacy = CpuState::legacy_real_entry([c.cs, c.ds, c.es, c.ss, c.fs, c.gs]).legacy;
    }
    if tf {
        cpu.cpu.rflags_system |= 0x100;
    }
    (vm, cpu)
}

fn run(path: Option<CpuModel>, code: &[u8], case: Case, fused: bool) -> Outcome {
    let Case {
        ss,
        sp,
        slice,
        tf,
        watch,
    } = case;
    x86jit_core::lift::set_stack_fusion_for_test(fused);
    let (vm, mut cpu) = setup(path, code, ss, sp, tf);
    let base = u64::from(ss) << 4;
    if watch {
        vm.watch_range(base, 0x1_0000);
    }
    let mut stops = Vec::new();
    let mut steps = 0;
    while steps < 100 && stops.len() < 200 {
        let Ok(result) = cpu.run_steps(&vm, slice.min(100 - steps), StepOptions::default()) else {
            stops.push((0, "error".to_string()));
            break;
        };
        let stop = format!("{:?}", result.stop());
        steps += result.steps();
        stops.push((result.steps(), stop));
        // A device stop is answered and the instruction replayed.
        match result.stop() {
            StepStop::Exit(Exit::BudgetExhausted) if result.steps() != 0 => {}
            StepStop::Exit(Exit::MmioRead { addr, .. }) => {
                let value = 0xbe00 | (*addr & 0xff);
                cpu.complete_mmio_read(value);
            }
            StepStop::Exit(Exit::MmioWrite { .. }) => cpu.complete_mmio_write(),
            _ => break,
        }
    }
    x86jit_core::lift::set_stack_fusion_for_test(true);
    let mut stack = vec![0; 0x200];
    if base + 0x1_0000 <= 0x4_0000 {
        vm.read_bytes(base + 0xff00, &mut stack[..0x100]).unwrap();
        vm.read_bytes(base, &mut stack[0x100..]).unwrap();
    }
    let mut data = vec![0; 0x18];
    vm.read_bytes(DATA, &mut data).unwrap();
    let mut code = vec![0; 0x40];
    vm.read_bytes(CODE, &mut code).unwrap();
    let c = &cpu.cpu;
    Outcome {
        stops,
        gpr: c.gpr,
        rip: c.rip,
        flags: format!("{:?}", c.flags),
        rflags_system: c.rflags_system,
        segments: [c.cs, c.ds, c.es, c.ss, c.fs, c.gs],
        retired: cpu.retired_instructions(),
        stack,
        data,
        code,
        dirty: vm.take_dirty_ranges(),
    }
}

/// Every program on `path`: each stack case at budgets of one and many
/// instructions, budgets of two and three on the ordinary stack, TF set, a
/// watched stack, a stack word across two RAM regions, and a stack inside
/// the running code.
fn census(path: Option<CpuModel>) {
    let case = |ss, sp, slice| Case {
        ss,
        sp,
        slice,
        tf: false,
        watch: false,
    };
    let mut cases = Vec::new();
    for sp in [0x100, 0, 1, 0xffff] {
        cases.push(case(RAM_SS, sp, 1));
        cases.push(case(RAM_SS, sp, 1000));
    }
    for (ss, sp) in [
        (TRAP_SS, 0x100),
        (UNMAPPED_SS, 0x100),
        // 2000h:8001h: the first push writes 27FFFh and 28000h.
        (RAM_SS, 0x8001),
        // 0:1008h: the pushes land on the program's own bytes.
        (CODE_SS, CODE + 8),
    ] {
        cases.push(case(ss, sp, 1));
        cases.push(case(ss, sp, 1000));
    }
    cases.push(case(RAM_SS, 0x100, 2));
    cases.push(case(RAM_SS, 0x100, 3));
    for slice in [1, 1000] {
        cases.push(Case {
            tf: true,
            ..case(RAM_SS, 0x100, slice)
        });
        for sp in [0x100, 0x8001] {
            cases.push(Case {
                watch: true,
                ..case(RAM_SS, sp, slice)
            });
        }
    }
    for (name, code) in PROGRAMS {
        for &case in &cases {
            let fused = run(path, code, case, true);
            let unfused = run(path, code, case, false);
            assert_eq!(fused, unfused, "{path:?} {name} {case:?}");
        }
    }
}

#[test]
fn alias_matches_the_unfused_walk() {
    census(None);
}

#[test]
fn i8088_matches_the_unfused_walk() {
    census(Some(CpuModel::I8088));
}

#[test]
fn i8086_matches_the_unfused_walk() {
    census(Some(CpuModel::I8086));
}

#[test]
fn nec_v20_matches_the_unfused_walk() {
    census(Some(CpuModel::NecV20));
}

#[test]
fn i80286_matches_the_unfused_walk() {
    census(Some(CpuModel::I80286));
}

#[test]
fn i80386_matches_the_unfused_walk() {
    census(Some(CpuModel::I80386));
}

#[test]
fn every_path_runs_the_headers_and_only_when_fused() {
    // The equivalence above is vacuous unless the fused runs take the
    // headers: count the headers each walk ran in place of its sequence.
    let registers = PROGRAMS[0].1;
    let call_ret = PROGRAMS[1].1;
    let case = Case {
        ss: RAM_SS,
        sp: 0x100,
        slice: 1000,
        tf: false,
        watch: false,
    };
    for path in PATHS {
        for (code, headers) in [(registers, 4), (call_ret, 4)] {
            for fused in [true, false] {
                x86jit_core::lift::take_fused_headers_for_test();
                let outcome = run(path, code, case, fused);
                assert_eq!(outcome.stops.last().unwrap().1, "Exit(Hlt)", "{path:?}");
                assert_eq!(
                    x86jit_core::lift::take_fused_headers_for_test(),
                    if fused { headers } else { 0 },
                    "{path:?} fused {fused}"
                );
            }
        }
    }
}

fn headers(ops: &[IrOp]) -> usize {
    ops.iter()
        .filter(|op| {
            matches!(
                op,
                IrOp::Legacy(
                    x86jit_core::legacy::LegacyOp::FusedPush { .. }
                        | x86jit_core::legacy::LegacyOp::FusedPop { .. }
                        | x86jit_core::legacy::LegacyOp::FusedPush20 { .. }
                        | x86jit_core::legacy::LegacyOp::FusedPop20 { .. }
                )
            )
        })
        .count()
}

#[test]
fn every_path_fuses_the_register_pushes_and_pops() {
    // The equivalence above is vacuous unless the fused runs take the headers.
    let (_, code) = PROGRAMS[0];
    for path in PATHS {
        let (vm, mut cpu) = setup(path, code, RAM_SS, 0x100, false);
        cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        let count = match path {
            None => headers(
                &x86jit_core::lift::lift_block(
                    &vm.mem,
                    x86jit_core::lift::FetchAddr::real16(0, CODE as u16),
                    CpuMode::Real16,
                )
                .unwrap()
                .ops,
            ),
            Some(_) => [true, false]
                .into_iter()
                .find_map(
                    |a20| match vm.cache.get(BlockKey::machine_direct(CODE, 0, a20)) {
                        Some(CachedBlock::Interpreted(ir)) => Some(headers(&ir.ops)),
                        _ => None,
                    },
                )
                .expect("Machine block cached"),
        };
        assert_eq!(count, 4, "{path:?}");
    }
}
