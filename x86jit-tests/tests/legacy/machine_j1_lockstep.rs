//! Decision-29 (J1): compiled Machine blocks against the interpreter, in
//! lockstep at every `run_steps` call. Each program runs on two Flat32 PG=0
//! machines with the Cranelift backend: the reference interprets, the subject
//! compiles every eligible block at its first execution (mode 2), enters it a
//! few turns later (mode 3) or runs scripted units (mode 4). After every call
//! the full state (flags included), the written RAM, the step result and the
//! retired count must agree, and all RAM at the end of a run; MMIO and port
//! exits are answered identically on both.
//!
//! The programs cover every op family and width the J1 lowering completes,
//! with operands drawn from a seeded generator; read-modify-write stores, whose
//! repetition after a wrongly committed defer shows in memory; stores into
//! code (an SMC page, a live field, the running block's next instruction,
//! where a DEFER in place of a STOP shows in registers); transfers split
//! across RAM, MMIO and ROM; A20 off, with the 1 MiB wrap; and the entry
//! refusals E3 (TF), E7 (MXCSR mode) and E8 (RF).
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::descriptor;
use iced_x86::code_asm::*;
use x86jit_core::{
    Backend, CpuModel, CpuState, ExceptionDelivery, Exit, MachineJitCounters, MachineJitMode, Prot,
    RegionKind, StepOptions, StepStop, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const MODELS: [CpuModel; 2] = [CpuModel::I80386, CpuModel::I80486Dx];
const SIZE: u64 = 0x120000;
const MMIO: u64 = 0x9000;
const ROM: u64 = 0xa000;
/// All RAM and ROM: everything but the MMIO page.
const SPANS: [(u64, u64); 3] = [(0, MMIO), (ROM, ROM + 0x1000), (ROM + 0x1000, SIZE)];

#[derive(Clone, Copy, Default)]
struct Setup {
    a20_off: bool,
    mxcsr_mode: bool,
    rf: bool,
    tf: bool,
}

fn machine(
    backend: JitBackend,
    model: CpuModel,
    code: &[u8],
    mode: MachineJitMode,
    setup: Setup,
    seed: u64,
) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(SIZE), Box::new(backend));
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.map(ROM, 0x1000, Prot::RX, RegionKind::Rom).unwrap();
    vm.map(
        ROM + 0x1000,
        (SIZE - ROM - 0x1000) as usize,
        Prot::RWX,
        RegionKind::Ram,
    )
    .unwrap();
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.set_machine_jit_for_test(mode);
    let rom: Vec<u8> = (0..0x1000u32).map(|i| (i * 7 + 3) as u8).collect();
    vm.write_bytes(ROM, &rom).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.gdtr.base = 0x2000;
    cpu.cpu.legacy.gdtr.limit = 0x7f;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        cache.base = 0;
        cache.limit = u32::MAX;
        cache.attributes = if i == 0 { 0x409b } else { 0x4093 };
    }
    cpu.cpu.cs = 8;
    cpu.cpu.ds = 16;
    cpu.cpu.es = 16;
    cpu.cpu.ss = 16;
    cpu.cpu.fs = 16;
    cpu.cpu.gs = 16;
    descriptor(&vm, 0x2008, 0, u32::MAX, 0xc09b);
    descriptor(&vm, 0x2010, 0, u32::MAX, 0xc093);
    cpu.set_a20_enabled(!setup.a20_off);
    // Operands from a seeded generator; the stack and EBP stay the program's.
    let mut z = seed;
    let mut next = || {
        z = z
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        z >> 32
    };
    for i in [0, 1, 2, 3, 6, 7] {
        cpu.cpu.gpr[i] = next() & 0xffff_ffff;
    }
    cpu.cpu.gpr[4] = 0x8000;
    let flags = next();
    cpu.cpu.flags.cf = flags & 1 != 0;
    cpu.cpu.flags.zf = flags & 2 != 0;
    cpu.cpu.flags.of = flags & 4 != 0;
    let data: Vec<u8> = (0..0x100).map(|_| next() as u8).collect();
    vm.write_bytes(0x6000, &data).unwrap();
    if setup.mxcsr_mode {
        cpu.cpu.mxcsr |= 1 << 13; // RC = round down (SDM Vol 1 §10.2.3).
    }
    if setup.rf {
        cpu.cpu.rflags_system |= 1 << 16;
    }
    if setup.tf {
        cpu.cpu.rflags_system |= 1 << 8;
    }
    (vm, cpu)
}

/// The registers without the retry ledger's dead entries.
fn live(cpu: &CpuState) -> String {
    let mut cpu = cpu.clone();
    cpu.mmio_parts[usize::from(cpu.mmio_parts_len)..].fill((0, 0));
    format!("{cpu:?}")
}

fn same_memory(avm: &Vm, bvm: &Vm, spans: &[(u64, u64)], what: &str) {
    for &(start, end) in spans {
        let mut am = vec![0; (end - start) as usize];
        let mut bm = vec![0; (end - start) as usize];
        avm.read_bytes(start, &mut am).unwrap();
        bvm.read_bytes(start, &mut bm).unwrap();
        if am == bm {
            continue;
        }
        if let Some(at) = (0..am.len()).find(|&i| am[i] != bm[i]) {
            panic!(
                "{what}: memory differs at {:#x}: {:#x} vs {:#x}",
                start + at as u64,
                am[at],
                bm[at]
            );
        }
    }
}

fn same(a: &Vcpu, b: &Vcpu, what: &str) {
    assert_eq!(live(&a.cpu), live(&b.cpu), "{what}");
    assert_eq!(a.retired_instructions(), b.retired_instructions(), "{what}");
}

/// Answer an exit identically on both machines; false when the run is over.
fn answer(cpu: &mut Vcpu, stop: &StepStop) -> bool {
    match stop {
        StepStop::Exit(Exit::BudgetExhausted) => true,
        StepStop::Exit(Exit::MmioRead { addr, .. }) => {
            cpu.complete_mmio_read(*addr ^ 0x5a5a_a5a5);
            true
        }
        StepStop::Exit(Exit::MmioWrite { .. }) => {
            cpu.complete_mmio_write();
            true
        }
        StepStop::Exit(Exit::PortIo { port, .. }) => {
            cpu.complete_port_in(u64::from(*port) * 3);
            true
        }
        // Outward delivery returns the single-step trap; carry on.
        StepStop::Exit(Exit::Exception { vector: 1, .. }) => true,
        _ => false,
    }
}

/// Run both machines with this budget schedule until the program ends.
fn lockstep(
    model: CpuModel,
    code: &[u8],
    mode: MachineJitMode,
    setup: Setup,
    seed: u64,
    budgets: &mut dyn FnMut(usize) -> u64,
    what: &str,
) -> MachineJitCounters {
    let (bvm, mut b) = machine(
        JitBackend::new(),
        model,
        code,
        MachineJitMode::Off,
        setup,
        seed,
    );
    let (avm, mut a) = machine(JitBackend::new(), model, code, mode, setup, seed);
    for call in 0..4000 {
        let budget = budgets(call);
        let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
        let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
        let at = format!("{what} call {call} budget {budget}");
        assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{at}");
        same(&a, &b, &at);
        // The pages the programs write, at each call; all RAM at the end.
        same_memory(&avm, &bvm, &[(0, MMIO), (0x100000, 0x101000)], &at);
        let go = answer(&mut a, ar.stop());
        answer(&mut b, br.stop());
        same(&a, &b, &at);
        if !go {
            break;
        }
    }
    same_memory(&avm, &bvm, &SPANS, what);
    a.machine_jit_counters_for_test()
}

fn asm(build: impl Fn(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(32).unwrap();
    build(&mut a);
    a.assemble(0x1000).unwrap()
}

/// `mov ebp, 6; L: body; dec ebp; jnz L; hlt`: the body runs six times on the
/// values it leaves behind.
fn looped(body: impl Fn(&mut CodeAssembler)) -> Vec<u8> {
    asm(|a| {
        let mut top = a.create_label();
        a.mov(ebp, 6u32).unwrap();
        a.set_label(&mut top).unwrap();
        body(a);
        a.dec(ebp).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
    })
}

/// Programs whose every instruction but HLT the J1 lowering completes: one per
/// op family, with every width it has.
fn families(model: CpuModel) -> Vec<(&'static str, Vec<u8>)> {
    // BSWAP is an Intel486 instruction (#UD on the 80386).
    let i486 = model != CpuModel::I80386;
    vec![
        (
            "alu",
            looped(|a| {
                a.add(eax, ecx).unwrap();
                a.adc(ebx, edx).unwrap();
                a.sub(esi, eax).unwrap();
                a.sbb(edi, ebx).unwrap();
                a.and(ecx, edx).unwrap();
                a.or(edx, esi).unwrap();
                a.xor(eax, edi).unwrap();
                a.cmp(ebx, ecx).unwrap();
                a.test(eax, edx).unwrap();
                a.add(eax, 0x1234_5678).unwrap();
                a.inc(esi).unwrap();
                a.dec(edi).unwrap();
                a.add(ax, cx).unwrap();
                a.adc(bx, dx).unwrap();
                a.sub(si, ax).unwrap();
                a.sbb(di, 0x7fff).unwrap();
                a.xor(cx, di).unwrap();
                a.add(al, cl).unwrap();
                a.adc(ah, dh).unwrap();
                a.sub(bl, 0x5a).unwrap();
                a.sbb(ch, al).unwrap();
                a.and(dl, bh).unwrap();
                a.or(bh, 0x7e).unwrap();
            }),
        ),
        (
            "shift",
            looped(|a| {
                a.shl(eax, 3).unwrap();
                a.shr(ebx, cl).unwrap();
                a.sar(edx, 1).unwrap();
                a.rol(esi, 5).unwrap();
                a.ror(edi, cl).unwrap();
                a.rcl(eax, 1).unwrap();
                a.rcr(ebx, cl).unwrap();
                a.shl(ax, cl).unwrap();
                a.sar(bx, 7).unwrap();
                a.rol(dx, cl).unwrap();
                a.rcr(si, 3).unwrap();
                a.shl(al, 2).unwrap();
                a.shr(bl, cl).unwrap();
                a.rol(dh, 3).unwrap();
                a.rcr(ah, cl).unwrap();
                a.inc(ecx).unwrap();
            }),
        ),
        (
            "double-shift",
            looped(|a| {
                a.shld(eax, ebx, 5).unwrap();
                a.shrd(edx, esi, cl).unwrap();
                a.add(ecx, 7).unwrap();
            }),
        ),
        // The 16-bit forms stay interpreted (`lowered_op`), but run here too.
        (
            "double-shift-16",
            looped(|a| {
                a.shld(si, di, 3).unwrap();
                a.shrd(ax, bx, cl).unwrap();
                a.add(ecx, 7).unwrap();
            }),
        ),
        (
            "multiply",
            looped(|a| {
                a.mul(ecx).unwrap();
                a.imul_2(ebx, esi).unwrap();
                a.imul_3(edx, eax, 7).unwrap();
                a.mul(cl).unwrap();
                a.imul(cx).unwrap();
                a.imul(esi).unwrap();
                a.imul_2(di, bx).unwrap();
                a.add(ecx, 0x3579).unwrap();
            }),
        ),
        (
            "bits",
            looped(|a| {
                a.or(ecx, 0x10).unwrap();
                a.bsf(eax, ecx).unwrap();
                a.bsr(ebx, ecx).unwrap();
                a.bt(esi, eax).unwrap();
                a.bts(edi, ebx).unwrap();
                a.btr(edx, ecx).unwrap();
                a.btc(esi, edx).unwrap();
                a.bt(ax, cx).unwrap();
                if i486 {
                    a.bswap(eax).unwrap();
                    a.bswap(edi).unwrap();
                }
                a.rol(ecx, 3).unwrap();
            }),
        ),
        (
            "flags",
            looped(|a| {
                a.lahf().unwrap();
                a.add(ah, 0x41).unwrap();
                a.sahf().unwrap();
                a.setb(al).unwrap();
                a.sete(bl).unwrap();
                a.setg(cl).unwrap();
                a.setnp(dl).unwrap();
                a.clc().unwrap();
                a.setb(dh).unwrap();
                a.stc().unwrap();
                a.adc(esi, edi).unwrap();
                a.std().unwrap();
                a.cld().unwrap();
                a.movsx(eax, bl).unwrap();
                a.movzx(edx, cx).unwrap();
                a.movsx(bx, al).unwrap();
                a.movzx(si, dl).unwrap();
                a.xor(edi, eax).unwrap();
            }),
        ),
        (
            "memory-widths",
            looped(|a| {
                a.mov(esi, 0x6000u32).unwrap();
                a.mov(al, byte_ptr(esi)).unwrap();
                a.mov(bx, word_ptr(esi + 2)).unwrap();
                a.mov(byte_ptr(esi + 4), bl).unwrap();
                a.mov(word_ptr(esi + 6), cx).unwrap();
                a.mov(edx, dword_ptr(esi + 9)).unwrap();
                a.movzx(eax, byte_ptr(esi + 1)).unwrap();
                a.movsx(ebx, word_ptr(esi + 3)).unwrap();
                a.mov(dword_ptr(esi + 0x11), edx).unwrap();
                a.add(ecx, eax).unwrap();
            }),
        ),
        // Non-idempotent stores: a defer after a committed write repeats it.
        (
            "read-modify-write",
            looped(|a| {
                a.mov(esi, 0x6000u32).unwrap();
                a.add(dword_ptr(esi), 1).unwrap();
                a.inc(byte_ptr(esi + 4)).unwrap();
                a.sub(word_ptr(esi + 6), 3).unwrap();
                a.xor(dword_ptr(esi + 8), eax).unwrap();
                a.adc(dword_ptr(esi + 12), ecx).unwrap();
                a.shl(dword_ptr(esi + 16), 1).unwrap();
                a.add(dword_ptr(esi + 21), edx).unwrap();
                a.sbb(byte_ptr(esi + 25), bl).unwrap();
                a.add(eax, dword_ptr(esi)).unwrap();
            }),
        ),
    ]
}

/// Programs that defer, stop, fault or exit on purpose.
fn programs(model: CpuModel) -> Vec<(String, Vec<u8>, Setup)> {
    let plain = Setup::default();
    // mov ebp, 8; L: inc eax; add byte [100Eh], 1; mov ebx, imm32 (its
    // immediate is 100Eh); add ecx, ebx; dec ebp; jnz L; hlt. The store
    // patches the running block's next instruction: a DEFER in place of the
    // STOP would run its stale immediate.
    let patch_next = b"\xbd\x08\x00\x00\x00\x40\x80\x05\x0e\x10\x00\x00\x01\xbb\x44\x33\x22\x11\x01\xd9\x4d\x75\xee\xf4".to_vec();
    assert_eq!(patch_next[0x0d], 0xbb);
    // mov ebp, 9; L: add byte [1041h], 3; jmp T; back: dec ebp; jnz L; hlt;
    // ... T (1040h): mov eax, imm32; add ebx, eax; jmp back. The patched
    // immediate of another block becomes a live field.
    let mut patch_other =
        b"\xbd\x09\x00\x00\x00\x80\x05\x41\x10\x00\x00\x03\xe9\x2f\x00\x00\x00\x4d\x75\xf1\xf4"
            .to_vec();
    patch_other.resize(0x40, 0x90);
    patch_other.extend(b"\xb8\x04\x03\x02\x01\x01\xc3\xe9\xc5\xff\xff\xff");
    let mut out: Vec<(String, Vec<u8>, Setup)> = vec![
        // mov edi, 40; L: inc eax; add ebx, eax; xor edx, ebx; dec edi; jnz L; hlt
        (
            "register".to_string(),
            b"\xbf\x28\x00\x00\x00\x40\x01\xc3\x31\xda\x4f\x75\xf8\xf4".to_vec(),
            plain,
        ),
        (
            "memory".to_string(),
            b"\xbe\x00\x60\x00\x00\xbf\x28\x00\x00\x00\x8b\x06\x83\xc0\x01\x89\x06\x4f\x75\xf6\xf4"
                .to_vec(),
            plain,
        ),
        (
            "callret".to_string(),
            b"\xbf\x28\x00\x00\x00\xe8\x04\x00\x00\x00\x4f\x75\xf8\xf4\x40\xc3".to_vec(),
            plain,
        ),
        // inc eax; mov byte [1008h], F4h; inc eax (now HLT); hlt.
        (
            "store-into-next".to_string(),
            b"\x40\xc6\x05\x08\x10\x00\x00\xf4\x40\xf4".to_vec(),
            plain,
        ),
        ("patch-next-immediate".to_string(), patch_next, plain),
        ("patch-other-block".to_string(), patch_other, plain),
        // mov edi, 6; L: mov byte [1020h], 41h (INC EBX there becomes INC ECX);
        // inc eax; dec edi; jnz L; jmp 1020h ... 1020h: inc ebx; hlt
        (
            "store-into-other".to_string(),
            {
                let mut code = b"\xbf\x06\x00\x00\x00\xc6\x05\x20\x10\x00\x00\x41\x40\x4f\x75\xf5\xe9\x0b\x00\x00\x00".to_vec();
                code.resize(0x20, 0x90);
                code.extend(b"\x43\xf4");
                code
            },
            plain,
        ),
        // mov eax, [9000h]; inc eax; mov [9004h], eax; add ebx, [9008h]; hlt
        (
            "mmio".to_string(),
            b"\xa1\x00\x90\x00\x00\x40\xa3\x04\x90\x00\x00\x03\x1d\x08\x90\x00\x00\xf4".to_vec(),
            plain,
        ),
        // Transfers split across RAM|MMIO and ROM|RAM, ROM reads and writes,
        // between read-modify-writes.
        (
            "split-and-rom".to_string(),
            asm(|a| {
                a.mov(esi, 0x8ffeu32).unwrap();
                a.add(dword_ptr(0x6000), 5).unwrap();
                a.mov(dword_ptr(esi), eax).unwrap();
                a.mov(ebx, dword_ptr(esi)).unwrap();
                a.mov(esi, 0xaffeu32).unwrap();
                a.mov(ecx, dword_ptr(esi)).unwrap();
                a.add(dword_ptr(0x6004), 7).unwrap();
                a.mov(dword_ptr(esi), edx).unwrap();
                a.mov(edi, dword_ptr(0xa100)).unwrap();
                a.add(dword_ptr(0xa100), 9).unwrap();
                a.add(edi, dword_ptr(0xa100)).unwrap();
                a.add(dword_ptr(0x6008), 1).unwrap();
                a.hlt().unwrap();
            }),
            plain,
        ),
    ];
    // Above 1 MiB, and a word and a dword at FFFFEh, with A20 on and off.
    let a20 = asm(|a| {
        a.mov(esi, 0x10_6000u32).unwrap();
        a.add(dword_ptr(esi), 1).unwrap();
        a.mov(eax, dword_ptr(esi + 4)).unwrap();
        a.mov(esi, 0xf_fffeu32).unwrap();
        a.mov(ebx, dword_ptr(esi)).unwrap();
        a.add(word_ptr(esi), 1).unwrap();
        a.mov(dword_ptr(esi), ecx).unwrap();
        a.add(dword_ptr(0x6000), 1).unwrap();
        a.hlt().unwrap();
    });
    out.push(("a20-on".to_string(), a20.clone(), plain));
    out.push((
        "a20-off".to_string(),
        a20,
        Setup {
            a20_off: true,
            ..plain
        },
    ));
    let prefix = b"\x40\x01\xc3\x31\xd9\xa3\x00\x60\x00\x00\x8b\x15\x00\x60\x00\x00\x50\x5e\x47";
    for (name, ending) in [
        ("hlt", &b"\xf4"[..]),
        ("ud", b"\x0f\x0b"),
        ("de", b"\x31\xc9\xf7\xf1"),
        ("gp", b"\xa1\xff\xff\xff\xff"),
        ("ss", b"\x36\xa1\xff\xff\xff\xff"),
        ("in", b"\xe4\x60"),
    ] {
        for ordinal in [0, 2] {
            let mut code = prefix.to_vec();
            code.extend(vec![0x40; ordinal]);
            code.extend_from_slice(ending);
            code.extend(b"\x43\x42\xf4");
            out.push((format!("{name}@{ordinal}"), code, plain));
        }
    }
    for (name, code) in families(model) {
        out.push((name.to_string(), code, plain));
    }
    out
}

fn add(total: &mut MachineJitCounters, c: MachineJitCounters) {
    total.compiled += c.compiled;
    total.entries += c.entries;
    total.compiled_retired += c.compiled_retired;
    total.continues += c.continues;
    total.stops += c.stops;
    total.unconditional_defers += c.unconditional_defers;
    total.helper_defers += c.helper_defers;
    total.defers_at_zero += c.defers_at_zero;
    total.refused += c.refused;
    for (t, s) in total.suffix.iter_mut().zip(c.suffix) {
        *t += s;
    }
}

const MODES: [MachineJitMode; 4] = [
    MachineJitMode::Forced,
    MachineJitMode::Delayed(1),
    MachineJitMode::Delayed(3),
    MachineJitMode::Scripted(7),
];

#[test]
fn compiled_machine_blocks_equal_the_interpreter_at_every_call_386() {
    compiled_machine_blocks_equal_the_interpreter_at_every_call(CpuModel::I80386);
}

#[test]
fn compiled_machine_blocks_equal_the_interpreter_at_every_call_486() {
    compiled_machine_blocks_equal_the_interpreter_at_every_call(CpuModel::I80486Dx);
}

fn compiled_machine_blocks_equal_the_interpreter_at_every_call(model: CpuModel) {
    let mut forced = MachineJitCounters::default();
    {
        for (index, (name, code, setup)) in programs(model).into_iter().enumerate() {
            for mode in MODES {
                for seed in [index as u64 * 31 + 1, index as u64 * 31 + 2] {
                    for constant in [1u64, 2, 7, 64] {
                        let what =
                            format!("{name} {model:?} {mode:?} seed={seed} budget={constant}");
                        let c = lockstep(model, &code, mode, setup, seed, &mut |_| constant, &what);
                        if mode == MachineJitMode::Forced {
                            add(&mut forced, c);
                        }
                    }
                }
                // Every first budget around the first blocks, then 7.
                for first in 0..=12u64 {
                    let what = format!("{name} {model:?} {mode:?} first={first}");
                    let c = lockstep(
                        model,
                        &code,
                        mode,
                        setup,
                        5,
                        &mut |call| if call == 0 { first } else { 7 },
                        &what,
                    );
                    if mode == MachineJitMode::Forced {
                        add(&mut forced, c);
                    }
                }
            }
        }
    }
    // Decision-29 §10 item 2: compiled code ran, and every exit row occurred.
    assert!(forced.compiled > 0, "{forced:?}");
    assert!(forced.compiled_retired > 10_000, "{forced:?}");
    assert!(forced.continues > 0, "{forced:?}");
    assert!(forced.stops > 0, "{forced:?}");
    assert!(forced.unconditional_defers > 0, "{forced:?}");
    assert!(forced.helper_defers > 0, "{forced:?}");
    // Suffixes that completed, exited to MMIO and faulted. HLT and IN are
    // singleton (non-ordinary) Flat32 blocks, so no suffix hands off: their
    // k+1 count stays the interpreted singleton's.
    for kind in [0, 2, 3] {
        assert!(forced.suffix[kind] > 0, "suffix kind {kind}: {forced:?}");
    }
}

/// Every instruction of a family program compiles: only the final HLT, a
/// singleton block, stays interpreted.
#[test]
fn every_lowered_family_runs_compiled() {
    for model in MODELS {
        for (name, code) in families(model) {
            if name == "double-shift-16" {
                continue;
            }
            let c = lockstep(
                model,
                &code,
                MachineJitMode::Forced,
                Setup::default(),
                3,
                &mut |_| 1000,
                name,
            );
            assert_eq!(c.unconditional_defers, 0, "{name} {model:?}: {c:?}");
            assert_eq!(c.helper_defers, 0, "{name} {model:?}: {c:?}");
            assert!(c.continues >= 6, "{name} {model:?}: {c:?}");
        }
    }
}

/// E3 (TF), E7 (MXCSR mode) and E8 (RF): the unit is not entered, and the
/// results equal the interpreter's.
#[test]
fn entry_refusals_keep_blocks_interpreted() {
    let (_, register, _) = programs(CpuModel::I80386).remove(0);
    for model in MODELS {
        for (what, setup) in [
            (
                "tf",
                Setup {
                    tf: true,
                    ..Setup::default()
                },
            ),
            (
                "mxcsr",
                Setup {
                    mxcsr_mode: true,
                    ..Setup::default()
                },
            ),
        ] {
            let c = lockstep(
                model,
                &register,
                MachineJitMode::Forced,
                setup,
                9,
                &mut |_| 64,
                what,
            );
            assert_eq!(c.entries, 0, "{what} {model:?}: {c:?}");
            assert!(c.refused > 0, "{what} {model:?}: {c:?}");
        }
        // RF=1 refuses the first block only: its first instruction clears RF.
        let rf = Setup {
            rf: true,
            ..Setup::default()
        };
        let c = lockstep(
            model,
            &register,
            MachineJitMode::Forced,
            rf,
            9,
            &mut |_| 64,
            "rf",
        );
        assert!(c.refused >= 1, "rf {model:?}: {c:?}");
        assert!(c.entries > 0, "rf {model:?}: {c:?}");
    }
}

/// Decision-29 §8: compiled units are owned by their vCPU and retired when
/// their block is released and when the vCPU is dropped. No unit outlives it.
#[test]
fn compiled_units_are_retired_and_none_leaks() {
    // 700 blocks of INC EAX; JMP $+2, then HLT: the proof map holds 64 at a
    // time, so most blocks are released while the run goes on.
    let chain = asm(|a| {
        for _ in 0..700 {
            let mut next = a.create_label();
            a.inc(eax).unwrap();
            a.jmp(next).unwrap();
            a.set_label(&mut next).unwrap();
        }
        a.hlt().unwrap();
    });
    let patching = programs(CpuModel::I80386)
        .into_iter()
        .find(|(name, ..)| name == "patch-other-block")
        .unwrap()
        .1;
    for (code, released) in [(chain, true), (patching, false)] {
        let backend = JitBackend::new();
        let lifetime = backend.code_lifetime().unwrap();
        let (vm, mut cpu) = machine(
            backend,
            CpuModel::I80386,
            &code,
            MachineJitMode::Forced,
            Setup::default(),
            1,
        );
        let run = cpu.run_steps(&vm, 100_000, StepOptions::default()).unwrap();
        assert!(matches!(run.stop(), StepStop::Exit(Exit::Hlt)), "{run:?}");
        let c = cpu.machine_jit_counters_for_test();
        assert!(c.compiled > 0, "{c:?}");
        assert!(!released || c.retired > 0, "{c:?}");
        let live = lifetime.live_units().len() as u64;
        assert_eq!(live, c.compiled - c.retired, "{c:?}");
        assert!(live <= 512, "{live} live units: {c:?}");
        drop(cpu);
        assert_eq!(lifetime.live_units().len(), 0, "a unit outlived its vCPU");
        drop(vm);
    }
}
