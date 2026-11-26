//! A hot loop that patches fields of its own instructions (xj-to9.46).
//! Arguments: iterations (default 200000), rounds (default 5), optional
//! path, optional workload. MACHINE32_MODEL=386 selects I80386; the default
//! is I80486Dx.
//!
//! Nineteen small blocks share one 4 KiB code page at CPL 0. One block's
//! first instruction is patched every seven iterations (about 430 retired
//! instructions) by a store on the same page, as a renderer that patches its
//! own inner loop does, and the loop keeps executing:
//! * imm: `MOV EAX, imm32`; a 4-byte `ADD [imm], 1` changes the immediate.
//! * disp: `MOV EAX, [disp32]`; a 4-byte `XOR [disp], A^B` switches the
//!   displacement between two data words.
//! * opcode (control): `INC EBX`; a 1-byte `XOR [opcode], 8` switches it to
//!   `DEC EBX` and back. It changes an opcode, so it must still relift.
//! * many: 304 `MOV EAX, imm32; ADD EBX, EAX` pairs in one page, 8 of whose
//!   immediates a loop of `ADD [ESI], 1` patches per iteration, round robin:
//!   more patched fields than the engine keeps live, so most relift.
//!
//! Paths: flat (unpaged Flat32), seg (CS.D=1, 64 KiB segments based at the
//! code page: Segmented), paged (Flat32 with identity paging). The guest
//! result is checked against a model of the loop. With `--features
//! test-hooks`, stderr gives blocks, reselections, lifts, proof hits and
//! walks per million retired.
use std::time::Instant;
use x86jit_core::{CpuMode, CpuModel, Exit, Prot, Reg, RegionKind, Vm, VmConfig};

#[cfg(target_os = "macos")]
fn instructions() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::uninit();
    // SAFETY: V4 writes the matching structure, read only after success.
    unsafe {
        assert_eq!(
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                usage.as_mut_ptr().cast()
            ),
            0
        );
        let count = usage.assume_init().ri_instructions;
        assert_ne!(count, 0, "host instruction counter unavailable");
        count
    }
}
#[cfg(not(target_os = "macos"))]
fn instructions() -> u64 {
    0
}

const PATHS: [&str; 3] = ["flat", "seg", "paged"];
const WORKS: [&str; 4] = ["imm", "disp", "opcode", "many"];
/// The many workload's patched fields: more than the engine keeps live.
const FIELDS: u32 = 304;
const GROUP: u32 = 8;
const CODE: u64 = 0x10000;
// Linear data addresses, inside the seg path's 64 KiB segments.
const DATA_A: u32 = CODE as u32 + 0x4000;
const DATA_B: u32 = CODE as u32 + 0x4010;
const WORD_A: u32 = 0x1357_9bdf;
const WORD_B: u32 = 0x0246_8ace;
const BLOCKS: u32 = 19;
const PATCHED: u32 = 9;
const PERIOD: u32 = 7;
const MEMORY: u64 = 0x20000;

fn constant(i: u32) -> u32 {
    i.wrapping_mul(0x0101_0101) | 1
}

/// The loop at IP `org` with data addressed from `data` (the segment
/// base's complement); returns the bytes and the patched field's offset.
fn code(work: &str, org: u64, data: u32) -> (Vec<u8>, u64) {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    let mut done = a.create_label();
    let mut field = a.create_label();
    a.set_label(&mut top).unwrap();
    for i in 0..BLOCKS {
        let mut next = a.create_label();
        if i == PATCHED {
            match work {
                "imm" => {
                    // MOV EAX, imm32 (B8 id): the immediate is one byte in.
                    a.db(&[0xb8]).unwrap();
                    a.set_label(&mut field).unwrap();
                    a.dd(&[0x1111_1111]).unwrap();
                }
                "disp" => {
                    // MOV EAX, [disp32] (8B 05 disp32): two bytes in.
                    a.db(&[0x8b, 0x05]).unwrap();
                    a.set_label(&mut field).unwrap();
                    a.dd(&[DATA_A.wrapping_sub(data)]).unwrap();
                }
                _ => {
                    // INC EBX (43); XOR 8 makes it DEC EBX (4B). The label
                    // marks the next instruction: the opcode is one byte back.
                    a.db(&[0x43]).unwrap();
                    a.set_label(&mut field).unwrap();
                }
            }
            a.add(ebx, eax).unwrap();
        } else {
            a.add(ebx, constant(i) as i32).unwrap();
            a.xor(edx, ebx).unwrap();
        }
        a.jmp(next).unwrap();
        a.set_label(&mut next).unwrap();
    }
    a.dec(edi).unwrap();
    a.je(done).unwrap();
    a.dec(ecx).unwrap();
    a.jne(top).unwrap();
    a.mov(ecx, PERIOD).unwrap();
    // The field's address is fixed below, after assembly.
    match work {
        "imm" => a.add(dword_ptr(0x7fff_fff0), 1).unwrap(),
        "disp" => a
            .xor(dword_ptr(0x7fff_fff0), (DATA_A ^ DATA_B) as i32)
            .unwrap(),
        _ => a.xor(byte_ptr(0x7fff_fff0), 8).unwrap(),
    }
    a.jmp(top).unwrap();
    a.set_label(&mut done).unwrap();
    a.hlt().unwrap();
    let result = a
        .assemble_options(
            org,
            iced_x86::BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS,
        )
        .unwrap();
    let field = result.label_ip(&field).unwrap() - u64::from(work == "opcode");
    let mut bytes = result.inner.code_buffer;
    let marker = 0x7fff_fff0u32.to_le_bytes();
    let at = bytes.windows(4).position(|w| w == marker).unwrap();
    bytes[at..at + 4].copy_from_slice(&(field as u32).to_le_bytes());
    (bytes, field - org)
}

/// The many workload at `org`: the fields at `org + 1 + 7 i`, the patch
/// pointer in ESI (from `org + 1`), the iteration count in EDI.
fn code_many(org: u32) -> Vec<u8> {
    let mut code = Vec::new();
    for i in 0..FIELDS {
        code.push(0xb8);
        code.extend_from_slice(&i.to_le_bytes());
        code.extend_from_slice(&[0x01, 0xc3]);
    }
    let tail = code.len() as i32;
    // MOV ECX, 8; L: ADD DWORD [ESI], 1; ADD ESI, 7; DEC ECX; JNZ L.
    code.extend_from_slice(&[
        0xb9,
        GROUP as u8,
        0,
        0,
        0,
        0x83,
        0x06,
        0x01,
        0x83,
        0xc6,
        0x07,
        0x49,
        0x75,
        0xf7,
    ]);
    // CMP ESI, end; JB skip; MOV ESI, start; skip: DEC EDI; JNZ top; HLT.
    code.extend_from_slice(&[0x81, 0xfe]);
    code.extend_from_slice(&(org + 1 + 7 * FIELDS).to_le_bytes());
    code.extend_from_slice(&[0x72, 0x05, 0xbe]);
    code.extend_from_slice(&(org + 1).to_le_bytes());
    code.extend_from_slice(&[0x4f, 0x0f, 0x85]);
    code.extend_from_slice(&(-(tail + 34)).to_le_bytes());
    code.push(0xf4);
    code
}

/// The many workload's effect from immediates `imm` and patch index `p`.
fn model_many(n: u64, mut imm: Vec<u32>, mut p: u32) -> Model {
    let mut m = Model {
        eax: 0,
        ebx: 0,
        edx: 0,
        field: 0,
        retired: 0,
    };
    for left in (1..=n).rev() {
        for &value in &imm {
            m.eax = value;
            m.ebx = m.ebx.wrapping_add(value);
        }
        for k in 0..GROUP {
            imm[(p + k) as usize] = imm[(p + k) as usize].wrapping_add(1);
        }
        p += GROUP;
        m.retired += u64::from(2 * FIELDS + 1 + 4 * GROUP + 2 + 2);
        if p == FIELDS {
            p = 0;
            m.retired += 1;
        }
        if left == 1 {
            m.retired += 1; // HLT.
        }
    }
    m
}

struct Model {
    eax: u32,
    ebx: u32,
    edx: u32,
    field: u32,
    retired: u64,
}

/// The loop's effect from a field holding `field` (the immediate, the
/// displacement or the opcode byte).
fn model(work: &str, n: u64, field: u32, data: u32) -> Model {
    let mut m = Model {
        eax: 0,
        ebx: 0,
        edx: 0,
        field,
        retired: 0,
    };
    let mut ecx = PERIOD;
    let mut edi = n;
    loop {
        for i in 0..BLOCKS {
            if i == PATCHED {
                match work {
                    "imm" => m.eax = m.field,
                    "disp" => {
                        m.eax = if m.field.wrapping_add(data) == DATA_A {
                            WORD_A
                        } else {
                            WORD_B
                        }
                    }
                    _ if m.field == 0x43 => m.ebx = m.ebx.wrapping_add(1),
                    _ => m.ebx = m.ebx.wrapping_sub(1),
                }
                m.ebx = m.ebx.wrapping_add(m.eax);
            } else {
                m.ebx = m.ebx.wrapping_add(constant(i));
                m.edx ^= m.ebx;
            }
            m.retired += 3;
        }
        edi -= 1;
        m.retired += 2;
        if edi == 0 {
            m.retired += 1; // HLT.
            return m;
        }
        ecx -= 1;
        m.retired += 2;
        if ecx == 0 {
            ecx = PERIOD;
            m.field = match work {
                "imm" => m.field.wrapping_add(1),
                "disp" => m.field ^ DATA_A ^ DATA_B,
                _ => m.field ^ 8,
            };
            m.retired += 3;
        }
    }
}

fn setup(path: &str, work: &str) -> (Vm, x86jit_core::Vcpu, u64) {
    let seg = path == "seg";
    let paged = path == "paged";
    let mut vm = Vm::new(VmConfig::flat(MEMORY));
    vm.set_cpu_mode(CpuMode::Compat32);
    vm.set_cpu_model(match std::env::var("MACHINE32_MODEL").as_deref() {
        Ok("386") => CpuModel::I80386,
        _ => CpuModel::I80486Dx,
    });
    vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    let base = if seg { CODE as u32 } else { 0 };
    let org = if seg { 0 } else { CODE };
    let (bytes, field) = if work == "many" {
        (code_many(org as u32), 1)
    } else {
        code(work, org, base)
    };
    vm.write_bytes(CODE, &bytes).unwrap();
    vm.write_bytes(u64::from(DATA_A), &WORD_A.to_le_bytes())
        .unwrap();
    vm.write_bytes(u64::from(DATA_B), &WORD_B.to_le_bytes())
        .unwrap();
    if paged {
        vm.write_bytes(0x1000, &0x2003u32.to_le_bytes()).unwrap();
        for page in 0..(MEMORY / 4096) as u32 {
            vm.write_bytes(
                0x2000 + u64::from(4 * page),
                &((page * 4096) | 3).to_le_bytes(),
            )
            .unwrap();
        }
    }
    let mut cpu = vm.new_vcpu();
    cpu.cpu.legacy.cr0 = if paged { 0x8000_0001 } else { 1 };
    cpu.cpu.legacy.cr3 = 0x1000;
    cpu.cpu.legacy.cpl = 0;
    for (i, segment) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        segment.base = base;
        // Byte-granular 64 KiB limits keep CS.D and SS.B at 1: Segmented.
        segment.limit = if seg { 0xffff } else { u32::MAX };
        segment.attributes = (if i == 0 { 0x9b } else { 0x93 }) | 0x4000;
        segment.unusable = 0;
    }
    cpu.set_a20_enabled(true);
    cpu.set_reg(Reg::Rsi, org + 1);
    (vm, cpu, CODE + field)
}

fn execute(vm: &Vm, cpu: &mut x86jit_core::Vcpu, work: &str, field_at: u64, n: u64) -> Model {
    if work == "many" {
        return execute_many(vm, cpu, field_at, n);
    }
    let width = if work == "opcode" { 1 } else { 4 };
    let mut field = [0; 4];
    vm.read_bytes(field_at, &mut field[..width]).unwrap();
    let base = cpu.cpu.legacy.segments[0].base;
    let expect = model(work, n, u32::from_le_bytes(field), base);
    let org = if base == 0 { CODE } else { 0 };
    cpu.set_reg(Reg::Rip, org);
    for reg in [Reg::Rax, Reg::Rbx, Reg::Rdx] {
        cpu.set_reg(reg, 0);
    }
    cpu.set_reg(Reg::Rcx, u64::from(PERIOD));
    cpu.set_reg(Reg::Rdi, n);
    cpu.set_reg(Reg::Rsp, 0x8000);
    let exit = cpu.run(vm, None);
    assert!(matches!(exit, Exit::Hlt), "{exit:?}");
    assert_eq!(cpu.reg(Reg::Rax) as u32, expect.eax);
    assert_eq!(cpu.reg(Reg::Rbx) as u32, expect.ebx);
    assert_eq!(cpu.reg(Reg::Rdx) as u32, expect.edx);
    vm.read_bytes(field_at, &mut field[..width]).unwrap();
    assert_eq!(u32::from_le_bytes(field), expect.field);
    expect
}

fn execute_many(vm: &Vm, cpu: &mut x86jit_core::Vcpu, first: u64, n: u64) -> Model {
    let imm = (0..FIELDS)
        .map(|i| {
            let mut field = [0; 4];
            vm.read_bytes(first + 7 * u64::from(i), &mut field).unwrap();
            u32::from_le_bytes(field)
        })
        .collect();
    let org = first - 1 - u64::from(cpu.cpu.legacy.segments[0].base);
    let p = ((cpu.reg(Reg::Rsi) - org - 1) / 7) as u32;
    let expect = model_many(n, imm, p);
    cpu.set_reg(Reg::Rip, org);
    for reg in [Reg::Rax, Reg::Rbx, Reg::Rdx] {
        cpu.set_reg(reg, 0);
    }
    cpu.set_reg(Reg::Rdi, n);
    let exit = cpu.run(vm, None);
    assert!(matches!(exit, Exit::Hlt), "{exit:?}");
    assert_eq!(cpu.reg(Reg::Rax) as u32, expect.eax);
    assert_eq!(cpu.reg(Reg::Rbx) as u32, expect.ebx);
    expect
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n = args.get(1).map_or(200_000, |s| s.parse::<u64>().unwrap());
    let rounds = args.get(2).map_or(5, |s| s.parse::<usize>().unwrap());
    assert!(n > 0 && n <= u32::MAX as u64);
    let paths: Vec<&str> = args
        .get(3)
        .map_or_else(|| PATHS.to_vec(), |s| vec![s.as_str()]);
    let works: Vec<&str> = args
        .get(4)
        .map_or_else(|| WORKS.to_vec(), |s| vec![s.as_str()]);
    println!(
        "workload,path,round,iterations,instructions,retired_counter,seconds,mips,host_per_step"
    );
    for round in 0..rounds {
        for &work in &works {
            for &path in &paths {
                let (vm, mut cpu, field_at) = setup(path, work);
                execute(&vm, &mut cpu, work, field_at, 2000);
                let retired_before = cpu.retired_instructions();
                #[cfg(feature = "test-hooks")]
                x86jit_core::Vcpu::take_machine_counters_for_test();
                let host_before = instructions();
                let start = Instant::now();
                let expect = execute(&vm, &mut cpu, work, field_at, n);
                let seconds = start.elapsed().as_secs_f64();
                let host = instructions() - host_before;
                let retired = cpu.retired_instructions() - retired_before;
                assert_eq!(retired, expect.retired);
                #[cfg(feature = "test-hooks")]
                eprintln!(
                    "counters smc-live {work} {path}: {:?}",
                    x86jit_core::Vcpu::take_machine_counters_for_test()
                        .map(|c| (c as f64 * 1e6 / retired.max(1) as f64).round())
                );
                println!(
                    "smc-live-{work},{path},{round},{n},{retired},{retired},{seconds:.9},{:.3},{:.3}",
                    retired as f64 / seconds / 1e6,
                    host as f64 / retired as f64
                );
            }
        }
    }
}
