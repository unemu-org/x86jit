//! A hot loop that patches fields of its own instructions (xj-to9.46).
//! Arguments: iterations (default 200000), rounds (default 5), optional
//! path, optional workload. MACHINE32_MODEL=386 selects I80386; the default
//! is I80486Dx.
//!
//! Nineteen small blocks share one 4 KiB code page at CPL 0. One block's
//! first instruction is patched every seven iterations (about 430 retired
//! instructions) by a store on the same page, as a renderer that patches its
//! own inner loop does (MACHINE32_SMC_PERIOD sets the iterations per
//! patch), and the loop keeps executing:
//! * imm: `MOV EAX, imm32`; a 4-byte `ADD [imm], 1` changes the immediate.
//! * disp: `MOV EAX, [disp32]`; a 4-byte `XOR [disp], A^B` switches the
//!   displacement between two data words.
//! * opcode (control): `INC EBX`; a 1-byte `XOR [opcode], 8` switches it to
//!   `DEC EBX` and back. It changes an opcode, so it must still relift.
//! * many: 304 `MOV EAX, imm32; ADD EBX, EAX` pairs in one page, 8 of whose
//!   immediates a loop of `ADD [ESI], 1` patches per iteration, round robin:
//!   more patched fields than the engine keeps live, so most relift.
//! * data (control): imm's loop, but the 4-byte `ADD [data], 1` writes a
//!   data word on another page: the cost of the same store off code pages.
//!
//! Further forms a drawer's inner loop may patch (xj-to9.53), named
//! explicitly (the workload argument takes a comma-separated list, or
//! `all`):
//! * imm8: `ADD EAX, simm8` (83 /0 ib), a 1-byte `ADD [imm8], 1`.
//! * imm16: `MOV AX, imm16` (66 B8 iw), a 2-byte `ADD [imm16], 1`.
//! * sib: `ADD EAX, [EBP+ESI*4+disp32]` (SIB), the displacement switched.
//! * shift: `SHR EAX, imm8` (C1 /5 ib), the count switched between 3 and 2.
//! * branch: `JMP rel8` (EB cb), switched between 0 and 3 to skip an ADD.
//! * imul: `IMUL EAX, EBX, imm32` (69 /r id), the immediate incremented.
//! * movzx: `MOVZX EAX, byte [disp32]` (0F B6 /r), the displacement switched.
//! * multi: four `MOV EAX, imm32` in one block, all four patched.
//! * entry: `MOV EAX, imm32` in a block entered at its start or, on odd
//!   iterations, at the patched instruction itself: two proofs hold it.
//! * self: the block patches its own `MOV EAX, imm32` just before it on
//!   every iteration, as well as every seventh.
//! * codedata: a 4-byte store into a data word on the code page, outside
//!   every instruction, which a `MOV EAX, [disp32]` reads.
//! * fs: imm, and each iteration first loads FS with the next of
//!   MACHINE32_SELECTORS (default 20) identical flat data selectors, which
//!   the code never uses: a context stamp per selector.
//! * reflect: imm, and each patch is followed by a walk through
//!   MACHINE32_SELECTORS data selectors with distinct bases (Segmented
//!   contexts) and back to the path's own data segment, as an
//!   extended-memory runtime's interrupt reflection walks through many
//!   context stamps.
//! * runtime: no patching; a game, a renderer and an extended-memory
//!   runtime's interrupt reflection with a real-mode round trip (the
//!   `runtime` module below; the iterations are frames). Flat path only.
//!
//! MACHINE32_BLOCKS sets the number of loop blocks (default 19); more than
//! the 64 unpaged proofs the engine retains exercises that bound.
//! Paths: flat (unpaged Flat32), seg (CS.D=1, 64 KiB segments based at the
//! code page: Segmented), paged (Flat32 with identity paging). The guest
//! result is checked against a model of the loop. With `--features
//! test-hooks`, stderr gives blocks, reselections, lifts, proof hits and
//! walks per million retired, and MACHINE32_DIAG=1 prints the VM's relift
//! diagnostics (`Vm::relift_diagnostics`) after each timed run.
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
const WORKS: [&str; 5] = ["imm", "disp", "opcode", "many", "data"];
const FORMS: [&str; 14] = [
    "imm8", "imm16", "sib", "shift", "branch", "imul", "movzx", "multi", "entry", "self",
    "codedata", "fs", "reflect", "runtime",
];
/// The fs workload's GDT (linear) and its rotating selector's word.
const GDT: u64 = 0x6000;
const SELECTOR_AT: u32 = DATA_A + 0x100;

fn selectors() -> u16 {
    std::env::var("MACHINE32_SELECTORS").map_or(20, |s| s.parse().unwrap())
}
/// The fields the multi workload patches.
const MULTI: usize = 4;
/// The many workload's patched fields: more than the engine keeps live.
const FIELDS: u32 = 304;
const GROUP: u32 = 8;
const CODE: u64 = 0x10000;
// Linear data addresses, inside the seg path's 64 KiB segments.
const DATA_A: u32 = CODE as u32 + 0x4000;
const DATA_B: u32 = CODE as u32 + 0x4010;
const WORD_A: u32 = 0x1357_9bdf;
const WORD_B: u32 = 0x0246_8ace;
// The data workload's counter, on the data words' page.
const DATA_C: u32 = CODE as u32 + 0x4020;
const IMM: u32 = 0x1111_1111;
const PATCHED: u32 = 9;

fn blocks() -> u32 {
    std::env::var("MACHINE32_BLOCKS").map_or(19, |s| s.parse::<u32>().unwrap().max(PATCHED + 1))
}
/// Iterations per patch: MACHINE32_SMC_PERIOD, default 7.
fn period() -> u32 {
    static PERIOD: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *PERIOD.get_or_init(|| std::env::var("MACHINE32_SMC_PERIOD").map_or(7, |s| s.parse().unwrap()))
}
const MEMORY: u64 = 0x20000;

fn constant(i: u32) -> u32 {
    i.wrapping_mul(0x0101_0101) | 1
}

/// The marker a patch store's address holds until assembly places field `k`.
fn marker(k: usize) -> u32 {
    0x7fff_ff00 + 4 * k as u32
}

/// The loop at IP `org` with data addressed from `data` (the segment
/// base's complement); returns the bytes and the patched fields' offsets.
fn code(work: &str, org: u64, data: u32) -> (Vec<u8>, Vec<u64>) {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    let mut done = a.create_label();
    let count = if work == "multi" { MULTI } else { 1 };
    let mut fields: Vec<_> = (0..count).map(|_| a.create_label()).collect();
    a.set_label(&mut top).unwrap();
    if work == "fs" {
        // MOV BP, [sel]; ADD BP, 8; CMP BP, end; JB ok; MOV BP, 8;
        // ok: MOV [sel], BP; MOV FS, BP. FS is never used.
        let mut ok = a.create_label();
        let at = SELECTOR_AT.wrapping_sub(data);
        a.mov(bp, word_ptr(at)).unwrap();
        a.add(bp, 8).unwrap();
        a.cmp(bp, 8 * (i32::from(selectors()) + 1)).unwrap();
        a.jb(ok).unwrap();
        a.mov(bp, 8).unwrap();
        a.set_label(&mut ok).unwrap();
        a.mov(word_ptr(at), bp).unwrap();
        a.mov(fs, bp).unwrap();
    }
    for i in 0..blocks() {
        let mut next = a.create_label();
        if i == PATCHED {
            let field = &mut fields[0];
            match work {
                "imm" | "fs" | "reflect" | "data" => {
                    // MOV EAX, imm32 (B8 id): the immediate is one byte in.
                    a.db(&[0xb8]).unwrap();
                    a.set_label(field).unwrap();
                    a.dd(&[IMM]).unwrap();
                }
                "disp" => {
                    // MOV EAX, [disp32] (8B 05 disp32): two bytes in.
                    a.db(&[0x8b, 0x05]).unwrap();
                    a.set_label(field).unwrap();
                    a.dd(&[DATA_A.wrapping_sub(data)]).unwrap();
                }
                "imm8" => {
                    // XOR EAX, EAX; ADD EAX, simm8 (83 C0 ib).
                    a.db(&[0x31, 0xc0, 0x83, 0xc0]).unwrap();
                    a.set_label(field).unwrap();
                    a.db(&[0x10]).unwrap();
                }
                "imm16" => {
                    // XOR EAX, EAX; MOV AX, imm16 (66 B8 iw).
                    a.db(&[0x31, 0xc0, 0x66, 0xb8]).unwrap();
                    a.set_label(field).unwrap();
                    a.dw(&[0x1234]).unwrap();
                }
                "sib" => {
                    // XOR EAX, EAX; ADD EAX, [EBP+ESI*4+disp32] (03 84 B5).
                    a.db(&[0x31, 0xc0, 0x03, 0x84, 0xb5]).unwrap();
                    a.set_label(field).unwrap();
                    a.dd(&[DATA_A.wrapping_sub(data)]).unwrap();
                }
                "shift" => {
                    // MOV EAX, EBX; SHR EAX, imm8 (C1 E8 ib).
                    a.db(&[0x89, 0xd8, 0xc1, 0xe8]).unwrap();
                    a.set_label(field).unwrap();
                    a.db(&[3]).unwrap();
                }
                "branch" => {
                    // JMP rel8 (EB cb) over ADD EBX, 5 (83 C3 05) or not.
                    a.db(&[0xeb]).unwrap();
                    a.set_label(field).unwrap();
                    a.db(&[0, 0x83, 0xc3, 0x05]).unwrap();
                }
                "imul" => {
                    // IMUL EAX, EBX, imm32 (69 C3 id).
                    a.db(&[0x69, 0xc3]).unwrap();
                    a.set_label(field).unwrap();
                    a.dd(&[3]).unwrap();
                }
                "movzx" => {
                    // MOVZX EAX, byte [disp32] (0F B6 05 disp32).
                    a.db(&[0x0f, 0xb6, 0x05]).unwrap();
                    a.set_label(field).unwrap();
                    a.dd(&[DATA_A.wrapping_sub(data)]).unwrap();
                }
                "multi" => {
                    for (k, field) in fields.iter_mut().enumerate() {
                        a.db(&[0xb8]).unwrap();
                        a.set_label(field).unwrap();
                        a.dd(&[0x1111_1111 * (k as u32 + 1)]).unwrap();
                        if k + 1 < MULTI {
                            a.add(ebx, eax).unwrap();
                        }
                    }
                }
                "entry" => {
                    // TEST EDI, 1; JNZ at; ADD EBX, 1; at: MOV EAX, imm32.
                    let mut at = a.create_label();
                    a.test(edi, 1).unwrap();
                    a.jne(at).unwrap();
                    a.add(ebx, 1).unwrap();
                    a.set_label(&mut at).unwrap();
                    a.db(&[0xb8]).unwrap();
                    a.set_label(field).unwrap();
                    a.dd(&[0x1111_1111]).unwrap();
                }
                "self" => {
                    a.add(dword_ptr(marker(0)), 1).unwrap();
                    a.db(&[0xb8]).unwrap();
                    a.set_label(field).unwrap();
                    a.dd(&[0x1111_1111]).unwrap();
                }
                "codedata" => {
                    // MOV EAX, [disp32] of a word after the code.
                    a.mov(eax, dword_ptr(marker(0))).unwrap();
                }
                _ => {
                    // INC EBX (43); XOR 8 makes it DEC EBX (4B). The label
                    // marks the next instruction: the opcode is one byte back.
                    a.db(&[0x43]).unwrap();
                    a.set_label(field).unwrap();
                }
            }
            if work != "branch" {
                a.add(ebx, eax).unwrap();
            }
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
    a.mov(ecx, period()).unwrap();
    // The fields' addresses are fixed below, after assembly.
    for k in 0..count {
        let at = dword_ptr(marker(k));
        match work {
            "imm" | "fs" | "reflect" | "imul" | "multi" | "entry" | "self" | "codedata"
            | "data" => a.add(at, 1).unwrap(),
            "imm8" => a.add(byte_ptr(marker(k)), 1).unwrap(),
            "imm16" => a.add(word_ptr(marker(k)), 1).unwrap(),
            "disp" | "sib" | "movzx" => a.xor(at, (DATA_A ^ DATA_B) as i32).unwrap(),
            "shift" => a.xor(byte_ptr(marker(k)), 1).unwrap(),
            "branch" => a.xor(byte_ptr(marker(k)), 3).unwrap(),
            _ => a.xor(byte_ptr(marker(k)), 8).unwrap(),
        }
    }
    if work == "reflect" {
        // MOV BP, 8k; MOV DS, BP; JMP $+2 for each selector, then the
        // path's own: each block is entered under the stamp it loaded.
        for k in 1..=selectors() + 1 {
            let mut next = a.create_label();
            a.mov(bp, 8 * k as i32).unwrap();
            a.mov(ds, bp).unwrap();
            a.jmp(next).unwrap();
            a.set_label(&mut next).unwrap();
        }
    }
    a.jmp(top).unwrap();
    a.set_label(&mut done).unwrap();
    a.hlt().unwrap();
    if work == "codedata" {
        a.set_label(&mut fields[0]).unwrap();
        a.dd(&[0x1111_1111]).unwrap();
    }
    let result = a
        .assemble_options(
            org,
            iced_x86::BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS,
        )
        .unwrap();
    let mut fields: Vec<u64> = fields
        .iter()
        .map(|field| result.label_ip(field).unwrap() - u64::from(work == "opcode"))
        .collect();
    if work == "data" {
        // The counter's address as the code names it, from the segment base.
        fields[0] = org + u64::from(DATA_C) - CODE;
    }
    let mut bytes = result.inner.code_buffer;
    for (k, field) in fields.iter().enumerate() {
        let marker = marker(k).to_le_bytes();
        while let Some(at) = bytes.windows(4).position(|w| w == marker) {
            bytes[at..at + 4].copy_from_slice(&(*field as u32).to_le_bytes());
        }
    }
    (bytes, fields.iter().map(|f| f - org).collect())
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
        field: [0; MULTI],
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
    field: [u32; MULTI],
    retired: u64,
}

/// The value a displacement field selects.
fn word(field: u32, data: u32) -> u32 {
    if field.wrapping_add(data) == DATA_A {
        WORD_A
    } else {
        WORD_B
    }
}

/// The loop's effect from fields holding `field` (the immediates, the
/// displacement, the opcode byte, the shift count, the branch displacement
/// or the data word).
fn model(work: &str, n: u64, field: [u32; MULTI], data: u32) -> Model {
    let mut m = Model {
        eax: 0,
        ebx: 0,
        edx: 0,
        field,
        retired: 0,
    };
    let count = if work == "multi" { MULTI } else { 1 };
    let mut ecx = period();
    let mut edi = n;
    let mut selector = 8;
    loop {
        if work == "fs" {
            selector += 8;
            m.retired += 6;
            if selector >= 8 * (selectors() + 1) {
                selector = 8;
                m.retired += 1;
            }
        }
        for i in 0..blocks() {
            if i == PATCHED {
                let f = m.field[0];
                // The instructions before the closing ADD EBX, EAX and JMP.
                let body = match work {
                    "imm" | "fs" | "reflect" => {
                        m.eax = f;
                        1
                    }
                    "disp" => {
                        m.eax = word(f, data);
                        1
                    }
                    "imm8" => {
                        m.eax = f as u8 as i8 as u32;
                        2
                    }
                    "imm16" => {
                        m.eax = f & 0xffff;
                        2
                    }
                    "sib" => {
                        m.eax = word(f, data);
                        2
                    }
                    "shift" => {
                        m.eax = m.ebx >> (f & 31);
                        2
                    }
                    "branch" => {
                        // JMP, then ADD EBX, 5 unless skipped; no ADD EBX, EAX.
                        if f == 0 {
                            m.ebx = m.ebx.wrapping_add(5);
                            m.retired += 1;
                        }
                        m.retired += 2;
                        continue;
                    }
                    "imul" => {
                        m.eax = m.ebx.wrapping_mul(f);
                        1
                    }
                    "movzx" => {
                        m.eax = word(f, data) & 0xff;
                        1
                    }
                    "multi" => {
                        for k in 0..MULTI - 1 {
                            m.ebx = m.ebx.wrapping_add(m.field[k]);
                        }
                        m.eax = m.field[MULTI - 1];
                        2 * MULTI as u64 - 1
                    }
                    "entry" => {
                        m.eax = f;
                        if edi & 1 == 0 {
                            m.ebx = m.ebx.wrapping_add(1);
                            4
                        } else {
                            3
                        }
                    }
                    "self" => {
                        m.field[0] = f.wrapping_add(1);
                        m.eax = m.field[0];
                        2
                    }
                    "codedata" => {
                        m.eax = f;
                        1
                    }
                    "data" => {
                        m.eax = IMM;
                        1
                    }
                    _ => {
                        if f == 0x43 {
                            m.ebx = m.ebx.wrapping_add(1);
                        } else {
                            m.ebx = m.ebx.wrapping_sub(1);
                        }
                        1
                    }
                };
                m.ebx = m.ebx.wrapping_add(m.eax);
                m.retired += body + 2;
            } else {
                m.ebx = m.ebx.wrapping_add(constant(i));
                m.edx ^= m.ebx;
                m.retired += 3;
            }
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
            ecx = period();
            for f in &mut m.field[..count] {
                *f = match work {
                    "imm" | "fs" | "reflect" | "imul" | "multi" | "entry" | "self" | "codedata"
                    | "data" => f.wrapping_add(1),
                    "imm8" => (*f + 1) & 0xff,
                    "imm16" => (*f + 1) & 0xffff,
                    "disp" | "sib" | "movzx" => *f ^ DATA_A ^ DATA_B,
                    "shift" => *f ^ 1,
                    "branch" => *f ^ 3,
                    _ => *f ^ 8,
                };
            }
            m.retired += 2 + count as u64;
            if work == "reflect" {
                m.retired += 3 * (u64::from(selectors()) + 1);
            }
        }
    }
}

fn setup(path: &str, work: &str) -> (Vm, x86jit_core::Vcpu, Vec<u64>) {
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
    let (bytes, fields) = if work == "many" {
        (code_many(org as u32), vec![1])
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
    if work == "fs" {
        // Identical flat writable data descriptors (G, B, DPL 0) at 8, 16, ...
        for k in 1..=u64::from(selectors()) {
            vm.write_bytes(GDT + 8 * k, &[0xff, 0xff, 0, 0, 0, 0x93, 0xcf, 0])
                .unwrap();
        }
        vm.write_bytes(u64::from(SELECTOR_AT), &8u16.to_le_bytes())
            .unwrap();
    }
    if work == "reflect" {
        // Data descriptors based at 100h * k (G, B), then one matching the
        // path's data segment cache.
        for k in 1..=u64::from(selectors()) {
            let b = (0x100 * k as u32).to_le_bytes();
            vm.write_bytes(
                GDT + 8 * k,
                &[0xff, 0xff, b[0], b[1], b[2], 0x93, 0xcf, b[3]],
            )
            .unwrap();
        }
        let b = base.to_le_bytes();
        let own = if seg {
            [0xff, 0xff, b[0], b[1], b[2], 0x93, 0x40, b[3]]
        } else {
            [0xff, 0xff, b[0], b[1], b[2], 0x93, 0xcf, b[3]]
        };
        vm.write_bytes(GDT + 8 * (u64::from(selectors()) + 1), &own)
            .unwrap();
    }
    let mut cpu = vm.new_vcpu();
    cpu.cpu.legacy.gdtr.base = GDT as u32;
    cpu.cpu.legacy.gdtr.limit = 8 * (selectors() + 2) - 1;
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
    (vm, cpu, fields.iter().map(|f| CODE + f).collect())
}

fn execute(vm: &Vm, cpu: &mut x86jit_core::Vcpu, work: &str, fields: &[u64], n: u64) -> Model {
    if work == "many" {
        return execute_many(vm, cpu, fields[0], n);
    }
    let width = match work {
        "opcode" | "imm8" | "shift" | "branch" => 1,
        "imm16" => 2,
        _ => 4,
    };
    let read = |at: u64| {
        let mut field = [0; 4];
        vm.read_bytes(at, &mut field[..width]).unwrap();
        u32::from_le_bytes(field)
    };
    let mut start = [0; MULTI];
    for (f, &at) in start.iter_mut().zip(fields) {
        *f = read(at);
    }
    let base = cpu.cpu.legacy.segments[0].base;
    let expect = model(work, n, start, base);
    let org = if base == 0 { CODE } else { 0 };
    cpu.set_reg(Reg::Rip, org);
    for reg in [Reg::Rax, Reg::Rbx, Reg::Rdx, Reg::Rbp] {
        cpu.set_reg(reg, 0);
    }
    if work == "sib" {
        cpu.set_reg(Reg::Rsi, 0);
    }
    if work == "fs" {
        vm.write_bytes(u64::from(SELECTOR_AT), &8u16.to_le_bytes())
            .unwrap();
    }
    cpu.set_reg(Reg::Rcx, u64::from(period()));
    cpu.set_reg(Reg::Rdi, n);
    cpu.set_reg(Reg::Rsp, 0x8000);
    let exit = cpu.run(vm, None);
    assert!(matches!(exit, Exit::Hlt), "{exit:?}");
    assert_eq!(cpu.reg(Reg::Rax) as u32, expect.eax, "{work}");
    assert_eq!(cpu.reg(Reg::Rbx) as u32, expect.ebx, "{work}");
    assert_eq!(cpu.reg(Reg::Rdx) as u32, expect.edx, "{work}");
    for (f, &at) in expect.field.iter().zip(fields) {
        assert_eq!(read(at), *f, "{work}");
    }
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

/// The runtime workload (xj-to9.56): a protected-mode program as a PC-class
/// embedder's game runs it, interrupted by an extended-memory runtime's
/// interrupt reflection. Per frame (EDI frames):
/// * game logic: MACHINE32_GAME (default 2000) Flat32 blocks, once;
/// * renderer: MACHINE32_RENDER (default 250) Flat32 blocks with stack
///   traffic, called MACHINE32_K1 (default 60) times under the flat stack
///   and MACHINE32_K2 (default 30) times under a stack segment with a base
///   and a 7D00h-byte limit, as the renderer runs under the runtime's
///   locked stack;
/// * reflection: a far call into the runtime's 32-bit code segment (base
///   104820h), whose MACHINE32_RT (default 150) blocks run under
///   fourteen DS/ES pairs (flat aliases and based data segments), then a
///   16-bit protected segment that loads the real-mode IDTR, a real-mode
///   round trip through four DS values, the protected IDTR again and five
///   16-bit data segments, and a far return.
///
/// MACHINE32_EXTRA adds based data segments to the runtime's pairs. Every
/// count is at least 1. Every block adds a constant to EBX and XORs one into EDX; the result is
/// checked against the counts of each body's executions.
mod runtime {
    use super::*;
    use iced_x86::code_asm::*;

    const MEMORY: u64 = 0x40_0000;
    const GDT: u64 = 0x6000;
    const IDTR_PROTECTED: u16 = 0x6800;
    const IDTR_REAL: u16 = 0x6808;
    const SAVE_ESP: u32 = 0x6810;
    const FLAT_CODE: u64 = 0x20_0000;
    const RT_BASE: u32 = 0x10_4820;
    const SIXTEEN_BASE: u32 = 0x2_8000;
    const STACK_BASE: u32 = 0x11_6330;
    const FLAT_ESP: u32 = 0x3f_0000;
    // Selectors: flat code, flat data, a flat data alias, the based stack,
    // the runtime's code, 16-bit code at SIXTEEN_BASE, a 64 KiB data
    // segment at 0, then the runtime's data segments.
    const CODE32: u16 = 0x08;
    const FLAT: u16 = 0x10;
    const ALIAS: u16 = 0x18;
    const STACK: u16 = 0x20;
    const RT: u16 = 0x28;
    const CODE16: u16 = 0x30;
    const DATA16: u16 = 0x38;
    const DATA0: u16 = 0x40;
    /// The runtime's data segment bases (selectors DATA0 + 8 k).
    const BASES: [u32; 8] = [
        0x10_4820, 0x11_6330, 0x0c_0000, 0x0a_0000, 0x5_6a00, 0x0e_0000, 0x400, 0x2_0000,
    ];
    fn data(k: usize) -> u16 {
        DATA0 + 8 * k as u16
    }
    /// The DS/ES pairs the runtime's body runs under.
    fn pairs() -> Vec<(u16, u16)> {
        vec![
            (FLAT, FLAT),
            (ALIAS, FLAT),
            (FLAT, ALIAS),
            (ALIAS, ALIAS),
            (data(0), FLAT),
            (data(0), data(1)),
            (data(1), data(1)),
            (data(2), FLAT),
            (data(3), data(3)),
            (data(4), data(0)),
            (data(5), FLAT),
            (data(6), data(6)),
            (data(7), ALIAS),
            (data(0), data(0)),
        ]
        .into_iter()
        .chain((0..extra()).map(|k| (data(BASES.len() + k as usize), FLAT)))
        .collect()
    }
    /// MACHINE32_EXTRA more data segments (bases 30000h + 1000h k), each a
    /// DS the runtime's body also runs under: more stamps than the engine
    /// retains, as a runtime that reflects with ever new buffers.
    fn extra() -> u32 {
        env("MACHINE32_EXTRA", 0)
    }
    const REAL_DS: [u16; 4] = [0x56a, 0xe000, 0x40, 0x1234];
    const SIXTEEN_DS: [u16; 5] = [DATA16, FLAT, 0x40 + 8 * 6, 0x40 + 8 * 7, 0x40 + 8 * 3];
    const REAL_CS: u16 = (SIXTEEN_BASE >> 4) as u16;

    fn env(name: &str, default: u32) -> u32 {
        std::env::var(name).map_or(default, |s| s.parse().unwrap())
    }
    struct Shape {
        game: u32,
        render: u32,
        k1: u32,
        k2: u32,
        rt: u32,
        sixteen: u32,
        real: u32,
    }
    fn shape() -> Shape {
        let at_least_one = |name, default| env(name, default).max(1);
        Shape {
            game: at_least_one("MACHINE32_GAME", 2000),
            render: at_least_one("MACHINE32_RENDER", 250),
            k1: at_least_one("MACHINE32_K1", 60),
            k2: at_least_one("MACHINE32_K2", 30),
            rt: at_least_one("MACHINE32_RT", 150),
            sixteen: at_least_one("MACHINE32_SIXTEEN", 24),
            real: at_least_one("MACHINE32_REAL", 24),
        }
    }

    /// Body constants: block `i` of body `salt` adds `add` and XORs `xor`.
    fn add(salt: u32, i: u32) -> u32 {
        constant(i.wrapping_mul(7).wrapping_add(salt))
    }
    fn xor(salt: u32, i: u32) -> u32 {
        add(salt, i).rotate_left(11) ^ salt
    }
    const GAME: u32 = 0x100_0000;
    const RENDER: u32 = 0x200_0000;
    const RTB: u32 = 0x300_0000;
    const SIXTEEN: u32 = 0x400_0000;
    const REAL: u32 = 0x500_0000;

    /// A body of `n` blocks ending in a near RET; `stack` adds PUSH EAX and
    /// POP EAX to each block.
    fn body(a: &mut CodeAssembler, salt: u32, n: u32, stack: bool) {
        for i in 0..n {
            let mut next = a.create_label();
            a.add(ebx, add(salt, i) as i32).unwrap();
            if stack {
                a.push(eax).unwrap();
            }
            a.xor(edx, xor(salt, i) as i32).unwrap();
            if stack {
                a.pop(eax).unwrap();
                a.add(eax, 1).unwrap();
            }
            a.jmp(next).unwrap();
            a.set_label(&mut next).unwrap();
        }
        a.ret().unwrap();
    }

    fn far32(a: &mut CodeAssembler, opcode: u8, offset: u32, selector: u16) {
        a.db(&[opcode]).unwrap();
        a.dd(&[offset]).unwrap();
        a.dw(&[selector]).unwrap();
    }

    struct Entries {
        rt_entry: u32,
        rt_tail: u32,
        sixteen_entry: u16,
        real_entry: u16,
        sixteen_back: u16,
    }

    fn assemble(s: &Shape, e: &Entries) -> ([Vec<u8>; 3], Entries) {
        // Flat main at FLAT_CODE (CS base 0).
        let mut a = CodeAssembler::new(32).unwrap();
        let (mut top, mut game, mut render) =
            (a.create_label(), a.create_label(), a.create_label());
        let (mut r1, mut r2) = (a.create_label(), a.create_label());
        a.set_label(&mut top).unwrap();
        a.call(game).unwrap();
        a.mov(ecx, s.k1).unwrap();
        a.set_label(&mut r1).unwrap();
        a.call(render).unwrap();
        a.dec(ecx).unwrap();
        a.jne(r1).unwrap();
        a.mov(dword_ptr(SAVE_ESP), esp).unwrap();
        a.mov(ax, STACK as u32).unwrap();
        a.mov(ss, ax).unwrap();
        a.mov(esp, 0x7c00).unwrap();
        a.mov(ecx, s.k2).unwrap();
        a.set_label(&mut r2).unwrap();
        a.call(render).unwrap();
        a.dec(ecx).unwrap();
        a.jne(r2).unwrap();
        a.mov(ax, FLAT as u32).unwrap();
        a.mov(ss, ax).unwrap();
        a.mov(esp, dword_ptr(SAVE_ESP)).unwrap();
        far32(&mut a, 0x9a, e.rt_entry, RT);
        a.dec(edi).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
        a.set_label(&mut game).unwrap();
        body(&mut a, GAME, s.game, false);
        a.set_label(&mut render).unwrap();
        body(&mut a, RENDER, s.render, true);
        let flat = a.assemble(FLAT_CODE).unwrap();

        // The runtime's 32-bit code, offsets from RT_BASE.
        let mut a = CodeAssembler::new(32).unwrap();
        let (mut entry, mut tail, mut rt) = (a.create_label(), a.create_label(), a.create_label());
        a.set_label(&mut entry).unwrap();
        for (ds_sel, es_sel) in pairs() {
            a.mov(ax, u32::from(ds_sel)).unwrap();
            a.mov(ds, ax).unwrap();
            a.mov(ax, u32::from(es_sel)).unwrap();
            a.mov(es, ax).unwrap();
            a.call(rt).unwrap();
        }
        far32(&mut a, 0xea, u32::from(e.sixteen_entry), CODE16);
        a.set_label(&mut tail).unwrap();
        a.mov(ax, FLAT as u32).unwrap();
        a.mov(ds, ax).unwrap();
        a.mov(es, ax).unwrap();
        a.mov(ss, ax).unwrap();
        a.mov(esp, dword_ptr(SAVE_ESP)).unwrap();
        a.sub(esp, 8).unwrap();
        a.retf().unwrap();
        a.set_label(&mut rt).unwrap();
        body(&mut a, RTB, s.rt, false);
        let result = a
            .assemble_options(
                0,
                iced_x86::BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS,
            )
            .unwrap();
        let rt_entry = result.label_ip(&entry).unwrap() as u32;
        let rt_tail = result.label_ip(&tail).unwrap() as u32;
        let rt_bytes = result.inner.code_buffer;

        // 16-bit code at SIXTEEN_BASE: protected (CODE16) and real (REAL_CS)
        // share its offsets.
        let mut a = CodeAssembler::new(16).unwrap();
        let (mut entry, mut real, mut back) =
            (a.create_label(), a.create_label(), a.create_label());
        let (mut sixteen, mut real_body) = (a.create_label(), a.create_label());
        a.set_label(&mut entry).unwrap();
        a.mov(ax, DATA16 as u32).unwrap();
        for seg in [ds, es, fs, gs] {
            a.mov(seg, ax).unwrap();
        }
        a.mov(ss, ax).unwrap();
        a.mov(sp, 0x5ff0).unwrap();
        // LIDT [IDTR_REAL]; the 16-bit form loads a 24-bit base.
        a.db(&[0x0f, 0x01, 0x1e]).unwrap();
        a.dw(&[IDTR_REAL]).unwrap();
        a.call(sixteen).unwrap();
        // MOV EAX, CR0; AND AL, FEh; MOV CR0, EAX; JMP FAR REAL_CS:real.
        a.db(&[0x0f, 0x20, 0xc0, 0x24, 0xfe, 0x0f, 0x22, 0xc0, 0xea])
            .unwrap();
        a.dw(&[e.real_entry, REAL_CS]).unwrap();
        a.set_label(&mut real).unwrap();
        a.xor(ax, ax).unwrap();
        for seg in [ss, fs, gs] {
            a.mov(seg, ax).unwrap();
        }
        for seg in REAL_DS {
            a.mov(ax, u32::from(seg)).unwrap();
            a.mov(ds, ax).unwrap();
            a.mov(es, ax).unwrap();
            a.call(real_body).unwrap();
        }
        // MOV EAX, CR0; OR AL, 1; MOV CR0, EAX; JMP FAR CODE16:back.
        a.db(&[0x0f, 0x20, 0xc0, 0x0c, 0x01, 0x0f, 0x22, 0xc0, 0xea])
            .unwrap();
        a.dw(&[e.sixteen_back, CODE16]).unwrap();
        a.set_label(&mut back).unwrap();
        a.mov(ax, DATA16 as u32).unwrap();
        a.mov(ss, ax).unwrap();
        for seg in [ds, es, fs, gs] {
            a.mov(seg, ax).unwrap();
        }
        a.call(sixteen).unwrap();
        a.db(&[0x0f, 0x01, 0x1e]).unwrap();
        a.dw(&[IDTR_PROTECTED]).unwrap();
        for seg in SIXTEEN_DS {
            a.mov(ax, u32::from(seg)).unwrap();
            a.mov(ds, ax).unwrap();
            a.call(sixteen).unwrap();
        }
        a.mov(ax, FLAT as u32).unwrap();
        a.mov(fs, ax).unwrap();
        a.mov(gs, ax).unwrap();
        // JMP FAR DWORD RT:tail (66 EA id iw).
        a.db(&[0x66]).unwrap();
        far32(&mut a, 0xea, e.rt_tail, RT);
        a.set_label(&mut sixteen).unwrap();
        body(&mut a, SIXTEEN, s.sixteen, false);
        a.set_label(&mut real_body).unwrap();
        body(&mut a, REAL, s.real, false);
        let result = a
            .assemble_options(
                0,
                iced_x86::BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS,
            )
            .unwrap();
        let entries = Entries {
            rt_entry,
            rt_tail,
            sixteen_entry: result.label_ip(&entry).unwrap() as u16,
            real_entry: result.label_ip(&real).unwrap() as u16,
            sixteen_back: result.label_ip(&back).unwrap() as u16,
        };
        ([flat, rt_bytes, result.inner.code_buffer], entries)
    }

    fn descriptor(base: u32, limit: u32, access: u8, flags: u8) -> [u8; 8] {
        let (b, l) = (base.to_le_bytes(), limit.to_le_bytes());
        [
            l[0],
            l[1],
            b[0],
            b[1],
            b[2],
            access,
            flags | (l[2] & 0x0f),
            b[3],
        ]
    }

    pub(super) fn setup() -> (Vm, x86jit_core::Vcpu) {
        let s = shape();
        let zero = Entries {
            rt_entry: 0,
            rt_tail: 0,
            sixteen_entry: 0,
            real_entry: 0,
            sixteen_back: 0,
        };
        let (_, entries) = assemble(&s, &zero);
        let ([flat, rt, sixteen], again) = assemble(&s, &entries);
        assert_eq!(again.rt_tail, entries.rt_tail);
        assert_eq!(again.sixteen_back, entries.sixteen_back);
        let mut vm = Vm::new(VmConfig::flat(MEMORY));
        vm.set_cpu_mode(CpuMode::Compat32);
        vm.set_cpu_model(match std::env::var("MACHINE32_MODEL").as_deref() {
            Ok("386") => CpuModel::I80386,
            _ => CpuModel::I80486Dx,
        });
        vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.write_bytes(FLAT_CODE, &flat).unwrap();
        vm.write_bytes(u64::from(RT_BASE), &rt).unwrap();
        vm.write_bytes(u64::from(SIXTEEN_BASE), &sixteen).unwrap();
        let mut gdt = vec![
            [0; 8],
            descriptor(0, 0xfffff, 0x9b, 0xc0),
            descriptor(0, 0xfffff, 0x93, 0xc0),
            descriptor(0, 0xfffff, 0x93, 0xc0),
            descriptor(STACK_BASE, 0x7cff, 0x93, 0x40),
            descriptor(RT_BASE, 0xffff, 0x9b, 0x40),
            descriptor(SIXTEEN_BASE, 0xffff, 0x9b, 0),
            descriptor(0, 0xffff, 0x93, 0),
        ];
        gdt.extend(BASES.map(|base| descriptor(base, 0xffff, 0x93, 0)));
        gdt.extend((0..extra()).map(|k| descriptor(0x3_0000 + 0x1000 * k, 0xffff, 0x93, 0)));
        vm.write_bytes(GDT, &gdt.concat()).unwrap();
        // The IDTR images LIDT loads: protected (limit 7FFh at 7000h) and
        // the real-mode IVT.
        vm.write_bytes(u64::from(IDTR_PROTECTED), &[0xff, 0x07, 0, 0x70, 0, 0])
            .unwrap();
        vm.write_bytes(u64::from(IDTR_REAL), &[0xff, 0x03, 0, 0, 0, 0])
            .unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.legacy.gdtr.base = GDT as u32;
        cpu.cpu.legacy.gdtr.limit = (8 * gdt.len() - 1) as u16;
        cpu.cpu.legacy.idtr.base = 0x7000;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        cpu.cpu.legacy.cr0 = 0x11;
        cpu.cpu.legacy.cpl = 0;
        for (i, segment) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
            segment.base = 0;
            segment.limit = u32::MAX;
            segment.attributes = (if i == 0 { 0x9b } else { 0x93 }) | 0xc000;
            segment.unusable = 0;
        }
        cpu.cpu.cs = CODE32;
        cpu.cpu.ds = FLAT;
        cpu.cpu.es = FLAT;
        cpu.cpu.ss = FLAT;
        cpu.cpu.fs = FLAT;
        cpu.cpu.gs = FLAT;
        cpu.set_a20_enabled(true);
        (vm, cpu)
    }

    fn sums(salt: u32, n: u32) -> (u32, u32) {
        (0..n).fold((0, 0), |(a, x), i| {
            (a.wrapping_add(add(salt, i)), x ^ xor(salt, i))
        })
    }

    /// Runs `n` frames and checks EBX and EDX; returns the retired count.
    pub(super) fn execute(vm: &Vm, cpu: &mut x86jit_core::Vcpu, n: u64) -> u64 {
        let s = shape();
        let per_frame = [
            (GAME, s.game, 1),
            (RENDER, s.render, s.k1 + s.k2),
            (RTB, s.rt, pairs().len() as u32),
            (SIXTEEN, s.sixteen, 2 + SIXTEEN_DS.len() as u32),
            (REAL, s.real, REAL_DS.len() as u32),
        ];
        let (mut sum, mut mix) = (0u32, 0u32);
        for (salt, blocks, calls) in per_frame {
            let (a, x) = sums(salt, blocks);
            let times = (u64::from(calls) * n) as u32;
            sum = sum.wrapping_add(a.wrapping_mul(times));
            if times & 1 != 0 {
                mix ^= x;
            }
        }
        cpu.set_reg(Reg::Rip, FLAT_CODE);
        for reg in [Reg::Rax, Reg::Rbx, Reg::Rdx] {
            cpu.set_reg(reg, 0);
        }
        cpu.set_reg(Reg::Rdi, n);
        cpu.set_reg(Reg::Rsp, u64::from(FLAT_ESP));
        let before = cpu.retired_instructions();
        let exit = cpu.run(vm, None);
        assert!(
            matches!(exit, Exit::Hlt),
            "{exit:?} at {:#x}",
            cpu.reg(Reg::Rip)
        );
        assert_eq!(cpu.reg(Reg::Rbx) as u32, sum, "runtime EBX");
        assert_eq!(cpu.reg(Reg::Rdx) as u32, mix, "runtime EDX");
        cpu.retired_instructions() - before
    }
}

fn run_runtime(round: usize, n: u64) {
    let (vm, mut cpu) = runtime::setup();
    runtime::execute(&vm, &mut cpu, 2);
    #[cfg(feature = "test-hooks")]
    vm.relift_diagnostics(true);
    #[cfg(feature = "test-hooks")]
    x86jit_core::Vcpu::take_machine_counters_for_test();
    let host_before = instructions();
    let start = Instant::now();
    let retired = runtime::execute(&vm, &mut cpu, n);
    let seconds = start.elapsed().as_secs_f64();
    let host = instructions() - host_before;
    #[cfg(feature = "test-hooks")]
    eprintln!(
        "counters smc-live runtime flat: {:?}; proof bytes (now, peak) {:?}",
        x86jit_core::Vcpu::take_machine_counters_for_test()
            .map(|c| (c as f64 * 1e6 / retired.max(1) as f64).round()),
        cpu.take_proof_bytes_for_test()
    );
    #[cfg(feature = "test-hooks")]
    if std::env::var_os("MACHINE32_DIAG").is_some() {
        eprintln!("{}", vm.relift_diagnostics(true));
    }
    println!(
        "smc-live-runtime,flat,{round},{n},{retired},{retired},{seconds:.9},{:.3},{:.3}",
        retired as f64 / seconds / 1e6,
        host as f64 / retired as f64
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n = args.get(1).map_or(200_000, |s| s.parse::<u64>().unwrap());
    let rounds = args.get(2).map_or(5, |s| s.parse::<usize>().unwrap());
    assert!(n > 0 && n <= u32::MAX as u64);
    let paths: Vec<&str> = args
        .get(3)
        .map_or_else(|| PATHS.to_vec(), |s| vec![s.as_str()]);
    let works: Vec<&str> = match args.get(4).map(String::as_str) {
        None => WORKS.to_vec(),
        Some("all") => WORKS.iter().chain(&FORMS).copied().collect(),
        Some(list) => list.split(',').collect(),
    };
    println!(
        "workload,path,round,iterations,instructions,retired_counter,seconds,mips,host_per_step"
    );
    for round in 0..rounds {
        for &work in &works {
            for &path in &paths {
                if work == "runtime" {
                    if path == "flat" {
                        run_runtime(round, n);
                    }
                    continue;
                }
                let (vm, mut cpu, fields) = setup(path, work);
                execute(&vm, &mut cpu, work, &fields, 2000);
                #[cfg(feature = "test-hooks")]
                vm.relift_diagnostics(true);
                let retired_before = cpu.retired_instructions();
                #[cfg(feature = "test-hooks")]
                x86jit_core::Vcpu::take_machine_counters_for_test();
                let host_before = instructions();
                let start = Instant::now();
                let expect = execute(&vm, &mut cpu, work, &fields, n);
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
                #[cfg(feature = "test-hooks")]
                if std::env::var_os("MACHINE32_DIAG").is_some() {
                    eprintln!("{}", vm.relift_diagnostics(true));
                }
                println!(
                    "smc-live-{work},{path},{round},{n},{retired},{retired},{seconds:.9},{:.3},{:.3}",
                    retired as f64 / seconds / 1e6,
                    host as f64 / retired as f64
                );
            }
        }
    }
}
