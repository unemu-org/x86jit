//! A hot loop that patches its own instruction immediates (xj-to9.45).
//! Arguments: iterations (default 200000), rounds (default 5), optional path.
//! MACHINE32_MODEL=486 selects I80486Dx; the default is I80386.
//!
//! Twenty-one small blocks share one 4 KiB code page at CPL 0. One block is
//! `MOV EAX, imm32`; every seven iterations (about 430 retired instructions)
//! a 4-byte `ADD [imm32], 1` on the same page changes that immediate and the
//! loop keeps executing, as a renderer that patches its own inner loop does.
//! Paths: flat (unpaged Flat32), seg (CS.D=1, 64 KiB segments based at the
//! code page: Segmented), paged (Flat32 with identity paging).
//! The guest result is checked against a model of the loop. With
//! `--features test-hooks`, stderr gives lifts and proof hits per million
//! retired.
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
const CODE: u64 = 0x10000;
const BLOCKS: u32 = 19;
const PATCHED: u32 = 9;
const PERIOD: u32 = 7;
const MEMORY: u64 = 0x20000;

fn constant(i: u32) -> u32 {
    i.wrapping_mul(0x0101_0101) | 1
}

/// The loop at IP `org`; returns the bytes and the immediate's offset.
fn code(org: u64) -> (Vec<u8>, u64) {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    let mut done = a.create_label();
    let mut patch = a.create_label();
    let mut imm = a.create_label();
    a.set_label(&mut top).unwrap();
    for i in 0..BLOCKS {
        let mut next = a.create_label();
        if i == PATCHED {
            // MOV EAX, imm32 (B8 id): the immediate is one byte in.
            a.db(&[0xb8]).unwrap();
            a.set_label(&mut imm).unwrap();
            a.dd(&[0x1111_1111]).unwrap();
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
    a.set_label(&mut patch).unwrap();
    a.mov(ecx, PERIOD).unwrap();
    // The immediate's address is fixed below, after assembly.
    a.add(dword_ptr(0x7fff_fff0), 1).unwrap();
    a.jmp(top).unwrap();
    a.set_label(&mut done).unwrap();
    a.hlt().unwrap();
    let result = a
        .assemble_options(
            org,
            iced_x86::BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS,
        )
        .unwrap();
    let imm = result.label_ip(&imm).unwrap();
    let mut bytes = result.inner.code_buffer;
    let marker = 0x7fff_fff0u32.to_le_bytes();
    let at = bytes.windows(4).position(|w| w == marker).unwrap();
    bytes[at..at + 4].copy_from_slice(&(imm as u32).to_le_bytes());
    (bytes, imm - org)
}

struct Model {
    eax: u32,
    ebx: u32,
    edx: u32,
    imm: u32,
    retired: u64,
}

fn model(n: u64, imm: u32) -> Model {
    let mut m = Model {
        eax: 0,
        ebx: 0,
        edx: 0,
        imm,
        retired: 0,
    };
    let mut ecx = PERIOD;
    let mut edi = n;
    loop {
        for i in 0..BLOCKS {
            if i == PATCHED {
                m.eax = m.imm;
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
            m.imm = m.imm.wrapping_add(1);
            m.retired += 3;
        }
    }
}

fn setup(path: &str) -> (Vm, x86jit_core::Vcpu, u64) {
    let seg = path == "seg";
    let paged = path == "paged";
    let mut vm = Vm::new(VmConfig::flat(MEMORY));
    vm.set_cpu_mode(CpuMode::Compat32);
    vm.set_cpu_model(match std::env::var("MACHINE32_MODEL").as_deref() {
        Ok("486") => CpuModel::I80486Dx,
        _ => CpuModel::I80386,
    });
    vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    let org = if seg { 0 } else { CODE };
    let (bytes, imm) = code(org);
    vm.write_bytes(CODE, &bytes).unwrap();
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
        segment.base = if seg { CODE as u32 } else { 0 };
        // Byte-granular 64 KiB limits keep CS.D and SS.B at 1: Segmented.
        segment.limit = if seg { 0xffff } else { u32::MAX };
        segment.attributes = (if i == 0 { 0x9b } else { 0x93 }) | 0x4000;
        segment.unusable = 0;
    }
    cpu.set_a20_enabled(true);
    (vm, cpu, CODE + imm)
}

fn execute(vm: &Vm, cpu: &mut x86jit_core::Vcpu, imm_at: u64, n: u64) -> Model {
    let mut imm = [0; 4];
    vm.read_bytes(imm_at, &mut imm).unwrap();
    let expect = model(n, u32::from_le_bytes(imm));
    let org = if cpu.cpu.legacy.segments[0].base == 0 {
        CODE
    } else {
        0
    };
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
    vm.read_bytes(imm_at, &mut imm).unwrap();
    assert_eq!(u32::from_le_bytes(imm), expect.imm);
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
    println!(
        "workload,path,round,iterations,instructions,retired_counter,seconds,mips,host_per_step"
    );
    for round in 0..rounds {
        for &path in &paths {
            let (vm, mut cpu, imm_at) = setup(path);
            execute(&vm, &mut cpu, imm_at, 2000);
            let retired_before = cpu.retired_instructions();
            #[cfg(feature = "test-hooks")]
            x86jit_core::Vcpu::take_machine_counters_for_test();
            let host_before = instructions();
            let start = Instant::now();
            let expect = execute(&vm, &mut cpu, imm_at, n);
            let seconds = start.elapsed().as_secs_f64();
            let host = instructions() - host_before;
            let retired = cpu.retired_instructions() - retired_before;
            assert_eq!(retired, expect.retired);
            // Per million retired: blocks, reselections, lifts, proof hits, walks.
            #[cfg(feature = "test-hooks")]
            eprintln!(
                "counters smc-patch {path}: {:?}",
                x86jit_core::Vcpu::take_machine_counters_for_test()
                    .map(|c| (c as f64 * 1e6 / retired.max(1) as f64).round())
            );
            println!(
                "smc-patch,{path},{round},{n},{retired},{retired},{seconds:.9},{:.3},{:.3}",
                retired as f64 / seconds / 1e6,
                host as f64 / retired as f64
            );
        }
    }
}
