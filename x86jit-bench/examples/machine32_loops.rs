//! Warm, finite five-instruction loops for the compiled-Machine J0 decision.
//! Arguments: iterations (default 2M), rounds (default 5), optional path/workload.
//! MACHINE32_MODEL=486 selects I80486Dx, 286 the I80286 (the unpaged
//! protected16 paths only); the default is I80386.
//! farcall uses MACHINE32_FAR_INTERVAL guest instructions per call (default 25).
//! farseg reloads DS/ES and calls far in protected16 or paged V86.
//! segmix mirrors a PC-class embedder's 16-bit protected and V86 body: per 25
//! retired, one DS and one ES reload, a near and a far CALL/RET pair (the far
//! callee in a second code segment) and three memory operands.
//! The 32seg paths run CS.D=1 code in 64 KiB segments with non-zero bases
//! (SS.B=1), as extended-memory runtimes and 16:32 driver-style code do;
//! their segmix is the same body in 32-bit code.
//! farcall-sliced and farseg-sliced use 64-step calls and board-style writes.
//! These cases place data 256 KiB from the descriptor page.
//! Application retirement is derived from the validated finite guest loop.
//! Path machine-paged-v86-gp (workloads v86gp, v86gp-free) runs paged V86 at
//! IOPL 0 with an IN every 266 retired that a ring-0 #GP monitor emulates;
//! v86gp-free replaces the IN and monitor with the same count of V86 work.
use std::sync::Arc;
use std::time::Instant;
use x86jit_core::{
    BlockKey, CachedBlock, CpuMode, CpuModel, CpuState, ExceptionDelivery, Exit, Prot, Reg,
    RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::{JitBackend, OptLevel};

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

const PATHS: [&str; 16] = [
    "machine-real16",
    "machine-protected32",
    "machine-paged32",
    "machine-protected32-cpl3",
    "machine-paged32-cpl3",
    "machine-protected16",
    "machine-paged16",
    "machine-protected16-cpl3",
    "machine-paged16-cpl3",
    "machine-paged-v86",
    "machine-protected32seg",
    "machine-paged32seg",
    "machine-protected32seg-cpl3",
    "machine-paged32seg-cpl3",
    "application-interp32",
    "application-jit32",
];
const WORKS: [&str; 6] = [
    "register", "memory", "callret", "farseg", "farcall", "segmix",
];

fn farcall_interval() -> u64 {
    let n = std::env::var("MACHINE32_FAR_INTERVAL").map_or(25, |s| s.parse().unwrap());
    assert!((5..=100).contains(&n));
    n
}
const CODE: u64 = 0x10000;
const DATA: u64 = 0x4000;
const STACK: u64 = 0x8000;

fn code(work: &str, real: bool) -> Vec<u8> {
    match work {
        // INC A; ADD B,A; XOR D,B; DEC DI; JNZ start.
        "register" => vec![0x40, 0x01, 0xc3, 0x31, 0xda, 0x4f, 0x75, 0xf8, 0xf4],
        // Same effects, four redundant segment prefixes: isolates fetch-length cost.
        "register-pad" => vec![
            0x2e, 0x40, 0x2e, 0x01, 0xc3, 0x2e, 0x31, 0xda, 0x2e, 0x4f, 0x75, 0xf4, 0xf4,
        ],
        // MOV A,[SI]; ADD A,1; MOV [SI],A; DEC DI; JNZ start.
        "memory" => vec![
            0x8b,
            if real { 0x04 } else { 0x06 },
            0x83,
            0xc0,
            0x01,
            0x89,
            if real { 0x04 } else { 0x06 },
            0x4f,
            0x75,
            0xf6,
            0xf4,
        ],
        // CALL sub; DEC DI; JNZ start; HLT; sub: INC A; RET.
        "callret" if real => vec![0xe8, 4, 0, 0x4f, 0x75, 0xfa, 0xf4, 0x40, 0xc3],
        "callret" => vec![0xe8, 4, 0, 0, 0, 0x4f, 0x75, 0xf8, 0xf4, 0x40, 0xc3],
        _ => panic!("unknown workload"),
    }
}

fn setup(path: &str, work: &str) -> (Vm, x86jit_core::Vcpu) {
    let sliced = work.ends_with("-sliced");
    let work = work.trim_end_matches("-sliced");
    let data = if sliced { 0x40000 } else { DATA };
    let gdt = if sliced { 0x800 } else { 0x3000 };
    let memory_size = if sliced { 0x80000 } else { 0x20000 };
    let mut vm = if path == "application-jit32" {
        Vm::with_backend(
            VmConfig::flat(memory_size),
            Box::new(JitBackend::with_opt_level(OptLevel::Speed)),
        )
    } else {
        Vm::new(VmConfig::flat(memory_size))
    };
    vm.set_cpu_mode(CpuMode::Compat32);
    let machine = path.starts_with("machine");
    let real = path == "machine-real16";
    let seg16 = path.contains("protected16") || path.contains("paged16");
    let seg32 = path.contains("32seg");
    let v86 = path == "machine-paged-v86";
    let user = path.ends_with("-cpl3") || v86;
    let paged = path.starts_with("machine-paged");
    if machine {
        vm.set_cpu_model(match std::env::var("MACHINE32_MODEL").as_deref() {
            Ok("486") => CpuModel::I80486Dx,
            Ok("286") => CpuModel::I80286,
            _ => CpuModel::I80386,
        });
    }
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.map(0, memory_size as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    let bytes = if work == "segmix" {
        if v86 {
            segmix_code((data >> 4) as u16, 0x1100, 16)
        } else {
            let rpl = if user { 3 } else { 0 };
            segmix_code(16 | rpl, 32 | rpl, if seg32 { 32 } else { 16 })
        }
    } else if work == "farseg" || work == "farcall" {
        let short = seg16 || v86;
        let mut bytes = if work == "farseg" {
            // MOV DX,DS; MOV DS,DX; MOV ES,DX.
            vec![0x8c, 0xda, 0x8e, 0xda, 0x8e, 0xc2]
        } else {
            vec![0x90; (farcall_interval() - 5) as usize]
        };
        if sliced {
            let load = if short {
                vec![0x8b, 0x16, 0, 0] // MOV DX,[0].
            } else {
                let mut load = vec![0x8b, 0x15]; // MOV EDX,[data].
                load.extend((data as u32).to_le_bytes());
                load
            };
            if work == "farcall" {
                bytes.splice(..1, load);
            } else {
                bytes.extend(load);
            }
        }
        let target = CODE + bytes.len() as u64 + if short { 9 } else { 11 };
        bytes.push(0x9a); // CALL FAR sub.
        if short {
            bytes.extend(((target - CODE) as u16).to_le_bytes());
        } else if seg32 {
            bytes.extend(((target - CODE) as u32).to_le_bytes());
        } else {
            bytes.extend((target as u32).to_le_bytes());
        }
        let selector: u16 = if v86 {
            0x1000
        } else if user {
            11
        } else {
            8
        };
        bytes.extend(selector.to_le_bytes());
        bytes.extend([0x4f, 0x75]); // DEC DI; JNZ start.
        bytes.push((-(bytes.len() as i16 + 1)) as u8);
        bytes.extend([0xf4, 0x40, 0xcb]); // HLT; sub: INC A; RETF.
        bytes
    } else {
        code(work, real || seg16 || v86)
    };
    vm.write_bytes(CODE, &bytes).unwrap();
    if work == "segmix" {
        // Far callee at offset 0 of the second code segment: XOR DX,AX; RETF.
        vm.write_bytes(CODE + 0x1000, &[0x31, 0xc2, 0xcb]).unwrap();
    }
    if path == "application-single32" {
        let bytes = code(work, false);
        let mut decoder =
            iced_x86::Decoder::with_ip(32, &bytes, CODE, iced_x86::DecoderOptions::NONE);
        while decoder.can_decode() {
            let addr = decoder.decode().ip();
            let ir = x86jit_core::lift::lift_one(
                &vm.mem,
                x86jit_core::lift::FetchAddr { pa: addr, ip: addr },
                CpuMode::Compat32,
            )
            .unwrap();
            let len = ir.guest_len;
            vm.cache.insert(
                BlockKey::new(addr, CpuMode::Compat32),
                CachedBlock::Interpreted(Arc::new(ir)),
                vec![(addr, len)],
                |_| vm.mem.mark_code(addr, len),
            );
        }
    }
    if paged {
        let rights = if user { 7 } else { 3 };
        vm.write_bytes(0x1000, &(0x2000u32 | rights).to_le_bytes())
            .unwrap();
        for page in 0..(memory_size / 4096) as u32 {
            vm.write_bytes(
                0x2000 + u64::from(4 * page),
                &((page * 4096) | rights).to_le_bytes(),
            )
            .unwrap();
        }
    }
    let mut cpu = vm.new_vcpu();
    if real {
        cpu.cpu = CpuState::legacy_real_entry([0x1000, 0, 0, 0, 0, 0]);
    } else if machine {
        // Same retained-cache fixture as legacy_paging.rs, with flat bases.
        cpu.cpu.legacy.cr0 = if paged { 0x80000001 } else { 1 };
        cpu.cpu.legacy.cr3 = 0x1000;
        cpu.cpu.legacy.cpl = if user { 3 } else { 0 };
        for (i, segment) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
            segment.base = 0;
            segment.limit = u32::MAX;
            segment.attributes =
                (if i == 0 { 0x9b } else { 0x93 }) | 0x4000 | (u16::from(cpu.cpu.legacy.cpl) << 5);
            segment.unusable = 0;
        }
    }
    if seg16 || seg32 || (work == "farcall" && machine) {
        cpu.cpu.cs = if user { 11 } else { 8 };
        cpu.cpu.ds = if user { 19 } else { 16 };
        cpu.cpu.es = cpu.cpu.ds;
        cpu.cpu.ss = if user { 27 } else { 24 };
        cpu.cpu.legacy.gdtr.base = gdt as u32;
        cpu.cpu.legacy.gdtr.limit = 39;
        for (i, segment) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
            segment.base = if !seg16 && !seg32 {
                0
            } else {
                match i {
                    0 => CODE as u32,
                    3 => 0x5000,
                    _ => data as u32,
                }
            };
            if seg16 {
                segment.limit = if i == 3 { 0x3fff } else { 0xfff };
                segment.attributes &= !0x4000;
            } else if seg32 {
                // Byte-granular 64 KiB limits; CS.D and SS.B stay 1.
                segment.limit = 0xffff;
            }
            if i < 4 && i != 2 {
                let index = match i {
                    0 => 1,
                    1 => 2,
                    _ => 3,
                };
                let limit = if seg16 || seg32 {
                    segment.limit
                } else {
                    0xfffff
                };
                let low = (limit & 0xffff) | (segment.base << 16);
                let high = ((segment.base >> 16) & 0xff)
                    | (u32::from(segment.attributes) << 8)
                    | (limit & 0xf0000)
                    | if seg16 || seg32 { 0 } else { 0x800000 };
                vm.write_bytes(gdt + index * 8, &low.to_le_bytes()).unwrap();
                vm.write_bytes(gdt + 4 + index * 8, &high.to_le_bytes())
                    .unwrap();
                if i == 0 {
                    // Index 4: the same code segment, based one page higher.
                    let base = segment.base + 0x1000;
                    let low = (limit & 0xffff) | (base << 16);
                    let high = (high & !0xff00_00ff) | ((base >> 16) & 0xff) | (base & 0xff00_0000);
                    vm.write_bytes(gdt + 32, &low.to_le_bytes()).unwrap();
                    vm.write_bytes(gdt + 36, &high.to_le_bytes()).unwrap();
                }
            }
        }
    }
    if v86 {
        cpu.cpu = CpuState::legacy_real_entry([
            0x1000,
            (data >> 4) as u16,
            (data >> 4) as u16,
            0x500,
            0,
            0,
        ]);
        cpu.cpu.legacy.cr0 = 0x80000001;
        cpu.cpu.legacy.cr3 = 0x1000;
        cpu.cpu.legacy.cpl = 3;
        cpu.cpu.rflags_system = 0x23000; // VM=1, IOPL=3 (SDM Vol 3A §21.2).
    }
    cpu.set_a20_enabled(true);
    (vm, cpu)
}

fn execute(vm: &Vm, cpu: &mut x86jit_core::Vcpu, real: bool, n: u64, sliced: bool) {
    // Real16 loops use DI, so split into runs without adding guest setup ops.
    let mut remaining = n;
    while remaining != 0 {
        let count = if real {
            remaining.min(60000)
        } else {
            remaining
        };
        let seg16 = cpu.cpu.legacy.cr0 & 1 != 0 && cpu.cpu.legacy.segments[0].base == CODE as u32;
        cpu.set_reg(Reg::Rip, if real || seg16 { 0 } else { CODE });
        cpu.set_reg(Reg::Rdi, count);
        cpu.set_reg(Reg::Rsi, if seg16 { 0 } else { DATA });
        cpu.set_reg(Reg::Rsp, if seg16 { 0x3000 } else { STACK });
        let exit = if sliced {
            loop {
                cpu.disable_delivery_log();
                cpu.set_a20_enabled(true);
                // A board may mirror live registers between device deadlines.
                for reg in [Reg::Rip, Reg::Rax, Reg::Rsp] {
                    cpu.set_reg(reg, cpu.reg(reg));
                }
                let run = cpu
                    .run_steps(vm, 64, x86jit_core::StepOptions::default())
                    .unwrap();
                if let x86jit_core::StepStop::Exit(exit) = run.into_stop() {
                    if !matches!(exit, Exit::BudgetExhausted) {
                        break exit;
                    }
                }
            }
        } else {
            cpu.run(vm, None)
        };
        if cpu.cpu.legacy.cpl == 3 {
            // SDM Vol 2A HLT: CPL 3 stops at #GP(0), without retiring HLT.
            assert!(matches!(exit, Exit::Exception { vector: 13, .. }));
            assert_eq!(
                exit.fault_metadata().unwrap().error_code(),
                x86jit_core::fault::ErrorCode::bits(0, 0xffff)
            );
        } else {
            assert!(matches!(exit, Exit::Hlt));
        }
        assert_eq!(cpu.reg(Reg::Rdi), 0);
        assert_eq!(cpu.reg(Reg::Rsp), if seg16 { 0x3000 } else { STACK });
        remaining -= count;
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n = args.get(1).map_or(2_000_000, |s| s.parse::<u64>().unwrap());
    let rounds = args.get(2).map_or(5, |s| s.parse::<usize>().unwrap());
    assert!(n > 0 && n <= u32::MAX as u64);
    if args.get(3).is_some_and(|s| s == "components") {
        components(n, rounds);
        return;
    }
    if args.get(3).is_some_and(|s| s == "machine-paged-v86-gp") {
        v86gp(n, rounds, args.get(4).map(String::as_str));
        return;
    }
    println!(
        "workload,path,round,iterations,instructions,retired_counter,seconds,mips,host_per_step"
    );
    for round in 0..rounds {
        let works: Vec<&str> = args
            .get(4)
            .map_or_else(|| WORKS.to_vec(), |s| vec![s.as_str()]);
        for work in works {
            let sliced = work.ends_with("-sliced");
            let base_work = work.trim_end_matches("-sliced");
            let paths: Vec<&str> = args.get(3).map_or_else(
                || {
                    (0..PATHS.len())
                        .map(|offset| PATHS[(round + offset) % PATHS.len()])
                        .collect()
                },
                |s| vec![s.as_str()],
            );
            for path in paths {
                let v86 = path == "machine-paged-v86";
                let seg16 = path.contains("protected16") || path.contains("paged16");
                let seg32 = path.contains("32seg");
                if ((base_work == "farseg" || base_work == "segmix") && !seg16 && !seg32 && !v86)
                    || (base_work == "segmix" && sliced)
                    || (base_work == "farcall"
                        && (!path.starts_with("machine") || path == "machine-real16"))
                {
                    continue;
                }
                let real = path == "machine-real16" || seg16 || v86;
                let (vm, mut cpu) = setup(path, work);
                execute(&vm, &mut cpu, real, 10000, sliced);
                cpu.set_reg(Reg::Rax, 0);
                cpu.set_reg(Reg::Rbx, 0);
                cpu.set_reg(Reg::Rdx, 0);
                vm.write_bytes(DATA, &[0; 4]).unwrap();
                vm.write_bytes(DATA + 0x100, &[0; 2]).unwrap();
                let retired_before = cpu.retired_instructions();
                #[cfg(feature = "test-hooks")]
                x86jit_core::Vcpu::take_machine_counters_for_test();
                let host_before = instructions();
                let start = Instant::now();
                execute(&vm, &mut cpu, real, n, sliced);
                let seconds = start.elapsed().as_secs_f64();
                let host = instructions() - host_before;
                let retired = cpu.retired_instructions() - retired_before;
                // Per 1000 retired, on stderr: blocks, reselections, lifts, proof hits.
                #[cfg(feature = "test-hooks")]
                eprintln!(
                    "counters {work} {path}: {:?}",
                    x86jit_core::Vcpu::take_machine_counters_for_test()
                        .map(|c| (c * 1000) as f64 / retired.max(1) as f64)
                );
                let count = (match base_work {
                    "farseg" => 8 + u64::from(sliced),
                    "farcall" => farcall_interval(),
                    "segmix" => 25,
                    _ => 5,
                }) * n
                    + if real {
                        n.div_ceil(60000) * u64::from(!path.ends_with("-cpl3") && !v86)
                    } else {
                        u64::from(!path.ends_with("-cpl3") && !v86)
                    };
                if path.starts_with("machine") {
                    assert_eq!(retired, count);
                }
                let mask = if real { 0xffff } else { 0xffff_ffff };
                if base_work == "segmix" {
                    let mut word = [0; 2];
                    vm.read_bytes(DATA + 0x100, &mut word).unwrap();
                    assert_eq!(u64::from(u16::from_le_bytes(word)), (3 * n) & 0xffff);
                    assert_eq!(cpu.reg(Reg::Rax) & 0xffff, (3 * n + 1) & 0xffff);
                } else {
                    assert_eq!(cpu.reg(Reg::Rax) & mask, n & mask);
                }
                if work == "memory" {
                    let mut data = [0; 4];
                    vm.read_bytes(DATA, &mut data).unwrap();
                    assert_eq!(u64::from(u32::from_le_bytes(data)) & mask, n & mask);
                }
                println!(
                    "{work},{path},{round},{n},{count},{retired},{seconds:.9},{:.3},{:.3}",
                    count as f64 / seconds / 1e6,
                    host as f64 / count as f64
                );
            }
        }
    }
}

// 25 retired per iteration (see the module comment). DS and ES reload `data`
// (a selector, or a V86 paragraph); the far callee is 1100h:0 in V86 and
// selector 20h | RPL otherwise, both one page above the loop's code. `bits`
// 32 assembles the same instructions with 32-bit operands and addresses.
fn segmix_code(data: u16, far: u16, bits: u32) -> Vec<u8> {
    use iced_x86::code_asm::*;
    if bits == 32 {
        return segmix32_code(data, far);
    }
    let mut a = CodeAssembler::new(16).unwrap();
    let mut top = a.create_label();
    let mut near = a.create_label();
    a.set_label(&mut top).unwrap();
    a.mov(bx, u32::from(data)).unwrap();
    a.add(bx, 0).unwrap();
    a.mov(ds, bx).unwrap();
    a.mov(es, bx).unwrap();
    a.mov(ax, word_ptr(0x100)).unwrap();
    a.add(ax, 3).unwrap();
    a.mov(word_ptr(0x100), ax).unwrap();
    a.push(ax).unwrap();
    a.xor(ax, 0x55aa).unwrap();
    a.pop(ax).unwrap();
    a.call(near).unwrap();
    a.db(&[0x9a, 0, 0, far as u8, (far >> 8) as u8]).unwrap(); // CALL FAR far:0.
    a.mov(word_ptr(0x102).es(), ax).unwrap();
    if std::env::var("MACHINE32_MODEL").as_deref() == Ok("286") {
        // The 80286 has no operand-size prefix: the same unit on BP.
        a.inc(bp).unwrap();
    } else {
        a.inc(ebp).unwrap();
    }
    a.add(si, ax).unwrap();
    a.rol(dx, 1).unwrap();
    a.add(cx, 7).unwrap();
    a.xor(cx, 0x1234).unwrap();
    a.inc(bx).unwrap();
    a.dec(di).unwrap();
    a.jne(top).unwrap();
    a.hlt().unwrap();
    a.set_label(&mut near).unwrap();
    a.add(ax, 1).unwrap();
    a.ret().unwrap();
    a.assemble(0).unwrap()
}

fn segmix32_code(data: u16, far: u16) -> Vec<u8> {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    let mut near = a.create_label();
    a.set_label(&mut top).unwrap();
    a.mov(ebx, u32::from(data)).unwrap();
    a.add(ebx, 0).unwrap();
    a.mov(ds, ebx).unwrap();
    a.mov(es, ebx).unwrap();
    a.mov(eax, dword_ptr(0x100)).unwrap();
    a.add(eax, 3).unwrap();
    a.mov(dword_ptr(0x100), eax).unwrap();
    a.push(eax).unwrap();
    a.xor(eax, 0x55aa).unwrap();
    a.pop(eax).unwrap();
    a.call(near).unwrap();
    // CALL FAR far:0, with a 32-bit offset.
    a.db(&[0x9a, 0, 0, 0, 0, far as u8, (far >> 8) as u8])
        .unwrap();
    a.mov(dword_ptr(0x104).es(), eax).unwrap();
    a.inc(ebp).unwrap();
    a.add(esi, eax).unwrap();
    a.rol(edx, 1).unwrap();
    a.add(ecx, 7).unwrap();
    a.xor(ecx, 0x1234).unwrap();
    a.inc(ebx).unwrap();
    a.dec(edi).unwrap();
    a.jne(top).unwrap();
    a.hlt().unwrap();
    a.set_label(&mut near).unwrap();
    a.add(eax, 1).unwrap();
    a.ret().unwrap();
    a.assemble(0).unwrap()
}

// Isolate the public decoder work repeated by Checked fetch. This excludes
// segment checks, bus reads, source proofs, dispatch and the IR interpreter.
fn components(n: u64, rounds: usize) {
    use iced_x86::{Decoder, DecoderOptions as D};
    let options = D::NO_PAUSE | D::NO_WBNOINVD | D::NO_MPFX_0FBC | D::NO_MPFX_0FBD | D::MOV_TR;
    println!(
        "workload,path,round,iterations,instructions,retired_counter,seconds,mips,host_per_step"
    );
    for round in 0..rounds {
        for work in ["register", "register-pad"] {
            let bytes = code(work, false);
            let mut decoder = Decoder::with_ip(32, &bytes, CODE, options);
            let lengths: Vec<_> = (0..5).map(|_| decoder.decode().len()).collect();
            let host = instructions();
            let start = Instant::now();
            for _ in 0..n {
                let mut offset = 0;
                for &len in &lengths {
                    for prefix in 1..=len {
                        let slice = std::hint::black_box(&bytes[offset..offset + prefix]);
                        let mut decoder =
                            Decoder::with_ip(32, slice, CODE + offset as u64, options);
                        std::hint::black_box(decoder.decode());
                    }
                    offset += len;
                }
            }
            let seconds = start.elapsed().as_secs_f64();
            let host = instructions() - host;
            let count = 5 * n;
            println!(
                "{work},component-decode,{round},{n},{count},0,{seconds:.9},{:.3},{:.3}",
                count as f64 / seconds / 1e6,
                host as f64 / count as f64
            );
        }
    }
}

// Paged V86 at IOPL 0 with a ring-0 #GP monitor. The I/O bitmap base lies
// beyond the TSS limit, so IN is #GP(0) (SDM Vol 3B §23.2.8.1 p. 23-11). The
// fault enters a 32-bit interrupt gate on the TSS ring-0 stack with the
// nine-dword V86 frame and an error code (Vol 3B §23.3.1.1 pp. 23-12–13);
// the monitor loads DS, reads saved CS:EIP, checks the opcode, advances EIP
// and IRETDs to V86 (Vol 2A IRET pp. 3-492–493). V86 HLT also traps; the
// monitor then halts at CPL 0.
const GP_BODIES: usize = 10;
const GP_V86_CS: u16 = 0x1000;
const GP_V86_DS: u16 = 0x2000;
const GP_V86_SS: u16 = 0x4000;
const GP_MONITOR: u64 = 0x9000;
const GP_COUNTER: u64 = 0x6000;

fn v86gp_code(free: bool) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(16).unwrap();
    let mut top = a.create_label();
    let mut near = a.create_label();
    a.set_label(&mut top).unwrap();
    for _ in 0..GP_BODIES {
        // 25 retired per body, including both callees.
        a.mov(bx, GP_V86_DS as u32).unwrap();
        a.mov(ds, bx).unwrap();
        a.mov(es, bx).unwrap();
        a.mov(ax, word_ptr(0x100)).unwrap();
        a.add(ax, 3).unwrap();
        a.mov(word_ptr(0x100), ax).unwrap();
        a.push(ax).unwrap();
        a.xor(ax, 0x55aa).unwrap();
        a.pop(ax).unwrap();
        a.call(near).unwrap();
        a.db(&[0x9a, 0, 0, 0x00, 0x11]).unwrap(); // CALL FAR 1100h:0.
        a.mov(word_ptr(0x102).es(), ax).unwrap();
        a.inc(bp).unwrap();
        a.add(si, ax).unwrap();
        a.rol(dx, 1).unwrap();
        a.add(bx, 7).unwrap();
        a.xor(bx, 0x1234).unwrap();
        a.add(si, bx).unwrap();
        a.inc(di).unwrap();
        a.xor(di, si).unwrap();
        a.ror(bx, 1).unwrap();
    }
    if free {
        a.mov(al, 0x5a).unwrap();
        for _ in 0..13 {
            a.nop().unwrap();
        }
    } else {
        a.in_(al, 0x80).unwrap();
    }
    a.dec(cx).unwrap();
    a.jne(top).unwrap();
    a.hlt().unwrap();
    a.set_label(&mut near).unwrap();
    a.add(ax, 1).unwrap();
    a.ret().unwrap();
    let v86 = a.assemble(0).unwrap();
    // Far callee at 1100h:0: XOR DX,AX; RETF.
    let far = vec![0x31, 0xc2, 0xcb];
    let mut m = CodeAssembler::new(32).unwrap();
    let mut halt = m.create_label();
    m.push(eax).unwrap();
    m.mov(ax, 0x10).unwrap();
    m.mov(ds, ax).unwrap();
    m.mov(eax, dword_ptr(esp + 12)).unwrap(); // Saved CS.
    m.shl(eax, 4).unwrap();
    m.add(eax, dword_ptr(esp + 8)).unwrap(); // Saved EIP.
    m.cmp(byte_ptr(eax), 0xe4).unwrap(); // IN AL,imm8.
    m.jne(halt).unwrap();
    m.inc(dword_ptr(GP_COUNTER)).unwrap();
    m.add(dword_ptr(esp + 8), 2).unwrap();
    m.pop(eax).unwrap();
    m.mov(al, 0x5a).unwrap();
    m.add(esp, 4).unwrap(); // Error code.
    m.iretd().unwrap();
    m.set_label(&mut halt).unwrap();
    m.hlt().unwrap();
    (v86, far, m.assemble(GP_MONITOR).unwrap())
}

fn v86gp_setup(free: bool, paged: bool) -> (Vm, x86jit_core::Vcpu) {
    use x86jit_core::legacy::{SegmentCache, SystemSegment};
    let memory_size = 0x50000;
    let mut vm = Vm::new(VmConfig::flat(memory_size));
    vm.set_cpu_mode(CpuMode::Compat32);
    vm.set_cpu_model(match std::env::var("MACHINE32_MODEL").as_deref() {
        Ok("486") => CpuModel::I80486Dx,
        _ => CpuModel::I80386,
    });
    vm.map(0, memory_size as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    let (v86, far, monitor) = v86gp_code(free);
    vm.write_bytes(u64::from(GP_V86_CS) << 4, &v86).unwrap();
    vm.write_bytes(0x11000, &far).unwrap();
    vm.write_bytes(GP_MONITOR, &monitor).unwrap();
    // GDT at 3800h: flat ring-0 code and data, accessed (SDM Vol 3A §3.4.5).
    vm.write_bytes(0x3808, &0x00cf_9b00_0000_ffffu64.to_le_bytes())
        .unwrap();
    vm.write_bytes(0x3810, &0x00cf_9300_0000_ffffu64.to_le_bytes())
        .unwrap();
    // IDT vector 13 at 3000h: 32-bit interrupt gate, DPL 0, to 8:monitor.
    let gate = (GP_MONITOR & 0xffff) | (8 << 16) | (0x8e00 << 32) | ((GP_MONITOR >> 16) << 48);
    vm.write_bytes(0x3000 + 13 * 8, &gate.to_le_bytes())
        .unwrap();
    // TSS at 5000h: ESP0=7000h, SS0=10h, I/O map base 68h past limit 67h.
    vm.write_bytes(0x5004, &0x7000u32.to_le_bytes()).unwrap();
    vm.write_bytes(0x5008, &0x10u32.to_le_bytes()).unwrap();
    vm.write_bytes(0x5066, &0x68u16.to_le_bytes()).unwrap();
    if paged {
        // Only the V86 code, callee, data and stack pages are user pages. The
        // V86 stack page 47h and the ring-0 stack page 6 share a TLB set, as
        // in the embedder's fixture.
        vm.write_bytes(0x1000, &0x2007u32.to_le_bytes()).unwrap();
        for page in 0..(memory_size / 4096) as u32 {
            let user = matches!(page, 0x10 | 0x11 | 0x20 | 0x47);
            let entry = (page * 4096) | if user { 7 } else { 3 };
            vm.write_bytes(0x2000 + u64::from(4 * page), &entry.to_le_bytes())
                .unwrap();
        }
    }
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([GP_V86_CS, GP_V86_DS, GP_V86_DS, GP_V86_SS, 0, 0]);
    cpu.cpu.legacy.cr0 = if paged { 0x80000001 } else { 1 };
    cpu.cpu.legacy.cr3 = 0x1000;
    cpu.cpu.legacy.cpl = 3;
    cpu.cpu.rflags_system = 0x20000; // VM=1, IOPL=0 (SDM Vol 3B §23.2.1).
    cpu.cpu.legacy.gdtr.base = 0x3800;
    cpu.cpu.legacy.gdtr.limit = 0x17;
    cpu.cpu.legacy.idtr.base = 0x3000;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    cpu.cpu.legacy.tr = SystemSegment {
        selector: 0x18,
        cache: SegmentCache {
            base: 0x5000,
            limit: 0x67,
            attributes: 0x8b,
            ..Default::default()
        },
        ..Default::default()
    };
    cpu.set_a20_enabled(true);
    (vm, cpu)
}

fn v86gp_execute(vm: &Vm, cpu: &mut x86jit_core::Vcpu, entry: &CpuState, n: u64) {
    cpu.cpu = entry.clone();
    cpu.set_reg(Reg::Rip, 0);
    cpu.set_reg(Reg::Rsp, 0x8000);
    cpu.set_reg(Reg::Rcx, n);
    let exit = cpu.run(vm, None);
    assert!(matches!(exit, Exit::Hlt), "{exit:?}");
    assert_eq!(cpu.cpu.legacy.cpl, 0);
    assert_eq!(cpu.reg(Reg::Rcx) & 0xffff, 0);
}

fn v86gp(n: u64, rounds: usize, only: Option<&str>) {
    assert!(n <= 0xffff, "CX counts the iterations");
    let path = "machine-paged-v86-gp";
    println!(
        "workload,path,round,iterations,instructions,retired_counter,seconds,mips,host_per_step"
    );
    for round in 0..rounds {
        for work in ["v86gp", "v86gp-free"] {
            if only.is_some_and(|o| o != work) {
                continue;
            }
            let free = work == "v86gp-free";
            let (vm, mut cpu) = v86gp_setup(free, true);
            let entry = cpu.cpu.clone();
            v86gp_execute(&vm, &mut cpu, &entry, 100);
            vm.write_bytes(GP_COUNTER, &[0; 4]).unwrap();
            vm.write_bytes(0x20100, &[0; 2]).unwrap();
            let retired_before = cpu.retired_instructions();
            let host_before = instructions();
            let start = Instant::now();
            v86gp_execute(&vm, &mut cpu, &entry, n);
            let seconds = start.elapsed().as_secs_f64();
            let host = instructions() - host_before;
            let retired = cpu.retired_instructions() - retired_before;
            // 252 V86 + 14 monitor per iteration; the final HLT traps and the
            // monitor retires 8 plus its own HLT.
            let count = 266 * n + 9;
            assert_eq!(retired, count);
            let mut word = [0; 4];
            vm.read_bytes(GP_COUNTER, &mut word).unwrap();
            assert_eq!(
                u64::from(u32::from_le_bytes(word)),
                if free { 0 } else { n }
            );
            vm.read_bytes(0x20100, &mut word[..2]).unwrap();
            assert_eq!(
                u64::from(u16::from_le_bytes([word[0], word[1]])),
                (30 * n) & 0xffff
            );
            println!(
                "{work},{path},{round},{n},{count},{retired},{seconds:.9},{:.3},{:.3}",
                count as f64 / seconds / 1e6,
                host as f64 / count as f64
            );
        }
    }
}
