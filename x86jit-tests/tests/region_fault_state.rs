//! SDM Vol 3A §7.5: a fault exposes the state preceding the instruction.
//! Capture real guard faults without resuming or installing a production handler.
#![cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]

use iced_x86::code_asm::*;
use std::sync::atomic::{AtomicPtr, Ordering};
use x86jit_core::{
    lift::lift_region, memory::HostRam, Backend, CachedBlock, CpuMode, CpuState, Exit,
    GuestAccessKind, MemConsistency, Memory, MemoryModel, Prot, RegionCaps, RegionKind, StepResult,
};
use x86jit_cranelift::{HostTarget, JitBackend, OptLevel};

const CODE: u64 = 0x1000;

#[repr(C)]
#[derive(Default)]
struct Capture {
    gpr: [u64; 16],
    rflags: u64,
    rip: u64,
    width: u8,
    access: u8,
    fault_addr: usize,
}

static CPU: AtomicPtr<CpuState> = AtomicPtr::new(std::ptr::null_mut());
static CAPTURE: AtomicPtr<Capture> = AtomicPtr::new(std::ptr::null_mut());
static INTERP_READ: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static INTERP_RIP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

extern "C" fn handler(_: libc::c_int, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
    // SAFETY: installed only in the forked child, with live CPU/capture pointers.
    // The handler reads raw memory and the signal-safe codemap, then calls _exit.
    unsafe {
        let context = &*context.cast::<libc::ucontext_t>();
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        let pc = context.uc_mcontext.gregs[libc::REG_RIP as usize] as usize;
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        let pc = context.uc_mcontext.pc as usize;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let pc = (*context.uc_mcontext).__ss.__pc as usize;
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        let pc = (*context.uc_mcontext).__ss.__rip as usize;
        let site = x86jit_core::codemap::lookup_trap(pc);
        let interp_rip = INTERP_RIP.load(Ordering::Relaxed);
        if site.is_none() && interp_rip == 0 {
            libc::_exit(90);
        }
        let cpu = CPU.load(Ordering::Relaxed);
        let capture = CAPTURE.load(Ordering::Relaxed);
        (*capture).gpr = (*cpu).gpr;
        (*capture).fault_addr = (*info).si_addr() as usize;
        (*capture).rflags = (*cpu).flags.to_rflags();
        (*capture).rip = site.map_or(interp_rip, |s| s.guest_rip);
        (*capture).width = site.map_or(8, |s| s.width);
        (*capture).access = match site.map_or(
            if INTERP_READ.load(Ordering::Relaxed) {
                GuestAccessKind::Read
            } else {
                GuestAccessKind::Write
            },
            |s| s.access,
        ) {
            GuestAccessKind::Read => 0,
            GuestAccessKind::Write => 1,
            GuestAccessKind::ReadWrite => 2,
        };
        libc::_exit(0);
    }
}

fn write_fault(kind: &str, tier: usize) -> Vec<String> {
    // SDM Vol 3A §5.6 permits reads from a write-protected page; §7.5
    // requires a fault to preserve instruction-entry registers and flags.
    // SAFETY: sysconf has no pointer arguments.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let data = (4 * page) as u64;
    let span = 8 * page;
    let mut failures = Vec::new();
    for (carry, value, live, count) in [
        (false, 0u64),
        (true, 0),
        (false, u64::MAX),
        (true, u64::MAX),
    ]
    .into_iter()
    .flat_map(|(carry, value)| {
        [false, true].into_iter().flat_map(move |live| {
            [0, 1, 64]
                .into_iter()
                .map(move |count| (carry, value, live, count))
        })
    }) {
        if !matches!(
            kind,
            "shl" | "shld" | "shrd" | "rcl" | "rcr" | "rol" | "ror"
        ) && count != 1
        {
            continue;
        }
        for after in [false, true] {
            if live && after {
                continue;
            }
            let mut a = CodeAssembler::new(64).unwrap();
            a.xor(eax, eax).unwrap();
            if carry {
                a.sub(rax, 1).unwrap();
            }
            a.mov(r12, 0x1234u64).unwrap();
            match kind {
                "add" => a.add(qword_ptr(rdi), 1).unwrap(),
                "sub" => a.sub(qword_ptr(rdi), 1).unwrap(),
                "and" => a.and(qword_ptr(rdi), 1).unwrap(),
                "or" => a.or(qword_ptr(rdi), 1).unwrap(),
                "xor" => a.xor(qword_ptr(rdi), 1).unwrap(),
                "inc" => a.inc(qword_ptr(rdi)).unwrap(),
                "dec" => a.dec(qword_ptr(rdi)).unwrap(),
                "neg" => a.neg(qword_ptr(rdi)).unwrap(),
                "not" => a.not(qword_ptr(rdi)).unwrap(),
                "adc" => a.adc(qword_ptr(rdi), 1).unwrap(),
                "sbb" => a.sbb(qword_ptr(rdi), 1).unwrap(),
                "shl" => a.shl(qword_ptr(rdi), cl).unwrap(),
                "shld" => a.shld(qword_ptr(rdi), r8, cl).unwrap(),
                "shrd" => a.shrd(qword_ptr(rdi), r8, cl).unwrap(),
                "rcl" => a.rcl(qword_ptr(rdi), cl).unwrap(),
                "rcr" => a.rcr(qword_ptr(rdi), cl).unwrap(),
                "rol" => a.rol(qword_ptr(rdi), cl).unwrap(),
                "ror" => a.ror(qword_ptr(rdi), cl).unwrap(),
                "bts_reg" => a.bts(qword_ptr(rdi), r8).unwrap(),
                "btr_reg" => a.btr(qword_ptr(rdi), r8).unwrap(),
                "btc_reg" => a.btc(qword_ptr(rdi), r8).unwrap(),
                "bts_imm" => a.bts(qword_ptr(rdi), 1).unwrap(),
                "btr_imm" => a.btr(qword_ptr(rdi), 1).unwrap(),
                "btc_imm" => a.btc(qword_ptr(rdi), 1).unwrap(),
                "lock_add" => a.lock().add(qword_ptr(rdi), 1).unwrap(),
                _ => unreachable!(),
            }
            if !live {
                // Observe a deferred RMW recipe before XOR kills its flags.
                a.mov(r11, qword_ptr(rdi + page as i32)).unwrap();
                a.xor(ebx, ebx).unwrap();
            }
            a.hlt().unwrap();
            let code = a.assemble(CODE).unwrap();
            let fault_rip = iced_x86::Decoder::with_ip(64, &code, CODE, 0)
                .into_iter()
                .filter(|i| {
                    i.op0_kind() == iced_x86::OpKind::Memory
                        || i.op1_kind() == iced_x86::OpKind::Memory
                })
                .nth(usize::from(after))
                .unwrap()
                .ip();
            let mem = memory(&code, span, (6 * page) as u64, page, true);
            mem.write_bytes(data, &value.to_le_bytes()).unwrap();
            let mut initial = CpuState::new();
            initial.gpr = std::array::from_fn(|i| 0x8000_0000_0000_0080 + i as u64);
            initial.gpr[7] = data;
            initial.gpr[1] = count;
            initial.gpr[8] = 1;
            initial.rip = CODE;
            let mut expected = initial.clone();
            let mut scratch = Vec::new();
            while expected.rip != fault_rip {
                assert!(matches!(
                    x86jit_core::interp::step_one(
                        &mem,
                        &mut expected,
                        CpuMode::Long64,
                        &mut scratch,
                        &mut Default::default()
                    ),
                    StepResult::Continue
                ));
            }
            mem.write_bytes(data, &value.to_le_bytes()).unwrap();
            let fault_addr = data + if after { page as u64 } else { 0 };
            for opt in [OptLevel::None, OptLevel::Speed] {
                let jit = JitBackend::with_options(None, HostTarget::Native, opt);
                let region = lift_region(
                    &mem,
                    CODE,
                    RegionCaps {
                        max_blocks: 16,
                        max_icount: 256,
                    },
                    CpuMode::Long64,
                )
                .unwrap();
                let compiled = if tier == 2 {
                    jit.materialize_region(&region, MemConsistency::Fast, None, 0)
                } else {
                    jit.materialize(&region.blocks[0], MemConsistency::Fast, None, 0)
                };
                let CachedBlock::Compiled { entry } = compiled else {
                    panic!("must compile")
                };
                // SAFETY: shared capture remains live until waitpid; only the child
                // writes it. Protection and handlers are confined to the child.
                unsafe {
                    let capture = libc::mmap(
                        std::ptr::null_mut(),
                        page,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED | libc::MAP_ANON,
                        -1,
                        0,
                    )
                    .cast::<Capture>();
                    assert_ne!(capture.cast(), libc::MAP_FAILED);
                    let mut cpu = initial.clone();
                    let mut ctx = x86jit_core::jit_abi::MemCtx::for_memory(&mem);
                    ctx.fuel = 128;
                    let pid = libc::fork();
                    assert!(pid >= 0);
                    if pid == 0 {
                        CPU.store(&mut cpu, Ordering::Relaxed);
                        CAPTURE.store(capture, Ordering::Relaxed);
                        INTERP_RIP.store(if tier == 0 { fault_rip } else { 0 }, Ordering::Relaxed);
                        INTERP_READ.store(after, Ordering::Relaxed);
                        let mut action: libc::sigaction = std::mem::zeroed();
                        action.sa_sigaction = handler as *const () as usize;
                        action.sa_flags = libc::SA_SIGINFO;
                        libc::sigemptyset(&mut action.sa_mask);
                        if libc::sigaction(libc::SIGSEGV, &action, std::ptr::null_mut()) != 0
                            || libc::sigaction(libc::SIGBUS, &action, std::ptr::null_mut()) != 0
                            || libc::mprotect(
                                (mem.host_base() as *mut u8).add(fault_addr as usize).cast(),
                                page,
                                if after {
                                    libc::PROT_NONE
                                } else {
                                    libc::PROT_READ
                                },
                            ) != 0
                        {
                            libc::_exit(91);
                        }
                        if tier == 0 {
                            for _ in 0..8 {
                                x86jit_core::interp::step_one(
                                    &mem,
                                    &mut cpu,
                                    CpuMode::Long64,
                                    &mut scratch,
                                    &mut Default::default(),
                                );
                            }
                        } else {
                            x86jit_core::jit_abi::call_block(entry, &mut cpu, &mut ctx);
                        }
                        libc::_exit(92);
                    }
                    let mut status = 0;
                    assert_eq!(libc::waitpid(pid, &mut status, 0), pid);
                    let actual = capture.read();
                    libc::munmap(capture.cast(), page);
                    assert_eq!(status, 0, "{kind}, tier={tier}, {opt:?}");
                    assert_eq!(actual.rip, fault_rip);
                    assert_eq!(
                        actual.width,
                        if !after && kind.ends_with("_reg") && tier != 0 {
                            1
                        } else {
                            8
                        }
                    );
                    assert_eq!(
                        actual.fault_addr,
                        mem.host_base() as usize + fault_addr as usize
                    );
                    assert_eq!(actual.gpr, expected.gpr, "{kind}, tier={tier}, {opt:?}");
                    assert_eq!(
                        actual.access,
                        if after {
                            0
                        } else if kind == "lock_add" && tier != 0 {
                            2
                        } else {
                            1
                        }
                    );
                    if actual.rflags != expected.flags.to_rflags() {
                        failures.push(format!("tier={tier}, carry={carry}, live={live}, count={count}, after={after}, {opt:?}: actual={:#x}, expected={:#x}", actual.rflags, expected.flags.to_rflags()));
                    }
                }
                mem.write_bytes(data, &value.to_le_bytes()).unwrap();
                let mut retired = initial.clone();
                let ir = &region.blocks[0];
                assert!(matches!(
                    x86jit_core::interp::interpret_block(
                        ir,
                        &mut retired,
                        &mem,
                        &mut scratch,
                        &mut Default::default()
                    ),
                    StepResult::Exit(Exit::Hlt)
                ));
                let mut expected_bytes = [0; 8];
                mem.read_bytes(data, &mut expected_bytes).unwrap();
                mem.write_bytes(data, &value.to_le_bytes()).unwrap();
                let mut actual = initial.clone();
                let mut ctx = x86jit_core::jit_abi::MemCtx::for_memory(&mem);
                ctx.fuel = 128;
                // SAFETY: entry was compiled for this live CPU and memory ABI.
                unsafe {
                    x86jit_core::jit_abi::call_block(entry, &mut actual, &mut ctx);
                }
                let mut actual_bytes = [0; 8];
                mem.read_bytes(data, &mut actual_bytes).unwrap();
                assert_eq!(
                    actual_bytes, expected_bytes,
                    "retired {kind}, tier={tier}, {opt:?}"
                );
                assert_eq!(
                    actual.gpr, retired.gpr,
                    "retired {kind}, tier={tier}, {opt:?}"
                );
                assert_eq!(
                    actual.flags.to_rflags(),
                    retired.flags.to_rflags(),
                    "retired {kind}, tier={tier}, live={live}, count={count}, {opt:?}"
                );
                mem.write_bytes(data, &value.to_le_bytes()).unwrap();
            }
        }
    }
    failures
}

macro_rules! write_fault_tests {
    ($($name:ident: $kind:literal),* $(,)?) => {$(
        #[test]
        fn $name() {
            let failures: Vec<_> = (0..3).flat_map(|tier| write_fault($kind, tier)).collect();
            assert!(failures.is_empty(), "{}: {failures:?}", $kind);
        }
    )*};
}

write_fault_tests! {
    write_fault_add: "add", write_fault_sub: "sub", write_fault_and: "and",
    write_fault_or: "or", write_fault_xor: "xor", write_fault_inc: "inc",
    write_fault_dec: "dec", write_fault_neg: "neg", write_fault_not: "not",
    write_fault_adc: "adc", write_fault_sbb: "sbb", write_fault_shl: "shl",
    write_fault_lock_add: "lock_add",
    write_fault_bts_reg: "bts_reg", write_fault_bts_imm: "bts_imm",
    write_fault_btr_reg: "btr_reg", write_fault_btr_imm: "btr_imm",
    write_fault_btc_reg: "btc_reg", write_fault_btc_imm: "btc_imm",
    write_fault_shld: "shld", write_fault_shrd: "shrd",
    write_fault_rcl: "rcl", write_fault_rcr: "rcr",
    write_fault_rol: "rol", write_fault_ror: "ror",
}

fn program(page: u64, vector_offset: i32) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    let mut left = a.create_label();
    let mut join = a.create_label();
    a.set_label(&mut top).unwrap();
    // A diamond and a back-edge exercise dirty-state joins, including a register
    // written only after the previous iteration's final memory access.
    a.test(cl, 1).unwrap();
    a.jz(left).unwrap();
    a.mov(r9, r15).unwrap();
    a.jmp(join).unwrap();
    a.set_label(&mut left).unwrap();
    a.mov(r10, r14).unwrap();
    a.jmp(join).unwrap();
    a.set_label(&mut join).unwrap();
    a.add(rax, rbx).unwrap();
    a.inc(r12).unwrap();
    a.mov(rbp, rsp).unwrap();
    a.mov(rdx, qword_ptr(rsi)).unwrap();
    a.add(r8, 17).unwrap();
    a.mov(qword_ptr(rsi + page as i32), r8).unwrap();
    a.add(r8, 19).unwrap();
    a.lock()
        .add(qword_ptr(rsi + (2 * page) as i32), r8)
        .unwrap();
    a.add(r8, 23).unwrap();
    a.lock().neg(qword_ptr(rsi + (3 * page) as i32)).unwrap();
    a.add(r8, 29).unwrap();
    a.lock()
        .cmpxchg(qword_ptr(rsi + (4 * page) as i32), r8)
        .unwrap();
    a.add(qword_ptr(rsi + (5 * page) as i32), r8).unwrap();
    a.add(r8, 31).unwrap();
    a.vmovdqu(ymm0, ymmword_ptr(rsi + (6 * page) as i32 + vector_offset))
        .unwrap();
    a.add(r8, 37).unwrap();
    a.vmovdqu(ymmword_ptr(rsi + (9 * page) as i32 + vector_offset), ymm0)
        .unwrap();
    a.xor(edi, edi).unwrap();
    a.add(r13, 41).unwrap();
    a.add(rsi, (12 * page) as i32).unwrap();
    a.dec(rcx).unwrap();
    a.jnz(top).unwrap();
    a.hlt().unwrap();
    a.assemble(CODE).unwrap()
}

fn memory(code: &[u8], span: usize, guard: u64, page: usize, host: bool) -> Memory {
    let model = MemoryModel::Flat { size: span as u64 };
    let mut mem = if host {
        // SAFETY: anonymous mapping is owned by HostRam until its destructor.
        unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                span,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            assert_ne!(ptr, libc::MAP_FAILED);
            Memory::from_host_ram(
                model,
                HostRam {
                    ptr: ptr.cast(),
                    len: span,
                    guest_base: 0,
                    protect: None,
                    dtor: Box::new(|ptr, len| {
                        libc::munmap(ptr.cast(), len);
                    }),
                },
            )
            .unwrap()
        }
    } else {
        Memory::new(model)
    };
    mem.map(0, guard as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    mem.map(
        guard + page as u64,
        span - guard as usize - page,
        Prot::RWX,
        RegionKind::Ram,
    )
    .unwrap();
    mem.write_bytes(CODE, code).unwrap();
    mem
}

#[test]
fn region_guard_fault_recovers_gprs_and_flags() {
    // SAFETY: sysconf has no pointer arguments.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(page > 0);
    let page = page as usize;
    let data = (4 * page) as u64;
    let span = 44 * page;
    let mut initial = CpuState::new();
    initial.gpr = std::array::from_fn(|i| 0x8000_0000_0000_0080 + i as u64);
    initial.gpr[1] = 3;
    initial.gpr[6] = data;
    initial.rip = CODE;
    // Guard both entry and the second iteration after traversing the back-edge.
    // Slide each wide access across the boundary to fault each host transfer.
    let transfer = if cfg!(target_arch = "x86_64") { 8 } else { 16 };
    for (iteration, site) in (0..8).map(|site| (1, site)).chain([(0, 0)]) {
        let offsets: Vec<i32> = if site >= 6 {
            (0..32).step_by(transfer).map(|v| -v).collect()
        } else {
            vec![0]
        };
        for offset in offsets {
            let code = program(page as u64, offset);
            let slot = [0, 1, 2, 3, 4, 5, 6, 9][site];
            let guard = data + ((12 * iteration + slot) * page) as u64;
            let interp = memory(&code, span, guard, page, false);
            let mut expected = initial.clone();
            let mut scratch = Vec::new();
            let mut faulted = false;
            for _ in 0..200 {
                let step = x86jit_core::interp::step_one(
                    &interp,
                    &mut expected,
                    CpuMode::Long64,
                    &mut scratch,
                    &mut Default::default(),
                );
                if let StepResult::Exit(exit) = step {
                    assert!(matches!(exit, Exit::UnmappedMemory { .. }), "{exit:?}");
                    faulted = true;
                    break;
                }
            }
            assert!(faulted);
            let mem = memory(&code, span, guard, page, true);
            let region = lift_region(
                &mem,
                CODE,
                RegionCaps {
                    max_blocks: 16,
                    max_icount: 256,
                },
                CpuMode::Long64,
            )
            .unwrap();
            assert!(region.has_loop);
            let mut targets = std::collections::HashMap::new();
            for block in &region.blocks {
                for op in &block.ops {
                    if let x86jit_core::ir::IrOp::Jump {
                        target: x86jit_core::ir::Val::Imm(target),
                    } = op
                    {
                        *targets.entry(*target).or_insert(0) += 1;
                    }
                }
            }
            assert!(
                targets.values().any(|&incoming| incoming >= 2),
                "diamond must join"
            );
            for opt in [OptLevel::None, OptLevel::Speed] {
                let jit = JitBackend::with_options(None, HostTarget::Native, opt);
                let CachedBlock::Compiled { entry } =
                    jit.materialize_region(&region, MemConsistency::Fast, None, 0)
                else {
                    panic!("region must compile")
                };
                // SAFETY: shared capture is valid through waitpid; only the child
                // writes it. All compilation/allocation happens before fork.
                unsafe {
                    let capture = libc::mmap(
                        std::ptr::null_mut(),
                        page,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED | libc::MAP_ANON,
                        -1,
                        0,
                    )
                    .cast::<Capture>();
                    assert_ne!(capture.cast(), libc::MAP_FAILED);
                    let mut cpu = initial.clone();
                    let mut ctx = x86jit_core::jit_abi::MemCtx::for_memory(&mem);
                    ctx.fuel = 128;
                    let pid = libc::fork();
                    assert!(pid >= 0);
                    if pid == 0 {
                        CPU.store(&mut cpu, Ordering::Relaxed);
                        CAPTURE.store(capture, Ordering::Relaxed);
                        let mut action: libc::sigaction = std::mem::zeroed();
                        action.sa_sigaction = handler as *const () as usize;
                        action.sa_flags = libc::SA_SIGINFO;
                        libc::sigemptyset(&mut action.sa_mask);
                        if libc::sigaction(libc::SIGSEGV, &action, std::ptr::null_mut()) != 0
                            || libc::sigaction(libc::SIGBUS, &action, std::ptr::null_mut()) != 0
                            || libc::mprotect(
                                (mem.host_base() as *mut u8).add(guard as usize).cast(),
                                page,
                                libc::PROT_NONE,
                            ) != 0
                        {
                            libc::_exit(91);
                        }
                        x86jit_core::jit_abi::call_block(entry, &mut cpu, &mut ctx);
                        libc::_exit(92);
                    }
                    let mut status = 0;
                    assert_eq!(libc::waitpid(pid, &mut status, 0), pid);
                    assert_eq!(status, 0, "site={site}, offset={offset}, opt={opt:?}");
                    let actual = capture.read();
                    libc::munmap(capture.cast(), page);
                    assert_eq!(
                        actual.rip, expected.rip,
                        "site={site}, offset={offset}, opt={opt:?}"
                    );
                    assert_eq!(
                        actual.gpr, expected.gpr,
                        "site={site}, offset={offset}, opt={opt:?}"
                    );
                    assert_eq!(
                        actual.rflags,
                        expected.flags.to_rflags(),
                        "site={site}, offset={offset}, opt={opt:?}"
                    );
                    assert_eq!(actual.width as usize, if site >= 6 { transfer } else { 8 });
                    assert_eq!(actual.access, [0, 1, 2, 0, 2, 0, 0, 1][site]);
                }
            }
        }
    }
}
