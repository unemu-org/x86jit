//! Instruction-entry fault state: SDM Vol 3A §7.5. The interpreter supplies
//! transfer splitting and access-base semantics for the public exit contract.
#![cfg(all(
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]

use iced_x86::code_asm::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use x86jit_core::jit_abi::fault::{classify, FrameRecipe, HostFaultKind, HostRegisters};
use x86jit_core::{
    memory::HostRam, Backend, CachedBlock, CpuState, Exit, InterpreterBackend, IrBlock, IrRegion,
    MemConsistency, Prot, Reg, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

struct ClassifierAllocator;
static CLASSIFIER_ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forward all allocation operations unchanged to the system allocator.
unsafe impl std::alloc::GlobalAlloc for ClassifierAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        if IN_CLASSIFIER.load(Ordering::Relaxed) {
            CLASSIFIER_ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        std::alloc::System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        std::alloc::System.dealloc(ptr, layout);
    }
}
#[global_allocator]
static ALLOCATOR: ClassifierAllocator = ClassifierAllocator;

#[path = "../../x86jit-cranelift/src/unwind_test_signal.rs"]
mod signal;
use signal::*;

static COMPILER_FP: AtomicU64 = AtomicU64::new(u64::MAX);

fn perturb_compiler_fp() {
    let value = COMPILER_FP.load(Ordering::Relaxed);
    if value == u64::MAX {
        return;
    }
    // SAFETY: test-only block materialization changes the calling thread's mode
    // to distinguish the run-entry value from the compiled-call entry value.
    unsafe {
        #[cfg(target_arch = "aarch64")]
        core::arch::asm!("msr fpcr, {}", in(reg) value, options(nomem, nostack));
        #[cfg(target_arch = "x86_64")]
        core::arch::asm!("ldmxcsr [{}]", in(reg) &value, options(nostack));
    }
}

struct Compiled {
    jit: JitBackend,
    region: bool,
}

impl Backend for Compiled {
    fn region_caps(&self) -> Option<x86jit_core::RegionCaps> {
        self.region.then_some(x86jit_core::RegionCaps {
            max_blocks: 16,
            max_icount: 256,
        })
    }

    fn materialize(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> CachedBlock {
        let block = self.jit.materialize(ir, consistency, mmio, guest_base);
        assert!(
            matches!(block, CachedBlock::Compiled { .. }),
            "must execute compiled code"
        );
        perturb_compiler_fp();
        block
    }

    fn materialize_region(
        &self,
        ir: &IrRegion,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        guest_base: u64,
    ) -> CachedBlock {
        assert!(ir.blocks.len() > 1 && ir.has_loop);
        let block = self
            .jit
            .materialize_region(ir, consistency, mmio, guest_base);
        assert!(
            matches!(block, CachedBlock::Compiled { .. }),
            "must execute compiled region"
        );
        perturb_compiler_fp();
        block
    }
}

fn vm(backend: Box<dyn Backend>, base: u64, page: usize) -> Vm {
    let len = page * 8;
    // SAFETY: the mapping is exclusively owned by HostRam until its destructor.
    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        assert_ne!(ptr, libc::MAP_FAILED);
        let mut vm = Vm::with_backend_host_ram(
            VmConfig::flat(base + len as u64),
            backend,
            HostRam {
                ptr: ptr.cast(),
                len,
                guest_base: base,
                protect: Some(Box::new(move |ptr, len, accessible| {
                    let host_prot = if accessible {
                        libc::PROT_READ | libc::PROT_WRITE
                    } else {
                        libc::PROT_NONE
                    };
                    let lo = ptr as usize & !(page - 1);
                    let hi = (ptr as usize + len + page - 1) & !(page - 1);
                    assert_eq!(
                        libc::mprotect(lo as *mut libc::c_void, hi - lo, host_prot),
                        0
                    );
                })),
                dtor: Box::new(|ptr, len| {
                    libc::munmap(ptr.cast(), len);
                }),
            },
        )
        .unwrap();
        vm.map(base, page * 4, Prot::RWX, RegionKind::Ram).unwrap();
        vm
    }
}

fn same_cpu(a: &CpuState, b: &CpuState) {
    assert_eq!(a.gpr, b.gpr);
    assert_eq!(a.rip, b.rip);
    assert_eq!(a.flags.to_rflags(), b.flags.to_rflags());
    assert_eq!(a.xmm, b.xmm);
    assert_eq!(a.ymm_hi, b.ymm_hi);
}

fn case(kind: &str, region: bool, straddle: bool, mapped: bool) {
    case_mode(kind, region, straddle, mapped, false);
}
fn case_mode(kind: &str, region: bool, straddle: bool, mapped: bool, routed: bool) {
    // SAFETY: sysconf has no pointer argument.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let base = 0x1_0000_0000;
    let code_addr = base + 0x1000;
    let guard = base + (page * 4) as u64;
    let addr = guard - if straddle { 4 } else { 0 };
    let mut asm = CodeAssembler::new(64).unwrap();
    asm.mov(ecx, 2).unwrap();
    let mut top = asm.create_label();
    asm.set_label(&mut top).unwrap();
    asm.add(r12, 7).unwrap();
    asm.sub(r13, 3).unwrap();
    asm.dec(ecx).unwrap();
    asm.jnz(top).unwrap();
    match kind {
        "load" => asm.mov(rax, qword_ptr(rdi)).unwrap(),
        "store" => asm.mov(qword_ptr(rdi), rax).unwrap(),
        "locked" => asm.lock().add(qword_ptr(rdi), 1).unwrap(),
        "vector" => asm.movdqu(xmm0, xmmword_ptr(rdi)).unwrap(),
        _ => unreachable!(),
    }
    asm.add(r14, 9).unwrap();
    asm.hlt().unwrap();
    let code = asm.assemble(code_addr).unwrap();
    let fault_rip = iced_x86::Decoder::with_ip(64, &code, code_addr, 0)
        .into_iter()
        .find(|i| {
            i.op0_kind() == iced_x86::OpKind::Memory || i.op1_kind() == iced_x86::OpKind::Memory
        })
        .unwrap()
        .ip();
    let mut interp = vm(Box::new(InterpreterBackend), base, page);
    let mut jit = vm(
        Box::new(Compiled {
            jit: JitBackend::new(),
            region,
        }),
        base,
        page,
    );
    jit.mem.set_host_fault_recovery(routed);
    for vm in [&interp, &jit] {
        vm.write_bytes(code_addr, &code).unwrap();
        if straddle {
            vm.write_bytes(addr, &[0x53; 4]).unwrap();
        }
    }
    let mut expected = interp.new_vcpu();
    expected.cpu.gpr = std::array::from_fn(|i| 0x8055_0000 + i as u64);
    expected.set_reg(Reg::Rdi, addr);
    expected.set_reg(Reg::Rip, code_addr);
    let mut actual = jit.new_vcpu();
    actual.cpu = expected.cpu.clone();
    let foreign = jit.new_vcpu();
    HANDLE.store(actual.fault_handle().as_usize(), Ordering::Relaxed);
    FOREIGN.store(foreign.fault_handle().as_usize(), Ordering::Relaxed);
    FAULTS.store(0, Ordering::Relaxed);
    let expected_exit = expected.run(&interp, None);
    assert!(matches!(expected_exit, Exit::UnmappedMemory { .. }));
    if mapped {
        jit.map(guard, page, Prot::RW, RegionKind::Ram).unwrap();
        // Simulate an embedder's protection independent of its region table.
        // SAFETY: page-aligned address in this child's owned mapping.
        unsafe {
            assert_eq!(
                libc::mprotect(
                    (jit.mem.host_base() as *mut u8).add(page * 4).cast(),
                    page,
                    libc::PROT_NONE
                ),
                0
            );
        }
    }
    // SAFETY: this function runs only in a forked child with live vCPUs.
    unsafe {
        install();
    }
    let _host_fp = HostFpMode::nondefault();
    let before_run_fp = fp_control();
    #[cfg(target_arch = "aarch64")]
    let changed_fp = before_run_fp ^ (1 << 22);
    #[cfg(target_arch = "x86_64")]
    let changed_fp = before_run_fp ^ (1 << 13);
    COMPILER_FP.store(changed_fp, Ordering::Relaxed);
    let actual_exit = actual.run(&jit, None);
    let after_run_fp = fp_control();
    COMPILER_FP.store(u64::MAX, Ordering::Relaxed);
    assert_eq!(after_run_fp, before_run_fp);
    assert_eq!(FAULTS.load(Ordering::Relaxed), 1);
    assert_eq!(ACCESS_FAULTS.load(Ordering::Relaxed), 1);
    assert_eq!(ALIGNMENTS.load(Ordering::Relaxed), 0);
    assert_eq!(CLASSIFIER_ALLOCS.load(Ordering::Relaxed), 0);
    assert_eq!(jit.cache.regions() > 0, region);
    assert!(WIDTH.load(Ordering::Relaxed) > 0);
    assert_eq!(actual.reg(Reg::Rip), fault_rip);
    if !mapped {
        assert!(
            matches!((&actual_exit, &expected_exit),
            (Exit::UnmappedMemory { addr: a, access: x, .. }, Exit::UnmappedMemory { addr: b, access: y, .. }) if a == b && x == y),
            "{kind}, region={region}, straddle={straddle}: {actual_exit:?} != {expected_exit:?}"
        );
        same_cpu(&actual.cpu, &expected.cpu);
        if straddle {
            // Memory requires one containing region for each scalar transfer.
            // Grow the original region across the newly accessible page.
            jit.unmap(base, page * 4).unwrap();
            jit.map(base, page * 5, Prot::RWX, RegionKind::Ram).unwrap();
        } else {
            jit.map(guard, page, Prot::RW, RegionKind::Ram).unwrap();
        }
    } else {
        assert!(
            matches!(actual_exit, Exit::UnmappedMemory { addr: a, access, .. }
            if a == guard && access == if kind == "load" { x86jit_core::AccessKind::Read } else { x86jit_core::AccessKind::Write })
        );
        // SAFETY: restore access to the mapped page before retrying.
        unsafe {
            assert_eq!(
                libc::mprotect(
                    (jit.mem.host_base() as *mut u8).add(page * 4).cast(),
                    page,
                    libc::PROT_READ | libc::PROT_WRITE
                ),
                0
            );
        }
    }
    if straddle {
        interp.unmap(base, page * 4).unwrap();
        interp
            .map(base, page * 5, Prot::RWX, RegionKind::Ram)
            .unwrap();
    } else {
        interp.map(guard, page, Prot::RW, RegionKind::Ram).unwrap();
    }
    assert!(matches!(expected.run(&interp, None), Exit::Hlt));
    assert!(matches!(actual.run(&jit, None), Exit::Hlt));
    same_cpu(&actual.cpu, &expected.cpu);
    let mut eb = [0; 32];
    let mut ab = [0; 32];
    for offset in [0usize, 16] {
        interp
            .mem
            .read_bytes(guard - 16 + offset as u64, &mut eb[offset..offset + 16])
            .unwrap();
        jit.mem
            .read_bytes(guard - 16 + offset as u64, &mut ab[offset..offset + 16])
            .unwrap();
    }
    assert_eq!(ab, eb);
    // Stale-but-live handles are harmless outside compiled execution.
    let mut regs = HostRegisters::default();
    // SAFETY: both vCPUs still live and are inactive on this thread.
    unsafe {
        assert!(!classify(
            HANDLE.load(Ordering::Relaxed),
            1,
            1,
            HostFaultKind::Access,
            &mut regs
        ));
    }
    HANDLE.store(0, Ordering::Relaxed);
    FOREIGN.store(0, Ordering::Relaxed);
}

macro_rules! recovery_test {
    ($name:ident, $kind:literal, $straddle:expr) => {
        #[test]
        fn $name() {
            for region in [false, true] {
                let status = child(|| case($kind, region, $straddle, false));
                assert_eq!(
                    status, 0,
                    "{} region={region}: child status {status}",
                    $kind
                );
            }
        }
    };
}
recovery_test!(landing_load_and_retry, "load", false);
recovery_test!(landing_store_and_retry, "store", false);
recovery_test!(landing_locked_rmw_and_retry, "locked", false);
recovery_test!(landing_vector_and_retry, "vector", false);
recovery_test!(landing_straddling_load_and_retry, "load", true);
recovery_test!(landing_straddling_store_and_retry, "store", true);
recovery_test!(landing_straddling_vector_and_retry, "vector", true);

#[test]
fn landing_mapped_protected_page_does_not_enter_rust_memory_access() {
    for region in [false, true] {
        for kind in ["load", "store", "locked"] {
            assert_eq!(
                child(|| case(kind, region, false, true)),
                0,
                "{kind}, region={region}"
            );
        }
    }
}

#[test]
fn landing_mapped_protected_page_with_recovery_returns_without_replay() {
    for region in [false, true] {
        for kind in ["load", "store", "locked"] {
            assert_eq!(child(|| case_mode(kind, region, false, true, true)), 0);
        }
    }
}

#[test]
fn landing_unrelated_host_fault_keeps_default_signal_death() {
    let status = child(|| {
        // SAFETY: deliberately fault on a valid PROT_NONE mapping in the child.
        unsafe {
            let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
            let vm = vm(Box::new(InterpreterBackend), 0, page);
            let cpu = vm.new_vcpu();
            HANDLE.store(cpu.fault_handle().as_usize(), Ordering::Relaxed);
            install();
            vm.mem.host_base().add(page * 4).read_volatile();
        }
    });
    assert!(libc::WIFSIGNALED(status));
    assert!([libc::SIGSEGV, libc::SIGBUS].contains(&libc::WTERMSIG(status)));
}

#[test]
fn landing_fault_inside_rust_helper_still_dies() {
    unsafe extern "C" fn helper(_: *mut u8, ctx: *mut u8) -> u64 {
        // SAFETY: deliberate host fault in a forked child, outside compiled code.
        unsafe {
            let ctx = &*ctx.cast::<x86jit_core::jit_abi::MemCtx>();
            (ctx.next_entry as *const u8).read_volatile();
        }
        0
    }
    let status = child(|| {
        // SAFETY: live ABI pointers and record; this child intentionally faults.
        unsafe {
            let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
            let vm = vm(Box::new(InterpreterBackend), 0, page);
            let mut cpu = CpuState::new();
            let mut ctx = x86jit_core::jit_abi::MemCtx::for_memory(&vm.mem);
            let mut record = x86jit_core::jit_abi::fault::EntryRecord::default();
            record.activate(&mut cpu, &mut ctx);
            ctx.next_entry = vm.mem.host_base() as u64 + (page * 4) as u64;
            HANDLE.store(record.handle().as_usize(), Ordering::Relaxed);
            install();
            x86jit_core::jit_abi::call_block(
                x86jit_core::CompiledPtr(helper as *const u8),
                &mut cpu,
                &mut ctx,
            );
        }
    });
    assert!(libc::WIFSIGNALED(status));
    assert!([libc::SIGSEGV, libc::SIGBUS].contains(&libc::WTERMSIG(status)));
}

// A compiled-block equivalent changes host rounding mode, then faults at a
// registered guest access. No guest MXCSR-mode implementation is needed.
#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    ".text", ".p2align 4",
    ".globl {block}", "{block}:",
    "stp x29, x30, [sp, #-16]!", "mov x29, sp", "sub sp, sp, #144",
    "stp x19, x20, [sp, #0]",
    "stp x21, x22, [sp, #16]",
    "stp x23, x24, [sp, #32]",
    "stp x25, x26, [sp, #48]",
    "stp x27, x28, [sp, #64]",
    "stp d8, d9, [sp, #80]",
    "stp d10, d11, [sp, #96]",
    "stp d12, d13, [sp, #112]",
    "stp d14, d15, [sp, #128]",
    "mov x19, #0",
    "mov x20, #0",
    "mov x21, #0",
    "mov x22, #0",
    "mov x23, #0",
    "mov x24, #0",
    "mov x25, #0",
    "mov x26, #0",
    "mov x27, #0",
    "mov x28, #0",
    "fmov d8, xzr",
    "fmov d9, xzr",
    "fmov d10, xzr",
    "fmov d11, xzr",
    "fmov d12, xzr",
    "fmov d13, xzr",
    "fmov d14, xzr",
    "fmov d15, xzr",

    // AArch64 FPCR.RMode occupies bits 23:22 (Arm ARM DDI0487J.a §C5.2.7, FPCR.RMode).
    "mrs x9, fpcr", "eor x9, x9, #0x400000", "msr fpcr, x9",
    "ldr x10, [x1, #40]",
    ".globl {site}", "{site}:", "ldr x0, [x10]",
    "ldp x19, x20, [sp, #0]",
    "ldp x21, x22, [sp, #16]",
    "ldp x23, x24, [sp, #32]",
    "ldp x25, x26, [sp, #48]",
    "ldp x27, x28, [sp, #64]",
    "ldp d8, d9, [sp, #80]",
    "ldp d10, d11, [sp, #96]",
    "ldp d12, d13, [sp, #112]",
    "ldp d14, d15, [sp, #128]",
    "add sp, sp, #144", "ldp x29, x30, [sp], #16", "ret",
    ".globl {end}", "{end}:",
    block = sym fp_block, site = sym fp_site, end = sym fp_end,
);
#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    ".text", ".p2align 4",
    ".globl {block}", "{block}:",
    "push rbp", "mov rbp, rsp", "sub rsp, 64",
    "mov [rbp - 40], rbx", "xor rbx, rbx",
    "mov [rbp - 32], r12", "xor r12, r12",
    "mov [rbp - 24], r13", "xor r13, r13",
    "mov [rbp - 16], r14", "xor r14, r14",
    "mov [rbp - 8], r15", "xor r15, r15",

    // SDM Vol 1 §10.2.3: MXCSR.RC is bits 14:13.
    "stmxcsr [rsp]", "xor dword ptr [rsp], 0x2000", "ldmxcsr [rsp]",
    "mov rax, [rsi + 40]",
    ".globl {site}", "{site}:", "mov rax, [rax]",
    "mov rbx, [rbp - 40]",
    "mov r12, [rbp - 32]",
    "mov r13, [rbp - 24]",
    "mov r14, [rbp - 16]",
    "mov r15, [rbp - 8]",
    "leave", "ret", ".globl {end}", "{end}:",
    block = sym fp_block, site = sym fp_site, end = sym fp_end,
);
extern "C" {
    fn fp_block(cpu: *mut u8, ctx: *mut u8) -> u64;
    fn fp_site();
    fn fp_end();
}

fn fp_control() -> u64 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: reading FPCR has no memory effects or privilege requirement.
    unsafe {
        let result: u64;
        core::arch::asm!("mrs {}, fpcr", out(reg) result, options(nomem, nostack));
        result
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: STMXCSR writes exactly four bytes to this local (SDM Vol 1 §10.2.3).
    unsafe {
        let mut result = 0u32;
        core::arch::asm!("stmxcsr [{}]", in(reg) &mut result, options(nostack));
        u64::from(result)
    }
}

#[test]
fn landing_restores_host_fp_control_and_saved_registers_before_callers_resume() {
    assert_eq!(
        child(|| {
            // SAFETY: all memory, CPU, entry storage, and signal pointers live through
            // the compiled call; assembly obeys the shared ABI and frame contract.
            unsafe {
                let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
                let vm = vm(Box::new(InterpreterBackend), 0, page);
                let mut cpu = CpuState::new();
                let mut ctx = x86jit_core::jit_abi::MemCtx::for_memory(&vm.mem);
                let mut record = x86jit_core::jit_abi::fault::EntryRecord::default();
                let _host_fp = HostFpMode::nondefault();
                record.activate(&mut cpu, &mut ctx);
                ctx.next_entry = vm.mem.host_base() as u64 + (page * 4) as u64;
                let entry = fp_block as *const () as usize;
                x86jit_core::codemap::register_with_recovery(
                    entry,
                    (fp_end as *const () as usize - entry) as u32,
                    Box::new([]),
                    Box::new([]),
                    vec![x86jit_core::GuestTrapSite {
                        host_offset: (fp_site as *const () as usize - entry) as u32,
                        guest_rip: 0x1_0000_1234,
                        access: x86jit_core::GuestAccessKind::Read,
                        width: 8,
                    }],
                    Some(FrameRecipe {
                        prologue_end: (fp_site as *const () as usize - entry) as u32,
                        return_auth: 0,
                        caller_sp: 16,
                        saved_fp: 0,
                        saved_return: 8,
                        gpr: std::array::from_fn(|i| {
                            if cfg!(target_arch = "aarch64") {
                                Some(-144 + i as i32 * 8)
                            } else if i < 5 {
                                Some(-40 + i as i32 * 8)
                            } else {
                                None
                            }
                        }),
                        simd: std::array::from_fn(|i| {
                            cfg!(target_arch = "aarch64").then_some(-64 + i as i32 * 8)
                        }),
                    }),
                )
                .unwrap();
                HANDLE.store(record.handle().as_usize(), Ordering::Relaxed);
                install();
                let before = fp_control();
                let patterns = register_probe::patterns();
                let mut observed = [0; register_probe::REGISTER_COUNT + 2];
                let result = register_probe::probe_call(
                    (&mut cpu as *mut CpuState).cast(),
                    (&mut ctx as *mut x86jit_core::jit_abi::MemCtx).cast(),
                    entry as *const u8,
                    observed.as_mut_ptr(),
                    patterns.as_ptr(),
                );
                let after = fp_control();
                record.deactivate();
                assert_eq!(after, before);
                assert_eq!(observed[..register_probe::REGISTER_COUNT], patterns);
                assert_eq!(result, x86jit_core::jit_abi::RET_HOST_FAULT);
                assert_eq!(cpu.rip, 0x1_0000_1234);
                assert_eq!(ctx.fault_size, 8);
                assert_eq!(ctx.fault_access, 0);
                assert_eq!(CLASSIFIER_ALLOCS.load(Ordering::Relaxed), 0);
                HANDLE.store(0, Ordering::Relaxed);
            }
        }),
        0
    );
}

#[path = "../../x86jit-cranelift/src/unwind_test_call.rs"]
mod register_probe;

#[test]
fn landing_restores_every_callee_saved_register_from_a_busy_region() {
    assert_eq!(
        child(|| {
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            let vm = vm(Box::new(InterpreterBackend), 0, page);
            let mut a = CodeAssembler::new(64).unwrap();
            let mut top = a.create_label();
            a.set_label(&mut top).unwrap();
            // Keep guest integer and vector values live through a helper call. The
            // host allocator must use preserved registers or spill them across it.
            for r in [
                rax, rbx, rcx, rdx, rsi, r8, r9, r10, r11, r12, r13, r14, r15,
            ] {
                a.add(r, qword_ptr(rdi)).unwrap();
            }
            a.cpuid().unwrap();
            for r in [
                xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12,
                xmm13, xmm14, xmm15,
            ] {
                a.addpd(r, xmmword_ptr(rdi)).unwrap();
            }
            a.dec(ebp).unwrap();
            a.jnz(top).unwrap();
            a.hlt().unwrap();
            vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
                .unwrap();
            let region = x86jit_core::lift::lift_region(
                &vm.mem,
                0x1000,
                x86jit_core::RegionCaps {
                    max_blocks: 16,
                    max_icount: 256,
                },
                x86jit_core::CpuMode::Long64,
            )
            .unwrap();
            let jit = JitBackend::new();
            let CachedBlock::Compiled { entry } =
                jit.materialize_region(&region, MemConsistency::Fast, None, 0)
            else {
                panic!("region must compile")
            };
            let mut cpu = CpuState::new();
            cpu.gpr[Reg::Rdi.gpr_index().unwrap()] = (page * 4) as u64;
            cpu.gpr[Reg::Rbp.gpr_index().unwrap()] = 2;
            cpu.rip = 0x1000;
            let mut ctx = x86jit_core::jit_abi::MemCtx::for_memory(&vm.mem);
            let mut record = x86jit_core::jit_abi::fault::EntryRecord::default();
            let patterns = register_probe::patterns();
            let mut observed = [0; register_probe::REGISTER_COUNT + 2];
            // SAFETY: pointers and standalone record stay fixed through this call;
            // the assembly caller preserves its own caller's ABI registers.
            unsafe {
                record.activate(&mut cpu, &mut ctx);
                HANDLE.store(record.handle().as_usize(), Ordering::Relaxed);
                install();
                let result = register_probe::probe_call(
                    (&mut cpu as *mut CpuState).cast(),
                    (&mut ctx as *mut x86jit_core::jit_abi::MemCtx).cast(),
                    entry.0,
                    observed.as_mut_ptr(),
                    patterns.as_ptr(),
                );
                record.deactivate();
                assert_eq!(result, x86jit_core::jit_abi::RET_HOST_FAULT);
            }
            assert_eq!(observed[..register_probe::REGISTER_COUNT], patterns);
            assert_eq!(FAULTS.load(Ordering::Relaxed), 1);
        }),
        0
    );
}

#[test]
fn alignment_faults_complete_locked_operations_in_block_and_region_modes() {
    // Intel SDM Vol 3A §9.1.1: locked operations preserve operand semantics
    // across alignment boundaries. Cross-thread replay atomicity is xj-0w3.5.
    for kind in ["add", "xchg", "cmpxchg"] {
        for region in [false, true] {
            assert_eq!(
                child(|| {
                    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
                    let base = 0x1_0000_0000;
                    let data = base + (page * 2) as u64;
                    let mut a = CodeAssembler::new(64).unwrap();
                    a.mov(ecx, 2).unwrap();
                    let mut top = a.create_label();
                    a.set_label(&mut top).unwrap();
                    match kind {
                        "add" => a.lock().add(qword_ptr(rdi + 12), 1).unwrap(),
                        "xchg" => a.xchg(qword_ptr(rdi + 12), rbx).unwrap(),
                        "cmpxchg" => a.lock().cmpxchg(qword_ptr(rdi + 12), rbx).unwrap(),
                        _ => unreachable!(),
                    }
                    a.dec(ecx).unwrap();
                    a.jnz(top).unwrap();
                    a.hlt().unwrap();
                    let code = a.assemble(base + 0x1000).unwrap();
                    let interp = vm(Box::new(InterpreterBackend), base, page);
                    let jit = vm(
                        Box::new(Compiled {
                            jit: JitBackend::new(),
                            region,
                        }),
                        base,
                        page,
                    );
                    for v in [&interp, &jit] {
                        v.write_bytes(base + 0x1000, &code).unwrap();
                        v.write_bytes(data + 12, &42u64.to_le_bytes()).unwrap();
                    }
                    let mut expected = interp.new_vcpu();
                    expected.set_reg(Reg::Rip, base + 0x1000);
                    expected.set_reg(Reg::Rdi, data);
                    expected.set_reg(Reg::Rax, 42);
                    expected.set_reg(Reg::Rbx, 77);
                    let mut actual = jit.new_vcpu();
                    actual.cpu = expected.cpu.clone();
                    assert!(matches!(expected.run(&interp, None), Exit::Hlt));
                    HANDLE.store(actual.fault_handle().as_usize(), Ordering::Relaxed);
                    unsafe {
                        install();
                    }
                    let exit = actual.run(&jit, None);
                    assert!(
                        matches!(exit, Exit::Hlt),
                        "{kind}, region={region}: {exit:?}"
                    );
                    same_cpu(&actual.cpu, &expected.cpu);
                    let mut eb = [0; 8];
                    let mut ab = [0; 8];
                    interp.mem.read_bytes(data + 12, &mut eb).unwrap();
                    jit.mem.read_bytes(data + 12, &mut ab).unwrap();
                    assert_eq!(ab, eb);
                    assert_eq!(jit.cache.regions() > 0, region);
                    #[cfg(target_arch = "aarch64")]
                    assert!(ALIGNMENTS.load(Ordering::Relaxed) > 0);
                }),
                0,
                "{kind}, region={region}"
            );
        }
    }
}

#[test]
fn user_sent_sigsegv_is_not_a_guest_memory_fault() {
    assert_eq!(fault_kind(libc::SIGSEGV, 0), HostFaultKind::Unsupported);
    assert_eq!(fault_kind(libc::SIGSEGV, -1), HostFaultKind::Unsupported);
    assert_eq!(
        fault_kind(libc::SIGSEGV, SEGV_MAPERR),
        HostFaultKind::Access
    );
    assert_eq!(
        fault_kind(libc::SIGSEGV, SEGV_ACCERR),
        HostFaultKind::Access
    );
    assert_eq!(
        fault_kind(libc::SIGBUS, libc::BUS_ADRERR),
        if cfg!(target_os = "freebsd") {
            HostFaultKind::Unsupported
        } else {
            HostFaultKind::Access
        }
    );
    assert_eq!(
        fault_kind(libc::SIGBUS, libc::BUS_ADRALN),
        HostFaultKind::Alignment
    );
    assert_eq!(
        fault_kind(libc::SIGBUS, libc::BUS_OBJERR),
        HostFaultKind::Unsupported
    );
    let status = child(|| unsafe {
        install();
        libc::raise(libc::SIGSEGV);
    });
    assert!(libc::WIFSIGNALED(status));
    assert_eq!(libc::WTERMSIG(status), libc::SIGSEGV);
}

#[test]
fn recovery_children_have_a_watchdog() {
    let status = child(|| unsafe {
        let remaining = libc::alarm(0);
        assert!((1..=15).contains(&remaining));
        libc::alarm(1);
        loop {
            libc::pause();
        }
    });
    assert!(libc::WIFSIGNALED(status));
    assert_eq!(libc::WTERMSIG(status), libc::SIGALRM);
}

struct HostFpMode(u64);
impl HostFpMode {
    fn nondefault() -> Self {
        let original = fp_control();
        #[cfg(target_arch = "aarch64")]
        let value = original | (1 << 24); // Arm ARM DDI0487J.a §C5.2.7, FPCR.FZ.
        #[cfg(target_arch = "x86_64")]
        let value = original | (1 << 13); // SDM Vol 1 §10.2.3.2, MXCSR.RC.
                                          // SAFETY: changes only a documented non-reserved FP control field.
        unsafe {
            set_fp_control(value);
        }
        assert_ne!(fp_control(), 0);
        Self(original)
    }
}
impl Drop for HostFpMode {
    fn drop(&mut self) {
        // SAFETY: restore the value read from this thread on construction.
        unsafe {
            set_fp_control(self.0);
        }
    }
}
unsafe fn set_fp_control(value: u64) {
    #[cfg(target_arch = "aarch64")]
    core::arch::asm!("msr fpcr, {}", in(reg) value, options(nomem, nostack));
    #[cfg(target_arch = "x86_64")]
    core::arch::asm!("ldmxcsr [{}]", in(reg) &value, options(nostack));
}

#[cfg(target_arch = "aarch64")]
#[test]
fn data_abort_syndrome_distinguishes_access_and_alignment() {
    for ec in [0x24, 0x25] {
        for dfsc in 4..=15 {
            assert_eq!(esr_fault_kind((ec << 26) | dfsc), HostFaultKind::Access);
        }
        assert_eq!(esr_fault_kind((ec << 26) | 0x21), HostFaultKind::Alignment);
        for dfsc in [0, 3, 0x10, 0x14, 0x3f] {
            assert_eq!(
                esr_fault_kind((ec << 26) | dfsc),
                HostFaultKind::Unsupported
            );
        }
    }
    assert_eq!(esr_fault_kind((0x20 << 26) | 7), HostFaultKind::Unsupported);
}
