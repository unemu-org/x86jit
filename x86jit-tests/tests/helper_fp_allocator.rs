//! Embedder allocators reached by compiled helpers require default host control.
use iced_x86::code_asm::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use x86jit_core::{Exit, GuestCpuFeatures, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

static SERIAL: Mutex<()> = Mutex::new(());
static ARMED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
static BAD_CONTROL: AtomicU64 = AtomicU64::new(0);
static BAD_RESULT: AtomicU64 = AtomicU64::new(0);
const ONE: u64 = 0x3ff0_0000_0000_0000;

fn control() -> u64 {
    // SAFETY: read the calling thread's FP control without changing it.
    unsafe {
        #[cfg(target_arch = "aarch64")]
        {
            let value;
            core::arch::asm!("mrs {}, fpcr", out(reg) value, options(nomem, nostack));
            value
        }
        #[cfg(target_arch = "x86_64")]
        {
            let mut value = 0u32;
            core::arch::asm!("stmxcsr [{}]", in(reg) &mut value, options(nostack));
            u64::from(value)
        }
    }
}

fn probe() {
    if !ARMED.load(Ordering::Relaxed) {
        return;
    }
    let fp = control();
    CALLS.fetch_add(1, Ordering::Relaxed);
    if fp != x86jit_core::jit_abi::fp::default_control() {
        BAD_CONTROL.store(fp, Ordering::Relaxed);
    }
    // SDM Vol 1 §4.8.4: 1 + 2^-53 is a tie, rounding to 1 under nearest-even.
    // Explicit arithmetic keeps this probe sensitive to the actual host control.
    let mut result = f64::from_bits(ONE);
    let half = f64::from_bits(0x3ca0_0000_0000_0000);
    // SAFETY: scalar FP with masked exceptions; restore status bits as well as
    // control so the probe itself cannot contaminate the next allocator entry.
    unsafe {
        #[cfg(target_arch = "aarch64")]
        core::arch::asm!("fadd {a:d}, {a:d}, {b:d}", a = inout(vreg) result, b = in(vreg) half, options(nostack));
        #[cfg(target_arch = "x86_64")]
        core::arch::asm!("addsd {a}, {b}", a = inout(xmm_reg) result, b = in(xmm_reg) half, options(nostack));
        x86jit_core::jit_abi::fp::set_control(fp);
    }
    if result.to_bits() != ONE {
        BAD_RESULT.store(result.to_bits(), Ordering::Relaxed);
    }
}

struct ProbeAllocator;
// SAFETY: delegate every allocation operation with its original arguments.
unsafe impl GlobalAlloc for ProbeAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        probe();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        probe();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        probe();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        probe();
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: ProbeAllocator = ProbeAllocator;

fn witness(kind: &str, mode: u32) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for timing in [false, true] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(JitBackend::new()));
        vm.set_guest_cpu_features(GuestCpuFeatures::v4());
        vm.set_time_accounting(timing);
        vm.map(0, 0x1f000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x1f000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        let mut a = CodeAssembler::new(64).unwrap();
        let helper = match kind {
            "scalar" => {
                a.mov(qword_ptr(rdi), rbx).unwrap();
                "note_write_helper"
            }
            "string" => {
                a.rep().stosq().unwrap();
                "string_helper::<true>"
            }
            "cas" | "cas_mmio" | "cas_align" | "cas_unmapped" => {
                a.cmpxchg16b(ptr(rdi)).unwrap();
                "cas_pair_helper"
            }
            "masked" => {
                a.vmovdqu32(zmmword_ptr(rdi).k1(), zmm0).unwrap();
                "vmaskmov_mem_helper"
            }
            "vec_masked" => {
                a.vmaskmovps(xmmword_ptr(rdi), xmm1, xmm0).unwrap();
                "vec_maskmov_mem_helper"
            }
            "narrow" => {
                a.vpmovqb(xmmword_ptr(rdi), zmm0).unwrap();
                "vpmov_narrow_mem_helper"
            }
            "fxsave" => {
                a.fxsave64(ptr(rdi)).unwrap();
                "fxstate_helper"
            }
            "xsave" => {
                a.xsave64(ptr(rdi)).unwrap();
                "xstate_helper"
            }
            "xsaveopt" => {
                a.xsaveopt64(ptr(rdi)).unwrap();
                "xstate_helper"
            }
            "xsavec" => {
                a.xsavec64(ptr(rdi)).unwrap();
                "xstate_helper"
            }
            _ => unreachable!(),
        };
        a.addsd(xmm2, xmm3).unwrap();
        a.hlt().unwrap();
        vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
            .unwrap();
        let mut cpu = vm.new_vcpu_with_mxcsr(mode);
        let seed = |cpu: &mut x86jit_core::Vcpu, addr| {
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rdi, addr);
            cpu.set_reg(Reg::Rax, 3);
            cpu.set_reg(Reg::Rdx, 0);
            cpu.set_reg(Reg::Rcx, 1);
            cpu.set_reg(Reg::Rbx, 7);
            cpu.cpu.kmask[1] = 1;
            cpu.set_xmm(1, u128::MAX);
            cpu.set_xmm(2, u128::from(ONE));
            cpu.set_xmm(3, 0x3ca0_0000_0000_0000);
        };
        // Compile and warm on ordinary data, then drive the cold path in that JIT.
        seed(&mut cpu, 0x18000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        if !matches!(kind, "cas_mmio" | "cas_align" | "cas_unmapped") && !kind.starts_with("xsave")
        {
            vm.mem.mark_code(0x18000, 512);
        }
        seed(
            &mut cpu,
            match kind {
                "cas_mmio" => 0x1f000,
                "cas_align" => 0x18001,
                "cas_unmapped" => 0x30000,
                _ => 0x18000,
            },
        );
        let count = || {
            vm.backend
                .helper_calls()
                .iter()
                .find(|(name, _)| *name == helper)
                .map_or(0, |(_, n)| *n)
        };
        let before = count();
        CALLS.store(0, Ordering::Relaxed);
        BAD_CONTROL.store(0, Ordering::Relaxed);
        BAD_RESULT.store(0, Ordering::Relaxed);
        ARMED.store(true, Ordering::Relaxed);
        let exit = cpu.run(&vm, None);
        ARMED.store(false, Ordering::Relaxed);
        let calls = CALLS.load(Ordering::Relaxed);
        let bad_control = BAD_CONTROL.load(Ordering::Relaxed);
        let bad_result = BAD_RESULT.load(Ordering::Relaxed);
        assert!(count() > before, "must execute {helper} through the JIT");
        if kind == "cas_mmio" {
            assert!(matches!(exit, Exit::UnknownInstruction { .. }));
        } else if kind == "cas_align" {
            assert!(matches!(exit, Exit::Exception { vector: 13, .. }));
        } else if kind == "cas_unmapped" {
            assert!(matches!(exit, Exit::UnmappedMemory { .. }));
        } else {
            assert!(matches!(exit, Exit::Hlt), "{exit:?}");
            assert_eq!(
                cpu.xmm(2) as u64,
                ONE + u64::from(mode == 0x5f80),
                "guest rounding restored after helper"
            );
        }
        assert!(calls > 0, "must reach the allocator on {kind}");
        assert_eq!(
            (bad_control, bad_result),
            (0, 0),
            "{kind}, MXCSR={mode:04x}, timing={timing}, allocator calls={calls}"
        );
    }
}

macro_rules! witnesses {
    ($($up:ident, $flush:ident => $kind:literal;)*) => { $(
        #[test] fn $up() { witness($kind, 0x5f80); }
        #[test] fn $flush() { witness($kind, 0x9fc0); }
    )* };
}
witnesses! {
    scalar_up, scalar_flush => "scalar";
    string_up, string_flush => "string";
    cas_up, cas_flush => "cas";
    cas_mmio_up, cas_mmio_flush => "cas_mmio";
    cas_align_up, cas_align_flush => "cas_align";
    cas_unmapped_up, cas_unmapped_flush => "cas_unmapped";
    masked_up, masked_flush => "masked";
    vec_masked_up, vec_masked_flush => "vec_masked";
    narrow_up, narrow_flush => "narrow";
    fxsave_up, fxsave_flush => "fxsave";
    xsave_up, xsave_flush => "xsave";
    xsaveopt_up, xsaveopt_flush => "xsaveopt";
    xsavec_up, xsavec_flush => "xsavec";
}

// These query implementations allocate their map buffer. Both host architectures
// use the same witnesses; deterministic read denial avoids OS protection differences.
#[cfg(any(target_os = "linux", target_os = "freebsd"))]
fn permission_witness(kind: &str, mode: u32) {
    use x86jit_core::HostRam;
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: the host page-size query has no pointer arguments.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let len = (page * 4).max(0x20000);
    let layout = Layout::from_size_align(len, page).unwrap();
    // SAFETY: nonzero, page-aligned backing, owned by HostRam until deallocation.
    let backing = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!backing.is_null());
    let ram = HostRam {
        ptr: backing,
        len: layout.size(),
        guest_base: 0,
        protect: None,
        dtor: Box::new(move |ptr, _| {
            // SAFETY: HostRam transfers the original allocation back exactly once.
            unsafe { std::alloc::dealloc(ptr, layout) };
        }),
    };
    let mut vm =
        Vm::with_backend_host_ram(VmConfig::flat(len as u64), Box::new(JitBackend::new()), ram)
            .unwrap();
    vm.set_guest_cpu_features(GuestCpuFeatures::v4());
    // codegen::preflight_split_store calls directly, without call_helper's
    // per-helper counter. Its account_helper timer still counts execution.
    vm.set_time_accounting(kind == "preflight");
    vm.mem.set_host_fault_recovery(true);
    vm.map(0, len, Prot::RWX, RegionKind::Ram).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    let helper = match kind {
        "preflight" => {
            a.vcvtps2ph(xmmword_ptr(rdi), ymm0, 0).unwrap();
            "vstore_preflight_helper"
        }
        "string" => {
            a.rep().stosq().unwrap();
            "string_helper::<true>"
        }
        "masked" => {
            a.vmovdqu32(zmmword_ptr(rdi).k1(), zmm0).unwrap();
            "vmaskmov_mem_helper"
        }
        "vec_masked" => {
            a.vmaskmovps(xmmword_ptr(rdi), xmm1, xmm0).unwrap();
            "vec_maskmov_mem_helper"
        }
        "narrow" => {
            a.vpmovqb(xmmword_ptr(rdi), zmm0).unwrap();
            "vpmov_narrow_mem_helper"
        }
        "fxsave" => {
            a.fxsave64(ptr(rdi)).unwrap();
            "fxstate_helper"
        }
        "xsavec" => {
            a.xsavec64(ptr(rdi)).unwrap();
            "xstate_helper"
        }
        _ => unreachable!(),
    };
    a.addsd(xmm2, xmm3).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
        .unwrap();
    let mut cpu = vm.new_vcpu_with_mxcsr(mode);
    let addr = (page * 3) as u64
        - if matches!(kind, "fxsave" | "xsavec") {
            256
        } else {
            1
        };
    let seed = |cpu: &mut x86jit_core::Vcpu| {
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.set_reg(Reg::Rdi, addr);
        cpu.set_reg(Reg::Rax, if kind == "xsavec" { 3 } else { 0 });
        cpu.set_reg(Reg::Rdx, 0);
        cpu.set_reg(Reg::Rcx, 1);
        cpu.cpu.kmask[1] = 1;
        cpu.set_xmm(1, u128::MAX);
        cpu.set_xmm(2, u128::from(ONE));
        cpu.set_xmm(3, 0x3ca0_0000_0000_0000);
    };
    seed(&mut cpu);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    seed(&mut cpu);
    let before = vm.backend.helper_calls();
    let helpers_before = vm.time_stats().helpers.calls;
    CALLS.store(0, Ordering::Relaxed);
    BAD_CONTROL.store(0, Ordering::Relaxed);
    BAD_RESULT.store(0, Ordering::Relaxed);
    let denied = backing as usize + page * 3;
    ARMED.store(true, Ordering::Relaxed);
    let exit =
        x86jit_core::with_copy_out_fault(denied..denied + page, usize::MAX, || cpu.run(&vm, None));
    ARMED.store(false, Ordering::Relaxed);
    let after = vm.backend.helper_calls();
    let count = |rows: &[(&str, u64)], helper: &str| {
        rows.iter()
            .find(|(name, _)| *name == helper)
            .map_or(0, |(_, n)| *n)
    };
    if kind == "preflight" {
        let counted = |rows: &[(&str, u64)]| rows.iter().map(|(_, n)| n).sum::<u64>();
        // The conversion helper is counted by both diagnostics; the crossing
        // preflight is the only call counted solely by the helper timer.
        assert_eq!(
            count(&after, "cvtps2ph_value_helper") - count(&before, "cvtps2ph_value_helper"),
            1,
            "must execute the conversion through the JIT"
        );
        assert_eq!(
            vm.time_stats().helpers.calls - helpers_before,
            counted(&after) - counted(&before) + 1,
            "must execute {helper}"
        );
    } else {
        assert!(
            count(&after, helper) > count(&before, helper),
            "must execute {helper}"
        );
    }
    assert!(matches!(exit, Exit::Hlt), "{exit:?}");
    assert!(
        CALLS.load(Ordering::Relaxed) > 0,
        "permission query must allocate"
    );
    assert_eq!(
        (
            BAD_CONTROL.load(Ordering::Relaxed),
            BAD_RESULT.load(Ordering::Relaxed)
        ),
        (0, 0)
    );
    assert_eq!(cpu.xmm(2) as u64, ONE + u64::from(mode == 0x5f80));
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
mod permission_queries {
    use super::*;
    macro_rules! queries {
        ($($up:ident, $flush:ident => $kind:literal;)*) => { $(
            #[test] fn $up() { permission_witness($kind, 0x5f80); }
            #[test] fn $flush() { permission_witness($kind, 0x9fc0); }
        )* };
    }
    queries! {
        preflight_up, preflight_flush => "preflight";
        string_up, string_flush => "string";
        masked_up, masked_flush => "masked";
        vec_masked_up, vec_masked_flush => "vec_masked";
        narrow_up, narrow_flush => "narrow";
        fxsave_up, fxsave_flush => "fxsave";
        xsavec_up, xsavec_flush => "xsavec";
    }
}
