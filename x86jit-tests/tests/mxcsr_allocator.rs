//! Allocator callbacks must enter Rust with default floating-point control.
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;
static ARMED: AtomicBool = AtomicBool::new(false);
static OBSERVED: AtomicU64 = AtomicU64::new(0);
static FLOAT_RESULT: AtomicU64 = AtomicU64::new(0);
fn control() -> u64 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v;
        core::arch::asm!("mrs {}, fpcr",out(reg)v,options(nomem,nostack));
        v
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut v = 0u32;
        core::arch::asm!("stmxcsr [{}]",in(reg)&mut v,options(nostack));
        u64::from(v)
    }
}
struct ProbeAllocator;
unsafe impl std::alloc::GlobalAlloc for ProbeAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            let fp = control();
            if fp != x86jit_core::jit_abi::fp::default_control() {
                OBSERVED.store(fp, Ordering::Relaxed);
                // SDM Vol 1 §4.8.4: this tie rounds down under nearest-even.
                let x = std::hint::black_box(1.0f64);
                let y = std::hint::black_box(f64::from_bits(0x3ca0000000000000));
                FLOAT_RESULT.store((x + y).to_bits(), Ordering::Relaxed);
            }
        }
        std::alloc::System.alloc(layout)
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: std::alloc::Layout) {
        std::alloc::System.dealloc(p, layout)
    }
}
#[global_allocator]
static ALLOCATOR: ProbeAllocator = ProbeAllocator;
#[test]
fn mxcsr_modes_ibtc_allocator_uses_default_fp() {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(JitBackend::new()));
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, &[0xff, 0xe0]).unwrap(); // jmp rax
    vm.write_bytes(0x2000, &[0xf4]).unwrap(); // hlt
    let mut cpu = vm.new_vcpu_with_mxcsr(0x5f80);
    cpu.set_reg(Reg::Rip, 0x2000);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt)); // seed fast cache
    cpu.set_reg(Reg::Rip, 0x1000);
    cpu.set_reg(Reg::Rax, 0x2000);
    ARMED.store(true, Ordering::Relaxed);
    let exit = cpu.run(&vm, None);
    ARMED.store(false, Ordering::Relaxed);
    assert!(matches!(exit, Exit::Hlt));
    assert!(
        cpu.fast_hits() > 0,
        "the indirect target must hit the fast cache"
    );
    assert_eq!(
        vm.cache.ibtc_filled(),
        1,
        "the miss must refill a descriptor"
    );
    let observed = OBSERVED.load(Ordering::Relaxed);
    let result = FLOAT_RESULT.load(Ordering::Relaxed);
    eprintln!("allocator control={observed:#x}, float result={result:#x}, expected=0x3ff0000000000000, IBTC fills={}",vm.cache.ibtc_filled());
    assert_eq!(observed, 0, "global allocator entered under guest FP mode");
}
