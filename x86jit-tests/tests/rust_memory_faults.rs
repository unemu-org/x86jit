//! Rust transfers recover without skipping frames. Guest fault state follows
//! Intel SDM Vol 3A §7.5; REP progress follows Vol 2B, REP—Repeat String Operation (Prefix),
//! Description. Aligned scalar atomicity follows Vol 3A §11.1.1.
#![cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]

use iced_x86::code_asm::*;
use std::sync::atomic::Ordering;
use x86jit_core::{
    memory::HostRam, Backend, CachedBlock, CpuState, Exit, InterpreterBackend, IrBlock, IrRegion,
    MemConsistency, Prot, Reg, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;
#[path = "../../x86jit-cranelift/src/unwind_test_signal.rs"]
#[allow(dead_code)]
mod signal;

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
        base: u64,
    ) -> CachedBlock {
        let block = self.jit.materialize(ir, consistency, mmio, base);
        assert!(matches!(block, CachedBlock::Compiled { .. }));
        block
    }
    fn materialize_region(
        &self,
        ir: &IrRegion,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        base: u64,
    ) -> CachedBlock {
        let block = self.jit.materialize_region(ir, consistency, mmio, base);
        assert!(matches!(block, CachedBlock::Compiled { .. }));
        block
    }
}
static FORK_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());
const BASE: u64 = 0x1_0000_0000;
fn page() -> usize {
    // SAFETY: sysconf has no pointer argument.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}
fn vm(tier: u8) -> Vm {
    let len = page() * 8;
    let backend: Box<dyn Backend> = if tier == 0 {
        Box::new(InterpreterBackend)
    } else {
        Box::new(Compiled {
            jit: JitBackend::new(),
            region: tier == 2,
        })
    };
    // SAFETY: the mapping stays owned by HostRam and is freed exactly once.
    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        assert_ne!(ptr, libc::MAP_FAILED);
        let mut vm = Vm::with_backend_host_ram(
            VmConfig::flat(BASE + len as u64),
            backend,
            HostRam {
                ptr: ptr.cast(),
                len,
                guest_base: BASE,
                protect: None,
                dtor: Box::new(|p, len| {
                    libc::munmap(p.cast(), len);
                }),
            },
        )
        .unwrap();
        vm.mem.set_host_fault_recovery(true);
        vm.map(BASE, len, Prot::RWX, RegionKind::Ram).unwrap();
        vm
    }
}
fn protect(vm: &Vm, enabled: bool) {
    // SAFETY: the page is aligned and belongs to this child's live mapping.
    unsafe {
        assert_eq!(
            libc::mprotect(
                vm.mem.host_base().add(page() * 4).cast_mut().cast(),
                page(),
                if enabled {
                    libc::PROT_NONE
                } else {
                    libc::PROT_READ | libc::PROT_WRITE
                }
            ),
            0
        );
    }
}
fn same_cpu(a: &CpuState, b: &CpuState) {
    assert_eq!(a.gpr, b.gpr);
    assert_eq!(a.rip, b.rip);
    assert_eq!(a.flags.to_rflags(), b.flags.to_rflags());
    assert_eq!(a.xmm, b.xmm);
    assert_eq!(a.ymm_hi, b.ymm_hi);
    assert_eq!(a.zmm_hi, b.zmm_hi);
    assert_eq!(a.fpr, b.fpr);
    assert_eq!(a.fpu_top, b.fpu_top);
    assert_eq!(a.fpu_sw, b.fpu_sw);
    assert_eq!(a.fpu_cw, b.fpu_cw);
    assert_eq!(a.fpu_empty, b.fpu_empty);
}
fn transfer_case(kind: &str, tier: u8, straddle: bool, readonly: bool) {
    transfer_mode_case(kind, tier, straddle, readonly, true);
}
fn transfer_mode_case(kind: &str, tier: u8, straddle: bool, readonly: bool, routed: bool) {
    let mut vm = vm(tier);
    vm.mem.set_host_fault_recovery(routed);
    let reference = self::vm(0);
    let code_addr = BASE + 0x1000;
    let guard = BASE + (page() * 4) as u64;
    let addr = guard
        - if straddle {
            match kind {
                "fxsave" | "fxrstor" => 256,
                "vector128_store" => 8,
                "vector" | "vector_store" | "masked_store" | "extract_store" | "dpps"
                | "blendv" | "blendi" | "round" | "psign" | "pmadd" | "kstore" | "kstore8" => 16,
                "narrow_store" => 8,
                "logic" | "popcnt" | "ternlog" | "packed" => 32,
                "rep" | "rep_store" | "rep_stos" => 2,
                _ => 4,
            }
        } else {
            0
        };
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    let mut tail = a.create_label();
    a.set_label(&mut top).unwrap();
    match kind {
        "load" => a.mov(rax, qword_ptr(rdi)).unwrap(),
        "store" => a.mov(qword_ptr(rdi), rax).unwrap(),
        "locked" => a.lock().add(qword_ptr(rdi), 1).unwrap(),
        "cas" => a.lock().cmpxchg(qword_ptr(rdi), rbx).unwrap(),
        "pair" => a.lock().cmpxchg16b(xmmword_ptr(rdi)).unwrap(),
        "x87" | "x87_empty" => a.fstp(qword_ptr(rdi)).unwrap(),
        "fxsave" => a.fxsave64(ptr(rdi)).unwrap(),
        "rep" => a.rep().movsb().unwrap(),
        "vector" => a.vmovdqu(ymm0, ymmword_ptr(rdi)).unwrap(),
        "vector128_store" => a.movdqu(xmmword_ptr(rdi), xmm0).unwrap(),
        "vector_store" => a.vmovdqu(ymmword_ptr(rdi), ymm0).unwrap(),
        "masked_store" => a.vmaskmovps(ymmword_ptr(rdi), ymm1, ymm0).unwrap(),
        "extract_store" => a.vextracti64x4(ymmword_ptr(rdi), zmm0, 0).unwrap(),
        "rep_store" => a.rep().movsb().unwrap(),
        "rep_stos" => a.rep().stosb().unwrap(),
        "x87_load" => a.fld(qword_ptr(rdi)).unwrap(),
        "fxrstor" => a.fxrstor64(ptr(rdi)).unwrap(),
        "dpps" => a.vdpps(ymm0, ymm0, ymmword_ptr(rdi), 0xff).unwrap(),
        "logic" => a.vpandd(zmm0, zmm0, zmmword_ptr(rdi)).unwrap(),
        "popcnt" => a.vpopcntd(zmm0, zmmword_ptr(rdi)).unwrap(),
        "blendv" => a.vblendvps(ymm0, ymm0, ymmword_ptr(rdi), ymm1).unwrap(),
        "blendi" => a.vblendps(ymm0, ymm0, ymmword_ptr(rdi), 0xff).unwrap(),
        "round" => a.vroundps(ymm0, ymmword_ptr(rdi), 0).unwrap(),
        "ternlog" => a.vpternlogd(zmm0, zmm1, zmmword_ptr(rdi), 0x96).unwrap(),
        "pmadd" => a.vpmaddwd(ymm0, ymm0, ymmword_ptr(rdi)).unwrap(),
        "psign" => a.vpsignd(ymm0, ymm0, ymmword_ptr(rdi)).unwrap(),
        "kstore" => a.vmovdqu32(ymmword_ptr(rdi).k1(), ymm0).unwrap(),
        "kstore8" => a.vmovdqu8(ymmword_ptr(rdi).k1(), ymm0).unwrap(),
        "narrow_store" => a.vpmovqd(xmmword_ptr(rdi), ymm0).unwrap(),
        "packed" => a.vpaddd(zmm0, zmm0, zmmword_ptr(rdi)).unwrap(),
        _ => unreachable!(),
    }
    a.add(r12, 1).unwrap();
    a.jmp(tail).unwrap();
    a.set_label(&mut tail).unwrap();
    a.dec(r8).unwrap();
    a.jnz(top).unwrap();
    a.hlt().unwrap();
    let code = a.assemble(code_addr).unwrap();
    for v in [&vm, &reference] {
        v.write_bytes(code_addr, &code).unwrap();
        v.write_bytes(guard - 512, &[0x53; 1024]).unwrap();
    }
    if matches!(kind, "rep" | "rep_store") {
        for v in [&vm, &reference] {
            v.write_bytes(
                if kind == "rep" { addr } else { guard - 512 },
                &[0xa1, 0xa2, 0xa3, 0xa4],
            )
            .unwrap();
        }
    }
    if kind == "psign" {
        for v in [&vm, &reference] {
            v.write_bytes(guard - 16, &[0xff; 32]).unwrap();
        }
    }
    let mut expected_prefix = [0; 512];
    vm.read_bytes(guard - 512, &mut expected_prefix).unwrap();
    let mut actual = vm.new_vcpu();
    actual.cpu.gpr = std::array::from_fn(|i| 0x8000 + i as u64);
    actual.set_reg(Reg::Rip, code_addr);
    actual.set_reg(Reg::Rdi, addr);
    actual.set_reg(Reg::Rsi, guard - 512);
    actual.set_reg(Reg::Rcx, 4);
    actual.set_reg(Reg::R8, 2);
    actual.cpu.fpu_empty = if kind == "x87_empty" { 0xff } else { 0 };
    actual
        .cpu
        .set_vec(0, [0x4000_0000_4000_0000_4000_0000_4000_0000; 4], 64);
    actual.cpu.set_vec(1, [u128::MAX; 4], 64);
    actual.cpu.kmask[1] = u64::MAX;
    if kind == "logic" {
        actual.cpu.set_vec(0, [u128::MAX; 4], 64);
    }
    if !matches!(
        kind,
        "logic" | "popcnt" | "ternlog" | "packed" | "extract_store"
    ) {
        actual.cpu.zmm_hi[0] = [0; 2];
    }
    if kind == "rep_stos" {
        actual.set_reg(Reg::Rax, 0xa5);
    }
    actual.cpu.fpr[0] = x86jit_core::f80::F80::from_i64(7).to_bytes();
    if kind == "rep" {
        actual.set_reg(Reg::Rsi, addr);
        actual.set_reg(Reg::Rdi, guard - 512);
    }
    let mut expected = reference.new_vcpu();
    expected.cpu = actual.cpu.clone();
    let mut before = actual.cpu.clone();
    if matches!(kind, "rep" | "rep_store" | "rep_stos") && straddle {
        // Completed iterations are architecturally committed; the faulting
        // iteration commits nothing (SDM Vol 2B, REP, Description).
        if kind != "rep_stos" {
            before.gpr[6] += 2;
        }
        before.gpr[7] += 2;
        before.gpr[1] -= 2;
        let offset = if kind == "rep" { 0 } else { 510 };
        expected_prefix[offset..offset + 2].copy_from_slice(if kind == "rep_stos" {
            &[0xa5, 0xa5]
        } else {
            &[0xa1, 0xa2]
        });
    }
    signal::HANDLE.store(actual.fault_handle().as_usize(), Ordering::Relaxed);
    protect(&vm, true);
    if readonly {
        // SAFETY: this page belongs to the child's mapping; reads remain accessible.
        unsafe {
            assert_eq!(
                libc::mprotect(
                    vm.mem.host_base().add(page() * 4).cast_mut().cast(),
                    page(),
                    libc::PROT_READ
                ),
                0
            );
        }
    }
    let access = if matches!(
        kind,
        "load"
            | "vector"
            | "rep"
            | "x87_load"
            | "fxrstor"
            | "dpps"
            | "logic"
            | "popcnt"
            | "blendv"
            | "blendi"
            | "round"
            | "ternlog"
            | "packed"
            | "psign"
            | "pmadd"
    ) {
        x86jit_core::AccessKind::Read
    } else {
        x86jit_core::AccessKind::Write
    };
    let exit = actual.run(&vm, None);
    assert!(
        matches!(exit, Exit::UnmappedMemory { addr, access: kind, .. } if addr == guard && kind == access),
        "{exit:?}: {kind} tier={tier} straddle={straddle}"
    );
    same_cpu(&actual.cpu, &before);
    let mut bytes = [0; 512];
    vm.read_bytes(guard - 512, &mut bytes).unwrap();
    if kind == "vector_store" && tier != 0 && straddle {
        // xj-obx.8: compiled 256-bit stores commit their first 128-bit transfer
        // before the high transfer faults. Resume rewrites those same bytes.
        expected_prefix[496..].copy_from_slice(&before.xmm[0].to_le_bytes());
    }
    // On AArch64 the compiled misaligned scalar straddle witnesses this host;
    // whole-store fault atomicity on other implementations is under xj-obx.8.
    assert_eq!(bytes, expected_prefix);
    protect(&vm, false);
    assert!(matches!(actual.run(&vm, None), Exit::Hlt));
    assert!(matches!(expected.run(&reference, None), Exit::Hlt));
    same_cpu(&actual.cpu, &expected.cpu);
    let mut ref_bytes = [0; 1024];
    let mut bytes = [0; 1024];
    vm.read_bytes(guard - 512, &mut bytes).unwrap();
    reference.read_bytes(guard - 512, &mut ref_bytes).unwrap();
    assert_eq!(bytes, ref_bytes);
    if tier == 2 {
        assert!(vm.cache.regions() > 0);
    }
}
#[test]
fn protected_rust_transfers_resume_in_all_tiers() {
    let _serial = FORK_TEST.lock().unwrap_or_else(|p| p.into_inner());
    let mut failures = Vec::new();
    for tier in 0..3 {
        for kind in [
            "load",
            "store",
            "locked",
            "cas",
            "pair",
            "x87",
            "x87_empty",
            "fxsave",
            "rep",
            "vector",
            "rep_store",
            "rep_stos",
            "x87_load",
            "fxrstor",
        ] {
            for straddle in [false, true] {
                if straddle && kind == "pair" {
                    continue;
                }
                for readonly in [false, true] {
                    if readonly
                        && matches!(
                            kind,
                            "load"
                                | "vector"
                                | "rep"
                                | "x87_load"
                                | "fxrstor"
                                | "dpps"
                                | "logic"
                                | "popcnt"
                                | "blendv"
                                | "blendi"
                                | "round"
                                | "ternlog"
                                | "packed"
                        )
                    {
                        continue;
                    }
                    let status = signal::child(|| {
                        unsafe {
                            signal::install();
                        }
                        transfer_case(kind, tier, straddle, readonly);
                    });
                    if status != 0 {
                        failures.push((kind, tier, straddle, readonly, status));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:?}");
}
#[test]
fn protected_fetch_preserves_valid_prefix() {
    let _serial = FORK_TEST.lock().unwrap_or_else(|p| p.into_inner());
    for tier in 0..3 {
        assert_eq!(
            signal::child(|| {
                unsafe {
                    signal::install();
                }
                let vm = vm(tier);
                let guard = BASE + (page() * 4) as u64;
                let start = guard - 5;
                vm.write_bytes(start, &[0x48, 0xff, 0xc0, 0x48, 0xb8])
                    .unwrap();
                vm.write_bytes(guard, &[0; 8]).unwrap();
                vm.write_bytes(guard + 8, &[0xf4]).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, start);
                signal::HANDLE.store(cpu.fault_handle().as_usize(), Ordering::Relaxed);
                protect(&vm, true);
                assert!(
                    matches!(cpu.run(&vm,None),Exit::UnmappedMemory { addr, access: x86jit_core::AccessKind::Execute, .. } if addr == guard)
                );
                assert_eq!(cpu.reg(Reg::Rax), 1);
                assert_eq!(cpu.reg(Reg::Rip), guard - 2);
                protect(&vm, false);
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            }),
            0
        );
    }
}
#[test]
fn direct_memory_without_active_vcpu_runs_drops_and_rejects_unrelated_pc() {
    let _serial = FORK_TEST.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(
        signal::child(|| {
            unsafe {
                signal::install();
            }
            let vm = vm(0);
            let guard = BASE + (page() * 4) as u64;
            let seed: Vec<u8> = (0..32).map(|i| 0xa0 + i).collect();
            vm.write_bytes(guard - 32, &seed).unwrap();
            protect(&vm, true);
            signal::HANDLE.store(0, Ordering::Relaxed);
            // A wide copy starting at each byte offset retries the faulting word
            // bytewise, preserving exactly the readable prefix.
            for prefix in 1..16 {
                let snapshot = vm.mem.code_slice(guard - prefix, 32).unwrap();
                assert_eq!(snapshot, seed[32 - prefix as usize..]);
                let mut copied = [0x55; 32];
                assert!(vm.mem.read_bytes(guard - prefix, &mut copied).is_err());
                assert_eq!(&copied[..prefix as usize], &seed[32 - prefix as usize..]);
            }
            struct Dropped<'a>(&'a std::cell::Cell<bool>);
            impl Drop for Dropped<'_> {
                fn drop(&mut self) {
                    self.0.set(true);
                }
            }
            let dropped = std::cell::Cell::new(false);
            let result = {
                let _guard = Dropped(&dropped);
                vm.mem.read(guard, 8)
            };
            assert!(
                matches!(result,Err(x86jit_core::memory::MemTrap::HostFault) if x86jit_core::MemTrap::HostFault.fault_addr(0) == guard)
            );
            assert!(dropped.get());
            let inactive = vm.new_vcpu();
            signal::HANDLE.store(inactive.fault_handle().as_usize(), Ordering::Relaxed);
            assert!(vm.mem.write(guard, 1, 8).is_err());
        }),
        0
    );
    let status = signal::child(|| {
        unsafe {
            signal::install();
        }
        let vm = vm(0);
        protect(&vm, true);
        // SAFETY: deliberate fault outside the fixup table in a disposable child.
        unsafe {
            core::ptr::read_volatile(vm.mem.host_base().add(page() * 4));
        }
    });
    assert!(libc::WIFSIGNALED(status));
    assert!([libc::SIGSEGV, libc::SIGBUS].contains(&libc::WTERMSIG(status)));
}

#[test]
fn routed_scalars_recover_and_remain_single_copy_atomic() {
    let _serial = FORK_TEST.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(
        signal::child(|| {
            unsafe {
                signal::install();
            }
            let vm = std::sync::Arc::new(vm(0));
            let guard = BASE + (page() * 4) as u64;
            let seed: Vec<u8> = (0..32).map(|i| 0xa0 + i).collect();
            vm.write_bytes(guard - 32, &seed).unwrap();
            protect(&vm, true);
            signal::HANDLE.store(0, Ordering::Relaxed);
            for size in [1, 2, 4, 8] {
                assert!(
                    matches!(vm.mem.read(guard,size),Err(x86jit_core::memory::MemTrap::HostFault) if x86jit_core::MemTrap::HostFault.fault_addr(0) == guard)
                );
                assert!(
                    matches!(vm.mem.write(guard,1,size),Err(x86jit_core::memory::MemTrap::HostFault) if x86jit_core::MemTrap::HostFault.fault_addr(0) == guard)
                );
            }
            protect(&vm, false);
            for size in [1, 2, 4, 8] {
                let mask = u64::MAX >> (64 - size * 8);
                let first = 0x5555_5555_5555_5555 & mask;
                let second = 0xaaaa_aaaa_aaaa_aaaa & mask;
                vm.mem.write(guard, first, size).unwrap();
                assert_eq!(
                    vm.mem.atomic_cas(guard, first, second, size).unwrap(),
                    first
                );
                assert_eq!(vm.mem.atomic_cas(guard, first, 0, size).unwrap(), second);
                assert_eq!(vm.mem.read(guard, size).unwrap(), second);
                assert_eq!(
                    vm.mem
                        .atomic_rmw(guard, first, size, x86jit_core::RmwOp::Xor)
                        .unwrap(),
                    second
                );
                assert_eq!(vm.mem.read(guard, size).unwrap(), mask);
                // SDM Vol 2A, CMPXCHG, Operation: a failed compare still requires
                // write access, even though the current value remains unchanged.
                // SAFETY: this page belongs to the child's mapping and stays readable.
                unsafe {
                    assert_eq!(
                        libc::mprotect(
                            vm.mem.host_base().add(page() * 4).cast_mut().cast(),
                            page(),
                            libc::PROT_READ
                        ),
                        0
                    );
                }
                assert!(
                    matches!(vm.mem.atomic_cas(guard, 0, first, size), Err(x86jit_core::memory::MemTrap::HostFault) if x86jit_core::MemTrap::HostFault.fault_addr(0) == guard)
                );
                assert_eq!(vm.mem.read(guard, size).unwrap(), mask);
                protect(&vm, false);
            }
        }),
        0
    );
    // Thread creation stays in the parent; forking a multithreaded test harness
    // and then initializing its thread runtime can inherit locked runtime state.
    let vm = std::sync::Arc::new(vm(0));
    let guard = BASE + (page() * 4) as u64;
    for size in [1, 2, 4, 8] {
        let mask = u64::MAX >> (64 - size * 8);
        let first = 0x5555_5555_5555_5555 & mask;
        let second = 0xaaaa_aaaa_aaaa_aaaa & mask;
        vm.mem.write(guard, first, size).unwrap();
        let writer = std::sync::Arc::clone(&vm);
        let start = std::sync::Arc::new(std::sync::Barrier::new(2));
        let start_writer = std::sync::Arc::clone(&start);
        let thread = std::thread::spawn(move || {
            start_writer.wait();
            for _ in 0..100_000 {
                writer.mem.write(guard, second, size).unwrap();
                writer.mem.write(guard, first, size).unwrap();
            }
        });
        start.wait();
        for _ in 0..100_000 {
            let value = vm.mem.read(guard, size).unwrap();
            assert!(
                value == first || value == second,
                "torn {size}-byte scalar: {value:x}"
            );
        }
        thread.join().unwrap();
    }
}

#[test]
fn lane_results_and_vector_stores_commit_after_all_accesses() {
    let _serial = FORK_TEST.lock().unwrap_or_else(|p| p.into_inner());
    let mut failures = Vec::new();
    for routed in [false, true] {
        for tier in 0..3 {
            if tier == 0 && !routed {
                continue;
            }
            for kind in [
                "dpps",
                "logic",
                "popcnt",
                "blendv",
                "blendi",
                "round",
                "ternlog",
                "packed",
                "psign",
                "pmadd",
                "kstore",
                "kstore8",
                "narrow_store",
                "masked_store",
                "vector128_store",
                "vector_store",
                "extract_store",
            ] {
                if tier != 0 && matches!(kind, "vector128_store" | "vector_store" | "extract_store")
                {
                    // Compiled split stores remain under xj-obx.8.
                    continue;
                }
                for readonly in [false, true] {
                    if !routed
                        && matches!(kind, "masked_store" | "kstore" | "kstore8" | "narrow_store")
                    {
                        // Default Rust helpers require host-accessible RAM.
                        continue;
                    }
                    // Masked/narrow stores use Rust helpers on every tier.
                    if readonly
                        && !matches!(
                            kind,
                            "kstore"
                                | "kstore8"
                                | "narrow_store"
                                | "masked_store"
                                | "vector128_store"
                                | "vector_store"
                                | "extract_store"
                        )
                    {
                        continue;
                    }
                    let status = signal::child(|| {
                        unsafe {
                            signal::install();
                        }
                        transfer_mode_case(kind, tier, true, readonly, routed);
                    });
                    if status != 0 {
                        failures.push((kind, tier, readonly, routed, status));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:?}");
}
#[test]
fn default_fetch_stops_before_an_unneeded_protected_page() {
    let _serial = FORK_TEST.lock().unwrap_or_else(|p| p.into_inner());
    for tier in 0..3 {
        assert_eq!(
            signal::child(|| {
                let mut vm = vm(tier);
                vm.mem.set_host_fault_recovery(false);
                let guard = BASE + (page() * 4) as u64;
                vm.write_bytes(guard - 1, &[0xf4]).unwrap();
                protect(&vm, true);
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, guard - 1);
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                cpu.set_reg(Reg::Rip, guard - 1);
                assert!(matches!(
                    cpu.step_instruction(&vm),
                    x86jit_core::StepResult::Exit(Exit::Hlt)
                ));
            }),
            0
        );
    }
}
#[test]
fn default_x87_fault_restores_the_entry_status() {
    let _serial = FORK_TEST.lock().unwrap_or_else(|p| p.into_inner());
    for tier in 0..3 {
        let mut vm = Vm::with_backend(
            VmConfig::flat(0x10000),
            if tier == 0 {
                Box::new(InterpreterBackend) as Box<dyn Backend>
            } else {
                Box::new(Compiled {
                    jit: JitBackend::new(),
                    region: tier == 2,
                })
            },
        );
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, &[0xdd, 0x1f, 0xf4]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.set_reg(Reg::Rdi, 0xffff);
        assert!(matches!(cpu.run(&vm, None), Exit::UnmappedMemory { .. }));
        assert_eq!(cpu.cpu.fpu_sw, 0);
    }
}

#[test]
fn compiled_split_stores_resume_by_rewriting_completed_bytes() {
    let _serial = FORK_TEST.lock().unwrap_or_else(|p| p.into_inner());
    for routed in [false, true] {
        for tier in 1..3 {
            assert_eq!(
                signal::child(|| {
                    unsafe {
                        signal::install();
                    }
                    transfer_mode_case("vector_store", tier, true, false, routed);
                }),
                0
            );
        }
    }
}

// The 16-byte backing offset puts the 4 KiB guest-region edge inside a
// host page on both 4 KiB and 16 KiB hosts. Region gaps stay host-accessible.
fn masked_store_region_gap_case(kind: &str, offset: u64, routed: bool, sparse: bool) {
    let len = 0x4000;
    // SAFETY: HostRam owns the shifted mapping; its destructor recovers
    // the mmap base and releases the complete allocation exactly once.
    let mut vm = unsafe {
        let allocation = len + page();
        let base = libc::mmap(
            std::ptr::null_mut(),
            allocation,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        assert_ne!(base, libc::MAP_FAILED);
        Vm::with_backend_host_ram(
            VmConfig::flat(len as u64),
            Box::new(InterpreterBackend),
            HostRam {
                ptr: base.cast::<u8>().add(16),
                len,
                guest_base: 0,
                protect: None,
                dtor: Box::new(move |p, _| {
                    libc::munmap(p.sub(16).cast(), allocation);
                }),
            },
        )
        .unwrap()
    };
    vm.mem.set_host_fault_recovery(routed);
    vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x3000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
    let address = 0x2000 - offset;
    let host = vm.mem.host_base() as usize;
    assert_eq!((host + address as usize) / page(), (host + 0x2000) / page());
    let mut a = CodeAssembler::new(64).unwrap();
    match kind {
        "vmaskmovps" => a.vmaskmovps(ymmword_ptr(rdi), ymm1, ymm0).unwrap(),
        "vmaskmovpd" => a.vmaskmovpd(ymmword_ptr(rdi), ymm1, ymm0).unwrap(),
        "kstore" => a.vmovdqu32(ymmword_ptr(rdi).k1(), ymm0).unwrap(),
        "kstore64" => a.vmovdqu64(ymmword_ptr(rdi).k1(), ymm0).unwrap(),
        "kstore8" => a.vmovdqu8(ymmword_ptr(rdi).k1(), ymm0).unwrap(),
        "narrow_store" => a.vpmovqd(xmmword_ptr(rdi), ymm0).unwrap(),
        _ => unreachable!(),
    }
    a.hlt().unwrap();
    vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
        .unwrap();
    vm.write_bytes(0x1ff0, &[0x53; 16]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x1000);
    cpu.set_reg(Reg::Rdi, address);
    cpu.cpu
        .set_vec(0, [0x4000_0000_4000_0000_4000_0000_4000_0000; 4], 64);
    cpu.cpu.set_vec(
        1,
        if sparse {
            [u128::MAX, 0, 0, 0]
        } else {
            [u128::MAX; 4]
        },
        64,
    );
    cpu.cpu.kmask[1] = if sparse {
        if kind == "kstore8" {
            0xffff
        } else {
            0x0f
        }
    } else {
        u64::MAX
    };
    let before = cpu.cpu.clone();
    let exit = cpu.run(&vm, None);
    if sparse {
        assert!(
            matches!(exit, Exit::Hlt),
            "{kind} routed={routed}: {exit:?}"
        );
        let mut bytes = [0; 16];
        vm.read_bytes(0x1ff0, &mut bytes).unwrap();
        assert_eq!(bytes, before.xmm[0].to_le_bytes(), "{kind} routed={routed}");
        return;
    }
    assert!(
        matches!(
            exit,
            Exit::UnmappedMemory {
                addr: 0x2000,
                access: x86jit_core::AccessKind::Write,
                ..
            }
        ),
        "{kind} routed={routed}: {exit:?}"
    );
    same_cpu(&cpu.cpu, &before);
    let mut bytes = [0; 16];
    vm.read_bytes(0x1ff0, &mut bytes).unwrap();
    assert_eq!(bytes, [0x53; 16], "{kind} routed={routed}");
}

#[test]
fn routed_vmaskmov_checks_regions_inside_a_host_page() {
    let _serial = FORK_TEST.lock().unwrap_or_else(|p| p.into_inner());
    for routed in [false, true] {
        masked_store_region_gap_case("vmaskmovps", 8, routed, false);
    }
}

#[test]
fn routed_kmasked_and_narrow_stores_check_every_selected_region() {
    let _serial = FORK_TEST.lock().unwrap_or_else(|p| p.into_inner());
    for kind in ["kstore", "kstore8", "narrow_store"] {
        masked_store_region_gap_case(
            kind,
            if kind == "narrow_store" { 2 } else { 8 },
            true,
            false,
        );
    }
}

#[test]
fn routed_masked_store_region_fault_names_the_first_unmapped_byte() {
    let _serial = FORK_TEST.lock().unwrap_or_else(|p| p.into_inner());
    for routed in [false, true] {
        masked_store_region_gap_case("vmaskmovpd", 4, routed, false);
    }
    masked_store_region_gap_case("kstore64", 4, true, false);
}

#[test]
fn masked_off_region_gaps_do_not_fault_in_either_mode() {
    let _serial = FORK_TEST.lock().unwrap_or_else(|p| p.into_inner());
    for routed in [false, true] {
        for kind in ["vmaskmovps", "kstore", "kstore8"] {
            masked_store_region_gap_case(kind, 16, routed, true);
        }
    }
}

#[test]
fn routed_region_preflight_faults_before_selected_mmio_writes() {
    let _serial = FORK_TEST.lock().unwrap_or_else(|p| p.into_inner());
    for routed in [false, true] {
        for kind in ["kstore", "kstore8", "narrow_store"] {
            let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(InterpreterBackend));
            vm.mem.set_host_fault_recovery(routed);
            vm.map(0, 0x1ff8, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x1ff8, 8, Prot::RW, RegionKind::Trap).unwrap();
            vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
            let mut a = CodeAssembler::new(64).unwrap();
            match kind {
                "kstore" => a.vmovdqu32(ymmword_ptr(rdi).k1(), ymm0).unwrap(),
                "kstore8" => a.vmovdqu8(ymmword_ptr(rdi).k1(), ymm0).unwrap(),
                "narrow_store" => a.vpmovqd(xmmword_ptr(rdi), ymm0).unwrap(),
                _ => unreachable!(),
            }
            a.hlt().unwrap();
            vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
                .unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rdi, 0x1ff8);
            cpu.cpu.set_vec(0, [u128::MAX; 4], 64);
            cpu.cpu.kmask[1] = u64::MAX;
            let elem = if kind == "kstore8" { 1 } else { 4 };
            let mut writes = 0;
            loop {
                match cpu.run(&vm, None) {
                    Exit::MmioWrite {
                        addr, size, value, ..
                    } => {
                        assert!(!routed, "{kind}: routed mode issued a device write");
                        assert_eq!(addr, 0x1ff8 + writes * elem);
                        assert_eq!(u64::from(size), elem);
                        let mask = u64::MAX >> (64 - elem * 8);
                        assert_eq!(value & mask, mask);
                        writes += 1;
                        assert!(writes <= 8 / elem);
                        cpu.complete_mmio_write();
                    }
                    Exit::UnmappedMemory {
                        addr: 0x2000,
                        access: x86jit_core::AccessKind::Write,
                        ..
                    } => break,
                    exit => panic!("{kind} routed={routed}: {exit:?}"),
                }
            }
            assert_eq!(writes, if routed { 0 } else { 8 / elem });
            assert_eq!(cpu.reg(Reg::Rip), 0x1000);
        }
    }
}
