//! Pure stores: SDM Vol 3A §11.1.1; Vol 2C, VMASKMOV/VPMASKMOV,
//! Operation; Vol 1 §§13.7, 13.9, 13.10 (XSAVE header reads vs XSAVEC).
#![cfg(all(
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
use iced_x86::code_asm::*;
use std::sync::atomic::Ordering;
use x86jit_core::{
    Backend, CachedBlock, Exit, HostRam, InterpreterBackend, IrBlock, MemConsistency, Prot, Reg,
    RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;
#[path = "../../x86jit-cranelift/src/unwind_test_signal.rs"]
#[allow(dead_code)]
mod signal;

const BASE: u64 = 0x2_0000_0000;
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
struct Compiled(JitBackend);
impl Backend for Compiled {
    fn materialize_with_split_stores(
        &self,
        ir: &IrBlock,
        consistency: MemConsistency,
        mmio: Option<(u64, u64)>,
        base: u64,
        mxcsr: u32,
        precise: bool,
    ) -> CachedBlock {
        let block =
            self.0
                .materialize_with_split_stores(ir, consistency, mmio, base, mxcsr, precise);
        assert!(
            matches!(block, CachedBlock::Compiled { .. }),
            "store must compile"
        );
        block
    }
    fn materialize(
        &self,
        ir: &IrBlock,
        tier: MemConsistency,
        mmio: Option<(u64, u64)>,
        base: u64,
    ) -> CachedBlock {
        let block = self.0.materialize(ir, tier, mmio, base);
        assert!(matches!(block, CachedBlock::Compiled { .. }));
        block
    }
}
fn page() -> usize {
    // SAFETY: sysconf has no pointer arguments.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}
fn vm(tier: MemConsistency, compiled: bool) -> Vm {
    let len = page() * 8;
    let mut config = VmConfig::flat(BASE + len as u64);
    config.consistency = tier;
    let backend: Box<dyn Backend> = if compiled {
        Box::new(Compiled(JitBackend::new()))
    } else {
        Box::new(InterpreterBackend)
    };
    // SAFETY: HostRam owns this complete mapping and releases it once.
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
            config,
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
        vm.map(BASE, len, Prot::RWX, RegionKind::Ram).unwrap();
        vm.mem.set_host_fault_recovery(true);
        vm.set_tier_up_after(None);
        vm
    }
}
fn program(name: &str) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    match name {
        "mov8" => a.mov(byte_ptr(rdi), al).unwrap(),
        "mov16" => a.mov(word_ptr(rdi), ax).unwrap(),
        "mov32" => a.mov(dword_ptr(rdi), eax).unwrap(),
        "mov64" => a.mov(qword_ptr(rdi), rax).unwrap(),
        "movq" => a.movq(qword_ptr(rdi), xmm0).unwrap(),
        "movups" => a.movups(xmmword_ptr(rdi), xmm0).unwrap(),
        "movaps" => a.movaps(xmmword_ptr(rdi), xmm0).unwrap(),
        "vmovdqu128" => a.vmovdqu(xmmword_ptr(rdi), xmm0).unwrap(),
        "vmovdqu256" => a.vmovdqu(ymmword_ptr(rdi), ymm0).unwrap(),
        "vmovdqu512" => a.vmovdqu32(zmmword_ptr(rdi), zmm0).unwrap(),
        "movnti" => a.movnti(qword_ptr(rdi), rax).unwrap(),
        "movntdq" => a.movntdq(xmmword_ptr(rdi), xmm0).unwrap(),
        "movntps" => a.movntps(xmmword_ptr(rdi), xmm0).unwrap(),
        "movntpd" => a.movntpd(xmmword_ptr(rdi), xmm0).unwrap(),
        "vmovntdq" => a.vmovntdq(ymmword_ptr(rdi), ymm0).unwrap(),
        "stosb" => a.rep().stosb().unwrap(),
        "stosq" => a.rep().stosq().unwrap(),
        "vmaskmovps" => a.vmaskmovps(ymmword_ptr(rdi), ymm1, ymm0).unwrap(),
        "vpmaskmovq" => a.vpmaskmovq(ymmword_ptr(rdi), ymm1, ymm0).unwrap(),
        "masked8" => a.vmovdqu8(zmmword_ptr(rdi).k1(), zmm0).unwrap(),
        "masked32" => a.vmovdqu32(zmmword_ptr(rdi).k1(), zmm0).unwrap(),
        "masked64" => a.vmovdqu64(zmmword_ptr(rdi).k1(), zmm0).unwrap(),
        "narrow" => a.vpmovdb(xmmword_ptr(rdi), zmm0).unwrap(),
        "fxsave" => a.fxsave64(ptr(rdi)).unwrap(),
        "xsavec" => a.xsavec64(ptr(rdi)).unwrap(),
        "xsave" => a.xsave64(ptr(rdi)).unwrap(),
        _ => unreachable!(),
    }
    a.hlt().unwrap();
    a.assemble(BASE + 0x1000).unwrap()
}
const CASES: &[&str] = &[
    "mov8",
    "mov16",
    "mov32",
    "mov64",
    "movq",
    "movups",
    "movaps",
    "vmovdqu128",
    "vmovdqu256",
    "vmovdqu512",
    "movnti",
    "movntdq",
    "movntps",
    "movntpd",
    "vmovntdq",
    "stosb",
    "stosq",
    "vmaskmovps",
    "vpmaskmovq",
    "masked8",
    "masked32",
    "masked64",
    "narrow",
    "fxsave",
    "xsavec",
];
fn protect(vm: &Vm, prot: i32) {
    // SAFETY: page-aligned subrange of the live allocation.
    unsafe {
        assert_eq!(
            libc::mprotect(
                vm.mem.host_base().add(page() * 4).cast_mut().cast(),
                page() * 2,
                prot
            ),
            0
        );
    }
}
fn execute(vm: &Vm, address: u64) -> Exit {
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, BASE + 0x1000);
    cpu.set_reg(Reg::Rdi, address);
    cpu.set_reg(Reg::Rax, 7);
    cpu.set_reg(Reg::Rcx, 3);
    cpu.cpu
        .set_vec(0, [0x1234_5678_9abc_def0_fedc_ba98_7654_3210; 4], 64);
    cpu.cpu
        .set_vec(1, [0x8000_0000_8000_0000_8000_0000_8000_0000; 4], 64);
    cpu.cpu.kmask[1] = 0x5555_5555_5555_5555;
    signal::HANDLE.store(cpu.fault_handle().as_usize(), Ordering::Relaxed);
    cpu.run(vm, None)
}
#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn accepted_probe_then_compiled_write_fault_keeps_latest_raw_evidence() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(
        signal::child(|| {
            // SAFETY: the child owns the file and replacement mapping. The
            // empty backing deliberately faults despite writable VM permissions.
            unsafe {
                signal::install();
                let mut vm = vm(MemConsistency::Fast, true);
                vm.set_precise_split_stores(true);
                vm.write_bytes(BASE + 0x1000, &program("movups")).unwrap();
                let file = libc::tmpfile();
                assert!(!file.is_null());
                let host = vm.mem.host_base().add(page() * 5).cast_mut();
                assert_eq!(
                    libc::mmap(
                        host.cast(),
                        page(),
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED | libc::MAP_FIXED,
                        libc::fileno(file),
                        0,
                    ),
                    host.cast()
                );
                libc::fclose(file);
                signal::FAULTS.store(0, Ordering::Relaxed);
                let address = BASE + (page() * 5 - 8) as u64;
                let exit = execute(&vm, address);
                assert!(matches!(exit, Exit::UnmappedMemory { .. }), "{exit:?}");
                assert!(signal::FAULTS.load(Ordering::Relaxed) >= 2);
                assert_ne!(
                    signal::WIDTH.load(Ordering::Relaxed),
                    0,
                    "compiled write landing"
                );
                let raw = exit.fault_metadata().unwrap().raw_host_fault().unwrap();
                let latest = signal::LAST_RAW_SYNDROME.load(Ordering::Relaxed);
                assert_eq!(raw.esr.or(raw.error_code).unwrap_or(0), latest, "{raw:?}");
            }
        }),
        0
    );
}

#[test]
fn host_write_only_mapping_capability() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(
        signal::child(|| {
            // SAFETY: handlers are confined to a disposable child.
            unsafe {
                signal::install();
            }
            let vm = vm(MemConsistency::Fast, false);
            let address = BASE + (page() * 4) as u64;
            protect(&vm, libc::PROT_WRITE);
            vm.mem.write(address, 0x5a, 1).unwrap();
            let read = vm.mem.read(address, 1);
            eprintln!(
                "MEASURED: PROT_WRITE permits native read = {}",
                read.is_ok()
            );
            if let Ok(value) = read {
                assert_eq!(value, 0x5a);
            }
        }),
        0
    );
}
#[test]
fn pure_stores_to_host_write_only_pages() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let status = signal::child(|| {
        // SAFETY: extend only this disposable child's timeout for the matrix.
        unsafe {
            libc::alarm(120);
        }
        // SAFETY: handler installation and mprotect stay in the child.
        unsafe {
            signal::install();
        }
        for tier in [
            MemConsistency::Fast,
            MemConsistency::AcqRel,
            MemConsistency::FullTso,
        ] {
            for compiled in [false, true] {
                for precise in [false, true] {
                    for &name in CASES {
                        for crossing in [false, true] {
                            let mut vm = vm(tier, compiled);
                            vm.set_precise_split_stores(precise);
                            let code = program(name);
                            vm.write_bytes(BASE + 0x1000, &code).unwrap();
                            let reference = self::vm(tier, false);
                            reference.write_bytes(BASE + 0x1000, &code).unwrap();
                            let offset = if !crossing {
                                if name == "xsavec" {
                                    64
                                } else {
                                    32
                                }
                            } else if matches!(name, "fxsave" | "xsavec") {
                                page() - 256
                            } else if matches!(
                                name,
                                "movaps" | "movntdq" | "movntps" | "movntpd" | "vmovntdq"
                            ) {
                                page() - 32
                            } else {
                                page() - 3
                            };
                            let address = BASE + (page() * 4 + offset) as u64;
                            assert!(
                                matches!(execute(&reference, address), Exit::Hlt),
                                "reference {name}"
                            );
                            protect(&vm, libc::PROT_WRITE);
                            let exit = execute(&vm, address);
                            protect(&vm, libc::PROT_READ | libc::PROT_WRITE);
                            assert!(matches!(exit, Exit::Hlt), "{name} {tier:?} compiled={compiled} precise={precise} crossing={crossing}: {exit:?}");
                            let mut expected = vec![0; page() * 2];
                            let mut actual = expected.clone();
                            reference
                                .read_bytes(BASE + (page() * 4) as u64, &mut expected)
                                .unwrap();
                            vm.read_bytes(BASE + (page() * 4) as u64, &mut actual)
                                .unwrap();
                            assert_eq!(actual, expected, "{name} {tier:?} compiled={compiled} precise={precise} crossing={crossing}");
                        }
                    }
                }
            }
        }
    });
    assert_eq!(status, 0);
}

#[test]
fn crossing_store_unmapped_second_page_preserves_original_fault_and_prefix() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(
        signal::child(|| {
            // SAFETY: handlers and mapping changes are confined to the disposable child.
            unsafe {
                signal::install();
            }
            for tier in [
                MemConsistency::Fast,
                MemConsistency::AcqRel,
                MemConsistency::FullTso,
            ] {
                for compiled in [false, true] {
                    for name in [
                        "mov64",
                        "movups",
                        "vmovdqu256",
                        "vmovdqu512",
                        "masked32",
                        "narrow",
                        "fxsave",
                        "xsavec",
                    ] {
                        let mut vm = vm(tier, compiled);
                        vm.set_precise_split_stores(true);
                        vm.write_bytes(BASE + 0x1000, &program(name)).unwrap();
                        let guard = BASE + (page() * 5) as u64;
                        let prefix = if matches!(name, "fxsave" | "xsavec") {
                            256
                        } else {
                            3
                        };
                        let address = guard - prefix;
                        vm.write_bytes(address, &vec![0x55; prefix as usize])
                            .unwrap();
                        // SAFETY: release exactly one page inside the owned mapping.
                        // The final munmap of the whole allocation also accepts this hole.
                        unsafe {
                            assert_eq!(
                                libc::munmap(
                                    vm.mem.host_base().add(page() * 5).cast_mut().cast(),
                                    page()
                                ),
                                0
                            );
                        }
                        let exit = execute(&vm, address);
                        let metadata = exit.fault_metadata().expect("host fault metadata");
                        assert_eq!(
                            metadata.origin(),
                            x86jit_core::FaultOrigin::HostAccess,
                            "{name}: {exit:?}"
                        );
                        assert_eq!(metadata.saved_rip(), BASE + 0x1000, "{name}: {exit:?}");
                        assert_eq!(
                            metadata.linear_address(),
                            x86jit_core::Knowledge::Known(x86jit_core::FaultAddress::new(
                                guard,
                                x86jit_core::AddressPrecision::ExactByte,
                                x86jit_core::AddressSpace::Linear
                            )),
                            "{name}: {exit:?}"
                        );
                        let mut actual = vec![0; prefix as usize];
                        vm.read_bytes(address, &mut actual).unwrap();
                        assert_eq!(
                            actual,
                            vec![0x55; prefix as usize],
                            "{name} {tier:?} compiled={compiled}: partial commit"
                        );
                    }
                }
            }
        }),
        0
    );
}

#[test]
fn compiled_conversion_store_respects_both_policies() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(
        signal::child(|| {
            // SAFETY: signal handlers and protections are confined to this child.
            unsafe {
                signal::install();
            }
            for tier in [
                MemConsistency::Fast,
                MemConsistency::AcqRel,
                MemConsistency::FullTso,
            ] {
                for recovery in [false, true] {
                    for precise in [false, true] {
                        for crossing in [false, true] {
                            // SDM Vol 1 §14.4: F16C packs single-precision lanes into half precision.
                            let code = [0xc4, 0xe3, 0x7d, 0x1d, 0x07, 0, 0xf4];
                            let mut vm = vm(tier, true);
                            vm.mem.set_host_fault_recovery(recovery);
                            vm.set_precise_split_stores(precise);
                            vm.write_bytes(BASE + 0x1000, &code).unwrap();
                            let reference = self::vm(tier, false);
                            reference.write_bytes(BASE + 0x1000, &code).unwrap();
                            let address =
                                BASE + (page() * 4 + if crossing { page() - 3 } else { 32 }) as u64;
                            assert!(matches!(execute(&reference, address), Exit::Hlt));
                            protect(&vm, libc::PROT_WRITE);
                            let exit = execute(&vm, address);
                            protect(&vm, libc::PROT_READ | libc::PROT_WRITE);
                            assert!(matches!(exit, Exit::Hlt), "{tier:?} recovery={recovery} precise={precise} crossing={crossing}: {exit:?}");
                            let mut actual = [0; 16];
                            let mut expected = [0; 16];
                            vm.read_bytes(address, &mut actual).unwrap();
                            reference.read_bytes(address, &mut expected).unwrap();
                            assert_eq!(actual, expected);
                        }
                    }
                }
            }
        }),
        0
    );
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn interpreted_crossing_store_sigbus_preserves_prefix() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(
        signal::child(|| {
            // SAFETY: handlers and the empty-file replacement mapping stay in the child.
            unsafe {
                signal::install();
                for (name, width) in [
                    ("mov64", 8),
                    ("movups", 16),
                    ("vmovdqu256", 32),
                    ("vmovdqu512", 64),
                ] {
                    let vm = vm(MemConsistency::Fast, false);
                    vm.write_bytes(BASE + 0x1000, &program(name)).unwrap();
                    let prefix = width / 2;
                    let address = BASE + (page() * 5 - prefix) as u64;
                    vm.write_bytes(address, &vec![0x55; prefix]).unwrap();
                    let file = libc::tmpfile();
                    assert!(!file.is_null());
                    let host = vm.mem.host_base().add(page() * 5).cast_mut();
                    assert_eq!(
                        libc::mmap(
                            host.cast(),
                            page(),
                            libc::PROT_READ | libc::PROT_WRITE,
                            libc::MAP_SHARED | libc::MAP_FIXED,
                            libc::fileno(file),
                            0,
                        ),
                        host.cast()
                    );
                    libc::fclose(file);
                    let exit = execute(&vm, address);
                    assert!(
                        matches!(exit, Exit::UnmappedMemory { .. }),
                        "{name}: {exit:?}"
                    );
                    let mut actual = vec![0; prefix];
                    vm.read_bytes(address, &mut actual).unwrap();
                    assert_eq!(
                        actual,
                        vec![0x55; prefix],
                        "{name}: SIGBUS committed a store prefix"
                    );
                }
            }
        }),
        0
    );
}

#[test]
fn compiled_conversion_fault_preserves_destination_without_precise_split_stores() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(
        signal::child(|| {
            // SAFETY: handlers and page protections stay in the child.
            unsafe { signal::install() };
            for tier in [
                MemConsistency::Fast,
                MemConsistency::AcqRel,
                MemConsistency::FullTso,
            ] {
                let mut vm = vm(tier, true);
                vm.set_precise_split_stores(false);
                // SDM Vol 1 §14.4: vcvtps2ph [rdi], ymm0, 0 stores sixteen bytes.
                vm.write_bytes(BASE + 0x1000, &[0xc4, 0xe3, 0x7d, 0x1d, 0x07, 0, 0xf4])
                    .unwrap();
                let address = BASE + (page() * 5 - 8) as u64;
                vm.write_bytes(address, &[0x55; 16]).unwrap();
                // SAFETY: one complete page within the live allocation.
                unsafe {
                    assert_eq!(
                        libc::mprotect(
                            vm.mem.host_base().add(page() * 5).cast_mut().cast(),
                            page(),
                            libc::PROT_NONE
                        ),
                        0
                    );
                }
                let exit = execute(&vm, address);
                assert!(
                    matches!(exit, Exit::UnmappedMemory { .. }),
                    "{tier:?}: {exit:?}"
                );
                // SAFETY: restore the page before inspecting all destination bytes.
                unsafe {
                    assert_eq!(
                        libc::mprotect(
                            vm.mem.host_base().add(page() * 5).cast_mut().cast(),
                            page(),
                            libc::PROT_READ | libc::PROT_WRITE
                        ),
                        0
                    );
                }
                let mut actual = [0; 16];
                vm.read_bytes(address, &mut actual).unwrap();
                assert_eq!(
                    actual, [0x55; 16],
                    "{tier:?}: conversion committed a prefix"
                );
            }
        }),
        0
    );
}

#[test]
fn mixed_rom_ram_crossing_store_fault_preserves_destination() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(
        signal::child(|| {
            // SAFETY: handlers and page protections stay in the child.
            unsafe { signal::install() };
            // SDM Vol 3A §§7.5–7.6: a fault leaves the destination unchanged.
            for tier in [
                MemConsistency::Fast,
                MemConsistency::AcqRel,
                MemConsistency::FullTso,
            ] {
                for compiled in [false, true] {
                    for name in ["vmovdqu256", "masked32"] {
                        for rom_offset in [-8, 0, 8] {
                            let mut vm = vm(tier, compiled);
                            vm.set_precise_split_stores(true);
                            let boundary = BASE + (page() * 5) as u64;
                            let address = boundary - 16;
                            // Seed backing before tagging the ROM bytes.
                            vm.write_bytes(address, &[0x55; 64]).unwrap();
                            vm.unmap(BASE, page() * 8).unwrap();
                            let rom = boundary.wrapping_add_signed(rom_offset);
                            vm.map(BASE, (rom - BASE) as usize, Prot::RWX, RegionKind::Ram)
                                .unwrap();
                            vm.map(rom, 8, Prot::RW, RegionKind::Rom).unwrap();
                            vm.map(
                                rom + 8,
                                page() * 8 - (rom + 8 - BASE) as usize,
                                Prot::RW,
                                RegionKind::Ram,
                            )
                            .unwrap();
                            vm.write_bytes(BASE + 0x1000, &program(name)).unwrap();
                            // SAFETY: one complete page within the live allocation.
                            let host = unsafe { vm.mem.host_base().add(page() * 5) };
                            unsafe {
                                assert_eq!(
                                    libc::mprotect(host.cast_mut().cast(), page(), libc::PROT_NONE),
                                    0
                                );
                            }
                            let exit = execute(&vm, address);
                            signal::HANDLE.store(0, Ordering::Relaxed);
                            let preflight = x86jit_core::interp::StrMem::preflight_store(
                                &vm.mem,
                                address,
                                u64::MAX,
                                8,
                                32,
                            );
                            // SAFETY: restore access before inspecting all destination bytes.
                            unsafe {
                                assert_eq!(
                                    libc::mprotect(
                                        host.cast_mut().cast(),
                                        page(),
                                        libc::PROT_READ | libc::PROT_WRITE
                                    ),
                                    0
                                );
                            }
                            assert!(
                                matches!(exit, Exit::UnmappedMemory { .. }),
                                "{name} {tier:?} compiled={compiled} rom_offset={rom_offset}: {exit:?}"
                            );
                            let metadata = exit.fault_metadata().unwrap();
                            assert_eq!(metadata.origin(), x86jit_core::FaultOrigin::HostAccess);
                            assert_eq!(metadata.saved_rip(), BASE + 0x1000);
                            if !compiled {
                                let fault = boundary + if rom_offset == 0 { 8 } else { 0 };
                                assert_eq!(
                                    metadata.linear_address(),
                                    x86jit_core::Knowledge::Known(x86jit_core::FaultAddress::new(
                                        fault,
                                        x86jit_core::AddressPrecision::ExactByte,
                                        x86jit_core::AddressSpace::Linear
                                    )),
                                    "{name} {tier:?}: {exit:?}"
                                );
                            }
                            let mut actual = [0; 64];
                            for (i, byte) in actual.iter_mut().enumerate() {
                                *byte = vm.mem.read(address + i as u64, 1).unwrap() as u8;
                            }
                            assert_eq!(
                                actual, [0x55; 64],
                                "{name} {tier:?} compiled={compiled} rom_offset={rom_offset}: partial commit"
                            );
                            assert!(
                                matches!(preflight, Err((at, x86jit_core::MemTrap::HostFault))
                                    if at == boundary + if rom_offset == 0 { 8 } else { 0 }),
                                "Memory preflight: {preflight:?}"
                            );
                        }
                    }
                }
            }
        }),
        0
    );
}

#[test]
fn interpreted_single_page_store_respects_rom_ram_splits() {
    for tier in [
        MemConsistency::Fast,
        MemConsistency::AcqRel,
        MemConsistency::FullTso,
    ] {
        for rom_first in [false, true] {
            let mut vm = vm(tier, false);
            let address = BASE + (page() * 4 + 32) as u64;
            vm.write_bytes(address, &[0x55; 32]).unwrap();
            vm.unmap(BASE, page() * 8).unwrap();
            let rom = address + if rom_first { 0 } else { 16 };
            vm.map(BASE, (rom - BASE) as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.map(rom, 16, Prot::RW, RegionKind::Rom).unwrap();
            vm.map(
                rom + 16,
                page() * 8 - (rom + 16 - BASE) as usize,
                Prot::RW,
                RegionKind::Ram,
            )
            .unwrap();
            vm.write_bytes(BASE + 0x1000, &program("vmovdqu256"))
                .unwrap();
            assert!(matches!(execute(&vm, address), Exit::Hlt));
            let mut actual = [0; 32];
            for (i, byte) in actual.iter_mut().enumerate() {
                *byte = vm.mem.read(address + i as u64, 1).unwrap() as u8;
            }
            let stored = 0x1234_5678_9abc_def0_fedc_ba98_7654_3210u128.to_le_bytes();
            let (first, second) = actual.split_at(16);
            let (ram, rom) = if rom_first {
                (second, first)
            } else {
                (first, second)
            };
            assert_eq!(ram, stored, "{tier:?} rom_first={rom_first}");
            assert_eq!(rom, [0x55; 16], "{tier:?} rom_first={rom_first}");
        }
    }
}

#[test]
fn xsave_rom_representative_cannot_hide_protected_ram_on_same_page() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    assert_eq!(
        signal::child(|| {
            // SAFETY: handlers and page protections stay in the child.
            unsafe { signal::install() };
            for compiled in [false, true] {
                for (name, prefix) in [("xsave", 576), ("xsavec", 256)] {
                    let mut vm = vm(MemConsistency::Fast, compiled);
                    vm.unmap(BASE, page() * 8).unwrap();
                    let boundary = BASE + (page() * 5) as u64;
                    vm.map(BASE, page() * 5, Prot::RWX, RegionKind::Ram)
                        .unwrap();
                    vm.map(boundary, 32, Prot::RW, RegionKind::Rom).unwrap();
                    vm.map(boundary + 32, page() * 3 - 32, Prot::RW, RegionKind::Ram)
                        .unwrap();
                    vm.write_bytes(BASE + 0x1000, &program(name)).unwrap();
                    let address = boundary - prefix as u64;
                    vm.write_bytes(address, &vec![0x55; prefix]).unwrap();
                    // SAFETY: protect the page shared by the ROM tag and later RAM bytes.
                    unsafe {
                        assert_eq!(
                            libc::mprotect(
                                vm.mem.host_base().add(page() * 5).cast_mut().cast(),
                                page(),
                                libc::PROT_NONE
                            ),
                            0
                        );
                    }
                    let exit = execute(&vm, address);
                    assert!(
                        matches!(exit, Exit::UnmappedMemory { .. }),
                        "compiled={compiled}: {exit:?}"
                    );
                    let mut actual = vec![0; prefix];
                    vm.read_bytes(address, &mut actual).unwrap();
                    assert_eq!(
                        actual,
                        vec![0x55; prefix],
                        "{name} compiled={compiled}: save committed a prefix"
                    );
                }
            }
        }),
        0
    );
}

#[test]
fn precise_store_bounds_reject_short_backing_and_address_wrap() {
    for span in [32, 128] {
        for address in [0, u64::MAX - 3] {
            if span == 128 && address == 0 {
                continue;
            }
            let mut vm =
                Vm::with_backend(VmConfig::flat(span), Box::new(Compiled(JitBackend::new())));
            vm.set_tier_up_after(None);
            vm.set_precise_split_stores(true);
            vm.map(0, span as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.write_bytes(0, &program("vmovdqu512")).unwrap();
            let mut before = vec![0; span as usize];
            vm.read_bytes(0, &mut before).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rdi, address);
            let exit = cpu.run(&vm, None);
            assert!(
                matches!(exit, Exit::UnmappedMemory { .. }),
                "{span} {address:x}: {exit:?}"
            );
            assert_eq!(
                exit.fault_metadata().unwrap().origin(),
                x86jit_core::FaultOrigin::GuestMap
            );
            let mut after = vec![0; span as usize];
            vm.read_bytes(0, &mut after).unwrap();
            assert_eq!(before, after, "faulting store changed backing");
        }
    }
}
