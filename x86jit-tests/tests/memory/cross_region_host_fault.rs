//! Host-protection faults on a store covering adjacent mapped regions.
//!
//! SDM Vol 3A §§7.5–7.6: a faulting store must be restartable with no destination
//! byte committed. A store whose bytes lie in adjacent RAM (and ROM) regions is
//! checked over its whole span before any byte is written, so it faults at the same
//! byte, with the same exit, as the same store inside one region.
//! Authored fixtures use real host protection; the guest `Prot` stays advisory.
#![cfg(all(
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]

use iced_x86::code_asm::*;
use std::sync::atomic::Ordering;
use x86jit_core::{
    memory::HostRam, AccessKind, Backend, Exit, InterpreterBackend, Prot, Reg, RegionCaps,
    RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;
#[path = "../../../x86jit-cranelift/src/unwind_test_signal.rs"]
#[allow(dead_code)]
mod signal;

const BASE: u64 = 0x1_0000_0000;
const CAPS: RegionCaps = RegionCaps {
    max_blocks: 16,
    max_icount: 256,
};
fn page() -> usize {
    // SAFETY: sysconf has no pointer argument.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// `(start, end, kind)` regions covering the whole mapping.
type Layout = Vec<(u64, u64, RegionKind)>;

/// Eight host pages at `BASE`; the fifth (`guard`) is the one the cases protect.
fn vm(tier: u8, routed: bool, layout: &[(u64, u64, RegionKind)]) -> Vm {
    let len = page() * 8;
    let backend: Box<dyn Backend> = match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        _ => Box::new(JitBackend::with_superblocks(CAPS)),
    };
    // SAFETY: the mapping stays owned by HostRam and is freed exactly once.
    let mut vm = unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        assert_ne!(ptr, libc::MAP_FAILED);
        Vm::with_backend_host_ram(
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
        .unwrap()
    };
    vm.set_tier_up_after(None);
    vm.set_precise_split_stores(true);
    vm.mem.set_host_fault_recovery(routed);
    for &(start, end, kind) in layout {
        vm.map(start, (end - start) as usize, Prot::RWX, kind)
            .unwrap();
    }
    vm
}

fn guard() -> u64 {
    BASE + (page() * 4) as u64
}

fn protect(vm: &Vm, prot: i32) {
    // SAFETY: the page is aligned and belongs to this child's live mapping.
    unsafe {
        assert_eq!(
            libc::mprotect(
                vm.mem.host_base().add(page() * 4).cast_mut().cast(),
                page(),
                prot
            ),
            0
        );
    }
}

/// Run one qword access at `guard - 4` with the guard page protected; return the
/// exit and the eight destination bytes (read back after unprotecting).
fn case(tier: u8, routed: bool, store: bool, layout: &[(u64, u64, RegionKind)]) -> (Exit, Vec<u8>) {
    let (exit, bytes, _, _) = case_at(tier, routed, store, layout, guard() - 4, false);
    (exit, bytes)
}

/// [`case`] with the operand at `at`, after `inc rcx` when `lead` (so the access
/// is not the first instruction of its block); also returns RCX and the
/// retired-instruction count.
fn case_at(
    tier: u8,
    routed: bool,
    store: bool,
    layout: &[(u64, u64, RegionKind)],
    at: u64,
    lead: bool,
) -> (Exit, Vec<u8>, u64, u64) {
    let vm = vm(tier, routed, layout);
    let code_addr = BASE + 0x1000;
    let mut a = CodeAssembler::new(64).unwrap();
    if lead {
        a.inc(rcx).unwrap();
    }
    if store {
        a.mov(qword_ptr(rdi), rax).unwrap();
    } else {
        a.mov(rbx, qword_ptr(rdi)).unwrap();
    }
    a.hlt().unwrap();
    vm.write_bytes(code_addr, &a.assemble(code_addr).unwrap())
        .unwrap();
    for i in 0..8 {
        vm.write_bytes(at + i, &[0x53]).unwrap();
    }
    if tier == 2 {
        let mut vm = vm;
        vm.test_support_force_region(code_addr, CAPS, 0x1f80)
            .unwrap();
        return run(vm, at, store);
    }
    run(vm, at, store)
}
fn run(vm: Vm, at: u64, store: bool) -> (Exit, Vec<u8>, u64, u64) {
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, BASE + 0x1000);
    cpu.set_reg(Reg::Rdi, at);
    cpu.set_reg(Reg::Rax, u64::MAX);
    signal::HANDLE.store(cpu.fault_handle().as_usize(), Ordering::Relaxed);
    protect(
        &vm,
        if store {
            libc::PROT_READ
        } else {
            libc::PROT_NONE
        },
    );
    let exit = cpu.run(&vm, None);
    protect(&vm, libc::PROT_READ | libc::PROT_WRITE);
    let bytes = (0..8)
        .map(|i| {
            let mut b = [0];
            vm.read_bytes(at + i, &mut b).unwrap();
            b[0]
        })
        .collect();
    (exit, bytes, cpu.reg(Reg::Rcx), cpu.retired_instructions())
}

fn layouts() -> Vec<(&'static str, Layout)> {
    let end = BASE + (page() * 8) as u64;
    let g = guard();
    vec![
        ("contained", vec![(BASE, end, RegionKind::Ram)]),
        (
            "ram/ram",
            vec![(BASE, g, RegionKind::Ram), (g, end, RegionKind::Ram)],
        ),
        (
            "ram/ram/ram",
            vec![
                (BASE, g - 2, RegionKind::Ram),
                (g - 2, g + 2, RegionKind::Ram),
                (g + 2, end, RegionKind::Ram),
            ],
        ),
        // A writable RAM piece, a ROM piece, then the protected RAM piece: the
        // first piece must not be written before the third one faults.
        (
            "ram/rom/ram",
            vec![
                (BASE, g - 2, RegionKind::Ram),
                (g - 2, g, RegionKind::Rom),
                (g, end, RegionKind::Ram),
            ],
        ),
    ]
}

#[test]
fn protected_split_access_faults_like_contained() {
    let mut failures = Vec::new();
    for tier in 0..3u8 {
        for store in [false, true] {
            for (name, layout) in layouts() {
                let status = signal::child(|| {
                    // SAFETY: the forked child owns its handlers and mapping.
                    unsafe {
                        signal::install();
                    }
                    let (exit, bytes) = case(tier, true, store, &layout);
                    let access = if store {
                        AccessKind::Write
                    } else {
                        AccessKind::Read
                    };
                    assert!(
                        matches!(exit, Exit::UnmappedMemory { addr, access: a, .. }
                            if addr == guard() && a == access),
                        "{name} tier={tier} store={store}: {exit:?}"
                    );
                    assert_eq!(bytes, [0x53; 8], "{name} tier={tier} store={store}");
                    // The whole exit, fault evidence included, matches the same
                    // access inside one region.
                    let (contained, _) = case(tier, true, store, &layouts()[0].1);
                    assert_eq!(
                        format!("{exit:?}"),
                        format!("{contained:?}"),
                        "{name} tier={tier} store={store}"
                    );
                });
                if status != 0 {
                    failures.push((name, tier, store, status));
                }
            }
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:?}");
}

/// The operand's first byte is host-protected and a region edge lies inside the
/// operand after it. The re-execution after the containment miss reaches the host
/// fault at the operand's first byte, the address of the first attempt's miss; the
/// exit keeps the host-fault evidence of the same access inside one region, on
/// every tier, for a load and a store, as the block's first instruction and after
/// an `inc rcx`.
#[test]
fn protected_first_byte_of_split_access_faults_like_contained() {
    let end = BASE + (page() * 8) as u64;
    let g = guard();
    let contained: Layout = vec![(BASE, end, RegionKind::Ram)];
    let split: Layout = vec![
        (BASE, g + 4, RegionKind::Ram),
        (g + 4, end, RegionKind::Ram),
    ];
    let mut failures = Vec::new();
    for tier in 0..3u8 {
        for store in [false, true] {
            for lead in [false, true] {
                let status = signal::child(|| {
                    // SAFETY: the forked child owns its handlers and mapping.
                    unsafe {
                        signal::install();
                    }
                    let (want, _, want_count, want_retired) =
                        case_at(tier, true, store, &contained, g, lead);
                    let (exit, bytes, count, retired) = case_at(tier, true, store, &split, g, lead);
                    assert_eq!(bytes, [0x53; 8], "tier={tier} store={store} lead={lead}");
                    assert_eq!(
                        format!("{exit:?}"),
                        format!("{want:?}"),
                        "tier={tier} store={store} lead={lead}"
                    );
                    assert_eq!(
                        (count, retired),
                        (want_count, want_retired),
                        "tier={tier} store={store} lead={lead}"
                    );
                });
                if status != 0 {
                    failures.push((tier, store, lead, status));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "failed cases (tier, store, lead, status): {failures:?}"
    );
}

/// The twenty-bit models with host-fault recovery on. A RAM word store whose
/// two bytes are consecutive takes one host access (`interp::i8086::write`);
/// one whose second byte wraps at offset FFFFh takes the byte path (8086
/// Family UM §2.3 p. 2-12). With the second byte on a host-protected page,
/// both fault at that byte and commit neither byte, with equal exits, IP and
/// retirement, on every twenty-bit model.
#[test]
fn twenty_bit_word_store_over_protected_page_commits_no_byte() {
    use x86jit_core::{CpuMode, CpuModel};
    const SIZE: usize = 0x10_0000;
    // The protected host page, its first byte a twenty-bit physical address.
    let g = (page() * 4) as u64;
    assert!(g + 0xffff + 2 < SIZE as u64 && page() <= 0x8000);
    // `segment:offset` of the word (DS:DI): at offset Fh of the paragraph
    // below the page, crossing into it (one access), or at offset FFFFh with
    // DS based on the page (two byte transfers, the second at its first byte).
    let run = |model: CpuModel, segment: u64, offset: u64| -> (Exit, [u8; 2], u64, u64) {
        // SAFETY: the mapping stays owned by HostRam and is freed exactly once.
        let mut vm = unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            assert_ne!(ptr, libc::MAP_FAILED);
            Vm::with_backend_host_ram(
                VmConfig::flat(SIZE as u64),
                Box::new(InterpreterBackend),
                HostRam {
                    ptr: ptr.cast(),
                    len: SIZE,
                    guest_base: 0,
                    protect: None,
                    dtor: Box::new(|p, len| {
                        libc::munmap(p.cast(), len);
                    }),
                },
            )
            .unwrap()
        };
        vm.set_cpu_mode(CpuMode::Real16);
        vm.set_cpu_model(model);
        vm.mem.set_host_fault_recovery(true);
        vm.map(0, SIZE, Prot::RWX, RegionKind::Ram).unwrap();
        // mov [di], ax ; hlt
        vm.write_bytes(0x1000, &[0x89, 0x05, 0xf4]).unwrap();
        let first = (segment << 4) + offset;
        let second = (segment << 4) + ((offset + 1) & 0xffff);
        assert_eq!(second, g);
        vm.write_bytes(first, &[0x53]).unwrap();
        vm.write_bytes(second, &[0x53]).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Cs, 0);
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.set_reg(Reg::Ds, segment);
        cpu.set_reg(Reg::Rdi, offset);
        cpu.set_reg(Reg::Rax, 0xa5c3);
        signal::HANDLE.store(cpu.fault_handle().as_usize(), Ordering::Relaxed);
        // SAFETY: the page is aligned and inside this child's live mapping.
        let protect = |prot| unsafe {
            assert_eq!(
                libc::mprotect(
                    vm.mem.host_base().add(g as usize).cast_mut().cast(),
                    page(),
                    prot
                ),
                0
            );
        };
        protect(libc::PROT_READ);
        let exit = cpu.run(&vm, None);
        protect(libc::PROT_READ | libc::PROT_WRITE);
        let mut bytes = [0; 2];
        vm.read_bytes(first, &mut bytes[..1]).unwrap();
        vm.read_bytes(second, &mut bytes[1..]).unwrap();
        (exit, bytes, cpu.reg(Reg::Rip), cpu.retired_instructions())
    };
    let mut failures = Vec::new();
    for model in [
        CpuModel::I8088,
        CpuModel::I8086,
        CpuModel::NecV20,
        CpuModel::NecV30,
    ] {
        let status = signal::child(|| {
            // SAFETY: the forked child owns its handlers and mapping.
            unsafe {
                signal::install();
            }
            let word = run(model, (g >> 4) - 1, 0xf);
            let bytes = run(model, g >> 4, 0xffff);
            for (name, (exit, stored, rip, retired)) in [("word", &word), ("bytes", &bytes)] {
                assert!(
                    matches!(exit, Exit::UnmappedMemory { addr, access: AccessKind::Write, .. }
                        if *addr == g),
                    "{model:?} {name}: {exit:?}"
                );
                assert_eq!(*stored, [0x53; 2], "{model:?} {name}");
                assert_eq!((*rip, *retired), (0x1000, 0), "{model:?} {name}");
            }
            assert_eq!(
                format!("{:?}", word.0),
                format!("{:?}", bytes.0),
                "{model:?}"
            );
        });
        if status != 0 {
            failures.push((model, status));
        }
    }
    assert!(failures.is_empty(), "failed models: {failures:?}");
}
