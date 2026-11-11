//! Exit-owned evidence: SDM Vol 3A §5.7, §7.13, §7.15, Events 0/1/3/6/12/13/16.
//! REP restart state: SDM Vol 2B, REP/REPE/REPNE, Description/Operation.
use x86jit_core::{
    Backend, ErrorCode, Exit, FaultAccess, FaultCause, FaultClass, FaultOrigin, InterpreterBackend,
    Knowledge::{self, Known, Unknown},
    MemoryEffects, Prot, Reg, RegionCaps, RegionKind, RegisterProgress, TransferUnit, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;
const CODE: u64 = 0x1000;
const DATA: u64 = 0x4000;
const HOLE: u64 = 0x7000;
const BAD: u64 = 0x0100_0000_0000_0000;

fn setup(tier: usize, code: &[u8]) -> (Vm, x86jit_core::Vcpu, u64) {
    let backend: Box<dyn Backend> = match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        _ => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 16,
            max_icount: 256,
        })),
    };
    let mut vm = Vm::with_backend(VmConfig::flat(HOLE), backend);
    vm.set_tier_up_after(None);
    vm.map(0, HOLE as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    // This loop forces a multi-block compiled region; no tested operand uses R8.
    let mut bytes = if tier == 2 {
        vec![
            0x41, 0xb8, 2, 0, 0, 0, 0xeb, 0, 0x41, 0xff, 0xc8, 0x75, 0xfb,
        ]
    } else {
        vec![]
    };
    let reporting = CODE + bytes.len() as u64;
    bytes.extend_from_slice(code);
    bytes.push(0xf4);
    vm.write_bytes(CODE, &bytes).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rbx, HOLE);
    cpu.set_reg(Reg::Rsp, DATA);
    (vm, cpu, reporting)
}
fn run(vm: &Vm, cpu: &mut x86jit_core::Vcpu, tier: usize) -> Exit {
    let exit = cpu.run(vm, Some(100));
    if tier == 2 {
        assert!(vm.cache.regions() > 0, "region was not compiled");
    }
    exit
}
macro_rules! tiers {
    ($name:ident, $body:ident) => {
        mod $name {
            #[test]
            fn interp() {
                super::$body(0);
            }
            #[test]
            fn block() {
                super::$body(1);
            }
            #[test]
            fn region() {
                super::$body(2);
            }
        }
    };
}
fn codes(tier: usize) {
    for (bytes, vector, cause, code, advance) in [
        (
            &[0x48, 0xf7, 0xf1][..],
            0,
            FaultCause::Divide,
            ErrorCode::NoCode,
            0,
        ),
        (
            &[0x0f, 0x0b][..],
            6,
            FaultCause::InvalidOpcode,
            ErrorCode::NoCode,
            0,
        ),
        (
            &[0xf0, 0x90][..],
            6,
            FaultCause::InvalidOpcode,
            ErrorCode::NoCode,
            0,
        ),
        (&[0xcc][..], 3, FaultCause::Breakpoint, ErrorCode::NoCode, 1),
        (&[0xf1][..], 1, FaultCause::DebugTrap, ErrorCode::NoCode, 1),
    ] {
        let (vm, mut cpu, rip) = setup(tier, bytes);
        let exit = run(&vm, &mut cpu, tier);
        let m = exit
            .fault_metadata()
            .expect("engine exception lacks metadata");
        assert_eq!(m.vector(), Known(vector), "{exit:?}");
        assert_eq!(m.error_code(), code, "{exit:?}");
        assert_eq!(m.origin(), FaultOrigin::GuestException(cause));
        assert_eq!(m.saved_rip(), rip + advance);
        assert_eq!(m.saved_rip(), cpu.reg(Reg::Rip));
        assert_eq!(m.instruction_rip(), Known(rip));
        assert_eq!(
            m.class(),
            if advance == 0 {
                FaultClass::Fault
            } else {
                FaultClass::Trap
            }
        );
        assert_eq!(m.architectural_error_code(), None);
        if advance != 0 {
            assert_eq!(m.instruction_len(), Known(1));
        }
    }
    // Quotient overflow is the same #DE cause, not a pushed zero error code.
    let (vm, mut cpu, _) = setup(tier, &[0x48, 0xf7, 0xf1]);
    cpu.set_reg(Reg::Rcx, 1);
    cpu.set_reg(Reg::Rdx, 1);
    assert_eq!(
        run(&vm, &mut cpu, tier)
            .fault_metadata()
            .unwrap()
            .error_code(),
        ErrorCode::NoCode
    );
}
tiers!(exception_classes, codes);
fn zero_codes(tier: usize) {
    for (bytes, reg, bad, vector, cause) in [
        (
            &[0x48, 0x8b, 0x03][..],
            Reg::Rbx,
            BAD,
            13,
            FaultCause::NonCanonical,
        ),
        (
            &[0x48, 0x8b, 0x45, 0][..],
            Reg::Rbp,
            BAD,
            12,
            FaultCause::NonCanonical,
        ),
        (
            &[0xdb, 0x2b][..],
            Reg::Rbx,
            BAD,
            13,
            FaultCause::NonCanonical,
        ),
        (
            &[0x0f, 0xae, 0x03][..],
            Reg::Rbx,
            BAD,
            13,
            FaultCause::NonCanonical,
        ),
        (
            &[0x0f, 0x28, 0x03][..],
            Reg::Rbx,
            DATA + 1,
            13,
            FaultCause::RequiredAlignment,
        ),
    ] {
        let (vm, mut cpu, rip) = setup(tier, bytes);
        cpu.set_reg(reg, bad);
        let exit = run(&vm, &mut cpu, tier);
        let m = exit.fault_metadata().unwrap();
        assert_eq!(m.vector(), Known(vector));
        assert_eq!(m.architectural_error_code(), Some(0), "{exit:?}");
        assert_eq!(m.origin(), FaultOrigin::GuestException(cause));
        assert_eq!(m.instruction_rip(), Known(rip));
        assert_eq!(m.instruction_len(), Known(bytes.len() as u8));
    }
    let (vm, mut cpu, _) = setup(tier, &[0x66; 16]);
    let exit = run(&vm, &mut cpu, tier);
    let m = exit.fault_metadata().unwrap();
    assert_eq!(m.architectural_error_code(), Some(0), "{exit:?}");
    assert_eq!(
        m.origin(),
        FaultOrigin::GuestException(FaultCause::InstructionLength)
    );
    assert_eq!(m.instruction_len(), Unknown);
}
tiers!(known_zero_causes, zero_codes);
fn map_facts(tier: usize) {
    for (code, intent, value) in [
        (&[0x48, 0x8b, 0x03][..], FaultAccess::Read, 0),
        (&[0x48, 0x89, 0x03][..], FaultAccess::Write, 2),
    ] {
        let (vm, mut cpu, rip) = setup(tier, code);
        let exit = run(&vm, &mut cpu, tier);
        let m = exit.fault_metadata().unwrap();
        assert_eq!(m.origin(), FaultOrigin::GuestMap);
        assert_eq!(m.instruction_rip(), Known(rip));
        assert_eq!(m.instruction_len(), Known(3));
        assert_eq!(m.operand_size(), Known(8));
        assert_eq!(m.operand_base(), Known(HOLE));
        assert_eq!(m.transfer_size(), Known(8));
        assert_eq!(m.access(), Known(intent));
        assert_eq!(m.transfer_access(), Known(intent));
        assert_eq!(m.vector(), Unknown);
        assert_eq!(m.error_code(), ErrorCode::Unknown);
        assert_eq!(m.architectural_error_code(), None);
        let projection = m.page_fault_projection().unwrap();
        assert_eq!(projection.error_code(), ErrorCode::bits(value, 0x13));
        for bit in [2, 3, 5, 6, 7, 15, 31] {
            assert_eq!(projection.error_code().bit(bit), None);
        }
    }
}
tiers!(map_projection, map_facts);
fn rmw(tier: usize) {
    for bytes in [
        &[0x48, 0x83, 0x03, 1][..],
        &[0xf0, 0x48, 0x83, 0x03, 1][..],
        &[0x48, 0x0f, 0xb1, 0x0b][..],
    ] {
        let (vm, mut cpu, _) = setup(tier, bytes);
        let exit = run(&vm, &mut cpu, tier);
        let m = exit.fault_metadata().unwrap();
        assert_eq!(m.access(), Known(FaultAccess::ReadWrite), "{exit:?}");
        assert_eq!(m.page_fault_projection().unwrap().error_code().bit(1), None);
        assert_eq!(m.transfer_size(), Known(8));
        if m.transfer_access() == Known(FaultAccess::Read) {
            assert_eq!(m.progress().ram(), MemoryEffects::None);
        }
    }
}
tiers!(rmw_intent, rmw);
fn split_width(tier: usize) {
    let (vm, mut cpu, _) = setup(tier, &[0x48, 0x8b, 0x03]);
    cpu.set_reg(Reg::Rbx, HOLE - 4);
    let exit = run(&vm, &mut cpu, tier);
    let m = exit.fault_metadata().unwrap();
    assert_eq!(
        m.transfer_size(),
        Known(8),
        "tail length is not transfer width: {exit:?}"
    );
    assert_eq!(m.operand_size(), Known(8));
    // Interpreter reports the transfer address, compiled bounds checks its absent tail.
    assert_eq!(
        m.linear_address().map_value(),
        if tier == 0 { HOLE - 4 } else { HOLE }
    );
    let (vm, mut cpu, _) = setup(tier, &[0xc5, 0xfe, 0x6f, 0x03]);
    cpu.set_reg(Reg::Rbx, HOLE - 16);
    let exit = run(&vm, &mut cpu, tier);
    let m = exit.fault_metadata().unwrap();
    assert_eq!(m.operand_size(), Known(32));
    assert_eq!(m.transfer_size(), Known(if tier == 0 { 8 } else { 32 }));
}
tiers!(requested_transfer_width, split_width);
fn rep(tier: usize) {
    for (src, dst, df) in [
        (DATA, HOLE - 2, false),
        (HOLE - 2, DATA, false),
        (DATA, HOLE, true),
    ] {
        let (vm, mut cpu, rip) = setup(tier, &[0xf3, 0xa4]);
        cpu.set_reg(Reg::Rsi, src);
        cpu.set_reg(Reg::Rdi, dst);
        cpu.set_reg(Reg::Rcx, 5);
        cpu.cpu.flags.df = df;
        let exit = run(&vm, &mut cpu, tier);
        let m = exit.fault_metadata().unwrap();
        let restart = m.progress().rep().expect("REP lacks restart state");
        assert_eq!(restart.source(), cpu.reg(Reg::Rsi));
        assert_eq!(restart.destination(), cpu.reg(Reg::Rdi));
        assert_eq!(restart.remaining(), cpu.reg(Reg::Rcx));
        assert_eq!(restart.direction_flag(), df);
        assert_eq!(restart.address_bits(), 64);
        assert_eq!(restart.element_size(), 1);
        assert_eq!(restart.completed_iterations(), Unknown);
        assert_eq!(m.progress().had_partial_progress(), Unknown);
        assert_eq!(m.progress().registers(), RegisterProgress::RepRestart);
        assert_eq!(m.instruction_rip(), Known(rip));
        assert_eq!(m.transfer_size(), Known(1));
    }
}
tiers!(rep_restart, rep);
fn fetch(tier: usize) {
    let (mut vm, mut cpu, _) = setup(tier, &[0xf4]);
    cpu.set_reg(Reg::Rip, HOLE);
    let exit = run(&vm, &mut cpu, 0);
    let m = exit.fault_metadata().unwrap();
    assert_eq!(m.origin(), FaultOrigin::GuestMap);
    assert_eq!(m.instruction_len(), Unknown);
    assert_eq!(m.transfer_size(), Known(1));
    assert_eq!(m.transfer_unit(), TransferUnit::FetchByte);
    assert_eq!(m.page_fault_projection().unwrap().error_code().bit(4), None);
    assert!(m.page_fault_projection().unwrap().is_fetch());
    vm.unmap(0, HOLE as usize).unwrap();
    vm.map(0, HOLE as usize, Prot::RWX, RegionKind::Trap)
        .unwrap();
    cpu.set_reg(Reg::Rip, CODE);
    let exit = cpu.run(&vm, Some(20));
    let m = exit.fault_metadata().unwrap();
    assert_eq!(m.origin(), FaultOrigin::TrapFetch);
    assert_eq!(m.page_fault_projection(), None);
}
tiers!(fetch_origin, fetch);
#[test]
fn legacy_constructors_and_owned_lifetime() {
    const LEGACY: Exit = Exit::exception(1, 13);
    const MAP: Exit = Exit::unmapped_memory(2, x86jit_core::AccessKind::Read);
    assert!(LEGACY.fault_metadata().is_none());
    assert!(MAP.fault_metadata().is_none());
    for vector in [10, 11, 12, 13, 14, 17] {
        let m = x86jit_core::FaultMetadata::exception(1, 1, vector, FaultCause::Unknown);
        assert_eq!(m.error_code(), ErrorCode::Unknown);
        assert_eq!(m.architectural_error_code(), None);
    }
    let (mut vm, mut cpu, _) = setup(1, &[0x48, 0x8b, 0x03]);
    let old = cpu.run(&vm, Some(20));
    let copy = old.clone();
    cpu.set_reg(Reg::Rbx, DATA);
    assert!(matches!(cpu.run(&vm, Some(20)), Exit::Hlt));
    vm.unmap(0, HOLE as usize).unwrap();
    assert_eq!(copy.fault_metadata().unwrap().transfer_size(), Known(8));
    assert_eq!(
        old.fault_metadata().unwrap().linear_address(),
        copy.fault_metadata().unwrap().linear_address()
    );
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Exit>();
}
trait AddressValue {
    fn map_value(self) -> u64;
}
impl AddressValue for Knowledge<x86jit_core::FaultAddress> {
    fn map_value(self) -> u64 {
        match self {
            Known(a) => a.value(),
            _ => panic!("missing address"),
        }
    }
}

fn pending_mf(tier: usize) {
    let (vm, mut cpu, rip) = setup(tier, &[0x9b]); // FWAIT reports an earlier event.
    cpu.cpu.fpu_sw = 0x81;
    let exit = run(&vm, &mut cpu, tier);
    let m = exit.fault_metadata().unwrap();
    assert_eq!(m.vector(), Known(16));
    assert_eq!(m.error_code(), ErrorCode::NoCode);
    assert_eq!(
        m.origin(),
        FaultOrigin::GuestException(FaultCause::PendingX87)
    );
    assert_eq!(m.instruction_rip(), Known(rip));
    assert_eq!(m.instruction_len(), Known(1));
}
tiers!(pending_x87_report, pending_mf);
fn iret_causes(tier: usize) {
    for (ip, cs, nt, expected) in [
        (CODE, 9, false, None),
        (BAD, 0, false, Some(0)),
        (CODE, 0, true, Some(0)),
    ] {
        let (vm, mut cpu, rip) = setup(tier, &[0x48, 0xcf]);
        for (i, value) in [ip, cs, 2, DATA, 0].into_iter().enumerate() {
            vm.write_bytes(DATA + (i * 8) as u64, &value.to_le_bytes())
                .unwrap();
        }
        if nt {
            cpu.cpu.rflags_system |= 1 << 14;
        }
        let exit = run(&vm, &mut cpu, tier);
        let m = exit.fault_metadata().unwrap();
        assert_eq!(m.vector(), Known(13));
        assert_eq!(m.instruction_rip(), Known(rip));
        assert_eq!(m.architectural_error_code(), expected, "{exit:?}");
        if expected.is_none() {
            assert_eq!(m.error_code(), ErrorCode::Unknown);
        }
    }
}
tiers!(iret_cause_codes, iret_causes);
fn helper_and_mask_widths(tier: usize) {
    use iced_x86::code_asm::*;
    let mut forms = Vec::new();
    let mut a = CodeAssembler::new(64).unwrap();
    a.vpaddd(zmm0, zmm1, dword_bcst(rbx)).unwrap();
    forms.push((a.assemble(CODE).unwrap(), 4));
    let mut a = CodeAssembler::new(64).unwrap();
    a.vmovdqu32(zmm0.k1(), zmmword_ptr(rbx)).unwrap();
    forms.push((a.assemble(CODE).unwrap(), 64));
    let mut a = CodeAssembler::new(64).unwrap();
    a.vpmovdb(xmmword_ptr(rbx), zmm0).unwrap();
    forms.push((a.assemble(CODE).unwrap(), 16));
    forms.push((vec![0xdb, 0x2b], 10)); // FLD m80, one Rust helper request.
    forms.push((vec![0x0f, 0xae, 0x03], 512)); // FXSAVE.
    for (bytes, width) in forms {
        for routed in [false, true] {
            let (mut vm, mut cpu, _) = setup(tier, &bytes);
            vm.mem.set_host_fault_recovery(routed);
            cpu.cpu.kmask[1] = 0x8000;
            let exit = run(&vm, &mut cpu, tier);
            let m = exit.fault_metadata().unwrap();
            assert_eq!(m.operand_size(), Known(width), "{bytes:02x?}: {exit:?}");
            assert_ne!(m.transfer_size(), Unknown);
        }
    }
    let (vm, mut cpu, _) = setup(tier, &[0x62, 0xf1, 0x7e, 0x49, 0x6f, 0x03]); // VMOVDQU32 zmm0{k1},[rbx].
    cpu.cpu.kmask[1] = 0;
    let exit = run(&vm, &mut cpu, tier);
    assert!(matches!(exit, Exit::Hlt));
    assert!(exit.fault_metadata().is_none());
}
tiers!(helper_mask_and_carrier_widths, helper_and_mask_widths);
#[test]
fn fetch_split_and_real16_address_space() {
    for tier in 0..3 {
        let (mut vm, mut cpu, _) = setup(tier, &[0xf4]);
        vm.unmap(0, HOLE as usize).unwrap();
        vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x2000, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
        vm.write_bytes(0x1fff, &[0x48]).unwrap();
        cpu.set_reg(Reg::Rip, 0x1fff);
        let exit = cpu.run(&vm, Some(20));
        let m = exit.fault_metadata().unwrap();
        assert_eq!(m.instruction_rip(), Known(0x1fff));
        assert_eq!(m.saved_rip(), 0x1fff);
        assert_eq!(m.linear_address().map_value(), 0x2000);
        assert_eq!(m.origin(), FaultOrigin::TrapFetch);
        assert_eq!(m.instruction_len(), Unknown);
    }
    let (mut vm, _, _) = setup(0, &[0xf4]);
    vm.set_cpu_mode(x86jit_core::CpuMode::Real16);
    let mut cpu = vm.new_vcpu();
    cpu.cpu.cs = 0x100;
    cpu.set_reg(Reg::Rip, 0x6000);
    let exit = cpu.run(&vm, Some(20));
    let m = exit.fault_metadata().unwrap();
    assert_eq!(m.saved_rip(), 0x6000);
    assert_eq!(m.real_mode_cs_ip(), Known((0x100, 0x6000)));
    assert_eq!(m.linear_address().map_value(), HOLE);
    assert!(
        matches!(m.linear_address(),Known(a) if a.space() == x86jit_core::AddressSpace::RealModePhysical)
    );
}

struct Counter;
thread_local! { static ALLOCATIONS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) }; }
// SAFETY: delegates every allocation operation and layout unchanged to System;
// the thread-local counter has a const initializer and never allocates.
unsafe impl std::alloc::GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| {
            if let Some(count) = n.get() {
                n.set(Some(count + 1));
            }
        });
        // SAFETY: the caller supplies a valid allocator layout.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: forwards the original allocation pointer and layout.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| {
            if let Some(count) = n.get() {
                n.set(Some(count + 1));
            }
        });
        // SAFETY: forwards the original allocation and requested size.
        unsafe { std::alloc::System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;
fn no_success_allocations(tier: usize) {
    for code in [
        &[0x90][..],
        &[0x48, 0x8b, 0x03][..],
        &[0x48, 0x89, 0x03][..],
        &[0xf3, 0xa4][..],
    ] {
        let (vm, mut cpu, _) = setup(tier, code);
        let reset = |cpu: &mut x86jit_core::Vcpu| {
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rbx, DATA);
            cpu.set_reg(Reg::Rsi, DATA);
            cpu.set_reg(Reg::Rdi, DATA + 256);
            cpu.set_reg(Reg::Rcx, 64);
        };
        reset(&mut cpu);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        reset(&mut cpu);
        assert!(matches!(cpu.run(&vm, Some(100)), Exit::Hlt));
        ALLOCATIONS.with(|n| n.set(Some(0)));
        reset(&mut cpu);
        let exit = cpu.run(&vm, Some(100));
        let count = ALLOCATIONS.with(|n| n.replace(None).unwrap());
        assert!(matches!(exit, Exit::Hlt));
        assert!(exit.fault_metadata().is_none());
        assert_eq!(count, 0, "{code:02x?}, tier={tier}");
    }
}
tiers!(success_allocations, no_success_allocations);

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod native {
    use x86jit_tests::{
        native::{run_native_fault_details, NativeFault},
        oracle::VectorInput,
        vector::{CpuSnapshot, MemChunk, MemKind, RunSpec},
    };
    #[test]
    fn native_split_store_fault_preserves_readable_bytes() {
        use iced_x86::code_asm::*;
        // SDM Vol 3A §§7.5–7.6: a fault restores instruction-entry state.
        const ENTRY: u64 = 0x21_0000;
        const GUARD: u64 = 0x25_0000;
        for width in [2, 4, 8, 16] {
            let mut a = CodeAssembler::new(64).unwrap();
            match width {
                2 => a.mov(word_ptr(rdi), ax).unwrap(),
                4 => a.mov(dword_ptr(rdi), eax).unwrap(),
                8 => a.mov(qword_ptr(rdi), rax).unwrap(),
                _ => a.movdqu(xmmword_ptr(rdi), xmm0).unwrap(),
            }
            a.hlt().unwrap();
            let address = GUARD - width / 2;
            let mut cpu = CpuSnapshot::default();
            cpu.gpr[7] = address;
            cpu.gpr[0] = 0x1212_1212_1212_1212;
            cpu.xmm[0] = u128::from_le_bytes([0x12; 16]);
            let prefix = MemChunk {
                addr: address,
                bytes: vec![0x35; width as usize / 2],
                kind: MemKind::Ram,
            };
            let input = VectorInput {
                cpu_init: cpu,
                mem_init: vec![
                    MemChunk {
                        addr: ENTRY,
                        bytes: a.assemble(ENTRY).unwrap(),
                        kind: MemKind::Ram,
                    },
                    prefix.clone(),
                ],
                entry: ENTRY,
                run: RunSpec::UntilExit,
            };
            let (out, evidence) =
                run_native_fault_details(&input, NativeFault::PageFault { addr: GUARD })
                    .expect("native split-store fault must be observable");
            assert_eq!(evidence.vector, 14);
            assert_eq!(out.cpu.rip, ENTRY);
            assert_eq!(out.mem[1].bytes, prefix.bytes);
        }
    }

    #[test]
    fn native_error_code_and_rmw_evidence() {
        const ENTRY: u64 = 0x21_0000;
        const UNMAPPED: u64 = 0x24_0000;
        for (bytes, fault, vector, mask, value) in [
            (
                &[0x48, 0x8b, 0x03][..],
                NativeFault::PageFault { addr: UNMAPPED },
                14,
                0x13,
                0,
            ),
            (
                &[0x48, 0x89, 0x03][..],
                NativeFault::PageFault { addr: UNMAPPED },
                14,
                0x13,
                2,
            ),
            // RMW projections leave W/R unknown even when this oracle observes 1.
            (
                &[0x48, 0x83, 0x03, 1][..],
                NativeFault::PageFault { addr: UNMAPPED },
                14,
                0x11,
                0,
            ),
            (
                &[0xf0, 0x48, 0x83, 0x03, 1][..],
                NativeFault::PageFault { addr: UNMAPPED },
                14,
                0x11,
                0,
            ),
            (
                &[0x48, 0x0f, 0xb1, 0x0b][..],
                NativeFault::PageFault { addr: UNMAPPED },
                14,
                0x11,
                0,
            ),
            (&[0x48, 0xf7, 0xf1][..], NativeFault::DivideError, 0, 0, 0),
            (&[0xf7, 0xf1][..], NativeFault::DivideError, 0, 0, 0),
            (&[0x0f, 0x0b][..], NativeFault::InvalidOpcode, 6, 0, 0),
            (&[0xff, 0xff][..], NativeFault::InvalidOpcode, 6, 0, 0),
            (&[0xcc][..], NativeFault::Breakpoint, 3, 0, 0),
            (&[0xf1][..], NativeFault::DebugTrap, 1, 0, 0),
            (
                &[0xd9, 0xee, 0xd9, 0xee, 0xde, 0xf9, 0x9b][..],
                NativeFault::X87FloatingPoint,
                16,
                0,
                0,
            ),
            (
                &[0x48, 0x8b, 0x03][..],
                NativeFault::GeneralProtection,
                13,
                u32::MAX,
                0,
            ),
            (
                &[0x48, 0x8b, 0x45, 0][..],
                NativeFault::StackFault,
                12,
                u32::MAX,
                0,
            ),
            (
                &[0x0f, 0x28, 0x03][..],
                NativeFault::GeneralProtection,
                13,
                u32::MAX,
                0,
            ),
            (
                &[0x66; 16][..],
                NativeFault::GeneralProtection,
                13,
                u32::MAX,
                0,
            ),
        ] {
            let mut init = CpuSnapshot::default();
            init.gpr[3] = UNMAPPED;
            if vector == 0 && bytes == [0xf7, 0xf1] {
                // SDM Vol 2A, DIV, Operation: EDX:EAX / ECX overflows 32 bits.
                init.gpr[0] = u32::MAX as u64;
                init.gpr[1] = 1;
                init.gpr[2] = 1;
            }
            if vector == 13 && bytes == [0x48, 0x8b, 0x03] {
                init.gpr[3] = 0x0100_0000_0000_0000;
            }
            if vector == 13 && bytes == [0x0f, 0x28, 0x03] {
                init.gpr[3] = ENTRY + 1;
            }
            if vector == 12 {
                init.gpr[5] = 0x0100_0000_0000_0000;
            }
            if vector == 16 {
                init.fpu_cw = 0x037e;
            }
            let input = VectorInput {
                cpu_init: init,
                mem_init: vec![MemChunk {
                    addr: ENTRY,
                    bytes: bytes.to_vec(),
                    kind: MemKind::Ram,
                }],
                entry: ENTRY,
                run: RunSpec::UntilExit,
            };
            let (out, evidence) = run_native_fault_details(&input, fault)
                .expect("designated native fault must be observable");
            assert_eq!(evidence.vector, vector);
            assert_eq!(
                out.cpu.rip,
                ENTRY
                    + if vector == 1 || vector == 3 {
                        1
                    } else if vector == 16 {
                        6
                    } else {
                        0
                    }
            );
            assert_eq!(evidence.error_code & u64::from(mask), value as u64);
            if vector == 14 {
                assert_eq!(evidence.fault_address, UNMAPPED);
            }
        }
    }
}

fn device_retry(tier: usize) {
    let (mut vm, mut cpu, _) = setup(tier, &[0xf3, 0x0f, 0x6f, 0x03]); // MOVDQU xmm0,[rbx].
    vm.unmap(0, HOLE as usize).unwrap();
    vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x5000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    cpu.set_reg(Reg::Rbx, 0x5ff8);
    let first = run(&vm, &mut cpu, tier);
    assert!(matches!(
        first,
        Exit::MmioRead {
            addr: 0x5ff8,
            size: 8,
            ..
        }
    ));
    assert!(first.fault_metadata().is_none());
    cpu.complete_mmio_read(0x1234);
    let exit = cpu.run(&vm, Some(100));
    let m = exit.fault_metadata().unwrap();
    assert!(matches!(exit, Exit::UnmappedMemory { addr: 0x6000, .. }));
    assert_eq!(m.operand_size(), Known(16));
    assert_eq!(m.transfer_size(), Known(8));
    assert_eq!(m.progress().acknowledged_transfers(), &[(0x5ff8, 0x1234)]);
    assert_eq!(m.progress().had_partial_progress(), Known(true));
    assert_eq!(m.progress().device_history_complete(), Unknown);
}
tiers!(acknowledged_device_retry, device_retry);
fn stack_faults(tier: usize) {
    for bytes in [&[0x50][..], &[0x58][..], &[0xe8, 0, 0, 0, 0][..]] {
        let (vm, mut cpu, rip) = setup(tier, bytes);
        let sp = if bytes[0] == 0x58 { HOLE } else { HOLE + 8 };
        cpu.set_reg(Reg::Rsp, sp);
        let exit = run(&vm, &mut cpu, tier);
        let m = exit.fault_metadata().unwrap();
        assert_eq!(cpu.reg(Reg::Rsp), sp);
        assert_eq!(m.saved_rip(), rip);
        assert_eq!(m.transfer_size(), Known(8));
        assert_eq!(m.progress().registers(), RegisterProgress::InstructionEntry);
        assert_eq!(m.progress().ram(), MemoryEffects::None, "{exit:?}");
    }
}
tiers!(stack_commit_last, stack_faults);
#[test]
fn generic_ir_does_not_prove_a_decode_cause() {
    use x86jit_core::{
        CachedBlock, CpuMode, CpuState, IrBlock, IrOp, MemConsistency, Memory, MemoryModel,
    };
    for vector in [10, 11, 12, 13, 14, 17] {
        let ir = IrBlock {
            legacy_context: None,
            mode: x86jit_core::CpuMode::Long64,
            guest_start: CODE,
            guest_len: 15,
            ops: vec![
                IrOp::InsnStart { guest_addr: CODE },
                IrOp::Trap { vector, advance: 0 },
            ],
            temp_count: 0,
            icount: 1,
        };
        let mem = Memory::new(MemoryModel::Flat { size: 0x2000 });
        let mut cpu = CpuState {
            rip: CODE,
            ..CpuState::default()
        };
        let result = x86jit_core::interp::interpret_block(
            &ir,
            &mut cpu,
            &mem,
            &mut vec![],
            &mut Default::default(),
        );
        let x86jit_core::StepResult::Exit(exit) = result else {
            panic!("expected exception")
        };
        assert_eq!(
            exit.fault_metadata().unwrap().error_code(),
            ErrorCode::Unknown
        );
        let jit = JitBackend::new();
        let CachedBlock::Compiled { entry, .. } =
            jit.materialize(&ir, MemConsistency::Fast, None, 0)
        else {
            panic!("must compile")
        };
        // SAFETY: entry is live in jit and was compiled for this CPU/context ABI.
        let result = unsafe {
            x86jit_core::jit_abi::run_compiled(
                entry,
                &mut cpu,
                &mem,
                CpuMode::Long64,
                x86jit_core::MemConsistency::Fast,
            )
        };
        let x86jit_core::StepResult::Exit(exit) = result else {
            panic!("expected exception")
        };
        assert_eq!(
            exit.fault_metadata().unwrap().error_code(),
            ErrorCode::Unknown
        );
    }
}
fn software_interrupt(tier: usize) {
    let (vm, mut cpu, rip) = setup(tier, &[0xcd, 0x0d]);
    let exit = run(&vm, &mut cpu, tier);
    let m = exit.fault_metadata().unwrap();
    assert_eq!(m.error_code(), ErrorCode::NoCode);
    assert_eq!(m.origin(), FaultOrigin::SoftwareInterrupt);
    assert_eq!(m.saved_rip(), rip + 2);
    assert_eq!(m.instruction_rip(), Known(rip));
    assert_eq!(m.architectural_error_code(), None);
}
tiers!(software_int_has_no_code, software_interrupt);

#[test]
#[should_panic]
fn error_code_rejects_values_outside_known_mask() {
    let _ = ErrorCode::bits(2, 1);
}
#[test]
fn layout_and_legacy_abi_validity() {
    use x86jit_core::{CpuMode, CpuState, Memory, MemoryModel};
    if cfg!(target_pointer_width = "64") {
        assert_eq!(std::mem::size_of::<Exit>(), 32);
    }
    assert_eq!(std::mem::size_of::<x86jit_core::MemTrap>(), 1);
    unsafe extern "C" fn old_backend(cpu: *mut u8, mem: *mut u8) -> u64 {
        // SAFETY: run_compiled supplies its exclusive live ABI records.
        unsafe {
            (*cpu.cast::<CpuState>()).rip = CODE;
            let mem = &mut *mem.cast::<x86jit_core::jit_abi::MemCtx>();
            mem.next_entry = u64::MAX;
            mem.link_slot = u64::MAX; // deliberately invalid detailed aliases
            mem.exception_vector = 17;
        }
        x86jit_core::jit_abi::RET_EXCEPTION
    }
    let mem = Memory::new(MemoryModel::Flat { size: 0x2000 });
    let mut cpu = CpuState::default();
    // SAFETY: this leaf implements the documented legacy compiled-block ABI.
    let result = unsafe {
        x86jit_core::jit_abi::run_compiled(
            x86jit_core::CompiledPtr(old_backend as *const u8),
            &mut cpu,
            &mem,
            CpuMode::Long64,
            x86jit_core::MemConsistency::Fast,
        )
    };
    let x86jit_core::StepResult::Exit(exit) = result else {
        panic!("expected legacy exception")
    };
    assert!(matches!(
        exit,
        Exit::Exception {
            addr: CODE,
            vector: 17,
            ..
        }
    ));
    let m = exit.fault_metadata().unwrap();
    assert_eq!(m.error_code(), ErrorCode::Unknown);
    assert_eq!(m.instruction_len(), Unknown);
}

fn rep_address_widths(tier: usize) {
    for (mode, bits, source, destination, count) in [
        (
            x86jit_core::CpuMode::Long64,
            32,
            DATA,
            HOLE - 2,
            0x1234_0000_0005,
        ),
        (
            x86jit_core::CpuMode::Compat32,
            16,
            0x1234_4000,
            0x5678_6ffe,
            0x9abc_0005,
        ),
    ] {
        let (mut vm, _, rip) = setup(tier, &[0x67, 0xf3, 0xa4]);
        vm.set_cpu_mode(mode);
        if tier == 2 && mode == x86jit_core::CpuMode::Compat32 {
            vm.write_bytes(
                CODE,
                &[
                    0xba, 2, 0, 0, 0, 0xeb, 0, 0x4a, 0x75, 0xfd, 0x90, 0x90, 0x90,
                ],
            )
            .unwrap();
        }
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rsi, source);
        cpu.set_reg(Reg::Rdi, destination);
        cpu.set_reg(Reg::Rcx, count);
        let exit = run(&vm, &mut cpu, tier);
        let m = exit.fault_metadata().unwrap();
        let restart = m.progress().rep().unwrap();
        let mask = if bits == 16 { 0xffff } else { 0xffff_ffff };
        assert_eq!(restart.source(), cpu.reg(Reg::Rsi) & mask);
        assert_eq!(restart.destination(), cpu.reg(Reg::Rdi) & mask);
        assert_eq!(restart.remaining(), cpu.reg(Reg::Rcx) & mask);
        assert_eq!(restart.remaining(), 3);
        assert_eq!(restart.address_bits(), bits);
        assert_eq!(m.instruction_rip(), Known(rip));
        assert_eq!(restart.completed_iterations(), Unknown);
    }
}
tiers!(rep_address_sizes, rep_address_widths);

fn stack_noncanonical(tier: usize) {
    for bytes in [&[0xe8, 0, 0, 0, 0][..], &[0xc3][..]] {
        let (vm, mut cpu, rip) = setup(tier, bytes);
        let stack = BAD + if bytes[0] == 0xe8 { 8 } else { 0 };
        cpu.set_reg(Reg::Rsp, stack);
        let exit = run(&vm, &mut cpu, tier);
        let m = exit.fault_metadata().unwrap();
        assert!(
            matches!(exit, Exit::Exception { vector: 12, .. }),
            "{exit:?}"
        );
        assert_eq!(m.architectural_error_code(), Some(0));
        assert_eq!(m.instruction_rip(), Known(rip));
        assert_eq!(cpu.reg(Reg::Rsp), stack);
    }
}
tiers!(stack_segment_metadata, stack_noncanonical);
#[test]
fn real16_failed_delivery_reports_the_delivery_access() {
    for stepping in [false, true] {
        let (mut vm, _, _) = setup(0, &[0xf7, 0xf1]);
        vm.set_cpu_mode(x86jit_core::CpuMode::Real16);
        let mut cpu = vm.new_vcpu();
        cpu.cpu.cs = 0x100;
        cpu.cpu.ss = 0x2000;
        cpu.set_reg(Reg::Rip, 0);
        cpu.set_reg(Reg::Rsp, 0);
        let exit = if stepping {
            match cpu.step_instruction(&vm) {
                x86jit_core::StepResult::Exit(e) => e,
                _ => panic!("delivery should fail"),
            }
        } else {
            cpu.run(&vm, Some(20))
        };
        assert!(
            matches!(exit, Exit::UnmappedMemory { addr: 0x2fffe, .. }),
            "{exit:?}"
        );
        let m = exit.fault_metadata().unwrap();
        assert_eq!(m.vector(), Unknown);
        assert_eq!(m.error_code(), ErrorCode::Unknown);
        assert_eq!(m.transfer_size(), Known(2));
        assert_eq!(m.real_mode_cs_ip(), Known((0x100, 0)));
        assert!(
            matches!(m.linear_address(),Known(a) if a.space()==x86jit_core::AddressSpace::RealModePhysical)
        );
    }
}
#[test]
fn containment_miss_does_not_prove_a_missing_byte() {
    let (mut vm, mut cpu, _) = setup(0, &[0x48, 0x8b, 0x03]);
    vm.unmap(0, HOLE as usize).unwrap();
    vm.map(0, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(0x5000, 0x2000, Prot::RW, RegionKind::Ram).unwrap();
    cpu.set_reg(Reg::Rbx, 0x4ffc);
    let exit = cpu.run(&vm, Some(20));
    let m = exit.fault_metadata().unwrap();
    assert!(matches!(exit, Exit::UnmappedMemory { addr: 0x4ffc, .. }));
    assert!(matches!(m.linear_address(), Known(address)
        if address.precision() == x86jit_core::AddressPrecision::TransferAddress));
    assert_eq!(m.page_fault_projection(), None);
    assert_eq!(m.architectural_error_code(), None);
}

fn split_store_progress(tier: usize) {
    use iced_x86::code_asm::*;
    let mut forms = Vec::new();
    let mut a = CodeAssembler::new(64).unwrap();
    a.movdqu(xmmword_ptr(rbx), xmm0).unwrap();
    forms.push((a.assemble(CODE).unwrap(), 16));
    let mut a = CodeAssembler::new(64).unwrap();
    a.vmovdqu(ymmword_ptr(rbx), ymm0).unwrap();
    forms.push((a.assemble(CODE).unwrap(), 32));
    let mut a = CodeAssembler::new(64).unwrap();
    a.vmovdqu32(zmmword_ptr(rbx), zmm0).unwrap();
    forms.push((a.assemble(CODE).unwrap(), 64));
    for (bytes, width) in forms {
        for (routed, precise) in [(false, false), (false, true), (true, false), (true, true)] {
            let (mut vm, mut cpu, rip) = setup(tier, &bytes);
            vm.mem.set_host_fault_recovery(routed);
            vm.set_precise_split_stores(precise);
            cpu.set_reg(Reg::Rbx, HOLE - width / 2);
            let exit = run(&vm, &mut cpu, tier);
            let m = exit.fault_metadata().unwrap();
            assert_eq!(m.instruction_rip(), Known(rip));
            assert_eq!(m.operand_size(), Known(width as u16));
            assert_eq!(m.progress().registers(), RegisterProgress::InstructionEntry);
            assert_eq!(
                m.progress().ram(),
                if precise && tier != 0 {
                    MemoryEffects::None
                } else {
                    MemoryEffects::PossiblyPartial
                }
            );
            // Device history remains independently qualified.
            assert_eq!(m.progress().had_partial_progress(), Unknown);
        }
    }
}
tiers!(split_store_preflight_effects, split_store_progress);

#[test]
fn raw_host_record_does_not_qualify_guest_error_code() {
    use x86jit_core::{FaultMetadata, HostFaultPlatform, RawHostFault};
    let unknown = FaultMetadata::unknown(CODE);
    assert_eq!(unknown.raw_host_fault(), None);
    let raw = RawHostFault {
        trap_number: Some(14),
        error_code: Some(7),
        fault_address: Some(0x1234),
        ..RawHostFault::new(HostFaultPlatform::LinuxX86_64)
    };
    let exit = Exit::unmapped_memory(DATA, x86jit_core::AccessKind::Read)
        .with_fault_metadata(unknown.with_raw_host_fault(raw));
    let copied = exit.clone();
    drop(exit);
    let m = copied.fault_metadata().unwrap();
    assert_eq!(m.raw_host_fault(), Some(&raw));
    assert_eq!(m.error_code(), ErrorCode::Unknown);
    assert_eq!(m.architectural_error_code(), None);
    assert_eq!(m.page_fault_projection(), None);
    assert_eq!(m.vector(), Unknown);
}
