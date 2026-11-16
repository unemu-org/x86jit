//! I80286 values applied through the catalogue (decision-23 §3, xj-50w.11).
//! Intel 80286 and 80287 Programmer's Reference Manual 210498-005 (1987),
//! page images checked:
//! - formed addresses: §6.4 p. 6-6 ("the resulting 24-bit physical
//!   address"), App. C hardware item 2 p. C-3, App. D item 2 p. D-1;
//! - LOCK: App. C item 11 p. C-2, App. B LOCK p. B-68;
//! - coprocessor operand overrun: App. B #MP p. B-9, §5.2 Table 5-2 p. 5-6
//!   and p. 5-7, App. C Table C-1 p. C-1.
//!
//! The 80386 keeps its own values (80386 PRM 230985-001): 32-bit addresses
//! (§14.7 item 18 p. 14-7), #UD for LOCK outside the listed forms (§14.7
//! item 9 p. 14-6) and 12/13 for an operand crossing the segment end
//! (§14.6 Table 14-1 p. 14-5).
use x86jit_core::fault::ErrorCode;
use x86jit_core::{
    Backend, CachedBlock, CpuModel, CpuState, ExceptionDelivery, Exit, FpuModel, IrBlock,
    Knowledge, MemConsistency, Prot, RegionKind, SegmentCache, SegmentId, StepOptions, StepStop,
    Vcpu, Vm, VmConfig,
};

struct NoBackend;
impl Backend for NoBackend {
    fn materialize(
        &self,
        _: &IrBlock,
        _: MemConsistency,
        _: Option<(u64, u64)>,
        _: u64,
    ) -> CachedBlock {
        panic!("Machine execution reached application backend")
    }
}

// RAM covers the first megabyte past 16 MiB, so a 32-bit formed address is
// visible as a distinct location.
const RAM: u64 = 0x100_2000;

fn machine(model: CpuModel, fpu: FpuModel, code: &[u8]) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(RAM), Box::new(NoBackend));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.set_fpu_model(fpu);
    vm.map(0, RAM as usize, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([0, 0x200, 0, 0x300, 0, 0]);
    cpu.cpu.fpu_model = fpu;
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}

fn cache(base: u32, limit: u32, attributes: u16) -> SegmentCache {
    SegmentCache {
        base,
        limit,
        attributes,
        ..SegmentCache::default()
    }
}

fn read(vm: &Vm, addr: u64, len: usize) -> Vec<u8> {
    let mut bytes = vec![0; len];
    vm.read_bytes(addr, &mut bytes).unwrap();
    bytes
}

fn run(vm: &Vm, cpu: &mut Vcpu, steps: u64) -> (u64, String) {
    let run = cpu.run_steps(vm, steps, StepOptions::default()).unwrap();
    (run.steps(), format!("{:?}", run.stop()))
}

fn exception(vm: &Vm, cpu: &mut Vcpu) -> (u8, ErrorCode, u64, Knowledge<u64>, Knowledge<u16>) {
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0, "{:?}", run.stop());
    let StepStop::Exit(exit @ Exit::Exception { addr, vector, .. }) = run.stop() else {
        panic!("{:?}", run.stop());
    };
    let m = exit.fault_metadata().expect("fault evidence");
    assert_eq!(*addr, m.saved_rip());
    (
        *vector,
        m.error_code(),
        m.saved_rip(),
        m.operand_base(),
        m.operand_size(),
    )
}

// §6.4 p. 6-6; App. D item 2 p. D-1: base FFFFF0h plus an offset past 0Fh
// reaches the first megabyte on the 80286 and the seventeenth on the 80386.
#[test]
fn data_addresses_wrap_at_the_model_physical_width() {
    for (model, wrapped) in [(CpuModel::I80286, true), (CpuModel::I80386, false)] {
        // MOV AL,[0020]; MOV BX,[000F]; MOV BYTE [0021],77h
        let (vm, mut cpu) = machine(
            model,
            FpuModel::Present,
            &[
                0xa0, 0x20, 0x00, 0x8b, 0x1e, 0x0f, 0x00, 0xc6, 0x06, 0x21, 0x00, 0x77,
            ],
        );
        cpu.cpu.legacy.cr0 = 1;
        cpu.cpu.legacy.segments[SegmentId::Ds as usize] = cache(0xff_fff0, 0xffff, 0x93);
        for (addr, byte) in [
            (0x10, 0x11),
            (0x100_0010, 0x22),
            (0xff_ffff, 0x33),
            (0x0, 0x44),
            (0x100_0000, 0x55),
        ] {
            vm.write_bytes(addr, &[byte]).unwrap();
        }
        assert_eq!(run(&vm, &mut cpu, 3).0, 3, "{model:?}");
        let low = if wrapped { 0 } else { 0x100_0000 };
        assert_eq!(cpu.cpu.gpr[0] & 0xff, if wrapped { 0x11 } else { 0x22 });
        // The word at FFFFFFh: its second byte follows the bus width.
        assert_eq!(
            cpu.cpu.gpr[3] & 0xffff,
            if wrapped { 0x4433 } else { 0x5533 }
        );
        assert_eq!(read(&vm, low + 0x11, 1), [0x77], "{model:?}");
        let other = if wrapped { 0x100_0011 } else { 0x11 };
        assert_eq!(read(&vm, other, 1), [0], "{model:?}");
    }
}

// The descriptor read, its accessed-bit update and an instruction fetch
// form their addresses on the same bus.
#[test]
fn descriptor_and_fetch_addresses_wrap_at_the_model_physical_width() {
    for (model, wrapped) in [(CpuModel::I80286, true), (CpuModel::I80386, false)] {
        // MOV AX,8; MOV DS,AX; MOV AL,[0]
        let (vm, mut cpu) = machine(
            model,
            FpuModel::Present,
            &[0xb8, 0x08, 0x00, 0x8e, 0xd8, 0xa0, 0x00, 0x00],
        );
        cpu.cpu.legacy.cr0 = 1;
        cpu.cpu.legacy.gdtr.base = 0xff_fff8;
        cpu.cpu.legacy.gdtr.limit = 0x0f;
        // 286 PRM §6.3 Fig. 6-3 p. 6-5: limit, base 23:0, access byte.
        vm.write_bytes(0, &[0xff, 0, 0, 0x50, 0, 0x92, 0, 0])
            .unwrap();
        vm.write_bytes(0x100_0000, &[0xff, 0, 0, 0x60, 0, 0x92, 0, 0])
            .unwrap();
        vm.write_bytes(0x5000, &[0x11]).unwrap();
        vm.write_bytes(0x6000, &[0x22]).unwrap();
        assert_eq!(run(&vm, &mut cpu, 3).0, 3, "{model:?}");
        assert_eq!(cpu.cpu.gpr[0] & 0xff, if wrapped { 0x11 } else { 0x22 });
        assert_eq!(read(&vm, 5, 1), [if wrapped { 0x93 } else { 0x92 }]);
        assert_eq!(
            read(&vm, 0x100_0005, 1),
            [if wrapped { 0x92 } else { 0x93 }]
        );

        // CS base FFFF00h, IP 0100h: MOV AL,imm8 at the formed address.
        let (vm, mut cpu) = machine(model, FpuModel::Present, &[]);
        cpu.cpu.legacy.cr0 = 1;
        cpu.cpu.legacy.segments[SegmentId::Cs as usize] = cache(0xff_ff00, 0xffff, 0x9b);
        cpu.cpu.rip = 0x100;
        vm.write_bytes(0, &[0xb0, 0x11]).unwrap();
        vm.write_bytes(0x100_0000, &[0xb0, 0x22]).unwrap();
        assert_eq!(run(&vm, &mut cpu, 1).0, 1, "{model:?}");
        assert_eq!(cpu.cpu.gpr[0] & 0xff, if wrapped { 0x11 } else { 0x22 });
    }
}

// A coprocessor operand split by the end of the 24-bit space: FILD QWORD
// [0004] through a real-mode cache based at FFFFF8h, then FISTP to ES:0100.
#[test]
fn coprocessor_transfers_wrap_at_the_model_physical_width() {
    for (model, wrapped) in [(CpuModel::I80286, true), (CpuModel::I80386, false)] {
        let (vm, mut cpu) = machine(
            model,
            FpuModel::Present,
            &[0xdf, 0x2e, 0x04, 0x00, 0x26, 0xdf, 0x3e, 0x00, 0x01],
        );
        cpu.cpu.legacy.segments[SegmentId::Ds as usize] = cache(0xff_fff8, 0xffff, 0x93);
        vm.write_bytes(0xff_fffc, &[1, 2, 3, 4]).unwrap();
        vm.write_bytes(0, &[5, 6, 7, 0x08]).unwrap();
        vm.write_bytes(0x100_0000, &[9, 10, 11, 0x0c]).unwrap();
        assert_eq!(run(&vm, &mut cpu, 2).0, 2, "{model:?}");
        let high = if wrapped {
            [5, 6, 7, 8]
        } else {
            [9, 10, 11, 12]
        };
        assert_eq!(read(&vm, 0x100, 8), [&[1, 2, 3, 4][..], &high].concat());
    }
}

const LOCKED_MOVES: &[&[u8]] = &[
    &[0x88, 0x07],             // MOV [BX],AL
    &[0x8a, 0x07],             // MOV AL,[BX]
    &[0xb8, 0x34, 0x12],       // MOV AX,1234h
    &[0xc7, 0x07, 0x78, 0x56], // MOV WORD [BX],5678h
    &[0xa3, 0x00, 0x02],       // MOV [0200],AX
    &[0xa0, 0x21, 0x00],       // MOV AL,[0021]
    &[0x8c, 0x07],             // MOV [BX],ES
    &[0x8e, 0xc0],             // MOV ES,AX
];
const LOCKED_STRINGS: &[&[u8]] = &[
    &[0xa5],       // MOVSW
    &[0xf3, 0xa4], // REP MOVSB
    &[0x6c],       // INSB
    &[0xf3, 0x6d], // REP INSW
    &[0x6e],       // OUTSB
    &[0xf3, 0x6f], // REP OUTSW
];

fn observe(model: CpuModel, code: &[u8], checked: bool) -> (u64, String, CpuState, Vec<u8>) {
    let (vm, mut cpu) = machine(model, FpuModel::Present, code);
    cpu.cpu.gpr[0] = 0x9a5a;
    cpu.cpu.gpr[1] = 3;
    cpu.cpu.gpr[2] = 0x60;
    cpu.cpu.gpr[3] = 0x10;
    cpu.cpu.gpr[6] = 0x20;
    cpu.cpu.gpr[7] = 0x30;
    vm.write_bytes(0x2000, &(0..0x80).collect::<Vec<u8>>())
        .unwrap();
    if checked {
        // FS attributes withhold the Direct proof; FS is not addressed.
        cpu.cpu.legacy.segments[SegmentId::Fs as usize].attributes = 0x92;
    }
    let (steps, stop) = run(&vm, &mut cpu, 8);
    (steps, stop, cpu.cpu.clone(), read(&vm, 0, 0x3000))
}

// App. C item 11 p. C-2: "LOCK should only be used with the XCHG, MOV,
// MOVS, INS, and OUTS instructions". A locked MOV executes through the
// engine's atomic path with the unprefixed form's result, one byte longer.
#[test]
fn lock_mov_executes_as_the_unprefixed_move() {
    for checked in [false, true] {
        for code in LOCKED_MOVES {
            let plain = observe(CpuModel::I80286, code, checked);
            let locked_code = [&[0xf0][..], code].concat();
            let mut locked = observe(CpuModel::I80286, &locked_code, checked);
            assert_eq!(locked.0, plain.0, "{code:02x?} {checked}");
            assert_eq!(locked.2.rip, plain.2.rip + 1, "{code:02x?}");
            locked.2.rip = plain.2.rip;
            assert_eq!(locked.1, plain.1, "{code:02x?} {checked}");
            assert_eq!(locked.2.gpr, plain.2.gpr, "{code:02x?} {checked}");
            assert_eq!(locked.2.rflags(), plain.2.rflags(), "{code:02x?}");
            assert_eq!(
                [locked.2.cs, locked.2.ds, locked.2.es, locked.2.ss],
                [plain.2.cs, plain.2.ds, plain.2.es, plain.2.ss]
            );
            assert_eq!(locked.2.legacy, plain.2.legacy, "{code:02x?} {checked}");
            // Memory outside the two code images.
            assert_eq!(locked.3[..0x1000], plain.3[..0x1000], "{code:02x?}");
            assert_eq!(locked.3[0x1010..], plain.3[0x1010..], "{code:02x?}");
            assert!(!plain.1.contains("Exception"), "{code:02x?}: {}", plain.1);
        }
    }
}

// The atomic path hands a locked device operand to the embedder as one
// instruction, as for every locked operand on this path, in Direct and
// Checked contexts; the unprefixed store is an ordinary device write. This
// tells the atomic path from a LOCK that is merely stripped. A locked load
// is one read: a device load reaches the embedder as the unprefixed one.
// Bead xj-ir6.33 changes every locked device operand to one MmioRead and one
// MmioWrite of the operand size; this witness pins the handoff until then.
#[test]
fn lock_mov_to_a_device_is_a_typed_handoff() {
    for checked in [false, true] {
        for (code, store) in [
            (&[0x88, 0x07][..], true),         // MOV [BX],AL
            (&[0xc7, 0x07, 0x78, 0x56], true), // MOV WORD [BX],5678h
            (&[0xa2, 0x10, 0x00], true),       // MOV [0010],AL
            (&[0x8c, 0x07], true),             // MOV [BX],ES
            (&[0x8a, 0x07], false),            // MOV AL,[BX]
        ] {
            for locked in [false, true] {
                let code = if locked {
                    [&[0xf0][..], code].concat()
                } else {
                    code.to_vec()
                };
                let mut vm = Vm::with_backend(VmConfig::flat(0x2_0000), Box::new(NoBackend));
                vm.set_cpu_model(CpuModel::I80286);
                vm.set_exception_delivery(ExceptionDelivery::Outward);
                vm.map(0, 0x1_0000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.map(0x1_0000, 0x1000, Prot::RW, RegionKind::Trap)
                    .unwrap();
                vm.write_bytes(0x1000, &code).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.cpu = CpuState::legacy_real_entry([0, 0x1000, 0, 0x300, 0, 0]);
                cpu.cpu.rip = 0x1000;
                cpu.cpu.gpr[0] = 0x5a;
                cpu.cpu.gpr[3] = 0x10;
                if checked {
                    cpu.cpu.legacy.segments[SegmentId::Fs as usize].attributes = 0x92;
                }
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                let label = format!("{code:02x?} checked={checked}");
                assert_eq!(run.steps(), 0, "{label}");
                match run.stop() {
                    StepStop::Exit(Exit::UnknownInstruction {
                        addr, bytes, len, ..
                    }) if locked && store => {
                        assert_eq!((*addr, usize::from(*len)), (0x1000, code.len()), "{label}");
                        assert_eq!(&bytes[..code.len()], &code[..], "{label}");
                    }
                    StepStop::Exit(Exit::MmioWrite { addr, .. }) if !locked && store => {
                        assert_eq!(*addr, 0x1_0010, "{label}");
                    }
                    StepStop::Exit(Exit::MmioRead { addr, size, .. }) if !store => {
                        assert_eq!((*addr, *size), (0x1_0010, 1), "{label}");
                    }
                    other => panic!("{label}: {other:?}"),
                }
                assert_eq!(cpu.cpu.rip, 0x1000, "{label}");
            }
        }
    }
}

// The locked-operand rule of xj-ir6.12: a locked operand lies in one region.
// A LOCK MOV whose word spans two adjacent regions (RAM and RAM, or RAM and
// ROM) stops with UnmappedMemory before any byte moves, where the unprefixed
// MOV completes (review probe zz_lock_mov_over_adjacent_regions).
#[test]
fn lock_mov_keeps_the_single_region_rule() {
    for second in [RegionKind::Ram, RegionKind::Rom] {
        for checked in [false, true] {
            for locked in [false, true] {
                let mut vm = Vm::with_backend(VmConfig::flat(0x2_0000), Box::new(NoBackend));
                vm.set_cpu_model(CpuModel::I80286);
                vm.set_exception_delivery(ExceptionDelivery::Outward);
                vm.map(0, 0x8000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.map(0x8000, 0x8000, Prot::RWX, second).unwrap();
                let code: &[u8] = if locked {
                    &[0xf0, 0x89, 0x07]
                } else {
                    &[0x89, 0x07]
                };
                vm.write_bytes(0x1000, code).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.cpu = CpuState::legacy_real_entry([0, 0, 0, 0x300, 0, 0]);
                cpu.cpu.rip = 0x1000;
                cpu.cpu.gpr[0] = 0x5a5a;
                cpu.cpu.gpr[3] = 0x7fff;
                cpu.cpu.gpr[4] = 0x7000;
                if checked {
                    cpu.cpu.legacy.segments[SegmentId::Fs as usize].attributes = 0x92;
                }
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                let label = format!("{second:?} checked={checked} locked={locked}");
                if locked {
                    assert_eq!(run.steps(), 0, "{label}");
                    assert!(
                        matches!(
                            run.stop(),
                            StepStop::Exit(Exit::UnmappedMemory {
                                addr: 0x7fff,
                                access: x86jit_core::AccessKind::Write,
                                ..
                            })
                        ),
                        "{label}: {:?}",
                        run.stop()
                    );
                    assert_eq!(read(&vm, 0x7fff, 1), [0], "{label}");
                } else {
                    assert_eq!(run.steps(), 1, "{label}");
                    assert_eq!(read(&vm, 0x7fff, 1), [0x5a], "{label}");
                }
            }
        }
    }
}

// A locked word split at FFFFFFh or by the A20 gate takes the locked-operand
// fallback and writes what the unprefixed MOV writes; a plain word read whose
// second byte wraps onto a device reads that byte at physical 0 (review probe
// zz_split_metadata_and_locked_split).
#[test]
fn lock_mov_split_by_the_bus_matches_the_unprefixed_move() {
    let mut vm = Vm::with_backend(VmConfig::flat(0x100_2000), Box::new(NoBackend));
    vm.set_cpu_model(CpuModel::I80286);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.map(0, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.map(0x1000, 0x100_1000, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.write_bytes(0x2000, &[0x8b, 0x1e, 0x0f, 0x00]).unwrap(); // MOV BX,[000F]
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([0x200, 0, 0, 0x300, 0, 0]);
    cpu.cpu.rip = 0;
    cpu.cpu.gpr[4] = 0x8000;
    cpu.cpu.legacy.segments[SegmentId::Ds as usize] = cache(0xff_fff0, 0xffff, 0x93);
    let first = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert!(
        matches!(
            first.stop(),
            StepStop::Exit(Exit::MmioRead {
                addr: 0,
                size: 1,
                ..
            })
        ),
        "{:?}",
        first.stop()
    );
    for (a20, base) in [(true, 0xff_fff0u32), (false, 0xf_fff0)] {
        let mut memories = Vec::new();
        for locked in [false, true] {
            let mut vm = Vm::with_backend(VmConfig::flat(0x100_2000), Box::new(NoBackend));
            vm.set_cpu_model(CpuModel::I80286);
            vm.set_exception_delivery(ExceptionDelivery::Outward);
            vm.map(0, 0x100_2000, Prot::RWX, RegionKind::Ram).unwrap();
            let code: &[u8] = if locked {
                &[0xf0, 0x89, 0x07]
            } else {
                &[0x89, 0x07]
            };
            vm.write_bytes(0x2000, code).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu = CpuState::legacy_real_entry([0x200, 0, 0, 0x300, 0, 0]);
            cpu.cpu.rip = 0;
            cpu.cpu.gpr[0] = 0xbbaa;
            cpu.cpu.gpr[3] = 0x0f;
            cpu.cpu.gpr[4] = 0x8000;
            cpu.cpu.legacy.segments[SegmentId::Ds as usize] = cache(base, 0xffff, 0x93);
            cpu.set_a20_enabled(a20);
            assert_eq!(run(&vm, &mut cpu, 1).0, 1, "a20={a20} locked={locked}");
            memories.push(
                [0xf_ffffu64, 0xff_ffff, 0, 0x10_0000, 0x100_0000].map(|a| read(&vm, a, 1)[0]),
            );
        }
        assert_eq!(memories[0], memories[1], "a20={a20}");
        let expected = if a20 {
            [0, 0xaa, 0xbb, 0, 0]
        } else {
            [0xaa, 0, 0xbb, 0, 0]
        };
        assert_eq!(memories[0], expected, "a20={a20}");
    }
}

// A locked MOVS, INS or OUTS would hold the bus across its transfers, which
// the engine cannot do: a typed handoff before anything commits (no #UD).
#[test]
fn lock_string_forms_are_a_typed_handoff() {
    for checked in [false, true] {
        for code in LOCKED_STRINGS {
            let plain = observe(CpuModel::I80286, code, checked);
            assert!(!plain.1.contains("Exception"), "{code:02x?}: {}", plain.1);
            let locked_code = [&[0xf0][..], code].concat();
            let (vm, mut cpu) = machine(CpuModel::I80286, FpuModel::Present, &locked_code);
            cpu.cpu.gpr[1] = 3;
            cpu.cpu.gpr[2] = 0x60;
            cpu.cpu.gpr[6] = 0x20;
            cpu.cpu.gpr[7] = 0x30;
            if checked {
                cpu.cpu.legacy.segments[SegmentId::Fs as usize].attributes = 0x92;
            }
            let before = cpu.cpu.clone();
            let memory = read(&vm, 0, 0x3000);
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            let StepStop::Exit(Exit::UnknownInstruction {
                addr, bytes, len, ..
            }) = run.stop()
            else {
                panic!("{code:02x?}: {:?}", run.stop());
            };
            // Both segment classes report the instruction's bytes.
            assert_eq!(*addr, 0x1000, "{code:02x?}");
            assert_eq!(usize::from(*len), locked_code.len(), "{code:02x?}");
            assert_eq!(&bytes[..locked_code.len()], &locked_code[..]);
            assert_eq!(run.steps(), 0);
            assert_eq!(cpu.cpu.gpr, before.gpr, "{code:02x?}");
            assert_eq!(cpu.cpu.rip, 0x1000);
            assert_eq!(read(&vm, 0, 0x3000), memory, "{code:02x?}");
        }
    }
}

#[test]
fn lock_keeps_its_other_dispositions() {
    // 80386 PRM §14.7 item 9 p. 14-6: #UD before any non-listed instruction.
    for code in LOCKED_MOVES.iter().chain(LOCKED_STRINGS) {
        let locked = [&[0xf0][..], code].concat();
        let (vm, mut cpu) = machine(CpuModel::I80386, FpuModel::Present, &locked);
        assert_eq!(exception(&vm, &mut cpu).0, 6, "{code:02x?}");
    }
    // The 80286 quirk names data movement only: other register forms and
    // string forms keep the strict decoder's #UD.
    for code in [&[0xf0, 0x00, 0xec][..], &[0xf0, 0xaa], &[0xf0, 0x87, 0xc3]] {
        let (vm, mut cpu) = machine(CpuModel::I80286, FpuModel::Present, code);
        assert_eq!(exception(&vm, &mut cpu).0, 6, "{code:02x?}");
    }
    // A locked read-modify-write form still executes.
    let (vm, mut cpu) = machine(CpuModel::I80286, FpuModel::Present, &[0xf0, 0x00, 0x07]);
    assert_eq!(run(&vm, &mut cpu, 1).0, 1);
}

// (bytes, operand offset, operand size, 80286 vector, 80386 vector);
// DS-relative unless the ModRM names BP. None executes.
type OverrunCase = (&'static [u8], u64, u16, Option<u8>, Option<u8>);
const OVERRUN: &[OverrunCase] = &[
    (&[0xdd, 0x06, 0xfc, 0xff], 0xfffc, 8, Some(9), Some(13)), // FLD QWORD [FFFC]
    (&[0xdd, 0x06, 0xf9, 0xff], 0xfff9, 8, Some(9), Some(13)), // FLD QWORD [FFF9]
    (&[0xdd, 0x06, 0xf8, 0xff], 0xfff8, 8, None, None),        // FLD QWORD [FFF8]
    (&[0xd9, 0x06, 0xfe, 0xff], 0xfffe, 4, Some(9), Some(13)), // FLD DWORD [FFFE]
    (&[0xd9, 0x06, 0xff, 0xff], 0xffff, 4, Some(9), Some(13)), // FLD DWORD [FFFF]
    (&[0xdf, 0x06, 0xff, 0xff], 0xffff, 2, Some(13), Some(13)), // FILD WORD [FFFF]
    (&[0xdf, 0x06, 0xfe, 0xff], 0xfffe, 2, None, None),        // FILD WORD [FFFE]
    (&[0xd9, 0x2e, 0xff, 0xff], 0xffff, 2, Some(13), Some(13)), // FLDCW [FFFF]
    (&[0xdd, 0x1e, 0xfc, 0xff], 0xfffc, 8, Some(9), Some(13)), // FSTP QWORD [FFFC]
    (&[0xdb, 0x2e, 0xf8, 0xff], 0xfff8, 10, Some(9), Some(13)), // FLD TBYTE [FFF8]
    (&[0xd9, 0x26, 0xf8, 0xff], 0xfff8, 14, Some(9), Some(13)), // FLDENV [FFF8]
    (&[0xdd, 0x36, 0xb0, 0xff], 0xffb0, 94, Some(9), Some(13)), // FNSAVE [FFB0]
    (
        &[0x26, 0xdd, 0x06, 0xfc, 0xff],
        0xfffc,
        8,
        Some(9),
        Some(13),
    ), // ES: FLD
    (&[0xdd, 0x46, 0xfc], 0xfffc, 8, Some(9), Some(12)),       // FLD QWORD [BP-4]
];

// App. B #MP p. B-9: an operand larger than one word with its second or
// later words outside the segment raises interrupt 9, no error code; a
// one-word operand takes exception 13. §5.2 p. 5-7: the saved CS:IP is the
// instruction's first byte. The coprocessor image and memory are unchanged.
#[test]
fn coprocessor_operand_overrun_raises_the_model_vector() {
    for (model, column) in [(CpuModel::I80286, 0), (CpuModel::I80386, 1)] {
        for checked in [false, true] {
            for &(code, offset, size, v286, v386) in OVERRUN {
                let vector = if column == 0 { v286 } else { v386 };
                let (vm, mut cpu) = machine(model, FpuModel::Present, code);
                cpu.cpu.gpr[5] = 0;
                if checked {
                    cpu.cpu.legacy.segments[SegmentId::Fs as usize].attributes = 0x92;
                }
                cpu.cpu.fpu_fip = 0x2345;
                cpu.cpu.fpu_fdp = 0x4567;
                cpu.cpu.fpu_fcs = 0x1234;
                cpu.cpu.fpu_fds = 0x2345;
                cpu.cpu.fpu_fop = 0x523;
                let image = |c: &CpuState| {
                    (
                        c.fpr,
                        c.fpu_sw,
                        c.fpu_cw,
                        c.fpu_top,
                        c.fpu_empty,
                        [c.fpu_fip, c.fpu_fdp],
                        [c.fpu_fcs, c.fpu_fds, c.fpu_fop],
                    )
                };
                let fpu = image(&cpu.cpu);
                let memory = read(&vm, 0, 0x2_0000);
                let Some(vector) = vector else {
                    assert_eq!(run(&vm, &mut cpu, 1).0, 1, "{model:?} {code:02x?}");
                    continue;
                };
                let (actual, error_code, saved, base, bytes) = exception(&vm, &mut cpu);
                let label = format!("{model:?} {code:02x?} checked={checked}");
                assert_eq!(actual, vector, "{label}");
                assert_eq!(error_code, ErrorCode::NoCode, "{label}");
                assert_eq!(saved, 0x1000, "{label}");
                assert_eq!(
                    (base, bytes),
                    (Knowledge::Known(offset), Knowledge::Known(size))
                );
                assert_eq!(image(&cpu.cpu), fpu, "{label}");
                assert_eq!(read(&vm, 0, 0x2_0000), memory, "{label}");
                assert_eq!(cpu.retired_instructions(), 0);
            }
        }
    }
}

// Decision-25: with an empty socket the CPU's whole-operand check stays.
#[test]
fn an_empty_socket_keeps_the_cpu_operand_check() {
    for &(code, _, size, v286, _) in OVERRUN {
        let (vm, mut cpu) = machine(CpuModel::I80286, FpuModel::None, code);
        cpu.cpu.gpr[5] = 0;
        let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        match (v286, result.stop()) {
            (None, _) => assert_eq!(result.steps(), 1, "{code:02x?}"),
            (Some(_), StepStop::Exit(Exit::Exception { vector, .. })) => {
                assert_eq!(*vector, 13, "{code:02x?} {size}")
            }
            (Some(_), other) => panic!("{code:02x?}: {other:?}"),
        }
    }
}

// A real-mode cache with limit 0FFFh (the Checked class): an operand whose
// start lies outside the limit keeps exception 13, whatever its segment;
// one whose later words lie outside raises interrupt 9. Coprocessor
// instructions in protected mode remain a typed refusal on this path.
#[test]
fn coprocessor_operands_check_their_start_before_their_extent() {
    for (code, vector) in [
        (&[0xdd, 0x06, 0xfc, 0x0f][..], 9),
        (&[0xdd, 0x06, 0x00, 0x10], 13),
        (&[0xdd, 0x46, 0x00], 9),
        (&[0x36, 0xdd, 0x06, 0x00, 0x10], 13),
        (&[0xdd, 0x06, 0xf8, 0x0f], 0),
    ] {
        let (vm, mut cpu) = machine(CpuModel::I80286, FpuModel::Present, code);
        cpu.cpu.legacy.segments[SegmentId::Ds as usize] = cache(0x4000, 0x0fff, 0x93);
        cpu.cpu.legacy.segments[SegmentId::Ss as usize] = cache(0x6000, 0x0fff, 0x93);
        cpu.cpu.gpr[5] = 0x0ffc;
        if vector == 0 {
            assert_eq!(run(&vm, &mut cpu, 1).0, 1, "{code:02x?}");
            continue;
        }
        let (actual, error, saved, ..) = exception(&vm, &mut cpu);
        assert_eq!((actual, error, saved), (vector, ErrorCode::NoCode, 0x1000));
    }
}

// Real-mode delivery through the IVT: interrupt 9 saves CS:IP of the
// first byte, prefixes included (App. C item 3 p. C-1; §5.2 p. 5-7).
#[test]
fn interrupt_9_is_delivered_through_the_real_mode_table() {
    let mut vm = Vm::with_backend(VmConfig::flat(0x2_0000), Box::new(NoBackend));
    vm.set_cpu_model(CpuModel::I80286);
    vm.set_exception_delivery(ExceptionDelivery::InGuest);
    vm.map(0, 0x2_0000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([0, 0x200, 0, 0x300, 0, 0]);
    cpu.cpu.fpu_model = FpuModel::Present;
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    vm.write_bytes(0x1000, &[0x26, 0xdd, 0x06, 0xfc, 0xff])
        .unwrap();
    vm.write_bytes(9 * 4, &[0x00, 0x05, 0x00, 0x00]).unwrap();
    vm.write_bytes(0x500, &[0xf4]).unwrap();
    let run = cpu.run_steps(&vm, 4, StepOptions::default()).unwrap();
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
        "{:?}",
        run.stop()
    );
    assert_eq!(cpu.cpu.rip, 0x501);
    assert_eq!(cpu.cpu.gpr[4] & 0xffff, 0x7ffa);
    let frame = read(&vm, 0x3000 + 0x7ffa, 6);
    assert_eq!(&frame[..4], &[0x00, 0x10, 0x00, 0x00]);
}

// App. B #MF p. B-9 tests the ERROR pin "at the beginning of most floating
// point instructions"; #MP p. B-9 arises "during the execution of the numeric
// instruction by the 80287". A pending unmasked error therefore raises 16
// before an overrunning operand can raise 9, in both segment classes (review
// probe zz_pending_error_and_overrun). FNSAVE, which tests no ERROR, keeps 9.
#[test]
fn a_pending_numeric_error_precedes_the_overrun() {
    for checked in [false, true] {
        for (code, pending, vector) in [
            (&[0xdd, 0x06, 0xfc, 0xff][..], true, 16), // FLD QWORD [FFFC]
            (&[0xdd, 0x06, 0x00, 0x00], true, 16),     // FLD QWORD [0000]
            (&[0xdd, 0x06, 0xfc, 0xff], false, 9),
            (&[0xdd, 0x36, 0xb0, 0xff], true, 9), // FNSAVE [FFB0]
        ] {
            let (vm, mut cpu) = machine(CpuModel::I80286, FpuModel::Present, code);
            if checked {
                cpu.cpu.legacy.segments[SegmentId::Fs as usize].attributes = 0x92;
            }
            if pending {
                // IE unmasked; IE and ES set in the status word.
                cpu.cpu.fpu_cw = 0x037e;
                cpu.cpu.fpu_sw = 0x0081;
            }
            let label = format!("{code:02x?} pending={pending} checked={checked}");
            let (actual, error, saved, ..) = exception(&vm, &mut cpu);
            assert_eq!(
                (actual, error, saved),
                (vector, ErrorCode::NoCode, 0x1000),
                "{label}"
            );
        }
    }
}

// The 80286 IDT, TSS, stack, string and port-input transfers form their
// addresses on the 24-bit bus too (§6.4 p. 6-6); the 80386 keeps 32 bits.
fn protected(vm: &Vm, cpu: &mut Vcpu) {
    // 286 PRM §6.3 Fig. 6-3 p. 6-5: code 8 and data 10h, base 0.
    vm.write_bytes(0x2008, &[0xff, 0xff, 0, 0, 0, 0x9b, 0, 0])
        .unwrap();
    vm.write_bytes(0x2010, &[0xff, 0xff, 0, 0, 0, 0x93, 0, 0])
        .unwrap();
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.gdtr.base = 0x2000;
    cpu.cpu.legacy.gdtr.limit = 0x3f;
    cpu.cpu.cs = 8;
    cpu.cpu.legacy.segments[SegmentId::Cs as usize] = cache(0, 0xffff, 0x9b);
    cpu.cpu.ss = 0x10;
    cpu.cpu.legacy.segments[SegmentId::Ss as usize] = cache(0, 0xffff, 0x93);
    cpu.cpu.ds = 0x10;
    cpu.cpu.legacy.segments[SegmentId::Ds as usize] = cache(0, 0xffff, 0x93);
    cpu.cpu.es = 0x10;
    cpu.cpu.legacy.segments[SegmentId::Es as usize] = cache(0, 0xffff, 0x93);
}

#[test]
fn idt_and_tss_addresses_wrap_at_the_model_physical_width() {
    for (model, wrapped) in [(CpuModel::I80286, true), (CpuModel::I80386, false)] {
        // INT 1 through an IDT based at FFFFF8h: gate 1 lies at 1000000h.
        let (vm, mut cpu) = machine(model, FpuModel::Present, &[0xcd, 0x01]);
        protected(&vm, &mut cpu);
        cpu.cpu.legacy.idtr.base = 0xff_fff8;
        cpu.cpu.legacy.idtr.limit = 0x7f;
        // 286 PRM §9.4 Fig. 9-3 p. 9-4: interrupt gates to 8:0500 and 8:0600.
        vm.write_bytes(0, &[0x00, 0x05, 0x08, 0x00, 0, 0x86, 0, 0])
            .unwrap();
        vm.write_bytes(0x100_0000, &[0x00, 0x06, 0x08, 0x00, 0, 0x86, 0, 0])
            .unwrap();
        assert_eq!(run(&vm, &mut cpu, 1).0, 1, "{model:?}");
        assert_eq!(
            cpu.cpu.rip,
            if wrapped { 0x500 } else { 0x600 },
            "{model:?}"
        );

        // JMP to a 286 TSS based at FFFFF0h: its fields from offset 10h on
        // lie past FFFFFFh (286 PRM §8.2 Fig. 8-1 p. 8-2).
        let (vm, mut cpu) = machine(
            model,
            FpuModel::Present,
            &[0xb8, 0x18, 0x00, 0x0f, 0x00, 0xd8, 0xea, 0, 0, 0x20, 0],
        );
        protected(&vm, &mut cpu);
        vm.write_bytes(0x2018, &[0x2b, 0, 0x00, 0x40, 0x00, 0x81, 0, 0])
            .unwrap();
        vm.write_bytes(0x2020, &[0x2b, 0, 0xf0, 0xff, 0xff, 0x81, 0, 0])
            .unwrap();
        vm.write_bytes(0xff_fffe, &0x1800u16.to_le_bytes()).unwrap();
        for (base, sp) in [(0u64, 0x9000u16), (0x100_0000, 0x7000)] {
            for (offset, value) in [
                (0x00, 2),
                (0x0a, sp),
                (0x12, 0x10),
                (0x14, 8),
                (0x16, 0x10),
                (0x18, 0x10),
            ] {
                vm.write_bytes(base + offset, &value.to_le_bytes()).unwrap();
            }
        }
        assert_eq!(run(&vm, &mut cpu, 3).0, 3, "{model:?}");
        assert_eq!(cpu.cpu.rip, 0x1800, "{model:?}");
        assert_eq!(
            cpu.cpu.gpr[4] & 0xffff,
            if wrapped { 0x9000 } else { 0x7000 }
        );
    }
}

#[test]
fn stack_string_and_port_input_addresses_wrap_at_the_model_physical_width() {
    for (model, wrapped) in [(CpuModel::I80286, true), (CpuModel::I80386, false)] {
        let low = |a: u64| if wrapped { a } else { a + 0x100_0000 };
        // PUSH AX through a real-mode SS cache based at FFFFF0h, SP=20h.
        let (vm, mut cpu) = machine(model, FpuModel::Present, &[0x50]);
        cpu.cpu.legacy.segments[SegmentId::Ss as usize] = cache(0xff_fff0, 0xffff, 0x93);
        cpu.cpu.gpr[0] = 0x6655;
        cpu.cpu.gpr[4] = 0x20;
        assert_eq!(run(&vm, &mut cpu, 1).0, 1, "{model:?}");
        assert_eq!(read(&vm, low(0xe), 2), [0x55, 0x66], "{model:?}");

        // MOVSB from a DS cache based at FFFFF0h, SI=20h, to ES:0500.
        let (vm, mut cpu) = machine(model, FpuModel::Present, &[0xa4]);
        cpu.cpu.legacy.segments[SegmentId::Ds as usize] = cache(0xff_fff0, 0xffff, 0x93);
        vm.write_bytes(0x10, &[0x11]).unwrap();
        vm.write_bytes(0x100_0010, &[0x22]).unwrap();
        cpu.cpu.gpr[6] = 0x20;
        cpu.cpu.gpr[7] = 0x500;
        assert_eq!(run(&vm, &mut cpu, 1).0, 1, "{model:?}");
        assert_eq!(read(&vm, 0x500, 1), [if wrapped { 0x11 } else { 0x22 }]);

        // INSB into an ES cache based at FFFFF0h, DI=20h: the store that
        // completes the port input goes to the formed address.
        let (vm, mut cpu) = machine(model, FpuModel::Present, &[0x6c]);
        cpu.cpu.legacy.segments[SegmentId::Es as usize] = cache(0xff_fff0, 0xffff, 0x93);
        cpu.cpu.gpr[2] = 0x60;
        cpu.cpu.gpr[7] = 0x20;
        let first = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(
                first.stop(),
                StepStop::Exit(Exit::PortIo { port: 0x60, .. })
            ),
            "{:?}",
            first.stop()
        );
        cpu.complete_port_in(0x5a);
        assert_eq!(run(&vm, &mut cpu, 1).0, 1, "{model:?}");
        assert_eq!(read(&vm, low(0x10), 1), [0x5a], "{model:?}");
        assert_eq!(read(&vm, if wrapped { 0x100_0010 } else { 0x10 }, 1), [0]);
    }
}

// ENTER's display evidence (`mmio_evidence`) re-forms the display's bus
// address from its retry identity. An ENTER 8,3 through an SS cache based at
// FFFFF0h reads its first display word on a device at SS:0010, which lies
// past FFFFFFh, then faults on the second, at SS:000E (FFFFFEh, unmapped).
// The fault reports the device answer at the wrapped address on the 80286
// and at 1000000h on the 80386.
#[test]
fn enter_display_evidence_wraps_at_the_model_physical_width() {
    for (model, display) in [(CpuModel::I80286, 0u64), (CpuModel::I80386, 0x100_0000)] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x100_3000), Box::new(NoBackend));
        vm.set_cpu_model(model);
        vm.set_exception_delivery(ExceptionDelivery::Outward);
        vm.map(0, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.map(0x1000, 0xffe000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.map(0x100_0000, 0x1000, Prot::RW, RegionKind::Trap)
            .unwrap();
        vm.map(0x100_1000, 0x2000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.write_bytes(0x2000, &[0xc8, 0x08, 0x00, 0x03]).unwrap(); // ENTER 8,3
        let mut cpu = vm.new_vcpu();
        cpu.cpu = CpuState::legacy_real_entry([0x200, 0, 0, 0x300, 0, 0]);
        cpu.cpu.rip = 0;
        cpu.cpu.legacy.segments[SegmentId::Ss as usize] = cache(0xff_fff0, 0xffff, 0x93);
        cpu.cpu.gpr[5] = 0x12; // BP: displays at SS:0010 and SS:000E.
        cpu.cpu.gpr[4] = 0x1012; // SP: the frame pushes go to RAM.
        let first = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(first.stop(), StepStop::Exit(Exit::MmioRead { addr, .. }) if *addr == display),
            "{model:?}: {:?}",
            first.stop()
        );
        cpu.complete_mmio_read(0xbeef);
        let second = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        let StepStop::Exit(exit @ Exit::UnmappedMemory { .. }) = second.stop() else {
            panic!("{model:?}: {:?}", second.stop());
        };
        let acknowledged = exit
            .fault_metadata()
            .expect("fault evidence")
            .progress()
            .acknowledged_transfers()
            .to_vec();
        assert!(
            acknowledged.contains(&(display, 0xbeef)),
            "{model:?}: {acknowledged:x?}"
        );
    }
}

// Review probes (xj-50w.11 review), adopted with their findings as values.

// Ten or more prefixes followed by an opcode the 80286 lacks: Direct and
// Checked both raise #GP (App. B #GP p. B-9); nine leave the opcode inside
// the limit (#UD, App. B #UD p. B-12), except 0F 20, whose unavailable byte
// is the eleventh.
#[test]
fn overlong_run_ending_in_an_unavailable_opcode() {
    for tail in [
        &[0x66u8, 0x90][..],
        &[0x64, 0x90],
        &[0x0f, 0x20, 0xc0],
        &[0x67, 0x90],
    ] {
        for run_len in [9usize, 10, 11] {
            let mut code = vec![0x26; run_len];
            code.extend_from_slice(tail);
            let expected = if run_len >= 10 || tail[0] == 0x0f {
                13
            } else {
                6
            };
            for checked in [false, true] {
                let (vm, mut cpu) = machine(CpuModel::I80286, FpuModel::Present, &code);
                if checked {
                    cpu.cpu.legacy.segments[SegmentId::Fs as usize].attributes = 0x92;
                }
                let (vector, _, saved, ..) = exception(&vm, &mut cpu);
                assert_eq!((vector, saved), (expected, 0x1000), "{code:02x?} {checked}");
            }
        }
    }
}

// Protected-mode coprocessor instructions with a coprocessor present remain a
// typed refusal before any operand test, under both delivery options, so
// interrupt 9 arises in real mode only.
#[test]
fn protected_mode_coprocessor_instructions_stay_refused() {
    for delivery in [ExceptionDelivery::Outward, ExceptionDelivery::InGuest] {
        for code in [
            &[0xdd, 0x06, 0xfc, 0xff][..],   // FLD QWORD [FFFC]
            &[0xdb, 0x2e, 0xf8, 0xff],       // FLD TBYTE [FFF8]
            &[0xd9, 0x26, 0xf8, 0xff],       // FLDENV [FFF8]
            &[0xdd, 0x06, 0x00, 0x00],       // FLD QWORD [0000]
            &[0xdd, 0x06, 0x00, 0x00, 0x9b], // FLD QWORD [0000]; WAIT
        ] {
            let mut vm = Vm::with_backend(VmConfig::flat(0x2_0000), Box::new(NoBackend));
            vm.set_cpu_model(CpuModel::I80286);
            vm.set_exception_delivery(delivery);
            vm.map(0, 0x2_0000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1000, code).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu = CpuState::legacy_real_entry([0, 0x200, 0, 0x300, 0, 0]);
            cpu.cpu.rip = 0x1000;
            cpu.cpu.gpr[4] = 0x8000;
            protected(&vm, &mut cpu);
            cpu.cpu.legacy.idtr.base = 0x4000;
            cpu.cpu.legacy.idtr.limit = 0x7ff;
            vm.write_bytes(0x4000 + 9 * 8, &[0x00, 0x18, 0x08, 0x00, 0, 0x86, 0, 0])
                .unwrap();
            vm.write_bytes(0x1800, &[0xf4]).unwrap();
            let run = cpu.run_steps(&vm, 3, StepOptions::default()).unwrap();
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnsupportedSystemContext { addr: 0x1000, .. })
                ),
                "{delivery:?} {code:02x?}: {:?}",
                run.stop()
            );
        }
    }
}

// WAIT is its own instruction (iced decodes 9Bh alone), so an FSAVE or
// FSTENV whose operand overruns raises 9 at the ESC byte, 1001h, after the
// WAIT retires (§5.2 p. 5-7), in Direct and Checked contexts alike.
#[test]
fn wait_combined_overrun_saves_the_esc_address() {
    for checked in [false, true] {
        for code in [
            &[0x9b, 0xdd, 0x36, 0xb0, 0xff][..], // FSAVE [FFB0]
            &[0x9b, 0xd9, 0x36, 0xf8, 0xff],     // FSTENV [FFF8]
        ] {
            let (vm, mut cpu) = machine(CpuModel::I80286, FpuModel::Present, code);
            if checked {
                cpu.cpu.legacy.segments[SegmentId::Fs as usize].attributes = 0x92;
            }
            let run = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
            assert_eq!(run.steps(), 1, "{code:02x?}");
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::Exception {
                        addr: 0x1001,
                        vector: 9,
                        ..
                    })
                ),
                "{code:02x?} {checked}: {:?}",
                run.stop()
            );
        }
    }
}

// A LOCK form at the end of the code segment, IP FFFEh: the fetch length of
// an unlisted form is the strict decoder's, as before; the listed LOCK MOV
// reads its third byte past the limit.
#[test]
fn lock_forms_at_the_segment_end() {
    for checked in [false, true] {
        for (code, vector) in [
            (&[0xf0, 0x05, 0x34, 0x12][..], 13), // LOCK ADD AX,1234h
            (&[0xf0, 0x40], 6),                  // LOCK INC AX
            (&[0xf0, 0xb0, 0x5a], 13),           // LOCK MOV AL,5Ah
        ] {
            let mut vm = Vm::with_backend(VmConfig::flat(0x2_0000), Box::new(NoBackend));
            vm.set_cpu_model(CpuModel::I80286);
            vm.set_exception_delivery(ExceptionDelivery::Outward);
            vm.map(0, 0x2_0000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0xfffe, code).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu = CpuState::legacy_real_entry([0, 0, 0, 0x300, 0, 0]);
            cpu.cpu.rip = 0xfffe;
            cpu.cpu.gpr[4] = 0x7000;
            if checked {
                cpu.cpu.legacy.segments[SegmentId::Fs as usize].attributes = 0x92;
            }
            let (actual, _, saved, ..) = exception(&vm, &mut cpu);
            assert_eq!((actual, saved), (vector, 0xfffe), "{code:02x?} {checked}");
        }
    }
}

// The strict decoder already reports NoMoreBytes on truncated LOCK forms, so
// the Checked fetch reads them whole; LOCK INC AX is complete and invalid.
// 9Bh before an ESC decodes as WAIT alone.
#[test]
fn iced_reports_truncated_lock_forms_as_incomplete() {
    use iced_x86::{Code, Decoder, DecoderError, DecoderOptions};
    for bytes in [
        &[0xf0, 0x05][..],
        &[0xf0, 0x05, 0x34],
        &[0xf0, 0xb0],
        &[0xf0, 0x88],
        &[0xf0, 0xc7],
        &[0xf0, 0xc7, 0x06],
        &[0xf0, 0xc7, 0x06, 0x34, 0x12],
        &[0xf0, 0xa1],
        &[0xf0, 0x80],
        &[0xf0, 0x80, 0xc0],
    ] {
        for options in [DecoderOptions::NONE, DecoderOptions::NO_INVALID_CHECK] {
            let mut decoder = Decoder::with_ip(16, bytes, 0, options);
            assert!(decoder.decode().is_invalid(), "{bytes:02x?}");
            assert_eq!(
                decoder.last_error(),
                DecoderError::NoMoreBytes,
                "{bytes:02x?}"
            );
        }
    }
    let mut decoder = Decoder::with_ip(16, &[0xf0, 0x40], 0, DecoderOptions::NONE);
    assert!(decoder.decode().is_invalid());
    assert_eq!(decoder.last_error(), DecoderError::InvalidInstruction);
    for bytes in [
        &[0x9b, 0xdd, 0x36, 0xb0, 0xff][..],
        &[0x9b, 0xd9, 0x36, 0xf8, 0xff],
    ] {
        let insn = Decoder::with_ip(16, bytes, 0x1000, DecoderOptions::NONE).decode();
        assert_eq!((insn.code(), insn.len()), (Code::Wait, 1));
    }
}

// The frozen Real16 alias keeps its behaviour: LOCK MOV executes there, and
// an x87 overrun and an eleven-byte instruction both deliver 13 through the
// IVT, saving IP 1000h.
#[test]
fn the_alias_keeps_its_behaviour() {
    for (code, vector) in [
        (&[0xf0, 0x88, 0x07][..], None),
        (&[0xdd, 0x06, 0xfc, 0xff], Some(13u64)),
        (
            &[
                0x26, 0x26, 0x26, 0x26, 0x26, 0x26, 0x26, 0x26, 0x26, 0xb0, 0x5a,
            ],
            Some(13),
        ),
    ] {
        let mut vm = Vm::new(VmConfig::flat(0x2_0000));
        vm.set_cpu_mode(x86jit_core::CpuMode::Real16);
        vm.set_exception_delivery(ExceptionDelivery::Outward);
        vm.map(0, 0x2_0000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, code).unwrap();
        // IVT: vector v -> 0:1800+v (HLT), so the stop IP names the vector.
        for v in 0..32u64 {
            vm.write_bytes(v * 4, &((0x1800 + v as u32) & 0xffff).to_le_bytes())
                .unwrap();
            vm.write_bytes(0x1800 + v, &[0xf4]).unwrap();
        }
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[4] = 0x8000;
        cpu.cpu.fpu_model = FpuModel::Present;
        let run = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
        match vector {
            Some(v) => {
                assert!(
                    matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
                    "{:?}",
                    run.stop()
                );
                assert_eq!(cpu.cpu.rip, 0x1800 + v + 1, "{code:02x?}");
                assert_eq!(read(&vm, 0x7ffa, 2), [0x00, 0x10], "{code:02x?}");
            }
            None => {
                assert_eq!(run.steps(), 2, "{:?}", run.stop());
                assert_eq!(cpu.cpu.rip, 0x1005, "{code:02x?}");
            }
        }
    }
}

// The 24-bit wrap composes with the A20 gate: with A20 disabled, a word at
// FFFFFFh reads EFFFFFh and 0 on the 80286 and EFFFFFh and 1000000h on the
// 80386, and a LOCK ADD there updates the same two bytes.
#[test]
fn the_wrap_composes_with_a20() {
    for (model, wrapped) in [(CpuModel::I80286, true), (CpuModel::I80386, false)] {
        // MOV BX,[000F]; LOCK ADD [000F],AX
        let (vm, mut cpu) = machine(
            model,
            FpuModel::Present,
            &[0x8b, 0x1e, 0x0f, 0x00, 0xf0, 0x01, 0x06, 0x0f, 0x00],
        );
        for (a, b) in [
            (0xff_ffff, 0x33),
            (0xef_ffff, 0x77),
            (0, 0x44),
            (0x100_0000, 0x55),
        ] {
            vm.write_bytes(a, &[b]).unwrap();
        }
        cpu.cpu.gpr[0] = 0x0101;
        cpu.cpu.legacy.segments[SegmentId::Ds as usize] = cache(0xff_fff0, 0xffff, 0x93);
        cpu.set_a20_enabled(false);
        assert_eq!(run(&vm, &mut cpu, 2).0, 2, "{model:?}");
        assert_eq!(
            cpu.cpu.gpr[3] & 0xffff,
            if wrapped { 0x4477 } else { 0x5577 }
        );
        let seen = [0xff_ffffu64, 0xef_ffff, 0, 0x100_0000].map(|a| read(&vm, a, 1)[0]);
        let expected = if wrapped {
            [0x33, 0x78, 0x45, 0x55]
        } else {
            [0x33, 0x78, 0x44, 0x56]
        };
        assert_eq!(seen, expected, "{model:?}");
    }
}

// The Direct resolver's typed refusals now report the refused instruction's
// bytes, as the Checked resolver does: an existing caller, LOCK ADC [BX],AX,
// which has no atomic path on either system model.
#[test]
fn direct_refusals_report_the_instruction_bytes() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for checked in [false, true] {
            let code = [0xf0, 0x11, 0x07];
            let (vm, mut cpu) = machine(model, FpuModel::Present, &code);
            if checked {
                cpu.cpu.legacy.segments[SegmentId::Fs as usize].attributes = 0x92;
            }
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            let StepStop::Exit(Exit::UnknownInstruction {
                addr, bytes, len, ..
            }) = run.stop()
            else {
                panic!("{model:?} {checked}: {:?}", run.stop());
            };
            assert_eq!((*addr, *len), (0x1000, 3), "{model:?} {checked}");
            assert_eq!(&bytes[..3], &code, "{model:?} {checked}");
        }
    }
}
