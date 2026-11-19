//! The delivery log of the Machine models (spec.md §17.10, Delivery log;
//! xj-ir6.37): one record per exception or interrupt that entered its handler,
//! in order, with the vector, kind, saved CS:IP and its linear address, the
//! error code, the CPL before and after, and the retired count. The frames
//! themselves are witnessed by legacy_real_exceptions, legacy_idt,
//! legacy_privilege, legacy_task_switch, legacy_i8086 and legacy_nec_v20;
//! this file pins what the log says about them.
use x86jit_core::legacy::SystemSegment;
use x86jit_core::{
    CpuMode, CpuModel, CpuState, DeliveryKind, DeliveryLog, DeliveryLogError, DeliveryLogOptions,
    DeliveryRecord, DeliveryRegs, Exit, I8080Entry, I8080Handback, Prot, RegionKind, SegmentCache,
    SegmentId, StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

const ALL: [CpuModel; 6] = [
    CpuModel::I8088,
    CpuModel::I8086,
    CpuModel::NecV20,
    CpuModel::NecV30,
    CpuModel::I80286,
    CpuModel::I80386,
];
/// The code segment's base in both modes: CS 100h in real mode, selector 8
/// in protected mode. Handlers live in the same segment.
const CODE_BASE: u32 = 0x1000;
const REAL_CS: u16 = 0x100;
const IP: u16 = 0x10;
const GDT: u64 = 0x2000;
const IDT: u64 = 0x2800;
const MEMORY: u64 = 0x20000;

fn twenty(model: CpuModel) -> bool {
    !matches!(model, CpuModel::I80286 | CpuModel::I80386)
}
/// Real mode on every model; protected mode on the 80286 and 80386.
fn modes(model: CpuModel) -> &'static [bool] {
    if twenty(model) {
        &[false]
    } else {
        &[false, true]
    }
}
/// An eight-byte descriptor with byte granularity (SDM Vol 3A §3.4.5
/// Fig. 3-8; 286 PRM §6.3 Fig. 6-3 p. 6-5).
fn descriptor(vm: &Vm, address: u64, base: u32, limit: u32, attr: u8) {
    let low = (limit & 0xffff) | (base << 16);
    let high = ((base >> 16) & 0xff) | (base & 0xff00_0000) | (u32::from(attr) << 8);
    vm.write_bytes(address, &low.to_le_bytes()).unwrap();
    vm.write_bytes(address + 4, &high.to_le_bytes()).unwrap();
}
fn cache(base: u32, attributes: u16) -> SegmentCache {
    SegmentCache {
        base,
        limit: 0xffff,
        attributes,
        unusable: 0,
        reserved: 0,
    }
}

/// A vCPU at CS:0010h with SS:SP = 0:8000h (selector 10h, base 0, in
/// protected mode), DS = ES = SS, the delivery log enabled.
fn machine(model: CpuModel, protected: bool, code: &[u8]) -> (Vm, Vcpu) {
    machine_with(model, protected, code, false)
}
/// [`machine`], with the stack page below 8000h a device region when
/// `device_stack` is set.
fn machine_with(model: CpuModel, protected: bool, code: &[u8], device_stack: bool) -> (Vm, Vcpu) {
    let (vm, mut cpu) = unlogged(model, protected, code, device_stack);
    cpu.enable_delivery_log(64).unwrap();
    (vm, cpu)
}
/// [`machine_with`] without the delivery log: no log API is called.
fn unlogged(model: CpuModel, protected: bool, code: &[u8], device_stack: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(MEMORY));
    vm.set_cpu_model(model);
    if device_stack {
        vm.map(0, 0x7000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0x7000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.map(
            0x8000,
            (MEMORY - 0x8000) as usize,
            Prot::RWX,
            RegionKind::Ram,
        )
        .unwrap();
    } else {
        vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
            .unwrap();
    }
    vm.write_bytes(u64::from(CODE_BASE) + u64::from(IP), code)
        .unwrap();
    let mut cpu = vm.new_vcpu();
    if protected {
        // 16-bit code at 1000h and data at 0 (286 PRM §6.3 Fig. 6-3 p. 6-5).
        descriptor(&vm, GDT + 8, CODE_BASE, 0xffff, 0x9a);
        descriptor(&vm, GDT + 16, 0, 0xffff, 0x92);
        cpu.cpu.cs = 8;
        for selector in [&mut cpu.cpu.ss, &mut cpu.cpu.ds, &mut cpu.cpu.es] {
            *selector = 16;
        }
        let segments = &mut cpu.cpu.legacy.segments;
        segments[SegmentId::Cs as usize] = cache(CODE_BASE, 0x9b);
        for id in [SegmentId::Ss, SegmentId::Ds, SegmentId::Es] {
            segments[id as usize] = cache(0, 0x93);
        }
        for id in [SegmentId::Fs, SegmentId::Gs] {
            segments[id as usize] = SegmentCache {
                unusable: 1,
                ..Default::default()
            };
        }
        cpu.cpu.legacy.cr0 = 1;
        cpu.cpu.legacy.gdtr.base = GDT as u32;
        cpu.cpu.legacy.gdtr.limit = 0x7f;
        cpu.cpu.legacy.idtr.base = IDT as u32;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
    } else {
        let real = CpuState::legacy_real_entry([REAL_CS, 0, 0, 0, 0, 0]);
        cpu.cpu.cs = REAL_CS;
        cpu.cpu.legacy.segments = real.legacy.segments;
    }
    cpu.cpu.rip = u64::from(IP);
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}
/// `code` at offset `at` of the code segment, the handler of `vector`: an
/// IVT entry in real mode, a DPL-0 286 interrupt gate in protected mode.
fn handler(vm: &Vm, protected: bool, vector: u8, at: u16, code: &[u8]) {
    vm.write_bytes(u64::from(CODE_BASE) + u64::from(at), code)
        .unwrap();
    if protected {
        let low = (8u32 << 16) | u32::from(at);
        vm.write_bytes(IDT + u64::from(vector) * 8, &low.to_le_bytes())
            .unwrap();
        vm.write_bytes(IDT + u64::from(vector) * 8 + 4, &0x8600u32.to_le_bytes())
            .unwrap();
    } else {
        let entry = u32::from(at) | (u32::from(REAL_CS) << 16);
        vm.write_bytes(u64::from(vector) * 4, &entry.to_le_bytes())
            .unwrap();
    }
}
fn cs(protected: bool) -> u16 {
    if protected {
        8
    } else {
        REAL_CS
    }
}
/// The fields of a [`DeliveryRecord`] as a value to compare: the record is
/// non-exhaustive, so only the engine builds one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rec {
    vector: u8,
    kind: DeliveryKind,
    saved_cs: u16,
    saved_ip: u32,
    saved_linear: u32,
    error_code: Option<u32>,
    cpl_before: u8,
    cpl_after: u8,
    retired: u64,
}
impl From<&DeliveryRecord> for Rec {
    fn from(r: &DeliveryRecord) -> Self {
        Self {
            vector: r.vector,
            kind: r.kind,
            saved_cs: r.saved_cs,
            saved_ip: r.saved_ip,
            saved_linear: r.saved_linear,
            error_code: r.error_code,
            cpl_before: r.cpl_before,
            cpl_after: r.cpl_after,
            retired: r.retired,
        }
    }
}
fn recs<'a>(records: impl IntoIterator<Item = &'a DeliveryRecord>) -> Vec<Rec> {
    records.into_iter().map(Rec::from).collect()
}

/// A record saved at `ip` of the code segment, with no error code, at CPL 0.
fn at(protected: bool, vector: u8, kind: DeliveryKind, ip: u16, retired: u64) -> Rec {
    Rec {
        vector,
        kind,
        saved_cs: cs(protected),
        saved_ip: u32::from(ip),
        saved_linear: CODE_BASE + u32::from(ip),
        error_code: None,
        cpl_before: 0,
        cpl_after: 0,
        retired,
    }
}
fn halt(vm: &Vm, cpu: &mut Vcpu) {
    let exit = cpu.run(vm, Some(100));
    assert!(matches!(exit, Exit::Hlt), "{exit:?}");
}
fn take(cpu: &mut Vcpu) -> Vec<Rec> {
    let batch = cpu.delivery_log_mut().unwrap().take();
    assert_eq!(batch.dropped, 0);
    assert_eq!(batch.total, cpu.delivery_log().unwrap().total());
    recs(&batch.records)
}

#[test]
fn a_boot_program_records_its_exceptions_in_order_on_every_model() {
    // The 80286 and 80386: #UD for 0F FF (I286 PRM §5.2.3 p. 5-7; I386 PRM
    // §14.6 Table 14-1 p. 14-5), whose handler reads the word at DS:FFFFh:
    // interrupt 13 in real mode (I286 PRM App. B MOV p. B-74; I386 PRM §14.6
    // Table 14-1), #GP(0) past the data limit in protected mode. The
    // twenty-bit models raise no processor interrupt above 4
    // (`ModelQuirks::max_processor_interrupt`): their program takes the
    // divide error, whose handler executes INT 3.
    for model in ALL {
        for &protected in modes(model) {
            let label = format!("{model:?} protected={protected}");
            let expected = if twenty(model) {
                let (vm, mut cpu) = machine(model, protected, b"\x90\xf6\xf3"); // NOP; DIV BL
                handler(&vm, protected, 0, 0x4000, b"\x90\xcc"); // NOP; INT 3
                handler(&vm, protected, 3, 0x4020, b"\xf4");
                halt(&vm, &mut cpu);
                let divide = if model.profile().quirks.divide_saves_next_ip {
                    0x13
                } else {
                    0x11
                };
                let records = take(&mut cpu);
                (
                    records,
                    vec![
                        at(protected, 0, DeliveryKind::Fault, divide, 1),
                        at(protected, 3, DeliveryKind::Software, 0x4002, 3),
                    ],
                )
            } else {
                let (vm, mut cpu) = machine(model, protected, b"\x90\x0f\xff"); // NOP; #UD
                handler(&vm, protected, 6, 0x4000, b"\x90\x3e\xa1\xff\xff"); // MOV AX,[FFFF]
                handler(&vm, protected, 13, 0x4020, b"\xf4");
                halt(&vm, &mut cpu);
                (
                    take(&mut cpu),
                    vec![
                        at(protected, 6, DeliveryKind::Fault, 0x11, 1),
                        Rec {
                            error_code: protected.then_some(0),
                            ..at(protected, 13, DeliveryKind::Fault, 0x4001, 2)
                        },
                    ],
                )
            };
            assert_eq!(expected.0, expected.1, "{label}");
        }
    }
}

/// Run `code` until its handler halts, and return the records.
fn records(
    model: CpuModel,
    protected: bool,
    code: &[u8],
    prepare: impl FnOnce(&Vm, &mut Vcpu),
) -> Vec<Rec> {
    let (vm, mut cpu) = machine(model, protected, code);
    for vector in [1, 2, 3, 4, 5, 0x40, 0x50] {
        handler(
            &vm,
            protected,
            vector,
            0x4000 + u16::from(vector) * 0x10,
            b"\xf4",
        );
    }
    prepare(&vm, &mut cpu);
    halt(&vm, &mut cpu);
    take(&mut cpu)
}

#[test]
fn software_interrupts_irq_nmi_and_the_step_trap_have_their_kinds() {
    use DeliveryKind::*;
    for model in ALL {
        for &protected in modes(model) {
            let label = format!("{model:?} protected={protected}");
            let none = |_: &Vm, _: &mut Vcpu| {};
            let r = |vector, kind, ip, retired| vec![at(protected, vector, kind, ip, retired)];
            // INT n and INT 3 save the next IP and retire (I286 PRM App. B INT/INTO
            // p. B-48; 80386 PRM §17 INT/INTO p. 17-69).
            assert_eq!(
                records(model, protected, b"\xcd\x40", none),
                r(0x40, Software, 0x12, 1),
                "{label}"
            );
            assert_eq!(
                records(model, protected, b"\xcc", none),
                r(3, Software, 0x11, 1),
                "{label}"
            );
            // INTO delivers only with OF set.
            assert_eq!(
                records(model, protected, b"\xce\xcc", none),
                r(3, Software, 0x12, 2),
                "{label}"
            );
            assert_eq!(
                records(model, protected, b"\xce", |_, cpu| cpu.cpu.flags.of = true),
                r(4, Software, 0x11, 1),
                "{label}"
            );
            // An IRQ and an NMI before the first instruction.
            assert_eq!(
                records(model, protected, b"\x90", |_, cpu| {
                    cpu.cpu.flags.if_ = true;
                    cpu.inject_irq(0x50);
                }),
                r(0x50, Irq, 0x10, 0),
                "{label}"
            );
            assert_eq!(
                records(model, protected, b"\x90", |_, cpu| cpu.raise_nmi().unwrap()),
                r(2, Nmi, 0x10, 0),
                "{label}"
            );
            // The single-step trap after one instruction.
            assert_eq!(
                records(model, protected, b"\x90", |_, cpu| cpu.cpu.rflags_system |=
                    0x100),
                r(1, Trap, 0x11, 1),
                "{label}"
            );
            // BOUND's #BR is a fault that saves BOUND (I286 PRM App. B BOUND
            // p. B-22; SDM Vol 3A §7.15, Event 5 p. 7-31); NEC CHKIND's
            // interrupt 5 is a software interrupt that saves the next IP (NEC
            // UM p. 12-170).
            // The 8086 and 8088 have no BOUND.
            let bound = |vm: &Vm, cpu: &mut Vcpu| {
                vm.write_bytes(0x400, &[0xfd, 0xff, 0x03, 0x00]).unwrap();
                cpu.cpu.gpr[0] = 4;
            };
            let chkind = matches!(model, CpuModel::NecV20 | CpuModel::NecV30);
            if !twenty(model) || chkind {
                assert_eq!(
                    records(model, protected, b"\x62\x06\x00\x04", bound),
                    if chkind {
                        r(5, Software, 0x14, 1)
                    } else {
                        r(5, Fault, 0x10, 0)
                    },
                    "{label}"
                );
            }
        }
    }
}

// A twenty-bit unit holds up to sixteen instructions and an interrupt ends it
// (xj-ir6.37 review P1-1): a record follows the interrupt the walk took, not
// the stack pointer or the unit's ops. PUSH AX and ADD SP move SP before the
// interrupt op. Every model runs the probes in each of its modes.
fn each_mode(probe: impl Fn(CpuModel, bool, &str)) {
    for model in ALL {
        for &protected in modes(model) {
            probe(
                model,
                protected,
                &format!("{model:?} protected={protected}"),
            );
        }
    }
}
fn none(_: &Vm, _: &mut Vcpu) {}

#[test]
fn an_into_not_taken_after_a_push_records_nothing() {
    // PUSH AX; INTO with OF clear; HLT.
    each_mode(|model, protected, label| {
        assert_eq!(
            records(model, protected, b"\x50\xce\xf4", none),
            [],
            "{label}"
        );
    });
}

#[test]
fn a_chkind_within_its_bounds_after_a_push_records_nothing() {
    // PUSH AX; CHKIND AX,[400h] with AX = 4 within 0..10h; HLT. NEC UM
    // p. 12-170: interrupt 5 only outside the bounds.
    for model in [CpuModel::NecV20, CpuModel::NecV30] {
        let records = records(model, false, b"\x50\x62\x06\x00\x04\xf4", |vm, cpu| {
            vm.write_bytes(0x400, &[0, 0, 0x10, 0]).unwrap();
            cpu.cpu.gpr[0] = 4;
        });
        assert_eq!(records, [], "{model:?}");
    }
}

#[test]
fn an_int_whose_frame_restores_the_stack_pointer_is_recorded() {
    // ADD SP,6; INT 40h: the frame brings SP back to its value at the start
    // of the unit.
    each_mode(|model, protected, label| {
        assert_eq!(
            records(model, protected, b"\x83\xc4\x06\xcd\x40", none),
            [at(protected, 0x40, DeliveryKind::Software, 0x15, 2)],
            "{label}"
        );
    });
}

#[test]
fn an_int_reached_in_a_later_slice_is_recorded_once() {
    // PUSH AX; INT 40h run as one step and then the rest: the first slice
    // retires only PUSH AX and records nothing.
    each_mode(|model, protected, label| {
        let (vm, mut cpu) = machine(model, protected, b"\x50\xcd\x40");
        handler(&vm, protected, 0x40, 0x4400, b"\xf4");
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 1, "{label}: {:?}", run.stop());
        assert!(cpu.delivery_log().unwrap().is_empty(), "{label}");
        halt(&vm, &mut cpu);
        assert_eq!(
            take(&mut cpu),
            [at(protected, 0x40, DeliveryKind::Software, 0x13, 2)],
            "{label}"
        );
    });
}

#[test]
fn interrupts_taken_after_a_push_are_recorded() {
    // PUSH AX, then INTO with OF set, INT 3, and on the NEC V20/V30 CHKIND
    // outside the bounds -3..3, whose interrupt 5 saves the next IP (NEC UM
    // p. 12-170).
    use DeliveryKind::Software;
    each_mode(|model, protected, label| {
        assert_eq!(
            records(model, protected, b"\x50\xce", |_, cpu| cpu.cpu.flags.of =
                true),
            [at(protected, 4, Software, 0x12, 2)],
            "{label}"
        );
        assert_eq!(
            records(model, protected, b"\x50\xcc", none),
            [at(protected, 3, Software, 0x12, 2)],
            "{label}"
        );
        if matches!(model, CpuModel::NecV20 | CpuModel::NecV30) {
            let records = records(model, protected, b"\x50\x62\x06\x00\x04", |vm, cpu| {
                vm.write_bytes(0x400, &[0xfd, 0xff, 0x03, 0x00]).unwrap();
                cpu.cpu.gpr[0] = 4;
            });
            assert_eq!(records, [at(protected, 5, Software, 0x15, 2)], "{label}");
        }
    });
}

/// Run a program of software interrupts between stack moves to its HLT, in
/// one run or in run_steps slices of `slice`, and return its records.
fn sliced(model: CpuModel, protected: bool, slice: Option<u64>) -> Vec<Rec> {
    let program: &[u8] = &[
        0x50, //             10: push ax
        0xcd, 0x40, //       11: int 40h
        0x58, //             13: pop ax
        0x83, 0xec, 0x02, // 14: sub sp,2
        0xce, //             17: into, with OF clear
        0xcc, //             18: int 3
        0x83, 0xc4, 0x02, // 19: add sp,2
        0xcd, 0x40, //       1C: int 40h
        0xf4, //             1E: hlt
    ];
    let (vm, mut cpu) = machine(model, protected, program);
    handler(&vm, protected, 0x40, 0x4000, b"\xcf");
    handler(&vm, protected, 3, 0x4010, b"\xcf");
    // SP 7000h: SUB SP,2 leaves OF clear.
    cpu.cpu.gpr[4] = 0x7000;
    match slice {
        None => halt(&vm, &mut cpu),
        Some(slice) => loop {
            let run = cpu.run_steps(&vm, slice, StepOptions::default()).unwrap();
            match run.stop() {
                StepStop::Exit(Exit::Hlt) => break,
                StepStop::Exit(Exit::BudgetExhausted) => {}
                stop => panic!("{model:?}: {stop:?}"),
            }
        },
    }
    take(&mut cpu)
}

#[test]
fn the_records_do_not_depend_on_how_the_run_is_sliced() {
    use DeliveryKind::Software;
    for model in ALL {
        for &protected in modes(model) {
            let label = format!("{model:?} protected={protected}");
            // Each INT and each IRET retires; INTO with OF clear retires and
            // delivers nothing.
            let expected = vec![
                at(protected, 0x40, Software, 0x13, 2),
                at(protected, 3, Software, 0x19, 7),
                at(protected, 0x40, Software, 0x1e, 10),
            ];
            for slice in [None, Some(1), Some(7), Some(400)] {
                assert_eq!(
                    sliced(model, protected, slice),
                    expected,
                    "{label} {slice:?}"
                );
            }
        }
    }
}

const K_CODE: u16 = 0x08;
const K_STACK: u16 = 0x10;
const U_CODE: u16 = 0x1b;
const U_DATA: u16 = 0x23;
const TSS_SEL: u16 = 0x28;
const TSS: u64 = 0x3000;

/// Ring-0 and ring-3 code and data at base 0 and a busy TSS whose ring-0
/// stack is 10h:9000h; the CPU runs `code` at 1B:1000h, CPL 3, on 23:8000h.
fn ring3(model: CpuModel, code: &[u8]) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(MEMORY));
    vm.set_cpu_model(model);
    vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.write_bytes(0x1000, code).unwrap();
    for (selector, attr) in [
        (K_CODE, 0x9a),
        (K_STACK, 0x92),
        (U_CODE, 0xfa),
        (U_DATA, 0xf2),
    ] {
        descriptor(&vm, GDT + u64::from(selector & !7), 0, 0xffff, attr);
    }
    // 286 PRM §8.2 Fig. 8-1 p. 8-2; 386 PRM §7.1 Fig. 7-1 p. 7-2.
    let wide = model == CpuModel::I80386;
    let (limit, attr) = if wide { (0x67, 0x8b) } else { (0x2b, 0x83) };
    descriptor(&vm, GDT + u64::from(TSS_SEL), TSS as u32, limit, attr);
    let (ss0, sp0, size) = if wide {
        (TSS + 8, TSS + 4, 4)
    } else {
        (TSS + 4, TSS + 2, 2)
    };
    vm.write_bytes(ss0, &K_STACK.to_le_bytes()).unwrap();
    vm.write_bytes(sp0, &0x9000u32.to_le_bytes()[..size])
        .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.gdtr.base = GDT as u32;
    cpu.cpu.legacy.gdtr.limit = 0x7f;
    cpu.cpu.legacy.idtr.base = IDT as u32;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    cpu.cpu.legacy.tr = SystemSegment {
        selector: TSS_SEL,
        cache: SegmentCache {
            base: TSS as u32,
            limit,
            attributes: u16::from(attr),
            ..Default::default()
        },
        ..Default::default()
    };
    cpu.cpu.legacy.cpl = 3;
    cpu.cpu.cs = U_CODE;
    cpu.cpu.ss = U_DATA;
    cpu.cpu.ds = U_DATA;
    cpu.cpu.es = U_DATA;
    cpu.cpu.fs = 0;
    cpu.cpu.gs = 0;
    let segments = &mut cpu.cpu.legacy.segments;
    segments[SegmentId::Cs as usize] = cache(0, 0xfb);
    for id in [SegmentId::Ss, SegmentId::Ds, SegmentId::Es] {
        segments[id as usize] = cache(0, 0xf3);
    }
    for id in [SegmentId::Fs, SegmentId::Gs] {
        segments[id as usize] = SegmentCache {
            unusable: 1,
            ..Default::default()
        };
    }
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    cpu.enable_delivery_log(8).unwrap();
    (vm, cpu)
}
fn ring0_gate(vm: &Vm, vector: u8, attr: u8) {
    let low = (u32::from(K_CODE) << 16) | 0x5000;
    vm.write_bytes(IDT + u64::from(vector) * 8, &low.to_le_bytes())
        .unwrap();
    vm.write_bytes(IDT + u64::from(vector) * 8 + 5, &[attr])
        .unwrap();
}

#[test]
fn inward_deliveries_record_cpl_3_to_0() {
    // 286 PRM §9.4 pp. 9-4–5; 386 PRM §9.6.1.1 p. 9-7: a gate to a more
    // privileged nonconforming segment switches to the TSS stack of its
    // DPL. The software interrupt goes through a DPL-3 gate; the #GP(10h) of
    // MOV DS with a DPL-0 data selector at CPL 3 and the IRQ ignore the gate
    // DPL (286 PRM §9.4 p. 9-4).
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let user = |vector, kind, ip, error_code, retired| Rec {
            vector,
            kind,
            saved_cs: U_CODE,
            saved_ip: ip,
            saved_linear: ip,
            error_code,
            cpl_before: 3,
            cpl_after: 0,
            retired,
        };
        let (vm, mut cpu) = ring3(model, b"\xcd\x40");
        ring0_gate(&vm, 0x40, 0xe6);
        vm.write_bytes(0x5000, b"\xf4").unwrap();
        halt(&vm, &mut cpu);
        assert_eq!(cpu.cpu.legacy.cpl, 0);
        assert_eq!(
            take(&mut cpu),
            [user(0x40, DeliveryKind::Software, 0x1002, None, 1)]
        );

        let (vm, mut cpu) = ring3(model, b"\x8e\xd8");
        ring0_gate(&vm, 13, 0x86);
        vm.write_bytes(0x5000, b"\xf4").unwrap();
        cpu.cpu.gpr[0] = u64::from(K_STACK);
        halt(&vm, &mut cpu);
        assert_eq!(
            take(&mut cpu),
            [user(13, DeliveryKind::Fault, 0x1000, Some(0x10), 0)]
        );

        let (vm, mut cpu) = ring3(model, b"\x90");
        ring0_gate(&vm, 0x50, 0x86);
        vm.write_bytes(0x5000, b"\xf4").unwrap();
        cpu.cpu.flags.if_ = true;
        cpu.inject_irq(0x50);
        halt(&vm, &mut cpu);
        assert_eq!(
            take(&mut cpu),
            [user(0x50, DeliveryKind::Irq, 0x1000, None, 0)]
        );
    }
}

/// Task A (selector 18h) runs `code` at 1000h after MOV AX,A; LTR AX, on
/// 10h:8000h; vectors 30h and 13 are task gates to task B (20h) at 1800h on
/// 10h:9000h, whose image the TSS layout of each model holds (286 PRM §8.2
/// Fig. 8-1 p. 8-2; 386 PRM §7.1 Fig. 7-1 p. 7-2). A's TSS is at 4000h and
/// B's at 5000h. `device` makes one page a device region: B's stack page
/// [`B_STACK_PAGE`], or A's TSS page [`A_TSS_PAGE`], whose image the
/// switch only writes.
fn task_gates(model: CpuModel, wide: bool, code: &[u8], device: Option<u64>) -> (Vm, Vcpu) {
    const A: u16 = 0x18;
    const B: u16 = 0x20;
    let mut program = vec![0xb8, A as u8, 0, 0x0f, 0x00, 0xd8]; // MOV AX,A; LTR AX
    program.extend(code);
    let mut vm = Vm::new(VmConfig::flat(MEMORY));
    vm.set_cpu_model(model);
    if let Some(page) = device {
        vm.map(0, page as usize, Prot::RWX, RegionKind::Ram)
            .unwrap();
        vm.map(page, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.map(
            page + 0x1000,
            (MEMORY - page - 0x1000) as usize,
            Prot::RWX,
            RegionKind::Ram,
        )
        .unwrap();
    } else {
        vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
            .unwrap();
    }
    vm.write_bytes(0x1000, &program).unwrap();
    descriptor(&vm, GDT + 8, 0, 0xffff, 0x9b);
    descriptor(&vm, GDT + 16, 0, 0xffff, 0x93);
    for (selector, ip, stack) in [(A, 0, 0x8000u32), (B, 0x1800, 0x9000)] {
        let base = 0x4000 + u64::from(selector - A) * 0x200;
        let (limit, attr) = if wide { (0x67, 0x89) } else { (0x2b, 0x81) };
        descriptor(&vm, GDT + u64::from(selector), base as u32, limit, attr);
        let fields: &[(u64, usize, u32)] = if wide {
            &[
                (0x20, 4, ip),
                (0x24, 4, 2),
                (0x38, 4, stack),
                (0x48, 2, 16),
                (0x4c, 2, 8),
                (0x50, 2, 16),
                (0x54, 2, 16),
                (0x66, 2, 0x68),
            ]
        } else {
            &[
                (0x0e, 2, ip),
                (0x10, 2, 2),
                (0x1a, 2, stack),
                (0x22, 2, 16),
                (0x24, 2, 8),
                (0x26, 2, 16),
                (0x28, 2, 16),
            ]
        };
        if device == Some(base) {
            continue;
        }
        for &(offset, size, value) in fields {
            vm.write_bytes(base + offset, &value.to_le_bytes()[..size])
                .unwrap();
        }
    }
    // Task gates for vectors 30h and 13 (286 PRM §9.5 p. 9-6).
    for vector in [0x30u8, 13] {
        vm.write_bytes(
            IDT + u64::from(vector) * 8,
            &[0, 0, B as u8, 0, 0, 0x85, 0, 0],
        )
        .unwrap();
    }
    vm.write_bytes(0x1800, b"\xf4").unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    cpu.cpu.cs = 8;
    cpu.cpu.ss = 16;
    cpu.cpu.ds = 16;
    cpu.cpu.es = 16;
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.gdtr.base = GDT as u32;
    cpu.cpu.legacy.gdtr.limit = 0x7f;
    cpu.cpu.legacy.idtr.base = IDT as u32;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    cpu.enable_delivery_log(8).unwrap();
    let run = cpu.run_steps(&vm, 2, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 2, "{:?}", run.stop());
    (vm, cpu)
}
const B_STACK_PAGE: u64 = 0x8000;
const A_TSS_PAGE: u64 = 0x4000;
/// The 80286 with a 286 TSS, and the 80386 with a 286 and with a 386 TSS.
const TASK_FORMATS: [(CpuModel, bool); 3] = [
    (CpuModel::I80286, false),
    (CpuModel::I80386, false),
    (CpuModel::I80386, true),
];
/// A protected-mode record in code selector 8, whose base is 0, at CPL 0.
fn flat(vector: u8, kind: DeliveryKind, ip: u32, error_code: Option<u32>, retired: u64) -> Rec {
    Rec {
        vector,
        kind,
        saved_cs: 8,
        saved_ip: ip,
        saved_linear: ip,
        error_code,
        cpl_before: 0,
        cpl_after: 0,
        retired,
    }
}
/// Run to HLT, completing every device write on the way; return how many.
fn halt_completing_writes(vm: &Vm, cpu: &mut Vcpu, label: &str) -> usize {
    let mut writes = 0;
    loop {
        match cpu.run(vm, Some(100)) {
            Exit::MmioWrite { .. } => {
                assert!(cpu.delivery_log().unwrap().is_empty(), "{label}");
                writes += 1;
                cpu.complete_mmio_write();
            }
            Exit::Hlt => return writes,
            exit => panic!("{label}: {exit:?}"),
        }
    }
}

#[test]
fn a_software_interrupt_whose_task_switch_is_handed_off_is_recorded_once() {
    // INT 30h through the task gate, with task A's TSS page a device region:
    // the switch saves A's state there before it reads B's (286 PRM App. B
    // pp. B-12–13; 386 PRM §7.5 steps 3–5 p. 7-8), so each write is an
    // MmioWrite the embedder completes. The
    // switch completes when the INT resumes, and the INT is recorded then,
    // once, as without the device (xj-ir6.37 re-review P2-1).
    for (model, wide) in TASK_FORMATS {
        let label = format!("{model:?} wide={wide}");
        let (vm, mut cpu) = task_gates(model, wide, b"\xcd\x30", Some(A_TSS_PAGE));
        assert!(halt_completing_writes(&vm, &mut cpu, &label) > 0, "{label}");
        assert_eq!(cpu.cpu.legacy.tr.selector, 0x20, "{label}");
        assert_eq!(
            take(&mut cpu),
            [flat(0x30, DeliveryKind::Software, 0x1008, None, 3)],
            "{label}"
        );
    }
}

#[test]
fn task_gate_deliveries_record_the_outgoing_task() {
    // 286 PRM §9.5 p. 9-6; 386 PRM §9.6.2 p. 9-9: a task gate saves the
    // interrupted CS:IP in the old TSS and pushes only the error code.
    for (model, wide) in TASK_FORMATS {
        for (code, vector, kind, ip, error_code, retired) in [
            (
                &b"\xcd\x30"[..],
                0x30,
                DeliveryKind::Software,
                0x1008,
                None,
                3,
            ),
            (b"\x90", 0x30, DeliveryKind::Irq, 0x1006, None, 2),
            (
                b"\xb8\x78\x00\x8e\xd8",
                13,
                DeliveryKind::Fault,
                0x1009,
                Some(0x78),
                3,
            ),
        ] {
            let label = format!("{model:?} wide={wide} vector={vector}");
            let (vm, mut cpu) = task_gates(model, wide, code, None);
            if kind == DeliveryKind::Irq {
                cpu.cpu.flags.if_ = true;
                cpu.inject_irq(0x30);
            }
            halt(&vm, &mut cpu);
            assert_eq!(cpu.cpu.legacy.tr.selector, 0x20, "{label}");
            assert_eq!(
                take(&mut cpu),
                [flat(vector, kind, ip, error_code, retired)],
                "{label}"
            );
        }
    }
}

#[test]
fn a_task_gate_delivery_handed_off_after_the_switch_records_the_outgoing_task() {
    // The #GP(78h) of MOV DS through a task gate: the switch commits, then
    // the error code goes onto task B's stack, a device region (286 PRM §9.5
    // p. 9-6). The record still names task A's MOV DS once the write
    // completes, as the outgoing TSS does (xj-ir6.37 review P2-1).
    for (model, wide) in TASK_FORMATS {
        let label = format!("{model:?} wide={wide}");
        let (vm, mut cpu) = task_gates(model, wide, b"\xb8\x78\x00\x8e\xd8", Some(B_STACK_PAGE));
        assert!(halt_completing_writes(&vm, &mut cpu, &label) > 0, "{label}");
        assert_eq!(cpu.cpu.legacy.tr.selector, 0x20, "{label}");
        assert_eq!(
            take(&mut cpu),
            [flat(13, DeliveryKind::Fault, 0x1009, Some(0x78), 3)],
            "{label}"
        );
    }
}

#[test]
fn a_fault_during_delivery_records_only_the_delivery_that_entered_a_handler() {
    // 286 PRM §9.6.2 p. 9-10; 386 PRM §9.8.8 Tables 9-3/9-4 p. 9-13: #NP
    // while delivering #GP is a double fault; #NP while delivering an IRQ is
    // delivered serially; a double fault whose delivery faults shuts the
    // processor down, and NMI ends shutdown (286 PRM §9.6.2 p. 9-10). The
    // attempts that faulted are not recorded. 286 PRM §9.4 Fig. 9-3: a gate
    // with P = 0.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        // #GP(0) whose gate is absent: #DF(0) is recorded, at the MOV.
        let (vm, mut cpu) = machine(model, true, b"\x3e\xa1\xff\xff");
        handler(&vm, true, 8, 0x4000, b"\xf4");
        handler(&vm, true, 13, 0x4020, b"\xf4");
        vm.write_bytes(IDT + 13 * 8 + 5, &[0x06]).unwrap();
        halt(&vm, &mut cpu);
        assert_eq!(
            take(&mut cpu),
            [Rec {
                error_code: Some(0),
                ..at(true, 8, DeliveryKind::Fault, 0x10, 0)
            }],
            "{model:?}"
        );

        // An IRQ whose gate is absent: #NP(EXT, IDT entry 50h) is recorded.
        let (vm, mut cpu) = machine(model, true, b"\x90");
        handler(&vm, true, 11, 0x4000, b"\xf4");
        handler(&vm, true, 0x50, 0x4020, b"\xf4");
        vm.write_bytes(IDT + 0x50 * 8 + 5, &[0x06]).unwrap();
        cpu.cpu.flags.if_ = true;
        cpu.inject_irq(0x50);
        halt(&vm, &mut cpu);
        assert_eq!(
            take(&mut cpu),
            [Rec {
                error_code: Some(0x50 * 8 + 3),
                ..at(true, 11, DeliveryKind::Fault, 0x10, 0)
            }],
            "{model:?}"
        );

        // Neither #GP nor #DF has a gate: shutdown, with nothing recorded;
        // the NMI that ends it is.
        let (vm, mut cpu) = machine(model, true, b"\x3e\xa1\xff\xff");
        handler(&vm, true, 2, 0x4000, b"\xf4");
        let exit = cpu.run(&vm, Some(100));
        assert!(matches!(exit, Exit::Shutdown { .. }), "{exit:?}");
        assert!(cpu.delivery_log().unwrap().is_empty());
        let held = cpu.cpu.rip as u16;
        cpu.raise_nmi().unwrap();
        halt(&vm, &mut cpu);
        assert_eq!(
            take(&mut cpu),
            [at(true, 2, DeliveryKind::Nmi, held, 0)],
            "{model:?}"
        );
    }
}

#[test]
fn a_software_interrupt_whose_delivery_faults_records_the_fault_at_the_int() {
    // 286 PRM §9.4 p. 9-4; 386 PRM §9.6.1.1 p. 9-7: INT n through a gate whose
    // DPL is below CPL raises #GP with the IDT entry's error code (vector × 8
    // + 2), saved at the INT, which does not retire.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let (vm, mut cpu) = ring3(model, b"\xcd\x40");
        ring0_gate(&vm, 0x40, 0x86);
        ring0_gate(&vm, 13, 0x86);
        vm.write_bytes(0x5000, b"\xf4").unwrap();
        halt(&vm, &mut cpu);
        assert_eq!(
            take(&mut cpu),
            [Rec {
                vector: 13,
                kind: DeliveryKind::Fault,
                saved_cs: U_CODE,
                saved_ip: 0x1000,
                saved_linear: 0x1000,
                error_code: Some(0x40 * 8 + 2),
                cpl_before: 3,
                cpl_after: 0,
                retired: 0,
            }],
            "{model:?}"
        );
    }
}

#[test]
fn a_fault_after_a_committed_task_switch_records_the_new_task() {
    // 286 App. B CALL/JMP pp. B-25/B-57–58; 386 Appendix E Table 8-1 note r
    // p. E-15: task B's IP 1800h past its CS limit 17FFh is #GP(0) in the
    // new task. The delivery through the task gate that committed the switch
    // is not recorded, and an INT whose delivery faulted does not retire; the
    // #GP, through an interrupt gate, saves B's CS:IP. Its error code has EXT
    // set during the IRQ's delivery and clear during the INT's (SDM Vol 3A
    // §7.13 p. 7-19).
    for (model, wide) in TASK_FORMATS {
        for irq in [false, true] {
            let label = format!("{model:?} wide={wide} irq={irq}");
            let (vm, mut cpu) =
                task_gates(model, wide, if irq { b"\x90" } else { b"\xcd\x30" }, None);
            descriptor(&vm, GDT + 8, 0, 0x17ff, 0x9b);
            // Vector 13: a DPL-0 286 interrupt gate to 8:1700h.
            vm.write_bytes(IDT + 13 * 8, &[0, 0x17, 8, 0, 0, 0x86, 0, 0])
                .unwrap();
            vm.write_bytes(0x1700, b"\xf4").unwrap();
            if irq {
                cpu.cpu.flags.if_ = true;
                cpu.inject_irq(0x30);
            }
            halt(&vm, &mut cpu);
            assert_eq!(cpu.cpu.legacy.tr.selector, 0x20, "{label}");
            assert_eq!(
                take(&mut cpu),
                [flat(
                    13,
                    DeliveryKind::Fault,
                    0x1800,
                    Some(u32::from(irq)),
                    2
                )],
                "{label}"
            );
        }
    }
}

#[test]
fn nec_brkem_and_calln_are_software_interrupts() {
    // NEC UM p. 12-169 BRKEM and p. 12-176 CALLN: vector, frame, next PC;
    // CALLN is decoded by the native side at an 8080-mode boundary.
    for model in [CpuModel::NecV20, CpuModel::NecV30] {
        let mut vm = Vm::new(VmConfig::flat(0x120000));
        vm.set_cpu_model(model);
        vm.map(0, 0x120000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1200, b"\x0f\xff\x80").unwrap(); // BRKEM 80h
        vm.write_bytes(0x80 * 4, &[0x00, 0x01, 0x00, 0x20]).unwrap(); // 2000:0100
        vm.write_bytes(0x40 * 4, &[0x00, 0x02, 0x00, 0x60]).unwrap(); // 6000:0200
        vm.write_bytes(0x60200, b"\xf4").unwrap();
        vm.write_bytes(0x20110, b"\xed\xed\x40").unwrap(); // CALLN 40h
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1200;
        cpu.cpu.gpr[4] = 0x8000;
        cpu.enable_delivery_log(8).unwrap();
        let run = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
        let StepStop::Exit(Exit::I8080Mode { view, entry, .. }) = run.stop() else {
            panic!("{:?}", run.stop())
        };
        assert_eq!(*entry, I8080Entry::Brkem { vector: 0x80 });
        let mut view = *view;
        view.pc = 0x110;
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        halt(&vm, &mut cpu);
        let software = |vector, cs: u16, ip: u32, retired| Rec {
            vector,
            kind: DeliveryKind::Software,
            saved_cs: cs,
            saved_ip: ip,
            saved_linear: (u32::from(cs) << 4) + ip,
            error_code: None,
            cpl_before: 0,
            cpl_after: 0,
            retired,
        };
        assert_eq!(
            take(&mut cpu),
            [
                software(0x80, 0, 0x1203, 1),
                software(0x40, 0x2000, 0x113, 2)
            ],
            "{model:?}"
        );
    }
}

#[test]
fn a_delivery_handed_off_to_a_device_stack_is_recorded_once_when_it_completes() {
    // The frame of INT 40h, and of an IRQ, goes to a device stack: each
    // write is an MmioWrite the embedder completes, and the delivery enters
    // its handler only after the last one (spec.md §5.2; decision-16 §3).
    for model in ALL {
        for &protected in modes(model) {
            for irq in [false, true] {
                let label = format!("{model:?} protected={protected} irq={irq}");
                let (vm, mut cpu) = machine_with(
                    model,
                    protected,
                    if irq { b"\x90" } else { b"\xcd\x40" },
                    true,
                );
                handler(&vm, protected, 0x40, 0x4000, b"\xf4");
                handler(&vm, protected, 0x50, 0x4000, b"\xf4");
                if irq {
                    cpu.cpu.flags.if_ = true;
                    cpu.inject_irq(0x50);
                }
                let mut writes = 0;
                let exit = loop {
                    let exit = cpu.run(&vm, Some(100));
                    match exit {
                        Exit::MmioWrite { .. } => {
                            assert!(cpu.delivery_log().unwrap().is_empty(), "{label}");
                            writes += 1;
                            cpu.complete_mmio_write();
                        }
                        exit => break exit,
                    }
                };
                assert!(matches!(exit, Exit::Hlt), "{label}: {exit:?}");
                assert!(writes > 0, "{label}");
                assert_eq!(
                    take(&mut cpu),
                    if irq {
                        [at(protected, 0x50, DeliveryKind::Irq, 0x10, 0)]
                    } else {
                        [at(protected, 0x40, DeliveryKind::Software, 0x12, 1)]
                    },
                    "{label}"
                );
            }
        }
    }
}

#[test]
fn a_software_interrupt_under_single_step_is_recorded_before_the_trap() {
    // 8086 Family UM §2.6 p. 2-28 and AP-67 p. A-28: on the 8086 and 8088
    // the step trap follows INT n, with the handler's address saved; the
    // engine gives the V20 and V30 the same order. The 80286 and 80386 take
    // no trap after a taken INT (I286 PRM §9.7.1 p. 9-14; 80386 PRM
    // §12.3.1.4 p. 12-8).
    for model in ALL {
        for &protected in modes(model) {
            let records = records(model, protected, b"\xcd\x40", |vm, cpu| {
                cpu.cpu.rflags_system |= 0x100;
                handler(vm, protected, 0x40, 0x4400, b"\x90\xf4");
            });
            let mut expected = vec![at(protected, 0x40, DeliveryKind::Software, 0x12, 1)];
            if twenty(model) {
                expected.push(at(protected, 1, DeliveryKind::Trap, 0x4400, 1));
            }
            assert_eq!(records, expected, "{model:?} protected={protected}");
        }
    }
}

#[test]
fn a_full_ring_keeps_the_last_records_and_counts_the_dropped_ones() {
    for model in ALL {
        // Five INT 40h, each returning through IRET, then HLT.
        let (vm, mut cpu) = machine(
            model,
            false,
            b"\xcd\x40\xcd\x40\xcd\x40\xcd\x40\xcd\x40\xf4",
        );
        handler(&vm, false, 0x40, 0x4000, b"\xcf");
        cpu.enable_delivery_log(2).unwrap();
        halt(&vm, &mut cpu);
        let log = cpu.delivery_log().unwrap();
        assert_eq!(
            (log.capacity(), log.len(), log.total(), log.dropped()),
            (2, 2, 5, 3)
        );
        let ips: Vec<u32> = log.iter().map(|r| r.saved_ip).collect();
        assert_eq!(ips, [0x18, 0x1a], "{model:?}: oldest first");
        let batch = cpu.delivery_log_mut().unwrap().take();
        assert_eq!((batch.dropped, batch.total), (3, 5));
        assert_eq!(
            batch.records.iter().map(|r| r.retired).collect::<Vec<_>>(),
            [7, 9],
            "{model:?}: each INT and its IRET retire"
        );
        let log = cpu.delivery_log().unwrap();
        assert_eq!((log.len(), log.dropped(), log.total()), (0, 0, 5));

        // drain_into appends to the caller's buffer and returns the same
        // counts; a second drain into the cleared buffer reuses it.
        let mut buffer = Vec::new();
        cpu.cpu.rip = u64::from(IP);
        halt(&vm, &mut cpu);
        let counts = cpu.delivery_log_mut().unwrap().drain_into(&mut buffer);
        assert_eq!((counts.dropped, counts.total), (3, 10));
        let ips: Vec<u32> = buffer.iter().map(|r| r.saved_ip).collect();
        assert_eq!(ips, [0x18, 0x1a], "{model:?}");
        let storage = buffer.as_ptr();
        buffer.clear();
        cpu.cpu.rip = u64::from(IP);
        halt(&vm, &mut cpu);
        let counts = cpu.delivery_log_mut().unwrap().drain_into(&mut buffer);
        assert_eq!((counts.dropped, counts.total, buffer.len()), (3, 15, 2));
        assert_eq!(buffer.as_ptr(), storage, "{model:?}: the buffer is reused");
        let log = cpu.delivery_log().unwrap();
        assert_eq!((log.len(), log.dropped(), log.total()), (0, 0, 15));

        // Clear discards the records and the dropped count unread; total
        // keeps counting.
        cpu.cpu.rip = u64::from(IP);
        halt(&vm, &mut cpu);
        let log = cpu.delivery_log_mut().unwrap();
        assert_eq!((log.len(), log.dropped(), log.total()), (2, 3, 20));
        log.clear();
        assert_eq!((log.len(), log.dropped(), log.total()), (0, 0, 20));
        // Disabling returns the log; a run without it records nothing.
        assert_eq!(cpu.disable_delivery_log().unwrap().total(), 20);
        cpu.cpu.rip = u64::from(IP);
        halt(&vm, &mut cpu);
        assert!(cpu.delivery_log().is_none());
    }
}

#[test]
fn the_log_refuses_a_bad_capacity_and_an_application_vcpu() {
    let (_, mut cpu) = machine(CpuModel::I8086, false, b"\xf4");
    for capacity in [0, DeliveryLog::MAX_CAPACITY + 1] {
        assert_eq!(
            cpu.enable_delivery_log(capacity),
            Err(DeliveryLogError::Capacity(capacity))
        );
        assert_eq!(cpu.delivery_log().unwrap().capacity(), 64);
    }
    cpu.enable_delivery_log(DeliveryLog::MAX_CAPACITY).unwrap();
    // An Application vCPU: the default Long64 one, and the Real16 alias,
    // which runs the frozen internal Machine entry (decision-23 §6).
    for mode in [None, Some(CpuMode::Real16)] {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        if let Some(mode) = mode {
            vm.set_cpu_mode(mode);
        }
        let mut cpu = vm.new_vcpu();
        assert_eq!(
            cpu.enable_delivery_log(8),
            Err(DeliveryLogError::NotMachine),
            "{mode:?}"
        );
        assert!(cpu.delivery_log().is_none(), "{mode:?}");
    }
}

/// A mixed workload: a loop of INT 40h, a counted loop with STI, a string
/// copy, the divide error and a final loop on itself, with an IRQ and an
/// NMI raised between slices. With `log` the delivery log of that capacity
/// is enabled and returned; without it no log API is called.
fn mixed(
    model: CpuModel,
    protected: bool,
    log: Option<usize>,
) -> (Vec<String>, Option<DeliveryLog>) {
    mixed_with(model, protected, log.map(|capacity| (capacity, None)))
}
/// [`mixed`], with the log enabled through `enable_delivery_log_with` when
/// options are given.
fn mixed_with(
    model: CpuModel,
    protected: bool,
    log: Option<(usize, Option<DeliveryLogOptions>)>,
) -> (Vec<String>, Option<DeliveryLog>) {
    let program: &[u8] = &[
        0xb9, 0x14, 0x00, // mov cx,20
        0xcd, 0x40, //       L: int 40h
        0x40, //             inc ax
        0xe2, 0xfb, //       loop L
        0xfb, //             sti
        0xb9, 0x32, 0x00, // mov cx,50
        0x43, //             L2: inc bx
        0xe2, 0xfd, //       loop L2
        0xbe, 0x00, 0x60, // mov si,6000h
        0xbf, 0x00, 0x70, // mov di,7000h
        0xb9, 0x10, 0x00, // mov cx,16
        0xf3, 0xa4, //       rep movsb
        0x31, 0xdb, //       xor bx,bx
        0xf6, 0xf3, //       div bl
        0x42, //             inc dx
        0xeb, 0xfe, //       jmp $
    ];
    let (vm, mut cpu) = unlogged(model, protected, program, false);
    match log {
        Some((capacity, None)) => cpu.enable_delivery_log(capacity).unwrap(),
        Some((capacity, Some(options))) => cpu.enable_delivery_log_with(capacity, options).unwrap(),
        None => {}
    }
    // The divide handler returns past DIV BL when the saved IP names it.
    let divide: &[u8] = if twenty(model) && model.profile().quirks.divide_saves_next_ip {
        b"\xcf"
    } else {
        b"\x55\x89\xe5\x83\x46\x02\x02\x5d\xcf" // push bp; mov bp,sp; add word [bp+2],2; pop bp; iret
    };
    handler(&vm, protected, 0, 0x4000, divide);
    for (vector, at) in [(2u8, 0x4100u16), (0x40, 0x4200), (0x50, 0x4300)] {
        handler(&vm, protected, vector, at, b"\xcf");
    }
    vm.write_bytes(0x6000, b"0123456789abcdef").unwrap();
    let mut seen = Vec::new();
    for (index, budget) in [7u64, 13, 29, 60, 400].into_iter().enumerate() {
        match index {
            1 | 3 => cpu.inject_irq(0x50),
            2 | 4 => cpu.raise_nmi().unwrap(),
            _ => {}
        }
        let run = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
        let mut memory = vec![0; 0x9000];
        vm.read_bytes(0, &mut memory).unwrap();
        seen.push(format!(
            "{} {:?} {} {:?} {:?}",
            run.steps(),
            run.stop(),
            cpu.retired_instructions(),
            cpu.cpu,
            memory
        ));
    }
    (seen, log.and_then(|_| cpu.disable_delivery_log()))
}

/// The deliveries of [`mixed`]. A step is one instruction and a delivery
/// none, so the NMI raised after 7 + 13 steps interrupts the LOOP of the
/// fifth INT 40h turn (MOV CX, then INT, IRET, INC AX and LOOP per turn).
/// The IRQ queued after 7 steps waits for STI and the instruction after it
/// (8086 Family UM §2.6 p. 2-24; I286 PRM §9.7.1 p. 9-14), and the NMI raised
/// after 109 steps interrupts the second loop. Its handler's IRET is the
/// 110th instruction; the 38 turns left, three MOVs, REP MOVSB and XOR BX
/// bring DIV BL's divide error to 191.
fn mixed_records(model: CpuModel, protected: bool) -> Vec<Rec> {
    use DeliveryKind::*;
    let mut expected = Vec::new();
    for turn in 0..20u64 {
        if turn == 5 {
            expected.push(at(protected, 2, Nmi, 0x16, 20));
        }
        let retired = 2 + 4 * turn + u64::from(turn >= 5);
        expected.push(at(protected, 0x40, Software, 0x15, retired));
    }
    expected.push(at(protected, 0x50, Irq, 0x1c, 84));
    expected.push(at(protected, 2, Nmi, 0x1c, 109));
    let divide = if twenty(model) && model.profile().quirks.divide_saves_next_ip {
        0x2e
    } else {
        0x2c
    };
    expected.push(at(protected, 0, Fault, divide, 191));
    expected
}

#[test]
fn the_log_changes_no_execution_state_or_retired_count() {
    for model in ALL {
        for &protected in modes(model) {
            let label = format!("{model:?} protected={protected}");
            // The run that never touches the log API is the baseline.
            let (off, none) = mixed(model, protected, None);
            assert!(none.is_none());
            let (on, log) = mixed(model, protected, Some(64));
            assert_eq!(off, on, "{label}");
            let log = log.unwrap();
            let records = recs(log.iter());
            assert_eq!(records, mixed_records(model, protected), "{label}");
            assert_eq!((log.total(), log.dropped()), (24, 0), "{label}");
            // A ring smaller than the run overwrites records only.
            let (small, ring) = mixed(model, protected, Some(1));
            assert_eq!(off, small, "{label}");
            let ring = ring.unwrap();
            assert_eq!(recs(ring.iter()), records[records.len() - 1..], "{label}");
            assert_eq!(
                (ring.total(), ring.dropped()),
                (log.total(), log.total() - 1),
                "{label}"
            );
        }
    }
}

// The registers option (spec.md §17.10, Delivery log, Registers; xj-ir6.40):
// each record carries the interrupted code's registers, as they were when the
// delivery began.

fn registers() -> DeliveryLogOptions {
    DeliveryLogOptions::default().registers(true)
}
/// The log's first record.
fn first(cpu: &Vcpu) -> DeliveryRecord {
    cpu.delivery_log().unwrap().iter().next().unwrap().clone()
}
/// A record's registers, which the registers option fills.
fn regs_of(record: &DeliveryRecord) -> DeliveryRegs {
    *record
        .regs
        .clone()
        .expect("the registers option fills regs")
}
/// The fields of a [`DeliveryRegs`] but FLAGS, as a value to compare.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Regs {
    /// EAX, EBX, ECX, EDX, ESI, EDI, EBP, ESP.
    gpr: [u32; 8],
    /// DS, ES, SS.
    selectors: [u16; 3],
    fs: Option<u16>,
    gs: Option<u16>,
}
impl From<&DeliveryRegs> for Regs {
    fn from(r: &DeliveryRegs) -> Self {
        Self {
            gpr: [r.eax, r.ebx, r.ecx, r.edx, r.esi, r.edi, r.ebp, r.esp],
            selectors: [r.ds, r.es, r.ss],
            fs: r.fs,
            gs: r.gs,
        }
    }
}
/// The general registers in `CpuState::gpr` order (AX, CX, DX, BX, SP, BP,
/// SI, DI) as `Regs::gpr` orders them.
fn ordered(gpr: [u32; 8]) -> [u32; 8] {
    [
        gpr[0], gpr[3], gpr[1], gpr[2], gpr[6], gpr[7], gpr[5], gpr[4],
    ]
}
const FLAGS_SET: u32 = 0x0001 | 0x0040 | 0x0200 | 0x0400 | 0x0800; // CF ZF IF DF OF

/// A vCPU running `code` with a distinct value in every register, handlers
/// that halt at 4000h for vectors 0, 13, 21h and 50h, and the delivery log
/// recording registers. AX..DI hold 1111h..7777h, BP excepted (with the
/// 80386's high halves 1..7), DX 0044h so that DH is zero; SS:SP is
/// 0400h:4000h in real mode and 10h:8000h in protected mode (ESP 0005xxxxh
/// on the 80386, whose stack is sixteen-bit), DS and ES 1234h
/// and 0567h or selectors 18h and 20h, FS and GS 0ABCh and 0DEFh in real mode
/// and null in protected mode. CF, ZF, IF, DF and OF are set. Returns the
/// registers a record of a delivery before the first instruction holds.
fn loaded(model: CpuModel, protected: bool, code: &[u8]) -> (Vm, Vcpu, Regs) {
    let (vm, mut cpu) = unlogged(model, protected, code, false);
    cpu.enable_delivery_log_with(64, registers()).unwrap();
    for vector in [0, 13, 0x21, 0x50] {
        handler(&vm, protected, vector, 0x4000, b"\xf4");
    }
    let wide = model == CpuModel::I80386;
    // CpuState::gpr order: AX, CX, DX, BX, SP, BP, SI, DI.
    let low: [u32; 8] = [0x1111, 0x3333, 0x0044, 0x2222, 0, 0x7777, 0x5555, 0x6666];
    let mut gpr = [0u32; 8];
    for (index, value) in low.into_iter().enumerate() {
        let high = if wide && index != 4 {
            (index as u32 + 1) << 16
        } else {
            0
        };
        gpr[index] = high | value;
    }
    let (ss, sp) = if protected {
        (0x10, 0x8000)
    } else {
        (0x400, 0x4000)
    };
    // The 80386's ESP has a high half too, which the sixteen-bit stack does
    // not use and the record carries (`DeliveryRegs::esp`).
    gpr[4] = if wide { 0x0005_0000 | sp } else { sp };
    for (index, value) in gpr.into_iter().enumerate() {
        cpu.cpu.gpr[index] = u64::from(value);
    }
    let (ds, es, fs, gs) = if protected {
        (0x18, 0x20, 0, 0)
    } else {
        (0x1234, 0x0567, 0x0abc, 0x0def)
    };
    if protected {
        descriptor(&vm, GDT + 0x18, 0, 0xffff, 0x92);
        descriptor(&vm, GDT + 0x20, 0, 0xffff, 0x92);
    }
    cpu.cpu.ss = ss;
    cpu.cpu.ds = ds;
    cpu.cpu.es = es;
    cpu.cpu.fs = fs;
    cpu.cpu.gs = gs;
    for (id, selector) in [
        (SegmentId::Ss, ss),
        (SegmentId::Ds, ds),
        (SegmentId::Es, es),
        (SegmentId::Fs, fs),
        (SegmentId::Gs, gs),
    ] {
        if !protected {
            cpu.cpu.legacy.segments[id as usize].base = u32::from(selector) << 4;
        }
    }
    let flags = &mut cpu.cpu.flags;
    (flags.cf, flags.zf, flags.if_, flags.df, flags.of) = (true, true, true, true, true);
    let expected = Regs {
        gpr: ordered(gpr),
        selectors: [ds, es, ss],
        fs: wide.then_some(fs),
        gs: wide.then_some(gs),
    };
    (vm, cpu, expected)
}
/// Run to the handler's HLT and return the one record's registers, after
/// checking that its FLAGS hold `set` (CF, ZF, IF, DF and OF, or fewer where
/// the raising instruction changes some of them), that the FLAGS word
/// the frame holds below the recorded SS:SP is its low half, and that the
/// handler runs with IF clear.
fn delivered(vm: &Vm, cpu: &mut Vcpu, protected: bool, set: u32, label: &str) -> Regs {
    halt(vm, cpu);
    let records: Vec<DeliveryRecord> = cpu.delivery_log().unwrap().iter().cloned().collect();
    assert_eq!(records.len(), 1, "{label}");
    let regs = *records[0]
        .regs
        .clone()
        .expect("the registers option fills regs");
    assert_eq!(regs.flags & set, set, "{label}: {:#x}", regs.flags);
    assert!(!cpu.cpu.flags.if_, "{label}");
    // Real-mode and 286-gate frames are words: FLAGS is the one pushed first,
    // just below the interrupted SP (8086 Family UM §2.6 pp. 2-25–28; 286 PRM
    // §9.4 Fig. 9-3).
    let base = if protected {
        0
    } else {
        u64::from(regs.ss) << 4
    };
    let mut word = [0; 2];
    vm.read_bytes(base + u64::from(regs.esp & 0xffff) - 2, &mut word)
        .unwrap();
    assert_eq!(
        regs.flags & 0xffff,
        u32::from(u16::from_le_bytes(word)),
        "{label}"
    );
    assert_eq!(regs.flags >> 16, 0, "{label}: no RF or VM");
    Regs::from(&regs)
}

#[test]
fn a_software_interrupt_records_its_inputs_on_every_model() {
    // INT n with a function number in AH: MOV AH, 30h; INT 21h. The record
    // holds the INT's inputs, SS:SP before its frame.
    for model in ALL {
        for &protected in modes(model) {
            let label = format!("{model:?} protected={protected}");
            let (vm, mut cpu, mut expected) = loaded(model, protected, b"\xb4\x30\xcd\x21");
            expected.gpr[0] = (expected.gpr[0] & !0xff00) | 0x3000;
            assert_eq!(
                delivered(&vm, &mut cpu, protected, FLAGS_SET, &label),
                expected,
                "{label}"
            );
            assert_ne!(cpu.cpu.gpr[4] as u32 & 0xffff, expected.gpr[7], "{label}");
        }
    }
}

#[test]
fn a_fault_and_an_irq_record_the_registers_at_that_moment() {
    // The fault: #GP for a word at DS:FFFFh on the 80286 and 80386 (interrupt
    // 13 in real mode, I286 PRM App. B MOV p. B-74; #GP(0) past the limit in
    // protected mode); the divide error of DIV DH on the twenty-bit models,
    // which raise no processor interrupt above 4. Neither writes a register
    // before it faults: the record holds the registers the faulting
    // instruction met, after the INC SI before it.
    for model in ALL {
        for &protected in modes(model) {
            let label = format!("{model:?} protected={protected}");
            let code: &[u8] = if twenty(model) {
                b"\x46\xf6\xf6" // INC SI; DIV DH
            } else {
                b"\x46\x3e\xa1\xff\xff" // INC SI; MOV AX, DS:[FFFFh]
            };
            let (vm, mut cpu, mut expected) = loaded(model, protected, code);
            expected.gpr[4] = (expected.gpr[4] & !0xffff) | 0x5556;
            // INC SI writes ZF and OF and keeps CF. DIV leaves the status
            // flags undefined (SDM Vol 2A §3.3 DIV, Flags Affected), and the
            // twenty-bit divide error leaves them as the engine's DIV wrote
            // them: there only IF and DF are pinned.
            let set = if twenty(model) { 0x0600 } else { 0x0601 };
            assert_eq!(
                delivered(&vm, &mut cpu, protected, set, &label),
                expected,
                "{label}"
            );
            let record = first(&cpu);
            assert_eq!(record.kind, DeliveryKind::Fault, "{label}");

            // An IRQ before the first instruction.
            let (vm, mut cpu, expected) = loaded(model, protected, b"\x90");
            cpu.inject_irq(0x50);
            assert_eq!(
                delivered(&vm, &mut cpu, protected, FLAGS_SET, &label),
                expected,
                "{label}"
            );
            let record = first(&cpu);
            assert_eq!(record.kind, DeliveryKind::Irq, "{label}");
        }
    }
}

#[test]
fn an_inward_delivery_records_the_interrupted_stack() {
    // 286 PRM §9.4 pp. 9-4–5; 386 PRM §9.6.1.1 p. 9-7: the gate switches to
    // the ring-0 stack 10h:9000h of the TSS; the record holds the ring-3
    // stack 23h:8000h the code was interrupted on.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let wide = model == CpuModel::I80386;
        for (code, vector, gate, kind) in [
            (&b"\xcd\x40"[..], 0x40, 0xe6, DeliveryKind::Software),
            (b"\x8e\xd8", 13, 0x86, DeliveryKind::Fault), // MOV DS, AX with AX = 10h
            (b"\x90", 0x50, 0x86, DeliveryKind::Irq),
        ] {
            let label = format!("{model:?} vector={vector:#x}");
            let (vm, mut cpu) = ring3(model, code);
            cpu.enable_delivery_log_with(8, registers()).unwrap();
            ring0_gate(&vm, vector, gate);
            vm.write_bytes(0x5000, b"\xf4").unwrap();
            let high = |index: u32| if wide { index << 16 } else { 0 };
            let gpr = [
                high(1) | u32::from(K_STACK),
                high(2) | 0x3333,
                high(3) | 0x4444,
                high(4) | 0x2222,
                0x8000,
                high(6) | 0x7777,
                high(7) | 0x5555,
                high(8) | 0x6666,
            ];
            for (index, value) in gpr.into_iter().enumerate() {
                cpu.cpu.gpr[index] = u64::from(value);
            }
            if kind == DeliveryKind::Irq {
                cpu.cpu.flags.if_ = true;
                cpu.inject_irq(0x50);
            }
            halt(&vm, &mut cpu);
            assert_eq!(
                (cpu.cpu.ss, cpu.cpu.gpr[4] & 0xffff < 0x9000),
                (K_STACK, true),
                "{label}"
            );
            let record = first(&cpu);
            assert_eq!((record.kind, record.cpl_before), (kind, 3), "{label}");
            assert_eq!(
                Regs::from(&regs_of(&record)),
                Regs {
                    gpr: ordered(gpr),
                    selectors: [U_DATA, U_DATA, U_DATA],
                    fs: wide.then_some(0),
                    gs: wide.then_some(0),
                },
                "{label}"
            );
            // The ring-3 frame on the ring-0 stack holds the same SS:SP.
            let mut frame = [0; 4];
            vm.read_bytes(0x9000 - 4, &mut frame).unwrap();
            assert_eq!(frame, [0x00, 0x80, U_DATA as u8, 0], "{label}");
        }
    }
}

#[test]
fn a_task_gate_delivery_records_the_outgoing_task_registers() {
    // 286 PRM §9.5 p. 9-6; 386 PRM §9.6.2 p. 9-9: the switch replaces every
    // register; the record holds task A's, as INT 30h met them, also when
    // the switch is handed off to the embedder to write A's TSS on a device.
    for (model, wide) in TASK_FORMATS {
        for device in [None, Some(A_TSS_PAGE)] {
            let label = format!("{model:?} wide={wide} device={device:?}");
            let (vm, mut cpu) = task_gates(model, wide, b"\xcd\x30", device);
            cpu.enable_delivery_log_with(8, registers()).unwrap();
            for (index, value) in [(1, 0x3333u64), (2, 0x4444), (3, 0x2222), (6, 0x5555)] {
                cpu.cpu.gpr[index] = value;
            }
            if device.is_some() {
                assert!(halt_completing_writes(&vm, &mut cpu, &label) > 0, "{label}");
            } else {
                halt(&vm, &mut cpu);
            }
            assert_eq!(
                (cpu.cpu.legacy.tr.selector, cpu.cpu.gpr[4]),
                (0x20, 0x9000),
                "{label}"
            );
            let record = first(&cpu);
            assert_eq!(
                Regs::from(&regs_of(&record)),
                Regs {
                    gpr: [0x18, 0x2222, 0x3333, 0x4444, 0x5555, 0, 0, 0x8000],
                    selectors: [16, 16, 16],
                    fs: (model == CpuModel::I80386).then_some(0),
                    gs: (model == CpuModel::I80386).then_some(0),
                },
                "{label}"
            );
        }
    }
}

#[test]
fn the_registers_option_changes_no_execution_state_or_retired_count() {
    for model in ALL {
        for &protected in modes(model) {
            let label = format!("{model:?} protected={protected}");
            let (off, _) = mixed(model, protected, None);
            let (plain, plain_log) = mixed_with(
                model,
                protected,
                Some((64, Some(DeliveryLogOptions::default()))),
            );
            let (on, log) = mixed_with(model, protected, Some((64, Some(registers()))));
            assert_eq!(off, plain, "{label}");
            assert_eq!(off, on, "{label}");
            let (plain_log, log) = (plain_log.unwrap(), log.unwrap());
            assert!(!plain_log.options().records_registers(), "{label}");
            assert!(log.options().records_registers(), "{label}");
            assert!(plain_log.iter().all(|r| r.regs.is_none()), "{label}");
            assert_eq!(recs(log.iter()), mixed_records(model, protected), "{label}");
            assert_eq!(recs(plain_log.iter()), recs(log.iter()), "{label}");
            let regs: Vec<DeliveryRegs> = log.iter().map(regs_of).collect();
            // The first INT 40h meets CX = 20 and SP = 8000h; every delivery
            // of the program interrupts code on that stack.
            assert_eq!((regs[0].ecx, regs[0].esp), (20, 0x8000), "{label}");
            assert!(regs.iter().all(|r| r.esp == 0x8000), "{label}");
        }
    }
}

#[test]
fn a_record_without_registers_stays_small() {
    // The registers are boxed: a record without them is one pointer larger
    // than the xj-ir6.37 record (32 bytes), not a whole DeliveryRegs.
    assert_eq!(std::mem::size_of::<DeliveryRecord>(), 40);
    assert_eq!(
        std::mem::size_of::<Option<Box<DeliveryRegs>>>(),
        std::mem::size_of::<usize>()
    );
}

#[test]
fn a_delivery_begun_before_the_registers_option_records_none() {
    // INT 30h through a task gate with task A's TSS on a device: the switch
    // is handed off to the embedder before it commits. Enabling the registers
    // option during the handoff starts an empty log; the INT, which began
    // before it, records no registers rather than a copy it never took.
    for (model, wide) in TASK_FORMATS {
        let label = format!("{model:?} wide={wide}");
        let (vm, mut cpu) = task_gates(model, wide, b"\xcd\x30", Some(A_TSS_PAGE));
        let mut enabled = false;
        loop {
            match cpu.run(&vm, Some(100)) {
                Exit::MmioWrite { .. } => {
                    if !enabled {
                        cpu.enable_delivery_log_with(8, registers()).unwrap();
                        enabled = true;
                    }
                    cpu.complete_mmio_write();
                }
                Exit::Hlt => break,
                exit => panic!("{label}: {exit:?}"),
            }
        }
        assert!(enabled, "{label}");
        let record = first(&cpu);
        assert_eq!(
            Rec::from(&record),
            flat(0x30, DeliveryKind::Software, 0x1008, None, 3),
            "{label}"
        );
        assert_eq!(record.regs, None, "{label}");
    }
}

#[test]
fn chained_and_handed_off_deliveries_record_where_they_began() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        // INT 40h through a gate with P = 0 raises #NP (286 PRM §9.4 Fig. 9-3
        // p. 9-5); a #GP whose gate is absent becomes #DF (286 PRM §9.6.2
        // p. 9-10). Each record holds the state the first delivery began
        // with: the INT's inputs, the MOV's.
        for (code, absent, vector) in [
            (&b"\xcd\x40"[..], 0x40u8, 11u8),
            (b"\x3e\xa1\xff\xff", 13, 8),
        ] {
            let label = format!("{model:?} vector={vector}");
            let (vm, mut cpu) = machine(model, true, code);
            cpu.enable_delivery_log_with(8, registers()).unwrap();
            handler(&vm, true, vector, 0x4000, b"\xf4");
            vm.write_bytes(IDT + u64::from(absent) * 8 + 5, &[0x06])
                .unwrap();
            cpu.cpu.gpr[1] = 0x1234;
            halt(&vm, &mut cpu);
            let record = first(&cpu);
            assert_eq!(record.vector, vector, "{label}");
            let regs = regs_of(&record);
            assert_eq!(
                (regs.ecx, regs.esp, regs.ss),
                (0x1234, 0x8000, 16),
                "{label}"
            );
        }
    }
    for (model, wide) in TASK_FORMATS {
        // A fault after a committed task switch belongs to task B: task B's
        // IP 1800h past its CS limit is #GP(0) in the new task (as
        // a_fault_after_a_committed_task_switch_records_the_new_task), and
        // the record holds B's registers, SP 9000h, not A's.
        let label = format!("{model:?} wide={wide}");
        let (vm, mut cpu) = task_gates(model, wide, b"\xcd\x30", None);
        cpu.enable_delivery_log_with(8, registers()).unwrap();
        descriptor(&vm, GDT + 8, 0, 0x17ff, 0x9b);
        vm.write_bytes(IDT + 13 * 8, &[0, 0x17, 8, 0, 0, 0x86, 0, 0])
            .unwrap();
        vm.write_bytes(0x1700, b"\xf4").unwrap();
        cpu.cpu.gpr[1] = 0x1234;
        halt(&vm, &mut cpu);
        let regs = regs_of(&first(&cpu));
        assert_eq!((regs.eax, regs.ecx, regs.esp), (0, 0, 0x9000), "{label}");

        // The #GP(78h) of MOV DS through a task gate, whose error code goes
        // to task B's stack on a device after the switch commits: the record
        // keeps task A's registers through the handoff. A copy taken again
        // when the switch resumes would hold B's (SP 9000h).
        let (vm, mut cpu) = task_gates(model, wide, b"\xb8\x78\x00\x8e\xd8", Some(B_STACK_PAGE));
        cpu.enable_delivery_log_with(8, registers()).unwrap();
        cpu.cpu.gpr[1] = 0x1234;
        assert!(halt_completing_writes(&vm, &mut cpu, &label) > 0, "{label}");
        assert_eq!(cpu.cpu.legacy.tr.selector, 0x20, "{label}");
        let regs = regs_of(&first(&cpu));
        assert_eq!(
            (regs.eax, regs.ecx, regs.esp, regs.ss, regs.ds),
            (0x78, 0x1234, 0x8000, 16, 16),
            "{label}"
        );
    }
    for model in [CpuModel::I8086, CpuModel::NecV30] {
        let label = format!("{model:?}");
        // INT 40h under TF: the INT records its inputs, TF set; the step trap
        // after it records the state the INT left, its frame pushed and IF
        // and TF clear (8086 Family UM §2.6 p. 2-28).
        let (vm, mut cpu) = machine(model, false, b"\xcd\x40");
        cpu.enable_delivery_log_with(8, registers()).unwrap();
        handler(&vm, false, 0x40, 0x4400, b"\x90\xf4");
        handler(&vm, false, 1, 0x4500, b"\xf4");
        cpu.cpu.flags.if_ = true;
        cpu.cpu.rflags_system |= 0x100;
        halt(&vm, &mut cpu);
        let regs: Vec<DeliveryRegs> = cpu.delivery_log().unwrap().iter().map(regs_of).collect();
        assert_eq!(regs.len(), 2, "{label}");
        assert_eq!(
            (regs[0].esp, regs[0].flags & 0x300),
            (0x8000, 0x300),
            "{label}"
        );
        assert_eq!((regs[1].esp, regs[1].flags & 0x300), (0x7ffa, 0), "{label}");

        // SP wraps: the frame of an INT at SP = 0, 2, 4 or 6 wraps in the
        // stack segment, and the record holds the SP the INT met.
        for sp in [0u16, 2, 4, 6] {
            let (vm, mut cpu) = machine(model, false, b"\xcd\x40");
            cpu.enable_delivery_log_with(8, registers()).unwrap();
            handler(&vm, false, 0x40, 0x4000, b"\xf4");
            cpu.cpu.gpr[4] = u64::from(sp);
            halt(&vm, &mut cpu);
            assert_eq!(regs_of(&first(&cpu)).esp, u32::from(sp), "{label} sp={sp}");
        }
    }
    for model in [CpuModel::NecV20, CpuModel::NecV30] {
        let label = format!("{model:?}");
        // CHKIND out of bounds: interrupt 5 with the CHKIND's inputs.
        let (vm, mut cpu) = machine(model, false, b"\x62\x06\x00\x04");
        cpu.enable_delivery_log_with(8, registers()).unwrap();
        handler(&vm, false, 5, 0x4000, b"\xf4");
        vm.write_bytes(0x400, &[0xfd, 0xff, 0x03, 0x00]).unwrap();
        cpu.cpu.gpr[0] = 4;
        cpu.cpu.gpr[1] = 0x1234;
        halt(&vm, &mut cpu);
        let regs = regs_of(&first(&cpu));
        assert_eq!(
            (regs.eax, regs.ecx, regs.esp),
            (4, 0x1234, 0x8000),
            "{label}"
        );

        // BRKEM from native mode saves MD = 1; CALLN from 8080 mode runs on
        // the native stack below BRKEM's frame and saves MD = 0 (NEC UM
        // p. 12-169 BRKEM, p. 12-176 CALLN).
        let mut vm = Vm::new(VmConfig::flat(0x120000));
        vm.set_cpu_model(model);
        vm.map(0, 0x120000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1200, b"\x0f\xff\x80").unwrap(); // BRKEM 80h
        vm.write_bytes(0x80 * 4, &[0x00, 0x01, 0x00, 0x20]).unwrap(); // 2000:0100
        vm.write_bytes(0x40 * 4, &[0x00, 0x02, 0x00, 0x60]).unwrap(); // 6000:0200
        vm.write_bytes(0x60200, b"\xf4").unwrap();
        vm.write_bytes(0x20110, b"\xed\xed\x40").unwrap(); // CALLN 40h
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1200;
        cpu.cpu.gpr[4] = 0x8000;
        cpu.enable_delivery_log_with(8, registers()).unwrap();
        let run = cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
        let StepStop::Exit(Exit::I8080Mode { view, .. }) = run.stop() else {
            panic!("{:?}", run.stop())
        };
        let mut view = *view;
        view.pc = 0x110;
        cpu.hand_back_i8080(&view, I8080Handback::EdEscape).unwrap();
        halt(&vm, &mut cpu);
        let regs: Vec<DeliveryRegs> = cpu.delivery_log().unwrap().iter().map(regs_of).collect();
        assert_eq!(regs.len(), 2, "{label}");
        assert_eq!(
            (regs[0].esp, regs[0].flags & 0x8000),
            (0x8000, 0x8000),
            "{label}"
        );
        assert_eq!(
            (regs[1].esp, regs[1].flags & 0x8000),
            (0x7ffa, 0),
            "{label}"
        );
    }
}

#[test]
fn the_80386_records_eflags_as_pushfd_forms_it() {
    // SDM Vol 2B §4.3 PUSHF/PUSHFD/PUSHFQ, Operation: the image has RF and
    // VM clear. A live RF at the INT does not reach the record.
    let (vm, mut cpu, _) = loaded(CpuModel::I80386, true, b"\xcd\x21");
    cpu.cpu.rflags_system |= 0x1_0000;
    halt(&vm, &mut cpu);
    let regs = regs_of(&first(&cpu));
    assert_eq!(regs.flags >> 16, 0, "{:#x}", regs.flags);
    assert_eq!(regs.flags & FLAGS_SET, FLAGS_SET, "{:#x}", regs.flags);
}
