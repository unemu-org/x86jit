//! The delivery log of the Machine models (spec.md §17.10, Delivery log;
//! xj-ir6.37): one record per exception or interrupt that entered its handler,
//! in order, with the vector, kind, saved CS:IP and its linear address, the
//! error code, the CPL before and after, and the retired count. The frames
//! themselves are witnessed by legacy_real_exceptions, legacy_idt,
//! legacy_privilege, legacy_task_switch, legacy_i8086 and legacy_nec_v20;
//! this file pins what the log says about them.
use x86jit_core::legacy::SystemSegment;
use x86jit_core::{
    CpuModel, CpuState, DeliveryKind, DeliveryLog, DeliveryLogError, DeliveryRecord, Exit,
    I8080Entry, I8080Handback, Prot, RegionKind, SegmentCache, SegmentId, StepOptions, StepStop,
    Vcpu, Vm, VmConfig,
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
    cpu.enable_delivery_log(64).unwrap();
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
/// A record saved at `ip` of the code segment, with no error code, at CPL 0.
fn at(protected: bool, vector: u8, kind: DeliveryKind, ip: u16, retired: u64) -> DeliveryRecord {
    DeliveryRecord {
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
fn take(cpu: &mut Vcpu) -> Vec<DeliveryRecord> {
    let batch = cpu.delivery_log_mut().unwrap().take();
    assert_eq!(batch.dropped, 0);
    batch.records
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
                        DeliveryRecord {
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
) -> Vec<DeliveryRecord> {
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
            // BOUND's #BR saves BOUND (I286 PRM App. B BOUND p. B-22); NEC
            // CHKIND's interrupt 5 saves the next IP (NEC UM p. 12-170).
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
                        r(5, Software, 0x10, 0)
                    },
                    "{label}"
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
        let user = |vector, kind, ip, error_code, retired| DeliveryRecord {
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

#[test]
fn task_gate_deliveries_record_the_outgoing_task() {
    // 286 PRM §9.5 p. 9-6; 386 PRM §9.6.2 p. 9-9: a task gate saves the
    // interrupted CS:IP in the old TSS and pushes only the error code.
    // Task A (selector 18h) runs at 1000h; the gate names task B (20h) at
    // 1800h, whose image the TSS layout of each model holds (286 PRM §8.2
    // Fig. 8-1 p. 8-2; 386 PRM §7.1 Fig. 7-1 p. 7-2).
    const A: u16 = 0x18;
    const B: u16 = 0x20;
    for (model, wide) in [
        (CpuModel::I80286, false),
        (CpuModel::I80386, false),
        (CpuModel::I80386, true),
    ] {
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
            let mut program = vec![0xb8, A as u8, 0, 0x0f, 0x00, 0xd8]; // MOV AX,A; LTR AX
            program.extend(code);
            let mut vm = Vm::new(VmConfig::flat(MEMORY));
            vm.set_cpu_model(model);
            vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.write_bytes(0x1000, &program).unwrap();
            descriptor(&vm, GDT + 8, 0, 0xffff, 0x9b);
            descriptor(&vm, GDT + 16, 0, 0xffff, 0x93);
            for (selector, ip, stack) in [(A, 0, 0x8000u32), (B, 0x1800, 0x9000)] {
                let base = 0x4000 + u64::from(selector - A) * 0x80;
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
            assert_eq!(run.steps(), 2, "{label}: {:?}", run.stop());
            if kind == DeliveryKind::Irq {
                cpu.cpu.flags.if_ = true;
                cpu.inject_irq(0x30);
            }
            halt(&vm, &mut cpu);
            assert_eq!(cpu.cpu.legacy.tr.selector, B, "{label}");
            assert_eq!(
                take(&mut cpu),
                [DeliveryRecord {
                    vector,
                    kind,
                    saved_cs: 8,
                    saved_ip: ip,
                    saved_linear: ip,
                    error_code,
                    cpl_before: 0,
                    cpl_after: 0,
                    retired,
                }],
                "{label}"
            );
        }
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
            [DeliveryRecord {
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
            [DeliveryRecord {
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
        let software = |vector, cs: u16, ip: u32, retired| DeliveryRecord {
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
        assert_eq!(batch.dropped, 3);
        assert_eq!(
            batch.records.iter().map(|r| r.retired).collect::<Vec<_>>(),
            [7, 9],
            "{model:?}: each INT and its IRET retire"
        );
        let log = cpu.delivery_log().unwrap();
        assert_eq!((log.len(), log.dropped(), log.total()), (0, 0, 5));

        // Clear empties the ring and its dropped count; total keeps counting.
        cpu.cpu.rip = u64::from(IP);
        halt(&vm, &mut cpu);
        let log = cpu.delivery_log_mut().unwrap();
        assert_eq!((log.len(), log.dropped(), log.total()), (2, 3, 10));
        log.clear();
        assert_eq!((log.len(), log.dropped(), log.total()), (0, 0, 10));
        // Disabling returns the log; a run without it records nothing.
        assert_eq!(cpu.disable_delivery_log().unwrap().total(), 10);
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
    let vm = Vm::new(VmConfig::flat(0x10000));
    let mut cpu = vm.new_vcpu();
    assert_eq!(
        cpu.enable_delivery_log(8),
        Err(DeliveryLogError::NotMachine)
    );
    assert!(cpu.delivery_log().is_none());
}

/// A mixed workload: a loop of INT 40h, a counted loop with STI, a string
/// copy, the divide error and a final loop on itself, with an IRQ and an
/// NMI raised between slices.
fn mixed(model: CpuModel, protected: bool, log: Option<usize>) -> Vec<String> {
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
    let (vm, mut cpu) = machine(model, protected, program);
    match log {
        Some(capacity) => cpu.enable_delivery_log(capacity).unwrap(),
        None => {
            let _ = cpu.disable_delivery_log();
        }
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
    if log.is_some_and(|capacity| capacity > 1) {
        let kinds: Vec<DeliveryKind> = cpu.delivery_log().unwrap().iter().map(|r| r.kind).collect();
        for kind in [
            DeliveryKind::Software,
            DeliveryKind::Irq,
            DeliveryKind::Nmi,
            DeliveryKind::Fault,
        ] {
            assert!(kinds.contains(&kind), "{model:?}: {kinds:?}");
        }
    }
    seen
}

#[test]
fn the_log_changes_no_execution_state_or_retired_count() {
    for model in ALL {
        for &protected in modes(model) {
            let off = mixed(model, protected, None);
            assert_eq!(
                off,
                mixed(model, protected, Some(64)),
                "{model:?} {protected}"
            );
            // A ring smaller than the run overwrites records only.
            assert_eq!(
                off,
                mixed(model, protected, Some(1)),
                "{model:?} {protected}"
            );
        }
    }
}
