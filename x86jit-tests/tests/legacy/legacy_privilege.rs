//! Privilege transitions on the I80286 and I80386 Machine models: interrupts
//! and exceptions to an inner ring, far CALL and JMP through call gates, RETF
//! and IRET to an outer ring, task switches into rings 1–3, and the CPL and
//! IOPL checks at CPL 1–3.
//! Sources pinned in oracles/intel: 80286 PRM 210498-005, 80386 PRM
//! 230985-001, SDM 325462-092US; printed page numbers. spec.md §17.10 records
//! the rows where they disagree and the choice (decisions O1–O14).
#[path = "../support/descriptor_fixture.rs"]
mod fixture;
use fixture::*;
use x86jit_core::fault::ErrorCode;
use x86jit_core::legacy::SystemSegment;
use x86jit_core::{
    CpuModel, ExceptionDelivery, Exit, Prot, RegionKind, SegmentCache, StepOptions, StepStop,
    UnsupportedSystemOperation, Vcpu, Vm,
};

const K_CODE: u16 = 0x08;
const K_STACK: u16 = 0x10;
const U_CODE: u16 = 0x1b;
const U_DATA: u16 = 0x23;
const TSS_SEL: u16 = 0x28;
const R1_CODE: u16 = 0x31;
const R1_STACK: u16 = 0x39;
const R2_STACK: u16 = 0x42;
const CONF3: u16 = 0x48;
const CONF0: u16 = 0x50;
const R2_CODE: u16 = 0x5a;
const TSS: u64 = 0x3000;
const IDT: u64 = 0x2800;
const CODE: [u16; 4] = [K_CODE, R1_CODE, R2_CODE, U_CODE];
const STACK: [u16; 4] = [K_STACK, R1_STACK, R2_STACK, U_DATA];
const TOP: [u32; 3] = [0x9000, 0xa000, 0xb000];
const MEMORY: u64 = 0x240000;

fn put(vm: &Vm, at: u64, size: usize, value: u64) {
    vm.write_bytes(at, &value.to_le_bytes()[..size]).unwrap();
}
fn cache(attr: u16) -> SegmentCache {
    SegmentCache {
        base: 0,
        limit: 0xffff,
        attributes: attr,
        unusable: 0,
        reserved: 0,
    }
}
fn null() -> SegmentCache {
    SegmentCache {
        unusable: 1,
        ..Default::default()
    }
}
fn wide(model: CpuModel) -> bool {
    model == CpuModel::I80386
}
/// The TSS field offsets of SS_n and (E)SP_n: 286 PRM §8.2 Fig. 8-1 p. 8-2;
/// 386 PRM §7.1 Fig. 7-1 p. 7-2.
fn tss_stack(model: CpuModel, ring: u64) -> (u64, u64, usize) {
    if wide(model) {
        (TSS + 8 + 8 * ring, TSS + 4 + 8 * ring, 4)
    } else {
        (TSS + 4 + 4 * ring, TSS + 2 + 4 * ring, 2)
    }
}
fn set_tss_stack(vm: &Vm, model: CpuModel, ring: u64, ss: u16, sp: u32) {
    let (ss_at, sp_at, size) = tss_stack(model, ring);
    put(vm, ss_at, 2, u64::from(ss));
    put(vm, sp_at, size, u64::from(sp));
}
fn gate(vm: &Vm, vector: u8, selector: u16, offset: u32, attr: u8) {
    let low = (u32::from(selector) << 16) | (offset & 0xffff);
    let high = (offset & 0xffff_0000) | (u32::from(attr) << 8);
    put(vm, IDT + u64::from(vector) * 8, 4, u64::from(low));
    put(vm, IDT + u64::from(vector) * 8 + 4, 4, u64::from(high));
}
fn enter(cpu: &mut Vcpu, ring: u8) {
    let r = usize::from(ring);
    let dpl = u16::from(ring) << 5;
    cpu.cpu.legacy.cpl = ring;
    cpu.cpu.cs = CODE[r];
    cpu.cpu.ss = STACK[r];
    cpu.cpu.ds = STACK[r];
    cpu.cpu.es = STACK[r];
    cpu.cpu.fs = 0;
    cpu.cpu.gs = 0;
    cpu.cpu.legacy.segments = [
        cache(0x9b | dpl),
        cache(0x93 | dpl),
        cache(0x93 | dpl),
        cache(0x93 | dpl),
        null(),
        null(),
    ];
}
/// A flat GDT with code and stack segments for every ring, a busy TSS whose
/// ring 0–2 stacks are SS_n:TOP[n], and an IDT at 2800h. The CPU runs `code`
/// at 1000h at CPL `ring`, on its stack at 8000h.
fn world(model: CpuModel, ring: u8, code: &[u8], delivery: ExceptionDelivery) -> (Vm, Vcpu) {
    let (vm, mut cpu) = if delivery == ExceptionDelivery::Outward {
        setup(model, code)
    } else {
        setup_delivery(model, code, delivery)
    };
    cpu.cpu.legacy.gdtr.limit = 0xff;
    for (selector, attr) in [
        (K_CODE, 0x9a),
        (K_STACK, 0x92),
        (U_CODE, 0xfa),
        (U_DATA, 0xf2),
        (R1_CODE, 0xba),
        (R1_STACK, 0xb2),
        (R2_STACK, 0xd2),
        (CONF3, 0xfe),
        (CONF0, 0x9e),
        (R2_CODE, 0xda),
    ] {
        descriptor(&vm, 0x2000 + u64::from(selector & !7), 0, 0xffff, attr);
    }
    let (limit, attr) = if wide(model) {
        (0x67, 0x8b)
    } else {
        (0x2b, 0x83)
    };
    descriptor(&vm, 0x2000 + u64::from(TSS_SEL), TSS as u32, limit, attr);
    if wide(model) {
        put(&vm, TSS + 0x66, 2, 0x68); // No bitmap within the TSS limit.
    }
    for ring in 0..3 {
        set_tss_stack(&vm, model, ring, STACK[ring as usize], TOP[ring as usize]);
    }
    cpu.cpu.legacy.tr = SystemSegment {
        selector: TSS_SEL,
        cache: SegmentCache {
            base: TSS as u32,
            limit,
            attributes: attr,
            ..Default::default()
        },
        ..Default::default()
    };
    cpu.cpu.legacy.idtr.base = IDT as u32;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    enter(&mut cpu, ring);
    (vm, cpu)
}
fn refused(cpu: &mut Vcpu, vm: &Vm, operation: UnsupportedSystemOperation) {
    let before = format!("{:?}", cpu.cpu);
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::UnsupportedSystemContext { operation: op, .. }) if *op == operation),
        "{:?}",
        run.stop()
    );
    assert_eq!(format!("{:?}", cpu.cpu), before);
}
fn halt(cpu: &mut Vcpu, vm: &Vm) {
    let exit = cpu.run(vm, Some(100));
    assert!(matches!(exit, Exit::Hlt), "{exit:?}");
}
fn frame(vm: &Vm, at: u64, size: usize, count: usize) -> Vec<u64> {
    (0..count)
        .map(|i| value(vm, at + (i * size) as u64, size))
        .collect()
}
/// Replace `[address, address + size)` with a device region.
fn trap_range(vm: &mut Vm, address: u64, size: u64) {
    let mut bytes = vec![0; MEMORY as usize];
    vm.read_bytes(0, &mut bytes).unwrap();
    vm.unmap(0, MEMORY as usize).unwrap();
    vm.map(0, address as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(address, size as usize, Prot::RW, RegionKind::Trap)
        .unwrap();
    vm.map(
        address + size,
        (MEMORY - address - size) as usize,
        Prot::RWX,
        RegionKind::Ram,
    )
    .unwrap();
    vm.write_bytes(0, &bytes[..address as usize]).unwrap();
    vm.write_bytes(address + size, &bytes[(address + size) as usize..])
        .unwrap();
}

#[test]
fn a_ring_0_kernel_irets_to_ring_3_and_takes_int_irq_and_gp_back_to_ring_0() {
    // The kernel copies its GDT, IDT and TSS images into place in real mode,
    // enters protected mode, loads TR and IRETs to ring 3. Ring 3 reloads DS
    // (the IRET nulled it), takes INT 40h, an IRQ and the #GP(0) of its HLT
    // into ring 0, each on the TSS's ring-0 stack.
    //   1000 cld; xor ax,ax; mov ds,ax; mov es,ax
    //        mov si,6000h; mov di,2000h; mov cx,40h; rep movsw   ; GDT
    //        mov si,6100h; mov di,2800h; mov cx,108h; rep movsw  ; IDT
    //        mov si,6400h; mov di,3000h; mov cx,34h; rep movsw   ; TSS
    //        lgdt [1076h]; lidt [107Ch]; mov ax,1; lmsw ax; jmp 8:103Dh
    //   103D mov ax,10h; mov ss,ax; mov sp,9000h; mov ds,ax; mov es,ax
    //        mov ax,28h; ltr ax
    //        push 23h; push 8000h; push 202h; push 1Bh; push 105Dh; iret
    //   105D mov ax,23h; mov ds,ax; int 40h; nop; nop; hlt
    //   1067 inc word [7000h]; iret          ; INT 40h, 16-bit gates
    //   106C inc word [7002h]; iret          ; IRQ 20h
    //   1071 pop ax; mov [7004h],ax; hlt     ; #GP
    //   1076 GDTR 7Fh:2000h, IDTR 7FFh:2800h
    //   1100 inc word [7000h]; iretd         ; 32-bit gates (I80386)
    //   1110 inc word [7002h]; iretd
    //   1120 pop eax; mov [7004h],ax; hlt
    const KERNEL: &str = "fc31c08ed88ec0be0060bf0020b94000f3a5be0061bf0028b90801f3a5be0064bf0030b93400f3a50f011676100f011e7c10b801000f01f0ea3d100800b810008ed0bc00908ed88ec0b828000f00d86a236800806802026a1b685d10cfb823008ed8cd409090f4ff060070cfff060270cf58a30470f47f0000200000ff0700280000";
    for model in MODELS {
        let (vm, mut cpu) = setup_delivery(model, &[], ExceptionDelivery::InGuest);
        let kernel = (0..KERNEL.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&KERNEL[i..i + 2], 16).unwrap())
            .collect::<Vec<_>>();
        vm.write_bytes(0x1000, &kernel).unwrap();
        vm.write_bytes(0x1100, b"\xff\x06\x00\x70\x66\xcf").unwrap();
        vm.write_bytes(0x1110, b"\xff\x06\x02\x70\x66\xcf").unwrap();
        vm.write_bytes(0x1120, b"\x66\x58\xa3\x04\x70\xf4").unwrap();
        // The images: the GDT at 6000h, the IDT at 6100h, the TSS at 6400h.
        for (selector, base, limit, attr) in [
            (K_CODE, 0, 0xffff, 0x9a),
            (K_STACK, 0, 0xffff, 0x92),
            (U_CODE, 0, 0xffff, 0xfa),
            (U_DATA, 0, 0xffff, 0xf2),
            (
                TSS_SEL,
                TSS as u32,
                if wide(model) { 0x67 } else { 0x2b },
                if wide(model) { 0x89 } else { 0x81 },
            ),
        ] {
            descriptor(&vm, 0x6000 + u64::from(selector & !7), base, limit, attr);
        }
        let (wide_gate, int40, irq20, gp): (u8, u64, u64, u64) = if wide(model) {
            (8, 0x1100, 0x1110, 0x1120)
        } else {
            (0, 0x1067, 0x106c, 0x1071)
        };
        for (vector, offset, attr) in [
            (13u64, gp, 0x86 | wide_gate),
            (0x20, irq20, 0x86 | wide_gate),
            (0x40, int40, 0xe7 | wide_gate),
        ] {
            put(&vm, 0x6100 + vector * 8, 4, (8 << 16) | offset);
            put(&vm, 0x6104 + vector * 8, 4, u64::from(attr) << 8);
        }
        let (ss0, sp0, size) = tss_stack(model, 0);
        put(&vm, ss0 - TSS + 0x6400, 2, u64::from(K_STACK));
        put(&vm, sp0 - TSS + 0x6400, size, 0x9000);
        cpu.cpu.legacy = x86jit_core::CpuState::legacy_real_entry([0; 6]).legacy;
        [
            cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs,
        ] = [0; 6];
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[4] = 0x7000;
        let mut guard = 0;
        while (cpu.cpu.cs, cpu.cpu.rip) != (U_CODE, 0x105d) {
            steps(&mut cpu, &vm, 1);
            guard += 1;
            assert!(guard < 2000, "{:?}", cpu.cpu);
        }
        // The IRET to ring 3 (286 PRM p. B-52; 386 PRM p. 17-77) nulled the
        // ring-0 DS and ES (SDM Vol 2A IRET p. 3-495).
        assert_eq!(cpu.cpu.legacy.cpl, 3);
        assert_eq!((cpu.cpu.ss, cpu.cpu.gpr[4] & 0xffff), (U_DATA, 0x8000));
        assert_eq!((cpu.cpu.ds, cpu.cpu.es), (0, 0));
        assert_eq!(cpu.cpu.legacy.segments[1].unusable, 1);
        assert!(cpu.cpu.flags.if_);
        assert_eq!(value(&vm, 0x202d, 1), if wide(model) { 0x8b } else { 0x83 });
        steps(&mut cpu, &vm, 3); // mov ax; mov ds; int 40h
        assert_eq!((cpu.cpu.cs, cpu.cpu.legacy.cpl), (K_CODE, 0));
        assert_eq!(
            (cpu.cpu.ss, cpu.cpu.gpr[4] & 0xffff),
            (K_STACK, 0x9000 - 10 * (size as u64 / 2))
        );
        steps(&mut cpu, &vm, 2); // inc; iret
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl),
            (U_CODE, 0x1064, 3)
        );
        assert_eq!((cpu.cpu.ds, cpu.cpu.gpr[4] & 0xffff), (U_DATA, 0x8000));
        assert_eq!(value(&vm, 0x7000, 2), 1);
        cpu.inject_irq(0x20);
        steps(&mut cpu, &vm, 1); // IRQ 20h, then inc
        assert_eq!((cpu.cpu.cs, cpu.cpu.legacy.cpl), (K_CODE, 0));
        assert!(!cpu.cpu.flags.if_);
        let pushed = frame(&vm, 0x9000 - 5 * size as u64, size, 5);
        assert_eq!(
            pushed,
            [0x1064, u64::from(U_CODE), 0x202, 0x8000, u64::from(U_DATA)]
        );
        steps(&mut cpu, &vm, 1); // iret
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl),
            (U_CODE, 0x1064, 3)
        );
        assert!(cpu.cpu.flags.if_);
        assert_eq!(value(&vm, 0x7002, 2), 1);
        // Ring 3 HLT is #GP(0) (286 PRM p. B-42; 386 PRM p. 6-17), delivered
        // with its error code on the ring-0 stack.
        halt(&mut cpu, &vm);
        assert_eq!((cpu.cpu.cs, cpu.cpu.legacy.cpl), (K_CODE, 0));
        // 386 PRM §12.3.1.1 p. 12-7: the fault frame sets RF.
        let flags = if wide(model) { 0x10202 } else { 0x202 };
        let pushed = frame(&vm, 0x9000 - 5 * size as u64, size, 5);
        assert_eq!(
            pushed,
            [0x1066, u64::from(U_CODE), flags, 0x8000, u64::from(U_DATA)]
        );
        assert_eq!(value(&vm, 0x7004, 2), 0);
        assert_eq!(value(&vm, 0x9000 - 6 * size as u64, size), 0);
    }
}

#[test]
fn int_n_from_ring_3_enters_ring_0_on_the_tss_stack() {
    // 286 PRM App. B INT p. B-49; 386 PRM §17 INT pp. 17-70–71; SDM Vol 2A
    // INT pp. 3-472–474. The frame: old SS, old (E)SP, FLAGS, CS, IP at the
    // gate's width; an interrupt gate clears IF, both clear TF and NT.
    for model in MODELS {
        for attr in [0xe6, 0xe7, 0xee, 0xef] {
            if !wide(model) && attr & 8 != 0 {
                continue;
            }
            let (vm, mut cpu) = world(model, 3, b"\xcd\x40", ExceptionDelivery::Outward);
            gate(&vm, 0x40, K_CODE, 0x5000, attr);
            cpu.cpu.gpr[4] = 0xdead_8000;
            cpu.cpu.flags.if_ = true;
            cpu.cpu.flags.cf = true;
            cpu.cpu.rflags_system = 0x4000;
            steps(&mut cpu, &vm, 1);
            let size = if attr & 8 != 0 { 4 } else { 2 };
            let sp = 0x9000 - 5 * size as u64;
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.ss, cpu.cpu.legacy.cpl),
                (K_CODE, 0x5000, K_STACK, 0)
            );
            assert_eq!(cpu.cpu.gpr[4], sp);
            assert_eq!(cpu.cpu.flags.if_, attr & 1 != 0);
            assert_eq!(cpu.cpu.rflags_system & 0x14100, 0);
            assert_eq!((cpu.cpu.ds, cpu.cpu.es), (U_DATA, U_DATA));
            let old_sp = if size == 4 { 0xdead_8000 } else { 0x8000 };
            assert_eq!(
                frame(&vm, sp, size, 5),
                [0x1002, u64::from(U_CODE), 0x4203, old_sp, u64::from(U_DATA)]
            );
            // The new CS and SS descriptors are marked accessed.
            assert_eq!((value(&vm, 0x200d, 1), value(&vm, 0x2015, 1)), (0x9b, 0x93));
            assert_eq!(value(&vm, 0x7ff0, 8) | value(&vm, 0x7ff8, 8), 0);
        }
    }
}

#[test]
fn ring_3_to_1_and_ring_2_to_0_use_the_target_rings_tss_stack() {
    for model in MODELS {
        for (from, handler, ring) in [(3, R1_CODE, 1), (2, K_CODE, 0)] {
            let (vm, mut cpu) = world(model, from, b"\xcd\x40", ExceptionDelivery::Outward);
            gate(&vm, 0x40, handler & !3, 0x5000, 0xe6);
            steps(&mut cpu, &vm, 1);
            let top = u64::from(TOP[ring as usize]);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.ss, cpu.cpu.legacy.cpl, cpu.cpu.gpr[4]),
                (handler, STACK[ring as usize], ring, top - 10)
            );
            assert_eq!(
                frame(&vm, top - 10, 2, 5),
                [
                    0x1002,
                    u64::from(CODE[from as usize]),
                    0x2,
                    0x8000,
                    u64::from(STACK[from as usize])
                ]
            );
        }
    }
}

#[test]
fn a_conforming_handler_runs_at_the_current_level() {
    // 286 PRM p. B-49; SDM Vol 3A §6.8.1.2 p. 6-12: no stack switch.
    for model in MODELS {
        let (vm, mut cpu) = world(model, 3, b"\xcd\x40", ExceptionDelivery::Outward);
        gate(&vm, 0x40, CONF0, 0x5000, 0xe6);
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.ss, cpu.cpu.legacy.cpl, cpu.cpu.gpr[4]),
            (CONF0 | 3, U_DATA, 3, 0x8000 - 6)
        );
    }
}

#[test]
fn an_exception_from_ring_3_pushes_its_error_code_and_ignores_gate_dpl() {
    // 286 PRM §9.4 p. 9-4 ("For exceptions and external interrupts, the CPL
    // of the program is ignored") and Fig. 9-4 pp. 9-4–5; SDM INT p. 3-474.
    // MOV DS of a DPL-0 segment at CPL 3 is #GP(10h). 386 PRM §12.3.1.1
    // p. 12-7: a fault frame sets RF.
    for model in MODELS {
        for attr in [0x86, 0x8e] {
            if !wide(model) && attr & 8 != 0 {
                continue;
            }
            let (vm, mut cpu) = world(model, 3, b"\x8e\xd8", ExceptionDelivery::InGuest);
            gate(&vm, 13, K_CODE, 0x5000, attr);
            vm.write_bytes(0x5000, b"\xf4").unwrap();
            cpu.cpu.gpr[0] = u64::from(K_STACK);
            halt(&mut cpu, &vm);
            let size = if attr & 8 != 0 { 4 } else { 2 };
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl),
                (K_CODE, 0x5001, 0)
            );
            assert_eq!(cpu.cpu.gpr[4], 0x9000 - 6 * size as u64);
            let flags = if size == 4 { 0x10002 } else { 0x2 };
            assert_eq!(
                frame(&vm, 0x9000 - 6 * size as u64, size, 6),
                [
                    0x10,
                    0x1000,
                    u64::from(U_CODE),
                    flags,
                    0x8000,
                    u64::from(U_DATA)
                ]
            );
            assert_eq!(cpu.cpu.ds, U_DATA);
        }
    }
}

#[test]
fn software_interrupts_check_gate_dpl_and_bound_does_not() {
    // 286 PRM p. B-49; 386 PRM p. 17-70: INT n, INT 3 and INTO through a
    // gate with DPL < CPL are #GP(vector*8+2); #BR is an exception and
    // ignores the gate's DPL (286 PRM §9.4 p. 9-4).
    for model in MODELS {
        for (code, vector) in [(&b"\xcd\x41"[..], 0x41), (b"\xcc", 3), (b"\xce", 4)] {
            let (vm, mut cpu) = world(model, 3, code, ExceptionDelivery::Outward);
            gate(&vm, vector, K_CODE, 0x5000, 0x86);
            cpu.cpu.flags.of = true;
            fault(&mut cpu, &vm, 13, Some(u32::from(vector) * 8 + 2));
            assert_eq!(value(&vm, 0x8ff6, 8) | value(&vm, 0x7ff8, 8), 0);
        }
        // BOUND AX,[6000h] with AX above the upper bound.
        let (vm, mut cpu) = world(model, 3, b"\x62\x06\x00\x60", ExceptionDelivery::InGuest);
        gate(&vm, 5, K_CODE, 0x5000, 0x86);
        vm.write_bytes(0x5000, b"\xf4").unwrap();
        put(&vm, 0x6000, 4, 0x0002_0001);
        cpu.cpu.gpr[0] = 5;
        halt(&mut cpu, &vm);
        assert_eq!((cpu.cpu.cs, cpu.cpu.legacy.cpl), (K_CODE, 0));
        assert_eq!(frame(&vm, 0x9000 - 10, 2, 2), [0x1000, u64::from(U_CODE)]);
    }
}

#[test]
fn irq_and_nmi_from_ring_3_enter_ring_0() {
    for model in MODELS {
        for nmi in [false, true] {
            let (vm, mut cpu) = world(model, 3, b"\x90\x90", ExceptionDelivery::Outward);
            let vector = if nmi { 2 } else { 0x30 };
            gate(&vm, vector, K_CODE, 0x5000, 0x86);
            vm.write_bytes(0x5000, b"\x90").unwrap();
            cpu.cpu.flags.if_ = true;
            if nmi {
                cpu.raise_nmi().unwrap();
            } else {
                cpu.inject_irq(vector);
            }
            steps(&mut cpu, &vm, 1);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl, cpu.cpu.gpr[4]),
                (K_CODE, 0x5001, 0, 0x9000 - 10)
            );
            assert_eq!(
                frame(&vm, 0x9000 - 10, 2, 5),
                [0x1000, u64::from(U_CODE), 0x202, 0x8000, u64::from(U_DATA)]
            );
        }
    }
}

/// One way to break an inward delivery, in the check order of §1.1 and §1.4
/// (SDM Vol 2A INT pp. 3-472–474; 286 PRM p. B-49; 386 PRM pp. 17-70–71).
fn break_inward(vm: &Vm, cpu: &mut Vcpu, model: CpuModel, index: usize) {
    let (ss_at, _, _) = tss_stack(model, 0);
    match index {
        0 => cpu.cpu.legacy.tr.cache.unusable = 1,
        1 => cpu.cpu.legacy.tr.cache.limit = (ss_at - TSS) as u32,
        2 => put(vm, ss_at, 2, 0),
        3 => put(vm, ss_at, 2, 0x1f8),
        4 => put(vm, ss_at, 2, 0x13),
        5 => put(vm, ss_at, 2, 0x20),
        6 => {
            let attr = value(vm, 0x2015, 1);
            put(vm, 0x2015, 1, (attr & 0x80) | 0x1a);
        }
        7 => {
            let attr = value(vm, 0x2015, 1);
            put(vm, 0x2015, 1, attr & 0x7f);
        }
        // SP0 9000h over an expand-down SS0 with limit 8FF8h: the 10 bytes
        // would start at 8FF6h, inside the limit (decision O2).
        8 => descriptor(vm, 0x2010, 0, 0x8ff8, 0x96),
        9 => descriptor(vm, 0x2008, 0, 0x4fff, 0x9a),
        _ => unreachable!(),
    }
}
fn inward_fault(index: usize) -> (u8, u16) {
    [
        (10, TSS_SEL),
        (10, TSS_SEL),
        (10, 0),
        (10, 0x1f8),
        (10, 0x10),
        (10, 0x20),
        (10, 0x10),
        (12, 0x10),
        (12, 0x10),
        (13, 0),
    ][index]
}

#[test]
fn inward_delivery_faults_precede_every_change_and_carry_ext() {
    // §1.1: TR and the TSS field (#TS(TR)); null SS #TS(EXT) (decision O5;
    // 386 PRM p. 17-70 prints #GP(EXT)); SS index, RPL, DPL and type
    // #TS(SS); not present #SS(SS); room #SS(SS) (decision O2); the gate
    // offset #GP(EXT) (SDM p. 3-474). EXT is set for an IRQ (SDM INT
    // p. 3-471, error_code(…, EXT)). Each pair pins the order of two checks.
    let mut cases = (0..10).map(|i| vec![i]).collect::<Vec<_>>();
    cases.extend([
        vec![0, 2],
        vec![1, 2],
        vec![2, 8],
        vec![3, 8],
        vec![5, 8],
        vec![6, 7],
        vec![7, 9],
        vec![8, 9],
    ]);
    for model in MODELS {
        for irq in [false, true] {
            for case in &cases {
                let (vm, mut cpu) = world(model, 3, b"\xcd\x40", ExceptionDelivery::Outward);
                gate(&vm, 0x40, K_CODE, 0x5000, 0xe6);
                for &index in case {
                    break_inward(&vm, &mut cpu, model, index);
                }
                if irq {
                    cpu.cpu.flags.if_ = true;
                    cpu.inject_irq(0x40);
                }
                let (vector, code) = inward_fault(case[0]);
                let ext = u32::from(irq);
                let gdt = (value(&vm, 0x200d, 1), value(&vm, 0x2015, 1));
                fault(&mut cpu, &vm, vector, Some(u32::from(code) | ext));
                assert_eq!(value(&vm, 0x8ff0, 8) | value(&vm, 0x8ff8, 8), 0, "{case:?}");
                assert_eq!((value(&vm, 0x200d, 1), value(&vm, 0x2015, 1)), gdt);
                assert!(!cpu.has_pending_irq());
            }
        }
    }
}

#[test]
fn a_failing_inward_delivery_escalates_to_double_fault_and_shutdown() {
    // 286 PRM §9.6.2 p. 9-10; 386 PRM Tables 9-3/9-4 p. 9-13: #TS during the
    // delivery of #GP is a double fault; a double fault whose delivery fails
    // shuts the processor down.
    for model in MODELS {
        let (vm, mut cpu) = world(model, 3, b"\x8e\xd8", ExceptionDelivery::InGuest);
        cpu.cpu.gpr[0] = u64::from(K_STACK);
        gate(&vm, 13, K_CODE, 0x5000, 0x86);
        gate(&vm, 8, R1_CODE & !3, 0x5100, 0x86);
        vm.write_bytes(0x5100, b"\x90").unwrap();
        put(&vm, tss_stack(model, 0).0, 2, 0);
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl),
            (R1_CODE, 0x5101, 1)
        );
        assert_eq!(frame(&vm, 0xa000 - 12, 2, 2), [0, 0x1000]);
        let (vm, mut cpu) = world(model, 3, b"\x8e\xd8", ExceptionDelivery::InGuest);
        cpu.cpu.gpr[0] = u64::from(K_STACK);
        gate(&vm, 13, K_CODE, 0x5000, 0x86);
        gate(&vm, 8, K_CODE, 0x5100, 0x86);
        put(&vm, tss_stack(model, 0).0, 2, 0);
        let exit = cpu.run(&vm, Some(10));
        assert!(matches!(exit, Exit::Shutdown { .. }), "{exit:?}");
    }
}

/// A RETF or IRET frame at ring 0: IP, CS, (FLAGS or n bytes), SP, SS.
#[allow(clippy::too_many_arguments)]
fn outer_frame(
    vm: &Vm,
    cpu: &mut Vcpu,
    size: usize,
    ip: u64,
    cs: u16,
    middle: &[u64],
    sp: u64,
    ss: u16,
) {
    let base = 0x8f00;
    cpu.cpu.gpr[4] = base;
    let mut at = base;
    for word in [ip, u64::from(cs)] {
        put(vm, at, size, word);
        at += size as u64;
    }
    for &word in middle {
        put(vm, at, 1, word);
        at += 1;
    }
    for word in [sp, u64::from(ss)] {
        put(vm, at, size, word);
        at += size as u64;
    }
}

#[test]
fn retf_to_an_outer_ring_reloads_ss_sp_and_nulls_data_segments() {
    // 286 PRM App. B RET pp. B-94–95; 386 PRM §17 RET pp. 17-142–144 and
    // §6.3.4.2 p. 6-16; SDM Vol 2B RET pp. 4-573–574. RET n releases n
    // bytes on both stacks (decision O9). DS holds DPL-0 data and is nulled;
    // ES (DPL 3) is kept; on the 386 FS holds conforming code (kept) and GS
    // non-conforming readable code with DPL 0 (nulled).
    for model in MODELS {
        for (code, size, n) in [
            (&b"\xcb"[..], 2usize, 0u16),
            (b"\xca\x04\x00", 2, 4),
            (b"\x66\xcb", 4, 0),
            (b"\x66\xca\x06\x00", 4, 6),
        ] {
            if !wide(model) && size == 4 {
                continue;
            }
            let (vm, mut cpu) = world(model, 0, code, ExceptionDelivery::Outward);
            let params = vec![0xaa; usize::from(n)];
            outer_frame(&vm, &mut cpu, size, 0x1234, U_CODE, &params, 0x7000, U_DATA);
            cpu.cpu.gpr[4] |= 0xdead_0000;
            cpu.cpu.es = U_DATA;
            cpu.cpu.legacy.segments[2] = cache(0xf3);
            if wide(model) {
                cpu.cpu.fs = CONF0;
                cpu.cpu.legacy.segments[4] = cache(0x9f);
                cpu.cpu.gs = K_CODE;
                cpu.cpu.legacy.segments[5] = cache(0x9b);
            }
            steps(&mut cpu, &vm, 1);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.ss, cpu.cpu.legacy.cpl),
                (U_CODE, 0x1234, U_DATA, 3)
            );
            // SDM RET p. 4-573: ESP := tempESP, then + n (DERIVED).
            assert_eq!(cpu.cpu.gpr[4], 0x7000 + u64::from(n));
            assert_eq!(cpu.cpu.legacy.segments[0].attributes, 0xfb);
            assert_eq!(cpu.cpu.legacy.segments[3].attributes, 0xf3);
            assert_eq!((cpu.cpu.ds, cpu.cpu.legacy.segments[1].unusable), (0, 1));
            assert_eq!(
                (cpu.cpu.es, cpu.cpu.legacy.segments[2].unusable),
                (U_DATA, 0)
            );
            if wide(model) {
                assert_eq!((cpu.cpu.fs, cpu.cpu.gs), (CONF0, 0));
                assert_eq!(cpu.cpu.legacy.segments[5].unusable, 1);
            }
            assert_eq!((value(&vm, 0x201d, 1), value(&vm, 0x2025, 1)), (0xfb, 0xf3));
        }
    }
}

/// One way to break an outward return, in the check order of §1.5 (286 PRM
/// p. B-95; 386 PRM Table 6-3 p. 6-17; SDM RET pp. 4-572–573).
#[derive(Clone, Copy)]
struct Outer {
    sp: u64,
    cs: u16,
    ss: u16,
    ip: u64,
}
fn break_outward(vm: &Vm, frame: &mut Outer, index: usize) {
    match index {
        0 => frame.sp = 0xfffa,
        1 => frame.cs = 3,
        2 => frame.cs = 0x1fb,
        3 => frame.cs = U_DATA,
        4 => frame.cs = R1_CODE | 3,
        5 => frame.cs = CONF3 | 2,
        6 => put(vm, 0x201d, 1, 0x7a),
        7 => frame.ss = 3,
        8 => frame.ss = 0x1fb,
        9 => frame.ss = U_DATA & !1,
        10 => frame.ss = U_CODE,
        11 => frame.ss = R2_STACK | 3,
        12 => put(vm, 0x2025, 1, 0x72),
        13 => descriptor(vm, 0x2018, 0, 0x0fff, 0xfa),
        _ => unreachable!(),
    }
}
fn outward_fault(index: usize) -> (u8, u16) {
    [
        (12, 0),
        (13, 0),
        (13, 0x1f8),
        (13, 0x20),
        (13, 0x30),
        (13, 0x48),
        (11, 0x18),
        (13, 0),
        (13, 0x1f8),
        (13, 0x20),
        (13, 0x18),
        (13, 0x40),
        (12, 0x20),
        (13, 0),
    ][index]
}

#[test]
fn outward_return_faults_precede_every_change() {
    // Null CS #GP(0); CS index, type and DPL #GP(CS) (a conforming CS needs
    // DPL <= RPL, decision O5); CS not present #NP(CS); SS null #GP(0); SS
    // index, RPL, type and DPL #GP(SS) (decision O4); SS not present #SS(SS)
    // (decision O3); IP beyond the CS limit #GP(0).
    let mut cases = (0..14).map(|i| vec![i]).collect::<Vec<_>>();
    cases.extend([
        vec![0, 1],
        vec![2, 7],
        vec![3, 12],
        vec![6, 7],
        vec![6, 12],
        vec![9, 12],
        vec![10, 13],
        vec![12, 13],
    ]);
    for model in MODELS {
        for iret in [false, true] {
            for case in &cases {
                let (vm, mut cpu) = world(
                    model,
                    0,
                    if iret { b"\xcf" } else { b"\xcb" },
                    ExceptionDelivery::Outward,
                );
                let mut outer = Outer {
                    sp: 0x8f00,
                    cs: U_CODE,
                    ss: U_DATA,
                    ip: 0x1234,
                };
                for &index in case {
                    break_outward(&vm, &mut outer, index);
                }
                let middle: &[u64] = if iret { &[2, 0] } else { &[] };
                outer_frame(
                    &vm, &mut cpu, 2, outer.ip, outer.cs, middle, 0x7000, outer.ss,
                );
                if outer.sp != 0x8f00 {
                    put(&vm, outer.sp, 2, outer.ip);
                    put(&vm, outer.sp + 2, 2, u64::from(outer.cs));
                    if iret {
                        put(&vm, outer.sp + 4, 2, 2);
                    }
                    cpu.cpu.gpr[4] = outer.sp;
                }
                let gdt = (value(&vm, 0x201d, 1), value(&vm, 0x2025, 1));
                let (vector, code) = outward_fault(case[0]);
                fault(&mut cpu, &vm, vector, Some(u32::from(code)));
                assert_eq!(
                    (value(&vm, 0x201d, 1), value(&vm, 0x2025, 1)),
                    gdt,
                    "{case:?}"
                );
            }
        }
    }
}

#[test]
fn iret_to_an_outer_ring_loads_if_and_iopl_by_the_old_cpl() {
    // SDM Vol 2A IRET p. 3-493: IF loads when CPL <= IOPL and IOPL when
    // CPL = 0, both before CPL := CS.RPL; 286 PRM p. B-51 ("IOPL ... changed
    // only when CPL=0"); 386 PRM p. 17-78.
    for model in MODELS {
        for (from, iopl) in [(0u8, 0u64), (1, 0), (1, 1), (2, 3), (2, 1)] {
            let (vm, mut cpu) = world(model, from, b"\xcf", ExceptionDelivery::Outward);
            cpu.cpu.rflags_system = iopl << 12;
            cpu.cpu.flags.if_ = false;
            outer_frame(
                &vm,
                &mut cpu,
                2,
                0x1234,
                U_CODE,
                &[0xc3, 0x32],
                0x7000,
                U_DATA,
            );
            steps(&mut cpu, &vm, 1);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.legacy.cpl, cpu.cpu.gpr[4]),
                (U_CODE, 3, 0x7000)
            );
            assert_eq!(cpu.cpu.flags.if_, from == 0 || u64::from(from) <= iopl);
            assert_eq!(
                (cpu.cpu.rflags_system >> 12) & 3,
                if from == 0 { 3 } else { iopl }
            );
            assert!(cpu.cpu.flags.cf && cpu.cpu.flags.zf && cpu.cpu.flags.sf);
        }
    }
}

#[test]
fn iret_to_an_outer_ring_restores_nt_tf_and_the_386_rf() {
    for model in MODELS {
        // TF traps after the first ring-3 instruction (286 PRM §9.7.1 p. 9-14).
        let (vm, mut cpu) = world(model, 0, b"\xcf", ExceptionDelivery::Outward);
        vm.write_bytes(0x1234, b"\x90").unwrap();
        outer_frame(
            &vm,
            &mut cpu,
            2,
            0x1234,
            U_CODE,
            &[0x02, 0x41],
            0x7000,
            U_DATA,
        );
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.rflags_system & 0x4100, 0x4100);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::Exception { vector: 1, .. })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!((cpu.cpu.rip, cpu.cpu.legacy.cpl), (0x1235, 3));
        if wide(model) {
            let (vm, mut cpu) = world(model, 0, b"\x66\xcf", ExceptionDelivery::Outward);
            outer_frame(
                &vm,
                &mut cpu,
                4,
                0x1234,
                U_CODE,
                &[0x02, 0, 1, 0],
                0x7000,
                U_DATA,
            );
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.rflags_system & 0x10000, 0x10000);
            // A frame with VM = 1 keeps its refusal.
            let (vm, mut cpu) = world(model, 0, b"\x66\xcf", ExceptionDelivery::Outward);
            outer_frame(
                &vm,
                &mut cpu,
                4,
                0x1234,
                U_CODE,
                &[0x02, 0, 2, 0],
                0x7000,
                U_DATA,
            );
            refused(
                &mut cpu,
                &vm,
                UnsupportedSystemOperation::InvalidLegacyContext(
                    x86jit_core::legacy::MachineContextError::Virtual8086Unsupported,
                ),
            );
        }
    }
}

#[test]
fn iret_and_retf_to_conforming_code_with_dpl_below_rpl_enter_ring_3() {
    // Decision O5: DPL <= RPL for a conforming return CS (286 PRM p. B-95;
    // SDM IRET p. 3-497), where 286 PRM p. B-52 and 386 PRM p. 17-77 print
    // "DPL must be > CPL" for IRET.
    for model in MODELS {
        for code in [b"\xcf", b"\xcb"] {
            let (vm, mut cpu) = world(model, 0, code, ExceptionDelivery::Outward);
            let middle: &[u64] = if code == b"\xcf" { &[2, 0] } else { &[] };
            outer_frame(&vm, &mut cpu, 2, 0x1234, CONF0 | 3, middle, 0x7000, U_DATA);
            steps(&mut cpu, &vm, 1);
            assert_eq!((cpu.cpu.cs, cpu.cpu.legacy.cpl), (CONF0 | 3, 3));
        }
    }
}

#[test]
fn popf_and_iret_at_each_cpl_keep_if_and_iopl_without_faulting() {
    // 286 PRM App. B POPF p. B-86, §10.1 p. 10-1; 386 PRM §8.3.1 p. 8-5,
    // §17 POPF p. 17-130; SDM Vol 2B POPF pp. 4-407–408.
    for model in MODELS {
        for iret in [false, true] {
            for cpl in 0..4u8 {
                for iopl in 0..4u64 {
                    let (vm, mut cpu) = world(
                        model,
                        cpl,
                        if iret { b"\xcf" } else { b"\x9d" },
                        ExceptionDelivery::Outward,
                    );
                    cpu.cpu.rflags_system = iopl << 12;
                    cpu.cpu.flags.if_ = false;
                    let image = 0x0203 | ((3 - iopl) << 12);
                    if iret {
                        put(&vm, 0x8000, 2, 0x1234);
                        put(&vm, 0x8002, 2, u64::from(CODE[usize::from(cpl)]));
                        put(&vm, 0x8004, 2, image);
                    } else {
                        put(&vm, 0x8000, 2, image);
                    }
                    steps(&mut cpu, &vm, 1);
                    assert_eq!(cpu.cpu.legacy.cpl, cpl);
                    assert_eq!(cpu.cpu.flags.if_, u64::from(cpl) <= iopl, "{cpl} {iopl}");
                    assert_eq!(
                        (cpu.cpu.rflags_system >> 12) & 3,
                        if cpl == 0 { 3 - iopl } else { iopl }
                    );
                    assert!(cpu.cpu.flags.cf);
                }
            }
        }
    }
}

#[test]
fn privileged_and_iopl_sensitive_instructions_check_before_any_effect() {
    // 286 PRM App. B pp. B-30, B-31, B-42, B-45, B-47, B-65, B-66, B-67,
    // B-72, B-81, B-82, B-106; 386 PRM §6.3.5.1 p. 6-17 and §8.3
    // pp. 8-4–5; SDM Vol 3A §6.9 p. 6-23. The memory forms (LGDT, LIDT,
    // LLDT, LTR and LMSW [BP+0]) address an SS operand beyond the SS
    // limit: the #GP(0) of the privilege check precedes its #SS(0)
    // (decision O10, ASSUMED). A missing I/O bitmap denies access on the
    // 80386 too (386 PRM §8.3.2 pp. 8-5–6). No PortIo exit is taken.
    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Cpl0,
        Iopl,
        Io,
    }
    let forms: [(&[u8], Kind, bool); 19] = [
        (b"\xf4", Kind::Cpl0, false),
        (b"\x0f\x00\x56\x00", Kind::Cpl0, false),
        (b"\x0f\x00\x5e\x00", Kind::Cpl0, false),
        (b"\x0f\x01\x56\x00", Kind::Cpl0, false),
        (b"\x0f\x01\x5e\x00", Kind::Cpl0, false),
        (b"\x0f\x01\xf0", Kind::Cpl0, false),
        (b"\x0f\x01\x76\x00", Kind::Cpl0, false),
        (b"\x0f\x06", Kind::Cpl0, false),
        (b"\x0f\x22\xc0", Kind::Cpl0, true),
        (b"\x0f\x20\xc0", Kind::Cpl0, true),
        (b"\x0f\x21\xf8", Kind::Cpl0, true),
        (b"\xfa", Kind::Iopl, false),
        (b"\xfb", Kind::Iopl, false),
        (b"\xe4\x60", Kind::Io, false),
        (b"\xe6\x80", Kind::Io, false),
        (b"\xec", Kind::Io, false),
        (b"\x6c", Kind::Io, false),
        (b"\x36\x6e", Kind::Io, false),
        (b"\xf3\x6e", Kind::Io, false),
    ];
    for model in MODELS {
        for (code, kind, only386) in forms {
            if only386 && !wide(model) {
                continue;
            }
            for cpl in 1..4u8 {
                for iopl in [0u64, 3] {
                    let (vm, mut cpu) = world(model, cpl, code, ExceptionDelivery::Outward);
                    cpu.cpu.rflags_system = iopl << 12;
                    cpu.cpu.flags.if_ = true;
                    cpu.cpu.gpr[5] = 0x2000;
                    cpu.cpu.gpr[6] = 0x2000;
                    cpu.cpu.gpr[1] = 2;
                    cpu.cpu.legacy.segments[3].limit = 0x0fff;
                    cpu.cpu.gpr[4] = 0x0f00;
                    let permitted = match kind {
                        Kind::Cpl0 => false,
                        Kind::Iopl | Kind::Io => u64::from(cpl) <= iopl,
                    };
                    if !permitted {
                        fault(&mut cpu, &vm, 13, Some(0));
                    } else if kind == Kind::Iopl {
                        steps(&mut cpu, &vm, 1);
                        assert_eq!(cpu.cpu.flags.if_, code == b"\xfb");
                    } else if code == b"\x36\x6e" {
                        // Permitted: the SS operand's own #SS(0) follows.
                        fault(&mut cpu, &vm, 12, Some(0));
                    } else {
                        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                        assert!(
                            matches!(run.stop(), StepStop::Exit(Exit::PortIo { .. })),
                            "{:?}",
                            run.stop()
                        );
                    }
                }
            }
        }
        // The unprivileged forms keep running at CPL 3: SMSW AX.
        let (vm, mut cpu) = world(model, 3, b"\x0f\x01\xe0", ExceptionDelivery::Outward);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[0] & 1, 1);
        // A reserved control register is #UD ahead of the privilege test
        // (ASSUMED: no source orders the two).
        if wide(model) {
            let (vm, mut cpu) = world(model, 3, b"\x0f\x20\xc8", ExceptionDelivery::Outward);
            fault(&mut cpu, &vm, 6, None);
        }
        // 80386 PRM §17 MOV p. 17-111: CPL 0 may read DR7.
        // ASSUMED: reserved/reset image from SDM Vol 3A §12.1
        // Table 12-1 p. 12-3.
        if wide(model) {
            let (vm, mut cpu) = world(model, 0, b"\x0f\x21\xf8", ExceptionDelivery::Outward);
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.gpr[0], 0x400);
        }
    }
}

/// An available TSS at 3100h, selector 60h, whose task runs at 1800h on
/// SS:8000h with the given CS, SS and DS, ES = U_DATA and SS0:SP0 =
/// K_STACK:C000h. Field offsets: 286 PRM §8.2 Fig. 8-1 p. 8-2; 386 PRM
/// §7.1 Fig. 7-1 p. 7-2.
fn incoming_task(vm: &Vm, model: CpuModel, cs: u16, ss: u16, ds: u16) {
    let incoming = 0x3100u64;
    let w = wide(model);
    descriptor(
        vm,
        0x2060,
        incoming as u32,
        if w { 0x67 } else { 0x2b },
        if w { 0x89 } else { 0x81 },
    );
    let fields: &[(u64, usize, u64)] = if w {
        &[
            (0x04, 4, 0xc000),
            (0x08, 2, K_STACK as u64),
            (0x20, 4, 0x1800),
            (0x24, 4, 0x0202),
            (0x38, 4, 0x8000),
            (0x48, 2, U_DATA as u64),
            (0x4c, 2, cs as u64),
            (0x50, 2, ss as u64),
            (0x54, 2, ds as u64),
        ]
    } else {
        &[
            (0x02, 2, 0xc000),
            (0x04, 2, K_STACK as u64),
            (0x0e, 2, 0x1800),
            (0x10, 2, 0x0202),
            (0x1a, 2, 0x8000),
            (0x22, 2, U_DATA as u64),
            (0x24, 2, cs as u64),
            (0x26, 2, ss as u64),
            (0x28, 2, ds as u64),
        ]
    };
    for &(offset, size, word) in fields {
        put(vm, incoming + offset, size, word);
    }
}

#[test]
fn a_task_switch_enters_ring_3_and_its_fault_is_delivered_inward() {
    // 286 PRM App. B SWITCH_TASKS p. B-13; 386 PRM §7.5 p. 7-8 and Table 7-1
    // p. 7-9: the incoming CPL is CS.RPL, and the incoming segments are
    // checked against it. The ring-3 HLT's #GP(0) enters ring 0 on the new
    // task's ring-0 stack, and so does the #TS(DS) of an incoming DS whose
    // DPL is 0, raised after the switch commits.
    for model in MODELS {
        for bad_ds in [false, true] {
            let (vm, mut cpu) = world(
                model,
                0,
                b"\xea\x00\x00\x60\x00",
                ExceptionDelivery::InGuest,
            );
            cpu.cpu.gpr[4] = 0x9000;
            let ds = if bad_ds { K_STACK } else { U_DATA };
            incoming_task(&vm, model, U_CODE, U_DATA, ds);
            vm.write_bytes(0x1800, b"\xf4").unwrap();
            gate(&vm, 13, K_CODE, 0x5000, 0x86);
            gate(&vm, 10, K_CODE, 0x5000, 0x86);
            vm.write_bytes(0x5000, b"\xf4").unwrap();
            if !bad_ds {
                steps(&mut cpu, &vm, 1);
                assert_eq!(
                    (
                        cpu.cpu.legacy.tr.selector,
                        cpu.cpu.cs,
                        cpu.cpu.ss,
                        cpu.cpu.legacy.cpl
                    ),
                    (0x60, U_CODE, U_DATA, 3)
                );
                assert_eq!(cpu.cpu.rip, 0x1800);
            }
            // The ring-3 HLT's #GP(0), or the #TS(DS) of the switch (386 PRM
            // Table 7-1 p. 7-9; 286 PRM p. B-13), then the handler's HLT.
            halt(&mut cpu, &vm);
            let code = if bad_ds { u64::from(K_STACK) } else { 0 };
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.legacy.cpl, cpu.cpu.gpr[4]),
                (K_CODE, 0, 0xc000 - 12)
            );
            assert_eq!(cpu.cpu.legacy.tr.selector, 0x60);
            assert_eq!(
                frame(&vm, 0xc000 - 12, 2, 6),
                [
                    code,
                    0x1800,
                    u64::from(U_CODE),
                    0x202,
                    0x8000,
                    u64::from(U_DATA)
                ]
            );
        }
    }
}

#[test]
fn an_embedder_state_at_cpl_3_runs() {
    for model in MODELS {
        for ring in 1..4 {
            let (vm, mut cpu) = world(model, ring, b"\x43", ExceptionDelivery::Outward);
            steps(&mut cpu, &vm, 1);
            assert_eq!((cpu.cpu.gpr[3], cpu.cpu.legacy.cpl), (1, ring));
        }
    }
}

#[test]
fn inward_frames_tss_fields_and_descriptors_on_devices_hand_off_exactly() {
    for model in MODELS {
        // The new stack's last slot (IP) is a device: the frame write stops
        // with nothing committed, and the completion finishes the delivery.
        let (mut vm, mut cpu) = world(model, 3, b"\xcd\x40", ExceptionDelivery::Outward);
        gate(&vm, 0x40, K_CODE, 0x5000, 0xe6);
        trap_range(&mut vm, 0x8ff6, 2);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0x8ff6,
                    size: 2,
                    value: 0x1002,
                    ..
                })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!(run.steps(), 0);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.ss, cpu.cpu.legacy.cpl, cpu.cpu.rip),
            (U_CODE, U_DATA, 3, 0x1000)
        );
        cpu.complete_mmio_write();
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl, cpu.cpu.gpr[4]),
            (K_CODE, 0x5000, 0, 0x8ff6)
        );
        assert_eq!(
            frame(&vm, 0x8ff8, 2, 4),
            [u64::from(U_CODE), 0x2, 0x8000, u64::from(U_DATA)]
        );
        // The TSS's ring-0 (E)SP is a device: its answer is the new stack.
        let (mut vm, mut cpu) = world(model, 3, b"\xcd\x40", ExceptionDelivery::Outward);
        gate(&vm, 0x40, K_CODE, 0x5000, 0xe6);
        let (_, sp_at, size) = tss_stack(model, 0);
        trap_range(&mut vm, sp_at, size as u64);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::MmioRead { addr, .. }) if *addr == sp_at),
            "{:?}",
            run.stop()
        );
        cpu.complete_mmio_read(0x7800);
        steps(&mut cpu, &vm, 1);
        assert_eq!((cpu.cpu.cs, cpu.cpu.gpr[4]), (K_CODE, 0x7800 - 10));
        assert_eq!(value(&vm, 0x7800 - 10, 2), 0x1002);
        // The new SS descriptor is a device and its accessed bit is clear:
        // its locked update is refused before the CS descriptor is touched.
        let (mut vm, mut cpu) = world(model, 3, b"\xcd\x40", ExceptionDelivery::Outward);
        gate(&vm, 0x40, K_CODE, 0x5000, 0xe6);
        trap_range(&mut vm, 0x2010, 8);
        for word in [0x0000_ffff, 0x0000_9200] {
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(run.stop(), StepStop::Exit(Exit::MmioRead { .. })),
                "{:?}",
                run.stop()
            );
            cpu.complete_mmio_read(word);
        }
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::UnsupportedSystemContext {
                    operation: UnsupportedSystemOperation::LegacyInstruction,
                    ..
                })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!((cpu.cpu.cs, cpu.cpu.legacy.cpl), (U_CODE, 3));
        assert_eq!(value(&vm, 0x200d, 1), 0x9a);
        assert_eq!(value(&vm, 0x8ff6, 8) | value(&vm, 0x8ffe, 2), 0);
    }
}

#[test]
fn error_codes_have_no_bits_above_15() {
    // 386 PRM §9.7 Fig. 9-7: a fault's error code defines bits 15:0 only.
    let (vm, mut cpu) = world(CpuModel::I80386, 3, b"\xcd\x40", ExceptionDelivery::Outward);
    gate(&vm, 0x40, K_CODE, 0x5000, 0xe6);
    put(&vm, tss_stack(CpuModel::I80386, 0).0, 2, 0x13);
    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    let StepStop::Exit(exit) = run.stop() else {
        panic!()
    };
    assert_eq!(
        exit.fault_metadata().unwrap().error_code(),
        ErrorCode::bits(0x10, 0xffff)
    );
}

#[test]
fn an_outward_iret_from_cpl_0_to_cpl_3_through_a_286_tss_and_an_int_back() {
    // The shape of an outward IRET (CF) from CPL 0 to CPL 3, on both
    // generations: TR holds a 286 TSS (SS0:SP0 set, SS1 and SS2 zero), LDTR
    // is loaded, and the ring-3 CS (FAh) and SS (F3h) alias the ring-0 bases.
    // 286 PRM App. B IRET p. B-52; 386 PRM §17 IRET pp. 17-77–78; SDM Vol 2A
    // IRET pp. 3-493–495; then INT n p. 3-473 and 286 PRM p. B-49.
    const BASE: u32 = 0x10000;
    for model in MODELS {
        let (vm, mut cpu) = world(model, 0, &[], ExceptionDelivery::Outward);
        for (selector, attr) in [
            (K_CODE, 0x9b),
            (K_STACK, 0x93),
            (U_CODE, 0xfa),
            (U_DATA, 0xf3),
        ] {
            descriptor(&vm, 0x2000 + u64::from(selector & !7), BASE, 0xffff, attr);
        }
        vm.write_bytes(TSS, &[0; 0x68]).unwrap();
        descriptor(&vm, 0x2028, TSS as u32, 0x2b, 0x83);
        put(&vm, TSS + 2, 2, 0x9000);
        put(&vm, TSS + 4, 2, u64::from(K_STACK));
        cpu.cpu.legacy.tr.cache = SegmentCache {
            base: TSS as u32,
            limit: 0x2b,
            attributes: 0x83,
            ..Default::default()
        };
        descriptor(&vm, 0x2060, 0x3800, 0x3f, 0x82);
        cpu.cpu.legacy.ldtr = SystemSegment {
            selector: 0x60,
            cache: SegmentCache {
                base: 0x3800,
                limit: 0x3f,
                attributes: 0x82,
                ..Default::default()
            },
            ..Default::default()
        };
        let based = |attr| SegmentCache {
            base: BASE,
            ..cache(attr)
        };
        cpu.cpu.legacy.segments = [
            based(0x9b),
            based(0x93),
            based(0xf3),
            based(0x93),
            null(),
            null(),
        ];
        cpu.cpu.es = U_DATA;
        let at = |offset: u64| u64::from(BASE) + offset;
        vm.write_bytes(at(0x1000), b"\xcf").unwrap();
        vm.write_bytes(at(0x1234), b"\xcd\x40\x90").unwrap();
        vm.write_bytes(at(0x5000), b"\xcf").unwrap();
        gate(&vm, 0x40, K_CODE, 0x5000, 0xe6);
        cpu.cpu.gpr[4] = 0x8f00;
        for (i, word) in [0x1234, u64::from(U_CODE), 0x3202, 0x7000, u64::from(U_DATA)]
            .into_iter()
            .enumerate()
        {
            put(&vm, at(0x8f00 + 2 * i as u64), 2, word);
        }
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.legacy.cpl, cpu.cpu.cs, cpu.cpu.rip),
            (3, U_CODE, 0x1234)
        );
        assert_eq!((cpu.cpu.ss, cpu.cpu.gpr[4]), (U_DATA, 0x7000));
        assert_eq!(cpu.cpu.legacy.segments[0], based(0xfb));
        assert_eq!(cpu.cpu.legacy.segments[3], based(0xf3));
        assert_eq!((cpu.cpu.rflags_system >> 12) & 7, 3);
        assert!(cpu.cpu.flags.if_);
        // Decision O6: the DPL-0 DS is nulled, the DPL-3 ES kept.
        assert_eq!((cpu.cpu.ds, cpu.cpu.legacy.segments[1].unusable), (0, 1));
        assert_eq!(
            (cpu.cpu.es, cpu.cpu.legacy.segments[2]),
            (U_DATA, based(0xf3))
        );
        // INT 40h through the DPL-3 interrupt gate switches to SS0:SP0.
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (
                cpu.cpu.legacy.cpl,
                cpu.cpu.cs,
                cpu.cpu.rip,
                cpu.cpu.ss,
                cpu.cpu.gpr[4]
            ),
            (0, K_CODE, 0x5000, K_STACK, 0x9000 - 10)
        );
        assert!(!cpu.cpu.flags.if_);
        assert_eq!(
            frame(&vm, at(0x9000 - 10), 2, 5),
            [0x1236, u64::from(U_CODE), 0x3202, 0x7000, u64::from(U_DATA)]
        );
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (
                cpu.cpu.legacy.cpl,
                cpu.cpu.cs,
                cpu.cpu.rip,
                cpu.cpu.ss,
                cpu.cpu.gpr[4]
            ),
            (3, U_CODE, 0x1236, U_DATA, 0x7000)
        );
        assert_eq!((cpu.cpu.rflags_system >> 12) & 7, 3);
        assert!(cpu.cpu.flags.if_);
        assert_eq!((cpu.cpu.ds, cpu.cpu.es), (0, U_DATA));
    }
}

#[test]
fn lock_is_iopl_sensitive_on_the_80286_only() {
    // 286 PRM App. B LOCK p. B-68: "#GP(0) if the current privilege level is
    // bigger (less privileged) than the I/O privilege level"; §3.11.2
    // p. 3-29. The test precedes the operand (decision O10, ASSUMED). The
    // 386 PRM §17 LOCK pp. 17-100–101 lists #UD only, for a form outside its
    // list (MOV and MOVS among them), and no IOPL test.
    const XCHG: &[u8] = b"\xf0\x86\x06\x00\x60"; // LOCK XCHG [6000h],AL
    const MOV: &[u8] = b"\xf0\x88\x07"; // LOCK MOV [BX],AL
    const MOVSB: &[u8] = b"\xf0\xa4"; // LOCK MOVSB
    for model in MODELS {
        for iopl in [0u64, 3] {
            for code in [XCHG, MOV, MOVSB] {
                let (vm, mut cpu) = world(model, 3, code, ExceptionDelivery::Outward);
                cpu.cpu.rflags_system = iopl << 12;
                cpu.cpu.gpr[0] = 0x5a;
                cpu.cpu.gpr[3] = 0x6000;
                cpu.cpu.gpr[6] = 0x6000;
                cpu.cpu.gpr[7] = 0x6100;
                put(&vm, 0x6000, 1, 0xa5);
                if !wide(model) && iopl == 0 {
                    fault(&mut cpu, &vm, 13, Some(0));
                } else if wide(model) && code != XCHG {
                    fault(&mut cpu, &vm, 6, None);
                } else if code == MOVSB {
                    // The locked string's handoff, as at CPL 0 (80286 PRM
                    // App. C item 11 p. C-2): nothing done.
                    let before = format!("{:?}", cpu.cpu);
                    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    assert_eq!(run.steps(), 0);
                    assert!(
                        matches!(
                            run.stop(),
                            StepStop::Exit(Exit::UnknownInstruction {
                                addr: 0x1000,
                                len: 2,
                                ..
                            })
                        ),
                        "{:?}",
                        run.stop()
                    );
                    assert_eq!(format!("{:?}", cpu.cpu), before);
                } else {
                    steps(&mut cpu, &vm, 1);
                    assert_eq!(value(&vm, 0x6000, 1), 0x5a);
                    if code == XCHG {
                        assert_eq!(cpu.cpu.gpr[0] & 0xff, 0xa5);
                    }
                    continue;
                }
                assert_eq!(value(&vm, 0x6000, 2), 0xa5, "{code:02x?}");
            }
        }
    }
}

#[test]
fn port_io_above_iopl_on_the_80386_faults_through_a_286_tss() {
    // 386 PRM §17.2.2.5 p. 17-15, I-O-Permission: "IF TSS type is 286 THEN
    // RETURN FALSE"; a 286 TSS has no I/O map (SDM Vol 3A §10.6 p. 10-18).
    // The access is #GP(0). With a 386 TSS, a missing bitmap denies access too.
    let forms: [&[u8]; 6] = [
        b"\xe4\x60",
        b"\xe6\x80",
        b"\xec",
        b"\x6c",
        b"\x6e",
        b"\xf3\x6e",
    ];
    for code in forms {
        for tss386 in [false, true] {
            for iopl in [0u64, 3] {
                let (vm, mut cpu) = world(CpuModel::I80386, 3, code, ExceptionDelivery::Outward);
                if !tss386 {
                    descriptor(&vm, 0x2028, TSS as u32, 0x2b, 0x83);
                    cpu.cpu.legacy.tr.cache.limit = 0x2b;
                    cpu.cpu.legacy.tr.cache.attributes = 0x83;
                }
                cpu.cpu.rflags_system = iopl << 12;
                cpu.cpu.gpr[1] = 2;
                cpu.cpu.gpr[6] = 0x2000;
                cpu.cpu.gpr[7] = 0x2000;
                if iopl == 3 {
                    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    assert!(
                        matches!(run.stop(), StepStop::Exit(Exit::PortIo { .. })),
                        "{:?}",
                        run.stop()
                    );
                } else {
                    fault(&mut cpu, &vm, 13, Some(0));
                }
            }
        }
    }
}

#[test]
fn rep_ins_and_outs_with_a_zero_count_make_no_io_test() {
    // SDM Vol 2B REP Operation p. 4-564: "WHILE CountReg ≠ 0 DO ... Execute
    // associated string instruction", so a zero count executes nothing and
    // makes no I/O test at CPL > IOPL (the maintainer's decision D-3). The
    // count is CX with a 16-bit address size and ECX with a 32-bit one.
    for model in MODELS {
        for code in [&b"\xf3\x6c"[..], b"\xf3\x6f", b"\xf2\x6e"] {
            let (vm, mut cpu) = world(model, 3, code, ExceptionDelivery::Outward);
            cpu.cpu.gpr[1] = 0x1_0000;
            steps(&mut cpu, &vm, 1);
            assert_eq!(
                (cpu.cpu.rip, cpu.cpu.gpr[1], cpu.cpu.legacy.cpl),
                (0x1002, 0x1_0000, 3)
            );
            // A count of 1 makes the test.
            let (vm, mut cpu) = world(model, 3, code, ExceptionDelivery::Outward);
            cpu.cpu.gpr[1] = 1;
            fault(&mut cpu, &vm, 13, Some(0));
        }
    }
    // ADDR32 REP INSB counts ECX: CX = 0 with ECX = 10000h makes the test.
    for ecx in [0u64, 0x1_0000] {
        let (vm, mut cpu) = world(
            CpuModel::I80386,
            3,
            b"\x67\xf3\x6c",
            ExceptionDelivery::Outward,
        );
        cpu.cpu.gpr[1] = ecx;
        if ecx == 0 {
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.rip, 0x1003);
        } else {
            fault(&mut cpu, &vm, 13, Some(0));
        }
    }
}

#[test]
fn same_level_far_transfers_and_segment_checks_at_cpl_3() {
    // 286 PRM App. B CALL p. B-24, JMP p. B-57, MOV p. B-75, INT p. B-49;
    // 386 PRM §17 CALL p. 17-39, JMP p. 17-85, MOV p. 17-97, INT p. 17-72.
    // A non-conforming target needs RPL <= CPL and DPL = CPL, a conforming
    // one DPL <= CPL, else #GP(selector); the CPL is kept and CS.RPL is CPL
    // (SDM Vol 2A CALL p. 3-131 "CS(RPL) := CPL"). MOV SS needs RPL = CPL
    // and DPL = CPL, else #GP(selector).
    let far = |op: u8, selector: u16| {
        let [low, high] = selector.to_le_bytes();
        vec![op, 0x00, 0x20, low, high]
    };
    for model in MODELS {
        for op in [0xea, 0x9a] {
            // To DPL-3 code, then RETF back for the CALL.
            let (vm, mut cpu) = world(model, 3, &far(op, U_CODE), ExceptionDelivery::Outward);
            vm.write_bytes(0x2000, b"\xcb").unwrap();
            steps(&mut cpu, &vm, 1);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl),
                (U_CODE, 0x2000, 3)
            );
            if op == 0x9a {
                assert_eq!(frame(&vm, 0x7ffc, 2, 2), [0x1005, u64::from(U_CODE)]);
                steps(&mut cpu, &vm, 1);
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4], cpu.cpu.legacy.cpl),
                    (U_CODE, 0x1005, 0x8000, 3)
                );
            }
            // To conforming DPL-0 code: CPL 3 kept, CS.RPL = 3.
            let (vm, mut cpu) = world(model, 3, &far(op, CONF0), ExceptionDelivery::Outward);
            steps(&mut cpu, &vm, 1);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl),
                (CONF0 | 3, 0x2000, 3)
            );
            // To non-conforming DPL-0 code from ring 3, and to DPL-3 code
            // from ring 2 (DPL != CPL).
            let (vm, mut cpu) = world(model, 3, &far(op, K_CODE), ExceptionDelivery::Outward);
            fault(&mut cpu, &vm, 13, Some(u32::from(K_CODE)));
            let (vm, mut cpu) = world(model, 2, &far(op, U_CODE), ExceptionDelivery::Outward);
            fault(&mut cpu, &vm, 13, Some(u32::from(U_CODE & !3)));
        }
        // MOV SS,AX with RPL 0, and with RPL 3 naming DPL-0 data.
        for (selector, code) in [(U_DATA & !3, U_DATA & !3), (K_STACK | 3, K_STACK)] {
            let (vm, mut cpu) = world(model, 3, b"\x8e\xd0", ExceptionDelivery::Outward);
            cpu.cpu.gpr[0] = u64::from(selector);
            fault(&mut cpu, &vm, 13, Some(u32::from(code)));
        }
        // INT 40h through a DPL-3 task gate switches tasks from ring 3; the
        // incoming ring-0 task links back to the ring-3 one.
        let (vm, mut cpu) = world(model, 3, b"\xcd\x40", ExceptionDelivery::Outward);
        incoming_task(&vm, model, K_CODE, K_STACK, K_STACK);
        gate(&vm, 0x40, 0x60, 0, 0xe5);
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (
                cpu.cpu.legacy.tr.selector,
                cpu.cpu.cs,
                cpu.cpu.rip,
                cpu.cpu.legacy.cpl
            ),
            (0x60, K_CODE, 0x1800, 0)
        );
        assert_eq!(cpu.cpu.rflags_system & 0x4000, 0x4000);
        assert_eq!(value(&vm, 0x3100, 2), u64::from(TSS_SEL));
        // Through a DPL-0 task gate it is #GP(vector * 8 + 2).
        let (vm, mut cpu) = world(model, 3, b"\xcd\x40", ExceptionDelivery::Outward);
        incoming_task(&vm, model, K_CODE, K_STACK, K_STACK);
        gate(&vm, 0x40, 0x60, 0, 0x85);
        fault(&mut cpu, &vm, 13, Some(0x40 * 8 + 2));
    }
}

#[test]
fn outward_return_bounds_count_the_operand_size_and_n() {
    // The top 8+n or 10 bytes, or 16+n or 20 with a 32-bit operand, must lie
    // within the stack limits, else #SS(0) (286 PRM App. B RET p. B-95, IRET
    // p. B-52; 386 PRM §17 RET p. 17-143, IRET p. 17-77; SDM Vol 2B RET
    // p. 4-572). The stack (SS.B = 0, limit FFFFh) ends the frame at FFFFh
    // exactly, or two bytes beyond. RETF n reads SP and SS past the n bytes.
    for model in MODELS {
        for (code, size, n, iret) in [
            (&b"\xcb"[..], 2u64, 0u64, false),
            (b"\xca\x04\x00", 2, 4, false),
            (b"\xcf", 2, 0, true),
            (b"\x66\xcb", 4, 0, false),
            (b"\x66\xca\x06\x00", 4, 6, false),
            (b"\x66\xcf", 4, 0, true),
        ] {
            if !wide(model) && size == 4 {
                continue;
            }
            let bytes = 4 * size + n + if iret { size } else { 0 };
            for over in [0u64, 2] {
                let (vm, mut cpu) = world(model, 0, code, ExceptionDelivery::Outward);
                let top = 0x1_0000 - bytes + over;
                let mut at = top;
                let middle = if iret { size } else { n };
                for (word, width) in [
                    (0x1234, size),
                    (u64::from(U_CODE), size),
                    (2, middle),
                    (0x7000, size),
                    (u64::from(U_DATA), size),
                ] {
                    if width != 0 && at + width <= 0x1_0000 {
                        put(&vm, at, width as usize, word);
                    }
                    at += width;
                }
                cpu.cpu.gpr[4] = top;
                if over == 0 {
                    steps(&mut cpu, &vm, 1);
                    assert_eq!(
                        (cpu.cpu.cs, cpu.cpu.ss, cpu.cpu.legacy.cpl, cpu.cpu.gpr[4]),
                        (U_CODE, U_DATA, 3, 0x7000 + n)
                    );
                } else {
                    fault(&mut cpu, &vm, 12, Some(0));
                }
            }
        }
    }
}

#[test]
fn outward_frames_and_descriptors_on_devices_hand_off_exactly() {
    for model in MODELS {
        // The SP and SS slots of the frame are a device: each read stops with
        // nothing committed, and the answers complete the return.
        let (mut vm, mut cpu) = world(model, 0, b"\xcb", ExceptionDelivery::Outward);
        outer_frame(&vm, &mut cpu, 2, 0x1234, U_CODE, &[], 0x7000, U_DATA);
        trap_range(&mut vm, 0x8f04, 4);
        for (addr, answer) in [(0x8f04u64, 0x7000u64), (0x8f06, u64::from(U_DATA))] {
            let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(run.stop(), StepStop::Exit(Exit::MmioRead { addr: a, size: 2, .. }) if *a == addr),
                "{:?}",
                run.stop()
            );
            assert_eq!(run.steps(), 0);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.ss, cpu.cpu.legacy.cpl, cpu.cpu.rip),
                (K_CODE, K_STACK, 0, 0x1000)
            );
            cpu.complete_mmio_read(answer);
        }
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.ss, cpu.cpu.legacy.cpl, cpu.cpu.gpr[4]),
            (U_CODE, U_DATA, 3, 0x7000)
        );
        // The return CS or SS descriptor is a device and its accessed bit is
        // clear: its locked update is refused before the other one changes.
        for (address, high, other) in [
            (0x2018u64, 0x0000_fa00u64, 0x2025u64),
            (0x2020, 0x0000_f200, 0x201d),
        ] {
            let (mut vm, mut cpu) = world(model, 0, b"\xcb", ExceptionDelivery::Outward);
            outer_frame(&vm, &mut cpu, 2, 0x1234, U_CODE, &[], 0x7000, U_DATA);
            trap_range(&mut vm, address, 8);
            let mut answers = [0x0000_ffff, high].into_iter();
            loop {
                let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                if let StepStop::Exit(Exit::MmioRead { .. }) = run.stop() {
                    cpu.complete_mmio_read(answers.next().unwrap());
                    continue;
                }
                assert!(
                    matches!(
                        run.stop(),
                        StepStop::Exit(Exit::UnsupportedSystemContext {
                            operation: UnsupportedSystemOperation::LegacyInstruction,
                            ..
                        })
                    ),
                    "{:?}",
                    run.stop()
                );
                assert_eq!(run.steps(), 0);
                break;
            }
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.ss, cpu.cpu.legacy.cpl, cpu.cpu.gpr[4]),
                (K_CODE, K_STACK, 0, 0x8f00)
            );
            assert_eq!(value(&vm, other, 1) & 1, 0);
        }
    }
}

#[test]
fn an_outward_return_to_ring_2_nulls_by_the_cached_dpl_only() {
    // SDM Vol 2B RET p. 4-574, Vol 2A IRET p. 3-495 (decision O6): data or
    // non-conforming code whose cached DPL is below the new CPL is nulled;
    // the rest is kept as it is. Return to ring 2: DS holds DPL-1 data
    // (nulled), ES a null selector over a usable DPL-3 data cache (kept:
    // the test reads the cache, not the selector); on the 386, FS holds
    // DPL-2 data (kept) and GS is already null (stays null).
    for model in MODELS {
        let (vm, mut cpu) = world(model, 0, b"\xcb", ExceptionDelivery::Outward);
        outer_frame(&vm, &mut cpu, 2, 0x1234, R2_CODE, &[], 0x7000, R2_STACK);
        cpu.cpu.ds = R1_STACK;
        cpu.cpu.legacy.segments[1] = cache(0xb3);
        cpu.cpu.es = 0;
        cpu.cpu.legacy.segments[2] = cache(0xf3);
        if wide(model) {
            cpu.cpu.fs = R2_STACK;
            cpu.cpu.legacy.segments[4] = cache(0xd3);
        }
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.ss, cpu.cpu.legacy.cpl),
            (R2_CODE, R2_STACK, 2)
        );
        assert_eq!((cpu.cpu.ds, cpu.cpu.legacy.segments[1].unusable), (0, 1));
        assert_eq!((cpu.cpu.es, cpu.cpu.legacy.segments[2]), (0, cache(0xf3)));
        if wide(model) {
            assert_eq!(
                (cpu.cpu.fs, cpu.cpu.legacy.segments[4]),
                (R2_STACK, cache(0xd3))
            );
            assert_eq!((cpu.cpu.gs, cpu.cpu.legacy.segments[5].unusable), (0, 1));
        }
    }
}

#[test]
fn a_tf_trap_after_an_outward_iret_is_delivered_to_ring_0() {
    // 286 PRM §9.7.1 p. 9-14: TF loaded by IRET traps after the next
    // instruction; the #DB trap from ring 3 enters ring 0 on the TSS stack
    // (286 PRM p. B-49; SDM INT p. 3-473) and the handler runs with TF and
    // NT clear. The frame's FLAGS hold the TF and NT the IRET loaded.
    for model in MODELS {
        let (vm, mut cpu) = world(model, 0, b"\xcf", ExceptionDelivery::InGuest);
        vm.write_bytes(0x1234, b"\x90").unwrap();
        gate(&vm, 1, K_CODE, 0x5000, 0x86);
        vm.write_bytes(0x5000, b"\x90").unwrap();
        outer_frame(
            &vm,
            &mut cpu,
            2,
            0x1234,
            U_CODE,
            &[0x02, 0x41],
            0x7000,
            U_DATA,
        );
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.legacy.cpl, 3);
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.ss, cpu.cpu.legacy.cpl),
            (K_CODE, 0x5000, K_STACK, 0)
        );
        assert_eq!(cpu.cpu.rflags_system & 0x4100, 0);
        assert_eq!(
            frame(&vm, 0x9000 - 10, 2, 5),
            [0x1235, u64::from(U_CODE), 0x4102, 0x7000, u64::from(U_DATA)]
        );
    }
}

#[test]
fn a_failing_nmi_delivery_from_ring_3_carries_ext() {
    // SDM Vol 2A INT p. 3-471: an externally generated event sets EXT in the
    // error code of a fault raised while delivering it. A null SS0 is
    // #TS(EXT) (decision O5b).
    for model in MODELS {
        let (vm, mut cpu) = world(model, 3, b"\x90", ExceptionDelivery::Outward);
        gate(&vm, 2, K_CODE, 0x5000, 0x86);
        put(&vm, tss_stack(model, 0).0, 2, 0);
        cpu.raise_nmi().unwrap();
        fault(&mut cpu, &vm, 10, Some(1));
    }
}

// Call gates (xj-srh.9, second landing). The gate selector GATE | 3 names a
// DPL-3 gate at GDT 70h; CPL-3 code at 1000h calls or jumps through it with a
// far pointer whose offset is ignored.
const GATE: u16 = 0x70;
const LDT: u64 = 0x2400;
/// A call-gate image: offset 15:0, selector, the parameter count in byte 4,
/// the access byte, offset 31:16 (286 PRM §7.5.1.1 Fig. 7-10 p. 7-17;
/// 386 PRM §6.3.4 Fig. 6-5 p. 6-11).
fn call_gate(vm: &Vm, address: u64, selector: u16, offset: u32, count: u8, attr: u8) {
    let low = (u32::from(selector) << 16) | (offset & 0xffff);
    let high = (offset & 0xffff_0000) | (u32::from(attr) << 8) | u32::from(count);
    put(vm, address, 4, u64::from(low));
    put(vm, address + 4, 4, u64::from(high));
}
fn far_ptr(op: u8, selector: u16) -> Vec<u8> {
    let [low, high] = selector.to_le_bytes();
    vec![op, 0x34, 0x12, low, high]
}
/// The gate types a model has: type 4 on both, type 12 on the 80386 only.
fn gate_types(model: CpuModel) -> &'static [u8] {
    if wide(model) {
        &[0x04, 0x0c]
    } else {
        &[0x04]
    }
}
fn gate_size(kind: u8) -> usize {
    if kind & 8 != 0 {
        4
    } else {
        2
    }
}
/// An LDT at 2400h, limit 3Fh, in LDTR (selector 68h).
fn with_ldt(vm: &Vm, cpu: &mut Vcpu) {
    descriptor(vm, 0x2068, LDT as u32, 0x3f, 0x82);
    cpu.cpu.legacy.ldtr = SystemSegment {
        selector: 0x68,
        cache: SegmentCache {
            base: LDT as u32,
            limit: 0x3f,
            attributes: 0x82,
            ..Default::default()
        },
        ..Default::default()
    };
}
/// CPL-3 code calls (9A) or jumps (EA) through GATE | 3, a DPL-3 gate of
/// `kind` to `target`:5000h with `count` parameters.
fn gate_world(model: CpuModel, op: u8, kind: u8, target: u16, count: u8) -> (Vm, Vcpu) {
    gate_world_delivery(model, op, kind, target, count, ExceptionDelivery::Outward)
}
fn gate_world_delivery(
    model: CpuModel,
    op: u8,
    kind: u8,
    target: u16,
    count: u8,
    delivery: ExceptionDelivery,
) -> (Vm, Vcpu) {
    let (vm, cpu) = world(model, 3, &far_ptr(op, GATE | 3), delivery);
    call_gate(&vm, 0x2070, target, 0x5000, count, 0xe0 | kind);
    (vm, cpu)
}

#[test]
fn a_call_gate_from_ring_3_enters_ring_0_on_the_tss_stack_and_retf_returns() {
    // The embedder's shape: CPL 3, a selector with RPL 3 naming a present
    // DPL-3 gate with no parameters, to DPL-0 non-conforming code (selector
    // RPL 0), with SS0:SP0 in the TSS. 286 PRM App. B CALL pp. B-24–25 and
    // §7.5.1.4 Fig. 7-12 p. 7-21; 386 PRM §17 CALL pp. 17-39–40; SDM Vol 2A
    // CALL pp. 3-131–132 and Vol 3A §6.8.5 p. 6-18. The new stack holds old
    // SS, old (E)SP, CS and (E)IP at the gate's width; CPL becomes 0 and
    // CS.RPL the CPL. RETF (RETFD through a 32-bit gate) returns to ring 3
    // (286 PRM p. B-95; SDM Vol 2B RET p. 4-573).
    for model in MODELS {
        for &kind in gate_types(model) {
            let size = gate_size(kind);
            let (vm, mut cpu) = gate_world(model, 0x9a, kind, K_CODE, 0);
            let ret: &[u8] = if size == 4 { b"\x66\xcb" } else { b"\xcb" };
            vm.write_bytes(0x5000, ret).unwrap();
            cpu.cpu.flags.if_ = true;
            steps(&mut cpu, &vm, 1);
            let sp = 0x9000 - 4 * size as u64;
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.ss, cpu.cpu.gpr[4]),
                (K_CODE, 0x5000, K_STACK, sp)
            );
            assert_eq!(cpu.cpu.legacy.cpl, 0);
            assert_eq!(
                frame(&vm, sp, size, 4),
                [0x1005, u64::from(U_CODE), 0x8000, u64::from(U_DATA)]
            );
            // The loaded caches, and the accessed bits in the GDT.
            let loaded = |attr| SegmentCache {
                base: 0,
                limit: 0xffff,
                attributes: attr,
                unusable: 0,
                reserved: 0,
            };
            assert_eq!(cpu.cpu.legacy.segments[0], loaded(0x9b));
            assert_eq!(cpu.cpu.legacy.segments[3], loaded(0x93));
            assert_eq!((value(&vm, 0x200d, 1), value(&vm, 0x2015, 1)), (0x9b, 0x93));
            // Nothing else changes: IF, DS and ES are kept, the old stack is
            // untouched.
            assert!(cpu.cpu.flags.if_);
            assert_eq!((cpu.cpu.ds, cpu.cpu.es), (U_DATA, U_DATA));
            assert_eq!(value(&vm, 0x7ff0, 8) | value(&vm, 0x7ff8, 8), 0);
            steps(&mut cpu, &vm, 1);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.ss, cpu.cpu.gpr[4]),
                (U_CODE, 0x1005, U_DATA, 0x8000)
            );
            assert_eq!(cpu.cpu.legacy.cpl, 3);
            assert_eq!((cpu.cpu.ds, cpu.cpu.es), (U_DATA, U_DATA));
        }
    }
}

#[test]
fn call_gates_in_the_ldt_and_to_ldt_code_enter_ring_0() {
    // A call gate may reside in the GDT or the LDT (286 PRM §7.5.1.1
    // p. 7-17); its target selector may name either table. The gate at LDT
    // index 1 (selector 0Fh) targets K_CODE; the GDT gate targets DPL-0 code
    // at LDT index 2 (selector 14h).
    for model in MODELS {
        let (vm, mut cpu) = world(model, 3, &far_ptr(0x9a, 0x0f), ExceptionDelivery::Outward);
        with_ldt(&vm, &mut cpu);
        call_gate(&vm, LDT + 8, K_CODE, 0x5000, 0, 0xe4);
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl),
            (K_CODE, 0x5000, 0)
        );
        let (vm, mut cpu) = gate_world(model, 0x9a, 4, 0x14, 0);
        with_ldt(&vm, &mut cpu);
        descriptor(&vm, LDT + 0x10, 0, 0xffff, 0x9a);
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl, cpu.cpu.ss),
            (0x14, 0x5000, 0, K_STACK)
        );
        assert_eq!(value(&vm, LDT + 0x15, 1), 0x9b);
    }
}

#[test]
fn a_call_gate_copies_its_parameters_and_retf_n_releases_them_on_both_stacks() {
    // The count in gate byte 4 names words for a type-4 gate and doublewords
    // for a type-12 gate (SDM Vol 3A §6.8.3 p. 6-14; 386 PRM §6.3.4.1
    // p. 6-14). They are copied in their order between old (E)SP and the
    // return address (SDM Vol 3A §6.8.5 step 7 p. 6-18, Fig. 6-13 p. 6-19).
    // RETF n pops the return address, skips n bytes, pops SS:SP and adds n to
    // the outer (E)SP (decision O9; SDM Vol 2B RET p. 4-574). With a new
    // stack that overlaps the parameters, every parameter is read before the
    // first write (decision O11).
    for model in MODELS {
        for &kind in gate_types(model) {
            for alias in [false, true] {
                let size = gate_size(kind);
                let n = 2 * size as u64;
                let (vm, mut cpu) = gate_world(model, 0x9a, kind, K_CODE, 2);
                let ret: &[u8] = if size == 4 {
                    b"\x66\xca\x08\x00"
                } else {
                    b"\xca\x04\x00"
                };
                vm.write_bytes(0x5000, ret).unwrap();
                let (first, second) = if size == 4 {
                    (0x1111_1111, 0x2222_2222)
                } else {
                    (0x1111, 0x2222)
                };
                // The caller pushed `first`, then `second`.
                let old = 0x8000 - n;
                cpu.cpu.gpr[4] = old;
                put(&vm, old, size, second);
                put(&vm, old + size as u64, size, first);
                // Aliased: SP0 is the caller's SP before the pushes, so the
                // first two frame slots overwrite both parameters.
                let top = if alias { 0x8000 } else { 0x9000 };
                set_tss_stack(&vm, model, 0, K_STACK, top);
                steps(&mut cpu, &vm, 1);
                let sp = u64::from(top) - 6 * size as u64;
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.gpr[4], cpu.cpu.legacy.cpl),
                    (K_CODE, sp, 0)
                );
                assert_eq!(
                    frame(&vm, sp, size, 6),
                    [
                        0x1005,
                        u64::from(U_CODE),
                        second,
                        first,
                        old,
                        u64::from(U_DATA)
                    ]
                );
                steps(&mut cpu, &vm, 1);
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.ss, cpu.cpu.gpr[4]),
                    (U_CODE, 0x1005, U_DATA, 0x8000)
                );
                assert_eq!(cpu.cpu.legacy.cpl, 3);
            }
        }
    }
}

#[test]
fn a_call_gate_counts_room_on_the_new_stack_by_gate_width_and_parameters() {
    // Room for the parameters plus 8 bytes, or plus 16 through a type-12
    // gate (286 PRM p. B-25; 386 PRM p. 17-40; SDM p. 3-132). SS0 is 80h,
    // DPL-0 data with limit 0FFFh: SP0 = 1000h fits exactly, SP0 = 1001h
    // ends one byte beyond the limit. The fault is #SS(0) on the 80286 and
    // #SS(new SS) on the 80386 (decision O1).
    for model in MODELS {
        for &kind in gate_types(model) {
            for count in [0u8, 3] {
                for top in [0x1000u32, 0x1001] {
                    let size = gate_size(kind) as u64;
                    let (vm, mut cpu) = gate_world(model, 0x9a, kind, K_CODE, count);
                    descriptor(&vm, 0x2080, 0, 0x0fff, 0x92);
                    set_tss_stack(&vm, model, 0, 0x80, top);
                    let used = (4 + u64::from(count)) * size;
                    if top == 0x1000 {
                        steps(&mut cpu, &vm, 1);
                        assert_eq!((cpu.cpu.ss, cpu.cpu.gpr[4]), (0x80, 0x1000 - used));
                    } else {
                        fault(&mut cpu, &vm, 12, Some(if wide(model) { 0x80 } else { 0 }));
                        assert_eq!(value(&vm, 0x2085, 1), 0x92);
                    }
                }
            }
        }
    }
}

/// One way to break a call through GATE, in the check order of 286 PRM
/// App. B CALL pp. B-24–25 and SDM Vol 2A CALL pp. 3-131–132, with the
/// expected vector and error code. Each world is a CPL-3 CALL through a
/// DPL-3 type-4 gate to K_CODE with two parameters.
fn break_call(vm: &Vm, cpu: &mut Vcpu, model: CpuModel, index: usize) -> (u8, u16) {
    let gate = |selector, attr| call_gate(vm, 0x2070, selector, 0x5000, 2, attr);
    match index {
        // Gate DPL < CPL, and gate DPL < RPL (at CPL 2, RPL 3), #GP(gate).
        0 => {
            gate(K_CODE, 0xc4);
            (13, GATE)
        }
        1 => {
            enter(cpu, 2);
            gate(K_CODE, 0xc4);
            (13, GATE)
        }
        // Gate not present, #NP(gate); with its DPL also too low, #GP first.
        2 => {
            gate(K_CODE, 0x64);
            (11, GATE)
        }
        3 => {
            gate(K_CODE, 0x44);
            (13, GATE)
        }
        // Gate CS null #GP(0); beyond the GDT limit #GP(CS).
        4 => {
            gate(3, 0xe4);
            (13, 0)
        }
        5 => {
            gate(0x1f8, 0xe4);
            (13, 0x1f8)
        }
        // Not code; code with DPL > CPL (from ring 0 to DPL-3 code).
        6 => {
            gate(K_STACK, 0xe4);
            (13, K_STACK)
        }
        7 => {
            enter(cpu, 0);
            gate(U_CODE, 0xe4);
            (13, U_CODE & !3)
        }
        // The target not present, #NP(CS) (decision O7); not present with
        // DPL > CPL as well, #GP(CS), as the type and DPL tests come first.
        8 => {
            descriptor(vm, 0x2078, 0, 0xffff, 0x1a);
            gate(0x78, 0xe4);
            (11, 0x78)
        }
        9 => {
            enter(cpu, 2);
            descriptor(vm, 0x2078, 0, 0xffff, 0x7a);
            gate(0x78, 0xe4);
            (13, 0x78)
        }
        // TSS: the ring-0 SS field one byte beyond the TR limit, #TS(TR).
        10 => {
            cpu.cpu.legacy.tr.cache.limit = if wide(model) { 8 } else { 4 };
            (10, TSS_SEL)
        }
        // A null SS0 #TS(0); SS0 with RPL 3 #TS(SS); SS0 not present #SS(SS).
        11 => {
            set_tss_stack(vm, model, 0, 0, 0x9000);
            (10, 0)
        }
        12 => {
            set_tss_stack(vm, model, 0, K_STACK | 3, 0x9000);
            (10, K_STACK)
        }
        13 => {
            descriptor(vm, 0x2080, 0, 0xffff, 0x12);
            set_tss_stack(vm, model, 0, 0x80, 0x9000);
            (12, 0x80)
        }
        // No room for 12 bytes (decision O1); with a bad offset too, room
        // first.
        14 | 15 => {
            set_tss_stack(vm, model, 0, K_STACK, 0x0b);
            if index == 15 {
                call_gate(vm, 0x2070, 0x88, 0x5000, 2, 0xe4);
            }
            (12, if wide(model) { K_STACK } else { 0 })
        }
        // The gate offset beyond the new CS limit (88h: limit 0FFFh), #GP(0);
        // with a parameter beyond the old SS limit too, the offset first
        // (decision O11).
        16 | 17 => {
            call_gate(vm, 0x2070, 0x88, 0x5000, 2, 0xe4);
            if index == 17 {
                cpu.cpu.legacy.segments[3].limit = 0x7ffd;
            }
            (13, 0)
        }
        // A parameter beyond the old SS limit, #SS(0) (decision O11).
        18 => {
            cpu.cpu.legacy.segments[3].limit = 0x7ffd;
            (12, 0)
        }
        // An unusable TR, #TS(TR) (as for an interrupt, SDM p. 3-472).
        19 => {
            cpu.cpu.legacy.tr.cache.unusable = 1;
            (10, TSS_SEL)
        }
        // SS0 beyond the GDT limit, SS0 DPL != 0 (20h: DPL-3 data, RPL 0) and
        // SS0 read-only data, each #TS(SS) (286 PRM p. B-25; SDM p. 3-131).
        20 => {
            set_tss_stack(vm, model, 0, 0x1f8, 0x9000);
            (10, 0x1f8)
        }
        21 => {
            set_tss_stack(vm, model, 0, U_DATA & !3, 0x9000);
            (10, U_DATA & !3)
        }
        22 => {
            descriptor(vm, 0x2080, 0, 0xffff, 0x90);
            set_tss_stack(vm, model, 0, 0x80, 0x9000);
            (10, 0x80)
        }
        _ => unreachable!(),
    }
}
const CALL_FAULTS: usize = 23;
/// The rows of `break_call` that break the ring-0 stack in the TSS, whose
/// fault an inward handler could not take.
fn breaks_the_ring_0_stack(index: usize) -> bool {
    matches!(index, 10..=15 | 19..=22)
}

#[test]
fn call_gate_faults_precede_every_change() {
    for model in MODELS {
        for index in 0..CALL_FAULTS {
            let (vm, mut cpu) = gate_world(model, 0x9a, 4, K_CODE, 2);
            descriptor(&vm, 0x2088, 0, 0x0fff, 0x9a);
            cpu.cpu.gpr[4] = 0x7ffc;
            put(&vm, 0x7ffc, 4, 0x1111_2222);
            let (vector, code) = break_call(&vm, &mut cpu, model, index);
            fault(&mut cpu, &vm, vector, Some(u32::from(code)));
            // The new stack and every accessed bit are untouched.
            assert_eq!(value(&vm, 0x8ff0, 8) | value(&vm, 0x8ff8, 8), 0, "{index}");
            for at in [0x200d, 0x2015, 0x2085] {
                assert_eq!(value(&vm, at, 1) & 1, 0, "{index} {at:x}");
            }
        }
    }
}

#[test]
fn call_gate_faults_are_delivered_in_the_guest_with_the_callers_state() {
    // The rows of `call_gate_faults_precede_every_change` with InGuest
    // delivery: #GP, #NP, #TS and #SS through the IDT, with the error code
    // and the caller's CS:IP and SS:SP in the frame, so the faulting CALL
    // committed nothing. A handler at K_CODE takes the fault inward on
    // SS0:SP0 (or at ring 0 from CPL 0); where the row breaks the ring-0
    // stack, a conforming handler (CONF0) takes it at CPL 3 on the caller's
    // stack (SDM Vol 2A INT pp. 3-471-474; 286 PRM §9.4 p. 9-4).
    for model in MODELS {
        for index in 0..CALL_FAULTS {
            let (vm, mut cpu) =
                gate_world_delivery(model, 0x9a, 4, K_CODE, 2, ExceptionDelivery::InGuest);
            descriptor(&vm, 0x2088, 0, 0x0fff, 0x9a);
            cpu.cpu.gpr[4] = 0x7ffc;
            put(&vm, 0x7ffc, 4, 0x1111_2222);
            let (vector, code) = break_call(&vm, &mut cpu, model, index);
            let conforming = breaks_the_ring_0_stack(index);
            let handler = if conforming { CONF0 } else { K_CODE };
            gate(&vm, vector, handler, 0x5800, 0x86);
            vm.write_bytes(0x5800, b"\x90").unwrap();
            let cpl = cpu.cpu.legacy.cpl;
            let (cs, ss) = (cpu.cpu.cs, cpu.cpu.ss);
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.rip, 0x5801, "{index}");
            let flags = 2;
            if conforming {
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.legacy.cpl, cpu.cpu.ss, cpu.cpu.gpr[4]),
                    (CONF0 | 3, 3, U_DATA, 0x7ff4),
                    "{index}"
                );
                assert_eq!(
                    frame(&vm, 0x7ff4, 2, 4),
                    [u64::from(code), 0x1000, u64::from(U_CODE), flags],
                    "{index}"
                );
                assert_eq!(value(&vm, 0x8ff0, 8) | value(&vm, 0x8ff8, 8), 0, "{index}");
            } else if cpl == 0 {
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.legacy.cpl, cpu.cpu.ss, cpu.cpu.gpr[4]),
                    (K_CODE, 0, K_STACK, 0x7ff4),
                    "{index}"
                );
                assert_eq!(
                    frame(&vm, 0x7ff4, 2, 4),
                    [u64::from(code), 0x1000, u64::from(K_CODE), flags],
                    "{index}"
                );
            } else {
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.legacy.cpl, cpu.cpu.ss, cpu.cpu.gpr[4]),
                    (K_CODE, 0, K_STACK, 0x8ff4),
                    "{index}"
                );
                assert_eq!(
                    frame(&vm, 0x8ff4, 2, 6),
                    [
                        u64::from(code),
                        0x1000,
                        u64::from(cs),
                        flags,
                        0x7ffc,
                        u64::from(ss)
                    ],
                    "{index}"
                );
                assert_eq!(value(&vm, 0x8ff0, 4), 0, "{index}");
            }
            // The CALL's own target and new-stack descriptors stay unaccessed,
            // and the caller's parameters are intact.
            for at in [0x2085, 0x208d] {
                assert_eq!(value(&vm, at, 1) & 1, 0, "{index} {at:x}");
            }
            assert_eq!(value(&vm, 0x7ffc, 4), 0x1111_2222, "{index}");
        }
    }
}

#[test]
fn a_tf_trap_after_a_call_gate_is_delivered_at_the_new_cpl() {
    // TF set before a CALL through a gate from ring 3 to ring 0: the #DB
    // trap follows the CALL and is taken at CPL 0 on the new stack, below
    // the intact gate frame, with the saved CS:IP of the gate target and TF
    // in the pushed FLAGS; the handler runs with TF clear (286 PRM §9.7.1
    // p. 9-14, App. B INT p. B-49; 386 PRM §12.3.1.4 p. 12-8, §17 INT
    // p. 17-71).
    for model in MODELS {
        let (vm, mut cpu) =
            gate_world_delivery(model, 0x9a, 4, K_CODE, 0, ExceptionDelivery::InGuest);
        gate(&vm, 1, K_CODE, 0x5800, 0x86);
        vm.write_bytes(0x5000, b"\x90").unwrap();
        vm.write_bytes(0x5800, b"\x90").unwrap();
        cpu.cpu.rflags_system |= 0x100;
        steps(&mut cpu, &vm, 2);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl, cpu.cpu.ss),
            (K_CODE, 0x5801, 0, K_STACK)
        );
        assert_eq!(cpu.cpu.gpr[4], 0x8ff2);
        assert_eq!(
            frame(&vm, 0x8ff2, 2, 7),
            [
                0x5000,
                u64::from(K_CODE),
                0x102,
                0x1005,
                u64::from(U_CODE),
                0x8000,
                u64::from(U_DATA)
            ]
        );
        assert_eq!(cpu.cpu.rflags_system & 0x100, 0);
    }
}

#[test]
fn a_type_12_gate_on_the_80286_is_a_general_protection_fault() {
    // On the 80286 descriptor type 12 is no gate: CALL and JMP take the
    // "ELSE #GP (code segment selector)" branch (286 PRM App. B CALL p. B-25,
    // JMP p. B-58; DERIVED).
    for op in [0x9a, 0xea] {
        let (vm, mut cpu) = gate_world(CpuModel::I80286, op, 0x0c, K_CODE, 0);
        fault(&mut cpu, &vm, 13, Some(u32::from(GATE)));
    }
}

#[test]
fn jmp_and_same_level_calls_through_a_gate_stay_at_cpl() {
    // A JMP through a call gate reaches non-conforming code with DPL = CPL
    // or conforming code with DPL <= CPL, and never changes privilege; an
    // inward JMP is #GP(CS) (286 PRM App. B JMP p. B-57, §7.5.1.3 p. 7-19;
    // 386 PRM §17 JMP pp. 17-85–86; SDM Vol 3A Table 6-1 p. 6-16). A CALL
    // through a gate to the same level, or to conforming code, makes no stack
    // switch: it pushes CS and IP at the gate's width on the current stack
    // (286 PRM p. B-25; 386 PRM pp. 17-40–41; SDM Vol 2A CALL pp. 3-133–134).
    // CS.RPL is the CPL in every case.
    for model in MODELS {
        for &kind in gate_types(model) {
            let size = gate_size(kind);
            for (target, cs) in [(U_CODE, U_CODE), (CONF0, CONF0 | 3)] {
                let (vm, mut cpu) = gate_world(model, 0xea, kind, target, 3);
                steps(&mut cpu, &vm, 1);
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl),
                    (cs, 0x5000, 3)
                );
                assert_eq!((cpu.cpu.ss, cpu.cpu.gpr[4]), (U_DATA, 0x8000));
                let (vm, mut cpu) = gate_world(model, 0x9a, kind, target, 3);
                let ret: &[u8] = if size == 4 { b"\x66\xcb" } else { b"\xcb" };
                vm.write_bytes(0x5000, ret).unwrap();
                steps(&mut cpu, &vm, 1);
                let sp = 0x8000 - 2 * size as u64;
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl, cpu.cpu.ss),
                    (cs, 0x5000, 3, U_DATA)
                );
                assert_eq!(cpu.cpu.gpr[4], sp);
                assert_eq!(frame(&vm, sp, size, 2), [0x1005, u64::from(U_CODE)]);
                assert_eq!(value(&vm, 0x8ff0, 8) | value(&vm, 0x8ff8, 8), 0);
                steps(&mut cpu, &vm, 1);
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.gpr[4]),
                    (U_CODE, 0x1005, 0x8000)
                );
            }
            // Inward: #GP(K_CODE); and to DPL-0 code from ring 0 it runs.
            let (vm, mut cpu) = gate_world(model, 0xea, kind, K_CODE, 0);
            fault(&mut cpu, &vm, 13, Some(u32::from(K_CODE)));
            let (vm, mut cpu) = gate_world(model, 0xea, kind, K_CODE, 0);
            enter(&mut cpu, 0);
            steps(&mut cpu, &vm, 1);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl),
                (K_CODE, 0x5000, 0)
            );
        }
        // A 32-bit gate supplies offset 31:16 too; a 16-bit gate does not.
        if wide(model) {
            let (vm, mut cpu) = gate_world(model, 0xea, 0x0c, K_CODE, 0);
            enter(&mut cpu, 0);
            descriptor(&vm, 0x2008, 0, 0xffff_ffff, 0x809a);
            call_gate(&vm, 0x2070, K_CODE, 0x0081_5000, 0, 0xec);
            steps(&mut cpu, &vm, 1);
            assert_eq!(cpu.cpu.rip, 0x0081_5000);
        }
    }
}

#[test]
fn a_call_gate_frame_on_a_device_hands_off_and_replays_the_copied_parameters() {
    // The new stack aliases the caller's parameters, and its last slot (IP)
    // is a device: the frame write stops with nothing committed, after RAM
    // slots have overwritten the parameters. The replay copies the values
    // first read (decision O11; spec.md §5.2).
    for model in MODELS {
        let (mut vm, mut cpu) = gate_world(model, 0x9a, 4, K_CODE, 2);
        cpu.cpu.gpr[4] = 0x7ffc;
        put(&vm, 0x7ffc, 2, 0x2222);
        put(&vm, 0x7ffe, 2, 0x1111);
        set_tss_stack(&vm, model, 0, K_STACK, 0x8000);
        trap_range(&mut vm, 0x7ff4, 2);
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        assert!(
            matches!(
                run.stop(),
                StepStop::Exit(Exit::MmioWrite {
                    addr: 0x7ff4,
                    size: 2,
                    value: 0x1005,
                    ..
                })
            ),
            "{:?}",
            run.stop()
        );
        assert_eq!(run.steps(), 0);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.ss, cpu.cpu.legacy.cpl, cpu.cpu.gpr[4]),
            (U_CODE, U_DATA, 3, 0x7ffc)
        );
        assert_eq!(value(&vm, 0x7ffe, 2), u64::from(U_DATA));
        cpu.complete_mmio_write();
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl, cpu.cpu.gpr[4]),
            (K_CODE, 0x5000, 0, 0x7ff4)
        );
        assert_eq!(
            frame(&vm, 0x7ff6, 2, 5),
            [u64::from(U_CODE), 0x2222, 0x1111, 0x7ffc, u64::from(U_DATA)]
        );
    }
}

#[test]
fn a_call_gate_frame_beyond_the_retry_ledger_is_refused_before_any_device_transfer() {
    // An inward CALL whose frame copy needs more retry-ledger entries than
    // the 64 that `CpuState` holds is refused as `LegacyInstruction` before
    // its first parameter read, with nothing changed and no device asked
    // (spec.md §17.10; §5.2). The count: two far-pointer keys, two per kept
    // descriptor (gate, CS, SS), one per packed parameter group (4 words or 2
    // dwords), one per device parameter read and one per device frame write.
    // The new stack sits just below the parameters and one device region
    // covers the frame and all parameters but the `ram` highest ones. Type 4
    // with 24 words: 8 + 6 + (24 - ram) + 28; type 12 with 21 dwords:
    // 8 + 11 + (21 - ram) + 25. One more parameter in RAM makes 64, which
    // completes with every device transfer asked once.
    for model in MODELS {
        for &kind in gate_types(model) {
            let size = gate_size(kind) as u64;
            let (count, over) = if size == 4 { (21u64, 0) } else { (24, 1) };
            for ram in [over, over + 1] {
                let (mut vm, mut cpu) = gate_world(model, 0x9a, kind, K_CODE, count as u8);
                let old = 0x8000 - count * size;
                let low = old - (4 + count) * size;
                let parameter =
                    |at: u64| (0x4000 + at) & if size == 4 { 0xffff_ffff } else { 0xffff };
                for k in 0..count {
                    put(
                        &vm,
                        old + k * size,
                        size as usize,
                        parameter(old + k * size),
                    );
                }
                cpu.cpu.gpr[4] = old;
                set_tss_stack(&vm, model, 0, K_STACK, old as u32);
                trap_range(&mut vm, low, 0x8000 - ram * size - low);
                if ram == over {
                    refused(&mut cpu, &vm, UnsupportedSystemOperation::LegacyInstruction);
                    continue;
                }
                let (mut reads, mut writes) = (Vec::new(), std::collections::BTreeMap::new());
                let mut asked = 0;
                loop {
                    let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                    if run.steps() == 1 {
                        break;
                    }
                    asked += 1;
                    assert!(asked <= 2 * (4 + 2 * count), "{:?}", run.stop());
                    match run.stop() {
                        StepStop::Exit(Exit::MmioRead {
                            addr, size: bytes, ..
                        }) => {
                            assert_eq!(u64::from(*bytes), size);
                            reads.push(*addr);
                            let answer = parameter(*addr);
                            cpu.complete_mmio_read(answer);
                        }
                        StepStop::Exit(Exit::MmioWrite { addr, value, .. }) => {
                            assert!(writes.insert(*addr, *value).is_none(), "{addr:x} again");
                            cpu.complete_mmio_write();
                        }
                        stop => panic!("{stop:?}"),
                    }
                }
                assert_eq!(reads.len() as u64, count - ram);
                let mut unique = reads.clone();
                unique.sort_unstable();
                unique.dedup();
                assert_eq!(unique.len(), reads.len());
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl, cpu.cpu.gpr[4]),
                    (K_CODE, 0x5000, 0, low)
                );
                let mut expected = vec![0x1005, u64::from(U_CODE)];
                expected.extend((0..count).map(|k| parameter(old + k * size)));
                expected.extend([old, u64::from(U_DATA)]);
                let written = (0..4 + count)
                    .map(|i| writes[&(low + i * size)])
                    .collect::<Vec<_>>();
                assert_eq!(written, expected);
                assert_eq!(writes.len() as u64, 4 + count);
            }
        }
    }
}

#[test]
fn call_gates_from_ring_3_to_1_and_ring_2_to_0_use_the_target_rings_tss_stack() {
    // The new stack is SS_n:(E)SP_n of the target CS DPL, n, and CS.RPL and
    // CPL become n (SDM Vol 3A §6.8.5 steps 1-3 p. 6-18; 286 PRM p. B-25;
    // 386 PRM p. 17-40). The gates name the target with RPL 0.
    for model in MODELS {
        for &kind in gate_types(model) {
            let size = gate_size(kind);
            for (from, target, ring) in [(3u8, R1_CODE, 1u8), (2, K_CODE, 0)] {
                let (vm, mut cpu) = gate_world(model, 0x9a, kind, target & !3, 0);
                enter(&mut cpu, from);
                steps(&mut cpu, &vm, 1);
                let r = usize::from(ring);
                let sp = u64::from(TOP[r]) - 4 * size as u64;
                assert_eq!(
                    (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.legacy.cpl),
                    (target, 0x5000, ring)
                );
                assert_eq!((cpu.cpu.ss, cpu.cpu.gpr[4]), (STACK[r], sp));
                let f = usize::from(from);
                assert_eq!(
                    frame(&vm, sp, size, 4),
                    [0x1005, u64::from(CODE[f]), 0x8000, u64::from(STACK[f])]
                );
            }
        }
    }
}

#[test]
fn a_call_gate_with_three_parameters_copies_them_in_order() {
    // Count 3: the frame holds IP, CS, the three parameters in the order the
    // caller pushed them, SP and SS (SDM Vol 3A §6.8.5 step 7 p. 6-18,
    // Fig. 6-13 p. 6-19); RETF 3n returns past them.
    for model in MODELS {
        for &kind in gate_types(model) {
            let size = gate_size(kind);
            let n = 3 * size as u64;
            let (vm, mut cpu) = gate_world(model, 0x9a, kind, K_CODE, 3);
            let ret: &[u8] = if size == 4 {
                b"\x66\xca\x0c\x00"
            } else {
                b"\xca\x06\x00"
            };
            vm.write_bytes(0x5000, ret).unwrap();
            let old = 0x8000 - n;
            let values = [0x0a0a_1111u64, 0x0b0b_2222, 0x0c0c_3333].map(|v| {
                if size == 4 {
                    v
                } else {
                    v & 0xffff
                }
            });
            for (k, &v) in values.iter().enumerate() {
                put(&vm, old + (k * size) as u64, size, v);
            }
            cpu.cpu.gpr[4] = old;
            steps(&mut cpu, &vm, 1);
            let sp = 0x9000 - 7 * size as u64;
            assert_eq!((cpu.cpu.cs, cpu.cpu.gpr[4]), (K_CODE, sp));
            let mut expected = vec![0x1005, u64::from(U_CODE)];
            expected.extend(values);
            expected.extend([old, u64::from(U_DATA)]);
            assert_eq!(frame(&vm, sp, size, 7), expected);
            steps(&mut cpu, &vm, 1);
            assert_eq!(
                (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.ss, cpu.cpu.gpr[4]),
                (U_CODE, 0x1005, U_DATA, 0x8000)
            );
        }
    }
}

#[test]
fn call_gates_on_the_80386_push_at_the_new_stack_width_with_32_bit_stacks() {
    // With SS.B = 1 the parameters are read at the old ESP and the frame is
    // pushed at NewESP, both 32-bit (SDM Vol 2A CALL p. 3-132; Vol 3A
    // §3.4.5). Old ESP 10100h and ESP0 10000h, two parameters:
    // - type 12, both stacks 32-bit: ESP FFE8h; RETFD 8 gives ESP 10108h;
    // - type 4, both stacks 32-bit: the old ESP slot holds SP only, 0100h;
    //   RETF 4 gives ESP 104h, the zero-extended SP plus 4 (spec.md §17.10,
    //   ESP after a 16-bit outward return);
    // - type 12 from a 16-bit stack at SP 7FF8h: RETFD 8 gives SP 8000h.
    let model = CpuModel::I80386;
    for (kind, old_wide) in [(0x0cu8, true), (0x04, true), (0x0c, false)] {
        let size = gate_size(kind);
        let (vm, mut cpu) = gate_world(model, 0x9a, kind, K_CODE, 2);
        descriptor(&vm, 0x2010, 0, 0xfffff, 0x4092);
        let (first, second) = if size == 4 {
            (0x1111_1111, 0x2222_2222)
        } else {
            (0x1111, 0x2222)
        };
        let old = if old_wide {
            descriptor(&vm, 0x2020, 0, 0xfffff, 0x40f2);
            cpu.cpu.legacy.segments[3] = SegmentCache {
                limit: 0xfffff,
                ..cache(0x40f3)
            };
            0x10100
        } else {
            0x7ff8
        };
        cpu.cpu.gpr[4] = old;
        put(&vm, old, size, second);
        put(&vm, old + size as u64, size, first);
        set_tss_stack(&vm, model, 0, K_STACK, 0x10000);
        let ret: &[u8] = if size == 4 {
            b"\x66\xca\x08\x00"
        } else {
            b"\xca\x04\x00"
        };
        vm.write_bytes(0x5000, ret).unwrap();
        steps(&mut cpu, &vm, 1);
        let sp = 0x10000 - 6 * size as u64;
        assert_eq!((cpu.cpu.cs, cpu.cpu.gpr[4]), (K_CODE, sp));
        let saved = old & if size == 4 { 0xffff_ffff } else { 0xffff };
        assert_eq!(
            frame(&vm, sp, size, 6),
            [
                0x1005,
                u64::from(U_CODE),
                second,
                first,
                saved,
                u64::from(U_DATA)
            ]
        );
        steps(&mut cpu, &vm, 1);
        let back = match (kind, old_wide) {
            (0x0c, true) => 0x10108,
            (_, true) => 0x104,
            _ => 0x8000,
        };
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.rip, cpu.cpu.ss, cpu.cpu.gpr[4]),
            (U_CODE, 0x1005, U_DATA, back)
        );
    }
}

#[test]
fn jmp_through_a_gate_faults_in_order_and_sets_the_accessed_bit() {
    // JMP through a gate: an offset beyond the target limit #GP(0); a target
    // not present #NP(CS); a conforming target with DPL > CPL #GP(CS), here
    // from CPL 0 (286 PRM App. B JMP p. B-57; 386 PRM §17 JMP pp. 17-85-86;
    // SDM Vol 2A JMP p. 3-509). Each fault leaves the descriptors unaccessed.
    // A JMP and a same-level CALL that succeed set the target's accessed bit
    // (286 PRM §7.4 p. 7-11; SDM Vol 3A §3.4.5.1).
    for model in MODELS {
        for &kind in gate_types(model) {
            let (vm, mut cpu) = gate_world(model, 0xea, kind, 0x88, 0);
            descriptor(&vm, 0x2088, 0, 0x0fff, 0xfa);
            fault(&mut cpu, &vm, 13, Some(0));
            assert_eq!(value(&vm, 0x208d, 1), 0xfa);
            let (vm, mut cpu) = gate_world(model, 0xea, kind, 0x78, 0);
            descriptor(&vm, 0x2078, 0, 0xffff, 0x7a);
            fault(&mut cpu, &vm, 11, Some(0x78));
            let (vm, mut cpu) = gate_world(model, 0xea, kind, CONF3, 0);
            enter(&mut cpu, 0);
            fault(&mut cpu, &vm, 13, Some(u32::from(CONF3)));
            assert_eq!(value(&vm, 0x204d, 1), 0xfe);
            for op in [0xea, 0x9a] {
                let (vm, mut cpu) = gate_world(model, op, kind, U_CODE, 0);
                assert_eq!(value(&vm, 0x201d, 1), 0xfa);
                steps(&mut cpu, &vm, 1);
                assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (U_CODE, 0x5000));
                assert_eq!(value(&vm, 0x201d, 1), 0xfb);
            }
        }
    }
}
