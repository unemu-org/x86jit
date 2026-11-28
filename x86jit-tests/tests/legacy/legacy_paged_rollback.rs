//! Paged instruction rollback at every operand page (paging design §3.4,
//! spec.md §16). The transaction lane runs an instruction twice, a preview
//! that plans its accesses and the commit; both run on the instruction's own
//! state and roll it back from fixed images on a fault or a device handoff.
//! 80386 PRM §9.8.14 pp. 9-17–19 and SDM Vol 3A §§7.5–7.6: a page fault
//! restarts the instruction with its architectural state as it was, CR2
//! apart. Every case faults or hands off at each page it touches, in a
//! position that puts the page boundary inside its access sequence, and is
//! checked against the complete `CpuState` and RAM of an unfaulted run.
use x86jit_core::legacy::{SegmentCache, SystemSegment};
use x86jit_core::{
    CpuModel, ExceptionDelivery, Exit, FpuModel, Prot, RegionKind, StepOptions, StepStop, Vcpu, Vm,
    VmConfig,
};

const MEMORY: u64 = 0x10000;
const PD: u32 = 0x1000;
const PT: u32 = 0x2000;
const GDT: u32 = 0x3000;
const CODE: u32 = 0x4000;
const IDT: u32 = 0x5000;
// Every page an operand can reach: the stack pages either side of 0x9000,
// two pages either side of each string pointer, and the tables.
const PAGES: [u32; 8] = [0x8000, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, GDT, IDT];
const TARGET: u32 = 0x4100;
const HANDLER: u32 = 0x4200;
// The far targets' code segment: a second flat code descriptor with another
// limit, so that a transfer changes the selector and the cache.
const FAR_CS: u32 = 0x18;
// Hardware tasks (80386 PRM §7.1 Fig. 7-1 pp. 7-2–3, §7.2 Fig. 7-2 p. 7-4):
// the running task's busy TSS in the GDT page, an available one in the IDT
// page (past the gates the cases use), and a not-present data descriptor.
const NEW_TSS_SELECTOR: u32 = 0x20;
const OLD_TSS_SELECTOR: u16 = 0x28;
const ABSENT_DATA: u32 = 0x30;
const OLD_TSS: u32 = GDT + 0x100;
const NEW_TSS: u32 = IDT + 0x100;

fn pattern(address: u64) -> u8 {
    (address.wrapping_mul(13).wrapping_add(7)) as u8
}

/// Register positions: inside one page, then with the stack, the string
/// pointers and the frame pointer a few bytes from a page end, so that the
/// boundary falls between an instruction's accesses or inside one.
const POSITIONS: [[u32; 4]; 3] = [
    // ESP, ESI, EDI, EBP
    [0x8f80, 0xa100, 0xc100, 0x8fa0],
    [0x9008, 0xaffe, 0xcffd, 0x9004],
    [0x8ff8, 0xafc0, 0xcfc0, 0x8ff8],
];

struct Case {
    name: &'static str,
    code: &'static [u8],
    /// Memory the instruction needs to complete, written at offsets from
    /// ESP (`stack`) or ESI (`source`).
    stack: &'static [u32],
    source: &'static [u32],
    /// An external interrupt for vector 30h is pending before the step, so
    /// that the step delivers it (the paged event transaction).
    irq: bool,
}

const CASES: &[Case] = &[
    case("push r", &[0x50], &[], &[]),
    case("pop r", &[0x5b], &[], &[]),
    case("push m", &[0xff, 0x36], &[], &[]),
    case("pop m", &[0x8f, 0x07], &[], &[]),
    case("call rel", &[0xe8, 0x00, 0x01, 0x00, 0x00], &[], &[]),
    case("ret", &[0xc3], &[TARGET], &[]),
    case("ret imm", &[0xc2, 0x08, 0x00], &[TARGET], &[]),
    case("pushad", &[0x60], &[], &[]),
    case("popad", &[0x61], &[], &[]),
    case("pushfd", &[0x9c], &[], &[]),
    case("popfd", &[0x9d], &[0x246], &[]),
    case("enter", &[0xc8, 0x10, 0x00, 0x02], &[], &[]),
    case("leave", &[0xc9], &[], &[]),
    case("movsd", &[0xa5], &[], &[]),
    case("rep movsb", &[0xf3, 0xa4], &[], &[]),
    case("stosd", &[0xab], &[], &[]),
    case("lodsb", &[0xac], &[], &[]),
    case("cmpsd", &[0xa7], &[], &[]),
    case("rep scasb", &[0xf3, 0xae], &[], &[]),
    case("xchg m", &[0x87, 0x06], &[], &[]),
    case("add m", &[0x01, 0x07], &[], &[]),
    case("inc m", &[0xff, 0x06], &[], &[]),
    case("lock xadd", &[0xf0, 0x0f, 0xc1, 0x07], &[], &[]),
    case("cmpxchg", &[0x0f, 0xb1, 0x0f], &[], &[]),
    case("fs load", &[0x64, 0x8b, 0x06], &[], &[]),
    case("gs store", &[0x65, 0x89, 0x07], &[], &[]),
    case("mov fs m", &[0x8e, 0x26], &[], &[0x10]),
    case("lgs", &[0x0f, 0xb5, 0x06], &[], &[0x1234_5678, 0x10]),
    case("bound", &[0x62, 0x06], &[], &[0, 0x7fff_ffff]),
    case("call far m", &[0xff, 0x1e], &[], &[TARGET, FAR_CS]),
    case("retf", &[0xcb], &[TARGET, FAR_CS], &[]),
    case("iretd", &[0xcf], &[TARGET, FAR_CS, 0x202], &[]),
    case("int", &[0xcd, 0x30], &[], &[]),
    case("jmp tss", &[0xea, 0, 0, 0, 0, 0x20, 0], &[], &[]),
    Case {
        name: "external interrupt",
        code: &[0x90],
        stack: &[],
        source: &[],
        irq: true,
    },
    case("fld m64", &[0xdd, 0x06], &[], &[]),
    case("fstp m64", &[0xdd, 0x1f], &[], &[]),
    case("fild m32", &[0xdb, 0x06], &[], &[]),
    case("fistp m32", &[0xdb, 0x1f], &[], &[]),
    case("fnstenv", &[0xd9, 0x37], &[], &[]),
    case("fldenv", &[0xd9, 0x26], &[], &[0x037f, 0, 0xffff]),
    case("fnsave", &[0xdd, 0x37], &[], &[]),
    case("frstor", &[0xdd, 0x26], &[], &[0x037f, 0, 0xffff]),
];

const fn case(
    name: &'static str,
    code: &'static [u8],
    stack: &'static [u32],
    source: &'static [u32],
) -> Case {
    Case {
        name,
        code,
        stack,
        source,
        irq: false,
    }
}

fn descriptor(base: u32, limit: u32, access: u8, flags: u8) -> u64 {
    u64::from(limit & 0xffff)
        | (u64::from(base & 0xff_ffff) << 16)
        | (u64::from(access) << 40)
        | (u64::from((limit >> 16) & 0xf) << 48)
        | (u64::from(flags) << 52)
        | (u64::from(base >> 24) << 56)
}

/// A ring-0 Flat32 machine with PG=1, identity-mapped below `MEMORY`, the
/// case's code at `CODE`, and `device` (if any) a Trap region.
fn machine(model: CpuModel, case: &Case, position: [u32; 4], device: Option<u32>) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(MEMORY));
    for page in (0..MEMORY).step_by(0x1000) {
        let kind = if device == Some(page as u32) {
            RegionKind::Trap
        } else {
            RegionKind::Ram
        };
        vm.map(page, 0x1000, Prot::RWX, kind).unwrap();
    }
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    let mut bytes: Vec<u8> = (0..MEMORY).map(pattern).collect();
    bytes[PD as usize..PD as usize + 4].copy_from_slice(&(PT | 7).to_le_bytes());
    for page in 0..(MEMORY as u32 >> 12) {
        let at = (PT + 4 * page) as usize;
        bytes[at..at + 4].copy_from_slice(&((page << 12) | 7).to_le_bytes());
    }
    write_ram(&vm, 0, &bytes, device);
    let put = |at: u32, value: u32| write_ram(&vm, u64::from(at), &value.to_le_bytes(), device);
    for (i, d) in [
        0,
        descriptor(0, 0xfffff, 0x9b, 0xc),
        descriptor(0, 0xfffff, 0x93, 0xc),
        descriptor(0, 0xffff0, 0x9b, 0xc),
        descriptor(NEW_TSS, 0x67, 0x89, 0),
        descriptor(OLD_TSS, 0x67, 0x8b, 0),
        descriptor(0, 0xfffff, 0x13, 0xc),
    ]
    .into_iter()
    .enumerate()
    {
        write_ram(&vm, u64::from(GDT) + 8 * i as u64, &d.to_le_bytes(), device);
    }
    // An interrupt gate (type 14) for vector 0x30 to FAR_CS:HANDLER (80386 PRM
    // §9.5 Fig. 9-3 p. 9-7).
    let gate = u64::from(HANDLER & 0xffff)
        | (u64::from(FAR_CS) << 16)
        | (0x8e << 40)
        | (u64::from(HANDLER >> 16) << 48);
    write_ram(&vm, u64::from(IDT) + 8 * 0x30, &gate.to_le_bytes(), device);
    // The incoming task: CR3, EIP, EFLAGS, ESP, then ES, CS, SS, DS, FS, GS
    // and a null LDT selector, and the I/O map base past the limit.
    for (offset, value) in [
        (0x1c, PD),
        (0x20, TARGET),
        (0x24, 0x202),
        (0x38, 0x8f00),
        (0x48, 0x10),
        (0x4c, 0x08),
        (0x50, 0x10),
        (0x54, 0x10),
        (0x58, 0x10),
        (0x5c, 0x10),
        (0x60, 0),
        (0x64, 0x68 << 16),
    ] {
        put(NEW_TSS + offset, value);
    }
    let mut code = case.code.to_vec();
    code.push(0xf4);
    write_ram(&vm, u64::from(CODE), &code, device);
    write_ram(&vm, u64::from(HANDLER), &[0x90, 0xf4], device);
    let [esp, esi, edi, ebp] = position;
    for (i, &value) in case.stack.iter().enumerate() {
        put(esp + 4 * i as u32, value);
    }
    for (i, &value) in case.source.iter().enumerate() {
        put(esi + 4 * i as u32, value);
    }
    let mut cpu = vm.new_vcpu();
    cpu.set_fpu_model(FpuModel::Present);
    let state = &mut cpu.cpu;
    state.rip = u64::from(CODE);
    state.legacy.cr0 = 0x8000_0011;
    state.legacy.cr2 = 0xdead_beef;
    state.legacy.cr3 = PD;
    state.legacy.cpl = 0;
    state.legacy.gdtr.base = GDT;
    state.legacy.gdtr.limit = 0x37;
    state.legacy.tr = SystemSegment {
        selector: OLD_TSS_SELECTOR,
        cache: SegmentCache {
            base: OLD_TSS,
            limit: 0x67,
            attributes: 0x8b,
            ..Default::default()
        },
        ..Default::default()
    };
    state.legacy.idtr.base = IDT;
    state.legacy.idtr.limit = 0x7ff;
    for (i, segment) in state.legacy.segments.iter_mut().enumerate() {
        segment.base = 0;
        segment.limit = u32::MAX;
        segment.attributes = if i == 0 { 0x9b } else { 0x93 } | (1 << 14) | (1 << 15);
        segment.unusable = 0;
    }
    (state.cs, state.ds, state.es, state.ss, state.fs, state.gs) =
        (0x08, 0x10, 0x10, 0x10, 0x10, 0x10);
    state.gpr = [
        0x1122_3344,
        2,
        0x5566_7788,
        0x99aa_bbcc,
        u64::from(esp),
        u64::from(ebp),
        u64::from(esi),
        u64::from(edi),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    // IF set, so that an interrupt gate's clear of it shows.
    state.flags.if_ = true;
    state.xmm[3] = 0x0123_4567_89ab_cdef_0011_2233_4455_6677;
    state.kmask[1] = 0x55;
    if case.irq {
        cpu.inject_irq(0x30);
    }
    (vm, cpu)
}

/// Writes that skip a device page, which holds no RAM.
fn write_ram(vm: &Vm, at: u64, bytes: &[u8], device: Option<u32>) {
    let mut address = at;
    let mut rest = bytes;
    while !rest.is_empty() {
        let page_end = (address | 0xfff) + 1;
        let len = rest.len().min((page_end - address) as usize);
        if device != Some((address & !0xfff) as u32) {
            vm.write_bytes(address, &rest[..len]).unwrap();
        }
        address += len as u64;
        rest = &rest[len..];
    }
}

fn step(vm: &Vm, cpu: &mut Vcpu) -> Exit {
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    match run.stop() {
        StepStop::Exit(exit) => exit.clone(),
        other => panic!("unexpected paged stop: {other:?}"),
    }
}

/// The complete state, the transfer ledger's live entries only: every
/// reader takes `mmio_parts[..mmio_parts_len]`, and a cleared ledger keeps
/// its old entries past the length (`CpuState::clear_mmio_parts`).
fn image(cpu: &Vcpu) -> String {
    let mut state = cpu.cpu.clone();
    state.mmio_parts[usize::from(state.mmio_parts_len)..].fill((0, 0));
    format!("{state:?}")
}

/// RAM outside the page tables, whose A and D bits record walks that the
/// rollback keeps (paging design §3.4), and outside the device page.
fn ram(vm: &Vm, device: Option<u32>) -> Vec<u8> {
    let mut bytes = vec![0; MEMORY as usize];
    for (page, chunk) in bytes.chunks_mut(0x1000).enumerate() {
        let page = page as u32 * 0x1000;
        if !(PD..=PT).contains(&page) && device != Some(page) {
            vm.read_bytes(u64::from(page), chunk).unwrap();
        }
    }
    bytes
}

fn set_present(vm: &Vm, page: u32, present: bool) {
    let entry = page | if present { 7 } else { 6 };
    vm.write_bytes(u64::from(PT + 4 * (page >> 12)), &entry.to_le_bytes())
        .unwrap();
}

/// The page faults and the device handoffs over every case, position and
/// page, pinned by the tests so that a case that stops reaching a page shows.
fn rollback_at_every_page(model: CpuModel) -> (usize, usize) {
    let (mut faults, mut handoffs) = (0, 0);
    for case in CASES {
        // Intel486 PRM §26 XADD p. 26-282, CMPXCHG p. 26-62: new in the 486.
        if model == CpuModel::I80386 && matches!(case.name, "lock xadd" | "cmpxchg") {
            continue;
        }
        for position in POSITIONS {
            let what = |page: u32, how: &str| {
                format!("{model:?} {} {position:x?} page {page:#x} {how}", case.name)
            };
            let (initial, _) = machine(model, case, position, None);
            let (rvm, mut reference) = machine(model, case, position, None);
            let exit = step(&rvm, &mut reference);
            assert!(
                matches!(exit, Exit::BudgetExhausted),
                "{}: {exit:?}",
                what(0, "reference")
            );
            reference.cpu.legacy.cr2 = 0;
            let expected = image(&reference);
            let expected_ram = ram(&rvm, None);
            for page in PAGES {
                // A fault at this page, then the same instruction again.
                let (vm, mut cpu) = machine(model, case, position, None);
                let before = image(&cpu);
                let before_ram = ram(&vm, None);
                set_present(&vm, page, false);
                match step(&vm, &mut cpu) {
                    Exit::BudgetExhausted => {}
                    exit @ Exit::Exception { vector: 14, .. } => {
                        faults += 1;
                        let linear = cpu.cpu.legacy.cr2;
                        assert_eq!(linear & !0xfff, page, "{}: {exit:?}", what(page, "CR2"));
                        cpu.cpu.legacy.cr2 = 0xdead_beef;
                        assert_eq!(image(&cpu), before, "{}", what(page, "rollback"));
                        assert_eq!(ram(&vm, None), before_ram, "{}", what(page, "rollback RAM"));
                        set_present(&vm, page, true);
                        // A delivery that faults leaves the request to the
                        // interrupt controller, which raises it again.
                        if case.irq && !cpu.has_pending_irq() {
                            cpu.inject_irq(0x30);
                        }
                        let exit = step(&vm, &mut cpu);
                        assert!(
                            matches!(exit, Exit::BudgetExhausted),
                            "{}: {exit:?}",
                            what(page, "restart")
                        );
                    }
                    exit => panic!("{}: {exit:?}", what(page, "fault")),
                }
                cpu.cpu.legacy.cr2 = 0;
                assert_eq!(image(&cpu), expected, "{}", what(page, "after fault"));
                assert_eq!(
                    ram(&vm, None),
                    expected_ram,
                    "{}",
                    what(page, "RAM after fault")
                );

                // The page as a device that answers as the reference RAM.
                if page == GDT || page == IDT {
                    // Descriptor tables in a device are typed refusals
                    // (`physical_table_rom_and_device_are_typed_refusals`).
                    continue;
                }
                let (vm, mut cpu) = machine(model, case, position, Some(page));
                let before = image(&cpu);
                let mut device = vec![0; 0x1000];
                initial.read_bytes(u64::from(page), &mut device).unwrap();
                let mut exits = 0;
                let mut refused = false;
                loop {
                    match step(&vm, &mut cpu) {
                        Exit::BudgetExhausted => break,
                        // An operand the Machine does not take from a device
                        // is refused before any effect.
                        Exit::UnsupportedSystemContext { .. } if exits == 0 => {
                            assert_eq!(image(&cpu), before, "{}", what(page, "refusal"));
                            refused = true;
                            break;
                        }
                        Exit::MmioRead { addr, size, .. } => {
                            let at = (addr - u64::from(page)) as usize;
                            let mut value = [0; 8];
                            value[..usize::from(size)]
                                .copy_from_slice(&device[at..at + usize::from(size)]);
                            cpu.complete_mmio_read(u64::from_le_bytes(value));
                        }
                        Exit::MmioWrite {
                            addr, size, value, ..
                        } => {
                            let at = (addr - u64::from(page)) as usize;
                            device[at..at + usize::from(size)]
                                .copy_from_slice(&value.to_le_bytes()[..usize::from(size)]);
                            cpu.complete_mmio_write();
                        }
                        exit => panic!("{}: {exit:?}", what(page, "device")),
                    }
                    exits += 1;
                    assert!(exits < 256, "{}", what(page, "device retry"));
                }
                if refused {
                    continue;
                }
                handoffs += usize::from(exits != 0);
                cpu.cpu.legacy.cr2 = 0;
                assert_eq!(image(&cpu), expected, "{}", what(page, "after device"));
                let mut after = ram(&vm, Some(page));
                after[page as usize..page as usize + 0x1000].copy_from_slice(&device);
                assert_eq!(after, expected_ram, "{}", what(page, "RAM after device"));
            }
        }
    }
    (faults, handoffs)
}

#[test]
fn every_operand_page_fault_and_device_handoff_restarts_exactly_386() {
    assert_eq!(rollback_at_every_page(CpuModel::I80386), (202, 145));
}

#[test]
fn every_operand_page_fault_and_device_handoff_restarts_exactly_486() {
    assert_eq!(rollback_at_every_page(CpuModel::I80486Dx), (210, 153));
}

/// A task switch whose incoming DS is not present faults after its commit
/// point, in the new task (80386 PRM §7.5 pp. 7-8–9): the transaction
/// lane's preview ends in that fault with the switch committed, and the
/// commit runs the switch again. The paged result, complete state and RAM,
/// equals the unpaged interpreter's.
#[test]
fn a_task_switch_faulting_after_its_commit_equals_the_unpaged_switch() {
    let case = case("jmp tss", &[0xea, 0, 0, 0, 0, 0x20, 0], &[], &[]);
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        let run = |paged: bool| {
            let (vm, mut cpu) = machine(model, &case, POSITIONS[0], None);
            vm.write_bytes(u64::from(NEW_TSS + 0x54), &ABSENT_DATA.to_le_bytes())
                .unwrap();
            if !paged {
                cpu.cpu.legacy.cr0 &= !(1 << 31);
            }
            let exit = step(&vm, &mut cpu);
            assert!(
                matches!(exit, Exit::Exception { vector: 11, .. }),
                "{model:?} paged={paged}: {exit:?}"
            );
            assert_eq!(cpu.cpu.legacy.tr.selector, NEW_TSS_SELECTOR as u16);
            cpu.cpu.legacy.cr0 &= !(1 << 31);
            (image(&cpu), ram(&vm, None))
        };
        let (paged, paged_ram) = run(true);
        let (unpaged, unpaged_ram) = run(false);
        assert_eq!(paged, unpaged, "{model:?}");
        assert_eq!(paged_ram, unpaged_ram, "{model:?}");
    }
}
