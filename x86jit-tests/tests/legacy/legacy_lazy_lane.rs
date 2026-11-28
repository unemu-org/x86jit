//! The lazy TLB-window lane of paged ordinary blocks, abandoned after its
//! instruction began: a `Fast` instruction (one RAM operand) runs in the
//! block walk and proves its operand window from the TLB at its first
//! access; an address op that faults first, or an access the TLB does not
//! prove (a miss, a clean page for a write, a window crossing into another
//! page), sends it to the operand transaction with only temporaries changed.
//! Each witness runs a block through Machine dispatch (`run_steps`) and the
//! same instructions one at a time on the singleton Checked reference entry
//! (`interpret_paged_for_test`, which never takes the ordinary walk) of a
//! fresh vCPU, and compares the exit, the registers, CR2 and every byte of
//! the page tables and data.
use x86jit_core::fault::ErrorCode;
use x86jit_core::{
    CpuModel, ExceptionDelivery, Exit, Prot, RegionKind, StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

const MODELS: [CpuModel; 2] = [CpuModel::I80386, CpuModel::I80486Dx];
const MEMORY: u64 = 0x20000;
const PD: u64 = 0x1000;
const PT: u64 = 0x2000;
const CODE: u64 = 0x4000;
const DATA: u32 = 0x8000;
const A: u32 = 0x20;
const D: u32 = 0x40;

/// How the data segments are described: base 0 and a 4 GiB limit (the
/// Flat32 class) or a byte-granular segment at `DATA` (the Segmented class).
#[derive(Clone, Copy, Debug)]
enum Data {
    Flat,
    Segmented { limit: u32 },
}

#[derive(Clone, Copy, Debug)]
struct Setup {
    model: CpuModel,
    cpl: u8,
    data: Data,
}

/// One paged machine: the first 32 pages identity-mapped (U/S, R/W, P, A
/// and D clear) through one table, `code` at `CODE`, the `pte` overrides
/// applied, and the registers set.
fn machine(setup: Setup, code: &[u8], pte: &[(u32, u32)], gpr: &[(usize, u64)]) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(MEMORY));
    // One region, so no region boundary refuses a window between pages.
    vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.set_cpu_model(setup.model);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.write_bytes(PD, &(PT as u32 | 7).to_le_bytes()).unwrap();
    for page in 0..(MEMORY >> 12) as u32 {
        let entry = pte
            .iter()
            .find(|&&(p, _)| p == page)
            .map_or((page << 12) | 7, |&(_, e)| e);
        vm.write_bytes(PT + 4 * u64::from(page), &entry.to_le_bytes())
            .unwrap();
    }
    vm.write_bytes(CODE, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = CODE;
    cpu.cpu.legacy.cr0 = 0x8000_0001;
    cpu.cpu.legacy.cr2 = 0xdead_beef;
    cpu.cpu.legacy.cr3 = PD as u32;
    cpu.cpu.legacy.cpl = setup.cpl;
    let dpl = u16::from(setup.cpl) << 5;
    for (i, segment) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        // CS, and SS, stay flat; DS and ES take the data class.
        let (base, limit) = match setup.data {
            Data::Segmented { limit } if i == 1 || i == 2 => (DATA, limit),
            _ => (0, u32::MAX),
        };
        segment.base = base;
        segment.limit = limit;
        segment.attributes = (if i == 0 { 0x9b } else { 0x93 }) | dpl | (1 << 14);
        segment.unusable = 0;
    }
    for &(reg, value) in gpr {
        cpu.cpu.gpr[reg] = value;
    }
    (vm, cpu)
}

/// Every byte the machine has: tables, code and data.
fn image(vm: &Vm) -> Vec<u8> {
    let mut bytes = vec![0; MEMORY as usize];
    for (page, bytes) in bytes.chunks_mut(0x1000).enumerate() {
        vm.read_bytes(page as u64 * 0x1000, bytes).unwrap();
    }
    bytes
}

fn word(vm: &Vm, at: u64) -> u32 {
    let mut bytes = [0; 4];
    vm.read_bytes(at, &mut bytes).unwrap();
    u32::from_le_bytes(bytes)
}

/// The block through Machine dispatch and, on a fresh machine, the same
/// instructions one at a time on the reference entry; both must stop with
/// the same exit, registers, CR2 and memory. Returns the dispatched machine
/// and its exit.
fn against_reference(
    setup: Setup,
    code: &[u8],
    pte: &[(u32, u32)],
    gpr: &[(usize, u64)],
) -> (Vm, Vcpu, Exit) {
    let (vm, mut cpu) = machine(setup, code, pte, gpr);
    let run = cpu.run_steps(&vm, 64, StepOptions::default()).unwrap();
    let StepStop::Exit(exit) = run.stop() else {
        panic!("{setup:?}: {run:?}")
    };
    let exit = exit.clone();
    let (reference_vm, mut reference) = machine(setup, code, pte, gpr);
    let mut steps = 0;
    let reference_exit = loop {
        if let Err(exit) = reference.interpret_paged_for_test(&reference_vm) {
            break exit;
        }
        steps += 1;
        assert!(steps < 64, "{setup:?}: the reference never stopped");
    };
    assert_eq!(
        format!("{exit:?}"),
        format!("{reference_exit:?}"),
        "{setup:?}: exit"
    );
    let (got, want) = (&cpu.cpu, &reference.cpu);
    assert_eq!(got.gpr, want.gpr, "{setup:?}: registers");
    assert_eq!(got.rip, want.rip, "{setup:?}: RIP");
    assert_eq!(
        format!("{:?}", got.flags),
        format!("{:?}", want.flags),
        "{setup:?}: flags"
    );
    assert_eq!(got.legacy.cr2, want.legacy.cr2, "{setup:?}: CR2");
    assert_eq!(
        format!("{:?}", got.legacy),
        format!("{:?}", want.legacy),
        "{setup:?}: system state"
    );
    assert!(
        image(&vm) == image(&reference_vm),
        "{setup:?}: memory differs from the reference"
    );
    (vm, cpu, exit)
}

fn exception(exit: &Exit) -> (u8, u32) {
    let Exit::Exception { vector, .. } = exit else {
        panic!("{exit:?}")
    };
    let code = match exit.fault_metadata().map(|m| m.error_code()) {
        Some(ErrorCode::Bits { value, .. }) => value,
        other => panic!("{exit:?}: {other:?}"),
    };
    (*vector, code)
}

const EAX: usize = 0;
const ECX: usize = 1;
const EDX: usize = 2;
const EBX: usize = 3;
const ESI: usize = 6;

/// A Segmented `MOV [m], r` whose address op exceeds the data segment's
/// limit: the address op fails before the operand's first access, so the
/// lazy instruction is abandoned and the transaction raises #GP(0) with
/// nothing written and CR2 kept (SDM Vol 3A §6.3; 80386 PRM §6.3.1.1,
/// §9.8.13 p. 9-17). With the page present, a store before it
/// leaves the page in the TLB and dirty, so the window the store would
/// prove exists; with the page absent, the #GP still comes first:
/// segmentation precedes paging (80386 PRM §§6.4.3–6.5 pp. 6-21–22;
/// Intel486 PRM §6.9 p. 6-25).
#[test]
fn a_segment_limit_fault_abandons_the_lazy_store() {
    for model in MODELS {
        for cpl in [0, 3] {
            for absent in [false, true] {
                let setup = Setup {
                    model,
                    cpl,
                    data: Data::Segmented { limit: 0x7ff },
                };
                // [mov [esi],ecx;] inc ecx; mov [ebx],eax; inc edx; ud2. The
                // dword at 7FEh straddles the limit, inside one page.
                let code: &[u8] = if absent {
                    &[0x41, 0x89, 0x03, 0x42, 0x0f, 0x0b]
                } else {
                    &[0x89, 0x0e, 0x41, 0x89, 0x03, 0x42, 0x0f, 0x0b]
                };
                let pte: &[(u32, u32)] = if absent { &[(8, 0)] } else { &[] };
                let gpr = [(EAX, 0x1122_3344), (ECX, 0x10), (EBX, 0x7fe), (ESI, 0)];
                let (vm, cpu, exit) = against_reference(setup, code, pte, &gpr);
                assert_eq!(exception(&exit), (13, 0), "{setup:?}");
                let at = if absent { 1 } else { 3 };
                assert_eq!(cpu.cpu.rip, CODE + at, "{setup:?}");
                assert_eq!(cpu.cpu.gpr[ECX], 0x11, "{setup:?}");
                assert_eq!(cpu.cpu.gpr[EDX], 0, "{setup:?}");
                assert_eq!(cpu.cpu.legacy.cr2, 0xdead_beef, "{setup:?}");
                assert_eq!(word(&vm, u64::from(DATA) + 0x7fc), 0, "{setup:?}");
                assert_eq!(word(&vm, u64::from(DATA) + 0x800), 0, "{setup:?}");
                let pte = if absent { 0 } else { DATA | 7 | A | D };
                assert_eq!(word(&vm, PT + 4 * 8), pte, "{setup:?}");
                if !absent {
                    assert_eq!(word(&vm, u64::from(DATA)), 0x10, "{setup:?}");
                }
            }
        }
    }
}

/// A `MOV [m]` whose page the TLB does not hold: the operand access finds
/// no window, the instruction is abandoned, and the transaction walks,
/// setting A (and D for a write) as the reference does (SDM Vol 3A §5.8;
/// 80386 PRM §5.2.4.3 p. 5-10), or faults with #PF naming the linear
/// address in CR2 and the access in the error code (SDM Vol 3A §7.15,
/// Event 14; 80386 PRM §9.8.14 pp. 9-17–19). The second store of the
/// block hits a page the TLB holds clean, which a write window does not
/// admit either.
#[test]
fn a_tlb_miss_abandons_the_lazy_access() {
    for model in MODELS {
        for cpl in [0, 3] {
            for data in [Data::Flat, Data::Segmented { limit: 0xffff }] {
                let base = match data {
                    Data::Flat => DATA,
                    Data::Segmented { .. } => 0,
                };
                let setup = Setup { model, cpl, data };
                let at = u64::from(base + 0x10);
                // mov eax,[ebx]; mov [ebx],ecx; mov [ebx+4],eax; inc edx; ud2
                let code = [0x8b, 0x03, 0x89, 0x0b, 0x89, 0x43, 0x04, 0x42, 0x0f, 0x0b];
                let gpr = [(ECX, 0x5566_7788), (EBX, at)];
                let (vm, cpu, exit) = against_reference(setup, &code, &[], &gpr);
                assert!(
                    matches!(exit, Exit::Exception { vector: 6, .. }),
                    "{setup:?}: {exit:?}"
                );
                assert_eq!(cpu.cpu.gpr[EDX], 1);
                assert_eq!(word(&vm, u64::from(DATA) + 0x10), 0x5566_7788);
                assert_eq!(word(&vm, PT + 4 * 8), DATA | 7 | A | D, "{setup:?}");
                // An absent page for a read and a write, and at CPL 3 a
                // read-only user page for a write and a supervisor page for a
                // read: (PTE, write, error code without U/S).
                for (entry, write, error) in [
                    (0, false, 0),
                    (0, true, 2),
                    (DATA | 5, true, 3),
                    (DATA | 3, false, 1),
                ] {
                    if entry != 0 && cpl != 3 {
                        continue;
                    }
                    // inc ecx; mov [ebx],ecx | mov eax,[ebx]; inc edx; ud2
                    let code: &[u8] = if write {
                        &[0x41, 0x89, 0x0b, 0x42, 0x0f, 0x0b]
                    } else {
                        &[0x41, 0x8b, 0x03, 0x42, 0x0f, 0x0b]
                    };
                    let (vm, cpu, exit) = against_reference(setup, code, &[(8, entry)], &gpr);
                    let error = error | if cpl == 3 { 4 } else { 0 };
                    assert_eq!(exception(&exit), (14, error), "{setup:?} {entry:#x}");
                    assert_eq!(cpu.cpu.legacy.cr2, DATA + 0x10, "{setup:?}");
                    assert_eq!(cpu.cpu.rip, CODE + 1, "{setup:?}");
                    assert_eq!(cpu.cpu.gpr[ECX], 0x5566_7789, "{setup:?}");
                    assert_eq!(word(&vm, u64::from(DATA) + 0x10), 0, "{setup:?}");
                }
            }
        }
    }
}

/// An instruction whose operand crosses into a second page that faults:
/// the first page is in the TLB, present and dirty, from the store before
/// it, so only the second page refuses; nothing of either part is written
/// (paging design §3.4; the #PF names the second page's first byte, SDM
/// Vol 3A §7.15, Event 14; 80386 PRM §9.8.14 pp. 9-17–19). A store and
/// a read-modify-write, both lazy `Fast` shapes.
#[test]
fn a_second_page_fault_commits_nothing_of_a_lazy_operand() {
    for model in MODELS {
        for cpl in [0, 3] {
            for data in [Data::Flat, Data::Segmented { limit: 0xffff }] {
                for rmw in [false, true] {
                    for next in [0, DATA + 0x1001] {
                        if next != 0 && cpl != 3 {
                            continue;
                        }
                        let base = match data {
                            Data::Flat => DATA,
                            Data::Segmented { .. } => 0,
                        };
                        let setup = Setup { model, cpl, data };
                        // mov [esi],ecx; mov|add [ebx],eax; inc edx; ud2
                        let code = [
                            0x89,
                            0x0e,
                            if rmw { 0x01 } else { 0x89 },
                            0x03,
                            0x42,
                            0x0f,
                            0x0b,
                        ];
                        let gpr = [
                            (EAX, 0x1122_3344),
                            (ECX, 0x0bad_f00d),
                            (EBX, u64::from(base + 0xffe)),
                            (ESI, u64::from(base + 0x800)),
                        ];
                        let (vm, cpu, exit) = against_reference(setup, &code, &[(9, next)], &gpr);
                        let error = 2 | u32::from(next != 0) | if cpl == 3 { 4 } else { 0 };
                        assert_eq!(exception(&exit), (14, error), "{setup:?} rmw={rmw}");
                        assert_eq!(cpu.cpu.legacy.cr2, DATA + 0x1000, "{setup:?}");
                        assert_eq!(cpu.cpu.rip, CODE + 2, "{setup:?}");
                        assert_eq!(word(&vm, u64::from(DATA) + 0x800), 0x0bad_f00d);
                        assert_eq!(word(&vm, u64::from(DATA) + 0xffc), 0, "{setup:?}");
                        assert_eq!(word(&vm, u64::from(DATA) + 0x1000), 0, "{setup:?}");
                        assert_eq!(word(&vm, PT + 4 * 8), DATA | 7 | A | D, "{setup:?}");
                    }
                }
            }
        }
    }
}
