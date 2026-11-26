//! One pass of the op walk per protected and paged block (xj-to9.35): each
//! instruction commits at the next instruction's marker, so a fault, an exit,
//! a budget cut, a single step or a code write in the middle of a block leaves
//! exactly the state of the Checked reference, which walks one instruction per
//! block. The dispatch counters show one op walk per block.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::descriptor;
use x86jit_core::{
    derive_legacy_context, CpuModel, CpuState, ExceptionDelivery, Exit, Prot, RegionKind,
    SegmentClass, StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

const MODELS: [CpuModel; 2] = [CpuModel::I80386, CpuModel::I80486Dx];
const PT: u64 = 0x1f000;

/// The walk under test and its reference. Flat32 and Segmented compare with
/// Checked over the same bytes (a cache that proves neither class: a short ES
/// limit for Flat32, a non-present GS cache, which is checked only on a load,
/// SDM Vol 3A §3.4.5). Paged V86 compares with unpaged V86 over identity pages.
#[derive(Clone, Copy, Debug, PartialEq)]
enum World {
    Flat32 { paged: bool },
    Segmented { paged: bool },
    V86,
}
use World::*;

const WORLDS: [World; 5] = [
    Flat32 { paged: false },
    Flat32 { paged: true },
    Segmented { paged: false },
    Segmented { paged: true },
    V86,
];

impl World {
    fn paged(self) -> bool {
        matches!(
            self,
            Flat32 { paged: true } | Segmented { paged: true } | V86
        )
    }
    fn wide(self) -> bool {
        matches!(self, Flat32 { .. })
    }
    /// Where the code starts in physical memory.
    fn code(self) -> u64 {
        if self.wide() {
            0x1000
        } else {
            0x1100
        }
    }
    fn class(self) -> SegmentClass {
        if self.wide() {
            SegmentClass::Flat32
        } else {
            SegmentClass::Segmented
        }
    }
}

fn identity_pages(vm: &Vm) {
    // 80386 PRM §5.2 pp. 5-7–11: one page table, identity frames, user pages.
    vm.write_bytes(0x1e000, &0x1f007u32.to_le_bytes()).unwrap();
    for page in 0..32u32 {
        vm.write_bytes(PT + u64::from(page * 4), &((page * 4096) | 7).to_le_bytes())
            .unwrap();
    }
}

/// `reference` builds the Checked (or unpaged V86) twin of the same machine.
fn make(world: World, model: CpuModel, code: &[u8], reference: bool) -> (Vm, Vcpu) {
    match world {
        Flat32 { paged } => {
            let (vm, mut cpu) = fixture::setup_delivery(model, code, ExceptionDelivery::Outward);
            for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
                cache.base = 0;
                cache.limit = u32::MAX;
                cache.attributes = if i == 0 { 0x409b } else { 0x4093 };
            }
            cpu.cpu.cs = 8;
            cpu.cpu.ds = 16;
            cpu.cpu.es = 16;
            cpu.cpu.ss = 16;
            descriptor(&vm, 0x2008, 0, u32::MAX, 0xc09b);
            descriptor(&vm, 0x2010, 0, u32::MAX, 0xc093);
            if reference {
                cpu.cpu.legacy.segments[2].limit -= 1;
                cpu.cpu.legacy.segments[5].attributes &= !0x80;
            }
            if paged {
                identity_pages(&vm);
                cpu.cpu.legacy.cr3 = 0x1e000;
                cpu.cpu.legacy.cr0 |= 1 << 31;
            }
            cpu.set_a20_enabled(true);
            (vm, cpu)
        }
        Segmented { paged } => {
            let mut vm = Vm::new(VmConfig::flat(0x20000));
            vm.set_cpu_model(model);
            vm.set_exception_delivery(ExceptionDelivery::Outward);
            vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1100, code).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_a20_enabled(true);
            cpu.cpu.rip = 0x100;
            cpu.cpu.gpr[4] = 0x1000;
            cpu.cpu.legacy.cr0 = if paged { 0x8000_0001 } else { 1 };
            cpu.cpu.legacy.cr3 = 0x1e000;
            cpu.cpu.legacy.gdtr.base = 0x3000;
            cpu.cpu.legacy.gdtr.limit = 0x7f;
            cpu.cpu.cs = 8;
            cpu.cpu.ds = 16;
            cpu.cpu.es = 16;
            cpu.cpu.ss = 24;
            cpu.cpu.fs = 16;
            cpu.cpu.gs = 16;
            // CS base 1000h, data at A000h with limit FFh, SS at 7000h.
            for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
                cache.base = [0x1000, 0xa000, 0xa000, 0x7000, 0xa000, 0xa000][i];
                cache.limit = if i == 0 || i == 3 { 0x1fff } else { 0xff };
                cache.attributes = if i == 0 { 0x9b } else { 0x93 };
                cache.unusable = 0;
            }
            descriptor(&vm, 0x3008, 0x1000, 0x1fff, 0x9b);
            descriptor(&vm, 0x3010, 0xa000, 0xff, 0x93);
            descriptor(&vm, 0x3018, 0x7000, 0x1fff, 0x93);
            if paged {
                identity_pages(&vm);
            }
            if reference {
                cpu.cpu.legacy.segments[5].attributes &= !0x80;
            }
            (vm, cpu)
        }
        V86 => {
            let mut vm = Vm::new(VmConfig::flat(0x20000));
            vm.set_cpu_model(model);
            vm.set_exception_delivery(ExceptionDelivery::Outward);
            vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.write_bytes(0x1100, code).unwrap();
            identity_pages(&vm);
            let mut cpu = vm.new_vcpu();
            cpu.cpu = CpuState::legacy_real_entry([0x100, 0xa00, 0xa00, 0x700, 0, 0]);
            cpu.cpu.rip = 0x100;
            cpu.cpu.gpr[4] = 0x1000;
            cpu.cpu.legacy.cr0 = if reference { 1 } else { 0x8000_0001 };
            cpu.cpu.legacy.cr3 = 0x1e000;
            cpu.cpu.legacy.cpl = 3;
            cpu.cpu.rflags_system = 0x23000; // VM=1, IOPL=3 (SDM Vol 3B §23.2.1).
            cpu.set_a20_enabled(true);
            (vm, cpu)
        }
    }
}

fn class(cpu: &Vcpu) -> SegmentClass {
    derive_legacy_context(cpu.legacy_cpu_model().unwrap(), &cpu.cpu)
        .unwrap()
        .segment_class
}

/// The registers without the retry ledger's dead entries.
fn live(cpu: &CpuState) -> String {
    let mut cpu = cpu.clone();
    cpu.mmio_parts[usize::from(cpu.mmio_parts_len)..].fill((0, 0));
    format!("{cpu:?}")
}

/// Full state against the reference: registers, retirement and all RAM, the
/// page tables' accessed and dirty bits included where the reference pages
/// (the unpaged V86 reference does not); device pages are skipped.
fn same(world: World, a: &Vcpu, b: &Vcpu, avm: &Vm, bvm: &Vm, what: &str) {
    let mut reference = b.cpu.clone();
    match world {
        Flat32 { .. } => {
            reference.legacy.segments[2].limit = a.cpu.legacy.segments[2].limit;
            reference.legacy.segments[5].attributes = a.cpu.legacy.segments[5].attributes;
        }
        Segmented { .. } => reference.legacy.segments[5].attributes |= 0x80,
        V86 => reference.legacy.cr0 |= 0x8000_0000,
    }
    assert_eq!(live(&a.cpu), live(&reference), "{what}");
    assert_eq!(a.retired_instructions(), b.retired_instructions(), "{what}");
    let pages = if world == V86 { 0x1e } else { 0x20 };
    for page in 0..pages {
        let mut am = vec![0; 0x1000];
        let mut bm = vec![0; 0x1000];
        let ar = avm.read_bytes(page * 0x1000, &mut am);
        let br = bvm.read_bytes(page * 0x1000, &mut bm);
        assert_eq!(ar.is_ok(), br.is_ok(), "{what}: page {page:#x}");
        assert!(am == bm, "{what}: memory page {page:#x}");
    }
}

/// INC, ADD, XOR, a store and a load of the same word, PUSH and POP, INC:
/// register, fast-operand and stack instructions in one block.
fn prefix(world: World) -> Vec<u8> {
    let mut code = b"\x40\x01\xc3\x31\xd9".to_vec();
    if world.wide() {
        code.extend(b"\xa3\x00\x60\x00\x00\x8b\x15\x00\x60\x00\x00");
    } else {
        code.extend(b"\xa3\x10\x00\x8b\x16\x10\x00");
    }
    code.extend(b"\x50\x5e\x47");
    code
}

/// The mid-block instructions that end the block: none, HLT, #UD, #DE, a
/// data-limit #GP(0), a stack-limit #SS(0), a not-present #PF (paged worlds
/// with a paged reference only), and an IN that exits to the embedder.
fn endings(world: World) -> Vec<(&'static str, Vec<u8>)> {
    let mut out = vec![
        ("none", vec![]),
        ("hlt", b"\xf4".to_vec()),
        ("ud", b"\x0f\x0b".to_vec()),
        ("de", b"\x31\xc9\xf7\xf1".to_vec()),
    ];
    match world {
        Flat32 { paged } => {
            out.push(("gp", b"\xa1\xff\xff\xff\xff".to_vec()));
            out.push(("ss", b"\x36\xa1\xff\xff\xff\xff".to_vec()));
            if paged {
                out.push(("pf", b"\xa1\x00\x90\x00\x00".to_vec()));
            }
            out.push(("in", b"\xe4\x60".to_vec()));
        }
        Segmented { paged } => {
            out.push(("gp", b"\xa1\x00\x01".to_vec()));
            out.push(("ss", b"\x36\xa1\xff\x1f".to_vec()));
            if paged {
                out.push(("pf", b"\x36\x8b\x0e\x00\x18".to_vec()));
            }
            out.push(("in", b"\xe4\x60".to_vec()));
        }
        V86 => {
            out.push(("gp", b"\x8b\x0e\xff\xff".to_vec()));
            out.push(("ss", b"\x36\x8b\x0e\xff\xff".to_vec()));
        }
    }
    out
}

/// The not-present page of the `pf` ending: 9000h for Flat32, SS:1800h
/// (8800h) for Segmented.
fn unmap_fault_page(world: World, vm: &Vm) {
    let page = if world.wide() { 9 } else { 8 };
    vm.write_bytes(PT + page * 4, &0u32.to_le_bytes()).unwrap();
}

fn program(world: World, ordinal: usize, ending: &[u8]) -> Vec<u8> {
    let mut code = prefix(world);
    code.extend(vec![0x40; ordinal]);
    code.extend_from_slice(ending);
    // INC BX; INC DX; HLT after the ending.
    code.extend(b"\x43\x42\xf4");
    code
}

fn pair(world: World, model: CpuModel, code: &[u8], tf: bool) -> ((Vm, Vcpu), (Vm, Vcpu)) {
    let mut out = [false, true].map(|reference| {
        let (vm, mut cpu) = make(world, model, code, reference);
        cpu.cpu.gpr[0] = 0x10;
        cpu.cpu.gpr[2] = 0x3;
        cpu.cpu.gpr[7] = 0x20;
        if world.paged() {
            unmap_fault_page(world, &vm);
        }
        if tf {
            cpu.cpu.rflags_system |= 0x100;
        }
        Some((vm, cpu))
    });
    let a = out[0].take().unwrap();
    let b = out[1].take().unwrap();
    if world != V86 {
        assert_eq!(class(&a.1), world.class(), "{world:?}");
        assert_eq!(class(&b.1), SegmentClass::Checked, "{world:?} reference");
    }
    (a, b)
}

#[test]
fn mid_block_faults_exits_budgets_and_single_steps_equal_the_reference() {
    // SDM Vol 3A §6.5 (faults restart at the faulting instruction with no
    // partial effect), §6.15 #GP/#SS/#PF; Vol 3B §17.3.1.4 (TF traps after
    // each instruction). Every budget cuts the block at a different
    // instruction, and each ending sits at two positions in the block.
    for world in WORLDS {
        for model in MODELS {
            for (name, ending) in endings(world) {
                for ordinal in [0, 2] {
                    let code = program(world, ordinal, &ending);
                    for tf in [false, true] {
                        for budget in 0..16 {
                            let ((avm, mut a), (bvm, mut b)) = pair(world, model, &code, tf);
                            let what = format!(
                                "{world:?} {model:?} {name} at {ordinal} TF={tf} budget={budget}"
                            );
                            let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                            let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                            assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what}");
                            same(world, &a, &b, &avm, &bvm, &what);
                            if budget == 15 && !tf {
                                let StepStop::Exit(exit) = ar.stop() else {
                                    panic!("{what}: {ar:?}")
                                };
                                let vector = match exit {
                                    Exit::Exception { vector, .. } => Some(*vector),
                                    _ => None,
                                };
                                let expected = match name {
                                    // HLT at CPL 3 is #GP(0) (SDM Vol 2A HLT).
                                    "none" | "hlt" if world == V86 => Some(13),
                                    "ud" => Some(6),
                                    "de" => Some(0),
                                    "gp" => Some(13),
                                    "ss" => Some(12),
                                    "pf" => Some(14),
                                    _ => None,
                                };
                                assert_eq!(vector, expected, "{what}: {exit:?}");
                                if expected.is_some() && !matches!(name, "none" | "hlt") {
                                    // The faulting instruction is the next to run:
                                    // the prefix, the INCs and, for #DE, the XOR.
                                    let retired = 8 + ordinal as u64 + u64::from(name == "de");
                                    assert_eq!(a.retired_instructions(), retired, "{what}");
                                    let ip =
                                        world.code() + code.len() as u64 - 3 - ending.len() as u64
                                            + if name == "de" { 2 } else { 0 };
                                    let linear = if world.wide() {
                                        a.cpu.rip
                                    } else {
                                        a.cpu.rip + 0x1000
                                    };
                                    assert_eq!(linear, ip, "{what}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn a_store_into_the_next_instruction_stops_the_block_before_it() {
    // SDM Vol 3A §11.6: a write to the code of the next instruction is
    // observed by that instruction. INC; MOV BYTE [1008h], F4h (HLT over the
    // INC that follows); INC; HLT. The block holds all four instructions.
    let code = b"\x40\xc6\x05\x08\x10\x00\x00\xf4\x40\x40\xf4";
    for world in [Flat32 { paged: false }, Flat32 { paged: true }] {
        for model in MODELS {
            for budget in [1, 2, 3, 20] {
                let ((avm, mut a), (bvm, mut b)) = pair(world, model, code, false);
                let what = format!("{world:?} {model:?} budget={budget}");
                let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what}");
                same(world, &a, &b, &avm, &bvm, &what);
                if budget == 20 {
                    // INC and the store retire, then the stored HLT: the
                    // overwritten INC never runs.
                    assert!(matches!(ar.stop(), StepStop::Exit(Exit::Hlt)), "{what}");
                    assert_eq!(ar.steps(), 3, "{what}");
                    assert_eq!(a.cpu.gpr[0], 0x11, "{what}");
                }
            }
        }
    }
}

#[test]
fn a_mid_block_device_exit_replays_only_its_instruction() {
    // A read and then a write of a device word in the middle of one block:
    // each exits before its instruction retires, the completion replays that
    // instruction alone, and the block's successors then run in the walk.
    for world in [
        Flat32 { paged: false },
        Flat32 { paged: true },
        Segmented { paged: false },
        Segmented { paged: true },
    ] {
        for model in MODELS {
            // INC; INC; MOV AX,[dev]; MOV [dev],AX; INC; INC BX; HLT.
            let code: &[u8] = if world.wide() {
                b"\x40\x40\xa1\x00\x60\x00\x00\xa3\x00\x60\x00\x00\x40\x43\xf4"
            } else {
                b"\x40\x40\xa1\x10\x00\xa3\x10\x00\x40\x43\xf4"
            };
            let device: u64 = if world.wide() { 0x6000 } else { 0xa000 };
            let total: usize = if world.wide() { 0x240000 } else { 0x20000 };
            let build = |reference| {
                let (mut vm, mut cpu) = make(world, model, code, reference);
                vm.unmap(0, total).unwrap();
                vm.map(0, device as usize, Prot::RWX, RegionKind::Ram)
                    .unwrap();
                vm.map(device, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                vm.map(
                    device + 0x1000,
                    total - device as usize - 0x1000,
                    Prot::RWX,
                    RegionKind::Ram,
                )
                .unwrap();
                // The remap dropped the image: write it again.
                rebuild(world, &vm, code);
                cpu.cpu.gpr[0] = 0x10;
                (vm, cpu)
            };
            let (avm, mut a) = build(false);
            let (bvm, mut b) = build(true);
            for stage in 0..3 {
                let what = format!("{world:?} {model:?} stage {stage}");
                let ar = a.run_steps(&avm, 20, StepOptions::default()).unwrap();
                let br = b.run_steps(&bvm, 20, StepOptions::default()).unwrap();
                assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what}");
                same(world, &a, &b, &avm, &bvm, &what);
                match stage {
                    0 => {
                        assert_eq!(ar.steps(), 2, "{what}");
                        assert!(matches!(ar.stop(), StepStop::Exit(Exit::MmioRead { .. })));
                        assert_eq!(a.retired_instructions(), 2, "{what}");
                        a.complete_mmio_read(41);
                        b.complete_mmio_read(41);
                    }
                    1 => {
                        assert_eq!(ar.steps(), 1, "{what}");
                        assert!(
                            matches!(ar.stop(), StepStop::Exit(Exit::MmioWrite { value: 41, .. })),
                            "{what}: {ar:?}"
                        );
                        assert_eq!(a.retired_instructions(), 3, "{what}");
                        a.complete_mmio_write();
                        b.complete_mmio_write();
                    }
                    _ => {
                        assert_eq!(ar.steps(), 4, "{what}");
                        assert!(matches!(ar.stop(), StepStop::Exit(Exit::Hlt)));
                        assert_eq!(a.retired_instructions(), 7, "{what}");
                        assert_eq!(a.cpu.gpr[0] & 0xffff, 42, "{what}");
                    }
                }
            }
        }
    }
}

/// The code, descriptors and page tables of `make` after a remap.
fn rebuild(world: World, vm: &Vm, code: &[u8]) {
    vm.write_bytes(world.code(), code).unwrap();
    match world {
        Flat32 { .. } => {
            descriptor(vm, 0x2008, 0, u32::MAX, 0xc09b);
            descriptor(vm, 0x2010, 0, u32::MAX, 0xc093);
        }
        _ => {
            descriptor(vm, 0x3008, 0x1000, 0x1fff, 0x9b);
            descriptor(vm, 0x3010, 0xa000, 0xff, 0x93);
            descriptor(vm, 0x3018, 0x7000, 0x1fff, 0x93);
        }
    }
    if world.paged() {
        identity_pages(vm);
    }
}

#[test]
fn a_warm_block_is_one_op_walk_and_a_budget_cut_keeps_it_one() {
    // A loop of register instructions, then one of the mixed prefix of the
    // fault witness, each closed by JNZ (the counts never reach zero) or by a
    // near JMP to its start, whose CS target check stays on the paged
    // register lane: after a warm-up run every block is entered once and
    // walked once, whether it runs to its end or a budget ends it in its middle.
    for world in WORLDS {
        for model in MODELS {
            for (body, close) in [
                b"\x40\x43\x01\xc3\x31\xd9\x47\x46\x42".to_vec(),
                prefix(world),
            ]
            .into_iter()
            .flat_map(|body| [(body.clone(), 0x75u8), (body, 0xeb)])
            {
                let mut code = body;
                let instructions = if code.len() == 9 { 9 } else { 8 };
                code.push(close);
                code.push((-(code.len() as i8 + 1)) as u8);
                for budget in [3, 4 * (instructions + 1) + 5] {
                    let ((avm, mut a), _) = pair(world, model, &code, false);
                    let what = format!("{world:?} {model:?} {code:02x?} budget={budget}");
                    let initial = a.cpu.clone();
                    a.run_steps(&avm, 40, StepOptions::default()).unwrap();
                    a.cpu = initial;
                    x86jit_core::Vcpu::take_machine_counters_for_test();
                    let run = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                    let [blocks, _, lifts, _, walks] =
                        x86jit_core::Vcpu::take_machine_counters_for_test();
                    assert_eq!(run.steps(), budget, "{what}: {run:?}");
                    assert_eq!(lifts, 0, "{what}: warm");
                    assert!(
                        blocks <= budget.div_ceil(instructions + 1) + 1,
                        "{what}: {blocks}"
                    );
                    assert_eq!(walks, blocks, "{what}: one op walk per block");
                }
            }
        }
    }
}
