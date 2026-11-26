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
    derive_legacy_context, CpuModel, CpuState, ExceptionDelivery, Exit, Memory, MemoryModel, Prot,
    RegionKind, SegmentClass, StepOptions, StepStop, Vcpu, Vm, VmConfig,
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

/// The guest memory `make` built, copied into a new `Memory` that replaces
/// `vm.mem` while the vCPU that ran on the old one stays live.
fn replace_memory(world: World, vm: &mut Vm) {
    let size = if world.wide() { 0x240000 } else { 0x20000 };
    let mut bytes = vec![0; size];
    vm.read_bytes(0, &mut bytes).unwrap();
    let mut memory = Memory::new(MemoryModel::Flat { size: size as u64 });
    memory.map(0, size, Prot::RWX, RegionKind::Ram).unwrap();
    memory.write_bytes(0, &bytes).unwrap();
    vm.mem = memory;
}

/// INC; a store of HLT over the INC that follows it; INC; HLT. Flat32 stores
/// through DS, Segmented through an ES that covers the code (its cache is
/// set to CS's base, as a load of a data alias would), V86 through CS.
fn store_into_next(world: World) -> Vec<u8> {
    match world {
        Flat32 { .. } => b"\x40\xc6\x05\x08\x10\x00\x00\xf4\x40\x40\xf4".to_vec(),
        Segmented { .. } => b"\x40\x26\xc6\x06\x07\x01\xf4\x40\x40\xf4".to_vec(),
        V86 => b"\x40\x2e\xc6\x06\x07\x01\xf4\x40\x40\xf4".to_vec(),
    }
}

/// The HLT at `ip` ran: it stops after itself, and in V86 (CPL 3) it raises
/// #GP(0) at itself (SDM Vol 2A HLT, Protected Mode Exceptions; Vol 3B
/// §20.2.7, the V86 sensitive instructions).
fn halted(world: World, run: &x86jit_core::StepRun, cpu: &Vcpu, ip: u64, what: &str) {
    if world == V86 {
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::Exception { vector: 13, addr, .. }) if *addr == ip),
            "{what}: {run:?}"
        );
        assert_eq!(cpu.cpu.rip, ip, "{what}");
    } else {
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::Hlt)),
            "{what}: {run:?}"
        );
        assert_eq!(cpu.cpu.rip, ip + 1, "{what}");
    }
}

#[test]
fn a_store_into_the_next_instruction_stops_the_block_before_it() {
    // SDM Vol 3A §11.6: a write to the code of the next instruction is
    // observed by that instruction. The block holds all four instructions.
    // `swapped` replaces `Vm::mem` after the vCPU was created: the walk polls
    // only the vCPU's dispatch word, which the run registers with the new
    // memory.
    for world in WORLDS {
        let code = store_into_next(world);
        for model in MODELS {
            for (budget, swapped) in [1, 2, 3, 20]
                .into_iter()
                .flat_map(|budget| [(budget, false), (budget, true)])
            {
                let ((mut avm, mut a), (mut bvm, mut b)) = pair(world, model, &code, false);
                if let Segmented { .. } = world {
                    for cpu in [&mut a, &mut b] {
                        let es = &mut cpu.cpu.legacy.segments[2];
                        (es.base, es.limit, es.attributes) = (0x1000, 0x1fff, 0x93);
                    }
                    assert_eq!(class(&a), world.class(), "{world:?}");
                }
                if swapped {
                    replace_memory(world, &mut avm);
                    replace_memory(world, &mut bvm);
                }
                let what = format!("{world:?} {model:?} budget={budget} swapped={swapped}");
                let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what}");
                same(world, &a, &b, &avm, &bvm, &what);
                if budget == 20 {
                    // INC and the store retire, then the stored HLT: the
                    // overwritten INC never runs.
                    let ip = if world.wide() { 0x1008 } else { 0x107 };
                    halted(world, &ar, &a, ip, &what);
                    assert_eq!(ar.steps(), if world == V86 { 2 } else { 3 }, "{what}");
                    assert_eq!(a.cpu.gpr[0], 0x11, "{what}");
                }
            }
        }
    }
}

#[test]
fn a_code_write_racing_with_the_walk_is_seen_at_a_later_boundary() {
    // SDM Vol 3A §11.6: another thread overwrites the first instruction of a
    // running loop (INC BX; MOV [mem],BX; INC DX; JMP back) with HLT once the
    // loop has stored BX at least once. The walk polls the dispatch word at
    // each boundary; whichever boundary sees the notice, the loop's next pass
    // runs the HLT and never the stale INC BX after it.
    for world in WORLDS {
        let (code, marker, at): (&[u8], u64, u64) = if world.wide() {
            (b"\x40\x43\x89\x1d\x00\x60\x00\x00\x42\xeb\xf6", 0x6000, 1)
        } else {
            (b"\x40\x43\x89\x1e\x10\x00\x42\xeb\xf8", 0xa010, 1)
        };
        for model in MODELS {
            let what = format!("{world:?} {model:?}");
            let ((vm, mut a), _) = pair(world, model, code, false);
            let hlt = world.code() + at;
            let stop = std::thread::scope(|scope| {
                scope.spawn(|| {
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
                    let mut stored = [0; 2];
                    while stored == [0; 2] && std::time::Instant::now() < deadline {
                        vm.read_bytes(marker, &mut stored).unwrap();
                        std::thread::yield_now();
                    }
                    vm.write_bytes(hlt, b"\xf4").unwrap();
                });
                loop {
                    let run = a.run_steps(&vm, 1 << 20, StepOptions::default()).unwrap();
                    if !matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)) {
                        break run;
                    }
                }
            });
            let ip = if world.wide() { 0x1000 } else { 0x100 };
            halted(world, &stop, &a, ip + at, &what);
            // The HLT stopped the loop at its top: BX, stored, and DX agree.
            let mut stored = [0; 2];
            vm.read_bytes(marker, &mut stored).unwrap();
            assert_eq!(u16::from_le_bytes(stored), a.cpu.gpr[3] as u16, "{what}");
            assert_eq!(
                a.cpu.gpr[2] as u16,
                (a.cpu.gpr[3] as u16).wrapping_add(3),
                "{what}"
            );
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
    // fault witness, each closed by JNZ (the counts never reach zero), by a
    // near JMP rel8 to its start or by a near JMP BP (FF E5) to it, whose CS
    // target check stays on the paged register lane (`register_lane`): an
    // instruction off that lane adds a walk of its own. After a warm-up run
    // every block is entered once and walked once, whether it runs to its
    // end or a budget ends it in its middle.
    for world in WORLDS {
        for model in MODELS {
            for (body, close) in [
                b"\x40\x43\x01\xc3\x31\xd9\x47\x46\x42".to_vec(),
                prefix(world),
            ]
            .into_iter()
            .flat_map(|body| {
                [
                    (body.clone(), b"\x75".as_slice()),
                    (body.clone(), b"\xeb"),
                    (body, b"\xff\xe5"),
                ]
            }) {
                let mut code = body;
                let instructions = if code.len() == 9 { 9 } else { 8 };
                code.extend_from_slice(close);
                if close.len() == 1 {
                    code.push((-(code.len() as i8 + 1)) as u8);
                }
                for budget in [3, 4 * (instructions + 1) + 5] {
                    let ((avm, mut a), _) = pair(world, model, &code, false);
                    // BP holds the loop's start: the code's offset in CS.
                    a.cpu.gpr[5] = a.cpu.rip;
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

/// A real-mode machine whose caches prove Direct; the reference flips the
/// accessed bit of the FS cache, which no instruction here addresses and a
/// real-mode access does not read, so it keeps Checked and nothing else
/// differs.
fn real(model: CpuModel, code: &[u8], reference: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x20000));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1100, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([0x100, 0xa00, 0xb00, 0x700, 0x800, 0x900]);
    cpu.cpu.rip = 0x100;
    cpu.cpu.gpr[4] = 0x1000;
    cpu.set_a20_enabled(true);
    if reference {
        cpu.cpu.legacy.segments[4].attributes ^= 1;
    }
    let expected = if reference {
        SegmentClass::Checked
    } else {
        SegmentClass::Direct
    };
    assert_eq!(class(&cpu), expected, "{model:?}");
    (vm, cpu)
}

/// Registers, retirement and all RAM of a real-mode run and its reference.
fn same_real(a: &Vcpu, b: &Vcpu, avm: &Vm, bvm: &Vm, what: &str) {
    let mut reference = b.cpu.clone();
    reference.legacy.segments[4].attributes ^= 1;
    assert_eq!(live(&a.cpu), live(&reference), "{what}");
    assert_eq!(a.retired_instructions(), b.retired_instructions(), "{what}");
    let mut am = vec![0; 0x20000];
    let mut bm = vec![0; 0x20000];
    avm.read_bytes(0, &mut am).unwrap();
    bvm.read_bytes(0, &mut bm).unwrap();
    assert!(am == bm, "{what}: memory");
}

/// The two runs' results, compared whole: steps and the stop, with the
/// exit's fault metadata.
fn same_stop(a: &x86jit_core::StepRun, b: &x86jit_core::StepRun, what: &str) {
    assert_eq!(format!("{a:?}"), format!("{b:?}"), "{what}");
}

#[test]
fn a_direct_rep_cut_between_iterations_resumes_and_faults_on_a_later_one() {
    // INC AX; REP MOVSW; INC BX; HLT with CX=5. A word at offset FFFFh
    // leaves its 64-KiB segment: the fourth iteration faults, through ES
    // (the destination) or DS (the source), with #GP (vector 13; 286 PRM
    // App. B MOV p. B-74; SDM Vol 3B §23.1.4 Table 23-1). REP is
    // interruptible between iterations with CX and the index registers
    // updated and IP at the instruction (SDM Vol 2B REP pp. 4-563–4-564), so each
    // budget leaves it unfinished at a different iteration, and the run
    // resumed from there ends at the same fault.
    let code = b"\x40\xf3\xa5\x43\xf4";
    for model in [CpuModel::I80286, CpuModel::I80386, CpuModel::I80486Dx] {
        for (si, di) in [(0x10u64, 0xfff9u64), (0xfff9, 0x20)] {
            for budget in 0..8 {
                let [(avm, mut a), (bvm, mut b)] = [false, true].map(|reference| {
                    let (vm, mut cpu) = real(model, code, reference);
                    cpu.cpu.gpr[1] = 5;
                    cpu.cpu.gpr[6] = si;
                    cpu.cpu.gpr[7] = di;
                    let data: Vec<u8> = (0..0x10000u32).map(|i| (i * 7 + 3) as u8).collect();
                    vm.write_bytes(0xa000, &data).unwrap();
                    (vm, cpu)
                });
                let what = format!("{model:?} SI={si:x} DI={di:x} budget={budget}");
                let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                same_stop(&ar, &br, &what);
                same_real(&a, &b, &avm, &bvm, &what);
                let ar = a.run_steps(&avm, 20, StepOptions::default()).unwrap();
                let br = b.run_steps(&bvm, 20, StepOptions::default()).unwrap();
                same_stop(&ar, &br, &format!("{what} resumed"));
                same_real(&a, &b, &avm, &bvm, &format!("{what} resumed"));
                let StepStop::Exit(Exit::Exception { vector, .. }) = ar.stop() else {
                    panic!("{what}: {ar:?}")
                };
                assert_eq!(*vector, 13, "{what}");
                // Three iterations completed; the REP is the next to run.
                assert_eq!(a.cpu.gpr[1], 2, "{what}");
                assert_eq!((a.cpu.gpr[6], a.cpu.gpr[7]), (si + 6, di + 6), "{what}");
                assert_eq!(a.cpu.rip, 0x101, "{what}");
                assert_eq!(a.cpu.gpr[0] & 0xffff, 1, "{what}");
            }
        }
    }
}

#[test]
fn a_strict_lane_step_out_resumes_the_walk() {
    // INC AX; MOV DS,[m16] (the selector it already holds); INC BX;
    // MOV CX,[m16]; INC DX; JMP back. The segment load is a system lane of
    // the paged walk; its descriptor window is proved from the selector the
    // previous pass loaded, but its selector read lies outside that window,
    // so the lane steps out, replays on the operand transaction and the walk
    // resumes at the next instruction (an extra op walk, two per step-out).
    // SDM Vol 2B MOV p. 4-35: the selector loads DS and its descriptor.
    for world in [Flat32 { paged: true }, Segmented { paged: true }] {
        for model in MODELS {
            let (code, selector_at): (&[u8], u64) = if world.wide() {
                (
                    b"\x40\x8e\x1d\x00\x60\x00\x00\x43\x8b\x0d\x02\x60\x00\x00\x42\xeb\xef",
                    0x6000,
                )
            } else {
                (
                    b"\x40\x8e\x1e\x10\x00\x43\x8b\x0e\x12\x00\x42\xeb\xf3",
                    0xa010,
                )
            };
            let build = |reference| {
                let (vm, cpu) = make(world, model, code, reference);
                vm.write_bytes(selector_at, &[0x10, 0, 0x34, 0x12]).unwrap();
                (vm, cpu)
            };
            for budget in [1, 2, 3, 6, 7, 13, 25] {
                let (avm, mut a) = build(false);
                let (bvm, mut b) = build(true);
                assert_eq!(class(&a), world.class());
                assert_eq!(class(&b), SegmentClass::Checked);
                let what = format!("{world:?} {model:?} budget={budget}");
                let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what}");
                same(world, &a, &b, &avm, &bvm, &what);
                assert_eq!(ar.steps(), budget, "{what}");
                // A warm loop: every pass after the first steps out once.
                x86jit_core::Vcpu::take_machine_counters_for_test();
                let ar = a.run_steps(&avm, 24, StepOptions::default()).unwrap();
                let [blocks, _, _, _, walks] = x86jit_core::Vcpu::take_machine_counters_for_test();
                let br = b.run_steps(&bvm, 24, StepOptions::default()).unwrap();
                assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what} warm");
                same(world, &a, &b, &avm, &bvm, &format!("{what} warm"));
                assert_eq!(a.cpu.gpr[1] & 0xffff, 0x1234, "{what}");
                assert!(walks > blocks, "{what}: no step-out ({blocks}, {walks})");
            }
        }
    }
}
