//! Data accesses covering adjacent mapped RAM and ROM regions.
//!
//! SDM Vol 1 §4.1.1: an unaligned operand can need more than one bus cycle, so a
//! single operand can cover bytes of two adjacent regions. When every byte lies in
//! a RAM or ROM region and the regions leave no gap, the engine performs the access:
//! loads return the bytes, stores commit the RAM bytes and drop the ROM bytes as a
//! store wholly inside a ROM region does. The rule holds per transfer: a transfer
//! touching a gap or a Trap region keeps the single-region rule (the same exit as
//! before), and so does every LOCK operand, including each A20 bus piece of one.
//! A 16-byte store checks both of its transfers before either is written.
//! The order of the per-region pieces is engine policy, not a hardware claim.
//! SDM Vol 3A §11.1.1: an aligned word/doubleword/quadword access is atomic.
use iced_x86::code_asm::*;
use x86jit_core::{
    AccessKind, Backend, CpuMode, CpuModel, CpuState, Exit, InterpreterBackend, Prot, Reg,
    RegionCaps, RegionKind, StepOptions, StepStop, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;
const LEFT: u64 = 0x3000;
const EDGE: u64 = 0x4000;
const RIGHT_END: u64 = 0x8000;
const CAPS: RegionCaps = RegionCaps {
    max_blocks: 16,
    max_icount: 256,
};
const KINDS: [(RegionKind, RegionKind); 3] = [
    (RegionKind::Ram, RegionKind::Ram),
    (RegionKind::Ram, RegionKind::Rom),
    (RegionKind::Rom, RegionKind::Ram),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tier {
    Interp,
    Block,
    Region,
}
const TIERS: [Tier; 3] = [Tier::Interp, Tier::Block, Tier::Region];

fn backend(tier: Tier) -> Box<dyn Backend> {
    match tier {
        Tier::Interp => Box::new(InterpreterBackend),
        Tier::Block => Box::new(JitBackend::new()),
        Tier::Region => Box::new(JitBackend::with_superblocks(CAPS)),
    }
}

/// Byte-wise, so a span covering two regions needs no multi-region host API.
fn poke(vm: &Vm, at: u64, bytes: &[u8]) {
    for (i, &b) in bytes.iter().enumerate() {
        vm.write_bytes(at + i as u64, &[b]).unwrap();
    }
}
fn peek(vm: &Vm, at: u64, len: usize) -> Vec<u8> {
    (0..len as u64)
        .map(|i| {
            let mut b = [0];
            vm.read_bytes(at + i, &mut b).unwrap();
            b[0]
        })
        .collect()
}
fn pattern(width: usize) -> Vec<u8> {
    (0..width).map(|i| 0x31 + i as u8).collect()
}
fn lanes(bytes: &[u8]) -> [u128; 4] {
    std::array::from_fn(|i| {
        let mut lane = [0; 16];
        for (j, b) in lane.iter_mut().enumerate() {
            *b = bytes.get(i * 16 + j).copied().unwrap_or(0);
        }
        u128::from_le_bytes(lane)
    })
}

/// One data move between RDI's operand and RAX/RBX or XMM0/XMM1 (YMM for 32).
fn moves(width: usize, bits: u32, store: bool) -> Vec<u8> {
    let mut a = CodeAssembler::new(bits).unwrap();
    match (width, store) {
        (2, true) => a.mov(word_ptr(edi), ax),
        (2, false) => a.mov(bx, word_ptr(edi)),
        (4, true) => a.mov(dword_ptr(edi), eax),
        (4, false) => a.mov(ebx, dword_ptr(edi)),
        (8, true) if bits == 64 => a.mov(qword_ptr(rdi), rax),
        (8, true) => a.movq(qword_ptr(edi), xmm0),
        (8, false) if bits == 64 => a.mov(rbx, qword_ptr(rdi)),
        (8, false) => a.movq(xmm1, qword_ptr(edi)),
        (16, true) => a.movdqu(xmmword_ptr(edi), xmm0),
        (16, false) => a.movdqu(xmm1, xmmword_ptr(edi)),
        (32, true) => a.vmovdqu(ymmword_ptr(edi), ymm0),
        (32, false) => a.vmovdqu(ymm1, ymmword_ptr(edi)),
        _ => unreachable!(),
    }
    .unwrap();
    a.hlt().unwrap();
    a.assemble(CODE).unwrap()
}

/// `window` maps Trap regions on both sides of the pair: the compiled tiers'
/// bounding Trap window then covers the pair, so compiled code hands each of
/// these accesses to the interpreter (§5.2) instead of accessing the host bytes.
fn app_vm(
    tier: Tier,
    mode: CpuMode,
    (left, right): (RegionKind, RegionKind),
    window: bool,
    routed: bool,
    code: &[u8],
) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(tier));
    vm.set_cpu_mode(mode);
    vm.set_tier_up_after(None);
    vm.mem.set_host_fault_recovery(routed);
    vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(LEFT, (EDGE - LEFT) as usize, Prot::RW, left)
        .unwrap();
    vm.map(EDGE, (RIGHT_END - EDGE) as usize, Prot::RW, right)
        .unwrap();
    if window {
        vm.map(0x2000, 0x10, Prot::RW, RegionKind::Trap).unwrap();
        vm.map(0x9000, 0x10, Prot::RW, RegionKind::Trap).unwrap();
    }
    vm.write_bytes(CODE, code).unwrap();
    if tier == Tier::Region {
        vm.test_support_force_region(CODE, CAPS, 0x1f80).unwrap();
    }
    vm
}

fn assert_compiled(vm: &Vm, tier: Tier) {
    match tier {
        Tier::Interp => {}
        Tier::Block => assert!(vm.cache.memory_stats().compiled_entries > 0),
        Tier::Region => assert!(vm.cache.regions() > 0),
    }
}

/// Inline compiled stores bound against the backing size and do not consult the
/// region kind, so they write ROM bytes (the documented `RegionKind::Rom` limit,
/// unchanged for a contained ROM store); every interpreter store drops them.
fn drops_rom(tier: Tier, window: bool) -> bool {
    tier == Tier::Interp || window
}

fn split_offsets(width: usize) -> Vec<u64> {
    if width <= 8 {
        (1..width as u64).collect()
    } else {
        let w = width as u64;
        vec![1, 4, 8, w / 2 + 1, w - 1]
    }
}

/// Collects failing cases so one run reports every failing configuration.
#[derive(Default)]
struct Failures(std::collections::BTreeMap<String, (usize, String)>);
impl Failures {
    fn check(&mut self, ok: bool, key: impl FnOnce() -> String, detail: impl FnOnce() -> String) {
        if !ok {
            let entry = self.0.entry(key()).or_insert_with(|| (0, detail()));
            entry.0 += 1;
        }
    }
    fn finish(self, name: &str) {
        for (key, (count, first)) in &self.0 {
            eprintln!("FAIL {name} {key}: {count} case(s); first: {first}");
        }
        assert!(
            self.0.is_empty(),
            "{name}: {} failing configuration(s)",
            self.0.len()
        );
    }
}

/// Expected destination bytes after a store of `bytes` over 0xcc-filled memory.
fn stored(at: u64, bytes: &[u8], kinds: (RegionKind, RegionKind), drop_rom: bool) -> Vec<u8> {
    (0..bytes.len())
        .map(|i| {
            let kind = if at + (i as u64) < EDGE {
                kinds.0
            } else {
                kinds.1
            };
            if kind == RegionKind::Rom && drop_rom {
                0xcc
            } else {
                bytes[i]
            }
        })
        .collect()
}

fn application_pairs(width: usize) {
    let mut failures = Failures::default();
    for kinds in KINDS {
        for tier in TIERS {
            for mode in [CpuMode::Long64, CpuMode::Compat32] {
                let bits = if mode == CpuMode::Long64 { 64 } else { 32 };
                for window in [false, true] {
                    for routed in [false, true] {
                        for store in [false, true] {
                            let key = format!(
                                "{kinds:?} {tier:?} {mode:?} window={window} routed={routed} \
                                 store={store}"
                            );
                            let code = moves(width, bits, store);
                            let vm = app_vm(tier, mode, kinds, window, routed, &code);
                            let mut cpu = vm.new_vcpu();
                            for split in split_offsets(width) {
                                let at = EDGE - split;
                                let bytes = pattern(width);
                                let fill = if store {
                                    vec![0xcc; width]
                                } else {
                                    bytes.clone()
                                };
                                poke(&vm, at, &fill);
                                let mut low = [0; 8];
                                low[..width.min(8)].copy_from_slice(&bytes[..width.min(8)]);
                                cpu.cpu = CpuState::new();
                                cpu.set_reg(Reg::Rip, CODE);
                                cpu.set_reg(Reg::Rdi, at);
                                cpu.set_reg(Reg::Rax, u64::from_le_bytes(low));
                                cpu.cpu.set_vec(0, lanes(&bytes), 32);
                                let exit = cpu.run(&vm, None);
                                let actual = if !matches!(exit, Exit::Hlt) {
                                    None
                                } else if store {
                                    Some(peek(&vm, at, width))
                                } else if width <= 8 && (width != 8 || bits == 64) {
                                    Some(cpu.reg(Reg::Rbx).to_le_bytes()[..width].to_vec())
                                } else {
                                    let received = cpu.cpu.vec_lanes(1);
                                    Some(
                                        received
                                            .iter()
                                            .flat_map(|l| l.to_le_bytes())
                                            .take(width)
                                            .collect(),
                                    )
                                };
                                let expected = if store {
                                    stored(at, &bytes, kinds, drops_rom(tier, window))
                                } else {
                                    bytes.clone()
                                };
                                failures.check(
                                    actual.as_ref() == Some(&expected),
                                    || key.clone(),
                                    || format!("split={split}: {exit:?} {actual:x?}"),
                                );
                            }
                            assert_compiled(&vm, tier);
                        }
                    }
                }
            }
        }
    }
    failures.finish(&format!("width={width}"));
}

#[test]
fn application_word_pairs() {
    application_pairs(2);
}
#[test]
fn application_dword_pairs() {
    application_pairs(4);
}
#[test]
fn application_qword_pairs() {
    application_pairs(8);
}
#[test]
fn application_xmm_pairs() {
    application_pairs(16);
}
#[test]
fn application_ymm_pairs() {
    application_pairs(32);
}

/// FLD/FSTP m80 take the interpreter's wide checked-transfer path; compiled code
/// accesses the host bytes inline, as for the scalar and vector moves above.
#[test]
fn x87_tbyte_pairs() {
    let image = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0xff, 0x3f]; // 1.5
    let mut failures = Failures::default();
    for kinds in KINDS {
        for tier in TIERS {
            for routed in [false, true] {
                for store in [false, true] {
                    let key = format!("{kinds:?} {tier:?} routed={routed} store={store}");
                    let mut a = CodeAssembler::new(64).unwrap();
                    if store {
                        a.fld(tword_ptr(rsi)).unwrap();
                        a.fstp(tword_ptr(rdi)).unwrap();
                    } else {
                        a.fld(tword_ptr(rdi)).unwrap();
                        a.fstp(tword_ptr(rsi)).unwrap();
                    }
                    a.hlt().unwrap();
                    let code = a.assemble(CODE).unwrap();
                    let vm = app_vm(tier, CpuMode::Long64, kinds, false, routed, &code);
                    let mut cpu = vm.new_vcpu();
                    for split in 1..10 {
                        let at = EDGE - split;
                        let contained = 0x1800;
                        poke(&vm, at, &if store { [0xcc; 10] } else { image });
                        poke(&vm, contained, &if store { image } else { [0xcc; 10] });
                        cpu.cpu = CpuState::new();
                        cpu.set_reg(Reg::Rip, CODE);
                        cpu.set_reg(Reg::Rdi, at);
                        cpu.set_reg(Reg::Rsi, contained);
                        let exit = cpu.run(&vm, None);
                        let actual = matches!(exit, Exit::Hlt)
                            .then(|| peek(&vm, if store { at } else { contained }, 10));
                        let expected = if store {
                            stored(at, &image, kinds, drops_rom(tier, false))
                        } else {
                            image.to_vec()
                        };
                        failures.check(
                            actual.as_ref() == Some(&expected),
                            || key.clone(),
                            || format!("split={split}: {exit:?} {actual:x?}"),
                        );
                    }
                    assert_compiled(&vm, tier);
                }
            }
        }
    }
    failures.finish("x87");
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Legacy {
    Alias,
    I286,
    I386,
}
const LEGACY: [Legacy; 3] = [Legacy::Alias, Legacy::I286, Legacy::I386];

fn legacy_vm(target: Legacy, size: u64) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(size), Box::new(InterpreterBackend));
    vm.set_cpu_mode(CpuMode::Real16);
    match target {
        Legacy::Alias => {}
        Legacy::I286 => vm.set_cpu_model(CpuModel::I80286),
        Legacy::I386 => vm.set_cpu_model(CpuModel::I80386),
    }
    vm
}
fn legacy_cpu(vm: &Vm, target: Legacy) -> Vcpu {
    let mut cpu = vm.new_vcpu();
    if target != Legacy::Alias {
        cpu.cpu = CpuState::legacy_real_entry([0; 6]);
    }
    cpu
}
fn legacy_run(vm: &Vm, cpu: &mut Vcpu, budget: u64) -> Exit {
    let result = cpu.run_steps(vm, budget, StepOptions::default()).unwrap();
    match result.stop() {
        StepStop::Exit(exit) => exit.clone(),
        other => panic!("{other:?}"),
    }
}

/// The embedder's repro: a 64 KiB frame mapped as independently remappable 16 KiB
/// RAM windows. MOV word [3FFFh] with DS=D000h covers D3FFFh and D4000h.
#[test]
fn real16_frame_word_across_windows() {
    for target in LEGACY {
        for windows in [false, true] {
            for routed in [false, true] {
                let mut vm = legacy_vm(target, 0x10_0000);
                vm.mem.set_host_fault_recovery(routed);
                vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
                if windows {
                    vm.map(0xd0000, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
                    vm.map(0xd4000, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
                } else {
                    vm.map(0xd0000, 0x8000, Prot::RWX, RegionKind::Ram).unwrap();
                }
                // mov ax,0D000h; mov ds,ax; mov word [3FFFh],1234h; mov ax,[3FFFh]; hlt
                vm.write_bytes(
                    0x7c00,
                    &[
                        0xb8, 0x00, 0xd0, 0x8e, 0xd8, 0xc7, 0x06, 0xff, 0x3f, 0x34, 0x12, 0xa1,
                        0xff, 0x3f, 0xf4,
                    ],
                )
                .unwrap();
                let mut cpu = legacy_cpu(&vm, target);
                cpu.set_reg(Reg::Rip, 0x7c00);
                let exit = legacy_run(&vm, &mut cpu, 20);
                let what = format!("{target:?} windows={windows} routed={routed}");
                assert!(matches!(exit, Exit::Hlt), "{what}: {exit:?}");
                assert_eq!(cpu.reg(Reg::Rax) & 0xffff, 0x1234, "{what}");
                assert_eq!(peek(&vm, 0xd3fff, 2), [0x34, 0x12], "{what}");
            }
        }
    }
}

/// Fetch already continues across adjacent RAM regions; this keeps a witness.
/// JMP FAR D000:3FFF enters `MOV AX,1234h` whose opcode is the last byte of the
/// first window; its immediate and the following HLT come from the second window.
#[test]
fn real16_frame_fetch_across_windows() {
    for target in LEGACY {
        let mut vm = legacy_vm(target, 0x10_0000);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0xd0000, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(0xd4000, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x7c00, &[0xea, 0xff, 0x3f, 0x00, 0xd0])
            .unwrap();
        vm.write_bytes(0xd3fff, &[0xb8]).unwrap();
        vm.write_bytes(0xd4000, &[0x34, 0x12, 0xf4]).unwrap();
        let mut cpu = legacy_cpu(&vm, target);
        cpu.set_reg(Reg::Rip, 0x7c00);
        let exit = legacy_run(&vm, &mut cpu, 20);
        assert!(matches!(exit, Exit::Hlt), "{target:?}: {exit:?}");
        assert_eq!(cpu.reg(Reg::Rax) & 0xffff, 0x1234, "{target:?}");
    }
}

/// Real-mode word and doubleword moves at every split position of adjacent
/// RAM/ROM pairs on the alias and the Machine models.
#[test]
fn real16_pairs() {
    let mut failures = Failures::default();
    for target in LEGACY {
        for kinds in KINDS {
            for width in [2u8, 4] {
                if width == 4 && target == Legacy::I286 {
                    continue;
                }
                for routed in [false, true] {
                    for store in [false, true] {
                        let key = format!(
                            "{target:?} {kinds:?} width={width} routed={routed} store={store}"
                        );
                        let mut vm = legacy_vm(target, 0x10000);
                        vm.mem.set_host_fault_recovery(routed);
                        vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
                        vm.map(LEFT, (EDGE - LEFT) as usize, Prot::RW, kinds.0)
                            .unwrap();
                        vm.map(EDGE, (RIGHT_END - EDGE) as usize, Prot::RW, kinds.1)
                            .unwrap();
                        let mut code = Vec::new();
                        if width == 4 {
                            code.push(0x66);
                        }
                        // mov [di],ax / mov bx,[di]
                        code.extend_from_slice(if store { &[0x89, 0x05] } else { &[0x8b, 0x1d] });
                        code.push(0xf4);
                        vm.write_bytes(CODE, &code).unwrap();
                        for split in 1..u64::from(width) {
                            let at = EDGE - split;
                            let width = usize::from(width);
                            let bytes = pattern(width);
                            let fill = if store {
                                vec![0xcc; width]
                            } else {
                                bytes.clone()
                            };
                            poke(&vm, at, &fill);
                            let mut cpu = legacy_cpu(&vm, target);
                            cpu.set_reg(Reg::Rip, CODE);
                            cpu.set_reg(Reg::Rdi, at);
                            cpu.set_reg(Reg::Rbx, 0);
                            let mut value = [0; 8];
                            value[..width].copy_from_slice(&bytes);
                            cpu.set_reg(Reg::Rax, u64::from_le_bytes(value));
                            let exit = legacy_run(&vm, &mut cpu, 8);
                            let actual = if !matches!(exit, Exit::Hlt) {
                                None
                            } else if store {
                                Some(peek(&vm, at, width))
                            } else {
                                Some(cpu.reg(Reg::Rbx).to_le_bytes()[..width].to_vec())
                            };
                            let expected = if store {
                                stored(at, &bytes, kinds, true)
                            } else {
                                bytes.clone()
                            };
                            failures.check(
                                actual.as_ref() == Some(&expected),
                                || key.clone(),
                                || format!("split={split}: {exit:?} {actual:x?}"),
                            );
                        }
                    }
                }
            }
        }
    }
    failures.finish("real16");
}

/// A span with a gap byte or a Trap byte keeps the containment rule. The expected
/// exits were measured on the revision before split support: the same exit kind,
/// address and access, and no byte written. Compiled code over a Vec backing has
/// no gap check and accesses the bytes natively; that result is kept too.
#[test]
fn gap_and_trap_spans_are_unchanged() {
    for tier in TIERS {
        for trap in [false, true] {
            for width in [2usize, 8, 16] {
                for store in [false, true] {
                    let what = format!("{tier:?} trap={trap} width={width} store={store}");
                    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(tier));
                    vm.set_tier_up_after(None);
                    vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
                    vm.map(LEFT, (EDGE - LEFT) as usize, Prot::RW, RegionKind::Ram)
                        .unwrap();
                    if trap {
                        vm.map(EDGE, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                    } else {
                        vm.map(EDGE + 1, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
                    }
                    vm.write_bytes(CODE, &moves(width, 64, store)).unwrap();
                    if tier == Tier::Region {
                        vm.test_support_force_region(CODE, CAPS, 0x1f80).unwrap();
                    }
                    vm.write_bytes(EDGE - 1, &[0x55]).unwrap();
                    let mut cpu = vm.new_vcpu();
                    cpu.set_reg(Reg::Rip, CODE);
                    cpu.set_reg(Reg::Rdi, EDGE - 1);
                    cpu.set_reg(Reg::Rax, u64::MAX);
                    cpu.cpu.set_vec(0, [u128::MAX; 4], 32);
                    let exit = cpu.run(&vm, None);
                    println!("{what}: {exit:?}");
                    let access = if store {
                        AccessKind::Write
                    } else {
                        AccessKind::Read
                    };
                    if tier != Tier::Interp && !trap {
                        assert!(matches!(exit, Exit::Hlt), "{what}: {exit:?}");
                        let byte = if store { 0xff } else { 0x55 };
                        assert_eq!(vm.mem.read(EDGE - 1, 1), Ok(byte), "{what}");
                        continue;
                    }
                    // A Trap span names the operand base. A gap span names the
                    // transfer base or its first unmapped byte, as each path did.
                    let expected = match (trap, width, store) {
                        (true, _, _) => EDGE - 1,
                        (false, 2 | 8, false) | (false, 16, true) => EDGE - 1,
                        (false, _, _) => EDGE,
                    };
                    assert!(
                        matches!(exit, Exit::UnmappedMemory { addr, access: a, .. }
                            if addr == expected && a == access),
                        "{what}: {exit:?}"
                    );
                    assert_eq!(vm.mem.read(EDGE - 1, 1), Ok(0x55), "{what}");
                }
            }
        }
    }
}

/// LOCK operands keep the single-region rule even across adjacent RAM regions.
#[test]
fn locked_operand_across_ram_regions_is_unchanged() {
    for tier in [Tier::Interp, Tier::Block] {
        let mut a = CodeAssembler::new(64).unwrap();
        a.lock().add(qword_ptr(rdi), 1).unwrap();
        a.hlt().unwrap();
        let code = a.assemble(CODE).unwrap();
        let vm = app_vm(
            tier,
            CpuMode::Long64,
            (RegionKind::Ram, RegionKind::Ram),
            true,
            false,
            &code,
        );
        poke(&vm, EDGE - 4, &[0; 8]);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rdi, EDGE - 4);
        let exit = cpu.run(&vm, None);
        println!("{tier:?}: {exit:?}");
        assert!(
            matches!(exit, Exit::UnmappedMemory { addr, access: AccessKind::Write, .. }
                if addr == EDGE - 4),
            "{tier:?}: {exit:?}"
        );
        assert_eq!(peek(&vm, EDGE - 4, 8), [0; 8], "{tier:?}");
    }
}

/// SDM Vol 3A §11.1.1: an aligned quadword access is atomic. A quadword whose two
/// halves lie in adjacent RAM regions is still one host access, so a reader never
/// observes a mix of two stores. Both vCPUs run guest code on separate threads.
#[test]
fn aligned_qword_across_ram_regions_never_tears() {
    use std::sync::{Arc, Barrier};
    const FLAG: u64 = 0x1800;
    const WORD: u64 = 0x4000;
    for routed in [false, true] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(InterpreterBackend));
        vm.mem.set_host_fault_recovery(routed);
        vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(LEFT, (WORD + 4 - LEFT) as usize, Prot::RW, RegionKind::Ram)
            .unwrap();
        vm.map(WORD + 4, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
        // Writer at 0x1000: alternate 0 and all-ones until the reader sets FLAG.
        let mut w = CodeAssembler::new(64).unwrap();
        let mut top = w.create_label();
        w.set_label(&mut top).unwrap();
        w.mov(qword_ptr(rdi), rax).unwrap();
        w.not(rax).unwrap();
        w.cmp(byte_ptr(rsi), 0).unwrap();
        w.je(top).unwrap();
        w.hlt().unwrap();
        vm.write_bytes(0x1000, &w.assemble(0x1000).unwrap())
            .unwrap();
        // Reader at 0x1400: RCX loads; stop early on a torn value (RCX != 0).
        let mut r = CodeAssembler::new(64).unwrap();
        let mut top = r.create_label();
        let mut ok = r.create_label();
        let mut done = r.create_label();
        r.set_label(&mut top).unwrap();
        r.mov(rbx, qword_ptr(rdi)).unwrap();
        r.test(rbx, rbx).unwrap();
        r.jz(ok).unwrap();
        r.cmp(rbx, -1).unwrap();
        r.je(ok).unwrap();
        r.jmp(done).unwrap();
        r.set_label(&mut ok).unwrap();
        r.dec(rcx).unwrap();
        r.jnz(top).unwrap();
        r.set_label(&mut done).unwrap();
        r.mov(byte_ptr(rsi), 1).unwrap();
        r.hlt().unwrap();
        vm.write_bytes(0x1400, &r.assemble(0x1400).unwrap())
            .unwrap();
        let vm = Arc::new(vm);
        let start = Arc::new(Barrier::new(2));
        let writer = {
            let vm = vm.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.set_reg(Reg::Rdi, WORD);
                cpu.set_reg(Reg::Rsi, FLAG);
                start.wait();
                let exit = cpu.run(&vm, None);
                assert!(matches!(exit, Exit::Hlt), "writer: {exit:?}");
            })
        };
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1400);
        cpu.set_reg(Reg::Rdi, WORD);
        cpu.set_reg(Reg::Rsi, FLAG);
        cpu.set_reg(Reg::Rcx, 200_000);
        start.wait();
        let exit = cpu.run(&vm, None);
        // A reader that stopped early still releases the writer.
        vm.write_bytes(FLAG, &[1]).unwrap();
        writer.join().unwrap();
        assert!(matches!(exit, Exit::Hlt), "reader: {exit:?}");
        assert_eq!(
            cpu.reg(Reg::Rcx),
            0,
            "routed={routed}: torn value {:#x}",
            cpu.reg(Reg::Rbx)
        );
    }
}

/// A 16-byte store is two 8-byte transfers. Layout: RAM [LEFT, EDGE) | RAM
/// [EDGE, EDGE+4) | gap. MOVDQU at EDGE-4: the first transfer covers both RAM
/// regions, the second covers [EDGE+4, EDGE+12), which is the gap.
fn vstore16_split_then_gap(tier: Tier, routed: bool, window: bool) -> (Exit, Vec<u8>) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend(tier));
    vm.set_cpu_mode(CpuMode::Long64);
    vm.set_tier_up_after(None);
    vm.mem.set_host_fault_recovery(routed);
    vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.map(LEFT, (EDGE - LEFT) as usize, Prot::RW, RegionKind::Ram)
        .unwrap();
    vm.map(EDGE, 4, Prot::RW, RegionKind::Ram).unwrap();
    if window {
        vm.map(0x2000, 0x10, Prot::RW, RegionKind::Trap).unwrap();
        vm.map(0x9000, 0x10, Prot::RW, RegionKind::Trap).unwrap();
    }
    let mut a = CodeAssembler::new(64).unwrap();
    a.movdqu(xmmword_ptr(rdi), xmm0).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
    if tier == Tier::Region {
        vm.test_support_force_region(CODE, CAPS, 0x1f80).unwrap();
    }
    poke(&vm, EDGE - 8, &[0xcc; 12]);
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rdi, EDGE - 4);
    cpu.set_xmm(
        0,
        u128::from_le_bytes(std::array::from_fn(|i| 0x31 + i as u8)),
    );
    let exit = cpu.run(&vm, Some(100));
    (exit, peek(&vm, EDGE - 8, 12))
}

/// SDM Vol 3A §7.5: a faulting MOVDQU leaves memory untouched; the first transfer
/// lying in two adjacent RAM regions does not let it commit before the second
/// transfer faults, with or without host-fault recovery. The fault names the
/// second transfer's first unmapped byte.
#[test]
fn vstore16_split_then_gap_commits_nothing() {
    for (tier, window) in [
        (Tier::Interp, false),
        (Tier::Block, true),
        (Tier::Region, true),
    ] {
        for routed in [false, true] {
            let what = format!("{tier:?} routed={routed}");
            let (exit, mem) = vstore16_split_then_gap(tier, routed, window);
            println!("{what}: {exit:?}");
            assert!(
                matches!(exit, Exit::UnmappedMemory { addr, access: AccessKind::Write, .. }
                    if addr == EDGE + 4),
                "{what}: {exit:?}"
            );
            assert_eq!(mem, [0xcc; 12], "{what}: partial commit, exit {exit:?}");
        }
    }
}

/// On I80386 with A20 off, a LOCK or XCHG dword at FFFF:000E is split at the bus
/// into [FFFFEh, 100000h) and [0, 2). A region boundary inside one piece keeps the
/// containment miss of a locked operand, as it does with A20 on.
#[test]
fn a20_split_lock_keeps_containment_miss() {
    for boundary in [0xf_ffffu64, 1] {
        for op in [&[0x66u8, 0xf0, 0x01, 0x04][..], &[0x66, 0x87, 0x04]] {
            let what = format!("boundary={boundary:#x} op={op:02x?}");
            let mut vm = legacy_vm(Legacy::I386, 0x12_0000);
            vm.map(0, boundary as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.map(
                boundary,
                (0x10_0000 - boundary) as usize,
                Prot::RWX,
                RegionKind::Ram,
            )
            .unwrap();
            // mov ax,0FFFFh ; mov ds,ax ; <op> [si] ; hlt
            let mut code = vec![0xb8, 0xff, 0xff, 0x8e, 0xd8];
            code.extend_from_slice(op);
            code.push(0xf4);
            vm.write_bytes(0x1000, &code).unwrap();
            poke(&vm, 0xf_fffe, &[0x5a; 2]);
            poke(&vm, 0, &[0x5a; 2]);
            let mut cpu = legacy_cpu(&vm, Legacy::I386);
            cpu.set_a20_enabled(false);
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rsi, 0x0e);
            cpu.set_reg(Reg::Rax, 0x1122_3344);
            let exit = legacy_run(&vm, &mut cpu, 10);
            println!("{what}: {exit:?}");
            let expected = if boundary == 1 { 0 } else { 0xf_fffe };
            assert!(
                matches!(exit, Exit::UnmappedMemory { addr, access: AccessKind::Write, .. }
                    if addr == expected),
                "{what}: {exit:?}"
            );
            assert_eq!(peek(&vm, 0xf_fffe, 2), [0x5a; 2], "{what}");
            assert_eq!(peek(&vm, 0, 2), [0x5a; 2], "{what}");
        }
    }
}

/// An automatic child view (`Vm::child_view`) joins adjacent regions of one kind
/// for data accesses as the parent does, so the parent interpreter and the child
/// agree on an operand over two RAM regions. A LOCK operand keeps the containment
/// fault in both.
#[test]
fn child_view_agrees_with_parent_on_split_operand() {
    use x86jit_core::{child, ChildArena, ChildStep};
    let codes: [(&str, &[u8]); 3] = [
        ("load", &[0x48, 0x8b, 0x18, 0xf4]),
        ("store", &[0x48, 0x89, 0x18, 0xf4]),
        ("lock add", &[0xf0, 0x48, 0x01, 0x18, 0xf4]),
    ];
    for (what, code) in codes {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(InterpreterBackend));
        vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(LEFT, (EDGE - LEFT) as usize, Prot::RW, RegionKind::Ram)
            .unwrap();
        vm.map(EDGE, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, code).unwrap();
        let mut parent = vm.new_vcpu();
        parent.set_reg(Reg::Rip, 0x1000);
        parent.set_reg(Reg::Rax, EDGE - 4);
        parent.set_reg(Reg::Rbx, 0x1122_3344_5566_7788);
        let mut cpu = parent.cpu.clone();
        let parent_ok = matches!(parent.run(&vm, Some(100)), Exit::Hlt);
        let view = vm.child_view(CpuMode::Long64);
        let mut arena = ChildArena::new(16);
        let first = child::step(&view, &mut cpu, &mut arena);
        let child_ok = matches!(first, ChildStep::Continue)
            && matches!(child::step(&view, &mut cpu, &mut arena), ChildStep::Hlt);
        println!("{what}: parent={parent_ok} child={child_ok} first={first:?}");
        assert_eq!(parent_ok, what != "lock add", "{what}");
        assert_eq!(parent_ok, child_ok, "{what}: first child step {first:?}");
    }
}

/// Witness of a known limit: a child view joins only regions of one kind, so a data
/// operand over adjacent RAM and ROM regions completes in the parent and faults in
/// the child at the operand base.
#[test]
fn child_view_keeps_the_containment_fault_over_ram_and_rom() {
    use x86jit_core::{child, ChildArena, ChildStep};
    for store in [false, true] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(InterpreterBackend));
        vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.map(LEFT, (EDGE - LEFT) as usize, Prot::RW, RegionKind::Ram)
            .unwrap();
        vm.map(EDGE, 0x1000, Prot::R, RegionKind::Rom).unwrap();
        let code: &[u8] = if store {
            &[0x48, 0x89, 0x18, 0xf4]
        } else {
            &[0x48, 0x8b, 0x18, 0xf4]
        };
        vm.write_bytes(0x1000, code).unwrap();
        let mut parent = vm.new_vcpu();
        parent.set_reg(Reg::Rip, 0x1000);
        parent.set_reg(Reg::Rax, EDGE - 4);
        let mut cpu = parent.cpu.clone();
        assert!(
            matches!(parent.run(&vm, Some(100)), Exit::Hlt),
            "store={store}"
        );
        let view = vm.child_view(CpuMode::Long64);
        let mut arena = ChildArena::new(16);
        let step = child::step(&view, &mut cpu, &mut arena);
        println!("store={store}: {step:?}");
        let ChildStep::Fault(fault) = step else {
            panic!("store={store}: {step:?}");
        };
        assert_eq!(fault.operand_base, x86jit_core::Knowledge::Known(EDGE - 4));
    }
}

/// The re-execution after a containment miss (`interp::split_block`): every case
/// compares a layout whose operand covers two adjacent RAM regions with the same
/// program and operand inside one region. Exit, registers, memory, retirement and
/// step counts must be equal, also when the split access is not the first
/// instruction of its block and under every `run_steps` budget.
mod reexecution {
    use super::{backend, peek, poke, Tier, CAPS, EDGE, TIERS};
    use iced_x86::code_asm::{
        qword_ptr, rax, rbx, rcx, rdi, rdx, rsi, xmm0, xmmword_ptr, CodeAssembler,
    };
    use x86jit_core::{
        CpuMode, CpuModel, CpuState, InterpreterBackend, Prot, Reg, RegionKind, StepOptions,
        StepStop, Vm, VmConfig,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Target {
        Alias,
        I286,
        I386,
        I8086,
        V30,
    }
    const TARGETS: [Target; 5] = [
        Target::Alias,
        Target::I286,
        Target::I386,
        Target::I8086,
        Target::V30,
    ];

    /// `split` maps 0x3000..EDGE and EDGE..0x8000 as two RAM regions; otherwise
    /// one RAM region covers 0x3000..0x8000. Code, IVT and handlers live below 0x2000.
    fn vm(target: Target, split: bool, mode: CpuMode) -> Vm {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10_0000), Box::new(InterpreterBackend));
        vm.set_cpu_mode(mode);
        match target {
            Target::Alias => {}
            Target::I286 => vm.set_cpu_model(CpuModel::I80286),
            Target::I386 => vm.set_cpu_model(CpuModel::I80386),
            Target::I8086 => vm.set_cpu_model(CpuModel::I8086),
            Target::V30 => vm.set_cpu_model(CpuModel::NecV30),
        }
        vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
        if split {
            vm.map(0x3000, (EDGE - 0x3000) as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.map(EDGE, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        } else {
            vm.map(0x3000, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
        }
        vm
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Outcome {
        exit: String,
        rip: u64,
        sp: u64,
        frame: Vec<u8>,
        retired: u64,
    }

    fn frame_run(target: Target, split: bool, code: &[u8], sp: u64, stepped: bool) -> Outcome {
        let vm = vm(target, split, CpuMode::Real16);
        vm.write_bytes(0x1000, code).unwrap();
        // IVT: vectors 3, 6 and 0x21 -> 0000:1800 (HLT).
        for v in [3u64, 6, 0x21] {
            vm.mem.write(v * 4, 0x1800, 4).unwrap();
        }
        vm.write_bytes(0x1800, &[0xf4]).unwrap();
        for i in 0..16u64 {
            vm.write_bytes(sp - 8 + i, &[0xcc]).unwrap();
        }
        let mut cpu = vm.new_vcpu();
        if matches!(target, Target::I286 | Target::I386) {
            cpu.cpu = CpuState::legacy_real_entry([0; 6]);
        }
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.set_reg(Reg::Rsp, sp);
        let exit = if stepped {
            let r = cpu.run_steps(&vm, 20, StepOptions::default()).unwrap();
            match r.stop() {
                StepStop::Exit(exit) => format!("{exit:?} steps={}", r.steps()),
                other => format!("{other:?}"),
            }
        } else {
            format!("{:?}", cpu.run(&vm, None))
        };
        let mut frame = vec![0; 16];
        for (i, b) in frame.iter_mut().enumerate() {
            let mut x = [0];
            vm.read_bytes(sp - 8 + i as u64, &mut x).unwrap();
            *b = x[0];
        }
        Outcome {
            exit,
            rip: cpu.reg(Reg::Rip),
            sp: cpu.reg(Reg::Rsp) & 0xffff,
            frame,
            retired: cpu.retired_instructions(),
        }
    }

    /// SDM Vol 3B §23.1.4: a real-address interrupt pushes FLAGS, CS and IP. A frame
    /// whose CS or IP word covers two adjacent RAM regions gives the same SP, frame
    /// and RIP as the same frame inside one region (the frame helpers commit SP per
    /// word, so they span the word themselves rather than re-executing).
    #[test]
    fn int_frame_across_ram_regions_matches_contained() {
        let mut failures = Vec::new();
        for target in TARGETS {
            // nop; int 21h  and  nop; int3 (a NOP first, so the gate is not the
            // block's first instruction)
            for code in [&[0x90u8, 0xcd, 0x21][..], &[0x90, 0xcc][..]] {
                // SP = EDGE+1: FLAGS word straddles; +3: CS word; +5: IP word.
                for sp in [EDGE + 1, EDGE + 3, EDGE + 5] {
                    for stepped in [false, true] {
                        let contained = frame_run(target, false, code, sp, stepped);
                        let split = frame_run(target, true, code, sp, stepped);
                        if contained != split {
                            failures.push(format!(
                                "{target:?} code={code:x?} sp={sp:#x} stepped={stepped}:\n  \
                                 contained={contained:x?}\n  split    ={split:x?}"
                            ));
                        }
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// Every real-mode stack helper that commits SP (or BP) between its transfers:
    /// far CALL, PUSHA, ENTER, PUSHF pushing down over EDGE, and IRET, RETF, POPA,
    /// POPF popping up over it, each after a NOP, with each word in turn straddling.
    /// State, frame, retirement and step counts equal the contained run.
    #[test]
    fn stack_frames_across_ram_regions_match_contained() {
        let snapshot = |target: Target, split: bool, code: &[u8], sp: u64, stepped: bool| {
            let vm = vm(target, split, CpuMode::Real16);
            vm.write_bytes(0x1000, code).unwrap();
            vm.write_bytes(0x1800, &[0xf4]).unwrap();
            for i in 0..48u64 {
                vm.write_bytes(sp - 24 + i, &[0xa0 | (i as u8 & 0xf)])
                    .unwrap();
            }
            // A return frame for the pops: IP=0x1800, CS=0, FLAGS=0x0202, then
            // values for POPA.
            for (i, w) in [0x1800u16, 0, 0x0202, 0x1234, 0x5678, 0x9abc, 0xdef0, 0x1357]
                .iter()
                .enumerate()
            {
                poke(&vm, sp + 2 * i as u64, &w.to_le_bytes());
            }
            let mut cpu = vm.new_vcpu();
            if matches!(target, Target::I286 | Target::I386) {
                cpu.cpu = CpuState::legacy_real_entry([0; 6]);
            }
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rsp, sp);
            cpu.set_reg(Reg::Rbp, sp + 4);
            cpu.set_reg(Reg::Rax, 0x1111);
            let stop = if stepped {
                let r = cpu.run_steps(&vm, 20, StepOptions::default()).unwrap();
                format!("steps={} {:?}", r.steps(), r.stop())
            } else {
                format!("{:?}", cpu.run(&vm, None))
            };
            let gprs: Vec<u64> = [
                Reg::Rax,
                Reg::Rcx,
                Reg::Rdx,
                Reg::Rbx,
                Reg::Rsp,
                Reg::Rbp,
                Reg::Rsi,
                Reg::Rdi,
            ]
            .iter()
            .map(|&r| cpu.reg(r))
            .collect();
            format!(
                "{stop} rip={:#x} cs={:#x} rflags={:#x} gpr={gprs:x?} retired={} executed={} frame={:x?}",
                cpu.reg(Reg::Rip),
                cpu.reg(Reg::Cs),
                cpu.cpu.rflags(),
                cpu.retired_instructions(),
                cpu.executed_instructions(),
                peek(&vm, sp - 24, 48)
            )
        };
        let pushes: [&[u8]; 4] = [
            &[0x90, 0x9a, 0x00, 0x18, 0x00, 0x00], // call far 0000:1800
            &[0x90, 0x60, 0xf4],                   // pusha
            &[0x90, 0xc8, 0x04, 0x00, 0x03, 0xf4], // enter 4, 3
            &[0x90, 0x9c, 0xf4],                   // pushf
        ];
        let pops: [&[u8]; 4] = [
            &[0x90, 0xcf],       // iret
            &[0x90, 0xcb],       // retf
            &[0x90, 0x61, 0xf4], // popa
            &[0x90, 0x9d, 0xf4], // popf
        ];
        let mut failures = Vec::new();
        for target in TARGETS {
            for stepped in [false, true] {
                let cases = pushes
                    .iter()
                    .flat_map(|&code| (0..9).map(move |k| (code, EDGE + 1 + 2 * k)))
                    .chain(
                        pops.iter()
                            .flat_map(|&code| (0..9).map(move |k| (code, EDGE - 1 - 2 * k))),
                    );
                for (code, sp) in cases {
                    let contained = snapshot(target, false, code, sp, stepped);
                    let split = snapshot(target, true, code, sp, stepped);
                    if contained != split {
                        failures.push(format!(
                            "{target:?} code={code:x?} sp={sp:#x} stepped={stepped}:\n  \
                             contained={contained}\n  split    ={split}"
                        ));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    fn snapshot(
        target: Target,
        split: bool,
        code: &[u8],
        regs: &[(Reg, u64)],
        budget: Option<u64>,
    ) -> String {
        let vm = vm(target, split, CpuMode::Real16);
        vm.write_bytes(0x1000, code).unwrap();
        for i in 0..0x20u64 {
            vm.write_bytes(EDGE - 0x10 + i, &[i as u8]).unwrap();
        }
        let mut cpu = vm.new_vcpu();
        if matches!(target, Target::I286 | Target::I386) {
            cpu.cpu = CpuState::legacy_real_entry([0; 6]);
        }
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.set_reg(Reg::Rsp, 0x1f00);
        for &(r, v) in regs {
            cpu.set_reg(r, v);
        }
        let stop = match budget {
            Some(b) => {
                let r = cpu.run_steps(&vm, b, StepOptions::default()).unwrap();
                format!("steps={} {:?}", r.steps(), r.stop())
            }
            None => format!("{:?}", cpu.run(&vm, None)),
        };
        let mem = peek(&vm, EDGE - 0x10, 0x20);
        let gprs: Vec<u64> = [
            Reg::Rax,
            Reg::Rcx,
            Reg::Rdx,
            Reg::Rbx,
            Reg::Rsp,
            Reg::Rsi,
            Reg::Rdi,
        ]
        .iter()
        .map(|&r| cpu.reg(r) & 0xffff)
        .collect();
        format!(
            "{stop} rip={:#x} gpr={gprs:x?} retired={} executed={} mem={mem:x?}",
            cpu.reg(Reg::Rip),
            cpu.retired_instructions(),
            cpu.executed_instructions()
        )
    }

    /// A named program with its initial registers.
    type Case<'a> = (&'a str, &'a [u8], Vec<(Reg, u64)>);

    /// A split access in the middle of a block (after a non-idempotent INC) and a
    /// REP MOVSW whose destination crosses the edge: state, step counts and
    /// retirement equal the contained run under `run` and `run_steps` budgets.
    #[test]
    fn real16_block_accounting_matches_contained() {
        // inc cx; mov [di],ax; inc dx; mov bx,[di]; inc si; hlt
        let straight: &[u8] = &[0x41, 0x89, 0x05, 0x42, 0x8b, 0x1d, 0x46, 0xf4];
        // cld; rep movsw; hlt  (DS:SI contained source, ES:DI destination over EDGE)
        let rep: &[u8] = &[0xfc, 0xf3, 0xa5, 0xf4];
        let cases: [Case; 3] = [
            (
                "straight",
                straight,
                vec![(Reg::Rdi, EDGE - 1), (Reg::Rax, 0xa1b2)],
            ),
            (
                "rep-movsw",
                rep,
                vec![(Reg::Rdi, EDGE - 3), (Reg::Rsi, 0x1100), (Reg::Rcx, 4)],
            ),
            (
                "rep-movsw-first",
                rep,
                vec![(Reg::Rdi, EDGE - 1), (Reg::Rsi, 0x1100), (Reg::Rcx, 4)],
            ),
        ];
        let mut failures = Vec::new();
        for target in TARGETS {
            for (name, code, regs) in &cases {
                for budget in [
                    None,
                    Some(1),
                    Some(2),
                    Some(3),
                    Some(4),
                    Some(5),
                    Some(6),
                    Some(7),
                    Some(9),
                ] {
                    let contained = snapshot(target, false, code, regs, budget);
                    let split = snapshot(target, true, code, regs, budget);
                    if contained != split {
                        failures.push(format!(
                            "{target:?} {name} budget={budget:?}\n  contained={contained}\n  split    ={split}"
                        ));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// A Long64 block with a scalar store, a 16-byte store and a scalar load over
    /// the edge between non-idempotent INCs, on every tier, with the compiled
    /// tiers' Trap window on and off (`window` hands the accesses to the
    /// interpreter). `run_steps` serves Real16 only, so retirement and the
    /// executed-instruction count stand for the step accounting here.
    #[test]
    fn long64_block_accounting_matches_contained() {
        let mut a = CodeAssembler::new(64).unwrap();
        a.inc(rcx).unwrap();
        a.mov(qword_ptr(rdi), rax).unwrap();
        a.inc(rdx).unwrap();
        a.movdqu(xmmword_ptr(rdi + 8), xmm0).unwrap();
        a.mov(rbx, qword_ptr(rdi)).unwrap();
        a.inc(rsi).unwrap();
        a.hlt().unwrap();
        let code = a.assemble(0x1000).unwrap();
        let mut failures = Vec::new();
        for tier in TIERS {
            for window in [false, true] {
                for at in [EDGE - 3, EDGE - 12, EDGE - 20] {
                    let contained = long64_run(tier, window, false, &code, at);
                    let split = long64_run(tier, window, true, &code, at);
                    if contained != split {
                        failures.push(format!(
                            "{tier:?} window={window} at={at:#x}\n  \
                             contained={contained}\n  split    ={split}"
                        ));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    fn long64_run(tier: Tier, window: bool, split: bool, code: &[u8], at: u64) -> String {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10_0000), backend(tier));
        vm.set_cpu_mode(CpuMode::Long64);
        vm.set_tier_up_after(None);
        vm.map(0, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
        if split {
            vm.map(0x3000, (EDGE - 0x3000) as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.map(EDGE, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        } else {
            vm.map(0x3000, 0x5000, Prot::RWX, RegionKind::Ram).unwrap();
        }
        if window {
            vm.map(0x2000, 0x10, Prot::RW, RegionKind::Trap).unwrap();
            vm.map(0x9000, 0x10, Prot::RW, RegionKind::Trap).unwrap();
        }
        vm.write_bytes(0x1000, code).unwrap();
        if tier == Tier::Region {
            vm.test_support_force_region(0x1000, CAPS, 0x1f80).unwrap();
        }
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        cpu.set_reg(Reg::Rsp, 0x1f00);
        cpu.set_reg(Reg::Rdi, at);
        cpu.set_reg(Reg::Rax, 0x1122_3344_5566_7788);
        let exit = cpu.run(&vm, None);
        format!(
            "{exit:?} rip={:#x} rbx={:#x} rcx={} rdx={} rsi={} retired={} executed={} mem={:x?}",
            cpu.reg(Reg::Rip),
            cpu.reg(Reg::Rbx),
            cpu.reg(Reg::Rcx),
            cpu.reg(Reg::Rdx),
            cpu.reg(Reg::Rsi),
            cpu.retired_instructions(),
            cpu.executed_instructions(),
            peek(&vm, at - 8, 0x30)
        )
    }
}
