//! Data-segment loads inside Seg16, Flat32 and V86 blocks, and same-level
//! far transfers on the paged lanes, against the Checked reference
//! (spec.md §17.10, xj-to9.18).
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::descriptor;
use x86jit_core::fault::ErrorCode;
use x86jit_core::{
    derive_legacy_context, CpuModel, CpuState, ExceptionDelivery, Exit, Prot, RegionKind,
    SegmentClass, StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

const MODELS: [CpuModel; 2] = [CpuModel::I80386, CpuModel::I80486Dx];

fn class(cpu: &Vcpu) -> SegmentClass {
    derive_legacy_context(cpu.legacy_cpu_model().unwrap(), &cpu.cpu)
        .unwrap()
        .segment_class
}

fn exit(run: &x86jit_core::StepRun) -> &Exit {
    let StepStop::Exit(exit) = run.stop() else {
        panic!("{run:?}")
    };
    exit
}

/// Seg16 at `cpl`: CS 08h (base 1000h), DS 10h (A000h, limit FFh), SS 18h,
/// with the extra descriptors the witnesses load. `checked` clears an unused
/// GS cache's P bit, which keeps the same program on the Checked class.
fn seg16(model: CpuModel, code: &[u8], cpl: u8, paged: bool, checked: bool) -> (Vm, Vcpu) {
    seg16_delivery(model, code, cpl, paged, checked, ExceptionDelivery::Outward)
}

fn seg16_delivery(
    model: CpuModel,
    code: &[u8],
    cpl: u8,
    paged: bool,
    checked: bool,
    delivery: ExceptionDelivery,
) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x20000));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(delivery);
    vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1100, code).unwrap();
    let dpl = u16::from(cpl) << 5;
    let mut cpu = vm.new_vcpu();
    cpu.set_a20_enabled(true);
    cpu.cpu.rip = 0x100;
    cpu.cpu.gpr[4] = 0x1000;
    cpu.cpu.legacy.cr0 = if paged { 0x80000001 } else { 1 };
    cpu.cpu.legacy.cr3 = 0x1e000;
    cpu.cpu.legacy.cpl = cpl;
    cpu.cpu.legacy.gdtr.base = 0x3000;
    cpu.cpu.legacy.gdtr.limit = 0x7f;
    cpu.cpu.cs = 8 | u16::from(cpl);
    cpu.cpu.ds = 16 | u16::from(cpl);
    cpu.cpu.es = cpu.cpu.ds;
    cpu.cpu.ss = 24 | u16::from(cpl);
    cpu.cpu.fs = cpu.cpu.ds;
    cpu.cpu.gs = cpu.cpu.ds;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        cache.base = [0x1000, 0xa000, 0xa000, 0x7000, 0xa000, 0xa000][i];
        cache.limit = if i == 0 || i == 3 { 0x1fff } else { 0xff };
        cache.attributes = (if i == 0 { 0x9b } else { 0x93 }) | dpl;
        cache.unusable = 0;
    }
    descriptor(&vm, 0x3008, 0x1000, 0x1fff, 0x9b | dpl);
    descriptor(&vm, 0x3010, 0xa000, 0xff, 0x93 | dpl);
    descriptor(&vm, 0x3018, 0x7000, 0x1fff, 0x93 | dpl);
    // A second code segment, and data with a sixteen-byte limit.
    descriptor(&vm, 0x3020, 0x4000, 0xfff, 0x9b | dpl);
    descriptor(&vm, 0x3028, 0xb000, 0xf, 0x93 | dpl);
    descriptor(&vm, 0x3030, 0x5000, 0x1fff, 0x93 | dpl);
    // Not present; execute-only code; DPL 0 data; an LDT descriptor.
    descriptor(&vm, 0x3038, 0x5000, 0xff, 0x13 | dpl);
    descriptor(&vm, 0x3040, 0x4000, 0xfff, 0x99 | dpl);
    descriptor(&vm, 0x3048, 0x5000, 0xff, 0x93);
    descriptor(&vm, 0x3050, 0x5000, 0xff, 0x82);
    // Data with A clear: a load sets it in memory and in the cache.
    descriptor(&vm, 0x3060, 0x5000, 0x1fff, 0x92 | dpl);
    // A writable alias of the second code segment.
    descriptor(&vm, 0x3070, 0x4000, 0xfff, 0x93 | dpl);
    // The far callee: INC DX; RETF.
    vm.write_bytes(0x4000, b"\x42\xcb").unwrap();
    if paged {
        vm.write_bytes(0x1e000, &0x1f007u32.to_le_bytes()).unwrap();
        for page in 0..32u32 {
            vm.write_bytes(
                0x1f000 + u64::from(page * 4),
                &((page * 4096) | 7).to_le_bytes(),
            )
            .unwrap();
        }
    }
    if checked {
        // SDM Vol 3A §3.4.5: P is checked on a descriptor load, not on an
        // access through the cache; no witness program touches GS.
        cpu.cpu.legacy.segments[5].attributes &= !0x80;
    }
    assert_eq!(
        class(&cpu),
        if checked {
            SegmentClass::Checked
        } else {
            SegmentClass::Seg16
        }
    );
    (vm, cpu)
}

/// The registers without the retry ledger's dead entries: only the first
/// `mmio_parts_len` entries of `mmio_parts` are ever read.
fn live(cpu: &CpuState) -> String {
    let mut cpu = cpu.clone();
    cpu.mmio_parts[usize::from(cpu.mmio_parts_len)..].fill((0, 0));
    format!("{cpu:?}")
}

fn same(a: &Vcpu, b: &mut Vcpu, avm: &Vm, bvm: &Vm, size: usize, what: &str) {
    b.cpu.legacy.segments[5].attributes |= 0x80;
    let equal = live(&a.cpu) == live(&b.cpu);
    b.cpu.legacy.segments[5].attributes &= !0x80;
    assert!(equal, "{what}\n{:?}\n{:?}", a.cpu, b.cpu);
    assert_eq!(a.retired_instructions(), b.retired_instructions(), "{what}");
    let mut am = vec![0; size];
    let mut bm = vec![0; size];
    avm.read_bytes(0, &mut am).unwrap();
    bvm.read_bytes(0, &mut bm).unwrap();
    assert!(am == bm, "{what}: memory");
}

// MOV DS,AX; MOV CX,[10h]; MOV ES,AX; MOV ES:[12h],CX; INC BX; PUSH DS;
// POP ES; MOV DX,ES:[14h]; LDS SI,[20h]; MOV DI,[SI]; CALL FAR 20h:0;
// MOV FS,AX; MOV BX,FS:[0]; MOV DS,BP; MOV CX,[2]; HLT.
fn census_program(cpl: u8) -> Vec<u8> {
    let mut code = b"\x8e\xd8\x8b\x0e\x10\x00\x8e\xc0\x26\x89\x0e\x12\x00\x43\x1e\x07".to_vec();
    code.extend(b"\x26\x8b\x16\x14\x00\xc5\x36\x20\x00\x8b\x3c");
    code.extend([0x9a, 0, 0, 0x20 | cpl, 0]);
    code.extend(b"\x8e\xe0\x64\x8b\x1e\x00\x00\x8e\xdd\x8b\x0e\x02\x00\xf4");
    code
}

#[test]
fn seg16_in_block_loads_and_far_transfers_match_checked() {
    // SDM Vol 2B MOV/POP/LDS (Protected Mode Exceptions), Vol 3A §§3.4.3,
    // 6.3–6.4, 6.8: the same descriptor checks, accessed-bit writes and
    // whole-operand limits against each newly loaded cache, at every budget.
    for model in MODELS {
        for cpl in [0, 3] {
            let rpl = u16::from(cpl);
            for paged in [false, true] {
                // DS sources: the entry DS, a larger segment, the 16-byte
                // segment (its [10h] faults), null (the access faults), and
                // a descriptor whose A bit the load sets.
                for selector in [0x10, 0x30, 0x28, 0, 0x60] {
                    let code = census_program(cpl);
                    for budget in [0, 1, 2, 3, 4, 6, 8, 10, 11, 12, 13, 14, 16, 40] {
                        let (avm, mut a) = seg16(model, &code, cpl, paged, false);
                        let (bvm, mut b) = seg16(model, &code, cpl, paged, true);
                        for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                            cpu.cpu.gpr[0] = u64::from(selector | rpl);
                            cpu.cpu.gpr[5] = u64::from(0x10 | rpl);
                            for base in [0xa020, 0x5020] {
                                vm.write_bytes(base, &[0x40, 0, 0x30 | cpl, 0]).unwrap();
                            }
                            vm.write_bytes(0x5040, &0x1234u16.to_le_bytes()).unwrap();
                        }
                        x86jit_core::Vcpu::take_machine_counters_for_test();
                        let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                        let [blocks, ..] = x86jit_core::Vcpu::take_machine_counters_for_test();
                        let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                        let what = format!(
                            "{model:?} CPL{cpl} PG={paged} DS={selector:#x} budget={budget}"
                        );
                        assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what}");
                        same(&a, &mut b, &avm, &bvm, 0x20000, &what);
                        if selector == 0x10 && budget == 40 {
                            // Every instruction retires; the loads did not
                            // split the blocks around them.
                            assert_eq!(a.retired_instructions(), 18 - u64::from(cpl == 3));
                            assert!(blocks <= 8, "{what}: {blocks} blocks");
                            assert_eq!(a.cpu.gpr[7], 0x1234, "{what}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn every_fault_of_an_in_block_load_matches_checked() {
    // SDM Vol 2B MOV, Protected Mode Exceptions; 80386 PRM §17 MOV
    // pp. 17-109–110: null SS #GP(0); index beyond the table, a system or
    // execute-only descriptor, or RPL/DPL > the data DPL #GP(selector); not
    // present #NP(selector), or #SS(selector) for SS. SDM Vol 3A §5.6.1;
    // 80386 PRM §6.5 p. 6-22: the descriptor read is a supervisor access,
    // so a missing descriptor page is #PF with U/S = 0 and CR2 = its address.
    for model in MODELS {
        for cpl in [0u8, 3] {
            let rpl = u16::from(cpl);
            for paged in [false, true] {
                let mut cases: Vec<(u8, u16, u8, u16)> = vec![
                    (0xd8, 0x38 | rpl, 11, 0x38),
                    (0xd8, 0x40 | rpl, 13, 0x40),
                    (0xd8, 0x50 | rpl, 13, 0x50),
                    (0xd8, 0x80 | rpl, 13, 0x80),
                    (0xd8, 0x0c | rpl, 13, 0x0c),
                    (0xe0, 0x38 | rpl, 11, 0x38),
                    (0xd0, rpl, 13, 0),
                    (0xd0, 0x38 | rpl, 12, 0x38),
                ];
                if cpl == 3 {
                    cases.push((0xc0, 0x48 | 3, 13, 0x48));
                } else {
                    // RPL 3 above DPL 0.
                    cases.push((0xc0, 0x48 | 3, 13, 0x48));
                    cases.push((0xd8, 0x10 | 3, 13, 0x10));
                }
                for (modrm, selector, vector, error) in cases {
                    // INC BX; MOV Sreg,AX; INC BX; MOV CX,[0].
                    let code = [0x43, 0x8e, modrm, 0x43, 0x8b, 0x0e, 0, 0];
                    let (avm, mut a) = seg16(model, &code, cpl, paged, false);
                    let (bvm, mut b) = seg16(model, &code, cpl, paged, true);
                    for cpu in [&mut a, &mut b] {
                        cpu.cpu.gpr[0] = u64::from(selector);
                    }
                    let ar = a.run_steps(&avm, 10, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, 10, StepOptions::default()).unwrap();
                    let what = format!("{model:?} CPL{cpl} PG={paged} {modrm:#x} {selector:#x}");
                    assert_eq!(ar.steps(), 1, "{what}");
                    assert_eq!(a.cpu.rip, 0x101, "{what}");
                    let Exit::Exception { vector: actual, .. } = exit(&ar) else {
                        panic!("{what}: {ar:?}")
                    };
                    assert_eq!(*actual, vector, "{what}");
                    assert_eq!(
                        exit(&ar).fault_metadata().unwrap().error_code(),
                        ErrorCode::bits(u32::from(error), 0xffff),
                        "{what}"
                    );
                    assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what}");
                    same(&a, &mut b, &avm, &bvm, 0x20000, &what);
                }
                if paged {
                    // The GDT page is not present: #PF(0) at the descriptor.
                    let code = [0x43, 0x8e, 0xd8, 0x43];
                    let (avm, mut a) = seg16(model, &code, cpl, true, false);
                    let (bvm, mut b) = seg16(model, &code, cpl, true, true);
                    for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                        cpu.cpu.gpr[0] = u64::from(0x30 | rpl);
                        vm.write_bytes(0x1f00c, &0u32.to_le_bytes()).unwrap();
                    }
                    let ar = a.run_steps(&avm, 10, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, 10, StepOptions::default()).unwrap();
                    assert_eq!(ar.steps(), 1);
                    assert!(matches!(exit(&ar), Exit::Exception { vector: 14, .. }));
                    assert_eq!(
                        exit(&ar).fault_metadata().unwrap().error_code(),
                        ErrorCode::bits(0, 7)
                    );
                    assert_eq!(a.cpu.legacy.cr2, 0x3030);
                    assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                    same(&a, &mut b, &avm, &bvm, 0x20000, "GDT page");
                }
            }
        }
    }
}

#[test]
fn an_access_after_an_in_block_load_checks_the_new_cache() {
    // SDM Vol 3A §§3.4.5.1, 6.3: the operand checks the limit of the
    // descriptor just loaded. [10h] is inside the entry DS (limit FFh) and
    // beyond the loaded one (limit Fh): #GP(0) after two retirements.
    for model in MODELS {
        for cpl in [0u8, 3] {
            for paged in [false, true] {
                for word in [false, true] {
                    // INC BX; MOV DS,AX; MOV CX,[0Fh or 10h]; INC BX.
                    let offset = if word { 0x0f } else { 0x10 };
                    let code = [0x43, 0x8e, 0xd8, 0x8b, 0x0e, offset, 0, 0x43];
                    let (avm, mut a) = seg16(model, &code, cpl, paged, false);
                    let (bvm, mut b) = seg16(model, &code, cpl, paged, true);
                    for cpu in [&mut a, &mut b] {
                        cpu.cpu.gpr[0] = u64::from(0x28 | u16::from(cpl));
                    }
                    let ar = a.run_steps(&avm, 10, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, 10, StepOptions::default()).unwrap();
                    assert_eq!(ar.steps(), 2);
                    assert_eq!(a.cpu.rip, 0x103);
                    assert_eq!(a.cpu.legacy.segments[1].limit, 0xf);
                    assert!(matches!(exit(&ar), Exit::Exception { vector: 13, .. }));
                    assert_eq!(format!("{ar:?}"), format!("{br:?}"));
                    same(&a, &mut b, &avm, &bvm, 0x20000, "new limit");
                }
            }
        }
    }
}

#[test]
fn a_store_into_a_retained_far_callee_runs_the_new_code() {
    // SDM Vol 3A §11.6: a store to code takes effect before it executes. The
    // block that reloads DS with a code alias patches the callee another
    // block already holds; the far call then runs the patched byte.
    for model in MODELS {
        for cpl in [0u8, 3] {
            for paged in [false, true] {
                // MOV DS,AX; MOV [0],CL; CALL FAR 20h:0; MOV DS,BP; HLT.
                let code = [0x8e, 0xd8, 0x88, 0x0e, 0, 0, 0x9a, 0, 0, 0x20 | cpl, 0];
                let mut code = code.to_vec();
                code.extend([0x8e, 0xdd, 0xf4]);
                let (avm, mut a) = seg16(model, &code, cpl, paged, false);
                let (bvm, mut b) = seg16(model, &code, cpl, paged, true);
                for (round, patch) in [0x42u8, 0x40, 0x41, 0x40].into_iter().enumerate() {
                    for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                        cpu.cpu.rip = 0x100;
                        cpu.cpu.gpr[0] = u64::from(0x70 | u16::from(cpl));
                        cpu.cpu.gpr[1] = u64::from(patch);
                        cpu.cpu.gpr[5] = u64::from(0x10 | u16::from(cpl));
                        let before = cpu.cpu.gpr;
                        let run = cpu.run_steps(vm, 20, StepOptions::default()).unwrap();
                        assert_eq!(run.steps(), 7 - u64::from(cpl == 3), "round {round}");
                        let reg = usize::from(patch - 0x40);
                        assert_eq!(cpu.cpu.gpr[reg], before[reg] + 1, "round {round}");
                    }
                    same(&a, &mut b, &avm, &bvm, 0x20000, "callee patch");
                }
            }
        }
    }
}

/// Flat32 at CPL 0 (`legacy_flat32.rs`): 08h flat code, 10h flat data, 18h
/// data with a 64-KiB limit, which ends the Flat32 proof when DS holds it.
fn flat32(model: CpuModel, code: &[u8], paged: bool, checked: bool) -> (Vm, Vcpu) {
    let (vm, mut cpu) = fixture::setup(model, code);
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
    descriptor(&vm, 0x2018, 0, 0xffff, 0x4093);
    // The far callee: INC EDX; RETF.
    vm.write_bytes(0x1800, b"\x42\xcb").unwrap();
    if paged {
        cpu.cpu.legacy.cr0 |= 0x8000_0000;
        cpu.cpu.legacy.cr3 = 0x1e000;
        vm.write_bytes(0x1e000, &0x1f003u32.to_le_bytes()).unwrap();
        for page in 0..0x240u32 {
            vm.write_bytes(
                0x1f000 + u64::from(page * 4),
                &((page * 4096) | 3).to_le_bytes(),
            )
            .unwrap();
        }
    }
    if checked {
        // ES:FFFFFFFF is never reached; this cache cannot prove Flat32.
        cpu.cpu.legacy.segments[2].limit -= 1;
    }
    cpu.set_a20_enabled(true);
    assert_eq!(
        class(&cpu),
        if checked {
            SegmentClass::Checked
        } else {
            SegmentClass::Flat32
        }
    );
    (vm, cpu)
}

#[test]
fn flat32_in_block_loads_keep_the_block_only_while_flat() {
    // SDM Vol 3A §§3.4.5.1, 6.3: after DS holds the 64-KiB segment, an
    // operand at 12345h is #GP(0); the Flat32 block must end at that load.
    // MOV DS,AX; MOV ECX,[3000h]; MOV FS,AX; MOV [3004h],ECX; CALL FAR
    // 8:1800h; MOV EDX,[12345h]; MOV DS,BP; MOV EBX,FS:[3000h]; HLT.
    let mut code = b"\x8e\xd8\x8b\x0d\x00\x30\x00\x00\x8e\xe0\x89\x0d\x04\x30\x00\x00".to_vec();
    code.extend(b"\x9a\x00\x18\x00\x00\x08\x00");
    code.extend(b"\x8b\x15\x45\x23\x01\x00\x8e\xdd\x64\x8b\x1d\x00\x30\x00\x00\xf4");
    for model in MODELS {
        for paged in [false, true] {
            for selector in [0x10u16, 0x18, 0] {
                for budget in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20] {
                    let (avm, mut a) = flat32(model, &code, paged, false);
                    let (bvm, mut b) = flat32(model, &code, paged, true);
                    for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                        cpu.cpu.gpr[0] = u64::from(selector);
                        cpu.cpu.gpr[5] = 0x10;
                        vm.write_bytes(0x3000, &0x89abcdefu32.to_le_bytes())
                            .unwrap();
                    }
                    let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                    let what = format!("{model:?} PG={paged} DS={selector:#x} budget={budget}");
                    assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what}");
                    b.cpu.legacy.segments[2].limit += 1;
                    assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu), "{what}");
                    b.cpu.legacy.segments[2].limit -= 1;
                    assert_eq!(a.retired_instructions(), b.retired_instructions());
                    let mut am = vec![0; 0x240000];
                    let mut bm = vec![0; 0x240000];
                    avm.read_bytes(0, &mut am).unwrap();
                    bvm.read_bytes(0, &mut bm).unwrap();
                    assert!(am == bm, "{what}: memory");
                    if budget == 20 {
                        assert_eq!(
                            class(&a),
                            if selector == 0x10 {
                                SegmentClass::Flat32
                            } else {
                                SegmentClass::Checked
                            },
                            "{what}"
                        );
                        if selector != 0x10 {
                            // MOV ECX,[3000h] faults through null DS, or
                            // MOV EDX,[12345h] beyond the 64-KiB DS.
                            assert!(matches!(exit(&ar), Exit::Exception { vector: 13, .. }));
                        }
                    }
                }
            }
        }
    }
}

/// Paged (or, as the reference, unpaged) V86 at IOPL 3: CS 100h, DS/ES
/// 0A00h, SS 700h, identity page tables at 1E000h with user pages.
fn v86(model: CpuModel, code: &[u8], paged: bool) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x20000));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1100, code).unwrap();
    // The far callee at 400h:0: INC DX; RETF.
    vm.write_bytes(0x4000, b"\x42\xcb").unwrap();
    vm.write_bytes(0x1e000, &0x1f007u32.to_le_bytes()).unwrap();
    for page in 0..32u32 {
        vm.write_bytes(
            0x1f000 + u64::from(page * 4),
            &((page * 4096) | 7).to_le_bytes(),
        )
        .unwrap();
    }
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([0x100, 0xa00, 0xa00, 0x700, 0, 0]);
    cpu.cpu.rip = 0x100;
    cpu.cpu.gpr[4] = 0x1000;
    cpu.cpu.legacy.cr0 = if paged { 0x80000001 } else { 1 };
    cpu.cpu.legacy.cr3 = 0x1e000;
    cpu.cpu.legacy.cpl = 3;
    cpu.cpu.rflags_system = 0x23000; // VM=1, IOPL=3 (SDM Vol 3B §23.2.1).
    cpu.set_a20_enabled(true);
    (vm, cpu)
}

#[test]
fn paged_v86_blocks_with_segment_loads_match_the_unpaged_singletons() {
    // SDM Vol 3B §23.2 p. 23-5; 80386 PRM §15.1.2 pp. 15-2–3: a V86 load
    // sets base = selector × 16 with no descriptor, and every later operand
    // uses the new base; §23.1.1: a whole operand beyond FFFFh is #GP(0) or
    // #SS(0). Paged V86 blocks run them inside one block; unpaged V86 keeps
    // the Checked singleton reference over the same identity-mapped bytes.
    // MOV DS,AX; MOV CX,[10h]; MOV ES,AX; MOV ES:[12h],CX; PUSH DS; POP ES;
    // LDS SI,[20h]; MOV DI,[SI]; CALL FAR 400h:0; MOV FS,AX; MOV BX,FS:[2];
    // MOV DS,BP; MOV CX,[0FFFFh]; HLT.
    let mut code = b"\x8e\xd8\x8b\x0e\x10\x00\x8e\xc0\x26\x89\x0e\x12\x00\x1e\x07".to_vec();
    code.extend(b"\xc5\x36\x20\x00\x8b\x3c\x9a\x00\x00\x00\x04\x8e\xe0\x64\x8b\x1e\x02\x00");
    code.extend(b"\x8e\xdd\x8b\x0e\xff\xff\xf4");
    for model in MODELS {
        for selector in [0x500u16, 0xa00, 0] {
            for budget in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 30] {
                let (avm, mut a) = v86(model, &code, true);
                let (bvm, mut b) = v86(model, &code, false);
                for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                    cpu.cpu.gpr[0] = u64::from(selector);
                    cpu.cpu.gpr[5] = 0xa00;
                    for base in [0x5020u64, 0xa020, 0x20] {
                        vm.write_bytes(base, &[0x40, 0, 0x00, 0x06]).unwrap();
                    }
                    vm.write_bytes(0x6040, &0x1234u16.to_le_bytes()).unwrap();
                }
                x86jit_core::Vcpu::take_machine_counters_for_test();
                let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                let [blocks, ..] = x86jit_core::Vcpu::take_machine_counters_for_test();
                let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                let what = format!("{model:?} DS={selector:#x} budget={budget}");
                assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what}");
                b.cpu.legacy.cr0 |= 0x8000_0000;
                assert_eq!(format!("{:?}", a.cpu), format!("{:?}", b.cpu), "{what}");
                assert_eq!(a.retired_instructions(), b.retired_instructions(), "{what}");
                let mut am = vec![0; 0x1e000];
                let mut bm = vec![0; 0x1e000];
                avm.read_bytes(0, &mut am).unwrap();
                bvm.read_bytes(0, &mut bm).unwrap();
                assert!(am == bm, "{what}: memory");
                if budget == 30 {
                    // The final word at DS:FFFFh is #GP(0) (SDM Vol 3B §23.1.1).
                    assert!(matches!(exit(&ar), Exit::Exception { vector: 13, .. }));
                    assert_eq!(a.retired_instructions(), 14, "{what}");
                    assert_eq!(a.cpu.gpr[7], 0x1234, "{what}");
                    assert!(blocks <= 6, "{what}: {blocks} blocks");
                }
            }
        }
    }
}

#[test]
fn paged_far_transfer_lanes_match_checked_at_every_frame_edge() {
    // SDM Vol 2A CALL, Vol 2B RET (Protected Mode Exceptions): a same-level
    // far CALL/RETF checks the descriptor, the new CS limit and the stack
    // frame; #SS(0) when the frame leaves SS, #PF on a missing stack page
    // (CR2 = the first missing byte), and the accessed bit is written.
    for model in MODELS {
        for cpl in [0u8, 3] {
            for (sp, target, stack_page, accessed) in [
                (0x1000u64, 0u16, true, true),
                (0x1000, 0, true, false),
                (0x0002, 0, true, true),
                (0x1000, 0x1000, true, true),
                (0x1000, 0, false, true),
                (0x1004, 0, true, true),
            ] {
                // INC BX; CALL FAR 20h:target; INC BX; HLT.
                let mut code = vec![0x43, 0x9a];
                code.extend(target.to_le_bytes());
                code.extend([0x20 | cpl, 0, 0x43, 0xf4]);
                for budget in [1, 2, 3, 4, 10] {
                    let (avm, mut a) = seg16(model, &code, cpl, true, false);
                    let (bvm, mut b) = seg16(model, &code, cpl, true, true);
                    for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                        cpu.cpu.gpr[4] = sp;
                        if !accessed {
                            descriptor(vm, 0x3020, 0x4000, 0xfff, 0x9a | (u16::from(cpl) << 5));
                        }
                        if !stack_page {
                            // The frame's page 7000h + FFCh is not present.
                            vm.write_bytes(0x1f01c, &0u32.to_le_bytes()).unwrap();
                        }
                    }
                    let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                    let what = format!(
                        "{model:?} CPL{cpl} SP={sp:#x} IP={target:#x} page={stack_page} A={accessed} budget={budget}"
                    );
                    assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what}");
                    same(&a, &mut b, &avm, &bvm, 0x20000, &what);
                }
            }
        }
    }
}

#[test]
fn in_block_reloads_keep_retained_proofs_warm() {
    // spec.md §17.10: an identical DS/ES reload and a far call/return pair
    // between two code segments lift nothing once warm.
    for model in MODELS {
        for paged in [false, true] {
            // MOV DS,AX; MOV ES,AX; MOV CX,[0]; CALL FAR 20h:0; JMP short -13.
            let code = [
                0x8e, 0xd8, 0x8e, 0xc0, 0x8b, 0x0e, 0, 0, 0x9a, 0, 0, 0x23, 0, 0xeb, 0xf1,
            ];
            let (vm, mut cpu) = seg16(model, &code, 3, paged, false);
            cpu.cpu.gpr[0] = 0x13;
            fixture::steps(&mut cpu, &vm, 70);
            x86jit_core::Vcpu::take_machine_counters_for_test();
            fixture::steps(&mut cpu, &vm, 700);
            let [blocks, reselections, lifts, _] =
                x86jit_core::Vcpu::take_machine_counters_for_test();
            assert_eq!(lifts, 0, "{blocks} {reselections}");
            // Per seven-instruction iteration, five blocks: the loads with
            // the MOV, the far call, INC DX, the RETF and the JMP; one
            // reselection each for the loads, the call and the return.
            assert_eq!(blocks, 500);
            assert_eq!(reselections, 300);
        }
    }
}

/// MOV ES,CX reloads the entry ES (CX = 10h | RPL): its descriptor read and
/// the PUSH/POP that follow leave TLB entries for the GDT page and the stack
/// page, so the next far transfer or load is a lane candidate.
const WARM: [u8; 4] = [0x8e, 0xc1, 0x50, 0x58];

fn warm(cpu: &mut Vcpu, cpl: u8) {
    cpu.cpu.gpr[1] = u64::from(0x10 | u16::from(cpl));
}

/// Run the lane variant and the Checked (transaction) variant at `budget`
/// and compare the stop, the registers, the retirements and the memory.
fn lane_against_checked(
    (avm, a): (&Vm, &mut Vcpu),
    (bvm, b): (&Vm, &mut Vcpu),
    budget: u64,
    what: &str,
) -> x86jit_core::StepRun {
    let ar = a.run_steps(avm, budget, StepOptions::default()).unwrap();
    let br = b.run_steps(bvm, budget, StepOptions::default()).unwrap();
    assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{what}");
    same(a, b, avm, bvm, 0x20000, what);
    ar
}

#[test]
fn retf_lane_matches_checked_at_the_stack_limit_and_with_a_release() {
    // SDM Vol 2B RET pp. 4-569–581 (Protected Mode Exceptions): #SS(0) if the return
    // frame is not within the stack limit; 80386 PRM §17 RET p. 17-145. The
    // same-level return pops CS:IP and then adds imm16 to SP. SS is
    // 18h (limit 1FFFh); the frame is IP, CS at the operand size.
    for model in MODELS {
        for cpl in [0u8, 3] {
            for (sp, release, size, accessed) in [
                (0x1002u16, None, 2u8, true),
                (0x1ffc, None, 2, true),
                (0x1ffe, None, 2, true),
                (0x1ffd, None, 2, true),
                (0x1ff8, None, 4, true),
                (0x1ffa, None, 4, true),
                (0x1ffc, None, 4, true),
                (0x1002, Some(8u16), 2, true),
                (0x1ffc, Some(0x10), 2, true),
                (0x1ff8, Some(0xfff0), 4, true),
                (0x1002, None, 2, false),
                (0x1ffc, Some(2), 2, false),
            ] {
                // MOV ES,CX; PUSH AX; POP AX; [66] RETF [imm16]; INC BX.
                let mut code = WARM.to_vec();
                if size == 4 {
                    code.push(0x66);
                }
                match release {
                    Some(imm) => {
                        code.push(0xca);
                        code.extend(imm.to_le_bytes());
                    }
                    None => code.push(0xcb),
                }
                code.push(0x43);
                for budget in [3, 4, 5, 10] {
                    let (avm, mut a) = seg16(model, &code, cpl, true, false);
                    let (bvm, mut b) = seg16(model, &code, cpl, true, true);
                    for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                        warm(cpu, cpl);
                        cpu.cpu.gpr[4] = u64::from(sp);
                        // The return target 20h:10h is HLT.
                        vm.write_bytes(0x4010, &[0xf4]).unwrap();
                        let frame = 0x7000 + u64::from(sp);
                        let cs = u64::from(0x20 | cpl);
                        let mut bytes = Vec::new();
                        bytes.extend(&0x10u64.to_le_bytes()[..usize::from(size)]);
                        bytes.extend(&cs.to_le_bytes()[..usize::from(size)]);
                        let room = (0x9000 - frame) as usize;
                        vm.write_bytes(frame, &bytes[..bytes.len().min(room)])
                            .unwrap();
                        if !accessed {
                            descriptor(vm, 0x3020, 0x4000, 0xfff, 0x9a | (u16::from(cpl) << 5));
                        }
                    }
                    let what = format!(
                        "{model:?} CPL{cpl} SP={sp:#x} imm={release:?} size={size} A={accessed} budget={budget}"
                    );
                    let run = lane_against_checked((&avm, &mut a), (&bvm, &mut b), budget, &what);
                    let fits = u32::from(sp) + 2 * u32::from(size) <= 0x2000;
                    if budget == 10 && fits {
                        assert_eq!(a.cpu.cs, u16::from(0x20 | cpl), "{what}");
                        // HLT at CPL 3 is #GP(0) (SDM Vol 2A HLT p. 3-439).
                        assert_eq!(a.cpu.rip, 0x10 + u64::from(cpl == 0), "{what}");
                        let popped = u64::from(sp) + 2 * u64::from(size);
                        let sp = (popped + u64::from(release.unwrap_or(0))) & 0xffff;
                        assert_eq!(a.cpu.gpr[4] & 0xffff, sp, "{what}");
                        assert_eq!(a.cpu.legacy.segments[0].attributes & 1, 1, "{what}");
                    } else if budget == 10 {
                        assert!(
                            matches!(exit(&run), Exit::Exception { vector: 12, .. }),
                            "{what}: {run:?}"
                        );
                        assert_eq!(
                            exit(&run).fault_metadata().unwrap().error_code(),
                            ErrorCode::bits(0, 0xffff),
                            "{what}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn far_jmp_lane_with_ptr16_16_and_ptr16_32_matches_checked() {
    // SDM Vol 2A JMP pp. 3-504–512 (Protected Mode Exceptions; Operation, "IF far jump
    // and ... conforming or nonconforming code segment"): the target IP
    // must be within the code limit (#GP(0)); a non-code, not-present or
    // privilege-mismatched selector is #GP(selector) or #NP(selector); the
    // accessed bit is set. 66h selects ptr16:32 in 16-bit code.
    for model in MODELS {
        for cpl in [0u8, 3] {
            let rpl = u16::from(cpl);
            for (wide, selector, ip, accessed) in [
                (false, 0x20 | rpl, 0x10u32, true),
                (true, 0x20 | rpl, 0x10, true),
                (false, 0x20 | rpl, 0x10, false),
                (true, 0x20 | rpl, 0x10, false),
                (false, 0x20 | rpl, 0xfff, true),
                (false, 0x20 | rpl, 0x1000, true),
                (true, 0x20 | rpl, 0x0001_0010, true),
                (false, 0x30 | rpl, 0x10, true),
                (false, 0x38 | rpl, 0x10, true),
                (false, rpl, 0x10, true),
            ] {
                // MOV ES,CX; PUSH AX; POP AX; JMP FAR selector:ip; INC BX.
                let mut code = WARM.to_vec();
                if wide {
                    code.extend([0x66, 0xea]);
                    code.extend(ip.to_le_bytes());
                } else {
                    code.push(0xea);
                    code.extend((ip as u16).to_le_bytes());
                }
                code.extend(selector.to_le_bytes());
                code.push(0x43);
                for budget in [3, 4, 5, 10] {
                    let (avm, mut a) = seg16(model, &code, cpl, true, false);
                    let (bvm, mut b) = seg16(model, &code, cpl, true, true);
                    for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                        warm(cpu, cpl);
                        vm.write_bytes(0x4010, &[0xf4]).unwrap();
                        vm.write_bytes(0x4fff, &[0xf4]).unwrap();
                        if !accessed {
                            descriptor(vm, 0x3020, 0x4000, 0xfff, 0x9a | (rpl << 5));
                        }
                    }
                    let what = format!(
                        "{model:?} CPL{cpl} wide={wide} {selector:#x}:{ip:#x} A={accessed} budget={budget}"
                    );
                    let run = lane_against_checked((&avm, &mut a), (&bvm, &mut b), budget, &what);
                    if budget == 10 && selector & !3 == 0x20 && ip < 0x1000 {
                        assert_eq!(a.cpu.cs, selector, "{what}");
                        assert_eq!(a.cpu.rip, u64::from(ip) + u64::from(cpl == 0), "{what}");
                        assert_eq!(a.cpu.legacy.segments[0].attributes & 1, 1, "{what}");
                    } else if budget == 10 {
                        assert!(matches!(exit(&run), Exit::Exception { .. }), "{what}");
                    }
                }
            }
        }
    }
}

/// Move the GDT to 2004h so that selector FF8h's descriptor occupies
/// 2FFCh–3003h, across the page boundary. 08h, 10h, 18h and 20h keep
/// their images; FF8h holds `attr` with base 4000h, limit FFFh.
fn straddling_gdt(vm: &Vm, cpu: &mut Vcpu, cpl: u8, attr: u16) {
    let dpl = u16::from(cpl) << 5;
    cpu.cpu.legacy.gdtr.base = 0x2004;
    cpu.cpu.legacy.gdtr.limit = 0xfff;
    descriptor(vm, 0x200c, 0x1000, 0x1fff, 0x9b | dpl);
    descriptor(vm, 0x2014, 0xa000, 0xff, 0x93 | dpl);
    descriptor(vm, 0x201c, 0x7000, 0x1fff, 0x93 | dpl);
    descriptor(vm, 0x2024, 0x4000, 0xfff, 0x9b | dpl);
    descriptor(vm, 0x2ffc, 0x4000, 0xfff, attr | dpl);
}

#[test]
fn a_descriptor_across_a_page_boundary_takes_the_transaction_path() {
    // SDM Vol 3A §5.6.1; 80386 PRM §6.5 p. 6-22: the descriptor read is a
    // supervisor access of eight bytes; when its second page is missing the
    // #PF has CR2 = 3000h, the first byte not translated (SDM Vol 3A §5.7
    // pp. 5-37–38), and the error code is P = 0, W/R = 0, U/S = 0. The
    // accessed bit (byte 5) lies on the second page: a missing page there is
    // #PF with CR2 = 3001h on the write if the read succeeded.
    for model in MODELS {
        for cpl in [0u8, 3] {
            let rpl = u16::from(cpl);
            // (program, descriptor attributes): MOV DS,AX; CALL FAR; JMP
            // FAR; RETF to FF8h:10h.
            let programs: [(&[u8], u16); 4] = [
                (&[0x8e, 0xd8], 0x93),
                (&[0x9a, 0x10, 0, 0xf8, 0x0f], 0x9b),
                (&[0xea, 0x10, 0, 0xf8, 0x0f], 0x9b),
                (&[0xcb], 0x9b),
            ];
            for (body, attr) in programs {
                // (A set, second page present, second page writable)
                for (accessed, present, writable) in [
                    (true, true, true),
                    (false, true, true),
                    (true, false, true),
                    (false, false, true),
                    (false, true, false),
                ] {
                    let mut code = WARM.to_vec();
                    code.extend(body);
                    code.push(0x43);
                    for budget in [3, 4, 10] {
                        let (avm, mut a) = seg16(model, &code, cpl, true, false);
                        let (bvm, mut b) = seg16(model, &code, cpl, true, true);
                        for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                            warm(cpu, cpl);
                            cpu.cpu.gpr[0] = u64::from(0xff8 | rpl);
                            cpu.cpu.gpr[4] = 0x1006;
                            if model == CpuModel::I80486Dx {
                                cpu.cpu.legacy.cr0 |= 1 << 16;
                            }
                            straddling_gdt(vm, cpu, cpl, attr & !u16::from(!accessed));
                            vm.write_bytes(0x8006, &[0x10, 0, 0xf8 | cpl, 0x0f])
                                .unwrap();
                            vm.write_bytes(0x4010, &[0xf4]).unwrap();
                            let pte = if !present {
                                0
                            } else if writable {
                                0x3007u32
                            } else {
                                0x3005
                            };
                            vm.write_bytes(0x1f00c, &pte.to_le_bytes()).unwrap();
                        }
                        let what = format!(
                            "{model:?} CPL{cpl} {body:x?} A={accessed} P={present} W={writable} budget={budget}"
                        );
                        let run =
                            lane_against_checked((&avm, &mut a), (&bvm, &mut b), budget, &what);
                        if budget != 10 {
                            continue;
                        }
                        let wp = model == CpuModel::I80486Dx && !writable && !accessed;
                        if !present || wp {
                            assert!(
                                matches!(exit(&run), Exit::Exception { vector: 14, .. }),
                                "{what}: {run:?}"
                            );
                            let (cr2, error) = if present { (0x3001, 3) } else { (0x3000, 0) };
                            assert_eq!(a.cpu.legacy.cr2, cr2, "{what}");
                            assert_eq!(
                                exit(&run).fault_metadata().unwrap().error_code(),
                                ErrorCode::bits(error, 7),
                                "{what}"
                            );
                        } else {
                            let mut byte = [0];
                            avm.read_bytes(0x3001, &mut byte).unwrap();
                            assert_eq!(byte[0] & 1, 1, "{what}: A in memory");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn a_write_protected_gdt_page_faults_the_accessed_bit_from_the_lane() {
    // Intel486 PRM §4.1.3 p. 4-7; SDM Vol 3A §§5.1.3, 5.6.1 pp. 5-5,
    // 5-33–35: with CR0.WP = 1 a supervisor write to a read-only page
    // faults; the I80386 has no WP and writes it. SDM Vol 3A §3.4.5.1:
    // a load of a descriptor whose A bit is clear writes A (byte 5); the
    // GDT access is a supervisor access regardless of CPL (§5.6.1), so the
    // #PF has CR2 = descriptor + 5 and error code P = 1, W/R = 1, U/S = 0
    // (§5.7 Fig. 5-12 pp. 5-37–38). The descriptor page is warm, so the
    // lane proves the read and the fault is raised on the lane path.
    for model in MODELS {
        for cpl in [0u8, 3] {
            let rpl = u16::from(cpl);
            // MOV DS,AX (60h); CALL FAR 20h:10h; JMP FAR 20h:10h; RETF.
            let programs: [(Vec<u8>, u64); 4] = [
                (vec![0x8e, 0xd8], 0x3060),
                (vec![0x9a, 0x10, 0, 0x20 | cpl, 0], 0x3020),
                (vec![0xea, 0x10, 0, 0x20 | cpl, 0], 0x3020),
                (vec![0xcb], 0x3020),
            ];
            for (body, address) in programs {
                let mut code = WARM.to_vec();
                code.extend(&body);
                code.push(0x43);
                let (avm, mut a) = seg16(model, &code, cpl, true, false);
                let (bvm, mut b) = seg16(model, &code, cpl, true, true);
                for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                    warm(cpu, cpl);
                    cpu.cpu.gpr[0] = u64::from(0x60 | rpl);
                    cpu.cpu.gpr[4] = 0x1006;
                    if model == CpuModel::I80486Dx {
                        cpu.cpu.legacy.cr0 |= 1 << 16;
                    }
                    descriptor(vm, 0x3020, 0x4000, 0xfff, 0x9a | (rpl << 5));
                    vm.write_bytes(0x8006, &[0x10, 0, 0x20 | cpl, 0]).unwrap();
                    vm.write_bytes(0x4010, &[0xf4]).unwrap();
                    // The GDT page, read-only.
                    vm.write_bytes(0x1f00c, &0x3005u32.to_le_bytes()).unwrap();
                }
                let what = format!("{model:?} CPL{cpl} {body:x?}");
                let run = lane_against_checked((&avm, &mut a), (&bvm, &mut b), 10, &what);
                if model == CpuModel::I80486Dx {
                    assert!(
                        matches!(exit(&run), Exit::Exception { vector: 14, .. }),
                        "{what}: {run:?}"
                    );
                    assert_eq!(run.steps(), 3, "{what}");
                    assert_eq!(a.cpu.rip, 0x104, "{what}");
                    assert_eq!(u64::from(a.cpu.legacy.cr2), address + 5, "{what}");
                    assert_eq!(
                        exit(&run).fault_metadata().unwrap().error_code(),
                        ErrorCode::bits(3, 7),
                        "{what}"
                    );
                    let mut byte = [0];
                    avm.read_bytes(address + 5, &mut byte).unwrap();
                    assert_eq!(byte[0] & 1, 0, "{what}: A stays clear");
                } else {
                    let mut byte = [0];
                    avm.read_bytes(address + 5, &mut byte).unwrap();
                    assert_eq!(byte[0] & 1, 1, "{what}: A written");
                }
            }
        }
    }
}

#[test]
fn in_guest_delivery_after_an_in_block_reload_matches_checked() {
    // SDM Vol 3A §7.12.1 pp. 7-13–14; 80386 PRM §9.6.1.1 p. 9-7: a fault through a gate
    // to the same level pushes FLAGS, CS, IP and the error code on the
    // current stack and enters the handler. #GP(0) from the 16-byte DS
    // after MOV DS,AX; #PF(0) from DS base 5000h on a missing page after the
    // reload (CR2 = 5000h, SDM Vol 3A §5.7 pp. 5-37–38).
    for model in MODELS {
        for paged in [false, true] {
            for (selector, vector) in [(0x28u16, 13u8), (0x30, 14)] {
                if vector == 14 && !paged {
                    continue;
                }
                // The warm-up (paged: MOV DS,AX runs on the lane); INC BX;
                // MOV DS,AX; MOV CX,[10h or 0]; INC BX.
                let offset = if vector == 13 { 0x10 } else { 0 };
                let mut code = WARM.to_vec();
                code.extend([0x43, 0x8e, 0xd8, 0x8b, 0x0e, offset, 0, 0x43]);
                for budget in [1, 3, 4, 5, 6, 10] {
                    let delivery = ExceptionDelivery::InGuest;
                    let (avm, mut a) = seg16_delivery(model, &code, 0, paged, false, delivery);
                    let (bvm, mut b) = seg16_delivery(model, &code, 0, paged, true, delivery);
                    for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                        warm(cpu, 0);
                        cpu.cpu.gpr[0] = u64::from(selector);
                        cpu.cpu.legacy.idtr.base = 0x2800;
                        cpu.cpu.legacy.idtr.limit = 0x7f;
                        for v in [13u64, 14] {
                            // 286 interrupt gate (type 6, P, DPL 0) to 20h:10h.
                            vm.write_bytes(0x2800 + v * 8, &[0x10, 0, 0x20, 0, 0, 0x86, 0, 0])
                                .unwrap();
                        }
                        vm.write_bytes(0x4010, &[0xf4]).unwrap();
                        if paged {
                            vm.write_bytes(0x1f014, &0u32.to_le_bytes()).unwrap();
                        }
                    }
                    let what = format!("{model:?} PG={paged} #{vector} budget={budget}");
                    lane_against_checked((&avm, &mut a), (&bvm, &mut b), budget, &what);
                    if budget != 10 {
                        continue;
                    }
                    assert_eq!(a.cpu.cs, 0x20, "{what}");
                    assert_eq!(a.cpu.rip, 0x11, "{what}");
                    assert_eq!(a.cpu.ds, selector, "{what}");
                    assert_eq!(a.cpu.gpr[3], 1, "{what}");
                    // Error code 0, IP of the faulting MOV, CS 08h, FLAGS.
                    let mut frame = [0u8; 6];
                    avm.read_bytes(0x8000 - 8, &mut frame).unwrap();
                    assert_eq!(frame, [0, 0, 7, 1, 8, 0], "{what}");
                    if vector == 14 {
                        assert_eq!(a.cpu.legacy.cr2, 0x5000, "{what}");
                    }
                }
            }
        }
    }
}
