//! Internal scalar paging witnesses. Paging design §5 step 1; bytes are supplied
//! directly to the private entry, so these tests make no paged-fetch claim.
use x86jit_core::fault::{
    AddressPrecision, AddressSpace, ErrorCode, FaultAccess, FaultClass, FaultOrigin, Knowledge,
    MemoryEffects,
};
use x86jit_core::{
    CpuModel, Exit, Prot, RegionKind, StepOptions, StepStop, UnsupportedSystemOperation, Vcpu, Vm,
    VmConfig,
};

const MODELS: [CpuModel; 3] = [
    CpuModel::I80386,
    CpuModel::I80486Dx,
    CpuModel::I80486DxCpuid,
];
const BASE: u32 = 0x8040_0000;
const OFFSET: u32 = 0x3000;
const LINEAR: u32 = BASE + OFFSET;
const PD: u64 = 0x1000;
const PT: u32 = 0x3000;
const DATA: u32 = 0x8000;
const NEXT: u32 = 0xb000;
const A: u32 = 0x20;
const D: u32 = 0x40;
const SOFTWARE: u32 = 0xe00;

struct Fixture {
    vm: Vm,
    cpu: Vcpu,
    model: CpuModel,
    code32: bool,
}
impl Fixture {
    fn new(model: CpuModel, code32: bool, cpl: u8) -> Self {
        let mut vm = Vm::new(VmConfig::flat(0x20000));
        for page in (0..0x20000).step_by(0x1000) {
            vm.map(page, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        }
        vm.set_cpu_model(model);
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x400;
        cpu.cpu.legacy.cr0 = 0x8000_0001;
        cpu.cpu.legacy.cr2 = 0xdead_beef;
        cpu.cpu.legacy.cr3 = PD as u32 | 0x18;
        cpu.cpu.legacy.cpl = cpl;
        for (i, segment) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
            segment.base = if i == 0 { 0 } else { BASE };
            segment.limit = u32::MAX;
            segment.attributes = (if i == 0 { 0x9b } else { 0x93 })
                | (u16::from(cpl) << 5)
                | (u16::from(code32) << 14);
            segment.unusable = 0;
        }
        let f = Self {
            vm,
            cpu,
            model,
            code32,
        };
        f.tables(7, 7);
        f
    }
    fn pde(&self) -> u64 {
        PD + 4 * u64::from(LINEAR >> 22)
    }
    fn pte(&self) -> u64 {
        u64::from(PT + 4 * ((LINEAR >> 12) & 0x3ff))
    }
    fn put(&self, at: u64, value: u32) {
        self.vm.write_bytes(at, &value.to_le_bytes()).unwrap();
    }
    fn get(&self, at: u64) -> u32 {
        let mut bytes = [0; 4];
        self.vm.read_bytes(at, &mut bytes).unwrap();
        u32::from_le_bytes(bytes)
    }
    fn tables(&self, pde: u32, pte: u32) {
        self.put(self.pde(), PT | pde);
        self.put(self.pte(), DATA | pte);
    }
    fn code(&self, write: bool, offset: u32) -> Vec<u8> {
        // MOV EAX,moffs32 / MOV moffs32,EAX, with the corresponding 16-bit
        // address/default-operand overrides for a P16 code segment.
        let mut code = Vec::new();
        if !self.code32 {
            code.push(0x66);
        }
        code.push(if write { 0xa3 } else { 0xa1 });
        if self.code32 {
            code.extend(offset.to_le_bytes());
        } else {
            code.extend((offset as u16).to_le_bytes());
        }
        code
    }
    fn execute(&mut self, code: &[u8]) -> Result<(), Exit> {
        x86jit_core::interp::interpret_paged_scalar_for_test(
            code,
            self.model,
            &mut self.cpu.cpu,
            &self.vm.mem,
        )
    }
    fn mov(&mut self, write: bool, offset: u32) -> Result<(), Exit> {
        self.execute(&self.code(write, offset))
    }
    fn pf(&mut self, write: bool, offset: u32, linear: u32, code: u32) -> Exit {
        let before_gpr = self.cpu.cpu.gpr;
        let before_flags = format!("{:?}", self.cpu.cpu.flags);
        let ip = self.cpu.cpu.rip;
        let exit = self.mov(write, offset).unwrap_err();
        assert!(
            matches!(exit, Exit::Exception { vector: 14, .. }),
            "{exit:?}"
        );
        let m = exit.fault_metadata().unwrap();
        assert_eq!(m.class(), FaultClass::Fault);
        assert!(matches!(m.origin(), FaultOrigin::GuestException(_)));
        assert_eq!(m.error_code(), ErrorCode::bits(code, 7));
        assert_eq!(m.error_code().bit(3), None);
        assert_eq!(m.saved_rip(), ip);
        assert_eq!(m.instruction_rip(), Knowledge::Known(ip));
        let Knowledge::Known(address) = m.linear_address() else {
            panic!("missing linear address")
        };
        assert_eq!(address.value(), u64::from(linear));
        assert_eq!(address.space(), AddressSpace::Linear);
        assert_eq!(address.precision(), AddressPrecision::ExactByte);
        assert_eq!(
            m.access(),
            Knowledge::Known(if write {
                FaultAccess::Write
            } else {
                FaultAccess::Read
            })
        );
        assert_ne!(m.progress().ram(), MemoryEffects::None);
        assert_eq!(self.cpu.cpu.legacy.cr2, linear);
        assert_eq!(self.cpu.cpu.rip, ip);
        assert_eq!(self.cpu.cpu.gpr, before_gpr);
        assert_eq!(format!("{:?}", self.cpu.cpu.flags), before_flags);
        exit
    }
}

#[test]
fn present_permissions_all_pde_pte_combinations() {
    // 80386 PRM §§6.4.1–6.4.2 pp. 6-20–22 Table 6-5;
    // Intel486 PRM §§6.8.1–6.8.2 pp. 6-23–25 Table 6-4, WP=0.
    for model in MODELS {
        for code32 in [false, true] {
            for cpl in 0..4 {
                let mut f = Fixture::new(model, code32, cpl);
                for pde in [1, 3, 5, 7] {
                    for pte in [1, 3, 5, 7] {
                        for preset in [0, A | D | SOFTWARE] {
                            for write in [false, true] {
                                f.cpu.cpu.rip = 0x400;
                                f.cpu.cpu.gpr[0] = 0x1234_5678;
                                f.tables(pde | preset, pte | preset);
                                f.put(u64::from(DATA), 0x7654_3210);
                                let rights = pde & pte;
                                let allowed =
                                    cpl != 3 || (rights & 4 != 0 && (!write || rights & 2 != 0));
                                if allowed {
                                    f.mov(write, OFFSET).unwrap();
                                    assert_eq!(
                                        f.get(u64::from(DATA)),
                                        if write { 0x1234_5678 } else { 0x7654_3210 }
                                    );
                                    assert_eq!(
                                        f.cpu.cpu.gpr[0],
                                        if write { 0x1234_5678 } else { 0x7654_3210 }
                                    );
                                    assert_eq!(f.cpu.cpu.legacy.cr2, 0xdead_beef);
                                } else {
                                    f.pf(write, OFFSET, LINEAR, 5 | (u32::from(write) << 1));
                                    assert_eq!(f.get(u64::from(DATA)), 0x7654_3210);
                                }
                                // 386 PRM §5.2.4.3 p. 5-10; 486 PRM §5.3.4.3
                                // pp. 5-21–22 and §9.9.14 p. 9-21: A on both;
                                // D only on an allowed write; preserve all other bits.
                                assert_eq!(f.get(f.pde()), PT | pde | preset | A);
                                assert_eq!(
                                    f.get(f.pte()),
                                    DATA | pte | preset | A | if allowed && write { D } else { 0 }
                                );
                                f.cpu.cpu.legacy.cr2 = 0xdead_beef;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn not_present_levels_and_low_three_error_bits() {
    // 80386 PRM §9.8.14 Fig. 9-8 pp. 9-17–18; Intel486 PRM
    // §9.9.14 pp. 9-21–23: not-present overrides permission bits.
    for model in MODELS {
        for code32 in [false, true] {
            for cpl in [0, 3] {
                for write in [false, true] {
                    for directory in [false, true] {
                        for absent in [0, 0xffe] {
                            let mut f = Fixture::new(model, code32, cpl);
                            f.tables(
                                if directory { absent } else { 1 },
                                if directory { 7 } else { absent },
                            );
                            let pd = f.get(f.pde());
                            let pt = f.get(f.pte());
                            f.pf(
                                write,
                                OFFSET,
                                LINEAR,
                                (u32::from(write) << 1) | if cpl == 3 { 4 } else { 0 },
                            );
                            assert_eq!(f.get(f.pde()), pd | if directory { 0 } else { A });
                            assert_eq!(f.get(f.pte()), pt);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn second_page_fault_preflights_load_and_store_before_operand_commit() {
    // 80386 PRM §9.8.14 pp. 9-17–19: CR2 names the failing linear
    // address, IP names the restarting instruction; paging design §3.4
    // strengthens scalar RAM atomicity while retaining page-walk side effects.
    for model in MODELS {
        for code32 in [false, true] {
            for write in [false, true] {
                for protection in [false, true] {
                    let mut f = Fixture::new(model, code32, 3);
                    f.put(f.pte() + 4, NEXT | if protection { 1 } else { SOFTWARE });
                    f.put(u64::from(DATA + 0xffc), 0x4433_2211);
                    f.put(u64::from(NEXT), 0x8877_6655);
                    f.pf(
                        write,
                        OFFSET + 0xffe,
                        LINEAR + 0x1000,
                        4 | (u32::from(write) << 1) | u32::from(protection),
                    );
                    assert_eq!(f.get(u64::from(DATA + 0xffc)), 0x4433_2211);
                    assert_eq!(f.get(u64::from(NEXT)), 0x8877_6655);
                    assert_eq!(f.get(f.pte()), DATA | 7 | A); // no premature D
                    assert_eq!(
                        f.get(f.pte() + 4),
                        NEXT | if protection { 1 | A } else { SOFTWARE }
                    );
                    // Repair and restart the very same instruction.
                    f.put(f.pte() + 4, NEXT | 7);
                    f.cpu.cpu.gpr[0] = 0xaabb_ccdd;
                    f.mov(write, OFFSET + 0xffe).unwrap();
                    if write {
                        assert_eq!(f.get(u64::from(DATA + 0xffc)), 0xccdd_2211);
                        assert_eq!(f.get(u64::from(NEXT)), 0x8877_aabb);
                        assert_eq!(f.get(f.pte()) & D, D);
                        assert_eq!(f.get(f.pte() + 4) & D, D);
                    } else {
                        assert_eq!(f.cpu.cpu.gpr[0], 0x6655_4433);
                    }
                }
            }
        }
    }
}

#[test]
fn segment_fault_precedes_walk_and_retains_cr2() {
    // 80386 PRM §§6.4.3–6.5 pp. 6-21–22; Intel486 PRM §6.9 p. 6-25.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 0);
            f.cpu.cpu.legacy.segments[1].limit = OFFSET - 1; // DS
            f.tables(0, 0);
            let exit = f.mov(false, OFFSET).unwrap_err();
            assert!(
                matches!(exit, Exit::Exception { vector: 13, .. }),
                "{exit:?}"
            );
            assert_eq!(f.cpu.cpu.legacy.cr2, 0xdead_beef);
            assert_eq!(f.get(f.pde()), PT);
        }
    }
}

#[test]
fn rmw_load_checks_write_rights_and_rollback_preserves_new_cr2() {
    // 80386 PRM §6.4.2 p. 6-21 and §9.8.14 pp. 9-17–19:
    // a memory ADD is a write access even when adding zero.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 3);
            f.tables(7, 5);
            f.cpu.cpu.gpr[0] = 0;
            let code: &[u8] = if code32 {
                &[0x01, 0x05, 0, 0x30, 0, 0]
            } else {
                &[0x66, 0x01, 0x06, 0, 0x30]
            };
            let exit = f.execute(code).unwrap_err();
            assert_eq!(
                exit.fault_metadata().unwrap().error_code(),
                ErrorCode::bits(7, 7)
            );
            assert_eq!(f.cpu.cpu.legacy.cr2, LINEAR);
            assert_eq!(f.cpu.cpu.rip, 0x400);
            assert_eq!(f.get(f.pte()), DATA | 5 | A);
        }
    }
}

#[test]
fn physical_table_failure_retains_cr2_and_never_becomes_pf() {
    // 80386 PRM §5.2.4.2 p. 5-9: CR3 has no present bit; a missing
    // physical table is an engine bus exit (paging design §1.3).
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.cpu.cpu.legacy.cr3 = 0x40000;
        let exit = f.mov(false, OFFSET).unwrap_err();
        assert!(matches!(exit, Exit::UnmappedMemory { .. }), "{exit:?}");
        let m = exit.fault_metadata().unwrap();
        assert!(
            matches!(m.linear_address(), Knowledge::Known(a) if a.value() == u64::from(LINEAR))
        );
        assert_eq!(
            m.physical_address(),
            Knowledge::Known(0x40000 + 4 * u64::from(LINEAR >> 22))
        );
        assert_eq!(f.cpu.cpu.legacy.cr2, 0xdead_beef);
    }
}

#[test]
fn public_paged_entry_and_cr0_enable_stay_refused() {
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 0);
            let before = format!("{:?}", f.cpu.cpu);
            let run = f.cpu.run_steps(&f.vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnsupportedSystemContext {
                        operation: UnsupportedSystemOperation::LegacyEntryFacility,
                        ..
                    })
                ),
                "{:?}",
                run.stop()
            );
            assert_eq!(run.steps(), 0);
            assert_eq!(format!("{:?}", f.cpu.cpu), before);
            f.cpu.cpu.legacy.cr0 = 1;
            f.cpu.cpu.gpr[0] = 0x8000_0001;
            f.vm.write_bytes(0x400, &[0x0f, 0x22, 0xc0]).unwrap(); // MOV CR0,EAX
            let run = f.cpu.run_steps(&f.vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(
                    run.stop(),
                    StepStop::Exit(Exit::UnsupportedSystemContext { .. })
                ),
                "{:?}",
                run.stop()
            );
            assert_eq!(f.cpu.cpu.legacy.cr0, 1);
            assert_eq!(f.cpu.cpu.rip, 0x400);
        }
    }
}

#[test]
fn physical_table_rom_and_device_are_typed_refusals() {
    // Engine RAM-table policy, paging design §§1.3, 3.3 (not an ISA restriction).
    for model in MODELS {
        for directory in [false, true] {
            for kind in [RegionKind::Rom, RegionKind::Trap] {
                let mut f = Fixture::new(model, true, 0);
                let page = if directory { PD } else { u64::from(PT) };
                f.vm.unmap(page, 0x1000).unwrap();
                f.vm.map(page, 0x1000, Prot::RWX, kind).unwrap();
                let exit = f.mov(false, OFFSET).unwrap_err();
                assert!(
                    matches!(
                        exit,
                        Exit::UnsupportedSystemContext {
                            operation: UnsupportedSystemOperation::PagingTableStorage,
                            ..
                        }
                    ),
                    "{exit:?}"
                );
                assert_eq!(f.cpu.cpu.legacy.cr2, 0xdead_beef);
            }
        }
    }
}

#[test]
fn span_plan_crosses_directory_and_linear_wrap_before_writing() {
    // 80386 PRM §5.2 Figs. 5-8–10 pp. 5-7–9; SDM Vol 3A §3.4
    // p. 3-10: 32-bit linear arithmetic. Paging design §3.4 retains the
    // plan when an operand aliases a table used by a later fragment.
    for model in MODELS {
        for code32 in [false, true] {
            for wrap in [false, true] {
                let mut f = Fixture::new(model, code32, 0);
                let first = if wrap { 0xffff_fffeu32 } else { 0xffbf_fffeu32 };
                let second = first.wrapping_add(2);
                f.cpu.cpu.legacy.segments[1].base = first.wrapping_sub(OFFSET + 0xffe);
                let next_table = 0x5000u32;
                f.put(PD + 4 * u64::from(first >> 22), PT | 7);
                f.put(PD + 4 * u64::from(second >> 22), next_table | 7);
                // Non-wrap case: the first write changes the high half of
                // the next PDE; re-walking after this commit would fail.
                let first_frame = if wrap { DATA } else { PD as u32 };
                f.put(u64::from(PT + 4 * ((first >> 12) & 0x3ff)), first_frame | 7);
                f.put(
                    u64::from(next_table + 4 * ((second >> 12) & 0x3ff)),
                    NEXT | 7,
                );
                f.cpu.cpu.gpr[0] = 0x1122_beef;
                f.mov(true, OFFSET + 0xffe).unwrap();
                assert_eq!(f.get(u64::from(first_frame + 0xffc)) >> 16, 0xbeef);
                assert_eq!(f.get(u64::from(NEXT)) & 0xffff, 0x1122);
                assert_eq!(f.cpu.cpu.legacy.cr2, 0xdead_beef);
            }
        }
    }
}

#[test]
fn scalar_word_and_byte_operands_translate() {
    // 80386 PRM §5.2 pp. 5-7–10: paging translates byte addresses
    // independently of the operand size selected by the instruction.
    for model in MODELS {
        for code32 in [false, true] {
            for byte in [false, true] {
                let mut f = Fixture::new(model, code32, 0);
                f.put(u64::from(DATA), 0x1234_abcd);
                let mut read = Vec::new();
                if !byte && code32 {
                    read.push(0x66);
                }
                read.push(if byte { 0xa0 } else { 0xa1 });
                if code32 {
                    read.extend(OFFSET.to_le_bytes());
                } else {
                    read.extend((OFFSET as u16).to_le_bytes());
                }
                f.cpu.cpu.gpr[0] = 0x7654_0000;
                f.execute(&read).unwrap();
                assert_eq!(
                    f.cpu.cpu.gpr[0],
                    if byte { 0x7654_00cd } else { 0x7654_abcd }
                );
                let opcode = usize::from(!byte && code32);
                read[opcode] += 2;
                f.cpu.cpu.gpr[0] = 0x8899;
                f.execute(&read).unwrap();
                assert_eq!(
                    f.get(u64::from(DATA)),
                    if byte { 0x1234_ab99 } else { 0x1234_8899 }
                );
                assert_eq!(f.get(f.pte()) & (A | D), A | D);
            }
        }
    }
}

#[test]
fn missing_second_page_precedes_first_page_device_handoff() {
    // Paging design §3.4: preflight the complete scalar span before any
    // operand MMIO. 80386 PRM §9.8.14 pp. 9-17–18 supplies the #PF/CR2 rule.
    for model in MODELS {
        for code32 in [false, true] {
            for write in [false, true] {
                let mut f = Fixture::new(model, code32, 3);
                f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
                f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Trap)
                    .unwrap();
                f.put(f.pte() + 4, 0);
                f.pf(
                    write,
                    OFFSET + 0xffe,
                    LINEAR + 0x1000,
                    if write { 6 } else { 4 },
                );
                assert_eq!(f.get(f.pte()), DATA | 7 | A);
            }
        }
    }
}

#[test]
fn split_device_operands_are_refused_before_reads_writes_or_rmw() {
    // Engine policy, paging design §§3.4, 5 step 5: no split device handoff
    // without a retained operand snapshot and translated continuation plan.
    for model in MODELS {
        for code32 in [false, true] {
            for devices in [0, 1, 2, 3] {
                for operation in 0..3 {
                    let mut f = Fixture::new(model, code32, 0);
                    f.put(f.pte() + 4, NEXT | 7);
                    f.put(u64::from(DATA + 0xffc), 0x1234_5678);
                    f.put(u64::from(NEXT), 0x8765_4321);
                    for (bit, page) in [(1, DATA), (2, NEXT)] {
                        if devices & bit != 0 {
                            f.vm.unmap(u64::from(page), 0x1000).unwrap();
                            f.vm.map(u64::from(page), 0x1000, Prot::RWX, RegionKind::Trap)
                                .unwrap();
                        }
                    }
                    f.cpu.cpu.gpr[0] = 0xaabb_ccdd;
                    let code = if operation == 2 {
                        // ADD dword [OFFSET+0xffe],1: RAM-first used to commit
                        // its low half before the MMIO write and add twice on retry.
                        let mut code = Vec::new();
                        if !code32 {
                            code.push(0x66);
                        }
                        code.extend([0x83, if code32 { 0x05 } else { 0x06 }]);
                        if code32 {
                            code.extend((OFFSET + 0xffe).to_le_bytes());
                        } else {
                            code.extend(((OFFSET + 0xffe) as u16).to_le_bytes());
                        }
                        code.push(1);
                        code
                    } else {
                        f.code(operation == 1, OFFSET + 0xffe)
                    };
                    if devices == 0 {
                        f.execute(&code).unwrap();
                        assert_eq!(
                            f.get(u64::from(DATA + 0xffc)),
                            [0x1234_5678, 0xccdd_5678, 0x1235_5678][operation]
                        );
                        assert_eq!(
                            f.get(u64::from(NEXT)),
                            if operation == 1 {
                                0x8765_aabb
                            } else {
                                0x8765_4321
                            }
                        );
                        if operation == 0 {
                            assert_eq!(f.cpu.cpu.gpr[0], 0x4321_1234);
                        }
                        continue;
                    }
                    let before = format!("{:?}", f.cpu.cpu);
                    for _ in 0..2 {
                        let exit = f.execute(&code).unwrap_err();
                        assert!(
                            matches!(
                                exit,
                                Exit::UnsupportedSystemContext {
                                    operation: UnsupportedSystemOperation::PagingSplitDevice,
                                    ..
                                }
                            ),
                            "{exit:?}"
                        );
                        assert_eq!(format!("{:?}", f.cpu.cpu), before);
                        if devices & 1 == 0 {
                            assert_eq!(f.get(u64::from(DATA + 0xffc)), 0x1234_5678);
                        }
                        if devices & 2 == 0 {
                            assert_eq!(f.get(u64::from(NEXT)), 0x8765_4321);
                        }
                        assert_eq!(f.get(f.pte()), DATA | 7 | A);
                        assert_eq!(f.get(f.pte() + 4), NEXT | 7 | A);
                    }
                }
            }
        }
    }
}

#[test]
fn device_region_split_within_one_page_is_also_refused() {
    // Paging design §5 step 5: physical-region and page splits both need the
    // same continuation, even when only one page translation is involved.
    for model in MODELS {
        for device_first in [false, true] {
            for write in [false, true] {
                let mut f = Fixture::new(model, true, 0);
                f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
                for (offset, size, device) in [(0, 4, device_first), (4, 0xffc, !device_first)] {
                    f.vm.map(
                        u64::from(DATA + offset),
                        size,
                        Prot::RWX,
                        if device {
                            RegionKind::Trap
                        } else {
                            RegionKind::Ram
                        },
                    )
                    .unwrap();
                }
                let ram = u64::from(DATA + if device_first { 4 } else { 0 });
                f.put(ram, 0x1234_5678);
                let before = format!("{:?}", f.cpu.cpu);
                let exit = f.mov(write, OFFSET + 2).unwrap_err();
                assert!(
                    matches!(
                        exit,
                        Exit::UnsupportedSystemContext {
                            operation: UnsupportedSystemOperation::PagingSplitDevice,
                            ..
                        }
                    ),
                    "{exit:?}"
                );
                assert_eq!(format!("{:?}", f.cpu.cpu), before);
                assert_eq!(f.get(ram), 0x1234_5678);
                assert_eq!(f.get(f.pte()), DATA | 7 | A);
            }
        }
    }
}

#[test]
fn memory_push_pop_are_refused_before_either_operand_translation() {
    // Step-1 policy, paging design §§3.4, 5: multiple operands require one
    // instruction-wide preflight before any MMIO or operand RAM transfer.
    for model in MODELS {
        for code32 in [false, true] {
            for pop in [false, true] {
                for absent_stack in [false, true] {
                    for mmio in [false, true] {
                        let mut f = Fixture::new(model, code32, 0);
                        let stack_offset = OFFSET + 0x1000;
                        f.cpu.cpu.gpr[4] = u64::from(stack_offset + if pop { 0 } else { 4 });
                        f.put(f.pte() + 4, NEXT | if absent_stack { 0 } else { 7 });
                        f.put(f.pte(), DATA | if absent_stack { 7 } else { 0 });
                        f.put(u64::from(DATA), 0x1234_5678);
                        f.put(u64::from(NEXT), 0x8765_4321);
                        let present_page = if absent_stack { DATA } else { NEXT };
                        if mmio {
                            f.vm.unmap(u64::from(present_page), 0x1000).unwrap();
                            f.vm.map(u64::from(present_page), 0x1000, Prot::RWX, RegionKind::Trap)
                                .unwrap();
                        }
                        let mut code = Vec::new();
                        if !code32 {
                            code.push(0x66);
                        }
                        code.extend([
                            if pop { 0x8f } else { 0xff },
                            (if pop { 0 } else { 0x30 }) | if code32 { 0x05 } else { 0x06 },
                        ]);
                        if code32 {
                            code.extend(OFFSET.to_le_bytes());
                        } else {
                            code.extend((OFFSET as u16).to_le_bytes());
                        }
                        let before = format!("{:?}", f.cpu.cpu);
                        let directory = f.get(f.pde());
                        let operand = f.get(f.pte());
                        let stack = f.get(f.pte() + 4);
                        let exit = f.execute(&code).unwrap_err();
                        assert!(
                            matches!(
                                exit,
                                Exit::UnsupportedSystemContext {
                                    operation: UnsupportedSystemOperation::LegacyInstruction,
                                    ..
                                }
                            ),
                            "{exit:?}"
                        );
                        assert_eq!(format!("{:?}", f.cpu.cpu), before);
                        assert_eq!(f.get(f.pde()), directory);
                        assert_eq!(f.get(f.pte()), operand);
                        assert_eq!(f.get(f.pte() + 4), stack);
                        for (page, value) in [(DATA, 0x1234_5678), (NEXT, 0x8765_4321)] {
                            if !mmio || page != present_page {
                                assert_eq!(f.get(u64::from(page)), value);
                            }
                        }
                    }
                }
            }
        }
    }
}
