//! Paging witnesses (paging design §5). Focused scalar/proof tests retain the
//! private entry; public witnesses exercise normal Machine dispatch and retries.
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
        Self::with_vm(Vm::new(VmConfig::flat(0x20000)), model, code32, cpl)
    }
    fn with_vm(vm: Vm, model: CpuModel, code32: bool, cpl: u8) -> Self {
        Self::with_delivery(
            vm,
            model,
            code32,
            cpl,
            x86jit_core::ExceptionDelivery::Outward,
        )
    }
    fn with_delivery(
        mut vm: Vm,
        model: CpuModel,
        code32: bool,
        cpl: u8,
        delivery: x86jit_core::ExceptionDelivery,
    ) -> Self {
        for page in (0..vm.mem.size()).step_by(0x1000) {
            vm.map(page, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
        }
        vm.set_cpu_model(model);
        vm.set_exception_delivery(delivery);
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
                            assert_eq!(f.get(f.pde()), pd);
                            assert_eq!(f.get(f.pte()), pt);
                        }
                    }
                }
            }
        }
    }
}

// ASSUMED (paging design §3.3): 80386 PRM §5.2.4.3 p. 5-10
// sets A before a page access, without specifying failed walks. A public CPU
// test suite falsified the previous PDE.A-on-absent-PTE choice.
#[test]
fn absent_pte_preserves_both_entries_for_read_write_and_fetch() {
    let mut changed = Vec::new();
    for model in MODELS {
        for code32 in [false, true] {
            for access in 0..3 {
                let mut f = Fixture::new(model, code32, 0);
                f.tables(3, 2);
                if access == 2 {
                    f.cpu.cpu.legacy.segments[0].base = BASE;
                    f.fetch_pf(u64::from(OFFSET), LINEAR, 0);
                } else {
                    f.pf(access == 1, OFFSET, LINEAR, if access == 1 { 2 } else { 0 });
                }
                if f.get(f.pde()) != PT | 3 || f.get(f.pte()) != DATA | 2 {
                    changed.push((model, code32, access, f.get(f.pde()), f.get(f.pte())));
                }
            }
        }
    }
    assert!(changed.is_empty(), "page-table changes: {changed:?}");
}

#[test]
fn present_protection_fault_sets_both_accessed_bits() {
    // Intel486 PRM §9.9.14 p. 9-21 requires both A bits. ASSUMED on
    // the 386 (80386 PRM §5.2.4.3 p. 5-10; paging design §3.3):
    // retain the same A-on-protection-fault policy, with D unchanged.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 3);
            f.tables(3, 3);
            f.pf(false, OFFSET, LINEAR, 5);
            assert_eq!(f.get(f.pde()), PT | 3 | A);
            assert_eq!(f.get(f.pte()), DATA | 3 | A);
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
fn public_paged_entry_and_guest_cr0_enable() {
    // 80386 PRM §5.2 pp. 5-7–11; Intel486 PRM §5.3.1 p. 5-18.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 0);
            f.code_mapping(0, CODE_FRAME);
            f.vm.write_bytes(0x400, &[0x0f, 0x22, 0xc0]).unwrap();
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x403, &[0xb0, 0x73])
                .unwrap();
            f.cpu.cpu.legacy.cr0 = 1;
            f.cpu.cpu.gpr[0] = 0x8000_0001;
            let run = f.cpu.run_steps(&f.vm, 2, StepOptions::default()).unwrap();
            assert!(
                matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                "{run:?}"
            );
            assert_eq!(run.steps(), 2);
            assert_eq!(f.cpu.cpu.legacy.cr0 & 0x8000_0001, 0x8000_0001);
            assert_eq!(f.cpu.cpu.gpr[0] & 0xff, 0x73);
            assert_eq!(f.cpu.cpu.rip, 0x405);
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
fn step4_memory_push_pop_preflight_both_operands() {
    // Paging design §3.4; 80386 PRM §9.8.14 pp. 9-17–18: either
    // operand can fault, before any stack/explicit-operand transfer.
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
                        let before = f.cpu.cpu.gpr;
                        let exit = f.execute(&code).unwrap_err();
                        assert!(
                            matches!(exit, Exit::Exception { vector: 14, .. }),
                            "{exit:?}"
                        );
                        assert_eq!(f.cpu.cpu.gpr, before);
                        assert_eq!(
                            f.cpu.cpu.legacy.cr2,
                            LINEAR + if absent_stack { 0x1000 } else { 0 }
                        );
                        assert_eq!(
                            exit.fault_metadata().unwrap().error_code(),
                            ErrorCode::bits(u32::from(pop != absent_stack) * 2, 7)
                        );
                        assert_eq!(f.get(f.pte()) & D, 0);
                        assert_eq!(f.get(f.pte() + 4) & D, 0);
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

const CODE_PT: u32 = 0x5000;
const CODE_FRAME: u32 = 0xc000;
const OTHER_FRAME: u32 = 0x10000;
const OTHER_PD: u32 = 0x2000;
const OTHER_PT: u32 = 0x6000;

impl Fixture {
    fn map_code(&self, directory: u32, table: u32, linear: u32, frame: u32, flags: u32) {
        self.put(u64::from(directory + 4 * (linear >> 22)), table | 7 | A);
        self.put(
            u64::from(table + 4 * ((linear >> 12) & 0x3ff)),
            frame | flags,
        );
    }
    fn code_mapping(&self, linear: u32, frame: u32) {
        self.map_code(PD as u32, CODE_PT, linear, frame, 7 | A);
    }
    fn fetch_at(&mut self, ip: u64) -> Result<(), Exit> {
        self.cpu.cpu.rip = ip;
        self.cpu.interpret_paged_for_test(&self.vm)
    }
    fn fetch_pf(&mut self, ip: u64, linear: u32, code: u32) {
        let registers = self.cpu.cpu.gpr;
        let flags = format!("{:?}", self.cpu.cpu.flags);
        let exit = self.fetch_at(ip).unwrap_err();
        assert!(
            matches!(exit, Exit::Exception { vector: 14, .. }),
            "{exit:?}"
        );
        let m = exit.fault_metadata().unwrap();
        assert_eq!(m.class(), FaultClass::Fault);
        assert_eq!(m.saved_rip(), ip);
        assert_eq!(m.instruction_rip(), Knowledge::Known(ip));
        assert_eq!(m.error_code(), ErrorCode::bits(code, 7));
        assert_eq!(m.error_code().bit(4), None); // no historical I/D bit
        assert_eq!(m.access(), Knowledge::Known(FaultAccess::Execute));
        let Knowledge::Known(address) = m.linear_address() else {
            panic!("linear evidence")
        };
        assert_eq!(address.value(), u64::from(linear));
        assert_eq!(self.cpu.cpu.legacy.cr2, linear);
        assert_eq!(self.cpu.cpu.rip, ip);
        assert_eq!(self.cpu.cpu.gpr, registers);
        assert_eq!(format!("{:?}", self.cpu.cpu.flags), flags);
    }
}

#[test]
fn paged_fetch_nonidentity_mapping_and_historical_fault_bits() {
    // 80386 PRM §5.2 pp. 5-7–10, §9.8.14 Fig. 9-8 pp. 9-17–18;
    // Intel486 PRM §9.9.14 Fig. 9-7 pp. 9-21–22: fetch reads, no I/D bit.
    for model in MODELS {
        for code32 in [false, true] {
            for cpl in [0, 3] {
                let mut f = Fixture::new(model, code32, cpl);
                f.code_mapping(0, CODE_FRAME);
                f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0xb0, 0x5a])
                    .unwrap();
                f.vm.write_bytes(0x400, &[0xb0, 0x11]).unwrap();
                f.fetch_at(0x400).unwrap();
                assert_eq!(f.cpu.cpu.gpr[0], 0x5a);
                assert_eq!(f.cpu.cpu.rip, 0x402);
                assert_eq!(f.get(u64::from(CODE_PT)), CODE_FRAME | 7 | A);
                assert_eq!(f.cpu.cpu.legacy.cr2, 0xdead_beef);
                // SDM Vol 3A §5.10.4.2: invalidate after table edits
                // before requiring the new permissions/presence.
                f.put(u64::from(CODE_PT), CODE_FRAME | 3 | A);
                f.cpu.invalidate_paging();
                if cpl == 3 {
                    f.fetch_pf(0x400, 0x400, 5);
                } else {
                    f.fetch_at(0x400).unwrap();
                }
                f.put(u64::from(CODE_PT), 0);
                f.cpu.invalidate_paging();
                f.fetch_pf(0x400, 0x400, if cpl == 3 { 4 } else { 0 });
                f.put(PD, 0);
                f.cpu.invalidate_paging();
                f.fetch_pf(0x400, 0x400, if cpl == 3 { 4 } else { 0 });
            }
        }
    }
}

#[test]
fn paged_fetch_physical_failures_report_actual_walk_writes() {
    // 80386 PRM §5.2.4.3 p. 5-10; Intel486 PRM §5.3.4.3 pp. 5-21–22:
    // A updates precede the physical fetch; progress must retain their effects.
    for model in MODELS {
        for code32 in [false, true] {
            for failure in 0..5 {
                for pde_a in [0, A] {
                    for pte_a in [0, A] {
                        let mut f = Fixture::new(model, code32, 0);
                        f.put(PD, CODE_PT | 7 | pde_a);
                        f.put(u64::from(CODE_PT), CODE_FRAME | 7 | pte_a);
                        let origin = match failure {
                            0 => {
                                f.vm.unmap(u64::from(CODE_FRAME), 0x1000).unwrap();
                                f.vm.map(
                                    u64::from(CODE_FRAME),
                                    0x1000,
                                    Prot::RWX,
                                    RegionKind::Trap,
                                )
                                .unwrap();
                                FaultOrigin::TrapFetch
                            }
                            1 => {
                                f.vm.unmap(u64::from(CODE_FRAME), 0x1000).unwrap();
                                FaultOrigin::GuestMap
                            }
                            2 | 3 => {
                                f.vm.set_fetch_policy(Box::new(move |_| {
                                    if failure == 2 {
                                        x86jit_core::FetchVerdict::DenyNotPresent
                                    } else {
                                        x86jit_core::FetchVerdict::DenyProtection
                                    }
                                }));
                                FaultOrigin::GuestMap
                            }
                            _ => {
                                f.vm.mem.set_host_fault_recovery(true);
                                FaultOrigin::HostAccess
                            }
                        };
                        let exit = if failure == 4 {
                            let host = f.vm.mem.host_base() as usize + CODE_FRAME as usize + 0x400;
                            x86jit_core::with_copy_out_fault(host..host + 1, usize::MAX, || {
                                f.fetch_at(0x400)
                            })
                        } else {
                            f.fetch_at(0x400)
                        }
                        .unwrap_err();
                        assert!(matches!(exit, Exit::UnmappedMemory { .. }), "{exit:?}");
                        let m = exit.fault_metadata().unwrap();
                        assert_eq!(m.origin(), origin);
                        assert_eq!(m.access(), Knowledge::Known(FaultAccess::Execute));
                        let wrote = pde_a == 0 || pte_a == 0;
                        assert_eq!(
                            m.progress().ram(),
                            if wrote {
                                MemoryEffects::PossiblyPartial
                            } else {
                                MemoryEffects::None
                            }
                        );
                        assert_eq!(m.progress().device(), MemoryEffects::None);
                        assert_eq!(
                            m.progress().had_partial_progress(),
                            if wrote {
                                Knowledge::Unknown
                            } else {
                                Knowledge::Known(false)
                            }
                        );
                        assert_eq!(f.get(PD), CODE_PT | 7 | A);
                        assert_eq!(f.get(u64::from(CODE_PT)), CODE_FRAME | 7 | A);
                        assert_eq!(f.cpu.cpu.rip, 0x400);
                        assert_eq!(f.cpu.cpu.legacy.cr2, 0xdead_beef);
                    }
                }
            }
        }
    }
}

#[test]
fn paged_fetch_does_not_speculate_past_one_byte_instruction() {
    // 80386 PRM §9.8.14 pp. 9-17–18: a required not-present translation
    // faults; paging design §3.5 requires incremental, one-instruction fetch.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 0);
            f.map_code(PD as u32, CODE_PT, 0, CODE_FRAME, 7);
            f.put(PD, CODE_PT | 7);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0xfff, &[0x40])
                .unwrap();
            f.cpu.cpu.gpr[0] = 7;
            f.fetch_at(0xfff).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 8);
            assert_eq!(f.cpu.cpu.rip, 0x1000);
            assert_eq!(f.get(PD), CODE_PT | 7 | A);
            assert_eq!(f.get(u64::from(CODE_PT)), CODE_FRAME | 7 | A);
            assert_eq!(f.get(u64::from(CODE_PT + 4)), 0);
            assert_eq!(f.cpu.cpu.legacy.cr2, 0xdead_beef);
        }
    }
}

#[test]
fn paged_fetch_cross_page_fault_restarts_before_execution() {
    // 80386 PRM §9.8.14 pp. 9-17–19; Intel486 PRM §9.9.14 pp. 9-21–23:
    // CR2 names the failing linear byte, saved IP the instruction start.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 0);
            f.code_mapping(0, CODE_FRAME);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0xfff, &[0xb0])
                .unwrap();
            f.cpu.cpu.gpr[0] = 0x1122;
            f.fetch_pf(0xfff, 0x1000, 0);
            f.code_mapping(0x1000, OTHER_FRAME);
            f.vm.write_bytes(u64::from(OTHER_FRAME), &[0x5a]).unwrap();
            f.fetch_at(0xfff).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x115a);
            assert_eq!(f.cpu.cpu.rip, 0x1001);
        }
    }
}

#[test]
fn paged_fetch_two_directories_and_equal_cr3_reload() {
    // 80386 PRM §5.2.3 p. 5-9, §5.2.5 p. 5-11;
    // Intel486 PRM §5.3.5 p. 5-22: CR3 selects the directory; reload flushes.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 0);
            f.cpu.cpu.legacy.cr3 = PD as u32;
            f.code_mapping(0, CODE_FRAME);
            f.map_code(OTHER_PD, OTHER_PT, 0, OTHER_FRAME, 7 | A);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0xb0, 0x11])
                .unwrap();
            f.vm.write_bytes(u64::from(OTHER_FRAME) + 0x400, &[0xb0, 0x22])
                .unwrap();
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x500, &[0x0f, 0x22, 0xd8])
                .unwrap(); // MOV CR3,EAX
            f.vm.write_bytes(u64::from(OTHER_FRAME) + 0x500, &[0x0f, 0x22, 0xd8])
                .unwrap();
            f.fetch_at(0x400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x11);
            let old = f.cpu.paging_cache_for_test();
            f.cpu.cpu.legacy.cr3 = OTHER_PD;
            f.fetch_at(0x400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x22);
            assert!(f.cpu.paging_cache_for_test().0 > old.0);
            f.cpu.cpu.legacy.cr3 = PD as u32;
            f.fetch_at(0x400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x11);
            let before = f.cpu.paging_cache_for_test();
            f.cpu.cpu.gpr[0] = PD;
            f.fetch_at(0x500).unwrap();
            assert_eq!(f.cpu.cpu.legacy.cr3, PD as u32);
            let after = f.cpu.paging_cache_for_test();
            assert!(after.0 > before.0);
            assert_eq!(after.1, 0);
            f.put(u64::from(CODE_PT), OTHER_FRAME | 7 | A);
            f.fetch_at(0x400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], PD | 0x22);
        }
    }
}

#[test]
fn paged_fetch_pg_transitions_and_explicit_embedder_invalidation() {
    // 80386 PRM §5.2 p. 5-7: PG selects translation. Equal-byte embedder
    // invalidation and bounded retention are engine policies, design §§3.2, 3.5.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 0);
            f.code_mapping(0, CODE_FRAME);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0xb0, 0x11])
                .unwrap();
            f.vm.write_bytes(0x400, &[0xb0, 0x22]).unwrap();
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x500, &[0x0f, 0x22, 0xc0])
                .unwrap(); // MOV CR0,EAX
            f.fetch_at(0x400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x11);
            let before = f.cpu.paging_cache_for_test().0;
            f.cpu.cpu.gpr[0] = 1;
            f.fetch_at(0x500).unwrap();
            assert_eq!(f.cpu.cpu.legacy.cr0 & 0x8000_0000, 0);
            assert!(f.cpu.paging_cache_for_test().0 > before);
            f.fetch_at(0x400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0] & 0xff, 0x22);
            let before = f.cpu.paging_cache_for_test().0;
            f.cpu.cpu.legacy.cr0 |= 0x8000_0000; // restore an already-paged context
            f.fetch_at(0x400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0] & 0xff, 0x11);
            assert!(f.cpu.paging_cache_for_test().0 > before);
            let before = f.cpu.paging_cache_for_test();
            f.cpu.invalidate_paging();
            assert!(f.cpu.paging_cache_for_test().0 > before.0);
            assert_eq!(f.cpu.paging_cache_for_test().1, 0);
            f.fetch_at(0x400).unwrap();
            assert!(f.cpu.paging_cache_for_test().2 > before.2);
            // Identical bytes in another frame still require new physical proof.
            f.vm.write_bytes(u64::from(OTHER_FRAME) + 0x400, &[0xb0, 0x11])
                .unwrap();
            f.put(u64::from(CODE_PT), OTHER_FRAME | 7 | A);
            f.cpu.invalidate_paging();
            let lifts = f.cpu.paging_cache_for_test().2;
            f.fetch_at(0x400).unwrap();
            assert!(f.cpu.paging_cache_for_test().2 > lifts);
            assert!(f.vm.mem.is_code_page(u64::from(OTHER_FRAME >> 12)));
            assert!(!f.vm.mem.is_code_page(u64::from(CODE_FRAME >> 12)));
        }
    }
}

#[test]
fn paged_fetch_physical_and_linear_alias_smc_cover_both_frames() {
    // Both linear aliases converge on one physical frame (80386 PRM §5.2
    // Fig. 5-9 pp. 5-8–9). Physical SMC invalidation is engine policy, §3.5.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 0);
            f.code_mapping(0, CODE_FRAME);
            f.code_mapping(0x1000, OTHER_FRAME);
            f.code_mapping(0x4000, CODE_FRAME);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0xfff, &[0xb0])
                .unwrap();
            f.vm.write_bytes(u64::from(OTHER_FRAME), &[0x11]).unwrap();
            f.fetch_at(0xfff).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x11);
            assert!(f.vm.mem.is_code_page(u64::from(CODE_FRAME >> 12)));
            assert!(f.vm.mem.is_code_page(u64::from(OTHER_FRAME >> 12)));
            f.vm.write_bytes(u64::from(OTHER_FRAME), &[0x22]).unwrap();
            f.fetch_at(0xfff).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x22);
            // A scalar store through a distinct linear alias modifies opcode.
            f.cpu.cpu.legacy.segments[1].base = 0;
            let store = if code32 {
                vec![0xc6, 0x05, 0xff, 0x4f, 0, 0, 0xb3]
            } else {
                vec![0xc6, 0x06, 0xff, 0x4f, 0xb3]
            }; // MOV byte [4fff],b3
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x600, &store)
                .unwrap();
            f.fetch_at(0x600).unwrap();
            f.cpu.cpu.gpr[0] = 0;
            f.fetch_at(0xfff).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0);
            assert_eq!(f.cpu.cpu.gpr[3] & 0xff, 0x22);
            // Exercise the other fetch alias after a physical opcode write.
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0xb0, 0x33])
                .unwrap();
            f.fetch_at(0x4400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x33);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x401, &[0x44])
                .unwrap();
            f.fetch_at(0x400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x44);
            f.fetch_at(0x4400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x44);
        }
    }
}

#[test]
fn paged_fetch_publication_write_race_retries_the_complete_source() {
    // Paging design §3.5: physical source registration rejects a fetched
    // snapshot invalidated before publication, including the second frame.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 0);
            f.code_mapping(0, CODE_FRAME);
            f.code_mapping(0x1000, OTHER_FRAME);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0xfff, &[0xb0])
                .unwrap();
            f.vm.write_bytes(u64::from(OTHER_FRAME), &[0x11]).unwrap();
            f.cpu.paging_publication_hook_for_test(|vm| {
                vm.write_bytes(u64::from(OTHER_FRAME), &[0x55]).unwrap();
            });
            f.fetch_at(0xfff).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x55);
            f.fetch_at(0xfff).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x55);
        }
    }
}

#[test]
fn paged_fetch_tlb_eviction_replaces_equal_byte_physical_proof() {
    // Paging design §§3.2, 3.5: replacement may observe a new frame;
    // equal instruction bytes alone cannot validate old physical sources.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 0);
            f.code_mapping(0, CODE_FRAME);
            for frame in [CODE_FRAME, OTHER_FRAME] {
                f.vm.write_bytes(u64::from(frame) + 0x400, &[0xb0, 0x11])
                    .unwrap();
            }
            f.fetch_at(0x400).unwrap();
            let before = f.cpu.paging_cache_for_test();
            f.code_mapping(0, OTHER_FRAME);
            // Force a direct-mapped TLB collision without a guest flush.
            f.code_mapping(0x40000, OTHER_FRAME);
            f.vm.write_bytes(u64::from(OTHER_FRAME) + 0x500, &[0x90])
                .unwrap();
            f.fetch_at(0x40500).unwrap();
            f.fetch_at(0x400).unwrap();
            let after = f.cpu.paging_cache_for_test();
            assert!(after.0 > before.0);
            assert_eq!(after.1, before.1);
            assert_eq!(after.2, before.2 + 2);
            assert!(f.vm.mem.is_code_page(u64::from(OTHER_FRAME >> 12)));
            assert!(!f.vm.mem.is_code_page(u64::from(CODE_FRAME >> 12)));
            f.vm.write_bytes(u64::from(OTHER_FRAME) + 0x401, &[0x22])
                .unwrap();
            f.fetch_at(0x400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x22);
        }
    }
}

#[test]
fn paged_fetch_ad_write_to_already_fetched_code_retries() {
    // 80386 PRM §5.2.4.3 p. 5-10; Intel486 PRM §5.3.4.3 pp. 5-21–22:
    // fetch sets A. Visibility to translated code follows design §§3.3, 3.5.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.cpu.cpu.legacy.segments[0].base = 0x3fe000;
        // First code frame is its own PTE table. The next page's entry at
        // table+ffc initially decodes as ADD EAX,imm32 (05 00 01 00 ...).
        // Walking the needed last immediate byte sets A in that same opcode,
        // changing it to AND EAX,imm32 (25 00 01 00 ...), so re-fetch is required.
        f.code_mapping(0x3fe000, CODE_PT);
        f.map_code(PD as u32, CODE_PT, 0x3ff000, OTHER_FRAME, 5);
        f.vm.write_bytes(u64::from(OTHER_FRAME), &[0]).unwrap();
        f.cpu.cpu.gpr[0] = 0xff;
        f.fetch_at(0xffc).unwrap();
        assert_eq!(f.cpu.cpu.gpr[0], 0);
        assert_eq!(f.cpu.cpu.rip, 0x1001);
        assert_eq!(f.get(u64::from(CODE_PT + 4 * 1023)), OTHER_FRAME | 5 | A);
    }
}

#[test]
fn paged_fetch_proofs_are_bounded_and_reset_discards_them() {
    // Bounded retention/reset are derived-state policies (paging design §§3.5, 3.7).
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.code_mapping(0, CODE_FRAME);
        f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0x40; 128])
            .unwrap();
        for ip in 0x400..0x480 {
            f.fetch_at(ip).unwrap();
            assert!(f.cpu.paging_cache_for_test().1 <= 64);
        }
        let before = f.cpu.paging_cache_for_test();
        f.cpu.reset_execution_state(Default::default());
        let after = f.cpu.paging_cache_for_test();
        assert!(after.0 > before.0);
        assert_eq!(after.1, 0);
        assert_eq!(after.2, before.2);
        assert!(!f.vm.mem.is_code_page(u64::from(CODE_FRAME >> 12)));
    }
}

#[test]
fn paged_fetch_eviction_releases_pages_but_preserves_shared_cache_tags() {
    // Paging design §3.5: physical source ownership is bounded with the proofs.
    use std::sync::Arc;
    use x86jit_core::{
        lift::{lift_block, FetchAddr},
        BlockKey, CachedBlock, CpuMode,
    };
    for model in MODELS {
        let mut f = Fixture::with_vm(Vm::new(VmConfig::flat(0x80000)), model, true, 0);
        let first = 0x10000;
        for index in 0..65 {
            let frame = first + index * 0x1000;
            f.code_mapping(index * 0x1000, frame);
            f.vm.write_bytes(u64::from(frame) + 0x400, &[0x40, 0xeb, 0xfd])
                .unwrap();
        }
        let bus = u64::from(first) + 0x400;
        let key = BlockKey::new(bus, CpuMode::Compat32);
        let ir = lift_block(&f.vm.mem, FetchAddr::flat(bus), CpuMode::Compat32).unwrap();
        f.vm.cache.insert(
            key,
            CachedBlock::Interpreted(Arc::new(ir)),
            vec![(bus, 3)],
            |spans| {
                for &(start, len) in spans {
                    f.vm.mem.mark_code(start, len);
                }
            },
        );
        for index in 0..65 {
            f.fetch_at(index * 0x1000 + 0x400).unwrap();
            assert!(f.cpu.paging_cache_for_test().1 <= 64);
        }
        assert_eq!(f.cpu.paging_cache_for_test().1, 1);
        assert!(f.vm.mem.is_code_page(u64::from(first >> 12)));
        assert!(f.vm.cache.get(key).is_some());
        for index in 1..64 {
            assert!(!f.vm.mem.is_code_page(u64::from((first >> 12) + index)));
        }
        let before = f.cpu.paging_cache_for_test();
        let generation = f.vm.cache.publication_generation();
        f.vm.write_bytes(u64::from(first) + 0x1400, &[0x48])
            .unwrap();
        f.fetch_at(64 * 0x1000 + 0x400).unwrap();
        assert_eq!(f.cpu.paging_cache_for_test(), before);
        assert_eq!(f.vm.cache.publication_generation(), generation);
        assert!(f.vm.cache.get(key).is_some());
        f.cpu.invalidate_paging();
        assert!(!f.vm.mem.is_code_page(u64::from((first >> 12) + 64)));
        assert!(f.vm.mem.is_code_page(u64::from(first >> 12)));
    }
}

#[test]
fn paged_fetch_private_tags_follow_vcpu_lifetimes() {
    // Paging design §§3.5, 3.7: releasing one owner preserves other vCPUs.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.code_mapping(0, CODE_FRAME);
        f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0x40, 0x40])
            .unwrap();
        f.fetch_at(0x400).unwrap();
        f.fetch_at(0x401).unwrap();
        let mut other = f.vm.new_vcpu();
        other.cpu.legacy = f.cpu.cpu.legacy;
        other.cpu.rip = 0x400;
        other.interpret_paged_for_test(&f.vm).unwrap();
        f.cpu.invalidate_paging();
        assert!(f.vm.mem.is_code_page(u64::from(CODE_FRAME >> 12)));
        other.reset_execution_state(Default::default());
        assert!(!f.vm.mem.is_code_page(u64::from(CODE_FRAME >> 12)));
        other.cpu.rip = 0x400;
        other.interpret_paged_for_test(&f.vm).unwrap();
        assert!(f.vm.mem.is_code_page(u64::from(CODE_FRAME >> 12)));
        drop(other);
        assert!(!f.vm.mem.is_code_page(u64::from(CODE_FRAME >> 12)));
        f.fetch_at(0x400).unwrap();
        // The registration owns only tracking metadata, so VM-first drop is safe.
        drop(f.vm);
        drop(f.cpu);
    }
}

#[test]
fn paged_fetch_segment_fault_precedes_walk_and_public_pg_write_succeeds() {
    // 80386 PRM §§6.4.3–6.5 pp. 6-21–22; Intel486 PRM §§6.8.3–6.9
    // pp. 6-24–25: segmentation precedes paging.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 0);
            f.map_code(PD as u32, CODE_PT, 0, CODE_FRAME, 7);
            f.put(PD, CODE_PT | 7);
            f.cpu.cpu.legacy.segments[0].limit = 0x3ff;
            let exit = f.fetch_at(0x400).unwrap_err();
            assert!(
                matches!(exit, Exit::Exception { vector: 13, .. }),
                "{exit:?}"
            );
            assert_eq!(f.cpu.cpu.legacy.cr2, 0xdead_beef);
            assert_eq!(f.get(PD), CODE_PT | 7);
            assert_eq!(f.get(u64::from(CODE_PT)), CODE_FRAME | 7);
            f.cpu.cpu.legacy.segments[0].limit = u32::MAX;
            f.vm.write_bytes(0x400, &[0x0f, 0x22, 0xc0]).unwrap();
            f.cpu.cpu.legacy.cr0 &= !0x8000_0000;
            f.cpu.cpu.gpr[0] = 0x8000_0001;
            let run = f.cpu.run_steps(&f.vm, 1, StepOptions::default()).unwrap();
            assert!(
                matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                "{run:?}"
            );
            assert_eq!(f.cpu.cpu.legacy.cr0 & 0x8000_0001, 0x8000_0001);
            assert_eq!(f.cpu.cpu.rip, 0x403);
        }
    }
}

#[test]
fn paging_wp_supervisor_writes_both_mapping_classes() {
    // Intel486 PRM §4.1.3 p. 4-7; SDM Vol 3A §§5.1.3, 5.6.1
    // pp. 5-5, 5-33–35: WP protects user AND supervisor read-only pages.
    // 80386 PRM §6.4.2 p. 6-21: supervisor writes ignore R/W.
    for model in MODELS {
        for code32 in [false, true] {
            for user in [0, 4] {
                for (pd_rw, pt_rw) in [(0, 0), (0, 2), (2, 0), (2, 2)] {
                    for wp in [false, true] {
                        let mut f = Fixture::new(model, code32, 0);
                        f.cpu.cpu.legacy.cr0 |= u32::from(wp) << 16;
                        let protected = wp && model != CpuModel::I80386;
                        f.tables(1 | user | pd_rw, 1 | user | pt_rw);
                        f.put(u64::from(DATA), 0x1122_3344);
                        f.cpu.cpu.gpr[0] = 0xaabb_ccdd;
                        if protected && (pd_rw == 0 || pt_rw == 0) {
                            f.pf(true, OFFSET, LINEAR, 3); // P=1, W=1, U=0
                            assert_eq!(f.get(u64::from(DATA)), 0x1122_3344);
                            assert_eq!(f.get(f.pte()) & (A | D), A);
                        } else {
                            f.mov(true, OFFSET).unwrap();
                            assert_eq!(f.get(u64::from(DATA)), 0xaabb_ccdd);
                            assert_eq!(f.get(f.pte()) & (A | D), A | D);
                        }
                        assert_eq!(f.get(f.pde()) & (A | D), A);
                    }
                }
            }
        }
    }
}

#[test]
fn paging_wp_changes_drop_proofs_and_public_cr0_enables_pg() {
    // Paging design §3.2, Q3: conservative generation change on WP changes.
    // 80386 PRM §4.1.3 p. 4-3: bit 16 is outside the 386 CR0 mask.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 0);
            f.code_mapping(0, CODE_FRAME);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0xb0, 0x11])
                .unwrap();
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x500, &[0x0f, 0x22, 0xc0])
                .unwrap();
            for wp in [true, false] {
                f.fetch_at(0x400).unwrap();
                let before = f.cpu.paging_cache_for_test();
                f.cpu.cpu.gpr[0] = u64::from(f.cpu.cpu.legacy.cr0) ^ (1 << 16);
                f.fetch_at(0x500).unwrap();
                if model != CpuModel::I80386 {
                    assert_eq!(f.cpu.cpu.legacy.cr0 & (1 << 16) != 0, wp);
                    assert!(f.cpu.paging_cache_for_test().0 > before.0);
                    assert_eq!(f.cpu.paging_cache_for_test().1, 0);
                } else {
                    assert_eq!(f.cpu.cpu.legacy.cr0 & (1 << 16), 0);
                }
            }
            // WP edited at positive entry must also invalidate.
            if model != CpuModel::I80386 {
                f.fetch_at(0x400).unwrap();
                let before = f.cpu.paging_cache_for_test();
                f.cpu.cpu.legacy.cr0 ^= 1 << 16;
                f.fetch_at(0x400).unwrap();
                assert!(f.cpu.paging_cache_for_test().0 > before.0);
                assert!(f.cpu.paging_cache_for_test().2 > before.2);
            }
            f.cpu.cpu.legacy.cr0 &= !0x8000_0000;
            f.cpu.cpu.gpr[0] = 0x8000_0001;
            f.vm.write_bytes(0x500, &[0x0f, 0x22, 0xc0]).unwrap();
            f.fetch_at(0x500).unwrap();
            assert_eq!(f.cpu.cpu.legacy.cr0 & 0x8000_0001, 0x8000_0001);
        }
    }
}

#[test]
fn paging_invlpg_is_addressed_privileged_and_drops_proofs() {
    // Intel486 PRM §26 INVLPG p. 26-172; SDM Vol 3A §5.10.4.1
    // pp. 5-48–49: an address, no operand read; full flush is allowed.
    for model in [
        CpuModel::I80486Dx,
        CpuModel::I80486DxCpuid,
        CpuModel::I80386,
    ] {
        for code32 in [false, true] {
            let mut f = Fixture::new(model, code32, 0);
            f.code_mapping(0, CODE_FRAME);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0xb0, 0x11])
                .unwrap();
            f.vm.write_bytes(u64::from(OTHER_FRAME) + 0x400, &[0xb0, 0x22])
                .unwrap();
            // A nonzero DS base, address override and indexed addressing. The
            // target page has no PTE; INVLPG must neither walk nor read it.
            let invlpg: &[u8] = if code32 {
                &[0x67, 0x0f, 0x01, 0x78, 0x10] // [BX+SI+10h]
            } else {
                &[0x67, 0x0f, 0x01, 0x7c, 0x8b, 0x10] // [EBX+ECX*4+10h]
            };
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x500, invlpg)
                .unwrap();
            f.vm.write_bytes(u64::from(OTHER_FRAME) + 0x500, invlpg)
                .unwrap();
            f.cpu.cpu.gpr[3] = 0x2000;
            f.cpu.cpu.gpr[6] = 0x30;
            f.cpu.cpu.gpr[1] = 0x0c;
            // INVLPG generates an address without accessing DS or its page.
            f.cpu.cpu.legacy.segments[1].limit = 0;
            f.cpu.cpu.legacy.segments[1].unusable = 1;
            f.fetch_at(0x400).unwrap();
            let before = f.cpu.paging_cache_for_test();
            let flags = format!("{:?}", f.cpu.cpu.flags);
            let cr2 = f.cpu.cpu.legacy.cr2;
            let tables = (f.get(f.pde()), f.get(f.pte()));
            if model == CpuModel::I80386 {
                let exit = f.fetch_at(0x500).unwrap_err();
                assert!(
                    matches!(exit, Exit::Exception { vector: 6, .. }),
                    "{exit:?}"
                );
            } else {
                f.put(u64::from(CODE_PT), OTHER_FRAME | 7 | A);
                f.fetch_at(0x500).unwrap();
                assert!(f.cpu.paging_cache_for_test().0 > before.0);
                assert_eq!(f.cpu.paging_cache_for_test().1, 0);
                f.fetch_at(0x400).unwrap();
                assert_eq!(f.cpu.cpu.gpr[0] & 0xff, 0x22);
            }
            assert_eq!(format!("{:?}", f.cpu.cpu.flags), flags);
            assert_eq!(f.cpu.cpu.legacy.cr2, cr2);
            assert_eq!((f.get(f.pde()), f.get(f.pte())), tables);
            for cpl in 1..4 {
                f.cpu.cpu.legacy.cpl = cpl;
                for segment in &mut f.cpu.cpu.legacy.segments {
                    segment.attributes = (segment.attributes & !0x60) | (u16::from(cpl) << 5);
                }
                let exit = f.fetch_at(0x500).unwrap_err();
                assert!(
                    matches!(exit, Exit::Exception { vector, .. }
                    if vector == if model == CpuModel::I80386 { 6 } else { 13 }),
                    "{exit:?}"
                );
            }
            f.cpu.cpu.legacy.cpl = 0;
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x600, &[0x0f, 0x01, 0xff])
                .unwrap();
            f.vm.write_bytes(u64::from(OTHER_FRAME) + 0x600, &[0x0f, 0x01, 0xff])
                .unwrap();
            let exit = f.fetch_at(0x600).unwrap_err();
            assert!(
                matches!(exit, Exit::Exception { vector: 6, .. }),
                "{exit:?}"
            );
        }
    }
}

#[test]
fn paging_read_then_write_sets_dirty_and_rewalks_changed_entries() {
    // 80386 PRM §5.2.4.3 p. 5-10; Intel486 PRM §5.3.4.3 pp. 5-21–22:
    // a read translation is not evidence that D has been set. The first
    // write through a clean TLB entry revalidates the mapping (design §3.2).
    for model in MODELS {
        for code32 in [false, true] {
            for remap in [false, true] {
                let mut f = Fixture::new(model, code32, 0);
                f.code_mapping(0, CODE_FRAME);
                let read = f.code(false, OFFSET);
                let write = f.code(true, OFFSET);
                f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &read)
                    .unwrap();
                f.vm.write_bytes(u64::from(CODE_FRAME) + 0x500, &write)
                    .unwrap();
                f.put(u64::from(DATA), 0x1122_3344);
                f.put(u64::from(NEXT), 0x5566_7788);
                f.fetch_at(0x400).unwrap();
                assert_eq!(f.get(f.pte()), DATA | 7 | A);
                if remap {
                    f.put(f.pte(), NEXT | 7 | A);
                }
                f.cpu.cpu.gpr[0] = 0xaabb_ccdd;
                f.fetch_at(0x500).unwrap();
                assert_eq!(
                    f.get(f.pte()),
                    (if remap { NEXT } else { DATA }) | 7 | A | D
                );
                assert_eq!(
                    f.get(u64::from(DATA)),
                    if remap { 0x1122_3344 } else { 0xaabb_ccdd }
                );
                assert_eq!(
                    f.get(u64::from(NEXT)),
                    if remap { 0xaabb_ccdd } else { 0x5566_7788 }
                );
            }
        }
    }
}

#[test]
fn paging_a20_change_drops_proofs_and_reselects_directory_and_frame() {
    // Paging design §§3.2, 3.6: A20 gates table traffic and final physical
    // traffic after translation; a changed gate invalidates retained proofs.
    for model in MODELS {
        for code32 in [false, true] {
            let mut f = Fixture::with_vm(Vm::new(VmConfig::flat(0x200000)), model, code32, 0);
            f.cpu.cpu.legacy.cr3 = PD as u32 | 0x100000;
            f.map_code(PD as u32, CODE_PT, 0, CODE_FRAME, 7 | A);
            f.map_code(
                PD as u32 | 0x100000,
                OTHER_PT,
                0,
                OTHER_FRAME | 0x100000,
                7 | A,
            );
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0xb0, 0x11])
                .unwrap();
            f.vm.write_bytes(u64::from(OTHER_FRAME) + 0x100400, &[0xb0, 0x22])
                .unwrap();
            for (enabled, value) in [(1, 0x22), (0, 0x11), (1, 0x22)] {
                let before = f.cpu.paging_cache_for_test();
                f.cpu.cpu.legacy.a20_enabled = enabled;
                f.fetch_at(0x400).unwrap();
                assert_eq!(f.cpu.cpu.gpr[0] & 0xff, value);
                assert!(f.cpu.paging_cache_for_test().0 > before.0);
                assert!(f.cpu.paging_cache_for_test().2 > before.2);
                assert_eq!(f.cpu.paging_cache_for_test().1, 1);
            }
        }
    }
}

#[test]
fn paging_two_vcpus_set_each_accessed_and_dirty_bit_once() {
    // SDM Vol 3A §11.1.2.1 p. 11-3; 80386 PRM §11.2.2 p. 11-5;
    // Intel486 PRM §13.1.2 p. 13-3: individual A/D updates are locked.
    // This witnesses shared table CAS, not multi-vCPU Machine execution.
    use std::sync::{Arc, Barrier, Mutex};
    for model in MODELS {
        for code32 in [false, true] {
            let f = Fixture::new(model, code32, 0);
            let barriers = Arc::new([Barrier::new(2), Barrier::new(2), Barrier::new(2)]);
            let counts = Arc::new(Mutex::new([(0, 0); 3])); // attempts, successful updates
            std::thread::scope(|scope| {
                for cpu_index in 0..2 {
                    let barriers = barriers.clone();
                    let counts = counts.clone();
                    let mut cpu = f.cpu.cpu.clone();
                    let code = f.code(true, OFFSET + cpu_index * 4);
                    let memory = &f.vm.mem;
                    let directory = f.pde();
                    let table = f.pte();
                    cpu.gpr[0] = 0x1234_5678;
                    scope.spawn(move || {
                        let mut first = [true; 3];
                        x86jit_core::interp::with_paging_cas_hook_for_test(
                            move |_, bus, expected, bits, observed| {
                                let i = match (bus, bits) {
                                    (b, A) if b == directory => 0,
                                    (b, A) if b == table => 1,
                                    (b, D) if b == table => 2,
                                    _ => panic!("unexpected CAS"),
                                };
                                if let Some(old) = observed {
                                    let mut counts = counts.lock().unwrap();
                                    counts[i].0 += 1;
                                    counts[i].1 += usize::from(old == expected);
                                } else if first[i] {
                                    first[i] = false;
                                    barriers[i].wait();
                                }
                            },
                            || {
                                x86jit_core::interp::interpret_paged_scalar_for_test(
                                    &code, model, &mut cpu, memory,
                                )
                                .unwrap()
                            },
                        );
                    });
                }
            });
            assert_eq!(*counts.lock().unwrap(), [(2, 1); 3]);
            assert_eq!(f.get(f.pde()), PT | 7 | A);
            assert_eq!(f.get(f.pte()), DATA | 7 | A | D);
            assert_eq!(f.get(u64::from(DATA)), 0x1234_5678);
            assert_eq!(f.get(u64::from(DATA) + 4), 0x1234_5678);
        }
    }
}

#[test]
fn paging_cas_retry_preserves_edits_and_restarts_identity_changes() {
    // Locked updates preserve competing edits (80386 PRM §11.2.2 p. 11-5;
    // Intel486 PRM §13.1.2 p. 13-3). Restart-on-identity-change and the
    // 386's failed-walk A order are engine policies, paging design §3.3 Q2.
    use std::cell::Cell;
    use std::rc::Rc;
    for model in MODELS {
        for code32 in [false, true] {
            for stage in 0..3 {
                for edit in 0..4 {
                    let mut f = Fixture::new(model, code32, 3);
                    f.put(u64::from(DATA), 0x1122_3344);
                    f.put(u64::from(NEXT), 0x5566_7788);
                    f.put(u64::from(OTHER_PT + 4 * ((LINEAR >> 12) & 0x3ff)), NEXT | 7);
                    let bus = if stage == 0 { f.pde() } else { f.pte() };
                    let bits = if stage == 2 { D } else { A };
                    let raced = Rc::new(Cell::new(false));
                    let flag = raced.clone();
                    let result = x86jit_core::interp::with_paging_cas_hook_for_test(
                        move |memory, at, expected, setting, observed| {
                            if at == bus
                                && setting == bits
                                && observed.is_none()
                                && !flag.replace(true)
                            {
                                let next = match edit {
                                    0 => expected | SOFTWARE | if stage == 0 { D } else { 0 },
                                    1 => {
                                        (expected & 0xfff)
                                            | if stage == 0 { OTHER_PT } else { NEXT }
                                    }
                                    2 => expected & !2, // revoke write permission
                                    3 => expected & !1, // remove presence
                                    _ => unreachable!(),
                                };
                                assert_eq!(
                                    memory
                                        .atomic_cas(at, u64::from(expected), u64::from(next), 4)
                                        .unwrap(),
                                    u64::from(expected)
                                );
                            }
                        },
                        || {
                            f.cpu.cpu.gpr[0] = 0xaabb_ccdd;
                            f.mov(true, OFFSET)
                        },
                    );
                    assert!(raced.get());
                    match edit {
                        0 => {
                            result.unwrap();
                            assert_eq!(f.get(bus) & SOFTWARE, SOFTWARE);
                            assert_eq!(f.get(f.pde()) & D, if stage == 0 { D } else { 0 });
                            assert_eq!(f.get(f.pte()) & (A | D), A | D);
                        }
                        1 => {
                            result.unwrap();
                            assert_eq!(f.get(u64::from(DATA)), 0x1122_3344);
                            assert_eq!(f.get(u64::from(NEXT)), 0xaabb_ccdd);
                            let pte = if stage == 0 {
                                u64::from(OTHER_PT + 4 * ((LINEAR >> 12) & 0x3ff))
                            } else {
                                f.pte()
                            };
                            assert_eq!(f.get(pte) & (A | D), A | D);
                        }
                        2 | 3 => {
                            let exit = result.unwrap_err();
                            assert!(
                                matches!(exit, Exit::Exception { vector: 14, .. }),
                                "{exit:?}"
                            );
                            assert_eq!(
                                exit.fault_metadata().unwrap().error_code(),
                                ErrorCode::bits(if edit == 2 { 7 } else { 6 }, 7)
                            );
                            assert_eq!(f.cpu.cpu.legacy.cr2, LINEAR);
                            assert_eq!(f.get(bus) & D, 0);
                            assert_eq!(f.get(bus) & A, if edit == 2 || stage == 2 { A } else { 0 });
                            assert_eq!(f.get(u64::from(DATA)), 0x1122_3344);
                            assert_eq!(f.get(u64::from(NEXT)), 0x5566_7788);
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
}

impl Fixture {
    fn memory_code(&self, opcode: &[u8], extension: u8, offset: u32, size: u8) -> Vec<u8> {
        let mut code = Vec::new();
        if (size == 4) != self.code32 {
            code.push(0x66);
        }
        code.extend(opcode);
        code.push((extension << 3) | if self.code32 { 5 } else { 6 });
        if self.code32 {
            code.extend(offset.to_le_bytes());
        } else {
            code.extend((offset as u16).to_le_bytes());
        }
        code
    }
    fn assert_pf(&mut self, code: &[u8], linear: u32, error: u32) {
        let gpr = self.cpu.cpu.gpr;
        let flags = format!("{:?}", self.cpu.cpu.flags);
        let ip = self.cpu.cpu.rip;
        let exit = self.execute(code).unwrap_err();
        assert!(
            matches!(exit, Exit::Exception { vector: 14, .. }),
            "{code:02x?}: {exit:?}"
        );
        assert_eq!(
            exit.fault_metadata().unwrap().error_code(),
            ErrorCode::bits(error, 7)
        );
        assert_eq!(self.cpu.cpu.legacy.cr2, linear, "{code:02x?}");
        assert_eq!(self.cpu.cpu.gpr, gpr, "{code:02x?}");
        assert_eq!(format!("{:?}", self.cpu.cpu.flags), flags);
        assert_eq!(self.cpu.cpu.rip, ip);
    }
    fn put64(&self, at: u64, value: u64) {
        self.vm.write_bytes(at, &value.to_le_bytes()).unwrap();
    }
    fn system_page(&self, index: u32, frame: u32, flags: u32) {
        self.put(u64::from(PT + index * 4), frame | flags);
    }
}

#[test]
fn step4_strings_preflight_each_operand_and_crossing() {
    // 80386 PRM §14.6 pp. 14-4–5, §17 MOVS/CMPS/LODS/SCAS/STOS;
    // §9.8.14 pp. 9-17–18: failed iterations preserve indices/count/flags.
    for model in MODELS {
        for code32 in [false, true] {
            for opcode in [0xa4, 0xa5, 0xaa, 0xab, 0xac, 0xad, 0xa6, 0xa7, 0xae, 0xaf] {
                for missing in [0, 1] {
                    let source = matches!(opcode, 0xa4..=0xa7 | 0xac | 0xad);
                    let dest = !matches!(opcode, 0xac | 0xad);
                    if (missing == 0 && !source) || (missing == 1 && !dest) {
                        continue;
                    }
                    for crossing in [false, true] {
                        let mut f = Fixture::new(model, code32, 3);
                        let size = if opcode & 1 == 0 { 1 } else { 4 };
                        let offset = if crossing { 0xfff } else { 0 };
                        f.cpu.cpu.gpr[1] = 2;
                        f.cpu.cpu.gpr[6] = u64::from(OFFSET + offset);
                        f.cpu.cpu.gpr[7] = u64::from(OFFSET + 0x2000 + offset);
                        f.system_page(4, NEXT, 7);
                        f.system_page(5, 0xe000, 7);
                        f.system_page(6, 0xf000, 7);
                        let page = 3 + missing * 2 + u32::from(crossing && size > 1);
                        f.system_page(page, 0, 0);
                        let linear =
                            BASE + page * 0x1000 + if crossing && size > 1 { 0 } else { offset };
                        let mut code = vec![0xf3];
                        if size == 4 && !code32 {
                            code.push(0x66);
                        }
                        code.push(opcode);
                        f.assert_pf(
                            &code,
                            linear,
                            if missing == 1 && matches!(opcode, 0xa4 | 0xa5 | 0xaa | 0xab) {
                                6
                            } else {
                                4
                            },
                        );
                        assert_eq!(f.get(u64::from(DATA)), 0);
                        assert_eq!(f.get(0xe000), 0);
                    }
                }
            }
        }
    }
}

#[test]
fn step4_rep_completed_iterations_survive_fault_and_checkpoint_restore() {
    // 80386 PRM §9.2.1 pp. 9-2–3, §9.8.14.1 p. 9-18, §14.6
    // pp. 14-4–5: REP restarts with the remaining count and advanced indices.
    for model in MODELS {
        for opcode in [0xa5, 0xab, 0xad, 0xa7, 0xaf] {
            let mut f = Fixture::new(model, true, 0);
            f.cpu.cpu.gpr[0] = 0x1234_5678;
            f.cpu.cpu.gpr[1] = 2;
            f.cpu.cpu.gpr[6] = u64::from(OFFSET + 0xffc);
            f.cpu.cpu.gpr[7] = u64::from(OFFSET + 0xffc);
            f.put(u64::from(DATA + 0xffc), 0x1234_5678);
            let code = [0xf3, opcode];
            f.mapped_instruction(&code);
            f.fetch_at(0x400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[1], 1);
            let saved = f.cpu.cpu.clone();
            let mut ram = vec![0; f.vm.mem.size() as usize];
            for (page, bytes) in ram.chunks_mut(0x1000).enumerate() {
                f.vm.read_bytes(page as u64 * 0x1000, bytes).unwrap();
            }
            let exit = f.cpu.interpret_paged_for_test(&f.vm).unwrap_err();
            assert!(
                matches!(exit, Exit::Exception { vector: 14, .. }),
                "{exit:?}"
            );
            assert_eq!(f.cpu.cpu.legacy.cr2, LINEAR + 0x1000);
            assert_eq!(f.cpu.cpu.gpr, saved.gpr);
            f.put(f.pte() + 4, NEXT | 7);
            f.put(u64::from(NEXT), 0x1234_5678);
            f.cpu.interpret_paged_for_test(&f.vm).unwrap();
            assert_eq!(f.cpu.cpu.gpr[1], 0);
            let finished = f.cpu.cpu.gpr;
            // Full checkpoint: CPU and RAM, followed by derived-state invalidation.
            f.cpu.cpu = saved;
            for (page, bytes) in ram.chunks(0x1000).enumerate() {
                f.vm.write_bytes(page as u64 * 0x1000, bytes).unwrap();
            }
            f.cpu.invalidate_paging();
            f.put(f.pte() + 4, NEXT | 7);
            f.put(u64::from(NEXT), 0x1234_5678);
            f.cpu.interpret_paged_for_test(&f.vm).unwrap();
            assert_eq!(f.cpu.cpu.gpr, finished);
        }
    }
}

#[test]
fn step4_stack_frames_all_widths_preflight_before_commit() {
    // 80386 PRM §17 PUSH/POP/PUSHA/POPA/ENTER/LEAVE/CALL/RET/IRET;
    // SDM Vol 3A §7.15 Event 14 pp. 7-47–50. Engine whole-frame restart.
    for model in MODELS {
        for code32 in [false, true] {
            for size in [2, 4] {
                for opcode in [
                    0x50, 0x58, 0x9c, 0x9d, 0x60, 0x61, 0xc8, 0xc9, 0xe8, 0xc3, 0xcb, 0xcf,
                ] {
                    let mut f = Fixture::new(model, code32, 0);
                    let pushing = matches!(opcode, 0x50 | 0x9c | 0x60 | 0xc8 | 0xe8);
                    let frame = matches!(opcode, 0x60 | 0xc8);
                    if pushing && frame {
                        f.system_page(3, 0, 0);
                        f.system_page(4, NEXT, 7);
                        f.cpu.cpu.gpr[4] = u64::from(OFFSET + 0x1000 + size);
                    } else {
                        f.cpu.cpu.gpr[4] =
                            u64::from(OFFSET + 0xfff + if pushing { size } else { 0 });
                    }
                    f.cpu.cpu.gpr[5] = u64::from(OFFSET + 0xfff);
                    f.cpu.cpu.gpr[0] = 0xdead_beef;
                    f.put(u64::from(DATA + 0xffc), 0x400);
                    f.put(u64::from(NEXT), 0xfeed_beef);
                    let mut code = Vec::new();
                    if (size == 4) != code32 {
                        code.push(0x66);
                    }
                    code.push(opcode);
                    if opcode == 0xc8 {
                        code.extend([0, 0, 2]);
                    }
                    if opcode == 0xe8 {
                        code.extend(vec![0; size as usize]);
                    }
                    f.assert_pf(
                        &code,
                        if pushing && frame {
                            LINEAR + 0x1000 - size
                        } else {
                            LINEAR + 0x1000
                        },
                        if pushing { 2 } else { 0 },
                    );
                    assert_eq!(f.get(u64::from(DATA + 0xffc)), 0x400);
                    assert_eq!(f.get(u64::from(NEXT)), 0xfeed_beef);
                }
            }
        }
    }
}

#[test]
fn step4_locked_scatter_operands_translate_and_are_restartable() {
    // SDM Vol 3A §11.1.2.2 pp. 11-4–5; paging design §3.4 retains
    // the existing single-vCPU scatter contract and split-device refusal.
    for model in MODELS {
        for opcode in [0x01, 0x87, 0xb1] {
            if opcode == 0xb1 && model == CpuModel::I80386 {
                continue;
            }
            let mut f = Fixture::new(model, true, 3);
            f.cpu.cpu.gpr[0] = 0x1234_5678;
            let mut code = vec![0xf0];
            if opcode == 0xb1 {
                code.push(0x0f);
            }
            code.extend(f.memory_code(&[opcode], 0, OFFSET + 0xffe, 4));
            f.put(u64::from(DATA + 0xffc), 0x0001_0000);
            f.assert_pf(&code, LINEAR + 0x1000, 6);
            assert_eq!(f.get(u64::from(DATA + 0xffc)), 0x0001_0000);
            f.system_page(4, NEXT, 7);
            f.execute(&code).unwrap();
            assert_eq!(f.get(f.pte()) & D, D);
            assert_eq!(f.get(f.pte() + 4) & D, D);
            assert_eq!(
                f.get(u64::from(DATA + 0xffc)) >> 16,
                if opcode == 0xb1 {
                    1
                } else if opcode == 0x01 {
                    0x5679
                } else {
                    0x5678
                }
            );
        }
    }
}

#[test]
fn step4_x87_full_images_and_bcd_cross_pages_without_partial_state() {
    // SDM Vol 3A §11.1.1 pp. 11-2–3: engine strong restart policy;
    // Vol 1 §8.1.9 pp. 8-10–12; 486 PRM FBLD/FBSTP pp. 26-77–80.
    for model in MODELS {
        for code32 in [false, true] {
            for (opcode, extension, writing) in [
                (0xd9, 4, false),
                (0xdd, 4, false),
                (0xd9, 6, true),
                (0xdd, 6, true),
                (0xdf, 4, false),
                (0xdf, 6, true),
                (0xdb, 5, false),
                (0xdb, 7, true),
            ] {
                let mut f = Fixture::new(model, code32, 0);
                f.cpu.set_fpu_model(x86jit_core::FpuModel::Present);
                let code = f.memory_code(
                    &[opcode],
                    extension,
                    OFFSET + 0xffc,
                    if code32 { 4 } else { 2 },
                );
                let fp = format!("{:?}", f.cpu.cpu);
                f.put(u64::from(DATA + 0xffc), 0x1122_3344);
                f.assert_pf(&code, LINEAR + 0x1000, if writing { 2 } else { 0 });
                let mut after = f.cpu.cpu.clone();
                after.legacy.cr2 = 0xdead_beef;
                assert_eq!(format!("{after:?}"), fp);
                assert_eq!(f.get(u64::from(DATA + 0xffc)), 0x1122_3344);
                f.system_page(4, NEXT, 7);
                f.execute(&code).unwrap();
                assert_eq!(f.get(f.pte()) & (A | D), if writing { A | D } else { A });
                assert_eq!(
                    f.get(f.pte() + 4) & (A | D),
                    if writing { A | D } else { A }
                );
            }
        }
    }
}

fn step4_descriptor(base: u32, limit: u32, attr: u16) -> u64 {
    u64::from(limit & 0xffff)
        | (u64::from(base & 0xffff) << 16)
        | (u64::from((base >> 16) & 0xff) << 32)
        | (u64::from(attr & 0xff) << 40)
        | (u64::from((limit >> 16) & 15) << 48)
        | (u64::from(attr >> 12) << 52)
        | (u64::from(base >> 24) << 56)
}

impl Fixture {
    fn system_setup(&mut self) {
        use x86jit_core::legacy::{SegmentCache, SystemSegment};
        self.system_page(6, 0x12000, 3);
        self.system_page(7, 0x13000, 3);
        self.system_page(8, 0x14000, 3);
        self.system_page(9, 0x15000, 3);
        self.system_page(10, 0x16000, 3);
        self.cpu.cpu.legacy.gdtr.base = BASE + 0x6000;
        self.cpu.cpu.legacy.gdtr.limit = 0x2f;
        self.cpu.cpu.legacy.idtr.base = BASE + 0x7000;
        self.cpu.cpu.legacy.idtr.limit = 0x7ff;
        self.cpu.cpu.cs = 8;
        self.cpu.cpu.ds = 16;
        self.cpu.cpu.es = 16;
        self.cpu.cpu.ss = 16;
        self.cpu.cpu.gpr[4] = u64::from(OFFSET + 0x800);
        self.put64(0x12008, step4_descriptor(0, 0xfffff, 0xc09b));
        self.put64(0x12010, step4_descriptor(BASE, 0xfffff, 0xc093));
        self.put64(0x12018, step4_descriptor(BASE + 0x8000, 0x100, 0x8b));
        self.put64(0x12020, step4_descriptor(BASE + 0x9000, 0x100, 0x89));
        self.cpu.cpu.legacy.tr = SystemSegment {
            selector: 24,
            cache: SegmentCache {
                base: BASE + 0x8000,
                limit: 0x100,
                attributes: 0x8b,
                ..Default::default()
            },
            ..Default::default()
        };
    }
    fn gate(&self, vector: u8, ip: u32, size: u8) {
        self.put64(
            0x13000 + u64::from(vector) * 8,
            u64::from(ip & 0xffff)
                | (8 << 16)
                | (u64::from(if size == 4 { 0x8e_u32 } else { 0x86 }) << 40)
                | (u64::from(ip & 0xffff_0000) << 32),
        );
    }
    fn mapped_instruction(&self, code: &[u8]) {
        self.code_mapping(0, CODE_FRAME);
        self.vm
            .write_bytes(u64::from(CODE_FRAME) + 0x400, code)
            .unwrap();
    }
}

#[test]
fn step4_descriptor_gdt_ldt_supervisor_reads_and_accessed_cas() {
    // SDM Vol 3A §5.6.1 p. 5-33 (implicit supervisor), §3.4.5.1
    // p. 3-13 (A); 486 WP §5.1.3 p. 5-5, §5.6.1 pp. 5-33–35.
    for model in MODELS {
        for ldt in [false, true] {
            for protected in [false, true] {
                let mut f = Fixture::new(model, true, 3);
                f.system_setup();
                f.cpu.cpu.legacy.cr0 |= 1 << 16;
                f.cpu.cpu.gpr[0] = if ldt { 15 } else { 19 };
                let at = if ldt { 0x12008 } else { 0x12010 };
                f.put64(at, step4_descriptor(BASE, 0xfffff, 0xc0f2));
                f.system_page(6, 0x12000, if protected { 1 } else { 3 });
                if ldt {
                    f.cpu.cpu.legacy.ldtr.selector = 40;
                    f.cpu.cpu.legacy.ldtr.cache = x86jit_core::legacy::SegmentCache {
                        base: BASE + 0x6000,
                        limit: 0x2f,
                        attributes: 0x82,
                        ..Default::default()
                    };
                }
                if protected && model != CpuModel::I80386 {
                    f.assert_pf(&[0x8e, 0xd8], BASE + 0x6000 + (at as u32 & 0xfff) + 5, 3);
                    assert_eq!(f.cpu.cpu.ds, 16);
                    assert_eq!((f.get(at + 4) >> 8) & 1, 0);
                } else {
                    f.execute(&[0x8e, 0xd8]).unwrap();
                    assert_eq!(f.cpu.cpu.ds, if ldt { 15 } else { 19 });
                    assert_eq!((f.get(at + 4) >> 8) & 1, 1);
                    assert_eq!(f.get(u64::from(PT + 6 * 4)) & D, D);
                }
            }
        }
    }
}

#[test]
fn step4_system_descriptor_and_table_images_cross_pages() {
    // SDM Vol 3A §5.6.1 p. 5-33; 80386 PRM §9.8.14 pp. 9-17–18:
    // implicit descriptor reads are supervisor; LGDT's operand is explicit.
    for model in MODELS {
        for ldt in [false, true] {
            let mut f = Fixture::new(model, true, 3);
            f.system_setup();
            f.cpu.cpu.gpr[0] = if ldt { 15 } else { 19 };
            if ldt {
                f.cpu.cpu.legacy.ldtr.selector = 40;
                f.cpu.cpu.legacy.ldtr.cache = x86jit_core::legacy::SegmentCache {
                    base: BASE + 0x6ff4,
                    limit: 0x2f,
                    attributes: 0x82,
                    ..Default::default()
                };
                f.put(0x12ffc, 0xffff);
            } else {
                f.cpu.cpu.legacy.gdtr.base = BASE + 0x6fec;
                f.put(0x12ffc, 0xffff);
            }
            f.system_page(7, 0, 0);
            f.assert_pf(&[0x8e, 0xd8], BASE + 0x7000, 0);
        }
        for load in [false, true] {
            let mut f = Fixture::new(model, true, 0);
            let code = f.memory_code(&[0x0f, 0x01], if load { 2 } else { 0 }, OFFSET + 0xffe, 4);
            f.put(u64::from(DATA + 0xffc), 0x1122_3344);
            let table = f.cpu.cpu.legacy.gdtr;
            f.assert_pf(&code, LINEAR + 0x1000, if load { 0 } else { 2 });
            assert_eq!(f.cpu.cpu.legacy.gdtr, table);
            assert_eq!(f.get(u64::from(DATA + 0xffc)), 0x1122_3344);
        }
    }
}

#[test]
fn step4_io_bitmap_and_port_strings_use_supervisor_paging() {
    // 386 PRM §§8.3, 17.2.2.5 pp. 8-4–6, 17-15; SDM Vol 3A
    // §5.6.1 p. 5-33: TSS/bitmap are implicit supervisor references.
    for model in MODELS {
        for opcode in [0xec, 0xee, 0x6d, 0x6f] {
            let mut f = Fixture::new(model, true, 3);
            f.system_setup();
            f.cpu.cpu.gpr[2] = 0x20;
            f.cpu.cpu.gpr[1] = 2;
            f.cpu.cpu.gpr[6] = u64::from(OFFSET);
            f.cpu.cpu.gpr[7] = u64::from(OFFSET);
            f.vm.write_bytes(0x14066, &0x80_u16.to_le_bytes()).unwrap();
            let code = if opcode < 0x80 {
                vec![0xf3, opcode]
            } else {
                vec![opcode]
            };
            f.mapped_instruction(&code);
            // Pointer page absent: #PF before a port handoff, U=0 at CPL3.
            f.system_page(8, 0, 0);
            let regs = f.cpu.cpu.gpr;
            let exit = f.fetch_at(0x400).unwrap_err();
            assert!(
                matches!(exit, Exit::Exception { vector: 14, .. }),
                "{exit:?}"
            );
            assert_eq!(f.cpu.cpu.legacy.cr2, BASE + 0x8066);
            assert_eq!(
                exit.fault_metadata().unwrap().error_code(),
                ErrorCode::bits(0, 7)
            );
            assert_eq!(f.cpu.cpu.gpr, regs);
            assert!(f.cpu.cpu.pending_port_in.is_none());
            f.system_page(8, 0x14000, 3);
            let exit = f.fetch_at(0x400).unwrap_err();
            assert!(matches!(exit, Exit::PortIo { .. }), "{exit:?}");
            if opcode == 0x6d {
                f.cpu.complete_port_in(0x1234_5678);
                f.cpu.interpret_paged_for_test(&f.vm).unwrap();
                assert_eq!(f.get(u64::from(DATA)), 0x1234_5678);
                assert_eq!(f.cpu.cpu.gpr[7], u64::from(OFFSET + 4));
                assert_eq!(f.cpu.cpu.gpr[1], 1);
            }
        }
    }
}

#[test]
fn step4_port_string_crossing_fault_precedes_port_effect() {
    // SDM Vol 2A INS pp. 3-464–465, Vol 2B OUTS pp. 4-173–175;
    // paging design §3.4: each iteration preflights its complete operand.
    for model in MODELS {
        for output in [false, true] {
            let mut f = Fixture::new(model, true, 0);
            f.cpu.cpu.gpr[1] = 2;
            f.cpu.cpu.gpr[6] = u64::from(OFFSET + 0xffe);
            f.cpu.cpu.gpr[7] = u64::from(OFFSET + 0xffe);
            f.mapped_instruction(&[0xf3, if output { 0x6f } else { 0x6d }]);
            let before = f.cpu.cpu.gpr;
            let exit = f.fetch_at(0x400).unwrap_err();
            assert!(
                matches!(exit, Exit::Exception { vector: 14, .. }),
                "{exit:?}"
            );
            assert_eq!(f.cpu.cpu.legacy.cr2, LINEAR + 0x1000);
            assert_eq!(f.cpu.cpu.gpr, before);
            assert!(f.cpu.cpu.port_string_in.is_none());
            assert!(f.cpu.cpu.pending_port_in.is_none());
        }
    }
}

#[test]
fn step4_in_guest_page_fault_frames_and_iret_both_gate_widths() {
    // 80386 PRM §9.8.14 pp. 9-17–19; §17 INT/IRET pp. 17-70–79:
    // #PF saves the original instruction IP and an error code at gate width.
    for model in MODELS {
        for size in [2, 4] {
            let mut f = Fixture::with_delivery(
                Vm::new(VmConfig::flat(0x20000)),
                model,
                true,
                0,
                x86jit_core::ExceptionDelivery::InGuest,
            );
            f.system_setup();
            f.gate(14, 0x800, size);
            f.mapped_instruction(&f.code(false, OFFSET + 0x1000));
            let old_sp = f.cpu.cpu.gpr[4];
            f.fetch_at(0x400).unwrap();
            assert_eq!(f.cpu.cpu.rip, 0x800);
            assert_eq!(f.cpu.cpu.legacy.cr2, LINEAR + 0x1000);
            assert_eq!(f.cpu.cpu.gpr[4], old_sp - u64::from(4 * size));
            let at = u64::from(DATA) + 0x800 - u64::from(4 * size);
            assert_eq!(f.vm.mem.read(at, size).unwrap(), 0);
            assert_eq!(f.vm.mem.read(at + u64::from(size), size).unwrap(), 0x400);
            f.system_page(4, NEXT, 7);
            f.cpu.cpu.gpr[4] += u64::from(size); // handler discards #PF's error code
            let code = if size == 2 {
                vec![0x66, 0xcf]
            } else {
                vec![0xcf]
            };
            f.execute(&code).unwrap();
            assert_eq!(f.cpu.cpu.rip, 0x400);
            assert_eq!(f.cpu.cpu.gpr[4], old_sp);
        }
    }
}

#[test]
fn step4_nested_delivery_updates_cr2_and_obeys_double_fault_table() {
    // 80386 PRM §9.8.8 Tables 9-3/9-4 p. 9-13; SDM Vol 3A
    // §7.15 Event 8, Table 7-5 pp. 7-36–37: #PF→#PF is #DF;
    // contributory→#PF is serial. CR2 is overwritten (§9.8.14 p. 9-18).
    for model in MODELS {
        for first_pf in [false, true] {
            let mut f = Fixture::with_delivery(
                Vm::new(VmConfig::flat(0x20000)),
                model,
                true,
                0,
                x86jit_core::ExceptionDelivery::InGuest,
            );
            f.system_setup();
            f.cpu.cpu.gpr[0] = 0x8000_0000;
            f.mapped_instruction(if first_pf {
                &[0xa1, 0, 0x40, 0, 0]
            } else {
                &[0x0f, 0x22, 0xc0]
            }); // MOV CR0,EAX with PE=0 → #GP
            if first_pf {
                // #PF's IDT entry on the absent page; #DF's entry remains mapped.
                f.cpu.cpu.legacy.idtr.base = BASE + 0x7f90;
                f.system_page(8, 0, 0);
                f.put64(0x13fd0, 0x900 | (8 << 16) | (0x8e_u64 << 40));
                f.fetch_at(0x400).unwrap();
                assert_eq!(f.cpu.cpu.rip, 0x900);
                assert_eq!(f.cpu.cpu.legacy.cr2, BASE + 0x8000);
                assert_eq!(f.vm.mem.read(u64::from(DATA) + 0x7f0, 4).unwrap(), 0);
            } else {
                // #GP's IDT entry absent; #PF's entry remains in the next page.
                f.cpu.cpu.legacy.idtr.base = BASE + 0x6f90;
                f.system_page(6, 0, 0);
                // This also hides GDT: install a separate present GDT mapping.
                f.cpu.cpu.legacy.gdtr.base = BASE + 0x9000;
                f.put64(0x15008, step4_descriptor(0, 0xfffff, 0xc09b));
                f.put64(0x13000, 0x900 | (8 << 16) | (0x8e_u64 << 40));
                f.fetch_at(0x400).unwrap();
                assert_eq!(f.cpu.cpu.rip, 0x900);
                assert_eq!(f.cpu.cpu.legacy.cr2, BASE + 0x6ff8);
            }
        }
        let mut f = Fixture::with_delivery(
            Vm::new(VmConfig::flat(0x20000)),
            model,
            true,
            0,
            x86jit_core::ExceptionDelivery::InGuest,
        );
        f.system_setup();
        f.gate(14, 0x800, 4);
        f.gate(8, 0x900, 4);
        f.cpu.cpu.gpr[4] = u64::from(OFFSET + 0x1002);
        f.mapped_instruction(&[0xa1, 0, 0x50, 0, 0]);
        let exit = f.fetch_at(0x400).unwrap_err();
        assert!(matches!(exit, Exit::Shutdown { .. }), "{exit:?}");
        assert_eq!(f.cpu.cpu.legacy.cr2, LINEAR + 0x1000);
    }
}

#[test]
fn step4_privilege_transition_tss_and_inner_stack_are_supervisor() {
    // SDM Vol 3A §5.6.1 p. 5-33; 80386 PRM §§6.5, 9.6 pp. 6-22,
    // 9-8–9; §17 INT pp. 17-70–71: inner stack and TSS use supervisor rights.
    for model in MODELS {
        let mut f = Fixture::with_delivery(
            Vm::new(VmConfig::flat(0x20000)),
            model,
            true,
            3,
            x86jit_core::ExceptionDelivery::InGuest,
        );
        f.system_setup();
        f.cpu.cpu.cs = 11;
        f.cpu.cpu.ss = 19;
        f.cpu.cpu.legacy.cr0 |= 1 << 16;
        f.put(0x14004, 0xa800);
        f.put(0x14008, 16);
        f.gate(14, 0x800, 4);
        f.mapped_instruction(&f.code(false, OFFSET + 0x1000));
        f.fetch_at(0x400).unwrap();
        assert_eq!(f.cpu.cpu.legacy.cpl, 0);
        assert_eq!(f.cpu.cpu.rip, 0x800);
        assert_eq!(f.cpu.cpu.gpr[4], 0xa800 - 24);
        assert_eq!(f.get(0x167e8), 4); // original user #PF error
        assert_eq!(f.get(0x167ec), 0x400);
        assert_eq!(f.get(0x167fc), 19); // old SS
        assert_eq!(f.get(u64::from(PT + 10 * 4)) & D, D);
    }
}

#[test]
fn step4_bitmap_second_page_fault_precedes_port_handoff() {
    // 386 PRM §8.3.2 pp. 8-5–6, §17.2.2.5 p. 17-15: two-byte
    // bitmap window, translated with implicit supervisor access at CPL3.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 3);
        f.system_setup();
        f.cpu.cpu.legacy.tr.cache.limit = 0x1100;
        f.cpu.cpu.gpr[2] = 0x20;
        f.vm.write_bytes(0x14066, &0xffb_u16.to_le_bytes()).unwrap();
        f.system_page(9, 0, 0);
        f.mapped_instruction(&[0xec]);
        let exit = f.fetch_at(0x400).unwrap_err();
        assert!(
            matches!(exit, Exit::Exception { vector: 14, .. }),
            "{exit:?}"
        );
        assert_eq!(f.cpu.cpu.legacy.cr2, BASE + 0x9000);
        assert_eq!(
            exit.fault_metadata().unwrap().error_code(),
            ErrorCode::bits(0, 7)
        );
        assert!(f.cpu.cpu.pending_port_in.is_none());
    }
}

impl Fixture {
    fn incoming_task(&mut self, new_directory: u32, postcommit_fault: bool) {
        self.system_setup();
        self.put(0x1501c, new_directory | 0x18);
        self.put(0x15020, 0x900);
        self.put(0x15024, 2);
        self.put(0x15028, 0xabcdef);
        self.put(0x15038, OFFSET + 0x700);
        for (offset, selector) in [(0x48, 16), (0x4c, 8), (0x50, 16), (0x54, 16)] {
            self.put(0x15000 + offset, selector);
        }
        self.put(0x15066, 0x80);
        self.put(0x1401c, 0xdead_beef);
        self.put(u64::from(new_directory + 4 * (BASE >> 22)), OTHER_PT | 7);
        self.put(
            u64::from(OTHER_PT + 6 * 4),
            0x17000 | if postcommit_fault { 0 } else { 3 },
        );
        self.put64(0x17008, step4_descriptor(0, 0xfffff, 0xc09b));
        self.put64(0x17010, step4_descriptor(BASE + 0x10000, 0xfffff, 0xc093));
        self.put(u64::from(OTHER_PT + 0x13 * 4), 0x18000 | 7);
        self.map_code(new_directory, 0x7000, 0, OTHER_FRAME, 7);
        self.vm
            .write_bytes(u64::from(OTHER_FRAME) + 0x900, &self.code(false, OFFSET))
            .unwrap();
        self.put(0x18000, 0x7654_3210);
        self.cpu.cpu.gpr[0] = 0x1122_3344;
    }
}

#[test]
fn step4_task_cr3_handoff_saves_old_space_and_loads_descriptors_in_new_space() {
    // 80386 PRM §7.1 Fig. 7-1 pp. 7-2–3, §7.5 pp. 7-8–9;
    // SDM Vol 3A §10.3 pp. 10-10–11: incoming PDBR precedes descriptor loads.
    for model in MODELS {
        for postcommit_fault in [false, true] {
            let mut f = Fixture::new(model, true, 0);
            f.incoming_task(OTHER_PD, postcommit_fault);
            let code = [0xea, 0, 0, 0, 0, 0x20, 0]; // supported far JMP to 386 TSS
            f.mapped_instruction(&code);
            let result = f.fetch_at(0x400);
            // SDM Vol 3A §10.3 steps 9, 14 pp. 10-11–13: descriptor
            // faults have the complete incoming register/selector context.
            assert_eq!(f.cpu.cpu.rip, 0x900);
            assert_eq!(f.cpu.cpu.gpr[0], 0xabcdef);
            assert_eq!(f.cpu.cpu.gpr[4], u64::from(OFFSET + 0x700));
            assert_eq!((f.cpu.cpu.cs, f.cpu.cpu.ss, f.cpu.cpu.ds), (8, 16, 16));
            assert_eq!(f.cpu.cpu.legacy.cpl, 0);
            if postcommit_fault {
                let exit = result.unwrap_err();
                assert!(
                    matches!(exit, Exit::Exception { vector: 14, .. }),
                    "{exit:?}"
                );
                assert_eq!(f.cpu.cpu.legacy.cr2, BASE + 0x6008);
                assert_eq!(exit.fault_metadata().unwrap().saved_rip(), 0x900);
            } else {
                result.unwrap();
                assert_eq!(f.cpu.cpu.legacy.segments[1].base, BASE + 0x10000);
                assert_eq!(f.cpu.cpu.legacy.segments[3].base, BASE + 0x10000);
                f.cpu.interpret_paged_for_test(&f.vm).unwrap();
                assert_eq!(f.cpu.cpu.gpr[0], 0x7654_3210);
            }
            assert_eq!(f.cpu.cpu.legacy.tr.selector, 32);
            assert_eq!(
                f.cpu.cpu.legacy.cr3,
                OTHER_PD | if model == CpuModel::I80386 { 0 } else { 0x18 }
            );
            assert_eq!(f.get(0x14020), 0x407);
            assert_eq!(f.get(0x14028), 0x1122_3344);
            assert_eq!(f.get(0x1401c), 0xdead_beef); // CR3 is static, never saved
            assert_eq!((f.get(0x1201c) >> 8) & 0xf, 9); // old busy cleared in old mapping
            assert_eq!((f.get(0x12024) >> 8) & 0xf, 11);
        }
    }
}

#[test]
fn step4_task_save_fault_before_commit_retains_outgoing_ownership() {
    // 80386 PRM §7.5 p. 7-8; SDM Vol 3A §10.3 pp. 10-10–11:
    // preflight of outgoing saves precedes the existing task commit seam.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.system_setup();
        f.cpu.cpu.legacy.tr.cache.base = BASE + 0xafe0;
        let old = f.cpu.cpu.legacy;
        f.assert_pf(&[0xea, 0, 0, 0, 0, 0x20, 0], BASE + 0xb000, 2);
        assert_eq!(f.cpu.cpu.legacy.cr3, old.cr3);
        assert_eq!(f.cpu.cpu.legacy.tr, old.tr);
        assert_eq!(f.get(0x14020), 0);
        assert_eq!((f.get(0x12024) >> 8) & 0xf, 9);
    }
}

#[test]
fn step4_task_image_fault_before_commit_retains_outgoing_task_ownership() {
    // SDM Vol 3A §10.3 step 5 p. 10-10 and notes p. 10-13: TSS
    // residency precedes saves/TR. 80386 PRM §9.8.14.1 p. 9-18:
    // later TSS-driven descriptor checks instead name the new task's IP.
    for model in MODELS {
        for page in [9, 10] {
            let mut f = Fixture::new(model, true, 0);
            f.system_setup();
            // At 9ff0h only static fields occupy the first page. Residency
            // includes that page even though register loads start on the next.
            f.put64(0x12020, step4_descriptor(BASE + 0x9ff0, 0x100, 0x89));
            f.system_page(page, 0, 0);
            let old = f.cpu.cpu.legacy;
            f.assert_pf(
                &[0xea, 0, 0, 0, 0, 0x20, 0],
                BASE + if page == 9 { 0x9ff0 } else { 0xa000 },
                0,
            );
            assert_eq!(f.cpu.cpu.legacy.cr3, old.cr3);
            assert_eq!(f.cpu.cpu.legacy.tr, old.tr);
            assert_eq!(f.get(0x14020), 0);
            assert_eq!((f.get(0x1201c) >> 8) & 0xf, 11);
            assert_eq!((f.get(0x12024) >> 8) & 0xf, 9);
        }
    }
}

#[test]
fn step4_task_descriptor_accessed_writes_set_clean_gdt_page_dirty() {
    // SDM Vol 3A §3.4.5.1 p. 3-13; 80386 PRM §5.2.4.3 p. 5-10;
    // Intel486 PRM §5.3.4.3 pp. 5-21–22: A-bit stores require PTE.D.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.incoming_task(OTHER_PD, false);
        f.put64(0x17008, step4_descriptor(0, 0xfffff, 0xc09a));
        f.put64(0x17010, step4_descriptor(BASE + 0x10000, 0xfffff, 0xc092));
        assert_eq!(f.get(u64::from(OTHER_PT + 6 * 4)) & D, 0);
        f.mapped_instruction(&[0xea, 0, 0, 0, 0, 0x20, 0]);
        f.fetch_at(0x400).unwrap();
        assert_eq!((f.get(0x1700c) >> 8) & 1, 1);
        assert_eq!((f.get(0x17014) >> 8) & 1, 1);
        assert_eq!(f.get(u64::from(OTHER_PT + 6 * 4)) & D, D);
    }
}

#[test]
fn step4_task_image_page_fault_handler_repairs_page_and_retries_switch() {
    // SDM Vol 3A §10.3 step 5 p. 10-10, notes p. 10-13: failed
    // residency checks leave the old task restartable. Descriptor faults
    // after step 13 instead belong to the new task (80386 PRM §9.8.14.1
    // p. 9-18); this handler repairs the incoming image before that point.
    for model in MODELS {
        let mut f = Fixture::with_delivery(
            Vm::new(VmConfig::flat(0x20000)),
            model,
            true,
            0,
            x86jit_core::ExceptionDelivery::InGuest,
        );
        f.incoming_task(OTHER_PD, false);
        let mut image = [0; 0x68];
        f.vm.read_bytes(0x15000, &mut image).unwrap();
        f.vm.write_bytes(0x15fe0, &image[..0x20]).unwrap();
        f.vm.write_bytes(0x16000, &image[0x20..]).unwrap();
        f.put64(0x12020, step4_descriptor(BASE + 0x9fe0, 0x100, 0x89));
        f.system_page(10, 0, 0);
        f.system_page(11, PT, 3); // handler's linear alias of the page table
        f.gate(14, 0x800, 4);
        f.mapped_instruction(&[0xea, 0, 0, 0, 0, 0x20, 0]);
        f.vm.write_bytes(
            u64::from(CODE_FRAME) + 0x800,
            &[
                0xc7, 0x05, 0x28, 0xb0, 0, 0, 3, 0x60, 1, 0, // MOV [b028h],16003h
                0x83, 0xc4, 4,    // discard #PF error code
                0xcf, // IRETD retries the far JMP
            ],
        )
        .unwrap();
        let old_tr = f.cpu.cpu.legacy.tr;
        let old_cr3 = f.cpu.cpu.legacy.cr3;
        f.fetch_at(0x400).unwrap();
        assert_eq!(f.cpu.cpu.rip, 0x800);
        assert_eq!(f.cpu.cpu.legacy.cr2, BASE + 0xa000);
        assert_eq!(f.cpu.cpu.legacy.tr, old_tr);
        assert_eq!(f.cpu.cpu.legacy.cr3, old_cr3);
        assert_eq!(f.cpu.cpu.gpr[0], 0x1122_3344);
        assert_eq!(f.get(0x14020), 0);
        assert_eq!((f.get(0x12024) >> 8) & 0xf, 9);
        f.cpu.interpret_paged_for_test(&f.vm).unwrap(); // repair
        assert_eq!(f.get(u64::from(PT + 10 * 4)), 0x16003);
        f.cpu.interpret_paged_for_test(&f.vm).unwrap(); // discard error
        f.cpu.interpret_paged_for_test(&f.vm).unwrap(); // IRETD
        assert_eq!(f.cpu.cpu.rip, 0x400);
        f.cpu.interpret_paged_for_test(&f.vm).unwrap(); // retry JMP
        assert_eq!(f.cpu.cpu.rip, 0x900);
        assert_eq!(f.cpu.cpu.legacy.tr.selector, 32);
        assert_eq!(f.cpu.cpu.legacy.cr3 & !0xfff, OTHER_PD);
        assert_eq!(f.cpu.cpu.gpr[0], 0xabcdef);
        assert_eq!(f.get(0x14020), 0x407);
        assert_eq!((f.get(0x12024) >> 8) & 0xf, 11);
    }
}

#[test]
fn step4_enter_allocation_page_fault_preserves_frame_and_registers() {
    // SDM Vol 2A §3.3 ENTER pp. 3-295–297: a write at the final SP
    // must be permitted. Intel486 PRM §4.1.3 p. 4-7: supervisor WP.
    for model in MODELS {
        for cpl in [0, 3] {
            for level in [0, 2] {
                for present in [false, true] {
                    if cpl == 0 && present && model == CpuModel::I80386 {
                        continue; // 386 has no supervisor write protection.
                    }
                    let mut f = Fixture::new(model, true, cpl);
                    f.cpu.cpu.legacy.cr0 |= 1 << 16;
                    f.cpu.cpu.gpr[4] = u64::from(OFFSET + 0x100);
                    f.cpu.cpu.gpr[5] = u64::from(OFFSET + 0x80);
                    f.put(u64::from(DATA + 0xfc), 0x1122_3344);
                    f.put(u64::from(DATA + 0xf8), 0x5566_7788);
                    f.put(u64::from(DATA + 0xf4), 0x99aa_bbcc);
                    f.system_page(2, NEXT, if present { 5 } else { 0 });
                    let pushes = if level == 0 { 1 } else { 3 };
                    let final_sp = OFFSET + 0x100 - pushes * 4 - 0x1000;
                    let code = [0xc8, 0, 0x10, level];
                    f.assert_pf(
                        &code,
                        BASE + final_sp,
                        u32::from(present) | 2 | if cpl == 3 { 4 } else { 0 },
                    );
                    assert_eq!(f.get(u64::from(DATA + 0xfc)), 0x1122_3344);
                    assert_eq!(f.get(u64::from(DATA + 0xf8)), 0x5566_7788);
                    assert_eq!(f.get(u64::from(DATA + 0xf4)), 0x99aa_bbcc);
                    assert_eq!(f.get(f.pte()) & D, 0);
                    f.system_page(2, NEXT, 7);
                    f.execute(&code).unwrap();
                    assert_eq!(f.cpu.cpu.gpr[4], u64::from(final_sp));
                    assert_eq!(f.cpu.cpu.gpr[5], u64::from(OFFSET + 0xfc));
                    assert_eq!(f.get(u64::from(DATA + 0xfc)), OFFSET + 0x80);
                }
            }
        }
    }
}

#[test]
fn step4_enter_frame_faults_precede_allocation_fault() {
    // SDM Vol 2A §3.3 ENTER pp. 3-295–297: display reads and frame
    // pushes precede allocation. Levels 0/1 never read a display.
    for model in MODELS {
        for code32 in [false, true] {
            let size = if code32 { 4 } else { 2 };
            for cpl in [0, 3] {
                for level in [0u8, 1, 2, 31] {
                    for missing_push in [false, true] {
                        let mut f = Fixture::new(model, code32, cpl);
                        f.cpu.cpu.gpr[4] = u64::from(OFFSET + 0x100);
                        // For level 31 only the last display read is absent.
                        let bp = OFFSET + 0x2000 + size * u32::from(level.saturating_sub(2));
                        f.cpu.cpu.gpr[5] = u64::from(bp);
                        f.system_page(2, 0, 0); // allocation
                        f.system_page(4, 0, 0); // display
                        f.system_page(5, NEXT, 7);
                        if missing_push {
                            f.system_page(3, 0, 0);
                        }
                        let sentinel = [0xa5; 0x100];
                        f.vm.write_bytes(u64::from(DATA), &sentinel).unwrap();
                        let pushes = if level == 0 { 1 } else { u32::from(level) + 1 };
                        let final_sp = OFFSET + 0x100 - pushes * size - 0x1000;
                        let code = [0xc8, 0, 0x10, level];
                        let (linear, write) = if missing_push {
                            (LINEAR + 0x100 - size, true)
                        } else if level > 1 {
                            (BASE + OFFSET + 0x2000 - size, false)
                        } else {
                            (BASE + final_sp, true)
                        };
                        f.assert_pf(
                            &code,
                            linear,
                            (u32::from(write) << 1) | if cpl == 3 { 4 } else { 0 },
                        );
                        let mut stack = [0; 0x100];
                        f.vm.read_bytes(u64::from(DATA), &mut stack).unwrap();
                        assert_eq!(stack, sentinel);
                        assert_eq!(f.get(f.pte()) & D, 0);
                        if !missing_push && level > 1 {
                            f.system_page(4, 0x19000, 7);
                            f.assert_pf(&code, BASE + final_sp, 2 | if cpl == 3 { 4 } else { 0 });
                            f.vm.read_bytes(u64::from(DATA), &mut stack).unwrap();
                            assert_eq!(stack, sentinel);
                            assert_eq!(f.get(f.pte()) & D, 0);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn step4_enter_allocation_checks_permissions_without_physical_access_or_ad() {
    // SDM Vol 2A §3.3 ENTER pp. 3-295–297: the allocation only checks
    // whether a write would #PF. ASSUMED: no A/D changes for this check
    // without an access (ENTER and Vol 3A §5.8 do not specify them).
    const ALLOCATION_PT: u32 = 0x5000;
    for model in MODELS {
        for code32 in [false, true] {
            let size = if code32 { 4 } else { 2 };
            for level in [0, 1, 2, 31] {
                for device in [false, true] {
                    for preset in [0, A | D | SOFTWARE] {
                        let mut f = Fixture::new(model, code32, 3);
                        // Put allocation and real transfers under separate PDEs
                        // so neither level's A/D assertion can be masked by a push.
                        f.cpu.cpu.legacy.segments[x86jit_core::legacy::SegmentId::Ss as usize]
                            .base = BASE - OFFSET;
                        f.cpu.cpu.gpr[4] = u64::from(OFFSET + 0x100);
                        f.cpu.cpu.gpr[5] = u64::from(OFFSET + 0x80);
                        f.system_page(0, DATA, 7);
                        let pde = f.pde() - 4;
                        let pte = u64::from(ALLOCATION_PT + 0xffc);
                        f.put(pde, ALLOCATION_PT | 7 | preset);
                        f.put(pte, NEXT | 7 | preset);
                        f.vm.unmap(u64::from(NEXT), 0x1000).unwrap();
                        if device {
                            f.vm.map(u64::from(NEXT), 0x1000, Prot::RWX, RegionKind::Trap)
                                .unwrap();
                        }
                        let pushes = if level == 0 { 1 } else { u32::from(level) + 1 };
                        let final_sp = OFFSET + 0x100 - pushes * size - 0x1000;
                        let code = [0xc8, 0, 0x10, level];
                        // A denied hypothetical write still #PFs before a commit,
                        // even though the physical destination has no RAM backing.
                        f.put(pte, NEXT | 5 | preset);
                        f.assert_pf(&code, BASE - OFFSET + final_sp, 7);
                        assert_eq!(f.get(pde), ALLOCATION_PT | 7 | preset);
                        assert_eq!(f.get(pte), NEXT | 5 | preset);
                        f.put(pte, NEXT | 7 | preset);
                        f.execute(&code).unwrap();
                        assert_eq!(f.cpu.cpu.gpr[4], u64::from(final_sp));
                        assert_eq!(f.cpu.cpu.gpr[5], u64::from(OFFSET + 0x100 - size));
                        assert_eq!(
                            f.get(u64::from(DATA + 0x100 - size))
                                & if code32 { u32::MAX } else { 0xffff },
                            OFFSET + 0x80
                        );
                        assert_eq!(f.cpu.cpu.legacy.cr2, BASE - OFFSET + final_sp);
                        assert_eq!(f.get(pde), ALLOCATION_PT | 7 | preset);
                        assert_eq!(f.get(pte), NEXT | 7 | preset);
                    }
                }
            }
        }
    }
}

#[test]
fn step4_enter_nesting_push_fault_precedes_every_frame_write() {
    // SDM Vol 2A §3.3 ENTER pp. 3-295–297: each display push requires
    // write permission even when the allocation's final page is writable.
    for model in MODELS {
        for cpl in [0, 3] {
            for present in [false, true] {
                if cpl == 0 && present && model == CpuModel::I80386 {
                    continue;
                }
                let mut f = Fixture::new(model, true, cpl);
                f.cpu.cpu.legacy.cr0 |= 1 << 16;
                f.cpu.cpu.gpr[4] = u64::from(OFFSET + 4);
                f.cpu.cpu.gpr[5] = u64::from(OFFSET + 0x80);
                f.put(u64::from(DATA), 0x1122_3344);
                f.system_page(0, 0x19000, 7); // allocation ends at offset ff8h
                f.system_page(2, NEXT, if present { 5 } else { 0 });
                f.assert_pf(
                    &[0xc8, 0, 0x20, 2],
                    LINEAR - 4,
                    u32::from(present) | 2 | if cpl == 3 { 4 } else { 0 },
                );
                assert_eq!(f.get(u64::from(DATA)), 0x1122_3344);
                assert_eq!(f.get(f.pte()) & D, 0);
            }
        }
    }
}

#[test]
fn step4_protection_faults_preserve_stack_string_and_x87_destinations() {
    // SDM Vol 3A §5.6.1 pp. 5-33–35; 80386 PRM §§6.4.1–6.4.2
    // pp. 6-20–22: CPL3 requires U/S, and writes additionally require R/W.
    for model in MODELS {
        for family in 0..16 {
            let mut f = Fixture::new(model, true, 3);
            f.cpu.set_fpu_model(x86jit_core::FpuModel::Present);
            f.cpu.cpu.gpr[4] = u64::from(OFFSET + 0x100);
            f.cpu.cpu.gpr[5] = u64::from(OFFSET + 0x100);
            f.cpu.cpu.gpr[6] = u64::from(OFFSET + 0x100);
            f.cpu.cpu.gpr[7] = u64::from(OFFSET + 0x200);
            f.cpu.cpu.gpr[1] = 2;
            let (code, writing, offset) = match family {
                0 => (vec![0x50], true, 0xfc),
                1 => (vec![0x58], false, 0x100),
                2 => (vec![0x60], true, 0xfc),
                3 => (vec![0x61], false, 0x100),
                4 => (vec![0xc8, 0, 0, 2], true, 0xfc),
                5 => (vec![0xc9], false, 0x100),
                6 => (vec![0xe8, 0, 0, 0, 0], true, 0xfc),
                7 => (vec![0xc3], false, 0x100),
                8 => (vec![0xcf], false, 0x100),
                9 => (vec![0xf3, 0xa5], true, 0x200),
                10 => (vec![0xf3, 0xab], true, 0x200),
                11 => (vec![0xf3, 0xad], false, 0x100),
                12 => (vec![0xf3, 0xa7], false, 0x100),
                13 => (vec![0xf3, 0xaf], false, 0x200),
                14 => (f.memory_code(&[0xd9], 4, OFFSET + 0x100, 4), false, 0x100),
                _ => (f.memory_code(&[0xdd], 6, OFFSET + 0x100, 4), true, 0x100),
            };
            f.tables(7, if writing { 5 } else { 3 });
            f.assert_pf(&code, LINEAR + offset, if writing { 7 } else { 5 });
            assert_eq!(f.get(u64::from(DATA)), 0);
            assert_eq!(f.get(f.pte()) & D, 0);
        }
    }
}

#[test]
fn step4_zero_count_rep_never_translates_operands_or_bitmap() {
    // SDM Vol 2B REP Operation p. 4-564; 80386 PRM §17 REP:
    // a zero count performs no iterations, memory references or port access.
    for model in MODELS {
        for opcode in [0xa5, 0xab, 0xad, 0xa7, 0xaf, 0x6d, 0x6f] {
            let mut f = Fixture::new(model, true, 3);
            f.cpu.cpu.gpr[1] = 0;
            f.tables(0, 0);
            let before = f.cpu.cpu.gpr;
            f.execute(&[0xf3, opcode]).unwrap();
            assert_eq!(f.cpu.cpu.gpr, before);
            assert_eq!(f.get(f.pde()) & A, 0);
            assert_eq!(f.cpu.cpu.legacy.cr2, 0xdead_beef);
        }
    }
}

#[test]
fn step4_far_call_and_return_frames_translate_every_slot() {
    // 80386 PRM §17 CALL/RET pp. 17-40–42, 17-141–145;
    // paging design §3.4: the whole frame precedes any operand commit.
    for model in MODELS {
        for size in [2, 4] {
            let mut f = Fixture::new(model, true, 0);
            f.system_setup();
            let mut code = Vec::new();
            if size == 2 {
                code.push(0x66);
            }
            code.push(0x9a);
            code.extend(0x800_u32.to_le_bytes()[..size as usize].iter());
            code.extend(8_u16.to_le_bytes());
            // First pushed slot mapped, second on an absent preceding page.
            f.system_page(3, 0, 0);
            f.system_page(4, NEXT, 7);
            f.cpu.cpu.gpr[4] = u64::from(OFFSET + 0x1000 + size);
            f.assert_pf(&code, LINEAR + 0x1000 - size, 2);
            assert_eq!(f.get(u64::from(NEXT)), 0);
            f.system_page(3, DATA, 7);
            f.execute(&code).unwrap();
            assert_eq!(f.cpu.cpu.rip, 0x800);
            let mut ret = Vec::new();
            if size == 2 {
                ret.push(0x66);
            }
            ret.push(0xcb);
            f.execute(&ret).unwrap();
            assert_eq!(f.cpu.cpu.rip, 0x400 + code.len() as u64);
            assert_eq!(f.cpu.cpu.gpr[4], u64::from(OFFSET + 0x1000 + size));
        }
    }
}

#[test]
fn step4_equal_task_cr3_load_discards_private_fetch_proofs() {
    // Design §3.2 conservative task-load invalidation, including equal CR3;
    // 80386 PRM §§5.2.5, 7.5 pp. 5-11, 7-8; SDM Vol 3A §5.10.4 p. 5-48.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.incoming_task(OTHER_PD, false);
        f.put(0x1501c, PD as u32 | 0x18);
        f.mapped_instruction(&[0xea, 0, 0, 0, 0, 0x20, 0]);
        f.fetch_at(0x400).unwrap();
        assert_eq!(
            f.cpu.cpu.legacy.cr3,
            PD as u32 | if model == CpuModel::I80386 { 0 } else { 0x18 }
        );
        assert_eq!(f.cpu.cpu.rip, 0x900);
        assert_eq!(f.cpu.paging_cache_for_test().1, 0);
    }
}

#[test]
fn step4_comparison_page_fault_restores_pre_rep_flags() {
    // SDM Vol 2B §4.3 REPE/REPZ p. 4-565 and REPNE/REPNZ p. 4-567:
    // completed comparison iterations retain indices/count; a fault restores
    // the flags from before the whole repeated instruction.
    for model in MODELS {
        for opcode in [0xa7, 0xaf] {
            let mut f = Fixture::new(model, true, 0);
            f.cpu.cpu.flags.cf = true;
            f.cpu.cpu.flags.of = true;
            f.cpu.cpu.flags.zf = false;
            let flags = f.cpu.cpu.flags;
            f.cpu.cpu.gpr[0] = 42;
            f.cpu.cpu.gpr[1] = 2;
            f.cpu.cpu.gpr[6] = u64::from(OFFSET + 0xffc);
            f.cpu.cpu.gpr[7] = u64::from(OFFSET + 0xffc);
            f.put(u64::from(DATA + 0xffc), 42);
            f.mapped_instruction(&[0xf3, opcode]);
            f.fetch_at(0x400).unwrap();
            assert!(f.cpu.cpu.flags.zf);
            let exit = f.cpu.interpret_paged_for_test(&f.vm).unwrap_err();
            assert!(
                matches!(exit, Exit::Exception { vector: 14, .. }),
                "{exit:?}"
            );
            assert_eq!(f.cpu.cpu.flags, flags);
            assert_eq!(f.cpu.cpu.gpr[1], 1);
            assert_eq!(f.cpu.cpu.gpr[7], u64::from(OFFSET + 0x1000));
        }
    }
}

#[test]
fn step4_handler_fetch_fault_is_separate_from_completed_delivery() {
    // SDM Vol 3A §7.15 Event 8 p. 7-37: faults on instruction prefetch
    // are outside the nested-delivery table; §5.7 pp. 5-37–38 defines CR2.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.system_setup();
        f.gate(0x20, 0x1800, 4);
        f.mapped_instruction(&[0xcd, 0x20]);
        f.fetch_at(0x400).unwrap();
        assert_eq!(f.cpu.cpu.rip, 0x1800);
        assert_eq!(f.cpu.cpu.gpr[4], u64::from(OFFSET + 0x800 - 12));
        let exit = f.cpu.interpret_paged_for_test(&f.vm).unwrap_err();
        assert!(
            matches!(exit, Exit::Exception { vector: 14, .. }),
            "{exit:?}"
        );
        assert_eq!(f.cpu.cpu.legacy.cr2, 0x1800);
        assert_eq!(exit.fault_metadata().unwrap().saved_rip(), 0x1800);
    }
}

#[test]
fn step4_ins_handoff_retains_cross_page_destination_plan() {
    // Paging design §3.4; 80386 PRM §14.6 pp. 14-4–5: one iteration
    // owns its destination before the irreversible port input. Page-table
    // edits during the handoff cannot redirect that already-preflighted store.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.system_page(4, NEXT, 7);
        f.cpu.cpu.gpr[1] = 2;
        f.cpu.cpu.gpr[7] = u64::from(OFFSET + 0xffe);
        f.mapped_instruction(&[0xf3, 0x6d]);
        let exit = f.fetch_at(0x400).unwrap_err();
        assert!(matches!(exit, Exit::PortIo { .. }), "{exit:?}");
        assert_eq!(f.get(f.pte()) & D, D);
        assert_eq!(f.get(f.pte() + 4) & D, D);
        f.put(f.pte(), 0);
        f.put(f.pte() + 4, 0);
        f.cpu.invalidate_paging();
        f.cpu.complete_port_in(0x1122_beef);
        f.cpu.interpret_paged_for_test(&f.vm).unwrap();
        assert_eq!(f.get(u64::from(DATA + 0xffc)) >> 16, 0xbeef);
        assert_eq!(f.get(u64::from(NEXT)) & 0xffff, 0x1122);
        assert_eq!(f.get(f.pte()), 0);
        assert_eq!(f.get(f.pte() + 4), 0);
        assert_eq!(f.cpu.cpu.gpr[7], u64::from(OFFSET + 0x1002));
        assert_eq!(f.cpu.cpu.gpr[1], 1);
    }
}

#[test]
fn step4_equal_task_cr3_load_observes_pte_written_by_outgoing_save() {
    // 80386 PRM §§5.2.5, 7.5 pp. 5-11, 7-8; SDM Vol 3A §10.3
    // pp. 10-10–11: save with the outgoing translation, reload CR3, then
    // translate incoming descriptors. Engine policy flushes equal CR3 too.
    for model in MODELS {
        for v86 in [false, true] {
            for incoming_pte in [0, 0x18003] {
                let mut f = Fixture::new(model, true, 0);
                f.incoming_task(OTHER_PD, false);
                f.put(0x1501c, PD as u32 | 0x18);
                f.put(0x15024, if v86 { 0x20002 } else { 2 });
                // TSS.EIP aliases PTE[6]. Saving next IP 17407h remaps the GDT
                // from frame 12000h to 17000h, where DS has a different base.
                f.cpu.cpu.legacy.tr.cache.base = BASE + 0x7ff8;
                f.system_page(8, PT, 3);
                // Preserve PTE[8]; the ECX save removes or redirects TSS PTE[9].
                f.cpu.cpu.gpr[0] = u64::from(PT | 3);
                f.cpu.cpu.gpr[1] = incoming_pte;
                f.code_mapping(0x17000, CODE_FRAME);
                f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0xea, 0, 0, 0, 0, 0x20, 0])
                    .unwrap();
                f.fetch_at(0x17400).unwrap();
                assert_eq!(f.get(u64::from(PT + 6 * 4)) & 0xfffff000, 0x17000);
                assert_eq!(f.get(u64::from(PT + 9 * 4)), incoming_pte as u32);
                assert_eq!(f.cpu.cpu.gpr[0], 0xabcdef);
                assert_eq!(f.cpu.cpu.task_io_map_base, 0x80);
                assert_eq!(
                    f.cpu.cpu.rflags_system & 0x20000,
                    if v86 { 0x20000 } else { 0 }
                );
                assert_eq!(f.cpu.cpu.legacy.cpl, if v86 { 3 } else { 0 });
                assert_eq!(
                    f.cpu.cpu.legacy.segments[1].base,
                    if v86 { 0x100 } else { BASE + 0x10000 }
                );
                assert_eq!(f.cpu.cpu.rip, 0x900);
                assert_eq!(f.cpu.cpu.legacy.tr.selector, 32);
            }
        }
    }
}

fn public_step(f: &mut Fixture) -> Exit {
    let run = f.cpu.run_steps(&f.vm, 1, StepOptions::default()).unwrap();
    match run.stop() {
        StepStop::Exit(exit) => exit.clone(),
        other => panic!("unexpected paged stop: {other:?}"),
    }
}

#[test]
fn public_split_device_retry_preserves_snapshot_and_transfer_order() {
    // 80386 PRM §9.8.14 pp. 9-17–19: restart at the original instruction.
    // Staged RAM and visit-ledger ordering are engine policy (paging §3.4).
    for model in MODELS {
        for devices in 1..=3 {
            for operation in 0..4 {
                let mut f = Fixture::new(model, true, 0);
                f.code_mapping(0, CODE_FRAME);
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
                let code = if operation >= 2 {
                    let mut code = if operation == 3 { vec![0xf0] } else { vec![] };
                    code.extend([0x83, 0x05]);
                    code.extend((OFFSET + 0xffe).to_le_bytes());
                    code.push(1);
                    code
                } else {
                    f.code(operation == 1, OFFSET + 0xffe)
                };
                f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
                    .unwrap();
                let mut reads = Vec::new();
                let mut writes = Vec::new();
                for attempt in 0..8 {
                    match public_step(&mut f) {
                        Exit::MmioRead { addr, size, .. } => {
                            assert_eq!(size, 2);
                            assert_eq!(
                                f.cpu.pending_mmio_linear_address(),
                                Some(
                                    LINEAR
                                        + if addr == u64::from(DATA + 0xffe) {
                                            0xffe
                                        } else {
                                            0x1000
                                        }
                                )
                            );
                            assert!(!reads.contains(&addr), "duplicate device read");
                            reads.push(addr);
                            f.cpu
                                .complete_mmio_read(if addr == u64::from(DATA + 0xffe) {
                                    0x1234
                                } else {
                                    0x4321
                                });
                        }
                        Exit::MmioWrite {
                            addr, size, value, ..
                        } => {
                            assert_eq!(size, 2);
                            assert_eq!(
                                f.cpu.pending_mmio_linear_address(),
                                Some(
                                    LINEAR
                                        + if addr == u64::from(DATA + 0xffe) {
                                            0xffe
                                        } else {
                                            0x1000
                                        }
                                )
                            );
                            assert!(!writes.contains(&addr), "duplicate device write");
                            assert_eq!(
                                value,
                                if operation == 1 {
                                    if addr == u64::from(DATA + 0xffe) {
                                        0xccdd
                                    } else {
                                        0xaabb
                                    }
                                } else if addr == u64::from(DATA + 0xffe) {
                                    0x1235
                                } else {
                                    0x4321
                                }
                            );
                            writes.push(addr);
                            // Operand RAM remains at entry until all device pieces finish.
                            if devices & 1 == 0 {
                                assert_eq!(f.get(u64::from(DATA + 0xffc)), 0x1234_5678);
                            }
                            f.cpu.complete_mmio_write();
                        }
                        Exit::BudgetExhausted => break,
                        exit => panic!("{model:?}/{devices}/{operation}/{attempt}: {exit:?}"),
                    }
                    assert!(attempt < 7, "retry did not finish");
                }
                assert_eq!(f.cpu.pending_mmio_linear_address(), None);
                let device_count = (devices as u32).count_ones() as usize;
                assert_eq!(reads.len(), if operation == 1 { 0 } else { device_count });
                assert_eq!(writes.len(), if operation == 0 { 0 } else { device_count });
                assert_eq!(f.cpu.cpu.rip, 0x400 + code.len() as u64);
                if devices & 1 == 0 {
                    assert_eq!(
                        f.get(u64::from(DATA + 0xffc)),
                        [0x1234_5678, 0xccdd_5678, 0x1235_5678, 0x1235_5678][operation]
                    );
                }
                if devices == 1 && operation >= 2 {
                    assert_eq!(f.get(u64::from(NEXT)), 0x8765_4321);
                }
                if operation == 0 {
                    assert_eq!(f.cpu.cpu.gpr[0], 0x4321_1234);
                }
            }
        }
    }
}

#[test]
fn public_device_region_split_and_multi_address_push() {
    // Paging §3.4: physical-region splits and explicit/implicit operands use
    // the same preflight and retained visit ledger.
    for model in MODELS {
        for device_first in [false, true] {
            let mut f = Fixture::new(model, true, 0);
            f.code_mapping(0, CODE_FRAME);
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
            let code = f.code(false, OFFSET + 2);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
                .unwrap();
            let ram = u64::from(DATA + if device_first { 4 } else { 0 });
            f.put(ram, 0x1234_5678);
            let exit = public_step(&mut f);
            assert!(matches!(exit, Exit::MmioRead { size: 2, .. }), "{exit:?}");
            f.cpu.complete_mmio_read(0xabcd);
            assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
            assert_eq!(
                f.cpu.cpu.gpr[0],
                if device_first {
                    0x5678_abcd
                } else {
                    0xabcd_1234
                }
            );
        }
        for source_device in [false, true] {
            let mut f = Fixture::new(model, true, 0);
            f.code_mapping(0, CODE_FRAME);
            f.put(f.pte() + 4, NEXT | 7);
            f.cpu.cpu.gpr[4] = u64::from(OFFSET + 0x1004);
            f.put(u64::from(DATA), 0x1234_5678);
            let page = if source_device { DATA } else { NEXT };
            f.vm.unmap(u64::from(page), 0x1000).unwrap();
            f.vm.map(u64::from(page), 0x1000, Prot::RWX, RegionKind::Trap)
                .unwrap();
            let mut code = vec![0xff, 0x35]; // PUSH dword [OFFSET]
            code.extend(OFFSET.to_le_bytes());
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
                .unwrap();
            let exit = public_step(&mut f);
            if source_device {
                assert!(
                    matches!(exit, Exit::MmioRead { addr, size: 4, .. } if addr == u64::from(DATA)),
                    "{exit:?}"
                );
                f.cpu.complete_mmio_read(0x1234_5678);
            } else {
                assert!(
                    matches!(exit, Exit::MmioWrite { addr, size: 4, value: 0x1234_5678, .. } if addr == u64::from(NEXT)),
                    "{exit:?}"
                );
                // The original RAM operand snapshot survives a host edit.
                f.put(u64::from(DATA), 0xaabb_ccdd);
                f.cpu.complete_mmio_write();
            }
            assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
            assert_eq!(f.cpu.cpu.gpr[4], u64::from(OFFSET + 0x1000));
            if source_device {
                assert_eq!(f.get(u64::from(NEXT)), 0x1234_5678);
            }
        }
    }
}

#[test]
fn public_retry_keeps_translation_after_page_table_edit_and_invalidation() {
    let mut f = Fixture::new(CpuModel::I80386, true, 0);
    f.code_mapping(0, CODE_FRAME);
    f.put(f.pte() + 4, NEXT | 7);
    f.vm.unmap(u64::from(NEXT), 0x1000).unwrap();
    f.vm.map(u64::from(NEXT), 0x1000, Prot::RWX, RegionKind::Trap)
        .unwrap();
    let code = f.code(true, OFFSET + 0xffe);
    f.cpu.cpu.gpr[0] = 0x1234_5678;
    f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
        .unwrap();
    assert!(
        matches!(public_step(&mut f), Exit::MmioWrite { addr, value: 0x1234, size: 2, .. } if addr == u64::from(NEXT))
    );
    f.put(f.pte(), OTHER_FRAME | 7);
    f.put(f.pte() + 4, OTHER_FRAME | 7);
    f.cpu.invalidate_paging();
    f.cpu.complete_mmio_write();
    assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
    assert_eq!(f.get(u64::from(DATA + 0xffc)) >> 16, 0x5678);
    assert_eq!(f.get(u64::from(OTHER_FRAME)), 0);
}

#[test]
fn public_sparse_and_host_ram_alias_smc_and_restore() {
    // 80386 PRM §5.2 Fig. 5-9 pp. 5-8–9: linear aliases select the same
    // physical frame. Paging §3.5/3.7: physical SMC and quiescent restore policy.
    for model in MODELS {
        for host in [false, true] {
            let vm = if host {
                let mut storage = vec![0u8; 0x20000].into_boxed_slice();
                let ptr = storage.as_mut_ptr();
                let len = storage.len();
                let mut storage = Some(storage);
                let ram = x86jit_core::HostRam {
                    ptr,
                    len,
                    guest_base: 0,
                    dtor: Box::new(move |_, _| drop(storage.take())),
                    protect: None,
                };
                Vm::with_backend_host_ram(
                    VmConfig::flat(0x20000),
                    Box::new(x86jit_core::InterpreterBackend),
                    ram,
                )
                .unwrap()
            } else {
                Vm::sparse(VmConfig::reserved(0x20000)).unwrap()
            };
            let mut f = Fixture::with_vm(vm, model, true, 0);
            f.code_mapping(0, CODE_FRAME);
            f.code_mapping(0x4000, CODE_FRAME);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0xb0, 0x11])
                .unwrap();
            for ip in [0x400, 0x4400] {
                f.cpu.cpu.rip = ip;
                assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
                assert_eq!(f.cpu.cpu.gpr[0] & 0xff, 0x11);
            }
            f.cpu.cpu.rip = 0x400;
            let snapshot = f.cpu.cpu.clone();
            let mut ram = vec![0; 0x20000];
            for (page, bytes) in ram.chunks_mut(0x1000).enumerate() {
                f.vm.read_bytes((page * 0x1000) as u64, bytes).unwrap();
            }
            // Write through a third linear alias using the normal paged store.
            f.put(f.pte(), CODE_FRAME | 7);
            let mut code = vec![0xc6, 0x05];
            code.extend((OFFSET + 0x401).to_le_bytes());
            code.push(0x22);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x500, &code)
                .unwrap();
            f.cpu.cpu.rip = 0x500;
            assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
            for ip in [0x400, 0x4400] {
                f.cpu.cpu.rip = ip;
                assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
                assert_eq!(f.cpu.cpu.gpr[0] & 0xff, 0x22);
            }
            for (page, bytes) in ram.chunks(0x1000).enumerate() {
                f.vm.write_bytes((page * 0x1000) as u64, bytes).unwrap();
            }
            f.cpu.cpu = snapshot;
            f.cpu.invalidate_paging();
            assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
            assert_eq!(f.cpu.cpu.gpr[0] & 0xff, 0x11);
        }
    }
}

#[test]
fn public_paging_a20_gates_tables_and_final_frames_after_translation() {
    // 80386 PRM §5.2 pp. 5-7–10 defines the translation; the board A20
    // gate applies to its physical traffic (paging design §3.6).
    for model in MODELS {
        let mut f = Fixture::with_vm(
            Vm::sparse(VmConfig::reserved(0x200000)).unwrap(),
            model,
            true,
            0,
        );
        f.cpu.cpu.legacy.cr3 = PD as u32 | 0x100000;
        f.map_code(PD as u32, CODE_PT | 0x100000, 0, CODE_FRAME | 0x100000, 7);
        f.map_code(
            PD as u32 | 0x100000,
            OTHER_PT | 0x100000,
            0,
            OTHER_FRAME | 0x100000,
            7,
        );
        // A20=0 selects the low directory, then gates its high table/frame.
        f.put(u64::from(CODE_PT), CODE_FRAME | 0x100007);
        f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0xb0, 0x11])
            .unwrap();
        f.vm.write_bytes(u64::from(OTHER_FRAME) + 0x100400, &[0xb0, 0x22])
            .unwrap();
        for (enabled, expected) in [(true, 0x22), (false, 0x11), (true, 0x22)] {
            f.cpu.set_a20_enabled(enabled);
            f.cpu.cpu.rip = 0x400;
            assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
            assert_eq!(f.cpu.cpu.gpr[0] & 0xff, expected);
        }
        // Linear bit 20 still selects PTE[256] with A20 off; only its
        // translated table/frame bus addresses pass through the board gate.
        f.put(u64::from(CODE_PT + 256 * 4), OTHER_FRAME | 0x100007);
        f.vm.write_bytes(u64::from(OTHER_FRAME) + 0x400, &[0xb0, 0x33])
            .unwrap();
        f.cpu.set_a20_enabled(false);
        f.cpu.cpu.rip = 0x100400;
        assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
        assert_eq!(f.cpu.cpu.gpr[0] & 0xff, 0x33);
        assert_eq!(f.get(u64::from(CODE_PT)) & A, A);
        assert_eq!(f.get(u64::from(OTHER_PT) + 0x100000) & A, A);
    }
}

#[test]
fn public_paging_table_device_refusal_and_286_rejection() {
    // Q1 is an engine limitation, not a hardware page fault. 80386 PRM
    // §5.2.3 pp. 5-8–9: CR3 and PDE table addresses are physical.
    for model in MODELS {
        for directory in [false, true] {
            let mut f = Fixture::new(model, true, 0);
            f.code_mapping(0, CODE_FRAME);
            let page = if directory { PD } else { u64::from(CODE_PT) };
            f.vm.unmap(page, 0x1000).unwrap();
            f.vm.map(page, 0x1000, Prot::RWX, RegionKind::Trap).unwrap();
            assert!(matches!(
                public_step(&mut f),
                Exit::UnsupportedSystemContext {
                    operation: UnsupportedSystemOperation::PagingTableStorage,
                    ..
                }
            ));
            assert_eq!(f.cpu.cpu.legacy.cr2, 0xdead_beef);
            assert!(!f.cpu.cpu.mmio_retry_active);
        }
    }
    let mut f = Fixture::new(CpuModel::I80286, false, 0);
    // Keep the 286 storage image valid apart from the requested PG bit.
    for segment in &mut f.cpu.cpu.legacy.segments {
        segment.base = 0;
        segment.limit = 0xffff;
    }
    assert!(matches!(
        public_step(&mut f),
        Exit::UnsupportedSystemContext {
            operation: UnsupportedSystemOperation::InvalidLegacyContext(
                x86jit_core::MachineContextError::PagingRequires386
            ),
            ..
        }
    ));
}

#[test]
fn public_device_retry_survives_host_stop_request_and_queued_a20_change() {
    let mut f = Fixture::new(CpuModel::I80386, true, 0);
    f.code_mapping(0, CODE_FRAME);
    f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
    f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Trap)
        .unwrap();
    let mut code = vec![0x83, 0x05];
    code.extend(OFFSET.to_le_bytes());
    code.push(1);
    f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
        .unwrap();
    assert!(matches!(public_step(&mut f), Exit::MmioRead { .. }));
    f.cpu.complete_mmio_read(41);
    assert!(matches!(
        public_step(&mut f),
        Exit::MmioWrite { value: 42, .. }
    ));
    f.cpu.set_a20_enabled(false);
    assert_eq!(f.cpu.cpu.legacy.a20_enabled, 1);
    f.cpu.exit_request().request_exit();
    let stop = public_step(&mut f);
    assert!(
        matches!(stop, Exit::Interrupted | Exit::MmioWrite { value: 42, .. }),
        "{stop:?}"
    );
    f.cpu.complete_mmio_write();
    let stop = public_step(&mut f);
    assert!(
        matches!(stop, Exit::BudgetExhausted | Exit::Interrupted),
        "{stop:?}"
    );
    assert_eq!(f.cpu.cpu.legacy.a20_enabled, 0);
    assert_eq!(f.cpu.cpu.rip, 0x400 + code.len() as u64);
}

#[test]
fn public_paged_cache_operations_preserve_flags() {
    // Intel486 PRM §26 INVD/WBINVD pp. 26-171, 26-281: privileged
    // cache operations preserve flags, independently of paging.
    for model in [CpuModel::I80486Dx, CpuModel::I80486DxCpuid] {
        for code in [[0x0f, 0x08], [0x0f, 0x09]] {
            let mut f = Fixture::new(model, true, 0);
            f.code_mapping(0, CODE_FRAME);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
                .unwrap();
            f.cpu.cpu.flags.cf = true;
            let flags = f.cpu.cpu.flags;
            assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
            assert_eq!(f.cpu.cpu.flags, flags);
        }
    }
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
#[path = "../../../x86jit-cranelift/src/unwind_test_signal.rs"]
mod paged_signal;

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod paged_host_fault {
    use super::paged_signal as signal;
    use super::*;

    #[test]
    fn public_host_ram_fault_keeps_linear_and_physical_evidence() {
        // Physical backing failure is a bus exit (paging §3.6), not the
        // not-present/protection #PF of 80386 PRM §9.8.14 pp. 9-17–19.
        let len = 0x20000;
        // SAFETY: HostRam owns the complete anonymous span until VM drop;
        // protection changes cover one aligned host page within that span.
        assert_eq!(
            signal::child(|| unsafe {
                signal::install();
                let ptr = libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                );
                assert_ne!(ptr, libc::MAP_FAILED);
                let ram = x86jit_core::HostRam {
                    ptr: ptr.cast(),
                    len,
                    guest_base: 0,
                    dtor: Box::new(|p, n| {
                        libc::munmap(p.cast(), n);
                    }),
                    protect: None,
                };
                let mut vm = Vm::with_backend_host_ram(
                    VmConfig::flat(len as u64),
                    Box::new(x86jit_core::InterpreterBackend),
                    ram,
                )
                .unwrap();
                vm.mem.set_host_fault_recovery(true);
                let mut f = Fixture::with_vm(vm, CpuModel::I80386, true, 0);
                f.code_mapping(0, CODE_FRAME);
                f.put(u64::from(DATA), 0x1234_5678);
                let code = f.code(false, OFFSET);
                f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
                    .unwrap();
                let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
                let data = ptr.cast::<u8>().add(DATA as usize).cast();
                assert_eq!(libc::mprotect(data, page, libc::PROT_NONE), 0);
                let exit = public_step(&mut f);
                assert!(matches!(exit, Exit::UnmappedMemory { .. }), "{exit:?}");
                let metadata = exit.fault_metadata().unwrap();
                assert!(
                    matches!(metadata.linear_address(), Knowledge::Known(a) if a.value() == u64::from(LINEAR))
                );
                assert_eq!(
                    metadata.physical_address(),
                    Knowledge::Known(u64::from(DATA))
                );
                assert_eq!(f.cpu.cpu.legacy.cr2, 0xdead_beef);
                assert_eq!(
                    libc::mprotect(data, page, libc::PROT_READ | libc::PROT_WRITE),
                    0
                );
                assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
                assert_eq!(f.cpu.cpu.gpr[0], 0x1234_5678);

                // Revoke backing only while a device write is held. The later
                // RAM commit fault retains that consumed acknowledgement.
                f.put(f.pte() + 4, NEXT | 7);
                f.put(u64::from(DATA + 0xffc), 0x1111_2222);
                f.vm.unmap(u64::from(NEXT), 0x1000).unwrap();
                f.vm.map(u64::from(NEXT), 0x1000, Prot::RWX, RegionKind::Trap)
                    .unwrap();
                let code = f.code(true, OFFSET + 0xffe);
                f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
                    .unwrap();
                f.cpu.cpu.rip = 0x400;
                assert!(
                    matches!(public_step(&mut f), Exit::MmioWrite { addr, size: 2, value: 0x1234, .. } if addr == u64::from(NEXT))
                );
                assert_eq!(libc::mprotect(data, page, libc::PROT_NONE), 0);
                f.cpu.complete_mmio_write();
                let exit = public_step(&mut f);
                assert!(matches!(exit, Exit::UnmappedMemory { .. }), "{exit:?}");
                assert_eq!(f.cpu.cpu.rip, 0x400);
                assert!(!f.cpu.cpu.pending_mmio_write);
                let metadata = exit.fault_metadata().unwrap();
                assert_eq!(metadata.saved_rip(), 0x400);
                assert!(
                    matches!(metadata.linear_address(), Knowledge::Known(a) if a.value() == u64::from(LINEAR + 0xffe))
                );
                assert!(!metadata.progress().acknowledged_transfers().is_empty());
                assert_eq!(
                    libc::mprotect(data, page, libc::PROT_READ | libc::PROT_WRITE),
                    0
                );
                assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
                assert_eq!(f.get(u64::from(DATA + 0xffc)), 0x5678_2222);
            }),
            0
        );
    }
}

#[test]
fn public_enter_device_displays_and_frame_complete_once() {
    // SDM Vol 2A §3.3 ENTER pp. 3-295–297: frame/display stack operations.
    // Retained display snapshots and transfer order follow paging §3.4 policy.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.code_mapping(0, CODE_FRAME);
        f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
        f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Trap)
            .unwrap();
        f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0xc8, 0, 0, 2])
            .unwrap();
        f.cpu.cpu.gpr[4] = u64::from(OFFSET + 0x100);
        f.cpu.cpu.gpr[5] = u64::from(OFFSET + 0x80);
        assert!(
            matches!(public_step(&mut f), Exit::MmioRead { addr, size: 4, .. } if addr == u64::from(DATA + 0x7c))
        );
        f.cpu.complete_mmio_read(0x1234_5678);
        for (offset, value) in [
            (0xfc, OFFSET + 0x80),
            (0xf8, 0x1234_5678),
            (0xf4, OFFSET + 0xfc),
        ] {
            let exit = public_step(&mut f);
            assert!(
                matches!(exit, Exit::MmioWrite { addr, size: 4, value: actual, .. } if addr == u64::from(DATA + offset) && actual == u64::from(value)),
                "{exit:?}"
            );
            f.cpu.complete_mmio_write();
        }
        assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
        assert_eq!(f.cpu.cpu.gpr[4], u64::from(OFFSET + 0xf4));
        assert_eq!(f.cpu.cpu.gpr[5], u64::from(OFFSET + 0xfc));
    }
}

#[test]
fn public_aliased_device_stores_have_separate_completion_owners() {
    // 80386 PRM §5.2 pp. 5-8–9: different linear pages may select one
    // physical frame. Paging §3.4: an answer never belongs to a later store.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.code_mapping(0, CODE_FRAME);
        f.put(f.pte() + 4, DATA | 7);
        f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
        f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Trap)
            .unwrap();
        f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0xab])
            .unwrap(); // STOSD
        for (offset, value) in [(OFFSET, 0x1234), (OFFSET + 0x1000, 0x5678)] {
            f.cpu.cpu.rip = 0x400;
            f.cpu.cpu.gpr[7] = u64::from(offset);
            f.cpu.cpu.gpr[0] = value;
            let exit = public_step(&mut f);
            assert!(
                matches!(exit, Exit::MmioWrite { addr, size: 4, value: actual, .. } if addr == u64::from(DATA) && actual == value),
                "{exit:?}"
            );
            f.cpu.complete_mmio_write();
            assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
        }
    }
}

#[test]
fn public_reset_abandons_paged_device_plan_and_snapshot() {
    let mut f = Fixture::new(CpuModel::I80386, true, 0);
    f.code_mapping(0, CODE_FRAME);
    f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
    f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Trap)
        .unwrap();
    let mut code = vec![0x83, 0x05];
    code.extend(OFFSET.to_le_bytes());
    code.push(1);
    f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
        .unwrap();
    assert!(matches!(public_step(&mut f), Exit::MmioRead { .. }));
    f.cpu.complete_mmio_read(41);
    assert!(matches!(
        public_step(&mut f),
        Exit::MmioWrite { value: 42, .. }
    ));
    f.cpu.reset_execution_state(Default::default());
    assert!(matches!(public_step(&mut f), Exit::MmioRead { .. }));
    f.cpu.complete_mmio_read(9);
    assert!(matches!(
        public_step(&mut f),
        Exit::MmioWrite { value: 10, .. }
    ));
    f.cpu.complete_mmio_write();
    assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
}

#[test]
fn public_device_write_acknowledgement_invalidates_declared_code_alias() {
    // Paging §3.5: the acknowledged physical bus write invalidates all
    // registered source aliases even when the embedder supplies equal bytes.
    let mut f = Fixture::new(CpuModel::I80386, true, 0);
    f.code_mapping(0, CODE_FRAME);
    f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
    f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Trap)
        .unwrap();
    f.vm.mem
        .declare_code_alias(u64::from(DATA), u64::from(CODE_FRAME), 0x1000)
        .unwrap();
    let code = f.code(true, OFFSET + 0x600);
    f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
        .unwrap();
    f.cpu.cpu.gpr[0] = 0; // equal-byte device write at the corresponding alias offset
    assert!(matches!(public_step(&mut f), Exit::MmioWrite { .. }));
    let before = f.cpu.paging_cache_for_test();
    f.cpu.complete_mmio_write();
    assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
    f.cpu.cpu.rip = 0x400;
    assert!(matches!(public_step(&mut f), Exit::MmioWrite { .. }));
    let after = f.cpu.paging_cache_for_test();
    assert!(after.0 > before.0);
    assert!(after.2 > before.2);
}

#[test]
fn public_pg_clear_abandons_old_device_completion_before_unpaged_access() {
    let mut f = Fixture::new(CpuModel::I80386, true, 0);
    f.code_mapping(0, CODE_FRAME);
    f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
    f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Trap)
        .unwrap();
    let code = f.code(true, OFFSET);
    f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
        .unwrap();
    f.cpu.cpu.gpr[0] = 1;
    assert!(matches!(
        public_step(&mut f),
        Exit::MmioWrite { value: 1, .. }
    ));
    f.cpu.complete_mmio_write();
    f.cpu.cpu.legacy.cr0 = 1;
    f.cpu.cpu.legacy.segments[1].base = 0;
    f.cpu.cpu.gpr[0] = 2;
    let mut code = vec![0xa3];
    code.extend(DATA.to_le_bytes());
    f.vm.write_bytes(0x400, &code).unwrap();
    assert!(
        matches!(public_step(&mut f), Exit::MmioWrite { addr, value: 2, .. } if addr == u64::from(DATA))
    );
    assert_eq!(f.cpu.pending_mmio_linear_address(), None);
    f.cpu.complete_mmio_write();
    assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
}

#[test]
fn public_x87_and_table_device_operands_retain_ram_snapshots() {
    // SDM Vol 2A §3.3 FLDCW p. 3-356; 80386 PRM §17 LGDT
    // pp. 17-94–95. Paging §3.4 retains each complete operand over MMIO.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.cpu.set_fpu_model(x86jit_core::FpuModel::Present);
        f.code_mapping(0, CODE_FRAME);
        f.put(f.pte() + 4, NEXT | 7);
        f.put(u64::from(DATA + 0xffc), 0x7f00_0000);
        f.vm.unmap(u64::from(NEXT), 0x1000).unwrap();
        f.vm.map(u64::from(NEXT), 0x1000, Prot::RWX, RegionKind::Trap)
            .unwrap();
        let code = f.memory_code(&[0xd9], 5, OFFSET + 0xfff, 4); // FLDCW
        f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
            .unwrap();
        let before = f.cpu.cpu.fpu_cw;
        let exit = public_step(&mut f);
        assert!(
            matches!(exit, Exit::MmioRead { addr, size: 1, .. } if addr == u64::from(NEXT)),
            "{exit:?}"
        );
        assert_eq!(f.cpu.cpu.fpu_cw, before);
        f.put(u64::from(DATA + 0xffc), 0);
        f.cpu.complete_mmio_read(0x0f);
        assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
        assert_eq!(f.cpu.cpu.fpu_cw, 0x0f7f);

        let code = f.memory_code(&[0x0f, 0x01], 2, OFFSET + 0xffe, 4); // LGDT
        f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
            .unwrap();
        f.cpu.cpu.rip = 0x400;
        f.put(u64::from(DATA + 0xffc), 0x1234_0000);
        let exit = public_step(&mut f);
        assert!(
            matches!(exit, Exit::MmioRead { addr, size: 4, .. } if addr == u64::from(NEXT)),
            "{exit:?}"
        );
        f.put(u64::from(DATA + 0xffc), 0);
        f.cpu.complete_mmio_read(0x1122_3344);
        assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
        assert_eq!(f.cpu.cpu.legacy.gdtr.base, 0x1122_3344);
        assert_eq!(f.cpu.cpu.legacy.gdtr.limit, 0x1234);
    }
}

#[test]
fn public_x87_image_store_acknowledgements_are_consumed_once() {
    // SDM Vol 2A §3.3 FSTENV/FNSTENV p. 3-390: save before masking
    // exceptions. Paging §3.4: rollback numeric outputs, retain bus completions.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.cpu.set_fpu_model(x86jit_core::FpuModel::Present);
        f.code_mapping(0, CODE_FRAME);
        f.put(f.pte() + 4, NEXT | 7);
        f.put(u64::from(DATA + 0xff8), 0x1122_3344);
        f.vm.unmap(u64::from(NEXT), 0x1000).unwrap();
        f.vm.map(u64::from(NEXT), 0x1000, Prot::RWX, RegionKind::Trap)
            .unwrap();
        let code = f.memory_code(&[0xd9], 6, OFFSET + 0xff8, 4); // FNSTENV, 28 bytes
        f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
            .unwrap();
        let before = f.cpu.cpu.fpu_cw;
        for (offset, bytes) in [(0, 8), (8, 8), (16, 4)] {
            let exit = public_step(&mut f);
            assert!(
                matches!(exit, Exit::MmioWrite { addr, size, .. } if addr == u64::from(NEXT + offset) && size == bytes),
                "{exit:?}"
            );
            assert_eq!(f.get(u64::from(DATA + 0xff8)), 0x1122_3344);
            assert_eq!(f.cpu.cpu.fpu_cw, before);
            f.cpu.complete_mmio_write();
        }
        assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
        assert_eq!(f.get(u64::from(DATA + 0xff8)) as u16, before);
    }
}

#[test]
fn public_task_switch_device_saves_resume_the_owned_phase_once() {
    // 80386 PRM §7.5 pp. 7-8–9; SDM Vol 3A §10.3 pp. 10-10–11:
    // outgoing saves use the old space, then incoming CR3 selects descriptors.
    for model in MODELS {
        for split in [false, true] {
            let mut f = Fixture::new(model, true, 0);
            f.incoming_task(OTHER_PD, false);
            if split {
                f.cpu.cpu.legacy.tr.cache.base = BASE + 0x7fdd;
                f.put(0x13ffc, 0xaabb_ccdd);
            }
            f.mapped_instruction(&[0xea, 0, 0, 0, 0, 0x20, 0]);
            f.vm.unmap(0x14000, 0x1000).unwrap();
            f.vm.map(0x14000, 0x1000, Prot::RWX, RegionKind::Trap)
                .unwrap();
            let mut saves = std::collections::BTreeMap::new();
            for attempt in 0..64 {
                match public_step(&mut f) {
                    Exit::MmioWrite {
                        addr, size, value, ..
                    } => {
                        assert!((0x14000..0x15000).contains(&addr));
                        assert!(
                            saves.insert(addr, (size, value)).is_none(),
                            "repeated task save"
                        );
                        if split {
                            assert_eq!(f.get(0x13ffc), 0xaabb_ccdd);
                        }
                        f.cpu.complete_mmio_write();
                    }
                    Exit::BudgetExhausted => break,
                    exit => panic!("{model:?}/{attempt}: {exit:?}"),
                }
                assert!(attempt < 63);
            }
            if split {
                assert_eq!(saves.get(&0x14000), Some(&(1, 0)));
                assert_eq!(saves.get(&0x14005), Some(&(4, 0x1122_3344)));
                assert_eq!(f.get(0x13ffc), 0x0004_07dd);
            } else {
                assert_eq!(saves.get(&0x14020), Some(&(4, 0x407)));
                assert_eq!(saves.get(&0x14028), Some(&(4, 0x1122_3344)));
                assert!(!saves.contains_key(&0x1401c)); // static CR3 is not saved
            }
            assert_eq!(f.cpu.cpu.rip, 0x900);
            assert_eq!(f.cpu.cpu.legacy.tr.selector, 32);
            assert_eq!(f.cpu.cpu.legacy.cr3 & 0xffff_f000, OTHER_PD);
            assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
            assert_eq!(f.cpu.cpu.gpr[0], 0x7654_3210);
        }
    }
}

#[test]
fn public_privilege_interrupt_device_stack_retries_all_gate_kinds() {
    // 80386 PRM §§9.5–9.6 pp. 9-6–9; SDM Vol 3A §7.12.1:
    // inner-ring delivery saves SS:ESP, FLAGS, CS:IP in the gate's width;
    // interrupt gates clear IF, trap gates retain it.
    for model in MODELS {
        for size in [2, 4] {
            for trap in [false, true] {
                let mut f = Fixture::new(model, true, 3);
                f.system_setup();
                f.cpu.cpu.cs = 11;
                f.cpu.cpu.ss = 19;
                f.cpu.cpu.flags.if_ = true;
                let old_sp = f.cpu.cpu.gpr[4];
                f.put(0x14004, 0xa800);
                f.put(0x14008, 16);
                f.gate(0x30, 0x800, size);
                if trap {
                    f.put(0x13184, if size == 4 { 0x8f00 } else { 0x8700 });
                }
                f.mapped_instruction(&[0x90]);
                f.vm.write_bytes(u64::from(CODE_FRAME) + 0x800, &[0x90])
                    .unwrap();
                f.vm.unmap(0x16000, 0x1000).unwrap();
                f.vm.map(0x16000, 0x1000, Prot::RWX, RegionKind::Trap)
                    .unwrap();
                f.cpu.inject_irq(0x30);
                let mut writes = std::collections::BTreeMap::new();
                for slot in 1..=5 {
                    let exit = public_step(&mut f);
                    let Exit::MmioWrite {
                        addr,
                        size: width,
                        value,
                        ..
                    } = exit
                    else {
                        panic!("{model:?}/{size}/{trap}/{slot}: {exit:?}");
                    };
                    assert_eq!(width, size);
                    assert_eq!(addr, 0x16800 - slot * u64::from(size));
                    assert!(writes.insert(addr, value).is_none());
                    assert_eq!(f.cpu.cpu.gpr[4], old_sp);
                    assert_eq!(f.cpu.cpu.legacy.cpl, 3);
                    // A held completion replays the same visit without advancing it.
                    assert!(matches!(public_step(&mut f), Exit::MmioWrite {
                        addr: again, size: bytes, value: stored, ..
                    } if (again, bytes, stored) == (addr, width, value)));
                    f.cpu.complete_mmio_write();
                }
                assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
                assert_eq!(f.cpu.cpu.legacy.cpl, 0);
                assert_eq!((f.cpu.cpu.cs, f.cpu.cpu.ss), (8, 16));
                assert_eq!(f.cpu.cpu.rip, 0x801);
                assert_eq!(f.cpu.cpu.gpr[4], 0xa800 - 5 * u64::from(size));
                assert_eq!(f.cpu.cpu.flags.if_, trap);
                assert_eq!(writes[&(0x16800 - u64::from(size))], 19);
                assert_eq!(writes[&(0x16800 - 2 * u64::from(size))], old_sp);
                assert_eq!(writes[&(0x16800 - 4 * u64::from(size))], 11);
                assert_eq!(writes[&(0x16800 - 5 * u64::from(size))], 0x400);
            }
        }
    }
}

#[test]
fn public_task_saves_precede_busy_clear_with_overlapping_storage() {
    // SDM Vol 3A §10.3 steps 6, 10 pp. 10-10–11: saves precede clearing
    // the outgoing busy flag. The overlap pins the same order as PG=0.
    for model in MODELS {
        for overlap in [false, true] {
            let mut f = Fixture::new(model, true, 0);
            f.incoming_task(OTHER_PD, false);
            if overlap {
                // EAX at TSS+0x28 aliases the outgoing descriptor access byte.
                f.cpu.cpu.legacy.tr.cache.base = BASE + 0x5ff5;
            }
            f.cpu.cpu.gpr[0] = 0x1122_338b;
            f.mapped_instruction(&[0xea, 0, 0, 0, 0, 0x20, 0]);
            assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
            let mut access = [0];
            f.vm.read_bytes(0x1201d, &mut access).unwrap();
            assert_eq!(access[0], 0x89, "{model:?}/{overlap}");
            if overlap {
                let mut saved = [0; 4];
                f.vm.read_bytes(0x1201d, &mut saved).unwrap();
                assert_eq!(u32::from_le_bytes(saved), 0x1122_3389);
            } else {
                assert_eq!(f.get(0x14028), 0x1122_338b);
            }
            assert_eq!(f.cpu.cpu.legacy.tr.selector, 32);
            assert_eq!(f.cpu.cpu.rip, 0x900);
            assert_eq!(f.cpu.cpu.gpr[0], 0xabcdef);
        }
    }
}

#[test]
fn public_x87_edit_abandons_retained_environment_and_stale_acknowledgement() {
    // SDM Vol 2A §3.3 FSTENV/FNSTENV p. 3-390: store the control word
    // before masking exceptions. Paging §3.4: edited inputs abandon the
    // retained image and its completions; the new attempt uses the edited CW.
    for model in MODELS {
        for completed_prefix in [false, true] {
            let mut f = Fixture::new(model, true, 0);
            f.cpu.set_fpu_model(x86jit_core::FpuModel::Present);
            f.code_mapping(0, CODE_FRAME);
            f.put(f.pte() + 4, NEXT | 7);
            f.put(u64::from(DATA + 0xff8), 0x1122_3344);
            f.vm.unmap(u64::from(NEXT), 0x1000).unwrap();
            f.vm.map(u64::from(NEXT), 0x1000, Prot::RWX, RegionKind::Trap)
                .unwrap();
            let code = f.memory_code(&[0xd9], 6, OFFSET + 0xff8, 4);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &code)
                .unwrap();
            assert!(
                matches!(public_step(&mut f), Exit::MmioWrite { addr, size: 8, .. }
                if addr == u64::from(NEXT))
            );
            if completed_prefix {
                f.cpu.complete_mmio_write();
                assert!(
                    matches!(public_step(&mut f), Exit::MmioWrite { addr, size: 8, .. }
                    if addr == u64::from(NEXT + 8))
                );
            }
            f.cpu.cpu.fpu_cw = 0x0240;
            f.cpu.complete_mmio_write();
            for (offset, width) in [(0, 8), (8, 8), (16, 4)] {
                let exit = public_step(&mut f);
                assert!(
                    matches!(exit, Exit::MmioWrite { addr, size, .. }
                    if addr == u64::from(NEXT + offset) && size == width),
                    "{exit:?}"
                );
                assert!(!f.cpu.cpu.pending_mmio_write);
                assert_eq!(f.get(u64::from(DATA + 0xff8)), 0x1122_3344);
                assert_eq!(f.cpu.cpu.fpu_cw, 0x0240);
                f.cpu.complete_mmio_write();
            }
            assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
            assert_eq!(f.get(u64::from(DATA + 0xff8)) as u16, 0x0240);
            assert_eq!(f.cpu.cpu.fpu_cw, 0x027f);
        }
    }
}

#[test]
fn public_event_stack_retry_abandons_edited_esp_or_cr3() {
    // Paging §3.4 and decision-16: an edited owner abandons the accepted
    // event and completion before replay; no acknowledgement moves to another
    // stack. SDM Vol 3A §7.12.1 defines the unedited interrupt frame.
    for model in MODELS {
        for edit_cr3 in [false, true] {
            for acknowledge in [false, true] {
                let mut f = Fixture::new(model, true, 0);
                f.system_setup();
                f.cpu.cpu.flags.if_ = true;
                f.gate(0x30, 0x800, 4);
                f.mapped_instruction(&[0x90]);
                f.put(u64::from(OTHER_PD + 4 * (BASE >> 22)), OTHER_PT | 7);
                for index in [3, 6, 7] {
                    f.put(
                        u64::from(OTHER_PT + index * 4),
                        if index == 3 {
                            NEXT | 7
                        } else {
                            f.get(u64::from(PT + index * 4))
                        },
                    );
                }
                f.map_code(OTHER_PD, 0x7000, 0, CODE_FRAME, 7);
                for frame in [DATA, NEXT] {
                    f.vm.unmap(u64::from(frame), 0x1000).unwrap();
                    f.vm.map(u64::from(frame), 0x1000, Prot::RWX, RegionKind::Trap)
                        .unwrap();
                }
                f.cpu.inject_irq(0x30);
                let exit = public_step(&mut f);
                assert!(
                    matches!(
                        exit,
                        Exit::MmioWrite {
                            addr: 0x87fc,
                            size: 4,
                            ..
                        }
                    ),
                    "{exit:?}"
                );
                assert!(f.cpu.has_pending_irq());
                if acknowledge {
                    f.cpu.complete_mmio_write();
                }
                if edit_cr3 {
                    f.cpu.cpu.legacy.cr3 = OTHER_PD;
                } else {
                    f.cpu.cpu.gpr[4] -= 0x100;
                }
                let esp = f.cpu.cpu.gpr[4];
                assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
                assert_eq!(f.cpu.cpu.rip, 0x401);
                assert_eq!(f.cpu.cpu.gpr[4], esp);
                assert!(!f.cpu.has_pending_irq());
                assert!(!f.cpu.cpu.pending_mmio_write);
                assert!(!f.cpu.cpu.mmio_retry_active);
                assert_eq!(f.cpu.pending_mmio_linear_address(), None);
            }
        }
    }
}

fn assert_paged_owner_refused(f: &mut Fixture) {
    for _ in 0..2 {
        let exit = public_step(f);
        assert!(
            matches!(
                exit,
                Exit::UnsupportedSystemContext {
                    operation: UnsupportedSystemOperation::LegacyEntryFacility,
                    ..
                }
            ),
            "{exit:?}"
        );
    }
}

#[test]
fn public_split_store_refuses_physical_map_edits() {
    // Paging §3.4: a physical plan cannot transfer its staged prefix or
    // acknowledged tail to a replacement mapping, even of the same kind.
    for model in MODELS {
        for page in [DATA, NEXT] {
            for replacement in [None, Some(RegionKind::Ram), Some(RegionKind::Trap)] {
                let mut f = Fixture::new(model, true, 0);
                f.put(f.pte() + 4, NEXT | 7);
                f.put(u64::from(DATA + 0xffc), 0x1234_5678);
                f.vm.unmap(u64::from(NEXT), 0x1000).unwrap();
                f.vm.map(u64::from(NEXT), 0x1000, Prot::RWX, RegionKind::Trap)
                    .unwrap();
                f.cpu.cpu.gpr[0] = 0xaabb_ccdd;
                f.mapped_instruction(&f.code(true, OFFSET + 0xffe));
                assert!(
                    matches!(public_step(&mut f), Exit::MmioWrite { addr, size: 2, value: 0xaabb, .. } if addr == u64::from(NEXT))
                );
                f.vm.unmap(u64::from(page), 0x1000).unwrap();
                if let Some(kind) = replacement {
                    f.vm.map(u64::from(page), 0x1000, Prot::RWX, kind).unwrap();
                }
                f.cpu.complete_mmio_write();
                let before = f.cpu.cpu.gpr;
                assert_paged_owner_refused(&mut f);
                assert_eq!(f.cpu.cpu.gpr, before);
                assert_eq!(f.cpu.cpu.rip, 0x400);
                assert!(f.cpu.cpu.pending_mmio_write);
                if page != DATA || replacement.is_some() {
                    assert_eq!(f.get(u64::from(DATA + 0xffc)), 0x1234_5678);
                }
            }
        }
        // A new tag elsewhere also changes the map generation.
        let mut f = Fixture::new(model, true, 0);
        f.vm.unmap(0x1f000, 0x1000).unwrap();
        f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
        f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Trap)
            .unwrap();
        f.mapped_instruction(&f.code(true, OFFSET));
        assert!(matches!(public_step(&mut f), Exit::MmioWrite { .. }));
        f.vm.map(0x1f000, 0x1000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        f.cpu.complete_mmio_write();
        assert_paged_owner_refused(&mut f);
    }
}

#[test]
fn public_event_refuses_physical_stack_remap() {
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.system_setup();
        f.cpu.cpu.flags.if_ = true;
        f.gate(0x30, 0x800, 4);
        f.mapped_instruction(&[0x90]);
        f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
        f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Trap)
            .unwrap();
        f.cpu.inject_irq(0x30);
        assert!(matches!(
            public_step(&mut f),
            Exit::MmioWrite { addr: 0x87fc, .. }
        ));
        f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
        f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        f.cpu.complete_mmio_write();
        assert_paged_owner_refused(&mut f);
        assert_eq!(f.cpu.cpu.gpr[4], u64::from(OFFSET + 0x800));
        assert_eq!(f.get(0x87fc), 0);
        assert!(f.cpu.has_pending_irq());
    }
}

#[test]
fn public_ins_refuses_edited_destination_owner() {
    // Paging §3.4: INS owns all destination/count inputs across the port
    // handoff and subsequent device store; only completion may change them.
    for model in MODELS {
        for rep in [false, true] {
            for device in [false, true] {
                for edit in 0..10 {
                    let mut f = Fixture::new(model, true, 0);
                    f.cpu.cpu.gpr[7] = u64::from(OFFSET);
                    f.cpu.cpu.gpr[1] = 2;
                    if device {
                        f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
                        f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Trap)
                            .unwrap();
                    }
                    f.mapped_instruction(if rep { &[0xf3, 0x6d] } else { &[0x6d] });
                    assert!(matches!(public_step(&mut f), Exit::PortIo { .. }));
                    f.cpu.complete_port_in(0x1122_3344);
                    if device {
                        assert!(matches!(public_step(&mut f), Exit::MmioWrite { .. }));
                        f.cpu.complete_mmio_write();
                    }
                    match edit {
                        0 => f.cpu.cpu.gpr[7] += 0x10,
                        1 => f.cpu.cpu.legacy.segments[2].base += 0x1000,
                        2 => f.cpu.cpu.legacy.cr3 = OTHER_PD,
                        3 => f.cpu.cpu.gpr[1] += 1,
                        4 => f.cpu.cpu.flags.df = true,
                        5 => f.cpu.cpu.port_string_in.as_mut().unwrap().next_index += 0x10,
                        6 => {
                            f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
                            f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Ram)
                                .unwrap();
                        }
                        7 => f.cpu.cpu.gpr[2] += 1,
                        8 if device => f.cpu.cpu.port_string_in.as_mut().unwrap().value = Some(7),
                        8 => f.cpu.cpu.pending_port_in = Some(4),
                        9 => f.cpu.cpu.flags.if_ = !f.cpu.cpu.flags.if_,
                        _ => unreachable!(),
                    }
                    let before = f.cpu.cpu.gpr;
                    let value = f.cpu.cpu.port_string_in.unwrap().value;
                    assert_paged_owner_refused(&mut f);
                    assert_eq!(f.cpu.cpu.gpr, before);
                    assert_eq!(f.get(u64::from(DATA)), 0);
                    assert_eq!(f.cpu.cpu.port_string_in.unwrap().value, value);
                }
            }
        }
    }
}

#[test]
fn public_task_incoming_eax_reads_committed_busy_clear() {
    // SDM Vol 3A §10.3 steps 6, 10, 12 pp. 10-10–12: the incoming
    // image observes committed saves and the subsequent descriptor CAS.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.incoming_task(OTHER_PD, false);
        f.system_page(5, 0x11000, 3);
        let base = BASE + 0x5ff5;
        f.cpu.cpu.legacy.tr.cache.base = base;
        f.put64(0x12020, step4_descriptor(base, 0x100, 0x89));
        f.put(0x12011, OTHER_PD | 0x18);
        f.cpu.cpu.gpr[0] = 0x1122_338b;
        f.mapped_instruction(&[0xea, 0, 0, 0, 0, 0x20, 0]);
        assert!(matches!(public_step(&mut f), Exit::BudgetExhausted));
        assert_eq!(f.get(0x1201d), 0x1122_3389);
        assert_eq!(f.cpu.cpu.gpr[0], 0x1122_3389);
        assert_eq!(f.cpu.cpu.legacy.tr.selector, 32);
        assert_eq!(f.cpu.cpu.rip, 0x407);
    }
}

#[test]
fn public_held_instruction_refuses_replacement_memory_identity() {
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
        f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Trap)
            .unwrap();
        f.mapped_instruction(&f.code(true, OFFSET));
        assert!(matches!(public_step(&mut f), Exit::MmioWrite { .. }));
        let mut replacement = Fixture::new(model, true, 0);
        replacement.vm.unmap(u64::from(DATA), 0x1000).unwrap();
        replacement
            .vm
            .map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Trap)
            .unwrap();
        replacement.mapped_instruction(&replacement.code(true, OFFSET));
        std::mem::swap(&mut f.vm, &mut replacement.vm);
        f.cpu.complete_mmio_write();
        assert_paged_owner_refused(&mut f);
        assert!(f.cpu.cpu.pending_mmio_write);
    }
}

#[test]
fn public_instruction_and_event_retries_cover_additional_cpu_inputs() {
    for model in MODELS {
        for event in [false, true] {
            for edit in 0..5 {
                let mut f = Fixture::new(model, true, 0);
                if event {
                    f.system_setup();
                    f.cpu.cpu.flags.if_ = true;
                    f.gate(0x30, 0x800, 4);
                    f.mapped_instruction(&[0x90]);
                    f.cpu.inject_irq(0x30);
                } else {
                    f.mapped_instruction(&f.code(true, OFFSET));
                }
                f.vm.unmap(u64::from(DATA), 0x1000).unwrap();
                f.vm.map(u64::from(DATA), 0x1000, Prot::RWX, RegionKind::Trap)
                    .unwrap();
                assert!(matches!(public_step(&mut f), Exit::MmioWrite { .. }));
                f.cpu.complete_mmio_write();
                match edit {
                    0 => f.cpu.cpu.random_state ^= 1,
                    1 => f.cpu.cpu.source_id ^= 1,
                    2 => f.cpu.cpu.task_io_map_base ^= 1,
                    3 => f.cpu.cpu.cpuid_overrides.xcr0 = Some(0),
                    4 => f.cpu.cpu.flags.if_ = !f.cpu.cpu.flags.if_,
                    _ => unreachable!(),
                }
                let exit = public_step(&mut f);
                if event {
                    assert!(
                        matches!(exit, Exit::BudgetExhausted),
                        "{model:?}/{edit}: {exit:?}"
                    );
                    assert!(!f.cpu.has_pending_irq());
                    assert_eq!(f.cpu.cpu.rip, 0x401);
                    assert_eq!(f.cpu.cpu.gpr[4], u64::from(OFFSET + 0x800));
                } else {
                    assert!(
                        matches!(exit, Exit::MmioWrite { .. }),
                        "{model:?}/{edit}: {exit:?}"
                    );
                }
                assert!(!f.cpu.cpu.pending_mmio_write);
                assert_eq!(f.get(if event { 0x87fc } else { u64::from(DATA) }), 0);
            }
        }
    }
}

#[test]
fn tlb_permissions_match_walks_for_every_combined_right_and_wp() {
    // 80386 PRM §§6.4.1–6.4.2 Table 6-5; SDM Vol 3A §5.6.1:
    // CPL0–2 bypass R/W only with WP clear, CPL3 requires combined U/S.
    for model in MODELS {
        for cpl in 0..4 {
            for wp in [false, true] {
                let mut f = Fixture::new(model, true, cpl);
                if wp && model != CpuModel::I80386 {
                    f.cpu.cpu.legacy.cr0 |= 1 << 16;
                }
                f.code_mapping(0, CODE_FRAME);
                for (ip, write) in [(0x400, false), (0x500, true)] {
                    f.vm.write_bytes(u64::from(CODE_FRAME) + ip, &f.code(write, OFFSET))
                        .unwrap();
                }
                for pde in [1, 3, 5, 7] {
                    for pte in [1, 3, 5, 7] {
                        f.cpu.invalidate_paging();
                        f.tables(pde | A, pte | A | D);
                        for write in [false, true] {
                            let allowed = (cpl != 3 || pde & pte & 4 != 0)
                                && (!write
                                    || (cpl != 3 && (!wp || model == CpuModel::I80386))
                                    || pde & pte & 2 != 0);
                            for _ in 0..2 {
                                f.cpu.cpu.gpr[0] = 0x1234_5678;
                                f.put(u64::from(DATA), 0x8765_4321);
                                let result = f.fetch_at(if write { 0x500 } else { 0x400 });
                                if allowed {
                                    result.unwrap();
                                    assert_eq!(
                                        f.cpu.cpu.gpr[0],
                                        if write { 0x1234_5678 } else { 0x8765_4321 }
                                    );
                                    assert_eq!(
                                        f.get(u64::from(DATA)),
                                        if write { 0x1234_5678 } else { 0x8765_4321 }
                                    );
                                } else {
                                    let exit = result.unwrap_err();
                                    assert!(
                                        matches!(exit, Exit::Exception { vector: 14, .. }),
                                        "{exit:?}"
                                    );
                                    assert_eq!(
                                        exit.fault_metadata().unwrap().error_code(),
                                        ErrorCode::bits(
                                            1 | (u32::from(write) << 1)
                                                | (u32::from(cpl == 3) << 2),
                                            7
                                        )
                                    );
                                    assert_eq!(f.cpu.cpu.gpr[0], 0x1234_5678);
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
fn tlb_stale_mapping_and_accessed_bits_last_until_explicit_invalidation() {
    // SDM Vol 3A §5.10.2.3: cached frame/rights may survive table stores;
    // clearing A requires invalidation before predictable re-setting.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.code_mapping(0, CODE_FRAME);
        f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &f.code(false, OFFSET))
            .unwrap();
        f.put(u64::from(DATA), 0x11);
        f.put(u64::from(NEXT), 0x22);
        f.fetch_at(0x400).unwrap();
        let proof = f.cpu.paging_cache_for_test();
        f.put(f.pte(), NEXT | 7);
        f.put(f.pde(), PT | 7);
        f.put(u64::from(CODE_PT), OTHER_FRAME); // not present, warm code proof survives
        f.fetch_at(0x400).unwrap();
        assert_eq!(f.cpu.cpu.gpr[0], 0x11);
        assert_eq!(f.get(f.pte()), NEXT | 7);
        assert_eq!(f.get(f.pde()), PT | 7);
        assert_eq!(f.cpu.paging_cache_for_test(), proof);
        f.code_mapping(0, CODE_FRAME);
        f.cpu.invalidate_paging();
        f.fetch_at(0x400).unwrap();
        assert_eq!(f.cpu.cpu.gpr[0], 0x22);
        assert_eq!(f.get(f.pte()), NEXT | 7 | A);
        assert_eq!(f.get(f.pde()), PT | 7 | A);
    }
}

#[test]
fn tlb_embedder_edits_map_identity_and_reset_flush_data_translations() {
    // Paging design §3.2: conservative flush on positive-entry edits/restore.
    for model in MODELS {
        for trigger in 0..7 {
            let mut f = Fixture::new(model, true, 0);
            f.code_mapping(0, CODE_FRAME);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &f.code(false, OFFSET))
                .unwrap();
            f.put(u64::from(DATA), 0x11);
            f.put(u64::from(NEXT), 0x22);
            f.fetch_at(0x400).unwrap();
            let old = f.cpu.paging_cache_for_test().0;
            f.put(f.pte(), NEXT | 7);
            match trigger {
                0 => f.cpu.invalidate_paging(),
                1 => f.cpu.cpu.legacy.segments[1].limit -= 1,
                2 => f.cpu.set_a20_enabled(false),
                3 => f.cpu.cpu.legacy.cr3 ^= 8,
                4 => {
                    f.vm.unmap(0x1f000, 0x1000).unwrap();
                }
                5 => f.cpu.reset_execution_state(Default::default()),
                6 => {
                    let mut replacement = Fixture::new(model, true, 0);
                    replacement.code_mapping(0, CODE_FRAME);
                    replacement
                        .vm
                        .write_bytes(
                            u64::from(CODE_FRAME) + 0x400,
                            &replacement.code(false, OFFSET),
                        )
                        .unwrap();
                    replacement.put(replacement.pte(), NEXT | 7);
                    replacement.put(u64::from(NEXT), 0x22);
                    std::mem::swap(&mut f.vm.mem, &mut replacement.vm.mem);
                }
                _ => unreachable!(),
            }
            f.fetch_at(0x400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x22, "trigger {trigger}");
            assert!(f.cpu.paging_cache_for_test().0 > old);
        }
    }
}

#[test]
fn tlb_vcpu_tokens_and_invalidation_are_independent() {
    // Decision-27: equal local epochs never authorize sharing different CR3s.
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.code_mapping(0, CODE_FRAME);
        f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &f.code(false, OFFSET))
            .unwrap();
        f.put(u64::from(DATA), 0x11);
        f.put(u64::from(NEXT), 0x22);
        f.fetch_at(0x400).unwrap();
        let mut other = f.vm.new_vcpu();
        other.cpu = f.cpu.cpu.clone();
        other.cpu.rip = 0x400;
        other.interpret_paged_for_test(&f.vm).unwrap();
        assert_ne!(
            other.paging_cache_for_test().0,
            f.cpu.paging_cache_for_test().0
        );
        let other_proof = other.paging_cache_for_test();
        f.put(f.pte(), NEXT | 7);
        f.cpu.invalidate_paging();
        f.fetch_at(0x400).unwrap();
        assert_eq!(f.cpu.cpu.gpr[0], 0x22);
        other.cpu.rip = 0x400;
        other.interpret_paged_for_test(&f.vm).unwrap();
        assert_eq!(other.cpu.gpr[0], 0x11);
        assert_eq!(other.paging_cache_for_test(), other_proof);
        other.invalidate_paging();
        other.cpu.rip = 0x400;
        other.interpret_paged_for_test(&f.vm).unwrap();
        assert_eq!(other.cpu.gpr[0], 0x22);
    }
}

#[test]
fn tlb_guest_control_invalidations_observe_edited_data_pte() {
    // Paging design §3.2; 80386 PRM §5.2.5; SDM Vol 3A §5.10.4.1.
    for model in MODELS {
        for trigger in 0..5 {
            if model == CpuModel::I80386 && (trigger == 2 || trigger == 3) {
                continue;
            }
            let mut f = Fixture::new(model, true, 0);
            f.cpu.cpu.legacy.cr3 = PD as u32;
            f.code_mapping(0, CODE_FRAME);
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &f.code(false, OFFSET))
                .unwrap();
            let control: &[u8] = match trigger {
                0 | 1 => &[0x0f, 0x22, 0xd8],         // MOV CR3,EAX
                2 | 4 => &[0x0f, 0x22, 0xc0],         // MOV CR0,EAX
                3 => &[0x0f, 0x01, 0x3d, 0, 0, 0, 0], // INVLPG [0], full flush
                _ => unreachable!(),
            };
            f.vm.write_bytes(u64::from(CODE_FRAME) + 0x600, control)
                .unwrap();
            f.put(u64::from(DATA), 0x11);
            f.put(u64::from(NEXT), 0x22);
            f.fetch_at(0x400).unwrap();
            let old = f.cpu.paging_cache_for_test().0;
            f.put(f.pte(), NEXT | 7);
            f.cpu.cpu.gpr[0] = match trigger {
                0 => PD,
                1 => {
                    f.map_code(OTHER_PD, CODE_PT, 0, CODE_FRAME, 7 | A);
                    f.put(u64::from(OTHER_PD + 4 * (LINEAR >> 22)), PT | 7 | A);
                    u64::from(OTHER_PD)
                }
                2 => u64::from(f.cpu.cpu.legacy.cr0 ^ (1 << 16)),
                3 => 0,
                4 => 1,
                _ => unreachable!(),
            };
            f.fetch_at(0x600).unwrap();
            assert!(f.cpu.paging_cache_for_test().0 > old);
            if trigger == 4 {
                // Re-enable PG through the unpaged instruction path.
                f.vm.write_bytes(0x600, control).unwrap();
                f.cpu.cpu.gpr[0] = 0x8000_0001;
                f.fetch_at(0x600).unwrap();
            }
            f.fetch_at(0x400).unwrap();
            assert_eq!(f.cpu.cpu.gpr[0], 0x22, "trigger {trigger}");
        }
    }
}

#[test]
fn tlb_warm_fetch_preserves_cr2_and_zero_budget_and_rechecks_fetch_policy() {
    // Paging design §3.2: CR2 is output-only and zero budget has no effects.
    // The embedder fetch policy remains a physical source proof dimension (§3.5).
    for model in MODELS {
        let mut f = Fixture::new(model, true, 0);
        f.code_mapping(0, CODE_FRAME);
        f.vm.write_bytes(u64::from(CODE_FRAME) + 0x400, &[0xb0, 0x11])
            .unwrap();
        f.fetch_at(0x400).unwrap();
        let proof = f.cpu.paging_cache_for_test();
        f.cpu.cpu.legacy.cr2 = 0x1234_5678;
        f.fetch_at(0x400).unwrap();
        assert_eq!(f.cpu.paging_cache_for_test(), proof);
        assert_eq!(f.cpu.cpu.legacy.cr2, 0x1234_5678);
        f.put(u64::from(CODE_PT), 0);
        f.cpu.cpu.rip = 0x400;
        let cr2 = f.cpu.cpu.legacy.cr2;
        let run = f.cpu.run_steps(&f.vm, 0, StepOptions::default()).unwrap();
        assert_eq!(run.steps(), 0);
        assert!(matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)));
        assert_eq!(f.cpu.cpu.legacy.cr2, cr2);
        assert_eq!(f.get(u64::from(CODE_PT)), 0);
        assert_eq!(f.cpu.paging_cache_for_test(), proof);
        f.code_mapping(0, CODE_FRAME);
        f.vm.set_fetch_policy(Box::new(|_| x86jit_core::FetchVerdict::DenyProtection));
        let exit = f.fetch_at(0x400).unwrap_err();
        assert!(
            matches!(
                exit,
                Exit::UnmappedMemory {
                    access: x86jit_core::AccessKind::Execute,
                    ..
                }
            ),
            "{exit:?}"
        );
    }
}
