//! The 80286 LOADALL instruction (0F 05) in the Machine profile.
//! Authority: Intel, "Undocumented iAPX 286 Test Instruction" (undated,
//! 15 pp., `oracles/intel/i286-loadall.pdf`), cited by the page numbers
//! printed on it; Intel 80286 and 80287 PRM 210498-005 (1987) for the
//! surrounding real- and protected-mode rules. The engine's choices where the
//! document is silent, and its table policy, are in spec.md §17.10.
#[path = "support/model_api.rs"]
mod model_api;
use x86jit_core::fault::ErrorCode;
use x86jit_core::lift::{lift_block, FetchAddr};
use x86jit_core::{
    Backend, CachedBlock, CpuGeneration, CpuMode, CpuModel, CpuState, Exit, InterpreterBackend,
    IrBlock, IrOp, LegacyCpuModel, MemConsistency, Prot, RegionKind, SegmentCache, StepOptions,
    StepStop, SystemSegment, TableRegister, UnsupportedSystemOperation, Vcpu, Vm, VmConfig,
};

const MODEL: LegacyCpuModel = LegacyCpuModel {
    generation: CpuGeneration::I286,
    model_semantics_id: 0x286,
};
struct NoBackend;
impl Backend for NoBackend {
    fn materialize(
        &self,
        _: &IrBlock,
        _: MemConsistency,
        _: Option<(u64, u64)>,
        _: u64,
    ) -> CachedBlock {
        panic!("legacy execution reached application backend")
    }
}

// p. 2, "LOADALL Memory Area Format": the physical address of each register.
// 800h-805h and 808h-815h load no register.
const MSW: usize = 0x806;
const TR: usize = 0x816;
const FLAGS: usize = 0x818;
const IP: usize = 0x81a;
const LDT: usize = 0x81c;
const DS: usize = 0x81e;
const SS: usize = 0x820;
const CS: usize = 0x822;
const ES: usize = 0x824;
const DI: usize = 0x826;
const SI: usize = 0x828;
const BP: usize = 0x82a;
const SP: usize = 0x82c;
const BX: usize = 0x82e;
const DX: usize = 0x830;
const CX: usize = 0x832;
const AX: usize = 0x834;
const ES_CACHE: usize = 0x836;
const CS_CACHE: usize = 0x83c;
const SS_CACHE: usize = 0x842;
const DS_CACHE: usize = 0x848;
const GDTR: usize = 0x84e;
const LDT_CACHE: usize = 0x854;
const IDTR: usize = 0x85a;
const TSS_CACHE: usize = 0x860;

/// The 102-byte table at physical 000800H (p. 2).
struct Table([u8; 102]);
impl Table {
    fn word(&mut self, address: usize, value: u16) -> &mut Self {
        self.0[address - 0x800..][..2].copy_from_slice(&value.to_le_bytes());
        self
    }
    /// p. 3: a 24-bit base, least significant byte first, the access rights
    /// byte, then the 16-bit limit. GDTR and IDTR hold zeroes in byte 3.
    fn entry(&mut self, address: usize, base: u32, access: u8, limit: u16) -> &mut Self {
        let bytes = &mut self.0[address - 0x800..][..6];
        bytes[..3].copy_from_slice(&base.to_le_bytes()[..3]);
        bytes[3] = access;
        bytes[4..].copy_from_slice(&limit.to_le_bytes());
        self
    }
    fn write(&self, vm: &Vm) {
        vm.write_bytes(0x800, &self.0).unwrap();
    }
}

/// A real-mode image: selectors zero, every cache a present, writable,
/// expand-up data segment of 64 KiB at base zero (p. 4 item 2 gives this as
/// the read/write/execute form of the CS cache), the IVT at zero.
fn real_table(ip: u16) -> Table {
    let mut table = Table([0; 102]);
    table.word(FLAGS, 0x0002).word(IP, ip).word(SP, 0x8000);
    for cache in [CS_CACHE, DS_CACHE, ES_CACHE, SS_CACHE] {
        table.entry(cache, 0, 0x93, 0xffff);
    }
    table.entry(IDTR, 0, 0, 0x3ff);
    table
}

fn setup(code: &[u8]) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x100_0000), Box::new(NoBackend));
    vm.map(0, 0x100_0000, Prot::RWX, RegionKind::Ram).unwrap();
    model_api::select(&mut vm, MODEL);
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([0; 6]);
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    (vm, cpu)
}
fn steps(cpu: &mut Vcpu, vm: &Vm, count: u64) {
    let run = cpu.run_steps(vm, count, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), count, "{:?}", run.stop());
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{:?}",
        run.stop()
    );
}
fn fault(cpu: &mut Vcpu, vm: &Vm, vector: u8, code: Option<u32>) {
    let before = format!("{:?}", cpu.cpu);
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    let StepStop::Exit(exit @ Exit::Exception { vector: actual, .. }) = run.stop() else {
        panic!("{:?}", run.stop());
    };
    assert_eq!(*actual, vector);
    let details = exit.fault_metadata().unwrap();
    assert_eq!(details.saved_rip(), cpu.cpu.rip);
    assert_eq!(
        details.error_code(),
        code.map_or(ErrorCode::NoCode, |c| ErrorCode::bits(c, 0xffff))
    );
    assert_eq!(format!("{:?}", cpu.cpu), before);
}
fn refused(cpu: &mut Vcpu, vm: &Vm) {
    let before = format!("{:?}", cpu.cpu);
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    assert!(
        matches!(
            run.stop(),
            StepStop::Exit(Exit::UnsupportedSystemContext {
                operation: UnsupportedSystemOperation::I286LoadallTable,
                ..
            })
        ),
        "{:?}",
        run.stop()
    );
    assert_eq!(format!("{:?}", cpu.cpu), before);
}
fn word(vm: &Vm, address: u64) -> u16 {
    let mut bytes = [0; 2];
    vm.read_bytes(address, &mut bytes).unwrap();
    u16::from_le_bytes(bytes)
}
fn cache(base: u32, limit: u32, attributes: u16) -> SegmentCache {
    SegmentCache {
        base,
        limit,
        attributes,
        unusable: 0,
        reserved: 0,
    }
}

#[test]
fn real_mode_ds_and_es_caches_reach_memory_above_one_mebibyte_cpu_model() {
    model_api::with_cpu_model(real_mode_ds_and_es_caches_reach_memory_above_one_mebibyte);
}

#[test]
fn real_mode_ds_and_es_caches_reach_memory_above_one_mebibyte() {
    // p. 1: "In real mode, LOADALL can provide addressability to the other
    // 15 Mbytes of the 286 physical address space"; the caches are used
    // independently of the selector values. The board's A20 gate still
    // masks the formed address of a read and of a write (I286 PRM App. C
    // hardware item 2 p. C-3): both bases have bit 20 set.
    for a20 in [true, false] {
        let (vm, mut cpu) = setup(b"\x0f\x05\xa1\x10\x00\x26\x89\x06\x20\x00\xf4");
        let mut table = real_table(0x1002);
        table
            .word(DS, 0x1234)
            .entry(DS_CACHE, 0x11_0000, 0x93, 0xffff)
            .word(ES, 0x5678)
            .entry(ES_CACHE, 0x30_0000, 0x93, 0xffff);
        table.write(&vm);
        vm.write_bytes(0x11_0010, &0xbeefu16.to_le_bytes()).unwrap();
        vm.write_bytes(0x01_0010, &0xcafeu16.to_le_bytes()).unwrap();
        vm.write_bytes(0x1_2350, &0x1111u16.to_le_bytes()).unwrap();
        cpu.set_a20_enabled(a20);
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            (cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.rip),
            (0, 0x1234, 0x5678, 0, 0x1002)
        );
        assert_eq!(cpu.cpu.legacy.segments[1], cache(0x11_0000, 0xffff, 0x93));
        assert_eq!(cpu.cpu.legacy.segments[2], cache(0x30_0000, 0xffff, 0x93));
        assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
        let expected = if a20 { 0xbeef } else { 0xcafe };
        assert_eq!(cpu.cpu.gpr[0] & 0xffff, expected);
        let (written, untouched) = if a20 {
            (0x30_0020, 0x20_0020)
        } else {
            (0x20_0020, 0x30_0020)
        };
        assert_eq!(u64::from(word(&vm, written)), expected);
        assert_eq!(word(&vm, untouched), 0);
        assert_eq!(word(&vm, 0x5_67a0), 0);
    }
}

#[test]
fn real_mode_block_copy_from_below_to_above_one_mebibyte_cpu_model() {
    model_api::with_cpu_model(real_mode_block_copy_from_below_to_above_one_mebibyte);
}

#[test]
fn real_mode_block_copy_from_below_to_above_one_mebibyte() {
    // The shape of an extended-memory move: LOADALL points DS below and ES
    // above 1 MiB, REP MOVSW copies, ordinary segment loads restore the
    // program's segments (p. 1; p. 3, reloading in real mode).
    let (vm, mut cpu) = setup(b"\x0f\x05");
    vm.write_bytes(0x1100, b"\xf3\xa5\x8c\xc8\x8e\xd8\x8e\xc0\xf4")
        .unwrap();
    let mut table = real_table(0x1100);
    table
        .word(DS, 0x2468)
        .entry(DS_CACHE, 0x02_4680, 0x93, 0xffff)
        .word(ES, 0x1a00)
        .entry(ES_CACHE, 0x1a_0000, 0x93, 0xffff)
        .word(CX, 0x100);
    table.write(&vm);
    let source: Vec<u8> = (0..0x200u32).map(|i| (i * 7 + 3) as u8).collect();
    vm.write_bytes(0x02_4680, &source).unwrap();
    assert!(matches!(cpu.run(&vm, Some(0x200)), Exit::Hlt));
    let mut copied = vec![0; 0x200];
    vm.read_bytes(0x1a_0000, &mut copied).unwrap();
    assert_eq!(copied, source);
    // The address keeps bit 20: nothing reached the wrapped alias.
    let mut alias = vec![0xff; 0x200];
    vm.read_bytes(0x0a_0000, &mut alias).unwrap();
    assert!(alias.iter().all(|b| *b == 0));
    assert_eq!(
        (cpu.cpu.gpr[1] & 0xffff, cpu.cpu.gpr[6], cpu.cpu.gpr[7]),
        (0, 0x200, 0x200)
    );
    assert_eq!((cpu.cpu.ds, cpu.cpu.es), (0, 0));
    for segment in [1, 2] {
        assert_eq!(cpu.cpu.legacy.segments[segment], cache(0, 0xffff, 0x93));
    }
}

#[test]
fn loaded_cache_survives_until_a_segment_load_then_reverts_cpu_model() {
    model_api::with_cpu_model(loaded_cache_survives_until_a_segment_load_then_reverts);
}

#[test]
fn loaded_cache_survives_until_a_segment_load_then_reverts() {
    // p. 1: offset checks against the loaded limit in either mode. p. 3: a
    // real-mode segment load puts the paragraph id in base bits 19:4, clears
    // the rest, resets the limit to FFFFH and makes the segment writable.
    let (vm, mut cpu) = setup(
        b"\x0f\x05\x8c\xd8\x8b\x1e\xf0\x00\x50\x58\xeb\x00\x8b\x0e\xff\x00\
          \x8e\xd8\x8b\x0e\xff\x00\xf4",
    );
    let mut table = real_table(0x1002);
    table
        .word(DS, 0x0040)
        .entry(DS_CACHE, 0x12_3450, 0x93, 0x00ff);
    table.write(&vm);
    vm.write_bytes(0x12_3540, &0x1357u16.to_le_bytes()).unwrap();
    vm.write_bytes(0x4f0, &0x9999u16.to_le_bytes()).unwrap();
    vm.write_bytes(0x4ff, &0x2468u16.to_le_bytes()).unwrap();
    // LOADALL, MOV AX,DS, MOV BX,[00F0], PUSH AX, POP AX, JMP SHORT.
    steps(&mut cpu, &vm, 6);
    assert_eq!(
        (cpu.cpu.gpr[0] & 0xffff, cpu.cpu.gpr[3] & 0xffff),
        (0x0040, 0x1357)
    );
    assert_eq!(cpu.cpu.legacy.segments[1], cache(0x12_3450, 0xff, 0x93));
    // MOV CX,[00FF]: the word's second byte lies past the loaded limit.
    // Real-mode limit violations are interrupt 13 without an error code
    // (I286 PRM Appendix B MOV p. B-74).
    assert_eq!(cpu.cpu.rip, 0x100c);
    fault(&mut cpu, &vm, 13, None);
    cpu.cpu.rip = 0x1010;
    // MOV DS,AX reloads the cache from the selector; MOV CX,[00FF] reads.
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.legacy.segments[1], cache(0x400, 0xffff, 0x93));
    assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
    assert_eq!(cpu.cpu.gpr[1] & 0xffff, 0x2468);
}

#[test]
fn a_far_jump_ends_a_loaded_cs_limit_cpu_model() {
    model_api::with_cpu_model(a_far_jump_ends_a_loaded_cs_limit);
}

#[test]
fn a_far_jump_ends_a_loaded_cs_limit() {
    // p. 1: a near target past the loaded CS limit faults. p. 3: a far
    // transfer reloads CS with the limit FFFFH, which its target then meets.
    // The rights 9Bh are the engine's real-mode CS label, which neither
    // source gives (ASSUMED, spec.md §17.10).
    for far in [false, true] {
        let (vm, mut cpu) = setup(b"\x0f\x05");
        vm.write_bytes(
            0x1002,
            if far {
                b"\xea\x00\x20\x00\x00"
            } else {
                b"\xe9\xfb\x0f\x90\x90"
            },
        )
        .unwrap();
        vm.write_bytes(0x2000, b"\xf4").unwrap();
        let mut table = real_table(0x1002);
        table.entry(CS_CACHE, 0, 0x93, 0x10ff);
        table.write(&vm);
        steps(&mut cpu, &vm, 1);
        if far {
            assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
            assert_eq!(cpu.cpu.legacy.segments[0], cache(0, 0xffff, 0x9b));
        } else {
            fault(&mut cpu, &vm, 13, None);
        }
    }
}

#[test]
fn ip_flags_registers_and_cs_cache_come_from_the_table_cpu_model() {
    model_api::with_cpu_model(ip_flags_registers_and_cs_cache_come_from_the_table);
}

#[test]
fn ip_flags_registers_and_cs_cache_come_from_the_table() {
    // p. 1: all CPU registers load from memory; p. 2: the table order.
    let (vm, mut cpu) = setup(b"\x0f\x05");
    let mut table = real_table(0x2000);
    table
        .word(CS, 0x0100)
        .entry(CS_CACHE, 0x5000, 0x93, 0xffff)
        .word(SS, 0x0300)
        .entry(SS_CACHE, 0x3_0000, 0x93, 0xffff)
        .word(FLAGS, 0x0cd5)
        .word(AX, 0x1111)
        .word(CX, 0x2222)
        .word(DX, 0x3333)
        .word(BX, 0x4444)
        .word(SP, 0x5554)
        .word(BP, 0x6666)
        .word(SI, 0x7777)
        .word(DI, 0x8888);
    table.write(&vm);
    // The CS cache base, not 0100h x 16, locates the next instruction.
    vm.write_bytes(0x7000, b"\xf4").unwrap();
    vm.write_bytes(0x3000, b"\x0f\x0b").unwrap();
    cpu.cpu.gpr[0] = 0x0123_4567_89ab_cdef;
    steps(&mut cpu, &vm, 1);
    assert_eq!(
        (cpu.cpu.cs, cpu.cpu.ss, cpu.cpu.rip),
        (0x100, 0x300, 0x2000)
    );
    assert_eq!(cpu.cpu.flags.to_flags16() & 0x0ed5, 0x0cd5);
    // A word register load keeps the bits a 16-bit write keeps.
    assert_eq!(cpu.cpu.gpr[0], 0x0123_4567_89ab_1111);
    assert_eq!(
        cpu.cpu.gpr[1..8]
            .iter()
            .map(|r| r & 0xffff)
            .collect::<Vec<_>>(),
        [0x2222, 0x3333, 0x4444, 0x5554, 0x6666, 0x7777, 0x8888]
    );
    assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
    assert_eq!(cpu.cpu.rip, 0x2001);
}

fn protected_setup() -> (Vm, Vcpu) {
    let (vm, mut cpu) = setup(b"\x0f\x05\xa1\x00\x00\x26\xa1\x00\x00");
    cpu.cpu.legacy.cr0 = 1;
    (vm, cpu)
}

#[test]
fn protected_mode_loadall_at_cpl_zero_loads_system_registers_cpu_model() {
    model_api::with_cpu_model(protected_mode_loadall_at_cpl_zero_loads_system_registers);
}

#[test]
fn protected_mode_loadall_at_cpl_zero_loads_system_registers() {
    // p. 1: CPL 0 in protected mode; LDTR, TR, GDTR, IDTR and MSW load.
    // p. 2: LOADALL cannot clear PE, so an MSW whose bit 0 is clear loads
    // without a check (spec.md §17.10). p. 3: no descriptor-table access,
    // and an invalid cache (present bit zero) faults on use with #GP(0).
    let (vm, mut cpu) = protected_setup();
    let mut table = real_table(0x1002);
    table
        .word(MSW, 0xfff0)
        .word(CS, 0x0008)
        .entry(CS_CACHE, 0, 0x9b, 0xffff)
        .word(SS, 0x0010)
        .word(DS, 0x0018)
        .entry(DS_CACHE, 0x30_0000, 0x93, 0x0fff)
        .word(ES, 0x0020)
        .entry(ES_CACHE, 0, 0x13, 0xffff)
        .word(FLAGS, 0x3002)
        .entry(GDTR, 0x2000, 0, 0x1f)
        .entry(IDTR, 0x3000, 0, 0x7ff)
        .word(LDT, 0x0028)
        .entry(LDT_CACHE, 0x4000, 0x82, 0x37)
        .word(TR, 0x0030)
        .entry(TSS_CACHE, 0x5000, 0x83, 0x2b);
    table.write(&vm);
    vm.write_bytes(0x30_0000, &0x55aau16.to_le_bytes()).unwrap();
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.legacy.cr0 & 0xf, 1);
    assert_eq!(cpu.cpu.legacy.cpl, 0);
    assert_eq!(cpu.cpu.rflags_system & 0x7000, 0x3000);
    assert_eq!(
        (cpu.cpu.legacy.gdtr, cpu.cpu.legacy.idtr),
        (
            TableRegister {
                base: 0x2000,
                limit: 0x1f,
                reserved: 0
            },
            TableRegister {
                base: 0x3000,
                limit: 0x7ff,
                reserved: 0
            }
        )
    );
    assert_eq!(
        (cpu.cpu.legacy.ldtr, cpu.cpu.legacy.tr),
        (
            SystemSegment {
                selector: 0x28,
                reserved: 0,
                cache: cache(0x4000, 0x37, 0x82)
            },
            SystemSegment {
                selector: 0x30,
                reserved: 0,
                cache: cache(0x5000, 0x2b, 0x83)
            }
        )
    );
    assert_eq!(
        cpu.cpu.legacy.segments[2],
        SegmentCache {
            unusable: 1,
            ..cache(0, 0xffff, 0x13)
        }
    );
    steps(&mut cpu, &vm, 1);
    assert_eq!(cpu.cpu.gpr[0] & 0xffff, 0x55aa);
    fault(&mut cpu, &vm, 13, Some(0));
}

#[test]
fn real_mode_loadall_with_pe_set_enters_protected_mode_cpu_model() {
    model_api::with_cpu_model(real_mode_loadall_with_pe_set_enters_protected_mode);
}

#[test]
fn real_mode_loadall_with_pe_set_enters_protected_mode() {
    // p. 1: LOADALL loads the MSW in real mode too; p. 3: the CS and SS
    // DPLs give the CPL.
    let (vm, mut cpu) = setup(b"\x0f\x05\xa1\x00\x00");
    let mut table = real_table(0x1002);
    table
        .word(MSW, 0xfff1)
        .word(CS, 0x0008)
        .entry(CS_CACHE, 0, 0x9b, 0xffff)
        .entry(DS_CACHE, 0x40_0000, 0x93, 0x0001);
    table.write(&vm);
    vm.write_bytes(0x40_0000, &0x4321u16.to_le_bytes()).unwrap();
    steps(&mut cpu, &vm, 2);
    assert_eq!(
        (cpu.cpu.legacy.cr0 & 0xf, cpu.cpu.legacy.cpl, cpu.cpu.cs),
        (1, 0, 8)
    );
    assert_eq!(cpu.cpu.gpr[0] & 0xffff, 0x4321);
}

#[test]
fn tables_outside_the_modelled_forms_are_refused_with_nothing_loaded_cpu_model() {
    model_api::with_cpu_model(tables_outside_the_modelled_forms_are_refused_with_nothing_loaded);
}

#[test]
fn tables_outside_the_modelled_forms_are_refused_with_nothing_loaded() {
    // p. 4 items 1-3 require these forms and describe no other result; the
    // engine's own limits refuse the rest (spec.md §17.10).
    type Edit = fn(&mut Table);
    let cases: &[(bool, Edit)] = &[
        // Real mode, p. 4 item 2: a read-only or an expand-down data CS.
        (false, |t| {
            t.entry(CS_CACHE, 0, 0x91, 0xffff);
        }),
        (false, |t| {
            t.entry(CS_CACHE, 0, 0x97, 0xffff);
        }),
        // Item 1: SS read-only, invalid, or code.
        (false, |t| {
            t.entry(SS_CACHE, 0, 0x91, 0xffff);
        }),
        (false, |t| {
            t.entry(SS_CACHE, 0, 0x13, 0xffff);
        }),
        (false, |t| {
            t.entry(SS_CACHE, 0, 0x9b, 0xffff);
        }),
        // Real-mode access checks every limit as expand-up.
        (false, |t| {
            t.entry(SS_CACHE, 0, 0x97, 0x0fff);
        }),
        (false, |t| {
            t.entry(DS_CACHE, 0x10_0000, 0x97, 0x0fff);
        }),
        (false, |t| {
            t.entry(ES_CACHE, 0x10_0000, 0x95, 0x0fff);
        }),
        // A valid DS with a system type.
        (false, |t| {
            t.entry(DS_CACHE, 0x10_0000, 0x82, 0x0fff);
        }),
        // A real-mode table that sets PE is checked as protected.
        (false, |t| {
            t.word(MSW, 1);
        }),
        // Protected mode: a data CS, unequal DPLs, a CS and SS at DPL 3
        // (the Machine runs only at CPL 0), a system-type ES.
        (true, |t| {
            t.entry(CS_CACHE, 0, 0x93, 0xffff);
        }),
        (true, |t| {
            t.entry(SS_CACHE, 0, 0xf3, 0xffff);
        }),
        (true, |t| {
            t.entry(CS_CACHE, 0, 0xfb, 0xffff)
                .entry(SS_CACHE, 0, 0xf3, 0xffff);
        }),
        (true, |t| {
            t.entry(ES_CACHE, 0x5000, 0x83, 0x2b);
        }),
    ];
    for (protected, edit) in cases {
        let (vm, mut cpu) = setup(b"\x0f\x05");
        let mut table = real_table(0x1002);
        if *protected {
            cpu.cpu.legacy.cr0 = 1;
            table.entry(CS_CACHE, 0, 0x9b, 0xffff);
        }
        edit(&mut table);
        table.write(&vm);
        refused(&mut cpu, &vm);
    }
}

#[test]
fn a_table_in_a_device_region_is_read_as_51_words_before_any_change_cpu_model() {
    model_api::with_cpu_model(a_table_in_a_device_region_is_read_as_51_words_before_any_change);
}

#[test]
fn a_table_in_a_device_region_is_read_as_51_words_before_any_change() {
    // p. 2: 102 bytes in 51 bus cycles, the rows that load no register
    // included. The document gives the count, not the order: the engine
    // reads ascending words, and each device stop restarts LOADALL with
    // nothing loaded.
    let mut vm = Vm::with_backend(VmConfig::flat(0x100_0000), Box::new(NoBackend));
    vm.map(0, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.map(0x1000, 0xff_f000, Prot::RWX, RegionKind::Ram)
        .unwrap();
    model_api::select(&mut vm, MODEL);
    vm.write_bytes(0x1000, b"\x0f\x05").unwrap();
    let mut table = real_table(0x1002);
    table
        .word(DS, 0x1234)
        .entry(DS_CACHE, 0x80_0000, 0x93, 0xffff)
        .word(BX, 0xb0b0);
    let mut cpu = vm.new_vcpu();
    cpu.cpu = CpuState::legacy_real_entry([0; 6]);
    cpu.cpu.rip = 0x1000;
    // The architectural registers; the MMIO completion slots change.
    let registers = |cpu: &Vcpu| {
        let c = &cpu.cpu;
        format!(
            "{:?}",
            (
                c.gpr,
                c.rip,
                c.flags,
                c.rflags_system,
                [c.cs, c.ds, c.es, c.ss],
                c.legacy
            )
        )
    };
    let before = registers(&cpu);
    let mut reads = Vec::new();
    loop {
        let run = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        match run.stop() {
            StepStop::Exit(Exit::MmioRead { addr, size: 2, .. }) => {
                assert_eq!(run.steps(), 0);
                assert_eq!(registers(&cpu), before);
                reads.push(*addr);
                let at = (*addr - 0x800) as usize;
                cpu.complete_mmio_read(u64::from(u16::from_le_bytes([
                    table.0[at],
                    table.0[at + 1],
                ])));
            }
            StepStop::Exit(Exit::BudgetExhausted) => break,
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(reads, (0..51).map(|i| 0x800 + 2 * i).collect::<Vec<u64>>());
    assert_eq!(
        (cpu.cpu.rip, cpu.cpu.ds, cpu.cpu.gpr[3] & 0xffff),
        (0x1002, 0x1234, 0xb0b0)
    );
    assert_eq!(cpu.cpu.legacy.segments[1], cache(0x80_0000, 0xffff, 0x93));
}

#[test]
fn opcode_0f_04_stays_undefined_cpu_model() {
    model_api::with_cpu_model(opcode_0f_04_stays_undefined);
}

#[test]
fn opcode_0f_04_stays_undefined() {
    // p. 13 calls LOADALL "opcode 0F04H" and gives it exception 13 at
    // CPL > 0; p. 2's encoding governs (spec.md §17.10), and 0F 04 keeps the
    // 80286 #UD (I286 PRM App. B #UD p. B-12) in both modes. The table area
    // is a device region, so a read of it would stop with an MMIO exit.
    for protected in [false, true] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x100_0000), Box::new(NoBackend));
        vm.map(0, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.map(0x1000, 0xff_f000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        model_api::select(&mut vm, MODEL);
        vm.write_bytes(0x1000, b"\x0f\x04").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu = CpuState::legacy_real_entry([0; 6]);
        cpu.cpu.rip = 0x1000;
        cpu.cpu.legacy.cr0 = u32::from(protected);
        fault(&mut cpu, &vm, 6, None);
    }
}

#[test]
fn other_generations_keep_the_undefined_opcode() {
    // 80386 PRM App. A two-byte map pp. A-6-A-7 leaves 0F 05 reserved, and
    // the LOADALL description says the instruction "will not work on an
    // iAPX 386" (p. 1). The table area is a device region, so a read of it
    // would stop with an MMIO exit instead of the #UD.
    let i386 = LegacyCpuModel {
        generation: CpuGeneration::I386,
        model_semantics_id: 0x386,
    };
    for cpu_model in [false, true] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x100_0000), Box::new(NoBackend));
        vm.map(0, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
        vm.map(0x1000, 0xff_f000, Prot::RWX, RegionKind::Ram)
            .unwrap();
        if cpu_model {
            vm.set_cpu_model(CpuModel::I80386);
            vm.set_exception_delivery(x86jit_core::ExceptionDelivery::Outward);
        } else {
            #[allow(deprecated)]
            vm.set_legacy_cpu_model(i386);
        }
        vm.write_bytes(0x1000, b"\x0f\x05").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu = CpuState::legacy_real_entry([0; 6]);
        cpu.cpu.rip = 0x1000;
        fault(&mut cpu, &vm, 6, None);
    }
}

#[test]
fn application_profiles_keep_syscall() {
    // The Application decoders never enable LOADALL decoding: 0F 05 stays
    // SYSCALL in every Application mode, lifted and executed as before.
    for mode in [CpuMode::Long64, CpuMode::Compat32, CpuMode::Real16] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10_0000), Box::new(InterpreterBackend));
        vm.set_cpu_mode(mode);
        vm.map(0, 0x10_0000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\x0f\x05").unwrap();
        real_table(0x2000).write(&vm);
        let block = lift_block(
            &vm.mem,
            FetchAddr {
                pa: 0x1000,
                ip: 0x1000,
            },
            mode,
        )
        .unwrap();
        assert!(
            block
                .ops
                .iter()
                .any(|op| matches!(op, IrOp::Syscall { is_amd64: true })),
            "{mode:?}: {:?}",
            block.ops
        );
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        cpu.cpu.gpr[4] = 0x8000;
        let exit = cpu.run(&vm, Some(1));
        assert!(matches!(exit, Exit::Syscall), "{mode:?}: {exit:?}");
    }
}

#[test]
fn real_mode_code_and_read_only_caches_load_cpu_model() {
    model_api::with_cpu_model(real_mode_code_and_read_only_caches_load);
}

#[test]
fn real_mode_code_and_read_only_caches_load() {
    // p. 4 item 2 lists execute-only and execute/read code for CS. The
    // engine's real-mode CS is 9Bh, so a far jump leaves that cache, and a
    // table that repeats it loads. Read-only and code DS and ES load too.
    // Real-mode access checks no rights (spec.md §17.10): the write through
    // the read-only DS and the read through execute-only code, which the
    // 80286 refuses with exception 13 (p. 1), succeed.
    for (cs, ds, es) in [(0x9b, 0x91, 0x9b), (0x98, 0x90, 0x99), (0x9e, 0x93, 0x9d)] {
        let (vm, mut cpu) = setup(b"\xea\x00\x11\x00\x00");
        vm.write_bytes(0x1100, b"\x0f\x05").unwrap();
        // MOV AX,[0000]; MOV [0002],AX; MOV BX,ES:[0000]; HLT.
        vm.write_bytes(0x1200, b"\xa1\x00\x00\xa3\x02\x00\x26\x8b\x1e\x00\x00\xf4")
            .unwrap();
        let mut table = real_table(0x1200);
        table
            .entry(CS_CACHE, 0, cs, 0xffff)
            .word(DS, 0x1000)
            .entry(DS_CACHE, 0x10_0000, ds, 0xffff)
            .word(ES, 0x2000)
            .entry(ES_CACHE, 0x20_0000, es, 0xffff);
        table.write(&vm);
        vm.write_bytes(0x10_0000, &0x1357u16.to_le_bytes()).unwrap();
        vm.write_bytes(0x20_0000, &0x2468u16.to_le_bytes()).unwrap();
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.legacy.segments[0], cache(0, 0xffff, 0x9b));
        steps(&mut cpu, &vm, 1);
        assert_eq!(
            cpu.cpu.legacy.segments[..3],
            [
                cache(0, 0xffff, cs.into()),
                cache(0x10_0000, 0xffff, ds.into()),
                cache(0x20_0000, 0xffff, es.into())
            ],
            "{cs:02x} {ds:02x} {es:02x}"
        );
        assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
        assert_eq!(word(&vm, 0x10_0002), 0x1357);
        assert_eq!(cpu.cpu.gpr[3] & 0xffff, 0x2468);
    }
}

#[test]
fn a_conforming_cs_and_a_nonzero_table_byte_three_load_cpu_model() {
    model_api::with_cpu_model(a_conforming_cs_and_a_nonzero_table_byte_three_load);
}

#[test]
fn a_conforming_cs_and_a_nonzero_table_byte_three_load() {
    // p. 4 item 2 names capabilities, and a conforming code segment has
    // one. GDTR and IDTR byte 3 "should be zeroes" (p. 3) and has no field
    // to load: it is ignored, as LGDT and LIDT ignore the last byte of
    // their operand (I286 PRM App. B LGDT/LIDT p. B-65). DERIVED.
    let (vm, mut cpu) = protected_setup();
    let mut table = real_table(0x1002);
    table
        .word(CS, 0x0008)
        .entry(CS_CACHE, 0, 0x9f, 0xffff)
        .word(SS, 0x0010)
        .word(DS, 0x0018)
        .entry(DS_CACHE, 0x30_0000, 0x93, 0x0fff)
        .word(ES, 0x0020)
        .entry(GDTR, 0x2000, 0x01, 0x1f)
        .entry(IDTR, 0x3000, 0xff, 0x7ff);
    table.write(&vm);
    vm.write_bytes(0x30_0000, &0x55aau16.to_le_bytes()).unwrap();
    steps(&mut cpu, &vm, 2);
    assert_eq!((cpu.cpu.legacy.cr0 & 1, cpu.cpu.legacy.cpl), (1, 0));
    assert_eq!(cpu.cpu.legacy.segments[0], cache(0, 0xffff, 0x9f));
    assert_eq!(
        (cpu.cpu.legacy.gdtr, cpu.cpu.legacy.idtr),
        (
            TableRegister {
                base: 0x2000,
                limit: 0x1f,
                reserved: 0
            },
            TableRegister {
                base: 0x3000,
                limit: 0x7ff,
                reserved: 0
            }
        )
    );
    assert_eq!(cpu.cpu.gpr[0] & 0xffff, 0x55aa);
}

#[test]
fn a_valid_ldt_cache_serves_the_ldt_under_a_null_selector_cpu_model() {
    model_api::with_cpu_model(a_valid_ldt_cache_serves_the_ldt_under_a_null_selector);
}

#[test]
fn a_valid_ldt_cache_serves_the_ldt_under_a_null_selector() {
    // p. 2: LOADALL loads the LDT cache directly; p. 3: no check is made
    // between a selector and its descriptor, and the present bit is the
    // valid bit. LLDT with a null selector still marks the LDTR invalid
    // (I286 PRM App. B LLDT p. B-66), so a later LDT reference raises
    // #GP(selector). DERIVED (spec.md §17.10).
    let (vm, mut cpu) = setup(b"\x0f\x05");
    // MOV AX,4; MOV DS,AX; MOV BX,[0000]; XOR CX,CX; LLDT CX; MOV DS,AX.
    vm.write_bytes(
        0x2000,
        b"\xb8\x04\x00\x8e\xd8\x8b\x1e\x00\x00\x31\xc9\x0f\x00\xd1\x8e\xd8",
    )
    .unwrap();
    cpu.cpu.legacy.cr0 = 1;
    let mut table = real_table(0x2000);
    table
        .word(CS, 0x0008)
        .entry(CS_CACHE, 0, 0x9b, 0xffff)
        .word(LDT, 0)
        .entry(LDT_CACHE, 0x6000, 0x82, 0x000f);
    table.write(&vm);
    // LDT entry 0: limit FFFFh, base 070000h, present writable data (I286
    // PRM §6.5 p. 6-8 and Fig. 6-7 p. 6-9, scan checked).
    vm.write_bytes(0x6000, &[0xff, 0xff, 0x00, 0x00, 0x07, 0x93, 0, 0])
        .unwrap();
    vm.write_bytes(0x7_0000, &0x4242u16.to_le_bytes()).unwrap();
    steps(&mut cpu, &vm, 4);
    assert_eq!(
        (cpu.cpu.ds, cpu.cpu.legacy.segments[1]),
        (4, cache(0x7_0000, 0xffff, 0x93))
    );
    assert_eq!(cpu.cpu.gpr[3] & 0xffff, 0x4242);
    steps(&mut cpu, &vm, 2);
    assert_eq!(cpu.cpu.legacy.ldtr.cache.unusable, 1);
    fault(&mut cpu, &vm, 13, Some(4));
}

#[test]
fn an_invalid_cache_faults_on_use_in_real_mode_cpu_model() {
    model_api::with_cpu_model(an_invalid_cache_faults_on_use_in_real_mode);
}

#[test]
fn an_invalid_cache_faults_on_use_in_real_mode() {
    // p. 3: a reference through an invalid cache raises exception 13 "with
    // an error code of zero", and the value loaded can be read without an
    // exception. In real mode the engine raises interrupt 13 without an
    // error code, as for its other real-mode segment faults (I286 PRM
    // App. B MOV p. B-74; spec.md §17.10).
    let (vm, mut cpu) = setup(b"\x0f\x05\x8c\xd8\xa1\x00\x00");
    let mut table = real_table(0x1002);
    table
        .word(DS, 0x1234)
        .entry(DS_CACHE, 0x10_0000, 0x13, 0xffff);
    table.write(&vm);
    steps(&mut cpu, &vm, 2);
    assert_eq!(cpu.cpu.gpr[0] & 0xffff, 0x1234);
    fault(&mut cpu, &vm, 13, None);
}

#[test]
fn a_step_trap_around_loadall_cpu_model() {
    model_api::with_cpu_model(a_step_trap_around_loadall);
}

#[test]
fn a_step_trap_around_loadall() {
    // The document is silent; the engine's choice is ASSUMED (spec.md
    // §17.10). A TF set before LOADALL traps after it, saving the loaded
    // CS:IP; a TF that LOADALL loads traps after the next instruction, as
    // after IRET or a task switch (I286 PRM §9.7.1 p. 9-14).
    for tf_before in [true, false] {
        let (vm, mut cpu) = setup(b"\x0f\x05");
        vm.write_bytes(0x2000, b"\x90\xf4").unwrap();
        vm.write_bytes(0x4000, b"\xf4").unwrap();
        // Vector 1 at 0000:4000.
        vm.write_bytes(4, &[0x00, 0x40, 0x00, 0x00]).unwrap();
        let mut table = real_table(0x2000);
        if tf_before {
            cpu.cpu.rflags_system |= 0x100;
        } else {
            table.word(FLAGS, 0x0102);
        }
        table.write(&vm);
        assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x4001), "{tf_before}");
        // The frame on the loaded stack: IP, CS, FLAGS.
        assert_eq!(
            [word(&vm, 0x7ffa), word(&vm, 0x7ffc), word(&vm, 0x7ffe)],
            if tf_before {
                [0x2000, 0, 0x0002]
            } else {
                [0x2001, 0, 0x0102]
            },
            "{tf_before}"
        );
    }
}

#[test]
fn loadall_after_other_instructions_in_one_block_cpu_model() {
    model_api::with_cpu_model(loadall_after_other_instructions_in_one_block);
}

#[test]
fn loadall_after_other_instructions_in_one_block() {
    // A Direct walk reads 0F 05 as SYSCALL and ends its run before it, so
    // LOADALL runs alone after the instructions before it.
    // MOV AX,1234; MOV [0500],AX; INC AX; LOADALL.
    let (vm, mut cpu) = setup(b"\xb8\x34\x12\xa3\x00\x05\x40\x0f\x05");
    vm.write_bytes(0x2000, b"\xf4").unwrap();
    let mut table = real_table(0x2000);
    table.word(AX, 0x5555).word(BX, 0x6666);
    table.write(&vm);
    assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
    assert_eq!(word(&vm, 0x500), 0x1234);
    assert_eq!(
        (
            cpu.cpu.gpr[0] & 0xffff,
            cpu.cpu.gpr[3] & 0xffff,
            cpu.cpu.rip
        ),
        (0x5555, 0x6666, 0x2001)
    );
}

#[test]
fn every_real_mode_data_segment_load_ends_a_loaded_cache_cpu_model() {
    model_api::with_cpu_model(every_real_mode_data_segment_load_ends_a_loaded_cache);
}

#[test]
fn every_real_mode_data_segment_load_ends_a_loaded_cache() {
    // p. 3: "Any subsequent segment register load instruction will reload
    // the associated descriptor cache register in the normal manner".
    // PUSH 0040; POP DS; LES DI,[0600]; LDS SI,[0604]; PUSH 0070; POP SS.
    let (vm, mut cpu) = setup(b"\x0f\x05");
    vm.write_bytes(
        0x1002,
        b"\x68\x40\x00\x1f\xc4\x3e\x00\x06\xc5\x36\x04\x06\x68\x70\x00\x17\xf4",
    )
    .unwrap();
    let mut table = real_table(0x1002);
    table
        .entry(DS_CACHE, 0x12_3450, 0x93, 0x00ff)
        .entry(ES_CACHE, 0x13_0000, 0x93, 0x0fff)
        .entry(SS_CACHE, 0x12_0000, 0x93, 0xffff);
    table.write(&vm);
    // The far pointers at 0040:0600 and 0040:0604.
    vm.write_bytes(0xa00, &[0x11, 0x11, 0x50, 0x00, 0x22, 0x22, 0x60, 0x00])
        .unwrap();
    assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt));
    // Both pushes went through the loaded SS base.
    assert_eq!(word(&vm, 0x12_7ffe), 0x0070);
    assert_eq!((cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss), (0x60, 0x50, 0x70));
    assert_eq!(
        cpu.cpu.legacy.segments[1..4],
        [
            cache(0x600, 0xffff, 0x93),
            cache(0x500, 0xffff, 0x93),
            cache(0x700, 0xffff, 0x93)
        ]
    );
    assert_eq!(
        (cpu.cpu.gpr[6] & 0xffff, cpu.cpu.gpr[7] & 0xffff),
        (0x2222, 0x1111)
    );
}

#[test]
fn every_real_mode_far_transfer_ends_a_loaded_cs_limit_cpu_model() {
    model_api::with_cpu_model(every_real_mode_far_transfer_ends_a_loaded_cs_limit);
}

#[test]
fn every_real_mode_far_transfer_ends_a_loaded_cs_limit() {
    // p. 3: a real-mode CS load resets the limit to FFFFH. Each target,
    // 0000:3000, lies past the loaded limit 2FFFh.
    for (name, code) in [
        ("CALL FAR", &b"\x9a\x00\x30\x00\x00"[..]),
        // PUSH 0; PUSH 3000; RETF.
        ("RETF", b"\x6a\x00\x68\x00\x30\xcb"),
        // PUSH 2; PUSH 0; PUSH 3000; IRET.
        ("IRET", b"\x6a\x02\x6a\x00\x68\x00\x30\xcf"),
        // Vector 20h at 0000:3000.
        ("INT 20h", b"\xcd\x20"),
    ] {
        let (vm, mut cpu) = setup(b"\x0f\x05");
        vm.write_bytes(0x1002, code).unwrap();
        vm.write_bytes(0x3000, b"\xf4").unwrap();
        vm.write_bytes(0x80, &[0x00, 0x30, 0x00, 0x00]).unwrap();
        let mut table = real_table(0x1002);
        table.entry(CS_CACHE, 0, 0x93, 0x2fff);
        table.write(&vm);
        steps(&mut cpu, &vm, 1);
        assert!(matches!(cpu.run(&vm, Some(10)), Exit::Hlt), "{name}");
        assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0, 0x3001), "{name}");
        assert_eq!(cpu.cpu.legacy.segments[0], cache(0, 0xffff, 0x9b), "{name}");
    }
}

#[test]
fn one_table_sets_every_register_cpu_model() {
    model_api::with_cpu_model(one_table_sets_every_register);
}

#[test]
fn one_table_sets_every_register() {
    // p. 2: every row of the table, each with its own value, in protected
    // mode at CPL 0; the rows that load no register hold 5Ah. The program
    // then observes the registers through the caches LOADALL loaded.
    let (vm, mut cpu) = setup(b"\x0f\x05");
    cpu.cpu.legacy.cr0 = 1;
    let mut table = Table([0x5a; 102]);
    table
        .word(MSW, 0xfff7)
        .word(TR, 0x0030)
        .word(FLAGS, 0x7cd7)
        .word(IP, 0x2000)
        .word(LDT, 0x0028)
        .word(DS, 0x0018)
        .word(SS, 0x0010)
        .word(CS, 0x0008)
        .word(ES, 0x0020)
        .word(DI, 0x1001)
        .word(SI, 0x2002)
        .word(BP, 0x3003)
        .word(SP, 0x8000)
        .word(BX, 0x5005)
        .word(DX, 0x6006)
        .word(CX, 0x7007)
        .word(AX, 0x8008)
        .entry(ES_CACHE, 0x50_0000, 0x92, 0x7fff)
        .entry(CS_CACHE, 0x01_0000, 0x9b, 0xffff)
        .entry(SS_CACHE, 0x2a_0000, 0x93, 0xffff)
        .entry(DS_CACHE, 0x30_0000, 0x93, 0x0fff)
        .entry(GDTR, 0x06_0000, 0, 0x00ff)
        .entry(LDT_CACHE, 0x04_5678, 0x82, 0x0037)
        .entry(IDTR, 0x07_0000, 0, 0x07ff)
        .entry(TSS_CACHE, 0x05_1234, 0x83, 0x002b);
    table.write(&vm);
    // PUSH AX; PUSHF; SMSW [0]; STR [2]; SLDT [4]; SGDT [6]; SIDT [C];
    // MOV [12],CS; MOV [14],SS; MOV [16],DS; MOV [18],ES;
    // MOV AX,ES:[0]; MOV [1A],AX; HLT.
    vm.write_bytes(
        0x1_2000,
        b"\x50\x9c\x0f\x01\x26\x00\x00\x0f\x00\x0e\x02\x00\x0f\x00\x06\x04\x00\
          \x0f\x01\x06\x06\x00\x0f\x01\x0e\x0c\x00\x8c\x0e\x12\x00\x8c\x16\x14\x00\
          \x8c\x1e\x16\x00\x8c\x06\x18\x00\x26\xa1\x00\x00\xa3\x1a\x00\xf4",
    )
    .unwrap();
    vm.write_bytes(0x50_0000, &0x9abcu16.to_le_bytes()).unwrap();
    steps(&mut cpu, &vm, 1);
    let c = &cpu.cpu;
    assert_eq!((c.legacy.cr0 & 0xffff, c.legacy.cpl), (7, 0));
    assert_eq!(
        (c.cs, c.ds, c.es, c.ss, c.rip),
        (0x08, 0x18, 0x20, 0x10, 0x2000)
    );
    assert_eq!(
        c.gpr[..8].iter().map(|r| r & 0xffff).collect::<Vec<_>>(),
        [0x8008, 0x7007, 0x6006, 0x5005, 0x8000, 0x3003, 0x2002, 0x1001]
    );
    assert_eq!(
        (c.flags.to_flags16() & 0x0fd7, c.rflags_system & 0x7100),
        (0x0cd7, 0x7000)
    );
    assert_eq!(
        c.legacy.segments[..4],
        [
            cache(0x01_0000, 0xffff, 0x9b),
            cache(0x30_0000, 0x0fff, 0x93),
            cache(0x50_0000, 0x7fff, 0x92),
            cache(0x2a_0000, 0xffff, 0x93)
        ]
    );
    assert_eq!(
        (c.legacy.gdtr, c.legacy.idtr),
        (
            TableRegister {
                base: 0x06_0000,
                limit: 0xff,
                reserved: 0
            },
            TableRegister {
                base: 0x07_0000,
                limit: 0x7ff,
                reserved: 0
            }
        )
    );
    assert_eq!(
        (c.legacy.ldtr, c.legacy.tr),
        (
            SystemSegment {
                selector: 0x28,
                reserved: 0,
                cache: cache(0x04_5678, 0x37, 0x82)
            },
            SystemSegment {
                selector: 0x30,
                reserved: 0,
                cache: cache(0x05_1234, 0x2b, 0x83)
            }
        )
    );
    assert!(matches!(cpu.run(&vm, Some(20)), Exit::Hlt));
    // The stack through the SS cache: AX, then the protected FLAGS image
    // with IOPL and NT (I286 PRM App. B IRET pp. B-51–52).
    assert_eq!(
        [word(&vm, 0x2a_7ffe), word(&vm, 0x2a_7ffc)],
        [0x8008, 0x7cd7]
    );
    // The stores through the DS cache.
    let mut stored = [0; 0x1c];
    vm.read_bytes(0x30_0000, &mut stored).unwrap();
    let at = |offset: usize| u16::from_le_bytes([stored[offset], stored[offset + 1]]);
    assert_eq!(at(0) & 0xf, 7);
    assert_eq!([at(2), at(4)], [0x30, 0x28]);
    assert_eq!(stored[6..11], [0xff, 0x00, 0x00, 0x00, 0x06]);
    assert_eq!(stored[0xc..0x11], [0xff, 0x07, 0x00, 0x00, 0x07]);
    assert_eq!(
        [at(0x12), at(0x14), at(0x16), at(0x18), at(0x1a)],
        [0x08, 0x10, 0x18, 0x20, 0x9abc]
    );
}

#[test]
fn prefixed_loadall_cpu_model() {
    model_api::with_cpu_model(prefixed_loadall);
}

#[test]
fn prefixed_loadall() {
    // 66h and 67h do not exist on the 80286 (I286 PRM App. B #UD p. B-12).
    // The 80286 entry admits LOCK on data movement only (App. C item 11
    // p. C-2); other forms keep the #UD (legacy_i286_quirks.rs). A segment
    // or repeat prefix selects nothing for a fixed physical table and is
    // ignored (ASSUMED: the document is silent).
    for (code, executes) in [
        (&b"\x66\x0f\x05"[..], false),
        (b"\x67\x0f\x05", false),
        (b"\xf0\x0f\x05", false),
        (b"\x2e\x0f\x05", true),
        (b"\x26\x0f\x05", true),
        (b"\xf3\x0f\x05", true),
        (b"\xf2\x0f\x05", true),
    ] {
        let (vm, mut cpu) = setup(code);
        let mut table = real_table(0x2000);
        table.word(AX, 0x5555);
        table.write(&vm);
        if executes {
            steps(&mut cpu, &vm, 1);
            assert_eq!(
                (cpu.cpu.rip, cpu.cpu.gpr[0] & 0xffff),
                (0x2000, 0x5555),
                "{code:02x?}"
            );
        } else {
            fault(&mut cpu, &vm, 6, None);
        }
    }
}
