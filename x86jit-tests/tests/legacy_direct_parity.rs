//! Decision-23 §5: Direct and Checked must expose the same Machine behavior.
use x86jit_core::{
    AccessKind, CpuModel, ExceptionDelivery, Exit, FaultAccess, Knowledge, Prot, RegionKind,
    SegmentId, StepOptions, StepStop, Vcpu, Vm, VmConfig,
};
#[cfg(debug_assertions)]
use x86jit_core::{BlockKey, CachedBlock};

fn setup(model: CpuModel, checked: bool, ip: u64, code: &[u8]) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(0x110000));
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    for (base, len) in [(0, 0x3000), (0x3000, 0x1000), (0x4000, 0x1c000)] {
        vm.map(base, len, Prot::RWX, RegionKind::Ram).unwrap();
    }
    vm.write_bytes(ip, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = ip;
    cpu.cpu.gpr[4] = 0x8800;
    for i in 0..32 {
        cpu.cpu.xmm[i] = (i as u128 + 1) << 80;
        cpu.cpu.ymm_hi[i] = (i as u128 + 2) << 64;
        cpu.cpu.zmm_hi[i] = [i as u128 + 3, i as u128 + 4];
    }
    cpu.cpu.kmask = [0x5a5a; 8];
    cpu.cpu.fpu_fip = 0x123456;
    cpu.cpu.fpu_fdp = 0x654321;
    if checked {
        // ES=92h withholds the Direct proof without changing these real accesses.
        // This forcing trick applies only while ES is not reloaded by the guest.
        cpu.cpu.legacy.segments[SegmentId::Es as usize].attributes = 0x92;
    }
    (vm, cpu)
}

#[derive(Debug, PartialEq, Eq)]
enum Stop {
    Budget,
    Hlt,
    Exception(u64, u8),
    Read(u64, u8),
    Write(u64, u8, u64),
    Unmapped(u64, AccessKind),
    Other(String),
}

#[derive(PartialEq, Eq)]
struct Observation {
    stop: Stop,
    steps: u64,
    retired: u64,
    gpr: [u64; 16],
    ip: u64,
    flags: u64,
    selectors: [u16; 6],
    architecture: String,
    exit: String,
    evidence: Option<(Knowledge<u64>, Knowledge<u16>, Knowledge<FaultAccess>)>,
    memory: Vec<Option<Vec<u8>>>,
}

impl std::fmt::Debug for Observation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Observation")
            .field("stop", &self.stop)
            .field("exit", &self.exit)
            .field("steps", &self.steps)
            .field("retired", &self.retired)
            .field("gpr", &self.gpr)
            .field("ip", &self.ip)
            .finish_non_exhaustive()
    }
}

/// The shared block at the entry point. The temp verifier it feeds exists only
/// in debug builds of the core (`ir.rs`).
#[cfg(debug_assertions)]
fn entry_key(cpu: &Vcpu) -> BlockKey {
    let a20_enabled = cpu.a20_enabled();
    let physical = ((u64::from(cpu.cpu.cs) << 4) + cpu.cpu.rip)
        & if a20_enabled { u64::MAX } else { !(1 << 20) };
    BlockKey::machine_direct(physical, 0, a20_enabled)
}

fn run(vm: &Vm, cpu: &mut Vcpu, budget: u64, trap: bool) -> Observation {
    #[cfg(debug_assertions)]
    let key = entry_key(cpu);
    let forced_checked = cpu.cpu.legacy.segments[SegmentId::Es as usize].attributes == 0x92;
    let result = cpu.run_steps(vm, budget, StepOptions::default()).unwrap();
    #[cfg(debug_assertions)]
    if let Some(CachedBlock::Interpreted(ir)) = vm.cache.get(key) {
        assert_eq!(ir.verify_temp_defs(), Ok(()));
    }
    let stop = match result.stop() {
        StepStop::Exit(Exit::BudgetExhausted) => Stop::Budget,
        StepStop::Exit(Exit::Hlt) => Stop::Hlt,
        StepStop::Exit(Exit::Exception { addr, vector, .. }) => Stop::Exception(*addr, *vector),
        StepStop::Exit(Exit::MmioRead { addr, size, .. }) => Stop::Read(*addr, *size),
        StepStop::Exit(Exit::MmioWrite {
            addr, size, value, ..
        }) => Stop::Write(*addr, *size, *value),
        StepStop::Exit(Exit::UnmappedMemory { addr, access, .. }) => Stop::Unmapped(*addr, *access),
        other => Stop::Other(format!("{other:?}")),
    };
    let memory = (0..0x110000)
        .step_by(0x1000)
        .map(|base| {
            if trap && base == 0x3000 {
                return None;
            }
            let mut bytes = vec![0; 0x1000];
            match vm.read_bytes(base, &mut bytes) {
                Ok(()) => Some(bytes),
                // Fixtures map at page-aligned bases; retain RAM in a partial
                // final page as well as ordinary whole pages.
                Err(_) if vm.read_bytes(base, &mut bytes[..1]).is_ok() => {
                    for (i, byte) in bytes.iter_mut().enumerate() {
                        let _ = vm.read_bytes(base + i as u64, std::slice::from_mut(byte));
                    }
                    Some(bytes)
                }
                Err(_) => None,
            }
        })
        .collect();
    // Compare every CPU field, including hidden caches/system, x87 and vector
    // state. Only the fixture's deliberate ES attribute and private retry
    // bookkeeping/identity differ between the two execution implementations.
    let mut architectural = cpu.cpu.clone();
    if forced_checked {
        assert_eq!(
            architectural.legacy.segments[SegmentId::Es as usize].attributes,
            0x92,
            "forcing cache changed"
        );
        assert_eq!(architectural.es, 0, "ES forcing fixture was reloaded");
        architectural.legacy.segments[SegmentId::Es as usize].attributes = 0x93;
    }
    architectural.mmio_parts = [(0, 0); 64];
    architectural.mmio_parts_len = 0;
    architectural.mmio_parts_rip = 0;
    architectural.mmio_retry_cs = 0;
    architectural.mmio_retry_active = false;
    architectural.source_id = 0;
    Observation {
        stop,
        exit: format!("{:?}", result.stop()),
        evidence: match result.stop() {
            StepStop::Exit(exit) => exit
                .fault_metadata()
                .map(|m| (m.operand_base(), m.operand_size(), m.access())),
            _ => None,
        },
        architecture: format!("{architectural:?}"),
        steps: result.steps(),
        retired: cpu.retired_instructions(),
        gpr: cpu.cpu.gpr,
        ip: cpu.cpu.rip,
        flags: cpu.cpu.rflags(),
        selectors: [
            cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs,
        ],
        memory,
    }
}

#[test]
fn scalar_mmio_rmw_completes_one_read_and_one_write() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let outcomes: Vec<_> = [true, false]
            .into_iter()
            .map(|checked| {
                let (mut vm, mut cpu) = setup(model, checked, 0x1000, b"\x83\x06\x00\x30\x01");
                vm.unmap(0x3000, 0x1000).unwrap();
                vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                let read = run(&vm, &mut cpu, 1, true);
                assert_eq!(
                    (read.steps, read.retired, &read.stop),
                    (0, 0, &Stop::Read(0x3000, 2))
                );
                cpu.complete_mmio_read(4);
                let write = run(&vm, &mut cpu, 1, true);
                assert_eq!(
                    (write.steps, write.retired, &write.stop),
                    (0, 0, &Stop::Write(0x3000, 2, 5))
                );
                cpu.complete_mmio_write();
                let done = run(&vm, &mut cpu, 1, true);
                assert_eq!(
                    (done.steps, done.retired, &done.stop),
                    (1, 1, &Stop::Budget),
                    "{model:?}, checked={checked}"
                );
                assert!(!cpu.cpu.mmio_retry_active);
                cpu.cpu.rip = 0x1000;
                let again = run(&vm, &mut cpu, 1, true);
                assert_eq!(
                    (&again.stop, again.steps, again.retired),
                    (&Stop::Read(0x3000, 2), 0, 1)
                );
                vec![read, write, done, again]
            })
            .collect();
        assert_eq!(outcomes[0], outcomes[1]);
    }
}

#[test]
fn pop_uses_sp_after_a_32_bit_esp_write() {
    // 80386 PRM §17.1: SS.B selects the stack-address size independently of 66h.
    let outcomes: Vec<_> = [true, false]
        .into_iter()
        .map(|checked| {
            let (vm, mut cpu) = setup(
                CpuModel::I80386,
                checked,
                0x1000,
                b"\x66\xbc\x00\x80\x01\x00\x58",
            );
            vm.write_bytes(0x8000, &0x1234u16.to_le_bytes()).unwrap();
            vm.write_bytes(0x18000, &0xabcdu16.to_le_bytes()).unwrap();
            let first = run(&vm, &mut cpu, 1, false);
            let second = run(&vm, &mut cpu, 1, false);
            assert_eq!(
                (second.gpr[0], second.gpr[4]),
                (0x1234, 0x18002),
                "checked={checked}"
            );
            vec![first, second]
        })
        .collect();
    assert_eq!(outcomes[0], outcomes[1]);
}

#[test]
fn stack_whole_operand_limit_faults_before_update() {
    // 80386 PRM §14.7 item 7 and Table 14-1: crossing SS faults before access.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for (opcode, sp) in [(0x58, 0xffff), (0x50, 1)] {
            let outcomes: Vec<_> = [true, false]
                .into_iter()
                .map(|checked| {
                    let (vm, mut cpu) = setup(model, checked, 0x1000, &[opcode]);
                    cpu.cpu.gpr[4] = sp;
                    cpu.cpu.gpr[0] = 0x5678;
                    let result = run(&vm, &mut cpu, 1, false);
                    let vector = if model == CpuModel::I80286 { 13 } else { 12 };
                    assert_eq!(
                        (
                            &result.stop,
                            result.steps,
                            result.retired,
                            result.ip,
                            result.gpr[4],
                            result.gpr[0]
                        ),
                        (&Stop::Exception(0x1000, vector), 0, 0, 0x1000, sp, 0x5678),
                        "{model:?}, checked={checked}, opcode={opcode:x}"
                    );
                    result
                })
                .collect();
            assert_eq!(outcomes[0], outcomes[1]);
        }
    }
}

#[test]
fn operand_override_loop_target_keeps_32_bits() {
    // 80386 PRM §17 LOOP/JCXZ: operand size controls the branch target width.
    for opcode in [0xe0, 0xe1, 0xe2, 0xe3] {
        let outcomes: Vec<_> = [true, false]
            .into_iter()
            .map(|checked| {
                let (vm, mut cpu) = setup(CpuModel::I80386, checked, 0xfff0, &[0x66, opcode, 0x20]);
                cpu.cpu.gpr[1] = if opcode == 0xe3 { 0 } else { 2 };
                cpu.cpu.flags.zf = opcode == 0xe1;
                let result = run(&vm, &mut cpu, 1, false);
                assert_eq!(
                    (&result.stop, result.steps, result.retired, result.ip),
                    (&Stop::Exception(0xfff0, 13), 0, 0, 0xfff0),
                    "opcode={opcode:x}, checked={checked}"
                );
                result
            })
            .collect();
        assert_eq!(outcomes[0], outcomes[1]);
    }
}

#[test]
fn edited_eip_outside_cs_faults_before_fetch() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let outcomes: Vec<_> = [true, false]
            .into_iter()
            .map(|checked| {
                let (vm, mut cpu) = setup(model, checked, 0x1000, b"\x90");
                let first = run(&vm, &mut cpu, 1, false);
                let mut results = vec![first];
                for ip in [0x10000, 0x10001, 0xffff_ffff] {
                    cpu.cpu.rip = ip;
                    let fault = run(&vm, &mut cpu, 1, false);
                    assert_eq!(
                        (&fault.stop, fault.steps, fault.retired),
                        (&Stop::Exception(ip, 13), 0, 1)
                    );
                    results.push(fault);
                }
                results
            })
            .collect();
        assert_eq!(outcomes[0], outcomes[1]);
    }
}

#[test]
fn out_of_limit_eip_cannot_reuse_a_valid_physical_cache_alias() {
    let results: Vec<_> = [true, false]
        .into_iter()
        .map(|checked| {
            let (vm, mut cpu) = setup(CpuModel::I80386, checked, 0x10001, b"\x90");
            cpu.cpu.cs = 0x1000;
            cpu.cpu.legacy.segments[SegmentId::Cs as usize].base = 0x10000;
            cpu.cpu.rip = 1;
            let first = run(&vm, &mut cpu, 1, false);
            cpu.cpu.cs = 0;
            cpu.cpu.legacy.segments[SegmentId::Cs as usize].base = 0;
            cpu.cpu.rip = 0x10001;
            let second = run(&vm, &mut cpu, 1, false);
            assert_eq!(
                (&second.stop, second.steps, second.retired),
                (&Stop::Exception(0x10001, 13), 0, 1)
            );
            vec![first, second]
        })
        .collect();
    assert_eq!(results[0], results[1]);
}

#[test]
fn stack_forms_match_checked_at_wrap_and_with_high_esp_bits() {
    // 80386 PRM §17.1 and §14.7 item 7: stack addressing and whole-operand
    // limits also govern flag images, interrupt frames and frame instructions.
    let forms: &[&[u8]] = &[
        b"\x50",
        b"\x58",
        b"\x54",
        b"\x5c",
        b"\x68\x34\x12",
        b"\xe8\x00\x00",
        b"\xff\xd0",
        b"\xc3",
        b"\xc2\x04\x00",
        b"\x9c",
        b"\x9d",
        b"\xcd\x20",
        b"\xcf",
        b"\xc8\x04\x00\x00",
        b"\xc8\x04\x00\x02",
        b"\xc9",
        b"\x60",
        b"\x61",
        b"\x0e",
        b"\x1f",
        b"\xff\x36\x00\x80",
        b"\x8f\x06\x00\x80",
    ];
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for form in forms {
            for prefix in [&[][..], &[0x66][..], &[0x67][..], &[0x66, 0x67][..]] {
                if model == CpuModel::I80286 && !prefix.is_empty() {
                    continue;
                }
                for sp in [0, 1, 0xfffe, 0xffff, 0x18000, 0x1ffff, 0x18001] {
                    let mut code = prefix.to_vec();
                    code.extend_from_slice(form);
                    let results: Vec<_> = [true, false]
                        .into_iter()
                        .map(|checked| {
                            let (vm, mut cpu) = setup(model, checked, 0x1000, &code);
                            for addr in [0, 0x8000, 0xfffe, 0x18000] {
                                vm.write_bytes(addr, &[0x00, 0x20, 0, 0, 2, 0, 0, 0])
                                    .unwrap();
                            }
                            vm.write_bytes(0x80, &[0, 0x20, 0, 0]).unwrap();
                            vm.write_bytes(0x2000, b"\xcf").unwrap();
                            cpu.cpu.gpr[0] = 0x2000;
                            cpu.cpu.gpr[4] = sp;
                            cpu.cpu.gpr[5] = 0x18000;
                            run(&vm, &mut cpu, 1, false)
                        })
                        .collect();
                    assert_eq!(
                        results[0], results[1],
                        "{model:?} code={code:02x?} sp={sp:x}"
                    );
                }
            }
        }
    }
}

#[test]
fn scalar_width_segment_and_branch_siblings_match_checked() {
    let forms: &[&[u8]] = &[
        b"\xa1\xff\xff",
        b"\xa3\xff\xff",
        b"\x83\x06\xff\xff\x01",
        b"\xff\x36\xff\xff",
        b"\xff\x16\xff\xff",
        b"\x66\xa1\xfe\xff",
        b"\x67\xa1\x01\x00\x01\x00",
        b"\x36\xa1\xff\xff",
        b"\x64\xa1\xff\xff",
        b"\x65\xa1\xff\xff",
        b"\x8b\x46\x00",
        b"\x66\x8b\x46\x00",
        b"\x67\x8b\x00",
        b"\xff\xe0",
        b"\xff\xd0",
        b"\x66\xff\xe0",
        b"\x66\xff\xd0",
        b"\x66\x0f\x84\x00\x00\x01\x00",
        b"\x66\xe9\x00\x00\x01\x00",
        b"\xe0\x20",
        b"\xe1\x20",
        b"\xe2\x20",
        b"\xe3\x20",
        b"\x67\xe2\x20",
        b"\x67\xe3\x20",
        b"\x66\x67\xe2\x20",
        b"\x0f\x00\xc0",
        b"\x0f\x00\xd8",
        b"\x0f\x02\xc0",
        b"\x63\xc0",
    ];
    for form in forms {
        for ip in [0x1000, 0xfff0] {
            let results: Vec<_> = [true, false]
                .into_iter()
                .map(|checked| {
                    let (vm, mut cpu) = setup(CpuModel::I80386, checked, ip, form);
                    cpu.cpu.gpr[0] = 0x12000;
                    cpu.cpu.gpr[1] = 0x10002;
                    cpu.cpu.gpr[3] = 0xffff;
                    cpu.cpu.gpr[5] = 0xffff;
                    cpu.cpu.flags.zf = true;
                    run(&vm, &mut cpu, 1, false)
                })
                .collect();
            assert_eq!(results[0], results[1], "code={form:02x?} ip={ip:x}");
        }
    }
}

#[test]
fn r5_cmps_same_device_address_has_two_reads() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let results: Vec<_> = [true, false]
            .into_iter()
            .map(|checked| {
                let (mut vm, mut cpu) = setup(model, checked, 0x1000, b"\xa6");
                vm.unmap(0x3000, 0x1000).unwrap();
                vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                cpu.cpu.gpr[6] = 0x3000;
                cpu.cpu.gpr[7] = 0x3000;
                let first = run(&vm, &mut cpu, 1, true);
                assert_eq!(first.stop, Stop::Read(0x3000, 1));
                cpu.complete_mmio_read(0x11);
                let second = run(&vm, &mut cpu, 1, true);
                cpu.complete_mmio_read(0x22);
                let done = run(&vm, &mut cpu, 1, true);
                assert_eq!((done.steps, done.retired, done.flags & 0x40), (1, 1, 0));
                (first, second, done)
            })
            .collect();
        assert_eq!(results[0], results[1], "{model:?}");
        assert_eq!(results[0].1.stop, Stop::Read(0x3000, 1));
    }
}

#[test]
fn r5_movsw_reads_source_before_destination_limit_fault() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for reply in [0x11, 0xdead_beef_0000_0011] {
            let results: Vec<_> = [true, false]
                .into_iter()
                .map(|checked| {
                    let (mut vm, mut cpu) = setup(model, checked, 0x1000, b"\xa5");
                    vm.unmap(0x3000, 0x1000).unwrap();
                    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                    cpu.cpu.gpr[6] = 0x3000;
                    cpu.cpu.gpr[7] = 0xffff;
                    let first = run(&vm, &mut cpu, 1, true);
                    cpu.complete_mmio_read(reply);
                    let fault = run(&vm, &mut cpu, 1, true);
                    assert_eq!(
                        (fault.steps, fault.retired, &fault.stop),
                        (0, 0, &Stop::Exception(0x1000, 13))
                    );
                    (first, fault)
                })
                .collect();
            assert_eq!(results[0], results[1], "{model:?}");
            assert_eq!(results[0].0.stop, Stop::Read(0x3000, 2));
        }
    }
}

#[test]
fn r5_crossing_store_reports_first_missing_byte() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let results: Vec<_> = [true, false]
            .into_iter()
            .map(|checked| {
                let (mut vm, mut cpu) = setup(model, checked, 0x1000, b"\xa3\xff\x3f");
                vm.unmap(0x4000, 0x1c000).unwrap();
                vm.map(0x5000, 0x1b000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.write_bytes(0x3fff, &[0xab]).unwrap();
                cpu.cpu.gpr[0] = 0x1234;
                let observation = run(&vm, &mut cpu, 1, false);
                let mut byte = [0];
                vm.read_bytes(0x3fff, &mut byte).unwrap();
                assert_eq!(byte, [0xab]);
                observation
            })
            .collect();
        assert_eq!(results[0], results[1], "{model:?}");
        assert_eq!(results[0].stop, Stop::Unmapped(0x4000, AccessKind::Write));
    }
}

#[test]
fn r5_stack_limit_fault_keeps_operand_evidence() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for (opcode, sp, access) in [
            (0x58, 0xffff, FaultAccess::Read),
            (0x50, 1, FaultAccess::Write),
        ] {
            let results: Vec<_> = [true, false]
                .into_iter()
                .map(|checked| {
                    let (vm, mut cpu) = setup(model, checked, 0x1000, &[opcode]);
                    cpu.cpu.gpr[4] = sp;
                    run(&vm, &mut cpu, 1, false)
                })
                .collect();
            assert_eq!(results[0], results[1], "{model:?}, opcode={opcode:x}");
            let offset = 0xffff;
            assert_eq!(
                results[0].evidence,
                Some((
                    Knowledge::Known(offset),
                    Knowledge::Known(2),
                    Knowledge::Known(access)
                ))
            );
            assert_eq!(
                (results[0].steps, results[0].retired, results[0].gpr[4]),
                (0, 0, sp)
            );
        }
    }
}

fn drain(
    vm: &Vm,
    cpu: &mut Vcpu,
    trap: bool,
    mut answer: impl FnMut(u64, u8) -> u64,
) -> Vec<Observation> {
    let entry = (cpu.cpu.cs, cpu.cpu.rip);
    let mut trace = Vec::new();
    for _ in 0..64 {
        let observation = run(vm, cpu, 1, trap);
        let done = match observation.stop {
            Stop::Read(addr, size) => {
                cpu.complete_mmio_read(answer(addr, size));
                false
            }
            Stop::Write(..) => {
                cpu.complete_mmio_write();
                false
            }
            Stop::Budget => (cpu.cpu.cs, cpu.cpu.rip) != entry,
            _ => true,
        };
        trace.push(observation);
        if done {
            return trace;
        }
    }
    panic!("device retry did not converge");
}

#[test]
fn r5_crossing_store_matches_application_address_contract() {
    let mut vm = Vm::new(VmConfig::flat(0x20000));
    vm.set_cpu_mode(x86jit_core::CpuMode::Real16);
    vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, b"\xa3\xff\x3f").unwrap();
    vm.write_bytes(0x3fff, &[0xab]).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[0] = 0x1234;
    let result = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    assert!(matches!(
        result.stop(),
        StepStop::Exit(Exit::UnmappedMemory {
            addr: 0x4000,
            access: AccessKind::Write,
            ..
        })
    ));
    assert_eq!((result.steps(), cpu.retired_instructions()), (0, 0));
    let mut byte = [0];
    vm.read_bytes(0x3fff, &mut byte).unwrap();
    assert_eq!(byte, [0xab]);
}

#[test]
fn r5_string_families_match_over_ram_and_devices() {
    // SDM Vol 2B §4.3 MOVS/CMPS/SCAS/LODS/STOS/REP: distinct operands and
    // iteration-local device obligations, also with 386 operand/address overrides.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for opcode in [0xa4, 0xa5, 0xa6, 0xa7, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf] {
            for prefix in [
                &[][..],
                &[0xf3][..],
                &[0xf2][..],
                &[0x66][..],
                &[0x67][..],
                &[0xf3, 0x66][..],
                &[0xf3, 0x67][..],
                &[0xf3, 0x66, 0x67][..],
            ] {
                if model == CpuModel::I80286 && prefix.iter().any(|p| matches!(p, 0x66 | 0x67)) {
                    continue;
                }
                for (si, di, trap) in [
                    (0x3008, 0x5008, false),
                    (0x3008, 0x5008, true),
                    (0x5008, 0x3008, true),
                    (0x3008, 0x3008, true),
                ] {
                    for df in [false, true] {
                        let mut code = prefix.to_vec();
                        code.push(opcode);
                        let results: Vec<_> = [true, false]
                            .into_iter()
                            .map(|checked| {
                                let (mut vm, mut cpu) = setup(model, checked, 0x1000, &code);
                                vm.write_bytes(0x3000, &[0x11; 32]).unwrap();
                                vm.write_bytes(0x5000, &[0x11; 32]).unwrap();
                                if trap {
                                    vm.unmap(0x3000, 0x1000).unwrap();
                                    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                                }
                                cpu.cpu.gpr[0] = 0x1111_1111;
                                cpu.cpu.gpr[1] = 3;
                                cpu.cpu.gpr[6] = si;
                                cpu.cpu.gpr[7] = di;
                                cpu.cpu.flags.df = df;
                                drain(&vm, &mut cpu, trap, |_, _| 0x1111_1111)
                            })
                            .collect();
                        assert_eq!(
                            results[0], results[1],
                            "{model:?} {code:02x?}, {si:x}/{di:x} trap={trap} df={df}"
                        );
                        assert_eq!(results[0].last().unwrap().stop, Stop::Budget);
                        assert_eq!(results[0].last().unwrap().retired, 1);
                    }
                }
            }
        }
    }
}

#[test]
fn r5_xlat_matches_with_ram_and_device_reads() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for trap in [false, true] {
            for prefix in [&[][..], &[0x66][..], &[0x67][..]] {
                if model == CpuModel::I80286 && !prefix.is_empty() {
                    continue;
                }
                let mut code = prefix.to_vec();
                code.push(0xd7);
                let results: Vec<_> = [true, false]
                    .into_iter()
                    .map(|checked| {
                        let (mut vm, mut cpu) = setup(model, checked, 0x1000, &code);
                        vm.write_bytes(0x3005, &[0xa7]).unwrap();
                        if trap {
                            vm.unmap(0x3000, 0x1000).unwrap();
                            vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                        }
                        cpu.cpu.gpr[3] = 0x3000;
                        cpu.cpu.gpr[0] = 5;
                        let trace = drain(&vm, &mut cpu, trap, |_, _| 0xa7);
                        assert_eq!(cpu.cpu.gpr[0], 0xa7);
                        trace
                    })
                    .collect();
                assert_eq!(results[0], results[1], "{model:?} {code:02x?} trap={trap}");
            }
        }
    }
}

#[test]
fn r5_far_transfers_match_with_pointer_device_reads() {
    // 80386 PRM §17 CALL/JMP/RET: pointer width, cached CS and far frames.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for wide in [false, true] {
            if model == CpuModel::I80286 && wide {
                continue;
            }
            for call in [false, true] {
                for indirect in [false, true] {
                    for trap in [false, true] {
                        if trap && !indirect {
                            continue;
                        }
                        let mut code = if wide { vec![0x66] } else { vec![] };
                        if indirect {
                            code.extend_from_slice(&[
                                0xff,
                                if call { 0x1e } else { 0x2e },
                                0,
                                0x70,
                            ]);
                        } else {
                            code.push(if call { 0x9a } else { 0xea });
                            code.extend_from_slice(if wide {
                                &[0, 0x20, 0, 0, 0, 0x10][..]
                            } else {
                                &[0, 0x20, 0, 0x10][..]
                            });
                        }
                        let results: Vec<_> = [true, false]
                            .into_iter()
                            .map(|checked| {
                                let (mut vm, mut cpu) = setup(model, checked, 0x1000, &code);
                                let ptr = if wide {
                                    &[0, 0x20, 0, 0, 0, 0x10][..]
                                } else {
                                    &[0, 0x20, 0, 0x10][..]
                                };
                                vm.write_bytes(0x7000, ptr).unwrap();
                                vm.write_bytes(
                                    0x12000,
                                    if wide { &[0x66, 0xcb][..] } else { &[0xcb][..] },
                                )
                                .unwrap();
                                if trap {
                                    vm.unmap(0x4000, 0x1c000).unwrap();
                                    for page in (0x4000..0x20000).step_by(0x1000) {
                                        vm.map(
                                            page,
                                            0x1000,
                                            Prot::RWX,
                                            if page == 0x7000 {
                                                RegionKind::Trap
                                            } else {
                                                RegionKind::Ram
                                            },
                                        )
                                        .unwrap();
                                    }
                                }
                                let mut trace = drain(&vm, &mut cpu, false, |address, _| {
                                    if address == 0x7000 {
                                        0x2000
                                    } else {
                                        0x1000
                                    }
                                });
                                assert_eq!((cpu.cpu.cs, cpu.cpu.rip), (0x1000, 0x2000));
                                if call {
                                    trace.push(run(&vm, &mut cpu, 1, false));
                                    assert_eq!(
                                        (cpu.cpu.cs, cpu.cpu.rip),
                                        (0, 0x1000 + code.len() as u64)
                                    );
                                }
                                trace
                            })
                            .collect();
                        assert_eq!(results[0], results[1], "{model:?} {code:02x?} trap={trap}");
                    }
                }
            }
        }
    }
}

#[test]
fn r5_a20_wrap_matches_for_reads_writes_and_same_bus_operands() {
    // Decision-22: the physical bit-20 gate applies separately to split parts.
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for enabled in [false, true] {
            for trap in [false, true] {
                for code in [&b"\xa1\x0f\x00"[..], &b"\xa3\x0f\x00"[..], &b"\xa6"[..]] {
                    let results: Vec<_> = [true, false]
                        .into_iter()
                        .map(|checked| {
                            let (mut vm, mut cpu) = setup(model, checked, 0x5000, code);
                            vm.map(0xff000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
                            vm.map(
                                0x100000,
                                0x1000,
                                Prot::RW,
                                if trap {
                                    RegionKind::Trap
                                } else {
                                    RegionKind::Ram
                                },
                            )
                            .unwrap();
                            vm.write_bytes(0xfffff, &[0xaa]).unwrap();
                            if !trap {
                                vm.write_bytes(0, &[0x11]).unwrap();
                                vm.write_bytes(0x100000, &[0x22]).unwrap();
                            } else {
                                vm.unmap(0, 0x3000).unwrap();
                                vm.map(0, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                                vm.map(0x1000, 0x2000, Prot::RWX, RegionKind::Ram).unwrap();
                            }
                            cpu.cpu.ds = 0xffff;
                            cpu.cpu.legacy.segments[SegmentId::Ds as usize].base = 0xffff0;
                            cpu.cpu.gpr[0] = 0x3344;
                            cpu.cpu.gpr[6] = 0x10;
                            cpu.cpu.gpr[7] = 0;
                            cpu.set_a20_enabled(enabled);
                            drain(&vm, &mut cpu, false, |_, _| 0x11)
                        })
                        .collect();
                    assert_eq!(
                        results[0], results[1],
                        "{model:?} A20={enabled}, trap={trap}, {code:02x?}"
                    );
                }
            }
        }
    }
}

#[test]
fn r5_read_then_write_device_families_keep_distinct_obligations() {
    let forms: &[(&[u8], u64, u64, usize)] = &[
        (b"\xff\x36\x00\x30", 0x3002, 0, 1), // PUSH [3000], same stack destination
        (b"\x8f\x06\x00\x30", 0x3000, 0, 1), // POP [3000], same stack source
        (b"\xff\x06\x00\x30", 0x8800, 0, 1), // INC
        (b"\xd1\x26\x00\x30", 0x8800, 0, 1), // SHL
        (b"\xf7\x16\x00\x30", 0x8800, 0, 1), // NOT
        (b"\xc8\x00\x00\x02", 0x3008, 0x3006, 3), // ENTER with overlapping frame reads/writes
    ];
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for &(code, sp, bp, writes) in forms {
            let results: Vec<_> = [true, false]
                .into_iter()
                .map(|checked| {
                    let (mut vm, mut cpu) = setup(model, checked, 0x1000, code);
                    vm.unmap(0x3000, 0x1000).unwrap();
                    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                    cpu.cpu.gpr[4] = sp;
                    cpu.cpu.gpr[5] = bp;
                    drain(&vm, &mut cpu, true, |_, _| 0x11)
                })
                .collect();
            assert_eq!(results[0], results[1], "{model:?} {code:02x?}");
            assert_eq!(
                results[0]
                    .iter()
                    .filter(|r| matches!(r.stop, Stop::Read(..)))
                    .count(),
                1
            );
            assert_eq!(
                results[0]
                    .iter()
                    .filter(|r| matches!(r.stop, Stop::Write(..)))
                    .count(),
                writes
            );
            assert_eq!(
                (
                    results[0].last().unwrap().steps,
                    results[0].last().unwrap().retired
                ),
                (1, 1)
            );
        }
    }
}

#[test]
fn r5_string_limit_faults_match_with_source_and_destination_devices() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for opcode in [0xa5, 0xa7, 0xab, 0xad, 0xaf] {
            for prefix in [
                &[][..],
                &[0xf3][..],
                &[0xf2][..],
                &[0x66][..],
                &[0xf3, 0x66][..],
                &[0x67][..],
                &[0xf3, 0x67][..],
            ] {
                if model == CpuModel::I80286 && prefix.iter().any(|p| matches!(p, 0x66 | 0x67)) {
                    continue;
                }
                for (si, di) in [(0x3000, 0xffff), (0xffff, 0x3000), (0xffff, 0xffff)] {
                    for trap in [false, true] {
                        let mut code = prefix.to_vec();
                        code.push(opcode);
                        let results: Vec<_> = [true, false]
                            .into_iter()
                            .map(|checked| {
                                let (mut vm, mut cpu) = setup(model, checked, 0x1000, &code);
                                if trap {
                                    vm.unmap(0x3000, 0x1000).unwrap();
                                    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                                }
                                cpu.cpu.gpr[1] = 2;
                                cpu.cpu.gpr[6] = si;
                                cpu.cpu.gpr[7] = di;
                                drain(&vm, &mut cpu, trap, |_, _| 0x11)
                            })
                            .collect();
                        assert_eq!(
                            results[0], results[1],
                            "{model:?} {code:02x?} si={si:x} di={di:x} trap={trap}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn r5_split_store_preflight_reports_first_missing_byte() {
    let results: Vec<_> = [true, false]
        .into_iter()
        .map(|checked| {
            let (mut vm, mut cpu) = setup(CpuModel::I80386, checked, 0x5000, b"\x66\xa3\x0e\x00");
            vm.map(0xff000, 0xfff, Prot::RW, RegionKind::Ram).unwrap();
            vm.map(0x100000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
            vm.write_bytes(0xffffe, &[0xaa]).unwrap();
            cpu.cpu.ds = 0xffff;
            cpu.cpu.legacy.segments[SegmentId::Ds as usize].base = 0xffff0;
            cpu.cpu.gpr[0] = 0x4433_2211;
            cpu.set_a20_enabled(false);
            let observation = run(&vm, &mut cpu, 1, false);
            let mut byte = [0];
            vm.read_bytes(0xffffe, &mut byte).unwrap();
            assert_eq!(byte, [0xaa]);
            observation
        })
        .collect();
    assert_eq!(results[0], results[1]);
    assert_eq!(results[0].stop, Stop::Unmapped(0xfffff, AccessKind::Write));
}

#[test]
fn r5_scalar_device_store_values_match_at_each_width() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for code in [
            &b"\xa2\x00\x30"[..],
            &b"\xa3\x00\x30"[..],
            &b"\x66\xa3\x00\x30"[..],
            &b"\x50"[..],
        ] {
            if model == CpuModel::I80286 && code[0] == 0x66 {
                continue;
            }
            let results: Vec<_> = [true, false]
                .into_iter()
                .map(|checked| {
                    let (mut vm, mut cpu) = setup(model, checked, 0x1000, code);
                    vm.unmap(0x3000, 0x1000).unwrap();
                    vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                    cpu.cpu.gpr[0] = 0x1122_3344_5566_7788;
                    cpu.cpu.gpr[4] = 0x3002;
                    drain(&vm, &mut cpu, true, |_, _| panic!("store requested a read"))
                })
                .collect();
            assert_eq!(results[0], results[1], "{model:?} {code:02x?}");
            let writes: Vec<_> = results[0]
                .iter()
                .filter_map(|r| {
                    if let Stop::Write(_, size, value) = r.stop {
                        Some((size, value))
                    } else {
                        None
                    }
                })
                .collect();
            let size = if code[0] == 0xa2 {
                1
            } else if code[0] == 0x66 {
                4
            } else {
                2
            };
            assert_eq!(
                writes,
                [(size, 0x1122_3344_5566_7788 & (u64::MAX >> (8 * (8 - size))))]
            );
            assert_eq!(results[0].last().unwrap().retired, 1);
        }
    }
}

#[test]
fn r5_unmapped_stack_operands_match_complete_exits() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for (code, sp) in [
            (&b"\x58"[..], 0x3fff),
            (&b"\x50"[..], 0x4001),
            (&b"\xc3"[..], 0x3fff),
            (&b"\xe8\x00\x00"[..], 0x4001),
            (&b"\x9c"[..], 0x4001),
            (&b"\x9d"[..], 0x3fff),
        ] {
            let results: Vec<_> = [true, false]
                .into_iter()
                .map(|checked| {
                    let (mut vm, mut cpu) = setup(model, checked, 0x1000, code);
                    vm.unmap(0x4000, 0x1c000).unwrap();
                    vm.map(0x5000, 0x1b000, Prot::RWX, RegionKind::Ram).unwrap();
                    cpu.cpu.gpr[4] = sp;
                    run(&vm, &mut cpu, 1, false)
                })
                .collect();
            assert_eq!(results[0], results[1], "{model:?} {code:02x?} SP={sp:x}");
            assert_eq!((results[0].steps, results[0].retired), (0, 0));
        }
    }
}

#[test]
fn r5_unmapped_string_operands_match_complete_exits() {
    for model in [CpuModel::I80286, CpuModel::I80386] {
        for opcode in [0xa5, 0xa7, 0xab, 0xad, 0xaf] {
            for (si, di) in [(0x3000, 0x3fff), (0x3fff, 0x3000)] {
                for trap in [false, true] {
                    let results: Vec<_> = [true, false]
                        .into_iter()
                        .map(|checked| {
                            let (mut vm, mut cpu) = setup(model, checked, 0x1000, &[opcode]);
                            vm.unmap(0x4000, 0x1c000).unwrap();
                            vm.map(0x5000, 0x1b000, Prot::RWX, RegionKind::Ram).unwrap();
                            if trap {
                                vm.unmap(0x3000, 0x1000).unwrap();
                                vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
                            }
                            cpu.cpu.gpr[6] = si;
                            cpu.cpu.gpr[7] = di;
                            drain(&vm, &mut cpu, trap, |_, _| 0x11)
                        })
                        .collect();
                    assert_eq!(
                        results[0], results[1],
                        "{model:?} {opcode:x} SI={si:x} DI={di:x} trap={trap}"
                    );
                }
            }
        }
    }
}

fn scalar_limit_write_intent(forms: &[(&[u8], u64, u16)]) {
    for model in [CpuModel::I80386, CpuModel::I80286] {
        for &(code, offset, size) in forms {
            if model == CpuModel::I80286 && code[0] == 0x66 {
                continue;
            }
            // Exercise both the shared walk and the retained A20-wrapping walk.
            for (ds, a20) in [(0, true), (0xffff, false)] {
                let results: Vec<_> = [true, false]
                    .into_iter()
                    .map(|checked| {
                        let (vm, mut cpu) = setup(model, checked, 0x1000, code);
                        cpu.cpu.ds = ds;
                        cpu.cpu.legacy.segments[SegmentId::Ds as usize].base = u32::from(ds) << 4;
                        cpu.set_a20_enabled(a20);
                        cpu.cpu.gpr[0] = 0x1234_5678;
                        let result = run(&vm, &mut cpu, 1, false);
                        assert_eq!(
                            result.evidence,
                            Some((
                                Knowledge::Known(offset),
                                Knowledge::Known(size),
                                Knowledge::Known(FaultAccess::Write)
                            )),
                            "{model:?} {code:02x?} checked={checked} A20={a20}"
                        );
                        assert_eq!(
                            (&result.stop, result.steps, result.retired),
                            (&Stop::Exception(0x1000, 13), 0, 0)
                        );
                        result
                    })
                    .collect();
                assert_eq!(results[0], results[1], "{model:?} {code:02x?} A20={a20}");
            }
        }
    }
}

#[test]
fn r6_scalar_store_limit_keeps_write_intent() {
    scalar_limit_write_intent(&[
        (b"\xa3\xff\xff", 0xffff, 2),
        (b"\x66\xa3\xfe\xff", 0xfffe, 4),
    ]);
}

#[test]
fn r6_scalar_rmw_limit_keeps_write_intent() {
    scalar_limit_write_intent(&[
        (b"\x83\x06\xff\xff\x01", 0xffff, 2),
        (b"\x66\x83\x06\xfe\xff\x01", 0xfffe, 4),
    ]);
}

#[test]
fn unified_far_and_interrupt_frames_match_checked() {
    // SDM Vol 2A §3.3 CALL/IRET and Vol 2B §4.3 RET;
    // 80386 PRM §17.1.3 pp. 17-1–17-2: 66h widens each slot,
    // while SP still wraps according to SS.B.
    for (form, frame) in [
        (&b"\x66\x9a\x00\x20\x00\x00\x00\x10"[..], &[0u32; 3][..]),
        (&b"\x66\xcb"[..], &[0x2000, 0x1000, 2][..]),
        (&b"\x66\xcf"[..], &[0x2000, 0x1000, 0x17002][..]),
    ] {
        for sp in [0, 0xffff, 0x8800] {
            let results: Vec<_> = [true, false]
                .into_iter()
                .map(|checked| {
                    let (vm, mut cpu) = setup(CpuModel::I80386, checked, 0x1000, form);
                    cpu.cpu.gpr[4] = sp;
                    if sp == 0x8800 {
                        for (i, value) in frame.iter().enumerate() {
                            vm.write_bytes(sp + i as u64 * 4, &value.to_le_bytes())
                                .unwrap();
                        }
                    }
                    run(&vm, &mut cpu, 1, false)
                })
                .collect();
            assert_eq!(results[0], results[1], "{form:02x?} SP={sp:x}");
        }
    }
}

#[test]
fn unified_enterd_device_display_keeps_restart_state() {
    // SDM Vol 2A §3.3 ENTER; Vol 3A §§7.5–7.6: a faulting
    // display read precedes architectural frame commit.
    let results: Vec<_> = [true, false]
        .into_iter()
        .map(|checked| {
            let (mut vm, mut cpu) =
                setup(CpuModel::I80386, checked, 0x1000, b"\x66\xc8\x08\x00\x02");
            vm.unmap(0x3000, 0x1000).unwrap();
            vm.map(0x3000, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
            cpu.cpu.gpr[4] = 0x8800;
            cpu.cpu.gpr[5] = 0x3004;
            let before = cpu.cpu.gpr;
            let first = run(&vm, &mut cpu, 1, true);
            assert_eq!((first.steps, first.retired), (0, 0));
            assert_eq!(first.gpr, before);
            assert_eq!(first.stop, Stop::Read(0x3000, 4));
            cpu.complete_mmio_read(0x1234_5678);
            let done = run(&vm, &mut cpu, 1, true);
            assert_eq!((done.steps, done.retired), (1, 1));
            assert_eq!((done.gpr[4], done.gpr[5]), (0x87ec, 0x87fc));
            (first, done)
        })
        .collect();
    assert_eq!(results[0], results[1]);
}

#[test]
fn unified_segment_pop_rechecks_a20_direct_bus_proof() {
    // SDM Vol 2B §4.3 POP loads the selector before the following instruction.
    // Decision-23 §5: that boundary must revalidate the Direct bus view.
    let results: Vec<_> = [true, false]
        .into_iter()
        .map(|checked| {
            let (mut vm, mut cpu) = setup(CpuModel::I80386, checked, 0x1000, b"\x1f\xa0\x10\x00");
            vm.map(0x100000, 0x1000, Prot::RW, RegionKind::Ram).unwrap();
            vm.write_bytes(0x8800, &0xffffu16.to_le_bytes()).unwrap();
            vm.write_bytes(0, &[0x11]).unwrap();
            vm.write_bytes(0x100000, &[0x22]).unwrap();
            cpu.set_a20_enabled(false);
            let done = run(&vm, &mut cpu, 2, false);
            assert_eq!((done.steps, done.retired, done.gpr[0] & 0xff), (2, 2, 0x11));
            done
        })
        .collect();
    assert_eq!(results[0], results[1]);
}
