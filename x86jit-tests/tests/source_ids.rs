use std::sync::{Arc, Mutex};
use x86jit_core::{
    Backend, BlockKey, CachedBlock, CpuSource, CpuidLeaf, CpuidOutOfRange, CpuidOverride,
    CpuidOverrides, CpuidRegisters as R, CpuidTable, Exit, InterpreterBackend, MemConsistency,
    Prot, RandomKind, Reg, RegionCaps, RegionKind, SourceIdError, TscSample, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const ENTRY: u64 = 0x1000;
const CAPS: RegionCaps = RegionCaps {
    max_blocks: 4,
    max_icount: 32,
};

#[derive(Debug, Default)]
struct Source {
    requests: Mutex<Vec<u64>>,
    aux: u32,
}

impl CpuSource for Source {
    fn read_tsc(&self, id: u64) -> TscSample {
        self.requests.lock().unwrap().push(id);
        TscSample {
            counter: 0x1234_5678_9abc_def0,
            aux: self.aux,
        }
    }

    fn read_random(&self, _: u64, _: RandomKind, _: u8) -> Option<u64> {
        unreachable!()
    }
}

fn identity_on_tier(tier: usize) {
    for (source_id, cpuid_id, aux, expected_ebx) in [
        (42, 42, 42, 0x2a12_3456),
        (0x1234_5678_0000_002a, 0x1234_56a7, 0x89ab_cdef, 0xa712_3456),
    ] {
        for consistency in [
            MemConsistency::Fast,
            MemConsistency::AcqRel,
            MemConsistency::FullTso,
        ] {
            let backend: Box<dyn Backend> = match tier {
                0 => Box::new(InterpreterBackend),
                1 => Box::new(JitBackend::new()),
                2 => Box::new(JitBackend::with_superblocks(CAPS)),
                _ => unreachable!(),
            };
            let mut config = VmConfig::flat(0x2000);
            config.consistency = consistency;
            let mut vm = Vm::with_backend(config, backend);
            vm.map(ENTRY, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
            vm.write_bytes(ENTRY, &[0x0f, 0xa2, 0xf4]).unwrap(); // cpuid; hlt
            vm.write_bytes(ENTRY + 0x10, &[0x0f, 0x01, 0xf9, 0xf4])
                .unwrap(); // rdtscp; hlt

            let mut table = CpuidTable::new(CpuidOutOfRange::Fixed(R::ZERO));
            table
                .leaves
                .insert(0, CpuidLeaf::Constant(R::new(0x1f, 0, 0, 0)));
            table
                .leaves
                .insert(1, CpuidLeaf::Constant(R::new(0, 0x123456, 0, 0)));
            for leaf in [0xb, 0x1f] {
                table.leaves.insert(
                    leaf,
                    CpuidLeaf::Indexed {
                        subleaves: [(0, R::new(1, 2, 0x100, 0)), (1, R::new(2, 4, 0x201, 0))]
                            .into(),
                        invalid_subleaf: R::ZERO,
                    },
                );
            }
            vm.set_cpuid_table(Some(table));
            let source = Arc::new(Source {
                requests: Mutex::default(),
                aux,
            });
            vm.set_cpu_source(Some(source.clone()));
            // The 64-bit source key, CPUID topology ID and TSC_AUX are independent.
            let parent = Vm::new(VmConfig::flat(0x1000));
            parent.new_vcpu_with_source_id(source_id).unwrap();
            let saved_cursor = parent.next_source_id().unwrap();
            vm.set_next_source_id(saved_cursor).unwrap();
            let mut cpu = vm.new_vcpu_with_source_id(source_id).unwrap();
            assert_eq!(vm.new_vcpu().cpu.source_id, saved_cursor);
            let mut overrides = CpuidOverrides::default();
            // SDM Vol 1 §21.3, Tables 21-8/31/70: initial APIC ID in
            // 01H EBX[31:24], x2APIC ID in 0BH/1FH EDX[31:0] on each subleaf.
            overrides.leaves.insert(
                (1, 0),
                CpuidOverride {
                    mask: R::new(0, 0xff00_0000, 0, 0),
                    value: R::new(0, (cpuid_id & 0xff) << 24, 0, 0),
                },
            );
            for leaf in [0xb, 0x1f] {
                for subleaf in [0, 1] {
                    overrides.leaves.insert(
                        (leaf, subleaf),
                        CpuidOverride {
                            mask: R::new(0, 0, 0, u32::MAX),
                            value: R::new(0, 0, 0, cpuid_id),
                        },
                    );
                }
            }
            cpu.cpu.cpuid_overrides = overrides;
            if tier == 2 {
                vm.test_support_force_region(ENTRY, CAPS, cpu.cpu.mxcsr)
                    .unwrap();
            }
            for (leaf, subleaf, reg, expected) in [
                (1, 0, Reg::Rbx, expected_ebx),
                (0xb, 0, Reg::Rdx, u64::from(cpuid_id)),
                (0xb, 1, Reg::Rdx, u64::from(cpuid_id)),
                (0x1f, 0, Reg::Rdx, u64::from(cpuid_id)),
                (0x1f, 1, Reg::Rdx, u64::from(cpuid_id)),
            ] {
                cpu.set_reg(Reg::Rip, ENTRY);
                cpu.set_reg(Reg::Rax, leaf);
                cpu.set_reg(Reg::Rcx, subleaf);
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                assert_eq!(cpu.reg(reg), expected);
            }
            let cached = vm
                .cache
                .get(BlockKey::with_mxcsr(ENTRY, vm.cpu_mode(), cpu.cpu.mxcsr))
                .unwrap();
            assert_eq!(matches!(cached, CachedBlock::Compiled { .. }), tier != 0);
            if tier == 2 {
                vm.test_support_force_region(ENTRY + 0x10, CAPS, cpu.cpu.mxcsr)
                    .unwrap();
            }
            cpu.set_reg(Reg::Rip, ENTRY + 0x10);
            for reg in [Reg::Rax, Reg::Rcx, Reg::Rdx] {
                cpu.set_reg(reg, u64::MAX);
            }
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            // SDM Vol 3B §20.17.2 and Vol 2B §4.3 RDTSCP: EDX:EAX is the
            // counter; ECX is the software-defined AUX signature, zero-extended.
            assert_eq!(cpu.reg(Reg::Rax), 0x9abc_def0);
            assert_eq!(cpu.reg(Reg::Rdx), 0x1234_5678);
            assert_eq!(cpu.reg(Reg::Rcx), u64::from(aux));
            assert_eq!(*source.requests.lock().unwrap(), [source_id]);
            let cached = vm
                .cache
                .get(BlockKey::with_mxcsr(
                    ENTRY + 0x10,
                    vm.cpu_mode(),
                    cpu.cpu.mxcsr,
                ))
                .unwrap();
            assert_eq!(matches!(cached, CachedBlock::Compiled { .. }), tier != 0);
            if tier == 2 {
                assert!(vm.cache.regions() >= 2);
            }
            // Without a source, AUX is engine policy and stays zero for any ID.
            cpu.cpu.cpu_source = None;
            cpu.set_reg(Reg::Rip, ENTRY + 0x10);
            cpu.set_reg(Reg::Rcx, u64::MAX);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rcx), 0);
            assert_eq!(*source.requests.lock().unwrap(), [source_id]);
        }
    }
}

#[test]
fn source_ids_identity_interp() {
    identity_on_tier(0);
}

#[test]
fn source_ids_identity_jit_block() {
    identity_on_tier(1);
}

#[test]
fn source_ids_identity_jit_region() {
    identity_on_tier(2);
}

#[test]
fn source_ids_explicit_reservations_skip_and_reject_duplicates() {
    let vm = Vm::new(VmConfig::flat(0x1000));
    assert_eq!(vm.next_source_id(), Some(0));
    assert_eq!(vm.new_vcpu().cpu.source_id, 0);
    let one = vm.new_vcpu_with_source_id(1).unwrap();
    assert_eq!(one.cpu.source_id, 1);
    assert_eq!(vm.new_vcpu().cpu.source_id, 2);
    assert_eq!(vm.next_source_id(), Some(3));
    assert_eq!(
        vm.new_vcpu_with_cpuid_overrides(CpuidOverrides::default())
            .cpu
            .source_id,
        3
    );
    assert!(matches!(
        vm.new_vcpu_with_source_id(1),
        Err(SourceIdError::InUse(1))
    ));
    drop(one);
    assert!(matches!(
        vm.new_vcpu_with_source_id(1),
        Err(SourceIdError::InUse(1))
    ));
    assert!(matches!(
        vm.new_vcpu_with_source_id(0),
        Err(SourceIdError::InUse(0))
    ));
    assert_eq!(vm.new_vcpu_with_mxcsr(0x1f80).cpu.source_id, 4);
    vm.new_vcpu_with_source_id(1000).unwrap();
    assert_eq!(vm.new_vcpu().cpu.source_id, 1001);
    assert_eq!(vm.new_vcpu_with_source_id(5).unwrap().cpu.source_id, 5);
    assert_eq!(vm.next_source_id(), Some(1002));
}

#[test]
fn source_ids_seed_is_forward_only_and_errors_leave_it_unchanged() {
    let vm = Vm::new(VmConfig::flat(0x1000));
    vm.set_next_source_id(100).unwrap();
    assert_eq!(vm.new_vcpu_with_source_id(7).unwrap().cpu.source_id, 7);
    assert_eq!(vm.next_source_id(), Some(100));
    assert_eq!(
        vm.set_next_source_id(99),
        Err(SourceIdError::Rewind {
            current: 100,
            requested: 99
        })
    );
    vm.set_next_source_id(100).unwrap();
    assert_eq!(vm.new_vcpu().cpu.source_id, 100);
    vm.new_vcpu_with_source_id(102).unwrap();
    assert_eq!(
        vm.set_next_source_id(102),
        Err(SourceIdError::Rewind {
            current: 103,
            requested: 102
        })
    );
    assert_eq!(vm.new_vcpu().cpu.source_id, 103);
}

#[test]
fn source_ids_rebuild_preserves_survivor_and_parent_cursor() {
    let parent = Vm::new(VmConfig::flat(0x1000));
    let cpus = (0..5).map(|_| parent.new_vcpu()).collect::<Vec<_>>();
    let cursor = parent.next_source_id().unwrap();
    let survivor_id = cpus[2].cpu.source_id;
    let child = Vm::new(VmConfig::flat(0x1000));
    child.set_next_source_id(cursor).unwrap();
    let survivor = child.new_vcpu_with_source_id(survivor_id).unwrap();
    assert_eq!(survivor.cpu.source_id, 2);
    assert_eq!(child.new_vcpu().cpu.source_id, 5);
    assert_eq!(parent.new_vcpu().cpu.source_id, 5);
}

#[test]
fn source_ids_fork_shares_reservations_and_seed() {
    let parent = Vm::new(VmConfig::flat(0x1000));
    parent.new_vcpu_with_source_id(3).unwrap();
    let child = parent
        .fork_with_backend(Box::new(InterpreterBackend))
        .unwrap();
    assert!(matches!(
        child.new_vcpu_with_source_id(3),
        Err(SourceIdError::InUse(3))
    ));
    child.set_next_source_id(6).unwrap();
    assert_eq!(parent.new_vcpu().cpu.source_id, 6);
    assert_eq!(child.new_vcpu().cpu.source_id, 7);
}

#[test]
fn source_ids_full_width_and_exhaustion_never_wrap() {
    let vm = Vm::new(VmConfig::flat(0x1000));
    vm.new_vcpu_with_source_id(u64::MAX).unwrap();
    assert_eq!(vm.next_source_id(), None);
    assert_eq!(vm.set_next_source_id(0), Err(SourceIdError::Exhausted));
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| vm.new_vcpu())).is_err());
    assert!(matches!(
        vm.new_vcpu_with_source_id(u64::MAX),
        Err(SourceIdError::InUse(u64::MAX))
    ));
    // Exhaustion of the monotonic automatic cursor leaves unused explicit IDs valid.
    assert_eq!(vm.new_vcpu_with_source_id(12).unwrap().cpu.source_id, 12);
    let automatic_max = Vm::new(VmConfig::flat(0x1000));
    automatic_max.set_next_source_id(u64::MAX).unwrap();
    assert_eq!(automatic_max.new_vcpu().cpu.source_id, u64::MAX);
    assert_eq!(automatic_max.next_source_id(), None);
}

#[test]
fn source_ids_concurrent_allocation_has_one_explicit_winner() {
    let vm = Arc::new(Vm::new(VmConfig::flat(0x1000)));
    let threads = (0..8)
        .map(|_| {
            let vm = vm.clone();
            std::thread::spawn(move || {
                let explicit = vm.new_vcpu_with_source_id(100).is_ok();
                let automatic = vm.new_vcpu().cpu.source_id;
                (explicit, automatic)
            })
        })
        .collect::<Vec<_>>();
    let results = threads
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(results.iter().filter(|(won, _)| *won).count(), 1);
    let mut ids = results.into_iter().map(|(_, id)| id).collect::<Vec<_>>();
    ids.sort_unstable();
    assert_eq!(ids, (101..109).collect::<Vec<_>>());
}
