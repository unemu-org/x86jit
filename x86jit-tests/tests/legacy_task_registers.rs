//! Separate task-register load/store part; no TSS execution or task switching.
#[path = "support/descriptor_fixture.rs"]
mod fixture;
use fixture::*;
use x86jit_core::CpuModel;

#[test]
fn str_stores_visible_selector() {
    // 286 PRM App. B p. B-108; 386 PRM §17 p. 17-166.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x0f\x00\xc8");
        cpu.cpu.legacy.tr.selector = 19;
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.gpr[0], 19);
    }
}

#[test]
fn ltr_str_cache_and_busy_writeback() {
    // 286 PRM App. B pp. B-72/B-108; 386 PRM §17 pp. 17-108/17-166;
    // SDM Vol 2A §3.3 LTR: load available TSS, mark busy, no task switch.
    for model in MODELS {
        for ty in if model == CpuModel::I80286 {
            &[1][..]
        } else {
            &[1, 9][..]
        } {
            let (vm, mut cpu) = setup(model, b"\x0f\x00\xd8\x0f\x00\xc9\x0f\x00\xd8");
            descriptor(&vm, 0x2010, 0x4000, 0x67, 0x80 | ty);
            cpu.cpu.gpr[0] = 19;
            steps(&mut cpu, &vm, 2);
            assert_eq!(cpu.cpu.gpr[1], 19);
            assert_eq!(cpu.cpu.legacy.tr.selector, 19);
            assert_eq!(cpu.cpu.legacy.tr.cache.base, 0x4000);
            assert_eq!(cpu.cpu.legacy.tr.cache.limit, 0x67);
            assert_eq!(value(&vm, 0x2015, 1), u64::from(0x82 | ty));
            fault(&mut cpu, &vm, 13, Some(16));
        }
    }
}

#[test]
fn ltr_rejects_null_ldt_wrong_type_absent_and_truncated() {
    // 286 PRM App. B p. B-72; 386 PRM §17 p. 17-108; SDM Vol 2A §3.3 LTR.
    for model in MODELS {
        for (selector, attr, limit, vector, error) in [
            (0, 0x81, 0x7f, 13, 0),
            (20, 0x81, 0x7f, 13, 20),
            (16, 0x83, 0x7f, 13, 16),
            (16, 0x82, 0x7f, 13, 16),
            (16, 0x01, 0x7f, 11, 16),
            (19, 0x81, 22, 13, 16),
        ] {
            let (vm, mut cpu) = setup(model, b"\x0f\x00\xd8");
            descriptor(&vm, 0x2010, 0x4000, 0x67, attr);
            cpu.cpu.gpr[0] = selector;
            cpu.cpu.legacy.gdtr.limit = limit;
            fault(&mut cpu, &vm, vector, Some(error));
            assert_eq!(value(&vm, 0x2015, 1), u64::from(attr));
        }
    }
}

#[test]
fn ltr_model_types_short_limit_and_cached_tr_survive_table_edits() {
    // SDM Vol 2A §3.3 LTR; 286 PRM App. B p. B-72;
    // 386 PRM §§7.2.2–7.2.3, §17 p. 17-108: only available model TSS
    // types load. LTR does not access the TSS or perform a task switch.
    for model in MODELS {
        for ty in 0..32 {
            let (vm, mut cpu) = setup(model, b"\x0f\x00\xd8");
            descriptor(&vm, 0x2010, 0x4000, 0, 0x80 | ty);
            cpu.cpu.gpr[0] = 16;
            if ty == 1 || (model == CpuModel::I80386 && ty == 9) {
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.legacy.tr.cache.limit, 0);
                let cache = cpu.cpu.legacy.tr;
                descriptor(&vm, 0x2010, 0x5000, 0x67, 0x81);
                vm.write_bytes(0x1003, b"\x0f\x00\xc8").unwrap();
                steps(&mut cpu, &vm, 1);
                assert_eq!(cpu.cpu.legacy.tr, cache);
                assert_eq!(cpu.cpu.gpr[0], 16);
            } else {
                fault(&mut cpu, &vm, 13, Some(16));
            }
        }
    }
}

#[test]
fn a20_gates_tr_descriptor_busy_writeback() {
    // Decision-22; SDM Vol 3A §11.7.13.4/Vol 2A §3.3 LTR.
    for model in MODELS {
        let (vm, mut cpu) = setup(model, b"\x0f\x00\xd8");
        cpu.cpu.legacy.gdtr.base = 0x102000;
        descriptor(&vm, 0x2010, 0x4000, 0x2b, 0x81);
        descriptor(&vm, 0x102010, 0x5000, 0x2b, 0x81);
        cpu.cpu.gpr[0] = 16;
        cpu.set_a20_enabled(false);
        steps(&mut cpu, &vm, 1);
        assert_eq!(cpu.cpu.legacy.tr.cache.base, 0x4000);
        assert_eq!(value(&vm, 0x2015, 1), 0x83);
        assert_eq!(value(&vm, 0x102015, 1), 0x81);
    }
}

#[test]
fn busy_device_update_requires_atomic_emulation_without_installing_tr() {
    // SDM Vol 2A §3.3 LTR p. 3-576: busy update is a locked RMW.
    // Separate MMIO read/write acknowledgments cannot implement that operation.
    for model in MODELS {
        let (mut vm, mut cpu) = setup(model, b"\x0f\x00\xd8");
        cpu.cpu.gpr[0] = 16;
        vm.unmap(0, 0x240000).unwrap();
        vm.map(
            0,
            0x2000,
            x86jit_core::Prot::RWX,
            x86jit_core::RegionKind::Ram,
        )
        .unwrap();
        vm.map(
            0x3000,
            0x23d000,
            x86jit_core::Prot::RWX,
            x86jit_core::RegionKind::Ram,
        )
        .unwrap();
        vm.map(
            0x2000,
            0x1000,
            x86jit_core::Prot::RW,
            x86jit_core::RegionKind::Trap,
        )
        .unwrap();
        let before = cpu.cpu.legacy.tr;
        for (address, response) in [(0x2010, 0x4000_002b), (0x2014, 0x0000_8100)] {
            let run = cpu
                .run_steps(&vm, 1, x86jit_core::StepOptions::default())
                .unwrap();
            assert_eq!(run.steps(), 0);
            assert!(
                matches!(run.stop(), x86jit_core::StepStop::Exit(x86jit_core::Exit::MmioRead { addr, size: 4, .. }) if *addr == address),
                "{:?}",
                run.stop()
            );
            assert_eq!(cpu.cpu.legacy.tr, before);
            cpu.complete_mmio_read(response);
        }
        let run = cpu
            .run_steps(&vm, 1, x86jit_core::StepOptions::default())
            .unwrap();
        assert_eq!(run.steps(), 0);
        assert!(matches!(
            run.stop(),
            x86jit_core::StepStop::Exit(x86jit_core::Exit::UnsupportedSystemContext {
                operation: x86jit_core::UnsupportedSystemOperation::LegacyInstruction,
                ..
            })
        ));
        assert_eq!(cpu.cpu.legacy.tr, before);
    }
}
