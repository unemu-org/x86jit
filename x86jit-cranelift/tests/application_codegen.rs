//! Deterministic application block bytes, captured at main 73ed5d6 with
//! rustc 1.98.1, Cranelift 0.135.8, native AArch64 Darwin, default backend options.
//! Address-bearing snapshots expand as crate unit tests with fixed helper, counter
//! and link-slot targets before lowering. Every emitted byte is compared; these
//! synthetic targets are never executed. No production API is added.
//! No x86_64 host is available for native capture here. Linux x86_64 snapshots must
//! be captured from 73ed5d6 by that CI host; cross-compilation is not execution.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
use std::sync::{Arc, Mutex};
use x86jit_core::{Backend, CachedBlock, CpuMode, MemConsistency, Prot, RegionKind};
use x86jit_cranelift::JitBackend;

#[test]
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn application_codegen_bytes() {
    use x86jit_core::{
        lift::{lift_block, FetchAddr},
        Memory, MemoryModel,
    };
    for (index, mode, code) in [
        (0, CpuMode::Long64, &b"\xb8\x78\x56\x34\x12\xf4"[..]),
        (1, CpuMode::Compat32, &b"\xb8\x78\x56\x34\x12\xf4"[..]),
        (2, CpuMode::Long64, &b"\x48\x89\xd8\x48\x01\xc8\xf4"[..]),
        (3, CpuMode::Compat32, &b"\x89\xd8\x01\xc8\xf4"[..]),
    ] {
        let mut mem = Memory::new(MemoryModel::Flat { size: 0x4000 });
        mem.map(0x1000, 4096, Prot::RWX, RegionKind::Ram).unwrap();
        for (i, byte) in code.iter().enumerate() {
            mem.write(0x1000 + i as u64, u64::from(*byte), 1).unwrap();
        }
        let ir = lift_block(&mem, FetchAddr::flat(0x1000), mode).unwrap();
        let backend = JitBackend::new();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let capture = observed.clone();
        backend.set_compiled_code_observer(move |event| {
            // SAFETY: the observer runs while the backend owns this published range.
            let bytes = unsafe {
                std::slice::from_raw_parts(event.host_start as *const u8, event.host_len as usize)
            };
            *capture.lock().unwrap() = bytes.to_vec();
        });
        assert!(matches!(
            backend.materialize(&ir, MemConsistency::Fast, None, 0),
            CachedBlock::Compiled { .. }
        ));
        let bytes = observed.lock().unwrap();
        let expected: &[&[u8]] = &[
            &[
                127, 35, 3, 213, 253, 123, 191, 169, 253, 3, 0, 145, 6, 207, 138, 82, 134, 70, 162,
                114, 198, 124, 64, 146, 6, 0, 0, 249, 199, 0, 130, 210, 7, 64, 0, 249, 64, 0, 128,
                210, 253, 123, 193, 168, 255, 15, 95, 214,
            ],
            &[
                127, 35, 3, 213, 253, 123, 191, 169, 253, 3, 0, 145, 6, 207, 138, 82, 134, 70, 162,
                114, 198, 124, 64, 146, 6, 0, 0, 249, 199, 0, 130, 210, 7, 64, 0, 249, 64, 0, 128,
                210, 253, 123, 193, 168, 255, 15, 95, 214,
            ],
            &[
                127, 35, 3, 213, 253, 123, 191, 169, 253, 3, 0, 145, 2, 12, 64, 249, 2, 0, 0, 249,
                3, 4, 64, 249, 68, 0, 3, 139, 159, 0, 2, 235, 229, 39, 159, 154, 129, 0, 0, 145,
                63, 0, 4, 235, 228, 39, 159, 154, 166, 0, 4, 42, 63, 0, 0, 241, 228, 23, 159, 154,
                37, 252, 127, 211, 71, 0, 3, 202, 72, 0, 1, 202, 7, 1, 39, 138, 231, 252, 127, 211,
                66, 0, 3, 202, 66, 0, 1, 202, 6, 96, 2, 57, 1, 100, 2, 57, 2, 104, 2, 57, 4, 108,
                2, 57, 5, 112, 2, 57, 7, 116, 2, 57, 1, 0, 0, 249, 225, 0, 130, 210, 1, 64, 0, 249,
                64, 0, 128, 210, 253, 123, 193, 168, 255, 15, 95, 214,
            ],
            &[
                127, 35, 3, 213, 253, 123, 191, 169, 253, 3, 0, 145, 1, 12, 64, 249, 33, 124, 64,
                146, 1, 0, 0, 249, 3, 4, 64, 249, 226, 3, 1, 42, 227, 3, 3, 42, 97, 64, 33, 139,
                33, 0, 0, 145, 36, 252, 96, 211, 132, 0, 64, 146, 225, 3, 1, 42, 63, 0, 0, 241,
                229, 23, 159, 154, 38, 252, 95, 211, 198, 0, 64, 146, 71, 0, 3, 202, 72, 0, 1, 202,
                7, 1, 39, 138, 231, 252, 95, 211, 231, 0, 64, 146, 66, 0, 3, 202, 66, 0, 1, 202, 4,
                96, 2, 57, 1, 100, 2, 57, 2, 104, 2, 57, 5, 108, 2, 57, 6, 112, 2, 57, 7, 116, 2,
                57, 33, 124, 64, 146, 1, 0, 0, 249, 161, 0, 130, 210, 1, 64, 0, 249, 64, 0, 128,
                210, 253, 123, 193, 168, 255, 15, 95, 214,
            ],
        ];
        assert_eq!(&**bytes, expected[index]);
    }
}

#[path = "support/application_fixtures.rs"]
pub(crate) mod fixtures;

#[test]
fn real16_lifted_ir_matches_main() {
    for (name, ir) in fixtures::real16() {
        assert!(ir.legacy_context.is_none());
        let observed = format!("{ir:#?}\n").replace("    legacy_context: None,\n", "");
        let path = format!(
            "{}/tests/snapshots/real16/{name}.txt",
            env!("CARGO_MANIFEST_DIR")
        );
        assert_eq!(observed, std::fs::read_to_string(path).unwrap(), "{name}");
    }
}

// The unit-test expansion fixes relocation targets before instruction selection.
// Real target addresses can change constant lowering and register allocation.
#[allow(unused_macros)]
macro_rules! relocated_codegen_witnesses {
    () => {
        #[test]
        fn application_relocated_codegen_bytes() {
            use std::sync::{Arc, Mutex};
            use x86jit_core::{CpuMode, MemConsistency};
            for mode in [CpuMode::Long64, CpuMode::Compat32] {
                let mut units = crate::application_codegen::fixtures::blocks(mode)
                    .into_iter()
                    .map(|(name, ir, mxcsr)| (name, Some(ir), None, mxcsr))
                    .collect::<Vec<_>>();
                units.push((
                    "region",
                    None,
                    Some(crate::application_codegen::fixtures::region(mode)),
                    0,
                ));
                if mode == CpuMode::Long64 {
                    units.push((
                        "rip-relative",
                        Some(crate::application_codegen::fixtures::block(
                            b"\x48\x8b\x05\x20\x00\x00\x00\x48\x89\x05\x20\x00\x00\x00\xf4",
                            mode,
                        )),
                        None,
                        0,
                    ));
                }
                for (name, ir, region, mxcsr) in units {
                    let backend = crate::JitBackend::new();
                    let observed = Arc::new(Mutex::new(Vec::new()));
                    let capture = observed.clone();
                    backend.set_compiled_code_observer(move |event| {
                        // SAFETY: the observer sees the backend's live executable range.
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                event.host_start as *const u8,
                                event.host_len as usize,
                            )
                        };
                        *capture.lock().unwrap() = bytes.to_vec();
                    });
                    let mut target_manifest = String::from(
                        "note_write_helper=0000123456781000\n\
                         counter:note_write_helper=0000123456782000\n\
                         string_helper=0000123456783000\n\
                         counter:string_helper=0000123456784000\n\
                         fp_boundary_helper=0000123456785000\n\
                         counter:fp_boundary_helper=0000123456786000\n\
                         set_control=0000123456787000\n",
                    );
                    let kind = if ir.is_some() {
                        crate::CompiledCodeKind::Block
                    } else {
                        crate::CompiledCodeKind::Region
                    };
                    let mut slot_index = 0u64;
                    backend
                        .shared
                        .compile_with(kind, 0x1000, mxcsr, |builder, mut helpers, _| {
                            helpers.note_watch.1 = 0x1234_5678_1000;
                            helpers.note_watch.2 = 0x1234_5678_2000;
                            helpers.string.1 = 0x1234_5678_3000;
                            helpers.string.2 = 0x1234_5678_4000;
                            helpers.fp_boundary.1 = 0x1234_5678_5000;
                            helpers.fp_boundary.2 = 0x1234_5678_6000;
                            helpers.switch_fp.1 = 0x1234_5678_7000;
                            helpers.switch_fp_changed = 0x1234_5678_7000;
                            let mut alloc_slot = || {
                                let address = 0x1234_5679_1000 + slot_index * 8;
                                target_manifest
                                    .push_str(&format!("link_slot:{slot_index}={address:016x}\n"));
                                slot_index += 1;
                                address
                            };
                            if let Some(ir) = ir {
                                crate::codegen::translate_block(
                                    builder,
                                    &ir,
                                    &backend.shared.offsets,
                                    &mut alloc_slot,
                                    helpers,
                                    MemConsistency::Fast,
                                    None,
                                    0,
                                    false,
                                )
                            } else {
                                crate::codegen::translate_region(
                                    builder,
                                    region.as_ref().unwrap(),
                                    &backend.shared.offsets,
                                    &mut alloc_slot,
                                    helpers,
                                    MemConsistency::Fast,
                                    None,
                                    0,
                                    false,
                                )
                            }
                        })
                        .unwrap();
                    let bytes = observed.lock().unwrap();
                    let hex = bytes
                        .chunks(16)
                        .map(|line| line.iter().map(|b| format!("{b:02x}")).collect::<String>())
                        .collect::<Vec<_>>()
                        .join("\n");
                    let captured = format!("{hex}\n\n{target_manifest}");
                    let path = format!(
                        "{}/tests/snapshots/application-aarch64-macos/{mode:?}-{name}.txt",
                        env!("CARGO_MANIFEST_DIR")
                    );
                    assert_eq!(
                        captured,
                        std::fs::read_to_string(path).unwrap(),
                        "{mode:?}-{name}"
                    );
                }
            }
        }
    };
}
#[allow(unused_imports)]
pub(crate) use relocated_codegen_witnesses;

#[test]
fn application_codegen_covers_blocks_regions_and_linked_execution() {
    use x86jit_core::{Exit, RegionCaps, Vm, VmConfig};
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for (_, ir, mxcsr) in fixtures::blocks(mode) {
            assert!(matches!(
                JitBackend::new().materialize_with_mxcsr(&ir, MemConsistency::Fast, None, 0, mxcsr),
                CachedBlock::Compiled { .. }
            ));
        }
        assert!(matches!(
            JitBackend::new().materialize_region(
                &fixtures::region(mode),
                MemConsistency::Fast,
                None,
                0
            ),
            CachedBlock::Compiled { .. }
        ));
        // Loop between two blocks so the dispatcher fills a real link slot.
        let mut vm = Vm::with_backend(VmConfig::flat(0x4000), Box::new(JitBackend::new()));
        vm.set_cpu_mode(mode);
        vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\xeb\x00\xeb\xfc").unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.rip = 0x1000;
        assert!(matches!(cpu.run(&vm, Some(20)), Exit::BudgetExhausted));
        assert!(vm.cache.chained() > 0);
        assert_eq!(cpu.cpu.rip, 0x1000);
        // A real region must execute its internal branches as well.
        let mut vm = Vm::with_backend(
            VmConfig::flat(0x4000),
            Box::new(JitBackend::with_superblocks(RegionCaps {
                max_blocks: 4,
                max_icount: 32,
            })),
        );
        vm.set_cpu_mode(mode);
        vm.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(0x1000, b"\x85\xc0\x74\x03\x83\xc0\x01\xf4")
            .unwrap();
        vm.test_support_force_region(
            0x1000,
            RegionCaps {
                max_blocks: 4,
                max_icount: 32,
            },
            0,
        )
        .unwrap();
        for eax in [0, 1] {
            let mut cpu = vm.new_vcpu();
            cpu.cpu.rip = 0x1000;
            cpu.cpu.gpr[0] = eax;
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.cpu.gpr[0], if eax == 0 { 0 } else { 2 });
        }
        assert!(vm.cache.regions() > 0);
    }
}
