use super::*;
use cranelift::codegen::isa;
use x86jit_core::lift::{lift_block, lift_region, CpuMode, FetchAddr};
use x86jit_core::{Memory, MemoryModel, Prot, RegionKind};

fn target(name: &str) -> cranelift::codegen::isa::OwnedTargetIsa {
    let mut flags = settings::builder();
    flags.set("use_colocated_libcalls", "false").unwrap();
    flags.set("is_pic", "false").unwrap();
    flags.set("preserve_frame_pointers", "true").unwrap();
    flags.set("enable_probestack", "false").unwrap();
    let mut isa = isa::lookup(name.parse().unwrap()).unwrap();
    if name.starts_with("x86_64") {
        isa.set("has_sse41", "false").unwrap();
        isa.set("has_ssse3", "false").unwrap();
    }
    isa.finish(settings::Flags::new(flags)).unwrap()
}

fn memory(bytes: &[u8]) -> Memory {
    let mut mem = Memory::new(MemoryModel::Flat { size: 0x4000 });
    mem.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
    mem.write_bytes(0x1000, bytes).unwrap();
    mem
}

#[test]
fn embedder_libcall_lowerings_finalize_on_both_isas() {
    for host in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        let mut module = JITModule::new(libcalls::builder(target(host)));
        let mut forms = Vec::new();
        // ROUND{PS,PD,SS,SD}, every immediate rounding mode, reg and memory.
        for opcode in 8..=11 {
            for mode in 0..4 {
                for operand in [0xc1, 0x01] {
                    forms.push(vec![0x66, 0x0f, 0x3a, opcode, operand, mode]);
                }
            }
        }
        forms.extend([
            vec![0x66, 0x0f, 0x38, 0x00, 0xc1], // pshufb
            vec![0x66, 0x0f, 0x38, 0x00, 0x01],
            vec![0x66, 0x0f, 0x5b, 0xc1], // cvtps2dq
            vec![0x66, 0x0f, 0x5b, 0x01],
            vec![0xf2, 0x0f, 0xe6, 0xc1], // cvtpd2dq
            vec![0xf2, 0x0f, 0xe6, 0x01],
            vec![0xc5, 0xfd, 0xe6, 0xc1], // vcvtpd2dq ymm
            vec![0xc5, 0xfd, 0xe6, 0x01],
            vec![0xc5, 0xfd, 0x5b, 0xc1], // vcvtps2dq ymm
            vec![0xc5, 0xfd, 0x5b, 0x01],
        ]);
        for prefix in [0xf2, 0xf3] {
            for operand in [0xc1, 0x01] {
                forms.push(vec![prefix, 0x0f, 0x2d, operand]); // cvt{s,d}2si
                forms.push(vec![prefix, 0x48, 0x0f, 0x2d, operand]);
            }
        }
        let mut seen = std::collections::HashSet::new();
        for (i, bytes) in forms.iter_mut().enumerate() {
            bytes.push(0xf4);
            let mem = memory(bytes);
            let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
            assert_eq!(ir.guest_len as usize, bytes.len(), "{bytes:02x?}");
            let mut ctx = module.make_context();
            ctx.func.signature = module.make_signature();
            ctx.func.signature.params = vec![AbiParam::new(types::I64); 2];
            ctx.func.signature.returns.push(AbiParam::new(types::I64));
            let mut fbctx = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
            let mut sig = module.make_signature();
            sig.params = vec![AbiParam::new(types::I64); 6];
            sig.returns.push(AbiParam::new(types::I64));
            let dummy = builder.import_signature(sig);
            let helpers = codegen::Helpers::all_dummy(&mut || (dummy, 0, 0), (dummy, 0, 0));
            codegen::translate_block(
                &mut builder,
                &ir,
                &cpu_offsets(),
                &mut || 0,
                helpers,
                MemConsistency::Fast,
                None,
                0,
                false,
            );
            builder.finalize();
            let id = module
                .declare_function(&format!("form_{i}"), Linkage::Export, &ctx.func.signature)
                .unwrap();
            module.define_function(id, &mut ctx).unwrap();
            for reloc in ctx.compiled_code().unwrap().buffer.relocs() {
                if let cranelift::codegen::FinalizedRelocTarget::ExternalName(
                    ir::ExternalName::LibCall(call),
                ) = &reloc.target
                {
                    seen.insert(*call);
                }
            }
            module.finalize_definitions().unwrap();
        }
        if host.starts_with("x86_64") {
            for call in [
                ir::LibCall::FloorF32,
                ir::LibCall::FloorF64,
                ir::LibCall::CeilF32,
                ir::LibCall::CeilF64,
                ir::LibCall::TruncF32,
                ir::LibCall::TruncF64,
                ir::LibCall::NearestF32,
                ir::LibCall::NearestF64,
                ir::LibCall::X86Pshufb,
            ] {
                assert!(seen.contains(&call), "lowering never exercised {call}");
            }
        }
    }
}

#[test]
fn embedder_observer_sees_blocks_and_background_regions() {
    let mem = memory(&[0x48, 0xff, 0xc8, 0x75, 0xfb, 0xf4]); // dec rax; jnz entry; hlt
    let jit = JitBackend::new();
    assert!(jit
        .codegen_description()
        .contains("preserve_frame_pointers=true"));
    assert!(jit
        .shared
        .jit()
        .module
        .isa()
        .flags()
        .preserve_frame_pointers());
    let observed = Arc::new(Mutex::new(Vec::new()));
    let output = observed.clone();
    jit.set_compiled_code_observer(move |code| {
        output
            .lock()
            .unwrap()
            .push((code, std::thread::current().id()));
    });
    let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let entry = jit
        .shared
        .compile(&ir, MemConsistency::Fast, None, 0)
        .unwrap();
    let region = lift_region(
        &mem,
        0x1000,
        RegionCaps {
            max_blocks: 8,
            max_icount: 128,
        },
        CpuMode::Long64,
    )
    .unwrap();
    let handle = jit.tier_up_handle();
    let request = TierUpRequest {
        pc: 0x1000,
        spans: vec![(0x1000, 6)],
        epoch: 0,
        unit: TierUpUnit::Region(Arc::new(region)),
        consistency: MemConsistency::Fast,
        mmio: None,
        guest_base: 0,
    };
    assert!(matches!(jit.tier_up_async(request), TierUpSubmit::Queued));
    handle.wait_idle();
    let observed = observed.lock().unwrap();
    assert_eq!(observed.len(), 2);
    assert_eq!(observed[0].0.host_start, entry.0 as usize);
    assert_eq!(observed[0].0.kind, CompiledCodeKind::Block);
    assert_eq!(observed[0].1, std::thread::current().id());
    assert_eq!(observed[1].0.kind, CompiledCodeKind::Region);
    assert_ne!(observed[1].1, std::thread::current().id());
    for (code, _) in observed.iter() {
        assert_eq!(code.guest_rip, 0x1000);
        assert!(code.host_len > 0);
        assert!((0..code.host_len).any(|offset| {
            x86jit_core::codemap::lookup(code.host_start + offset as usize) == Some(0x1000)
        }));
    }
}

#[test]
fn embedder_flag_elision_options_preserve_execution() {
    let mem = memory(&[0x83, 0xc0, 1, 0x83, 0xc0, 2, 0xf4]);
    let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let mut expected = x86jit_core::CpuState::new();
    expected.rip = 0x1000;
    let expected_exit = x86jit_core::interp::interpret_block(
        &ir,
        &mut expected,
        &mem,
        &mut Vec::new(),
        &mut Default::default(),
    );
    assert!(matches!(
        expected_exit,
        x86jit_core::StepResult::Exit(x86jit_core::Exit::Hlt)
    ));
    let lengths = Arc::new(Mutex::new(Vec::new()));
    for enabled in [false, true] {
        let jit = JitBackend::new();
        jit.set_flag_elision(enabled);
        let lengths = lengths.clone();
        jit.set_compiled_code_observer(move |code| lengths.lock().unwrap().push(code.host_len));
        let entry = jit
            .shared
            .compile(&ir, MemConsistency::Fast, None, 0)
            .unwrap();
        let mut actual = x86jit_core::CpuState::new();
        actual.rip = 0x1000;
        // SAFETY: entry uses the compiled-block ABI and both state and memory
        // remain live throughout execution.
        let exit = unsafe {
            x86jit_core::jit_abi::run_compiled(entry, &mut actual, &mem, CpuMode::Long64)
        };
        assert!(matches!(
            exit,
            x86jit_core::StepResult::Exit(x86jit_core::Exit::Hlt)
        ));
        assert_eq!(actual.gpr, expected.gpr);
        assert_eq!(actual.flags, expected.flags);
        assert!(jit
            .codegen_description()
            .contains(&format!("flag_elision={enabled}")));
    }
    let lengths = lengths.lock().unwrap();
    assert!(
        lengths[0] > lengths[1],
        "disabling elision must retain flag code"
    );
}

#[test]
#[cfg(feature = "environment")]
fn embedder_legacy_jit_environment_requires_api_opt_in() {
    const CHILD: &str = "X86JIT_JIT_OPTIONS_CHILD";
    if std::env::var_os(CHILD).is_none() {
        assert!(std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "embedder_tests::embedder_legacy_jit_environment_requires_api_opt_in"
            ])
            .env(CHILD, "1")
            .env("X86JIT_PERF_MAP", "1")
            .env("X86JIT_NO_FLAG_ELISION", "1")
            .status()
            .unwrap()
            .success());
        return;
    }
    let path = std::path::PathBuf::from(format!("/tmp/perf-{}.map", std::process::id()));
    assert!(!path.exists());
    let jit = JitBackend::new();
    let mem = memory(&[0xf4]);
    let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    jit.shared
        .compile(&ir, MemConsistency::Fast, None, 0)
        .unwrap();
    assert!(
        !path.exists(),
        "default construction and compilation must ignore env"
    );
    assert!(jit.codegen_description().contains("flag_elision=true"));
    jit.enable_environment_diagnostics();
    assert!(jit.codegen_description().contains("flag_elision=false"));
    let entry = jit
        .shared
        .compile(&ir, MemConsistency::Fast, None, 0)
        .unwrap();
    let map = std::fs::read_to_string(&path).unwrap();
    assert!(map.starts_with(&format!("{:x} ", entry.0 as usize)));
    assert!(map.ends_with(" jit_0x1000\n"));
    drop(jit);
    std::fs::remove_file(path).unwrap();
}

#[test]
#[cfg(target_arch = "x86_64")]
fn embedder_baseline_swizzle_libcall_obeys_vector_abi() {
    // SDM Vol 2B, PSHUFB, Operation (128-bit SSE), p. 4-423:
    // the low nibble selects a byte, and bit 7 selects zero.
    let mem = memory(&[0x66, 0x0f, 0x38, 0x00, 0xc1, 0xf4]);
    let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let jit = JitBackend::new();
    let mut state = JitBackend::build_jit(HostTarget::Baseline, OptLevel::None);
    state.module = JITModule::new(libcalls::builder(target("x86_64-unknown-linux-gnu")));
    *jit.shared.inner.lock().unwrap() = Some(state);
    let entry = jit
        .shared
        .compile(&ir, MemConsistency::Fast, None, 0)
        .unwrap();
    for base in (0..=255).step_by(16) {
        let mut cpu = x86jit_core::CpuState::new();
        cpu.rip = 0x1000;
        cpu.xmm[0] = u128::from_le_bytes(std::array::from_fn(|i| i as u8 + 1));
        cpu.xmm[1] = u128::from_le_bytes(std::array::from_fn(|i| (base + i) as u8));
        let expected = u128::from_le_bytes(std::array::from_fn(|i| {
            let index = (base + i) as u8;
            if index & 0x80 != 0 {
                0
            } else {
                (index & 15) + 1
            }
        }));
        // SAFETY: native-ISA entry with live CpuState and Memory, compiled-block ABI.
        unsafe { x86jit_core::jit_abi::run_compiled(entry, &mut cpu, &mem, CpuMode::Long64) };
        assert_eq!(cpu.xmm[0], expected);
    }
}

#[test]
fn review_observer_is_outside_module_lock() {
    let jit = Arc::new(JitBackend::new());
    let backend = Arc::downgrade(&jit);
    let observed = Arc::new(AtomicUsize::new(0));
    let output = observed.clone();
    let mem = memory(&[0xf4]);
    let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let nested = ir.clone();
    jit.set_compiled_code_observer(move |code| {
        let jit = backend.upgrade().unwrap();
        // Record the lock probe outside any assertion that panic handling could
        // swallow. Bound re-entry to one nested materialization.
        if jit.shared.inner.try_lock().is_err() {
            return;
        }
        assert!((0..code.host_len).any(|off| {
            x86jit_core::codemap::lookup(code.host_start + off as usize) == Some(code.guest_rip)
        }));
        if output.fetch_add(1, Ordering::Relaxed) == 0 {
            jit.invalidate_links();
            assert!(matches!(
                jit.materialize(&nested, MemConsistency::Fast, None, 0),
                CachedBlock::Compiled { .. }
            ));
        }
    });
    jit.shared
        .compile(&ir, MemConsistency::Fast, None, 0)
        .unwrap();
    assert_eq!(
        observed.load(Ordering::Relaxed),
        2,
        "observer must support re-entry after registration"
    );
}

#[test]
fn review_observer_panic_does_not_strand_background_queue() {
    let jit = JitBackend::new();
    let called = Arc::new(AtomicUsize::new(0));
    let output = called.clone();
    jit.set_compiled_code_observer(move |_| {
        output.fetch_add(1, Ordering::Relaxed);
        panic!("observer panic probe");
    });
    let mem = memory(&[0xf4]);
    let ir = Arc::new(lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap());
    let request = || TierUpRequest {
        pc: 0x1000,
        spans: vec![(0x1000, 1)],
        epoch: 0,
        unit: TierUpUnit::Block(ir.clone()),
        consistency: MemConsistency::Fast,
        mmio: None,
        guest_base: 0,
    };
    let handle = jit.tier_up_handle();
    for _ in 0..2 {
        assert!(matches!(jit.tier_up_async(request()), TierUpSubmit::Queued));
        let deadline = Instant::now() + std::time::Duration::from_secs(3);
        while jit.shared.queue.lock().unwrap().outstanding != 0 {
            assert!(
                !jit.worker.lock().unwrap().as_ref().unwrap().is_finished(),
                "observer panic terminated compiler"
            );
            assert!(
                Instant::now() < deadline,
                "observer panic stalled background compile"
            );
            std::thread::yield_now();
        }
        handle.wait_idle();
    }
    assert!(!jit.worker.lock().unwrap().as_ref().unwrap().is_finished());
    assert_eq!(jit.tier_up_finished().len(), 2);
    assert_eq!(
        called.load(Ordering::Relaxed),
        1,
        "panicking observer must be disabled"
    );
}

#[test]
fn review_foreground_observer_panic_keeps_replacement() {
    let jit = Arc::new(JitBackend::new());
    let backend = Arc::downgrade(&jit);
    let observed = Arc::new(AtomicUsize::new(0));
    let output = observed.clone();
    jit.set_compiled_code_observer(move |_| {
        let output = output.clone();
        backend
            .upgrade()
            .unwrap()
            .set_compiled_code_observer(move |_| {
                output.fetch_add(1, Ordering::Relaxed);
            });
        panic!("replaced observer panic probe");
    });
    let mem = memory(&[0xf4]);
    let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    for _ in 0..2 {
        assert!(matches!(
            jit.materialize(&ir, MemConsistency::Fast, None, 0),
            CachedBlock::Compiled { .. }
        ));
    }
    assert_eq!(observed.load(Ordering::Relaxed), 1);
}

#[test]
#[cfg(feature = "environment")]
fn review_legacy_perfmap_multiple_backends_preserve_ranges() {
    const CHILD: &str = "X86JIT_REVIEW_PERFMAP_CHILD";
    if std::env::var_os(CHILD).is_none() {
        assert!(std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "embedder_tests::review_legacy_perfmap_multiple_backends_preserve_ranges"
            ])
            .env(CHILD, "1")
            .env("X86JIT_PERF_MAP", "1")
            .status()
            .unwrap()
            .success());
        return;
    }
    let path = std::path::PathBuf::from(format!("/tmp/perf-{}.map", std::process::id()));
    assert!(!path.exists());
    std::fs::write(&path, "1 1 retained\n").unwrap();
    let mem = memory(&[0xf4]);
    let ir = lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let jit1 = JitBackend::new();
    jit1.enable_environment_diagnostics();
    let first = jit1
        .shared
        .compile(&ir, MemConsistency::Fast, None, 0)
        .unwrap();
    let jit2 = JitBackend::new();
    jit2.enable_environment_diagnostics();
    let second = jit2
        .shared
        .compile(&ir, MemConsistency::Fast, None, 0)
        .unwrap();
    jit1.enable_environment_diagnostics();
    let map = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(
        map.lines().count(),
        3,
        "second backend truncated first backend's live code range"
    );
    assert!(map.starts_with("1 1 retained\n"));
    assert!(map
        .lines()
        .any(|line| line.starts_with(&format!("{:x} ", first.0 as usize))));
    assert!(map
        .lines()
        .any(|line| line.starts_with(&format!("{:x} ", second.0 as usize))));
}

#[test]
fn review_panic_payload_drop_keeps_worker_alive() {
    struct Payload;
    impl Drop for Payload {
        fn drop(&mut self) {
            panic!("panic payload destructor");
        }
    }
    let jit = JitBackend::new();
    let called = Arc::new(AtomicUsize::new(0));
    let output = called.clone();
    jit.set_compiled_code_observer(move |_| {
        output.fetch_add(1, Ordering::Relaxed);
        std::panic::panic_any(Payload);
    });
    let mem = memory(&[0xf4]);
    let ir = Arc::new(lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap());
    let request = || TierUpRequest {
        pc: 0x1000,
        spans: vec![(0x1000, 1)],
        epoch: 0,
        unit: TierUpUnit::Block(ir.clone()),
        consistency: MemConsistency::Fast,
        mmio: None,
        guest_base: 0,
    };
    let handle = jit.tier_up_handle();
    for _ in 0..2 {
        assert!(matches!(jit.tier_up_async(request()), TierUpSubmit::Queued));
        let deadline = Instant::now() + std::time::Duration::from_secs(3);
        while jit.shared.queue.lock().unwrap().outstanding != 0 {
            assert!(
                !jit.worker.lock().unwrap().as_ref().unwrap().is_finished(),
                "observer panic terminated compiler; outstanding={}",
                jit.shared.queue.lock().unwrap().outstanding
            );
            assert!(
                Instant::now() < deadline,
                "observer panic stalled background compile"
            );
            std::thread::yield_now();
        }
        handle.wait_idle();
    }
    assert!(!jit.worker.lock().unwrap().as_ref().unwrap().is_finished());
    assert_eq!(jit.tier_up_finished().len(), 2);
    assert_eq!(
        called.load(Ordering::Relaxed),
        1,
        "panicking observer must be disabled"
    );
}

#[test]
fn review_observer_capture_drop_keeps_worker_alive() {
    struct Payload;
    impl Drop for Payload {
        fn drop(&mut self) {
            panic!("observer capture destructor");
        }
    }
    let jit = JitBackend::new();
    let called = Arc::new(AtomicUsize::new(0));
    let output = called.clone();
    let capture = Payload;
    jit.set_compiled_code_observer(move |_| {
        let _ = &capture;
        output.fetch_add(1, Ordering::Relaxed);
        panic!("observer body panic");
    });
    let mem = memory(&[0xf4]);
    let ir = Arc::new(lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap());
    let request = || TierUpRequest {
        pc: 0x1000,
        spans: vec![(0x1000, 1)],
        epoch: 0,
        unit: TierUpUnit::Block(ir.clone()),
        consistency: MemConsistency::Fast,
        mmio: None,
        guest_base: 0,
    };
    let handle = jit.tier_up_handle();
    for _ in 0..2 {
        assert!(matches!(jit.tier_up_async(request()), TierUpSubmit::Queued));
        let deadline = Instant::now() + std::time::Duration::from_secs(3);
        while jit.shared.queue.lock().unwrap().outstanding != 0 {
            assert!(
                !jit.worker.lock().unwrap().as_ref().unwrap().is_finished(),
                "observer panic terminated compiler; outstanding={}",
                jit.shared.queue.lock().unwrap().outstanding
            );
            assert!(
                Instant::now() < deadline,
                "observer panic stalled background compile"
            );
            std::thread::yield_now();
        }
        handle.wait_idle();
    }
    assert!(!jit.worker.lock().unwrap().as_ref().unwrap().is_finished());
    assert_eq!(jit.tier_up_finished().len(), 2);
    assert_eq!(
        called.load(Ordering::Relaxed),
        1,
        "panicking observer must be disabled"
    );
}

#[test]
fn review_observer_retirement_drops_outside_backend_locks() {
    struct Payload;
    impl Drop for Payload {
        fn drop(&mut self) {
            panic!("secondary panic payload destructor");
        }
    }
    struct Capture {
        shared: Weak<Shared>,
        dropped: Arc<AtomicUsize>,
    }
    impl Drop for Capture {
        fn drop(&mut self) {
            if let Some(shared) = self.shared.upgrade() {
                assert!(shared.inner.try_lock().is_ok());
                assert!(shared.observer.try_lock().is_ok());
            }
            self.dropped.fetch_add(1, Ordering::Relaxed);
            std::panic::panic_any(Payload);
        }
    }
    let jit = JitBackend::new();
    let dropped = Arc::new(AtomicUsize::new(0));
    for _ in 0..2 {
        let capture = Capture {
            shared: Arc::downgrade(&jit.shared),
            dropped: dropped.clone(),
        };
        jit.set_compiled_code_observer(move |_| {
            let _ = &capture;
        });
    }
    let replaced = dropped.load(Ordering::Relaxed);
    drop(jit);
    assert_eq!(replaced, 1);
    assert_eq!(dropped.load(Ordering::Relaxed), 2);
}

#[test]
fn review_compile_completion_notifies_during_unwind() {
    let jit = JitBackend::new();
    jit.shared.queue.lock().unwrap().outstanding = 1;
    let shared = jit.shared.clone();
    let (started, ready) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        let queue = shared.queue.lock().unwrap();
        started.send(()).unwrap();
        let (queue, timeout) = shared
            .idle_cv
            .wait_timeout_while(queue, std::time::Duration::from_secs(3), |q| {
                q.outstanding != 0
            })
            .unwrap();
        assert!(!timeout.timed_out(), "completion must notify idle waiters");
        assert_eq!(queue.outstanding, 0);
    });
    ready.recv().unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _completion = CompileCompletion(&jit.shared);
        panic!("compile unwind probe");
    }));
    assert!(result.is_err());
    waiter.join().unwrap();
    jit.tier_up_handle().wait_idle();
}
