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
        .jit(0)
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
        .compile(&ir, MemConsistency::Fast, None, 0, 0)
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
        mxcsr_mode: 0,
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
            .compile(&ir, MemConsistency::Fast, None, 0, 0)
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
        .compile(&ir, MemConsistency::Fast, None, 0, 0)
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
        .compile(&ir, MemConsistency::Fast, None, 0, 0)
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
        .compile(&ir, MemConsistency::Fast, None, 0, 0)
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
        .compile(&ir, MemConsistency::Fast, None, 0, 0)
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
        mxcsr_mode: 0,
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
        .compile(&ir, MemConsistency::Fast, None, 0, 0)
        .unwrap();
    let jit2 = JitBackend::new();
    jit2.enable_environment_diagnostics();
    let second = jit2
        .shared
        .compile(&ir, MemConsistency::Fast, None, 0, 0)
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
        mxcsr_mode: 0,
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
        mxcsr_mode: 0,
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

fn telemetry_ir() -> IrBlock {
    let mem = memory(&[0xf4]);
    lift_block(&mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap()
}

fn assert_lock_sample(sample: &LockTelemetry, count: u64) {
    assert_eq!(sample.acquisitions, count);
    // No lower bound on the hold time: a coarse monotonic clock (41.7 ns per
    // tick at 24 MHz) can measure a short uncontended hold as 0 ns.
    assert!(sample.total_hold_ns >= sample.max_hold_ns);
    assert!(sample.total_wait_ns >= sample.max_wait_ns);
    assert_eq!(sample.hold_histogram.iter().sum::<u64>(), count);
    assert_eq!(sample.wait_histogram.iter().sum::<u64>(), count);
}

fn wait_for_telemetry_waiters(jit: &JitBackend, expected: &[(LockKind, usize)]) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while !expected
        .iter()
        .all(|&(kind, count)| jit.shared.lock_telemetry.waiting(kind) == count)
    {
        assert!(
            std::time::Instant::now() < deadline,
            "timed lock waiters did not start"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
fn lock_telemetry_switch_modes_and_reset() {
    let jit = JitBackend::new();
    let ir = telemetry_ir();
    for mode in [0, 1] {
        jit.shared
            .compile(&ir, MemConsistency::Fast, None, 0, mode)
            .unwrap();
    }
    jit.invalidate_links();
    assert_eq!(jit.lock_telemetry(), LockTelemetrySnapshot::default());
    jit.set_lock_telemetry(true);
    for mode in [0, 1, 0] {
        jit.shared
            .compile(&ir, MemConsistency::Fast, None, 0, mode)
            .unwrap();
    }
    let snapshot = jit.lock_telemetry();
    assert_lock_sample(&snapshot.compile_serial, 3);
    assert_lock_sample(&snapshot.default_module, 2);
    assert_lock_sample(&snapshot.nondefault_module, 1);
    jit.set_lock_telemetry(false);
    jit.shared
        .compile(&ir, MemConsistency::Fast, None, 0, 0)
        .unwrap();
    jit.invalidate_links();
    assert_eq!(jit.lock_telemetry(), snapshot);
    jit.reset_lock_telemetry();
    assert_eq!(jit.lock_telemetry(), LockTelemetrySnapshot::default());
    jit.set_lock_telemetry(true);
    jit.shared
        .compile(&ir, MemConsistency::Fast, None, 0, 1)
        .unwrap();
    assert_lock_sample(&jit.lock_telemetry().nondefault_module, 1);
    assert_eq!(jit.lock_telemetry().default_module.acquisitions, 0);
}

#[test]
fn lock_telemetry_invalidation_does_not_build_modules() {
    let jit = JitBackend::new();
    jit.set_lock_telemetry(true);
    jit.invalidate_links();
    let snapshot = jit.lock_telemetry();
    assert_lock_sample(&snapshot.default_module, 1);
    assert_lock_sample(&snapshot.nondefault_module, 1);
    assert!(jit.shared.inner.try_lock().unwrap().is_none());
    assert!(jit.shared.nondefault_inner.try_lock().unwrap().is_none());
}

#[test]
fn lock_telemetry_compile_errors_release_guards() {
    for stage in 1..=4 {
        let jit = JitBackend::new();
        jit.set_lock_telemetry(true);
        jit.shared
            .fail_compile_stage
            .store(stage, Ordering::Relaxed);
        assert!(jit
            .shared
            .compile(&telemetry_ir(), MemConsistency::Fast, None, 0, 0)
            .is_err());
        assert_lock_sample(&jit.lock_telemetry().default_module, 1);
        assert_lock_sample(&jit.lock_telemetry().compile_serial, 1);
        assert!(jit.shared.compile_serial.try_lock().is_ok());
        assert!(jit.shared.inner.try_lock().is_ok());
        let snapshot = jit.lock_telemetry();
        assert!(jit
            .shared
            .compile(&telemetry_ir(), MemConsistency::Fast, None, 0, 0)
            .is_err());
        assert_eq!(jit.lock_telemetry(), snapshot);
    }
    let jit = JitBackend::new();
    jit.set_lock_telemetry(true);
    jit.set_compile_limit(0);
    assert!(jit
        .shared
        .compile(&telemetry_ir(), MemConsistency::Fast, None, 0, 0)
        .is_err());
    assert_lock_sample(&jit.lock_telemetry().default_module, 1);
    assert_lock_sample(&jit.lock_telemetry().compile_serial, 1);
}

#[test]
fn lock_telemetry_compile_unwind_and_poison_recovery() {
    let jit = JitBackend::new();
    jit.set_lock_telemetry(true);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = jit
            .shared
            .compile_with(CompiledCodeKind::Block, 0x1000, 0, |_, _, _| {
                panic!("compile telemetry unwind probe");
            });
    }));
    assert!(result.is_err());
    assert!(jit.shared.inner.is_poisoned());
    assert!(jit.shared.compile_serial.is_poisoned());
    assert_lock_sample(&jit.lock_telemetry().default_module, 1);
    assert_lock_sample(&jit.lock_telemetry().compile_serial, 1);
    // Reacquire the poisoned compilation locks through their timed paths.
    // The injected panic may also poison unrelated compiler state.
    let serial = jit.shared.lock_telemetry.lock(
        &jit.shared.compile_serial,
        LockKind::CompileSerial,
        "JIT compilation",
    );
    drop(jit.shared.jit(0));
    drop(serial);
    assert_lock_sample(&jit.lock_telemetry().default_module, 2);
    assert_lock_sample(&jit.lock_telemetry().compile_serial, 2);
}

#[test]
fn lock_telemetry_observer_runs_after_release() {
    let jit = Arc::new(JitBackend::new());
    jit.set_lock_telemetry(true);
    let weak = Arc::downgrade(&jit);
    let seen = Arc::new(AtomicUsize::new(0));
    let output = seen.clone();
    jit.set_compiled_code_observer(move |_| {
        let jit = weak.upgrade().unwrap();
        if jit.shared.inner.try_lock().is_ok()
            && jit.shared.nondefault_inner.try_lock().is_ok()
            && jit.shared.compile_serial.try_lock().is_ok()
            && jit.lock_telemetry().default_module.acquisitions == 1
            && jit.lock_telemetry().compile_serial.acquisitions == 1
        {
            jit.reset_lock_telemetry();
            output.fetch_add(1, Ordering::Relaxed);
        }
    });
    jit.shared
        .compile(&telemetry_ir(), MemConsistency::Fast, None, 0, 0)
        .unwrap();
    assert_eq!(seen.load(Ordering::Relaxed), 1);
    assert_eq!(jit.lock_telemetry(), LockTelemetrySnapshot::default());
}

#[test]
fn lock_telemetry_compilers_wait_for_module() {
    use std::sync::mpsc::channel;
    use std::time::Duration;
    let jit = Arc::new(JitBackend::new());
    jit.set_lock_telemetry(true);
    // Keep module ownership deterministic while both compiling threads start.
    let held = jit.shared.jit(0);
    let (started, ready) = channel();
    let (finished, done) = channel();
    let mut workers = Vec::new();
    for _ in 0..2 {
        let jit = jit.clone();
        let started = started.clone();
        let finished = finished.clone();
        workers.push(std::thread::spawn(move || {
            let ir = telemetry_ir();
            started.send(()).unwrap();
            jit.shared
                .compile(&ir, MemConsistency::Fast, None, 0, 0)
                .unwrap();
            finished.send(()).unwrap();
        }));
    }
    for _ in 0..2 {
        ready.recv_timeout(Duration::from_secs(3)).unwrap();
    }
    // One compiler waits on the held module, keeping the other on serial.
    wait_for_telemetry_waiters(
        &jit,
        &[(LockKind::DefaultModule, 1), (LockKind::CompileSerial, 1)],
    );
    assert!(done.recv_timeout(Duration::from_millis(30)).is_err());
    drop(held);
    for _ in 0..2 {
        done.recv_timeout(Duration::from_secs(3)).unwrap();
    }
    for worker in workers {
        worker.join().unwrap();
    }
    let snapshot = jit.lock_telemetry();
    assert_lock_sample(&snapshot.default_module, 3);
    assert!(snapshot.default_module.max_wait_ns >= 10_000_000);
    assert_lock_sample(&snapshot.compile_serial, 2);
    assert!(snapshot.compile_serial.max_wait_ns >= 10_000_000);
}

#[test]
fn lock_telemetry_switch_and_reset_with_active_guards() {
    let jit = JitBackend::new();
    jit.set_lock_telemetry(true);
    let module = jit.shared.lock_module(0);
    jit.reset_lock_telemetry();
    assert_eq!(jit.lock_telemetry(), LockTelemetrySnapshot::default());
    jit.set_lock_telemetry(false);
    drop(module);
    assert_lock_sample(&jit.lock_telemetry().default_module, 1);
    let module = jit.shared.lock_module(1);
    jit.set_lock_telemetry(true);
    drop(module);
    assert_eq!(jit.lock_telemetry().nondefault_module.acquisitions, 0);
    jit.reset_lock_telemetry();
    drop(jit.shared.lock_module(1));
    assert_lock_sample(&jit.lock_telemetry().nondefault_module, 1);
}

#[test]
fn lock_telemetry_compilers_wait_for_serial_separately() {
    use std::sync::mpsc::channel;
    use std::time::Duration;

    let jit = Arc::new(JitBackend::new());
    jit.set_lock_telemetry(true);
    let (holding, held) = channel();
    let (release, released) = channel();
    let (compiler_finished, compiler_done) = channel();
    let first = jit.clone();
    let compiler = std::thread::spawn(move || {
        first
            .shared
            .compile_with(
                CompiledCodeKind::Block,
                0x1000,
                0,
                |builder, helpers, slots| {
                    holding.send(()).unwrap();
                    released.recv_timeout(Duration::from_secs(3)).unwrap();
                    codegen::translate_block(
                        builder,
                        &telemetry_ir(),
                        &first.shared.offsets,
                        slots,
                        helpers,
                        MemConsistency::Fast,
                        None,
                        0,
                        false,
                    )
                },
            )
            .unwrap();
        compiler_finished.send(()).unwrap();
    });
    held.recv_timeout(Duration::from_secs(3)).unwrap();
    assert!(jit.shared.compile_serial.try_lock().is_err());
    // The other mode's module is free while the first compiler holds serial.
    assert!(jit.shared.nondefault_inner.try_lock().is_ok());
    let second = jit.clone();
    let (starting, started) = channel();
    let (finished, done) = channel();
    let waiter = std::thread::spawn(move || {
        let ir = telemetry_ir();
        starting.send(()).unwrap();
        second
            .shared
            .compile(&ir, MemConsistency::Fast, None, 0, 1)
            .unwrap();
        finished.send(()).unwrap();
    });
    started.recv_timeout(Duration::from_secs(3)).unwrap();
    wait_for_telemetry_waiters(&jit, &[(LockKind::CompileSerial, 1)]);
    assert_eq!(
        jit.shared
            .lock_telemetry
            .waiting(LockKind::NondefaultModule),
        0
    );
    assert_eq!(jit.lock_telemetry().nondefault_module.acquisitions, 0);
    let blocked = done.recv_timeout(Duration::from_millis(30)).is_err();
    release.send(()).unwrap();
    compiler_done.recv_timeout(Duration::from_secs(3)).unwrap();
    done.recv_timeout(Duration::from_secs(3)).unwrap();
    compiler.join().unwrap();
    waiter.join().unwrap();
    assert!(blocked);
    let snapshot = jit.lock_telemetry();
    assert_lock_sample(&snapshot.compile_serial, 2);
    assert_lock_sample(&snapshot.default_module, 1);
    assert_lock_sample(&snapshot.nondefault_module, 1);
    assert!(snapshot.compile_serial.max_wait_ns >= 10_000_000);
    assert!(snapshot.compile_serial.max_hold_ns >= 10_000_000);
}
