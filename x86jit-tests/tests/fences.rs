//! SDM Vol 3A §11.2.3.4 permits the SB (0,0) outcome without a fence;
//! Vol 2B §4.3 MFENCE forbids it with a full fence.
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use iced_x86::code_asm::*;
use x86jit_core::{
    Backend, Exit, InterpreterBackend, MemConsistency, Prot, Reg, RegionCaps, RegionKind, Vm,
    VmConfig,
};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;
const X: u64 = 0x8000;
const Y: u64 = 0x8100;
const ITERS: usize = 100_000;

fn program(regions: bool) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    a.set_label(&mut top).unwrap();
    if regions {
        let mut body = a.create_label();
        a.jmp(body).unwrap();
        a.set_label(&mut body).unwrap();
    }
    a.mov(qword_ptr(rbx), 1).unwrap();
    a.mfence().unwrap();
    a.mov(rax, qword_ptr(rdx)).unwrap();
    if regions {
        a.dec(rcx).unwrap();
        a.jnz(top).unwrap();
    }
    a.hlt().unwrap();
    a.assemble(CODE).unwrap()
}

#[test]
fn fences_lift_to_explicit_ir() {
    use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
    // SDM Vol 2B §4.3 MFENCE, Vol 2A §3.3 LFENCE, Vol 2B §4.3 SFENCE.
    use x86jit_core::{FenceKind, IrOp};
    for (modrm, kind) in [
        (0xf0, FenceKind::Full),
        (0xe8, FenceKind::Load),
        (0xf8, FenceKind::Store),
    ] {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(CODE, &[0x0f, 0xae, modrm, 0xf4]).unwrap();
        let ir = lift_block(&vm.mem, FetchAddr::flat(CODE), CpuMode::Long64).unwrap();
        let fences: Vec<_> = ir
            .ops
            .iter()
            .filter_map(|op| match op {
                IrOp::Fence { kind } => Some(*kind),
                _ => None,
            })
            .collect();
        assert_eq!(fences, [kind]);
    }
}

fn helper_mp_program(regions: bool, writer: bool) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    a.set_label(&mut top).unwrap();
    if regions {
        let mut body = a.create_label();
        a.jmp(body).unwrap();
        a.set_label(&mut body).unwrap();
    }
    if writer {
        a.rep().stosb().unwrap();
        a.mov(qword_ptr(Y), 1).unwrap();
    } else {
        a.mov(rax, qword_ptr(Y)).unwrap();
        a.mfence().unwrap();
        a.mov(rcx, qword_ptr(X)).unwrap();
    }
    if regions {
        a.dec(r8).unwrap();
        a.jnz(top).unwrap();
    }
    a.hlt().unwrap();
    a.assemble(CODE + if writer { 0 } else { 0x100 }).unwrap()
}

#[test]
fn helper_stos_message_passing_jit_block_and_region() {
    // SDM Vol 3A §11.2.4.1 item 3: string operations are not reordered
    // with other stores. The reader MFENCE isolates the writer's ordering.
    let mut failures = Vec::new();
    for mode in ["block", "regions"] {
        for tier in [
            MemConsistency::Fast,
            MemConsistency::AcqRel,
            MemConsistency::FullTso,
        ] {
            let count = race(tier, mode, true);
            if tier != MemConsistency::Fast && count != 0 {
                failures.push(format!("{tier:?}/{mode}: {count}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "helper MP allowed Y=1,X=0: {failures:?}"
    );
}

#[derive(Default)]
struct Rendezvous {
    epoch: AtomicUsize,
    done: [AtomicUsize; 2],
    result: [AtomicU64; 2],
    stop: AtomicBool,
}

fn wait_yield(spins: &mut u32) {
    *spins += 1;
    if *spins == 64 {
        thread::yield_now();
        *spins = 0;
    } else {
        std::hint::spin_loop();
    }
}

fn race(tier: MemConsistency, mode: &str, helper_mp: bool) -> usize {
    let iters = if helper_mp { 200_000 } else { ITERS };
    let regions = mode == "regions";
    let backend: Box<dyn Backend> = match mode {
        "interp" => Box::new(InterpreterBackend),
        "block" => Box::new(JitBackend::new()),
        "regions" => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 4,
            max_icount: 32,
        })),
        _ => unreachable!(),
    };
    let mut config = VmConfig::flat(0x10000);
    config.consistency = tier;
    let mut vm = Vm::with_backend(config, backend);
    vm.map(0, 0x10000, Prot::RW, RegionKind::Ram).unwrap();
    for id in 0..2 {
        let code = if helper_mp {
            helper_mp_program(regions, id == 0)
        } else {
            program(regions)
        };
        vm.write_bytes(CODE + id * 0x100, &code).unwrap();
    }
    // Materialize before the race: neither compilation nor a first code-page
    // registration should add incidental synchronization to the measured pair.
    for id in 0..2 {
        let mut warm = vm.new_vcpu();
        warm.set_reg(Reg::Rip, CODE + id * 0x100);
        warm.set_reg(Reg::Rcx, 1);
        warm.set_reg(Reg::Rax, 1);
        warm.set_reg(Reg::Rdi, X);
        warm.set_reg(Reg::Rbx, X);
        warm.set_reg(Reg::Rdx, Y);
        warm.set_reg(Reg::R8, 1);
        assert!(matches!(warm.run(&vm, Some(8)), Exit::Hlt));
    }
    if regions {
        assert!(vm.cache.regions() > 0, "region path must actually execute");
    }
    let vm = Arc::new(vm);
    let sync = Arc::new(Rendezvous::default());
    let deadline = Instant::now() + Duration::from_secs(30);
    let handles: Vec<_> = (0..2)
        .map(|id| {
            let vm = vm.clone();
            let sync = sync.clone();
            thread::spawn(move || {
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rbx, if id == 0 { X } else { Y });
                cpu.set_reg(Reg::Rdx, if id == 0 { Y } else { X });
                for epoch in 1..=iters {
                    let mut spins = 0;
                    while sync.epoch.load(Ordering::Acquire) != epoch {
                        if sync.stop.load(Ordering::Relaxed) || Instant::now() >= deadline {
                            sync.stop.store(true, Ordering::Relaxed);
                            return;
                        }
                        wait_yield(&mut spins);
                    }
                    cpu.set_reg(Reg::Rip, CODE + id as u64 * 0x100);
                    cpu.set_reg(Reg::Rcx, 1);
                    cpu.set_reg(Reg::Rax, 1);
                    cpu.set_reg(Reg::Rdi, X);
                    cpu.set_reg(Reg::R8, 1);
                    assert!(matches!(cpu.run(&vm, Some(8)), Exit::Hlt));
                    let result = if helper_mp {
                        u64::from(cpu.reg(Reg::Rax) == 1 && cpu.reg(Reg::Rcx) == 0)
                    } else {
                        cpu.reg(Reg::Rax)
                    };
                    sync.result[id].store(result, Ordering::Relaxed);
                    sync.done[id].store(epoch, Ordering::Release);
                }
            })
        })
        .collect();
    let mut relaxed = 0;
    let mut timed_out = false;
    for epoch in 1..=iters {
        vm.mem.write(X, 0, 8).unwrap();
        vm.mem.write(Y, 0, 8).unwrap();
        sync.epoch.store(epoch, Ordering::Release);
        let mut spins = 0;
        while sync.done.iter().any(|d| d.load(Ordering::Acquire) != epoch) {
            if Instant::now() >= deadline {
                timed_out = true;
                break;
            }
            wait_yield(&mut spins);
        }
        if timed_out {
            break;
        }
        relaxed += usize::from(if helper_mp {
            sync.result[1].load(Ordering::Relaxed) == 1
        } else {
            sync.result.iter().all(|r| r.load(Ordering::Relaxed) == 0)
        });
    }
    sync.stop.store(true, Ordering::Relaxed);
    for handle in handles {
        handle.join().unwrap();
    }
    assert!(
        !timed_out,
        "rendezvous timed out: {tier:?}/{mode}, helper_mp={helper_mp}"
    );
    eprintln!("helper_mp={helper_mp} {tier:?}/{mode}: {relaxed}/{iters} relaxed outcomes");
    relaxed
}

#[test]
fn sb_mfence_all_tiers_and_backends() {
    let mut failures = Vec::new();
    for tier in [
        MemConsistency::Fast,
        MemConsistency::AcqRel,
        MemConsistency::FullTso,
    ] {
        for mode in ["interp", "block", "regions"] {
            let count = race(tier, mode, false);
            if count != 0 {
                failures.push(format!("{tier:?}/{mode}: {count}"));
            }
        }
    }
    assert!(failures.is_empty(), "SB+mfence allowed (0,0): {failures:?}");
}

#[test]
#[ignore = "release-only fence cost measurement; run explicitly"]
fn fence_loop_costs() {
    const UNROLL: u64 = 32;
    const LOOPS: u64 = 20_000;
    for tier in [
        MemConsistency::Fast,
        MemConsistency::AcqRel,
        MemConsistency::FullTso,
    ] {
        for mode in ["interp", "block", "regions"] {
            for kind in ["MFENCE", "LFENCE", "SFENCE"] {
                let mut times = [0u128; 2];
                for (fenced, time) in times.iter_mut().enumerate() {
                    let backend: Box<dyn Backend> = match mode {
                        "interp" => Box::new(InterpreterBackend),
                        "block" => Box::new(JitBackend::new()),
                        _ => Box::new(JitBackend::with_superblocks(RegionCaps {
                            max_blocks: 4,
                            max_icount: 256,
                        })),
                    };
                    let mut a = CodeAssembler::new(64).unwrap();
                    let mut top = a.create_label();
                    let mut body = a.create_label();
                    a.set_label(&mut top).unwrap();
                    a.jmp(body).unwrap();
                    a.set_label(&mut body).unwrap();
                    for _ in 0..UNROLL {
                        if fenced == 1 {
                            match kind {
                                "MFENCE" => a.mfence().unwrap(),
                                "LFENCE" => a.lfence().unwrap(),
                                "SFENCE" => a.sfence().unwrap(),
                                _ => unreachable!(),
                            }
                        } else {
                            a.nop().unwrap();
                        }
                    }
                    a.dec(rcx).unwrap();
                    a.jnz(top).unwrap();
                    a.hlt().unwrap();
                    let mut config = VmConfig::flat(0x10000);
                    config.consistency = tier;
                    let mut vm = Vm::with_backend(config, backend);
                    vm.map(0, 0x10000, Prot::RW, RegionKind::Ram).unwrap();
                    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
                    let mut cpu = vm.new_vcpu();
                    // Compile and warm before timing; report the best of three.
                    *time = u128::MAX;
                    for sample in 0..4 {
                        cpu.set_reg(Reg::Rip, CODE);
                        cpu.set_reg(Reg::Rcx, LOOPS);
                        let start = Instant::now();
                        assert!(matches!(cpu.run(&vm, Some(LOOPS * 3 + 4)), Exit::Hlt));
                        if sample > 0 {
                            *time = (*time).min(start.elapsed().as_nanos());
                        }
                    }
                    if mode == "regions" {
                        assert!(vm.cache.regions() > 0);
                    }
                }
                eprintln!("{kind} cost {tier:?}/{mode}: control={:.3} ns/slot, fenced={:.3} ns/fence, delta={:.3} ns/fence",
                times[0] as f64 / (UNROLL * LOOPS) as f64,
                times[1] as f64 / (UNROLL * LOOPS) as f64,
                (times[1] as f64 - times[0] as f64) / (UNROLL * LOOPS) as f64);
            }
        }
    }
}

#[test]
fn serializing_instructions_lift_with_a_full_fence() {
    use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
    use x86jit_core::{FenceKind, IrOp};
    // SDM Vol 3A §11.3: CPUID and IRET serialize execution. IRET orders prior
    // memory accesses before its frame loads and completes those loads before return.
    // Other listed instructions must remain unsupported.
    for (bytes, mode, supported) in [
        (&[0x0f, 0xa2, 0xf4][..], CpuMode::Long64, true),
        (&[0xcf][..], CpuMode::Real16, true),
        (&[0x66, 0xcf][..], CpuMode::Long64, true),
        (&[0xcf][..], CpuMode::Long64, true),
        (&[0x48, 0xcf][..], CpuMode::Long64, true),
        (&[0x66, 0xcf][..], CpuMode::Compat32, true),
        (&[0xcf][..], CpuMode::Compat32, true),
        (&[0x0f, 0xaa][..], CpuMode::Long64, false), // RSM
        (&[0x0f, 0x01, 0xe8][..], CpuMode::Long64, false), // SERIALIZE
        (&[0x0f, 0x22, 0xc0][..], CpuMode::Long64, false), // MOV CR0,RAX
        (&[0x0f, 0x22, 0xd8][..], CpuMode::Long64, false), // MOV CR3,RAX
        (&[0x0f, 0x23, 0xc0][..], CpuMode::Long64, false), // MOV DR0,RAX
        (&[0x0f, 0x08][..], CpuMode::Long64, false), // INVD
        (&[0x0f, 0x09][..], CpuMode::Long64, false), // WBINVD
        (&[0x0f, 0x30][..], CpuMode::Long64, false), // WRMSR
        (&[0x0f, 0x01, 0x10][..], CpuMode::Long64, false), // LGDT [RAX]
        (&[0x0f, 0x01, 0x18][..], CpuMode::Long64, false), // LIDT [RAX]
        (&[0x0f, 0x00, 0xd0][..], CpuMode::Long64, false), // LLDT AX
        (&[0x0f, 0x00, 0xd8][..], CpuMode::Long64, false), // LTR AX
        (&[0x0f, 0x01, 0x38][..], CpuMode::Long64, false), // INVLPG [RAX]
        (&[0x66, 0x0f, 0x38, 0x80, 0x00][..], CpuMode::Long64, false), // INVEPT
        (&[0x66, 0x0f, 0x38, 0x81, 0x00][..], CpuMode::Long64, false), // INVVPID
    ] {
        let mut vm = Vm::new(VmConfig::flat(0x10000));
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(CODE, bytes).unwrap();
        let ir = lift_block(&vm.mem, FetchAddr::flat(CODE), mode);
        if supported {
            let ir = ir.unwrap();
            if let Some(nt_check) = ir.ops.iter().position(|op| matches!(op, IrOp::CheckIretNt)) {
                let fences: Vec<_> = ir
                    .ops
                    .iter()
                    .enumerate()
                    .filter_map(|(index, op)| {
                        matches!(
                            op,
                            IrOp::Fence {
                                kind: FenceKind::Full
                            }
                        )
                        .then_some(index)
                    })
                    .collect();
                assert_eq!(fences.len(), 2, "{mode:?}/{bytes:x?}: {ir:?}");
                let first_load = ir
                    .ops
                    .iter()
                    .position(|op| matches!(op, IrOp::Load { .. }))
                    .unwrap();
                let last_load = ir
                    .ops
                    .iter()
                    .rposition(|op| matches!(op, IrOp::Load { .. }))
                    .unwrap();
                let frame_check = ir
                    .ops
                    .iter()
                    .position(|op| matches!(op, IrOp::CheckIret { .. }))
                    .unwrap();
                let restore = ir
                    .ops
                    .iter()
                    .position(|op| matches!(op, IrOp::RestoreFlags { iret: true, .. }))
                    .unwrap();
                assert!(
                    nt_check < fences[0]
                        && fences[0] < first_load
                        && last_load < frame_check
                        && frame_check < fences[1]
                        && fences[1] < restore,
                    "{mode:?}/{bytes:x?}: {ir:?}"
                );
                continue;
            }
            let position = ir
                .ops
                .iter()
                .position(|op| matches!(op, IrOp::Cpuid | IrOp::IretReal))
                .unwrap();
            assert!(matches!(
                ir.ops[position - 1],
                IrOp::Fence {
                    kind: FenceKind::Full
                }
            ));
        } else {
            assert!(
                matches!(ir, Err(x86jit_core::lift::LiftError::Unsupported { .. })),
                "unexpected serializing coverage: {bytes:x?}: {ir:?}"
            );
        }
    }
}

/// SDM Vol 3A §11.2.2 and Vol 2B §4.3 SFENCE: observing publication Y
/// requires observing the preceding data X. This exposes the interpreter's
/// host fence on weak hosts, unlike the slower SB store-to-load path.
#[cfg(target_arch = "aarch64")]
#[test]
fn interpreter_fences_order_message_passing() {
    const ROUNDS: u64 = 2_000_000;
    let mut writer = CodeAssembler::new(64).unwrap();
    let mut top = writer.create_label();
    writer.set_label(&mut top).unwrap();
    writer.inc(rdi).unwrap();
    writer.mov(qword_ptr(rbx), rdi).unwrap();
    writer.sfence().unwrap();
    writer.mov(qword_ptr(rdx), rdi).unwrap();
    writer.dec(r9).unwrap();
    writer.jnz(top).unwrap();
    writer.hlt().unwrap();
    let mut reader = CodeAssembler::new(64).unwrap();
    let mut top = reader.create_label();
    let mut ordered = reader.create_label();
    reader.set_label(&mut top).unwrap();
    reader.mov(rax, qword_ptr(rdx)).unwrap();
    reader.lfence().unwrap();
    reader.mov(rcx, qword_ptr(rbx)).unwrap();
    reader.cmp(rcx, rax).unwrap();
    reader.jae(ordered).unwrap();
    reader.inc(r8).unwrap();
    reader.set_label(&mut ordered).unwrap();
    reader.dec(r9).unwrap();
    reader.jnz(top).unwrap();
    reader.hlt().unwrap();
    let mut vm = Vm::new(VmConfig::flat(0x10000));
    vm.map(0, 0x10000, Prot::RW, RegionKind::Ram).unwrap();
    vm.write_bytes(CODE, &writer.assemble(CODE).unwrap())
        .unwrap();
    vm.write_bytes(CODE + 0x100, &reader.assemble(CODE + 0x100).unwrap())
        .unwrap();
    let vm = Arc::new(vm);
    let start = Arc::new(std::sync::Barrier::new(4));
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = std::sync::mpsc::channel();
    let mut handles = Vec::new();
    for addr in [CODE, CODE + 0x100] {
        let vm = vm.clone();
        let start = start.clone();
        let tx = tx.clone();
        handles.push(thread::spawn(move || {
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, addr);
            cpu.set_reg(Reg::R9, ROUNDS);
            cpu.set_reg(Reg::Rbx, X);
            cpu.set_reg(Reg::Rdx, Y);
            start.wait();
            let exit = cpu.run(&vm, Some(ROUNDS * 3 + 4));
            tx.send((exit, cpu.reg(Reg::R8))).unwrap();
        }));
    }
    let disturbance = {
        let vm = vm.clone();
        let start = start.clone();
        let stop = stop.clone();
        thread::spawn(move || {
            start.wait();
            let mut value = 0;
            while !stop.load(Ordering::Relaxed) {
                // Independent cache-line traffic makes delayed X publication
                // observable without adding ordering to either guest thread.
                for _ in 0..64 {
                    vm.mem.write(X + 8, value, 8).unwrap();
                    value = value.wrapping_add(1);
                }
                thread::yield_now();
            }
        })
    };
    start.wait();
    let results: Vec<_> = (0..2)
        .map(|_| rx.recv_timeout(Duration::from_secs(30)))
        .collect();
    stop.store(true, Ordering::Relaxed);
    disturbance.join().unwrap();
    // On timeout do not wait unboundedly for a broken execution path. Guest
    // runs also have a finite block budget; dropped handles can finish alone.
    assert!(
        results.iter().all(Result::is_ok),
        "interpreter MP timed out: {results:?}"
    );
    for h in handles {
        h.join().unwrap();
    }
    for (exit, forbidden) in results.into_iter().map(Result::unwrap) {
        assert!(matches!(exit, Exit::Hlt));
        assert_eq!(
            forbidden, 0,
            "interpreter fence allowed publication before data"
        );
    }
}

#[test]
fn real16_iret_uses_interpreter_with_either_backend() {
    use x86jit_core::CpuMode;
    // SDM Vol 3A §11.3: the serializing return includes stack reads.
    // Real16 dispatch must use the same interpreter with an injected JIT backend.
    for backend in [
        Box::new(InterpreterBackend) as Box<dyn Backend>,
        Box::new(JitBackend::new()),
    ] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
        vm.set_cpu_mode(CpuMode::Real16);
        vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.write_bytes(CODE, &[0xcf]).unwrap();
        vm.write_bytes(0x2000, &[0xb8, 0x34, 0x12, 0xf4]).unwrap(); // mov ax,0x1234; hlt
        vm.write_bytes(0x8000, &[0x00, 0x20, 0, 0, 0x02, 0x02])
            .unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rsp, 0x8000);
        assert!(matches!(cpu.run(&vm, Some(4)), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 0x1234);
        assert_eq!(cpu.reg(Reg::Rsp), 0x8006);
        assert!(cpu.flags().if_);
    }
}

#[test]
#[ignore = "release-only helper cost measurement; run explicitly"]
fn helper_string_loop_costs() {
    const LOOPS: u64 = 20_000;
    for tier in [
        MemConsistency::Fast,
        MemConsistency::AcqRel,
        MemConsistency::FullTso,
    ] {
        for mode in ["block", "regions"] {
            for copy in [false, true] {
                for bytes in [1u64, 64, 4096] {
                    let mut a = CodeAssembler::new(64).unwrap();
                    let mut top = a.create_label();
                    a.set_label(&mut top).unwrap();
                    a.mov(rdi, X).unwrap();
                    a.mov(rsi, 0xa000u64).unwrap();
                    a.mov(rcx, bytes).unwrap();
                    if copy {
                        a.rep().movsb().unwrap();
                    } else {
                        a.rep().stosb().unwrap();
                    }
                    a.dec(r8).unwrap();
                    a.jnz(top).unwrap();
                    a.hlt().unwrap();
                    let code = a.assemble(CODE).unwrap();
                    let backend: Box<dyn Backend> = if mode == "regions" {
                        Box::new(JitBackend::with_superblocks(RegionCaps {
                            max_blocks: 4,
                            max_icount: 32,
                        }))
                    } else {
                        Box::new(JitBackend::new())
                    };
                    let mut config = VmConfig::flat(0x10000);
                    config.consistency = tier;
                    let mut vm = Vm::with_backend(config, backend);
                    vm.map(0, 0x10000, Prot::RW, RegionKind::Ram).unwrap();
                    vm.write_bytes(CODE, &code).unwrap();
                    vm.write_bytes(0xa000, &vec![0x5a; 4096]).unwrap();
                    let mut times = Vec::new();
                    for sample in 0..8 {
                        let mut cpu = vm.new_vcpu();
                        cpu.set_reg(Reg::Rip, CODE);
                        cpu.set_reg(Reg::Rax, 0x5a);
                        cpu.set_reg(Reg::R8, LOOPS);
                        let start = Instant::now();
                        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                        let elapsed = start.elapsed().as_nanos();
                        if sample != 0 {
                            times.push(elapsed);
                        }
                        let mut out = vec![0; bytes as usize];
                        vm.mem.read_bytes(X, &mut out).unwrap();
                        assert!(out.iter().all(|b| *b == 0x5a));
                    }
                    if mode == "regions" {
                        assert!(vm.cache.regions() > 0);
                    }
                    times.sort_unstable();
                    let median = times[times.len() / 2];
                    let mut deviations: Vec<_> = times.iter().map(|v| v.abs_diff(median)).collect();
                    deviations.sort_unstable();
                    eprintln!("helper-cost {tier:?}/{mode}/{} bytes={bytes} loops={LOOPS} median_ns={median} mad_ns={}",
                        if copy { "movsb" } else { "stosb" }, deviations[deviations.len()/2]);
                }
            }
        }
    }
}
