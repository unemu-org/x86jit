use iced_x86::code_asm::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use x86jit_core::{
    Backend, CpuSource, Exit, Feature, Flags, GuestCpuFeatures, InterpreterBackend, Prot,
    RandomKind, Reg, RegionCaps, RegionKind, TscSample, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

fn backends() -> [Box<dyn Backend>; 2] {
    [Box::new(InterpreterBackend), Box::new(JitBackend::new())]
}

fn vm(backend: Box<dyn Backend>, bytes: &[u8]) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x3000), backend);
    vm.map(0x1000, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, bytes).unwrap();
    vm
}

#[test]
fn cpu_source_instructions_execute_on_both_backends() {
    let mut exits = Vec::new();
    for bytes in [
        &[0x0f, 0x01, 0xf9, 0xf4][..],
        &[0x48, 0x0f, 0xc7, 0xf0, 0xf4][..],
        &[0x48, 0x0f, 0xc7, 0xf8, 0xf4][..],
    ] {
        for backend in backends() {
            let vm = vm(backend, bytes);
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            exits.push(cpu.run(&vm, None));
        }
    }
    assert!(
        exits.iter().all(|exit| matches!(exit, Exit::Hlt)),
        "{exits:?}"
    );
}

#[test]
fn cpu_source_default_rdtscp() {
    // SDM Vol 2B RDTSCP, Description/Flags Affected: zero-extended outputs,
    // no flag changes. The values themselves are the engine's deterministic policy.
    for backend in backends() {
        let vm = vm(backend, &[0x0f, 0x01, 0xf9, 0xf4]);
        let mut cpu = vm.new_vcpu();
        for reg in [Reg::Rax, Reg::Rcx, Reg::Rdx] {
            cpu.set_reg(reg, u64::MAX);
        }
        let flags = cpu.cpu.flags;
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 0x1234_5678);
        assert_eq!(cpu.reg(Reg::Rdx), 0);
        assert_eq!(cpu.reg(Reg::Rcx), 0);
        assert_eq!(cpu.cpu.flags, flags);
    }
}

#[test]
fn cpu_source_default_rdrand_lifts() {
    for backend in backends() {
        let vm = vm(backend, &[0x48, 0x0f, 0xc7, 0xf0, 0xf4]);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert!(cpu.cpu.flags.cf);
    }
}

#[test]
fn cpu_source_default_rdseed_lifts() {
    for backend in backends() {
        let vm = vm(backend, &[0x48, 0x0f, 0xc7, 0xf8, 0xf4]);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert!(cpu.cpu.flags.cf);
    }
}

const STATUS: u64 = 0x8d5;
const SAMPLE: u64 = 0xfedc_ba98_7654_3210;

fn dirty_flags() -> Flags {
    let mut flags = Flags::default();
    flags.set_flags16((STATUS | 0x600) as u16);
    flags
}

#[derive(Debug, PartialEq, Eq)]
enum Request {
    Tsc(u64),
    Random(u64, RandomKind, u8),
}

#[derive(Debug, Default)]
struct Source {
    ticks: AtomicU64,
    random_calls: AtomicU64,
    requests: Mutex<Vec<Request>>,
}

impl CpuSource for Source {
    fn read_tsc(&self, vcpu_id: u64) -> TscSample {
        self.requests.lock().unwrap().push(Request::Tsc(vcpu_id));
        TscSample {
            counter: 0x1234_5678_ffff_fffe + self.ticks.fetch_add(1, Ordering::SeqCst),
            aux: 0x8000_0000 | vcpu_id as u32,
        }
    }

    fn read_random(&self, vcpu_id: u64, kind: RandomKind, size: u8) -> Option<u64> {
        self.requests
            .lock()
            .unwrap()
            .push(Request::Random(vcpu_id, kind, size));
        // Fail once, then succeed, independently of the engine's status flags.
        (self.random_calls.fetch_add(1, Ordering::SeqCst) % 2 == 1).then_some(SAMPLE)
    }
}

fn run(vm: &Vm, cpu: &mut Vcpu) {
    cpu.set_reg(Reg::Rip, 0x1000);
    assert!(matches!(cpu.run(vm, None), Exit::Hlt));
}

#[test]
fn cpu_source_tsc_samples_once_per_instruction_for_two_vcpus() {
    // SDM Vol 2B RDTSC/RDTSCP, Description/Flags Affected; Vol 3B §20.17.2:
    // paired TSC/AUX sample and zero-extended output registers, flags unchanged.
    for aux in [false, true] {
        for backend in backends() {
            let bytes: &[u8] = if aux {
                &[0x0f, 0x01, 0xf9, 0xf4]
            } else {
                &[0x0f, 0x31, 0xf4]
            };
            let mut vm = vm(backend, bytes);
            let source = Arc::new(Source::default());
            vm.set_cpu_source(Some(source.clone()));
            let mut cpus = [vm.new_vcpu(), vm.new_vcpu()];
            assert_ne!(cpus[0].cpu.source_id, cpus[1].cpu.source_id);
            let mut previous = 0;
            for iteration in 0..6 {
                let cpu = &mut cpus[iteration % 2];
                for reg in [Reg::Rax, Reg::Rdx, Reg::Rcx] {
                    cpu.set_reg(reg, u64::MAX);
                }
                cpu.set_flags(dirty_flags());
                let flags = cpu.flags();
                run(&vm, cpu);
                assert!(cpu.reg(Reg::Rax) <= u64::from(u32::MAX));
                assert!(cpu.reg(Reg::Rdx) <= u64::from(u32::MAX));
                let counter = (cpu.reg(Reg::Rdx) << 32) | cpu.reg(Reg::Rax);
                assert_eq!(counter, 0x1234_5678_ffff_fffe + iteration as u64);
                assert!(counter >= previous);
                previous = counter;
                assert_eq!(
                    cpu.reg(Reg::Rcx),
                    if aux {
                        0x8000_0000 | cpu.cpu.source_id
                    } else {
                        u64::MAX
                    }
                );
                assert_eq!(cpu.flags(), flags);
            }
            assert_eq!(source.ticks.load(Ordering::SeqCst), 6);
            assert_eq!(
                *source.requests.lock().unwrap(),
                (0..6)
                    .map(|i| Request::Tsc(cpus[i % 2].cpu.source_id))
                    .collect::<Vec<_>>()
            );
        }
    }
}

fn random_bytes(kind: RandomKind, size: u8) -> Vec<u8> {
    let mut asm = CodeAssembler::new(64).unwrap();
    match (kind, size) {
        (RandomKind::Rdrand, 2) => asm.rdrand(r9w),
        (RandomKind::Rdrand, 4) => asm.rdrand(r9d),
        (RandomKind::Rdrand, 8) => asm.rdrand(r9),
        (RandomKind::Rdseed, 2) => asm.rdseed(r9w),
        (RandomKind::Rdseed, 4) => asm.rdseed(r9d),
        (RandomKind::Rdseed, 8) => asm.rdseed(r9),
        _ => unreachable!(),
    }
    .unwrap();
    asm.hlt().unwrap();
    asm.assemble(0x1000).unwrap()
}

#[test]
fn cpu_source_random_widths_failure_success_and_flags() {
    // SDM Vol 2B RDRAND/RDSEED, Operation/Flags Affected: failure zeroes
    // the destination width, CF reports success, OF/SF/ZF/AF/PF are always clear.
    // SDM Vol 1 §3.4.1.1: 16-bit preserves upper bits; 32-bit zero-extends.
    for kind in [RandomKind::Rdrand, RandomKind::Rdseed] {
        for size in [2, 4, 8] {
            for backend in backends() {
                let mut vm = vm(backend, &random_bytes(kind, size));
                let source = Arc::new(Source::default());
                vm.set_cpu_source(Some(source.clone()));
                let mut cpu = vm.new_vcpu();
                for success in [false, true] {
                    cpu.set_reg(Reg::R9, u64::MAX);
                    cpu.set_flags(dirty_flags());
                    run(&vm, &mut cpu);
                    let expected = match (success, size) {
                        (false, 2) => 0xffff_ffff_ffff_0000,
                        (false, _) => 0,
                        (true, 2) => 0xffff_ffff_ffff_3210,
                        (true, 4) => 0x7654_3210,
                        (true, 8) => SAMPLE,
                        _ => unreachable!(),
                    };
                    assert_eq!(cpu.reg(Reg::R9), expected, "{kind:?}/{size}/{success}");
                    assert_eq!(cpu.flags().to_rflags() & STATUS, u64::from(success));
                    assert_eq!(u64::from(cpu.flags().to_flags16()) & 0x600, 0x600);
                }
                assert_eq!(
                    *source.requests.lock().unwrap(),
                    vec![
                        Request::Random(cpu.cpu.source_id, kind, size),
                        Request::Random(cpu.cpu.source_id, kind, size)
                    ]
                );
            }
        }
    }
}

#[derive(Debug)]
struct ZeroSource;

impl CpuSource for ZeroSource {
    fn read_tsc(&self, _: u64) -> TscSample {
        TscSample { counter: 0, aux: 0 }
    }

    fn read_random(&self, _: u64, _: RandomKind, _: u8) -> Option<u64> {
        Some(0)
    }
}

#[test]
fn cpu_source_successful_zero_is_not_failure() {
    // SDM Vol 2B RDRAND/RDSEED, Operation: CF reports availability, not value.
    for kind in [RandomKind::Rdrand, RandomKind::Rdseed] {
        for backend in backends() {
            let mut vm = vm(backend, &random_bytes(kind, 8));
            vm.set_cpu_source(Some(Arc::new(ZeroSource)));
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::R9, u64::MAX);
            cpu.set_flags(dirty_flags());
            run(&vm, &mut cpu);
            assert_eq!(cpu.reg(Reg::R9), 0);
            assert_eq!(cpu.flags().to_rflags() & STATUS, 1);
        }
    }
}

#[test]
fn cpu_source_is_not_sampled_past_an_earlier_fault() {
    // SDM Vol 3A §7.5: instructions after a fault have not executed.
    for instruction in [
        &[0x0f, 0x31][..],
        &[0x0f, 0x01, 0xf9][..],
        &[0x49, 0x0f, 0xc7, 0xf1][..],
        &[0x49, 0x0f, 0xc7, 0xf9][..],
    ] {
        let mut bytes = vec![0x48, 0x8b, 0x0e]; // mov rcx,[rsi]
        bytes.extend(instruction);
        bytes.push(0xf4);
        for backend in backends() {
            let mut vm = vm(backend, &bytes);
            let source = Arc::new(Source::default());
            vm.set_cpu_source(Some(source.clone()));
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            cpu.set_reg(Reg::Rsi, 0x3000);
            assert!(matches!(cpu.run(&vm, None), Exit::UnmappedMemory { .. }));
            assert!(source.requests.lock().unwrap().is_empty());
        }
    }
}

#[test]
fn cpu_source_random_flags_and_intervening_reader_survive_elision() {
    // SDM Vol 2A ADD, Operation/Flags Affected: 8000000000000000H doubled
    // produces CF=OF=1. SETB reads that CF before the random instruction;
    // SETO after it must observe OF=0 (Vol 2B RDRAND/RDSEED, Flags Affected).
    for kind in [RandomKind::Rdrand, RandomKind::Rdseed] {
        for backend in backends() {
            let mut asm = CodeAssembler::new(64).unwrap();
            asm.add(rax, rax).unwrap();
            asm.setb(bl).unwrap();
            match kind {
                RandomKind::Rdrand => asm.rdrand(r9),
                RandomKind::Rdseed => asm.rdseed(r9),
            }
            .unwrap();
            asm.seto(cl).unwrap();
            asm.hlt().unwrap();
            let mut vm = vm(backend, &asm.assemble(0x1000).unwrap());
            let source = Arc::new(Source::default());
            vm.set_cpu_source(Some(source));
            let mut cpu = vm.new_vcpu();
            for success in [false, true] {
                cpu.set_reg(Reg::Rax, 0x8000_0000_0000_0000);
                cpu.set_reg(Reg::Rcx, u64::MAX);
                cpu.set_reg(Reg::Rbx, 0);
                run(&vm, &mut cpu);
                assert_eq!(cpu.reg(Reg::Rbx), 1);
                assert_eq!(cpu.reg(Reg::Rcx) & 0xff, 0);
                assert_eq!(cpu.flags().to_rflags() & STATUS, u64::from(success));
            }
        }
    }
}

#[test]
fn cpu_source_random_flags_at_later_fault() {
    // SDM Vol 3A §7.5: a fault exposes the preceding instruction's flags.
    // The final XOR makes preceding ALU flags dead on the successful path.
    for kind in [RandomKind::Rdrand, RandomKind::Rdseed] {
        for backend in backends() {
            let mut asm = CodeAssembler::new(64).unwrap();
            asm.add(rax, rax).unwrap();
            match kind {
                RandomKind::Rdrand => asm.rdrand(r9),
                RandomKind::Rdseed => asm.rdseed(r9),
            }
            .unwrap();
            let fault = 0x1000 + asm.assemble(0x1000).unwrap().len() as u64;
            asm.mov(rcx, qword_ptr(rsi)).unwrap();
            asm.xor(eax, eax).unwrap();
            asm.hlt().unwrap();
            let mut vm = vm(backend, &asm.assemble(0x1000).unwrap());
            let source = Arc::new(Source::default());
            vm.set_cpu_source(Some(source.clone()));
            let mut cpu = vm.new_vcpu();
            for success in [false, true] {
                cpu.set_reg(Reg::Rip, 0x1000);
                cpu.set_reg(Reg::Rax, 0x8000_0000_0000_0000);
                cpu.set_reg(Reg::Rsi, 0x3000);
                assert!(matches!(cpu.run(&vm, None), Exit::UnmappedMemory { .. }));
                assert_eq!(cpu.reg(Reg::Rip), fault);
                assert_eq!(cpu.reg(Reg::R9), if success { SAMPLE } else { 0 });
                assert_eq!(cpu.flags().to_rflags() & STATUS, u64::from(success));
            }
            assert_eq!(source.random_calls.load(Ordering::SeqCst), 2);
        }
    }
}

#[test]
fn cpu_source_defaults_are_pinned_and_per_vcpu() {
    for backend in backends() {
        let vm = vm(backend, &[0x0f, 0x31, 0xf4]);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rax, u64::MAX);
        cpu.set_reg(Reg::Rdx, u64::MAX);
        cpu.set_reg(Reg::Rcx, u64::MAX);
        cpu.set_flags(dirty_flags());
        let flags = cpu.flags();
        run(&vm, &mut cpu);
        assert_eq!(cpu.reg(Reg::Rax), 0x1234_5678);
        assert_eq!(cpu.reg(Reg::Rdx), 0);
        assert_eq!(cpu.reg(Reg::Rcx), u64::MAX);
        assert_eq!(cpu.flags(), flags);
    }
    for backend in backends() {
        // RDRAND and RDSEED share one deterministic per-vCPU fallback stream.
        let vm = vm(
            backend,
            &[0x48, 0x0f, 0xc7, 0xf0, 0x48, 0x0f, 0xc7, 0xfb, 0xf4],
        );
        let mut cpus = [vm.new_vcpu(), vm.new_vcpu()];
        for cpu in &mut cpus {
            run(&vm, cpu);
            assert_eq!(cpu.reg(Reg::Rax), 0xff1c_1522_4c36_be38);
            assert_eq!(cpu.reg(Reg::Rbx), 0x44a7_2703_e495_37d9);
            assert_eq!(cpu.flags().to_rflags() & STATUS, 1);
        }
        let saved = cpus[0].cpu.clone();
        run(&vm, &mut cpus[0]);
        let expected = (cpus[0].reg(Reg::Rax), cpus[0].reg(Reg::Rbx));
        assert_eq!(expected, (0x85fb_c6a8_223c_7826, 0xba00_8c15_3a59_f8af));
        cpus[0].cpu = saved;
        run(&vm, &mut cpus[0]);
        assert_eq!((cpus[0].reg(Reg::Rax), cpus[0].reg(Reg::Rbx)), expected);
    }
}

#[test]
fn cpu_source_region_loop_samples_each_executed_instruction() {
    let mut asm = CodeAssembler::new(64).unwrap();
    let mut top = asm.create_label();
    asm.set_label(&mut top).unwrap();
    asm.rdtscp().unwrap();
    asm.mov(r10d, ecx).unwrap();
    asm.rdrand(r9).unwrap();
    asm.dec(r8d).unwrap();
    asm.jnz(top).unwrap();
    asm.hlt().unwrap();
    let backend = JitBackend::with_superblocks(RegionCaps {
        max_blocks: 8,
        max_icount: 64,
    });
    let mut vm = vm(Box::new(backend), &asm.assemble(0x1000).unwrap());
    let source = Arc::new(Source::default());
    vm.set_cpu_source(Some(source.clone()));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::R8, 4);
    run(&vm, &mut cpu);
    assert!(vm.cache.regions() > 0);
    assert_eq!(cpu.reg(Reg::R10), 0x8000_0000 | cpu.cpu.source_id);
    assert_eq!(cpu.reg(Reg::Rax), 1);
    assert_eq!(cpu.reg(Reg::Rdx), 0x1234_5679);
    assert_eq!(cpu.reg(Reg::R9), SAMPLE);
    assert_eq!(source.ticks.load(Ordering::SeqCst), 4);
    assert_eq!(source.random_calls.load(Ordering::SeqCst), 4);
}

#[test]
fn cpu_source_handle_lifecycle_and_fork() {
    for backend in backends() {
        let mut vm = vm(backend, &[0x0f, 0x01, 0xf9, 0xf4]);
        let mut original = vm.new_vcpu();
        let source = Arc::new(Source::default());
        vm.set_cpu_source(Some(source.clone()));
        let mut cpu = vm.new_vcpu();
        vm.set_cpu_source(None);
        let mut removed = vm.new_vcpu();
        run(&vm, &mut original);
        run(&vm, &mut removed);
        assert_eq!(original.reg(Reg::Rax), 0x1234_5678);
        assert_eq!(removed.reg(Reg::Rax), 0x1234_5678);
        run(&vm, &mut cpu);
        assert_eq!(source.ticks.load(Ordering::SeqCst), 1);

        vm.set_cpu_source(Some(source.clone()));
        let child = vm.fork_with_backend(Box::new(JitBackend::new())).unwrap();
        let mut cloned = child.new_vcpu();
        let fresh_id = cloned.cpu.source_id;
        cloned.cpu = cpu.cpu.clone();
        assert_eq!(cloned.cpu.source_id, cpu.cpu.source_id);
        run(&child, &mut cloned);
        let mut fresh = child.new_vcpu();
        let parent_fresh = vm.new_vcpu();
        assert_ne!(fresh.cpu.source_id, fresh_id);
        assert_ne!(fresh.cpu.source_id, parent_fresh.cpu.source_id);
        run(&child, &mut fresh);
        assert_eq!(source.ticks.load(Ordering::SeqCst), 3);
    }
}

#[test]
fn cpu_source_features_have_witness_bits_and_stay_opt_in() {
    // SDM Vol 2B RDRAND/RDSEED/RDTSCP, #UD feature tests; Vol 3B §20.17.2.
    for (feature, leaf, word, bit) in [
        (Feature::Rdrand, 1, 2, 30),
        (Feature::Rdseed, 7, 1, 18),
        (Feature::Rdtscp, 0x8000_0001, 3, 27),
    ] {
        let words = |f: GuestCpuFeatures| {
            let r = f.cpuid(leaf, 0);
            [r.eax, r.ebx, r.ecx, r.edx]
        };
        assert_eq!(words(GuestCpuFeatures::default())[word] & (1 << bit), 0);
        assert_eq!(
            words(GuestCpuFeatures::empty().with(feature))[word] & (1 << bit),
            1 << bit
        );
    }
}
