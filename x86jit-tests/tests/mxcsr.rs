//! Architectural MXCSR storage (SDM Vol 1 §§10.2.3, 11.6.4–11.6.6).

use iced_x86::code_asm::*;
use x86jit_core::{Backend, Exit, InterpreterBackend, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

const CODE: u64 = 0x1000;
const DATA: u64 = 0x8000;
const ORACLE_CODE: u64 = 0x210000;
const ORACLE_DATA: u64 = 0x220000;
// SDM Vol 1 §10.2.3, Figure 10-3: RC=11, FTZ, DAZ, all masks, IE and PE.
const MODE: u32 = 0xffe1;

fn backends() -> [Box<dyn Backend>; 2] {
    [Box::new(InterpreterBackend), Box::new(JitBackend::new())]
}

fn vm(backend: Box<dyn Backend>, build: impl FnOnce(&mut CodeAssembler)) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    build(&mut a);
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
    vm
}

fn read_u32(vm: &Vm, addr: u64) -> u32 {
    let mut bytes = [0; 4];
    vm.read_bytes(addr, &mut bytes).unwrap();
    u32::from_le_bytes(bytes)
}

#[test]
fn mxcsr_legacy_load_store_round_trip() {
    load_store_round_trip(false);
}

#[test]
fn mxcsr_vex_load_store_round_trip() {
    load_store_round_trip(true);
}

fn load_store_round_trip(vex: bool) {
    for backend in backends() {
        let vm = vm(backend, |a| {
            if vex {
                a.vldmxcsr(dword_ptr(DATA)).unwrap();
                a.vstmxcsr(dword_ptr(DATA + 4)).unwrap();
            } else {
                a.ldmxcsr(dword_ptr(DATA)).unwrap();
                a.stmxcsr(dword_ptr(DATA + 4)).unwrap();
            }
            a.hlt().unwrap();
        });
        vm.write_bytes(DATA, &MODE.to_le_bytes()).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(read_u32(&vm, DATA + 4), MODE);
    }
}

#[test]
fn mxcsr_fxsave_fxrstor_round_trip() {
    for backend in backends() {
        let vm = vm(backend, |a| {
            a.ldmxcsr(dword_ptr(DATA)).unwrap();
            a.fxsave(xmmword_ptr(DATA + 64)).unwrap();
            a.ldmxcsr(dword_ptr(DATA + 4)).unwrap();
            a.fxrstor(xmmword_ptr(DATA + 64)).unwrap();
            a.stmxcsr(dword_ptr(DATA + 8)).unwrap();
            a.hlt().unwrap();
        });
        vm.write_bytes(DATA, &MODE.to_le_bytes()).unwrap();
        vm.write_bytes(DATA + 4, &0x1f80u32.to_le_bytes()).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        // SDM Vol 2A §3.3 FXSAVE, Tables 3-48/3-49: MXCSR at 24, mask at 28.
        assert_eq!(read_u32(&vm, DATA + 64 + 24), MODE);
        assert_eq!(read_u32(&vm, DATA + 64 + 28), 0xffff);
        assert_eq!(read_u32(&vm, DATA + 8), MODE);
    }
}

#[test]
fn mxcsr_ldmxcsr_reserved_bits_fault_without_commit() {
    reserved_bits_fault_without_commit(false, false);
}

#[test]
fn mxcsr_vldmxcsr_reserved_bits_fault_without_commit() {
    reserved_bits_fault_without_commit(false, true);
}

#[test]
fn mxcsr_fxrstor_reserved_bits_fault_without_commit() {
    reserved_bits_fault_without_commit(true, false);
}

fn reserved_bits_fault_without_commit(restore: bool, vex: bool) {
    for backend in backends() {
        let vm = vm(backend, |a| {
            if restore {
                a.fxrstor(xmmword_ptr(DATA)).unwrap();
            } else if vex {
                a.vldmxcsr(dword_ptr(DATA)).unwrap();
            } else {
                a.ldmxcsr(dword_ptr(DATA)).unwrap();
            }
            a.hlt().unwrap();
        });
        for bit in 16..32 {
            let mut image = [0x55; 512];
            let offset = if restore { 24 } else { 0 };
            image[offset..offset + 4].copy_from_slice(&(1u32 << bit).to_le_bytes());
            vm.write_bytes(DATA, &image).unwrap();
            let mut cpu = vm.new_vcpu_with_mxcsr(MODE);
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_xmm(0, 0x1234);
            cpu.set_fpu_cw(0x077f);
            cpu.set_fpr_bytes(3, &[0x42; 10]);
            let before = cpu.cpu.clone();
            // SDM Vol 1 §10.2.3; Vol 2A §3.3 LDMXCSR/FXRSTOR, Exceptions.
            assert!(matches!(
                cpu.run(&vm, None),
                Exit::Exception {
                    addr: CODE,
                    vector: 13,
                    ..
                }
            ));
            assert_eq!(cpu.reg(Reg::Rip), CODE);
            assert_eq!(format!("{:?}", cpu.cpu), format!("{before:?}"));
        }
    }
}

#[test]
fn mxcsr_fenv_round_trip() {
    for backend in backends() {
        let vm = vm(backend, |a| {
            a.ldmxcsr(dword_ptr(DATA)).unwrap();
            a.fnstenv(dword_ptr(DATA + 64)).unwrap();
            a.stmxcsr(dword_ptr(DATA + 92)).unwrap();
            a.ldmxcsr(dword_ptr(DATA + 4)).unwrap();
            a.fldenv(dword_ptr(DATA + 64)).unwrap();
            a.ldmxcsr(dword_ptr(DATA + 92)).unwrap();
            a.stmxcsr(dword_ptr(DATA + 8)).unwrap();
            a.hlt().unwrap();
        });
        vm.write_bytes(DATA, &MODE.to_le_bytes()).unwrap();
        vm.write_bytes(DATA + 4, &0x1f80u32.to_le_bytes()).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_fpu_cw(0x0b7f);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.fpu_cw(), 0x0b7f);
        assert_eq!(read_u32(&vm, DATA + 8), MODE);
    }
}

#[test]
fn mxcsr_embedder_seed_accessors_and_vcpu_isolation() {
    for backend in backends() {
        let vm = vm(backend, |a| {
            a.stmxcsr(dword_ptr(DATA)).unwrap();
            a.hlt().unwrap();
        });
        assert_eq!(vm.new_vcpu().mxcsr(), 0x1f80);
        let mut seeded = vm.new_vcpu_with_mxcsr(0x9fc0);
        let mut other = vm.new_vcpu();
        assert_eq!(seeded.mxcsr(), 0x9fc0);
        assert_eq!(seeded.cpu.clone().mxcsr, 0x9fc0);
        seeded.set_reg(Reg::Rip, CODE);
        assert!(matches!(seeded.run(&vm, None), Exit::Hlt));
        assert_eq!(read_u32(&vm, DATA), 0x9fc0);
        seeded.set_mxcsr(MODE);
        assert_eq!(seeded.mxcsr(), MODE);
        assert_eq!(other.mxcsr(), 0x1f80);
        other.set_reg(Reg::Rip, CODE);
        assert!(matches!(other.run(&vm, None), Exit::Hlt));
        assert_eq!(read_u32(&vm, DATA), 0x1f80);
        assert_eq!(seeded.mxcsr(), MODE);
    }
}

#[test]
fn mxcsr_embedder_rejects_reserved_bits_without_commit() {
    let vm = Vm::new(VmConfig::flat(0x1000));
    let mut cpu = vm.new_vcpu_with_mxcsr(0x9fc0);
    for bit in 16..32 {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cpu.set_mxcsr(1 << bit);
        }));
        assert!(result.is_err());
        assert_eq!(cpu.mxcsr(), 0x9fc0);
    }
}

#[test]
fn mxcsr_load_memory_fault_preserves_state() {
    for vex in [false, true] {
        for backend in backends() {
            let vm = vm(backend, |a| {
                if vex {
                    a.vldmxcsr(dword_ptr(0xfffeu64)).unwrap();
                } else {
                    a.ldmxcsr(dword_ptr(0xfffeu64)).unwrap();
                }
                a.hlt().unwrap();
            });
            let mut cpu = vm.new_vcpu_with_mxcsr(MODE);
            cpu.set_reg(Reg::Rip, CODE);
            let before = cpu.cpu.clone();
            assert!(matches!(cpu.run(&vm, None), Exit::UnmappedMemory { .. }));
            assert_eq!(format!("{:?}", cpu.cpu), format!("{before:?}"));
        }
    }
}

#[test]
fn mxcsr_fault_preserves_prior_instruction_results() {
    for restore in [false, true] {
        for backend in backends() {
            let vm = vm(backend, |a| {
                a.mov(eax, -1i32).unwrap();
                a.add(eax, 1).unwrap();
                if restore {
                    a.fxrstor(xmmword_ptr(DATA)).unwrap();
                } else {
                    a.ldmxcsr(dword_ptr(DATA)).unwrap();
                }
                a.hlt().unwrap();
            });
            vm.write_bytes(
                DATA + if restore { 24 } else { 0 },
                &0x10000u32.to_le_bytes(),
            )
            .unwrap();
            let mut cpu = vm.new_vcpu_with_mxcsr(MODE);
            cpu.set_reg(Reg::Rip, CODE);
            let mut bytes = [0; 16];
            vm.read_bytes(CODE, &mut bytes).unwrap();
            let mut decoder =
                iced_x86::Decoder::with_ip(64, &bytes, CODE, iced_x86::DecoderOptions::NONE);
            let _ = decoder.decode();
            let _ = decoder.decode();
            let fault_ip = decoder.ip();
            let exit = cpu.run(&vm, None);
            assert!(matches!(exit, Exit::Exception { addr, vector: 13, .. } if addr == fault_ip));
            assert_eq!(cpu.reg(Reg::Rip), fault_ip);
            assert_eq!(cpu.reg(Reg::Rax), 0);
            assert!(cpu.cpu.flags.cf);
            assert!(cpu.cpu.flags.zf);
            assert_eq!(cpu.mxcsr(), MODE);
        }
    }
}

fn differential_input() -> x86jit_tests::oracle::VectorInput {
    use x86jit_tests::oracle::VectorInput;
    use x86jit_tests::vector::{CpuSnapshot, MemChunk, MemKind, RunSpec};
    let mut a = CodeAssembler::new(64).unwrap();
    a.stmxcsr(dword_ptr(ORACLE_DATA)).unwrap();
    a.ldmxcsr(dword_ptr(ORACLE_DATA + 4)).unwrap();
    a.stmxcsr(dword_ptr(ORACLE_DATA + 8)).unwrap();
    a.hlt().unwrap();
    let mut bytes = vec![0; 16];
    bytes[4..8].copy_from_slice(&MODE.to_le_bytes());
    VectorInput {
        cpu_init: CpuSnapshot {
            mxcsr: 0x9fc0,
            ..Default::default()
        },
        mem_init: vec![
            MemChunk {
                addr: ORACLE_CODE,
                bytes: a.assemble(ORACLE_CODE).unwrap(),
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: ORACLE_DATA,
                bytes,
                kind: MemKind::Ram,
            },
        ],
        entry: ORACLE_CODE,
        run: RunSpec::UntilExit,
    }
}

#[test]
fn mxcsr_engine_snapshots_seed_and_capture_all_bits() {
    let input = differential_input();
    // The native oracle refuses mappings below its minimum guest address.
    assert!(input.mem_init.iter().all(|chunk| chunk.addr >= 0x10000));
    for backend in backends() {
        let outcome = x86jit_tests::oracle::run_with_backend(&input, backend);
        assert_eq!(outcome.cpu.mxcsr, MODE);
        let bytes = &outcome
            .mem
            .iter()
            .find(|m| m.addr == ORACLE_DATA)
            .unwrap()
            .bytes;
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 0x9fc0);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), MODE);
    }
}

#[cfg(feature = "unicorn")]
#[test]
fn mxcsr_seed_matches_unicorn() {
    let vector = x86jit_tests::builder::Vector::asm(|a| {
        a.stmxcsr(dword_ptr(DATA)).unwrap();
        a.hlt().unwrap();
    })
    .init(|cpu| cpu.mxcsr = 0x9fc0);
    vector.assert_matches_unicorn();
    assert_eq!(vector.interpret().cpu.mxcsr, 0x9fc0);
    assert_eq!(vector.unicorn().cpu.mxcsr, 0x9fc0);
}

#[cfg(feature = "unicorn")]
#[test]
fn mxcsr_nondefault_round_trip_matches_unicorn() {
    use x86jit_tests::oracle::Oracle;
    let input = differential_input();
    let reference = x86jit_tests::unicorn::UnicornOracle.run(&input);
    assert_eq!(reference.cpu.mxcsr, MODE);
    for backend in backends() {
        let outcome = x86jit_tests::oracle::run_with_backend(&input, backend);
        assert_eq!(outcome.cpu.mxcsr, reference.cpu.mxcsr);
        assert!(x86jit_tests::compare::compare(&reference, &outcome, &[]).is_none());
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn mxcsr_nondefault_round_trip_matches_native() {
    let input = differential_input();
    let reference = x86jit_tests::native::run_native(&input).expect("host supports SSE");
    assert_eq!(reference.cpu.mxcsr, MODE);
    for backend in backends() {
        let outcome = x86jit_tests::oracle::run_with_backend(&input, backend);
        assert_eq!(outcome.cpu.mxcsr, reference.cpu.mxcsr);
        assert!(x86jit_tests::compare::compare(&reference, &outcome, &[]).is_none());
    }
}
