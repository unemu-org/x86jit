//! SDM Vol 1 §§8.1.8–8.1.9 and §10.5.1: x87 pointer/opcode witnesses.
use iced_x86::code_asm::*;
use std::cell::RefCell;
use x86jit_core::{
    x87::{exec_fxstate, FpMem},
    Backend, CpuMode, CpuState, Exit, InterpreterBackend, Prot, Reg, RegionKind, Vm, VmConfig,
    X87PointerModel,
};

const CODE: u64 = 0x21_0000;
const DATA: u64 = 0x22_0000;
const OUT: u64 = 0x23_0000;

// AMD64 APM Vol 1 §6.2.5.2; Vol 5 FSTENV/FNSTENV: last non-control
// opcode is recorded and saved even when no unmasked exception is pending.
fn opcode_models(backend: Box<dyn Backend>) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x30_0000), backend);
    for addr in [CODE, DATA, OUT] {
        vm.map(addr, 4096, Prot::RWX, RegionKind::Ram).unwrap();
    }
    let mut a = CodeAssembler::new(64).unwrap();
    a.fld(qword_ptr(DATA)).unwrap(); // DD /0, absolute SIB: FOP=0x504.
    a.fnstenv(ptr(OUT)).unwrap();
    a.fld1().unwrap(); // D9 E8: register form also records FOP.
    a.fnstenv(ptr(OUT + 32)).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
    vm.write_bytes(DATA, &2f64.to_le_bytes()).unwrap();
    for model in SAVE_POLICIES {
        let mut cpu = vm.new_vcpu();
        cpu.cpu.x87_pointer_model = model;
        cpu.set_reg(Reg::Rip, CODE);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        let mut out = [0; 60];
        vm.read_bytes(OUT, &mut out).unwrap();
        assert_eq!(cpu.cpu.fpu_sw & 0x80, 0);
        let expected = if model == X87PointerModel::Amd {
            [0x504u16, 0x1e8]
        } else {
            [0, 0]
        };
        for (base, opcode) in [0, 32].into_iter().zip(expected) {
            assert_eq!(
                u16::from_le_bytes(out[base + 18..base + 20].try_into().unwrap()),
                opcode
            );
        }
        assert_eq!(cpu.cpu.fpu_fop, expected[1]);
    }
}

fn image() -> [u8; 512] {
    let mut b = [0; 512];
    b[0..2].copy_from_slice(&0x037fu16.to_le_bytes());
    b[6..8].copy_from_slice(&0x0523u16.to_le_bytes());
    b[8..16].copy_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
    b[16..24].copy_from_slice(&0x2345_6789_abcd_ef01u64.to_le_bytes());
    b[24..28].copy_from_slice(&0x1f80u32.to_le_bytes());
    b
}

#[test]
fn fxstate_helper_retains_edited_pointers() {
    struct Image(RefCell<[u8; 512]>);
    impl FpMem for Image {
        fn load(&self, addr: u64, buf: &mut [u8]) -> bool {
            assert_eq!(addr, 0);
            buf.copy_from_slice(&*self.0.borrow());
            true
        }
        fn store(&self, addr: u64, buf: &[u8]) -> bool {
            assert_eq!(addr, 0);
            self.0.borrow_mut().copy_from_slice(buf);
            true
        }
    }
    let expected = image();
    let mem = Image(RefCell::new(expected));
    let mut cpu = CpuState::default();
    // SDM Vol 2A §3.3 CPUID, leaf 7 EBX[6,13]: ordinary FDP updates, live selectors.
    assert_eq!(cpu.cpuid(7, 0).ebx & ((1 << 6) | (1 << 13)), 0);
    assert_eq!(exec_fxstate(&mut cpu, &mem, 0, true), None);
    assert_eq!(exec_fxstate(&mut cpu, &mem, 0, false), None);
    assert_eq!(&mem.0.borrow()[6..24], &expected[6..24]);
}

// SDM Vol 1 §§8.1.8–8.1.9; AMD64 APM Vol 2 §11.4.4.3, Vol 5 FXSAVE:
// the AMD-style save policy tests ES, not merely a sticky exception flag.
const SAVE_POLICIES: [X87PointerModel; 2] = [X87PointerModel::Sdm, X87PointerModel::Amd];

#[test]
fn x87_pointer_model_policy_fxstate_export() {
    struct Image(RefCell<[u8; 512]>);
    impl FpMem for Image {
        fn load(&self, _: u64, buf: &mut [u8]) -> bool {
            buf.copy_from_slice(&*self.0.borrow());
            true
        }
        fn store(&self, _: u64, buf: &[u8]) -> bool {
            self.0.borrow_mut().copy_from_slice(buf);
            true
        }
    }
    for policy in SAVE_POLICIES {
        for sw in [0u16, 1, 0x8081] {
            let mut initial = image();
            initial[2..4].copy_from_slice(&sw.to_le_bytes());
            let mem = Image(RefCell::new(initial));
            let mut cpu = CpuState {
                x87_pointer_model: policy,
                ..CpuState::default()
            };
            assert_eq!(exec_fxstate(&mut cpu, &mem, 0, true), None);
            assert_eq!(cpu.fpu_fip, u64_at(&initial, 8));
            assert_eq!(cpu.fpu_fdp, u64_at(&initial, 16));
            assert_eq!(cpu.fpu_fop, 0x523);
            assert_eq!(exec_fxstate(&mut cpu, &mem, 0, false), None);
            let saved = mem.0.borrow();
            if policy == X87PointerModel::Amd && sw & 0x80 == 0 {
                assert_eq!(&saved[6..24], &[0; 18]);
            } else {
                assert_eq!(&saved[6..24], &initial[6..24]);
            }
            assert_eq!(&saved[..6], &initial[..6]);
            assert_eq!(&saved[24..28], &initial[24..28]);
        }
    }
}

fn save_policy_images(backend: Box<dyn Backend>) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x30_0000), backend);
    vm.map(0, 0x30_0000, Prot::RWX, RegionKind::Ram).unwrap();
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        vm.set_cpu_mode(mode);
        for policy in SAVE_POLICIES {
            vm.set_x87_pointer_model(policy);
            for sw in [0u16, 1, 0x8081] {
                // Imports are deliberately tested even with ES clear: the policy
                // filters exports without discarding restored state.
                for import in 0..3 {
                    if mode == CpuMode::Compat32 && import == 0 {
                        continue;
                    }
                    let mut a =
                        CodeAssembler::new(if mode == CpuMode::Long64 { 64 } else { 32 }).unwrap();
                    if import == 2 {
                        a.fldenv(ptr(DATA + 512)).unwrap();
                    } else {
                        fx(&mut a, true, import == 0, DATA);
                    }
                    fx(&mut a, false, mode == CpuMode::Long64, OUT);
                    fx(&mut a, false, false, OUT + 512);
                    a.fnstenv(ptr(OUT + 1024)).unwrap();
                    a.hlt().unwrap();
                    let mut initial = image().to_vec();
                    initial[2..4].copy_from_slice(&sw.to_le_bytes());
                    let fip = if import == 0 {
                        u64_at(&initial, 8)
                    } else {
                        u64_at(&initial, 8) as u32 as u64
                    };
                    let fdp = if import == 0 {
                        u64_at(&initial, 16)
                    } else {
                        u64_at(&initial, 16) as u32 as u64
                    };
                    let (fcs, fds) = if import == 0 {
                        (0u16, 0u16)
                    } else {
                        (0x5678, 0x6789)
                    };
                    let mut env = [0u8; 28];
                    env[..2].copy_from_slice(&0x037fu16.to_le_bytes());
                    env[4..6].copy_from_slice(&sw.to_le_bytes());
                    env[8..10].copy_from_slice(&0xffffu16.to_le_bytes());
                    env[12..16].copy_from_slice(&(fip as u32).to_le_bytes());
                    env[16..18].copy_from_slice(&fcs.to_le_bytes());
                    env[18..20].copy_from_slice(&0x0523u16.to_le_bytes());
                    env[20..24].copy_from_slice(&(fdp as u32).to_le_bytes());
                    env[24..26].copy_from_slice(&fds.to_le_bytes());
                    initial.extend_from_slice(&env);
                    vm.write_bytes(DATA, &initial).unwrap();
                    let code = a.assemble(CODE).unwrap();
                    vm.write_bytes(CODE, &code).unwrap();
                    let mut cpu = vm.new_vcpu();
                    cpu.set_reg(Reg::Rip, CODE);
                    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                    assert_eq!(
                        (cpu.cpu.fpu_fip, cpu.cpu.fpu_fdp, cpu.cpu.fpu_fop),
                        (fip, fdp, 0x523)
                    );
                    assert_eq!((cpu.cpu.fpu_fcs, cpu.cpu.fpu_fds), (fcs, fds));
                    let mut out = [0; 1536];
                    vm.read_bytes(OUT, &mut out).unwrap();
                    let visible = policy == X87PointerModel::Sdm || sw & 0x80 != 0;
                    for base in [0, 512] {
                        assert_eq!(&out[base + 2..base + 4], &sw.to_le_bytes());
                        assert_eq!(
                            &out[base + 6..base + 8],
                            &(if visible { 0x523u16 } else { 0 }).to_le_bytes()
                        );
                        if base == 0 && mode == CpuMode::Long64 {
                            assert_eq!(u64_at(&out, base + 8), if visible { fip } else { 0 });
                            assert_eq!(u64_at(&out, base + 16), if visible { fdp } else { 0 });
                        } else {
                            assert_eq!(
                                &out[base + 8..base + 12],
                                &(if visible { fip as u32 } else { 0 }).to_le_bytes()
                            );
                            assert_eq!(
                                &out[base + 16..base + 20],
                                &(if visible { fdp as u32 } else { 0 }).to_le_bytes()
                            );
                            assert_eq!(
                                &out[base + 12..base + 14],
                                &(if visible { fcs } else { 0 }).to_le_bytes()
                            );
                            assert_eq!(
                                &out[base + 20..base + 22],
                                &(if visible { fds } else { 0 }).to_le_bytes()
                            );
                        }
                    }
                    assert_eq!(&out[1028..1030], &sw.to_le_bytes());
                    // AMD64 APM Vol 5 FSTENV (FNSTENV): save the current
                    // environment regardless of ES, unlike FXSAVE.
                    assert_eq!(&out[1036..1050], &env[12..26]);
                }
            }
        }
    }
}

fn run(code: &[u8], data: &[u8], backend: Box<dyn Backend>) -> Vec<u8> {
    run_mode(code, data, backend, x86jit_core::CpuMode::Long64)
}
fn run_mode(
    code: &[u8],
    data: &[u8],
    backend: Box<dyn Backend>,
    mode: x86jit_core::CpuMode,
) -> Vec<u8> {
    run_config(code, data, backend, mode, |_| {})
}
fn run_config(
    code: &[u8],
    data: &[u8],
    backend: Box<dyn Backend>,
    mode: x86jit_core::CpuMode,
    configure: impl FnOnce(&mut CpuState),
) -> Vec<u8> {
    let mut vm = Vm::with_backend(VmConfig::flat(0x30_0000), backend);
    vm.set_cpu_mode(mode);
    vm.map(0, 0x30_0000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(CODE, code).unwrap();
    vm.write_bytes(DATA, data).unwrap();
    let mut cpu = vm.new_vcpu();
    if mode == x86jit_core::CpuMode::Compat32 {
        cpu.cpu.cs = 0x33;
        cpu.cpu.ds = 0x2b;
        cpu.cpu.ss = 0x2b;
    }
    configure(&mut cpu.cpu);
    cpu.set_reg(Reg::Rip, CODE);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let mut result = vec![0; 2048];
    vm.read_bytes(OUT, &mut result).unwrap();
    result
}
macro_rules! tiers {
    ($interp:ident, $jit:ident, $case:ident) => {
        #[test]
        fn $interp() {
            $case(Box::new(InterpreterBackend));
        }
        #[test]
        fn $jit() {
            $case(Box::new(x86jit_cranelift::JitBackend::new()));
        }
    };
}
tiers!(
    x87_pointer_model_policy_images_interp,
    x87_pointer_model_policy_images_jit,
    save_policy_images
);

fn save_policy_retains_recording(backend: Box<dyn Backend>) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x30_0000), backend);
    vm.map(0, 0x30_0000, Prot::RWX, RegionKind::Ram).unwrap();
    assert_eq!(vm.x87_pointer_model(), X87PointerModel::Sdm);
    let original = vm.new_vcpu();
    vm.set_x87_pointer_model(X87PointerModel::Amd);
    assert_eq!(original.cpu.x87_pointer_model, X87PointerModel::Sdm);
    let child = vm.fork_with_backend(Box::new(InterpreterBackend)).unwrap();
    assert_eq!(child.new_vcpu().cpu.x87_pointer_model, X87PointerModel::Amd);

    let mut a = CodeAssembler::new(64).unwrap();
    a.fld(qword_ptr(DATA)).unwrap();
    let save_ip = CODE + a.assemble(CODE).unwrap().len() as u64;
    fx(&mut a, false, true, OUT);
    a.hlt().unwrap();
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
    vm.write_bytes(DATA, &2f64.to_le_bytes()).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!((cpu.cpu.fpu_fip, cpu.cpu.fpu_fdp), (CODE, DATA));
    let mut out = [0; 512];
    vm.read_bytes(OUT, &mut out).unwrap();
    assert_eq!(&out[6..24], &[0; 18]);
    // The same recorded state becomes visible after either a policy change or ES.
    for policy in SAVE_POLICIES {
        cpu.cpu.x87_pointer_model = policy;
        cpu.cpu.fpu_sw = if policy == X87PointerModel::Sdm {
            0
        } else {
            0x80
        };
        cpu.set_reg(Reg::Rip, save_ip);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        vm.read_bytes(OUT, &mut out).unwrap();
        assert_eq!(u64_at(&out, 8), CODE);
        assert_eq!(u64_at(&out, 16), DATA);
    }
}
tiers!(
    x87_pointer_model_policy_recording_interp,
    x87_pointer_model_policy_recording_jit,
    save_policy_retains_recording
);

#[test]
fn x87_pointer_model_policy_real_environment() {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(InterpreterBackend));
    vm.set_cpu_mode(CpuMode::Real16);
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x100, &[0x66, 0xd9, 0x36, 0x00, 0x08, 0xf4])
        .unwrap();
    for policy in SAVE_POLICIES {
        vm.set_x87_pointer_model(policy);
        for sw in [0u16, 1, 0x8081] {
            let mut cpu = vm.new_vcpu();
            cpu.cpu.fpu_sw = sw;
            cpu.cpu.fpu_fip = 0x12345678;
            cpu.cpu.fpu_fdp = 0x23456789;
            cpu.cpu.fpu_fop = 0x523;
            cpu.set_reg(Reg::Rip, 0x100);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            let mut out = [0; 28];
            vm.read_bytes(0x800, &mut out).unwrap();
            // AMD64 APM Vol 5 FSTENV (FNSTENV): no ES condition.
            let expected = [0x5678u32, 0x01234523, 0x6789, 0x02345000];
            for (offset, value) in [12, 16, 20, 24].into_iter().zip(expected) {
                assert_eq!(&out[offset..offset + 4], &value.to_le_bytes());
            }
            assert_eq!(&out[4..6], &sw.to_le_bytes());
        }
    }
}
fn fx(asm: &mut CodeAssembler, restore: bool, wide: bool, addr: u64) {
    // SDM Vol 2A §3.3 FXSAVE/FXRSTOR: REX.W selects the 64-bit pointer image.
    if wide {
        if restore {
            asm.fxrstor64(ptr(addr)).unwrap();
        } else {
            asm.fxsave64(ptr(addr)).unwrap();
        }
    } else if restore {
        asm.fxrstor(ptr(addr)).unwrap();
    } else {
        asm.fxsave(ptr(addr)).unwrap();
    }
}

fn fx_images_round_trip(backend: Box<dyn Backend>) {
    {
        let mut a = CodeAssembler::new(64).unwrap();
        fx(&mut a, true, true, DATA);
        fx(&mut a, false, true, OUT);
        fx(&mut a, false, false, OUT + 512);
        fx(&mut a, true, false, DATA);
        fx(&mut a, false, true, OUT + 1024);
        fx(&mut a, false, false, OUT + 1536);
        a.hlt().unwrap();
        let initial = image();
        let out = run(&a.assemble(CODE).unwrap(), &initial, backend);
        assert_eq!(&out[6..24], &initial[6..24]);
        for base in [512, 1536] {
            assert_eq!(&out[base + 6..base + 12], &initial[6..12]);
            assert_eq!(&out[base + 16..base + 20], &initial[16..20]);
        }
        // REX.W restore clears selectors; legacy restore loads them.
        assert_eq!(&out[524..526], &[0; 2]);
        assert_eq!(&out[532..534], &[0; 2]);
        assert_eq!(&out[1548..1550], &initial[12..14]);
        assert_eq!(&out[1556..1558], &initial[20..22]);
        assert_eq!(&out[1030..1036], &initial[6..12]);
        assert_eq!(&out[1036..1040], &[0; 4]);
        assert_eq!(&out[1040..1044], &initial[16..20]);
        assert_eq!(&out[1044..1048], &[0; 4]);
    }
}

fn legacy_environment_round_trip(backend: Box<dyn Backend>) {
    let mut env = [0; 28];
    env[0..2].copy_from_slice(&0x037fu16.to_le_bytes());
    env[8..10].copy_from_slice(&0xffffu16.to_le_bytes());
    env[12..16].copy_from_slice(&0x9abcdef0u32.to_le_bytes());
    env[16..18].copy_from_slice(&0x1234u16.to_le_bytes());
    env[18..20].copy_from_slice(&0x0523u16.to_le_bytes());
    env[20..24].copy_from_slice(&0xabcdef01u32.to_le_bytes());
    env[24..26].copy_from_slice(&0x2345u16.to_le_bytes());
    {
        let mut a = CodeAssembler::new(64).unwrap();
        fx(&mut a, true, true, DATA + 512);
        a.fldenv(ptr(DATA)).unwrap();
        fx(&mut a, false, true, OUT);
        a.fnstenv(ptr(OUT + 512)).unwrap();
        a.hlt().unwrap();
        let mut data = env.to_vec();
        data.resize(512, 0);
        data.extend_from_slice(&image());
        let out = run(&a.assemble(CODE).unwrap(), &data, backend);
        assert_eq!(&out[6..8], &env[18..20]);
        assert_eq!(&out[8..12], &env[12..16]);
        assert_eq!(&out[12..16], &[0; 4]);
        assert_eq!(&out[16..20], &env[20..24]);
        assert_eq!(&out[20..24], &[0; 4]);
        assert_eq!(&out[524..538], &env[12..26]);
    }
}

fn instruction_code() -> (Vec<u8>, u64, u64) {
    let mut a = CodeAssembler::new(64).unwrap();
    a.fld(qword_ptr(DATA)).unwrap();
    let register_ip = CODE + a.assemble(CODE).unwrap().len() as u64;
    a.fabs().unwrap();
    fx(&mut a, false, true, OUT);
    a.fldcw(word_ptr(DATA + 8)).unwrap();
    a.fnstcw(word_ptr(DATA + 10)).unwrap();
    a.fnstsw(ax).unwrap();
    a.fnclex().unwrap();
    a.wait().unwrap();
    fx(&mut a, false, true, OUT + 512);
    let nop_ip = CODE + a.assemble(CODE).unwrap().len() as u64;
    a.fnop().unwrap();
    fx(&mut a, false, true, OUT + 1024);
    a.fninit().unwrap();
    fx(&mut a, false, true, OUT + 1536);
    a.hlt().unwrap();
    (a.assemble(CODE).unwrap(), register_ip, nop_ip)
}
fn instruction_data() -> [u8; 12] {
    let mut b = [0; 12];
    b[..8].copy_from_slice(&(-2f64).to_le_bytes());
    b[8..10].copy_from_slice(&0x037fu16.to_le_bytes());
    b
}
fn u64_at(out: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(out[off..off + 8].try_into().unwrap())
}
fn instruction_updates(backend: Box<dyn Backend>) {
    let (code, register_ip, nop_ip) = instruction_code();
    {
        let out = run(&code, &instruction_data(), backend);
        for base in [0, 512] {
            assert_eq!(u64_at(&out, base + 8), register_ip);
            assert_eq!(u64_at(&out, base + 16), DATA);
            // SDM Vol 1 §8.1.9: masked operations do not update FOP.
            assert_eq!(&out[base + 6..base + 8], &[0; 2]);
        }
        assert_eq!(u64_at(&out, 1032), nop_ip);
        assert_eq!(&out[1542..1560], &[0; 18]);
    }
}

fn unmasked_exception_records_opcode(backend: Box<dyn Backend>) {
    {
        let mut a = CodeAssembler::new(64).unwrap();
        a.fldcw(word_ptr(DATA)).unwrap();
        a.fld1().unwrap();
        a.fldz().unwrap();
        let ip = CODE + a.assemble(CODE).unwrap().len() as u64;
        a.fdivp(st1, st0).unwrap();
        // Non-waiting save observes the raising instruction before deferred #MF.
        fx(&mut a, false, true, OUT);
        a.fnstenv(ptr(OUT + 512)).unwrap();
        a.fnclex().unwrap();
        a.fldcw(word_ptr(DATA)).unwrap();
        a.fld1().unwrap();
        let mem_ip = CODE + a.assemble(CODE).unwrap().len() as u64;
        a.fdiv(qword_ptr(DATA + 8)).unwrap();
        fx(&mut a, false, true, OUT + 1024);
        a.fnstenv(ptr(OUT + 1536)).unwrap();
        a.hlt().unwrap();
        let mut data = [0; 16];
        data[..2].copy_from_slice(&0x037bu16.to_le_bytes());
        let mut code = a.assemble(CODE).unwrap();
        code.insert((mem_ip - CODE) as usize, 0x66);
        let out = run(&code, &data, backend);
        assert_eq!(u64_at(&out, 8), ip);
        assert_eq!(&out[6..8], &0x06f9u16.to_le_bytes());
        assert_eq!(&out[530..532], &0x06f9u16.to_le_bytes());
        assert_eq!(u64_at(&out, 1032), mem_ip);
        assert_eq!(u64_at(&out, 1040), DATA + 8);
        assert_eq!(&out[1030..1032], &0x0434u16.to_le_bytes());
        assert_eq!(&out[1554..1556], &0x0434u16.to_le_bytes());
    }
}

tiers!(
    x87_pointer_fx_images_round_trip_interp,
    x87_pointer_fx_images_round_trip_jit,
    fx_images_round_trip
);

tiers!(
    x87_pointer_legacy_environment_round_trip_interp,
    x87_pointer_legacy_environment_round_trip_jit,
    legacy_environment_round_trip
);

tiers!(
    x87_pointer_instruction_updates_interp,
    x87_pointer_instruction_updates_jit,
    instruction_updates
);

tiers!(
    x87_pointer_unmasked_exception_records_opcode_interp,
    x87_pointer_unmasked_exception_records_opcode_jit,
    unmasked_exception_records_opcode
);

fn prefix_and_control(backend: Box<dyn Backend>) {
    let mut a = CodeAssembler::new(64).unwrap();
    fx(&mut a, true, true, DATA);
    a.fldcw(word_ptr(DATA)).unwrap();
    a.fnstcw(word_ptr(DATA + 512)).unwrap();
    a.fnstsw(ax).unwrap();
    a.fnclex().unwrap();
    a.wait().unwrap();
    a.fnstenv(ptr(DATA + 512)).unwrap();
    fx(&mut a, false, true, OUT);
    let mut code = a.assemble(CODE).unwrap();
    let ip = CODE + code.len() as u64;
    // Redundant operand-size prefix: FIP includes it; FOP excludes it.
    code.extend_from_slice(&[0x66, 0xd9, 0xe8]);
    let mut tail = CodeAssembler::new(64).unwrap();
    fx(&mut tail, false, true, OUT + 512);
    tail.hlt().unwrap();
    code.extend(tail.assemble(CODE + code.len() as u64).unwrap());
    let initial = image();
    let out = run(&code, &initial, backend);
    assert_eq!(&out[6..24], &initial[6..24]);
    assert_eq!(u64_at(&out, 520), ip);
    assert_eq!(&out[518..520], &initial[6..8]);
    assert_eq!(&out[528..536], &initial[16..24]);
}
tiers!(
    x87_pointer_prefix_and_control_interp,
    x87_pointer_prefix_and_control_jit,
    prefix_and_control
);

fn compat_selectors(backend: Box<dyn Backend>) {
    let mut a = CodeAssembler::new(32).unwrap();
    a.fld(qword_ptr(DATA)).unwrap();
    a.fxsave(ptr(OUT)).unwrap();
    a.fnstenv(ptr(OUT + 512)).unwrap();
    a.hlt().unwrap();
    let out = run_mode(
        &a.assemble(CODE).unwrap(),
        &2f64.to_le_bytes(),
        backend,
        x86jit_core::CpuMode::Compat32,
    );
    assert_eq!(
        u32::from_le_bytes(out[8..12].try_into().unwrap()),
        CODE as u32
    );
    assert_eq!(
        u32::from_le_bytes(out[16..20].try_into().unwrap()),
        DATA as u32
    );
    assert_eq!(
        u32::from_le_bytes(out[524..528].try_into().unwrap()),
        CODE as u32
    );
    assert_eq!(
        u32::from_le_bytes(out[532..536].try_into().unwrap()),
        DATA as u32
    );
    assert_eq!(&out[12..14], &0x33u16.to_le_bytes());
    assert_eq!(&out[20..22], &0x2bu16.to_le_bytes());
    assert_eq!(&out[528..530], &0x33u16.to_le_bytes());
    assert_eq!(&out[536..538], &0x2bu16.to_le_bytes());
}
tiers!(
    x87_pointer_compat_interp,
    x87_pointer_compat_jit,
    compat_selectors
);

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn native_output(code: Vec<u8>, data: Vec<u8>) -> Vec<u8> {
    use x86jit_tests::{
        oracle::VectorInput,
        vector::{CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec},
    };
    let input = VectorInput {
        cpu_init: CpuSnapshot::default(),
        mem_init: vec![
            MemChunk {
                addr: CODE,
                bytes: code,
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: DATA,
                bytes: data,
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: OUT,
                bytes: vec![0; 2048],
                kind: MemKind::Ram,
            },
        ],
        entry: CODE,
        run: RunSpec::UntilExit,
    };
    let out = x86jit_tests::native::run_native(&input).expect("native pointer witness");
    assert_eq!(out.exit, ExitKind::Hlt);
    out.mem[2].bytes.clone()
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn host_is_amd() -> bool {
    // AMD64 APM Vol 3 CPUID, Table 3-2: vendor string in EBX, EDX, ECX.
    // SAFETY: CPUID leaf 0 is available on every x86-64 host.
    #[allow(unused_unsafe)] // CPUID intrinsics are unsafe on the workspace MSRV.
    let vendor = unsafe { std::arch::x86_64::__cpuid(0) };
    (vendor.ebx, vendor.edx, vendor.ecx)
        == (
            u32::from_le_bytes(*b"Auth"),
            u32::from_le_bytes(*b"enti"),
            u32::from_le_bytes(*b"cAMD"),
        )
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn native_save_policy(backend: Box<dyn Backend>) {
    if !host_is_amd() {
        return;
    }
    let mut a = CodeAssembler::new(64).unwrap();
    a.fld(qword_ptr(DATA)).unwrap();
    fx(&mut a, false, true, OUT);
    fx(&mut a, false, false, OUT + 512);
    a.fnstenv(ptr(OUT + 480)).unwrap();
    a.fldcw(word_ptr(DATA + 8)).unwrap();
    a.fdiv(qword_ptr(DATA + 16)).unwrap();
    fx(&mut a, false, true, OUT + 1024);
    fx(&mut a, false, false, OUT + 1536);
    a.fnstenv(ptr(OUT + 1504)).unwrap();
    a.hlt().unwrap();
    let code = a.assemble(CODE).unwrap();
    let mut data = [0; 24];
    data[..8].copy_from_slice(&2f64.to_le_bytes());
    data[8..10].copy_from_slice(&0x037bu16.to_le_bytes());
    // On AMD the state save and restore a kernel uses across a task switch do
    // not carry the x87 pointer registers while ES is clear (APM Vol 2
    // §11.4.4.3), so a capture preempted between the FLD and an FNSTENV can
    // read zeros there. Preemption can only lose the pointers, never invent
    // them: keep the first capture whose two FNSTENV images both carry a
    // nonzero instruction pointer.
    let native = (0..32)
        .map(|_| native_output(code.clone(), data.to_vec()))
        .find(|out| {
            [480, 1504]
                .iter()
                .all(|&base| out[base + 12..base + 16].iter().any(|&b| b != 0))
        })
        .expect("a native capture that was not preempted inside the sequence");
    let ours = run_config(&code, &data, backend, CpuMode::Long64, |cpu| {
        // SDM Vol 1 §8.1.8: live selectors are saved when EBX[13] is clear.
        // Zen 4 CI measured native FCS=0x33; match its user CS rather than
        // the vCPU's default CS=0, retaining exact selector comparisons below.
        assert_eq!(cpu.cpuid(7, 0).ebx & (1 << 13), 0);
        cpu.cs = 0x33;
        cpu.x87_pointer_model = X87PointerModel::Amd;
    });
    // APM Vol 2 §11.4.4.3; Vol 5 FXSAVE: exact pointer/opcode comparisons
    // on AMD for ES clear and set, without the default policy's exclusions.
    // Vol 5 FSTENV (FNSTENV) saves the environment regardless of ES;
    // Zen 4 measured nonzero FIP/FCS/FOP/FDP with ES clear.
    for base in [0, 512, 1024, 1536] {
        assert_eq!(&ours[base + 2..base + 4], &native[base + 2..base + 4]);
        assert_eq!(&ours[base + 6..base + 24], &native[base + 6..base + 24]);
    }
    for base in [480, 1504] {
        assert_eq!(&ours[base + 4..base + 6], &native[base + 4..base + 6]);
        assert_eq!(&ours[base + 12..base + 26], &native[base + 12..base + 26]);
    }
}
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
tiers!(
    x87_pointer_model_policy_native_interp,
    x87_pointer_model_policy_native_jit,
    native_save_policy
);

// AMD64 APM Vol 2 §11.4.4.3; Vol 5 FXSAVE/FXRSTOR: error pointers are
// saved/restored only with ES=1. Ryzen 7 7840HS (Zen 4) measured FOP/FIP/FDP
// as zero with ES=0, including after FXRSTOR of nonzero edited fields.
// Exclude those native fields on AMD only; engine witnesses remain unconditional
// (SDM Vol 1 §§8.1.8–8.1.9; Vol 2A §3.3 FXSAVE).
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn x87_pointer_native_instruction_witness() {
    let (code, ip, nop_ip) = instruction_code();
    let native = native_output(code, instruction_data().to_vec());
    // Register-form FDP is undefined (SDM Vol 1 §8.1.8), so only compare FIP.
    // Report the actual image if a vendor's pointer-saving policy differs.
    for (base, expected) in [(0, ip), (512, ip), (1024, nop_ip), (1536, 0)] {
        if !host_is_amd() || base == 1536 {
            assert_eq!(
                u64_at(&native, base + 8),
                expected,
                "native environment={:02x?}",
                &native[base + 6..base + 24]
            );
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn x87_pointer_native_edited_image_witness() {
    let mut a = CodeAssembler::new(64).unwrap();
    fx(&mut a, true, true, DATA);
    fx(&mut a, false, true, OUT);
    a.hlt().unwrap();
    let initial = image();
    let native = native_output(a.assemble(CODE).unwrap(), initial.to_vec());
    assert_eq!(&native[..6], &initial[..6]);
    assert_eq!(&native[24..28], &initial[24..28]);
    if !host_is_amd() {
        // ES=0: see the measured AMD divergence above the instruction witness.
        assert_eq!(&native[6..24], &initial[6..24]);
    }
}

fn fault_retains_environment(backend: Box<dyn Backend>) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x30_0000), backend);
    for addr in [CODE, DATA, OUT] {
        vm.map(addr, 4096, Prot::RWX, RegionKind::Ram).unwrap();
    }
    let initial = image();
    vm.write_bytes(DATA, &initial).unwrap();
    for op in [[0xdd, 0x00], [0xdd, 0x18]] {
        let mut a = CodeAssembler::new(64).unwrap();
        fx(&mut a, true, true, DATA);
        let mut code = a.assemble(CODE).unwrap();
        let fault_ip = CODE + code.len() as u64;
        code.extend_from_slice(&op);
        let resume_ip = CODE + code.len() as u64;
        let mut tail = CodeAssembler::new(64).unwrap();
        fx(&mut tail, false, true, OUT);
        tail.hlt().unwrap();
        code.extend(tail.assemble(resume_ip).unwrap());
        vm.write_bytes(CODE, &code).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rax, 0x40_0000);
        assert!(matches!(
            cpu.run(&vm, None),
            Exit::UnmappedMemory {
                addr: 0x40_0000,
                ..
            }
        ));
        assert_eq!(cpu.reg(Reg::Rip), fault_ip);
        cpu.set_reg(Reg::Rip, resume_ip);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        let mut out = [0; 512];
        vm.read_bytes(OUT, &mut out).unwrap();
        // SDM Vol 3A §7.5: a memory fault leaves instruction-entry state intact.
        assert_eq!(&out[6..24], &initial[6..24]);
    }
}
tiers!(
    x87_pointer_fault_retains_environment_interp,
    x87_pointer_fault_retains_environment_jit,
    fault_retains_environment
);

fn segmented_operand(backend: Box<dyn Backend>) {
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(rax, DATA - 0x1000).unwrap();
    let mut code = a.assemble(CODE).unwrap();
    let ip = CODE + code.len() as u64;
    code.extend_from_slice(&[0x64, 0xdd, 0x00]); // fld qword fs:[rax]
    let mut tail = CodeAssembler::new(64).unwrap();
    fx(&mut tail, false, false, OUT);
    fx(&mut tail, false, true, OUT + 512);
    tail.hlt().unwrap();
    code.extend(tail.assemble(CODE + code.len() as u64).unwrap());
    let out = run_config(
        &code,
        &2f64.to_le_bytes(),
        backend,
        x86jit_core::CpuMode::Long64,
        |cpu| {
            cpu.fs_base = 0x1000;
            cpu.fs = 0x53;
            cpu.cs = 0x33;
        },
    );
    // SDM Vol 1 §8.1.8; AMD64 APM Vol 1 §6.2.5: offsets and selectors.
    assert_eq!(
        u32::from_le_bytes(out[8..12].try_into().unwrap()),
        ip as u32
    );
    assert_eq!(
        u32::from_le_bytes(out[16..20].try_into().unwrap()),
        (DATA - 0x1000) as u32
    );
    assert_eq!(&out[12..14], &0x33u16.to_le_bytes());
    assert_eq!(&out[20..22], &0x53u16.to_le_bytes());
    assert_eq!(u64_at(&out, 520), ip);
    assert_eq!(u64_at(&out, 528), DATA - 0x1000);
}
tiers!(
    x87_pointer_segmented_operand_interp,
    x87_pointer_segmented_operand_jit,
    segmented_operand
);

fn cpuid_pointer_policy(backend: Box<dyn Backend>) {
    use x86jit_core::features::{CpuidOverride, CpuidRegisters};
    let mut a = CodeAssembler::new(64).unwrap();
    fx(&mut a, true, false, DATA);
    let ip = CODE + a.assemble(CODE).unwrap().len() as u64;
    a.fld(qword_ptr(DATA + 512)).unwrap();
    fx(&mut a, false, false, OUT);
    a.fnstenv(ptr(OUT + 512)).unwrap();
    a.hlt().unwrap();
    let mut data = image().to_vec();
    data.extend_from_slice(&2f64.to_le_bytes());
    let out = run_config(
        &a.assemble(CODE).unwrap(),
        &data,
        backend,
        x86jit_core::CpuMode::Long64,
        |cpu| {
            let bits = (1 << 6) | (1 << 13);
            cpu.cpuid_overrides.leaves.insert(
                (7, 0),
                CpuidOverride {
                    mask: CpuidRegisters::new(0, bits, 0, 0),
                    value: CpuidRegisters::new(0, bits, 0, 0),
                },
            );
        },
    );
    // SDM Vol 1 §8.1.8: exception-only FDP; deprecated selectors save as zero.
    assert_eq!(
        u32::from_le_bytes(out[8..12].try_into().unwrap()),
        ip as u32
    );
    assert_eq!(&out[16..20], &data[16..20]);
    assert_eq!(&out[12..14], &[0; 2]);
    assert_eq!(&out[20..22], &[0; 2]);
    assert_eq!(&out[528..530], &[0; 2]);
    assert_eq!(&out[536..538], &[0; 2]);
    assert_eq!(&out[532..536], &data[16..20]);
}
tiers!(
    x87_pointer_cpuid_policy_interp,
    x87_pointer_cpuid_policy_jit,
    cpuid_pointer_policy
);

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn x87_pointer_native_memory_and_exception_witness() {
    let mut a = CodeAssembler::new(64).unwrap();
    a.fld(qword_ptr(DATA)).unwrap();
    fx(&mut a, false, true, OUT);
    a.fldcw(word_ptr(DATA + 8)).unwrap();
    a.fld1().unwrap();
    a.fldz().unwrap();
    let ip = CODE + a.assemble(CODE).unwrap().len() as u64;
    a.fdivp(st1, st0).unwrap();
    fx(&mut a, false, true, OUT + 512);
    a.fnstenv(ptr(OUT + 1024)).unwrap();
    a.hlt().unwrap();
    let mut data = 2f64.to_le_bytes().to_vec();
    data.extend_from_slice(&0x037bu16.to_le_bytes());
    let out = native_output(a.assemble(CODE).unwrap(), data);
    if !host_is_amd() {
        // ES=0: see the measured AMD divergence above the instruction witness.
        assert_eq!(
            u64_at(&out, 8),
            CODE,
            "native memory environment={:02x?}",
            &out[6..24]
        );
        assert_eq!(u64_at(&out, 16), DATA);
    }
    // APM Vol 2 §11.4.4.3; Vol 5 FXSAVE: ES=1 must save the error pointers
    // on AMD too. Keep this native witness even if a host returns zero pointers.
    assert_eq!(
        u16::from_le_bytes(out[514..516].try_into().unwrap()) & 0x84,
        0x84
    );
    assert_eq!(u64_at(&out, 520), ip);
    assert_eq!(&out[518..520], &0x06f9u16.to_le_bytes());
}

// SDM Vol 1 §8.1.8: owned pointers are offsets plus selectors, even when
// real-mode memory accesses add a nonzero segment base.
#[test]
fn real16_fx_pointers_keep_segment_offsets() {
    for (prefix, selector, base) in [
        (None, 0x2345, 0x23450),
        (Some(0x36), 0x3456, 0x34560),
        (Some(0x26), 0x4567, 0x45670),
        (Some(0x2e), 0x1234, 0x12340),
    ] {
        let mut vm = Vm::with_backend(VmConfig::flat(0x100000), Box::new(InterpreterBackend));
        vm.set_cpu_mode(x86jit_core::CpuMode::Real16);
        vm.map(0, 0x100000, Prot::RWX, RegionKind::Ram).unwrap();
        let mut code = prefix.into_iter().collect::<Vec<_>>();
        code.extend_from_slice(&[0xdd, 0x06, 0x00, 0x02, 0x0f, 0xae, 0x06, 0x00, 0x04, 0xf4]);
        vm.write_bytes(0x12440, &code).unwrap();
        vm.write_bytes(base + 0x200, &1f64.to_le_bytes()).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.cpu.cs = 0x1234;
        cpu.cpu.ds = 0x2345;
        cpu.cpu.ss = 0x3456;
        cpu.cpu.es = 0x4567;
        cpu.set_reg(Reg::Rip, 0x100);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.cpu.fpu_fip, 0x100);
        assert_eq!(cpu.cpu.fpu_fcs, 0x1234);
        assert_eq!(cpu.cpu.fpu_fdp, 0x200);
        assert_eq!(cpu.cpu.fpu_fds, selector);
        let mut out = [0; 512];
        vm.read_bytes(0x23850, &mut out).unwrap();
        assert_eq!(&out[8..12], &0x100u32.to_le_bytes());
        assert_eq!(&out[12..14], &0x1234u16.to_le_bytes());
        assert_eq!(&out[16..20], &0x200u32.to_le_bytes());
        assert_eq!(&out[20..22], &selector.to_le_bytes());
    }
}

#[test]
fn real16_env28_packs_and_restores_linear_pointers() {
    // SDM Vol 1 §8.1.10, Fig. 8-10: 32-bit real-address environment.
    // The high pointer bits occupy bits 27:12 of the following dword.
    let mut vm = Vm::with_backend(VmConfig::flat(0x100000), Box::new(InterpreterBackend));
    vm.set_cpu_mode(x86jit_core::CpuMode::Real16);
    vm.map(0, 0x100000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(
        0x12440,
        &[
            0xdd, 0x06, 0x00, 0x02, // FLD DS:[0200]
            0x66, 0xd9, 0x36, 0x00, 0x04, // FNSTENV DS:[0400], 28 bytes
            0xdb, 0xe3, // FNINIT
            0x66, 0xd9, 0x26, 0x00, 0x06, // FLDENV DS:[0600], 28 bytes
            0x66, 0xd9, 0x36, 0x00, 0x08, // FNSTENV DS:[0800], 28 bytes
            0xf4,
        ],
    )
    .unwrap();
    vm.write_bytes(0x23650, &1f64.to_le_bytes()).unwrap();
    let mut initial = [0; 28];
    initial[..2].copy_from_slice(&0x037fu16.to_le_bytes());
    initial[8..10].copy_from_slice(&0xffffu16.to_le_bytes());
    initial[12..16].copy_from_slice(&0xdef0u32.to_le_bytes());
    initial[16..20].copy_from_slice(&0x09abc523u32.to_le_bytes());
    initial[20..24].copy_from_slice(&0x5678u32.to_le_bytes());
    initial[24..28].copy_from_slice(&0x01234000u32.to_le_bytes());
    vm.write_bytes(0x23a50, &initial).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.cs = 0x1234;
    cpu.cpu.ds = 0x2345;
    cpu.set_reg(Reg::Rip, 0x100);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    let mut out = [0; 28];
    vm.read_bytes(0x23850, &mut out).unwrap();
    assert_eq!(&out[12..16], &0x2440u32.to_le_bytes());
    assert_eq!(&out[16..20], &0x1000u32.to_le_bytes());
    assert_eq!(&out[20..24], &0x3650u32.to_le_bytes());
    assert_eq!(&out[24..28], &0x2000u32.to_le_bytes());
    assert_eq!(cpu.cpu.fpu_fip, 0x9abcdef0);
    assert_eq!(cpu.cpu.fpu_fdp, 0x12345678);
    assert_eq!(cpu.cpu.fpu_fop, 0x523);
    // SDM Vol 1 §8.1.8: real-address FLDENV clears the selectors.
    assert_eq!((cpu.cpu.fpu_fcs, cpu.cpu.fpu_fds), (0, 0));
    vm.read_bytes(0x23c50, &mut out).unwrap();
    assert_eq!(&out[12..28], &initial[12..28]);
}

fn opcode_requires_a_new_unmasked_exception(backend: Box<dyn Backend>) {
    // SDM Vol 1 §§8.1.9, 8.7.1: FOP changes for an instruction that incurs
    // an unmasked exception. Restored sticky flags alone do not constitute one.
    let mut initial = image();
    initial[0..2].copy_from_slice(&0x037eu16.to_le_bytes()); // unmask invalid
    initial[2..4].copy_from_slice(&1u16.to_le_bytes()); // sticky IE, ES clear
    initial[4] = 1; // occupied ST(0), containing zero
    let mut a = CodeAssembler::new(64).unwrap();
    fx(&mut a, true, true, DATA);
    a.fabs().unwrap(); // valid zero operand: no exception raised
    fx(&mut a, false, true, OUT);
    a.hlt().unwrap();
    let out = run(&a.assemble(CODE).unwrap(), &initial, backend);
    assert_eq!(&out[6..8], &initial[6..8]);
    assert_eq!(u16::from_le_bytes(out[2..4].try_into().unwrap()) & 0x81, 1);
}
tiers!(
    x87_pointer_sticky_flags_preserve_opcode_interp,
    x87_pointer_sticky_flags_preserve_opcode_jit,
    opcode_requires_a_new_unmasked_exception
);

tiers!(
    x87_opcode_models_interp,
    x87_opcode_models_jit,
    opcode_models
);
