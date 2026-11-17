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

/// Image FIP values, each with the FIP that a REX.W restore loads from it. The engine's
/// linear-address width is 48 bits, so bits 63:48 copy bit 47 (SDM Vol 3A §4.5 p. 4-4;
/// §4.5.3 p. 4-6). The third is the FIP of [`image`].
const FIP_LOADS: [(u64, u64); 4] = [
    (0xffff_8765_4321_0fed, 0xffff_8765_4321_0fed), // upper half: kept
    (0x0000_7fff_ffff_fff0, 0x0000_7fff_ffff_fff0), // lower half: kept
    (0x1234_5678_9abc_def0, 0x0000_5678_9abc_def0), // not canonical, bit 47 clear
    (0x0123_8765_4321_0fed, 0xffff_8765_4321_0fed), // not canonical, bit 47 set
];

fn image_with_fip(fip: u64) -> [u8; 512] {
    let mut b = image();
    b[8..16].copy_from_slice(&fip.to_le_bytes());
    b
}

/// [`image`] as a REX.W restore loads it: FIP canonical, FDP unchanged.
fn loaded_image() -> [u8; 512] {
    image_with_fip(FIP_LOADS[2].1)
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
    // The helper restores as FXRSTOR64 does: FOP and FDP as edited, FIP canonical
    // (SDM Vol 3A §4.5.3 p. 4-6).
    assert_eq!(&mem.0.borrow()[6..24], &loaded_image()[6..24]);
    for (fip, loaded) in FIP_LOADS {
        *mem.0.borrow_mut() = image_with_fip(fip);
        assert_eq!(exec_fxstate(&mut cpu, &mem, 0, true), None);
        assert_eq!(cpu.fpu_fip, loaded, "image FIP {fip:#x}");
        assert_eq!(cpu.fpu_fdp, u64_at(&expected, 16));
    }
    // A direct write to the field is kept as written, and a save stores it.
    cpu.fpu_fip = FIP_LOADS[2].0;
    assert_eq!(exec_fxstate(&mut cpu, &mem, 0, false), None);
    assert_eq!(u64_at(&mem.0.borrow()[..], 8), FIP_LOADS[2].0);
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
            assert_eq!(cpu.fpu_fip, FIP_LOADS[2].1);
            assert_eq!(cpu.fpu_fdp, u64_at(&initial, 16));
            assert_eq!(cpu.fpu_fop, 0x523);
            assert_eq!(exec_fxstate(&mut cpu, &mem, 0, false), None);
            let saved = mem.0.borrow();
            if policy == X87PointerModel::Amd && sw & 0x80 == 0 {
                assert_eq!(&saved[6..24], &[0; 18]);
            } else {
                assert_eq!(&saved[6..24], &loaded_image()[6..24]);
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
                        FIP_LOADS[2].1
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
        // FXRSTOR64 loads FIP canonical (SDM Vol 3A §4.5.3 p. 4-6); bits 31:0 survive.
        assert_eq!(&out[6..24], &loaded_image()[6..24]);
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

// Every FXRSTOR and XRSTOR image form. With REX.W = 1, FIP is loaded canonical at
// the engine's 48 bits (SDM Vol 3A §4.5.3 p. 4-6) and FDP loads all 64 bits. With
// REX.W = 0 both load bits 31:0 and clear the upper 32 (Vol 1 §8.1.8 p. 8-10).
fn restore_forms_load_fip_canonical(backend: Box<dyn Backend>) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x30_0000), backend);
    vm.map(0, 0x30_0000, Prot::RWX, RegionKind::Ram).unwrap();
    // (form, REX.W, XCOMP_BV of an XSAVE image or None for an FXSAVE image)
    let forms = [
        ("FXRSTOR64", true, None),
        ("XRSTOR64, standard", true, Some(0)),
        ("XRSTOR64, compacted", true, Some((1u64 << 63) | 1)),
        ("FXRSTOR", false, None),
        ("XRSTOR, standard", false, Some(0)),
    ];
    for (i, (form, wide, xcomp)) in forms.into_iter().enumerate() {
        let at = CODE + i as u64 * 0x100;
        let mut a = CodeAssembler::new(64).unwrap();
        // EDX:EAX = 1 requests the x87 component only (SDM Vol 1 §13.8).
        a.mov(eax, 1).unwrap();
        a.xor(edx, edx).unwrap();
        match (xcomp, wide) {
            (None, _) => fx(&mut a, true, wide, DATA),
            (Some(_), true) => a.xrstor64(ptr(DATA)).unwrap(),
            (Some(_), false) => a.xrstor(ptr(DATA)).unwrap(),
        }
        fx(&mut a, false, true, OUT);
        a.hlt().unwrap();
        vm.write_bytes(at, &a.assemble(at).unwrap()).unwrap();
        for (fip, loaded) in FIP_LOADS {
            let mut area = image_with_fip(fip).to_vec();
            if let Some(xcomp) = xcomp {
                // SDM Vol 1 §13.4.2: XSTATE_BV, then XCOMP_BV; bit 0 is the x87 state.
                let mut header = [0u8; 64];
                header[..8].copy_from_slice(&1u64.to_le_bytes());
                header[8..16].copy_from_slice(&xcomp.to_le_bytes());
                area.extend_from_slice(&header);
            }
            vm.write_bytes(DATA, &area).unwrap();
            let fdp = u64_at(&area, 16);
            let expected = if wide {
                (loaded, fdp)
            } else {
                (fip as u32 as u64, fdp as u32 as u64)
            };
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, at);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt), "{form}");
            assert_eq!(
                (cpu.cpu.fpu_fip, cpu.cpu.fpu_fdp),
                expected,
                "{form}: image FIP {fip:#x}"
            );
            let mut out = [0; 512];
            vm.read_bytes(OUT, &mut out).unwrap();
            assert_eq!(
                (u64_at(&out, 8), u64_at(&out, 16)),
                expected,
                "{form}: FXSAVE64 after image FIP {fip:#x}"
            );
        }
    }
}
tiers!(
    x87_pointer_restore_forms_load_fip_canonical_interp,
    x87_pointer_restore_forms_load_fip_canonical_jit,
    restore_forms_load_fip_canonical
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
    assert_eq!(&out[6..24], &loaded_image()[6..24]);
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

// AMD64 APM Vol 2 §11.4.4.3 p. 354; Vol 5 FXSAVE p. 26 and FXRSTOR p. 25: error
// pointers are saved/restored only with ES=1. Ryzen 7 7840HS (Zen 4) measured
// FOP/FIP/FDP as zero with ES=0, including after FXRSTOR of nonzero edited fields.
// Intel documents two more host differences (SDM Vol 1 §8.1.8 p. 8-9): with
// CPUID.(EAX=07H,ECX=0):EBX[6] = 1 only an unmasked exception updates FDP, and with
// EBX[13] = 1 FCS and FDS are saved as 0000H. A native witness compares a pointer
// field only where `native::host_x87_pointers` measured that this host stores it, and
// prints why it skipped one. Engine witnesses remain unconditional (SDM Vol 1
// §§8.1.8–8.1.9; Vol 2A §3.3 FXSAVE).
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn x87_pointer_native_instruction_witness() {
    use x86jit_tests::native::{host_x87_pointers, HostX87Pointers};
    let host = host_x87_pointers();
    let (code, ip, nop_ip) = instruction_code();
    let native = native_output(code, instruction_data().to_vec());
    // Register-form FDP is undefined (SDM Vol 1 §8.1.8), so only compare FIP.
    // Report the actual image if a vendor's pointer-saving policy differs.
    for (base, expected) in [(0, ip), (512, ip), (1024, nop_ip), (1536, 0)] {
        if host.fxsave_pointers_without_es || base == 1536 {
            assert_eq!(
                u64_at(&native, base + 8),
                expected,
                "native environment={:02x?}",
                &native[base + 6..base + 24]
            );
        } else {
            eprintln!(
                "FXSAVE64 image at +{base}: FIP not compared: {}",
                HostX87Pointers::NO_FXSAVE_POINTERS
            );
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn x87_pointer_native_edited_image_witness() {
    use x86jit_tests::native::{host_x87_pointers, HostX87Pointers};
    let host = host_x87_pointers();
    // One FXRSTOR64/FXSAVE64 pair per FIP_LOADS image, 512 bytes apart.
    let mut a = CodeAssembler::new(64).unwrap();
    let mut data = Vec::new();
    for (i, (fip, _)) in FIP_LOADS.into_iter().enumerate() {
        fx(&mut a, true, true, DATA + i as u64 * 512);
        fx(&mut a, false, true, OUT + i as u64 * 512);
        data.extend_from_slice(&image_with_fip(fip));
    }
    a.hlt().unwrap();
    let code = a.assemble(CODE).unwrap();
    let native = native_output(code.clone(), data.clone());
    // The engine loads FIP at its own 48 bits on every tier, whatever the host does.
    let ours = run(&code, &data, Box::new(InterpreterBackend));
    let jit = run(&code, &data, Box::new(x86jit_cranelift::JitBackend::new()));
    assert_eq!(ours, jit);
    for (i, (_, loaded)) in FIP_LOADS.into_iter().enumerate() {
        let base = i * 512;
        assert_eq!(u64_at(&ours, base + 8), loaded);
        assert_eq!(&ours[base + 16..base + 24], &image()[16..24]);
        assert_eq!(&native[base..base + 6], &data[base..base + 6]);
        assert_eq!(&native[base + 24..base + 28], &data[base + 24..base + 28]);
    }
    if !host.fxsave_pointers_without_es {
        // ES=0: see the measured AMD divergence above the instruction witness.
        eprintln!(
            "FXSAVE64 FOP/FIP/FDP not compared: {}",
            HostX87Pointers::NO_FXSAVE_POINTERS
        );
        return;
    }
    // SDM Vol 3A §4.5.3 p. 4-6: a load of FIP ignores the image bits beyond the
    // maximum linear-address width and sign-extends the top kept bit, so FIP is CPU
    // canonical: 48-bit with 4-level paging only, 57-bit with 5-level paging (Vol 3A
    // §4.5 p. 4-4). The probe measures the width. CPUID.(EAX=07H,ECX=0):ECX[16]
    // enumerates 5-level paging (Vol 1 Table 21-20 p. 21-28), so a host that reports it
    // must have measured 57. A hypervisor can hide that bit from its guest, so a host
    // that does not report it may have measured either width. A GitHub-hosted runner
    // measured 57. FDP is not a register that section lists: all 64 bits load.
    // SAFETY: CPUID exists on every x86-64 host; leaf 0 reports the highest basic leaf.
    #[allow(unused_unsafe)] // CPUID intrinsics are unsafe on the workspace MSRV.
    let la57 = unsafe {
        std::arch::x86_64::__cpuid(0).eax >= 7
            && std::arch::x86_64::__cpuid_count(7, 0).ecx & (1 << 16) != 0
    };
    assert!(
        host.fip_bits == Some(57) || (!la57 && host.fip_bits == Some(48)),
        "host FIP width {:?} with CPUID LA57 = {la57}",
        host.fip_bits
    );
    for (i, (fip, _)) in FIP_LOADS.into_iter().enumerate() {
        let base = i * 512;
        assert_eq!(
            &native[base + 6..base + 8],
            &data[base + 6..base + 8],
            "FOP"
        );
        assert_eq!(
            &native[base + 16..base + 24],
            &data[base + 16..base + 24],
            "FDP"
        );
        let saved = u64_at(&native, base + 8);
        let engine = u64_at(&ours, base + 8);
        assert_eq!(Some(saved), host.loaded_fip(fip), "image FIP {fip:#x}");
        // A 48-bit host loads what the engine loads. A 57-bit host differs from the
        // engine only for an image FIP that is not 48-bit canonical.
        if host.fip_bits == Some(48) {
            assert_eq!(saved, engine, "image FIP {fip:#x}");
        } else if saved != engine {
            eprintln!(
                "FXSAVE64 FIP after image FIP {fip:#x} not compared: host {saved:#x}, \
                 engine {engine:#x}: {}",
                HostX87Pointers::WIDER_FIP
            );
        }
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
        assert_eq!(&out[6..24], &loaded_image()[6..24]);
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
    use x86jit_tests::native::{host_x87_pointers, HostX87Pointers};
    let host = host_x87_pointers();
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
    // ES=0: see the host differences above the instruction witness.
    if !host.fxsave_pointers_without_es {
        eprintln!(
            "FXSAVE64 FIP/FDP with ES=0 not compared: {}",
            HostX87Pointers::NO_FXSAVE_POINTERS
        );
    } else {
        assert_eq!(
            u64_at(&out, 8),
            CODE,
            "native memory environment={:02x?}",
            &out[6..24]
        );
        if host.fdp_without_exception {
            assert_eq!(u64_at(&out, 16), DATA);
        } else {
            eprintln!(
                "FXSAVE64 FDP with ES=0 not compared: {}",
                HostX87Pointers::NO_FDP_UPDATE
            );
        }
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
