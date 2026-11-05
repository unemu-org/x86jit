use std::io::Write;
use std::sync::{Arc, Mutex};
use x86jit_core::lift::{elide_dead_flags_with_options, lift_block, CpuMode, FetchAddr};
use x86jit_core::lockstep::{Trace, TraceOptions};
use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn vm(bytes: &[u8]) -> Vm {
    let mut vm = Vm::new(VmConfig::flat(0x4000));
    vm.map(0x1000, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
    vm.write_bytes(0x1000, bytes).unwrap();
    vm
}

#[test]
fn embedder_flag_elision_is_an_explicit_lift_option() {
    let vm = vm(&[0x83, 0xc0, 1, 0x83, 0xc0, 2, 0xf4]);
    let ir = lift_block(&vm.mem, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
    let original = format!("{:?}", ir.ops);
    let mut disabled = ir.ops.clone();
    let mut enabled = ir.ops.clone();
    let masks_off = elide_dead_flags_with_options(&mut disabled, false);
    let masks_on = elide_dead_flags_with_options(&mut enabled, true);
    assert_eq!(masks_off, masks_on);
    assert_eq!(format!("{disabled:?}"), original);
    assert_ne!(format!("{enabled:?}"), original);
}

#[test]
fn embedder_lockstep_writer_window_and_cap_work_without_environment() {
    // Scalar tracing requires the explicitly supplied address window.
    let mut vm = vm(&[0xb8, 42, 0, 0, 0, 0x83, 0xc0, 1, 0xf4]);
    let output = Capture::default();
    let trace = Arc::new(Trace::new(
        output.clone(),
        TraceOptions {
            window: Some((0x1005, 0x1008)),
            max_records: Some(1),
        },
    ));
    vm.set_lockstep_trace(Some(trace));
    for _ in 0..2 {
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, Some(4)), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 43);
    }
    let data = output.0.lock().unwrap();
    assert_eq!(u64::from_le_bytes(data[..8].try_into().unwrap()), 0x1005);
    // v3 record: address, byte length, bytes, memory flag/address, two side states.
    assert_eq!(
        data.len(),
        8 + 1 + 3 + 1 + 8 + 2 * (16 * 8 + 8 + 64 + 32 * 16)
    );
    assert_eq!(&data[9..12], &[0x83, 0xc0, 1]);
}

#[test]
fn embedder_defaults_ignore_legacy_environment() {
    const CHILD: &str = "X86JIT_EMBEDDER_OPTIONS_CHILD";
    if let Some(path) = std::env::var_os(CHILD) {
        let vm = vm(&[0x66, 0x0f, 0xef, 0xc0, 0xf4]); // pxor xmm0,xmm0; hlt
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, Some(4)), Exit::Hlt));
        embedder_flag_elision_is_an_explicit_lift_option();
        assert!(!std::path::Path::new(&path).exists());
        return;
    }
    let path = std::env::temp_dir().join(format!("x86jit-trace-opt-in-{}", std::process::id()));
    assert!(!path.exists());
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "embedder_defaults_ignore_legacy_environment"])
        .env(CHILD, &path)
        .env("X86JIT_LOCKSTEP", &path)
        .env("X86JIT_NO_FLAG_ELISION", "1")
        .status()
        .unwrap();
    assert!(status.success());
    assert!(!path.exists());
}

#[test]
#[cfg(feature = "environment")]
fn embedder_legacy_lockstep_environment_requires_api_opt_in() {
    const CHILD: &str = "X86JIT_LOCKSTEP_OPTIONS_CHILD";
    if let Some(path) = std::env::var_os(CHILD) {
        let trace = Trace::from_environment().expect("explicit legacy trace");
        let mut vm = vm(&[0x66, 0x0f, 0xef, 0xc0, 0xf4]);
        vm.set_lockstep_trace(Some(Arc::new(trace)));
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, Some(4)), Exit::Hlt));
        let data = std::fs::read(&path).unwrap();
        assert!(!data.is_empty());
        assert_eq!(u64::from_le_bytes(data[..8].try_into().unwrap()), 0x1000);
        drop(vm);
        std::fs::remove_file(path).unwrap();
        return;
    }
    let path = std::env::temp_dir().join(format!("x86jit-legacy-trace-{}", std::process::id()));
    assert!(!path.exists());
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "embedder_legacy_lockstep_environment_requires_api_opt_in",
        ])
        .env(CHILD, &path)
        .env("X86JIT_LOCKSTEP", &path)
        .env_remove("X86JIT_LOCKSTEP_LO")
        .env_remove("X86JIT_LOCKSTEP_HI")
        .env("X86JIT_LOCKSTEP_MAX", "1")
        .status()
        .unwrap();
    assert!(status.success());
    assert!(!path.exists());
}
