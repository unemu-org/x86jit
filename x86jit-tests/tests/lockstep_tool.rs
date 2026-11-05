#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;

fn capture(path: &Path) -> Command {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut command = Command::new("bash");
    command
        .current_dir(root)
        .arg(root.join("scripts/lockstep.sh"))
        .args(["capture", "--max", "0", "--out"])
        .arg(path)
        .arg("--");
    command
}

fn trace_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("x86jit-{name}-{}.trace", std::process::id()))
}

#[test]
fn lockstep_capture_rejects_missing_and_empty_output() {
    let path = trace_path("missing-output");
    for args in [vec!["true"], vec!["sh", "-c", ": > \"$X86JIT_LOCKSTEP\""]] {
        let output = capture(&path).args(args).output().unwrap();
        if path.exists() {
            std::fs::remove_file(&path).unwrap();
        }
        assert!(
            !output.status.success(),
            "capture must reject missing or empty output"
        );
        assert!(String::from_utf8_lossy(&output.stderr).contains("capture produced no trace"));
    }
}

#[test]
fn lockstep_capture_accepts_trace_api_consumer() {
    const CHILD: &str = "X86JIT_LOCKSTEP_TOOL_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let mut vm = x86jit_core::Vm::new(x86jit_core::VmConfig::flat(0x4000));
        let trace = x86jit_core::lockstep::Trace::from_environment().expect("requested trace");
        vm.set_lockstep_trace(Some(std::sync::Arc::new(trace)));
        vm.map(
            0x1000,
            0x1000,
            x86jit_core::Prot::RWX,
            x86jit_core::RegionKind::Ram,
        )
        .unwrap();
        vm.write_bytes(0x1000, &[0x66, 0x0f, 0xef, 0xc0, 0xf4])
            .unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(x86jit_core::Reg::Rip, 0x1000);
        assert!(matches!(cpu.run(&vm, Some(2)), x86jit_core::Exit::Hlt));
        return;
    }
    let path = trace_path("api-consumer");
    let output = capture(&path)
        .arg(std::env::current_exe().unwrap())
        .args(["--exact", "lockstep_capture_accepts_trace_api_consumer"])
        .env(CHILD, "1")
        .env("X86JIT_LOCKSTEP_MAX", "0")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!std::fs::read(&path).unwrap().is_empty());
    std::fs::remove_file(path).unwrap();
}
