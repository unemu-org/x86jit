//! Isolate signal state and enforce a wall-clock limit even on engine deadlock.
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub fn isolated(name: &str, body: impl FnOnce(Instant)) {
    isolated_for(name, Duration::from_secs(90), body);
}

pub fn isolated_for(name: &str, cap: Duration, body: impl FnOnce(Instant)) {
    let start = Instant::now();
    if std::env::var("X86JIT_STRESS_CHILD").as_deref() == Ok(name) {
        body(start + cap - Duration::from_secs(10));
        eprintln!("MEASURED {name}: {:.3}s", start.elapsed().as_secs_f64());
        return;
    }
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            name,
            "--test-threads",
            "2",
            "--nocapture",
            "--include-ignored",
        ])
        .env("X86JIT_STRESS_CHILD", name)
        .stdin(Stdio::null())
        .spawn()
        .unwrap();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "{name}: {status}");
            return;
        }
        if start.elapsed() >= cap {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("{name}: killed after {cap:?}");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

pub fn remaining(deadline: Instant) -> Duration {
    deadline
        .checked_duration_since(Instant::now())
        .expect("stress deadline exceeded")
}

#[allow(dead_code)] // Also included by raw-provider tests, which have no polling.
pub fn until(deadline: Instant, mut done: impl FnMut() -> bool) {
    while !done() {
        remaining(deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
}
