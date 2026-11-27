//! Host filesystem access and legacy diagnostics behind the environment feature.

use crate::lockstep::{Trace, TraceOptions};

pub(crate) fn lockstep_trace() -> Option<Trace> {
    let path = std::env::var_os("X86JIT_LOCKSTEP")?;
    let parse = |key| {
        std::env::var(key)
            .ok()
            .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
    };
    let window = parse("X86JIT_LOCKSTEP_LO").zip(parse("X86JIT_LOCKSTEP_HI"));
    let max_records = std::env::var("X86JIT_LOCKSTEP_MAX")
        .ok()
        .and_then(|s| s.parse().ok());
    let file = std::fs::File::create(path).ok()?;
    Some(Trace::new(
        file,
        TraceOptions {
            window,
            max_records,
        },
    ))
}

/// The test-only compiled-Machine mode named by `X86JIT_TEST_MACHINE_JIT`
/// (decision-29 §9): `forced`, `delayed:K` or `scripted:SEED`.
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn test_machine_jit() -> Option<String> {
    std::env::var("X86JIT_TEST_MACHINE_JIT").ok()
}

#[cfg(target_os = "linux")]
pub(crate) fn host_writable_region_end(at: usize) -> Option<usize> {
    // Linux proc_pid_maps(5): ordered address ranges followed by rwxs permissions.
    // Read the kernel's map metadata, never the memory represented by it.
    let maps = std::fs::read("/proc/self/maps").ok()?;
    for line in maps
        .split(|&byte| byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        let mut fields = line
            .split(|byte| byte.is_ascii_whitespace())
            .filter(|field| !field.is_empty());
        // Mapping paths can contain non-UTF-8 bytes. Only the numeric range is text.
        let (start, end) = std::str::from_utf8(fields.next()?).ok()?.split_once('-')?;
        let start = usize::from_str_radix(start, 16).ok()?;
        let end = usize::from_str_radix(end, 16).ok()?;
        let permissions = fields.next()?;
        if start <= at && at < end {
            return (permissions.get(1) == Some(&b'w')).then_some(end);
        }
    }
    None
}
