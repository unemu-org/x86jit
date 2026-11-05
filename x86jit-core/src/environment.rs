//! Legacy diagnostics, enabled only by an explicit API call.

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
