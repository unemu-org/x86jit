//! Explicitly enabled legacy environment diagnostics.

use std::fs::{File, OpenOptions};
use std::io::LineWriter;
use std::sync::OnceLock;

use crate::{perfmap::PerfMap, JitBackend};

pub(crate) fn configure(backend: &JitBackend) {
    backend.set_flag_elision(std::env::var_os("X86JIT_NO_FLAG_ELISION").is_none());
    if std::env::var_os("X86JIT_PERF_MAP").as_deref() == Some(std::ffi::OsStr::new("1")) {
        static WRITER: OnceLock<Option<PerfMap<LineWriter<File>>>> = OnceLock::new();
        if let Some(map) = WRITER.get_or_init(|| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("/tmp/perf-{}.map", std::process::id()))
                .ok()
                .map(|file| PerfMap::new(LineWriter::new(file)))
        }) {
            backend.set_compiled_code_observer(move |code| map.record(code));
        }
    }
}
