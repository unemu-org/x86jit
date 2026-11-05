//! Perf-map observer for an embedder-supplied writer. No environment or file access.

use std::io::Write;
use std::sync::Mutex;

use crate::{CompiledCode, CompiledCodeKind};

/// Append perf symbols to a caller-supplied sink. Write errors are ignored because
/// profiler output must not interrupt compilation. Use with
/// [`JitBackend::set_compiled_code_observer`](crate::JitBackend::set_compiled_code_observer).
/// The sink is serialized across foreground and background compiler threads.
pub struct PerfMap<W: Write + Send> {
    writer: Mutex<W>,
}

impl<W: Write + Send> PerfMap<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer: Mutex::new(writer),
        }
    }

    pub fn record(&self, code: CompiledCode) {
        let mut writer = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let _ = format_line(
            &mut *writer,
            code.host_start,
            code.host_len,
            code.kind,
            code.guest_rip,
        );
    }
}

fn format_line(
    out: &mut impl Write,
    start: usize,
    len: u32,
    kind: CompiledCodeKind,
    guest: u64,
) -> std::io::Result<()> {
    let prefix = match kind {
        CompiledCodeKind::Block => "jit_0x",
        CompiledCodeKind::Region => "jit_region_0x",
    };
    writeln!(out, "{start:x} {len:x} {prefix}{guest:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_line_block_is_well_formed_no_0x_prefixes() {
        let mut buf = Vec::new();
        format_line(
            &mut buf,
            0x7f00_1000,
            0x40,
            CompiledCodeKind::Block,
            0x0040_3146,
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(buf).unwrap(),
            "7f001000 40 jit_0x403146\n"
        );
    }

    #[test]
    fn format_line_region_uses_region_prefix() {
        let mut buf = Vec::new();
        format_line(
            &mut buf,
            0x1000,
            0x1_2345,
            CompiledCodeKind::Region,
            0x400000,
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(buf).unwrap(),
            "1000 12345 jit_region_0x400000\n"
        );
    }

    #[test]
    fn format_line_handles_zero_and_large_values() {
        let mut buf = Vec::new();
        format_line(&mut buf, 0, 0, CompiledCodeKind::Block, 0).unwrap();
        assert_eq!(String::from_utf8(buf).unwrap(), "0 0 jit_0x0\n");

        let mut buf = Vec::new();
        format_line(
            &mut buf,
            usize::MAX,
            u32::MAX,
            CompiledCodeKind::Block,
            0xdead_beef_cafe_babe,
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(buf).unwrap(),
            format!("{:x} ffffffff jit_0xdeadbeefcafebabe\n", usize::MAX)
        );
    }
}
