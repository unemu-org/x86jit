//! Private, destructor-free interpreter returns (§8.1, decision-14).
//!
//! Public Exit owns a box. Keeping it inside every helper's Option adds implicit
//! drop guards and cold payload copies to the successful op walk. This carrier
//! keeps the old scalar return shape. A fault-only detailed tag transfers a temporary
//! owned record into the public Exit before it escapes; success never reads,
//! writes, clears or allocates that record. Non-fault returns never consume the pending record.
use super::{AccessKind, Exit, PortDir};

use std::cell::Cell;

thread_local! {
    static PENDING: Cell<Option<((u64, u16), Exit)>> = const { Cell::new(None) };
}
pub(super) fn memory_key(addr: u64, access: AccessKind) -> (u64, u16) {
    (
        addr,
        256 + match access {
            AccessKind::Read => 0,
            AccessKind::Write => 1,
            AccessKind::Execute => 2,
        },
    )
}
#[cold]
#[inline(never)]
fn record(key: (u64, u16), exit: Exit) {
    PENDING.with(|slot| slot.set(Some((key, exit))));
}
#[cold]
#[inline(never)]
pub(super) fn consume(key: (u64, u16)) -> Option<Exit> {
    PENDING.with(|slot| match slot.take() {
        Some((owner, exit)) if owner == key => Some(exit),
        other => {
            slot.set(other);
            None
        }
    })
}

#[cold]
#[inline(never)]
pub(super) fn consume_current() -> Option<Exit> {
    PENDING.with(|slot| slot.take().map(|(_, exit)| exit))
}

// No Drop implementation: every exiting carrier is consumed exactly once by
// into_exit before another fault can be recorded. A pending record abandoned by a panic is bounded to one box and is
// replaced by the next recorded fault or destroyed with the thread's TLS.
pub(crate) enum InterpExit {
    Syscall,
    Hlt,
    DetailedUnmapped {
        addr: u64,
        access: AccessKind,
    },
    MmioRead {
        addr: u64,
        size: u8,
        total_size: u16,
        part: u8,
    },
    MmioWrite {
        addr: u64,
        size: u8,
        value: u64,
        total_size: u16,
        part: u8,
    },
    UnknownInstruction {
        addr: u64,
        bytes: [u8; 15],
        len: u8,
    },
    DetailedException {
        addr: u64,
        vector: u8,
    },
    PortIo {
        port: u16,
        size: u8,
        dir: PortDir,
        value: u64,
    },
    BudgetExhausted,
    Interrupted,
    CodeSpaceExhausted {
        addr: u64,
    },
    UnsupportedSystemContext {
        addr: u64,
        operation: crate::UnsupportedSystemOperation,
    },
    Shutdown {
        addr: u64,
    },
}
impl InterpExit {
    #[inline(always)]
    pub(super) fn from_exit(exit: Exit) -> Self {
        match exit {
            Exit::UnsupportedSystemContext { addr, operation } => {
                Self::UnsupportedSystemContext { addr, operation }
            }
            Exit::Shutdown { addr } => Self::Shutdown { addr },
            Exit::Syscall => Self::Syscall,
            Exit::Hlt => Self::Hlt,
            exit @ Exit::UnmappedMemory { addr, access, .. } => {
                record(memory_key(addr, access), exit);
                Self::DetailedUnmapped { addr, access }
            }
            exit @ Exit::Exception { addr, vector, .. } => {
                record((addr, u16::from(vector)), exit);
                Self::DetailedException { addr, vector }
            }
            Exit::MmioRead {
                addr,
                size,
                total_size,
                part,
            } => Self::MmioRead {
                addr,
                size,
                total_size,
                part,
            },
            Exit::MmioWrite {
                addr,
                size,
                value,
                total_size,
                part,
            } => Self::MmioWrite {
                addr,
                size,
                value,
                total_size,
                part,
            },
            Exit::UnknownInstruction { addr, bytes, len } => {
                Self::UnknownInstruction { addr, bytes, len }
            }
            Exit::PortIo {
                port,
                size,
                dir,
                value,
            } => Self::PortIo {
                port,
                size,
                dir,
                value,
            },
            Exit::BudgetExhausted => Self::BudgetExhausted,
            // No walk produces the NEC 8080-mode exit; its only producer is
            // the dispatcher's 8080-mode boundary in vm/nec.rs, which returns
            // it directly (pinned by `i8080_mode_exit_has_one_producer`).
            // Sharing an arm keeps this conversion, inlined into every walk,
            // free of a panic path.
            Exit::Interrupted | Exit::I8080Mode { .. } => Self::Interrupted,
            Exit::CodeSpaceExhausted { addr } => Self::CodeSpaceExhausted { addr },
        }
    }
    #[inline(always)]
    pub(crate) fn into_exit(self) -> Exit {
        match self {
            Self::UnsupportedSystemContext { addr, operation } => {
                Exit::UnsupportedSystemContext { addr, operation }
            }
            Self::Shutdown { addr } => Exit::Shutdown { addr },
            Self::Syscall => Exit::Syscall,
            Self::Hlt => Exit::Hlt,
            Self::DetailedUnmapped { addr, access } => consume(memory_key(addr, access))
                .unwrap_or_else(|| Exit::unmapped_memory(addr, access)),
            Self::DetailedException { addr, vector } => {
                consume((addr, u16::from(vector))).unwrap_or_else(|| Exit::exception(addr, vector))
            }
            Self::MmioRead {
                addr,
                size,
                total_size,
                part,
            } => Exit::MmioRead {
                addr,
                size,
                total_size,
                part,
            },
            Self::MmioWrite {
                addr,
                size,
                value,
                total_size,
                part,
            } => Exit::MmioWrite {
                addr,
                size,
                value,
                total_size,
                part,
            },
            Self::UnknownInstruction { addr, bytes, len } => {
                Exit::UnknownInstruction { addr, bytes, len }
            }
            Self::PortIo {
                port,
                size,
                dir,
                value,
            } => Exit::PortIo {
                port,
                size,
                dir,
                value,
            },
            Self::BudgetExhausted => Exit::BudgetExhausted,
            Self::Interrupted => Exit::Interrupted,
            Self::CodeSpaceExhausted { addr } => Exit::CodeSpaceExhausted { addr },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FaultMetadata;
    #[test]
    fn nonfault_return_cannot_reuse_an_abandoned_record() {
        let abandoned = InterpExit::from_exit(
            Exit::exception(1, 13).with_fault_metadata(FaultMetadata::unknown(1)),
        );
        assert!(InterpExit::Hlt.into_exit().fault_metadata().is_none());
        let _ = abandoned; // an unwound fault is replaced, never exposed as legacy evidence
        let current = InterpExit::from_exit(
            Exit::exception(2, 13).with_fault_metadata(FaultMetadata::unknown(2)),
        );
        let owned = current.into_exit();
        assert_eq!(owned.fault_metadata().unwrap().saved_rip(), 2);
        let later = InterpExit::from_exit(
            Exit::exception(3, 13).with_fault_metadata(FaultMetadata::unknown(3)),
        )
        .into_exit();
        assert_eq!(owned.fault_metadata().unwrap().saved_rip(), 2);
        assert_eq!(later.fault_metadata().unwrap().saved_rip(), 3);
        assert!(InterpExit::Hlt.into_exit().fault_metadata().is_none());
    }
    // The shared arm of from_exit maps Exit::I8080Mode to Interrupted, which is
    // right only while no walk produces it. Its one producer is the 8080-mode
    // boundary in vm/nec.rs; a new construction must revisit from_exit first.
    #[test]
    fn i8080_mode_exit_has_one_producer() {
        fn visit(dir: &std::path::Path, root: &std::path::Path, out: &mut Vec<(String, String)>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    visit(&path, root, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let name = path
                        .strip_prefix(root)
                        .unwrap()
                        .components()
                        .map(|c| c.as_os_str().to_string_lossy().into_owned())
                        .collect::<Vec<_>>()
                        .join("/");
                    // String literals, such as this test's expectations, are skipped.
                    let needle = concat!("I8080Mode", " {");
                    for line in std::fs::read_to_string(&path).unwrap().lines() {
                        let code = line.trim();
                        if code.starts_with('"') || code.starts_with("(\"") {
                            continue;
                        }
                        if line.contains(needle) {
                            out.push((name.clone(), line.trim().to_string()));
                        }
                    }
                }
            }
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut found = Vec::new();
        visit(&root, &root, &mut found);
        found.sort();
        let expected: Vec<(String, String)> = [
            (
                "exit.rs",
                "I8080Mode { view: I8080View, entry: I8080Entry },",
            ),
            (
                "interp/result.rs",
                "Exit::Interrupted | Exit::I8080Mode { .. } => Self::Interrupted,",
            ),
            ("vm/nec.rs", "Err(Exit::I8080Mode {"),
        ]
        .iter()
        .map(|(f, l)| (f.to_string(), l.to_string()))
        .collect();
        assert_eq!(found, expected);
    }
    #[test]
    fn private_carriers_have_no_implicit_drop_work() {
        assert!(!std::mem::needs_drop::<InterpExit>());
        assert!(!std::mem::needs_drop::<super::super::InterpResult>());
        assert!(!std::mem::needs_drop::<super::super::BlockResult>());
        assert_eq!(
            std::mem::size_of::<InterpExit>(),
            std::mem::size_of::<Exit>()
        );
    }
}
