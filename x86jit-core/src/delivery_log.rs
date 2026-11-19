//! The delivery log of a Machine vCPU (spec.md §17.10, Delivery log): an
//! opt-in ring of the last N exceptions and interrupts the guest took, for an
//! embedder that runs with [`crate::ExceptionDelivery::InGuest`] and so never
//! sees a delivery as an exit. It lives in the `Vcpu`, outside `CpuState`,
//! `jit_abi` and any checkpoint an embedder takes of `CpuState`; the engine
//! writes it only at a delivery that has entered its handler, from a cold
//! call, and never reads it back.
use std::collections::VecDeque;

/// How a recorded delivery arose, and so what its saved IP names (spec.md
/// §17.10, Delivery log).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeliveryKind {
    /// A processor exception raised by an instruction or by a delivery: the
    /// fault and abort classes (SDM Vol 3A §7.5), #DF included, and BOUND's
    /// #BR, a fault that saves the BOUND instruction (SDM Vol 3A §7.15, Event 5
    /// p. 7-31). The saved IP is the faulting instruction, which `retired`
    /// does not count. One exception: the twenty-bit divide error saves the IP its
    /// model saves (`ModelQuirks::divide_saves_next_ip`), the next instruction
    /// on the models that set it.
    Fault,
    /// The single-step debug trap that follows an instruction or a delivery
    /// with TF set. The saved IP is the next instruction to execute: after a
    /// twenty-bit software interrupt, the first instruction of its handler.
    Trap,
    /// A software interrupt: INT n, INT 3, a taken INTO and INT1 on the 80386
    /// (SDM Vol 3A §7.4.2); on the NEC V20/V30 also CHKIND's interrupt 5,
    /// BRKEM and the 8080-mode CALLN. The saved IP is the next instruction,
    /// and `retired` counts the interrupting instruction.
    Software,
    /// A maskable external interrupt queued with `Vcpu::inject_irq`. The saved
    /// IP is the instruction it interrupted, which has not run.
    Irq,
    /// The non-maskable interrupt (`Vcpu::raise_nmi`, `NmiLine::raise`). The
    /// saved IP is the instruction it interrupted, which has not run.
    Nmi,
}

/// One delivery that entered its handler (spec.md §17.10, Delivery log).
/// Records are made by the engine only; new fields may be added.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct DeliveryRecord {
    /// The vector the delivery used.
    pub vector: u8,
    pub kind: DeliveryKind,
    /// The return address the delivery saved: on the new stack, or in the old
    /// task's TSS for a task gate. What it names depends on [`Self::kind`].
    pub saved_cs: u16,
    pub saved_ip: u32,
    /// The linear address of the instruction `saved_cs:saved_ip` names: the CS
    /// base in effect before the delivery plus `saved_ip`, before the A20 gate.
    /// On the twenty-bit models it wraps at 20 bits as their address formation
    /// does.
    pub saved_linear: u32,
    /// The error code the delivery pushed; `None` when it pushed none, which
    /// includes every real-address and twenty-bit delivery.
    pub error_code: Option<u32>,
    /// CPL before the delivery and in its handler (0 outside protected mode).
    pub cpl_before: u8,
    pub cpl_after: u8,
    /// `Vcpu::retired_instructions` when the handler was entered: a software
    /// interrupt's own instruction is included, a faulting instruction is not.
    pub retired: u64,
}

/// What [`DeliveryLog::take`] returns: the records, oldest first, with the
/// log's counts at the take. New fields may be added.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeliveryBatch {
    pub records: Vec<DeliveryRecord>,
    /// Records the full ring overwrote since the previous take or drain,
    /// which this batch therefore lacks.
    pub dropped: u64,
    /// [`DeliveryLog::total`] at the take: the records are the deliveries
    /// numbered `total - records.len()` up to `total`, counting from 0, so
    /// batches taken one after another order themselves.
    pub total: u64,
}

/// What [`DeliveryLog::drain_into`] returns besides the records it appends:
/// the same counts as [`DeliveryBatch`]. New fields may be added.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct DeliveryCounts {
    /// As [`DeliveryBatch::dropped`].
    pub dropped: u64,
    /// As [`DeliveryBatch::total`].
    pub total: u64,
}

/// Why [`crate::Vcpu::enable_delivery_log`] refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeliveryLogError {
    /// The vCPU does not run [`crate::ExecutionProfile::Machine`].
    NotMachine,
    /// The capacity is zero or above [`DeliveryLog::MAX_CAPACITY`].
    Capacity(usize),
}

impl std::fmt::Display for DeliveryLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotMachine => {
                f.write_str("the delivery log exists only on ExecutionProfile::Machine vCPUs")
            }
            Self::Capacity(capacity) => write!(
                f,
                "delivery log capacity {capacity} is outside 1..={}",
                DeliveryLog::MAX_CAPACITY
            ),
        }
    }
}

impl std::error::Error for DeliveryLogError {}

/// The ring of the last [`Self::capacity`] deliveries of one vCPU, read
/// between runs (spec.md §17.10, Delivery log). A full ring overwrites its
/// oldest record and counts it in [`Self::dropped`]. Its storage grows with
/// the records it holds, up to the capacity; nothing is reserved when it is
/// enabled.
///
/// [`Self::take`] and [`Self::drain_into`] are the reads that consume: each
/// removes the records and hands over the dropped count with the total.
/// [`Self::iter`], [`Self::dropped`] and [`Self::total`] only look.
#[derive(Clone, Debug)]
pub struct DeliveryLog {
    records: VecDeque<DeliveryRecord>,
    capacity: usize,
    total: u64,
    dropped: u64,
}

impl DeliveryLog {
    /// The largest capacity [`crate::Vcpu::enable_delivery_log`] accepts.
    pub const MAX_CAPACITY: usize = 1 << 16;

    pub(crate) fn new(capacity: usize) -> Result<Self, DeliveryLogError> {
        if capacity == 0 || capacity > Self::MAX_CAPACITY {
            return Err(DeliveryLogError::Capacity(capacity));
        }
        Ok(Self {
            records: VecDeque::new(),
            capacity,
            total: 0,
            dropped: 0,
        })
    }

    pub(crate) fn push(&mut self, record: DeliveryRecord) {
        let len = self.records.len();
        if len == self.capacity {
            self.records.pop_front();
            self.dropped += 1;
        } else if len == self.records.capacity() {
            // Grow by doubling, but never past the capacity.
            self.records
                .reserve_exact(len.max(8).min(self.capacity - len));
        }
        self.records.push_back(record);
        self.total += 1;
    }

    /// The number of records the ring keeps.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of records it holds now.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The records it holds, oldest first, without removing them.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &DeliveryRecord> + DoubleEndedIterator {
        self.records.iter()
    }

    /// Records overwritten since the previous [`Self::take`],
    /// [`Self::drain_into`] or [`Self::clear`].
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Deliveries recorded since the log was enabled, overwritten, taken and
    /// cleared ones included.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Discard every record and the dropped count, unread; [`Self::total`]
    /// keeps counting. To read both, use [`Self::take`] or
    /// [`Self::drain_into`] instead.
    pub fn clear(&mut self) {
        self.records.clear();
        self.dropped = 0;
    }

    /// Remove and return every record, oldest first, with [`Self::dropped`],
    /// which is then reset, and [`Self::total`]. The records go into a new
    /// `Vec`; [`Self::drain_into`] reuses the caller's.
    pub fn take(&mut self) -> DeliveryBatch {
        let records = self.records.drain(..).collect();
        let counts = self.counts();
        DeliveryBatch {
            records,
            dropped: counts.dropped,
            total: counts.total,
        }
    }

    /// Remove every record and append it to `records`, oldest first, without
    /// clearing what `records` already holds; return [`Self::dropped`], which
    /// is then reset, and [`Self::total`]. An embedder that drains after every
    /// run reuses one buffer.
    pub fn drain_into(&mut self, records: &mut Vec<DeliveryRecord>) -> DeliveryCounts {
        records.extend(self.records.drain(..));
        self.counts()
    }

    fn counts(&mut self) -> DeliveryCounts {
        DeliveryCounts {
            dropped: std::mem::take(&mut self.dropped),
            total: self.total,
        }
    }
}
