//! The delivery log of a Machine vCPU (spec.md §17.10, Delivery log): an
//! opt-in ring of the last N exceptions and interrupts the guest took, for an
//! embedder that runs with [`crate::ExceptionDelivery::InGuest`] and so never
//! sees a delivery as an exit. It lives in the `Vcpu`, outside `CpuState` and
//! `jit_abi`; the engine writes it only at a delivery that has entered its
//! handler, from a cold call, and never reads it back.
use std::collections::VecDeque;

/// How a recorded delivery arose (spec.md §17.10, Delivery log).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeliveryKind {
    /// A processor exception raised by an instruction or by a delivery: the
    /// fault and abort classes (SDM Vol 3A §7.5), #DF included, and the
    /// twenty-bit divide error, whose saved IP is the one its model saves.
    Fault,
    /// The single-step debug trap that follows an instruction or a delivery
    /// with TF set.
    Trap,
    /// A software interrupt: INT n, INT 3, a taken INTO, INT1 on the 80386, and
    /// BOUND's exception 5 (SDM Vol 3A §§7.3.3, 7.4.2); on the NEC V20/V30 also
    /// CHKIND's interrupt 5, BRKEM and the 8080-mode CALLN.
    Software,
    /// A maskable external interrupt queued with `Vcpu::inject_irq`.
    Irq,
    /// The non-maskable interrupt (`Vcpu::raise_nmi`, `NmiLine::raise`).
    Nmi,
}

/// One delivery that entered its handler (spec.md §17.10, Delivery log).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeliveryRecord {
    /// The vector the delivery used.
    pub vector: u8,
    pub kind: DeliveryKind,
    /// The return address the delivery saved: on the new stack, or in the old
    /// task's TSS for a task gate.
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

/// What [`DeliveryLog::take`] returns: the records, oldest first, and how
/// many older ones the ring overwrote since the previous take or clear.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeliveryBatch {
    pub records: Vec<DeliveryRecord>,
    pub dropped: u64,
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
/// oldest record and counts it in [`Self::dropped`].
#[derive(Clone, Debug)]
pub struct DeliveryLog {
    records: VecDeque<DeliveryRecord>,
    capacity: usize,
    total: u64,
    dropped: u64,
    /// The twenty-bit dispatcher's CS and SP before the unit it is about to
    /// walk: a software interrupt inside the walk saves this CS, and moves SP.
    pub(crate) walk_origin: (u16, u16),
}

impl DeliveryLog {
    /// The largest capacity [`crate::Vcpu::enable_delivery_log`] accepts.
    pub const MAX_CAPACITY: usize = 1 << 16;

    pub(crate) fn new(capacity: usize) -> Result<Self, DeliveryLogError> {
        if capacity == 0 || capacity > Self::MAX_CAPACITY {
            return Err(DeliveryLogError::Capacity(capacity));
        }
        Ok(Self {
            records: VecDeque::with_capacity(capacity),
            capacity,
            total: 0,
            dropped: 0,
            walk_origin: (0, 0),
        })
    }

    pub(crate) fn push(&mut self, record: DeliveryRecord) {
        if self.records.len() == self.capacity {
            self.records.pop_front();
            self.dropped += 1;
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

    /// The records it holds, oldest first.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &DeliveryRecord> + DoubleEndedIterator {
        self.records.iter()
    }

    /// Records overwritten since the previous [`Self::take`] or [`Self::clear`].
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Deliveries recorded since the log was enabled, overwritten, taken and
    /// cleared ones included.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Remove every record and reset [`Self::dropped`]; [`Self::total`] keeps
    /// counting.
    pub fn clear(&mut self) {
        self.records.clear();
        self.dropped = 0;
    }

    /// Remove and return every record, oldest first, with [`Self::dropped`],
    /// which is then reset.
    pub fn take(&mut self) -> DeliveryBatch {
        DeliveryBatch {
            records: self.records.drain(..).collect(),
            dropped: std::mem::take(&mut self.dropped),
        }
    }
}
