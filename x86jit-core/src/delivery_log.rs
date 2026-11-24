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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    /// The registers of the interrupted code at the delivery, when the log
    /// was enabled with [`DeliveryLogOptions::registers`]; `None` otherwise,
    /// and `None` for a delivery that began before the log was enabled with
    /// them (an 80286/80386 task switch handed off to the embedder across the
    /// enable). Boxed, so that a record without them stays one pointer wide.
    pub regs: Option<Box<DeliveryRegs>>,
    /// EFLAGS.VM before delivery and in the entered handler (design v86 §9 Q4).
    pub virtual_8086_before: bool,
    pub virtual_8086_after: bool,
}

/// The registers of the interrupted code at one delivery (spec.md §17.10,
/// Delivery log), recorded with [`DeliveryLogOptions::registers`].
///
/// Every field holds its value at the moment the delivery began: after the
/// raising instruction's own effects and before the delivery's, so before the
/// frame push, the stack switch, the task switch and the FLAGS changes a
/// delivery makes. For a software interrupt (INT n, INT 3, INTO) these are
/// the instruction's inputs; for a fault, the state the handler is entered
/// from, with the faulting instruction rolled back where the model rolls it
/// back; for an IRQ or NMI, the state after the last instruction that
/// completed. A delivery that follows another one without an
/// instruction between them (an NMI after a fault's frame, a fault raised by
/// a delivery) records the state the earlier delivery left. After a task
/// switch has committed, a fault belongs to the new task and records its
/// state. Records are made by the engine only; new fields may be added.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct DeliveryRegs {
    /// The general registers. The twenty-bit models and the 80286 fill the
    /// low 16 bits (AX, BX, ...) and leave the high 16 zero; the 80386 fills
    /// all 32.
    pub eax: u32,
    pub ebx: u32,
    pub ecx: u32,
    pub edx: u32,
    pub esi: u32,
    pub edi: u32,
    pub ebp: u32,
    /// The interrupted code's stack pointer register, which with [`Self::ss`]
    /// names the interrupted stack: the value before the delivery pushed its
    /// frame, and on an inward delivery the outer stack's, not the handler's.
    /// On the 80386 it is the whole ESP register: where SS is a sixteen-bit
    /// stack segment (always in real mode) the stack pointer is its low 16
    /// bits, and the high half is carried as the register holds it, not
    /// zeroed.
    pub esp: u32,
    pub ds: u16,
    pub es: u16,
    pub ss: u16,
    /// FS and GS on the 80386; `None` on the models without them.
    pub fs: Option<u16>,
    pub gs: Option<u16>,
    /// The FLAGS image (EFLAGS on the 80386) of the interrupted code as PUSHF
    /// would form it on the model, the bits the model reads as set included.
    /// On the 80386 RF and VM are therefore zero, as PUSHFD stores them (SDM
    /// Vol 2B §4.3 PUSHF/PUSHFD/PUSHFQ, Operation), whatever their live
    /// value; the frame of a fault can hold RF set. On the twenty-bit models
    /// it is the image the delivery saved in its frame, which an interrupt
    /// taken in NEC 8080 mode saves with MD clear.
    pub flags: u32,
}

// The registers `cpu` holds, as the delivery log records them on a model of
// `generation`, with the FLAGS image `flags` and the stack pointer `sp`
// (spec.md §17.10, Delivery log). The twenty-bit models and the 80286 have
// sixteen-bit registers (8086 Family UM §2.2; I286 PRM §2.3.1), and only
// the 80386 has FS and GS (80386 PRM §2.3.2).
pub(crate) fn regs_at(
    cpu: &crate::CpuState,
    generation: crate::CpuGeneration,
    flags: u32,
    sp: u64,
) -> DeliveryRegs {
    let wide = generation == crate::CpuGeneration::I386;
    let mask = if wide { u64::from(u32::MAX) } else { 0xffff };
    let gpr = |index: usize| (cpu.gpr[index] & mask) as u32;
    DeliveryRegs {
        eax: gpr(0),
        ebx: gpr(3),
        ecx: gpr(1),
        edx: gpr(2),
        esi: gpr(6),
        edi: gpr(7),
        ebp: gpr(5),
        esp: (sp & mask) as u32,
        ds: cpu.ds,
        es: cpu.es,
        ss: cpu.ss,
        fs: wide.then_some(cpu.fs),
        gs: wide.then_some(cpu.gs),
        flags,
    }
}

/// What [`crate::Vcpu::enable_delivery_log_with`] records besides the default
/// fields. New options may be added; start from [`Default`], which records
/// nothing extra. A log keeps the options it was enabled with: they change
/// only through another `enable_delivery_log_with`, which starts an empty log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct DeliveryLogOptions {
    registers: bool,
}

impl DeliveryLogOptions {
    /// These options with [`DeliveryRecord::regs`] filled when `registers` is
    /// set. The registers are copied from the vCPU state at each delivery;
    /// nothing is copied while this is off.
    pub fn registers(mut self, registers: bool) -> Self {
        self.registers = registers;
        self
    }

    /// Whether records carry [`DeliveryRecord::regs`].
    pub fn records_registers(&self) -> bool {
        self.registers
    }
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
    options: DeliveryLogOptions,
    total: u64,
    dropped: u64,
}

impl DeliveryLog {
    /// The largest capacity [`crate::Vcpu::enable_delivery_log`] accepts.
    pub const MAX_CAPACITY: usize = 1 << 16;

    pub(crate) fn new(
        capacity: usize,
        options: DeliveryLogOptions,
    ) -> Result<Self, DeliveryLogError> {
        if capacity == 0 || capacity > Self::MAX_CAPACITY {
            return Err(DeliveryLogError::Capacity(capacity));
        }
        Ok(Self {
            records: VecDeque::new(),
            capacity,
            options,
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

    /// The options the log was enabled with.
    pub fn options(&self) -> DeliveryLogOptions {
        self.options
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
