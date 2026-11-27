//! Relift diagnostics for Machine private proofs (xj-to9.53), compiled only
//! with the `test-hooks` feature: why the Flat32, Segmented and paged proof
//! lifts of one VM happened, what the live-field table (`live.rs`) did, and
//! the most recent relifts and refused fields with the instruction bytes
//! around their trigger, so a guest's patched instruction forms can be named.
//! Read through [`crate::Vm::relift_diagnostics`].
use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

/// The proof map a lift fills.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProofClass {
    /// Unpaged Flat32 (`flat32.rs`).
    Flat32,
    /// Unpaged Segmented (`flat32.rs`).
    Segmented,
    /// Any class with CR0.PG set (`paging.rs`).
    Paged,
}

impl ProofClass {
    pub const ALL: [Self; 3] = [Self::Flat32, Self::Segmented, Self::Paged];
    pub(crate) fn of(class: crate::SegmentClass, paged: bool) -> Self {
        match (paged, class) {
            (true, _) => Self::Paged,
            (false, crate::SegmentClass::Flat32) => Self::Flat32,
            (false, _) => Self::Segmented,
        }
    }
}

/// Which bytes of the dropped proof a recorded store overlapped, classified
/// against the bytes the relift decodes there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoreClass {
    /// Wholly inside the immediate or displacement of a form the live-field
    /// path reads live (MOV, LEA, ALU, CMP, TEST, shifts and rotates, IMUL,
    /// MOVZX, MOVSX, a near relative JMP): such a store is absorbed
    /// once its field is promoted, so this counts the patches before it
    /// (two per field) and those to a field that was refused or demoted.
    LiveField,
    /// Wholly inside an immediate, displacement or branch displacement of
    /// another form (Jcc/CALL/LOOP, PUSH, ENTER, RET, ...).
    OtherField,
    /// Touching a prefix, opcode, ModRM or SIB byte, or crossing a field.
    Opcode,
    /// No instruction the relift decodes contains the store's first byte.
    Outside,
}

/// Why a write record has a gap, which drops every private proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GapKind {
    /// More recorded advances than the write log retains (256).
    Overflow,
    /// An unrecorded invalidation: `invalidate_overlapping` (mapping,
    /// protection, region-kind, backing or fetch-policy change, or one an
    /// embedder issues).
    Invalidate,
    /// A drained store before any private proof existed.
    UnrecordedDrain,
    /// A child arena's pages reconciled after the fact.
    ChildArena,
    /// A code-lifetime reset of the shared cache.
    Reset,
}

/// Why a lift was repeated inside one resolve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RetryKind {
    /// The source bytes or the fetch permit changed while lifting.
    SourceChanged,
    /// The lift baked a live field, which it demoted.
    Demotion,
    /// The proof generation moved before registration.
    Registration,
    /// The lift promoted a field (one extra lift per promotion).
    Promotion,
}

/// Why a proof lift happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReliftCause {
    /// The key's first lift on this vCPU (also after its history, bounded
    /// to 65,536 keys per proof map, was forgotten).
    First,
    /// A recorded store overlapped the proof's bytes.
    Store(StoreClass),
    /// A live-field promotion advanced the proof generation over the proof.
    Promotion,
    /// A gap in the write record dropped every proof.
    Gap(GapKind),
    /// The proof map reached its bound (8192 unpaged, 4096 paged, or their IR budgets) and swept
    /// a quarter of its proofs.
    Bound,
    /// The proof's context identity was evicted from the retained contexts
    /// (paged only: an unpaged proof of an evicted identity is never
    /// returned again and leaves by the bound).
    ContextEvicted,
    /// A mapping change dropped every proof.
    Mapping,
    /// Paged only: a TLB flush (a CR3 load, a change of CR0's paging bits,
    /// A20 or the model, an explicit invalidation), a full context table or
    /// a reset dropped every proof.
    Paging,
    /// Paged only: after a TLB generation advance (an eviction or INVLPG of
    /// a page a fetch used), a code page of the proof no longer translated
    /// to the frame it was fetched from, or its fetch would fault.
    Translation,
    /// The proof's fetch permit went stale.
    Permit,
    /// The lift was repeated inside the same resolve.
    Retry(RetryKind),
    /// The key's previous lift ran once without a proof: its source could
    /// not be published.
    Unpublished,
}

impl ReliftCause {
    pub const ALL: [Self; CAUSES] = [
        Self::First,
        Self::Store(StoreClass::LiveField),
        Self::Store(StoreClass::OtherField),
        Self::Store(StoreClass::Opcode),
        Self::Store(StoreClass::Outside),
        Self::Promotion,
        Self::Gap(GapKind::Overflow),
        Self::Gap(GapKind::Invalidate),
        Self::Gap(GapKind::UnrecordedDrain),
        Self::Gap(GapKind::ChildArena),
        Self::Gap(GapKind::Reset),
        Self::Bound,
        Self::ContextEvicted,
        Self::Mapping,
        Self::Paging,
        Self::Translation,
        Self::Permit,
        Self::Retry(RetryKind::SourceChanged),
        Self::Retry(RetryKind::Demotion),
        Self::Retry(RetryKind::Registration),
        Self::Retry(RetryKind::Promotion),
        Self::Unpublished,
    ];
    fn index(self) -> usize {
        Self::ALL.iter().position(|&c| c == self).unwrap()
    }
}
const CAUSES: usize = 22;

/// Why a paged vCPU's TLB generation advanced (xj-to9.70). A generation
/// names the TLB's fetched entries: it advances when one of them leaves or is
/// replaced, and with every flush. Paged proofs revalidate their code pages
/// against the TLB once per generation (`paging.rs`); only a flush drops them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TlbAdvance {
    /// The translation context changed: CR3, CR0.PG/CD/NW/WP/PE, A20 or the
    /// model's flags mask (`Tlb::reconcile`).
    ContextFlush,
    /// An explicit flush: a CR3 load (equal values too), a task switch's CR3,
    /// a reset, a mapping change or the embedder's paging invalidation.
    ExplicitFlush,
    /// A fill replaced a fetched entry of another page (eviction).
    Replaced,
    /// A #PF on a cached entry removed a fetched one.
    FaultRemoved,
    /// A write through a clean fetched entry, or a 486 protection retry,
    /// removed it to walk again (paging design §3.2).
    Rewalked,
    /// A failed D-bit update removed a fetched entry.
    DirtyFailed,
    /// INVLPG of a page.
    InvalidatePage,
    /// A replayed lane restored its slots and their generation.
    LaneUndone,
}

impl TlbAdvance {
    pub const ALL: [Self; TLB_ADVANCES] = [
        Self::ContextFlush,
        Self::ExplicitFlush,
        Self::Replaced,
        Self::FaultRemoved,
        Self::Rewalked,
        Self::DirtyFailed,
        Self::InvalidatePage,
        Self::LaneUndone,
    ];
}
pub(crate) const TLB_ADVANCES: usize = 8;

/// Paged-proof translation checks (xj-to9.70): the TLB generation advances by
/// site, and the proof hits that revalidated their code pages after one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TlbStats {
    /// Generation advances by site; nonzero only.
    pub advances: Vec<(TlbAdvance, u64)>,
    /// Proof hits that translated their code pages again: the frames
    /// matched and the proof was kept, or one differed and it relifted.
    pub revalidated: u64,
    pub revalidation_failed: u64,
}

/// Why a patched range was not read live; it then relifts on every patch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LiveRefusal {
    /// The instruction is not a live-capable form (`lift::legacy::live_fields`:
    /// not MOV, LEA, ALU, CMP, TEST, a shift or rotate, IMUL, MOVZX, MOVSX or
    /// a near relative JMP, a segment or control register operand, or a
    /// second immediate).
    IneligibleForm,
    /// The instruction's bytes are not one bus range (it straddles pages
    /// mapped apart).
    NonContiguous,
    /// The patched range lies in the instruction but not inside one field.
    OutsideField,
    /// The exactness probe refused the field.
    Probe(ProbeFailure),
    /// The live table was full of fields whose code is still held.
    TableFull,
}

/// Why `lift::legacy::live_instruction` refused a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProbeFailure {
    /// The fetched bytes are not exactly one instruction.
    Length,
    /// No zero, sign or plain extension reproduces the decoder's value.
    Extension,
    /// The lowering did not read the field through `live_field`.
    NoLiveRead,
    /// The lift traps, is unsupported or refuses.
    Trap,
    /// The lift differs between the current bytes, their complement and zero.
    ValueDependent,
    /// A probe lift failed.
    LiftFailed,
}

impl LiveRefusal {
    pub const ALL: [Self; REFUSALS] = [
        Self::IneligibleForm,
        Self::NonContiguous,
        Self::OutsideField,
        Self::Probe(ProbeFailure::Length),
        Self::Probe(ProbeFailure::Extension),
        Self::Probe(ProbeFailure::NoLiveRead),
        Self::Probe(ProbeFailure::Trap),
        Self::Probe(ProbeFailure::ValueDependent),
        Self::Probe(ProbeFailure::LiftFailed),
        Self::TableFull,
    ];
    fn index(self) -> usize {
        Self::ALL.iter().position(|&c| c == self).unwrap()
    }
}
const REFUSALS: usize = 10;

/// Why a live field returned to normal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LiveDemotion {
    /// A lift decoded the field's bytes without reading it live (another
    /// decoding, or an entry inside the instruction).
    Baked,
    /// A relift of the field's instruction failed the exactness probe.
    ProbeFailed,
    /// A store of at most four bytes touched another byte of the field's
    /// instruction, or crossed the field.
    StoreAround,
    /// A wider store overlapped the field's instruction.
    WideStore,
}

impl LiveDemotion {
    pub const ALL: [Self; DEMOTIONS] = [
        Self::Baked,
        Self::ProbeFailed,
        Self::StoreAround,
        Self::WideStore,
    ];
    fn index(self) -> usize {
        Self::ALL.iter().position(|&c| c == self).unwrap()
    }
}
const DEMOTIONS: usize = 4;

/// Live-field counters (`live.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Live {
    /// `LiveFields::instruction` calls: one per instruction a proof lift decodes.
    Calls,
    /// Calls before any small code write was counted: the plain lift.
    Inactive,
    /// Calls whose instruction holds no live field and no patched range.
    Plain,
    /// Table snapshots taken (the table lock, once per lift and page).
    Snapshots,
    /// Calls that decoded the instruction's fields.
    Slow,
    /// Exactness probes run (three lifts each).
    Probes,
    /// Instructions lifted with a live read.
    Lowered,
    /// Fields promoted.
    Promotions,
    /// Stores absorbed by a live field.
    Absorbed,
    /// Stores inside a live field not absorbed: a shared unit covers them.
    SharedCovered,
    /// Small code stores counted toward a promotion.
    Counted,
    /// Counted-store table clears (more than 1,024 ranges).
    CountClears,
}
const LIVE: usize = 12;

/// One recent relift.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReliftRecord {
    pub class: ProofClass,
    pub cause: ReliftCause,
    /// The block's decode IP (EIP) and the bus address of its first byte,
    /// when the dropped proof's sources name it.
    pub block_ip: u32,
    pub block_bus: Option<u64>,
    /// The triggering store's bus address and length, when one is known.
    pub write: Option<(u64, u32)>,
    /// The instruction containing the store: its bus address, up to 16 of
    /// its current bytes, and its text.
    pub instruction: Option<(u64, Vec<u8>, String)>,
    /// The proof key, `name=value` (hex) per field.
    pub key: String,
    /// The fields that differ from the previous lift's key at the same
    /// linear entry address, `name old->new`; empty for its first lift.
    pub key_diff: String,
}

/// A proof key's named fields, for the relift diagnostics.
pub(crate) trait DiagKey {
    /// The linear entry address the key names.
    fn site(&self) -> u64;
    fn fields(&self) -> Vec<(&'static str, u64)>;
}

/// A context-identity event of `Execution::sync_class`, buffered per
/// thread (the Execution has no VM) and drained into the VM's diagnostics
/// at the next resolve.
pub(crate) enum ContextNote {
    /// The stamp changed.
    Switch,
    /// It matched one of the recent identities.
    Recent,
    /// It matched a retained identity.
    Retained,
    /// A new identity, and the previous stamp's fields.
    Created {
        id: u64,
        previous: Option<Vec<(&'static str, String)>>,
        fields: Vec<(&'static str, String)>,
    },
    /// The least recently used identity, and its stamp's fields.
    Evicted {
        id: u64,
        fields: Vec<(&'static str, String)>,
    },
    /// An instruction invalidated every translation of the vCPU.
    Invalidated,
}

thread_local! {
    static NOTES: std::cell::RefCell<Vec<ContextNote>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub(crate) fn note_context(note: ContextNote) {
    NOTES.with(|n| {
        let mut n = n.borrow_mut();
        if n.len() < 4096 {
            n.push(note);
        }
    });
}

/// Move this thread's context notes into `diag`.
pub(crate) fn drain_contexts(diag: &Diag) {
    let notes = NOTES.with(|n| {
        let mut n = n.borrow_mut();
        if n.is_empty() {
            Vec::new()
        } else {
            std::mem::take(&mut *n)
        }
    });
    if notes.is_empty() {
        return;
    }
    let mut c = diag.contexts.lock().unwrap();
    for note in notes {
        match note {
            ContextNote::Switch => c.stats.switches += 1,
            ContextNote::Recent => c.stats.recent_hits += 1,
            ContextNote::Retained => c.stats.retained_hits += 1,
            ContextNote::Invalidated => c.stats.invalidations += 1,
            ContextNote::Created {
                id,
                previous,
                fields,
            } => {
                c.stats.created += 1;
                let mut text = format!("context {id} created");
                if let Some(previous) = previous {
                    let mut changed = Vec::new();
                    for ((name, old), (_, new)) in previous.iter().zip(&fields) {
                        if old != new {
                            changed.push(format!("{name} {old}->{new}"));
                            *c.created_by_field.entry(name).or_default() += 1;
                        }
                    }
                    text += &format!(" from the previous stamp: {}", changed.join(", "));
                } else {
                    text += &format!(" (first): {}", join(&fields));
                }
                push_ring(&mut c.recent, text);
            }
            ContextNote::Evicted { id, fields } => {
                c.stats.evicted += 1;
                push_ring(
                    &mut c.recent,
                    format!("context {id} evicted: {}", join(&fields)),
                );
            }
        }
    }
}

fn join(fields: &[(&'static str, String)]) -> String {
    fields
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn push_ring<T>(ring: &mut VecDeque<T>, item: T) {
    if ring.len() == RING {
        ring.pop_front();
    }
    ring.push_back(item);
}

/// Context-identity statistics of the vCPUs' Machine dispatch.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ContextStats {
    /// Stamp changes (`Execution::sync_class`).
    pub switches: u64,
    /// Of those: matched a recent identity, a retained one, or created one.
    pub recent_hits: u64,
    pub retained_hits: u64,
    pub created: u64,
    /// Identities evicted from the 256 retained.
    pub evicted: u64,
    /// Instructions that invalidated every translation (CR0 or CR3 writes,
    /// descriptor-table loads and the like).
    pub invalidations: u64,
    /// Created identities by the stamp field that differed from the
    /// previous stamp.
    pub created_by_field: Vec<(String, u64)>,
    /// The last 64 creations and evictions, with the fields.
    pub recent: Vec<String>,
}

/// Proof-key statistics: distinct keys per linear entry address.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeyStats {
    /// Entry addresses lifted, and distinct keys lifted over all of them.
    pub sites: usize,
    pub keys: usize,
    /// The entry addresses with the most distinct keys: (address, keys).
    pub top: Vec<(u64, usize)>,
    /// Lifts whose key differed from the previous lift's at the same
    /// address, by differing field.
    pub changes_by_field: Vec<(String, u64)>,
}

#[derive(Default)]
struct Contexts {
    stats: ContextStats,
    created_by_field: std::collections::BTreeMap<&'static str, u64>,
    recent: VecDeque<String>,
}

#[derive(Default)]
struct Keys {
    sites: std::collections::HashMap<u64, (Vec<u64>, std::collections::HashSet<Vec<u64>>)>,
    changes: std::collections::BTreeMap<&'static str, u64>,
}

/// One recent refused field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefusalRecord {
    pub reason: LiveRefusal,
    /// The patched range's bus address and length.
    pub patched: (u64, u32),
    /// The instruction's bus address, bytes and text.
    pub instruction: (u64, Vec<u8>, String),
}

const RING: usize = 64;

fn ring<T: Clone>(ring: &Mutex<VecDeque<T>>, take: bool) -> Vec<T> {
    let mut ring = ring.lock().unwrap();
    if take {
        std::mem::take(&mut *ring).into()
    } else {
        ring.iter().cloned().collect()
    }
}

#[derive(Default)]
pub(crate) struct Diag {
    lifts: [[AtomicU64; CAUSES]; 3],
    uncached: [AtomicU64; 3],
    live: [AtomicU64; LIVE],
    refusals: [AtomicU64; REFUSALS],
    demotions: [AtomicU64; DEMOTIONS],
    recent: Mutex<VecDeque<ReliftRecord>>,
    refused: Mutex<VecDeque<RefusalRecord>>,
    contexts: Mutex<Contexts>,
    keys: Mutex<Keys>,
    tlb: [AtomicU64; TLB_ADVANCES],
    revalidated: [AtomicU64; 2],
}

fn push<T>(ring: &Mutex<VecDeque<T>>, item: T) {
    let mut ring = ring.lock().unwrap();
    if ring.len() == RING {
        ring.pop_front();
    }
    ring.push_back(item);
}

/// Formats one instruction at the start of `bytes`, decoded with `bits`.
pub(crate) fn text(bytes: &[u8], ip: u32, bits: u8) -> (usize, String) {
    use iced_x86::Formatter;
    let insn = iced_x86::Decoder::with_ip(
        u32::from(bits),
        bytes,
        u64::from(ip),
        iced_x86::DecoderOptions::NONE,
    )
    .decode();
    let mut text = String::new();
    iced_x86::IntelFormatter::new().format(&insn, &mut text);
    (insn.len(), text)
}

impl Diag {
    pub(crate) fn lift(&self, class: ProofClass, cause: ReliftCause) {
        self.lifts[class as usize][cause.index()].fetch_add(1, Relaxed);
    }
    pub(crate) fn uncached(&self, class: ProofClass) {
        self.uncached[class as usize].fetch_add(1, Relaxed);
    }
    pub(crate) fn live(&self, counter: Live) {
        self.live[counter as usize].fetch_add(1, Relaxed);
    }
    pub(crate) fn demote(&self, why: LiveDemotion) {
        self.demotions[why.index()].fetch_add(1, Relaxed);
    }
    pub(crate) fn refuse(&self, record: RefusalRecord) {
        self.refusals[record.reason.index()].fetch_add(1, Relaxed);
        push(&self.refused, record);
    }
    pub(crate) fn record(&self, record: ReliftRecord) {
        push(&self.recent, record);
    }
    /// Add a vCPU's TLB generation advances, by `TlbAdvance` index.
    pub(crate) fn tlb(&self, advances: &mut [u64; TLB_ADVANCES]) {
        for (counter, n) in self.tlb.iter().zip(advances.iter_mut()) {
            if *n != 0 {
                counter.fetch_add(std::mem::take(n), Relaxed);
            }
        }
    }
    /// A proof hit revalidated its code pages: kept (`true`) or not.
    pub(crate) fn revalidated(&self, kept: bool) {
        self.revalidated[usize::from(!kept)].fetch_add(1, Relaxed);
    }

    /// Note a lift of `key`; returns the key's text and its difference from
    /// the previous key lifted at the same entry address.
    pub(crate) fn key(&self, key: &impl DiagKey) -> (String, String) {
        let fields = key.fields();
        let values: Vec<u64> = fields.iter().map(|&(_, v)| v).collect();
        let text = fields
            .iter()
            .map(|(name, value)| format!("{name}={value:x}"))
            .collect::<Vec<_>>()
            .join(" ");
        let mut keys = self.keys.lock().unwrap();
        let keys = &mut *keys;
        if keys.sites.len() >= 1 << 16 && !keys.sites.contains_key(&key.site()) {
            keys.sites.clear();
        }
        let (last, seen) = keys.sites.entry(key.site()).or_default();
        let mut diff = Vec::new();
        if !last.is_empty() {
            for ((name, new), old) in fields.iter().zip(last.iter()) {
                if new != old {
                    diff.push(format!("{name} {old:x}->{new:x}"));
                    *keys.changes.entry(name).or_default() += 1;
                }
            }
        }
        if seen.len() < 4096 {
            seen.insert(values.clone());
        }
        *last = values;
        (text, diff.join(", "))
    }

    /// A snapshot, resetting every counter and ring when `take`.
    pub(crate) fn snapshot(&self, take: bool, table: (usize, usize, usize)) -> ReliftDiagnostics {
        let read = |c: &AtomicU64| {
            if take {
                c.swap(0, Relaxed)
            } else {
                c.load(Relaxed)
            }
        };
        let mut lifts = Vec::new();
        for class in ProofClass::ALL {
            for cause in ReliftCause::ALL {
                let n = read(&self.lifts[class as usize][cause.index()]);
                if n != 0 {
                    lifts.push((class, cause, n));
                }
            }
        }
        let live = self.live.each_ref().map(read);
        ReliftDiagnostics {
            lifts,
            uncached: self.uncached.each_ref().map(read),
            live: LiveFieldStats {
                calls: live[Live::Calls as usize],
                inactive: live[Live::Inactive as usize],
                plain: live[Live::Plain as usize],
                snapshots: live[Live::Snapshots as usize],
                slow: live[Live::Slow as usize],
                probes: live[Live::Probes as usize],
                live_lifts: live[Live::Lowered as usize],
                promotions: live[Live::Promotions as usize],
                absorbed: live[Live::Absorbed as usize],
                shared_covered: live[Live::SharedCovered as usize],
                counted: live[Live::Counted as usize],
                count_clears: live[Live::CountClears as usize],
                refusals: LiveRefusal::ALL
                    .into_iter()
                    .map(|r| (r, read(&self.refusals[r.index()])))
                    .filter(|&(_, n)| n != 0)
                    .collect(),
                demotions: LiveDemotion::ALL
                    .into_iter()
                    .map(|d| (d, read(&self.demotions[d.index()])))
                    .filter(|&(_, n)| n != 0)
                    .collect(),
                live_fields: table.0,
                counted_ranges: table.1,
                refused_ranges: table.2,
            },
            recent: ring(&self.recent, take),
            refused: ring(&self.refused, take),
            contexts: {
                let mut c = self.contexts.lock().unwrap();
                let mut stats = c.stats.clone();
                stats.created_by_field = c
                    .created_by_field
                    .iter()
                    .map(|(k, v)| (k.to_string(), *v))
                    .collect();
                stats.recent = c.recent.iter().cloned().collect();
                if take {
                    *c = Contexts::default();
                }
                stats
            },
            keys: {
                let mut k = self.keys.lock().unwrap();
                let mut top: Vec<(u64, usize)> = k
                    .sites
                    .iter()
                    .map(|(&s, (_, seen))| (s, seen.len()))
                    .collect();
                top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                let stats = KeyStats {
                    sites: k.sites.len(),
                    keys: k.sites.values().map(|(_, seen)| seen.len()).sum(),
                    top: top.into_iter().take(8).collect(),
                    changes_by_field: k.changes.iter().map(|(n, v)| (n.to_string(), *v)).collect(),
                };
                if take {
                    *k = Keys::default();
                }
                stats
            },
            tlb: TlbStats {
                advances: TlbAdvance::ALL
                    .into_iter()
                    .zip(&self.tlb)
                    .map(|(a, n)| (a, read(n)))
                    .filter(|&(_, n)| n != 0)
                    .collect(),
                revalidated: read(&self.revalidated[0]),
                revalidation_failed: read(&self.revalidated[1]),
            },
        }
    }
}

/// Live-field statistics of one VM (xj-to9.46, `vm/legacy/live.rs`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LiveFieldStats {
    /// Instructions proof lifts decoded through the live-field path.
    pub calls: u64,
    /// Of `calls`: before any small code store was counted (plain lift).
    pub inactive: u64,
    /// Of `calls`: holding no live field and no patched range (plain lift).
    pub plain: u64,
    /// Table snapshots (one table lock per lift and code page).
    pub snapshots: u64,
    /// Of `calls`: decoded their fields (a live or patched range in them).
    pub slow: u64,
    /// Exactness probes run (three lifts each).
    pub probes: u64,
    /// Instructions lifted reading a field live.
    pub live_lifts: u64,
    pub promotions: u64,
    /// Stores absorbed: no proof dropped.
    pub absorbed: u64,
    /// Stores inside a live field that a shared unit covered (not absorbed).
    pub shared_covered: u64,
    /// Small code stores counted toward a promotion.
    pub counted: u64,
    /// Clears of the counted-store table (more than 1,024 ranges).
    pub count_clears: u64,
    pub refusals: Vec<(LiveRefusal, u64)>,
    pub demotions: Vec<(LiveDemotion, u64)>,
    /// Table sizes now: live fields, counted ranges, refused ranges.
    pub live_fields: usize,
    pub counted_ranges: usize,
    pub refused_ranges: usize,
}

/// Relift diagnostics of one VM, from [`crate::Vm::relift_diagnostics`].
/// Its `Display` prints every nonzero counter and both rings.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReliftDiagnostics {
    /// Proof lifts (each lift attempt) by class and cause; nonzero only.
    /// Their sum is the `lifts` cost-probe counter of the vCPUs' threads.
    pub lifts: Vec<(ProofClass, ReliftCause, u64)>,
    /// Lifts executed once without a proof (source not publishable), by
    /// [`ProofClass`] index.
    pub uncached: [u64; 3],
    pub live: LiveFieldStats,
    /// The last 64 lifts, oldest first.
    pub recent: Vec<ReliftRecord>,
    /// The last 64 refused fields, oldest first.
    pub refused: Vec<RefusalRecord>,
    /// Context identities: switches, creations by changed stamp field,
    /// evictions.
    pub contexts: ContextStats,
    /// Distinct proof keys per entry address, and the key fields that
    /// changed between lifts at one address.
    pub keys: KeyStats,
    /// Paged TLB generation advances and proof revalidations.
    pub tlb: TlbStats,
}

impl ReliftDiagnostics {
    /// Lifts of `class` (all classes when `None`) with `cause`.
    pub fn lifts(&self, class: Option<ProofClass>, cause: ReliftCause) -> u64 {
        self.lifts
            .iter()
            .filter(|(c, k, _)| class.is_none_or(|class| class == *c) && *k == cause)
            .map(|(.., n)| n)
            .sum()
    }
    /// Every lift of `class` (all classes when `None`).
    pub fn total(&self, class: Option<ProofClass>) -> u64 {
        self.lifts
            .iter()
            .filter(|(c, ..)| class.is_none_or(|class| class == *c))
            .map(|(.., n)| n)
            .sum()
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

impl fmt::Display for ReliftDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "relift diagnostics: {} lifts", self.total(None))?;
        for (class, cause, n) in &self.lifts {
            writeln!(f, "  lifts {class:?} {cause:?}: {n}")?;
        }
        for class in ProofClass::ALL {
            if self.uncached[class as usize] != 0 {
                writeln!(f, "  uncached {class:?}: {}", self.uncached[class as usize])?;
            }
        }
        let l = &self.live;
        writeln!(
            f,
            "  live: calls {} inactive {} plain {} snapshots {} slow {} probes {} \
             live-lifts {} promotions {} absorbed {} shared-covered {} counted {} \
             count-clears {}; table live {} counted {} refused {}",
            l.calls,
            l.inactive,
            l.plain,
            l.snapshots,
            l.slow,
            l.probes,
            l.live_lifts,
            l.promotions,
            l.absorbed,
            l.shared_covered,
            l.counted,
            l.count_clears,
            l.live_fields,
            l.counted_ranges,
            l.refused_ranges
        )?;
        for (why, n) in &l.refusals {
            writeln!(f, "  refused {why:?}: {n}")?;
        }
        for (why, n) in &l.demotions {
            writeln!(f, "  demoted {why:?}: {n}")?;
        }
        for r in &self.recent {
            write!(
                f,
                "  relift {:?} {:?} ip {:#x}",
                r.class, r.cause, r.block_ip
            )?;
            if let Some(bus) = r.block_bus {
                write!(f, " bus {bus:#x}")?;
            }
            if let Some((at, len)) = r.write {
                write!(f, " write {at:#x}+{len}")?;
            }
            if let Some((at, bytes, text)) = &r.instruction {
                write!(f, " insn {at:#x} [{}] {text}", hex(bytes))?;
            }
            write!(f, " key {{{}}}", r.key)?;
            if !r.key_diff.is_empty() {
                write!(f, " changed {{{}}}", r.key_diff)?;
            }
            writeln!(f)?;
        }
        let c = &self.contexts;
        writeln!(
            f,
            "  contexts: switches {} recent-hits {} retained-hits {} created {} evicted {} invalidations {}",
            c.switches, c.recent_hits, c.retained_hits, c.created, c.evicted, c.invalidations
        )?;
        for (field, n) in &c.created_by_field {
            writeln!(f, "  context created, {field} changed: {n}")?;
        }
        for line in &c.recent {
            writeln!(f, "  {line}")?;
        }
        let k = &self.keys;
        writeln!(
            f,
            "  keys: {} entry addresses, {} distinct keys",
            k.sites, k.keys
        )?;
        for (site, n) in &k.top {
            writeln!(f, "  keys at {site:#x}: {n}")?;
        }
        for (field, n) in &k.changes_by_field {
            writeln!(f, "  key changed at one address, {field}: {n}")?;
        }
        let t = &self.tlb;
        write!(
            f,
            "  tlb: revalidated {} failed {}",
            t.revalidated, t.revalidation_failed
        )?;
        for (site, n) in &t.advances {
            write!(f, " {site:?} {n}")?;
        }
        writeln!(f)?;
        for r in &self.refused {
            let (at, bytes, text) = &r.instruction;
            writeln!(
                f,
                "  refusal {:?} patched {:#x}+{} insn {at:#x} [{}] {text}",
                r.reason,
                r.patched.0,
                r.patched.1,
                hex(bytes)
            )?;
        }
        Ok(())
    }
}

/// Classify a recorded store `write` (`[lo, hi)` on the bus) into a dropped
/// proof's block, decoded from its current bytes at `sources` (bus ranges
/// `[lo, hi)` in fetch order) from `ip`; also the containing instruction.
#[allow(clippy::type_complexity)]
pub(crate) fn classify(
    vm: &crate::Vm,
    sources: &[(u64, u64)],
    write: (u64, u64),
    ip: u32,
    context: crate::legacy::MachineContext,
) -> Option<(StoreClass, Option<(u64, Vec<u8>, String)>)> {
    let mut bytes = Vec::new();
    let mut at = None;
    let mut bus = Vec::new();
    for &(lo, hi) in sources {
        let start = bytes.len();
        bytes.resize(start + (hi - lo) as usize, 0);
        vm.mem.read_bytes(lo, &mut bytes[start..]).ok()?;
        if (lo..hi).contains(&write.0) {
            at = Some(start + (write.0 - lo) as usize);
        }
        bus.extend(lo..hi);
    }
    let Some(at) = at else {
        return Some((StoreClass::Outside, None));
    };
    let end = at + (write.1 - write.0) as usize;
    let bits = context.default_operand_bits;
    let mut decoder = iced_x86::Decoder::with_ip(
        u32::from(bits),
        &bytes,
        u64::from(ip),
        crate::model::decode::decoder_options_for_generation(context.generation),
    );
    while decoder.can_decode() {
        let start = decoder.position();
        let insn = decoder.decode();
        let len = insn.len();
        if insn.is_invalid() || at >= start + len {
            if insn.is_invalid() {
                break;
            }
            continue;
        }
        if at < start {
            break;
        }
        let code = &bytes[start..start + len];
        let inside =
            |offset: usize, size: usize| start + offset <= at && end <= start + offset + size;
        let class = if end > start + len {
            StoreClass::Opcode
        } else if crate::lift::legacy::live_fields(code, ip.wrapping_add(start as u32), context)
            .is_some_and(|f| {
                f.fields
                    .iter()
                    .flatten()
                    .any(|&(o, n)| inside(usize::from(o), usize::from(n)))
            })
        {
            StoreClass::LiveField
        } else {
            let offsets = decoder.get_constant_offsets(&insn);
            let mut fields = Vec::new();
            if offsets.has_immediate() {
                fields.push((offsets.immediate_offset(), offsets.immediate_size()));
            }
            if offsets.has_immediate2() {
                fields.push((offsets.immediate_offset2(), offsets.immediate_size2()));
            }
            if offsets.has_displacement() {
                fields.push((offsets.displacement_offset(), offsets.displacement_size()));
            }
            if fields.iter().any(|&(o, n)| inside(o, n)) {
                StoreClass::OtherField
            } else {
                StoreClass::Opcode
            }
        };
        let shown = &code[..len.min(16)];
        let (_, text) = text(shown, ip.wrapping_add(start as u32), bits);
        return Some((class, Some((bus[start], shown.to_vec(), text))));
    }
    Some((StoreClass::Outside, None))
}

impl From<crate::lift::legacy::ProbeRefusal> for ProbeFailure {
    fn from(r: crate::lift::legacy::ProbeRefusal) -> Self {
        use crate::lift::legacy::ProbeRefusal as R;
        match r {
            R::Length => Self::Length,
            R::Extension => Self::Extension,
            R::NoLiveRead => Self::NoLiveRead,
            R::Trap => Self::Trap,
            R::ValueDependent => Self::ValueDependent,
            R::LiftFailed => Self::LiftFailed,
        }
    }
}
