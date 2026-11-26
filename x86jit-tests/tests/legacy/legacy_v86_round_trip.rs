//! xj-to9.42: paged V86 at IOPL 0, a ring-0 #GP monitor and IRETD back.
//! Deliveries, IRETs, refused instructions and read-modify-writes whose every
//! access hits RAM through the TLB skip the two-phase transaction; each case
//! here runs twice, with those lanes and on the transaction path alone
//! (`without_paging_fast_lanes_for_test`), and must leave the same exits,
//! registers, flags, CR2, frames, page tables and TLB contents.
//!
//! Sources: SDM Vol 3B §23.3.1.1 pp. 23-12–13 (the nine-slot ring-0 exit
//! frame and the null data segments), Vol 2A §3.3 INT pp. 3-472–477 and IRET
//! pp. 3-491–497; 80386 PRM §15.3.2 pp. 15-6–7, §9.8 pp. 9-11–18, §5.2.4.3
//! p. 5-10; Intel486 PRM §23.3.2 pp. 23-7–8 and §4.1.3 p. 4-7 (CR0.WP).
use x86jit_core::legacy::{SegmentCache, SystemSegment};
use x86jit_core::{CpuModel, CpuState, Prot, RegionKind, StepOptions, Vcpu, Vm, VmConfig};

const MEMORY: usize = 0x60000;
const PD: u64 = 0x1000;
const PT: u64 = 0x2000;
const IDT: u64 = 0x3000;
const GDT: u64 = 0x3800;
const TSS: u32 = 0x5000;
const COUNTER: u64 = 0x6000;
const MONITOR: u32 = 0x9000;
const RETURN: u32 = 0x9100;
const STOP: u32 = 0x9200;
/// The ring-0 service behind INT 22h (a DPL 3 gate).
const SERVICE: u32 = 0x9300;
/// The handler behind #UD.
const UD: u32 = 0x9400;
/// A ring-3 handler (CS 33h, a DPL 3 code segment).
const USER_HANDLER: u32 = 0x30000;
/// ADD DWORD [ESP+4],1; ADD ESP,4; IRETD: skip a one-byte faulting
/// instruction and drop the error code.
const SKIP_ONE: [u8; 9] = [0x83, 0x44, 0x24, 0x04, 0x01, 0x83, 0xc4, 0x04, 0xcf];
const V86_CS: u16 = 0x1000;
const V86_DS: u16 = 0x5100;
const V86_SS: u16 = 0x4000;
/// The paged pages the V86 task may touch (CPL 3 accesses).
const USER: [u32; 4] = [0x10, 0x47, 0x51, 0x30];

/// PUSH EAX; MOV AX,10h; MOV DS,AX; INC DWORD [6000h];
/// ADD DWORD [ESP+8],2; POP EAX; ADD ESP,4; then `before_iret`; IRETD.
fn monitor(before_iret: &[u8]) -> Vec<u8> {
    let mut code = vec![
        0x50, 0x66, 0xb8, 0x10, 0x00, 0x8e, 0xd8, 0xff, 0x05, 0x00, 0x60, 0x00, 0x00, 0x83, 0x44,
        0x24, 0x08, 0x02, 0x58, 0x83, 0xc4, 0x04,
    ];
    code.extend_from_slice(before_iret);
    code.push(0xcf);
    code
}

/// IN AL,80h; PUSH AX; POP AX; INC BX; JMP to the IN.
const V86_IN: [u8; 7] = [0xe4, 0x80, 0x50, 0x58, 0x43, 0xeb, 0xf9];

struct World {
    model: CpuModel,
    paged: bool,
    v86: Vec<u8>,
    monitor: Vec<u8>,
    /// Ring-0 ESP from the TSS.
    esp0: u32,
    /// PTEs of the ring-0 stack pages 6 and 7.
    stack_ptes: [u32; 2],
    /// CR0.WP (Intel486 PRM §4.1.3 p. 4-7).
    wp: bool,
    eflags: u64,
    tr_limit: u32,
    /// SS0 selector: 10h flat, 20h with limit 6FFFh.
    ss0: u16,
    /// Vector 13's gate attributes (high word bits 15:8).
    gp_gate: u16,
    /// INT 21h's gate DPL.
    int_dpl: u16,
    /// Vector 13's handler: (CS, offset).
    gp_target: (u16, u32),
    /// The ring-0 code segment's access byte (9Bh accessed, 9Ah clear).
    code_access: u8,
    /// The ring-0 data segment's access byte (93h accessed, 92h clear).
    data_access: u8,
    /// An I/O permission bitmap byte for ports 80h-87h (TSS+78h; the TSS
    /// limit grows to 7Fh to hold it and the byte after it).
    bitmap: Option<u8>,
    /// A page mapped as a device instead of RAM.
    mmio_page: Option<u32>,
    /// Start in protected mode at this CPL (flat segments) instead of V86;
    /// the code is `v86` at 10000h.
    protected: Option<u8>,
    /// The page tables (page 2) user-writable.
    user_tables: bool,
    service: Vec<u8>,
    /// Whether some delivery, IRET or refusal completes in a lane.
    lanes: bool,
}

impl World {
    fn new(model: CpuModel, paged: bool) -> Self {
        Self {
            model,
            paged,
            v86: V86_IN.to_vec(),
            monitor: monitor(&[]),
            esp0: 0x7000,
            stack_ptes: [0x6063, 0x7063],
            wp: false,
            eflags: 0x20002,
            tr_limit: 0x67,
            ss0: 0x10,
            gp_gate: 0x8e00,
            int_dpl: 0,
            gp_target: (0x08, MONITOR),
            code_access: 0x9b,
            data_access: 0x93,
            bitmap: None,
            mmio_page: None,
            protected: None,
            user_tables: false,
            service: vec![0xcf],
            lanes: true,
        }
    }

    fn on_device(&self, addr: u64) -> bool {
        self.mmio_page
            .is_some_and(|page| addr >> 12 == u64::from(page))
    }

    fn build(&self) -> (Vm, Vcpu) {
        let mut vm = Vm::new(VmConfig::flat(MEMORY as u64));
        vm.set_cpu_model(self.model);
        if let Some(page) = self.mmio_page {
            let at = page as usize * 0x1000;
            vm.map(0, at, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(at as u64, 0x1000, Prot::RW, RegionKind::Trap)
                .unwrap();
            vm.map(
                at as u64 + 0x1000,
                MEMORY - at - 0x1000,
                Prot::RWX,
                RegionKind::Ram,
            )
            .unwrap();
        } else {
            vm.map(0, MEMORY, Prot::RWX, RegionKind::Ram).unwrap();
        }
        // The device page holds nothing; its tables are read as device data.
        let put = |addr: u64, bytes: &[u8]| {
            if !self.on_device(addr) {
                vm.write_bytes(addr, bytes).unwrap();
            }
        };
        if self.paged {
            put(PD, &(PT as u32 | 0x27).to_le_bytes());
            for page in 0..(MEMORY / 0x1000) as u32 {
                // A and D set on every page but the two stack pages.
                let entry = match page {
                    6 => self.stack_ptes[0],
                    7 => self.stack_ptes[1],
                    2 if self.user_tables => 0x2067,
                    _ if USER.contains(&page) => (page << 12) | 0x67,
                    _ => (page << 12) | 0x63,
                };
                put(PT + u64::from(4 * page), &entry.to_le_bytes());
            }
        }
        // Flat ring-0 code and data, and a data segment limited to 6FFFh, all
        // accessed (SDM Vol 3A §3.4.5).
        for (selector, descriptor) in [
            (
                0x08,
                0x00cf_0000_0000_ffffu64 | u64::from(self.code_access) << 40,
            ),
            (
                0x10,
                0x00cf_0000_0000_ffff | u64::from(self.data_access) << 40,
            ),
            // Flat ring-3 code and data (DPL 3).
            (0x30, 0x00cf_fb00_0000_ffff),
            (0x38, 0x00cf_f300_0000_ffff),
            (0x20, 0x0040_9300_0000_6fff),
            // Busy 386 TSSs: the current one at 5000h and the task it was
            // nested in at 8800h, for an IRETD with NT set (80386 PRM §7.6
            // p. 7-10, §17 IRET p. 17-74).
            (0x18, 0x0000_8b00_5000_0067),
            (0x28, 0x0000_8b00_8800_0067),
        ] {
            put(GDT + selector, &descriptor.to_le_bytes());
        }
        let gate = |cs: u16, offset: u32, attributes: u16| {
            u64::from(offset & 0xffff)
                | (u64::from(cs) << 16)
                | (u64::from(attributes) << 32)
                | (u64::from(offset >> 16) << 48)
        };
        for vector in 0..0x23u64 {
            let entry = match vector {
                13 => gate(self.gp_target.0, self.gp_target.1, self.gp_gate),
                6 => gate(8, UD, 0x8e00),
                1 | 2 | 0x20 => gate(8, RETURN, 0x8e00),
                0x21 => gate(8, RETURN, 0x8e00 | (self.int_dpl << 13)),
                0x22 => gate(8, SERVICE, 0xee00),
                _ => gate(8, STOP, 0x8e00),
            };
            put(IDT + vector * 8, &entry.to_le_bytes());
        }
        put(u64::from(TSS) + 4, &self.esp0.to_le_bytes());
        put(u64::from(TSS) + 8, &u32::from(self.ss0).to_le_bytes());
        put(u64::from(TSS) + 0x66, &0x68u16.to_le_bytes());
        put(u64::from(TSS), &0x28u32.to_le_bytes());
        if let Some(bits) = self.bitmap {
            // 80386 PRM §8.3.2 pp. 8-5–6: ports 80h-87h are byte 10h of the
            // map; two bytes are always read.
            put(u64::from(TSS) + 0x78, &[bits, 0xff]);
        }
        // The outer task resumes at the HLT loop on its own stack, with the
        // same directory (80386 PRM §7.1 Fig. 7-1 p. 7-2).
        for (field, value) in [
            (0x1c, PD as u32),
            (0x20, STOP),
            (0x24, 2),
            (0x38, 0x7800),
            (0x48, 0x10),
            (0x4c, 0x08),
            (0x50, 0x10),
            (0x54, 0x10),
            (0x66, 0x68),
        ] {
            put(0x8800 + field, &value.to_le_bytes());
        }
        put(u64::from(MONITOR), &self.monitor);
        put(u64::from(RETURN), &[0xcf]);
        put(u64::from(STOP), &[0xf4, 0xeb, 0xfd]);
        put(u64::from(SERVICE), &self.service);
        // ADD DWORD [ESP],2; IRETD: past a two-byte UD2.
        put(u64::from(UD), &[0x83, 0x04, 0x24, 0x02, 0xcf]);
        put(u64::from(USER_HANDLER), &SKIP_ONE);
        put(u64::from(V86_CS) << 4, &self.v86);
        let mut cpu = vm.new_vcpu();
        cpu.cpu = CpuState::legacy_real_entry([V86_CS, V86_DS, V86_DS, V86_SS, 0, 0]);
        cpu.cpu.legacy.cr0 = if self.paged { 0x8000_0001 } else { 1 } | u32::from(self.wp) << 16;
        cpu.cpu.legacy.cr3 = PD as u32;
        cpu.cpu.legacy.cpl = 3;
        // TF, IOPL, NT, RF and VM; IF lives with the arithmetic flags.
        cpu.cpu.rflags_system = self.eflags & 0x37100;
        cpu.cpu.flags.if_ = self.eflags & 0x200 != 0;
        cpu.cpu.legacy.gdtr.base = GDT as u32;
        cpu.cpu.legacy.gdtr.limit = 0x3f;
        cpu.cpu.legacy.idtr.base = IDT as u32;
        cpu.cpu.legacy.idtr.limit = 0x7ff;
        cpu.cpu.legacy.tr = SystemSegment {
            selector: 0x18,
            cache: SegmentCache {
                base: TSS,
                limit: if self.bitmap.is_some() {
                    0x7f
                } else {
                    self.tr_limit
                },
                attributes: 0x8b,
                ..Default::default()
            },
            ..Default::default()
        };
        cpu.cpu.gpr[4] = 0x8000;
        if let Some(cpl) = self.protected {
            // Flat segments at `cpl`, FS and GS null (SDM Vol 3A §3.4.3).
            let flat = |access: u16| SegmentCache {
                base: 0,
                limit: u32::MAX,
                attributes: 0xc000 | access,
                ..Default::default()
            };
            let null = SegmentCache {
                unusable: 1,
                ..Default::default()
            };
            let (code, data, cs, ds) = if cpl == 3 {
                (0xfb, 0xf3, 0x33, 0x3b)
            } else {
                (0x9b, 0x93, 0x08, 0x10)
            };
            cpu.cpu.legacy.segments = [flat(code), flat(data), flat(data), flat(data), null, null];
            [
                cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs,
            ] = [cs, ds, ds, ds, 0, 0];
            cpu.cpu.legacy.cpl = cpl;
            cpu.cpu.rflags_system &= !0x20000;
            cpu.cpu.rip = u64::from(V86_CS) << 4;
            cpu.cpu.gpr[4] = if cpl == 3 { 0x48000 } else { 0x7800 };
        }
        cpu.set_a20_enabled(true);
        (vm, cpu)
    }
}

/// What one slice of a run left: its result, the whole CPU state (CR2, the
/// segment caches, CPL and every flag included) and all of memory (frames,
/// page-table A/D bits, the monitor's counter).
#[derive(PartialEq)]
struct Snapshot {
    result: String,
    retired: u64,
    cpu: String,
    tlb: String,
    /// The guest-visible registers alone, to compare paged with unpaged.
    registers: String,
    memory: Vec<u8>,
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} retired {} {}", self.result, self.retired, self.cpu)
    }
}

fn snapshot(vm: &Vm, cpu: &Vcpu, result: String) -> Snapshot {
    let mut memory = vec![0; MEMORY];
    // A device page reads as zeros.
    for (page, bytes) in memory.chunks_mut(0x1000).enumerate() {
        let _ = vm.read_bytes(page as u64 * 0x1000, bytes);
    }
    Snapshot {
        result,
        retired: cpu.retired_instructions(),
        cpu: format!("{:?}", cpu.cpu),
        tlb: format!("{:x?}", cpu.paging_tlb_for_test()),
        registers: format!(
            "{:x?}",
            (
                cpu.cpu.gpr,
                cpu.cpu.rip,
                [cpu.cpu.cs, cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs],
                cpu.cpu.rflags_system,
                cpu.cpu.flags,
                cpu.cpu.legacy.segments,
                cpu.cpu.legacy.cpl,
                cpu.cpu.legacy.cr2,
            )
        ),
        memory,
    }
}

/// Run `slices` (step budgets, each optionally after an NMI edge or an IRQ)
/// on a fresh world, recording every slice.
fn trace(world: &World, slices: &[(u64, Event)]) -> Vec<Snapshot> {
    let (vm, mut cpu) = world.build();
    let mut out = Vec::new();
    for &(budget, event) in slices {
        match event {
            Event::None => {}
            Event::Nmi => cpu.raise_nmi().unwrap(),
            Event::Irq(vector) => cpu.inject_irq(vector),
        }
        let result = cpu.run_steps(&vm, budget, StepOptions::default());
        out.push(snapshot(&vm, &cpu, format!("{result:?}")));
    }
    out
}

#[derive(Clone, Copy)]
enum Event {
    None,
    Nmi,
    Irq(u8),
}

/// The lanes and the transaction path agree on every slice.
/// Every case takes at least one lane, and none with the lanes off.
fn parity(world: &World, slices: &[(u64, Event)]) -> Vec<Snapshot> {
    let hits = x86jit_core::interp::paging_fast_lane_hits_for_test;
    let before = hits();
    let fast = trace(world, slices);
    let lanes = hits() - before;
    let slow = x86jit_core::interp::without_paging_fast_lanes_for_test(|| trace(world, slices));
    assert_eq!(
        hits() - before,
        lanes,
        "{:?}: a lane ran while off",
        world.model
    );
    if world.paged {
        assert!(lanes > 0 || !world.lanes, "{:?}: no lane ran", world.model);
    }
    for (index, (fast, slow)) in fast.iter().zip(&slow).enumerate() {
        assert_eq!(
            fast.result, slow.result,
            "{:?} slice {index}: result",
            world.model
        );
        assert_eq!(fast.cpu, slow.cpu, "{:?} slice {index}: CPU", world.model);
        assert_eq!(fast.tlb, slow.tlb, "{:?} slice {index}: TLB", world.model);
        if fast.memory != slow.memory {
            let at = fast
                .memory
                .iter()
                .zip(&slow.memory)
                .position(|(a, b)| a != b)
                .unwrap();
            panic!(
                "{:?} slice {index}: memory differs first at {at:#x}",
                world.model
            );
        }
        assert_eq!(
            fast.retired, slow.retired,
            "{:?} slice {index}",
            world.model
        );
    }
    fast
}

const MODELS: [CpuModel; 2] = [CpuModel::I80386, CpuModel::I80486Dx];

fn counter(snapshot: &Snapshot) -> u32 {
    let at = COUNTER as usize;
    u32::from_le_bytes(snapshot.memory[at..at + 4].try_into().unwrap())
}

#[test]
fn round_trips_match_the_transaction_path_and_the_unpaged_machine() {
    // One period: IN (#GP, 0 steps), the monitor's eight, then PUSH, POP,
    // INC and JMP: twelve steps. The odd budgets stop inside the monitor
    // and inside V86, so slices resume on both sides of each transfer.
    let slices: Vec<_> = [12, 12, 5, 7, 3, 9, 12, 1, 11, 120]
        .into_iter()
        .map(|budget| (budget, Event::None))
        .collect();
    for model in MODELS {
        let paged = parity(&World::new(model, true), &slices);
        let unpaged = trace(&World::new(model, false), &slices);
        assert_eq!(counter(paged.last().unwrap()), 16, "{model:?}");
        // Registers, selectors and caches after each slice, and the frames
        // and counter: paging changes none of them here (SDM Vol 2A IRET
        // p. 3-493 restores all six V86 selectors).
        for (index, (paged, unpaged)) in paged.iter().zip(&unpaged).enumerate() {
            assert_eq!(paged.result, unpaged.result, "{model:?} slice {index}");
            assert_eq!(
                paged.registers, unpaged.registers,
                "{model:?} slice {index}"
            );
            assert_eq!(
                paged.memory[0x6f00..0x8000],
                unpaged.memory[0x6f00..0x8000],
                "{model:?} slice {index}: ring-0 frames"
            );
            assert_eq!(counter(paged), counter(unpaged), "{model:?} slice {index}");
        }
    }
}

#[test]
fn a_frame_fault_at_every_slot_matches_the_transaction_path() {
    // ESP0 = 7000h + 4(k - 1): the first k - 1 slots of the ten-slot #GP
    // frame lie on page 7 and slot k on page 6, whose PTE is not present,
    // read-only supervisor, user, user read-only, clean (D clear, which the
    // lanes leave to the transaction path: 80386 PRM §5.2.4.3 p. 5-10) or
    // not yet accessed (A clear, set by the walk). On the 486 with CR0.WP a
    // read-only page faults a supervisor write (Intel486 PRM §4.1.3 p. 4-7).
    // The monitor starts with a NOP and the run goes one step at a time, so
    // a slice ends after each delivery, before any other store to its page.
    let mut slices = vec![(1, Event::None); 30];
    slices.push((40, Event::None));
    for model in MODELS {
        for pte in [
            0x6000, 0x6061, 0x6065, 0x6067, 0x6023, 0x6027, 0x6003, 0x6043,
        ] {
            for wp in [false, true] {
                if wp && model == CpuModel::I80386 {
                    continue;
                }
                for slot in 1..=10 {
                    let mut world = World::new(model, true);
                    world.monitor.insert(0, 0x90);
                    world.esp0 = 0x7000 + 4 * (slot - 1);
                    world.stack_ptes[0] = pte;
                    world.wp = wp;
                    parity(&world, &slices);
                }
            }
        }
    }
}

#[test]
fn ss0_tss_and_gate_faults_match_the_transaction_path() {
    let slices = [(12, Event::None), (30, Event::None)];
    for model in MODELS {
        let cases: [&dyn Fn(&mut World); 5] = [
            // SS0 limit 6FFFh: the frame below ESP0 7000h leaves it
            // (#SS(SS0), SDM Vol 2A INT p. 3-476).
            &|w| {
                w.ss0 = 0x20;
            },
            // The TSS ends before SS0 (#TS(TR), SDM p. 3-473).
            &|w| w.tr_limit = 0x08,
            // Vector 13's gate not present (#NP), a trap gate, a 16-bit
            // interrupt gate (refused before any effect) and a task gate.
            &|w| w.gp_gate = 0x0e00,
            &|w| w.gp_gate = 0x8f00,
            &|w| w.gp_gate = 0x8600,
        ];
        for case in cases {
            let mut world = World::new(model, true);
            case(&mut world);
            parity(&world, &slices);
        }
        let mut world = World::new(model, true);
        world.gp_gate = 0x8500;
        parity(&world, &slices);
    }
}

#[test]
fn int_n_through_a_gate_dpl_matches_the_transaction_path() {
    // V86 at IOPL 3: INT 21h reaches the IDT, whose gate DPL is checked
    // against CPL 3 (#GP(21h*8+2) at DPL 0; SDM Vol 2A INT p. 3-475).
    // INT 21h; INC BX; JMP to the INT.
    for model in MODELS {
        for dpl in [0, 3] {
            let mut world = World::new(model, true);
            world.v86 = vec![0xcd, 0x21, 0x43, 0xeb, 0xfb];
            world.eflags = 0x23002;
            world.int_dpl = dpl;
            parity(&world, &[(9, Event::None), (40, Event::None)]);
        }
    }
}

#[test]
fn bad_iretd_frames_match_the_transaction_path() {
    // Each patch runs before the monitor's IRETD.
    let patches: [&[u8]; 4] = [
        // MOV DWORD [ESP],10000h: EIP beyond FFFFh is #GP(0) (SDM Vol 2A
        // IRET p. 3-493).
        &[0xc7, 0x04, 0x24, 0x00, 0x00, 0x01, 0x00],
        // MOV DWORD [201Ch],0; MOV EAX,CR3; MOV CR3,EAX: page 7, which
        // holds the frame's selector slots, is no longer present (#PF).
        &[
            0xc7, 0x05, 0x1c, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x20, 0xd8, 0x0f,
            0x22, 0xd8,
        ],
        // AND DWORD [ESP+8],FFFDFFFFh: VM clear, a protected IRETD to CS
        // 1000h (outward, faults on the selector).
        &[0x81, 0x64, 0x24, 0x08, 0xff, 0xff, 0xfd, 0xff],
        // PUSHFD; OR DWORD [ESP],4000h; POPFD: NT set, a return to the
        // task at 8800h, which saves this one first.
        &[0x9c, 0x81, 0x0c, 0x24, 0x00, 0x40, 0x00, 0x00, 0x9d],
    ];
    for model in MODELS {
        for (index, patch) in patches.iter().enumerate() {
            let mut world = World::new(model, true);
            world.monitor = monitor(patch);
            if index == 1 {
                // Slots 1-4 (GS, FS, DS, ES) on page 7.
                world.esp0 = 0x7010;
            }
            parity(&world, &[(12, Event::None), (40, Event::None)]);
        }
    }
}

#[test]
fn tf_nmi_and_irq_around_the_round_trip_match_the_transaction_path() {
    for model in MODELS {
        // TF in V86: a #DB trap after every V86 instruction, delivered to
        // ring 0 and returned by IRETD; the monitor's IRETD restores TF
        // (80386 PRM §12.3.1.4 p. 12-8).
        let mut world = World::new(model, true);
        world.eflags = 0x20102;
        parity(
            &world,
            &[(5, Event::None), (7, Event::None), (60, Event::None)],
        );
        // IF in V86: an IRQ taken in V86, and one queued while the monitor
        // runs with IF clear (interrupt gate) and taken after its IRETD.
        let mut world = World::new(model, true);
        world.eflags = 0x20202;
        parity(
            &world,
            &[
                (2, Event::Irq(0x20)),
                (3, Event::None),
                (3, Event::Irq(0x20)),
                (30, Event::None),
            ],
        );
        // NMI edges in V86 and inside the monitor (vector 2 to IRETD).
        let world = World::new(model, true);
        parity(
            &world,
            &[
                (2, Event::Nmi),
                (3, Event::None),
                (3, Event::Nmi),
                (30, Event::None),
            ],
        );
    }
}

#[test]
fn monitor_read_modify_writes_match_the_transaction_path() {
    // ESP0 = 8000h keeps the frame on page 7; the monitor's INC DWORD
    // [6000h] is a read-modify-write of page 6, which is read-only (a #PF
    // for the store on the 486 with CR0.WP, Intel486 PRM §4.1.3 p. 4-7),
    // clean (D set by the store, 80386 PRM §5.2.4.3 p. 5-10), not yet
    // accessed, or writable. ADD DWORD [ESP+8],2 updates the frame in place.
    let mut slices = vec![(1, Event::None); 30];
    slices.push((40, Event::None));
    for model in MODELS {
        for pte in [0x6061, 0x6023, 0x6003, 0x6063] {
            for wp in [false, true] {
                let mut world = World::new(model, true);
                world.esp0 = 0x8000;
                world.stack_ptes[0] = pte;
                world.wp = wp;
                parity(&world, &slices);
            }
        }
    }
}

fn stepwise() -> Vec<(u64, Event)> {
    let mut slices = vec![(1, Event::None); 30];
    slices.push((40, Event::None));
    slices
}

#[test]
fn a_stale_tlb_entry_meets_both_paths_alike() {
    // SDM Vol 3A §5.10.2: an edited PTE may go unseen until its entry
    // leaves the TLB. Ring 3 makes its stack page 47h supervisor-only, a
    // ring-0 service (INT 22h) reads it, filling a supervisor entry, and
    // makes it user again without a flush. The CLI that follows is #GP to a
    // ring-3 handler, whose frame goes to that page through the stale
    // entry: #PF on the 80386 (80386 PRM §5.2.5, §6.4 pp. 6-20–22), a fresh
    // walk on the 486 (Intel486 PRM §9.9.14 p. 9-21 rewalks a protection
    // failure). ESP0 8000h keeps the ring-0 frames off 47h's TLB set.
    // MOV DWORD [211Ch],47063h; INT 22h; CLI; INC EBX; JMP to the MOV.
    let code = [
        0xc7, 0x05, 0x1c, 0x21, 0x00, 0x00, 0x63, 0x70, 0x04, 0x00, 0xcd, 0x22, 0xfa, 0x43, 0xeb,
        0xf0,
    ];
    // MOV EAX,[47FF0h]; MOV DWORD [211Ch],47067h; IRETD.
    let service = [
        0xa1, 0xf0, 0x7f, 0x04, 0x00, 0xc7, 0x05, 0x1c, 0x21, 0x00, 0x00, 0x67, 0x70, 0x04, 0x00,
        0xcf,
    ];
    for model in MODELS {
        let mut world = World::new(model, true);
        world.protected = Some(3);
        world.v86 = code.to_vec();
        world.service = service.to_vec();
        world.user_tables = true;
        world.esp0 = 0x8000;
        world.gp_target = (0x33, USER_HANDLER);
        world.gp_gate = 0xee00;
        parity(&world, &stepwise());
    }
}

#[test]
fn protected_deliveries_and_irets_match_the_transaction_path() {
    for model in MODELS {
        // Ring 3 to ring 0 and back: CLI at IOPL 0 is #GP (80386 PRM §8.3
        // pp. 8-4–5), the ring-0 handler skips it and IRETD returns
        // outward (SDM Vol 2A IRET p. 3-495). CLI; INC EBX; JMP to the CLI.
        let mut world = World::new(model, true);
        world.protected = Some(3);
        world.v86 = vec![0xfa, 0x43, 0xeb, 0xfc];
        world.monitor = SKIP_ONE.to_vec();
        parity(&world, &stepwise());
        // The same #GP to a DPL 3 handler: a same-level frame on the ring-3
        // stack (SDM Vol 2A INT p. 3-478).
        world.gp_target = (0x33, USER_HANDLER);
        parity(&world, &stepwise());
        // Ring 0: UD2 is #UD, delivered and returned at the same level.
        // UD2; INC EBX; JMP to the UD2.
        let mut world = World::new(model, true);
        world.protected = Some(0);
        world.v86 = vec![0x0f, 0x0b, 0x43, 0xeb, 0xfb];
        parity(&world, &stepwise());
    }
}

#[test]
fn accessed_bits_bitmaps_devices_and_awkward_frames_match_the_transaction_path() {
    for model in MODELS {
        let cases: [&dyn Fn(&mut World); 10] = [
            // A code or stack descriptor with A clear: the CAS of SDM Vol 3A
            // §3.4.5.1 replays on the transaction path.
            &|w| w.code_access = 0x9a,
            &|w| w.data_access = 0x92,
            // A TSS that holds the bitmap: port 80h denied, then permitted
            // (the IN goes out to the host; 80386 PRM §8.3.2 pp. 8-5–6).
            &|w| w.bitmap = Some(0x01),
            &|w| {
                w.bitmap = Some(0x00);
                w.lanes = false;
            },
            // The IDT and GDT page, the TSS page and the ring-0 stack page
            // on a device: the lanes take RAM only, and the run stops at the
            // first device transfer for the host.
            &|w| {
                w.mmio_page = Some(3);
                w.lanes = false;
            },
            &|w| {
                w.mmio_page = Some(5);
                w.lanes = false;
            },
            &|w| {
                w.esp0 = 0x8000;
                w.mmio_page = Some(7);
                w.lanes = false;
            },
            // The frame on the V86 code page (a registered code page).
            &|w| w.esp0 = 0x10800,
            // ESP0 7002h: the first slot crosses from page 6 into page 7.
            &|w| w.esp0 = 0x7002,
            &|w| w.esp0 = 0x7ffe,
        ];
        for case in cases {
            let mut world = World::new(model, true);
            world.monitor.insert(0, 0x90);
            case(&mut world);
            parity(&world, &stepwise());
        }
    }
}

#[test]
fn a_v86_iret_at_iopl_3_matches_the_transaction_path() {
    // At IOPL 3 a V86 IRET is not sensitive and stays in V86, popping IP, CS
    // and FLAGS from the user stack (SDM Vol 2A IRET pp. 3-491–492).
    // PUSH 2; PUSH CS; PUSH 7; IRET; INC BX; JMP to the start.
    for model in MODELS {
        let mut world = World::new(model, true);
        world.v86 = vec![0x6a, 0x02, 0x0e, 0x68, 0x07, 0x00, 0xcf, 0x43, 0xeb, 0xf6];
        world.eflags = 0x23002;
        parity(&world, &stepwise());
    }
}
