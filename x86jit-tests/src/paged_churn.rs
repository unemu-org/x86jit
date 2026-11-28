//! A paged ring-0 loop that turns the functional TLB over (xj-to9.70), shared by
//! the relift witness (`tests/legacy/legacy_relift_diag.rs`) and the
//! `machine32_paged_churn` measurement. It reproduces the shape of a
//! protected-mode operating system's paged kernel phases: about two dozen
//! block entries per iteration that read 20 data pages of a 4 MiB sweep, more
//! pages than the 256-entry TLB holds, so fills keep replacing the entries the
//! loop's own code was fetched through; an INVLPG of a data page; an
//! equal-value CR3 reload, which flushes the TLB; CR0.TS set
//! before a V86 entry and cleared by CLTS after the exit (FPU ownership); an
//! IRETD into a short V86 body whose IN traps to a ring-0 #GP monitor; and a
//! second flat data selector, so the loop runs under several context stamps.
//!
//! The rotating variants (`setup_with(.., ROTATE)` and `ROTATE_MANY`,
//! xj-to9.75) also model the data-segment reloads of an operating system's
//! 16-bit and V86 transitions: each iteration first loads DS, ES, FS and GS
//! from a table of 32 or 128 entries, DS with one of four flat aliases, ES
//! with a flat alias or a byte-granular data segment, FS and GS with the
//! null selector, a flat alias or a non-flat data segment. The same kernel
//! code then runs under that many data segment sets, Flat32 while ES is flat
//! and Segmented while it is not. `ROTATE` keeps under 64 distinct Segmented
//! stamps; `ROTATE_MANY` (five non-flat ES of eight) cycles through about
//! 240, more than the 64 paged identities retained, so they are evicted.
//!
//! Layout (identity-mapped, every page user/writable): GDT at 500h, IDT at
//! 800h, ring-0 stack below 9000h, TSS at 5000h, kernel code at 10000h, V86
//! code at 20000h (CS 2000h), the #GP monitor at 30000h, V86 data and stack at
//! 40000h (DS = SS = 4000h), page directory 1000h, tables 2000h-3fffh, the
//! selector table at 7100h, and the sweep at 100000h-4fffffh.
use x86jit_core::legacy::{SegmentCache, SystemSegment};
use x86jit_core::{CpuModel, CpuState, Prot, RegionKind, Vcpu, Vm, VmConfig};

/// Guest memory: 8 MiB of RAM, all mapped through two page tables.
pub const MEMORY: u64 = 0x80_0000;
/// The kernel loop's entry point.
pub const KERNEL: u32 = 0x10000;
const V86_CS: u16 = 0x2000;
const MONITOR: u32 = 0x30000;
const V86_DS: u16 = 0x4000;
/// The V86 body's counter word (DS:100h), 3 per iteration.
pub const V86_COUNTER: u64 = 0x40100;
/// The monitor's exit count, one per iteration.
pub const MONITOR_COUNTER: u64 = 0x6000;
const SWEEP: u32 = 0x10_0000;
const SWEEP_END: u32 = 0x50_0000;
/// Data pages each iteration reads, one block each.
pub const READS: u32 = 20;
const STACK: u32 = 0x9000;
/// The rotating variants' selector table: entries of DS, ES, FS and GS, one
/// word each.
const TABLE: u32 = 0x7100;
/// `setup_with` without rotation.
pub const FIXED: u32 = 0;
/// `setup_with` rotating through 32 data segment sets.
pub const ROTATE: u32 = 32;
/// `setup_with` rotating through 128 data segment sets.
pub const ROTATE_MANY: u32 = 128;

/// Entry `i` of a table of `rotations`: DS, ES, FS, GS. 10h, 18h, 28h and
/// 30h are flat data; 38h, 48h, 50h, 58h and 60h byte-granular 32-bit data
/// segments at 7800h, 7900h, 7A00h, 7B00h and 7C00h, 40h a 16-bit one at
/// 7C00h; 0 is the null selector, which FS and GS may hold.
fn rotation(i: u32, rotations: u32) -> [u16; 4] {
    const DS: [u16; 4] = [0x10, 0x18, 0x28, 0x30];
    const ES: [u16; 8] = [0x10, 0x18, 0x28, 0x38, 0x48, 0x50, 0x58, 0x60];
    const FS_GS: [u16; 4] = [0, 0x10, 0x38, 0x40];
    let i = i as usize;
    if rotations == ROTATE_MANY {
        return [
            DS[i % 4],
            ES[(i / 4) % 8],
            FS_GS[(i / 32) % 4],
            FS_GS[(i / 2) % 4],
        ];
    }
    [
        DS[i % 4],
        ES[(i / 4) % 4],
        FS_GS[(i / 16) % 2 + 2 * (i % 2)],
        FS_GS[(i / 2) % 4],
    ]
}

/// The kernel loop: `iterations` in EDI; HLT at ring 0 when done.
fn kernel(rotations: u32) -> Vec<u8> {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    let mut swept = a.create_label();
    a.set_label(&mut top).unwrap();
    if rotations != FIXED {
        // This iteration's data segments, DS last: the table is read
        // through the flat DS the monitor loaded.
        a.mov(eax, edi).unwrap();
        a.and(eax, (rotations - 1) as i32).unwrap();
        for (k, segment) in [(1, es), (2, fs), (3, gs)] {
            a.movzx(ecx, word_ptr(eax * 8 + (TABLE + 2 * k) as i32))
                .unwrap();
            a.mov(segment, ecx).unwrap();
        }
        a.movzx(ecx, word_ptr(eax * 8 + TABLE as i32)).unwrap();
        a.mov(ds, ecx).unwrap();
    }
    for k in 0..READS {
        // One block per data page: read it, fold it in, end the block.
        let mut next = a.create_label();
        a.add(ebp, dword_ptr(esi + (k * 0x1000) as i32)).unwrap();
        a.inc(ebx).unwrap();
        a.jmp(next).unwrap();
        a.set_label(&mut next).unwrap();
    }
    // INVLPG the first page this iteration read (Intel486 PRM §26 INVLPG).
    a.invlpg(dword_ptr(esi)).unwrap();
    a.add(esi, (READS * 0x1000) as i32).unwrap();
    a.cmp(esi, SWEEP_END as i32).unwrap();
    a.jb(swept).unwrap();
    a.mov(esi, SWEEP).unwrap();
    a.set_label(&mut swept).unwrap();
    // Two blocks under a second flat data selector (another context stamp).
    let mut alias = a.create_label();
    a.mov(eax, 0x18u32).unwrap();
    a.mov(ds, eax).unwrap();
    a.add(ebp, dword_ptr(0x7000)).unwrap();
    a.jmp(alias).unwrap();
    a.set_label(&mut alias).unwrap();
    a.mov(eax, 0x10u32).unwrap();
    a.mov(ds, eax).unwrap();
    // CR0.TS before the V86 entry (SDM Vol 3A §2.5: TS gates ESC and WAIT).
    a.mov(eax, cr0).unwrap();
    a.or(eax, 8).unwrap();
    a.mov(cr0, eax).unwrap();
    // IRETD at CPL 0 with EFLAGS.VM set returns to V86 and pops ES, DS, FS
    // and GS as well (SDM Vol 2A IRET p. 3-490,
    // RETURN-TO-VIRTUAL-8086-MODE).
    a.push(0).unwrap(); // GS
    a.push(0).unwrap(); // FS
    a.push(u32::from(V86_DS)).unwrap(); // DS
    a.push(u32::from(V86_DS)).unwrap(); // ES
    a.push(u32::from(V86_DS)).unwrap(); // SS
    a.push(0xfffe).unwrap(); // SP
    a.push(0x2_0002).unwrap(); // EFLAGS: VM, IOPL 0, IF clear
    a.push(u32::from(V86_CS)).unwrap(); // CS
    a.push(0).unwrap(); // IP
    a.iretd().unwrap();
    // The monitor returns here at CPL 0 with DS = ES = 10h.
    a.clts().unwrap();
    // Reload CR3 with its own value, the classic way to flush the TLB
    // (80386 PRM §5.2.5 p. 5-11: a load flushes even when equal).
    a.mov(eax, cr3).unwrap();
    a.mov(cr3, eax).unwrap();
    a.dec(edi).unwrap();
    a.jne(top).unwrap();
    a.hlt().unwrap();
    a.assemble(u64::from(KERNEL)).unwrap()
}

/// The return point after the V86 round trip: the CLTS after the IRETD.
fn kernel_back(code: &[u8]) -> u32 {
    // CLTS (0F 06) is the only one in the loop.
    let at = code
        .windows(2)
        .position(|w| w == [0x0f, 0x06])
        .expect("CLTS");
    KERNEL + at as u32
}

fn v86() -> Vec<u8> {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(16).unwrap();
    a.mov(ax, word_ptr(0x100)).unwrap();
    a.add(ax, 3).unwrap();
    a.mov(word_ptr(0x100), ax).unwrap();
    a.inc(dx).unwrap();
    // IN is not IOPL-sensitive in V86; the TSS's I/O map base lies past its
    // limit, so every port faults with #GP(0) (SDM Vol 3B §§23.2.7, 23.2.8.1).
    a.in_(al, 0x80).unwrap();
    a.hlt().unwrap();
    a.assemble(0).unwrap()
}

/// The #GP monitor: count the exit, drop the error code and the V86 frame
/// (SDM Vol 3B §23.3.1.1, Figure 23-4: EIP, CS, EFLAGS, ESP, SS, ES, DS, FS
/// and GS), load the kernel's data segments, which the exit cleared, and
/// continue the kernel loop.
fn monitor(back: u32) -> Vec<u8> {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(32).unwrap();
    a.mov(eax, 0x10u32).unwrap();
    a.mov(ds, eax).unwrap();
    a.mov(es, eax).unwrap();
    a.inc(dword_ptr(MONITOR_COUNTER as u32)).unwrap();
    a.add(esp, 40).unwrap();
    a.mov(eax, back).unwrap();
    a.jmp(eax).unwrap();
    a.assemble(u64::from(MONITOR)).unwrap()
}

/// A VM and a vCPU at the kernel loop's entry, ring 0, paged, for `model`.
pub fn setup(model: CpuModel, iterations: u32) -> (Vm, Vcpu) {
    setup_with(model, iterations, FIXED)
}

/// `setup` (`FIXED`), or a rotating variant (`ROTATE`, `ROTATE_MANY`), which
/// loads DS, ES, FS and GS from the selector table at the top of each
/// iteration.
pub fn setup_with(model: CpuModel, iterations: u32, rotations: u32) -> (Vm, Vcpu) {
    assert!(matches!(rotations, FIXED | ROTATE | ROTATE_MANY));
    let mut vm = Vm::new(VmConfig::flat(MEMORY));
    vm.set_cpu_model(model);
    vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    let code = kernel(rotations);
    vm.write_bytes(u64::from(KERNEL), &code).unwrap();
    vm.write_bytes(u64::from(V86_CS) << 4, &v86()).unwrap();
    vm.write_bytes(u64::from(MONITOR), &monitor(kernel_back(&code)))
        .unwrap();
    // GDT (SDM Vol 3A §3.4.5): 8 flat ring-0 code, 10h, 18h, 28h and 30h
    // flat data, 20h the busy 32-bit TSS at 5000h, 38h data at 7800h (limit
    // 7FFh, B set, G clear), 40h data at 7C00h (limit 3FFh, B clear), 48h,
    // 50h, 58h and 60h data at 7900h, 7A00h, 7B00h and 7C00h (B set).
    vm.write_bytes(0x508, &0x00cf_9b00_0000_ffffu64.to_le_bytes())
        .unwrap();
    vm.write_bytes(0x510, &0x00cf_9300_0000_ffffu64.to_le_bytes())
        .unwrap();
    vm.write_bytes(0x518, &0x00cf_9300_0000_ffffu64.to_le_bytes())
        .unwrap();
    vm.write_bytes(0x520, &0x0000_8b00_5000_0067u64.to_le_bytes())
        .unwrap();
    for (at, descriptor) in [
        (0x528, 0x00cf_9300_0000_ffffu64),
        (0x530, 0x00cf_9300_0000_ffff),
        (0x538, 0x0040_9300_7800_07ff),
        (0x540, 0x0000_9300_7c00_03ff),
        (0x548, 0x0040_9300_7900_06ff),
        (0x550, 0x0040_9300_7a00_05ff),
        (0x558, 0x0040_9300_7b00_04ff),
        (0x560, 0x0040_9300_7c00_03ff),
    ] {
        vm.write_bytes(at, &descriptor.to_le_bytes()).unwrap();
    }
    let table: Vec<u8> = (0..rotations)
        .flat_map(|i| rotation(i, rotations))
        .flat_map(u16::to_le_bytes)
        .collect();
    vm.write_bytes(u64::from(TABLE), &table).unwrap();
    // IDT vector 13: 32-bit interrupt gate, DPL 0, to 8:MONITOR.
    let gate =
        u64::from(MONITOR & 0xffff) | (8 << 16) | (0x8e00 << 32) | (u64::from(MONITOR >> 16) << 48);
    vm.write_bytes(0x800 + 13 * 8, &gate.to_le_bytes()).unwrap();
    // TSS: ESP0, SS0 = 10h, I/O map base past the limit.
    vm.write_bytes(0x5004, &STACK.to_le_bytes()).unwrap();
    vm.write_bytes(0x5008, &0x10u32.to_le_bytes()).unwrap();
    vm.write_bytes(0x5066, &0x68u16.to_le_bytes()).unwrap();
    // 80386 PRM §5.2 pp. 5-7-11: two tables identity-map 8 MiB, present,
    // writable, user.
    for table in 0..2u32 {
        vm.write_bytes(
            0x1000 + 4 * u64::from(table),
            &((0x2000 + 0x1000 * table) | 7).to_le_bytes(),
        )
        .unwrap();
    }
    let mut entries = Vec::with_capacity(8 * 1024);
    for page in 0..(MEMORY >> 12) as u32 {
        entries.extend(((page << 12) | 7).to_le_bytes());
    }
    vm.write_bytes(0x2000, &entries).unwrap();
    // Distinct sweep values, so the sum checks every read.
    let mut sweep = Vec::with_capacity((SWEEP_END - SWEEP) as usize / 0x1000 * 4);
    for page in 0..(SWEEP_END - SWEEP) / 0x1000 {
        sweep.extend(page.to_le_bytes());
    }
    for (i, word) in sweep.chunks(4).enumerate() {
        vm.write_bytes(u64::from(SWEEP) + 0x1000 * i as u64, word)
            .unwrap();
    }
    let mut cpu = vm.new_vcpu();
    cpu.cpu.legacy.cr0 = 0x8000_0011;
    cpu.cpu.legacy.cr3 = 0x1000;
    cpu.cpu.legacy.cpl = 0;
    cpu.cpu.cs = 8;
    cpu.cpu.ds = 0x10;
    cpu.cpu.es = 0x10;
    cpu.cpu.ss = 0x10;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        *cache = SegmentCache {
            base: 0,
            limit: u32::MAX,
            attributes: (if i == 0 { 0x9b } else { 0x93 }) | 0x4000 | 0x8000,
            ..Default::default()
        };
    }
    // FS and GS hold the null selector, as the monitor leaves them.
    for i in [4, 5] {
        cpu.cpu.legacy.segments[i] = SegmentCache {
            unusable: 1,
            ..Default::default()
        };
    }
    cpu.cpu.legacy.gdtr.base = 0x500;
    cpu.cpu.legacy.gdtr.limit = 0x67;
    cpu.cpu.legacy.idtr.base = 0x800;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    cpu.cpu.legacy.tr = SystemSegment {
        selector: 0x20,
        cache: SegmentCache {
            base: 0x5000,
            limit: 0x67,
            attributes: 0x8b,
            ..Default::default()
        },
        ..Default::default()
    };
    cpu.set_a20_enabled(true);
    let state: &mut CpuState = &mut cpu.cpu;
    state.rip = u64::from(KERNEL);
    state.gpr[4] = u64::from(STACK);
    state.gpr[6] = u64::from(SWEEP);
    state.gpr[7] = u64::from(iterations);
    (vm, cpu)
}

/// Guest instructions retired by `iterations` and the final HLT: per
/// iteration 3 per read block, then INVLPG, ADD, CMP, Jcc (and MOV at the
/// wrap), 4 + 2 under the alias selector, 3 for TS, 9 PUSH and the IRETD, 4
/// in V86 (the faulting IN retires none), 7 in the monitor, CLTS, 2 for the
/// CR3 reload, DEC, JNZ.
pub fn expected_retired(iterations: u32) -> u64 {
    expected_retired_with(iterations, FIXED)
}

/// `expected_retired` of `setup_with`: a rotating variant adds 10 per
/// iteration (MOV, AND, then a MOVZX and a segment load for each of ES, FS,
/// GS and DS).
pub fn expected_retired_with(iterations: u32, rotations: u32) -> u64 {
    let rotate = if rotations == FIXED { 0 } else { 10 };
    let per = 3 * u64::from(READS) + 4 + 6 + 3 + 10 + 4 + 7 + 5 + rotate;
    let pages = (SWEEP_END - SWEEP) / 0x1000;
    let mut start = 0;
    let mut wraps = 0;
    for _ in 0..iterations {
        start += READS;
        if start >= pages {
            start = 0;
            wraps += 1;
        }
    }
    per * u64::from(iterations) + wraps + 1
}

/// The EBP sum after `iterations`: every read adds its page's index.
pub fn expected_sum(iterations: u32) -> u32 {
    let pages = (SWEEP_END - SWEEP) / 0x1000;
    let mut start = 0;
    let mut sum = 0u32;
    for _ in 0..iterations {
        for k in 0..READS {
            // Pages past the sweep hold zero.
            if start + k < pages {
                sum = sum.wrapping_add(start + k);
            }
        }
        start += READS;
        if start >= pages {
            start = 0;
        }
    }
    sum
}

/// The wide variant (xj-to9.75): a flat ring-0 loop of `blocks` blocks
/// (ADD EBP,EBX; INC EBX; JMP to the next), then a CR3 load from a table of
/// `directories` page directories that all map the same identity tables,
/// as a kernel entered from many processes runs the same code under each
/// address space. Its working set is `blocks` code blocks under
/// `directories` CR3 values. Directories at 8000h + 1000h k, the CR3 table
/// at 7000h, the stack below 80000h, the code at 100000h.
pub fn setup_wide(model: CpuModel, iterations: u32, blocks: u32, directories: u32) -> (Vm, Vcpu) {
    use iced_x86::code_asm::*;
    assert!(directories.is_power_of_two() && directories <= 32);
    let (vm, mut cpu) = flat_ring0(model, directories);
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    a.set_label(&mut top).unwrap();
    for _ in 0..blocks {
        let mut next = a.create_label();
        a.add(ebp, ebx).unwrap();
        a.inc(ebx).unwrap();
        a.jmp(next).unwrap();
        a.set_label(&mut next).unwrap();
    }
    // 80386 PRM §5.2.5 p. 5-11: the load flushes the TLB.
    a.mov(eax, edi).unwrap();
    a.and(eax, (directories - 1) as i32).unwrap();
    a.mov(eax, dword_ptr(eax * 4 + 0x7000)).unwrap();
    a.mov(cr3, eax).unwrap();
    a.dec(edi).unwrap();
    a.jne(top).unwrap();
    a.hlt().unwrap();
    let code = a.assemble(0x10_0000).unwrap();
    vm.write_bytes(0x10_0000, &code).unwrap();
    cpu.cpu.rip = 0x10_0000;
    cpu.cpu.gpr[7] = u64::from(iterations);
    (vm, cpu)
}

// A VM of `MEMORY` bytes of RAM, `directories` page directories at 8000h +
// 1000h k that all name the same two tables, which identity-map 8 MiB,
// present, writable, user, and their CR3 values at 7000h; and a vCPU at
// ring 0 under the first, flat CS, DS, ES and SS, the stack below 80000h.
fn flat_ring0(model: CpuModel, directories: u32) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(MEMORY));
    vm.set_cpu_model(model);
    vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    // 80386 PRM §5.2 pp. 5-7-11: each directory names the same two tables,
    // which identity-map 8 MiB, present, writable, user.
    let mut entries = Vec::with_capacity(8 * 1024);
    for page in 0..(MEMORY >> 12) as u32 {
        entries.extend(((page << 12) | 7).to_le_bytes());
    }
    vm.write_bytes(0x2000, &entries).unwrap();
    for k in 0..directories {
        let directory = 0x8000 + 0x1000 * k;
        for table in 0..2u32 {
            vm.write_bytes(
                u64::from(directory + 4 * table),
                &((0x2000 + 0x1000 * table) | 7).to_le_bytes(),
            )
            .unwrap();
        }
        vm.write_bytes(u64::from(0x7000 + 4 * k), &directory.to_le_bytes())
            .unwrap();
    }
    let mut cpu = vm.new_vcpu();
    cpu.cpu.legacy.cr0 = 0x8000_0011;
    cpu.cpu.legacy.cr3 = 0x8000;
    cpu.cpu.legacy.cpl = 0;
    cpu.cpu.cs = 8;
    cpu.cpu.ds = 0x10;
    cpu.cpu.es = 0x10;
    cpu.cpu.ss = 0x10;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        *cache = SegmentCache {
            base: 0,
            limit: u32::MAX,
            attributes: (if i == 0 { 0x9b } else { 0x93 }) | 0x4000 | 0x8000,
            ..Default::default()
        };
    }
    cpu.set_a20_enabled(true);
    cpu.cpu.gpr[4] = 0x8_0000;
    (vm, cpu)
}

// Bytes of each code page the store and context variants fill with blocks
// before the JMP to the next page; the store variant's data words follow.
const PAGE_CODE: usize = 0xf00;
// A code page's data words: 8 dwords, past every byte its blocks fetch.
const PAGE_DATA: u32 = 0xf80;
// The non-code data page the store variant's `Stores::Data` writes.
const DATA_PAGE: u32 = 0x30_0000;
const CODE: u32 = 0x10_0000;

/// Where `setup_stores` writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stores {
    /// A data word on the store's own code page, past its code: every store
    /// is a code-page write, drained and reconciled against the proofs.
    CodePage,
    /// A word of a page no code is fetched from.
    Data,
}

// `blocks` blocks (ADD EBP,EBX; INC EBX; JMP to the next), each `stride`th
// one led by `MOV [target],EBX` (`store(page, i)` names the target), filled
// page by page from `CODE` (`PAGE_CODE` bytes each), the last JMP to `tail`.
fn block_pages(
    blocks: u32,
    stride: u32,
    tail: u32,
    store: impl Fn(u32, u32) -> Option<u32>,
) -> Vec<(u32, Vec<u8>)> {
    use iced_x86::code_asm::*;
    let mut pages = Vec::new();
    let mut i = 0;
    let mut page = CODE;
    while i < blocks {
        let mut a = CodeAssembler::new(32).unwrap();
        let mut size = 0;
        // Each block is at most 11 bytes; the page's last JMP rel32 is 5.
        loop {
            if stride != 0 && i % stride == 0 {
                if let Some(target) = store(page, i) {
                    a.mov(dword_ptr(target), ebx).unwrap();
                    size += 6;
                }
            }
            a.add(ebp, ebx).unwrap();
            a.inc(ebx).unwrap();
            size += 3;
            i += 1;
            if i == blocks || size + 2 + 11 + 5 > PAGE_CODE {
                break;
            }
            let mut next = a.create_label();
            a.jmp(next).unwrap();
            a.set_label(&mut next).unwrap();
            size += 2;
        }
        let next = if i < blocks { page + 0x1000 } else { tail };
        a.jmp(u64::from(next)).unwrap();
        pages.push((page, a.assemble(u64::from(page)).unwrap()));
        page += 0x1000;
    }
    pages
}

/// The store variant (xj-to9.85): a flat ring-0 loop of `blocks` blocks
/// over consecutive code pages (up to 768 a page), each `stride`th of which
/// first stores EBX to a data word: on its own code page past the code
/// (`Stores::CodePage`), as an operating system's data sits beside its
/// code, or on a page without code (`Stores::Data`). A code-page store
/// overlaps no block, so no proof drops; each is drained and reconciled
/// against every proof held. One CR3, no TLB churn.
pub fn setup_stores(
    model: CpuModel,
    iterations: u32,
    blocks: u32,
    stride: u32,
    stores: Stores,
) -> (Vm, Vcpu) {
    use iced_x86::code_asm::*;
    let (vm, mut cpu) = flat_ring0(model, 1);
    let tail = CODE + 0x1000 * blocks.div_ceil(250);
    let pages = block_pages(blocks, stride, tail, |page, i| {
        Some(match stores {
            Stores::CodePage => page + PAGE_DATA + 4 * ((i / stride) % 8),
            Stores::Data => DATA_PAGE + 4 * ((i / stride) % 1024),
        })
    });
    assert!(pages.last().unwrap().0 < tail);
    for (page, code) in &pages {
        assert!(code.len() <= PAGE_CODE);
        vm.write_bytes(u64::from(*page), code).unwrap();
    }
    let mut a = CodeAssembler::new(32).unwrap();
    a.dec(edi).unwrap();
    a.jne(u64::from(CODE)).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(u64::from(tail), &a.assemble(u64::from(tail)).unwrap())
        .unwrap();
    cpu.cpu.rip = u64::from(CODE);
    cpu.cpu.gpr[7] = u64::from(iterations);
    (vm, cpu)
}

/// Guest instructions `setup_stores` retires: 3 per block (the page's last
/// JMP counts as its block's), 1 per store, 2 per iteration, and the HLT.
pub fn expected_retired_stores(iterations: u32, blocks: u32, stride: u32) -> u64 {
    let stores = if stride == 0 {
        0
    } else {
        blocks.div_ceil(stride)
    };
    (3 * u64::from(blocks) + u64::from(stores) + 2) * u64::from(iterations) + 1
}

/// The context variant (xj-to9.85): `blocks` flat ring-0 blocks run once
/// (resident proofs), then a loop that loads ES with the next of `stamps`
/// byte-granular data segments at distinct bases (selector 18h + 8 k),
/// runs two blocks in that Segmented state and reloads the flat ES. Each
/// iteration's paged identity is a complete stamp that names ES, so more
/// stamps than the 64 retained and 4,096 retired ones evict a retained
/// identity and forget a retired one, with its proofs, every iteration.
/// GDT at 400000h.
pub fn setup_contexts(model: CpuModel, iterations: u32, blocks: u32, stamps: u32) -> (Vm, Vcpu) {
    use iced_x86::code_asm::*;
    assert!((1..=8189).contains(&stamps));
    let (vm, mut cpu) = flat_ring0(model, 1);
    let tail = CODE + 0x1000 * blocks.div_ceil(250).max(1);
    for (page, code) in block_pages(blocks, 0, tail, |_, _| None) {
        vm.write_bytes(u64::from(page), &code).unwrap();
    }
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    a.set_label(&mut top).unwrap();
    a.xor(edx, edx).unwrap();
    a.mov(eax, edi).unwrap();
    a.mov(ecx, stamps).unwrap();
    a.div(ecx).unwrap();
    a.lea(eax, dword_ptr(edx * 8 + 0x18)).unwrap();
    a.mov(es, eax).unwrap();
    for _ in 0..2 {
        let mut next = a.create_label();
        a.add(ebp, ebx).unwrap();
        a.inc(ebx).unwrap();
        a.jmp(next).unwrap();
        a.set_label(&mut next).unwrap();
    }
    a.mov(eax, 0x10u32).unwrap();
    a.mov(es, eax).unwrap();
    a.dec(edi).unwrap();
    a.jne(top).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(u64::from(tail), &a.assemble(u64::from(tail)).unwrap())
        .unwrap();
    // GDT (SDM Vol 3A §3.4.5): 8 flat ring-0 code, 10h flat data, 18h + 8 k
    // data at 10h k (limit FFFFh, B set, G clear).
    const GDT: u32 = 0x40_0000;
    let mut gdt = Vec::with_capacity(8 * (3 + stamps as usize));
    gdt.extend(0u64.to_le_bytes());
    gdt.extend(0x00cf_9b00_0000_ffffu64.to_le_bytes());
    gdt.extend(0x00cf_9300_0000_ffffu64.to_le_bytes());
    for k in 0..u64::from(stamps) {
        let base = 0x10 * k;
        let descriptor = 0x0040_9300_0000_ffff | ((base & 0xff_ffff) << 16) | ((base >> 24) << 56);
        gdt.extend(descriptor.to_le_bytes());
    }
    vm.write_bytes(u64::from(GDT), &gdt).unwrap();
    cpu.cpu.legacy.gdtr.base = GDT;
    cpu.cpu.legacy.gdtr.limit = (gdt.len() - 1) as u16;
    cpu.cpu.rip = u64::from(CODE);
    cpu.cpu.gpr[7] = u64::from(iterations);
    (vm, cpu)
}

/// Guest instructions `setup_contexts` retires: 3 per warm block, then 16
/// per iteration (6 to select and load ES, 6 in two blocks, 2 to reload ES,
/// DEC, JNZ), and the HLT.
pub fn expected_retired_contexts(iterations: u32, blocks: u32) -> u64 {
    3 * u64::from(blocks) + 16 * u64::from(iterations) + 1
}

/// Guest instructions `setup_wide` retires: 3 per block and 6 per
/// iteration, and the final HLT.
pub fn expected_retired_wide(iterations: u32, blocks: u32) -> u64 {
    (3 * u64::from(blocks) + 6) * u64::from(iterations) + 1
}
