//! A paged ring-0 loop that turns the functional TLB over (xj-to9.70), shared by
//! the relift witness (`tests/legacy/legacy_relift_diag.rs`) and the
//! `machine32_paged_churn` measurement. It reproduces the shape of a
//! protected-mode operating system's paged kernel phases: about two dozen
//! block entries per iteration that read 20 data pages of a 4 MiB sweep, more
//! pages than the 256-entry TLB holds, so fills keep replacing the entries the
//! loop's own code was fetched through; an INVLPG of a data page; CR0.TS set
//! before a V86 entry and cleared by CLTS after the exit (FPU ownership); an
//! IRETD into a short V86 body whose IN traps to a ring-0 #GP monitor; and a
//! second flat data selector, so the loop runs under several context stamps.
//!
//! Layout (identity-mapped, every page user/writable): GDT at 500h, IDT at
//! 800h, ring-0 stack below 9000h, TSS at 5000h, kernel code at 10000h, V86
//! code at 20000h (CS 2000h), the #GP monitor at 30000h, V86 data and stack at
//! 40000h (DS = SS = 4000h), page directory 1000h, tables 2000h-3fffh, and
//! the sweep at 100000h-4fffffh.
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

/// The kernel loop: `iterations` in EDI; HLT at ring 0 when done.
fn kernel() -> Vec<u8> {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    let mut swept = a.create_label();
    a.set_label(&mut top).unwrap();
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
    let mut vm = Vm::new(VmConfig::flat(MEMORY));
    vm.set_cpu_model(model);
    vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    let code = kernel();
    vm.write_bytes(u64::from(KERNEL), &code).unwrap();
    vm.write_bytes(u64::from(V86_CS) << 4, &v86()).unwrap();
    vm.write_bytes(u64::from(MONITOR), &monitor(kernel_back(&code)))
        .unwrap();
    // GDT (SDM Vol 3A §3.4.5): 8 flat ring-0 code, 10h and 18h flat data,
    // 20h the busy 32-bit TSS at 5000h.
    vm.write_bytes(0x508, &0x00cf_9b00_0000_ffffu64.to_le_bytes())
        .unwrap();
    vm.write_bytes(0x510, &0x00cf_9300_0000_ffffu64.to_le_bytes())
        .unwrap();
    vm.write_bytes(0x518, &0x00cf_9300_0000_ffffu64.to_le_bytes())
        .unwrap();
    vm.write_bytes(0x520, &0x0000_8b00_5000_0067u64.to_le_bytes())
        .unwrap();
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
    cpu.cpu.legacy.gdtr.limit = 0x27;
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
/// in V86 (the faulting IN retires none), 7 in the monitor, CLTS, DEC, JNZ.
pub fn expected_retired(iterations: u32) -> u64 {
    let per = 3 * u64::from(READS) + 4 + 6 + 3 + 10 + 4 + 7 + 3;
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
