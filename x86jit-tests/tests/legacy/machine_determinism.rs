//! xj-to9.80: a protected-mode operating system in miniature, run in budgeted
//! slices on the Intel486 Machine model, compiled against interpreted. The
//! guest is a paged ring-0 kernel loop (calls, read-modify-writes, REP STOSD
//! and REP MOVSD across demand-paged pages, CLI/STI, POPFD and MOV SS
//! windows, A and D bit aging without INVLPG, unmapping with INVLPG, CR3
//! reloads) that drops to a ring-3 task through IRETD and back through
//! INT 80h, then to a V86 task (IOPL 3: PUSHF/POPF/CLI/STI, INT 81h back to
//! ring 0). A ring-0 #PF handler maps the faulting page with A and D clear;
//! a timer handler runs at every injected vector 20h.
//!
//! An embedder raises its interrupts between run calls, so the harness
//! injects the timer whenever the retired count crosses a fixed period, on
//! both machines alike. After every slice the step result, the retired
//! count, the full register state (CR2, the segment caches, CPL and every
//! flag included), the TLB contents and the written memory must agree; all
//! memory is compared every few slices and at the end. Slices are 1, 7, 97,
//! 997 and 10,000 steps, and a mixed schedule; the subject compiles in the
//! Forced, Delayed and Scripted test modes and with background tier-up.
//!
//! Sources: SDM Vol 3A §6.12.1 (interrupt gates, IF cleared), §6.8.3 and
//! §6.8.1 (the STI and MOV SS shadows), Vol 3B §20.3 (V86 entry and exit
//! frames through IRETD and interrupt gates), Vol 3A §4.8 / 80386 PRM §5.2.4
//! (A and D bits), Intel486 PRM §26 INVLPG.
use super::machine_j1_lockstep::{live, Subject};
use iced_x86::code_asm::*;
use x86jit_core::{
    ChainEdgeAction, CpuModel, DeviceContext, DeviceDeadline, DeviceIo, DeviceRange, DeviceRequest,
    DeviceResult, DeviceSpace, DeviceStop, ExceptionDelivery, Exit, MachineJitCounters,
    MachineJitMode, Prot, RegionKind, SegmentCache, StepOptions, StepStop, SystemSegment,
    TableRegister, Vcpu, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const MEMORY: u64 = 0x18_0000;
const PD: u64 = 0x1000;
const PT: u64 = 0x2000;
const IDT: u64 = 0x3000;
const GDT: u64 = 0x3800;
const TSS: u64 = 0x5000;
/// Kernel variables: iteration 6000h, CLI/STI counters 6010h-6014h,
/// INT 80h/81h counts 6020h-6024h, ticks 6100h, #PF count 6104h, A and D
/// counts 6108h-610Ch; tables from 6400h.
const DATA: u64 = 0x6000;
const STACK_TOP: u32 = 0xf000;
const KERNEL: u64 = 0x10000;
const AFTER_USER: u64 = 0x11000;
const AFTER_V86: u64 = 0x11800;
const KSUB: u64 = 0x12000;
/// The routines of [`shapes_loop`] at `ROUTINE` and `ROUTINE + 100h`.
const ROUTINE: u64 = 0x14000;
/// `cli; inc eax; sti; ret` and `pushfd; cli; inc [6018h]; popfd; ret`:
/// returns in the STI shadow and right after POPFD sets IF.
const WINDOWS: u64 = 0x13000;
/// `mov ecx, imm32; ret`, its immediate rewritten every iteration.
const PATCH: u64 = 0x12400;
/// A routine calling each of `FUNCTIONS` small functions at `FUNCTION`
/// (32 bytes apart): a working set of thousands of blocks.
const CALLER: u64 = 0x40000;
const FUNCTION: u64 = 0x60000;
const FUNCTIONS: u64 = 400;
const TIMER: u64 = 0x18000;
const PAGE_FAULT: u64 = 0x18400;
const INT80: u64 = 0x18800;
const INT81: u64 = 0x18c00;
const STRAY: u64 = 0x18f00;
const USER_CODE: u64 = 0x20000;
const USER_SUB: u64 = 0x20800;
const USER_STACK_TOP: u32 = 0x23000;
const V86_CS: u16 = 0x3000;
const V86_DS: u16 = 0x3100;
const V86_SS: u16 = 0x3200;
/// 64 pages, not present at the start, mapped by the #PF handler.
const DEMAND: u32 = 0x10_0000;
const TIMER_VECTOR: u8 = 0x20;
/// A device page (a Trap region) and ports 40h-43h, served in process by
/// [`Device`] or answered at their exits.
const MMIO: u64 = 0x17_0000;
/// The kernel runs at linear C0000000h and up, a second mapping of the low
/// frames through a supervisor directory entry.
const HIGH: u64 = 0xc000_0000;

fn asm32(at: u64, build: impl Fn(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(32).unwrap();
    build(&mut a);
    a.assemble(at).unwrap()
}

/// The kernel's main loop, to the IRETD into ring 3.
fn kernel_loop(a: &mut CodeAssembler) {
    let mut top = a.create_label();
    let mut compute = a.create_label();
    let mut probe = a.create_label();
    let mut windows = a.create_label();
    let mut age = a.create_label();
    let mut not_accessed = a.create_label();
    let mut clean = a.create_label();
    let mut unmap = a.create_label();
    let mut no_unmap = a.create_label();
    let mut no_cr3 = a.create_label();
    a.set_label(&mut top).unwrap();
    a.inc(dword_ptr(DATA)).unwrap();
    // Self-modifying code: patch the loop count PATCH loads, then call it.
    a.mov(eax, dword_ptr(DATA)).unwrap();
    a.and(eax, 63).unwrap();
    a.add(eax, 120).unwrap();
    a.mov(dword_ptr(HIGH + PATCH + 1), eax).unwrap();
    a.call(HIGH + PATCH).unwrap();
    // Thousands of blocks.
    a.call(HIGH + CALLER).unwrap();
    // Compiled work: calls, read-modify-writes, a loop.
    a.set_label(&mut compute).unwrap();
    a.add(ebx, ecx).unwrap();
    a.xor(edx, ebx).unwrap();
    a.mov(esi, ecx).unwrap();
    a.and(esi, 0x3f).unwrap();
    a.add(dword_ptr(esi * 4 + 0x6400), edx).unwrap();
    a.call(HIGH + KSUB).unwrap();
    a.call(HIGH + WINDOWS).unwrap();
    a.call(HIGH + WINDOWS + 0x10).unwrap();
    a.pushad().unwrap();
    a.call(HIGH + KSUB).unwrap();
    a.popad().unwrap();
    a.rol(edx, 3).unwrap();
    let mut quiet = a.create_label();
    if !DEVICES.with(std::cell::Cell::get) {
        a.jmp(quiet).unwrap();
    }
    a.mov(eax, dword_ptr(HIGH + MMIO + 8)).unwrap();
    a.add(ebx, eax).unwrap();
    a.mov(dword_ptr(HIGH + MMIO + 4), edx).unwrap();
    a.add(ebx, dword_ptr(HIGH + MMIO)).unwrap();
    a.test(cl, 15).unwrap();
    a.jne(quiet).unwrap();
    a.mov(dword_ptr(HIGH + MMIO + 0xc), ebx).unwrap();
    a.set_label(&mut quiet).unwrap();
    a.dec(ecx).unwrap();
    a.jne(compute).unwrap();
    // REP STOSD and REP MOVSD across demand-paged pages.
    a.mov(eax, dword_ptr(DATA)).unwrap();
    a.imul_3(edi, eax, 0x1a44).unwrap();
    a.and(edi, 0x1fffc).unwrap();
    a.add(edi, DEMAND).unwrap();
    a.mov(ecx, 3000u32).unwrap();
    a.mov(eax, ebx).unwrap();
    a.cld().unwrap();
    a.rep().stosd().unwrap();
    a.mov(eax, dword_ptr(DATA)).unwrap();
    a.imul_3(esi, eax, 0x2468).unwrap();
    a.and(esi, 0x1fffc).unwrap();
    a.add(esi, DEMAND).unwrap();
    a.imul_3(edi, eax, 0x1357).unwrap();
    a.and(edi, 0x1fffc).unwrap();
    a.add(edi, DEMAND + 0x2_0000).unwrap();
    a.mov(ecx, 1500u32).unwrap();
    a.rep().movsd().unwrap();
    // Probes: a read and a read-modify-write on every third page.
    a.mov(ecx, 16u32).unwrap();
    a.mov(esi, DEMAND + 0x40).unwrap();
    a.set_label(&mut probe).unwrap();
    a.add(ebp, dword_ptr(esi)).unwrap();
    a.add(dword_ptr(esi + 8), ecx).unwrap();
    a.add(esi, 0x3000).unwrap();
    a.dec(ecx).unwrap();
    a.jne(probe).unwrap();
    // Interrupt windows: CLI ... STI (shadow), POPFD clearing and setting
    // IF, MOV SS (shadow).
    a.cli().unwrap();
    a.add(eax, ebx).unwrap();
    a.add(ebx, ecx).unwrap();
    a.sti().unwrap();
    a.add(ecx, edx).unwrap();
    a.pushfd().unwrap();
    a.and(dword_ptr(esp), !0x200u32 as i32).unwrap();
    a.popfd().unwrap();
    a.add(edx, eax).unwrap();
    a.inc(dword_ptr(DATA + 0x10)).unwrap();
    a.pushfd().unwrap();
    a.or(dword_ptr(esp), 0x200).unwrap();
    a.popfd().unwrap();
    a.mov(ax, ss).unwrap();
    a.mov(ss, ax).unwrap();
    a.add(eax, 7).unwrap();
    a.mov(ecx, 20u32).unwrap();
    a.set_label(&mut windows).unwrap();
    a.cli().unwrap();
    a.add(dword_ptr(DATA + 0x14), ecx).unwrap();
    a.sti().unwrap();
    a.dec(ecx).unwrap();
    a.jne(windows).unwrap();
    // Aging: count and clear A and D in the demand PTEs, no INVLPG, with
    // interrupts held off through the unmapping (PUSHFD; CLI ... POPFD).
    a.pushfd().unwrap();
    a.cli().unwrap();
    a.mov(ecx, 64u32).unwrap();
    a.mov(edx, (PT + 4 * u64::from(DEMAND >> 12)) as u32)
        .unwrap();
    a.set_label(&mut age).unwrap();
    a.mov(eax, dword_ptr(edx)).unwrap();
    a.test(al, 0x20).unwrap();
    a.je(not_accessed).unwrap();
    a.inc(dword_ptr(DATA + 0x108)).unwrap();
    a.set_label(&mut not_accessed).unwrap();
    a.test(al, 0x40).unwrap();
    a.je(clean).unwrap();
    a.inc(dword_ptr(DATA + 0x10c)).unwrap();
    a.set_label(&mut clean).unwrap();
    a.and(eax, !0x60u32 as i32).unwrap();
    a.mov(dword_ptr(edx), eax).unwrap();
    a.add(edx, 4).unwrap();
    a.dec(ecx).unwrap();
    a.jne(age).unwrap();
    // Every other iteration, unmap eight pages with INVLPG.
    a.test(byte_ptr(DATA), 1).unwrap();
    a.je(no_unmap).unwrap();
    a.mov(eax, dword_ptr(DATA)).unwrap();
    a.imul_3(eax, eax, 7).unwrap();
    a.mov(ecx, 8u32).unwrap();
    a.set_label(&mut unmap).unwrap();
    a.and(eax, 63).unwrap();
    a.mov(edx, eax).unwrap();
    a.and(
        dword_ptr(edx * 4 + (PT + 4 * u64::from(DEMAND >> 12)) as i32),
        !1u32 as i32,
    )
    .unwrap();
    a.shl(edx, 12).unwrap();
    a.add(edx, DEMAND).unwrap();
    a.invlpg(byte_ptr(edx)).unwrap();
    a.add(eax, 5).unwrap();
    a.dec(ecx).unwrap();
    a.jne(unmap).unwrap();
    a.set_label(&mut no_unmap).unwrap();
    a.popfd().unwrap();
    // Every fourth iteration, reload CR3.
    a.test(byte_ptr(DATA), 3).unwrap();
    a.jne(no_cr3).unwrap();
    a.mov(eax, cr3).unwrap();
    a.mov(cr3, eax).unwrap();
    a.set_label(&mut no_cr3).unwrap();
    // To ring 3 (SDM Vol 2A IRET): SS, ESP, EFLAGS (IF), CS, EIP.
    a.mov(ax, 0x23).unwrap();
    a.mov(ds, ax).unwrap();
    a.mov(es, ax).unwrap();
    a.mov(fs, ax).unwrap();
    a.mov(gs, ax).unwrap();
    a.push(0x23).unwrap();
    a.push(USER_STACK_TOP).unwrap();
    a.push(0x202).unwrap();
    a.push(0x1b).unwrap();
    a.push(USER_CODE as u32).unwrap();
    a.iretd().unwrap();
}

/// Two near-twin ring-0 routines: PUSHAD, a flag word tested and changed by
/// BTR (BTS), a structure read through it, LIDT, two backward REP MOVSD
/// after STD, CLD, a read-modify-write, POPAD, RET.
fn routine(a: &mut CodeAssembler, twin: bool) {
    let mut out = a.create_label();
    a.pushad().unwrap();
    a.mov(ebx, dword_ptr(DATA + 0x40)).unwrap();
    if twin {
        a.bts(dword_ptr(ebx), 5).unwrap();
        a.jb(out).unwrap();
    } else {
        a.btr(dword_ptr(ebx), 3).unwrap();
        a.jae(out).unwrap();
    }
    a.mov(edi, dword_ptr(DATA + 0x44)).unwrap();
    if twin {
        a.or(byte_ptr(edi + 8), 1).unwrap();
    } else {
        a.and(byte_ptr(edi + 8), 0xfe).unwrap();
    }
    a.mov(ebp, dword_ptr(ebx + 4)).unwrap();
    a.mov(eax, dword_ptr(ebp + 8)).unwrap();
    a.lidt(dword_ptr(DATA + 0x50)).unwrap();
    a.std().unwrap();
    a.lea(esi, dword_ptr(ebp + 0x7c)).unwrap();
    a.lea(edi, dword_ptr(esi + 0x60)).unwrap();
    a.mov(ecx, 18u32).unwrap();
    a.rep().movsd().unwrap();
    a.lea(esi, dword_ptr(ebp + 0x180)).unwrap();
    a.mov(ecx, 9u32).unwrap();
    a.rep().movsd().unwrap();
    a.cld().unwrap();
    a.and(eax, 0xffff_00ffu32 as i32).unwrap();
    a.or(eax, 0x100).unwrap();
    a.mov(dword_ptr(ebp + 8), eax).unwrap();
    if twin {
        a.sub(dword_ptr(DATA + 0x60), 1).unwrap();
    } else {
        a.add(dword_ptr(DATA + 0x60), 1).unwrap();
    }
    a.set_label(&mut out).unwrap();
    a.popad().unwrap();
    a.ret().unwrap();
}

/// The compared variable of [`compare_first`] and [`compare_second`]: in
/// RAM, or with [`DEVICES`] a register of the device page.
fn compared() -> u64 {
    if DEVICES.with(std::cell::Cell::get) {
        HIGH + MMIO + 0x10
    } else {
        HIGH + 0x9710
    }
}

/// `push edx; push ecx; mov edx,[m]; add edx,ebx; add [edx+8],ecx;
/// cmp ebx,[v]; jne; mov ecx,[edx+8]; cmp ecx,[edx+8]; jne; call [m'];
/// pop ecx; pop edx; ret`.
fn compare_first(a: &mut CodeAssembler) {
    let mut out = a.create_label();
    a.push(edx).unwrap();
    a.push(ecx).unwrap();
    a.mov(edx, dword_ptr(DATA + 0x70)).unwrap();
    a.add(edx, ebx).unwrap();
    a.add(dword_ptr(edx + 8), ecx).unwrap();
    a.cmp(ebx, dword_ptr(compared())).unwrap();
    a.jne(out).unwrap();
    a.mov(ecx, dword_ptr(edx + 8)).unwrap();
    a.cmp(ecx, dword_ptr(edx + 8)).unwrap();
    a.jne(out).unwrap();
    a.call(dword_ptr(DATA + 0x68)).unwrap();
    a.set_label(&mut out).unwrap();
    a.pop(ecx).unwrap();
    a.pop(edx).unwrap();
    a.ret().unwrap();
}

/// `movzx edx,byte [eax+d32]; mov edx,[edx*4+d32]; cmp ebx,[v]; jne; ...`.
fn compare_second(a: &mut CodeAssembler) {
    let mut out = a.create_label();
    a.push(eax).unwrap();
    a.push(edx).unwrap();
    a.and(eax, 0x3f).unwrap();
    a.movzx(edx, byte_ptr(eax + (DATA + 0x400) as i32)).unwrap();
    a.and(edx, 0x3f).unwrap();
    a.mov(edx, dword_ptr(edx * 4 + (DATA + 0x500) as i32))
        .unwrap();
    a.cmp(ebx, dword_ptr(compared())).unwrap();
    a.jne(out).unwrap();
    a.inc(dword_ptr(DATA + 0x74)).unwrap();
    a.set_label(&mut out).unwrap();
    a.pop(edx).unwrap();
    a.pop(eax).unwrap();
    a.ret().unwrap();
}

/// The routines' caller, and short blocks that switch stacks with XCHG ESP:
/// a compare and branch, a STI window, an indirect jump.
fn shapes_loop(a: &mut CodeAssembler) {
    let mut top = a.create_label();
    let mut equal = a.create_label();
    let mut joined = a.create_label();
    let mut below = a.create_label();
    a.cli().unwrap();
    a.set_label(&mut top).unwrap();
    a.inc(dword_ptr(DATA)).unwrap();
    a.xor(dword_ptr(DATA + 0x200), 0x28).unwrap();
    a.call(HIGH + ROUTINE).unwrap();
    a.call(HIGH + ROUTINE + 0x100).unwrap();
    a.and(ebx, 0xff0).unwrap();
    // The compared variable's translation leaves the TLB every other pass,
    // so the compiled load misses the view (decision-32 §3).
    let mut kept = a.create_label();
    a.test(byte_ptr(DATA), 1).unwrap();
    a.je(kept).unwrap();
    a.invlpg(byte_ptr(compared())).unwrap();
    a.set_label(&mut kept).unwrap();
    a.call(HIGH + ROUTINE + 0x300).unwrap();
    a.call(HIGH + ROUTINE + 0x400).unwrap();
    a.mov(eax, dword_ptr(DATA)).unwrap();
    a.and(eax, 7).unwrap();
    a.xchg(dword_ptr(DATA + 0x30), esp).unwrap();
    a.cmp(eax, 5).unwrap();
    a.je(equal).unwrap();
    a.mov(ebp, esp).unwrap();
    a.movzx(ecx, byte_ptr(ebp - 4)).unwrap();
    a.jmp(joined).unwrap();
    a.set_label(&mut equal).unwrap();
    a.inc(ecx).unwrap();
    a.set_label(&mut joined).unwrap();
    a.xchg(dword_ptr(DATA + 0x30), esp).unwrap();
    a.mov(edx, esp).unwrap();
    a.xchg(esp, edx).unwrap();
    a.sti().unwrap();
    a.push(eax).unwrap();
    a.cmp(ecx, 3).unwrap();
    a.jb(below).unwrap();
    a.mov(ebx, dword_ptr(esp)).unwrap();
    a.set_label(&mut below).unwrap();
    a.pop(eax).unwrap();
    a.cli().unwrap();
    // An indirect jump to `ROUTINE + 200h: jmp top`.
    a.mov(eax, (HIGH + ROUTINE + 0x200) as u32).unwrap();
    a.jmp(eax).unwrap();
}

/// After INT 80h: to the V86 task through IRETD (SDM Vol 3B §20.3: GS, FS,
/// DS, ES, SS, ESP, EFLAGS with VM, CS, EIP).
fn after_user(a: &mut CodeAssembler) {
    a.sti().unwrap();
    a.add(eax, dword_ptr(DATA)).unwrap();
    a.push(0).unwrap();
    a.push(0).unwrap();
    a.push(u32::from(V86_DS)).unwrap();
    a.push(u32::from(V86_DS)).unwrap();
    a.push(u32::from(V86_SS)).unwrap();
    a.push(0xff0).unwrap();
    // VM, IOPL 3, IF.
    a.push(0x2_3202).unwrap();
    a.push(u32::from(V86_CS)).unwrap();
    a.push(0).unwrap();
    a.iretd().unwrap();
}

fn after_v86(a: &mut CodeAssembler) {
    a.pushfd().unwrap();
    a.or(dword_ptr(esp), 0x3000).unwrap();
    a.popfd().unwrap();
    a.sti().unwrap();
    a.jmp(HIGH + KERNEL).unwrap();
}

/// A subroutine whose counter shares its code page.
fn ksub(a: &mut CodeAssembler) {
    a.add(eax, edx).unwrap();
    a.inc(dword_ptr(HIGH + KSUB + 0x800)).unwrap();
    a.push(eax).unwrap();
    a.mov(eax, dword_ptr(esp)).unwrap();
    a.pop(eax).unwrap();
    a.ret().unwrap();
}

fn timer(a: &mut CodeAssembler) {
    let mut table = a.create_label();
    a.push(eax).unwrap();
    a.push(ecx).unwrap();
    a.push(ds).unwrap();
    a.mov(ax, 0x10).unwrap();
    a.mov(ds, ax).unwrap();
    a.inc(dword_ptr(DATA + 0x100)).unwrap();
    a.in_(al, 0x40).unwrap();
    a.out(0x43, al).unwrap();
    a.mov(eax, dword_ptr(DATA + 0x100)).unwrap();
    a.and(eax, 0x3f).unwrap();
    a.add(dword_ptr(eax * 4 + 0x6600), eax).unwrap();
    a.mov(ecx, 8u32).unwrap();
    a.set_label(&mut table).unwrap();
    a.add(dword_ptr(ecx * 4 + 0x6700), ecx).unwrap();
    a.dec(ecx).unwrap();
    a.jne(table).unwrap();
    a.pop(ds).unwrap();
    a.pop(ecx).unwrap();
    a.pop(eax).unwrap();
    a.iretd().unwrap();
}

/// Map CR2's page with A and D clear, user read-write.
fn page_fault(a: &mut CodeAssembler) {
    a.push(eax).unwrap();
    a.push(ebx).unwrap();
    a.push(ds).unwrap();
    a.mov(ax, 0x10).unwrap();
    a.mov(ds, ax).unwrap();
    a.mov(eax, cr2).unwrap();
    a.shr(eax, 12).unwrap();
    a.mov(ebx, eax).unwrap();
    a.shl(ebx, 12).unwrap();
    a.or(ebx, 7).unwrap();
    a.mov(dword_ptr(eax * 4 + PT as i32), ebx).unwrap();
    a.inc(dword_ptr(DATA + 0x104)).unwrap();
    a.pop(ds).unwrap();
    a.pop(ebx).unwrap();
    a.pop(eax).unwrap();
    a.add(esp, 4).unwrap();
    a.iretd().unwrap();
}

/// INT 80h from ring 3 or INT 81h from V86: back to the kernel on its own
/// stack with flat ring-0 data segments.
fn service(a: &mut CodeAssembler, counter: u64, to: u64) {
    a.mov(esp, HIGH as u32 + STACK_TOP).unwrap();
    a.mov(ax, 0x10).unwrap();
    a.mov(ds, ax).unwrap();
    a.mov(es, ax).unwrap();
    a.mov(fs, ax).unwrap();
    a.mov(gs, ax).unwrap();
    a.inc(dword_ptr(counter)).unwrap();
    a.jmp(HIGH + to).unwrap();
}

fn user(a: &mut CodeAssembler) {
    let mut top = a.create_label();
    a.mov(ecx, 300u32).unwrap();
    a.set_label(&mut top).unwrap();
    a.add(eax, ecx).unwrap();
    a.mov(ebx, ecx).unwrap();
    a.and(ebx, 0xff).unwrap();
    a.add(dword_ptr(ebx * 4 + 0x21000), eax).unwrap();
    a.call(USER_SUB).unwrap();
    a.dec(ecx).unwrap();
    a.jne(top).unwrap();
    a.mov(esi, 0x21000u32).unwrap();
    a.mov(edi, 0x21800u32).unwrap();
    a.mov(ecx, 400u32).unwrap();
    a.cld().unwrap();
    a.rep().movsd().unwrap();
    // Demand pages from CPL 3.
    a.mov(esi, DEMAND + 0x3_0010).unwrap();
    a.add(eax, dword_ptr(esi)).unwrap();
    a.add(dword_ptr(esi + 0x1000), eax).unwrap();
    a.int(0x80).unwrap();
}

/// The V86 task at IOPL 3: IF-sensitive instructions execute (80386 PRM
/// §15.2.1), INT 81h goes to its ring-0 gate (SDM Vol 3B §20.3.3).
fn v86() -> Vec<u8> {
    let mut a = CodeAssembler::new(16).unwrap();
    let mut top = a.create_label();
    a.mov(cx, 200u32).unwrap();
    a.set_label(&mut top).unwrap();
    a.add(ax, cx).unwrap();
    a.mov(bx, cx).unwrap();
    a.and(bx, 0x7e).unwrap();
    a.add(word_ptr(bx), ax).unwrap();
    a.xor(dx, ax).unwrap();
    a.pushf().unwrap();
    a.pop(si).unwrap();
    a.push(si).unwrap();
    a.popf().unwrap();
    a.cli().unwrap();
    a.inc(di).unwrap();
    a.sti().unwrap();
    a.add(bp, dx).unwrap();
    a.loop_(top).unwrap();
    a.int(0x81).unwrap();
    a.assemble(0).unwrap()
}

fn build(subject: Subject) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(MEMORY), Box::new(JitBackend::new()));
    vm.map(0, MMIO as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.map(MMIO, 0x1000, Prot::RW, RegionKind::Trap).unwrap();
    vm.map(
        MMIO + 0x1000,
        (MEMORY - MMIO - 0x1000) as usize,
        Prot::RWX,
        RegionKind::Ram,
    )
    .unwrap();
    vm.set_cpu_model(CpuModel::I80486Dx);
    vm.register_device_range(DeviceRange::Mmio(MMIO..MMIO + 0x100))
        .unwrap();
    vm.register_device_range(DeviceRange::Ports(0x40..0x44))
        .unwrap();
    vm.set_exception_delivery(ExceptionDelivery::InGuest);
    match subject {
        Subject::Mode(mode) => vm.set_machine_jit_for_test(mode),
        Subject::Background(t1) => {
            vm.set_machine_jit_for_test(MachineJitMode::Off);
            vm.set_tier_up_after(Some(t1));
            vm.set_tier_up_background(true);
        }
    }
    let put = |addr: u64, bytes: &[u8]| vm.write_bytes(addr, bytes).unwrap();
    // Paging (80386 PRM §5.2): one table, A and D clear everywhere.
    put(PD, &(PT as u32 | 7).to_le_bytes());
    put(PD + 4 * (HIGH >> 22), &(PT as u32 | 3).to_le_bytes());
    for page in 0..(MEMORY >> 12) as u32 {
        let entry = match page {
            0x20..=0x23 | 0x30..=0x32 => (page << 12) | 7,
            _ if (DEMAND >> 12..(DEMAND >> 12) + 64).contains(&page) => 0,
            _ if page >= 0x140 => (page << 12) | 7,
            _ => (page << 12) | 3,
        };
        put(PT + 4 * u64::from(page), &entry.to_le_bytes());
    }
    for (selector, descriptor) in [
        (0x08u64, 0x00cf_9b00_0000_ffffu64),
        (0x10, 0x00cf_9300_0000_ffff),
        (0x18, 0x00cf_fb00_0000_ffff),
        (0x20, 0x00cf_f300_0000_ffff),
        // The busy 386 TSS at 5000h.
        (0x28, 0x0000_8b00_5000_0067),
    ] {
        put(GDT + selector, &descriptor.to_le_bytes());
    }
    let gate = |offset: u64, attributes: u64| {
        (offset & 0xffff) | (8 << 16) | (attributes << 32) | ((offset >> 16) << 48)
    };
    for vector in 0..256u64 {
        let entry = match vector {
            14 => gate(HIGH + PAGE_FAULT, 0x8e00),
            0x20 => gate(HIGH + TIMER, 0x8e00),
            0x80 => gate(HIGH + INT80, 0xee00),
            0x81 => gate(HIGH + INT81, 0xee00),
            _ => gate(HIGH + STRAY, 0x8e00),
        };
        put(IDT + vector * 8, &entry.to_le_bytes());
    }
    put(TSS + 4, &(HIGH as u32 + STACK_TOP).to_le_bytes());
    put(TSS + 8, &0x10u32.to_le_bytes());
    put(TSS + 0x66, &0x68u16.to_le_bytes());
    if SHAPES.with(std::cell::Cell::get) {
        put(KERNEL, &asm32(HIGH + KERNEL, shapes_loop));
        put(ROUTINE, &asm32(HIGH + ROUTINE, |a| routine(a, false)));
        put(
            ROUTINE + 0x100,
            &asm32(HIGH + ROUTINE + 0x100, |a| routine(a, true)),
        );
        put(
            ROUTINE + 0x200,
            &asm32(HIGH + ROUTINE + 0x200, |a| {
                a.jmp(HIGH + KERNEL + 1).unwrap();
            }),
        );
        put(
            ROUTINE + 0x300,
            &asm32(HIGH + ROUTINE + 0x300, compare_first),
        );
        put(
            ROUTINE + 0x400,
            &asm32(HIGH + ROUTINE + 0x400, compare_second),
        );
        put(
            ROUTINE + 0x500,
            &asm32(HIGH + ROUTINE + 0x500, |a| {
                a.inc(dword_ptr(DATA + 0x64)).unwrap();
                a.ret().unwrap();
            }),
        );
        put(
            DATA + 0x68,
            &(HIGH as u32 + ROUTINE as u32 + 0x500).to_le_bytes(),
        );
        put(DATA + 0x70, &(HIGH as u32 + 0x7000).to_le_bytes());
        // The routines' data: flag words, a structure, an IDT image, an
        // alternate stack (`shapes_loop`).
        put(DATA + 0x40, &(DATA as u32 + 0x200).to_le_bytes());
        put(DATA + 0x44, &(DATA as u32 + 0x300).to_le_bytes());
        put(DATA + 0x204, &(HIGH as u32 + 0x7000).to_le_bytes());
        put(DATA + 0x50, &0x7ffu16.to_le_bytes());
        put(DATA + 0x52, &(IDT as u32).to_le_bytes());
        put(DATA + 0x30, &(HIGH as u32 + 0xd000).to_le_bytes());
    } else {
        put(KERNEL, &asm32(HIGH + KERNEL, kernel_loop));
    }
    put(AFTER_USER, &asm32(HIGH + AFTER_USER, after_user));
    put(AFTER_V86, &asm32(HIGH + AFTER_V86, after_v86));
    put(KSUB, &asm32(HIGH + KSUB, ksub));
    put(
        CALLER,
        &asm32(HIGH + CALLER, |a| {
            for f in 0..FUNCTIONS {
                a.call(HIGH + FUNCTION + 32 * f).unwrap();
            }
            a.ret().unwrap();
        }),
    );
    for f in 0..FUNCTIONS {
        put(
            FUNCTION + 32 * f,
            &asm32(HIGH + FUNCTION + 32 * f, |a| {
                a.add(eax, f as u32 as i32).unwrap();
                a.xor(dword_ptr(DATA + 0x800 + 4 * (f % 64)), eax).unwrap();
                a.rol(eax, (f % 7) as u32).unwrap();
                a.ret().unwrap();
            }),
        );
    }
    put(
        WINDOWS,
        &asm32(HIGH + WINDOWS, |a| {
            a.cli().unwrap();
            a.inc(eax).unwrap();
            a.sti().unwrap();
            a.ret().unwrap();
        }),
    );
    put(
        WINDOWS + 0x10,
        &asm32(HIGH + WINDOWS + 0x10, |a| {
            a.pushfd().unwrap();
            a.cli().unwrap();
            a.inc(dword_ptr(DATA + 0x18)).unwrap();
            a.popfd().unwrap();
            a.ret().unwrap();
        }),
    );
    put(
        PATCH,
        &asm32(HIGH + PATCH, |a| {
            a.mov(ecx, 150u32).unwrap();
            a.ret().unwrap();
        }),
    );
    put(TIMER, &asm32(HIGH + TIMER, timer));
    put(PAGE_FAULT, &asm32(HIGH + PAGE_FAULT, page_fault));
    put(
        INT80,
        &asm32(HIGH + INT80, |a| service(a, DATA + 0x20, AFTER_USER)),
    );
    put(
        INT81,
        &asm32(HIGH + INT81, |a| service(a, DATA + 0x24, AFTER_V86)),
    );
    put(STRAY, &[0xf4, 0xeb, 0xfd]);
    put(USER_CODE, &asm32(USER_CODE, user));
    put(
        USER_SUB,
        &asm32(USER_SUB, |a| {
            a.xor(edx, eax).unwrap();
            a.add(dword_ptr(USER_SUB + 0x400), edx).unwrap();
            a.ret().unwrap();
        }),
    );
    put(u64::from(V86_CS) << 4, &v86());
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = HIGH + KERNEL;
    cpu.cpu.legacy.cr0 = 0x8000_0001;
    cpu.cpu.legacy.cr3 = PD as u32;
    cpu.cpu.legacy.gdtr = TableRegister {
        base: GDT as u32,
        limit: 0x3f,
        reserved: 0,
    };
    cpu.cpu.legacy.idtr = TableRegister {
        base: IDT as u32,
        limit: 0x7ff,
        reserved: 0,
    };
    cpu.cpu.legacy.tr = SystemSegment {
        selector: 0x28,
        reserved: 0,
        cache: SegmentCache {
            base: TSS as u32,
            limit: 0x67,
            attributes: 0x8b,
            ..Default::default()
        },
    };
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        cache.base = 0;
        cache.limit = u32::MAX;
        cache.attributes = if i == 0 { 0xc09b } else { 0xc093 };
    }
    cpu.cpu.cs = 8;
    cpu.cpu.ds = 0x10;
    cpu.cpu.es = 0x10;
    cpu.cpu.ss = 0x10;
    cpu.cpu.fs = 0x10;
    cpu.cpu.gs = 0x10;
    cpu.cpu.legacy.cpl = 0;
    cpu.cpu.gpr[4] = HIGH + u64::from(STACK_TOP);
    cpu.cpu.flags.if_ = true;
    // IOPL 3 in ring 0, as some protected-mode operating systems run.
    cpu.cpu.rflags_system |= 0x3000;
    cpu.set_a20_enabled(true);
    if DELIVERY_LOG.with(std::cell::Cell::get) {
        cpu.enable_delivery_log(64).unwrap();
    }
    (vm, cpu)
}

fn hash(vm: &Vm) -> u64 {
    let mut bytes = vec![0; MMIO as usize];
    vm.read_bytes(0, &mut bytes).unwrap();
    let mut rest = vec![0; (MEMORY - MMIO - 0x1000) as usize];
    vm.read_bytes(MMIO + 0x1000, &mut rest).unwrap();
    bytes.extend(rest);
    let (words, _) = bytes.as_chunks::<8>();
    words.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, c| {
        (h ^ u64::from_le_bytes(*c)).wrapping_mul(0x0100_0000_01b3)
    })
}

thread_local! {
    /// Whether [`run_with`] serves the device in process.
    static DEVICES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Whether [`build`] runs [`shapes_loop`] instead of [`kernel_loop`].
    static SHAPES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Whether [`build`] enables the delivery log (spec.md §17.10).
    static DELIVERY_LOG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The in-process device: reads answer from the request's address and
/// position, and some accesses act on the run: a write at +4 requests a host
/// stop, a read at +8 sets a step deadline, a write at +0Ch writes RAM.
#[derive(Default)]
struct Device {
    transcript: Vec<DeviceRequest>,
}

impl DeviceIo for Device {
    fn access(&mut self, r: &DeviceRequest, context: &mut DeviceContext<'_>) -> DeviceResult {
        self.transcript.push(r.clone());
        let n = self.transcript.len() as u64;
        let value = match (r.space, r.address) {
            (DeviceSpace::Mmio, a) if a == MMIO + 0x10 => {
                if n.is_multiple_of(5) {
                    context.request_host_stop();
                } else if n.is_multiple_of(3) {
                    let _ = context.set_deadline(DeviceDeadline::Steps(r.position.steps + 9));
                }
                r.position.retired_instructions & 0xff
            }
            (DeviceSpace::Mmio, a) if a == MMIO + 4 => {
                if n.is_multiple_of(11) {
                    context.request_host_stop();
                }
                0
            }
            (DeviceSpace::Mmio, a) if a == MMIO + 8 => {
                if n.is_multiple_of(7) {
                    let _ = context.set_deadline(DeviceDeadline::Steps(r.position.steps + 40));
                }
                r.position.steps & 0xffff
            }
            (DeviceSpace::Mmio, a) if a == MMIO + 0xc => {
                let word = (r.value ^ r.position.retired_instructions) as u32;
                context
                    .write_ram(DATA + 0x900 + (n % 8) * 4, &word.to_le_bytes())
                    .unwrap();
                0
            }
            (_, a) => a ^ r.position.retired_instructions,
        };
        DeviceResult::Handled(value)
    }
}

/// Answer a stop identically on both machines.
fn answer(cpu: &mut Vcpu, stop: &StepStop, at: &str) {
    match stop {
        StepStop::Exit(Exit::MmioRead { addr, .. }) => cpu.complete_mmio_read(*addr ^ 0x5a5a),
        StepStop::Exit(Exit::MmioWrite { .. }) => cpu.complete_mmio_write(),
        StepStop::Exit(Exit::PortIo { port, .. }) => cpu.complete_port_in(u64::from(*port) * 3),
        StepStop::Exit(
            Exit::BudgetExhausted | Exit::Interrupted | Exit::Device(DeviceStop::Deadline { .. }),
        )
        | StepStop::IrqReady => {}
        other => panic!("{at}: {other:?}"),
    }
}

/// The pages every slice compares: tables, kernel data and stack, the ring-3
/// data and stack, the V86 data and stack.
const HOT: [(u64, u64); 4] = [
    (PD, PT + 0x1000),
    (DATA, 0x8000),
    (0xe000, 0xf000),
    (0x21000, 0x23000),
];

fn same_hot(a: &Vm, b: &Vm, what: &str) {
    super::machine_j1_lockstep::same_memory(a, b, &HOT, what);
    super::machine_j1_lockstep::same_memory(a, b, &[(0x31000, 0x33000)], what);
}

/// Run the guest for `total` steps in slices from `budgets`, injecting the
/// timer every `period` retired instructions, in lockstep with the
/// interpreter; the subject's counters.
fn run(
    subject: Subject,
    budgets: &mut dyn FnMut(usize) -> u64,
    total: u64,
    period: u64,
    what: &str,
) -> MachineJitCounters {
    run_with(subject, budgets, total, period, what, &mut |_| {})
}

/// `run`, with `before` called ahead of every subject call.
fn run_with(
    subject: Subject,
    budgets: &mut dyn FnMut(usize) -> u64,
    total: u64,
    period: u64,
    what: &str,
    before: &mut dyn FnMut(usize),
) -> MachineJitCounters {
    let (bvm, mut b) = build(Subject::Mode(MachineJitMode::Off));
    let (avm, mut a) = build(subject);
    a.enable_class_census_for_test();
    let devices = DEVICES.with(std::cell::Cell::get);
    let (mut subject_device, mut reference_device) = (Device::default(), Device::default());
    let mut next_tick = period;
    // An embedder's interrupt controller: a request raised at a fixed
    // retired count is injected between calls once the vCPU can take it;
    // until then each call stops at the first boundary that can.
    let mut requested = false;
    let mut steps = 0;
    let mut slice = 0;
    while steps < total {
        if b.retired_instructions() >= next_tick {
            requested = true;
            next_tick += period;
        }
        assert_eq!(a.irq_ready(), b.irq_ready(), "{what} slice {slice}");
        if requested && b.irq_ready() {
            a.inject_irq(TIMER_VECTOR);
            b.inject_irq(TIMER_VECTOR);
            requested = false;
        }
        let options = StepOptions::default().with_stop_when_irq_ready(requested);
        let budget = budgets(slice);
        before(slice);
        let (ar, br) = if devices {
            (
                a.run_steps_with_devices(&avm, budget, options, Some(&mut subject_device))
                    .unwrap()
                    .into_run(),
                b.run_steps_with_devices(&bvm, budget, options, Some(&mut reference_device))
                    .unwrap()
                    .into_run(),
            )
        } else {
            (
                a.run_steps(&avm, budget, options).unwrap(),
                b.run_steps(&bvm, budget, options).unwrap(),
            )
        };
        let at = format!(
            "{what} slice {slice} budget {budget} retired {} at {:x}:{:x}",
            b.retired_instructions(),
            b.cpu.cs,
            b.cpu.rip
        );
        assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{at}");
        assert_eq!(a.retired_instructions(), b.retired_instructions(), "{at}");
        assert_eq!(live(&a.cpu), live(&b.cpu), "{at}");
        assert_eq!(
            a.paging_tlb_for_test(),
            b.paging_tlb_for_test(),
            "{at}: TLB"
        );
        same_hot(&avm, &bvm, &at);
        if budget >= 997 || slice % 61 == 0 {
            assert_eq!(hash(&avm), hash(&bvm), "{at}: memory");
        }
        assert!(b.cpu.rip & !1 != HIGH + STRAY, "{at}: stray");
        assert_eq!(
            subject_device.transcript, reference_device.transcript,
            "{at}: device transcripts"
        );
        answer(&mut a, br.stop(), &at);
        answer(&mut b, br.stop(), &at);
        assert_eq!(live(&a.cpu), live(&b.cpu), "{at}: answered");
        steps += br.steps();
        slice += 1;
    }
    assert_eq!(hash(&avm), hash(&bvm), "{what}: memory at the end");
    let mut vars = [0u8; 0x110];
    bvm.read_bytes(DATA, &mut vars).unwrap();
    let word = |at: usize| u32::from_le_bytes(vars[at..at + 4].try_into().unwrap());
    // The guest went round: ring 3, V86, timer ticks and demand faults.
    assert!(
        SHAPES.with(std::cell::Cell::get)
            || (word(0x20) > 0 && word(0x24) > 0 && word(0x100) > 0 && word(0x104) > 0),
        "{what}: iterations {} int80 {} int81 {} ticks {} faults {}",
        word(0),
        word(0x20),
        word(0x24),
        word(0x100),
        word(0x104)
    );
    let c = a.machine_jit_counters_for_test();
    println!(
        "{what}: retired {}, compiled_retired {}",
        a.retired_instructions(),
        c.compiled_retired
    );
    for ((name, n), r) in MachineJitCounters::REFUSAL_NAMES
        .iter()
        .zip(c.refusals)
        .zip(c.refusals_retired)
    {
        if n != 0 {
            println!("  refusal {name}: {n} turns, {r} retired");
        }
    }
    if let Some(census) = a.class_census_for_test() {
        let mut rows: Vec<_> = census.classes.into_iter().collect();
        rows.sort_by_key(|row| std::cmp::Reverse(row.1 .0));
        for ((row, class), (turns, retired)) in rows {
            println!("  class {row} / {class}: {turns} turns, {retired} retired");
        }
    }
    c
}

const SUBJECTS: [Subject; 5] = [
    Subject::Mode(MachineJitMode::Forced),
    Subject::Mode(MachineJitMode::Delayed(1)),
    Subject::Mode(MachineJitMode::Delayed(3)),
    Subject::Mode(MachineJitMode::Scripted(7)),
    Subject::Background(50),
];

fn sliced(budget: u64, total: u64) {
    for subject in SUBJECTS {
        let what = format!("{subject:?} slices of {budget}");
        let c = run(subject, &mut |_| budget, total, 3001, &what);
        if subject == Subject::Mode(MachineJitMode::Forced) && budget >= 97 {
            assert!(
                c.compiled_retired > 0 && c.paged_entries > 0 && c.compiled_retired_cpl3 > 0,
                "{what}: {c:?}"
            );
        }
    }
}

#[test]
fn an_operating_system_in_slices_of_1_equals_the_interpreter() {
    sliced(1, 30_000);
}

#[test]
fn an_operating_system_in_slices_of_7_equals_the_interpreter() {
    sliced(7, 60_000);
}

#[test]
fn an_operating_system_in_slices_of_97_equals_the_interpreter() {
    sliced(97, 200_000);
}

#[test]
fn an_operating_system_in_slices_of_997_equals_the_interpreter() {
    sliced(997, 300_000);
}

#[test]
fn an_operating_system_in_slices_of_10000_equals_the_interpreter() {
    sliced(10_000, 400_000);
}

#[test]
fn an_operating_system_in_mixed_slices_equals_the_interpreter() {
    const MIX: [u64; 9] = [10_000, 1, 997, 7, 97, 3, 10_000, 64, 2];
    for subject in SUBJECTS {
        let what = format!("{subject:?} mixed slices");
        run(subject, &mut |i| MIX[i % MIX.len()], 300_000, 2003, &what);
    }
}

/// Requests every 97 retired instructions: most arrive with IF clear, so
/// calls stop at the first boundary that can take them.
#[test]
fn an_operating_system_under_frequent_interrupts_equals_the_interpreter() {
    const MIX: [u64; 7] = [10_000, 997, 65_536, 97, 7, 4093, 1];
    for subject in SUBJECTS {
        let what = format!("{subject:?} frequent interrupts");
        run(subject, &mut |i| MIX[i % MIX.len()], 300_000, 97, &what);
    }
}

/// The state at every `period` steps of a run whose calls take budgets from
/// `budgets`, clipped to end at those points, with a request raised there.
fn checkpoints(
    subject: Subject,
    budgets: &dyn Fn(usize) -> u64,
    total: u64,
    period: u64,
) -> Vec<(u64, String, u64)> {
    let (vm, mut cpu) = build(subject);
    let mut out = Vec::new();
    let mut requested = false;
    let mut steps = 0;
    let mut call = 0;
    while steps < total {
        if steps % period == 0 && out.len() as u64 == steps / period {
            out.push((cpu.retired_instructions(), live(&cpu.cpu), hash(&vm)));
            requested = true;
        }
        if requested && cpu.irq_ready() {
            cpu.inject_irq(TIMER_VECTOR);
            requested = false;
        }
        let budget = budgets(call).min(period - steps % period);
        let options = StepOptions::default().with_stop_when_irq_ready(requested);
        let r = cpu.run_steps(&vm, budget, options).unwrap();
        answer(&mut cpu, r.stop(), "checkpoints");
        steps += r.steps();
        call += 1;
    }
    out
}

/// Slicing is invisible: the interpreter and every compiled mode reach the
/// same state at every checkpoint whatever budgets their calls take.
#[test]
fn an_operating_system_is_invariant_under_slicing() {
    const MIX: [u64; 9] = [10_000, 1, 997, 7, 97, 3, 10_000, 64, 2];
    let mix = |i: usize| MIX[i % MIX.len()];
    let reference = checkpoints(
        Subject::Mode(MachineJitMode::Off),
        &|_| 10_000,
        200_000,
        5000,
    );
    type Budgets<'a> = &'a dyn Fn(usize) -> u64;
    let runs: [(&str, Subject, Budgets); 8] = [
        ("off 1", Subject::Mode(MachineJitMode::Off), &|_| 1),
        ("off 7", Subject::Mode(MachineJitMode::Off), &|_| 7),
        ("off 97", Subject::Mode(MachineJitMode::Off), &|_| 97),
        ("off mixed", Subject::Mode(MachineJitMode::Off), &mix),
        (
            "forced 10000",
            Subject::Mode(MachineJitMode::Forced),
            &|_| 10_000,
        ),
        ("forced mixed", Subject::Mode(MachineJitMode::Forced), &mix),
        (
            "delayed mixed",
            Subject::Mode(MachineJitMode::Delayed(3)),
            &mix,
        ),
        ("background 10000", Subject::Background(50), &|_| 10_000),
    ];
    for (name, subject, budgets) in runs {
        let got = checkpoints(subject, budgets, 200_000, 5000);
        for (i, (a, b)) in got.iter().zip(&reference).enumerate() {
            assert_eq!(a.0, b.0, "{name}: retired at checkpoint {i}");
            assert_eq!(a.1, b.1, "{name}: state at checkpoint {i}");
            assert_eq!(a.2, b.2, "{name}: memory at checkpoint {i}");
        }
    }
}

/// Long enough for background tier-up at its production threshold to
/// publish most of the working set while the run goes on.
#[test]
fn an_operating_system_with_background_tier_up_equals_the_interpreter() {
    const MIX: [u64; 5] = [10_000, 65_536, 9_973, 4_096, 12_345];
    for subject in [Subject::Background(50), Subject::Background(2)] {
        let what = format!("{subject:?} long run");
        run(subject, &mut |i| MIX[i % MIX.len()], 1_500_000, 3001, &what);
    }
}

/// Engine events at chain edges that change nothing the guest sees: a unit
/// retirement (the epoch advances, slots and the return ring are dropped),
/// exhausted code space, a write of unchanged bytes into running code.
#[test]
fn an_operating_system_with_engine_events_at_chain_edges_equals_the_interpreter() {
    const MIX: [u64; 5] = [10_000, 997, 4_093, 97, 65_536];
    for action in [
        ChainEdgeAction::Retire,
        ChainEdgeAction::CodeSpaceExhausted,
        ChainEdgeAction::CodeWrite(KERNEL + 0x40),
        ChainEdgeAction::CodeWrite(FUNCTION + 0x200),
    ] {
        for subject in [
            Subject::Mode(MachineJitMode::Forced),
            Subject::Background(2),
        ] {
            let what = format!("{subject:?} {action:?}");
            let mut z = 0x9e37_79b9u64;
            run_with(
                subject,
                &mut |i| MIX[i % MIX.len()],
                400_000,
                3001,
                &what,
                &mut |_| {
                    z = z.wrapping_mul(6364136223846793005).wrapping_add(1);
                    Vcpu::raise_at_chain_edge_for_test((z >> 33) % 200, action);
                },
            );
        }
    }
}

/// The guest's device served in process (decision-28): compiled paged
/// blocks reach it inside their block, and its effects (a host stop, a
/// step deadline, a RAM write) end the run where the interpreter's would.
#[test]
fn an_operating_system_with_in_process_devices_equals_the_interpreter() {
    DEVICES.with(|d| d.set(true));
    const MIX: [u64; 7] = [10_000, 997, 65_536, 97, 7, 4093, 1];
    for subject in SUBJECTS {
        let what = format!("{subject:?} devices");
        run(subject, &mut |i| MIX[i % MIX.len()], 300_000, 211, &what);
    }
    for subject in [Subject::Background(50), Subject::Background(2)] {
        let what = format!("{subject:?} devices long run");
        run(
            subject,
            &mut |i| [10_000, 65_536, 9_973][i % 3],
            1_500_000,
            3001,
            &what,
        );
    }
    DEVICES.with(|d| d.set(false));
}

/// With the delivery log on, compiled blocks take the cold epilogue: no
/// chains, and every completion goes through the outcome arm.
#[test]
fn an_operating_system_with_the_delivery_log_equals_the_interpreter() {
    DELIVERY_LOG.with(|d| d.set(true));
    const MIX: [u64; 7] = [10_000, 997, 65_536, 97, 7, 4093, 1];
    for subject in SUBJECTS {
        let what = format!("{subject:?} delivery log");
        run(subject, &mut |i| MIX[i % MIX.len()], 300_000, 2003, &what);
    }
    DELIVERY_LOG.with(|d| d.set(false));
}

fn run_shapes(subject: Subject, budgets: &mut dyn FnMut(usize) -> u64, total: u64, what: &str) {
    // No interrupts: the captured slices took none.
    let c = run(subject, budgets, total, u64::MAX / 2, what);
    if subject == Subject::Mode(MachineJitMode::Forced) {
        assert!(
            c.compiled_retired > 0 && c.paged_entries > 0,
            "{what}: {c:?}"
        );
    }
}

/// The shapes an embedder's operating system accrued phantom counts in:
/// PUSHAD ... LIDT, STD, backward REP MOVSD, CLD ... POPAD, RET twins, and
/// XCHG ESP blocks around a compare, a STI window and an indirect jump,
/// with the delivery log off and on.
#[test]
fn routine_shapes_equal_the_interpreter() {
    SHAPES.with(|d| d.set(true));
    const MIX: [u64; 7] = [10_000, 997, 65_536, 97, 7, 4093, 1];
    for log in [false, true] {
        DELIVERY_LOG.with(|d| d.set(log));
        for subject in SUBJECTS {
            let what = format!("{subject:?} shapes log={log}");
            run_shapes(subject, &mut |i| MIX[i % MIX.len()], 200_000, &what);
        }
    }
    DELIVERY_LOG.with(|d| d.set(false));
    SHAPES.with(|d| d.set(false));
}

/// The same shapes with the compared variable in a device page served in
/// process (decision-28; decision-32 Q9: paged compiled accesses reach it
/// through `in_block_access`), the device's stops and deadlines included.
#[test]
fn routine_shapes_with_devices_equal_the_interpreter() {
    SHAPES.with(|d| d.set(true));
    DEVICES.with(|d| d.set(true));
    const MIX: [u64; 7] = [10_000, 997, 65_536, 97, 7, 4093, 1];
    for log in [false, true] {
        DELIVERY_LOG.with(|d| d.set(log));
        for subject in SUBJECTS {
            let what = format!("{subject:?} shapes devices log={log}");
            run_shapes(subject, &mut |i| MIX[i % MIX.len()], 200_000, &what);
        }
    }
    DELIVERY_LOG.with(|d| d.set(false));
    DEVICES.with(|d| d.set(false));
    SHAPES.with(|d| d.set(false));
}

/// xj-to9.80: a compiled unit is the block of one lift. A block ends before
/// a non-ordinary instruction (here CLI); when the guest patches that
/// instruction into an ordinary one (here INC ESI) — a protected-mode
/// operating system's dynamic link patches its service traps in place — the
/// unit's own bytes are unchanged and it stays valid, but the next lift at
/// the same address is longer. Entering the shorter unit for the longer
/// block would charge instructions it never ran.
#[test]
fn a_block_that_grows_past_a_patched_instruction_gets_its_own_unit() {
    use super::machine_j1_lockstep::{asm, machine, same, Setup};
    let mut patch = 0;
    let code = asm(|a| {
        let mut top = a.create_label();
        let mut at = a.create_label();
        a.mov(ebp, 400u32).unwrap();
        a.set_label(&mut top).unwrap();
        a.add(eax, 1).unwrap();
        a.add(ebx, eax).unwrap();
        a.xor(ecx, ebx).unwrap();
        a.add(edx, ecx).unwrap();
        a.set_label(&mut at).unwrap();
        a.cli().unwrap();
        a.inc(edi).unwrap();
        a.dec(ebp).unwrap();
        a.jne(top).unwrap();
        a.hlt().unwrap();
        let _ = &mut at;
    });
    // The CLI: the only FAh byte of the program.
    patch += 0x1000 + code.iter().position(|&b| b == 0xfa).unwrap() as u64;
    let setup = Setup {
        paged: true,
        ..Setup::default()
    };
    {
        let paged = setup.paged;
        for subject in [
            Subject::Mode(MachineJitMode::Forced),
            Subject::Mode(MachineJitMode::Delayed(1)),
            Subject::Background(1),
        ] {
            let what = format!("{subject:?} paged={paged}");
            let model = CpuModel::I80486Dx;
            let (bvm, mut b) = machine(
                JitBackend::new(),
                model,
                &code,
                Subject::Mode(MachineJitMode::Off),
                setup,
                5,
            );
            let (avm, mut a) = machine(JitBackend::new(), model, &code, subject, setup, 5);
            for call in 0..400 {
                let at = format!("{what} call {call}");
                if call == 8 {
                    // INC ESI over the CLI, then every proof lifts again.
                    for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                        vm.write_bytes(patch, &[0x46]).unwrap();
                        cpu.invalidate_paging();
                    }
                }
                let ar = a.run_steps(&avm, 64, StepOptions::default()).unwrap();
                let br = b.run_steps(&bvm, 64, StepOptions::default()).unwrap();
                assert_eq!(format!("{ar:?}"), format!("{br:?}"), "{at}");
                same(&a, &b, &at);
                if matches!(br.stop(), StepStop::Exit(Exit::Hlt)) {
                    break;
                }
            }
        }
    }
}
