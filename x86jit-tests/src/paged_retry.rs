//! The paged transaction lane and its fault retries on the Machine
//! interpreter (xj-to9.83), shared by the `machine32_paged_retry`
//! measurement and the allocation witness (`tests/paged_allocations.rs`).
//! A ring-0 paged loop on the I80486Dx:
//!
//! - [`Workload::Stack`]: instructions that take the two-phase transaction
//!   (PUSH and POP of memory, CALL/RET, PUSHAD/POPAD, MOVSD, CMPSD,
//!   ENTER/LEAVE).
//! - [`Workload::Fault`]: each iteration clears the present bit of four
//!   operand pages and runs a load, a MOVSD whose destination faults after
//!   its source read, a PUSH of memory whose source faults, and a POPAD whose
//!   fifth read crosses into an absent stack page. Each page fault goes to a
//!   #PF handler that sets the PTE present and returns with IRETD, and the
//!   instruction runs again (80386 PRM §9.8.14 pp. 9-17–19).
//!
//! Layout (identity-mapped by one table): page directory 1000h, table
//! 2000h, GDT 3000h, IDT 3800h, saved registers 6000h, stack below 9000h,
//! code 10000h, the #PF handler 18000h, operand pages 40000h-43000h and
//! 50000h-51000h.
use x86jit_core::legacy::SegmentCache;
use x86jit_core::{CpuModel, ExceptionDelivery, Prot, RegionKind, Vcpu, Vm, VmConfig};

/// Which loop [`setup`] builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Workload {
    Stack,
    Fault,
}

/// The loop's entry point.
pub const CODE: u32 = 0x10000;
/// Page faults per iteration of [`Workload::Fault`].
pub const FAULTS: u32 = 4;

const MEMORY: u64 = 0x10_0000;
const PD: u32 = 0x1000;
const PT: u32 = 0x2000;
const GDT: u32 = 0x3000;
const IDT: u32 = 0x3800;
const HANDLER: u32 = 0x18000;
const STACK: u32 = 0x9000;
// Operand pages of the `fault` loop.
const LOAD: u32 = 0x40000;
const SOURCE: u32 = 0x41000;
const DESTINATION: u32 = 0x42000;
const PUSHED: u32 = 0x43000;
// The stack page above `STACK`, which POPAD crosses into.
const UPPER: u32 = STACK;
const SAVED: u32 = 0x6000;

fn pte(page: u32) -> u32 {
    PT + 4 * (page >> 12)
}

fn stack_loop() -> Vec<u8> {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    let mut callee = a.create_label();
    a.set_label(&mut top).unwrap();
    a.mov(esi, 0x50000u32).unwrap();
    a.mov(edi, 0x51000u32).unwrap();
    a.push(dword_ptr(esi)).unwrap();
    a.pop(dword_ptr(edi)).unwrap();
    a.call(callee).unwrap();
    a.pushad().unwrap();
    a.popad().unwrap();
    a.movsd().unwrap();
    a.cmpsd().unwrap();
    a.enter(16, 0).unwrap();
    a.leave().unwrap();
    a.dec(ecx).unwrap();
    a.jne(top).unwrap();
    a.hlt().unwrap();
    a.set_label(&mut callee).unwrap();
    a.ret().unwrap();
    a.assemble(u64::from(CODE)).unwrap()
}

fn fault_loop() -> Vec<u8> {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(32).unwrap();
    let mut top = a.create_label();
    a.set_label(&mut top).unwrap();
    for page in [LOAD, DESTINATION, PUSHED, UPPER] {
        a.and(dword_ptr(pte(page)), !1u32 as i32).unwrap();
        a.invlpg(dword_ptr(page)).unwrap();
    }
    a.add(ebp, dword_ptr(LOAD + 0x10)).unwrap();
    a.mov(esi, SOURCE + 0x20).unwrap();
    a.mov(edi, DESTINATION + 0x40).unwrap();
    a.movsd().unwrap();
    a.push(dword_ptr(PUSHED + 0x80)).unwrap();
    a.pop(eax).unwrap();
    // POPAD from 10h below the absent page: its fifth read faults. It
    // loads the counter and keeps ESP, which are saved around it.
    a.mov(dword_ptr(SAVED), ebx).unwrap();
    a.mov(dword_ptr(SAVED + 4), esp).unwrap();
    a.mov(esp, STACK - 0x10).unwrap();
    a.popad().unwrap();
    a.mov(esp, dword_ptr(SAVED + 4)).unwrap();
    a.mov(ebx, dword_ptr(SAVED)).unwrap();
    a.dec(ebx).unwrap();
    a.jne(top).unwrap();
    a.hlt().unwrap();
    a.assemble(u64::from(CODE)).unwrap()
}

/// The #PF handler: set the faulting page's PTE present, drop the error
/// code and return to the faulting instruction (80386 PRM §9.8.14).
fn handler() -> Vec<u8> {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(32).unwrap();
    a.push(eax).unwrap();
    a.mov(eax, cr2).unwrap();
    a.shr(eax, 12).unwrap();
    a.or(dword_ptr(eax * 4 + PT as i32), 1).unwrap();
    a.pop(eax).unwrap();
    a.add(esp, 4).unwrap();
    a.iretd().unwrap();
    a.assemble(u64::from(HANDLER)).unwrap()
}

/// A VM and a vCPU at the loop's entry, ring 0, paged, running `iterations`
/// of `workload` to a HLT.
pub fn setup(workload: Workload, iterations: u32) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(MEMORY));
    vm.set_cpu_model(CpuModel::I80486Dx);
    vm.set_exception_delivery(ExceptionDelivery::InGuest);
    vm.map(0, MEMORY as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    let code = if workload == Workload::Stack {
        stack_loop()
    } else {
        fault_loop()
    };
    vm.write_bytes(u64::from(CODE), &code).unwrap();
    vm.write_bytes(u64::from(HANDLER), &handler()).unwrap();
    vm.write_bytes(u64::from(GDT) + 8, &0x00cf_9b00_0000_ffffu64.to_le_bytes())
        .unwrap();
    vm.write_bytes(u64::from(GDT) + 16, &0x00cf_9300_0000_ffffu64.to_le_bytes())
        .unwrap();
    // IDT vector 14: 32-bit interrupt gate, DPL 0, to 8:HANDLER.
    let gate =
        u64::from(HANDLER & 0xffff) | (8 << 16) | (0x8e00 << 32) | (u64::from(HANDLER >> 16) << 48);
    vm.write_bytes(u64::from(IDT) + 14 * 8, &gate.to_le_bytes())
        .unwrap();
    // 80386 PRM §5.2 pp. 5-7–11: one table identity-maps the first 4 MiB.
    vm.write_bytes(u64::from(PD), &(PT | 7).to_le_bytes())
        .unwrap();
    let mut entries = Vec::new();
    for page in 0..(MEMORY >> 12) as u32 {
        entries.extend(((page << 12) | 7).to_le_bytes());
    }
    vm.write_bytes(u64::from(PT), &entries).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.legacy.cr0 = 0x8000_0011;
    cpu.cpu.legacy.cr3 = PD;
    cpu.cpu.legacy.cpl = 0;
    cpu.cpu.cs = 8;
    (cpu.cpu.ds, cpu.cpu.es, cpu.cpu.ss, cpu.cpu.fs, cpu.cpu.gs) = (0x10, 0x10, 0x10, 0x10, 0x10);
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        *cache = SegmentCache {
            base: 0,
            limit: u32::MAX,
            attributes: (if i == 0 { 0x9b } else { 0x93 }) | 0x4000 | 0x8000,
            ..Default::default()
        };
    }
    cpu.cpu.legacy.gdtr.base = GDT;
    cpu.cpu.legacy.gdtr.limit = 0x17;
    cpu.cpu.legacy.idtr.base = IDT;
    cpu.cpu.legacy.idtr.limit = 0x7ff;
    cpu.set_a20_enabled(true);
    restart(&mut cpu, workload, iterations);
    (vm, cpu)
}

/// Put `cpu` back at the loop's entry with `iterations` to run.
pub fn restart(cpu: &mut Vcpu, workload: Workload, iterations: u32) {
    cpu.cpu.rip = u64::from(CODE);
    cpu.cpu.gpr[4] = u64::from(STACK - 0x100);
    cpu.cpu.gpr[5] = u64::from(STACK - 0x40);
    cpu.cpu.gpr[if workload == Workload::Stack { 1 } else { 3 }] = u64::from(iterations);
}
