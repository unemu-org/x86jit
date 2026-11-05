//! Measurement-only access loops; not part of the shipped performance gate.
use super::workloads::{Counters, TierCfg, Workload};
use x86jit_core::{Backend, Exit, Prot, Reg, RegionKind, Vm, VmConfig};
const KCODE: u64 = 0x1000;
// Ordinary memory and stack controls for the alignment/cold-fault cost comparison.
pub fn all() -> Vec<Workload> {
    macro_rules! probe {
        ($name:literal, $guest:ident, $expect:literal) => {
            Workload {
                name: $name,
                kind: "memory-access",
                guest: $guest,
                native: None,
                expect: $expect,
            }
        };
    }
    vec![
        probe!("movaps", guest_movaps, b"movaps=35353535"),
        probe!("scalar_ld", guest_scalar_ld, b"probe=35353535"),
        probe!("scalar_st", guest_scalar_st, b"probe=35353535"),
        probe!("pushpop", guest_pushpop, b"probe=35353535"),
        probe!("rsp_ld", guest_rsp_ld, b"probe=35353535"),
        probe!("movups", guest_movups, b"probe=35353535"),
        probe!("addps_m128", guest_addps_m128, b"probe=007a1200"),
    ]
}

macro_rules! access_probe {
    ($guest:ident, $kind:literal) => {
        fn $guest(backend: Box<dyn Backend>, tier: TierCfg) -> (Vec<u8>, Counters) {
            guest_access_probe(backend, tier, $kind)
        }
    };
}
access_probe!(guest_scalar_ld, "scalar_ld");
access_probe!(guest_scalar_st, "scalar_st");
access_probe!(guest_pushpop, "pushpop");
access_probe!(guest_rsp_ld, "rsp_ld");
access_probe!(guest_movups, "movups");
access_probe!(guest_addps_m128, "addps_m128");

fn guest_access_probe(backend: Box<dyn Backend>, tier: TierCfg, kind: &str) -> (Vec<u8>, Counters) {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(64).unwrap();
    let mut body = a.create_label();
    a.mov(ebx, 0x4_0000u32).unwrap();
    a.mov(edi, 0x5_0000u32).unwrap();
    a.mov(ecx, 1_000_000u32).unwrap();
    a.mov(eax, 0x3535_3535u32).unwrap();
    a.mov(qword_ptr(rsp + 8), rax).unwrap();
    a.xorps(xmm0, xmm0).unwrap();
    a.set_label(&mut body).unwrap();
    for _ in 0..8 {
        match kind {
            "scalar_ld" => a.mov(rax, qword_ptr(rbx)).unwrap(),
            "scalar_st" => a.mov(qword_ptr(rbx), rax).unwrap(),
            "pushpop" => {
                a.push(rax).unwrap();
                a.pop(rax).unwrap();
            }
            "rsp_ld" => a.mov(rax, qword_ptr(rsp + 8)).unwrap(),
            "movups" => {
                a.movups(xmm0, xmmword_ptr(rbx)).unwrap();
                a.movups(xmmword_ptr(rdi), xmm0).unwrap();
            }
            "addps_m128" => a.addps(xmm0, xmmword_ptr(rbx)).unwrap(),
            _ => unreachable!(),
        }
    }
    a.dec(ecx).unwrap();
    a.jnz(body).unwrap();
    if kind == "addps_m128" {
        a.cvttss2si(eax, xmm0).unwrap();
    }
    if kind == "movups" {
        a.movd(eax, xmm0).unwrap();
    }
    a.hlt().unwrap();
    let data = if kind == "addps_m128" {
        1.0f32.to_le_bytes().repeat(4)
    } else {
        vec![0x35; 16]
    };
    let (value, counters) = run_code(
        &a.assemble(KCODE).unwrap(),
        &[(0x4_0000, data)],
        backend,
        tier,
    );
    (format!("probe={value:08x}").into_bytes(), counters)
}

fn guest_movaps(backend: Box<dyn Backend>, tier: TierCfg) -> (Vec<u8>, Counters) {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(64).unwrap();
    let mut body = a.create_label();
    a.mov(rsi, 0x4_0000u64).unwrap();
    a.mov(rdi, 0x5_0000u64).unwrap();
    a.mov(ecx, 1_000_000u32).unwrap();
    a.set_label(&mut body).unwrap();
    a.movaps(xmm0, xmmword_ptr(rsi)).unwrap();
    a.movaps(xmmword_ptr(rdi), xmm0).unwrap();
    a.dec(ecx).unwrap();
    a.jnz(body).unwrap();
    a.mov(eax, dword_ptr(rdi)).unwrap();
    a.hlt().unwrap();
    let (value, counters) = run_code(
        &a.assemble(KCODE).unwrap(),
        &[(0x4_0000, vec![0x35; 16])],
        backend,
        tier,
    );
    (format!("movaps={value:08x}").into_bytes(), counters)
}

fn run_code(
    code: &[u8],
    data: &[(u64, Vec<u8>)],
    backend: Box<dyn Backend>,
    tier: TierCfg,
) -> (u32, Counters) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10_0000), backend);
    vm.set_tier_up_after(tier.after);
    vm.set_tier_up_background(tier.background);
    vm.set_tier_up_region_after(tier.region_after);
    vm.map(0, 0x10_0000, Prot::RW, RegionKind::Ram).unwrap();
    for (addr, bytes) in data {
        vm.write_bytes(*addr, bytes).unwrap();
    }
    vm.write_bytes(KCODE, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, KCODE);
    cpu.set_reg(Reg::Rsp, 0x8_0000);
    loop {
        match cpu.run(&vm, None) {
            Exit::Hlt => break,
            Exit::BudgetExhausted => continue,
            other => panic!("kernel exited unexpectedly: {other:?}"),
        }
    }
    let counters = Counters {
        chained: vm.cache.chained(),
        ibtc_filled: vm.cache.ibtc_filled(),
        fast_hits: cpu.fast_hits(),
        misses: vm.cache.misses(),
        compile_ns: vm.backend.compile_ns(),
        executed: cpu.executed_instructions(),
        helper_calls: vm.backend.helper_calls().iter().map(|(_, n)| n).sum(),
        top_helper: vm
            .backend
            .helper_calls()
            .first()
            .map(|(name, _)| *name)
            .unwrap_or(""),
        top_helper_calls: vm
            .backend
            .helper_calls()
            .first()
            .map(|(_, n)| *n)
            .unwrap_or(0),
    };
    (cpu.reg(Reg::Rax) as u32, counters)
}
