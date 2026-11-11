//! Long64 REP MOVSB between RAM buffers in a VM with two device regions.
//! Release mode; optional arguments: 16 KiB copies (default 8192), then `ram-only`.
use iced_x86::code_asm::*;
use std::time::Instant;
use x86jit_core::{Exit, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;

fn main() {
    const CODE: u64 = 0x1000;
    const SOURCE: u64 = 0x8000;
    const DEST: u64 = 0x10000;
    const BYTES: u64 = 16384;
    let copies: u64 = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(8192);
    let ram_only = std::env::args().nth(2).is_some_and(|s| s == "ram-only");
    let mut vm = Vm::with_backend(VmConfig::flat(0x20000), Box::new(JitBackend::new()));
    for (start, len, kind) in [
        (0, 0x4000, RegionKind::Ram),
        (0x4000, 0x1000, RegionKind::Trap),
        (0x5000, 0x13000, RegionKind::Ram),
        (0x18000, 0x1000, RegionKind::Trap),
        (0x19000, 0x7000, RegionKind::Ram),
    ] {
        let kind = if ram_only { RegionKind::Ram } else { kind };
        vm.map(start, len, Prot::RWX, kind).unwrap();
    }
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    a.set_label(&mut top).unwrap();
    a.mov(ecx, BYTES as u32).unwrap();
    a.mov(esi, SOURCE as u32).unwrap();
    a.mov(edi, DEST as u32).unwrap();
    a.rep().movsb().unwrap();
    a.dec(ebx).unwrap();
    a.jnz(top).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
    vm.write_bytes(SOURCE, &vec![0x5a; BYTES as usize]).unwrap();
    let mut cpu = vm.new_vcpu();
    for iterations in [64, copies] {
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rbx, iterations);
        let start = Instant::now();
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        if iterations == copies {
            println!(
                "rep-movsb-{}-vm,{copies},{:.6},helpers={:?}",
                if ram_only { "ram" } else { "trap" },
                start.elapsed().as_secs_f64() * 1e9 / (copies * BYTES) as f64,
                vm.backend.helper_calls()
            );
        }
    }
    let mut dest = vec![0; BYTES as usize];
    vm.read_bytes(DEST, &mut dest).unwrap();
    assert!(dest.iter().all(|&v| v == 0x5a));
}
