//! Release-mode SMC scaling witness. Setup is excluded; one call/return and one
//! store per iteration. Filler translations occupy distinct pages. Arguments:
//! interp|jit, cached block count, data|overlap, store count (default 10000).
use iced_x86::code_asm::*;
use std::time::Instant;
use x86jit_core::{
    Backend, BlockKey, CachedBlock, CpuMode, Exit, InterpreterBackend, IrBlock, IrOp, Prot, Reg,
    RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let tier = args.get(1).map(String::as_str).unwrap_or("interp");
    let count: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let overlap = args.get(3).is_some_and(|s| s == "overlap");
    let stores: u64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(10000);
    let backend: Box<dyn Backend> = match tier {
        "interp" => Box::new(InterpreterBackend),
        "jit" => Box::new(JitBackend::new()),
        _ => panic!("expected interp|jit"),
    };
    let size = 0x10000 + count * 4096;
    let mut vm = Vm::with_backend(VmConfig::flat(size), backend);
    vm.set_tier_up_after(None);
    vm.map(0, size as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    // mov eax,42; ret. The overlap case rewrites the same immediate byte.
    vm.write_bytes(0x1000, &[0xb8, 42, 0, 0, 0, 0xc3]).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    let mut top = a.create_label();
    a.set_label(&mut top).unwrap();
    a.call(rbx).unwrap();
    a.mov(byte_ptr(rdi), 42).unwrap();
    a.dec(rcx).unwrap();
    a.jnz(top).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(0x4000, &a.assemble(0x4000).unwrap())
        .unwrap();
    // Populate the unrelated cache without measuring compilation/setup. These
    // entries are never executed; their spans and page tags are real cache state.
    for i in 0..count {
        let pc = 0x10000 + i * 4096;
        let ir = IrBlock {
            mode: CpuMode::Long64,
            guest_start: pc,
            guest_len: 1,
            icount: 1,
            temp_count: 0,
            ops: vec![IrOp::Hlt],
        };
        vm.cache.insert(
            BlockKey::new(pc, CpuMode::Long64),
            CachedBlock::Interpreted(std::sync::Arc::new(ir)),
            vec![(pc, 1)],
            |spans| {
                for &(start, len) in spans {
                    vm.mem.mark_code(start, len);
                }
            },
        );
    }
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rsp, 0x8000);
    cpu.set_reg(Reg::Rbx, 0x1000);
    cpu.set_reg(Reg::Rdi, if overlap { 0x1001 } else { 0x1800 });
    let mut run = |n| {
        cpu.set_reg(Reg::Rip, 0x4000);
        cpu.set_reg(Reg::Rcx, n);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 42);
    };
    run(100);
    assert!(
        !vm.mem.is_code_page((0x8000 - 8) >> 12),
        "stack must be data-only"
    );
    assert!(
        vm.cache
            .get(BlockKey::new(0x10000 + (count - 1) * 4096, CpuMode::Long64))
            .is_some(),
        "warmup must retain the filler translations"
    );
    let misses = vm.cache.misses();
    let start = Instant::now();
    run(stores);
    let secs = start.elapsed().as_secs_f64();
    let lifts = vm.cache.misses() - misses;
    println!(
        "{tier},{count},{},{stores},{:.3},{lifts},{:.1}",
        if overlap { "overlap" } else { "data" },
        secs * 1e9 / stores as f64,
        lifts as f64 / secs
    );
}
