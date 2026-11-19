//! SDM Vol 1 §8.5.1.2 and Vol 2A §3.3 FIST/FISTP, FISTTP:
//! an abandoned store must not notify watched memory.
use iced_x86::code_asm::*;
use x86jit_core::{Backend, Exit, InterpreterBackend, Prot, Reg, RegionKind, Vm, VmConfig};

#[test]
fn suppressed_float_stores_do_not_dirty_watched_memory() {
    // SDM Vol 1 §8.5.1.1: unmasked stack underflow preserves memory and TOP;
    // masking it permits the indefinite store and the instruction's usual pop.
    let mut unexpected_writes = Vec::new();
    for jit in [false, true] {
        for (width, pop) in [(4, false), (4, true), (8, false), (8, true), (10, true)] {
            let backend: Box<dyn Backend> = if jit {
                Box::new(x86jit_cranelift::JitBackend::new())
            } else {
                Box::new(InterpreterBackend)
            };
            let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
            vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(0x2000, 0x2000, Prot::RW, RegionKind::Ram).unwrap();
            let mut asm = CodeAssembler::new(64).unwrap();
            asm.fninit().unwrap();
            asm.fldcw(word_ptr(0x2000)).unwrap();
            match (width, pop) {
                (4, false) => asm.fst(dword_ptr(0x3000)),
                (4, true) => asm.fstp(dword_ptr(0x3000)),
                (8, false) => asm.fst(qword_ptr(0x3000)),
                (8, true) => asm.fstp(qword_ptr(0x3000)),
                (10, true) => asm.fstp(tbyte_ptr(0x3000)),
                _ => unreachable!(),
            }
            .unwrap();
            asm.fnstsw(ax).unwrap();
            asm.hlt().unwrap();
            vm.write_bytes(0x1000, &asm.assemble(0x1000).unwrap())
                .unwrap();
            vm.write_bytes(0x2000, &0x037eu16.to_le_bytes()).unwrap();
            vm.write_bytes(0x3000, &[0xa5; 10]).unwrap();
            vm.watch_range(0x3000, 10);
            assert!(vm.take_dirty_ranges().is_empty());
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.reg(Reg::Rax) as u16 & 0x82ff, 0x80c1);
            assert_eq!(cpu.fpu_top(), 0);
            let mut dest = [0; 10];
            vm.read_bytes(0x3000, &mut dest).unwrap();
            assert_eq!(dest, [0xa5; 10]);
            if !vm.take_dirty_ranges().is_empty() {
                unexpected_writes.push((jit, width, pop));
            }
            vm.write_bytes(0x2000, &0x037fu16.to_le_bytes()).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, 0x1000);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            assert_eq!(cpu.fpu_top(), u32::from(pop));
            assert_eq!(vm.take_dirty_ranges(), vec![(0x3000, 4096)]);
            vm.read_bytes(0x3000, &mut dest).unwrap();
            assert_ne!(dest, [0xa5; 10]);
        }
    }
    assert!(
        unexpected_writes.is_empty(),
        "unexpected writes: {unexpected_writes:?}"
    );
}

#[test]
fn suppressed_integer_stores_do_not_dirty_watched_memory() {
    let mut bad = 0;
    for jit in [false, true] {
        for width in [2, 4, 8] {
            for trunc in [false, true] {
                let backend: Box<dyn Backend> = if jit {
                    Box::new(x86jit_cranelift::JitBackend::new())
                } else {
                    Box::new(InterpreterBackend)
                };
                let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
                vm.map(0x1000, 0x1000, Prot::RWX, RegionKind::Ram).unwrap();
                vm.map(0x2000, 0x2000, Prot::RW, RegionKind::Ram).unwrap();
                let mut a = CodeAssembler::new(64).unwrap();
                a.fninit().unwrap();
                a.fldcw(word_ptr(0x2000)).unwrap();
                a.fld(tbyte_ptr(0x2010)).unwrap();
                match (width, trunc) {
                    (2, false) => a.fistp(word_ptr(0x3000)),
                    (4, false) => a.fistp(dword_ptr(0x3000)),
                    (8, false) => a.fistp(qword_ptr(0x3000)),
                    (2, true) => a.fisttp(word_ptr(0x3000)),
                    (4, true) => a.fisttp(dword_ptr(0x3000)),
                    (8, true) => a.fisttp(qword_ptr(0x3000)),
                    _ => unreachable!(),
                }
                .unwrap();
                a.fnstsw(ax).unwrap();
                a.hlt().unwrap();
                vm.write_bytes(0x1000, &a.assemble(0x1000).unwrap())
                    .unwrap();
                vm.write_bytes(0x2000, &0x37eu16.to_le_bytes()).unwrap();
                vm.write_bytes(
                    0x2010,
                    &x86jit_core::f80::F80::from_f64(1e30f64.to_bits()).to_bytes(),
                )
                .unwrap();
                vm.write_bytes(0x3000, &[0xa5; 8]).unwrap();
                vm.watch_range(0x3000, 8);
                assert!(vm.take_dirty_ranges().is_empty());
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, 0x1000);
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                assert_eq!(cpu.reg(Reg::Rax) as u16 & 0x82ff, 0x8081);
                assert_eq!(cpu.fpu_top(), 7);
                let mut dest = [0; 8];
                vm.read_bytes(0x3000, &mut dest).unwrap();
                assert_eq!(dest, [0xa5; 8]);
                let dirty = vm.take_dirty_ranges();
                eprintln!("jit={jit} width={width} trunc={trunc} unchanged=true dirty={dirty:?}");
                if !dirty.is_empty() {
                    bad += 1;
                }
                // Positive control: the same masked invalid operation really stores.
                vm.write_bytes(0x2000, &0x37fu16.to_le_bytes()).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, 0x1000);
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                let dirty = vm.take_dirty_ranges();
                assert_eq!(dirty, vec![(0x3000, 4096)]);
                vm.read_bytes(0x3000, &mut dest).unwrap();
                assert_ne!(dest, [0xa5; 8]);
            }
        }
    }
    assert_eq!(bad, 0);
}
