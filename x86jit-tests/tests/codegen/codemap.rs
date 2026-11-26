//! Compiled source locations preserve full guest addresses independently of RAM placement.

use std::collections::BTreeSet;

use iced_x86::code_asm::*;
use x86jit_core::ir::{IrBlock, IrOp, IrRegion, Val};
use x86jit_core::lift::{lift_block, CpuMode, FetchAddr};
use x86jit_core::{
    codemap, Backend, CachedBlock, HostRam, MemConsistency, Prot, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

fn compiled_rips(block: CachedBlock) -> BTreeSet<u64> {
    let CachedBlock::Compiled { entry } = block else {
        panic!("JIT backend must compile the unit");
    };
    let mut rips = BTreeSet::new();
    // The entry can precede the first source location (the function prologue).
    // Stop at the end of this function's registered range once coverage starts.
    for off in 0..4096 {
        match codemap::lookup(entry.0 as usize + off) {
            Some(rip) => {
                rips.insert(rip);
            }
            None if !rips.is_empty() => return rips,
            None => {}
        }
    }
    panic!("expected a complete codemap range within 4096 host bytes");
}

fn block_rips_round_trip(guest_base: u64, code_addr: u64) {
    let backing = vec![0u8; 0x4000].into_boxed_slice();
    let len = backing.len();
    let ptr = Box::into_raw(backing) as *mut u8;
    let ram = HostRam {
        ptr,
        len,
        guest_base,
        dtor: Box::new(|ptr, len| {
            // SAFETY: these are the pointer and length of the boxed slice above;
            // HostRam owns the allocation and calls this destructor exactly once.
            drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
        }),
        protect: None,
    };
    let mut vm = Vm::with_backend_host_ram(
        VmConfig::reserved(guest_base + len as u64),
        Box::new(JitBackend::new()),
        ram,
    )
    .unwrap();
    vm.map(code_addr, 0x1000, Prot::RX, RegionKind::Ram)
        .unwrap();
    let mut asm = CodeAssembler::new(64).unwrap();
    asm.mov(rax, qword_ptr(rbx)).unwrap();
    asm.mov(rcx, qword_ptr(rdx)).unwrap();
    asm.hlt().unwrap();
    vm.write_bytes(code_addr, &asm.assemble(code_addr).unwrap())
        .unwrap();
    let ir = lift_block(&vm.mem, FetchAddr::flat(code_addr), CpuMode::Long64).unwrap();
    let expected = ir
        .ops
        .iter()
        .filter_map(|op| match op {
            IrOp::InsnStart { guest_addr } => Some(*guest_addr),
            _ => None,
        })
        .collect();
    let compiled = vm.backend.materialize(
        &ir,
        vm.consistency,
        vm.mem.trap_window(),
        vm.mem.guest_base(),
    );
    assert_eq!(compiled_rips(compiled), expected);
}

#[test]
fn codemap_preserves_high_guest_base_rips() {
    block_rips_round_trip(0x1_0000_0000, 0x1_0000_2000);
}

#[test]
fn codemap_preserves_low_guest_rips() {
    block_rips_round_trip(0, 0x2000);
}

#[test]
fn codemap_preserves_rips_across_low_word_wrap() {
    block_rips_round_trip(0x1_ffff_e000, 0x1_ffff_ffff);
}

#[test]
fn codemap_preserves_noncontiguous_region_rips() {
    let first = 0x1_0000_2000;
    let second = 0x3_0000_2000;
    // Explicit IR connects blocks more than 4 GiB apart with identical low words.
    let block = |guest_start, target| IrBlock {
        legacy_context: None,
        mode: x86jit_core::CpuMode::Long64,
        guest_start,
        guest_len: 1,
        icount: 1,
        temp_count: 0,
        ops: vec![
            IrOp::InsnStart {
                guest_addr: guest_start,
            },
            IrOp::Jump {
                target: Val::Imm(target),
            },
        ],
        link_exit: false,
    };
    let region = IrRegion {
        entry: first,
        blocks: vec![block(first, second), block(second, first)],
        has_loop: true,
    };
    let jit = JitBackend::new();
    let compiled = jit.materialize_region(&region, MemConsistency::Fast, None, 0);
    assert_eq!(compiled_rips(compiled), BTreeSet::from([first, second]));
}
