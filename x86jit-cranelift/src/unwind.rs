//! One-frame recovery from Cranelift's prologue unwind description (§8.2).

use cranelift::codegen::isa::unwind::UnwindInst;
use cranelift::codegen::CompiledCode;
use x86jit_core::jit_abi::fault::FrameRecipe;

pub(crate) fn uses_bkey(isa: &dyn cranelift::codegen::isa::TargetIsa) -> bool {
    isa.isa_flags()
        .iter()
        .any(|v| v.name == "sign_return_address_with_bkey" && v.as_bool() == Some(true))
}

pub(crate) fn recipe(
    code: &CompiledCode,
    bkey: bool,
    call_conv: cranelift::codegen::isa::CallConv,
    default_call_conv: cranelift::codegen::isa::CallConv,
) -> Option<FrameRecipe> {
    if call_conv != default_call_conv {
        return None;
    }
    extract(&code.buffer.unwind_info, bkey)
}

fn extract(insts: &[(u32, UnwindInst)], bkey: bool) -> Option<FrameRecipe> {
    if !cfg!(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "freebsd"
        )
    )) {
        return None;
    }
    let mut pushed = None;
    let mut frame = None;
    let mut clobbers = 0;
    let mut last_pc = 0;
    let mut return_auth = 0;
    for (pc, inst) in insts {
        if *pc < last_pc {
            return None;
        }
        last_pc = *pc;
        match *inst {
            UnwindInst::PushFrameRegs {
                offset_upward_to_caller_sp,
            } => {
                if pushed.is_some() || frame.is_some() {
                    return None;
                }
                pushed = Some(i32::try_from(offset_upward_to_caller_sp).ok()?);
            }
            UnwindInst::DefineNewFrame {
                offset_upward_to_caller_sp,
                offset_downward_to_clobbers,
            } => {
                if frame.is_some() {
                    return None;
                }
                let caller_sp = i32::try_from(offset_upward_to_caller_sp).ok()?;
                let saved_fp = caller_sp.checked_sub(pushed?)?;
                clobbers = i32::try_from(offset_downward_to_clobbers).ok()?;
                // Interpret PushFrameRegs exactly as Cranelift 0.115.1's
                // isa/unwind/systemv.rs create_unwind_info_from_insts: saved FP
                // is CFA - push_size. AArch64's RegisterMapper::lr_offset() is
                // 8; x64's CIE places its hardware-pushed RA at CFA - 8.
                let saved_return = if cfg!(target_arch = "aarch64") {
                    saved_fp.checked_add(8)?
                } else {
                    caller_sp.checked_sub(8)?
                };
                frame = Some(FrameRecipe {
                    prologue_end: *pc,
                    return_auth,
                    caller_sp,
                    saved_fp,
                    saved_return,
                    gpr: [None; 10],
                    simd: [None; 8],
                });
            }
            UnwindInst::SaveReg {
                clobber_offset,
                reg,
            } => {
                let frame = frame.as_mut()?;
                let off = i32::try_from(clobber_offset).ok()?.checked_sub(clobbers)?;
                if off & 7 != 0 || off >= frame.saved_fp {
                    return None;
                }
                // regalloc2::RegClass discriminants, exposed by RealReg::class:
                // Int = 0, Float = 1, Vector = 2. hw_enc is the ISA encoding.
                let slot = match (
                    cfg!(target_arch = "aarch64"),
                    reg.class() as u8,
                    reg.hw_enc(),
                ) {
                    (true, 0, n @ 19..=28) => &mut frame.gpr[(n - 19) as usize],
                    (true, 1, n @ 8..=15) => &mut frame.simd[(n - 8) as usize],
                    (false, 0, n) => {
                        let index = [3, 12, 13, 14, 15].iter().position(|r| *r == n)?;
                        &mut frame.gpr[index]
                    }
                    _ => return None,
                };
                if slot.replace(off).is_some() {
                    return None;
                }
            }
            UnwindInst::StackAlloc { .. } if frame.is_some() => {}
            UnwindInst::Aarch64SetPointerAuth { return_addresses } if frame.is_none() => {
                return_auth = if return_addresses {
                    if bkey {
                        2
                    } else {
                        1
                    }
                } else {
                    0
                };
            }
            // In particular, alternate SP rules need
            // explicit support rather than interpreting them as ordinary frames.
            _ => return None,
        }
    }
    let mut frame = frame?;
    // The default ABIs use this frame record. Other shapes (including tail-call
    // stack resizing) need a distinct return/authentication contract.
    if frame.saved_fp != 0 || frame.caller_sp != 16 {
        return None;
    }
    frame.prologue_end = last_pc;
    Some(frame)
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[path = "unwind_test_signal.rs"]
mod signal;

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[path = "unwind_test_call.rs"]
mod register_probe;

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "macos", target_os = "freebsd"),
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod tests {
    use super::*;
    use cranelift::prelude::*;
    use cranelift_module::{Linkage, Module};

    struct Probe {
        recipe: FrameRecipe,
        out: *const u64,
        corrupt: u8,
    }

    unsafe extern "C" fn inspect(fp: u64, probe: *const Probe) -> u64 {
        // SAFETY: called by the generated function with its real frame pointer;
        // Probe and the assembly caller's output remain live through the call.
        unsafe {
            let probe = &*probe;
            let r = probe.recipe;
            let patterns = register_probe::patterns();
            let saved =
                |off: i32| (fp.checked_add_signed(i64::from(off)).unwrap() as *const u64).read();
            let fp_index = if cfg!(target_arch = "aarch64") { 10 } else { 5 };
            let mut valid = saved(r.saved_fp) == patterns[fp_index]
                && fp + r.caller_sp as u64 == probe.out.add(register_probe::REGISTER_COUNT).read();
            for (i, off) in r.gpr.into_iter().enumerate() {
                if let Some(off) = off {
                    valid &= saved(off + if probe.corrupt == 1 { 8 } else { 0 }) == patterns[i];
                }
            }
            for (i, off) in r.simd.into_iter().enumerate() {
                if let Some(off) = off {
                    valid &=
                        saved(off + if probe.corrupt == 2 { 8 } else { 0 }) == patterns[11 + i];
                }
            }
            #[allow(unused_mut)]
            let mut ra = saved(r.saved_return);
            #[cfg(target_arch = "aarch64")]
            {
                let sp = probe.out.add(register_probe::REGISTER_COUNT).read();
                if r.return_auth == 2 {
                    core::arch::asm!(".arch_extension pauth", "autib {ra}, {sp}",
                        ra = inout(reg) ra, sp = in(reg) sp, options(nomem, nostack));
                } else if r.return_auth == 1 {
                    core::arch::asm!(".arch_extension pauth", "autia {ra}, {sp}",
                        ra = inout(reg) ra, sp = in(reg) sp, options(nomem, nostack));
                }
            }
            valid &= ra == probe.out.add(register_probe::REGISTER_COUNT + 1).read();
            u64::from(valid)
        }
    }

    #[test]
    fn real_prologue_slots_match_unwind_recipe() {
        let mut jit =
            crate::JitBackend::build_jit(crate::HostTarget::Native, crate::OptLevel::Speed);
        let mut ctx = jit.module.make_context();
        ctx.func
            .signature
            .params
            .extend([AbiParam::new(types::I64); 2]);
        ctx.func.signature.returns.push(AbiParam::new(types::I64));
        let mut fbctx = FunctionBuilderContext::new();
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
            let block = b.create_block();
            b.append_block_params_for_function_params(block);
            b.switch_to_block(block);
            b.seal_block(block);
            let data = b.block_params(block)[0];
            let probe = b.block_params(block)[1];
            let mut integers = Vec::new();
            let mut floats = Vec::new();
            for i in 0..32 {
                integers.push(b.ins().load(types::I64, MemFlags::trusted(), data, i * 8));
                floats.push(
                    b.ins()
                        .load(types::F64, MemFlags::trusted(), data, 256 + i * 8),
                );
            }
            let mut sig = jit.module.make_signature();
            sig.params.extend([AbiParam::new(types::I64); 2]);
            sig.returns.push(AbiParam::new(types::I64));
            let sig = b.import_signature(sig);
            let target = b.ins().iconst(types::I64, inspect as *const () as i64);
            let fp = b.ins().get_frame_pointer(types::I64);
            // A balanced tree keeps enough independent FP inputs live to use
            // the allocator's non-preferred preserved vector registers too.
            let mut layer = floats.clone();
            while layer.len() > 1 {
                layer = layer
                    .chunks_exact(2)
                    .map(|pair| b.ins().fadd(pair[0], pair[1]))
                    .collect();
            }
            b.ins().store(MemFlags::trusted(), layer[0], data, 0);
            let call = b.ins().call_indirect(sig, target, &[fp, probe]);
            let result = b.inst_results(call)[0];
            // Consume all values after the call to require real callee saves.
            for (i, value) in integers.into_iter().enumerate() {
                let value = b.ins().iadd_imm(value, 1);
                b.ins()
                    .store(MemFlags::trusted(), value, data, i as i32 * 8);
            }
            for (i, value) in floats.into_iter().enumerate() {
                let value = b.ins().fadd(value, value);
                b.ins()
                    .store(MemFlags::trusted(), value, data, 256 + i as i32 * 8);
            }
            b.ins().return_(&[result]);
            b.finalize();
        }
        let id = jit
            .module
            .declare_function("frame_probe", Linkage::Local, &ctx.func.signature)
            .unwrap();
        jit.module.define_function(id, &mut ctx).unwrap();
        let code = ctx.compiled_code().unwrap();
        let r = recipe(
            code,
            jit.return_address_bkey,
            ctx.func.signature.call_conv,
            jit.module.isa().default_call_conv(),
        )
        .expect("real function must have a supported recipe");
        assert!(
            r.gpr.iter().flatten().count() >= if cfg!(target_arch = "aarch64") { 8 } else { 4 }
        );
        if cfg!(target_arch = "aarch64") {
            assert_eq!(r.simd.iter().flatten().count(), 8);
        }
        #[cfg(target_arch = "aarch64")]
        {
            // Arm ARM DDI0487J.a §C6.2, PACIB/PACIA encodings: SP modifier,
            // rather than the zero-modifier PACIBZ form (0xD503235F).
            let first = u32::from_le_bytes(code.code_buffer()[..4].try_into().unwrap());
            let signing = jit
                .module
                .isa()
                .isa_flags()
                .iter()
                .any(|v| v.name == "sign_return_address" && v.as_bool() == Some(true));
            if cfg!(target_os = "macos") {
                assert_eq!(r.return_auth, 2);
                assert_eq!(first, 0xD503237F);
                assert_ne!(first, 0xD503235F);
            } else if signing {
                assert_eq!(r.return_auth, if jit.return_address_bkey { 2 } else { 1 });
                assert_eq!(
                    first,
                    if jit.return_address_bkey {
                        0xD503237F
                    } else {
                        0xD503233F
                    }
                );
            } else {
                assert_eq!(r.return_auth, 0);
                assert!(![0xD503237F, 0xD503235F, 0xD503233F, 0xD503231F].contains(&first));
            }
        }
        assert!(recipe(
            code,
            jit.return_address_bkey,
            cranelift::codegen::isa::CallConv::Tail,
            jit.module.isa().default_call_conv()
        )
        .is_none());
        for (push, caller) in [(16, 24), (24, 24)] {
            assert!(extract(
                &[
                    (
                        0,
                        UnwindInst::PushFrameRegs {
                            offset_upward_to_caller_sp: push
                        }
                    ),
                    (
                        4,
                        UnwindInst::DefineNewFrame {
                            offset_upward_to_caller_sp: caller,
                            offset_downward_to_clobbers: 0
                        }
                    )
                ],
                false
            )
            .is_none());
        }
        assert!(extract(&[], false).is_none());
        let mut incomplete = code.buffer.unwind_info.to_vec();
        incomplete.retain(|(_, inst)| !matches!(inst, UnwindInst::DefineNewFrame { .. }));
        assert!(extract(&incomplete, false).is_none());
        jit.module.finalize_definitions().unwrap();
        let entry = jit.module.get_finalized_function(id);
        let mut data = [0u64; 64];
        let mut out = [0; register_probe::REGISTER_COUNT + 2];
        let patterns = register_probe::patterns();
        let mut probe = Probe {
            recipe: r,
            out: out.as_ptr(),
            corrupt: 0,
        };
        for corrupt in 0..if cfg!(target_arch = "aarch64") { 3 } else { 2 } {
            probe.corrupt = corrupt;
            // SAFETY: executable code, data, output and probe are live and obey
            // the test caller's ABI. The deliberately wrong recipe only shifts
            // GPR reads into other readable slots of this frame.
            let result = unsafe {
                register_probe::probe_call(
                    data.as_mut_ptr().cast(),
                    (&mut probe as *mut Probe).cast(),
                    entry,
                    out.as_mut_ptr(),
                    patterns.as_ptr(),
                )
            };
            assert_eq!(result, u64::from(corrupt == 0));
            assert_eq!(out[..register_probe::REGISTER_COUNT], patterns);
        }
    }
    use super::signal;

    #[test]
    fn compiled_region_recovers_lr_and_simd_after_a_helper() {
        use cranelift::codegen::cursor::{Cursor, FuncCursor};
        use std::sync::atomic::Ordering;
        use x86jit_core::jit_abi::fault::EntryRecord;
        use x86jit_core::jit_abi::{MemCtx, RET_HOST_FAULT};
        use x86jit_core::{
            CpuMode, CpuState, MemConsistency, MemoryModel, Prot, RegionCaps, RegionKind,
        };
        assert_eq!(
            signal::child(|| {
                // SAFETY: child owns this page-aligned mapping until Memory drops it.
                let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
                let len = page * 8;
                let ptr = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        len,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANON,
                        -1,
                        0,
                    )
                };
                assert_ne!(ptr, libc::MAP_FAILED);
                let mut mem = x86jit_core::Memory::from_host_ram(
                    MemoryModel::Flat { size: len as u64 },
                    x86jit_core::memory::HostRam {
                        ptr: ptr.cast(),
                        len,
                        guest_base: 0,
                        protect: None,
                        dtor: Box::new(|ptr, len| unsafe {
                            libc::munmap(ptr.cast(), len);
                        }),
                    },
                )
                .unwrap();
                mem.map(0, page * 4, Prot::RWX, RegionKind::Ram).unwrap();
                // mov ebp,2; top: cpuid; mov rax,[rdi]; dec ebp; jnz top; hlt.
                // The guest access follows a real helper call in a lifted loop region.
                mem.write_bytes(
                    0x1000,
                    &[
                        0xbd, 2, 0, 0, 0, 0x0f, 0xa2, 0x48, 0x8b, 0x07, 0xff, 0xcd, 0x75, 0xf7,
                        0xf4,
                    ],
                )
                .unwrap();
                let region = x86jit_core::lift::lift_region(
                    &mem,
                    0x1000,
                    RegionCaps {
                        max_blocks: 16,
                        max_icount: 256,
                    },
                    CpuMode::Long64,
                )
                .unwrap();
                assert!(region.has_loop && region.blocks.len() > 1);
                let jit = crate::JitBackend::new();
                let mut pressure = [1.0f64; 32];
                let pressure_ptr = pressure.as_mut_ptr() as u64;
                let entry = jit
                    .shared
                    .compile_with(
                        crate::perfmap::Kind::Region,
                        region.entry,
                        |b, helpers, slots| {
                            let rips = crate::codegen::translate_region(
                                b,
                                &region,
                                &jit.shared.offsets,
                                slots,
                                helpers,
                                MemConsistency::Fast,
                                None,
                                0,
                                false,
                            );
                            // Test-only pressure around the real region: values loaded in
                            // a dominating prelude remain live to every ordinary return.
                            // This forces real vector saves without changing production codegen.
                            let f = &mut *b.func;
                            let old_entry = f.layout.entry_block().unwrap();
                            let prelude = f.dfg.make_block();
                            let cpu = f.dfg.append_block_param(prelude, types::I64);
                            let ctx = f.dfg.append_block_param(prelude, types::I64);
                            let mut c = FuncCursor::new(f);
                            c.goto_top(old_entry);
                            c.insert_block(prelude);
                            let data = c.ins().iconst(types::I64, pressure_ptr as i64);
                            let values: Vec<_> = (0..32)
                                .map(|i| c.ins().load(types::F64, MemFlags::trusted(), data, i * 8))
                                .collect();
                            let mut tree = values.clone();
                            while tree.len() > 1 {
                                tree = tree
                                    .chunks_exact(2)
                                    .map(|p| c.ins().fadd(p[0], p[1]))
                                    .collect();
                            }
                            c.ins().store(MemFlags::trusted(), tree[0], data, 0);
                            c.ins().jump(old_entry, &[cpu, ctx]);
                            let returns: Vec<_> = c
                                .func
                                .layout
                                .blocks()
                                .flat_map(|block| c.func.layout.block_insts(block))
                                .filter(|inst| {
                                    c.func.dfg.insts[*inst].opcode()
                                        == cranelift::codegen::ir::Opcode::Return
                                })
                                .collect();
                            for ret in returns {
                                c.goto_inst(ret);
                                let mut tree = values.clone();
                                while tree.len() > 1 {
                                    tree = tree
                                        .chunks_exact(2)
                                        .map(|p| c.ins().fadd(p[0], p[1]))
                                        .collect();
                                }
                                c.ins().store(MemFlags::trusted(), tree[0], data, 0);
                            }
                            rips
                        },
                    )
                    .expect("lifted region with pressure must compile");
                let mut cpu = CpuState::new();
                cpu.gpr[7] = (page * 4) as u64;
                cpu.rip = region.entry;
                let mut ctx = MemCtx::for_memory(&mem);
                let mut record = EntryRecord::default();
                let patterns = register_probe::patterns();
                let mut out = [0; register_probe::REGISTER_COUNT + 2];
                // SAFETY: guard page and handler are confined to the child; all ABI
                // pointers, record and pressure data remain live until the return.
                unsafe {
                    assert_eq!(
                        libc::mprotect(
                            ptr.cast::<u8>().add(page * 4).cast(),
                            page,
                            libc::PROT_NONE
                        ),
                        0
                    );
                    record.activate(&mut cpu, &mut ctx);
                    signal::HANDLE.store(record.handle().as_usize(), Ordering::Relaxed);
                    signal::install();
                    let result = register_probe::probe_call(
                        (&mut cpu as *mut CpuState).cast(),
                        (&mut ctx as *mut MemCtx).cast(),
                        entry.0,
                        out.as_mut_ptr(),
                        patterns.as_ptr(),
                    );
                    record.deactivate();
                    assert_eq!(result, RET_HOST_FAULT);
                }
                assert_eq!(out[..register_probe::REGISTER_COUNT], patterns);
                assert_eq!(signal::FAULTS.load(Ordering::Relaxed), 1);
                #[cfg(target_arch = "aarch64")]
                {
                    assert_eq!(signal::SIMD_SLOTS.load(Ordering::Relaxed), 8);
                    assert!(signal::LR_CHANGED.load(Ordering::Relaxed));
                }
                // Keep this test's pressure buffer live through execution.
                std::hint::black_box(pressure);
            }),
            0
        );
    }
}
