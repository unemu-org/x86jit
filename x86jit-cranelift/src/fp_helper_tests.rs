//! FPCR witnesses for the explicitly integer-only helper table.
#![cfg(target_arch = "aarch64")]
use super::*;
use x86jit_core::jit_abi::MemCtx;
use x86jit_core::{CpuState, GuestCpuFeatures, Memory, MemoryModel, Prot, RegionKind};

struct Fixture {
    case: &'static str,
    cpu: CpuState,
    ctx: MemCtx,
    out: [u64; 8],
    packed: u128,
    observed: [u64; 2],
    variant: u64,
    timing: u64,
}

// Every integer classification needs a callable witness. This match is also
// checked when the production table is built by tests.
const INTEGER_NAMES: &[&str] = &[
    "aes_helper",
    "aes_mem_helper",
    "bmi_helper",
    "broadcast_lane_helper",
    "broadcast_lane_mem_helper",
    "cpuid_helper",
    "crc32_helper",
    "div_helper",
    "fp_boundary_helper",
    "cvtps2ph_helper",
    "cvtps2ph_value_helper",
    "fxstate_helper",
    "gather_helper",
    "gf2p8_helper",
    "gf2p8_mem_helper",
    "gfni_helper",
    "gfni_mem_helper",
    "mpsadbw_helper",
    "note_write_helper",
    "pclmul_helper",
    "pclmul_mem_helper",
    "pcmpstr_helper",
    "pcmpstr_mem_helper",
    "pcmpstrm_helper",
    "pcmpstrm_mem_helper",
    "phminposuw_helper",
    "pmaddwd_helper",
    "sha_helper",
    "sha_mem_helper",
    "shift_reg_helper",
    "string_helper",
    "valign_helper",
    "var_shift_helper",
    "var_shift_mem_helper",
    "vec_maskmov_mem_helper",
    "vhint_helper",
    "vhint_mem_helper",
    "vmasked_logic_helper",
    "vmasked_packed_helper",
    "vmasked_shift_helper",
    "vmaskmov_helper",
    "vmaskmov_mem_helper",
    "vp_blendm_helper",
    "vp_multishift_helper",
    "vp_unary_lane_helper",
    "vpabs_helper",
    "vpack_helper",
    "vpack_mem_helper",
    "vperm1_helper",
    "vperm1_mem_helper",
    "vpermt2_helper",
    "vpermt2_mem_helper",
    "vpmadd_helper",
    "vpmadd_mem_helper",
    "vpmov_extend_wide_helper",
    "vpmov_narrow_helper",
    "vpmov_narrow_mem_helper",
    "vpshufb_wide_helper",
    "vshuf_lane_helper",
    "vshuffle32_wide_helper",
    "vstore_preflight_helper",
    "xgetbv_helper",
];
pub(super) fn has_witness(name: &str) -> bool {
    INTEGER_NAMES.contains(&name.trim_end_matches("::<true>"))
}

fn read_fpcr() -> u64 {
    let value;
    // SAFETY: Arm ARM DDI0487J.a §C5.2.7: FPCR is readable at EL0.
    unsafe {
        core::arch::asm!("mrs {}, fpcr", out(reg) value, options(nomem, nostack));
    }
    value
}

#[unsafe(naked)]
unsafe extern "C" fn invoke(_fixture: *mut Fixture, _control: u64) {
    // SAFETY: all cases execute only integer operations with host traps masked;
    // restore the calling thread's exact FPCR before returning to test code.
    core::arch::naked_asm!(
        "stp x29, x30, [sp, #-32]!", "mov x29, sp",
        "mrs x9, fpcr", "str x9, [sp, #16]", "msr fpcr, x1",
        "bl {entry}", "ldr x9, [sp, #16]", "msr fpcr, x9",
        "ldp x29, x30, [sp], #32", "ret", entry = sym exercise,
    );
}

unsafe extern "C" fn exercise(f: *mut Fixture) {
    // SAFETY: invoke supplies the exclusive, live fixture and its memory context.
    let f = unsafe { &mut *f };
    let cpu = &mut f.cpu as *mut CpuState as *mut u8;
    let mem = &mut f.ctx as *mut MemCtx as *mut u8;
    f.observed[0] = read_fpcr();
    // SAFETY: fixtures provide live ABI objects, mapped aligned buffers, valid
    // vector indices/widths and an eight-word output slot for every case.
    unsafe {
        match f.case {
            "aes_helper" => {
                aes_helper(cpu, 0, 1, 2, 0, 3, f.timing);
            }
            "aes_mem_helper" => {
                aes_mem_helper(cpu, 0, 1, 0x807fff0123456789, 0x12345678, 0, 3, f.timing);
            }
            "bmi_helper" => {
                bmi_helper(1, 2, 0, 8, f.out.as_mut_ptr(), f.timing);
            }
            "broadcast_lane_helper" => {
                broadcast_lane_helper(cpu, 0, 1, 16, 4, 16, 1, 1, 1, f.timing);
            }
            "broadcast_lane_mem_helper" => {
                f.out[7] = broadcast_lane_mem_helper(
                    cpu, mem, 0, 0x8000, 16, 4, 16, 1, 1, 1, 0x1000, f.timing,
                );
            }
            "cpuid_helper" => {
                cpuid_helper(cpu, f.timing);
            }
            "crc32_helper" => {
                f.out[7] = crc32_helper(0, 1, 8, f.timing);
            }
            "cvtps2ph_helper" => {
                cvtps2ph_helper(cpu, 0, 1, 8, 4, f.timing);
            }
            "cvtps2ph_value_helper" => {
                cvtps2ph_value_helper(cpu, 1, 8, 4, &mut f.packed, f.timing);
            }
            "fp_boundary_helper" => {
                fp_boundary_helper(
                    0x3f800001,
                    0x3ff0000000000001,
                    0x40000000,
                    0x4000000000000000,
                    f.variant,
                    f.variant & 1,
                    u64::from(f.cpu.mxcsr),
                    &mut f.packed,
                    f.timing,
                );
            }
            "div_helper" => {
                f.out[7] = div_helper(0, 0x807fff0123456789, 7, 8, 1, f.out.as_mut_ptr(), f.timing);
            }
            "fxstate_helper" => {
                f.out[7] = fxstate_helper(cpu, mem, 0x8000, f.variant & 1, 0x1000, f.timing);
            }
            "gather_helper" => {
                f.out[7] = gather_helper(
                    cpu, mem, 0, 2, 3, 0x8000, 0, 4, 4, 16, 1, 0, 0x1000, f.timing,
                );
            }
            "gf2p8_helper" => {
                gf2p8_helper(cpu, 0, 1, 2, 3, 0, 1, 1, 16, f.timing);
            }
            "gf2p8_mem_helper" => {
                f.out[7] =
                    gf2p8_mem_helper(cpu, mem, 0, 1, 0x8000, 3, 0, 1, 1, 16, 0x1000, f.timing);
            }
            "gfni_helper" => {
                gfni_helper(cpu, 0, 1, 2, 0, 3, f.timing);
            }
            "gfni_mem_helper" => {
                gfni_mem_helper(cpu, 0, 1, 0x807fff0123456789, 0x12345678, 0, 3, f.timing);
            }
            "mpsadbw_helper" => {
                mpsadbw_helper(cpu, 0, 1, 0x807fff0123456789, 0x12345678, 3, 0, f.timing);
            }
            "note_write_helper" => {
                note_write_helper(f.ctx.mem_self, 0x8000, 8, f.timing);
            }
            "pclmul_helper" => {
                pclmul_helper(cpu, 0, 1, 2, 3, f.timing);
            }
            "pclmul_mem_helper" => {
                pclmul_mem_helper(cpu, 0, 1, 0x807fff0123456789, 0x12345678, 3, f.timing);
            }
            "pcmpstr_helper" => {
                pcmpstr_helper(cpu.cast_const(), 1, 2, 3, 0, f.out.as_mut_ptr(), f.timing);
            }
            "pcmpstr_mem_helper" => {
                pcmpstr_mem_helper(
                    cpu.cast_const(),
                    1,
                    0x807fff0123456789,
                    0x12345678,
                    3,
                    0,
                    f.out.as_mut_ptr(),
                    f.timing,
                );
            }
            "pcmpstrm_helper" => {
                pcmpstrm_helper(cpu, 1, 2, 3, 0, f.out.as_mut_ptr(), f.timing);
            }
            "pcmpstrm_mem_helper" => {
                pcmpstrm_mem_helper(
                    cpu,
                    1,
                    0x807fff0123456789,
                    0x12345678,
                    3,
                    0,
                    f.out.as_mut_ptr(),
                    f.timing,
                );
            }
            "phminposuw_helper" => {
                phminposuw_helper(cpu, 0, 1, f.timing);
            }
            "pmaddwd_helper" => {
                pmaddwd_helper(cpu, 0, 1, 2, f.timing);
            }
            "sha_helper" => {
                sha_helper(cpu, 0, 1, 2, 0, 3, f.timing);
            }
            "sha_mem_helper" => {
                sha_mem_helper(cpu, 0, 1, 0x807fff0123456789, 0x12345678, 0, 3, f.timing);
            }
            "shift_reg_helper" => {
                shift_reg_helper(cpu, 0, 1, 2, 4, 0, 0, 1, 1, 16, f.timing);
            }
            "string_helper" => {
                f.out[7] = string_helper::<true>(cpu, mem, 0, 1, 1, 0x1000, 64, 0, f.timing);
            }
            "valign_helper" => {
                valign_helper(cpu, 0, 1, 2, 1, 4, 16, f.timing);
            }
            "var_shift_helper" => {
                var_shift_helper(cpu, 0, 1, 2, 4, 0, 0, 1, 1, 16, f.timing);
            }
            "var_shift_mem_helper" => {
                var_shift_mem_helper(
                    cpu,
                    0,
                    1,
                    0x807fff0123456789,
                    0x12345678,
                    0,
                    4,
                    0,
                    0,
                    f.timing,
                );
            }
            "vec_maskmov_mem_helper" => {
                f.out[7] = vec_maskmov_mem_helper(
                    cpu,
                    mem,
                    0,
                    0x8000,
                    1,
                    4,
                    16,
                    f.variant & 1,
                    0x1000,
                    f.timing,
                );
            }
            "vhint_helper" => {
                vhint_helper(cpu, 0, 1, 2, 0, 16, f.timing);
            }
            "vhint_mem_helper" => {
                vhint_mem_helper(
                    cpu,
                    0,
                    1,
                    0x807fff0123456789,
                    0x12345678,
                    0x807fff0123456789,
                    0x12345678,
                    0,
                    16,
                    f.timing,
                );
            }
            "vmasked_logic_helper" => {
                vmasked_logic_helper(cpu, 0, 0, 1, 2, 1, 4, 1, 16, f.timing);
            }
            "vmasked_packed_helper" => {
                vmasked_packed_helper(cpu, 0, 0, 1, 2, 1, 4, 1, 16, f.timing);
            }
            "vmasked_shift_helper" => {
                vmasked_shift_helper(cpu, 0, 1, 3, 4, 0, 0, 1, 1, 16, f.timing);
            }
            "vmaskmov_helper" => {
                vmaskmov_helper(cpu, 0, 1, 1, 4, 1, 16, f.timing);
            }
            "vmaskmov_mem_helper" => {
                f.out[7] = vmaskmov_mem_helper(
                    cpu,
                    mem,
                    0,
                    0x8000,
                    1,
                    4,
                    1,
                    16,
                    f.variant & 1,
                    0x1000,
                    0,
                    f.timing,
                );
            }
            "vp_blendm_helper" => {
                vp_blendm_helper(cpu, 0, 1, 2, 1, 4, 16, 1, f.timing);
            }
            "vp_multishift_helper" => {
                vp_multishift_helper(cpu, 0, 1, 2, 16, 1, 1, 1, f.timing);
            }
            "vp_unary_lane_helper" => {
                vp_unary_lane_helper(cpu, 0, 1, 0, 3, 4, 16, 1, 1, 1, f.timing);
            }
            "vpabs_helper" => {
                vpabs_helper(cpu, 0, 1, 4, 16, 1, 1, 1, f.timing);
            }
            "vpack_helper" => {
                vpack_helper(cpu, 0, 1, 2, 2, 1, 16, f.timing);
            }
            "vpack_mem_helper" => {
                vpack_mem_helper(cpu, 0, 1, 0x807fff0123456789, 0x12345678, 2, 1, 0, f.timing);
            }
            "vperm1_helper" => {
                vperm1_helper(cpu, 0, 3, 1, 4, 16, 1, 1, 1, f.timing);
            }
            "vperm1_mem_helper" => {
                f.out[7] =
                    vperm1_mem_helper(cpu, mem, 0, 3, 0x8000, 4, 1, 1, 1, 16, 0x1000, f.timing);
            }
            "vpermt2_helper" => {
                vpermt2_helper(cpu, 0, 3, 2, 4, 1, 1, 1, 16, 0, f.timing);
            }
            "vpermt2_mem_helper" => {
                f.out[7] =
                    vpermt2_mem_helper(cpu, mem, 0, 3, 0x8000, 4, 1, 1, 1, 16, 0, 0x1000, f.timing);
            }
            "vpmadd_helper" => {
                vpmadd_helper(cpu, 0, 1, 2, 0, 16, f.timing);
            }
            "vpmadd_mem_helper" => {
                vpmadd_mem_helper(cpu, 0, 1, 0x807fff0123456789, 0x12345678, 0, 0, f.timing);
            }
            "vpmov_extend_wide_helper" => {
                vpmov_extend_wide_helper(cpu, 0, 1, 1, 4, 1, 16, 1, 1, 1, f.timing);
            }
            "vpmov_narrow_helper" => {
                vpmov_narrow_helper(cpu, 0, 1, 4, 2, 16, 1, 1, 1, f.timing);
            }
            "vpmov_narrow_mem_helper" => {
                f.out[7] = vpmov_narrow_mem_helper(cpu, mem, 1, 0x8000, 4, 2, 16, 0x1000, f.timing);
            }
            "vpshufb_wide_helper" => {
                vpshufb_wide_helper(cpu, 0, 1, 3, 16, 1, 1, 1, f.timing);
            }
            "vshuf_lane_helper" => {
                vshuf_lane_helper(cpu, 0, 1, 2, 3, 4, 16, 1, 1, 1, f.timing);
            }
            "vshuffle32_wide_helper" => {
                vshuffle32_wide_helper(cpu, 0, 1, 3, 16, 1, 1, 1, f.timing);
            }
            "vstore_preflight_helper" => {
                f.out[7] = vstore_preflight_helper(mem, 16, 0x8000, f.timing);
            }
            "xgetbv_helper" => {
                f.out[7] = xgetbv_helper(cpu, mem, 0x1000, f.timing);
            }
            _ => unreachable!(),
        }
    }
    f.observed[1] = read_fpcr();
}

fn fixture(name: &'static str, memory: &Memory, variant: u64, timing: u64) -> Fixture {
    let mut cpu = CpuState::new();
    cpu.features = GuestCpuFeatures::v4();
    cpu.mxcsr = if variant == 0 { 0x1f80 } else { 0xdfc0 };
    cpu.gpr[0] = 7;
    cpu.gpr[1] = if name == "xgetbv_helper" { 0 } else { 16 };
    cpu.gpr[6] = 0x8000;
    cpu.gpr[7] = 0x9000;
    for i in 0..32 {
        cpu.xmm[i] = 0xff807fff_00800001_ffff0000_01234567u128.rotate_left(i as u32);
        cpu.ymm_hi[i] = !cpu.xmm[i];
        cpu.zmm_hi[i] = [cpu.xmm[i], !cpu.xmm[i]];
    }
    if name.starts_with("cvtps2ph") {
        cpu.xmm[1] = 0x7f800001_b3800000_3f801000_33800000;
        cpu.ymm_hi[1] = 0xff800000_3f800fff_00000001_7f800000;
    }
    cpu.xmm[3] = 0; // gather indices stay within the mapped buffer
    cpu.kmask[1] = 0xa5a5;
    Fixture {
        case: name,
        cpu,
        ctx: MemCtx::for_memory(memory),
        out: [0; 8],
        packed: 0,
        observed: [0; 2],
        variant,
        timing,
    }
}

#[test]
fn every_helper_has_an_explicit_classification_and_integer_witness() {
    // Building one table checks every registration, including unused helpers.
    use x86jit_core::{Backend, CachedBlock, CpuMode, IrBlock, IrOp};
    let block = IrBlock {
        mode: CpuMode::Long64,
        guest_start: 0x1000,
        guest_len: 1,
        icount: 1,
        temp_count: 0,
        ops: vec![IrOp::Hlt],
        legacy_context: None,
    };
    assert!(matches!(
        JitBackend::new().materialize(&block, MemConsistency::Fast, None, 0),
        CachedBlock::Compiled { .. }
    ));
}

#[test]
fn integer_helpers_are_independent_of_fpcr() {
    let mut controls = vec![0];
    // Arm ARM DDI0487J.a §C5.2.7: RMode[23:22], FZ[24], AH[1], FIZ[0].
    // AH/FIZ are RES0 without FEAT_AFP; do not claim a hardware witness for them.
    let afp = x86jit_core::jit_abi::fp::has_afp();
    for rc in 0..4 {
        for bits in 0..if afp { 8 } else { 2 } {
            controls.push((rc << 22) | ((bits & 1) << 24) | ((bits >> 1) & 3));
        }
    }
    eprintln!(
        "integer helper FPCR witness: AFP={afp}, controls={}",
        controls.len()
    );
    for &name in INTEGER_NAMES {
        eprintln!("FPCR integer witness: {name}");
        assert!(has_witness(name), "missing witness: {name}");
        for variant in 0..if name == "fp_boundary_helper" { 7 } else { 2 } {
            for timing in 0..2 {
                let mut expected = None;
                for &control in &controls {
                    let mut memory = Memory::new(MemoryModel::Flat { size: 0x10000 });
                    memory.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
                    for (i, b) in (0..128).map(|i| (i, (i * 17) as u8)) {
                        memory.write(0x8000 + i, u64::from(b), 1).unwrap();
                    }
                    memory.write(0x8018, 0x1f80, 4).unwrap();
                    memory.mark_code(0x8000, 0x1000);
                    memory.mark_code(0x9000, 32);
                    let mut f = fixture(name, &memory, variant, timing);
                    // SAFETY: the fixture owns all ABI objects for the synchronous call.
                    unsafe {
                        invoke(&mut f, control);
                    }
                    assert_eq!(f.observed, [control; 2], "{name}: helper changed FPCR");
                    // Preflight returns a host address, which is allocation-specific.
                    if name == "vstore_preflight_helper" {
                        f.out[7] = u64::from(f.out[7] != 0);
                    }
                    let state = format!("{:?}", f.cpu);
                    let mut bytes = vec![0; 0x1000];
                    memory.read_bytes(0x8000, &mut bytes).unwrap();
                    let mut dest = [0; 32];
                    memory.read_bytes(0x9000, &mut dest).unwrap();
                    bytes.extend_from_slice(&dest);
                    let result = (
                        state,
                        bytes,
                        f.out,
                        f.packed,
                        f.ctx.fault_addr,
                        f.ctx.exception_vector,
                    );
                    if let Some(expected) = &expected {
                        assert_eq!(&result, expected, "{name}, FPCR={control:x}");
                    } else {
                        expected = Some(result);
                    }
                }
            }
        }
    }
}

unsafe extern "C" fn observing_fp(
    cpu: *mut u8,
    dst: u64,
    a: u64,
    b: u64,
    op: u64,
    double: u64,
    bytes: u64,
    timing: u64,
) {
    // SAFETY: generated code supplies the exclusive CPU; the witness is called
    // with default FPCR even when the guest requests upward rounding and FZ.
    unsafe {
        (*cpu.cast::<CpuState>()).gpr[14] = read_fpcr();
    }
    let rounded: u32;
    // Arm ARM DDI0487J.a §C5.2.7: default RMode rounds a half-ULP tie to even.
    // SAFETY: no traps are enabled; explicit operands prevent constant folding.
    unsafe {
        core::arch::asm!(
            "fmov s0, {one:w}", "fmov s1, {half:w}", "fadd s0, s0, s1",
            "fmov {result:w}, s0",
            one = in(reg) 0x3f800000u32, half = in(reg) 0x33800000u32,
            result = out(reg) rounded, out("v0") _, out("v1") _,
            options(nomem, nostack),
        );
        vhfloat_helper(cpu, dst, a, b, op, double, bytes, timing);
    }
    unsafe {
        (*cpu.cast::<CpuState>()).gpr[15] = u64::from(rounded);
    }
}

#[test]
fn fp_and_unclassified_helpers_enter_default_environment() {
    use x86jit_core::{
        lift::{lift_block, FetchAddr},
        CpuMode,
    };
    for class in [
        codegen::HelperFp::NeedsDefault,
        codegen::HelperFp::Unclassified,
    ] {
        for rc in 0..4 {
            let mut memory = Memory::new(MemoryModel::Flat { size: 0x4000 });
            memory.map(0, 0x4000, Prot::RWX, RegionKind::Ram).unwrap();
            // HADDPS xmm0,xmm1; HLT.
            for (i, byte) in [0xf2, 0x0f, 0x7c, 0xc1, 0xf4].into_iter().enumerate() {
                memory.write(0x1000 + i as u64, byte, 1).unwrap();
            }
            let block = lift_block(&memory, FetchAddr::flat(0x1000), CpuMode::Long64).unwrap();
            let mode = 0x9fc0 | (rc << 13);
            let backend = JitBackend::new();
            let entry = backend
                .shared
                .compile_with(
                    CompiledCodeKind::Block,
                    0x1000,
                    mode & x86jit_core::jit_abi::fp::MODE_MASK,
                    |builder, mut helpers, alloc| {
                        helpers.vhfloat.1 = observing_fp as *const () as u64;
                        helpers.vhfloat.3 = class;
                        codegen::translate_block(
                            builder,
                            &block,
                            &backend.shared.offsets,
                            alloc,
                            helpers,
                            MemConsistency::Fast,
                            None,
                            0,
                            false,
                        )
                    },
                )
                .unwrap();
            let mut cpu = CpuState::new();
            cpu.mxcsr = mode;
            cpu.xmm[1] = 0x3f800000_3f800000_3f800000_3f800000;
            let saved = read_fpcr();
            // SAFETY: the backend retains the live entry and memory for this call.
            unsafe {
                x86jit_core::jit_abi::call_block(entry, &mut cpu, &mut MemCtx::for_memory(&memory));
            }
            assert_eq!(cpu.gpr[14], 0, "{class:?}, RC={rc}");
            assert_eq!(cpu.gpr[15], 0x3f800000);
            assert_eq!(read_fpcr(), saved);
        }
    }
}
