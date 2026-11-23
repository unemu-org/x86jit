//! Model decode sets (spec.md §17.10; decision-23 §§2–3).
//!
//! CPU forms: 80286 PRM Part I Appendix A pp. A-2–A-5 / Appendix B;
//! 80386 PRM Appendix A pp. A-2–A-7 / §17. Coprocessor forms retain the
//! existing pairing policy (decision-25); SDM Vol 3B §25.18.9 covers aliases.
//! These are encoding names, not decoder feature tags. Bit 0 selects 286,
//! bit 1 selects 386, bit 2 selects 8086/8088 (UM §4.2 Tables 4-12/4-13).
//! Bit 2 marks only rows that the 8086 policy in `decode/i8086.rs` admits;
//! there, WAIT and ESC stay separate instructions. Bit 3 marks the
//! 80186-class rows NEC V20/V30 UM §12 lists for native mode; the V20/V30
//! set is bits 2 and 3. Bit 4 selects early 486 additions; bit 5 selects the
//! CPUID-capable 486. Both also select bit 1 to inherit the 386 set
//! (Intel486 PRM App. A pp. A-4–7). Width variants are separate encodings.
use super::ModelIsa;
mod i8086;
use iced_x86::{Code, Instruction, Register};

// SDM Vol 3B §25.15 pp. 25-5–6: absent reserved encodings raise #UD;
// defined aliases are decoded as their canonical instruction. None means a
// typed refusal for a model whose manual does not define the encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodeAction {
    Defined,
    Alias(Code),
    Absent(Option<u8>),
}

pub(super) const ENCODINGS: &[(Code, u8)] = &[
    // Intel486 PRM App. A pp. A-4–7; §26 instruction entries.
    (Code::Bswap_r16, 48),
    (Code::Bswap_r32, 48),
    (Code::Xadd_rm8_r8, 48),
    (Code::Xadd_rm16_r16, 48),
    (Code::Xadd_rm32_r32, 48),
    (Code::Cmpxchg_rm8_r8, 48),
    (Code::Cmpxchg_rm16_r16, 48),
    (Code::Cmpxchg_rm32_r32, 48),
    (Code::Invd, 48),
    (Code::Wbinvd, 48),
    (Code::Invlpg_m, 48),
    // Embedded Intel486 DM §D.1/Table D-1 pp. D-1–3.
    (Code::Cpuid, 32),
    (Code::Add_rm8_r8, 7),
    (Code::Add_rm16_r16, 7),
    (Code::Add_rm32_r32, 2),
    (Code::Add_r8_rm8, 7),
    (Code::Add_r16_rm16, 7),
    (Code::Add_r32_rm32, 2),
    (Code::Add_AL_imm8, 7),
    (Code::Add_AX_imm16, 7),
    (Code::Add_EAX_imm32, 2),
    (Code::Pushw_ES, 7),
    (Code::Pushd_ES, 2),
    (Code::Popw_ES, 7),
    (Code::Popd_ES, 2),
    (Code::Or_rm8_r8, 7),
    (Code::Or_rm16_r16, 7),
    (Code::Or_rm32_r32, 2),
    (Code::Or_r8_rm8, 7),
    (Code::Or_r16_rm16, 7),
    (Code::Or_r32_rm32, 2),
    (Code::Or_AL_imm8, 7),
    (Code::Or_AX_imm16, 7),
    (Code::Or_EAX_imm32, 2),
    (Code::Pushw_CS, 7),
    (Code::Pushd_CS, 2),
    (Code::Adc_rm8_r8, 7),
    (Code::Adc_rm16_r16, 7),
    (Code::Adc_rm32_r32, 2),
    (Code::Adc_r8_rm8, 7),
    (Code::Adc_r16_rm16, 7),
    (Code::Adc_r32_rm32, 2),
    (Code::Adc_AL_imm8, 7),
    (Code::Adc_AX_imm16, 7),
    (Code::Adc_EAX_imm32, 2),
    (Code::Pushw_SS, 7),
    (Code::Pushd_SS, 2),
    (Code::Popw_SS, 7),
    (Code::Popd_SS, 2),
    (Code::Sbb_rm8_r8, 7),
    (Code::Sbb_rm16_r16, 7),
    (Code::Sbb_rm32_r32, 2),
    (Code::Sbb_r8_rm8, 7),
    (Code::Sbb_r16_rm16, 7),
    (Code::Sbb_r32_rm32, 2),
    (Code::Sbb_AL_imm8, 7),
    (Code::Sbb_AX_imm16, 7),
    (Code::Sbb_EAX_imm32, 2),
    (Code::Pushw_DS, 7),
    (Code::Pushd_DS, 2),
    (Code::Popw_DS, 7),
    (Code::Popd_DS, 2),
    (Code::And_rm8_r8, 7),
    (Code::And_rm16_r16, 7),
    (Code::And_rm32_r32, 2),
    (Code::And_r8_rm8, 7),
    (Code::And_r16_rm16, 7),
    (Code::And_r32_rm32, 2),
    (Code::And_AL_imm8, 7),
    (Code::And_AX_imm16, 7),
    (Code::And_EAX_imm32, 2),
    (Code::Daa, 7),
    (Code::Sub_rm8_r8, 7),
    (Code::Sub_rm16_r16, 7),
    (Code::Sub_rm32_r32, 2),
    (Code::Sub_r8_rm8, 7),
    (Code::Sub_r16_rm16, 7),
    (Code::Sub_r32_rm32, 2),
    (Code::Sub_AL_imm8, 7),
    (Code::Sub_AX_imm16, 7),
    (Code::Sub_EAX_imm32, 2),
    (Code::Das, 7),
    (Code::Xor_rm8_r8, 7),
    (Code::Xor_rm16_r16, 7),
    (Code::Xor_rm32_r32, 2),
    (Code::Xor_r8_rm8, 7),
    (Code::Xor_r16_rm16, 7),
    (Code::Xor_r32_rm32, 2),
    (Code::Xor_AL_imm8, 7),
    (Code::Xor_AX_imm16, 7),
    (Code::Xor_EAX_imm32, 2),
    (Code::Aaa, 7),
    (Code::Cmp_rm8_r8, 7),
    (Code::Cmp_rm16_r16, 7),
    (Code::Cmp_rm32_r32, 2),
    (Code::Cmp_r8_rm8, 7),
    (Code::Cmp_r16_rm16, 7),
    (Code::Cmp_r32_rm32, 2),
    (Code::Cmp_AL_imm8, 7),
    (Code::Cmp_AX_imm16, 7),
    (Code::Cmp_EAX_imm32, 2),
    (Code::Aas, 7),
    (Code::Inc_r16, 7),
    (Code::Inc_r32, 2),
    (Code::Dec_r16, 7),
    (Code::Dec_r32, 2),
    (Code::Push_r16, 7),
    (Code::Push_r32, 2),
    (Code::Pop_r16, 7),
    (Code::Pop_r32, 2),
    (Code::Pushaw, 11),
    (Code::Pushad, 2),
    (Code::Popaw, 11),
    (Code::Popad, 2),
    (Code::Bound_r16_m1616, 11),
    (Code::Bound_r32_m3232, 2),
    (Code::Arpl_rm16_r16, 3),
    (Code::Arpl_r32m16_r32, 2),
    (Code::Push_imm16, 11),
    (Code::Pushd_imm32, 2),
    (Code::Imul_r16_rm16_imm16, 11),
    (Code::Imul_r32_rm32_imm32, 2),
    (Code::Pushw_imm8, 11),
    (Code::Pushd_imm8, 2),
    (Code::Imul_r16_rm16_imm8, 11),
    (Code::Imul_r32_rm32_imm8, 2),
    (Code::Insb_m8_DX, 11),
    (Code::Insw_m16_DX, 11),
    (Code::Insd_m32_DX, 2),
    (Code::Outsb_DX_m8, 11),
    (Code::Outsw_DX_m16, 11),
    (Code::Outsd_DX_m32, 2),
    (Code::Jo_rel8_16, 7),
    (Code::Jo_rel8_32, 2),
    (Code::Jno_rel8_16, 7),
    (Code::Jno_rel8_32, 2),
    (Code::Jb_rel8_16, 7),
    (Code::Jb_rel8_32, 2),
    (Code::Jae_rel8_16, 7),
    (Code::Jae_rel8_32, 2),
    (Code::Je_rel8_16, 7),
    (Code::Je_rel8_32, 2),
    (Code::Jne_rel8_16, 7),
    (Code::Jne_rel8_32, 2),
    (Code::Jbe_rel8_16, 7),
    (Code::Jbe_rel8_32, 2),
    (Code::Ja_rel8_16, 7),
    (Code::Ja_rel8_32, 2),
    (Code::Js_rel8_16, 7),
    (Code::Js_rel8_32, 2),
    (Code::Jns_rel8_16, 7),
    (Code::Jns_rel8_32, 2),
    (Code::Jp_rel8_16, 7),
    (Code::Jp_rel8_32, 2),
    (Code::Jnp_rel8_16, 7),
    (Code::Jnp_rel8_32, 2),
    (Code::Jl_rel8_16, 7),
    (Code::Jl_rel8_32, 2),
    (Code::Jge_rel8_16, 7),
    (Code::Jge_rel8_32, 2),
    (Code::Jle_rel8_16, 7),
    (Code::Jle_rel8_32, 2),
    (Code::Jg_rel8_16, 7),
    (Code::Jg_rel8_32, 2),
    (Code::Add_rm8_imm8, 7),
    (Code::Or_rm8_imm8, 7),
    (Code::Adc_rm8_imm8, 7),
    (Code::Sbb_rm8_imm8, 7),
    (Code::And_rm8_imm8, 7),
    (Code::Sub_rm8_imm8, 7),
    (Code::Xor_rm8_imm8, 7),
    (Code::Cmp_rm8_imm8, 7),
    (Code::Add_rm16_imm16, 7),
    (Code::Add_rm32_imm32, 2),
    (Code::Or_rm16_imm16, 7),
    (Code::Or_rm32_imm32, 2),
    (Code::Adc_rm16_imm16, 7),
    (Code::Adc_rm32_imm32, 2),
    (Code::Sbb_rm16_imm16, 7),
    (Code::Sbb_rm32_imm32, 2),
    (Code::And_rm16_imm16, 7),
    (Code::And_rm32_imm32, 2),
    (Code::Sub_rm16_imm16, 7),
    (Code::Sub_rm32_imm32, 2),
    (Code::Xor_rm16_imm16, 7),
    (Code::Xor_rm32_imm32, 2),
    (Code::Cmp_rm16_imm16, 7),
    (Code::Cmp_rm32_imm32, 2),
    (Code::Add_rm8_imm8_82, 7),
    (Code::Or_rm8_imm8_82, 7),
    (Code::Adc_rm8_imm8_82, 7),
    (Code::Sbb_rm8_imm8_82, 7),
    (Code::And_rm8_imm8_82, 7),
    (Code::Sub_rm8_imm8_82, 7),
    (Code::Xor_rm8_imm8_82, 7),
    (Code::Cmp_rm8_imm8_82, 7),
    (Code::Add_rm16_imm8, 7),
    (Code::Add_rm32_imm8, 2),
    (Code::Or_rm16_imm8, 7),
    (Code::Or_rm32_imm8, 2),
    (Code::Adc_rm16_imm8, 7),
    (Code::Adc_rm32_imm8, 2),
    (Code::Sbb_rm16_imm8, 7),
    (Code::Sbb_rm32_imm8, 2),
    (Code::And_rm16_imm8, 7),
    (Code::And_rm32_imm8, 2),
    (Code::Sub_rm16_imm8, 7),
    (Code::Sub_rm32_imm8, 2),
    (Code::Xor_rm16_imm8, 7),
    (Code::Xor_rm32_imm8, 2),
    (Code::Cmp_rm16_imm8, 7),
    (Code::Cmp_rm32_imm8, 2),
    (Code::Test_rm8_r8, 7),
    (Code::Test_rm16_r16, 7),
    (Code::Test_rm32_r32, 2),
    (Code::Xchg_rm8_r8, 7),
    (Code::Xchg_rm16_r16, 7),
    (Code::Xchg_rm32_r32, 2),
    (Code::Mov_rm8_r8, 7),
    (Code::Mov_rm16_r16, 7),
    (Code::Mov_rm32_r32, 2),
    (Code::Mov_r8_rm8, 7),
    (Code::Mov_r16_rm16, 7),
    (Code::Mov_r32_rm32, 2),
    (Code::Mov_rm16_Sreg, 7),
    (Code::Mov_r32m16_Sreg, 2),
    (Code::Lea_r16_m, 7),
    (Code::Lea_r32_m, 2),
    (Code::Mov_Sreg_rm16, 7),
    (Code::Mov_Sreg_r32m16, 2),
    (Code::Pop_rm16, 7),
    (Code::Pop_rm32, 2),
    (Code::Nopw, 7),
    (Code::Nopd, 3),
    (Code::Nopq, 3),
    (Code::Xchg_r16_AX, 7),
    (Code::Xchg_r32_EAX, 2),
    (Code::Cbw, 7),
    (Code::Cwde, 2),
    (Code::Cwd, 7),
    (Code::Cdq, 2),
    (Code::Call_ptr1616, 7),
    (Code::Call_ptr1632, 2),
    (Code::Wait, 7),
    (Code::Pushfw, 7),
    (Code::Pushfd, 2),
    (Code::Popfw, 7),
    (Code::Popfd, 2),
    (Code::Sahf, 7),
    (Code::Lahf, 7),
    (Code::Mov_AL_moffs8, 7),
    (Code::Mov_AX_moffs16, 7),
    (Code::Mov_EAX_moffs32, 2),
    (Code::Mov_moffs8_AL, 7),
    (Code::Mov_moffs16_AX, 7),
    (Code::Mov_moffs32_EAX, 2),
    (Code::Movsb_m8_m8, 7),
    (Code::Movsw_m16_m16, 7),
    (Code::Movsd_m32_m32, 2),
    (Code::Cmpsb_m8_m8, 7),
    (Code::Cmpsw_m16_m16, 7),
    (Code::Cmpsd_m32_m32, 2),
    (Code::Test_AL_imm8, 7),
    (Code::Test_AX_imm16, 7),
    (Code::Test_EAX_imm32, 2),
    (Code::Stosb_m8_AL, 7),
    (Code::Stosw_m16_AX, 7),
    (Code::Stosd_m32_EAX, 2),
    (Code::Lodsb_AL_m8, 7),
    (Code::Lodsw_AX_m16, 7),
    (Code::Lodsd_EAX_m32, 2),
    (Code::Scasb_AL_m8, 7),
    (Code::Scasw_AX_m16, 7),
    (Code::Scasd_EAX_m32, 2),
    (Code::Mov_r8_imm8, 7),
    (Code::Mov_r16_imm16, 7),
    (Code::Mov_r32_imm32, 2),
    (Code::Rol_rm8_imm8, 11),
    (Code::Ror_rm8_imm8, 11),
    (Code::Rcl_rm8_imm8, 11),
    (Code::Rcr_rm8_imm8, 11),
    (Code::Shl_rm8_imm8, 11),
    (Code::Shr_rm8_imm8, 11),
    (Code::Sal_rm8_imm8, 3),
    (Code::Sar_rm8_imm8, 11),
    (Code::Rol_rm16_imm8, 11),
    (Code::Rol_rm32_imm8, 2),
    (Code::Ror_rm16_imm8, 11),
    (Code::Ror_rm32_imm8, 2),
    (Code::Rcl_rm16_imm8, 11),
    (Code::Rcl_rm32_imm8, 2),
    (Code::Rcr_rm16_imm8, 11),
    (Code::Rcr_rm32_imm8, 2),
    (Code::Shl_rm16_imm8, 11),
    (Code::Shl_rm32_imm8, 2),
    (Code::Shr_rm16_imm8, 11),
    (Code::Shr_rm32_imm8, 2),
    (Code::Sal_rm16_imm8, 3),
    (Code::Sal_rm32_imm8, 2),
    (Code::Sar_rm16_imm8, 11),
    (Code::Sar_rm32_imm8, 2),
    (Code::Retnw_imm16, 7),
    (Code::Retnd_imm16, 2),
    (Code::Retnw, 7),
    (Code::Retnd, 2),
    (Code::Les_r16_m1616, 7),
    (Code::Les_r32_m1632, 2),
    (Code::Lds_r16_m1616, 7),
    (Code::Lds_r32_m1632, 2),
    (Code::Mov_rm8_imm8, 7),
    (Code::Mov_rm16_imm16, 7),
    (Code::Mov_rm32_imm32, 2),
    (Code::Enterw_imm16_imm8, 11),
    (Code::Enterd_imm16_imm8, 2),
    (Code::Leavew, 11),
    (Code::Leaved, 2),
    (Code::Retfw_imm16, 7),
    (Code::Retfd_imm16, 2),
    (Code::Retfw, 7),
    (Code::Retfd, 2),
    (Code::Int3, 7),
    (Code::Int_imm8, 7),
    (Code::Into, 7),
    (Code::Iretw, 7),
    (Code::Iretd, 2),
    (Code::Rol_rm8_1, 7),
    (Code::Ror_rm8_1, 7),
    (Code::Rcl_rm8_1, 7),
    (Code::Rcr_rm8_1, 7),
    (Code::Shl_rm8_1, 7),
    (Code::Shr_rm8_1, 7),
    (Code::Sal_rm8_1, 3),
    (Code::Sar_rm8_1, 7),
    (Code::Rol_rm16_1, 7),
    (Code::Rol_rm32_1, 2),
    (Code::Ror_rm16_1, 7),
    (Code::Ror_rm32_1, 2),
    (Code::Rcl_rm16_1, 7),
    (Code::Rcl_rm32_1, 2),
    (Code::Rcr_rm16_1, 7),
    (Code::Rcr_rm32_1, 2),
    (Code::Shl_rm16_1, 7),
    (Code::Shl_rm32_1, 2),
    (Code::Shr_rm16_1, 7),
    (Code::Shr_rm32_1, 2),
    (Code::Sal_rm16_1, 3),
    (Code::Sal_rm32_1, 2),
    (Code::Sar_rm16_1, 7),
    (Code::Sar_rm32_1, 2),
    (Code::Rol_rm8_CL, 7),
    (Code::Ror_rm8_CL, 7),
    (Code::Rcl_rm8_CL, 7),
    (Code::Rcr_rm8_CL, 7),
    (Code::Shl_rm8_CL, 7),
    (Code::Shr_rm8_CL, 7),
    (Code::Sal_rm8_CL, 3),
    (Code::Sar_rm8_CL, 7),
    (Code::Rol_rm16_CL, 7),
    (Code::Rol_rm32_CL, 2),
    (Code::Ror_rm16_CL, 7),
    (Code::Ror_rm32_CL, 2),
    (Code::Rcl_rm16_CL, 7),
    (Code::Rcl_rm32_CL, 2),
    (Code::Rcr_rm16_CL, 7),
    (Code::Rcr_rm32_CL, 2),
    (Code::Shl_rm16_CL, 7),
    (Code::Shl_rm32_CL, 2),
    (Code::Shr_rm16_CL, 7),
    (Code::Shr_rm32_CL, 2),
    (Code::Sal_rm16_CL, 3),
    (Code::Sal_rm32_CL, 2),
    (Code::Sar_rm16_CL, 7),
    (Code::Sar_rm32_CL, 2),
    (Code::Aam_imm8, 7),
    (Code::Aad_imm8, 7),
    (Code::Salc, 3),
    (Code::Xlat_m8, 7),
    (Code::Fadd_m32fp, 7),
    (Code::Fmul_m32fp, 7),
    (Code::Fcom_m32fp, 7),
    (Code::Fcomp_m32fp, 7),
    (Code::Fsub_m32fp, 7),
    (Code::Fsubr_m32fp, 7),
    (Code::Fdiv_m32fp, 7),
    (Code::Fdivr_m32fp, 7),
    (Code::Fadd_st0_sti, 7),
    (Code::Fmul_st0_sti, 7),
    (Code::Fcom_st0_sti, 7),
    (Code::Fcomp_st0_sti, 7),
    (Code::Fsub_st0_sti, 7),
    (Code::Fsubr_st0_sti, 7),
    (Code::Fdiv_st0_sti, 7),
    (Code::Fdivr_st0_sti, 7),
    (Code::Fld_m32fp, 7),
    (Code::Fst_m32fp, 7),
    (Code::Fstp_m32fp, 7),
    (Code::Fldenv_m14byte, 7),
    (Code::Fldenv_m28byte, 2),
    (Code::Fldcw_m2byte, 7),
    (Code::Fnstenv_m14byte, 7),
    (Code::Fstenv_m14byte, 3),
    (Code::Fnstenv_m28byte, 2),
    (Code::Fstenv_m28byte, 2),
    (Code::Fnstcw_m2byte, 7),
    (Code::Fstcw_m2byte, 3),
    (Code::Fld_sti, 7),
    (Code::Fxch_st0_sti, 7),
    (Code::Fnop, 7),
    (Code::Fstpnce_sti, 7),
    (Code::Fchs, 7),
    (Code::Fabs, 7),
    (Code::Ftst, 7),
    (Code::Fxam, 7),
    (Code::Fld1, 7),
    (Code::Fldl2t, 7),
    (Code::Fldl2e, 7),
    (Code::Fldpi, 7),
    (Code::Fldlg2, 7),
    (Code::Fldln2, 7),
    (Code::Fldz, 7),
    (Code::F2xm1, 7),
    (Code::Fyl2x, 7),
    (Code::Fptan, 7),
    (Code::Fpatan, 7),
    (Code::Fxtract, 7),
    (Code::Fprem1, 2),
    (Code::Fdecstp, 7),
    (Code::Fincstp, 7),
    (Code::Fprem, 7),
    (Code::Fyl2xp1, 7),
    (Code::Fsqrt, 7),
    (Code::Fsincos, 2),
    (Code::Frndint, 7),
    (Code::Fscale, 7),
    (Code::Fsin, 2),
    (Code::Fcos, 2),
    (Code::Fiadd_m32int, 7),
    (Code::Fimul_m32int, 7),
    (Code::Ficom_m32int, 7),
    (Code::Ficomp_m32int, 7),
    (Code::Fisub_m32int, 7),
    (Code::Fisubr_m32int, 7),
    (Code::Fidiv_m32int, 7),
    (Code::Fidivr_m32int, 7),
    (Code::Fucompp, 2),
    (Code::Fild_m32int, 7),
    (Code::Fist_m32int, 7),
    (Code::Fistp_m32int, 7),
    (Code::Fld_m80fp, 7),
    (Code::Fstp_m80fp, 7),
    (Code::Fneni, 7),
    (Code::Feni, 3),
    (Code::Fndisi, 7),
    (Code::Fdisi, 3),
    (Code::Fnclex, 7),
    (Code::Fclex, 3),
    (Code::Fninit, 7),
    (Code::Finit, 3),
    (Code::Fnsetpm, 7),
    (Code::Fsetpm, 3),
    (Code::Fadd_m64fp, 7),
    (Code::Fmul_m64fp, 7),
    (Code::Fcom_m64fp, 7),
    (Code::Fcomp_m64fp, 7),
    (Code::Fsub_m64fp, 7),
    (Code::Fsubr_m64fp, 7),
    (Code::Fdiv_m64fp, 7),
    (Code::Fdivr_m64fp, 7),
    (Code::Fadd_sti_st0, 7),
    (Code::Fmul_sti_st0, 7),
    (Code::Fcom_st0_sti_DCD0, 7),
    (Code::Fcomp_st0_sti_DCD8, 7),
    (Code::Fsubr_sti_st0, 7),
    (Code::Fsub_sti_st0, 7),
    (Code::Fdivr_sti_st0, 7),
    (Code::Fdiv_sti_st0, 7),
    (Code::Fld_m64fp, 7),
    (Code::Fst_m64fp, 7),
    (Code::Fstp_m64fp, 7),
    (Code::Frstor_m94byte, 7),
    (Code::Frstor_m108byte, 2),
    (Code::Fnsave_m94byte, 7),
    (Code::Fsave_m94byte, 3),
    (Code::Fnsave_m108byte, 2),
    (Code::Fsave_m108byte, 2),
    (Code::Fnstsw_m2byte, 7),
    (Code::Fstsw_m2byte, 3),
    (Code::Ffree_sti, 7),
    (Code::Fxch_st0_sti_DDC8, 7),
    (Code::Fst_sti, 7),
    (Code::Fstp_sti, 7),
    (Code::Fucom_st0_sti, 7),
    (Code::Fucomp_st0_sti, 7),
    (Code::Fiadd_m16int, 7),
    (Code::Fimul_m16int, 7),
    (Code::Ficom_m16int, 7),
    (Code::Ficomp_m16int, 7),
    (Code::Fisub_m16int, 7),
    (Code::Fisubr_m16int, 7),
    (Code::Fidiv_m16int, 7),
    (Code::Fidivr_m16int, 7),
    (Code::Faddp_sti_st0, 7),
    (Code::Fmulp_sti_st0, 7),
    (Code::Fcomp_st0_sti_DED0, 7),
    (Code::Fcompp, 7),
    (Code::Fsubrp_sti_st0, 7),
    (Code::Fsubp_sti_st0, 7),
    (Code::Fdivrp_sti_st0, 7),
    (Code::Fdivp_sti_st0, 7),
    (Code::Fild_m16int, 7),
    (Code::Fist_m16int, 7),
    (Code::Fistp_m16int, 7),
    (Code::Fbld_m80bcd, 7),
    (Code::Fild_m64int, 7),
    (Code::Fbstp_m80bcd, 7),
    (Code::Fistp_m64int, 7),
    (Code::Ffreep_sti, 7),
    (Code::Fxch_st0_sti_DFC8, 7),
    (Code::Fstp_sti_DFD0, 7),
    (Code::Fstp_sti_DFD8, 7),
    (Code::Fnstsw_AX, 7),
    (Code::Fstsw_AX, 3),
    (Code::Loopne_rel8_16_CX, 7),
    (Code::Loopne_rel8_32_CX, 2),
    (Code::Loopne_rel8_16_ECX, 2),
    (Code::Loopne_rel8_32_ECX, 2),
    (Code::Loope_rel8_16_CX, 7),
    (Code::Loope_rel8_32_CX, 2),
    (Code::Loope_rel8_16_ECX, 2),
    (Code::Loope_rel8_32_ECX, 2),
    (Code::Loop_rel8_16_CX, 7),
    (Code::Loop_rel8_32_CX, 2),
    (Code::Loop_rel8_16_ECX, 2),
    (Code::Loop_rel8_32_ECX, 2),
    (Code::Jcxz_rel8_16, 7),
    (Code::Jcxz_rel8_32, 2),
    (Code::Jecxz_rel8_16, 2),
    (Code::Jecxz_rel8_32, 2),
    (Code::In_AL_imm8, 7),
    (Code::In_AX_imm8, 7),
    (Code::In_EAX_imm8, 2),
    (Code::Out_imm8_AL, 7),
    (Code::Out_imm8_AX, 7),
    (Code::Out_imm8_EAX, 2),
    (Code::Call_rel16, 7),
    (Code::Call_rel32_32, 2),
    (Code::Jmp_rel16, 7),
    (Code::Jmp_rel32_32, 2),
    (Code::Jmp_ptr1616, 7),
    (Code::Jmp_ptr1632, 2),
    (Code::Jmp_rel8_16, 7),
    (Code::Jmp_rel8_32, 2),
    (Code::In_AL_DX, 7),
    (Code::In_AX_DX, 7),
    (Code::In_EAX_DX, 2),
    (Code::Out_DX_AL, 7),
    (Code::Out_DX_AX, 7),
    (Code::Out_DX_EAX, 2),
    (Code::Int1, 2),
    (Code::Hlt, 7),
    (Code::Cmc, 7),
    (Code::Test_rm8_imm8, 7),
    (Code::Test_rm8_imm8_F6r1, 7),
    (Code::Not_rm8, 7),
    (Code::Neg_rm8, 7),
    (Code::Mul_rm8, 7),
    (Code::Imul_rm8, 7),
    (Code::Div_rm8, 7),
    (Code::Idiv_rm8, 7),
    (Code::Test_rm16_imm16, 7),
    (Code::Test_rm32_imm32, 2),
    (Code::Test_rm16_imm16_F7r1, 7),
    (Code::Test_rm32_imm32_F7r1, 2),
    (Code::Not_rm16, 7),
    (Code::Not_rm32, 2),
    (Code::Neg_rm16, 7),
    (Code::Neg_rm32, 2),
    (Code::Mul_rm16, 7),
    (Code::Mul_rm32, 2),
    (Code::Imul_rm16, 7),
    (Code::Imul_rm32, 2),
    (Code::Div_rm16, 7),
    (Code::Div_rm32, 2),
    (Code::Idiv_rm16, 7),
    (Code::Idiv_rm32, 2),
    (Code::Clc, 7),
    (Code::Stc, 7),
    (Code::Cli, 7),
    (Code::Sti, 7),
    (Code::Cld, 7),
    (Code::Std, 7),
    (Code::Inc_rm8, 7),
    (Code::Dec_rm8, 7),
    (Code::Inc_rm16, 7),
    (Code::Inc_rm32, 2),
    (Code::Dec_rm16, 7),
    (Code::Dec_rm32, 2),
    (Code::Call_rm16, 7),
    (Code::Call_rm32, 2),
    (Code::Call_m1616, 7),
    (Code::Call_m1632, 2),
    (Code::Jmp_rm16, 7),
    (Code::Jmp_rm32, 2),
    (Code::Jmp_m1616, 7),
    (Code::Jmp_m1632, 2),
    (Code::Push_rm16, 7),
    (Code::Push_rm32, 2),
    (Code::Sldt_rm16, 3),
    (Code::Sldt_r32m16, 2),
    (Code::Str_rm16, 3),
    (Code::Str_r32m16, 2),
    (Code::Lldt_rm16, 3),
    (Code::Lldt_r32m16, 2),
    (Code::Ltr_rm16, 3),
    (Code::Ltr_r32m16, 2),
    (Code::Verr_rm16, 3),
    (Code::Verr_r32m16, 2),
    (Code::Verw_rm16, 3),
    (Code::Verw_r32m16, 2),
    (Code::Sgdt_m1632_16, 3),
    (Code::Sgdt_m1632, 2),
    (Code::Sidt_m1632_16, 3),
    (Code::Sidt_m1632, 2),
    (Code::Lgdt_m1632_16, 3),
    (Code::Lgdt_m1632, 2),
    (Code::Lidt_m1632_16, 3),
    (Code::Lidt_m1632, 2),
    (Code::Smsw_rm16, 3),
    (Code::Smsw_r32m16, 2),
    (Code::Lmsw_rm16, 3),
    (Code::Lmsw_r32m16, 2),
    (Code::Lar_r16_rm16, 3),
    (Code::Lar_r32_r32m16, 2),
    (Code::Lsl_r16_rm16, 3),
    (Code::Lsl_r32_r32m16, 2),
    (Code::Storeall, 3),
    // Intel, "Undocumented iAPX 286 Test Instruction"
    // (`oracles/intel/i286-loadall.pdf`), pp. 1-2: LOADALL is 0F 05 on the
    // 80286 and "will not work on an iAPX 386". p. 13 says "opcode 0F04H"
    // and that 0F05H halts; p. 2's bit encoding governs (DERIVED, spec.md).
    (Code::Loadall286, 1),
    (Code::Clts, 3),
    (Code::Mov_r32_cr, 2),
    (Code::Mov_r32_dr, 2),
    (Code::Mov_cr_r32, 2),
    (Code::Mov_dr_r32, 2),
    (Code::Jo_rel16, 2),
    (Code::Jo_rel32_32, 2),
    (Code::Jno_rel16, 2),
    (Code::Jno_rel32_32, 2),
    (Code::Jb_rel16, 2),
    (Code::Jb_rel32_32, 2),
    (Code::Jae_rel16, 2),
    (Code::Jae_rel32_32, 2),
    (Code::Je_rel16, 2),
    (Code::Je_rel32_32, 2),
    (Code::Jne_rel16, 2),
    (Code::Jne_rel32_32, 2),
    (Code::Jbe_rel16, 2),
    (Code::Jbe_rel32_32, 2),
    (Code::Ja_rel16, 2),
    (Code::Ja_rel32_32, 2),
    (Code::Js_rel16, 2),
    (Code::Js_rel32_32, 2),
    (Code::Jns_rel16, 2),
    (Code::Jns_rel32_32, 2),
    (Code::Jp_rel16, 2),
    (Code::Jp_rel32_32, 2),
    (Code::Jnp_rel16, 2),
    (Code::Jnp_rel32_32, 2),
    (Code::Jl_rel16, 2),
    (Code::Jl_rel32_32, 2),
    (Code::Jge_rel16, 2),
    (Code::Jge_rel32_32, 2),
    (Code::Jle_rel16, 2),
    (Code::Jle_rel32_32, 2),
    (Code::Jg_rel16, 2),
    (Code::Jg_rel32_32, 2),
    (Code::Seto_rm8, 2),
    (Code::Setno_rm8, 2),
    (Code::Setb_rm8, 2),
    (Code::Setae_rm8, 2),
    (Code::Sete_rm8, 2),
    (Code::Setne_rm8, 2),
    (Code::Setbe_rm8, 2),
    (Code::Seta_rm8, 2),
    (Code::Sets_rm8, 2),
    (Code::Setns_rm8, 2),
    (Code::Setp_rm8, 2),
    (Code::Setnp_rm8, 2),
    (Code::Setl_rm8, 2),
    (Code::Setge_rm8, 2),
    (Code::Setle_rm8, 2),
    (Code::Setg_rm8, 2),
    (Code::Pushw_FS, 2),
    (Code::Pushd_FS, 2),
    (Code::Popw_FS, 2),
    (Code::Popd_FS, 2),
    (Code::Bt_rm16_r16, 2),
    (Code::Bt_rm32_r32, 2),
    (Code::Shld_rm16_r16_imm8, 2),
    (Code::Shld_rm32_r32_imm8, 2),
    (Code::Shld_rm16_r16_CL, 2),
    (Code::Shld_rm32_r32_CL, 2),
    (Code::Pushw_GS, 2),
    (Code::Pushd_GS, 2),
    (Code::Popw_GS, 2),
    (Code::Popd_GS, 2),
    (Code::Bts_rm16_r16, 2),
    (Code::Bts_rm32_r32, 2),
    (Code::Shrd_rm16_r16_imm8, 2),
    (Code::Shrd_rm32_r32_imm8, 2),
    (Code::Shrd_rm16_r16_CL, 2),
    (Code::Shrd_rm32_r32_CL, 2),
    (Code::Imul_r16_rm16, 2),
    (Code::Imul_r32_rm32, 2),
    (Code::Lss_r16_m1616, 2),
    (Code::Lss_r32_m1632, 2),
    (Code::Btr_rm16_r16, 2),
    (Code::Btr_rm32_r32, 2),
    (Code::Lfs_r16_m1616, 2),
    (Code::Lfs_r32_m1632, 2),
    (Code::Lgs_r16_m1616, 2),
    (Code::Lgs_r32_m1632, 2),
    (Code::Movzx_r16_rm8, 2),
    (Code::Movzx_r32_rm8, 2),
    (Code::Movzx_r16_rm16, 2),
    (Code::Movzx_r32_rm16, 2),
    (Code::Bt_rm16_imm8, 2),
    (Code::Bt_rm32_imm8, 2),
    (Code::Bts_rm16_imm8, 2),
    (Code::Bts_rm32_imm8, 2),
    (Code::Btr_rm16_imm8, 2),
    (Code::Btr_rm32_imm8, 2),
    (Code::Btc_rm16_imm8, 2),
    (Code::Btc_rm32_imm8, 2),
    (Code::Btc_rm16_r16, 2),
    (Code::Btc_rm32_r32, 2),
    (Code::Bsf_r16_rm16, 2),
    (Code::Bsf_r32_rm32, 2),
    (Code::Bsr_r16_rm16, 2),
    (Code::Bsr_r32_rm32, 2),
    (Code::Movsx_r16_rm8, 2),
    (Code::Movsx_r32_rm8, 2),
    (Code::Movsx_r16_rm16, 2),
    (Code::Movsx_r32_rm16, 2),
    // 80386 PRM §17 MOV to/from Special Registers p. 17-111:
    // the shared test-register encoding exists; only TR6/TR7 exist.
    (Code::Mov_r32_tr, 2),
    (Code::Mov_tr_r32, 2),
];

const TABLE_LEN: usize = {
    let mut max = 0;
    let mut i = 0;
    while i < ENCODINGS.len() {
        if ENCODINGS[i].0 as usize > max {
            max = ENCODINGS[i].0 as usize;
        }
        i += 1;
    }
    max + 1
};
static MODELS_BY_CODE: [u8; TABLE_LEN] = {
    let mut table = [0; TABLE_LEN];
    let mut i = 0;
    while i < ENCODINGS.len() {
        table[ENCODINGS[i].0 as usize] = ENCODINGS[i].1
            | if defined_alias(ENCODINGS[i].0).is_some() {
                0x80
            } else {
                0
            };
        i += 1;
    }
    table
};

struct DecodeSet {
    models: u8,
    options: u32,
    absent: Option<u8>,
    legacy_segments_only: bool,
}

// 80286 PRM App.B #UD p.B-12; 80386 PRM §14.6 Table14-1 p.14-5:
// both models have #UD. None is reserved for a model with typed refusals.
// MOV_TR exposes the 386 special-register forms (PRM §17 p.17-111);
// encoding membership still denies them on the 286. LOADALL286 names 0F 05
// LOADALL (Intel, "Undocumented iAPX 286 Test Instruction", p. 2) and 0F 04
// STOREALL, which `unavailable_prefix` keeps #UD (App. B p. B-12) although
// that document's p. 13 gives 0F04H exception 13 at CPL > 0 (spec.md).
const I286_SET: DecodeSet = DecodeSet {
    models: 1,
    options: iced_x86::DecoderOptions::MOV_TR | iced_x86::DecoderOptions::LOADALL286,
    absent: Some(6),
    legacy_segments_only: true,
};
const I486_SET: DecodeSet = DecodeSet {
    models: 18,
    options: iced_x86::DecoderOptions::MOV_TR,
    absent: Some(6),
    legacy_segments_only: false,
};
const I486_CPUID_SET: DecodeSet = DecodeSet {
    models: 34,
    ..I486_SET
};
const I386_SET: DecodeSet = DecodeSet {
    models: 2,
    options: iced_x86::DecoderOptions::MOV_TR,
    absent: Some(6),
    legacy_segments_only: false,
};

// The existing system VM handles these two rows after the twenty-bit dispatch.
// Keep the common decoder flags a catalogue-derived constant on that path.
// They decide where a fetch ends and which forms the shared walk may take.
// The 286's LOADALL286 renames only the two-byte 0F 04 and 0F 05, neither
// of which the shared walk takes, so the common flags omit it.
pub(crate) const SYSTEM_DECODER_OPTIONS: u32 = {
    assert!(I286_SET.options & !iced_x86::DecoderOptions::LOADALL286 == I386_SET.options);
    I386_SET.options
};

// The compatibility-generation adapter uses the same rows. Const blocks avoid
// selecting a profile and a second table pointer merely to read decoder flags.
// Like profile_for_generation, the I8086 generation keeps its C1 projection;
// I8088/I8086 fetch through their own entry (vm/i8086.rs).
#[inline(always)]
pub(crate) const fn decoder_options_for_generation(generation: crate::CpuGeneration) -> u32 {
    match generation {
        crate::CpuGeneration::I286 | crate::CpuGeneration::I8086 => const { I286_SET.options },
        crate::CpuGeneration::I386 => const { I386_SET.options },
    }
}

// Project immutable catalogue rows for incremental compatibility fetch. The
// per-byte path needs prefix policy, without selecting two table pointers.
#[inline(always)]
pub(crate) fn unavailable_prefix_for_generation(
    generation: crate::CpuGeneration,
    code: &[u8],
) -> bool {
    match generation {
        crate::CpuGeneration::I286 | crate::CpuGeneration::I8086 => {
            const ISA: ModelIsa = super::CpuModel::I80286.profile().isa;
            (const { ISA.legacy_segments_only() }) && ISA.unavailable_prefix(code)
        }
        crate::CpuGeneration::I386 => {
            const ISA: ModelIsa = super::CpuModel::I80386.profile().isa;
            (const { ISA.legacy_segments_only() }) && ISA.unavailable_prefix(code)
        }
    }
}

impl ModelIsa {
    const fn set(self) -> &'static DecodeSet {
        match self {
            Self::I8086 => &i8086::SET,
            Self::NecV20 => &i8086::NEC_SET,
            Self::I80286 => &I286_SET,
            Self::I80386 => &I386_SET,
            Self::I80486Dx => &I486_SET,
            Self::I80486DxCpuid => &I486_CPUID_SET,
        }
    }
    pub(super) const fn model_mask(self) -> u8 {
        self.set().models
    }
    pub(crate) const fn legacy_segments_only(self) -> bool {
        self.set().legacy_segments_only
    }
    pub(crate) const fn decoder_options(self) -> u32 {
        self.set().options
    }
    pub(crate) const fn absent_encoding(self) -> Option<u8> {
        self.set().absent
    }
    /// Earliest byte at which the manual establishes an invalid encoding.
    pub(crate) fn unavailable_prefix(self, code: &[u8]) -> bool {
        if self == Self::I8086 {
            return i8086::known_absent(code);
        }
        if self == Self::NecV20 {
            return i8086::known_absent_nec(code);
        }
        let mut opcode = code
            .iter()
            .skip_while(|b| matches!(b, 0x26 | 0x2e | 0x36 | 0x3e | 0xf0 | 0xf2 | 0xf3));
        // 80286 PRM App. B #UD p. B-12: first/second opcode, extension,
        // invalid Sreg or register in a memory-only form are independently #UD.
        // 0F 05 is LOADALL (Intel, "Undocumented iAPX 286 Test Instruction",
        // p. 2); 0F 04 stays #UD.
        if self.set().legacy_segments_only {
            match opcode.next() {
                Some(0x64..=0x67) => true,
                Some(0x0f) => match opcode.next() {
                    Some(0x00) => opcode.next().is_some_and(|b| b & 0x38 >= 0x30),
                    Some(0x01) => opcode.next().is_some_and(|b| {
                        matches!(b & 0x38, 0x28 | 0x38) || (b & 0xc0 == 0xc0 && b & 0x38 <= 0x18)
                    }),
                    Some(0x02 | 0x03 | 0x05 | 0x06) | None => false,
                    Some(_) => true,
                },
                _ => false,
            }
        } else {
            false
        }
    }

    #[inline]
    pub(crate) fn decode(self, bytes: &[u8], insn: &Instruction) -> DecodeAction {
        if self == Self::I8086 {
            return i8086::decode(bytes, insn);
        }
        if self == Self::NecV20 {
            return i8086::decode_nec(bytes, insn);
        }
        let mask = self.model_mask();
        let absent = DecodeAction::Absent(self.absent_encoding());
        let models = MODELS_BY_CODE
            .get(insn.code() as usize)
            .copied()
            .unwrap_or(0);
        if models & mask == 0 {
            return absent;
        }
        if self.set().legacy_segments_only
            && (self.unavailable_prefix(bytes)
                || (0..insn.op_count())
                    .any(|i| matches!(insn.op_register(i), Register::FS | Register::GS)))
        {
            return absent;
        }
        // Only these admitted forms can use the memory-only table slots.
        // Other instructions in those slots are already absent from the set.
        if matches!(
            insn.code(),
            Code::Sgdt_m1632_16
                | Code::Sgdt_m1632
                | Code::Sidt_m1632_16
                | Code::Sidt_m1632
                | Code::Lgdt_m1632_16
                | Code::Lgdt_m1632
                | Code::Lidt_m1632_16
                | Code::Lidt_m1632
        ) && insn.op0_kind() != iced_x86::OpKind::Memory
        {
            return absent;
        }
        if matches!(insn.code(), Code::Mov_r32_tr | Code::Mov_tr_r32)
            && !(0..insn.op_count()).any(|i| {
                matches!(insn.op_register(i), Register::TR6 | Register::TR7)
                    || matches!(self, Self::I80486Dx | Self::I80486DxCpuid)
                        && matches!(
                            insn.op_register(i),
                            Register::TR3 | Register::TR4 | Register::TR5
                        )
            })
        {
            return absent;
        }
        if models & 0x80 == 0 {
            DecodeAction::Defined
        } else {
            DecodeAction::Alias(defined_alias(insn.code()).expect("catalogue alias"))
        }
    }
}

/// 80286 PRM App. C item 11 p. C-2: LOCK is meant for XCHG, MOV, MOVS, INS
/// and OUTS, and App. B LOCK p. B-68 gives no real-mode exception for the
/// prefix. The strict decoder rejects it on the last four; for an entry whose
/// `lock_data_movement` quirk admits them, this decodes `bytes` (one whole
/// instruction) with the prefix kept. Bytes the strict decoder rejects for
/// any other reason stay rejected: without its LOCK prefixes the instruction
/// must decode strictly, to the same encoding and length.
pub(crate) fn locked_data_movement(
    bytes: &[u8],
    bitness: u32,
    ip: u64,
    options: u32,
) -> Option<Instruction> {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic::*};
    let prefixes = bytes
        .iter()
        .take_while(|b| {
            matches!(
                b,
                0x26 | 0x2e | 0x36 | 0x3e | 0x64..=0x67 | 0xf0 | 0xf2 | 0xf3
            )
        })
        .count();
    if bytes.len() > 15 || !bytes[..prefixes].contains(&0xf0) {
        return None;
    }
    let mut unlocked = [0; 15];
    let mut len = 0;
    for &byte in bytes[..prefixes].iter().filter(|b| **b != 0xf0) {
        unlocked[len] = byte;
        len += 1;
    }
    unlocked[len..len + bytes.len() - prefixes].copy_from_slice(&bytes[prefixes..]);
    len += bytes.len() - prefixes;
    let plain = Decoder::with_ip(bitness, &unlocked[..len], ip, options).decode();
    if plain.is_invalid()
        || plain.len() != len
        || !matches!(
            plain.mnemonic(),
            Mov | Movsb | Movsw | Movsd | Insb | Insw | Insd | Outsb | Outsw | Outsd
        )
    {
        return None;
    }
    let locked = Decoder::with_ip(
        bitness,
        bytes,
        ip,
        options | DecoderOptions::NO_INVALID_CHECK,
    )
    .decode();
    (!locked.is_invalid()
        && locked.len() == bytes.len()
        && locked.code() == plain.code()
        && locked.has_lock_prefix())
    .then_some(locked)
}

pub(super) const fn defined_alias(code: Code) -> Option<Code> {
    // SDM Vol 3B §25.15 p.25-6: 82h, group 2 /6 and group 3 /1 are defined aliases.
    use Code::*;
    let canonical = match code {
        Add_rm8_imm8_82 => Add_rm8_imm8,
        Or_rm8_imm8_82 => Or_rm8_imm8,
        Adc_rm8_imm8_82 => Adc_rm8_imm8,
        Sbb_rm8_imm8_82 => Sbb_rm8_imm8,
        And_rm8_imm8_82 => And_rm8_imm8,
        Sub_rm8_imm8_82 => Sub_rm8_imm8,
        Xor_rm8_imm8_82 => Xor_rm8_imm8,
        Cmp_rm8_imm8_82 => Cmp_rm8_imm8,
        Test_rm8_imm8_F6r1 => Test_rm8_imm8,
        Test_rm16_imm16_F7r1 => Test_rm16_imm16,
        Test_rm32_imm32_F7r1 => Test_rm32_imm32,
        Sal_rm8_imm8 => Shl_rm8_imm8,
        Sal_rm16_imm8 => Shl_rm16_imm8,
        Sal_rm32_imm8 => Shl_rm32_imm8,
        Sal_rm8_1 => Shl_rm8_1,
        Sal_rm16_1 => Shl_rm16_1,
        Sal_rm32_1 => Shl_rm32_1,
        Sal_rm8_CL => Shl_rm8_CL,
        Sal_rm16_CL => Shl_rm16_CL,
        Sal_rm32_CL => Shl_rm32_CL,
        _ => return None,
    };
    Some(canonical)
}

#[cfg(test)]
mod lock_tests {
    use super::*;
    use iced_x86::{Decoder, DecoderError, DecoderOptions};

    // 80286 PRM App. C item 11 p. C-2; App. B LOCK p. B-68.
    #[test]
    fn locked_data_movement_admits_only_the_listed_forms() {
        let options = I286_SET.options;
        for (bytes, code) in [
            (&[0xf0, 0x88, 0x07][..], Code::Mov_rm8_r8),
            (&[0xf0, 0x89, 0x47, 0x02], Code::Mov_rm16_r16),
            (&[0xf0, 0x8b, 0x07], Code::Mov_r16_rm16),
            (&[0xf0, 0xb0, 0x5a], Code::Mov_r8_imm8),
            (
                &[0xf0, 0xc7, 0x06, 0x34, 0x12, 0x78, 0x56],
                Code::Mov_rm16_imm16,
            ),
            (&[0xf0, 0xa1, 0x34, 0x12], Code::Mov_AX_moffs16),
            (&[0xf0, 0x8e, 0xd8], Code::Mov_Sreg_rm16),
            (&[0x26, 0xf0, 0xa5], Code::Movsw_m16_m16),
            (&[0xf0, 0xf3, 0xa4], Code::Movsb_m8_m8),
            (&[0xf0, 0x6c], Code::Insb_m8_DX),
            (&[0xf0, 0xf3, 0x6f], Code::Outsw_DX_m16),
        ] {
            assert!(Decoder::with_ip(16, bytes, 0, options)
                .decode()
                .is_invalid());
            let insn = locked_data_movement(bytes, 16, 0, options).expect("admitted");
            assert_eq!(
                (insn.code(), insn.len()),
                (code, bytes.len()),
                "{bytes:02x?}"
            );
            assert!(insn.has_lock_prefix());
            // A shorter fetch still asks for more bytes, so incremental
            // fetching reads the whole instruction.
            for end in 1..bytes.len() {
                let mut decoder = Decoder::with_ip(
                    16,
                    &bytes[..end],
                    0,
                    options | DecoderOptions::NO_INVALID_CHECK,
                );
                assert!(decoder.decode().is_invalid());
                assert_eq!(decoder.last_error(), DecoderError::NoMoreBytes);
            }
        }
        for bytes in [
            &[0xf0, 0x00, 0xec][..], // LOCK ADD AH,CH: not a listed form
            &[0xf0, 0x87, 0xc3],     // LOCK XCHG BX,AX: not in this quirk
            &[0xf0, 0xaa],           // LOCK STOSB
            &[0xf0, 0x8e, 0xc8],     // MOV CS,AX: invalid without LOCK too
            &[0x88, 0x07],           // no LOCK prefix
            &[0xf0, 0x88],           // truncated
        ] {
            assert!(
                locked_data_movement(bytes, 16, 0, options).is_none(),
                "{bytes:02x?}"
            );
        }
        // LOCK on a lockable read-modify-write form is valid without this path.
        assert!(!Decoder::with_ip(16, &[0xf0, 0x00, 0x07], 0, options)
            .decode()
            .is_invalid());
    }
}
