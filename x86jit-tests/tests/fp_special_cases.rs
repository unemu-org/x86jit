//! Exact SIMD special values: SDM Vol 1 §§4.8.3.5, 4.8.3.7; Vol 2 §3.3/§4.3/§5.2.

use iced_x86::code_asm::*;
use x86jit_core::{
    Backend, Exit, InterpreterBackend, Prot, Reg, RegionCaps, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::{HostTarget, JitBackend, OptLevel};

macro_rules! op {
    ($name:literal, $double:literal, $op:ident($($arg:expr),*)) => {
        ($name, $double, |a: &mut CodeAssembler| { a.$op($($arg),*).unwrap(); })
    };
}

type Recipe = (&'static str, bool, fn(&mut CodeAssembler));

fn recipes() -> Vec<Recipe> {
    vec![
        op!("cmpss_imm0", false, cmpss(xmm0, xmm1, 0)),
        op!("vcmpss_imm0", false, vcmpss(xmm0, xmm0, xmm1, 0)),
        op!("cmpss_imm1", false, cmpss(xmm0, xmm1, 1)),
        op!("vcmpss_imm1", false, vcmpss(xmm0, xmm0, xmm1, 1)),
        op!("cmpss_imm2", false, cmpss(xmm0, xmm1, 2)),
        op!("vcmpss_imm2", false, vcmpss(xmm0, xmm0, xmm1, 2)),
        op!("cmpss_imm3", false, cmpss(xmm0, xmm1, 3)),
        op!("vcmpss_imm3", false, vcmpss(xmm0, xmm0, xmm1, 3)),
        op!("cmpss_imm4", false, cmpss(xmm0, xmm1, 4)),
        op!("vcmpss_imm4", false, vcmpss(xmm0, xmm0, xmm1, 4)),
        op!("cmpss_imm5", false, cmpss(xmm0, xmm1, 5)),
        op!("vcmpss_imm5", false, vcmpss(xmm0, xmm0, xmm1, 5)),
        op!("cmpss_imm6", false, cmpss(xmm0, xmm1, 6)),
        op!("vcmpss_imm6", false, vcmpss(xmm0, xmm0, xmm1, 6)),
        op!("cmpss_imm7", false, cmpss(xmm0, xmm1, 7)),
        op!("vcmpss_imm7", false, vcmpss(xmm0, xmm0, xmm1, 7)),
        op!("vcmpss_imm8", false, vcmpss(xmm0, xmm0, xmm1, 8)),
        op!("vcmpss_imm9", false, vcmpss(xmm0, xmm0, xmm1, 9)),
        op!("vcmpss_imm10", false, vcmpss(xmm0, xmm0, xmm1, 10)),
        op!("vcmpss_imm11", false, vcmpss(xmm0, xmm0, xmm1, 11)),
        op!("vcmpss_imm12", false, vcmpss(xmm0, xmm0, xmm1, 12)),
        op!("vcmpss_imm13", false, vcmpss(xmm0, xmm0, xmm1, 13)),
        op!("vcmpss_imm14", false, vcmpss(xmm0, xmm0, xmm1, 14)),
        op!("vcmpss_imm15", false, vcmpss(xmm0, xmm0, xmm1, 15)),
        op!("vcmpss_imm16", false, vcmpss(xmm0, xmm0, xmm1, 16)),
        op!("vcmpss_imm17", false, vcmpss(xmm0, xmm0, xmm1, 17)),
        op!("vcmpss_imm18", false, vcmpss(xmm0, xmm0, xmm1, 18)),
        op!("vcmpss_imm19", false, vcmpss(xmm0, xmm0, xmm1, 19)),
        op!("vcmpss_imm20", false, vcmpss(xmm0, xmm0, xmm1, 20)),
        op!("vcmpss_imm21", false, vcmpss(xmm0, xmm0, xmm1, 21)),
        op!("vcmpss_imm22", false, vcmpss(xmm0, xmm0, xmm1, 22)),
        op!("vcmpss_imm23", false, vcmpss(xmm0, xmm0, xmm1, 23)),
        op!("vcmpss_imm24", false, vcmpss(xmm0, xmm0, xmm1, 24)),
        op!("vcmpss_imm25", false, vcmpss(xmm0, xmm0, xmm1, 25)),
        op!("vcmpss_imm26", false, vcmpss(xmm0, xmm0, xmm1, 26)),
        op!("vcmpss_imm27", false, vcmpss(xmm0, xmm0, xmm1, 27)),
        op!("vcmpss_imm28", false, vcmpss(xmm0, xmm0, xmm1, 28)),
        op!("vcmpss_imm29", false, vcmpss(xmm0, xmm0, xmm1, 29)),
        op!("vcmpss_imm30", false, vcmpss(xmm0, xmm0, xmm1, 30)),
        op!("vcmpss_imm31", false, vcmpss(xmm0, xmm0, xmm1, 31)),
        op!("vsqrtss_mem", false, vsqrtss(xmm0, xmm2, dword_ptr(rax))),
        op!("vrcpss_mem", false, vrcpss(xmm0, xmm2, dword_ptr(rax))),
        op!("vrsqrtss_mem", false, vrsqrtss(xmm0, xmm2, dword_ptr(rax))),
        op!("cmpsd_imm0", true, cmpsd_3(xmm0, xmm1, 0)),
        op!("vcmpsd_imm0", true, vcmpsd(xmm0, xmm0, xmm1, 0)),
        op!("cmpsd_imm1", true, cmpsd_3(xmm0, xmm1, 1)),
        op!("vcmpsd_imm1", true, vcmpsd(xmm0, xmm0, xmm1, 1)),
        op!("cmpsd_imm2", true, cmpsd_3(xmm0, xmm1, 2)),
        op!("vcmpsd_imm2", true, vcmpsd(xmm0, xmm0, xmm1, 2)),
        op!("cmpsd_imm3", true, cmpsd_3(xmm0, xmm1, 3)),
        op!("vcmpsd_imm3", true, vcmpsd(xmm0, xmm0, xmm1, 3)),
        op!("cmpsd_imm4", true, cmpsd_3(xmm0, xmm1, 4)),
        op!("vcmpsd_imm4", true, vcmpsd(xmm0, xmm0, xmm1, 4)),
        op!("cmpsd_imm5", true, cmpsd_3(xmm0, xmm1, 5)),
        op!("vcmpsd_imm5", true, vcmpsd(xmm0, xmm0, xmm1, 5)),
        op!("cmpsd_imm6", true, cmpsd_3(xmm0, xmm1, 6)),
        op!("vcmpsd_imm6", true, vcmpsd(xmm0, xmm0, xmm1, 6)),
        op!("cmpsd_imm7", true, cmpsd_3(xmm0, xmm1, 7)),
        op!("vcmpsd_imm7", true, vcmpsd(xmm0, xmm0, xmm1, 7)),
        op!("vcmpsd_imm8", true, vcmpsd(xmm0, xmm0, xmm1, 8)),
        op!("vcmpsd_imm9", true, vcmpsd(xmm0, xmm0, xmm1, 9)),
        op!("vcmpsd_imm10", true, vcmpsd(xmm0, xmm0, xmm1, 10)),
        op!("vcmpsd_imm11", true, vcmpsd(xmm0, xmm0, xmm1, 11)),
        op!("vcmpsd_imm12", true, vcmpsd(xmm0, xmm0, xmm1, 12)),
        op!("vcmpsd_imm13", true, vcmpsd(xmm0, xmm0, xmm1, 13)),
        op!("vcmpsd_imm14", true, vcmpsd(xmm0, xmm0, xmm1, 14)),
        op!("vcmpsd_imm15", true, vcmpsd(xmm0, xmm0, xmm1, 15)),
        op!("vcmpsd_imm16", true, vcmpsd(xmm0, xmm0, xmm1, 16)),
        op!("vcmpsd_imm17", true, vcmpsd(xmm0, xmm0, xmm1, 17)),
        op!("vcmpsd_imm18", true, vcmpsd(xmm0, xmm0, xmm1, 18)),
        op!("vcmpsd_imm19", true, vcmpsd(xmm0, xmm0, xmm1, 19)),
        op!("vcmpsd_imm20", true, vcmpsd(xmm0, xmm0, xmm1, 20)),
        op!("vcmpsd_imm21", true, vcmpsd(xmm0, xmm0, xmm1, 21)),
        op!("vcmpsd_imm22", true, vcmpsd(xmm0, xmm0, xmm1, 22)),
        op!("vcmpsd_imm23", true, vcmpsd(xmm0, xmm0, xmm1, 23)),
        op!("vcmpsd_imm24", true, vcmpsd(xmm0, xmm0, xmm1, 24)),
        op!("vcmpsd_imm25", true, vcmpsd(xmm0, xmm0, xmm1, 25)),
        op!("vcmpsd_imm26", true, vcmpsd(xmm0, xmm0, xmm1, 26)),
        op!("vcmpsd_imm27", true, vcmpsd(xmm0, xmm0, xmm1, 27)),
        op!("vcmpsd_imm28", true, vcmpsd(xmm0, xmm0, xmm1, 28)),
        op!("vcmpsd_imm29", true, vcmpsd(xmm0, xmm0, xmm1, 29)),
        op!("vcmpsd_imm30", true, vcmpsd(xmm0, xmm0, xmm1, 30)),
        op!("vcmpsd_imm31", true, vcmpsd(xmm0, xmm0, xmm1, 31)),
        op!("vsqrtsd_mem", true, vsqrtsd(xmm0, xmm2, qword_ptr(rax))),
        op!("cmpps_imm0", false, cmpps(xmm0, xmm1, 0)),
        op!("vcmpps_imm0", false, vcmpps(xmm0, xmm0, xmm1, 0)),
        op!("vcmpps_ymm0", false, vcmpps(ymm0, ymm0, ymm1, 0)),
        op!("cmpps_imm1", false, cmpps(xmm0, xmm1, 1)),
        op!("vcmpps_imm1", false, vcmpps(xmm0, xmm0, xmm1, 1)),
        op!("vcmpps_ymm1", false, vcmpps(ymm0, ymm0, ymm1, 1)),
        op!("cmpps_imm2", false, cmpps(xmm0, xmm1, 2)),
        op!("vcmpps_imm2", false, vcmpps(xmm0, xmm0, xmm1, 2)),
        op!("vcmpps_ymm2", false, vcmpps(ymm0, ymm0, ymm1, 2)),
        op!("cmpps_imm3", false, cmpps(xmm0, xmm1, 3)),
        op!("vcmpps_imm3", false, vcmpps(xmm0, xmm0, xmm1, 3)),
        op!("vcmpps_ymm3", false, vcmpps(ymm0, ymm0, ymm1, 3)),
        op!("cmpps_imm4", false, cmpps(xmm0, xmm1, 4)),
        op!("vcmpps_imm4", false, vcmpps(xmm0, xmm0, xmm1, 4)),
        op!("vcmpps_ymm4", false, vcmpps(ymm0, ymm0, ymm1, 4)),
        op!("cmpps_imm5", false, cmpps(xmm0, xmm1, 5)),
        op!("vcmpps_imm5", false, vcmpps(xmm0, xmm0, xmm1, 5)),
        op!("vcmpps_ymm5", false, vcmpps(ymm0, ymm0, ymm1, 5)),
        op!("cmpps_imm6", false, cmpps(xmm0, xmm1, 6)),
        op!("vcmpps_imm6", false, vcmpps(xmm0, xmm0, xmm1, 6)),
        op!("vcmpps_ymm6", false, vcmpps(ymm0, ymm0, ymm1, 6)),
        op!("cmpps_imm7", false, cmpps(xmm0, xmm1, 7)),
        op!("vcmpps_imm7", false, vcmpps(xmm0, xmm0, xmm1, 7)),
        op!("vcmpps_ymm7", false, vcmpps(ymm0, ymm0, ymm1, 7)),
        op!("vcmpps_imm8", false, vcmpps(xmm0, xmm0, xmm1, 8)),
        op!("vcmpps_ymm8", false, vcmpps(ymm0, ymm0, ymm1, 8)),
        op!("vcmpps_imm9", false, vcmpps(xmm0, xmm0, xmm1, 9)),
        op!("vcmpps_ymm9", false, vcmpps(ymm0, ymm0, ymm1, 9)),
        op!("vcmpps_imm10", false, vcmpps(xmm0, xmm0, xmm1, 10)),
        op!("vcmpps_ymm10", false, vcmpps(ymm0, ymm0, ymm1, 10)),
        op!("vcmpps_imm11", false, vcmpps(xmm0, xmm0, xmm1, 11)),
        op!("vcmpps_ymm11", false, vcmpps(ymm0, ymm0, ymm1, 11)),
        op!("vcmpps_imm12", false, vcmpps(xmm0, xmm0, xmm1, 12)),
        op!("vcmpps_ymm12", false, vcmpps(ymm0, ymm0, ymm1, 12)),
        op!("vcmpps_imm13", false, vcmpps(xmm0, xmm0, xmm1, 13)),
        op!("vcmpps_ymm13", false, vcmpps(ymm0, ymm0, ymm1, 13)),
        op!("vcmpps_imm14", false, vcmpps(xmm0, xmm0, xmm1, 14)),
        op!("vcmpps_ymm14", false, vcmpps(ymm0, ymm0, ymm1, 14)),
        op!("vcmpps_imm15", false, vcmpps(xmm0, xmm0, xmm1, 15)),
        op!("vcmpps_ymm15", false, vcmpps(ymm0, ymm0, ymm1, 15)),
        op!("vcmpps_imm16", false, vcmpps(xmm0, xmm0, xmm1, 16)),
        op!("vcmpps_ymm16", false, vcmpps(ymm0, ymm0, ymm1, 16)),
        op!("vcmpps_imm17", false, vcmpps(xmm0, xmm0, xmm1, 17)),
        op!("vcmpps_ymm17", false, vcmpps(ymm0, ymm0, ymm1, 17)),
        op!("vcmpps_imm18", false, vcmpps(xmm0, xmm0, xmm1, 18)),
        op!("vcmpps_ymm18", false, vcmpps(ymm0, ymm0, ymm1, 18)),
        op!("vcmpps_imm19", false, vcmpps(xmm0, xmm0, xmm1, 19)),
        op!("vcmpps_ymm19", false, vcmpps(ymm0, ymm0, ymm1, 19)),
        op!("vcmpps_imm20", false, vcmpps(xmm0, xmm0, xmm1, 20)),
        op!("vcmpps_ymm20", false, vcmpps(ymm0, ymm0, ymm1, 20)),
        op!("vcmpps_imm21", false, vcmpps(xmm0, xmm0, xmm1, 21)),
        op!("vcmpps_ymm21", false, vcmpps(ymm0, ymm0, ymm1, 21)),
        op!("vcmpps_imm22", false, vcmpps(xmm0, xmm0, xmm1, 22)),
        op!("vcmpps_ymm22", false, vcmpps(ymm0, ymm0, ymm1, 22)),
        op!("vcmpps_imm23", false, vcmpps(xmm0, xmm0, xmm1, 23)),
        op!("vcmpps_ymm23", false, vcmpps(ymm0, ymm0, ymm1, 23)),
        op!("vcmpps_imm24", false, vcmpps(xmm0, xmm0, xmm1, 24)),
        op!("vcmpps_ymm24", false, vcmpps(ymm0, ymm0, ymm1, 24)),
        op!("vcmpps_imm25", false, vcmpps(xmm0, xmm0, xmm1, 25)),
        op!("vcmpps_ymm25", false, vcmpps(ymm0, ymm0, ymm1, 25)),
        op!("vcmpps_imm26", false, vcmpps(xmm0, xmm0, xmm1, 26)),
        op!("vcmpps_ymm26", false, vcmpps(ymm0, ymm0, ymm1, 26)),
        op!("vcmpps_imm27", false, vcmpps(xmm0, xmm0, xmm1, 27)),
        op!("vcmpps_ymm27", false, vcmpps(ymm0, ymm0, ymm1, 27)),
        op!("vcmpps_imm28", false, vcmpps(xmm0, xmm0, xmm1, 28)),
        op!("vcmpps_ymm28", false, vcmpps(ymm0, ymm0, ymm1, 28)),
        op!("vcmpps_imm29", false, vcmpps(xmm0, xmm0, xmm1, 29)),
        op!("vcmpps_ymm29", false, vcmpps(ymm0, ymm0, ymm1, 29)),
        op!("vcmpps_imm30", false, vcmpps(xmm0, xmm0, xmm1, 30)),
        op!("vcmpps_ymm30", false, vcmpps(ymm0, ymm0, ymm1, 30)),
        op!("vcmpps_imm31", false, vcmpps(xmm0, xmm0, xmm1, 31)),
        op!("vcmpps_ymm31", false, vcmpps(ymm0, ymm0, ymm1, 31)),
        op!(
            "vaddps_ymm_mem",
            false,
            vaddps(ymm0, ymm0, ymmword_ptr(rax))
        ),
        op!(
            "vsubps_ymm_mem",
            false,
            vsubps(ymm0, ymm0, ymmword_ptr(rax))
        ),
        op!(
            "vmulps_ymm_mem",
            false,
            vmulps(ymm0, ymm0, ymmword_ptr(rax))
        ),
        op!(
            "vdivps_ymm_mem",
            false,
            vdivps(ymm0, ymm0, ymmword_ptr(rax))
        ),
        op!(
            "vminps_ymm_mem",
            false,
            vminps(ymm0, ymm0, ymmword_ptr(rax))
        ),
        op!(
            "vmaxps_ymm_mem",
            false,
            vmaxps(ymm0, ymm0, ymmword_ptr(rax))
        ),
        op!("cmppd_imm0", true, cmppd(xmm0, xmm1, 0)),
        op!("vcmppd_imm0", true, vcmppd(xmm0, xmm0, xmm1, 0)),
        op!("vcmppd_ymm0", true, vcmppd(ymm0, ymm0, ymm1, 0)),
        op!("cmppd_imm1", true, cmppd(xmm0, xmm1, 1)),
        op!("vcmppd_imm1", true, vcmppd(xmm0, xmm0, xmm1, 1)),
        op!("vcmppd_ymm1", true, vcmppd(ymm0, ymm0, ymm1, 1)),
        op!("cmppd_imm2", true, cmppd(xmm0, xmm1, 2)),
        op!("vcmppd_imm2", true, vcmppd(xmm0, xmm0, xmm1, 2)),
        op!("vcmppd_ymm2", true, vcmppd(ymm0, ymm0, ymm1, 2)),
        op!("cmppd_imm3", true, cmppd(xmm0, xmm1, 3)),
        op!("vcmppd_imm3", true, vcmppd(xmm0, xmm0, xmm1, 3)),
        op!("vcmppd_ymm3", true, vcmppd(ymm0, ymm0, ymm1, 3)),
        op!("cmppd_imm4", true, cmppd(xmm0, xmm1, 4)),
        op!("vcmppd_imm4", true, vcmppd(xmm0, xmm0, xmm1, 4)),
        op!("vcmppd_ymm4", true, vcmppd(ymm0, ymm0, ymm1, 4)),
        op!("cmppd_imm5", true, cmppd(xmm0, xmm1, 5)),
        op!("vcmppd_imm5", true, vcmppd(xmm0, xmm0, xmm1, 5)),
        op!("vcmppd_ymm5", true, vcmppd(ymm0, ymm0, ymm1, 5)),
        op!("cmppd_imm6", true, cmppd(xmm0, xmm1, 6)),
        op!("vcmppd_imm6", true, vcmppd(xmm0, xmm0, xmm1, 6)),
        op!("vcmppd_ymm6", true, vcmppd(ymm0, ymm0, ymm1, 6)),
        op!("cmppd_imm7", true, cmppd(xmm0, xmm1, 7)),
        op!("vcmppd_imm7", true, vcmppd(xmm0, xmm0, xmm1, 7)),
        op!("vcmppd_ymm7", true, vcmppd(ymm0, ymm0, ymm1, 7)),
        op!("vcmppd_imm8", true, vcmppd(xmm0, xmm0, xmm1, 8)),
        op!("vcmppd_ymm8", true, vcmppd(ymm0, ymm0, ymm1, 8)),
        op!("vcmppd_imm9", true, vcmppd(xmm0, xmm0, xmm1, 9)),
        op!("vcmppd_ymm9", true, vcmppd(ymm0, ymm0, ymm1, 9)),
        op!("vcmppd_imm10", true, vcmppd(xmm0, xmm0, xmm1, 10)),
        op!("vcmppd_ymm10", true, vcmppd(ymm0, ymm0, ymm1, 10)),
        op!("vcmppd_imm11", true, vcmppd(xmm0, xmm0, xmm1, 11)),
        op!("vcmppd_ymm11", true, vcmppd(ymm0, ymm0, ymm1, 11)),
        op!("vcmppd_imm12", true, vcmppd(xmm0, xmm0, xmm1, 12)),
        op!("vcmppd_ymm12", true, vcmppd(ymm0, ymm0, ymm1, 12)),
        op!("vcmppd_imm13", true, vcmppd(xmm0, xmm0, xmm1, 13)),
        op!("vcmppd_ymm13", true, vcmppd(ymm0, ymm0, ymm1, 13)),
        op!("vcmppd_imm14", true, vcmppd(xmm0, xmm0, xmm1, 14)),
        op!("vcmppd_ymm14", true, vcmppd(ymm0, ymm0, ymm1, 14)),
        op!("vcmppd_imm15", true, vcmppd(xmm0, xmm0, xmm1, 15)),
        op!("vcmppd_ymm15", true, vcmppd(ymm0, ymm0, ymm1, 15)),
        op!("vcmppd_imm16", true, vcmppd(xmm0, xmm0, xmm1, 16)),
        op!("vcmppd_ymm16", true, vcmppd(ymm0, ymm0, ymm1, 16)),
        op!("vcmppd_imm17", true, vcmppd(xmm0, xmm0, xmm1, 17)),
        op!("vcmppd_ymm17", true, vcmppd(ymm0, ymm0, ymm1, 17)),
        op!("vcmppd_imm18", true, vcmppd(xmm0, xmm0, xmm1, 18)),
        op!("vcmppd_ymm18", true, vcmppd(ymm0, ymm0, ymm1, 18)),
        op!("vcmppd_imm19", true, vcmppd(xmm0, xmm0, xmm1, 19)),
        op!("vcmppd_ymm19", true, vcmppd(ymm0, ymm0, ymm1, 19)),
        op!("vcmppd_imm20", true, vcmppd(xmm0, xmm0, xmm1, 20)),
        op!("vcmppd_ymm20", true, vcmppd(ymm0, ymm0, ymm1, 20)),
        op!("vcmppd_imm21", true, vcmppd(xmm0, xmm0, xmm1, 21)),
        op!("vcmppd_ymm21", true, vcmppd(ymm0, ymm0, ymm1, 21)),
        op!("vcmppd_imm22", true, vcmppd(xmm0, xmm0, xmm1, 22)),
        op!("vcmppd_ymm22", true, vcmppd(ymm0, ymm0, ymm1, 22)),
        op!("vcmppd_imm23", true, vcmppd(xmm0, xmm0, xmm1, 23)),
        op!("vcmppd_ymm23", true, vcmppd(ymm0, ymm0, ymm1, 23)),
        op!("vcmppd_imm24", true, vcmppd(xmm0, xmm0, xmm1, 24)),
        op!("vcmppd_ymm24", true, vcmppd(ymm0, ymm0, ymm1, 24)),
        op!("vcmppd_imm25", true, vcmppd(xmm0, xmm0, xmm1, 25)),
        op!("vcmppd_ymm25", true, vcmppd(ymm0, ymm0, ymm1, 25)),
        op!("vcmppd_imm26", true, vcmppd(xmm0, xmm0, xmm1, 26)),
        op!("vcmppd_ymm26", true, vcmppd(ymm0, ymm0, ymm1, 26)),
        op!("vcmppd_imm27", true, vcmppd(xmm0, xmm0, xmm1, 27)),
        op!("vcmppd_ymm27", true, vcmppd(ymm0, ymm0, ymm1, 27)),
        op!("vcmppd_imm28", true, vcmppd(xmm0, xmm0, xmm1, 28)),
        op!("vcmppd_ymm28", true, vcmppd(ymm0, ymm0, ymm1, 28)),
        op!("vcmppd_imm29", true, vcmppd(xmm0, xmm0, xmm1, 29)),
        op!("vcmppd_ymm29", true, vcmppd(ymm0, ymm0, ymm1, 29)),
        op!("vcmppd_imm30", true, vcmppd(xmm0, xmm0, xmm1, 30)),
        op!("vcmppd_ymm30", true, vcmppd(ymm0, ymm0, ymm1, 30)),
        op!("vcmppd_imm31", true, vcmppd(xmm0, xmm0, xmm1, 31)),
        op!("vcmppd_ymm31", true, vcmppd(ymm0, ymm0, ymm1, 31)),
        op!("vaddpd_ymm_mem", true, vaddpd(ymm0, ymm0, ymmword_ptr(rax))),
        op!("vsubpd_ymm_mem", true, vsubpd(ymm0, ymm0, ymmword_ptr(rax))),
        op!("vmulpd_ymm_mem", true, vmulpd(ymm0, ymm0, ymmword_ptr(rax))),
        op!("vdivpd_ymm_mem", true, vdivpd(ymm0, ymm0, ymmword_ptr(rax))),
        op!("vminpd_ymm_mem", true, vminpd(ymm0, ymm0, ymmword_ptr(rax))),
        op!("vmaxpd_ymm_mem", true, vmaxpd(ymm0, ymm0, ymmword_ptr(rax))),
        op!("comiss", false, comiss(xmm0, xmm1)),
        op!("ucomiss", false, ucomiss(xmm0, xmm1)),
        op!("comisd", true, comisd(xmm0, xmm1)),
        op!("ucomisd", true, ucomisd(xmm0, xmm1)),
        op!("vcvtps2pd_ymm", false, vcvtps2pd(ymm0, xmm1)),
        op!("vcvtdq2pd_ymm", false, vcvtdq2pd(ymm0, xmm1)),
        op!("vcvtpd2ps_ymm", true, vcvtpd2ps(xmm0, ymm1)),
        op!("vcvtpd2dq_ymm", true, vcvtpd2dq(xmm0, ymm1)),
        op!("vcvttpd2dq_ymm", true, vcvttpd2dq(xmm0, ymm1)),
        op!("vcvtdq2ps_ymm", false, vcvtdq2ps(ymm0, ymm1)),
        op!("vcvtps2dq_ymm", false, vcvtps2dq(ymm0, ymm1)),
        op!("vcvttps2dq_ymm", false, vcvttps2dq(ymm0, ymm1)),
        op!("addss", false, addss(xmm0, xmm1)),
        op!("addss_mem", false, addss(xmm0, dword_ptr(rax))),
        op!("vaddss", false, vaddss(xmm0, xmm0, xmm1)),
        op!("vaddss_mem", false, vaddss(xmm0, xmm0, dword_ptr(rax))),
        op!("subss", false, subss(xmm0, xmm1)),
        op!("subss_mem", false, subss(xmm0, dword_ptr(rax))),
        op!("vsubss", false, vsubss(xmm0, xmm0, xmm1)),
        op!("vsubss_mem", false, vsubss(xmm0, xmm0, dword_ptr(rax))),
        op!("mulss", false, mulss(xmm0, xmm1)),
        op!("mulss_mem", false, mulss(xmm0, dword_ptr(rax))),
        op!("vmulss", false, vmulss(xmm0, xmm0, xmm1)),
        op!("vmulss_mem", false, vmulss(xmm0, xmm0, dword_ptr(rax))),
        op!("divss", false, divss(xmm0, xmm1)),
        op!("divss_mem", false, divss(xmm0, dword_ptr(rax))),
        op!("vdivss", false, vdivss(xmm0, xmm0, xmm1)),
        op!("vdivss_mem", false, vdivss(xmm0, xmm0, dword_ptr(rax))),
        op!("minss", false, minss(xmm0, xmm1)),
        op!("minss_mem", false, minss(xmm0, dword_ptr(rax))),
        op!("vminss", false, vminss(xmm0, xmm0, xmm1)),
        op!("vminss_mem", false, vminss(xmm0, xmm0, dword_ptr(rax))),
        op!("maxss", false, maxss(xmm0, xmm1)),
        op!("maxss_mem", false, maxss(xmm0, dword_ptr(rax))),
        op!("vmaxss", false, vmaxss(xmm0, xmm0, xmm1)),
        op!("vmaxss_mem", false, vmaxss(xmm0, xmm0, dword_ptr(rax))),
        op!("sqrtss", false, sqrtss(xmm0, xmm1)),
        op!("sqrtss_mem", false, sqrtss(xmm0, dword_ptr(rax))),
        op!("vsqrtss", false, vsqrtss(xmm0, xmm2, xmm1)),
        op!("vrcpss", false, vrcpss(xmm0, xmm2, xmm1)),
        op!("vrsqrtss", false, vrsqrtss(xmm0, xmm2, xmm1)),
        op!("roundss0", false, roundss(xmm0, xmm1, 0)),
        op!("vroundss0", false, vroundss(xmm0, xmm2, xmm1, 0)),
        op!("roundss1", false, roundss(xmm0, xmm1, 1)),
        op!("vroundss1", false, vroundss(xmm0, xmm2, xmm1, 1)),
        op!("roundss2", false, roundss(xmm0, xmm1, 2)),
        op!("vroundss2", false, vroundss(xmm0, xmm2, xmm1, 2)),
        op!("roundss3", false, roundss(xmm0, xmm1, 3)),
        op!("vroundss3", false, vroundss(xmm0, xmm2, xmm1, 3)),
        op!("vfmadd132ss", false, vfmadd132ss(xmm0, xmm2, xmm1)),
        op!("vfmadd213ss", false, vfmadd213ss(xmm0, xmm2, xmm1)),
        op!("vfmadd231ss", false, vfmadd231ss(xmm0, xmm2, xmm1)),
        op!("vfmsub132ss", false, vfmsub132ss(xmm0, xmm2, xmm1)),
        op!("vfmsub213ss", false, vfmsub213ss(xmm0, xmm2, xmm1)),
        op!("vfmsub231ss", false, vfmsub231ss(xmm0, xmm2, xmm1)),
        op!("vfnmadd132ss", false, vfnmadd132ss(xmm0, xmm2, xmm1)),
        op!("vfnmadd213ss", false, vfnmadd213ss(xmm0, xmm2, xmm1)),
        op!("vfnmadd231ss", false, vfnmadd231ss(xmm0, xmm2, xmm1)),
        op!("vfnmsub132ss", false, vfnmsub132ss(xmm0, xmm2, xmm1)),
        op!("vfnmsub213ss", false, vfnmsub213ss(xmm0, xmm2, xmm1)),
        op!("vfnmsub231ss", false, vfnmsub231ss(xmm0, xmm2, xmm1)),
        op!("addsd", true, addsd(xmm0, xmm1)),
        op!("addsd_mem", true, addsd(xmm0, qword_ptr(rax))),
        op!("vaddsd", true, vaddsd(xmm0, xmm0, xmm1)),
        op!("vaddsd_mem", true, vaddsd(xmm0, xmm0, qword_ptr(rax))),
        op!("subsd", true, subsd(xmm0, xmm1)),
        op!("subsd_mem", true, subsd(xmm0, qword_ptr(rax))),
        op!("vsubsd", true, vsubsd(xmm0, xmm0, xmm1)),
        op!("vsubsd_mem", true, vsubsd(xmm0, xmm0, qword_ptr(rax))),
        op!("mulsd", true, mulsd(xmm0, xmm1)),
        op!("mulsd_mem", true, mulsd(xmm0, qword_ptr(rax))),
        op!("vmulsd", true, vmulsd(xmm0, xmm0, xmm1)),
        op!("vmulsd_mem", true, vmulsd(xmm0, xmm0, qword_ptr(rax))),
        op!("divsd", true, divsd(xmm0, xmm1)),
        op!("divsd_mem", true, divsd(xmm0, qword_ptr(rax))),
        op!("vdivsd", true, vdivsd(xmm0, xmm0, xmm1)),
        op!("vdivsd_mem", true, vdivsd(xmm0, xmm0, qword_ptr(rax))),
        op!("minsd", true, minsd(xmm0, xmm1)),
        op!("minsd_mem", true, minsd(xmm0, qword_ptr(rax))),
        op!("vminsd", true, vminsd(xmm0, xmm0, xmm1)),
        op!("vminsd_mem", true, vminsd(xmm0, xmm0, qword_ptr(rax))),
        op!("maxsd", true, maxsd(xmm0, xmm1)),
        op!("maxsd_mem", true, maxsd(xmm0, qword_ptr(rax))),
        op!("vmaxsd", true, vmaxsd(xmm0, xmm0, xmm1)),
        op!("vmaxsd_mem", true, vmaxsd(xmm0, xmm0, qword_ptr(rax))),
        op!("sqrtsd", true, sqrtsd(xmm0, xmm1)),
        op!("sqrtsd_mem", true, sqrtsd(xmm0, qword_ptr(rax))),
        op!("vsqrtsd", true, vsqrtsd(xmm0, xmm2, xmm1)),
        op!("roundsd0", true, roundsd(xmm0, xmm1, 0)),
        op!("vroundsd0", true, vroundsd(xmm0, xmm2, xmm1, 0)),
        op!("roundsd1", true, roundsd(xmm0, xmm1, 1)),
        op!("vroundsd1", true, vroundsd(xmm0, xmm2, xmm1, 1)),
        op!("roundsd2", true, roundsd(xmm0, xmm1, 2)),
        op!("vroundsd2", true, vroundsd(xmm0, xmm2, xmm1, 2)),
        op!("roundsd3", true, roundsd(xmm0, xmm1, 3)),
        op!("vroundsd3", true, vroundsd(xmm0, xmm2, xmm1, 3)),
        op!("vfmadd132sd", true, vfmadd132sd(xmm0, xmm2, xmm1)),
        op!("vfmadd213sd", true, vfmadd213sd(xmm0, xmm2, xmm1)),
        op!("vfmadd231sd", true, vfmadd231sd(xmm0, xmm2, xmm1)),
        op!("vfmsub132sd", true, vfmsub132sd(xmm0, xmm2, xmm1)),
        op!("vfmsub213sd", true, vfmsub213sd(xmm0, xmm2, xmm1)),
        op!("vfmsub231sd", true, vfmsub231sd(xmm0, xmm2, xmm1)),
        op!("vfnmadd132sd", true, vfnmadd132sd(xmm0, xmm2, xmm1)),
        op!("vfnmadd213sd", true, vfnmadd213sd(xmm0, xmm2, xmm1)),
        op!("vfnmadd231sd", true, vfnmadd231sd(xmm0, xmm2, xmm1)),
        op!("vfnmsub132sd", true, vfnmsub132sd(xmm0, xmm2, xmm1)),
        op!("vfnmsub213sd", true, vfnmsub213sd(xmm0, xmm2, xmm1)),
        op!("vfnmsub231sd", true, vfnmsub231sd(xmm0, xmm2, xmm1)),
        op!("addps", false, addps(xmm0, xmm1)),
        op!("addps_mem", false, addps(xmm0, xmmword_ptr(rax))),
        op!("vaddps", false, vaddps(xmm0, xmm0, xmm1)),
        op!("vaddps_mem", false, vaddps(xmm0, xmm0, xmmword_ptr(rax))),
        op!("vaddps_ymm", false, vaddps(ymm0, ymm0, ymm1)),
        op!("subps", false, subps(xmm0, xmm1)),
        op!("subps_mem", false, subps(xmm0, xmmword_ptr(rax))),
        op!("vsubps", false, vsubps(xmm0, xmm0, xmm1)),
        op!("vsubps_mem", false, vsubps(xmm0, xmm0, xmmword_ptr(rax))),
        op!("vsubps_ymm", false, vsubps(ymm0, ymm0, ymm1)),
        op!("mulps", false, mulps(xmm0, xmm1)),
        op!("mulps_mem", false, mulps(xmm0, xmmword_ptr(rax))),
        op!("vmulps", false, vmulps(xmm0, xmm0, xmm1)),
        op!("vmulps_mem", false, vmulps(xmm0, xmm0, xmmword_ptr(rax))),
        op!("vmulps_ymm", false, vmulps(ymm0, ymm0, ymm1)),
        op!("divps", false, divps(xmm0, xmm1)),
        op!("divps_mem", false, divps(xmm0, xmmword_ptr(rax))),
        op!("vdivps", false, vdivps(xmm0, xmm0, xmm1)),
        op!("vdivps_mem", false, vdivps(xmm0, xmm0, xmmword_ptr(rax))),
        op!("vdivps_ymm", false, vdivps(ymm0, ymm0, ymm1)),
        op!("minps", false, minps(xmm0, xmm1)),
        op!("minps_mem", false, minps(xmm0, xmmword_ptr(rax))),
        op!("vminps", false, vminps(xmm0, xmm0, xmm1)),
        op!("vminps_mem", false, vminps(xmm0, xmm0, xmmword_ptr(rax))),
        op!("vminps_ymm", false, vminps(ymm0, ymm0, ymm1)),
        op!("maxps", false, maxps(xmm0, xmm1)),
        op!("maxps_mem", false, maxps(xmm0, xmmword_ptr(rax))),
        op!("vmaxps", false, vmaxps(xmm0, xmm0, xmm1)),
        op!("vmaxps_mem", false, vmaxps(xmm0, xmm0, xmmword_ptr(rax))),
        op!("vmaxps_ymm", false, vmaxps(ymm0, ymm0, ymm1)),
        op!("sqrtps", false, sqrtps(xmm0, xmm1)),
        op!("sqrtps_mem", false, sqrtps(xmm0, xmmword_ptr(rax))),
        op!("vsqrtps", false, vsqrtps(xmm0, xmm1)),
        op!("vsqrtps_ymm", false, vsqrtps(ymm0, ymm1)),
        op!("vsqrtps_ymm_mem", false, vsqrtps(ymm0, ymmword_ptr(rax))),
        op!("vrcpps", false, vrcpps(xmm0, xmm1)),
        op!("vrcpps_ymm", false, vrcpps(ymm0, ymm1)),
        op!("vrcpps_ymm_mem", false, vrcpps(ymm0, ymmword_ptr(rax))),
        op!("vrsqrtps", false, vrsqrtps(xmm0, xmm1)),
        op!("vrsqrtps_ymm", false, vrsqrtps(ymm0, ymm1)),
        op!("vrsqrtps_ymm_mem", false, vrsqrtps(ymm0, ymmword_ptr(rax))),
        op!("roundps0", false, roundps(xmm0, xmm1, 0)),
        op!("vroundps0", false, vroundps(ymm0, ymm1, 0)),
        op!("roundps1", false, roundps(xmm0, xmm1, 1)),
        op!("vroundps1", false, vroundps(ymm0, ymm1, 1)),
        op!("roundps2", false, roundps(xmm0, xmm1, 2)),
        op!("vroundps2", false, vroundps(ymm0, ymm1, 2)),
        op!("roundps3", false, roundps(xmm0, xmm1, 3)),
        op!("vroundps3", false, vroundps(ymm0, ymm1, 3)),
        op!("vfmadd132ps", false, vfmadd132ps(ymm0, ymm2, ymm1)),
        op!("vfmadd213ps", false, vfmadd213ps(ymm0, ymm2, ymm1)),
        op!("vfmadd231ps", false, vfmadd231ps(ymm0, ymm2, ymm1)),
        op!("vfmsub132ps", false, vfmsub132ps(ymm0, ymm2, ymm1)),
        op!("vfmsub213ps", false, vfmsub213ps(ymm0, ymm2, ymm1)),
        op!("vfmsub231ps", false, vfmsub231ps(ymm0, ymm2, ymm1)),
        op!("vfnmadd132ps", false, vfnmadd132ps(ymm0, ymm2, ymm1)),
        op!("vfnmadd213ps", false, vfnmadd213ps(ymm0, ymm2, ymm1)),
        op!("vfnmadd231ps", false, vfnmadd231ps(ymm0, ymm2, ymm1)),
        op!("vfnmsub132ps", false, vfnmsub132ps(ymm0, ymm2, ymm1)),
        op!("vfnmsub213ps", false, vfnmsub213ps(ymm0, ymm2, ymm1)),
        op!("vfnmsub231ps", false, vfnmsub231ps(ymm0, ymm2, ymm1)),
        op!("vfmaddsub132ps", false, vfmaddsub132ps(ymm0, ymm2, ymm1)),
        op!("vfmaddsub213ps", false, vfmaddsub213ps(ymm0, ymm2, ymm1)),
        op!("vfmaddsub231ps", false, vfmaddsub231ps(ymm0, ymm2, ymm1)),
        op!("vfmsubadd132ps", false, vfmsubadd132ps(ymm0, ymm2, ymm1)),
        op!("vfmsubadd213ps", false, vfmsubadd213ps(ymm0, ymm2, ymm1)),
        op!("vfmsubadd231ps", false, vfmsubadd231ps(ymm0, ymm2, ymm1)),
        op!("addpd", true, addpd(xmm0, xmm1)),
        op!("addpd_mem", true, addpd(xmm0, xmmword_ptr(rax))),
        op!("vaddpd", true, vaddpd(xmm0, xmm0, xmm1)),
        op!("vaddpd_mem", true, vaddpd(xmm0, xmm0, xmmword_ptr(rax))),
        op!("vaddpd_ymm", true, vaddpd(ymm0, ymm0, ymm1)),
        op!("subpd", true, subpd(xmm0, xmm1)),
        op!("subpd_mem", true, subpd(xmm0, xmmword_ptr(rax))),
        op!("vsubpd", true, vsubpd(xmm0, xmm0, xmm1)),
        op!("vsubpd_mem", true, vsubpd(xmm0, xmm0, xmmword_ptr(rax))),
        op!("vsubpd_ymm", true, vsubpd(ymm0, ymm0, ymm1)),
        op!("mulpd", true, mulpd(xmm0, xmm1)),
        op!("mulpd_mem", true, mulpd(xmm0, xmmword_ptr(rax))),
        op!("vmulpd", true, vmulpd(xmm0, xmm0, xmm1)),
        op!("vmulpd_mem", true, vmulpd(xmm0, xmm0, xmmword_ptr(rax))),
        op!("vmulpd_ymm", true, vmulpd(ymm0, ymm0, ymm1)),
        op!("divpd", true, divpd(xmm0, xmm1)),
        op!("divpd_mem", true, divpd(xmm0, xmmword_ptr(rax))),
        op!("vdivpd", true, vdivpd(xmm0, xmm0, xmm1)),
        op!("vdivpd_mem", true, vdivpd(xmm0, xmm0, xmmword_ptr(rax))),
        op!("vdivpd_ymm", true, vdivpd(ymm0, ymm0, ymm1)),
        op!("minpd", true, minpd(xmm0, xmm1)),
        op!("minpd_mem", true, minpd(xmm0, xmmword_ptr(rax))),
        op!("vminpd", true, vminpd(xmm0, xmm0, xmm1)),
        op!("vminpd_mem", true, vminpd(xmm0, xmm0, xmmword_ptr(rax))),
        op!("vminpd_ymm", true, vminpd(ymm0, ymm0, ymm1)),
        op!("maxpd", true, maxpd(xmm0, xmm1)),
        op!("maxpd_mem", true, maxpd(xmm0, xmmword_ptr(rax))),
        op!("vmaxpd", true, vmaxpd(xmm0, xmm0, xmm1)),
        op!("vmaxpd_mem", true, vmaxpd(xmm0, xmm0, xmmword_ptr(rax))),
        op!("vmaxpd_ymm", true, vmaxpd(ymm0, ymm0, ymm1)),
        op!("sqrtpd", true, sqrtpd(xmm0, xmm1)),
        op!("sqrtpd_mem", true, sqrtpd(xmm0, xmmword_ptr(rax))),
        op!("vsqrtpd", true, vsqrtpd(xmm0, xmm1)),
        op!("vsqrtpd_ymm", true, vsqrtpd(ymm0, ymm1)),
        op!("vsqrtpd_ymm_mem", true, vsqrtpd(ymm0, ymmword_ptr(rax))),
        op!("roundpd0", true, roundpd(xmm0, xmm1, 0)),
        op!("vroundpd0", true, vroundpd(ymm0, ymm1, 0)),
        op!("roundpd1", true, roundpd(xmm0, xmm1, 1)),
        op!("vroundpd1", true, vroundpd(ymm0, ymm1, 1)),
        op!("roundpd2", true, roundpd(xmm0, xmm1, 2)),
        op!("vroundpd2", true, vroundpd(ymm0, ymm1, 2)),
        op!("roundpd3", true, roundpd(xmm0, xmm1, 3)),
        op!("vroundpd3", true, vroundpd(ymm0, ymm1, 3)),
        op!("vfmadd132pd", true, vfmadd132pd(ymm0, ymm2, ymm1)),
        op!("vfmadd213pd", true, vfmadd213pd(ymm0, ymm2, ymm1)),
        op!("vfmadd231pd", true, vfmadd231pd(ymm0, ymm2, ymm1)),
        op!("vfmsub132pd", true, vfmsub132pd(ymm0, ymm2, ymm1)),
        op!("vfmsub213pd", true, vfmsub213pd(ymm0, ymm2, ymm1)),
        op!("vfmsub231pd", true, vfmsub231pd(ymm0, ymm2, ymm1)),
        op!("vfnmadd132pd", true, vfnmadd132pd(ymm0, ymm2, ymm1)),
        op!("vfnmadd213pd", true, vfnmadd213pd(ymm0, ymm2, ymm1)),
        op!("vfnmadd231pd", true, vfnmadd231pd(ymm0, ymm2, ymm1)),
        op!("vfnmsub132pd", true, vfnmsub132pd(ymm0, ymm2, ymm1)),
        op!("vfnmsub213pd", true, vfnmsub213pd(ymm0, ymm2, ymm1)),
        op!("vfnmsub231pd", true, vfnmsub231pd(ymm0, ymm2, ymm1)),
        op!("vfmaddsub132pd", true, vfmaddsub132pd(ymm0, ymm2, ymm1)),
        op!("vfmaddsub213pd", true, vfmaddsub213pd(ymm0, ymm2, ymm1)),
        op!("vfmaddsub231pd", true, vfmaddsub231pd(ymm0, ymm2, ymm1)),
        op!("vfmsubadd132pd", true, vfmsubadd132pd(ymm0, ymm2, ymm1)),
        op!("vfmsubadd213pd", true, vfmsubadd213pd(ymm0, ymm2, ymm1)),
        op!("vfmsubadd231pd", true, vfmsubadd231pd(ymm0, ymm2, ymm1)),
        op!("haddps", false, haddps(xmm0, xmm1)),
        op!("vhaddps_ymm", false, vhaddps(ymm0, ymm0, ymm1)),
        op!("vhaddps_mem", false, vhaddps(ymm0, ymm0, ymmword_ptr(rax))),
        op!("hsubps", false, hsubps(xmm0, xmm1)),
        op!("vhsubps_ymm", false, vhsubps(ymm0, ymm0, ymm1)),
        op!("vhsubps_mem", false, vhsubps(ymm0, ymm0, ymmword_ptr(rax))),
        op!("addsubps", false, addsubps(xmm0, xmm1)),
        op!("vaddsubps_ymm", false, vaddsubps(ymm0, ymm0, ymm1)),
        op!(
            "vaddsubps_mem",
            false,
            vaddsubps(ymm0, ymm0, ymmword_ptr(rax))
        ),
        op!("dpps", false, dpps(xmm0, xmm1, 0xff)),
        op!("vdpps", false, vdpps(xmm0, xmm0, xmm1, 0xff)),
        op!("haddpd", true, haddpd(xmm0, xmm1)),
        op!("vhaddpd_ymm", true, vhaddpd(ymm0, ymm0, ymm1)),
        op!("vhaddpd_mem", true, vhaddpd(ymm0, ymm0, ymmword_ptr(rax))),
        op!("hsubpd", true, hsubpd(xmm0, xmm1)),
        op!("vhsubpd_ymm", true, vhsubpd(ymm0, ymm0, ymm1)),
        op!("vhsubpd_mem", true, vhsubpd(ymm0, ymm0, ymmword_ptr(rax))),
        op!("addsubpd", true, addsubpd(xmm0, xmm1)),
        op!("vaddsubpd_ymm", true, vaddsubpd(ymm0, ymm0, ymm1)),
        op!(
            "vaddsubpd_mem",
            true,
            vaddsubpd(ymm0, ymm0, ymmword_ptr(rax))
        ),
        op!("dppd", true, dppd(xmm0, xmm1, 0xff)),
        op!("vdppd", true, vdppd(xmm0, xmm0, xmm1, 0xff)),
        op!("vdpps_ymm", false, vdpps(ymm0, ymm0, ymm1, 0xff)),
        op!("cvtps2pd", false, cvtps2pd(xmm0, xmm1)),
        op!("vcvtps2pd", false, vcvtps2pd(xmm0, xmm1)),
        op!("cvtpd2ps", true, cvtpd2ps(xmm0, xmm1)),
        op!("vcvtpd2ps", true, vcvtpd2ps(xmm0, xmm1)),
        op!("cvtdq2ps", false, cvtdq2ps(xmm0, xmm1)),
        op!("vcvtdq2ps", false, vcvtdq2ps(xmm0, xmm1)),
        op!("cvtdq2pd", false, cvtdq2pd(xmm0, xmm1)),
        op!("vcvtdq2pd", false, vcvtdq2pd(xmm0, xmm1)),
        op!("cvtps2dq", false, cvtps2dq(xmm0, xmm1)),
        op!("vcvtps2dq", false, vcvtps2dq(xmm0, xmm1)),
        op!("cvttps2dq", false, cvttps2dq(xmm0, xmm1)),
        op!("vcvttps2dq", false, vcvttps2dq(xmm0, xmm1)),
        op!("cvtpd2dq", true, cvtpd2dq(xmm0, xmm1)),
        op!("vcvtpd2dq", true, vcvtpd2dq(xmm0, xmm1)),
        op!("cvttpd2dq", true, cvttpd2dq(xmm0, xmm1)),
        op!("vcvttpd2dq", true, vcvttpd2dq(xmm0, xmm1)),
        op!("cvtss2sd", false, cvtss2sd(xmm0, xmm1)),
        op!("vcvtss2sd", false, vcvtss2sd(xmm0, xmm2, xmm1)),
        op!("cvtsd2ss", true, cvtsd2ss(xmm0, xmm1)),
        op!("vcvtsd2ss", true, vcvtsd2ss(xmm0, xmm2, xmm1)),
        op!("cvtss2sieax", false, cvtss2si(eax, xmm1)),
        op!("cvtss2sirax", false, cvtss2si(rax, xmm1)),
        op!("cvttss2sieax", false, cvttss2si(eax, xmm1)),
        op!("cvttss2sirax", false, cvttss2si(rax, xmm1)),
        op!("cvtsd2sieax", true, cvtsd2si(eax, xmm1)),
        op!("cvtsd2sirax", true, cvtsd2si(rax, xmm1)),
        op!("cvttsd2sieax", true, cvttsd2si(eax, xmm1)),
        op!("cvttsd2sirax", true, cvttsd2si(rax, xmm1)),
        op!("cvtsi2sseax", false, cvtsi2ss(xmm0, eax)),
        op!("cvtsi2ssrax", false, cvtsi2ss(xmm0, rax)),
        op!("cvtsi2sdeax", true, cvtsi2sd(xmm0, eax)),
        op!("cvtsi2sdrax", true, cvtsi2sd(xmm0, rax)),
        op!("vcvtph2ps", false, vcvtph2ps(ymm0, xmm1)),
        op!("vcvtps2ph0", false, vcvtps2ph(xmm0, ymm1, 0)),
        op!("vcvtps2ph1", false, vcvtps2ph(xmm0, ymm1, 1)),
        op!("vcvtps2ph2", false, vcvtps2ph(xmm0, ymm1, 2)),
        op!("vcvtps2ph3", false, vcvtps2ph(xmm0, ymm1, 3)),
    ]
}

fn random_bits(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn operand(seed: &mut u64, double: bool, sample: usize) -> [u128; 2] {
    let mut out = [0; 2];
    for half in &mut out {
        for lane in 0..if double { 2 } else { 4 } {
            let r = random_bits(seed);
            let sign = (r & 1) << if double { 63 } else { 31 };
            let (inf, quiet, frac, one) = if double {
                (
                    0x7ff0000000000000,
                    0x8000000000000,
                    0xfffffffffffff,
                    0x3ff0000000000000,
                )
            } else {
                (0x7f800000, 0x400000, 0x7fffff, 0x3f800000)
            };
            let bits = match (sample + lane) % 8 {
                0 => sign,
                1 => sign | inf,
                2 => sign | inf | quiet | (r & frac),
                3 => sign | inf | ((r & (quiet - 1)) | 1),
                4 => sign | ((r & frac) | 1),
                5 => sign | one,
                6 => sign | (frac + 1),
                _ => {
                    if double {
                        r
                    } else {
                        u64::from(r as u32)
                    }
                }
            };
            *half |= u128::from(bits) << (lane * if double { 64 } else { 32 });
        }
    }
    out
}

#[test]
fn fp_default_randomized_tiers() {
    // Strict bit comparison includes NaN signs/payloads. Use mixed lane classes
    // so a vector's exceptional lane cannot change its ordinary lanes.
    let mut seed = 0x1276_93ab_d057_ee41;
    let mut mismatches = Vec::new();
    let mut executions = 0;
    for (name, double, build) in recipes() {
        let machines: Vec<_> = (0..4).map(|tier| machine(tier, build)).collect();
        for sample in 0..96 {
            let x = operand(&mut seed, double, sample / 8);
            let y = operand(&mut seed, double, sample);
            let z = operand(&mut seed, double, sample / 3);
            let expected = execute(&machines[0], x, y, z);
            for (tier, vm) in machines.iter().enumerate().skip(1) {
                let actual = execute(vm, x, y, z);
                executions += 1;
                if expected != actual
                    && !mismatches
                        .iter()
                        .any(|m: &String| m.starts_with(&format!("{name} tier={tier} ")))
                {
                    mismatches.push(format!("{name} tier={tier} sample={sample} x={x:x?} y={y:x?} z={z:x?}\ninterp={expected:x?}\njit={actual:x?}"));
                }
            }
        }
        assert!(
            machines[2].cache.regions() > 0,
            "{name}: region must execute"
        );
    }
    eprintln!(
        "{executions} strict comparisons across {} instruction recipes",
        recipes().len()
    );
    assert!(
        mismatches.is_empty(),
        "{} mismatching op/tier pairs in {executions} comparisons:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

const ENTRY: u64 = 0x1000;
const DATA: u64 = 0x8000;

fn machine(tier: usize, build: impl FnOnce(&mut CodeAssembler)) -> Vm {
    let backend: Box<dyn Backend> = match tier {
        0 => Box::new(InterpreterBackend),
        1 => Box::new(JitBackend::new()),
        2 => Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 8,
            max_icount: 128,
        })),
        _ => Box::new(JitBackend::with_options(
            None,
            HostTarget::Baseline,
            OptLevel::None,
        )),
    };
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), backend);
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    let mut a = CodeAssembler::new(64).unwrap();
    let mut body = a.create_label();
    let mut done = a.create_label();
    a.mov(ecx, 1).unwrap();
    a.jmp(body).unwrap();
    a.set_label(&mut body).unwrap();
    build(&mut a);
    a.pushfq().unwrap();
    a.pop(rdx).unwrap();
    a.jmp(done).unwrap();
    a.set_label(&mut done).unwrap();
    a.dec(ecx).unwrap();
    a.jnz(body).unwrap();
    a.hlt().unwrap();
    vm.write_bytes(ENTRY, &a.assemble(ENTRY).unwrap()).unwrap();
    vm
}

fn execute(
    vm: &Vm,
    x: [u128; 2],
    y: [u128; 2],
    z: [u128; 2],
) -> ([u128; 16], [u128; 16], [u64; 16]) {
    vm.write_bytes(DATA, &y[0].to_le_bytes()).unwrap();
    vm.write_bytes(DATA + 16, &y[1].to_le_bytes()).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, ENTRY);
    cpu.set_reg(Reg::Rax, DATA);
    cpu.set_reg(Reg::Rsp, 0xf000);
    for (r, value) in [x, y, z].into_iter().enumerate() {
        cpu.cpu.xmm[r] = value[0];
        cpu.cpu.ymm_hi[r] = value[1];
    }
    let exit = cpu.run(vm, Some(16));
    assert!(matches!(exit, Exit::Hlt), "{exit:?}");
    (
        cpu.cpu.xmm[..16].try_into().unwrap(),
        cpu.cpu.ymm_hi[..16].try_into().unwrap(),
        cpu.cpu.gpr,
    )
}

fn splat(x: u64, double: bool) -> [u128; 2] {
    let v = if double {
        u128::from(x) | u128::from(x) << 64
    } else {
        let x = u128::from(x as u32);
        x | x << 32 | x << 64 | x << 96
    };
    [v; 2]
}

#[test]
fn fp_default_invalid_and_propagation_witness() {
    // SDM Vol 1 §§4.8.3.5 (Table 4-8), 4.8.3.7 (Table 4-3).
    for tier in 0..4 {
        for double in [false, true] {
            let (neg, inf, nan, snan, indefinite) = if double {
                (
                    0x8000000000000001,
                    0x7ff0000000000000,
                    0xfff8123456789abc,
                    0x7ff0000000000001,
                    0xfff8000000000000,
                )
            } else {
                (0x80000001, 0x7f800000, 0xffc12345, 0x7f800001, 0xffc00000)
            };
            for op in 0..5 {
                let vm = machine(tier, |a| match (op, double) {
                    (0, false) => a.sqrtps(xmm0, xmm0).unwrap(),
                    (1, false) => a.addps(xmm0, xmm1).unwrap(),
                    (2, false) => a.subps(xmm0, xmm1).unwrap(),
                    (3, false) => a.mulps(xmm0, xmm1).unwrap(),
                    (4, false) => a.divps(xmm0, xmm1).unwrap(),
                    (0, true) => a.sqrtpd(xmm0, xmm0).unwrap(),
                    (1, true) => a.addpd(xmm0, xmm1).unwrap(),
                    (2, true) => a.subpd(xmm0, xmm1).unwrap(),
                    (3, true) => a.mulpd(xmm0, xmm1).unwrap(),
                    (4, true) => a.divpd(xmm0, xmm1).unwrap(),
                    _ => unreachable!(),
                });
                let (x, y) = match op {
                    0 => (neg, 0),
                    1 => (inf, inf | neg & (1 << if double { 63 } else { 31 })),
                    2 | 4 => (inf, inf),
                    _ => (0, inf),
                };
                let out = execute(&vm, splat(x, double), splat(y, double), [0; 2]);
                assert_eq!(
                    out.0[0],
                    splat(indefinite, double)[0],
                    "tier={tier} double={double} op={op}"
                );
                let out = execute(&vm, splat(nan, double), splat(snan, double), [0; 2]);
                assert_eq!(
                    out.0[0],
                    splat(nan, double)[0],
                    "NaN priority tier={tier} op={op}"
                );
            }
        }
    }
}

#[test]
fn fp_estimate_special_values_witness() {
    // SDM Vol 2B §4.3 RCPPS/RCPSS/RSQRTPS/RSQRTSS: signed-zero
    // treatment of denormals, negative RSQRT indefinite, and RCP underflow.
    for tier in 0..4 {
        for rsqrt in [false, true] {
            let vm = machine(tier, |a| {
                if rsqrt {
                    a.vrsqrtps(xmm0, xmm1)
                } else {
                    a.vrcpps(xmm0, xmm1)
                }
                .unwrap();
            });
            for (x, expected) in [
                (0, 0x7f800000),
                (0x80000000, 0xff800000),
                (1, 0x7f800000),
                (0x007fffff, 0x7f800000),
                (0x80000001, 0xff800000),
                (0x807fffff, 0xff800000),
                (0x7f800000, 0),
                (0xff800000, if rsqrt { 0xffc00000 } else { 0x80000000 }),
                (0xbf800000, if rsqrt { 0xffc00000 } else { 0xbf800000 }),
                (0x7f800001, 0x7fc00001),
                (0xff812345, 0xffc12345),
                (0x7fc54321, 0x7fc54321),
            ]
            .into_iter()
            .chain((!rsqrt).then_some((0x7f7fffff, 0)))
            {
                let out = execute(&vm, [0; 2], splat(x, false), [0; 2]);
                assert_eq!(
                    out.0[0],
                    splat(expected, false)[0],
                    "tier={tier} rsqrt={rsqrt} x={x:x}"
                );
            }
        }
    }
}

#[test]
fn fp_half_nan_quieting_witness() {
    // SDM Vol 2C §5.2 VCVTPH2PS; Vol 1 §14.4, Table 14-9:
    // preserve NaN sign/payload and quiet SNaNs; half denormals convert exactly.
    for tier in 0..4 {
        let vm = machine(tier, |a| a.vcvtph2ps(ymm0, xmm1).unwrap());
        let halves = [0x7c01u16, 0xfc01, 0x7e55, 0xfe55, 0x7c00, 0xfc00, 1, 0x8001];
        let src = halves
            .iter()
            .enumerate()
            .fold(0, |v, (i, h)| v | u128::from(*h) << (16 * i));
        let out = execute(&vm, [0; 2], [src, 0], [0; 2]);
        let expected = [
            0x7fc02000u32,
            0xffc02000,
            0x7fcaa000,
            0xffcaa000,
            0x7f800000,
            0xff800000,
            0x33800000,
            0xb3800000,
        ];
        for (i, bits) in expected.into_iter().enumerate() {
            let lane = if i < 4 { out.0[0] } else { out.1[0] };
            assert_eq!(
                (lane >> (32 * (i % 4))) as u32,
                bits,
                "tier={tier} half={:04x}",
                halves[i]
            );
        }
    }
}

#[test]
fn fp_dot_product_nan_and_invalid_witness() {
    // SDM Vol 2A §3.3 DPPD/DPPS and Vol 1 §§4.8.3.5, 4.8.3.7.
    // Select one product so horizontal NaN priority (implementation dependent
    // for DPPS/DPPD) cannot affect the witness.
    for tier in 0..4 {
        for double in [false, true] {
            let vm = machine(tier, |a| {
                if double {
                    a.dppd(xmm0, xmm1, 0x11)
                } else {
                    a.dpps(xmm0, xmm1, 0x11)
                }
                .unwrap();
            });
            let (inf, qnan, snan, indefinite) = if double {
                (
                    0x7ff0000000000000,
                    0xfff8123456789abc,
                    0x7ff0000000000001,
                    0xfff8000000000000,
                )
            } else {
                (0x7f800000, 0xffc12345, 0x7f800001, 0xffc00000)
            };
            for (x, y, expected) in [(0, inf, indefinite), (qnan, snan, qnan)] {
                let out = execute(&vm, splat(x, double), splat(y, double), [0; 2]);
                assert_eq!(
                    out.0[0],
                    u128::from(expected),
                    "tier={tier} double={double}"
                );
            }
        }
    }
}
