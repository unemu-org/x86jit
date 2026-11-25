//! Opt-in runtime census. Run with `--ignored --nocapture`; writes CENSUS_386_OUT.
//! D8–DF probes attach `FpuModel::Present` (the I80386 387 pairing).
//! The opcode and group masks follow 80386 PRM Appendix A, pp. A-4–A-8.
//! Decoder metadata supplies lengths/labels, never the historical opcode membership.
#[path = "../support/descriptor_fixture.rs"]
#[allow(dead_code)]
mod descriptor_fixture;

use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic as M, OpKind};
use std::collections::BTreeMap;
use std::fmt::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use x86jit_core::lift::FetchAddr;
use x86jit_core::{
    CpuMode, CpuModel, Exit, FpuModel, Prot, RegionKind, SegmentCache, StepOptions, StepStop, Vcpu,
    Vm, VmConfig,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    V86,
    PagedV86,
    PagedP32,
    PagedP16,
    P32,
    P16,
    Real,
}
impl Mode {
    fn v86(self) -> bool {
        matches!(self, Self::V86 | Self::PagedV86)
    }
    fn paged(self) -> bool {
        matches!(self, Self::PagedP32 | Self::PagedP16 | Self::PagedV86)
    }
    fn bits(self) -> u32 {
        if matches!(self, Self::P32 | Self::PagedP32) {
            32
        } else {
            16
        }
    }
}
#[derive(Clone)]
struct Probe {
    bytes: Vec<u8>,
    insn: Instruction,
    shape: &'static str,
    x87: bool,
    reserved: bool,
}

fn modrm(two: bool, op: u8) -> bool {
    if two {
        !matches!(op, 0x06 | 0x08..=0x09 | 0x80..=0x8f | 0xa0..=0xa2 | 0xa8..=0xa9 | 0xc8..=0xcf)
    } else {
        (op < 0x40 && op & 7 <= 3)
            || matches!(op, 0x62..=0x63 | 0x69 | 0x6b | 0x80..=0x8f | 0xc0..=0xc1 | 0xc4..=0xc7 | 0xd0..=0xd3 | 0xd8..=0xdf | 0xf6..=0xf7 | 0xfe..=0xff)
    }
}
fn defined(model: CpuModel, two: bool, op: u8) -> bool {
    // Intel486 PRM App. A pp. A-4–7; Embedded 486 DM §D.1 p. D-1.
    if two
        && model.i486_signature().is_some()
        && (matches!(op, 0x08..=0x09 | 0xb0..=0xb1 | 0xc0..=0xc1 | 0xc8..=0xcf)
            || op == 0xa2 && model == CpuModel::I80486DxCpuid)
    {
        return true;
    }
    if two {
        matches!(op, 0x00..=0x03 | 0x06 | 0x20..=0x24 | 0x26 | 0x80..=0x9f | 0xa0..=0xa1 | 0xa3..=0xa5 | 0xa8..=0xa9 | 0xab..=0xad | 0xaf | 0xb2..=0xb7 | 0xba..=0xbf)
    } else {
        !matches!(op, 0x0f | 0x26 | 0x2e | 0x36 | 0x3e | 0x64..=0x67 | 0x82 | 0xd6 | 0xf0..=0xf3)
    }
}
fn group(two: bool, op: u8) -> bool {
    if two {
        matches!(op, 0x00 | 0x01 | 0xba)
    } else {
        matches!(
            op,
            0x80 | 0x81 | 0x83 | 0x8f | 0xc0 | 0xc1 | 0xc6 | 0xc7 | 0xd0
                ..=0xd3 | 0xf6 | 0xf7 | 0xfe | 0xff
        )
    }
}
fn legal(model: CpuModel, two: bool, op: u8, reg: u8, shape: u8) -> bool {
    // PRM Appendix A p. A-8; §17 MOV special-register encodings.
    if two {
        match op {
            0x00 => reg <= 5,
            0x01 => {
                (matches!(reg, 0..=4 | 6)
                    || reg == 7 && model.i486_signature().is_some() && shape != 0)
                    && (reg >= 4 || shape != 0)
            }
            0x20 | 0x22 => matches!(reg, 0 | 2 | 3) && shape == 0,
            0x21 | 0x23 => matches!(reg, 0..=3 | 6..=7) && shape == 0,
            0x24 | 0x26 => (reg >= 6 || model.i486_signature().is_some() && reg >= 3) && shape == 0,
            0xb2 | 0xb4 | 0xb5 => shape != 0,
            0xba => reg >= 4,
            _ => true,
        }
    } else {
        match op {
            0x62 | 0x8d | 0xc4 | 0xc5 => shape != 0,
            0x8c => reg <= 5,
            0x8e => matches!(reg, 0 | 2..=5),
            0x8f | 0xc6 | 0xc7 => reg == 0,
            0xc0..=0xc1 | 0xd0..=0xd3 => reg != 6,
            0xf6 | 0xf7 => reg != 1,
            0xfe => reg <= 1,
            0xff => reg <= 6 && (!matches!(reg, 3 | 5) || shape != 0),
            _ => true,
        }
    }
}
fn probes(model: CpuModel, mode: Mode) -> Vec<Probe> {
    let mut out = Vec::new();
    for two in [false, true] {
        for op in 0..=255u8 {
            if !defined(model, two, op) {
                continue;
            }
            let has_modrm = modrm(two, op);
            for p66 in [false, true] {
                for p67 in [false, true] {
                    let addr32 = (mode.bits() == 32) != p67;
                    let operand32 = (mode.bits() == 32) != p66;
                    let strings = !two && matches!(op, 0x6c..=0x6f | 0xa4..=0xa7 | 0xaa..=0xaf);
                    let reps: &[u8] = if strings { &[0, 0xf2, 0xf3] } else { &[0] };
                    for &rep in reps {
                        for reg in 0..if has_modrm { 8 } else { 1 } {
                            for shape in 0..if has_modrm { 3 } else { 1 } {
                                if shape == 2 && !addr32 {
                                    continue;
                                }
                                let reserved = group(two, op) && !legal(model, two, op, reg, 1);
                                if !legal(model, two, op, reg, shape) && !reserved {
                                    continue;
                                }
                                let mut b = Vec::new();
                                if p66 {
                                    b.push(0x66);
                                }
                                if p67 {
                                    b.push(0x67);
                                }
                                if rep != 0 {
                                    b.push(rep);
                                }
                                if two {
                                    b.push(0x0f);
                                }
                                b.push(op);
                                if has_modrm {
                                    match shape {
                                        0 => b.push(0xc3 | reg << 3),
                                        1 if addr32 => {
                                            b.push(5 | reg << 3);
                                            b.extend(0x6000u32.to_le_bytes());
                                        }
                                        1 => {
                                            b.push(6 | reg << 3);
                                            b.extend(0x6000u16.to_le_bytes());
                                        }
                                        _ => {
                                            b.push(4 | reg << 3);
                                            b.push(0x0d);
                                            b.extend(0x5fffu32.to_le_bytes());
                                        }
                                    }
                                }
                                if !two && matches!(op, 0xa0..=0xa3) {
                                    b.extend(if addr32 {
                                        0x6000u32.to_le_bytes().to_vec()
                                    } else {
                                        0x6000u16.to_le_bytes().to_vec()
                                    });
                                }
                                if !two && matches!(op, 0x9a | 0xea) {
                                    b.extend(if operand32 {
                                        0x1800u32.to_le_bytes().to_vec()
                                    } else {
                                        0x1800u16.to_le_bytes().to_vec()
                                    });
                                    b.extend(
                                        (if mode == Mode::Real { 0u16 } else { 8 }).to_le_bytes(),
                                    );
                                }
                                let head = b.len();
                                b.resize(15, 0);
                                let insn = Decoder::with_ip(
                                    mode.bits(),
                                    &b,
                                    0x1000,
                                    DecoderOptions::MOV_TR,
                                )
                                .decode();
                                if insn.is_invalid() && !reserved {
                                    continue;
                                }
                                if !two
                                    && matches!(op, 0xd8..=0xdf)
                                    && !insn.cpuid_features().iter().all(|f| {
                                        matches!(
                                            f,
                                            iced_x86::CpuidFeature::FPU
                                                | iced_x86::CpuidFeature::FPU287
                                                | iced_x86::CpuidFeature::FPU387
                                        )
                                    })
                                {
                                    continue;
                                }
                                if insn.is_invalid() {
                                    let immediate = if two && op == 0xba
                                        || !two && matches!(op, 0xc0 | 0xc1 | 0xc6 | 0xf6)
                                    {
                                        1
                                    } else if !two && matches!(op, 0xc7 | 0xf7) {
                                        if operand32 {
                                            4
                                        } else {
                                            2
                                        }
                                    } else {
                                        0
                                    };
                                    b.truncate(head + immediate);
                                } else {
                                    b.truncate(insn.len());
                                }
                                // PRM §17 AAM; group 8 exercises the reported bit 21.
                                if !two && matches!(op, 0xd4 | 0xd5) {
                                    b[head] = 10;
                                }
                                if matches!(
                                    insn.op0_kind(),
                                    OpKind::NearBranch16 | OpKind::NearBranch32
                                ) {
                                    b[head] = 16;
                                }
                                if two && op == 0xba {
                                    b[head] = 21;
                                }
                                if (!two
                                    && matches!(op, 0xc0..=0xc1 | 0xc6..=0xc7 | 0x69 | 0x6b | 0x68 | 0x6a | 0x80..=0x81 | 0x83 | 0xb0..=0xbf))
                                    || (two && matches!(op, 0xa4 | 0xac))
                                {
                                    b[head] = 1;
                                }
                                if !two && op == 0xc8 {
                                    b[head] = 16;
                                    b[head + 2] = 2;
                                }
                                if !two && op == 0xcd {
                                    b[head] = 0x30;
                                }
                                let insn = Decoder::with_ip(
                                    mode.bits(),
                                    &b,
                                    0x1000,
                                    DecoderOptions::MOV_TR,
                                )
                                .decode();
                                out.push(Probe {
                                    bytes: b,
                                    insn,
                                    shape: if !has_modrm {
                                        "implicit"
                                    } else {
                                        ["reg", "mem", "sib"][shape as usize]
                                    },
                                    x87: !two && matches!(op, 0xd8..=0xdf),
                                    reserved,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    // The embedder's exact witness, alongside systematic /r coverage.
    let bytes = vec![0x0f, 0xba, 0xe2, 0x15];
    let insn = Decoder::with_ip(mode.bits(), &bytes, 0x1000, DecoderOptions::MOV_TR).decode();
    out.push(Probe {
        bytes,
        insn,
        shape: "reported",
        x87: false,
        reserved: false,
    });
    out
}

fn descriptor(vm: &Vm, selector: u16, base: u32, limit: u32, attr: u16) {
    descriptor_fixture::descriptor(vm, 0x2000 + u64::from(selector), base, limit, attr);
}
fn setup(model: CpuModel, p: &Probe, mode: Mode, application: bool) -> (Vm, Vcpu) {
    let mut bytes = p.bytes.clone();
    bytes.extend([0xf4; 32]);
    let (vm, mut cpu) = if application {
        let mut vm = Vm::new(VmConfig::flat(0x20000));
        vm.map(0, 0x20000, Prot::RWX, RegionKind::Ram).unwrap();
        vm.set_cpu_mode(CpuMode::Compat32);
        vm.write_bytes(0x1000, &bytes).unwrap();
        let cpu = vm.new_vcpu();
        (vm, cpu)
    } else {
        descriptor_fixture::setup(model, &bytes)
    };
    cpu.set_fpu_model(if p.x87 || model.i486_signature().is_some() {
        FpuModel::Present
    } else {
        FpuModel::None
    });
    vm.write_bytes(0x1800, &[0xf4]).unwrap();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[..8].copy_from_slice(&[7, 1, 0, 0x90f0, 0x8000, 0x8100, 0x6000, 0x7000]);
    cpu.cpu.flags = Default::default();
    cpu.cpu.legacy.cr0 = u32::from(mode != Mode::Real);
    cpu.cpu.legacy.gdtr.base = 0x2000;
    cpu.cpu.legacy.gdtr.limit = 0x3f;
    cpu.cpu.legacy.idtr.base = if mode == Mode::Real { 0 } else { 0x3000 };
    cpu.cpu.legacy.idtr.limit = if mode == Mode::Real { 0x3ff } else { 0x7ff };
    let wide = matches!(mode, Mode::P32 | Mode::PagedP32) || mode.v86();
    descriptor(&vm, 8, 0, 0xffff_ffff, if wide { 0xc09b } else { 0x809b });
    descriptor(&vm, 16, 0, 0xffff_ffff, if wide { 0xc093 } else { 0x8093 });
    descriptor(&vm, 24, 0x4000, 0xff, 0x82); // LDT
    descriptor(&vm, 32, 0x5000, 0x67, 0x89); // available 386 TSS
    let paragraphs = mode == Mode::Real || mode.v86();
    cpu.cpu.cs = if paragraphs { 0 } else { 8 };
    let data = if paragraphs { 0 } else { 16 };
    cpu.cpu.ds = data;
    cpu.cpu.es = data;
    cpu.cpu.ss = data;
    cpu.cpu.fs = data;
    cpu.cpu.gs = data;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        *cache = SegmentCache {
            base: 0,
            limit: if paragraphs { 0xffff } else { u32::MAX },
            attributes: if i == 0 { 0x9b } else { 0x93 }
                | if wide && !mode.v86() {
                    0xc000
                } else if !paragraphs {
                    0x8000
                } else {
                    0
                },
            unusable: 0,
            reserved: 0,
        };
    }
    // PRM §9.4 / Fig. 9-3 and §9.5: valid IVT/IDT handlers at CPL 0.
    for vector in 0..256u64 {
        if mode == Mode::Real {
            vm.write_bytes(vector * 4, &0x1800u32.to_le_bytes())
                .unwrap();
        } else {
            vm.write_bytes(
                0x3000 + vector * 8,
                &[0, 0x18, 8, 0, 0, if mode.v86() { 0xee } else { 0x8e }, 0, 0],
            )
            .unwrap();
        }
    }
    vm.write_bytes(0x6000, &0x89abcdefu32.to_le_bytes())
        .unwrap();
    vm.write_bytes(0x7000, &19u32.to_le_bytes()).unwrap();
    let m = p.insn.mnemonic();
    if matches!(m, M::Lar | M::Lsl | M::Verr | M::Verw)
        || m == M::Mov
            && (p.insn.op0_register().is_segment_register()
                || p.insn.op1_register().is_segment_register())
    {
        cpu.cpu.gpr[3] = u64::from(data);
        vm.write_bytes(0x6000, &data.to_le_bytes()).unwrap();
    }
    if m == M::Mov
        && (p.insn.op0_register().is_dr()
            || p.insn.op1_register().is_dr()
            || p.insn.op0_register().is_tr()
            || p.insn.op1_register().is_tr())
    {
        cpu.cpu.gpr[3] = 0;
    }
    let operand32 = p.insn.code().op_code().operand_size() == 32;
    let size = if operand32 { 4 } else { 2 };
    let selector = if paragraphs { 0u16 } else { 8 };
    vm.write_bytes(0x8000, &0x1800u32.to_le_bytes()[..size])
        .unwrap();
    vm.write_bytes(0x8000 + size as u64, &selector.to_le_bytes())
        .unwrap();
    vm.write_bytes(0x8000 + 2 * size as u64, &2u32.to_le_bytes()[..size])
        .unwrap();
    if matches!(m, M::Pop) {
        vm.write_bytes(0x8000, &data.to_le_bytes()).unwrap();
    }
    if matches!(m, M::Popf | M::Popfd) {
        vm.write_bytes(0x8000, &2u32.to_le_bytes()).unwrap();
    }
    if matches!(m, M::Lds | M::Les | M::Lss | M::Lfs | M::Lgs) {
        vm.write_bytes(0x6000, &0x7000u32.to_le_bytes()).unwrap();
        vm.write_bytes(0x6000 + size as u64, &data.to_le_bytes())
            .unwrap();
    }
    if matches!(m, M::Call | M::Jmp) && matches!(p.insn.op0_kind(), OpKind::Memory) {
        vm.write_bytes(0x6000, &0x1800u32.to_le_bytes()).unwrap();
        vm.write_bytes(0x6000 + size as u64, &selector.to_le_bytes())
            .unwrap();
    }
    if matches!(m, M::Call | M::Jmp) && p.insn.op0_kind() == OpKind::Register {
        cpu.cpu.gpr[3] = 0x1800;
    }
    if matches!(m, M::Bound) {
        let bounds = if operand32 {
            [i32::MIN.to_le_bytes(), i32::MAX.to_le_bytes()].concat()
        } else {
            [i16::MIN.to_le_bytes(), i16::MAX.to_le_bytes()].concat()
        };
        vm.write_bytes(0x6000, &bounds).unwrap();
    }
    if matches!(m, M::Lldt | M::Ltr | M::Lmsw) {
        let value: u16 = match m {
            M::Lldt => 24,
            M::Ltr => 32,
            _ => u16::from(mode != Mode::Real),
        };
        cpu.cpu.gpr[3] = u64::from(value);
        vm.write_bytes(0x6000, &value.to_le_bytes()).unwrap();
    }
    if matches!(m, M::Lgdt | M::Lidt) {
        let base = if m == M::Lgdt { 0x2000u32 } else { 0x3000 };
        vm.write_bytes(0x6000, &0x3ffu16.to_le_bytes()).unwrap();
        vm.write_bytes(0x6002, &base.to_le_bytes()).unwrap();
    }
    if m == M::Mov && (p.insn.op0_register().is_cr() || p.insn.op1_register().is_cr()) {
        cpu.cpu.gpr[3] = u64::from(mode != Mode::Real);
    }
    if m == M::Into {
        cpu.cpu.flags.of = true;
    }
    if mode.v86() && !application {
        // SDM Vol 3B §§23.2.1–23.2.8 pp. 23-6–11; 386 PRM
        // §§15.2–15.5 pp. 15-4–10: canonical paragraphs, CPL=3,
        // IOPL=3, and a live permissive bitmap (also required at IOPL=3).
        cpu.cpu.legacy.cpl = 3;
        cpu.cpu.rflags_system = (1 << 17) | 0x3000;
        cpu.cpu.legacy.tr = x86jit_core::legacy::SystemSegment {
            selector: 0x28,
            cache: SegmentCache {
                base: 0x9000,
                limit: 0x2100,
                attributes: 0x8b,
                ..Default::default()
            },
            ..Default::default()
        };
        vm.write_bytes(0x9004, &0x18000u32.to_le_bytes()).unwrap();
        vm.write_bytes(0x9008, &16u16.to_le_bytes()).unwrap();
        vm.write_bytes(0x9066, &0x100u16.to_le_bytes()).unwrap();
    }
    if mode.paged() && !application {
        // 80386 PRM §5.2 pp. 5-7–11; Intel486 PRM §5.3 pp. 5-18–22:
        // one identity-mapped directory/table, separate from probe storage.
        vm.write_bytes(0x1e000, &0x1f007u32.to_le_bytes()).unwrap();
        for page in 0..32u32 {
            vm.write_bytes(
                0x1f000 + u64::from(page * 4),
                &((page * 0x1000) | 7).to_le_bytes(),
            )
            .unwrap();
        }
        cpu.cpu.legacy.cr3 = 0x1e000;
        cpu.cpu.legacy.cr0 |= 0x8000_0000;
    }
    (vm, cpu)
}

fn comparable(p: &Probe) -> bool {
    use M::*;
    !p.x87
        && !matches!(p.insn.mnemonic(), Mov if p.insn.op0_register().is_segment_register() || p.insn.op1_register().is_segment_register() || p.insn.op0_register().is_cr() || p.insn.op1_register().is_cr() || p.insn.op0_register().is_dr() || p.insn.op1_register().is_dr() || p.insn.op0_register().is_tr() || p.insn.op1_register().is_tr())
        && !matches!(
            p.insn.mnemonic(),
            Arpl | Lar
                | Lsl
                | Verr
                | Verw
                | Lldt
                | Ltr
                | Sldt
                | Str
                | Lgdt
                | Lidt
                | Sgdt
                | Sidt
                | Lmsw
                | Smsw
                | Clts
                | Lds
                | Les
                | Lss
                | Lfs
                | Lgs
                | Int
                | Int3
                | Into
                | Iret
                | Iretd
                | Retf
                | Cli
                | Sti
                | In
                | Out
                | Insb
                | Insw
                | Insd
                | Outsb
                | Outsw
                | Outsd
                | Wait
                | Cpuid
                | Invd
                | Wbinvd
                | Invlpg
        )
        && !matches!(p.insn.op0_kind(), OpKind::FarBranch16 | OpKind::FarBranch32)
        && !(matches!(p.insn.mnemonic(), Push | Pop) && p.insn.op0_register().is_segment_register())
        && !(matches!(p.insn.mnemonic(), Call | Jmp) && p.insn.is_jmp_far_indirect()
            || p.insn.is_call_far_indirect())
}
fn same_application_decode(p: &Probe) -> bool {
    let app = Decoder::with_ip(32, &p.bytes, 0x1000, DecoderOptions::NONE).decode();
    !app.is_invalid()
        && app.code() == p.insn.code()
        && app.len() == p.insn.len()
        && app.op_kinds().eq(p.insn.op_kinds())
        && (0..app.op_count()).all(|i| app.op_register(i) == p.insn.op_register(i))
        && app.memory_base() == p.insn.memory_base()
        && app.memory_index() == p.insn.memory_index()
        && app.memory_displacement64() == p.insn.memory_displacement64()
}
fn compare(p: &Probe, mode: Mode, cpu: &Vcpu, vm: &Vm) -> String {
    let (app, mut other) = setup(CpuModel::I80386, p, mode, true);
    let ir = match x86jit_core::lift::lift_one(&app.mem, FetchAddr::flat(0x1000), CpuMode::Compat32)
    {
        Ok(ir) => ir,
        Err(e) => return format!("unavailable:{e:?}"),
    };
    let result = x86jit_core::interp::interpret_block(
        &ir,
        &mut other.cpu,
        &app.mem,
        &mut Vec::new(),
        &mut Default::default(),
    );
    if let x86jit_core::StepResult::Exit(ref e) = result {
        if !matches!(e, Exit::Hlt) {
            return format!("unavailable:{e:?}");
        }
    }
    let undefined = p.insn.rflags_undefined();
    let mut mask = 0xcd5u64;
    for (iced, architectural) in [
        (iced_x86::RflagsBits::CF, 1),
        (iced_x86::RflagsBits::PF, 4),
        (iced_x86::RflagsBits::AF, 16),
        (iced_x86::RflagsBits::ZF, 64),
        (iced_x86::RflagsBits::SF, 128),
        (iced_x86::RflagsBits::DF, 1024),
        (iced_x86::RflagsBits::OF, 2048),
    ] {
        if undefined & iced != 0 {
            mask &= !architectural;
        }
    }
    let mut differences = Vec::new();
    for i in 0..8 {
        if cpu.cpu.gpr[i] as u32 != other.cpu.gpr[i] as u32 {
            differences.push(format!("r{i}:{:x}!={:x}", cpu.cpu.gpr[i], other.cpu.gpr[i]));
        }
    }
    if cpu.cpu.rip != other.cpu.rip {
        differences.push(format!("ip:{:x}!={:x}", cpu.cpu.rip, other.cpu.rip));
    }
    if (cpu.cpu.flags.to_rflags() ^ other.cpu.flags.to_rflags()) & mask != 0 {
        differences.push(format!(
            "flags:{:x}!={:x}/mask{mask:x}",
            cpu.cpu.flags.to_rflags(),
            other.cpu.flags.to_rflags()
        ));
    }
    let mut a = vec![0; 0x4000];
    let mut b = a.clone();
    vm.read_bytes(0x5000, &mut a).unwrap();
    app.read_bytes(0x5000, &mut b).unwrap();
    if a != b {
        let at = a.iter().zip(&b).position(|(a, b)| a != b).unwrap();
        differences.push(format!(
            "mem@{:x}:{:02x}!={:02x}",
            at + 0x5000,
            a[at],
            b[at]
        ));
    }
    if differences.is_empty() {
        "equal".into()
    } else {
        format!("DIFF:{}", differences.join(","))
    }
}

#[test]
#[ignore = "manual 386 opcode census; emits a TSV runtime artifact"]
fn runtime_census() {
    runtime_census_for(CpuModel::I80386, &[Mode::P32, Mode::P16, Mode::Real]);
}

#[test]
#[ignore = "manual 486 opcode census; emits TSV runtime artifacts"]
fn runtime_census_i486() {
    for model in [CpuModel::I80486Dx, CpuModel::I80486DxCpuid] {
        runtime_census_for(model, &[Mode::P32, Mode::P16, Mode::Real]);
    }
}

#[test]
#[ignore = "manual paged P32 opcode census; emits TSV runtime artifacts"]
fn runtime_census_paged() {
    for model in [
        CpuModel::I80386,
        CpuModel::I80486Dx,
        CpuModel::I80486DxCpuid,
    ] {
        runtime_census_for(model, &[Mode::PagedP32]);
    }
}

#[test]
#[ignore = "manual V86 and paged V86 census; emits TSV runtime artifacts"]
fn runtime_census_v86() {
    for model in [
        CpuModel::I80386,
        CpuModel::I80486Dx,
        CpuModel::I80486DxCpuid,
    ] {
        runtime_census_for(model, &[Mode::V86, Mode::PagedV86]);
    }
}

// Fail the manual census on any unexplained V86 outcome, including reserved
// controls. This is a runtime audit, not an instruction-membership oracle.
fn v86_rule(p: &Probe, stop: &StepStop, steps: u64) -> &'static str {
    match stop {
        StepStop::Exit(Exit::Exception { vector, .. }) => {
            assert_eq!(steps, 0);
            let rule = if p.reserved {
                assert_eq!(*vector, 6);
                "reserved encoding: 386 PRM App. A pp. A-4–8; 486 PRM App. A pp. A-4–7"
            } else {
                let (expected, rule) = match p.insn.mnemonic() {
                    M::Arpl => (6, "protected-only #UD: SDM Vol 2A ARPL pp. 3-75–76"),
                    M::Lar => (6, "protected-only #UD: SDM Vol 2A LAR p. 3-536"),
                    M::Lsl => (6, "protected-only #UD: SDM Vol 2A LSL p. 3-575"),
                    M::Verr | M::Verw => (6, "protected-only #UD: SDM Vol 2C VERR/VERW p. 5-163"),
                    M::Lldt => (6, "protected-only #UD: SDM Vol 2A LLDT p. 3-559"),
                    M::Sldt => (6, "protected-only #UD: SDM Vol 2B SLDT p. 4-657"),
                    M::Ltr => (6, "protected-only #UD: SDM Vol 2A LTR p. 3-577"),
                    M::Str => (6, "protected-only #UD: SDM Vol 2B STR p. 4-679"),
                    M::Hlt | M::Lgdt | M::Lidt | M::Lmsw | M::Clts
                    | M::Invd | M::Wbinvd | M::Invlpg => (
                        13,
                        "CPL=3 privileged #GP(0): SDM Vol 3B §23.2.6 pp. 23-9–10; 486 PRM §23.7.1 p. 23-14",
                    ),
                    M::Mov if p.insn.op0_register().is_cr()
                        || p.insn.op1_register().is_cr()
                        || p.insn.op0_register().is_dr()
                        || p.insn.op1_register().is_dr()
                        || p.insn.op0_register().is_tr()
                        || p.insn.op1_register().is_tr() => (
                        13,
                        "special-register #GP(0): SDM Vol 2B MOV CR/DR pp. 4-32–36; 386 PRM §10.6.2 pp. 10-15–16; 486 PRM §10.7.2 pp. 10-13–17",
                    ),
                    _ => panic!("unexplained V86 fault: {}: {stop:?}", p.insn),
                };
                assert_eq!(*vector, expected, "{}: {stop:?}", p.insn);
                rule
            };
            if *vector == 13 {
                let StepStop::Exit(exit) = stop else {
                    unreachable!()
                };
                assert_eq!(
                    exit.fault_metadata().unwrap().error_code(),
                    x86jit_core::fault::ErrorCode::bits(0, 0xffff)
                );
            }
            rule
        }
        StepStop::Exit(Exit::UnsupportedSystemContext { operation, .. }) => {
            assert_eq!(steps, 0);
            assert!(!p.reserved && p.insn.mnemonic() == M::Bswap);
            assert_eq!(p.insn.code().op_code().operand_size(), 16);
            assert_eq!(
                *operation,
                x86jit_core::UnsupportedSystemOperation::LegacyInstruction
            );
            "undefined 16-bit BSWAP refused: 486 PRM §26 BSWAP p. 26-35"
        }
        StepStop::Exit(Exit::BudgetExhausted | Exit::PortIo { .. }) => "",
        _ => panic!("unexplained V86 outcome: {}: {stop:?}", p.insn),
    }
}

fn runtime_census_for(model: CpuModel, modes: &[Mode]) {
    let mut report =
        String::from("model\tmode\tbytes\tmnemonic\tshape\tclass\texit\tcomparison\trule\n");
    let mut summary = String::from(
        "model\tmode\tinteger total\texecutes\trefused\tfault\tport I/O\tunknown\tpanic\n",
    );
    for &mode in modes {
        let mut counts = BTreeMap::<String, usize>::new();
        let mut comparisons = BTreeMap::<String, usize>::new();
        let mut families = BTreeMap::<String, usize>::new();
        let mut integer_refusals = BTreeMap::<String, usize>::new();
        let generated = probes(model, mode);
        for two in [false, true] {
            for op in 0..=255u8 {
                if defined(model, two, op) {
                    assert!(
                        generated.iter().any(|p| {
                            let raw = p
                                .bytes
                                .iter()
                                .copied()
                                .skip_while(|b| matches!(b, 0x66 | 0x67 | 0xf2 | 0xf3))
                                .collect::<Vec<_>>();
                            if two {
                                raw.starts_with(&[0x0f, op])
                            } else {
                                raw.first() == Some(&op)
                            }
                        }),
                        "missing opcode {two}/{op:02x}"
                    );
                }
            }
        }
        for p in generated {
            let (vm, mut cpu) = setup(model, &p, mode, false);
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                let run = cpu.run_steps(&vm, 1, StepOptions::default());
                let rule = if mode.v86() {
                    let r = run.as_ref().expect("V86 stepping failed");
                    v86_rule(&p, r.stop(), r.steps())
                } else {
                    ""
                };
                let (class, exit) = match run {
                    Ok(r) => {
                        match r.stop() {
                            StepStop::Exit(Exit::UnsupportedSystemContext {
                                operation, ..
                            }) if !p.x87 && !p.reserved => {
                                *integer_refusals
                                    .entry(format!("{:?}/{operation:?}", p.insn.mnemonic()))
                                    .or_default() += 1;
                            }
                            _ => {}
                        }
                        let class = match r.stop() {
                            StepStop::Exit(Exit::UnsupportedSystemContext { .. }) => "refused",
                            StepStop::Exit(Exit::UnknownInstruction { .. }) => "unknown",
                            StepStop::Exit(Exit::Exception { .. }) => "fault",
                            StepStop::Exit(Exit::PortIo { .. }) => "io-expected",
                            StepStop::Exit(Exit::BudgetExhausted | Exit::Hlt) => "executes",
                            _ => "other",
                        };
                        (
                            class.to_owned(),
                            format!("{:?};steps={}", r.stop(), r.steps()),
                        )
                    }
                    Err(e) => ("step-error".into(), format!("{e:?}")),
                };
                let comparison = if !p.reserved
                    && class == "executes"
                    && comparable(&p)
                    && same_application_decode(&p)
                {
                    compare(&p, mode, &cpu, &vm)
                } else {
                    "excluded".into()
                };
                (class, exit, comparison, rule)
            }));
            let (class, exit, comparison, rule) = outcome.unwrap_or_else(|e| {
                (
                    "panic".into(),
                    e.downcast_ref::<String>()
                        .cloned()
                        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default(),
                    "excluded".into(),
                    "panic",
                )
            });
            if mode.v86() {
                assert!(
                    !matches!(class.as_str(), "panic" | "unknown" | "step-error" | "other"),
                    "{model:?}/{mode:?}: {}: {exit}",
                    p.insn
                );
                assert!(!comparison.starts_with("DIFF:"), "{}: {comparison}", p.insn);
            }
            let category = if p.reserved {
                format!("reserved-{class}")
            } else if p.x87 {
                format!("x87-{class}")
            } else {
                class.clone()
            };
            *counts.entry(category.clone()).or_default() += 1;
            if matches!(
                class.as_str(),
                "refused" | "unknown" | "panic" | "step-error"
            ) || comparison.starts_with("DIFF:")
            {
                *families
                    .entry(format!(
                        "{category}/{:?}/{}",
                        p.insn.mnemonic(),
                        comparison.split(':').next().unwrap()
                    ))
                    .or_default() += 1;
            }
            *comparisons
                .entry(comparison.split(':').next().unwrap().into())
                .or_default() += 1;
            let bytes: String = p.bytes.iter().map(|b| format!("{b:02X}")).collect();
            writeln!(
                report,
                "{model:?}\t{mode:?}\t{bytes}\t{}\t{}\t{category}\t{exit}\t{comparison}\t{rule}",
                p.insn, p.shape
            )
            .unwrap();
        }
        assert_eq!(
            counts.get("x87-refused").copied().unwrap_or(0),
            0,
            "x87 must not refuse the Machine context in {mode:?}"
        );
        if mode.paged() && !mode.v86() {
            // Existing explicit limitations: test-register commands (386 PRM
            // §10.6.2 p. 10-14; Intel486 PRM §4.5 pp. 4-14–17), and
            // undefined 16-bit BSWAP (Intel486 PRM §26 p. 26-35).
            for reason in integer_refusals.keys() {
                assert!(
                    matches!(
                        reason.as_str(),
                        "Mov/LegacyTlbTest" | "Mov/LegacyCacheTest" | "Bswap/LegacyInstruction"
                    ),
                    "unexplained paged refusal: {reason}"
                );
            }
            for category in ["panic", "unknown", "step-error"] {
                assert_eq!(
                    counts.get(category).copied().unwrap_or(0),
                    0,
                    "{model:?}/{category}"
                );
            }
            // Machine retains the historical F2 string repetition policy;
            // the Application decoder ignores F2 on MOVS/STOS/LODS. These
            // same 24 RCX-only comparisons differ in the unpaged census.
            writeln!(report, "# F2 MOVS/STOS/LODS comparisons retain the unpaged Machine/Application prefix-policy difference").unwrap();
        }
        let count = |category: &str| counts.get(category).copied().unwrap_or(0);
        let integer_total: usize = counts
            .iter()
            .filter(|(category, _)| {
                !category.starts_with("reserved-") && !category.starts_with("x87-")
            })
            .map(|(_, count)| count)
            .sum();
        writeln!(
            summary,
            "{model:?}\t{mode:?}\t{integer_total}\t{}\t{}\t{}\t{}\t{}\t{}",
            count("executes"),
            count("refused"),
            count("fault"),
            count("io-expected"),
            count("unknown"),
            count("panic")
        )
        .unwrap();
        eprintln!("CENSUS {model:?} {mode:?}: {counts:?}; integer refusals={integer_refusals:?}; comparison={comparisons:?}; families={families:?}");
        writeln!(report, "# {model:?} {mode:?}: {counts:?}; integer refusals={integer_refusals:?}; comparison={comparisons:?}; families={families:?}").unwrap();
    }
    eprintln!("INTEGER CENSUS (excludes D8–DF and reserved group controls)\n{summary}");
    report.push_str(&format!("# INTEGER SUMMARY\n{summary}"));
    let path = std::env::var("CENSUS_386_OUT").unwrap_or_else(|_| {
        std::env::temp_dir()
            .join(format!(
                "census-{model:?}-runtime{}.tsv",
                if modes[0].v86() {
                    "-v86"
                } else if modes[0].paged() {
                    "-paged"
                } else {
                    ""
                }
            ))
            .display()
            .to_string()
    });
    std::fs::write(&path, report).unwrap();
    eprintln!("Census rows: {path}");
}

/// Sweep defined one-byte/0F CPU forms with redundant prefixes. Keep modern
/// decoder labels in the report: a changed label is evidence of reinterpretation,
/// not evidence that the historical model has that instruction. See
/// legacy_redundant_prefix for the per-generation REP assumptions and citations.
#[test]
#[ignore = "manual redundant-prefix census; emits TSV runtime artifacts"]
fn redundant_prefix_census() {
    for model in [
        CpuModel::I80386,
        CpuModel::I80486Dx,
        CpuModel::I80486DxCpuid,
    ] {
        let mut report = String::from("model\tmode\tprefix\tbytes\tdecoded\tbaseline\toutcome\n");
        for mode in [Mode::Real, Mode::P16, Mode::P32] {
            let mut seen = std::collections::BTreeSet::new();
            let mut counts = BTreeMap::<String, usize>::new();
            let mut regressions = BTreeMap::<String, usize>::new();
            for p in probes(model, mode) {
                if p.x87
                    || p.reserved
                    || p.bytes
                        .first()
                        .is_some_and(|b| matches!(b, 0x66 | 0x67 | 0xf2 | 0xf3))
                    || !seen.insert((
                        p.insn.code() as u32,
                        p.shape,
                        p.insn.op0_register() as u32,
                        p.insn.op1_register() as u32,
                    ))
                {
                    continue;
                }
                let baseline = prefix_outcome(model, &p, mode);
                for prefix in [0xf2, 0xf3, 0x26, 0x2e, 0x36, 0x3e, 0x64, 0x65, 0x66, 0x67] {
                    let mut bytes = vec![prefix];
                    bytes.extend(&p.bytes);
                    bytes.resize(15, 0);
                    let insn =
                        Decoder::with_ip(mode.bits(), &bytes, 0x1000, DecoderOptions::MOV_TR)
                            .decode();
                    // An operand-size prefix may change immediate length; padding
                    // is deliberate. Do not discard decoder-invalid forms.
                    bytes.truncate(insn.len());
                    let q = Probe {
                        bytes,
                        insn,
                        ..p.clone()
                    };
                    let outcome = prefix_outcome(model, &q, mode);
                    *counts.entry(outcome.clone()).or_default() += 1;
                    if outcome != baseline
                        && matches!(
                            outcome.as_str(),
                            "ud" | "refused" | "unknown" | "panic" | "step-error"
                        )
                    {
                        *regressions
                            .entry(format!("{prefix:02X}/{}/{outcome}", p.insn))
                            .or_default() += 1;
                    }
                    let bytes: String = q.bytes.iter().map(|b| format!("{b:02X}")).collect();
                    writeln!(
                        report,
                        "{model:?}\t{mode:?}\t{prefix:02X}\t{bytes}\t{}\t{baseline}\t{outcome}",
                        q.insn
                    )
                    .unwrap();
                }
            }
            eprintln!(
                "PREFIX CENSUS {model:?}/{mode:?}: {counts:?}; new refusals/UD={regressions:?}"
            );
            writeln!(
                report,
                "# {model:?}/{mode:?}: {counts:?}; new refusals/UD={regressions:?}"
            )
            .unwrap();
            assert_eq!(counts.get("panic"), None);
            assert_eq!(counts.get("step-error"), None);
        }
        let path = std::env::temp_dir().join(format!("census-{model:?}-prefixes.tsv"));
        std::fs::write(&path, report).unwrap();
        eprintln!("Prefix census rows: {}", path.display());
    }
}

fn prefix_outcome(model: CpuModel, p: &Probe, mode: Mode) -> String {
    catch_unwind(AssertUnwindSafe(|| {
        let (vm, mut cpu) = setup(model, p, mode, false);
        match cpu.run_steps(&vm, 1, StepOptions::default()) {
            Ok(run) => match run.stop() {
                StepStop::Exit(Exit::Exception { vector: 6, .. }) => "ud",
                StepStop::Exit(Exit::Exception { .. }) => "fault",
                StepStop::Exit(Exit::UnsupportedSystemContext { .. }) => "refused",
                StepStop::Exit(Exit::UnknownInstruction { .. }) => "unknown",
                StepStop::Exit(Exit::BudgetExhausted | Exit::Hlt) => "executes",
                StepStop::Exit(Exit::PortIo { .. }) => "io",
                _ => "other",
            },
            Err(_) => "step-error",
        }
    }))
    .unwrap_or("panic")
    .to_owned()
}

#[test]
#[ignore = "manual absent-coprocessor x87 census; emits a TSV runtime artifact"]
fn runtime_census_x87_absent() {
    let mut report = String::from("mode\tbytes\tmnemonic\tclass\texit\n");
    for mode in [
        Mode::Real,
        Mode::P16,
        Mode::P32,
        Mode::V86,
        Mode::PagedP32,
        Mode::PagedV86,
    ] {
        let mut counts = BTreeMap::<String, usize>::new();
        for p in probes(CpuModel::I80386, mode).into_iter().filter(|p| p.x87) {
            let (vm, mut cpu) = setup(CpuModel::I80386, &p, mode, false);
            cpu.set_fpu_model(FpuModel::None);
            let r = cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
            let class = match r.stop() {
                StepStop::Exit(Exit::BudgetExhausted) => "executes",
                StepStop::Exit(Exit::UnknownInstruction { .. }) => "unknown",
                StepStop::Exit(Exit::Exception { .. }) => "fault",
                other => panic!("{mode:?}/{}: {other:?}", p.insn),
            };
            *counts.entry(class.into()).or_default() += 1;
            let bytes: String = p.bytes.iter().map(|b| format!("{b:02X}")).collect();
            writeln!(
                report,
                "{mode:?}\t{bytes}\t{}\t{class}\t{:?}",
                p.insn,
                r.stop()
            )
            .unwrap();
        }
        println!("I80386/None/{mode:?}: {counts:?}");
    }
    let path = std::env::temp_dir().join("census-x87-absent.tsv");
    std::fs::write(&path, report).unwrap();
    println!("{}", path.display());
}

#[test]
fn flat32_checked_census_forms_are_equal() {
    flat32_checked_census(0);
}

#[test]
fn flat32_cpl3_checked_census_forms_are_equal() {
    flat32_checked_census(3);
}

fn flat32_checked_census(cpl: u8) {
    use x86jit_core::{derive_legacy_context, SegmentClass};
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for mode in [Mode::P32, Mode::PagedP32] {
            let mut count = 0;
            let mut sequences = 0;
            // Retain register, absolute and SIB forms, both size overrides and
            // string policies from the same public opcode census generator.
            for mut p in probes(model, mode)
                .into_iter()
                .filter(comparable)
                .step_by(if mode.paged() { 1 } else { 17 })
            {
                // Carry each result into INC EBX; XOR EDX,EBX; HLT. Budgets
                // stop both within the sequence and after its terminal exit.
                p.bytes.extend_from_slice(b"\x43\x31\xda\xf4");
                for budget in [1, 3, 5] {
                    let (avm, mut a) = setup(model, &p, mode, false);
                    let (bvm, mut b) = setup(model, &p, mode, false);
                    for cpu in [&mut a, &mut b] {
                        cpu.cpu.legacy.cpl = cpl;
                        for cache in &mut cpu.cpu.legacy.segments {
                            cache.attributes |= u16::from(cpl) << 5;
                        }
                        cpu.cpu.cs |= u16::from(cpl);
                        cpu.cpu.ss |= u16::from(cpl);
                        cpu.cpu.ds |= u16::from(cpl);
                        cpu.cpu.es |= u16::from(cpl);
                    }
                    assert_eq!(
                        derive_legacy_context(a.legacy_cpu_model().unwrap(), &a.cpu)
                            .unwrap()
                            .segment_class,
                        SegmentClass::Flat32
                    );
                    // ES fails the Flat32 proof; a non-present hidden GS
                    // keeps Segmented off (SDM Vol 3A §3.4.5: P is checked
                    // on a load). No probe reloads GS.
                    b.cpu.legacy.segments[2].limit -= 1;
                    b.cpu.legacy.segments[5].attributes &= !0x80;
                    assert_eq!(
                        derive_legacy_context(b.legacy_cpu_model().unwrap(), &b.cpu)
                            .unwrap()
                            .segment_class,
                        SegmentClass::Checked
                    );
                    let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                    assert_eq!(
                        format!("{ar:?}"),
                        format!("{br:?}"),
                        "{:02x?} budget={budget}",
                        p.bytes
                    );
                    sequences += usize::from(ar.steps() > 1);
                    b.cpu.legacy.segments[2].limit += 1;
                    b.cpu.legacy.segments[5].attributes |= 0x80;
                    assert_eq!(
                        format!("{:?}", a.cpu),
                        format!("{:?}", b.cpu),
                        "{:02x?}",
                        p.bytes
                    );
                    assert_eq!(a.retired_instructions(), b.retired_instructions());
                    let mut am = vec![0; 0x20000];
                    let mut bm = vec![0; 0x20000];
                    avm.read_bytes(0, &mut am).unwrap();
                    bvm.read_bytes(0, &mut bm).unwrap();
                    assert_eq!(am, bm, "{:02x?}", p.bytes);
                }
                count += 1;
            }
            eprintln!("{model:?}/{mode:?}/CPL{cpl}: {count} forms, {} comparisons, {sequences} multi-instruction runs", count * 3);
            assert!(count > 100, "census unexpectedly small: {count}");
            assert!(
                sequences > 100,
                "too few multi-instruction runs: {sequences}"
            );
        }
    }
}

#[test]
fn seg32_checked_census_forms_are_equal() {
    seg32_checked_census(0);
}

#[test]
fn seg32_cpl3_checked_census_forms_are_equal() {
    seg32_checked_census(3);
}

/// CS.D=1 in 64 KiB segments with a non-zero CS base: the Segmented class
/// against Checked over every comparable form, including the 66h/67h
/// variants (16-bit addressing in 32-bit code). Alternate forms run with
/// SS.B=0 (SP, SDM Vol 1 §6.2.3 p. 6-3) and with G=1 DS/ES caches whose
/// scaled limit is 1FFFFh (SDM Vol 3A §3.4.5 p. 3-11).
fn seg32_checked_census(cpl: u8) {
    use x86jit_core::{derive_legacy_context, SegmentClass};
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for mode in [Mode::P32, Mode::PagedP32] {
            let mut count = 0;
            let mut sequences = 0;
            for (index, mut p) in probes(model, mode)
                .into_iter()
                .filter(comparable)
                .enumerate()
                .filter(|(i, _)| mode.paged() || i % 3 == 0)
            {
                p.bytes.extend_from_slice(b"\x43\x31\xda\xf4");
                for budget in [1, 3, 5] {
                    let (avm, mut a) = setup(model, &p, mode, false);
                    let (bvm, mut b) = setup(model, &p, mode, false);
                    for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                        cpu.cpu.legacy.cpl = cpl;
                        for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
                            cache.attributes |= u16::from(cpl) << 5;
                            cache.attributes &= !0x8000;
                            cache.limit = 0xffff;
                            cache.base = if i == 0 { 0x100 } else { 0 };
                            if index % 4 == 1 && (i == 1 || i == 2) {
                                cache.attributes |= 0x8000;
                                cache.limit = 0x1_ffff;
                            }
                        }
                        if index % 2 == 1 {
                            cpu.cpu.legacy.segments[3].attributes &= !0x4000;
                        }
                        cpu.cpu.cs |= u16::from(cpl);
                        cpu.cpu.ss |= u16::from(cpl);
                        cpu.cpu.ds |= u16::from(cpl);
                        cpu.cpu.es |= u16::from(cpl);
                        vm.write_bytes(0x1100, &p.bytes).unwrap();
                    }
                    let context =
                        derive_legacy_context(a.legacy_cpu_model().unwrap(), &a.cpu).unwrap();
                    assert_eq!(context.segment_class, SegmentClass::Segmented);
                    assert_eq!(context.default_operand_bits, 32);
                    b.cpu.legacy.segments[5].attributes &= !0x80;
                    assert_eq!(
                        derive_legacy_context(b.legacy_cpu_model().unwrap(), &b.cpu)
                            .unwrap()
                            .segment_class,
                        SegmentClass::Checked
                    );
                    let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                    assert_eq!(
                        format!("{ar:?}"),
                        format!("{br:?}"),
                        "{model:?}/{mode:?} CPL{cpl} {:02x?} budget={budget}",
                        p.bytes
                    );
                    sequences += usize::from(ar.steps() > 1);
                    b.cpu.legacy.segments[5].attributes |= 0x80;
                    assert_eq!(
                        format!("{:?}", a.cpu),
                        format!("{:?}", b.cpu),
                        "{:02x?}",
                        p.bytes
                    );
                    assert_eq!(a.retired_instructions(), b.retired_instructions());
                    let mut am = vec![0; 0x20000];
                    let mut bm = vec![0; 0x20000];
                    avm.read_bytes(0, &mut am).unwrap();
                    bvm.read_bytes(0, &mut bm).unwrap();
                    assert_eq!(am, bm, "{:02x?}", p.bytes);
                }
                count += 1;
            }
            eprintln!(
                "Segmented CS.D=1 {model:?}/{mode:?}/CPL{cpl}: {count} forms, {} comparisons, {sequences} multi-instruction runs",
                count * 3
            );
            assert!(count > 100, "census unexpectedly small: {count}");
            assert!(
                sequences > 100,
                "too few multi-instruction runs: {sequences}"
            );
        }
    }
}

#[test]
fn seg16_checked_census_forms_are_equal() {
    seg16_checked_census(0);
}

#[test]
fn seg16_cpl3_checked_census_forms_are_equal() {
    seg16_checked_census(3);
}

fn seg16_checked_census(cpl: u8) {
    use x86jit_core::{derive_legacy_context, SegmentClass};
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        for mode in [Mode::P16, Mode::PagedP16] {
            let mut count = 0;
            for mut p in probes(model, mode).into_iter().filter(comparable) {
                p.bytes.extend_from_slice(b"\x43\x31\xda\xf4");
                for budget in [1, 3, 5] {
                    let (avm, mut a) = setup(model, &p, mode, false);
                    let (bvm, mut b) = setup(model, &p, mode, false);
                    for (vm, cpu) in [(&avm, &mut a), (&bvm, &mut b)] {
                        cpu.cpu.legacy.cpl = cpl;
                        for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
                            cache.attributes |= u16::from(cpl) << 5;
                            cache.limit = 0xffff;
                            cache.base = if i == 0 { 0x100 } else { 0 };
                        }
                        cpu.cpu.cs |= u16::from(cpl);
                        cpu.cpu.ss |= u16::from(cpl);
                        cpu.cpu.ds |= u16::from(cpl);
                        cpu.cpu.es |= u16::from(cpl);
                        vm.write_bytes(0x1100, &p.bytes).unwrap();
                    }
                    assert_eq!(
                        derive_legacy_context(a.legacy_cpu_model().unwrap(), &a.cpu)
                            .unwrap()
                            .segment_class,
                        SegmentClass::Segmented
                    );
                    // P is checked on segment load, not cached operand access
                    // (SDM Vol 3A §3.4.5). No probe reloads GS in this census.
                    b.cpu.legacy.segments[5].attributes &= !0x80;
                    assert_eq!(
                        derive_legacy_context(b.legacy_cpu_model().unwrap(), &b.cpu)
                            .unwrap()
                            .segment_class,
                        SegmentClass::Checked
                    );
                    let ar = a.run_steps(&avm, budget, StepOptions::default()).unwrap();
                    let br = b.run_steps(&bvm, budget, StepOptions::default()).unwrap();
                    assert_eq!(
                        format!("{ar:?}"),
                        format!("{br:?}"),
                        "{model:?}/{mode:?} CPL{cpl} {:02x?} budget={budget}",
                        p.bytes
                    );
                    b.cpu.legacy.segments[5].attributes |= 0x80;
                    assert_eq!(
                        format!("{:?}", a.cpu),
                        format!("{:?}", b.cpu),
                        "{:02x?}",
                        p.bytes
                    );
                    assert_eq!(a.retired_instructions(), b.retired_instructions());
                    let mut am = vec![0; 0x20000];
                    let mut bm = vec![0; 0x20000];
                    avm.read_bytes(0, &mut am).unwrap();
                    bvm.read_bytes(0, &mut bm).unwrap();
                    assert_eq!(am, bm, "{:02x?}", p.bytes);
                }
                count += 1;
            }
            eprintln!(
                "Segmented {model:?}/{mode:?}/CPL{cpl}: {count} forms, {} comparisons",
                count * 3
            );
            assert!(count > 100);
        }
    }
}
