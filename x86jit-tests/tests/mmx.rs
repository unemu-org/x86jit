//! SDM Vol 1 §9.5.1, §9.6.2; Vol 3A §15.2 Table 15-2.
//! Physical MM slots, x87 transitions, precise memory widths and shared packed semantics.
use iced_x86::code_asm::*;
use x86jit_core::{
    Backend, CpuMode, Exit, GuestCpuFeatures, InterpreterBackend, Prot, Reg, RegionCaps,
    RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;
use x86jit_tests::fuzz::mmx;
use x86jit_tests::oracle::{run_with_backend_mode, RunOutcome, VectorInput};
use x86jit_tests::vector::{CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec};

const CODE: u64 = 0x21_0000;
const DATA: u64 = 0x22_0000;

fn backends() -> Vec<Box<dyn Backend>> {
    vec![
        Box::new(InterpreterBackend),
        Box::new(JitBackend::new()),
        Box::new(JitBackend::with_superblocks(RegionCaps {
            max_blocks: 8,
            max_icount: 128,
        })),
    ]
}

fn program(emit: impl FnOnce(&mut CodeAssembler)) -> VectorInput {
    let mut a = CodeAssembler::new(64).unwrap();
    emit(&mut a);
    a.hlt().unwrap();
    VectorInput {
        cpu_init: CpuSnapshot::default(),
        entry: CODE,
        run: RunSpec::UntilExit,
        mem_init: vec![
            MemChunk {
                addr: CODE,
                bytes: a.assemble(CODE).unwrap(),
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: DATA,
                bytes: vec![0; 4096],
                kind: MemKind::Ram,
            },
        ],
    }
}

fn data(out: &RunOutcome) -> &[u8] {
    &out.mem.iter().find(|m| m.addr == DATA).unwrap().bytes
}
fn word(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
}
fn qword(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}
fn run(p: &VectorInput, mut check: impl FnMut(&RunOutcome)) {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        let out = x86jit_tests::native::run_native(p).expect("native MMX witness");
        check(&out);
    }
    for b in backends() {
        let out = run_with_backend_mode(p, b, GuestCpuFeatures::v2(), CpuMode::Long64);
        assert_eq!(out.exit, ExitKind::Hlt);
        check(&out);
    }
}

#[test]
fn mmx_all_forms_lift_and_differential() {
    for mode in [CpuMode::Long64, CpuMode::Compat32] {
        for &code in mmx::CODES {
            if mode == CpuMode::Compat32 && !code.op_code().mode32() {
                continue;
            }
            assert_eq!(
                x86jit_tests::compat::probe_code_in(code, mode),
                Some(x86jit_tests::compat::Probe::Lifted),
                "{code:?} {mode:?}"
            );
            for memory in [false, true] {
                for seed in [1, 64, 255] {
                    let p = mmx::input(&[code], seed, memory, mode);
                    mmx::check(&p, mode).unwrap_or_else(|e| {
                        panic!("{code:?} {mode:?} memory={memory} seed={seed}: {e}")
                    });
                }
            }
        }
    }
}

#[test]
fn mmx_pending_x87_exception_precedes_execution() {
    // SDM Vol 3A §15.5.1; Vol 2A §3.3 EMMS/CVTPI2PD/CVTPI2PS and CVT*2PI;
    // Vol 2B §4.3 MOVDQ2Q/MOVQ2DQ. CVTPI2PD xmm,m64 is exempt.
    for opcode in [
        &[0x0f, 0xef, 0xc0][..],
        &[0x0f, 0xef, 0x00],
        &[0x0f, 0x77],
        &[0x0f, 0x2a, 0xc0],
        &[0x0f, 0x2a, 0x00],
        &[0x66, 0x0f, 0x2a, 0xc0],
        &[0x66, 0x0f, 0x2a, 0x00],
        &[0x66, 0x0f, 0x2d, 0xc0],
        &[0x66, 0x0f, 0x2d, 0x00],
        &[0x0f, 0x2d, 0xc0],
        &[0x0f, 0x2d, 0x00],
        &[0x66, 0x0f, 0x2c, 0xc0],
        &[0x66, 0x0f, 0x2c, 0x00],
        &[0x0f, 0x2c, 0xc0],
        &[0x0f, 0x2c, 0x00],
        &[0xf2, 0x0f, 0xd6, 0xc0],
        &[0xf3, 0x0f, 0xd6, 0xc0],
        &[0x9b], // FWAIT positive control.
    ] {
        for backend in backends() {
            let mut vm = Vm::with_backend(VmConfig::flat(0x30_0000), backend);
            vm.map(CODE, 4096, Prot::RWX, RegionKind::Ram).unwrap();
            vm.map(DATA, 4096, Prot::RWX, RegionKind::Ram).unwrap();
            let mut a = CodeAssembler::new(64).unwrap();
            a.fldcw(word_ptr(DATA)).unwrap();
            a.fld1().unwrap();
            a.fldz().unwrap();
            a.fdivp(st1, st0).unwrap();
            a.fxsave(qword_ptr(DATA + 512)).unwrap();
            let ip = CODE + a.assemble(CODE).unwrap().len() as u64;
            a.db(opcode).unwrap();
            a.fxsave(qword_ptr(DATA + 1024)).unwrap();
            a.hlt().unwrap();
            vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
            vm.write_bytes(DATA, &(0x037fu16 & !4).to_le_bytes())
                .unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rax, DATA + 128);
            cpu.set_xmm(0, 0x1234);
            let exit = cpu.run(&vm, None);
            let exempt = opcode == [0x66, 0x0f, 0x2a, 0x00];
            if exempt {
                assert!(matches!(exit, Exit::Hlt), "{opcode:02x?}: {exit:?}");
            } else {
                assert!(
                    matches!(exit, Exit::Exception { vector: 16, addr, .. } if addr == ip),
                    "{opcode:02x?}: {exit:?}"
                );
                assert_eq!(cpu.reg(Reg::Rip), ip);
                assert_eq!(cpu.xmm(0), 0x1234);
                cpu.set_reg(Reg::Rip, ip + opcode.len() as u64);
                assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            }
            assert_eq!(cpu.fpu_top(), 6);
            assert_eq!(cpu.reg(Reg::Rax), DATA + 128);
            let mut before = [0; 512];
            let mut after = [0; 512];
            vm.read_bytes(DATA + 512, &mut before).unwrap();
            vm.read_bytes(DATA + 1024, &mut after).unwrap();
            // The exempt form writes XMM0; all x87 image bytes must survive.
            assert_eq!(&before[..160], &after[..160], "{opcode:02x?}");
        }
    }
}

#[test]
fn mmx_pd_conversion_alignment_preserves_state_on_fault() {
    // SDM Vol 3B §25.25.3 Table 25-4: CVT{T}PD2PI m128 requires 16 bytes.
    // CVTPS2PI m64 is the unaligned control (Table 25-5).
    for opcode in [
        &[0x66, 0x0f, 0x2d, 0x00][..],
        &[0x66, 0x0f, 0x2c, 0x00],
        &[0x0f, 0x2d, 0x00],
    ] {
        for offset in [0, 1, 8, 15] {
            for backend in backends() {
                let mut vm = Vm::with_backend(VmConfig::flat(0x30_0000), backend);
                vm.map(CODE, 4096, Prot::RWX, RegionKind::Ram).unwrap();
                vm.map(DATA, 4096, Prot::RWX, RegionKind::Ram).unwrap();
                let mut a = CodeAssembler::new(64).unwrap();
                a.fxsave(qword_ptr(DATA + 512)).unwrap();
                let ip = CODE + a.assemble(CODE).unwrap().len() as u64;
                a.db(opcode).unwrap();
                a.fxsave(qword_ptr(DATA + 1024)).unwrap();
                a.hlt().unwrap();
                vm.write_bytes(CODE, &a.assemble(CODE).unwrap()).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, CODE);
                cpu.set_reg(Reg::Rax, DATA + offset);
                cpu.set_fpu_top(5);
                cpu.set_xmm(0, u128::MAX);
                for n in 0..8 {
                    cpu.set_fpr_bytes(n, &x86jit_core::f80::F80::from_i64(n as i64 + 1).to_bytes());
                }
                let exit = cpu.run(&vm, None);
                if offset != 0 && opcode[0] == 0x66 {
                    assert!(
                        matches!(exit, Exit::Exception { vector: 13, addr, .. } if addr == ip),
                        "{opcode:02x?} offset={offset}: {exit:?}"
                    );
                    assert_eq!(cpu.reg(Reg::Rip), ip);
                    assert_eq!(cpu.fpu_top(), 5);
                    assert_eq!(cpu.reg(Reg::Rax), DATA + offset);
                    assert_eq!(cpu.xmm(0), u128::MAX);
                    cpu.set_reg(Reg::Rip, ip + opcode.len() as u64);
                    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
                    let mut before = [0; 512];
                    let mut after = [0; 512];
                    vm.read_bytes(DATA + 512, &mut before).unwrap();
                    vm.read_bytes(DATA + 1024, &mut after).unwrap();
                    assert_eq!(before, after);
                } else {
                    assert!(matches!(exit, Exit::Hlt), "{opcode:02x?}: {exit:?}");
                    assert_eq!(cpu.fpu_top(), 0);
                }
            }
        }
    }
}

#[test]
fn mmx_state_images_physical_aliases_and_emms() {
    for emms in [false, true] {
        let mut p = program(|a| {
            a.fldenv(dword_ptr(DATA)).unwrap();
            a.pxor(mm3, mm3).unwrap();
            if emms {
                a.emms().unwrap();
            }
            a.fxsave(qword_ptr(DATA + 512)).unwrap();
            a.fnstenv(dword_ptr(DATA + 128)).unwrap();
        });
        let page = &mut p.mem_init[1].bytes;
        page[..2].copy_from_slice(&0x077fu16.to_le_bytes());
        page[4..6].copy_from_slice(&(0x4500u16 | (5 << 11)).to_le_bytes());
        page[8..10].copy_from_slice(&0xffffu16.to_le_bytes());
        p.cpu_init.st = [x86jit_core::f80::F80::from_i64(1).to_bytes(); 8];
        run(&p, |out| {
            let bytes = data(out);
            assert_eq!(word(bytes, 512), 0x077f);
            assert_eq!(word(bytes, 514), 0x4500);
            assert_eq!(bytes[516], if emms { 0 } else { 0xff });
            let tag = word(bytes, 136);
            assert_eq!(tag, if emms { 0xffff } else { 2 << 6 });
            assert_eq!(
                &bytes[512 + 32 + 3 * 16..512 + 32 + 3 * 16 + 10],
                &[0, 0, 0, 0, 0, 0, 0, 0, 255, 255]
            );
            for n in [0, 1, 2, 4, 5, 6, 7] {
                assert_eq!(
                    &bytes[512 + 32 + n * 16..512 + 42 + n * 16],
                    &x86jit_core::f80::F80::from_i64(1).to_bytes()
                );
            }
        });
    }
}

#[test]
fn mmx_read_transitions_preserve_exponents() {
    let mut p = program(|a| {
        a.movq(rax, mm5).unwrap();
        a.movq2dq(xmm2, mm1).unwrap();
        a.fxsave(qword_ptr(DATA + 512)).unwrap();
    });
    p.cpu_init.fpu_top = 3;
    for (n, slot) in p.cpu_init.st.iter_mut().enumerate() {
        *slot = x86jit_core::f80::F80::from_i64(n as i64 + 1).to_bytes();
    }
    let physical5 = p.cpu_init.st[2];
    let physical1 = p.cpu_init.st[6];
    run(&p, |out| {
        assert_eq!(out.cpu.fpu_top, 0);
        assert_eq!(
            out.cpu.gpr[0],
            u64::from_le_bytes(physical5[..8].try_into().unwrap())
        );
        assert_eq!(
            out.cpu.xmm[2],
            u64::from_le_bytes(physical1[..8].try_into().unwrap()) as u128
        );
        assert_eq!(data(out)[516], 0xff);
        for n in 0..8 {
            assert_eq!(out.cpu.st[n], p.cpu_init.st[(n + 5) & 7]);
        }
    });
}

#[test]
fn mmx_emms_allows_x87_push_and_missing_emms_overflows() {
    // SDM Vol 1 §8.5.1.1: masked stack overflow sets IE/SF/C1 and pushes indefinite.
    for emms in [false, true] {
        let p = program(|a| {
            a.pxor(mm0, mm0).unwrap();
            if emms {
                a.emms().unwrap();
            }
            a.fld1().unwrap();
            a.fxsave(qword_ptr(DATA + 512)).unwrap();
            a.fnstenv(dword_ptr(DATA + 128)).unwrap();
        });
        run(&p, |out| {
            let sw = word(data(out), 132);
            assert_eq!(sw & 0x0241, if emms { 0 } else { 0x0241 });
            assert_eq!(out.cpu.fpu_top, 7);
            assert_eq!(
                out.cpu.st[0],
                if emms {
                    x86jit_core::f80::F80::from_i64(1).to_bytes()
                } else {
                    [0, 0, 0, 0, 0, 0, 0, 192, 255, 255]
                }
            );
            assert_eq!(data(out)[516], if emms { 0x80 } else { 0xff });
        });
    }
}

#[test]
fn mmx_integer_family_witnesses() {
    // SDM Vol 2B §4.3: PXOR/PADDSB/PADDUSB/PMADDWD/PACKSSWB/PALIGNR/PSHUFB.
    let mut p = program(|a| {
        a.movq(mm0, qword_ptr(DATA)).unwrap();
        a.movq(mm1, qword_ptr(DATA + 8)).unwrap();
        a.paddsb(mm0, mm1).unwrap();
        a.movq(qword_ptr(DATA + 64), mm0).unwrap();
        a.movq(mm0, qword_ptr(DATA)).unwrap();
        a.paddusb(mm0, mm1).unwrap();
        a.movq(qword_ptr(DATA + 72), mm0).unwrap();
        a.movq(mm0, qword_ptr(DATA + 16)).unwrap();
        a.pmaddwd(mm0, mm0).unwrap();
        a.movq(qword_ptr(DATA + 80), mm0).unwrap();
        a.movq(mm0, qword_ptr(DATA + 24)).unwrap();
        a.packsswb(mm0, mm0).unwrap();
        a.movq(qword_ptr(DATA + 88), mm0).unwrap();
        a.pxor(mm0, mm0).unwrap();
        a.movq(qword_ptr(DATA + 96), mm0).unwrap();
    });
    let b = &mut p.mem_init[1].bytes;
    b[..8].copy_from_slice(&[127, 128, 255, 1, 0, 100, 200, 250]);
    b[8..16].copy_from_slice(&[1, 255, 1, 255, 0, 100, 100, 10]);
    b[16..24].copy_from_slice(&0x8000_8000_8000_8000u64.to_le_bytes());
    b[24..32].copy_from_slice(&0x7fff_8000_0100_ff00u64.to_le_bytes());
    run(&p, |out| {
        let b = data(out);
        assert_eq!(&b[64..72], &[127, 128, 0, 0, 0, 127, 44, 4]);
        assert_eq!(&b[72..80], &[128, 255, 255, 255, 0, 200, 255, 255]);
        assert_eq!(qword(b, 80), 0x8000_0000_8000_0000);
        assert_eq!(qword(b, 88), 0x7f80_7f80_7f80_7f80);
        assert_eq!(qword(b, 96), 0);
    });
}

#[test]
fn mmx_faults_commit_no_state_and_load_exact_width() {
    const SPAN: u64 = 0x30_0000;
    for (opcode, width) in [
        (&[0x0f, 0x6f, 0x00][..], 8),
        (&[0x0f, 0xef, 0x00], 8),
        (&[0x0f, 0x7f, 0x00], 8),
        (&[0x0f, 0xe7, 0x00], 8),
        (&[0x0f, 0x6e, 0x00], 4),
        (&[0x0f, 0x60, 0x00], 4),
        (&[0x0f, 0xc4, 0x00, 0x00], 2),
        (&[0x66, 0x0f, 0x2d, 0x00], 16),
        (&[0x66, 0x0f, 0x2c, 0x00], 16),
        (&[0x0f, 0x2a, 0x00], 8),
        (&[0x66, 0x0f, 0x2a, 0x00], 8),
    ] {
        for b in backends() {
            let mut vm = Vm::with_backend(VmConfig::flat(SPAN), b);
            vm.map(0, SPAN as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
            let mut a = CodeAssembler::new(64).unwrap();
            a.fxsave(qword_ptr(DATA)).unwrap();
            a.db(opcode).unwrap();
            a.fxsave(qword_ptr(DATA + 512)).unwrap();
            a.hlt().unwrap();
            let code = a.assemble(CODE).unwrap();
            let mut decoder = iced_x86::Decoder::with_ip(64, &code, CODE, 0);
            let _ = decoder.decode();
            let fault_ip = decoder.decode().ip();
            let tail = decoder.decode().ip();
            vm.write_bytes(CODE, &code).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            // Aligned m128 must fault on memory, rather than on alignment.
            cpu.set_reg(Reg::Rax, if width == 16 { SPAN } else { SPAN - width / 2 });
            cpu.set_fpu_top(5);
            for n in 0..8 {
                cpu.set_fpr_bytes(n, &x86jit_core::f80::F80::from_i64(n as i64 + 1).to_bytes());
            }
            assert!(matches!(cpu.run(&vm, None), Exit::UnmappedMemory { .. }));
            assert_eq!(cpu.reg(Reg::Rip), fault_ip);
            assert_eq!(cpu.fpu_top(), 5);
            cpu.set_reg(Reg::Rip, tail);
            assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
            let mut before = [0; 512];
            let mut after = [0; 512];
            vm.read_bytes(DATA, &mut before).unwrap();
            vm.read_bytes(DATA + 512, &mut after).unwrap();
            assert_eq!(before, after);
        }
    }
    // SDM Vol 2B §4.3 MOVD/MOVQ and PUNPCKL*: m32, not m64.
    for opcode in [
        &[0x0f, 0x6e, 0x00][..],
        &[0x0f, 0x60, 0x00],
        &[0x0f, 0x61, 0x00],
        &[0x0f, 0x62, 0x00],
    ] {
        for b in backends() {
            let mut vm = Vm::with_backend(VmConfig::flat(SPAN), b);
            vm.map(0, SPAN as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
            let mut code = opcode.to_vec();
            code.push(0xf4);
            vm.write_bytes(CODE, &code).unwrap();
            vm.write_bytes(SPAN - 4, &0xffff_ffffu32.to_le_bytes())
                .unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rax, SPAN - 4);
            assert!(
                matches!(cpu.run(&vm, None), Exit::Hlt),
                "m32 boundary {opcode:02x?}"
            );
        }
    }
}

#[test]
fn mmx_all_forms_execute_inside_regions() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use x86jit_core::{CachedBlock, IrBlock, IrRegion, MemConsistency};
    struct Regions {
        jit: JitBackend,
        count: Arc<AtomicUsize>,
    }
    impl Backend for Regions {
        fn region_caps(&self) -> Option<RegionCaps> {
            Some(RegionCaps {
                max_blocks: 8,
                max_icount: 128,
            })
        }
        fn materialize(
            &self,
            ir: &IrBlock,
            c: MemConsistency,
            m: Option<(u64, u64)>,
            b: u64,
        ) -> CachedBlock {
            self.jit.materialize(ir, c, m, b)
        }
        fn materialize_region(
            &self,
            ir: &IrRegion,
            c: MemConsistency,
            m: Option<(u64, u64)>,
            b: u64,
        ) -> CachedBlock {
            assert!(ir.has_loop);
            let block = self.jit.materialize_region(ir, c, m, b);
            assert!(matches!(block, CachedBlock::Compiled { .. }));
            self.count.fetch_add(1, Ordering::Relaxed);
            block
        }
    }
    for &code in mmx::CODES {
        let mut p = mmx::input(&[code], 31, true, CpuMode::Long64);
        let mut decoder = iced_x86::Decoder::with_ip(64, &p.mem_init[0].bytes, CODE, 0);
        let len = decoder.decode().len();
        let mut a = CodeAssembler::new(64).unwrap();
        let mut top = a.create_label();
        a.mov(r15, 3u64).unwrap();
        a.jmp(top).unwrap();
        a.set_label(&mut top).unwrap();
        a.db(&p.mem_init[0].bytes[..len]).unwrap();
        a.dec(r15).unwrap();
        a.jnz(top).unwrap();
        a.fxsave(qword_ptr(mmx::IMAGE)).unwrap();
        a.hlt().unwrap();
        p.mem_init[0].bytes = a.assemble(CODE).unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let mut region = run_with_backend_mode(
            &p,
            Box::new(Regions {
                jit: JitBackend::new(),
                count: count.clone(),
            }),
            GuestCpuFeatures::v2(),
            CpuMode::Long64,
        );
        let mut interp = run_with_backend_mode(
            &p,
            Box::new(InterpreterBackend),
            GuestCpuFeatures::v2(),
            CpuMode::Long64,
        );
        assert_eq!(region.exit, ExitKind::Hlt, "{code:?}");
        mmx::normalize(&mut region, false);
        mmx::normalize(&mut interp, false);
        assert!(
            x86jit_tests::compare::compare(&interp, &region, &[]).is_none(),
            "{code:?}"
        );
        assert!(
            count.load(Ordering::Relaxed) > 0,
            "{code:?} did not compile a region"
        );
    }
}

#[test]
fn mmx_emms_alone_empties_tags_and_resets_top() {
    let mut p = program(|a| {
        a.fldenv(dword_ptr(DATA)).unwrap();
        a.emms().unwrap();
        a.fxsave(qword_ptr(DATA + 512)).unwrap();
        a.fnstenv(dword_ptr(DATA + 128)).unwrap();
    });
    p.mem_init[1].bytes[..2].copy_from_slice(&0x037fu16.to_le_bytes());
    p.mem_init[1].bytes[4..6].copy_from_slice(&(5u16 << 11).to_le_bytes());
    p.cpu_init.st = [x86jit_core::f80::F80::from_i64(1).to_bytes(); 8];
    run(&p, |out| {
        assert_eq!(out.cpu.fpu_top, 0);
        assert_eq!(data(out)[516], 0);
        assert_eq!(word(data(out), 136), 0xffff);
    });
}

#[test]
fn mmx_conversions_rounding_and_xmm_preservation() {
    // SDM Vol 2A §3.3 CVTPI2PS: MXCSR.RC; preserve bits 127:64 and above.
    for rc in 0..4 {
        let mut p = program(|a| {
            a.movq(mm0, qword_ptr(DATA)).unwrap();
            a.cvtpi2ps(xmm0, mm0).unwrap();
            a.cvtps2pi(mm1, xmm0).unwrap();
            a.cvttps2pi(mm2, xmm0).unwrap();
        });
        p.cpu_init.mxcsr = 0x1f80 | (rc << 13);
        p.cpu_init.xmm[0] = 0x1234_5678_90ab_cdefu128 << 64;
        p.cpu_init.ymm_hi[0] = u128::MAX;
        let ints = 16777217u64 | ((-16777217i32 as u32 as u64) << 32);
        p.mem_init[1].bytes[..8].copy_from_slice(&ints.to_le_bytes());
        let positive = if rc == 2 { 0x4b80_0001u32 } else { 0x4b80_0000 };
        let negative = if rc == 1 { 0xcb80_0001u32 } else { 0xcb80_0000 };
        run(&p, |out| {
            assert_eq!(
                out.cpu.xmm[0],
                p.cpu_init.xmm[0] | u128::from(positive) | (u128::from(negative) << 32)
            );
            assert_eq!(out.cpu.ymm_hi[0], u128::MAX);
        });
    }
}

#[test]
fn mmx_cvtpi2pd_memory_preserves_x87_state() {
    // SDM Vol 2A §3.3 CVTPI2PD explicitly exempts xmm,m64 from the MMX transition.
    let mut p = program(|a| {
        a.cvtpi2pd(xmm0, qword_ptr(DATA)).unwrap();
        a.fxsave(qword_ptr(DATA + 512)).unwrap();
    });
    p.cpu_init.fpu_top = 5;
    p.mem_init[1].bytes[..8].copy_from_slice(&0xffff_fffe_0000_0003u64.to_le_bytes());
    run(&p, |out| {
        assert_eq!(out.cpu.fpu_top, 5);
        assert_eq!(data(out)[516], 0);
        assert_eq!(
            out.cpu.xmm[0],
            u128::from(3.0f64.to_bits()) | (u128::from((-2.0f64).to_bits()) << 64)
        );
    });
}

#[test]
fn mmx_comparator_observes_tags_top_and_all_register_bits() {
    let p = mmx::input(&[iced_x86::Code::Pxor_mm_mmm64], 1, false, CpuMode::Long64);
    let baseline = run_with_backend_mode(
        &p,
        Box::new(InterpreterBackend),
        GuestCpuFeatures::v2(),
        CpuMode::Long64,
    );
    assert_eq!(baseline.exit, ExitKind::Hlt);
    for offset in [0x802, 0x804, 0x820, 0x828] {
        let mut a = baseline.clone();
        let mut b = baseline.clone();
        b.mem[1].bytes[offset] ^= 1;
        mmx::normalize(&mut a, false);
        mmx::normalize(&mut b, false);
        assert!(
            x86jit_tests::compare::compare(&a, &b, &[]).is_some(),
            "uncompared FXSAVE byte {offset:#x}"
        );
    }
    let mut a = baseline.clone();
    let mut b = baseline.clone();
    b.mem[1].bytes[0x828] ^= 1;
    b.cpu.st[0][8] ^= 1;
    mmx::normalize(&mut a, true);
    mmx::normalize(&mut b, true);
    assert!(x86jit_tests::compare::compare(&a, &b, &[]).is_none());
    b.mem[1].bytes[0x804] ^= 1;
    assert!(x86jit_tests::compare::compare(&a, &b, &[]).is_some());
}

#[test]
fn mmx_movd_clears_the_upper_dword_in_both_directions() {
    // SDM Vol 2B §4.3 MOVD/MOVQ: MOVD zero-extends its 32-bit input into MMn.
    let mut p = program(|a| {
        a.movd(mm0, eax).unwrap();
        a.movq(rdx, mm0).unwrap();
        a.movd(ebx, mm0).unwrap();
    });
    p.cpu_init.gpr[0] = 0xdead_beef_1234_5678;
    p.cpu_init.gpr[3] = u64::MAX;
    run(&p, |out| {
        assert_eq!(out.cpu.gpr[2], 0x1234_5678);
        assert_eq!(out.cpu.gpr[3], 0x1234_5678);
        assert_eq!(&out.cpu.st[0][8..], &[255, 255]);
    });
}
