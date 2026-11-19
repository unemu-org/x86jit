//! SDM Vol 1 §15.6.1: merge/zero writemasks; Vol 2 §2.7.7: embedded broadcast.
use iced_x86::{Decoder, DecoderOptions, Encoder, Instruction};
use x86jit_core::{Backend, Exit, InterpreterBackend, Prot, Reg, RegionKind, Vm, VmConfig};
use x86jit_cranelift::JitBackend;
use x86jit_tests::{
    compare::compare,
    oracle::{run_with_backend_mode, RunOutcome, VectorInput},
    vector::{CpuSnapshot, ExitKind, MemChunk, MemKind, RunSpec},
};

const CODE: u64 = 0x1_0000;
const END: u64 = 0x2_0000;

macro_rules! check_eq {
    ($actual:expr, $expected:expr $(,)?) => {
        check_eq!($actual, $expected, "")
    };
    ($actual:expr, $expected:expr, $($context:tt)+) => {
        match (&$actual, &$expected) {
            (actual, expected) => assert!(
                actual == expected,
                "{} expected={expected:x?} engine={actual:x?}; {}",
                stringify!($actual),
                format_args!($($context)+),
            ),
        }
    };
}

fn instruction_context(i: &Instruction, mask: u64) -> String {
    format!(
        "{:?} {:?},{:?} {:?}={mask:#x} z={} bcst={}",
        i.mnemonic(),
        i.op0_register(),
        i.op1_register(),
        i.op_mask(),
        i.zeroing_masking(),
        i.is_broadcast(),
    )
}

fn check_hlt(exit: Exit, context: &str) {
    assert!(
        matches!(exit, Exit::Hlt),
        "exit expected=Hlt engine={exit:?}; {context}"
    );
}

fn check_vector(actual: [u128; 4], expected: [u128; 4], reg: usize, context: &str) {
    for (chunk, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert!(
            actual == expected,
            "zmm{reg} chunk={chunk} expected={expected:#x} engine={actual:#x}; {context}"
        );
    }
}

fn lifted(i: &iced_x86::Instruction) -> Option<String> {
    use x86jit_core::{
        lift::{lift_one, FetchAddr},
        CpuMode, Memory, MemoryModel,
    };
    let mut encoder = iced_x86::Encoder::new(64);
    encoder.encode(i, CODE).ok()?;
    let mut mem = Memory::new(MemoryModel::Flat { size: END });
    mem.map(0, END as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    mem.write_bytes(CODE, &encoder.take_buffer()).unwrap();
    lift_one(&mem, FetchAddr::flat(CODE), CpuMode::Long64)
        .ok()
        .map(|b| {
            b.ops
                .iter()
                .map(|op| {
                    format!("{op:?}").replace(&format!("{:?}", op.memory_access()), "<access>")
                })
                .collect::<Vec<_>>()
                .join(";")
        })
}

#[test]
fn evex_modifier_sweep_never_silently_drops_bits() {
    let mut masked = std::collections::BTreeSet::new();
    let mut broadcast = std::collections::BTreeSet::new();
    let forms = x86jit_tests::compat::evex_modifier_forms();
    assert!(forms.len() > 3000, "incomplete EVEX sweep: {}", forms.len());
    let codes: std::collections::BTreeSet<_> = forms.iter().map(|i| i.code()).collect();
    let unencodable: Vec<_> = iced_x86::Code::values()
        .filter(|c| {
            let info = c.op_code();
            info.encoding() == iced_x86::EncodingKind::EVEX
                && info.mode64()
                && (info.can_use_op_mask_register() || info.can_broadcast())
                && !codes.contains(c)
        })
        .collect();
    eprintln!(
        "EVEX modifier sweep: {} forms, {} Codes; unencodable: {unencodable:?}",
        forms.len(),
        codes.len()
    );
    assert!(
        unencodable.is_empty(),
        "missing EVEX modifier templates: {unencodable:?}"
    );
    for i in forms {
        let Some(actual) = lifted(&i) else {
            continue;
        };
        let mut plain = i;
        if i.has_op_mask() {
            plain.set_op_mask(iced_x86::Register::None);
            plain.set_zeroing_masking(false);
            if lifted(&plain).as_ref() == Some(&actual) {
                masked.insert(format!("{:?}", i.mnemonic()));
            }
        }
        if i.is_broadcast() {
            plain = i;
            plain.set_is_broadcast(false);
            if lifted(&plain).as_ref() == Some(&actual) {
                broadcast.insert(format!("{:?}", i.mnemonic()));
            }
        }
    }
    assert!(
        masked.is_empty() && broadcast.is_empty(),
        "ignored writemask ({}): {masked:?}\nignored broadcast ({}): {broadcast:?}",
        masked.len(),
        broadcast.len()
    );
}

fn backends() -> Vec<Box<dyn Backend>> {
    vec![
        Box::new(InterpreterBackend),
        Box::new(JitBackend::new()),
        Box::new(JitBackend::with_superblocks(x86jit_core::RegionCaps {
            max_blocks: 16,
            max_icount: 256,
        })),
    ]
}

fn encode(i: &Instruction) -> Vec<u8> {
    let mut e = Encoder::new(64);
    e.encode(i, CODE).unwrap();
    let bytes = e.take_buffer();
    let decoded = Decoder::with_ip(64, &bytes, CODE, DecoderOptions::NONE).decode();
    check_eq!(decoded.code(), i.code());
    check_eq!(decoded.op_mask(), i.op_mask());
    check_eq!(decoded.is_broadcast(), i.is_broadcast());
    bytes
}

fn random(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn random_registers(mut i: Instruction, seed: &mut u64) -> Instruction {
    let dst = (random(seed) % 32) as usize;
    for op in 0..i.op_count() {
        let old = i.op_register(op);
        if i.op_kind(op) == iced_x86::OpKind::Register && old.is_vector_register() {
            let number = if op == 0 || random(seed) & 1 == 0 {
                dst
            } else {
                (random(seed) % 32) as usize
            };
            let base = old as usize - old.number();
            i.set_op_register(op, iced_x86::Register::try_from(base + number).unwrap());
        }
    }
    i
}

fn sample(i: &Instruction, seed: &mut u64) -> VectorInput {
    let mut cpu = CpuSnapshot {
        rip: CODE,
        ..Default::default()
    };
    cpu.gpr[0] = 0x1_8000u64.wrapping_sub(i.memory_displacement64());
    for r in 0..32 {
        let mut lane = || u128::from(random(seed)) | (u128::from(random(seed)) << 64);
        cpu.xmm[r] = lane();
        cpu.ymm_hi[r] = lane();
        cpu.zmm_hi[r] = [lane(), lane()];
    }
    for k in &mut cpu.kmask {
        *k = random(seed);
    }
    let mut bytes = encode(i);
    bytes.push(0xf4);
    VectorInput {
        cpu_init: cpu,
        entry: CODE,
        run: RunSpec::UntilExit,
        mem_init: vec![
            MemChunk {
                addr: CODE,
                bytes,
                kind: MemKind::Ram,
            },
            MemChunk {
                addr: 0x1_8000,
                bytes: (0..64).map(|_| random(seed) as u8).collect(),
                kind: MemKind::Ram,
            },
        ],
    }
}

fn run(input: &VectorInput, jit: bool) -> RunOutcome {
    run_with_backend_mode(
        input,
        if jit {
            Box::new(JitBackend::new())
        } else {
            Box::new(InterpreterBackend)
        },
        x86jit_core::GuestCpuFeatures::default(),
        x86jit_core::CpuMode::Long64,
    )
}

fn equal(mut expected: RunOutcome, mut actual: RunOutcome, context: &str) {
    let label = if context.starts_with("native/") {
        "native"
    } else {
        "expected"
    };
    assert!(
        expected.exit == ExitKind::Hlt && actual.exit == ExitKind::Hlt,
        "exit {label}={:?} engine={:?}; {context}; expected=Hlt",
        expected.exit,
        actual.exit
    );
    expected.cpu.rip = 0;
    actual.cpu.rip = 0;
    // The transformed references execute different instruction bytes.
    expected.mem.retain(|c| c.addr != CODE);
    actual.mem.retain(|c| c.addr != CODE);
    if let Some(diff) = compare(&expected, &actual, &[]) {
        let first = diff.to_string();
        let first = first
            .lines()
            .next()
            .unwrap()
            .trim()
            .replace("expected ", &format!("{label}="))
            .replace("  got ", " engine=");
        panic!("{first}; {context}");
    }
}

#[test]
fn evex_divergence_and_instruction_fit_the_status_prefix() {
    use iced_x86::{Code, Register};
    let mut i = Instruction::with2(
        Code::EVEX_Vmovupd_xmm_k1z_xmmm128,
        Register::XMM30,
        Register::XMM30,
    )
    .unwrap();
    i.set_op_mask(Register::K1);
    i.set_zeroing_masking(true);
    let context = instruction_context(&i, 1);
    let expected = RunOutcome {
        cpu: CpuSnapshot::default(),
        mem: vec![MemChunk {
            addr: 0x1_8000,
            bytes: vec![0],
            kind: MemKind::Ram,
        }],
        exit: ExitKind::Hlt,
    };
    for memory in [false, true] {
        let mut actual = expected.clone();
        let prefix = if memory {
            actual.mem[0].bytes[0] = 1;
            "mem 0x18000: native=0x00 engine=0x01;"
        } else {
            actual.cpu.xmm[30] = u128::MAX;
            "xmm30: native=0x00000000000000000000000000000000 engine=0xffffffffffffffffffffffffffffffff;"
        };
        let failure = std::panic::catch_unwind(|| {
            equal(
                expected.clone(),
                actual,
                &format!("native/interp {context}"),
            );
        })
        .expect_err("injected divergence must fail");
        let message = failure.downcast_ref::<String>().unwrap();
        let status: String = message.chars().take(240).collect();
        assert!(
            status.starts_with(prefix),
            "diagnostic expected={prefix:?} engine={status:?}"
        );
        assert!(
            status.contains(&context),
            "diagnostic expected={context:?} engine={status:?}"
        );
    }
}

fn clear_upper(cpu: &mut CpuSnapshot, i: &Instruction) {
    if i.op0_register().is_vector_register() {
        let dst = i.op0_register().number();
        if i.op0_register().size() < 32 {
            cpu.ymm_hi[dst] = 0;
        }
        if i.op0_register().size() < 64 {
            cpu.zmm_hi[dst] = [0; 2];
        }
    }
}

#[test]
fn evex_random_masks_and_broadcast_match_jit_and_expanded_memory() {
    let mut seed = 0xa4093822299f31d0;
    let mut checked = 0;
    let mut broadcasts = 0;
    for i in x86jit_tests::compat::evex_modifier_forms() {
        if lifted(&i).is_none() {
            continue;
        }
        let changed = random_registers(i, &mut seed);
        let i = if lifted(&changed).is_some() {
            changed
        } else {
            i
        };
        let input = sample(&i, &mut seed);
        let expected = run(&input, false);
        equal(
            expected.clone(),
            run(&input, true),
            &instruction_context(&i, input.cpu_init.kmask[1]),
        );
        if i.is_broadcast() {
            let mut plain = i;
            plain.set_is_broadcast(false);
            let mut expanded = input.clone();
            let width = i.memory_size().size();
            let scalar = expanded.mem_init[1].bytes[..width].to_vec();
            expanded.mem_init[1].bytes = scalar.iter().copied().cycle().take(64).collect();
            expanded.mem_init[0].bytes = encode(&plain);
            expanded.mem_init[0].bytes.push(0xf4);
            // Some masked cross-lane memory forms are deliberately unsupported;
            // the unmasked comparison still witnesses their broadcast value.
            if lifted(&plain).is_some() {
                let mut reference = run(&expanded, false);
                // Only the computation is compared: the reference expands scalar data.
                reference.mem = expected.mem.clone();
                clear_upper(&mut reference.cpu, &i);
                equal(
                    reference,
                    expected.clone(),
                    &format!(
                        "expanded {}",
                        instruction_context(&i, input.cpu_init.kmask[1])
                    ),
                );
            }
            broadcasts += 1;
        }
        if i.has_op_mask()
            && i.op0_register().is_vector_register()
            && lifted(&i).unwrap().contains("VReadChunk")
        {
            let mut plain = i;
            plain.set_op_mask(iced_x86::Register::None);
            plain.set_zeroing_masking(false);
            if lifted(&plain).is_none() {
                checked += 1;
                continue;
            }
            let mut unmasked = input.clone();
            unmasked.mem_init[0].bytes = encode(&plain);
            unmasked.mem_init[0].bytes.push(0xf4);
            let mut reference = run(&unmasked, false);
            let elem = match i.mnemonic() {
                iced_x86::Mnemonic::Vpacksswb | iced_x86::Mnemonic::Vpackuswb => 1,
                iced_x86::Mnemonic::Vpackssdw | iced_x86::Mnemonic::Vpackusdw => 2,
                _ => i.memory_size().element_size(),
            };
            let dst = i.op0_register().number();
            let width = i.op0_register().size();
            let mut lanes = [
                reference.cpu.xmm[dst],
                reference.cpu.ymm_hi[dst],
                reference.cpu.zmm_hi[dst][0],
                reference.cpu.zmm_hi[dst][1],
            ];
            let merge = if matches!(
                i.mnemonic(),
                iced_x86::Mnemonic::Vpblendmd | iced_x86::Mnemonic::Vpblendmq
            ) {
                i.op1_register().number()
            } else {
                dst
            };
            let original = [
                input.cpu_init.xmm[merge],
                input.cpu_init.ymm_hi[merge],
                input.cpu_init.zmm_hi[merge][0],
                input.cpu_init.zmm_hi[merge][1],
            ];
            for byte in 0..width {
                if input.cpu_init.kmask[1] & (1 << (byte / elem)) == 0 {
                    let shift = (byte % 16) * 8;
                    let selected = if i.zeroing_masking() {
                        0
                    } else {
                        original[byte / 16]
                    };
                    lanes[byte / 16] =
                        (lanes[byte / 16] & !(0xff << shift)) | (selected & (0xff << shift));
                }
            }
            reference.cpu.xmm[dst] = lanes[0];
            reference.cpu.ymm_hi[dst] = lanes[1];
            reference.cpu.zmm_hi[dst] = [lanes[2], lanes[3]];
            clear_upper(&mut reference.cpu, &i);
            equal(
                reference,
                expected,
                &format!(
                    "independent mask {}",
                    instruction_context(&i, input.cpu_init.kmask[1])
                ),
            );
        }
        checked += 1;
    }
    assert!(
        checked > 1000 && broadcasts > 100,
        "checked={checked}, broadcasts={broadcasts}"
    );
    eprintln!("EVEX differential samples: {checked}, broadcasts: {broadcasts}");
}

fn native_samples() -> impl Iterator<Item = (Instruction, VectorInput)> {
    use iced_x86::CpuidFeature::*;
    let mut seed = 0x13198a2e03707344;
    x86jit_tests::compat::evex_modifier_forms()
        .into_iter()
        .filter(|i| {
            i.code()
                .cpuid_features()
                .iter()
                .all(|f| matches!(f, AVX512F | AVX512VL | AVX512BW | AVX512DQ))
                && lifted(i).is_some()
        })
        .map(move |i| {
            let changed = random_registers(i, &mut seed);
            let i = if lifted(&changed).is_some() {
                changed
            } else {
                i
            };
            let mut input = sample(&i, &mut seed);
            // The native harness rejects the SYSCALL byte pair in every memory
            // chunk, including data. Keep random fixture bytes clear of that pair.
            let data_byte = |b| if b == 5 { 0x85 } else { b };
            for v in input
                .cpu_init
                .xmm
                .iter_mut()
                .chain(input.cpu_init.ymm_hi.iter_mut())
                .chain(input.cpu_init.zmm_hi.iter_mut().flatten())
            {
                *v = u128::from_le_bytes(v.to_le_bytes().map(data_byte));
            }
            for k in &mut input.cpu_init.kmask {
                *k = u64::from_le_bytes(k.to_le_bytes().map(data_byte));
            }
            for b in &mut input.mem_init[1].bytes {
                *b = data_byte(*b);
            }
            (i, input)
        })
}

fn guest_vector_setup(mut input: VectorInput) -> VectorInput {
    use iced_x86::{Code, MemoryOperand, Register};
    const STATE: u64 = 0x1_9000;
    let mut setup = Vec::new();
    let mut data = Vec::new();
    // SDM Vol 2B §4.3, VMOVDQU64; Vol 2A §3.3, KMOVQ: establish the
    // complete vector/opmask state through instructions run by both engines.
    for r in 0..32 {
        setup.extend(encode(
            &Instruction::with2(
                Code::EVEX_Vmovdqu64_zmm_k1z_zmmm512,
                Register::try_from(Register::ZMM0 as usize + r).unwrap(),
                MemoryOperand::with_base_displ(Register::None, (STATE + data.len() as u64) as i64),
            )
            .unwrap(),
        ));
        for lane in [
            input.cpu_init.xmm[r],
            input.cpu_init.ymm_hi[r],
            input.cpu_init.zmm_hi[r][0],
            input.cpu_init.zmm_hi[r][1],
        ] {
            data.extend(lane.to_le_bytes());
        }
    }
    for k in 0..8 {
        setup.extend(encode(
            &Instruction::with2(
                Code::VEX_Kmovq_kr_km64,
                Register::try_from(Register::K0 as usize + k).unwrap(),
                MemoryOperand::with_base_displ(Register::None, (STATE + data.len() as u64) as i64),
            )
            .unwrap(),
        ));
        data.extend(input.cpu_init.kmask[k].to_le_bytes());
    }
    setup.extend_from_slice(&input.mem_init[0].bytes);
    input.mem_init[0].bytes = setup;
    input.mem_init.push(MemChunk {
        addr: STATE,
        bytes: data,
        kind: MemKind::Ram,
    });
    input.cpu_init.xmm.fill(0);
    input.cpu_init.ymm_hi.fill(0);
    input.cpu_init.zmm_hi.fill([0; 2]);
    input.cpu_init.kmask.fill(0);
    input
}

#[test]
fn evex_native_samples_establish_extended_state_in_guest() {
    let mut checked = 0;
    for (i, input) in native_samples() {
        let setup = guest_vector_setup(input.clone());
        let context = instruction_context(&i, input.cpu_init.kmask[1]);
        for r in 0..32 {
            check_vector(
                [
                    setup.cpu_init.xmm[r],
                    setup.cpu_init.ymm_hi[r],
                    setup.cpu_init.zmm_hi[r][0],
                    setup.cpu_init.zmm_hi[r][1],
                ],
                [0; 4],
                r,
                &context,
            );
        }
        for k in 0..8 {
            check_eq!(setup.cpu_init.kmask[k], 0, "k{k}; {context}");
        }
        for chunk in &setup.mem_init {
            for (offset, pair) in chunk.bytes.windows(2).enumerate() {
                assert!(
                    pair != [0x0f, 0x05],
                    "mem@{:#x} expected=no-syscall engine={pair:02x?}; {context}",
                    chunk.addr + offset as u64
                );
            }
        }
        if checked == 0 {
            let mut expected = input.clone();
            expected.mem_init[0].bytes = vec![0xf4];
            let setup = guest_vector_setup(expected.clone());
            let mut reference = run(&expected, false);
            reference.mem = setup.mem_init.clone();
            for backend in backends() {
                equal(
                    reference.clone(),
                    run_with_backend_mode(
                        &setup,
                        backend,
                        x86jit_core::GuestCpuFeatures::default(),
                        x86jit_core::CpuMode::Long64,
                    ),
                    "guest vector setup",
                );
            }
        }
        checked += 1;
    }
    assert!(checked > 1000, "incomplete native samples: {checked}");
    eprintln!("Native-compatible EVEX samples: {checked}");
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn evex_modifiers_match_native() {
    if !(std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512vl")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512dq"))
    {
        return;
    }
    for (i, input) in native_samples() {
        let context = instruction_context(&i, input.cpu_init.kmask[1]);
        let input = guest_vector_setup(input);
        let native = x86jit_tests::native::run_native(&input)
            .unwrap_or_else(|| panic!("native capture unavailable; {context}"));
        equal(
            native.clone(),
            run(&input, false),
            &format!("native/interp {context}"),
        );
        equal(native, run(&input, true), &format!("native/JIT {context}"));
    }
}

fn vm(bytes: &[u8], backend: Box<dyn Backend>) -> Vm {
    let mut vm = Vm::with_backend(VmConfig::flat(END), backend);
    vm.map(0, END as usize, Prot::RWX, RegionKind::Ram).unwrap();
    let mut looped = vec![0xb9, 3, 0, 0, 0, 0xeb, 0];
    looped.extend_from_slice(&bytes[..bytes.len() - 1]);
    looped.extend_from_slice(&[0xff, 0xc9, 0x75, (-(bytes.len() as i8 + 3)) as u8, 0xf4]);
    vm.write_bytes(CODE, &looped).unwrap();
    vm
}

#[test]
fn evex_double_moves_mask_whole_qwords_interp() {
    double_moves_mask_whole_qwords(0);
}

#[test]
fn evex_double_moves_mask_whole_qwords_jit() {
    double_moves_mask_whole_qwords(1);
}

#[test]
fn evex_double_moves_mask_whole_qwords_region() {
    double_moves_mask_whole_qwords(2);
}

fn double_moves_mask_whole_qwords(selected_tier: usize) {
    use iced_x86::{
        Code::*,
        Register::{K1, XMM30, YMM30, ZMM30},
    };
    // SDM Vol 2A §2.7.4; Vol 2B §4.3, MOVAPD/MOVUPD: each mask bit
    // selects a whole qword; register destinations clear bits MAXVL-1:VL.
    for (code, reg, width) in [
        (EVEX_Vmovupd_xmm_k1z_xmmm128, XMM30, 16),
        (EVEX_Vmovupd_ymm_k1z_ymmm256, YMM30, 32),
        (EVEX_Vmovupd_zmm_k1z_zmmm512, ZMM30, 64),
        (EVEX_Vmovapd_xmm_k1z_xmmm128, XMM30, 16),
        (EVEX_Vmovapd_ymm_k1z_ymmm256, YMM30, 32),
        (EVEX_Vmovapd_zmm_k1z_zmmm512, ZMM30, 64),
    ] {
        for zero in [false, true] {
            for mask in [0, 1, 2, 5, 0x80, u64::MAX] {
                let mut i = Instruction::with2(code, reg, reg).unwrap();
                i.set_op_mask(K1);
                i.set_zeroing_masking(zero);
                let context = instruction_context(&i, mask);
                let mut bytes = encode(&i);
                bytes.push(0xf4);
                for (tier, backend) in backends()
                    .into_iter()
                    .enumerate()
                    .filter(|(tier, _)| *tier == selected_tier)
                {
                    let vm = vm(&bytes, backend);
                    let mut cpu = vm.new_vcpu();
                    cpu.set_reg(Reg::Rip, CODE);
                    let original = 0x88776655_44332211_ffeeddcc_bbaa9988u128;
                    for r in 0..32 {
                        cpu.set_xmm(r, original);
                        cpu.set_ymm_hi(r, original);
                        cpu.set_zmm_hi(r, 0, original);
                        cpu.set_zmm_hi(r, 1, original);
                    }
                    cpu.set_kmask(1, mask);
                    let exit = cpu.run(&vm, None);
                    check_hlt(exit, &context);
                    for r in 0..32 {
                        let actual = [
                            cpu.xmm(r),
                            cpu.ymm_hi(r),
                            cpu.zmm_hi(r, 0),
                            cpu.zmm_hi(r, 1),
                        ];
                        for (chunk, actual) in actual.into_iter().enumerate() {
                            let mut expected = original;
                            if r == 30 {
                                expected = 0;
                                for qword in 0..2 {
                                    let lane = chunk * 2 + qword;
                                    if lane < width / 8 && (!zero || mask & (1 << lane) != 0) {
                                        expected |=
                                            original & (u128::from(u64::MAX) << (qword * 64));
                                    }
                                }
                            }
                            check_eq!(
                                actual,
                                expected,
                                "zmm{r} chunk={chunk}; tier={tier}; {context}"
                            );
                        }
                    }
                    check_eq!(cpu.kmask(1), mask, "{context}");
                    if tier == 2 {
                        assert!(
                            vm.cache.regions() > 0,
                            "regions engine=0 expected=>0; {context}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn evex_double_move_memory_transfers_whole_qwords() {
    use iced_x86::{
        Code::*,
        MemoryOperand,
        Register::{K1, RAX, XMM30},
    };
    // SDM Vol 2B §4.3, MOVUPD: an active mask bit transfers eight bytes.
    // Vol 2A §2.8.4 (Type E4.nb): inactive memory elements suppress faults.
    for store in [false, true] {
        let memory = MemoryOperand::with_base(RAX);
        let mut i = if store {
            Instruction::with2(EVEX_Vmovupd_xmmm128_k1z_xmm, memory, XMM30).unwrap()
        } else {
            Instruction::with2(EVEX_Vmovupd_xmm_k1z_xmmm128, XMM30, memory).unwrap()
        };
        i.set_op_mask(K1);
        for mask in [1, 2] {
            let context = instruction_context(&i, mask);
            let mut bytes = encode(&i);
            bytes.push(0xf4);
            for backend in backends() {
                let vm = vm(&bytes, backend);
                let value = 0x11223344_55667788u64;
                vm.write_bytes(END - 8, &value.to_le_bytes()).unwrap();
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, CODE);
                cpu.set_reg(Reg::Rax, END - 8);
                cpu.set_xmm(30, u128::MAX);
                cpu.set_ymm_hi(30, u128::MAX);
                cpu.set_zmm_hi(30, 0, u128::MAX);
                cpu.set_zmm_hi(30, 1, u128::MAX);
                cpu.set_kmask(1, mask);
                let exit = cpu.run(&vm, None);
                let fault = mask == 2;
                if fault {
                    assert!(
                        matches!(exit, Exit::UnmappedMemory { addr: END, .. }),
                        "exit engine={exit:?} expected=fault@{END:#x}; {context}"
                    );
                } else {
                    check_hlt(exit, &context);
                }
                let expected = if store || fault {
                    u128::MAX
                } else {
                    (u128::MAX << 64) | u128::from(value)
                };
                check_eq!(cpu.xmm(30), expected, "{context}");
                let upper = if store || fault { u128::MAX } else { 0 };
                check_eq!(cpu.ymm_hi(30), upper, "{context}");
                check_eq!(cpu.zmm_hi(30, 0), upper, "{context}");
                check_eq!(cpu.zmm_hi(30, 1), upper, "{context}");
                let mut actual = [0; 8];
                vm.read_bytes(END - 8, &mut actual).unwrap();
                let expected = if store && !fault { u64::MAX } else { value };
                check_eq!(
                    u64::from_le_bytes(actual),
                    expected,
                    "mem@{:#x}; {context}",
                    END - 8
                );
            }
        }
    }
}

#[test]
fn evex_vpermilps_handwritten_mask_merges_and_zeroes() {
    for zero in [false, true] {
        for backend in backends() {
            // VPERMILPS xmm1{k1}{z?}, xmm2, 0x1b: reverse the four dwords.
            let bytes = [
                0x62,
                0xf3,
                0x7d,
                if zero { 0x89 } else { 9 },
                4,
                0xca,
                0x1b,
                0xf4,
            ];
            let decoded = Decoder::new(64, &bytes, DecoderOptions::NONE).decode();
            let context = instruction_context(&decoded, 5);
            check_eq!(decoded.op_mask(), iced_x86::Register::K1, "{context}");
            check_eq!(decoded.zeroing_masking(), zero, "{context}");
            let vm = vm(&bytes, backend);
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_xmm(1, 0xdddddddd_cccccccc_bbbbbbbb_aaaaaaaa);
            cpu.set_xmm(2, 0x00000004_00000003_00000002_00000001);
            cpu.set_kmask(1, 5);
            check_hlt(cpu.run(&vm, None), &context);
            check_eq!(
                cpu.xmm(1),
                if zero {
                    0x00000000_00000002_00000000_00000004
                } else {
                    0xdddddddd_00000002_bbbbbbbb_00000004
                },
                "{context}"
            );
        }
    }
}

#[test]
fn evex_fma_family_broadcast_masks_and_accumulators() {
    // SDM Vol 2C §5.2, VFMADD/VFMSUB/VFNMADD/VFNMSUB/VFMADDSUB/VFMSUBADD:
    // 132 = dst*mem+src, 213 = src*dst+mem, 231 = src*mem+dst;
    // the alternating families subtract on even/odd lanes respectively.
    for (opcode, neg_prod, neg_add, alternating) in [
        (0x98, false, false, 0),
        (0x9a, false, true, 0),
        (0x9c, true, false, 0),
        (0x9e, true, true, 0),
        (0x96, false, false, 1),
        (0x97, false, false, 2),
    ] {
        for order in 0..3 {
            for double in [false, true] {
                let elem = if double { 8 } else { 4 };
                for (vl, width) in [(0, 16), (0x20, 32), (0x40, 64)] {
                    let count = width / elem;
                    for mask in [
                        None,
                        Some(0),
                        Some(0xa5a5),
                        Some(1 << count),
                        Some(u64::MAX),
                    ] {
                        for zero in [false, true] {
                            if zero && mask.is_none() {
                                continue;
                            }
                            // Exercise op0 == op1 as well as distinct sources.
                            let alias = zero;
                            let p1 =
                                if alias { 0x7d } else { 0x75 } | if double { 0x80 } else { 0 };
                            let bytes = [
                                0x62,
                                0xf2,
                                p1,
                                vl | 0x18 | u8::from(mask.is_some()) | if zero { 0x80 } else { 0 },
                                opcode + order * 0x10,
                                0x03,
                                0xf4,
                            ];
                            let i = Decoder::new(64, &bytes, DecoderOptions::NONE).decode();
                            let context = instruction_context(&i, mask.unwrap_or(0));
                            check_eq!(i.is_broadcast(), true, "bytes={bytes:02x?}; {context}");
                            check_eq!(i.op0_register().size(), width, "{context}");
                            check_eq!(i.memory_size().size(), elem, "{context}");
                            let pack = |base: f64| -> [u128; 4] {
                                let mut chunks = [0; 4];
                                for lane in 0..64 / elem {
                                    let value = base + (lane % 2) as f64;
                                    let bits = if double {
                                        value.to_bits()
                                    } else {
                                        u64::from((value as f32).to_bits())
                                    };
                                    chunks[lane * elem / 16] |=
                                        u128::from(bits) << ((lane * elem % 16) * 8);
                                }
                                chunks
                            };
                            let mut expected = [0u128; 4];
                            for lane in 0..count {
                                let mut dst = 2.0 + (lane % 2) as f64;
                                for _ in 0..3 {
                                    if mask.is_none_or(|k| k & (1 << lane) != 0) {
                                        let src = if alias { dst } else { 5.0 + (lane % 2) as f64 };
                                        let (x, y, z) = match order {
                                            0 => (dst, 4.0, src),
                                            1 => (src, dst, 4.0),
                                            _ => (src, 4.0, dst),
                                        };
                                        let sub = if alternating == 0 {
                                            neg_add
                                        } else {
                                            lane % 2 == usize::from(alternating == 2)
                                        };
                                        dst = (if neg_prod { -x * y } else { x * y })
                                            + if sub { -z } else { z };
                                    } else if zero {
                                        dst = 0.0;
                                    }
                                }
                                let bits = if double {
                                    dst.to_bits()
                                } else {
                                    u64::from((dst as f32).to_bits())
                                };
                                expected[lane * elem / 16] |=
                                    u128::from(bits) << ((lane * elem % 16) * 8);
                            }
                            for (tier, backend) in backends().into_iter().enumerate() {
                                let vm = vm(&bytes, backend);
                                let scalar = if double {
                                    4f64.to_bits()
                                } else {
                                    u64::from(4f32.to_bits())
                                };
                                vm.write_bytes(END - elem as u64, &scalar.to_le_bytes()[..elem])
                                    .unwrap();
                                let mut cpu = vm.new_vcpu();
                                cpu.set_reg(Reg::Rip, CODE);
                                cpu.set_reg(Reg::Rbx, END - elem as u64);
                                for r in 0..32 {
                                    let chunks = match r {
                                        0 => pack(2.0),
                                        1 => pack(5.0),
                                        _ => [u128::MAX; 4],
                                    };
                                    cpu.set_xmm(r, chunks[0]);
                                    cpu.set_ymm_hi(r, chunks[1]);
                                    cpu.set_zmm_hi(r, 0, chunks[2]);
                                    cpu.set_zmm_hi(r, 1, chunks[3]);
                                }
                                cpu.set_kmask(0, 0);
                                cpu.set_kmask(1, mask.unwrap_or(0));
                                check_hlt(
                                    cpu.run(&vm, None),
                                    &instruction_context(&i, mask.unwrap_or(0)),
                                );
                                if tier == 2 {
                                    assert!(
                                        vm.cache.regions() > 0,
                                        "regions expected=>0 engine=0; {context}"
                                    );
                                }
                                for r in 0..32 {
                                    let actual = [
                                        cpu.xmm(r),
                                        cpu.ymm_hi(r),
                                        cpu.zmm_hi(r, 0),
                                        cpu.zmm_hi(r, 1),
                                    ];
                                    let wanted = match r {
                                        0 => expected,
                                        1 => pack(5.0),
                                        _ => [u128::MAX; 4],
                                    };
                                    check_vector(
                                        actual,
                                        wanted,
                                        r,
                                        &instruction_context(&i, mask.unwrap_or(0)),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn evex_fma_memory_faults_preserve_all_vector_inputs() {
    // SDM Vol 2A §2.8.2 (Type E2): inactive elements suppress memory faults.
    // Vol 2C §5.2, VFMADD231PS: broadcast reads one dword if any lane is active.
    for broadcast in [false, true] {
        for zero in [false, true] {
            for tail in [0, 2, 4] {
                for mask in [0, 1, 2, 5, 1 << 16] {
                    for (tier, backend) in backends().into_iter().enumerate() {
                        let bytes = [
                            0x62,
                            0xf2,
                            0x75,
                            0x49 | if broadcast { 0x10 } else { 0 } | if zero { 0x80 } else { 0 },
                            0xb8,
                            0x03,
                            0xf4,
                        ];
                        let context = format!(
                            "Vfmadd231ps K1={mask:#x} z={zero} bcst={broadcast} tail={tail}"
                        );
                        let vm = vm(&bytes, backend);
                        vm.write_bytes(END - 4, &4f32.to_le_bytes()).unwrap();
                        let mut cpu = vm.new_vcpu();
                        cpu.set_reg(Reg::Rip, CODE);
                        cpu.set_reg(Reg::Rbx, END - tail);
                        let original = 0x40000000_40000000_40000000_40000000u128;
                        for r in 0..32 {
                            cpu.set_xmm(r, original);
                            cpu.set_ymm_hi(r, original);
                            cpu.set_zmm_hi(r, 0, original);
                            cpu.set_zmm_hi(r, 1, original);
                        }
                        cpu.set_kmask(1, mask);
                        let exit = cpu.run(&vm, None);
                        let active = mask & 0xffff;
                        let faulting = if broadcast {
                            if tail < 4 {
                                active
                            } else {
                                0
                            }
                        } else if tail < 4 {
                            active
                        } else {
                            active & !1
                        };
                        let fault = faulting != 0;
                        if fault {
                            let address = (END - tail
                                + if broadcast {
                                    0
                                } else {
                                    u64::from(faulting.trailing_zeros()) * 4
                                })
                            .max(END);
                            assert!(
                                matches!(exit, Exit::UnmappedMemory { addr, .. } if addr == address),
                                "exit engine={exit:?}; {context}"
                            );
                            check_eq!(cpu.reg(Reg::Rip), CODE + 7, "{context}");
                        } else {
                            check_hlt(exit, &context);
                        }
                        if tier == 2 {
                            assert!(
                                vm.cache.regions() > 0,
                                "regions expected=>0 engine=0; {context}"
                            );
                        }
                        for r in 0..32 {
                            let actual = [
                                cpu.xmm(r),
                                cpu.ymm_hi(r),
                                cpu.zmm_hi(r, 0),
                                cpu.zmm_hi(r, 1),
                            ];
                            if fault || r != 0 {
                                check_vector(actual, [original; 4], r, &format!("Vfmadd231ps K1={mask:#x} z={zero} bcst={broadcast} tail={tail}"));
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn evex_broadcast_reads_only_the_last_mapped_dword() {
    // SDM Vol 2B §4.3, PACKSSDW: a dword broadcast feeds word-sized results.
    // Bit 7 alone must still load the scalar, although there are only four source dwords.
    for backend in backends() {
        let mut i = Instruction::default();
        i.set_code(iced_x86::Code::EVEX_Vpackssdw_xmm_k1z_xmm_xmmm128b32);
        i.set_op0_register(iced_x86::Register::XMM1);
        i.set_op1_register(iced_x86::Register::XMM2);
        i.set_op2_kind(iced_x86::OpKind::Memory);
        i.set_memory_base(iced_x86::Register::RAX);
        i.set_op_mask(iced_x86::Register::K1);
        i.set_is_broadcast(true);
        let mut bytes = encode(&i);
        bytes.push(0xf4);
        let context = instruction_context(&i, 0x80);
        let vm = vm(&bytes, backend);
        vm.write_bytes(END - 4, &0x1234u32.to_le_bytes()).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rax, END - 4);
        cpu.set_xmm(1, u128::MAX);
        cpu.set_kmask(1, 0x80);
        check_hlt(cpu.run(&vm, None), &context);
        check_eq!(
            cpu.xmm(1),
            0x1234ffff_ffffffff_ffffffff_ffffffff,
            "{context}"
        );
    }
    for (p1, elem, value, expected) in [
        (
            0x6c,
            4,
            u64::from(1f32.to_bits()),
            0x3f800000_3f800000_3f800000_3f800000u128,
        ),
        (0xed, 8, 1f64.to_bits(), 0x3ff0000000000000_3ff0000000000000),
    ] {
        for backend in backends() {
            // VADDPS/VADDPD xmm1, xmm2, scalar bcst [rax].
            let bytes = [0x62, 0xf1, p1, 0x18, 0x58, 8, 0xf4];
            let decoded = Decoder::new(64, &bytes, DecoderOptions::NONE).decode();
            let context = instruction_context(&decoded, 0);
            check_eq!(decoded.is_broadcast(), true, "{context}");
            check_eq!(decoded.memory_size().size(), elem, "{context}");
            let vm = vm(&bytes, backend);
            vm.write_bytes(END - elem as u64, &value.to_le_bytes()[..elem])
                .unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.set_reg(Reg::Rip, CODE);
            cpu.set_reg(Reg::Rax, END - elem as u64);
            check_hlt(cpu.run(&vm, None), &context);
            check_eq!(cpu.xmm(1), expected, "{context}");
        }
    }
    // SDM Vol 1 §15.6.1.1: encoded k0 means all elements, independent of k0's value.
    let blend = x86jit_tests::compat::evex_modifier_forms()
        .into_iter()
        .find(|i| {
            i.mnemonic() == iced_x86::Mnemonic::Vpblendmd
                && i.op0_register().size() == 16
                && i.is_broadcast()
                && !i.has_op_mask()
        })
        .unwrap();
    let mut bytes = encode(&blend);
    bytes.push(0xf4);
    for backend in backends() {
        let context = instruction_context(&blend, 0);
        let vm = vm(&bytes, backend);
        vm.write_bytes(END - 4, &1u32.to_le_bytes()).unwrap();
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rax, END - 4);
        cpu.set_kmask(0, 0);
        check_hlt(cpu.run(&vm, None), &context);
        check_eq!(
            cpu.xmm(0),
            0x00000001_00000001_00000001_00000001,
            "{context}"
        );
    }
}

#[test]
fn evex_masked_memory_suppresses_inactive_elements_and_preserves_fault_state() {
    compare_and_blend_memory_faults();
    // SDM Vol 1 §15.6.1, Vol 2A §2.8.2: masked-off VADDPS elements
    // suppress memory faults; a fault in an active element commits no destination.
    for broadcast in [false, true] {
        for zero in [false, true] {
            for mask in [0, 1, 2, 5, 0xffff_ffff_ffff_0000] {
                for backend in backends() {
                    let p2 = 9 | if broadcast { 0x10 } else { 0 } | if zero { 0x80 } else { 0 };
                    let bytes = [0x62, 0xf1, 0x6c, p2, 0x58, 8, 0xf4];
                    let i = Decoder::new(64, &bytes, DecoderOptions::NONE).decode();
                    let context = instruction_context(&i, mask);
                    let vm = vm(&bytes, backend);
                    vm.write_bytes(END - 4, &1f32.to_le_bytes()).unwrap();
                    let mut cpu = vm.new_vcpu();
                    cpu.set_reg(Reg::Rip, CODE);
                    cpu.set_reg(Reg::Rax, END - 4);
                    for r in 0..32 {
                        cpu.set_xmm(r, u128::MAX);
                        cpu.set_ymm_hi(r, u128::MAX);
                        cpu.set_zmm_hi(r, 0, u128::MAX);
                        cpu.set_zmm_hi(r, 1, u128::MAX);
                    }
                    cpu.set_xmm(2, 0);
                    cpu.set_kmask(1, mask);
                    let exit = cpu.run(&vm, None);
                    let fault = !broadcast && mask & 0xe != 0;
                    if fault {
                        let address = END - 4 + u64::from((mask & 0xe).trailing_zeros()) * 4;
                        assert!(
                            matches!(exit, Exit::UnmappedMemory { addr, .. } if addr == address),
                            "exit engine={exit:?}; {context}"
                        );
                        check_eq!(cpu.reg(Reg::Rip), CODE + 7, "{context}");
                    } else {
                        check_hlt(exit, &context);
                        let mut expected = if zero { 0 } else { u128::MAX };
                        for n in 0..4 {
                            if mask & (1 << n) != 0 {
                                expected = (expected & !(0xffff_ffff << (32 * n)))
                                    | (0x3f80_0000 << (32 * n));
                            }
                        }
                        check_eq!(cpu.xmm(1), expected, "{context}");
                        check_eq!(cpu.ymm_hi(1), 0, "{context}");
                        check_eq!(cpu.zmm_hi(1, 0), 0, "{context}");
                        check_eq!(cpu.zmm_hi(1, 1), 0, "{context}");
                    }
                    for r in 0..32 {
                        if r == 1 && !fault {
                            continue;
                        }
                        check_eq!(
                            cpu.xmm(r),
                            if r == 2 { 0 } else { u128::MAX },
                            "register {r}; {context}"
                        );
                        check_eq!(cpu.ymm_hi(r), u128::MAX, "register {r}; {context}");
                        check_eq!(cpu.zmm_hi(r, 0), u128::MAX, "register {r}; {context}");
                        check_eq!(cpu.zmm_hi(r, 1), u128::MAX, "register {r}; {context}");
                    }
                }
            }
        }
    }
}

fn compare_and_blend_memory_faults() {
    // SDM Vol 2A §2.8.4 (Type E4), Vol 2C §5.2, VPCMPD/VPTESTMD/VPBLENDMD.
    for mut i in x86jit_tests::compat::evex_modifier_forms()
        .into_iter()
        .filter(|i| {
            matches!(
                i.mnemonic(),
                iced_x86::Mnemonic::Vpcmpd
                    | iced_x86::Mnemonic::Vptestmd
                    | iced_x86::Mnemonic::Vpblendmd
            ) && i.op1_register().size() == 16
                && i.op2_kind() == iced_x86::OpKind::Memory
                && i.has_op_mask()
                && !i.zeroing_masking()
                && !i.is_broadcast()
        })
    {
        for broadcast in [false, true] {
            i.set_is_broadcast(broadcast);
            let mut bytes = encode(&i);
            bytes.push(0xf4);
            for mask in [0u64, 1, 3] {
                let context = instruction_context(&i, mask);
                for backend in backends() {
                    let vm = vm(&bytes, backend);
                    vm.write_bytes(END - 4, &1u32.to_le_bytes()).unwrap();
                    let mut cpu = vm.new_vcpu();
                    cpu.set_reg(Reg::Rip, CODE);
                    cpu.set_reg(Reg::Rax, END - 4);
                    cpu.set_xmm(0, u128::MAX);
                    cpu.set_kmask(1, mask);
                    cpu.set_kmask(2, u64::MAX);
                    let exit = cpu.run(&vm, None);
                    let vector_dst = i.op0_register().is_vector_register();
                    if mask == 3 && !broadcast {
                        assert!(
                            matches!(exit, Exit::UnmappedMemory { addr: END, .. }),
                            "exit engine={exit:?}; {}",
                            instruction_context(&i, mask)
                        );
                        check_eq!(cpu.reg(Reg::Rip), CODE + 7, "{context}");
                        check_eq!(cpu.xmm(0), u128::MAX, "{context}");
                        check_eq!(cpu.kmask(2), u64::MAX, "{context}");
                    } else {
                        check_hlt(exit, &instruction_context(&i, mask));
                        if vector_dst {
                            let value = (0..4)
                                .fold(0u128, |v, n| v | (u128::from((mask >> n) & 1) << (n * 32)));
                            check_eq!(cpu.xmm(0), value, "{context}");
                        } else {
                            check_eq!(
                                cpu.kmask(2),
                                if i.mnemonic() == iced_x86::Mnemonic::Vpcmpd {
                                    mask
                                } else {
                                    0
                                },
                                "{context}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn evex_broadcast_fault_names_the_first_missing_byte() {
    let context = "Vaddps XMM1,XMM2 unmasked bcst=true";
    for backend in backends() {
        let vm = vm(&[0x62, 0xf1, 0x6c, 0x18, 0x58, 8, 0xf4], backend);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, CODE);
        cpu.set_reg(Reg::Rax, END - 2);
        cpu.set_xmm(1, u128::MAX);
        let exit = cpu.run(&vm, None);
        assert!(
            matches!(exit, Exit::UnmappedMemory { addr: END, .. }),
            "exit expected=fault@{END:#x} engine={exit:?}; {context}"
        );
        check_eq!(cpu.xmm(1), u128::MAX, "{context}");
    }
}

#[test]
fn evex_e4nf_broadcast_reads_with_an_empty_destination_mask() {
    use iced_x86::Mnemonic::*;
    // SDM Vol 2A §2.8.4, Tables 2-45/2-52: E4NF has no fault suppression.
    // VPERMD ymm0{k1}, ymm1, [rax]{1to8}.
    let reproduction = Decoder::new(
        64,
        &[0x62, 0xf2, 0x75, 0x39, 0x36, 0x00],
        DecoderOptions::NONE,
    )
    .decode();
    let context = instruction_context(&reproduction, 0);
    check_eq!(reproduction.mnemonic(), Vpermd, "{context}");
    check_eq!(reproduction.is_broadcast(), true, "{context}");
    check_eq!(reproduction.op_mask(), iced_x86::Register::K1, "{context}");
    let mut checked = std::collections::BTreeSet::new();
    let mut forms = 0;
    for i in std::iter::once(reproduction)
        .chain(x86jit_tests::compat::evex_modifier_forms())
        .filter(|i| {
            i.is_broadcast()
                && i.has_op_mask()
                && matches!(
                    i.mnemonic(),
                    Vpackssdw
                        | Vpackusdw
                        | Vpermd
                        | Vpermpd
                        | Vpermps
                        | Vpermq
                        | Vpermilpd
                        | Vpermilps
                        | Vpshufd
                        | Vpunpckhdq
                        | Vpunpckhqdq
                        | Vpunpckldq
                        | Vpunpcklqdq
                        | Vshufpd
                        | Vshufps
                        | Vunpckhpd
                        | Vunpckhps
                        | Vunpcklpd
                        | Vunpcklps
                )
                && lifted(i).is_some()
        })
    {
        forms += 1;
        checked.insert(i.mnemonic());
        let mut bytes = encode(&i);
        bytes.push(0xf4);
        let elem = i.memory_size().size() as u64;
        let dst = i.op0_register().number();
        for (tier, backend) in backends().into_iter().enumerate() {
            let vm = vm(&bytes, backend);
            for mask in [0, 1u64 << 63] {
                let context = instruction_context(&i, mask);
                // Warm the loop with a mapped element so the region tier also
                // executes the faulting instruction from its compiled region.
                let mut cpu = vm.new_vcpu();
                cpu.set_reg(Reg::Rip, CODE);
                cpu.set_reg(Reg::Rax, END - elem);
                cpu.set_kmask(1, mask);
                cpu.set_xmm(dst, u128::MAX);
                cpu.set_ymm_hi(dst, u128::MAX);
                cpu.set_zmm_hi(dst, 0, u128::MAX);
                cpu.set_zmm_hi(dst, 1, u128::MAX);
                check_hlt(cpu.run(&vm, None), &instruction_context(&i, mask));
                if tier == 2 {
                    assert!(
                        vm.cache.regions() > 0,
                        "regions engine=0 expected=>0; {}",
                        instruction_context(&i, mask)
                    );
                }
                let width = i.op0_register().size();
                check_eq!(
                    cpu.xmm(dst),
                    if i.zeroing_masking() { 0 } else { u128::MAX },
                    "{context}"
                );
                check_eq!(
                    cpu.ymm_hi(dst),
                    if width < 32 || i.zeroing_masking() {
                        0
                    } else {
                        u128::MAX
                    },
                    "{context}"
                );
                for half in 0..2 {
                    check_eq!(
                        cpu.zmm_hi(dst, half),
                        if width < 64 || i.zeroing_masking() {
                            0
                        } else {
                            u128::MAX
                        },
                        "{context}"
                    );
                }
                for addr in [END, END - elem + 1] {
                    cpu.set_reg(Reg::Rip, CODE);
                    cpu.set_reg(Reg::Rax, addr);
                    for r in 0..32 {
                        cpu.set_xmm(r, u128::MAX);
                        cpu.set_ymm_hi(r, u128::MAX);
                        cpu.set_zmm_hi(r, 0, u128::MAX);
                        cpu.set_zmm_hi(r, 1, u128::MAX);
                    }
                    let exit = cpu.run(&vm, None);
                    assert!(
                        matches!(
                            exit,
                            Exit::UnmappedMemory {
                                addr: END,
                                access: x86jit_core::AccessKind::Read,
                                ..
                            }
                        ),
                        "exit engine={exit:?}; {}",
                        instruction_context(&i, mask)
                    );
                    check_eq!(cpu.reg(Reg::Rip), CODE + 7, "{context}");
                    for r in 0..32 {
                        check_eq!(
                            cpu.xmm(r),
                            u128::MAX,
                            "register {r}; {}",
                            instruction_context(&i, mask)
                        );
                        check_eq!(
                            cpu.ymm_hi(r),
                            u128::MAX,
                            "register {r}; {}",
                            instruction_context(&i, mask)
                        );
                        check_eq!(
                            cpu.zmm_hi(r, 0),
                            u128::MAX,
                            "register {r}; {}",
                            instruction_context(&i, mask)
                        );
                        check_eq!(
                            cpu.zmm_hi(r, 1),
                            u128::MAX,
                            "register {r}; {}",
                            instruction_context(&i, mask)
                        );
                    }
                    check_eq!(cpu.kmask(1), mask, "{context}");
                }
            }
        }
    }
    check_eq!(
        checked.len(),
        19,
        "missing E4NF families: {checked:?}; {context}"
    );
    eprintln!(
        "E4NF broadcast witnesses: {forms} forms, {} mnemonics, all three tiers",
        checked.len()
    );
}

#[test]
fn evex_modifiers_execute_in_a_compiled_region() {
    let context = "Vaddps XMM1,XMM2 K1=0x5 z=true bcst=true";
    let backend = JitBackend::with_superblocks(x86jit_core::RegionCaps {
        max_blocks: 16,
        max_icount: 256,
    });
    let vm = vm(&[0x62, 0xf1, 0x6c, 0x99, 0x58, 8, 0xf4], Box::new(backend));
    vm.write_bytes(END - 4, &1f32.to_le_bytes()).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, CODE);
    cpu.set_reg(Reg::Rax, END - 4);
    cpu.set_kmask(1, 5);
    check_hlt(cpu.run(&vm, None), context);
    check_eq!(
        cpu.xmm(1),
        0x00000000_3f800000_00000000_3f800000,
        "{context}"
    );
    assert!(
        vm.cache.regions() > 0,
        "regions expected=>0 engine=0; {context}"
    );
}
