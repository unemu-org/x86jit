//! The relift diagnostics (`Vm::relift_diagnostics`, xj-to9.53) name why a
//! Flat32 proof relifted: a patched live-capable immediate relifts until it
//! is promoted and then is absorbed; a patched shift count, which no live
//! read covers, is refused as an ineligible form and relifts on every patch
//! with its instruction in the ring; an opcode store is an opcode store; a
//! loop of more blocks than the unpaged proof map retains clears at its
//! bound.
use x86jit_core::{
    CpuModel, Exit, LiveRefusal, ProofClass, Prot, RegionKind, ReliftCause, StoreClass, Vcpu, Vm,
    VmConfig,
};

const SIZE: u64 = 0x20000;
const CODE: u64 = 0x10000;
const N: u64 = 40;

fn setup(code: &[u8]) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(SIZE));
    vm.set_cpu_model(CpuModel::I80486Dx);
    vm.map(0, SIZE as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.write_bytes(CODE, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.legacy.cr0 = 1;
    for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
        cache.base = 0;
        cache.limit = u32::MAX;
        cache.attributes = (if i == 0 { 0x9b } else { 0x93 }) | 0x4000;
        cache.unusable = 0;
    }
    cpu.set_a20_enabled(true);
    cpu.cpu.rip = CODE;
    cpu.cpu.gpr[4] = 0x8000;
    cpu.cpu.gpr[7] = N;
    (vm, cpu)
}

fn run(vm: &Vm, cpu: &mut Vcpu) {
    assert!(matches!(cpu.run(vm, None), Exit::Hlt));
}

/// A: MOV EAX, imm32; ADD EBX, EAX; JMP $+2. B: MOV EDX, EBX; SHR EDX, 3;
/// JMP $+2. C: ADD [A+1], 1; XOR [B+4], 1; DEC EDI; JNZ A; HLT. Returns
/// the code and the bus addresses of the immediate and the count.
fn patching_loop() -> (Vec<u8>, u64, u64) {
    let imm = CODE + 1;
    let mut c = vec![0xb8, 0, 0, 0, 0, 0x01, 0xc3, 0xeb, 0x00];
    // B at A+9: MOV EDX, EBX (89 DA); SHR EDX, 3 (C1 EA 03); JMP $+2.
    c.extend([0x89, 0xda, 0xc1, 0xea, 0x03, 0xeb, 0x00]);
    let shift = CODE + 9 + 4;
    assert_eq!(c[(shift - CODE) as usize], 0x03);
    // C: ADD DWORD [imm], 1 (83 05 disp32 01); XOR BYTE [shift], 1 (80 35 disp32 01).
    c.extend([0x83, 0x05]);
    c.extend((imm as u32).to_le_bytes());
    c.push(1);
    c.extend([0x80, 0x35]);
    c.extend((shift as u32).to_le_bytes());
    c.push(1);
    // DEC EDI; JNZ A (rel32); HLT.
    c.push(0x4f);
    let rel = CODE as i64 - (CODE as i64 + c.len() as i64 + 6);
    c.extend([0x0f, 0x85]);
    c.extend((rel as i32).to_le_bytes());
    c.push(0xf4);
    (c, imm, shift)
}

#[test]
fn a_patched_immediate_is_absorbed_and_a_patched_shift_count_is_named() {
    let (code, imm, shift) = patching_loop();
    let (vm, mut cpu) = setup(&code);
    vm.relift_diagnostics(true);
    run(&vm, &mut cpu);
    let d = vm.relift_diagnostics(true);
    let text = d.to_string();
    let flat = Some(ProofClass::Flat32);
    // Two patches promote the immediate; later ones are absorbed.
    let live = d.lifts(flat, ReliftCause::Store(StoreClass::LiveField));
    assert!((1..=3).contains(&live), "{text}");
    assert_eq!(d.live.promotions, 1, "{text}");
    assert!(d.live.absorbed >= N - 4, "{text}");
    // Every patch of the count relifts B.
    let other = d.lifts(flat, ReliftCause::Store(StoreClass::OtherField));
    assert!(other >= N - 2, "{text}");
    assert!(
        d.live
            .refusals
            .iter()
            .any(|&(r, n)| r == LiveRefusal::IneligibleForm && n >= 1),
        "{text}"
    );
    let refusal = d
        .refused
        .iter()
        .find(|r| r.patched == (shift, 1))
        .unwrap_or_else(|| panic!("{text}"));
    assert_eq!(refusal.instruction.0, shift - 2, "{text}");
    assert_eq!(refusal.instruction.1[..2], [0xc1, 0xea], "{text}");
    assert_eq!(refusal.instruction.1.len(), 3, "{text}");
    assert!(refusal.instruction.2.starts_with("shr"), "{text}");
    let record = d
        .recent
        .iter()
        .rev()
        .find(|r| r.write == Some((shift, 1)))
        .unwrap_or_else(|| panic!("{text}"));
    assert_eq!(
        record.cause,
        ReliftCause::Store(StoreClass::OtherField),
        "{text}"
    );
    let (at, bytes, insn) = record.instruction.as_ref().unwrap();
    assert_eq!((*at, &bytes[..2]), (shift - 2, &[0xc1, 0xea][..]), "{text}");
    assert!(insn.starts_with("shr edx"), "{text}");
    assert!(d.recent.iter().all(|r| r.write != Some((imm, 4))
        || r.cause == ReliftCause::Store(StoreClass::LiveField)
        || r.cause == ReliftCause::Retry(x86jit_core::RetryKind::Promotion)
        || r.cause == ReliftCause::Promotion));
    // The counts are per cause and add up to the lifts made.
    assert_eq!(d.total(None), d.total(flat), "{text}");
    assert!(vm.relift_diagnostics(false).lifts.is_empty());
}

#[test]
fn an_opcode_store_and_the_proof_bound_are_named() {
    // A: INC EBP (45); JMP $+2. B: XOR BYTE [A], 8; DEC EDI; JNZ A; HLT.
    let mut c = vec![0x45, 0xeb, 0x00, 0x80, 0x35];
    c.extend((CODE as u32).to_le_bytes());
    c.push(8);
    c.push(0x4f);
    let rel = CODE as i64 - (CODE as i64 + c.len() as i64 + 6);
    c.extend([0x0f, 0x85]);
    c.extend((rel as i32).to_le_bytes());
    c.push(0xf4);
    let (vm, mut cpu) = setup(&c);
    vm.relift_diagnostics(true);
    run(&vm, &mut cpu);
    let d = vm.relift_diagnostics(true);
    let opcode = d.lifts(None, ReliftCause::Store(StoreClass::Opcode));
    assert!(opcode >= N - 2, "{d}");
    assert_eq!(cpu.cpu.gpr[5], 0, "INC and DEC alternate");

    // 1100 blocks of ADD EBX, 1; JMP $+2, more than the 1024 proofs kept,
    // then DEC EDI; JNZ top; HLT.
    let mut c = Vec::new();
    for _ in 0..1100 {
        c.extend([0x83, 0xc3, 0x01, 0xeb, 0x00]);
    }
    c.push(0x4f);
    let rel = CODE as i64 - (CODE as i64 + c.len() as i64 + 6);
    c.extend([0x0f, 0x85]);
    c.extend((rel as i32).to_le_bytes());
    c.push(0xf4);
    let (vm, mut cpu) = setup(&c);
    run(&vm, &mut cpu);
    let d = vm.relift_diagnostics(true);
    assert_eq!(cpu.cpu.gpr[3], 1100 * N);
    assert!(d.lifts(None, ReliftCause::First) >= 1101, "{d}");
    assert!(d.lifts(None, ReliftCause::Bound) > 0, "{d}");
}

/// A Flat32 loop that loads FS with each of 20 selectors in turn (bases
/// 10h apart) and reads FS:[0]: one context identity, so its blocks stay
/// proved, and each read sees the FS it loaded (xj-to9.53). Before, every
/// selector was a context stamp of its own: 20 stamps cycling through the
/// 16 retained identities relifted every block of every iteration.
#[test]
fn reloading_fs_keeps_flat32_proofs() {
    const GDT: u64 = 0x6000;
    const BASES: u64 = 0x8000;
    const SELECTORS: u64 = 20;
    // ADD SI, 8; CMP SI, 8 * 21; JB +4; MOV SI, 8; MOV FS, SI;
    // MOV EAX, FS:[0]; ADD EBX, EAX; JMP $+2.
    let mut c = vec![
        0x66, 0x83, 0xc6, 0x08, 0x66, 0x81, 0xfe, 0xa8, 0x00, 0x72, 0x04, 0x66, 0xbe, 0x08, 0x00,
        0x8e, 0xe6, 0x64, 0xa1, 0, 0, 0, 0, 0x01, 0xc3, 0xeb, 0x00,
    ];
    // Ten blocks of ADD ECX, 1; JMP $+2; then DEC EDI; JNZ top; HLT.
    for _ in 0..10 {
        c.extend([0x83, 0xc1, 0x01, 0xeb, 0x00]);
    }
    c.push(0x4f);
    let rel = CODE as i64 - (CODE as i64 + c.len() as i64 + 6);
    c.extend([0x0f, 0x85]);
    c.extend((rel as i32).to_le_bytes());
    c.push(0xf4);
    let (vm, mut cpu) = setup(&c);
    let value = |k: u64| (k * k + 1) as u32;
    for k in 1..=SELECTORS {
        let base = (BASES + 0x10 * k) as u32;
        let b = base.to_le_bytes();
        // Writable data, limit FFFFFh pages, G and B, DPL 0.
        vm.write_bytes(
            GDT + 8 * k,
            &[0xff, 0xff, b[0], b[1], b[2], 0x93, 0xcf, b[3]],
        )
        .unwrap();
        vm.write_bytes(u64::from(base), &value(k).to_le_bytes())
            .unwrap();
    }
    cpu.cpu.legacy.gdtr.base = GDT as u32;
    cpu.cpu.legacy.gdtr.limit = (8 * (SELECTORS + 1) - 1) as u16;
    cpu.cpu.gpr[6] = 8;
    vm.relift_diagnostics(true);
    run(&vm, &mut cpu);
    let d = vm.relift_diagnostics(true);
    let mut selector = 8;
    let mut ebx = 0u32;
    for _ in 0..N {
        selector += 8;
        if selector >= 8 * (SELECTORS + 1) {
            selector = 8;
        }
        ebx = ebx.wrapping_add(value(selector / 8));
    }
    assert_eq!(cpu.cpu.gpr[3] as u32, ebx, "{d}");
    assert_eq!(cpu.cpu.gpr[1], 10 * N, "{d}");
    // The blocks lift once, the FS load and the read in it once per
    // selector at most (a singleton Checked load is no proof).
    assert!(
        d.total(Some(ProofClass::Flat32)) <= 2 * SELECTORS + 16,
        "{d}"
    );
    assert_eq!(d.contexts.evicted, 0, "{d}");
}

/// A paged loop: a hot inner loop of 10 blocks, which links, then a sweep
/// of 300 blocks, more than the 256 paged proofs kept, each adding its own
/// immediate. The sweep fills the map, which drops a quarter of its proofs
/// (the links into them end with the map's version); the hot loop entered
/// again relinks, and every block executes its own bytes, so the sum is
/// exact, and most sweep proofs stay (xj-to9.53).
#[test]
fn a_full_paged_proof_map_drops_a_quarter_and_relinks() {
    const PD: u32 = 0x4000;
    const PT: u32 = 0x5000;
    const HOT: u32 = 10;
    const SWEEP: u32 = 300;
    const INNER: u32 = 50;
    let block = |c: &mut Vec<u8>, imm: u32| {
        // ADD EBX, imm32 (81 C3 id); JMP $+2.
        c.extend([0x81, 0xc3]);
        c.extend(imm.to_le_bytes());
        c.extend([0xeb, 0x00]);
    };
    // top: MOV ECX, INNER; hot: HOT blocks; DEC ECX; JNZ hot (rel32);
    // SWEEP blocks; DEC EDI; JNZ top (rel32); HLT. 2.5 KiB, one page.
    let mut c = vec![0xb9];
    c.extend(INNER.to_le_bytes());
    let hot = c.len() as i64;
    for i in 1..=HOT {
        block(&mut c, 1000 * i);
    }
    c.push(0x49);
    let rel = hot - (c.len() as i64 + 6);
    c.extend([0x0f, 0x85]);
    c.extend((rel as i32).to_le_bytes());
    for i in 1..=SWEEP {
        block(&mut c, i);
    }
    c.push(0x4f);
    let rel = -(c.len() as i64 + 6);
    c.extend([0x0f, 0x85]);
    c.extend((rel as i32).to_le_bytes());
    c.push(0xf4);
    assert!(c.len() < 0x1000);
    let (vm, mut cpu) = setup(&c);
    // 80386 PRM §5.2 pp. 5-7–11: identity-map the RAM, present, writable.
    vm.write_bytes(u64::from(PD), &(PT | 3).to_le_bytes())
        .unwrap();
    for page in 0..(SIZE >> 12) as u32 {
        vm.write_bytes(u64::from(PT + 4 * page), &((page << 12) | 3).to_le_bytes())
            .unwrap();
    }
    cpu.cpu.legacy.cr3 = PD;
    cpu.cpu.legacy.cr0 = 0x8000_0001;
    vm.relift_diagnostics(true);
    Vcpu::take_machine_counters_for_test();
    run(&vm, &mut cpu);
    let [_, _, lifts, _, _, links] = Vcpu::take_machine_counters_for_test();
    let d = vm.relift_diagnostics(true);
    let hot_sum = u64::from(1000 * HOT * (HOT + 1) / 2 * INNER);
    let sweep_sum = u64::from(SWEEP * (SWEEP + 1) / 2);
    assert_eq!(cpu.cpu.gpr[3], N * (hot_sum + sweep_sum), "{d}");
    assert!(
        d.lifts(Some(ProofClass::Paged), ReliftCause::Bound) > 0,
        "{d}"
    );
    // A cleared map would relift about every sweep block of every pass.
    let blocks = u64::from(HOT + SWEEP + 3);
    assert!(lifts <= blocks + (N - 1) * 200, "{lifts} lifts\n{d}");
    assert!(links > N, "{links} links\n{d}");
}

/// A hot Flat32 loop whose every iteration walks through 20 data selectors
/// with distinct bases (Segmented contexts) and back to a flat one, as an
/// extended-memory runtime's interrupt reflection does: 21 stamps per
/// iteration, more than the 16 identities once retained. With 64 retained
/// no identity is evicted, so every block keeps its key and its proof, and
/// each DS-relative read sees the base it loaded (xj-to9.53).
#[test]
fn a_reflection_path_through_many_contexts_keeps_identities() {
    const GDT: u64 = 0x6000;
    const SELECTORS: u32 = 20;
    let flat = 8 * (SELECTORS + 1);
    let mut c = Vec::new();
    // Hot: ten blocks of ADD EBX, imm32; JMP $+2.
    for i in 1..=10u32 {
        c.extend([0x81, 0xc3]);
        c.extend((1000 * i).to_le_bytes());
        c.extend([0xeb, 0x00]);
    }
    // For each selector: MOV BP, sel; MOV DS, BP; ADD ECX, [0]; JMP $+2.
    for k in 1..=SELECTORS {
        c.extend([0x66, 0xbd]);
        c.extend(((8 * k) as u16).to_le_bytes());
        c.extend([0x8e, 0xdd, 0x03, 0x0d, 0, 0, 0, 0, 0xeb, 0x00]);
    }
    // MOV BP, flat; MOV DS, BP; DEC EDI; JNZ top; HLT.
    c.extend([0x66, 0xbd]);
    c.extend((flat as u16).to_le_bytes());
    c.extend([0x8e, 0xdd, 0x4f]);
    let rel = CODE as i64 - (CODE as i64 + c.len() as i64 + 6);
    c.extend([0x0f, 0x85]);
    c.extend((rel as i32).to_le_bytes());
    c.push(0xf4);
    let (vm, mut cpu) = setup(&c);
    for k in 1..=SELECTORS + 1 {
        // Writable data, limit FFFFFh pages, G and B, DPL 0; based at
        // 100h * k, the last one flat.
        let base = if k == SELECTORS + 1 { 0 } else { 0x100 * k };
        let b = base.to_le_bytes();
        vm.write_bytes(
            GDT + 8 * u64::from(k),
            &[0xff, 0xff, b[0], b[1], b[2], 0x93, 0xcf, b[3]],
        )
        .unwrap();
        vm.write_bytes(u64::from(base), &k.to_le_bytes()).unwrap();
    }
    cpu.cpu.legacy.gdtr.base = GDT as u32;
    cpu.cpu.legacy.gdtr.limit = (8 * (SELECTORS + 2) - 1) as u16;
    vm.relift_diagnostics(true);
    run(&vm, &mut cpu);
    let d = vm.relift_diagnostics(true);
    assert_eq!(cpu.cpu.gpr[3], N * 1000 * 55, "{d}");
    assert_eq!(
        cpu.cpu.gpr[1],
        N * u64::from(SELECTORS * (SELECTORS + 1) / 2),
        "{d}"
    );
    assert_eq!(d.contexts.evicted, 0, "{d}");
    // Every block lifts once: no identity, so no key, is minted twice.
    assert!(d.total(None) <= 4 * u64::from(SELECTORS) + 30, "{d}");
}
