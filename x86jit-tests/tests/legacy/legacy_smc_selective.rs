//! A code write drops only the proofs whose source bytes it overlaps
//! (xj-to9.45). A loop of three blocks on one page stores into an earlier
//! block's immediate and into the next instruction of its own block; the
//! third block is never written. Every store is seen by the next fetch of the
//! bytes it changed (spec.md §10), on Flat32, Segmented and paged 32-bit code
//! and in real mode, for every budget slicing. The unwritten blocks stay
//! proved: the loop lifts a bounded number of blocks per iteration.
use x86jit_core::{CpuModel, Exit, Prot, RegionKind, StepOptions, StepStop, Vcpu, Vm, VmConfig};

const SIZE: u64 = 0x20000;
const CODE: u64 = 0x10000;
const PD: u32 = 0x4000;
const PT: u32 = 0x5000;
const N: u64 = 60;
const A0: u32 = 0x1234_5600;
const C0: u32 = 0x0abc_0000;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Path {
    Flat32,
    Seg32,
    Paged32,
    Real,
}

/// The loop at IP `org`; DS has CS's base, so its stores address IPs.
/// `bits` 16 uses the same opcodes with 16-bit immediates and addresses.
fn code(org: u64, bits: u32) -> Vec<u8> {
    let w = (bits / 8) as usize;
    let le = |v: u64| v.to_le_bytes()[..w].to_vec();
    // Block A: MOV eAX, imm; ADD eBX, eAX; JMP $+2.
    let imm_a = org + 1;
    let mut c = vec![0xb8];
    c.extend(le(u64::from(A0)));
    c.extend([0x01, 0xc3, 0xeb, 0x00]);
    // Block B: ADD [imm_a], 1; ADD [imm_c], 1; MOV eDX, imm_c; ADD eSI, eDX; JMP $+2.
    let store = |c: &mut Vec<u8>, target: u64| {
        c.extend([0x83, if bits == 32 { 0x05 } else { 0x06 }]);
        c.extend(le(target));
        c.push(1);
    };
    store(&mut c, imm_a);
    let imm_c = org + c.len() as u64 + (2 + w as u64 + 1) + 1;
    store(&mut c, imm_c);
    assert_eq!(org + c.len() as u64 + 1, imm_c);
    c.push(0xba);
    c.extend(le(u64::from(C0)));
    c.extend([0x01, 0xd6, 0xeb, 0x00]);
    // Block C, never written: ADD eBP, 3; DEC eDI; JNZ top; HLT.
    c.extend([0x83, 0xc5, 0x03, 0x4f, 0x75]);
    c.push((-(c.len() as i64 + 1)) as u8);
    c.push(0xf4);
    c
}

fn setup(path: Path, model: CpuModel) -> (Vm, Vcpu) {
    let mut vm = Vm::new(VmConfig::flat(SIZE));
    vm.set_cpu_model(model);
    vm.map(0, SIZE as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    let mut cpu = vm.new_vcpu();
    let (org, ds, bits) = match path {
        Path::Flat32 | Path::Paged32 => (CODE, 0, 32),
        Path::Seg32 => (0, CODE, 32),
        Path::Real => (0, CODE, 16),
    };
    vm.write_bytes(CODE, &code(org, bits)).unwrap();
    if path == Path::Real {
        cpu.cpu.legacy = x86jit_core::CpuState::legacy_real_entry([0x1000; 6]).legacy;
        cpu.cpu.cs = 0x1000;
        cpu.cpu.ds = 0x1000;
    } else {
        cpu.cpu.legacy.cr0 = 1;
        for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
            cache.base = ds as u32;
            // Byte-granular 64 KiB limits keep CS.D=1 code Segmented.
            cache.limit = if path == Path::Seg32 {
                0xffff
            } else {
                u32::MAX
            };
            cache.attributes = (if i == 0 { 0x9b } else { 0x93 }) | 0x4000;
            cache.unusable = 0;
        }
    }
    if path == Path::Paged32 {
        // 80386 PRM §5.2 pp. 5-7–11: identity-map the RAM, present, writable.
        vm.write_bytes(u64::from(PD), &(PT | 3).to_le_bytes())
            .unwrap();
        for page in 0..(SIZE >> 12) as u32 {
            vm.write_bytes(u64::from(PT + 4 * page), &((page << 12) | 3).to_le_bytes())
                .unwrap();
        }
        cpu.cpu.legacy.cr3 = PD;
        cpu.cpu.legacy.cr0 = 0x8000_0001;
    }
    cpu.set_a20_enabled(true);
    cpu.cpu.rip = org;
    cpu.cpu.gpr[7] = N;
    (vm, cpu)
}

fn run(vm: &Vm, cpu: &mut Vcpu, slice: u64) {
    for _ in 0..100_000 {
        let r = cpu.run_steps(vm, slice, StepOptions::default()).unwrap();
        match r.stop() {
            StepStop::Exit(Exit::Hlt) => return,
            StepStop::Exit(Exit::BudgetExhausted) => {}
            stop => panic!("{stop:?}"),
        }
    }
    panic!("the loop did not halt");
}

const CASES: [(Path, CpuModel); 7] = [
    (Path::Flat32, CpuModel::I80386),
    (Path::Flat32, CpuModel::I80486Dx),
    (Path::Seg32, CpuModel::I80386),
    (Path::Seg32, CpuModel::I80486Dx),
    (Path::Paged32, CpuModel::I80386),
    (Path::Paged32, CpuModel::I80486Dx),
    (Path::Real, CpuModel::I80386),
];

#[test]
fn stores_into_one_block_reach_the_next_fetch_of_that_block() {
    for (path, model) in CASES {
        let mask = if path == Path::Real {
            0xffff
        } else {
            0xffff_ffff
        };
        for slice in [1, 2, 5, 1 << 40] {
            let (vm, mut cpu) = setup(path, model);
            run(&vm, &mut cpu, slice);
            // Iteration i (from 0) reads A0 + i, stored by iteration i - 1,
            // and C0 + i + 1, stored by the instruction just before it.
            let ebx = (0..N).fold(0u64, |s, i| s.wrapping_add(u64::from(A0) + i));
            let esi = (0..N).fold(0u64, |s, i| s.wrapping_add(u64::from(C0) + i + 1));
            let what = format!("{path:?} {model:?} slice {slice}");
            assert_eq!(cpu.cpu.gpr[3] & mask, ebx & mask, "{what}");
            assert_eq!(cpu.cpu.gpr[6] & mask, esi & mask, "{what}");
            assert_eq!(cpu.cpu.gpr[5] & mask, 3 * N, "{what}");
            assert_eq!(cpu.cpu.gpr[7] & mask, 0, "{what}");
            assert_eq!(cpu.retired_instructions(), 11 * N + 1, "{what}");
        }
    }
}

#[test]
fn a_store_into_one_block_keeps_its_neighbours_proved() {
    for (path, model) in CASES {
        let (vm, mut cpu) = setup(path, model);
        x86jit_core::Vcpu::take_machine_counters_for_test();
        run(&vm, &mut cpu, 1 << 40);
        let [blocks, _, lifts, proofs, ..] = x86jit_core::Vcpu::take_machine_counters_for_test();
        let what = format!("{path:?} {model:?}: {blocks} blocks, {lifts} lifts, {proofs} proofs");
        // Each store ends its block, so an iteration enters A and B at each
        // of its three starts, all four rewritten by the iteration's stores
        // (B's later starts reach its own patched immediate), and re-lifts
        // them. Block C is lifted once and then proved; dropping every proof
        // on each store would re-lift it every iteration too.
        // Real mode keeps a rewritten page's instructions as singletons
        // checked against their bytes: only the two patched ones re-lift.
        if path == Path::Real {
            assert_eq!(blocks, 11 * N + 1, "{what}");
            assert!(lifts <= 2 * N + 10, "{what}");
        } else {
            assert_eq!(blocks, 5 * N + 1, "{what}");
            assert!(lifts <= 4 * N + 2, "{what}");
        }
        assert!(proofs >= N - 1, "{what}");
    }
}
