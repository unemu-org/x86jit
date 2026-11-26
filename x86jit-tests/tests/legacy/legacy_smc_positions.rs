//! Self-modifying code at every position of a block (xj-to9.31): a loop whose
//! body stores into its own instructions — an earlier one, the storing one,
//! the next one, a later one, the last byte of a sixteen-instruction block,
//! a length-changing opcode, bytes across a 4 KiB page boundary — on the
//! twenty-bit models, the 80286 and 80386 in real mode, protected 16-bit code
//! (80286 Segmented and Checked, 80386) and 80386 paged 16-bit code. Each
//! store is seen by the next fetch of the bytes it changed (the engine's SMC
//! contract, spec.md §10; decision-23 excludes a prefetch queue), so every
//! result is the analytic one, for every budget slicing.
use x86jit_core::{CpuModel, Exit, Prot, RegionKind, StepOptions, StepStop, Vcpu, Vm, VmConfig};

const N: u8 = 40;
// LODSB data: byte k at LODS + k.
const LODS: u16 = 0x7d00;
const SIZE: u64 = 0x20000;
const PD: u32 = 0x10000;
const PT: u32 = 0x11000;

#[derive(Clone, Copy, Debug)]
enum Path {
    Twenty(CpuModel),
    Real(CpuModel),
    Protected16 { model: CpuModel, checked: bool },
    Paged16(CpuModel),
}

const PATHS: [Path; 11] = [
    Path::Twenty(CpuModel::I8088),
    Path::Twenty(CpuModel::I8086),
    Path::Twenty(CpuModel::NecV20),
    Path::Twenty(CpuModel::NecV30),
    Path::Real(CpuModel::I80286),
    Path::Real(CpuModel::I80386),
    Path::Protected16 {
        model: CpuModel::I80286,
        checked: false,
    },
    Path::Protected16 {
        model: CpuModel::I80286,
        checked: true,
    },
    Path::Protected16 {
        model: CpuModel::I80386,
        checked: false,
    },
    Path::Paged16(CpuModel::I80386),
    Path::Paged16(CpuModel::I80486Dx),
];

fn setup(path: Path, base: u16, code: &[u8]) -> (Vm, Vcpu) {
    let model = match path {
        Path::Twenty(model) | Path::Real(model) | Path::Paged16(model) => model,
        Path::Protected16 { model, .. } => model,
    };
    let mut vm = Vm::new(VmConfig::flat(SIZE));
    vm.set_cpu_model(model);
    vm.map(0, SIZE as usize, Prot::RWX, RegionKind::Ram)
        .unwrap();
    vm.write_bytes(u64::from(base), code).unwrap();
    vm.write_bytes(u64::from(LODS), &(0..64).collect::<Vec<u8>>())
        .unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = u64::from(base);
    cpu.cpu.gpr[4] = 0x7000;
    cpu.cpu.gpr[6] = u64::from(LODS);
    match path {
        Path::Twenty(_) => {}
        Path::Real(_) => {
            cpu.cpu.legacy = x86jit_core::CpuState::legacy_real_entry([0; 6]).legacy;
            cpu.set_a20_enabled(true);
        }
        Path::Protected16 { .. } | Path::Paged16(_) => {
            cpu.set_a20_enabled(true);
            cpu.cpu.legacy.cr0 = 1;
            for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
                cache.base = 0;
                cache.limit = 0xffff;
                cache.attributes = if i == 0 { 0x9b } else { 0x93 };
                cache.unusable = 0;
            }
            // The 80286 Checked reference: a GS cache without its P bit
            // (never addressed here) keeps the context out of Segmented.
            if matches!(path, Path::Protected16 { checked: true, .. }) {
                cpu.cpu.legacy.segments[5].attributes &= !0x80;
            }
            if matches!(path, Path::Paged16(_)) {
                // 80386 PRM §5.2 pp. 5-7–11: identity-map the RAM, present,
                // writable, user.
                vm.write_bytes(u64::from(PD), &(PT | 7).to_le_bytes())
                    .unwrap();
                for page in 0..(SIZE >> 12) as u32 {
                    vm.write_bytes(u64::from(PT + 4 * page), &((page << 12) | 7).to_le_bytes())
                        .unwrap();
                }
                cpu.cpu.legacy.cr3 = PD;
                cpu.cpu.legacy.cr0 = 0x8000_0001;
            }
        }
    }
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

// MOV CX,N; the body at `base + 3`; LOOP to the body; HLT.
fn looped(base: u16, body: &[u8]) -> Vec<u8> {
    let mut code = vec![0xb9, N, 0];
    code.extend_from_slice(body);
    code.push(0xe2);
    code.push((-(body.len() as i8) - 2) as u8);
    code.push(0xf4);
    assert!(usize::from(base) + code.len() < 0x10000);
    code
}

fn inc(target: u16) -> Vec<u8> {
    let [lo, hi] = target.to_le_bytes();
    vec![0xfe, 0x06, lo, hi] // INC byte [target]
}
fn store(op: u8, modrm: u8, target: u16, imm: u8) -> Vec<u8> {
    let [lo, hi] = target.to_le_bytes();
    vec![op, modrm, lo, hi, imm]
}
const ADD_DL_1: [u8; 3] = [0x80, 0xc2, 0x01];

struct Case {
    name: &'static str,
    base: u16,
    body: Vec<u8>,
    dl: u8,
    al: u8,
    /// A byte the loop leaves in memory, and its value.
    memory: Option<(u16, u8)>,
}

fn sum(f: impl Fn(u8) -> u8) -> u8 {
    (1..=N).fold(0u8, |acc, i| acc.wrapping_add(f(i)))
}

fn cases() -> Vec<Case> {
    let body = 0x7c03;
    let concat = |parts: &[&[u8]]| parts.concat();
    vec![
        Case {
            // The next instruction: iteration i adds 1 + i.
            name: "next",
            base: 0x7c00,
            body: concat(&[&inc(body + 6), &ADD_DL_1]),
            dl: sum(|i| 1 + i),
            al: 0,
            memory: None,
        },
        Case {
            name: "later",
            base: 0x7c00,
            body: concat(&[&inc(body + 8), &[0x90, 0x90], &ADD_DL_1]),
            dl: sum(|i| 1 + i),
            al: 0,
            memory: None,
        },
        Case {
            // An instruction before the store, the block's first: seen from
            // the next iteration.
            name: "before",
            base: 0x7c00,
            body: concat(&[&ADD_DL_1, &inc(body + 2)]),
            dl: sum(|i| i),
            al: 0,
            memory: None,
        },
        Case {
            // The storing instruction's own immediate: ADD byte [imm],imm
            // doubles it; ADD DL,[imm] accumulates 2^i.
            name: "self",
            base: 0x7c00,
            body: concat(&[&store(0x80, 0x06, body + 4, 1), &[0x02, 0x16, 0x07, 0x7c]]),
            dl: sum(|i| 1u8.checked_shl(u32::from(i)).unwrap_or(0)),
            al: 0,
            memory: Some((body + 4, 0)),
        },
        Case {
            // Sixteen instructions: the changed byte is the block's last.
            name: "last-of-sixteen",
            base: 0x7c00,
            body: concat(&[&inc(body + 4 + 14 + 2), &[0x90; 14], &ADD_DL_1]),
            dl: sum(|i| 1 + i),
            al: 0,
            memory: None,
        },
        Case {
            // The LOOP displacement, the last byte, rewritten unchanged.
            name: "same-value-last-byte",
            base: 0x7c00,
            body: concat(&[&store(0xc6, 0x06, body + 8 + 1, 0xf6), &ADD_DL_1]),
            dl: N,
            al: 0,
            memory: None,
        },
        Case {
            // XOR toggles 90h (NOP) and 04h (ADD AL,imm8 consuming the
            // following 90h): the instruction boundaries after it change.
            name: "length-change",
            base: 0x7c00,
            body: concat(&[
                &store(0x80, 0x36, body + 5, 0x94),
                &[0x90, 0x90, 0x90],
                &ADD_DL_1,
            ]),
            dl: N,
            al: 0x90u8.wrapping_mul(N.div_ceil(2)),
            memory: None,
        },
        Case {
            // The first instruction's opcode: NOP, NOP and INC DX, NOP.
            name: "first-opcode",
            base: 0x7c00,
            body: concat(&[&[0x90, 0x90], &store(0x80, 0x36, body, 0xd2)]),
            dl: N / 2,
            al: 0,
            memory: None,
        },
        Case {
            // Several stores per iteration: into an earlier instruction (seen
            // next iteration), the instruction after the stores and a later
            // one. Iteration i adds i + 2 (1 + i).
            name: "several",
            base: 0x7c00,
            body: concat(&[
                &ADD_DL_1,
                &inc(body + 2),
                &inc(body + 17),
                &inc(body + 21),
                &ADD_DL_1,
                &[0x90],
                &ADD_DL_1,
            ]),
            dl: sum(|i| 3 * i + 2),
            al: 0,
            memory: None,
        },
        Case {
            // XOR toggles NOP and LODSB after a NOP: a string past the
            // block's entry ends the block before it, so its byte is fetched
            // but outside the block's instructions; a change there relifts
            // after the whole block. Odd iterations load bytes 0, 1, ... 19.
            name: "string-after-entry",
            base: 0x7c00,
            body: concat(&[&store(0x80, 0x36, body + 6, 0x3c), &[0x90, 0x90], &ADD_DL_1]),
            dl: N,
            al: N / 2 - 1,
            memory: None,
        },
        Case {
            // Data on the code's page, outside every block.
            name: "data-beside-code",
            base: 0x7c00,
            body: concat(&[&inc(0x7e00), &ADD_DL_1]),
            dl: N,
            al: 0,
            memory: Some((0x7e00, N)),
        },
        Case {
            // The changed instruction starts the next 4 KiB page.
            name: "next-page",
            base: 0x7ff8,
            body: concat(&[&inc(0x8002), &[0x90], &ADD_DL_1]),
            dl: sum(|i| 1 + i),
            al: 0,
            memory: None,
        },
        Case {
            // The changed instruction straddles the page boundary.
            name: "across-pages",
            base: 0x7ff7,
            body: concat(&[&inc(0x8001), &[0x90], &ADD_DL_1]),
            dl: sum(|i| 1 + i),
            al: 0,
            memory: None,
        },
    ]
}

#[test]
fn stores_into_every_position_of_a_block_are_fetched() {
    for case in cases() {
        let code = looped(case.base, &case.body);
        for path in PATHS {
            let mut retired = None;
            for slice in [u64::MAX, 1, 2, 3, 7, 64] {
                let what = format!("{} {path:?} slice {slice}", case.name);
                let (vm, mut cpu) = setup(path, case.base, &code);
                cpu.cpu.ds = 0;
                run(&vm, &mut cpu, slice);
                assert_eq!(cpu.cpu.gpr[2] as u8, case.dl, "{what}");
                assert_eq!(cpu.cpu.gpr[0] as u8, case.al, "{what}");
                if let Some((at, value)) = case.memory {
                    assert_eq!(
                        vm.mem.read(u64::from(at), 1).unwrap(),
                        u64::from(value),
                        "{what}"
                    );
                }
                let count = cpu.retired_instructions();
                assert_eq!(*retired.get_or_insert(count), count, "{what}");
                // The host rewrites the program once its pages have taken
                // self-modifying stores: the next run fetches the new bytes.
                vm.write_bytes(u64::from(case.base), &code).unwrap();
                cpu.cpu.rip = u64::from(case.base);
                cpu.cpu.gpr[0] = 0;
                cpu.cpu.gpr[2] = 0;
                cpu.cpu.gpr[6] = u64::from(LODS);
                if let Some((at, _)) = case.memory {
                    let offset = usize::from(at.wrapping_sub(case.base));
                    vm.write_bytes(u64::from(at), &[code.get(offset).copied().unwrap_or(0)])
                        .unwrap();
                }
                run(&vm, &mut cpu, slice);
                assert_eq!(cpu.cpu.gpr[2] as u8, case.dl, "{what}, rewritten");
                assert_eq!(cpu.cpu.gpr[0] as u8, case.al, "{what}, rewritten");
            }
        }
    }
}

#[test]
fn a_host_write_reaches_a_block_of_a_self_modifying_page() {
    // After its own stores have dropped blocks of the page, a host write
    // between runs replaces the INC with NOPs and the ADD immediate with 10h.
    let base = 0x7c00;
    let code = looped(base, &[&inc(0x7c09)[..], &ADD_DL_1].concat());
    for path in PATHS {
        let (vm, mut cpu) = setup(path, base, &code);
        run(&vm, &mut cpu, u64::MAX);
        assert_eq!(cpu.cpu.gpr[2] as u8, sum(|i| 1 + i), "{path:?}");
        vm.write_bytes(0x7c03, &[0x90; 4]).unwrap();
        vm.write_bytes(0x7c09, &[0x10]).unwrap();
        cpu.cpu.rip = base.into();
        cpu.cpu.gpr[2] = 0;
        run(&vm, &mut cpu, u64::MAX);
        assert_eq!(cpu.cpu.gpr[2] as u8, 0x10u8.wrapping_mul(N), "{path:?}");
    }
}

// The twenty-bit models' own wraps: one loop whose stored-into ADD straddles
// the sixteen-bit IP wrap (CS 1000h, IP FFFEh to 0000h; 8086 Family UM §2.2
// p. 2-7) and one whose ADD straddles the twenty-bit bus wrap (FFFFFh to
// 00000h; §2.3 p. 2-12). Each block ends at the wrap.
#[test]
fn twenty_bit_stores_across_the_ip_and_bus_wraps() {
    for (cs, ip, ds, target) in [(0x1000u16, 0xfff3u16, 0x1000u16, 0u16), (0xffff, 3, 0, 0)] {
        let body = [&inc(target)[..], &[0x90; 4], &ADD_DL_1].concat();
        let code = looped(0, &body);
        for path in PATHS {
            let Path::Twenty(model) = path else { continue };
            for slice in [u64::MAX, 1, 3] {
                let mut vm = Vm::new(VmConfig::flat(0x100000));
                vm.set_cpu_model(model);
                vm.map(0, 0x100000, Prot::RWX, RegionKind::Ram).unwrap();
                for (k, byte) in code.iter().enumerate() {
                    let offset = ip.wrapping_add(k as u16);
                    let linear = ((u64::from(cs) << 4) + u64::from(offset)) & 0xfffff;
                    vm.write_bytes(linear, &[*byte]).unwrap();
                }
                let mut cpu = vm.new_vcpu();
                cpu.cpu.cs = cs;
                cpu.cpu.ds = ds;
                cpu.cpu.rip = u64::from(ip);
                run(&vm, &mut cpu, slice);
                let what = format!("{cs:x}:{ip:x} {path:?} slice {slice}");
                assert_eq!(cpu.cpu.gpr[2] as u8, sum(|i| 1 + i), "{what}");
                assert_eq!(
                    cpu.cpu.rip,
                    u64::from(ip.wrapping_add(code.len() as u16)),
                    "{what}"
                );
            }
        }
    }
}

// A store through a declared code alias (two guest views of one host page,
// as a board's mirrored RAM) into the next block. Its page is
// self-modifying, but it stays with the shared cache, whose code tags on
// both views drop the stored-into block (spec.md §10).
#[cfg(unix)]
#[test]
fn twenty_bit_stores_through_a_code_alias_reach_the_next_block() {
    use std::os::fd::{AsRawFd, FromRawFd};
    const LEN: usize = 0x20000;
    for path in PATHS {
        let Path::Twenty(model) = path else { continue };
        let file = {
            let mut name = std::env::temp_dir()
                .join("x86jit-smc20-XXXXXX")
                .as_os_str()
                .as_encoded_bytes()
                .to_vec();
            name.push(0);
            // SAFETY: mkstemp takes a writable NUL-terminated template; the
            // descriptor moves to File, and unlink removes only its name.
            unsafe {
                let fd = libc::mkstemp(name.as_mut_ptr().cast());
                assert!(fd >= 0);
                assert_eq!(libc::unlink(name.as_ptr().cast()), 0);
                let file = std::fs::File::from_raw_fd(fd);
                file.set_len(0x4000).unwrap();
                file
            }
        };
        // SAFETY: a fresh anonymous reservation; MAP_FIXED replaces only
        // pages inside it with two views of the same file.
        let ptr = unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                LEN,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            assert_ne!(ptr, libc::MAP_FAILED);
            for offset in [0x4000, 0x18000] {
                assert_eq!(
                    libc::mmap(
                        ptr.cast::<u8>().add(offset).cast(),
                        0x4000,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED | libc::MAP_FIXED,
                        file.as_raw_fd(),
                        0,
                    ),
                    ptr.cast::<u8>().add(offset).cast()
                );
            }
            ptr.cast::<u8>()
        };
        let ram = x86jit_core::HostRam {
            ptr,
            len: LEN,
            guest_base: 0,
            dtor: Box::new(|ptr, len| {
                // SAFETY: the reservation above, including its views.
                assert_eq!(unsafe { libc::munmap(ptr.cast(), len) }, 0);
            }),
            protect: None,
        };
        let mut vm = Vm::with_backend_host_ram(
            VmConfig::reserved(LEN as u64),
            Box::new(x86jit_core::InterpreterBackend),
            ram,
        )
        .unwrap();
        vm.set_cpu_model(model);
        vm.map(0, LEN, Prot::RWX, RegionKind::Ram).unwrap();
        vm.mem.declare_code_alias(0x4000, 0x18000, 0x4000).unwrap();
        // The "next" loop at 4000h, a JMP to the ADD ending the storing
        // block; DS 1400h makes each offset X address 14000h + X, the second
        // view of 4000h + (X - 4000h).
        let base = 0x4000;
        let code = looped(
            base,
            &[&inc(base + 3 + 8)[..], &[0xeb, 0], &ADD_DL_1].concat(),
        );
        vm.write_bytes(u64::from(base), &code).unwrap();
        for slice in [u64::MAX, 1, 3] {
            vm.write_bytes(u64::from(base), &code).unwrap();
            let mut cpu = vm.new_vcpu();
            cpu.cpu.ds = 0x1400;
            cpu.cpu.rip = u64::from(base);
            run(&vm, &mut cpu, slice);
            assert_eq!(
                cpu.cpu.gpr[2] as u8,
                sum(|i| 1 + i),
                "{path:?} slice {slice}"
            );
        }
    }
}
