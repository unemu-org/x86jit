//! Self-modifying code through a declared code alias (`Memory::declare_code_alias`):
//! two guest views of one host page set, code fetched through one view and
//! patched through the other. A store through either view is seen by the next
//! fetch of the bytes it changed, inside the running block too (spec.md §10;
//! decision-23 excludes a prefetch queue), on the twenty-bit models, the
//! 80286 and 80386 in real mode, protected 16-bit code (80286 Segmented and
//! Checked, 80386), 80386/80486 paged 16-bit code and the Application Real16
//! personality, for every budget slicing; for a block that never entered the
//! cache too (a failed source barrier), with three views, and for a store
//! that spans two pages.
#![cfg(unix)]

use std::os::fd::{AsRawFd, FromRawFd};
use x86jit_core::{
    CpuMode, CpuModel, Exit, Prot, RegionKind, StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

const LEN: usize = 0x20000;
/// The two views of one 16 KiB shared file.
const VIEWS: [u32; 2] = [0x4000, 0x18000];
/// Three views of it, declared as a chain: the first and the last are
/// aliases only through the middle one.
const VIEWS3: [u32; 3] = [0x4000, 0xc000, 0x18000];
const VIEW_LEN: usize = 0x4000;
/// The code's offset in either view, and its segment offset.
const BASE: u16 = 0x3c00;
const BODY: u16 = BASE + 3;
const PD: u32 = 0x10000;
const PT: u32 = 0x11000;
const N: u8 = 40;

#[derive(Clone, Copy, Debug)]
enum Path {
    Twenty(CpuModel),
    Real(CpuModel),
    Protected16 {
        model: CpuModel,
        checked: bool,
    },
    Paged16(CpuModel),
    /// The Application personality's Real16 mode: no CPU model.
    Application,
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

fn shared_file() -> std::fs::File {
    let mut name = std::env::temp_dir()
        .join("x86jit-alias-smc-XXXXXX")
        .as_os_str()
        .as_encoded_bytes()
        .to_vec();
    name.push(0);
    // SAFETY: mkstemp takes a writable NUL-terminated template; the
    // descriptor moves to File, and unlink removes only its fresh name.
    unsafe {
        let fd = libc::mkstemp(name.as_mut_ptr().cast());
        assert!(fd >= 0);
        assert_eq!(libc::unlink(name.as_ptr().cast()), 0);
        let file = std::fs::File::from_raw_fd(fd);
        file.set_len(VIEW_LEN as u64).unwrap();
        file
    }
}

/// A Machine VM whose RAM at both `VIEWS` is one shared page set, declared
/// to the engine as a code alias.
fn alias_vm(path: Path) -> Vm {
    views_vm(path, &VIEWS, &[(0, 1)])
}

/// A VM for `path` whose RAM at every view in `views` is one shared page
/// set; each `(i, j)` declares views `i` and `j` a code alias. The process
/// barrier is constructed available, so `with_barrier_flush_failure` keeps
/// every newly lifted block out of the cache.
fn views_vm(path: Path, views: &[u32], pairs: &[(usize, usize)]) -> Vm {
    let file = shared_file();
    // SAFETY: a fresh anonymous reservation; MAP_FIXED replaces only pages
    // inside it with two views of the same file.
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
        for &view in views {
            let at = ptr.cast::<u8>().add(view as usize).cast();
            assert_eq!(
                libc::mmap(
                    at,
                    VIEW_LEN,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    file.as_raw_fd(),
                    0,
                ),
                at
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
    let mut vm = x86jit_core::with_barrier_success(|| {
        Vm::with_backend_host_ram(
            VmConfig::reserved(LEN as u64),
            Box::new(x86jit_core::InterpreterBackend),
            ram,
        )
    })
    .unwrap();
    match model(path) {
        Some(model) => vm.set_cpu_model(model),
        None => vm.set_cpu_mode(CpuMode::Real16),
    }
    vm.map(0, LEN, Prot::RWX, RegionKind::Ram).unwrap();
    for &(i, j) in pairs {
        vm.mem
            .declare_code_alias(u64::from(views[i]), u64::from(views[j]), VIEW_LEN)
            .unwrap();
    }
    vm
}

/// A vCPU fetching through view `code` (CS base) and addressing data and its
/// stack through view `data` (DS and SS bases), both with offsets `BASE..`.
fn vcpu(vm: &Vm, path: Path, code: u32, data: u32) -> Vcpu {
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = u64::from(BASE);
    cpu.cpu.gpr[4] = 0x3000;
    match path {
        Path::Twenty(_) | Path::Application => {
            cpu.cpu.cs = (code >> 4) as u16;
            cpu.cpu.ds = (data >> 4) as u16;
            cpu.cpu.ss = (data >> 4) as u16;
        }
        Path::Real(_) => {
            let data = (data >> 4) as u16;
            let selectors = [(code >> 4) as u16, data, 0, data, 0, 0];
            let entry = x86jit_core::CpuState::legacy_real_entry(selectors);
            cpu.cpu.cs = entry.cs;
            cpu.cpu.ds = entry.ds;
            cpu.cpu.ss = entry.ss;
            cpu.cpu.legacy = entry.legacy;
            cpu.set_a20_enabled(true);
        }
        Path::Protected16 { .. } | Path::Paged16(_) => {
            cpu.set_a20_enabled(true);
            cpu.cpu.legacy.cr0 = 1;
            for (i, cache) in cpu.cpu.legacy.segments.iter_mut().enumerate() {
                cache.base = match i {
                    0 => code,
                    1 | 3 => data,
                    _ => 0,
                };
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
                for page in 0..(LEN >> 12) as u32 {
                    vm.write_bytes(u64::from(PT + 4 * page), &((page << 12) | 7).to_le_bytes())
                        .unwrap();
                }
                cpu.cpu.legacy.cr3 = PD;
                cpu.cpu.legacy.cr0 = 0x8000_0001;
            }
        }
    }
    cpu
}

/// Run to HLT in `slice`-step slices; `uncached` fails every source
/// barrier, so each block runs once, uncached. Otherwise the barrier
/// succeeds without a host command, on every host.
fn run_as(vm: &Vm, cpu: &mut Vcpu, slice: u64, uncached: bool) {
    if !uncached {
        return x86jit_core::with_barrier_success(|| run(vm, cpu, slice));
    }
    let before = vm.memory_stats().source_barrier_uncached_executions;
    x86jit_core::with_barrier_flush_failure(|| run(vm, cpu, slice));
    assert!(vm.memory_stats().source_barrier_uncached_executions > before);
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

// MOV CX,N; the body at BODY; LOOP to the body; HLT.
fn looped(body: &[u8]) -> Vec<u8> {
    let mut code = vec![0xb9, N, 0];
    code.extend_from_slice(body);
    code.push(0xe2);
    code.push((-(body.len() as i8) - 2) as u8);
    code.push(0xf4);
    code
}

fn inc(target: u16) -> Vec<u8> {
    let [lo, hi] = target.to_le_bytes();
    vec![0xfe, 0x06, lo, hi] // INC byte [target]
}
const ADD_DL_1: [u8; 3] = [0x80, 0xc2, 0x01];

fn sum(f: impl Fn(u8) -> u8) -> u8 {
    (1..=N).fold(0u8, |acc, i| acc.wrapping_add(f(i)))
}

/// (name, loop body, expected DL)
fn cases() -> Vec<(&'static str, Vec<u8>, u8)> {
    vec![
        // The next instruction of the storing block: iteration i adds 1 + i.
        (
            "next",
            [&inc(BODY + 6)[..], &ADD_DL_1].concat(),
            sum(|i| 1 + i),
        ),
        // A locked read-modify-write into the next instruction.
        (
            "locked next",
            [&[0xf0][..], &inc(BODY + 7), &ADD_DL_1].concat(),
            sum(|i| 1 + i),
        ),
        // The same increment, its target past a JMP that ends the block.
        (
            "later block",
            [&inc(BODY + 8)[..], &[0xeb, 0], &ADD_DL_1].concat(),
            sum(|i| 1 + i),
        ),
        // The storing instruction's own immediate: ADD byte [imm],imm doubles
        // it each iteration; ADD DL,[imm] accumulates 2^i.
        (
            "own bytes",
            [
                &[0x80, 0x06, (BODY + 4) as u8, ((BODY + 4) >> 8) as u8, 1][..],
                &[0x02, 0x16, (BODY + 4) as u8, ((BODY + 4) >> 8) as u8],
            ]
            .concat(),
            sum(|i| 1u8.checked_shl(u32::from(i)).unwrap_or(0)),
        ),
    ]
}

/// Every ordered pair of distinct views.
fn directions(views: &[u32]) -> Vec<(u32, u32)> {
    let mut pairs = Vec::new();
    for &code in views {
        for &data in views {
            if code != data {
                pairs.push((code, data));
            }
        }
    }
    pairs
}

/// The loop cases on `paths`, with the views of `views_vm(_, views, pairs)`.
fn loops(paths: &[Path], views: &[u32], pairs: &[(usize, usize)], uncached: bool, slices: &[u64]) {
    let mut failed = Vec::new();
    for &path in paths {
        let vm = views_vm(path, views, pairs);
        for (code_view, data_view) in directions(views) {
            for (name, body, dl) in cases() {
                let code = looped(&body);
                for &slice in slices {
                    vm.write_bytes(u64::from(code_view + u32::from(BASE)), &code)
                        .unwrap();
                    let mut cpu = vcpu(&vm, path, code_view, data_view);
                    run_as(&vm, &mut cpu, slice, uncached);
                    let got = cpu.cpu.gpr[2] as u8;
                    if got != dl {
                        failed.push(format!(
                            "{path:?} {name}: code at {code_view:#x}, stores through \
                             {data_view:#x}, slice {slice}: DL {got} != {dl}"
                        ));
                    }
                }
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
fn store_through_one_view_is_fetched_through_the_other() {
    loops(&PATHS, &VIEWS, &[(0, 1)], false, &[u64::MAX, 1, 3]);
}

/// The paths whose blocks consult the source barrier before they enter the
/// cache, the Application personality included: the 80286 Checked and the
/// paged walks lift without it.
fn uncached_paths() -> Vec<Path> {
    let mut paths: Vec<_> = PATHS
        .into_iter()
        .filter(|path| {
            !matches!(
                path,
                Path::Protected16 { checked: true, .. } | Path::Paged16(_)
            )
        })
        .collect();
    paths.push(Path::Application);
    paths
}

#[test]
fn uncached_block_sees_a_store_through_the_other_view() {
    // A block the failed source barrier kept out of the cache: its lift
    // ends before its walk starts (spec.md §10).
    loops(&uncached_paths(), &VIEWS, &[(0, 1)], true, &[u64::MAX, 3]);
}

#[test]
fn store_through_a_third_view_is_fetched_through_the_first() {
    loops(&PATHS, &VIEWS3, &[(0, 1), (1, 2)], false, &[u64::MAX]);
    loops(
        &uncached_paths(),
        &VIEWS3,
        &[(0, 1), (1, 2)],
        true,
        &[u64::MAX],
    );
}

fn model(path: Path) -> Option<CpuModel> {
    match path {
        Path::Twenty(model) | Path::Real(model) | Path::Paged16(model) => Some(model),
        Path::Protected16 { model, .. } => Some(model),
        Path::Application => None,
    }
}

#[test]
fn system_table_store_through_one_view_is_fetched_through_the_other() {
    // 80286 PRM §B SIDT, SDM Vol 2B SIDT: a 16-bit SIDT stores the limit
    // word, then the 24-bit base, then one more byte. The stored image is
    // MOV DL,2Ah; JMP $+4 over its last byte; the original is MOV DL,1 and
    // the same JMP, so DL tells which bytes the next instruction ran.
    const X: u16 = BASE + 5;
    let [lo, hi] = X.to_le_bytes();
    let code = [
        &[0x0f, 0x01, 0x0e, lo, hi][..], // SIDT [X]
        &[0xb2, 0x01, 0xeb, 0x02, 0x00, 0x00],
        &[0xf4],
    ]
    .concat();
    let mut failed = Vec::new();
    for path in PATHS {
        if matches!(path, Path::Twenty(_)) {
            continue;
        }
        let vm = alias_vm(path);
        for (code_view, data_view) in [(VIEWS[0], VIEWS[1]), (VIEWS[1], VIEWS[0])] {
            for slice in [u64::MAX, 1] {
                vm.write_bytes(u64::from(code_view + u32::from(BASE)), &code)
                    .unwrap();
                let mut cpu = vcpu(&vm, path, code_view, data_view);
                cpu.cpu.legacy.idtr.limit = 0x2ab2;
                cpu.cpu.legacy.idtr.base = 0x02eb;
                run_as(&vm, &mut cpu, slice, false);
                let got = cpu.cpu.gpr[2] as u8;
                if got != 0x2a {
                    failed.push(format!(
                        "{path:?}: code at {code_view:#x}, stores through \
                         {data_view:#x}, slice {slice}: DL {got:#x}"
                    ));
                }
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Run `code` (at `BASE`, ending in HLT) on every path in both directions;
/// its store writes the immediate at `BASE + imm` of a following
/// MOV DX,1234h, so DX equals the stored word only if that instruction ran
/// its new bytes.
fn one_shot(code: &[u8], imm: u16, sp: u16, paths: impl Fn(Path) -> bool) {
    let paths: Vec<_> = PATHS.into_iter().filter(|&path| paths(path)).collect();
    one_shot_at(&paths, BASE, code, imm, sp, false);
}

/// `one_shot` with the code at offset `base` of either view, on `paths`;
/// `uncached` keeps every block out of the cache. AX holds 0BEEFh.
fn one_shot_at(paths: &[Path], base: u16, code: &[u8], imm: u16, sp: u16, uncached: bool) {
    let mut failed = Vec::new();
    for &path in paths {
        let vm = alias_vm(path);
        for (code_view, data_view) in [(VIEWS[0], VIEWS[1]), (VIEWS[1], VIEWS[0])] {
            for slice in [u64::MAX, 1] {
                vm.write_bytes(u64::from(code_view + u32::from(base)), code)
                    .unwrap();
                let mut cpu = vcpu(&vm, path, code_view, data_view);
                cpu.cpu.rip = u64::from(base);
                cpu.cpu.gpr[0] = 0xbeef;
                cpu.cpu.gpr[4] = u64::from(sp);
                run_as(&vm, &mut cpu, slice, uncached);
                let at = u64::from(code_view + u32::from(base + imm));
                let stored = vm.mem.read(at, 2).unwrap();
                let got = cpu.cpu.gpr[2] as u16;
                if u64::from(got) != stored || stored == 0x1234 {
                    failed.push(format!(
                        "{path:?}: code at {code_view:#x}, stores through \
                         {data_view:#x}, slice {slice}, uncached {uncached}: \
                         DX {got:#x}, stored {stored:#x}"
                    ));
                }
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
fn stack_store_through_one_view_is_fetched_through_the_other() {
    // PUSHF stores FLAGS at SS:SP-2, the next instruction's immediate.
    // PUSHF; MOV DX,1234h; HLT
    one_shot(&[0x9c, 0xba, 0x34, 0x12, 0xf4], 2, BASE + 4, |_| true);
}

#[test]
fn x87_store_through_one_view_is_fetched_through_the_other() {
    // FNSTCW m16 stores the control word over the next instruction's
    // immediate; real-address paths (the protected ones would need an
    // ESC-capable CR0 and IDT, which this fixture does not build).
    let [lo, hi] = (BASE + 7).to_le_bytes();
    // FNINIT; FNSTCW [BASE+7]; MOV DX,1234h; HLT
    one_shot(
        &[0xdb, 0xe3, 0xd9, 0x3e, lo, hi, 0xba, 0x34, 0x12, 0xf4],
        7,
        0x3000,
        |path| matches!(path, Path::Twenty(_) | Path::Real(_)),
    );
}

#[test]
fn application_stack_store_through_one_view_is_fetched_through_the_other() {
    // The Application Real16 personality runs PUSHF through its shared
    // helper on the raw memory. PUSHF; MOV DX,1234h; HLT
    for uncached in [false, true] {
        one_shot_at(
            &[Path::Application],
            BASE,
            &[0x9c, 0xba, 0x34, 0x12, 0xf4],
            2,
            BASE + 4,
            uncached,
        );
    }
}

#[test]
fn store_spanning_two_pages_is_fetched_through_the_other_view() {
    // The block crosses a page edge of either view; the word store writes
    // the next instruction's immediate, one byte on each page.
    // MOV [P-1],AX; MOV DX,1234h; HLT, with P = 3000h the page edge.
    const AT: u16 = 0x3000 - 5;
    let [lo, hi] = (AT + 4).to_le_bytes();
    let code = [0xa3, lo, hi, 0xba, 0x34, 0x12, 0xf4];
    let mut paths = PATHS.to_vec();
    paths.push(Path::Application);
    one_shot_at(&paths, AT, &code, 4, 0x3000, false);
    one_shot_at(&uncached_paths(), AT, &code, 4, 0x3000, true);
}

#[test]
fn views_follow_the_memory_across_threads_and_vms() {
    // The walk maps a running block's views per thread
    // (`Memory::running_view_reach`): two memories with different views at
    // the same addresses, and a vCPU moved to another thread mid-loop, each
    // see only their own declarations.
    let path = Path::Twenty(CpuModel::I8086);
    let two = views_vm(path, &VIEWS, &[(0, 1)]);
    // A second, unrelated declaration: both memories have made as many.
    two.mem.declare_code_alias(0x1000, 0x2000, 0x100).unwrap();
    let three = views_vm(path, &VIEWS3, &[(0, 1), (1, 2)]);
    let (_, body, dl) = cases().remove(0);
    let code = looped(&body);
    let start = |vm: &Vm, code_view: u32, data_view: u32| {
        vm.write_bytes(u64::from(code_view + u32::from(BASE)), &code)
            .unwrap();
        vcpu(vm, path, code_view, data_view)
    };
    let finish = |vm: &Vm, mut cpu: Vcpu| {
        run_as(vm, &mut cpu, 3, false);
        cpu.cpu.gpr[2] as u8
    };
    // Part of the loop here, the rest on another thread.
    let half = |vm: &Vm, cpu: &mut Vcpu| {
        for _ in 0..N / 2 {
            let r = x86jit_core::with_barrier_success(|| {
                cpu.run_steps(vm, 3, StepOptions::default()).unwrap()
            });
            assert!(matches!(r.stop(), StepStop::Exit(Exit::BudgetExhausted)));
        }
    };
    let mut a = start(&two, VIEWS[0], VIEWS[1]);
    half(&two, &mut a);
    let mut b = start(&three, VIEWS3[0], VIEWS3[2]);
    half(&three, &mut b);
    std::thread::scope(|scope| {
        let a = scope.spawn(|| finish(&two, a));
        let b = scope.spawn(|| finish(&three, b));
        assert_eq!(a.join().unwrap(), dl);
        assert_eq!(b.join().unwrap(), dl);
    });
    // The same block, on one thread, in either memory.
    assert_eq!(finish(&two, start(&two, VIEWS[0], VIEWS[1])), dl);
    assert_eq!(finish(&three, start(&three, VIEWS3[0], VIEWS3[1])), dl);
}

#[test]
fn uncached_block_tags_end_with_its_lift_without_aliases() {
    // With no alias declared, a block that never entered the cache leaves no
    // code tag once its lift ends, whatever the thread runs next.
    let path = Path::Twenty(CpuModel::I8086);
    let first = views_vm(path, &VIEWS, &[]);
    let second = views_vm(path, &VIEWS, &[]);
    let (_, body, dl) = cases().remove(0);
    let code = looped(&body);
    let page = u64::from(VIEWS[0] + u32::from(BASE)) >> 12;
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                for (vm, uncached) in [(&first, true), (&second, false)] {
                    vm.write_bytes(u64::from(VIEWS[0] + u32::from(BASE)), &code)
                        .unwrap();
                    let mut cpu = vcpu(vm, path, VIEWS[0], VIEWS[0]);
                    run_as(vm, &mut cpu, u64::MAX, uncached);
                    assert_eq!(cpu.cpu.gpr[2] as u8, dl);
                }
                assert!(!first.mem.is_code_page(page));
            })
            .join()
            .unwrap();
    });
    assert!(!first.mem.is_code_page(page));
}
