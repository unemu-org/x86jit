//! Successor-link witnesses (xj-to9.27): every linked path, run with links and
//! with every link refused (`links::OFF`), under one budget schedule each, ends
//! in the same state, retired count and delivery log, and the linked run
//! follows links.
use super::*;
use crate::{CpuModel, Prot, Reg, RegionKind, VmConfig};

const CODE: u64 = 0x1000;
const GDT: u64 = 0xa000;
const DATA_SELECTORS: usize = MAX_RETAINED_CONTEXTS + 8;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Setup {
    Real(CpuModel),
    Flat32,
    Segmented(CpuModel),
    Paged(CpuModel),
}

fn machine(setup: Setup, code: &[u8], extra: &[(u64, &[u8])]) -> (Vm, Vcpu) {
    let model = match setup {
        Setup::Real(model) | Setup::Segmented(model) | Setup::Paged(model) => model,
        Setup::Flat32 => CpuModel::I80386,
    };
    let mut vm = Vm::new(VmConfig::flat(0x10_0000));
    vm.set_cpu_model(model);
    vm.map(0, 0x10_0000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(CODE, code).unwrap();
    let mut cpu = vm.new_vcpu();
    match setup {
        Setup::Real(_) => {
            for (reg, value) in [
                (Reg::Cs, 0),
                (Reg::Rip, CODE),
                (Reg::Ds, 0),
                (Reg::Es, 0),
                (Reg::Ss, 0),
                (Reg::Rsp, 0x7000),
            ] {
                cpu.set_reg(reg, value);
            }
        }
        Setup::Flat32 | Setup::Segmented(_) | Setup::Paged(_) => {
            let flat = !matches!(setup, Setup::Segmented(_));
            // GDT at A000h: selector 10h + 8k is a data segment based at
            // k * 10h (`DATA_SELECTORS` of them), with the setup's limit.
            cpu.cpu.legacy.gdtr.base = GDT as u32;
            cpu.cpu.legacy.gdtr.limit = (8 * (2 + DATA_SELECTORS) - 1) as u16;
            for k in 0..DATA_SELECTORS {
                let base = (k * 0x10) as u32;
                let [b0, b1, b2, _] = base.to_le_bytes();
                let descriptor = if flat {
                    [0xff, 0xff, b0, b1, b2, 0x93, 0xcf, 0]
                } else {
                    [0xff, 0xff, b0, b1, b2, 0x93, 0, 0]
                };
                vm.write_bytes(GDT + 0x10 + 8 * k as u64, &descriptor)
                    .unwrap();
            }
            cpu.cpu.rip = CODE;
            cpu.cpu.gpr[4] = 0x7000;
            cpu.cpu.legacy.cr0 = 1;
            if matches!(setup, Setup::Paged(_)) {
                // Page directory 8000h, table 9000h: linear pages 0-15 map
                // their own frames, except where a test remaps one.
                cpu.cpu.legacy.cr0 |= 0x8000_0000;
                cpu.cpu.legacy.cr3 = 0x8000;
                vm.write_bytes(0x8000, &0x9007u32.to_le_bytes()).unwrap();
                for page in 0..16u32 {
                    vm.write_bytes(
                        0x9000 + u64::from(page) * 4,
                        &((page << 12) | 7).to_le_bytes(),
                    )
                    .unwrap();
                }
            }
            for (i, cache) in cpu.cpu.legacy.segments[..4].iter_mut().enumerate() {
                cache.base = 0;
                cache.limit = if flat { u32::MAX } else { 0xffff };
                cache.attributes = if i == 0 { 0x9b } else { 0x93 } | if flat { 0x4000 } else { 0 };
            }
            let context =
                derive_legacy_context(model.legacy_model(), &cpu.cpu).expect("protected context");
            assert_eq!(
                context.segment_class,
                if flat {
                    crate::SegmentClass::Flat32
                } else {
                    crate::SegmentClass::Segmented
                },
                "{setup:?}"
            );
        }
    }
    // After the page tables: an extra write may replace an entry.
    for (address, bytes) in extra {
        vm.write_bytes(*address, bytes).unwrap();
    }
    (vm, cpu)
}

struct Outcome {
    gpr: [u64; 16],
    state: String,
    retired: u64,
    log: String,
    followed: u64,
    cleared: u64,
    // Links followed after a near RET (`return_witness.rs`).
    returns: u64,
}

/// Runs `steps` units in runs of `slice` (one run when `None`), with links
/// allowed or refused, on a VM that `vm_setup` finished.
#[allow(clippy::too_many_arguments)]
fn run(
    setup: Setup,
    code: &[u8],
    extra: &[(u64, &[u8])],
    steps: u64,
    slice: Option<u64>,
    links: bool,
    prepare: impl Fn(&mut Vcpu),
    vm_setup: fn(&mut Vm),
) -> Outcome {
    let (mut vm, mut cpu) = machine(setup, code, extra);
    vm_setup(&mut vm);
    cpu.enable_delivery_log(64).unwrap();
    prepare(&mut cpu);
    links::OFF.with(|off| off.set(!links));
    // A first run validates the entry's Direct bus (`validated_direct_bus`);
    // until then the real-mode paths take the Checked lookup.
    let mut done = cpu
        .run_steps(&vm, 1, StepOptions::default())
        .unwrap()
        .steps();
    while done < steps {
        let budget = slice.map_or(steps - done, |slice| slice.min(steps - done));
        let run = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
        assert!(
            matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
            "{setup:?} {:?}",
            run.stop()
        );
        done += run.steps();
    }
    links::OFF.with(|off| off.set(false));
    let links = &cpu.legacy_execution.as_ref().unwrap().links;
    let (followed, cleared, returns) = (links.followed, links.cleared, links.returns);
    Outcome {
        gpr: cpu.cpu.gpr,
        state: format!("{:?}", cpu.cpu),
        retired: cpu.retired,
        log: format!("{:?}", cpu.delivery_log()),
        followed,
        cleared,
        returns,
    }
}

/// The linked run of every schedule matches the refused one, and the
/// unsliced linked run follows links. Returns the reference outcome.
fn witness(
    setup: Setup,
    code: &[u8],
    extra: &[(u64, &[u8])],
    steps: u64,
    links: Option<bool>,
    prepare: impl Fn(&mut Vcpu) + Copy,
) -> Outcome {
    witness_on(setup, code, extra, steps, links, prepare, |_| {})
}

/// `witness` on a VM that `vm_setup` finished.
fn witness_on(
    setup: Setup,
    code: &[u8],
    extra: &[(u64, &[u8])],
    steps: u64,
    links: Option<bool>,
    prepare: impl Fn(&mut Vcpu) + Copy,
    vm_setup: fn(&mut Vm),
) -> Outcome {
    let reference = run(setup, code, extra, steps, None, false, prepare, vm_setup);
    assert_eq!(reference.followed, 0);
    for slice in [None, Some(1), Some(2), Some(7), Some(64)] {
        let linked = run(setup, code, extra, steps, slice, true, prepare, vm_setup);
        assert_eq!(linked.state, reference.state, "{setup:?} slice {slice:?}");
        assert_eq!(
            linked.retired, reference.retired,
            "{setup:?} slice {slice:?}"
        );
        assert_eq!(linked.log, reference.log, "{setup:?} slice {slice:?}");
        if let (None, Some(expected)) = (slice, links) {
            assert_eq!(
                linked.followed != 0,
                expected,
                "{setup:?}: {} links followed",
                linked.followed
            );
        }
        if slice == Some(1) {
            // A one-step budget never links.
            assert_eq!(linked.followed, 0, "{setup:?}");
        }
    }
    reference
}

const REAL: [Setup; 3] = [
    Setup::Real(CpuModel::I8086),
    Setup::Real(CpuModel::I80286),
    Setup::Real(CpuModel::I80386),
];
const PROTECTED: [Setup; 4] = [
    Setup::Flat32,
    Setup::Segmented(CpuModel::I80386),
    Setup::Segmented(CpuModel::I80286),
    Setup::Paged(CpuModel::I80386),
];

#[test]
fn links_follow_direct_exits_on_every_linked_path() {
    // mov cx,100; L: inc bx; dec cx; jnz L; jmp $
    let real = [0xb9, 100, 0, 0x43, 0x49, 0x75, 0xfc, 0xeb, 0xfe];
    // The same with ECX in 32-bit code.
    let flat = [0xb9, 100, 0, 0, 0, 0x43, 0x49, 0x75, 0xfc, 0xeb, 0xfe];
    for setup in REAL.into_iter().chain(PROTECTED) {
        let code: &[u8] = if matches!(setup, Setup::Flat32 | Setup::Paged(_)) {
            &flat
        } else {
            &real
        };
        let outcome = witness(setup, code, &[], 400, Some(true), |_| {});
        let gpr = outcome.gpr;
        assert_eq!(gpr[3] & 0xffff, 100, "{setup:?}");
        assert_eq!(gpr[1] & 0xffff, 0, "{setup:?}");
    }
}

#[test]
fn a_store_into_a_linked_successor_runs_its_new_bytes() {
    // Across blocks. L: jmp T is a direct exit into T, linked after the
    // first iteration; at CX = 8 a later block rewrites T's INC BX to INC AX.
    //   mov cx,16; L: jmp T; T: inc bx; cmp cx,8; jne S;
    //   mov byte [T],40h; S: loop L; jmp $
    let real = [
        0xb9, 16, 0, 0xeb, 0x00, 0x43, 0x83, 0xf9, 0x08, 0x75, 0x05, 0xc6, 0x06, 0x05, 0x10, 0x40,
        0xe2, 0xf1, 0xeb, 0xfe,
    ];
    let flat = [
        0xb9, 16, 0, 0, 0, 0xeb, 0x00, 0x43, 0x83, 0xf9, 0x08, 0x75, 0x07, 0xc6, 0x05, 0x07, 0x10,
        0, 0, 0x40, 0xe2, 0xef, 0xeb, 0xfe,
    ];
    // Same block. L rewrites its own first byte and loops to itself, after a
    // linked count-down W on the same page.
    //   mov cx,8; mov dx,50; W: dec dx; jnz W;
    //   L: inc bx; mov byte [L],40h; loop L; jmp $
    let real_self = [
        0xb9, 8, 0, 0xba, 50, 0, 0x4a, 0x75, 0xfd, 0x43, 0xc6, 0x06, 0x09, 0x10, 0x40, 0xe2, 0xf8,
        0xeb, 0xfe,
    ];
    let flat_self = [
        0xb9, 8, 0, 0, 0, 0xba, 50, 0, 0, 0, 0x4a, 0x75, 0xfd, 0x43, 0xc6, 0x05, 0x0d, 0x10, 0, 0,
        0x40, 0xe2, 0xf6, 0xeb, 0xfe,
    ];
    for setup in REAL.into_iter().chain(PROTECTED) {
        let wide = matches!(setup, Setup::Flat32 | Setup::Paged(_));
        // A block that stores into its own page every iteration starts a new
        // generation each time, and the real-mode Direct class retains such a
        // page as Checked instructions (`smc_pages`): only W links.
        for (code, bx, ax) in [
            (if wide { &flat[..] } else { &real[..] }, 9, 7),
            (if wide { &flat_self[..] } else { &real_self[..] }, 1, 7),
        ] {
            let outcome = witness(setup, code, &[], 300, Some(true), |_| {});
            let gpr = outcome.gpr;
            assert_eq!(gpr[3] & 0xffff, bx, "{setup:?}");
            assert_eq!(gpr[0] & 0xffff, ax, "{setup:?}");
        }
    }
}

#[test]
fn a_remapped_linked_page_runs_its_new_frame() {
    // L: JMP 2000h is a direct exit into linear page 2, frame 5 (INC EBX),
    // linked after the first iteration. At ECX = 8 the PTE is pointed at
    // frame 6 (INC EAX) and the translation flushed: a CR3 reload on the
    // I80386, INVLPG on the I80486 (SDM Vol 3A §4.10.4.1).
    //   mov ecx,16; L: jmp 2000h; R: cmp ecx,8; jne S;
    //   mov dword [9008h],6007h; <flush>; S: loop L; jmp $
    let mut code = vec![
        0xb9, 16, 0, 0, 0, 0xe9, 0xf6, 0x0f, 0, 0, 0x83, 0xf9, 0x08, 0x75, 0x11,
    ];
    code.extend([0xc7, 0x05, 0x08, 0x90, 0, 0, 0x07, 0x60, 0, 0]);
    let flush_offset = code.len();
    code.extend([0; 7]);
    code.extend([0xe2, 0xe3, 0xeb, 0xfe]);
    // inc ebx / inc eax; jmp 100Ah
    let frame5: &[u8] = &[0x43, 0xe9, 0x04, 0xf0, 0xff, 0xff];
    let frame6: &[u8] = &[0x40, 0xe9, 0x04, 0xf0, 0xff, 0xff];
    let pte: &[u8] = &0x5007u32.to_le_bytes();
    for (model, flush) in [
        // mov edx,cr3; mov cr3,edx; nop
        (CpuModel::I80386, [0x0f, 0x20, 0xda, 0x0f, 0x22, 0xda, 0x90]),
        // invlpg [2000h]
        (CpuModel::I80486Dx, [0x0f, 0x01, 0x3d, 0x00, 0x20, 0, 0]),
    ] {
        code[flush_offset..flush_offset + 7].copy_from_slice(&flush);
        let setup = Setup::Paged(model);
        let extra = [(0x5000, frame5), (0x6000, frame6), (0x9008, pte)];
        let outcome = witness(setup, &code, &extra, 300, Some(true), |_| {});
        let gpr = outcome.gpr;
        assert_eq!(gpr[3], 9, "{model:?}");
        assert_eq!(gpr[0], 7, "{model:?}");
    }
}

#[test]
fn interrupts_and_single_steps_at_link_points_keep_their_boundaries() {
    // INT 20h at CX = 32, then STI, which opens the window for the IRQ 21h
    // queued before the run (handlers: INC DX / INC SI, IRET).
    //   mov cx,64; L: inc bx; cmp cx,32; jne S; int 20h; sti; S: loop L; jmp $
    let interrupts = [
        0xb9, 64, 0, 0x43, 0x83, 0xf9, 0x20, 0x75, 0x03, 0xcd, 0x20, 0xfb, 0xe2, 0xf5, 0xeb, 0xfe,
    ];
    // At CX = 8, PUSHF; POP AX; OR AH,1; PUSH AX; POPF sets TF: every later
    // instruction traps to INC DI, IRET.
    //   mov cx,32; L: inc bx; cmp cx,8; jne S; <set TF>; S: loop L; jmp $
    let single_step = [
        0xb9, 32, 0, 0x43, 0x83, 0xf9, 0x08, 0x75, 0x07, 0x9c, 0x58, 0x80, 0xcc, 0x01, 0x50, 0x9d,
        0xe2, 0xf1, 0xeb, 0xfe,
    ];
    let ivt: &[u8] = &[
        0x20, 0x20, 0, 0, // vector 1: 0000:2020h
    ];
    let vectors: &[u8] = &[
        0x00, 0x20, 0, 0, // vector 20h: 0000:2000h
        0x10, 0x20, 0, 0, // vector 21h: 0000:2010h
    ];
    let handlers: &[(u64, &[u8])] = &[
        (4, ivt),
        (0x80, vectors),
        (0x2000, &[0x42, 0xcf]),
        (0x2010, &[0x46, 0xcf]),
        (0x2020, &[0x47, 0xcf]),
    ];
    for setup in REAL {
        let outcome = witness(setup, &interrupts, handlers, 400, Some(true), |cpu| {
            cpu.inject_irq(0x21)
        });
        let gpr = outcome.gpr;
        assert_eq!(gpr[3] & 0xffff, 64, "{setup:?}");
        assert_eq!(gpr[2] & 0xffff, 1, "{setup:?}");
        assert!(outcome.log.contains("Irq"), "{setup:?} {}", outcome.log);
        let outcome = witness(setup, &single_step, handlers, 400, Some(true), |_| {});
        assert!(outcome.log.contains("Trap"), "{setup:?} {}", outcome.log);
    }
}

static NMI: std::sync::Mutex<Option<crate::NmiLine>> = std::sync::Mutex::new(None);
static EXIT: std::sync::Mutex<Option<crate::ExitRequest>> = std::sync::Mutex::new(None);

#[test]
fn nmi_and_exit_requests_inside_a_linked_run_stop_at_the_same_boundary() {
    // mov cx,32; L: inc bx; loop L; inc cx; inc cx; inc cx; jmp $. The loop
    // links to itself; the lift of the block after it raises an NMI edge, or
    // a host exit request, from the source-read hook, mid-run. The NMI
    // handler (vector 2) is INC DX, IRET.
    let code = [0xb9, 32, 0, 0x43, 0xe2, 0xfd, 0x41, 0x41, 0x41, 0xeb, 0xfe];
    let handlers: &[(u64, &[u8])] = &[(8, &[0x00, 0x20, 0, 0]), (0x2000, &[0x42, 0xcf])];
    for setup in [Setup::Real(CpuModel::I80286), Setup::Real(CpuModel::I80386)] {
        for nmi in [true, false] {
            let mut outcomes = Vec::new();
            for links in [false, true] {
                let (vm, mut cpu) = machine(setup, &code, handlers);
                cpu.enable_delivery_log(16).unwrap();
                links::OFF.with(|off| off.set(!links));
                // The entry and loop blocks are lifted here.
                cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
                cpu.run_steps(&vm, 10, StepOptions::default()).unwrap();
                *NMI.lock().unwrap() = cpu.nmi_line();
                *EXIT.lock().unwrap() = Some(cpu.exit_request());
                *vm.source_read_hook.lock().unwrap() = Some(if nmi {
                    |_| NMI.lock().unwrap().as_ref().unwrap().raise()
                } else {
                    |_| EXIT.lock().unwrap().as_ref().unwrap().request_exit()
                });
                let run = cpu.run_steps(&vm, 200, StepOptions::default()).unwrap();
                links::OFF.with(|off| off.set(false));
                let followed = cpu.legacy_execution.as_ref().unwrap().links.followed;
                assert_eq!(followed != 0, links, "{setup:?}");
                if nmi {
                    assert_eq!(cpu.cpu.gpr[2] & 0xffff, 1, "{setup:?}");
                    assert_eq!(run.steps(), 200, "{setup:?}");
                } else {
                    assert!(
                        matches!(run.stop(), StepStop::Exit(Exit::Interrupted)),
                        "{setup:?} {:?}",
                        run.stop()
                    );
                }
                outcomes.push((
                    run.steps(),
                    format!("{:?}", cpu.cpu),
                    cpu.retired,
                    format!("{:?}", cpu.delivery_log()),
                ));
            }
            assert_eq!(outcomes[0], outcomes[1], "{setup:?} nmi {nmi}");
        }
    }
}

#[test]
fn links_never_outlive_their_run() {
    // mov cx,0; L: inc bx; loop L; jmp $, then INC BX patched to INC AX
    // between runs, in this VM and in a fork the vCPU then runs in.
    let code = [0xb9, 0, 0, 0x43, 0xe2, 0xfd, 0xeb, 0xfe];
    for setup in REAL.into_iter().chain([
        Setup::Segmented(CpuModel::I80386),
        Setup::Segmented(CpuModel::I80286),
    ]) {
        let (vm, mut cpu) = machine(setup, &code, &[]);
        cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        cpu.run_steps(&vm, 499, StepOptions::default()).unwrap();
        assert_ne!(cpu.legacy_execution.as_ref().unwrap().links.followed, 0);
        let bx = cpu.cpu.gpr[3];
        vm.write_bytes(CODE + 3, &[0x40]).unwrap();
        cpu.run_steps(&vm, 500, StepOptions::default()).unwrap();
        assert_eq!(cpu.cpu.gpr[3], bx, "{setup:?}");
        assert_eq!(cpu.cpu.gpr[0] & 0xffff, 250, "{setup:?}");
        let child = vm
            .fork_with_backend(Box::new(super::super::InterpreterBackend))
            .unwrap();
        child.write_bytes(CODE + 3, &[0x46]).unwrap();
        cpu.run_steps(&child, 500, StepOptions::default()).unwrap();
        assert_eq!(cpu.cpu.gpr[0] & 0xffff, 250, "{setup:?}");
        assert_eq!(cpu.cpu.gpr[6] & 0xffff, 250, "{setup:?}");
        cpu.run_steps(&vm, 500, StepOptions::default()).unwrap();
        assert_eq!(cpu.cpu.gpr[0] & 0xffff, 500, "{setup:?}");
        assert_eq!(cpu.cpu.gpr[6] & 0xffff, 250, "{setup:?}");
    }
}

/// xj-to9.45: a store into one block of a page drops only that block's
/// proof, and no link outlives the generation advance that drops it (links
/// name the generation, so a selective drop ends every link by design). A
/// linked count-down W shares the page with B, whose immediate low byte the
/// loop stores; W stays proved across the stores and B runs its new bytes.
#[test]
fn a_selective_proof_drop_ends_links_and_keeps_its_neighbours_proved() {
    //   mov cx,8; L: mov dx,20; W: dec dx; jnz W; mov [B+1],cl; jmp B;
    //   B: mov ax,imm; add bx,ax; loop L; jmp $
    let code = |w: usize| {
        let imm = |v: u32| v.to_le_bytes()[..w].to_vec();
        let mut c = vec![0xb9];
        c.extend(imm(8));
        let l = c.len();
        c.push(0xba);
        c.extend(imm(20));
        c.extend([0x4a, 0x75, 0xfd, 0x88, if w == 4 { 0x0d } else { 0x0e }]);
        let at = c.len();
        c.extend(imm(0));
        c.extend([0xeb, 0x00]);
        let b = c.len();
        c[at..at + w].copy_from_slice(&imm(CODE as u32 + b as u32 + 1));
        c.push(0xb8);
        c.extend(imm(1 << (8 * w - 4)));
        c.extend([0x01, 0xc3, 0xe2]);
        c.push((l as isize - c.len() as isize - 1) as u8);
        c.extend([0xeb, 0xfe]);
        c
    };
    for setup in PROTECTED {
        let w = if matches!(setup, Setup::Flat32 | Setup::Paged(_)) {
            4
        } else {
            2
        };
        let outcome = witness(setup, &code(w), &[], 600, Some(true), |_| {});
        let mask = if w == 4 { 0xffff_ffff } else { 0xffff };
        // Iterations see CL = 8, 7, ... 1 in the immediate's low byte.
        let ebx = (8 << (8 * w - 4)) + 36;
        assert_eq!(outcome.gpr[3] & mask, ebx, "{setup:?}");
        // Proofs: each iteration re-lifts B only. Dropping every proof on
        // each store would re-lift W, L and the store's block too.
        let (vm, mut cpu) = machine(setup, &code(w), &[]);
        cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        cpu.run_steps(&vm, 599, StepOptions::default()).unwrap();
        let execution = cpu.legacy_execution.as_ref().unwrap();
        let lifts = match setup {
            Setup::Paged(_) => execution.paging.as_ref().unwrap().lifts,
            _ => execution.flat32.lifts,
        };
        // Under 32: at most the first lifts and about one per iteration.
        assert!(lifts <= 2 * 8 + 4, "{setup:?}: {lifts} lifts");
    }
}

/// A call-heavy loop whose block starts lie a few bytes apart (the
/// `real16_loops` dos-mixed body): its links are made once and followed,
/// not remade on every visit.
#[test]
fn neighbouring_blocks_keep_their_links() {
    // cld; mov si,1000h; mov di,2000h; mov cx,32; rep movsw; mov bx,1000h;
    // mov cx,16; body: lodsw; add ax,[bx]; xor ah,al; shl ax,1; shr dx,1;
    // adc dx,ax; and ax,7fffh; stosw; inc bx; push ax; call sub; pop ax;
    // test al,1; jz +1; inc dx; loop body; jmp start;
    // sub: push bx; add bx,dx; ror bx,1; xor al,bh; pop bx; ret
    let mut code = vec![
        0xfc, 0xbe, 0, 0x10, 0xbf, 0, 0x20, 0xb9, 32, 0, 0xf3, 0xa5, 0xbb, 0, 0x10, 0xb9, 16, 0,
    ];
    let body = code.len();
    code.extend([
        0xad, 0x03, 0x07, 0x30, 0xc4, 0xd1, 0xe0, 0xd1, 0xea, 0x11, 0xc2, 0x25, 0xff, 0x7f, 0xab,
        0x43, 0x50, 0xe8, 0, 0, 0x58, 0xa8, 1, 0x74, 1, 0x42,
    ]);
    let call = code.len() - 9;
    code.extend([0xe2, (body as isize - code.len() as isize - 2) as u8]);
    let back = (-(code.len() as i16 + 3)).to_le_bytes();
    code.extend([0xe9, back[0], back[1]]);
    let target = (code.len() as i16 - (call + 3) as i16).to_le_bytes();
    code[call + 1..call + 3].copy_from_slice(&target);
    code.extend([0x53, 0x01, 0xd3, 0xd1, 0xcb, 0x30, 0xf8, 0x5b, 0xc3]);
    let (vm, mut cpu) = machine(Setup::Real(CpuModel::I8086), &code, &[]);
    for (reg, value) in [(Reg::Ds, 0x1000), (Reg::Es, 0x2000), (Reg::Ss, 0x4000)] {
        cpu.set_reg(reg, value);
    }
    cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
    cpu.run_steps(&vm, 100_000, StepOptions::default()).unwrap();
    let links = &cpu.legacy_execution.as_ref().unwrap().links;
    assert!(links.made < 32, "{} made", links.made);
    assert!(links.followed > 10_000, "{} followed", links.followed);
}

#[test]
fn a_block_entered_in_two_contexts_follows_each_context_s_own_link() {
    // S reads DS:[7000h] and is entered by a direct jump with DS = 10h (base
    // 0) and with DS = 18h (base 10h), alternately: each context's lookup of
    // S, and of what follows it, returns that context's block.
    //   mov cx,64; L: mov ax,10h; mov ds,ax; xor bp,bp; jmp S;
    //   B: mov ax,18h; mov ds,ax; mov bp,1; jmp S; R: loop L; jmp $;
    //   S: mov bx,[7000h]; add si,bx; test bp,bp; jz B; jmp R
    let narrow = [
        0xb9, 64, 0, 0xb8, 0x10, 0, 0x8e, 0xd8, 0x31, 0xed, 0xeb, 0x0e, 0xb8, 0x18, 0, 0x8e, 0xd8,
        0xbd, 1, 0, 0xeb, 0x04, 0xe2, 0xeb, 0xeb, 0xfe, 0x8b, 0x1e, 0, 0x70, 0x01, 0xde, 0x85,
        0xed, 0x74, 0xe8, 0xeb, 0xf0,
    ];
    // The same in 32-bit code.
    let wide = [
        0xb9, 64, 0, 0, 0, 0x66, 0xb8, 0x10, 0, 0x8e, 0xd8, 0x31, 0xed, 0xeb, 0x11, 0x66, 0xb8,
        0x18, 0, 0x8e, 0xd8, 0xbd, 1, 0, 0, 0, 0xeb, 0x04, 0xe2, 0xe7, 0xeb, 0xfe, 0x8b, 0x1d, 0,
        0x70, 0, 0, 0x01, 0xde, 0x85, 0xed, 0x74, 0xe3, 0xeb, 0xee,
    ];
    let data: &[(u64, &[u8])] = &[(0x7000, &[1, 0, 0, 0]), (0x7010, &[0, 1, 0, 0])];
    for setup in PROTECTED {
        let code: &[u8] = if matches!(setup, Setup::Flat32 | Setup::Paged(_)) {
            &wide
        } else {
            &narrow
        };
        let outcome = witness(setup, code, data, 64 * 18 + 20, Some(true), |_| {});
        assert_eq!(outcome.gpr[6] & 0xffff, 64 * 0x101, "{setup:?}");
    }
}

#[test]
fn a_reused_context_identity_is_never_followed() {
    // Each outer iteration loads DS with DATA_SELECTORS data selectors in
    // turn (bases 0, 10h, ..), more contexts than are retained
    // (MAX_RETAINED_CONTEXTS): the first context's identity is evicted, and
    // it returns under another on the next outer iteration. S, a self-loop
    // entered by direct branches, adds DS:[7000h] = k + 1 three times under
    // the first selector (k = 0) and under the one at k = MAX_RETAINED_CONTEXTS,
    // whose context is created by an eviction; a link of S made under one
    // identity must not be followed under another.
    //   mov cx,30; O: mov dx,10h; P: mov ax,dx; mov ds,ax; mov bp,3;
    //   cmp dx,10h; je S; cmp dx,LATER; je S; jmp Q; nop; nop;
    //   S: add si,[7000h]; dec bp; jnz S; jmp Q; nop; nop;
    //   Q: add dx,8; cmp dx,END; jb P; loop O; jmp $
    let later = (0x10 + 8 * MAX_RETAINED_CONTEXTS as u16).to_le_bytes();
    let end = (0x10 + 8 * DATA_SELECTORS as u16).to_le_bytes();
    let code = [
        0xb9, 30, 0, 0xba, 0x10, 0, 0x89, 0xd0, 0x8e, 0xd8, 0xbd, 3, 0, 0x83, 0xfa, 0x10, 0x74,
        0x0a, 0x81, 0xfa, later[0], later[1], 0x74, 0x04, 0xeb, 0x0d, 0x90, 0x90, 0x03, 0x36, 0,
        0x70, 0x4d, 0x75, 0xf9, 0xeb, 0x02, 0x90, 0x90, 0x83, 0xc2, 0x08, 0x81, 0xfa, end[0],
        end[1], 0x72, 0xd6, 0xe2, 0xd1, 0xeb, 0xfe,
    ];
    let words: Vec<[u8; 2]> = (0..DATA_SELECTORS as u16)
        .map(|k| (k + 1).to_le_bytes())
        .collect();
    let data: Vec<(u64, &[u8])> = words
        .iter()
        .enumerate()
        .map(|(k, word)| (0x7000 + 0x10 * k as u64, &word[..]))
        .collect();
    const { assert!(DATA_SELECTORS > MAX_RETAINED_CONTEXTS) };
    // 30 outer iterations of 2 + 18 + 20 + (DATA_SELECTORS - 2) * 11 steps.
    let steps = 30 * (2 + 18 + 20 + (DATA_SELECTORS as u64 - 2) * 11) + 100;
    for model in [CpuModel::I80386, CpuModel::I80286] {
        let setup = Setup::Segmented(model);
        let outcome = witness(setup, &code, &data, steps, Some(true), |_| {});
        let later = 3 * (MAX_RETAINED_CONTEXTS as u64 + 1);
        assert_eq!(
            outcome.gpr[6] & 0xffff,
            (30 * (3 + later)) & 0xffff,
            "{model:?}"
        );
        assert_ne!(outcome.cleared, 0, "{model:?}: no identity was evicted");
    }
}

#[test]
fn retained_twenty_bit_pages_never_link() {
    // A rewrites B's immediate every iteration; B jumps to C past a gap and
    // C loops back to A, so C and A are entered by direct exits on turns that
    // drained no store. Once the page is retained (`retained20`), its blocks
    // are checked against their bytes on every lookup and no link stands in.
    //   mov cx,200; A: mov byte [B+1],2Ah; B: mov al,0; add bx,ax; jmp C;
    //   nop x4; C: dec cx; jnz A; jmp $
    let code = [
        0xb9, 200, 0, 0xc6, 0x06, 0x09, 0x10, 0x2a, 0xb0, 0, 0x01, 0xc3, 0xeb, 0x04, 0x90, 0x90,
        0x90, 0x90, 0x49, 0x75, 0xee, 0xeb, 0xfe,
    ];
    for model in [CpuModel::I8088, CpuModel::I8086, CpuModel::NecV30] {
        let setup = Setup::Real(model);
        let outcome = witness(setup, &code, &[], 200 * 6 + 20, None, |_| {});
        assert_eq!(outcome.gpr[3] & 0xffff, 200 * 0x2a, "{model:?}");
        let (vm, mut cpu) = machine(setup, &code, &[]);
        cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        cpu.run_steps(&vm, 200 * 6, StepOptions::default()).unwrap();
        let execution = cpu.legacy_execution.as_ref().unwrap();
        assert_ne!(execution.retained20.len(), 0, "{model:?}");
        assert_eq!(execution.links.followed, 0, "{model:?}");
    }
}

// The generation witness's rewrite: once the guest has stored the marker,
// the next source read of a lift (`Vm::after_source_read`) rewrites B's
// INC BX to INC AX. The resolver drains that store itself, so the turn after
// it sees no dirty word and B's page is never retained.
const MARKER: u64 = 0x6000;
const REWRITTEN: u64 = 0x1003;
fn rewrite_after_marker(vm: &Vm) {
    let mut marker = [0];
    vm.read_bytes(MARKER, &mut marker).unwrap();
    if marker[0] == 0 {
        *vm.source_read_hook.lock().unwrap() = Some(rewrite_after_marker);
        return;
    }
    let mut byte = [0];
    vm.read_bytes(REWRITTEN, &mut byte).unwrap();
    if byte[0] == 0x43 {
        vm.write_bytes(REWRITTEN, &[0x40]).unwrap();
    }
}

#[test]
fn a_store_drained_inside_a_resolver_ends_its_links() {
    // B is linked from the loop's LOOP. At CX = 20 the guest stores the
    // marker and jumps to C, whose first lift rewrites B: every later B runs
    // INC AX. Only the publication generation tells the link that its block
    // left the cache (`Link::generation`).
    //   mov cx,40; B: inc bx; cmp cx,20; jne S; mov byte [6000h],1; jmp C;
    //   S: loop B; jmp $; (12 nops); C: nop; jmp S
    let mut code = vec![
        0xb9, 40, 0, 0x43, 0x83, 0xf9, 0x14, 0x75, 0x07, 0xc6, 0x06, 0x00, 0x60, 0x01, 0xeb, 0x10,
        0xe2, 0xf1, 0xeb, 0xfe,
    ];
    code.resize(0x20, 0x90);
    code.extend([0x90, 0xeb, 0xed]);
    let setups = [
        Setup::Real(CpuModel::I80286),
        Setup::Real(CpuModel::I80386),
        Setup::Segmented(CpuModel::I80386),
        Setup::Segmented(CpuModel::I80286),
    ];
    for setup in setups {
        let outcome = witness_on(
            setup,
            &code,
            &[],
            40 * 5 + 40,
            Some(true),
            |_| {},
            |vm| {
                *vm.source_read_hook.lock().unwrap() = Some(rewrite_after_marker);
            },
        );
        assert_eq!(outcome.gpr[3] & 0xffff, 21, "{setup:?}");
        assert_eq!(outcome.gpr[0] & 0xffff, 19, "{setup:?}");
    }
}

#[test]
fn the_same_ip_under_two_code_segments_follows_each_segment_s_link() {
    // Two copies of one loop at 0100:0000 and 0200:0000 that differ only in
    // Q (INC BX / INC SI) and far-jump to each other: Q is entered by a
    // direct jump at the same IP in both, so only the physical fetch address
    // tells their links apart (`Link::physical`).
    //   dec cx; jz done; jmp Q; nop; Q: inc bx | inc si; jmp far other:0;
    //   (nops); done: jmp $
    let segment = |q: u8, other: u8| {
        let mut code = vec![0x49, 0x74, 0x1d, 0xeb, 0x01, 0x90, q, 0xea, 0, 0, 0, other];
        code.resize(0x20, 0x90);
        code.extend([0xeb, 0xfe]);
        code
    };
    let a = segment(0x43, 0x02);
    let b = segment(0x46, 0x01);
    for setup in REAL {
        let outcome = witness(setup, &a, &[(0x2000, &b)], 40 * 5 + 20, Some(true), |cpu| {
            cpu.set_reg(Reg::Cs, 0x100);
            // The real-mode CS cache the 80286 and 80386 fetch through.
            cpu.cpu.legacy.segments[SegmentId::Cs as usize].base = 0x1000;
            cpu.set_reg(Reg::Rip, 0);
            cpu.set_reg(Reg::Rcx, 40);
        });
        assert_eq!(outcome.gpr[3] & 0xffff, 20, "{setup:?}");
        assert_eq!(outcome.gpr[6] & 0xffff, 19, "{setup:?}");
    }
}

#[test]
fn retained_twenty_bit_pages_under_a_fetch_policy_never_link() {
    // Under a fetch policy a retained twenty-bit page keeps the shared
    // resolver (`retained20` is false), whose blocks are published and would
    // link. The store into L drops it once, which retains the page
    // (`smc_pages`); the self-loop L then runs on turns that drain no store.
    // The links refuse the retained page (`link_path`): the counter shows it,
    // since a link there would return the block the resolver returns.
    //   mov cx,1; L: inc bx; dec cx; jnz L; mov byte [L],43h; mov cx,200; jmp L
    let code = [
        0xb9, 1, 0, 0x43, 0x49, 0x75, 0xfc, 0xc6, 0x06, 0x03, 0x10, 0x43, 0xb9, 200, 0, 0xeb, 0xf2,
    ];
    for model in [CpuModel::I8088, CpuModel::I8086, CpuModel::NecV30] {
        let setup = Setup::Real(model);
        witness_on(
            setup,
            &code,
            &[],
            1000,
            Some(false),
            |_| {},
            |vm| {
                vm.set_fetch_policy(Box::new(|_| crate::FetchVerdict::Allow));
            },
        );
        let (mut vm, mut cpu) = machine(setup, &code, &[]);
        vm.set_fetch_policy(Box::new(|_| crate::FetchVerdict::Allow));
        cpu.run_steps(&vm, 1, StepOptions::default()).unwrap();
        cpu.run_steps(&vm, 1000, StepOptions::default()).unwrap();
        let execution = cpu.legacy_execution.as_ref().unwrap();
        assert!(execution.smc_pages.contains(&1), "{model:?}");
        assert_eq!(execution.links.followed, 0, "{model:?}");
    }
}

#[test]
fn a_translation_flush_ends_every_link() {
    // S adds DS:[7000h] eight times with DS base 0, the run's first context.
    // DS is then loaded with base 10h and CR3 written, which flushes the
    // translations (`invalidate_translations`): identities and the proof map
    // version restart, so the flushed context takes the first context's
    // identity again. S must not follow the link made before the flush.
    //   mov cx,8; xor bp,bp; jmp S; P: mov ax,18h; mov ds,ax; mov eax,cr3;
    //   mov cr3,eax; mov cx,8; mov bp,1; jmp S; done: jmp $;
    //   S: add si,[7000h]; dec cx; jnz S; test bp,bp; jz P; jmp done
    let code = [
        0xb9, 8, 0, 0x31, 0xed, 0xeb, 0x15, 0xb8, 0x18, 0, 0x8e, 0xd8, 0x0f, 0x20, 0xd8, 0x0f,
        0x22, 0xd8, 0xb9, 8, 0, 0xbd, 1, 0, 0xeb, 0x02, 0xeb, 0xfe, 0x03, 0x36, 0, 0x70, 0x49,
        0x75, 0xf9, 0x85, 0xed, 0x74, 0xe0, 0xeb, 0xf1,
    ];
    let data: &[(u64, &[u8])] = &[(0x7000, &[1, 0]), (0x7010, &[0, 1])];
    let setup = Setup::Segmented(CpuModel::I80386);
    let outcome = witness(setup, &code, data, 8 * 3 * 2 + 40, Some(true), |_| {});
    assert_eq!(outcome.gpr[6] & 0xffff, 8 + 8 * 0x100);
}

/// Ports and the handled MMIO page of the device witness.
const DEVICE_MMIO: u64 = 0xc000;

#[derive(Default)]
struct Recorder {
    effects: Vec<(crate::DeviceSpace, u64, crate::DeviceDirection, u8, u64)>,
}
impl crate::DeviceIo for Recorder {
    fn access(
        &mut self,
        r: &crate::DeviceRequest,
        _: &mut crate::DeviceContext<'_>,
    ) -> crate::DeviceResult {
        let answer = 0x35 + self.effects.len() as u64 % 7;
        self.effects
            .push((r.space, r.address, r.direction, r.size, r.value));
        crate::DeviceResult::Handled(answer)
    }
}

#[test]
fn device_transfers_between_linked_blocks_keep_their_order() {
    // Port OUT and IN, a handled MMIO store and load, REP OUTSB and REP INSB
    // (served handoffs and in-block transfers) run between the loop's
    // direct exits; the linked run serves the same requests in the same
    // order and ends in the same state.
    //   mov bp,40; L: out 80h,al; in al,80h; add bx,ax; mov [C000h],al;
    //   mov dl,[C001h]; add bx,dx; mov cx,2; mov si,5000h; rep outsb;
    //   mov cx,2; mov di,5100h; rep insb; dec bp; jnz L; jmp $
    let code = [
        0xbd, 40, 0, 0xe6, 0x80, 0xe4, 0x80, 0x01, 0xc3, 0xa2, 0x00, 0xc0, 0x8a, 0x16, 0x01, 0xc0,
        0x01, 0xd3, 0xb9, 2, 0, 0xbe, 0x00, 0x50, 0xf3, 0x6e, 0xb9, 2, 0, 0xbf, 0x00, 0x51, 0xf3,
        0x6c, 0x4d, 0x75, 0xde, 0xeb, 0xfe,
    ];
    // The NEC V30 stands for the twenty-bit bus: the 8086 has no INS/OUTS.
    for model in [CpuModel::NecV30, CpuModel::I80286, CpuModel::I80386] {
        let mut outcomes = Vec::new();
        for (links, slice) in [
            (false, None),
            (true, None),
            (true, Some(7)),
            (true, Some(64)),
        ] {
            let mut vm = Vm::new(VmConfig::flat(0x10_0000));
            vm.set_cpu_model(model);
            vm.map(0, DEVICE_MMIO as usize, Prot::RWX, RegionKind::Ram)
                .unwrap();
            vm.map(DEVICE_MMIO, 0x1000, Prot::RW, RegionKind::Trap)
                .unwrap();
            vm.map(
                DEVICE_MMIO + 0x1000,
                0x10_0000 - 0xd000,
                Prot::RWX,
                RegionKind::Ram,
            )
            .unwrap();
            vm.write_bytes(CODE, &code).unwrap();
            let mut cpu = vm.new_vcpu();
            for (reg, value) in [
                (Reg::Cs, 0),
                (Reg::Rip, CODE),
                (Reg::Ds, 0),
                (Reg::Es, 0),
                (Reg::Ss, 0),
                (Reg::Rsp, 0x7000),
            ] {
                cpu.set_reg(reg, value);
            }
            vm.register_device_range(crate::DeviceRange::Ports(0..65536))
                .unwrap();
            vm.register_device_range(crate::DeviceRange::Mmio(DEVICE_MMIO..DEVICE_MMIO + 0x1000))
                .unwrap();
            let mut device = Recorder::default();
            links::OFF.with(|off| off.set(!links));
            cpu.run_steps_with_devices(&vm, 1, StepOptions::default(), Some(&mut device))
                .unwrap();
            let mut done = 1;
            while done < 40 * 14 {
                let budget = slice.unwrap_or(40 * 14 - done).min(40 * 14 - done);
                let run = cpu
                    .run_steps_with_devices(&vm, budget, StepOptions::default(), Some(&mut device))
                    .unwrap();
                assert!(
                    matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
                    "{model:?} {:?}",
                    run.stop()
                );
                done += run.steps();
            }
            links::OFF.with(|off| off.set(false));
            let followed = cpu.legacy_execution.as_ref().unwrap().links.followed;
            if links && slice.is_none() {
                assert_ne!(followed, 0, "{model:?}");
            }
            outcomes.push((
                format!("{:?}", cpu.cpu),
                cpu.retired,
                format!("{:?}", device.effects),
            ));
        }
        for outcome in &outcomes[1..] {
            assert_eq!(outcome, &outcomes[0], "{model:?}");
        }
        assert!(outcomes[0].2.contains("Mmio"), "{model:?}");
    }
}

#[path = "return_witness.rs"]
mod returns;
