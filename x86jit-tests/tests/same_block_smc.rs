//! SDM Vol 3A §14.6, Self-Modifying Code: writes affecting prefetched/decoded
//! instructions invalidate them. The engine observes the new bytes after the
//! storing instruction completes, independently of dispatch and step budgets.

use x86jit_core::{
    CpuMode, Exit, InterpreterBackend, Prot, Reg, RegionKind, StepOptions, StepStop, Vcpu, Vm,
    VmConfig,
};

const START: u64 = 0x7c00;

fn setup(mode: CpuMode, code: &[u8]) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x10000), Box::new(InterpreterBackend));
    vm.set_cpu_mode(mode);
    vm.set_tier_up_after(None);
    vm.map(0, 0x10000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.write_bytes(START, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, START);
    (vm, cpu)
}

fn forward(mode: CpuMode) -> (Vm, Vcpu) {
    let code: &[u8] = match mode {
        // mov byte [7c06h],42h; mov al,0; hlt
        CpuMode::Real16 => &[0xc6, 0x06, 0x06, 0x7c, 0x42, 0xb0, 0, 0xf4],
        // mov byte [rip+1],42h; mov al,0; hlt
        CpuMode::Long64 => &[0xc6, 0x05, 1, 0, 0, 0, 0x42, 0xb0, 0, 0xf4],
        CpuMode::Compat32 => &[0xc6, 0x05, 0x08, 0x7c, 0, 0, 0x42, 0xb0, 0, 0xf4],
    };
    setup(mode, code)
}

fn stepped(budget: u64) {
    let (vm, mut cpu) = forward(CpuMode::Real16);
    let mut total = 0;
    for _ in 0..4 {
        let result = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
        assert!(result.steps() <= budget);
        total += result.steps();
        match result.stop() {
            StepStop::Exit(Exit::Hlt) => {
                assert_eq!(cpu.reg(Reg::Rax), 0x42);
                assert_eq!(total, 3);
                assert_eq!(cpu.retired_instructions(), 3);
                return;
            }
            StepStop::Exit(Exit::BudgetExhausted) => {}
            other => panic!("{other:?}"),
        }
    }
    panic!("did not halt");
}

#[test]
fn real16_budget_1() {
    stepped(1);
}

#[test]
fn real16_budget_2() {
    stepped(2);
}

#[test]
fn real16_budget_64() {
    stepped(64);
}

fn unbounded(mode: CpuMode) {
    let (vm, mut cpu) = forward(mode);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rax), 0x42, "{mode:?}");
    assert_eq!(cpu.retired_instructions(), 3, "{mode:?}");
}

#[test]
fn real16_run() {
    unbounded(CpuMode::Real16);
}

#[test]
fn compat32_run() {
    unbounded(CpuMode::Compat32);
}

#[test]
fn long64_run() {
    unbounded(CpuMode::Long64);
}

#[test]
fn segmented_real16() {
    for budget in [1, 2, 64] {
        let (vm, mut cpu) = forward(CpuMode::Real16);
        cpu.set_reg(Reg::Cs, 0x700);
        cpu.set_reg(Reg::Rip, 0xc00);
        for _ in 0..4 {
            let result = cpu.run_steps(&vm, budget, StepOptions::default()).unwrap();
            match result.stop() {
                StepStop::Exit(Exit::BudgetExhausted) => {}
                StepStop::Exit(Exit::Hlt) => break,
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(cpu.reg(Reg::Rax), 0x42, "budget={budget}");
        assert_eq!(cpu.reg(Reg::Rip), 0xc08);
        assert_eq!(cpu.retired_instructions(), 3);
    }
    let (vm, mut cpu) = forward(CpuMode::Real16);
    cpu.set_reg(Reg::Cs, 0x700);
    cpu.set_reg(Reg::Rip, 0xc00);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rax), 0x42);
    assert_eq!(cpu.retired_instructions(), 3);
}

#[test]
fn forward_store_retires_before_refetch() {
    for mode in [CpuMode::Real16, CpuMode::Compat32, CpuMode::Long64] {
        let (vm, mut cpu) = forward(mode);
        let next = START + if mode == CpuMode::Real16 { 5 } else { 7 };
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        assert_eq!(cpu.reg(Reg::Rip), next, "{mode:?}");
        assert_eq!(cpu.retired_instructions(), 1, "{mode:?}");
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rax), 0x42, "{mode:?}");
        assert_eq!(cpu.retired_instructions(), 3, "{mode:?}");
    }
}

#[test]
fn earlier_instruction_does_not_end_block() {
    for mode in [CpuMode::Real16, CpuMode::Compat32, CpuMode::Long64] {
        // mov al,0; mov byte [bx/ebx/rbx],42h; mov al,17h; hlt.
        let addressing = if mode == CpuMode::Real16 { 0x07 } else { 0x03 };
        let (vm, mut cpu) = setup(mode, &[0xb0, 0, 0xc6, addressing, 0x42, 0xb0, 0x17, 0xf4]);
        cpu.set_reg(Reg::Rbx, START + 1);
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::Hlt), "{mode:?}");
        assert_eq!(cpu.reg(Reg::Rax), 0x17, "{mode:?}");
        assert_eq!(cpu.retired_instructions(), 4, "{mode:?}");
        assert_eq!(vm.mem.read(START + 1, 1).unwrap(), 0x42);
    }
}

#[test]
fn non_code_store_does_not_end_block() {
    for mode in [CpuMode::Real16, CpuMode::Compat32, CpuMode::Long64] {
        let addressing = if mode == CpuMode::Real16 { 0x07 } else { 0x03 };
        let (vm, mut cpu) = setup(mode, &[0xc6, addressing, 0x42, 0xb0, 0x17, 0xf4]);
        cpu.set_reg(Reg::Rbx, 0x9000);
        assert!(!vm.mem.is_code_page(9));
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::Hlt), "{mode:?}");
        assert_eq!(cpu.reg(Reg::Rax), 0x17, "{mode:?}");
        assert_eq!(cpu.retired_instructions(), 3, "{mode:?}");
        assert_eq!(vm.mem.read(0x9000, 1).unwrap(), 0x42);
        assert!(!vm.mem.has_dirty_code());
    }
}

#[test]
fn unrelated_bytes_on_code_page_do_not_end_block() {
    let (vm, mut cpu) = setup(CpuMode::Long64, &[0xc6, 0x03, 0x42, 0xb0, 0x17, 0xf4]);
    cpu.set_reg(Reg::Rbx, START + 0x100);
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rax), 0x17);
    assert_eq!(cpu.retired_instructions(), 3);
    assert_eq!(vm.mem.read(START + 0x100, 1).unwrap(), 0x42);
}

#[test]
fn block_and_store_cross_code_page_boundary() {
    // The word store replaces MOV AL,0 across the 4 KiB boundary with MOV AL,42h.
    let (vm, mut cpu) = setup(CpuMode::Real16, &[]);
    vm.write_bytes(0x7ff9, &[0xc7, 0x06, 0xff, 0x7f, 0xb0, 0x42, 0xb0, 0, 0xf4])
        .unwrap();
    cpu.set_reg(Reg::Rip, 0x7ff9);
    assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
    assert_eq!(cpu.reg(Reg::Rip), 0x7fff);
    assert_eq!(cpu.retired_instructions(), 1);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(cpu.reg(Reg::Rax), 0x42);
    assert_eq!(cpu.retired_instructions(), 3);
}

#[test]
fn entire_writing_instruction_completes() {
    // SDM Vol 2D §6.1, XADD: exchange the old destination into the source
    // register and set flags for the sum, in addition to the memory write.
    for mode in [CpuMode::Real16, CpuMode::Compat32, CpuMode::Long64] {
        let addressing = if mode == CpuMode::Real16 { 0x07 } else { 0x03 };
        // xadd byte [bx/ebx/rbx],al; mov dl,al; mov cl,1; hlt
        let (vm, mut cpu) = setup(mode, &[0x0f, 0xc0, addressing, 0x88, 0xc2, 0xb1, 1, 0xf4]);
        cpu.set_reg(Reg::Rbx, START + 6);
        cpu.set_reg(Reg::Rax, 0x41);
        let mut flags = cpu.flags();
        flags.zf = true;
        cpu.set_flags(flags);
        assert!(matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted));
        assert_eq!(cpu.reg(Reg::Rip), START + 3, "{mode:?}");
        assert_eq!(cpu.reg(Reg::Rax), 1, "{mode:?}");
        assert!(!cpu.flags().zf, "{mode:?}");
        assert_eq!(cpu.retired_instructions(), 1);
        assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
        assert_eq!(cpu.reg(Reg::Rdx), 1, "{mode:?}");
        assert_eq!(cpu.reg(Reg::Rcx), 0x42, "{mode:?}");
        assert_eq!(cpu.retired_instructions(), 4);
    }
}

#[test]
fn another_vcpu_clears_tag_before_running_block_patches_itself() {
    use std::io::{self, Write};
    use std::sync::{mpsc, Arc};
    use std::time::Duration;
    use x86jit_core::lockstep::{Trace, TraceOptions};

    struct PauseAfterStore {
        entered: mpsc::Sender<()>,
        resume: mpsc::Receiver<()>,
    }
    impl Write for PauseAfterStore {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.entered.send(()).unwrap();
            self.resume.recv_timeout(Duration::from_secs(10)).unwrap();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    // SDM Vol 3A §14.6: serialization resynchronizes modified instruction bytes.
    // mov byte [rbx],17h; mov byte [rdi],42h; cpuid; mov al,0; hlt
    let (mut vm, mut a) = setup(
        CpuMode::Long64,
        &[
            0xc6, 0x03, 0x17, 0xc6, 0x07, 0x42, 0x0f, 0xa2, 0xb0, 0, 0xf4,
        ],
    );
    a.set_reg(Reg::Rbx, START + 0x100);
    a.set_reg(Reg::Rdi, START + 9);
    vm.write_bytes(0x9000, &[0xf4]).unwrap();
    let mut b = vm.new_vcpu();
    b.set_reg(Reg::Rip, 0x9000);
    let (entered_tx, entered_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    vm.set_lockstep_trace(Some(Arc::new(Trace::new(
        PauseAfterStore {
            entered: entered_tx,
            resume: resume_rx,
        },
        TraceOptions {
            window: Some((START, START + 11)),
            max_records: Some(1),
        },
    ))));

    std::thread::scope(|scope| {
        let running = scope.spawn(|| {
            let exit = a.run(&vm, None);
            (exit, a)
        });
        // The trace flush is after the second instruction's boundary check,
        // before its store. A still owns its cloned IR throughout B's dispatch.
        entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let tagged_before = vm.mem.is_code_page(START >> 12);
        let dirty_before = vm.mem.has_dirty_code();
        let b_exit = b.run(&vm, None);
        let tagged_after = vm.mem.is_code_page(START >> 12);
        resume_tx.send(()).unwrap();
        let (exit, a) = running.join().unwrap();
        assert!(tagged_before && dirty_before);
        assert!(matches!(b_exit, Exit::Hlt));
        assert!(!tagged_after, "B must clear the running block's code tag");
        assert!(matches!(exit, Exit::Hlt));
        assert_eq!(vm.mem.read(START + 9, 1).unwrap(), 0x42);
        assert_eq!(a.reg(Reg::Rax) & 0xff, 0x42);
        assert_eq!(a.retired_instructions(), 5);
    });
}

#[cfg(unix)]
#[test]
fn serialized_store_through_declared_alias_refetches_running_block() {
    use std::os::fd::{AsRawFd, FromRawFd};

    let mut name = std::env::temp_dir()
        .join("x86jit-same-block-smc-XXXXXX")
        .as_os_str()
        .as_encoded_bytes()
        .to_vec();
    name.push(0);
    // SAFETY: mkstemp receives a writable NUL-terminated template. File owns
    // the resulting descriptor; unlink removes only this fresh temporary name.
    let file = unsafe {
        let fd = libc::mkstemp(name.as_mut_ptr().cast());
        assert!(fd >= 0);
        assert_eq!(libc::unlink(name.as_ptr().cast()), 0);
        std::fs::File::from_raw_fd(fd)
    };
    file.set_len(0x4000).unwrap();
    // SAFETY: MAP_FIXED replaces only pages within this fresh owned reservation.
    let ptr = unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            0x10000,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        assert_ne!(ptr, libc::MAP_FAILED);
        for offset in [0x4000, 0xc000] {
            let view = ptr.cast::<u8>().add(offset).cast();
            assert_eq!(
                libc::mmap(
                    view,
                    0x4000,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    file.as_raw_fd(),
                    0,
                ),
                view,
            );
        }
        ptr.cast::<u8>()
    };
    let ram = x86jit_core::HostRam {
        ptr,
        len: 0x10000,
        guest_base: 0,
        dtor: Box::new(|ptr, len| {
            // SAFETY: this is the reservation, including both shared views.
            assert_eq!(unsafe { libc::munmap(ptr.cast(), len) }, 0);
        }),
        protect: None,
    };
    let mut vm = Vm::with_backend_host_ram(
        VmConfig::reserved(0x10000),
        Box::new(InterpreterBackend),
        ram,
    )
    .unwrap();
    vm.set_tier_up_after(None);
    vm.map(0x4000, 0x4000, Prot::RX, RegionKind::Ram).unwrap();
    vm.map(0xc000, 0x4000, Prot::RW, RegionKind::Ram).unwrap();
    vm.mem.declare_code_alias(0x4000, 0xc000, 0x4000).unwrap();
    // SDM Vol 3A §14.6: CPUID resynchronizes code fetched through another
    // linear address. mov byte [rdi],42h; cpuid; target: mov al,0; hlt
    vm.write_bytes(0x4000, &[0xc6, 0x07, 0x42, 0x0f, 0xa2, 0xb0, 0, 0xf4])
        .unwrap();
    // Alias expansion is reached through the store page's shared code tag.
    vm.mem.mark_code(0x4000, 8);
    assert!(vm.mem.is_code_page(0xc));
    vm.mem.clear_code_page(4);
    assert!(!vm.mem.is_code_page(0xc));
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, 0x4000);
    cpu.set_reg(Reg::Rdi, 0xc006);
    assert!(matches!(cpu.run(&vm, None), Exit::Hlt));
    assert_eq!(vm.mem.read(0x4006, 1).unwrap(), 0x42);
    assert_eq!(cpu.reg(Reg::Rax) & 0xff, 0x42);
    assert_eq!(cpu.retired_instructions(), 4);
}

#[test]
fn helper_stores_cut_before_later_decoded_instructions() {
    use iced_x86::code_asm::*;

    // SDM Vol 3A §14.6: the store's entry point must not change refetch behavior.
    // Transfers deliberately overwrite decoded successors; a one-block budget
    // must stop before executing any of those old successors.
    for kind in [
        "vector", "masked", "x87", "fxsave", "xsave", "rep", "locked",
    ] {
        let mut a = CodeAssembler::new(64).unwrap();
        match kind {
            "vector" => a.movdqu(xmmword_ptr(rdi), xmm0).unwrap(),
            "masked" => a.vmaskmovps(ymmword_ptr(rdi), ymm1, ymm0).unwrap(),
            "x87" => a.fstp(tbyte_ptr(rdi)).unwrap(),
            "fxsave" => a.fxsave64(ptr(rdi)).unwrap(),
            "xsave" => a.xsave64(ptr(rdi)).unwrap(),
            "rep" => a.rep().stosb().unwrap(),
            "locked" => a.lock().add(byte_ptr(rdi), 1).unwrap(),
            _ => unreachable!(),
        }
        let store_len = a.assemble(START).unwrap().len();
        a.mov(bl, 0x17).unwrap();
        a.hlt().unwrap();
        let (mut vm, _) = setup(CpuMode::Long64, &a.assemble(START).unwrap());
        vm.set_guest_cpu_features(x86jit_core::GuestCpuFeatures::v4());
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, START);
        cpu.set_reg(
            Reg::Rdi,
            if matches!(kind, "rep" | "locked") {
                START + store_len as u64 + 1
            } else {
                START
            },
        );
        cpu.set_reg(Reg::Rcx, 1);
        cpu.set_reg(Reg::Rax, 1);
        cpu.cpu.ymm_hi[1] = u128::MAX;
        cpu.set_xmm(1, u128::MAX);
        assert!(
            matches!(cpu.run(&vm, Some(1)), Exit::BudgetExhausted),
            "{kind}"
        );
        assert_eq!(cpu.reg(Reg::Rip), START + store_len as u64, "{kind}");
        assert_eq!(cpu.reg(Reg::Rbx), 0, "{kind}");
        assert_eq!(cpu.retired_instructions(), 1, "{kind}");
    }
}
