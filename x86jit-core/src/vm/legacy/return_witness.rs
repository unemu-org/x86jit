//! Return-link witnesses (xj-to9.30): a block that ends in a near RET
//! consults the successor links (`links.rs`) like a direct exit. The link
//! is keyed on the IP the RET popped, never on the CALL that pushed it, so a
//! rewritten return address, a stack switch, a `longjmp`, recursion of any
//! depth, SMC and remapping of the return page and the events between CALL
//! and RET all reach the block the ordinary lookup returns. Each runs with
//! links and with every link refused (`links::OFF`) under the budgets of
//! `witness`.
use super::*;

/// `link_witness::witness`, where `returns` is whether the unsliced linked
/// run follows a link after a near RET (`Links::returns`), not after any
/// exit: the CALL's and the loop's own links follow in every one.
fn witness(
    setup: Setup,
    code: &[u8],
    extra: &[(u64, &[u8])],
    steps: u64,
    returns: Option<bool>,
    prepare: impl Fn(&mut Vcpu) + Copy,
) -> Outcome {
    witness_on(setup, code, extra, steps, returns, prepare, |_| {})
}

/// `witness` on a VM that `vm_setup` finished.
fn witness_on(
    setup: Setup,
    code: &[u8],
    extra: &[(u64, &[u8])],
    steps: u64,
    returns: Option<bool>,
    prepare: impl Fn(&mut Vcpu) + Copy,
    vm_setup: fn(&mut Vm),
) -> Outcome {
    let reference = super::witness_on(setup, code, extra, steps, None, prepare, vm_setup);
    if let Some(expected) = returns {
        let linked = run(setup, code, extra, steps, None, true, prepare, vm_setup);
        assert_eq!(
            linked.returns != 0,
            expected,
            "{setup:?}: {} links followed after a near RET",
            linked.returns
        );
    }
    reference
}

/// The 16-bit or the 32-bit form of a witness, by the setup's code size.
fn sized<'a, T>(setup: Setup, narrow: &'a [T], wide: &'a [T]) -> &'a [T] {
    if matches!(setup, Setup::Flat32 | Setup::Paged(_)) {
        wide
    } else {
        narrow
    }
}

#[test]
fn returns_follow_links_on_every_linked_path() {
    // mov cx,100; L: call S; dec cx; jnz L; jmp $; S: inc bx; ret
    let narrow = [
        0xb9, 100, 0, 0xe8, 5, 0, 0x49, 0x75, 0xfa, 0xeb, 0xfe, 0x43, 0xc3,
    ];
    // The same with ECX in 32-bit code.
    let wide = [
        0xb9, 100, 0, 0, 0, 0xe8, 5, 0, 0, 0, 0x49, 0x75, 0xf8, 0xeb, 0xfe, 0x43, 0xc3,
    ];
    for setup in REAL.into_iter().chain(PROTECTED) {
        let code = sized(setup, &narrow, &wide);
        let outcome = witness(setup, code, &[], 100 * 5 + 20, Some(true), |_| {});
        assert_eq!(outcome.gpr[3] & 0xffff, 100, "{setup:?}");
        assert_eq!(outcome.gpr[1] & 0xffff, 0, "{setup:?}");
        // Three transfers an iteration, the RET's included: the CALL's and
        // the JNZ's alone are two.
        let linked = run(setup, code, &[], 100 * 5 + 20, None, true, |_| {}, |_| {});
        assert!(linked.followed > 250, "{setup:?}: {}", linked.followed);
    }
}

#[test]
fn a_rewritten_return_address_reaches_the_block_it_names() {
    // S adds R2 - R1 to its return address on odd BX: the RET alternates
    // between R1 and R2, each linked under its own IP.
    //   mov cx,64; L: call S; R1: dec cx; jnz L; jmp $;
    //   R2: inc si; dec cx; jnz L; jmp $;
    //   S: inc bx; test bl,1; jz T; mov bp,sp; add word [bp],5; T: ret
    let narrow = [
        0xb9, 64, 0, 0xe8, 0x0b, 0, 0x49, 0x75, 0xfa, 0xeb, 0xfe, 0x46, 0x49, 0x75, 0xf4, 0xeb,
        0xfe, 0x43, 0xf6, 0xc3, 1, 0x74, 0x06, 0x89, 0xe5, 0x83, 0x46, 0, 5, 0xc3,
    ];
    // The same in 32-bit code, with add dword [esp],5.
    let wide = [
        0xb9, 64, 0, 0, 0, 0xe8, 0x0b, 0, 0, 0, 0x49, 0x75, 0xf8, 0xeb, 0xfe, 0x46, 0x49, 0x75,
        0xf2, 0xeb, 0xfe, 0x43, 0xf6, 0xc3, 1, 0x74, 0x04, 0x83, 0x04, 0x24, 5, 0xc3,
    ];
    for setup in REAL.into_iter().chain(PROTECTED) {
        let code = sized(setup, &narrow, &wide);
        let outcome = witness(setup, code, &[], 64 * 10, Some(true), |_| {});
        assert_eq!(outcome.gpr[3] & 0xffff, 64, "{setup:?}");
        assert_eq!(outcome.gpr[6] & 0xffff, 32, "{setup:?}");
        assert_eq!(outcome.gpr[1] & 0xffff, 0, "{setup:?}");
    }
}

#[test]
fn a_return_through_a_switched_stack_reaches_the_address_it_pops() {
    // On odd CL, S loads SS with a second stack segment whose word at the
    // same SP holds R2, planted before the run: the RET pops R2 there, and R2
    // restores the first stack.
    //   mov cx,32; L: call S; R1: inc bx; N: dec cx; jnz L; jmp $;
    //   R2: mov ax,K0; mov ss,ax; inc si; jmp N;
    //   S: test cl,1; jz T; mov ax,K1; mov ss,ax; T: ret
    let code = |k0: u16, k1: u16| {
        let [a, b] = k0.to_le_bytes();
        let [c, d] = k1.to_le_bytes();
        vec![
            0xb9, 32, 0, 0xe8, 0x0e, 0, 0x43, 0x49, 0x75, 0xf9, 0xeb, 0xfe, 0xb8, a, b, 0x8e, 0xd0,
            0x46, 0xeb, 0xf3, 0xf6, 0xc1, 1, 0x74, 0x05, 0xb8, c, d, 0x8e, 0xd0, 0xc3,
        ]
    };
    let r2: &[u8] = &0x100cu16.to_le_bytes();
    for setup in REAL.into_iter().chain([
        Setup::Segmented(CpuModel::I80386),
        Setup::Segmented(CpuModel::I80286),
    ]) {
        // Real mode: paragraph 500h (base 5000h). Protected: selector 18h
        // (base 10h), and 10h (base 0) restores the first.
        let (code, plant) = if matches!(setup, Setup::Real(_)) {
            (code(0, 0x500), 0x5000 + 0x6ffe)
        } else {
            (code(0x10, 0x18), 0x10 + 0x6ffe)
        };
        let outcome = witness(
            setup,
            &code,
            &[(plant, r2)],
            32 * 10 + 20,
            Some(true),
            |_| {},
        );
        assert_eq!(outcome.gpr[3] & 0xffff, 16, "{setup:?}");
        assert_eq!(outcome.gpr[6] & 0xffff, 16, "{setup:?}");
        assert_eq!(outcome.gpr[4] & 0xffff, 0x7000, "{setup:?}");
    }
}

#[test]
fn a_longjmp_past_pending_returns_keeps_every_return_target() {
    // On odd CL, B restores the SP saved before the call of A and jumps past
    // both returns; on even CL both return normally.
    //   mov cx,32; L: mov di,sp; call A; R: inc bx; J: dec cx; jnz L; jmp $;
    //   A: call B; inc si; ret; B: test cl,1; jz T; mov sp,di; jmp J; T: ret
    let code = [
        0xb9, 32, 0, 0x89, 0xe7, 0xe8, 0x06, 0, 0x43, 0x49, 0x75, 0xf7, 0xeb, 0xfe, 0xe8, 0x02, 0,
        0x46, 0xc3, 0xf6, 0xc1, 1, 0x74, 0x04, 0x89, 0xfc, 0xeb, 0xed, 0xc3,
    ];
    for setup in REAL.into_iter().chain([
        Setup::Segmented(CpuModel::I80386),
        Setup::Segmented(CpuModel::I80286),
    ]) {
        let outcome = witness(setup, &code, &[], 32 * 10 + 20, Some(true), |_| {});
        assert_eq!(outcome.gpr[3] & 0xffff, 16, "{setup:?}");
        assert_eq!(outcome.gpr[6] & 0xffff, 16, "{setup:?}");
        assert_eq!(outcome.gpr[4] & 0xffff, 0x7000, "{setup:?}");
    }
}

#[test]
fn deep_recursion_returns_through_every_frame() {
    // Forty nested calls of S, each returning to B, then to the loop.
    //   mov cx,8; L: mov dx,40; call S; dec cx; jnz L; jmp $;
    //   S: dec dx; jz B; call S; B: inc bx; ret
    let narrow = [
        0xb9, 8, 0, 0xba, 40, 0, 0xe8, 5, 0, 0x49, 0x75, 0xf7, 0xeb, 0xfe, 0x4a, 0x74, 0x03, 0xe8,
        0xfa, 0xff, 0x43, 0xc3,
    ];
    // The same with ECX and EDX in 32-bit code.
    let wide = [
        0xb9, 8, 0, 0, 0, 0xba, 40, 0, 0, 0, 0xe8, 5, 0, 0, 0, 0x49, 0x75, 0xf3, 0xeb, 0xfe, 0x4a,
        0x74, 0x05, 0xe8, 0xf8, 0xff, 0xff, 0xff, 0x43, 0xc3,
    ];
    for setup in REAL.into_iter().chain(PROTECTED) {
        let code = sized(setup, &narrow, &wide);
        let outcome = witness(setup, code, &[], 8 * 210, Some(true), |_| {});
        assert_eq!(outcome.gpr[3] & 0xffff, 8 * 40, "{setup:?}");
        assert_eq!(outcome.gpr[4] & 0xffff, 0x7000, "{setup:?}");
    }
}

#[test]
fn ret_imm16_returns_follow_links() {
    // mov cx,32; L: push cx; push cx; call S; dec cx; jnz L; jmp $;
    // S: mov bp,sp; add bx,[bp+2]; ret 4
    let narrow = [
        0xb9, 32, 0, 0x51, 0x51, 0xe8, 5, 0, 0x49, 0x75, 0xf8, 0xeb, 0xfe, 0x89, 0xe5, 0x03, 0x5e,
        2, 0xc2, 4, 0,
    ];
    // The same in 32-bit code: add ebx,[esp+4]; ret 8.
    let wide = [
        0xb9, 32, 0, 0, 0, 0x51, 0x51, 0xe8, 5, 0, 0, 0, 0x49, 0x75, 0xf6, 0xeb, 0xfe, 0x03, 0x5c,
        0x24, 4, 0xc2, 8, 0,
    ];
    for setup in REAL.into_iter().chain(PROTECTED) {
        let code = sized(setup, &narrow, &wide);
        let outcome = witness(setup, code, &[], 32 * 8 + 20, Some(true), |_| {});
        assert_eq!(outcome.gpr[3] & 0xffff, 32 * 33 / 2, "{setup:?}");
        assert_eq!(outcome.gpr[4] & 0xffff, 0x7000, "{setup:?}");
    }
}

#[test]
fn near_returns_to_the_same_ip_under_two_code_segments_follow_each_segment_s_link() {
    // Two copies of one loop at 0100:0000 and 0200:0000 that differ only in
    // Q (INC BX / INC SI), entered by a near RET at the same IP in both; a
    // far RET (not linked) moves to the other copy. Only the physical fetch
    // address tells the return links apart (`Link::physical`).
    //   dec cx; jz done; call N; Q: inc bx | inc si; mov ax,other; push ax;
    //   xor ax,ax; push ax; retf; N: ret; (nops); done: jmp $
    let segment = |q: u8, other: u8| {
        let mut code = vec![
            0x49, 0x74, 0x1d, 0xe8, 0x09, 0, q, 0xb8, 0, other, 0x50, 0x31, 0xc0, 0x50, 0xcb, 0xc3,
        ];
        code.resize(0x20, 0x90);
        code.extend([0xeb, 0xfe]);
        code
    };
    let a = segment(0x43, 0x02);
    let b = segment(0x46, 0x01);
    for setup in REAL {
        let outcome = witness(
            setup,
            &a,
            &[(0x2000, &b)],
            40 * 10 + 20,
            Some(true),
            |cpu| {
                cpu.set_reg(Reg::Cs, 0x100);
                // The real-mode CS cache the 80286 and 80386 fetch through.
                cpu.cpu.legacy.segments[SegmentId::Cs as usize].base = 0x1000;
                cpu.set_reg(Reg::Rip, 0);
                cpu.set_reg(Reg::Rcx, 40);
            },
        );
        assert_eq!(outcome.gpr[3] & 0xffff, 20, "{setup:?}");
        assert_eq!(outcome.gpr[6] & 0xffff, 19, "{setup:?}");
    }
}

#[test]
fn a_store_into_the_return_block_before_the_ret_runs_its_new_bytes() {
    // At CX = 8, S rewrites R's INC BX to INC AX before it returns: in the
    // RET's own block (NOP NOP), or in a block before it (JMP to the next
    // instruction ends the block, so the RET's turn drains no store).
    //   mov cx,16; L: call S; R: inc bx; loop L; jmp $;
    //   S: cmp cx,8; jne T; mov byte [R],40h; nop; nop | jmp T; T: ret
    let narrow = |tail: [u8; 2]| {
        vec![
            0xb9, 16, 0, 0xe8, 5, 0, 0x43, 0xe2, 0xfa, 0xeb, 0xfe, 0x83, 0xf9, 8, 0x75, 7, 0xc6,
            0x06, 0x06, 0x10, 0x40, tail[0], tail[1], 0xc3,
        ]
    };
    let wide = |tail: [u8; 2]| {
        vec![
            0xb9, 16, 0, 0, 0, 0xe8, 5, 0, 0, 0, 0x43, 0xe2, 0xf8, 0xeb, 0xfe, 0x83, 0xf9, 8, 0x75,
            9, 0xc6, 0x05, 0x0a, 0x10, 0, 0, 0x40, tail[0], tail[1], 0xc3,
        ]
    };
    for setup in REAL.into_iter().chain(PROTECTED) {
        for tail in [[0x90, 0x90], [0xeb, 0x00]] {
            let code = if matches!(setup, Setup::Flat32 | Setup::Paged(_)) {
                wide(tail)
            } else {
                narrow(tail)
            };
            let outcome = witness(setup, &code, &[], 16 * 8 + 20, Some(true), |_| {});
            assert_eq!(outcome.gpr[3] & 0xffff, 8, "{setup:?} {tail:?}");
            assert_eq!(outcome.gpr[0] & 0xffff, 8, "{setup:?} {tail:?}");
        }
    }
}

#[test]
fn interrupts_and_single_steps_between_call_and_ret_keep_their_boundaries() {
    // INT 20h at CX = 32 inside S, then STI, which opens the window for the
    // IRQ 21h queued before the run, before S returns.
    //   mov cx,64; L: call S; R: inc bx; loop L; jmp $;
    //   S: cmp cx,32; jne T; int 20h; sti; T: ret
    let interrupts = [
        0xb9, 64, 0, 0xe8, 5, 0, 0x43, 0xe2, 0xfa, 0xeb, 0xfe, 0x83, 0xf9, 0x20, 0x75, 0x03, 0xcd,
        0x20, 0xfb, 0xc3,
    ];
    // At CX = 8, S sets TF before it returns: every later instruction traps.
    //   S: cmp cx,8; jne T; pushf; pop ax; or ah,1; push ax; popf; T: ret
    let single_step = [
        0xb9, 32, 0, 0xe8, 5, 0, 0x43, 0xe2, 0xfa, 0xeb, 0xfe, 0x83, 0xf9, 0x08, 0x75, 0x07, 0x9c,
        0x58, 0x80, 0xcc, 0x01, 0x50, 0x9d, 0xc3,
    ];
    let ivt: &[u8] = &[0x20, 0x20, 0, 0];
    let vectors: &[u8] = &[0x00, 0x20, 0, 0, 0x10, 0x20, 0, 0];
    let handlers: &[(u64, &[u8])] = &[
        (4, ivt),
        (0x80, vectors),
        (0x2000, &[0x42, 0xcf]),
        (0x2010, &[0x46, 0xcf]),
        (0x2020, &[0x47, 0xcf]),
    ];
    for setup in REAL {
        let outcome = witness(setup, &interrupts, handlers, 64 * 7, Some(true), |cpu| {
            cpu.inject_irq(0x21)
        });
        assert_eq!(outcome.gpr[3] & 0xffff, 64, "{setup:?}");
        assert_eq!(outcome.gpr[2] & 0xffff, 1, "{setup:?}");
        assert_eq!(outcome.gpr[6] & 0xffff, 1, "{setup:?}");
        assert!(outcome.log.contains("Irq"), "{setup:?} {}", outcome.log);
        let outcome = witness(setup, &single_step, handlers, 400, Some(true), |_| {});
        assert!(outcome.log.contains("Trap"), "{setup:?} {}", outcome.log);
    }
}

#[test]
fn a_remapped_return_page_runs_its_new_frame() {
    // L jumps to C on linear page 2 (frame 5), which calls S on page 1 and
    // returns to INC EBX there. At ECX = 8, S points page 2 at frame 6 (INC
    // EAX at the same offset) and flushes the translation, a CR3 reload on
    // the I80386 and INVLPG on the I80486 (SDM Vol 3A §4.10.4.1), before it
    // returns: that RET and every later one run frame 6.
    //   mov ecx,16; L: jmp 2000h; R: loop L; jmp $;
    //   S: cmp ecx,8; jne T; mov dword [9008h],6007h; <flush>; T: ret
    //   2000h: call S; inc ebx | inc eax; jmp R
    let mut code = vec![
        0xb9, 16, 0, 0, 0, 0xe9, 0xf6, 0x0f, 0, 0, 0xe2, 0xf9, 0xeb, 0xfe, 0x83, 0xf9, 0x08, 0x75,
        0x11,
    ];
    code.extend([0xc7, 0x05, 0x08, 0x90, 0, 0, 0x07, 0x60, 0, 0]);
    let flush_offset = code.len();
    code.extend([0; 7]);
    code.push(0xc3);
    let frame5: &[u8] = &[
        0xe8, 0x09, 0xf0, 0xff, 0xff, 0x43, 0xe9, 0xff, 0xef, 0xff, 0xff,
    ];
    let frame6: &[u8] = &[
        0xe8, 0x09, 0xf0, 0xff, 0xff, 0x40, 0xe9, 0xff, 0xef, 0xff, 0xff,
    ];
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
        let outcome = witness(setup, &code, &extra, 16 * 9 + 20, Some(true), |_| {});
        assert_eq!(outcome.gpr[3], 8, "{model:?}");
        assert_eq!(outcome.gpr[0], 8, "{model:?}");
    }
}

#[test]
fn only_near_returns_end_in_a_return_link_exit() {
    // 16-bit forms: ret, ret 4, retf, retf 4, iret, jmp ax, jmp [bx],
    // jmp [bp+0], call ax, call [bx], call [bp+0], pop ax.
    let narrow: [(&[u8], bool); 12] = [
        (&[0xc3], true),
        (&[0xc2, 4, 0], true),
        (&[0xcb], false),
        (&[0xca, 4, 0], false),
        (&[0xcf], false),
        (&[0xff, 0xe0], false),
        (&[0xff, 0x27], false),
        (&[0xff, 0x66, 0], false),
        (&[0xff, 0xd0], false),
        (&[0xff, 0x17], false),
        (&[0xff, 0x56, 0], false),
        (&[0x58], false),
    ];
    // 32-bit forms: ret, ret 8, retf, jmp eax, jmp [ebx], call eax,
    // call [ebx], call [esp].
    let wide: [(&[u8], bool); 8] = [
        (&[0xc3], true),
        (&[0xc2, 8, 0], true),
        (&[0xcb], false),
        (&[0xff, 0xe0], false),
        (&[0xff, 0x23], false),
        (&[0xff, 0xd0], false),
        (&[0xff, 0x13], false),
        (&[0xff, 0x14, 0x24], false),
    ];
    for setup in [
        Setup::Real(CpuModel::I80286),
        Setup::Real(CpuModel::I80386),
        Setup::Segmented(CpuModel::I80286),
        Setup::Segmented(CpuModel::I80386),
        Setup::Flat32,
        Setup::Paged(CpuModel::I80386),
    ] {
        let (vm, cpu) = machine(setup, &[], &[]);
        let model = match setup {
            Setup::Real(model) | Setup::Segmented(model) | Setup::Paged(model) => model,
            Setup::Flat32 => CpuModel::I80386,
        };
        let context = derive_legacy_context(model.legacy_model(), &cpu.cpu).unwrap();
        let profile = vm.machine_model_profile(context.generation);
        let forms: &[(&[u8], bool)] = sized(setup, &narrow, &wide);
        for &(bytes, near) in forms {
            let ir =
                crate::lift::legacy::instruction_for_profile(bytes, CODE as u32, context, profile)
                    .unwrap();
            assert_eq!(links::ends_near_return(&ir), near, "{setup:?} {bytes:02x?}");
            assert!(!ir.ends_direct(), "{setup:?} {bytes:02x?}");
        }
    }
}

#[test]
fn a_return_block_rewritten_inside_a_resolver_runs_its_new_bytes() {
    // R, the return site at REWRITTEN, is linked from S's RET. At CX = 20 S
    // stores the marker and jumps to C, whose first lift rewrites R's INC
    // BX to INC AX (`rewrite_after_marker`); C then jumps to the RET, whose
    // turn drains no store. Only the publication generation tells R's link
    // that its block left the cache (`Link::generation`).
    //   L: call S; R: inc bx; loop L; jmp $; S: cmp cx,20; jne T;
    //   mov byte [6000h],1; jmp C; T: ret; (nops); C: nop; jmp T
    let mut code = vec![
        0xe8, 5, 0, 0x43, 0xe2, 0xfa, 0xeb, 0xfe, 0x83, 0xf9, 20, 0x75, 7, 0xc6, 0x06, 0x00, 0x60,
        0x01, 0xeb, 0x0c, 0xc3,
    ];
    assert_eq!(CODE + 3, REWRITTEN);
    code.resize(0x20, 0x90);
    code.extend([0x90, 0xeb, 0xf1]);
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
            40 * 6 + 40,
            Some(true),
            |cpu| cpu.set_reg(Reg::Rcx, 40),
            |vm| {
                *vm.source_read_hook.lock().unwrap() = Some(rewrite_after_marker);
            },
        );
        assert_eq!(outcome.gpr[3] & 0xffff, 20, "{setup:?}");
        assert_eq!(outcome.gpr[0] & 0xffff, 20, "{setup:?}");
    }
}

/// Virtual-8086 mode at IOPL 3 and CPL 3, unpaged or under the setup's
/// identity paging, with the real-mode setup's CS:IP and SS:SP.
fn v86(paged: bool) -> impl Fn(&mut Vcpu) + Copy {
    move |cpu: &mut Vcpu| {
        cpu.cpu = CpuState::legacy_real_entry([0; 6]);
        cpu.cpu.rip = CODE;
        cpu.cpu.gpr[4] = 0x7000;
        cpu.cpu.legacy.cr0 = if paged { 0x8000_0001 } else { 1 };
        cpu.cpu.legacy.cr3 = 0x8000;
        cpu.cpu.legacy.cpl = 3;
        cpu.cpu.rflags_system = (1 << 17) | 0x3000;
    }
}

#[test]
fn virtual_8086_returns_follow_links() {
    // The 16-bit loop of `returns_follow_links_on_every_linked_path`, in
    // V86 mode. Paged V86 links like any paged context; unpaged V86 takes
    // no successor link at all (`link_path`), RET's included.
    //   mov cx,100; L: call S; dec cx; jnz L; jmp $; S: inc bx; ret
    let code = [
        0xb9, 100, 0, 0xe8, 5, 0, 0x49, 0x75, 0xfa, 0xeb, 0xfe, 0x43, 0xc3,
    ];
    for (setup, paged) in [
        (Setup::Real(CpuModel::I80386), false),
        (Setup::Paged(CpuModel::I80386), true),
        (Setup::Paged(CpuModel::I80486Dx), true),
    ] {
        let outcome = witness(setup, &code, &[], 100 * 5 + 20, Some(paged), v86(paged));
        // VM and IOPL 3 (SDM Vol 3A §2.3): the run stayed in V86 mode.
        assert!(outcome.state.contains("rflags_system: 143360"), "{setup:?}");
        assert_eq!(outcome.gpr[3] & 0xffff, 100, "{setup:?}");
        assert_eq!(outcome.gpr[4] & 0xffff, 0x7000, "{setup:?}");
    }
}

#[test]
fn operand_size_prefixed_returns_follow_links() {
    // 16-bit code, a 32-bit RET (SDM Vol 2B §4.3 RET: the operand size
    // selects the width popped): mov cx,32; L: push dword R; o32 ret;
    // R: inc bx; loop L; jmp $, and with push 0; ...; o32 ret 2.
    let narrow = [
        0xb9, 32, 0, 0x66, 0x68, 0x0b, 0x10, 0, 0, 0x66, 0xc3, 0x43, 0xe2, 0xf5, 0xeb, 0xfe,
    ];
    let narrow_imm = [
        0xb9, 32, 0, 0x6a, 0, 0x66, 0x68, 0x0f, 0x10, 0, 0, 0x66, 0xc2, 2, 0, 0x43, 0xe2, 0xf1,
        0xeb, 0xfe,
    ];
    // 32-bit code, a 16-bit RET: mov ecx,32; L: push word R; o16 ret; R:
    // inc ebx; loop L; jmp $, and with o16 push 0; ...; o16 ret 2.
    let wide = [
        0xb9, 32, 0, 0, 0, 0x66, 0x68, 0x0b, 0x10, 0x66, 0xc3, 0x43, 0xe2, 0xf7, 0xeb, 0xfe,
    ];
    let wide_imm = [
        0xb9, 32, 0, 0, 0, 0x66, 0x6a, 0, 0x66, 0x68, 0x10, 0x10, 0x66, 0xc2, 2, 0, 0x43, 0xe2,
        0xf2, 0xeb, 0xfe,
    ];
    // The 80286 has no operand-size prefix.
    for setup in [
        Setup::Real(CpuModel::I80386),
        Setup::Segmented(CpuModel::I80386),
        Setup::Flat32,
        Setup::Paged(CpuModel::I80386),
    ] {
        for code in [
            sized(setup, &narrow[..], &wide[..]),
            sized(setup, &narrow_imm, &wide_imm),
        ] {
            let outcome = witness(setup, code, &[], 32 * 6 + 20, Some(true), |_| {});
            assert_eq!(outcome.gpr[3] & 0xffff, 32, "{setup:?} {code:02x?}");
            assert_eq!(outcome.gpr[4] & 0xffff, 0x7000, "{setup:?} {code:02x?}");
        }
    }
}

/// A flat 32-bit ring-0 code segment, selector 8 of the setup's GDT.
const CODE_DESCRIPTOR: (u64, &[u8]) = (GDT + 8, &[0xff, 0xff, 0, 0, 0, 0x9b, 0xcf, 0]);

#[test]
fn far_returns_in_protected_code_between_linked_near_returns() {
    // The near RET links; the far RET (to selector 8, a flat code segment)
    // loads CS and does not (`links::ends_near_return`).
    //   mov ecx,32; L: call S; R: inc ebx; push 8; push F; retf; jmp $;
    //   F: inc esi; loop L; jmp $; S: ret
    let code = [
        0xb9, 32, 0, 0, 0, 0xe8, 0x10, 0, 0, 0, 0x43, 0x6a, 0x08, 0x68, 0x15, 0x10, 0, 0, 0xcb,
        0xeb, 0xfe, 0x46, 0xe2, 0xed, 0xeb, 0xfe, 0xc3,
    ];
    for setup in [Setup::Flat32, Setup::Paged(CpuModel::I80386)] {
        let outcome = witness(
            setup,
            &code,
            &[CODE_DESCRIPTOR],
            32 * 8 + 20,
            Some(true),
            |cpu| cpu.cpu.cs = 8,
        );
        assert_eq!(outcome.gpr[3], 32, "{setup:?}");
        assert_eq!(outcome.gpr[6], 32, "{setup:?}");
        assert_eq!(outcome.gpr[4], 0x7000, "{setup:?}");
    }
}

// Vectors 12 (#SS) and 13 (#GP) of the real-mode IVT, and their handler at
// 2000h: inc si; mov sp,7000h; jmp R (1006h, or 1009h for the 32-bit call).
fn fault_handlers(r: u16) -> [u8; 7] {
    let [lo, hi] = r.wrapping_sub(0x2007).to_le_bytes();
    [0x46, 0xbc, 0x00, 0x70, 0xe9, lo, hi]
}

#[test]
fn a_return_that_faults_on_its_stack_read_delivers_the_fault() {
    // At CX = 8, S sets SP to FFFFh: the RET's word read leaves the stack
    // segment and faults (SDM Vol 2B §4.3 RET, real-address mode: #SS);
    // the handler resumes at R.
    //   mov cx,16; L: call S; R: inc bx; loop L; jmp $;
    //   S: cmp cx,8; jne T; mov sp,FFFFh; T: ret
    let code = [
        0xb9, 16, 0, 0xe8, 5, 0, 0x43, 0xe2, 0xfa, 0xeb, 0xfe, 0x83, 0xf9, 8, 0x75, 3, 0xbc, 0xff,
        0xff, 0xc3,
    ];
    let handler = fault_handlers(0x1006);
    let vectors: &[u8] = &[0x00, 0x20, 0, 0, 0x00, 0x20, 0, 0];
    let extra: &[(u64, &[u8])] = &[(0x30, vectors), (0x2000, &handler)];
    for model in [CpuModel::I80286, CpuModel::I80386] {
        let setup = Setup::Real(model);
        let outcome = witness(setup, &code, extra, 16 * 7 + 20, Some(true), |_| {});
        assert_eq!(outcome.gpr[3] & 0xffff, 16, "{model:?}");
        assert_eq!(outcome.gpr[6] & 0xffff, 1, "{model:?}");
        assert!(
            outcome.log.contains("kind: Fault"),
            "{model:?} {}",
            outcome.log
        );
    }
}

#[test]
fn a_return_past_the_code_segment_limit_delivers_the_fault() {
    // A 32-bit CALL and RET in real mode. At CX = 8, S sets bit 16 of its
    // return EIP: the RET's target is past CS's 64 KiB limit (SDM Vol 2B
    // §4.3 RET, real-address mode: #GP); the handler resumes at R.
    //   mov cx,16; L: o32 call S; R: inc bx; loop L; jmp $;
    //   S: cmp cx,8; jne T; mov bp,sp; mov byte [bp+2],1; T: o32 ret
    let code = [
        0xb9, 16, 0, 0x66, 0xe8, 5, 0, 0, 0, 0x43, 0xe2, 0xf7, 0xeb, 0xfe, 0x83, 0xf9, 8, 0x75, 6,
        0x89, 0xe5, 0xc6, 0x46, 2, 1, 0x66, 0xc3,
    ];
    let handler = fault_handlers(0x1009);
    let vectors: &[u8] = &[0x00, 0x20, 0, 0, 0x00, 0x20, 0, 0];
    let extra: &[(u64, &[u8])] = &[(0x30, vectors), (0x2000, &handler)];
    let setup = Setup::Real(CpuModel::I80386);
    let outcome = witness(setup, &code, extra, 16 * 7 + 20, Some(true), |_| {});
    assert_eq!(outcome.gpr[3] & 0xffff, 16);
    assert_eq!(outcome.gpr[6] & 0xffff, 1);
    assert!(outcome.log.contains("kind: Fault"), "{}", outcome.log);
}

#[test]
fn a_return_from_a_not_present_stack_page_delivers_the_page_fault() {
    // At ECX = 8, S sets ESP to 10000h, the first byte of linear page 16,
    // which the setup's page table leaves not present: the RET's read faults
    // (SDM Vol 3A §4.7: #PF) and the fault frame lands on page 15. The #PF
    // handler (an interrupt gate to 8:3000h) resumes at R.
    //   mov ecx,16; L: call S; R: inc ebx; loop L; jmp $;
    //   S: cmp ecx,8; jne T; mov esp,10000h; T: ret
    let code = [
        0xb9, 16, 0, 0, 0, 0xe8, 5, 0, 0, 0, 0x43, 0xe2, 0xf8, 0xeb, 0xfe, 0x83, 0xf9, 8, 0x75, 5,
        0xbc, 0, 0, 1, 0, 0xc3,
    ];
    // inc esi; mov esp,7000h; jmp 100Ah
    let handler: &[u8] = &[0x46, 0xbc, 0, 0x70, 0, 0, 0xe9, 0xff, 0xdf, 0xff, 0xff];
    // Vector 14: a 32-bit interrupt gate, DPL 0, to 8:3000h.
    let gate: &[u8] = &[0x00, 0x30, 0x08, 0x00, 0x00, 0x8e, 0x00, 0x00];
    let extra = [CODE_DESCRIPTOR, (0xb000 + 14 * 8, gate), (0x3000, handler)];
    for model in [CpuModel::I80386, CpuModel::I80486Dx] {
        let setup = Setup::Paged(model);
        let outcome = witness(setup, &code, &extra, 16 * 6 + 40, Some(true), |cpu| {
            cpu.cpu.cs = 8;
            cpu.cpu.legacy.idtr.base = 0xb000;
            cpu.cpu.legacy.idtr.limit = 0x7ff;
        });
        assert_eq!(outcome.gpr[3], 16, "{model:?}");
        assert_eq!(outcome.gpr[6], 1, "{model:?}");
        assert_eq!(outcome.gpr[4], 0x7000, "{model:?}");
    }
}
