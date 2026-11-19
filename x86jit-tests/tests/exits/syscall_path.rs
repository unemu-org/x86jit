//! Non-fault exit identity (§5.2). These tests exercise the engine's syscall
//! interception contract, including Compat32 forms already accepted by the lift.

use x86jit_core::{
    Backend, BlockKey, CachedBlock, CpuMode, Exit, FaultPath, InterpreterBackend, Prot, Reg,
    RegionCaps, RegionKind, Vm, VmConfig,
};
use x86jit_cranelift::JitBackend;

const ENTRY: u64 = 0x1000;
const CAPS: RegionCaps = RegionCaps {
    max_blocks: 8,
    max_icount: 64,
};

#[derive(Clone, Copy, Debug)]
enum Tier {
    Interpreter,
    Block,
    Region,
}

fn vm(tier: Tier, mode: CpuMode, code: &[u8]) -> Vm {
    let backend: Box<dyn Backend> = match tier {
        Tier::Interpreter => Box::new(InterpreterBackend),
        Tier::Block => Box::new(JitBackend::new()),
        Tier::Region => Box::new(JitBackend::with_superblocks(CAPS)),
    };
    let mut vm = Vm::with_backend(VmConfig::flat(0x3000), backend);
    vm.set_cpu_mode(mode);
    vm.map(ENTRY, 0x1000, Prot::RX, RegionKind::Ram).unwrap();
    vm.write_bytes(ENTRY, code).unwrap();
    vm.write_bytes(ENTRY + 0x100, &[0xf4]).unwrap();
    vm
}

fn syscall_loop(tier: Tier, mode: CpuMode, gate: &[u8]) {
    // mov eax,39; jmp next; next: nop; <gate>; jmp ENTRY.
    // SDM Vol 2B §4.3, SYSCALL: 0F 05; Vol 2A §3.2, INT n: CD ib.
    // A prefix-length variant guards against deriving the RIP by subtracting two.
    let mut code = vec![0xb8, 39, 0, 0, 0, 0xeb, 0, 0x90];
    let instruction_rip = ENTRY + code.len() as u64;
    code.extend_from_slice(gate);
    let resume_rip = ENTRY + code.len() as u64;
    let displacement = -(code.len() as i8 + 2);
    code.extend_from_slice(&[0xeb, displacement as u8]);
    let mut vm = vm(tier, mode, &code);
    if matches!(tier, Tier::Region) {
        // Force the selected tier independently of automatic warmup decisions.
        vm.set_tier_up_after(Some(u32::MAX));
        vm.set_tier_up_background(true);
        vm.test_support_force_region(ENTRY, CAPS, 0x1f80).unwrap();
    }
    let mut cpu = vm.new_vcpu();
    cpu.set_reg(Reg::Rip, ENTRY);
    assert!(cpu.syscall_evidence().is_none());
    let mut saved = None;
    for _ in 0..4 {
        assert!(matches!(cpu.run(&vm, Some(32)), Exit::Syscall));
        assert_eq!(cpu.reg(Reg::Rax), 39);
        assert_eq!(cpu.cpu.rip, resume_rip);
        assert!(cpu.interruption_evidence().is_none());
        let evidence = cpu.syscall_evidence().expect("syscall execution evidence");
        assert_eq!(evidence.rip(), instruction_rip);
        let compiled_entry = match tier {
            Tier::Interpreter => {
                assert_eq!(evidence.path(), FaultPath::Interpreter);
                None
            }
            Tier::Block => match evidence.path() {
                FaultPath::CompiledBlock { host_pc, site } => {
                    assert_eq!(site, x86jit_core::codemap::lookup_trap(host_pc));
                    Some((ENTRY + 7, host_pc))
                }
                path => panic!("expected compiled block, got {path:?}"),
            },
            Tier::Region => match evidence.path() {
                FaultPath::Region {
                    entry_rip,
                    host_pc,
                    site,
                } => {
                    assert_eq!(entry_rip, ENTRY);
                    assert_eq!(site, x86jit_core::codemap::lookup_trap(host_pc));
                    Some((ENTRY, host_pc))
                }
                path => panic!("expected forced region, got {path:?}"),
            },
        };
        if let Some((entry_rip, host_pc)) = compiled_entry {
            assert!(x86jit_core::codemap::contains(host_pc));
            let block = vm
                .cache
                .get(BlockKey::with_mxcsr(entry_rip, mode, cpu.cpu.mxcsr))
                .unwrap();
            let CachedBlock::Compiled { entry } = block else {
                panic!("evidence must identify a compiled cache entry");
            };
            assert_eq!(host_pc, entry.0 as usize);
        }
        if let Some(previous) = saved {
            assert_eq!(previous, evidence);
        }
        saved = Some(evidence);
        cpu.set_reg(Reg::Rax, 1234); // embedder response
    }
    assert!(matches!(cpu.run(&vm, Some(0)), Exit::BudgetExhausted));
    assert!(cpu.syscall_evidence().is_none());
    assert_eq!(saved.unwrap().rip(), instruction_rip);

    // Replacing syscall evidence with interruption evidence must not conflate them.
    assert!(matches!(cpu.run(&vm, Some(32)), Exit::Syscall));
    cpu.exit_request().request_exit();
    assert!(matches!(cpu.run(&vm, None), Exit::Interrupted));
    assert!(cpu.syscall_evidence().is_none());
    assert_eq!(cpu.interruption_evidence().unwrap().rip(), resume_rip);
    assert!(matches!(cpu.run(&vm, Some(32)), Exit::Syscall));
    cpu.set_reg(Reg::Rip, ENTRY + 0x100);
    assert!(matches!(cpu.run(&vm, Some(32)), Exit::Hlt));
    assert!(cpu.syscall_evidence().is_none());
    assert!(cpu.interruption_evidence().is_none());
}

macro_rules! syscall_test {
    ($name:ident, $tier:expr, $mode:expr, $gates:expr) => {
        #[test]
        fn $name() {
            for gate in $gates {
                syscall_loop($tier, $mode, gate);
            }
        }
    };
}

const LONG_GATES: &[&[u8]] = &[&[0x0f, 0x05], &[0x66, 0x0f, 0x05]];
const COMPAT_GATES: &[&[u8]] = &[&[0x0f, 0x05], &[0xcd, 0x80], &[0x66, 0xcd, 0x80]];

syscall_test!(
    syscall_path_interpreter,
    Tier::Interpreter,
    CpuMode::Long64,
    LONG_GATES
);
syscall_test!(syscall_path_block, Tier::Block, CpuMode::Long64, LONG_GATES);
syscall_test!(
    syscall_path_region,
    Tier::Region,
    CpuMode::Long64,
    LONG_GATES
);
syscall_test!(
    syscall_path_compat_interpreter,
    Tier::Interpreter,
    CpuMode::Compat32,
    COMPAT_GATES
);
syscall_test!(
    syscall_path_compat_block,
    Tier::Block,
    CpuMode::Compat32,
    COMPAT_GATES
);
syscall_test!(
    syscall_path_compat_region,
    Tier::Region,
    CpuMode::Compat32,
    COMPAT_GATES
);

#[test]
fn syscall_path_sysenter_remains_unsupported() {
    // SDM Vol 2B §4.3, SYSENTER: 0F 34. This is a coverage assertion,
    // not an assertion about hardware's treatment of the instruction.
    for tier in [Tier::Interpreter, Tier::Block] {
        let vm = vm(tier, CpuMode::Compat32, &[0x0f, 0x34]);
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, ENTRY);
        assert!(matches!(
            cpu.run(&vm, None),
            Exit::UnknownInstruction { .. }
        ));
        assert!(cpu.syscall_evidence().is_none());
    }
}

#[test]
fn syscall_path_clears_on_unwinding_run_entry() {
    for tier in [Tier::Interpreter, Tier::Block, Tier::Region] {
        let mut vm = vm(tier, CpuMode::Long64, &[0x0f, 0x05]);
        if matches!(tier, Tier::Region) {
            vm.test_support_force_region(ENTRY, CAPS, 0x1f80).unwrap();
        }
        let mut cpu = vm.new_vcpu();
        cpu.set_reg(Reg::Rip, ENTRY);
        assert!(matches!(cpu.run(&vm, None), Exit::Syscall));
        let saved = cpu.syscall_evidence().unwrap();
        vm.set_fetch_policy(Box::new(|_| panic!("fetch fixture")));
        cpu.set_reg(Reg::Rip, ENTRY + 0x100);
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cpu.run(&vm, None)));
        assert!(unwind.is_err());
        assert!(cpu.syscall_evidence().is_none());
        assert_eq!(saved.rip(), ENTRY);
    }
}
