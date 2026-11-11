//! Descriptor fixtures shared by the LDT/query and task-register witnesses.
use x86jit_core::fault::ErrorCode;
use x86jit_core::{
    Backend, CachedBlock, CpuModel, ExceptionDelivery, ExecutionProfile, Exit, IrBlock,
    MemConsistency, Prot, RegionKind, StepOptions, StepStop, Vcpu, Vm, VmConfig,
};

pub const MODELS: [CpuModel; 2] = [CpuModel::I80286, CpuModel::I80386];
struct NoBackend;
impl Backend for NoBackend {
    fn materialize(
        &self,
        _: &IrBlock,
        _: MemConsistency,
        _: Option<(u64, u64)>,
        _: u64,
    ) -> CachedBlock {
        panic!("Machine execution reached an application backend")
    }
}
pub fn setup(model: CpuModel, code: &[u8]) -> (Vm, Vcpu) {
    let mut vm = Vm::with_backend(VmConfig::flat(0x240000), Box::new(NoBackend));
    vm.map(0, 0x240000, Prot::RWX, RegionKind::Ram).unwrap();
    vm.set_cpu_model(model);
    vm.set_exception_delivery(ExceptionDelivery::Outward);
    assert_eq!(vm.execution_profile(), ExecutionProfile::Machine);
    vm.write_bytes(0x1000, code).unwrap();
    let mut cpu = vm.new_vcpu();
    cpu.cpu.rip = 0x1000;
    cpu.cpu.gpr[4] = 0x8000;
    cpu.cpu.legacy.cr0 = 1;
    cpu.cpu.legacy.gdtr.base = 0x2000;
    cpu.cpu.legacy.gdtr.limit = 0x7f;
    descriptor(&vm, 0x2008, 0x3000, 0x3f, 0x82);
    (vm, cpu)
}
pub fn descriptor(vm: &Vm, address: u64, base: u32, limit: u32, attr: u16) {
    // SDM Vol 3A §3.4.5 Fig. 3-8; 286 PRM §6.3 Fig. 6-3 p. 6-5;
    // 386 PRM §5.1.1 Fig. 5-3: eight-byte descriptor images.
    let raw = if attr & 0x8000 != 0 {
        limit >> 12
    } else {
        limit
    };
    let low = (raw & 0xffff) | (base << 16);
    let high = ((base >> 16) & 0xff)
        | (base & 0xff00_0000)
        | (u32::from(attr & 0xf0ff) << 8)
        | (raw & 0xf0000);
    vm.write_bytes(address, &low.to_le_bytes()).unwrap();
    vm.write_bytes(address + 4, &high.to_le_bytes()).unwrap();
}
pub fn value(vm: &Vm, address: u64, size: usize) -> u64 {
    let mut bytes = [0; 8];
    vm.read_bytes(address, &mut bytes[..size]).unwrap();
    u64::from_le_bytes(bytes)
}
pub fn steps(cpu: &mut Vcpu, vm: &Vm, count: u64) {
    let run = cpu.run_steps(vm, count, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), count, "{:?}", run.stop());
    assert!(
        matches!(run.stop(), StepStop::Exit(Exit::BudgetExhausted)),
        "{:?}",
        run.stop()
    );
}
pub fn fault(cpu: &mut Vcpu, vm: &Vm, vector: u8, code: Option<u32>) {
    let before = format!("{:?}", cpu.cpu);
    let run = cpu.run_steps(vm, 1, StepOptions::default()).unwrap();
    assert_eq!(run.steps(), 0);
    let StepStop::Exit(exit @ Exit::Exception { vector: actual, .. }) = run.stop() else {
        panic!("{:?}", run.stop());
    };
    assert_eq!(*actual, vector);
    let details = exit.fault_metadata().unwrap();
    assert_eq!(details.saved_rip(), cpu.cpu.rip);
    assert_eq!(
        details.error_code(),
        code.map_or(ErrorCode::NoCode, |c| ErrorCode::bits(c, 0xffff))
    );
    assert_eq!(format!("{:?}", cpu.cpu), before);
}
