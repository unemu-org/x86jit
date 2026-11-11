//! Run the existing Machine witnesses through both selection APIs.
use std::cell::Cell;
use x86jit_core::{CpuGeneration, CpuModel, LegacyCpuModel, Vm};

thread_local! {
    static USE_CPU_MODEL: Cell<bool> = const { Cell::new(false) };
}

pub fn with_cpu_model(test: impl FnOnce()) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            USE_CPU_MODEL.set(false);
        }
    }
    USE_CPU_MODEL.set(true);
    let _reset = Reset;
    test();
}

pub fn select(vm: &mut Vm, model: LegacyCpuModel) {
    if USE_CPU_MODEL.get() {
        let model = match model.generation {
            CpuGeneration::I286 => CpuModel::I80286,
            CpuGeneration::I386 => CpuModel::I80386,
            _ => panic!("fixture has no catalogue model"),
        };
        vm.set_cpu_model(model);
        assert_eq!(vm.cpu_model(), Some(model));
        // Fault witnesses inspect outward exits through either selection API.
        vm.set_exception_delivery(x86jit_core::ExceptionDelivery::Outward);
    } else {
        // This leg explicitly witnesses the deprecated selector.
        #[allow(deprecated)]
        vm.set_legacy_cpu_model(model);
        assert_eq!(vm.cpu_model(), None);
    }
    assert_eq!(
        vm.execution_profile(),
        x86jit_core::ExecutionProfile::Machine
    );
}
