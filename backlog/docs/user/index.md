---
id: doc-20
title: 'x86jit — user documentation'
type: guide
created_date: '2026-07-06 11:25'
---

# x86jit — user documentation

Docs for humans embedding `x86jit` as a library. Usage, examples, integration guides. Not for coding agents — agents read [`../../tasks/`](../../tasks/).

`x86jit` is a guest-agnostic x86-64 recompiler engine: you give it a memory map plus an entry point, it executes guest instructions and hands control back through `Exit` whenever it hits something it doesn't handle (syscall, MMIO, unknown instruction). File-format parsing (ELF/PE), OS syscall emulation, and devices live in **your** code, not the core (§1).

## Minimal shape

```rust
use x86jit_core::{Vm, VmConfig, MemoryModel, MemConsistency, Reg, Exit, Prot, RegionKind};

// Default backend is the interpreter. For the JIT:
//   Vm::with_backend(cfg, Box::new(x86jit_cranelift::JitBackend::new(..)))
let mut vm = Vm::new(VmConfig {
    memory_model: MemoryModel::Flat { size: 64 << 20 },
    // Consistency tier (matters on ARM hosts only): Fast → AcqRel → FullTso.
    // Escalate per workload if a multithreaded guest misbehaves (spec §8.2.3).
    consistency: MemConsistency::Fast,
});

// map + load guest bytes (an ELF loader would do this for you — see x86jit-elf)
vm.map(0x1000, 0x1000, Prot::RX, RegionKind::Ram)?;
vm.write_bytes(0x1000, &code)?;

let mut cpu = vm.new_vcpu();
cpu.set_reg(Reg::Rip, 0x1000);

loop {
    match cpu.run(&vm, Some(100_000)) {
        Exit::Syscall => { /* read args via cpu.reg(), set result, continue */ }
        Exit::Hlt => break,
        other => { /* handle mmio / unknown / budget */ break }
    }
}
```

<TBD: expand with a runnable end-to-end example once M2 (static ELF hello-world) lands.>


## In-process Machine devices

Select a `CpuModel`, map MMIO as `RegionKind::Trap`, and register physical MMIO
and/or port ranges with `Vm::register_device_range`. Lend your `DeviceIo`
implementation to `cpu.run_steps_with_devices(&vm, budget, options, Some(&mut
board))`; the same board may borrow a single-threaded device graph for the call.
Return `Handled(value)` to continue, `Fallback` for ordinary Exit/completion, or
`DeviceFault(error)` for explicit host recovery. Make no device-private changes
when declining/faulting. Use the restricted context for tracked RAM writes and
board signals. INTR is a level: on `Exit::Device(DeviceStop::IntrReady)`, let
your interrupt controller
choose a vector and use `inject_irq`; deassert with `set_intr(false)` when the
controller lowers the line. Set an exact step/retirement deadline when a device
needs service before the run budget ends. Drain the normal delivery log at your
chosen scheduling boundaries. Registration changes and snapshots require
quiescence with completed or explicitly abandoned handoffs; reconstruct ranges
and device state on restore. Treat handler panic poisoning as terminal for that
VM. See [spec §5.4](../design/spec.md#54-borrowed-machine-devices-decision-28-stages-23).
