//! Closed libcall table for the pinned Cranelift lowerings.

use cranelift::codegen::{ir::LibCall, isa::OwnedTargetIsa};
use cranelift_jit::JITBuilder;

macro_rules! unary {
    ($name:ident, $ty:ty, $method:ident) => {
        extern "C" fn $name(value: $ty) -> $ty {
            value.$method()
        }
    };
}
unary!(ceil_f32, f32, ceil);
unary!(ceil_f64, f64, ceil);
unary!(floor_f32, f32, floor);
unary!(floor_f64, f64, floor);
unary!(trunc_f32, f32, trunc);
unary!(trunc_f64, f64, trunc);
unary!(nearest_f32, f32, round_ties_even);
unary!(nearest_f64, f64, round_ties_even);

extern "C" fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}
extern "C" fn fma_f64(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

// Link these addresses into the host binary, including when it is static.
unsafe extern "C" {
    fn memcpy(dst: *mut u8, src: *const u8, len: usize) -> *mut u8;
    fn memmove(dst: *mut u8, src: *const u8, len: usize) -> *mut u8;
    fn memset(dst: *mut u8, byte: i32, len: usize) -> *mut u8;
    fn memcmp(a: *const u8, b: *const u8, len: usize) -> i32;
}

// Cranelift 0.135.8's X86Pshufb signature passes two i8x16 values in xmm0/xmm1
// and returns i8x16 in xmm0. Stable Rust cannot express that vector signature.
// This leaf uses only baseline SSE2 and caller-saved integer registers. It
// implements PSHUFB: SDM Vol 2B, PSHUFB, Operation (128-bit SSE), p. 4-423.
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "C" fn swizzle() {
    core::arch::naked_asm!(
        "sub rsp, 56",
        "movdqu [rsp], xmm0",
        "movdqu [rsp + 16], xmm1",
        "xor eax, eax",
        "2:",
        "movzx ecx, byte ptr [rsp + rax + 16]",
        "xor edx, edx",
        "test ecx, 128",
        "jnz 3f",
        "and ecx, 15",
        "mov dl, byte ptr [rsp + rcx]",
        "3:",
        "mov byte ptr [rsp + rax + 32], dl",
        "inc eax",
        "cmp eax, 16",
        "jb 2b",
        "movdqu xmm0, [rsp + 32]",
        "add rsp, 56",
        "ret",
    );
}

// Only cross-ISA compilation can reference X86Pshufb on another host. Such code
// may be finalized for relocation tests but must never be executed here.
#[cfg(not(target_arch = "x86_64"))]
extern "C" fn swizzle() {
    panic!("X86Pshufb requires an x86-64 execution host");
}

fn address(call: LibCall) -> *const u8 {
    use LibCall::*;
    match call {
        CeilF32 => ceil_f32 as *const u8,
        CeilF64 => ceil_f64 as *const u8,
        FloorF32 => floor_f32 as *const u8,
        FloorF64 => floor_f64 as *const u8,
        TruncF32 => trunc_f32 as *const u8,
        TruncF64 => trunc_f64 as *const u8,
        NearestF32 => nearest_f32 as *const u8,
        NearestF64 => nearest_f64 as *const u8,
        FmaF32 => fma_f32 as *const u8,
        FmaF64 => fma_f64 as *const u8,
        Memcpy => memcpy as *const u8,
        Memmove => memmove as *const u8,
        Memset => memset as *const u8,
        Memcmp => memcmp as *const u8,
        X86Pshufb => swizzle as *const u8,
        // The engine emits no TLS IR and explicitly disables outline stack probes.
        Probestack | ElfTlsGetAddr | ElfTlsGetOffset => {
            panic!("unsupported Cranelift libcall: {call}")
        }
    }
}

pub(crate) fn builder(isa: OwnedTargetIsa) -> JITBuilder {
    let mut builder = JITBuilder::with_isa(isa, Box::new(|call| format!("x86jit_libcall_{call}")));
    for &call in LibCall::all_libcalls() {
        if matches!(
            call,
            LibCall::Probestack | LibCall::ElfTlsGetAddr | LibCall::ElfTlsGetOffset
        ) {
            continue;
        }
        builder.symbol(format!("x86jit_libcall_{call}"), address(call));
    }
    // Lookup functions run in reverse registration order. Never return None:
    // that would allow Cranelift's built-in dlsym lookup to run next.
    builder.symbol_lookup_fn(Box::new(|name| panic!("unregistered JIT symbol: {name}")));
    builder.memory_provider(crate::code_memory::provider(std::sync::Arc::new(
        crate::memory_stats::CodeCounters::default(),
    )));
    builder
}

#[cfg(test)]
pub(crate) fn assert_supported_relocations(code: &cranelift::codegen::CompiledCode) {
    use cranelift::codegen::{binemit::Reloc, ir::ExternalName, FinalizedRelocTarget};
    for reloc in code.buffer.relocs() {
        assert_eq!(
            reloc.kind,
            Reloc::Abs8,
            "unexpected JIT relocation: {reloc:?}"
        );
        let FinalizedRelocTarget::ExternalName(ExternalName::LibCall(call)) = &reloc.target else {
            panic!("unexpected JIT relocation target: {reloc:?}");
        };
        assert_eq!(reloc.addend, 0, "unexpected libcall addend: {reloc:?}");
        assert!(!address(*call).is_null());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift::codegen::ir;
    use cranelift::prelude::*;
    use cranelift_jit::JITModule;
    use cranelift_module::{Linkage, Module};

    #[test]
    fn embedder_libcall_all_signatures_finalize_on_both_isas() {
        for host in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
            let mut flags = settings::builder();
            flags.set("use_colocated_libcalls", "false").unwrap();
            let isa = cranelift::codegen::isa::lookup(host.parse().unwrap())
                .unwrap()
                .finish(settings::Flags::new(flags))
                .unwrap();
            let mut module = JITModule::new(builder(isa));
            for &call in LibCall::all_libcalls() {
                if matches!(
                    call,
                    LibCall::Probestack | LibCall::ElfTlsGetAddr | LibCall::ElfTlsGetOffset
                ) {
                    continue;
                }
                let mut ctx = module.make_context();
                ctx.func.signature = call.signature(module.isa().default_call_conv(), types::I64);
                let signature = ctx.func.signature.clone();
                let ext = ctx.func.import_signature(signature);
                let callee = ctx.func.import_function(ir::ExtFuncData {
                    name: ir::ExternalName::LibCall(call),
                    signature: ext,
                    colocated: false,
                    patchable: false,
                });
                let mut fbctx = FunctionBuilderContext::new();
                let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
                let block = b.create_block();
                b.append_block_params_for_function_params(block);
                b.switch_to_block(block);
                let params = b.block_params(block).to_vec();
                let inst = b.ins().call(callee, &params);
                let results = b.inst_results(inst).to_vec();
                b.ins().return_(&results);
                b.seal_all_blocks();
                b.finalize(module.isa().frontend_config());
                let id = module
                    .declare_function(
                        &format!("test_{call}"),
                        Linkage::Export,
                        &ctx.func.signature,
                    )
                    .unwrap();
                module.define_function(id, &mut ctx).unwrap();
                assert_supported_relocations(ctx.compiled_code().unwrap());
                module.finalize_definitions().unwrap();
                if (host.starts_with("aarch64") && cfg!(target_arch = "aarch64"))
                    || (host.starts_with("x86_64") && cfg!(target_arch = "x86_64"))
                {
                    check_native_call(call, module.get_finalized_function(id));
                }
            }
        }
    }

    fn check_native_call(call: LibCall, entry: *const u8) {
        // SAFETY: the trampoline was compiled with LibCall::signature, whose C
        // ABI is exactly the function type selected below. All memory operands
        // refer to live local buffers of the lengths passed.
        unsafe {
            match call {
                LibCall::CeilF32 | LibCall::FloorF32 | LibCall::TruncF32 | LibCall::NearestF32 => {
                    let f: extern "C" fn(f32) -> f32 = core::mem::transmute(entry);
                    assert_eq!(f(2.5), if call == LibCall::CeilF32 { 3.0 } else { 2.0 });
                    assert_eq!(f(-0.0).to_bits(), (-0.0f32).to_bits());
                    if call == LibCall::NearestF32 {
                        assert_eq!(f(3.5), 4.0);
                    }
                }
                LibCall::CeilF64 | LibCall::FloorF64 | LibCall::TruncF64 | LibCall::NearestF64 => {
                    let f: extern "C" fn(f64) -> f64 = core::mem::transmute(entry);
                    assert_eq!(f(2.5), if call == LibCall::CeilF64 { 3.0 } else { 2.0 });
                    assert_eq!(f(-0.0).to_bits(), (-0.0f64).to_bits());
                    if call == LibCall::NearestF64 {
                        assert_eq!(f(3.5), 4.0);
                    }
                }
                LibCall::FmaF32 => {
                    let f: extern "C" fn(f32, f32, f32) -> f32 = core::mem::transmute(entry);
                    assert_eq!(f(2.0, 3.0, 4.0), 10.0);
                }
                LibCall::FmaF64 => {
                    let f: extern "C" fn(f64, f64, f64) -> f64 = core::mem::transmute(entry);
                    assert_eq!(f(2.0, 3.0, 4.0), 10.0);
                }
                LibCall::Memcpy | LibCall::Memmove => {
                    let f: extern "C" fn(*mut u8, *const u8, usize) -> *mut u8 =
                        core::mem::transmute(entry);
                    let src = [1, 2, 3, 4];
                    let mut dst = [0; 4];
                    assert_eq!(f(dst.as_mut_ptr(), src.as_ptr(), 4), dst.as_mut_ptr());
                    assert_eq!(dst, src);
                    if call == LibCall::Memmove {
                        f(dst.as_mut_ptr().add(1), dst.as_ptr(), 3);
                        assert_eq!(dst, [1, 1, 2, 3]);
                    }
                }
                LibCall::Memset => {
                    let f: extern "C" fn(*mut u8, i32, usize) -> *mut u8 =
                        core::mem::transmute(entry);
                    let mut dst = [0; 4];
                    assert_eq!(f(dst.as_mut_ptr(), 0x1ab, 4), dst.as_mut_ptr());
                    assert_eq!(dst, [0xab; 4]);
                }
                LibCall::Memcmp => {
                    let f: extern "C" fn(*const u8, *const u8, usize) -> i32 =
                        core::mem::transmute(entry);
                    let a = [0u8, 255];
                    let b = [0u8, 128];
                    assert!(f(a.as_ptr(), b.as_ptr(), 2) > 0);
                    assert!(f(b.as_ptr(), a.as_ptr(), 2) < 0);
                    assert_eq!(f(a.as_ptr(), a.as_ptr(), 2), 0);
                }
                // The vector ABI has a separate native x86-64 execution witness.
                LibCall::X86Pshufb => {}
                _ => unreachable!(),
            }
        }
    }

    #[test]
    #[should_panic(expected = "unregistered JIT symbol: memcpy")]
    fn embedder_libcall_rejects_process_symbols_by_name() {
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let mut module = JITModule::new(builder(isa));
        let signature = LibCall::Memcpy.signature(module.isa().default_call_conv(), types::I64);
        let callee = module
            .declare_function("memcpy", Linkage::Import, &signature)
            .unwrap();
        let id = module
            .declare_function("test_unknown", Linkage::Export, &signature)
            .unwrap();
        let mut ctx = module.make_context();
        ctx.func.signature = signature;
        let callee = module.declare_func_in_func(callee, &mut ctx.func);
        let mut fbctx = FunctionBuilderContext::new();
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let block = b.create_block();
        b.append_block_params_for_function_params(block);
        b.switch_to_block(block);
        let params = b.block_params(block).to_vec();
        let inst = b.ins().call(callee, &params);
        let results = b.inst_results(inst).to_vec();
        b.ins().return_(&results);
        b.seal_all_blocks();
        b.finalize(module.isa().frontend_config());
        module.define_function(id, &mut ctx).unwrap();
        module.finalize_definitions().unwrap();
    }
}
