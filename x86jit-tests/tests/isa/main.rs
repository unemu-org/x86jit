//! Instruction semantics: x87, MXCSR and floating-point special values, MMX,
//! SSE, AVX and AVX-512 forms, XSAVE, flags and CPUID.
//!
//! One module per former test binary; a test's path starts with its module.

mod cpuid;
mod cvt_integer_indefinite;
mod evex_modifiers;
mod flag_stack;
mod fp_special_cases;
mod gathers;
mod machine_empty_fpu;
mod mmx;
mod mxcsr;
mod mxcsr_modes;
mod real16_x87_model;
mod sse4a_adx;
mod sse_memory_forms;
mod v3_memory_forms;
mod x87_control_word;
mod x87_encodings;
mod x87_exception_flags;
mod x87_faults;
mod x87_forms;
mod x87_integer_store;
mod x87_mf;
mod x87_pointers;
mod x87_rest;
mod x87_suppressed_store_watch;
mod xsave;
