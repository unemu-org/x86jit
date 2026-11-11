#[cfg(target_os = "macos")]
pub fn retired() -> u64 {
    // Apple SDK sys/resource.h, rusage_info_v4: ri_instructions follows 29 u64s.
    #[repr(C)]
    #[derive(Default)]
    struct Usage {
        uuid: [u8; 16],
        before: [u64; 29],
        instructions: u64,
        after: [u64; 5],
    }
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut Usage) -> i32;
    }
    let mut usage = Usage::default();
    // SAFETY: the SDK V4 layout is fully allocated and lives through the call.
    assert_eq!(
        unsafe { proc_pid_rusage(std::process::id() as i32, 4, &mut usage) },
        0
    );
    assert_ne!(usage.instructions, 0, "instruction counter unavailable");
    usage.instructions
}
#[cfg(not(target_os = "macos"))]
pub fn retired() -> u64 {
    0
}
