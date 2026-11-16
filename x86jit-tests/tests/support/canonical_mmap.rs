//! Anonymous host mappings that an identity-mapped `Vm` can use.
//!
//! An identity mapping makes each guest address equal to its host address, so the
//! whole host span must be canonical as a guest span. The lower canonical half ends
//! at 2^47 (SDM Vol 1 §3.3.7.1 p. 3-10). `Memory::from_host_ram` rejects any span
//! above that with `HostRamError::NonCanonicalSpan`. `mmap(NULL, …)` gives no such
//! promise: aarch64 Linux places anonymous mappings near the top of its 48-bit user
//! range. [`map_below_canonical_top`] asks the kernel for a low placement and panics,
//! with the addresses it tried, when it cannot get one.

/// Exclusive top of the lower canonical half (SDM Vol 1 §3.3.7.1 p. 3-10).
const CANONICAL_TOP: u64 = 1 << 47;
/// Distance between successive placement hints, and the largest mapping served.
const STRIDE: u64 = 1 << 30;
/// Hints tried below the canonical top before giving up: 1 TiB of address space.
const ATTEMPTS: u64 = 1024;

/// Flags that make a placement hint exact without replacing an existing mapping.
/// Linux before 4.17 ignores `MAP_FIXED_NOREPLACE` and treats the address as a plain
/// hint; macOS has no such flag and honours a free hint. Every result is checked.
#[cfg(any(target_os = "linux", target_os = "android"))]
const NO_REPLACE: libc::c_int = libc::MAP_FIXED_NOREPLACE;
#[cfg(target_os = "freebsd")]
const NO_REPLACE: libc::c_int = libc::MAP_FIXED | libc::MAP_EXCL;
#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "freebsd")))]
const NO_REPLACE: libc::c_int = 0;

/// Map `len` bytes of private anonymous memory with protection `prot`. The mapping
/// and the byte after it both lie below 2^47, so the mapping can back an identity
/// mapping and a test can still address one past its end. The caller owns the
/// mapping and releases it with `munmap`.
///
/// # Panics
/// Panics when the host grants no placement below 2^47. A test that needs an
/// identity mapping cannot run without one, and must not pass silently.
pub fn map_below_canonical_top(len: usize, prot: libc::c_int) -> *mut libc::c_void {
    assert!(
        len as u64 <= STRIDE,
        "{len} bytes exceed the placement stride"
    );
    let flags = libc::MAP_PRIVATE | libc::MAP_ANON;
    let low = |p: *mut libc::c_void| {
        p != libc::MAP_FAILED && (p as u64).saturating_add(len as u64) < CANONICAL_TOP
    };
    let map = |hint: u64, extra: libc::c_int| {
        // SAFETY: an anonymous mapping that replaces nothing: either no hint, or a
        // hint with `NO_REPLACE` (or a plain hint the kernel may ignore).
        let p = unsafe { libc::mmap(hint as *mut _, len, prot, flags | extra, -1, 0) };
        if p != libc::MAP_FAILED && !low(p) {
            // SAFETY: `p` is the mapping just created above, of exactly `len` bytes.
            unsafe { libc::munmap(p, len) };
        }
        p
    };
    // Keep the kernel's default placement where it is already low (x86-64 Linux,
    // macOS, FreeBSD on amd64).
    let default = map(0, 0);
    if low(default) {
        return default;
    }
    // Otherwise search downward from the canonical top, one stride at a time.
    let mut hint = CANONICAL_TOP - STRIDE;
    for _ in 0..ATTEMPTS {
        let p = map(hint, NO_REPLACE);
        if low(p) {
            return p;
        }
        hint -= STRIDE;
    }
    panic!(
        "no {len}-byte anonymous mapping could be placed below 2^47 on this host: the \
         default placement was {:#x}, and hints from {:#x} down to {:#x} were \
         refused or ignored. An identity-mapped Vm needs a canonical host span \
         (SDM Vol 1 §3.3.7.1 p. 3-10), so this test cannot run here.",
        default as u64,
        CANONICAL_TOP - STRIDE,
        hint + STRIDE,
    );
}
