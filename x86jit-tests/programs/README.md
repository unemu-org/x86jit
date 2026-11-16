# Guest programs

One file. The program is our own work; the binary statically links musl libc, whose
licence notice is [below](#licence-notice-musl-libc-in-pthreadself):

| File | What | Used by |
|---|---|---|
| `pthreads.elf` | A static-musl C program: four pthreads each incrementing a shared counter under a mutex 100 000 times, so the result is deterministically `400000` — but only if guest threads, cross-thread atomics, and the futex-backed mutex and join all work. Statically linked, not stripped. | `tests/mt.rs` (`pthreads_counter_{interp,jit,jit_background}`) |

`mt.rs` services this program's syscalls inline rather than through a shim, which is why
the test belongs in this repository at all: it exercises the M7 threading stack — `clone`
spawning a host thread per guest thread over one `Arc<Vm>`, a real `futex` blocking and
waking them — without needing a Linux userland.

**Its C source was not preserved.** That is a gap, recorded rather than glossed: the binary
is reproducible in behaviour (any equivalent four-thread mutex program would do) but not
byte-for-byte. Reconstructing the source is open work.

## Licence notice: musl libc in `pthreads.elf`

`pthreads.elf` is statically linked against [musl libc](https://musl.libc.org/), so the
binary contains musl's code. musl is MIT-licensed, and the MIT licence requires its
copyright and permission notice to accompany every copy of the binary. The musl release
it was linked against was not recorded.

musl's copyright and permission notice is in [`MUSL-COPYRIGHT`](MUSL-COPYRIGHT), copied
verbatim from the `COPYRIGHT` file of musl 1.2.6. musl is MIT-licensed as a whole; the
notice of the release actually linked was not recorded, and the 1.2.6 file carries the
same MIT terms.

## Where everything else went

The third-party guest binaries this directory used to hold — busybox, sqlite3, lua,
`djpeg`, CPython and its stdlib subset, the glibc and musl loaders, the Go servers — moved
to `unemulinux` (a separate repository) with the syscall shim that makes
them runnable, and are fetched and checksum-pinned there rather than committed.

That move also ended this repository's redistribution of `busybox` (GPL-2.0), the glibc
loader (LGPL-2.1), and `lua`/`python3` — both of which statically link GNU Readline and are
therefore GPL-3.0 combined works — none of which had a corresponding-source offer or a
licence text alongside them. See [`PROVENANCE.md`](../../PROVENANCE.md) §5.
