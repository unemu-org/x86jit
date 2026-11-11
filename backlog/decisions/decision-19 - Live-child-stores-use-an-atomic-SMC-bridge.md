---
id: decision-19
title: Live child stores use an atomic SMC bridge
date: '2026-11-11 07:34'
status: accepted
---
## Context

READ: decision-15's RFMEM/vfork contract shares guest bytes with running parent
vCPUs. `Memory::clone` is the separate private-copy primitive; `Vm::child_view`
keeps the same backing address. A page untagged at snapshot creation may acquire
a cached translation before a child writes it. Post-exit reconciliation leaves
that translation executable throughout the child's lifetime.

The ordinary `Memory::note_write` notifier takes locks and allocates. Calling it
from a child would break the allocation-free, no-parent-lock contract.

## Decision

Amend decision-15's SMC protocol and blanket prohibition on shared engine atomics.
Keep live sharing supported. Keep snapshot-tagged stores as preflight refusals,
and use only a narrow atomic bridge for stores to pages tagged later:

1. Commit bytes and journal written pages in a per-view preallocated atomic
   bitmap covering mapped RAM. No child allocation, locking or reference-count
   operation is needed.
2. Execute a SeqCst fence, then check the stable live atomic code tags. This is
   the writer side of spec §10's tag-transition protocol. Use the full fence even
   with a process-barrier parent: a separate child process sharing the address
   space is outside PRIVATE_EXPEDITED's process scope.
3. On a live tag hit, publish the bridge-global pending flag and notify every vCPU
   through stable append-only dispatch nodes, including registrations after view
   creation. Notification executes a SeqCst fence between publishing that flag
   and reading the links. Registration executes a SeqCst fence between publishing
   its link and reading the pending flag. A child can then signal a guest peer.
4. Parent dispatch drains page journals under normal SMC serialization, expands
   code aliases and invalidates affected sources and cache entries. Alias removal
   changes the active parent-side graph immediately. While **any** child view is
   live, it retains the removed declaration in shared parent-side history and
   pins every currently set code tag in the affected pre-removal components.
   Ordinary registration releases, page clears and whole-cache clears preserve
   those pinned bits without a transient clear. Drains expand child journals
   through both active and retired declarations, including transitive paths.
   Keep history and pinned tags across every drain and across individual view
   drops, including when another view is created after removal. View registration,
   removal, journal expansion and parent-side view drop use the alias mutex; the
   journal registry is locked after it. The last view's drop recomputes pinned
   tags from active owners and declarations, then releases the retired history.
   Without live views, removal clears tags immediately. Removal needs no fence,
   pending publication or broadcast of its own.
5. A per-vCPU epoch brackets drain/acknowledgement and preserves notifications
   racing the acknowledgement, assuming fewer than 2^64 notifications to that
   node between epoch samples; other dispatch bits remain independently owned.

READ ordering proof: the two SeqCst fences form a publish/read handshake. If a
notifier misses a newly published link, registration's pending read must observe
the notification; if registration misses the pending publication, notification
must observe its link. Both reads missing would require each fence to precede
the other in the SeqCst order. The parent's dispatch-word mutex serializes
registration's pending read with drains that clear the flag. This proof does
not depend on stronger ordering inside OnceLock.

READ racing-write coverage: the child orders `store -> journal -> SeqCst fence
-> tag read -> pending publication -> notify`. A child may pause after committing
its bytes but before publishing its journal bit. Removal retains its tag and
relationship under the alias mutex. Dispatch may drain an empty journal and
acknowledge while that child is paused; when the child resumes, its journal
publication precedes its fence and its tag read still sees the set bit. It
therefore publishes a new notification. The subsequent Acquire journal drain
expands the write through the retained relationship and invalidates the formerly
aliased translation. If an earlier drain already consumed the bit, that drain
already supplied the invalidation. The per-vCPU epoch preserves notification
racing acknowledgement. Forcing a drain during removal cannot cover a journal
publication that has not happened yet, so that broadcast is unnecessary with
retained tags and insufficient with immediate tag clearing.

READ lifetime coverage: safe view borrows prevent the last view from dropping
while a step still uses it. Parent-side drop follows child exit/exec and final
written-page reconciliation. The alias mutex serializes last-drop tag recomputation
and history release with both registration of a new view and journal expansion.
All tag-clear paths preserve pinned bits during that interval; the watermark
continues to widen monotonically and the tag epoch is not reset. Active direct
translations, lift/fetch owners and declarations determine the tags that survive
last drop. Child stepping uses only the atomic bridge; the parent Memory reference
is used exclusively by the view's parent-side destructor.

ASSUMED epoch bound: fewer than 2^64 notifications occur to one dispatch node
during one drain/acknowledgement interval. A MAX → 0 transition changes equality
and is detected. Exactly a full modulus of increments restores equality and can
erase a racing notice; the global dirty flag alone cannot wake a compiled chain
polling its cleared per-vCPU word. The protocol's no-loss claim is conditional
on this bound.

## Consequences

Child stepping borrows only atomic code tags and the bridge, alongside the view's
owned mapping, fetch and CPU snapshots. The existing VM lifetime keeps those
atomics, nodes and backing alive. Parent locks, callbacks, caches and destructors remain unavailable
to child execution. Partial host stores report their completed prefixes; deferred
vector commits, REP elements and atomic transfers use the same hook.

This costs a full fence per committed transfer, another per live-tag notification,
one per dispatch-node registration, and an atomic journal bitmap per view (one bit
per mapped RAM page, rounded per region). Live code writes broadcast
to the vCPU registry; registered dispatch nodes and words survive until Memory
drops once child views have enabled the bridge. Parent invalidation is page-granular.
The ordinary arena log still supports final reconciliation and watched data.
Removed alias declarations consume shared parent-side history until the last
live view is dropped. Pinned tags can cause conservative extra notifications and
child invalidations of formerly related pages during that interval; ordinary
writes expand only through active declarations. A new view can conservatively
snapshot pinned tags and refuse stores to them under the existing snapshot-code
preflight rule. History is unbounded while views overlap continuously. Last drop
releases it with the deferred tag clear; an ordinary drain cannot establish
quiescence for an unpublished child journal bit. Alias removal adds no forced
notification. View drop runs parent locks and cleanup and must stay in the parent.

A restriction requiring all parent execution and translation to remain quiescent
would preserve a fully isolated snapshot, but would remove decision-15's intended
RFMEM use case. Calling the ordinary notifier would instead sacrifice child lock
and allocation safety. The atomic bridge retains both intended properties.

MEASURED witnesses belong to `child_step` and core unit tests: snapshot before
tagging, translate on two vCPUs, child store, signal a sibling and observe changed
code before post-exit reconciliation; and a second child store during parent
acknowledgement. These exercise interpreter/JIT execution, scalar/vector/REP/atomic
stores, late vCPU registration and independent exit/exhaustion notices.
On ARM64 macOS, the late-tag witness fails against the original implementation:
both warmed interpreter vCPUs return the old value 0 after the child patches it
to 2. With the bridge, both backends observe 2 before parent reconciliation.

## Accepted amendments — 2026-11-11

The maintainer's acceptance remains in force. Review amendments preserve alias
relationships for deferred child writes, make the registration/notification
ordering explicit with paired SeqCst fences, qualify epoch equality with its
full-modulus bound, and correct the pending flag's ownership to bridge-global.
Regression witnesses cover removal between a notified child store and dispatch
(interpreter and JIT), removed transitive aliases, sibling registration concurrent
with a child store, and a MAX → 0 epoch transition during acknowledgement.
Deterministic cfg(test) hooks run after a committed guest store both before journal
publication and after publication but before the fence/tag read. Each timing
covers dispatch before or after the child resumes. The pre-publication witness
allows dispatch to drain and acknowledge the still-empty journal, then checks that
the resumed child notifies both vCPUs and evicts the formerly aliased cached
translation. The witnesses also cover another live journal, independent dispatch
bits, history across drains, individual and overlapping view lifetimes, ordinary
and bulk tag clears, and immediate removal after all views have dropped.
Concurrent host execution is a witness, not a memory-model proof; the ordering
argument above is READ evidence and the full-modulus bound remains ASSUMED.

MEASURED: bypassing retired-alias expansion makes the removal-before-dispatch
regression return the stale value 1 instead of the patched value 2. With expansion
enabled, the interpreter and JIT both return 2 before final reconciliation.

MEASURED: temporarily restoring immediate tag clearing plus a forced removal
notification makes the pre-journal-publication witness fail at `no formerly
aliased translation survives`. Dispatch drains and acknowledges before the child
publishes its bit; its later cleared-tag read produces no new notification. With
deferred clearing restored, both pause points and both dispatch timings pass.
