# .beads

The beads (`bd`) issue database for x86jit: issue prefix `xj`, embedded Dolt in
`embeddeddolt/` (gitignored). It reaches other machines through `bd dolt push` /
`bd bootstrap` via `refs/dolt/data` on `origin`, not through git commits.

How this project uses it: [`backlog/docs/working-on-tasks.md`](../backlog/docs/working-on-tasks.md)
and [`backlog/docs/agent-workflow.md`](../backlog/docs/agent-workflow.md). Install and fresh-clone
setup: [`backlog/docs/dev-setup.md`](../backlog/docs/dev-setup.md) § beads.
