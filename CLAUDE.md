# Project conventions

## Development logs (`logs.md`)

Every OpenSpec change folder under `docs/specs/changes/*/` keeps a `logs.md` of
development logs, one `## commit-N-logs` section per commit.

**Log only what fell outside the plan, or bugs encountered during implementation.**

- Do NOT restate the plan's tasks — those already live in `tasks/commit-N.md` and the
  commit message. Re-describing them is noise.
- Do NOT document the same thing more than once.
- Record the non-obvious: deviations from the design, constraints discovered while
  implementing, and bugs hit (and how they were resolved).
- Keep entries terse. If a commit had no deviations and no bugs, it needs no log entry.
