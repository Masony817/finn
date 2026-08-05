---
name: finn-docs-sync
description: Update the Finn agent guide (CLAUDE.md, symlinked as AGENTS.md) so it matches the current repo. Use after adding or removing a package, pipeline stage, generated artifact, config contract, safety gate, firmware environment, or skill, when a command or layout in the guide has gone stale, or when asked to refresh/audit the agent instructions.
---

# Finn Docs Sync

Keep `CLAUDE.md` true and the same size. It is the shared agent guide; `AGENTS.md`
is a symlink to it. Edit `CLAUDE.md` only.

Growth is the failure mode. Every session that appends "one more note" turns a
guide into a wall nobody reads. Assume the file is already the right length: a
new entry should usually replace or absorb an old one, not sit beside it.

## What belongs

Include only what an agent needs *before* it can work safely and would otherwise
get wrong:

- Repo structure and what each area is for.
- Contracts between artifacts (which file is the source, which is generated).
- Conventions a new file must match to look like the ones around it.
- Rules whose violation is expensive: safety gates, generated-file edits, what
  never gets committed.
- Entry-point commands, and the non-obvious invocations (a platform workaround, a
  required flag).

## What does not

- Anything a competent agent derives by reading one file it was already going to open.
- Current findings, open questions, roadmap, run results, or task state. Those
  belong in `docs/`, a run artifact, or the user's memory.
- Prose duplicated from `README.md`, `docs/`, or `packages/scopik/README.md`.
  Link instead, in the voice of "read X before doing Y".
- Speculative guidance for code that does not exist yet.

## Procedure

1. **Detect drift before editing.** Do not rewrite from impression. Check the
   claims the guide actually makes:

   ```bash
   git log --oneline -20
   git diff --stat HEAD~10..HEAD -- ':!logs'
   git ls-files | grep -v '^\.platformio' | sed 's|/[^/]*$||' | sort -u
   ```

   Then verify each concrete reference in `CLAUDE.md` still resolves: paths exist,
   scripts still take the flags shown, `[env:*]` names match `platformio.ini`,
   the ruff/pytest/CI commands match `pyproject.toml` and `.github/workflows/ci.yml`,
   and the skill table matches `.agents/skills/`.

2. **Classify each change.** For anything new in the repo, ask whether an agent
   would get it wrong without being told. If no, it does not go in the file.
   New pipeline stages, generated artifacts, source-of-truth configs, and safety
   gates almost always qualify. New helper functions, tests, and log directories
   almost never do.

3. **Edit in place, at the existing altitude.** Use the section that already
   exists; add a section only for a genuinely new area of the repo. Match the
   surrounding density and the imperative voice. When a rule is superseded,
   delete the old one in the same edit.

4. **Respect the hard rules while editing them.** The numbered rules in
   `## Hard rules` encode physical safety and derived-artifact integrity. Loosen
   or remove one only when the user explicitly says the underlying constraint has
   changed, and say so plainly in your summary.

5. **Verify.**

   ```bash
   ls -l AGENTS.md              # must still be a symlink -> CLAUDE.md
   ```

   Confirm every path and command you added or touched. Prefer running a command
   over trusting its name. Never leave a reference that does not resolve.

6. **Report** what you changed and what you deliberately left out, so the user can
   push back on either.

## Style

- No em dashes anywhere. Plain `-`.
- Imperative, specific, and falsifiable. "Never hand-edit `lqr_seeded_config.h`;
  regenerate it with `run_lqr_sim.py --firmware-header`" beats "be careful with
  generated files".
- One idea per bullet. Tables for enumerations, fenced blocks for commands and
  layout.
- State the reason when the rule is non-obvious, in the same sentence, briefly.

## Adding a skill

New skills go in `.agents/skills/<name>/SKILL.md` (canonical, harness-neutral)
with a `.claude/skills/<name>/SKILL.md` pointer that redirects to it, mirroring
`finn-gap-review`. Add `agents/openai.yaml` when Codex should surface it. Then add
one row to the skill table in `CLAUDE.md`, and nothing more.
