---
name: finn-docs-sync
description: Update the Finn agent guide (CLAUDE.md, symlinked as AGENTS.md) so it matches the current repo. Use after adding or removing a package, pipeline stage, generated artifact, config contract, safety gate, firmware environment, or skill, when a command or layout in the guide has gone stale, or when asked to refresh/audit the agent instructions.
---

Read `.agents/skills/finn-docs-sync/SKILL.md` completely and follow it as the canonical workflow. Resolve all references relative to that canonical skill directory. Do not maintain a separate Claude-specific copy of the guide rules.
