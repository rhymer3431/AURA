# Vendored `interface-design` Snapshot

- Upstream repository: `https://github.com/Dammyjay93/interface-design`
- Upstream ref: `main`
- Snapshot commit: `8c407c1c42890010a9eb403a9f419b1eeadcfdad`
- Retrieved for this repo on: `2026-03-22`

## Included Paths

- `.claude/skills/interface-design/**`
- `.claude/commands/*.md`
- `reference/**`
- `.claude-plugin/plugin.json`

## Local Integration Notes

- This snapshot is vendored under `dashboard/tools/interface-design/`.
- Codex loads the project-local wrapper at `.claude/skills/interface-design/SKILL.md`.
- The vendored snapshot keeps the upstream `.claude` and `.claude-plugin` paths for compatibility and provenance, but the content is adapted for Codex use in this repository.
- The wrapper constrains this skill to dashboard frontend work and maps the design memory file to `dashboard/.interface-design/system.md`.
- The local reference files under `.claude/skills/interface-design/references/` keep their historical filenames for compatibility, but their content is adapted to the approved AURA Pipeline Overview dashboard target.
- No global installation under `~/.codex/skills` is part of this integration.
