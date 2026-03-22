---
name: interface-design
description: Project-local wrapper for the vendored interface-design skill. Use for `dashboard/` frontend UI work only.
---

# Dashboard Interface Design

This repository vendors the upstream `Dammyjay93/interface-design` skill under `dashboard/tools/interface-design/`.

Before doing dashboard UI work with this skill, read these files in order:

1. `dashboard/tools/interface-design/.claude/skills/interface-design/SKILL.md`
2. `dashboard/tools/interface-design/.claude/skills/interface-design/references/principles.md`
3. `dashboard/tools/interface-design/.claude/skills/interface-design/references/validation.md`

Load extra references only when needed:

- `dashboard/tools/interface-design/.claude/skills/interface-design/references/critique.md`
  Use when reviewing or refining an existing UI after the first pass.
- `dashboard/tools/interface-design/.claude/skills/interface-design/references/example.md`
  Use when you need a concrete example of the intended design reasoning pattern.
- `dashboard/tools/interface-design/reference/system-template.md`
  Use when creating or rewriting the project design memory file.
- `dashboard/tools/interface-design/reference/examples/system-precision.md`
- `dashboard/tools/interface-design/reference/examples/system-warmth.md`
  Use these example systems only as reference material, not as copy-paste defaults.

## Project Overrides

Apply the upstream guidance with these repository-specific overrides:

- Scope is limited to `dashboard/` unless the user explicitly expands it.
- Treat upstream references to `.interface-design/system.md` as `dashboard/.interface-design/system.md`.
- Default audit/extract targets are:
  - `dashboard/src`
  - `dashboard/src/styles`
  - `dashboard/guidelines/Guidelines.md`
- Preserve the existing dashboard/Tauri product context. Do not drift into marketing-site patterns.

## Command Mapping

The vendored upstream `.claude/commands/*.md` files are reference workflows only. Codex does not expose them as native slash commands in this repo.

Map them to normal Codex behavior instead:

- `init`: establish or refine dashboard UI direction, then optionally create/update `dashboard/.interface-design/system.md`
- `status`: read and summarize `dashboard/.interface-design/system.md`
- `audit`: inspect `dashboard/src`, `dashboard/src/styles`, and `dashboard/guidelines/Guidelines.md` against the saved system
- `extract`: derive reusable dashboard design patterns from the same paths and offer to save them
- `critique`: run an internal design critique and improve the UI before presenting it

## Working Rule

If the user asks for frontend work outside `dashboard/`, do not use this skill unless they explicitly want the dashboard design language applied there too.
