---
name: interface-design
description: Project-local wrapper for the vendored interface-design skill. Use for `dashboard/` frontend UI work only.
---

# AURA Dashboard Interface Design

This repository vendors the upstream `Dammyjay93/interface-design` skill under `dashboard/tools/interface-design/`.

The vendored snapshot keeps its upstream `.claude` path layout for compatibility and provenance, but the canonical entrypoint for this repository is this Codex wrapper.

The compatibility filenames under `.claude/skills/interface-design/references/` remain, but in this repository they no longer mean "SnowUI-like quiet workspace." Treat them as the local AURA dashboard target references derived from the approved Pipeline Overview screen.

Before doing dashboard UI work with this skill, read these files in order:

1. `dashboard/tools/interface-design/.claude/skills/interface-design/SKILL.md`
2. `dashboard/tools/interface-design/.claude/skills/interface-design/references/principles.md`
3. `dashboard/tools/interface-design/.claude/skills/interface-design/references/validation.md`
4. `.claude/skills/interface-design/references/snowui-design-system.md`
5. `dashboard/.interface-design/system.md`
6. `.claude/skills/interface-design/references/snowui-dashboard.md`

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
- Treat `.claude/skills/interface-design/references/snowui-design-system.md` as the canonical local design-system brief for the approved AURA dashboard target.
- Treat `.claude/skills/interface-design/references/snowui-dashboard.md` as the canonical local layout and composition brief for the approved AURA dashboard target.
- Default audit/extract targets are:
  - `dashboard/src`
  - `dashboard/src/styles`
  - `dashboard/guidelines/Guidelines.md`
- Preserve the existing dashboard/Tauri product context. Do not drift into marketing-site patterns.
- Use the local dashboard target references to refine region hierarchy, pastel KPI grouping, panel scale, process cards, and the live-view workstation framing from the approved image.
- Reject Claude-like or generic "quiet workspace" defaults:
  - same-background sidebar with almost invisible separation
  - monochrome shell-first hierarchy where every panel lives on the same plane
  - tiny control pills and shortcut chips as the main organizing device
  - chat-product framing, composer-like bottom tooling, or settings-page density
- Prefer distinct but still refined region grouping:
  - dedicated left navigation rail
  - breadcrumb/status top strip
  - colored KPI row
  - large neutral work panels
  - stacked right-side support modules
- When the local target references conflict with the upstream generic guidance, the local target references win for this repository.

## Local Target Contract

Use the local AURA target references as the default operating contract for dashboard UI work:

- Reuse existing tokens and existing components before adding new ones.
- Prefer semantic tokens, but allow a small number of intentional dashboard-specific surface families when they are repeated:
  - warm canvas
  - fog panel
  - white module
  - planner blue tile
  - perception lilac tile
  - status mint accents
- Keep the first read obvious at a glance. The operator should understand the main regions before reading small text.
- Prefer base components and variants; page-specific components are the fallback when reuse truly breaks down.
- If behavior is the same and only presentation shifts, absorb it into variants, slots, or wrappers instead of minting a new base.
- If a pattern repeats three times or more, promote it into a reusable base/variant or saved system pattern.
- Favor large readable blocks over whisper-light card stacks.
- Use soft shadows and low-contrast borders together when they clarify major panel groups.
- Let telemetry color families stay visible and useful instead of muting them into a grayscale shell.

## Command Mapping

The vendored upstream `.claude/commands/*.md` files are reference workflows only. In this repo they are Codex-oriented workflow references, not native slash commands.

Map them to normal Codex behavior instead:

- `init`: establish or refine dashboard UI direction, then optionally create/update `dashboard/.interface-design/system.md`
- `status`: read and summarize `dashboard/.interface-design/system.md`
- `audit`: inspect `dashboard/src`, `dashboard/src/styles`, and `dashboard/guidelines/Guidelines.md` against the saved system
- `extract`: derive reusable dashboard design patterns from the same paths and persist them only when the task requires saved system updates
- `critique`: run an internal design critique and improve the UI before presenting it

Treat those workflow names as documentation labels, not as callable slash commands.

## Working Rule

If the user asks for frontend work outside `dashboard/`, do not use this skill unless they explicitly want the dashboard design language applied there too.

If the user asks to align the dashboard with the current project reference, update `dashboard/.interface-design/system.md` first when the design direction changes materially, then implement UI changes against that saved system.

When aligning to the approved AURA target, preserve the runtime-ops purpose:

- bright but disciplined operator-console framing
- generous rounded panels instead of whisper-light shells
- pastel KPI surfaces for high-level telemetry families
- a dominant live-view workstation
- a right-side module column for health, process, and sensor context
- compact text where needed, but not chat-app minimalism
