---
name: interface-design:init
description: Codex workflow reference for establishing or refining interface direction with craft and consistency.
---

## Purpose

Use this workflow when you need to establish or refine interface direction for dashboards, admin panels, apps, tools, or other product UI.

Do not use it for landing pages or marketing surfaces. Switch to a separate frontend design workflow instead.

## Required Reading

Before writing code, read:

1. `../skills/interface-design/SKILL.md`

Do not skip it. The craft principles and validation checks live there.

## Intent First

Before touching code, state the answers to these questions in your working notes:

- Who is the actual human using this interface?
- What must they accomplish?
- What should the interface feel like?

If any of those answers materially affect the design direction and cannot be derived from the repo or prompt, ask the user a concise question. Otherwise, state your assumption and continue.

## Before Writing Each Component

State the intent and the technical approach:

```text
Intent: [who, what they need to do, how it should feel]
Palette: [foundation + accent — and why they fit the product's world]
Depth: [borders / subtle shadows / layered — and why]
Surfaces: [your elevation scale — and why this temperature]
Typography: [your typeface choice — and why it fits the intent]
Spacing: [your base unit]
```

Every choice must be explainable. If the rationale is only "it's common" or "it works," the decision defaulted.

## Communication

Do the work directly. Keep reasoning concrete, but do not narrate fake modes or meta workflows.

Bad:

- "I'm in ESTABLISH MODE"
- "Let me check system.md..."

Good:

- State the explored domain, proposed direction, and implementation reasoning.

## Direction Workflow

1. Read the required files above, even if `system.md` already exists.
2. Check whether `.interface-design/system.md` exists.
3. If it exists, apply the established decisions unless the user asked to change direction.
4. If it does not exist, assess the domain and produce:
   - domain concepts
   - color world
   - signature element
   - default patterns to reject
5. Propose a direction that ties those four outputs together.
6. Ask the user only if a high-impact direction choice remains ambiguous.
7. Build and then run the craft checks before presenting results.

## Persistence

Do not automatically interrupt the task to ask whether patterns should be saved.

Update `.interface-design/system.md` only when one of these is true:

- the user explicitly asks to save or persist the design system
- the task explicitly includes establishing or refining the saved system
- the current work materially changes the saved design direction and the system file must stay authoritative
