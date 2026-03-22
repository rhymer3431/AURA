---
name: interface-design:status
description: Codex workflow reference for summarizing the current saved interface system.
---

# interface-design status

Use this workflow to show the current design system state from `.interface-design/system.md`.

## What to Show

If `.interface-design/system.md` exists, summarize:

```text
Design System: [Project Name]

Direction: [Precision & Density / Warmth / etc]
Foundation: [Cool slate / Warm stone / etc]
Depth: [Borders-only / Subtle shadows / Layered]

Tokens:
- Spacing base: 4px
- Radius scale: 4px, 6px, 8px
- Colors: [count] defined

Patterns:
- Button Primary (36px h, 16px px, 6px radius)
- Card Default (border, 16px pad)
- [other patterns...]

Last updated: [from git or file mtime]
```

If no `system.md` exists, report that clearly and point to the next useful Codex workflow:

```text
No design system found.

Next steps:
1. Build UI and establish the system during implementation
2. Perform the extract workflow to derive patterns from existing code
```

## Implementation

1. Read `.interface-design/system.md`
2. Parse direction, tokens, and patterns
3. Format a concise summary
4. If no system exists, suggest the next workflow without using slash-command language
