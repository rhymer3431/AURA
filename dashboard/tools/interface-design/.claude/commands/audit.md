---
name: interface-design:audit
description: Codex workflow reference for checking UI code against the saved interface system.
---

# interface-design audit

Use this workflow to compare existing UI code against the saved design system.

## Recommended Targets

Audit a specific file or directory when the user scopes the task.

Otherwise, use the default UI targets for this repository:

- `dashboard/src`
- `dashboard/src/styles`
- `dashboard/guidelines/Guidelines.md`

## What to Check

If `.interface-design/system.md` exists, check:

1. **Spacing violations**
   - Find spacing values not on the defined grid
   - Example: 17px when the base is 4px

2. **Depth violations**
   - Borders-only system: flag shadows
   - Subtle system: flag layered shadows
   - Allow ring shadows such as `0 0 0 1px`

3. **Color violations**
   - If a palette is defined, flag colors outside it
   - Allow neutral semantic grays if they fit the saved system

4. **Pattern drift**
   - Find buttons that diverge from the saved button pattern
   - Find cards that diverge from the saved card pattern

Report in this shape:

```text
Audit Results: src/components/

Violations:
  Button.tsx:12 - Height 38px (pattern: 36px)
  Card.tsx:8 - Shadow used (system: borders-only)
  Input.tsx:20 - Spacing 14px (grid: 4px, nearest: 12px or 16px)

Suggestions:
  - Update Button height to match pattern
  - Replace shadow with border
  - Adjust spacing to grid
```

If no `system.md` exists, say so and redirect to a Codex-native next step:

```text
No design system to audit against.

Create a system first:
1. Establish the system while building UI
2. Perform the extract workflow to derive a system from existing code
```

## Implementation

1. Check for `system.md`
2. Parse the saved rules
3. Read target files (`tsx`, `jsx`, `css`, `scss`, and relevant docs)
4. Compare them against the saved system
5. Report violations with concrete suggestions
