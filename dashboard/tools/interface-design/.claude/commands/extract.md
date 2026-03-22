---
name: interface-design:extract
description: Codex workflow reference for extracting reusable interface patterns from existing code.
---

# interface-design extract

Use this workflow to derive a saved interface system from existing UI code.

## Recommended Targets

Extract from the user-specified directory when one is given.

Otherwise, use the common UI paths for this repository:

- `dashboard/src`
- `dashboard/src/styles`
- `dashboard/guidelines/Guidelines.md`

## What to Extract

Scan UI files (`tsx`, `jsx`, `vue`, `svelte`, CSS) for:

1. **Repeated spacing values**
   ```text
   Found: 4px (12x), 8px (23x), 12px (18x), 16px (31x), 24px (8x)
   Suggests: Base 4px, Scale: 4, 8, 12, 16, 24
   ```

2. **Repeated radius values**
   ```text
   Found: 6px (28x), 8px (5x)
   Suggests: Radius scale: 6px, 8px
   ```

3. **Button patterns**
   ```text
   Found 8 buttons:
   - Height: 36px (7/8), 40px (1/8)
   - Padding: 12px 16px (6/8), 16px (2/8)
   Suggests: Button pattern: 36px h, 12px 16px padding
   ```

4. **Card patterns**
   ```text
   Found 12 cards:
   - Border: 1px solid (10/12), none (2/12)
   - Padding: 16px (9/12), 20px (3/12)
   Suggests: Card pattern: 1px border, 16px padding
   ```

5. **Depth strategy**
   ```text
   box-shadow found: 2x
   border found: 34x
   Suggests: Borders-only depth
   ```

Then present the extracted system in a customizable summary:

```text
Extracted patterns:

Spacing:
  Base: 4px
  Scale: 4, 8, 12, 16, 24, 32

Depth: Borders-only (34 borders, 2 shadows)

Patterns:
  Button: 36px h, 12px 16px pad, 6px radius
  Card: 1px border, 16px pad
```

If the user asked to persist the result, or the task explicitly includes creating the saved system, write `.interface-design/system.md` after confirming any high-impact adjustments.

## Implementation

1. Glob the relevant UI files
2. Parse repeated values and component patterns
3. Identify the dominant system by frequency
4. Summarize the proposed system
5. Persist `system.md` only when the task or user intent requires it
