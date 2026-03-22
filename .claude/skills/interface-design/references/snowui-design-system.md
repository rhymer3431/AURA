# SnowUI Design System Reference

Source Figma:

- File: `SnowUI Design System (Community)`
- Canvas: `🟡 Design system`
- URL: `https://www.figma.com/design/WaLIQwZJ1YeuNccnqqiiel/SnowUI-Design-System--Community-?node-id=60755-3905`

Use this file when the task needs SnowUI's broader token, variable, and component-system rules. Keep using `snowui-dashboard.md` for layout and framing guidance.

## Skill Purpose

This reference exists to make dashboard UI work follow SnowUI as an operating system, not as a screenshot to imitate.

- Use SnowUI to keep the design system small, variable-driven, theme-aware, and reusable.
- Use SnowUI to improve token discipline, component taxonomy, and reuse inside `dashboard/`.
- Do not use SnowUI to erase AURA's runtime-console identity, telemetry semantics, or operator density.

## Core Philosophy

### 90% Principle

SnowUI does not try to systemize every possible edge case.

- Keep roughly 90% of recurring UI inside the design system.
- Allow the remaining 10% to stay local when it is rare, page-specific, or not worth promoting.
- Do not add a new token or component just because a single screen wants a slightly different value.
- When a rare need does not repeat, solve it locally and leave the core system lean.

### Variables First

SnowUI treats variables as the primary control surface for UI.

- Prefer token or variable references over direct values.
- Treat changes to shared variables as high-impact because they affect many screens at once.
- Minimize raw `px`, hex, direct radius, or ad hoc shadow recipes.
- Favor semantic tokens that survive theme switching instead of mode-specific hardcoded values.
- Assume light and dark mode both matter unless the user explicitly says otherwise.

### Lean System

SnowUI values a small system that is easy to learn, reuse, and ship with.

- Recompose existing primitives before minting a new primitive.
- When three similar patterns appear, unify them into a base plus variants or a saved pattern.
- Do not allow business-level components to drift into their own styling language.
- Avoid page-by-page styling logic when a reusable token or variant already solves the problem.

## Token Rules

### Spacing, Icon Size, Radius

- Keep spacing on a `4px` grid.
- Keep spacing, icon-size, and corner-radius scales compact, ideally under `16` steps each.
- Avoid one-off values such as `13px`, `22px`, or `7px` radius unless a truly local exception is justified.
- Promote a new value only when it is clearly repeatable across the product.

Recommended scale:

```text
space-0  = 0
space-1  = 4
space-2  = 8
space-3  = 12
space-4  = 16
space-5  = 20
space-6  = 24
space-7  = 28
space-8  = 32
space-10 = 40
space-12 = 48
space-20 = 80

radius-sm  = 8
radius-md  = 12
radius-lg  = 16
radius-xl  = 20
radius-2xl = 24

icon-sm = 16
icon-md = 20
icon-lg = 24
icon-xl = 32
```

### Colors

SnowUI separates theme-driven meaning from fixed visual assets.

- Prefer semantic color tokens over raw palette references.
- Separate canvas, surface, text, border, accent, and state responsibilities.
- Expose brand through `accent-*` semantics instead of scattering brand hex values through components.
- Keep the state palette limited and consistent: success, warning, error, info.
- Treat light/dark compatibility as part of the token design, not as a later patch.

Recommended semantic structure:

```text
bg.canvas
bg.surface
bg.elevated
text.primary
text.secondary
text.muted
border.default
border.subtle
accent.primary
accent.secondary
state.success
state.warning
state.error
state.info
```

### Text Styles

- Keep the type system restrained and reusable.
- Favor heading, body, label, and caption tiers over many near-duplicate styles.
- Do not solve hierarchy by inventing endless font-size and weight combinations.
- Use spacing, weight, and contrast before adding new text tiers.
- Keep the number of text styles low, ideally under `16`.

Recommended style ladder:

```text
display-lg
heading-xl
heading-lg
heading-md
body-lg
body-md
body-sm
label-md
label-sm
caption
```

### Effects

- Keep shadows and blur recipes to a small reusable set.
- Prefer spacing, border logic, and contrast before adding stronger effects.
- Do not create a different shadow recipe for every page.
- Use glass or blur treatments sparingly and intentionally.
- Keep the effect inventory small, ideally under `8`.

Recommended effect set:

```text
shadow-sm
shadow-md
shadow-lg
blur-bg-sm
blur-bg-md
```

## Component Rules

### Base First

Start with reusable building blocks before creating page-specific pieces.

Preferred base layer:

```text
ButtonBase
InputBase
SelectBase
CardBase
ListItemBase
TagBase or BadgeBase
AvatarBase
IconBase
ModalBase
TabBase
```

Rules:

- Do not start from a page-specific component if the pattern is a generic control.
- If the UI meaning matches an existing base, extend that base instead of cloning it.
- Prefer base-level composition and slotting before inventing a parallel control family.

### Variant-Driven Expansion

- Expand via `size`, `tone`, `variant`, and `state`.
- If the component behavior is the same and the layout only shifts slightly, use variants or slots.
- Use wrappers for business-specific composition, but keep the base API recognizable.
- Common examples such as icon-only, leading-icon, trailing-icon, destructive, soft, and ghost should stay inside one family where possible.

### Core vs Business Separation

- Core components are reusable and presentation-focused.
- Business components may include domain logic, data binding, and workflow-specific composition.
- Business components should consume core tokens and core components instead of creating a second styling system.

Example split:

```text
ui/
  button/
  input/
  card/
  tabs/

features/
  billing/
    plan-card/
  analytics/
    metric-card/
  users/
    user-filter-panel/
```

## Layout Rules

SnowUI encourages flexible product layouts, not one rigid template.

- Make `1`, `2`, and `3` column layouts easy to compose.
- Prefer card and grid primitives that can be rearranged without restyling the whole page.
- Preserve room for density changes, panel resizing, and reorderable work surfaces where the product benefits.
- Keep layout freedom high, but keep spacing, radius, type scale, and token usage system-bound.
- Do not confuse layout freedom with permission to introduce arbitrary values or component-local styling languages.

## Resource Rules

Resources should stay replaceable and loosely coupled to the UI system.

- Treat icons, avatars, logos, emoji, and illustrations as assets, not as the system itself.
- Keep the icon library loosely coupled to the UI layer.
- Align icon size, stroke, and weight to the token scale.
- Do not mix unrelated SVG styles casually inside the same product surface.
- Treat illustrations as supporting assets, not the main structural voice of the screen.
- Let avatar and logo assets carry content identity; let the system tokens own the container, spacing, and framing.

## Working Guidance

### When Generating UI

- Reuse existing tokens and existing components first.
- Add a new token or component only when repeated use is likely.
- Keep spacing, radius, icon size, and typography on the system scale.
- Prefer semantic colors and theme-ready variables.
- Minimize hardcoded colors, spacing, radii, and effects.
- Keep dark mode compatibility intact.
- Allow free layouts, but keep style rules controlled.

### When Refactoring

- If similar UI appears three or more times, absorb it into a base, variant, or saved system pattern.
- Remove duplicated business-layer styling when a core component already exists.
- Replace ad hoc CSS with token-based styling where possible.
- Prefer design-token aliases over direct values.

### Prohibited Patterns

- Do not create page-specific versions of generic components by default.
- Do not duplicate similar button, input, or card families.
- Do not repeatedly hardcode hex colors, pixel values, or corner radii.
- Do not decide styling from light mode alone.

## Translation Into AURA

- Use SnowUI to tighten reuse, not to flatten AURA into a generic admin template.
- Preserve semantic runtime colors for safety, warnings, service health, and telemetry meaning.
- Prefer shared tokens for shell, input, text hierarchy, and status treatments before adding component-local rules.
- Keep rare AURA-specific exceptions local unless they become repeatable.
- When a system decision conflicts with AURA's saved project memory or runtime meaning, prefer `dashboard/.interface-design/system.md` and `snowui-dashboard.md` for the final repo-specific choice.

## Practical Checks

- If a new color, radius, or effect appears once, question whether it belongs in the system.
- If a visual difference can be expressed by a variant, avoid adding a new component family.
- If the request is bespoke but low-repeat, keep it in the allowed 10% exception zone.
- If a light-theme choice fails in dark mode, the token design is incomplete.
