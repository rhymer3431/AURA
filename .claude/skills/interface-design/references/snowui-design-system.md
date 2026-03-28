# AURA Console Design System Reference

Compatibility note:

- The filename is kept for compatibility with the existing repo-local skill wiring.
- The content below defines the local AURA dashboard system target.

Source reference:

- Approved AURA Pipeline Overview screenshot supplied with the task
- Supporting product context from the existing AURA dashboard

Use this file when the task needs the broader token, variable, and component-system rules behind the local AURA dashboard target. Keep using `snowui-dashboard.md` for layout and composition guidance.

## Skill Purpose

This reference exists to keep dashboard UI work aligned with the approved AURA operations-board style instead of drifting back into a Claude-like workspace shell.

- Keep the system small, variable-driven, and reusable.
- Preserve distinct dashboard region families:
  - canvas
  - shell panel
  - inner module
  - KPI surface families
  - semantic state accents
- Do not flatten AURA's runtime-console identity into a monochrome settings workspace.

## Core Philosophy

### System First, But Not Flat

The design system should absorb recurring structure without erasing the page's hierarchy.

- Reuse repeated surfaces, radii, spacings, and state treatments.
- Allow a small number of dashboard-specific surface families when they repeat across the product.
- Do not add a new token for a one-off whim.
- Do not force every region onto the same visual plane just to make the system look "clean."

### Variables First

Variables remain the primary control surface for UI.

- Prefer token or variable references over direct values.
- Treat changes to shared variables as high-impact because they affect many screens at once.
- Minimize raw `px`, hex, direct radius, or ad hoc shadow recipes.
- Favor semantic tokens that survive theme switching instead of mode-specific hardcoded values.
- When a dashboard family truly needs a dedicated fill, encode it as a named semantic family instead of sprinkling ad hoc pastels through components.

### Region-Aware System

The system should be easy to learn and still support the approved board composition.

- Recompose existing primitives before minting a new primitive.
- When three similar patterns appear, unify them into a base plus variants or a saved pattern.
- Do not allow business-level components to drift into their own styling language.
- Avoid page-by-page styling logic when a reusable token or variant already solves the problem.
- Let navigation rail, KPI tiles, shell panels, and white modules feel intentionally different when their jobs differ.

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

Separate theme-driven meaning from fixed visual assets.

- Prefer semantic color tokens over raw palette references.
- Separate canvas, surface, text, border, accent, and state responsibilities.
- Expose family color through semantic groups instead of scattering brand hex values through components.
- Keep the state palette limited and consistent: success, warning, error, info.
- Treat light/dark compatibility as part of the token design, not as a later patch.

Recommended semantic structure:

```text
bg.canvas
bg.shell
bg.module
bg.kpi.planner
bg.kpi.perception
bg.kpi.services
bg.kpi.runtime
text.primary
text.secondary
text.muted
border.default
border.subtle
accent.action
accent.live
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
- Prefer spacing, panel grouping, and border logic before adding stronger effects.
- Do not create a different shadow recipe for every page.
- Use blur or glass sparingly and intentionally.
- Keep the effect inventory small, ideally under `8`.

Recommended effect set:

```text
shadow-shell
shadow-module
shadow-float
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

The system should allow flexible product layouts, but the default dashboard board composition matters.

- Make `1`, `2`, and `3` column layouts easy to compose.
- Prefer card and grid primitives that can be rearranged without restyling the whole page.
- Preserve room for density changes, panel resizing, and reorderable work surfaces where the product benefits.
- Keep layout freedom high, but keep spacing, radius, type scale, and token usage system-bound.
- Do not confuse layout freedom with permission to introduce arbitrary values or component-local styling languages.
- For the core dashboard, expect:
  - fixed left navigation rail
  - KPI ribbon near the top
  - dominant center workbench
  - stacked right support column

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
- Preserve the visual hierarchy from the approved screenshot before exploring new compositions.

### When Refactoring

- If similar UI appears three or more times, absorb it into a base, variant, or saved system pattern.
- Remove duplicated business-layer styling when a core component already exists.
- Replace ad hoc CSS with token-based styling where possible.
- Prefer design-token aliases over direct values.
- If a refactor makes the dashboard flatter, grayer, or more workspace-like than the approved target, treat that as a regression.

### Prohibited Patterns

- Do not create page-specific versions of generic components by default.
- Do not duplicate similar button, input, or card families.
- Do not repeatedly hardcode hex colors, pixel values, or corner radii.
- Do not decide styling from light mode alone.

## Translation Into AURA

- Use the local AURA reference system to tighten reuse, not to flatten AURA into a generic admin template.
- Preserve semantic runtime colors for safety, warnings, service health, and telemetry meaning.
- Prefer shared tokens for shell, input, text hierarchy, and status treatments before adding component-local rules.
- Keep rare AURA-specific exceptions local unless they become repeatable.
- When a system decision conflicts with AURA's saved project memory or runtime meaning, prefer `dashboard/.interface-design/system.md` and `snowui-dashboard.md` for the final repo-specific choice.

## Practical Checks

- If a new color, radius, or effect appears once, question whether it belongs in the system.
- If a visual difference can be expressed by a variant, avoid adding a new component family.
- If the request is bespoke but low-repeat, keep it in the allowed 10% exception zone.
- If a light-theme choice fails in dark mode, the token design is incomplete.
