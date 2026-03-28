# Core Craft Principles

These apply regardless of design direction. This is the quality floor.

---

## Surface & Token Architecture

Professional interfaces don't pick colors randomly — they build systems. Understanding this architecture is the difference between "looks okay" and "feels like a real product."

### The Primitive Foundation

Every color in your interface should trace back to a small set of primitives:

- **Foreground** — text colors (primary, secondary, muted)
- **Background** — surface colors (base, elevated, overlay)
- **Border** — edge colors (default, subtle, strong)
- **Brand** — your primary accent
- **Semantic** — functional colors (destructive, warning, success)

Don't invent new colors. Map everything to these primitives.

### Surface Elevation Hierarchy

Surfaces still stack, but operator dashboards often need stronger regional grouping than generic workspace apps. A dropdown sits above a module which sits inside a shell which sits on the page.

```
Level 0: Base background (the app canvas)
Level 1: Cards, panels (same visual plane as base)
Level 2: Dropdowns, popovers (floating above)
Level 3: Nested dropdowns, stacked overlays
Level 4: Highest elevation (rare)
```

In dark mode, higher elevation = slightly lighter. In light mode, higher elevation can use lighter fill, border separation, or soft shadow. The principle: **elevated surfaces need clear visual distinction from what's beneath them.**

### The Readable Separation Principle

This is where many dashboards fail. The AI default is either:

- everything on one quiet plane, or
- every card loudly outlined

Good operator UI lands in the middle: region ownership is immediately readable, but the page is still refined.

**For surfaces:** Elevation differences should be visible enough to separate navigation, shell panels, and inner modules. In bright dashboards, that may mean warm canvas + fog shell + white module instead of one barely changing neutral.

**For borders:** Borders should define regions without turning into wireframes. Use low opacity for inner modules and slightly stronger definition for major shells when needed.

**The test:** Glance at your interface. You should immediately perceive navigation, KPI summary, main workbench, and support modules. If everything merges into one workspace plane, increase separation. If borders are all you see, reduce them.

**Common AI mistakes to avoid:**
- Borders that are too visible (1px solid gray instead of subtle rgba)
- Surface jumps that are arbitrary instead of region-driven
- Using pastel colors randomly instead of assigning them to repeated families
- Hiding major region changes behind overly subtle shells

### Text Hierarchy via Tokens

Don't just have "text" and "gray text." Build four levels:

- **Primary** — default text, highest contrast
- **Secondary** — supporting text, slightly muted
- **Tertiary** — metadata, timestamps, less important
- **Muted** — disabled, placeholder, lowest contrast

Use all four consistently. If you're only using two, your hierarchy is too flat.

### Border Progression

Borders aren't binary. Build a scale:

- **Default** — standard borders
- **Subtle/Muted** — softer separation
- **Strong** — emphasis, hover states
- **Stronger** — maximum emphasis, focus rings

Match border intensity to the importance of the boundary.

### Dedicated Control Tokens

Form controls (inputs, checkboxes, selects) have specific needs. Don't just reuse surface tokens — create dedicated ones:

- **Control background** — often different from surface backgrounds
- **Control border** — needs to feel interactive
- **Control focus** — clear focus indication

This separation lets you tune controls independently from layout surfaces.

### Context-Aware Bases

Different areas of your app might need different base surfaces:

- **Marketing pages** — might use darker/richer backgrounds
- **Dashboard/app** — might use neutral working backgrounds
- **Sidebar** — may intentionally differ from the main canvas when stable orientation matters

The surface hierarchy works the same way — it just starts from a different base.

### Alternative Backgrounds for Depth

Beyond shadows, use contrasting backgrounds to create depth. An "alternative" or "inset" background makes content feel recessed. Useful for:

- Empty states in data grids
- Code blocks
- Inset panels
- Visual grouping without borders

---

## Spacing System

Pick a base unit (4px and 8px are common) and use multiples throughout. The specific number matters less than consistency — every spacing value should be explainable as "X times the base unit."

Build a scale for different contexts:
- Micro spacing (icon gaps, tight element pairs)
- Component spacing (within buttons, inputs, cards)
- Section spacing (between related groups)
- Major separation (between distinct sections)

## Symmetrical Padding

TLBR must match. If top padding is 16px, left/bottom/right must also be 16px. Exception: when content naturally creates visual balance.

```css
/* Good */
padding: 16px;
padding: 12px 16px; /* Only when horizontal needs more room */

/* Bad */
padding: 24px 16px 12px 16px;
```

## Border Radius Consistency

Sharper corners feel technical, rounder corners feel friendly. Pick a scale that fits your product's personality and use it consistently.

The key is having a system: small radius for inputs and buttons, medium for cards, large for modals or containers. Don't mix sharp and soft randomly — inconsistent radius is as jarring as inconsistent spacing.

## Depth & Elevation Strategy

Match your depth approach to your design direction. Choose ONE and commit:

**Borders-only (flat)** — Clean, technical, dense. Best when the product truly benefits from a tight utility feel.

**Subtle single shadows** — Soft lift without complexity. A simple `0 1px 3px rgba(0,0,0,0.08)` can be enough. Works for approachable products that want gentle depth.

**Layered shadows** — Rich, premium, dimensional. Multiple shadow layers create realistic depth. Stripe and Mercury use this approach. Best for cards that need to feel like physical objects.

**Surface color shifts** — Background families establish hierarchy without dramatic effects. A fog shell around white modules often works better for operations dashboards than a page full of identical cards.

```css
/* Borders-only approach */
--border: rgba(0, 0, 0, 0.08);
--border-subtle: rgba(0, 0, 0, 0.05);
border: 0.5px solid var(--border);

/* Single shadow approach */
--shadow: 0 1px 3px rgba(0, 0, 0, 0.08);

/* Layered shadow approach */
--shadow-layered:
  0 0 0 0.5px rgba(0, 0, 0, 0.05),
  0 1px 2px rgba(0, 0, 0, 0.04),
  0 2px 4px rgba(0, 0, 0, 0.03),
  0 4px 8px rgba(0, 0, 0, 0.02);
```

## Card Layouts

Monotonous card layouts are lazy design. A KPI tile should not look like a process list card should not look like a live-view workstation panel.

Design each card's internal structure for its specific content — but keep the surface treatment consistent: same border weight, shadow depth, corner radius, padding scale, typography.

## Isolated Controls

UI controls deserve container treatment. Date pickers, filters, dropdowns — these should feel like crafted objects.

**Never use native form elements for styled UI.** Native `<select>`, `<input type="date">`, and similar elements render OS-native dropdowns that cannot be styled. Build custom components instead:

- Custom select: trigger button + positioned dropdown menu
- Custom date picker: input + calendar popover
- Custom checkbox/radio: styled div with state management

Custom select triggers must use `display: inline-flex` with `white-space: nowrap` to keep text and chevron icons on the same row.

## Typography Hierarchy

Build distinct levels that are visually distinguishable at a glance:

- **Headlines** — heavier weight, tighter letter-spacing for presence
- **Body** — comfortable weight for readability
- **Labels/UI** — medium weight, works at smaller sizes
- **Data** — often monospace, needs `tabular-nums` for alignment

Don't rely on size alone. Combine size, weight, and letter-spacing to create clear hierarchy. If you squint and can't tell headline from body, the hierarchy is too weak.

## Monospace for Data

Numbers, IDs, codes, timestamps belong in monospace. Use `tabular-nums` for columnar alignment. Mono signals "this is data."

## Iconography

Icons clarify, not decorate — if removing an icon loses no meaning, remove it. Choose a consistent icon set and stick with it throughout the product.

Give standalone icons presence with subtle background containers. Icons next to text should align optically, not mathematically.

## Animation

Keep it fast and functional. Micro-interactions (hover, focus) should feel instant — around 150ms. Larger transitions (modals, panels) can be slightly longer — 200-250ms.

Use smooth deceleration easing (ease-out variants). Avoid spring/bounce effects in professional interfaces — they feel playful, not serious.

## Contrast Hierarchy

Build a four-level system: foreground (primary) → secondary → muted → faint. Use all four consistently.

## Color Carries Meaning

Gray builds structure. Color communicates — status, action, emphasis, identity. Unmotivated color is noise. Color that reinforces the product's world is character.

## Navigation Context

Screens need grounding. A data table floating in space feels like a component demo, not a product. Consider including:

- **Navigation** — sidebar or top nav showing where you are in the app
- **Location indicator** — breadcrumbs, page title, or active nav state
- **User context** — who's logged in, what workspace/org

When building sidebars, choose deliberately:

- same background as the canvas when the UI should feel extremely unified
- a dedicated rail background when stable orientation matters more than minimalist blending

For operations dashboards, a dedicated rail is often the better default.

## Dark Mode

Dark interfaces have different needs:

**Borders over shadows** — Shadows are less visible on dark backgrounds. Lean more on borders for definition.

**Adjust semantic colors** — Status colors (success, warning, error) often need to be slightly desaturated for dark backgrounds.

**Same structure, different values** — The hierarchy system still applies, just with inverted values.
