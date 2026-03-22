# SnowUI Dashboard Reference

Source Figma:

- File: `SnowUI Design System (Community)`
- Canvas: `🔷 Dashboard`
- URL: `https://www.figma.com/design/jWo6r4s6gxj5X1ffae11LA/SnowUI-Design-System--Community-?node-id=73957-26057`

Use this file as a distilled visual reference. Prefer these extracted rules over re-reading the full Figma unless the task needs a specific component variant.

## What SnowUI Is Doing

- Bright neutral canvas with near-white working surfaces
- Hairline structure using `0.5px` borders and very low-contrast black alpha
- Small, quiet type scale centered around `12px` and `14px`
- Compact header with breadcrumb, search capsule, and lightweight icon controls
- Sidebar navigation built from soft active pills instead of heavy blocks
- Rounded controls with `12px` to `16px` radii
- Primary emphasis uses black fill, not saturated brand gradients
- Icons are simple, thin, and frequent but visually quiet

## Practical Values To Borrow

- Base spacing: `4px`
- Common control heights: `24px`, `36px`
- Common radii:
  - Micro text/key chip: `4px`
  - Small text container: `8px`
  - Control pill: `12px`
  - Search / active nav / grouped control: `16px`
- Border treatment:
  - Dividers: `0.5px solid rgba(0, 0, 0, 0.10)`
  - Quiet fills: `rgba(0, 0, 0, 0.04)`
  - Muted text: `rgba(0, 0, 0, 0.40)` and `rgba(0, 0, 0, 0.20)`
- Typography:
  - UI body: `12px` and `14px`
  - Default weight: `400`
  - Voice: restrained, not editorial

## Header Patterns

- Put breadcrumb and search/actions on the same line
- Keep the header shallow and compact
- Use subtle bottom separators, not card-like header containers
- Search is a soft pill with an embedded shortcut chip
- Utility icons should read as lightweight controls, not feature cards

## Sidebar Patterns

- Default width sits around `212px` to `220px`
- Collapsed width can go down to `80px`
- Brand sits at top, user/account anchors at bottom
- Organize items into labeled groups
- Active item is a filled neutral pill
- Nested items should indent softly rather than using hard boxes

## What To Preserve In AURA

SnowUI is the reference, not the product identity. Keep these AURA-specific constraints:

- Runtime diagnostics and telemetry remain primary use cases
- Semantic colors still matter for safety, warnings, and service health
- Data-heavy regions can stay denser than SnowUI examples
- Monospace remains appropriate for logs, metrics, IDs, and machine state
- The UI should feel like a calm operator console, not a generic SaaS admin

## Translation Rules For This Repo

- Prefer SnowUI's quiet neutral shells for navigation, search, and framing
- Reserve accent color for runtime meaning, not blanket decoration
- Use black or graphite for primary actions when emphasis is needed
- Reduce visual weight before adding new decoration
- Prefer pills, dividers, and soft inset fills over card stacks and loud badges
- When in doubt, make hierarchy clearer by spacing and border logic first, color second
