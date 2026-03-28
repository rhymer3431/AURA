# AURA Dashboard Target Reference

Compatibility note:

- The filename is kept for compatibility with the existing repo-local skill wiring.
- The content below is the actual AURA target dashboard reference.

Source reference:

- Approved AURA Pipeline Overview screenshot supplied with the task
- Use this as the primary visual anchor for dashboard composition and tone

Use this file as the primary visual reference. Prefer these extracted rules over older quiet-workspace patterns.

## What The Target Is Doing

- Warm-white canvas with a dedicated left navigation rail
- Small breadcrumb strip above the page title, not a chat-style workspace header
- Four wide KPI tiles with distinct pastel fills for different telemetry families
- One dominant live-robot workbench panel occupying the main center column
- One stacked support column on the right for process, sensor, and module health cards
- Large rounded outer panels with lighter inner white modules
- Green status dots, red live badge, and restrained operational accents that stay readable
- Clear region ownership: sidebar, KPI ribbon, live stage, right support column, lower diagnostic panels

## Practical Values To Borrow

- Base spacing: `4px`
- Common control heights: `32px`, `40px`, `48px`
- Common radii:
  - Micro chip: `8px`
  - Inline control: `14px`
  - KPI / list card: `20px` to `24px`
  - Major panel shell: `28px` to `32px`
- Border treatment:
  - Panel border: `1px solid rgba(15, 23, 42, 0.05)`
  - Inner module border: `1px solid rgba(15, 23, 42, 0.06)`
  - Supporting dividers can stay subtle, but major panel grouping should be readable without squinting
- Typography:
  - UI body: `13px` and `14px`
  - Section title: `22px` to `28px`
  - KPI value: `40px` to `48px`
  - Voice: operational, precise, approachable

## Header Patterns

- Put breadcrumb and search/actions on the same line
- Keep the header shallow, but let the page title and section heading breathe
- Use a simple strip with icons and breadcrumb, not a floating utility shell
- Do not turn the top bar into the main visual event; the KPI row and work panels should carry the page

## Sidebar Patterns

- Default width sits around `220px` to `236px`
- Collapsed width can go down to `80px`
- Brand sits at top, app shortcut or utility stays anchored at bottom
- Organize items into labeled groups
- Active item is a pale highlighted row with comfortable padding
- Section labels should be visible and warm-gray, not hidden or ultra-faint
- The rail can own its own background tone; it does not need to disappear into the canvas

## KPI Tile Patterns

- Use four-across tiles when the width allows, with each tile holding:
  - family label
  - large primary number
  - small delta or trend at the far edge
- Encode family by pastel tile background, not just by micro-accent dots
- Keep tile chrome minimal; the shape and fill should do most of the work
- Avoid tiny monochrome stat pills that undersell the importance of top-line telemetry

## Main Workbench Patterns

- The live-view panel should be the visual anchor of the page
- Use an outer fog panel plus an inner white media stage
- Keep the panel header actionable:
  - section icon
  - title
  - live badge
  - overlay toggle
  - expand or utility affordance
- Let the camera or visualization surface be large enough to feel operational, not decorative
- Add a compact inference strip or metadata footer inside the panel instead of pushing all detail elsewhere

## Right Column Patterns

- Right-column cards should be stacked inside large rounded shells
- Each shell may contain several smaller white modules
- Good content:
  - process composition
  - sensor input state
  - health summaries
  - pipeline or inference breakdowns
- Use white inner cards plus green or semantic state markers to keep the stack readable

## What To Preserve In AURA

- Runtime diagnostics and telemetry remain primary use cases
- Semantic colors still matter for safety, warnings, and service health
- Data-heavy regions can stay denser than the hero surfaces
- Monospace remains appropriate for logs, metrics, IDs, and machine state
- The UI should feel like a bright robotics operations board, not a Claude-like workspace shell and not a generic SaaS admin

## Translation Rules For This Repo

- Prefer readable panel clusters over whisper-light layering
- Reserve accent color for runtime meaning, but allow pastel family blocks for KPI grouping
- Use white modules inside larger fog shells to create depth
- Let side navigation, KPI tiles, and the live workbench be visually distinct regions
- When in doubt, solve hierarchy with panel scale, region grouping, and color-family assignment before resorting to tiny chips or hidden dividers

## Layout Translation For This Repo

- Make the default page read in this order:
  - left navigation rail
  - breadcrumb/title strip
  - KPI row
  - large center workbench plus right support column
  - lower diagnostic panels
- Prefer `2+1` compositions in the main body where the center workbench dominates and the right column supports it
- Use full-width panel shells for major sections instead of many unrelated small cards
- Keep the layout flexible, but preserve the first-read hierarchy from the approved screenshot
- When a layout feels too flat, increase contrast between regions instead of muting everything into one workspace plane
