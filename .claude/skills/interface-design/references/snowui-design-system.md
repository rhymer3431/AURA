# SnowUI Design System Reference

Source Figma:

- File: `SnowUI Design System (Community)`
- Canvas: `🟡 Design system`
- URL: `https://www.figma.com/design/WaLIQwZJ1YeuNccnqqiiel/SnowUI-Design-System--Community-?node-id=60755-3905`

Use this file when the task needs SnowUI's broader token, variable, and component-system rules. Keep using `snowui-dashboard.md` for layout and framing guidance.

## What This Board Adds

- System-wide token discipline instead of a single dashboard composition
- Variable categories for colors, spacing, size, radius, font, font weight, paragraph spacing, and visibility toggles
- Explicit guidance on keeping the system small and widely reused
- A component taxonomy that separates base, common, chart, mobile, and page-specific pieces

## Core Rules Extracted

### 90% Principle

- SnowUI expects about 90% of a product to come from the design system
- The remaining 10% can break system rules when a page-specific need justifies it
- Use this to avoid bloating the system with one-off tokens or components

### Variable Discipline

- Prefer editing variables before introducing raw values in components
- Keep the number of variables small
- Treat changes to variables as high-impact because they affect many components and pages
- Main variable groups shown in the board:
  - Colors
  - Corner Radius
  - Font
  - Font Weight
  - Paragraph Spacing
  - Show or Hide
  - Size
  - Spacing

### Spacing, Size, Corner Radius

- Keep values on multiples of `4`
- Try to keep the number of spacing, icon-size, and corner-radius values below `16`
- Prefer a compact reusable range over a long tail of one-off values
- The board highlights smaller operational values like `4`, `8`, `12`, `16`, `20`, `24`, `28`, `32`, `40`, `48`, `80`

### Colors

- Keep the number of colors as small as possible
- Prefer one primary family plus neutral black/white scales and a restrained secondary set
- The board organizes color usage across light and dark modes instead of redefining color meaning per screen
- Color groups shown in the board:
  - Primary
  - Black
  - White
  - Secondary semantic hues: purple, indigo, blue, cyan, mint, green, yellow, orange, red
  - Background
  - Surface

### Text Styles

- Keep the number of text styles below `20`
- Preferred font is `Inter`
- Other recommended fonts shown: `SF Pro`, `Roboto`, `Averta`
- The board emphasizes a restrained scale built from repeated `Regular` and `Semibold` weights
- Visible sizes include `12`, `14`, `16`, `18`, `24`, `32`, `48`, `64`

### Components

- Keep the number of components as small as possible
- Prefer extending base components with variants or replaceable internals instead of creating near-duplicate components
- The board's component buckets are:
  - Base Components
  - Common Components
  - Chart Components
  - Mobile Components
  - Page Components
- Page components may exist, but they should still conform to the system rather than redefine it

## How To Translate This Into AURA

- Use SnowUI's system board to tighten token reuse, not to erase AURA's runtime-console identity
- Keep AURA's semantic runtime colors meaningful and limited
- Prefer shared tokens for shell, input, text hierarchy, and status treatments before adding component-local values
- When a new dashboard pattern would be used rarely, keep it local rather than promoting it into `dashboard/.interface-design/system.md`
- When a new pattern crosses repeated use, save it as a reusable rule instead of duplicating ad hoc styling

## Practical Checks For Skill Users

- If a new color or radius appears only once, question whether it should exist
- If a component variant can be achieved by slot replacement or size variables, avoid creating a new component
- If a page asks for bespoke visuals, keep them inside the allowed 10% rather than expanding the core system blindly
- If a dashboard decision conflicts with this reference, prefer the project memory and `snowui-dashboard.md` for AURA-specific direction
