# AURA Dashboard Interface System

## Intent
- Human: 로봇 런타임을 모니터링하고 장애 징후를 빠르게 판독해야 하는 운영자/개발자
- Primary job: 현재 세션 상태, gateway/telemetry, planner, memory, diagnostics를 한 화면에서 빠르게 스캔
- Feel: bright robotics operations board, readable at a glance, large-paneled, calm but not muted, technical without looking like a chat workspace

## Domain
- control room
- runtime telemetry
- field diagnostics
- sensor fusion
- vision workstation
- process lane
- sensor rack
- orchestration console
- safety checkpoint

## Color World
- warm white canvas
- fog-gray panel shells
- white instrument modules
- planner ice-blue KPI tiles
- perception lilac KPI tiles
- runtime mint state lights
- live coral badges
- slate ink for metrics and labels

## Signature
- a top KPI ribbon feeding into one dominant live-view workbench, with a stacked right support column and a clearly separated left navigation rail

## Defaults Replaced
- Quiet settings workspace shell -> explicit operations-board region hierarchy
- Same-plane monochrome cards -> layered fog shells with white inner modules
- Tiny neutral stat pills -> wide pastel KPI tiles with large black values
- Claude-like sidebar/header restraint -> visible navigation rail, breadcrumb strip, and strong workbench focus
- Generic body typography -> `Pretendard` for UI with `IBM Plex Mono` reserved for telemetry and machine-readable data

## Tokens
- Base spacing: 4px
- Radius scale:
  - Micro: 8px
  - Small: 14px
  - Control: 16px
  - Tile: 24px
  - Panel: 30px
- Depth:
  - Panel-cluster first
  - Low-contrast borders plus soft shadows for major shells
  - Inner white modules can use lighter lift than the outer fog shell
  - Avoid glassy or ultra-flat shells that erase region ownership
- Typography:
  - UI: Pretendard
  - Data: IBM Plex Mono
  - Core sizes: 13px, 14px, 16px, 24px, 44px
  - Default weight: 500 for labels, 700 for KPI values and section titles
- Border language:
  - Divider: rgba(15, 23, 42, 0.08)
  - Panel border: rgba(15, 23, 42, 0.05)
  - Module border: rgba(15, 23, 42, 0.06)
  - Semantic emphasis only when state matters
- Surface language:
  - Canvas: warm white
  - Shells: fog-gray rounded work surfaces
  - Modules: white cards nested inside shells
  - Utility fills: pale neutral or semantic pastel, never dark chat-like capsules
- Color usage:
  - Use pastel families for KPI grouping when those families repeat across the page
  - Use green states, red live badges, and amber warnings as operational signals
  - Black/slate text should stay crisp inside pastel tiles and white modules
  - Decorative gradients are unnecessary; the product should read through region color and surface scale

## Reusable Patterns

### Shell Navigation
- Sidebar width: 220px to 236px
- Collapsed sidebar width: 80px
- Item height: 40px to 44px
- Active state: pale highlighted row with comfortable horizontal padding
- Section labels: 12px, visible muted slate, not ultra-faint
- Use grouping: brand top, section labels and nav in the middle, utility anchor at the bottom
- The rail can own a distinct background tone from the main canvas

### Top Bar
- Height target: 56px to 64px
- Breadcrumb, search, and action icons live on one horizontal band
- Use a simple strip with breadcrumb and utilities; the bar should support the page, not dominate it
- Avoid composer-like search capsules and shortcut chips as the main visual motif
- Primary content tabs can sit below the page header when needed, but the KPI row should remain the first major visual block

### Panel Frame
- Outer panel: 28px to 32px radius
- Padding: 18px to 24px
- Border: 1px low-contrast slate
- Background: fog shell, with white modules nested where detail density increases
- Use soft lift to separate major shells when they sit on the same canvas row

### KPI Card
- Use wide pastel tiles as the page's first-read telemetry row
- Value: 40px to 48px
- Meta: 13px to 14px, mono only when the value is machine-formatted
- Delta or trend can sit at the far edge in smaller text
- Tile families should repeat consistently:
  - planner blue
  - perception lilac
  - services ice
  - runtime mint or neutral family as needed
- On narrower work surfaces, stack before squeezing below readable tile width

### Data Strip
- Font: IBM Plex Mono 11px
- Background: pale module inset
- Border radius: 14px to 16px
- Padding: 8px 12px

### Live Workbench
- The live-view panel should dominate the main canvas
- Layout: header row, media stage, compact inference footer
- Use outer fog shell plus a white inner stage around the media frame
- Header should support:
  - title
  - live badge
  - overlay toggle
  - utility action
- The stage should feel operational and large enough to inspect detections, not like a decorative preview

### Section Tabs
- Tabs are text-first with clear active emphasis
- Keep them horizontally aligned and scrollable on smaller widths
- Tabs should switch sibling pages within the same navigation group

### Right Support Column
- Width: 340px to 400px when present
- Use stacked fog shells with smaller white modules inside
- Good content:
  - process composition
  - sensor input state
  - active services
  - pipeline summaries
- This column supports the workbench; it should not visually outrank the live-view panel

### Search Capsule
- Use only when the page truly needs search
- Avoid making capsule search a defining motif for the dashboard shell
- If present, match the larger rounded system and keep it secondary to telemetry and work panels

### Utility Icon Group
- Icon size: 16px to 20px
- Container padding stays light
- Icons should feel aligned and precise, never like decorative badges

### Status Badge
- Height: 22px to 24px
- Font: IBM Plex Mono 11px medium
- Dot + label, or compact pill label for compact contexts
- Use one of: mint, amber, coral, slate
- Live or critical states can be more visible than the rest of the shell

## Validation
- Spacing stays on the 4px grid
- Region hierarchy is obvious without squinting:
  - navigation rail
  - KPI ribbon
  - main live workbench
  - right support column
- KPI tiles use stable family colors and crisp value typography
- Major shells read as intentional groups, not as flat workspace cards
- Data readouts use mono where scan alignment matters
- First screen on 1440px-wide desktop should expose breadcrumb strip, page heading, KPI row, live workbench, and at least the top of the right support column without scrolling
- No new random hex values inside components when a token or pattern already exists

## Reference
- Primary visual reference: the approved AURA Pipeline Overview screenshot supplied with the task
- Rule: keep AURA's telemetry semantics and operator-console purpose while matching the screenshot's large-panel hierarchy, pastel KPI row, and live-workbench composition
