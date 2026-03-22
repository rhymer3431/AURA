# AURA Dashboard Interface System

## Intent
- Human: 로봇 런타임을 모니터링하고 장애 징후를 빠르게 판독해야 하는 운영자/개발자
- Primary job: 현재 세션 상태, gateway/telemetry, planner, memory, diagnostics를 한 화면에서 빠르게 스캔
- Feel: quiet operations console with SnowUI-like restraint, dense but calm, diagnostic not decorative

## Domain
- control room
- runtime telemetry
- field diagnostics
- sensor fusion
- lab notebook
- command arbitration
- safety rail
- instrument sheet
- system checklist

## Color World
- soft porcelain console shells
- graphite ink and hairline dividers
- telemetry cyan LEDs
- actuator green status lamps
- caution amber indicators
- fault coral annotations
- quiet black action pills

## Signature
- SnowUI-style quiet shell: hairline dividers, compact pills, shallow header/search controls, with telemetry accents reserved for runtime meaning

## Defaults Replaced
- Default template admin shell -> compact SnowUI-derived header and sidebar framing
- Loud colored UI accents -> neutral shell with semantic color reserved for runtime states
- Large soft dashboard cards -> tighter panels, pills, and inset surfaces with clearer hierarchy
- Generic body typography -> `Inter` for UI with `IBM Plex Mono` for machine-readable data

## Tokens
- Base spacing: 4px
- Radius scale:
  - Micro: 4px
  - Small: 8px
  - Control: 12px
  - Capsule: 16px
  - Panel: 24px
- Depth:
  - Borders-first
  - Hairline dividers are allowed at 0.5px
  - No heavy shadows in standard dashboard flow
  - Inset regions use `rgba(0, 0, 0, 0.04)` to `rgba(0, 0, 0, 0.06)` style fills instead of shadow
- Typography:
  - UI: Inter
  - Data: IBM Plex Mono
  - Core sizes: 12px, 14px, 16px, 24px
  - Default weight: 400
- Border language:
  - Divider: rgba(0, 0, 0, 0.10)
  - Quiet border: rgba(24, 33, 37, 0.08)
  - Semantic emphasis only when state matters
- Surface language:
  - Canvas: light neutral, not pure white
  - Panels: near-white or pale porcelain
  - Utility fills: soft graphite alpha
- Color usage:
  - Black/graphite is valid for primary action emphasis
  - Telemetry cyan, emerald, amber, coral, violet are status colors first, decoration second

## Reusable Patterns

### Shell Navigation
- Sidebar width: 272px
- Collapsed sidebar width: 80px
- Item height: 40px to 44px
- Active state: soft neutral pill, not a loud filled block
- Section labels: 11px to 12px, low-contrast, compact
- Use SnowUI-style grouping: brand top, navigation center, account/utilities bottom

### Top Bar
- Height should stay visually shallow and compact
- Breadcrumb, search, and action icons live on one horizontal band
- Use bottom divider or same-plane separation instead of elevated card treatment
- Search control is a rounded capsule with inline shortcut/meta chip

### Panel Frame
- Outer panel: 20px to 24px radius
- Padding: 16px to 24px
- Border: 0.5px to 1px low-contrast graphite
- Background: near-white porcelain, not pure white
- Do not stack unnecessary shadows to separate adjacent panels

### KPI Card
- Prefer quiet stat blocks over loud marketing cards
- Value: 24px to 32px depending on density
- Meta: mono 11px to 12px
- Accent rail is optional, not mandatory
- If color appears, it should encode runtime state or service health

### Data Strip
- Font: IBM Plex Mono 11px
- Background: soft inset graphite/porcelain tone
- Border radius: 12px to 16px
- Padding: 10px 12px

### Search Capsule
- Width: around 160px minimum in compact top bars
- Height: 32px to 36px
- Fill: quiet neutral alpha
- Embedded shortcut chip uses hairline border and 12px text

### Utility Icon Group
- Icon size: 16px to 20px
- Container padding stays light
- Icons should feel aligned and quiet, never like decorative badges

### Status Badge
- Height: 24px to 26px
- Font: IBM Plex Mono 11px medium
- Dot + label, or tight pill label for compact contexts
- Use one of: cyan, emerald, amber, coral, violet, slate
- Avoid oversized rounded chips unless they communicate a strong status

## Validation
- Spacing stays on the 4px grid
- Quiet surface hierarchy remains consistent before adding color
- Hairline borders and inset fills should do most of the structural work
- Data readouts use mono where scan alignment matters
- Navigation and top bar should stay compact, not drift back into bulky admin templates
- No new random hex values inside components when a token or pattern already exists

## Reference
- External reference: SnowUI dashboard canvas in the linked Figma file
- Rule: borrow structure, density, and restraint from SnowUI; keep AURA's telemetry semantics and operator-console purpose
