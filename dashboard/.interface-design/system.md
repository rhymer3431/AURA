# AURA Dashboard Interface System

## Intent
- Human: 로봇 런타임을 모니터링하고 장애 징후를 빠르게 판독해야 하는 운영자/개발자
- Primary job: 현재 세션 상태, gateway/telemetry, planner, memory, diagnostics를 한 화면에서 빠르게 스캔
- Feel: quiet operations console, dense but calm, diagnostic not decorative

## Domain
- control room
- runtime telemetry
- field diagnostics
- sensor fusion
- lab notebook
- command arbitration
- safety rail

## Color World
- matte graphite equipment panels
- warm lab paper
- telemetry cyan LEDs
- actuator green status lamps
- caution amber indicators
- fault coral annotations

## Signature
- Thin telemetry rail and mono data strip running through shell, cards, and diagnostics blocks

## Defaults Replaced
- Default `Inter` dashboard typography -> `IBM Plex Sans` + `IBM Plex Mono`
- Bright pastel metric cards -> muted paper panels with one telemetry rail per card
- Floating white-on-white sections -> same-plane shell with borders-only elevation

## Tokens
- Base spacing: 4px
- Radius scale:
  - Micro: 14px
  - Control: 18px
  - Panel: 26px
- Depth:
  - Borders-only
  - No heavy shadows
  - Inset regions use a darker paper tone instead of shadow
- Typography:
  - UI: IBM Plex Sans
  - Data: IBM Plex Mono

## Reusable Patterns

### Shell Navigation
- Sidebar width: 272px
- Item height: 44px minimum
- Active state: inset paper fill + left telemetry rail
- Section labels: 11px mono uppercase

### Panel Frame
- Outer panel: 26px radius
- Padding: 20px to 24px
- Border: 1px low-contrast graphite
- Background: off-white paper, not pure white

### KPI Card
- Rail: 3px accent stripe on left edge
- Value: 32px semibold, negative tracking
- Meta: mono 11px
- Height: 132px minimum

### Data Strip
- Font: IBM Plex Mono 11px
- Background: inset paper tone
- Border radius: 16px
- Padding: 10px 12px

### Status Badge
- Height: 26px
- Font: IBM Plex Mono 11px medium
- Dot + label
- Use one of: cyan, emerald, amber, coral, violet, slate

## Validation
- Spacing stays on the 4px grid
- Borders-only depth remains consistent
- Data readouts use mono where scan alignment matters
- No new random hex values inside components when a token or pattern already exists
