# AURA Dashboard

이 디렉터리는 AURA 런타임을 제어하고 WebRTC viewer를 표시하는 React + Vite + Tauri 대시보드입니다.

## Run

로컬 개발 실행은 저장소 루트에서 아래 PowerShell 런처를 사용합니다.

```powershell
.\scripts\powershell\run_dashboard.ps1
```

이 런처는 다음을 함께 실행합니다.

- Python `aiohttp` dashboard backend (`http://127.0.0.1:8095`)
- Tauri desktop app (`npm run tauri:dev`)

런처는 이제 backend `http://127.0.0.1:8095/api/bootstrap` 가 준비된 뒤에 Tauri를 시작합니다. 따라서 full-stack 실행에서는 백엔드가 아직 뜨지 않은 순간의 Vite mock fallback으로 대시보드가 먼저 고정되는 문제를 줄입니다.

## Frontend-only commands

```bash
npm install
npm run dev
npm run build
npm run tauri:dev
npm run tauri:build
npm run test:run
```

`npm run tauri:dev` now auto-selects the next available localhost port when `127.0.0.1:5173` is already occupied. Set `AURA_DASHBOARD_DEV_PORT` if you need to pin a different preferred dev port.

`npm run dev` now falls back to a lightweight mock `/api` surface when the dashboard backend is not listening on `127.0.0.1:8095`. This keeps the React shell usable for UI work instead of flooding Vite with `ECONNREFUSED` proxy errors.

For a live backend during browser development:

- run `.\scripts\powershell\run_dashboard.ps1` from the repository root, or
- set `VITE_AURA_API_BASE` to call a backend directly from the browser, or
- set `AURA_DASHBOARD_PROXY_TARGET` before `npm run dev` if the backend is on a different host or port.

## Project-Local Codex Skill

This dashboard now vendors the upstream `Dammyjay93/interface-design` frontend skill as a repo-local Codex skill.

- Vendored upstream snapshot: `dashboard/tools/interface-design/`
- Pinned upstream commit: `8c407c1c42890010a9eb403a9f419b1eeadcfdad`
- Codex project entrypoint: `.claude/skills/interface-design/SKILL.md`
- Project design memory path: `dashboard/.interface-design/system.md`
- Default audit/extract targets: `dashboard/src`, `dashboard/src/styles`, `dashboard/guidelines/Guidelines.md`
- This integration is repo-local only. It does not install anything into `~/.codex/skills`.
- Start a new Codex session or restart Codex after pulling these files so local skill discovery can reload.

## Notes

- 브라우저 dev server 기본 포트는 `http://127.0.0.1:5173` 이지만, 해당 포트가 이미 사용 중이면 `npm run tauri:dev` 는 다음 사용 가능한 localhost 포트로 자동 이동합니다.
- packaged Tauri 앱은 backend를 `http://127.0.0.1:8095` 로 직접 호출합니다.
- Windows에서 Tauri를 실행하려면 Rust toolchain이 필요합니다.
