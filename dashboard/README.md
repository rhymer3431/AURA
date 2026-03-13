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

## Frontend-only commands

```bash
npm install
npm run dev
npm run build
npm run tauri:dev
npm run tauri:build
npm run test:run
```

## Notes

- 브라우저 dev server는 여전히 `http://127.0.0.1:5173` 에서 동작하지만, 기본 사용 경로는 브라우저가 아니라 Tauri desktop shell입니다.
- packaged Tauri 앱은 backend를 `http://127.0.0.1:8095` 로 직접 호출합니다.
- Windows에서 Tauri를 실행하려면 Rust toolchain이 필요합니다.
