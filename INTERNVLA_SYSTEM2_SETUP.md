# InternVLA System2 Setup

## Canonical Launcher
- Canonical: `.\scripts\powershell\run_internvla_system2.ps1`
- Compatibility: `.\run_internvla_system2.ps1`

## Default Asset Locations
- Model: `artifacts/models/InternVLA-N1-System2.Q4_K_M.gguf`
- MMProj: `artifacts/models/InternVLA-N1-System2.mmproj-Q8_0.gguf`
- Chat template: `scripts/powershell/internvla_system2_chat_template.jinja`
- Existing root file names are still accepted as fallback paths.

## Prerequisites
1. `llama-server` is installed and reachable, or `LLAMA_SERVER_EXE` points to it.
2. Use a GPU-enabled `llama.cpp` build. CPU-only builds are rejected by the launcher.
3. CUDA 13.1 runtime DLLs must be available next to `llama-server`; the launcher refreshes them when needed.

## Quick Start
```powershell
.\run_internvla_system2.ps1
```

## Common Variants
```powershell
.\run_internvla_system2.ps1 -Port 8081 -ContextSize 12288
.\run_internvla_system2.ps1 -LlamaServer "C:\tools\llama.cpp\llama-server.exe"
.\run_internvla_system2.ps1 -- -n 256 --temp 0.2
```

## Environment Variables
- `INTERNVLA_HOST`
- `INTERNVLA_PORT`
- `INTERNVLA_MODEL_PATH`
- `INTERNVLA_MMPROJ_PATH`
- `INTERNVLA_CHAT_TEMPLATE_FILE`
- `LLAMA_SERVER_EXE`
- `INTERNVLA_HF_REPO`
- `INTERNVLA_HF_MMPROJ_FILE`
- `INTERNVLA_CTX_SIZE`
- `INTERNVLA_GPU_LAYERS`
- `INTERNVLA_REASONING_BUDGET`

## Runtime Notes
- The launcher resolves project-relative paths from the repo root, not from the script directory.
- The launcher prefers artifacts under `artifacts/models/` and only falls back to the legacy root filenames when those files still exist.
- The launcher now pins the official InternVLA multimodal chat template instead of relying on GGUF metadata inference.
- The default reasoning budget is `0` so the dual-system JSON response path does not wait on extra thinking tokens.
- The dual server now requests System2 outputs with an explicit JSON schema and sane sampling overrides instead of relying on the GGUF sampling metadata defaults.
