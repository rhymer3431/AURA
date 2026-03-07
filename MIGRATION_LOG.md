# Migration Log

## Function-Oriented `src/` Refactor
- Removed the old canonical wrapper package layer under `src/`.
- Moved canonical modules directly into `src/adapters`, `src/apps`, `src/common`, `src/control`, `src/inference`, `src/locomotion`, `src/runtime`, `src/services`, and `src/vendor`.
- Kept `navdp/`, `navdp_sidecar/`, and `g1_play/` as compatibility-only shims that bootstrap `src` and forward to canonical modules.
- Restored `play_g1_keyboard_onnx.py` as a compatibility shim to `locomotion.entrypoint`.
- Pointed canonical PowerShell launchers at `runtime.g1_bridge`, `apps.navdp_server_app`, and `apps.dual_server_app`.
- Moved shared transform helpers into `src/common/geometry.py` and re-exported them from `src/adapters/navdp_http.py` for compatibility.
- Removed the duplicated `navdp_sidecar/depth_anything` tree so `src/vendor/depth_anything` is the only vendor copy.
