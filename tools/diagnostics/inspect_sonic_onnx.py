from pathlib import Path

import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "apps" / "gear_sonic_deploy"

models = {
    "encoder": MODEL_DIR / "model_encoder.onnx",
    "decoder": MODEL_DIR / "model_decoder.onnx",
    "planner": MODEL_DIR / "planner_sonic.onnx",
}

for name, path in models.items():
    print(f"\n{'='*50}")
    print(f"Model: {name}  ({path})")
    sess = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    print("  INPUTS:")
    for inp in sess.get_inputs():
        print(f"    name={inp.name}  shape={inp.shape}  dtype={inp.type}")

    print("  OUTPUTS:")
    for out in sess.get_outputs():
        print(f"    name={out.name}  shape={out.shape}  dtype={out.type}")
