import onnxruntime as ort

models = {
    "encoder": "gear_sonic_deploy/model_encoder.onnx",
    "decoder": "gear_sonic_deploy/model_decoder.onnx",
    "planner": "gear_sonic_deploy/planner_sonic.onnx",
}

for name, path in models.items():
    print(f"\n{'='*50}")
    print(f"Model: {name}  ({path})")
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    print("  INPUTS:")
    for inp in sess.get_inputs():
        print(f"    name={inp.name}  shape={inp.shape}  dtype={inp.type}")

    print("  OUTPUTS:")
    for out in sess.get_outputs():
        print(f"    name={out.name}  shape={out.shape}  dtype={out.type}")
