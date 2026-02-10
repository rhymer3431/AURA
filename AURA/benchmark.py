"""
Benchmark speed (end-to-end predict) for:
- YOLO-World: yolov8s-worldv2.pt
- YOLOE11:   yoloe-11s-seg.pt
- YOLOE26:   yoloe-26s-seg.pt

Measures: latency (ms) + FPS, including preprocess + model + postprocess.
Usage:
  pip install -U ultralytics opencv-python torch

  python bench_yolo_openvocab.py --source path/to/image.jpg --imgsz 640 --iters 200 --warmup 30 --device 0
"""

import argparse
import time
from statistics import mean, median

import torch
from ultralytics import YOLOE, YOLOWorld

DEFAULT_CLASSES = ["person", "chair", "bottle", "cup", "laptop"]


def _sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_model(model, source, imgsz, iters, warmup, device, classes=None):
    # Set open-vocab classes if supported
    if classes is not None and hasattr(model, "set_classes"):
        model.set_classes(classes)

    # Warmup
    for _ in range(warmup):
        _sync(device)
        _ = model.predict(source=source, imgsz=imgsz, device=device, verbose=False)
        _sync(device)

    # Timed runs
    times = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        _ = model.predict(source=source, imgsz=imgsz, device=device, verbose=False)
        _sync(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    ms = [t * 1000.0 for t in times]
    return {
        "mean_ms": mean(ms),
        "median_ms": median(ms),
        "p95_ms": sorted(ms)[int(0.95 * (len(ms) - 1))],
        "fps_mean": 1000.0 / mean(ms),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True, help="image/video path or webcam index like 0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--device", type=str, default="0", help='CUDA device id like "0" or "cpu"')
    ap.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES), help="comma-separated prompts/classes")
    args = ap.parse_args()

    # Device string for Ultralytics
    device = "cpu" if args.device.lower() == "cpu" else f"cuda:{args.device}"
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    print(f"Device: {device}")
    print(f"Source: {args.source}")
    print(f"imgsz={args.imgsz}, warmup={args.warmup}, iters={args.iters}")
    print(f"Classes: {classes}")
    print("-" * 72)

    # Models (weights names from Ultralytics docs)
    models = [
        ("YOLO-World (v2)", YOLOWorld("yolov8s-worldv2.pt"), classes),
        ("YOLOE-11S-SEG",   YOLOE("yoloe-11s-seg.pt"),      classes),
        ("YOLOE-26S-SEG",   YOLOE("yoloe-26s-seg.pt"),      classes),
    ]

    # Optional: enable half precision on GPU for more realistic speed
    # (Ultralytics will generally choose reasonable defaults; this is just a note)
    # If you need forced fp16: pass half=True in predict calls (not all exports support it).

    results = []
    for name, model, cls in models:
        out = bench_model(
            model=model,
            source=args.source,
            imgsz=args.imgsz,
            iters=args.iters,
            warmup=args.warmup,
            device=device,
            classes=cls,
        )
        results.append((name, out))

    # Print table
    header = f"{'Model':<18} | {'Mean(ms)':>9} | {'Median(ms)':>10} | {'P95(ms)':>8} | {'FPS':>8}"
    print(header)
    print("-" * len(header))
    for name, out in results:
        print(
            f"{name:<18} | "
            f"{out['mean_ms']:9.2f} | {out['median_ms']:10.2f} | {out['p95_ms']:8.2f} | {out['fps_mean']:8.2f}"
        )


if __name__ == "__main__":
    main()
