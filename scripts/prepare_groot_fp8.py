#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


LOGGER = logging.getLogger("prepare_groot_fp8")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _download_model_if_needed(repo_id: str, model_dir: Path, skip_download: bool) -> None:
    required = [model_dir / "config.json", model_dir / "model.safetensors.index.json"]
    if all(path.exists() for path in required):
        LOGGER.info("Model files already present at %s", model_dir)
        return

    if skip_download:
        missing = [str(path) for path in required if not path.exists()]
        raise FileNotFoundError(
            "Missing required model files while --skip-download is set: " + ", ".join(missing)
        )

    LOGGER.info("Downloading checkpoint from %s -> %s", repo_id, model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(model_dir))
    LOGGER.info("Checkpoint download complete")


def _load_model_config(model_dir: Path) -> Dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _prepare_gr00t_import(groot_repo_root: Path) -> None:
    root_str = str(groot_repo_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    model_pkg = "gr00t.model"
    if model_pkg not in sys.modules:
        pkg = types.ModuleType(model_pkg)
        pkg.__path__ = [str(groot_repo_root / "gr00t" / "model")]
        sys.modules[model_pkg] = pkg

    modules_pkg = "gr00t.model.modules"
    if modules_pkg not in sys.modules:
        pkg = types.ModuleType(modules_pkg)
        pkg.__path__ = [str(groot_repo_root / "gr00t" / "model" / "modules")]
        sys.modules[modules_pkg] = pkg


def _build_dit_module(config: Dict, groot_repo_root: Path) -> tuple[torch.nn.Module, bool]:
    _prepare_gr00t_import(groot_repo_root)
    dit_module = importlib.import_module("gr00t.model.modules.dit")

    diff_cfg = dict(config.get("diffusion_model_cfg", {}))
    cross_attention_dim = int(config.get("backbone_embedding_dim", 2048))
    use_alternate = bool(config.get("use_alternate_vl_dit", True))
    attend_text_every_n_blocks = int(config.get("attend_text_every_n_blocks", 2))

    if use_alternate:
        model = dit_module.AlternateVLDiT(
            **diff_cfg,
            cross_attention_dim=cross_attention_dim,
            attend_text_every_n_blocks=attend_text_every_n_blocks,
        )
    else:
        model = dit_module.DiT(**diff_cfg, cross_attention_dim=cross_attention_dim)

    return model, use_alternate


def _load_dit_weights(model_dir: Path, prefix: str = "action_head.model.") -> Dict[str, torch.Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    selected_keys = [name for name in weight_map if name.startswith(prefix)]
    if not selected_keys:
        raise RuntimeError(f"No keys with prefix '{prefix}' found in {index_path}")

    shard_to_keys: Dict[str, list[str]] = defaultdict(list)
    for key in selected_keys:
        shard_to_keys[weight_map[key]].append(key)

    state_dict: Dict[str, torch.Tensor] = {}
    for shard_name, shard_keys in sorted(shard_to_keys.items()):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard: {shard_path}")
        LOGGER.info("Reading %s (%s tensors)", shard_name, len(shard_keys))
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in shard_keys:
                state_dict[key[len(prefix) :]] = handle.get_tensor(key)

    LOGGER.info("Loaded %s DiT tensors from safetensors shards", len(state_dict))
    return state_dict


def _export_onnx(
    model: torch.nn.Module,
    *,
    use_alternate: bool,
    config: Dict,
    onnx_path: Path,
    device: torch.device,
    batch_size: int,
    opt_sa_len: int,
    opt_vl_len: int,
) -> None:
    input_embedding_dim = int(config.get("input_embedding_dim", 1536))
    backbone_embedding_dim = int(config.get("backbone_embedding_dim", 2048))
    dtype = torch.bfloat16

    class AlternateWrapper(torch.nn.Module):
        def __init__(self, base: torch.nn.Module) -> None:
            super().__init__()
            self.base = base

        def forward(
            self,
            sa_embs: torch.Tensor,
            vl_embs: torch.Tensor,
            timestep: torch.Tensor,
            image_mask: torch.Tensor,
            backbone_attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            return self.base(
                sa_embs,
                vl_embs,
                timestep,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )

    class BasicWrapper(torch.nn.Module):
        def __init__(self, base: torch.nn.Module) -> None:
            super().__init__()
            self.base = base

        def forward(
            self,
            sa_embs: torch.Tensor,
            vl_embs: torch.Tensor,
            timestep: torch.Tensor,
        ) -> torch.Tensor:
            return self.base(sa_embs, vl_embs, timestep)

    sa_embs = torch.randn(
        (batch_size, opt_sa_len, input_embedding_dim),
        device=device,
        dtype=dtype,
    )
    vl_embs = torch.randn(
        (batch_size, opt_vl_len, backbone_embedding_dim),
        device=device,
        dtype=dtype,
    )
    timestep = torch.ones((batch_size,), device=device, dtype=torch.int64)

    dynamic_axes = {
        "sa_embs": {0: "batch_size", 1: "sa_seq_len"},
        "vl_embs": {0: "batch_size", 1: "vl_seq_len"},
        "timestep": {0: "batch_size"},
        "output": {0: "batch_size", 1: "sa_seq_len"},
    }

    if use_alternate:
        wrapper = AlternateWrapper(model).to(device=device)
        image_mask = torch.ones((batch_size, opt_vl_len), device=device, dtype=torch.bool)
        backbone_attention_mask = torch.ones((batch_size, opt_vl_len), device=device, dtype=torch.bool)
        export_inputs = (sa_embs, vl_embs, timestep, image_mask, backbone_attention_mask)
        input_names = [
            "sa_embs",
            "vl_embs",
            "timestep",
            "image_mask",
            "backbone_attention_mask",
        ]
        dynamic_axes["image_mask"] = {0: "batch_size", 1: "vl_seq_len"}
        dynamic_axes["backbone_attention_mask"] = {0: "batch_size", 1: "vl_seq_len"}
    else:
        wrapper = BasicWrapper(model).to(device=device)
        export_inputs = (sa_embs, vl_embs, timestep)
        input_names = ["sa_embs", "vl_embs", "timestep"]

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Exporting ONNX to %s", onnx_path)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            export_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=["output"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            export_params=True,
        )
    LOGGER.info("ONNX export complete")


def _build_tensorrt_engine(
    *,
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    workspace_mb: int,
    batch_size: int,
    input_embedding_dim: int,
    backbone_embedding_dim: int,
    min_sa_len: int,
    opt_sa_len: int,
    max_sa_len: int,
    min_vl_len: int,
    opt_vl_len: int,
    max_vl_len: int,
) -> None:
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    LOGGER.info("Parsing ONNX: %s", onnx_path)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("Failed to parse ONNX:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1024**2))
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    precision = precision.lower().strip()
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    elif precision == "fp8":
        config.set_flag(trt.BuilderFlag.FP8)
    elif precision != "fp32":
        raise ValueError(f"Unsupported precision: {precision}")

    profile = builder.create_optimization_profile()
    shape_ranges = {
        "sa_embs": (
            (batch_size, min_sa_len, input_embedding_dim),
            (batch_size, opt_sa_len, input_embedding_dim),
            (batch_size, max_sa_len, input_embedding_dim),
        ),
        "vl_embs": (
            (batch_size, min_vl_len, backbone_embedding_dim),
            (batch_size, opt_vl_len, backbone_embedding_dim),
            (batch_size, max_vl_len, backbone_embedding_dim),
        ),
        "timestep": (
            (batch_size,),
            (batch_size,),
            (batch_size,),
        ),
        "image_mask": (
            (batch_size, min_vl_len),
            (batch_size, opt_vl_len),
            (batch_size, max_vl_len),
        ),
        "backbone_attention_mask": (
            (batch_size, min_vl_len),
            (batch_size, opt_vl_len),
            (batch_size, max_vl_len),
        ),
    }

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        if inp.name not in shape_ranges:
            continue
        min_shape, opt_shape, max_shape = shape_ranges[inp.name]
        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
        LOGGER.info("Profile %s -> min=%s opt=%s max=%s", inp.name, min_shape, opt_shape, max_shape)

    config.add_optimization_profile(profile)

    LOGGER.info("Building TensorRT engine (%s)", precision.upper())
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT failed to build engine")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized_engine))
    LOGGER.info("Engine written: %s (%.2f MB)", engine_path, engine_path.stat().st_size / (1024**2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download GR00T checkpoint and build FP8 TensorRT DiT engine")
    parser.add_argument(
        "--repo-id",
        default="nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
        help="Hugging Face repo id",
    )
    parser.add_argument(
        "--model-dir",
        default="models/gr00t_n1_6_g1_pnp_apple_to_plate",
        help="Local checkpoint directory",
    )
    parser.add_argument(
        "--groot-repo-root",
        default=r"C:/Users/mango/project/Isaac-GR00T-tmp",
        help="Path to Isaac-GR00T checkout used to import DiT module definitions",
    )
    parser.add_argument(
        "--output-dir",
        default="models/gr00t_n1_6_g1_pnp_apple_to_plate/trt_fp8",
        help="Directory for exported ONNX and TRT engine",
    )
    parser.add_argument("--onnx-name", default="dit_model.onnx")
    parser.add_argument("--engine-name", default="dit_model_fp8.trt")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16", "fp8"],
        default="fp8",
        help="TensorRT precision mode",
    )
    parser.add_argument("--workspace-mb", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-sa-len", type=int, default=1)
    parser.add_argument("--opt-sa-len", type=int, default=51)
    parser.add_argument("--max-sa-len", type=int, default=256)
    parser.add_argument("--min-vl-len", type=int, default=1)
    parser.add_argument("--opt-vl-len", type=int, default=122)
    parser.add_argument("--max-vl-len", type=int, default=512)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = _parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    onnx_path = output_dir / args.onnx_name
    engine_path = output_dir / args.engine_name

    _download_model_if_needed(args.repo_id, model_dir, args.skip_download)
    config = _load_model_config(model_dir)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    needs_export = args.force_export or not onnx_path.exists()
    if needs_export:
        model, use_alternate = _build_dit_module(config, Path(args.groot_repo_root))
        state_dict = _load_dit_weights(model_dir)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Unexpected load_state_dict output: missing={missing}, unexpected={unexpected}")
        model = model.eval().to(device=device, dtype=torch.bfloat16)
        _export_onnx(
            model,
            use_alternate=use_alternate,
            config=config,
            onnx_path=onnx_path,
            device=device,
            batch_size=args.batch_size,
            opt_sa_len=args.opt_sa_len,
            opt_vl_len=args.opt_vl_len,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        LOGGER.info("Skipping ONNX export. Existing file found: %s", onnx_path)

    needs_build = args.force_build or not engine_path.exists()
    if needs_build:
        _build_tensorrt_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision=args.precision,
            workspace_mb=args.workspace_mb,
            batch_size=args.batch_size,
            input_embedding_dim=int(config.get("input_embedding_dim", 1536)),
            backbone_embedding_dim=int(config.get("backbone_embedding_dim", 2048)),
            min_sa_len=args.min_sa_len,
            opt_sa_len=args.opt_sa_len,
            max_sa_len=args.max_sa_len,
            min_vl_len=args.min_vl_len,
            opt_vl_len=args.opt_vl_len,
            max_vl_len=args.max_vl_len,
        )
    else:
        LOGGER.info("Skipping TensorRT build. Existing file found: %s", engine_path)

    manifest = {
        "repo_id": args.repo_id,
        "model_dir": str(model_dir),
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "sa_len": {"min": args.min_sa_len, "opt": args.opt_sa_len, "max": args.max_sa_len},
        "vl_len": {"min": args.min_vl_len, "opt": args.opt_vl_len, "max": args.max_vl_len},
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Saved build manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
