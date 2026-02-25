#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import logging
import sys
import types
from pathlib import Path


LOGGER = logging.getLogger("run_groot_policy_server_fp8")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Isaac-GR00T policy server with TensorRT DiT replacement."
    )
    parser.add_argument(
        "--groot-repo-root",
        default=r"C:/Users/mango/project/Isaac-GR00T-tmp",
        help="Path to Isaac-GR00T checkout",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local GR00T checkpoint path (directory with config.json + safetensors shards)",
    )
    parser.add_argument(
        "--trt-engine-path",
        required=True,
        help="TensorRT engine path for action-head DiT (.trt)",
    )
    parser.add_argument("--embodiment-tag", default="UNITREE_G1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trt-device-index", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--strict", dest="strict", action="store_true")
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    parser.add_argument("--use-sim-policy-wrapper", action="store_true")
    return parser.parse_args()


def _resolve_embodiment_tag(raw: str, embodiment_enum_cls):
    token = raw.strip()
    if token in embodiment_enum_cls.__members__:
        return embodiment_enum_cls[token]

    lowered = token.lower()
    for item in embodiment_enum_cls:
        if item.value.lower() == lowered:
            return item

    options = ", ".join(sorted(embodiment_enum_cls.__members__.keys()))
    raise ValueError(f"Unknown embodiment tag '{raw}'. Available: {options}")


def _install_transformers_compat_patches() -> None:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.modeling_utils import PreTrainedModel

    if not getattr(PretrainedConfig, "_groot_fp8_attn_patch", False):
        original_getattribute = PretrainedConfig.__getattribute__

        def patched_getattribute(self, name: str):
            if name == "_attn_implementation_autoset":
                try:
                    return original_getattribute(self, name)
                except AttributeError:
                    return False
            if name == "initializer_range":
                try:
                    return original_getattribute(self, name)
                except AttributeError:
                    return 0.02
            if name == "initializer_factor":
                try:
                    return original_getattribute(self, name)
                except AttributeError:
                    return 1.0
            return original_getattribute(self, name)

        PretrainedConfig.__getattribute__ = patched_getattribute  # type: ignore[assignment]
        PretrainedConfig._groot_fp8_attn_patch = True  # type: ignore[attr-defined]

    if not getattr(PreTrainedModel, "_groot_fp8_flash_patch", False):
        if hasattr(PreTrainedModel, "_check_and_adjust_attn_implementation"):
            original_adjust = PreTrainedModel._check_and_adjust_attn_implementation

            def patched_adjust(self, attn_implementation, is_init_check: bool = False):
                if isinstance(attn_implementation, str) and attn_implementation.startswith(
                    "flash_attention"
                ):
                    attn_implementation = "sdpa"
                try:
                    return original_adjust(
                        self, attn_implementation=attn_implementation, is_init_check=is_init_check
                    )
                except Exception as exc:  # pragma: no cover - fallback path
                    logging.warning(
                        "Attention implementation '%s' unavailable. Using compatibility fallback: %s",
                        attn_implementation,
                        exc,
                    )
                    return "sdpa"

            PreTrainedModel._check_and_adjust_attn_implementation = patched_adjust  # type: ignore[assignment]

        if hasattr(PreTrainedModel, "_flash_attn_2_can_dispatch"):
            original_dispatch = PreTrainedModel._flash_attn_2_can_dispatch

            def patched_dispatch(self, is_init_check: bool = False):
                try:
                    return original_dispatch(self, is_init_check=is_init_check)
                except Exception as exc:  # pragma: no cover - fallback path
                    logging.warning(
                        "FlashAttention2 unavailable. Falling back to SDPA/eager attention: %s",
                        exc,
                    )
                    if hasattr(self.config, "_attn_implementation"):
                        self.config._attn_implementation = "sdpa"
                    if hasattr(self.config, "_attn_implementation_internal"):
                        self.config._attn_implementation_internal = "sdpa"
                    return False

            PreTrainedModel._flash_attn_2_can_dispatch = patched_dispatch  # type: ignore[assignment]
        elif hasattr(PreTrainedModel, "_check_and_enable_flash_attn_2"):
            original_check = PreTrainedModel._check_and_enable_flash_attn_2

            @classmethod
            def patched_check(cls, config, *args, **kwargs):
                try:
                    return original_check.__func__(cls, config, *args, **kwargs)  # type: ignore[attr-defined]
                except Exception as exc:  # pragma: no cover - fallback path
                    logging.warning(
                        "FlashAttention2 unavailable. Falling back to SDPA/eager attention: %s",
                        exc,
                    )
                    if hasattr(config, "_attn_implementation"):
                        config._attn_implementation = "sdpa"
                    if hasattr(config, "_attn_implementation_internal"):
                        config._attn_implementation_internal = "sdpa"
                    return config

            PreTrainedModel._check_and_enable_flash_attn_2 = patched_check  # type: ignore[assignment]

        PreTrainedModel._groot_fp8_flash_patch = True  # type: ignore[attr-defined]


def _install_transformers_image_utils_compat() -> None:
    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput") and hasattr(image_utils, "ImageInput"):
        # Older/newer transformers builds do not always export VideoInput.
        image_utils.VideoInput = image_utils.ImageInput  # type: ignore[attr-defined]

    if not hasattr(image_utils, "make_batched_videos"):
        # Compatibility shim for Eagle fast image processor import-time expectations.
        def make_batched_videos(videos):
            if videos is None:
                return []

            if isinstance(videos, (list, tuple)):
                if len(videos) == 0:
                    return []
                first = videos[0]
                if isinstance(first, (list, tuple)):
                    return [list(video) for video in videos]
                return [list(videos)]

            return [[videos]]

        image_utils.make_batched_videos = make_batched_videos  # type: ignore[attr-defined]


def _set_attn_impl(config, impl: str) -> None:
    for attr in ("_attn_implementation", "_attn_implementation_internal"):
        if hasattr(config, attr):
            try:
                setattr(config, attr, impl)
            except Exception:
                pass

    if hasattr(config, "attn_implementation"):
        try:
            setattr(config, "attn_implementation", impl)
        except Exception:
            pass


def _wrap_ctor_force_sdpa(cls):
    if getattr(cls, "_groot_sdpa_ctor_patch", False):
        return cls

    class _Wrapped(cls):  # type: ignore[misc,valid-type]
        def __init__(self, config, *args, **kwargs):
            _set_attn_impl(config, "sdpa")
            super().__init__(config, *args, **kwargs)

    _Wrapped.__name__ = cls.__name__
    _Wrapped.__qualname__ = cls.__qualname__
    _Wrapped.__module__ = cls.__module__
    _Wrapped._groot_sdpa_ctor_patch = True  # type: ignore[attr-defined]
    return _Wrapped


def _patch_eagle_remote_module_for_sdpa() -> None:
    cache_root = Path.home() / ".cache" / "huggingface" / "modules"
    if str(cache_root) not in sys.path:
        sys.path.insert(0, str(cache_root))

    module_name = "transformers_modules.Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2.modeling_eagle3_vl"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        LOGGER.warning("Could not import Eagle remote module for SDPA patching: %s", exc)
        return

    if getattr(module, "_groot_sdpa_patch", False):
        return

    for cls_name in (
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "SiglipVisionModel",
        "Siglip2VisionModel",
    ):
        cls = getattr(module, cls_name, None)
        if cls is None:
            continue
        setattr(module, cls_name, _wrap_ctor_force_sdpa(cls))

    eagle_cls = getattr(module, "Eagle3_VLForConditionalGeneration", None)
    if eagle_cls is not None and not getattr(eagle_cls, "_groot_sdpa_patch", False):
        original_init = eagle_cls.__init__

        def patched_init(self, config, *args, **kwargs):
            text_config = getattr(config, "text_config", None)
            _set_attn_impl(config, "sdpa")
            if text_config is not None:
                _set_attn_impl(text_config, "sdpa")
            return original_init(self, config, *args, **kwargs)

        eagle_cls.__init__ = patched_init  # type: ignore[assignment]
        eagle_cls._groot_sdpa_patch = True  # type: ignore[attr-defined]

    module._groot_sdpa_patch = True  # type: ignore[attr-defined]


def _patch_eagle_processor_module_compat() -> None:
    cache_root = Path.home() / ".cache" / "huggingface" / "modules"
    if str(cache_root) not in sys.path:
        sys.path.insert(0, str(cache_root))

    module_name = "transformers_modules.Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2.processing_eagle3_vl"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        LOGGER.warning("Could not import Eagle processor module for compatibility patching: %s", exc)
        return

    processor_cls = getattr(module, "Eagle3_VLProcessor", None)
    if processor_cls is not None and not getattr(processor_cls, "_groot_kwargs_patch", False):
        original_from_args_and_dict = processor_cls.from_args_and_dict

        @classmethod
        def patched_from_args_and_dict(cls, args, processor_dict, **kwargs):
            try:
                return original_from_args_and_dict.__func__(cls, args, processor_dict, **kwargs)  # type: ignore[attr-defined]
            except ValueError as exc:
                if "dictionary update sequence element" not in str(exc):
                    raise

                processor_dict = dict(processor_dict or {})
                return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
                processor_dict.pop("processor_class", None)

                unused_kwargs = cls.validate_init_kwargs(
                    processor_config=processor_dict, valid_kwargs=cls.valid_kwargs
                )
                processor = cls(*args, **processor_dict)

                for key in list(kwargs.keys()):
                    if hasattr(processor, key):
                        setattr(processor, key, kwargs.pop(key))

                if isinstance(unused_kwargs, dict):
                    kwargs.update(unused_kwargs)
                elif isinstance(unused_kwargs, tuple):
                    for item in unused_kwargs:
                        if isinstance(item, dict):
                            kwargs.update(item)
                elif not isinstance(unused_kwargs, (list, set, type(None))):
                    LOGGER.debug(
                        "Ignoring unsupported unused_kwargs type from Eagle processor: %s",
                        type(unused_kwargs).__name__,
                    )

                if return_unused_kwargs:
                    return processor, kwargs
                return processor

        processor_cls.from_args_and_dict = patched_from_args_and_dict  # type: ignore[assignment]
        processor_cls._groot_kwargs_patch = True  # type: ignore[attr-defined]

    image_module_name = (
        "transformers_modules.Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2.image_processing_eagle3_vl_fast"
    )
    try:
        image_module = importlib.import_module(image_module_name)
    except Exception as exc:
        LOGGER.warning("Could not import Eagle image processor module for compatibility patching: %s", exc)
        return

    image_cls = getattr(image_module, "Eagle3_VLImageProcessorFast", None)
    if image_cls is None or getattr(image_cls, "_groot_prepare_input_patch", False):
        return

    if not hasattr(image_cls, "_prepare_input_images") and hasattr(image_cls, "_prepare_image_like_inputs"):
        def _prepare_input_images(self, images, do_convert_rgb=None, input_data_format=None, device=None):
            return self._prepare_image_like_inputs(
                images=images,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )

        image_cls._prepare_input_images = _prepare_input_images  # type: ignore[attr-defined]

    image_cls._groot_prepare_input_patch = True  # type: ignore[attr-defined]


def _register_gr00t_modules_lightweight(groot_root: Path) -> None:
    model_pkg = "gr00t.model"
    if model_pkg not in sys.modules:
        pkg = types.ModuleType(model_pkg)
        pkg.__path__ = [str(groot_root / "gr00t" / "model")]
        sys.modules[model_pkg] = pkg

    n1d6_pkg = "gr00t.model.gr00t_n1d6"
    if n1d6_pkg not in sys.modules:
        pkg = types.ModuleType(n1d6_pkg)
        pkg.__path__ = [str(groot_root / "gr00t" / "model" / "gr00t_n1d6")]
        sys.modules[n1d6_pkg] = pkg

    modules_pkg = "gr00t.model.modules"
    if modules_pkg not in sys.modules:
        pkg = types.ModuleType(modules_pkg)
        pkg.__path__ = [str(groot_root / "gr00t" / "model" / "modules")]
        sys.modules[modules_pkg] = pkg

    importlib.import_module("gr00t.model.gr00t_n1d6.gr00t_n1d6")
    importlib.import_module("gr00t.model.gr00t_n1d6.processing_gr00t_n1d6")


def main() -> None:
    _configure_logging()
    args = _parse_args()

    groot_root = Path(args.groot_repo_root).resolve()
    if not groot_root.exists():
        raise FileNotFoundError(f"Invalid --groot-repo-root: {groot_root}")

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Invalid --model-path: {model_path}")

    trt_engine_path = Path(args.trt_engine_path).resolve()
    if not trt_engine_path.exists():
        raise FileNotFoundError(f"Invalid --trt-engine-path: {trt_engine_path}")

    sys.path.insert(0, str(groot_root))
    sys.path.insert(0, str(groot_root / "scripts" / "deployment"))

    _install_transformers_compat_patches()
    _install_transformers_image_utils_compat()
    _patch_eagle_remote_module_for_sdpa()
    _patch_eagle_processor_module_compat()
    _register_gr00t_modules_lightweight(groot_root)

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer
    from standalone_inference_script import replace_dit_with_tensorrt

    embodiment_tag = _resolve_embodiment_tag(args.embodiment_tag, EmbodimentTag)
    LOGGER.info(
        "Starting FP8 policy server: model=%s engine=%s embodiment=%s host=%s port=%s",
        model_path,
        trt_engine_path,
        embodiment_tag.value,
        args.host,
        args.port,
    )

    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=str(model_path),
        device=args.device,
        strict=args.strict,
    )
    replace_dit_with_tensorrt(policy, str(trt_engine_path), device=args.trt_device_index)

    if args.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(policy=policy, host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
