import importlib
import sys
from pathlib import Path

cache_root = Path.home() / ".cache" / "huggingface" / "modules"
if str(cache_root) not in sys.path:
    sys.path.insert(0, str(cache_root))

module_name = "transformers_modules.Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2.image_processing_eagle3_vl_fast"
module = importlib.import_module(module_name)
image_cls = module.Eagle3_VLImageProcessorFast

methods = [m for m in dir(image_cls) if not m.startswith("__")]
print("=== All available methods on Eagle3_VLImageProcessorFast ===")
for m in methods:
    print(f"  {m}")

candidates = [
    "_prepare_input_images",
    "_prepare_image_like_inputs",
    "preprocess",
    "_preprocess",
    "_prepare_images_and_videos",
    "_prepare_images",
    "prepare_images",
]
print("\n=== Candidate method availability ===")
for c in candidates:
    print(f"  {c}: {hasattr(image_cls, c)}")

print(
    f"\n_groot_prepare_input_patch already set: "
    f"{getattr(image_cls, '_groot_prepare_input_patch', False)}"
)
