"""Download the TRELLIS.2 / Pixal3D / Hi3D model files into ComfyUI's model folders.

Skips files that already exist with the right size, so it is safe to run on every boot.

  python download_models.py --comfy /workspace/ComfyUI-Easy-Install/ComfyUI --models trellis2,pixal3d,hi3d

Env: HF_TOKEN (gated repos), HI3D_PRECISION=int8|fp16, PIXAL3D_MULTIVIEW=1
"""
import argparse
import os
import shutil
import sys

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
try:
    import hf_transfer  # noqa: F401
except ImportError:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import HfApi, hf_hub_download

# (repo, path in repo or None = find by file name, ComfyUI models/ subfolder)
SHARED_3D = [
    ("Comfy-Org/Pixal3D", "vae/trellis_2_shape_vae_bf16.safetensors", "vae"),
    ("Comfy-Org/Pixal3D", "vae/trellis_2_texture_vae_bf16.safetensors", "vae"),
    ("Comfy-Org/Pixal3D", "clip_vision/dino_v3_L_naf_fp32.safetensors", "clip_vision"),
    ("Comfy-Org/MoGe", "geometry_estimation/moge_2_vitl_normal_fp16.safetensors", "geometry_estimation"),
    ("Comfy-Org/BiRefNet", "background_removal/birefnet.safetensors", "background_removal"),
]

GROUPS = {
    # The official "Pixal3D: Image to Model" template switches between both diffusion models.
    "trellis2": SHARED_3D + [
        ("Comfy-Org/TRELLIS.2", "diffusion_models/trellis_2_int8_convrot.safetensors", "diffusion_models"),
    ],
    "pixal3d": SHARED_3D + [
        ("Comfy-Org/Pixal3D", "diffusion_models/pixal3d_int8_convrot.safetensors", "diffusion_models"),
    ],
    "hi3d": [
        ("Aero-Ex/ComfyUI-Hi3D", None, "checkpoints",
         "hi3d_v1.safetensors" if os.environ.get("HI3D_PRECISION", "int8") == "fp16" else "hi3d_v1_int8.safetensors"),
        ("Aero-Ex/ComfyUI-Hi3D", None, "clip_vision", "dinov2-large.safetensors"),
        ("Aero-Ex/ComfyUI-Hi3D", None, "clip_vision", "clip-vit-large-patch14.safetensors"),
        ("Comfy-Org/BiRefNet", "background_removal/birefnet.safetensors", "background_removal"),
    ],
}
if os.environ.get("PIXAL3D_MULTIVIEW", "0") == "1":
    GROUPS["pixal3d"].append(("Comfy-Org/Pixal3D", "diffusion_models/pixal3d_multiview_int8_convrot.safetensors", "diffusion_models"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True, help="ComfyUI folder")
    ap.add_argument("--models", default="trellis2,pixal3d,hi3d")
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    models_dir = os.path.join(args.comfy, "models")
    staging = os.path.join(models_dir, ".download_staging")
    repo_files = {}

    wanted, seen = [], set()
    for group in [g.strip() for g in args.models.split(",") if g.strip()]:
        if group not in GROUPS:
            print(f"Unknown model group '{group}' (choose from {', '.join(GROUPS)})")
            continue
        for entry in GROUPS[group]:
            key = (entry[0], entry[1] or entry[3], entry[2])
            if key not in seen:
                seen.add(key)
                wanted.append(entry)

    failed = 0
    for entry in wanted:
        repo, path, folder = entry[0], entry[1], entry[2]
        name = os.path.basename(path) if path else entry[3]
        target = os.path.join(models_dir, folder, name)
        try:
            if path is None:
                if repo not in repo_files:
                    repo_files[repo] = api.list_repo_files(repo)
                matches = [f for f in repo_files[repo] if os.path.basename(f) == name]
                if not matches:
                    raise FileNotFoundError(f"{name} not found in {repo}")
                path = matches[0]

            size = None
            try:
                size = api.get_paths_info(repo, [path])[0].size
            except Exception:
                pass
            if os.path.isfile(target) and (size is None or os.path.getsize(target) == size):
                print(f"  ok      {folder}/{name}")
                continue

            print(f"  get     {folder}/{name}  ({size / 1e9:.2f} GB)" if size else f"  get     {folder}/{name}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            tmp = hf_hub_download(repo, path, local_dir=staging, token=api.token)
            shutil.move(tmp, target)
        except Exception as e:
            failed += 1
            print(f"  FAILED  {folder}/{name} from {repo}: {e}")
            print("          If the repo is gated: accept its license on huggingface.co and set HF_TOKEN in the pod.")

    shutil.rmtree(staging, ignore_errors=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
