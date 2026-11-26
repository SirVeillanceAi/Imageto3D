# Imageto3D – RunPod template

A RunPod template that installs ComfyUI with **[ComfyUI-Easy-Install](https://github.com/Tavris1/ComfyUI-Easy-Install)** (Linux branch) and sets it up for image-to-3D on the GPU:

| Model | How it runs | What it does |
|---|---|---|
| **TRELLIS.2** | Native ComfyUI nodes (ComfyUI ≥ 0.34) | Image → textured 3D mesh |
| **Pixal3D** | Native ComfyUI nodes | Pixel-aligned image → textured 3D mesh (PBR) |
| **Hi3D v1.0** | [Aero-Ex ComfyUI-Hi3D](https://github.com/Aero-Ex/Aero-Ex-ComfyUI-Hi3D) node | Refines a coarse mesh + reference image up to 1536³ detail (geometry only) |

GPU only: the pod refuses to start without an NVIDIA GPU or with less than 10 GB VRAM, and checks that PyTorch can actually use the GPU before launching ComfyUI.

## Quick start

1. Create the template in RunPod using the values in **[runpod/template.md](runpod/template.md)**.
2. Deploy a pod from it on a 10 GB+ GPU. Set **Filter → CUDA Versions → 13.0**.
3. The first boot installs everything into `/workspace` (about 15–30 min plus roughly 20 GB of model downloads). Follow along in the pod's **Logs** tab.
4. When the log says `Starting ComfyUI`, click **Connect → HTTP 8188**.
5. Later boots skip the install and start in a minute or two.

## Using the models

- **TRELLIS.2 / Pixal3D:** in ComfyUI open **Templates → 3D → "Pixal3D: Image to Model"**. The `Switch to Trellis2` boolean selects TRELLIS.2 or Pixal3D. Results save to `ComfyUI/output/3d/`.
- **Hi3D refine:** open **Workflows → Imageto3D → Hi3D Refine Chain** (512 → 1024 → 1536) or **Hi3D Refine Direct** (one pass). Feed it the GLB you made with TRELLIS.2/Pixal3D plus the same input image.

## Files

| Path | Purpose |
|---|---|
| `scripts/start.sh` | Pod entrypoint: GPU checks, first-boot install, Torch/driver match, model check, starts ComfyUI on port 8188 |
| `scripts/install.sh` | Runs `ComfyUI-Easy-Install.sh` unattended and adds the Hi3D node |
| `scripts/download_models.py` | Downloads model files into the right ComfyUI folders; skips ones already there |
| `docker/Dockerfile` | Small Ubuntu 24.04 image with the system packages pre-installed |
| `.github/workflows/docker.yml` | Rebuilds that image to `ghcr.io/sirveillanceai/imageto3d:latest` on every push |
| `runpod/template.md` | Exact values for the RunPod template form |

## Notes

- **Why not the Easy-Install "Trellis2" add-on?** It installs an older custom node with compiled CUDA wheels that only support Torch 2.7–2.9 and pins numpy 1.26. ComfyUI now runs TRELLIS.2 and Pixal3D natively in pure PyTorch, so this template keeps Easy-Install's Torch 2.11 + CUDA 13.0 and skips that add-on.
- **Torch vs. driver:** Easy-Install ships Torch 2.11.0+cu130 (needs a CUDA 13 driver). If a pod lands on a CUDA 12.8 host, `start.sh` switches to Torch 2.8.0+cu128 using Easy-Install's Torch-Pack, and switches back automatically on a CUDA 13 host.
- **VRAM:** all three models ship int8 checkpoints and run through ComfyUI's dynamic VRAM management, so no `--lowvram` flag is used. The Hi3D node author measured the full 512→1536 chain on a 6 GB card.
- **Disk:** Easy-Install also installs its standard Pixaroma node set (WanVideoWrapper, KJNodes, etc.). Install + models need roughly 60–70 GB, so a 100 GB volume leaves room for outputs.
- **Licenses:** Pixal3D and Hi3D have their own model licenses (Hi3D Open License v1.0). Check them before commercial use.
- **Logs:** `/workspace/logs/install.log`, `models.log`, `comfyui.log`.

## GitHub basics for this repo

- **Changing a script:** open the file on github.com, click the pencil icon, edit, then **Commit changes**. That's it – new pods pick it up.
- **Option A (Docker image):** each commit to `main` triggers the **Actions** tab to rebuild the image (about 5 min). The first time, go to your GitHub profile → **Packages → imageto3d → Package settings → Change visibility → Public**, so RunPod can download it without a login.
- **Option B (no image):** the start command clones this repo fresh on every boot, so edits apply on the next pod restart.
