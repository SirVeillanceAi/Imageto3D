# RunPod template settings

In RunPod go to **Templates → New Template** and fill in the fields below.

| Field | Value |
|---|---|
| Template Name | `Imageto3D - ComfyUI Easy Install (TRELLIS.2 / Pixal3D / Hi3D)` |
| Template Type | Pod |
| Compute Type | **Nvidia GPU** |
| Container Image | `ghcr.io/sirveillanceai/imageto3d:latest` *(Option A)* or `ubuntu:24.04` *(Option B)* |
| Container Start Command | Option A: leave empty. Option B: see below |
| Container Disk | `30 GB` |
| Volume Disk | `100 GB` |
| Volume Mount Path | `/workspace` |
| Expose HTTP Ports | `8188` |
| Expose TCP Ports | `22` |

### Option B start command (no Docker image needed)

Paste this into **Container Start Command**. It pulls the latest scripts from this repo on every boot.

```
bash -c "apt-get update && apt-get install -y git ca-certificates && rm -rf /opt/imageto3d && git clone --depth 1 https://github.com/SirVeillanceAi/Imageto3D /opt/imageto3d && bash /opt/imageto3d/scripts/start.sh"
```

## Environment variables (all optional)

| Key | Default | What it does |
|---|---|---|
| `MODELS` | `trellis2,pixal3d,hi3d` | Which model sets to download. Remove one to save disk. |
| `HI3D_PRECISION` | `int8` | `int8` (~4.3 GB, faster) or `fp16` for the Hi3D checkpoint. |
| `PIXAL3D_MULTIVIEW` | `0` | `1` also downloads the Pixal3D multi-view model. |
| `HF_TOKEN` | – | Hugging Face token, only needed if a model repo is gated. Use a RunPod **Secret**. |
| `TORCH_CHANNEL` | `auto` | `auto` picks cu130 on CUDA 13 hosts, cu128 on CUDA 12.8 hosts. |
| `MIN_VRAM_GB` | `10` | Pods with less VRAM refuse to start. |
| `ALLOW_LOW_VRAM` | `0` | `1` lets a smaller GPU start anyway. |
| `UPDATE_ON_START` | `0` | `1` = git pull ComfyUI and the Hi3D node each boot. |
| `COMFY_ARGS` | – | Extra ComfyUI flags. Leave empty unless you need one. |
| `CEI_REF` | tested commit | Set to `MAC-Linux` to use the newest Easy-Install. |

## When deploying a pod

- **GPU:** any NVIDIA GPU with 10 GB+ VRAM (RTX 3080 10GB, A4000, RTX 4070 Ti, A5000, L4, RTX 4090, A40, L40S, RTX 5090, ...).
- **Filter → CUDA Versions: 13.0.** ComfyUI wants Torch cu130, which needs a CUDA 13 driver. 12.8 hosts still work (Torch 2.8 cu128 fallback) but are not recommended.
- Attach a **Network Volume** if you want the install and models to survive deleting the pod.
