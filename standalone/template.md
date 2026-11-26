# RunPod template settings – Standalone (Gradio)

In RunPod go to **Templates → New Template** and fill in the fields below.

| Field | Value |
|---|---|
| Template Name | `Imageto3D Gradio - TRELLIS.2 / Pixal3D / Hi3D` |
| Template Type | Pod |
| Compute Type | **Nvidia GPU** |
| Container Image | `ghcr.io/sirveillanceai/imageto3d-gradio:latest` *(Option A)* or `ubuntu:24.04` *(Option B)* |
| Container Start Command | Option A: leave empty. Option B: see below |
| Container Disk | `30 GB` |
| Volume Disk | `60 GB` |
| Volume Mount Path | `/workspace` |
| Expose HTTP Ports | `7860` (add `8188` only if you set `EXPOSE_COMFY=1`) |
| Expose TCP Ports | `22` |

### Option B start command (no Docker image needed)

```
bash -c "apt-get update && apt-get install -y git ca-certificates && rm -rf /opt/imageto3d && git clone --depth 1 https://github.com/SirVeillanceAi/Imageto3D /opt/imageto3d && bash /opt/imageto3d/standalone/start.sh"
```

## Environment variables (all optional)

| Key | Default | What it does |
|---|---|---|
| `GRADIO_USERNAME` / `GRADIO_PASSWORD` | – | **Recommended.** Adds a login to the app – the RunPod link is otherwise open to anyone who has it. Use RunPod **Secrets**. |
| `MODELS` | `trellis2,pixal3d,hi3d` | Which model sets to download. |
| `HI3D_PRECISION` | `int8` | `int8` (4.3 GB, faster) or `fp16` (7.7 GB). |
| `HF_TOKEN` | – | Only needed if a Hugging Face repo becomes gated. |
| `TORCH_CHANNEL` | `auto` | `auto` picks cu130 on CUDA 13 hosts, cu128 on CUDA 12.8 hosts. |
| `MIN_VRAM_GB` / `ALLOW_LOW_VRAM` | `10` / `0` | VRAM guard. |
| `EXPOSE_COMFY` | `0` | `1` also serves the ComfyUI interface on port 8188. |
| `COMFY_REF` / `HI3D_REF` | tested commits | Set to `master` / `main` for the newest code. |

## When deploying a pod

- Any NVIDIA GPU with **10 GB+ VRAM**. **Filter → CUDA Versions: 13.0.**
- First boot: about 10 min install, then the app opens while ~23 GB of models download in the background. The **Status** tab shows which models are ready.
- Click **Connect → HTTP 7860** to open the app.
