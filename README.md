# Imageto3D – RunPod templates

Two RunPod templates for image → 3D on a rented GPU (10 GB+ VRAM):

| Template | You get | Setup guide |
|---|---|---|
| **ComfyUI (Easy-Install)** | Full ComfyUI with node graphs to edit | [runpod/template.md](runpod/template.md) |
| **Standalone (Gradio)** | A simple web app: upload image → download GLB. No node graphs | [standalone/template.md](standalone/template.md) |

Both run the same models the same way, so results match.

---

## Standalone Gradio app

- **Image → 3D:** upload a picture, pick **TRELLIS.2** or **Pixal3D**, optionally bake a PBR texture (base color, metal, roughness, normal, AO), optionally run **Hi3D** refinement (512 → 1024 → 1536). Preview in the browser and download every GLB.
- **Multi-view → 3D (Pixal3D):** up to four photos of the same object (front, left, back, right, 90° apart) for a more accurate shape. Includes a one-click example (a marble whale tail: front, right and back).
- **Hi3D refine (your mesh):** upload your own coarse mesh (.glb/.gltf/.obj/.stl) plus a reference image of the same object to get a much denser mesh.
- **3D Viewer:** every GLB in the output folder (or one you upload) in the [Babylon.js viewer](https://doc.babylonjs.com/features/featuresDeepDive/babylonViewer), with a one-click **Babylon Sandbox** link (inspector, materials, measurements).
- **CPU view (no GPU / no WebGL):** 3D viewers run in *your* browser and need WebGL. Computers without a GPU often have none, because Chrome is dropping its software (SwiftShader) fallback. The app then renders the model on the pod's CPU (`standalone/render_cpu.py`) and shows still views you can turn 45° at a time. After every job you also get four CPU-rendered views (front, 90°, back, 270°).
- **Status:** GPU, engine and model-download status, plus the file-manager link.
- **File manager:** [File Browser](https://filebrowser.org) on port **8080** serves `/workspace`, so you can download, upload, rename, delete and zip outputs, inputs, models and logs. The login is your `FILEBROWSER_*` or `GRADIO_*` username and password. If you set neither, the user is `admin` with a random password, printed once in the pod logs and saved in `/workspace/.filebrowser/login.txt`.

Under the hood a headless, pinned ComfyUI runs on `127.0.0.1:8188` as the GPU engine. That is the only way to run all three models on a 10 GB card: the official TRELLIS.2, Pixal3D and Hi3D code each need their own Python/CUDA stack, and official Hi3D was only validated on a 96 GB GPU. ComfyUI's int8 versions and memory manager make them fit. The graphs in `standalone/pipeline.py` copy the node settings from ComfyUI's official Pixal3D/TRELLIS.2 template and PixelArtistry's Hi3D workflows.

Set `GRADIO_USERNAME` and `GRADIO_PASSWORD` on the pod – otherwise anyone with the RunPod link can use the app.

Files: `standalone/app.py` (Gradio UI + ComfyUI client + viewer routes), `standalone/pipeline.py` (graphs), `standalone/viewer.html` (Babylon viewer page), `standalone/render_cpu.py` (CPU previews), `standalone/filebrowser.sh` (file manager), `standalone/install.sh`, `standalone/start.sh`, `standalone/Dockerfile`, `standalone/template.md`. Shared GPU/driver checks live in `scripts/common.sh`; models use `scripts/download_models.py`.

---

## ComfyUI (Easy-Install) template

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
3. The first boot installs everything into `/workspace` (about 15–30 min plus about 23 GB of model downloads). Follow along in the pod's **Logs** tab.
4. When the log says `Starting ComfyUI`, click **Connect → HTTP 8188**.
5. Later boots skip the install and start in a minute or two.

## Using the models

Everything is in the ComfyUI workflow sidebar under **Imageto3D**:

| Workflow | What it does |
|---|---|
| `PixelArtistry_Hi3D_01_Image_to_Hi3D` | **Start here.** Upload an image → TRELLIS.2 or Pixal3D (one switch) builds the coarse mesh → Hi3D refines 512 → 1024 → 1536 |
| `PixelArtistry_Hi3D_02_Your_Mesh_Refine_Chain` | Your own mesh + its image → Hi3D 512 → 1024 → 1536 |
| `PixelArtistry_Hi3D_03_Your_Mesh_Single_Pass` | One Hi3D pass at the resolution you pick |
| `Hi3D Refine Chain` / `Hi3D Refine Direct` | AeroX's original example workflows |

For TRELLIS.2 / Pixal3D alone (with textures), use **Templates → 3D → "Pixal3D: Image to Model"**. Outputs save to `ComfyUI/output/3d/`.

Workflows 01–03 come from [PixelArtistry-Hi3D](https://github.com/pixelartistry/PixelArtistry-Hi3D) (MIT) – see [pixel-artistry.com/Hi3D](https://pixel-artistry.com/Hi3D/). Their workflow 04 is left out because it needs the UltraTex and Animated-Characters node packs.

## Files

| Path | Purpose |
|---|---|
| `scripts/start.sh` | Pod entrypoint: GPU checks, first-boot install, Torch/driver match, model check, starts ComfyUI on port 8188 |
| `scripts/install.sh` | Runs `ComfyUI-Easy-Install.sh` unattended, adds the Hi3D node (pinned to `227d18f`, the version PixelArtistry tested) and the PixelArtistry workflows |
| `scripts/common.sh` | GPU/VRAM/driver checks, SSH and system packages shared by both templates |
| `scripts/download_models.py` | Downloads model files into the right ComfyUI folders; skips ones already there |
| `docker/Dockerfile` | Small Ubuntu 24.04 image with the system packages pre-installed |
| `.github/workflows/docker.yml` | Rebuilds both images (`imageto3d` and `imageto3d-gradio`) on every push |
| `runpod/template.md` | Exact values for the RunPod template form |

## Notes

- **Why not the Easy-Install "Trellis2" add-on?** It installs an older custom node with compiled CUDA wheels that only support Torch 2.7–2.9 and pins numpy 1.26. ComfyUI now runs TRELLIS.2 and Pixal3D natively in pure PyTorch, so this template keeps Easy-Install's Torch 2.11 + CUDA 13.0 and skips that add-on.
- **Torch vs. driver:** Easy-Install ships Torch 2.11.0+cu130 (needs a CUDA 13 driver). If a pod lands on a CUDA 12.8 host, `start.sh` switches to Torch 2.8.0+cu128 using Easy-Install's Torch-Pack, and switches back automatically on a CUDA 13 host.
- **ComfyUI version:** Hi3D nodes need ComfyUI 0.42.0+ (Easy-Install pulls the latest, 0.46.0 at time of writing); the installer checks and updates if needed.
- **VRAM:** all three models ship int8 checkpoints and run through ComfyUI's dynamic VRAM management, so no `--lowvram` flag is used. The Hi3D node author measured the full 512→1536 chain on a 6 GB card.
- **Disk:** Easy-Install also installs its standard Pixaroma node set (WanVideoWrapper, KJNodes, etc.). Install + models need roughly 60–70 GB, so a 100 GB volume leaves room for outputs.
- **Licenses:** Pixal3D and Hi3D have their own model licenses (Hi3D Open License v1.0). Check them before commercial use.
- **Logs:** `/workspace/logs/install.log`, `models.log`, `comfyui.log` (standalone adds `standalone-install.log`, `gradio.log`).

## GitHub basics for this repo

- **Changing a script:** open the file on github.com, click the pencil icon, edit, then **Commit changes**. That's it – new pods pick it up.
- **Option A (Docker image):** each commit to `main` triggers the **Actions** tab to rebuild the image (about 5 min). The first time, go to your GitHub profile → **Packages**, open **imageto3d** and **imageto3d-gradio**, and for each one choose **Package settings → Change visibility → Public**, so RunPod can download them without a login.
- **Option B (no image):** the start command clones this repo fresh on every boot, so edits apply on the next pod restart.
