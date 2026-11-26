"""Imageto3D standalone: a Gradio front end for TRELLIS.2, Pixal3D and Hi3D v1.0.

ComfyUI runs headless on localhost as the inference engine (int8 models, dynamic VRAM,
no compiled CUDA extensions); this app builds the graphs, queues them and shows the GLBs.
"""
import json
import os
import random
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import websocket
from PIL import Image

import pipeline as P

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = os.environ.get("COMFY_DIR", "/workspace/imageto3d-standalone/ComfyUI")
INPUT_DIR = os.path.join(COMFY_DIR, "input")
OUTPUT_DIR = os.path.join(COMFY_DIR, "output")
MODELS_DIR = os.path.join(COMFY_DIR, "models")
CLIENT_ID = str(uuid.uuid4())

REQUIRED = {
    "trellis2": [("diffusion_models", P.TRELLIS2_UNET), ("vae", P.SHAPE_VAE), ("clip_vision", P.DINO_V3)],
    "pixal3d": [("diffusion_models", P.PIXAL3D_UNET), ("vae", P.SHAPE_VAE), ("clip_vision", P.DINO_V3),
                ("geometry_estimation", P.MOGE)],
    "texture": [("vae", P.TEXTURE_VAE)],
    "background": [("background_removal", P.BIREFNET)],
    "hi3d": [("checkpoints", P.HI3D_CKPT), ("clip_vision", P.DINO_V2), ("clip_vision", P.CLIP_L)],
}


# ---------------------------------------------------------------- ComfyUI client
def _request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(COMFY_URL + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"{}")


def comfy_ready():
    try:
        _request("/system_stats")
        return True
    except Exception:
        return False


def missing_models(groups):
    return [f"{folder}/{name}" for g in groups for folder, name in REQUIRED[g]
            if not os.path.isfile(os.path.join(MODELS_DIR, folder, name))]


def run_graph(graph, progress):
    """Queue a graph, report progress, return {save_node_id: [glb paths]}."""
    labels = dict(P.LABELS)
    ws = websocket.create_connection(f"{COMFY_URL.replace('http', 'ws', 1)}/ws?clientId={CLIENT_ID}", timeout=5)
    try:
        try:
            prompt_id = _request("/prompt", {"prompt": graph, "client_id": CLIENT_ID})["prompt_id"]
        except urllib.error.HTTPError as e:
            detail = json.loads(e.read() or b"{}")
            errs = [f"{labels.get(n, n)}: {x['message']} {x.get('details', '')}"
                    for n, v in detail.get("node_errors", {}).items() for x in v.get("errors", [])]
            raise gr.Error("ComfyUI rejected the job: " + ("; ".join(errs) or detail.get("error", {}).get("message", str(e))))

        done_nodes, total = set(), len(graph)
        current = "Queued"
        progress(0, desc=current)
        while True:
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not isinstance(msg, str):
                continue  # binary preview frames
            m = json.loads(msg)
            data = m.get("data", {})
            if data.get("prompt_id") not in (None, prompt_id):
                continue
            t = m["type"]
            if t == "executing":
                if data.get("node") is None:
                    break
                done_nodes.add(data["node"])
                current = labels.get(data["node"], data["node"])
                progress(len(done_nodes) / total, desc=current)
            elif t == "execution_cached":
                done_nodes.update(data.get("nodes", []))
            elif t == "progress" and data.get("max"):
                progress(len(done_nodes) / total, desc=f"{current} ({data['value']}/{data['max']})")
            elif t == "execution_error":
                text = f"{labels.get(data.get('node_id'), data.get('node_type'))}: {data.get('exception_message', '').strip()}"
                if "out of memory" in text.lower():
                    text += "  - try a lower resolution, smaller texture size, or a GPU with more VRAM."
                raise gr.Error(text)
            elif t == "execution_interrupted":
                raise gr.Error("Job was cancelled.")
            elif t == "execution_success":
                break
    finally:
        ws.close()

    outputs = _request(f"/history/{prompt_id}").get(prompt_id, {}).get("outputs", {})
    files = {}
    for node_id, out in outputs.items():
        for items in out.values():
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict) and str(item.get("filename", "")).endswith(".glb"):
                    files.setdefault(node_id, []).append(
                        os.path.join(OUTPUT_DIR, item.get("subfolder", ""), item["filename"]))
    return files


# ---------------------------------------------------------------- helpers
def _stage_image(path):
    name = f"imageto3d_{uuid.uuid4().hex[:10]}.png"
    Image.open(path).save(os.path.join(INPUT_DIR, name))  # PNG keeps any alpha channel
    return name


def _stage_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".glb", ".gltf", ".obj", ".stl"):
        raise gr.Error("Mesh must be .glb, .gltf, .obj or .stl")
    name = f"imageto3d_{uuid.uuid4().hex[:10]}{ext}"
    os.makedirs(os.path.join(INPUT_DIR, "3d"), exist_ok=True)
    shutil.copy(path, os.path.join(INPUT_DIR, "3d", name))
    return name


def _check(groups):
    if not comfy_ready():
        raise gr.Error("The ComfyUI engine is still starting (or crashed). Wait a minute and try again; see /workspace/logs/comfyui.log.")
    missing = missing_models(groups)
    if missing:
        raise gr.Error("Models not downloaded yet: " + ", ".join(missing) + ". See the Status tab.")


def _seed(seed, randomize):
    return random.randint(0, 2**31 - 1) if randomize else int(seed)


# ---------------------------------------------------------------- tab handlers
def generate(image, model, remove_bg, resolution, texture, texture_size, face_count, hi3d, seed, randomize,
             progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload an image first.")
    model = "trellis2" if model.startswith("TRELLIS") else "pixal3d"
    hi3d = "off" if hi3d == "Off" else hi3d
    groups = [model] + (["texture"] if texture else []) + (["hi3d"] if hi3d != "off" else []) \
        + (["background"] if remove_bg else [])
    _check(groups)
    seed = _seed(seed, randomize)
    run = time.strftime("%Y%m%d-%H%M%S")
    graph = P.image_to_3d(_stage_image(image), model=model, remove_background=remove_bg, seed=seed,
                          resolution=int(resolution), texture=texture, texture_size=int(texture_size),
                          face_count=int(face_count), hi3d=hi3d, prefix=f"imageto3d/{run}_{model}")
    t0 = time.time()
    files = run_graph(graph, progress)

    order = ["save_textured"] + [f"save_hi3d_{r}" for r in reversed(P.HI3D_LADDER)] + ["save_shape"]
    all_files = [f for k in order for f in files.get(k, [])]
    if not all_files:
        raise gr.Error("The job finished but produced no GLB file - check /workspace/logs/comfyui.log.")
    # Show the textured mesh if made, otherwise the most refined geometry
    preview = all_files[0]
    note = f"Done in {time.time() - t0:.0f} s - {('TRELLIS.2' if model == 'trellis2' else 'Pixal3D')}, seed {seed}."
    if hi3d != "off":
        note += " Hi3D files are geometry only (no texture)."
    return preview, all_files, note, seed


def refine(mesh, image, target, remove_bg, seed, randomize, progress=gr.Progress()):
    if mesh is None or image is None:
        raise gr.Error("Upload both a mesh and the reference image of the same object.")
    _check(["hi3d", "background"] if remove_bg else ["hi3d"])
    seed = _seed(seed, randomize)
    run = time.strftime("%Y%m%d-%H%M%S")
    graph = P.refine_mesh(_stage_image(image), _stage_mesh(mesh), target=target, remove_background=remove_bg,
                          seed=seed, prefix=f"imageto3d/{run}_refine")
    t0 = time.time()
    files = run_graph(graph, progress)
    all_files = [f for r in reversed(P.HI3D_LADDER) for f in files.get(f"save_hi3d_{r}", [])]
    if not all_files:
        raise gr.Error("The job finished but produced no GLB file - check /workspace/logs/comfyui.log.")
    return all_files[0], all_files, f"Done in {time.time() - t0:.0f} s, seed {seed}. Geometry only (no texture).", seed


def status():
    lines = ["### Engine", f"- ComfyUI at `{COMFY_URL}`: " + ("**running**" if comfy_ready() else "**not reachable yet**")]
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        lines.append(f"- GPU: {gpu or 'none detected'}")
    except Exception:
        lines.append("- GPU: nvidia-smi not available")
    lines.append("\n### Models")
    seen = set()
    for group, files in REQUIRED.items():
        for folder, name in files:
            if (folder, name) in seen:
                continue
            seen.add((folder, name))
            ok = os.path.isfile(os.path.join(MODELS_DIR, folder, name))
            lines.append(f"- {'✅' if ok else '⏳ missing'} `{folder}/{name}` ({group})")
    return "\n".join(lines)


# ---------------------------------------------------------------- UI
with gr.Blocks(title="Imageto3D - TRELLIS.2 / Pixal3D / Hi3D") as demo:
    gr.Markdown("# Imageto3D\nImage → 3D with **TRELLIS.2** or **Pixal3D**, plus **Hi3D v1.0** detail refinement. "
                "Runs on the GPU; one job at a time.")

    with gr.Tab("Image → 3D"):
        with gr.Row():
            with gr.Column(scale=1):
                in_image = gr.Image(label="Input image", type="filepath", image_mode="RGBA", height=320)
                in_model = gr.Radio(["TRELLIS.2", "Pixal3D"], value="TRELLIS.2", label="Model",
                                    info="Pixal3D keeps the front view pixel-aligned to your image.")
                in_bg = gr.Checkbox(value=True, label="Remove background (BiRefNet)",
                                    info="Turn off if your image already has a transparent background.")
                in_res = gr.Radio(["1024", "1536"], value="1024", label="Shape resolution",
                                  info="1536 = more detail, more VRAM and time.")
                in_tex = gr.Checkbox(value=True, label="PBR texture (base color, metal, roughness, normal, AO)")
                with gr.Accordion("Texture & mesh settings", open=False):
                    in_texsize = gr.Radio(["1024", "2048", "4096"], value="2048", label="Texture size")
                    in_faces = gr.Slider(50000, 1500000, value=700000, step=10000, label="Textured mesh face count")
                in_hi3d = gr.Radio(["Off", "512", "1024", "1536"], value="Off", label="Hi3D refine (geometry only)",
                                   info="Adds fine detail to the untextured mesh: 512 → 1024 → 1536.")
                with gr.Row():
                    in_seed = gr.Number(value=42, precision=0, label="Seed")
                    in_rand = gr.Checkbox(value=False, label="Random seed")
                btn_gen = gr.Button("Generate 3D", variant="primary")
            with gr.Column(scale=1):
                out_model = gr.Model3D(label="Preview", height=480, clear_color=(0.15, 0.15, 0.15, 1.0))
                out_files = gr.File(label="Download GLB files", file_count="multiple")
                out_note = gr.Markdown()
        btn_gen.click(generate, [in_image, in_model, in_bg, in_res, in_tex, in_texsize, in_faces, in_hi3d,
                                 in_seed, in_rand], [out_model, out_files, out_note, in_seed])

    with gr.Tab("Hi3D refine (your mesh)"):
        gr.Markdown("Bring a coarse mesh and a reference image of the **same object**. "
                    "Hi3D re-samples the geometry against the image and returns a denser, untextured mesh.")
        with gr.Row():
            with gr.Column(scale=1):
                r_mesh = gr.File(label="Coarse mesh (.glb / .gltf / .obj / .stl)", file_types=[".glb", ".gltf", ".obj", ".stl"], type="filepath")
                r_image = gr.Image(label="Reference image", type="filepath", image_mode="RGBA", height=280)
                r_target = gr.Radio(["512", "1024", "1536"], value="1536", label="Refine up to")
                r_bg = gr.Checkbox(value=True, label="Remove background (BiRefNet)")
                with gr.Row():
                    r_seed = gr.Number(value=42, precision=0, label="Seed")
                    r_rand = gr.Checkbox(value=False, label="Random seed")
                btn_ref = gr.Button("Refine with Hi3D", variant="primary")
            with gr.Column(scale=1):
                r_out = gr.Model3D(label="Preview", height=480, clear_color=(0.15, 0.15, 0.15, 1.0))
                r_files = gr.File(label="Download GLB files", file_count="multiple")
                r_note = gr.Markdown()
        btn_ref.click(refine, [r_mesh, r_image, r_target, r_bg, r_seed, r_rand], [r_out, r_files, r_note, r_seed])

    with gr.Tab("Status"):
        status_md = gr.Markdown()
        gr.Button("Refresh").click(status, None, status_md)
        demo.load(status, None, status_md)


if __name__ == "__main__":
    user, pw = os.environ.get("GRADIO_USERNAME"), os.environ.get("GRADIO_PASSWORD")
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_PORT", 7860)),
        auth=(user, pw) if user and pw else None, allowed_paths=[OUTPUT_DIR], show_error=True)
