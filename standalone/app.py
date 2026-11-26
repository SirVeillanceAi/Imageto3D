"""Imageto3D standalone: a Gradio front end for TRELLIS.2, Pixal3D and Hi3D v1.0.

ComfyUI runs headless on localhost as the inference engine (int8 models, dynamic VRAM,
no compiled CUDA extensions); this app builds the graphs, queues them and shows the GLBs.
"""
import hashlib
import hmac
import json
import os
import random
import secrets
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import websocket
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from PIL import Image
from starlette.responses import FileResponse

import pipeline as P
import render_cpu

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = os.environ.get("COMFY_DIR", "/workspace/imageto3d-standalone/ComfyUI")
INPUT_DIR = os.path.join(COMFY_DIR, "input")
OUTPUT_DIR = os.path.join(COMFY_DIR, "output")
MODELS_DIR = os.path.join(COMFY_DIR, "models")
CLIENT_ID = str(uuid.uuid4())
APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_SUBDIR = "viewer_uploads"
POD_ID = os.environ.get("RUNPOD_POD_ID", "")
FILEBROWSER_PORT = os.environ.get("FILEBROWSER_PORT", "8080")
VIEWER_ONLY = os.environ.get("VIEWER_ONLY") == "1"   # pod started without a GPU

REQUIRED = {
    "trellis2": [("diffusion_models", P.TRELLIS2_UNET), ("vae", P.SHAPE_VAE), ("clip_vision", P.DINO_V3)],
    "pixal3d": [("diffusion_models", P.PIXAL3D_UNET), ("vae", P.SHAPE_VAE), ("clip_vision", P.DINO_V3),
                ("geometry_estimation", P.MOGE)],
    "pixal3d_mv": [("diffusion_models", P.PIXAL3D_MV_UNET), ("vae", P.SHAPE_VAE), ("clip_vision", P.DINO_V3)],
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
    if VIEWER_ONLY:
        raise gr.Error("This pod has no GPU, so it runs in viewer-only mode. Stop the pod and start it again "
                       "with a GPU to generate. The 3D Viewer tab and the file manager work now.")
    if not comfy_ready():
        raise gr.Error("The ComfyUI engine is still starting (or crashed). Wait a minute and try again; see /workspace/logs/comfyui.log.")
    missing = missing_models(groups)
    if missing:
        raise gr.Error("Models not downloaded yet: " + ", ".join(missing) + ". See the Status tab.")


def _seed(seed, randomize):
    return random.randint(0, 2**31 - 1) if randomize else int(seed)


# ---------------------------------------------------------------- 3D viewer: signed file links
def _load_secret():
    path = os.path.join(os.path.dirname(COMFY_DIR), ".viewer_secret")
    try:
        with open(path) as f:
            key = f.read().strip()
        if len(key) >= 32:
            return key.encode()
    except OSError:
        pass
    key = secrets.token_hex(32)
    try:
        with open(path, "w") as f:
            f.write(key)
        os.chmod(path, 0o600)
    except OSError:
        pass  # not persisted: links just stop working after a restart
    return key.encode()


SECRET = _load_secret()
OUTPUT_REAL = os.path.realpath(OUTPUT_DIR)


def _sign(rel):
    return hmac.new(SECRET, rel.encode(), hashlib.sha256).hexdigest()[:32]


def _resolve(token, rel):
    """Map a signed link back to a file inside the output folder (404 otherwise)."""
    if not hmac.compare_digest(token, _sign(rel)):
        raise HTTPException(404)
    path = os.path.realpath(os.path.join(OUTPUT_REAL, rel))
    if not path.startswith(OUTPUT_REAL + os.sep) or not os.path.isfile(path):
        raise HTTPException(404)
    return path


def glb_link(path):
    rel = os.path.relpath(os.path.realpath(path), OUTPUT_REAL).replace(os.sep, "/")
    return f"/glb/{_sign(rel)}/{urllib.parse.quote(rel)}"


def _public_base(request):
    h = request.headers if request is not None else {}
    host = h.get("x-forwarded-host") or h.get("host") or (f"{POD_ID}-7860.proxy.runpod.net" if POD_ID else "localhost:7860")
    proto = h.get("x-forwarded-proto") or ("https" if "proxy.runpod.net" in host else "http")
    return f"{proto.split(',')[0].strip()}://{host.split(',')[0].strip()}"


def viewer_links(path, request=None):
    link = glb_link(path)
    viewer = "/viewer?src=" + urllib.parse.quote(link, safe="")
    sandbox = "https://sandbox.babylonjs.com/?assetUrl=" + urllib.parse.quote(_public_base(request) + link, safe="")
    a = lambda href, text: f'<a href="{href}" target="_blank" rel="noopener">{text}</a>'
    return " · ".join([a(viewer, "🔍 Open in 3D viewer"), a(sandbox, "Babylon Sandbox ↗"),
                       a(viewer + "&mode=cpu", "CPU view (no GPU/WebGL)")])


def viewer_iframe(path, height=560):
    if not path:
        return "<div style='padding:40px;text-align:center;opacity:.7'>Pick or upload a model.</div>"
    src = "/viewer?src=" + urllib.parse.quote(glb_link(path), safe="")
    return (f'<iframe src="{src}" style="width:100%;height:{height}px;border:0;border-radius:8px;background:#262626" '
            f'allow="fullscreen; xr-spatial-tracking"></iframe>')


def cpu_previews(path):
    try:
        return [(img, f"{os.path.basename(path)} - {cap}") for img, cap in render_cpu.render_views(path)]
    except render_cpu.NotEnoughMemory as e:
        gr.Warning(str(e))
        return []
    except Exception as e:  # preview is a convenience; never fail the job over it
        gr.Warning(f"CPU preview failed: {e}")
        return []


def after_job(path, request: gr.Request):
    """Runs after a generation: viewer links + CPU-rendered views (shown even without WebGL)."""
    if not path or not os.path.isfile(path):
        return "", []
    return viewer_links(path, request), cpu_previews(path)


def list_models():
    found = []
    for root, _dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f.lower().endswith(".glb"):
                p = os.path.join(root, f)
                found.append((os.path.getmtime(p), os.path.relpath(p, OUTPUT_DIR)))
    return [rel for _t, rel in sorted(found, reverse=True)]


def refresh_models():
    models = list_models()
    return gr.update(choices=models, value=models[0] if models else None)


def _picked(rel):
    path = os.path.realpath(os.path.join(OUTPUT_REAL, rel or ""))
    if not rel or not path.startswith(OUTPUT_REAL + os.sep) or not os.path.isfile(path):
        raise gr.Error("File not found - press Refresh.")
    return path


def view_model(rel, request: gr.Request):
    """Cheap: the 3D view renders in the browser. CPU renders only run when asked for."""
    if not rel:
        return viewer_iframe(None), "", []
    path = _picked(rel)
    return viewer_iframe(path), viewer_links(path, request), []


def render_picked(rel):
    if not rel:
        raise gr.Error("Pick a model first.")
    return cpu_previews(_picked(rel))


def upload_model(file):
    if not file:
        return gr.update()
    if not file.lower().endswith(".glb"):
        raise gr.Error("The viewer takes .glb files.")
    os.makedirs(os.path.join(OUTPUT_DIR, UPLOAD_SUBDIR), exist_ok=True)
    name = os.path.basename(file)
    dest = os.path.join(OUTPUT_DIR, UPLOAD_SUBDIR, name)
    shutil.copy(file, dest)
    rel = os.path.relpath(dest, OUTPUT_DIR)
    return gr.update(choices=list_models(), value=rel)


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
    return preview, all_files, note, seed, preview


def multiview(front, left, back, right, remove_bg, fov, resolution, texture, texture_size, face_count, hi3d,
              seed, randomize, progress=gr.Progress()):
    given = {k: v for k, v in {"front": front, "left": left, "back": back, "right": right}.items() if v}
    if not given:
        raise gr.Error("Upload at least one view (front recommended).")
    hi3d = "off" if hi3d == "Off" else hi3d
    _check(["pixal3d_mv"] + (["texture"] if texture else []) + (["hi3d"] if hi3d != "off" else [])
           + (["background"] if remove_bg else []))
    seed = _seed(seed, randomize)
    run = time.strftime("%Y%m%d-%H%M%S")
    graph = P.multiview_to_3d({k: _stage_image(v) for k, v in given.items()}, remove_background=remove_bg,
                              seed=seed, fov=fov, resolution=int(resolution), texture=texture,
                              texture_size=int(texture_size), face_count=int(face_count), hi3d=hi3d,
                              prefix=f"imageto3d/{run}_multiview")
    t0 = time.time()
    files = run_graph(graph, progress)
    order = ["save_textured"] + [f"save_hi3d_{r}" for r in reversed(P.HI3D_LADDER)] + ["save_shape"]
    all_files = [f for k in order for f in files.get(k, [])]
    if not all_files:
        raise gr.Error("The job finished but produced no GLB file - check /workspace/logs/comfyui.log.")
    note = f"Done in {time.time() - t0:.0f} s - Pixal3D multi-view ({', '.join(given)}), seed {seed}."
    if hi3d != "off":
        note += " Hi3D files are geometry only (no texture)."
    return all_files[0], all_files, note, seed, all_files[0]


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
    return (all_files[0], all_files, f"Done in {time.time() - t0:.0f} s, seed {seed}. Geometry only (no texture).",
            seed, all_files[0])


def status():
    lines = ["### Engine", f"- ComfyUI at `{COMFY_URL}`: " + ("**running**" if comfy_ready() else "**not reachable yet**")]
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        lines.append(f"- GPU: {gpu or 'none detected'}")
    except Exception:
        lines.append("- GPU: nvidia-smi not available")
    fb = f"https://{POD_ID}-{FILEBROWSER_PORT}.proxy.runpod.net" if POD_ID else f"port {FILEBROWSER_PORT}"
    lines.append(f"- File manager (File Browser): {fb} - login shown in the pod logs and saved in "
                 "`/workspace/.filebrowser/login.txt`")
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
    if VIEWER_ONLY:
        gr.Markdown("⚠️ **Viewer-only mode: this pod has no GPU.** Generation is off. Use the **3D Viewer** tab "
                    "(CPU view works without a GPU) and the file manager on port "
                    f"{FILEBROWSER_PORT}. To generate, stop the pod and start it again once a GPU is available.")

    with gr.Tab("Image → 3D"):
        with gr.Row():
            with gr.Column(scale=1):
                in_image = gr.Image(label="Input image", type="filepath", image_mode=None, height=320)
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
                out_links = gr.Markdown()
                out_cpu = gr.Gallery(label="CPU preview (works without a GPU or WebGL)", columns=4, height=200,
                                     object_fit="contain")
                out_path = gr.State()
        btn_gen.click(generate, [in_image, in_model, in_bg, in_res, in_tex, in_texsize, in_faces, in_hi3d,
                                 in_seed, in_rand], [out_model, out_files, out_note, in_seed, out_path]) \
            .then(after_job, out_path, [out_links, out_cpu])

    with gr.Tab("Multi-view → 3D (Pixal3D)"):
        gr.Markdown("Photos of the **same object** from views 90° apart: front, left side, back, right side. "
                    "Any subset works; front is recommended. Keep the object about the same size in every photo.")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    mv_front = gr.Image(label="Front", type="filepath", image_mode=None, height=200)
                    mv_left = gr.Image(label="Left side", type="filepath", image_mode=None, height=200)
                with gr.Row():
                    mv_back = gr.Image(label="Back", type="filepath", image_mode=None, height=200)
                    mv_right = gr.Image(label="Right side", type="filepath", image_mode=None, height=200)
                examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
                if os.path.isfile(os.path.join(examples_dir, "tailfin_front.jpg")):
                    gr.Examples([[os.path.join(examples_dir, f"tailfin_{v}.jpg") if v else None
                                  for v in ("front", None, "back", "right")]],
                                inputs=[mv_front, mv_left, mv_back, mv_right], label="Example: marble whale tail")
                mv_bg = gr.Checkbox(value=True, label="Remove background (BiRefNet)")
                mv_fov = gr.Slider(5, 90, value=20, step=1, label="Camera field of view (degrees)",
                                   info="20 suits most photos taken from a distance; raise it for close-up wide-angle shots.")
                mv_res = gr.Radio(["1024", "1536"], value="1024", label="Shape resolution")
                mv_tex = gr.Checkbox(value=True, label="PBR texture")
                with gr.Accordion("Texture & mesh settings", open=False):
                    mv_texsize = gr.Radio(["1024", "2048", "4096"], value="2048", label="Texture size")
                    mv_faces = gr.Slider(50000, 1500000, value=700000, step=10000, label="Textured mesh face count")
                mv_hi3d = gr.Radio(["Off", "512", "1024", "1536"], value="Off", label="Hi3D refine (geometry only)")
                with gr.Row():
                    mv_seed = gr.Number(value=42, precision=0, label="Seed")
                    mv_rand = gr.Checkbox(value=False, label="Random seed")
                btn_mv = gr.Button("Generate 3D from views", variant="primary")
            with gr.Column(scale=1):
                mv_out = gr.Model3D(label="Preview", height=480, clear_color=(0.15, 0.15, 0.15, 1.0))
                mv_files = gr.File(label="Download GLB files", file_count="multiple")
                mv_note = gr.Markdown()
                mv_links = gr.Markdown()
                mv_cpu = gr.Gallery(label="CPU preview (works without a GPU or WebGL)", columns=4, height=200,
                                    object_fit="contain")
                mv_path = gr.State()
        btn_mv.click(multiview, [mv_front, mv_left, mv_back, mv_right, mv_bg, mv_fov, mv_res, mv_tex, mv_texsize,
                                 mv_faces, mv_hi3d, mv_seed, mv_rand], [mv_out, mv_files, mv_note, mv_seed, mv_path]) \
            .then(after_job, mv_path, [mv_links, mv_cpu])

    with gr.Tab("Hi3D refine (your mesh)"):
        gr.Markdown("Bring a coarse mesh and a reference image of the **same object**. "
                    "Hi3D re-samples the geometry against the image and returns a denser, untextured mesh.")
        with gr.Row():
            with gr.Column(scale=1):
                r_mesh = gr.File(label="Coarse mesh (.glb / .gltf / .obj / .stl)", file_types=[".glb", ".gltf", ".obj", ".stl"], type="filepath")
                r_image = gr.Image(label="Reference image", type="filepath", image_mode=None, height=280)
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
                r_links = gr.Markdown()
                r_cpu = gr.Gallery(label="CPU preview (works without a GPU or WebGL)", columns=4, height=200,
                                   object_fit="contain")
                r_path = gr.State()
        btn_ref.click(refine, [r_mesh, r_image, r_target, r_bg, r_seed, r_rand], [r_out, r_files, r_note, r_seed, r_path]) \
            .then(after_job, r_path, [r_links, r_cpu])

    with gr.Tab("3D Viewer"):
        gr.Markdown("Browse every model in the output folder, or upload a .glb. The 3D view uses the "
                    "[Babylon.js viewer](https://doc.babylonjs.com/features/featuresDeepDive/babylonViewer) in your browser. "
                    "On a device without a GPU or WebGL, the viewer switches to **CPU view**, which the pod renders on its own CPU.")
        with gr.Row():
            v_pick = gr.Dropdown(label="Model (newest first)", choices=[], scale=4, allow_custom_value=False)
            v_refresh = gr.Button("Refresh", scale=1)
            v_upload = gr.UploadButton("Upload .glb", file_types=[".glb"], type="filepath", scale=1)
        v_frame = gr.HTML(viewer_iframe(None))
        v_links = gr.Markdown()
        v_render = gr.Button("Render CPU preview images (front / 90° / back / 270°)")
        v_cpu = gr.Gallery(label="CPU preview", columns=4, height=220, object_fit="contain")
        v_refresh.click(refresh_models, None, v_pick)
        v_upload.upload(upload_model, v_upload, v_pick)
        v_pick.change(view_model, v_pick, [v_frame, v_links, v_cpu])
        v_render.click(render_picked, v_pick, v_cpu)
        demo.load(refresh_models, None, v_pick)

    with gr.Tab("Status"):
        status_md = gr.Markdown()
        gr.Button("Refresh").click(status, None, status_md)
        demo.load(status, None, status_md)


# ---------------------------------------------------------------- web routes (viewer, signed files, CPU renders)
web = FastAPI()
with open(os.path.join(APP_DIR, "viewer.html"), encoding="utf-8") as _f:
    VIEWER_HTML = _f.read()
_png_cache = OrderedDict()


@web.get("/viewer", response_class=HTMLResponse)
def viewer_page():
    return HTMLResponse(VIEWER_HTML, headers={"Cache-Control": "no-cache"})


@web.get("/glb/{token}/{rel:path}")
def glb_file(token: str, rel: str):
    path = _resolve(token, rel)
    # CORS lets sandbox.babylonjs.com fetch the file; the unguessable token is the access check.
    return FileResponse(path, media_type="model/gltf-binary", filename=os.path.basename(path),
                        content_disposition_type="inline",
                        headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "private, max-age=3600"})


@web.get("/glb-render/{token}/{rel:path}")
def glb_render(token: str, rel: str, az: float = 0, el: float = 15, size: int = 768):
    path = _resolve(token, rel)
    az, el = round(az / 15) * 15 % 360, max(-60, min(75, round(el / 15) * 15))
    size = max(256, min(1024, int(size) // 128 * 128))
    key = (path, os.path.getmtime(path), az, el, size)
    png = _png_cache.get(key)
    if png is None:
        try:
            png = render_cpu.render_png(path, az, el, size)
        except render_cpu.NotEnoughMemory as e:
            raise HTTPException(507, str(e))
        except Exception as e:
            raise HTTPException(500, f"render failed: {e}")
        _png_cache[key] = png
        while len(_png_cache) > 96:
            _png_cache.popitem(last=False)
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


if __name__ == "__main__":
    import uvicorn

    user, pw = os.environ.get("GRADIO_USERNAME"), os.environ.get("GRADIO_PASSWORD")
    demo.queue(default_concurrency_limit=1)
    app = gr.mount_gradio_app(
        web, demo, path="/", auth=(user, pw) if user and pw else None,
        allowed_paths=[OUTPUT_DIR, os.path.join(APP_DIR, "examples")], show_error=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("GRADIO_PORT", 7860)), log_level="warning")
