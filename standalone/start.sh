#!/usr/bin/env bash
# Imageto3D standalone - RunPod entrypoint.
# Gradio UI on :7860; ComfyUI runs headless on 127.0.0.1:8188 as the GPU engine.
# Installs once into /workspace, then later boots start in a minute or two.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
BASE_DIR="${BASE_DIR:-$WORKSPACE/imageto3d-standalone}"
LOG_DIR="$WORKSPACE/logs"
MIN_VRAM_GB="${MIN_VRAM_GB:-10}"
MODELS="${MODELS:-trellis2,pixal3d,hi3d}"
export PIXAL3D_MULTIVIEW="${PIXAL3D_MULTIVIEW:-1}"   # multi-view tab model
TORCH_CHANNEL="${TORCH_CHANNEL:-auto}"
GRADIO_PORT="${GRADIO_PORT:-7860}"
COMFY_PORT="${COMFY_PORT:-8188}"
EXPOSE_COMFY="${EXPOSE_COMFY:-0}"   # 1 = also serve the ComfyUI interface on port 8188
COMFY_ARGS="${COMFY_ARGS:-}"

source "$REPO_DIR/scripts/common.sh"
source "$REPO_DIR/standalone/filebrowser.sh"
PY="$BASE_DIR/venv/bin/python"

log "Repo: $REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
install_system_packages
start_ssh

# ---- No GPU (e.g. RunPod restarted the pod with "0 GPUs"): viewer + file manager only.
if ! nvidia-smi -L >/dev/null 2>&1; then
    warn "No NVIDIA GPU on this pod - starting in VIEWER-ONLY mode (3D viewer, CPU previews, file manager)."
    warn "Generation needs a GPU: stop the pod and start it again once a GPU is available."
    VPY="$BASE_DIR/venv/bin/python"
    if ! "$VPY" -c "import gradio, trimesh, websocket" 2>/dev/null; then
        VPY="$BASE_DIR/venv-viewer/bin/python"
        if ! "$VPY" -c "import gradio, trimesh, websocket" 2>/dev/null; then
            log "Installing the light viewer environment (no PyTorch)"
            mkdir -p "$BASE_DIR"
            python3.12 -m venv "$BASE_DIR/venv-viewer"
            "$VPY" -m pip install -q --upgrade pip
            "$VPY" -m pip install -q "gradio==6.32.0" "websocket-client==1.9.2" "trimesh==5.2.2" pillow numpy scipy \
                || die "Could not install the viewer environment."
        fi
    fi
    mkdir -p "$BASE_DIR/ComfyUI/output" "$BASE_DIR/ComfyUI/input" "$BASE_DIR/ComfyUI/models"
    start_filebrowser
    export VIEWER_ONLY=1 COMFY_URL="http://127.0.0.1:$COMFY_PORT" COMFY_DIR="$BASE_DIR/ComfyUI" GRADIO_PORT GRADIO_ANALYTICS_ENABLED=False
    log "Starting the viewer on port $GRADIO_PORT  ->  https://${RUNPOD_POD_ID:-<pod-id>}-${GRADIO_PORT}.proxy.runpod.net"
    cd "$REPO_DIR/standalone"
    while true; do
        "$VPY" app.py 2>&1 | tee -a "$LOG_DIR/gradio.log"
        warn "App exited; restarting in 5 s"
        sleep 5
    done
fi

check_gpu

WANT="$(pick_torch_channel)"
[ "$WANT" = "none" ] && die "Host driver supports CUDA ${DRIVER_CUDA}; need 12.8+ (13.0 recommended). In RunPod, filter 'CUDA Versions' to 13.0 when deploying."

if [ ! -f "$BASE_DIR/.installed" ]; then
    log "First boot: installing into $BASE_DIR (about 10 min). Log: $LOG_DIR/standalone-install.log"
    BASE_DIR="$BASE_DIR" TORCH_TARGET="$WANT" bash "$REPO_DIR/standalone/install.sh" 2>&1 | tee -a "$LOG_DIR/standalone-install.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "Install failed - see $LOG_DIR/standalone-install.log"
fi

# Volume moved to a host with a different driver? Swap the Torch build.
HAVE="$("$PY" -c 'import torch;print("cu"+torch.version.cuda.replace(".",""))' 2>/dev/null || echo none)"
if [ "$HAVE" != "$WANT" ]; then
    log "Switching Torch build: $HAVE -> $WANT"
    [ "$WANT" = "cu128" ] && warn "Driver only supports CUDA ${DRIVER_CUDA}; using Torch 2.8.0+cu128. ComfyUI recommends cu130."
    if [ "$WANT" = "cu128" ]; then
        "$PY" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    else
        "$PY" -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
    fi
fi
verify_cuda

# Extras added after the first release: installed on boot so existing volumes upgrade themselves.
"$PY" -c "import trimesh" 2>/dev/null || { log "Installing trimesh (CPU model previews)"; "$PY" -m pip install -q "trimesh==5.2.2"; }

# Models download in the background; the Status tab shows what is ready.
log "Downloading models in the background ($MODELS). Log: $LOG_DIR/models.log"
"$PY" -I "$REPO_DIR/scripts/download_models.py" --comfy "$BASE_DIR/ComfyUI" --models "$MODELS" \
    >> "$LOG_DIR/models.log" 2>&1 &

LISTEN=127.0.0.1
[ "$EXPOSE_COMFY" = "1" ] && LISTEN=0.0.0.0
log "Starting ComfyUI engine on $LISTEN:$COMFY_PORT. Log: $LOG_DIR/comfyui.log"
(
    cd "$BASE_DIR/ComfyUI"
    while true; do
        "$PY" -W ignore::FutureWarning main.py --listen "$LISTEN" --port "$COMFY_PORT" --disable-auto-launch $COMFY_ARGS \
            >> "$LOG_DIR/comfyui.log" 2>&1
        echo "[imageto3d] ComfyUI exited; restarting in 5 s" >> "$LOG_DIR/comfyui.log"
        sleep 5
    done
) &

start_filebrowser

[ -n "${GRADIO_PASSWORD:-}" ] || warn "No GRADIO_USERNAME/GRADIO_PASSWORD set - anyone with the pod's link can use the app."
log "Starting Gradio on port $GRADIO_PORT  ->  https://${RUNPOD_POD_ID:-<pod-id>}-${GRADIO_PORT}.proxy.runpod.net"
export COMFY_URL="http://127.0.0.1:$COMFY_PORT" COMFY_DIR="$BASE_DIR/ComfyUI" GRADIO_PORT GRADIO_ANALYTICS_ENABLED=False
cd "$REPO_DIR/standalone"
while true; do
    "$PY" app.py 2>&1 | tee -a "$LOG_DIR/gradio.log"
    warn "Gradio exited; restarting in 5 s"
    sleep 5
done
