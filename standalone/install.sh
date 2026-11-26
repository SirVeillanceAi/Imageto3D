#!/usr/bin/env bash
# Standalone install: Python 3.12 venv + pinned ComfyUI (headless engine) + Hi3D nodes + Gradio.
# No ComfyUI-Easy-Install and no extra node packs - only what TRELLIS.2, Pixal3D and Hi3D need.

set -euo pipefail

BASE_DIR="${BASE_DIR:-/workspace/imageto3d-standalone}"
TORCH_TARGET="${TORCH_TARGET:-cu130}"
# ComfyUI 0.46.0 (2026-11-26): native TRELLIS.2 / Pixal3D nodes, tested with this app's graphs.
COMFY_REF="${COMFY_REF:-8b0f107157b58f9ec79a19fb3182e78cb30f0a23}"
# Hi3D nodes at the version PixelArtistry-Hi3D tested.
HI3D_REF="${HI3D_REF:-227d18f}"

log() { echo -e "\033[92m[install]\033[0m $*"; }

if [ -e "$BASE_DIR" ] && [ ! -f "$BASE_DIR/.installed" ]; then
    mv "$BASE_DIR" "$BASE_DIR.failed-$(date +%Y%m%d-%H%M%S)"
    log "Moved incomplete install aside"
fi
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

log "Creating Python 3.12 venv"
python3.12 -m venv venv
PY="$BASE_DIR/venv/bin/python"
"$PY" -m pip install --upgrade pip wheel

log "Installing PyTorch ($TORCH_TARGET)"
if [ "$TORCH_TARGET" = "cu128" ]; then
    "$PY" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
else
    "$PY" -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
fi

log "Installing ComfyUI @ $COMFY_REF"
git clone https://github.com/Comfy-Org/ComfyUI ComfyUI
git -C ComfyUI checkout -q "$COMFY_REF"
"$PY" -m pip install -r ComfyUI/requirements.txt

log "Installing Hi3D nodes @ $HI3D_REF"
git clone https://github.com/Aero-Ex/Aero-Ex-ComfyUI-Hi3D ComfyUI/custom_nodes/ComfyUI-Hi3D
git -C ComfyUI/custom_nodes/ComfyUI-Hi3D checkout -q "$HI3D_REF"

log "Installing Gradio app dependencies"
"$PY" -m pip install "gradio==6.32.0" "websocket-client==1.9.2" "huggingface_hub>=0.30" hf_transfer pillow "trimesh==5.2.2"

touch "$BASE_DIR/.installed"
log "Install complete"
