#!/usr/bin/env bash
# Runs the ComfyUI-Easy-Install Linux installer unattended, then adds the Hi3D v1.0 node.
# TRELLIS.2 and Pixal3D are native ComfyUI nodes (ComfyUI >= 0.34.0, no compiled CUDA
# extensions), so the Easy-Install "Trellis2" add-on is NOT used - it would force Torch 2.9
# wheels and numpy 1.26 and is replaced by the native nodes.

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/workspace/ComfyUI-Easy-Install}"
CEI_REPO="https://github.com/Tavris1/ComfyUI-Easy-Install"
# Tested commit of the MAC-Linux branch (v3.30.3, 2026-11-19). Set CEI_REF=MAC-Linux for latest.
CEI_REF="${CEI_REF:-ab2d28c6b936d9745a4127567d795043e15608f6}"
HI3D_REPO="https://github.com/Aero-Ex/Aero-Ex-ComfyUI-Hi3D"
# Version tested by PixelArtistry-Hi3D (https://pixel-artistry.com/Hi3D/). Set HI3D_REF=main for latest.
HI3D_REF="${HI3D_REF:-227d18f}"
PA_REPO="https://github.com/pixelartistry/PixelArtistry-Hi3D"
MIN_COMFY="0.42.0"   # Hi3D nodes need >= 0.42.0; native TRELLIS.2/Pixal3D need >= 0.34.0

log() { echo -e "\033[92m[install]\033[0m $*"; }

# A half-finished previous attempt is moved aside, never deleted.
if [ -e "$INSTALL_DIR" ] && [ ! -f "$INSTALL_DIR/.imageto3d_installed" ]; then
    mv "$INSTALL_DIR" "$INSTALL_DIR.failed-$(date +%Y%m%d-%H%M%S)"
    log "Moved incomplete install aside"
fi

log "Cloning ComfyUI-Easy-Install (MAC-Linux @ $CEI_REF)"
git clone -b MAC-Linux --single-branch "$CEI_REPO" "$INSTALL_DIR"
git -C "$INSTALL_DIR" checkout -q "$CEI_REF"

log "Running ComfyUI-Easy-Install.sh (unattended)"
cd "$INSTALL_DIR"
chmod +x ComfyUI-Easy-Install.sh
# stdin from /dev/null answers its "press any key" prompts
bash ./ComfyUI-Easy-Install.sh </dev/null || true

PY="$INSTALL_DIR/python_embeded/python"
[ -x "$PY" ] && [ -f "$INSTALL_DIR/ComfyUI/main.py" ] || { echo "Easy-Install did not finish (python_embeded or ComfyUI missing)"; exit 1; }

COMFY_VER="$(grep -oP '__version__ = "\K[^"]+' ComfyUI/comfyui_version.py)"
log "ComfyUI $COMFY_VER"
if [ "$(printf '%s\n%s\n' "$COMFY_VER" "$MIN_COMFY" | sort -V | head -n1)" != "$MIN_COMFY" ]; then
    git -C ComfyUI pull --ff-only
    "$PY" -I -m pip install --no-warn-script-location -r ComfyUI/requirements.txt
fi
[ -f ComfyUI/comfy_extras/nodes_trellis2.py ] || { echo "This ComfyUI has no native TRELLIS.2 / Pixal3D nodes"; exit 1; }
log "Native TRELLIS.2 / Pixal3D nodes present"

log "Installing Hi3D v1.0 node (Aero-Ex, dependency-free)"
rm -rf ComfyUI/custom_nodes/ComfyUI-Hi3D
git clone "$HI3D_REPO" ComfyUI/custom_nodes/ComfyUI-Hi3D
git -C ComfyUI/custom_nodes/ComfyUI-Hi3D checkout -q "$HI3D_REF"

log "Fetching PixelArtistry Hi3D workflows (image -> TRELLIS.2/Pixal3D -> Hi3D)"
rm -rf Add-Ons/PixelArtistry-Hi3D
git clone --depth 1 "$PA_REPO" Add-Ons/PixelArtistry-Hi3D || log "PixelArtistry workflows unavailable - skipping"

"$PY" -I -m pip install --no-warn-script-location "huggingface_hub>=0.30" hf_transfer

touch "$INSTALL_DIR/.imageto3d_installed"
log "Install complete"
