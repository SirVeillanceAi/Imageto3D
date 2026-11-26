#!/usr/bin/env bash
# Imageto3D - RunPod entrypoint
# ComfyUI-Easy-Install + native TRELLIS.2 / Pixal3D + Hi3D v1.0 refinement, GPU only.
# Installs once into /workspace (persistent volume), then just starts ComfyUI on later boots.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
INSTALL_DIR="${INSTALL_DIR:-$WORKSPACE/ComfyUI-Easy-Install}"
LOG_DIR="$WORKSPACE/logs"
COMFY_PORT="${COMFY_PORT:-8188}"
MIN_VRAM_GB="${MIN_VRAM_GB:-10}"
MODELS="${MODELS:-trellis2,pixal3d,hi3d}"
TORCH_CHANNEL="${TORCH_CHANNEL:-auto}"      # auto | cu130 | cu128
UPDATE_ON_START="${UPDATE_ON_START:-0}"     # 1 = git pull ComfyUI + Hi3D node every boot
COMFY_ARGS="${COMFY_ARGS:-}"

source "$REPO_DIR/scripts/common.sh"

# ---------- 4. Torch: match the wheel to this host's driver ----------
PY="$INSTALL_DIR/python_embeded/python"
align_torch() {
    local want; want="$(pick_torch_channel)"
    [ "$want" = "none" ] && die "Host driver supports CUDA ${DRIVER_CUDA}; need 12.8+ (13.0 recommended). In RunPod, filter 'CUDA Versions' to 13.0 when deploying."
    local have; have="$("$PY" -c 'import torch;print("cu"+torch.version.cuda.replace(".",""))' 2>/dev/null || echo none)"
    if [ "$have" = "$want" ]; then log "Torch build $have matches this host"; return; fi

    log "Switching Torch build: $have -> $want"
    if [ "$want" = "cu128" ]; then
        warn "Driver only supports CUDA ${DRIVER_CUDA}. Using Torch 2.8.0+cu128. ComfyUI recommends cu130 - pick a CUDA 13 host for best results."
        bash "$INSTALL_DIR/Add-Ons/Torch-Pack/Torch2.8.0+cu128.sh" NoPause </dev/null || true
        "$PY" -c 'import torch;assert torch.version.cuda.startswith("12.8")' 2>/dev/null || \
            "$PY" -I -m pip install --no-warn-script-location torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    else
        # Same pins the Easy-Install main installer uses
        "$PY" -I -m pip install --no-warn-script-location torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
    fi
}

# ---------- main ----------
log "Repo: $REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
install_system_packages
start_ssh
check_gpu

if [ ! -f "$INSTALL_DIR/.imageto3d_installed" ]; then
    log "First boot: installing ComfyUI-Easy-Install into $INSTALL_DIR (15-30 min). Log: $LOG_DIR/install.log"
    INSTALL_DIR="$INSTALL_DIR" bash "$REPO_DIR/scripts/install.sh" 2>&1 | tee -a "$LOG_DIR/install.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "Install failed - see $LOG_DIR/install.log"
elif [ "$UPDATE_ON_START" = "1" ]; then
    log "Updating ComfyUI and Hi3D node..."
    git -C "$INSTALL_DIR/ComfyUI" pull --ff-only || warn "ComfyUI update failed"
    "$PY" -I -m pip install --no-warn-script-location -r "$INSTALL_DIR/ComfyUI/requirements.txt" || warn "requirements update failed"
    git -C "$INSTALL_DIR/ComfyUI/custom_nodes/ComfyUI-Hi3D" checkout -q main && git -C "$INSTALL_DIR/ComfyUI/custom_nodes/ComfyUI-Hi3D" pull --ff-only || warn "Hi3D node update failed"
fi

align_torch
verify_cuda

log "Checking models: $MODELS"
"$PY" -I "$REPO_DIR/scripts/download_models.py" --comfy "$INSTALL_DIR/ComfyUI" --models "$MODELS" \
    2>&1 | tee -a "$LOG_DIR/models.log" || warn "Some models failed to download - see $LOG_DIR/models.log"

# Put the Hi3D workflows into the ComfyUI workflow sidebar (Imageto3D folder).
# PixelArtistry 04 is skipped: it needs the UltraTex / Animated-Characters packs.
WF_DIR="$INSTALL_DIR/ComfyUI/user/default/workflows/Imageto3D"
mkdir -p "$WF_DIR"
for wf in "$INSTALL_DIR"/Add-Ons/PixelArtistry-Hi3D/workflows/PixelArtistry_Hi3D_0[123]_*.json \
          "$INSTALL_DIR"/ComfyUI/custom_nodes/ComfyUI-Hi3D/workflows/*.json; do
    [ -f "$wf" ] || continue
    dest="$WF_DIR/$(basename "$wf")"
    [ -f "$dest" ] && continue
    cp "$wf" "$dest"
    [ "${HI3D_PRECISION:-int8}" = "fp16" ] && sed -i 's/hi3d_v1_int8\.safetensors/hi3d_v1.safetensors/g' "$dest"
done

log "Starting ComfyUI on port $COMFY_PORT  ->  https://${RUNPOD_POD_ID:-<pod-id>}-${COMFY_PORT}.proxy.runpod.net"
log "Start with Workflows > Imageto3D > PixelArtistry_Hi3D_01_Image_to_Hi3D (image -> TRELLIS.2/Pixal3D -> Hi3D)."
cd "$INSTALL_DIR"
while true; do
    # No --cpu / --lowvram: ComfyUI's dynamic VRAM handles 10 GB+ cards on GPU.
    "$PY" -W ignore::FutureWarning ComfyUI/main.py --listen 0.0.0.0 --port "$COMFY_PORT" $COMFY_ARGS \
        2>&1 | tee -a "$LOG_DIR/comfyui.log"
    warn "ComfyUI exited; restarting in 5 s"
    sleep 5
done
