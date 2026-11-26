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

export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE/.cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/.cache/uv}"
export DEBIAN_FRONTEND=noninteractive

mkdir -p "$LOG_DIR" "$HF_HOME"
log()  { echo -e "\033[92m[imageto3d]\033[0m $*"; }
warn() { echo -e "\033[93m[imageto3d] WARNING:\033[0m $*"; }
die()  { echo -e "\033[91m[imageto3d] ERROR:\033[0m $*"; echo "Container kept alive for debugging (web terminal / SSH)."; sleep infinity; }

# ---------- 1. System packages (skipped when using the prebuilt image) ----------
install_system_packages() {
    command -v apt-get >/dev/null 2>&1 || { warn "apt-get not found; make sure git, curl, unzip and python3.12 exist."; return; }
    if command -v git >/dev/null && command -v python3.12 >/dev/null && command -v unzip >/dev/null \
       && python3.12 -c "import venv, ensurepip" 2>/dev/null && command -v ffmpeg >/dev/null; then
        return
    fi
    log "Installing system packages..."
    apt-get update -y
    apt-get install -y --no-install-recommends \
        git git-lfs curl wget ca-certificates unzip zip bc procps psmisc nano \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        build-essential cmake ninja-build pkg-config \
        ffmpeg sox libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
        openssh-server
    rm -rf /var/lib/apt/lists/*
}

# ---------- 2. SSH (RunPod injects PUBLIC_KEY) ----------
start_ssh() {
    [ -n "${PUBLIC_KEY:-}" ] || return
    command -v sshd >/dev/null 2>&1 || return
    mkdir -p ~/.ssh /run/sshd && chmod 700 ~/.ssh
    grep -qF "$PUBLIC_KEY" ~/.ssh/authorized_keys 2>/dev/null || echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    ssh-keygen -A >/dev/null 2>&1
    /usr/sbin/sshd && log "SSH started"
}

# ---------- 3. GPU checks (GPU only - no CPU fallback) ----------
GPU_NAME=""; VRAM_MIB=0; DRIVER_CUDA="0.0"; COMPUTE_CAP="0.0"
check_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || die "No NVIDIA GPU visible. This template is GPU-only - deploy it on a GPU pod."
    nvidia-smi >/dev/null 2>&1 || die "nvidia-smi failed - the GPU/driver is not usable on this host."
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
    VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
    DRIVER_CUDA="$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -n1)"
    log "GPU: $GPU_NAME | VRAM: ${VRAM_MIB} MiB | compute ${COMPUTE_CAP:-?} | driver CUDA ${DRIVER_CUDA:-?}"

    # 10 GB cards report ~10240 MiB; allow a little slack.
    if [ "$VRAM_MIB" -lt $(( MIN_VRAM_GB * 1000 )) ]; then
        [ "${ALLOW_LOW_VRAM:-0}" = "1" ] && warn "Only ${VRAM_MIB} MiB VRAM - continuing because ALLOW_LOW_VRAM=1." \
            || die "GPU has ${VRAM_MIB} MiB VRAM; this template needs ${MIN_VRAM_GB} GB+. (Set ALLOW_LOW_VRAM=1 to override.)"
    fi
}

ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }

pick_torch_channel() {
    if [ "$TORCH_CHANNEL" != "auto" ]; then echo "$TORCH_CHANNEL"; return; fi
    if ver_ge "${DRIVER_CUDA:-0}" "13.0" && ver_ge "${COMPUTE_CAP:-0}" "7.5"; then echo cu130
    elif ver_ge "${DRIVER_CUDA:-0}" "12.8"; then echo cu128
    else echo none; fi
}

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

verify_cuda() {
    "$PY" - <<'EOF' || die "PyTorch cannot use the GPU. Check $WORKSPACE/logs and the driver/CUDA version of this pod."
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.randn(1024, 1024, device="cuda"); (x @ x).sum().item()
print(f"[imageto3d] torch {torch.__version__} on {torch.cuda.get_device_name(0)} - OK")
EOF
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
    git -C "$INSTALL_DIR/ComfyUI/custom_nodes/ComfyUI-Hi3D" pull --ff-only || warn "Hi3D node update failed"
fi

align_torch
verify_cuda

log "Checking models: $MODELS"
"$PY" -I "$REPO_DIR/scripts/download_models.py" --comfy "$INSTALL_DIR/ComfyUI" --models "$MODELS" \
    2>&1 | tee -a "$LOG_DIR/models.log" || warn "Some models failed to download - see $LOG_DIR/models.log"

# Ship the Hi3D refine workflows into the ComfyUI workflow browser
WF_DIR="$INSTALL_DIR/ComfyUI/user/default/workflows/Imageto3D"
mkdir -p "$WF_DIR"
cp -n "$INSTALL_DIR"/ComfyUI/custom_nodes/ComfyUI-Hi3D/workflows/*.json "$WF_DIR/" 2>/dev/null || true

log "Starting ComfyUI on port $COMFY_PORT  ->  https://${RUNPOD_POD_ID:-<pod-id>}-${COMFY_PORT}.proxy.runpod.net"
log "TRELLIS.2 / Pixal3D: Templates > 3D > 'Pixal3D: Image to Model'. Hi3D: Workflows > Imageto3D."
cd "$INSTALL_DIR"
while true; do
    # No --cpu / --lowvram: ComfyUI's dynamic VRAM handles 10 GB+ cards on GPU.
    "$PY" -W ignore::FutureWarning ComfyUI/main.py --listen 0.0.0.0 --port "$COMFY_PORT" $COMFY_ARGS \
        2>&1 | tee -a "$LOG_DIR/comfyui.log"
    warn "ComfyUI exited; restarting in 5 s"
    sleep 5
done
