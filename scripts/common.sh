# Shared helpers for the Imageto3D RunPod entrypoints (sourced, not run).
# Expects WORKSPACE, LOG_DIR, MIN_VRAM_GB, TORCH_CHANNEL to be set; verify_cuda uses $PY.

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

verify_cuda() {
    "$PY" - <<'EOF' || die "PyTorch cannot use the GPU. Check $WORKSPACE/logs and the driver/CUDA version of this pod."
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.randn(1024, 1024, device="cuda"); (x @ x).sum().item()
print(f"[imageto3d] torch {torch.__version__} on {torch.cuda.get_device_name(0)} - OK")
EOF
}

