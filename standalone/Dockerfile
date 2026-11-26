# Imageto3D standalone (Gradio) RunPod image: system packages only. The venv, ComfyUI engine
# and models are installed on first boot into /workspace so they persist on your volume.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        git git-lfs curl wget ca-certificates unzip zip bc procps psmisc nano \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        build-essential cmake ninja-build pkg-config \
        ffmpeg sox libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
        openssh-server \
    && rm -rf /var/lib/apt/lists/*

COPY scripts /opt/imageto3d/scripts
COPY standalone /opt/imageto3d/standalone
RUN chmod +x /opt/imageto3d/scripts/*.sh /opt/imageto3d/standalone/*.sh

WORKDIR /workspace
EXPOSE 7860 22
CMD ["/opt/imageto3d/standalone/start.sh"]
