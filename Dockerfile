# thecollabagepatch/magenta:latest
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# CUDA libs present + on loader path
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-12-4 && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/compat:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
RUN ln -sf /usr/local/cuda/targets/x86_64-linux/lib /usr/local/cuda/lib64 || true

# Ensure the NVIDIA repo key is present (non-interactive) and install cuDNN 9.8
RUN set -eux; \
  apt-get update && apt-get install -y --no-install-recommends gnupg ca-certificates curl; \
  install -d -m 0755 /usr/share/keyrings; \
  # Refresh the *same* keyring the base source uses (no second source file)
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    | gpg --batch --yes --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg; \
  apt-get update; \
  # If libcudnn is "held", unhold it so we can move to 9.8
  apt-mark unhold libcudnn9-cuda-12 || true; \
  # Install cuDNN 9.8 for CUDA 12 (correct dev package name!)
  apt-get install -y --no-install-recommends \
      'libcudnn9-cuda-12=9.8.*' \
      'libcudnn9-dev-cuda-12=9.8.*' \
      --allow-downgrades --allow-change-held-packages; \
  apt-mark hold libcudnn9-cuda-12 || true; \
  ldconfig; \
  rm -rf /var/lib/apt/lists/*

# (optional) preload workaround if still needed
ENV LD_PRELOAD=/usr/local/cuda/lib64/libcusparse.so.12:/usr/local/cuda/lib64/libcublas.so.12:/usr/local/cuda/lib64/libcublasLt.so.12:/usr/local/cuda/lib64/libcufft.so.11:/usr/local/cuda/lib64/libcusolver.so.11

# Better allocator (less fragmentation than BFC during XLA autotune)
ENV TF_GPU_ALLOCATOR=cuda_malloc_async

# Let cuBLAS use TF32 fast path on Ada (L40S) for big GEMMs
ENV TF_ENABLE_CUBLAS_TF32=1 NVIDIA_TF32_OVERRIDE=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

ENV JAX_PLATFORMS=""

# --- OS deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates git \
    libsndfile1 ffmpeg \
    build-essential pkg-config \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3 => 3.11 for convenience
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && python -m pip install --upgrade pip

# --- Python deps (pin order matters!) ---
# 1) JAX CUDA pins
RUN python -m pip install "jax[cuda12]==0.6.2" "jaxlib==0.6.2"

# 2) Lock seqio early to avoid backtracking madness
RUN python -m pip install "seqio==0.0.11"

# 3) Install Magenta RT *without* deps so we control pins
RUN python -m pip install --no-deps 'git+https://github.com/magenta/magenta-realtime#egg=magenta_rt[gpu]'

# 4) TF nightlies (MATCH DATES!)
RUN python -m pip install \
    "tf_nightly==2.20.0.dev20250619" \
    "tensorflow-text-nightly==2.20.0.dev20250316" \
    "tf-hub-nightly"

# 5) tf2jax pinned alongside tf_nightly so pip doesn’t drag stable TF
RUN python -m pip install tf2jax "tf_nightly==2.20.0.dev20250619"

# 6) The rest of MRT deps + API runtime deps
RUN python -m pip install \
    gin-config librosa resampy soundfile \
    google-auth google-auth-oauthlib google-auth-httplib2 \
    google-api-core googleapis-common-protos google-resumable-media \
    google-cloud-storage requests tqdm typing-extensions numpy==2.1.3 \
    fastapi uvicorn[standard] python-multipart pyloudnorm

# 7) Exact commits for T5X/Flaxformer as in pyproject
RUN python -m pip install \
    "t5x @ git+https://github.com/google-research/t5x.git@92c5b46" \
    "flaxformer @ git+https://github.com/google/flaxformer@399ea3a"

# ---- FINAL: enforce TF nightlies and clean any stable TF ----
RUN python - <<'PY'
import sys, sysconfig, glob, os, shutil
# Find a writable site dir (site-packages OR dist-packages)
cands = [sysconfig.get_paths().get('purelib'), sysconfig.get_paths().get('platlib')]
cands += [p for p in sys.path if p and p.endswith(('site-packages','dist-packages'))]
site = next(p for p in cands if p and os.path.isdir(p))

patterns = [
  "tensorflow", "tensorflow-*.dist-info", "tensorflow-*.egg-info",
  "tf-nightly-*.dist-info", "tf_nightly-*.dist-info",
  "tensorflow_text", "tensorflow_text-*.dist-info",
  "tf-hub-nightly-*.dist-info", "tf_hub_nightly-*.dist-info",
  "tf_keras-nightly-*.dist-info", "tf_keras_nightly-*.dist-info",
  "tensorboard*", "tb-nightly-*.dist-info",
  "keras*",  # remove stray keras
  "tensorflow_hub*", "tensorflow_io*",
]
for pat in patterns:
  for path in glob.glob(os.path.join(site, pat)):
    if os.path.isdir(path): shutil.rmtree(path, ignore_errors=True)
    else:
      try: os.remove(path)
      except FileNotFoundError: pass

print("TF/Hub/Text cleared in:", site)
PY

# Reinstall pinned nightlies in ONE transaction
RUN python -m pip install --no-cache-dir --force-reinstall \
    "tf-nightly==2.20.0.dev20250619" \
    "tensorflow-text-nightly==2.20.0.dev20250316" \
    "tf-hub-nightly"

RUN python -m pip install huggingface_hub

RUN python -m pip install --no-cache-dir --force-reinstall "protobuf==4.25.3"

RUN python -m pip install gradio



# Switch to Spaces’ preferred user
# Switch to Spaces’ preferred user
RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

# Copy from *build context* into image, owned by appuser
COPY --chown=appuser:appuser app.py /home/appuser/app/app.py

# NEW: shared utils + worker
COPY --chown=appuser:appuser utils.py /home/appuser/app/utils.py
COPY --chown=appuser:appuser jam_worker.py /home/appuser/app/jam_worker.py

USER appuser

EXPOSE 7860
CMD ["bash", "-lc", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
