FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# ^ pick 12.4 OR 12.6 everywhere; 12.4 shown for consistency with your LD paths

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils python3-pip \
    libsndfile1 ffmpeg git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -y
ENV PATH="/root/.local/bin:${PATH}"

# CUDA loader path (avoid hard pin to a different minor)
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# TF/GPU niceties
ENV TF_FORCE_GPU_ALLOW_GROWTH=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_PLATFORMS=cuda,cpu

# copy project manifest and lock it deterministically
WORKDIR /opt/app
COPY pyproject.toml ./

# produce a lock (or check in uv.lock and just COPY it instead)
RUN uv lock

# sync deps into a venv at /opt/venv (fast, reproducible)
RUN uv sync --frozen --python=/usr/bin/python3.11 --no-dev

# show JAX versions (build-time sanity)
RUN /opt/venv/bin/python - <<'PY'
import jax, jaxlib
print("JAX:", jax.__version__)
print("JAXLIB:", jaxlib.__version__)
try:
    import importlib
    print("CUDA plugin:", importlib.metadata.version("jax-cuda12-plugin"))
except Exception as e:
    print("CUDA plugin:", "not found?", e)
PY

# app files
COPY app.py utils.py jam_worker.py ./

EXPOSE 7860
CMD ["/opt/venv/bin/uvicorn","app:app","--host","0.0.0.0","--port","7860"]
