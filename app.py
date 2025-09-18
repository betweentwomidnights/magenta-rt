import os

# ---- Space mode gating (place above any JAX import!) ----
SPACE_MODE = os.getenv("SPACE_MODE")
if SPACE_MODE is None:
    try:
        import jax
        SPACE_MODE = "serve" if any(getattr(d, "platform", "") in ("gpu","cuda","rocm") for d in jax.devices()) else "template"
    except Exception:
        SPACE_MODE = "template"



if SPACE_MODE != "serve":
    # In template mode, force JAX to CPU so it won't try to load CUDA plugins
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
else:
    # Only set GPU-friendly XLA flags when we actually intend to serve on GPU
    os.environ.setdefault(
        "XLA_FLAGS",
        " ".join([
            "--xla_gpu_enable_triton_gemm=true",
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            "--xla_gpu_autotune_level=2",
        ])
    )

# Optional: persist JAX compile cache across restarts (reduces warmup time)
os.environ.setdefault("JAX_CACHE_DIR", "/home/appuser/.cache/jax")

import jax
# ✅ Valid choices include: "default", "high", "highest", "tensorfloat32", "float32", etc.
# TF32 is the sweet spot on Ampere/Ada GPUs for ~1.1–1.3× matmul speedups.
try:
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
except Exception:
    jax.config.update("jax_default_matmul_precision", "high")  # older alias

# Initialize the on-disk compilation cache (best-effort)
try:
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.initialize_cache(os.environ["JAX_CACHE_DIR"])
except Exception:
    pass
# --------------------------------------------------------------------



from magenta_rt import system, audio as au
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException, Response, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import tempfile, io, base64, math, threading
from fastapi.middleware.cors import CORSMiddleware
from contextlib import contextmanager
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly
from utils import (
    match_loudness_to_reference, stitch_generated, hard_trim_seconds,
    apply_micro_fades, make_bar_aligned_context, take_bar_aligned_tail,
    resample_and_snap, wav_bytes_base64
)

from jam_worker import JamWorker, JamParams, JamChunk
from one_shot_generation import generate_loop_continuation_with_mrt, generate_style_only_with_mrt

import uuid, threading

import logging

import gradio as gr
from typing import Optional, Union, Literal


import json, asyncio, base64
import time



from starlette.websockets import WebSocketState
try:
    from uvicorn.protocols.utils import ClientDisconnected  # uvicorn >= 0.20
except Exception:
    class ClientDisconnected(Exception):  # fallback
        pass

import re, tarfile
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

from pydantic import BaseModel

from model_management import CheckpointManager, AssetManager, ModelSelector, ModelSelect

def _gpu_probe() -> dict:
    """
    Returns:
      {
        "ok": bool,
        "backend": str | None,        # "gpu" | "cpu" | "tpu" | None
        "has_gpu": bool,
        "devices": list[str],         # e.g. ["gpu:0", "gpu:1"]
        "error": str | None,
      }
    """
    try:
        import jax
        try:
            backend = jax.default_backend()  # "gpu", "cpu", "tpu"
        except Exception:
            from jax.lib import xla_bridge
            backend = getattr(xla_bridge.get_backend(), "platform", None)

        try:
            devices = jax.devices()
            has_gpu = any(getattr(d, "platform", "") in ("gpu", "cuda", "rocm") for d in devices)
            dev_list = [f"{getattr(d, 'platform', '?')}:{getattr(d, 'id', '?')}" for d in devices]
            return {"ok": True, "backend": backend, "has_gpu": has_gpu, "devices": dev_list, "error": None}
        except Exception as e:
            return {"ok": False, "backend": backend, "has_gpu": False, "devices": [], "error": f"jax.devices failed: {e}"}
    except Exception as e:
        return {"ok": False, "backend": None, "has_gpu": False, "devices": [], "error": f"jax import failed: {e}"}

# ---- Finetune assets (mean & centroids) --------------------------------------
# _FINETUNE_REPO_DEFAULT = os.getenv("MRT_ASSETS_REPO", "thepatch/magenta-ft")
_ASSETS_REPO_ID: str | None = None
_MEAN_EMBED: np.ndarray | None = None           # shape (D,) dtype float32
_CENTROIDS: np.ndarray | None = None            # shape (K, D) dtype float32

# _STEP_RE = re.compile(r"(?:^|/)checkpoint_(\d+)(?:/|\.tar\.gz|\.tgz)?$")

# Create instances (these don't modify globals)
asset_manager = AssetManager()
model_selector = ModelSelector(CheckpointManager(), asset_manager)

def _sync_assets_globals_from_manager():
    # Keeps /model/config in sync with what the asset manager has
    global _MEAN_EMBED, _CENTROIDS, _ASSETS_REPO_ID
    _MEAN_EMBED = asset_manager.mean_embed
    _CENTROIDS = asset_manager.centroids
    _ASSETS_REPO_ID = asset_manager.assets_repo_id

def _any_jam_running() -> bool:
    with jam_lock:
        return any(w.is_alive() for w in jam_registry.values())

def _stop_all_jams(timeout: float = 5.0):
    with jam_lock:
        for sid, w in list(jam_registry.items()):
            if w.is_alive():
                w.stop()
                w.join(timeout=timeout)
                jam_registry.pop(sid, None)


async def send_json_safe(ws: WebSocket, obj) -> bool:
    """Try to send. Returns False if the socket is (or becomes) closed."""
    if ws.client_state == WebSocketState.DISCONNECTED or ws.application_state == WebSocketState.DISCONNECTED:
        return False
    try:
        await ws.send_text(json.dumps(obj))
        return True
    except (WebSocketDisconnect, ClientDisconnected, RuntimeError):
        return False
    except Exception:
        return False

# --- Patch T5X mesh helpers for GPUs on JAX >= 0.7 (coords present, no core_on_chip) ---
def _patch_t5x_for_gpu_coords():
    try:
        import jax
        from t5x import partitioning as _t5x_part

        old_bounds = getattr(_t5x_part, "bounds_from_last_device", None)
        old_getcoords = getattr(_t5x_part, "get_coords", None)

        def _bounds_from_last_device_gpu_safe(last_device):
            # TPU: coords + core_on_chip
            core = getattr(last_device, "core_on_chip", None)
            coords = getattr(last_device, "coords", None)
            if coords is not None and core is not None:
                x, y, z = coords
                return x + 1, y + 1, z + 1, core + 1
            # Non-TPU (or GPU lacking core_on_chip): hosts x local_devices
            return jax.host_count(), jax.local_device_count()

        def _get_coords_gpu_safe(device):
            core = getattr(device, "core_on_chip", None)
            coords = getattr(device, "coords", None)
            if coords is not None and core is not None:
                return (*coords, core)
            # Fallback that works on CPU/GPU
            return (device.process_index, device.id % jax.local_device_count())

        _t5x_part.bounds_from_last_device = _bounds_from_last_device_gpu_safe
        _t5x_part.get_coords = _get_coords_gpu_safe
        import logging; logging.info("Patched t5x.partitioning for GPU coords without core_on_chip.")
    except Exception as e:
        import logging; logging.exception("t5x GPU-coords patch failed: %s", e)

# Call the patch immediately at import time (before MagentaRT init)
_patch_t5x_for_gpu_coords()

jam_registry: dict[str, JamWorker] = {}
jam_lock = threading.Lock()

@contextmanager
def mrt_overrides(mrt, **kwargs):
    """Temporarily set attributes on MRT if they exist; restore after."""
    old = {}
    try:
        for k, v in kwargs.items():
            if hasattr(mrt, k):
                old[k] = getattr(mrt, k)
                setattr(mrt, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(mrt, k, v)

# loudness utils
try:
    import pyloudnorm as pyln
    _HAS_LOUDNORM = True
except Exception:
    _HAS_LOUDNORM = False

# def _combine_styles(mrt, styles_str: str = "", weights_str: str = ""):
#     extra = [s.strip() for s in (styles_str or "").split(",") if s.strip()]
#     if not extra:
#         return mrt.embed_style("warmup")
#     sw = [float(x) for x in (weights_str or "").split(",") if x.strip()]
#     embeds, weights = [], []
#     for i, s in enumerate(extra):
#         embeds.append(mrt.embed_style(s))
#         weights.append(sw[i] if i < len(sw) else 1.0)
#     wsum = sum(weights) or 1.0
#     weights = [w/wsum for w in weights]
#     import numpy as np
#     return np.sum([w*e for w, e in zip(weights, embeds)], axis=0).astype(np.float32)

def build_style_vector(
    mrt,
    *,
    text_styles: list[str] | None = None,
    text_weights: list[float] | None = None,
    loop_embed: np.ndarray | None = None,
    loop_weight: float | None = None,
    mean_weight: float | None = None,
    centroid_weights: list[float] | None = None,
) -> np.ndarray:
    """
    Returns a single style embedding combining:
      - loop embedding (optional)
      - one or more text style embeddings (optional)
      - mean finetune embedding (optional)
      - centroid embeddings (optional)
    All weights are normalized so they sum to 1 if > 0.
    """
    comps: list[np.ndarray] = []
    weights: list[float] = []

    # loop component
    if loop_embed is not None and (loop_weight or 0) > 0:
        comps.append(loop_embed.astype(np.float32, copy=False))
        weights.append(float(loop_weight))

    # text components
    if text_styles:
        for i, s in enumerate(text_styles):
            s = s.strip()
            if not s:
                continue
            w = 1.0
            if text_weights and i < len(text_weights):
                try: w = float(text_weights[i])
                except: w = 1.0
            if w <= 0: 
                continue
            e = mrt.embed_style(s)
            comps.append(e.astype(np.float32, copy=False))
            weights.append(w)

    # mean finetune
    if mean_weight and (_MEAN_EMBED is not None) and mean_weight > 0:
        comps.append(_MEAN_EMBED)
        weights.append(float(mean_weight))

    # centroid components
    if centroid_weights and _CENTROIDS is not None:
        K = _CENTROIDS.shape[0]
        for k, w in enumerate(centroid_weights[:K]):
            try: w = float(w)
            except: w = 0.0
            if w <= 0: 
                continue
            comps.append(_CENTROIDS[k])
            weights.append(w)

    if not comps:
        # fallback: neutral style if nothing provided
        return mrt.embed_style("")

    wsum = sum(weights)
    if wsum <= 0:
        return mrt.embed_style("")
    weights = [w/wsum for w in weights]

    # weighted sum
    out = np.zeros_like(comps[0], dtype=np.float32)
    for w, e in zip(weights, comps):
        out += w * e.astype(np.float32, copy=False)
    return out



# ----------------------------
# FastAPI app with lazy, thread-safe model init
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or lock to your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MRT = None
_MRT_LOCK = threading.Lock()

def get_mrt():
    global _MRT
    if _MRT is None:
        with _MRT_LOCK:
            if _MRT is None:
                ckpt_dir = CheckpointManager.resolve_checkpoint_dir()  # uses MRT_CKPT_REPO/STEP if present
                _MRT = system.MagentaRT(
                    tag=os.getenv("MRT_SIZE", "large"),
                    guidance_weight=5.0,
                    device="gpu",
                    checkpoint_dir=ckpt_dir,
                    lazy=False
                )
                # If no assets loaded yet, and a repo is configured, load them now.
                if asset_manager.mean_embed is None and asset_manager.centroids is None:
                    repo = os.getenv("MRT_ASSETS_REPO") or os.getenv("MRT_CKPT_REPO")
                    if repo:
                        asset_manager.load_finetune_assets_from_hf(repo, None)
                        _sync_assets_globals_from_manager()
    return _MRT

_WARMED = False
_WARMUP_LOCK = threading.Lock()

def _mrt_warmup():
    """
    Build a minimal, bar-aligned silent context and run one 2s generate_chunk
    to trigger XLA JIT & autotune so first real request is fast.
    """
    global _WARMED
    with _WARMUP_LOCK:
        if _WARMED:
            return
        try:
            mrt = get_mrt()

            # --- derive timing from model config ---
            codec_fps = float(mrt.codec.frame_rate)
            ctx_seconds = float(mrt.config.context_length_frames) / codec_fps
            sr = int(mrt.sample_rate)

            # We'll align to 120 BPM, 4/4, and generate one ~2s chunk
            bpm = 120.0
            beats_per_bar = 4

            # --- build a silent, stereo context of ctx_seconds ---
            samples = int(max(1, round(ctx_seconds * sr)))
            silent = np.zeros((samples, 2), dtype=np.float32)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, silent, sr, subtype="PCM_16")
                tmp_path = tmp.name

            try:
                # Load as Waveform and take a tail of exactly ctx_seconds
                loop = au.Waveform.from_file(tmp_path).resample(sr).as_stereo()
                seconds_per_bar = beats_per_bar * (60.0 / bpm)
                ctx_tail = take_bar_aligned_tail(loop, bpm, beats_per_bar, ctx_seconds)

                # Tokens for context window
                tokens_full = mrt.codec.encode(ctx_tail).astype(np.int32)
                tokens = tokens_full[:, :mrt.config.decoder_codec_rvq_depth]
                context_tokens = make_bar_aligned_context(
                    tokens,
                    bpm=bpm,
                    fps=float(mrt.codec.frame_rate),
                    ctx_frames=mrt.config.context_length_frames,
                    beats_per_bar=beats_per_bar,
                )

                # Init state and a basic style vector (text token is fine)
                state = mrt.init_state()
                state.context_tokens = context_tokens
                style_vec = mrt.embed_style("warmup")

                # --- one throwaway chunk (~2s) ---
                _wav, _state = mrt.generate_chunk(state=state, style=style_vec)

                logging.info("MagentaRT warmup complete.")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            _WARMED = True
        except Exception as e:
            # Never crash on warmup errors; log and continue serving
            logging.exception("MagentaRT warmup failed (continuing without warmup): %s", e)


# ----------------------------
# startup and model selection
# ----------------------------

@app.on_event("startup")
def _boot():
    # 1) Load finetune assets up front (only if envs are present)
    repo = os.getenv("MRT_ASSETS_REPO") or os.getenv("MRT_CKPT_REPO")
    if repo:
        ok, msg = asset_manager.load_finetune_assets_from_hf(repo, None)
        _sync_assets_globals_from_manager()  # keep /model/config in sync
        logging.info("Startup asset load from %s: %s", repo, "ok" if ok else msg)
    else:
        logging.info("Startup asset load: no repo env set; skipping.")

    # 2) Start warmup in the background (unchanged behavior)
    if os.getenv("MRT_WARMUP", "1") != "0":
        threading.Thread(target=_mrt_warmup, name="mrt-warmup", daemon=True).start()

@app.get("/model/status")
def model_status():
    mrt = get_mrt()
    return {
        "tag": getattr(mrt, "_tag", "unknown"),
        "using_checkpoint_dir": True,
        "codec_frame_rate": float(mrt.codec.frame_rate),
        "decoder_rvq_depth": int(mrt.config.decoder_codec_rvq_depth),
        "context_seconds": float(mrt.config.context_length),
        "chunk_seconds": float(mrt.config.chunk_length),
        "crossfade_seconds": float(mrt.config.crossfade_length),
        "selected_step": os.getenv("MRT_CKPT_STEP"),
        "repo": os.getenv("MRT_CKPT_REPO"),
    }

@app.post("/model/swap")
def model_swap(step: int = Form(...)):
    # stop any active jam if you want to be strict (not shown)
    os.environ["MRT_CKPT_STEP"] = str(step)
    global _MRT
    with _MRT_LOCK:
        _MRT = None  # force re-create on next get_mrt()
    # optionally pre-warm here by calling get_mrt()
    return {"reloaded": True, "step": step}

@app.post("/model/assets/load")
def model_assets_load(repo_id: str = Form(None)):
    global _MEAN_EMBED, _CENTROIDS, _ASSETS_REPO_ID
    ok, msg = asset_manager.load_finetune_assets_from_hf(repo_id, get_mrt())
    # Sync globals after loading
    _MEAN_EMBED = asset_manager.mean_embed
    _CENTROIDS = asset_manager.centroids
    _ASSETS_REPO_ID = asset_manager.assets_repo_id
    return {"ok": ok, "message": msg, "repo_id": _ASSETS_REPO_ID,
            "mean": _MEAN_EMBED is not None,
            "centroids": None if _CENTROIDS is None else int(_CENTROIDS.shape[0])}

@app.get("/model/assets/status")
def model_assets_status():
    d = None
    try:
        d = int(get_mrt().style_model.config.embedding_dim)
    except Exception:
        pass
    return {
        "repo_id": _ASSETS_REPO_ID,
        "mean_loaded": _MEAN_EMBED is not None,
        "centroids_loaded": False if _CENTROIDS is None else True,
        "centroid_count": None if _CENTROIDS is None else int(_CENTROIDS.shape[0]),
        "embedding_dim": d,
    }

@app.get("/model/config")
def model_config():
    """
    Lightweight config snapshot:
      - never calls get_mrt() (no model build / no downloads)
      - never calls snapshot_download()
      - reports whether a model instance is currently loaded in memory
      - best-effort local checkpoint presence (no network)
    """
    # Read-only snapshot of in-memory model presence
    with _MRT_LOCK:
        loaded = (_MRT is not None)

    size   = os.getenv("MRT_SIZE", "large")
    repo   = os.getenv("MRT_CKPT_REPO")
    rev    = os.getenv("MRT_CKPT_REV", "main")
    step   = os.getenv("MRT_CKPT_STEP")
    assets = os.getenv("MRT_ASSETS_REPO")

    # Use CheckpointManager for local cache probe (no network)
    local_ckpt = None
    if step:
        try:
            from pathlib import Path
            import re
            step_escaped = re.escape(str(step))
            candidates = []
            for root in ("/home/appuser/.cache/mrt_ckpt/extracted",
                         "/home/appuser/.cache/mrt_ckpt/repo"):
                p = Path(root)
                if not p.exists():
                    continue
                # Look for exact "checkpoint_<step>" directories anywhere under these roots
                for d in p.rglob(f"checkpoint_{step}"):
                    if d.is_dir():
                        candidates.append(str(d))
            local_ckpt = candidates[0] if candidates else None
        except Exception:
            local_ckpt = None

    return {
        "size": size,
        "repo": repo,
        "revision": rev,
        "selected_step": step,
        "assets_repo": assets,

        # in-memory + local cache hints (no network, no model build)
        "loaded": loaded,
        "active_jam": _any_jam_running(),
        "local_checkpoint_dir": local_ckpt,   # None if not found locally

        # steering assets currently resident in memory
        "mean_loaded": (_MEAN_EMBED is not None),
        "centroids_loaded": (_CENTROIDS is not None),
        "centroid_count": (None if _CENTROIDS is None else int(_CENTROIDS.shape[0])),
        "warmup_done": bool(_WARMED),
    }

@app.get("/model/checkpoints")
def model_checkpoints(repo_id: str, revision: str = "main"):
    steps = CheckpointManager.list_ckpt_steps(repo_id, revision)
    return {"repo": repo_id, "revision": revision, "steps": steps, "latest": (steps[-1] if steps else None)}

@app.post("/model/select")
def model_select(req: ModelSelect):
    """
    Swap model/checkpoint/assets. If req.prewarm is True, run the full bar-aligned warmup
    (_mrt_warmup) synchronously so we only report warmed once the new model is actually ready.
    """
    global _MRT, _MEAN_EMBED, _CENTROIDS, _ASSETS_REPO_ID, _WARMED

    # 1) Validate the request (no side-effects)
    success, validation_result = model_selector.validate_selection(req)
    if not success:
        if "error" in validation_result:
            raise HTTPException(status_code=400, detail=validation_result["error"])
        return {"ok": False, **validation_result}
    


    # Augment response surface
    validation_result["active_jam"] = _any_jam_running()

    # Dry-run path
    if req.dry_run:
        return {"ok": True, "dry_run": True, **validation_result}
    
    if req.ckpt_step == "none":  # user asked for stock base
        asset_manager.clear_assets()       # implement .clear_assets() to set embeds/centroids to None
        _sync_assets_globals_from_manager()

    # 2) Handle jam policy
    if _any_jam_running():
        if req.stop_active:
            _stop_all_jams()
        else:
            raise HTTPException(status_code=409, detail="A jam is running; retry with stop_active=true")

    # 3) Compute environment changes (no mutation yet)
    env_changes = model_selector.prepare_env_changes(req, validation_result)

    # Keep current env for rollback
    old_env = {
        "MRT_SIZE": os.getenv("MRT_SIZE"),
        "MRT_CKPT_REPO": os.getenv("MRT_CKPT_REPO"),
        "MRT_CKPT_REV": os.getenv("MRT_CKPT_REV"),
        "MRT_CKPT_STEP": os.getenv("MRT_CKPT_STEP"),
        "MRT_ASSETS_REPO": os.getenv("MRT_ASSETS_REPO"),
    }

    try:
        # 4) Apply env atomically
        for key, value in env_changes.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)

        # 5) Force rebuild of the model and reset warmup state
        with _MRT_LOCK:
            _MRT = None
        with _WARMUP_LOCK:
            _WARMED = False  # ← critical: don't leak previous model's warmed state

        # 6) Load finetune assets if requested (mean/centroids)
        if req.sync_assets and validation_result.get("assets_repo"):
            ok, msg = asset_manager.load_finetune_assets_from_hf(
                validation_result["assets_repo"],
                None  # don't implicitly instantiate model here; we'll do it below
            )
            if ok:
                _MEAN_EMBED = asset_manager.mean_embed
                _CENTROIDS = asset_manager.centroids
                _ASSETS_REPO_ID = asset_manager.assets_repo_id
            else:
                logging.warning("Asset sync skipped/failed: %s", msg)

        # 7) Prewarm behavior:
        #    - If prewarm=True, run the *real* bar-aligned warmup synchronously.
        #    - This will instantiate the new MRT and set _WARMED=True on success.
        if req.prewarm:
            _mrt_warmup()  # builds MRT internally via get_mrt(), runs generate_chunk, sets _WARMED

        # Optional: if you want to always ensure MRT exists (even without prewarm), uncomment:
        # else:
        #     _ = get_mrt()

        return {
            "ok": True,
            **validation_result,
            "warmup_done": bool(_WARMED),
        }

    except Exception as e:
        # 8) Roll back env on failure
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        # Also reset model pointer & warmed flag to a safe state
        with _MRT_LOCK:
            _MRT = None
        with _WARMUP_LOCK:
            _WARMED = False
        logging.exception("Model select failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Model select failed: {e}")
    


# ----------------------------
# one-shot generation
# ----------------------------



@app.post("/generate")
def generate(
    loop_audio: UploadFile = File(...),
    bpm: float = Form(...),
    bars: int = Form(8),
    beats_per_bar: int = Form(4),
    styles: str = Form("acid house"),
    style_weights: str = Form(""),
    loop_weight: float = Form(1.0),
    loudness_mode: str = Form("auto"),
    loudness_headroom_db: float = Form(1.0),
    guidance_weight: float = Form(5.0),
    temperature: float = Form(1.1),
    topk: int = Form(40),
    target_sample_rate: int | None = Form(None),
    intro_bars_to_drop: int = Form(0),          # <— NEW
):
    # Read file
    data = loop_audio.file.read()
    if not data:
        return {"error": "Empty file"}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    # Parse styles + weights
    extra_styles = [s for s in (styles.split(",") if styles else []) if s.strip()]
    weights = [float(x) for x in style_weights.split(",")] if style_weights else None

    mrt = get_mrt()  # warm once, in this worker thread
    # Temporarily override MRT inference knobs for this request
    with mrt_overrides(mrt,
                       guidance_weight=guidance_weight,
                       temperature=temperature,
                       topk=topk):
        wav, loud_stats = generate_loop_continuation_with_mrt(
            mrt,
            input_wav_path=tmp_path,
            bpm=bpm,
            extra_styles=extra_styles,
            style_weights=weights,
            bars=bars,
            beats_per_bar=beats_per_bar,
            loop_weight=loop_weight,
            loudness_mode=loudness_mode,
            loudness_headroom_db=loudness_headroom_db,
            intro_bars_to_drop=intro_bars_to_drop,   # <— pass through
        )

    # 1) Figure out the desired SR
    inp_info = sf.info(tmp_path)
    input_sr = int(inp_info.samplerate)
    target_sr = int(target_sample_rate or input_sr)

    # 2) Convert to target SR + snap to exact bars
    cur_sr = int(mrt.sample_rate)
    x = wav.samples if wav.samples.ndim == 2 else wav.samples[:, None]
    seconds_per_bar = (60.0 / float(bpm)) * int(beats_per_bar)
    expected_secs = float(bars) * seconds_per_bar
    x = resample_and_snap(x, cur_sr=cur_sr, target_sr=target_sr, seconds=expected_secs)

    # 3) Encode WAV once (no extra write)
    audio_b64, total_samples, channels = wav_bytes_base64(x, target_sr)
    loop_duration_seconds = total_samples / float(target_sr)

    # 4) Metadata
    metadata = {
        "bpm": int(round(bpm)),
        "bars": int(bars),
        "beats_per_bar": int(beats_per_bar),
        "styles": extra_styles,
        "style_weights": weights,
        "loop_weight": loop_weight,
        "loudness": loud_stats,
        "sample_rate": int(target_sr),
        "channels": int(channels),
        "crossfade_seconds": mrt.config.crossfade_length,
        "total_samples": int(total_samples),
        "seconds_per_bar": seconds_per_bar,
        "loop_duration_seconds": loop_duration_seconds,
        "guidance_weight": guidance_weight,
        "temperature": temperature,
        "topk": topk,
    }
    return {"audio_base64": audio_b64, "metadata": metadata}

# new endpoint to return a bar-aligned chunk without the need for combined audio

@app.post("/generate_style")
def generate_style(
    bpm: float = Form(...),
    bars: int = Form(8),
    beats_per_bar: int = Form(4),
    styles: str = Form("warmup"),
    style_weights: str = Form(""),
    guidance_weight: float = Form(1.1),
    temperature: float = Form(1.1),
    topk: int = Form(40),
    target_sample_rate: int | None = Form(None),
    intro_bars_to_drop: int = Form(0),
):
    """
    Style-only, bar-aligned generation (no input audio).
    Seeds with 10s of silent context; outputs exactly `bars` at the requested BPM.
    """
    mrt = get_mrt()

    # Override sampling knobs just for this request
    with mrt_overrides(mrt,
                       guidance_weight=guidance_weight,
                       temperature=temperature,
                       topk=topk):
        wav, _ = generate_style_only_with_mrt(
            mrt,
            bpm=bpm,
            bars=bars,
            beats_per_bar=beats_per_bar,
            styles=styles,
            style_weights=style_weights,
            intro_bars_to_drop=intro_bars_to_drop,
        )

    # Determine target SR (defaults to model SR = 48k)
    cur_sr = int(mrt.sample_rate)
    target_sr = int(target_sample_rate or cur_sr)
    x = wav.samples if wav.samples.ndim == 2 else wav.samples[:, None]

    seconds_per_bar = (60.0 / float(bpm)) * int(beats_per_bar)
    expected_secs   = float(bars) * seconds_per_bar

    # Snap exactly to musical length at the requested sample rate
    x = resample_and_snap(x, cur_sr=cur_sr, target_sr=target_sr, seconds=expected_secs)

    audio_b64, total_samples, channels = wav_bytes_base64(x, target_sr)

    metadata = {
        "bpm": int(round(bpm)),
        "bars": int(bars),
        "beats_per_bar": int(beats_per_bar),
        "styles": [s.strip() for s in (styles.split(",") if styles else []) if s.strip()],
        "style_weights": [float(y) for y in style_weights.split(",")] if style_weights else None,
        "sample_rate": int(target_sr),
        "channels": int(channels),
        "crossfade_seconds": mrt.config.crossfade_length,
        "seconds_per_bar": seconds_per_bar,
        "loop_duration_seconds": total_samples / float(target_sr),
        "guidance_weight": guidance_weight,
        "temperature": temperature,
        "topk": topk,
    }
    return {"audio_base64": audio_b64, "metadata": metadata}


# ----------------------------
# the 'keep jamming' button
# ----------------------------

@app.post("/jam/start")
def jam_start(
    loop_audio: UploadFile = File(...),
    bpm: float = Form(...),
    bars_per_chunk: int = Form(4),
    beats_per_bar: int = Form(4),
    styles: str = Form(""),
    style_weights: str = Form(""),
    loop_weight: float = Form(1.0),

    # NEW steering params:
    mean: float = Form(0.0),
    centroid_weights: str = Form(""),

    loudness_mode: str = Form("auto"),
    loudness_headroom_db: float = Form(1.0),
    guidance_weight: float = Form(1.1),
    temperature: float = Form(1.1),
    topk: int = Form(40),
    target_sample_rate: int | None = Form(None),
):
    asset_manager.ensure_assets_loaded(get_mrt())

    # enforce single active jam per GPU
    with jam_lock:
        for sid, w in list(jam_registry.items()):
            if w.is_alive():
                raise HTTPException(status_code=429, detail="A jam is already running. Try again later.")

    # read input + prep context/style (reuse your existing code)
    data = loop_audio.file.read()
    if not data: raise HTTPException(status_code=400, detail="Empty file")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(data); tmp_path = tmp.name

    mrt = get_mrt()
    loop = au.Waveform.from_file(tmp_path).resample(mrt.sample_rate).as_stereo()

    # build tail context + style vec (tail-biased)
    codec_fps = float(mrt.codec.frame_rate)
    ctx_seconds = float(mrt.config.context_length_frames) / codec_fps
    loop_tail = take_bar_aligned_tail(loop, bpm, beats_per_bar, ctx_seconds)

    # Parse client style fields (preserves your semantics)
    text_list = [s.strip() for s in (styles.split(",") if styles else []) if s.strip()]
    try:
        tw = [float(x) for x in style_weights.split(",")] if style_weights else []
    except ValueError:
        tw = []
    try:
        cw = [float(x) for x in centroid_weights.split(",")] if centroid_weights else []
    except ValueError:
        cw = []

    # Compute loop-tail embed once (same as before)
    loop_tail_embed = mrt.embed_style(loop_tail)

    # Build final style vector:
    # - identical to your previous mix when mean==0 and cw is empty
    # - otherwise includes mean and centroid components (weights auto-normalized)
    style_vec = build_style_vector(
        mrt,
        text_styles=text_list,
        text_weights=tw,
        loop_embed=loop_tail_embed,
        loop_weight=float(loop_weight),
        mean_weight=float(mean),
        centroid_weights=cw,
    ).astype(np.float32, copy=False)

    # target SR (default input SR)
    inp_info = sf.info(tmp_path)
    input_sr = int(inp_info.samplerate)
    target_sr = int(target_sample_rate or input_sr)

    params = JamParams(
        bpm=bpm, 
        beats_per_bar=beats_per_bar, 
        bars_per_chunk=bars_per_chunk,
        target_sr=target_sr, 
        loudness_mode=loudness_mode, 
        headroom_db=loudness_headroom_db,
        style_vec=style_vec, 
        ref_loop=loop_tail,                    # For loudness matching
        combined_loop=loop,                    # NEW: Full loop for context setup
        guidance_weight=guidance_weight, 
        temperature=temperature, 
        topk=topk
    )

    worker = JamWorker(mrt, params)
    sid = str(uuid.uuid4())
    with jam_lock:
        jam_registry[sid] = worker
    worker.start()

    return {"session_id": sid}

@app.get("/jam/next")
def jam_next(session_id: str):
    """
    Get the next sequential chunk in the jam session.
    This ensures chunks are delivered in order without gaps.
    """
    with jam_lock:
        worker = jam_registry.get(session_id)
    if worker is None or not worker.is_alive():
        raise HTTPException(status_code=404, detail="Session not found")

    # Get the next sequential chunk (this blocks until ready)
    chunk = worker.get_next_chunk()
    
    if chunk is None:
        raise HTTPException(status_code=408, detail="Chunk not ready within timeout")

    return {
        "chunk": {
            "index": chunk.index,
            "audio_base64": chunk.audio_base64,
            "metadata": chunk.metadata
        }
    }

@app.post("/jam/consume")
def jam_consume(session_id: str = Form(...), chunk_index: int = Form(...)):
    """
    Mark a chunk as consumed by the frontend.
    This helps the worker manage its buffer and generation flow.
    """
    with jam_lock:
        worker = jam_registry.get(session_id)
    if worker is None or not worker.is_alive():
        raise HTTPException(status_code=404, detail="Session not found")

    worker.mark_chunk_consumed(chunk_index)
    
    return {"consumed": chunk_index}



@app.post("/jam/stop")
def jam_stop(session_id: str = Body(..., embed=True)):
    with jam_lock:
        worker = jam_registry.get(session_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="Session not found")

    worker.stop()
    worker.join(timeout=5.0)
    if worker.is_alive():
        # It’s daemon=True, so it won’t block process exit, but report it
        print(f"⚠️ JamWorker {session_id} did not stop within timeout")

    with jam_lock:
        jam_registry.pop(session_id, None)
    return {"stopped": True}

@app.post("/jam/update")
def jam_update(
    session_id: str = Form(...),

    # knobs
    guidance_weight: Optional[float] = Form(None),
    temperature: Optional[float]     = Form(None),
    topk: Optional[int]              = Form(None),

    # styles
    styles: str                      = Form(""),
    style_weights: str               = Form(""),
    loop_weight: Optional[float]     = Form(None),
    use_current_mix_as_style: bool   = Form(False),

    # NEW steering
    mean: Optional[float]            = Form(None),
    centroid_weights: str            = Form(""),
):
    asset_manager.ensure_assets_loaded(get_mrt())

    with jam_lock:
        worker = jam_registry.get(session_id)
    if worker is None or not worker.is_alive():
        raise HTTPException(status_code=404, detail="Session not found")

    # 1) fast knob updates
    if any(v is not None for v in (guidance_weight, temperature, topk)):
        worker.update_knobs(
            guidance_weight=guidance_weight,
            temperature=temperature,
            topk=topk
        )

    # 2) rebuild style only if asked
    wants_style_update = (
        use_current_mix_as_style
        or (styles.strip() != "")
        or (mean is not None)
        or (centroid_weights.strip() != "")
    )
    if not wants_style_update:
        return {"ok": True}

    # --- parse inputs (robust) ---
    text_list = [s.strip() for s in (styles.split(",") if styles else []) if s.strip()]
    try:
        tw = [float(x) for x in style_weights.split(",")] if style_weights else []
    except ValueError:
        tw = []
    try:
        cw = [float(x) for x in centroid_weights.split(",")] if centroid_weights else []
    except ValueError:
        cw = []

    # Clamp centroid weights to available centroids (if loaded)
    max_c = 0 if _CENTROIDS is None else int(_CENTROIDS.shape[0])
    if max_c and len(cw) > max_c:
        cw = cw[:max_c]

    # Snapshot minimal state under lock
    with worker._lock:
        combined_loop = worker.params.combined_loop if use_current_mix_as_style else None
        lw = None
        if use_current_mix_as_style:
            lw = 1.0 if (loop_weight is None) else float(loop_weight)
        mrt = worker.mrt

    # Heavy work OUTSIDE the lock
    loop_embed = None
    if combined_loop is not None:
        loop_embed = mrt.embed_style(combined_loop)

    style_vec = build_style_vector(
        mrt,
        text_styles=text_list,
        text_weights=tw,
        loop_embed=loop_embed,             # None => ignored by builder
        loop_weight=lw,                    # None => ignored by builder
        mean_weight=(None if mean is None else float(mean)),
        centroid_weights=cw,               # [] => ignored by builder
    ).astype(np.float32, copy=False)

    # Swap atomically
    with worker._lock:
        worker.params.style_vec = style_vec

    return {"ok": True}


@app.post("/jam/reseed")
def jam_reseed(session_id: str = Form(...), loop_audio: UploadFile = File(None)):
    with jam_lock:
        worker = jam_registry.get(session_id)
    if worker is None or not worker.is_alive():
        raise HTTPException(status_code=404, detail="Session not found")

    # Option 1: use uploaded new “combined” bounce from the app
    if loop_audio is not None:
        data = loop_audio.file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(data); path = tmp.name
        wav = au.Waveform.from_file(path).resample(worker.mrt.sample_rate).as_stereo()
    else:
        # Option 2: reseed from what we’ve been streaming (the model side)
        # (Usually better to reseed from the Swift-side “combined” mix you trust.)

        s = getattr(worker, "_stream", None)
        if s is None or s.shape[0] == 0:
            raise HTTPException(status_code=400, detail="No internal stream to reseed from")
        wav = au.Waveform(s.astype(np.float32, copy=False), int(worker.mrt.sample_rate)).as_stereo()

    worker.reseed_from_waveform(wav)
    return {"ok": True}

@app.post("/jam/reseed_splice")
def jam_reseed_splice(
    session_id: str = Form(...),
    anchor_bars: float = Form(2.0),              # how much of the original to re-inject
    combined_audio: UploadFile = File(None),     # preferred: Swift supplies the current combined mix
):
    worker = jam_registry.get(session_id)
    if worker is None or not worker.is_alive():
        raise HTTPException(status_code=404, detail="Session not found")

    # Build a waveform to reseed from

    wav = None

    if combined_audio is not None:
        data = combined_audio.file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty combined_audio")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(data)
            path = tmp.name
        wav = au.Waveform.from_file(path).resample(worker.mrt.sample_rate).as_stereo()
    else:
        # Fallback: reseed from the model’s internal stream (less ideal than the Swift-side bounce)
        s = getattr(worker, "_stream", None)
        if s is None or s.shape[0] == 0:
            raise HTTPException(status_code=400, detail="No audio available to reseed from")
        wav = au.Waveform(s.astype(np.float32, copy=False), int(worker.mrt.sample_rate)).as_stereo()

    # Perform the splice reseed
    worker.reseed_splice(wav, anchor_bars=float(anchor_bars))
    return {"ok": True, "anchor_bars": float(anchor_bars)}

@app.get("/jam/status")
def jam_status(session_id: str):
    with jam_lock:
        worker = jam_registry.get(session_id)

    if worker is None:
        raise HTTPException(status_code=404, detail="Session not found")

    running = worker.is_alive()

    # Snapshot safely
    with worker._lock:
        last_generated = int(worker.idx)
        last_delivered = int(worker._last_delivered_index)
        queued = len(worker.outbox)
        buffer_ahead = last_generated - last_delivered
        p = worker.params
        spb = p.beats_per_bar * (60.0 / p.bpm)
        chunk_secs = p.bars_per_chunk * spb

    return {
        "running": running,
        "last_generated_index": last_generated,       # Last chunk that finished generating
        "last_delivered_index": last_delivered,       # Last chunk sent to frontend
        "buffer_ahead": buffer_ahead,                  # How many chunks ahead we are
        "queued_chunks": queued,                       # Total chunks in outbox
        "bpm": p.bpm,
        "beats_per_bar": p.beats_per_bar,
        "bars_per_chunk": p.bars_per_chunk,
        "seconds_per_bar": spb,
        "chunk_duration_seconds": chunk_secs,
        "target_sample_rate": p.target_sr,
        "last_chunk_started_at": worker.last_chunk_started_at,
        "last_chunk_completed_at": worker.last_chunk_completed_at,
    }


@app.get("/health")
def health():
    # 1) Template mode → not ready (encourage duplication on GPU)
    if SPACE_MODE != "serve":
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "status": "template_mode",
                "message": "This Space is a GPU template. Duplicate it and select an L40s/A100-class runtime to use the API.",
                "mode": SPACE_MODE,
            },
        )

    # 2) Runtime hardware probe
    probe = _gpu_probe()
    if not probe["ok"] or not probe["has_gpu"] or probe.get("backend") != "gpu":
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "status": "gpu_unavailable",
                "message": "GPU is not visible to JAX. Select a GPU runtime (e.g., L40s) to serve.",
                "probe": probe,
                "mode": SPACE_MODE,
            },
        )

    # 3) Ready; include operational hints
    warmed = bool(_WARMED)
    with jam_lock:
        active_jams = sum(1 for w in jam_registry.values() if w.is_alive())
    return {
        "ok": True,
        "status": "ready" if warmed else "initializing",
        "mode": SPACE_MODE,
        "warmed": warmed,
        "active_jams": active_jams,
        "probe": probe,
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", "-")
    print(f"📥 {request.method} {request.url.path}?{request.url.query} [rid={rid}]")
    try:
        response = await call_next(request)
    except Exception as e:
        print(f"💥 exception for {request.url.path} [rid={rid}]: {e}")
        raise
    print(f"📤 {response.status_code} {request.url.path} [rid={rid}]")
    return response





# ----------------------------
# websockets route
# ----------------------------


@app.websocket("/ws/jam")
async def ws_jam(websocket: WebSocket):
    await websocket.accept()
    sid = None
    worker = None
    binary_audio = False
    mode = "rt"  # or "bar"

    # NEW: capture ws in closure
    async def send_json(obj):
        return await send_json_safe(websocket, obj)

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            # --- START ---
            if mtype == "start":
                binary_audio = bool(msg.get("binary_audio", False))
                mode = msg.get("mode", "rt")
                params = msg.get("params", {}) or {}
                sid = msg.get("session_id")

                # attach or create
                if sid:
                    with jam_lock:
                        worker = jam_registry.get(sid)
                    if worker is None or not worker.is_alive():
                        await send_json({"type":"error","error":"Session not found"})
                        continue
                else:
                    # optionally accept base64 loop and start a new worker (bar-mode)
                    if mode == "bar":
                        loop_b64 = msg.get("loop_audio_b64")
                        if not loop_b64:
                            await send_json({"type":"error","error":"loop_audio_b64 required for mode=bar when no session_id"})
                            continue
                        loop_bytes = base64.b64decode(loop_b64)
                        # mimic /jam/start
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(loop_bytes); tmp_path = tmp.name
                        # build JamParams similar to /jam/start
                        mrt = get_mrt()
                        model_sr = int(mrt.sample_rate)  # typically 48000
                        # Defaults for WS: raw loudness @ model SR, unless overridden by client:
                        target_sr = int(params.get("target_sr", model_sr))
                        loudness_mode = params.get("loudness_mode", "none")
                        headroom_db = float(params.get("headroom_db", 1.0))
                        loop = au.Waveform.from_file(tmp_path).resample(mrt.sample_rate).as_stereo()

                        codec_fps = float(mrt.codec.frame_rate)
                        ctx_seconds = float(mrt.config.context_length_frames) / codec_fps
                        bpm = float(params.get("bpm", 120.0))
                        bpb = int(params.get("beats_per_bar", 4))
                        loop_tail = take_bar_aligned_tail(loop, bpm, bpb, ctx_seconds)

                        # style vector (loop + extra styles)
                        embeds, weights = [mrt.embed_style(loop_tail)], [float(params.get("loop_weight", 1.0))]
                        extra = [s for s in (params.get("styles","").split(",")) if s.strip()]
                        sw = [float(x) for x in params.get("style_weights","").split(",") if x.strip()]
                        for i, s in enumerate(extra):
                            embeds.append(mrt.embed_style(s.strip()))
                            weights.append(sw[i] if i < len(sw) else 1.0)
                        wsum = sum(weights) or 1.0
                        weights = [w/wsum for w in weights]
                        style_vec = np.sum([w*e for w, e in zip(weights, embeds)], axis=0).astype(np.float32)

                        # target SR fallback: input SR
                        inp_info = sf.info(tmp_path)
                        target_sr = int(params.get("target_sr", int(inp_info.samplerate)))

                        # Build JamParams for WS bar-mode
                        jp = JamParams(
                            bpm=bpm, beats_per_bar=bpb, bars_per_chunk=int(params.get("bars_per_chunk", 8)),
                            target_sr=target_sr,
                            loudness_mode=loudness_mode, headroom_db=headroom_db,
                            style_vec=style_vec,
                            ref_loop=None if loudness_mode == "none" else loop_tail,  # disable match by default
                            combined_loop=loop,
                            guidance_weight=float(params.get("guidance_weight", 1.1)),
                            temperature=float(params.get("temperature", 1.1)),
                            topk=int(params.get("topk", 40)),
                        )
                        worker = JamWorker(get_mrt(), jp)
                        sid = str(uuid.uuid4())
                        with jam_lock:
                            # single active jam per GPU, mirroring /jam/start
                            for _sid, w in list(jam_registry.items()):
                                if w.is_alive():
                                    await send_json({"type":"error","error":"A jam is already running"})
                                    worker = None; sid = None
                                    break
                            if worker is not None:
                                jam_registry[sid] = worker
                                worker.start()

                    else:
                        # mode == "rt" (Colab-style, no loop context)
                        mrt = get_mrt()
                        state = mrt.init_state()

                        # Build silent context (10s) tokens
                        codec_fps   = float(mrt.codec.frame_rate)
                        ctx_seconds = float(mrt.config.context_length_frames) / codec_fps
                        sr = int(mrt.sample_rate)
                        samples = int(max(1, round(ctx_seconds * sr)))
                        silent = au.Waveform(np.zeros((samples, 2), np.float32), sr)
                        tokens = mrt.codec.encode(silent).astype(np.int32)[:, :mrt.config.decoder_codec_rvq_depth]
                        state.context_tokens = tokens

                        # Parse params (including steering)
                        asset_manager.ensure_assets_loaded(get_mrt())
                        styles_str        = params.get("styles", "warmup") or ""
                        style_weights_str = params.get("style_weights", "") or ""
                        mean_w            = float(params.get("mean", 0.0) or 0.0)
                        cw_str            = str(params.get("centroid_weights", "") or "")

                        text_list = [s.strip() for s in styles_str.split(",") if s.strip()]
                        try:
                            text_w = [float(x) for x in style_weights_str.split(",")] if style_weights_str else []
                        except ValueError:
                            text_w = []
                        try:
                            cw = [float(x) for x in cw_str.split(",") if x.strip() != ""]
                        except ValueError:
                            cw = []

                        # Clamp centroid weights to available centroids
                        if _CENTROIDS is not None and len(cw) > int(_CENTROIDS.shape[0]):
                            cw = cw[: int(_CENTROIDS.shape[0])]

                        # Build initial style vector (no loop_embed in rt mode)
                        style_vec = build_style_vector(
                            mrt,
                            text_styles=text_list,
                            text_weights=text_w,
                            loop_embed=None,
                            loop_weight=None,
                            mean_weight=mean_w,
                            centroid_weights=cw,
                        )

                        # Stash rt session fields
                        websocket._mrt   = mrt
                        websocket._state = state
                        websocket._style_cur = style_vec
                        websocket._style_tgt = style_vec
                        websocket._style_ramp_s = float(params.get("style_ramp_seconds", 0.0))

                        websocket._rt_mean              = mean_w
                        websocket._rt_centroid_weights  = cw
                        websocket._rt_running           = True
                        websocket._rt_sr                = sr
                        websocket._rt_topk              = int(params.get("topk", 40))
                        websocket._rt_temp              = float(params.get("temperature", 1.1))
                        websocket._rt_guid              = float(params.get("guidance_weight", 1.1))
                        websocket._pace                 = params.get("pace", "asap")  # "realtime" | "asap"

                        # (Optional) report whether steering assets were loaded
                        assets_ok = (_MEAN_EMBED is not None) or (_CENTROIDS is not None)
                        await send_json({"type": "started", "mode": "rt", "steering_assets": "loaded" if assets_ok else "none"})

                        # kick off the ~2s streaming loop
                        async def _rt_loop():
                            try:
                                mrt = websocket._mrt
                                chunk_secs = (mrt.config.chunk_length_frames * mrt.config.frame_length_samples) / float(mrt.sample_rate)
                                target_next = time.perf_counter()
                                while websocket._rt_running:
                                    mrt.guidance_weight = websocket._rt_guid
                                    mrt.temperature     = websocket._rt_temp
                                    mrt.topk            = websocket._rt_topk

                                    # ramp style
                                    ramp = float(getattr(websocket, "_style_ramp_s", 0.0) or 0.0)
                                    if ramp <= 0.0:
                                        websocket._style_cur = websocket._style_tgt
                                    else:
                                        step = min(1.0, chunk_secs / ramp)
                                        websocket._style_cur = websocket._style_cur + step * (websocket._style_tgt - websocket._style_cur)

                                    wav, new_state = mrt.generate_chunk(state=websocket._state, style=websocket._style_cur)
                                    websocket._state = new_state

                                    x = wav.samples.astype(np.float32, copy=False)
                                    buf = io.BytesIO()
                                    sf.write(buf, x, mrt.sample_rate, subtype="FLOAT", format="WAV")

                                    ok = True
                                    if binary_audio:
                                        try:
                                            await websocket.send_bytes(buf.getvalue())
                                            ok = await send_json({"type": "chunk_meta", "metadata": {"sample_rate": mrt.sample_rate}})
                                        except Exception:
                                            ok = False
                                    else:
                                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                                        ok = await send_json({"type": "chunk", "audio_base64": b64,
                                                            "metadata": {"sample_rate": mrt.sample_rate}})

                                    if not ok:
                                        break

                                    if getattr(websocket, "_pace", "asap") == "realtime":
                                        t1 = time.perf_counter()
                                        target_next += chunk_secs
                                        sleep_s = max(0.0, target_next - t1 - 0.02)
                                        if sleep_s > 0:
                                            await asyncio.sleep(sleep_s)
                            except asyncio.CancelledError:
                                pass
                            except Exception:
                                pass

                        websocket._rt_task = asyncio.create_task(_rt_loop())
                        continue  # skip the “bar-mode started” message below

                await send_json({"type":"started","session_id": sid, "mode": mode})

                # if we’re in bar-mode, begin pushing chunks as they arrive
                if mode == "bar" and worker is not None:
                    async def _pump():
                        while True:
                            if not worker.is_alive():
                                break
                            chunk = worker.get_next_chunk(timeout=60.0)
                            if chunk is None:
                                continue
                            if binary_audio:
                                await websocket.send_bytes(base64.b64decode(chunk.audio_base64))
                                await send_json({"type":"chunk_meta","index":chunk.index,"metadata":chunk.metadata})
                            else:
                                await send_json({"type":"chunk","index":chunk.index,
                                                 "audio_base64":chunk.audio_base64,"metadata":chunk.metadata})
                    asyncio.create_task(_pump())

            # --- UPDATES (bar or rt) ---
            elif mtype == "update":
                if mode == "bar":
                    if not sid:
                        await send_json({"type":"error","error":"No session_id yet"}); return
                    # fan values straight into your existing HTTP handler:
                    res = jam_update(
                        session_id=sid,
                        guidance_weight=msg.get("guidance_weight"),
                        temperature=msg.get("temperature"),
                        topk=msg.get("topk"),
                        styles=msg.get("styles",""),
                        style_weights=msg.get("style_weights",""),
                        loop_weight=msg.get("loop_weight"),
                        use_current_mix_as_style=bool(msg.get("use_current_mix_as_style", False)),
                    )
                    await send_json({"type":"status", **res})  # {"ok": True}
                else:
                    # rt-mode: there’s no JamWorker; update the local knobs/state
                    websocket._rt_temp = float(msg.get("temperature", websocket._rt_temp))
                    websocket._rt_topk = int(msg.get("topk", websocket._rt_topk))
                    websocket._rt_guid = float(msg.get("guidance_weight", websocket._rt_guid))

                    # NEW steering fields
                    if "mean" in msg and msg["mean"] is not None:
                        try: websocket._rt_mean = float(msg["mean"])
                        except: websocket._rt_mean = 0.0

                    if "centroid_weights" in msg:
                        cw = [w.strip() for w in str(msg["centroid_weights"]).split(",") if w.strip() != ""]
                        try:
                            websocket._rt_centroid_weights = [float(x) for x in cw]
                        except:
                            websocket._rt_centroid_weights = []

                    # styles / text weights (optional, comma-separated)
                    styles_str = msg.get("styles", None)
                    style_weights_str = msg.get("style_weights", "")

                    text_list = [s for s in (styles_str.split(",") if styles_str else []) if s.strip()]
                    text_w = [float(x) for x in style_weights_str.split(",")] if style_weights_str else []

                    asset_manager.ensure_assets_loaded(get_mrt())
                    websocket._style_tgt = build_style_vector(
                        websocket._mrt,
                        text_styles=text_list,
                        text_weights=text_w,
                        loop_embed=None,
                        loop_weight=None,
                        mean_weight=float(websocket._rt_mean),
                        centroid_weights=websocket._rt_centroid_weights,
                    )
                    # optionally allow live changes to ramp:
                    if "style_ramp_seconds" in msg:
                        try: websocket._style_ramp_s = float(msg["style_ramp_seconds"])
                        except: pass
                    await send_json({"type":"status","updated":"rt-knobs+style"})

            elif mtype == "consume" and mode == "bar":
                with jam_lock:
                    worker = jam_registry.get(msg.get("session_id"))
                if worker is not None:
                    worker.mark_chunk_consumed(int(msg.get("chunk_index", -1)))

            elif mtype == "reseed" and mode == "bar":
                with jam_lock:
                    worker = jam_registry.get(msg.get("session_id"))
                if worker is None or not worker.is_alive():
                    await send_json({"type":"error","error":"Session not found"}); continue
                loop_b64 = msg.get("loop_audio_b64")
                if not loop_b64:
                    await send_json({"type":"error","error":"loop_audio_b64 required"}); continue
                loop_bytes = base64.b64decode(loop_b64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(loop_bytes); path = tmp.name
                wav = au.Waveform.from_file(path).resample(worker.mrt.sample_rate).as_stereo()
                worker.reseed_from_waveform(wav)
                await send_json({"type":"status","reseeded":True})

            elif mtype == "reseed_splice" and mode == "bar":
                with jam_lock:
                    worker = jam_registry.get(msg.get("session_id"))
                if worker is None or not worker.is_alive():
                    await send_json({"type":"error","error":"Session not found"}); continue
                anchor = float(msg.get("anchor_bars", 2.0))
                b64 = msg.get("combined_audio_b64")
                if b64:
                    data = base64.b64decode(b64)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(data); path = tmp.name
                    wav = au.Waveform.from_file(path).resample(worker.mrt.sample_rate).as_stereo()
                    worker.reseed_splice(wav, anchor_bars=anchor)
                else:
                    # fallback: model-side stream splice
                    worker.reseed_splice(worker.params.combined_loop, anchor_bars=anchor)
                await send_json({"type":"status","splice":anchor})

            elif mtype == "stop":
                if mode == "rt":
                    websocket._rt_running = False
                    task = getattr(websocket, "_rt_task", None)
                    if task is not None:
                        task.cancel()
                        try: await task
                        except asyncio.CancelledError: pass
                    await send_json({"type":"stopped"})
                    break  # <- add this if you want to end the socket after stop

            elif mtype == "ping":
                await send_json({"type":"pong"})

            else:
                await send_json({"type":"error","error":f"Unknown type {mtype}"})

    except WebSocketDisconnect:
        # best-effort cleanup for bar-mode sessions started within this socket (optional)
        pass
    except Exception as e:
        try:
            await send_json({"type":"error","error":str(e)})
        except Exception:
            pass
    finally:
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                await websocket.close()
        except Exception:
            pass


@app.get("/ping")
def ping():
    return {"ok": True}

@app.get("/", response_class=Response)
def read_root():
    """Root endpoint that explains what this API does"""
    try:
        html_file = Path(__file__).parent / "documentation.html"
        html_content = html_file.read_text(encoding='utf-8')
    except FileNotFoundError:
        # Fallback if file is missing
        html_content = """
        <!DOCTYPE html>
        <html><body>
        <h1>MagentaRT Research API</h1>
        <p>Documentation file not found. Please check documentation.html</p>
        </body></html>
        """
    return Response(content=html_content, media_type="text/html")

@app.get("/lil_demo_540p.mp4")
def demo_video():
    return FileResponse(Path(__file__).parent / "lil_demo_540p.mp4", media_type="video/mp4")

@app.get("/tester", response_class=HTMLResponse)
def tester():
    html_path = Path(__file__).parent / "magentaRT_rt_tester.html"
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"}  # avoid sticky caches while iterating
    )