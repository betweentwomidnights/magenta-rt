import os
# Useful XLA GPU optimizations (harmless if a flag is unknown)
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

# ---- Finetune assets (mean & centroids) --------------------------------------
_FINETUNE_REPO_DEFAULT = os.getenv("MRT_ASSETS_REPO", "thepatch/magenta-ft")
_ASSETS_REPO_ID: str | None = None
_MEAN_EMBED: np.ndarray | None = None           # shape (D,) dtype float32
_CENTROIDS: np.ndarray | None = None            # shape (K, D) dtype float32

_STEP_RE = re.compile(r"(?:^|/)checkpoint_(\d+)(?:/|\.tar\.gz|\.tgz)?$")

def _list_ckpt_steps(repo_id: str, revision: str = "main") -> list[int]:
    """
    List available checkpoint steps in a HF model repo without downloading all weights.
    Looks for:
      checkpoint_<step>/
      checkpoint_<step>.tgz | .tar.gz
      archives/checkpoint_<step>.tgz | .tar.gz
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
    steps = set()
    for f in files:
        m = _STEP_RE.search(f)
        if m:
            try:
                steps.add(int(m.group(1)))
            except:
                pass
    return sorted(steps)

def _step_exists(repo_id: str, revision: str, step: int) -> bool:
    return step in _list_ckpt_steps(repo_id, revision)

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

def _load_finetune_assets_from_hf(repo_id: str | None) -> tuple[bool, str]:
    """
    Download & load mean_style_embed.npy and cluster_centroids.npy from a HF model repo.
    Safe to call multiple times; will overwrite globals if successful.
    """
    global _ASSETS_REPO_ID, _MEAN_EMBED, _CENTROIDS
    repo_id = repo_id or _FINETUNE_REPO_DEFAULT
    try:
        from huggingface_hub import hf_hub_download
        mean_path = None
        cent_path = None
        try:
            mean_path = hf_hub_download(repo_id, filename="mean_style_embed.npy", repo_type="model")
        except Exception:
            pass
        try:
            cent_path = hf_hub_download(repo_id, filename="cluster_centroids.npy", repo_type="model")
        except Exception:
            pass

        if mean_path is None and cent_path is None:
            return False, f"No finetune asset files found in repo {repo_id}"

        if mean_path is not None:
            m = np.load(mean_path)
            if m.ndim != 1:
                return False, f"mean_style_embed.npy must be 1-D (got {m.shape})"
        else:
            m = None

        if cent_path is not None:
            c = np.load(cent_path)
            if c.ndim != 2:
                return False, f"cluster_centroids.npy must be 2-D (got {c.shape})"
        else:
            c = None

        # Optional: shape check vs model embedding dim once model is alive
        try:
            d = int(get_mrt().style_model.config.embedding_dim)
            if m is not None and m.shape[0] != d:
                return False, f"mean_style_embed dim {m.shape[0]} != model dim {d}"
            if c is not None and c.shape[1] != d:
                return False, f"cluster_centroids dim {c.shape[1]} != model dim {d}"
        except Exception:
            # Model not built yet; we’ll trust the files and rely on runtime checks later
            pass

        _MEAN_EMBED = m.astype(np.float32, copy=False) if m is not None else None
        _CENTROIDS = c.astype(np.float32, copy=False) if c is not None else None
        _ASSETS_REPO_ID = repo_id
        logging.info("Loaded finetune assets from %s (mean=%s, centroids=%s)",
                     repo_id,
                     "yes" if _MEAN_EMBED is not None else "no",
                     f"{_CENTROIDS.shape[0]}x{_CENTROIDS.shape[1]}" if _CENTROIDS is not None else "no")
        return True, "ok"
    except Exception as e:
        logging.exception("Failed to load finetune assets: %s", e)
        return False, str(e)

def _ensure_assets_loaded():
    # Best-effort lazy load if nothing is loaded yet
    if _MEAN_EMBED is None and _CENTROIDS is None:
        _load_finetune_assets_from_hf(_ASSETS_REPO_ID or _FINETUNE_REPO_DEFAULT)
# ------------------------------------------------------------------------------

def _resolve_checkpoint_dir() -> str | None:
    repo_id = os.getenv("MRT_CKPT_REPO")
    if not repo_id:
        return None
    step = os.getenv("MRT_CKPT_STEP")  # e.g. "1863001"

    root = Path(snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=os.getenv("MRT_CKPT_REV", "main"),
        local_dir="/home/appuser/.cache/mrt_ckpt/repo",
        local_dir_use_symlinks=False,
    ))

    # Prefer an archive if present (more reliable for Zarr/T5X)
    arch_names = [
        f"checkpoint_{step}.tgz",
        f"checkpoint_{step}.tar.gz",
        f"archives/checkpoint_{step}.tgz",
        f"archives/checkpoint_{step}.tar.gz",
    ] if step else []

    cache_root = Path("/home/appuser/.cache/mrt_ckpt/extracted")
    cache_root.mkdir(parents=True, exist_ok=True)
    for name in arch_names:
        arch = root / name
        if arch.is_file():
            out_dir = cache_root / f"checkpoint_{step}"
            marker = out_dir.with_suffix(".ok")
            if not marker.exists():
                out_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(arch, "r:*") as tf:
                    tf.extractall(out_dir)
                marker.write_text("ok")
            # sanity: require .zarray to exist inside the extracted tree
            if not any(out_dir.rglob(".zarray")):
                raise RuntimeError(f"Extracted archive missing .zarray files: {out_dir}")
            return str(out_dir / f"checkpoint_{step}") if (out_dir / f"checkpoint_{step}").exists() else str(out_dir)

    # No archive; try raw folder from repo and sanity check.
    if step:
        raw = root / f"checkpoint_{step}"
        if raw.is_dir():
            if not any(raw.rglob(".zarray")):
                raise RuntimeError(
                    f"Downloaded checkpoint_{step} appears incomplete (no .zarray). "
                    "Upload as a .tgz or push via git from a Unix shell."
                )
            return str(raw)

    # Pick latest if no step
    step_dirs = [d for d in root.iterdir() if d.is_dir() and re.match(r"checkpoint_\\d+$", d.name)]
    if step_dirs:
        pick = max(step_dirs, key=lambda d: int(d.name.split('_')[-1]))
        if not any(pick.rglob(".zarray")):
            raise RuntimeError(f"Downloaded {pick} appears incomplete (no .zarray).")
        return str(pick)

    return None


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

def create_documentation_interface():
    """Create a Gradio interface for documentation and transparency"""
    with gr.Blocks(title="MagentaRT Research API", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            r"""
# 🎵 MagentaRT Live Music Generation Research API

**Research-only implementation for iOS/web app development**

This API uses Google's [MagentaRT](https://github.com/magenta/magenta-realtime) to generate
continuous music either as **bar-aligned chunks over HTTP** or as **low-latency realtime chunks via WebSocket**.
            """
        )

        with gr.Tabs():
            # ------------------------------------------------------------------
            # About & current status
            # ------------------------------------------------------------------
            with gr.Tab("📖 About & Status"):
                gr.Markdown(
                    r"""
## What this is
We're exploring AI‑assisted loop‑based music creation that can run on GPUs (not just TPUs) and stream to apps in realtime.

### Implemented backends
- **HTTP (bar‑aligned):** `/generate`, `/jam/start`, `/jam/next`, `/jam/stop`, `/jam/update`, etc.
- **WebSocket (realtime):** `ws://…/ws/jam` with `mode="rt"` (Colab‑style continuous chunks). New in this build.

## What we learned (GPU notes)
- **L40S 48GB:** comfortably **faster than realtime** → we added a `pace: "realtime"` switch so the server doesn’t outrun playback.
- **L4 24GB:** **consistently just under realtime**; even with pre‑roll buffering, TF32/JAX tunings, reduced chunk size, and the **base** checkpoint, we still see eventual under‑runs.
- **Implication:** For production‑quality realtime, aim for ~**40GB VRAM** per user/session (e.g., **A100 40GB**, or MIG slices ≈ **35–40GB** on newer parts). Smaller GPUs can demo, but sustained realtime is not reliable.

## Model / audio specs
- **Model:** MagentaRT (T5X; decoder RVQ depth = 16)
- **Audio:** 48 kHz stereo, 2.0 s chunks by default, 40 ms crossfade
- **Context:** 10 s rolling context window
                    """
                )

            # ------------------------------------------------------------------
            # HTTP API
            # ------------------------------------------------------------------
            with gr.Tab("🔧 API (HTTP)"):
                gr.Markdown(
                    r"""
### Single Generation
```bash
curl -X POST \
  "$HOST/generate" \
  -F "loop_audio=@drum_loop.wav" \
  -F "bpm=120" \
  -F "bars=8" \
  -F "styles=acid house,techno" \
  -F "guidance_weight=5.0" \
  -F "temperature=1.1"
```

### Continuous Jamming (bar‑aligned, HTTP)
```bash
# 1) Start a session
echo $(curl -s -X POST "$HOST/jam/start" \
  -F "loop_audio=@loop.wav" \
  -F "bpm=120" \
  -F "bars_per_chunk=8") | jq .
# → {"session_id":"…"}

# 2) Pull next chunk (repeat)
curl "$HOST/jam/next?session_id=$SESSION"

# 3) Stop
curl -X POST "$HOST/jam/stop" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"'$SESSION'"}'
```

### Common parameters
- **bpm** *(int)* – beats per minute
- **bars / bars_per_chunk** *(int)* – musical length
- **styles** *(str)* – comma‑separated text prompts (mixed internally)
- **guidance_weight** *(float)* – style adherence (CFG weight)
- **temperature / topk** – sampling controls
- **intro_bars_to_drop** *(int, /generate)* – generate-and-trim intro
                    """
                )

            # ------------------------------------------------------------------
            # WebSocket API: realtime (‘rt’ mode)
            # ------------------------------------------------------------------
            with gr.Tab("🧩 API (WebSocket • rt mode)"):
                gr.Markdown(
                    r"""
Connect to `wss://…/ws/jam` and send a **JSON control stream**. In `rt` mode the server emits ~2 s WAV chunks (or binary frames) continuously.

### Start (client → server)
```jsonc
{
  "type": "start",
  "mode": "rt",
  "binary_audio": false,          // true → raw WAV bytes + separate chunk_meta
  "params": {
    "styles": "heavy metal",     // or "jazz, hiphop"
    "style_weights": "1.0,1.0",  // optional, auto‑normalized
    "temperature": 1.1,
    "topk": 40,
    "guidance_weight": 1.1,
    "pace": "realtime",          // "realtime" | "asap" (default)
    "max_decode_frames": 50       // 50≈2.0s; try 36–45 on smaller GPUs
  }
}
```

### Server events (server → client)
- `{"type":"started","mode":"rt"}` – handshake
- `{"type":"chunk","audio_base64":"…","metadata":{…}}` – base64 WAV
  - `metadata.sample_rate` *(int)* – usually 48000
  - `metadata.chunk_frames` *(int)* – e.g., 50
  - `metadata.chunk_seconds` *(float)* – frames / 25.0
  - `metadata.crossfade_seconds` *(float)* – typically 0.04
- `{"type":"chunk_meta","metadata":{…}}` – sent **after** a binary frame when `binary_audio=true`
- `{"type":"status",…}`, `{"type":"error",…}`, `{"type":"stopped"}`

### Update (client → server)
```jsonc
{
  "type": "update",
  "styles": "jazz, hiphop",
  "style_weights": "1.0,0.8",
  "temperature": 1.2,
  "topk": 64,
  "guidance_weight": 1.0,
  "pace": "realtime",            // optional live flip
  "max_decode_frames": 40         // optional; <= 50
}
```

### Stop / ping
```json
{"type":"stop"}
{"type":"ping"}
```

### Browser quick‑start (schedules seamlessly with 25–40 ms crossfade)
```html
<script>
const XFADE = 0.025; // 25 ms
let ctx, gain, ws, nextTime = 0;
async function start(){
  ctx = new (window.AudioContext||window.webkitAudioContext)();
  gain = ctx.createGain(); gain.connect(ctx.destination);
  ws = new WebSocket("wss://YOUR_SPACE/ws/jam");
  ws.onopen = ()=> ws.send(JSON.stringify({
    type:"start", mode:"rt", binary_audio:false,
    params:{ styles:"warmup", temperature:1.1, topk:40, guidance_weight:1.1, pace:"realtime" }
  }));
  ws.onmessage = async ev => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "chunk" && msg.audio_base64){
      const bin = atob(msg.audio_base64); const buf = new Uint8Array(bin.length);
      for (let i=0;i<bin.length;i++) buf[i] = bin.charCodeAt(i);
      const ab = buf.buffer; const audio = await ctx.decodeAudioData(ab);
      const src = ctx.createBufferSource(); const g = ctx.createGain();
      src.buffer = audio; src.connect(g); g.connect(gain);
      if (nextTime < ctx.currentTime + 0.05) nextTime = ctx.currentTime + 0.12;
      const startAt = nextTime, dur = audio.duration;
      nextTime = startAt + Math.max(0, dur - XFADE);
      g.gain.setValueAtTime(0, startAt);
      g.gain.linearRampToValueAtTime(1, startAt + XFADE);
      g.gain.setValueAtTime(1, startAt + Math.max(0, dur - XFADE));
      g.gain.linearRampToValueAtTime(0, startAt + dur);
      src.start(startAt);
    }
  };
}
</script>
```

### Python client (async)
```python
import asyncio, json, websockets, base64, soundfile as sf, io
async def run(url):
  async with websockets.connect(url) as ws:
    await ws.send(json.dumps({"type":"start","mode":"rt","binary_audio":False,
      "params": {"styles":"warmup","temperature":1.1,"topk":40,"guidance_weight":1.1,"pace":"realtime"}}))
    while True:
      msg = json.loads(await ws.recv())
      if msg.get("type") == "chunk":
        wav = base64.b64decode(msg["audio_base64"])  # bytes of a WAV
        x, sr = sf.read(io.BytesIO(wav), dtype="float32")
        print("chunk", x.shape, sr)
      elif msg.get("type") in ("stopped","error"): break
asyncio.run(run("wss://YOUR_SPACE/ws/jam"))
```
                    """
                )

            # ------------------------------------------------------------------
            # Performance & hardware guidance
            # ------------------------------------------------------------------
            with gr.Tab("📊 Performance & Hardware"):
                gr.Markdown(
                    r"""
### Current observations
- **L40S 48GB** → faster than realtime. Use `pace:"realtime"` to avoid client over‑buffering.
- **L4 24GB** → slightly **below** realtime even with pre‑roll buffering, TF32/Autotune, smaller chunks (`max_decode_frames`), and the **base** checkpoint.

### Practical guidance
- For consistent realtime, target **~40GB VRAM per active stream** (e.g., **A100 40GB**, or MIG slices ≈ **35–40GB** on newer GPUs).
- Keep client‑side **overlap‑add** (25–40 ms) for seamless chunk joins.
- Prefer **`pace:"realtime"`** once playback begins; use **ASAP** only to build a short pre‑roll if needed.
- Optional knob: **`max_decode_frames`** (default **50** ≈ 2.0 s). Reducing to **36–45** can lower per‑chunk latency/VRAM, but doesn’t increase frames/sec throughput.

### Concurrency
This research build is designed for **one active jam per GPU**. Concurrency would require GPU partitioning (MIG) or horizontal scaling with a session scheduler.
                    """
                )

            # ------------------------------------------------------------------
            # Changelog & legal
            # ------------------------------------------------------------------
            with gr.Tab("🗒️ Changelog & Legal"):
                gr.Markdown(
                    r"""
### Recent changes
- New **WebSocket realtime** route: `/ws/jam` (`mode:"rt"`)
- Added server pacing flag: `pace: "realtime" | "asap"`
- Exposed `max_decode_frames` for shorter chunks on smaller GPUs
- Client test page now does proper **overlap‑add** crossfade between chunks

### Licensing
This project uses MagentaRT under:
- **Code:** Apache 2.0
- **Model weights:** CC‑BY 4.0
Please review the MagentaRT repo for full terms.
                    """
                )

        gr.Markdown(
            r"""
---
**🔬 Research Project** | **📱 iOS/Web Development** | **🎵 Powered by MagentaRT**
            """
        )

    return interface

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

# ----------------------------
# Main generation (single combined style vector)
# ----------------------------
def generate_loop_continuation_with_mrt(
    mrt,
    input_wav_path: str,
    bpm: float,
    extra_styles=None,
    style_weights=None,
    bars: int = 8,
    beats_per_bar: int = 4,
    loop_weight: float = 1.0,
    loudness_mode: str = "auto",
    loudness_headroom_db: float = 1.0,
    intro_bars_to_drop: int = 0,             # <— NEW
):
    # Load & prep (unchanged)
    loop = au.Waveform.from_file(input_wav_path).resample(mrt.sample_rate).as_stereo()

    # Use tail for context (your recent change)
    codec_fps   = float(mrt.codec.frame_rate)
    ctx_seconds = float(mrt.config.context_length_frames) / codec_fps
    loop_for_context = take_bar_aligned_tail(loop, bpm, beats_per_bar, ctx_seconds)

    tokens_full = mrt.codec.encode(loop_for_context).astype(np.int32)
    tokens = tokens_full[:, :mrt.config.decoder_codec_rvq_depth]

    # Bar-aligned token window (unchanged)
    context_tokens = make_bar_aligned_context(
        tokens, bpm=bpm, fps=float(mrt.codec.frame_rate),
        ctx_frames=mrt.config.context_length_frames, beats_per_bar=beats_per_bar
    )
    state = mrt.init_state()
    state.context_tokens = context_tokens

    # STYLE embed (optional: switch to loop_for_context if you want stronger “recent” bias)
    loop_embed = mrt.embed_style(loop_for_context)
    embeds, weights = [loop_embed], [float(loop_weight)]
    if extra_styles:
        for i, s in enumerate(extra_styles):
            if s.strip():
                embeds.append(mrt.embed_style(s.strip()))
                w = style_weights[i] if (style_weights and i < len(style_weights)) else 1.0
                weights.append(float(w))
    wsum = float(sum(weights)) or 1.0
    weights = [w / wsum for w in weights]
    combined_style = np.sum([w * e for w, e in zip(weights, embeds)], axis=0).astype(loop_embed.dtype)

    # --- Length math ---
    seconds_per_bar = beats_per_bar * (60.0 / bpm)
    total_secs      = bars * seconds_per_bar
    drop_bars       = max(0, int(intro_bars_to_drop))
    drop_secs       = min(drop_bars, bars) * seconds_per_bar       # clamp to <= bars
    gen_total_secs  = total_secs + drop_secs                       # generate extra

    # Chunk scheduling to cover gen_total_secs
    chunk_secs = mrt.config.chunk_length_frames * mrt.config.frame_length_samples / mrt.sample_rate  # ~2.0
    steps = int(math.ceil(gen_total_secs / chunk_secs)) + 1  # pad then trim

    # Generate
    chunks = []
    for _ in range(steps):
        wav, state = mrt.generate_chunk(state=state, style=combined_style)
        chunks.append(wav)

    # Stitch continuous audio
    stitched = stitch_generated(chunks, mrt.sample_rate, mrt.config.crossfade_length).as_stereo()

    # Trim to generated length (bars + dropped bars)
    stitched = hard_trim_seconds(stitched, gen_total_secs)

    # 👉 Drop the intro bars
    if drop_secs > 0:
        n_drop = int(round(drop_secs * stitched.sample_rate))
        stitched = au.Waveform(stitched.samples[n_drop:], stitched.sample_rate)

    # Final exact-length trim to requested bars
    out = hard_trim_seconds(stitched, total_secs)

    # Final polish AFTER drop
    out = out.peak_normalize(0.95)
    apply_micro_fades(out, 5)

    # Loudness match to input (after drop) so bar 1 sits right
    out, loud_stats = match_loudness_to_reference(
        ref=loop, target=out,
        method=loudness_mode, headroom_db=loudness_headroom_db
    )

    return out, loud_stats

# untested.
# not sure how it will retain the input bpm. we may want to use a metronome instead of silence. i think google might do that.
# does a generation with silent context rather than a combined loop
def generate_style_only_with_mrt(
    mrt,
    bpm: float,
    bars: int = 8,
    beats_per_bar: int = 4,
    styles: str = "warmup",
    style_weights: str = "",
    intro_bars_to_drop: int = 0,
):
    """
    Style-only, bar-aligned generation using a silent context (no input audio).
    Returns: (au.Waveform out, dict loud_stats_or_None)
    """
    # ---- Build a 10s silent context, tokenized for the model ----
    codec_fps   = float(mrt.codec.frame_rate)
    ctx_seconds = float(mrt.config.context_length_frames) / codec_fps
    sr          = int(mrt.sample_rate)

    silent = au.Waveform(np.zeros((int(round(ctx_seconds * sr)), 2), np.float32), sr)
    tokens_full = mrt.codec.encode(silent).astype(np.int32)
    tokens = tokens_full[:, :mrt.config.decoder_codec_rvq_depth]

    state = mrt.init_state()
    state.context_tokens = tokens

    # ---- Style vector (text prompts only, normalized weights) ----
    prompts = [s.strip() for s in (styles.split(",") if styles else []) if s.strip()]
    if not prompts:
        prompts = ["warmup"]
    sw = [float(x) for x in style_weights.split(",")] if style_weights else []
    embeds, weights = [], []
    for i, p in enumerate(prompts):
        embeds.append(mrt.embed_style(p))
        weights.append(sw[i] if i < len(sw) else 1.0)
    wsum = float(sum(weights)) or 1.0
    weights = [w / wsum for w in weights]
    style_vec = np.sum([w * e for w, e in zip(weights, embeds)], axis=0).astype(np.float32)

    # ---- Target length math ----
    seconds_per_bar = beats_per_bar * (60.0 / bpm)
    total_secs      = bars * seconds_per_bar
    drop_bars       = max(0, int(intro_bars_to_drop))
    drop_secs       = min(drop_bars, bars) * seconds_per_bar
    gen_total_secs  = total_secs + drop_secs

    # ~2.0s chunk length from model config
    chunk_secs = (mrt.config.chunk_length_frames * mrt.config.frame_length_samples) / float(mrt.sample_rate)

    # Generate enough chunks to cover total, plus a pad chunk for crossfade headroom
    steps = int(math.ceil(gen_total_secs / chunk_secs)) + 1

    chunks = []
    for _ in range(steps):
        wav, state = mrt.generate_chunk(state=state, style=style_vec)
        chunks.append(wav)

    # Stitch & trim to exact musical length
    stitched = stitch_generated(chunks, mrt.sample_rate, mrt.config.crossfade_length).as_stereo()
    stitched = hard_trim_seconds(stitched, gen_total_secs)

    if drop_secs > 0:
        n_drop = int(round(drop_secs * stitched.sample_rate))
        stitched = au.Waveform(stitched.samples[n_drop:], stitched.sample_rate)

    out = hard_trim_seconds(stitched, total_secs)
    out = out.peak_normalize(0.95)
    apply_micro_fades(out, 5)

    return out, None  # loudness stats not applicable (no reference)

def _combine_styles(mrt, styles_str: str = "", weights_str: str = ""):
    extra = [s.strip() for s in (styles_str or "").split(",") if s.strip()]
    if not extra:
        return mrt.embed_style("warmup")
    sw = [float(x) for x in (weights_str or "").split(",") if x.strip()]
    embeds, weights = [], []
    for i, s in enumerate(extra):
        embeds.append(mrt.embed_style(s))
        weights.append(sw[i] if i < len(sw) else 1.0)
    wsum = sum(weights) or 1.0
    weights = [w/wsum for w in weights]
    import numpy as np
    return np.sum([w*e for w, e in zip(weights, embeds)], axis=0).astype(np.float32)

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
                ckpt_dir = _resolve_checkpoint_dir()  # ← points to checkpoint_1863001
                _MRT = system.MagentaRT(
                    tag=os.getenv("MRT_SIZE", "large"),  # keep 'large' if finetuned from large
                    guidance_weight=5.0,
                    device="gpu",
                    checkpoint_dir=ckpt_dir,             # ← uses your finetune
                    lazy=False,
                )
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

# Kick it off in the background on server start
@app.on_event("startup")
def _kickoff_warmup():
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
    ok, msg = _load_finetune_assets_from_hf(repo_id)
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
    mrt = None
    try:
        mrt = get_mrt()
    except Exception:
        pass
    return {
        "size": os.getenv("MRT_SIZE", "large"),
        "repo": os.getenv("MRT_CKPT_REPO"),
        "revision": os.getenv("MRT_CKPT_REV", "main"),
        "selected_step": os.getenv("MRT_CKPT_STEP"),
        "resolved_ckpt_dir": _resolve_checkpoint_dir(),  # may be None if not yet downloaded
        "loaded": bool(mrt),
    }

@app.get("/model/checkpoints")
def model_checkpoints(repo_id: str, revision: str = "main"):
    steps = _list_ckpt_steps(repo_id, revision)
    return {"repo": repo_id, "revision": revision, "steps": steps, "latest": (steps[-1] if steps else None)}

class ModelSelect(BaseModel):
    size: Optional[Literal["base","large"]] = None
    repo_id: Optional[str] = None
    revision: Optional[str] = "main"
    step: Optional[Union[int, str]] = None   # allow "latest"
    assets_repo_id: Optional[str] = None     # default: follow repo_id
    sync_assets: bool = True                 # load mean/centroids from repo
    prewarm: bool = False                    # call get_mrt() to build right away
    stop_active: bool = True                 # auto-stop jams; else 409
    dry_run: bool = False                    # validate only, don't swap

@app.post("/model/select")
def model_select(req: ModelSelect):
    # --- Current env defaults ---
    cur = {
        "size":    os.getenv("MRT_SIZE", "large"),
        "repo":    os.getenv("MRT_CKPT_REPO"),
        "rev":     os.getenv("MRT_CKPT_REV", "main"),
        "step":    os.getenv("MRT_CKPT_STEP"),
        "assets":  os.getenv("MRT_ASSETS_REPO", _FINETUNE_REPO_DEFAULT),
    }

    # --- Flags for special step values ---
    no_ckpt = isinstance(req.step, str) and req.step.lower() == "none"
    latest  = isinstance(req.step, str) and req.step.lower() == "latest"

    # --- Target selection (do not require repo when no_ckpt) ---
    tgt = {
        "size":   (req.size or cur["size"]),
        "repo":   (None if no_ckpt else (req.repo_id or cur["repo"])),
        "rev":    (req.revision if req.revision is not None else cur["rev"]),
        # None => resolve to "latest" below. Keep None for no_ckpt as well.
        "step":   (None if (no_ckpt or latest) else (str(req.step) if req.step is not None else cur["step"])),
        "assets": (req.assets_repo_id or req.repo_id or cur["assets"]),
    }

    # ---------- CASE 1: run with NO FINETUNE (stock base/large) ----------
    if no_ckpt:
        preview = {
            "target_size": tgt["size"],
            "target_repo": None,
            "target_revision": None,
            "target_step": None,
            "assets_repo": None,
            "assets_probe": {"ok": True, "message": "skipped"},
            "active_jam": _any_jam_running(),
        }
        if req.dry_run:
            return {"ok": True, "dry_run": True, **preview}

        # Jam policy
        if _any_jam_running():
            if req.stop_active:
                _stop_all_jams()
            else:
                raise HTTPException(status_code=409, detail="A jam is running; retry with stop_active=true")

        # Clear checkpoint + asset env so get_mrt() uses stock weights
        for k in ("MRT_CKPT_REPO", "MRT_CKPT_REV", "MRT_CKPT_STEP", "MRT_ASSETS_REPO"):
            os.environ.pop(k, None)
        os.environ["MRT_SIZE"] = str(tgt["size"])

        # Rebuild model and optionally prewarm
        global _MRT
        with _MRT_LOCK:
            _MRT = None
        if req.prewarm:
            get_mrt()

        return {"ok": True, **preview}

    # ---------- CASE 2: select a repo + step (supports "latest") ----------
    if not tgt["repo"]:
        raise HTTPException(status_code=400, detail="repo_id is required for model selection.")

    # 1) enumerate available steps
    steps = _list_ckpt_steps(tgt["repo"], tgt["rev"])
    if not steps:
        return {"ok": False, "error": f"No checkpoint files found in {tgt['repo']}@{tgt['rev']}", "discovered_steps": steps}

    # 2) choose step (explicit or latest)
    chosen_step = int(tgt["step"]) if tgt["step"] is not None else steps[-1]
    if chosen_step not in steps:
        return {"ok": False, "error": f"checkpoint_{chosen_step} not present in {tgt['repo']}@{tgt['rev']}", "discovered_steps": steps}

    # 3) optional finetune assets probe (no downloads, just listing)
    assets_ok, assets_msg = True, "skipped"
    if req.sync_assets:
        try:
            api = HfApi()
            files = set(api.list_repo_files(repo_id=tgt["assets"], repo_type="model"))
            if ("mean_style_embed.npy" not in files) and ("cluster_centroids.npy" not in files):
                assets_ok, assets_msg = False, f"No finetune asset files in {tgt['assets']}"
            else:
                assets_msg = "found"
        except Exception as e:
            assets_ok, assets_msg = False, f"probe failed: {e}"

    preview = {
        "target_size": tgt["size"],
        "target_repo": tgt["repo"],
        "target_revision": tgt["rev"],
        "target_step": chosen_step,
        "assets_repo": (tgt["assets"] if req.sync_assets else None),
        "assets_probe": {"ok": assets_ok, "message": assets_msg},
        "active_jam": _any_jam_running(),
    }

    if req.dry_run:
        return {"ok": True, "dry_run": True, **preview}

    # Jam policy
    if _any_jam_running():
        if req.stop_active:
            _stop_all_jams()
        else:
            raise HTTPException(status_code=409, detail="A jam is running; retry with stop_active=true")

    # 4) atomic swap with rollback
    old_env = {
        "MRT_SIZE":         os.getenv("MRT_SIZE"),
        "MRT_CKPT_REPO":    os.getenv("MRT_CKPT_REPO"),
        "MRT_CKPT_REV":     os.getenv("MRT_CKPT_REV"),
        "MRT_CKPT_STEP":    os.getenv("MRT_CKPT_STEP"),
        "MRT_ASSETS_REPO":  os.getenv("MRT_ASSETS_REPO"),
    }
    try:
        os.environ["MRT_SIZE"]      = str(tgt["size"])
        os.environ["MRT_CKPT_REPO"] = str(tgt["repo"])
        os.environ["MRT_CKPT_REV"]  = str(tgt["rev"])
        os.environ["MRT_CKPT_STEP"] = str(chosen_step)
        if req.sync_assets:
            os.environ["MRT_ASSETS_REPO"] = str(tgt["assets"])

        # force rebuild
        global _MRT
        with _MRT_LOCK:
            _MRT = None

        # optionally load finetune assets now
        if req.sync_assets:
            _load_finetune_assets_from_hf(os.getenv("MRT_ASSETS_REPO"))

        # optional prewarm to amortize JIT
        if req.prewarm:
            get_mrt()

        return {"ok": True, **preview}
    except Exception as e:
        # rollback on error
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        with _MRT_LOCK:
            _MRT = None
        try:
            get_mrt()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Swap failed: {e}")



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
    _ensure_assets_loaded()

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
    _ensure_assets_loaded()

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
    return {"ok": True}

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
                        _ensure_assets_loaded()
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

                    _ensure_assets_loaded()
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
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>MagentaRT Research API</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 860px; margin: 48px auto; padding: 0 20px; color:#111; }
        code, pre { background:#f6f8fa; border:1px solid #eaecef; border-radius:6px; padding:2px 6px; }
        pre { padding:12px; overflow:auto; }
        .muted { color:#555; }
        ul { line-height: 1.8; }
      </style>
    </head>
    <body>
      <h1>🎵 MagentaRT Research API</h1>
      <p class="muted"><strong>Purpose:</strong> AI music generation for iOS/web app research using Google's MagentaRT.</p>

      <h2>Available Endpoints</h2>
      <ul>
        <li><code>POST /generate</code> – Generate 4–8 bars of music (HTTP, bar-aligned)</li>
        <li><code>POST /jam/start</code> – Start continuous jamming (HTTP)</li>
        <li><code>GET /jam/next</code> – Get next chunk (HTTP)</li>
        <li><code>POST /jam/consume</code> – Confirm a chunk as consumed (HTTP)</li>
        <li><code>POST /jam/stop</code> – End session (HTTP)</li>
        <li><code>WEBSOCKET /ws/jam</code> – Realtime streaming (<code>mode="rt"</code>)</li>
        <li><code>GET /docs</code> – API documentation (Gradio)</li>
      </ul>

      <h2>WebSocket Quick Start (rt mode)</h2>
      <p>Connect to <code>wss://&lt;your-space&gt;/ws/jam</code> and send:</p>
      <pre>{
  "type": "start",
  "mode": "rt",
  "binary_audio": false,
  "params": {
    "styles": "warmup",
    "temperature": 1.1,
    "topk": 40,
    "guidance_weight": 1.1,
    "pace": "realtime",          // or "asap" to bootstrap quickly
    "max_decode_frames": 50      // default ~2.0s; try 36–45 on smaller GPUs
  }
}</pre>
      <p>Update parameters live:</p>
      <pre>{
  "type": "update",
  "styles": "jazz, hiphop",
  "style_weights": "1.0,0.8",
  "temperature": 1.2,
  "topk": 64,
  "guidance_weight": 1.0,
  "pace": "realtime",
  "max_decode_frames": 40
}</pre>
      <p>Stop:</p>
      <pre>{"type":"stop"}</pre>

      <h2>Notes</h2>
      <ul>
        <li>Audio: 48 kHz stereo, ~2.0 s chunks by default with ~40 ms crossfade.</li>
        <li>L40S 48GB: faster than realtime → prefer <code>pace: "realtime"</code>.</li>
        <li>L4 24GB: slightly under realtime even with pre-roll and tuning.</li>
        <li>For sustained realtime, target ~40 GB VRAM per active stream (e.g., A100 40GB or ≈35–40 GB MIG slice).</li>
      </ul>

      <p class="muted"><strong>Licensing:</strong> Uses MagentaRT (Apache 2.0 + CC-BY 4.0). Users are responsible for outputs.</p>
      <p>See <a href="/docs">/docs</a> for full API details and client examples.</p>
    </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")