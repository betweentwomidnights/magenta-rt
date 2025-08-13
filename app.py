from magenta_rt import system, audio as au
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException, Response
import tempfile, io, base64, math, threading
from fastapi.middleware.cors import CORSMiddleware
from contextlib import contextmanager
import soundfile as sf
import numpy as np
from math import gcd
from scipy.signal import resample_poly
from utils import (
    match_loudness_to_reference, stitch_generated, hard_trim_seconds,
    apply_micro_fades, make_bar_aligned_context, take_bar_aligned_tail,
    resample_and_snap, wav_bytes_base64
)

from jam_worker import JamWorker, JamParams, JamChunk
import uuid, threading

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
        tokens, bpm=bpm, fps=int(mrt.codec.frame_rate),
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
                _MRT = system.MagentaRT(tag="base", guidance_weight=1.0, device="gpu", lazy=False)
    return _MRT

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
    loudness_mode: str = Form("auto"),
    loudness_headroom_db: float = Form(1.0),
    guidance_weight: float = Form(1.1),
    temperature: float = Form(1.1),
    topk: int = Form(40),
    target_sample_rate: int | None = Form(None),
):
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

    # style vec = normalized mix of loop_tail + extra styles
    embeds, weights = [mrt.embed_style(loop_tail)], [float(loop_weight)]
    extra = [s for s in (styles.split(",") if styles else []) if s.strip()]
    sw = [float(x) for x in style_weights.split(",")] if style_weights else []
    for i, s in enumerate(extra):
        embeds.append(mrt.embed_style(s.strip()))
        weights.append(sw[i] if i < len(sw) else 1.0)
    wsum = sum(weights) or 1.0
    weights = [w / wsum for w in weights]
    style_vec = np.sum([w * e for w, e in zip(weights, embeds)], axis=0).astype(embeds[0].dtype)

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
def jam_update(session_id: str = Form(...),
               guidance_weight: float | None = Form(None),
               temperature: float | None = Form(None),
               topk: int | None = Form(None)):
    with jam_lock:
        worker = jam_registry.get(session_id)
    if worker is None or not worker.is_alive():
        raise HTTPException(status_code=404, detail="Session not found")
    worker.update_knobs(guidance_weight=guidance_weight, temperature=temperature, topk=topk)
    return {"ok": True}

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