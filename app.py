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

import gradio as gr

def create_documentation_interface():
    """Create a Gradio interface for documentation and transparency"""
    
    with gr.Blocks(title="MagentaRT Research API", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🎵 MagentaRT Live Music Generation Research API
        
        **Research-only implementation for iOS app development**
        
        This API uses Google's [MagentaRT](https://github.com/magenta/magenta-realtime) to generate 
        continuous music based on input audio loops for experimental iOS app development.
        """)
        
        with gr.Tabs():
            with gr.Tab("📖 About This Research"):
                gr.Markdown("""
                ## What This API Does
                
                We're exploring AI-assisted loop-based music creation for mobile apps. Websockets are notoriously annoying in ios-swift apps, so I tried to come up with an http version tailored to the loop based nature of an existing swift app. This API provides:
                
                ### 🎹 Single Generation (`/generate`)
                - Upload audio loop + BPM + style parameters
                - Returns 4-8 bars of AI-generated continuation
                - **Performance**: 4 bars in ~9s, 8 bars in ~16s (L40S GPU)
                
                ### 🔄 Continuous Jamming (`/jam/*`)
                - `/jam/start` - Begin continuous generation session
                - `/jam/next` - Get next bar-aligned chunk
                - `/jam/stop` - End session
                - **Performance**: Real-time 8-bar chunks after warmup
                
                ## Technical Specs
                - **Model**: MagentaRT (800M parameter transformer)
                - **Quality**: 48kHz stereo output
                - **Context**: 10-second audio analysis window
                - **Styles**: Text descriptions (e.g., "acid house, techno")
                
                ## Research Goals
                - Seamless AI music generation for loop-based composition
                - Real-time parameter adjustment during generation
                - Mobile-optimized music creation workflows
                """)
            
            with gr.Tab("🔧 API Documentation"):
                gr.Markdown("""
                ## Single Generation Example
                ```bash
                curl -X POST "/generate" \\
                     -F "loop_audio=@drum_loop.wav" \\
                     -F "bpm=120" \\
                     -F "bars=8" \\
                     -F "styles=acid house,techno" \\
                     -F "guidance_weight=5.0" \\
                     -F "temperature=1.1"
                ```
                
                ## Continuous Jamming Example
                ```bash
                # 1. Start session
                SESSION=$(curl -X POST "/jam/start" \\
                    -F "loop_audio=@loop.wav" \\
                    -F "bpm=120" \\
                    -F "bars_per_chunk=8" | jq -r .session_id)
                
                # 2. Get chunks in real-time
                curl "/jam/next?session_id=$SESSION"
                
                # 3. Stop when done
                curl -X POST "/jam/stop" \\
                     -H "Content-Type: application/json" \\
                     -d "{\\"session_id\\": \\"$SESSION\\"}"
                ```
                
                ## Key Parameters
                - **bpm**: 60-200 (beats per minute)
                - **bars**: 1-16 (bars to generate)
                - **styles**: Text descriptions, comma-separated
                - **guidance_weight**: 0.1-10.0 (style adherence)
                - **temperature**: 0.1-2.0 (randomness)
                - **intro_bars_to_drop**: Skip N bars from start
                
                ## Response Format
                ```json
                {
                  "audio_base64": "...",
                  "metadata": {
                    "bpm": 120,
                    "bars": 8,
                    "sample_rate": 48000,
                    "loop_duration_seconds": 16.0
                  }
                }
                ```
                """)
            
            with gr.Tab("📱 iOS App Integration"):
                gr.Markdown("""
                ## How Our iOS App Uses This API
                
                ### User Flow
                1. **Record/Import**: User provides drum or instrument loop
                2. **Parameter Setup**: Set BPM, style, generation settings
                3. **Continuous Generation**: App calls `/jam/start`
                4. **Real-time Playback**: App fetches chunks via `/jam/next`
                5. **Seamless Mixing**: Generated audio mixed into live stream
                
                ### Technical Implementation
                - **Audio Format**: 48kHz WAV for consistency
                - **Chunk Size**: 8 bars (~16 seconds at 120 BPM)
                - **Buffer Management**: 3-5 chunks ahead for smooth playback
                - **Style Updates**: Real-time parameter adjustment via `/jam/update`
                
                ### Networking Considerations
                - **Latency**: ~2-3 seconds per chunk after warmup
                - **Bandwidth**: ~500KB per 8-bar chunk (compressed)
                - **Reliability**: Automatic retry with exponential backoff
                - **Caching**: Local buffer for offline resilience
                """)
            
            with gr.Tab("⚖️ Licensing & Legal"):
                gr.Markdown("""
                ## MagentaRT Licensing
                
                This project uses Google's MagentaRT model under:
                - **Source Code**: Apache License 2.0
                - **Model Weights**: Creative Commons Attribution 4.0 International
                - **Usage Terms**: [See MagentaRT repository](https://github.com/magenta/magenta-realtime)
                
                ### Key Requirements
                - ✅ **Attribution**: Credit MagentaRT in derivative works
                - ✅ **Responsible Use**: Don't infringe copyrights
                - ✅ **No Warranties**: Use at your own risk
                - ✅ **Patent License**: Explicit patent grants included
                
                ## Our Implementation
                - **Purpose**: Research and development only
                - **Non-Commercial**: Experimental iOS app development
                - **Open Source**: Will release implementation under Apache 2.0
                - **Attribution**: Proper credit to Google Research team
                
                ### Required Attribution
                ```
                Generated using MagentaRT
                Copyright 2024 Google LLC
                Licensed under Apache 2.0 and CC-BY 4.0
                Implementation for research purposes
                ```
                """)
            
            with gr.Tab("📊 Performance & Limits"):
                gr.Markdown("""
                ## Current Performance (L40S 48GB)
                
                ### ⚡ Single Generation
                - **4 bars @ 100 BPM**: ~9 seconds
                - **8 bars @ 100 BPM**: ~16 seconds
                - **Memory usage**: ~40GB VRAM during generation
                
                ### 🔄 Continuous Jamming
                - **Warmup**: ~10-15 seconds first chunk
                - **8-bar chunks @ 120 BPM**: Real-time delivery
                - **Buffer ahead**: 3-5 chunks for smooth playback
                
                ## Known Limitations
                
                ### 🎵 Model Limitations (MagentaRT)
                - **Context**: 10-second maximum memory
                - **Training**: Primarily Western instrumental music
                - **Vocals**: Non-lexical only, no lyric conditioning
                - **Structure**: No long-form song arrangement
                - **Inside Swift**: After a few turns of continuous chunks, the swift app works best if you restart the jam from the combined audio again. In this way you might end up with a real jam.
                
                ### 🖥️ Infrastructure Limitations
                - **Concurrency**: Single user jam sessions only
                - **GPU Memory**: 40GB+ VRAM required for stable operation
                - **Latency**: 2+ second minimum for style changes
                - **Uptime**: Research setup, no SLA guarantees
                
                ## Resource Requirements
                - **Minimum**: 24GB VRAM (basic operation, won't operate realtime enough for new chunks coming in)
                - **Recommended**: 48GB VRAM (stable performance) 
                - **CPU**: 8+ cores
                - **System RAM**: 32GB+
                - **Storage**: 50GB+ for model weights
                """)
                
        gr.Markdown("""
        ---
        
        **🔬 Research Project** | **📱 iOS Development** | **🎵 Powered by MagentaRT**
        
        This API is part of ongoing research into AI-assisted music creation for mobile devices.
        For technical details, see the API documentation tabs above.
        """)
    
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

@app.get("/", response_class=Response)
def read_root():
    """Root endpoint that explains what this API does"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>MagentaRT Research API</title></head>
    <body style="font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px;">
        <h1>🎵 MagentaRT Research API</h1>
        <p><strong>Purpose:</strong> AI music generation for iOS app research using Google's MagentaRT</p>
        <h2>Available Endpoints:</h2>
        <ul>
            <li><code>POST /generate</code> - Generate 4-8 bars of music</li>
            <li><code>POST /jam/start</code> - Start continuous jamming</li>
            <li><code>GET /jam/next</code> - Get next chunk</li>
            <li><code>GET /jam/consume</code> - confirm a chunk as consumed</li>
            <li><code>POST /jam/stop</code> - End session</li>
            <li><code>GET /docs</code> - API documentation</li>
        </ul>
        <p><strong>Research Only:</strong> Experimental implementation for iOS app development.</p>
        <p><strong>Licensing:</strong> Uses MagentaRT (Apache 2.0 + CC-BY 4.0). Users responsible for outputs.</p>
        <p>Visit <a href="/docs">/docs</a> for detailed API documentation.</p>
    </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")