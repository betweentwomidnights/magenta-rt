"""
One-shot music generation functions for MagentaRT.

This module contains the core generation functions extracted from the main app
that can be used independently for single-shot music generation tasks.
"""
import math
import numpy as np
from magenta_rt import audio as au
from utils import (
    match_loudness_to_reference, 
    stitch_generated, 
    hard_trim_seconds,
    apply_micro_fades, 
    make_bar_aligned_context, 
    take_bar_aligned_tail
)


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
    intro_bars_to_drop: int = 0,
    progress_cb=None
):
    """
    Generate a continuation of an input loop using MagentaRT.
    """
    
    # ===== CRITICAL FIX: Force codec state isolation =====
    # Create a completely isolated encoding session to prevent
    # audio from previous generations bleeding into this one
    
    # Save original codec state (if any)
    original_codec_state = {}
    codec_attrs_to_clear = [
        '_encode_state', '_decode_state', 
        '_last_encoded', '_last_decoded',
        '_encoder_cache', '_decoder_cache',
        '_buffer', '_frame_buffer'
    ]
    
    for attr in codec_attrs_to_clear:
        if hasattr(mrt.codec, attr):
            original_codec_state[attr] = getattr(mrt.codec, attr)
            setattr(mrt.codec, attr, None)
    
    # Also clear any MRT-level generation state
    mrt_attrs_to_clear = ['_last_state', '_generation_cache']
    for attr in mrt_attrs_to_clear:
        if hasattr(mrt, attr):
            original_codec_state[f'mrt_{attr}'] = getattr(mrt, attr)
            setattr(mrt, attr, None)
    
    try:
        # ============================================================
        
        # Load & prep - Force FRESH file read (no caching)
        loop = au.Waveform.from_file(input_wav_path).resample(mrt.sample_rate).as_stereo()
        
        # CRITICAL: Create a detached copy to prevent reference issues
        loop = au.Waveform(
            loop.samples.copy(),  # Force array copy
            loop.sample_rate
        )

        # Use tail for context
        codec_fps   = float(mrt.codec.frame_rate)
        ctx_seconds = float(mrt.config.context_length_frames) / codec_fps
        loop_for_context = take_bar_aligned_tail(loop, bpm, beats_per_bar, ctx_seconds)
        
        # CRITICAL: Another detached copy before encoding
        loop_for_context = au.Waveform(
            loop_for_context.samples.copy(),
            loop_for_context.sample_rate
        )

        # Force fresh encoding with explicit copy flags
        tokens_full = mrt.codec.encode(loop_for_context).astype(np.int32, copy=True)
        tokens = tokens_full[:, :mrt.config.decoder_codec_rvq_depth]
        
        # CRITICAL: Ensure tokens are not a view
        tokens = np.array(tokens, dtype=np.int32, copy=True, order='C')

        # Bar-aligned token window
        context_tokens = make_bar_aligned_context(
            tokens, bpm=bpm, fps=float(mrt.codec.frame_rate),
            ctx_frames=mrt.config.context_length_frames, beats_per_bar=beats_per_bar
        )
        
        # CRITICAL: Force contiguous memory layout
        context_tokens = np.ascontiguousarray(context_tokens, dtype=np.int32)
        
        # Create completely fresh state
        state = mrt.init_state()
        state.context_tokens = context_tokens

        # STYLE embed - force fresh
        loop_embed = mrt.embed_style(loop_for_context)
        embeds, weights = [np.array(loop_embed, copy=True)], [float(loop_weight)]
        
        if extra_styles:
            for i, s in enumerate(extra_styles):
                if s.strip():
                    e = mrt.embed_style(s.strip())
                    embeds.append(np.array(e, copy=True))
                    w = style_weights[i] if (style_weights and i < len(style_weights)) else 1.0
                    weights.append(float(w))
        
        wsum = float(sum(weights)) or 1.0
        weights = [w / wsum for w in weights]
        combined_style = np.sum([w * e for w, e in zip(weights, embeds)], axis=0)
        combined_style = np.ascontiguousarray(combined_style, dtype=np.float32)

        # --- Length math ---
        seconds_per_bar = beats_per_bar * (60.0 / bpm)
        total_secs      = bars * seconds_per_bar
        drop_bars       = max(0, int(intro_bars_to_drop))
        drop_secs       = min(drop_bars, bars) * seconds_per_bar
        gen_total_secs  = total_secs + drop_secs

        chunk_secs = mrt.config.chunk_length_frames * mrt.config.frame_length_samples / mrt.sample_rate
        steps = int(math.ceil(gen_total_secs / chunk_secs)) + 1

        if progress_cb:
            progress_cb(0, steps)

        # Generate with state isolation
        chunks = []
        for i in range(steps):
            wav, state = mrt.generate_chunk(state=state, style=combined_style)
            # Force copy the waveform samples to prevent reference issues
            wav = au.Waveform(wav.samples.copy(), wav.sample_rate)
            chunks.append(wav)
            if progress_cb:
                progress_cb(i + 1, steps)

        # Rest unchanged...
        stitched = stitch_generated(chunks, mrt.sample_rate, mrt.config.crossfade_length).as_stereo()
        stitched = hard_trim_seconds(stitched, gen_total_secs)

        if drop_secs > 0:
            n_drop = int(round(drop_secs * stitched.sample_rate))
            stitched = au.Waveform(stitched.samples[n_drop:], stitched.sample_rate)

        out = hard_trim_seconds(stitched, total_secs)

        out, loud_stats = apply_barwise_loudness_match(
            out=out,
            ref_loop=loop,
            bpm=bpm,
            beats_per_bar=beats_per_bar,
            method=loudness_mode,
            headroom_db=loudness_headroom_db,
            smooth_ms=50,
        )

        apply_micro_fades(out, 5)

        return out, loud_stats
        
    finally:
        # ===== CLEANUP: Clear codec state after generation =====
        # This prevents audio from THIS generation leaking into the NEXT one
        for attr in codec_attrs_to_clear:
            if hasattr(mrt.codec, attr):
                setattr(mrt.codec, attr, None)
        
        for attr in mrt_attrs_to_clear:
            if hasattr(mrt, attr):
                setattr(mrt, attr, None)
        # =======================================================


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


# loudness matching helper for /generate:

def apply_barwise_loudness_match(
    out: au.Waveform,
    ref_loop: au.Waveform,
    *,
    bpm: float,
    beats_per_bar: int,
    method: str = "auto",
    headroom_db: float = 1.0,
    smooth_ms: int = 50,
) -> tuple[au.Waveform, dict]:
    """
    Bar-locked loudness matching that establishes the correct starting level
    then maintains consistency. Only the first bar is matched to the reference;
    subsequent bars use the same gain to maintain relative dynamics.
    """
    sr = int(out.sample_rate)
    spb = (60.0 / float(bpm)) * int(beats_per_bar)
    bar_len = int(round(spb * sr))

    y = out.samples.astype(np.float32, copy=False)
    if y.ndim == 1: y = y[:, None]
    
    if ref_loop.sample_rate != sr:
        ref = ref_loop.resample(sr).as_stereo().samples.astype(np.float32, copy=False)
    else:
        ref = ref_loop.as_stereo().samples.astype(np.float32, copy=False)

    if ref.ndim == 1: ref = ref[:, None]
    if ref.shape[1] == 1: ref = np.repeat(ref, 2, axis=1)

    from utils import match_loudness_to_reference

    # Measure reference loudness once
    ref_bar_len = min(ref.shape[0], bar_len)
    ref_bar = au.Waveform(ref[:ref_bar_len], sr)
    
    gains_db = []
    out_adj = y.copy()
    need = y.shape[0]
    n_bars = max(1, int(np.ceil(need / float(bar_len))))
    ramp = int(max(0, round(smooth_ms * sr / 1000.0)))
    min_lufs_samples = int(0.4 * sr)
    
    # Calculate gain from bar 0 matching
    first_bar_gain_linear = 1.0
    
    for i in range(n_bars):
        s = i * bar_len
        e = min(need, s + bar_len)
        if e <= s: 
            break
        
        bar_samples = e - s
        tgt_bar = au.Waveform(y[s:e], sr)  # Always read from ORIGINAL

        # First bar: match to reference to establish gain
        if i == 0:
            effective_method = "rms" if bar_samples < min_lufs_samples else method
            matched_bar, stats = match_loudness_to_reference(
                ref_bar, tgt_bar, method=effective_method, headroom_db=headroom_db
            )
            
            # Calculate the linear gain that was applied
            eps = 1e-12
            first_bar_gain_linear = float(np.sqrt(
                (np.mean(matched_bar.samples**2) + eps) / 
                (np.mean(tgt_bar.samples**2) + eps)
            ))
            g = matched_bar.samples.astype(np.float32, copy=False)
        else:
            # Subsequent bars: apply the same gain from bar 0
            g = (tgt_bar.samples * first_bar_gain_linear).astype(np.float32, copy=False)

        # Calculate gain in dB for stats
        if tgt_bar.samples.size > 0:
            eps = 1e-12
            g_lin = float(np.sqrt((np.mean(g**2) + eps) / (np.mean(tgt_bar.samples**2) + eps)))
        else:
            g_lin = 1.0
        gains_db.append(20.0 * np.log10(max(g_lin, 1e-6)))

        # Apply with ramp for smoothness
        if i > 0 and ramp > 0:
            ramp_len = min(ramp, e - s)
            t = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)[:, None]
            out_adj[s:s+ramp_len] = (1.0 - t) * out_adj[s:s+ramp_len] + t * g[:ramp_len]
            out_adj[s+ramp_len:e] = g[ramp_len:e-s]
        else:
            out_adj[s:e] = g

    out.samples = out_adj.astype(np.float32, copy=False)
    return out, {"per_bar_gain_db": gains_db}