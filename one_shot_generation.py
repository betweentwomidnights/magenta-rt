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
):
    """
    Generate a continuation of an input loop using MagentaRT.
    
    Args:
        mrt: MagentaRT instance
        input_wav_path: Path to input audio file
        bpm: Beats per minute
        extra_styles: List of additional text style prompts (optional)
        style_weights: List of weights for style prompts (optional)
        bars: Number of bars to generate
        beats_per_bar: Beats per bar (typically 4)
        loop_weight: Weight for the input loop's style embedding
        loudness_mode: Loudness matching method ("auto", "lufs", "rms", "none")
        loudness_headroom_db: Headroom in dB for peak limiting
        intro_bars_to_drop: Number of intro bars to generate then drop
        
    Returns:
        Tuple of (au.Waveform output, dict loudness_stats)
    """
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

    # STYLE embed (optional: switch to loop_for_context if you want stronger "recent" bias)
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

    # (optional) keep micro fades
    apply_micro_fades(out, 5)

    # Bar-wise loudness match so bar 1 sits right even if the model ramps up
    out, loud_stats = apply_barwise_loudness_match(
        out,
        ref_loop=loop,                 # same source the jam path tiles per chunk
        bpm=bpm,
        beats_per_bar=beats_per_bar,
        method=loudness_mode,
        headroom_db=loudness_headroom_db,
    )

    # Optionally finish with a light peak cap to ~-1 dBFS (no re-scaling)
    out = out.peak_normalize(0.95)


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
    smooth_ms: int = 50,          # small ramp between bars
) -> tuple[au.Waveform, dict]:
    """
    Bar-locked loudness matching. Tiles ref_loop to cover out, then
    per-bar calls match_loudness_to_reference() and applies gains with
    a short cross-ramp between bars for smoothness.
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

    # tile reference to length of out
    need = y.shape[0]
    reps = int(np.ceil(need / float(ref.shape[0]))) if ref.shape[0] else 1
    ref_tiled = np.tile(ref, (max(1, reps), 1))[:need]

    gains_db = []
    out_adj = y.copy()
    n_bars = max(1, int(np.ceil(need / float(bar_len))))
    ramp = int(max(0, round(smooth_ms * sr / 1000.0)))

    for i in range(n_bars):
        s = i * bar_len
        e = min(need, s + bar_len)
        if e <= s: break

        ref_bar = au.Waveform(ref_tiled[s:e], sr)
        tgt_bar = au.Waveform(out_adj[s:e], sr)

        matched_bar, stats = match_loudness_to_reference(
            ref_bar, tgt_bar, method=method, headroom_db=headroom_db
        )
        # compute linear gain we actually applied
        g = matched_bar.samples.astype(np.float32, copy=False)
        if tgt_bar.samples.size > 0:
            # avoid divide-by-zero; infer average gain over the bar
            eps = 1e-12
            g_lin = float(np.sqrt((np.mean(g**2) + eps) / (np.mean(tgt_bar.samples**2) + eps)))
        else:
            g_lin = 1.0
        gains_db.append(20.0 * np.log10(max(g_lin, 1e-6)))

        # write with a short cross-ramp from previous bar
        if i > 0 and ramp > 0:
            r0 = max(s, s + ramp - (e - s))  # clamp if last bar shorter
            t = np.linspace(0.0, 1.0, r0 - s, dtype=np.float32)[:, None]
            out_adj[s:r0] = (1.0 - t) * out_adj[s:r0] + t * g[:r0-s]
            out_adj[r0:e] = g[r0-s:e-s]
        else:
            out_adj[s:e] = g

    out.samples = out_adj.astype(np.float32, copy=False)
    return out, {"per_bar_gain_db": gains_db}