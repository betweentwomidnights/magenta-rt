# utils.py
from __future__ import annotations
import io, base64, math
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# Magenta RT audio types
from magenta_rt import audio as au

# Optional loudness
try:
    import pyloudnorm as pyln
    _HAS_LOUDNORM = True
except Exception:
    _HAS_LOUDNORM = False


# ---------- Loudness ----------
def _measure_lufs(wav: au.Waveform) -> float:
    meter = pyln.Meter(wav.sample_rate)  # BS.1770-4
    return float(meter.integrated_loudness(wav.samples))

def _rms(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    return float(np.sqrt(np.mean(x**2)))

def match_loudness_to_reference(
    ref: au.Waveform,
    target: au.Waveform,
    method: str = "auto",   # "auto"|"lufs"|"rms"|"none"
    headroom_db: float = 1.0
) -> tuple[au.Waveform, dict]:
    stats = {"method": method, "applied_gain_db": 0.0}
    if method == "none":
        return target, stats

    if method == "auto":
        method = "lufs" if _HAS_LOUDNORM else "rms"

    if method == "lufs" and _HAS_LOUDNORM:
        L_ref = _measure_lufs(ref)
        L_tgt = _measure_lufs(target)
        delta_db = L_ref - L_tgt
        gain = 10.0 ** (delta_db / 20.0)
        y = target.samples.astype(np.float32) * gain
        stats.update({"ref_lufs": L_ref, "tgt_lufs_before": L_tgt, "applied_gain_db": delta_db})
    else:
        ra = _rms(ref.samples)
        rb = _rms(target.samples)
        if rb <= 1e-12:
            return target, stats
        gain = ra / rb
        y = target.samples.astype(np.float32) * gain
        stats.update({"ref_rms": ra, "tgt_rms_before": rb, "applied_gain_db": 20*np.log10(max(gain,1e-12))})

    # simple peak “limiter” to keep headroom
    limit = 10 ** (-headroom_db / 20.0)   # e.g., -1 dBFS
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > limit:
        y *= (limit / peak)
        stats["post_peak_limited"] = True
    else:
        stats["post_peak_limited"] = False

    target.samples = y.astype(np.float32)
    return target, stats


# ---------- Stitch / fades / trims ----------
def stitch_generated(chunks, sr: int, xfade_s: float, drop_first_pre_roll: bool = True):
    if not chunks:
        raise ValueError("no chunks")
    xfade_n = int(round(xfade_s * sr))
    if xfade_n <= 0:
        return au.Waveform(np.concatenate([c.samples for c in chunks], axis=0), sr)

    t = np.linspace(0, np.pi/2, xfade_n, endpoint=False, dtype=np.float32)
    eq_in, eq_out = np.sin(t)[:, None], np.cos(t)[:, None]

    first = chunks[0].samples
    if first.shape[0] < xfade_n:
        raise ValueError("chunk shorter than crossfade prefix")

    # 🔧 key change:
    out = first[xfade_n:].copy() if drop_first_pre_roll else first.copy()

    for i in range(1, len(chunks)):
        cur = chunks[i].samples
        if cur.shape[0] < xfade_n:
            continue
        head, tail = cur[:xfade_n], cur[xfade_n:]
        mixed = out[-xfade_n:] * eq_out + head * eq_in
        out = np.concatenate([out[:-xfade_n], mixed, tail], axis=0)

    return au.Waveform(out, sr)

def hard_trim_seconds(wav: au.Waveform, seconds: float) -> au.Waveform:
    n = int(round(seconds * wav.sample_rate))
    return au.Waveform(wav.samples[:n], wav.sample_rate)

def apply_micro_fades(wav: au.Waveform, ms: int = 5) -> None:
    n = int(wav.sample_rate * ms / 1000.0)
    if n > 0 and wav.samples.shape[0] > 2*n:
        env = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        wav.samples[:n]  *= env
        wav.samples[-n:] *= env[::-1]


# ---------- Token context helpers ----------
def make_bar_aligned_context(tokens, bpm, fps=25, ctx_frames=250, beats_per_bar=4):
    frames_per_bar_f = (beats_per_bar * 60.0 / bpm) * fps
    frames_per_bar = int(round(frames_per_bar_f))
    if abs(frames_per_bar - frames_per_bar_f) > 1e-3:
        reps = int(np.ceil(ctx_frames / len(tokens)))
        return np.tile(tokens, (reps, 1))[-ctx_frames:]
    reps = int(np.ceil(ctx_frames / len(tokens)))
    tiled = np.tile(tokens, (reps, 1))
    end = (len(tiled) // frames_per_bar) * frames_per_bar
    if end < ctx_frames:
        return tiled[-ctx_frames:]
    start = end - ctx_frames
    return tiled[start:end]

def take_bar_aligned_tail(wav: au.Waveform, bpm: float, beats_per_bar: int, ctx_seconds: float, max_bars=None) -> au.Waveform:
    spb = (60.0 / bpm) * beats_per_bar
    bars_needed = max(1, int(round(ctx_seconds / spb)))
    if max_bars is not None:
        bars_needed = min(bars_needed, max_bars)
    tail_seconds = bars_needed * spb
    n = int(round(tail_seconds * wav.sample_rate))
    if n >= wav.samples.shape[0]:
        return wav
    return au.Waveform(wav.samples[-n:], wav.sample_rate)


# ---------- SR normalize + snap ----------
def resample_and_snap(x: np.ndarray, cur_sr: int, target_sr: int, seconds: float) -> np.ndarray:
    """
    x: np.ndarray shape (S, C), float32
    Returns: exact-length array (round(seconds*target_sr), C)
    """
    if x.ndim == 1:
        x = x[:, None]
    if cur_sr != target_sr:
        g = gcd(cur_sr, target_sr)
        up, down = target_sr // g, cur_sr // g
        x = resample_poly(x, up, down, axis=0)

    expected_len = int(round(seconds * target_sr))
    if x.shape[0] < expected_len:
        pad = np.zeros((expected_len - x.shape[0], x.shape[1]), dtype=x.dtype)
        x = np.vstack([x, pad])
    elif x.shape[0] > expected_len:
        x = x[:expected_len, :]
    return x.astype(np.float32, copy=False)


# ---------- WAV encode ----------
def wav_bytes_base64(x: np.ndarray, sr: int) -> tuple[str, int, int]:
    """
    x: np.ndarray shape (S, C)
    returns: (base64_wav, total_samples, channels)
    """
    buf = io.BytesIO()
    sf.write(buf, x, sr, subtype="FLOAT", format="WAV")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return b64, int(x.shape[0]), int(x.shape[1])
