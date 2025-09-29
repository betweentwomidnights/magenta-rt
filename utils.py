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
def make_bar_aligned_context(tokens, bpm, fps=25.0, ctx_frames=250, beats_per_bar=4):
    """
    Return a ctx_frames-long slice of `tokens` whose **end** lands on an integer
    bar boundary in codec-frame space (model runs at `fps`, typically 25).
    """

    if tokens is None:
        raise ValueError("tokens is None")
    tokens = np.asarray(tokens)
    if tokens.ndim == 1:
        tokens = tokens[:, None]

    T = tokens.shape[0]
    if T == 0:
        return tokens

    fps = float(fps)

    # float frames per bar (e.g., ~65.934 at 91 BPM for 4/4 @ 25fps)
    frames_per_bar_f = (beats_per_bar * 60.0 / float(bpm)) * fps

    # >>> KEY FIX: quantize bar length to an integer number of codec frames
    frames_per_bar_i = max(1, int(round(frames_per_bar_f)))

    # Tile so we can always snap the *end* to a bar boundary and still have ctx_frames
    reps = int(np.ceil((ctx_frames + T) / float(T))) + 1
    tiled = np.tile(tokens, (reps, 1))
    total = tiled.shape[0]

    # How many whole integer bars fit in the tiled sequence?
    k_bars = total // frames_per_bar_i
    if k_bars <= 0:
        return tiled[-ctx_frames:]

    # Snap END to an exact integer multiple of frames_per_bar_i
    end_idx = int(k_bars * frames_per_bar_i)
    end_idx = min(max(end_idx, ctx_frames), total)
    start_idx = end_idx - ctx_frames
    if start_idx < 0:
        start_idx = 0
        end_idx = ctx_frames

    window = tiled[start_idx:end_idx]

    # Guard off-by-one
    if window.shape[0] < ctx_frames:
        pad = np.tile(tokens, (int(np.ceil((ctx_frames - window.shape[0]) / T)), 1))
        window = np.vstack([window, pad])[:ctx_frames]
    elif window.shape[0] > ctx_frames:
        window = window[-ctx_frames:]

    return window



def take_bar_aligned_tail(
    wav: au.Waveform,
    bpm: float,
    beats_per_bar: int,
    ctx_seconds: float,
    max_bars=None
) -> au.Waveform:
    """
    Take a tail whose length is an integer number of bars, with the END aligned
    to a bar boundary. Uses ceil for bars_needed so we never under-fill the context.
    """
    import math

    # seconds per bar
    spb = (60.0 / float(bpm)) * float(beats_per_bar)

    # Pick enough whole bars to cover ctx_seconds (avoid underfilling on round-down).
    # The small epsilon avoids an extra bar due to FP jitter when ctx_seconds ~= k * spb.
    eps = 1e-9
    bars_needed = max(1, int(math.ceil((float(ctx_seconds) - eps) / spb)))

    if max_bars is not None:
        bars_needed = min(bars_needed, int(max_bars))

    # Convert bars -> samples (do rounding once at the end for stability)
    samples_per_bar_f = spb * float(wav.sample_rate)
    n = int(round(bars_needed * samples_per_bar_f))

    total = int(wav.samples.shape[0])
    if n >= total:
        # Not enough audio to take that many bars—return as-is (current behavior).
        return wav

    start = total - n
    return au.Waveform(wav.samples[start:], wav.sample_rate)


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

def _ratio(out_sr: int, in_sr: int) -> tuple[int, int]:
    g = gcd(int(out_sr), int(in_sr))
    return int(out_sr) // g, int(in_sr) // g

class StreamingResampler:
    """
    Stateful streaming resampler.
    Prefers soxr (best), then libsamplerate; final fallback is block resample_poly.
    Always pass float32 arrays shaped (S, C).
    """
    def __init__(self, in_sr: int, out_sr: int, channels: int = 2, quality: str = "VHQ"):
        self.in_sr = int(in_sr)
        self.out_sr = int(out_sr)
        self.channels = int(channels)
        self.quality = quality
        self._backend = None

        # Try soxr first
        try:
            import soxr  # pip install soxr
            self._backend = "soxr"
            # dtype float32 keeps things consistent with the rest of your code
            self._rs = soxr.Resampler(
                self.in_sr,
                self.out_sr,
                channels=self.channels,
                dtype="float32",
                quality=self.quality,  # "Q", "HQ", "VHQ"
            )
        except Exception:
            # Try libsamplerate
            try:
                import samplerate  # pip install samplerate
                self._backend = "samplerate"
                # sinc_best == highest quality; you can choose 'sinc_medium' for speed
                self._rs = samplerate.Resampler(converter_type="sinc_best", channels=self.channels)
            except Exception:
                # Last resort: block resample (not truly streaming)
                from scipy.signal import resample_poly
                self._backend = "scipy"
                self._resample_poly = resample_poly
                self._L, self._M = _ratio(self.out_sr, self.in_sr)
                # Keep a tiny tail to help transitions (still not perfect vs true streaming)
                self._hist = np.zeros((0, self.channels), dtype=np.float32)

    def process(self, x: np.ndarray, final: bool = False) -> np.ndarray:
        """Feed a chunk (S, C) and get resampled chunk (S', C). Keep calling in order."""
        if x.size == 0 and not final:
            # nothing to do
            return np.zeros((0, self.channels), dtype=np.float32)

        if self._backend == "soxr":
            return self._rs.process(x, final=final)

        elif self._backend == "samplerate":
            import samplerate
            ratio = float(self.out_sr) / float(self.in_sr)
            # end_of_input=True flushes tail on the last call
            y = self._rs.process(x, ratio, end_of_input=final)
            # libsamplerate returns (S', C)
            return y.astype(np.float32, copy=False)

        # --- scipy fallback (block, not truly streaming) ---
        # We concatenate a short history to reduce block edge artifacts
        x_ext = x if self._hist.size == 0 else np.vstack([self._hist, x])
        y = self._resample_poly(x_ext, up=self._L, down=self._M, axis=0).astype(np.float32, copy=False)

        # Heuristic: drop the portion corresponding roughly to the history to avoid duplicate content
        # (Not perfect, but helps a lot when chunks are reasonably sized.)
        drop = int(round(self._hist.shape[0] * self.out_sr / self.in_sr))
        y = y[drop:] if drop < y.shape[0] else np.zeros((0, self.channels), dtype=np.float32)

        # Keep a small input tail for the next call (say ~ 4 ms at in_sr)
        tail_samples = max(int(0.004 * self.in_sr), 1)
        self._hist = x[-tail_samples:] if x.shape[0] >= tail_samples else x.copy()
        if final:
            self._hist = np.zeros((0, self.channels), dtype=np.float32)
        return y

    def flush(self) -> np.ndarray:
        """Drain converter tail (call at stop)."""
        if self._backend == "soxr":
            return self._rs.process(np.zeros((0, self.channels), dtype=np.float32), final=True)
        elif self._backend == "samplerate":
            ratio = float(self.out_sr) / float(self.in_sr)
            return self._rs.process(np.zeros((0, self.channels), dtype=np.float32), ratio, end_of_input=True)
        else:
            # nothing meaningful to flush in scipy fallback
            return np.zeros((0, self.channels), dtype=np.float32)
