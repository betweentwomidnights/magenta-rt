# jam_worker.py - Bar-locked spool rewrite
from __future__ import annotations

import os

import threading, time
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Dict, Tuple, List

import numpy as np
from magenta_rt import audio as au

from utils import (
    StreamingResampler,
    match_loudness_to_reference,
    make_bar_aligned_context,
    take_bar_aligned_tail,
    wav_bytes_base64,
)

# -----------------------------
# Data classes
# -----------------------------

@dataclass
class JamParams:
    bpm: float
    beats_per_bar: int
    bars_per_chunk: int
    target_sr: int
    loudness_mode: str = "auto"
    headroom_db: float = 1.0
    style_vec: Optional[np.ndarray] = None
    ref_loop: Optional[au.Waveform] = None
    combined_loop: Optional[au.Waveform] = None
    guidance_weight: float = 1.1
    temperature: float = 1.1
    topk: int = 40
    style_ramp_seconds: float = 8.0  # 0 => instant (current behavior), try 6.0–10.0 for gentle glides


@dataclass
class JamChunk:
    index: int
    audio_base64: str
    metadata: dict


# -----------------------------
# Helpers
# -----------------------------

class BarClock:
    """Sample-domain bar clock with drift-free absolute boundaries."""
    def __init__(self, target_sr: int, bpm: float, beats_per_bar: int, base_offset_samples: int = 0):
        self.sr = int(target_sr)
        self.bpm = Fraction(str(bpm))  # exact decimal to avoid FP drift
        self.beats_per_bar = int(beats_per_bar)
        self.bar_samps = Fraction(self.sr * 60 * self.beats_per_bar, 1) / self.bpm
        self.base = int(base_offset_samples)

    def bounds_for_chunk(self, chunk_index: int, bars_per_chunk: int) -> Tuple[int, int]:
        start_f = self.base + self.bar_samps * (chunk_index * bars_per_chunk)
        end_f   = self.base + self.bar_samps * ((chunk_index + 1) * bars_per_chunk)
        return int(round(start_f)), int(round(end_f))

    def seconds_per_bar(self) -> float:
        return float(self.beats_per_bar) * (60.0 / float(self.bpm))


# -----------------------------
# Worker
# -----------------------------

class JamWorker(threading.Thread):
    FRAMES_PER_SECOND: float | None = None  # filled in __init__ once codec is available
    """Generates continuous audio with MagentaRT, spools it at target SR,
    and emits *sample-accurate*, bar-aligned chunks (no FPS drift)."""

    def __init__(self, mrt, params: JamParams):
        super().__init__(daemon=True)
        self.mrt = mrt
        self.params = params

        # external callers (FastAPI endpoints) use this for atomic updates
        self._lock = threading.RLock()

        # generation state
        self.state = self.mrt.init_state()
        self.mrt.guidance_weight = float(self.params.guidance_weight)
        self.mrt.temperature     = float(self.params.temperature)
        self.mrt.topk            = int(self.params.topk)

        

        # codec/setup
        self._codec_fps = float(self.mrt.codec.frame_rate)
        JamWorker.FRAMES_PER_SECOND = self._codec_fps
        self._ctx_frames = int(self.mrt.config.context_length_frames)
        self._ctx_seconds = self._ctx_frames / self._codec_fps

        # model stream (model SR) for internal continuity/crossfades
        self._model_stream: Optional[np.ndarray] = None
        self._model_sr = int(self.mrt.sample_rate)

        # style vector (already normalized upstream)
        self._style_vec = (None if self.params.style_vec is None
                   else np.array(self.params.style_vec, dtype=np.float32, copy=True))
        self._chunk_secs = (
            self.mrt.config.chunk_length_frames * self.mrt.config.frame_length_samples
        ) / float(self._model_sr)  # ≈ 2.0 s by default

        # target-SR in-RAM spool (what we cut loops from)
        if int(self.params.target_sr) != int(self._model_sr):
            self._rs = StreamingResampler(self._model_sr, int(self.params.target_sr), channels=2)
        else:
            self._rs = None
        self._spool = np.zeros((0, 2), dtype=np.float32)   # (S,2) target SR
        self._spool_written = 0                            # absolute frames written into spool

        self._pending_tail_model = None      # type: Optional[np.ndarray]  # last tail at model SR
        self._pending_tail_target_len = 0    # number of target-SR samples last tail contributed

        # bar clock: start with offset 0; if you have a downbeat estimator, set base later
        self._bar_clock = BarClock(self.params.target_sr, self.params.bpm, self.params.beats_per_bar, base_offset_samples=0)

        # emission counters
        self.idx = 0  # next chunk index to *produce*
        self._next_to_deliver = 0  # next chunk index to hand out via get_next_chunk()
        self._last_consumed_index = -1  # updated via mark_chunk_consumed(); generation throttle uses this

        # outbox and synchronization
        self._outbox: Dict[int, JamChunk] = {}
        self._cv = threading.Condition()

        # control flags
        self._stop_event = threading.Event()
        self._max_buffer_ahead = 1

        # reseed queues (install at next bar boundary after emission)
        self._pending_reseed: Optional[dict] = None           # legacy full reset path (kept for fallback)
        self._pending_token_splice: Optional[dict] = None     # seamless token splice

        # Prepare initial context from combined loop (best musical alignment)
        if self.params.combined_loop is not None:
            self._install_context_from_loop(self.params.combined_loop)

    # ---------- lifecycle ----------

    def set_buffer_seconds(self, seconds: float):
        """Clamp how far ahead we allow, in *seconds* of audio."""
        chunk_secs = float(self.params.bars_per_chunk) * self._bar_clock.seconds_per_bar()
        max_chunks = max(0, int(round(seconds / max(chunk_secs, 1e-6))))
        with self._cv:
            self._max_buffer_ahead = max_chunks

    def set_buffer_chunks(self, k: int):
        with self._cv:
            self._max_buffer_ahead = max(0, int(k))

    def stop(self):
        self._stop_event.set()

    # FastAPI reads this to block until the next sequential chunk is ready
    def get_next_chunk(self, timeout: float = 30.0) -> Optional[JamChunk]:
        deadline = time.time() + timeout
        with self._cv:
            while True:
                c = self._outbox.get(self._next_to_deliver)
                if c is not None:
                    self._next_to_deliver += 1
                    return c
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=min(0.25, remaining))

    def mark_chunk_consumed(self, chunk_index: int):
        # This lets the generator run ahead, but not too far
        with self._cv:
            self._last_consumed_index = max(self._last_consumed_index, int(chunk_index))
            # purge old chunks to cap memory
            for k in list(self._outbox.keys()):
                if k < self._last_consumed_index - 1:
                    self._outbox.pop(k, None)

    def update_knobs(self, *, guidance_weight=None, temperature=None, topk=None):
        with self._lock:
            if guidance_weight is not None:
                self.params.guidance_weight = float(guidance_weight)
            if temperature is not None:
                self.params.temperature = float(temperature)
            if topk is not None:
                self.params.topk = int(topk)
            # push into mrt
            self.mrt.guidance_weight = float(self.params.guidance_weight)
            self.mrt.temperature     = float(self.params.temperature)
            self.mrt.topk            = int(self.params.topk)

    # ---------- context / reseed ----------

    def _expected_token_shape(self) -> Tuple[int, int]:
        F = int(self._ctx_frames)
        D = int(self.mrt.config.decoder_codec_rvq_depth)
        return F, D

    def _coerce_tokens(self, toks: np.ndarray) -> np.ndarray:
        """Force tokens to (context_length_frames, rvq_depth), padding/trimming as needed.
        Pads missing frames by repeating the last frame (safer than zeros for RVQ stacks)."""
        F, D = self._expected_token_shape()
        if toks.ndim != 2:
            toks = np.atleast_2d(toks)
        # depth first
        if toks.shape[1] > D:
            toks = toks[:, :D]
        elif toks.shape[1] < D:
            pad_cols = np.tile(toks[:, -1:], (1, D - toks.shape[1]))
            toks = np.concatenate([toks, pad_cols], axis=1)
        # frames
        if toks.shape[0] < F:
            if toks.shape[0] == 0:
                toks = np.zeros((1, D), dtype=np.int32)
            pad = np.repeat(toks[-1:, :], F - toks.shape[0], axis=0)
            toks = np.concatenate([pad, toks], axis=0)
        elif toks.shape[0] > F:
            toks = toks[-F:, :]
        if toks.dtype != np.int32:
            toks = toks.astype(np.int32, copy=False)
        return toks

    def _encode_exact_context_tokens(self, loop: au.Waveform) -> np.ndarray:
        """Build *exactly* context_length_frames worth of tokens (e.g., 250 @ 25fps),
        while ensuring the *end* of the audio lands on a bar boundary.
        Strategy: take the largest integer number of bars <= ctx_seconds as the tail,
        then left-fill from just before that tail (wrapping if needed) to reach exactly
        ctx_seconds; finally, pad/trim to exact samples and, as a last resort, pad/trim
        tokens to the expected frame count.
        """
        wav = loop.as_stereo().resample(self._model_sr)
        data = wav.samples.astype(np.float32, copy=False)
        if data.ndim == 1:
            data = data[:, None]

        spb = self._bar_clock.seconds_per_bar()
        ctx_sec = float(self._ctx_seconds)
        sr = int(self._model_sr)

        # bars that fit fully inside ctx_sec (at least 1)
        bars_fit = max(1, int(ctx_sec // spb))
        tail_len_samps = int(round(bars_fit * spb * sr))

        # ensure we have enough source by tiling
        need = int(round(ctx_sec * sr)) + tail_len_samps
        if data.shape[0] == 0:
            data = np.zeros((1, 2), dtype=np.float32)
        reps = int(np.ceil(need / float(data.shape[0])))
        tiled = np.tile(data, (reps, 1))

        end = tiled.shape[0]
        tail = tiled[end - tail_len_samps:end]

        # left-fill to reach exact ctx samples (keeps end-of-bar alignment)
        ctx_samps = int(round(ctx_sec * sr))
        pad_len = ctx_samps - tail.shape[0]
        if pad_len > 0:
            pre = tiled[end - tail_len_samps - pad_len:end - tail_len_samps]
            ctx = np.concatenate([pre, tail], axis=0)
        else:
            ctx = tail[-ctx_samps:]

        # final snap to *exact* ctx samples
        if ctx.shape[0] < ctx_samps:
            pad = np.zeros((ctx_samps - ctx.shape[0], ctx.shape[1]), dtype=np.float32)
            ctx = np.concatenate([pad, ctx], axis=0)
        elif ctx.shape[0] > ctx_samps:
            ctx = ctx[-ctx_samps:]

        exact = au.Waveform(ctx, sr)
        tokens_full = self.mrt.codec.encode(exact).astype(np.int32)
        depth = int(self.mrt.config.decoder_codec_rvq_depth)
        tokens = tokens_full[:, :depth]

        # Force expected (F,D) at *return time*
        tokens = self._coerce_tokens(tokens)
        return tokens

    def _encode_exact_context_tokens(self, loop: au.Waveform) -> np.ndarray:
        """Build *exactly* context_length_frames worth of tokens (e.g., 250 @ 25fps),
        while ensuring the *end* of the audio lands on a bar boundary.
        Strategy: take the largest integer number of bars <= ctx_seconds as the tail,
        then left-fill from just before that tail (wrapping if needed) to reach exactly
        ctx_seconds; finally, pad/trim to exact samples and, as a last resort, pad/trim
        tokens to the expected frame count.
        """
        wav = loop.as_stereo().resample(self._model_sr)
        data = wav.samples.astype(np.float32, copy=False)
        if data.ndim == 1:
            data = data[:, None]

        spb = self._bar_clock.seconds_per_bar()
        ctx_sec = float(self._ctx_seconds)
        sr = int(self._model_sr)

        # bars that fit fully inside ctx_sec (at least 1)
        bars_fit = max(1, int(ctx_sec // spb))
        tail_len_samps = int(round(bars_fit * spb * sr))

        # ensure we have enough source by tiling
        need = int(round(ctx_sec * sr)) + tail_len_samps
        if data.shape[0] == 0:
            data = np.zeros((1, 2), dtype=np.float32)
        reps = int(np.ceil(need / float(data.shape[0])))
        tiled = np.tile(data, (reps, 1))

        end = tiled.shape[0]
        tail = tiled[end - tail_len_samps:end]

        # left-fill to reach exact ctx samples (keeps end-of-bar alignment)
        ctx_samps = int(round(ctx_sec * sr))
        pad_len = ctx_samps - tail.shape[0]
        if pad_len > 0:
            pre = tiled[end - tail_len_samps - pad_len:end - tail_len_samps]
            ctx = np.concatenate([pre, tail], axis=0)
        else:
            ctx = tail[-ctx_samps:]

        # final snap to *exact* ctx samples
        if ctx.shape[0] < ctx_samps:
            pad = np.zeros((ctx_samps - ctx.shape[0], ctx.shape[1]), dtype=np.float32)
            ctx = np.concatenate([pad, ctx], axis=0)
        elif ctx.shape[0] > ctx_samps:
            ctx = ctx[-ctx_samps:]

        exact = au.Waveform(ctx, sr)
        tokens_full = self.mrt.codec.encode(exact).astype(np.int32)
        depth = int(self.mrt.config.decoder_codec_rvq_depth)
        tokens = tokens_full[:, :depth]

        # Last defense: force expected frame count
        frames = tokens.shape[0]
        exp = int(self._ctx_frames)
        if frames < exp:
            # repeat last frame
            pad = np.repeat(tokens[-1:, :], exp - frames, axis=0)
            tokens = np.concatenate([pad, tokens], axis=0)
        elif frames > exp:
            tokens = tokens[-exp:, :]
        return tokens


    def _install_context_from_loop(self, loop: au.Waveform):
        # Build exact-length, bar-locked context tokens
        context_tokens = self._encode_exact_context_tokens(loop)
        s = self.mrt.init_state()
        s.context_tokens = context_tokens
        self.state = s
        self._original_context_tokens = np.copy(context_tokens)

    def reseed_from_waveform(self, wav: au.Waveform):
        """Immediate reseed: replace context from provided wave (bar-locked, exact length)."""
        context_tokens = self._encode_exact_context_tokens(wav)
        with self._lock:
            s = self.mrt.init_state()
            s.context_tokens = context_tokens
            self.state = s
            self._model_stream = None  # drop model-domain continuity so next chunk starts cleanly
            self._original_context_tokens = np.copy(context_tokens)

    def reseed_splice(self, recent_wav: au.Waveform, anchor_bars: float):
        """Queue a *seamless* reseed by token splicing instead of full restart.
        We compute a fresh, bar-locked context token tensor of exact length
        (e.g., 250 frames), then splice only the *tail* corresponding to
        `anchor_bars` so generation continues smoothly without resetting state.
        """
        new_ctx = self._encode_exact_context_tokens(recent_wav)  # coerce to (F,D)
        F, D = self._expected_token_shape()

        # how many frames correspond to the requested anchor bars
        spb = self._bar_clock.seconds_per_bar()
        frames_per_bar = max(1, int(round(self._codec_fps * spb)))
        splice_frames = max(1, min(int(round(max(1.0, float(anchor_bars)) * frames_per_bar)), F))

        with self._lock:
            # snapshot current context
            cur = getattr(self.state, "context_tokens", None)
            if cur is None:
                # fall back to full reseed (still coerced)
                self._pending_reseed = {"ctx": new_ctx}
                return
            cur = self._coerce_tokens(cur)

            # build the spliced tensor: keep left (F - splice) from cur, take right (splice) from new
            left = cur[:F - splice_frames, :]
            right = new_ctx[F - splice_frames:, :]
            spliced = np.concatenate([left, right], axis=0)
            spliced = self._coerce_tokens(spliced)

            # queue for install at the *next bar boundary* right after emission
            self._pending_token_splice = {
                "tokens": spliced,
                "debug": {"F": F, "D": D, "splice_frames": splice_frames, "frames_per_bar": frames_per_bar}
            }
            


    def reseed_from_waveform(self, wav: au.Waveform):
        """Immediate reseed: replace context from provided wave (bar-aligned tail)."""
        wav = wav.as_stereo().resample(self._model_sr)
        tail = take_bar_aligned_tail(wav, self.params.bpm, self.params.beats_per_bar, self._ctx_seconds)
        tokens_full = self.mrt.codec.encode(tail).astype(np.int32)
        depth = int(self.mrt.config.decoder_codec_rvq_depth)
        context_tokens = tokens_full[:, :depth]

        s = self.mrt.init_state()
        s.context_tokens = context_tokens
        self.state = s
        # reset model stream so next generate starts cleanly
        self._model_stream = None

        # optional loudness match will be applied per-chunk on emission

        # also remember this as new "original"
        self._original_context_tokens = np.copy(context_tokens)

    # ---------- core streaming helpers ----------

    def _append_model_chunk_and_spool(self, wav: au.Waveform) -> None:
        """
        Conservative boundary fix:
        - Emit body+tail immediately (target SR), unchanged from your original behavior.
        - On *next* call, compute the mixed overlap (prev tail ⨉ cos + new head ⨉ sin),
            resample it, and overwrite the last `_pending_tail_target_len` samples in the
            target-SR spool with that mixed overlap. Then emit THIS chunk's body+tail and
            remember THIS chunk's tail length at target SR for the next correction.

        This keeps external timing and bar alignment identical, but removes the audible
        fade-to-zero at chunk ends.
        """
        

        # ---- unpack model-rate samples ----
        s = wav.samples.astype(np.float32, copy=False)
        if s.ndim == 1:
            s = s[:, None]
        n_samps, _ = s.shape
        if n_samps == 0:
            return

        # crossfade length in model samples
        try:
            xfade_s = float(self.mrt.config.crossfade_length)
        except Exception:
            xfade_s = 0.0
        xfade_n = int(round(max(0.0, xfade_s) * float(self._model_sr)))

        # helper: resample to target SR via your streaming resampler
        def to_target(y: np.ndarray) -> np.ndarray:
            return y if self._rs is None else self._rs.process(y, final=False)

        # ------------------------------------------
        # (A) If we have a pending model tail, fix the last emitted tail at target SR
        # ------------------------------------------
        if self._pending_tail_model is not None and self._pending_tail_model.shape[0] == xfade_n and xfade_n > 0 and n_samps >= xfade_n:
            head = s[:xfade_n, :]
            t = np.linspace(0.0, np.pi/2.0, xfade_n, endpoint=False, dtype=np.float32)[:, None]
            cosw = np.cos(t, dtype=np.float32)
            sinw = np.sin(t, dtype=np.float32)
            mixed_model = (self._pending_tail_model * cosw) + (head * sinw)  # [xfade_n, C] at model SR

            y_mixed = to_target(mixed_model.astype(np.float32))
            Lcorr = int(y_mixed.shape[0])  # exact target-SR samples to write

            # Overwrite the last `_pending_tail_target_len` samples of the spool with `y_mixed`.
            # Use the *smaller* of the two lengths to be safe.
            Lpop = min(self._pending_tail_target_len, self._spool.shape[0], Lcorr)
            if Lpop > 0 and self._spool.size:
                # Trim last Lpop samples
                self._spool = self._spool[:-Lpop, :]
                self._spool_written -= Lpop
                # Append corrected overlap (trim/pad to Lpop to avoid drift)
                if Lcorr != Lpop:
                    if Lcorr > Lpop:
                        y_m = y_mixed[-Lpop:, :]
                    else:
                        pad = np.zeros((Lpop - Lcorr, y_mixed.shape[1]), dtype=np.float32)
                        y_m = np.concatenate([y_mixed, pad], axis=0)
                else:
                    y_m = y_mixed
                self._spool = np.concatenate([self._spool, y_m], axis=0) if self._spool.size else y_m
                self._spool_written += y_m.shape[0]

            # For internal continuity, update _model_stream like before
            if self._model_stream is None or self._model_stream.shape[0] < xfade_n:
                self._model_stream = s[xfade_n:].copy()
            else:
                self._model_stream = np.concatenate([self._model_stream[:-xfade_n], mixed_model, s[xfade_n:]], axis=0)
        else:
            # First-ever call or too-short to mix: maintain _model_stream minimally
            if xfade_n > 0 and n_samps > xfade_n:
                self._model_stream = s[xfade_n:].copy() if self._model_stream is None else np.concatenate([self._model_stream, s[xfade_n:]], axis=0)
            else:
                self._model_stream = s.copy() if self._model_stream is None else np.concatenate([self._model_stream, s], axis=0)

        # ------------------------------------------
        # (B) Emit THIS chunk's body and tail (same external behavior)
        # ------------------------------------------
        if xfade_n > 0 and n_samps >= (2 * xfade_n):
            body = s[xfade_n:-xfade_n, :]
            if body.size:
                y_body = to_target(body.astype(np.float32))
                if y_body.size:
                    self._spool = np.concatenate([self._spool, y_body], axis=0) if self._spool.size else y_body
                    self._spool_written += y_body.shape[0]
        else:
            # If chunk too short for head+tail split, treat all (minus preroll) as body
            if xfade_n > 0 and n_samps > xfade_n:
                body = s[xfade_n:, :]
                y_body = to_target(body.astype(np.float32))
                if y_body.size:
                    self._spool = np.concatenate([self._spool, y_body], axis=0) if self._spool.size else y_body
                    self._spool_written += y_body.shape[0]
                # No tail to remember this round
                self._pending_tail_model = None
                self._pending_tail_target_len = 0
                return

        # Tail (always remember how many TARGET samples we append)
        if xfade_n > 0 and n_samps >= xfade_n:
            tail = s[-xfade_n:, :]
            y_tail = to_target(tail.astype(np.float32))
            Ltail = int(y_tail.shape[0])
            if Ltail:
                self._spool = np.concatenate([self._spool, y_tail], axis=0) if self._spool.size else y_tail
                self._spool_written += Ltail
                self._pending_tail_model = tail.copy()
                self._pending_tail_target_len = Ltail
            else:
                # Nothing appended (resampler returning nothing yet) — keep model tail but mark zero target len
                self._pending_tail_model = tail.copy()
                self._pending_tail_target_len = 0
        else:
            self._pending_tail_model = None
            self._pending_tail_target_len = 0


    def _should_generate_next_chunk(self) -> bool:
        # Allow running ahead relative to whichever is larger: last *consumed*
        # (explicit ack from client) or last *delivered* (implicit ack).
        implicit_consumed = self._next_to_deliver - 1  # last chunk handed to client
        horizon_anchor = max(self._last_consumed_index, implicit_consumed)
        return self.idx <= (horizon_anchor + self._max_buffer_ahead)

    def _emit_ready(self):
        """Emit next chunk(s) if the spool has enough samples. With robust RMS debug."""
        

        QDB_SILENCE = -55.0
        EPS = 1e-12

        def rms_dbfs(x: np.ndarray) -> float:
            if x.ndim == 2:
                x = x.mean(axis=1)
            rms = float(np.sqrt(np.mean(np.square(x)) + EPS))
            return 20.0 * np.log10(max(rms, EPS))

        def qbar_rms_dbfs(x: np.ndarray, seg_len: int) -> list[float]:
            if x.ndim == 2:
                mono = x.mean(axis=1)
            else:
                mono = x
            N = mono.shape[0]
            vals = []
            for i in range(0, N, seg_len):
                seg = mono[i:min(i + seg_len, N)]
                if seg.size == 0:
                    break
                r = float(np.sqrt(np.mean(seg * seg) + EPS))
                vals.append(20.0 * np.log10(max(r, EPS)))
            return vals

        def fmt_db_list(vals):
            return ['%5.1f' % v for v in vals[:8]]

        def extract_gain_db(g):
            # Accept float/int, dict{'gain_db': ...}, tuple/list, or None
            if g is None:
                return None
            if isinstance(g, (int, float)):
                return float(g)
            if isinstance(g, dict):
                for k in ('gain_db', 'gain', 'applied_gain_db'):
                    if k in g:
                        try:
                            return float(g[k])
                        except Exception:
                            pass
                return None
            if isinstance(g, (list, tuple)) and g:
                try:
                    return float(g[0])
                except Exception:
                    return None
            return None

        while True:
            start, end = self._bar_clock.bounds_for_chunk(self.idx, self.params.bars_per_chunk)
            if end > self._spool_written:
                break

            loop = self._spool[start:end]

            # ---- pre-LM diagnostics ----
            spb = self._bar_clock.bar_samps
            qlen = max(1, spb // 4)
            q_rms_pre = qbar_rms_dbfs(loop, qlen)
            silent_marks_pre = ["🟢" if v > QDB_SILENCE else "🟥" for v in q_rms_pre[:8]]
            print(f"[emit idx={self.idx}] pre-LM qRMS dBFS: {fmt_db_list(q_rms_pre)} {''.join(silent_marks_pre)}")

            # Loudness match (optional)
            gain_db_applied_raw = None
            if self.params.ref_loop is not None and self.params.loudness_mode != "none":
                ref = self.params.ref_loop.as_stereo().resample(self.params.target_sr)
                wav = au.Waveform(loop.copy(), int(self.params.target_sr))
                try:
                    matched, gain_db_applied_raw = match_loudness_to_reference(
                        ref, wav,
                        method=self.params.loudness_mode,
                        headroom_db=self.params.headroom_db
                    )
                    loop = matched.samples
                except Exception as e:
                    print(f"[emit idx={self.idx}] loudness-match ERROR: {e}; proceeding with un-matched audio")

            gain_db = extract_gain_db(gain_db_applied_raw)

            # ---- post-LM diagnostics ----
            q_rms_post = qbar_rms_dbfs(loop, qlen)
            silent_marks_post = ["🟢" if v > QDB_SILENCE else "🟥" for v in q_rms_post[:8]]
            if gain_db is None:
                print(f"[emit idx={self.idx}] post-LM qRMS dBFS: {fmt_db_list(q_rms_post)} {''.join(silent_marks_post)} (LM: none)")
            else:
                print(f"[emit idx={self.idx}] post-LM qRMS dBFS: {fmt_db_list(q_rms_post)} {''.join(silent_marks_post)} (LM gain {gain_db:+.2f} dB)")

            # Encode & ship
            audio_b64, total_samples, channels = wav_bytes_base64(loop, int(self.params.target_sr))
            meta = {
                "bpm": float(self.params.bpm),
                "bars": int(self.params.bars_per_chunk),
                "beats_per_bar": int(self.params.beats_per_bar),
                "sample_rate": int(self.params.target_sr),
                "channels": int(channels),
                "total_samples": int(total_samples),
                "seconds_per_bar": self._bar_clock.seconds_per_bar(),
                "loop_duration_seconds": self.params.bars_per_chunk * self._bar_clock.seconds_per_bar(),
                "guidance_weight": float(self.params.guidance_weight),
                "temperature": float(self.params.temperature),
                "topk": int(self.params.topk),
            }
            chunk = JamChunk(index=self.idx, audio_base64=audio_b64, metadata=meta)

            with self._cv:
                self._outbox[self.idx] = chunk
                self._cv.notify_all()

            print(f"[emit idx={self.idx}] slice [{start}:{end}] (len={end-start}), spool_written={self._spool_written}")
            self.idx += 1

            # Apply pending splices/reseeds immediately after a completed emit
            with self._lock:
                if self._pending_token_splice is not None:
                    spliced = self._coerce_tokens(self._pending_token_splice["tokens"])
                    try:
                        self.state.context_tokens = spliced
                        self._pending_token_splice = None
                        print(f"[emit idx={self.idx}] installed token splice (in-place)")
                    except Exception:
                        new_state = self.mrt.init_state()
                        new_state.context_tokens = spliced
                        self.state = new_state
                        self._model_stream = None
                        self._pending_token_splice = None
                        print(f"[emit idx={self.idx}] installed token splice (reinit state)")
                elif self._pending_reseed is not None:
                    ctx = self._coerce_tokens(self._pending_reseed["ctx"])
                    new_state = self.mrt.init_state()
                    new_state.context_tokens = ctx
                    self.state = new_state
                    self._model_stream = None
                    self._pending_reseed = None
                    print(f"[emit idx={self.idx}] performed full reseed")


    # ---------- main loop ----------

    def run(self):
        # generate until stopped
        while not self._stop_event.is_set():
            # throttle generation if we are far ahead
            if not self._should_generate_next_chunk():
                # still try to emit if spool already has enough
                self._emit_ready()
                time.sleep(0.01)
                continue

            # generate next model chunk
            # snapshot current style vector under lock for this step
            with self._lock:
                target = self.params.style_vec
                if target is None:
                    style_to_use = None
                else:
                    if self._style_vec is None:  # first use: start exactly at initial style (no glide)
                        self._style_vec = np.array(target, dtype=np.float32, copy=True)
                    else:
                        ramp = float(self.params.style_ramp_seconds or 0.0)
                        step = 1.0 if ramp <= 0.0 else min(1.0, self._chunk_secs / ramp)
                        # linear ramp in embedding space
                        self._style_vec += step * (target.astype(np.float32, copy=False) - self._style_vec)
                    style_to_use = self._style_vec

            wav, self.state = self.mrt.generate_chunk(state=self.state, style=style_to_use)
            # append and spool
            self._append_model_chunk_and_spool(wav)
            # try emitting zero or more chunks if available
            self._emit_ready()

        # finalize resampler (flush) — not strictly necessary here
        tail = self._rs.process(np.zeros((0,2), np.float32), final=True)
        if tail.size:
            self._spool = np.concatenate([self._spool, tail], axis=0)
            self._spool_written += tail.shape[0]
        # one last emit attempt
        self._emit_ready()
