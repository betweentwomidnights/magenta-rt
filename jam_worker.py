# jam_worker.py - Bar-locked spool rewrite
from __future__ import annotations

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

        # style vector (already normalized upstream)
        self._style_vec = self.params.style_vec

        # codec/setup
        self._codec_fps = float(self.mrt.codec.frame_rate)
        self._ctx_frames = int(self.mrt.config.context_length_frames)
        self._ctx_seconds = self._ctx_frames / self._codec_fps

        # model stream (model SR) for internal continuity/crossfades
        self._model_stream: Optional[np.ndarray] = None
        self._model_sr = int(self.mrt.sample_rate)

        # target-SR in-RAM spool (what we cut loops from)
        self._rs = StreamingResampler(self._model_sr, int(self.params.target_sr), channels=2)
        self._spool = np.zeros((0, 2), dtype=np.float32)   # (S,2) target SR
        self._spool_written = 0                            # absolute frames written into spool

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
        self._max_buffer_ahead = 5

        # reseed queue (install at next safe point)
        self._pending_reseed: Optional[dict] = None

        # Prepare initial context from combined loop (best musical alignment)
        if self.params.combined_loop is not None:
            self._install_context_from_loop(self.params.combined_loop)

    # ---------- lifecycle ----------

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
        """Queue a splice reseed to be applied right after the next emitted loop."""
        new_ctx = self._encode_exact_context_tokens(recent_wav)
        with self._lock:
            self._pending_reseed = {"ctx": new_ctx}


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

    def reseed_splice(self, recent_wav: au.Waveform, anchor_bars: float):
        """Queue a splice reseed to be applied right after the next emitted loop.
        For now, we simply replace the context by recent wave tail; anchor is accepted
        for API compatibility and future crossfade/token-splice logic."""
        recent_wav = recent_wav.as_stereo().resample(self._model_sr)
        tail = take_bar_aligned_tail(recent_wav, self.params.bpm, self.params.beats_per_bar, self._ctx_seconds)
        tokens_full = self.mrt.codec.encode(tail).astype(np.int32)
        depth = int(self.mrt.config.decoder_codec_rvq_depth)
        new_ctx = tokens_full[:, :depth]
        self._pending_reseed = {"ctx": new_ctx}

    # ---------- core streaming helpers ----------

    def _append_model_chunk_and_spool(self, wav: au.Waveform):
        """Crossfade into the model-rate stream and write the *non-overlapped*
        tail to the target-SR spool."""
        s = wav.samples.astype(np.float32, copy=False)
        if s.ndim == 1:
            s = s[:, None]
        sr = self._model_sr
        xfade_s = float(self.mrt.config.crossfade_length)
        xfade_n = int(round(max(0.0, xfade_s) * sr))

        if self._model_stream is None:
            # first chunk: drop the preroll (xfade) then spool
            new_part = s[xfade_n:] if xfade_n < s.shape[0] else s[:0]
            self._model_stream = new_part.copy()
            if new_part.size:
                y = self._rs.process(new_part, final=False)
                self._spool = np.concatenate([self._spool, y], axis=0)
                self._spool_written += y.shape[0]
            return

        # crossfade into existing stream
        if xfade_n > 0 and self._model_stream.shape[0] >= xfade_n and s.shape[0] >= xfade_n:
            tail = self._model_stream[-xfade_n:]
            head = s[:xfade_n]
            t = np.linspace(0, np.pi/2, xfade_n, endpoint=False, dtype=np.float32)[:, None]
            mixed = tail * np.cos(t) + head * np.sin(t)
            self._model_stream = np.concatenate([self._model_stream[:-xfade_n], mixed, s[xfade_n:]], axis=0)
            new_part = s[xfade_n:]
        else:
            self._model_stream = np.concatenate([self._model_stream, s], axis=0)
            new_part = s

        # spool only the *new* non-overlapped part
        if new_part.size:
            y = self._rs.process(new_part.astype(np.float32, copy=False), final=False)
            if y.size:
                self._spool = np.concatenate([self._spool, y], axis=0)
                self._spool_written += y.shape[0]

    def _should_generate_next_chunk(self) -> bool:
        # Don't let generation run too far ahead of consumption
        return self.idx <= (self._last_consumed_index + self._max_buffer_ahead)

    def _emit_ready(self):
        """Emit next chunk(s) if the spool has enough samples."""
        while True:
            start, end = self._bar_clock.bounds_for_chunk(self.idx, self.params.bars_per_chunk)
            if end > self._spool_written:
                break  # need more audio
            loop = self._spool[start:end]

            # Loudness match to reference loop (optional)
            if self.params.ref_loop is not None and self.params.loudness_mode != "none":
                ref = self.params.ref_loop.as_stereo().resample(self.params.target_sr)
                wav = au.Waveform(loop.copy(), int(self.params.target_sr))
                matched, _ = match_loudness_to_reference(ref, wav, method=self.params.loudness_mode, headroom_db=self.params.headroom_db)
                loop = matched.samples

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
            self.idx += 1

            # If a reseed is queued, install it *right after* we finish a chunk
            with self._lock:
                if self._pending_reseed is not None:
                    new_state = self.mrt.init_state()
                    new_state.context_tokens = self._pending_reseed["ctx"]
                    self.state = new_state
                    self._model_stream = None
                    self._pending_reseed = None

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
                style_vec = self._style_vec
            wav, self.state = self.mrt.generate_chunk(state=self.state, style=style_vec)
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
