# jam_worker.py - Updated with robust silence handling
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

def _dbg_rms_dbfs(x: np.ndarray) -> float:
    if x.ndim == 2:
        x = x.mean(axis=1)
    r = float(np.sqrt(np.mean(x * x) + 1e-12))
    return 20.0 * np.log10(max(r, 1e-12))

def _dbg_rms_dbfs_model(x: np.ndarray) -> float:
    # x is model-rate, shape [S,C] or [S]
    if x.ndim == 2:
        x = x.mean(axis=1)
    r = float(np.sqrt(np.mean(x * x) + 1e-12))
    return 20.0 * np.log10(max(r, 1e-12))

def _dbg_shape(x):
    return tuple(x.shape) if hasattr(x, "shape") else ("-",)

def _is_silent(audio: np.ndarray, threshold_db: float = -60.0) -> bool:
    """Check if audio is effectively silent."""
    if audio.size == 0:
        return True
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    rms = float(np.sqrt(np.mean(audio**2)))
    return 20.0 * np.log10(max(rms, 1e-12)) < threshold_db

def _has_energy(audio: np.ndarray, threshold_db: float = -40.0) -> bool:
    """Check if audio has significant energy (stricter than just non-silent)."""
    return not _is_silent(audio, threshold_db)

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
    style_ramp_seconds: float = 8.0


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

        # Health monitoring
        self._silence_streak = 0  # consecutive silent chunks
        self._last_good_context_tokens = None  # backup of last known good context

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
            # Save this as our "good" context backup
            if hasattr(self.state, 'context_tokens') and self.state.context_tokens is not None:
                self._last_good_context_tokens = np.copy(self.state.context_tokens)

    # ---------- NEW: Health monitoring methods ----------

    def _check_model_health(self, new_chunk: np.ndarray) -> bool:
        """Check if the model output looks healthy."""
        if _is_silent(new_chunk, threshold_db=-80.0):
            self._silence_streak += 1
            print(f"⚠️  Silent chunk detected (streak: {self._silence_streak})")
            return False
        else:
            if self._silence_streak > 0:
                print(f"✅ Audio resumed after {self._silence_streak} silent chunks")
            self._silence_streak = 0
            return True

    def _recover_from_silence(self):
        """Attempt to recover from silence by restoring last good context."""
        print("🔧 Attempting recovery from silence...")
        
        if self._last_good_context_tokens is not None:
            # Restore last known good context
            try:
                new_state = self.mrt.init_state()
                new_state.context_tokens = np.copy(self._last_good_context_tokens)
                self.state = new_state
                self._model_stream = None  # Reset stream to start fresh
                print("   Restored last good context")
            except Exception as e:
                print(f"   Context restoration failed: {e}")
        
        # If we have the original loop, rebuild context from it
        elif self.params.combined_loop is not None:
            try:
                self._install_context_from_loop(self.params.combined_loop)
                self._model_stream = None
                print("   Rebuilt context from original loop")
            except Exception as e:
                print(f"   Context rebuild failed: {e}")

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
        """Build *exactly* context_length_frames worth of tokens, ensuring bar alignment."""
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
            # Instead of zero padding, repeat the audio to fill
            shortfall = ctx_samps - ctx.shape[0]
            if ctx.shape[0] > 0:
                fill = np.tile(ctx, (int(np.ceil(shortfall / ctx.shape[0])) + 1, 1))[:shortfall]
                ctx = np.concatenate([fill, ctx], axis=0)
            else:
                print("⚠️  Zero-length context, using fallback")
                ctx = np.zeros((ctx_samps, 2), dtype=np.float32)
        elif ctx.shape[0] > ctx_samps:
            ctx = ctx[-ctx_samps:]

        exact = au.Waveform(ctx, sr)
        tokens_full = self.mrt.codec.encode(exact).astype(np.int32)
        depth = int(self.mrt.config.decoder_codec_rvq_depth)
        tokens = tokens_full[:, :depth]

        # Force expected (F,D) at *return time*
        tokens = self._coerce_tokens(tokens)
        
        # Validate that we don't have a silent context
        if _is_silent(ctx, threshold_db=-80.0):
            print("⚠️  Generated silent context - this may cause issues")
        
        return tokens

    def _install_context_from_loop(self, loop: au.Waveform):
        # Build exact-length, bar-locked context tokens
        context_tokens = self._encode_exact_context_tokens(loop)
        s = self.mrt.init_state()
        s.context_tokens = context_tokens
        self.state = s
        self._last_good_context_tokens = np.copy(context_tokens)

    def reseed_from_waveform(self, wav: au.Waveform):
        """Immediate reseed: replace context from provided wave (bar-locked, exact length)."""
        context_tokens = self._encode_exact_context_tokens(wav)
        with self._lock:
            s = self.mrt.init_state()
            s.context_tokens = context_tokens
            self.state = s
            self._model_stream = None  # drop model-domain continuity so next chunk starts cleanly
            self._last_good_context_tokens = np.copy(context_tokens)
            self._silence_streak = 0  # Reset health monitoring

    def reseed_splice(self, recent_wav: au.Waveform, anchor_bars: float):
        """Queue a *seamless* reseed by token splicing instead of full restart."""
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

    # ---------- REWRITTEN: core streaming helpers ----------

    def _append_model_chunk_and_spool(self, wav: au.Waveform) -> None:
        """
        REWRITTEN: Robust audio processing that rejects silent chunks entirely.
        
        Strategy:
        1. Validate input chunk for silence/issues
        2. REJECT silent chunks - don't add them to spool or model stream
        3. Use healthy crossfading only between good audio
        4. Aggressive recovery when silence detected
        """
        # Unpack model-rate samples
        s = wav.samples.astype(np.float32, copy=False)
        if s.ndim == 1:
            s = s[:, None]
        n_samps, _ = s.shape
        if n_samps == 0:
            return

        # Health check on new chunk - use stricter threshold
        is_healthy = self._check_model_health(s)
        is_very_quiet = _is_silent(s, threshold_db=-50.0)  # stricter than default -60
        
        # Get crossfade params
        try:
            xfade_s = float(self.mrt.config.crossfade_length)
        except Exception:
            xfade_s = 0.0
        xfade_n = int(round(max(0.0, xfade_s) * float(self._model_sr)))

        print(f"[model] chunk len={n_samps} rms={_dbg_rms_dbfs_model(s):+.1f} dBFS healthy={is_healthy} quiet={is_very_quiet}")

        # --- REJECT PROBLEMATIC CHUNKS ---
        if not is_healthy or is_very_quiet:
            print(f"[REJECT] Discarding unhealthy/quiet chunk - not adding to spool or model stream")
            
            # Trigger recovery immediately on first bad chunk
            if self._silence_streak >= 1:
                self._recover_from_silence()
                
            # Don't process this chunk at all - return early
            return

        # Reset silence streak on good chunk
        if self._silence_streak > 0:
            print(f"✅ Audio resumed after {self._silence_streak} rejected chunks")
        self._silence_streak = 0

        # Helper: resample to target SR
        def to_target(y: np.ndarray) -> np.ndarray:
            return y if self._rs is None else self._rs.process(y, final=False)

        # --- SIMPLIFIED CROSSFADE LOGIC (only for healthy audio) ---
        
        if self._model_stream is None:
            # First chunk - no crossfading needed
            self._model_stream = s.copy()
            
        elif xfade_n <= 0 or n_samps < xfade_n:
            # No crossfade configured or chunk too short - simple append
            self._model_stream = np.concatenate([self._model_stream, s], axis=0)
            
        elif _is_silent(self._model_stream[-xfade_n:], threshold_db=-50.0):
            # Previous tail is quiet - don't crossfade, just replace
            print(f"[crossfade] Replacing quiet tail with new audio")
            # Remove quiet tail and append new chunk
            self._model_stream = np.concatenate([self._model_stream[:-xfade_n], s], axis=0)
            
        else:
            # Normal crossfade between healthy audio
            tail = self._model_stream[-xfade_n:]
            head = s[:xfade_n]
            body = s[xfade_n:] if n_samps > xfade_n else np.zeros((0, s.shape[1]), dtype=np.float32)
            
            # Equal power crossfade
            t = np.linspace(0.0, 1.0, xfade_n, dtype=np.float32)[:, None]
            fade_out = np.cos(t * np.pi / 2.0)
            fade_in = np.sin(t * np.pi / 2.0)
            
            mixed = tail * fade_out + head * fade_in
            
            print(f"[crossfade] tail rms={_dbg_rms_dbfs_model(tail):+.1f} head rms={_dbg_rms_dbfs_model(head):+.1f} mixed rms={_dbg_rms_dbfs_model(mixed):+.1f}")
            
            # Update model stream: remove old tail, add mixed section, add body
            self._model_stream = np.concatenate([
                self._model_stream[:-xfade_n],
                mixed,
                body
            ], axis=0)

        # --- CONVERT AND APPEND TO SPOOL (only healthy audio reaches here) ---
        
        # Take the new audio from this iteration
        if xfade_n > 0 and n_samps >= xfade_n:
            # Normal case: body after crossfade region
            new_audio = s[xfade_n:] if n_samps > xfade_n else s
        else:
            # Short chunk or no crossfade: use entire chunk
            new_audio = s
            
        if new_audio.shape[0] > 0:
            target_audio = to_target(new_audio)
            if target_audio.shape[0] > 0:
                print(f"[append] body len={target_audio.shape[0]} rms={_dbg_rms_dbfs(target_audio):+.1f} dBFS")
                self._spool = np.concatenate([self._spool, target_audio], axis=0) if self._spool.size else target_audio
                self._spool_written += target_audio.shape[0]

        # --- SAVE GOOD CONTEXT ---
        # Only save context from healthy chunks
        if hasattr(self.state, 'context_tokens') and self.state.context_tokens is not None:
            self._last_good_context_tokens = np.copy(self.state.context_tokens)

        # Trim model stream to reasonable length (keep ~30 seconds)
        max_model_samples = int(30.0 * self._model_sr)
        if self._model_stream.shape[0] > max_model_samples:
            self._model_stream = self._model_stream[-max_model_samples:]

    def _should_generate_next_chunk(self) -> bool:
        # Allow running ahead relative to whichever is larger: last *consumed*
        # (explicit ack from client) or last *delivered* (implicit ack).
        implicit_consumed = self._next_to_deliver - 1  # last chunk handed to client
        horizon_anchor = max(self._last_consumed_index, implicit_consumed)
        return self.idx <= (horizon_anchor + self._max_buffer_ahead)

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
                "silence_streak": self._silence_streak,  # Add health info
            }
            chunk = JamChunk(index=self.idx, audio_base64=audio_b64, metadata=meta)

            if os.getenv("MRT_DEBUG_RMS", "0") == "1":
                spb = self._bar_clock.bar_samps
                seg = int(max(1, spb // 4))  # quarter-bar window
                
                rms = [float(np.sqrt(np.mean(loop[i:i+seg]**2))) for i in range(0, loop.shape[0], seg)]
                print(f"[emit idx={self.idx}] quarter-bar RMS: {rms[:8]}")

            with self._cv:
                self._outbox[self.idx] = chunk
                self._cv.notify_all()
            self.idx += 1

            # If a reseed is queued, install it *right after* we finish a chunk
            with self._lock:
                # Prefer seamless token splice when available
                if self._pending_token_splice is not None:
                    spliced = self._coerce_tokens(self._pending_token_splice["tokens"])
                    try:
                        # inplace update (no reset)
                        self.state.context_tokens = spliced
                        self._pending_token_splice = None
                        print("[reseed] Token splice applied")
                    except Exception:
                        # fallback: full reseed using spliced tokens
                        new_state = self.mrt.init_state()
                        new_state.context_tokens = spliced
                        self.state = new_state
                        self._model_stream = None
                        self._pending_token_splice = None
                        print("[reseed] Token splice fallback to full reset")
                elif self._pending_reseed is not None:
                    ctx = self._coerce_tokens(self._pending_reseed["ctx"])
                    new_state = self.mrt.init_state()
                    new_state.context_tokens = ctx
                    self.state = new_state
                    self._model_stream = None
                    self._pending_reseed = None
                    print("[reseed] Full reseed applied")

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
        if self._rs is not None:
            tail = self._rs.process(np.zeros((0,2), np.float32), final=True)
            if tail.size:
                self._spool = np.concatenate([self._spool, tail], axis=0)
                self._spool_written += tail.shape[0]
        # one last emit attempt
        self._emit_ready()