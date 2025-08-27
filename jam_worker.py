# jam_worker.py - SIMPLE FIX VERSION
import threading, time, base64, io, uuid
from dataclasses import dataclass, field
import numpy as np
import soundfile as sf
from magenta_rt import audio as au
from threading import RLock
from utils import (
    match_loudness_to_reference, stitch_generated, hard_trim_seconds,
    apply_micro_fades, make_bar_aligned_context, take_bar_aligned_tail,
    resample_and_snap, wav_bytes_base64
)
from math import floor, ceil

@dataclass
class JamParams:
    bpm: float
    beats_per_bar: int
    bars_per_chunk: int
    target_sr: int
    loudness_mode: str = "auto"
    headroom_db: float = 1.0
    style_vec: np.ndarray | None = None
    ref_loop: any = None
    combined_loop: any = None
    guidance_weight: float = 1.1
    temperature: float = 1.1
    topk: int = 40

@dataclass
class JamChunk:
    index: int
    audio_base64: str
    metadata: dict

class JamWorker(threading.Thread):
    def __init__(self, mrt, params: JamParams):
        super().__init__(daemon=True)
        self.mrt = mrt
        self.params = params
        self.state = mrt.init_state()

        # ✅ init synchronization + placeholders FIRST
        self._lock = threading.Lock()
        self._original_context_tokens = None   # so hasattr checks are cheap/clear

        if params.combined_loop is not None:
            self._setup_context_from_combined_loop()

        self.idx = 0
        self.outbox: list[JamChunk] = []
        self._stop_event = threading.Event()

        self._stream = None
        self._next_emit_start = 0

        # NEW: Track delivery state
        self._last_delivered_index = 0
        self._max_buffer_ahead = 5

        # Timing info
        self.last_chunk_started_at = None
        self.last_chunk_completed_at = None

        self._pending_reseed = None        # {"ctx": np.ndarray, "ref": au.Waveform|None}
        self._needs_bar_realign = False    # request a one-shot downbeat alignment
        self._reseed_ref_loop = None       # which loop to align against after reseed


    def _setup_context_from_combined_loop(self):
        """Set up MRT context tokens from the combined loop audio"""
        try:
            from utils import make_bar_aligned_context, take_bar_aligned_tail

            codec_fps = float(self.mrt.codec.frame_rate)
            ctx_seconds = float(self.mrt.config.context_length_frames) / codec_fps

            loop_for_context = take_bar_aligned_tail(
                self.params.combined_loop,
                self.params.bpm,
                self.params.beats_per_bar,
                ctx_seconds
            )

            tokens_full = self.mrt.codec.encode(loop_for_context).astype(np.int32)
            tokens = tokens_full[:, :self.mrt.config.decoder_codec_rvq_depth]

            context_tokens = make_bar_aligned_context(
                tokens,
                bpm=self.params.bpm,
                fps=float(self.mrt.codec.frame_rate),  # keep fractional fps
                ctx_frames=self.mrt.config.context_length_frames,
                beats_per_bar=self.params.beats_per_bar
            )

            # Install fresh context
            self.state.context_tokens = context_tokens
            print(f"✅ JamWorker: Set up fresh context from combined loop")

            # NEW: keep a copy of the *original* context tokens for future splice-reseed
            # (guard so we only set this once, at jam start)
            with self._lock:
                if not hasattr(self, "_original_context_tokens") or self._original_context_tokens is None:
                    self._original_context_tokens = np.copy(context_tokens)  # shape: [T, depth]

        except Exception as e:
            print(f"❌ Failed to setup context from combined loop: {e}")

    def stop(self):
        self._stop_event.set()

    def update_knobs(self, *, guidance_weight=None, temperature=None, topk=None):
        with self._lock:
            if guidance_weight is not None: self.params.guidance_weight = float(guidance_weight)
            if temperature is not None:     self.params.temperature     = float(temperature)
            if topk is not None:            self.params.topk            = int(topk)

    def get_next_chunk(self) -> JamChunk | None:
        """Get the next sequential chunk (blocks/waits if not ready)"""
        target_index = self._last_delivered_index + 1
        
        # Wait for the target chunk to be ready (with timeout)
        max_wait = 30.0  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait and not self._stop_event.is_set():
            with self._lock:
                # Look for the exact chunk we need
                for chunk in self.outbox:
                    if chunk.index == target_index:
                        self._last_delivered_index = target_index
                        print(f"📦 Delivered chunk {target_index}")
                        return chunk
            
            # Not ready yet, wait a bit
            time.sleep(0.1)
        
        # Timeout or stopped
        return None

    def mark_chunk_consumed(self, chunk_index: int):
        """Mark a chunk as consumed by the frontend"""
        with self._lock:
            self._last_delivered_index = max(self._last_delivered_index, chunk_index)
            print(f"✅ Chunk {chunk_index} consumed")

    def _should_generate_next_chunk(self) -> bool:
        """Check if we should generate the next chunk (don't get too far ahead)"""
        with self._lock:
            # Don't generate if we're already too far ahead
            if self.idx > self._last_delivered_index + self._max_buffer_ahead:
                return False
            return True

    def _seconds_per_bar(self) -> float:
        return self.params.beats_per_bar * (60.0 / self.params.bpm)

    def _snap_and_encode(self, y, seconds, target_sr, bars):
        cur_sr = int(self.mrt.sample_rate)
        x = y.samples if y.samples.ndim == 2 else y.samples[:, None]
        x = resample_and_snap(x, cur_sr=cur_sr, target_sr=target_sr, seconds=seconds)
        b64, total_samples, channels = wav_bytes_base64(x, target_sr)
        meta = {
            "bpm": int(round(self.params.bpm)),
            "bars": int(bars),
            "beats_per_bar": int(self.params.beats_per_bar),
            "sample_rate": int(target_sr),
            "channels": channels,
            "total_samples": total_samples,
            "seconds_per_bar": self._seconds_per_bar(),
            "loop_duration_seconds": bars * self._seconds_per_bar(),
            "guidance_weight": self.params.guidance_weight,
            "temperature": self.params.temperature,
            "topk": self.params.topk,
        }
        return b64, meta

    def _append_model_chunk_to_stream(self, wav):
        """Incrementally append a model chunk with equal-power crossfade."""
        xfade_s = float(self.mrt.config.crossfade_length)
        sr = int(self.mrt.sample_rate)
        xfade_n = int(round(xfade_s * sr))

        s = wav.samples if wav.samples.ndim == 2 else wav.samples[:, None]

        if getattr(self, "_stream", None) is None:
            # First chunk: drop model pre-roll (xfade head)
            if s.shape[0] > xfade_n:
                self._stream = s[xfade_n:].astype(np.float32, copy=True)
            else:
                self._stream = np.zeros((0, s.shape[1]), dtype=np.float32)
            self._next_emit_start = 0  # pointer into _stream (model SR samples)
            return

        # Crossfade last xfade_n samples of _stream with head of new s
        if s.shape[0] <= xfade_n or self._stream.shape[0] < xfade_n:
            # Degenerate safeguard
            self._stream = np.concatenate([self._stream, s], axis=0)
            return

        tail = self._stream[-xfade_n:]
        head = s[:xfade_n]

        # Equal-power envelopes
        t = np.linspace(0, np.pi/2, xfade_n, endpoint=False, dtype=np.float32)[:, None]
        eq_in, eq_out = np.sin(t), np.cos(t)
        mixed = tail * eq_out + head * eq_in

        self._stream = np.concatenate([self._stream[:-xfade_n], mixed, s[xfade_n:]], axis=0)

    def reseed_from_waveform(self, wav):
        # 1) Re-init state
        new_state = self.mrt.init_state()

        # 2) Build bar-aligned context tokens from provided audio
        codec_fps   = float(self.mrt.codec.frame_rate)
        ctx_seconds = float(self.mrt.config.context_length_frames) / codec_fps
        from utils import take_bar_aligned_tail, make_bar_aligned_context

        tail = take_bar_aligned_tail(wav, self.params.bpm, self.params.beats_per_bar, ctx_seconds)
        tokens_full = self.mrt.codec.encode(tail).astype(np.int32)
        tokens = tokens_full[:, :self.mrt.config.decoder_codec_rvq_depth]
        context_tokens = make_bar_aligned_context(tokens,
            bpm=self.params.bpm, fps=float(self.mrt.codec.frame_rate),
            ctx_frames=self.mrt.config.context_length_frames,
            beats_per_bar=self.params.beats_per_bar
        )
        new_state.context_tokens = context_tokens
        self.state = new_state
        self._prepare_stream_for_reseed_handoff()

    def _frames_per_bar(self) -> int:
        # codec frame-rate (frames/s) -> frames per musical bar
        fps = float(self.mrt.codec.frame_rate)
        sec_per_bar = (60.0 / float(self.params.bpm)) * float(self.params.beats_per_bar)
        return int(round(fps * sec_per_bar))

    def _ctx_frames(self) -> int:
        # how many codec frames fit in the model’s conditioning window
        return int(self.mrt.config.context_length_frames)

    def _make_recent_tokens_from_wave(self, wav) -> np.ndarray:
        """
        Encode waveform and produce a BAR-ALIGNED context token window.
        """
        tokens_full = self.mrt.codec.encode(wav).astype(np.int32)           # [T, rvq_total]
        tokens      = tokens_full[:, :self.mrt.config.decoder_codec_rvq_depth]

        from utils import make_bar_aligned_context
        ctx = make_bar_aligned_context(
            tokens,
            bpm=self.params.bpm,
            fps=float(self.mrt.codec.frame_rate),  # keep fractional fps
            ctx_frames=self.mrt.config.context_length_frames,
            beats_per_bar=self.params.beats_per_bar
        )
        return ctx

    def _bar_aligned_tail(self, tokens: np.ndarray, bars: float) -> np.ndarray:
        """
        Take a tail slice that is an integer number of codec frames corresponding to `bars`.
        We round to nearest frame to stay phase-consistent with codec grid.
        """
        frames_per_bar = self._frames_per_bar()
        want = max(frames_per_bar * int(round(bars)), 0)
        if want == 0:
            return tokens[:0]  # empty
        if tokens.shape[0] <= want:
            return tokens
        return tokens[-want:]

    def _splice_context(self, original_tokens: np.ndarray, recent_tokens: np.ndarray,
                    anchor_bars: float) -> np.ndarray:
        import math
        ctx_frames = self._ctx_frames()
        depth = original_tokens.shape[1]
        frames_per_bar = self._frames_per_bar()

        # 1) Anchor tail (whole bars)
        anchor = self._bar_aligned_tail(original_tokens, math.floor(anchor_bars))

        # 2) Fill remainder with recent (prefer whole bars)
        a = anchor.shape[0]
        remain = max(ctx_frames - a, 0)

        recent = recent_tokens[:0]
        used_recent = 0  # frames taken from the END of recent_tokens
        if remain > 0:
            bars_fit = remain // frames_per_bar
            if bars_fit >= 1:
                want_recent_frames = int(bars_fit * frames_per_bar)
                used_recent = min(want_recent_frames, recent_tokens.shape[0])
                recent = recent_tokens[-used_recent:] if used_recent > 0 else recent_tokens[:0]
            else:
                used_recent = min(remain, recent_tokens.shape[0])
                recent = recent_tokens[-used_recent:] if used_recent > 0 else recent_tokens[:0]

        # 3) Concat in order [anchor, recent]
        if anchor.size or recent.size:
            out = np.concatenate([anchor, recent], axis=0)
        else:
            # fallback: just take the last ctx window from recent
            out = recent_tokens[-ctx_frames:]

        # 4) Trim if we overshot
        if out.shape[0] > ctx_frames:
            out = out[-ctx_frames:]

        # 5) Snap the **END** to the nearest LOWER bar boundary
        if frames_per_bar > 0:
            max_bar_aligned = (out.shape[0] // frames_per_bar) * frames_per_bar
        else:
            max_bar_aligned = out.shape[0]
        if max_bar_aligned > 0 and out.shape[0] != max_bar_aligned:
            out = out[-max_bar_aligned:]

        # 6) Left-fill to reach ctx_frames **without moving the END**
        deficit = ctx_frames - out.shape[0]
        if deficit > 0:
            left_parts = []

            # Prefer frames immediately BEFORE the region we used from 'recent_tokens'
            if used_recent < recent_tokens.shape[0]:
                take = min(deficit, recent_tokens.shape[0] - used_recent)
                if used_recent > 0:
                    left_parts.append(recent_tokens[-(used_recent + take) : -used_recent])
                else:
                    left_parts.append(recent_tokens[-take:])

            # Then take frames immediately BEFORE the 'anchor' in original_tokens
            if sum(p.shape[0] for p in left_parts) < deficit and anchor.shape[0] > 0:
                need = deficit - sum(p.shape[0] for p in left_parts)
                a_len = anchor.shape[0]
                avail = max(original_tokens.shape[0] - a_len, 0)
                take2 = min(need, avail)
                if take2 > 0:
                    left_parts.append(original_tokens[-(a_len + take2) : -a_len])

            # Still short? tile from what's available
            have = sum(p.shape[0] for p in left_parts)
            if have < deficit:
                base = out if out.shape[0] > 0 else (recent_tokens if recent_tokens.shape[0] > 0 else original_tokens)
                reps = int(np.ceil((deficit - have) / max(1, base.shape[0])))
                left_parts.append(np.tile(base, (reps, 1))[: (deficit - have)])

            left = np.concatenate(left_parts, axis=0)
            out = np.concatenate([left[-deficit:], out], axis=0)

        # 7) Final guard to exact length
        if out.shape[0] > ctx_frames:
            out = out[-ctx_frames:]
        elif out.shape[0] < ctx_frames:
            reps = int(np.ceil(ctx_frames / max(1, out.shape[0])))
            out = np.tile(out, (reps, 1))[-ctx_frames:]

        # 8) Depth guard
        if out.shape[1] != depth:
            out = out[:, :depth]
        return out

    
    def _realign_emit_pointer_to_bar(self, sr_model: int):
        """Advance _next_emit_start to the next bar boundary in model-sample space."""
        bar_samps = int(round(self._seconds_per_bar() * sr_model))
        if bar_samps <= 0:
            return
        phase = self._next_emit_start % bar_samps
        if phase != 0:
            self._next_emit_start += (bar_samps - phase)
    
    def _prepare_stream_for_reseed_handoff(self):
        # OLD: keep crossfade tail -> causes phase offset
        # sr = int(self.mrt.sample_rate)
        # xfade_s = float(self.mrt.config.crossfade_length)
        # xfade_n = int(round(xfade_s * sr))
        # if getattr(self, "_stream", None) is not None and self._stream.shape[0] > 0:
        #     tail = self._stream[-xfade_n:] if self._stream.shape[0] > xfade_n else self._stream
        #     self._stream = tail.copy()
        # else:
        #     self._stream = None

        # NEW: throw away the tail completely; start fresh
        self._stream = None

        self._next_emit_start = 0
        self._needs_bar_realign = True

    def reseed_splice(self, recent_wav, anchor_bars: float):
        """
        Token-splice reseed queued for the next bar boundary between chunks.
        """
        with self._lock:
            if not hasattr(self, "_original_context_tokens") or self._original_context_tokens is None:
                self._original_context_tokens = np.copy(self.state.context_tokens)

            recent_tokens = self._make_recent_tokens_from_wave(recent_wav)  # [T, depth]
            new_ctx = self._splice_context(self._original_context_tokens, recent_tokens, anchor_bars)

            # Queue it; the run loop will install right after we finish the current slice
            self._pending_reseed = {"ctx": new_ctx, "ref": recent_wav}

            # install the new context window
            new_state = self.mrt.init_state()
            new_state.context_tokens = new_ctx
            self.state = new_state

            self._prepare_stream_for_reseed_handoff()

            # optional: ask streamer to drop an intro crossfade worth of audio right after reseed
            self._pending_drop_intro_bars = getattr(self, "_pending_drop_intro_bars", 0) + 1

    def run(self):
        """Main worker loop — generate into a continuous stream, then emit bar-aligned slices."""
        spb = self._seconds_per_bar()                     # seconds per bar
        chunk_secs = self.params.bars_per_chunk * spb
        xfade = float(self.mrt.config.crossfade_length)   # seconds
        sr = int(self.mrt.sample_rate)
        chunk_step_f = chunk_secs * sr    # float samples per chunk
        self._emit_phase = getattr(self, "_emit_phase", 0.0)

        def _need(first_chunk_extra: bool = False) -> int:
            """
            How many more samples we still need in the stream to emit the next slice.
            Uses the fractional step (chunk_step_f) + current _emit_phase to compute
            the *integer* number of samples required for the next chunk, without
            mutating _emit_phase here.
            """
            start = getattr(self, "_next_emit_start", 0)
            total = 0 if getattr(self, "_stream", None) is None else self._stream.shape[0]
            have = max(0, total - start)

            # Compute the integer step we'd use for the next emit, non-mutating.
            emit_phase = float(getattr(self, "_emit_phase", 0.0))
            step_int = int(floor(chunk_step_f + emit_phase))  # matches the logic used when advancing

            # How much we want available beyond 'start' for this emit.
            want = step_int
            if first_chunk_extra:
                # Reserve two extra bars so the first-chunk onset alignment has material.
                # Use ceil to be conservative so we don't under-request.
                want += int(ceil(2.0 * spb * sr))

            return max(0, want - have)

        def _mono_env(x: np.ndarray, sr: int, win_ms: float = 10.0) -> np.ndarray:
            if x.ndim == 2: x = x.mean(axis=1)
            x = np.abs(x).astype(np.float32)
            w = max(1, int(round(win_ms * 1e-3 * sr)))
            if w > 1:
                kern = np.ones(w, dtype=np.float32) / float(w)
                x = np.convolve(x, kern, mode="same")
            d = np.diff(x, prepend=x[:1])
            d[d < 0] = 0.0
            return d

        def _estimate_first_offset_samples(ref_loop_wav, gen_head_wav, sr: int, spb: float) -> int:
            """Tempo-aware first-downbeat offset (positive => model late)."""
            try:
                max_ms = int(max(160.0, min(0.25 * spb * 1000.0, 450.0)))
                ref = ref_loop_wav if ref_loop_wav.sample_rate == sr else ref_loop_wav.resample(sr)
                n_bar = int(round(spb * sr))
                ref_tail = ref.samples[-n_bar:, :] if ref.samples.shape[0] >= n_bar else ref.samples
                gen_head = gen_head_wav.samples[: int(2 * n_bar), :]
                if ref_tail.size == 0 or gen_head.size == 0:
                    return 0

                # envelopes + z-score
                def _z(a):
                    m, s = float(a.mean()), float(a.std() or 1.0); return (a - m) / s
                e_ref = _z(_mono_env(ref_tail, sr)).astype(np.float32)
                e_gen = _z(_mono_env(gen_head, sr)).astype(np.float32)

                # upsample x4 for finer lag
                def _upsample(a, r=4):
                    n = len(a); grid = np.arange(n, dtype=np.float32)
                    fine = np.linspace(0, n - 1, num=n * r, dtype=np.float32)
                    return np.interp(fine, grid, a).astype(np.float32)
                up = 4
                e_ref_u, e_gen_u = _upsample(e_ref, up), _upsample(e_gen, up)

                max_lag_u = int(round((max_ms / 1000.0) * sr * up))
                seg = min(len(e_ref_u), len(e_gen_u))
                e_ref_u = e_ref_u[-seg:]
                pad = np.zeros(max_lag_u, dtype=np.float32)
                e_gen_u_pad = np.concatenate([pad, e_gen_u, pad])

                best_lag_u, best_score = 0, -1e9
                for lag_u in range(-max_lag_u, max_lag_u + 1):
                    start = max_lag_u + lag_u
                    b = e_gen_u_pad[start : start + seg]
                    denom = (np.linalg.norm(e_ref_u) * np.linalg.norm(b)) or 1.0
                    score = float(np.dot(e_ref_u, b) / denom)
                    if score > best_score:
                        best_score, best_lag_u = score, lag_u
                return int(round(best_lag_u / up))
            except Exception:
                return 0

        print("🚀 JamWorker started (bar-aligned streaming)…")

        while not self._stop_event.is_set():
            if not self._should_generate_next_chunk():
                time.sleep(0.25)
                continue

            # 1) Generate until we have enough material in the stream
            need = _need(first_chunk_extra=(self.idx == 0))
            while need > 0 and not self._stop_event.is_set():
                with self._lock:
                    style_vec = self.params.style_vec
                    self.mrt.guidance_weight = float(self.params.guidance_weight)
                    self.mrt.temperature     = float(self.params.temperature)
                    self.mrt.topk            = int(self.params.topk)
                wav, self.state = self.mrt.generate_chunk(state=self.state, style=style_vec)
                self._append_model_chunk_to_stream(wav)   # equal-power xfade into a persistent stream
                need = _need(first_chunk_extra=(self.idx == 0))

            if self._stop_event.is_set():
                break

            # 2) One-time: align the emit pointer to the groove
            if (self.idx == 0 and self.params.combined_loop is not None) or self._needs_bar_realign:
                ref_loop = self._reseed_ref_loop or self.params.combined_loop
                if ref_loop is not None:
                    head_len = min(self._stream.shape[0] - self._next_emit_start, int(round(2 * spb * sr)))
                    seg = self._stream[self._next_emit_start : self._next_emit_start + head_len]
                    gen_head = au.Waveform(seg.astype(np.float32, copy=False), sr).as_stereo()
                    offs = _estimate_first_offset_samples(ref_loop, gen_head, sr, spb)
                    if offs != 0:
                        self._next_emit_start = max(0, self._next_emit_start + offs)
                        print(f"🎯 Offset compensation: {offs/sr:+.3f}s")
                    self._realign_emit_pointer_to_bar(sr)
                self._needs_bar_realign = False
                self._reseed_ref_loop = None

            # 3) Emit exactly bars_per_chunk × spb from the stream
            start = self._next_emit_start
            step_total = chunk_step_f + self._emit_phase
            step_int   = int(np.floor(step_total))
            self._emit_phase = float(step_total - step_int)
            end = start + step_int
            if end > self._stream.shape[0]:
                # shouldn't happen often; generate a bit more and loop
                continue

            slice_ = self._stream[start:end]
            self._next_emit_start = end

            y = au.Waveform(slice_.astype(np.float32, copy=False), sr).as_stereo()

            # 4) Post-processing / loudness
            if self.idx == 0 and self.params.ref_loop is not None:
                y, _ = match_loudness_to_reference(
                    self.params.ref_loop, y,
                    method=self.params.loudness_mode,
                    headroom_db=self.params.headroom_db
                )
            else:
                apply_micro_fades(y, 3)

            # 5) Resample + exact-length snap + encode
            b64, meta = self._snap_and_encode(
                y, seconds=chunk_secs, target_sr=self.params.target_sr, bars=self.params.bars_per_chunk
            )
            meta["xfade_seconds"] = xfade

            # 6) Publish
            with self._lock:
                self.idx += 1
                self.outbox.append(JamChunk(index=self.idx, audio_base64=b64, metadata=meta))
                if len(self.outbox) > 10:
                    cutoff = self._last_delivered_index - 5
                    self.outbox = [ch for ch in self.outbox if ch.index > cutoff]

                # 👉 If a reseed was requested, apply it *now*, between chunks
                if self._pending_reseed is not None:
                    pkg = self._pending_reseed
                    self._pending_reseed = None

                    new_state = self.mrt.init_state()
                    new_state.context_tokens = pkg["ctx"]          # exact (ctx_frames, depth)
                    self.state = new_state

                    # start a fresh stream and schedule one-time alignment
                    self._stream = None
                    self._next_emit_start = 0
                    self._reseed_ref_loop = pkg.get("ref") or self.params.combined_loop
                    self._needs_bar_realign = True

                    print("🔁 Reseed installed at bar boundary; will realign before next slice")

            print(f"✅ Completed chunk {self.idx}")

        print("🛑 JamWorker stopped")

