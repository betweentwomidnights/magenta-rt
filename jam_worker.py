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
        """
        Main worker loop:
        • Generate continuous audio at model/native SR (sr_in).
        • Maintain input-domain emit pointer for groove realign.
        • Maintain an OUTPUT-domain streaming resampler (sr_out = 44100 by default).
        • Emit EXACTLY bars_per_chunk at sr_out using a fractional phase accumulator.
        • No per-chunk resampling; resampler carries state across chunks => seamless.
        """
        import numpy as np
        import time
        from math import floor, ceil
        from utils import wav_bytes_base64, match_loudness_to_reference, apply_micro_fades

        # ---------- Session timing ----------
        spb         = self._seconds_per_bar()                         # seconds per bar
        chunk_secs  = float(self.params.bars_per_chunk) * float(spb)  # seconds per emitted chunk

        # ---------- Sample rates ----------
        sr_in  = int(self.mrt.sample_rate)                            # model/native SR (e.g., 48000)
        sr_out = int(getattr(self.params, "target_sr", 44100) or 44100)  # desired client SR (44.1k by default)
        self.params.target_sr = sr_out  # reflect back in metadata

        # ---------- Crossfade (model-side stitching), seconds ----------
        xfade_seconds = float(self.mrt.config.crossfade_length)

        # ---------- INPUT-domain emit step (used for groove realign + generation need) ----------
        chunk_step_in_f   = chunk_secs * sr_in               # float samples per chunk (input domain)
        self._emit_phase  = float(getattr(self, "_emit_phase", 0.0))  # carry across loops

        # ---------- OUTPUT-domain emit step (controls exact client length) ----------
        chunk_step_out_f   = chunk_secs * sr_out
        self._emit_phase_out = float(getattr(self, "_emit_phase_out", 0.0))
        self._next_emit_start_out = int(getattr(self, "_next_emit_start_out", 0))

        # ---------- Continuous resampler state (into sr_out) ----------
        self._resampler   = None
        self._stream_out  = np.zeros((0, int(self.params.channels or 2)), dtype=np.float32)
        if sr_out != sr_in:
            # Lazy import to avoid hard dep if not needed
            from utils import StreamingResampler
            ch = int(self.params.channels or 2)
            self._resampler = StreamingResampler(in_sr=sr_in, out_sr=sr_out, channels=ch, quality="VHQ")

        # ---------- INPUT stream / pointers ----------
        # self._stream: np.ndarray (S_in, C) grows as we generate
        # self._next_emit_start: input-domain pointer we realign to bar boundary once at start / reseed
        self._stream = getattr(self, "_stream", None)
        self._next_emit_start = int(getattr(self, "_next_emit_start", 0))
        self._needs_bar_realign = bool(getattr(self, "_needs_bar_realign", True))

        # How much of INPUT we have already fed into the resampler (in samples @ sr_in)
        input_consumed = int(getattr(self, "_input_consumed", 0))

        # Delivery bookkeeping
        self.idx = int(getattr(self, "idx", 0))
        self._last_delivered_index = int(getattr(self, "_last_delivered_index", 0))
        self.outbox = getattr(self, "outbox", [])

        print("🚀 JamWorker started (bar-aligned streaming, stateful resampler)…")

        # ---------- Helpers inside run() ----------
        def _need_input(first_chunk_extra: bool = False) -> int:
            """
            How many INPUT-domain samples we still need in self._stream to be comfortable
            before emitting the next slice. Mirrors your fractional step math without
            mutating _emit_phase here.
            """
            total = 0 if self._stream is None else self._stream.shape[0]
            start = int(getattr(self, "_next_emit_start", 0))
            have  = max(0, total - start)

            # Integer step we will advance by (input domain), non-mutating:
            step_int = int(floor(chunk_step_in_f + float(getattr(self, "_emit_phase", 0.0))))

            want = step_int
            if first_chunk_extra:
                # reserve 2 extra bars for downbeat/onset alignment safety
                want += int(ceil(2.0 * spb * sr_in))

            return max(0, want - have)

        def _feed_resampler_as_needed():
            """
            Ensure OUTPUT buffer (_stream_out) has resampled audio for any new INPUT
            samples appended to self._stream since we last consumed it.
            """
            nonlocal input_consumed, sr_in, sr_out
            total_in = 0 if self._stream is None else self._stream.shape[0]
            if total_in <= input_consumed:
                return  # nothing new to feed

            # Slice the new INPUT region and push through streaming resampler (or pass-through)
            new_in = self._stream[input_consumed:total_in]
            if new_in.size == 0:
                return

            if self._resampler is not None:
                y_out = self._resampler.process(new_in, final=False)
            else:
                # No resampling needed; alias output to input
                y_out = new_in

            if y_out.size:
                self._stream_out = y_out if self._stream_out.size == 0 else np.vstack([self._stream_out, y_out])

            input_consumed = total_in  # we've fed all available input into the (re)sampler

        def _output_have():
            """How many OUTPUT-domain samples are available to emit from current pointer."""
            total_out = 0 if self._stream_out is None else self._stream_out.shape[0]
            return max(0, total_out - self._next_emit_start_out)

        def _compute_step_in() -> int:
            """Integer input-domain step for internal pointer (non-mutating)."""
            return int(floor(chunk_step_in_f + float(getattr(self, "_emit_phase", 0.0))))

        def _compute_step_out() -> int:
            """Integer output-domain step for emission (non-mutating)."""
            return int(floor(chunk_step_out_f + float(getattr(self, "_emit_phase_out", 0.0))))

        def _advance_input_pointer():
            """Advance input emit pointer by the integer step and carry fractional phase."""
            step_total = chunk_step_in_f + self._emit_phase
            step_int   = int(floor(step_total))
            self._emit_phase = float(step_total - step_int)
            self._next_emit_start += step_int

        def _advance_output_pointer():
            """Advance output emit pointer by the integer step and carry fractional phase."""
            step_total = chunk_step_out_f + self._emit_phase_out
            step_int   = int(floor(step_total))
            self._emit_phase_out = float(step_total - step_int)
            self._next_emit_start_out += step_int

        def _trim_buffers_if_needed():
            """
            Keep memory bounded by dropping already-emitted OUTPUT and corresponding INPUT,
            while keeping indices consistent.
            """
            # Drop OUTPUT head
            if self._next_emit_start_out > 3 * int(chunk_step_out_f or sr_out):
                cut = int(self._next_emit_start_out)
                self._stream_out = self._stream_out[cut:]
                self._next_emit_start_out -= cut

            # Drop INPUT head **only** if we've consumed it into resampler AND it's before emit start
            # (emit start is for alignment math; after first chunk we keep advancing anyway)
            head_can_drop = min(input_consumed, self._next_emit_start)
            if head_can_drop > sr_in * 8:  # keep a few bars as safety
                drop = head_can_drop - int(sr_in * 4)
                if drop > 0:
                    self._stream = self._stream[drop:]
                    self._next_emit_start -= drop
                    input_consumed -= drop

        # ---------- Main loop ----------
        while not self._stop_event.is_set():
            # Throttle if we're too far ahead of the consumer
            if not self._should_generate_next_chunk():
                time.sleep(0.25)
                continue

            # 1) Ensure we have enough INPUT material for the next slice (and first-chunk extra)
            need_in = _need_input(first_chunk_extra=(self.idx == 0))
            while need_in > 0 and not self._stop_event.is_set():
                # Model generation step; xfade into persistent INPUT stream
                with self._lock:
                    style_vec = self.params.style_vec
                    self.mrt.guidance_weight = float(self.params.guidance_weight)
                    self.mrt.temperature     = float(self.params.temperature)
                    self.mrt.topk            = int(self.params.topk)

                wav, self.state = self.mrt.generate_chunk(state=self.state, style=style_vec)
                self._append_model_chunk_to_stream(wav)  # equal-power crossfade into self._stream

                # Feed any newly appended INPUT into the OUTPUT resampler
                _feed_resampler_as_needed()

                need_in = _need_input(first_chunk_extra=(self.idx == 0))

            if self._stop_event.is_set():
                break

            # 2) One-time: tempo/bar realign in INPUT domain before emitting the *first* chunk
            if self._needs_bar_realign:
                self._realign_emit_pointer_to_bar(sr_in)
                self._emit_phase = 0.0  # reset input fractional phase after snapping to grid

                # Set INPUT→RESAMPLER start so the very first OUTPUT sample corresponds to _next_emit_start
                input_consumed = max(input_consumed, self._next_emit_start)
                self._needs_bar_realign = False

                # Feed any post-snap INPUT into OUTPUT resampler so we have aligned OUTPUT available
                _feed_resampler_as_needed()

            # 3) Ensure OUTPUT buffer has enough samples for the next emission step
            step_out_int = _compute_step_out()
            while _output_have() < step_out_int and not self._stop_event.is_set():
                # If OUTPUT is short, try feeding more INPUT into resampler; if INPUT has no new data, generate more
                _feed_resampler_as_needed()
                if _output_have() < step_out_int:
                    # generate another model chunk
                    with self._lock:
                        style_vec = self.params.style_vec
                        self.mrt.guidance_weight = float(self.params.guidance_weight)
                        self.mrt.temperature     = float(self.params.temperature)
                        self.mrt.topk            = int(self.params.topk)
                    wav, self.state = self.mrt.generate_chunk(state=self.state, style=style_vec)
                    self._append_model_chunk_to_stream(wav)
                    _feed_resampler_as_needed()

            if self._stop_event.is_set():
                break

            # 4) Slice OUTPUT-domain chunk exactly step_out_int long and (optionally) loudness-align the first one
            start_out = int(self._next_emit_start_out)
            end_out   = start_out + int(step_out_int)

            total_out = 0 if self._stream_out is None else self._stream_out.shape[0]
            if end_out > total_out:
                # Should be rare due to loop above, but guard anyway
                time.sleep(0.01)
                continue

            y_send = self._stream_out[start_out:end_out]

            if self.idx == 0 and getattr(self.params, "ref_loop", None) is not None:
                # First chunk: match loudness to reference if requested
                y_send, _ = match_loudness_to_reference(
                    self.params.ref_loop, y_send,
                    method=getattr(self.params, "loudness_mode", "integrated"),
                    headroom_db=getattr(self.params, "headroom_db", 1.0),
                )
            else:
                # With a continuous stateful resampler, no per-chunk fades are needed.
                # If you *really* want safety fades, do 1 ms only on first/last when stopping.
                pass

            # 5) Encode WAV (already exact length at sr_out)
            b64, total_samples, channels = wav_bytes_base64(y_send, sr_out)
            meta = {
                "bpm": float(self.params.bpm),
                "bars": int(self.params.bars_per_chunk),
                "seconds": float(chunk_secs),
                "sample_rate": int(sr_out),
                "samples": int(total_samples),
                "channels": int(channels),
                "xfade_seconds": float(xfade_seconds),
            }

            # 6) Publish + advance both emit pointers
            with self._lock:
                self.idx += 1
                self.outbox.append(JamChunk(index=self.idx, audio_base64=b64, metadata=meta))
                # prune outbox to keep memory in check
                if len(self.outbox) > 10:
                    cutoff = self._last_delivered_index - 5
                    self.outbox = [ch for ch in self.outbox if ch.index > cutoff]

                # Handle reseed requests BETWEEN chunks
                if getattr(self, "_pending_reseed", None) is not None:
                    pkg = self._pending_reseed
                    self._pending_reseed = None

                    # Reset model state with fresh bar-aligned context tokens
                    new_state = self.mrt.init_state()
                    new_state.context_tokens = pkg["ctx"]
                    self.state = new_state

                    # Reset INPUT stream and schedule one-time bar realign
                    self._stream = None
                    self._next_emit_start = 0
                    self._reseed_ref_loop = pkg.get("ref") or self.params.combined_loop
                    self._needs_bar_realign = True

                    # Reset OUTPUT-domain streaming state
                    self._stream_out = np.zeros((0, int(self.params.channels or 2)), dtype=np.float32)
                    self._next_emit_start_out = 0
                    self._emit_phase_out = 0.0
                    input_consumed = 0
                    if self._resampler is not None:
                        # Rebuild the resampler to clear its filter tail
                        from utils import StreamingResampler
                        ch = int(self.params.channels or 2)
                        self._resampler = StreamingResampler(in_sr=sr_in, out_sr=sr_out, channels=ch, quality="VHQ")

                    print("🔁 Reseed installed at bar boundary; will realign before next slice")

            # Advance both emit pointers for next round
            _advance_input_pointer()
            _advance_output_pointer()

            # Keep memory tidy
            _trim_buffers_if_needed()

            print(f"✅ Completed chunk {self.idx}")

        # Stop: flush tail from resampler (optional)
        if self._resampler is not None:
            tail = self._resampler.flush()
            if tail.size:
                self._stream_out = tail if self._stream_out.size == 0 else np.vstack([self._stream_out, tail])

        print("🛑 JamWorker stopped")
