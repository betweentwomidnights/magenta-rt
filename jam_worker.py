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

        # NEW: Track delivery state
        self._last_delivered_index = 0
        self._max_buffer_ahead = 5

        # Timing info
        self.last_chunk_started_at = None
        self.last_chunk_completed_at = None


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
                fps=int(self.mrt.codec.frame_rate),
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
            bpm=self.params.bpm, fps=int(self.mrt.codec.frame_rate),
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
            fps=int(self.mrt.codec.frame_rate),
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

        # 1) Anchor tail
        # Use floor, not round, to avoid grabbing an extra bar.
        anchor = self._bar_aligned_tail(original_tokens, math.floor(anchor_bars))

        # 2) Fill remainder with recent (in whole bars when possible)
        a = anchor.shape[0]
        remain = max(ctx_frames - a, 0)
        if remain > 0:
            bars_fit = remain // frames_per_bar
            if bars_fit >= 1:
                want_recent_frames = int(bars_fit * frames_per_bar)
                recent = recent_tokens[-want_recent_frames:] if recent_tokens.shape[0] > want_recent_frames else recent_tokens
            else:
                recent = recent_tokens[-remain:] if recent_tokens.shape[0] > remain else recent_tokens
        else:
            recent = recent_tokens[:0]

        out = np.concatenate([anchor, recent], axis=0) if (anchor.size or recent.size) else recent_tokens[-ctx_frames:]
        if out.shape[0] > ctx_frames:
            out = out[-ctx_frames:]

        # --- NEW: force total length to a whole number of bars
        max_bar_aligned = (out.shape[0] // frames_per_bar) * frames_per_bar
        if max_bar_aligned > 0 and out.shape[0] != max_bar_aligned:
            out = out[-max_bar_aligned:]

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
        Token-splice reseed:
        - original = the context we captured when the jam started
        - recent   = tokens from the provided recent waveform (usually Swift-combined mix)
        - anchor_bars controls how much of the original vibe we re-inject
        """
        with self._lock:
            if not hasattr(self, "_original_context_tokens") or self._original_context_tokens is None:
                # Fallback: if we somehow don’t have originals, treat current as originals
                self._original_context_tokens = np.copy(self.state.context_tokens)

            recent_tokens = self._make_recent_tokens_from_wave(recent_wav)          # [T, depth]
            new_ctx = self._splice_context(self._original_context_tokens, recent_tokens, anchor_bars)

            # install the new context window
            new_state = self.mrt.init_state()
            new_state.context_tokens = new_ctx
            self.state = new_state

            self._prepare_stream_for_reseed_handoff()

            # optional: ask streamer to drop an intro crossfade worth of audio right after reseed
            self._pending_drop_intro_bars = getattr(self, "_pending_drop_intro_bars", 0) + 1

    def run(self):
        """Main worker loop - generate chunks continuously but don't get too far ahead"""
        spb = self._seconds_per_bar()
        chunk_secs = self.params.bars_per_chunk * spb
        xfade = float(self.mrt.config.crossfade_length)  # seconds

        # local fallback stitcher that *keeps* the first head if utils.stitch_generated
        # doesn't yet support drop_first_pre_roll
        def _stitch_keep_head(chunks, sr: int, xfade_s: float):
            from magenta_rt import audio as au
            import numpy as _np
            if not chunks:
                raise ValueError("no chunks to stitch")
            xfade_n = int(round(max(0.0, xfade_s) * sr))
            # Fast-path: no crossfade
            if xfade_n <= 0:
                out = _np.concatenate([c.samples for c in chunks], axis=0)
                return au.Waveform(out, sr)
            # build equal-power curves
            t = _np.linspace(0, _np.pi / 2, xfade_n, endpoint=False, dtype=_np.float32)
            eq_in, eq_out = _np.sin(t)[:, None], _np.cos(t)[:, None]

            first = chunks[0].samples
            if first.shape[0] < xfade_n:
                raise ValueError("chunk shorter than crossfade prefix")
            out = first.copy()  # 👈 keep the head for live seam

            for i in range(1, len(chunks)):
                cur = chunks[i].samples
                if cur.shape[0] < xfade_n:
                    # too short to crossfade; just butt-join
                    out = _np.concatenate([out, cur], axis=0)
                    continue
                head, tail = cur[:xfade_n], cur[xfade_n:]
                mixed = out[-xfade_n:] * eq_out + head * eq_in
                out = _np.concatenate([out[:-xfade_n], mixed, tail], axis=0)
            return au.Waveform(out, sr)

        print("🚀 JamWorker started with flow control...")

        while not self._stop_event.is_set():
            # Don’t get too far ahead of the consumer
            if not self._should_generate_next_chunk():
                # We're ahead enough, wait a bit for frontend to catch up
                # (kept short so stop() stays responsive)
                time.sleep(0.5)
                continue

            # Snapshot knobs + compute index atomically
            with self._lock:
                style_vec = self.params.style_vec
                self.mrt.guidance_weight = float(self.params.guidance_weight)
                self.mrt.temperature     = float(self.params.temperature)
                self.mrt.topk            = int(self.params.topk)
                next_idx = self.idx + 1

            print(f"🎹 Generating chunk {next_idx}...")
            self.last_chunk_started_at = time.time()

            # ---- Generate enough model sub-chunks to yield *audible* chunk_secs ----
            # Count the first chunk at full length L, and each subsequent at (L - xfade)
            assembled = 0.0
            chunks = []

            while assembled < chunk_secs and not self._stop_event.is_set():
                # generate_chunk returns (au.Waveform, new_state)
                wav, self.state = self.mrt.generate_chunk(state=self.state, style=style_vec)
                chunks.append(wav)
                L = wav.samples.shape[0] / float(self.mrt.sample_rate)
                assembled += L if len(chunks) == 1 else max(0.0, L - xfade)

            if self._stop_event.is_set():
                break

            # ---- Stitch and trim at model SR (keep first head for seamless handoff) ----
            try:
                # Preferred path if you've added the new param in utils.stitch_generated
                y = stitch_generated(chunks, self.mrt.sample_rate, xfade, drop_first_pre_roll=False).as_stereo()
            except TypeError:
                # Backward-compatible: local stitcher that keeps the head
                y = _stitch_keep_head(chunks, int(self.mrt.sample_rate), xfade).as_stereo()

            # Hard trim to the exact musical duration (still at model SR)
            y = hard_trim_seconds(y, chunk_secs)

            # ---- Post-processing ----
            if next_idx == 1 and self.params.ref_loop is not None:
                # match loudness to the provided reference on the very first audible chunk
                y, _ = match_loudness_to_reference(
                    self.params.ref_loop, y,
                    method=self.params.loudness_mode,
                    headroom_db=self.params.headroom_db
                )
            else:
                # light micro-fades to guard against clicks
                apply_micro_fades(y, 3)

            # ---- Resample + bar-snap + encode ----
            b64, meta = self._snap_and_encode(
                y,
                seconds=chunk_secs,
                target_sr=self.params.target_sr,
                bars=self.params.bars_per_chunk
            )
            # small hint for the client if you want UI butter between chunks
            meta["xfade_seconds"] = xfade

            # ---- Publish the completed chunk ----
            with self._lock:
                self.idx = next_idx
                self.outbox.append(JamChunk(index=next_idx, audio_base64=b64, metadata=meta))
                # Keep outbox bounded (trim far-behind entries)
                if len(self.outbox) > 10:
                    cutoff = self._last_delivered_index - 5
                    self.outbox = [ch for ch in self.outbox if ch.index > cutoff]

            self.last_chunk_completed_at = time.time()
            print(f"✅ Completed chunk {next_idx}")

        print("🛑 JamWorker stopped")
