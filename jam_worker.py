# jam_worker.py - COMPREHENSIVE REWRITE FOR PRECISE TIMING
import threading, time, base64, io, uuid, math
from dataclasses import dataclass, field
import numpy as np
import soundfile as sf
from magenta_rt import audio as au
from threading import RLock
from utils import (
    match_loudness_to_reference, stitch_generated, hard_trim_seconds,
    apply_micro_fades, make_bar_aligned_context, take_bar_aligned_tail,
    resample_and_snap, wav_bytes_base64, StreamingResampler
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

@dataclass
class TimingState:
    """Precise timing state tracking"""
    # Fractional bar position (never rounded until final emission)
    emit_position_bars: float = 0.0
    
    # Sample-accurate positions in the stream
    stream_position_samples: int = 0
    
    # Accumulated timing error for correction
    fractional_error_bars: float = 0.0
    
    # Codec frame timing
    frames_per_bar: float = 0.0
    samples_per_bar: float = 0.0
    
    def advance_by_bars(self, bars: float):
        """Advance timing by exact fractional bars"""
        self.emit_position_bars += bars
        self.fractional_error_bars += bars - int(bars)
        
        # Correct for accumulated error when it gets significant
        if abs(self.fractional_error_bars) > 0.5:
            correction = int(round(self.fractional_error_bars))
            self.fractional_error_bars -= correction
            return correction  # bars to skip/rewind
        return 0

class JamWorker(threading.Thread):
    def __init__(self, mrt, params: JamParams):
        super().__init__(daemon=True)
        self.mrt = mrt
        self.params = params
        self.state = mrt.init_state()

        # Core timing calculations (keep as floats for precision)
        self._codec_fps = float(self.mrt.codec.frame_rate)  # 25.0
        self._model_sr = int(self.mrt.sample_rate)          # 48000
        self._target_sr = int(params.target_sr)
        
        # Critical: these stay as floats to preserve fractional precision
        self._seconds_per_bar = float(params.beats_per_bar * 60.0 / params.bpm)
        self._frames_per_bar = self._seconds_per_bar * self._codec_fps
        self._samples_per_bar_model = self._seconds_per_bar * self._model_sr
        self._samples_per_bar_target = self._seconds_per_bar * self._target_sr
        
        # Timing state
        self._timing = TimingState(
            frames_per_bar=self._frames_per_bar,
            samples_per_bar=self._samples_per_bar_model
        )
        
        # Warn about problematic BPMs
        frame_error = abs(self._frames_per_bar - round(self._frames_per_bar))
        if frame_error > 0.01:
            print(f"⚠️ Warning: {params.bpm} BPM creates {frame_error:.3f} frame drift per bar")
            print(f"   This may cause gradual timing drift in long jams")

        # Synchronization + placeholders
        self._lock = threading.Lock()
        self._original_context_tokens = None

        if params.combined_loop is not None:
            self._setup_context_from_combined_loop()

        self.idx = 0
        self.outbox: list[JamChunk] = []
        self._stop_event = threading.Event()

        # Stream state
        self._stream = None
        self._stream_write_pos = 0  # Where we append new model output
        
        # Delivery tracking
        self._last_delivered_index = 0
        self._max_buffer_ahead = 5

        # Streaming resampler for precise SR conversion
        self._resampler = None
        if self._target_sr != self._model_sr:
            self._resampler = StreamingResampler(
                in_sr=self._model_sr,
                out_sr=self._target_sr, 
                channels=2,
                quality="VHQ"
            )

        # Timing info
        self.last_chunk_started_at = None
        self.last_chunk_completed_at = None

        # Control flags
        self._pending_reseed = None
        self._needs_bar_realign = False
        self._reseed_ref_loop = None

    def _setup_context_from_combined_loop(self):
        """Set up MRT context tokens from the combined loop audio"""
        try:
            from utils import make_bar_aligned_context, take_bar_aligned_tail

            codec_fps = self._codec_fps
            ctx_seconds = float(self.mrt.config.context_length_frames) / codec_fps

            loop_for_context = take_bar_aligned_tail(
                self.params.combined_loop,
                self.params.bpm,
                self.params.beats_per_bar,
                ctx_seconds
            )

            tokens_full = self.mrt.codec.encode(loop_for_context).astype(np.int32)
            tokens = tokens_full[:, :self.mrt.config.decoder_codec_rvq_depth]

            # Use enhanced context alignment for fractional BPMs
            context_tokens = make_bar_aligned_context(
                tokens,
                bpm=self.params.bpm,
                fps=self._codec_fps,
                ctx_frames=self.mrt.config.context_length_frames,
                beats_per_bar=self.params.beats_per_bar,
                precise_timing=True  # Use new precise mode
            )

            self.state.context_tokens = context_tokens
            print(f"Context setup: {context_tokens.shape[0]} frames, {self._frames_per_bar:.3f} frames/bar")

            # Store original context for splice reseeding
            with self._lock:
                if not hasattr(self, "_original_context_tokens") or self._original_context_tokens is None:
                    self._original_context_tokens = np.copy(context_tokens)

        except Exception as e:
            print(f"Failed to setup context from combined loop: {e}")

    def stop(self):
        self._stop_event.set()

    def update_knobs(self, *, guidance_weight=None, temperature=None, topk=None):
        with self._lock:
            if guidance_weight is not None: 
                self.params.guidance_weight = float(guidance_weight)
            if temperature is not None:     
                self.params.temperature = float(temperature)
            if topk is not None:            
                self.params.topk = int(topk)

    def get_next_chunk(self) -> JamChunk | None:
        """Get the next sequential chunk (blocks/waits if not ready)"""
        target_index = self._last_delivered_index + 1
        
        max_wait = 30.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait and not self._stop_event.is_set():
            with self._lock:
                for chunk in self.outbox:
                    if chunk.index == target_index:
                        self._last_delivered_index = target_index
                        print(f"Delivered chunk {target_index} (bars {chunk.metadata.get('bar_range', 'unknown')})")
                        return chunk
            time.sleep(0.1)
        
        return None

    def mark_chunk_consumed(self, chunk_index: int):
        """Mark a chunk as consumed by the frontend"""
        with self._lock:
            self._last_delivered_index = max(self._last_delivered_index, chunk_index)

    def _should_generate_next_chunk(self) -> bool:
        """Check if we should generate the next chunk"""
        with self._lock:
            return self.idx <= self._last_delivered_index + self._max_buffer_ahead

    def _get_precise_chunk_samples(self, bars: float) -> int:
        """Get exact sample count for fractional bars at model SR"""
        exact_seconds = bars * self._seconds_per_bar
        return int(round(exact_seconds * self._model_sr))

    def _append_model_chunk_to_stream(self, wav):
        """Append model output to continuous stream with crossfading"""
        xfade_s = float(self.mrt.config.crossfade_length)
        sr = self._model_sr
        xfade_n = int(round(xfade_s * sr))

        s = wav.samples if wav.samples.ndim == 2 else wav.samples[:, None]

        if self._stream is None:
            # First chunk: drop model pre-roll
            if s.shape[0] > xfade_n:
                self._stream = s[xfade_n:].astype(np.float32, copy=True)
            else:
                self._stream = np.zeros((0, s.shape[1]), dtype=np.float32)
            self._stream_write_pos = self._stream.shape[0]
            return

        # Crossfade with equal-power curves
        if s.shape[0] <= xfade_n or self._stream.shape[0] < xfade_n:
            # Degenerate case
            self._stream = np.concatenate([self._stream, s], axis=0)
            self._stream_write_pos = self._stream.shape[0]
            return

        # Standard crossfade
        tail = self._stream[-xfade_n:]
        head = s[:xfade_n]

        t = np.linspace(0, np.pi/2, xfade_n, endpoint=False, dtype=np.float32)[:, None]
        eq_in, eq_out = np.sin(t), np.cos(t)
        mixed = tail * eq_out + head * eq_in

        self._stream = np.concatenate([self._stream[:-xfade_n], mixed, s[xfade_n:]], axis=0)
        self._stream_write_pos = self._stream.shape[0]

    def _extract_precise_chunk(self, start_bars: float, chunk_bars: float) -> np.ndarray:
        """Extract exactly chunk_bars worth of audio starting at start_bars"""
        start_samples = self._get_precise_chunk_samples(start_bars)
        chunk_samples = self._get_precise_chunk_samples(chunk_bars)
        end_samples = start_samples + chunk_samples
        
        if end_samples > self._stream.shape[0]:
            return None  # Not enough audio generated yet
            
        return self._stream[start_samples:end_samples]

    def _perform_onset_alignment(self, ref_loop: au.Waveform) -> float:
        """Estimate timing offset between generated audio and reference"""
        if self._stream is None or self._stream.shape[0] < self._model_sr:
            return 0.0
            
        try:
            # Take first ~2 seconds of generated audio
            gen_samples = min(int(2.0 * self._model_sr), self._stream.shape[0])
            gen_head = au.Waveform(
                self._stream[:gen_samples].astype(np.float32, copy=False), 
                self._model_sr
            ).as_stereo()
            
            # Reference: last bar of the loop
            ref_samples = int(self._seconds_per_bar * ref_loop.sample_rate)
            if ref_loop.samples.shape[0] >= ref_samples:
                ref_tail = au.Waveform(
                    ref_loop.samples[-ref_samples:], 
                    ref_loop.sample_rate
                ).resample(self._model_sr).as_stereo()
            else:
                ref_tail = ref_loop.resample(self._model_sr).as_stereo()
            
            # Cross-correlation based alignment
            def envelope(x, sr):
                if x.ndim == 2:
                    x = x.mean(axis=1)
                x = np.abs(x).astype(np.float32)
                # Simple smoothing
                win = max(1, int(0.01 * sr))  # 10ms window
                if win > 1:
                    kernel = np.ones(win) / win
                    x = np.convolve(x, kernel, mode='same')
                return x
            
            env_ref = envelope(ref_tail.samples, self._model_sr)
            env_gen = envelope(gen_head.samples, self._model_sr)
            
            # Limit search range to reasonable offset
            max_offset_samples = int(0.2 * self._model_sr)  # 200ms max
            
            # Normalize for correlation
            env_ref = (env_ref - env_ref.mean()) / (env_ref.std() + 1e-8)
            env_gen = (env_gen - env_gen.mean()) / (env_gen.std() + 1e-8)
            
            # Find best correlation
            best_offset = 0
            best_corr = -1.0
            
            search_len = min(len(env_ref), len(env_gen) - max_offset_samples)
            if search_len > 0:
                for offset in range(0, max_offset_samples, 4):  # subsample for speed
                    if offset + search_len >= len(env_gen):
                        break
                    corr = np.corrcoef(env_ref[:search_len], env_gen[offset:offset+search_len])[0,1]
                    if not np.isnan(corr) and corr > best_corr:
                        best_corr = corr
                        best_offset = offset
                        
            offset_seconds = best_offset / self._model_sr
            print(f"Onset alignment: {offset_seconds:.3f}s offset (correlation: {best_corr:.3f})")
            return offset_seconds
            
        except Exception as e:
            print(f"Onset alignment failed: {e}")
            return 0.0

    def _align_to_bar_boundary(self):
        """Align timing state to next bar boundary"""
        current_bar = self._timing.emit_position_bars
        next_bar = math.ceil(current_bar)
        
        if abs(next_bar - current_bar) > 1e-6:
            skip_bars = next_bar - current_bar
            skip_samples = self._get_precise_chunk_samples(skip_bars)
            self._timing.stream_position_samples += skip_samples
            self._timing.emit_position_bars = next_bar
            print(f"Aligned to bar {next_bar:.0f}, skipped {skip_bars:.4f} bars")

    def reseed_from_waveform(self, wav):
        """Full context replacement reseed"""
        new_state = self.mrt.init_state()
        
        # Build new context from waveform
        codec_fps = self._codec_fps
        ctx_seconds = float(self.mrt.config.context_length_frames) / codec_fps
        
        tail = take_bar_aligned_tail(wav, self.params.bpm, self.params.beats_per_bar, ctx_seconds)
        tokens_full = self.mrt.codec.encode(tail).astype(np.int32)
        tokens = tokens_full[:, :self.mrt.config.decoder_codec_rvq_depth]
        
        context_tokens = make_bar_aligned_context(
            tokens,
            bpm=self.params.bpm, 
            fps=self._codec_fps,
            ctx_frames=self.mrt.config.context_length_frames,
            beats_per_bar=self.params.beats_per_bar,
            precise_timing=True
        )
        
        new_state.context_tokens = context_tokens
        self.state = new_state
        
        # Reset stream
        self._stream = None
        self._stream_write_pos = 0
        self._timing = TimingState(
            frames_per_bar=self._frames_per_bar,
            samples_per_bar=self._samples_per_bar_model
        )
        self._needs_bar_realign = True
        self._reseed_ref_loop = wav

    def reseed_splice(self, recent_wav, anchor_bars: float):
        """Token-splice reseed"""
        with self._lock:
            if not hasattr(self, "_original_context_tokens") or self._original_context_tokens is None:
                self._original_context_tokens = np.copy(self.state.context_tokens)

            # Build new context via splicing
            recent_tokens = self._make_recent_tokens_from_wave(recent_wav)
            new_ctx = self._splice_context(self._original_context_tokens, recent_tokens, anchor_bars)

            self._pending_reseed = {"ctx": new_ctx, "ref": recent_wav}
            
            # Install immediately
            new_state = self.mrt.init_state()
            new_state.context_tokens = new_ctx
            self.state = new_state

            # Reset stream state  
            self._stream = None
            self._stream_write_pos = 0
            self._timing = TimingState(
                frames_per_bar=self._frames_per_bar,
                samples_per_bar=self._samples_per_bar_model
            )
            self._needs_bar_realign = True

    def _make_recent_tokens_from_wave(self, wav) -> np.ndarray:
        """Encode waveform to context tokens with precise alignment"""
        tokens_full = self.mrt.codec.encode(wav).astype(np.int32)
        tokens = tokens_full[:, :self.mrt.config.decoder_codec_rvq_depth]
        
        context_tokens = make_bar_aligned_context(
            tokens,
            bpm=self.params.bpm,
            fps=self._codec_fps,
            ctx_frames=self.mrt.config.context_length_frames,
            beats_per_bar=self.params.beats_per_bar,
            precise_timing=True
        )
        return context_tokens

    def _splice_context(self, original_tokens: np.ndarray, recent_tokens: np.ndarray, anchor_bars: float) -> np.ndarray:
        """Enhanced context splicing with fractional bar handling"""
        ctx_frames = int(self.mrt.config.context_length_frames)
        
        # Convert anchor bars to codec frames (keep fractional precision)
        anchor_frames_f = anchor_bars * self._frames_per_bar
        anchor_frames = int(round(anchor_frames_f))
        
        # Take anchor from original
        anchor = original_tokens[-anchor_frames:] if anchor_frames <= original_tokens.shape[0] else original_tokens
        
        # Fill remainder with recent tokens
        remain_frames = ctx_frames - anchor.shape[0]
        if remain_frames > 0:
            recent = recent_tokens[-remain_frames:] if remain_frames <= recent_tokens.shape[0] else recent_tokens
        else:
            recent = recent_tokens[:0]  # empty
        
        # Combine
        if anchor.size > 0 and recent.size > 0:
            spliced = np.concatenate([recent, anchor], axis=0)
        elif anchor.size > 0:
            spliced = anchor
        else:
            spliced = recent_tokens[-ctx_frames:]
        
        # Ensure exact length
        if spliced.shape[0] > ctx_frames:
            spliced = spliced[-ctx_frames:]
        elif spliced.shape[0] < ctx_frames:
            # Tile to fill
            reps = int(np.ceil(ctx_frames / max(1, spliced.shape[0])))
            spliced = np.tile(spliced, (reps, 1))[-ctx_frames:]
        
        return spliced

    def run(self):
        """Main generation loop with precise timing"""
        chunk_bars = float(self.params.bars_per_chunk)
        chunk_samples = self._get_precise_chunk_samples(chunk_bars)
        xfade_s = float(self.mrt.config.crossfade_length)

        def _samples_needed(first_chunk_extra=False):
            """Calculate samples needed in stream for next emission"""
            available = 0 if self._stream is None else (
                self._stream.shape[0] - self._timing.stream_position_samples
            )
            required = chunk_samples
            if first_chunk_extra:
                # Extra material for onset alignment
                extra_samples = self._get_precise_chunk_samples(2.0)
                required += extra_samples
            return max(0, required - available)

        print(f"JamWorker started: {self.params.bpm} BPM, {self._frames_per_bar:.3f} frames/bar, {chunk_bars} bars/chunk")

        while not self._stop_event.is_set():
            if not self._should_generate_next_chunk():
                time.sleep(0.25)
                continue

            # 1) Generate until we have enough audio
            needed = _samples_needed(first_chunk_extra=(self.idx == 0))
            while needed > 0 and not self._stop_event.is_set():
                with self._lock:
                    style_vec = self.params.style_vec
                    self.mrt.guidance_weight = float(self.params.guidance_weight)
                    self.mrt.temperature = float(self.params.temperature)
                    self.mrt.topk = int(self.params.topk)
                    
                wav, self.state = self.mrt.generate_chunk(state=self.state, style=style_vec)
                self._append_model_chunk_to_stream(wav)
                needed = _samples_needed(first_chunk_extra=(self.idx == 0))

            if self._stop_event.is_set():
                break

            # 2) First chunk: perform onset alignment
            if (self.idx == 0 and self.params.combined_loop is not None) or self._needs_bar_realign:
                ref_loop = self._reseed_ref_loop or self.params.combined_loop
                if ref_loop is not None:
                    offset_seconds = self._perform_onset_alignment(ref_loop)
                    if abs(offset_seconds) > 0.01:  # More than 10ms
                        offset_samples = int(round(offset_seconds * self._model_sr))
                        self._timing.stream_position_samples = max(0, offset_samples)
                        print(f"Applied onset offset: {offset_seconds:.3f}s")
                
                self._align_to_bar_boundary()
                self._needs_bar_realign = False
                self._reseed_ref_loop = None

            # 3) Extract precise chunk
            chunk_start_bars = self._timing.emit_position_bars
            slice_audio = self._extract_precise_chunk(chunk_start_bars, chunk_bars)
            
            if slice_audio is None:
                continue  # Need more generation

            # Update timing state
            correction = self._timing.advance_by_bars(chunk_bars)
            if correction != 0:
                print(f"Applied {correction} bar timing correction")
            
            self._timing.stream_position_samples += chunk_samples

            # 4) Create waveform and process
            y = au.Waveform(slice_audio.astype(np.float32, copy=False), self._model_sr).as_stereo()

            # Loudness matching and fades
            if self.idx == 0 and self.params.ref_loop is not None:
                y, _ = match_loudness_to_reference(
                    self.params.ref_loop, y,
                    method=self.params.loudness_mode,
                    headroom_db=self.params.headroom_db
                )
            else:
                apply_micro_fades(y, 3)

            # 5) Sample rate conversion
            if self._resampler is not None:
                # Use streaming resampler for precise conversion
                resampled = self._resampler.process(y.samples, final=False)
                
                # Ensure exact target length
                target_samples = int(round(chunk_bars * self._samples_per_bar_target))
                if resampled.shape[0] != target_samples:
                    if resampled.shape[0] < target_samples:
                        pad_samples = target_samples - resampled.shape[0]
                        pad = np.zeros((pad_samples, resampled.shape[1]), dtype=resampled.dtype)
                        resampled = np.vstack([resampled, pad])
                    else:
                        resampled = resampled[:target_samples]
                
                final_audio = resampled
                final_sr = self._target_sr
            else:
                # No resampling needed
                final_audio = y.samples
                final_sr = self._model_sr

            # 6) Encode to base64
            b64, total_samples, channels = wav_bytes_base64(final_audio, final_sr)
            
            # 7) Create metadata with timing info
            actual_duration = total_samples / final_sr
            bar_range = f"{chunk_start_bars:.2f}-{self._timing.emit_position_bars:.2f}"
            
            meta = {
                "bpm": int(round(self.params.bpm)),
                "bars": int(self.params.bars_per_chunk),
                "beats_per_bar": int(self.params.beats_per_bar),
                "sample_rate": int(final_sr),
                "channels": int(channels),
                "total_samples": int(total_samples),
                "seconds_per_bar": self._seconds_per_bar,
                "loop_duration_seconds": actual_duration,
                "bar_range": bar_range,
                "timing_state": {
                    "emit_position_bars": self._timing.emit_position_bars,
                    "frames_per_bar": self._frames_per_bar,
                    "fractional_error": self._timing.fractional_error_bars,
                },
                "xfade_seconds": xfade_s,
                "guidance_weight": self.params.guidance_weight,
                "temperature": self.params.temperature,
                "topk": self.params.topk,
            }

            # 8) Publish chunk
            with self._lock:
                self.idx += 1
                chunk = JamChunk(index=self.idx, audio_base64=b64, metadata=meta)
                self.outbox.append(chunk)
                
                # Cleanup old chunks
                if len(self.outbox) > 10:
                    cutoff = self._last_delivered_index - 5
                    self.outbox = [ch for ch in self.outbox if ch.index > cutoff]

                # Handle pending reseeds
                if self._pending_reseed is not None:
                    pkg = self._pending_reseed
                    self._pending_reseed = None

                    new_state = self.mrt.init_state()
                    new_state.context_tokens = pkg["ctx"]
                    self.state = new_state

                    # Reset timing and stream
                    self._stream = None
                    self._stream_write_pos = 0
                    self._timing = TimingState(
                        frames_per_bar=self._frames_per_bar,
                        samples_per_bar=self._samples_per_bar_model
                    )
                    self._reseed_ref_loop = pkg.get("ref")
                    self._needs_bar_realign = True

                    print("Reseed applied at bar boundary")

            drift_ms = abs(self._timing.fractional_error_bars) * self._seconds_per_bar * 1000
            print(f"Completed chunk {self.idx} ({bar_range} bars, {drift_ms:.1f}ms drift)")

        print("JamWorker stopped")
        
        # Clean up resampler
        if self._resampler is not None:
            try:
                self._resampler.flush()
            except:
                pass