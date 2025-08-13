# jam_worker.py - SIMPLE FIX VERSION
import threading, time, base64, io, uuid
from dataclasses import dataclass, field
import numpy as np
import soundfile as sf

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
        
        if params.combined_loop is not None:
            self._setup_context_from_combined_loop()
            
        self.idx = 0
        self.outbox: list[JamChunk] = []
        self._stop_event = threading.Event()
        
        # NEW: Track delivery state
        self._last_delivered_index = 0
        self._max_buffer_ahead = 5  # Don't generate more than 3 chunks ahead
        
        # Timing info
        self.last_chunk_started_at = None
        self.last_chunk_completed_at = None
        self._lock = threading.Lock()

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
            
            self.state.context_tokens = context_tokens
            print(f"✅ JamWorker: Set up fresh context from combined loop")
            
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

    def run(self):
        """Main worker loop - generate chunks continuously but don't get too far ahead"""
        spb = self._seconds_per_bar()
        chunk_secs = self.params.bars_per_chunk * spb
        xfade = self.mrt.config.crossfade_length

        print("🚀 JamWorker started with flow control...")
        
        while not self._stop_event.is_set():
            # Check if we should generate the next chunk
            if not self._should_generate_next_chunk():
                # We're ahead enough, wait a bit for frontend to catch up
                print(f"⏸️  Buffer full, waiting for consumption...")
                time.sleep(0.5)
                continue

            # Generate the next chunk
            with self._lock:
                style_vec = self.params.style_vec
                self.mrt.guidance_weight = self.params.guidance_weight
                self.mrt.temperature = self.params.temperature
                self.mrt.topk = self.params.topk
                next_idx = self.idx + 1

            print(f"🎹 Generating chunk {next_idx}...")
            
            # Generate enough model chunks to cover chunk_secs
            need = chunk_secs
            chunks = []
            self.last_chunk_started_at = time.time()
            
            while need > 0 and not self._stop_event.is_set():
                wav, self.state = self.mrt.generate_chunk(state=self.state, style=style_vec)
                chunks.append(wav)
                need -= (wav.samples.shape[0] / float(self.mrt.sample_rate))

            if self._stop_event.is_set():
                break

            # Stitch and trim to exact seconds at model SR
            y = stitch_generated(chunks, self.mrt.sample_rate, xfade).as_stereo()
            y = hard_trim_seconds(y, chunk_secs)

            # Post-process
            if next_idx == 1 and self.params.ref_loop is not None:
                y, _ = match_loudness_to_reference(
                    self.params.ref_loop, y,
                    method=self.params.loudness_mode,
                    headroom_db=self.params.headroom_db
                )
            else:
                apply_micro_fades(y, 3)

            # Resample + snap + b64
            b64, meta = self._snap_and_encode(
                y, seconds=chunk_secs,
                target_sr=self.params.target_sr,
                bars=self.params.bars_per_chunk
            )

            # Store the completed chunk
            with self._lock:
                self.idx = next_idx
                self.outbox.append(JamChunk(index=next_idx, audio_base64=b64, metadata=meta))
                
                # Keep outbox bounded (remove old chunks)
                if len(self.outbox) > 10:
                    # Remove chunks that are way behind the delivery point
                    self.outbox = [ch for ch in self.outbox if ch.index > self._last_delivered_index - 5]

            self.last_chunk_completed_at = time.time()
            print(f"✅ Completed chunk {next_idx}")

        print("🛑 JamWorker stopped")