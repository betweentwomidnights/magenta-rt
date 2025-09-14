### Current observations
- **L40S 48GB** → faster than realtime. Use `pace:"realtime"` to avoid client over‑buffering.
- **L4 24GB** → slightly **below** realtime even with pre‑roll buffering, TF32/Autotune, smaller chunks (`max_decode_frames`), and the **base** checkpoint.

### Practical guidance
- For consistent realtime, target **~40GB VRAM per active stream** (e.g., **A100 40GB**, or MIG slices ≈ **35–40GB** on newer GPUs).
- Keep client‑side **overlap‑add** (25–40 ms) for seamless chunk joins.
- Prefer **`pace:"realtime"`** once playback begins; use **ASAP** only to build a short pre‑roll if needed.
- Optional knob: **`max_decode_frames`** (default **50** ≈ 2.0 s). Reducing to **36–45** can lower per‑chunk latency/VRAM, but doesn't increase frames/sec throughput.

### Concurrency
This research build is designed for **one active jam per GPU**. Concurrency would require GPU partitioning (MIG) or horizontal scaling with a session scheduler.