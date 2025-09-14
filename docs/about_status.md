## What this is
We're exploring AI‑assisted loop‑based music creation that can run on GPUs (not just TPUs) and stream to apps in realtime.

### Implemented backends
- **HTTP (bar‑aligned):** `/generate`, `/jam/start`, `/jam/next`, `/jam/stop`, `/jam/update`, etc.
- **WebSocket (realtime):** `ws://…/ws/jam` with `mode="rt"` (Colab‑style continuous chunks). New in this build.

## What we learned (GPU notes)
- **L40S 48GB:** comfortably **faster than realtime** → we added a `pace: "realtime"` switch so the server doesn't outrun playback.
- **L4 24GB:** **consistently just under realtime**; even with pre‑roll buffering, TF32/JAX tunings, reduced chunk size, and the **base** checkpoint, we still see eventual under‑runs.
- **Implication:** For production‑quality realtime, aim for ~**40GB VRAM** per user/session (e.g., **A100 40GB**, or MIG slices ≈ **35–40GB** on newer parts). Smaller GPUs can demo, but sustained realtime is not reliable.

## Model / audio specs
- **Model:** MagentaRT (T5X; decoder RVQ depth = 16)
- **Audio:** 48 kHz stereo, 2.0 s chunks by default, 40 ms crossfade
- **Context:** 10 s rolling context window