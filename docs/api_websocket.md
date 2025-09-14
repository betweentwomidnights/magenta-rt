Connect to `wss://…/ws/jam` and send a **JSON control stream**. In `rt` mode the server emits ~2 s WAV chunks (or binary frames) continuously.

### Start (client → server)
```jsonc
{
  "type": "start",
  "mode": "rt",
  "binary_audio": false,          // true → raw WAV bytes + separate chunk_meta
  "params": {
    "styles": "heavy metal",     // or "jazz, hiphop"
    "style_weights": "1.0,1.0",  // optional, auto‑normalized
    "temperature": 1.1,
    "topk": 40,
    "guidance_weight": 1.1,
    "pace": "realtime",          // "realtime" | "asap" (default)
    "max_decode_frames": 50       // 50≈2.0s; try 36–45 on smaller GPUs
  }
}
```

### Server events (server → client)
- `{"type":"started","mode":"rt"}` – handshake
- `{"type":"chunk","audio_base64":"…","metadata":{…}}` – base64 WAV
  - `metadata.sample_rate` *(int)* – usually 48000
  - `metadata.chunk_frames` *(int)* – e.g., 50
  - `metadata.chunk_seconds` *(float)* – frames / 25.0
  - `metadata.crossfade_seconds` *(float)* – typically 0.04
- `{"type":"chunk_meta","metadata":{…}}` – sent **after** a binary frame when `binary_audio=true`
- `{"type":"status",…}`, `{"type":"error",…}`, `{"type":"stopped"}`

### Update (client → server)
```jsonc
{
  "type": "update",
  "styles": "jazz, hiphop",
  "style_weights": "1.0,0.8",
  "temperature": 1.2,
  "topk": 64,
  "guidance_weight": 1.0,
  "pace": "realtime",            // optional live flip
  "max_decode_frames": 40         // optional; <= 50
}
```

### Stop / ping
```json
{"type":"stop"}
{"type":"ping"}
```

### Browser quick‑start (schedules seamlessly with 25–40 ms crossfade)
```html
<script>
const XFADE = 0.025; // 25 ms
let ctx, gain, ws, nextTime = 0;
async function start(){
  ctx = new (window.AudioContext||window.webkitAudioContext)();
  gain = ctx.createGain(); gain.connect(ctx.destination);
  ws = new WebSocket("wss://YOUR_SPACE/ws/jam");
  ws.onopen = ()=> ws.send(JSON.stringify({
    type:"start", mode:"rt", binary_audio:false,
    params:{ styles:"warmup", temperature:1.1, topk:40, guidance_weight:1.1, pace:"realtime" }
  }));
  ws.onmessage = async ev => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "chunk" && msg.audio_base64){
      const bin = atob(msg.audio_base64); const buf = new Uint8Array(bin.length);
      for (let i=0;i<bin.length;i++) buf[i] = bin.charCodeAt(i);
      const ab = buf.buffer; const audio = await ctx.decodeAudioData(ab);
      const src = ctx.createBufferSource(); const g = ctx.createGain();
      src.buffer = audio; src.connect(g); g.connect(gain);
      if (nextTime < ctx.currentTime + 0.05) nextTime = ctx.currentTime + 0.12;
      const startAt = nextTime, dur = audio.duration;
      nextTime = startAt + Math.max(0, dur - XFADE);
      g.gain.setValueAtTime(0, startAt);
      g.gain.linearRampToValueAtTime(1, startAt + XFADE);
      g.gain.setValueAtTime(1, startAt + Math.max(0, dur - XFADE));
      g.gain.linearRampToValueAtTime(0, startAt + dur);
      src.start(startAt);
    }
  };
}
</script>
```

### Python client (async)
```python
import asyncio, json, websockets, base64, soundfile as sf, io
async def run(url):
  async with websockets.connect(url) as ws:
    await ws.send(json.dumps({"type":"start","mode":"rt","binary_audio":False,
      "params": {"styles":"warmup","temperature":1.1,"topk":40,"guidance_weight":1.1,"pace":"realtime"}}))
    while True:
      msg = json.loads(await ws.recv())
      if msg.get("type") == "chunk":
        wav = base64.b64decode(msg["audio_base64"])  # bytes of a WAV
        x, sr = sf.read(io.BytesIO(wav), dtype="float32")
        print("chunk", x.shape, sr)
      elif msg.get("type") in ("stopped","error"): break
asyncio.run(run("wss://YOUR_SPACE/ws/jam"))
```