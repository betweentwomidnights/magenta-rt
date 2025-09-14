### Single Generation
```bash
curl -X POST \
  "$HOST/generate" \
  -F "loop_audio=@drum_loop.wav" \
  -F "bpm=120" \
  -F "bars=8" \
  -F "styles=acid house,techno" \
  -F "guidance_weight=5.0" \
  -F "temperature=1.1"
```

### Continuous Jamming (bar‑aligned, HTTP)
```bash
# 1) Start a session
echo $(curl -s -X POST "$HOST/jam/start" \
  -F "loop_audio=@loop.wav" \
  -F "bpm=120" \
  -F "bars_per_chunk=8") | jq .
# → {"session_id":"…"}

# 2) Pull next chunk (repeat)
curl "$HOST/jam/next?session_id=$SESSION"

# 3) Stop
curl -X POST "$HOST/jam/stop" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"'$SESSION'"}'
```

### Common parameters
- **bpm** *(int)* – beats per minute
- **bars / bars_per_chunk** *(int)* – musical length
- **styles** *(str)* – comma‑separated text prompts (mixed internally)
- **guidance_weight** *(float)* – style adherence (CFG weight)
- **temperature / topk** – sampling controls
- **intro_bars_to_drop** *(int, /generate)* – generate-and-trim intro