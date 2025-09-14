### Recent changes
- New **WebSocket realtime** route: `/ws/jam` (`mode:"rt"`)
- Added server pacing flag: `pace: "realtime" | "asap"`
- Exposed `max_decode_frames` for shorter chunks on smaller GPUs
- Client test page now does proper **overlap‑add** crossfade between chunks

### Licensing
This project uses MagentaRT under:
- **Code:** Apache 2.0
- **Model weights:** CC‑BY 4.0
Please review the MagentaRT repo for full terms.