# HYPOTHESES

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | Slow tokens come from ollama splitting piccolo across CPU/GPU because cvoiced holds VRAM | **Confirmed** | 18/33 layers: 14.3/10.5 tok/s; 33/33: 43.9/41.3 tok/s, same prompts (data/bench-ollama.jsonl) |
| H2 | piccolo and cvoiced can coexist at some context | **Rejected** at 64k | ollama alone at 64k q8_0 = 6454 MiB; +2408 cvoiced > 8188 |
| H3 | turbo3 KV costs little decode speed | **Confirmed (≤5%)** | tg128 at 16k depth: 39.1 vs q8_0 41.1 |
| H4 | turbo3 KV is near-lossless | **Rejected** for this model | KLD 0.0246, top-1 80%; q4_0 does better (0.0090, 85%); q8_0 0.0025, 95%. A screen (n=60, top-10, 2k context), not a verdict |
| H5 | turbo3 is the only way to 262k on 8 GB | **Confirmed**, with a variant | turbo3 7390 MiB; q4_0-K+turbo3-V 7742 MiB but deadlocks in prefill |
| H6 | A watcher on /api/ps is enough for cvoice to yield | **Rejected by argument** | ollama picks its layer split at load time from free VRAM; a watcher reacts after the split is decided, and the model stays split until it unloads. The yield has to happen before the load, so it belongs in the request path |
| H7 | cvoiced /unload is enough to free VRAM | **Rejected** (cvoice's own measurement) | leaks ~780 MB per cycle; process exit is the full reclaim |
