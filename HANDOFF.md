# HANDOFF (2026-11-02, gpu-share agent)

Report: https://claude.ai/artifact/M2dXnVNrndfKRULptyBgcL (private; source in report/index.html)

## State of the world (verify, don't trust: run these)
- `gpu-share status`, `gpu-share-frontdoor status`, `systemctl --user status gpu-share cvoiced`
- Live since 17:18: the broker (ownbox-installed, user unit gpu-share.service) owns *:11434.
  Ollama runs on 127.0.0.1:11435 via /etc/systemd/system/ollama.service.d/40-gpu-share.conf.
  Revert with `gpu-share-frontdoor off`.
- piccolo-gorgone:9b on /v1 = turbo3 @262144 (~/data/llama-turbo3/build/bin/llama-server,
  spawned on demand, stopped after 300 s idle). piccolo-gorgone:9b-q8 = q8_0 @131072.
  Which is the default name is Bogdan's open decision; the operator said keep turbo3.
- cvoice: shelved on ownbox (6ed3f18), POST /yield (a9b5f5d). The token-printing fix went to
  a subagent; see LOG for the result.
- hotline-calld: leases are wired (hotline-call c8d3720, switch `gpu_lease`) and verified
  live through :11434 by manager-6.
- Device map: archserver/gpu-share claimed and verified.

## Open / next
- Bogdan: default KV profile (turbo3 262k, KLD 0.0246, vs q8_0 128k, KLD 0.0025).
- Not built: an /api/chat and /api/generate translator for routed models (they go to stock ollama).
- Not on ownbox: ~/data/llama-turbo3 (the fork). It's a CUDA build of several GB, so shelving
  it means a build in ~/.local/share/ownbox/tools on a root disk with ~4.7 GB free. That
  needs a decision on where the build goes (/mnt/offload now has room). gpu-share's
  config points at its binary path.
- Not leased: batch cvoiced users outside calls (kiri-wake, standin).
- Not exercised live: a real phone call under the broker (only lease round-trips), and the
  fallback to ollama when llama-server is missing (code path only).
