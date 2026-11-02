# gpu-share

A small HTTP broker that sits on ollama's port and decides who gets the one 8 GB
GPU. It makes cvoiced (the TTS daemon) give its VRAM back before an LLM loads,
keeps the card free for live phone calls, and serves selected models through the
turbo3 llama.cpp fork so they get a 262k context.

## Why it exists

On 2026-11-02 piccolo-gorgone was generating at 10–14 tok/s instead of 41–44.
cvoiced held 2.4 GB, so ollama loaded the model 18/33 layers on GPU and the rest
on the CPU. Ollama evicts its *own* models when it needs room, but it cannot
evict another process. Nothing else on the box could either. Measured on the
RTX 4060, same prompts, temperature 0:

| Setup | Layers on GPU | Decode, short prompt | Decode, 15k-token prompt | Prefill |
|---|---:|---:|---:|---:|
| ollama q8_0 @64k, cvoiced resident | 18/33 | 14.3 tok/s | 10.5 tok/s | 1071 tok/s |
| ollama q8_0 @64k, GPU alone | 33/33 | 43.9 tok/s | 41.3 tok/s | 1885 tok/s |

piccolo at 64k alone uses 6.45 GB, which leaves no room for cvoiced. They cannot
coexist at any useful context, so the fix has to be cvoiced yielding the card.

## How it works

```
clients (OpenHands, agents, hotline-calld)
        │  :11434
        ▼
   gpu-share ──── /v1/* for routed models ──► turbo3 llama-server (child, :11436)
        │    └─── everything else ───────────► ollama (:11435)
        │
        ├─ before any load: snapshot VRAM (nvidia-smi, ollama /api/ps, cvoiced's PID)
        │  and evict what has to go: cvoiced first, then idle LLMs, largest first
        ├─ call lease (hotline-calld): reserve 5.1 GB from ring to hang-up
        └─ reconcile every 5 s: health-poll calls, stop an idle llama-server,
           evict an ollama model left split across CPU and GPU once it fits fully
```

- **cvoiced yields by exiting.** It is sent `POST /yield` and exits after its
  in-flight synthesis; systemd restarts it empty, holding 0 MiB. Older
  cvoiced builds without `/yield` get `systemctl --user restart` instead.
  `/unload` is deliberately not used: it leaks ~780 MB per cycle (cvoice's own
  measurement), so only a process exit gives all the memory back. cvoiced
  reloads lazily on its next request, and hotline-calld already warms it at ring time.
- **Why it sits in the request path rather than watching.** Ollama picks the
  layer split at load time, from the VRAM free at that moment. A watcher on
  `/api/ps` would only see the split after it was decided, and the model stays
  split until it unloads. So the yield must happen *before* the load, which
  means the request has to pass through something.
- **Calls outrank LLMs.** Between `call/begin` and `call/end` the call's budget
  is off limits: Whisper large-v3 ~1.9 GB, cvoiced ~2.4 GB, bge-m3 0.66 GB.
  An LLM load that cannot fit beside it is held, not split. After `hold_s`
  (120 s) it gets `503` and `Retry-After: 30`. A call evicts even an in-flight
  LLM (`call.evict_inflight`), because the call is interactive and an LLM client can retry.
  An LLM load never interrupts another LLM's request.
- **Routing.** `models` maps a client-visible name to a `profile` (GGUF, ctx,
  K/V cache types). `/v1/chat/completions` and `/v1/completions` for a routed
  name go to llama-server. One profile runs at a time; switching restarts it
  (2–5 s once the weights are in page cache). The GGUF is resolved through
  ollama's manifest, never by a remembered blob hash. Sampling defaults come
  from the model's ollama params (temperature 1, top_k 20, top_p 0.95,
  presence_penalty 1.5 for piccolo), so a client sending none samples as before.
  `/api/show` reports the profile's context in `num_ctx`, and `/api/tags`,
  `/v1/models` and `/api/ps` list the alias names.
- **Degrades instead of failing.** No nvidia-smi or a failed snapshot: requests
  are forwarded unmanaged. llama-server missing or crashing on load: the routed
  model is served by stock ollama.

### The turbo3 fork, measured

`~/data/llama-turbo3` is [Madreag/turbo3-cuda](https://github.com/Madreag/turbo3-cuda)
(from TheTom/llama-cpp-turboquant), forked to BogdanStamenovic/turbo3-cuda with a
CUDA 13 build fix. piccolo-gorgone (qwen35, 262144 native, only 8 of 32 layers
carry KV):

| KV (K/V) | Largest ctx that fits 8 GB | VRAM at that ctx | Decode @0 / @16k | KLD vs f16 KV | Top-1 agree |
|---|---:|---:|---:|---:|---:|
| f16 | <128k | — | 44.2 / 40.9 | 0 (reference) | 100% |
| q8_0 | 128k | 7656 MiB | 43.9 / 41.1 | 0.0025 | 95% |
| q4_0 | 192k | 7332 MiB | — | 0.0090 | 85% |
| q8_0 / turbo3 | <262k | — | — | 0.0171 | 85% |
| turbo4 | <262k | — | — | 0.0213 | 85% |
| **turbo3** | **262k** | **7384 MiB (7746 peak)** | **43.8 / 39.1** | **0.0246** | **80%** |

At a real 89,845-token prompt through the broker, turbo3 prefilled at 1444 tok/s
and decoded at 30.3 tok/s, and answered correctly.

**Read the KLD column honestly.** turbo3 buys 4× today's context at ≤5% speed,
but its error is ~10× q8_0's, and worse than plain q4_0's. The KLD is the fork's
own `quality-tests/kl_divergence.py`: top-10 logprobs, next token, 60 wikitext-2
prompts of ~2k tokens. An f16-vs-f16 rerun gives exactly 0, so the gaps are not
run-to-run noise. It is still a screen, not a verdict: small n, a short context,
a top-k approximation. Hence two profiles:

| Name clients send | Profile | Use it for |
|---|---|---|
| `piccolo-gorgone:9b` | turbo3 @262144 | the long context he asked for |
| `piccolo-gorgone:9b-q8` | q8_0 @131072 | 2× the old 64k, near-lossless |

Which one is the default name is his call. It is one line in `config.json`.

## Install

```sh
ownbox install gpu-share        # builds (needs go), installs the user unit, starts it
gpu-share-frontdoor on          # moves ollama to :11435 (sudo) and the broker onto :11434
```

Setup asks before the front-door step. Without a TTY it skips it, and the broker
runs on a side port (127.0.0.1:11437) in front of the untouched ollama. Update
rebuilds, keeps `~/.config/gpu-share/config.json`, and restarts the unit. Remove
puts ollama back on :11434 first, so ollama is never stranded, and asks before
deleting config and logs.

## Usage

```sh
gpu-share status                    # tenants, VRAM, leases, recent evictions
gpu-share plan piccolo-gorgone:9b   # dry run: what that load would evict
gpu-share args piccolo-gorgone:9b   # the llama-server command line it would run
gpu-share call begin|renew|end      # the call lease, by hand
gpu-share config                    # effective config (file overrides built-in defaults)
gpu-share-frontdoor status|on|off [--dry-run]
```

Call lease contract (used by hotline-calld):

| Request | Response |
|---|---|
| `POST /gpu-share/call/begin {"id":"…"}` | `{"lease","ttl_s":90,"evicting",plan}`, at once (~50 ms); evictions run in the background |
| `POST /gpu-share/call/renew {"lease"}` | `{"ok":true}`, or 404 if expired (begin again) |
| `POST /gpu-share/call/end {"lease"}` | `{"ok":true}`; held loads proceed |

Renew every 30 s. As a backstop, the broker polls `127.0.0.1:8792/health`
(`inCall`/`active_calls`), so a call without a lease is still protected and a
crashed calld's lease expires after 90 s.

## Limitations

What works today has been exercised on the 4060 (2026-11-02, side port): yield,
cold start, tool calls, a 90k prompt, call hold and release, profile switch,
eviction for an ollama load, split repair, idle stop. What does not exist yet, or is weak:

- **Ollama-native `/api/chat` and `/api/generate` for a routed model go to stock
  ollama** (64k, q8_0), after the llama-server is stopped. llama-server does not
  speak that protocol, and a translator is not built. OpenHands uses `/v1`, so it
  gets turbo3. A local agent on `/api/chat` does not.
- **Reasoning field name differs.** Ollama's `/v1` returns `message.reasoning`;
  llama-server returns `reasoning_content`. litellm reads the latter. A client
  that hard-codes `reasoning` would see nothing.
- **cvoiced outside calls is not leased.** Batch users (kiri-wake, standin)
  reload cvoiced on their own. If an LLM holds 7.7 GB at that moment, cvoiced's
  load falls back to CPU (slow, still works). Only calls are protected.
- **Cold-start cost of yielding.** cvoiced: 0.8 s process + 8.0 s model load;
  a fully cold `/speak` is 6.0 s, warm 0.4 s. hotline-calld warms it at ring
  time, so a call answered after ~9 s of ringing hears no delay. An inbound call
  auto-answered at 1.5 s may hear several seconds of silence before the first word.
- **One llama-server, one profile at a time.** Two requests for different
  profiles serialize behind a 2–5 s restart.
- **VRAM need for an ollama model never seen loaded** is estimated as weights ×
  1.15 + 512 MiB. Once seen fully loaded, the real size is learned (in memory only).
- **Killing the broker** stops the llama-server child (Pdeathsig, and
  KillMode=control-group), and so drops any in-flight turbo request. That
  includes `ownbox update`, which restarts the unit.
- The q4_0-K + turbo3-V combination deadlocks in prefill on this build, and
  q8_0-K + q4_0-V runs on a slow path. Neither is offered as a profile.
- turbo3 at 262k leaves ~440 MiB spare at peak. Larger batches or `-np > 1`
  will not fit.
