# LOG

Everything tried, in order, including dead ends. Times CET, 2026-11-02.

## Task
Bogdan (Discord 1566834402431074494, provenance verified): "check out why is token
generation so slow. Cvoice should unload whenever a model wants to load. And also i
need the turbo3 cuda fork used for models to get more context". Relayed by hotline-80.

## 16:20 Survey
- nvidia-smi: cvoiced 2408 MiB + ollama llama-server 4318 MiB on an 8188 MiB card.
- Root disk 1.5 GB free (the seed said ~12G); /mnt/offload 100% full (1.3 GB). Bulky
  data goes to /mnt/windows/ml/gpu-share only.
- The "turbo3 cuda fork" already exists: ~/data/llama-turbo3 = Madreag/turbo3-cuda
  (from TheTom/llama-cpp-turboquant), forked to BogdanStamenovic/turbo3-cuda, built
  2026-08-29 with a CUDA 13 include fix. Its docs/RTX4060-8GB-262K.md: piccolo @262144
  fits only with turbo3 (1600 MiB KV); f16/q8_0 OOM. No speed or quality numbers.
  The binary still links against CUDA 13.4 (ldd clean).
- Ollama endpoint usage, 7 days of journal: OpenHands (100.103.46.118) uses
  /v1/chat/completions, /api/show, /v1/models, /api/tags. ~1000 /api/embed from ::1,
  which is hotline-calld speculate.py (bge-m3), per manager-6.
- Passive tok/s from ollama's own print_timing, 14 days (data/ollama-timing-14d.log):
  piccolo @65536 at 33/33 layers ~40 tok/s, 32/33 36.5, 25/33 8.3, 18/33 11.5.

## 16:23-16:59 GPU window (granted by hotline-80, cvoiced stopped by me)
Controlled runs, bench/ollama_tps.py, temp 0, seed 1, raw prompts. data/bench-ollama.jsonl.
- With cvoiced resident: 18/33 layers, decode 14.3 (short) / 10.5 (14859-token prompt), prefill 1071.
- cvoiced stopped: 33/33, decode 43.9 / 41.3, prefill 1885. Ollama alone at 64k q8_0 = 6454 MiB,
  so piccolo and cvoiced (2.4 GB) cannot both fit even at a perfect fit.
- llama-bench on the fork (data/llama-bench.jsonl), pp512/tg128, depth 0 and 16384:
  f16 44.2/40.9, q8_0 43.9/41.1, turbo3 43.8/39.1 tok/s decode. Prefill 2027-2078 at depth 0.
- Dead end: the first capacity loop and the first KLD loop ran under zsh, where
  `set -- $cfg` and `./kld.sh $c` do not word-split. Nothing ran. Rewritten as bash scripts.
- Capacity (data/capacity.txt): q8_0 128k fits (7656 MiB), 160k OOM; q4_0 192k fits (7332),
  262k OOM; turbo4 262k OOM; q8_0-K+turbo3-V 262k OOM; q4_0-K+turbo3-V 262k fits (7742);
  turbo3 262k fits (7390).
- KLD vs f16 KV, the fork's quality-tests/kl_divergence.py (top-10 logprobs, next token),
  60 wikitext-2 test prompts of ~2048 tokens, -c 4096:
  f16 rerun 0.0000 / 100% top-1 (deterministic, so the differences are real),
  q8_0 0.0025 / 95%, q4_0 0.0090 / 85%, q8K-turbo3V 0.0171 / 85%, turbo4 0.0213 / 85%,
  turbo3 0.0246 / 80%.
- Dead end: q4_0-K + turbo3-V deadlocks mid-prefill (GPU 0% for 10 min on prompt 149).
  Killed by PID. q8_0-K + q4_0-V runs but slowly (probably no fused FA kernel for that pair),
  so I abandoned it to keep within the window.
- cvoiced cold start (data/cvoice-load.json, data/cvoice-speak.txt): process up 0.84 s,
  /load 8.0 s, first /speak after load 3.74 s, warm /speak 0.40 s; fully cold /speak with
  no /load 5.99 s.
- cvoiced restarted at 16:59 and reported to hotline-80.

## 17:00 Design agreed
- With manager-6 (hotline-call): the call outranks the LLM. Reserve ~5.0 GB (Whisper 1.9 +
  cvoiced 2.4 + bge-m3 0.66) from ring to hang-up. Lease: begin/renew/end, 30 s renewal;
  begin must return within ~1 s and evict in the background. Backstop: poll 8792/health.
- hotline-80: the broker plan is approved. Putting cvoice on the ownbox shelf is required.
  The KV default is Bogdan's call (put to him). Moving ollama to :11435 comes back as an ask with the diff.

## 17:00-17:20 Broker build
- Chose Go, stdlib only: a long-running streaming proxy with no venv (the disks were
  reported full at the time) and one binary for ownbox to build.
- Planner is a pure function (internal/broker/plan.go) with 8 unit tests, so the
  policy is testable without a GPU and `gpu-share plan` can show it without acting.
- Dead end: the zsh word-split trap a third time (`$G status` with G holding a command
  plus flags). Ran the commands literally.
- Delegated cvoice /yield + ownbox shelving to an Opus subagent (reports to me).
  Result: cvoice a9b5f5d (/yield), 6ed3f18 (lifecycle). Empty cvoiced = 0 MiB.
  /yield: 2408 MiB released, back in 6.8 s. It found and fixed: the old remove deleting
  profiles and the token unasked, and an unattended install defaulting to the client role.

## 17:09-17:14 GPU window 2 (granted by hotline-80): end to end on a side port 127.0.0.1:11437
- piccolo /v1 cold: cvoiced restarted (/yield not deployed yet) -> turbo3 @262k ready
  5.0 s -> 42.3 tok/s decode, 7384 MiB. 15.1 s wall including the cold start.
- Tool call through llama-server --jinja: correct get_weather call.
- 89,845-token prompt: prefill 1444 tok/s, decode 30.3 tok/s, correct answer, peak
  7746 MiB, so the profile MiB was raised from 7400 to 7750.
- call begin 46 ms -> turbo evicted within 1 s -> a piccolo request held for 10 s -> call end
  -> loaded 3.1 s, served.
- Profile switch to q8_0 @128k: 2.1 s, 42.5 tok/s, 7650 MiB.
- An ollama model load evicted the idle llama-server.
- Split repair: piccolo loaded straight into ollama beside cvoiced (4299/7126 MiB on GPU).
  After cvoiced left, the reconciler evicted it within 5 s.
- Idle stop after keep_alive (15 s test).
- Added: fallback to stock ollama if llama-server cannot start.

## 17:16 ownbox
- `ownbox install gpu-share --yes`: built, unit installed and started on the side port
  (no TTY, so the front-door step was skipped and printed). `ownbox update --yes` on a
  populated config: kept, restarted. `ownbox uninstall --yes`: unit gone, config kept;
  reinstalled after.
- Found: ownbox does not export OWNBOX_TOOL_DIR to lifecycle commands; lifecycle.sh
  falls back to its own location.
- Asked hotline-80 for the :11434 front-door move with the exact diff.

## 17:18 Front door on (approved by hotline-80, with conditions)
- Dry run saved: data/frontdoor-dryrun.txt. 8792/health just before: inCall False, active_calls 0.
- `gpu-share-frontdoor on` 17:18:24 -> 17:18:27 (~3 s with no ollama). The broker owns *:11434;
  ollama is on 127.0.0.1:11435 via /etc/systemd/system/ollama.service.d/40-gpu-share.conf.
- Checked from the tailnet address 100.72.2.62:11434: /api/tags and /v1/models match the
  before lists (data/frontdoor-*-*.txt), plus the alias piccolo-gorgone:9b-q8. bge-m3 /api/embed:
  2.16 s cold (ollama had just restarted), then 9.6 ms on ::1 and 6.5 ms on 127.0.0.1. piccolo
  /v1/chat/completions: correct, 42.3 tok/s via turbo3 (the start evicted idle bge-m3; ready 2.3 s).
- Device map: claimed archserver/gpu-share (unit gpu-share.service), verified.
- Asked manager-6 for a round trip through hotline-calld's own client on :11434.
