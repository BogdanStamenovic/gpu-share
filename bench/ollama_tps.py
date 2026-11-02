#!/usr/bin/env python3
"""Decode/prefill tok/s for one model through an ollama-compatible /api/generate.

Uses ollama's own eval_count/eval_duration counters, so the numbers are what the
server measured, not wall clock. Fixed prompts and seed so runs are comparable.
"""
import json, sys, time, urllib.request, argparse

WIKI = "/mnt/windows/ml/gpu-share/wikitext-2-raw/wiki.test.raw"

def gen(base, model, prompt, n, num_ctx):
    body = {"model": model, "prompt": prompt, "stream": False, "raw": True,
            "options": {"num_predict": n, "temperature": 0, "seed": 1, "num_ctx": num_ctx}}
    req = urllib.request.Request(base + "/api/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    r["_wall"] = time.time() - t
    return r

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:11434")
ap.add_argument("--model", default="piccolo-gorgone:9b")
ap.add_argument("--ctx", type=int, default=65536)
ap.add_argument("--label", required=True)
ap.add_argument("--long-chars", type=int, default=60000)
a = ap.parse_args()
text = open(WIKI, encoding="utf-8").read()
cases = [("short", "The history of the Roman Empire", 256),
         ("long", text[:a.long_chars] + "\n\nSummarize the above:", 128)]
gen(a.base, a.model, "hi", 1, a.ctx)  # load, not measured
for name, p, n in cases:
    r = gen(a.base, a.model, p, n, a.ctx)
    out = {"label": a.label, "case": name,
           "prompt_tokens": r.get("prompt_eval_count"),
           "prefill_tps": r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9) if r.get("prompt_eval_duration") else None,
           "gen_tokens": r.get("eval_count"),
           "decode_tps": r["eval_count"] / (r["eval_duration"] / 1e9),
           "wall_s": round(r["_wall"], 2)}
    print(json.dumps(out), flush=True)
