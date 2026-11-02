"""Peak VRAM of a llama-server profile on long prompts: does it fit the card?

The turbo3 262k profile aborted in flash attention under long prompts
(2026-11-03: ggml_cuda_pool_vmm::alloc <- launch_fattn, six times in 13 min).
This starts llama-server standalone with the production arguments plus a
variant, sends prompts of a target token count, samples nvidia-smi every
100 ms, and reports peak VRAM, prompt speed and whether the server survived.

    python3 bench/fit.py OUT.json VARIANT [VARIANT...]
    VARIANT = name=extra args, e.g. 'ub256=-ub 256' or 'ctx196k=-c 196608'

Run only with the GPU otherwise idle (gpu-share has nothing resident).
"""

import json
import shlex
import subprocess
import sys
import threading
import time
import urllib.request

BIN = "/home/bodas/data/llama-turbo3/build/bin/llama-server"
MODEL = ("/mnt/windows/Users/Korisnik/.ollama/models/blobs/"
         "sha256-18b2ed082f5d418ba40c4b4656bc9a0430f8faac98999dcc028856815dcb38bd")
BASE = ["-m", MODEL, "-c", "262144", "-ngl", "99", "-fa", "on", "-ctk", "turbo3",
        "-ctv", "turbo3", "-np", "1", "--host", "127.0.0.1", "--port", "11499", "--jinja"]
URL = "http://127.0.0.1:11499"
TARGETS = (22000, 90000)
PARA = ("Ovo je dugačak tekst za merenje memorije grafičke kartice pri obradi velikih "
        "upita. Svaka rečenica dodaje nekoliko tokena, a cilj je da upit dostigne "
        "zadatu dužinu bez ponavljanja istog niza na granici blokova. ")


def used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip())


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def prompt_of(tokens: int) -> str:
    n = post("/tokenize", {"content": PARA})["tokens"]
    reps = max(1, tokens // len(n))
    return "".join(f"[{i}] {PARA}" for i in range(reps))


def run(name: str, extra: list[str]) -> dict:
    cmd = [BIN] + BASE + extra
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    while time.time() - t0 < 240:
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            break
        except Exception:
            if proc.poll() is not None:
                return {"variant": name, "error": "exited during load", "stderr": proc.stderr.read()[-800:]}
            time.sleep(0.5)
    out = {"variant": name, "extra": extra, "loaded_mib": used_mib(), "prompts": []}
    for target in TARGETS:
        peak, stop = [0], threading.Event()

        def sample():
            while not stop.is_set():
                peak[0] = max(peak[0], used_mib())
                time.sleep(0.1)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        row = {"target_tokens": target}
        try:
            text = prompt_of(target)
            t = time.time()
            r = post("/completion", {"prompt": text, "n_predict": 4, "cache_prompt": False})
            row.update(ok=True, prompt_tokens=r.get("timings", {}).get("prompt_n"),
                       prompt_per_s=round(r.get("timings", {}).get("prompt_per_second", 0), 1),
                       wall_s=round(time.time() - t, 1))
        except Exception as exc:
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                       server_alive=proc.poll() is None)
        stop.set()
        th.join()
        row["peak_mib"] = peak[0]
        out["prompts"].append(row)
        print(json.dumps({"variant": name, **row}), flush=True)
        if proc.poll() is not None:
            out["crashed"] = True
            out["stderr_tail"] = proc.stderr.read()[-1200:]
            break
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(3)
    return out


def agentic(name: str, extra: list[str], turns: int = 60, start: int = 18000, step: int = 400) -> dict:
    """piccolo-bench's pattern: one conversation, many turns, prompt cache on.

    Hermes drove 40-100 tool-call turns per task, each a chat completion on the
    growing conversation, so the server reuses its cache and keeps context
    checkpoints ("32/32 context checkpoints" before the 2026-11-03 aborts). A
    single long prompt (`run`) did not reproduce the crash; this is the shape
    that did, in production."""
    proc = subprocess.Popen([BIN] + BASE + extra, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    while time.time() - t0 < 240:
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            break
        except Exception:
            if proc.poll() is not None:
                return {"variant": name, "error": "exited during load"}
            time.sleep(0.5)
    base_text = prompt_of(start)
    msgs = [{"role": "system", "content": base_text}, {"role": "user", "content": "Počni."}]
    out = {"variant": name, "extra": extra, "mode": "agentic", "turns": []}
    peak_all = 0
    for i in range(turns):
        peak, stop = [0], threading.Event()

        def sample():
            while not stop.is_set():
                peak[0] = max(peak[0], used_mib())
                time.sleep(0.1)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        row = {"turn": i + 1}
        try:
            r = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 48,
                                              "chat_template_kwargs": {"enable_thinking": False}})
            reply = r["choices"][0]["message"].get("content") or "ok"
            row.update(ok=True, prompt_tokens=r.get("usage", {}).get("prompt_tokens"))
            msgs.append({"role": "assistant", "content": reply})
            msgs.append({"role": "user", "content": f"Rezultat alata {i}: " + PARA * max(1, step // 40)})
        except Exception as exc:
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        stop.set(); th.join()
        row["peak_mib"] = peak[0]
        peak_all = max(peak_all, peak[0])
        out["turns"].append(row)
        if i % 10 == 0 or not row["ok"]:
            print(json.dumps({"variant": name, **row}), flush=True)
        if proc.poll() is not None or not row["ok"]:
            out["crashed"] = proc.poll() is not None
            out["stderr_tail"] = (proc.stderr.read() if proc.poll() is not None else "")[-1500:]
            break
    out["peak_mib"] = peak_all
    print(json.dumps({"variant": name, "summary_peak_mib": peak_all, "turns_done": len(out["turns"]),
                      "crashed": out.get("crashed", False)}), flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(3)
    return out


if __name__ == "__main__":
    if sys.argv[2] == "--agentic":
        results = []
        for spec in sys.argv[3:]:
            name, _, args = spec.partition("=")
            results.append(agentic(name, shlex.split(args)))
        json.dump(results, open(sys.argv[1], "w"), indent=1)
        sys.exit(0)
    results = []
    for spec in sys.argv[2:]:
        name, _, args = spec.partition("=")
        results.append(run(name, shlex.split(args)))
    json.dump(results, open(sys.argv[1], "w"), indent=1)
