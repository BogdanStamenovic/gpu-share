#!/bin/bash
# frontdoor.sh on|off|status [--dry-run]
#
# "on":  ollama moves to 127.0.0.1:11435 (a system drop-in, needs sudo) and the
#        gpu-share broker takes 0.0.0.0:11434, so every existing client (OpenHands,
#        hotline-calld, agents) goes through it without changing anything.
# "off": the reverse. The broker stays up on a side port (127.0.0.1:11437) in
#        front of ollama's :11434, harmless and testable.
# Ollama is unreachable for a few seconds while it restarts either way.
set -euo pipefail
DROPIN_SRC="$(cd "$(dirname "$0")/.." && pwd)/systemd/ollama-40-gpu-share.conf"
DROPIN=/etc/systemd/system/ollama.service.d/40-gpu-share.conf
CFG="$HOME/.config/gpu-share/config.json"
DRY=0; [[ "${2:-}" == "--dry-run" ]] && DRY=1
run() { if ((DRY)); then echo "would: $*"; else "$@"; fi; }

set_mode() { # front|side
  ((DRY)) && { echo "would: set $CFG to $1 mode"; return; }
  mkdir -p "$(dirname "$CFG")"; [[ -f "$CFG" ]] || echo '{}' > "$CFG"
  python3 - "$CFG" "$1" <<'PY'
import json, sys
p, mode = sys.argv[1], sys.argv[2]
c = json.load(open(p))
if mode == "front":
    c.pop("listen", None); c.pop("ollama", None)   # defaults: 0.0.0.0:11434 -> 127.0.0.1:11435
else:
    c["listen"], c["ollama"] = "127.0.0.1:11437", "http://127.0.0.1:11434"
json.dump(c, open(p, "w"), indent=2); print(f"{p}: {mode} mode")
PY
}

wait_port() { for _ in $(seq 60); do curl -sf "http://$1/api/version" >/dev/null && return 0; sleep 0.5; done; echo "nothing answering on $1" >&2; return 1; }

case "${1:-status}" in
  on)
    run sudo install -m 0644 "$DROPIN_SRC" "$DROPIN"
    run sudo systemctl daemon-reload
    run sudo systemctl restart ollama
    ((DRY)) || wait_port 127.0.0.1:11435
    set_mode front
    run systemctl --user restart gpu-share.service
    ((DRY)) || { wait_port 127.0.0.1:11434; echo "front door on: :11434 -> gpu-share -> ollama :11435"; }
    ;;
  off)
    set_mode side
    run systemctl --user restart gpu-share.service
    run sudo rm -f "$DROPIN"
    run sudo systemctl daemon-reload
    run sudo systemctl restart ollama
    ((DRY)) || { wait_port 127.0.0.1:11434; echo "front door off: ollama back on :11434, broker on 127.0.0.1:11437"; }
    ;;
  status)
    [[ -f "$DROPIN" ]] && echo "ollama drop-in: present ($DROPIN)" || echo "ollama drop-in: absent"
    ss -ltnp 2>/dev/null | awk '$4 ~ /:(11434|11435|11436|11437)$/ {print "listening:", $4, $NF}'
    ;;
  *) echo "usage: frontdoor.sh on|off|status [--dry-run]" >&2; exit 2 ;;
esac
