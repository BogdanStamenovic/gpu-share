#!/bin/bash
# Ownbox lifecycle for gpu-share: setup | update | remove.
# Runs in the checkout ($OWNBOX_TOOL_DIR). Optional steps are asked on a TTY;
# without one (agents run ownbox --yes) the safe default is taken and printed.
set -euo pipefail
TOOL_DIR="${OWNBOX_TOOL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$TOOL_DIR"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/gpu-share.service"
CFG_DIR="$HOME/.config/gpu-share"
STATE_DIR="$HOME/.local/state/gpu-share"
DROPIN=/etc/systemd/system/ollama.service.d/40-gpu-share.conf

ask() { # ask "question" default(y|n)
  if [[ -t 0 ]]; then
    local a; read -r -p "$1 [${2}] " a; a="${a:-$2}"; [[ "$a" =~ ^[Yy] ]]
  else
    echo "(no TTY) $1 -> ${2}"; [[ "$2" == y ]]
  fi
}

build() {
  command -v go >/dev/null || { echo "gpu-share: go is required to build (pacman -S go)" >&2; exit 1; }
  # Build to a temp name and rename, so a failed build leaves the running binary alone.
  go build -o bin/gpu-share.new ./cmd/gpu-share
  mv -f bin/gpu-share.new bin/gpu-share
}

render_unit() {
  mkdir -p "$UNIT_DIR"
  sed "s|@TOOL_DIR@|$TOOL_DIR|g" systemd/gpu-share.service.in > "$UNIT"
  systemctl --user daemon-reload
}

ensure_config() {
  mkdir -p "$CFG_DIR" "$STATE_DIR"
  if [[ ! -f "$CFG_DIR/config.json" ]]; then
    # Only overrides live here; `gpu-share config` prints the effective config.
    if [[ -f "$DROPIN" ]]; then echo '{}' > "$CFG_DIR/config.json"
    else printf '{\n  "listen": "127.0.0.1:11437",\n  "ollama": "http://127.0.0.1:11434"\n}\n' > "$CFG_DIR/config.json"; fi
    echo "wrote $CFG_DIR/config.json"
  fi
}

check_deps() {
  local ls; ls=$(bin/gpu-share -config "$CFG_DIR/config.json" config | python3 -c 'import json,sys;print(json.load(sys.stdin)["llama_server"])')
  [[ -x "$ls" ]] || echo "WARNING: $ls not found: routed models fall back to stock ollama until the turbo3 fork is built (~/data/llama-turbo3)."
  command -v nvidia-smi >/dev/null || echo "WARNING: nvidia-smi missing: the broker forwards unmanaged."
}

case "${1:-}" in
  setup)
    build; ensure_config; render_unit; check_deps
    systemctl --user enable gpu-share.service >/dev/null
    systemctl --user restart gpu-share.service
    sleep 1; systemctl --user is-active gpu-share.service
    if [[ ! -f "$DROPIN" ]]; then
      echo "gpu-share runs on a side port (127.0.0.1:11437); ollama still owns :11434."
      if ask "Move ollama to :11435 and put gpu-share on :11434 now (sudo, restarts ollama)?" n; then
        scripts/frontdoor.sh on
      else
        echo "skipped. To switch later: gpu-share-frontdoor on"
      fi
    fi
    ;;
  update)
    build; ensure_config; render_unit; check_deps
    # Config and state are never rewritten here. Unknown config keys are
    # ignored, so a rollback to an older build reads a newer config fine.
    if systemctl --user is-active -q gpu-share.service; then
      systemctl --user restart gpu-share.service; echo "restarted gpu-share"
    fi
    ;;
  remove)
    if [[ -f "$DROPIN" ]]; then
      # Leaving the drop-in would strand ollama on :11435 with nothing on :11434.
      echo "ollama is behind gpu-share; putting it back on :11434."
      scripts/frontdoor.sh off || echo "WARNING: could not revert; run: sudo rm $DROPIN && sudo systemctl daemon-reload && sudo systemctl restart ollama" >&2
    fi
    systemctl --user disable --now gpu-share.service 2>/dev/null || true
    rm -f "$UNIT"; systemctl --user daemon-reload
    if ask "Delete config and logs ($CFG_DIR, $STATE_DIR)?" n; then rm -rf "$CFG_DIR" "$STATE_DIR"; echo deleted; else echo "kept $CFG_DIR and $STATE_DIR"; fi
    ;;
  *) echo "usage: lifecycle.sh setup|update|remove" >&2; exit 2 ;;
esac
