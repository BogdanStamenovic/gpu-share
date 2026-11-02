#!/bin/bash
cd /mnt/windows/ml/gpu-share; mkdir -p cap
B=/mnt/windows/Users/Korisnik/.ollama/models/blobs/sha256-18b2ed082f5d418ba40c4b4656bc9a0430f8faac98999dcc028856815dcb38bd
for cfg in "$@"; do set -- $cfg; L=cap/$1-$2-$3.log
  ~/data/llama-turbo3/build/bin/llama-server -m $B -ngl 99 -fa on -ctk $1 -ctv $2 -c $3 -np 1 --port 8091 --host 127.0.0.1 > $L 2>&1 & P=$!
  ok=no; for i in $(seq 120); do curl -sf 127.0.0.1:8091/health >/dev/null && { ok=yes; break; }; kill -0 $P 2>/dev/null || break; sleep 1; done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  kv=$(grep -oE 'KV buffer size = +[0-9.]+ MiB' $L | awk '{s+=$(NF-1)} END{print s}')
  echo "K=$1 V=$2 ctx=$3 up=$ok kv=${kv}MiB vram_used=${used}MiB $(grep -m1 -oE 'cudaMalloc failed.*|failed to allocate.*' $L)"
  kill $P 2>/dev/null; wait $P 2>/dev/null; sleep 1
done
