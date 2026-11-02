#!/bin/bash
# usage: kld.sh K V LABEL
cd /mnt/windows/ml/gpu-share
B=/mnt/windows/Users/Korisnik/.ollama/models/blobs/sha256-18b2ed082f5d418ba40c4b4656bc9a0430f8faac98999dcc028856815dcb38bd
~/data/llama-turbo3/build/bin/llama-server -m $B -ngl 99 -fa on -ctk $1 -ctv $2 -c 4096 -np 1 --port 8091 --host 127.0.0.1 > kld/server-$3.log 2>&1 & P=$!
for i in $(seq 90); do curl -sf 127.0.0.1:8091/health >/dev/null && break; sleep 1; done
UV_CACHE_DIR=/mnt/windows/ml/gpu-share/uv-cache uv run -q --with requests python ~/data/llama-turbo3/quality-tests/kl_divergence.py --port 8091 --host 127.0.0.1 --type $3 --wiki wikitext-2-raw/wiki.test.raw --n-prompts 60 --prompt-tokens 2048 --output-dir kld 2>&1 | grep -E 'KL|Top-1|Delta'
kill $P; wait $P 2>/dev/null
