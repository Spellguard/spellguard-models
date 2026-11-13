#!/usr/bin/env bash
# Reproduce the published results end to end: rebuild the test sets, download the model from Hugging Face,
# serve it with llama-server, score every item and compare the metrics with ours.
#
#   LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server ./run.sh [spellguard|qwen3guard]
#
# Needs uv, a llama.cpp build (see README.md for the commit and build flags) and, for a full run in a few
# hours, a GPU with 12 GB or more. WORK (default ./work) holds the data, the model and the scores.
set -euo pipefail

MODEL="${1:-spellguard}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/work}"
PORT="${PORT:-8080}"
LLAMA_SERVER="${LLAMA_SERVER:?set LLAMA_SERVER to your llama.cpp llama-server binary}"
mkdir -p "$WORK"

case "$MODEL" in
  spellguard)
    REPO=Spellguard/Spellguard-1.0-8B-GGUF
    REVISION=8e6d60ab6858e25c9d70815f93527a5a3f3c1e12
    FILE=Spellguard-1.0-8B-Q4_K_M.gguf
    SHA256=6c9d37c0877e04f1b1e21405d1fec817d7e36458a7476bb9358024743a090971
    KEY=spellguard_1.0_8b; NAME=spellguard-1.0-8b
    GGUF="$WORK/models/$FILE"
    if [ ! -f "$GGUF" ]; then
      uvx --from 'huggingface_hub>=0.34' hf download "$REPO" "$FILE" --revision "$REVISION" --local-dir "$WORK/models"
    fi
    ;;
  qwen3guard)
    # The baseline: a Q4_K_M we made from Qwen/Qwen3Guard-Gen-8B (see make_baseline_gguf.sh). A rebuild on
    # another machine can differ from ours in a few bytes; set GGUF to use your own file.
    SHA256=91e221f38006762715f0ef69043d54452ed107fc31aafe10f8482b04759fe6e8
    KEY=qwen3guard_gen_8b; NAME=qwen3guard-gen-8b
    GGUF="${GGUF:-$WORK/models/Qwen3Guard-Gen-8B-Q4_K_M.gguf}"
    [ -f "$GGUF" ] || { echo "no $GGUF: run make_baseline_gguf.sh first" >&2; exit 1; }
    ;;
  *) echo "usage: run.sh [spellguard|qwen3guard]" >&2; exit 2 ;;
esac

got=$(sha256sum "$GGUF" | cut -d' ' -f1)
if [ "$got" != "$SHA256" ]; then
  echo "note: $GGUF has sha256 $got, not the file the published results used ($SHA256)" >&2
fi

# 1. The test sets, rebuilt from their public sources and checked against the published items.
[ -d "$WORK/data/phtest" ] || uv run --script "$HERE/fetch.py" all "$WORK/data"
uv run --script "$HERE/items.py" "$WORK/data" "$WORK/items.jsonl"  # exits 1 if any set differs

# 2. The server, with the arguments the published results used.
"$LLAMA_SERVER" -m "$GGUF" --host 127.0.0.1 --port "$PORT" \
  -ngl 99 -c 16384 -np 16 -kvu -fa on -b 2048 -ub 512 > "$WORK/llama-server-$NAME.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  kill -0 $SERVER_PID 2>/dev/null || { echo "llama-server exited; see $WORK/llama-server-$NAME.log" >&2; exit 1; }
  sleep 2
done

# 3. Every item. Rerunning resumes where an interrupted run stopped.
mkdir -p "$WORK/scores"
uv run --script "$HERE/score.py" --items "$WORK/items.jsonl" --gguf "$GGUF" \
  --server "http://127.0.0.1:$PORT" --out "$WORK/scores/$NAME.jsonl" --concurrency 16

# 4. The metrics, against the published headlines and the published per-item scores.
uv run --script "$HERE/metrics.py" --items "$WORK/items.jsonl" --scores "$WORK/scores/$NAME.jsonl" \
  --published "$HERE/../results/spellguard-1.0-8b.json" --model "$KEY" \
  --reference "$HERE/../results/scores/$NAME.jsonl.gz" --json "$WORK/metrics-$NAME.json"
