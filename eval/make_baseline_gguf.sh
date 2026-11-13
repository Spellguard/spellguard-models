#!/usr/bin/env bash
# Build the baseline's Q4_K_M GGUF from Qwen/Qwen3Guard-Gen-8B, the way ours was built: llama.cpp's converter
# to BF16, then llama-quantize to Q4_K_M.
#
#   LLAMA_CPP=/path/to/llama.cpp ./make_baseline_gguf.sh
#
# LLAMA_CPP is a llama.cpp checkout at the commit in README.md, built (it needs build/bin/llama-quantize).
# Our file's sha256 is 91e221f38006762715f0ef69043d54452ed107fc31aafe10f8482b04759fe6e8. The quantiser's
# arithmetic can differ between CPU architectures, so a file built elsewhere may differ from ours in a small
# share of bytes; its scores then differ from ours in the last digits, and a few decisions near 0.5 can flip.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/work}"
LLAMA_CPP="${LLAMA_CPP:?set LLAMA_CPP to a built llama.cpp checkout}"
REVISION=4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb
mkdir -p "$WORK/models"

uvx --from 'huggingface_hub>=0.34' hf download Qwen/Qwen3Guard-Gen-8B --revision "$REVISION" \
  --local-dir "$WORK/models/Qwen3Guard-Gen-8B"
uv run --no-project --with-requirements "$LLAMA_CPP/requirements/requirements-convert_hf_to_gguf.txt" \
  python "$LLAMA_CPP/convert_hf_to_gguf.py" "$WORK/models/Qwen3Guard-Gen-8B" --outtype bf16 \
  --outfile "$WORK/models/Qwen3Guard-Gen-8B-BF16.gguf"
"$LLAMA_CPP/build/bin/llama-quantize" "$WORK/models/Qwen3Guard-Gen-8B-BF16.gguf" \
  "$WORK/models/Qwen3Guard-Gen-8B-Q4_K_M.gguf" Q4_K_M
sha256sum "$WORK/models/Qwen3Guard-Gen-8B-Q4_K_M.gguf"
