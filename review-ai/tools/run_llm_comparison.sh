#!/usr/bin/env bash
# Part B of docs/model_comparison_plan.md: the four LLM settings on the full test set.
# Skips settings whose metrics file exists. WAIT_FOR=<pid> waits for another GPU job first.
#
# Usage:  GPU=1 bash tools/run_llm_comparison.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-1}"
PY=.venv/bin/python
LOG_DIR=logs/model_comparison
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false
export HF_HOME="$(cd .. && pwd)/hf_cache/home"  # everything else Hugging Face writes (xet download logs, ...)
export HF_HUB_CACHE="$(cd .. && pwd)/hf_cache"  # project-local model cache; never write outside gujincheng/

if [[ -n "${WAIT_FOR:-}" ]]; then
  echo "[$(date +%F' '%T)] waiting for pid $WAIT_FOR"
  while kill -0 "$WAIT_FOR" 2>/dev/null; do sleep 60; done
fi

# Qwen2.5-7B is already in the shared cache (someone else's download): read it in place, read-only,
# via its snapshot directory (revision a09a35458c702b33eeacc393d103063234e8bc28). Nothing is written there.
QWEN="/home/etl/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
SWALLOW="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3 --revision 0858f50ecfbc61eab25f353013e0be542e1312ac"

free_gb() { df -BG --output=avail "$HF_HUB_CACHE" | tail -1 | tr -dc '0-9'; }
has_space() {  # need_gb: model size; keeps a 25 GB margin for other users of this disk
  local need=$(( $1 + 25 )) free; free=$(free_gb)
  if (( free < need )); then echo "[$(date +%F' '%T)] SKIP: ${free}G free < ${need}G needed"; return 1; fi
}
drop_model() {  # delete a model we downloaded, only inside the project cache
  local dir="$HF_HUB_CACHE/models--${1//\//--}"
  [[ "$dir" == */gujincheng/hf_cache/models--* && -d "$dir" ]] && rm -rf "$dir" "$HF_HUB_CACHE/.locks/models--${1//\//--}" \
    && echo "[$(date +%F' '%T)] removed $1 from project cache ($(free_gb)G free)"
  return 0  # nothing to remove is not an error
}

run() {  # name model-and-revision mode
  local name=$1 model=$2 mode=$3
  if [[ -f "results/${name}_metrics.json" ]]; then echo "[skip] $name"; return; fi
  echo "[$(date +%F' '%T)] $name"
  $PY tools/llm_absa_eval.py --model $model --mode "$mode" --split test --name "$name" > "$LOG_DIR/$name.log" 2>&1
}

run LLM_qwen25_7b_zero   "$QWEN"    zero
run LLM_qwen25_7b_few    "$QWEN"    few
run LLM_swallow8b_zero   "$SWALLOW" zero
run LLM_swallow8b_few    "$SWALLOW" few
drop_model tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3
echo "[$(date +%F' '%T)] llm comparison done"
