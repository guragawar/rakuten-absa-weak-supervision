#!/usr/bin/env bash
# Part A of docs/model_comparison_plan.md: fine-tune other encoders with exactly the E1 settings.
# Runs are skipped if results/<name>_metrics.json exists; checkpoints are deleted after each run
# (metrics + test npz are kept).
#
# Usage:  GPU=1 bash tools/run_model_comparison.sh [model_key ...]   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-1}"
PY=.venv/bin/python
LOG_DIR=logs/model_comparison
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false
export HF_HOME="$(cd .. && pwd)/hf_cache/home"  # everything else Hugging Face writes (xet download logs, ...)
export HF_HUB_CACHE="$(cd .. && pwd)/hf_cache"  # project-local model cache; never write outside gujincheng/
export HF_DATASETS_CACHE="$(cd .. && pwd)/hf_cache/datasets"  # datasets cache holds licensed text: keep it inside gujincheng/
FT="--loss_type asl --asl_gamma_neg 4.0 --asl_clip 0.05 --no_pos_weight --tune_threshold"

declare -A MODEL=(
  [xlmr_base]=xlm-roberta-base
  [deberta_ja]=ku-nlp/deberta-v3-base-japanese
  [luke_ja]=studio-ousia/luke-japanese-base-lite
  [modernbert_ja_130m]=sbintuitions/modernbert-ja-130m
  [modernbert_ja_310m]=sbintuitions/modernbert-ja-310m
  [bert_large_ja]=tohoku-nlp/bert-large-japanese-v2
  [xlmr_large]=xlm-roberta-large
)
ORDER=(xlmr_base deberta_ja luke_ja modernbert_ja_130m modernbert_ja_310m bert_large_ja xlmr_large)

run() {  # key seed [extra args]
  local key=$1 seed=$2; shift 2
  local name="CMP_E1_${key}_seed${seed}${LR_TAG:-}"
  if [[ -f "results/${name}_metrics.json" ]]; then echo "[skip] $name"; return; fi
  echo "[$(date +%F' '%T)] $name (${MODEL[$key]})"
  $PY train_absa_baseline.py $FT --model_name "${MODEL[$key]}" --seed "$seed" "$@" \
      --output_dir "checkpoints/tmp_$name" --experiment_name "$name" > "$LOG_DIR/$name.log" 2>&1
  rm -rf "checkpoints/tmp_$name"
}

keys=("${@:-${ORDER[@]}}")
for key in ${keys[@]}; do
  for seed in 42 43 44; do run "$key" "$seed"; done
done
echo "[$(date +%F' '%T)] model comparison done"
