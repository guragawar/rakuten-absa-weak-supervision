#!/usr/bin/env bash
# Follow-ups triggered by the pre-registered rules of docs/model_comparison_plan.md (Part A):
#   1. divergence rule: ModernBERT-ja-130m seed 44 collapsed (valid negative F1 0.41 < 0.50), so all three
#      seeds are re-run with lr 1e-5; the lr is then chosen by the valid mean and both are reported.
#   2. A': several new encoders beat DeBERTa on valid; the best by valid (ModernBERT-ja-310m, 0.7735) gets
#      E7 (2021-24 pool, 1 epoch, same arguments as E7_recent2M) + E8 x 3 seeds.
# Runs are skipped if their output exists.  Usage: WAIT_FOR=<pid> GPU=1 bash tools/run_model_comparison_followup.sh
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

if [[ -n "${WAIT_FOR:-}" ]]; then
  echo "[$(date +%F' '%T)] waiting for pid $WAIT_FOR"
  while kill -0 "$WAIT_FOR" 2>/dev/null; do sleep 60; done
fi

e8() {  # name seed extra-args...
  local name=$1 seed=$2; shift 2
  if [[ -f "results/${name}_metrics.json" ]]; then echo "[skip] $name"; return; fi
  echo "[$(date +%F' '%T)] $name"
  $PY train_absa_baseline.py $FT --seed "$seed" "$@" --output_dir "checkpoints/tmp_$name" \
      --experiment_name "$name" > "$LOG_DIR/$name.log" 2>&1
  rm -rf "checkpoints/tmp_$name"
}

# 1. divergence rule
for s in 42 43 44; do
  e8 "CMP_E1_modernbert_ja_130m_lr1e-5_seed$s" $s --model_name sbintuitions/modernbert-ja-130m --learning_rate 1e-5
done

# 2. A': E7 + E8 on ModernBERT-ja-310m
E7=Aprime_E7_modernbert310m_recent2M
if [[ ! -f "checkpoints/$E7/encoder/model.safetensors" ]]; then
  echo "[$(date +%F' '%T)] $E7"
  $PY train_panel_weak_pretrain.py --model_name sbintuitions/modernbert-ja-310m \
      --train_file data/weak_all/panel_weak_train.jsonl --validation_file data/weak_all/panel_weak_valid.jsonl \
      --stats_file data/weak_all/panel_weak_label_stats_all.json --num_epochs 1 --max_valid_samples 50000 \
      --min_source_year 2021 --pos_weight_mode sqrt --output_dir "checkpoints/$E7" --experiment_name "$E7" \
      > "$LOG_DIR/$E7.log" 2>&1
  rm -rf "checkpoints/$E7"/checkpoint-*
fi
for s in 42 43 44; do
  e8 "Aprime_E8_modernbert310m_seed$s" $s --pretrained_encoder "checkpoints/$E7/encoder"
done
echo "[$(date +%F' '%T)] follow-up done"
