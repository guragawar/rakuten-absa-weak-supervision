#!/usr/bin/env bash
# Run the pre-registered ablations A0/A5/A6/A7 (docs/ablation_plan_A5_A7.md) in sequence.
# Finished work is skipped, so the queue can be re-run after an interruption: an ablation whose
# E8 results all exist is skipped entirely (its E7 encoder may have been deleted), otherwise
# each E7/E8 run is skipped if its output exists. E8 checkpoints are deleted after the run
# (npz + metrics are kept); E7 keeps only encoder/.
#
# Usage:  GPU=1 bash tools/run_ablation_queue.sh  [steps...]   (default: A0 A5 A6 A7; A5b on request)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-1}"
PY=.venv/bin/python
LOG_DIR=logs/ablation
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false
export HF_HOME="$(cd .. && pwd)/hf_cache/home"  # everything else Hugging Face writes (xet download logs, ...)
export HF_HUB_CACHE="$(cd .. && pwd)/hf_cache"  # project-local model cache; never write outside gujincheng/
export HF_DATASETS_CACHE="$(cd .. && pwd)/hf_cache/datasets"  # datasets cache holds licensed text: keep it inside gujincheng/

all_done() {  # true if every listed E8 run already has its metrics file
  local n
  for n in "$@"; do [[ -f "results/${n}_metrics.json" ]] || return 1; done
}

e7() {  # name, extra args...
  local name=$1; shift
  if [[ -f "checkpoints/$name/encoder/model.safetensors" ]]; then echo "[skip] E7 $name"; return; fi
  echo "[$(date +%F' '%T)] E7 $name"
  $PY train_panel_weak_pretrain.py --pos_weight_mode sqrt --output_dir "checkpoints/$name" \
      --experiment_name "$name" "$@" > "$LOG_DIR/$name.log" 2>&1
  rm -rf "checkpoints/$name"/checkpoint-*
}

e8() {  # name, encoder_dir, seed
  local name=$1 enc=$2 seed=$3
  if [[ -f "results/${name}_metrics.json" ]]; then echo "[skip] E8 $name"; return; fi
  echo "[$(date +%F' '%T)] E8 $name (encoder=$enc seed=$seed)"
  $PY train_absa_baseline.py --train_file data/absa_train.csv --validation_file data/absa_valid.csv \
      --test_file data/absa_test.csv --labels_file data/absa_label_columns.json \
      --loss_type asl --asl_gamma_neg 4.0 --asl_clip 0.05 --no_pos_weight --tune_threshold \
      --pretrained_encoder "$enc" --seed "$seed" \
      --output_dir "checkpoints/tmp_$name" --experiment_name "$name" > "$LOG_DIR/$name.log" 2>&1
  rm -rf "checkpoints/tmp_$name"
}

A0() {
  for s in 42 43 44; do e8 "A0_ref_E8_seed$s" checkpoints/absa_weak_pretrain_v1/encoder $s; done
}
A5() {
  if all_done A5_E8_meal_split_seed{42,43,44}; then echo "[skip] A5 (all E8 results exist)"; return; fi
  local d=data/ablation/A5_meal_split
  e7 A5_E7_meal_split --heads 1,2,3b,3d,4,5,6 --train_file $d/panel_weak_train.jsonl \
      --validation_file $d/panel_weak_valid.jsonl --stats_file $d/panel_weak_label_stats.json
  for s in 42 43 44; do e8 "A5_E8_meal_split_seed$s" checkpoints/A5_E7_meal_split/encoder $s; done
}
A6() {
  if all_done A6_E8_noise_filter_seed{42,43,44}; then echo "[skip] A6 (all E8 results exist)"; return; fi
  local d=data/ablation/A6_noise_filter
  e7 A6_E7_noise_filter --train_file $d/panel_weak_train.jsonl \
      --validation_file $d/panel_weak_valid.jsonl --stats_file $d/panel_weak_label_stats.json
  for s in 42 43 44; do e8 "A6_E8_noise_filter_seed$s" checkpoints/A6_E7_noise_filter/encoder $s; done
}
A7() {
  for es in 43 44; do
    if all_done A7_E7s${es}_E8_seed{42,43,44}; then echo "[skip] A7 E7 seed $es (all E8 results exist)"; continue; fi
    e7 "A7_E7_seed$es" --seed $es
    for s in 42 43 44; do e8 "A7_E7s${es}_E8_seed$s" "checkpoints/A7_E7_seed$es/encoder" $s; done
  done
}

A5b() {  # post-hoc, pre-registered in docs/ablation_plan_A5_A7.md: meal split on the 2021-24 pool, 5 seeds
  local seeds=(42 43 44 45 46)
  for s in "${seeds[@]}"; do e8 "A5b_ref_E8_seed$s" checkpoints/E7_recent2M/encoder $s; done
  if all_done $(printf "A5b_E8_meal_split_seed%s " "${seeds[@]}"); then echo "[skip] A5b (all E8 results exist)"; return; fi
  local d=data/ablation/A5b_meal_split_all
  [[ -f $d/panel_weak_label_stats.json ]] || { echo "missing $d (run prepare_ablation_weak_labels.py first)"; exit 1; }
  e7 A5b_E7_meal_split_recent --heads 1,2,3b,3d,4,5,6 --train_file $d/panel_weak_train.jsonl \
      --validation_file $d/panel_weak_valid.jsonl --stats_file $d/panel_weak_label_stats.json \
      --num_epochs 1 --max_valid_samples 50000 --min_source_year 2021
  for s in "${seeds[@]}"; do e8 "A5b_E8_meal_split_seed$s" checkpoints/A5b_E7_meal_split_recent/encoder $s; done
}

steps=("${@:-A0 A5 A6 A7}")
for step in ${steps[@]}; do "$step"; done
echo "[$(date +%F' '%T)] queue done"
