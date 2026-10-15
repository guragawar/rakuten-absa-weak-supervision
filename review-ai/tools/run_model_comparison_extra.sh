#!/usr/bin/env bash
# Additions to docs/model_comparison_plan.md (section "追加"): mBERT, llm-jp-3.1-13B, gpt-oss-20B.
# Each LLM is first format-checked on the first 200 valid sentences; gpt-oss is also load-tested and
# skipped (with the reason logged) if it cannot run on this GPU. WAIT_FOR=<pid> waits for another job.
#
# Usage:  WAIT_FOR=<pid> GPU=1 bash tools/run_model_comparison_extra.sh
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

# ── mBERT (encoder, E1 settings) ──
for seed in 42 43 44; do
  name="CMP_E1_mbert_seed${seed}"
  if [[ -f "results/${name}_metrics.json" ]]; then echo "[skip] $name"; continue; fi
  echo "[$(date +%F' '%T)] $name"
  $PY train_absa_baseline.py $FT --model_name google-bert/bert-base-multilingual-cased --seed "$seed" \
      --output_dir "checkpoints/tmp_$name" --experiment_name "$name" > "$LOG_DIR/$name.log" 2>&1
  rm -rf "checkpoints/tmp_$name"
done

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

llm() {  # name mode extra-args...
  local name=$1 mode=$2; shift 2
  if [[ ! -f "results/LLMCHECK_${name#LLM_}_metrics.json" ]]; then
    echo "[$(date +%F' '%T)] format check $name"
    $PY tools/llm_absa_eval.py "$@" --mode "$mode" --split valid --max_rows 200 \
        --name "LLMCHECK_${name#LLM_}" > "$LOG_DIR/LLMCHECK_${name#LLM_}.log" 2>&1 \
      || { echo "[$(date +%F' '%T)] FAILED format check $name"; return 0; }
    grep "RESULT" "$LOG_DIR/LLMCHECK_${name#LLM_}.log" || true
  fi
  if [[ -f "results/${name}_metrics.json" ]]; then echo "[skip] $name"; return; fi
  echo "[$(date +%F' '%T)] $name"
  $PY tools/llm_absa_eval.py "$@" --mode "$mode" --split test --name "$name" > "$LOG_DIR/$name.log" 2>&1 \
    || echo "[$(date +%F' '%T)] FAILED $name (see $LOG_DIR/$name.log)"
}

# ── llm-jp-3.1-13B-instruct4 (NII) ──
LLMJP="--model llm-jp/llm-jp-3.1-13b-instruct4 --revision da86a55d7a754d1a45bc2d860f8548841a84f42f --batch_size 4"  # 16 ran out of memory with few-shot prompts
if has_space 27; then
  llm LLM_llmjp31_13b_zero zero $LLMJP
  llm LLM_llmjp31_13b_few  few  $LLMJP
  [[ -f results/LLM_llmjp31_13b_zero_metrics.json && -f results/LLM_llmjp31_13b_few_metrics.json ]] \
    && drop_model llm-jp/llm-jp-3.1-13b-instruct4  # keep the weights if a run failed
fi

# ── gpt-oss-20B (OpenAI open weights), reasoning effort low ──
GPTOSS="--model openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee --reasoning_effort low --max_new_tokens 512 --batch_size ${GPTOSS_BATCH:-1}"
# bf16 weights take 41.8 GB on a 48 GB GPU: the zero-shot run used batch 2; few-shot prompts ran out of memory
# at batch 2 (2026-10-13), so few-shot uses batch 1. Greedy decoding, so batch size does not change the method.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%F' '%T)] gpt-oss load test"
if has_space 14 && $PY - > "$LOG_DIR/gptoss_load_test.log" 2>&1 <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m = "openai/gpt-oss-20b"; rev = "6cee5e81ee83917806bbde320786a8fb61efebee"
tok = AutoTokenizer.from_pretrained(m, revision=rev)
model = AutoModelForCausalLM.from_pretrained(m, revision=rev, torch_dtype="auto", device_map="cuda")
print("dtype", next(model.parameters()).dtype, "gpu GB", round(torch.cuda.memory_allocated() / 1e9, 1))
enc = tok.apply_chat_template([{"role": "user", "content": "1+1=?"}], add_generation_prompt=True,
                              reasoning_effort="low", return_tensors="pt", return_dict=True).to("cuda")
out = model.generate(**enc, max_new_tokens=64, do_sample=False)
print("ok", tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)[-80:])
PY
then
  echo "gpt-oss load test passed: $(grep -E '^(dtype|ok)' "$LOG_DIR/gptoss_load_test.log" | tr '\n' ' ')"
  llm LLM_gptoss20b_zero zero $GPTOSS
  llm LLM_gptoss20b_few  few  $GPTOSS
  [[ -f results/LLM_gptoss20b_zero_metrics.json && -f results/LLM_gptoss20b_few_metrics.json ]] \
    && drop_model openai/gpt-oss-20b  # keep the weights if a run failed
else
  echo "gpt-oss load test FAILED, skipped: $(tail -1 "$LOG_DIR/gptoss_load_test.log")"
fi
echo "[$(date +%F' '%T)] extra comparison done"
