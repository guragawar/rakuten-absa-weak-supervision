#!/usr/bin/env bash
# 実時監控 E7 弱監督預訓練（Ctrl+C 退出）
# 使い方: bash tools/watch_e7.sh
OUT="/tmp/claude-1000/-home-etl-gujincheng/e602ceb2-78c6-448a-a286-b8048e69563e/scratchpad/e7_deberta.log"
CKPT="/home/etl/gujincheng/review-ai/checkpoints/E7_deberta_recent2M"
TOTAL=63550

while true; do
  clear 2>/dev/null || printf '\033[H\033[2J'
  echo "════════ E7-DeBERTa (recent2M) 実時監控（Ctrl+C で終了）════════"
  echo "  $(date '+%Y-%m-%d %H:%M:%S')   ログ: $OUT"
  echo
  echo "▍GPU1"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
             --format=csv,noheader -i 1 2>/dev/null | sed 's/^/    /'
  echo
  STEP=$(grep -oE "[0-9]+/${TOTAL}" "$OUT" 2>/dev/null | tail -1 | cut -d/ -f1)
  STEP=${STEP:-0}
  PCT=$(awk "BEGIN{printf \"%.1f\", $STEP/$TOTAL*100}")
  echo "▍進度: ${STEP} / ${TOTAL} steps  (${PCT}%)"
  grep -oE "[0-9]+%\|[^|]*\| [0-9]+/${TOTAL} \[[^]]*\]" "$OUT" 2>/dev/null | tail -1 | sed 's/^/    /'
  echo
  echo "▍最近 loss（下降していれば正常）"
  grep -oE "\{'loss': [0-9.]+.*'epoch': [0-9.]+\}" "$OUT" 2>/dev/null | tail -3 | sed 's/^/    /'
  echo
  echo "▍検証 macro_f1（参考線 ~0.55、XLM-R版E7と同水準なら良）"
  grep -oE "'eval_macro_f1': [0-9.]+" "$OUT" 2>/dev/null | tail -8 | nl -w2 -s'. ' | sed 's/^/    /'
  echo
  CKN=$(ls -d "$CKPT"/checkpoint-* 2>/dev/null | wc -l)
  echo "▍保存済 checkpoint: ${CKN} 個"
  if grep -q "ACCEPTANCE" "$OUT" 2>/dev/null; then
    echo
    echo "════════ 完了 ════════"
    grep "ACCEPTANCE" "$OUT" | tail -1 | sed 's/^/    /'
  fi
  sleep 3
done
