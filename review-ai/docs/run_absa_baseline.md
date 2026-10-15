# ABSA baseline 运行说明

> **状态（2026-10-12）**：2026-07 的首次 E1 运行说明，保留作历史。论文实验的完整命令见根目录 `REPRODUCE.md`。注意本文的训练命令是早期版本（未用 ASL 与阈值调优）；关于 Gold TSV 的解析注意事项仍然准确。

本实验使用 `gold/data/travel_aspect_sentiment.tsv` 做日语酒店评论方面级情感分析（ABSA）baseline。

## 任务定义

输入：一句日语酒店评论。

输出：14 个多标签分类目标：

- `朝食_ポジティブ`, `朝食_ネガティブ`
- `夕食_ポジティブ`, `夕食_ネガティブ`
- `風呂_ポジティブ`, `風呂_ネガティブ`
- `サービス_ポジティブ`, `サービス_ネガティブ`
- `立地_ポジティブ`, `立地_ネガティブ`
- `設備・アメニティ_ポジティブ`, `設備・アメニティ_ネガティブ`
- `部屋_ポジティブ`, `部屋_ネガティブ`

一句话可以同时命中多个标签，因此这是 multi-label classification，不是单标签三分类。

## 数据准备

项目维护两种数据口径（同一 seed=42，按 review_id 分组切分）：

| 口径 | 位置 | 说明 |
|------|------|------|
| annotated-only | `data/absa_*.csv` | 只保留至少含一个方面标签的句子（53,192 句）。当前 v1 实验（travel/focal/asl）使用这份。 |
| full setting | `data/absa_full/absa_*.csv` | 保留无标注句作为全 0 背景样本（76,624 句）。用于评估真实输入场景（E4）。 |

生成命令（脚本在 `tools/`）：

```bash
cd /home/etl/gujincheng/review-ai
source .venv/bin/activate

# annotated-only（当前 data/ 内容）
python tools/prepare_absa_dataset.py \
  --input gold/data/travel_aspect_sentiment.tsv \
  --output_dir data \
  --drop_unlabeled \
  --seed 42

# full setting
python tools/prepare_absa_dataset.py \
  --input gold/data/travel_aspect_sentiment.tsv \
  --output_dir data/absa_full \
  --seed 42
```

每个输出目录包含 `absa_train/valid/test.csv`、`absa_label_columns.json`、`absa_split_summary.json`。

> 历史说明：早期 v1 实验的 `experiment_config.json` 中记录的路径是 `data/travel_absa/`，该目录后来重命名为 `data/`（内容一致，annotated-only，seed=42）。

## 训练 baseline

```bash
python train_absa_baseline.py \
  --train_file data/absa_train.csv \
  --validation_file data/absa_valid.csv \
  --test_file data/absa_test.csv \
  --labels_file data/absa_label_columns.json \
  --output_dir checkpoints/absa_xlmr_baseline \
  --model_name xlm-roberta-base \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5
```

输出：

- `checkpoints/absa_xlmr_baseline/experiment_config.json`
- `checkpoints/absa_xlmr_baseline/pos_weight.json`
- `results/absa_xlmr_baseline_metrics.json`
- `results/absa_xlmr_baseline_report.txt`

## 指标口径

脚本默认阈值为 `0.5`，即 `sigmoid(logit) >= 0.5` 视为预测命中。

主要看：

- `micro_f1`：整体标签级性能，适合不均衡多标签任务的主指标
- `macro_f1`：14 个标签平均性能，更能暴露少数类短板
- `sample_f1`：按句子计算的多标签 F1
- `exact_match_accuracy`：14 个标签完全一致才算对，通常很严格
- 每个标签的 precision / recall / F1：用于找出具体弱项

## 注意事项

`travel_aspect_sentiment.tsv` 有一个解析坑：表头是 18 列，但数据行是 17 列，开头多了一个空的索引列名。另有少量评论文本含制表符。不要直接用默认 `pandas.read_csv` 对齐训练；应使用 `tools/prepare_absa_dataset.py` 生成 CSV。

训练脚本默认使用 `BCEWithLogitsLoss(pos_weight=...)` 处理标签不均衡。如需关闭，可加：

```bash
--no_pos_weight
```

其他损失函数选项（当前推荐 ASL + 阈值调优）：

```bash
--loss_type asl --asl_gamma_neg 4.0 --asl_clip 0.05 --no_pos_weight --tune_threshold
# 或
--loss_type focal --focal_gamma 2.0 --max_pos_weight 8.0 --tune_threshold
```
