# Reproducing the thesis results

Every number in the thesis is registered in [`review-ai/docs/thesis_numbers.md`](review-ai/docs/thesis_numbers.md)
together with the file it comes from. This document gives the commands that produce those files.

Status marks used below:

| Mark | Meaning |
|---|---|
| ✅ | Re-run on 2026-10-12; output **byte-identical** to the file used in the experiments |
| ☑️ | Re-run on 2026-10-12; content identical (format differs only as noted) |
| 🔁 | Command known from records made at run time (metrics JSON, checkpoint config, queue log, or the project log); not re-run |
| ⚠️ | **Reconstructed**: the data arguments of these runs were not logged at the time; values are inferred from run names, step counts and data sizes, as explained in the step |

## 0. Setup

```bash
bash setup_env.sh                 # creates review-ai/.venv (Python 3.10, pinned requirements)
cd review-ai                      # every command below runs from review-ai/
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
# Keep every cache inside the project: the server account is shared, and the datasets cache holds
# Arrow copies of the licensed weak-label files. The queue scripts in tools/ set these themselves.
export HF_HUB_CACHE=/home/etl/gujincheng/hf_cache HF_DATASETS_CACHE=/home/etl/gujincheng/hf_cache/datasets
```

Hardware used: one NVIDIA RTX 6000 Ada (48 GB) per run. Approximate times on an otherwise idle GPU:
E1/E8 fine-tuning ≈ 5 min; E7 on the 2024 pool (2 epochs) ≈ 1 h; E7 on 2M reviews (1 epoch) ≈ 1.5 h;
E7 on the full 6.08M pool (1 epoch) ≈ 4–5 h.

Pretrained models (Hugging Face revisions that were used; pass `revision=` or pre-download them to pin):

| Model | Revision |
|---|---|
| `xlm-roberta-base` | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` |
| `ku-nlp/deberta-v3-base-japanese` | `0c320ba8f63dcbb8adb049a79349437fa5728ac9` |
| `cl-tohoku/bert-base-japanese-v3` | `65243d6e5629b969c77309f217bd7b1a79d43c7e` |

### Data (licensed, not in this repository)

| File | Content | Obtain from |
|---|---|---|
| `gold/data/travel_aspect_sentiment.tsv` | Gold ABSA corpus: 楽天トラベルレビュー：アスペクト・センチメントタグ付きコーパス (Nakayama et al., LREC 2022) | Rakuten Data Release via NII-IDR |
| `gold/data/raw/01_Travel_Review.{2009..2024}.tsv` | Rakuten Travel review panel, 16 yearly files (2016 file is truncated: 82,856 rows) | Rakuten Data Release via NII-IDR |

Both are distributed under the NII-IDR agreement. Do not copy them, or anything derived that contains
review text, outside the lab server.

## 1. Data preparation

### 1.1 Gold ABSA splits ✅

```bash
$PY tools/prepare_absa_dataset.py --drop_unlabeled          # -> data/absa_{train,valid,test}.csv
```
80:10:10 split grouped by `review_id` (seed 42), sentences with at least one label only:
42,469 / 5,367 / 5,356 sentences. `--drop_unlabeled` is required; the default keeps unlabeled sentences.

### 1.2 Explicit / implicit tags (Task 1, used by E2 and all implicit-negative metrics) 🔁

```bash
$PY tools/split_explicit_implicit.py --report               # -> data/absa_explicit_implicit_tags.csv
```
Uses the `oseti` sentiment lexicon; sentence results are cached in `data/oseti_sentence_cache.json`.

### 1.3 Parse the panel 🔁

```bash
$PY tools/parse_travel_review_panel.py --export_dir data/panel_parsed --report   # all years
# -> data/panel_parsed/panel_{YYYY}.csv  (6,996,903 rows in total)
```

### 1.4 Exclude Gold reviews from the panel

The exclusion list `data/panel_exclude_text_hashes.jsonl` (11,369 text hashes: 9,911 exact + 1,458
near-duplicate matches with Gold reviews) **is committed** and is the list all experiments used.
To reproduce the experiments, use it as is and skip to 1.5.

Rebuilding it from scratch (not re-run on 2026-10-12; order matters, the first command overwrites the list):
```bash
$PY tools/check_panel_gold_overlap.py                                   # exact matches, all years
$PY tools/build_panel_train_pool.py --panel_csv "data/panel_parsed/panel_*.csv" \
    --out_pool data/panel_train_pool_all.csv --out_stats data/panel_pool_stats_all.json
$PY tools/approx_dedup_panel_gold.py --panel_csv data/panel_train_pool_all.csv \
    --out results/approx_dedup_panel_gold_allyears.json --update_exclude  # MinHash J >= 0.85, appends
```
then rebuild the pools in 1.5.

### 1.5 Weak-supervision pools ✅

```bash
# 2024 pool of the main E7 run. --no_dedup: this pool was built before duplicate removal existed
$PY tools/build_panel_train_pool.py --no_dedup                   # -> data/panel_train_pool.csv (583,777 rows)

# all-year pool for the data-size, year and DeBERTa experiments
$PY tools/build_panel_train_pool.py --panel_csv "data/panel_parsed/panel_*.csv" \
    --out_pool data/panel_train_pool_all.csv --out_stats data/panel_pool_stats_all.json
# -> 6,759,618 rows: 11,445 Gold-overlap rows and 225,840 duplicate texts removed
```
Train/valid assignment is a hash of the review text (90/10, seed 42), so duplicates never straddle the split.

### 1.6 Weak labels ☑️

Rating 1–2 → negative (1), 3–5 → non-negative (0), 0 (not rated) → masked. Head order:
1 立地, 2 部屋, 3 食事, 4 風呂, 5 サービス, 6 設備 (column mapping: `docs/data_spec_rakuten_travel.md`).
```bash
$PY tools/prepare_panel_weak_labels.py                           # -> data/panel_weak_{train,valid}.jsonl
$PY tools/prepare_panel_weak_labels.py --pool_csv data/panel_train_pool_all.csv \
    --out_dir data/weak_all --stats_path data/weak_all/panel_weak_label_stats_all.json
```
☑️ for the 2024 files: all original fields identical on every row; the current script also writes a
`source_year` field (added later for year filtering; ignored unless `--min/max_source_year` is set).

### 1.7 Ablation pools (A5, A6) 🔁

```bash
$PY tools/prepare_ablation_weak_labels.py --mode meal_split   --out_dir data/ablation/A5_meal_split
$PY tools/prepare_ablation_weak_labels.py --mode noise_filter --out_dir data/ablation/A6_noise_filter
```

## 2. Training runs

### 2.1 Command templates

E1 (fine-tune a pretrained model on Gold) and E8 (fine-tune an E7 encoder on Gold) use the same script
and the same settings. The defaults of `train_absa_baseline.py` are **not** the thesis settings
(default loss is BCE without threshold tuning), so these flags are always needed:

```bash
FT="--loss_type asl --asl_gamma_neg 4.0 --asl_clip 0.05 --no_pos_weight --tune_threshold"
# E1
$PY train_absa_baseline.py $FT --model_name MODEL --seed S --output_dir checkpoints/NAME --experiment_name NAME
# E8
$PY train_absa_baseline.py $FT --pretrained_encoder checkpoints/E7_NAME/encoder --seed S \
    --output_dir checkpoints/NAME --experiment_name NAME
```
With `--pretrained_encoder`, both model and tokenizer are loaded from the encoder directory and
`--model_name` is ignored (older `experiment_config.json` files still show the default `xlm-roberta-base`).
Every run writes `results/NAME_metrics.json` (test metrics as `eval_*`, validation metrics as `valid_*`,
and since 2026-10-12 the full `args`) and `results/test_probs/NAME.npz` (test probabilities).

E7 (weak-supervision pretraining) writes `checkpoints/NAME/encoder/` and `results/NAME_metrics.json`:
```bash
$PY train_panel_weak_pretrain.py --output_dir checkpoints/NAME --experiment_name NAME [data args]
```
Defaults = main E7 run: 2024 pool, 6 heads, 2 epochs, `sqrt` positive weights, max length 256, seed 42.

### 2.2 Run list

| Thesis item | Runs | Command | |
|---|---|---|---|
| E1 XLM-R baseline | seeds 42–46 | E1 template, `--model_name xlm-roberta-base` | 🔁 |
| E1 Japanese backbones | DeBERTa seeds 42–46; Tohoku seeds 42–44 | `--model_name ku-nlp/deberta-v3-base-japanese` / `cl-tohoku/bert-base-japanese-v3` | 🔁 |
| E2 explicit/implicit diagnosis | on an E1 checkpoint | `$PY tools/evaluate_by_source.py --checkpoint_dir CKPT` | 🔁 |
| E7 main | 1 | `train_panel_weak_pretrain.py --output_dir checkpoints/absa_weak_pretrain_v1 --experiment_name E7_panel_weak_pretrain` | 🔁 |
| E8 main (A1) | E7 seeds 42/43/44 × E8 seeds 42/43/44 | `bash tools/run_ablation_queue.sh A0 A7` (E7 seed 42 = main E7) | 🔁 |
| A2 fine-tuning strategy | E8 seed 42 | E8 template + `--encoder_lr 1e-5` (A2b) or `--freeze_encoder` (A2c) | 🔁 |
| A3 E7 epochs | E7 + E8 | E7 `--num_epochs 1`, then E8 | 🔁 |
| A4 overall head | E7 + E8 | E7 `--include_overall`, then E8 | 🔁 |
| A5 / A6 / A7 | see `docs/ablation_plan_A5_A7.md` | `bash tools/run_ablation_queue.sh A5 A6 A7` | 🔁 |
| Data size 0.58M / 2M / 6.08M | E7 + E8 seed 42 each | see 2.3 | ⚠️ |
| Recent (2021–24) vs early (2009–13) | E7 + E8 (recent: 3 E8 seeds) | see 2.3 | ⚠️ |
| DeBERTa + E7/E8 | E7 + E8 seeds 42–44 | see 2.3 | ⚠️ |
| SCL vs BCE (DeBERTa) | E7-SCL + E8 seeds 42–44 | `$PY train_panel_scl_pretrain.py` (defaults = the run), then E8 | 🔁 |

### 2.3 All-year E7 runs ⚠️

Shared arguments of all all-year runs (1 epoch was chosen after A3; 50k validation rows keep evaluation fast):
```bash
ALL="--train_file data/weak_all/panel_weak_train.jsonl --validation_file data/weak_all/panel_weak_valid.jsonl \
     --stats_file data/weak_all/panel_weak_label_stats_all.json --num_epochs 1 --max_valid_samples 50000"
$PY train_panel_weak_pretrain.py $ALL --max_train_samples 583778  --output_dir checkpoints/E7_scale_583k  --experiment_name E7_scale_583k
$PY train_panel_weak_pretrain.py $ALL --max_train_samples 2000000 --output_dir checkpoints/E7_scale_2M    --experiment_name E7_scale_2M
$PY train_panel_weak_pretrain.py $ALL                             --output_dir checkpoints/E7_scale_6.76M --experiment_name E7_scale_6.76M
$PY train_panel_weak_pretrain.py $ALL --min_source_year 2021      --output_dir checkpoints/E7_recent2M    --experiment_name E7_recent2M
$PY train_panel_weak_pretrain.py $ALL --min_source_year 2009 --max_source_year 2013 \
                                                                  --output_dir checkpoints/E7_early2M     --experiment_name E7_early2M
$PY train_panel_weak_pretrain.py $ALL --min_source_year 2021 --model_name ku-nlp/deberta-v3-base-japanese \
                                                                  --output_dir checkpoints/E7_deberta_recent2M --experiment_name E7_deberta_recent2M
```
How the values were reconstructed:
- `E7_recent2M`: the surviving checkpoint trained 63,550 steps × batch 32 = 2.03M rows, which is all
  2021–2024 training rows, so no subsampling.
- `E7_early2M`: 2009–2013 has 2,048,313 raw rows; after deduplication and the 10% validation split this is
  ≈1.80M, the size recorded for this run, so no subsampling.
- `E7_scale_583k`: 583,778 is the size of the 2024 file, matching the run name; the exact value was not logged.
- `E7_scale_6.76M`: "6.76M" is the size of the whole pool; training used its 6,084,468 training rows.

Each E7 is followed by E8 (template in 2.1) with the matching `--pretrained_encoder`.

## 3. Statistical analysis

Runs from before 2026-10-08 have no automatic `.npz`; their test probabilities were saved with
`tools/save_test_probs.py --checkpoint_dir CKPT --out results/test_probs/NAME.npz` before the checkpoints
were deleted. All tests below are seed-averaged paired bootstraps (2,000 resamples, seed 42) over test sentences.

```bash
P=results/test_probs
B="$PY tools/multiseed_f1_bootstrap.py"
# A1 main: E1 (5 runs) vs E8 (3 E7 seeds x 3 E8 seeds)
$B --probs_a $P/E1_xlmr_seed4{2,3,4,5,6}.npz \
   --probs_b $P/A0_ref_E8_seed4{2,3,4}.npz $P/A7_E7s4{3,4}_E8_seed4{2,3,4}.npz \
   --label_a xlmr_E1_5runs --label_b xlmr_E8_9runs \
   --labels 朝食_ネガティブ 夕食_ネガティブ 立地_ネガティブ 部屋_ネガティブ 設備・アメニティ_ネガティブ 風呂_ネガティブ サービス_ネガティブ \
   --out results/ablation_A1_main_E1x5_vs_E8x9.json
# explicit/implicit gap (RQ3)
$PY tools/multiseed_gap_bootstrap.py --probs_a $P/E1_xlmr_seed4{2,3,4,5,6}.npz --probs_b $P/A0_ref_E8_seed4{2,3,4}.npz \
   --label_a xlmr_E1 --label_b xlmr_E8_A0 --out results/ablation_A1_gap_E1_vs_A0.json
# ablations against the A0 reference
$B --probs_a $P/A0_ref_E8_seed4{2,3,4}.npz --probs_b $P/A5_E8_meal_split_seed4{2,3,4}.npz --label_a A0 --label_b A5_meal_split \
   --labels 朝食_ネガティブ 夕食_ネガティブ 立地_ネガティブ --out results/ablation_A5_vs_A0.json
$B --probs_a $P/A0_ref_E8_seed4{2,3,4}.npz --probs_b $P/A6_E8_noise_filter_seed4{2,3,4}.npz --label_a A0 --label_b A6_noise_filter \
   --labels 朝食_ネガティブ 夕食_ネガティブ 立地_ネガティブ --out results/ablation_A6_vs_A0.json
# backbone, stacking, SCL
$B --probs_a $P/E1_xlmr_seed4{2,3,4,5,6}.npz --probs_b $P/deberta_v3_seed4{2,3,4,5,6}.npz \
   --label_a xlmr_E1 --label_b deberta_E1 --out results/repro/xlmr_vs_deberta_5seed.json
$B --probs_a $P/deberta_v3_seed4{2,3,4}.npz --probs_b $P/E8_deberta_recent2M_seed4{2,3,4}.npz \
   --label_a deberta_E1 --label_b deberta_E8_bce --out results/repro/stacking_3seed.json
$B --probs_a $P/E8_deberta_recent2M_seed4{2,3,4}.npz --probs_b $P/E8_scl_deberta_recent2M_seed4{2,3,4}.npz \
   --label_a deberta_E8_bce --label_b deberta_E8_scl --out results/repro/scl_vs_bce_3seed.json
# mechanism of the DeBERTa gain on implicit negatives
$PY tools/mechanism_implicit_neg.py --probs_a $P/E1_xlmr_seed4{2,3,4,5,6}.npz --probs_b $P/deberta_v3_seed4{2,3,4,5,6}.npz \
   --label_a E1_xlmr --label_b deberta_v3 --out results/repro/mechanism_xlmr_vs_deberta.json
# RQ4: rating/text agreement (keyword-based)
$PY tools/rq4_rating_text_consistency.py --inputs data/panel_weak_train.jsonl data/panel_weak_valid.jsonl \
   --out results/rq4_rating_text_consistency_2024.json
```

## 4. Checking the repository

```bash
$PY tools/verify_repo.py
```
Runs on CPU in about a minute: checks that every thesis script starts (`--help`), and recomputes the
headline numbers from `results/test_probs/` and compares them with `docs/thesis_numbers.md`.

## 5. Known limits

- **GPU non-determinism.** The same configuration and seed can differ by up to ≈0.7 pp in negative F1
  between runs (observed for E1 with XLM-R and DeBERTa). Conclusions therefore rest on distributions over
  runs, never on a single run; a re-run will not reproduce individual numbers exactly.
- **Runs before 2026-10-08** saved test metrics only, and most of their checkpoints were deleted to save
  disk. July-era E8 seeds 43/44 and the data-size/year points can be re-run but not re-tested from saved
  predictions. Their numbers come from `results/*_metrics.json`.
- **Selection on test.** The data-size and year experiments were compared on the test set, one run each.
  They are reported as analysis, not used to choose the main configuration (see `docs/thesis_numbers.md` §0).
- **2016 panel file** is truncated (≈20% of the year).
