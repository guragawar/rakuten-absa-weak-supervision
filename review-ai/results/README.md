# Results index

Every thesis number cites its file in `../docs/thesis_numbers.md`; this page explains the naming.

## File types

| Pattern | Written by | Content |
|---|---|---|
| `NAME_metrics.json` | `train_absa_baseline.py` (E1/E8) | Test metrics as `eval_*`; since 2026-10-08 also validation metrics as `valid_*`; since 2026-10-12 the full command line as `args` |
| `NAME_report.txt` | `train_absa_baseline.py` | Human-readable version of the same run |
| `E7*_metrics.json`, `A*_E7_*_metrics.json` | `train_panel_weak_pretrain.py`, `train_panel_scl_pretrain.py` | Pretraining validation metrics on the panel (not a thesis metric; the value of E7 is measured through E8) |
| `test_probs/NAME.npz` | `train_absa_baseline.py`, `tools/save_test_probs.py` | Test probabilities, thresholds and labels per run; input of every significance test. **Not in git** (holds per-sentence Gold labels) |

`seed42` is the default and is often omitted from names; `seed43r`/`seed44r` mark same-seed reruns
(used to measure GPU non-determinism).

## Groups

| Experiment | Files |
|---|---|
| E1 baselines | `E1_baseline_rerun` (XLM-R seed 42), `E1_seed4{3,4}` (XLM-R, July), `E1_xlmr_seed4{3..6}` (XLM-R, reruns), `E1_deberta_v3*`, `E1_tohoku_v3*` |
| Early E1 loss exploration (before ASL was chosen) | `absa_xlmr_baseline`, `absa_travel_v1`, `absa_focal_v1`, `absa_asl_v1` |
| E2 explicit / implicit diagnosis | `E2_explicit_vs_implicit_*`, `E2_gap_*`, `task1_explicit_implicit_report.json` |
| E7 pretraining | `E7_panel_weak_pretrain` (main), `E7_A3_epoch1`, `E7_A4_7head`, `E7_scale_*`, `E7_recent2M`, `E7_early2M`, `E7_deberta_recent2M`, `E7_scl_deberta_recent2M` |
| E8 main (July) | `E8_weak_pretrain_finetune` (seed 42), `E8_seed4{3,4}`, `E8_implicit_subset_diagnosis.json` |
| A2–A4 ablations | `E8_A2b_layered_lr`, `E8_A2c_frozen`, `E8_A3_epoch1`, `E8_A4_7head` |
| Data size and year (analysis) | `E8_scale_{583k,2M,6.76M}`, `E8_recent2M*`, `E8_early2M` |
| Backbone stacking, SCL | `E8_deberta_recent2M*`, `E8_scl_deberta_recent2M*` |
| A0/A5/A6/A7 (pre-registered, Oct) | `A0_ref_E8_seed*`, `A5_*`, `A6_*`, `A7_*` |
| **Significance tests (current)** | `ablation_*.json` (A1 main and A5–A7), `repro/*.json` (backbone, stacking, SCL, mechanism) |
| RQ4 | `rq4_rating_text_consistency_2024.json` |
| Panel parsing / profiling | `travel_review_panel_*`, `parse_report_2016_2024.json` |
| Figures | `figures/` |

## Superseded files

These are kept because older notes cite them. Use the current files in the right column.

| Superseded | Use instead | Why |
|---|---|---|
| `A1_paired_bootstrap_E1_vs_E8.json` (single seed) | `ablation_A1_main_E1x5_vs_E8x9.json` | Main comparison now uses 5 E1 runs and 9 E8 runs |
| `implicit_neg_f1_multiseed_bootstrap_E1_vs_deberta.json`, `gap_multiseed_bootstrap_*` | `repro/xlmr_vs_deberta_5seed.json` | Same test, produced by a committed script |
| `stacking_3seed_bootstrap_summary.json`, `stacking_bootstrap_*` | `repro/stacking_3seed.json` | Same |
| `scl_vs_bce_3seed_bootstrap.json` | `repro/scl_vs_bce_3seed.json`, `repro/scl_vs_e1_3seed.json` | Same |
| `mechanism_implicit_neg_*.json` | `repro/mechanism_xlmr_vs_deberta.json` | Regexes are now fixed in `tools/mechanism_implicit_neg.py` |

## Not in git

`test_probs/`, `travel_review_2024_profile.json`, `approx_dedup_panel_gold*.json`, `uploaded_data_profile.json`
contain review text or per-sentence Gold labels (see the root `.gitignore`).
