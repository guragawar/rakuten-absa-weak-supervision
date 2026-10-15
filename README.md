# Aspect-rating weak supervision for Japanese hotel ABSA

Master's thesis project (GU JINCHENG). Code, results and documentation for
**aspect-based sentiment analysis (ABSA) of Japanese hotel reviews**, using the per-aspect star ratings
that guests leave on Rakuten Travel as weak supervision.

## The question

The Gold ABSA corpus labels each review sentence for 7 aspects (breakfast, dinner, bath, service,
location, facilities, room) × positive/negative, a 14-label multi-label task. Most negative labels are
*implicit*: the sentence contains no sentiment word (about 92% of negative labels), and these are harder
to detect. Guests also rate six aspects of their stay from 1 to 5. These ratings exist for about
7 million reviews, at no annotation cost, but they describe the whole stay, not a sentence.

Can pretraining an encoder on these ratings improve sentence-level ABSA, especially negative detection?

## Method

| Step | ID | What happens |
|---|---|---|
| Baseline | **E1** | Fine-tune a pretrained encoder (XLM-R; also DeBERTa-v3-japanese, Tohoku BERT) on Gold |
| Weak pretraining | **E7** | Train the encoder on panel reviews with 6 aspect heads: rating 1–2 = negative, 3–5 = non-negative, 0 = masked |
| Fine-tuning | **E8** | Fine-tune the E7 encoder on Gold with exactly the E1 settings; E8 − E1 is the effect of E7 |

Gold reviews that also appear in the panel (11,369 texts, exact or MinHash near-duplicates) are removed
from the pretraining data, so the test set is never seen during pretraining.

## Main results

Gold test set, means over runs, seed-averaged paired bootstrap (95% CI). Source of every number:
[`review-ai/docs/thesis_numbers.md`](review-ai/docs/thesis_numbers.md).

| Comparison | Negative F1 | Notes |
|---|---|---|
| E1 → E8, XLM-R (5 runs vs 9 runs) | 0.7066 → 0.7257, **+1.91 pp** [+1.10, +2.75] | Positive F1 +0.17 pp (n.s.); implicit-negative F1 +1.43 pp [+0.69, +2.18] |
| per aspect | location +6.07, bath +4.06, room +3.56, facilities +2.42 pp (significant) | breakfast, dinner, service: n.s. — the panel has one "meal" rating, so breakfast and dinner get no specific signal |
| E1 backbone: XLM-R → DeBERTa-v3-japanese | **+1.89 pp** [+0.94, +2.77] | Gain concentrated on sentences with few surface cues |
| E7/E8 on top of DeBERTa | **+1.85 pp** [+0.88, +2.86] | The two gains add up: 0.7066 → 0.7447 (+3.81 pp) |
| Contrastive (SCL) instead of aspect heads | −1.88 pp vs aspect heads | A single review-level polarity is not enough; aspect-specific signal matters |
| Keyword filter on weak labels (A6) | −1.08 pp | Ratings whose aspect is not mentioned in the text still carry useful signal |

Also analysed: pretraining data size (gains level off by about 2M reviews) and review year
(recent reviews help about twice as much as 2009–2013 reviews). These were compared on the test set,
one run each, so they are reported as analysis and not used to pick the configuration.

## Repository layout

```
.
├── README.md, REPRODUCE.md        how to reproduce every result
├── requirements.txt               direct dependencies (pinned); requirements-lock.txt = full environment
├── setup_env.sh                   builds review-ai/.venv
├── meetings/                      advisor meeting materials
└── review-ai/
    ├── train_absa_baseline.py     E1 / E8: 14-label ABSA fine-tuning (ASL loss, per-label thresholds)
    ├── train_panel_weak_pretrain.py   E7: aspect-rating weak pretraining
    ├── train_panel_scl_pretrain.py    E7-SCL: supervised-contrastive variant (comparison)
    ├── tools/                     data preparation, analysis, significance tests, verify_repo.py
    ├── docs/                      research plan, related work, number ledger (index: docs/README.md)
    ├── results/                   metrics and test JSON of every run (index: results/README.md)
    ├── data/                      derived data; files with review text are not in git
    ├── gold/data/                 licensed source data (not in git)
    ├── checkpoints/               model checkpoints (not in git)
    └── legacy/mercari_3class/     archived earlier project (3-class Mercari sentiment), not part of the thesis
```

## Quick start

```bash
bash setup_env.sh                        # Python 3.10 environment in review-ai/.venv
cd review-ai
.venv/bin/python tools/verify_repo.py    # self-check: scripts start, headline numbers recompute from saved predictions
```
Full pipeline, from raw data to every table: [`REPRODUCE.md`](REPRODUCE.md).

## Data and licensing

The source data are **not** in this repository and must not be redistributed:

- Rakuten Travel review panel 2009–2024 and the Gold ABSA corpus
  (楽天トラベルレビュー：アスペクト・センチメントタグ付きコーパス), obtained from the Rakuten Data Release
  under the NII-IDR agreement;
- Mercari comments used by the archived legacy project.

`.gitignore` excludes these files and every derived file that contains review text or per-sentence
Gold labels; `tools/verify_repo.py` checks that no tracked file contains review text.

> **Before publishing any code:** this repository has no remote, and its history before commit `39e3c20`
> contains licensed files. Do not push it. Create a fresh repository from an export of the current tree,
> without `review-ai/legacy/` (its annotation guides quote Mercari comments).

## References for the data

- Y. Nakayama, K. Murakami, G. Kumar, S. Bhingardive, I. Hardaway. A Large-Scale Japanese Dataset for
  Aspect-based Sentiment Analysis. LREC 2022, pp. 7014–7021.
- Y. Nakayama, K. Hikichi, Y. C. Liu, Y. Hirate. Rakuten Data Release: A Large-Scale and Long-Term Reviews
  Corpus for Hotel Domain. arXiv:2512.15151, 2025.
