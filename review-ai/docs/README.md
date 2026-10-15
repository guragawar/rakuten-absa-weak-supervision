# Documentation index

Working documents are in Chinese; repository-level guides (`../../README.md`, `../../REPRODUCE.md`) are in English.

## Current

| Document | What it is | Use it when |
|---|---|---|
| [`thesis_numbers.md`](thesis_numbers.md) | **Single source of truth** for every number in the thesis, with the file each one comes from and the conventions (metric definitions, which runs count) | Writing any number into the thesis or slides. Check with `tools/verify_repo.py` |
| [`research_plan.md`](research_plan.md) | Research questions RQ1–RQ6, method, experiment matrix, with the status of each RQ | Framing the thesis; answering "what did we set out to show" |
| [`related_work.md`](related_work.md) | Literature review: 34 references, follow-up studies on the same corpus, closest prior work, comparability notes | Writing the related-work chapter and the comparison with prior results |
| [`ablation_plan_A5_A7.md`](ablation_plan_A5_A7.md) | Pre-registered ablations A5–A7 (written before the runs) and their results | Citing A5–A7; showing the decision rules were fixed in advance |
| [`data_spec_rakuten_travel.md`](data_spec_rakuten_travel.md) | Official field specification of the Rakuten Travel panel and the verified rating-column → aspect mapping | Anything that touches panel ratings (the column order is easy to get wrong) |

## Historical (kept as a record, not updated)

| Document | Date | Notes |
|---|---|---|
| [`implicit_sentiment.md`](implicit_sentiment.md) | 2026-07 | Implementation plan for the weak-supervision pipeline (tasks and acceptance criteria) |
| [`asset_inventory.md`](asset_inventory.md) | 2026-07-07 | Inventory of data, models and code before the legacy project was archived |
| [`data_review_travel_review_2024.md`](data_review_travel_review_2024.md) | 2026-07 | First assessment of the 2024 panel file |
| [`run_absa_baseline.md`](run_absa_baseline.md) | 2026-07 | First E1 run guide; superseded by `REPRODUCE.md` (its parsing notes on the Gold TSV are still accurate) |

Documents of the archived Mercari 3-class project are in `../legacy/mercari_3class/docs/`.
