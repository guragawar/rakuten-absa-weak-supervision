# About this repository

This is a fresh export of the working repository at commit `fdb9f50`, without its history.
The working repository is never pushed, because its early history contains licensed data.

Left out of the export:
- `review-ai/legacy/`: an earlier, unrelated project (Mercari comments, 3-class sentiment).
- `review-ai/data/panel_exclude_text_hashes.jsonl`: hashes of licensed review texts used to remove
  Gold sentences from the pretraining pool; rebuilt from the licensed data by `tools/check_panel_gold_overlap.py`
  and `tools/approx_dedup_panel_gold.py` (REPRODUCE.md, section on the exclusion list).

No Rakuten Travel data, Gold corpus text or derived per-sentence files are included; see README.md
("Data") for how to obtain the data under the NII-IDR agreement.
