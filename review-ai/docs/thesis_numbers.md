# 论文数字台账（Single Source of Truth）

> 规则：论文、幻灯片、要旨中出现的每一个数字都必须能在本表找到来源。
> 新增或修改数字时先更新本表，再改文稿。建立：2026-09-29。

## 0. 口径约定

| 项 | 约定 |
|---|---|
| 负面聚合 F1（agg-neg） | 7 个负面标签**合并计算**（pooled）的 F1，即 `eval_agg_negative_f1`，不是 7 个 F1 的平均 |
| 点估计 | metrics.json（训练脚本内评估）与 npz 重算（`results/test_probs/`）相差 ≤0.001，来自推理 batching/padding 差异。**显著性检验一律用 npz/bootstrap 口径** |
| 同 seed 不可复现 | 同配置同 seed 的 E1 重跑相差最多约 0.7pp（XLM-R seed43：0.7044 vs 0.7109；DeBERTa seed43：0.7182 vs 0.7253），属于 GPU 非确定性。**基线用全部运行的分布，不用单次运行** |
| 选模 | E8 只保存了 test 指标，scaling 各点的 checkpoint 已删除 → scaling/年份实验**定位为分析性实验**（全部报告，不据此选主配置） |

## 1★. 主线 A1 的论文主数字（2026-10-09 起采用，完全可由 npz 复算）

E1（5 seed）vs E8（9 次 = E7 seed 42/43/44 × E8 seed 42/43/44），seed 平均配对 bootstrap 2000 次。来源：`results/ablation_A1_main_E1x5_vs_E8x9.json`。

| 指标 | E1 | E8 | Δ | 95% CI |
|---|---|---|---|---|
| 负面聚合 F1 | 0.7066 | 0.7257 | **+1.91pp** | [+1.10, +2.75] |
| 正面聚合 F1 | 0.8605 | 0.8623 | +0.17pp | [−0.14, +0.51] n.s. |
| micro F1 | 0.8237 | 0.8292 | +0.55pp | [+0.22, +0.89] |
| 隐式负面 F1 | 0.8403 | 0.8546 | +1.43pp | [+0.69, +2.18] |
| 各方面负面 | | | 立地 +6.07 / 部屋 +3.56 / 設備 +2.42 / 風呂 +4.06（均显著）；サービス +0.60 / 朝食 −0.16 / 夕食 +0.39（n.s.） | |

- 稳健性：9 个 E8 值 0.7204–0.7333 全部高于 E1 全部 7 次运行的最大值 0.7143；E7 seed 引起的波动≈0（E8 seed 间 sd 0.39pp）。
- 隐式-显式差距：E1 +4.34 → E8 +4.99pp，缩小量 n.s.（`results/ablation_A1_gap_E1_vs_A0.json`）→ 增益不集中在隐式负面。
- 精确率/召回率（2026-10-13 由同一批 npz 计算，E1 5 次 / E8 9 次的平均）：负面精确率 0.6856 → 0.7027，召回率 0.7292 → 0.7504；预测为负面的格数 2,353 → 2,363（gold 负面 2,212）。即增益不是靠多报负面得到的。
- 隐式/显式负面的检出率（召回，同上）：隐式 0.7247 → 0.7461（n=2,065），显式 0.7918 → 0.8103（n=147）。**注意口径**：表中的「隐式负面 F1」只在 gold 为负面的格上计算（tags 文件只给 gold 为 1 的格打标），不含误报，等价于检出率 R 的函数 2R/(1+R)，因此不能与负面聚合 F1 比大小；论文中建议直接报告检出率或写明定义。
- 与旧数字的关系：下面 §1 的 +1.60pp（7 月 E8 3 次 vs E1 7 次，metrics 口径）与此一致；10/1 报告用的是 +1.60pp。论文改用本节数字，因为它覆盖 E7 随机性且全部可复算。
- 注意 7 月单 seed 的各方面结果（風呂 +0.2）已被多运行结果取代（風呂 +4.06 显著）。

## 1. 主线 A1：XLM-R，E1 vs E8（2024 单年，E7 2ep）

| 数字 | 值 | 来源 |
|---|---|---|
| E1 agg-neg，全部 7 次运行 | 0.7033 / 0.7044 / 0.7143 / 0.7109 / 0.7118 / 0.7055 / 0.7021；均值 **0.7075**，sd 0.0048 | `E1_baseline_rerun`、`E1_seed{43,44}`、`E1_xlmr_seed{43..46}` metrics |
| E8 agg-neg，3 seed | 0.7224 / 0.7190 / 0.7291；均值 **0.7235**，sd 0.0052 | `E8_weak_pretrain_finetune`、`E8_seed{43,44}` metrics |
| Δ（E8 均值 − E1 全运行均值） | **+1.60pp**；E8 最差（0.7190）> E1 最好（0.7143） | 上两行计算 |
| Δ（配对 seed，旧口径） | +1.62 ± 0.25pp（n=3） | 同上，与 seed 配对 |
| paired bootstrap（seed42） | agg-neg Δ **+2.00pp** CI[+0.80, +3.21] P=0.999；micro Δ +0.70pp 有意；agg-pos Δ +0.30pp CI[−0.25, +0.83] **n.s.** | `A1_paired_bootstrap_E1_vs_E8.json` |
| 各方面负面 F1 Δ（seed42） | 立地 +6.9 / 部屋 +5.6 / 設備 +3.8 / 風呂 +0.2 / サービス −0.0 / 朝食 −0.4 / 夕食 +0.1（pp） | `E1_baseline_rerun` vs `E8_weak_pretrain_finetune` metrics |

## 2. 消融（seed42，agg-neg / micro）

| 实验 | agg-neg | micro | 来源 |
|---|---|---|---|
| A2a 全参数（=E8） | 0.7224 | 0.8303 | `E8_weak_pretrain_finetune` |
| A2b 分层 lr | 0.7031 | 0.8178 | `E8_A2b_layered_lr` |
| A2c 冻结 encoder | 0.6040 | 0.6277 | `E8_A2c_frozen` |
| A3 E7 1 epoch | 0.7194 | 0.8317 | `E8_A3_epoch1` |
| A4 7 头（含総合） | 0.7215 | 0.8300 | `E8_A4_7head` |

## 2b. 追加消融 A5–A7（预注册，`docs/ablation_plan_A5_A7.md`，均对参照 A0 = 主线 E7 × E8 3 seed）

| 消融 | 主要结果（test，3 seed 平均配对 bootstrap） | 判定 | 来源 |
|---|---|---|---|
| A5 食事拆分（朝食/夕食头） | 朝食负面 +0.41pp CI[−1.22, +1.97]；夕食负面 +1.66pp CI[−0.63, +3.84]；负面聚合 +0.02pp | 部分成立（valid 朝食 +1.27pp 方向一致，test 不显著） | `results/ablation_A5_vs_A0.json` |
| A6 关键词噪声过滤 | 负面聚合 **−1.08pp** CI[−1.87, −0.24]；micro −0.56、正面 −0.39 也显著下降 | H6b：过滤有害（未提及关键词的负面评分仍有用） | `results/ablation_A6_vs_A0.json` |
| A7 E7 seed 43/44 | 对 E1：+1.96 / +1.71pp（均显著）；9 个 E8 值全部 > E1 最大值 | 结论不依赖 E7 seed | `results/ablation_A7_E1_vs_E7s4{3,4}.json` |

## 3. 分析性实验：数据量与年份（XLM-R，E7 1ep，seed42，单次运行）

| 实验 | agg-neg | Δ vs 0.7033 | micro | 来源 |
|---|---|---|---|---|
| 随机 58 万 | 0.7135 | +1.02 | 0.8275 | `E8_scale_583k` |
| 随机 200 万 | 0.7341 | +3.08 | 0.8335 | `E8_scale_2M` |
| 随机 676 万 | 0.7268 | +2.35 | 0.8343 | `E8_scale_6.76M` |
| 近年 2021–24，203 万 | 0.7345 | +3.12 | 0.8356 | `E8_recent2M` |
| 早年 2009–13，180 万 | 0.7188 | +1.55 | 0.8285 | `E8_early2M` |
| 近年 200 万，3 seed | 0.7345 / 0.7337 / 0.7251；均值 0.7311 | +2.37（vs E1 全运行均值） | — | `E8_recent2M{,_seed43,_seed44}` |

注：单次运行的 Δ 需要对照同 seed 不可复现噪声（最多约 0.7pp）来解读。

## 4. 骨干模型与叠加（DeBERTa-ja = ku-nlp/deberta-v3-base-japanese）

| 数字 | 值 | 来源 |
|---|---|---|
| DeBERTa-E1 vs XLM-R-E1，agg-neg（5 seed） | Δ **+1.89pp** CI[+0.94, +2.77] P=0.9995 | `results/repro/xlmr_vs_deberta_5seed.json`（原：`implicit_neg_f1_multiseed_bootstrap_E1_vs_deberta.json`） |
| 同上，隐式负面 F1（n=2065 对） | 0.8403 → 0.8501，Δ **+0.98pp** CI[+0.18, +1.76] P=0.9935 | 同上 |
| 隐式-显式 gap 缩小 | 点估计 +1.83pp，CI[−0.88, +4.66] **n.s.**（显式负面仅 147 对） | `results/gap_*bootstrap*.json` |
| DeBERTa 上叠加 E7/E8（3 seed） | agg-neg Δ **+1.85pp** CI[+0.88, +2.86]；隐式负面 Δ +1.45pp；agg-pos Δ +0.50pp（也有意） | `results/repro/stacking_3seed.json`（原：`stacking_3seed_bootstrap_summary.json`） |
| 累计 | XLM-R-E1（5 seed）0.7066 → DeBERTa+E7/E8（3 seed）0.7447 = **+3.81pp** | 同上 `total_vs_xlmr_e1` |
| 机制：DeBERTa 优势集中在线索少的句子（5 seed 多数投票，2,065 个隐式负面格） | 有否定形态 +0.22pp（n=916）/ 无否定 **+2.35pp**（n=1,149）/ 纯隐式 **+2.99pp**（n=603）/ 短句 ≤35 字 **+2.66pp**（n=1,051）/ 长句 +0.10pp；won 131 / lost 102 / 都漏 420 | `results/repro/mechanism_xlmr_vs_deberta.json`（规则定义写在脚本和 json 里） |

## 5. 否定性结果：SCL（SCAPT 核心）

各组 3 seed 均值（DeBERTa 底座，近年 200 万，`results/repro/scl_vs_bce_3seed.json`、`results/repro/scl_vs_e1_3seed.json`）：

| 组 | 负面聚合 F1 | 隐式负面 F1 |
|---|---|---|
| DeBERTa E1（无预训练，seed 42–44） | 0.7262 | 0.8495 |
| + E7-BCE / E8 | 0.7447 | 0.8640 |
| + E7-SCL / E8 | 0.7259 | 0.8603 |

| 比较 | agg-neg Δ | 隐式负面 Δ | 来源 |
|---|---|---|---|
| SCL vs BCE | **−1.88pp** CI[−2.84, −0.93] 显著为负 | −0.37pp n.s. | `results/repro/scl_vs_bce_3seed.json` |
| SCL vs E1（无预训练） | −0.03pp n.s. | +1.08pp 显著 CI[+0.47, +1.75] | `results/repro/scl_vs_e1_3seed.json` |

## 6. 数据

| 数字 | 值 | 来源 |
|---|---|---|
| 面板全量 | 6,996,903 条（2009–2024，2016 残缺仅 82,856 条） | 项目记录 / `data/panel_pool_stats_all.json` |
| Gold 泄漏排除 | 排除清单 11,369 个 text_hash（精确 9,911 + 近似 1,458）；在全年份面板中命中 **11,445 行**（同一文本可出现多次） | `data/panel_exclude_text_hashes.jsonl`、`data/panel_pool_stats_all.json` |
| 重复文本排除 | **225,840 行（3.2%）**：跨年 176,681 + 年内 49,159 | `data/panel_pool_stats_all.json`；年内/跨年拆分来自 2026-10-12 用改进后的 `build_panel_train_pool.py` 重算（输出与原池逐字节一致） |
| 最终预训练池（全年份） | **6,759,618 行**（train 6,084,468 / valid 675,150） | 同上；`data/weak_all/panel_weak_label_stats_all.json` |
| 2024 年池（主线 E7） | 583,777 行（排除 1 行；**未去重**，含年内重复文本 11,910 行≈2.0%，按 text_hash 划分 train/valid，重复文本不会跨 train/valid） | `data/panel_pool_stats.json`；复现需 `--no_dedup` |
| 更正记录 | 2026-10-12 前本表写的 6,761,076 / 225,849 是加入近似去重之前那一版池子的数字，实验实际使用的是上面的 6,759,618 | — |
| Gold 隐式标签比例 | 86.5%；负面中隐式约 92% | E2 诊断 |
| Gold 标签数（`--drop_unlabeled` 后的 53,192 句） | 92,273 个（正面 68,613 / 负面 23,660 = 25.6%）；负面中隐式 91.95%（train 17,503/19,038，valid 2,188/2,410，test 2,065/2,212）。显式 = 句中有 oseti 负面词命中（`tools/split_explicit_implicit.py` 的 `judge_source`） | `data/absa_explicit_implicit_tags.csv`（2026-10-13 统计） |

## 6b. RQ4：评分与正文的一致性（关键词法，2024 年 583,777 条，2026-10-08）

| 方面 | 负面标签数 | 负面中提及率 | 非负面中提及率 | lift | 负面但正文未提及（上限） |
|---|---|---|---|---|---|
| 立地 | 15,072 | 41.3% | 36.2% | 1.14 | 58.7% |
| 部屋 | 39,124 | 69.3% | 48.2% | 1.44 | 30.7% |
| 食事 | 23,144 | 74.1% | 64.5% | 1.15 | 25.9% |
| 風呂 | 41,213 | 54.3% | 38.9% | 1.39 | 45.7% |
| サービス | 27,374 | 56.1% | 38.9% | 1.44 | 43.9% |
| 設備 | 37,794 | 52.2% | 27.9% | 1.87 | 47.8% |

来源：`results/rq4_rating_text_consistency_2024.json`（`tools/rq4_rating_text_consistency.py`，词表写在 json 里）。
注意：关键词召回率有限（表达多样），「未提及」是上限估计；词表里「味」「感じ」等较泛，会抬高两侧提及率。待用句子级模型做第二版。
更正：此前口头提到的「食事差评 44% 未提吃饭」只用了早餐/晚餐关键词，完整食事词表下为 25.9%。

## 6c. 与其他模型的比较（预注册 `docs/model_comparison_plan.md`，2026-10-13 完成）

来源：`results/model_comparison_summary.json`；配对 bootstrap 在 `results/model_comparison/`。编码器为 seed 42/43/44 的均值；LLM 各 1 次运行（贪心解码）。

| 组 | 模型 | test 负面 F1 | 备注 |
|---|---|---|---|
| 编码器 E1 | mBERT | 0.6622 | |
| | XLM-R base（本批重跑） | 0.7079 | 主线台账 5 seed 为 0.7066，运行不同 |
| | DeBERTa-v3-ja（本批重跑） | 0.7239 | 主线台账 3 seed 为 0.7262，运行不同 |
| | 东北大 BERT large | 0.7387 | |
| | ModernBERT-ja 310M | 0.7459 | valid 最高（0.7735） |
| | XLM-R large | 0.7492 | |
| | LUKE-ja（lite） | 0.7559 | test 最高；未做 E7/E8 |
| | ModernBERT-ja 130M | 0.6242（lr 1e-5）/ 0.6206（lr 2e-5） | seed 44 在两个学习率下都崩溃；seed 42/43 为 0.743 / 0.738 |
| 编码器 + E7/E8 | XLM-R base | 0.7257 | §1★ |
| | DeBERTa-v3-ja | 0.7447 | §4 |
| | ModernBERT-ja 310M（A′） | 0.7564 | 对 E1 +1.05pp CI[+0.16, +1.95]；隐式负面 +1.68pp CI[+0.90, +2.48]；正面 +0.06 n.s.；valid 0.7735 → 0.7690（方向相反） |
| 本地 LLM（严格 / 宽松解析） | Qwen2.5-7B 零样本 / 少样本 | 0.4620 / 0.4620；0.5971 / 0.5971 | |
| | Swallow-8B 零样本 / 少样本 | 0.5633 / 0.5633；**0.6278 / 0.6287** | LLM 中最好；对 XLM-R base E1 −8.01pp（显著） |
| | llm-jp-3.1-13B 零样本 / 少样本 | 0.2577 / 0.4893；0.4059 / 0.4143 | 零样本有 2,643 句不是 JSON 输出 |
| | gpt-oss-20B 零样本 / 少样本 | 0.5539 / 0.5539；0.6120 / 0.6129 | reasoning low |

- 底座间的差：mBERT → LUKE-ja 约 9.4pp；LUKE-ja 对 DeBERTa + E7/E8 为 +1.12pp（显著）。
- GPT-4o 等外部 API 未实施（利用規約第3条第5項；NII IDR FAQ 4-1）。

## 6d. 追加实验 L / O / M（预注册 `docs/followup_experiments_plan.md`，2026-10-14）

**L 标注量曲线**（XLM-R base，E1 对 E8 = 主线 E7 编码器，各 5 seed；轮数按 E1 的 valid 选为「更新步数与全量相同」的 L-b）。来源：`results/label_efficiency_summary.json`。

| Gold 比例 | E1 → E8（test 负面 F1） | Δ [95% CI] |
|---|---|---|
| 10% | 0.6633 → 0.6777 | +1.43 [+0.55, +2.32] |
| 25% | 0.6831 → 0.7004 | +1.73 [+0.81, +2.62] |
| 50% | 0.6980 → 0.7113 | +1.33 [+0.41, +2.22] |
| 100% | 0.7066 → 0.7252 | +1.86 [+0.96, +2.79]（E8 5 次；主表 9 次为 +1.91） |

- 增益不随 Gold 量变化：Δ(10%) − Δ(100%) = −0.43pp [−1.36, +0.50]。
- E8@50% − E1@100% = +0.47pp [−0.42, +1.33]；E8@25% − E1@100% = −0.62pp [−1.55, +0.30]。等价标注比例（log₂ 插值，描述性）≈ **37%**。
- 固定 3 epoch（L-a）时 10% 的增益为 +5.03pp，但这是基线训练不足造成的（E1 valid 0.593 对 L-b 的 0.686），不作为结论。

**O 总评分对照**（只用総合评分 1 个输出的 BCE 预训练）。来源：`results/followup/O_*.json`。

| 底座 / 池 | E1 | 分项 BCE | 总评分 BCE | 总评分 SCL | 分项 − 总评分（BCE） | 总评分 BCE − E1 |
|---|---|---|---|---|---|---|
| DeBERTa / 2021–24 | 0.7262 | 0.7447 | 0.6818 | 0.7259 | +6.29 [+5.13, +7.46] | −4.44 [−5.52, −3.31] |
| XLM-R / 2024 | 0.7066 | 0.7273（A0 3 次） | 0.6891 | — | +3.82 [+2.78, +4.84] | −1.75 [−2.56, −0.89] |

- 总评分信号下 BCE 比 SCL 低 4.41pp [−5.46, −3.35]。总评分输出在 E7 的 valid F1 为 0.678 / 0.672（学得正常）。

**B 强底座上的 E7/E8**（2021–24 年池 1 epoch + E8 × 3 seed，对模型对比时的 E1 × 3）。来源：`results/followup/B_*_E7E8_vs_E1.json`。

| 底座 | E1 → E7/E8 | 负面 Δ [95% CI] | 隐式负面 Δ | valid Δ |
|---|---|---|---|---|
| BERT large | 0.7387 → 0.7499 | +1.12 [+0.12, +2.09] | +1.22（显著） | −0.23 |
| ModernBERT-ja 310M | 0.7459 → 0.7564 | +1.05 [+0.16, +1.95] | +1.68（显著） | −0.45 |
| XLM-R large | 0.7492 → 0.7434 | −0.58 [−1.53, +0.32] | −0.26 | −0.56 |
| LUKE-ja | 0.7559 → 0.7574 | +0.15 [−0.67, +0.99] | +0.26 | −0.25 |

- 4 个强底座中，test 显著且 valid 同方向的为 0 个。XLM-R base +1.91 对 XLM-R large −0.58。

**M 增益来源**（隐式负面 2,065 格，多数决检出率）。来源：`results/followup/mechanism_*_E1_vs_E8.json`。

| 底座 | 检出率 | 纯隐式 | 短句 ≤35 字 | 长句 |
|---|---|---|---|---|
| XLM-R base | 73.32% → 75.69%（+2.37） | +1.66 | +1.62 | +3.16 |
| DeBERTa-v3-ja | 74.38% → 76.22%（+1.84） | +1.66 | −0.38 | +4.14 |
| ModernBERT-ja 310M | 74.24% → 77.14%（+2.91） | +1.82 | +2.95 | +2.86 |

## 7. 可复现性状况

| 结果 | 能否从 npz 复算 | 备注 |
|---|---|---|
| E1 XLM-R 5 seed、DeBERTa 5 seed、DeBERTa 叠加 3 seed、SCL 3 seed | ✅ | `results/test_probs/` |
| E8 主线 seed42、近年 200 万 seed42 | ✅（2026-09-29 补存，已核对与 bootstrap 值一致） | `E8_xlmr_main2024_seed42.npz`、`E8_xlmr_recent2M_seed42.npz` |
| E8 主线 seed43/44、scaling 各点、早年 | ❌ 仅 metrics | checkpoint 已删除 |
| DeBERTa 对比 / 叠加 / SCL 的 bootstrap | ✅ 2026-10-08 用 `tools/multiseed_f1_bootstrap.py` 复算，点估计与 CI 和原 json 一致（差异仅为原 json 的取整，SCL 的 CI 差 ≤0.04pp 属 bootstrap 随机误差） | `results/repro/*.json`，**今后引用这些文件** |
| 机制分析 | ✅ 2026-10-08 用 `tools/mechanism_implicit_neg.py` 重算。原结果的否定/转折正则未保存，无法逐字还原；新脚本的规则已固定。won/lost、短句/长句完全一致，否定分层数字略变但结论不变（且更明显） | `results/repro/mechanism_xlmr_vs_deberta.json` |
