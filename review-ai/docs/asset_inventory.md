# 项目资产清单（详细版）

> **状态（2026-10-12）**：本文是 2026-07 的盘点快照，保留作历史，正文未更新。之后的变化：旧三分类项目（Mercari）的代码、文档、gold3 数据与结果已移至 `review-ai/legacy/mercari_3class/`（见其 README）；含评论原文的数据文件已全部移出 git；`implicit_sentiment.md` 移至 `docs/`。当前结构见根目录 `README.md`。

**盘点日期**：2026-07-07
**项目位置**：`192.168.10.199:/home/etl/gujincheng`（总体积约 54G）
**版本控制**：Git 已初始化（7/7），初始提交 `2a8f042`，94 个文件入库；
venv、checkpoint、可再生大数据已通过 `.gitignore` 排除。

**价值分级说明**

| 级别 | 含义 |
|------|------|
| ★★★ 核心 | 研究结论直接依赖，丢失后无法低成本重建（人工标注、原始数据、最佳模型） |
| ★★ 重要 | 支撑研究可复现与可解释，重建需要较多算力或工时 |
| ★ 辅助 | 有复用潜力但当前未直接使用，或可由脚本再生 |
| ○ 可清理 | 已被取代或冗余，确认后可删除 |

---

## 1. 数据资产

### 1.1 ABSA 主数据（当前研究主线）

#### `gold/data/travel_aspect_sentiment.tsv` — ★★★ 核心

- **类型**：人工标注的方面级情感语料（原始 TSV，8.7MB）
- **规模**：76,624 句，来自约 12,476 条酒店评论；每条评论最多 44 句
- **标注体系**：7 方面（朝食/夕食/風呂/サービス/立地/設備・アメニティ/部屋）× 2 极性 = 14 个 0/1 标注列；标注命中共 92,273 次（正 68,613 / 负 23,660，约 3:1）
- **价值**：
  - 全项目**规模最大、粒度最细**的人工标注资产，是 ABSA 主线的根基；
  - 按市面日语标注成本估算，7.6 万句 × 14 标签的人工标注属于难以复购的资源；
  - 多标签结构支持的衍生研究：方面共现分析、按酒店聚合口碑、正负并存句研究。
- **注意事项**：表头 18 列而数据行 17 列（开头多一个空索引列名），且 105 行文本内含制表符——**必须**用 `tools/prepare_absa_dataset.py` 解析，直接 `pandas.read_csv` 会整体错位一列。

#### 训练切分（两种口径）— ★★

| 口径 | 位置 | train/valid/test | 用途 |
|------|------|------------------|------|
| annotated-only | `data/absa_{train,valid,test}.csv` | 42,469 / 5,367 / 5,356 | v1 实验（E1–E3）所用；只保留至少含 1 个方面标签的句子 |
| full setting | `data/absa_full/` | 61,046 / 7,678 / 7,900 | E4 真实场景评估；含 23,432 句全 0 背景句 |

- **产生方式**：`tools/prepare_absa_dataset.py`，seed=42，按 `review_id` 分组切分（同一评论的句子不会跨集合，防泄漏）
- **价值**：实验可比性的基准。切分摘要（`absa_split_summary.json`，含每标签逐集合计数）+ 标签定义（`absa_label_columns.json`）已入 Git，任何人可用同一命令逐字节复现。
- **注意事项**：两口径**不可混用对比**；测试集最少的标签 `立地_ネガティブ` 仅约 112–125 个正例，单标签指标方差偏大。

### 1.2 三分类数据（前期工作）

#### Gold 人工标注 — ★★★ 核心

| 文件 | 规模 | 说明 |
|------|------|------|
| `gold/data/reviewed/mercari_final_gold3_ja_pos_neg_neu.csv` | 3,638 条 | 人工终审的三分类标注（源头文件） |
| `gold/data/{train,valid,test}/gold3_*.csv` | 2,910 / 364 / 364 | 官方切分 |
| `data/gold_{train,valid,test}_unified.csv` | 同上 | 统一格式版（text / label_name / label_id） |
| `gold/data/patches/`、`refined/` | v1.1 | 标注修正闭环的补丁与修正版数据集机制（已验证可用，当前补丁为空） |

- **价值**：Mercari 域的唯一高质量标注；是"弱标注 vs 人工标注"这一研究叙事的证据基础；也是未来做两阶段训练（弱→Gold 微调）的微调集。

#### 边界样本 — ★★（未兑现资产）

- **位置**：`gold/data/reviewed/boundary_samples/`，195 条待复核清单 + 按 5 类切分的子文件（ambiguous_negative 22 / ambiguous_neutral 46 / error_neg→neu 17 / error_neu→neg 61 / high_uncertainty 49）
- **产生方式**：`tools/extract_boundary_samples.py` 基于模型不确定性与错误分析自动挑选
- **价值**：主动学习/标注高效化研究的现成实验材料——这 195 条是"最值得人工标注的样本"，配套的中日文标注指南（12 个 reason_tag 体系）和合并/统计/补丁工具链全部就绪；
- **现状**：`reviewed_label` 全空，复核进度 0%。**工具链的前期投入尚未产生回报**，若启动约 1–2 天人工可完成。

#### 弱标注与流水线产物 — ★

| 文件 | 规模 | 说明 |
|------|------|------|
| `outputs/stage1_clean.csv` | 约 182MB | 清洗后全量 Mercari 评论（弱标注前的原料池） |
| `outputs/stage2_weaklabels_sample.csv` | 约 50K | 关键词规则弱标注样本 |
| `outputs/stage3_{train,valid,test}.csv` | 约 45K 合计 | 弱标注训练切分 + 类权重 |
| `data/weak_stage{2,3}_unified.csv`、`weak_stage2_stage3_merged.csv` | 50K/50K/45K | 统一格式版（stage3 标签优先的合并规则） |
| `data_raw/` | 156MB | 原始 Mercari 评论 |
| `outputs/stage2_reviewpack_{neg,neu,pos}_300.csv` | 各 300 条 | 人工审核用抽样包 |

- **价值**：弱监督研究线的完整素材；`stage1_clean` 是唯一的"未标注大池子"，未来做自训练（self-training）或数据增强时的采样源。全部可由 `process.py` 从 `data_raw` 再生，因此不入 Git。

### 1.3 辅助数据 — ★

| 资产 | 规模 | 价值 |
|------|------|------|
| 筑波语料 `annotation01_tsukuba_corpus_20140930.tsv` | 5,308 句 / 1,000 条评论 | **双标注**（两位标注者，一致率 69%）是其独特价值：可做标注一致性分析、人类基准（human ceiling）讨论、或 ABSA 外部验证集。标签为字母代码（p/Z/k/e/y/o），使用前需对照原始标注规范确认含义。**至今未用于任何实验，是明确的低垂果实。** |
| 酒店维表 `02_Travel_HotelMaster.tsv` | 27,603 家（ID→名称，无重复无缺失） | 支持把句子级预测聚合到酒店级：如"各酒店的方面口碑画像"，是研究落地应用（demo/可视化）的关键连接件。 |

---

## 2. 模型资产（共约 39G，未入 Git，均为 XLM-RoBERTa-base 微调）

### 2.1 ABSA 模型

#### `checkpoints/absa_asl_v1/checkpoint-3984` — ★★★ 当前最佳

- **训练配置**：ASL（γ_neg=4, clip=0.05）、无 pos_weight、逐标签阈值调优、3 epoch、annotated-only 口径
- **测试表现**：Micro F1 **0.822** / Macro F1 0.773 / Sample F1 0.818 / 严格全对率 0.647；正面聚合 F1 0.860，负面聚合 F1 0.705
- **附属文件**：`experiment_config.json`（完整超参）、`thresholds.json`（14 个调优阈值，0.50–0.70）
- **价值**：论文主结果的载体；也是后续错误分析（E5）和推理应用的默认模型。丢失需约 6 分钟 GPU 重训（成本低），但其**阈值文件与配置是精确复现结果的凭证**（已在 checkpoint 目录，建议单独备份小文件）。

#### 其他 ABSA checkpoint

| 模型 | 体积 | 表现 | 级别 | 说明 |
|------|------|------|------|------|
| `absa_focal_v1` | 3.2G | Micro F1 0.821 | ★★ | 消融对照（Focal γ=2 + pos_weight≤8 + 调阈值），论文对比表需要 |
| `absa_travel_v1` | 3.2G | Micro F1 0.766 | ★★ | BCE 基线（t=0.5），证明改进幅度的锚点 |
| `absa_xlmr_baseline` | 6.3G（2 ckpt） | Micro F1 0.611 | ★ | 最早的 full 口径 BCE 实验（E0）；结果 JSON 保留即可，checkpoint 被 E4 取代后可删 |

### 2.2 三分类模型（前期）

| 模型 | 体积 | 表现 | 级别 | 说明 |
|------|------|------|------|------|
| `gold_only_on_gold`（3 ckpt） | 7.6G | Macro F1 0.7125 | ★★ | 前期结论的证据模型；动机章节引用 |
| `weak_baseline_on_gold` | 6.3G | Macro F1 0.5656 | ★★ | 同上，对照组 |
| `gold/models/baseline_gold3_xlmr_base` | 6.3G | — | ★ | Gold 子项目冻结基线，与 gold_only 功能重叠 |
| `hf_xlmr_base_e3` | 3.2G | — | ○ 可清理 | 最早期弱标注实验，已被全面取代 |
| `hf_xlmr_base`、`weak_smoketest_on_gold` | <1MB | — | ○ | 空目录/冒烟测试残留 |

> **清理建议**：删除 `hf_xlmr_base_e3`（3.2G）、`absa_xlmr_baseline` 旧 ckpt（E4 跑完后，6.3G）、
> 视情况删 `gold/models/baseline_gold3_xlmr_base`（6.3G，若 gold_only_on_gold 足够）→ 最多释放约 **16G**。
> 删除前保留各自的 metrics/report（已入 Git，不占空间）。

---

## 3. 代码资产（22 个脚本，全部入 Git，语法与冒烟测试通过）

### 3.1 训练与推理（`review-ai/` 顶层）— ★★★

| 脚本 | 行数级 | 功能与价值 |
|------|--------|-----------|
| `train_absa_baseline.py` | ~550 行 | **主线核心**。ABSA 多标签训练一站式：BCE/Focal/ASL 三种损失、pos_weight 上限、负样本过采样、逐标签阈值调优（验证集网格搜索）、正负聚合指标、自动落盘 metrics/report/thresholds。所有 CLI 参数均已验证生效（含 7/7 修复的 `--asl_clip`）。 |
| `train_unified.py` | ~400 行 | 三分类统一入口：显式指定三个数据文件、固定标签映射、数据来源追踪防泄漏、类权重。前期两组对照实验的产生器。 |
| `training_utils.py` | ~80 行 | 共享 `WeightedTrainer` + effective-number 类权重（7/7 新增，消除三份重复实现）。 |
| `train_gold3_baseline.py` / `train_hf_gpu.py` | ~250/200 行 | Gold3 / 弱标注专用训练（历史入口，仍可用）。 |
| `process.py` | ~250 行 | 弱标注流水线：清洗 → 关键词弱标注 → 切分。弱监督线的数据再生器。 |
| `predict_xlmr.py` | ~100 行 | 通用推理：任意 CSV → 标签 + 概率。应用落地入口。 |

### 3.2 工具脚本（`tools/`，15 个）— ★★

| 分组 | 脚本 | 价值 |
|------|------|------|
| ABSA 数据 | `prepare_absa_dataset.py` | 唯一正确解析 travel TSV 的入口（处理列错位/内嵌 tab）；分组切分 + 统计摘要。**任何 ABSA 数据操作都应经过它。** |
| 评估 | `evaluate_checkpoint.py` | 独立评估任意 checkpoint，不必重训 |
| 数据画像 | `analyze_uploaded_data.py` | 三份新数据的 EDA（输出 `results/uploaded_data_profile.json`） |
| Gold 闭环 | `extract_boundary_samples.py` → 人工标注 → `generate_boundary_review_summary.py` → `make_gold_patch_and_refined.py` → `annotation_loop_merge.py` | **完整的标注-修正-再训练闭环**，主动学习研究的基础设施；另有 `check_gold_file.py`/`check_gold_xlsx.py` 做格式校验、`split_gold3.py` 切分 |
| 实验自动化 | `run_and_fill_experiments.sh`、`monitor_and_complete_experiments.sh`、`fill_summary_template.py` | 一键跑对照实验并把指标填进报告模板 |
| 其他 | `unify_datasets.py`、`analyze_gold3_errors.py` | 数据格式统一；三分类错误分析 |

### 3.3 代码质量现状

- 三分类脚本共享逻辑已抽取（`training_utils.py`），重复代码已消除；
- 全部脚本 `py_compile` 通过、主要入口 `--help` 冒烟通过、无 linter 错误；
- 已知残留：无自动化测试（tests/），实验管理靠 shell 脚本而非实验追踪工具（当前规模下可接受）。

---

## 4. 结果与文档资产（全部入 Git）

### 4.1 实验结果（`results/`）— ★★★

| 结果组 | 文件 | 关键数值 | 论文角色 |
|--------|------|---------|---------|
| ABSA ASL（E3） | `absa_asl_v1_{metrics.json,report.txt}` | Micro 0.822 / Macro 0.773 / 负面聚合 0.705；含 14 标签逐项 P/R/F1 + support + 调优阈值 | **主结果** |
| ABSA Focal（E2） | `absa_focal_v1_*` | Micro 0.821 | 消融对比 |
| ABSA BCE（E1） | `absa_travel_v1_*` | Micro 0.766 | 基线锚点 |
| ABSA full BCE（E0） | `absa_xlmr_baseline_*` + 225KB 训练日志 | Micro 0.611 | 口径差异讨论 |
| 三分类对照 | `{weak_baseline,gold_only}_on_gold_*` | Macro 0.5656 vs 0.7125 | 动机章节 |
| 数据画像 | `uploaded_data_profile.json` | 三份新数据完整统计 | 数据章节素材 |

- **价值**：这些 JSON/TXT 是**所有已花费 GPU 时间的最终凝结**（合计十余次训练），
  且每份都带数据来源追踪字段，审稿可查。体积极小但价值最高，务必随 Git 保留。

### 4.2 文档（`docs/` 15 篇 + `gold/docs/` + 根 README）— ★★

| 类别 | 文件 | 价值 |
|------|------|------|
| 研究管理 | `research_plan.md`（7/7，E0–E6 矩阵 + 10 周计划）、`asset_inventory.md`（本文档） | 与老师沟通、交接的直接材料 |
| 运行手册 | `run_absa_baseline.md`（两口径说明 + 推荐参数 + TSV 解析坑）、`run_weak_on_gold.md`、`README_unify_datasets.md` | 复现指南；新人可按文档独立跑通 |
| 技术记录 | `TRAINING_EVAL_SUMMARY.md`、`train_hf_gpu_CODE_REVIEW.md`、`train_hf_gpu_DATA_INFO.md`、`data_unification.md` | 决策留痕（如已知参数风险、数据合并规则） |
| 报告模板 | `weak_vs_gold_summary_{template,filled,example}.md`、`experiment_log_template.md`、`weak_baseline_on_gold_results_template.md` | 中日双语汇报模板（filled 版含真实数值） |
| 标注体系 | `gold/docs/ANNOTATION_GUIDE_{CN,JP}.md` | Neutral/Negative 判定规则 + 12 reason_tag 体系；**标注标准本身是可复用的方法论资产** |
| 对照报告 | `gold/reports/comparison_weak_vs_gold{,_JP}.md`、`boundary_review_summary.{md,json}` | 前期结论的成文版本 |
| 入口 | 根 `README.md` | 双轨工作流 + 结构图 + 最佳结果表（6/25 重写，与现状一致） |

### 4.3 错误分析素材（`gold/metadata/`）— ★

- `error_analysis_all_predictions.csv`（176KB，全测试集预测记录）、`error_analysis_errors_only.csv`、`error_analysis_report.json`
- 三分类时代的错误分析产物；E5 做 ABSA 错误分析时可参考其字段设计。

---

## 5. 环境与基础设施资产

| 项 | 明细 | 级别 |
|----|------|------|
| GPU | 2 × NVIDIA RTX 6000 Ada（各 48G 显存）。GPU 0 带桌面进程，训练用 GPU 1（一次 ABSA 3-epoch 训练约 6 分钟 / 6.5G 显存，容量非常充裕） | ★★★ |
| 主 Python 环境 | `review-ai/.venv`：Python 3.10 + PyTorch（bf16/tf32）+ Transformers + datasets + scikit-learn + pandas 2.3，7.3G | ★★ |
| 冗余环境 | 根 `venv`：Python 3.11，7.4G。README 已不再引用 | ○ 可清理 |
| 版本控制 | Git（7/7 建立），`.gitignore` 策略：代码/文档/结果/小数据入库，venv/模型/可再生大数据排除 | ★★ |
| 依赖清单 | 根 `requirements.txt`（torch/transformers/datasets 等核心 8 项） | ★ |

---

## 6. 资产健康度与行动建议

| 维度 | 状态 | 建议 |
|------|------|------|
| 可复现性 | 好：切分脚本+seed+摘要入 Git，实验配置随 checkpoint 保存 | 每完成一个实验就 `git commit` 一次 |
| 存储 | 54G 中约 16G 为冗余（双 venv 之一 7.4G + 过时 ckpt 6–9G） | E4 跑完后统一清理 |
| 备份 | 尚无远程备份；最不可再生的是 gold 标注与 travel TSV（已入 Git 但仓库仍在本机） | 给 Git 加一个远程（GitHub 私库或内网 bare repo） |
| 未兑现资产 | 195 条边界样本（复核 0%）、筑波语料（未用） | 前者对应主动学习方向；后者可做 ABSA 外部验证 |
| 当前断点 | E4 数据就绪未跑；E5 错误分析未开始 | 按研究计划第 1 周执行 |
