# 日语酒店评论方面级情感分析 —— 弱监督预训练工程实施文档（路线 A）

> **状态（2026-10-12）**：本文档是 2026-07 的实施记录，保留作历史。当前的目录结构、复现命令与论文数字以根目录 `README.md`、`REPRODUCE.md` 与 `docs/thesis_numbers.md` 为准。旧三分类项目的脚本（含 `training_utils.py`、`tools/evaluate_checkpoint.py`）已移至 `review-ai/legacy/mercari_3class/`。

> 本文档面向 AI 编程助手（Cursor / Claude Code）与开发者本人，用于指导本项目的代码实现。
> 学术方向与实验设计见 `docs/research_plan.md`（路线 A：分项评分弱监督预训练）；本文档只关注**要写什么代码、按什么顺序写、验收标准是什么**。

---

## 0. 项目基本信息

- **项目根目录**：`192.168.10.199:/home/etl/gujincheng`
- **代码主目录**：`review-ai/`
- **Git**：已初始化；每完成一个 Phase/Task 提交一次
- **GPU**：2 × RTX 6000 Ada（48G），训练用 **GPU 1**（GPU 0 有桌面进程占用）
- **Python 环境**：`review-ai/.venv`，Python 3.10 + PyTorch(bf16/tf32) + Transformers + datasets + scikit-learn + pandas 2.3
- **模型底座**：XLM-RoBERTa-base（沿用现有训练框架）
- **主训练数据**：`gold/data/raw/01_Travel_Review.2024.tsv`（583,778 条，弱监督预训练）
- **评测锚点**：Gold ABSA `data/absa_{train,valid,test}.csv`（53,192 句，test 永不参与面板训练）
- **实验编号**：E1/E2（诊断，已完成）→ **E7/E8**（弱监督主线）→ E8-diag（子集评估）

---

## 1. 现有资产（直接复用，不要重新造轮子）

| 资产 | 路径 | 用途 |
|---|---|---|
| **主训练数据（路线 A）** | `gold/data/raw/01_Travel_Review.2024.tsv` | 583,778 条评论 + 6 分项评分，**E7 弱监督预训练语料** |
| 面板解析工具 | `tools/parse_travel_review_panel.py` + `tools/travel_review_panel_utils.py` | 解析 18 列 TSV（含内嵌 tab 修复） |
| 重叠检查工具 | `tools/check_panel_gold_overlap.py` | 面板 vs Gold 文本去重，输出剔除清单 |
| 面板画像工具 | `tools/profile_travel_review_panel.py` | 年度面板统计 |
| Gold 标注数据 | `gold/data/travel_aspect_sentiment.tsv` | 76,624 句，**唯一句子级 ABSA 评测标准** |
| 数据解析脚本 | `tools/prepare_absa_dataset.py` | Gold TSV 唯一正确解析入口 |
| Gold 切分 | `data/absa_{train,valid,test}.csv` | E1/E8 微调与评测 |
| 训练脚本 | `train_absa_baseline.py` | ABSA 多标签训练（E1 已完成；E8 在此基础上加 `--pretrained_encoder`） |
| 评估脚本 | ~~`tools/evaluate_checkpoint.py`~~（三分类用，已移至 legacy；ABSA 用 `tools/save_test_probs.py`） | 独立评估任意 checkpoint |
| 子集评估 | `tools/evaluate_by_source.py` | E2 / E8-diag 显式/隐式子集评估（✅ 已实现） |
| 显式/隐式切分 | `tools/split_explicit_implicit.py` + `data/absa_explicit_implicit_tags.csv` | Task 1（✅ 已完成） |
| 共享工具 | ~~`training_utils.py`~~（仅三分类脚本使用，已移至 legacy） | `WeightedTrainer` + 类权重逻辑 |
| 边界样本 | `legacy/mercari_3class/gold/data/reviewed/boundary_samples/`（195 条，Mercari，已移至 legacy） | Task 2 人工验证种子集 |
| E1 基线 | `checkpoints/absa_asl_v2_baseline/checkpoint-3984` | 统一对比起点（✅ 已完成） |
| 数据评估报告 | `docs/data_review_travel_review_2024.md` | 面板字段说明与使用注意 |

**明确不使用**：Mercari 弱标注/Gold 三分类数据、旧三分类 checkpoint。

**HotelMaster**（`gold/data/02_Travel_HotelMaster.tsv`）：本路线不用于训练，可选用于酒店聚合 demo。

---

## 2. 依赖

### 2.1 已安装

```bash
source review-ai/.venv/bin/activate
# oseti 0.4.3.1 — Task 1 显式/隐式切分用（✅ 已验证）
pip install oseti
```

### 2.2 面板数据许可

使用前确认 NII-IDR / 实验室数据协议。**原始 TSV 不得入 Git**：

```bash
# 确认 .gitignore 包含：
# review-ai/gold/data/raw/
```

---

## 3. 任务分解（按顺序执行）

### Phase 0｜面板数据就绪（阻塞 E7，当前最高优先级）

#### P0-1｜重叠检查

**已有脚本**：`tools/check_panel_gold_overlap.py`

```bash
python tools/check_panel_gold_overlap.py
# 输出：
#   data/panel_overlap_with_gold.json
#   data/panel_exclude_text_hashes.jsonl
```

**验收标准**：
- 报告 Gold 评论与面板的精确匹配数、重叠率；
- `panel_exclude_text_hashes.jsonl` 可用于后续过滤；
- **重叠率写入论文数据节**；在重叠检查完成前**禁止开始 E7**。

#### P0-2｜面板解析与训练池构建

**已有脚本**：`tools/parse_travel_review_panel.py`

```bash
python tools/parse_travel_review_panel.py --years 2024 --export_dir data/panel_parsed
```

**新建脚本**：`tools/build_panel_train_pool.py`

**逻辑**：
1. 读取 `data/panel_parsed/2024.csv`
2. 剔除 `panel_exclude_text_hashes.jsonl` 中的重叠行
3. 剔除 `review_text` 为空的行
4. 保留字段：`text_hash, hotel_id, posted_at, overall_rating, rating_1..6, review_text`
5. 面板内部 90/10 切分（`seed=42`）→ `panel_train.csv` / `panel_valid.csv`

**输出**：
- `data/panel_train_pool.csv`（或拆分为 train/valid 两个文件）
- `data/panel_pool_stats.json`（条数、剔除数、各分项有效评分分布）

**验收标准**：训练池条数 ≈ 58 万 − 重叠数；打印各分项 rating=0 占比与 1–2 分占比。

#### P0-3｜分项评分列 → 方面映射核验

**新建脚本**：`tools/verify_panel_rating_columns.py`

**逻辑**（半天工作量）：
1. 对每个 `rating_1..6` 列，抽取 1–2 分评论各 100 条
2. 用 `absa_asl_v1` 预测 14 标签，统计哪个方面负面命中率最高
3. 确认 6 列与 7 方面（サービス/立地/部屋/設備・アメニティ/風呂/食事）的对应关系
4. 顺带统计「低分但模型预测无负面」的比率（评分–文本不一致率）

**输出**：`data/panel_rating_column_map.json`

```json
{
  "rating_1": {"aspect": "サービス", "confidence": "high"},
  "rating_2": {"aspect": "立地", "confidence": "high"},
  ...
}
```

**验收标准**：6 列映射置信度 ≥ 5/6 为 high；不一致率数字写入论文。

---

### Task 1｜显式/隐式自动切分（✅ 已完成）

**文件**：`tools/split_explicit_implicit.py`  
**输出**：`data/absa_explicit_implicit_tags.csv`  
**报告**：`results/task1_explicit_implicit_report.json`

> 本 Task 为诊断动机服务，不阻塞 E7/E8，但 E8-diag 依赖其输出。

---

### Task 2｜人工验证子集（可选，与 Phase 2 并行）

**文件**：`tools/build_verification_sample.py`（待建）

- 抽样 300–500 条（含 195 边界样本 + implicit 分层 + explicit 对照）
- 输出 `gold/data/verification_sample_to_annotate.xlsx`
- 标注完成后跑 `tools/check_oseti_accuracy.py`

> 路线 A 不依赖此 Task 跑通 E7/E8，但可支撑 RQ5 与论文局限讨论。

---

### Task 3｜Baseline 重新训练（E1，✅ 已完成）

```bash
python train_absa_baseline.py \
  --train_file data/absa_train.csv \
  --validation_file data/absa_valid.csv \
  --test_file data/absa_test.csv \
  --labels_file data/absa_label_columns.json \
  --loss_type asl --asl_gamma_neg 4.0 --asl_clip 0.05 \
  --no_pos_weight \
  --tune_threshold \
  --output_dir checkpoints/absa_asl_v2_baseline \
  --experiment_name E1_baseline_rerun
```

**验收**：`results/E1_baseline_rerun_metrics.json`（Micro F1 0.824，✅ 通过）

---

### Task 4｜显式/隐式子集诊断（E2，✅ 已完成）

**文件**：`tools/evaluate_by_source.py`  
**输出**：`results/E2_explicit_vs_implicit_diagnosis.json`

**关键数字**：隐式标签对负面 F1 0.826 vs 显式 0.869（差距 −4.3 pp）

---

### Task 5｜弱监督训练集格式化（E7 前置）

**新建脚本**：`tools/prepare_panel_weak_labels.py`

**逻辑**：
1. 读取 `data/panel_train_pool.csv` + `data/panel_rating_column_map.json`
2. 为每条评论生成 6 个弱标签：
   - `rating_k == 0` → `mask_k = 0`（不参与损失）
   - `rating_k ∈ {1,2}` → `label_k = 1`（负面）
   - `rating_k ∈ {3,4,5}` → `label_k = 0`（非负面）
3. 输出 HuggingFace datasets 可读格式：`data/panel_weak_{train,valid}.jsonl`

**字段**：`text, label_1..6, mask_1..6, text_hash, hotel_id`

**验收标准**：打印各头有效样本数与正负比例；与 `panel_pool_stats.json` 一致。

---

### Task 6｜分项评分弱监督预训练（E7，核心）

**新建文件**：`train_panel_weak_pretrain.py`

**架构**：
- `XLMRobertaModel` 共享 encoder
- 6 个 `nn.Linear(hidden, 1)` 二分类头（对应 6 分项评分）
- 损失：6 头 masked BCE 均值（`mask=0` 的列跳过）

**训练命令**（初版）：

```bash
python train_panel_weak_pretrain.py \
  --train_file data/panel_weak_train.jsonl \
  --validation_file data/panel_weak_valid.jsonl \
  --rating_map data/panel_rating_column_map.json \
  --model_name xlm-roberta-base \
  --output_dir checkpoints/absa_weak_pretrain_v1 \
  --experiment_name E7_panel_weak_pretrain \
  --num_epochs 2 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5
```

**实现要点**：
- 流式 DataLoader（58 万条，不要一次性 load 到内存）
- 先跑 1000 条冒烟测试，再上全量
- 保存 **encoder only**（`save_encoder_only`）供 E8 加载
- `seed=42`

**验收标准**：
- `results/E7_panel_weak_pretrain_metrics.json` 落盘
- 面板 valid 上 6 分项平均 F1 > 0.60（弱标签基线，不要求很高）
- checkpoint 可被 E8 脚本正确加载

---

### Task 7｜Gold ABSA 微调（E8，核心）

**方案**：在 `train_absa_baseline.py` 基础上新增 `--pretrained_encoder` 参数，或新建 `train_absa_from_pretrain.py`（推荐后者，避免污染原脚本）。

**新建文件**：`train_absa_from_pretrain.py`

**逻辑**：
1. 加载 E7 的 encoder 权重到 `XLMRobertaModel`
2. 初始化 14 标签 ABSA 分类头（随机初始化，不继承 E7 的 6 头）
3. 在 Gold `absa_train.csv` 上微调，配置与 E1 完全对齐：
   - ASL（γ_neg=4.0, clip=0.05）
   - 无 pos_weight
   - 逐标签阈值调优
4. 支持 `--encoder_lr`（默认 1e-5）与 `--head_lr`（默认 2e-5）分层学习率

```bash
python train_absa_from_pretrain.py \
  --pretrained_encoder checkpoints/absa_weak_pretrain_v1/encoder \
  --train_file data/absa_train.csv \
  --validation_file data/absa_valid.csv \
  --test_file data/absa_test.csv \
  --labels_file data/absa_label_columns.json \
  --loss_type asl --asl_gamma_neg 4.0 --asl_clip 0.05 \
  --no_pos_weight \
  --tune_threshold \
  --encoder_lr 1e-5 --head_lr 2e-5 \
  --output_dir checkpoints/absa_weak_finetune_v1 \
  --experiment_name E8_weak_pretrain_finetune
```

**验收标准**：
- `results/E8_weak_pretrain_finetune_metrics.json` 落盘
- 与 E1 对比：**总体 Micro F1 不得下降超过 0.01**；若负面聚合 F1 提升 ≥ 0.02 即为成功
- 记录 E1 vs E8 差值到 results JSON 的 `delta_vs_E1` 字段

**消融**（同一脚本，不同参数）：
- A2a：全参数统一 lr 2e-5
- A2b：冻结 encoder，仅训头
- A3：E7 epoch=1 vs epoch=2

---

### Task 8｜隐式子集评估（E8-diag）

复用 `tools/evaluate_by_source.py`，指向 E8 checkpoint：

```bash
python tools/evaluate_by_source.py \
  --checkpoint_dir checkpoints/absa_weak_finetune_v1/checkpoint-XXXX \
  --experiment_name E8_implicit_subset_diagnosis
```

**输出**：`results/E8_implicit_subset_diagnosis.json`

**验收标准**：
- 产出 E1 vs E8 的显式/隐式/隐式负面子集 F1 三方对比表
- 若 E8 在隐式负面子集上 F1 提升 > E8 在总体上的提升 → RQ3 支持
- Bootstrap 95% CI（复用 E2 已有逻辑）

---

### Task 9｜LLM 少样本对照（E5，可选）

**文件**：`tools/llm_cot_baseline.py`（待建）

- 隐式子集 100–150 句
- 对比 E1 / E8 / LLM 在隐式子集上的 F1 与 token 成本

---

### 附录 Task｜对比学习（原 E3/E4，不在路线 A 必做范围）

若 E8 增益不足，可考虑：
- `contrastive_absa_model.py` + `train_contrastive_absa.py`
- `tools/build_contrastive_pairs.py`

详见旧版计划；当前**不优先实现**。

---

## 4. 目录结构约定（路线 A 新增文件）

```
review-ai/
├── tools/
│   ├── parse_travel_review_panel.py       # P0-2（已有）
│   ├── check_panel_gold_overlap.py        # P0-1（已有）
│   ├── build_panel_train_pool.py          # P0-2（待建）
│   ├── verify_panel_rating_columns.py     # P0-3（待建）
│   ├── prepare_panel_weak_labels.py       # Task 5（待建）
│   ├── split_explicit_implicit.py         # Task 1（✅）
│   ├── evaluate_by_source.py              # Task 4 / Task 8（✅）
│   ├── build_verification_sample.py       # Task 2（可选）
│   └── llm_cot_baseline.py                # Task 9（可选）
├── train_panel_weak_pretrain.py           # Task 6 / E7（待建）
├── train_absa_from_pretrain.py            # Task 7 / E8（待建）
├── train_absa_baseline.py                 # Task 3 / E1（✅）
├── data/
│   ├── panel_parsed/2024.csv
│   ├── panel_train_pool.csv
│   ├── panel_weak_{train,valid}.jsonl
│   ├── panel_rating_column_map.json
│   ├── panel_overlap_with_gold.json
│   ├── panel_exclude_text_hashes.jsonl
│   └── absa_explicit_implicit_tags.csv    # Task 1（✅）
├── checkpoints/
│   ├── absa_asl_v2_baseline/              # E1（✅）
│   ├── absa_weak_pretrain_v1/             # E7
│   └── absa_weak_finetune_v1/             # E8
└── results/
    ├── E1_baseline_rerun_metrics.json     # ✅
    ├── E2_explicit_vs_implicit_diagnosis.json  # ✅
    ├── E7_panel_weak_pretrain_metrics.json
    ├── E8_weak_pretrain_finetune_metrics.json
    ├── E8_implicit_subset_diagnosis.json
    └── figures/
```

---

## 5. 执行顺序与里程碑检查点

```
Phase 0：P0-1 重叠检查 → P0-2 训练池 → P0-3 列映射  （阻塞 E7）
    ↓
Task 5：弱标签格式化
    ↓
Task 6 / E7：面板弱监督预训练
    ↓
Task 7 / E8：Gold ABSA 微调
    ↓
Task 8 / E8-diag：隐式子集评估（vs E2）
    ↓
[决策点] E8 负面聚合 F1 比 E1 提升 ≥ 0.02？
    ├─ 是 → 论文主线成立，进入撰写
    └─ 否 → 分析弱标签噪声；尝试 A2/A3 消融；考虑附录对比学习
```

**已完成、不阻塞主线**：Task 1（✅）、Task 3/E1（✅）、Task 4/E2（✅）

**可并行**：Task 2 人工验证（与 E7 训练并行）

**每完成一个 Task/Phase**：`git commit`，格式 `[Task N] 描述 + 关键指标`

---

## 6. 给 Cursor / Claude Code 的执行提示

- **当前最高优先级**：Phase 0（重叠检查 → 训练池 → 列映射），完成前不要开 E7
- 面板训练脚本先用 **1000 条**冒烟，再上全量 58 万
- 复用现有代码风格（argparse CLI、json/csv 落盘、`{experiment_name}_metrics.json` 命名）
- E8 微调参数必须与 E1 对齐（ASL、无 pos_weight、阈值调优），否则对比无效
- 所有随机过程 `seed=42`
- `gold/data/raw/` 不入 Git；论文只报告统计，不公开原始 TSV
- E7 训练用流式 DataLoader；`per_device_train_batch_size=32` 起步，OOM 则减 batch 加 grad_accum

---

## 7. 附录：弱标签损失伪代码（E7）

```python
def masked_bce_loss(logits, labels, masks):
    # logits: (batch, 6), labels: (batch, 6), masks: (batch, 6) bool
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    per_head = loss_fn(logits, labels.float())  # (batch, 6)
    per_head = per_head * masks.float()
    # 每个样本按有效头数平均，再 batch 平均
    valid_counts = masks.float().sum(dim=1).clamp(min=1)
    per_sample = per_head.sum(dim=1) / valid_counts
    return per_sample.mean()
```

---

## 8. 附录：Anthropic API 调用示例（Task 9 / E5，可选）

```python
import anthropic

client = anthropic.Anthropic()  # 从环境变量读取 API key

def cot_predict(sentence, aspect, few_shot_examples):
    prompt = f"""以下は日本語のホテルレビュー文です。「{aspect}」という観点について、
まず判断理由を述べてから、最後に「ポジティブ」「ネガティブ」「該当なし」のいずれかで判定してください。

例:
{few_shot_examples}

対象文: {sentence}
"""
    response = client.messages.create(
        model=MODEL_NAME,  # 运行前用 client.models.list() 确认
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```
