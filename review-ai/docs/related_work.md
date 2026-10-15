# 相关工作（文献综述）

**撰写日期**：2026-07-20
**配套文档**：[`research_plan.md`](research_plan.md)（定学术方向）、[`data_spec_rakuten_travel.md`](data_spec_rakuten_travel.md)（数据仕様）
**核实说明**：本综述中标注 **[✓已核实]** 的文献均在撰写时通过检索确认了作者、venue、年份、页码等书目信息；标注 **[待核实]** 的文献凭已有知识记载，正式提交前需自行核对确切出处，避免以讹传讹。已知一处因数据库付费墙无法确认作者姓名，如实标注，不作猜测填充。

---

## 0. 综述结构与本研究的定位

本研究处于四条文献脉络的交汇处：**(1)** 方面级情感分析（ABSA）任务本身，**(2)** 弱监督/远程监督情感分析——即用评分代替人工标注，**(3)** 隐式情感分析——本研究诊断动机（RQ1）的理论基础，**(4)** 领域自适应预训练——本研究两阶段方法的直接方法论来源。此外还涉及类别不平衡损失、近似去重、统计显著性检验等工程方法学，以及面向未来工作的多示例学习（MIL）。**§4.4** 补充记录基座模型选择这一维度（补充实验线：日语专用 DeBERTaV3 对隐式的针对性提升）；**§9** 系统评估面向隐式识别改进的方法路线（SCAPT / THOR / 知识注入），并以「能否嫁接到现有流水线」为标准给出优先级建议。第 10 节给出本研究相对于最相关先行工作（DSPN、DAPT）的差异化定位。

---

## 1. 方面级情感分析（ABSA）：任务定义与日语语境

方面级情感分析作为标准任务由 **SemEval-2014 Task 4** 确立 **[✓已核实]**：

> Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., & Manandhar, S. (2014). SemEval-2014 Task 4: Aspect Based Sentiment Analysis. *Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014)*, Dublin, pp. 27–35.

该任务确立了「方面项抽取 + 方面级极性分类」的标准范式，本研究的 14 标签多标签设置（7 观点 × 正/负）是其在日语酒店评论域的延伸。

任务全景可参考近期综述 **[✓已核实]**：

> Zhang, W., Li, X., Deng, Y., Bing, L., & Lam, W. (2023). A Survey on Aspect-Based Sentiment Analysis: Tasks, Methods, and Challenges. *IEEE Transactions on Knowledge and Data Engineering*, 35(11), 11019–11038.（arXiv 预印本 2022年3月，arXiv:2203.01054）

该综述系统梳理了 ABSA 从传统特征工程到 BERT 时代端到端方法的演进，可作为本论文「相关工作」章节方法演进部分的骨架参考。

### 1.1 日语 ABSA 与 Gold 数据集的直接来源 ★核心引用

日语因缺乏标准数据集，ABSA 研究长期滞后于英语。本研究所用的 Gold ABSA 数据集（53,192 句、7 观点 × 正/负）**直接来自以下工作** **[✓已核实，重要发现]**：

> Nakayama, Y., Murakami, K., Kumar, G., Bhingardive, S., & Hardaway, I. (2022). A Large-Scale Japanese Dataset for Aspect-based Sentiment Analysis. *Proceedings of the Thirteenth Language Resources and Evaluation Conference (LREC 2022)*, Marseille, pp. 7014–7021.

该论文明确报告「12,000 条楽天トラベル酒店评论」「53,192 句」「7 个观点类别（食事/夕食/朝食/立地/設備/部屋/風呂 等）」——这些数字与本项目全程使用的 Gold 数据集规模**逐一吻合**，可确认为同一数据集的原始发布论文。该论文亦指出 BERT 类上下文模型在该任务上效果显著，为本研究选用 XLM-R 作为基座提供了先例支持。

值得注意的是，该论文第一作者 Nakayama, Y. 与本研究面板数据的来源论文（见 §1.2）第一作者相同，说明 Gold 标注数据与 700 万条弱监督面板数据出自同一楽天技術研究所团队的连贯数据管线——这是本研究「弱监督预训练 + Gold 微调」两阶段设计在数据层面天然成立的基础。

#### 1.1.1 原论文的基线成绩与可比性（2026-10-08 读原文补记）

原论文 §4 在 80:10:10 划分（按 14 标签分布分层）上比较了 BERT-MLC（cl-tohoku bert-japanese 旧版，14 维 sigmoid，阈值固定 0.5）、BERT-SPC（句子-方面对分类）与 Attention-LSTM。Table 6 报告的 14 标签 F1（论文称 Macro F1）：

| 模型 | 14 标签平均 | 负面各标签 |
|---|---|---|
| BERT-SPC | 0.692 | 朝食 0.614 / 夕食 0.545 / 風呂 0.634 / サービス 0.630 / 立地 0.638 / 設備 0.656 / 部屋 0.579 |
| **BERT-MLC（最好）** | **0.711** | 朝食 0.680 / 夕食 0.658 / 風呂 0.689 / サービス 0.675 / 立地 0.623 / 設備 0.651 / 部屋 0.609 |

论文也把错误原因归为零形式照应、相近类别混淆、以及「日本文化导致的隐式负面意见」——与本研究 RQ1 的问题设定直接对应。

**可比性（写论文时必须注明）**：
1. 官方发布的 `travel_aspect_sentiment.tsv` **不含划分**；本研究按 review_id 随机划分（同为 80:10:10），测试集不同。
2. 本研究在验证集上逐标签调阈值并使用 ASL，原论文为固定阈值 0.5 与 BCE。
3. 原论文未明确每标签 F1 的计算方式（从数值看应为正类二分类 F1，与本研究一致，但无法完全确认）。

因此与原论文数字只作**参考比较**（本研究 14 标签平均：XLM-R E1 0.776 → DeBERTa+E7/E8 0.801）。**同一划分上的公平比较**：本研究的 Tohoku-v3 E1 即原论文 BERT-MLC 方法（同系列模型的新版），可作为「先行手法の再実装」基线（14 标签平均 0.789）。

#### 1.1.2 使用同一语料的后续研究（2026-10-08 检索，数字已对照原文核实）

检索范围：Google Scholar 被引列表（13 件）、按语料名「アスペクト・センチメントタグ付きコーパス」的全文检索、J-STAGE、言語処理学会論文集、CiNii。**没有一项研究沿用原论文的划分**，均为各自划分或筛选，数字只能作参考比较。

| 研究 | 任务与数据设定 | 报告的成绩 |
|---|---|---|
| 中山ら，NLP2021 P2-12（原论文日文版） | 14 标签多标签，8:1:1，BERT | macro-F 0.68 / micro-F 0.72 |
| Nakayama et al., LREC 2022 | 14 标签，80:10:10 分层，BERT-MLC | 14 标签平均 0.711（见 §1.1.1） |
| 味岡・岡田・森，JSAI 2024 2G6-GS-6-02 | 14 标签；76,624 句中筛出 48,353 句（去掉无标签句、同一方面正负并存句等），30,000 训练 / 10,000 测试，5 折 | micro-F1：Mpm+T 0.775、XMpm+T 0.765 |
| 西元・森・岡田，JSAI 2025 3G4-GS-6-03 | 同上筛选与划分 | micro-F1：改良 Mpm+T 0.782、Mpm+T 0.777、TIES-Merging 0.748 |
| 林・浦野，JSAI 2026 1Yin-B-28 | 14 标签，LUKE vs BERT，划分未写明 | Pos/Neg/「Macro」F1：BERT 0.74/0.63/0.69，LUKE 0.75/0.66/0.71（原文称 Macro，实为**加权平均**） |

本研究（自有划分，review_id 单位 80:10:10；与 JSAI 2024/2025 一样**去掉了无标签句**（53,192 句，`--drop_unlabeled`），但**保留**同一方面正负并存的句子；多 seed 平均）：14 标签平均 F1 0.776（XLM-R E1）→ 0.801（DeBERTa+E7/E8）；micro-F1 0.824 → 0.841。基线 E1 已高于上述报告值，差距主要来自更新的底座模型与逐标签调阈值，**不可归因于本研究的方法**；方法的贡献只用同一划分内的比较（E1 vs E8）论证。

另：Kusumoto・Okada・Mori（AROB 2023, pp.1146–1149）为 Mpm+T 原论文，全文未能获取；北大 Zhang ら（NLP2023 A3-2）明言该语料非自由公开而改用 chABSA；Hanafusa et al.（LREC 2026）为另一个新建的日语 ABSA 数据集（基于 WRIME），不使用本语料。

### 1.2 面板数据来源

> Nakayama, Y., Hikichi, K., Liu, Y. C., & Hirate, Y. (2025). Rakuten Data Release: A Large-Scale and Long-Term Reviews Corpus for Hotel Domain. *arXiv:2512.15151*. **[✓已核实]**

该论文报告面板语料含 **729 万条**跨 16 年（2009–2024）的楽天トラベル评论，并**明确分析了 2019–2024 年间的 data drift（数据漂移）**——这与本研究在 §6 规模/年份分析中独立发现的「近年数据优于早年数据」现象直接呼应，构成了外部文献对本研究核心发现的支持证据（详见 §4.3 与 §9 的讨论）。


### 1.3 同一平台上的最新日语 ABSA 评测（2026-10-08 补记）

> Yu, L.-C., Becker, J., Muhammad, S. H., et al. (2026). SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DimABSA). *arXiv:2604.07066*. **[✓已核实]**

多语言共享任务，其中 `jpn-hot` 数据集为**新爬取的楽天トラベル评论**，标注为效价–唤醒度（valence–arousal）连续值，以 RMSE 评估。说明日语酒店 ABSA 仍是活跃的研究对象；但任务形式（连续值回归）与数据均不同于本研究（7 方面 × 正负的多标签分类），**不可直接比较数字**，仅在相关工作中提及。

同日的补充检索（Google Scholar 被引列表、J-STAGE、言語処理学会論文集等）找到了使用本语料的后续研究，见 §1.1.2。

---

## 2. 弱监督 / 远程监督情感分析：把评分当标签

### 2.1 方法论谱系起点

将「廉价但间接的信号」当作训练标签这一范式，最早在关系抽取领域被系统化 **[✓已核实]**：

> Mintz, M., Bills, S., Snow, R., & Jurafsky, D. (2009). Distant supervision for relation extraction without labeled data. *Proceedings of the Joint Conference of the 47th Annual Meeting of the ACL and the 4th IJCNLP*, Suntec, pp. 1003–1011.

该文提出的「远程监督」（distant supervision）概念——用已有知识库自动标注文本——是本研究「用面板评分自动生成方面级弱标签」这一设计思路的方法论源头，尽管领域不同（关系抽取 vs. 情感分析）。

情感分析领域最早的远程监督实践之一 **[✓已核实]**：

> Go, A., Bhayani, R., & Huang, L. (2009). Twitter Sentiment Classification using Distant Supervision. *Stanford CS224N Project Report*, pp. 1–12.

该工作用推文中的表情符号（😊/😞）作为免费弱标签训练情感分类器，是「用户生成的伴随信号 → 弱标签」这一思路在情感分析中的早期范例，本研究的评分弱标签可视为其在结构化评分场景下的延伸。

### 2.2 评分作为弱标签的已知局限

评分与文本情感不一致的问题在多篇工作中被讨论：用户评分可能被单一维度主导，而文本涵盖多个方面，导致评分-文本错位——这正是本研究 Phase 0 中「評分–文本一致性」核验（P0-3/P0-4）以及数据质量讨论章节的直接动机。

### 2.3 最相关的先行工作：DSPN ★核心对比对象

与本研究方法**最为接近**的先行工作是：

> Li, W., Chen, Y., Zheng, S., Wang, L., & Lalor, J. (2024). Stars Are All You Need: A Distantly Supervised Pyramid Network for Unified Sentiment Analysis. *Proceedings of the 9th Workshop on Noisy and User-generated Text (W-NUT 2024)*.（arXiv:2305.01710） **[✓已核实]**

DSPN 同样**仅用星级评分**作为弱监督信号，通过金字塔结构在词/方面/文档三个层级联合建模，端到端完成方面类别检测（ACD）、方面级情感分类（ACSA）与评分预测三个任务，在英语和中文的多方面评论数据集上取得了与有监督基线相当的效果。

**本研究与 DSPN 的关键差异**（详细定位见 §9）：DSPN 是**单阶段、端到端**的弱监督分类器，直接在评分弱标签上训练并评测；本研究采用**两阶段迁移学习范式**——先用评分弱标签对 encoder 做预训练，再在独立的高质量 Gold 标注集上微调至句子级 14 标签任务，且额外系统性研究了预训练数据的**规模效应**与**年份（时间域）效应**，这在 DSPN 及相关弱监督 ABSA 工作中均未见系统讨论。


### 2.4 同一数据上用评分辅助 ABSA 的先行研究 ★必须引用并区分

> 小泉さやか（2025）．文書と属性に対する感情分析のマルチタスク学習に関する研究．北陸先端科学技術大学院大学 修士論文（指導教員：白井清昭）．https://dspace.jaist.ac.jp/dspace/bitstream/10119/20050/5/paper.pdf **[✓已核实全文]**

用楽天トラベルの**総合評価**（星 1–2＝否定 / 3＝中立 / 4–5＝肯定）构造文档级情感分类，作为多任务学习的辅助任务来帮助本语料上的方面情感分类（每方面 3 分类、类别均衡抽样 1 万条、7:2:1）。主要结论：**训练数据充足时多任务学习无显著提升**（7 方面平均正确率 0.815 vs 单任务 0.814），仅在训练数据减到 1/10 时有效（0.750 vs 0.720）。该文也构造了 6 个方面评分的数据，并指出「评分与正文不一致」「正文未提及该方面却打 5 星」等问题，与本研究 RQ4 的观察一致。

| | 小泉（2025） | 本研究 |
|---|---|---|
| 评分信号 | 総合評価（每条评论 1 个极性） | 6 个方面评分（每条评论 6 个信号） |
| 规模 | 1 万条 | 58 万–676 万条 |
| 用法 | 多任务联合训练 | 弱监督预训练 → Gold 微调 |
| 评测 | 每方面 3 分类的均衡子集 | 标准 14 标签多标签，全量 Gold |
| 训练数据充足时 | 无提升 | 负面 F1 +1.60pp（显著，E8 3 次全部高于 E1 7 次） |

与本研究自身的对照一致：只用文档级极性的 SCL 在总体负面 F1 上无提升（§9），总合头无效（A4）。即**方面特异的信号 + 规模**是评分弱监督起作用的关键，可作为论文贡献的定位依据。

---

## 3. 隐式情感分析：为什么负面识别难（RQ1 的理论基础）

### 3.1 隐式 ABSA 作为独立研究方向

隐式方面情感（无显式情感词的方面表达）近年成为 ABSA 领域的重要子方向，近期系统综述指出该方向研究量持续增长：

> "Implicit Aspect-Based Sentiment Analysis: A Systematic Review." *ACM Computing Surveys*. DOI: 10.1145/3799234. 系统综述 2019–2024 年间 85 篇同行评审文献。**[部分确认——DOI/期刊/年份范围已核实，作者姓名因数据库访问限制未能确认，请勿在未核实前引用作者名]**

该综述的存在与规模（85 篇文献覆盖 5 年）本身即可作为「隐式 ABSA 是活跃研究方向」的证据，支持本研究 RQ1（隐式标签难度量化）的选题动机。

### 3.2 本研究隐式/显式切分所用工具的技术来源

Task 1（显式/隐式自动切分）所用的 `oseti` 工具 **[✓已核实]**：

> Ikegami, Y. `oseti`: Dictionary-based Sentiment Analysis for Japanese. GitHub: ikegami-yukino/oseti. MIT License.

其判定逻辑基于两部日语情感极性词典，本研究亦需明确标注这两部词典的原始出处：

> Takamura, H., Inui, T., & Okumura, M. (2005). Extracting Semantic Orientations of Words using Spin Model. *Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL 2005)*.（名詞編，スピンモデルによる意味的極性抽出）**[✓已核实]**

> 小林のぞみ、乾健太郎、松本裕治、立石健二 他 (2005)．意見抽出のための評価表現の収集．『自然言語処理』第12巻第3号，pp. 203–222．（用言編）**[✓已核实]**

Task 1 的输出（`data/absa_explicit_implicit_tags.csv`）依赖 oseti 的 `analyze_detail()` 判定每个活跃标签对是否命中情感词典——这一工具链的可靠性直接影响 RQ1/RQ3/RQ5 结论的稳健性。**该信度验证（Task 2）经与指导者/研究者确认后明确列为本研究的局限（limitation），不在本轮实验中执行**：验证样本（400 句分层抽样）与打分脚本（`tools/check_oseti_accuracy.py`）均已备妥，但真人标注环节的本质要求独立于 oseti 的判断作为 ground truth——若由 AI 系统代为标注，将不再是「人工核验」，而是变成另一个不同性质的实验（LLM 与 oseti 的一致性比较）。因此本研究选择诚实记录这一未验证环节，而非以自动化方式冒充人工验证。此决定与结论应在论文第 7 章「结论与局限」中明确说明。

---

## 4. 领域自适应预训练：两阶段方法的直接灵感 ★核心引用

### 4.1 方法论直接来源

本研究「面板弱监督预训练 → Gold 微调」两阶段框架的**直接方法论灵感**来自：

> Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*. **[✓已核实]**

该论文系统验证了「继续预训练」（continued pretraining）在四个领域（生物医学、计算机科学文献、新闻、**评论**）、八个分类任务上的增益，证明领域自适应预训练（DAPT）与任务自适应预训练（TAPT）均能带来显著且稳健的性能提升，尤其在低资源场景下更明显。**评论（reviews）恰是该论文验证的四个领域之一**，与本研究的酒店评论场景直接对应。

### 4.2 基座编码器

> Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *Proceedings of ACL 2020*, pp. 8440–8451.（XLM-RoBERTa） **[✓已核实]**

本研究 E1/E7/E8 全部以 `xlm-roberta-base` 为基座编码器，其多语言预训练特性适配日语文本处理。

### 4.4 日语专用基座模型（补充实验线，2026-07-23/24）★新增

主线全程使用多语言模型 XLM-R。补充实验线（详见补充研究报告）在严格控制变量（同代码・同超参・同损失，仅替换 `--model_name`）的条件下检验日语专用预训练模型是否更适配隐式情感识别，涉及以下模型：

> He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. *Proceedings of ICLR 2023*, Kigali.（arXiv:2111.09543） **[✓已核实]**

> He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. *Proceedings of ICLR 2021*.（arXiv:2006.03654） **[✓已核实]**

DeBERTaV3 用替换词检测（RTD）取代 MLM，并以「梯度解耦的嵌入共享」解决 ELECTRA 判别器/生成器的 tug-of-war 问题，是近年判别式预训练的代表架构。本研究使用的日语专用 DeBERTaV3 由京都大学黑桥研究室发布：

> Kurohashi-Chu-Murawaki Lab, Kyoto University. `ku-nlp/deberta-v3-base-japanese`. HuggingFace Model Hub.（模型发布，无独立论文；架构见上 DeBERTaV3） **[模型发布——引用 model card，勿虚构论文]**

作为次要候选比较的东北大学 BERT：

> Tohoku NLP Group. `cl-tohoku/bert-base-japanese-v3`. HuggingFace Model Hub / GitHub.（形态素级分词 + 纯日语语料 CC-100 ja + Wikipedia ja，模型发布） **[模型发布——引用 model card，勿虚构论文]**

**补充实验线的核心发现**：在直接对应研究目标的隐式子集负面 F1 上，DeBERTaV3-japanese 相对 XLM-R 基线取得 5-seed × 配对 bootstrap 硬显著提升（+0.98pp，P=0.994）；机制分析表明该优势系统性集中在表层线索最少的纯隐式句/短句上；叠加实验进一步证明日语专用基座与评分弱监督（E7/E8）**互补可叠加**——两者合计使负面 F1 从 XLM-R 基线的 0.7066 提升至 0.7447（+3.81pp，全项目最高）。这一发现将 §4.1 的 DAPT 框架从「继续预训练的数据」维度扩展到「基座模型选择」维度：对目标语言的专用预训练本身即是一种领域适配。

### 4.3 本研究对 DAPT 框架的方法论延伸

DAPT 原论文的「领域」指**主题域**（biomedical vs. news vs. reviews 等）。本研究第 6 节的规模/年份分析意外发现了一种 DAPT 原论文未直接讨论的漂移形式：**同一主题域内的时间域（temporal domain）漂移**——即同为「酒店评论」主题，但 2009 年与 2024 年的文本因用户表达习惯、评分标准演变而产生分布差异，导致早年数据（2009–2013）的预训练价值仅为近年数据（2021–2024）的约一半（+1.54pp vs. +3.12pp，详见 §6）。这一发现与 §1.2 引用的 Rakuten 面板数据论文中「分析 2019–2024 data drift」的官方结论相互印证，可视为对 DAPT 理论框架在时间维度上的一个具体化延伸讨论。

---

## 5. 多标签分类中的类别不平衡

本研究负面标签的稀缺性（负例仅占正例的 31%）是多标签分类中典型的类别不平衡问题，相关损失函数设计如下：

> Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *Proceedings of ICCV 2017*, Venice, pp. 2999–3007. **[✓已核实]**

Focal Loss 首次系统提出通过动态降权简单样本来缓解极端类别不平衡，是不对称损失设计思路的奠基工作，本研究代码中实现该损失作为可选对照（`--loss_type focal`）。

E1/E8 主线所用的损失函数：

> Ridnik, T., Ben-Baruch, E., Zamir, N., Noy, A., Friedman, I., Protter, M., & Zelnik-Manor, L. (2021). Asymmetric Loss for Multi-Label Classification. *Proceedings of ICCV 2021*. **[✓已核实]**

ASL（Asymmetric Loss，γ_neg=4.0, clip=0.05）针对多标签场景中「正标签少、负标签多」的不对称性设计，通过对负样本施加更强的聚焦参数与硬阈值裁剪来抑制易分类负样本的梯度贡献，与本研究 14 标签 ABSA 中「非活跃标签占绝大多数」的场景高度契合。E7 弱监督预训练中使用的 per-head pos_weight（√(neg/pos)）与 ASL 思路同源，均是对正负样本不平衡的损失层面补偿。

---

## 6. 数据处理方法论：近似重复检测

面板与 Gold 数据集之间的近似重复检测（`tools/approx_dedup_panel_gold.py`）基于 MinHash + LSH 技术：

> Broder, A. Z. (1997). On the Resemblance and Containment of Documents. *Proceedings of Compression and Complexity of Sequences (SEQUENCES 1997)*, Positano, pp. 21–29. **[✓已核实]**

MinHash 算法将文档相似度问题转化为可高效随机采样估计的集合交集问题，最早被用于 AltaVista 搜索引擎的近重复网页过滤。

> Indyk, P., & Motwani, R. (1998). Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality. *Proceedings of STOC 1998*, pp. 604–613. **[✓已核实]**

局部敏感哈希（LSH）提供了本研究实现中「分桶（band）索引 + 候选筛选」策略的理论基础，使得 676 万条面板评论与 12,476 条 Gold 评论的近似匹配可在合理时间内完成（避免暴力两两比较）。

---

## 7. 统计显著性检验

主结果（A1：E1 vs. E8）的显著性检验采用配对 bootstrap 方法，其在 NLP 领域的标准应用参考：

> Koehn, P. (2004). Statistical Significance Tests for Machine Translation Evaluation. *Proceedings of EMNLP 2004*, Barcelona, pp. 388–395. **[✓已核实]**

该论文验证了即使在小规模测试集（300 句量级）上，bootstrap 重采样方法也能可靠判断两系统性能差异是否具有统计意义——这与本研究 Gold test（5,356 句）的规模量级相当，为采用该方法提供了直接先例。

Bootstrap 方法本身的统计学原典：

> Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. *The Annals of Statistics*, 7(1), 1–26. **[✓已核实]**

---

## 8. 面向未来工作：多示例学习（MIL）

本研究 §7/§8 指出的「评论级弱标签 vs. 句子级 ABSA」粒度错配问题（朝食/夕食标签零增益即为实证），可参考多示例学习框架进行根治，相关奠基性文献：

> Dietterich, T. G., Lathrop, R. H., & Lozano-Pérez, T. (1997). Solving the Multiple Instance Problem with Axis-Parallel Rectangles. *Artificial Intelligence*, 89(1–2), 31–71. **[✓已核实]**

该文首次系统提出「包（bag）级标签、实例（instance）级隐变量」的多示例学习范式，与本研究「评论（bag）级评分、句子（instance）级情感」的结构完全对应。

> Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. *Proceedings of ICML 2018*, PMLR 80, pp. 2127–2136. **[✓已核实]**

该文提出基于注意力机制的可学习聚合算子，使 MIL 具备良好的可解释性——若应用于本研究场景，可望直接可视化「模型认为评论中哪些句子对应哪个方面的负面评分」，兼具方法价值与论文图表价值。

---

## 9. 面向隐式识别改进的方法路线（未来工作的文献基础）★新增

本研究的核心瓶颈已被机制分析精确定位：两个模型都漏检的约 420 个「纯隐式」负例——无否定形态、无转折/遗憾标记、无冗余语境的短句。针对这一瓶颈，文献中存在三条成体系的方法路线。以「能否嫁接到本研究现有两阶段流水线」为标准评估如下。

### 9.1 SCAPT：监督对比预训练 ★与本研究流水线同源，最高优先级

> Li, Z., Zou, Y., Zhang, C., Zhang, Q., & Wei, Z. (2021). Learning Implicit Sentiment in Aspect-based Sentiment Analysis with Supervised Contrastive Pre-Training. *Proceedings of EMNLP 2021*, pp. 246–256. **[✓已核实]**（[代码开源](https://github.com/Tribleave/SCAPT-ABSA)）

**为什么高度相关**：SCAPT 的预训练语料与本研究**同源**——它从 Yelp/Amazon 评论中抽取五星/一星评论，用星级作为正/负弱标签（Yelp 156 万句 + Amazon 51 万句），这正是本研究面板评分→弱标签 pipeline 的英语对应物。二者都属「评分弱监督预训练 encoder」范式，SCAPT 恰好填充本研究 E7 的同一槽位。

**方法要点**（三个联合预训练目标，L^pre = L^sup + α·L^rec + β·L^map，α=β=1，backbone 为 BERT-base）：
1. **监督对比学习（SCL）**：把「隐式情感表达」的句子表征拉向「同极性的显式表达」、推开异极性——这一步直接教模型「即使没有情感词也能对齐到正确极性」，是针对隐式的核心机制；
2. **评论重构（RR）**：Transformer 解码器从编码表征重构整句（去噪自编码）；
3. **掩码方面预测（MAP）**：遮蔽方面词并预测（方面词 80% 遮蔽 + 随机遮蔽至总 15%）。

**报告效果**（SemEval-2014，隐式 ISE 子集 vs 显式 ESE 子集，准确率）：Restaurant ISE 63.67→**72.28（+8.61）**、ESE 92.73→94.37（+1.64）；Laptop ISE 68.39→**77.59（+9.20）**、ESE 83.84→84.70（+0.86）。**即隐式大涨、显式几乎不动——正是本研究追求但 masked-BCE 版 E7 未能实现的「gap 缩小」模式。**

**嫁接评估**：
- ✅ 数据已具备：面板评论 + 评分（可按 SCAPT 方式抽取极端评分作对比学习的极性锚点）；
- ✅ 框架槽位对应：SCAPT 产出一个 encoder，可直接进入本研究 E8 微调阶段，可与 DeBERTaV3-japanese 基座组合；
- ✅ 本研究已有的按方面评分（6 观点）比 SCAPT 的文档级单极性更细，理论上可做**方面条件化对比学习**（比原版 SCAPT 更强）；
- ⚠️ 需新增组件：对比学习需要构造正/负样本对与温度超参；RR 需额外 Transformer 解码器；MAP 需方面词表。工程量中等。
- ⚠️ 任务差异：SCAPT 为经典「给定方面判极性」单标签设定，本研究为 14 标签多标签存在性检测——SCL 的表征层目标可迁移（这也是推荐它的原因），但评测协议与头部结构需适配。

**一句话定位（原假设）**：SCAPT ≈「一个专门为隐式情感设计的 E7 替代方案」，把 E7 的训练目标从 masked-BCE 预测换成监督对比对齐。

**★实测结果（2026-07-24，已否定"替代"假设）**：本研究实现了 SCAPT 核心的监督对比损失（SCAPT-core SCL，文档级极性锚点 = 総合评分，类别平衡 batch，不含 RR/MAP 辅助目标），在完全相同的控制条件下（DeBERTa 基座、近年 2M 池、E8 协议、3 seed）与 E7-BCE 对比：
- **SCL 不是上位替代，反被 BCE 严格支配**。整体负面 F1：SCL 0.7259 vs BCE 0.7447，配对 bootstrap Δ**−1.87pp（CI[−2.84,−0.93]，显著为负）**；且 SCL 相对"无预训练"的整体增益 ≈0（Δ−0.04pp，n.s.）。
- **但 SCL 的核心思想确实起作用**：隐式负面 F1 上 SCL 显著优于无预训练（+1.10pp，P=1.000），忠实于 SCAPT 设计初衷。只是 BCE 在隐式上做得**不比 SCL 少**（+1.44pp），还额外拿下了 SCL 完全抓不到的整体负面 +1.86pp。
- **机制解释**：SCL 用文档级单极性锚点，每条评论仅 1 比特监督；E7-BCE 用 6 个按方面的头，提供 6 比特（哪些方面偏负）。整体负面 F1 依赖这种按方面特异信号（§4.4 已证的"方面覆盖"机制），SCL 的单极性对齐无法复现。这解释了为何 SCAPT 在英语单标签 SemEval 上的惊艳数字（ISE +8~9pp）**未迁移到本研究的多标签设定**——任务结构差异（多标签方面存在性检测 vs 单方面极性）是关键。
- 结果文件：`results/scl_vs_bce_3seed_bootstrap.json`；实现：`train_panel_scl_pretrain.py`。**教训**：文献的强结果需在本任务设定下实测验证，不能先验假定迁移。

### 9.2 THOR：链式推理 ★攻最难层，但范式不同

> Fei, H., Li, B., Liu, Q., Bing, L., Li, F., & Chua, T.-S. (2023). Reasoning Implicit Sentiment with Chain-of-Thought Prompting. *Proceedings of ACL 2023 (Vol. 2: Short Papers)*, Toronto, pp. 1171–1182.（arXiv:2305.11255） **[✓已核实]**（[代码开源](https://github.com/scofield7419/THOR-ISA)）

三跳推理（THree-hop Reasoning）：先推隐含的方面 → 再推隐含的观点 → 最后推极性，模拟人类多跳常识推理。关键优势是**不必依赖巨型 LLM**——论文用 Flan-T5 微调 + CoT 三跳，比 zero-shot 至少提升 5%。适合攻本研究那 420 个「需常识推理而非模式匹配」的最难纯隐式句。**嫁接评估**：范式为生成式推理，与本研究判别式多标签差异较大，适配成本高于 SCAPT；更适合作为独立的对比方法或最难层的专项实验，而非主流水线替换。

### 9.3 图结构 / 外部知识注入（2024–2025）

近期多项工作向模型注入外部情感知识以「点亮」隐式信息：SenticNet 情感强度增强、心理学知识图注意力网络、多提示图融合等（见 §3.1 综述所归类）。这类方法可作为叠加组件，但相对零散、缺乏统一框架，优先级低于 SCAPT。相关综述已系统归类这些方向（§3.1 的 ACM Computing Surveys 隐式 ABSA 综述），写论文相关工作章节时可据其分类坐标定位本研究。

### 9.4 小结：推荐路线

**原假设**：以投入产出比排序，SCAPT（§9.1）为首选——数据同源、槽位对应、直击隐式且已有开源实现。**★但已实测否定（2026-07-24）**：SCAPT-core SCL 在本研究多标签设定下**被现有 E7-BCE 严格支配**（整体负面 −1.87pp 显著为负，隐式上无优势），原因是文档级单极性锚点丢失了 BCE 的按方面特异信号（详见 §9.1 实测结果）。**修正后的推荐**：(1) 现有 E7-BCE 已是合理选择，无需替换；(2) 若仍要探索 SCL 的隐式针对性，正确方向是**方面条件化对比**（用 6 观点评分而非单一総合评分做对比锚点，理论上可兼得方面特异 + 极性对齐）或 **BCE+SCL 联合目标**（互补而非替代），而非原版文档级 SCL；(3) THOR（§9.2）作为最难层的专项探索仍有价值。均属未来工作候选。

---

## 10. 本研究在文献中的定位（综合小结）

综合以上四条脉络，本研究可定位为：

1. **相对于 DSPN（§2.3）**——同属「评分作为唯一弱监督信号」的方法论谱系，但本研究采用**两阶段迁移学习**而非单阶段端到端训练，代价是需要独立的高质量微调集（Gold ABSA），换来的是可以复用已有的、更细粒度的句子级评测基准，并可对**预训练数据规模与时间跨度**做系统消融——这是 DSPN 论文未涉及的维度。

2. **相对于 DAPT（§4）**——本研究将「继续预训练」的对象从「同主题域的无标注文本」（DAPT 原设计）替换为「同主题域、结构化评分弱监督」，即在 DAPT 的 domain-adaptive pretraining 基础上叠加了一层弱监督任务信号。同时，本研究发现的「时间域漂移」现象（§4.3）为 DAPT 的「领域」概念提供了一个更细粒度的观察角度，且与 Rakuten 官方语料论文（§1.2）的 data drift 分析相互印证。

3. **相对于隐式 ABSA 综述（§3.1）**——本研究通过 RQ1/RQ3 的诊断实验为该领域提供了一个**具体的、量化的负例**：证明「评论级弱监督预训练」这一直觉上合理的方法**并不能**缩小隐式-显式性能差距（gap 从 4.32pp 扩大至 4.68pp），机制上指向粒度错配而非隐式表达本身——这一负结果本身是对该综述所述研究方向的有益补充。**需注意**：该结论的「隐式/显式」切分本身依赖 oseti 工具，其信度未经独立人工核验（§3.2 已述局限），论文中应对此保持相应的表述克制，避免过度断言切分本身的绝对准确性。

4. **方法论工程层面**——ASL（§5）、MinHash/LSH（§6）、paired bootstrap（§7）均为成熟技术在新场景（日语酒店 ABSA 弱监督流水线）下的**组合应用**，其组合方式（尤其是"评分列语义核验" + "多年跨度去重" + "规模/年份联合消融"的完整数据工程流水线）是本研究在方法论工程层面的贡献，而非单点技术创新。

---

## 参考文献总表（按引用顺序）

| # | 文献 | 章节 | 状态 |
|---|------|------|------|
| 1 | Pontiki et al., SemEval-2014 Task 4 | §1 | ✓已核实 |
| 2 | Zhang, Li, Deng, Bing, Lam, IEEE TKDE 2023 (arXiv:2203.01054) | §1 | ✓已核实 |
| 3 | Nakayama, Murakami, Kumar, Bhingardive, Hardaway, LREC 2022 | §1.1 | ✓已核实（Gold数据集来源） |
| 4 | Nakayama, Hikichi, Liu, Hirate, arXiv:2512.15151 | §1.2 | ✓已核实（面板数据来源） |
| 5 | Mintz, Bills, Snow, Jurafsky, ACL 2009 | §2.1 | ✓已核实 |
| 6 | Go, Bhayani, Huang, Stanford CS224N 2009 | §2.1 | ✓已核实 |
| 7 | Li, Chen, Zheng, Wang, Lalor, WNUT 2024 (arXiv:2305.01710) | §2.3 | ✓已核实（DSPN，核心对比） |
| 8 | "Implicit ABSA: A Systematic Review", ACM CSUR, DOI:10.1145/3799234 | §3.1 | 部分確認（作者未確認） |
| 9 | Ikegami, oseti (GitHub) | §3.2 | ✓已核实 |
| 10 | Takamura, Inui, Okumura, ACL 2005 | §3.2 | ✓已核实 |
| 11 | 小林・乾・松本・立石ら，自然言語処理 2005 | §3.2 | ✓已核实 |
| 12 | Gururangan et al., ACL 2020 (DAPT) | §4.1 | ✓已核实（方法灵感） |
| 13 | Conneau et al., ACL 2020 (XLM-R) | §4.2 | ✓已核实 |
| 14 | Lin, Goyal, Girshick, He, Dollár, ICCV 2017 | §5 | ✓已核实 |
| 15 | Ridnik et al., ICCV 2021 (ASL) | §5 | ✓已核实 |
| 16 | Broder, SEQUENCES 1997 (MinHash) | §6 | ✓已核实 |
| 17 | Indyk, Motwani, STOC 1998 (LSH) | §6 | ✓已核实 |
| 18 | Koehn, EMNLP 2004 | §7 | ✓已核实 |
| 19 | Efron, Annals of Statistics 1979 | §7 | ✓已核实 |
| 20 | Dietterich, Lathrop, Lozano-Pérez, AIJ 1997 (MIL) | §8 | ✓已核实 |
| 21 | Ilse, Tomczak, Welling, ICML 2018 | §8 | ✓已核实 |
| 22 | He, Gao, Chen, ICLR 2023 (DeBERTaV3, arXiv:2111.09543) | §4.4 | ✓已核实 |
| 23 | He, Liu, Gao, Chen, ICLR 2021 (DeBERTa, arXiv:2006.03654) | §4.4 | ✓已核实 |
| 24 | ku-nlp/deberta-v3-base-japanese（京大黑桥研，HuggingFace） | §4.4 | 模型发布（引用model card） |
| 25 | cl-tohoku/bert-base-japanese-v3（东北大，HuggingFace） | §4.4 | 模型发布（引用model card） |
| 26 | Li, Zou, Zhang, Zhang, Wei, EMNLP 2021 (SCAPT), pp.246–256 | §9.1 | ✓已核实（隐式改进·实测被BCE支配） |
| 27 | Fei, Li, Liu, Bing, Li, Chua, ACL 2023 (THOR), pp.1171–1182 (arXiv:2305.11255) | §9.2 | ✓已核实 |
| 28 | Yu, Becker, Muhammad et al., SemEval-2026 Task 3 DimABSA (arXiv:2604.07066) | §1.3 | ✓已核实（jpn-hot＝楽天トラベル新爬取，VA 回归，不可直接比较） |
| 29 | 中山・村上・Kumar・Bhingardive・Hardaway，NLP2021 P2-12 | §1.1.2 | ✓已核实（原论文日文版） |
| 30 | 味岡・岡田・森，JSAI 2024 2G6-GS-6-02 | §1.1.2 | ✓已核实 |
| 31 | 西元・森・岡田，JSAI 2025 3G4-GS-6-03 | §1.1.2 | ✓已核实 |
| 32 | 林・浦野，JSAI 2026 1Yin-B-28 | §1.1.2 | ✓已核实（「Macro F1」实为加权平均） |
| 33 | 小泉，JAIST 修士論文 2025 | §2.4 | ✓已核实全文（最接近的先行研究） |
| 34 | Hanafusa et al., LREC 2026 (2026.lrec-1.640) | §1.1.2 | ✓已核实摘要（另一日语 ABSA 数据集） |

**使用说明**：本表可直接作为论文 BibTeX / 参考文献列表的底稿。「✓已核实」条目的书目信息（作者全名、venue、页码）已通过检索确认，但**建议在最终投稿前仍通过 Google Scholar / ACL Anthology / DBLP 等权威源做二次核对**，尤其是页码等细节可能因预印本版本不同而有出入。「部分確認」条目（#8）请勿引用具体作者姓名，仅可引用 DOI 与已确认的规模/年份信息，或替换为其他可完整核实的隐式 ABSA 相关文献。「模型发布」条目（#24/#25）无独立学术论文，引用时指向 HuggingFace model card 与其架构论文（DeBERTaV3/BERT），勿虚构论文出处。
