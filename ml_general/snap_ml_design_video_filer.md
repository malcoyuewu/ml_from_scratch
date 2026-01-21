考虑trade off之前，加不加component也是trade off。

例如，为什么要加cache，不加cache有什么好处。
常见的还有at least once vs at most once.
read your write consistency vs eventual consistency
这些在DDIA都提过。

关于tradeoff, 我能想到是

1: write heavy (LSM tree) vs read heavy (mysql + cache)
2: pull vs post
3: latency sensitive (sync + db) vs throughput sensitive (async + message queue)

https://www.1point3acres.com/bbs/thread-1086103-6-1.html

https://www.1point3acres.com/bbs/thread-1155409-1-1.html

. Χ
感觉看eng blog确实比较好，可以把online inference + logging和offline feature/data/training pipeline各环节作为关键词，搜各公司ml infra blog。两个feature store例子：
Netflix的feature store Axion, 处理offline batch updates
Pins的real-time features，包括普通的aggregation features和sequence features。

https://www.1point3acres.com/bbs/thread-1038372-1-1.html

-- please help me design with question caliabration --> goal -> high level model proposal --> features --> candidates generator --> fine rnker --> rerank --> trade off naalysis: "也是系统设计，推荐snap filter，偏model的那种。按Alex Xu的ML design讲就行。"
下面给你一套 **“Alex Xu 风格 ML system design”** 的完整答题骨架，场景是 **Snap Filter 推荐（偏模型）**，并按你要的结构：

**Question Calibration → Goal → High-level proposal → Features → Candidate Gen → Fine Ranker → Rerank → Trade-off analysis**

---

## 1) Question Calibration（先把题目“锁死”）

面试官说“推荐 Snap filter，偏 model”，你要立刻问 6~8 个关键问题，把 scope 固定：

**产品/入口**

* 推荐发生在什么入口？（Camera 首页 / Lens Explorer / 搜索 / Chat 中贴纸）
* 推荐对象是 Lens/Filter（UGC + 官方）还是只官方？是否含 Sponsored Lens？

**约束**

* 强实时还是弱实时？（打开相机后 100ms 内必须出结果？）
* 需要多样性/新颖性约束吗？（避免同质、避免刷屏）
* 是否需要安全/政策过滤（未成年人、敏感内容、版权）？

**反馈信号与标签**

* 主要优化哪个行为：**Apply / Share / Save / Dwell / Return / Hide**？
* 是否有显式 dislike？是否有 session-level 目标（本次相机 session 的满意度）？

**冷启动**

* 新用户、新 Lens 冷启动怎么处理？是否允许探索？

**评估**

* 线上 KPI：Apply rate、D7 retention、session length、latency、GPU cost？
* 离线：AUC/NDCG 还是 counterfactual/off-policy 评估？

> 你问完这些，立刻给一句“我先假设”：
> **“我假设入口是 Camera 首页推荐 Top-K Lens，目标是 maximize apply+share，同时满足 P99<150ms 与 policy safety。”**

---

## 2) Goal（目标拆成：业务 + 体验 + 工程）

**业务目标（primary objective）**

* 最大化 **Expected Utility**：
  [
  U = w_1 \cdot P(apply) + w_2 \cdot P(share) + w_3 \cdot dwell - w_4 \cdot hide
  ]
  （不必写公式也行，但讲清楚：多目标加权）

**用户体验目标（guardrails）**

* 多样性（类别/creator/风格）
* 新颖性（不能总是热门）
* 公平性（避免只推头部）
* 安全合规（policy, age gating）

**工程目标**

* **延迟**：P99 < 150ms（举例）
* **成本**：每请求 CPU/GPU 预算
* **可观测**：曝光→点击→apply→share 全链路可追踪

---

## 3) High-level Model Proposal（端到端架构一句话）

用你要求的 pipeline 串起来：

**Retrieval（Candidate Gen）→ Pre-rank（可选）→ Fine Rank（LTR/Deep）→ Re-rank（Diversity/Policy/Ads）→ Serve**

推荐你用 **两塔 + 特征交叉 + 列表重排** 的经典组合：

* **Candidate Generator**：Two-Tower / ANN（向量召回）
* **Fine Ranker**：DCN-v2 / DLRM / Transformer-based ranking（特征交叉 + 序列特征）
* **Re-rank**：多样性 + 新颖性 + policy + ads（MMR / xQuAD / constrained optimization）

---

## 4) Features（特征体系：User / Context / Item / Cross / Sequence）

按 Alex Xu 的方式，你要列“可落地”的特征族，并强调哪些是实时哪些是离线。

### 4.1 User features

* 长期兴趣：Lens 类别偏好（beauty, funny, anime, sports…）
* 社交图：好友最近使用的 Lens embedding/统计
* 历史行为：apply/share/save/hide 的序列
* 设备/网络：机型、OS、GPU 能力（影响可运行的 AR Lens 类型）

### 4.2 Context features（强实时）

* 时间/地点（local time, geo bucket）
* session intent：刚打开相机 vs 刚看完故事 vs 刚聊天
* 光照/摄像头（前置/后置），可选：视觉场景类别（但注意隐私与延迟）

### 4.3 Item (Lens) features

* 内容 embedding：Lens 视觉效果/语义标签/描述文本 embedding
* 质量：历史 apply rate、completion、crash rate（AR Lens 可能 crash）
* 新鲜度：发布时间、热度趋势（trend slope）
* creator 画像：creator embedding、违规记录、头部/长尾

### 4.4 Cross features（交叉）

* user_interest ⊗ item_category
* device_capability ⊗ lens_complexity
* friend_used ⊗ same_creator

### 4.5 Sequence features（关键加分）

* 最近 N 次 camera session 行为序列
* 最近 K 次 “apply→share” 的路径
* session-level embedding（用 GRU/Transformer 做短期兴趣）

---

## 5) Candidate Generator（召回：高覆盖、低延迟）

目标：从百万 Lens → 2k~10k candidates

### 5.1 召回策略（多路）

* **Personalized ANN**：Two-Tower(user emb, lens emb) + 向量检索
* **Trending recall**：按地区/时间段热榜（强新颖性）
* **Social recall**：好友最近用过/分享过
* **Category recall**：用户偏好类别内召回
* **Exploration recall**：新 Lens / 长尾 creator（少量比例）

最后做 **merge + 去重 + policy filter**，控制候选规模。

### 5.2 Two-Tower 训练要点

* 正样本：apply / share（可分权重）
* 负样本：曝光未 apply；或 in-batch negatives
* 去偏：曝光偏置（position bias）→ 可加 IPS / propensity features（至少要提到“去偏问题”）

---

## 6) Fine Ranker（精排：学习复杂交互，输出最终 Top-N）

目标：2k → 100（或 200）

### 6.1 模型选择（讲 trade-offs）

* **DCN-v2 / DLRM**：工业常用、特征交叉强、延迟可控
* **Transformer ranker**：序列/上下文更强，但成本更高
* 多目标：多头输出（apply/share/hide），或单头“期望效用”

### 6.2 Label / Loss（要会说）

* 多任务学习：

  * head1: P(apply)
  * head2: P(share)
  * head3: P(hide)
* final score = 加权组合 + 校准（calibration）
* 训练数据的时间窗口、样本重加权（新 Lens 要加权）

### 6.3 Serving 细节（加分点）

* 特征：在线 feature store（实时）+ 离线特征（近实时更新）
* 模型：CPU 推理为主（Snap 这类强延迟场景通常更偏 CPU），复杂模型可做蒸馏
* 缓存：Top trending 可缓存；个性化部分做 session cache

---

## 7) Re-rank（重排：把“好看”变成“好用”）

精排出来的 Top-100 还不够，因为你还要满足：

**(1) 多样性**：类别/creator/风格去同质

* **MMR**：在 relevance 和 diversity 之间平衡
* 或 **xQuAD**：覆盖多意图/多主题

**(2) 新颖性/探索**

* ε-greedy / Thompson Sampling（轻量可控）
* 或基于 uncertainty 的探索（需要模型输出方差/ensemble）

**(3) Policy & Safety**（必须硬约束）

* age gating、敏感标签、版权、creator 风险
* 先 hard filter，再 soft penalize（避免“全屏被删空”）

**(4) Ads / Sponsored Lens**（如果有）

* 约束式插入：比如每 K 个插一个赞助 Lens，但要 obey relevance 阈值
* 多目标：收入 vs 用户体验（guardrail KPI）

---

## 8) Trade-off Analysis（面试最爱：你要“选型有理”）

下面这段建议你背熟，几乎所有推荐系统都能用。

### 8.1 质量 vs 延迟

* 更强模型（Transformer）↑质量，但 ↑P99 与成本
* 解决：**两阶段（召回+精排）**、蒸馏、量化、特征裁剪、缓存

### 8.2 个性化 vs 覆盖/冷启动

* 强个性化容易陷入过滤泡泡、对新 Lens 不友好
* 解决：多路召回 + exploration bucket + trend recall + creator boosting

### 8.3 去偏与训练数据质量

* 曝光位置、UI 影响 labels（position bias）
* 解决：IPS/propensity、随机化流量做校准集、counterfactual evaluation

### 8.4 实时特征 vs 一致性

* 实时特征能抓意图，但可能 stale/延迟不稳定
* 解决：feature TTL、fallback、分层 feature（critical vs optional）

### 8.5 多样性/安全约束 vs relevance

* 强多样性可能降低短期 CTR/apply
* 解决：重排层做可控权重 + guardrail（长期 retention）

---

## 你可以直接用的 60 秒“串讲版本”

> “我把 Snap Filter 推荐建成两阶段：多路召回把百万 Lens 拉到几千候选（Two-Tower ANN + 热榜 + 社交 + 类别 + 探索），然后用 DCN-v2/DLRM 做精排输出 Top-100，多任务预测 apply/share/hide 并做 score calibration。最后在重排层做多样性、新颖性、policy safety 和可能的 sponsored 约束插入。核心权衡是质量 vs P99 延迟、个性化 vs 冷启动覆盖、去偏训练 vs 数据可得性，我会通过蒸馏/量化/缓存、探索桶、以及随机化流量+IPS 来解决。”

---

如果你愿意，我可以把这套答案进一步“Snap 面试风格化”成：

* **白板图**（每层输入输出、延迟预算）
* **指标表**（offline/online/guardrails）
* **follow-up 问答库**（比如“你怎么做 policy filter？”“如何做 embedding 更新？”“如何做 session-based ranking？”）

-- please also cover logging and online near real time online features store

下面把你缺的两块 **Logging/Observability** 和 **Online / Near-Real-Time Feature Store** 直接嵌进这套 Snap Lens 推荐设计里（依然是 ML system design 口径，偏可落地工程）。

---

## A) Logging & Observability（必须讲“全链路 + 可回放 + 可归因”）

### A1. 你要记录什么（按事件流拆）

**1) Request / Context log（每次打开相机/刷新推荐）**

* `request_id`, `user_id`, `session_id`, `timestamp`
* 入口/场景：camera_home / explorer / search
* 设备/网络：device_model, os, app_version, net_type
* 实时上下文：geo_bucket, local_time_bucket, camera_front/back
* 实验信息：`exp_id`, `variant_id`

**2) Candidate log（召回阶段）**

* 每路召回的：

  * `recall_source`（ANN / trending / social / category / explore）
  * 候选列表 item_ids（可采样/截断）
  * 召回分数（embedding dot / popularity score）
  * **召回耗时**、候选规模（num_candidates）
* Merge 结果：去重后总候选数

**3) Ranking log（精排/重排）**

* 精排输入：Top-M candidates（可采样）
* 每个 item 的：

  * `rank_score`（以及多头：p_apply, p_share, p_hide）
  * 关键特征快照（只存“可解释 subset”，避免全量特征爆炸）
  * 模型版本：`model_name`, `model_version`, `feature_schema_version`
* 重排输出：最终 Top-K + 重排原因（diversity bucket / policy filter / sponsored insert）

**4) Impression log（曝光）**

* `position`, `impression_time`, `view_duration`（可选）
* 是否被 policy filter 降权/屏蔽（用于审计）

**5) Interaction / Outcome log（反馈）**

* apply / share / save / hide / dwell / bounce
* 关键：把 outcome 事件带上 `request_id` / `impression_id`，用于 join 训练

---

### A2. 你怎么用这些 log（面试加分点）

**在线监控（SLO / 数据健康）**

* 服务：P50/P95/P99 延迟、超时率、召回缺失率、feature missing rate
* 质量：apply rate / share rate / hide rate，按入口/设备/地区分层
* 漂移：embedding 分布漂移、特征分布漂移（PSI / KL）
* 训练数据：label 延迟、join 成功率、负样本比例稳定性

**Debug / Replay（能回放一次请求）**

* 用 `request_id` 拉出：context → candidates(按路) → rank scores → final list → subsequent outcomes
* 解决“线上掉量”时能快速定位：是召回变差、特征缺失、模型 bug、还是 policy 误杀

**归因与实验**

* 全链路带 `exp_id/variant`，做 A/B 守护指标（crash rate、latency、policy violation）
* 多目标：短期 apply vs 长期 retention 的 guardrail

> Snap/Lens 这种还要特别强调 **Crash/Performance logs**（AR Lens 复杂度可能导致帧率下降/闪退），这是业务强相关信号。

---

## B) Online / Near-Real-Time Feature Store（核心是“分层 + TTL + 回填 + 一致性策略”）

### B1. 特征分层（你答题最好按这 3 层讲）

**Layer 0：Request-time / Context features（0ms，来自请求）**

* device, network, geo, time bucket, entry surface
* 优点：稳定、零依赖
* 作用：所有阶段可用（召回/精排/重排）

**Layer 1：Online real-time features（毫秒级读取，秒级更新）**
典型：近实时用户状态、session 状态、近期交互计数

* `user_recent_apply_count_1h/24h`
* `user_last_k_lens_embeddings`（或聚合后的 session embedding）
* `friend_recent_lens_topN`（可做近实时聚合）
  **存储**：Redis / RocksDB-based KV / Scylla/Cassandra（看公司栈）
* Key：`(user_id)` 或 `(user_id, session_id)`
* Value：protobuf / thrift 压缩结构（固定 schema）
* TTL：小时级或天级（避免陈旧）

**Layer 2：Near-real-time aggregates（分钟级~小时级）**
典型：趋势、热度、全局统计、creator 质量分

* `lens_trending_score_5m/1h`
* `lens_apply_rate_1h`（滑窗）
* `lens_crash_rate_1h`
  **实现**：流式计算（Kafka/Flink/Spark Streaming）→ 写入 Feature Store / OLAP

---

### B2. 写入链路（Streaming Feature Pipeline）

**事件产生 → 消息队列 → 流处理聚合 → 写入 online store**

* Events：impression/apply/share/hide/crash
* Kafka topic：按事件类型分 topic（或统一 envelope）
* Flink job：

  * 滑窗聚合（5m/1h/24h）
  * 去重（按 impression_id）
  * 输出到 Online Feature Store（Redis/Scylla）和离线仓（S3/Hive）双写
* Exactly-once：不一定强求，但要有 **幂等写**（event_id 去重）或事务语义

---

### B3. 读取策略（Serving path 的“延迟预算”）

推荐你把读特征拆成两段：

**召回阶段（最敏感）**

* 只读：Layer0 + 少量 Layer1（比如 user embedding / session embedding）
* 严控：≤ 5~10ms（含网络）
* 缓存：session cache（同一 session 多次刷新复用）

**精排阶段**

* 读：Layer0 + Layer1 + 部分 Layer2（趋势/质量）
* 做批量读取（multi-get），减少 round trips
* 对重排策略相关的特征（policy/safety）尽量本地化或提前过滤

---

### B4. 一致性与降级（必须讲，否则面试官会追）

**一致性模型**

* Online FS 通常是 **最终一致**；训练与 serving 可能存在 skew
* 解决：

  * feature schema versioning
  * training-serving parity：同一套聚合逻辑（或共享代码）
  * 记录 feature snapshot hash 到 ranking log（便于排查 skew）

**Fallback / Degrade**

* 特征缺失时：

  * 召回 fallback：热榜/类别召回兜底
  * 精排 fallback：用 lightweight model 或 rule score
* 超时策略：

  * 召回：超时直接返回部分候选
  * 精排：超时返回上一次 session cache Top-K

---

### B5. “在线近实时”与“离线训练”怎么对齐（训练数据生成）

* Impression / ranking logs → 离线仓（数据湖）
* 通过 `impression_id` join outcome（apply/share/hide）
* 生成训练样本时要处理：

  * label delay（share 可能延迟）
  * position bias（带 position / UI features）
  * sample reweight（新 Lens/长尾 creator）

---

## C) 你可以直接塞进答案里的 30 秒段落

> “我会做全链路 logging：request/context、各路召回候选与耗时、精排/重排分数与模型版本、最终曝光列表及位置、以及 apply/share/hide/crash 等反馈，并用 request_id 做 replay，支持 A/B 归因和 drift 监控。特征上我会分层：请求即时特征、秒级更新的在线 KV 特征（Redis/Scylla，TTL 控制）、以及分钟级滑窗聚合的 near-real-time 趋势/质量特征（Kafka+Flink 写回 online store）。Serving 读特征做 multi-get 与超时降级，保证 P99。”

---

如果你想，我可以把这两块进一步补成 **白板图 + SLO 数字**（例如每一层预算：召回 20ms、精排 80ms、重排 20ms、网络与开销 30ms），以及 Snap 面试常见追问的标准回答（例如“怎么避免特征爆炸”“怎么做 feature freshness”“怎么做 exactly-once/幂等”）。


-- how does counterfactual/off-policy 评估？ work?
下面我用 **“面试可讲清 + 数学直觉正确 + 工程可落地”** 的方式，把 **counterfactual / off-policy evaluation（OPE）** 系统性讲清楚，并且直接对齐你前面那个 **Snap Filter / Lens 推荐** 场景。

---

# 1️⃣ 为什么需要 Counterfactual / Off-policy 评估？

### 问题本质

我们想回答的是：

> **“如果我上线一个新推荐策略 π_new，而不真的上线它，我能不能提前估计它的效果？”**

困难在于：

* 真实数据是由 **旧策略 π_old（logging policy）** 产生的
* 每个用户 **只看到了 π_old 给他的结果**
* **没有看到** π_new 会给他什么（反事实 counterfactual）

👉 这就是 **selection bias / exposure bias**
👉 传统离线 AUC / NDCG **不可靠**

---

# 2️⃣ 核心思想（一句话）

> **用“旧策略下观察到的反馈”，通过概率重加权，来估计“新策略会发生什么”**

关键词：

* **Logging policy π₀**
* **Target policy π₁**
* **Propensity（曝光概率）**
* **Importance Sampling（重要性采样）**

---

# 3️⃣ 最基础方法：IPS（Inverse Propensity Scoring）

## 3.1 设定

对每一次曝光（impression）记录：

* context：x（用户 + 场景）
* action：a（展示的 Lens）
* reward：r（apply / share / dwell）
* propensity：π₀(a | x)（旧策略选这个 Lens 的概率）

目标：估计新策略 π₁ 的期望 reward

---

## 3.2 IPS 公式（面试必会）

[
\mathbb{E}*{\pi_1}[r] \approx
\frac{1}{N} \sum*{i=1}^N
\frac{\pi_1(a_i \mid x_i)}{\pi_0(a_i \mid x_i)} \cdot r_i
]

直觉：

* 如果某个样本 **在旧策略下很少出现**，但 **新策略很想选它**
* 那这个样本的 reward 就要被 **放大**

---

## 3.3 Lens 推荐里的例子

* 旧策略：热门 Lens 占 80% 曝光
* 新策略：更偏长尾 + 新创作者
* 某 Lens：

  * π₀ = 0.01
  * π₁ = 0.05
  * 用户 apply = 1

👉 权重 = 5
👉 说明：这个 Lens 在新策略下“更重要”

---

## 3.4 IPS 的致命问题（面试官一定会追）

❌ **高方差**

* π₀ 很小 → 权重爆炸
* 少量样本主导估计

❌ **propensity 不准**

* 如果 π₀ 记录不准 → 评估完全失真

---

# 4️⃣ 工业级改进方法（重点）

## 4.1 SNIPS（Self-Normalized IPS）

[
\hat{V}_{SNIPS} =
\frac{\sum_i w_i r_i}{\sum_i w_i}
\quad \text{where } w_i = \frac{\pi_1}{\pi_0}
]

✔ 优点：

* 显著降低方差
* 工业界非常常用

❌ 缺点：

* 引入一点 bias（但通常值得）

👉 **面试推荐说：默认用 SNIPS**

---

## 4.2 Clipped IPS（权重裁剪）

[
w_i = \min\left(\frac{\pi_1}{\pi_0}, c\right)
]

✔ 控制极端样本
✔ 工程上非常实用
❌ 引入 bias

---

## 4.3 Doubly Robust（DR）⭐（高级加分）

结合：

1. **直接 reward 模型**（Q(x,a)）
2. **IPS 校正项**

[
\hat{V}_{DR} =
\frac{1}{N} \sum_i
\Big[
\hat{Q}(x_i, \pi_1)

* w_i \cdot (r_i - \hat{Q}(x_i, a_i))
  \Big]
  ]

直觉：

* 如果 reward 模型准 → 即使 propensity 有点错也 OK
* 如果 propensity 准 → 即使 reward 模型有点错也 OK

👉 **只要其中一个对，就不崩**

📌 在推荐系统 / 广告系统 / RLHF 中非常受欢迎

---

# 5️⃣ Ranking / Slate 场景（Snap Lens 的关键点）

Lens 推荐不是选 **一个 action**，而是 **一个列表（Top-K）**

### 常见做法（面试可讲）

1. **Position-based IPS**

   * 记录每个位置的 propensity
   * π₀(a at position j | x)

2. **Top-K truncation**

   * 只对前 K 位做 OPE
   * 避免低位 noisy 样本

3. **List-wise reward**

   * reward = Σ position_weight × apply_j

---

# 6️⃣ 数据怎么收？（这是工程关键）

## 6.1 必须做“随机化流量”

你**不能**完全相信 deterministic 推荐系统

### 常见策略

* ε-greedy（1%–5%）
* Uniform random on candidate set
* Random position swap

目的：

* 覆盖 action space
* 让 π₀(a|x) > 0

👉 **没有 exploration = 没法做 counterfactual**

---

## 6.2 Logging 必须包含

* action probabilities（propensity）
* 完整 candidate set（或至少 Top-M）
* position
* model / policy version

---

# 7️⃣ 在 Snap Lens 里的完整使用流程

**Offline**

1. 用随机化流量收集 logged data
2. 训练新 ranker
3. 用 SNIPS / DR 评估：

   * apply rate
   * share rate
   * hide rate
4. 看 uplift + 方差 + confidence interval

**Online**
5. 小流量 A/B 验证
6. 对比 OPE 预测 vs 实际 uplift
7. 校准 OPE pipeline

---

# 8️⃣ 面试官最爱追问 & 标准回答

### Q1：为什么不用普通离线 AUC？

> 因为数据是由旧策略产生的，有 exposure bias，模型只在“被展示过的分布”上学得好。

---

### Q2：propensity 怎么算？

> 来自 logging policy：softmax score / sampling probability；如果是 deterministic，会通过随机化 bucket 构建。

---

### Q3：IPS 和 DR 你用哪个？

> 默认 SNIPS；在 reward 模型比较成熟时用 Doubly Robust。

---

### Q4：什么时候 OPE 不可信？

* 探索太少
* π₁ 与 π₀ 分布差太远
* propensity 记录不准确
* slate/long-term reward 未建模

---

# 9️⃣ 30 秒面试总结版（你可以直接背）

> “Counterfactual 或 off-policy 评估用于在不实际上线新策略的情况下估计其效果。核心是用旧策略下的数据，通过 propensity-based importance sampling 重加权来估计新策略的期望 reward。实践中我会用 SNIPS 或 Doubly Robust 来降低方差，并通过少量随机化流量保证 action coverage。对于 Lens 这种 Top-K 排序场景，会做 position-aware OPE，并最终用小流量 A/B 验证校准离线估计。”

---

如果你愿意，我可以：

* 给你一份 **OPE vs A/B 对照表**
* 写一个 **Snap Lens 风格的 DR 评估伪代码**
* 或把它提升到 **RL / bandit / policy learning** 角度（PPO / slate bandit）


-- follow-up 问答库（比如“你怎么做 policy filter？”“如何做 embedding 更新？”“如何做 session-based ranking？”）
下面给你一套 **“Snap / 推荐系统 ML Design Follow-up 问答库”**，全部是**面试官高频追问 + Staff/L5/L6 级标准回答**，我会刻意用 **可白板 + 可工程落地** 的表述方式。
（你可以把它当成“随身小抄”）

---

# Ⅰ. Policy / Safety / Trust & Integrity

## Q1️⃣ 你怎么做 policy filter？是在模型前还是模型后？

**标准答案（分层）**

> 我会做 **三层 policy 防护**：前置 hard filter + ranking soft constraint + post-ranking audit。

**1）Pre-ranking Hard Filter（必须）**

* 在 **candidate generation 之后、精排之前**
* 规则型 / ML classifier：

  * age gating（未成年人）
  * 敏感内容标签
  * creator 风险（违规历史）
* 特点：**零容忍、低延迟**
* 目的：保证模型**永远看不到非法候选**

**2）In-ranking Soft Constraint**

* 把 policy risk 当作 feature / penalty
* 在 score 中扣分，而不是直接删除
* 避免过度误杀（false positive）

**3）Post-ranking Audit / Override**

* 重排后再次检查（尤其是 sponsored / UGC）
* 兜底逻辑，防止 edge case 泄露

👉 **总结句（面试好用）**

> “Hard filter 保证合规底线，soft constraint 平衡体验，post-ranking 是最后一道保险。”

---

## Q2️⃣ policy 规则变化频繁，怎么避免频繁重训模型？

**答案要点**

* **Policy decoupling**

  * 规则与模型解耦
  * rule engine / config-driven
* 模型只学“偏好”，不学“法律”
* 规则变更 → 热更新，不重训模型

---

# Ⅱ. Embedding 体系（召回 & 表示学习）

## Q3️⃣ Lens embedding 是怎么训练和更新的？

**标准结构化回答**

### 1）Embedding 类型

* **Item embedding（Lens）**

  * 内容：视觉效果 / 文本描述 / creator 标签
* **User embedding**

  * 长期兴趣（apply/share）
  * 短期 session embedding

### 2）训练方式

* **Two-Tower / Contrastive Learning**
* 正样本：apply / share
* 负样本：

  * in-batch negatives
  * impression-but-no-apply
* 去偏：加 position / exposure feature

### 3）更新策略（面试必讲）

| 类型                | 更新频率         | 方式                 |
| ----------------- | ------------ | ------------------ |
| Item embedding    | 小时级 / 天级     | 离线批量               |
| User embedding    | 秒~分钟级        | online aggregation |
| Session embedding | request-time | on-the-fly         |

---

## Q4️⃣ 新 Lens / 冷启动 Lens 怎么办？

**高分回答**

* 内容 embedding（不依赖用户行为）
* creator embedding 继承
* exploration bucket（ε-greedy）
* 召回层保证最低曝光

👉 关键词：**coverage guarantee**

---

## Q5️⃣ embedding 漂移 / 过期怎么监控？

**面试加分**

* embedding distribution drift（L2 / cosine）
* ANN recall@K 稳定性
* online apply rate by embedding age
* 自动回滚旧 embedding 版本

---

# Ⅲ. Session-based Ranking（Snap 强考点）

## Q6️⃣ 什么是 session-based ranking？为什么 Snap 特别重要？

**一句话定义**

> Session-based ranking 利用用户**短期意图**（当前 camera session），而不是长期历史。

**Snap 场景**

* 打开相机 → 强意图、低容忍
* 当前 5–10 分钟行为 ≫ 过去一周历史

---

## Q7️⃣ 你怎么建模 session？

**标准三层方案**

### 1）规则型（baseline）

* 最近 K 次 apply 的 Lens 类别
* 最近使用的 creator
* 时间衰减权重

### 2）Embedding 聚合

* 最近 N 个 Lens embedding
* time-decay average / attention pooling

### 3）Sequence Model（高阶）

* GRU / Transformer
* 输入：Lens embedding + context
* 输出：session embedding

👉 精排时用：`user_long ⊕ session_short ⊕ item`

---

## Q8️⃣ session 特征怎么做在线服务？会不会太慢？

**工程回答**

* session embedding 存 Redis（TTL=30min）
* request 内增量更新
* 超时 fallback 到 user long-term embedding

---

# Ⅳ. Ranking / Modeling 追问

## Q9️⃣ 为什么不用一个超大模型直接端到端？

**标准 trade-off 回答**

* latency / cost 不可控
* 可解释性差
* policy / ads / diversity 难插入

👉 所以用 **multi-stage ranking**

---

## Q1️⃣0️⃣ 多目标（apply/share/hide）怎么优化？

**两种主流方式**

1. Multi-head prediction + 加权
2. Scalarized utility（业务调权）

**Guardrail**

* hide / crash 作为 hard constraint

---

## Q1️⃣1️⃣ 如何防止模型学会“点而不用”的 Lens？

**回答要点**

* apply > click
* dwell threshold
* delayed reward
* negative signal（hide / bounce）

---

# Ⅴ. Logging / Evaluation（延伸）

## Q1️⃣2️⃣ 离线评估你信什么？

**优先级**

1. Counterfactual OPE（SNIPS / DR）
2. Replay metrics
3. Small A/B

👉 永远不信单纯 AUC

---

## Q1️⃣3️⃣ 线上掉量你怎么排查？

**黄金排查顺序**

1. policy filter 是否误杀
2. 特征缺失 / 延迟
3. 召回覆盖下降
4. embedding / model version mismatch
5. UI / 客户端变更

---

# Ⅵ. 系统级追问（Staff / Principal）

## Q1️⃣4️⃣ 推荐系统最大的失败模式是什么？

**成熟回答**

* 过拟合旧分布（filter bubble）
* policy/模型耦合
* offline 指标与线上脱钩
* 冷启动/探索不足

---

## Q1️⃣5️⃣ 如果 apply rate ↑ 但留存 ↓，你怎么办？

**高段位回答**

* 多目标未平衡
* short-term reward over-optimized
* 提高 diversity / novelty
* 引入 long-term proxy（D7 return）

---

# Ⅶ. 30 秒「Follow-up 问答总结版」

> “我会用分层 policy filter 保证合规，用 Two-Tower embedding 支撑多路召回，并通过 session-based ranking 捕捉短期意图。Embedding 离线更新、session 在线聚合，精排用多任务模型，重排处理多样性和安全约束。评估上依赖 counterfactual OPE + 小流量 A/B，确保线上长期指标不被短期点击劫持。”

---

如果你愿意，下一步我可以帮你做三件**非常 Snap 面试向**的事之一：
1️⃣ 把这套问答压缩成 **一页白板答题模板**
2️⃣ 出一轮 **“面试官连环追问模拟”**（我扮面试官）
3️⃣ 针对 **L5 vs L6** 给你不同深度版本（答多深刚刚好）


-- follow-up 问答库（比如“你怎么做 policy filter？”“如何做 embedding 更新？”“如何做 session-based ranking？”）
下面我直接给你 **“同一道 Snap 推荐系统 ML 设计题，在 L5 vs L6 的答题深度差异”**。
你可以把它当成 **“同一张白板，讲到哪一层停”** 的参考。

---

# 总览对比（面试官心智模型）

| 维度   | L5（Senior MLE） | L6（Staff MLE） |
| ---- | -------------- | ------------- |
| 关注点  | 模型 + 系统能跑对     | 架构正确 + 长期可扩展  |
| 设计视角 | 单系统最优          | 多系统权衡         |
| 模型   | 用什么模型          | 为什么 + 什么时候不用  |
| 数据   | 特征怎么来          | 数据怎么坏、怎么修     |
| 决策   | 能实现            | 能为团队定方向       |

---

# 1️⃣ Question Calibration

## L5：确认输入输出即可

> “这是 Camera 首页 Lens 推荐，目标是 maximize apply + share，延迟要求 P99 < 150ms，对吗？”

✔ 清楚
✔ 不发散

---

## L6：先锁 business boundary

> “这是 Camera 首页的核心入口，强实时高并发。我会假设它是 Snap 的 DAU 心跳，因此我会优先优化长期留存，其次才是短期 apply。”

✔ 把系统**放进公司战略**
✔ 已经在“帮面试官做判断”

---

# 2️⃣ Goal 设定

## L5：目标可量化

* maximize apply / share
* guardrail：hide rate、latency、crash rate

**停在这里完全 OK**

---

## L6：目标是 trade-off surface

* 短期：apply ↑
* 中期：session depth ↑
* 长期：creator ecosystem 健康度 ↑
* 工程：模型复杂度 vs 迭代速度

👉 **L6 关键词：second-order effect**

---

# 3️⃣ High-level Architecture

## L5：标准 pipeline

> “我会用多路召回 + 精排 + 重排的经典推荐架构。”

* Two-Tower retrieval
* DCN/DLRM ranker
* MMR rerank

✔ 工业正确
✔ 风险低

---

## L6：架构为什么是“分阶段”

> “我刻意不做端到端，是因为 policy、ads、diversity 都是独立演进的约束点，如果耦合进模型，组织成本会指数级上升。”

👉 **L6 = 技术 + 组织设计**

---

# 4️⃣ Feature & Embedding

## L5：列清楚就够

* user / item / context
* session features
* online feature store

---

## L6：关注 feature failure modes

* 哪些特征最容易 stale？
* 哪些特征对 P99 最敏感？
* 哪些特征一坏会“静默伤害指标”？

👉 例句：

> “session embedding 是高价值但高风险特征，我会给它 TTL + fallback。”

---

# 5️⃣ Candidate Generation

## L5：覆盖多路召回

* ANN personalized
* trending
* social
* exploration

---

## L6：召回 = 产品策略入口

> “召回层决定了生态公平性，我会在召回层显式给新 Lens 和新 creator 留 quota，而不是指望精排模型学会‘善良’。”

---

# 6️⃣ Fine Ranker

## L5：模型细节

* DCN-v2 / multi-head
* label / loss / calibration

---

## L6：模型是可替换部件

> “我不赌某个模型结构，而是赌可替换性：feature contract、线上评估、快速回滚。”

👉 强调：

* model registry
* shadow mode
* fast rollback

---

# 7️⃣ Re-rank / Policy

## L5：知道怎么做

* MMR
* policy filter
* sponsored insertion

---

## L6：谁该为什么负责

> “policy 不应该由模型 owner 扛全部责任，它应该是平台能力，服务所有推荐系统。”

👉 这是 **Staff thinking**

---

# 8️⃣ Logging & Evaluation

## L5：会用 OPE

* SNIPS / DR
* small A/B

---

## L6：评估系统本身要被评估

> “我会持续对比 OPE 预测 vs 实际 A/B uplift，监控 bias 漂移。”

---

# 9️⃣ Failure Handling

## L5：会兜底

* fallback
* cache
* timeout degrade

---

## L6：提前设计失败

> “系统默认会失败，我设计的是失败时伤害最小。”

---

# 🔟 30 秒总结对比（面试官视角）

### L5 说完，面试官想的是：

> “这个人能独立负责一个复杂推荐系统。”

### L6 说完，面试官想的是：

> “这个人能定义我们以后怎么做推荐系统。”

---

# 🧠 一句话判断你该讲到哪一层

* **被问“你会怎么设计？” → L5 深度**
* **被问“你为什么这么设计？” → L6 深度**
* **被反复追问 trade-off / edge case → 已经是 L6 场**

---

如果你愿意，下一步我可以：

* 🎯 帮你 **把同一题拆成 L5 vs L6 的两套“逐字答案”**
* 🧑‍⚖️ 模拟一轮 **Snap L6 面试官连环追问**
* 🧪 给你一套 **“什么时候别装 L6” 的风险提示（非常重要）**

你要哪一个？
-- Dive

下面给你一份**可直接上白板**的 Snap Filter / Lens 推荐（偏 model）系统设计稿，严格按你要的顺序，并包含 **每层输入输出 + 延迟预算**（Alex Xu ML Design 口径）。你可以照着画、照着讲。

---

# 1) Question Calibration（把范围锁死）

我会先问这 8 个问题（白板左上角写“Assumptions”）：

1. 入口：Camera 首页推荐？Lens Explorer？Search？
2. 物品：推荐的是 Lens/Filter（UGC+官方）？是否含 Sponsored？
3. Top-K：一次返回多少？（Top 20/50）
4. 目标：优化 apply / share / save / hide / dwell 哪些？主目标是什么？
5. 约束：P99 延迟要求？（假设 P99 < 150ms）
6. policy：年龄分级、敏感内容、版权是否需要强过滤？
7. 冷启动：新用户、新 Lens 是否需要探索配额？
8. 评估：上线前离线用什么，线上 A/B guardrails 是什么？

**默认假设（写在白板上）**

* Camera 首页个性化推荐 **Top-20 Lens**
* Primary：**Expected Utility = apply + share – hide – crash**
* Guardrail：**P99 < 150ms**，policy 100% 合规，crash rate 不升

---

# 2) Goal（业务 + 体验 + 工程）

白板写三行：

* **Business:** maximize Apply/Share (session utility)
* **UX Guardrails:** diversity / novelty / fairness / safety
* **Eng:** P99 latency, cost, debuggability (replay), A/B agility

---

# 3) High-level Model Proposal（总架构一句话）

> 多路召回（ANN + trending + social + explore）→ 精排（multi-task ranker）→ 重排（diversity/policy/ads）→ serve + log

---

# 4) Whiteboard 图（每层 I/O + 延迟预算）

把下面这张“ASCII 白板图”画出来即可（你也可以照抄到白板）：

```
[Client Camera Open]
   |
   |  (0-5ms) request context
   v
[Gateway / Rec Service]
   |
   |--(0-10ms) Online Feature Fetch (critical)
   |     Input: user_id, session_id, context
   |     Output: user_long_emb, session_emb, realtime counters
   |
   |--(20-40ms) Candidate Generation (multi-recall)
   |     Input: embeddings + context + policy allowlist
   |     Output: ~2k-10k candidates + recall_source tags
   |
   |--(5-10ms) Pre-rank (optional lightweight)
   |     Input: candidates + sparse features
   |     Output: top ~1k
   |
   |--(50-70ms) Fine Ranker (deep LTR, multi-task)
   |     Input: top1k + rich features
   |     Output: top100 with p_apply/p_share/p_hide + score
   |
   |--(5-15ms) Re-rank (constraints)
   |     Input: top100 + constraints (diversity/policy/ads)
   |     Output: final top20
   |
   |--(5-10ms) Response + Logging async
   v
[Client renders Lens shelf]

Total P99 Budget: 120-150ms (server side ~110-140ms + network)
```

### 推荐的 P99 拆分（更像面试官期待的表述）

* Feature fetch（critical only）: **≤10ms**
* Candidate gen（ANN+merge）: **≤40ms**
* Pre-rank（可选）: **≤10ms**
* Fine rank: **≤70ms**
* Re-rank: **≤15ms**
* Overhead（serialize, RPC）: **≤10ms**

> 若面试官追问“怎么保证”：回答 **批量 multi-get、并行召回、超时降级、缓存**。

---

# 5) Features（按“可落地”讲，强调实时 vs 离线）

### 5.1 User features

* long-term兴趣：类别偏好、creator 偏好、embed
* 历史行为：apply/share/hide 序列统计
* 设备能力：机型/GPU/帧率能力（Lens 复杂度 gating）

### 5.2 Context（强实时）

* time_bucket, geo_bucket, entry_surface
* camera_state（front/back）、network_type
* session state：最近一次操作、session embedding

### 5.3 Item (Lens) features

* 内容 embedding（视觉效果/文本tag）
* 质量与风险：crash_rate、policy_risk_score
* 热度/趋势：trending_slope、freshness

### 5.4 Cross / Sequence（加分）

* user_interest ⊗ lens_category
* device_capability ⊗ lens_complexity
* session sequence model embedding（短期意图）

---

# 6) Candidate Generator（多路召回：覆盖 + 低延迟）

目标：百万 Lens → 2k~10k

**Recall routes（并行）**

1. **Personalized ANN**（Two-Tower：user_emb · lens_emb）
2. **Trending recall**（本地/全局热榜，按 geo/time）
3. **Social recall**（好友最近用过/分享）
4. **Category recall**（用户偏好类目内）
5. **Exploration bucket**（新 Lens / 长尾 creator 配额）

**Merge & filter**

* 去重 + policy allowlist + 复杂度 gating（弱机型不推重 Lens）
* 输出带上 `recall_source`（用于 debug/归因）

> L6 说法加分：**召回层决定生态公平性（长尾/新创作者 quota）**。

---

# 7) Fine Ranker（精排：multi-task + 可校准）

输入：top ~1k candidates + rich features
输出：top100 + 多头概率

**模型选择（稳）**

* DCN-v2 / DLRM / MLP with cross（CPU 友好）
* Multi-head：p_apply / p_share / p_hide / p_crash

**打分**

* `score = w1*E[apply] + w2*E[share] - w3*E[hide] - w4*E[crash]`
* 做 **calibration**（比如分桶/温度缩放），保证概率可比

**Serving 关键**

* 批量特征读取（multi-get）
* 轻量化：蒸馏/量化/特征裁剪（P99 保证）

---

# 8) Re-rank（重排：把“相关”变成“好体验 + 合规”）

输入：top100
输出：final top20

**约束层次**

1. **Hard policy filter（必须）**：敏感/版权/年龄 gating（兜底再跑一次）
2. **Diversity**：MMR / xQuAD（类别/creator 去同质）
3. **Novelty / Explore**：控制探索比例（例如 10–20%）
4. **Sponsored（如有）**：约束式插入（每 K 个最多 1 个，且 relevance≥阈值）

---

# 9) Logging / Online Feature Store（白板右侧写“Observability”）

## 9.1 必须记录的日志（用 request_id 串联可回放）

* **request/context log**：user/session/context/exp_id
* **candidate log**：每路召回候选数、topN、耗时、recall_source
* **ranking log**：模型版本、feature_schema、top items 的 score/heads（采样）
* **impression log**：position、曝光列表、policy 是否介入
* **outcome log**：apply/share/hide/dwell/crash（带 impression_id）

**用途**

* replay debug（一次请求全链路回放）
* drift/数据健康（feature missing、分布漂移）
* 训练集 join（曝光→反馈闭环）
* A/B 归因（按 exp_id 分层）

## 9.2 Online / Near Real-time Feature Store（3 层）

* Layer0：请求即时特征（0ms）
* Layer1：在线 KV（Redis/Scylla）秒~分钟级更新（session embedding、近期计数，TTL）
* Layer2：流式聚合（Kafka+Flink）分钟级滑窗趋势/质量（trending/crash）

**降级策略**

* 特征超时 → 用 cached/session fallback
* 召回缺失 → trending/category 兜底
* 精排超时 → 返回 pre-rank/top cached

---

# 10) Trade-off Analysis（面试官最爱）

白板底部写 5 条：

1. **Quality vs Latency**：更强模型↑质量但↑P99 → multi-stage、蒸馏、量化、缓存
2. **Personalization vs Cold-start**：强个性化伤新 Lens → 多路召回 + exploration quota
3. **Diversity vs CTR**：多样性可能降短期 apply → 用 guardrail 看长期 retention
4. **Freshness vs Consistency**：实时特征更准但易 stale → TTL + fallback + schema version
5. **Bias/Exposure**：离线评估偏差 → 随机化流量 + OPE（SNIPS/DR）+ 小流量 A/B

---

## 你上白板时的“讲法节奏”（建议）

1. 30s calibration+assumptions
2. 60s 画 pipeline + latency budget
3. 2–3min 讲召回/精排/重排（每层 I/O）
4. 60s logging + feature store + degrade
5. 30s trade-offs + evaluation（OPE/A-B）

---

如果你要我继续加码（更贴 Snap 面试），我可以再给你：

* **L5 版（讲到能落地即可） vs L6 版（强调生态/组织/可演进）逐字稿**
* **面试官连环追问清单**（比如“怎么算 propensity？”“policy 误杀怎么查？”“ANN 索引怎么更新？”）并附标准回答
<img width="1200" height="903" alt="image" src="https://github.com/user-attachments/assets/23896d45-9f25-49d6-9a77-2e28ed59385c" />

https://medium.com/@zaiinn440/one-stop-guide-for-production-recommendation-systems-9491f68d92e3

下面是一轮 **“Snap 面试官连环追问模拟（CNN + Multimodal）”**，风格完全贴近 **Snap Camera / Lens / Multimodal MLE** 面试。
我会按 **真实面试节奏** 来：**面试官追问 → 你该怎么答 → L6 加分点 → 常见踩雷**。
你可以直接当 **mock interview 脚本** 用。

---

# 🎤 Scene Setting（面试官开场）

> *“You mentioned CNN-based perception and multimodal LLMs for Snap Camera and Lens. Let’s go deeper.”*

---

## Round 1：Vision 基础但带坑（5–7 分钟）

### Q1️⃣ 面试官

**“Why CNNs instead of ViTs for Snap Camera perception?”**

#### ✅ 合格回答（L5）

> CNNs are better suited for real-time, on-device inference. They have lower latency, smaller memory footprint, and more predictable performance across devices, which is critical for camera-first products like Snap.

#### ⭐ L6 加分

> This is fundamentally a **latency–semantic capacity trade-off**. For on-device, frame-by-frame perception, CNNs give us deterministic P99 latency and stable FPS. ViTs make more sense server-side or for offline understanding tasks where semantic richness matters more than strict real-time constraints.

#### ❌ 踩雷

* “CNN is better than ViT in general”
* “ViT is too new”

---

### Q2️⃣ 面试官

**“What vision tasks are actually critical for Lens recommendation?”**

#### ✅ 合格回答

> We don’t use CNNs to directly recommend lenses. CNNs provide structured perception signals—scene type, face attributes, segmentation masks—that become features for downstream retrieval and ranking.

#### ⭐ L6 加分

> CNN answers *“what the user is seeing”*, while ranking answers *“what we should show next”*. Mixing those responsibilities usually leads to brittle systems.

#### ❌ 踩雷

* “CNN directly ranks lenses”
* “We feed raw pixels to ranking”

---

## Round 2：系统与实时性（Snap 非常看重）

### Q3️⃣ 面试官

**“How do you guarantee camera smoothness if CNN inference is heavy?”**

#### ✅ 合格回答

> We use lightweight architectures like MobileNet/EfficientNet, quantize to INT8, and deploy different models based on device capability. We also decouple vision inference from ranking so delays don’t block rendering.

#### ⭐ L6 加分

> Camera systems are designed assuming failure. If perception lags, we fall back to cached or previous-frame signals. A single dropped frame is worse than slightly stale semantics.

#### ❌ 踩雷

* “Add more GPU”
* “Accuracy is more important than FPS”

---

### Q4️⃣ 面试官

**“How do you handle temporal consistency across frames?”**

#### ✅ 合格回答

> We use temporal smoothing—EMA over embeddings or labels—and optionally sequence models to stabilize predictions across frames.

#### ⭐ L6 加分

> Single-frame accuracy is less important than **temporal coherence**. Jittery perception breaks user trust much faster than slightly incorrect labels.

---

## Round 3：Multimodal LLM 介入（重点）

### Q5️⃣ 面试官

**“Where exactly does a multimodal LLM fit into this pipeline?”**

#### ✅ 合格回答

> The multimodal LLM sits above perception. CNNs or ViTs extract visual embeddings; the LLM performs cross-modal reasoning—like interpreting a scene and mapping it to user intent or creative suggestions.

#### ⭐ L6 加分

> The key is **decoupling**: perception models convert pixels to symbols; LLMs operate on symbols, not raw frames. This keeps the system modular and debuggable.

#### ❌ 踩雷

* “LLM replaces CNN”
* “End-to-end multimodal is better”

---

### Q6️⃣ 面试官

**“Why not let the LLM rank lenses directly?”**

#### ✅ 合格回答

> LLMs are too slow and unstable for large-scale ranking. They’re better used to generate features, constraints, or reasoning signals for a traditional ranker.

#### ⭐ L6 加分

> LLMs are **copilots, not engines**. Deterministic ranking with guardrails belongs to numerical models; LLMs add semantic lift where it matters.

#### ❌ 踩雷

* “LLMs are smarter so they should rank”
* “We just cache LLM results”

---

## Round 4：Multimodal Alignment（CLIP-style）

### Q7️⃣ 面试官

**“How do you align images and text in Snap?”**

#### ✅ 合格回答

> We use contrastive learning, similar to CLIP, where image and text embeddings are trained to be close for matched pairs and far apart otherwise.

#### ⭐ L6 加分

> Alignment quality depends more on **data diversity and hard negatives** than model size. Lens descriptions are noisy, so we often augment them with auto-generated captions and user behavior signals.

#### ❌ 踩雷

* “Just use off-the-shelf CLIP”
* “More data always fixes alignment”

---

### Q8️⃣ 面试官

**“What if Lens text descriptions are wrong or missing?”**

#### ✅ 合格回答

> We generate captions from vision models, normalize text across languages, and correct embeddings using apply/share feedback.

#### ⭐ L6 加分

> User behavior is the strongest alignment signal. If users consistently apply a lens in a given scene, that implicitly corrects semantic mismatches.

---

## Round 5：Safety & Hallucination（Snap 必考）

### Q9️⃣ 面试官

**“How do you prevent multimodal hallucinations?”**

#### ✅ 合格回答

> We enforce grounding by requiring visual evidence, constrain outputs to structured schemas, and fall back when confidence is low.

#### ⭐ L6 加分

> In creative tools, abstention is a valid output. A system that sometimes says “I’m not sure” is safer than one that’s always confident.

#### ❌ 踩雷

* “We trust the model”
* “RLHF solves hallucination”

---

## Round 6：Evaluation & Failure Modes

### Q1️⃣0️⃣ 面试官

**“How do you evaluate CNN + multimodal systems offline and online?”**

#### ✅ 合格回答

> Offline we evaluate perception accuracy and alignment metrics; online we look at apply/share uplift, session depth, and safety violations.

#### ⭐ L6 加分

> The real signal is **counterfactual consistency**: do offline gains predict online uplift? If not, the problem is usually feature integration, not the model itself.

---

### Q1️⃣1️⃣ 面试官（终极问题）

**“When should Snap *not* use multimodal LLMs?”**

#### ⭐ L6 完美答案

> When latency is critical, the task is rule-based, or ROI is unclear. Choosing not to use an LLM is often the most senior architectural decision.

#### ❌ 踩雷

* “Always use LLMs”
* “Because it’s trendy”

---

# 🧠 30 秒终极总结（CNN + Multimodal，Snap 风格）

> “In Snap’s camera-first ecosystem, CNNs handle real-time perception with strict latency guarantees, while multimodal LLMs sit above perception to provide semantic reasoning and creative assistance. The two are decoupled by design: CNNs convert pixels into stable signals; LLMs operate on those signals without blocking the camera or ranking pipeline. This separation lets us scale safely, iterate quickly, and maintain user trust.”

---

如果你愿意，下一步我可以继续 **高强度 Snap 模拟**：

* 🎭 **你来答，我来当 Snap 面试官追问（实战模式）**
* 🧩 给你一份 **“Snap CNN + Multimodal 必死坑清单”**
* 📐 帮你画一张 **Camera + Lens 多模态系统终极白板图**

你想直接进入 **实战 mock** 吗？

<img width="1788" height="1346" alt="image" src="https://github.com/user-attachments/assets/3a613442-1950-42d5-8d39-0c897d1e3b2f" />

I generated images with the prompt: 'dedicated diagram image of a whiteboard-style ML system design for Snap filter recommendation, showing candidate generator, fine ranker, and diversity-oriented reranker with latency budgets and diversity metrics, inspired by Alex Xu's approach'

-- follow up questions and answers about logging please

下面是一套 **Snap 面试高频 · Logging & Observability 连环 follow-up 问答库**，完全贴合 **Camera / Lens / 推荐 / CNN / Multimodal / MTML** 场景。
风格是 **“面试官刁钻追问 → 标准答案 → L6 加分 → 踩雷点”**，你可以直接用来 mock。

---

# 一、Logging 的总体设计

## Q1️⃣ 面试官

**“Why is logging critical in Snap’s recommendation and camera systems?”**

### ✅ L5 合格回答

> Logging is critical for debugging, offline training, and evaluation. Without logs, we can’t understand why a model made a decision or reproduce issues.

### ⭐ L6 加分回答

> Logging is the **only source of truth** connecting perception, ranking, and user outcomes. In Snap, where camera, ML models, and policy interact in real time, logs are how we ensure debuggability, accountability, and long-term model health.

### ❌ 踩雷

* “Logging is mainly for offline training”
* “Metrics dashboards are enough”

---

# 二、你到底 log 什么？（必考）

## Q2️⃣ 面试官

**“What exactly do you log for a Lens recommendation request?”**

### ✅ L5 合格回答

> We log request context, candidate lists, ranking scores, impressions, and user interactions like apply or share.

### ⭐ L6 加分回答（结构化）

> I log **five layers**, all tied by `request_id`:

1. **Request / Context**: user, session, device, entry surface, experiment id
2. **Candidate logs**: recall source, candidate IDs, recall scores, latency
3. **Ranking logs**: model version, feature schema, per-head scores (sampled)
4. **Impression logs**: final top-K with positions and policy flags
5. **Outcome logs**: apply / share / hide / crash with impression_id

> This enables full replay and counterfactual analysis.

### ❌ 踩雷

* “We log predictions”
* “We only log final results”

---

## Q3️⃣ 面试官

**“Do you log all features used by the model?”**

### ✅ L5 合格回答

> No. Logging all features would be too expensive. We log a subset.

### ⭐ L6 加分回答

> Never log full feature vectors.
> I log:

* feature **presence / missing flags**
* a small **explainability subset**
* a **feature schema version hash**
  This avoids privacy issues and still allows debugging training-serving skew.

### ❌ 踩雷

* “Yes, for reproducibility”
* “We log raw images”

---

# 三、Logging × CNN / Vision（Snap 专属）

## Q4️⃣ 面试官

**“What do you log from CNN perception models?”**

### ✅ L5 合格回答

> We log model outputs like scene labels or face attributes.

### ⭐ L6 加分回答

> We never log raw images.
> We log:

* high-level semantic labels (scene, face present)
* embedding version + stats (norm, age)
* confidence scores
  This balances privacy, debuggability, and safety auditing.

### ❌ 踩雷

* “We store frames for debugging”
* “We log pixel data”

---

## Q5️⃣ 面试官

**“How do you debug perception jitter or instability?”**

### ⭐ L6 标准答案

> I log **temporal sequences** of perception outputs (sampled), then check:

* frame-to-frame variance
* confidence oscillation
* correlation with device / lighting
  This helps distinguish model instability from sensor noise.

---

# 四、Logging × MTML 精排（高频）

## Q6️⃣ 面试官

**“How do you log MTML predictions without exploding storage?”**

### ✅ L5 合格回答

> We sample ranking logs and only log top candidates.

### ⭐ L6 加分回答

> We:

* log **only top-N** items
* sample traffic (e.g., 1–5%)
* log **per-head outputs**, not intermediate tensors
  This allows head-level debugging (apply vs hide) at manageable cost.

### ❌ 踩雷

* “We don’t log per-head outputs”
* “Storage is cheap”

---

## Q7️⃣ 面试官

**“How do you detect that one MTML head is dominating others?”**

### ⭐ L6 标准答案

> By logging:

* per-head score distributions
* per-head online metrics

* 补充一个常用tradeoff:
embedding和模型一起训练可能会得到更好的效果，但是embedding layer通常是模型中参数最多的部分，所以latency限制比较紧张， 或者separately trained (pretrained) embedding足够好的的时候也可以在two-tower 训练的时候freeze emebdding

https://www.1point3acres.com/bbs/thread-1158161-1-1.html

https://www.1point3acres.com/bbs/thread-1155409-1-1.html

*** https://www.1point3acres.com/bbs/thread-1088518-1-1.html
* score correlation over time
  If apply ↑ but hide ↑, logs immediately show head imbalance.

---

# 五、Logging × Evaluation（OPE / A/B）

## Q8️⃣ 面试官

**“What logging is required for counterfactual evaluation?”**

### ✅ L5 合格回答

> We need logged actions, rewards, and propensities.

### ⭐ L6 加分回答

> Specifically:

* action (lens_id, position)
* logging policy probability (propensity)
* candidate set context
* outcome labels
  Without **accurate propensities**, IPS / DR estimates are meaningless.

### ❌ 踩雷

* “We can infer propensities”
* “We don’t need randomization”

---

## Q9️⃣ 面试官

**“How do you validate offline OPE against online A/B?”**

### ⭐ L6 标准答案

> I log OPE predictions alongside experiment IDs, then compare predicted uplift vs actual A/B uplift. Persistent divergence signals bias or logging errors, not model quality.

---

# 六、Failure & Debugging（真实面试最像）

## Q1️⃣0️⃣ 面试官

**“Apply rate dropped overnight. What logs do you check first?”**

### ✅ L5 合格回答

> I’d check model metrics and recent deployments.

### ⭐ L6 加分回答（顺序很重要）

1. policy filter hit rate (误杀？)
2. feature missing / timeout logs
3. recall coverage by source
4. model / feature version mismatch
5. client-side or UI changes

> Logging lets me rule out whole classes of failures quickly.

### ❌ 踩雷

* “Rollback the model immediately”
* “Probably data drift”

---

## Q1️⃣1️⃣ 面试官

**“How do you support full request replay?”**

### ⭐ L6 标准答案

> All logs share a `request_id`. Given a request, I can reconstruct:
> context → candidates → scores → final ranking → outcomes.
> This is essential for trust, debugging, and incident reviews.

---

# 七、Privacy & Compliance（Snap 非常敏感）

## Q1️⃣2️⃣ 面试官

**“How do you balance logging and user privacy?”**

### ✅ L5 合格回答

> We avoid logging raw data and anonymize user IDs.

### ⭐ L6 加分回答

> Principles:

* log **signals, not content**
* hash or bucket user identifiers
* strict TTL and access control
* policy-driven redaction
  Logging must be **privacy-first by design**, not an afterthought.

### ❌ 踩雷

* “Engineers only access logs”
* “We delete later”

---

# 八、终极问题（Staff / L6 分水岭）

## Q1️⃣3️⃣ 面试官

**“What’s the biggest logging mistake you’ve seen?”**

### ⭐ L6 完美答案

> Logging predictions without logging **decisions and context**.
> Without knowing *what alternatives were considered*, logs are useless for learning or accountability.

---

# 🧠 30 秒 Logging 总结（Snap 风格）

> “In Snap systems, logging is not an afterthought—it’s the backbone of debugging, evaluation, and trust. I design logs to capture context, candidate alternatives, model decisions, and outcomes, all tied by request IDs. I avoid logging raw content, sample aggressively, and log just enough structure to replay decisions, support counterfactual evaluation, and detect failures early.”

---

如果你愿意，下一步我可以：

* 🎭 **做一轮 Logging 专项 mock（你答，我追问）**
* 🧾 给你一份 **Snap Logging 必死坑清单**
* 🧩 把 logging + feature store + OPE 画成一张 **终极白板图**

你想继续哪一个？


