# JS 阈值驱动的跨轮阶段蒸馏：按探索阶段动态分配学生计算

> 研究问题：怎样识别并迁移深轮教师中“对正确性有贡献、且学生学得动”的关键跨轮探索，并按探索阶段动态分配学生循环，在减少平均推理开销的同时保留推理能力？
>
> 本文是一份研究方案，不是实验报告。方案更新于 2026-11-13。当前主路线使用 JS 阈值划分教师阶段，学生目标轮数随阶段数变化；固定学生预算下的 JS Top-S 选点移至对比实验。论文与组件支持不等于完整方案已被验证。

## 1. 研究背景与要解决的问题

Ouro、Huginn 等循环语言模型在生成一个 token 前，会重复运行共享 Transformer 模块。更多循环可能改善推理，也会增加计算开销。[R2, R3]

LoopOPD 让浅轮学生生成解答，再由深轮教师在相同学生前缀上提供预测分布。例如，第四轮教师监督第二轮学生。[R1] 但只学习教师终点，可能不足以迁移教师中间的关键修正；反过来，把所有中间轮都交给学生，也可能重复学习变化很小的预测。[R4, R5, R23]

我们将同一 token 输出前，多轮内部计算逐步调整表示和预测的过程称为“跨轮隐式探索”。希望将预测接近的连续轮压缩为一个阶段，把明显转折保留下来，让学生每轮承接一个阶段结果。

因此，主路线不再预先规定每个 token 必须使用两轮、八轮或十六轮。**先用一个 JS 阈值划分教师轨迹，再用阶段数量确定学生的目标循环数。** 平稳轨迹分段少，学生目标轮数少；转折较多的轨迹保留更多阶段。研究目标由“固定浅轮预算”调整为“按阶段动态压缩，改善精度与平均计算开销的关系”。

## 2. 具体方案面临的两个挑战

### 2.1 挑战一：怎样识别并迁移关键跨轮探索？

教师终点依赖前面的循环计算。Cross-Loop Query Handoff 的查询冻结与移植实验说明，部分跨轮更新确实承担具体计算作用。[R4] 因此，不能只因最终预测相同，就认为中间计算全部没有作用。

主方案用相邻轮 token 分布的 JS 衡量预测转折，按阈值分段，每段取末轮，依次监督学生。**JS 用于提出阶段划分，不直接证明该变化有益或不可替代。** 能否更好保留纠错能力、学生能否学会，需要与终点蒸馏、均匀映射等方法比较。[R5, R28, R29, R32]

### 2.2 挑战二：更深的教师不一定更好，监督应到哪轮？

增加循环可能收益饱和，也可能改错或在超出常用深度后退化。[R2, R3, R26] 主路线沿用基模已有的动态退出规则，在当前学生前缀上决定教师监督终点 $n_t$，并设置最大上限 $K$：

- Ouro 使用学习到的退出分布与累计退出概率阈值。[R2]
- Huginn 先采用原生的相邻预测 KL 收敛退出规则。[R3]

原生退出代表基模建议停止，不保证本轮正确或最优。有答案或验证器时，可增加一次教师动态作答的结果筛选；它筛选题目，不搜索多个固定深度，也不把独立作答的退出轮次复用为 OPD 标签。[R19]

## 3. 主体方案：原生退出定终点，JS 阈值定阶段数

**整体流程：学生按当前动态策略生成 → 教师读取同一学生前缀并选择退出终点 → 用 JS 阈值分段 → 段数作为学生目标轮数，段末作为监督目标 → 学生同时学习阶段预测和退出。**

| 符号 | 含义 |
|---|---|
| $K$ | 教师最大循环上限；第一版学生也用同一上限兜底，不强制每次运行到上限 |
| $n_t$ | 教师在当前 token 前缀上选择的监督终点，$1\leq n_t\leq K$ |
| $\tau$ | 教师相邻轮 JS 的分段阈值，同一设置下对所有 token 使用同一值 |
| $S_t$ | 教师分段后得到的阶段数，也是学生训练的目标循环数 |
| $\widehat S_t$ | 部署时由学生自己的退出头决定的实际循环数，可能与目标 $S_t$ 不同 |

### 3.1 第一步：得到同前缀的教师动态轨迹

学生按当前策略生成回答。教师读取相同学生前缀 $c_t=(x,y^{\mathrm{student}}_{<t})$，使用已启用的基模退出规则确定 $n_t$，保留前 $n_t$ 轮的预测分布 $q_1,\ldots,q_{n_t}$。[R1–R3]

读出使用模型原有输出路径：

- Ouro：循环末状态经过原有归一化、LM head、softmax。
- Huginn：经过完整 coda 预测路径与 softmax，不能将原始循环隐层直接接到 LM head。[M3, M4]

教师使用自己的状态与缓存；这里不为每个中间轮另外生成完整文本。教师独立作答时的退出轮数，也不直接代替当前学生前缀上的 $n_t$。

### 3.2 第二步：只用 JS 阈值划分阶段

计算相邻教师轮的变化：

\[
J_{t,d}=\operatorname{JS}(q_{d-1},q_d),\qquad d=2,\ldots,n_t.
\]

**保留所有超过阈值的连接作为分段边界，不排序、不做 Top-S 截断：**

\[
\mathcal B_t=\{d:J_{t,d}>\tau\},\qquad
S_t=1+|\mathcal B_t|.
\]

在连接 $d-1\rightarrow d$ 处分段，取前一段末轮 $d-1$；最后一段取教师退出轮 $n_t$。按顺序得到阶段末轮：

\[
(e_{t,1},\ldots,e_{t,S_t})
=\operatorname{sort}\left(\{d-1:d\in\mathcal B_t\}\cup\{n_t\}\right).
\]

这意味着：

- 所有相邻变化都不超过阈值：一段，$S_t=1$，只学习教师终点。
- 出现一个超阈值转折：两段，学生目标两轮。
- 出现两个超阈值转折：三段，学生目标三轮。
- 每个连接都超过阈值：$S_t=n_t$，该位置暂不压缩。

阈值 $\tau$ 在开发集确定。第一版固定 softmax 温度为 1，JS 使用自然对数；不同基模可分别校准阈值。$\tau$ 越小，保留的阶段通常越多；$\tau$ 越大，压缩越强。相邻变化小也可能累积成明显漂移，这是当前分段规则需要通过实验检查的局限。

### 3.3 第三步：每个学生循环学习一个教师阶段

对于当前前缀，学生目标运行 $S_t$ 轮：

| 学生循环 | 对应教师监督 |
|---|---|
| 第 1 轮 | 第一段末轮 $e_{t,1}$ 的分布 |
| 第 2 轮 | 第二段末轮 $e_{t,2}$ 的分布 |
| …… | …… |
| 第 $S_t$ 轮 | 最后一段末轮，即教师退出轮 $n_t$ 的分布 |

沿用 LoopOPD 的 reverse KL，对每个 token 的阶段损失取平均：

\[
\mathcal L_{\mathrm{stage},t}
=\frac{1}{S_t}\sum_{j=1}^{S_t}
D_{\mathrm{KL}}\!\left(
p_S^{(j)}(\cdot\mid c_t)\,\Vert\,
q_{e_{t,j}}(\cdot\mid c_t)
\right).
\]

按 $S_t$ 平均，避免阶段多的 token 仅因目标数量更多而占据更大损失权重。教师分布、边界和目标轮数停止梯度；只更新学生。若当前学生 rollout 提前退出，训练回放时补算监督所需的学生循环，并将其计入训练成本。

### 3.4 第四步：让学生学会自己退出

**教师分段只能提供训练标签；部署时学生看不到教师，不能直接知道 $S_t$。** 因此增加一个简单的退出学习任务：

    学生第 1～S_t−1 轮：继续
    学生第 S_t 轮：停止

Ouro 复用已有 gate 作为学生退出头；Huginn 增加一个读取循环末状态的线性二分类头。按上述标签训练二元交叉熵 $\mathcal L_{\mathrm{stop},t}$，与阶段蒸馏联合优化：

\[
\mathcal L=\mathbb E_t[
\mathcal L_{\mathrm{stage},t}
+\beta\mathcal L_{\mathrm{stop},t}].
\]

第一版学生在每轮的停止概率达到 0.5 时退出，未达到则运行至上限 $K$。这是学生学到的停止策略；教师仍使用各基模原有的退出策略。首批 rollout 可沿用基模原有的动态退出配置，随后使用更新后的学生策略。

用 oracle 深度标签训练轻量退出决策器，可参考 Think-at-Hard、CALM 与冻结轨迹 gate 训练工作；这里以 JS 阶段数构造停止标签，是我们需要验证的设计。[R26, R27, R36]

部署时只运行学生，实际轮数记为 $\widehat S_t$。退出学习可能不准确，因此不能把训练标签 $S_t$ 的均值直接当作推理成本，必须实测 $\widehat S_t$ 与耗时。

### 3.5 例子：同样六轮教师，可以对应不同学生轮数

以下仅为说明，设教师当前退出轮次 $n_t=6$，JS 阈值 $\tau=0.05$：

| 相邻轮 JS | 1→2 | 2→3 | 3→4 | 4→5 | 5→6 | 分段与学生目标 |
|---|---:|---:|---:|---:|---:|---|
| token A | 0.01 | 0.02 | 0.30 | 0.01 | 0.02 | [1–3] [4–6]；学生两轮学 3、6 |
| token B | 0.01 | 0.20 | 0.01 | 0.18 | 0.01 | [1–2] [3–4] [5–6]；学生三轮学 2、4、6 |
| token C | 0.01 | 0.02 | 0.03 | 0.01 | 0.02 | [1–6]；学生一轮学 6 |

所以学生目标轮数由实际分段决定，不必为了统一两轮而舍弃 token B 的一个阶段，也不必让 token C 运行两轮。实际推理能否学到这种分配，由学生退出结果检验。

## 4. 方案参考及证据范围

| 工作 | 可以借鉴什么 | 不能直接推出什么 |
|---|---|---|
| LoopOPD [R1] | 同学生前缀的跨轮 on-policy 蒸馏、reverse KL | 其固定深浅轮结果不等于本方案动态学生已有效 |
| Ouro/Huginn [R2, R3] | 共享参数循环、原生动态退出、循环状态与缓存 | 支持动态不代表默认开启，也不代表退出即正确 |
| Cross-Loop Query Handoff [R4] | 部分跨轮修正确有具体计算作用 | 不证明所有 JS 大的变化都必要 |
| Prediction Dynamics [R28] | 区分纠错、改错、错换错 | 变化量不等于收益 |
| F-PABEE / PABEE [R35, R34] | 分布稳定与连续稳定判据 | 原任务是分类；F-PABEE 的 JSKD 不等于标准 JS |
| Think-at-Hard [R26] | 参考标签监督轻量迭代决策器 | 它是改造的 Qwen3，不是原生 Ouro/Huginn 上的同一方法 |
| Adaptive Depth / CALM [R27, R36] | 退出读出、轻量停止学习与校准 | 不保证阶段数标签就是最优预算 |
| PKD / RAIL-KD [R23, R29] | 中间目标和简单映射基线 | 不直接解决循环探索阶段的识别 |

已有论文支持组件和动机；“阈值分段 → 阶段蒸馏 → 学习动态退出”的完整组合，需要我们自己的实验。

## 5. 设计边界与对照思路

### 5.1 固定预算的 JS Top-S 改为对比实验

旧方案先固定学生预算 $S$，在超阈值连接中最多保留 $S-1$ 个最大 JS 边界，强制最多分成 $S$ 段。它适用于严格固定轮次的场景，但可能删掉超出预算的转折。

主方案只使用阈值：所有超阈值边界都保留，$S_t$ 随段数变化。固定预算 Top-S 保留为对照，比较不同固定预算的精度与成本曲线。为避免命名混淆，旧方案严格说是“最多 Top-(S−1) 个边界，再保留最终段末”，不是 Top-S 个边界。

### 5.2 选段末，不代表删除教师中间计算

教师第 3、6 轮作为目标时，仍需完整执行相应前置计算才能得到真实状态。学生学习这些阶段目标，使用自己的隐藏状态与 KV，不直接继承教师缓存，也不运行教师跳轮路径。[R2, R3, M3, M4]

段末提供统一的阶段结束目标；是否优于段首、随机映射或均匀映射，通过对照实验回答。[R29, R32] 第一版不增加轮间加权融合和隐层辅助损失。

### 5.3 阈值分段不保证每个 token 都能压缩

当所有连接都超阈值时，$S_t=n_t$；阶段多可能来自有用修正，也可能来自无益振荡。我们不强制每个位置都压缩，也不把 JS 直接称为正确性收益。

“学生学得动”通过阶段预测拟合与实际答题结果检验，不通过多路径试训挑选。相邻小变化的累积漂移、学生停止过早、保留噪声阶段，是重点误差来源；它们不另组成一套复杂评分器。

## 6. 对比实验与成本

### 6.1 核心实验

| 实验 | 方法 | 主要回答什么 |
|---|---|---|
| A | 原生动态基模，直接推理 | 不蒸馏时的精度与计算开销 |
| B | 原始 LoopOPD，固定学生轮次 | 直接跨轮蒸馏基线；比较多个预算 |
| C | 动态学生，只学习教师终点，使用同样的 $S_t$ 退出标签 | 只学退出与终点，是否已经足够 |
| D | 使用同样的阶段数与退出标签，均匀选教师阶段末轮 | 需要 JS 定位，还是增加中间目标即可 |
| E | JS 阈值分段、段末蒸馏与学生退出学习 | 主方案 |
| F | 固定学生预算，超阈值边界中选最大 Top-(S−1) | 用户指定保留的固定预算对照 |

另可加入保持顺序的随机选点，参考 RAIL-KD；与 D 一样保持目标数量可比。[R29, R32] C/D/E 的退出标签规则相同，用来区分“动态计算分配”和“阶段目标选择”的作用。

全部方法使用相同基模、数据、教师上限、结果筛选设置和可比训练资源。动态方法与固定预算方法比较**精度—实际平均循环数、精度—实际耗时**曲线，不要求平均轮数天然相同。通过开发集选择可比开销的运行点；最终测试集不调阈值。

### 6.2 与问题接近的论文 baseline

| 论文 | 与我们的共同问题 | 实验定位 |
|---|---|---|
| LoopOPD [R1] | 把深轮能力转移给更省计算的循环学生 | 最直接的蒸馏基线 |
| Ouro/Huginn 原生退出 [R2, R3] | 按输入与 token 分配循环计算 | 不做阶段蒸馏的动态推理基线 |
| LoopRPT [R15] | 改善 Ouro 的中间预测质量与动态计算效率 | 条件允许时加入的原生 Ouro 方法对比；训练机制不同 |
| Think-at-Hard [R26] | 选择何时继续潜在循环、避免无益迭代 | 相关动态计算工作；其改造 Qwen3 结果不能直接与 Ouro/Huginn 数值横比 |
| PKD / RAIL-KD [R23, R29] | 利用中间教师目标帮助学生 | 将其有序均匀或随机映射思想迁移到同基模，标明是适配基线 |

这些工作的目标与约束并非完全相同。特别是动态退出本身不等于关键跨轮探索的蒸馏，应区分直接复现、同基模适配和相关工作讨论。

### 6.3 必须报告的成本与指标

- 完整测试集准确率、生成长度、学生实际平均循环数及分位数、端到端延迟。
- 教师退出深度 $n_t$、阶段数 $S_t$、实际学生深度 $\widehat S_t$ 的分布；两种学生轮数不要混报。
- 相对教师轨迹的阶段压缩比例、$S_t=n_t$ 的位置比例，以及达到最大上限的比例。
- 训练总开销：教师轨迹和逐轮读出、学生补算、退出头训练、可选结果筛选。
- JS 阈值变化时的精度与成本曲线，以及退出头对目标轮数的预测误差。

核对的 Ouro 发布实现可能先算全部配置轮，再选择退出读出；动态监督终点不自动带来实际少算。Huginn 的自适应接口也需处理批处理与缺失 KV。先保留所用实现的缓存语义，额外提前停算或调度优化单独验证。[M3, M4, R40]

Cross-Loop Query Handoff 式冻结与恢复只作为小规模机制验证，不参与主训练流程。[R4]

## 7. 第一版实现约定

- **教师动态退出**：Ouro 上限先用 4；Huginn 上限先以 32 为参考。明确启用退出规则。Ouro-1.4B-Thinking 的默认退出阈值 1.0、Huginn 自适应接口默认 criterion="none"，均不能直接当成已启用提前退出。[M3–M5]
- **学生动态循环**：主方案不固定二轮、八轮或十六轮。第一版学生最大上限与教师 $K$ 相同，以容纳全部分段；上限是兜底，实际由学生退出头选择。
- **JS 阈值**：保留所有 $J_{t,d}>\tau$ 的边界，取段末，$S_t=1+|\mathcal B_t|$；不限制为 Top-S。阈值按开发集上的精度与实际成本确定。
- **学生退出**：Ouro 复用已有 gate 参数，Huginn 增加线性退出头；训练以阶段末“停止”、前面“继续”为标签。主方案的学生停止规则是所提出训练方法的一部分。
- **边界情况**：$n_t=1$ 或无超阈值边界时，目标一轮；所有连接超阈值时保留全部阶段。推理预测不准时仍受 $K$ 上限约束。
- **OPD 与筛选**：师生读取相同学生前缀。结果筛选只缓存题目是否保留；终点和分段在当前前缀上重算。退出规则、温度、JS 阈值、生成配置与验证器版本均记录。
- **训练深度背景**：Ouro 早期八轮、后续主要四轮；Huginn 使用随机深度训练，标称约 32、最后八轮反传。这些训练设置不能当作每个 token 的实际退出轮数。[R2, R3]
- **当前范围**：先使用预测分布和简单退出头，不增加 U-Space、多教师、加权融合或多路径试训。LoopOPD 的教师参数刷新与本方案的动态循环数是两个独立维度。

## 8. 参考文献

以下编号与前文一致。近期工作多为预印本；它们支持动机和组件选择，不能代替我们自己的实验。

- **[R1] Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models.** [arXiv:2610.10623v2](https://arxiv.org/abs/2610.10623v2)。核心对照；固定四轮教师/二轮学生、冻结与动态教师、学生深度消融、共享参数局部理论。重点看 §4.3、Table 5、训练配置与 Appendix A。
- **[R2] Scaling Latent Reasoning via Looped Language Models.** [arXiv:2510.25741v5](https://arxiv.org/abs/2510.25741v5)。Ouro 基模；Table 1、§4.3 的八轮到四轮训练历史，Table 12/13 的 Thinking 深度曲线。本文数字采用 v5。
- **[R3] Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.** [arXiv:2502.05171v2](https://arxiv.org/abs/2502.05171v2)。Huginn 架构、随机深度训练、截断反传及自适应计算；§3、§4.1 与深度/上下文分析。
- **[R4] Cross-Loop Query Handoff: How Later Loops Redirect Two-Hop Retrieval in the Ouro Looped Language Models.** [arXiv:2610.34639v1](https://arxiv.org/abs/2610.34639v1)。查询冻结/移植、匹配控制和任务局限；为“有计算贡献的更新”提供干预范式。
- **[R5] Monitoring latent reasoning in looped language models.** [arXiv:2610.18127v1](https://arxiv.org/abs/2610.18127v1)。中间读出、正确性与最终一致性的区别；Ouro/Huginn 行为差异、跨轮读出可迁移性的限制。特别注意 teacher-forced 分析的范围。
- **[R10] Less is More: Task-aware Layer-wise Distillation for Language Model Compression.** [arXiv:2210.01351v3](https://arxiv.org/abs/2210.01351v3)。TED；任务相关过滤，减少有限学生不需要或难吸收的信息。支持“选择内容”，不直接证明循环更新可压缩。
- **[R11] Improved Knowledge Distillation via Teacher Assistant.** [arXiv:1902.03393v2](https://arxiv.org/abs/1902.03393v2)。能力差距与中间教师；作为分阶段、较小跨度监督的依据。
- **[R12] BERT-EMD: Many-to-Many Layer Mapping for BERT Compression with Earth Mover’s Distance.** [arXiv:2010.06133v1](https://arxiv.org/abs/2010.06133v1)。多对多层映射；可借鉴匹配方法，不能直接等同跨循环因果路径。
- **[R13] ALP-KD: Attention-Based Layer Projection for Knowledge Distillation.** [arXiv:2012.14022v1](https://arxiv.org/abs/2012.14022v1)。注意力式层融合；支持多层信息利用的可行性，无法替代任务收益判断。
- **[R14] U-Space: Uncovering When and Why Uncertainty Arises in Language Models.** [arXiv:2610.09087v1](https://arxiv.org/abs/2610.09087v1)。J-Lens/Jacobian 语义不确定性基和 U-Lens；用于可选特征。原实验不包含 Ouro/Huginn，需重新校准。
- **[R15] LoopRPT: Reinforcement Pre-Training for Looped Language Models.** [arXiv:2603.19714v2](https://arxiv.org/abs/2603.19714v2)。EMA 参考教师、退出深度、next-token 进展与时间成本；动态循环相关的重要方法对照。
- **[R18] Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.** [arXiv:2604.13016v2](https://arxiv.org/abs/2604.13016v2)。教师能力、推理模式兼容性与知识迁移；支持“更高分教师不自动等于更好训练教师”。
- **[R19] Outcome-Guided On-Policy Self-Distillation.** [arXiv:2610.05070v1](https://arxiv.org/abs/2610.05070v1)。结果、教师熵和分歧联合筛选；应作为接近的已有工作，不能把结果+不确定性门控单独声称为创新。
- **[R20] PHF: Privileged Hidden Flow for On-Policy Self-Distillation.** [arXiv:2606.29340v1](https://arxiv.org/abs/2606.29340v1)。隐藏转移方向/几何监督；可借鉴关系蒸馏，其 token 间设置与本文 loop 间设置不同。
- **[R23] Patient Knowledge Distillation for BERT Model Compression.** [arXiv:1908.09355v1](https://arxiv.org/abs/1908.09355v1)。中间状态蒸馏的基础参考；有序监督需与此类已有方法区分。
- **[R26] Think-at-Hard: Dynamic Looped Transformers for Improved Reasoning.** [arXiv:2511.08577v4](https://arxiv.org/abs/2511.08577v4)。latent overthinking、depth-aware LoRA、duo-causal attention 与静态 oracle 训练；其 Ouro 风格对照为基于 Qwen3-Base/Open-R1 的重实现，不能混同官方 Ouro checkpoint。
- **[R27] Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts.** [arXiv:2607.20519v1](https://arxiv.org/abs/2607.20519v1)。包括 Ouro-1.4B/2.6B；区分轨迹形成与退出读出，冻结轨迹拟合 gate，比较 post-hoc confidence 与 ponder gate；不能据此推出某种信号保证正确退出。
- **[R28] Prediction Dynamics in Depth-Recurrent Language Models.** [arXiv:2609.21383v1](https://arxiv.org/abs/2609.21383v1)。Huginn/Ouro 的候选答案分数与轨迹，§4.5 区分纠错、改错和错换错；终点一致性不等于正确性，回看完整轨迹的分析不等于在线保证。
- **[M3] Ouro-1.4B-Thinking 官方模型实现。** [modeling_ouro.py](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/blob/main/modeling_ouro.py)。用于核对循环×物理层 KV 索引和循环状态传递；与 [R2] 的缓存实验结合理解。
- **[M4] Huginn-0125 官方模型实现。** [raven_modeling_minimal.py](https://huggingface.co/tomg-group-umd/huginn-0125/blob/main/raven_modeling_minimal.py)。predict_from_latents 接口执行归一化、coda、归一化与 LM head；用于核对中间循环状态的原生预测读出。
- **[M5] Ouro-1.4B-Thinking 官方模型说明。** [模型卡](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking)。Recurrent Steps and Adaptive Exit 说明 total_ut_steps 默认 4，early_exit_threshold 默认 1.0、降低阈值鼓励更早退出；是否提前停止实际计算需结合 [M3] 核对。


- **[R29] RAIL-KD: RAndom Intermediate Layer Mapping for Knowledge Distillation.** [arXiv:2109.10164v2](https://arxiv.org/abs/2109.10164v2)。按 epoch 随机选教师层，并保持对应顺序；降低映射搜索负担，是软对齐必须比较的低成本基线。原任务为 BERT/GLUE 等，不是循环模型。
- **[R30] Improving Reasoning Capabilities in Small Models through Mixture-of-Layers Distillation with Stepwise Attention on Key Information.** [arXiv:2604.15701v1](https://arxiv.org/abs/2604.15701v1)。MoLSAKI；聚合关键 token 的分步注意力，教师权重使用注意力变化量，学生采用可学习路由。它不是“每条映射各训一次”，也未直接研究跨循环压缩。
- **[R31] Align-to-Distill: Trainable Attention Alignment for Knowledge Distillation in Neural Machine Translation.** [arXiv:2403.01479v3](https://arxiv.org/abs/2403.01479v3)。A2D；训练中学习跨层注意力头对齐，不预设固定映射；完整注意力图的计算不适合未经评估直接用于长序列。
- **[R32] Revisiting Intermediate-Layer Matching in Knowledge Distillation: Layer-Selection Strategy Doesn’t Matter (Much).** [arXiv:2502.04499v2](https://arxiv.org/abs/2502.04499v2)。在 BERT/BART/T5 和部分 Qwen3 任务上研究多种映射策略，观察到差异常有限；不能直接外推所有循环轨迹，但要求我们比较简单与随机映射。


- **[R33] Two-Scale Latent Dynamics for Recurrent-Depth Transformers.** [arXiv:2509.23314v2](https://arxiv.org/abs/2509.23314v2)。§2、§4；GPT-2 风格分组循环模型，隐藏更新长度、相邻更新夹角与二阶差分，two-hit 停止规则，无需 LM head。不能宣称已在原生 Ouro/Huginn 证实。
- **[R34] BERT Loses Patience: Fast and Robust Inference with Early Exit.** [arXiv:2006.04152v3](https://arxiv.org/abs/2006.04152v3)。PABEE；连续预测不变的 patience，BERT/ALBERT 分类与回归。
- **[R35] F-PABEE: Flexible-patience-based Early Exiting for Single-label and Multi-label text Classification Tasks.** [arXiv:2305.11916v1](https://arxiv.org/abs/2305.11916v1)。§3；相邻预测分布距离与 patience 联合控制。其 JSKD 是 JS 风格交叉熵目标，不应直接写成与标准 JS 完全相同。
- **[R36] Confident Adaptive Language Modeling.** [arXiv:2207.07061v2](https://arxiv.org/abs/2207.07061v2)。CALM，§3.5 的 hidden-state cosine saturation、top-2 margin 和小型分类器；处理局部退出与序列约束、缺失隐藏态，并强调校准。
- **[R37] Consistent Accelerated Inference via Confident Adaptive Transformers.** [arXiv:2104.08803v2](https://arxiv.org/abs/2104.08803v2)。CATs；元一致性分类器与校准，保证范围主要是相对完整模型的一致性。
- **[R38] DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference.** [arXiv:2004.12993v1](https://arxiv.org/abs/2004.12993v1)。中间 off-ramp 的预测熵阈值，适合作为单点置信度基线。
- **[R39] FastBERT: a Self-distilling BERT with Adaptive Inference Time.** [arXiv:2004.02178v2](https://arxiv.org/abs/2004.02178v2)。自蒸馏早期分支、归一化熵式不确定性退出；分类模型参考，不是原生循环模型。
- **[R40] Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching.** [arXiv:2608.09444v2](https://arxiv.org/abs/2608.09444v2)。原生 Ouro/Huginn 的调度、KV 与实际延迟；支持开销核算，不提供关键探索的正确性判据。
