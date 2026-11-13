# 关键跨轮探索的阶段蒸馏：原生动态退出与段末监督

> 研究问题：在固定的浅轮推理预算下，怎样识别并转移深轮教师中“对正确性有贡献、且浅轮学生学得动”的计算更新？
>
> 本文是一份研究方案，不是实验报告。方案更新于 2026-11-13；论文核对截至 2026-11-12，并核对了官方代码的动态退出开关。下文把已有论文的观察、可以推导的条件性结论、以及需要我们验证的新假设分开表述。Ouro 的数值引用明确采用原论文 v5；不同论文的评测结果不作直接受控比较。

## 1. 研究背景与要解决的问题

循环语言模型（如 Ouro、Huginn）会在生成每个 token 前，把共享的 Transformer 模块重复运行多轮。增加循环有机会提高推理能力，但也增加了计算开销。因此，我们希望部署时只运行固定的少数轮，同时尽量保留深轮模型的能力。[R2, R3]

LoopOPD 提供了一种思路：让浅轮学生生成解答，再由同一模型的深轮教师，在相同文本前缀上提供预测分布，指导浅轮学生学习。例如，用第四轮教师监督第二轮学生。[R1] 这使模型可以利用额外循环产生的能力进行自蒸馏。

不过，深轮教师产生的变化并不都值得学生照单全收。有些中间变化可能是在修正错误、补充信息，为后续正确预测做准备；另一些变化可能重复已有信息，甚至把原本正确的预测改错。[R4, R5, R26, R28] 同时，即使教师的修正有用，浅轮学生也未必能在更少轮计算内直接学会。[R1, R10, R11]

因此，我们希望从深轮计算中挑出真正有价值、又适合浅轮学生学习的部分，把有限的学生计算预算用在这些更新上。本文要解决的总体问题是：

> **在固定的浅轮推理预算下，怎样识别并转移深轮教师中“对正确性有贡献、且浅轮学生学得动”的计算更新？**

这里“固定浅轮预算”只约束学生：部署时每个 token 始终运行预先设定的 $S$ 轮。训练阶段的教师启用基模已有的动态退出规则，在每个学生前缀上选择实际监督终点 $n_t$。主路线不逐题搜索最小正确轮次。Ouro 可先设置最多四轮、学生固定二轮；Huginn 可先设置最多 32 轮、学生固定八轮或十六轮。这些数值是实验上限或参考预算，不代表每个 token 都运行相同轮数。[R1–R3]

## 2. 具体方案面临的两个挑战

### 2.1 挑战一：重要的中间修正，可能没有被充分迁移，我们怎么迁移？

LoopOPD 主要让浅轮学生学习深轮教师在每个 token 位置的最终预测。但教师得到这个预测之前，可能已经经历了多次有用的修正。例如，先发现信息不足，再调整检索方向，最后得到正确答案。我们需要检验：**只对齐最深轮的预测分布，是否足以让浅轮学生吸收这些对正确性有贡献的中间更新？**

本文将模型在生成同一个 token 前，通过多轮内部计算逐步调整表示和预测的过程称为“跨轮隐式探索”。其中，为后续纠错或正确预测提供重要信息的部分，称为“关键跨轮探索”：有些中间轮看起来没有立即变好，却可能为下一轮继续计算提供条件。Cross-Loop Query Handoff 对后轮查询更新的干预，说明部分跨轮变化确实承担具体计算作用。[R4] 我们希望识别这些关键探索，并找到适合浅轮学生的迁移方式。

实现上有两点需要注意：一是直接匹配最深轮，可能跨越学生当前能学会的计算跨度；二是中间状态和 KV 记忆可能影响后续计算，压缩时需要保留相应功能。[R1–R3, R11] 这些是方案设计与验证的细节，不另立为总体研究问题。

### 2.2 挑战二：更深的教师，不一定是更好的监督，怎样控制监督范围？

增加循环并不保证预测持续改善。有时是超出模型主要训练深度后表现退化，有时是继续计算把正确预测改错，也可能只是后续收益已经饱和。[R2, R3, R26]

主路线沿用基模已有的退出机制：**由教师根据当前学生前缀决定监督到第几轮，记为 $n_t$。** Ouro 使用学习到的退出概率，Huginn 使用其已有的预测收敛规则。我们设置统一的最大循环上限 $K$，但不要求每个位置都监督到 $K$，也不先生成每个固定深度的完整解答。[R2, R3]

原生退出表达的是“基模认为可以停止”，不等于已经证明本轮答对或达到逐题最优。若有标准答案或验证器，可保留一次教师完整作答的结果筛选：教师按同一动态策略作答，答对的题用于该蒸馏项，答错的题跳过。该筛选不为 OPD 缓存题目级深度；训练时仍在当前学生前缀上重新确定 $n_t$。结果筛选不能保证每个中间预测都正确。[R19, R26–R28]

### 2.3 对应的总体思路

我们提出跨轮阶段蒸馏：**学生生成前缀后，教师按原生动态退出规则确定当前监督终点 $n_t$；在前 $n_t$ 轮内读取 token 分布，按预测变化分段，每段选择最后一轮，依次监督固定浅轮学生。** 有结果标签时，可增加一次教师动态作答的样本筛选。

后文按实际执行顺序展开：先生成学生前缀，再由教师自身规则选择退出轮次，随后在该范围内分段选点并蒸馏。中间修正能否因此更好地迁移，由实验检验。

## 3. 主体方案：原生退出定终点，JS 分段选中间监督点

**整体流程：固定浅轮学生生成 → 教师读取同一学生前缀，按原生规则确定退出轮次 n_t → 在前 n_t 轮内用 JS 分段、取段末 → 蒸馏固定浅轮学生。**

先统一三个数量：

| 符号 | 含义 |
|---|---|
| $K$ | 教师允许的最大循环数，同一实验设置中固定 |
| $n_t$ | 当前学生前缀上，原生退出规则选择的教师监督终点；不同 token 可以不同，且 $n_t\leq K$ |
| $S$ | 学生固定循环预算，例如 $S=2$；训练和部署时均不改变 |

$n_t$ 决定当前 token 的监督范围，$S$ 决定最多保留多少个阶段目标。下文以二轮学生说明。“跳循环”在这里指将一段教师计算压缩给一次学生更新；教师提供的第 $n_t$ 轮预测仍依赖前面各轮的完整计算。实际代码是否提前停算，另见开销说明。

### 3.1 第一步：学生生成前缀，教师用原生退出规则确定 n_t

学生固定运行 $S$ 轮生成回答。对每个 token 位置，教师读取同一个学生前缀 $c_t=(x_i,y^S_{<t})$，使用已启用的基模原生退出规则选择 $n_t$。

| 基模 | 原生动态退出依据 | 第一版教师上限 $K$ | 学生固定预算 $S$ |
|---|---|---:|---:|
| Ouro | 学习到的退出分布，其累计概率达到阈值 | 先用 4 | 2 |
| Huginn | 相邻循环预测分布收敛，先采用原生 KL 退出规则 | 先以 32 为参考 | 8 或 16，分别实验 |

例如，教师对第一个 token 在第二轮退出，对第二个 token 在第四轮退出，则对应 $n_1=2$、$n_2=4$，无需为这道题统一设定一个“最佳轮次”。退出阈值在开发数据上设定；未触发退出时使用第 $K$ 轮。[R2, R3]

两种基模都支持动态循环，但不代表加载后默认开启。核对的 Ouro-1.4B-Thinking 模型说明中，退出阈值默认 1.0，表示使用全部步骤；Huginn 的 generate_with_adaptive_compute 中 criterion="none" 默认不提前退出。实验必须明确开启并报告所用规则与阈值。Huginn 原论文 KL 阈值为 $5\times10^{-4}$，核对代码的 KL 自动阈值为 $10^{-3}$，二者不可混写。[R2, R3, M3–M5]

可选的结果筛选只需教师按同一动态策略完整作答一次并缓存是否正确，不搜索多个固定深度。原生退出确定监督终点，结果验证筛选题目，两者作用不同。[R19]

### 3.2 第二步：读取前 n_t 轮的 token 分布

**学生和教师读取同一个学生生成前缀** $c_t$。在这个前缀上，保留教师从第一轮到原生退出轮 $n_t$ 的读出；每轮使用教师自身状态和缓存，不能以学生的浅轮缓存代替。[R1–R3]

每轮末状态通过模型原有输出端读成 next-token 分布：

- **Ouro：**每轮末状态经过原有 final norm、LM head 和 softmax；若提取的状态已经归一化，直接接 LM head。
- **Huginn：**每轮末状态经过完整 coda 预测路径和 softmax，包含相应归一化、coda 层与 LM head。[R3, M3, M4]

得到同一前缀上的 $q_1,\ldots,q_{n_t}$。这里读取概率分布，不为每个中间轮另外生成文本。教师独立作答时的退出轮次不直接用于当前学生前缀；蒸馏监督终点在当前前缀上决定。

### 3.3 第三步：用 JS 找转折，分段后选择最后一轮

在每个学生前缀上，计算相邻教师轮的分布变化：

\[
J_d=\mathrm{JS}(q_{d-1},q_d),\qquad d=2,\ldots,n_t.
\]

JS 小，表示这两轮的预测接近；JS 大，表示出现明显预测转折。Huginn 用相邻 KL 判断收敛，F-PABEE 借助预测分布相似性判断稳定；这里选择标准 JS 作为分段指标，并将 KL 作为对照。[R3, R35]

**具体选点规则：**

1. 将 $J_d>\tau$ 的连接作为候选分段边界；$\tau$ 在训练开发数据上固定。
2. 二轮学生最多保留一个边界，选择 JS 最大的候选；一般 $S$ 轮学生最多保留 $S-1$ 个最大边界。选出的边界按教师轮次排序。
3. 在第 $d-1$ 与第 $d$ 轮之间划分阶段，每段选择最后一轮作为监督点；最后一段的末轮始终是 $n_t$。
4. 没有明显转折时，只保留第 $n_t$ 轮作为终点监督。若 $n_t\leq S$，第一版也只保留终点监督，不增加中间阶段压缩项。

例如，当前 $n_t=6$，最大的变化发生在 **3→4**，就划为 **1～3、4～6** 两段，选 **3、6**。这里的大 JS 用来定位阶段边界；选段末则保留该段执行结束时的真实教师预测。

因此，应写成“**在当前 token 的前 n_t 轮内，选择不超过 S 个阶段的末轮**”。$n_t$ 是教师监督深度，不是监督点数量。不同 token 的中间监督点和终点都可以不同，学生的循环预算 $S$ 始终固定。

### 3.4 第四步：用选中的段末分布训练学生

对于二轮学生：

| 分段结果 | 学生监督方式 |
|---|---|
| 两段，教师末轮为 $d_1$ 和 $d_2=n_t$ | 学生第一轮学习 $q_{d_1}$，第二轮学习 $q_{n_t}$ |
| 无明显转折，或 $n_t\leq S$ | 仅用 $q_{n_t}$ 监督学生第二轮 |

两段时，沿用 LoopOPD 的 reverse KL：

\[
\mathcal L
=\frac12\left[
D_{\mathrm{KL}}\!\left(p_S^{(1)}(\cdot\mid c_t)\Vert q_{d_1}(\cdot\mid c_t)\right)
+D_{\mathrm{KL}}\!\left(p_S^{(2)}(\cdot\mid c_t)\Vert q_{n_t}(\cdot\mid c_t)\right)
\right].
\]

两侧条件都是同一个学生前缀 $c_t$。教师、分段结果和选中轮次停止梯度，只更新学生参数。第一版不增加教师轮融合、隐层辅助损失或额外选轮网络。

学生仍独立执行自己的第 1、2 轮；教师的监督分布来自原生退出轨迹。学生第一轮承接前一段教师计算的阶段结果，第二轮承接后一段结果。部署时只运行学生固定的 $S$ 轮。

### 3.5 完整例子：某个 token 的六轮教师轨迹压缩为二轮学生

为说明分段方式，假设教师上限允许六轮以上，当前 token 的原生退出轮次恰好为 $n_t=6$，学生固定 $S=2$。六轮只是示意，不代表直接把 Ouro 主设置改为六轮。

训练时，学生固定两轮生成前缀；教师在同一前缀上选择第六轮退出，读取前六轮分布。若 JS 表现为：

    1→2：小
    2→3：小
    3→4：大
    4→5：小
    5→6：小

则：

    教师完整计算： [1 → 2 → 3]   [4 → 5 → 6]
    选中监督轮：          3                6
                          ↓                ↓
    学生固定计算：      第一轮 ────────→ 第二轮

下一个 token 可能在第三轮被选择退出，对应 $n_{t+1}=3$；学生仍然固定两轮。原生退出确定各 token 的监督终点，轮间 JS 选择中间监督点。JS 转折是否对应值得迁移的关键探索，由蒸馏效果与对照实验检验。

## 4. 动态退出论文具体提供哪些判据

主路线沿用 Ouro/Huginn 已有的动态退出规则确定监督终点，再使用 JS 分段选择中间监督点。停止规则与分段规则的作用不同：前者决定当前监督到哪轮，后者决定保留哪些阶段结果。PABEE/F-PABEE 的连续稳定判据等可作为对照。

| 论文 | 原方法检查什么 | 我们如何使用 | 适用范围 |
|---|---|---|---|
| Huginn [R3] | 相邻循环预测 KL，低于阈值则退出 | 判断预测平稳，筛出阶段末端 | 原生 Huginn；原阈值不能直接移植到 JS 或其他模型 |
| Adaptive Depth [R27] | Ouro 上的 hidden cosine/L2、entropy、margin、max probability、ponder gate | 比较低成本状态与预测规则 | 原生 Ouro；没有一个规则在所有任务上最好 |
| Two-Scale Latent Dynamics [R33] | 隐藏更新的一阶与二阶变化，连续命中再退出 | 低成本隐层判据对照，不作为主选点指标 | GPT-2 风格分组循环模型，并非原生 Ouro/Huginn |
| CALM [R36] | 隐藏态余弦饱和、top-2 margin、小型退出分类器，并做序列层面校准 | 低成本指标与阈值校准 | 自回归生成；其保证有条件，不能等同答案总是正确 |
| LoopRPT [R15] | 参考 token likelihood 相对参考深度的增益 | 在参考前缀上检验深轮是否提供预测进展 | Ouro；不是任意学生前缀上的正确性标签 |
| PABEE [R34] | 连续若干层预测标签不变 | patience，减少偶然平稳的误判 | BERT/ALBERT 分类 |
| F-PABEE [R35] | 连续分布距离低于阈值，而不只看 argmax | 用分布变化定义平稳段 | 单/多标签分类；JSKD 与标准 JS 不完全相同 |
| DeeBERT [R38] | 中间分类分布熵低于阈值 | 单点置信度基线 | BERT 分类，不足以识别完整转折 |
| FastBERT [R39] | 自蒸馏分支的归一化熵/不确定性 | 检验便宜置信度筛选是否已经够用 | BERT 分类，先训练早期输出分支 |
| CATs [R37] | 预测一致性分类器与校准 | 校准“何时接受较早输出”的思想 | 保证与原模型一致的范围，不等于真值正确 |
| Prediction Dynamics [R28] | 区分纠错、改错、错换错 | 防止把分布变化量直接当收益 | Huginn/Ouro 的受控候选答案轨迹 |
| Think-at-Hard [R26] | 哪些 token 值得继续迭代、是否发生 latent overthinking | 说明明显后轮更新也可能有害 | 循环改造模型，非原生 Ouro checkpoint |

这些论文支持“变化/稳定怎样测量”。**按转折分阶段、选择各段末轮，并进行固定浅轮蒸馏，是我们要检验的组合。**

## 5. 三个容易混淆的判断

### 5.1 教师内部相似，与师生相似不是一回事

**教师第 2、3 轮的读出分布相似**，说明可以考虑只保留一个代表预测阶段，减少重复监督；不意味着内部计算完全相同，也不意味着这两轮都需要单独蒸馏。

**学生第一轮与教师某轮相似**，才与师生对齐难度有关；这里也可用预测分布距离比较。但相似不自动代表能获得新能力，差异大也不自动代表学不会。第一版先按教师轨迹分阶段，不叠加新的可学习性评分器。

“学生学得动”由中间阶段承接目标和固定浅轮训练结果检验，不能由阶段内分布相似直接推出。师生距离可以作为后续便宜的兼容性门控对照，不做多路径试训。

### 5.2 为什么主方案固定选段末

段末选择提供统一的阶段结束目标，不增加融合参数，也无需搜索段内代表轮。这个设计的依据是压缩轨迹的阶段结构，不是“深一轮总会更好”。是否优于段首、均匀映射或阶段内平均，由实验回答。

RAIL-KD 和选层策略研究表明，随机或固定映射也可能很好，必须保留简单映射对照。[R29, R32] ALP-KD/MoLSAKI 的软对齐可作为融合方法参考，但不是当前主方案。[R13, R30]

### 5.3 选监督点，不等于删除教师计算

训练读出时，教师仍完整执行第 1 至 $n_t$ 轮内的所有循环，才能找到其中的转折。学生独立执行固定浅轮，构建自己的 KV；第一版不做教师阶段跳轮或师生 KV 移植。教师自身的动态退出缓存沿用所核对实现的语义。

选择第 3、6 轮提供监督时，教师仍执行全部中间循环，得到真实的第 3、6 轮状态。学生独立学习这些阶段目标，不直接运行教师跳轮路径，也不继承教师缓存。[R2, R3, M3]

## 6. 最小实验与成本

### 6.1 先做哪些实验

| 实验 | 方法 | 回答什么 |
|---|---|---|
| A | 原始 LoopOPD：固定教师深度，只监督终点 | 基础表现 |
| B | 原生动态退出教师，只监督退出终点 | 沿用动态教师终点的作用 |
| C | 与 B 使用同一退出策略，在前 $n_t$ 轮内均匀分段、选择段末 | 单纯增加中间监督的作用 |
| D | 与 B 使用同一退出策略，在前 $n_t$ 轮内用 JS 分段、选择段末 | 是否需要识别探索阶段，即完整方案 |

主对照 A/B/C/D 使用相同样本集合、学生预算和教师最大上限。若采用结果筛选，各方法使用同一份筛选结果，并另做有无筛选对照。比较 C/D 时保持监督点数量可比，单独统计 D 退化为仅终点监督的位置。由于学生训练后生成的前缀可能不同，不要求跨实验强行共享同一个 $n_t$ 标签。所有学生在完整独立测试集上报告结果。

先报告固定浅轮精度、教师退出轮次分布、可选正确性筛选成本、训练成本、生成长度和部署时间。需要进一步解释结果时，再增加段首/段末、JS/KL 和最小正确深度搜索等对照；不将它们混入主体训练步骤。[R29, R32]

Cross-Loop Query Handoff 式冻结/恢复实验可用于小规模机制验证，检查所选阶段是否涉及检索修正；它不参与主训练流程。[R4]

### 6.2 开销如何控制

**不做逐深度完整作答搜索。** 教师在当前学生前缀上使用原生规则选择 $n_t$。若保留结果筛选，每题只按同一动态策略完整作答一次并缓存是否正确，单独统计这部分生成成本。

**动态选终点，不自动等于真正提前停算。** 核对的 Ouro 发布实现先计算配置中的全部循环，再按退出阈值选择 token 读出；这种实现可以用于动态监督选点，但不能声称已经节省了后续循环。Huginn 提供逐轮检查的自适应生成接口，其普通批处理缓存也可能影响退出评测。首先忠实保留所用实现的缓存语义，报告实际执行轮数与耗时；额外的提前停算或批处理优化单独验证。[M3, M4, R40]

**读出成本。** JS 分段需要前 $n_t$ 轮的 token 分布，Ouro 可按 token 分块计算；Huginn 需要 coda 读出。退出检查本身若已计算分布，应复用该结果。学生只训练一次，部署预算仍固定为 $S$。[R3, M4]

参考解答损失筛选、最小观测正确深度搜索等作为扩展实验，其标注与验证开销单独报告。调研依据见《降低每题教师监督深度选择成本：论文与可借鉴方案》。退出阈值与分段阈值仅在训练开发数据上校准；原生退出不等于正确性或最优深度保证。

## 7. 基模与实现约定

- **Ouro 主设置**：教师上限先用四轮，启用累计退出概率阈值，学生固定二轮。当前 $n_t=4$ 时可选择 2/4、3/4 等监督目标；若教师选择更早退出，则以该位置的 $n_t$ 为终点。
- **Huginn 验证**：教师上限先以 32 为参考，启用原生 KL 退出规则；学生固定八轮或十六轮，分别实验。更换上限与退出阈值作为对照。
- **训练深度**：Ouro 早期八轮、后续主要四轮；Huginn 随机深度训练，标称约 32、最后八轮反传。六轮示意不代表六轮一定是更强教师。[R2, R3]
- **读出一致性**：同一完整轨迹的中间读出，未必等于独立固定该深度生成的策略；论文要说明对象。必要时核对缓存；Huginn 预测读出需经过 coda。[R3, M3]
- **当前主分段指标**：计算前 $n_t$ 轮的相邻 JS，最多选择 $S-1$ 个超过阈值的最大转折作为边界，取各段末轮。若选出少于 $S$ 个阶段，将这些目标按顺序均匀映射到学生检查点，最后目标始终匹配学生第 $S$ 轮。
- **可选离线筛选**：先冻结教师，固定动态退出、解码配置、长度上限与验证器版本，按动态策略完整作答一次并缓存正确性。只缓存样本筛选结果，不复用其退出轮次作为 OPD 标签。
- **边界位置**：$n_t\leq S$ 时仅保留终点监督，不增加中间阶段压缩项，并报告比例。若使用结果筛选，报告教师作答失败的样本比例；这不表示所有深度都无法答对。
- **OPD 范围**：离线正确性标签只负责样本筛选；训练时师生读取相同学生前缀。第一版不增加专门的前缀纠错模块。
- **暂不增加 U-Space、额外教师和新 gate**：沿用基模已有退出规则，先检验 JS 段末监督，再逐项扩展。这里的“动态退出教师”与 LoopOPD 的教师参数刷新是两个不同维度。

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
