# 谈谈DeepSeek mHC

## TL;DR


DeepSeek发布了一篇新的论文《mHC: Manifold-Constrained Hyper-Connections》[^1]. 这篇论文有一个非常好的Taste, 一方面是算法层面上, 流形(Manifold)和基于Optimal Transport(OT)理论的Sinkhorn-Knopp算法. 另一方面是工程实现上的处理也非常优雅.

在2024年字节发布Hyper-Connections前几个月, 我也写过一篇文章[《谈谈DeepMind会做算法导论的TransNAR并引出基于SAE-GNN的可组合Transformer猜想》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490297&idx=1&sn=7d758e84bdce7ae4f20f031f4ac3f221&scene=21#wechat_redirect)谈到过一个架构, 基于一些可解释性的视角, 即类似于SAE和TransNAR的方式来扩展残差流宽度和连接多样性.

然后进一步基于范畴论的视角上, 如果将Attention作为一个态射, 那么也需要一些高阶范畴做一些约束, 其实这些才是Hyper-Connection的实质, 而不是现在一些算法工程师简单理解的Residual的处理. 下面我们在展开分析mHC之前, 先补充一些数学相关的内容. 本文目录如下:

```
0. 一些数学的背景
0.1 Manifold视角
0.2 Hyper-Connection Overview
0.3 基于范畴论的视角看Hyper-Connection
0.4 Optimal Transport
1. mHC Overview
1.1 HC如何破坏恒等映射
1.2 Mainfold Constrained
2. 相关工作
2.1 微观设计
2.2 宏观设计
3. 预备知识
3.1 数值稳定性分析
3.2 系统开销
4. mHC算法和实现
4.1 流形约束超连接
4.2 参数化与流形投影
4.3 mHC工程实现
4.3.1 算子融合
4.3.2 Recomputing
4.3.3 流水线排布
5. 实验
5.1 Setup
5.2 主要结果
5.3 Scaling
5.4 稳定性分析
6. 结论和展望

```


# 0. 一些数学的背景


mHC本身的实现是非常简单而且Straightforward的, 这里补充一些内容来展开阐述一下流形和HC, 因为可能在mHC的基础上还可以发展成更多的算法, 流形约束不光只是在HC的训练稳定性上, 其实还有更多的扩展, 或者更多的流形可以考虑, 例如基于流形约束的视角控制Attention? DeepSeek在mHC的论文结尾也谈到:

但该框架同样支持探索为特定学习目标量身定制的、多样化的流形约束.

因此在前置的这一章, 我想多展开一下这些数学上观念, 虽然它们中的一些内容和mHC本身没有太直接的关联.

## 0.1 Manifold视角


先来用一个比较浅显的一段解释一下Manifold.  这个观点来自于DeepMind的Demis Hassabis, 他眼中的大千世界：世界存在某种结构压缩的低维流形（low-dimensional manifold).

Hassabis 在解释 AlphaFold 成功的关键时指出, 蛋白质的理论构象空间是  $10^{300}$  之巨, 完全无法穷举或物理模拟. 但自然界中蛋白能在毫秒级自动完成折叠, 说明自然并没有在“全空间乱跑”, 而是压缩演化在一条低维流形上. 这是 AlphaFold 能够成功预测结构的根本前提. 他强调：自然现象之所以“可预测”, 不是因为我们写出了完美的方程, 而是因为自然的行为模式在高维空间中稀疏分布、结构清晰、路径稳定, 它们集中在一种可压缩、可调度的结构空间中, 这就是所谓的“流形（manifold）”.

AlphaFold 并不是像传统方法那样模拟蛋白质在全空间内的动力学演化过程, 而是通过从大量蛋白质序列与已知结构的样本中学习, 采集到了这个结构流形. 它不是试图穷举所有可能构象, 而是在训练中逐步内化出一个可以导航的、从序列到结构的映射空间, 也就是一个“潜在折叠流形”. 最终, 在预测新蛋白结构时, 它并不是搜索整个空间, 而是在这个内化的流形中直接定位最可能的位置, 完成高效预测.

另外一个例子是在具身智能上, 个机器人的所有可能状态(如关节角度, 位置, 姿态)可以构成一个流形. 例如, 一个三维空间中刚体的姿态(旋转)可以由一个三维流形SO(3)来描述.

![](images/img_002.jpg)


拉长整个时间线来看, 历史的长河里, 世界万物如同一个流形(Manifold)静静地流淌着, 又因为文字形成的记忆和智能把时空折叠成一个高维的空间... 治史之旨, 在钩沉索隐, 以解读过去; 在洞幽烛微, 以阐明现在; 更在观测瞻望, 以指引未来. 此即史学之求索, 亦是人类前行之罗盘. 似乎模型的训练也是如此. 例如LLM这些基于自然语言训练的模型.

也就是说, 世界本是高熵混沌的, 但其中蕴含可提取的结构秩序: 只要识别出这些结构, 就能实现压缩、调度与迁移, 从而在无序中建立智能与控制. 以前我们把这些结构秩序用自然语言的方式称为经验或者总结成一些定理.

而现在用深度神经网络从数据中提取出了低维流形, 并在这个结构压缩空间中完成了调度和推理. 它不是理解了所有物理机制, 而是掌握了“自然允许你走的那些路径”.  智能的任务不是重建现实, 而是压缩现实, 压缩成一个可以学习、调用、迁移的结构流形.

![](images/img_003.jpg)


结论: 利用流形来对深度神经网络做一些可学习的约束就成了一个非常straightforward的路径, 那么需要探讨的问题是这些约束施加在神经网络的什么地方? mHC就是一个很好的例子.

## 0.2 Hyper-Connection Overview


我们来简单阐述一下《Hyper-Connection》[^2], 它是Seed在2024年9月的一个工作, Defa在知乎上有一个解释《都2025年了, 我不允许你还在用残差连接！》[^3]

Transformer中残差连接主要就两种变体Pre-Norm 和 Post-Norm各自都有其局限性, 这里苏剑林的博客有过分析.

- `Pre-Norm`：在每个残差块之前做Norm, 能够有效地减少梯度消失问题. Pre-Norm的问题在于后面的层的输出太像, 以至于削弱了模型的学习能力.
- `Post-Norm`：在残差块之后做norm, 有助于减少表示崩溃问题, 但也会重新引入梯度消失问题.


Hyper-Connections的灵魂在于通过动态调整不同层之间的连接权重, 弥补了残差连接在梯度消失（Gradients Vanishing）和表示崩溃（representation collapse）之间的跷跷板现象. 最后发现, 不仅训练比Pre-Norm稳定, 层间相似度更低, 相似度范围更广, 效果更好.

问题的根源在于, 传统的残差连接以一种固定的, 不可学习的方式来组合层输入和输出, 能否让网络自主学习如何组合不同层级的特征, 以打破这种固定的权衡?为了实现这一点, HC引入了两个关键概念, 一个是宽度扩展, 另一个是可学习的连接. Hyper-Connection在Transformer上完整的网络结构图如下

![](images/img_004.png)


后续的工作是Seed的《Virtual Width Networks》[^4]

![](images/img_005.png)


Hyper-Connections会对这些hidden向量建立以下两类连接：

- 深度连接（Depth-Connections）：类似于残差连接, 但通过为输入与输出之间的连接分配可学习的权重, 允许网络灵活调整不同层之间的连接强度.
- 宽度连接（Width-Connections）：在每一层中实现Hidden向量之间的信息交互, 增强特征融合能力, 从而提升模型的表示效果.


我们以Dynamic HC(DHC)为例, 来简单解释一下算法

将第 k 层的输入视为隐向量  $h_{k-1} \in \mathbb{R}^d$  (或  $h_{k-1} \in \mathbb{R}^{d \times 1}$ ), 网络的初始输入为  $h_0$ . 最初,  $h_0 \in \mathbb{R}^d$  被复制 n 次, 形成初始的超隐状态矩阵 $H_0 = (h_0 \ h_0 \ \dots \ h_0)^\intercal \in \mathbb{R}^{n \times d}$

对于第 k 层, 其输入是来自前一层的超隐状态矩阵

$$
H_{k-1} = (h_{k-1}^1 \ h_{k-1}^2 \ \dots \ h_{k-1}^n)^\intercal \in \mathbb{R}^{n \times d}
$$


Hyper-Connection 可以用一个矩阵 HC 来表示, 其中每个元素定义了一个连接权重. 该矩阵的结构如下: $HC(H) = \begin{pmatrix} 0_{1 \times 1} & B(H) \\ A_m(H) & A_r(H) \end{pmatrix}$

![](images/img_006.png)


深度连接关注的是垂直方向的信息流动, 即输入与输出之间的关系,如上图 (c) 所示. 深度连接可以被解耦为如下矩阵 : $DC(H) = \begin{pmatrix} B(H) \\ \text{diag}(A_r(H)) \end{pmatrix}$

宽度连接关注的是水平方向的信息流动, 即通道与通道之间的关系, 如上图 (d) 所示, 矩阵可以定义如下: $WC(H) = (A_m(H) \ A_r(H))$

**渣注**


简单来说, 动态深度连接为每个 token 动态调整新旧知识在每个通道中的混合比例. 动态宽度连接为每个 token 动态调整通道间的信息交换和汇聚方式, 从而实现计算图上Hidden Vector的动态重排.

为了更便于理解, 大致的算法用一个"智能音频混合器"来类比Dynamic Hyper-Connection:

![](images/img_007.jpg)


- **输入 H**: 想象有 `n` 条音轨 (例如, 鼓, 贝斯, 吉他, 人声), 每条音轨都是一个信息通道.
- **步骤 1 (动态权重生成)**: 混音师 (网络) 听了一下当前的音乐片段 (输入 token), 决定了所有旋钮的调校方案 ( $A_m(H), A_r(H), B(H)$ ).
- **步骤 2 (宽度连接)**:

  - `h0` 的计算: 混音师根据  $A_m(H)$  方案, 从 `n` 条音轨中混合出一个新的声音, 送入一个效果器 (如混响, 对应 `T` 层).
  - `H'` 的计算: 同时, 混音师根据  $A_r(H)$  方案, 对原始的 `n` 条音轨进行内部交叉混合, 准备好作为背景声.

- **步骤 3 (核心计算)**: 效果器处理混合声, 产生新的效果声  $h'_0$ .
- **步骤 4 (深度连接)**: 混音师根据  $B(H)$  方案, 将效果声  $h'_0$  混合回 `n` 条背景声  $H'$  中, 形成最终输出的 `n` 条新音轨  $\hat{H}$ , 传递给下一个混音站.


**渣注**


结合前面一节, 我们可以将 DHC 理解为一种**自适应的、数据驱动的动态流形约束**.

传统的残差连接  $h_{out} = h_{in} + F(h_{in})$  可以被看作一种最简单的流形约束. 它假设  $F(h_{in})$  是一个在  $h_{in}$  点的微小扰动或更新. 因此, 输出  $h_{out}$  仍然非常靠近输入  $h_{in}$ . 如果  $h_{in}$  在流形上, 那么  $h_{out}$  也大概率在流形附近. 这就像在地球表面上走一小步, 你仍然在地球表面上. 实质上这里也阐述了残差连接的恒等映射. 这个约束是静态且统一的. 无论数据点  $h_{in}$  位于流形的哪个位置 (是平坦区域还是高度弯曲的区域), 它都采用完全相同的约束方式 (简单相加). 它对流形本身的局部几何结构一无所知 (manifold-agnostic).

DHC 通过其多通道设计和动态权重, 实现了一种**流形感知 (manifold-aware)** 的动态约束. DHC的 n 个通道可以被理解为在学习一个动态的、局部的坐标系, 这个坐标系近似了表示流形在当前点 H 处的切空间.超隐状态矩阵  $H = (h_1, h_2, \dots, h_n)^\intercal$  中的 n 个向量, 不再是孤立的点, 而是可以被看作是张成 (span) 一个 n 维子空间的基向量.

一旦有了这个局部坐标系 (切空间), DHC 的所有动态操作都可以被理解为在这个坐标系内进行的、有意义的几何变换.

-  $A_m(H)$ 构成了一个在流形上的方向选择器,  $h_0^\intercal = A_m(H)^\intercal H$  这个操作, 是在由 n 个基向量张成的切空间中, 进行动态的线性组合, 形成一个特定的方向向量
-  $A_r(H)$ 构成了在流形上的局部线性变换器 (旋转/缩放),  $H' = A_r(H)^\intercal H$  这个操作, 是对 n 个基向量本身进行一次动态的线性变换
-  $B(H)$ 构成了一个更新投影器,  $\hat{H} = B(H)^\intercal (T(h_0))^\intercal + H'$  这个操作, 是将 T 层计算出的更新量  $T(h_0)$  (它本身也是一个向量), 通过  $B(H)$  投影回 n 个新的基向量方向上.


由于 DHC 在每一步 (每一层) 都会重新评估当前位置的局部几何 (通过动态生成  $A_m, A_r, B$ ), 它的逐层演进过程更有可能**沿着流形的内在曲率前进**, 从而更紧密地逼近表示空间中的**测地线流**. 这意味着从输入到输出, 信息的演化路径更"自然"、更"高效", 避免了在弯路上走直线的浪费(传统的ResNet像是在高维环境空间中走直线小步), 从而获得更好的性能.

关于HC的训练稳定性的问题, 以及mHC的算法我们将在稍后的章节展开.

## 0.3 基于范畴论的视角看Hyper-Connection


另一个视角是来自范畴论, 在[《大模型时代的数学基础(2)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488528&idx=1&sn=fa49e334201e738e7ddb4258030798b3&scene=21#wechat_redirect)中有一些范畴论相关的简单介绍, 米田引理 (Yoneda Lemma) 是范畴论的基石..大白话来说如果能够理解：“人的本质是一切社会关系的总和”, 它的核心思想是:**一个对象完全由它与范畴中所有其他对象的关系 (即"箭头"或"态射") 所决定**. 换句话说, 你不需要知道一个对象的"内部构造", 只需要知道它如何与外界"交互"(即所有射向其他对象的箭头集合), 就能完全理解它.

对应来看, 以范畴论的视角来看, 我们对于一个Token可以看作由它的Attention所决定的. 于是token和attention某种意义上可以构成一个范畴. 当前的大模型训练实际上利用Transformer在通过Attention构造的态射来构建世界的一个可表示的预层(representable presheaf)

![](images/img_008.jpg)


那么在这个视角下, 结合Manifold的观点, 事实上我们就需要对模型本身的Attention实施一些约束, 那么很自然的就需要Attention旁路上添加一些运算, 那么很自然的一个想法就是构造Hyper-Connection, 进一步这些信息的传递在模型的层间, 进一步引出了范畴论中的Nerve[^5](神经)构造. 具体内容可以参考[ 《谈谈 Hierarchical Sparse Attention (HSA)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496275&idx=1&sn=5f8a8d8efff22033d3f2aed8a5844e53&scene=21#wechat_redirect)的第三章.

DeepSeek进一步把流形约束的想法和Hyper-Connection结合, 提升了Hyper-Connection训练稳定性的工作.

## 0.4 Optimal Transport


另一个需要补充的背景是DeepSeek所使用的`Sinkhorn-Knopp`算法, 它来自于Optimal Transport理论, 从Optimal Transport的视角看Attention也是一个非常有趣的话题, 去年8月有一篇文章[《大模型时代的数学基础(9)- SDPA和最优传输, 强化学习及信息几何的联系》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494688&idx=1&sn=3d589f6d4be56ee372d5db4f8631b0cc&scene=21#wechat_redirect), 其中介绍了Sinkhorn算法和Sinkhorn Attention.

Sinkhorn Attention是一个双向匹配:算法目的是在Query和Key之间找到一个“最优匹配”或“最优运输方案”, 使得每个Query都分配完自己的预算, 同时每个Key接收到的总预算也是固定的(例如, 平均分配) 这是一个双随机(doubly stochastic)矩阵. 排序矩阵用 Sinkhorn-Knopp 迭代强制变为双随机矩阵(行、列和均为1), 保证可导且可并行.

![](images/img_009.png)


而mHC的核心思想是将残差映射矩阵  $\mathcal{H}_{l}^{\text{res}}$  约束为**双随机矩阵 (doubly stochastic matrix)**. 解决HC不稳定的问题.

![](images/img_010.jpg)


**为何选择双随机矩阵?**


- 范数保持: 双随机矩阵的谱范数不大于1, 即  $\|\mathcal{H}_{l}^{\text{res}}\|_2 \leq 1$ . 这意味着该操作是能有效缓解梯度爆炸.
- 组合闭包性: 双随机矩阵的乘积仍然是双随机矩阵. 这保证了跨多层的复合映射  $\prod \mathcal{H}^{\text{res}}$  仍然保持稳定性, 从而在整个网络深度上维持了信号的稳定传播.
- 几何解释: 所有  $n \times n$  双随机矩阵的集合构成了**Birkhoff多面体**, 它是所有  $n \times n$  置换矩阵的凸包. 这意味着  $\mathcal{H}^{\text{res}}$  的作用可以被解释为对不同残差流进行置换的凸组合, 是一种鲁棒的特征融合机制.


实质是: Sinkhorn-Knopp算法将一个"危险"的无约束矩阵  $\tilde{\mathcal{H}}_{l}^{\text{res}}$  变成了一个"安全"的双随机矩阵  $\mathcal{H}_{l}^{\text{res}}$

# 1. mHC Overview


## 1.1 HC如何破坏恒等映射


作者在Abstract中提到:

HC通过扩展残差流 (residual stream) 的宽度和多样化其连接模式, 拓展了过去十年间已经无处不在的残差连接范式. 尽管这种多样化带来了显著的性能提升, 但它从根本上损害了残差连接固有的**恒等映射 (identity mapping)** 属性, 从而导致了严重的训练不稳定性和受限的可扩展性, 此外还引入了显著的内存访问开销.

首先, 我们需要理解标准残差连接是如何实现并受益于恒等映射的.

在一个标准的ResNet块中, 输出  $x_{l+1}$  是输入  $x_l$  和一个非线性变换  $\mathcal{F}(x_l)$  的和:

 $x_{l+1} = x_l + \mathcal{F}(x_l, W_l)$

当我们把这个公式从浅层  $l$  递归地展开到深层  $L$  时, 会得到:

 $x_L = x_{L-1} + \mathcal{F}(x_{L-1}, W_{L-1})$  $= (x_{L-2} + \mathcal{F}(x_{L-2}, W_{L-2})) + \mathcal{F}(x_{L-1}, W_{L-1})$  $= \dots$  $= x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, W_i)$

现在我们来看HC的公式. HC首先将残差流扩展为  $n$  个并行的流, 我们用一个  $n \times C$  的矩阵  $x_l$  来表示. 其单层更新规则为: $x_{l+1} = \mathcal{H}_{l}^{\text{res}} x_l + (\text{残差部分})$

这里的  $\mathcal{H}_{l}^{\text{res}}$  是一个  $n \times n$  的可学习矩阵. 让我们像之前一样, 把这个公式从浅层  $l$  递归展开到深层  $L$ . 为了简化, 我们暂时忽略残差部分, 只看主干的传播:

$$
\begin{aligned}
x_L &= \mathcal{H}_{L-1}^{\text{res}} x_{L-1} + (\dots) \\
&= \mathcal{H}_{L-1}^{\text{res}} (\mathcal{H}_{L-2}^{\text{res}} x_{L-2}) + (\dots) \\
&= \dots \\
&= (\mathcal{H}_{L-1}^{\text{res}} \mathcal{H}_{L-2}^{\text{res}} \dots \mathcal{H}_{l}^{\text{res}}) x_l + (\text{残差部分})
\end{aligned}
$$


我们将这个复合映射记为  $\mathcal{H}_{\text{composite}} = \prod_{i=l}^{L-1} \mathcal{H}_{i}^{\text{res}}$ . 那么: $x_L = \mathcal{H}_{\text{composite}} \ x_l + (\text{残差部分})$

如下图所示, ResNet结构来自任意浅层  $l$  的信息, 可以通过一个不含任何变换的、纯粹的加法路径, 直接传递到任意深层  $L$ 这条纯粹的加法路径就是恒等映射 (Identity Mapping).

![](images/img_011.jpg)


而在HC的设计中,  $\mathcal{H}_{i}^{\text{res}}$  是一个**无约束的 (unconstrained)** 可学习矩阵. 这意味着它的元素可以是任意值, 它的性质 (如范数, 行列和) 是完全不可控的.矩阵连乘会产生指数级的放大或缩小效应.如果每个  $\mathcal{H}_{i}^{\text{res}}$  的谱范数略大于1, 比如1.1, 那么经过  $k$  层传播后, 复合矩阵的范数可能会增长到  $1.1^k$ . 当网络很深时,  $k$  很大, 这会导致信号的**指数级爆炸**. 反之, 如果范数略小于1, 比如0.9, 就会导致信号的**指数级消失**.

## 1.2 Mainfold Constrained


基于这个问题, DeepSeek提出了**流形约束超连接 (Manifold-Constrained Hyper-Connections, mHC)**. 它将HC的残差连接空间投影到一个特定的流形上, 以恢复恒等映射属性, 同时结合了严格的基础设施优化来保证效率.

具体来说, mHC利用**Sinkhorn-Knopp算法** (Sinkhorn and Knopp, 1967) 将  $\mathcal{H}_{l}^{\text{res}}$  进行熵投影到Birkhoff多面体 (Birkhoff polytope)上. 这个操作有效地将残差连接矩阵约束在由**双随机矩阵 (doubly stochastic matrices)** 构成的流形内. 由于这些矩阵的行和与列和都等于1,  $\mathcal{H}_{l}^{\text{res}} x_l$  这个操作起到了对输入特征进行凸组合 (convex combination)的作用. 这一特性促进了一种稳态的信号传播, 其中特征均值得以守恒, 并且信号范数被严格正则化, 有效地减轻了信号消失或爆炸的风险. 此外, 由于双随机矩阵对于矩阵乘法是闭合的 (closure), 复合映射  $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$  保持了这种守恒属性. 因此, mHC有效地在任意深度之间维持了恒等映射的稳定性.

另一方面, 为了确保效率, 采用了**算子融合 (kernel fusion)** 并利用Tilelang开发了混合精度算子. 此外通过选择性重计算 (selective recomputing)减轻了内存占用, 并在DualPipe调度细致地Overlap了通信.当扩展率  $n=4$  时, mHC支持规模化训练, 并且仅引入了6.7%的额外时间开销.

**渣注**


总体来看, 整个Abstract和Introduction指出了HC存在的三个重要的问题:

- HC破坏了 "恒等映射". 恒等映射保证了信号和梯度可以无损地在网络中深层传播, 是解决深度网络梯度消失/爆炸问题的基石. 一旦破坏, 信号在逐层传播中可能被无界地放大或衰减. 作者用 "fundamentally compromises" (从根本上损害) 这个词, 足见其问题的严重性.
- 实验中的一些后果:

  - **训练不稳定性**: 理论上的信号衰变/放大直接导致训练过程中的损失尖峰 (loss spike) 和梯度爆炸.
  - **可扩展性受限**: 这种不稳定性会随着模型深度和规模的增加而被放大, 使得HC难以应用于真正的大规模模型训练.

- 内存访问开销: 扩展的残差流意味着需要从内存中读写更多的数据. 在现代计算中, 内存带宽往往是瓶颈 (所谓的"内存墙"). HC虽然计算量增加不多, 但其巨大的内存访问成本会严重拖慢实际训练速度.


DeepSeek的做法是: 算法层面使用了来自数学中微分几何和拓扑学的概念**流形 (Manifold)**. 具体来说, 是将连接矩阵 "投影" 到一个具有特定良好性质的流形上. 这个投影操作的目的是 "恢复恒等映射属性", 直接对标之前发现的核心问题. 这是一种非常优雅且有力的解决方案.

在Infra层面, 通过一系列优化, 直接回应了HC带来的内存访问开销问题.

"流形约束" 这个提法非常高明. 它将一个工程问题(如何约束矩阵)提升到了一个更抽象和普适的数学框架下. 这不仅让解决方案更具理论美感, 也为未来的工作(例如, 探索其他流形)打开了想象空间.

# 2. 相关工作


这一章对于深度学习中的架构进展从微观设计 (micro-design) 和宏观设计 (macro-design) 两个维度进行了分析.

- **微观设计**: 关注"积木块"本身的设计 (如Attention, FFN, 卷积核).
- **宏观设计**: 关注如何"搭建积木"(层与层之间的连接方式).


## 2.1 微观设计


大致阐述了从CNN处理结构化信号到Transformers (Vaswani et al., 2017) 的出现确立了**注意力机制 (Attention)** 和**前馈网络 (Feed-Forward Networks, FFNs)** 作为现代架构的基础构建模块. 注意力机制促进了全局信息的传播, 而FFN则增强了单个特征的表示能力. 然后展开介绍了一下Attention Block的演进, 例如MQA/GQA/MLA等.. 也阐述了FFNs也通过MoE范式被推广为稀疏计算, 允许在不按比例增加计算成本的情况下实现大规模的参数扩展.

## 2.2 宏观设计


宏观设计主导了网络的全局拓扑结构, 例如ResNet/DenseNet/FractalNet等架构分别通过稠密连接和多路径结构来增加拓扑复杂性, 以期提升性能. 深度层聚合(DLA)通过在不同深度和分辨率上递归地聚合特征, 进一步扩展了这一范式.

然后展开介绍了近期的工作, 宏观设计的焦点已经转向**扩展残差流的宽度**. Hyper-Connections引入了可学习的矩阵来调节不同深度特征之间的连接强度, Residual Matrix Transformer, RMT则用一个外积内存矩阵替换了标准的残差流以促进特征存储. MUDDFormer采用多路动态稠密连接来优化跨层信息流.

紧接着话锋一转, 点出了这些结构的问题:它们损害了残差连接固有的**恒等映射属性**, 从而引入了不稳定性并阻碍了可扩展性. 另一部分的问题是, 由于特征宽度的扩展, 它们也带来了显著的内存访问开销.

最后介绍了mHC的区别, 在HC的基础上将残差连接空间约束在一个特定的流形上以恢复恒等映射属性, 同时还结合了基础设施优化来确保效率. 这种方法在保持扩展连接所带来的拓扑优势的同时, 增强了稳定性和可扩展性.

# 3. 预备知识


作者在mHC论文的第三章介绍了一些预备知识. 首先是这篇论文使用的符号体系:

在HC的公式中, 第  $l$  层的输入  $x_l \in \mathbb{R}^{1 \times C}$  被扩展  $n$  倍, 以构建一个隐矩阵  $x_l = (x_{l,0}^\top, \dots, x_{l,n-1}^\top)^\top \in \mathbb{R}^{n \times C}$ , 这可以被看作是  $n$  流的残差. 这个操作有效地加宽了残差流的宽度. 为了控制这个流的读出, 写入和更新过程, HC引入了三个可学习的线性映射:

$$
\mathcal{H}_{l}^{\text{pre}}, \mathcal{H}_{l}^{\text{post}} \in \mathbb{R}^{1 \times n}$ 和 $\mathcal{H}_{l}^{\text{res}} \in \mathbb{R}^{n \times n}
$$

.   这些映射修改了标准残差连接公式 $x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, W_i)$

得到了如下公式:

$$
x_{l+1} = \mathcal{H}_{l}^{\text{res}} x_l + (\mathcal{H}_{l}^{\text{post}})^\top \mathcal{F}(\mathcal{H}_{l}^{\text{pre}} x_l, W_l)
$$


在HC的公式中, 可学习的映射由两部分系数组成: 依赖于输入的部分和全局的部分, 分别被称为动态映射 (dynamic mappings) 和静态映射 (static mappings). 形式上, HC如下计算这些系数:

$$
\begin{cases} \tilde{x}_l = \text{RMSNorm}(x_l) \\ \mathcal{H}_{l}^{\text{pre}} = \alpha_{l}^{\text{pre}} \cdot \tanh(\theta_{l}^{\text{pre}} \tilde{x}_l^\top) + b_{l}^{\text{pre}} \\ \mathcal{H}_{l}^{\text{post}} = \alpha_{l}^{\text{post}} \cdot \tanh(\theta_{l}^{\text{post}} \tilde{x}_l^\top) + b_{l}^{\text{post}} \\ \mathcal{H}_{l}^{\text{res}} = \alpha_{l}^{\text{res}} \cdot \tanh(\theta_{l}^{\text{res}} \tilde{x}_l^\top) + b_{l}^{\text{res}} \end{cases}
$$


其中RMSNorm(·)应用于最后一个维度, 标量  $\alpha_{l}^{\text{pre}}, \alpha_{l}^{\text{post}}$  和  $\alpha_{l}^{\text{res}} \in \mathbb{R}$  是被初始化为很小值的可学习门控因子.

- 动态映射通过由  $\theta_{l}^{\text{pre}}, \theta_{l}^{\text{post}} \in \mathbb{R}^{1 \times C}$  和  $\theta_{l}^{\text{res}} \in \mathbb{R}^{n \times C}$  参数化的线性投影导出
- 静态映射则由可学习的偏置  $b_{l}^{\text{pre}}, b_{l}^{\text{post}} \in \mathbb{R}^{1 \times n}$  和  $b_{l}^{\text{res}} \in \mathbb{R}^{n \times n}$  表示


值得注意的是, 引入这些映射  $\mathcal{H}_{l}^{\text{pre}}, \mathcal{H}_{l}^{\text{post}}$  和  $\mathcal{H}_{l}^{\text{res}}$ ——所带来的计算开销可以忽略不计, 因为典型的扩展率  $n$  (例如4) 远小于输入维度  $C$ . 通过这种设计, HC有效地将残差流的信息容量与模型的计算复杂度 (FLOPs) (该复杂度与层的输入维度强相关) 解耦. 因此, HC通过调整残差流宽度, 为模型扩展提供了一条新的途径, 补充了预训练scaling laws 中讨论的传统扩展维度, 即模型FLOPs和训练数据大小.

尽管HC需要三个映射来管理残差流和层输入之间的维度不匹配, 但下表中展示HC组件的消融研究的实验表明, 残差映射  $\mathcal{H}_{l}^{\text{res}}$  产生了最显著的性能增益. 这一发现强调了在残差流内部进行有效信息交换的至关重要的作用.

![](images/img_012.png)


注: HC组件的消融研究: 当某个特定映射 ( $\mathcal{H}_{l}^{\text{pre}}$ ,  $\mathcal{H}_{l}^{\text{post}}$ , 或  $\mathcal{H}_{l}^{\text{res}}$ )被禁用时, 采用一个固定的映射来保持维度一致性: 对 $\mathcal{H}_{l}^{\text{pre}}$ 使用权重为 $1/n$ 的均匀映射, 对 $\mathcal{H}_{l}^{\text{post}}$ 使用权重为1的均匀映射, 对 $\mathcal{H}_{l}^{\text{res}}$ 使用单位矩阵

## 3.1 数值稳定性分析


尽管残差映射  $\mathcal{H}_{l}^{\text{res}}$  对性能至关重要, 但其顺序应用对数值稳定性构成了重大风险. 正如下面公式:

$$
x_L = \left(\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}\right) x_l + \sum_{i=l}^{L-1} \left[ \left(\prod_{j=1}^{L-1-i} \mathcal{H}_{L-j}^{\text{res}}\right) (\mathcal{H}_{i}^{\text{post}})^\top \mathcal{F}(\mathcal{H}_{i}^{\text{pre}} x_i, W_i) \right]
$$


当HC扩展到多层时, 从层  $l$ 到  $L$  的有效信号传播由复合映射  $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$  控制. 由于可学习映射  $\mathcal{H}_{l}^{\text{res}}$  是无约束的, 这个复合映射不可避免地会偏离恒等映射. 因此, 信号幅度在正向传播和反向传播中都容易发生爆炸或消失. 这种现象破坏了残差学习依赖于无阻碍信号流的基本前提, 从而在更深或更大规模的模型中破坏了训练过程的稳定性.

经验证据支持了这一分析. 作者在大规模实验中观察到不稳定的损失行为, 如图所示. 以mHC为基线, HC在约12k步时表现出意外的损失飙升, 这与梯度范数的不稳定性高度相关.

![](images/img_013.png)


此外, 对  $\mathcal{H}_{l}^{\text{res}}$  的分析验证了这种不稳定性的机理. 为了量化复合映射  $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$  如何沿残差流放大信号, 作者使用了两个指标.

- 基于复合映射的**行和的最大绝对值**, 捕捉了前向传播中的最坏情况扩展.
- 基于**列和的最大绝对值**, 对应于反向传播.


并将这些指标称为复合映射的**Amax增益幅度 (Amax Gain Magnitude)**. 如下图(b)所示, Amax增益幅度产生了极值, 峰值达到3000, 巨大差异证实了残差流爆炸的存在.

![](images/img_014.png)


**渣注:**


作者定义了一个非常巧妙的指标: Amax增益幅度:

- 对于一个矩阵  $A$ , Forward中 Amax增益幅度是  $\max_i \sum_j |A_{ij}|$ , 即**最大绝对行和**.
- 对于一个矩阵  $A$ , Backward中 Amax增益幅度是  $\max_j \sum_i |A_{ij}|$ , 即**最大绝对列和**.


#### 为什么Forward是行和?


考虑一个输入向量  $x$ , 输出  $y = Ax$ . 输出的第  $i$  个分量是  $y_i = \sum_j A_{ij}x_j$ . 根据不等式:  $|y_i| \le \sum_j |A_{ij}||x_j|$ . 如果我们假设输入向量  $x$  的所有元素绝对值最多为  $M_x$  (即  $\|x\|_\infty \le M_x$ ), 那么  $|y_i| \le (\sum_j |A_{ij}|) M_x$ .

因此, 行和  $\sum_j |A_{ij}|$  直接决定了输入信号在传播到输出的第  $i$  个分量时可能被放大的最大倍数. 最大绝对行和就代表了在所有输出分量中最坏情况下的放大因子.

#### 为什么Backward是列和?


考虑反向传播, 损失  $\mathcal{L}$  对输入  $x$  的梯度是

$$
\frac{\partial \mathcal{L}}{\partial x_j} = \sum_i \frac{\partial \mathcal{L}}{\partial y_i} A_{ij}
$$


这可以写成梯度向量的形式:

$$
(\frac{\partial \mathcal{L}}{\partial x})^\top = (\frac{\partial \mathcal{L}}{\partial y})^\top A
$$


根据不等式:

$$
|\frac{\partial \mathcal{L}}{\partial x_j}| \le \sum_i |\frac{\partial \mathcal{L}}{\partial y_i}| |A_{ij}|
$$


如果我们假设上游梯度向量的绝对值最大为  $M_g$ , 那么  $|\frac{\partial \mathcal{L}}{\partial x_j}| \le (\sum_i |A_{ij}|) M_g$ .

因此, 列和  $\sum_i |A_{ij}|$  直接决定了上游梯度在反向传播到输入第  $j$  个分量时可能被放大的最大倍数. 最大绝对列和就代表了在所有输入分量中最坏情况下的梯度放大因子.

#### 对上图的展开分析


上图中的(a)展示了单个  $\mathcal{H}_{l}^{\text{res}}$  矩阵的增益幅度, 即使是单个矩阵, 其Amax增益幅度也已经偏离了1, 大部分值在1到2之间, 有些甚至接近10.这表明HC的可学习映射在每一层都引入了微小但确定的放大效应. 虽然单层的放大看起来不严重(最多10倍), 但这正是问题的根源. 就像一个微小的正反馈, 它为后续的指数级爆炸埋下了伏笔. 这就像俗话说的"千里之堤, 毁于蚁穴".

图(b)是整篇论文中最具说服力的部分. 它展示了多个  $\mathcal{H}_{l}^{\text{res}}$  矩阵**连乘**后的复合效应. 随着层数的增加(x轴从左到右), 复合映射的增益幅度呈指数级增长. 注意y轴是对数尺度, 曲线近似线性就意味着指数增长.在网络的中后段, Amax增益幅度达到了  $10^3$  到  $10^4$  的量级, 也就是1000到10000倍, 论文中提到的"峰值达到3000"就是来源于此.

这张图证明了HC的内在不稳定性. 它清晰地可视化了在引言和第三章中描述的理论问题: **无约束矩阵的连乘导致信号/梯度的指数级爆炸**.

## 3.2 系统开销


虽然由于额外映射的线性特性, HC的计算复杂度仍然可控, 但系统级开销带来了不可忽略的挑战, 特别来说就是关于访问内存的开销.

下表分析了由  $n$  流残差设计在单个残差层中引入的每个token的内存访问开销.

![](images/img_016.png)


分析显示, HC将内存访问成本增加了大约与  $n$  成正比的一个因子. 如果没有融合算子的缓解, 这种过度的I/O需求会显著降低训练吞吐量.

此外, 由于  $\mathcal{H}_{l}^{\text{pre}}, \mathcal{H}_{l}^{\text{post}}$  和  $\mathcal{H}_{l}^{\text{res}}$  涉及可学习参数, 它们的反向传播需要中间激活值. 这导致GPU内存占用的巨大增加, 通常需要梯度Checkpoint来维持可行的内存使用.

此外, HC在流水线并行中需要  $n$  倍的通信成本, 导致更大的bubbles和训练吞吐量的下降.

**渣注:**


标准残差连接的总I/O约为  $3C$ . HC的总I/O约为  $8nC + \dots$ . 当  $n=4$  时, I/O成本大约是原来的  $10$  倍以上, 这是一个巨大的开销. 这个量化分析清晰地揭示了问题的严重性.

# 4. mHC算法和实现


## 4.1 流形约束超连接


mHC是从ResNet中的恒等映射原理中汲取灵感, 核心前提是将残差映射  $\mathcal{H}_{l}^{\text{res}}$  约束在一个特定的流形上. 原始的恒等映射通过强制  $\mathcal{H}_{l}^{\text{res}} = I$  (单位矩阵) 来确保稳定性, 但它从根本上排除了残差流内部的信息交换, 而这对于最大化多流架构的潜力至关重要. 因此作者建议将残差映射投影到一个既能维持跨层信号传播稳定性, 又能促进残差流之间相互作用以保持模型表达能力的流形上.

为此, 作者将  $\mathcal{H}_{l}^{\text{res}}$  限制为一个**双随机矩阵**, 其具有非负的元素, 并且行和与列和均为1.形式上, 令  $\mathcal{M}_{\text{res}}$  表示双随机矩阵的流形(也被称为Birkhoff多面体). 将  $\mathcal{H}_{l}^{\text{res}}$  约束为  $\mathcal{P}_{\mathcal{M}_{\text{res}}}(\mathcal{H}_{l}^{\text{res}})$ , 定义为:

$$
\mathcal{P}_{\mathcal{M}_{\text{res}}}(\mathcal{H}_{l}^{\text{res}}) := \left\{ \mathcal{H}_{l}^{\text{res}} \in \mathbb{R}^{n \times n} \mid \mathcal{H}_{l}^{\text{res}} \mathbf{1}_n = \mathbf{1}_n, \mathbf{1}_n^\top \mathcal{H}_{l}^{\text{res}} = \mathbf{1}_n^\top, \mathcal{H}_{l}^{\text{res}} \geq 0 \right\}
$$


其中  $\mathbf{1}_n$  代表全1的  $n$  维向量.值得注意的是, 当  $n=1$  时, 双随机条件退化为标量1, 从而恢复了原始的恒等映射.

然后作者展开讲了一下选择双随矩阵的流形的理论依据:

- **范数保持 (Norm Preservation)**: 双随机矩阵的谱范数被1所界定 (即,  $\|\mathcal{H}_{l}^{\text{res}}\|_2 \leq 1$ ). 这意味着该可学习映射是**非扩张的 (non-expansive)**, 有效地缓解了梯度爆炸问题.
- **组合闭包性 (Compositional Closure)**: 双随机矩阵的集合在矩阵乘法下是封闭的. 这确保了跨多层的复合残差映射  $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$  仍然是双随机的, 从而在模型的整个深度上保持了稳定性.
- **通过Birkhoff多面体的几何解释 (Geometric Interpretation via the Birkhoff Polytope)**:  $\mathcal{M}_{\text{res}}$  集合构成了Birkhoff多面体, 它是置换矩阵集合的凸包. 这提供了一个清晰的几何解释: 残差映射扮演着**置换的凸组合 (convex combination of permutations)** 的角色. 从数学上讲, 重复应用此类矩阵倾向于单调地增加跨流信息的混合, 有效地充当了一种鲁棒的特征融合机制.


此外,对输入映射  $\mathcal{H}_{l}^{\text{pre}}$  和输出映射  $\mathcal{H}_{l}^{\text{post}}$  施加了非负性约束. 这种约束防止了由正负系数组合引起的信号抵消, 这也可以被视为一种特殊的流形投影.

#### 双随机矩阵


一个  $n \times n$  的实数矩阵  $A$  被称为双随机矩阵, 如果它满足以下三个条件:

- `非负性`: 矩阵中所有元素都大于等于 0. $A_{ij} \ge 0$  for all  $i, j$ .
- `行和为1`:  $\sum_{j=1}^{n} A_{ij} = 1$  for all  $i = 1, \dots, n$ .
- `列和为1`:  $\sum_{i=1}^{n} A_{ij} = 1$  for all  $j = 1, \dots, n$ .


例子 ( $n=3$ ),下面这个矩阵就是一个双随机矩阵:

$$
A = \begin{pmatrix}
0.5 & 0.2 & 0.3 \\
0.1 & 0.7 & 0.2 \\
0.4 & 0.1 & 0.5
\end{pmatrix}
$$


可以验证:

- 所有元素非负.
- 第一行和:  $0.5 + 0.2 + 0.3 = 1$ . 第二行和:  $0.1 + 0.7 + 0.2 = 1$ . 第三行和:  $0.4 + 0.1 + 0.5 = 1$ .
- 第一列和:  $0.5 + 0.1 + 0.4 = 1$ . 第二列和:  $0.2 + 0.7 + 0.1 = 1$ . 第三列和:  $0.3 + 0.2 + 0.5 = 1$ .


#### Birkhoff多面体


Birkhoff 多面体  $B_n$  就是所有满足上述条件的  $n \times n$  矩阵所构成的集合.将一个  $n \times n$  矩阵看作  $n^2$  维空间中的一个点, 那么 Birkhoff 多面体  $B_n$  就是这个高维空间中的一个**凸多面体(convex polytope)**.

- **凸性(Convexity)**: 如果有两个双随机矩阵  $A$  和  $B$ , 那么它们的任意凸组合  $\lambda A + (1-\lambda)B$  (其中  $0 \le \lambda \le 1$ ) 也是一个双随机矩阵. 这意味着这个集合是凸的.
- **多面体(Polytope)**: 这个集合是由一系列线性等式(行和/列和为1)和线性不等式(元素非负)所界定的有界区域, 因此它是一个凸多面体.


展开一下这个多面体的顶点是什么? 一个多面体的几何性质很大程度上由它的顶点决定, 因为多面体内的任何一点都可以表示为其顶点的凸组合.

例子 ( $n=2$ ), 一个  $2 \times 2$  的双随机矩阵形如:

$$
A = \begin{pmatrix}
x & 1-x \\
1-x & x
\end{pmatrix}
$$


其中  $0 \le x \le 1$ .这个集合在几何上是一条连接以下两个点的线段:

- 当  $x=1$  时,  $A_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$
- 当  $x=0$  时,  $A_2 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$


这两个矩阵  $A_1$  和  $A_2$  就是  $B_2$  的顶点. 它们是两个  $2 \times 2$  的置换矩阵(permutation matrices).

#### Birkhoff-von Neumann 定理


Birkhoff 多面体  $B_n$  是所有  $n \times n$  置换矩阵的**凸包(convex hull)**. 换句话说,  $B_n$  的顶点恰好是  $n!$  个  $n \times n$  的置换矩阵.

也就是说, 如果 $X$ 是一个双随机矩阵, 那么存在 $\theta_1,...,\theta_k \ge 0, \sum_{i=1}^k \theta_i = 1$ 和置换矩阵  $P_1,...P_k$ 使得: $X = \theta_1P_1 + ... + \theta_kP_k$

#### 这对mHC的流形约束有什么作用?


对  $\mathcal{H}_{l}^{\text{res}}$  操作, 本质是"加权平均的洗牌", 对输入  $x_l$  应用  $\mathcal{H}_{l}^{\text{res}}$  操作, 即计算  $\mathcal{H}_{l}^{\text{res}} x_l$ , 等价于:

$$
\mathcal{H}_{l}^{\text{res}} x_l = (\sum \theta_k P_k) x_l = \sum \theta_k (P_k x_l)
$$


这个式子清晰地表明, mHC的残差更新过程, 是对输入  $x_l$  进行所有可能的  $n!$  种"洗牌"(置换)操作, 然后将这些"洗牌"后的结果进行加权平均. "加权平均"是一种典型的混合操作. 它将来自不同流的信息平滑地聚合在一起. 因为所有的系数  $\theta_k$  都是非负且和为1, 这个过程天然地不具备放大信号的能力. 任何一个输出流的范数都不会超过所有输入流范数的最大值. 这从几何上保证了信号的稳定性.

论文中提到mHC是一个"鲁棒的特征融合机制". 这里的鲁棒性就来源于Birkhoff多面体的凸性. 无论网络学习到的  $\mathcal{H}_{l}^{\text{res}}$  在多面体内部的哪个位置, 它都逃不出"加权平均洗牌"这个范畴, 其行为是可预测和有界的.

另一方面一个重要的性质是**乘法下的闭包性**: 两个双随机矩阵的乘积仍然是一个双随机矩阵.这是保证mHC**可扩展到任意深度**的关键. 考虑跨多层的复合映射  $\mathcal{H}_{\text{composite}} = \prod_{i=l}^{L-1} \mathcal{H}_{i}^{\text{res}}$ . 由于每个  $\mathcal{H}_{i}^{\text{res}}$  都是Birkhoff多面体中的一个点, 它们的乘积  $\mathcal{H}_{\text{composite}}$  也必定是Birkhoff多面体中的一个点. 这意味着, 无论网络有多深, 从任意浅层到任意深层的净残差传播效应, 始终等效于单个双随机矩阵的操作. 稳定性不会因为深度的增加而退化, 完美解决了HC的指数累积爆炸问题.

通过Birkhoff多面体的性质, 我们可以将mHC的流形约束理解为:

![](images/img_017.jpg)


## 4.2 参数化与流形投影


在本节中, 作者详细介绍mHC中  $\mathcal{H}_{l}^{\text{pre}}$ ,  $\mathcal{H}_{l}^{\text{post}}$  和  $\mathcal{H}_{l}^{\text{res}}$  的计算过程.

给定第  $l$  层的输入隐矩阵  $x_l \in \mathbb{R}^{n \times C}$ , 首先将其展平为一个向量  $\hat{x}_l = \text{vec}(x_l) \in \mathbb{R}^{1 \times nC}$  以保留完整的上下文信息.

然后, 遵循原始HC的公式来获得动态映射和静态映射, 如下所示:

$$
\begin{cases} \hat{x}'_l = \text{RMSNorm}(\hat{x}_l) \\ \tilde{\mathcal{H}}_{l}^{\text{pre}} = \alpha_{l}^{\text{pre}} \cdot (\hat{x}'_l \varphi_{l}^{\text{pre}}) + b_{l}^{\text{pre}} \\ \tilde{\mathcal{H}}_{l}^{\text{post}} = \alpha_{l}^{\text{post}} \cdot (\hat{x}'_l \varphi_{l}^{\text{post}}) + b_{l}^{\text{post}} \\ \tilde{\mathcal{H}}_{l}^{\text{res}} = \alpha_{l}^{\text{res}} \cdot \text{mat}(\hat{x}'_l \varphi_{l}^{\text{res}}) + b_{l}^{\text{res}} \end{cases}
$$


其中  $\varphi_{l}^{\text{pre}}, \varphi_{l}^{\text{post}} \in \mathbb{R}^{nC \times n}$  和  $\varphi_{l}^{\text{res}} \in \mathbb{R}^{nC \times n^2}$  是用于动态映射的线性投影, 而  $\text{mat}(\cdot)$  是一个从  $\mathbb{R}^{1 \times n^2}$  到  $\mathbb{R}^{n \times n}$  的reshape函数.

####  $vec(x_l)$ 的处理


将  $n \times C$  的多流输入矩阵展平成一个  $1 \times nC$  的长向量. 这是一个关键的改进. 在原始HC的公式中, 输入是  $RMSNorm(x_l)$ , 这通常意味着对每个流(每一行)单独做Norm来生成映射矩阵. 而mHC通过  $vec(x_l)$  保留了所有流的全部信息, 将它们拼接在一起. 这使得后续的线性投影  $\varphi$  可以访问到所有流的完整上下文, 从而能够学习到更复杂的流间依赖关系来生成映射矩阵.

例如,  $\varphi$  的权重可以学习到"如果第1个流的某个特征很强, 并且第3个流的某个特征很弱, 那么在  $\mathcal{H}^{\text{res}}$  中应该增强从第3流到第1流的连接强度". 这种全局上下文感知是原始HC所不具备的.

#### 动态映射与静态映射


- **动态部分**:  $(\hat{x}'_l \varphi_l)$ . 这是一个标准的线性层, 将  $1 \times nC$  的输入向量映射到目标尺寸 ( $1 \times n$  或  $1 \times n^2$ ). 这部分使得生成的映射矩阵是输入依赖的, 能够根据不同的输入token动态调整连接模式.
- **静态部分**:  $+ b_l$ . 这是一个可学习的偏置项, 代表了独立于输入的全局连接偏好.
- **门控因子**:  $\alpha_l$ . 这个因子与原始HC一致, 在训练初期通过小值初始化, 起到稳定训练的作用, 使得初始的映射矩阵接近于静态偏置  $b_l$ .


然后, 最终的约束映射通过以下方式获得:

$$
\begin{cases} \mathcal{H}_{l}^{\text{pre}} = \sigma(\tilde{\mathcal{H}}_{l}^{\text{pre}}) \\ \mathcal{H}_{l}^{\text{post}} = 2\sigma(\tilde{\mathcal{H}}_{l}^{\text{post}}) \\ \mathcal{H}_{l}^{\text{res}} = \text{Sinkhorn-Knopp}(\tilde{\mathcal{H}}_{l}^{\text{res}}) \end{cases}
$$


其中  $\sigma(\cdot)$  表示Sigmoid函数. $\text{Sinkhorn-Knopp}(\cdot)$  算子首先通过一个指数算子使所有元素为正, 然后执行一个交替重新缩放行和列以使它们的和为1的迭代归一化过程.

具体来说, 给定一个正矩阵  $M^{(0)} = \exp(\tilde{\mathcal{H}}_{l}^{\text{res}})$  作为起点, 归一化迭代过程如下: $M^{(t)} = T_r(T_c(M^{(t-1)}))$

其中  $T_r$  和  $T_c$  分别表示行和列的归一化. 当  $t_{\max} \to \infty$  时, 这个过程收敛到一个双随机矩阵  $\mathcal{H}_{l}^{\text{res}} = M^{(t_{\max})}$ . 在实验中, 作者选择  $t_{\max}=20$ .

这个阶段是mHC的核心创新所在, 它将阶段一生成的"原始"矩阵  $\tilde{\mathcal{H}}$  投影到目标流形上.

####  $\mathcal{H}_{l}^{\text{pre}}$  和  $\mathcal{H}_{l}^{\text{post}}$  的约束


原始的HC使用的函数为tanh,值域: (-1,1),可能产生负数.
