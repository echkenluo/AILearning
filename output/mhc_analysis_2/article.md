# 谈谈mHC-Lite:无需使用Sinkhorn-Knopp迭代的算法

## TL;DR


今天看到一篇论文《mHC-lite: You Don’t Need 20 Sinkhorn-Knopp Iterations》[^1]. 大概的内容是在DeepSeek mHC的实现时, 采用了固定的20次的Sinkhorn-Knopp迭代, 但是这个迭代产生的矩阵只是一个数值上近似于双随机矩阵.  然而我们可以根据Birkhoff–von Neumann 定理来处理, 该定理在前一篇文章[《谈谈DeepSeek mHC》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247497138&idx=1&sn=8215a15d8e196d412ab908ec3302c857&scene=21#wechat_redirect)中有详细阐述.

#### Birkhoff-von Neumann 定理


Birkhoff 多面体  $B_n$  是所有  $n \times n$  置换矩阵的**凸包(convex hull)**. 换句话说,  $B_n$  的顶点恰好是  $n!$  个  $n \times n$  的置换矩阵.

也就是说, 如果 $X$ 是一个双随机矩阵, 那么存在 $\theta_1,...,\theta_k \ge 0, \sum_{i=1}^k \theta_i = 1$ 和置换矩阵  $P_1,...P_k$ 使得: $X = \theta_1P_1 + ... + \theta_kP_k$

我们可以预先生成这  $n!$  个置换矩阵, 然后通过训练计算出参数  $\theta_i$ 的方式, 避免迭代. 例如  $n=4, 4!=24$ . 在  $n$  相对较小时计算复杂度也是可控的.

另外DeepSeek刚发布了一篇论文《Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models》[^2] 也挺有意思的, 过两天详细写一个分析.

# 1. 概述


HC无约束的残差矩阵可能会损害训练的稳定性. 为了解决这个问题, DeepSeek 团队的流形约束超连接 (mHC) 方法通过迭代式的 Sinkhorn–Knopp(SK)算法处理, 将这些矩阵**近似地**投影到 Birkhoff 多面体上. 但是这种做法作者认为存在局限性:

- 有限的 SK 迭代次数无法保证精确的双随机性, 留下了一个可能随网络深度累积并破坏稳定性的近似误差
- 高效的 SK 迭代需要特殊的Kernel和一些工程上的技巧.


受 Birkhoff–von Neumann 定理的启发, 作者提出了一种简单的重参数化方法 mHC-Lite, 它通过置换矩阵的凸组合来显式地构造双随机矩阵. 这种方法通过构造保证了精确的双随机性, 并且仅使用原生的矩阵运算即可实现.

![](images/img_003.png)


作者对于20次迭代的SK计算举了一个例子:

![](images/img_004.png)


经过20次 SK 迭代后, 输出矩阵的列和分别为1.92, 0.59, 和 0.59, 这与双随机性有巨大偏差, 这是潜在的会导致训练不稳定的因素.

# 2. SK迭代算法稳定性分析


在 mHC 中, 当收敛缓慢时, 固定次数的 SK 迭代 (例如, mHC 中的20次) 并不保证高质量的近似. 对于一般的非负矩阵, SK 算法只带有一个最坏情况下的迭代上界: 为了获得一个  $\ell_1$  误差最多为  $\epsilon$  的双随机近似, 可能需要多达  $O\left(\frac{n^2 \log(n/\nu)}{\epsilon^2}\right)$  次迭代, 其中相对范围  $\nu$  定义为: $\nu := \frac{\min_{i,j: x_{i,j}>0} x_{i,j}}{\max_{i,j} x_{i,j}}$

此处  $x_{i,j}$  是矩阵  $X$  的第  $(i,j)$  个元素. 即使对于严格为正的矩阵, 收敛速度仍然对  $1/\nu$  敏感, 当  $1/\nu$  很大时, 收敛可能极其缓慢.

这个问题在 mHC 中是实际存在的. SK 算法的输入是通过对一个特征的仿射函数进行指数化得到的, 这可能会产生具有非常大相对范围的病态矩阵. 在作者的测量中,  大约 27.9% 的 SK 输入满足  $1/\nu \ge 10^{13}$ . 在这种输入下, 固定的 SK 迭代预算可能无法生成一个接近双随机的矩阵, 如下图所示:

![](images/img_005.png)


下图显示, mHC 中单个残差矩阵的列和可能偏离1高达100%.

![](images/img_006.png)


更重要的是, 这些逐层的偏差会在深度上累积: 在一个24层的网络中,  $\prod_l H_{\text{res}}^l$  的列和可能偏离1高达220%, 这意味着当模型进一步扩大时存在不稳定的风险.

**渣注**


mHC 的计算流程  $H_{\text{res}}^l = \text{SK}(\exp(\dots))$  是产生病态矩阵的根源. exp 函数会极大地放大其输入的差异. 比如, 一个小的负数 exp(-10) 会变成一个极小的值 ( $\approx 4.5 \times 10^{-5}$ ), 而一个大的正数 exp(10) 会变成一个极大的值 ( $\approx 22026$ ). 这种巨大的数值差异导致了极小的  $\nu$  值和极大的  $1/\nu$ , 使得 SK 算法收敛困难.

当然从算法本身来看, 对于LLM模型的深度和迭代算法不收敛的影响在一个几十层的模型上影响还是可控的.

# 3. mHC-Lite的方法


基于 Birkhoff-von Neumann 定理, 直接将双随机矩阵表示为置换矩阵的凸组合. 这种参数化方法保证了生成的矩阵是**精确**双随机的. 此外, 通过消除迭代近似, 这种参数化移除了它们在训练和推理中的计算开销, 避免了对高度专业化基础设施的重度依赖.

在 mHC-lite 中, 为了控制混淆变量, 我们保持 mHC 的结构不变, 只改变了  $H_{\text{res}}^l$  的计算. 设  $x^l \in \mathbb{R}^{n \times C}$  是第 l 层的输入特征,  $\hat{x}^l \in \mathbb{R}^{1 \times nC}$  是其展平后的特征. 然后我们基于  $x^l$  动态构建映射  $H_{\text{res}}^l, H_{\text{pre}}^l$  和  $H_{\text{post}}^l$  如下: $\hat{x}'^l = \text{RMSNorm}(\hat{x}^l)$

$$
H_{\text{pre}}^l = \text{sigmoid}(\alpha_{\text{pre}}^l \hat{x}'^l W_{\text{pre}}^l + b_{\text{pre}}^l)
$$


$$
H_{\text{post}}^l = 2 \cdot \text{sigmoid}(\alpha_{\text{post}}^l \hat{x}'^l W_{\text{post}}^l + b_{\text{post}}^l)
$$


mHC-Lite的修改如下:

$$
a^l = \text{softmax}(\alpha_{\text{res}}^l \hat{x}'^l W_{\text{res}}^l + b_{\text{res}}^l)
$$

 $H_{\text{res}}^l = \sum_{k=1}^{n!} a_{l,k} P^k$

首先通过一个带有 softmax 激活函数的线性层计算出一个动态权重向量  $a^l = (a_{l,1}, \dots, a_{l,n!}) \in \mathbb{R}^{n!}$ . 然后  $P^k$  是一个预先生成的  $n!$  个置换矩阵. 因此只需要简单的一次矩阵乘法即可.

**渣注:**


其实整个算法还是很巧妙的, 该方法的可行性建立在  $n$  较小的基础上. 幸运的是, 受到传输带宽的约束, 在已有的工作中  $n=4$  是一个常用且效果好的选择, 此时  $n!=24$ , 完全在可接受的范围内. 那么我们仅需要生成24个置换矩阵, 总共有 24 * n^2 = 384个元素, 这是一个非常小的矩阵并可以放置在SMEM中.

然后将加权求和  $\sum_{k=1}^{n!} a_{l,k} P^k$  变成一次矩阵乘法 `a @ P_all`, 因此无需定制任何特殊的Kernel.它将一个约束优化问题 (找到最近的双随机矩阵) 转化为了一个无约束优化问题 (学习线性层的权重  $W_{\text{res}}^l$ ). 网络的梯度可以平滑地通过 `softmax` 和线性层反向传播, 优化过程非常标准.

然后整个实验可能是受到算力约束, 是基于nanoGPT框架做了几个小模型验证S (6层, 约45M参数), M (12层, 约0.12B参数), 和 L (24层, 约0.36B参数). 就不展开分析了.

---

## 参考资料

[1] 

mHC-lite: You Don’t Need 20 Sinkhorn-Knopp Iterations: *https://arxiv.org/pdf/2601.05732*

[2]
Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models: *https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf*


预览时标签不可点

微信扫一扫
关注该公众号        继续滑动看下一个         轻触阅读原文

![](images/img_007.png)


                        zartbot                                          向上滑动看下一个

知道了
          微信扫一扫
使用小程序        ****    取消 允许        ****    取消 允许        ****    取消 允许     × 分析      **

![](images/img_007.png)


微信扫一扫可打开此内容，
使用完整服务          ： ， ， ， ， ， ， ， ， ， ， ， ， 。   视频 小程序 赞 ，轻点两下取消赞 在看 ，轻点两下取消在看 分享 留言 收藏 听过
