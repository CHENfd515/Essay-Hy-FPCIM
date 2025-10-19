# Vision Transformer (ViT-B/16) 模型结构
## [ 注！本分析后经检查发现有误 ]
基于 `torchvision.models.vit_b_16` 模型，分析其参数量（Num Params）和计算量（Mult-Adds / FLOPs）在各个主要模块中的分布和占比。

---

### I. 模型整体资源概览

ViT-B/16 是一个具有 **8580 万参数**（约 $86$M）和约 **5655 万次浮点运算**（FLOPs）的图像分类模型。

| 指标 (Metric) | 数量 (ViT-B/16) | 描述 |
| :--- | :--- | :--- |
| **总参数量** | $\approx 85.8$ Million | 衡量模型大小和内存占用。 |
| **总计算量 (Mult-Adds)** | $\approx 56.55$ Million | 衡量模型前向推理所需的计算资源。 |
| **Embedding Dimension** | $768$ | 序列中每个 Token 的特征维度。 |
| **Encoder Layer 数量** | $12$ | Transformer Block 的数量。 |

---

### II. 参数量 (Num Params) 分布占比

参数量决定了模型的存储大小和内存需求。在 ViT 中，**99% 的参数**都集中在 **Transformer Encoder** 内部。

| 模块 (Module) | 参数量总和 (Params) | **占总参数量的百分比** | 作用 |
| :--- | :--- | :--- | :--- |
| **Transformer Encoder (12 层)** | $84,957,000$ | $\mathbf{99.02\%}$ | 负责序列的自注意力计算和特征学习。 |
| **MLP Head (分类头)** | $769,000$ | $\approx 0.90\%$ | 将 Class Token 映射到 $1000$ 个分类输出。 |
| **Patch Embedding (Conv2d)** | $590,592$ | $\approx 0.69\%$ | 将图像块投影到 $768$ 维特征空间。 |
| **Positional/Class Tokens** | $151,776$ | $\approx 0.18\%$ | 可学习的位置编码和分类起点。 |

**结论：**

ViT 的参数分布是**高度集中**的。如果需要优化模型大小，主要的优化目标是 **12 个 Transformer Encoder Layer** 内部的线性层权重。

---

### III. 计算量 (Mult-Adds / FLOPs) 分布占比

计算量决定了模型的运行速度。ViT 是一个典型的**计算密集型**模型，其绝大部分 FLOPs 集中在核心的 $12$ 层。

| 模块 (Module) | 近似 Mult-Adds 总量 | **占总计算量的百分比** | 性能影响 |
| :--- | :--- | :--- | :--- |
| **Transformer Encoder (12 层)** | $\approx 56.4$ Million | $\approx \mathbf{99.8\%}$ | **决定模型推理速度的瓶颈。** |
| **Patch Embedding (Conv2d)** | $\approx 0.147$ Million | $\approx 0.2\%$ | 仅执行一次预处理，计算开销可忽略。 |
| **MLP Head (分类头)** | $\approx 1500$ 次 | $\approx 0.003\%$ | 最终一步映射，计算量极小。 |

**结论：**

ViT 的 **99.8% 以上的计算量**都发生在 **Transformer Encoder** 中。这意味着，提升 ViT 的推理速度，必须专注于**优化或减少 Transformer Block 的数量或维度**。

---

### IV. 单个 Transformer Layer 内部计算分解

进一步分析单个 Encoder Layer，可以定位计算最昂贵的子模块。

| Layer 内部模块 (Module) | Mult-Adds (单层) | **占该 Layer 总计算量的百分比** | 优化方向 |
| :--- | :--- | :--- | :--- |
| **MLP Block (FeedForward)** | $\approx 4.70$ Million | $\approx \mathbf{67\%}$ | **计算瓶颈：** 由于 $4$ 倍的维度膨胀（$768 \rightarrow 3072$）导致高 FLOPs。 |
| **Multihead Attention (MHA)** | $\approx 2.36$ Million | $\approx 33\%$ | 相比 MLP，FLOPs 相对较低，但负责序列的核心交互。 |
| **LayerNorm** | (极少) | $\approx 0\%$ | 归一化操作，计算开销极低。 |

**总结：**

从 **FLOPs 角度**来看，**MLP 块是单个 Transformer Layer 中计算工作量的主要来源。**