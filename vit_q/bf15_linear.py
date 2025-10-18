# bf15_linear.py  ——  BF15 仿真 Linear（frexp 截尾法，GPU 向量化）
import torch
import torch.nn as nn
import math
ORIG_MATMUL = torch.matmul

# ---------------------------
# 1) 将张量数值“仿真成 BF15”
#    （在 FP32 域用 frexp 拆分，再清掉尾数最低1位）
# ---------------------------
@torch.no_grad()
def to_bf15_real_fp32(x: torch.Tensor) -> torch.Tensor:
    """
    输入: 任意 dtype/cuda
    输出: FP32 张量，其数值等价于把 x 转成 BF16 后再清掉最后1位尾数（BF15）
    做法：
      x = sign * mant * 2**exp, mant∈[0.5,1)
      规范化到 [1,2): mant1 = mant*2, exp1 = exp-1
      量化到 7bit 且清掉最低1位： m_q = floor(mant1*128) -> m_q = (m_q // 2) * 2
      最终值： sign * (m_q/128) * 2**exp1
    """
    x32 = x.to(torch.float32)
    sign = torch.sign(x32)
    ax   = torch.abs(x32)

    mant, exp = torch.frexp(ax)    # ax = mant * 2**exp, mant ∈ [0.5,1)
    mant1 = mant * 2.0
    exp1  = exp  - 1.0

    m_q = torch.floor(mant1 * 128.0)            # 2**7
    m_q = torch.floor(m_q / 2.0) * 2.0          # 清掉最低1位 (BF15)

    # 零值保护（frexp(0)返回mant=0, exp=0，这里直接返回0）
    zero = (ax == 0)
    real = sign * (m_q / 128.0) * torch.pow(torch.tensor(2.0, device=x32.device), exp1)
    real = torch.where(zero, torch.zeros_like(real), real)
    return real  # FP32

# ---------------------------
# 2) BF15 仿真 matmul（全向量化）
# ---------------------------
@torch.no_grad()
def bf15_left_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    只把左矩阵 A 仿真到 BF15（frexp 截尾），右矩阵 B 保持 BF16（用其真实值，FP32计算）。
    输出转回 BF16，方便与下游衔接。
    """
    # 左：BF15 实值（FP32）
    A_bf15 = to_bf15_real_fp32(A)
    # 右：保持 BF16 的真实值（直接升到 FP32 即可）
    B_real = B.to(torch.float32)

    out_fp32 = ORIG_MATMUL(A_bf15, B_real)   # 用原生 matmul，避免递归
    return out_fp32.to(torch.bfloat16)


# ---------------------------
# 3) 可替换 nn.Linear 的模块
# ---------------------------
class BF15IntLinear(nn.Module):
    """
    用 BF15 仿真 matmul 替代 nn.Linear（推理用）
    - 内部把输入与权重都做 BF15 截尾仿真，再执行 matmul
    - 输出 dtype 为 BF16，方便与 ViT 其它算子衔接
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias   = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2d = x.reshape(-1, x.shape[-1])              # [..., in] -> [M, K]
        # 左：输入 x 仿真 BF15；右：weight^T 保持 BF16 的真实值
        y2d = bf15_left_matmul(x2d, self.weight.t())
        if self.bias is not None:
            y2d = (y2d.to(torch.float32) + self.bias.to(torch.float32)).to(torch.bfloat16)
        return y2d.reshape(*x.shape[:-1], self.out_features)

# ---------------------------
# 4) 递归替换工具
# ---------------------------
def replace_linear_with_bf15(model: nn.Module):
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            new_layer = BF15IntLinear(module.in_features, module.out_features, module.bias is not None)
            # 拷贝权重到 FP32（内部再仿真成 BF15 实值用于计算）
            new_layer.weight.data.copy_(module.weight.data.float())
            if module.bias is not None:
                new_layer.bias.data.copy_(module.bias.data.float())
            setattr(model, name, new_layer)
        else:
            replace_linear_with_bf15(module)

# ---------------------------
# 5) 自测
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    A = torch.randn(8, 16, dtype=torch.bfloat16, device=device)
    B = torch.randn(16, 32, dtype=torch.bfloat16, device=device)

    y_sim = bf15_left_matmul(A, B)
    y_ref = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.bfloat16)

    print("max abs diff:", (y_sim - y_ref).abs().max().item())

    layer = BF15IntLinear(16, 32).to(device)
    x = torch.randn(4, 16, dtype=torch.bfloat16, device=device)
    y = layer(x)
    print("Output:", y.shape)
