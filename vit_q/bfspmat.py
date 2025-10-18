import torch
import torch.nn.functional as F

# --------- 和原实现保持一致的拆分函数（支持 left=BF15 截尾） ----------
@torch.no_grad()
def split_to_bf_parts(x: torch.Tensor, left: bool):
    """
    输出:
      m_signed: int32, 代表带符号的“规格化尾数 * 128”（[±128..±255]）
      e_unbias: int32, 无偏指数（2 的指数），满足:  real ~= (m_signed / 128) * 2**e_unbias
    left=True 时对 m_signed 清掉最低 1 bit → 对应 BF15（左矩阵用）
    """
    x32  = x.to(torch.float32)
    zero = (x32 == 0)
    sgn  = torch.sign(x32)
    ax   = torch.abs(x32)

    mant, exp = torch.frexp(ax)          # ax = mant * 2**exp, mant∈[0.5,1)
    mant1 = mant * 2.0                   # → [1,2)
    e_unb = exp - 1.0

    m_scaled = torch.floor(mant1 * 128.0).to(torch.int32)   # [128..255]
    if left:
        m_scaled = (m_scaled >> 1) << 1                     # 清掉 LSB（BF15）

    m_signed = torch.where(sgn < 0, -m_scaled, m_scaled).to(torch.int32)
    m_signed = torch.where(zero, torch.zeros_like(m_signed), m_signed)
    e_unb    = torch.where(zero, torch.zeros_like(e_unb),    e_unb).to(torch.int32)
    return m_signed, e_unb


# ------------------ 优化版：全 GPU 并行的块内整数仿真 + 块间浮点累加 ------------------
@torch.no_grad()
def bf15_left_exp_int_matmul(A: torch.Tensor, B: torch.Tensor, tile_k: int = 64) -> torch.Tensor:
    """
    A: [M,K]（左矩阵量化为 BF15：清尾数 LSB）
    B: [K,N]（右矩阵保持 BF16 精度：7-bit 尾数，不清 LSB）
    过程：
      - 块内（tile_k）：整数域仿真（指数对齐 + 尾数右移“趋零截断” + 沿K求和）
      - 块间（所有 tile）：一次性浮点累加（batch reduce），保持与旧版一致的误差
    返回：FP32（可按需 .to(torch.bfloat16)）
    """
    assert A.dim() == 2 and B.dim() == 2
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimension mismatch"

    device = A.device
    # 1) 拆分（保持与旧逻辑一致）
    Am, Ae = split_to_bf_parts(A, left=True)    # int32
    Bm, Be = split_to_bf_parts(B, left=False)   # int32

    # 2) 将 K 维分块并“批处理”（把所有 tile 堆到 batch 维，GPU 并行处理）
    T = (K + tile_k - 1) // tile_k                        # tile 个数
    pad_k = T * tile_k - K                                # 末块 padding 的长度

    if pad_k > 0:
        # 左矩阵沿 K 维 pad 在末尾（0 填充不影响数值）
        Am = F.pad(Am, (0, pad_k), value=0)
        Ae = F.pad(Ae, (0, pad_k), value=0)
        # 右矩阵沿 K 维 pad 在开头或末尾均可（与上面保持一致的方向）
        Bm = F.pad(Bm, (0, 0, 0, pad_k), value=0)  # pad on dim=0 (K)
        Be = F.pad(Be, (0, 0, 0, pad_k), value=0)

    # 现在 K_pad = T * tile_k，可 reshape 为 [T, *]
    K_pad = T * tile_k
    Am_t = Am.reshape(M, T, tile_k).transpose(0, 1).contiguous()      # [T, M, tile_k]
    Ae_t = Ae.reshape(M, T, tile_k).transpose(0, 1).contiguous()      # [T, M, tile_k]
    Bm_t = Bm.reshape(T, tile_k, N).contiguous()                       # [T, tile_k, N]
    Be_t = Be.reshape(T, tile_k, N).contiguous()                       # [T, tile_k, N]

    # 3) 计算每个 tile 的逐项乘积与指数和 （全部张量化，GPU 一次性并行）
    # prod: [T, M, tile_k, N]   （int32）
    prod  = Am_t.unsqueeze(-1) * Bm_t.unsqueeze(1)                     # (T,M,t,N)
    e_sum = Ae_t.unsqueeze(-1) + Be_t.unsqueeze(1)                     # (T,M,t,N)

    # 4) 块内指数对齐：E_ref_tile = max_k(e_sum)，右移“趋零截断”
    E_ref = e_sum.max(dim=2, keepdim=True).values                      # (T,M,1,N)
    shift = (E_ref - e_sum).clamp_min_(0)                              # (T,M,t,N), int32

    # —— 用整数位移模拟 trunc toward 0:   aligned = trunc(prod / 2**shift)
    #     算法：abs() 右移，最后按符号恢复
    abs_prod    = prod.abs()
    aligned_mag = abs_prod >> shift                                    # (T,M,t,N), int32
    aligned     = torch.where(prod >= 0, aligned_mag, -aligned_mag)    # (T,M,t,N), int32

    # 5) 沿 k（tile_k）求和得到每个 tile 的 S_tile、E_tile
    S_tile = aligned.sum(dim=2, dtype=torch.int32)                     # (T,M,N), int32
    E_tile = E_ref.squeeze(2).to(torch.int32)                          # (T,M,N), int32

    # 6) 块间：浮点域一次性累加（与旧版语义一致）
    #    两个 7-bit mantissa 相乘 → 总指数需减 14（2**14）
    #    使用 torch.ldexp(mantissa, exponent) 更稳定：mantissa * 2**exponent
    tile_fp32 = torch.ldexp(S_tile.to(torch.float32), (E_tile - 14))   # (T,M,N), float32
    Y_fp32    = tile_fp32.sum(dim=0)                                   # (M,N),  float32

    del Am, Ae, Bm, Be, e_sum, prod, aligned_mag, aligned, S_tile, E_tile, tile_fp32, abs_prod, shift, E_ref, Am_t, Ae_t, Bm_t, Be_t
    torch.cuda.empty_cache()
    
    return Y_fp32  # .to(torch.bfloat16) 也可以

# ====================== 额外集成：BF15 线性层 & ViT 一键替换 ======================
import torch
import torch.nn as nn
import math

@torch.no_grad()
def to_bf15_real_fp32(x: torch.Tensor) -> torch.Tensor:
    """
    将张量数值“仿真”为 BF15 的实值（返回 FP32）：
      1) frexp 拆分 x = sign * mant * 2**exp, mant∈[0.5,1)
      2) 映射到 [1,2): mant1=mant*2, exp1=exp-1
      3) 量化到 7bit，再把最低 1 位清零（BF15）
      4) 还原：sign * (m_q/128) * 2**exp1
    说明：仅对“左操作数”做 BF15 截尾时，用这个输出作为 matmul 的输入更方便。
    """
    x32  = x.to(torch.float32)
    zero = (x32 == 0)
    sgn  = torch.sign(x32)
    ax   = torch.abs(x32)

    mant, exp = torch.frexp(ax)      # mant∈[0.5,1)
    mant1 = mant * 2.0               # [1,2)
    exp1  = exp  - 1.0

    m_q = torch.floor(mant1 * 128.0) # 7bit
    m_q = torch.floor(m_q / 2.0) * 2.0   # 清掉 LSB → BF15

    real = sgn * (m_q / 128.0) * torch.pow(torch.tensor(2.0, device=x32.device), exp1)
    real = torch.where(zero, torch.zeros_like(real), real)
    return real  # FP32


class BF15IntLinear(nn.Module):
    """
    用你的 bf15_left_exp_int_matmul 做核心 GEMM 的 Linear：
      - 左输入仿真为 BF15（指数/尾数按你的实现来）
      - 权重保持 BF16 实值路径（内部算法会拆分指数/尾数并对齐）
      - 输出默认转回 BF16，方便接到 ViT 其它算子
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
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [*, in] → [M, K]
        x2d = x.reshape(-1, x.shape[-1])
        # 用你的 BF15 左乘仿真 matmul
        y2d = bf15_left_exp_int_matmul(x2d, self.weight.t())
        if self.bias is not None:
            y2d = (y2d.to(torch.float32) + self.bias.to(torch.float32)).to(torch.bfloat16)
        # [M, out] → [*, out]
        return y2d.reshape(*x.shape[:-1], self.out_features)


@torch.no_grad()
def replace_linear_with_bf15(model: nn.Module):
    """
    递归把模型里所有 nn.Linear 换成 BF15IntLinear，并拷贝参数。
    用法：replace_linear_with_bf15(model)
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            new_layer = BF15IntLinear(module.in_features, module.out_features, module.bias is not None)
            new_layer.weight.data.copy_(module.weight.data.float())
            if module.bias is not None:
                new_layer.bias.data.copy_(module.bias.data.float())
            setattr(model, name, new_layer)
        else:
            replace_linear_with_bf15(module)


# ===================== Demo =======================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    A = torch.randn(128, 256, dtype=torch.bfloat16, device=device)
    B = torch.randn(256, 128, dtype=torch.bfloat16, device=device)

    y_sim = bf15_left_exp_int_matmul(A, B, tile_k=64)
    y_ref = torch.matmul(A.to(torch.float32), B.to(torch.float32))

    diff = (y_sim - y_ref).abs()
    print("max abs diff:", float(diff.max()))
    print("mean abs diff:", float(diff.mean()))
    print("Output:", y_sim.shape)
