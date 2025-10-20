# vit_hw_eval.py
import torch
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights
from torchvision import transforms
from PIL import Image
import time, json, os

# ===============================
# 导入 BF15 仿真函数与计数工具
# ===============================
from bf15_linear import (
    replace_linear_with_bf15,
    bf15_left_matmul,   # 我们会统计 tile 数
    reset_hw_counter,
    report_hw_counter,
    count_matmul_tiles
)

# ===============================
# 单张图片加载
# ===============================
def load_one_image(img_path):
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    img = Image.open(img_path).convert("RGB")
    return preprocess(img).unsqueeze(0), weights


# ===============================
# Patch torch.matmul 统计 ViT 内部所有 matmul
# ===============================
class PatchMatmulLeftBF15:
    def __enter__(self):
        # 记录原始实现，避免递归
        self._orig_matmul = torch.matmul
        self._orig_linear = F.linear

        def hooked_matmul(a, b):
            # 仅处理浮点二维/批量 matmul
            if not (torch.is_floating_point(a) and torch.is_floating_point(b)):
                return self._orig_matmul(a, b)
            try:
                # 2D
                if a.dim() == 2 and b.dim() == 2:
                    M, K = a.shape
                    K2, N = b.shape
                    if K == K2:
                        count_matmul_tiles(M, K, N, kind="matmul")
                    return bf15_left_matmul(a, b)
                # batched: 展平逐批处理（形状 [B, M, K] @ [B, K, N]）
                elif a.dim() >= 3 and b.dim() >= 3 and a.shape[-1] == b.shape[-2]:
                    a2 = a.reshape(-1, a.shape[-2], a.shape[-1])
                    b2 = b.reshape(-1, b.shape[-2], b.shape[-1])
                    outs = []
                    for i in range(a2.shape[0]):
                        M, K = a2[i].shape
                        K2, N = b2[i].shape
                        if K == K2:
                            count_matmul_tiles(M, K, N, kind="matmul")
                        outs.append(bf15_left_matmul(a2[i], b2[i]))
                    return torch.stack(outs, dim=0).reshape(*a.shape[:-2], a.shape[-2], b.shape[-1])
                else:
                    return self._orig_matmul(a, b)
            except Exception as e:
                print(f"[WARN] matmul fallback: {e}")
                return self._orig_matmul(a, b)

        def hooked_linear(input, weight, bias=None):
            # F.linear: out = input @ weight.T + bias
            # 仅处理浮点；其余保持原实现
            if not (torch.is_floating_point(input) and torch.is_floating_point(weight)):
                return self._orig_linear(input, weight, bias)

            try:
                # 将最后一维视为 K，先展平成 [M, K] 做一次 matmul，再还原形状
                in_shape = input.shape
                K = in_shape[-1]
                N = weight.shape[0]   # out_features
                M = int(input.numel() // K)
                x2d = input.reshape(M, K)
                # 统计 tile（x @ weight.T）
                count_matmul_tiles(M, K, N, kind="matmul")
                y2d = bf15_left_matmul(x2d, weight.t())
                if bias is not None:
                    y2d = (y2d.to(torch.float32) + bias.to(torch.float32)).to(torch.bfloat16)
                return y2d.reshape(*in_shape[:-1], N)
            except Exception as e:
                print(f"[WARN] linear fallback: {e}")
                return self._orig_linear(input, weight, bias)

        # 打补丁
        torch.matmul = hooked_matmul
        F.linear = hooked_linear
        return self

    def __exit__(self, exc_type, exc, tb):
        # 还原
        torch.matmul = self._orig_matmul
        F.linear = self._orig_linear
        return False

# ===============================
# 主流程
# ===============================
if __name__ == "__main__":
    # 1) 选择一张图像路径
    img_path = r"D:\Awesome-Backbones-main\imgnet\val\n01440764\ILSVRC2012_val_00000293.JPEG"
    x, weights = load_one_image(img_path)

    # 2) 加载 ViT 模型
    model = models.vit_b_16(weights=weights)
    model.eval()

    # 3) 替换所有 Linear -> BF15 仿真模块
    replace_linear_with_bf15(model)
    print("已替换所有 nn.Linear -> BF15IntLinear。")

    # 4) 模型与输入设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    x = x.to(device, dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32)

    # 5) 复位硬件计数器
    reset_hw_counter()

    # 6) 单张图片推理
    print("\n开始 ViT-B/16 单张推理与硬件统计...")
    with torch.no_grad():
        t0 = time.time()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            with PatchMatmulLeftBF15():
                y = model(x)
        t1 = time.time()

    # 7) 打印结果
    prob = torch.softmax(y[0], dim=0)
    top5 = torch.topk(prob, 5)
    print("\nTop-5:")
    for i in range(5):
        print(f"{i+1}: id={top5.indices[i].item()}, p={top5.values[i].item():.4f}")

    print(f"\n实际推理耗时: {(t1 - t0):.4f} s")

    print("\n--- 硬件模拟统计（所有 matmul，包括注意力/MLP）---")
    report_hw_counter(freq_hz=200_000_000)
