# res50_one_image_hw_eval.py
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from PIL import Image
import time

# 从 bf15_linear 导入：线性层替换 & 计数工具 & matmul实现（matmul会自动计数）
from bf15_linear import replace_linear_with_bf15, bf15_left_matmul
from bf15_linear import reset_hw_counter, report_hw_counter, count_matmul_tiles

# ---------- 单张图片 ----------
def load_one_image(img_path):
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    img = Image.open(img_path).convert("RGB")
    return preprocess(img).unsqueeze(0), weights

# ---------- 给 Conv2d 安装 forward hook，按等价 GEMM 统计 tiles ----------
def _register_conv_tile_counter(model):
    handles = []

    def conv_hook(mod, inp, out):
        # inp: Tuple[tensor]; out: tensor
        # 统计等价 GEMM:  A=[B*Hout*Wout, Cin_per_g * Kh*Kw],  B=[Cin_per_g*Kh*Kw, Cout_per_g]
        if not isinstance(mod, torch.nn.Conv2d):
            return
        x = inp[0]
        y = out
        if x is None or y is None:
            return
        B = x.shape[0]
        Cout, Hout, Wout = y.shape[1], y.shape[2], y.shape[3]
        Kh, Kw = mod.kernel_size
        Cin = mod.in_channels
        groups = mod.groups
        Cin_per_g = Cin // groups
        M = B * Hout * Wout
        K = Cin_per_g * Kh * Kw
        N = Cout // groups
        # 每个 group 都有一个这样的 GEMM
        tiles_per_group = count_matmul_tiles(M, K, N, kind="conv")
        # 乘以 groups
        # 这里分两种实现方式：
        #   1) 直接再加 (groups-1) 次；简单但直观
        #   2) 修改 count_matmul_tiles 接口接收 multiplier；为了最少改 bf15_linear，这里选 1)
        for _ in range(groups - 1):
            count_matmul_tiles(M, K, N, kind="conv")

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
    return handles

# ---------- 主流程：单张图 ----------
if __name__ == '__main__':
    # 1) 选择一张验证图片（请改成你机器上的路径）
    img_path = r"D:\Awesome-Backbones-main\imgnet\val\n01440764\ILSVRC2012_val_00000293.JPEG"
    x, weights = load_one_image(img_path)

    # 2) 模型
    model = models.resnet50(weights=weights)
    model.eval()

    # 3) 替换所有 Linear -> BF15 仿真（fc 会走 bf15_left_matmul，并自动计数）
    replace_linear_with_bf15(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    x = x.to(device, dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32)

    # 4) 复位计数器 & 安装 Conv2d 统计 hook
    reset_hw_counter()
    conv_hooks = _register_conv_tile_counter(model)

    # 5) 推理（单张）
    with torch.no_grad():
        t0 = time.time()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            y = model(x)
        t1 = time.time()

    # 6) 打印 top-5
    prob = torch.softmax(y[0], dim=0)
    top5 = torch.topk(prob, 5)
    print("\nTop-5:")
    for i in range(5):
        print(f"{i+1}: id={top5.indices[i].item()}, p={top5.values[i].item():.4f}")

    print(f"\n实际推理耗时: {(t1 - t0):.4f} s")

    # 7) 硬件统计（包含 fc 的 matmul + 全部 Conv2d）
    print("\n--- 硬件模拟统计（含卷积等价 GEMM）---")
    report_hw_counter(freq_hz=200_000_000)

    # 8) 清理 hook
    for h in conv_hooks:
        h.remove()
