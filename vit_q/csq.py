import torch
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
from torchvision import transforms
from PIL import Image
import os, time, json
from torch.utils.data import Dataset, DataLoader
from bfspmat import replace_linear_with_bf15, to_bf15_real_fp32   # ✅ 新版导入

# ===============================
# ImageNet 验证集 Dataset
# ===============================
class ImageNetValDataset(Dataset):
    def __init__(self, val_root, wnid_map, transform):
        self.samples = []
        self.transform = transform
        self.wnid_map = wnid_map
        for wnid in sorted(os.listdir(val_root)):
            class_dir = os.path.join(val_root, wnid)
            if not os.path.isdir(class_dir):
                continue
            target = wnid_map.get(wnid)
            if target is None:
                continue
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, img_file), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, target


# ===============================
# BF15 Matmul Patch（新版）
# ===============================
from bfspmat import bf15_left_exp_int_matmul   # ✅ 导入你的函数

class PatchMatmulLeftBF15:
    def __enter__(self):
        # 记录原始 matmul
        self._orig = torch.matmul

        def hooked_matmul(a, b):
            # 只处理浮点张量
            if not (torch.is_floating_point(a) and torch.is_floating_point(b)):
                return self._orig(a, b)

            # ✅ 直接使用你的 bf15_left_exp_int_matmul 进行替换
            try:
                # 检查是否满足 2D 要求（你的函数要求 A,B 是二维矩阵）
                if a.dim() == 2 and b.dim() == 2:
                    return bf15_left_exp_int_matmul(a, b, tile_k=64).to(torch.bfloat16)
                else:
                    # 对于多维情况（如 batch matmul），可以逐批调用
                    a2 = a.reshape(-1, a.shape[-2], a.shape[-1])
                    b2 = b.reshape(-1, b.shape[-2], b.shape[-1])
                    outs = []
                    for i in range(a2.shape[0]):
                        outs.append(bf15_left_exp_int_matmul(a2[i], b2[i], tile_k=64))
                    return torch.stack(outs, dim=0).to(torch.bfloat16)
            except Exception as e:
                print(f"[WARN] bf15_left_exp_int_matmul fallback: {e}")
                return self._orig(a, b)

        torch.matmul = hooked_matmul
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.matmul = self._orig
        return False



# ===============================
# 主流程
# ===============================
if __name__ == '__main__':
    imagenet_root = "D:\\Awesome-Backbones-main\\imgnet\\val"
    map_filename = "D:\\Awesome-Backbones-main\\imgnet\\imagenet_wnid_to_model_id.json"

    with open(map_filename, 'r') as f:
        wnid_map = json.load(f)
    print(f"加载了 {len(wnid_map)} 条 WNID 映射。")

    weights_sel = ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights_sel)
    preprocess = weights_sel.transforms()
    model.eval()

    # ✅ 替换 Linear 层为 BF15 仿真模块
    replace_linear_with_bf15(model)
    print("已替换所有 nn.Linear -> BF15IntLinear。")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
        print("模型权重已转换为 BF16 存储。")
    else:
        print("当前设备不支持 BF16，保持 FP32。")
    print(f"使用设备: {device}")

    dataset = ImageNetValDataset(imagenet_root, wnid_map, preprocess)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print(f"验证集共 {len(dataset)} 张图片，批量大小 64")

    total, correct_top1, correct_top5, total_inference_time = 0, 0, 0, 0.0

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            start_time = time.time()
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
                with PatchMatmulLeftBF15():
                    outputs = model(imgs)
            end_time = time.time()

            total_inference_time += (end_time - start_time)
            total += targets.size(0)

            _, pred = outputs.topk(5, 1, True, True)
            correct = pred.eq(targets.view(-1, 1).expand_as(pred))
            correct_top1 += correct[:, :1].sum().item()
            correct_top5 += correct[:, :5].sum().item()

            if total % 3200 == 0:
                print(f"已推理 {total}/{len(dataset)} 张...")

    acc1 = correct_top1 / total * 100
    acc5 = correct_top5 / total * 100
    avg_time = total_inference_time / total

    print(f"\n--- ViT-B/16 (BF15仿真Linear) 验证结果 ---")
    print(f"Top-1 准确率: {acc1:.2f}%")
    print(f"Top-5 准确率: {acc5:.2f}%")
    print(f"平均单张推理时间: {avg_time:.4f} 秒")

    print(f"\n官方指标: {weights_sel.meta.get('_metrics', {}).get('ImageNet-1K', {})}")
