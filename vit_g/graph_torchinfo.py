import torch
import torchvision.models as models
from torchinfo import summary

# --- ViT 模型参数 ---
# 我们使用 ViT-B/16 作为示例：
# B: Base Size
# 16: Patch Size (16x16)
BATCH_SIZE = 1
IMAGE_SIZE = 224
NUM_CLASSES = 1000 
# Note: ViT-B/16 的 embedding dimension 是 768，Transformer 深度是 12。

def print_vit_summary():
    print("Loading ViT-B/16 model from torchvision...")
    # 实例化模型 (weights=None 表示不加载预训练权重，只加载结构)
    model = models.vit_b_16(weights=None) 
    
    # 确保模型在评估模式下，以获得准确的摘要
    model.eval() 

    # 打印模型摘要
    # input_size 定义了虚拟输入张量的形状：(Batch Size, Channels, Height, Width)
    # depth=4 将深度展开到第四层，以便清晰地看到 MultiheadAttention 和 MLP 的结构
    summary(
        model, 
        input_size=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=4
    )

if __name__ == "__main__":
    print_vit_summary()