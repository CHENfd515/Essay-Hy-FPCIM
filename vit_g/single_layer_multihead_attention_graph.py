import torch
import torch.nn as nn
from torchviz import make_dot
import subprocess
import os

# ViT-B/16 的配置参数
EMBED_DIM = 768  # 特征维度 D
NUM_HEADS = 12   # 注意力头数量
SEQUENCE_LENGTH = 197 # 序列长度 (196 patches + 1 CLS token)

def visualize_multihead_attention_svg():
    print("--- 检查 Graphviz ---")
    try:
        # 验证 Graphviz 是否安装并配置
        subprocess.run(['dot', '-V'], check=True, capture_output=True)
        print("Graphviz 已找到。")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("------------------------------------------------------------------")
        print("ERROR: Graphviz is NOT found or NOT correctly configured.")
        print("请先安装 Graphviz (如: apt install graphviz) 并确保它在 PATH 中。")
        print("------------------------------------------------------------------")
        return

    # 1. 实例化 MultiheadAttention 模块
    print("实例化 MultiheadAttention 模块...")
    mha_layer = nn.MultiheadAttention(
        embed_dim=EMBED_DIM, 
        num_heads=NUM_HEADS, 
        batch_first=True
    )
    mha_layer.eval()

    # 2. 创建虚拟输入 (Self-Attention: Q=K=V)
    dummy_input = torch.randn(1, SEQUENCE_LENGTH, EMBED_DIM) 
    query = key = value = dummy_input
    
    print(f"使用虚拟输入形状 (B, L, D): {query.shape}")

    # 3. 执行前向传播
    attn_output, _ = mha_layer(query, key, value)

    # 4. 使用 make_dot 生成计算图
    graph = make_dot(
        attn_output, 
        params=dict(mha_layer.named_parameters())
    )

    # 5. 渲染并保存图形文件为 SVG 格式
    file_name = "single_layer_multihead_attention_graph"
    # 关键修改：format="svg"
    graph.render(file_name, format="svg", cleanup=True)
    
    print("\n可视化成功！")
    print(f"结构图已保存为: {file_name}.svg")
    print("您可以使用浏览器打开此文件，并进行无损缩放查看。")

if __name__ == "__main__":
    visualize_multihead_attention_svg()