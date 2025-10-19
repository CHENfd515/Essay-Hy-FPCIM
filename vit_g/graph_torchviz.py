import subprocess
import torch
import torchvision.models as models
from torchviz import make_dot
import os

# --- 核心可视化脚本 ---

def visualize_vit_with_torchviz():
    # 1. 检查 Graphviz 是否可用 (可选但推荐)
    try:
        # 尝试运行 'dot -V' 来验证 Graphviz 是否在 PATH 中
        subprocess.run(['dot', '-V'], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("------------------------------------------------------------------")
        print("ERROR: Graphviz is NOT found or NOT correctly configured.")
        print("Please install Graphviz on your system and ensure it is in your PATH.")
        print("------------------------------------------------------------------")
        return

    # 2. 实例化 ViT 模型
    # 使用 vit_b_16 结构，不加载预训练权重
    print("Loading ViT-B/16 model...")
    model = models.vit_b_16(weights=None)
    model.eval() # 将模型设置为评估模式

    # 3. 创建虚拟输入 (Batch Size 1, 3 Channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Using dummy input shape: {dummy_input.shape}")

    # 4. 执行前向传播并生成计算图
    try:
        output = model(dummy_input)
        
        # 使用 make_dot 生成图形
        # params=dict(model.named_parameters()) 将模型参数添加到图中作为节点
        graph = make_dot(output, params=dict(model.named_parameters()))
        
        # 5. 渲染并保存图形文件
        file_name = "vit_b_16_graph"
        # format="png" 或 "svg" (SVG 文件在浏览器中查看可缩放，适合大型图)
        graph.render(file_name, format="png", cleanup=True)
        
        print("\nVisualization successful!")
        print(f"Output file saved as: {file_name}.png")
        print("NOTE: This is a very large, nested graph. Use SVG format or zoom heavily.")
        
    except Exception as e:
        print(f"\nAn error occurred during visualization: {e}")
        print("Ensure PyTorch and other dependencies are correctly installed.")


if __name__ == "__main__":
    visualize_vit_with_torchviz()