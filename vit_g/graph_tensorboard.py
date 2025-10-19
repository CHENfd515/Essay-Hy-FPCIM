import torch
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

# 1. 加载模型
model = models.vit_b_16(weights=None) 

# 2. 创建虚拟输入
# (Batch Size, Channels, Height, Width)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. 创建 Writer 并添加图
writer = SummaryWriter('runs/vit_visualization') # log 目录
writer.add_graph(model, dummy_input)
writer.close()

print("TensorBoard log files created in the 'runs/vit_visualization' directory.")
print("Run 'tensorboard --logdir=runs' in your terminal to view the graph.")