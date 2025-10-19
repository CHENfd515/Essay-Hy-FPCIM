from torchgraph import dispatch_capture, aot_capture, compile_capture
# torchgraph from https://github.com/alpha0422/torch-graph.git
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义多头注意力模型
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 确保嵌入维度能被头数整除
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        # 定义线性层
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # 计算Q、K、V
        qkv = self.qkv_proj(x)  # shape: (batch_size, seq_len, 3*embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # shape: (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv.unbind(0)  # 分别获取Q、K、V
        
        # 注意力计算
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # 点积注意力
        attn_scores = attn_scores / (self.head_dim ** 0.5)  # 缩放
        attn_probs = F.softmax(attn_scores, dim=-1)  # 注意力权重
        
        # 加权求和
        attn_output = torch.matmul(attn_probs, v)  # 加权求和
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)  # 重塑
        
        # 输出投影
        output = self.out_proj(attn_output)
        return output

# 初始化模型和输入
embed_dim = 64
num_heads = 4
model = MultiHeadAttention(embed_dim, num_heads).cuda()
x = torch.randn((2, 10, embed_dim), requires_grad=True, device="cuda")  # (batch_size, seq_len, embed_dim)

# 抓取计算图（三种方式）
dispatch_capture(model, x)  # 通过dispatch模式抓取（包含前向+反向）
aot_capture(model, x)      # 通过AOTAutograd抓取（前向和反向分离）
compile_capture(model, x)  # 通过torch.compile抓取（前向）