"""
Multi-Head Attention（多头注意力）示例
- 每个头独立执行缩放点积注意力，然后拼接并线性整合
- 维度关系：d_model = num_heads * d_k
输入形状：
- q, k, v: [batch, seq, d_model]
输出：
- out: [batch, seq, d_model]
- attn_weights: [batch, heads, seq, seq]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v 的形状均为: (batch_size, num_heads, seq_len, d_k)
    """
    d_k = q.size(-1)
    
    # 1. 计算点积得分: Q * K^T
    # transpose(-2, -1) 是为了将最后两个维度转置，以便进行矩阵乘法
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 如果有掩码 (Mask)，将对应位置设为极小值，这样 Softmax 后的权重接近 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 3. Softmax 归一化得到注意力权重
    attn_weights = F.softmax(scores, dim=-1)
    
    # 4. 加权求和得到最终输出
    output = torch.matmul(attn_weights, v)
    
    return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 定义四个线性层 (Q, K, V 的投影以及最后的输出投影)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. 线性变换并拆分为多个头
        # (batch, seq, d_model) -> (batch, seq, heads, d_k) -> (batch, heads, seq, d_k)
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 调用刚才写的 Scaled Dot-Product Attention
        x, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # 3. 拼接 (Concat) 所有头的输出
        # (batch, heads, seq, d_k) -> (batch, seq, heads, d_k) -> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. 最后的线性整合
        return self.fc_out(x), attn_weights

# 测试一下维度
mha = MultiHeadAttention(d_model=512, num_heads=8)
dummy_input = torch.randn(1, 10, 512) # batch=1, seq_len=10, dim=512
output, weights = mha(dummy_input, dummy_input, dummy_input)
print(f"输出形状: {output.shape}")
