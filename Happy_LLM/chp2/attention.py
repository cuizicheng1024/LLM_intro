"""
Scaled Dot-Product Attention（缩放点积注意力）
核心计算：
- scores = (Q @ K^T) / sqrt(d_k)
- weights = softmax(scores, dim=-1)
- output  = weights @ V
形状约定：
- Q, K, V: [batch, heads, seq, d_k]
- 可选 mask: 与 scores 同形状或可广播，0 表示屏蔽
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v 的形状均为: (batch_size, num_heads, seq_len, d_k)
    """
    
#  q：是你传入的查询（Query）张量，形状是 (batch_size, num_heads, seq_len_q, d_k)；
# .size()：PyTorch 张量的方法，返回张量各维度的大小（等价于 .shape，但 .size() 可以传索引，更灵活）；
# -1：倒数第一个维度的索引（Python 中负数索引表示从后往前数）。
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
