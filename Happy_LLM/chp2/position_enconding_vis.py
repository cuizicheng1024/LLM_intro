import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
正余弦位置编码（Sinusoidal Positional Encoding）实现与可视化
- 作用：为序列中的每个 token 引入位置信息，使模型能够区分不同位置
- 公式核心（d_model 为偶数）：
  pe[pos, 2i]   = sin(pos / 10000^(2i/d_model))
  pe[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 1) 预分配位置编码矩阵（形状：max_len × d_model）
        pe = torch.zeros(max_len, d_model)
        
        # 2) 位置索引向量（0..max_len-1），并扩展维度为列向量 (max_len × 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3) 频率因子 div_term，对偶数维度计算 10000^(-2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # 4) 按照公式填充：偶数维用 sin，奇数维用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 5) 增加 batch 维度 -> (1 × max_len × d_model)，并注册为 buffer（不参与训练）
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 输入 x 形状: (batch_size × seq_len × d_model)
        # 取与当前序列长度匹配的前 seq_len 行位置编码，并与嵌入相加
        x = x + self.pe[:, :x.size(1)]
        return x

# 可视化：展示位置编码矩阵的热力图（行：位置 pos；列：特征维度 d_model）
pe_model = PositionalEncoding(d_model=128, max_len=100)
plt.figure(figsize=(8, 5))
plt.imshow(pe_model.pe[0].cpu().numpy(), cmap='RdBu')
plt.title("Positional Encoding Heatmap")
plt.xlabel("Dimension (d_model)")
plt.ylabel("Position (pos)")
plt.colorbar()
plt.savefig('./Happy_LLM/chp2/positional_encoding.png')

