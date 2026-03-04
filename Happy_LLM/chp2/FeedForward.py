"""
Transformer 前馈网络 (Position-wise FeedForward) 模块
- 结构：线性升维 -> 激活 -> Dropout -> 线性降维
- 作用：对每个位置独立地进行非线性变换，提升表示能力
"""
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 典型结构：d_model -> d_ff -> d_model
        # 激活常见选择：ReLU/GELU；d_ff 通常为 2~4 倍 d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # 输入输出形状一致：[batch, seq, d_model]
        print(f"FeedForward input shape: {x.shape}")
        return self.net(x)
