"""
Layer Normalization 示例
- 对最后一个维度进行标准化：y = (x - mean) / sqrt(var + eps) * gamma + beta
- 相比 BatchNorm，不依赖 batch 统计量，适合 NLP 等可变序列长度场景
"""
import torch
import torch.nn as nn

# 创建 LayerNorm 层
# normalized_shape 指定要归一化的维度大小（通常是特征维度 d_model）
# bias=True 表示同时学习平移参数 beta（若为 False，则仅有缩放 gamma）
layer_norm = nn.LayerNorm(normalized_shape=4, bias=True)

# 输入数据（形状：[batch, seq_len=1, feature=4] 的简化示例，这里用二维张量演示）
x = torch.tensor([[22.0, 5.0, 6.0, 8.0]])

# 应用 LayerNorm
# 计算步骤：
# 1) 计算最后维度的均值与方差
# 2) 标准化并进行仿射变换（学习到的 gamma、beta）
y = layer_norm(x)
print(y)  # 输出接近 [1.71, -0.76, -0.62, -0.33]
