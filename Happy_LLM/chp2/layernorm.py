# 代码示例
import torch
import torch.nn as nn

# 创建 LayerNorm 层
layer_norm = nn.LayerNorm(normalized_shape=4, bias=True)

# 输入数据
x = torch.tensor([[22.0, 5.0, 6.0, 8.0]])

# 应用 LayerNorm
y = layer_norm(x)
print(y)  # 输出接近 [1.71, -0.76, -0.62, -0.33]