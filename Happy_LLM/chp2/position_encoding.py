"""
Sinusoidal 位置编码示例
- 经典公式（d_model 偶数）：对偶数/奇数维分别用 sin/cos
- P[k, 2i]   = sin(k / n^(2i/d_model))
- P[k, 2i+1] = cos(k / n^(2i/d_model))
参数:
- seq_len: 序列长度（token 数）
- d_model: 特征维度（模型隐藏维度，通常与嵌入维度一致）
- n: 缩放基数（常用 10000）
返回:
- P: [seq_len, d_model] 的位置编码矩阵
"""
import numpy as np
import matplotlib.pyplot as plt
def PositionEncoding(seq_len, d_model, n=10000):
    P = np.zeros((seq_len, d_model))
    for k in range(seq_len):
        for i in np.arange(int(d_model/2)):
            denominator = np.power(n, 2*i/d_model)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

# 示例：生成一个较小维度的编码便于直观打印
P = PositionEncoding(seq_len=4, d_model=4, n=50)
print(P)
