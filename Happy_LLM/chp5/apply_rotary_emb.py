import torch

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    将 RoPE（Rotary Positional Embedding）应用到注意力的 query/key 上。
    
    参数:
    - x: 张量，形状为 [batch, seq_len, n_heads, head_dim]。通常是多头注意力中的 q 或 k。
    - freqs_cis: 预先计算好的复数形式的旋转因子，形状为 [seq_len, head_dim // 2]，
                 其中每一行对应某个位置（token index）的余弦/正弦对。
    
    返回:
    - 与 x 相同 dtype 的张量，形状与 x 相同。表示对最后一维做旋转后的结果。
    
    说明:
    - 实现思路是将最后一维（head_dim）按（实部、虚部）两两配对映射为复数，在复平面上乘以 e^{iθ}
      实现旋转，最后再还原回实数拼接的形式。
    """
    # x 的形状应为: [batch, seq_len, n_heads, head_dim]
    # freqs_cis 的形状应为: [seq_len, head_dim // 2]
    
    # 1) 将 x 的最后一个维度两两配对为复数的 (real, imag)
    #    例如 head_dim=64，则变为 32 对复数。为了使用 view_as_complex，需要先转 float（避免半精度不被支持），
    #    再用 reshape(..., -1, 2) 把最后维度拆成两列，最后交给 view_as_complex 变为复表示。
    #    变换后形状: [batch, seq_len, n_heads, head_dim // 2]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # 2) 调整 freqs_cis 的维度，方便与 x_complex 做广播相乘
    #    [seq_len, head_dim // 2] -> [1, seq_len, 1, head_dim // 2]
    #    这样 batch 和 n_heads 维度会通过广播共享同一个旋转因子。
    freqs_cis = freqs_cis.view(1, x.shape[1], 1, x_complex.shape[-1])
    
    # 3) 在复平面上做旋转：z' = z * (cosθ + i·sinθ)
    #    这里 freqs_cis 已经是 torch.polar 得到的复数形式。
    x_rotated = x_complex * freqs_cis
    
    # 4) 将复数结果还原回实数张量：
    #    view_as_real -> 在最后一维拆成 (real, imag)，然后 flatten 合并回 head_dim
    #    形状: [batch, seq_len, n_heads, head_dim // 2, 2] -> [batch, seq_len, n_heads, head_dim]
    out = torch.view_as_real(x_rotated).flatten(3)
    # 保持与输入 x 相同的数据类型（可能是半精度/混合精度）
    return out.type_as(x)
