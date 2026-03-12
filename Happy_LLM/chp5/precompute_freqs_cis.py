import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算 RoPE 使用的复数旋转因子（cos + i·sin），用于长度为 end 的序列。
    
    参数:
    - dim: head_dim，每个注意力头向量的维度。
    - end: 最大序列长度（通常等于模型支持的 max_seq_len）。
    - theta: 基础角频率缩放常数，通常取 10000.0（与 Transformer 位置编码一致）。
    
    返回:
    - freqs_cis: 形状为 [end, dim // 2] 的复数张量。其中第 m 行对应该位置 m 的旋转因子。
    """
    # dim: 每个头的维度 (head_dim)
    # end: 最大长度 (max_seq_len)
    
    # 1) 计算不同维度上的基础频率系数（等比数列）
    #    索引从 0 开始，每隔 2 个取一个，共 dim // 2 个频率
    #    形状: (dim // 2,)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 2) 生成位置索引 m ∈ [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)  
    
    # 3) 外积计算角度矩阵: m * θ_i
    #    形状: (end, dim // 2)
    freqs = torch.outer(t, freqs).float()
    
    # 4) 将角度转为复数极坐标表示：cos(mθ) + i·sin(mθ)
    #    形状: (end, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
    return freqs_cis
