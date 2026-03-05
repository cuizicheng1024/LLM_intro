import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 1. 创建一个足够长的 position encoding 矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 2. 生成位置序列 [0, 1, 2, ..., max_len-1] 并增加一个维度变成 (max_len, 1)，拍平为 (max_len,)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. 计算公式中的分母部分 (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # 4. 填充 PE 矩阵：偶数维用 sin，奇数维用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 5. 增加 batch 维度 (1, max_len, d_model) 并注册为 buffer（不参与梯度下降）
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 形状: (batch_size, seq_len, d_model)
        # 将 PE 加到输入 Embedding 上（只取当前句子长度的部分）
        x = x + self.pe[:, :x.size(1)]
        return x

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

import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 典型的升维 -> 激活 -> 降维 结构
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        # 引用我们之前写好的模块
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 典型的 Pre-LN 结构
        # 1. Self-Attention 部分
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, mask)[0])
        # 2. Feed Forward 部分
        x2 = self.norm2(x)
        x = x + self.dropout(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.src_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # 1. Self-Attention 部分
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, tgt_mask)[0])
        # 2. Encoder-Decoder Attention 部分
        x2 = self.norm2(x)
        x = x + self.dropout(self.src_attn(x2, enc_out, enc_out, src_mask)[0])
        # 3. Feed Forward 部分
        x2 = self.norm3(x)
        x = x + self.dropout(self.ff(x2))
        return x



class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_len):
        super().__init__()
        # 1. 词嵌入与位置编码
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len) 
        
        # 2. 堆叠编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # 3. 堆叠解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # 4. 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # --- Encoder 阶段 ---
        # src shape: [batch, src_len]
        src = self.dropout(self.pe(self.src_embedding(src))) # [batch, src_len, d_model]
        
        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask) # [batch, src_len, d_model]
            
        # --- Decoder 阶段 ---
        # tgt shape: [batch, tgt_len]
        # memory 就是 encoder 的输出: [batch, src_len, d_model]
        tgt = self.dropout(self.pe(self.tgt_embedding(tgt))) # [batch, tgt_len, d_model]
        
        dec_output = tgt
        for layer in self.decoder_layers:
            # Decoder 同时看自己的 tgt 和 Encoder 的输出
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask) # [batch, tgt_len, d_model]
            
        # --- 输出映射 ---
        logits = self.fc_out(dec_output) # [batch, tgt_len, tgt_vocab_size]
        return logits

# --- 冒烟测试 (Smoke Test) ---
def smoke_test():
    # 参数设置
    params = {
        'src_vocab_size': 100,
        'tgt_vocab_size': 100,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 512,
        'dropout': 0.1,
        'max_len': 50
    }
    
    model = Transformer(**params)
    
    # 模拟输入数据
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    src = torch.randint(0, 100, (batch_size, src_len))
    tgt = torch.randint(0, 100, (batch_size, tgt_len))
    
    # 模拟掩码 (先设为 None 或 全1)
    src_mask = None 
    tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(batch_size, 1, tgt_len, tgt_len)
    
    # 运行
    output = model(src, tgt, src_mask, tgt_mask)
    
    print(f"输入 Src 形状: {src.shape}")
    print(f"输入 Tgt 形状: {tgt.shape}")
    print(f"模型输出形状: {output.shape}") 
    
    if output.shape == (batch_size, tgt_len, params['tgt_vocab_size']):
        print("✅ 冒烟测试通过：维度完全正确！")
    else:
        print("❌ 维度不匹配，请检查代码。")

smoke_test()
