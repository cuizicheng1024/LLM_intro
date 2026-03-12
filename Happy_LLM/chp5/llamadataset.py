import torch
from torch.utils.data import Dataset
import numpy as np

class LlamaDataset(Dataset):
    """
    将预分词后的连续 ID 序列（train.bin）按滑动窗口切分为 (x, y) 训练样本的 Dataset。
    
    约定:
    - 数据文件是使用 np.tofile 写出的连续 uint16 ID（或其它 dtype）；
    - 每个样本长度为 max_seq_len，标签 y 为 x 右移一位（因果语言建模）。
    """
    def __init__(self, bin_file, max_seq_len):
        self.max_seq_len = max_seq_len
        # 使用内存映射加载二进制文件，避免一次性读入占用大量内存
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        
    def __len__(self):
        # 减去 max_seq_len 和 1 是为了保证 __getitem__ 在取 (x, y) 时不越界
        return len(self.data) - self.max_seq_len - 1

    def __getitem__(self, index):
        # 1) 连续取出长度为 max_seq_len + 1 的片段
        #    之所以多取 1，是为了构造右移一位的 (x, y) 对
        chunk = self.data[index : index + self.max_seq_len + 1]
        # PyTorch 的 Embedding/损失一般期望 int64（long），因此做类型转换
        chunk = chunk.astype(np.int64) # 转为 PyTorch 需要的 int64
        
        # 2) 构造训练对
        #    x: 输入序列 [t0, t1, ..., t_{n-1}]
        #    y: 目标序列 [t1, t2, ..., t_n] (即 x 右移一位)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        
        return x, y

# --- 实验验证 ---
dataset = LlamaDataset("train.bin", max_seq_len=128)
x, y = dataset[0]

print(f"输入 x 的前 5 个 ID: {x[:5]}")
print(f"标签 y 的前 5 个 ID: {y[:5]}")
print(f"x 和 y 的形状是否一致: {x.shape == y.shape}")
