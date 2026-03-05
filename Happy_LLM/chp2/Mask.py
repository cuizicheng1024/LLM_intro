import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
自回归（因果）注意力掩码可视化

- 目的：展示 Decoder 中常用的 Look-ahead/Causal Mask（下三角矩阵），
        保证位置 i 只能看到自己及之前的位置，防止“偷看”未来词。
- 实现：torch.tril(ones) 生成下三角 0/1 掩码；1 表示保留，0 表示屏蔽（在注意力得分中会被置为极小值）
- 可视化：使用 matplotlib 绘制热力图，并在每个格子标注 Keep/Mask 便于直观理解
"""
import matplotlib.pyplot as plt
import torch

def visualize_causal_mask(size):
    # 生成下三角掩码矩阵，形状为 [size, size]
    # 行表示 Query 位置，列表示 Key 位置
    # 例如 size=5 时，第 3 行（位置 3 的 Query）只能看到列 0..3（含自己）
    mask = torch.tril(torch.ones(size, size))
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='Blues')
    plt.title("Look-ahead (Causal) Mask")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    
    # 在每个格子上添加文本：1 为 Keep（可见），0 为 Mask（不可见）
    for i in range(size):
        for j in range(size):
            text = "Keep" if mask[i, j] == 1 else "Mask"
            plt.text(j, i, text, ha="center", va="center", color="black" if mask[i,j] == 1 else "grey")
            
    # 注意：在脚本环境下 plt.show() 可能弹出窗口；如在无图形界面的环境，可注释掉
    # plt.show()
    plt.savefig("./Happy_LLM/chp2/causal_mask.png")

visualize_causal_mask(size=5)
