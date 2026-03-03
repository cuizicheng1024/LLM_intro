"""
Softmax 函数示例
- 输入通常是未归一化的分数（logits），Softmax 将其映射为概率分布
- 归一化维度 dim 的选择非常重要：对一维向量用 dim=0；对二维矩阵按行或列选择  
- 数值特性：输出非负，且在指定维度上求和为 1
"""
import torch
import torch.nn.functional as F

# 输入数据（logits）：未归一化的类别分数
logits = torch.tensor([3.01, 0.09, 2.48, 1.95])

# 应用 Softmax（在向量维度上归一化）
# 典型选择：对一维向量使用 dim=0；对形如 [batch, num_classes] 的张量使用 dim=1
probs = F.softmax(logits, dim=0)

# 输出为概率分布（每个元素 ∈ [0,1]，总和为 1）
print(probs)        # tensor([0.5028, 0.0271, 0.2959, 0.1742])
print(probs.sum())  # tensor(1.0000)

# 直觉说明：
# - Softmax 对分数进行“相对比较”，较大的分数得到更高概率
# - 分母是所有分数的指数和，确保归一化
# - 若需要控制“平滑程度”，可以配合温度系数 T：softmax(logits / T)
