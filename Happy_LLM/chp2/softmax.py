# 代码示例
import torch
import torch.nn.functional as F

# 输入数据（logits）
logits = torch.tensor([3.01, 0.09, 2.48, 1.95])

# 应用 Softmax
probs = F.softmax(logits, dim=0)
print(probs)  # tensor([0.5028, 0.0271, 0.2959, 0.1742])
print(probs.sum())  # tensor(1.0000)