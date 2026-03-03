# 第 20 章：手写 Inference.py

Original URL: https://www.waylandz.com/llm-transformer-book/第20章-手写Inference.py-推理逻辑

> **一句话总结**：推理就是：加载模型 → 输入 prompt → 自回归生成 → 解码输出。代码只有 30 行，但这是模型"开口说话"的时刻。

> 📦 **完整代码仓库**：[github.com/waylandzhang/Transformer-from-scratch](https://github.com/waylandzhang/Transformer-from-scratch)

---

## 20.1 推理 vs 训练

### 20.1.1 回顾第 16 章的对比

|  | 训练 | 推理 |
| --- | --- | --- |
| **目的** | 学习参数 | 生成文本 |
| **输入** | 完整序列 + 目标 | 只有 prompt |
| **输出** | 损失值 | 生成的文本 |
| **参数更新** | 是 | 否 |
| **Dropout** | 开启 | 关闭 |

### 20.1.2 推理的核心流程

```
1. 加载训练好的模型
2. 将 prompt 编码为 token IDs
3. 自回归生成（一个词一个词）
4. 解码回文本
```

---

## 20.2 加载模型

### 20.2.1 加载检查点

```
# 加载模型
import torch
import tiktoken
from model import Model

# 加载检查点
checkpoint = torch.load('model/model.ckpt')

# 从检查点恢复超参数
h_params = checkpoint['h_params']

# 重建模型
model = Model(h_params)

# 加载参数
model.load_state_dict(checkpoint['model_state_dict'])

# 切换到评估模式
model.eval()

# 移到正确的设备
model.to(h_params['device'])
```

### 20.2.2 为什么需要 `model.eval()`？

`model.eval()` 做了两件事：

1. **关闭 Dropout**：推理时不需要随机丢弃
2. **固定 BatchNorm**：使用训练时的统计量

不切换到 eval 模式，每次推理结果可能不一样！

---

## 20.3 准备输入

### 20.3.1 编码 prompt

```
# 编码输入
encoding = tiktoken.get_encoding("cl100k_base")

# 你想让模型续写什么？
start = "农夫山泉 "

# 编码为 token IDs
start_ids = encoding.encode(start)
print(f"Prompt: {start}")
print(f"Token IDs: {start_ids}")

# 转为 Tensor
x = torch.tensor(start_ids, dtype=torch.long, device=h_params['device'])
x = x.unsqueeze(0)  # 增加 batch 维度：[seq_len] → [1, seq_len]

print(f"Input shape: {x.shape}")
```

输出示例：

```
Prompt: 农夫山泉
Token IDs: [161, 253, 109, 26288, 239, 103]
Input shape: torch.Size([1, 6])
```

---

## 20.4 生成文本

### 20.4.1 调用生成函数

```
# 生成文本
with torch.no_grad():  # 不计算梯度
    y = model.generate(
        x,
        max_new_tokens=200,   # 最多生成 200 个 token
        temperature=0.5,       # 温度：越低越确定
        top_k=None            # 不使用 top-k
    )

# 解码
output_text = encoding.decode(y[0].tolist())

print('---------------')
print(output_text)
print('---------------')
```

### 20.4.2 生成结果示例

```
---------------
农夫山泉 天然水 550ml 瓶装
农夫山泉 东方树叶 茉莉花茶 500ml
农夫山泉 NFC 橙汁 300ml
农夫山泉 维他命水 柠檬味 500ml
---------------
```

模型学会了生成看起来像商品名称的文本！

---

## 20.5 生成参数详解

### 20.5.1 Temperature

```
y = model.generate(x, temperature=0.5)
```

Temperature 控制输出的"随机性"：

| Temperature | 效果 | 适用场景 |
| --- | --- | --- |
| 0.1-0.3 | 非常确定，重复性高 | 事实问答 |
| 0.5-0.7 | 平衡随机和确定 | 通用场景 |
| 0.8-1.0 | 较随机，多样性高 | 创意写作 |
| > 1.0 | 非常随机，可能不连贯 | 实验用 |

### 20.5.2 Top-K Sampling

```
y = model.generate(x, top_k=50)
```

只从概率最高的 K 个词中采样：

```
原始概率分布：
  "天" = 0.3, "矿" = 0.2, "冰" = 0.15, ...（100k 个词）

Top-K=3 后：
  "天" = 0.5, "矿" = 0.33, "冰" = 0.17
  （重新归一化到这 3 个词）
```

**好处**：避免采样到低概率的奇怪词。

### 20.5.3 Max New Tokens

```
y = model.generate(x, max_new_tokens=200)
```

控制生成长度：

* 太短：可能生成不完整
* 太长：浪费计算，可能产生重复

---

## 20.6 检查模型参数

### 20.6.1 打印参数量

```
# 统计参数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数量: {total_params:,}")
```

输出示例：

```
模型参数量: 8,234,560
```

### 20.6.2 查看每层参数

```
# 打印每层参数名和形状
for name, param in model.state_dict().items():
    print(f"{name}: {param.shape}")
```

输出示例：

```
token_embedding_lookup_table.weight: torch.Size([100256, 80])
transformer_blocks.0.ln1.weight: torch.Size([80])
transformer_blocks.0.ln1.bias: torch.Size([80])
transformer_blocks.0.mha.heads.0.Wq.weight: torch.Size([20, 80])
transformer_blocks.0.mha.heads.0.Wk.weight: torch.Size([20, 80])
transformer_blocks.0.mha.heads.0.Wv.weight: torch.Size([20, 80])
...
model_out_linear_layer.weight: torch.Size([100256, 80])
model_out_linear_layer.bias: torch.Size([100256])
```

---

## 20.7 完整 inference.py 代码

```
# -*- coding: utf-8 -*-
"""
Sample from a trained model
"""
import torch
import tiktoken
from model import Model

# 加载模型和超参数
checkpoint = torch.load('model/model.ckpt')
h_params = checkpoint['h_params']
model = Model(h_params)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(h_params['device'])

# 加载分词器
encoding = tiktoken.get_encoding("cl100k_base")

# 输入 prompt
start = "农夫山泉 "
start_ids = encoding.encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=h_params['device'])[None, ...]

# 生成
with torch.no_grad():
    y = model.generate(x, max_new_tokens=200, temperature=0.5, top_k=None)
    print('---------------')
    print(encoding.decode(y[0].tolist()))
    print('---------------')

# 打印模型参数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model param size: {total_params:,}")

# 打印模型参数
for name in model.state_dict().keys():
    print(name, model.state_dict()[name].shape)
```

---

## 20.8 不同 Prompt 的效果

### 20.8.1 尝试不同输入

```
# 尝试不同的 prompt
prompts = [
    "农夫山泉",
    "可口可乐",
    "奥利奥",
    "蒙牛"
]

for prompt in prompts:
    x = torch.tensor(encoding.encode(prompt), dtype=torch.long, device=h_params['device'])[None, ...]
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=50, temperature=0.5)
    print(f"Prompt: {prompt}")
    print(f"Output: {encoding.decode(y[0].tolist())}")
    print("---")
```

### 20.8.2 观察生成效果

模型会根据训练数据的模式来生成：

* 如果训练数据是商品名称，它会生成商品名称风格的文本
* 如果训练数据是小说，它会生成小说风格的文本
* 如果训练数据是代码，它会生成代码风格的文本

**模型学到的是数据中的模式，而不是"理解"内容。**

---

## 20.9 自回归生成的可视化

### 20.9.1 逐步生成过程

```
# 可视化生成过程
def generate_with_trace(model, x, max_new_tokens=10, temperature=1.0):
    """带追踪的生成"""
    encoding = tiktoken.get_encoding("cl100k_base")

    print(f"初始 prompt: {encoding.decode(x[0].tolist())}")
    print("---")

    for i in range(max_new_tokens):
        # 前向传播
        with torch.no_grad():
            logits, _ = model(x[:, -model.context_length:])

        # 获取最后位置的预测
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)

        # 获取 top-5 候选
        top5_probs, top5_ids = torch.topk(probs[0], 5)
        print(f"Step {i+1} 候选:")
        for prob, idx in zip(top5_probs, top5_ids):
            print(f"  '{encoding.decode([idx.item()])}': {prob.item():.3f}")

        # 采样
        idx_next = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, idx_next), dim=1)

        print(f"  → 选择: '{encoding.decode([idx_next[0].item()])}'")
        print(f"  当前序列: {encoding.decode(x[0].tolist())}")
        print("---")

    return x
```

### 20.9.2 输出示例

```
初始 prompt: 农夫山泉
---
Step 1 候选:
  '天': 0.312
  '矿': 0.198
  '有': 0.087
  '纯': 0.076
  '水': 0.065
  → 选择: '天'
  当前序列: 农夫山泉天
---
Step 2 候选:
  '然': 0.421
  '山': 0.156
  '地': 0.089
  '的': 0.067
  '下': 0.054
  → 选择: '然'
  当前序列: 农夫山泉天然
---
...
```

---

## 20.10 常见问题

### 20.10.1 生成重复内容

**问题**：模型不断重复相同的词或短语。

**原因**：

* Temperature 太低
* 训练数据本身有重复
* 模型过拟合

**解决**：

* 提高 Temperature
* 使用 Top-K 或 Top-P 采样
* 添加 repetition penalty

### 20.10.2 生成乱码

**问题**：输出是乱码或不连贯的文本。

**原因**：

* 模型训练不足
* prompt 不在训练分布内
* Temperature 太高

**解决**：

* 训练更多步
* 使用更合适的 prompt
* 降低 Temperature

### 20.10.3 速度太慢

**问题**：生成每个 token 都很慢。

**原因**：

* 没有使用 GPU
* 没有 KV Cache
* 模型太大

**解决**：

* 使用 GPU（如果有）
* 实现 KV Cache（第 22 章）
* 使用更小的模型

---

## 20.11 本章总结

### 20.11.1 推理三步曲

```
1. 加载模型
   checkpoint = torch.load('model.ckpt')
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()

2. 编码 prompt
   start_ids = encoding.encode(prompt)
   x = torch.tensor(start_ids)[None, ...]

3. 生成
   with torch.no_grad():
       y = model.generate(x, max_new_tokens=200)
   output = encoding.decode(y[0].tolist())
```

### 20.11.2 关键参数

| 参数 | 作用 | 建议值 |
| --- | --- | --- |
| `max_new_tokens` | 最大生成长度 | 50-500 |
| `temperature` | 随机性控制 | 0.5-0.8 |
| `top_k` | 限制候选词数量 | 50-100 |

### 20.11.3 核心认知

> **inference.py 只有 30 行代码，但它是我们整个旅程的终点——让模型真正"开口说话"。加载参数、编码 prompt、自回归生成、解码输出，这就是 GPT 推理的全部。理解了这些，你就理解了 ChatGPT 是如何回复你的每一句话的。**

---

## 本章交付物

学完这一章，你应该能够：

* 加载训练好的模型检查点
* 理解 `model.eval()` 的作用
* 使用不同的 Temperature 和 Top-K 参数
* 独立运行推理脚本

---

## Part 5 总结

恭喜！你已经完成了**代码实现**部分：

| 章节 | 内容 | 代码量 |
| --- | --- | --- |
| 第 18 章 | Model.py - 模型定义 | ~200 行 |
| 第 19 章 | Train.py - 训练循环 | ~100 行 |
| 第 20 章 | Inference.py - 推理逻辑 | ~30 行 |

**总共不到 400 行代码**，你就实现了一个完整的 Transformer！

这些代码虽然简化，但包含了真正 GPT 的核心逻辑。理解了这些，你就能读懂 Hugging Face transformers、LLaMA、GPT-NeoX 等开源项目的源码。

---

## 完整代码

Part 5 的完整实现可在 GitHub 获取：

> 📦 **[github.com/waylandzhang/Transformer-from-scratch](https://github.com/waylandzhang/Transformer-from-scratch)**

包含：

* `model.py` - 完整模型定义
* `train.py` - 训练脚本
* `inference.py` - 推理脚本
* `step-by-step.ipynb` - 逐步讲解的 Jupyter notebook

---

## 下一章预告

我们的模型能工作了，但速度不够快。每生成一个 token，都要重新计算整个序列的 Attention——太浪费了！

下一章，我们进入 **Part 6：生产优化**，学习 **Flash Attention** 和 **KV Cache**，让推理速度提升数倍！