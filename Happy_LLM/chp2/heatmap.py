"""
可视化 Transformer 注意力权重的热力图
- 依赖：transformers, seaborn, matplotlib, torch
- 流程：加载模型 -> 生成注意力 -> 提取某层某头矩阵 -> 绘制热力图并高亮关键 token
"""
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# -------------------------- 1. 配置模型和输入 --------------------------
# 选择轻量级预训练模型（DistilBERT，速度快，注意力机制完整）
model_name = "distilbert-base-uncased"
# 目标句子
input_text = "The animal didn't cross the street because it was too tired"

# 加载分词器和模型（启用返回注意力权重）
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)  # 关键：output_attentions=True

# -------------------------- 2. 预处理输入 --------------------------
# 分词并转换为模型输入格式
inputs = tokenizer(
    input_text,
    return_tensors="pt",  # 返回PyTorch张量
    add_special_tokens=True  # 添加[CLS]/[SEP]等特殊token
)
# 获取token列表（用于热力图标注）
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# -------------------------- 3. 前向传播获取注意力权重 --------------------------
# 禁用梯度计算（仅推理，加快速度）
with torch.no_grad():
    outputs = model(**inputs)

# 提取注意力权重：attentions是tuple，每个元素是 [batch, num_heads, seq_len, seq_len]
# 取最后一层、第一个头的注意力矩阵（可调整层/头，比如layers[-2]取倒数第二层）
attention = outputs.attentions[-1]  # 最后一层注意力
attention_matrix = attention[0][0].cpu().numpy()  # 取第0个batch、第0个头，转numpy

# -------------------------- 4. 绘制注意力热力图 --------------------------
# 设置画布大小
plt.figure(figsize=(24, 20))
# 绘制热力图（cmap选冷暖色，annot=True显示数值）
ax = sns.heatmap(
    attention_matrix,
    annot=True,  # 显示每个格子的注意力值
    fmt=".4f",   # 数值保留2位小数
    cmap="YlOrRd",  # 配色方案（黄→橙→红，值越高越红）
    xticklabels=tokens,  # x轴标注为token
    yticklabels=tokens   # y轴标注为token
)
# 设置标题和轴标签
plt.title("Attention Matrix Heatmap (Last Layer, Head 0)", fontsize=14)
plt.xlabel("Key Tokens", fontsize=12)
plt.ylabel("Query Tokens", fontsize=12)
# 旋转x轴标签，避免重叠
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

# -------------------------- 5. 标注关键token（it和animal） --------------------------
# 找到"it"和"animal"的token索引
it_idx = tokens.index("it")
animal_idx = tokens.index("animal")
# 标注"it"（query）对"animal"（key）的注意力格子
ax.add_patch(
    plt.Rectangle((animal_idx, it_idx), 1, 1, fill=False, edgecolor="blue", linewidth=3)
)
# it → animal
plt.text(
    animal_idx + 0.5, it_idx + 1.5, 
    "", 
    ha="center", va="center", 
    color="blue", fontweight="bold"
)

# 保存图片（可选）
plt.tight_layout()  # 自动调整布局
output_path = Path(__file__).resolve().parent / "attention_heatmap.png"
plt.savefig(output_path, dpi=300)
plt.show()

# -------------------------- 6. 输出关键注意力值 --------------------------
print(f"Token列表: {tokens}")
print(f"'it'（索引{it_idx}）对'animal'（索引{animal_idx}）的注意力值: {attention_matrix[it_idx][animal_idx]:.4f}")
# 对比："it"对"street"的注意力值（作为参考）
street_idx = tokens.index("street")
print(f"'it'（索引{it_idx}）对'street'（索引{street_idx}）的注意力值: {attention_matrix[it_idx][street_idx]:.4f}")
