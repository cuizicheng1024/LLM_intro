# LLM_intro Project

欢迎来到 LLM_intro 项目！本项目旨在通过深入浅出的理论讲解和代码实践，帮助你掌握大语言模型（LLM）的核心概念。

## 📋 前置知识要求

为了更好地理解本项目的内容，建议具备以下基础：
- **数学基础**：具备**线性代数**知识（矩阵运算、向量空间等是理解 Transformer 的关键）。
- **编程基础**：了解 Python 编程及 PyTorch 基础。

## 📂 项目目录结构说明

本项目主要包含以下几个部分的学习资料：

### 1. Transformer 深入浅出 (`Transformer介绍`)
这部分是《Transformer 架构：从直觉到实现》的完整教程，旨在通过直观的几何解释和代码实现，带你彻底搞懂 Transformer 架构。

*   **`Transformer介绍/`**
    *   包含从 "GPT 是什么" 到 "后 Transformer 架构" 的 30+ 章节 Markdown 文档。
    *   涵盖内容：
        *   **基础直觉**：GPT 发展史、大模型本质
        *   **核心组件**：Tokenization、Positional Encoding、LayerNorm、Softmax
        *   **Attention 机制**：线性变换几何意义、QKV 的本质、Multi-Head Attention
        *   **完整架构**：残差连接、前向传播、训练与推理
        *   **手写实现**：手写 Model.py, Train.py, Inference.py
        *   **进阶优化**：Flash Attention, KV Cache, LoRA, Quantization
    *   文档中包含大量可视化图片（存储于 `images/` 目录），帮助从几何角度理解复杂的数学原理。

### 2. HappyLLM 学习资料 (`Happy_LLM/chp1`, `Happy_LLM/chp2`)
这部分代码和文档侧重于 LLM 基础概念的代码实现与可视化。

*   **`Happy_LLM/chp1/` - NLP 基础概念**
    *   `第一章 NLP基础概念.md`: 理论笔记
    *   `day1_onehot.py`: One-Hot 编码实现
    *   `day1_tokenizer.py`: 分词器（Tokenizer）基础
    *   `day1_embeddding.py`: 词嵌入（Word Embedding）实现
    *   `day1_PCA_visual.py`: 词嵌入的 PCA 降维可视化
    *   `gpt_tokenization.py`: GPT 分词示例
    *   `word_embedding_pca.png`: 可视化结果图
    *   `glove.6B.100d.txt`: 预训练词向量文件（已在 `.gitignore` 中忽略，不会上传到 GitHub）

*   **`Happy_LLM/chp2/` - 神经网络核心组件**
    *   `softmax.py`: Softmax 函数实现与理解
    *   `layernorm.py`: Layer Normalization（层归一化）实现
    *   `position_encoding.py`: 位置编码示例实现
    *   `第二章 Transformer架构.md`: Transformer 架构学习笔记（来源于 happy-llm 课程）
## 🚀 如何开始

1.  **环境准备**：确保安装 Python 3.x 及 PyTorch，建议使用Trae等编辑器。
2.  **基础学习**：阅读 `Transformer介绍` 中的文档，配合 `Happy_LLM/chp2` 的代码理解核心组件。
3.  **进阶学习**：从 `Happy_LLM/chp1` 开始，运行代码理解 NLP 的基础输入表示。
4.  **动手实践**：跟随教程尝试手写 Transformer 模型。

enjoy Coding! 🚀
