# LLM_intro Project

欢迎来到 LLM_intro 项目！本项目旨在通过深入浅出的理论讲解和代码实践，帮助你掌握大语言模型（LLM）的核心概念。

## 📋 前置知识要求

为了更好地理解本项目的内容，建议具备以下基础：
- **数学基础**：具备**线性代数**知识（矩阵运算、向量空间等是理解 Transformer 的关键）。
- **编程基础**：了解 Python 编程。

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
    *   文档目录：
        - [前言](Transformer介绍/前言.md)
        - [第 1 章：GPT 是什么](Transformer介绍/第%201%20章：GPT%20是什么.md)
        - [第 2 章：大模型的本质](Transformer介绍/第%202%20章：大模型的本质.md)
        - [第 3 章：Transformer 全景图](Transformer介绍/第%203%20章：Transformer%20全景图.md)
        - [第 4 章：Tokenization](Transformer介绍/第%204%20章：Tokenization.md)
        - [第 5 章：Positional Encoding](Transformer介绍/第%205%20章：Positional%20Encoding.md)
        - [第 6 章：LayerNorm 与 Softmax](Transformer介绍/第%206%20章：LayerNorm%20与%20Softmax.md)
        - [第 7 章：神经网络层](Transformer介绍/第%207%20章：神经网络层.md)
        - [第 8 章：线性变换的几何意义](Transformer介绍/第%208%20章：线性变换的几何意义.md)
        - [第 9 章：Attention 的几何逻辑](Transformer介绍/第%209%20章：Attention%20的几何逻辑.md)
        - [第 10 章：QKV 到底是什么](Transformer介绍/第%2010%20章：QKV%20到底是什么.md)
        - [第 11 章：Multi-Head Attention](Transformer介绍/第%2011%20章：Multi-Head%20Attention.md)
        - [第 12 章：QKV 输出的本质](Transformer介绍/第%2012%20章：QKV%20输出的本质.md)
        - [第 13 章：残差连接与 Dropout](Transformer介绍/第%2013%20章：残差连接与%20Dropout.md)
        - [第 14 章：词嵌入与位置信息](Transformer介绍/第%2014%20章：词嵌入与位置信息.md)
        - [第 15 章：完整前向传播](Transformer介绍/第%2015%20章：完整前向传播.md)
        - [第 16 章：训练与推理的异同](Transformer介绍/第%2016%20章：训练与推理的异同.md)
        - [第 17 章：学习率的理解](Transformer介绍/第%2017%20章：学习率的理解.md)
        - [第 18 章：手写 Model.py](Transformer介绍/第%2018%20章：手写%20Model.py.md)
        - [第 19 章：手写 Train.py](Transformer介绍/第%2019%20章：手写%20Train.py.md)
        - [第 20 章：手写 Inference.py](Transformer介绍/第%2020%20章：手写%20Inference.py.md)
        - [第 21 章：Flash Attention](Transformer介绍/第%2021%20章：Flash%20Attention.md)
        - [第 22 章：KV Cache](Transformer介绍/第%2022%20章：KV%20Cache.md)
        - [第 23 章：MHA 到 MQA 到 GQA](Transformer介绍/第%2023%20章：MHA%20到%20MQA%20到%20GQA.md)
        - [第 24 章：Sparse 与 Infinite Attention](Transformer介绍/第%2024%20章：Sparse%20与%20Infinite%20Attention.md)
        - [第 25 章：位置编码演进](Transformer介绍/第%2025%20章：位置编码演进.md)
        - [第 26 章：LoRA 与 QLoRA](Transformer介绍/第%2026%20章：LoRA%20与%20QLoRA.md)
        - [第 27 章：模型量化](Transformer介绍/第%2027%20章：模型量化.md)
        - [第 28 章：Prompt Engineering](Transformer介绍/第%2028%20章：Prompt%20Engineering.md)
        - [第 29 章：RLHF 与偏好学习](Transformer介绍/第%2029%20章：RLHF%20与偏好学习.md)
        - [第 30 章：Mixture of Experts](Transformer介绍/第%2030%20章：Mixture%20of%20Experts.md)
        - [第 31 章：推理模型革命](Transformer介绍/第%2031%20章：推理模型革命.md)
        - [第 32 章：后 Transformer 架构](Transformer介绍/第%2032%20章：后%20Transformer%20架构.md)
        - [附录 A：Scaling Law](Transformer介绍/附录%20A：Scaling%20Law.md)
        - [附录 B：解码策略详解](Transformer介绍/附录%20B：解码策略详解.md)
        - [附录 C：常见问题 FAQ](Transformer介绍/附录%20C：常见问题%20FAQ.md)

### 2. HappyLLM 学习资料 (`Happy_LLM/chp1`, `Happy_LLM/chp2`)
*   **`Happy_LLM/chp1/` - NLP 基础概念**
    *   文档目录：
        - [第一章 NLP基础概念](Happy_LLM/chp1/第一章%20NLP基础概念.md)

*   **`Happy_LLM/chp2/` - 神经网络核心组件**
    *   文档目录：
        - [第二章 Transformer架构](Happy_LLM/chp2/第二章%20Transformer架构.md)
## 🚀 如何开始

1.  **环境准备**：确保安装 Python 3.x 及 PyTorch，建议使用Trae等编辑器。
2.  **基础学习**：阅读 `Transformer介绍` 中的文档，配合 `Happy_LLM/chp2` 的代码理解核心组件。
3.  **进阶学习**：从 `Happy_LLM/chp1` 开始，运行代码理解 NLP 的基础输入表示。
4.  **动手实践**：跟随教程尝试手写 Transformer 模型。

Enjoy Coding! 🚀
