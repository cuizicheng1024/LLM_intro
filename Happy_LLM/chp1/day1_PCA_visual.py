import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim
import numpy as np

# Word Embedding Visualization (词嵌入可视化) 演示
# 词向量通常是高维的（如 100维, 300维），无法直接在二维屏幕上展示。
# 本示例使用 PCA (主成分分析) 技术将高维向量降维到 2D 平面进行可视化，
# 以直观观察词与词之间的聚类关系。

def load_glove_vectors(glove_file_path):
    """
    加载 GloVe 词向量文件
    :param glove_file_path: GloVe 文件路径（如 glove.6B.100d.txt）
    :return: gensim的KeyedVectors对象（可直接调用词向量）
    """
    print(f"正在加载 GloVe 向量文件: {glove_file_path} ...")
    # 初始化词向量字典
    word_vectors = {}
    try:
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 拆分每行：词 + 向量值
                parts = line.strip().split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                word_vectors[word] = vector
        
        # 转换为Gensim的KeyedVectors（方便后续操作）
        vocab_size = len(word_vectors)
        if vocab_size == 0:
            return None

        vector_dim = len(next(iter(word_vectors.values())))
        kv = gensim.models.KeyedVectors(vector_dim)
        kv.add_vectors(list(word_vectors.keys()), list(word_vectors.values()))
        print(f"加载完成，共 {vocab_size} 个词向量。")
        return kv
    except FileNotFoundError:
        print(f"[错误] 找不到文件: {glove_file_path}")
        return None

# ==========================================
# 主程序逻辑
# ==========================================

# 1. 加载 GloVe 词向量
# 需先下载: https://nlp.stanford.edu/projects/glove/
glove_path = "./chp1/glove.6B.100d.txt"  # 替换为你的GloVe文件路径
wv = load_glove_vectors(glove_path)

if wv:
    # 2. 定义要可视化的词列表
    # 选择几组具有明显语义类别的词，观察它们在空间中的分布
    # - 皇室/人物类: king, queen, prince, princess
    # - 水果类: apple, banana, orange, fruit
    # - 科技类: computer, laptop, software, keyboard
    words_to_plot = [
        "king", "queen", "prince", "princess", 
        "apple", "banana", "orange", "fruit", "tree","water",
        "computer", "laptop", "software", "keyboard", 
        "mouse","cat","dog",
    ]

    # 3. 提取这些词的高维向量
    # 列表推导式: 遍历每个词，从 wv 中取出对应的向量
    # 结果是一个 numpy 数组，形状为 (词数, 向量维度)
    # 注意：如果词表中不存在某个词，这里会报错，实际使用需做检查
    word_vectors_list = []
    valid_words = []
    for w in words_to_plot:
        if w in wv:
            word_vectors_list.append(wv[w])
            valid_words.append(w)
        else:
            print(f"警告: 词 '{w}' 不在词表中，已跳过。")
            
    word_vectors = np.array(word_vectors_list)

    if len(word_vectors) > 0:
        # 4. 使用 PCA 进行降维
        # PCA (Principal Component Analysis) 是一种常用的线性降维算法
        # n_components=2: 将数据降维到 2 维
        pca = PCA(n_components=2)
        
        # fit_transform: 先拟合数据（计算主成分），再进行转换
        # 输出 coords 的形状为 (词数, 2)
        coords = pca.fit_transform(word_vectors)

        # 5. 绘图 (Matplotlib)
        plt.figure(figsize=(10, 8))
        
        # 绘制散点图
        # x 轴: coords[:, 0] (第一主成分)
        # y 轴: coords[:, 1] (第二主成分)
        plt.scatter(coords[:, 0], coords[:, 1], edgecolors='k', c='r', s=100)

        # 为每个点添加文本标签
        for i, word in enumerate(valid_words):
            # xy: 点的坐标
            # xytext: 文本的偏移位置 (可选)
            plt.annotate(word, xy=(coords[i, 0], coords[i, 1]), xytext=(5, 2), 
                         textcoords='offset points', size=12)

        plt.title("Word Embedding 2D Visualization (PCA)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 保存图片到本地
        output_file = "./chp1/word_embedding_pca.png"
        plt.savefig(output_file)
        print(f"已完成绘制。")
        print(f"可视化图表已保存至: {output_file}")
        
        # 显示图表 (如果在支持 GUI 的环境中运行)
        # plt.show() 
    else:
        print("没有有效的词向量可供绘图。")
else:
    print("词向量加载失败，请检查文件路径。")
