import gensim
import numpy as np

# Word Embedding (词嵌入) 演示
# 词嵌入将词语映射到低维稠密向量空间，能够捕捉词与词之间的语义关系。
# 本示例演示如何手动加载 GloVe 预训练词向量并进行操作。

def load_glove_vectors(glove_file_path):
    """
    加载 GloVe 格式的词向量文件。
    
    GloVe (Global Vectors for Word Representation) 文件通常是文本格式，
    每行包含一个词及其对应的向量值，以空格分隔。
    例如: "the 0.418 0.24968 -0.41242 ..."
    
    :param glove_file_path: GloVe 文件路径（如 glove.6B.100d.txt）
    :return: gensim的KeyedVectors对象（提供 most_similar 等便捷方法）
    """
    print(f"正在加载 GloVe 向量文件: {glove_file_path} ...")
    
    # 1. 读取文件并将词向量存入字典
    word_vectors = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # strip() 去除首尾空白符，split() 按空格切分每行
            parts = line.strip().split()
            
            # 第一个元素是词 (Key)
            word = parts[0]
            
            # 剩余元素是向量的分量 (Values)，转换为 float32 类型的 numpy 数组
            # 使用 float32 可以节省内存
            vector = np.array(parts[1:], dtype=np.float32)
            
            word_vectors[word] = vector
    
    # 2. 转换为 Gensim 的 KeyedVectors 对象
    # KeyedVectors 是 Gensim 库中用于高效存储和检索词向量的类
    vocab_size = len(word_vectors)
    if vocab_size == 0:
        return None
        
    # 获取向量维度 (通常是 50, 100, 200, 300 等)
    vector_dim = len(next(iter(word_vectors.values())))
    
    # 初始化 KeyedVectors 容器
    kv = gensim.models.KeyedVectors(vector_dim)
    
    # 批量添加向量数据 (Keys 和 Weights)
    kv.add_vectors(list(word_vectors.keys()), list(word_vectors.values()))
    
    print(f"加载完成，共 {vocab_size} 个词向量，维度为 {vector_dim}。")
    return kv

# ==========================================
# 主程序逻辑
# ==========================================

# 1. 设置 GloVe 文件路径
# 注意: 运行此代码前，需下载 GloVe 预训练向量文件
# 下载地址: https://nlp.stanford.edu/projects/glove/
# 建议下载 glove.6B.zip 并解压，使用其中的 glove.6B.100d.txt
glove_path = "./chp1/glove.6B.100d.txt" 

try:
    # 尝试加载词向量
    wv = load_glove_vectors(glove_path)

    if wv:
        # 2. 基础测试：获取并打印单个词的向量
        # 词向量是一个稠密向量 (Dense Vector)
        print("\n'king' 的词向量前 5 维数据：", wv["king"][:5])
        
        # 3. 语义相似度测试：寻找最相似的词
        # 原理：计算余弦相似度 (Cosine Similarity)
        # 两个向量夹角越小（余弦值越接近 1），语义越相似
        print("\n与 'king' 最相似的 5 个词：")
        # most_similar 返回 (词, 相似度分数) 的列表
        for word, score in wv.most_similar("king", topn=5):
            print(f"  {word}: {score:.4f}")

        # 4. 著名的类比推理 (Analogy Task)
        # 验证公式: King - Man + Woman ≈ Queen
        # 意义：词向量空间具有线性子结构，可以通过向量加减法来推理语义关系
        print("\n类比推理测试: King - Man + Woman = ?")
        # positive=['king', 'woman']: 加上这两个词的向量
        # negative=['man']: 减去这个词的向量
        result = wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
        
        # 打印结果 (预期结果应该是 'queen')
        print(f"计算结果: {result[0][0]} (相似度: {result[0][1]:.4f})")
        
except FileNotFoundError:
    print(f"\n[错误] 找不到文件: {glove_path}")
    print("请确保已下载 glove.6B.100d.txt 并放置在代码同一目录下。")
except Exception as e:
    print(f"\n[错误] 发生异常: {e}")
