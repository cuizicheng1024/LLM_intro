import numpy as np

# One-Hot 编码 (独热编码) 是一种最简单的文本表示方法。
# 核心思想：
# 1. 建立一个包含语料库中所有唯一词的词表 (Vocabulary)。
# 2. 每个词用一个长度为词表大小 (V) 的向量表示。
# 3. 该向量中，仅在该词对应的索引位置为 1，其余位置均为 0。

# 1. 原始数据 (Toy Corpus)
sentences = [
    "I love AI",
    "I love NLP",
    "AI is cool"
]

# 2. 构建词表 (Vocabulary)
# 第一步：分词并去重
words = set()
for s in sentences:
    # 简单的按空格分词 (split())
    for word in s.split():
        words.add(word)

# 第二步：排序，确保词表的顺序是固定的（确定性）
# 这样每次运行生成的索引映射都是一致的
vocab = sorted(list(words))

# 第三步：建立词到索引的映射 (Word to Index)
# 例如: {'AI': 0, 'I': 1, 'NLP': 2, ...}
word_to_idx = {word: i for i, word in enumerate(vocab)}

print("词表 (Vocabulary Size = {}):".format(len(vocab)), vocab)
print("词到索引的映射:", word_to_idx)

# 3. 生成 One-hot 向量函数
def get_one_hot(word):
    """
    输入一个词，返回其对应的 One-hot 向量。
    """
    # 初始化一个全零向量，长度等于词表大小
    vector = np.zeros(len(vocab))
    
    # 如果词在词表中，将对应索引位置设为 1
    if word in word_to_idx:
        index = word_to_idx[word]
        vector[index] = 1
    # 如果词不在词表中（OOV - Out of Vocabulary），通常返回全零向量或特殊标记向量
    
    return vector

# 4. 测试与观察
word1 = "AI"
word2 = "NLP"

v1 = get_one_hot(word1)
v2 = get_one_hot(word2)

print(f"\n'{word1}' 的向量:\n{v1}")
print(f"\n'{word2}' 的向量:\n{v2}")

# 计算两个向量的点积 (Dot Product)
# 在几何意义上，点积可以反映两个向量的相似程度（未归一化的余弦相似度）
# 对于 One-hot 向量：
# - 如果两个词不同，它们的 '1' 位于不同位置，点积为 0 (正交)。
# - 如果两个词相同，点积为 1。
dot_product = np.dot(v1, v2)
print(f"\n两个向量的点积 (相似度): {dot_product}") 

# 结论：
# One-hot 编码的主要缺点是无法表示词与词之间的语义相似性。
# "AI" 和 "NLP" 虽然语义相关，但它们的 One-hot 向量是正交的 (相似度为 0)。
