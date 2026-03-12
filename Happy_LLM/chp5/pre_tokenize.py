import numpy as np
import sentencepiece as spm
from tqdm import tqdm

def pre_tokenize():
    """
    使用训练好的 SentencePiece 模型对原始文本进行分词并保存为二进制 ID 序列。
    
    流程:
    1) 加载训练好的分词模型（my_tokenizer.model）；
    2) 逐行读取原始语料，对每行进行分词并转为 ID；
    3) 为每行追加 EOS（句子结束符）以便模型学习句边界；
    4) 将所有 ID 拼接为一个一维数组并保存为 train.bin。
    """
    # 1. 加载我们在 Phase 3 训练好的模型
    sp = spm.SentencePieceProcessor(model_file='my_tokenizer.model')
    
    # 2. 读取原始语料 (假设是一个很大的 txt)
    with open("train.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    all_ids = []
    for line in tqdm(lines, desc="Tokenizing"):
        # 分词并转为 ID（返回的是 Python 列表）
        ids = sp.encode_as_ids(line)
        # 加上句子结束符 (EOS)：帮助模型建模句子边界与拼接
        ids.append(sp.eos_id())
        all_ids.extend(ids)
    
    # 3. 转换为 Numpy 数组并保存为二进制文件
    #    选择 uint16 的前提是词表大小 < 65535；若词表更大需改为 uint32。
    all_ids = np.array(all_ids, dtype=np.uint16) # 词表<65535用uint16即可
    all_ids.tofile("train.bin")
    print(f"\n预处理完成！总共 {len(all_ids)} 个 Tokens 已存入 train.bin")

pre_tokenize()
