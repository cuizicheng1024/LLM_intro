import sentencepiece as sp

# 本脚本演示如何使用 SentencePiece 训练一个简化的 BPE 分词器并进行基本验证。
# 流程：
# 1) 准备一份中英文混合的示例语料，写入 train.txt；
# 2) 使用 SentencePieceTrainer 训练 BPE 模型（得到 my_tokenizer.model/.vocab）；
# 3) 使用训练好的模型进行编码、分词和解码测试，观察效果。

# 模拟一个简单的中英文混合语料（为了演示，重复多次扩大体量）
data = """
I love deep learning. 我爱深度学习。
Large language models are amazing. 大语言模型太神奇了。
Transformer is the core of LLaMA. Transformer 是 LLaMA 的核心。
""" * 1000  # 复制多次模拟大数据量

with open("./train.txt", "w", encoding="utf-8") as f:
    f.write(data)


# 训练 BPE 模型
# 说明：
# - vocab_size：实际项目常用 32k 或更大，这里缩小以加快演示；
# - character_coverage：对中英文混合语料常设为 0.9995，尽量覆盖字符集；
# - byte_fallback：遇到未登录字符回退到字节表示，避免无法编码的异常。
sp.SentencePieceTrainer.train(
    input='train.txt',           # 输入文件
    model_prefix='my_tokenizer', # 输出模型文件名前缀
    vocab_size=483,             # 词表大小（实战通常为32000，此处设小方便演示）
    character_coverage=0.9995,   # 覆盖 99.95% 的字符，适合中英文
    model_type='bpe',            # 使用 BPE 算法
    user_defined_symbols=['<pad>', '<mask>'], # 自定义特殊符号
    byte_fallback=True           # 遇到不认识的字符回退到字节表示，永不报错
)

print("训练完成！已生成 my_tokenizer.model 和 my_tokenizer.vocab")

# 加载模型
sp_model = sp.SentencePieceProcessor(model_file='my_tokenizer.model')

# 测试用例
text_en = "I love deep learning"
text_cn = "我爱深度学习"

# 1) 编码 (Text -> IDs)：得到整数 ID 序列，供模型使用
ids_en = sp_model.encode_as_ids(text_en)
ids_cn = sp_model.encode_as_ids(text_cn)

# 2) 分词 (Text -> Tokens)：得到 piece 级 token 列表，便于观察切分
tokens_en = sp_model.encode_as_pieces(text_en)
tokens_cn = sp_model.encode_as_pieces(text_cn)

print(f"英文分词: {tokens_en}")
print(f"英文 ID: {ids_en}")
print(f"\n中文分词: {tokens_cn}")
print(f"中文 ID: {ids_cn}")

# 3) 解码 (IDs -> Text)：将 ID 序列还原为文本，校验可逆性
print(f"\n解码测试: {sp_model.decode(ids_cn)}")
