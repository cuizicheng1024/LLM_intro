from transformers import AutoTokenizer

# 1. 加载 GPT-2 的分词器 (经典 BPE) 
# 如果无法访问 HuggingFace 仓库，则需要手动下载GPT2 文件，并指定本地路径
# GPT-2 使用 Byte-Pair Encoding (BPE) 算法，适合处理英文等字母语言
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")

# 2. 加载一个中文模型分词器 (如 BERT-base-Chinese) 
# BERT 使用 WordPiece 分词算法，对于中文通常是按字分词
# 同样，如果无法在线下载，需要先下载模型文件到本地
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-chinese")

# 定义测试文本
text_en = "Learning NLP is unhappily difficult but exciting!"
text_cn = "我喜欢学习自然语言处理，虽然它是个痛苦的过程"

# 3. 打印 GPT-2 的英文分词结果
# tokenize() 方法将文本分割成 token（词元）
print("#####################################")
print("GPT-2 English Tokens:", tokenizer_gpt2.tokenize(text_en))
# encode() 方法将文本转换为模型所需的 token ID 序列
print("GPT-2 English IDs:", tokenizer_gpt2.encode(text_en))

# 4. 打印 BERT 的中文分词结果
# 注意观察中文分词的结果，通常是单字
print("#####################################")
print("\nBERT Chinese Tokens:", tokenizer_bert.tokenize(text_cn))
print("BERT Chinese IDs:", tokenizer_bert.encode(text_cn))

# 5. 观察 [CLS] 和 [SEP] 特殊标记
# 直接调用分词器对象会返回包含特殊标记的字典
# input_ids: 包含 [CLS] (开始) 和 [SEP] (结束) 的 token ID 序列
# token_type_ids: 用于区分句子对（在单句任务中通常为0）
# attention_mask: 用于指示哪些 token 是实际内容，哪些是 padding
encoded_input = tokenizer_bert(text_cn)
print("\nBERT Full Encoded (with special tokens):", encoded_input['input_ids'])
# 可以使用 decode 将 ID 转回文本，以验证特殊标记的存在
print("Decoded:", tokenizer_bert.decode(encoded_input['input_ids']))
