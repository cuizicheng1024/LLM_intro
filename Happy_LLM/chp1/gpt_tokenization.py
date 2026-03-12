# 代码示例
import tiktoken

# 使用 GPT-4 的编码器
enc = tiktoken.get_encoding("cl100k_base")

# 编码一个字符串
text = "小沈阳江西演唱会邀请了"
tokens = enc.encode(text)
print(f"Token IDs: {tokens}")
print(f"Token 数量: {len(tokens)}")

# 解码（把 Token ID 转回文字）
decoded = enc.decode(tokens)
print(f"解码结果: {decoded}")

# 查看每个 Token 对应的文字
for token_id in tokens:
    print(f"  {token_id} → '{enc.decode([token_id])}'")