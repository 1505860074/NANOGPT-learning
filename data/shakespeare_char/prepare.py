"""
为字符级（character-level）语言建模准备莎士比亚数据集。
也就是说，我们不用 GPT-2 的 BPE token 来编码，而是直接把字符映射成整数。
会保存包含这些 id 的 train.bin 和 val.bin，以及包含编码器、解码器和
其他一些相关信息的 meta.pkl。
(Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.)
"""
import os
import pickle
import requests
import numpy as np

# 下载 tiny shakespeare（迷你莎士比亚）数据集 (download the tiny shakespeare dataset)
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# 取出这段文本中出现过的所有不重复字符 (get all the unique characters that occur in this text)
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# 建立一个从字符到整数的映射 (create a mapping from characters to integers)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # 编码器：输入一个字符串，输出一个整数列表 (encoder: take a string, output a list of integers)
def decode(l):
    return ''.join([itos[i] for i in l]) # 解码器：输入一个整数列表，输出一个字符串 (decoder: take a list of integers, output a string)

# 创建训练集和测试集的划分 (create the train and test splits)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 把两部分都编码成整数 (encode both to integers)
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# 导出为 bin 文件 (export to bin files)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# 同时保存元信息（meta），方便我们之后做编码/解码 (save the meta information as well, to help us encode/decode later)
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# 数据集的字符长度 (length of dataset in characters):  1115394
# 所有不重复的字符 (all the unique characters):
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# 词表大小 (vocab size): 65
# 训练集有 1003854 个 token (train has 1003854 tokens)
# 验证集有 111540 个 token (val has 111540 tokens)
