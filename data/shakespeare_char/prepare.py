"""
把莎士比亚数据集准备成"字符级"语言建模用的形式。
也就是说，不用 GPT-2 的 BPE token 做编码，而是直接把每个字符映射成整数。
会保存 train.bin、val.bin（存放这些 id），以及 meta.pkl（存放编码器、解码器
和一些其他相关信息）。

--- 以下为英文原文 ---
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy

# 下载 tiny shakespeare 数据集
# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# 取出这段文本里出现过的所有不重复字符
# get all the unique characters that occur in this text
characters = sorted(list(set(data)))
vocabulary_size = len(characters)
print("all the unique characters:", ''.join(characters))
print(f"vocab size: {vocabulary_size:,}")

# 建立"字符 -> 整数"的映射
# create a mapping from characters to integers
string_to_integer = { character:index for index,character in enumerate(characters) }
integer_to_string = { index:character for index,character in enumerate(characters) }
def encode(string):
    return [string_to_integer[character] for character in string] # 编码器：输入字符串，输出整数列表
def decode(token_list):
    return ''.join([integer_to_string[token_id] for token_id in token_list]) # 解码器：输入整数列表，输出字符串

# 划分训练集和测试集
# create the train and test splits
dataset_length = len(data)
train_data = data[:int(dataset_length*0.9)]
validation_data = data[int(dataset_length*0.9):]

# 把两份数据都编码成整数
# encode both to integers
train_token_ids = encode(train_data)
validation_token_ids = encode(validation_data)
print(f"train has {len(train_token_ids):,} tokens")
print(f"val has {len(validation_token_ids):,} tokens")

# 导出成 bin 二进制文件
# export to bin files
train_token_ids = numpy.array(train_token_ids, dtype=numpy.uint16)
validation_token_ids = numpy.array(validation_token_ids, dtype=numpy.uint16)
train_token_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
validation_token_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# 顺便把元信息也保存下来，方便之后做编码/解码
# save the meta information as well, to help us encode/decode later
# 注意：下面这几个字符串键名是 meta.pkl 的数据格式，不随变量改名而变
metadata = {
    'vocab_size': vocabulary_size,
    'itos': integer_to_string,
    'stoi': string_to_integer,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(metadata, f)

# 以下是这个脚本运行后打印出来的实际结果（原样保留，方便对照终端输出）：
# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
