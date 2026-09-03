import os
import requests
import tiktoken
import numpy

# 下载 tiny shakespeare 数据集
# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
dataset_length = len(data)
train_data = data[:int(dataset_length*0.9)]
validation_data = data[int(dataset_length*0.9):]

# 用 tiktoken 的 gpt2 bpe 做编码
# encode with tiktoken gpt2 bpe
encoder = tiktoken.get_encoding("gpt2")
train_token_ids = encoder.encode_ordinary(train_data)
validation_token_ids = encoder.encode_ordinary(validation_data)
print(f"train has {len(train_token_ids):,} tokens")
print(f"val has {len(validation_token_ids):,} tokens")

# 导出成 bin 二进制文件
# export to bin files
train_token_ids = numpy.array(train_token_ids, dtype=numpy.uint16)
validation_token_ids = numpy.array(validation_token_ids, dtype=numpy.uint16)
train_token_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
validation_token_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin 有 301,966 个 token
# val.bin 有 36,059 个 token
# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
