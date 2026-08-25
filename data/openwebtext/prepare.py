# 把 openwebtext 数据集保存成一个二进制文件以供训练使用。下面这个链接很有帮助：
# (saves the openwebtext dataset to a binary file for training. following was helpful:)
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface 的 datasets 库 (huggingface datasets)

# .map() 调用中的工作进程数量 (number of workers in .map() call)
# 一个合适的取值大约是 cpu 核心数 // 2 这个量级 (good number to use is ~order number of cpu cores // 2)
num_proc = 8

# load_dataset() 调用中的工作进程数量 (number of workers in load_dataset() call)
# 最佳取值可能和上面的 num_proc 不同，因为它还取决于网络（NW）速度。
# 不过通常来说比设成 1 要好。
# (best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though)
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # 在 huggingface 的 .cache 目录里占用 54GB，约有 8M（800 万）篇文档（8,013,769 篇）
    # (takes 54GB in huggingface .cache dir, about 8M documents (8,013,769))
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # owt 默认只包含 'train' 这一个划分（split），所以要创建一个 test 划分 (owt by default only contains the 'train' split, so create a test split)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # 把 test 划分改名为 val (rename the test split to val)

    # 结果如下： (this results in:)
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # 现在我们要对数据集做分词（tokenize）。先定义编码函数（gpt2 bpe）(we now want to tokenize the dataset. first define the encoding function (gpt2 bpe))
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary 会忽略所有特殊 token (encode_ordinary ignores any special tokens)
        ids.append(enc.eot_token) # 添加文本结束（end of text）token，比如 gpt2 bpe 里是 50256 (add the end of text token, e.g. 50256 for gpt2 bpe)
        # 注意：我觉得 eot 应该加在开头而不是末尾……嗯。不过它毕竟叫 "eot"（文本结束）……
        # (note: I think eot should be prepended not appended... hmm. it's called "eot" though...)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # 对数据集做分词 (tokenize the dataset)
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # 把每个数据集里所有的 id 拼接成一个可用于训练的大文件 (concatenate all the ids in each dataset into one large file we can use for training)
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # （可以这么做，因为 enc.max_token_value == 50256 小于 2**16）((can do since enc.max_token_value == 50256 is < 2**16))
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # 把样本打包成批，以加快写入速度 (Batch together samples for faster write)
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # 写入 mmap（内存映射文件）(Write into mmap)
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin 约 17GB，val.bin 约 8.5MB (train.bin is ~17GB, val.bin ~8.5MB)
    # 训练集约有 9B（90 亿）个 token（9,035,582,198）(train has ~9B tokens)
    # 验证集约有 4M（400 万）个 token（4,434,897）(val has ~4M tokens)

    # 之后想读取这些 bin 文件，比如用 numpy： (to read the bin files later, e.g. with numpy:)
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
