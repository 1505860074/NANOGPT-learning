# 把 openwebtext 数据集保存成用于训练的二进制文件。下面这个链接很有参考价值：
# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
import tqdm
import numpy
import tiktoken
import datasets # huggingface 的 datasets 库

# .map() 调用时使用的工作进程数
# 一个好的取值大约是 cpu 核心数 // 2 这个量级
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
number_of_processes = 8

# load_dataset() 调用时使用的工作进程数
# 最佳取值可能和上面的 number_of_processes 不同，因为它还取决于网络速度。
# 不过通常设成比 1 大总是更好的
# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
number_of_processes_for_load_dataset = number_of_processes

encoder = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # 会在 huggingface 的 .cache 目录里占用 54GB，大约 800 万篇文档（8,013,769 篇）
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = datasets.load_dataset("openwebtext", num_proc=number_of_processes_for_load_dataset)

    # owt 默认只包含 'train' 这一个切分，所以这里另外划分出一个 test 切分
    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # 把 test 切分改名成 val

    # 结果长这样：
    # this results in:
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

    # 现在要对数据集做分词。先定义编码函数（gpt2 bpe）
    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        token_ids = encoder.encode_ordinary(example['text']) # encode_ordinary 会忽略所有特殊 token
        token_ids.append(encoder.eot_token) # 追加文本结束 token，比如 gpt2 bpe 里是 50256
        # 注：我觉得 eot 应该放在开头而不是结尾……唔，不过它毕竟叫 "eot"（end of text）
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        output = {'ids': token_ids, 'len': len(token_ids)}
        return output

    # 对数据集做分词
    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=number_of_processes,
    )

    # 把每个切分里的所有 id 拼接成一个大文件，供训练使用
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dataset_split in tokenized.items():
        array_length = numpy.sum(dataset_split['len'], dtype=numpy.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        data_type = numpy.uint16 # （可以这么用，因为 encoder.max_token_value == 50256 小于 2**16）
        memmap_array = numpy.memmap(filename, dtype=data_type, mode='w+', shape=(array_length,))
        total_batches = 1024

        write_index = 0
        for batch_index in tqdm.tqdm(range(total_batches), desc=f'writing {filename}'):
            # 把样本攒成一批一起写，写得更快
            # Batch together samples for faster write
            batch = dataset_split.shard(num_shards=total_batches, index=batch_index, contiguous=True).with_format('numpy')
            batch_array = numpy.concatenate(batch['ids'])
            # 写入 mmap
            # Write into mmap
            memmap_array[write_index : write_index + len(batch_array)] = batch_array
            write_index += len(batch_array)
        memmap_array.flush()

    # train.bin 约 17GB，val.bin 约 8.5MB
    # 训练集约有 90 亿个 token（9,035,582,198）
    # 验证集约有 400 万个 token（4,434,897）
    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # 之后要读取这些 bin 文件，比如用 numpy：
    # to read the bin files later, e.g. with numpy:
    # m = numpy.memmap('train.bin', dtype=numpy.uint16, mode='r')
