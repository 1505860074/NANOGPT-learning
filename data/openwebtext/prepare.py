"""
把 openwebtext 数据集保存成用于训练的二进制文件（train.bin / val.bin）。参考：
https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
"""
import os
# os（Operating System，操作系统）：Python 标准库，提供对系统的接口，本文件用到 os.path.join / os.path.dirname。
import tqdm
# tqdm（读法近似 "taqaddum"（阿拉伯语"进步"），也常念 "progress"）：一个进度条库。
# 它能在一行里实时刷新显示循环进度，长任务（如下载、遍历）时直观看到完成比例。
import numpy
# numpy（Numerical Python，数值 Python）：科学计算核心库，本文件用 numpy.memmap / numpy.sum / numpy.concatenate。
import tiktoken
# tiktoken：OpenAI 官方分词器库，本文件用它把 800 万篇网页文本切成 GPT 模型的 token。
import datasets
# datasets（HuggingFace 的数据集库）：简称 HF Datasets。专门用来加载/管理大规模机器学习数据集，
#   支持从 HuggingFace Hub 一键下载、并行处理、流式读取。本文件就用它加载 openwebtext。

# .map() 调用时使用的工作进程数，好的取值大约是 cpu 核心数 // 2
number_of_processes = 8
# // 整除运算符。
# load_dataset() 调用时使用的工作进程数，通常设成比 1 大总是更好的
number_of_processes_for_load_dataset = number_of_processes

encoder = tiktoken.get_encoding("gpt2")
# tiktoken.get_encoding("gpt2")：拿到 GPT-2 的 BPE 编码器对象，供下方分词用（全局复用一次）。


def load_and_split_dataset() -> dict:
    """加载 openwebtext 数据集，并从 train 切分出 val 验证集。

    输入: 无。
    输出: split_dataset（DatasetDict）— 含 'train' 与 'val' 两个切分的字典。
    """
    # 会在 huggingface 的 .cache 目录里占用 54GB，大约 800 万篇文档
    dataset = datasets.load_dataset("openwebtext", num_proc=number_of_processes_for_load_dataset)
    # datasets.load_dataset(name, num_proc)：从 HuggingFace Hub 把 openwebtext 数据集下载并缓存到本地。
    #   num_proc：用多少个进程并行下载/处理，加快速度。

    # owt 默认只包含 'train' 这一个切分，所以这里另外划分出一个验证集
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    # .train_test_split(...)：把训练集随机切成"训练 + 测试"两份（机器学习最基本的数据划分方法）。
    #   test_size=0.0005：测试集占万分之五；seed 随机种子（固定可复现）；shuffle=True 先打乱再切。
    split_dataset['val'] = split_dataset.pop('test') # 把 test 切分改名成 val
    # dict.pop('test')：取出 'test' 这组数据并从字典中删除它，再放回槽位 'val'。
    return split_dataset


def make_tokenize_function() -> callable:
    """构建单篇文档的分词函数（闭包，复用全局 encoder）。

    输入: 无。
    输出: process（Callable[[dict], dict]）— 输入含 'text' 的样本，输出含 'ids' 与 'len'。
    """
    def process(example):
        token_ids = encoder.encode_ordinary(example['text']) # encode_ordinary 会忽略所有特殊 token
        # encoder.encode_ordinary(文本)：把一篇文章切成 token 编号列表（不处理特殊 token）。
        token_ids.append(encoder.eot_token) # 追加文本结束 token（gpt2 bpe 里是 50256）
        # .append(x)：列表方法，在末尾加上一个元素。
        # encoder.eot_token：文本结束符（end of text）的编号，帮模型知道"这篇文章讲完了"。
        output = {'ids': token_ids, 'len': len(token_ids)}
        return output
    return process


def tokenize_dataset(split_dataset: dict) -> dict:
    """对整个数据集做 BPE 分词，去掉原 'text' 列。

    输入: split_dataset（DatasetDict）— load_and_split_dataset 的返回值。
    输出: tokenized（DatasetDict）— 各切分表里含 'ids' 与 'len' 列。
    """
    process = make_tokenize_function()
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=number_of_processes,
    )
    # .map(函数, ...)：HuggingFace 数据集的核心方法，把 process 应用到"每一行样本"上，
    #   自动做并行化与进度条。remove_columns 删掉用不上的原始 'text' 列（省内存）。
    #   desc：显示在进度条上的描述文字。num_proc：并行进程数。
    return tokenized


def write_bin_files(tokenized: dict) -> None:
    """把每个切分里的 token id 全部拼接，写入对应的 *.bin（uint16 内存映射文件）。

    输入: tokenized（DatasetDict）— tokenize_dataset 的返回值。
    输出: 无（生成 train.bin / val.bin；train 约 17GB、90 亿 token）。
    """
    # 把每个切分里的所有 id 拼接成一个大文件，供训练使用
    for split, dataset_split in tokenized.items():
        # dict.items()：逐对返回 (键, 值)，这里键是 'train'/'val'，值是对应的数据表。
        array_length = numpy.sum(dataset_split['len'], dtype=numpy.uint64)
        # numpy.sum(数组, dtype)：把所有样本长度加起来 = 总 token 数，作为数组总长度。
        #   dtype=numpy.uint64 无符号 64 位整数，因为 90 亿远超 32 位能表示的范围。
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        data_type = numpy.uint16 # encoder.max_token_value == 50256 < 2**16，可安全用 uint16
        # numpy.uint16：无符号 16 位整数，最大 65535，装得下 50256 的 token 编号，省内存。
        memmap_array = numpy.memmap(filename, dtype=data_type, mode='w+', shape=(array_length,))
        # numpy.memmap(path, dtype, mode, shape)：创建内存映射文件——文件在磁盘上，但你可以像操作数组一样读写它。
        #   mode='w+'：写模式，用于新建/覆盖文件。shape 是数组形状（这里是一维、总 token 数）。
        #   好处：900M 个元素的文件不用全部加载进内存，边写边落盘。
        total_batches = 1024

        write_index = 0
        for batch_index in tqdm.tqdm(range(total_batches), desc=f'writing {filename}'):
            # tqdm.tqdm(可迭代对象, desc)：给 for 循环包一层进度条，实时显示写到了第几批。
            # 把样本攒成一批一起写，写得更快
            batch = dataset_split.shard(num_shards=total_batches, index=batch_index, contiguous=True).with_format('numpy')
            # .shard(num_shards, index, contiguous)：把数据集平均切成 1024 片，取第 index 片。
            #   contiguous=True 保证取到的是连续完整的一块（不交错）。
            # .with_format('numpy')：让这一片的数据以 numpy 数组形式返回（方便后续运算）。
            batch_array = numpy.concatenate(batch['ids'])
            # numpy.concatenate(数组列表)：把所有样本的 token 列表首尾拼接成一整个大数组。
            memmap_array[write_index : write_index + len(batch_array)] = batch_array
            # 切片赋值 memmap[起:止] = 数组：把这批 token 写到内存映射数组的对应下标区间。
            write_index += len(batch_array)
        memmap_array.flush()
        # .flush()：把缓冲区里的数据真的刷写到磁盘上（保证文件完整）。

    # 之后要读取这些 bin 文件，例如：
    # m = numpy.memmap('train.bin', dtype=numpy.uint16, mode='r')


def main():
    """OpenWebText 数据准备主流程：加载并切分 → 分词 → 拼接导出 bin 文件。"""
    # ---- 1. 加载数据集并划分验证集 ----
    split_dataset = load_and_split_dataset()

    # ---- 2. 对整个数据集做 gpt2 BPE 分词 ----
    tokenized = tokenize_dataset(split_dataset)

    # ---- 3. 拼接各切分的 token 并写进二进制文件 ----
    write_bin_files(tokenized)


if __name__ == '__main__':
    # __name__=='__main__'：直接运行本脚本时执行 main()；被 import 时不执行。
    main()