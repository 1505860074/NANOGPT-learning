"""
把莎士比亚数据集准备成 BPE（GPT-2 tokenizer）编码的二进制文件：train.bin / val.bin。
"""
import os
# os（Operating System，操作系统）：Python 标准库，提供对系统的接口，本文件用到 os.path.join / os.path.dirname / os.path.exists。
import requests
# requests（请求）：最常见的 HTTP 客户端库，专门用来"发网络请求下载数据"，本文件用 requests.get() 下载数据集。
import tiktoken
# tiktoken（"token" 编码器）：OpenAI 官方分词器库，提供 GPT 系列模型使用的 BPE 分词。
import numpy
# numpy（Numerical Python，数值 Python）：科学计算核心库，本文件用 numpy.array 编码并落盘。

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
SCRIPT_DIRECTORY = os.path.dirname(__file__)
# __file__：Python 自动注入的"当前文件路径"；os.path.dirname() 取出所在目录，保证文件落到脚本旁边。
"""本脚本所在目录，输入（input.txt）与输出（bin 文件）都放在这里。"""


def download_dataset() -> str:
    """若 input.txt 不存在则下载，并返回其完整文本。

    输入: 无。
    输出: data（str）— 莎士比亚原始文本。
    """
    input_file_path = os.path.join(SCRIPT_DIRECTORY, 'input.txt')
    # os.path.join(a, b)：把路径片段拼成完整路径。
    if not os.path.exists(input_file_path):
        # os.path.exists(path)：判断路径是否存在。
        with open(input_file_path, 'w', encoding='utf-8') as f:
            # open(path, 'w', encoding)：写入模式打开文件；encoding='utf-8' 指定用 UTF-8 字符集写入。
            f.write(requests.get(DATA_URL).text)
            # requests.get(url).text：发出 GET 请求并取回服务器返回的文本内容。
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data


def split_train_validation(data: str) -> tuple:
    """把文本按 90/10 切分成训练集与验证集。

    输入: data（str）— 原始文本。
    输出: (train_data, validation_data)（tuple[str, str]）。
    """
    dataset_length = len(data)
    train_data = data[:int(dataset_length*0.9)]
    validation_data = data[int(dataset_length*0.9):]
    # 字符串切片 data[a:b]：取下标 a 到 b-1；int(x)：转成整数。前 90% 训练、后 10% 验证。
    return train_data, validation_data


def encode_with_gpt2(text: str) -> list:
    """用 tiktoken 的 gpt2 BPE 把文本编码成 token 整数列表。

    输入: text（str）— 待编码文本。
    输出: token_ids（list[int]）— 编码后的 token 序列。
    """
    encoder = tiktoken.get_encoding("gpt2")
    # tiktoken.get_encoding("gpt2")：取到 GPT-2 的 BPE 编码器对象。
    return encoder.encode_ordinary(text)
    # encoder.encode_ordinary(文本)：把文本切成 token 编号列表。
    #   "ordinary"（普通的）指不把特殊 token（如 <|endoftext|>）当字面量，只做常规分词。


def export_binary_files(train_token_ids: list, validation_token_ids: list) -> None:
    """把 token 序列转成 uint16 数组并写入 train.bin / val.bin。

    输入:
        train_token_ids（list[int]）— 训练集 token 序列。
        validation_token_ids（list[int]）— 验证集 token 序列。
    输出: 无（生成 train.bin / val.bin 文件）。
    """
    train_token_ids = numpy.array(train_token_ids, dtype=numpy.uint16)
    validation_token_ids = numpy.array(validation_token_ids, dtype=numpy.uint16)
    # numpy.array(列表, dtype)：普通列表 → numpy 数组；numpy.uint16 表示无符号 16 位整数（省内存）。
    train_token_ids.tofile(os.path.join(SCRIPT_DIRECTORY, 'train.bin'))
    validation_token_ids.tofile(os.path.join(SCRIPT_DIRECTORY, 'val.bin'))
    # .tofile(path)：把数组原始二进制内容直接写进文件。


def main():
    """BPE 数据准备主流程：下载 → 切分 → GPT-2 编码 → 导出 bin 文件。"""
    # ---- 1. 下载并读取原始文本 ----
    data = download_dataset()

    # ---- 2. 切分训练集 / 验证集 ----
    train_data, validation_data = split_train_validation(data)

    # ---- 3. 用 gpt2 BPE 编码两份文本 ----
    train_token_ids = encode_with_gpt2(train_data)
    validation_token_ids = encode_with_gpt2(validation_data)
    print(f"train has {len(train_token_ids):,} tokens")
    print(f"val has {len(validation_token_ids):,} tokens")

    # ---- 4. 导出成 bin 二进制文件 ----
    export_binary_files(train_token_ids, validation_token_ids)
    # train.bin 有 301,966 个 token；val.bin 有 36,059 个 token


if __name__ == '__main__':
    # __name__=='__main__'：直接运行本脚本时执行 main()；被 import 时不执行。
    main()