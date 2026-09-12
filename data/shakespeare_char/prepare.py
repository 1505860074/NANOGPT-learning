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
# os（Operating System，操作系统）：Python 标准库，提供对系统的接口，本文件用到 os.path.join / os.path.dirname / os.path.exists。
import pickle
# pickle（可译为"腌制品"，寓意把数据"腌制"存起来）：Python 标准库，负责 Python 对象序列化。
# 序列化=把内存对象（字典/列表等）保存成二进制文件并能在之后原样读回。
import requests
# requests（请求）：Python 社区最流行的 HTTP 客户端库，专门用来"发网络请求下载数据"。
# 本文件用 requests.get() 下载莎士比亚数据集。这是本文件第一个第三方库。
import numpy
# numpy（Numerical Python，数值 Python）：科学计算核心库，提供高性能多维数组运算，本文件用 numpy.array 编码并落盘。

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
SCRIPT_DIRECTORY = os.path.dirname(__file__)
# 变量 __file__：Python 自动注入的"当前这个文件本身的路径"。
# os.path.dirname(path)：取出路径里的"目录部分"（去掉文件名）。
# 两行连起来得到"prepare.py 自己所在的目录"，这样输入/输出文件不管从哪个目录运行，都落在正确的地方。
"""本脚本所在目录，输入（input.txt）与输出（bin / pkl 文件）都放在这里。"""


def download_dataset() -> str:
    """若 input.txt 不存在则下载，并返回其完整文本。

    输入: 无。
    输出: data（str）— 莎士比亚原始文本（约 110 万个字符）。
    """
    input_file_path = os.path.join(SCRIPT_DIRECTORY, 'input.txt')
    # os.path.join(a, b)：把路径片段拼成完整路径，自动按系统规则加分隔符。
    if not os.path.exists(input_file_path):
        # os.path.exists(path)：判断路径是否存在，存在返回 True。
        with open(input_file_path, 'w') as f:
            # open(path, 'w')：以"写入（write）"模式打开/新建文件。
            f.write(requests.get(DATA_URL).text)
            # requests.get(url)：向 url 发一个 HTTP GET 请求，把服务器返回的内容封装成对象。
            #   .text：取出返回的文本内容（这里就是莎士比亚全集原文）。
    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")
    # len(x)：内建函数，求长度（字符串=字符数）。f-string 里 :, 是千分位分隔符（1,115,394）。
    return data


def build_character_codec(data: str) -> dict:
    """从文本推导字符表，并构建"字符 <-> 整数"的双向映射与编解码函数。

    输入: data（str）— 原始文本。
    输出: dict，含 vocab_size（int）、stoi（dict[str, int]）、itos（dict[int, str]）、
          encode（Callable[[str], list[int]]）、decode（Callable[[list[int]], str]）。
    """
    # 取出这段文本里出现过的所有不重复字符
    characters = sorted(list(set(data)))
    # set(data)：内建函数，把字符串转成"集合（set）"——自动去掉重复字符，剩下一堆唯一字符。
    # list(...)：转回列表。sorted(...)：内建函数，排序（按字符编码从小到大）。
    vocabulary_size = len(characters)
    print("all the unique characters:", ''.join(characters))
    # ''.join(列表)：用空字符串把列表元素拼成一个大字符串，这里用于打印所有字符。
    print(f"vocab size: {vocabulary_size:,}")

    # 建立"字符 -> 整数"的映射
    string_to_integer = { character:index for index,character in enumerate(characters) }
    integer_to_string = { index:character for index,character in enumerate(characters) }
    # enumerate(列表)：返回 (下标, 元素) 的有序迭代器，这里同时拿到"字符"和它的"编号"。
    # 字典推导式 {键:值 for ...}：一行构造字典的语法，建立双向映射。
    def encode(string):
        return [string_to_integer[character] for character in string] # 编码器：输入字符串，输出整数列表
    # 列表推导式 [表达式 for ...]：把字符串逐字符映射成整数编号。
    def decode(token_list):
        return ''.join([integer_to_string[token_id] for token_id in token_list]) # 解码器：输入整数列表，输出字符串
    return {
        'vocab_size': vocabulary_size,
        'stoi': string_to_integer,
        'itos': integer_to_string,
        'encode': encode,
        'decode': decode,
    }


def export_binary_files(data: str, encode: callable) -> tuple:
    """按 90/10 切分文本并编码，导出 train.bin / val.bin（uint16）。

    输入:
        data（str）— 原始文本。
        encode（Callable[[str], list[int]]）— 字符级编码函数。
    输出: (train_token_count, val_token_count)（tuple[int, int]）— 两个切分各自的 token 数。
    """
    # 划分训练集和验证集
    dataset_length = len(data)
    train_data = data[:int(dataset_length*0.9)]
    validation_data = data[int(dataset_length*0.9):]
    # 字符串切片 data[a:b]：取下标 a 到 b-1 的子串。int(x)：内建函数转整数。
    #   前 90% 给训练、后 10% 给验证。

    # 把两份数据都编码成整数
    train_token_ids = encode(train_data)
    validation_token_ids = encode(validation_data)
    print(f"train has {len(train_token_ids):,} tokens")
    print(f"val has {len(validation_token_ids):,} tokens")

    # 导出成 bin 二进制文件
    train_token_ids = numpy.array(train_token_ids, dtype=numpy.uint16)
    validation_token_ids = numpy.array(validation_token_ids, dtype=numpy.uint16)
    # numpy.array(列表, dtype)：把普通列表变成 numpy 数组，dtype 指定元素类型。
    #   numpy.uint16：无符号 16 位整数（每个 token 占 2 字节），比 Python 默认的 int 大幅省内存。
    train_token_ids.tofile(os.path.join(SCRIPT_DIRECTORY, 'train.bin'))
    validation_token_ids.tofile(os.path.join(SCRIPT_DIRECTORY, 'val.bin'))
    # .tofile(path)：numpy 数组方法，把数组原始二进制内容直接写入文件（不加任何格式头）。
    return len(train_token_ids), len(validation_token_ids)


def write_metadata(codec: dict) -> None:
    """把词表大小与编解码表写进 meta.pkl，供后续训练脚本读取。

    输入: codec（dict）— build_character_codec 的返回值。
    输出: 无（生成 meta.pkl 文件）。
    """
    # 注意：下面这几个字符串键名是 meta.pkl 的数据格式，不随变量改名而变
    metadata = {
        'vocab_size': codec['vocab_size'],
        'itos': codec['itos'],
        'stoi': codec['stoi'],
    }
    with open(os.path.join(SCRIPT_DIRECTORY, 'meta.pkl'), 'wb') as f:
        # 'wb'：以"写入二进制（write binary）"模式打开。
        pickle.dump(metadata, f)
        # pickle.dump(对象, 文件)：把对象序列化后写入文件（存成二进制）。
        #   之后可用 pickle.load() 原样读回。


def main():
    """字符级数据准备主流程：下载 → 建字符词表 → 切分编码导出 bin → 写 meta.pkl。"""
    # ---- 1. 下载并读取原始文本 ----
    data = download_dataset()
    # ---- 2. 构建字符级词表与编解码函数 ----
    codec = build_character_codec(data)
    # ---- 3. 切分、编码并导出二进制文件 ----
    export_binary_files(data, codec['encode'])
    # ---- 4. 保存元信息 ----
    write_metadata(codec)


if __name__ == '__main__':
    # __name__=='__main__'：直接运行本脚本时执行 main()；被 import 时不执行（脚本/库两用）。
    main()