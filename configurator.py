"""
穷人版配置器（Poor Man's Configurator）。这大概是个糟糕的主意。用法示例：
$ python train.py config/override_file.py --batch_size=32
它会先运行 config/override_file.py，然后再把 batch_size 覆盖成 32

这个文件里的代码会以如下方式被 train.py 之类的脚本执行：
>>> exec(open('configurator.py').read())

所以它并不是一个 Python 模块，只是把这段代码从 train.py 里挪了出来。
脚本里的代码随后会直接覆盖 globals()（全局变量）。

我知道大家不会喜欢这种写法，我只是实在讨厌配置系统的复杂度，
也讨厌每个变量前面都得加个 config. 前缀。如果有人能想出更好的
简单 Python 方案，我洗耳恭听。

--- 以下为英文原文 ---
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
# sys（system，系统）：Python 标准库，访问与 Python 解释器相关的系统信息。
# 本文件用到 sys.argv——命令行参数列表（argv 全称 argument vector，参数向量）。
import ast
# ast（Abstract Syntax Tree，抽象语法树）：Python 标准库，用来解析 Python 代码结构的工具。
# 本文件用 ast.literal_eval() 安全地把字符串"字面量"（数字/布尔/字符串等）还原成对应类型的值，不做危险计算。

for argument in sys.argv[1:]:
    # sys.argv：命令行参数组成的列表。sys.argv[0] 是脚本名本身，sys.argv[1:] 才是用户传进来的参数。
    #   比如 python train.py --BATCH_SIZE=32 → argv = ['train.py', '--BATCH_SIZE=32']。
    # 列表切片 [1:]：从下标 1 开始取到末尾（去掉脚本名）。
    if '=' not in argument:
        # 认为它是一个配置文件的文件名
        # assume it's the name of a config file
        assert not argument.startswith('--')
        # assert：断言，条件为假就抛异常。这里确保配置文件路径不以 '--' 开头。
        config_file = argument
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            # with ... as f：上下文管理器自动关闭文件；open() 默认文本读模式。
            print(f.read())
        exec(open(config_file).read())
        # exec(代码字符串)：内建函数，把字符串当作 Python 代码执行。
        #   这里运行的是配置文件全文（里面是 WANDB_LOG = False 之类的大写变量赋值），
        #   因为是在调用者（如 train.py）的全局空间里 exec，所以这些赋值直接改写了调用者的模块级变量。
    else:
        # 认为它是一个 --key=value 形式的参数
        # assume it's a --key=value argument
        assert argument.startswith('--')
        key, value = argument.split('=')
        # str.split('=')：按分隔符把字符串切成列表，这里切成 [key部分, value部分]。
        key = key[2:]
        # 字符串切片 [2:]：去掉前两个字符 '--'，得到真正的键名。
        if key in globals():
            # globals()：内建函数，返回当前模块所有"全局变量"构成的字典（键是变量名）。
            #   这解释了为什么超参数必须写在大写顶格——configurator 只能覆盖已经存在的全局键。
            try:
                # try ... except：异常处理。先试着把 value 解析成非字符串类型。
                attempt = ast.literal_eval(value)
                # ast.literal_eval("32") → 32（int）；"True" → True（bool）；"0.1" → 0.1（float）。
                #   literal_eval 只解析"字面量"，绝不执行任意代码，因此是安全的"字符串按类型还原"函数。
            except (SyntaxError, ValueError):
                # 如果求值失败，就当成字符串用
                # if that goes wrong, just use the string
                attempt = value
                # except (类型列表)：捕获这两种异常，说明它不是能解析的字面量，只能当字符串。
            # 确保类型对得上
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # type(x)：内建函数，返回 x 的类型对象。这里强制命令行覆盖的类型和原全局变量类型一致，
            #   比如不能把原本是 int 的 BATCH_SIZE 覆盖成字符串 'abc'。
            # 祈祷一切顺利
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
            # 关键一步：把覆盖后的新值写回全局字典，从而修改影调用者的超参数。
        else:
            raise ValueError(f"Unknown config key: {key}")
            # raise ValueError：主动抛出异常，报告"这个配置键不存在"。
