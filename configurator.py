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
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # 认为它是一个配置文件的文件名
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # 认为它是一个 --key=value 形式的参数
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # 尝试对它求值（比如它是布尔值、数字等等）
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # 如果求值失败，就当成字符串用
                # if that goes wrong, just use the string
                attempt = val
            # 确保类型对得上
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # 祈祷一切顺利
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
