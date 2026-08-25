"""
穷人版配置器（Poor Man's Configurator）。这大概是个糟糕的主意。用法示例：
$ python train.py config/override_file.py --batch_size=32
这会先运行 config/override_file.py，然后再把 batch_size 覆盖为 32

本文件中的代码会以如下方式从 train.py 之类的地方运行：
>>> exec(open('configurator.py').read())

所以它不是一个 Python 模块，只是把这段代码从 train.py 里挪出去而已。
本脚本中的代码随后会覆盖 globals()（全局变量）

我知道大家不会喜欢这种做法，我只是真的很不喜欢配置的复杂性、
不喜欢在每一个变量前面都得加上 config. 前缀。如果有人想出了更好的
简洁 Python 方案，我非常愿意听。
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:  
        # 假定它是一个配置文件的名字 (assume it's the name of a config file)
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # 假定它是一个 --key=value 形式的参数 (assume it's a --key=value argument)
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # 尝试对它求值（比如布尔值、数字等等）(attempt to eval it it (e.g. if bool, number, or etc))
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # 如果求值失败，就直接当字符串用 (if that goes wrong, just use the string)
                attempt = val
            # 确保类型能对上 (ensure the types match ok)
            assert type(attempt) == type(globals()[key])
            # 祈祷别出错 (cross fingers)
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
