"""
从训练好的模型中采样生成文本。

--- 以下为英文原文 ---
Sample from a trained model
"""
import os
# os（Operating System，操作系统）：Python 标准库，提供对系统的接口。
# 本文件用到 os.path.join（拼路径）、os.path.exists（判断文件是否存在）、os.path.dirname（取目录名）等路径处理函数。
import pickle
# pickle（可翻译为"腌制品"：寓意把数据"腌制"保存起来）：Python 标准库，负责 Python 对象的序列化。
# 序列化=把内存里的对象（字典、列表等）保存成二进制文件，之后可再读回成对象；本文件用它读取检查点里的元信息 meta.pkl。
import contextlib
# contextlib（context + lib，上下文管理库）：Python 标准库，提供制作/操作"上下文管理器"的工具。
# 本文件用于 nullcontext()——一个"什么都不做"的上下文管理器，用来在 CPU 分支下占位。
import torch
# torch：「PyTorch」的官方包名。开源深度学习框架，提供 GPU 张量计算与自动求导。
import tiktoken
# tiktoken（通俗理解为 "token" 开头的编码器）：OpenAI 官方的分词器库。
# 分词器的作用：把人类文本切成模型能懂的"token（词元）"，或把 token 拼回文本。gpt2 编码器是它的默认实现。
import model
# model：本仓库自己的模块（即同目录下的 model.py），里面定义了 GPT 模型。

# -----------------------------------------------------------------------------
# 超参数（必须留在模块顶层：configurator.py 通过 globals() 覆盖这些值）
# -----------------------------------------------------------------------------
INITIALIZE_FROM = 'resume' # 取 'resume'（从 output_directory 恢复）或某个 gpt2 变体（如 'gpt2-xl'）
"""从哪里加载模型：'resume' 读本地检查点，或 'gpt2' / 'gpt2-xl' 等加载 OpenAI 预训练权重。"""
OUTPUT_DIRECTORY = 'out' # 若 initialize_from 不是 'resume' 则忽略此项
"""存放 ckpt.pt 的目录。"""
START = "\n" # 也可以是 "<|endoftext|>" 等。还能指定文件，写成："FILE:prompt.txt"
"""生成起始提示词文本；写成 "FILE:xxx.txt" 则从该文件读取内容。"""
NUMBER_OF_SAMPLES = 10 # 采样生成的样本条数
"""要生成多少条样本。"""
MAX_NEW_TOKENS = 500 # 每条样本生成的 token 数
"""每条样本生成多少个新 token。"""
TEMPERATURE = 0.8 # 1.0 = 不改变预测分布，< 1.0 = 更确定，> 1.0 = 更随机
"""采样温度，越小越保守、越大越随机。"""
TOP_K = 200 # 只保留概率最高的 top_k 个 token，其余的概率钳成 0
"""每步只从概率最高的 k 个候选里采样。"""
SEED = 1337
"""随机种子，固定它可以复现同样的生成结果。"""
DEVICE = 'cuda' # 例如：'cpu'、'cuda'、'cuda:0'、'cuda:1' 等
"""推理使用的设备。"""
DATA_TYPE = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 可选 'float32'、'bfloat16'、'float16'
"""推理用的浮点精度。"""
COMPILE = False # 用 PyTorch 2.0 编译模型以提速
"""是否用 PyTorch 2.0 编译模型；生成少量样本时通常不值得。"""
exec(open('configurator.py').read()) # 从命令行或配置文件读取覆盖项
# exec(代码字符串)：内建函数，把字符串当作 Python 代码执行。
# open(path)：内建函数，打开文件并返回文件对象；.read() 读完全文。
# 整个过程：运行 configurator.py 的代码 → 它解析命令行参数（如 --DEVICE=cpu）→ 覆盖上方的大写超参数。


def setup_runtime() -> dict:
    """初始化随机种子、TF32 开关与混合精度上下文。

    输入: 无。
    输出: dict，含 device_type（str）与 autocast_context（上下文管理器）。
    """
    torch.manual_seed(SEED)
    # torch.manual_seed(n)：给 CPU 的随机数生成器种下种子 n。之后所有随机操作序列固定可复现。
    torch.cuda.manual_seed(SEED)
    # torch.cuda.manual_seed(n)：同上，但针对 GPU 的随机生成器，与上一句配套使用。
    torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cuda.*：PyTorch 存放"后端开关"的命名空间。这里允许了矩阵乘用 TF32（一种半精度计算），GPU 上更快。
    torch.backends.cudnn.allow_tf32 = True
    # cudnn（CUDA Deep Neural Network library）：英伟达的深度网络加速库。让它也用 TF32，整体提速。
    device_type = 'cuda' if 'cuda' in DEVICE else 'cpu' # 供 torch.autocast 使用
    pytorch_data_type = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[DATA_TYPE]
    # 字典取值：用字符串键从刚才那个字典里取出对应的 torch 数据类型对象。
    autocast_context = contextlib.nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=pytorch_data_type)
    # torch.amp.autocast(device_type, dtype)：混合精度上下文管理器。进入 with 块后，
    #   能半精度的运算自动降到低精度跑（更快省显存），需要精度的运算留在 float32（不降精度出错）。
    #   dtype 传 bfloat16 或 float16（半精度浮点）。
    # contextlib.nullcontext()：占位用的空上下文（什么都不做），CPU 上没有 autocast 加速，就用它占位保持代码结构一致。
    return {'device_type': device_type, 'autocast_context': autocast_context}


def load_model(device: str) -> tuple:
    """加载模型：'resume' 从检查点恢复，'gpt2*' 加载预训练权重，并搬上指定设备。

    可以拆成三步：
    1) 按 INITIALIZE_FROM 分支重建模型（检查点或预训练权重）
    2) 剥掉 torch.compile 加的 '_orig_mod.' 键名前缀后装载权重
    3) 切成 eval 模式、搬到 device，可选地 torch.compile

    输入: device（str）— 目标设备，如 'cuda'。
    输出: (gpt_model, checkpoint)
          gpt_model（model.GPT）— 加载好的模型，eval 态且已在 device 上。
          checkpoint（dict|None）— resume 模式读到的检查点（供后续取数据集的 meta），其余模式为 None。
    """
    checkpoint = None
    if INITIALIZE_FROM == 'resume':
        # 从保存在指定目录下的模型初始化
        checkpoint_path = os.path.join(OUTPUT_DIRECTORY, 'ckpt.pt')
        # os.path.join(a, b)：把路径片段拼成一个完整路径，自动按系统规则加分隔符（Windows 用 \）。
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # torch.load(path, map_location)：从磁盘读回之前 torch.save 的检查点对象（本质是字典）。
        #   map_location=device：把里面的张量直接加载到指定设备上。
        # 用检查点里记录的模型结构参数重建配置
        gpt_config = model.GPTConfig(**checkpoint['model_args'])
        # ** 字典拆包：把字典展开成关键字参数一次性传给构造函数。
        # checkpoint['model_args'] 里存着之前的模型结构参数（层数、维度等）。
        gpt_model = model.GPT(gpt_config)
        state_dictionary = checkpoint['model']
        # 剥掉 torch.compile 加上的权键名前缀
        unwanted_prefix = '_orig_mod.'
        for key, value in list(state_dictionary.items()):
            # dict.items()：返回 (键, 值) 组成的视图；list(...) 转成普通列表，避免边遍历边改字典出问题。
            # 点名 list(state_dictionary.items())：因为下面要改键名，必须先复制一份快照再改。
            if key.startswith(unwanted_prefix):
                # str.startswith(s)：判断字符串是否以 s 开头。torch.compile 会在键名前加 '_orig_mod.'。
                state_dictionary[key[len(unwanted_prefix):]] = state_dictionary.pop(key)
                # str[s:]：字符串切片，从第 s 个字符开始截取，这里就是"去掉前缀"。
                # dict.pop(key)：取出该键的值并从字典中删除它。
                # 综合：把 '前缀+原名' 的键，改成纯 '原名'，值不变。
        gpt_model.load_state_dict(state_dictionary)
        # load_state_dict(dict)：把权重字典按参数名一一填进模型，恢复训练时的状态。
    elif INITIALIZE_FROM.startswith('gpt2'):
        # 从指定的 GPT-2 模型初始化
        gpt_model = model.GPT.from_pretrained(INITIALIZE_FROM, dict(dropout=0.0))
    else:
        raise ValueError(f"未知的 INITIALIZE_FROM: {INITIALIZE_FROM}")
        # raise ValueError(...)：主动抛出"值错误"异常（不符合预期的值）。f"..." 是 f-string 格式化字符串。

    gpt_model.eval()
    # eval()：切成评估模式——关闭 dropout 等训练专用行为，保证生成结果稳定。
    gpt_model.to(device)
    # .to(device)：把模型（包括所有参数）搬到目标设备。CPU 上为空操作。
    if COMPILE:
        gpt_model = torch.compile(gpt_model) # 需要 PyTorch 2.0（可选）
        # torch.compile(model)：PyTorch 2.0 的即时编译器，把模型整段优化成更快的执行图，需几秒到几分钟的编译时间。
    return gpt_model, checkpoint


def build_codec(checkpoint: dict) -> tuple:
    """构建"文本 <-> token 整数"的编解码函数。

    优先用数据集目录里的 meta.pkl（字符级模型用）；没有就用 GPT-2 的 BPE 编码兜底。

    输入: checkpoint（dict|None）— resume 模式的检查点，用于定位对应数据集的 meta.pkl。
    输出: (encode, decode)
          encode（Callable[[str], list[int]]）— 把字符串编码成 token 整数列表。
          decode（Callable[[list[int]], str]）— 把 token 整数列表解码成字符串。
    """
    # 到数据集目录里找找有没有 meta 这个 pickle 文件
    load_metadata = False
    if INITIALIZE_FROM == 'resume' and 'config' in checkpoint:
        # 检查点里存的是大写 'DATASET'，早期版本存小写 'dataset'，两种都认
        checkpoint_config = checkpoint['config']
        checkpoint_dataset = checkpoint_config.get('DATASET', checkpoint_config.get('dataset'))
        # dict.get(key, 默认值)：安全取字典值，键不存在时返回默认值（而不是抛 KeyError）。
        if checkpoint_dataset is not None:
            metadata_path = os.path.join('data', checkpoint_dataset, 'meta.pkl')
            load_metadata = os.path.exists(metadata_path)
            # os.path.exists(path)：判断路径是否存在，存在返回 True。

    if load_metadata:
        print(f"Loading meta from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            # with ... as f：上下文管理器 + 文件句柄，离开 with 块文件会自动关闭（省去手动 close）。
            # 'rb'：读取二进制模式（read binary）。
            metadata = pickle.load(f)
            # pickle.load(f)：从文件里把序列化对象读回成 Python 对象。
        # 'stoi'/'itos' 是 meta.pkl 的数据格式键名，不随变量改名而变
        string_to_integer, integer_to_string = metadata['stoi'], metadata['itos']
        # 'stoi'（string to integer，字符到整数）、'itos'（integer to string，整数到字符）。
        encode = lambda string: [string_to_integer[character] for character in string]
        # lambda 形参: 表达式 = 匿名函数（一次性的小函数）。这行定义 encode：把字符串里每个字符查表换成整数。
        decode = lambda token_list: ''.join([integer_to_string[token_id] for token_id in token_list])
        # ''.join(列表)：用 ''（空串）把列表里的元素拼成一个大字符串。
    else:
        # 没有 meta.pkl，默认按 gpt-2 的编码方式来
        print("No meta.pkl found, assuming GPT-2 encodings...")
        encoder = tiktoken.get_encoding("gpt2")
        # tiktoken.get_encoding("gpt2")：拿到 GPT-2 使用的 BPE 编码器对象。
        #   BPE（Byte Pair Encoding）：一种把文本切成"次词元"子词的分词算法，输入文本、输出 token 编号。
        encode = lambda string: encoder.encode(string, allowed_special={"<|endoftext|>"})
        # encoder.encode(text)：文本 → token 编号列表。allowed_special 告诉它允许哪些特殊 token（如文本结束符）。
        decode = lambda token_list: encoder.decode(token_list)
        # encoder.decode(ids)：token 编号列表 → 文本。
    return encode, decode


def main():
    """采样主流程：初始化运行环境 → 加载模型 → 构建编解码 → 逐条生成并打印。"""
    # ---- 1. 初始化运行环境（随机种子 + 混合精度） ----
    runtime = setup_runtime()

    # ---- 2. 加载模型 ----
    gpt_model, checkpoint = load_model(DEVICE)

    # ---- 3. 构建文本编解码函数 ----
    encode, decode = build_codec(checkpoint)

    # ---- 4. 把起始提示词编码成输入张量 ----
    start_text = START
    if START.startswith('FILE:'):
        # 从文件读取真正的提示词文本
        with open(START[5:], 'r', encoding='utf-8') as f:
            # 'r' 读取文本模式；encoding='utf-8' 指定用 UTF-8 字符集解码，中文注释这样才不乱码。
            start_text = f.read()
    start_token_ids = encode(start_text)
    input_token_indices = torch.tensor(start_token_ids, dtype=torch.long, device=DEVICE)[None, ...] # 前面补一个批维度
    # torch.tensor(列表, dtype, device)：用 Python 列表直接构造张量，可指定数据类型与设备。
    #   [None, ...]：给张量加一个"尺寸 1 的新维度"放在最前面。模型输入要求 (批, 序列)，原来只有 (序列)，
    #   所以用 None 扩维；... 是省略号，表示"剩下的维原样不动"。

    # ---- 5. 采样生成并打印 ----
    with torch.no_grad():
        # torch.no_grad()：不记录梯度（推理专用，省内存、提速）。
        with runtime['autocast_context']:
            for sample_index in range(NUMBER_OF_SAMPLES):
                # range(n)：内建函数，生成 0,1,...,n-1。
                generated_indices = gpt_model.generate(input_token_indices, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
                print(decode(generated_indices[0].tolist()))
                # generated_indices[0]：取批维度上的第 0 条样本（我们只生成 1 条）。
                # .tolist()：张量 → Python 列表（decode 需要普通列表）。
                print('---------------')


if __name__ == '__main__':
    # __name__：当前模块名。直接运行本文件时 __name__ == '__main__'，于是执行 main()；
    # 被别的文件 import 时则不会执行——这是"把脚本同时当程序跑、当库导入"的标准写法。
    main()