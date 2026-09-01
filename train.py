"""
这个训练脚本既可以在单张 GPU 上以调试模式运行，
也可以用分布式数据并行（DDP, distributed data parallel）做更大规模的训练。

在单张 GPU 上运行的示例：
$ python train.py --batch_size=32 --compile=False

在 1 个节点的 4 张 GPU 上用 DDP 运行的示例：
$ torchrun --standalone --nproc_per_node=4 train.py

跨 2 个节点、共 4 张 GPU 用 DDP 运行的示例：
- 在第一个（主）节点上运行，示例 IP 为 123.456.123.456：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- 在工作节点（worker node）上运行：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
（如果你的集群没有 Infiniband 互联，请在命令前加上 NCCL_IB_DISABLE=1）

【本文件在项目中的定位】
train.py 是 nanoGPT 的训练入口脚本：读取超参数配置 -> 按需初始化分布式训练 ->
构建/恢复模型 -> 进入训练主循环（前向、反向、更新参数、定期评估并保存 checkpoint）。
被训练出的模型权重（checkpoint）会存到 out_dir 下，供 sample.py 之类的脚本加载生成文本。

【全局名词解释（后面正文再遇到这些词，只会给极短提示，不再重复讲）】
- DDP（DistributedDataParallel）：PyTorch 提供的"分布式数据并行"方案。把同一个模型复制到多张 GPU
  （甚至多台机器）上，每张卡各自跑一部分数据的前向/反向，然后自动把各卡算出的梯度做平均同步，
  等效于用更大的 batch size 训练，从而加速训练。
- torchrun：PyTorch 自带的多进程启动器，负责在每张卡上各拉起一个 train.py 进程，并通过环境变量
  RANK / LOCAL_RANK / WORLD_SIZE 告诉每个进程"我是第几个进程、用第几张卡、总共几个进程"。
- 混合精度训练（mixed precision）：正常训练用 32 位浮点数（float32）存储数字，精度高但慢、占显存多。
  混合精度指训练时部分计算改用更"窄"的格式（bfloat16 / float16），速度更快、显存更省，
  同时靠 autocast、GradScaler 等机制保住足够的数值精度不让训练跑飞。
- autocast：一个上下文管理器（context manager，即 `with ... :` 语法背后的对象），进入它包裹的代码块后，
  PyTorch 会自动决定哪些运算可以安全地用低精度（如 bfloat16/float16）计算、哪些仍需 float32，无需手动改代码。
- GradScaler（梯度缩放器）：float16 数值范围较窄，很小的梯度可能直接下溢变成 0，导致训练学不到东西。
  GradScaler 先把 loss 乘大一个倍数再反向传播（梯度也跟着变大，避免下溢），更新参数前再除回去。
  用 bfloat16 时数值范围足够，不需要它，这时它会被设成"空操作"（enabled=False）。
- 梯度累积（gradient accumulation）：显存有限，没法一次塞下很大的 batch。做法是把一个大 batch
  拆成若干个小的"微批"（micro-batch），依次算each微批的梯度并累加起来，累加够了才更新一次参数，
  效果上等价于用了更大的 batch size。
- torch.compile：PyTorch 2.0 引入的即时编译（JIT）功能，把模型的计算图编译成更高效的底层代码，
  运行更快，代价是第一次调用前要花时间编译。
- checkpoint（检查点）：训练过程中定期把模型参数、优化器状态、当前迭代步数等打包保存到磁盘的文件，
  用于训练中断后续训、或者训练完之后加载权重去做推理/生成。
- MFU（Model FLOPs Utilization，模型算力利用率）：实际达到的每秒浮点运算次数，占 GPU 理论峰值算力的
  百分比，是衡量"训练效率高不高、有没有让 GPU 吃满"的常用指标。
"""

# ==================== 依赖导入 ====================
import os
import time
import math
import pickle
from contextlib import nullcontext # nullcontext：一个什么都不做的"空"上下文管理器，见下方 ctx 变量的用法

import numpy as np # 用来通过 memmap 高效读取预处理好的二进制 token 文件（见 get_batch）
import torch
from torch.nn.parallel import DistributedDataParallel as DDP # 多卡训练容器，见文件头 DDP 名词解释
from torch.distributed import init_process_group, destroy_process_group # 分布式进程组的建立/销毁

from model import GPTConfig, GPT # 项目自己实现的 GPT 模型结构与配置类，定义在同目录 model.py 里

# ==================== 默认超参数配置 ====================
# 默认配置值，是为在 OpenWebText 上训练一个 gpt2 (124M) 而设计的 (default config values designed to train a gpt2 (124M) on OpenWebText)
# I/O（输入输出）
out_dir = 'out' # 检查点和日志的输出目录 (output directory for checkpoints and logs)
eval_interval = 2000 # 每隔多少次迭代评估一次训练/验证损失 (how many iterations between evaluations)
log_interval = 1 # 每隔多少次迭代在终端打印一次训练日志 (how many iterations between printing training log)
eval_iters = 200 # 评估时用多少个批次估算损失，取平均以降低波动 (how many batches to average over when estimating loss)
eval_only = False # 如果为 True，脚本在第一次评估后立刻退出 (if True, script exits right after the first eval)
always_save_checkpoint = True # 如果为 True，每次评估后都保存一个检查点 (if True, always save a checkpoint after each eval)
init_from = 'scratch' # 'scratch'（从零开始）或 'resume'（从检查点续训）或 'gpt2*'
# wandb 日志记录 (wandb logging)
wandb_log = False # 默认关闭 (disabled by default)
wandb_project = 'owt' # wandb 项目名 (wandb project name)
wandb_run_name = 'gpt2' # wandb 运行名，默认可以是 'run' + str(time.time()) ('run' + str(time.time()))
# 数据 (data)
dataset = 'openwebtext' # 数据集名称，对应 data/ 目录下的子目录名 (dataset name, matches subdirectory name under data/)
gradient_accumulation_steps = 5 * 8 # 用于模拟更大的批大小 (used to simulate larger batch sizes)
batch_size = 12 # 如果 gradient_accumulation_steps > 1，这个值是微批（micro-batch）大小 (if gradient_accumulation_steps > 1, this is the micro-batch size)
block_size = 1024 # 上下文长度，即模型一次能看到的最大 token 数 (context length, max number of tokens the model attends to at once)
# 模型 (model)
n_layer = 12 # Transformer 层数 (number of transformer layers)
n_head = 12 # 每层的注意力头数 (number of attention heads per layer)
n_embd = 768 # 嵌入维度，也是模型的隐藏层宽度 (embedding dimension, i.e. model's hidden width)
dropout = 0.0 # 预训练时用 0 比较好，微调时可以试试 0.1 以上 (for pretraining 0 is good, for finetuning try 0.1+)
bias = False # 我们是否在 LayerNorm 和 Linear 层里使用偏置？ (do we use bias inside LayerNorm and Linear layers?)
# adamw 优化器 (adamw optimizer)
learning_rate = 6e-4 # 最大学习率 (max learning rate)
max_iters = 600000 # 训练迭代的总次数 (total number of training iterations)
weight_decay = 1e-1 # 权重衰减系数，用于 L2 正则化 (weight decay coefficient, for L2 regularization)
beta1 = 0.9 # AdamW 一阶动量的衰减系数 (AdamW's beta1, decay rate for the first moment estimate)
beta2 = 0.95 # AdamW 二阶动量的衰减系数 (AdamW's beta2, decay rate for the second moment estimate)
grad_clip = 1.0 # 把梯度裁剪到这个值，若等于 0.0 则禁用裁剪 (clip gradients at this value, or disable if == 0.0)
# 学习率衰减设置 (learning rate decay settings)
decay_lr = True # 是否对学习率做衰减 (whether to decay the learning rate)
warmup_iters = 2000 # 预热（warm up）多少步 (how many steps to warm up for)
lr_decay_iters = 600000 # 按 Chinchilla 论文的做法，应当约等于 max_iters (should be ~= max_iters per Chinchilla)
min_lr = 6e-5 # 最小学习率，按 Chinchilla 论文的做法，应当约等于 learning_rate/10 (minimum learning rate, should be ~= learning_rate/10 per Chinchilla)
# DDP（分布式数据并行）设置 (DDP settings)
backend = 'nccl' # 'nccl'、'gloo' 等等 ('nccl', 'gloo', etc.)
# 系统 (system)
device = 'cuda' # 例如：'cpu'、'cuda'、'cuda:0'、'cuda:1' 等等，在 macbook 上可以试试 'mps' (examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks)
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32'、'bfloat16' 或 'float16'，最后一种会自动启用 GradScaler ('float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler)
compile = True # 使用 PyTorch 2.0 编译模型以提速 (use PyTorch 2.0 to compile the model to be faster)
# -----------------------------------------------------------------------------
# 先把当前 globals()（也就是上面那一大串超参数变量）里所有"简单类型"变量名记下来，
# 这样待会儿不管 configurator.py 往 globals() 里加了/改了什么，都知道哪些是"配置项"
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# nanoGPT 的配置覆盖技巧：exec() 会把 configurator.py 文件内容当 Python 代码直接执行一遍。
# configurator.py 会读命令行参数（如 --batch_size=32）或指定的配置文件，直接对本文件里
# 同名的全局变量重新赋值，从而不用写一大堆 argparse 样板代码就能实现"命令行覆盖超参数"。
exec(open('configurator.py').read()) # 来自命令行或配置文件的覆盖项 (overrides from command line or config file)
config = {k: globals()[k] for k in config_keys} # 把最终生效的配置值收集成一个字典，对日志记录会有用 (will be useful for logging)
# -----------------------------------------------------------------------------

# ==================== 分布式训练（DDP）初始化 ====================
# 各种初始化、派生属性、I/O 设置 (various inits, derived attributes, I/O setup)
# 条件：环境变量 RANK 是否存在（torchrun 启动多进程时会自动设置它），用来判断这次运行是不是 DDP 多卡训练
ddp = int(os.environ.get('RANK', -1)) != -1 # 这是一次 dd1·p 运行吗？ (is this a ddp run?)
if ddp:
    # 分支：DDP 多卡/多机训练。每张卡各自跑一个独立的 train.py 进程，这里做的是"每个进程认清自己身份"
    init_process_group(backend=backend) # 建立进程组，让各进程之间能通信、同步梯度；backend 见上方配置项
    ddp_rank = int(os.environ['RANK']) # 本进程在"所有机器所有卡"里的全局编号
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # 本进程在"当前这台机器"上的编号，用来选定用第几张卡
    ddp_world_size = int(os.environ['WORLD_SIZE']) # 参与训练的进程（等于 GPU）总数
    device = f'cuda:{ddp_local_rank}' # 本进程专属的 GPU，例如 'cuda:0'、'cuda:1'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # 这个进程负责日志记录、保存检查点等工作 (this process will do logging, checkpointing etc.)
    seed_offset = ddp_rank # 每个进程拿到不同的随机种子 (each process gets a different seed)
    # 会有 world_size 个进程同时训练，所以我们可以按比例把
    # 每个进程期望的梯度累积迭代次数等比缩小
    # (world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally)
    # 目的：不管用几张卡，每次参数更新实际处理的总 token 数（tokens_per_iter）保持不变，
    # 所以卡数越多，每张卡自己需要做的梯度累积次数就等比减少
    assert gradient_accumulation_steps % ddp_world_size == 0 # 兜底校验：必须能整除，否则没法按比例平均分配
    gradient_accumulation_steps //= ddp_world_size
else:
    # 兜底分支：不是 DDP，说明是在单张 GPU（或 CPU）、单个进程上运行 (if not ddp, we are running on a single gpu, and one process)
    master_process = True # 只有一个进程，自然它自己就是"主进程"
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process: # 只让主进程建目录，避免多进程同时创建时产生竞争/重复日志
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset) # 固定随机种子保证可复现；加 seed_offset 让 DDP 各进程种子不同，采样到不同数据
# 在gpu运算上的性能优化
# tf32（TensorFloat-32）：NVIDIA Ampere 及以后架构 GPU 支持的一种运算精度，矩阵乘法速度比 float32 快很多，
# 精度损失很小，训练场景通常可以放心开启
torch.backends.cuda.matmul.allow_tf32 = True # 允许矩阵乘法使用 tf32 (allow tf32 on matmul)
torch.backends.cudnn.allow_tf32 = True # 允许 cudnn 使用 tf32 (allow tf32 on cudnn)

# ==================== 混合精度上下文（autocast / dtype）====================
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后面 torch.autocast 使用 (for later use in torch.autocast)
# 注意：float16 数据类型会自动使用 GradScaler (note: float16 data type will automatically use a GradScaler)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype] # 把字符串配置映射成真正的 torch 数据类型
# ctx：训练循环里 `with ctx:` 包住前向传播的那个上下文管理器。
# CPU 上没有意义就用 nullcontext()（什么都不做的占位符，见文件头解释），保持 `with ctx:` 写法统一；
# GPU 上用 torch.amp.autocast 开启自动混合精度（见文件头 autocast 名词解释）
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ==================== 数据加载器 ====================
# 穷人版数据加载器 (poor man's data loader)：没有用 PyTorch 的 Dataset/DataLoader 体系，
# 而是直接手写"从二进制文件里随机取一段"，因为 nanoGPT 的训练数据是预先分词好、拼接成一整条
# uint16 token 序列存在磁盘上的，比常规的按样本加载简单、也更快
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # 我们每个批次都重新创建 np.memmap，以避免内存泄漏，依据如下讨论：
    # (We recreate np.memmap every batch to avoid a memory leak, as per)
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    # memmap：NumPy 提供的"内存映射文件"，不会把整个文件读进内存，而是像访问内存数组一样按需从磁盘读取，
    # 这样即便 train.bin 有几十 GB 大，也能随机访问其中任意一段而不用整个装进内存
    if split == 'train': # 条件：请求的是训练集切分
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else: # 兜底：非 'train' 一律当作验证集切分
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 随机采样 batch_size 个起始位置
    # x：输入序列；y：把同一段窗口整体右移一位得到的"下一个 token"标签——这就是语言模型
    # "预测下一个 token"训练目标的构造方式，x[i] 位置的正确答案就是 y[i]
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda': # 条件：目标设备是 GPU，值得做锁页内存 + 异步搬运这套优化
        # 把数组 x、y 固定（pin）在内存中，这样我们就能异步地把它们搬到 GPU 上（non_blocking=True）(pin arrays x,y, which allows us to move them to GPU asynchronously)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else: # 兜底：CPU 训练，没有"内存搬显存"的开销，直接搬过去即可
        x, y = x.to(device), y.to(device)
    return x, y

# ==================== 训练状态初始化 & 词表大小推导 ====================
# 先在这里初始化这些变量，如果 init_from='resume'（即从检查点恢复）可以被覆盖 (init these up here, can override if init_from='resume' (i.e. from a checkpoint))
iter_num = 0
best_val_loss = 1e9

# 尝试从数据集中推导出 vocab_size（词表大小）(attempt to derive vocab_size from the dataset)
# vocab_size 决定模型最后输出层的维度（要在这么多个候选 token 里选一个），数据预处理脚本
# 会把"词表大小 + token 与 id 的映射关系"存进 meta.pkl，这里尝试读出来复用，保证和数据对齐
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path): # 条件：这个数据集在预处理时生成了 meta.pkl（自定义词表，如字符级数据集）
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f) # pickle：Python 标准库的序列化格式，用来把字典/对象存成二进制文件再读回来
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
# 没有 meta.pkl 时（如常见的 GPT-2 BPE 分词数据集），vocab_size 会在下面按 GPT-2 的默认值兜底

# ==================== 构建模型：从零训练 / 续训 / 加载 GPT-2 预训练权重 ====================
# 模型初始化 (model init)
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # 先用命令行传入的 model_args 作为起点 (start with model_args from command line)
if init_from == 'scratch':
    # 分支：从零开始训练一个全新模型，参数随机初始化，不加载任何已有权重
    print("Initializing a new model from scratch")
    # 确定从零训练时我们要用的词表大小 (determine the vocab size we'll use for from-scratch training)
    if meta_vocab_size is None: # 兜底：数据集没提供自定义词表（没有 meta.pkl）
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304 # 50304 是把 GPT-2 真实词表 50257 向上取整到 128 的倍数，便于 GPU 做高效矩阵运算
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    # 分支：从之前中断的训练检查点续训，而不是重新开始
    print(f"Resuming training from {out_dir}")
    # 从一个检查点恢复训练。(resume training from a checkpoint.)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device) # map_location：把保存时在别的设备上的张量重新映射加载到当前 device 上
    checkpoint_model_args = checkpoint['model_args']
    # 强制让这些配置属性保持一致，否则根本没法恢复训练；
    # 其余属性（比如 dropout）可以保留命令行里想要的值
    # (force these config attributes to be equal otherwise we can't even resume training;
    # the rest of the attributes (e.g. dropout) can stay as desired from command line)
    # 原因：这些属性决定了模型每一层参数张量的形状，必须和检查点里保存的权重形状完全一致，
    # 否则 load_state_dict 会直接因为形状不匹配而报错
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # 创建模型 (create the model)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 修正状态字典（state dict）里的键名 :(
    # 老实说不知道为什么检查点有时会带上这个前缀，还得再排查
    # (fix the keys of the state dictionary :( honestly no idea how checkpoints sometimes
    # get this prefix, have to debug more)
    # 补充说明：这个前缀实际来自下方的 torch.compile(model)——编译后的模型内部会把原模型包一层，
    # 保存权重时键名就会多出 "_orig_mod." 前缀；这里统一把它剥掉，不管保存时是否编译过都能正常加载
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix): # 条件：这个参数名带有编译产生的多余前缀
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k) # 去掉前缀后用新键名重新放回字典
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num'] # 恢复训练进度，接着上次的迭代步数继续计数
    best_val_loss = checkpoint['best_val_loss'] # 恢复"历史最佳验证损失"记录，用于后续判断要不要保存新 checkpoint
elif init_from.startswith('gpt2'):
    # 分支：加载 OpenAI 官方发布的 GPT-2 预训练权重作为起点（常用于微调场景），
    # init_from 此时的值形如 'gpt2'、'gpt2-medium' 等，对应不同规模的预训练模型
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # 从 OpenAI 的 GPT-2 权重初始化 (initialize from OpenAI GPT-2 weights)
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args) # from_pretrained 定义在 model.py，负责下载/转换 OpenAI 权重
    # 读出创建好的配置参数，这样我们才能正确地把它们存入检查点 (read off the created config params, so we can store them into checkpoint correctly)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# 如果需要，用「模型手术」把模型的 block size 裁小 (crop down the model block size if desired, using model surgery)
# "模型手术"：不是重新训练，而是直接在已经构建好的模型上，把位置编码等和 block_size 相关的张量
# 原地裁剪成更小的尺寸，从而降低显存占用和计算量——只能裁小，不能裁大
if block_size < model.config.block_size: # 条件：当前想要的上下文长度比模型原本支持的更短
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # 这样检查点里存的才是正确的值 (so that the checkpoint will have the right value)
model.to(device) # 把模型的所有参数搬到目标设备（CPU/GPU），用法同前面讲过的张量 .to()

# 初始化一个 GradScaler（梯度缩放器）。如果 enabled=False，scaler 就是个空操作 (initialize a GradScaler. If enabled=False scaler is a no-op)
# 详见文件头 GradScaler 名词解释：只有 dtype 是 float16 时才真正启用，bfloat16/float32 不需要
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# ==================== 优化器 / 模型编译 / DDP 包装 ====================
# 优化器 (optimizer)
# configure_optimizers 定义在 model.py 里，负责把参数分组（例如给权重矩阵用 weight decay、
# 给偏置/LayerNorm 等一维参数不用），再用给定的学习率、动量系数构造出 AdamW 优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume': # 条件：是续训场景，优化器的动量等内部状态也要恢复，否则相当于重新开始热身
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # 释放内存 (free up memory)：checkpoint 字典里含有完整模型权重的拷贝，用完及时丢弃

# 编译模型 (compile the model)
if compile: # 条件：配置里开启了 torch.compile 加速（见文件头名词解释）
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model # 保留一份编译前的原始模型引用，备用（比如某些场景编译后的模型不便直接操作）
    model = torch.compile(model) # 需要 PyTorch 2.0 (requires PyTorch 2.0)

# 把模型包进 DDP 容器 (wrap model into DDP container)
if ddp: # 条件：多卡训练，需要用 DDP 包一层来自动同步各卡梯度（见文件头 DDP 名词解释）
    model = DDP(model, device_ids=[ddp_local_rank])

# ==================== 训练/评估辅助函数 ====================
# 借助多个批次，对训练集或验证集估算出一个任意精度的损失值 (helps estimate an arbitrarily accurate loss over either split using many batches)
# 单个 batch 的 loss 波动很大，跑 eval_iters 个 batch 取平均，能更稳定地反映模型当前真实水平
@torch.no_grad() # 装饰器（decorator）：给函数"贴"上一个功能。这里表示函数体内的所有张量运算都不记录梯度，
                 # 因为评估阶段不需要反向传播，关掉梯度记录能省显存、算得更快
def estimate_loss():
    out = {}
    model.eval() # 切到"评估模式"：像 dropout 这类训练/评估行为不同的层会切换成评估时的行为（本项目 dropout=0.0 时影响不大，但仍是标准做法）
    for split in ['train', 'val']: # 分别在训练集、验证集上各估一次损失，便于对比是否过拟合
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx: # 复用前面定义的混合精度上下文，评估时也享受同样的加速
                logits, loss = model(X, Y)
            losses[k] = loss.item() # .item()：把只有一个数的张量转成普通 Python 浮点数，方便后续存进普通 tensor/做打印
        out[split] = losses.mean()
    model.train() # 评估完切回"训练模式"，不要忘了切回去，否则接下来的训练会一直用评估模式的行为
    return out

# 学习率衰减调度器（带预热的余弦衰减）(learning rate decay scheduler (cosine with warmup))
# "预热"（warmup）：训练刚开始时模型参数还很随机，用较大学习率容易直接训崩，所以先从很小的学习率
# 线性增大到目标学习率；过了预热期再用余弦曲线慢慢降到一个较小的下限，兼顾前期稳定和后期精细收敛
def get_lr(it):
    # 分支 1) 当前迭代数还在预热期内 -> 学习率从接近 0 线性增大到 learning_rate (linear warmup for warmup_iters steps)
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 分支 2) 已经超过衰减计划的总步数 -> 兜底：固定用最小学习率，不再继续下降 (if it > lr_decay_iters, return min learning rate)
    if it > lr_decay_iters:
        return min_lr
    # 分支 3) 介于预热结束和衰减终点之间 -> 用余弦曲线平滑地从 learning_rate 降到 min_lr (in between, use cosine decay down to min learning rate)
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters) # 当前处于衰减阶段的进度，0 表示刚开始衰减，1 表示衰减完毕
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff 的取值范围是 0..1 (coeff ranges 0..1)；余弦曲线让下降前后段更平滑，不是匀速直线下降
    return min_lr + coeff * (learning_rate - min_lr)

# ==================== 实验日志（wandb）====================
# 日志记录 (logging)
if wandb_log and master_process: # 条件：用户开启了 wandb 上报，且只让主进程上报（避免多卡重复上报同一份数据）
    import wandb # wandb（Weights & Biases）：一个训练实验跟踪/可视化平台，能把每一步的指标画成网页上的曲线图
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# ==================== 训练主循环 ====================
# 训练循环 (training loop)
X, Y = get_batch('train') # 取出最开始的第一个批次 (fetch the very first batch)
t0 = time.time()
local_iter_num = 0 # 本进程生命周期内的迭代次数 (number of iterations in the lifetime of this process)
raw_model = model.module if ddp else model # 如有需要，从 DDP 容器里把模型解包出来 (unwrap DDP container if needed)；DDP 包装后真正的模型被存在 .module 属性里
running_mfu = -1.0 # MFU 的指数滑动平均，-1.0 表示"还没有有效数据"，见文件头 MFU 名词解释
while True: # 没有固定的 for 循环范围，靠循环体末尾的迭代计数 + 下面的终止条件手动 break 退出

    # 确定并设置本次迭代的学习率 (determine and set the learning rate for this iteration)
    lr = get_lr(iter_num) if decay_lr else learning_rate # 条件表达式：开启衰减就按调度器算，否则整个训练过程固定用配置的学习率
    for param_group in optimizer.param_groups: # PyTorch 优化器可以给不同参数组设不同学习率，这里统一改成同一个 lr
        param_group['lr'] = lr

    # 在训练集/验证集上评估损失，并写出检查点 (evaluate the loss on train/val sets and write checkpoints)
    if iter_num % eval_interval == 0 and master_process: # 条件：到了预定的评估间隔，且只让主进程做评估/打印/保存，避免多卡重复工作
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log: # 条件：开启了 wandb，把这一次评估的指标上报上去
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # 转换成百分比 (convert to percentage)
            })
        if losses['val'] < best_val_loss or always_save_checkpoint: # 条件：验证损失刷新了历史最优，或者配置要求"每次评估都无条件保存"
            best_val_loss = losses['val']
            if iter_num > 0: # 条件：跳过训练刚开始（第 0 步）时的保存，此时模型还没训练，没有保存的意义
                checkpoint = {
                    'model': raw_model.state_dict(), # 用 raw_model 而不是可能被 DDP/compile 包装过的 model，保存"干净"的原始权重
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only: # 兜底：配置成"只评估、不训练"（常用于调试），第一次评估完就直接退出，不进入下面的训练步骤
        break

    # 前向、反向、参数更新；可选地做梯度累积来模拟更大的批大小，
    # 并且在数据类型为 float16 时使用 GradScaler
    # (forward backward update, with optional gradient accumulation to simulate larger
    # batch size and using the GradScaler if data type is float16)
    for micro_step in range(gradient_accumulation_steps): # 依次处理每个"微批"，梯度会自动累加在 .grad 里，不会被清零
        if ddp: # 条件：多卡训练时，梯度同步（跨卡通信）很耗时，没必要每个微批都做一次
            # 在 DDP 训练中，我们只需要在最后一个微步（micro step）同步梯度。
            # 官方做法是使用 model.no_sync() 上下文管理器，但我实在不喜欢
            # 那样会让代码变臃肿、还迫使我们重复写代码；
            # 看了那个上下文管理器的源码，它其实就是切换这个变量而已
            # (in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable)
            # require_backward_grad_sync：只在最后一个微批设为 True，backward() 才会真正跨卡同步梯度，
            # 前面的微批设为 False，各卡只在本地累加梯度、不通信，省下大量不必要的跨卡流量
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx: # 混合精度上下文，前向传播在这里面执行
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # 对损失做缩放，以抵消梯度累积带来的影响 (scale the loss to account for gradient accumulation)：因为梯度会累加 gradient_accumulation_steps 次，提前除掉这个倍数，效果才等价于一次性用大 batch 算出的平均梯度
        # 在模型于 GPU 上做前向传播的同时，立刻异步预取下一个批次 (immediately async prefetch next batch while model is doing the forward pass on the GPU)
        X, Y = get_batch('train')
        # 反向传播，如果用 fp16 训练则配合梯度缩放 (backward pass, with gradient scaling if training in fp16)
        scaler.scale(loss).backward() # scaler.scale(loss)：先放大 loss 再反传，配合下面 scaler.step 里的自动缩小，防止 float16 梯度下溢（见文件头 GradScaler 名词解释）
    # 裁剪梯度 (clip the gradient)
    if grad_clip != 0.0: # 条件：配置开启了梯度裁剪（值为 0 表示不裁剪）
        scaler.unscale_(optimizer) # 裁剪前必须先把 GradScaler 放大过的梯度还原回真实尺度，否则裁剪阈值就失去意义
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # 把所有参数梯度的整体范数限制在 grad_clip 以内，防止个别 batch 梯度过大把参数"训飞"
    # 让优化器和 scaler 走一步（如果用 fp16 训练）(step the optimizer and scaler if training in fp16)
    scaler.step(optimizer) # 内部会检查梯度里有没有 inf/NaN，正常才真正调用 optimizer.step() 更新参数，否则跳过这一步
    scaler.update() # 根据这一步有没有出现数值溢出，动态调整下一步用来放大 loss 的缩放系数
    # 尽早清空梯度，这块内存不再需要了 (flush the gradients as soon as we can, no need for this memory anymore)
    optimizer.zero_grad(set_to_none=True) # set_to_none=True：把梯度直接设为 None 而不是清零成同尺寸的全 0 张量，省一次显存分配和写入，是官方推荐的更高效写法

    # 计时与日志 (timing and logging)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process: # 条件：到了打印间隔，且只让主进程打印，避免多卡重复刷屏
        # 把损失取成浮点数。注意：这里是一个 CPU-GPU 同步点
        # 乘回去以撤销上面的除法，从而近似出真正的总损失（严格来说应该是求和）
        # (get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss
        # (exact would have been a sum))
        # "CPU-GPU 同步点"：loss.item() 会强制等待 GPU 排队中的所有计算跑完、把结果拷回 CPU 才能拿到值，
        # 这会打断 CPU、GPU 原本各干各的、异步流水线式的执行节奏，所以不能每一步都调用，只在打印时才用
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # 让训练循环先稳定一会儿 (let the training loop settle a bit)：前几步涉及 GPU 预热/编译等开销，耗时不具代表性，跳过不纳入 MFU 统计
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt) # estimate_mfu 定义在 model.py，按模型结构估算理论算力需求，除以实际耗时得到利用率
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu # 指数滑动平均：新值权重 0.1，历史值权重 0.9，让打印出的 MFU 数值更平滑、不来回跳动
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # 终止条件 (termination conditions)
    if iter_num > max_iters: # 兜底：达到配置的总迭代次数，训练结束，跳出 while True 无限循环
        break

if ddp: # 条件：多卡训练收尾时，要把进程组资源释放掉，否则进程退出时可能残留僵尸通信资源
    destroy_process_group()
