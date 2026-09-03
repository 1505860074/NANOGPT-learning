"""
这个训练脚本既可以在单张 GPU 上以调试模式运行，
也可以用分布式数据并行（ddp）做更大规模的训练。

在单张 GPU 上运行的示例：
$ python train.py --batch_size=32 --compile=False

在 1 个节点的 4 张 GPU 上用 DDP 运行的示例：
$ torchrun --standalone --nproc_per_node=4 train.py

跨 2 个节点、共 4 张 GPU 用 DDP 运行的示例：
- 在第一个（主）节点上运行，示例 IP 为 123.456.123.456：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- 在工作节点上运行：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
（如果你的集群没有 Infiniband 互联，请在命令前加上 NCCL_IB_DISABLE=1）

--- 以下为英文原文 ---
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import contextlib
import numpy
import torch
import model

# -----------------------------------------------------------------------------
# 下面这些默认配置值，是为了在 OpenWebText 上训练一个 gpt2 (124M) 而设计的
# default config values designed to train a gpt2 (124M) on OpenWebText
# 输入输出（I/O）
output_directory = 'out'
"""检查点和日志的输出目录。"""
evaluation_interval = 2000
"""每隔多少次迭代做一次评估。"""
log_interval = 1
"""每隔多少次迭代打印一次训练日志。"""
evaluation_iterations = 200
"""每次评估取多少个批次求平均。"""
evaluation_only = False # if True, script exits right after the first eval
"""若为 True，脚本在第一次评估结束后立刻退出。"""
always_save_checkpoint = True # if True, always save a checkpoint after each eval
"""若为 True，每次评估后都保存检查点。"""
initialize_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
"""从哪里初始化模型：'scratch' 从零训练、'resume' 从检查点续训、'gpt2*' 加载 OpenAI 预训练权重。"""
# wandb 日志记录
# wandb logging
wandb_log = False # disabled by default
"""是否把训练指标上报到 wandb，默认关闭。"""
wandb_project = 'owt'
"""wandb 的项目名。"""
wandb_run_name = 'gpt2' # 'run' + str(time.time())
"""wandb 里这次运行的名字。"""
# 数据
# data
dataset = 'openwebtext'
"""数据集名称，对应 data/ 下的子目录名。"""
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
"""梯度累积次数，攒够这么多个微批才更新一次参数。"""
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
"""单个微批（micro-batch）的样本条数。"""
block_size = 1024
"""上下文长度，模型一次最多能看见多少个 token。"""
# 模型
# model
number_of_layers = 12
"""Transformer 的层数，也就是堆叠多少个 Block。"""
number_of_attention_heads = 12
"""每层的注意力头数。"""
embedding_dimension = 768
"""嵌入维度，也是模型的隐藏层宽度。"""
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
"""dropout 比率，预训练用 0 比较好，微调时可以试试 0.1 以上。"""
bias = False # do we use bias inside LayerNorm and Linear layers?
"""是否在 LayerNorm 和 Linear 层里使用偏置。"""
# adamw 优化器
# adamw optimizer
learning_rate = 6e-4 # max learning rate
"""最大学习率，也就是预热结束时达到的峰值。"""
maximum_iterations = 600000 # total number of training iterations
"""训练迭代的总次数，到这个数就停。"""
weight_decay = 1e-1
"""权重衰减系数，相当于 L2 正则化的强度。"""
beta1 = 0.9
"""AdamW 的一阶动量衰减率。"""
beta2 = 0.95
"""AdamW 的二阶动量衰减率。"""
gradient_clip_value = 1.0 # clip gradients at this value, or disable if == 0.0
"""梯度裁剪的阈值，等于 0.0 表示不裁剪。"""
# 学习率衰减相关设置
# learning rate decay settings
decay_learning_rate = True # whether to decay the learning rate
"""是否对学习率做衰减，False 则全程用固定学习率。"""
warmup_iterations = 2000 # how many steps to warm up for
"""学习率线性预热的步数。"""
learning_rate_decay_iterations = 600000 # should be ~= max_iters per Chinchilla
"""余弦衰减走完所需的步数，按 Chinchilla 的建议应该 ≈ maximum_iterations。"""
minimum_learning_rate = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
"""最小学习率，也就是余弦衰减的下限。"""
# DDP 相关设置
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
"""DDP 多卡之间通信用的后端。"""
# 系统
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
"""训练使用的设备。"""
data_type = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
"""训练用的浮点精度，float16 会自动启用 GradScaler。"""
compile = True # use PyTorch 2.0 to compile the model to be faster
"""是否用 PyTorch 2.0 编译模型以提速。"""
# -----------------------------------------------------------------------------
config_keys = [name for name, value in globals().items() if not name.startswith('_') and isinstance(value, (int, float, bool, str))]
"""上面所有超参数的名字，供命令行覆盖和日志记录时遍历。"""
exec(open('configurator.py').read()) # 从命令行或配置文件读取覆盖项
config = {name: globals()[name] for name in config_keys} # 之后记录日志时会用到
"""本次运行的超参数快照，会一起存进检查点、也会上报给 wandb。"""
# -----------------------------------------------------------------------------

# 各种初始化、派生属性，以及 I/O 的设置
# various inits, derived attributes, I/O setup
is_distributed_data_parallel = int(os.environ.get('RANK', -1)) != -1 # 这次是不是 ddp 运行？
"""本次是否以 DDP 多进程方式运行，靠 torchrun 设置的 RANK 环境变量判断。"""
if is_distributed_data_parallel:
    torch.distributed.init_process_group(backend=backend)
    distributed_data_parallel_rank = int(os.environ['RANK'])
    """本进程在所有进程里的全局编号，从 0 开始。"""
    distributed_data_parallel_local_rank = int(os.environ['LOCAL_RANK'])
    """本进程在本机内的编号，决定它用哪一张显卡。"""
    distributed_data_parallel_world_size = int(os.environ['WORLD_SIZE'])
    """参与本次训练的进程总数。"""
    device = f'cuda:{distributed_data_parallel_local_rank}'
    torch.cuda.set_device(device)
    master_process = distributed_data_parallel_rank == 0 # 由这个进程负责记日志、存检查点等工作
    """本进程是否为主进程，只有主进程负责记日志和存检查点。"""
    seed_offset = distributed_data_parallel_rank # 每个进程拿到不同的随机种子
    """随机种子的偏移量，让每个进程抽到不同的数据。"""
    # 会有 world_size 个进程同时训练，所以可以按比例调小每个进程需要的梯度累积次数
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % distributed_data_parallel_world_size == 0
    gradient_accumulation_steps //= distributed_data_parallel_world_size
else:
    # 如果不是 ddp，那就是在单张 gpu、单个进程上运行
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    distributed_data_parallel_world_size = 1
tokens_per_iteration = gradient_accumulation_steps * distributed_data_parallel_world_size * batch_size * block_size
"""每次迭代（也就是每更新一次参数）实际吃掉的 token 总数。"""
print(f"tokens per iteration will be: {tokens_per_iteration:,}")

if master_process:
    os.makedirs(output_directory, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # 矩阵乘法允许使用 tf32
torch.backends.cudnn.allow_tf32 = True # cudnn 允许使用 tf32
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后面 torch.autocast 使用
"""设备的大类，只区分 'cuda' 和 'cpu'。"""
# 注意：float16 这种数据类型会自动启用 GradScaler
# note: float16 data type will automatically use a GradScaler
pytorch_data_type = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[data_type]
"""data_type 这个字符串对应的 torch 数据类型对象。"""
autocast_context = contextlib.nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=pytorch_data_type)
"""混合精度的上下文管理器；在 CPU 上退化成什么都不做的空上下文。"""

# 穷人版数据加载器
# poor man's data loader
data_directory = os.path.join('data', dataset)
"""当前数据集所在的目录，里面放着 train.bin / val.bin。"""
def get_batch(split):
    # 每取一个批次都重新创建 numpy.memmap，以避免内存泄漏，依据是：
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = numpy.memmap(os.path.join(data_directory, 'train.bin'), dtype=numpy.uint16, mode='r')
    else:
        data = numpy.memmap(os.path.join(data_directory, 'val.bin'), dtype=numpy.uint16, mode='r')
    random_start_indices = torch.randint(len(data) - block_size, (batch_size,))
    input_batch = torch.stack([torch.from_numpy((data[start_index:start_index+block_size]).astype(numpy.int64)) for start_index in random_start_indices])
    target_batch = torch.stack([torch.from_numpy((data[start_index+1:start_index+1+block_size]).astype(numpy.int64)) for start_index in random_start_indices])
    if device_type == 'cuda':
        # 把输入和目标批次固定（pin）在内存里，这样就能异步地把它们搬到 GPU 上（non_blocking=True）
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        input_batch, target_batch = input_batch.pin_memory().to(device, non_blocking=True), target_batch.pin_memory().to(device, non_blocking=True)
    else:
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    return input_batch, target_batch

# 先在这里初始化这两个值，如果 initialize_from='resume'（即从检查点续训）会被覆盖
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iteration_number = 0
"""当前的全局迭代步数，从检查点续训时会被恢复成上次的值。"""
best_validation_loss = 1e9
"""至今为止见过的最好（最低）验证损失。"""

# 尝试从数据集里推导出词表大小
# attempt to derive vocab_size from the dataset
metadata_path = os.path.join(data_directory, 'meta.pkl')
"""数据集元信息文件 meta.pkl 的路径。"""
metadata_vocabulary_size = None
"""从 meta.pkl 读到的词表大小；数据集没提供时为 None。"""
if os.path.exists(metadata_path):
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        """meta.pkl 里的元信息字典，含词表大小和字符编解码表。"""
    metadata_vocabulary_size = metadata['vocab_size'] # 'vocab_size' 是 meta.pkl 的数据格式键名，不随变量改名而变
    print(f"found vocab_size = {metadata_vocabulary_size} (inside {metadata_path})")

# 模型初始化
# model init
model_arguments = dict(number_of_layers=number_of_layers, number_of_attention_heads=number_of_attention_heads,
                       embedding_dimension=embedding_dimension, block_size=block_size,
                       bias=bias, vocabulary_size=None, dropout=dropout) # 先用命令行传进来的参数作为起点
"""构造 GPTConfig 用的参数字典，同时也会原样存进检查点。"""
if initialize_from == 'scratch':
    # 从零开始初始化一个新模型
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # 确定从零训练时要用的词表大小
    # determine the vocab size we'll use for from-scratch training
    if metadata_vocabulary_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_arguments['vocabulary_size'] = metadata_vocabulary_size if metadata_vocabulary_size is not None else 50304
    gpt_config = model.GPTConfig(**model_arguments)
    """模型的结构配置对象。"""
    gpt_model = model.GPT(gpt_config)
    """GPT 模型本体；后面可能被 torch.compile 和 DDP 层层包装。"""
elif initialize_from == 'resume':
    print(f"Resuming training from {output_directory}")
    # 从一个检查点继续训练。
    # resume training from a checkpoint.
    checkpoint_path = os.path.join(output_directory, 'ckpt.pt')
    """要续训的检查点文件路径。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    """从磁盘读出来的检查点内容，含权重、优化器状态和超参数。"""
    # 老版本的检查点用的是缩写字段名（n_layer 等），这里统一翻译成现在的全称字段名。
    # 对已经是全称的新检查点，这个转换不会有任何影响。
    checkpoint_model_arguments = model.convert_legacy_model_arguments(checkpoint['model_args'])
    """检查点里记录的模型结构参数，字段名已翻译成全称。"""
    # 强制让这几个配置项保持一致，否则连续训都做不了
    # 其余属性（比如 dropout）则可以沿用命令行里想要的值
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for name in ['number_of_layers', 'number_of_attention_heads', 'embedding_dimension', 'block_size', 'bias', 'vocabulary_size']:
        model_arguments[name] = checkpoint_model_arguments[name]
    # 创建模型
    # create the model
    gpt_config = model.GPTConfig(**model_arguments)
    gpt_model = model.GPT(gpt_config)
    state_dictionary = checkpoint['model']
    """检查点里存的权重字典。"""
    # 修正状态字典里的键名 :(
    # 说实话我也不知道检查点为什么有时会带上这个前缀，还得再排查排查
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    """torch.compile 会给权重键名加上的前缀，加载前要剥掉。"""
    for key, value in list(state_dictionary.items()):
        if key.startswith(unwanted_prefix):
            state_dictionary[key[len(unwanted_prefix):]] = state_dictionary.pop(key)
    # 老检查点的权重键名同样是缩写形式，一并翻译成全称键名（新检查点不受影响）
    state_dictionary = model.convert_legacy_state_dictionary(state_dictionary)
    gpt_model.load_state_dict(state_dictionary)
    iteration_number = checkpoint['iter_num']
    best_validation_loss = checkpoint['best_val_loss']
elif initialize_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {initialize_from}")
    # 从 OpenAI 的 GPT-2 权重初始化
    # initialize from OpenAI GPT-2 weights
    override_arguments = dict(dropout=dropout)
    """要覆盖预训练模型默认配置的参数，这里只允许改 dropout。"""
    gpt_model = model.GPT.from_pretrained(initialize_from, override_arguments)
    # 把创建出来的配置参数读回来，这样才能正确地存进检查点
    # read off the created config params, so we can store them into checkpoint correctly
    for name in ['number_of_layers', 'number_of_attention_heads', 'embedding_dimension', 'block_size', 'bias', 'vocabulary_size']:
        model_arguments[name] = getattr(gpt_model.config, name)
# 如果需要，用"模型手术"的方式把模型的 block size 裁小
# crop down the model block size if desired, using model surgery
if block_size < gpt_model.config.block_size:
    gpt_model.crop_block_size(block_size)
    model_arguments['block_size'] = block_size # 这样检查点里存的才是正确的值
gpt_model.to(device)

# 初始化一个 GradScaler。如果 enabled=False，这个 scaler 就是个空操作
# initialize a GradScaler. If enabled=False scaler is a no-op
gradient_scaler = torch.cuda.amp.GradScaler(enabled=(data_type == 'float16'))
"""float16 训练用的梯度缩放器，把 loss 放大以免小梯度下溢；其他精度下是空操作。"""

# 优化器
# optimizer
optimizer = gpt_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
"""AdamW 优化器，二维参数做权重衰减、偏置和 LayerNorm 不做。"""
if initialize_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # 释放内存

# 编译模型
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_gpt_model = gpt_model
    """编译前的原始模型，留一份引用备用。"""
    gpt_model = torch.compile(gpt_model) # 需要 PyTorch 2.0

# 把模型包进 DDP 容器里
# wrap model into DDP container
if is_distributed_data_parallel:
    gpt_model = torch.nn.parallel.DistributedDataParallel(gpt_model, device_ids=[distributed_data_parallel_local_rank])

# 用很多个批次来估算训练集/验证集上的损失，想估多准就能估多准
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    output_losses = {}
    gpt_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evaluation_iterations)
        for batch_index in range(evaluation_iterations):
            input_batch, target_batch = get_batch(split)
            with autocast_context:
                logits, loss = gpt_model(input_batch, target_batch)
            losses[batch_index] = loss.item()
        output_losses[split] = losses.mean()
    gpt_model.train()
    return output_losses

# 学习率衰减调度器（带预热的余弦衰减）
# learning rate decay scheduler (cosine with warmup)
def get_learning_rate(iteration):
    # 1) 前 warmup_iterations 步做线性预热
    # 1) linear warmup for warmup_iters steps
    if iteration < warmup_iterations:
        return learning_rate * (iteration + 1) / (warmup_iterations + 1)
    # 2) 如果 iteration > learning_rate_decay_iterations，直接返回最小学习率
    # 2) if it > lr_decay_iters, return min learning rate
    if iteration > learning_rate_decay_iterations:
        return minimum_learning_rate
    # 3) 在这两者之间，用余弦衰减一路降到最小学习率
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iteration - warmup_iterations) / (learning_rate_decay_iterations - warmup_iterations)
    assert 0 <= decay_ratio <= 1
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coefficient 的取值范围是 0..1
    return minimum_learning_rate + coefficient * (learning_rate - minimum_learning_rate)

# 日志记录
# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# 训练主循环
# training loop
input_batch, target_batch = get_batch('train') # 取出最开始的第一个批次
iteration_start_time = time.time()
"""本次迭代的起始时刻，用来算每步耗时。"""
local_iteration_number = 0 # 本进程从启动到现在跑过的迭代次数
"""本进程自启动以来跑过的迭代次数，和 iteration_number 不同，续训不会累加。"""
raw_gpt_model = gpt_model.module if is_distributed_data_parallel else gpt_model # 需要的话，把 DDP 容器拆开取出原始模型
"""脱掉 DDP 外壳后的模型，存检查点和算 MFU 时要用它。"""
running_model_flops_utilization = -1.0
"""模型算力利用率（MFU）的滑动平均值，-1.0 表示还没开始统计。"""
while True:

    # 确定并设置本次迭代使用的学习率
    # determine and set the learning rate for this iteration
    current_learning_rate = get_learning_rate(iteration_number) if decay_learning_rate else learning_rate
    """本次迭代实际使用的学习率。"""
    for parameter_group in optimizer.param_groups:
        parameter_group['lr'] = current_learning_rate

    # 在训练集/验证集上评估损失，并写出检查点
    # evaluate the loss on train/val sets and write checkpoints
    if iteration_number % evaluation_interval == 0 and master_process:
        losses = estimate_loss()
        """训练集和验证集上的平均损失。"""
        print(f"step {iteration_number}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iteration_number,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": current_learning_rate,
                "mfu": running_model_flops_utilization*100, # 换算成百分比
            })
        if losses['val'] < best_validation_loss or always_save_checkpoint:
            best_validation_loss = losses['val']
            if iteration_number > 0:
                # 下面这些字符串键名是检查点的数据格式，保持不变以便新旧检查点互通
                checkpoint = {
                    'model': raw_gpt_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_arguments,
                    'iter_num': iteration_number,
                    'best_val_loss': best_validation_loss,
                    'config': config,
                }
                """准备写入磁盘的检查点内容。"""
                print(f"saving checkpoint to {output_directory}")
                torch.save(checkpoint, os.path.join(output_directory, 'ckpt.pt'))
    if iteration_number == 0 and evaluation_only:
        break

    # 前向、反向、更新参数；可选地用梯度累积来模拟更大的批大小，
    # 如果数据类型是 float16 还会用上 GradScaler
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if is_distributed_data_parallel:
            # 在 DDP 训练里，只需要在最后一个微步（micro step）同步梯度。
            # 官方做法是用 no_sync() 上下文管理器，但我实在不喜欢它
            # 让代码变得臃肿、还逼我们重复写代码；
            # 翻了下那个上下文管理器的源码，它其实就是在切换下面这个变量
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            gpt_model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with autocast_context:
            logits, loss = gpt_model(input_batch, target_batch)
            loss = loss / gradient_accumulation_steps # 对损失做缩放，以抵消梯度累积带来的放大
        # 趁模型正在 GPU 上做前向传播，立刻异步预取下一个批次
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        input_batch, target_batch = get_batch('train')
        # 反向传播；如果用 fp16 训练，还会做梯度缩放
        # backward pass, with gradient scaling if training in fp16
        gradient_scaler.scale(loss).backward()
    # 裁剪梯度
    # clip the gradient
    if gradient_clip_value != 0.0:
        gradient_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), gradient_clip_value)
    # 让优化器走一步；如果用 fp16 训练，scaler 也跟着走一步
    # step the optimizer and scaler if training in fp16
    gradient_scaler.step(optimizer)
    gradient_scaler.update()
    # 尽早清空梯度，这块内存已经用不上了
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # 计时与日志
    # timing and logging
    iteration_end_time = time.time()
    """本次迭代的结束时刻。"""
    elapsed_time = iteration_end_time - iteration_start_time
    """本次迭代耗时，单位是秒。"""
    iteration_start_time = iteration_end_time
    if iteration_number % log_interval == 0 and master_process:
        # 把 loss 取成 float。注意：这里是一个 CPU-GPU 同步点
        # 乘回去以抵消上面的除法，近似还原出真实的总损失（严格来说应该是求和）
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        loss_value = loss.item() * gradient_accumulation_steps
        """还原成真实尺度后的损失数值，仅用于打印。"""
        if local_iteration_number >= 5: # 先让训练循环稳定一小会儿
            model_flops_utilization = raw_gpt_model.estimate_model_flops_utilization(batch_size * gradient_accumulation_steps, elapsed_time)
            """本次迭代的模型算力利用率。"""
            running_model_flops_utilization = model_flops_utilization if running_model_flops_utilization == -1.0 else 0.9*running_model_flops_utilization + 0.1*model_flops_utilization
        print(f"iter {iteration_number}: loss {loss_value:.4f}, time {elapsed_time*1000:.2f}ms, mfu {running_model_flops_utilization*100:.2f}%")
    iteration_number += 1
    local_iteration_number += 1

    # 终止条件
    # termination conditions
    if iteration_number > maximum_iterations:
        break

if is_distributed_data_parallel:
    torch.distributed.destroy_process_group()
