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
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 下面这些默认配置值，是为了在 OpenWebText 上训练一个 gpt2 (124M) 而设计的
# default config values designed to train a gpt2 (124M) on OpenWebText
# 输入输出（I/O）
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # 若为 True，脚本在第一次评估结束后立刻退出
always_save_checkpoint = True # 若为 True，每次评估后都保存一个检查点
init_from = 'scratch' # 可选 'scratch'（从零开始）、'resume'（续训）或 'gpt2*'
# wandb 日志记录
# wandb logging
wandb_log = False # 默认关闭
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# 数据
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # 用来模拟更大的批大小
batch_size = 12 # 若 gradient_accumulation_steps > 1，这里指的是微批（micro-batch）大小
block_size = 1024
# 模型
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # 预训练用 0 比较好，微调时可以试试 0.1 以上
bias = False # 是否在 LayerNorm 和 Linear 层里使用偏置？
# adamw 优化器
# adamw optimizer
learning_rate = 6e-4 # 最大学习率
max_iters = 600000 # 训练迭代的总次数
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # 把梯度裁剪到这个值；等于 0.0 表示不裁剪
# 学习率衰减相关设置
# learning rate decay settings
decay_lr = True # 是否对学习率做衰减
warmup_iters = 2000 # 预热（warmup）多少步
lr_decay_iters = 600000 # 按 Chinchilla 的建议，应该 ≈ max_iters
min_lr = 6e-5 # 最小学习率，按 Chinchilla 的建议应该 ≈ learning_rate/10
# DDP 相关设置
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# 系统
# system
device = 'cuda' # 例如：'cpu'、'cuda'、'cuda:0'、'cuda:1' 等；macbook 上可以试试 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 可选 'float32'、'bfloat16' 或 'float16'，最后一个会自动启用 GradScaler
compile = True # 用 PyTorch 2.0 编译模型以提速
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # 从命令行或配置文件读取覆盖项
config = {k: globals()[k] for k in config_keys} # 之后记录日志时会用到
# -----------------------------------------------------------------------------

# 各种初始化、派生属性，以及 I/O 的设置
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # 这次是不是 ddp 运行？
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # 由这个进程负责记日志、存检查点等工作
    seed_offset = ddp_rank # 每个进程拿到不同的随机种子
    # 会有 world_size 个进程同时训练，所以可以按比例调小每个进程需要的梯度累积次数
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # 如果不是 ddp，那就是在单张 gpu、单个进程上运行
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # 矩阵乘法允许使用 tf32
torch.backends.cudnn.allow_tf32 = True # cudnn 允许使用 tf32
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后面 torch.autocast 使用
# 注意：float16 这种数据类型会自动启用 GradScaler
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 穷人版数据加载器
# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # 每取一个批次都重新创建 np.memmap，以避免内存泄漏，依据是：
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # 把数组 x、y 固定（pin）在内存里，这样就能异步地把它们搬到 GPU 上（non_blocking=True）
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# 先在这里初始化这两个值，如果 init_from='resume'（即从检查点续训）会被覆盖
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# 尝试从数据集里推导出 vocab_size（词表大小）
# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# 模型初始化
# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # 先用命令行传进来的 model_args 作为起点
if init_from == 'scratch':
    # 从零开始初始化一个新模型
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # 确定从零训练时要用的词表大小
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # 从一个检查点继续训练。
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # 强制让这几个配置项保持一致，否则连续训都做不了
    # 其余属性（比如 dropout）则可以沿用命令行里想要的值
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # 创建模型
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 修正状态字典里的键名 :(
    # 说实话我也不知道检查点为什么有时会带上这个前缀，还得再排查排查
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # 从 OpenAI 的 GPT-2 权重初始化
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # 把创建出来的配置参数读回来，这样才能正确地存进检查点
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# 如果需要，用"模型手术"的方式把模型的 block size 裁小
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # 这样检查点里存的才是正确的值
model.to(device)

# 初始化一个 GradScaler。如果 enabled=False，这个 scaler 就是个空操作
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 优化器
# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # 释放内存

# 编译模型
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # 需要 PyTorch 2.0

# 把模型包进 DDP 容器里
# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 用很多个批次来估算训练集/验证集上的损失，想估多准就能估多准
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 学习率衰减调度器（带预热的余弦衰减）
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) 前 warmup_iters 步做线性预热
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) 如果 it > lr_decay_iters，直接返回最小学习率
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) 在这两者之间，用余弦衰减一路降到最小学习率
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff 的取值范围是 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# 日志记录
# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# 训练主循环
# training loop
X, Y = get_batch('train') # 取出最开始的第一个批次
t0 = time.time()
local_iter_num = 0 # 本进程从启动到现在跑过的迭代次数
raw_model = model.module if ddp else model # 需要的话，把 DDP 容器拆开取出原始模型
running_mfu = -1.0
while True:

    # 确定并设置本次迭代使用的学习率
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 在训练集/验证集上评估损失，并写出检查点
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # 换算成百分比
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # 前向、反向、更新参数；可选地用梯度累积来模拟更大的批大小，
    # 如果数据类型是 float16 还会用上 GradScaler
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # 在 DDP 训练里，只需要在最后一个微步（micro step）同步梯度。
            # 官方做法是用 model.no_sync() 上下文管理器，但我实在不喜欢它
            # 让代码变得臃肿、还逼我们重复写代码；
            # 翻了下那个上下文管理器的源码，它其实就是在切换下面这个变量
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # 对损失做缩放，以抵消梯度累积带来的放大
        # 趁模型正在 GPU 上做前向传播，立刻异步预取下一个批次
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # 反向传播；如果用 fp16 训练，还会做梯度缩放
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # 裁剪梯度
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 让优化器走一步；如果用 fp16 训练，scaler 也跟着走一步
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # 尽早清空梯度，这块内存已经用不上了
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # 计时与日志
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # 把 loss 取成 float。注意：这里是一个 CPU-GPU 同步点
        # 乘回去以抵消上面的除法，近似还原出真实的总损失（严格来说应该是求和）
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # 先让训练循环稳定一小会儿
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # 终止条件
    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
