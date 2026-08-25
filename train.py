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
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # 来自命令行或配置文件的覆盖项 (overrides from command line or config file)
config = {k: globals()[k] for k in config_keys} # 对日志记录会有用 (will be useful for logging)
# -----------------------------------------------------------------------------

# 各种初始化、派生属性、I/O 设置 (various inits, derived attributes, I/O setup)
ddp = int(os.environ.get('RANK', -1)) != -1 # 这是一次 dd1·p 运行吗？ (is this a ddp run?)
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # 这个进程负责日志记录、保存检查点等工作 (this process will do logging, checkpointing etc.)
    seed_offset = ddp_rank # 每个进程拿到不同的随机种子 (each process gets a different seed)
    # 会有 world_size 个进程同时训练，所以我们可以按比例把
    # 每个进程期望的梯度累积迭代次数等比缩小
    # (world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally)
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # 如果不是 ddp，那我们就是在单张 GPU、单个进程上运行 (if not ddp, we are running on a single gpu, and one process)
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
# 在gpu运算上的性能优化
torch.backends.cuda.matmul.allow_tf32 = True # 允许矩阵乘法使用 tf32 (allow tf32 on matmul)
torch.backends.cudnn.allow_tf32 = True # 允许 cudnn 使用 tf32 (allow tf32 on cudnn)

device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后面 torch.autocast 使用 (for later use in torch.autocast)
# 注意：float16 数据类型会自动使用 GradScaler (note: float16 data type will automatically use a GradScaler)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 穷人版数据加载器 (poor man's data loader)
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # 我们每个批次都重新创建 np.memmap，以避免内存泄漏，依据如下讨论：
    # (We recreate np.memmap every batch to avoid a memory leak, as per)
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # 把数组 x、y 固定（pin）在内存中，这样我们就能异步地把它们搬到 GPU 上（non_blocking=True）(pin arrays x,y, which allows us to move them to GPU asynchronously)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# 先在这里初始化这些变量，如果 init_from='resume'（即从检查点恢复）可以被覆盖 (init these up here, can override if init_from='resume' (i.e. from a checkpoint))
iter_num = 0
best_val_loss = 1e9

# 尝试从数据集中推导出 vocab_size（词表大小）(attempt to derive vocab_size from the dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# 模型初始化 (model init)
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # 先用命令行传入的 model_args 作为起点 (start with model_args from command line)
if init_from == 'scratch':
    # 从零初始化一个新模型 (init a new model from scratch)
    print("Initializing a new model from scratch")
    # 确定从零训练时我们要用的词表大小 (determine the vocab size we'll use for from-scratch training)
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # 从一个检查点恢复训练。(resume training from a checkpoint.)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # 强制让这些配置属性保持一致，否则根本没法恢复训练；
    # 其余属性（比如 dropout）可以保留命令行里想要的值
    # (force these config attributes to be equal otherwise we can't even resume training;
    # the rest of the attributes (e.g. dropout) can stay as desired from command line)
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
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # 从 OpenAI 的 GPT-2 权重初始化 (initialize from OpenAI GPT-2 weights)
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # 读出创建好的配置参数，这样我们才能正确地把它们存入检查点 (read off the created config params, so we can store them into checkpoint correctly)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# 如果需要，用「模型手术」把模型的 block size 裁小 (crop down the model block size if desired, using model surgery)
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # 这样检查点里存的才是正确的值 (so that the checkpoint will have the right value)
model.to(device)

# 初始化一个 GradScaler（梯度缩放器）。如果 enabled=False，scaler 就是个空操作 (initialize a GradScaler. If enabled=False scaler is a no-op)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 优化器 (optimizer)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # 释放内存 (free up memory)

# 编译模型 (compile the model)
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # 需要 PyTorch 2.0 (requires PyTorch 2.0)

# 把模型包进 DDP 容器 (wrap model into DDP container)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 借助多个批次，对训练集或验证集估算出一个任意精度的损失值 (helps estimate an arbitrarily accurate loss over either split using many batches)
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

# 学习率衰减调度器（带预热的余弦衰减）(learning rate decay scheduler (cosine with warmup))
def get_lr(it):
    # 1) 前 warmup_iters 步做线性预热 (linear warmup for warmup_iters steps)
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) 如果 it > lr_decay_iters，返回最小学习率 (if it > lr_decay_iters, return min learning rate)
    if it > lr_decay_iters:
        return min_lr
    # 3) 介于两者之间时，使用余弦衰减一直降到最小学习率 (in between, use cosine decay down to min learning rate)
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff 的取值范围是 0..1 (coeff ranges 0..1)
    return min_lr + coeff * (learning_rate - min_lr)

# 日志记录 (logging)
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# 训练循环 (training loop)
X, Y = get_batch('train') # 取出最开始的第一个批次 (fetch the very first batch)
t0 = time.time()
local_iter_num = 0 # 本进程生命周期内的迭代次数 (number of iterations in the lifetime of this process)
raw_model = model.module if ddp else model # 如有需要，从 DDP 容器里把模型解包出来 (unwrap DDP container if needed)
running_mfu = -1.0
while True:

    # 确定并设置本次迭代的学习率 (determine and set the learning rate for this iteration)
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 在训练集/验证集上评估损失，并写出检查点 (evaluate the loss on train/val sets and write checkpoints)
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # 转换成百分比 (convert to percentage)
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

    # 前向、反向、参数更新；可选地做梯度累积来模拟更大的批大小，
    # 并且在数据类型为 float16 时使用 GradScaler
    # (forward backward update, with optional gradient accumulation to simulate larger
    # batch size and using the GradScaler if data type is float16)
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # 在 DDP 训练中，我们只需要在最后一个微步（micro step）同步梯度。
            # 官方做法是使用 model.no_sync() 上下文管理器，但我实在不喜欢
            # 那样会让代码变臃肿、还迫使我们重复写代码；
            # 看了那个上下文管理器的源码，它其实就是切换这个变量而已
            # (in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable)
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # 对损失做缩放，以抵消梯度累积带来的影响 (scale the loss to account for gradient accumulation)
        # 在模型于 GPU 上做前向传播的同时，立刻异步预取下一个批次 (immediately async prefetch next batch while model is doing the forward pass on the GPU)
        X, Y = get_batch('train')
        # 反向传播，如果用 fp16 训练则配合梯度缩放 (backward pass, with gradient scaling if training in fp16)
        scaler.scale(loss).backward()
    # 裁剪梯度 (clip the gradient)
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 让优化器和 scaler 走一步（如果用 fp16 训练）(step the optimizer and scaler if training in fp16)
    scaler.step(optimizer)
    scaler.update()
    # 尽早清空梯度，这块内存不再需要了 (flush the gradients as soon as we can, no need for this memory anymore)
    optimizer.zero_grad(set_to_none=True)

    # 计时与日志 (timing and logging)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # 把损失取成浮点数。注意：这里是一个 CPU-GPU 同步点
        # 乘回去以撤销上面的除法，从而近似出真正的总损失（严格来说应该是求和）
        # (get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss
        # (exact would have been a sum))
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # 让训练循环先稳定一会儿 (let the training loop settle a bit)
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # 终止条件 (termination conditions)
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
