"""
train.py 的一个精简得多的版本，用于性能基准测试（benchmarking）
(A much shorter version of train.py for benchmarking)
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
batch_size = 12
block_size = 1024
bias = False
real_data = True
seed = 1337
device = 'cuda' # 例如：'cpu'、'cuda'、'cuda:0'、'cuda:1' 等等 (examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.)
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' 或 'bfloat16' 或 'float16'
compile = True # 使用 PyTorch 2.0 编译模型以提速 (use PyTorch 2.0 to compile the model to be faster)
profile = False # 使用 pytorch 性能分析器（profiler），还是只做简单的基准测试？ (use pytorch profiler, or just simple benchmarking?)
exec(open('configurator.py').read()) # 来自命令行或配置文件的覆盖项 (overrides from command line or config file)
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # 允许矩阵乘法使用 tf32 (allow tf32 on matmul)
torch.backends.cudnn.allow_tf32 = True # 允许 cudnn 使用 tf32 (allow tf32 on cudnn)
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后面 torch.autocast 使用 (for later use in torch.autocast)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 数据加载初始化 (data loading init)
if real_data:
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data # 注意：基准测试脚本里忽略 split 参数 (note ignore split in benchmarking script)
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
else:
    # 另一种做法：如果想用固定数据、从而不关心数据加载的开销 (alternatively, if fixed data is desired to not care about data loading)
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

# 模型初始化 (model init)
gptconf = GPTConfig(
    block_size = block_size, # 模型能往回看多远？也就是上下文长度 (how far back does the model look? i.e. context size)
    n_layer = 12, n_head = 12, n_embd = 768, # 模型的规模 (size of the model)
    dropout = 0, # 为了保证确定性（可复现）(for determinism)
    bias = bias,
)
model = GPT(gptconf)
model.to(device)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

if compile:
    print("Compiling model...")
    model = torch.compile(model) # 需要 pytorch 2.0

if profile:
    # 关于 pytorch 性能分析器的有用文档： (useful docs on pytorch profiler:)
    # - 教程 https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - API 文档 https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False, # 会带来额外开销，不需要就关掉 (incurs an additional overhead, disable if not needed)
        with_flops=True,
        with_modules=False, # 目前仅适用于 torchscript 模型 (only for torchscript models atm)
    ) as prof:

        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

            prof.step() # 在每一步结束时通知性能分析器 (notify the profiler at end of each step)

else:

    # 简单的基准测试 (simple benchmarking)
    torch.cuda.synchronize()
    for stage, num_steps in enumerate([10, 20]): # 先预热（burnin），然后才真正测速 (burnin, then benchmark)
        t0 = time.time()
        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1-t0
        mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
        if stage == 1:
            print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
