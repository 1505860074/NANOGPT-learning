"""
train.py 的一个精简得多的版本，用于跑基准测试（benchmark）

--- 以下为英文原文 ---
A much shorter version of train.py for benchmarking
"""
import os
import contextlib
import numpy
import time
import torch
import model

# -----------------------------------------------------------------------------
batch_size = 12
"""每个批次的样本条数。"""
block_size = 1024
"""上下文长度，模型一次能看见多少个 token。"""
bias = False
"""是否在 LayerNorm 和 Linear 层里使用偏置。"""
real_data = True
"""True 用真实的 openwebtext 数据，False 用随机张量（排除数据加载的干扰）。"""
seed = 1337
"""随机种子。"""
device = 'cuda' # 例如：'cpu'、'cuda'、'cuda:0'、'cuda:1' 等
"""测速使用的设备。"""
data_type = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 可选 'float32'、'bfloat16'、'float16'
"""测速用的浮点精度。"""
compile = True # 用 PyTorch 2.0 编译模型以提速
"""是否用 PyTorch 2.0 编译模型。"""
profile = False # 用 pytorch 性能分析器，还是只做简单的计时测速？
"""True 走 PyTorch 性能分析器，False 只做简单计时。"""
exec(open('configurator.py').read()) # 从命令行或配置文件读取覆盖项
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # 矩阵乘法允许使用 tf32
torch.backends.cudnn.allow_tf32 = True # cudnn 允许使用 tf32
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后面 torch.autocast 使用
"""设备的大类，只区分 'cuda' 和 'cpu'。"""
pytorch_data_type = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[data_type]
"""data_type 这个字符串对应的 torch 数据类型对象。"""
autocast_context = contextlib.nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=pytorch_data_type)
"""混合精度的上下文管理器；在 CPU 上退化成空上下文。"""

# 数据加载的初始化
# data loading init
if real_data:
    dataset = 'openwebtext'
    data_directory = os.path.join('data', dataset)
    train_data = numpy.memmap(os.path.join(data_directory, 'train.bin'), dtype=numpy.uint16, mode='r')
    """以内存映射方式打开的训练集 token 数组。"""
    def get_batch(split):
        data = train_data # 注意：基准测试脚本里忽略 split 参数
        random_start_indices = torch.randint(len(data) - block_size, (batch_size,))
        input_batch = torch.stack([torch.from_numpy((data[start_index:start_index+block_size]).astype(numpy.int64)) for start_index in random_start_indices])
        target_batch = torch.stack([torch.from_numpy((data[start_index+1:start_index+1+block_size]).astype(numpy.int64)) for start_index in random_start_indices])
        input_batch, target_batch = input_batch.pin_memory().to(device, non_blocking=True), target_batch.pin_memory().to(device, non_blocking=True)
        return input_batch, target_batch
else:
    # 或者，如果希望用固定数据、不想被数据加载干扰，就走这条分支
    # alternatively, if fixed data is desired to not care about data loading
    input_batch = torch.randint(50304, (batch_size, block_size), device=device)
    target_batch = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (input_batch, target_batch)

# 模型初始化
# model init
gpt_config = model.GPTConfig(
    block_size = block_size, # 模型能往回看多远？也就是上下文长度
    number_of_layers = 12, number_of_attention_heads = 12, embedding_dimension = 768, # 模型规模
    dropout = 0, # 为了保证结果可复现
    bias = bias,
)
gpt_model = model.GPT(gpt_config)
"""GPT 模型本体。"""
gpt_model.to(device)

optimizer = gpt_model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)
"""AdamW 优化器。"""

if compile:
    print("Compiling model...")
    gpt_model = torch.compile(gpt_model) # pytorch 2.0

if profile:
    # pytorch 性能分析器的实用文档：
    # useful docs on pytorch profiler:
    # - 教程 https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - API https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5
    number_of_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False, # 会带来额外开销，不需要就关掉
        with_flops=True,
        with_modules=False, # 目前仅对 torchscript 模型有效
    ) as profiler:

        input_batch, target_batch = get_batch('train')
        for step in range(number_of_steps):
            with autocast_context:
                logits, loss = gpt_model(input_batch, target_batch)
            input_batch, target_batch = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            print(f"{step}/{number_of_steps} loss: {loss_value:.4f}")

            profiler.step() # 每一步结束时通知性能分析器

else:

    # 简单的计时测速
    # simple benchmarking
    torch.cuda.synchronize()
    for stage, number_of_steps in enumerate([10, 20]): # 先预热（burnin），再正式测速
        iteration_start_time = time.time()
        input_batch, target_batch = get_batch('train')
        for step in range(number_of_steps):
            with autocast_context:
                logits, loss = gpt_model(input_batch, target_batch)
            input_batch, target_batch = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            print(f"{step}/{number_of_steps} loss: {loss_value:.4f}")
        torch.cuda.synchronize()
        iteration_end_time = time.time()
        elapsed_time = iteration_end_time - iteration_start_time
        model_flops_utilization = gpt_model.estimate_model_flops_utilization(batch_size * 1 * number_of_steps, elapsed_time)
        if stage == 1:
            print(f"time per iteration: {elapsed_time/number_of_steps*1000:.4f}ms, MFU: {model_flops_utilization*100:.2f}%")
