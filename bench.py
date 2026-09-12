"""
train.py 的一个精简得多的版本，用于跑基准测试（benchmark）。

--- 以下为英文原文 ---
A much shorter version of train.py for benchmarking
"""
import os
# os（Operating System，操作系统）：Python 标准库，提供操作系统接口，本文件用到 os.path.join（拼路径）。
import contextlib
# contextlib（context + lib，上下文管理库）：Python 标准库，提供上下文管理器工具，本文件用 nullcontext() 占位。
import numpy
# numpy（Numerical Python，数值 Python）：Python 最核心的科学计算库，主打高性能多维数组（ndarray）运算。
#   全称常写作 NumPy，这是本文件第一个第三方库，也是深度学习所有张量计算的地基。
import time
# time：Python 标准库，提供与时间相关的函数，本文件用 time.time() 记录时刻来测速。
import torch
# torch：「PyTorch」的官方包名。开源深度学习框架，提供 GPU 张量计算与自动求导。
import model
# model：本仓库自己的模块（同目录下的 model.py），定义了 GPT 模型。

# -----------------------------------------------------------------------------
# 超参数（必须留在模块顶层：configurator.py 通过 globals() 覆盖这些值）
# -----------------------------------------------------------------------------
BATCH_SIZE = 12
"""每个批次的样本条数。"""
BLOCK_SIZE = 1024
"""上下文长度，模型一次能看见多少个 token。"""
BIAS = False
"""是否在 LayerNorm 和 Linear 层里使用偏置。"""
REAL_DATA = True
"""True 用真实的 openwebtext 数据，False 用随机张量（排除数据加载的干扰）。"""
SEED = 1337
"""随机种子。"""
DEVICE = 'cuda' # 例如：'cpu'、'cuda'、'cuda:0'、'cuda:1' 等
"""测速使用的设备。"""
DATA_TYPE = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 可选 'float32'、'bfloat16'、'float16'
"""测速用的浮点精度。"""
COMPILE = True # 用 PyTorch 2.0 编译模型以提速
"""是否用 PyTorch 2.0 编译模型。"""
PROFILE = False # 用 pytorch 性能分析器，还是只做简单的计时测速？
"""True 走 PyTorch 性能分析器，False 只做简单计时。"""
exec(open('configurator.py').read()) # 从命令行或配置文件读取覆盖项
# exec(代码字符串)：内建函数，把字符串当作代码执行；open(path).read() 读出 configurator.py 全文并运行之。
# -----------------------------------------------------------------------------


def setup_runtime() -> dict:
    """初始化随机种子、TF32 开关与混合精度上下文。

    输入: 无。
    输出: dict，含 device_type（str）与 autocast_context（上下文管理器）。
    """
    torch.manual_seed(SEED)
    # torch.manual_seed(n)：给 CPU 随机源下种子，固定随机序列以便复现。
    torch.cuda.manual_seed(SEED)
    # torch.cuda.manual_seed(n)：给 GPU 随机源下种子，与上一句配套。
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # torch.backends.cuda.* / cudnn.*：PyTorch 的后端开关命名空间；打开 TF32（半精度）让 GPU 矩阵运算更快。
    device_type = 'cuda' if 'cuda' in DEVICE else 'cpu' # 供 torch.autocast 使用
    pytorch_data_type = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[DATA_TYPE]
    # 字典取值：把字符串键换成对应的 torch 数据类型对象。
    autocast_context = contextlib.nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=pytorch_data_type)
    # torch.amp.autocast：混合精度上下文，进入后部分运算自动跑低精度加速。
    # contextlib.nullcontext()：空上下文占位，CPU 上不会有加速但仍保持代码结构统一。
    return {'device_type': device_type, 'autocast_context': autocast_context}


def make_batch_loader() -> callable:
    """构造批次加载器。

    REAL_DATA=True 时从 openwebtext 的 train.bin 采样；否则返回固定随机张量，排除数据加载对测速的干扰。

    输入: 无。
    输出: get_batch（Callable[[str], tuple[torch.Tensor, torch.Tensor]]）— 输入 split，返回 (input, target) 两个批次张量。
    """
    if REAL_DATA:
        dataset = 'openwebtext'
        data_directory = os.path.join('data', dataset)
        train_data = numpy.memmap(os.path.join(data_directory, 'train.bin'), dtype=numpy.uint16, mode='r')
        # numpy.memmap(path, dtype, mode)：内存映射数组——不把整个大文件读进内存，
        #   而是建立"磁盘文件 <-> 内存视图"的映射，用的时候像普通数组一样切片访问。
        #   dtype=numpy.uint16：元素类型是无符号 16 位整数（token 编号用 2 字节存，省内存）。
        #   mode='r'：只读模式（read）。
        def get_batch(split):
            data = train_data # 基准测试脚本里忽略 split 参数
            random_start_indices = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
            # torch.randint(上限, 形状)：生成给定形状的随机整数张量，值的范围是 [0, 上限)。
            #   len(data)：序列长度（=随机取样时不允许取到末尾，所以减掉 BLOCK_SIZE）。
            input_batch = torch.stack([torch.from_numpy((data[start_index:start_index+BLOCK_SIZE]).astype(numpy.int64)) for start_index in random_start_indices])
            target_batch = torch.stack([torch.from_numpy((data[start_index+1:start_index+1+BLOCK_SIZE]).astype(numpy.int64)) for start_index in random_start_indices])
            # torch.from_numpy(ndarray)：把 NumPy 数组转成 PyTorch 张量。
            # .astype(numpy.int64)：把数组元素类型转成 64 位有符号整数（模型计算需要）。
            # torch.stack(张量列表)：把一列张量沿"新维度"堆叠起来——这里把 BATCH_SIZE 个样本拼成 (批, 序列) 一块。
            # 列表推导式 [表达式 for ...]：一行生成列表的写法，这里分别给每批样本采一段连续的上下文。
            input_batch, target_batch = input_batch.pin_memory().to(DEVICE, non_blocking=True), target_batch.pin_memory().to(DEVICE, non_blocking=True)
            # .pin_memory()：把 CPU 张量固定在"页锁定内存"中，好让 CPU→GPU 的搬运走异步快通道。
            # .to(DEVICE, non_blocking=True)：搬到 GPU；non_blocking=True 表示异步搬运（不阻塞 CPU 干别的），
            #   前提是已经 pin 过。target 比 input 右移一位：让模型用当前位置预测下一个 token。
            return input_batch, target_batch
    else:
        # 用固定随机数据，不受数据加载速度影响
        input_batch = torch.randint(50304, (BATCH_SIZE, BLOCK_SIZE), device=DEVICE)
        target_batch = torch.randint(50304, (BATCH_SIZE, BLOCK_SIZE), device=DEVICE)
        get_batch = lambda split: (input_batch, target_batch)
        # lambda split: ...：匿名函数，忽略入参直接返回固定张量。
    return get_batch


def build_model() -> dict:
    """构建 GPT 模型与优化器，搬到 DEVICE，并按需 torch.compile。

    输入: 无。
    输出: dict，含 gpt_model（被编译后的模型）与 optimizer（AdamW）。
    """
    gpt_config = model.GPTConfig(
        block_size = BLOCK_SIZE,
        number_of_layers = 12, number_of_attention_heads = 12, embedding_dimension = 768, # 模型规模
        dropout = 0, # 为了保证结果可复现
        bias = BIAS,
    )
    gpt_model = model.GPT(gpt_config)
    gpt_model.to(DEVICE)

    # device_type 由 setup_runtime 提供，这里直接从 DEVICE 推导（与 setup_runtime 保持一致）
    device_type = 'cuda' if 'cuda' in DEVICE else 'cpu'
    optimizer = gpt_model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)
    # configure_optimizers：model.GPT 里自定义的方法，返回一个配好的 AdamW 优化器。

    if COMPILE:
        print("Compiling model...")
        gpt_model = torch.compile(gpt_model) # pytorch 2.0
        # torch.compile(model)：PyTorch 2.0 即时编译器，把模型优化成更快的执行图（编译期约一分钟）。
    return {'gpt_model': gpt_model, 'optimizer': optimizer}


def run_profile_benchmark(gpt_model: model.GPT, get_batch: callable, optimizer: torch.optim.Optimizer, autocast_context: contextlib.AbstractContextManager):
    """用 PyTorch 性能分析器测速，把 trace 写到 ./bench_log。

    输入:
        gpt_model（model.GPT）— 待测模型。
        get_batch（Callable[[str], tuple]）— 批次加载器。
        optimizer（torch.optim.Optimizer）— 优化器。
        autocast_context（contextlib.AbstractContextManager）— 混合精度上下文。
    输出: 无（打印每步 loss，输出性能分析 trace）。
    """
    # 性能分析器文档：https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    wait, warmup, active = 5, 5, 5
    number_of_steps = wait + warmup + active
    with torch.profiler.profile(
        # torch.profiler.profile(...)：PyTorch 性能分析器（profiler=分析器），进入 with 块后统计每一步的耗时。
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        # activities：要分析的硬件，CPU 与 GPU 都分析。
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        # schedule：时间表——先等 wait 步、热身 warmup 步（结果丢弃）、再正式收集 active 步。
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        # on_trace_ready：收集完成后把结果存成 TensorBoard 可读的 trace，放在 ./bench_log。
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=True,
        # with_flops=True：顺便估算每次运算的浮点次数（FLOPs）。
        with_modules=False,
    ) as profiler:
        input_batch, target_batch = get_batch('train')
        for step in range(number_of_steps):
            with autocast_context:
                logits, loss = gpt_model(input_batch, target_batch)
            input_batch, target_batch = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            # optimizer.zero_grad()：清空上一次反向传播留下的梯度（不清会累积）。
            #   set_to_none=True：用置 None 的方式清，比置零省内存、更快。
            loss.backward()
            # loss.backward()：反向传播，自动算网络里每个参数的梯度。
            optimizer.step()
            # optimizer.step()：用刚算出的梯度更新一次模型权重。
            loss_value = loss.item()
            # .item()：把只有一个数的张量取出来变成普通 Python 数（loss.item() 通常打印用）。
            print(f"{step}/{number_of_steps} loss: {loss_value:.4f}")
            # f-string 里 :.4f 表示保留 4 位小数。
            profiler.step() # 每一步结束通知性能分析器


def run_simple_benchmark(gpt_model: model.GPT, get_batch: callable, optimizer: torch.optim.Optimizer, autocast_context: contextlib.AbstractContextManager):
    """简单计时测速：先预热再正式测，最后打印每迭代耗时与 MFU。

    输入:
        gpt_model（model.GPT）— 待测模型。
        get_batch（Callable[[str], tuple]）— 批次加载器。
        optimizer（torch.optim.Optimizer）— 优化器。
        autocast_context（contextlib.AbstractContextManager）— 混合精度上下文。
    输出: 无（打印每步 loss、每迭代耗时与 MFU）。
    """
    torch.cuda.synchronize()
    # torch.cuda.synchronize()：强制等 GPU 上所有异步任务全部跑完（CPU/GPU 之间的"对表"），
    #   这样后面的计时才准确（不会把 GPU 还没算完就提前计时）。
    for stage, number_of_steps in enumerate([10, 20]): # 先预热（burnin），再正式测速
        # enumerate(列表)：返回 (下标, 元素) 对的迭代器。这里 [10,20] 表示第一阶段跑 10 步（预热）、第二阶段 20 步（正式）。
        iteration_start_time = time.time()
        # time.time()：返回当前时刻（1970 年以来的秒数），两次取差就得到耗时。
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
        model_flops_utilization = gpt_model.estimate_model_flops_utilization(BATCH_SIZE * 1 * number_of_steps, elapsed_time)
        # estimate_model_flops_utilization：model.GPT 里自定义的方法，估算模型算力利用率（MFU，实际/理论峰值）。
        if stage == 1:
            print(f"time per iteration: {elapsed_time/number_of_steps*1000:.4f}ms, MFU: {model_flops_utilization*100:.2f}%")


def main():
    """基准测试主流程：初始化环境 → 加载数据 → 构建模型 → 按 PROFILE 选择测速方式。"""
    # ---- 1. 初始化运行环境（随机种子 + 混合精度） ----
    runtime = setup_runtime()

    # ---- 2. 数据加载初始化 ----
    get_batch = make_batch_loader()

    # ---- 3. 构建模型与优化器 ----
    built = build_model()
    gpt_model, optimizer = built['gpt_model'], built['optimizer']

    # ---- 4. 执行测速 ----
    if PROFILE:
        run_profile_benchmark(gpt_model, get_batch, optimizer, runtime['autocast_context'])
    else:
        run_simple_benchmark(gpt_model, get_batch, optimizer, runtime['autocast_context'])


if __name__ == '__main__':
    # __name__ == '__main__'：直接运行时才执行 main()；被 import 时不执行（脚本/库两用的标准写法）。
    main()