"""
从一个训练好的模型中采样（生成文本）
(Sample from a trained model)
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # 可以是 'resume'（从某个 out_dir 恢复）或某个 gpt2 变体（例如 'gpt2-xl'）(either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl'))
out_dir = 'out' # 如果 init_from 不是 'resume'，此项被忽略 (ignored if init_from is not 'resume')
start = "\n" # 也可以是 "<|endoftext|>" 等等。还可以指定一个文件，用法为："FILE:prompt.txt" (or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt")
num_samples = 10 # 要采样生成多少个样本 (number of samples to draw)
max_new_tokens = 500 # 每个样本中生成的 token 数量 (number of tokens generated in each sample)
temperature = 0.8 # 1.0 = 不改变，< 1.0 = 预测更保守（随机性更低），> 1.0 = 预测更随机 (1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions)
top_k = 200 # 只保留概率最高的 top_k 个 token，其余的概率强制归零 (retain only the top_k most likely tokens, clamp others to have 0 probability)
seed = 1337
device = 'cuda' # 例如：'cpu'、'cuda'、'cuda:0'、'cuda:1' 等等 (examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.)
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' 或 'bfloat16' 或 'float16'
compile = False # 使用 PyTorch 2.0 编译模型以提速 (use PyTorch 2.0 to compile the model to be faster)
exec(open('configurator.py').read()) # 来自命令行或配置文件的覆盖项 (overrides from command line or config file)
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # 允许矩阵乘法使用 tf32 (allow tf32 on matmul)
torch.backends.cudnn.allow_tf32 = True # 允许 cudnn 使用 tf32 (allow tf32 on cudnn)
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后面 torch.autocast 使用 (for later use in torch.autocast)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 模型 (model)
if init_from == 'resume':
    # 从保存在指定目录中的模型初始化 (init from a model saved in a specific directory)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # 从指定的 GPT-2 模型初始化 (init from a given GPT-2 model)
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # 需要 PyTorch 2.0（可选）(requires PyTorch 2.0 (optional))

# 查找 meta pickle 文件，以防它在数据集目录里是可用的 (look for the meta pickle in case it is available in the dataset folder)
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # 更早的检查点可能没有这些字段…… (older checkpoints might not have these...)
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO 想把这里做得更通用，以支持任意的编码器/解码器方案 (TODO want to make this more general to arbitrary encoder/decoder schemes)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # 好吧，那就默认假定使用 gpt-2 的编码方式 (ok let's assume gpt-2 encodings by default)
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# 对提示词（prompt）的开头部分做编码 (encode the beginning of the prompt)
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# 运行生成 (run generation)
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
