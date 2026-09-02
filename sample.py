"""
从训练好的模型中采样生成文本

--- 以下为英文原文 ---
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # 取 'resume'（从 out_dir 恢复）或某个 gpt2 变体（如 'gpt2-xl'）
out_dir = 'out' # 若 init_from 不是 'resume' 则忽略此项
start = "\n" # 也可以是 "<|endoftext|>" 等。还能指定文件，写成："FILE:prompt.txt"
num_samples = 10 # 采样生成的样本条数
max_new_tokens = 500 # 每条样本生成的 token 数
temperature = 0.8 # 1.0 = 不改变预测分布，< 1.0 = 更确定，> 1.0 = 更随机
top_k = 200 # 只保留概率最高的 top_k 个 token，其余的概率钳成 0
seed = 1337
device = 'cuda' # 例如：'cpu'、'cuda'、'cuda:0'、'cuda:1' 等
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 可选 'float32'、'bfloat16'、'float16'
compile = False # 用 PyTorch 2.0 编译模型以提速
exec(open('configurator.py').read()) # 从命令行或配置文件读取覆盖项
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # 矩阵乘法允许使用 tf32
torch.backends.cudnn.allow_tf32 = True # cudnn 允许使用 tf32
device_type = 'cuda' if 'cuda' in device else 'cpu' # 供后面 torch.autocast 使用
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 模型
# model
if init_from == 'resume':
    # 从保存在指定目录下的模型初始化
    # init from a model saved in a specific directory
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
    # 从指定的 GPT-2 模型初始化
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # 需要 PyTorch 2.0（可选）

# 到数据集目录里找找有没有 meta 这个 pickle 文件
# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # 老一些的检查点可能没有这些字段……
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO 想把这里做得更通用，以支持任意的编码器/解码器方案
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # 好吧，那就默认按 gpt-2 的编码方式来
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# 对提示词（prompt）的开头部分做编码
# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# 执行生成
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
