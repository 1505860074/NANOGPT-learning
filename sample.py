"""
从训练好的模型中采样生成文本

--- 以下为英文原文 ---
Sample from a trained model
"""
import os
import pickle
import contextlib
import torch
import tiktoken
import model

# -----------------------------------------------------------------------------
INITIALIZE_FROM = 'resume' # 取 'resume'（从 output_directory 恢复）或某个 gpt2 变体（如 'gpt2-xl'）
"""从哪里加载模型：'resume' 读本地检查点，或 'gpt2' / 'gpt2-xl' 等加载 OpenAI 预训练权重。"""
OUTPUT_DIRECTORY = 'out' # 若 initialize_from 不是 'resume' 则忽略此项
"""存放 ckpt.pt 的目录。"""
START = "\n" # 也可以是 "<|endoftext|>" 等。还能指定文件，写成："FILE:prompt.txt"
"""生成起始提示词的配置值；写成 "FILE:xxx.txt" 则从该文件读取内容。"""
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
# -----------------------------------------------------------------------------

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True # 矩阵乘法允许使用 tf32
torch.backends.cudnn.allow_tf32 = True # cudnn 允许使用 tf32
device_type = 'cuda' if 'cuda' in DEVICE else 'cpu' # 供后面 torch.autocast 使用
"""设备的大类，只区分 'cuda' 和 'cpu'。"""
pytorch_data_type = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[DATA_TYPE]
"""DATA_TYPE 这个字符串对应的 torch 数据类型对象。"""
autocast_context = contextlib.nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=pytorch_data_type)
"""混合精度的上下文管理器；在 CPU 上退化成空上下文。"""

# 模型
# model
if INITIALIZE_FROM == 'resume':
    # 从保存在指定目录下的模型初始化
    # init from a model saved in a specific directory
    checkpoint_path = os.path.join(OUTPUT_DIRECTORY, 'ckpt.pt')
    """检查点文件的路径。"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    """从磁盘读出来的检查点内容。"""
    # 用检查点里记录的模型结构参数重建配置
    gpt_config = model.GPTConfig(**checkpoint['model_args'])
    gpt_model = model.GPT(gpt_config)
    """GPT 模型本体。"""
    state_dictionary = checkpoint['model']
    """检查点里存的权重字典。"""
    unwanted_prefix = '_orig_mod.'
    """torch.compile 会给权重键名加上的前缀，加载前要剥掉。"""
    for key, value in list(state_dictionary.items()):
        if key.startswith(unwanted_prefix):
            state_dictionary[key[len(unwanted_prefix):]] = state_dictionary.pop(key)
    gpt_model.load_state_dict(state_dictionary)
elif INITIALIZE_FROM.startswith('gpt2'):
    # 从指定的 GPT-2 模型初始化
    # init from a given GPT-2 model
    gpt_model = model.GPT.from_pretrained(INITIALIZE_FROM, dict(dropout=0.0))

gpt_model.eval()
gpt_model.to(DEVICE)
if COMPILE:
    gpt_model = torch.compile(gpt_model) # 需要 PyTorch 2.0（可选）

# 到数据集目录里找找有没有 meta 这个 pickle 文件
# look for the meta pickle in case it is available in the dataset folder
load_metadata = False
"""是否找到了数据集的 meta.pkl（字符级模型才有）。"""
if INITIALIZE_FROM == 'resume' and 'config' in checkpoint: # 老一些的检查点可能没有这些字段……
    # 检查点里 config 的键名：本项目现在存的是大写的 'DATASET'，早期版本存的是小写的
    # 'dataset'，这里两种都认，好让老检查点仍然能找到自己的 meta.pkl
    checkpoint_config = checkpoint['config']
    """检查点里记录的那次运行的超参数快照。"""
    checkpoint_dataset = checkpoint_config.get('DATASET', checkpoint_config.get('dataset'))
    """检查点对应的数据集名；新旧两种键名都尝试过。"""
    if checkpoint_dataset is not None:
        metadata_path = os.path.join('data', checkpoint_dataset, 'meta.pkl')
        load_metadata = os.path.exists(metadata_path)
if load_metadata:
    print(f"Loading meta from {metadata_path}...")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    # TODO 想把这里做得更通用，以支持任意的编码器/解码器方案
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    # 'stoi'/'itos' 是 meta.pkl 的数据格式键名，不随变量改名而变
    string_to_integer, integer_to_string = metadata['stoi'], metadata['itos']
    encode = lambda string: [string_to_integer[character] for character in string]
    decode = lambda token_list: ''.join([integer_to_string[token_id] for token_id in token_list])
else:
    # 好吧，那就默认按 gpt-2 的编码方式来
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    encoder = tiktoken.get_encoding("gpt2")
    """GPT-2 的 BPE 分词器。"""
    encode = lambda string: encoder.encode(string, allowed_special={"<|endoftext|>"})
    decode = lambda token_list: encoder.decode(token_list)

# 对提示词（prompt）的开头部分做编码
# encode the beginning of the prompt
# 这是运行时变量：START 指向文件时，真正的提示词要从文件里读出来
start_text = START
"""运行时实际使用的提示词文本。"""
if START.startswith('FILE:'):
    with open(START[5:], 'r', encoding='utf-8') as f:
        start_text = f.read()
start_token_ids = encode(start_text)
"""起始提示词编码后的 token 序列。"""
input_token_indices = (torch.tensor(start_token_ids, dtype=torch.long, device=DEVICE)[None, ...])
"""喂给模型的输入张量，前面补了一个批维度。"""

# 执行生成
# run generation
with torch.no_grad():
    with autocast_context:
        for sample_index in range(NUMBER_OF_SAMPLES):
            generated_indices = gpt_model.generate(input_token_indices, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
            print(decode(generated_indices[0].tolist()))
            print('---------------')
