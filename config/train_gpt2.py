# 用于训练 GPT-2 (124M) 的配置，在 1 个节点、8 张 A100 40GB 上可把损失降到很不错的 ~2.85
# (config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB)
# 按如下方式启动（比如放在一个 screen 会话里），然后等大约 5 天：
# (launch as the following (e.g. in a screen session) and wait ~5 days:)
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# 这些设置让总批大小达到约 0.5M (these make the total batch size be ~0.5M)
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520（批大小 × 上下文长度 × 梯度累积 × GPU 数）
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# 这让 token 总量达到 300B（3000 亿）(this makes total number of tokens be 300B)
max_iters = 600000
lr_decay_iters = 600000

# 评估相关设置 (eval stuff)
eval_interval = 1000
eval_iters = 200
log_interval = 10

# 权重衰减 (weight decay)
weight_decay = 1e-1
