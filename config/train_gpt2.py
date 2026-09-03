# 在 1 个节点 8 张 A100 40GB 上训练 GPT-2 (124M)，可把损失降到 ~2.85 这一相当不错的水平
# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# 用下面的命令启动（比如放在 screen 会话里），然后等大约 5 天：
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

WANDB_LOG = True
WANDB_PROJECT = 'owt'
WANDB_RUN_NAME='gpt2-124M'

# 这几个值让总批大小约为 0.5M
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
BATCH_SIZE = 12
BLOCK_SIZE = 1024
GRADIENT_ACCUMULATION_STEPS = 5 * 8

# 这让总的 token 数达到 300B
# this makes total number of tokens be 300B
MAXIMUM_ITERATIONS = 600000
LEARNING_RATE_DECAY_ITERATIONS = 600000

# 评估相关
# eval stuff
EVALUATION_INTERVAL = 1000
EVALUATION_ITERATIONS = 200
LOG_INTERVAL = 10

# 权重衰减
# weight decay
WEIGHT_DECAY = 1e-1
