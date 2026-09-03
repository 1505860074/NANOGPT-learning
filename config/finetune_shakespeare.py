import time

OUTPUT_DIRECTORY = 'out-shakespeare'
EVALUATION_INTERVAL = 5
EVALUATION_ITERATIONS = 40
WANDB_LOG = False # 想开就开
WANDB_PROJECT = 'shakespeare'
WANDB_RUN_NAME = 'ft-' + str(time.time())

DATASET = 'shakespeare'
INITIALIZE_FROM = 'gpt2-xl' # 这是最大的那个 GPT-2 模型

# 只在验证损失变好时才保存检查点
# only save checkpoints if the validation loss improves
ALWAYS_SAVE_CHECKPOINT = False

# 每次迭代处理的样本量：
# 1 BATCH_SIZE * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# 莎士比亚数据集共 301,966 个 token，所以 1 个 epoch ≈ 9.2 次迭代
# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
MAXIMUM_ITERATIONS = 20

# 用恒定学习率做微调
# finetune at constant LR
LEARNING_RATE = 3e-5
DECAY_LEARNING_RATE = False
