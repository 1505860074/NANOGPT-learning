# 训练一个迷你的、字符级的莎士比亚模型
# 适合用来调试，也适合在 macbook 之类的机器上把玩
# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

OUTPUT_DIRECTORY = 'out-shakespeare-char'
EVALUATION_INTERVAL = 250 # 评估勤一点，因为我们会过拟合
EVALUATION_ITERATIONS = 200
LOG_INTERVAL = 10 # 别打印得太太频繁

# 在这么小的数据集上预计会过拟合，所以只在验证集变好时才保存
# we expect to overfit on this small dataset, so only save when val improves
ALWAYS_SAVE_CHECKPOINT = False

WANDB_LOG = False # 想开的话可以从命令行覆盖
WANDB_PROJECT = 'shakespeare-char'
WANDB_RUN_NAME = 'mini-gpt'

DATASET = 'shakespeare_char'
GRADIENT_ACCUMULATION_STEPS = 1
BATCH_SIZE = 64
BLOCK_SIZE = 256 # 上下文最多回看 256 个字符

# 婴儿版 GPT 模型 :)
# baby GPT model :)
NUMBER_OF_LAYERS = 6
NUMBER_OF_ATTENTION_HEADS = 6
EMBEDDING_DIMENSION = 384
DROPOUT = 0.2

LEARNING_RATE = 1e-3 # 小网络可以把学习率开高一点
MAXIMUM_ITERATIONS = 5000
LEARNING_RATE_DECAY_ITERATIONS = 5000 # 通常设成和 MAXIMUM_ITERATIONS 相等
MINIMUM_LEARNING_RATE = 1e-4 # 通常取 LEARNING_RATE / 10
BETA2 = 0.99 # 调大一点，因为每次迭代的 token 数很少

WARMUP_ITERATIONS = 100 # 可能不是特别必要

# 在 macbook 上还要加上
# on macbook also add
# DEVICE = 'cpu'  # 只用 cpu 跑
# COMPILE = False # 不要对模型做 torch compile
