import time

output_directory = 'out-shakespeare'
evaluation_interval = 5
evaluation_iterations = 40
wandb_log = False # 想开就开
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'
initialize_from = 'gpt2-xl' # 这是最大的那个 GPT-2 模型

# 只在验证损失变好时才保存检查点
# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# 每次迭代处理的样本量：
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# 莎士比亚数据集共 301,966 个 token，所以 1 个 epoch ≈ 9.2 次迭代
# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
maximum_iterations = 20

# 用恒定学习率做微调
# finetune at constant LR
learning_rate = 3e-5
decay_learning_rate = False
