import time

out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
wandb_log = False # 想开就开 (feel free to turn on)
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2-xl' # 这是最大的那个 GPT-2 模型 (this is the largest GPT-2 model)

# 只在验证损失有改善时才保存检查点 (only save checkpoints if the validation loss improves)
always_save_checkpoint = False

# 每次迭代的样本数量： (the number of examples per iter:)
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter（每次迭代 32,768 个 token）
# shakespeare 数据集有 301,966 个 token，所以 1 个 epoch ≈ 9.2 次迭代 (shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters)
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# 以恒定学习率（LR）进行微调 (finetune at constant LR)
learning_rate = 3e-5
decay_lr = False
