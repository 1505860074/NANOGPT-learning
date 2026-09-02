# 训练一个迷你的、字符级的莎士比亚模型
# 适合用来调试，也适合在 macbook 之类的机器上把玩
# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # 评估勤一点，因为我们会过拟合
eval_iters = 200
log_interval = 10 # 别打印得太太频繁

# 在这么小的数据集上预计会过拟合，所以只在验证集变好时才保存
# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # 想开的话可以从命令行覆盖
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # 上下文最多回看 256 个字符

# 婴儿版 GPT 模型 :)
# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # 小网络可以把学习率开高一点
max_iters = 5000
lr_decay_iters = 5000 # 通常设成和 max_iters 相等
min_lr = 1e-4 # 通常取 learning_rate / 10
beta2 = 0.99 # 调大一点，因为每次迭代的 token 数很少

warmup_iters = 100 # 可能不是特别必要

# 在 macbook 上还要加上
# on macbook also add
# device = 'cpu'  # 只用 cpu 跑
# compile = False # 不要对模型做 torch compile
