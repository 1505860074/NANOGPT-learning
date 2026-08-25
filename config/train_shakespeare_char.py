# 训练一个微型的字符级（character-level）莎士比亚模型 (train a miniature character-level shakespeare model)
# 适合在 macbook 之类的机器上调试和玩玩 (good for debugging and playing on macbooks and such)

out_dir = 'out-shakespeare-char'
eval_interval = 250 # 保持频繁评估，因为我们会过拟合 (keep frequent because we'll overfit)
eval_iters = 200
log_interval = 10 # 别打印得太太频繁 (don't print too too often)

# 我们预计会在这个小数据集上过拟合，所以只在验证损失改善时才保存 (we expect to overfit on this small dataset, so only save when val improves)
always_save_checkpoint = False

wandb_log = False # 如果你愿意，可以通过命令行覆盖 (override via command line if you like)
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # 上下文最多可回看前面 256 个字符 (context of up to 256 previous characters)

# 婴儿版 GPT 模型 :) (baby GPT model :))
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # 网络很小的时候，学习率可以适当调高一些 (with baby networks can afford to go a bit higher)
max_iters = 5000
lr_decay_iters = 5000 # 通常设成和 max_iters 相等 (make equal to max_iters usually)
min_lr = 1e-4 # 通常取 learning_rate / 10 (learning_rate / 10 usually)
beta2 = 0.99 # 调大一点，因为每次迭代的 token 数量很少 (make a bit bigger because number of tokens per iter is small)

warmup_iters = 100 # 可能不是特别必要 (not super necessary potentially)

# 在 macbook 上还要加上 (on macbook also add)
# device = 'cpu'  # 只在 cpu 上运行 (run on cpu only)
# compile = False # 不要用 torch 编译模型 (do not torch compile the model)
