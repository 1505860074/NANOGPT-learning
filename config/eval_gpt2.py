# 评测原始的 gpt2 模型
# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M 参数量
# 124M parameters
batch_size = 8
eval_iters = 500 # 多跑一些迭代，估计得更准
eval_only = True
wandb_log = False
init_from = 'gpt2'
