# 评估基础版 gpt2 (evaluate the base gpt2)
# n_layer=24, n_head=16, n_embd=1024
# 350M 参数 (350M parameters)
batch_size = 8
eval_iters = 500 # 用更多迭代次数以得到较好的估计 (use more iterations to get good estimate)
eval_only = True
wandb_log = False
init_from = 'gpt2-medium'
