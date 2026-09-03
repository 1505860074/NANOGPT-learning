# 评测原始的 gpt2 模型
# evaluate the base gpt2
# number_of_layers=12, number_of_attention_heads=12, embedding_dimension=768
# 124M 参数量
# 124M parameters
batch_size = 8
evaluation_iterations = 500 # 多跑一些迭代，估计得更准
evaluation_only = True
wandb_log = False
initialize_from = 'gpt2'
