"""
GPT 语言模型的完整定义，全部都在这一个文件里。
参考资料：
1) OpenAI 官方发布的 GPT-2 TensorFlow 实现：
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers 的 PyTorch 实现：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm，但 bias（偏置）是可选的。PyTorch 不支持简单地写 bias=False
    (LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False) """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 所有注意力头的 key、query、value 投影，但打包成一个批量算 (key, query, value projections for all heads, but in a batch)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影 (output projection)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 正则化 (regularization)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention 能让 GPU 火力全开，但只有 PyTorch >= 2.0 才支持 (flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 因果掩码，确保注意力只作用于输入序列中当前位置左侧的内容 (causal mask to ensure that attention is only applied to the left in the input sequence)
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # 批大小、序列长度、嵌入维度 (batch size, sequence length, embedding dimensionality (n_embd))

        # 批量计算所有注意力头的 query、key、value，并把 head 维度前移，使其成为批维度 (calculate query, key, values for all heads in batch and move head forward to be the batch dim)
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 因果自注意力；自注意力计算：(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T) (causal self-attention; Self-attend)
        if self.flash:
            # 使用 Flash Attention 的 CUDA 核心做高效注意力计算 (efficient attention using Flash Attention CUDA kernels)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # 注意力的手工实现版本 (manual implementation of attention)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 把所有注意力头的输出并排重新拼装起来 (re-assemble all head outputs side by side)

        # 输出投影 (output projection)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 的 vocab_size 是 50257，为了效率向上填充到最近的 64 的倍数 (GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True：像 GPT-2 那样在 Linear 和 LayerNorm 里带偏置。False：效果稍好一点、也更快 (True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 在使用权重绑定（weight tying）并配合 torch.compile() 时会产生一些警告：
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # （即：functional_call 收到了绑定权重的多个取值，该行为已废弃，未来版本会报错）
        # 不能 100% 确定这是怎么回事，目前看来无害。TODO 待研究
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # 初始化所有权重 (init all weights)
        self.apply(self._init_weights)
        # 按照 GPT-2 论文的做法，对残差投影层施加特殊的缩放初始化 (apply special scaled init to the residual projections, per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # 报告参数数量 (report number of parameters)
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        返回模型中的参数数量。
        对于「不含嵌入层」的计数方式（默认），会减去位置嵌入（position embeddings）的参数。
        token 嵌入本来也该减去，但由于参数共享（weight tying），这些参数实际上被当作
        最后一层的权重在使用，所以我们把它们计入。
        (Return the number of parameters in the model. For non-embedding count (default),
        the position embeddings get subtracted. The token embeddings would too, except due
        to the parameter sharing these params are actually used as weights in the final
        layer, so we include them.)
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # 形状为 (t) (shape (t))

        # 前向传播 GPT 模型本身 (forward the GPT model itself)
        tok_emb = self.transformer.wte(idx) # token 嵌入，形状为 (b, t, n_embd) (token embeddings of shape (b, t, n_embd))
        pos_emb = self.transformer.wpe(pos) # 位置嵌入，形状为 (t, n_embd) (position embeddings of shape (t, n_embd))
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 如果传入了期望的目标值（targets），就顺便计算损失 (if we are given some desired targets also calculate the loss)
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段的小优化：只对最后一个位置跑 lm_head (inference-time mini-optimization: only forward the lm_head on the very last position)
            logits = self.lm_head(x[:, [-1], :]) # 注意：这里用列表 [-1] 是为了保留时间维度 (note: using list [-1] to preserve the time dim)
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # 必要时通过「模型手术」来减小 block size（上下文长度）
        # 例如我们可能加载了 GPT2 的预训练检查点（block size 为 1024），
        # 但想在某个更小、更简单的模型上使用更小的 block size
        # (model surgery to decrease the block size if necessary; e.g. we may load the GPT2
        # pretrained model checkpoint (block size 1024) but want to use a smaller block size
        # for some smaller, simpler model)
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # 默认为空字典 (default to empty dict)
        # 只有 dropout 可以被覆盖，更多说明见下文 (only dropout can be overridden see more notes below)
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer、n_head 和 n_embd 由 model_type 决定 (n_layer, n_head and n_embd are determined from model_type)
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M 参数 (124M params)
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M 参数 (350M params)
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M 参数 (774M params)
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M 参数 (1558M params)
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # GPT 模型检查点里始终是 50257 (always 50257 for GPT model checkpoints)
        config_args['block_size'] = 1024 # GPT 模型检查点里始终是 1024 (always 1024 for GPT model checkpoints)
        config_args['bias'] = True # GPT 模型检查点里始终为 True (always True for GPT model checkpoints)
        # 如果需要，我们可以覆盖 dropout 比率 (we can override the dropout rate, if desired)
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # 创建一个从零初始化的 minGPT 模型 (create a from-scratch initialized minGPT model)
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # 丢弃这个掩码/缓冲区，它不是参数 (discard this mask / buffer, not a param)

        # 初始化一个 huggingface/transformers 模型 (init a huggingface/transformers model)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 拷贝参数，同时确保所有参数在名称和形状上都能一一对齐 (copy while ensuring all of the parameters are aligned and match in names and shapes)
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # 忽略这些，只是缓冲区 (ignore these, just a buffer)
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # 同上，只是掩码（缓冲区）(same, just the mask (buffer))
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 简单说，openai 的检查点用的是 "Conv1D" 模块，而我们只想用普通的 Linear，
        # 这意味着导入这些权重时必须做转置
        # (basically the openai checkpoints use a "Conv1D" module, but we only want to use a
        # vanilla Linear; this means that we have to transpose these weights when we import them)
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 对需要转置的 Conv1D 权重做特殊处理 (special treatment for the Conv1D weights we need to transpose)
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 其余参数直接普通拷贝 (vanilla copy over the other parameters)
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 先从所有候选参数开始 (start with all of the candidate parameters)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉那些不需要梯度的参数 (filter out those that do not require grad)
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 创建优化器参数组。所有 2 维（及以上）的参数都会做权重衰减，其余不做。
        # 也就是说：矩阵乘法里的所有权重张量 + 嵌入层会衰减，所有偏置和 layernorm 参数不衰减。
        # (create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # 创建 AdamW 优化器，如果 fused（融合）版本可用就使用它 (Create AdamW optimizer and use the fused version if it is available)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ 估算模型算力利用率（MFU, model flops utilization），以 A100 的 bfloat16 峰值 FLOPS 为单位
        (estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS) """
        # 首先估算每次迭代我们做了多少 flops (first estimate the number of flops we do per iteration.)
        # 参见 PaLM 论文附录 B：https://arxiv.org/abs/2204.02311 (see PaLM paper Appendix B as ref)
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # 把我们的 flops 吞吐量表示为 A100 bfloat16 峰值 flops 的比例 (express our flops throughput as ratio of A100 bfloat16 peak flops)
        flops_achieved = flops_per_iter * (1.0/dt) # 每秒 (per second)
        flops_promised = 312e12 # A100 GPU 的 bfloat16 峰值算力是 312 TFLOPS (A100 GPU bfloat16 peak flops is 312 TFLOPS)
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        接收一个作为条件的索引序列 idx（形状为 (b,t) 的 LongTensor），
        并把这个序列续写 max_new_tokens 次，每次都把预测结果重新喂回模型。
        使用本方法时，你大概会想确保模型处于 model.eval() 模式下。
        (Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.)
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文变得太长，必须把它裁剪到 block_size (if the sequence context is growing too long we must crop it at block_size)
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 前向传播模型，得到序列中该索引对应的 logits (forward the model to get the logits for the index in the sequence)
            logits, _ = self(idx_cond)
            # 取出最后一步的 logits，并按所需的温度（temperature）做缩放 (pluck the logits at the final step and scale by desired temperature)
            logits = logits[:, -1, :] / temperature
            # 可选地把 logits 裁剪为只保留概率最高的 k 个选项 (optionally crop the logits to only the top k options)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 应用 softmax，把 logits 转换成（归一化的）概率 (apply softmax to convert logits to (normalized) probabilities)
            probs = F.softmax(logits, dim=-1)
            # 从该分布中采样 (sample from the distribution)
            idx_next = torch.multinomial(probs, num_samples=1)
            # 把采样到的索引追加到当前序列后面，然后继续 (append sampled index to the running sequence and continue)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
