"""
GPT 语言模型的完整定义，全部内容都在这一个文件里。
参考资料：
1) OpenAI 官方发布的 GPT-2 TensorFlow 实现：
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers 的 PyTorch 实现：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

--- 以下为英文原文 ---
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
import dataclasses

import torch

# 本项目把模块属性名和配置字段名都改成了完整全称，而 HuggingFace 的 GPT-2 权重、
# 以及本项目早期版本存下的检查点，用的都是缩写形式的旧键名。下面两张映射表负责在
# 加载权重时把旧键名翻译成现在的全称键名，从而保住向后兼容。

# 旧的模块属性名（也就是权重字典里的键名分段） -> 现在的全称
LEGACY_TO_CURRENT_MODULE_NAMES = {
    'wte':     'word_token_embedding',
    'wpe':     'word_position_embedding',
    'drop':    'embedding_dropout',
    'h':       'blocks',
    'ln_f':    'final_layer_normalization',
    'ln_1':    'layer_normalization_before_attention',
    'ln_2':    'layer_normalization_before_multi_layer_perceptron',
    'attn':    'attention',
    'mlp':     'multi_layer_perceptron',
    'c_attn':  'combined_query_key_value_projection',
    'c_fc':    'expansion_projection',
    'c_proj':  'output_projection',
    'lm_head': 'language_model_head',
}

# 旧的配置字段名 -> 现在的全称（检查点里的 model_args 用的是旧名）
LEGACY_TO_CURRENT_CONFIG_FIELDS = {
    'vocab_size': 'vocabulary_size',
    'n_layer':    'number_of_layers',
    'n_head':     'number_of_attention_heads',
    'n_embd':     'embedding_dimension',
}

def convert_legacy_state_dictionary_key(legacy_key):
    """把一个旧的权重键名翻译成现在的全称键名，例如
    'transformer.h.0.attn.c_attn.weight'
        -> 'transformer.blocks.0.attention.combined_query_key_value_projection.weight'
    """
    parts = legacy_key.split('.')
    # 注意力模块里那个叫 bias 的其实是因果掩码缓冲区，和 LayerNorm 的偏置参数重名了，
    # 所以要先单独处理它，避免把 LayerNorm 的 bias 也误改掉
    if len(parts) >= 2 and parts[-1] == 'bias' and parts[-2] == 'attn':
        parts[-1] = 'causal_mask'
    return '.'.join(LEGACY_TO_CURRENT_MODULE_NAMES.get(part, part) for part in parts)

def convert_legacy_state_dictionary(legacy_state_dictionary):
    """把整份旧权重字典的键名批量翻译成全称键名。"""
    return {convert_legacy_state_dictionary_key(key): value
            for key, value in legacy_state_dictionary.items()}

def convert_legacy_model_arguments(legacy_model_arguments):
    """把检查点里旧的 model_args 字段名翻译成 GPTConfig 现在的全称字段名。"""
    return {LEGACY_TO_CURRENT_CONFIG_FIELDS.get(name, name): value
            for name, value in legacy_model_arguments.items()}

class LayerNorm(torch.nn.Module):
    """
    LayerNorm，但偏置（bias）是可选的。PyTorch 不支持简单地写 bias=False

    --- 以下为英文原文 ---
    LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False
    """

    def __init__(self, number_of_dimensions, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(number_of_dimensions))
        """缩放参数，逐维放缩归一化之后的数值。"""
        self.bias = torch.nn.Parameter(torch.zeros(number_of_dimensions)) if bias else None
        """平移参数；bias=False 时为 None。"""

    def forward(self, input_tensor):
        return torch.nn.functional.layer_norm(input_tensor, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.embedding_dimension % config.number_of_attention_heads == 0
        # 所有注意力头的 key、query、value 投影，但打包成一次批量计算
        # key, query, value projections for all heads, but in a batch
        self.combined_query_key_value_projection = torch.nn.Linear(config.embedding_dimension, 3 * config.embedding_dimension, bias=config.bias)
        """把 query、key、value 三个投影打包成一次矩阵乘法，所以输出宽度是嵌入维度的 3 倍。"""
        # 输出投影
        # output projection
        self.output_projection = torch.nn.Linear(config.embedding_dimension, config.embedding_dimension, bias=config.bias)
        """注意力的输出投影，把多头拼接后的结果映射回嵌入维度。"""
        # 正则化
        # regularization
        self.attention_dropout = torch.nn.Dropout(config.dropout)
        """作用在注意力权重上的 dropout。"""
        self.residual_dropout = torch.nn.Dropout(config.dropout)
        """作用在输出投影结果上的 dropout。"""
        self.number_of_attention_heads = config.number_of_attention_heads
        """注意力头数。"""
        self.embedding_dimension = config.embedding_dimension
        """嵌入维度。"""
        self.dropout_probability = config.dropout
        """dropout 比率，会作为参数传给 Flash Attention。"""
        # flash attention 能让 GPU 火力全开，但只有 PyTorch >= 2.0 才支持
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.use_flash_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        """当前 PyTorch 版本是否支持 Flash Attention。"""
        if not self.use_flash_attention:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 因果掩码，确保注意力只作用于输入序列中当前位置左边的内容
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        batch_size, sequence_length, embedding_dimension = x.size() # 批大小、序列长度、嵌入维度

        # 批量计算所有注意力头的 query、key、value，并把 head 维前移，使其变成批维度
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = self.combined_query_key_value_projection(x).split(self.embedding_dimension, dim=2)
        head_dimension = embedding_dimension // self.number_of_attention_heads
        key   = key.view(batch_size, sequence_length, self.number_of_attention_heads, head_dimension).transpose(1, 2)   # (批, 头数, 序列, 每头维度)
        query = query.view(batch_size, sequence_length, self.number_of_attention_heads, head_dimension).transpose(1, 2) # (批, 头数, 序列, 每头维度)
        value = value.view(batch_size, sequence_length, self.number_of_attention_heads, head_dimension).transpose(1, 2) # (批, 头数, 序列, 每头维度)

        # 因果自注意力；自注意力计算：(批, 头数, 序列, 每头维度) x (批, 头数, 每头维度, 序列) -> (批, 头数, 序列, 序列)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.use_flash_attention:
            # 用 Flash Attention 的 CUDA 核函数做高效注意力计算
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.dropout_probability if self.training else 0, is_causal=True)
        else:
            # 手工实现的注意力计算
            # manual implementation of attention
            attention_weights = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
            attention_weights = attention_weights.masked_fill(self.causal_mask[:,:,:sequence_length,:sequence_length] == 0, float('-inf'))
            attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)
            attention_weights = self.attention_dropout(attention_weights)
            y = attention_weights @ value # (批, 头数, 序列, 序列) x (批, 头数, 序列, 每头维度) -> (批, 头数, 序列, 每头维度)
        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dimension) # 把所有头的输出并排拼回去

        # 输出投影
        # output projection
        y = self.residual_dropout(self.output_projection(y))
        return y

class MLP(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.expansion_projection = torch.nn.Linear(config.embedding_dimension, 4 * config.embedding_dimension, bias=config.bias)
        """升维投影，把嵌入维度放大到 4 倍。"""
        self.gelu_activation      = torch.nn.GELU()
        """GELU 激活函数，Block 里的非线性主要由它提供。"""
        self.output_projection    = torch.nn.Linear(4 * config.embedding_dimension, config.embedding_dimension, bias=config.bias)
        """降维投影，把 4 倍宽度压回嵌入维度。"""
        self.dropout_layer        = torch.nn.Dropout(config.dropout)
        """作用在 MLP 输出上的 dropout。"""

    def forward(self, x):
        x = self.expansion_projection(x)
        x = self.gelu_activation(x)
        x = self.output_projection(x)
        x = self.dropout_layer(x)
        return x

class Block(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer_normalization_before_attention = LayerNorm(config.embedding_dimension, bias=config.bias)
        """进注意力子层之前的层归一化（Pre-LN 的做法）。"""
        self.attention = CausalSelfAttention(config)
        """注意力子层，负责在词与词之间搬运信息。"""
        self.layer_normalization_before_multi_layer_perceptron = LayerNorm(config.embedding_dimension, bias=config.bias)
        """进 MLP 子层之前的层归一化。"""
        self.multi_layer_perceptron = MLP(config)
        """MLP 子层，负责逐个位置地把信息加工深。"""

    def forward(self, x):
        x = x + self.attention(self.layer_normalization_before_attention(x))
        x = x + self.multi_layer_perceptron(self.layer_normalization_before_multi_layer_perceptron(x))
        return x

@dataclasses.dataclass
class GPTConfig:
    block_size: int = 1024
    """上下文长度，模型一次最多能看见多少个 token。"""
    vocabulary_size: int = 50304 # GPT-2 词表大小本为 50257，为效率补齐到最近的 64 的倍数
    """词表大小，也就是输出层要预测多少个候选 token。"""
    number_of_layers: int = 12
    """Transformer 的层数，也就是堆叠多少个 Block。"""
    number_of_attention_heads: int = 12
    """每层的注意力头数，必须能整除 embedding_dimension。"""
    embedding_dimension: int = 768
    """嵌入维度，也是模型的隐藏层宽度。"""
    dropout: float = 0.0
    """dropout 比率。"""
    bias: bool = True # True：像 GPT-2 那样在 Linear 和 LayerNorm 里带偏置；False：效果略好且更快
    """是否在 Linear 和 LayerNorm 里使用偏置。"""

class GPT(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocabulary_size is not None
        assert config.block_size is not None
        self.config = config
        """模型的结构配置对象。"""

        self.transformer = torch.nn.ModuleDict(dict(
            word_token_embedding = torch.nn.Embedding(config.vocabulary_size, config.embedding_dimension),
            word_position_embedding = torch.nn.Embedding(config.block_size, config.embedding_dimension),
            embedding_dropout = torch.nn.Dropout(config.dropout),
            blocks = torch.nn.ModuleList([Block(config) for _ in range(config.number_of_layers)]),
            final_layer_normalization = LayerNorm(config.embedding_dimension, bias=config.bias),
        ))
        self.language_model_head = torch.nn.Linear(config.embedding_dimension, config.vocabulary_size, bias=False)
        """输出头，把每个位置的向量映射成整个词表上的分数（logits）。"""
        # 启用权重共享（weight tying）后，用 torch.compile() 会产生一些警告：
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # 不能百分百确定这是怎么回事，目前看来无害。TODO 待调查
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.word_token_embedding.weight = self.language_model_head.weight # https://paperswithcode.com/method/weight-tying

        # 初始化所有权重
        # init all weights
        self.apply(self._initialize_weights)
        # 按 GPT-2 论文的做法，对残差投影层施加特殊的缩放初始化
        # apply special scaled init to the residual projections, per GPT-2 paper
        for parameter_name, parameter in self.named_parameters():
            if parameter_name.endswith('output_projection.weight'):
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02/math.sqrt(2 * config.number_of_layers))

        # 报告参数量
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_number_of_parameters()/1e6,))

    def get_number_of_parameters(self, non_embedding=True):
        """
        返回模型的参数量。
        统计"非嵌入"参数量时（这是默认行为），会减去位置嵌入的参数。
        本来 token 嵌入也该减去，但由于参数共享，这部分参数实际上被当作
        最后一层的权重在用，所以把它们算进来。

        --- 以下为英文原文 ---
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        number_of_parameters = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            number_of_parameters -= self.transformer.word_position_embedding.weight.numel()
        return number_of_parameters

    def _initialize_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_indices, targets=None):
        device = token_indices.device
        batch_size, sequence_length = token_indices.size()
        assert sequence_length <= self.config.block_size, f"Cannot forward sequence of length {sequence_length}, block size is only {self.config.block_size}"
        positions = torch.arange(0, sequence_length, dtype=torch.long, device=device) # 形状为 (序列长度)

        # 执行 GPT 模型本身的前向传播
        # forward the GPT model itself
        token_embeddings = self.transformer.word_token_embedding(token_indices) # token 嵌入，形状为 (批, 序列, 嵌入维度)
        position_embeddings = self.transformer.word_position_embedding(positions) # 位置嵌入，形状为 (序列, 嵌入维度)
        x = self.transformer.embedding_dropout(token_embeddings + position_embeddings)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.final_layer_normalization(x)

        if targets is not None:
            # 如果传入了期望的目标值，就顺便计算损失
            # if we are given some desired targets also calculate the loss
            logits = self.language_model_head(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理时的小优化：只对最后一个位置跑 language_model_head
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.language_model_head(x[:, [-1], :]) # 注意：用列表 [-1] 是为了保住时间维
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # 必要时给模型"动个手术"，把 block size 改小
        # 比如我们可能加载了 GPT2 预训练检查点（block size 为 1024），
        # 但想在某个更小、更简单的模型上用更小的 block size
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.word_position_embedding.weight = torch.nn.Parameter(self.transformer.word_position_embedding.weight[:block_size])
        for block in self.transformer.blocks:
            if hasattr(block.attention, 'causal_mask'):
                block.attention.causal_mask = block.attention.causal_mask[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_arguments=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_arguments = override_arguments or {} # 默认为空字典
        # 只有 dropout 可以被覆盖，详见下面的说明
        # only dropout can be overridden see more notes below
        assert all(name == 'dropout' for name in override_arguments)
        import transformers
        print("loading weights from pretrained gpt: %s" % model_type)

        # 层数、注意力头数和嵌入维度由 model_type 决定
        # n_layer, n_head and n_embd are determined from model_type
        config_arguments = {
            'gpt2':         dict(number_of_layers=12, number_of_attention_heads=12, embedding_dimension=768),  # 124M 参数
            'gpt2-medium':  dict(number_of_layers=24, number_of_attention_heads=16, embedding_dimension=1024), # 350M 参数
            'gpt2-large':   dict(number_of_layers=36, number_of_attention_heads=20, embedding_dimension=1280), # 774M 参数
            'gpt2-xl':      dict(number_of_layers=48, number_of_attention_heads=25, embedding_dimension=1600), # 1558M 参数
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_arguments['vocabulary_size'] = 50257 # GPT 模型检查点里这个值恒为 50257
        config_arguments['block_size'] = 1024 # GPT 模型检查点里这个值恒为 1024
        config_arguments['bias'] = True # GPT 模型检查点里这个值恒为 True
        # 如果需要，我们可以覆盖 dropout 比率
        # we can override the dropout rate, if desired
        if 'dropout' in override_arguments:
            print(f"overriding dropout rate to {override_arguments['dropout']}")
            config_arguments['dropout'] = override_arguments['dropout']
        # 创建一个从零初始化的 minGPT 模型
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_arguments)
        gpt_model = GPT(config)
        state_dictionary = gpt_model.state_dict()

        # 初始化一个 huggingface/transformers 的模型
        # init a huggingface/transformers model
        huggingface_model = transformers.GPT2LMHeadModel.from_pretrained(model_type)
        huggingface_state_dictionary = huggingface_model.state_dict()

        # 逐个拷贝，同时确保所有参数在名称和形状上都对得上。
        # HuggingFace 那边用的是缩写形式的旧键名，所以每个键都要先翻译成我们的全称键名。
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        huggingface_keys = [key for key in huggingface_state_dictionary.keys()
                            if not key.endswith('.attn.masked_bias')  # 忽略它们，只是缓冲区
                            and not key.endswith('.attn.bias')]       # 同上，只是掩码（缓冲区）
        our_keys = [key for key in state_dictionary.keys()
                    if not key.endswith('.attention.causal_mask')]    # 丢弃这个掩码/缓冲区，它不是参数
        # 这些权重在 HuggingFace 那边是以 Conv1D 的转置形式存放的（下方注释有说明）
        transposed_huggingface_suffixes = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 简单说，openai 的检查点用的是 "Conv1D" 模块，而我们只想用普通的 Linear
        # 这意味着导入这些权重时必须先做转置
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(huggingface_keys) == len(our_keys), f"mismatched keys: {len(huggingface_keys)} != {len(our_keys)}"
        for huggingface_key in huggingface_keys:
            our_key = convert_legacy_state_dictionary_key(huggingface_key)
            assert our_key in state_dictionary, f"converted key not found in model: {our_key}"
            if any(huggingface_key.endswith(suffix) for suffix in transposed_huggingface_suffixes):
                # 对需要转置的 Conv1D 权重做特殊处理
                # special treatment for the Conv1D weights we need to transpose
                assert huggingface_state_dictionary[huggingface_key].shape[::-1] == state_dictionary[our_key].shape
                with torch.no_grad():
                    state_dictionary[our_key].copy_(huggingface_state_dictionary[huggingface_key].t())
            else:
                # 其余参数直接原样拷贝
                # vanilla copy over the other parameters
                assert huggingface_state_dictionary[huggingface_key].shape == state_dictionary[our_key].shape
                with torch.no_grad():
                    state_dictionary[our_key].copy_(huggingface_state_dictionary[huggingface_key])

        return gpt_model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 先取出所有候选参数
        # start with all of the candidate parameters
        parameter_dictionary = {parameter_name: parameter for parameter_name, parameter in self.named_parameters()}
        # 过滤掉那些不需要梯度的参数
        # filter out those that do not require grad
        parameter_dictionary = {parameter_name: parameter for parameter_name, parameter in parameter_dictionary.items() if parameter.requires_grad}
        # 创建优化器参数组。所有二维的参数都会做权重衰减，其余的不做。
        # 也就是说：矩阵乘法里的权重张量 + 嵌入层会衰减，所有偏置和 layernorm 不衰减。
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_parameters = [parameter for name, parameter in parameter_dictionary.items() if parameter.dim() >= 2]
        no_decay_parameters = [parameter for name, parameter in parameter_dictionary.items() if parameter.dim() < 2]
        optimizer_groups = [
            {'params': decay_parameters, 'weight_decay': weight_decay},
            {'params': no_decay_parameters, 'weight_decay': 0.0}
        ]
        number_of_decay_parameters = sum(parameter.numel() for parameter in decay_parameters)
        number_of_no_decay_parameters = sum(parameter.numel() for parameter in no_decay_parameters)
        print(f"num decayed parameter tensors: {len(decay_parameters)}, with {number_of_decay_parameters:,} parameters")
        print(f"num non-decayed parameter tensors: {len(no_decay_parameters)}, with {number_of_no_decay_parameters:,} parameters")
        # 创建 AdamW 优化器，如果有 fused（融合）版本就用它
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_arguments = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate, betas=betas, **extra_arguments)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_model_flops_utilization(self, forward_backward_per_iteration, elapsed_time):
        """
        估算模型算力利用率（MFU），以 A100 在 bfloat16 下的峰值 FLOPS 为基准单位

        --- 以下为英文原文 ---
        estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        """
        # 首先估算每次迭代要做多少次浮点运算（flops）。
        # 参考 PaLM 论文附录 B：https://arxiv.org/abs/2204.02311
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        number_of_parameters = self.get_number_of_parameters()
        config = self.config
        number_of_layers = config.number_of_layers
        number_of_attention_heads = config.number_of_attention_heads
        head_dimension = config.embedding_dimension // config.number_of_attention_heads
        sequence_length = config.block_size
        floating_point_operations_per_token = 6*number_of_parameters + 12*number_of_layers*number_of_attention_heads*head_dimension*sequence_length
        floating_point_operations_per_forward_backward = floating_point_operations_per_token * sequence_length
        floating_point_operations_per_iteration = floating_point_operations_per_forward_backward * forward_backward_per_iteration
        # 把我们的 flops 吞吐量表示为 A100 bfloat16 峰值算力的百分比
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        floating_point_operations_achieved = floating_point_operations_per_iteration * (1.0/elapsed_time) # 每秒
        floating_point_operations_promised = 312e12 # A100 GPU 在 bfloat16 下的峰值算力是 312 TFLOPS
        model_flops_utilization = floating_point_operations_achieved / floating_point_operations_promised
        return model_flops_utilization

    @torch.no_grad()
    def generate(self, token_indices, max_new_tokens, temperature=1.0, top_k=None):
        """
        接收一段作为条件的索引序列 token_indices（形状为 (批, 序列) 的 LongTensor），把这个序列
        续写 max_new_tokens 次，每次都把预测结果重新喂回模型。
        用它的时候，多半应该先确保模型处于 eval() 模式。

        --- 以下为英文原文 ---
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # 如果上下文序列变得太长，必须把它裁剪到 block_size
            # if the sequence context is growing too long we must crop it at block_size
            conditioning_token_indices = token_indices if token_indices.size(1) <= self.config.block_size else token_indices[:, -self.config.block_size:]
            # 前向跑一遍模型，得到序列中该位置的 logits
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(conditioning_token_indices)
            # 取出最后一步的 logits，并按设定的温度做缩放
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # 可选：把 logits 裁剪成只保留概率最高的 k 个选项
            # optionally crop the logits to only the top k options
            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = -float('Inf')
            # 用 softmax 把 logits 转换成（归一化后的）概率
            # apply softmax to convert logits to (normalized) probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            # 从这个分布里采样
            # sample from the distribution
            next_token_index = torch.multinomial(probabilities, num_samples=1)
            # 把采样到的索引追加到当前序列后面，然后继续
            # append sampled index to the running sequence and continue
            token_indices = torch.cat((token_indices, next_token_index), dim=1)

        return token_indices
