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
# math（Mathematical，数学）：Python 标准库，提供数学函数与常量。本文件用到 sqrt() 开平方、pi、cos 等。
import inspect
# inspect（inspection，检查/审视）：Python 标准库，用来"检视"程序运行中的对象，如查看函数签名。
# 本文件用 inspect.signature() 查看 torch.optim.AdamW 的参数，判断当前 PyTorch 是否支持 fused 优化器。
import dataclasses
# dataclasses（data + classes，数据类）：Python 标准库，用 @dataclasses.dataclass 修饰类后，自动生成 __init__、__repr__ 等样板代码。
# 本文件用它把纯数据的配置类 GPTConfig 定义成"数据类"。

import torch
# torch：「PyTorch」的官方包名（torch 原本是"火把"的意思，寓意像火炬一样点亮深度学习）。
# 一个开源深度学习框架，提供 GPU 张量计算、自动求导（autograd）与神经网络模块（torch.nn）。
# 注意：全称是 PyTorch，包名是 torch，二者指同一个东西。

# 本项目把模块属性名改成了完整全称，而 HuggingFace 的 GPT-2 权重用的是缩写形式的
# 旧键名。下面的映射表负责在加载权重时把旧键名翻译成现在的全称键名。

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

class LayerNorm(torch.nn.Module):
    """
    LayerNorm，但偏置（bias）是可选的。PyTorch 不支持简单地写 bias=False

    --- 以下为英文原文 ---
    LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False
    """
    # torch.nn：（neural network，神经网络）PyTorch 的神经网络模块集合。
    # torch.nn.Module：神经网络组件的基类（base class），你想做任何网络层都得继承它。
    # 继承后自动获得 parameter 管理、train()/eval() 切换、state_dict() 保存等功能。

    def __init__(self, number_of_dimensions, bias):
        super().__init__()
        # super().__init__()：super（superior，上级）= 指父类；这句表示"调用父类 torch.nn.Module 的初始化"，
        # 必须先执行，父类才能正确登记好本层内部的状态。
        self.weight = torch.nn.Parameter(torch.ones(number_of_dimensions))
        # torch.ones(n)：创建一个全为 1、长度为 n 的一维张量（tensor，张量=多维数组）。
        # torch.nn.Parameter(...)：把普通张量声明为"可学习参数"，交给 PyTorch 的自动求导系统跟踪它的梯度。
        # 这里 weight 是 LayerNorm 的"缩放（scale）"系数，初始全为 1，训练时会不断更新。
        """缩放参数，逐维放缩归一化之后的数值。"""
        self.bias = torch.nn.Parameter(torch.zeros(number_of_dimensions)) if bias else None
        # torch.zeros(n)：创建全为 0 的一维张量。bias 真则给 LayerNorm 配可学习的"平移（shift）"系数。
        """平移参数；bias=False 时为 None。"""

    def forward(self, input_tensor):
        return torch.nn.functional.layer_norm(input_tensor, self.weight.shape, self.weight, self.bias, 1e-5)
        # torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)：
        #   functional 意为"函数接口层"，里面是与 torch.nn 各层等价的纯函数版本。
        #   作用：对最后一维做层归一化——沿该维求均值与方差，再把数据减均值、除标准差，
        #   最后乘 weight（缩放）、加 bias（平移）。5 个参数：输入张量、归一化形状、权重、偏置、eps。
        #   eps=1e-5 是个极小的数，加在分母里防止除以 0。

class CausalSelfAttention(torch.nn.Module):

    def __init__(self, config: GPTConfig):
        """构建因果自注意力子层：拼接投影、输出投影、dropout，并在不支持 Flash Attention 时注册因果掩码。

        输入: config（GPTConfig）— 含 embedding_dimension / number_of_attention_heads / dropout / bias / block_size。
        输出: 无（按 config 初始化子层可学习参数与缓冲）。
        """
        super().__init__()
        assert config.embedding_dimension % config.number_of_attention_heads == 0
        # assert（断言）：检查一个条件，为假就抛出 AssertionError 终止程序。
        # 这里确保"嵌入维度能被注意力头数整除"，否则下面没法把头均分。
        # 所有注意力头的 key、query、value 投影，但打包成一次批量计算
        # key, query, value projections for all heads, but in a batch
        self.combined_query_key_value_projection = torch.nn.Linear(config.embedding_dimension, 3 * config.embedding_dimension, bias=config.bias)
        # torch.nn.Linear(in, out, bias)：全连接（线性）层，做 y = xWᵀ + b 的矩阵乘法。
        #   参数：in_features 输入维度、out_features 输出维度、bias 是否带偏置。
        #   内部自动创建可学习权重 W（形状 out×in），训练时更新。
        """把 query、key、value 三个投影打包成一次矩阵乘法，所以输出宽度是嵌入维度的 3 倍。"""
        # 输出投影
        # output projection
        self.output_projection = torch.nn.Linear(config.embedding_dimension, config.embedding_dimension, bias=config.bias)
        """注意力的输出投影，把多头拼接后的结果映射回嵌入维度。"""
        # 正则化
        # regularization
        self.attention_dropout = torch.nn.Dropout(config.dropout)
        # torch.nn.Dropout(p)：随机"丢弃"层。前向时按概率 p 把部分元素置 0，训练时启用、评估时自动关闭。
        #   作用：强迫网络别过度依赖少数神经元，减轻过拟合。p 是丢弃的比例，如 0 表示不丢。
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
        # hasattr(obj, name)：Python 内建函数，判断对象 obj 是否具有名为 name 的属性/方法，返回 True/False。
        #   这里检查当前 PyTorch 是否提供了 scaled_dot_product_attention（Flash 注意力），有就能用快速版。
        """当前 PyTorch 版本是否支持 Flash Attention。"""
        if not self.use_flash_attention:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 因果掩码，确保注意力只作用于输入序列中当前位置左边的内容
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
            # register_buffer(name, tensor)：把张量注册成模型的"缓冲区（buffer）"。
            #   缓冲区和 Parameter 的区别：它不参与梯度更新（不是可学习参数），但会随模型一起迁移设备、随 state_dict 保存。
            # torch.ones(n, n)：建一个 n×n 的全 1 矩阵（n = block_size）。
            # torch.tril(...)：tril = triangle lower，取矩阵的"下三角部分"，把上三角（含右上）全置 0。
            #   [1 1 1]  tril 后得 [1 0 0]
            #   [1 1 1]            [1 1 0]
            #   [1 1 1]            [1 1 1]  —— 这就是"因果掩码"：位置 i 只能看到 ≤ i 的位置。
            # .view(...)：view 是"按新形状排列元素"但共享底层内存（reshape 的轻量版）。
            #   这里把 2 维矩阵改成 4 维 (1,1,block_size,block_size)，为了跟上注意力矩阵的维度对齐。

    def forward(self, x):
        batch_size, sequence_length, embedding_dimension = x.size() # 批大小、序列长度、嵌入维度
        # x.size()：返回张量各维大小组成的元组，这里拆包赋给三个变量。

        # 批量计算所有注意力头的 query、key、value，并把 head 维前移，使其变成批维度
        # calculate query, key, valuecombined_query_key_value_projections for all heads in batch and move head forward to be the batch dim
        query, key, value = self.combined_query_key_value_projection(x).split(self.embedding_dimension, dim=2)
        # .split(size, dim)：按大小把张量沿指定维切成若干块（切完再拼回原 shape 原本有多个块）。
        #   第一次用的张量 split 需要了解 dim 参数：dim=2 表示沿第 3 个维（长度=3*emb_dim）切成 3 块。
        head_dimension = embedding_dimension // self.number_of_attention_heads
        # // 整除：结果向下取整，不走浮点。如 768 // 12 = 64（每头的向量长度）。
        key   = key.view(batch_size, sequence_length, self.number_of_attention_heads, head_dimension).transpose(1, 2)   # (批, 头数, 序列, 每头维度)
        # .view(...)：重新排列元素成新形状（不复制数据）。把一整块 (B,T,dim) 拆成 (B,T,头数,每头维度)。
        # .transpose(dim0, dim1)：交换两个维（转置）。把第 1、2 维互换，变成 (B,头数,T,每头维度)，
        #   让"头"变成一个独立维度，方便对每个头单独做注意力。
        query = query.view(batch_size, sequence_length, self.number_of_attention_heads, head_dimension).transpose(1, 2) # (批, 头数, 序列, 每头维度)
        value = value.view(batch_size, sequence_length, self.number_of_attention_heads, head_dimension).transpose(1, 2) # (批, 头数, 序列, 每头维度)

        # 因果自注意力；自注意力计算：(批, 头数, 序列, 每头维度) x (批, 头数, 每头维度, 序列) -> (批, 头数, 序列, 序列)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.use_flash_attention:
            # 用 Flash Attention 的 CUDA 核函数做高效注意力计算
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                query, 
                key, 
                value, 
                attn_mask=None, 
                dropout_p=self.dropout_probability if self.training else 0, 
                is_causal=True)
            # torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)：
            #   PyTorch >= 2.0 的快速注意力实现。内部自动完成 softmax(QKᵀ/√d)V，还内置因果掩码优化。
            #   参数：q/k/v 是三个投影；attn_mask 显式掩码（这里用 None）；dropout_p 注意力的丢弃率；
            #   is_causal=True 告诉它"只看左边"（因果），这是去掩码就很快的关键。
        else:
            # 手工实现的注意力计算
            # manual implementation of attention
            attention_weights = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
            # @ 是矩阵乘法运算符（Python 3.5+ 引入，等价 numpy 的 matmul）。query @ keyᵀ 得注意力得分。
            # key.transpose(-2, -1)：-2 是倒数第 2 维，-1 是最后一维；转置它们的目的是让矩阵维度能对齐相乘。
            # math.sqrt(x)：开平方根。除以 √d（d=每头维度）缩放得分，防止数值过大，让 softmax 梯度更稳。
            attention_weights = attention_weights.masked_fill(self.causal_mask[:,:,:sequence_length,:sequence_length] == 0, float('-inf'))
            # .masked_fill(mask, value)：张量方法。mask 里为 True 的位置填入 value。
            #   把"上三角=0"的位置（未来位置）填成 -inf，softmax 后它们概率约为 0，实现只看过去。
            # float('-inf')：负无穷，参与 softmax 时 exp(-inf)≈0。
            attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)
            # torch.nn.functional.softmax(x, dim)：沿 dim 维算"softmax 归一化"——
            #   把任意大小的数值转成一组和为 1 的概率值（越大越接近 1）。dim=-1 指最后一维（各 token 之间）。
            attention_weights = self.attention_dropout(attention_weights)
            y = attention_weights @ value # (批, 头数, 序列, 序列) x (批, 头数, 序列, 每头维度) -> (批, 头数, 序列, 每头维度)
        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dimension) # 把所有头的输出并排拼回去
        # .contiguous()：确保张量在内存中是连续排列的。转置/切片常打乱内存顺序，.view() 前应先调它保证可行。

        # 输出投影
        # output projection
        y = self.residual_dropout(self.output_projection(y))
        return y

class MLP(torch.nn.Module):

    def __init__(self, config: GPTConfig):
        """构建 MLP 子层：升维投影（×4）→ GELU → 降维投影 → dropout。

        输入: config（GPTConfig）— 含 embedding_dimension / dropout / bias。
        输出: 无（按 config 初始化各子层可学习参数）。
        """
        super().__init__()
        self.expansion_projection = torch.nn.Linear(config.embedding_dimension, 4 * config.embedding_dimension, bias=config.bias)
        """升维投影，把嵌入维度放大到 4 倍。"""
        self.gelu_activation      = torch.nn.GELU()
        # torch.nn.GELU()：GELU（Gaussian Error Linear Unit，高斯误差线性单元），一种激活函数。
        #   给网络引入非线性。公式近似 x·Φ(x)（Φ 是标准正态分布的累积分布函数），
        #   对"该保留哪些信息"用 0~1 区间软性放行，如今 Transformer 里最常用。
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

    def __init__(self, config: GPTConfig):
        """构建一个 Transformer Block：两个 Pre-LN 残差子层（自注意力 + MLP）。

        输入: config（GPTConfig）— 模型结构配置。
        输出: 无（初始化 attention 与 multi_layer_perceptron 两个子层）。
        """
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
# @dataclasses.dataclass：装饰器的语法形式是 @名字 放在类/函数定义前一行，含义是"用后面这个函数处理下面的类"。
#   这里 dataclass 会看类体里的"带类型注解的变量"（如 block_size: int），自动为类生成 __init__、__repr__ 等方法，
#   使 GPTConfig 成为一个纯数据的配置容器，用法：GPTConfig(block_size=128, ...)。
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
        # torch.nn.ModuleDict(dict)：一个"字典型"容器，可以用名字访问内部的子模块（如 self.transformer['blocks']）。
        # torch.nn.ModuleList([...])：一个"列表型"容器，存放数量可变的子模块。
        # torch.nn.Embedding(vocab_size, dim)：查表层（embedding=嵌入）。
        #   输入是整数 token 下标，输出是对应下标那一行的向量；词越多表越大，向量是可学习的。
        # range(n)：内建函数，生成 0..n-1 的整数序列。列表推导式 [Block(cfg) for _ in range(n)] 生成 n 个 Block。
        # dict(...)：内建函数，构造字典。注意用等号传参时，键名就是参数名（这里等于键值对）。
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
        # self.apply(func)：Module 的方法，会把 func 递归地作用在这棵组件的"每一个子模块"上。
        #   这里等于对所有 Linear/Embedding 层调用 _initialize_weights 完成初始化。
        # 按 GPT-2 论文的做法，对残差投影层施加特殊的缩放初始化
        # apply special scaled init to the residual projections, per GPT-2 paper
        for parameter_name, parameter in self.named_parameters():
            # self.named_parameters()：逐个返回 (参数名, 参数张量)，参数名里带完整的模块路径，如 "blocks.3.attention.output_projection.weight"。
            if parameter_name.endswith('output_projection.weight'):
                # str.endswith(suffix)：字符串方法，判断是否以给定后缀结尾。
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02/math.sqrt(2 * config.number_of_layers))
                # torch.nn.init.normal_(tensor, mean, std)：原地把 tensor 填成"正态分布"随机数。
                #   注意函数名以下划线结尾「_」是 PyTorch 惯例：表示这是个原地（in-place）修改操作，会改写参数本身。
                # std=0.02/√(2*n_layers)：越深的层初始化得越小，避免信号在深网络中层层放大而爆炸。

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
        # self.parameters()：遍历返回所有可学习的参数张量。
        # parameter.numel()：numel = number of elements，返回张量里元素总数（参数量）。
        # sum(...) / sum(生成器)：内建函数求和，这里数所有参数的个数之和。
        if non_embedding:
            number_of_parameters -= self.transformer.word_position_embedding.weight.numel()
        return number_of_parameters

    def _initialize_weights(self, module: torch.nn.Module):
        """按 GPT 惯例初始化单个子层：Linear/Embedding 权重用 N(0, 0.02)，Linear 偏置置零。

        输入: module（torch.nn.Module）— 待初始化的子层。
        输出: 无（原地改写 module 的参数）。
        """
        if isinstance(module, torch.nn.Linear):
            # isinstance(obj, 类型)：内建函数，判断 obj 是否是这个类型（或其子类）的实例，返回 True/False。
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                # torch.nn.init.zeros_(tensor)：原地把所有元素置 0（初始化偏置用）。
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_indices, targets=None):
        device = token_indices.device
        batch_size, sequence_length = token_indices.size()
        assert sequence_length <= self.config.block_size, f"Cannot forward sequence of length {sequence_length}, block size is only {self.config.block_size}"
        positions = torch.arange(0, sequence_length, dtype=torch.long, device=device) # 形状为 (序列长度)
        # torch.arange(start, end, dtype, device)：生成从 start 到 end-1 的整数序列（可指定数据类型和设备）。
        #   这里得到 0,1,2,...,T-1，供位置嵌入表按"位置下标"取向量。dtype=torch.long = 64 位有符号整数。

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
                # torch.nn.functional.cross_entropy(input, target, ignore_index)：交叉熵损失，
                #   衡量"预测的分布"和"真实下标"之间的差距，值越小越准；LLM 训练的目标就是不断减小它。
                # .view(-1, ...)：-1 是"自动推算"的意思，PyTorch 会按元素总数反推出这一维的实际大小，把高维数组摊平成 2 维。
                # ignore_index=-1：跳过目标值为 -1 的位置（padding/占位），不算进损失。
        else:
            # 推理时的小优化：只对最后一个位置跑 language_model_head
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.language_model_head(x[:, [-1], :]) # 注意：用列表 [-1] 是为了保住时间维
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int):
        """"模型手术"：把 block_size 裁小。用于加载大块头预训练检查点后部署到更小的上下文。

        输入: block_size（int）— 裁小后的目标上下文长度，须 <= 当前 block_size。
        输出: 无（原地修改 config.block_size、位置嵌入与因果掩码）。
        """
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
    # @classmethod：类方法装饰器。被它修饰的方法第一个参数自动接收"类本身"（习惯命名 cls），
    #   因此可以用 GPT.from_pretrained(...) 直接通过类调用，不用先创建实例。
    def from_pretrained(cls, model_type, override_arguments=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # {...} 是集合（set）：一组不重复元素；x in 集合 判断"x 是否在里面"。
        override_arguments = override_arguments or {} # 默认为空字典
        # A or B：A 为假（如 None/空）则取 B。所以传 None 时回退成空字典 {}。
        # 只有 dropout 可以被覆盖，详见下面的说明
        # only dropout can be overridden see more notes below
        assert all(name == 'dropout' for name in override_arguments)
        # all(...)：内建函数，序列里所有元素都为真才返回 True。这里确保覆盖项只能是 dropout。
        import transformers
        # transformers（转换器/变形金刚）：HuggingFace 出品的深度学习库，内置大量预训练模型（GPT-2、BERT、LLaMA 等）。
        #   这里是为了拿到"官方 GPT-2 权重"，再搬运进我们自己的模型结构。
        print("loading weights from pretrained gpt: %s" % model_type)
        # print("..." % 值)：旧式字符串格式化（% 占位符），%s 表示把后面的字符串填进来。

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
        # **config_arguments：字典拆包（dictionary unpacking），把字典展开成关键字参数。
        #   即 GPTConfig(number_of_layers=12, number_of_attention_heads=12, ...)。
        gpt_model = GPT(config)
        state_dictionary = gpt_model.state_dict()
        # state_dict()：Module 的方法，返回一个字典，键是参数路径名，值是对应的权重张量。
        #   它是 PyTorch 存取权重的标准出口，也是保存检查点的核心。

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
                # torch.no_grad()：上下文管理器，进入后所有运算都不记录梯度、不生成计算图。
                #   只用来拷贝/读取权重时开着它，省内存又提速。
                    state_dictionary[our_key].copy_(huggingface_state_dictionary[huggingface_key].t())
                # .t()：矩阵转置（transpose 的简写，只对 2 维张量）。把 Conv1D 权重转回 Linear 的布局。
                # .copy_(src)：原地把 src 的元素复制到本张量（下划线结尾 = 原地操作）。
            else:
                # 其余参数直接原样拷贝
                # vanilla copy over the other parameters
                assert huggingface_state_dictionary[huggingface_key].shape == state_dictionary[our_key].shape
                with torch.no_grad():
                    state_dictionary[our_key].copy_(huggingface_state_dictionary[huggingface_key])

        return gpt_model

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple, device_type: str) -> torch.optim.AdamW:
        """配置 AdamW 优化器：二维参数（权重、嵌入）做权重衰减，一维参数（偏置、LayerNorm）不做；CUDA 上优先用融合版。

        输入:
            weight_decay（float）— 应用于二维参数的权重衰减系数。
            learning_rate（float）— 初始学习率。
            betas（tuple[float, float]）— AdamW 的一阶、二阶动量衰减率。
            device_type（str）— 'cuda' 或 'cpu'，决定是否使用 fused AdamW。
        输出: optimizer（torch.optim.AdamW）— 配置好的优化器。
        """
        # 先取出所有候选参数
        # start with all of the caload_state_dictndidate parameters
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
        # inspect.signature(可调用对象)：返回它的"签名"对象；.parameters 属性里装着所有参数名。
        #   技巧：把 'fused' in 参数字典 当判断条件用，看当前版本是否支持 fused 参数。
        # torch.optim.AdamW：PyTorch 优化器（optimizer=优化器，AdamW=Adam 的权重衰减改良版，LLM 训练标配）。
        use_fused = fused_available and device_type == 'cuda'
        extra_arguments = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate, betas=betas, **extra_arguments)
        # 优化器的职责：反向传播算出梯度后，用它更新模型的权重，让损失逐步变小。
        #   lr（learning rate，学习率）：每次都权重更新的步长；betas：Adam 的两个动量系数。
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
    # 这里 @torch.no_grad() 用法=函数装饰器：把 generate 整个包在"不记录梯度"的上下文里执行（推理用）。
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
                # torch.topk(x, k)：取出张量里最大的 k 个数，返回 (数值序列, 对应下标序列)。_ 表示"不要第二个返回值"。
                # min(a, b)：内建函数，取较小者（防止要取的 k 超过词表大小）。
                logits[logits < top_values[:, [-1]]] = -float('Inf')
                # 布尔索引：把"小于第 k 大值"的位置全部改成 -inf，从而在 softmax 后概率为 0——只保留 top-k 候选。
            # 用 softmax 把 logits 转换成（归一化后的）概率
            # apply softmax to convert logits to (normalized) probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            # 从这个分布里采样
            # sample from the distribution
            next_token_index = torch.multinomial(probabilities, num_samples=1)
            # torch.multinomial(probabilities, num_samples)：按给定概率分布随机抽样 num_samples 次，
            #   概率高的 token 更容易被抽到（而不是总选最大的）。这一步带来生成的多样性。
            # 把采样到的索引追加到当前序列后面，然后继续
            # append sampled index to the running sequence and continue
            token_indices = torch.cat((token_indices, next_token_index), dim=1)
            # torch.cat((a, b), dim)：沿着 dim 维把若干个张量拼接起来（concatenate）。
            #   这里把新采到的 token 拼到原序列的时间维（第 1 维）末尾，实现"自我续写"。

        return token_indices
