# nanoGPT 面试记录

> 本文档记录一场以 nanoGPT 项目为背景的模拟面试。
> 角色：面试官（用户） / 面试者（Claude）。
> 每一轮包含三部分：**问题**、**回答**、**点评**。
> 编号约定：`## 第 N 题` 为主问题；其下的追问/子问题用 `#### N.1`、`#### N.2` …… 缩进编号。

---

## 术语表（初学者友好）

> 收录本文档中出现的、不常见或初学者容易卡住的术语，用大白话解释，并说明它在 nanoGPT 里起什么作用。随着面试推进会持续补充。
>
> 注：本项目已把源码里的缩写变量名统一改成完整全称（例如 `wte` → `word_token_embedding`）。术语表采用新名字，并在需要处标注上游 nanoGPT 的原名，方便对照阅读原版代码。

### 一、模型结构相关

- **Transformer**：一种神经网络架构，靠「注意力机制」让序列里每个词都能参考其他词。GPT 就是它的一种。名字取自开创它的论文《Attention Is All You Need》，"transform" 指把一串输入向量变换成一串输出向量。
- **decoder-only（仅解码器）**：Transformer 原本有「编码器+解码器」两半；GPT 只保留解码器那一半，专门做「根据前文预测下一个词」。所以叫 decoder-only。
- **Block（块 / 层）**：GPT 主干里可以重复堆叠的「一层」，每层内部 = 一个注意力子层 + 一个 MLP 子层。nanoGPT 默认堆 12 个 Block。代码里就是 `Block` 这个类（`model.py:167`）。
- **子层（sub-layer）**：一个 Block 内部串联的两个小模块，即「注意力子层」和「MLP 子层」，每个外面都套着同一套固定套路——先 LayerNorm、算完再残差加回来。叫「子」层是因为它比 Block 这个「层」低一级，所以 12 层的 GPT-2 实际上有 24 个子层。两者分工互补：注意力子层让**词与词之间交换信息**，MLP 子层让**每个词各自把信息加工深**，缺任何一个模型都不成立（`model.py:176-178`）。
- **word_token_embedding（token 嵌入表）**：把每个词（token）的编号翻译成一串数字向量，让模型能计算。上游 nanoGPT 里这个属性叫 `wte`，是 **w**ord **t**oken **e**mbedding 的缩写，本项目已改成完整全称（`model.py:200`）。
- **word_position_embedding（位置嵌入表）**：告诉模型「这个词排在第几个位置」，因为注意力本身不区分先后。上游叫 `wpe`，是 **w**ord **p**osition **e**mbedding 的缩写，本项目已改成完整全称。用的是「可学习的绝对位置嵌入」（`model.py:201`）。
- **embedding（嵌入）**：把离散的东西（如一个词）映射成连续向量的做法。好比给每个词发一张「数字身份证」，意思相近的词身份证也相近。
- **embedding_dimension / number_of_attention_heads / number_of_layers / block_size / vocabulary_size**：模型的超参数。分别是：嵌入向量维度、注意力头数、Block 层数、模型一次能看的最大 token 数（上下文长度）、词表大小。上游对应的缩写名依次是 `n_embd` / `n_head` / `n_layer` / `block_size` / `vocab_size`。
- **LayerNorm（层归一化）**：把一层的输出数值重新缩放到「均值 0、方差 1」附近，让训练更稳。好比每过一关就把数值「拉回正轨」，防止越滚越大或越缩越小。
- **Pre-LN（前置归一化）**：指「先做 LayerNorm，再进注意力/MLP」的顺序（对应 Post-LN 是先算再归一化）。Pre-LN 训练更稳、更容易堆深层。nanoGPT 用的是 Pre-LN（`model.py:177`）。
- **Post-LN（后置归一化）**：与 Pre-LN 相反的顺序——先算子层，再把「残差相加后的结果」整个归一化，即 `x = LN(x + f(x))`。2017 年 Transformer 原论文用的就是这种，但堆深层时训练不稳、必须小心翼翼地做学习率预热，所以 GPT-2 之后主流都改用了 Pre-LN。
- **残差连接（residual connection）**：`x = x + f(x)` 这种写法——把子层的输出加回到输入上。好处是给梯度留了一条「高速公路」，深层网络也不容易训练失败。
- **注意力（attention）/ 自注意力（self-attention）**：让序列里每个位置去「关注」其他位置并加权汇总信息的机制。自注意力就是序列关注它自己。
- **cross-attention（交叉注意力）**：Query 来自一个序列、Key/Value 来自另一个序列的注意力，典型场景是翻译——用译文当前词去「查」原文该看哪里。与之相对，self-attention 的 Q/K/V 全部来自同一个序列。GPT 是 decoder-only，**只有自注意力，没有交叉注意力**。
- **因果（causal）**：限制每个位置只能看它「左边（前面）」的词，不能偷看后面的。因为生成时后文还没出现。也叫 masked（带掩码的）注意力。
- **多头（multi-head）**：把注意力拆成多个「头」并行地从不同角度看关系，再拼起来。好比多个专家分别关注不同侧面。
- **Q/K/V（Query/Key/Value，查询/键/值）**：注意力的三种角色。Query 是「我想找什么」，Key 是「我有什么标签」，Value 是「我实际携带的信息」。Query 和 Key 匹配算权重，再去加权 Value。
- **Flash Attention**：一种把注意力算得又快又省显存的 GPU 实现，PyTorch ≥ 2.0 通过 `scaled_dot_product_attention` 提供。名字里的 "flash" 就是形容快（`model.py:111`）。
- **MLP（Multi-Layer Perceptron，多层感知机）/ 前馈（feed-forward）**：Block 里的另一个子层，就是两层全连接网络，中间升维再降回来，负责「逐位置地加工特征」。它是 Block 里的**参数大户**——约占一个 Block 参数量的 2/3（注意力子层占 1/3），一般认为模型记住的事实知识主要存放在这里（`model.py:151`）。
- **FFN（Feed-Forward Network，前馈网络）**：MLP 子层在 Transformer 原论文里的叫法，指的是同一个东西。所以「MLP 子层 = FFN 子层 = 前馈子层」三名一物，读不同论文会碰到不同叫法。「前馈」指信号只从输入单向流向输出、中间没有回路，是相对于 RNN 的「循环」而言的。
- **感知机（Perceptron）**：MLP 里那个 "P" 的来源。1958 年 Rosenblatt 提出的最早的神经网络模型，名字取自 "perception"（感知），因为它模仿神经元「接收信号 → 加权求和 → 超过阈值就激活」的过程。单层感知机连异或（XOR）都学不会，1969 年被 Minsky 批评后一度让神经网络研究停滞十年；加上隐藏层变成「多层」感知机后才有了强大的拟合能力，MLP 这个略显复古的名字就是这段历史留下的。
- **4 倍升维（4x expansion）**：MLP 子层先把维度从 `embedding_dimension` 放大到 `4*embedding_dimension`，激活之后再降回原维度（`model.py:155-157`）。中间「胖」出来的那层给了模型更大的施展空间，好比把材料摊到大案板上加工完再收拢。4 倍是 Transformer 原论文定下的经验值，一直沿用至今。
- **token mixing / channel mixing（跨位置混合 / 逐通道加工）**：概括两个子层分工的一对说法。注意力是 token mixing，在**位置维度**上混合信息——`attention_weights` 的形状是 `(批, 头数, 序列, 序列)`，最后那个「序列×序列」正是「每个词对每个词」的关注权重表（`model.py:139`）；MLP 是 channel mixing，只在**特征维度**上做变换，各个位置之间互不相干——`torch.nn.Linear` 只作用在最后一维（`model.py:161-163`）。记住这一对，就抓住了两个子层的本质区别。
- **GELU（Gaussian Error Linear Unit）**：一种激活函数，作用类似 ReLU 但更平滑。GPT 系列常用它（`model.py:156`）。
- **非线性（non-linearity）**：激活函数带来的「掰弯」能力。如果不加激活函数，多层全连接叠起来在数学上等价于一层，模型再深也白搭。Block 里的非线性主要由 MLP 子层的 GELU 提供——这正是「光有注意力不够、必须配一个 MLP」的原因：单看注意力，它对 value 做的基本是加权平均，表达能力很有限。
- **dropout（随机失活）**：训练时随机把一部分神经元输出临时置 0，逼模型别过度依赖某几个特征，是一种防过拟合的正则化手段。nanoGPT 默认 `dropout=0.0`（预训练时通常关掉）。
- **language_model_head（语言模型头）**：模型最后一层线性变换，把每个位置的向量映射回「整个词表」的分数。"head" 指接在主干后面的任务输出头。上游叫 `lm_head`（`model.py:206`）。
- **logits**：模型输出的、还没归一化成概率的「原始分数」。经过 softmax 才变成概率。
- **weight tying（权重共享 / 权重绑定）**：让输入的 token 嵌入表和输出的 language_model_head 共用同一套权重，省参数也常常更稳（`model.py:215`）。
- **softmax**：把一串任意实数分数变成「加起来等于 1」的概率分布的函数。分数越大概率越高。
- **cross-entropy（交叉熵）**：分类任务常用的损失函数，衡量「模型预测的概率分布」和「真实答案」差多少，越小越好（`model.py:275`）。
- **置换不变（permutation invariance）**：指一个运算「不在乎输入顺序」——把输入打乱，结果只是跟着换位置、数值不变。自注意力本身就是置换不变的，所以它「看不出词序」，必须靠位置嵌入额外补上「谁前谁后」的信息。
- **广播（broadcasting）**：NumPy/PyTorch 里两个形状不完全一样的张量做逐元素运算时，自动把较小的那个「复制扩展」到匹配形状的机制。例如 `(批, 序列, 维度)` 加 `(序列, 维度)` 时，位置嵌入会被广播到每个 batch 上（`model.py:293`）。名字来自广播电台「一份信号发给所有听众」的比喻。
- **正则化（regularization）**：一大类「防止模型死记硬背训练数据、提升泛化能力」的手段的统称，dropout、weight decay 都属于它。"regular" 有「规整、约束」之意，即给模型加约束别让它太放飞。
- **token（词元）**：文本被切分后的最小单位，可能是一个词、一个子词、甚至一个字符。模型不直接处理文字，而是处理 token 对应的整数编号。名字就是英文「符号 / 记号」的意思。
- **tokenizer（分词器）**：把原始文本切成 token 并映射成整数编号的工具（GPT-2 用的是 BPE 分词）。它在模型之外的数据预处理阶段工作，模型只接收它吐出的整数序列。
- **nn.Embedding（嵌入层）**：PyTorch 里实现「查表式词嵌入」的模块，内部是一个 `(词表大小, 嵌入维度)` 的可学习矩阵，输入整数下标、输出对应行向量。本质是一次按行索引的查表（`model.py:226`）。
- **one-hot（独热编码）**：把一个类别编号表示成「只有对应那一位是 1、其余全 0」的向量。查表式嵌入在数学上等价于「one-hot 向量 × 嵌入矩阵」，但实现上直接索引、跳过 one-hot。名字里 "one hot" 指「只有一位是热的（点亮为 1）」。
- **gather（按索引收集）**：按给定的下标从张量里「挑出」对应元素/行的操作。`nn.Embedding` 的查表底层就是 gather，比造一个巨大 one-hot 再做矩阵乘法省得多。
- **combined_query_key_value_projection（合并 QKV 投影）**：把 Query/Key/Value 三个线性投影打包成一个 `Linear(嵌入维度 → 3×嵌入维度)` 一次算完，再切成三块。合并是为 GPU 效率（一个大矩阵乘法快过三个小的），与三个独立投影数学等价。上游对应 `c_attn`（`model.py:100`）。
- **head_dimension（每头维度）**：多头注意力里每个头分到的维度 = `嵌入维度 ÷ 头数`（默认 768÷12=64）。拆头就是把 768 维重新解释成「12 头 × 64 维」（`model.py:135`）。
- **缩放点积注意力（scaled dot-product attention）**：注意力的标准算法——`softmax(Q·Kᵀ / √head_dimension) · V`。除以 `√head_dimension` 是「缩放」，防止点积数值过大把 softmax 推到梯度极小的饱和区（`model.py:149`）。
- **view / reshape（改形状不改数据）**：把张量在不复制数据的前提下重新解释成另一个形状，前提是元素总数不变。多头注意力用 `view` 把 `(批,序列,768)` 变成 `(批,序列,12,64)` 来「免费」分头（`model.py:136`）。
- **transpose（转置/交换维度）**：交换张量的两个维度。注意力里用 `transpose(1,2)` 把「头」维挪到「批」维旁边，好让每个头像 batch 一样被独立批量计算（`model.py:136`）。
- **split（切分）**：沿某一维把张量切成若干块。合并 QKV 投影后用 `.split(768, dim=2)` 把 2304 维切回 Q/K/V 三段（`model.py:134`）。

### 二、训练 / 自动微分相关

- **nn.Parameter（参数）**：PyTorch 里一种特殊张量，专门用来装「模型要学习的权重」。把张量包成 `Parameter` 后，它会自动 `requires_grad=True` 并被 `model.parameters()` 收集，从而能被优化器更新。模型里所有权重（嵌入表、Linear 的 weight/bias 等）本质都是 `Parameter`。
- **requires_grad（是否需要梯度）**：张量上的一个布尔开关。为 `True` 时，autograd 会记录它参与的运算并在反向时给它算梯度——这正是「可学习」的技术开关。普通张量默认为 `False`。
- **autograd（自动微分引擎）**：PyTorch 内置的「自动求导」系统。前向计算时它自动记录运算连成计算图，调用 `backward()` 时自动用链式法则算出所有梯度，省去手写求导。名字是 **auto**matic **grad**ient（自动梯度）的缩写。
- **计算图（computation graph）**：前向传播时 autograd 在背后搭起的一张「谁由谁、经过什么运算得来」的关系图，是反向求导的地图。PyTorch 的图是「动态图」——每次前向都重新搭一遍。
- **梯度（gradient）**：loss 对某个参数的偏导数，指向「让 loss 增大最快」的方向。训练时朝它的反方向走一小步就能减小 loss。反向传播算完后存在参数的 `.grad` 属性里。
- **反向传播（backpropagation / backward）**：从 loss 出发、沿计算图倒着用链式法则逐层求出每个参数梯度的过程。代码里就是 `loss.backward()`（`train.py:448`）。"back" 指方向与前向相反。
- **链式法则（chain rule）**：微积分里「复合函数求导 = 各层导数相乘」的法则。反向传播就是把它自动化地套在整张计算图上。
- **optimizer（优化器）/ optimizer.step()**：拿着梯度去真正更新参数的组件。`step()` 按 `参数 ← 参数 − 学习率 × 更新量` 就地改参数。nanoGPT 用的是 **AdamW**（`train.py:456`）。
- **AdamW**：一种常用优化器，在 Adam（自适应学习率）基础上把权重衰减（weight decay）做对。名字 = **Adam** + **W**eight decay。nanoGPT 里对 2 维以上参数施加权重衰减、其余不加（见 `configure_optimizers`）。
- **zero_grad（清空梯度）**：把参数的 `.grad` 清零。因为 PyTorch 的梯度是「累加」的，每轮更新完必须清零，否则下一轮会和这轮叠加（`train.py:460`）。
- **学习率（learning rate）**：每次更新「走多大一步」的系数。太大容易震荡发散，太小学得慢。
- **梯度累积（gradient accumulation）**：连续跑几个小批次、把梯度累加起来再更新一次，用来在显存有限时「模拟」大 batch 的效果（`train.py:439-442`）。
- **梯度裁剪（gradient clipping）**：当梯度整体过大时把它按范数缩小，防止「梯度爆炸」导致训练崩掉。代码里是 `clip_grad_norm_`（`train.py:453`）。
- **混合精度 / GradScaler（梯度缩放器）**：用 float16 加速训练时，梯度数值可能小到下溢为 0，于是先把 loss 放大（scale）再 backward、更新前再缩回（unscale），`GradScaler` 负责这套缩放（`train.py:448-457`）。

---

## 项目文件导览（各脚本功能）

> 面试前先看这张"仓库地图"：每个脚本是干嘛的、入口在哪、怎么跑，先有个整体印象。
> 这样被问到"这个文件是干什么的"就能立刻对上，也为后面每一题提供上下文。
> 符号约定：`有 main` = 脚本有 `main()` 入口函数并用 `if __name__ == '__main__'` 管起来；`被注入` = 没有自己的入口，是被别的脚本 `exec` 进去执行的。

### 一、仓库总览

整个仓库只围绕三件事转：

1. **定义模型** — `model.py`：GPT 长什么样（唯一被当作"库"来 `import` 的文件）。
2. **跑模型** — `train.py`（训练）、`sample.py`（生成文本）、`bench.py`（测速）。
3. **后勤** — `config/` + `configurator.py` 负责喂配置，`data/*/prepare.py` 负责造数据。

一条典型的主链路：

```
data/.../prepare.py  把原始文本切成 token，存成 train.bin / val.bin（有的带 meta.pkl）
        ↓
train.py             读 bin 文件训练，产出 out/ckpt.pt（权重+优化器状态+超参快照）
        ↓
sample.py            加载 ckpt.pt 或 OpenAI GPT-2 权重，按提示词续写文本
```

想换超参时，靠 `configurator.py` 解析命令行 `--KEY=value` 或配置文件，直接覆盖各脚本**顶格大写的全局超参**（这就是超参必须留在模块顶层的根本原因）。

### 二、逐文件说明

#### 1. `model.py` — GPT 模型定义（库，无 main）

- **是什么**：整个 GPT 模型唯一的定义处，一个大文件装下了从"一层归一化"到"完整模型"的全部组件。其他脚本都 `import model` 用它，所以它**没有也不该有** `main()`。
- **包含的组件**（自底向上）：
  - `LayerNorm`：可选 bias 的层归一化。
  - `CausalSelfAttention`：因果多头自注意力（Flash 版 / 手写版二选一）。
  - `MLP`：升维 4 倍 → GELU → 降维的两层网络。
  - `Block`：一个 Transformer 层 = Pre-LN 注意力子层 + Pre-LN MLP 子层，都带残差。
  - `GPTConfig`：数据类（`@dataclasses.dataclass`），装全部超参数。
  - `GPT`：把以上拼起来的完整模型，附带 5 个方法 — `forward`（训练/推理双路径）、`from_pretrained`（搬运 OpenAI 预训练权重）、`configure_optimizers`（二维参数/一维参数分两组）、`generate`（自回归采样）、`estimate_model_flops_utilization`（估算 MFU）。
- **面试地位**：下方"第 1 题"和附录题库里，几乎所有模型结构问题都以这个文件为主战场。

#### 2. `train.py` — 训练主脚本（有 main）

- **是什么**：训练 GPT 的核心流程，同时支持单卡调试与 DDP 多卡训练。
- **主流程**（`main()` 里按阶段走）：
  1. **初始化环境**：DDP 进程组（如多卡）、随机种子（每进程偏移）、设备、混合精度上下文。
  2. **加载数据**：从 `meta.pkl` 推导词表大小、构造"贫穷版批次加载器"。
  3. **构建模型**：按 `INITIALIZE_FROM` 三选一——`scratch` 从零 / `resume` 从检查点续训 / `gpt2*` 加载 OpenAI 权重；需要时做 block size 裁剪并搬上设备。
  4. **建优化器**：`configure_optimizers` 得到 AdamW；resume 模式恢复优化器状态；float16 时配 `GradScaler`。
  5. **编译 + DDP 包装**：可选 `torch.compile`，多卡再包 `DistributedDataParallel`。
  6. **准备基础函数**：损失评估器（`estimate_loss`）、学习率调度器（warmup + 余弦衰减）。
  7. **wandb 日志**（可选）。
  8. **训练主循环**：定学习率 → 定期评估并保存检查点 → 梯度累积多次前向/反向 → 梯度裁剪 → 优化器 step。
- **关键产出**：`out/ckpt.pt`，供 `sample.py` 读取。
- **怎么跑**：单卡 `python train.py --BATCH_SIZE=32 --COMPILE=False`；4 卡 `torchrun --standalone --nproc_per_node=4 train.py`。

#### 3. `sample.py` — 从模型生成文本（有 main）

- **是什么**：加载训练好的模型（`resume` 读本地检查点）或 OpenAI GPT-2 预训练权重，根据一段起始提示词续写出若干条文本。
- **主流程**（`main()` 里 5 步）：
  1. 初始化随机种子 + TF32 + 混合精度上下文。
  2. 加载模型（`load_model`）：resume 剥掉 `_orig_mod.` 前缀后 `load_state_dict`，或 `from_pretrained` 拉 GPT-2 权重；切成 eval 态、搬设备、可选 `torch.compile`。
  3. 构建编解码函数（`build_codec`）：数据集有 `meta.pkl` 用字符级 `stoi/itos`，没有则回退 GPT-2 BPE（`tiktoken`）。
  4. 把起始提示词编码成输入张量（`[None, ...]` 补批维度）。
  5. 循环 `NUMBER_OF_SAMPLES` 次调用 `GPT.generate` 并打印解码结果。
- **怎么跑**：`python sample.py`；命令行覆盖示例 `python sample.py --NUMBER_OF_SAMPLES=3 --MAX_NEW_TOKENS=200`。

#### 4. `bench.py` — 基准测速（有 main）

- **是什么**：`train.py` 的精简加速版本，只关心"每迭代耗时 + 算力利用率（MFU）"，用于衡量硬件与实现效率。
- **主流程**（`main()` 里 4 步）：
  1. 初始化运行环境（种子 + 混合精度）。
  2. 构造批次加载器：`REAL_DATA=True` 读真实 openwebtext 的 `train.bin`；`False` 用固定随机张量，排除数据加载对测速的干扰。
  3. 建 GPT 模型 + AdamW 优化器（`configure_optimizers`），可选 `torch.compile`。
  4. 按 `PROFILE` 二选一：`torch.profiler` 出详细 trace（写到 `./bench_log`），或简单计时打印 `time per iteration` 与 `MFU`。
- **怎么跑**：`python bench.py`。

#### 5. `configurator.py` — 穷人版配置器（被注入，无 main）

- **是什么**：这段代码会被各脚本顶层的 `exec(open('configurator.py').read())` **直接执行**，用来解析命令行参数（`--KEY=value`）或配置文件，覆盖脚本顶层的全局超参。
- **工作逻辑**：遍历 `sys.argv[1:]`（命令行参数，去掉脚本名自己）→ 不含 `=` 的当成**配置文件**，`exec` 执行它；含 `=` 的切成键值 → 用 `ast.literal_eval` 安全地把字符串还原成数字/布尔等类型 → 类型核对后 `globals()[key] = value` 写回全局字典。这就是"所有超参必须顶格大写"的原因——覆盖只能落到已存在的全局键上。
- **为什么这样设计**：作者自嘲"Probably a terrible idea"，为了避开配置系统的复杂度，选择了最朴素粗暴的方案。

#### 6. `config/*.py` — 配置文件（被 configurator exec）

| 文件                                                                                      | 用途                                                                                              |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `train_gpt2.py`                                                                         | 在 8×A100 上从零训 GPT-2 124M 到损失约 2.85 的正式配置（约 5 天，总 batch ~0.5M）。              |
| `train_shakespeare_char.py`                                                             | 字符级莎士比亚迷你模型（6 层 384 维），适合单卡/笔记本调试。                                      |
| `finetune_shakespeare.py`                                                               | 从`gpt2-xl` 预训练权重微调莎士比亚文本（恒定小学习率）。                                        |
| `eval_gpt2.py` / `eval_gpt2_medium.py` / `eval_gpt2_large.py` / `eval_gpt2_xl.py` | 分别评估 GPT-2 四个尺寸（124M / 350M / 774M / 1558M）在数据上的损失（`EVALUATION_ONLY=True`）。 |

> 它们本质是"纯变量赋值的文件"，被 `configurator.py` exec 进去，所以没有 import、没有 main 是正确设计。

#### 7. `data/*/prepare.py` — 数据预处理（有 main）

三个目录对应三种数据原料，`prepare.py` 把原始文本预处理成训练脚本能直接读的 `train.bin` / `val.bin`（字符级那套还会产出 `meta.pkl`）：

| 文件                            | 编码方式                  | 产出                                     | 规模                           |
| ------------------------------- | ------------------------- | ---------------------------------------- | ------------------------------ |
| `shakespeare_char/prepare.py` | 字符级（每个字符一个 id） | `train.bin`、`val.bin`、`meta.pkl` | ~1.1M 字符，65 个不同字符      |
| `shakespeare/prepare.py`      | GPT-2 BPE（`tiktoken`） | `train.bin`、`val.bin`               | ~30 万 token                   |
| `openwebtext/prepare.py`      | GPT-2 BPE（`tiktoken`） | `train.bin`、`val.bin`（无 meta）    | ~90 亿 token，HF 缓存占地 54GB |

- **主线一致**：下载或加载数据 → 90/10 切分训练/验证 → 编码成 token → 用 `numpy` 数组（`uint16` 省内存）写成二进制文件落盘。
- **细节差异**：字符级版自己从文本里数出词表并生成 `meta.pkl`；openwebtext 版用 HuggingFace `datasets` 拉取并行分词、用 `numpy.memmap` 把 90 亿 token 直接映射写盘（不占内存）。
- **怎么跑**：在对应目录运行 `python prepare.py`。

---

## 训练全景：数学 ↔ 代码对照（速览图）

> 这一节把"模型长什么样、训练怎么跑"用**公式 ↔ 代码行号**的画法整体串一遍，适合通读源码前后对照着看。
> 为避免重复：名词解释看「术语表」、脚本职责看「项目文件导览」、前向结构的详细问答看「第 1 题」——本节只画图、给行号，不重复展开原理。

### 一、一次参数更新的完整流水线（8 站）

这 8 站 = 训练主循环里跑一圈的核心动作（`train.py:727-772`）：

```
【1】随机抽样一批数据                 get_batch('train')           train.py:279
   数学：从数据集均匀随机抽 B 个起点各截 T 长；
        xᵢ = token[i : i+T]，yᵢ = token[i+1 : i+T+1]（右移一位=“下一个词”的答案）
   代码：data[start:start+BLOCK_SIZE] / 偏移 +1 的 target_batch    train.py:303/314

【2】前向传播 → logits + loss         gpt_model(input_batch, target_batch)  train.py:749
   数学：(a) 嵌入相加  h₀ = E(x) + P(位置)
        (b) N 个 Block  hₐ = Blockₐ(hₐ₋₁)（内部见下方视图二）
        (c) 出分数    logits = W_head · LN(h₁₂)                    model.py:378-388
        (d) 交叉熵    ℒ = −(1/Σ)·Σ log P(yₜ | x<ₜ)                 model.py:389

【3】除以累积步数                     loss / GRADIENT_ACCUMULATION_STEPS  train.py:750
   数学：ℒ ← ℒ/G。攒 G 个微批的梯度加总，尺度才等于一个“大 Batch”

【4】放大损失（仅 fp16）             gradient_scaler.scale(loss)  train.py:759
   数学：ℒ̃ = 65536·ℒ —— 不改极值点，防微小梯度下溢成 0

【5】反向传播（算梯度）              .backward()                 train.py:759
   数学：∇θℒ̃ = 65536·∇θℒ，即每个参数的梯度放大 65536 倍
   实现：链式法则沿计算图倒走，把偏导存进「参数.grad」

【6】还原梯度 + 裁剪                 unscale_ + clip_grad_norm_  train.py:764-765
   数学：∇ ← ∇/65536；若 ‖∇‖ > g 则 ∇ ← ∇·g/‖∇‖（限幅防梯度爆炸）

【7】优化器真正更新参数              gradient_scaler.step(optimizer)  train.py:768
   数学：AdamW（动量 + 自适应学习率，公式见下方表格）
   溢出检查通过才执行；梯度若已 inf/NaN 则本轮跳过更新

【8】清空梯度                        optimizer.zero_grad(set_to_none=True)  train.py:772
   数学：.grad 置 0，参数不动，下一轮从干净状态重新累积
```

第 1~8 站 = 一次“大更新”；外层 `while True` 每圈走一次（`train.py:685`）。

### 二、前向传播内部展开（model.py）

```
x(整数id) ──词嵌入表──→ E(x) (B,T,768)
位置 0..T-1 ──位置表──→ P      (B,T,768)      h₀ = E + P         model.py:378-380
                          ▼
          ┌──── Block ×N（model.py:381-382）─┐
          │ ① 注意力（跨位置交换信息）  CausalSelfAttention      model.py:98
          │    h ← h + Attn( LN(h) )      ← 残差 + PreLN
          │    Q,K,V = h·W（一次性大投影再 split）               model.py:134
          │    A = softmax(QKᵀ/√d_head + 因果掩码)；out = A·V     model.py:145-153
          │    多头拼回 → 输出投影 Wₒ      （多头细节见 1.5）
          │ ② MLP（逐位置深加工）         MLP                     model.py:155
          │    h ← h + W₂·GELU( W₁·LN(h) )（中间宽 4×768，见 1.4）
          └──────────────────────────────────────┘
                          ▼
     h₁₂ ──Final LN──→ ──W_head(与嵌入共享，weight tying)──→ logits   model.py:383-388
                          ▼
     ℒ = cross_entropy(logits, y)：逐 token 的“下一个词预测误差”      model.py:389
```

### 三、外层训练大纲（迭代 + 调度 + 评估 + 存盘）

```
while True:                                                  train.py:685
├─【定学习率】 η = get_learning_rate(iteration_number)       train.py:689
│    · t < 预热: η = η_max·(t+1)/(T_warm+1)            线性升温   train.py:620
│    · t > 衰减: η = η_min                              固定最低   train.py:626
│    · 中间:   η = η_min + (η_max−η_min)·½(1+cos(π·r)) 余弦衰减   train.py:630-633
│
├─【定期评估】it % EVALUATION_INTERVAL == 0                    train.py:696
│    train/val 各取 E 批、纯前向求平均损失 → 打印               estimate_loss  train.py:583
│    val loss 创新低 → 记录 best_validation_loss（最优模型依据） train.py:710
│
├─【定期存盘】it % CHECKPOINT_INTERVAL == 0                     train.py:710-723
│    存：模型权重 / AdamW 状态 / iteration / 学习率 / best_val → resume 无缝续训 train.py:453
│
├─【跑一遍第 1~8 站】（for micro_step in range(G)）              train.py:731
├─【日志】打印 loss / 耗时 / MFU                               train.py:785-796
├─ iteration_number += 1;  local_iteration_number += 1          train.py:797-798
└─【终止】iteration_number > MAXIMUM_ITERATIONS → break          train.py:802
```

### 四、速查表：数学符号 ↔ 代码位置

| 数学上                         | 公式                              | 代码                            | 位置                |
| ------------------------------ | --------------------------------- | ------------------------------- | ------------------- |
| 采样一个 batch（下一个词预测） | (xᵢ, xᵢ+1) 错一位               | `get_batch`                   | train.py:279        |
| 嵌入相加                       | h₀ = E + P                       | `embedding_dropout(emb+pos)`  | model.py:380        |
| 因果注意力                     | A = softmax(QKᵀ/√d + 掩码) · V | `CausalSelfAttention.forward` | model.py:98/145-158 |
| MLP 前馈                       | W₂·GELU(W₁x)                   | `MLP.forward`                 | model.py:155        |
| 语言模型损失                   | ℒ = −Σlog P(yₜ\|x<ₜ)         | `cross_entropy(...)`          | model.py:389        |
| 梯度                           | ∇θℒ                            | `loss.backward()`             | train.py:759        |
| 梯度缩放（fp16）               | ℒ̃ = s·ℒ                      | `GradScaler.scale`            | train.py:759        |
| 梯度裁剪                       | 超范数限幅                        | `clip_grad_norm_`             | train.py:765        |
| 优化器更新 (AdamW)             | θ ← θ − η·m̂/(√v̂+ε)    | `GradScaler.step(optimizer)`  | train.py:768        |
| 热身+余弦学习率                | 见视图三                          | `get_learning_rate`           | train.py:616        |

一句话把整个训练串起来：**不停地「随机抽一批 → 前向算误差 → 反向求梯度 → 裁剪后沿下坡方向挪参数」，期间按节奏调学习率、评估、存盘，直到迭代数用尽——模型从随机初始值被一圈圈滚进损失的低谷。**

---

## 第 1 题：讲讲 nanoGPT 里 GPT 的整体结构

**问题**

讲讲 nanoGPT 里 GPT 的整体结构。

**回答**

nanoGPT 的整个模型定义都在 `model.py` 一个文件里，是一个标准的 **decoder-only Transformer**（GPT-2 结构）。从**数据流**自顶向下看：

**1. 输入嵌入层**

- Token 嵌入 `word_token_embedding`（`torch.nn.Embedding(vocabulary_size, embedding_dimension)`）：token id → `embedding_dimension` 维向量。
- 位置嵌入 `word_position_embedding`（`torch.nn.Embedding(block_size, embedding_dimension)`）：**可学习的绝对位置嵌入**，不是 sin/cos，也不是 RoPE。
- 两者**相加**后过 dropout（`model.py:266`），得到主干输入 `x`，形状 `(批, 序列, embedding_dimension)`。

**2. 主干：N 个 Block 堆叠**
`number_of_layers` 个 `Block` 串联（默认 12 层），每个 Block 用 **Pre-LN**（先归一化再进子层），残差写法（`model.py:176-178`）：

```python
x = x + self.attention(self.layer_normalization_before_attention(x))                            # 自注意力子层
x = x + self.multi_layer_perceptron(self.layer_normalization_before_multi_layer_perceptron(x))  # 前馈子层
```

- **CausalSelfAttention**：`combined_query_key_value_projection` 一次算出 Q/K/V（打包成 `3*embedding_dimension` 再 split），reshape 成多头 `(批, 头数, 序列, 每头维度)`；核心是**因果**（每个位置只能看左边）；有 Flash Attention 走 `scaled_dot_product_attention`，否则手写 softmax；最后 `output_projection` 投影回去。
- **MLP**：两层全连接，中间升维到 `4*embedding_dimension`，激活用 **GELU**（`model.py:155-157`）。

**3. 输出层MLP**

- final LayerNorm `final_layer_normalization`。
- `language_model_head`（`torch.nn.Linear(embedding_dimension, vocabulary_size, bias=False)`）→ 词表 logits。

**值得强调的设计点**

1. **权重共享（weight tying）**：`word_token_embedding.weight = language_model_head.weight`（`model.py:215`），输入嵌入与输出投影共用权重。
2. **LayerNorm 带可选 bias**：自定义 `LayerNorm` 类以支持 `bias=False`。
3. **残差投影特殊初始化**：`output_projection.weight` 用 `std=0.02/sqrt(2*number_of_layers)`（`model.py:224`），GPT-2 论文做法，防止深层方差爆炸。
4. **vocabulary_size 补齐到 50304**：本为 50257，凑 64 倍数为 GPU 效率。
5. **训练/推理两条路径**：给 `targets` 算 cross-entropy loss；不给则只对最后一个位置跑 `language_model_head`（`model.py:279`）。

一句话：**嵌入相加 → N 层 (Pre-LN + 因果多头注意力 + GELU MLP，均带残差) → final LN → 权重共享的 language_model_head 出 logits**。

**点评**

回答覆盖了整体结构、数据流、以及 5 个容易被追问的设计细节（weight tying、Pre-LN、缩放初始化、vocab padding、双路径 forward），并落到具体行号，属于比较完整的答法。可进一步深挖的追问方向：Pre-LN vs Post-LN 的区别、为什么残差投影要那样缩放初始化、绝对位置嵌入的局限（外推能力差）。

#### 1.1 为什么要将 word_token_embedding 和 word_position_embedding 二者相加后过一个 dropout，分别介绍这两步（子问题）

**问题**

为什么要将 `word_token_embedding`（上游 `wte`）和 `word_position_embedding`（上游 `wpe`）二者相加后过一个 dropout？分别介绍这两步。

对应代码（`model.py:291-293`）：

```python
token_embeddings = self.transformer.word_token_embedding(token_indices)        # (批, 序列, 嵌入维度)
position_embeddings = self.transformer.word_position_embedding(positions)      # (序列, 嵌入维度)
x = self.transformer.embedding_dropout(token_embeddings + position_embeddings)
```

**回答**

这行代码是两步：**先把两个嵌入相加，再对相加结果做一次 dropout**。

**第一步：为什么是「相加」token 嵌入 + 位置嵌入**

两个嵌入各自只回答半个问题：

- `word_token_embedding` 只回答「**这是哪个词**」。盲点是——同一个词无论排第几位，查出来的向量完全一样。
- `word_position_embedding` 只回答「**这是第几个位置**」。

为什么必须补位置信息：自注意力是**置换不变**的，它把输入当无序集合，打乱词序结果只跟着换位置、数值不变。光有 token 嵌入，模型眼里「猫 追 狗」和「狗 追 猫」没区别，所以必须注入词序。

为什么是相加而不是拼接（concat）：

1. **维度不变、零额外开销**：相加后仍是 `embedding_dimension` 维，后面所有层不用改；拼接会让维度翻倍，整个网络得变宽。
2. **信息没真丢**：`embedding_dimension` 很高维（默认 768），模型能在训练中自己学会用不同方向分别编码 token 信息和位置信息，后续有能力再拆开利用。
3. **广播自动对齐**：token 嵌入 `(批, 序列, 维度)` + 位置嵌入 `(序列, 维度)`，靠 PyTorch **广播**把位置嵌入复制到每个 batch——即同一 batch 里所有样本第 t 个位置都加同一个位置向量。

相加后，每个位置的向量同时带上「**是什么词 + 在什么位置**」，才够资格进注意力层。

**第二步：为什么相加后要过一次 dropout**

- **是什么**：训练时随机把向量里一部分元素临时置 0（其余按比例放大），每个 step 丢的位置都不同，是一种**正则化**，逼模型别过度依赖某几个特征、缓解过拟合。
- **为什么放这里**：这是模型入口，在入口注入随机扰动，正则化效果贯穿后面所有层。代码里叫 `embedding_dropout`（`model.py:293`），专作用于嵌入层输出。
- **关键前提**：dropout **只在训练时生效**，推理（`model.eval()`）时自动变成原样通过，否则生成结果会带随机扰动。
- **nanoGPT 实际取值**：默认 `dropout=0.0`，**预训练时通常关掉**——预训练数据海量、基本只过一遍，过拟合风险低；它主要在**数据量小、要微调**时才调大（如 0.1）。这行是「预留开关」，不是「一定开着」。

**一句话总结**：相加是为了让每个位置同时带上「词身份 + 位置」两重信息（选相加而非拼接是为维度不变、省算力，靠高维空间和广播保证信息不丢）；dropout 是入口处一道可选正则化，防过拟合、增强鲁棒，只在训练时起作用，nanoGPT 预训练默认关闭。

**点评**

答题把「一行代码 = 两步」拆开，各自讲清「是什么 / 为什么需要 / 为什么这样设计」。两个高频追问点回答到位：**相加 vs 拼接**（考点是维度与信息保留的权衡）、**dropout 训练/推理行为差异 + nanoGPT 默认为 0**（考点是知不知道预训练一般不用 dropout）。还可延伸：绝对位置嵌入 vs RoPE/ALiBi、为什么位置嵌入不做 dropout 单列而是和 token 嵌入相加后一起做。

#### 1.2 讲解词嵌入的具体过程（对应到实际代码）（子问题）

**问题**

讲解词嵌入（`word_token_embedding`）的具体过程，并对应到实际代码。

**回答**

「词嵌入」在代码里其实就是**一次查表**。按「建表 → 输入 → 查表 → 反向更新 → 复用」的顺序，全程对到代码：

**第 0 步：输入是什么**

进入模型的不是文字，而是一串**整数**（token id）。文本在上游已被 tokenizer 切成 token 并映射成编号，喂给模型的是 `token_indices`，形状 `(batch_size, sequence_length)`，每个元素是 `0 ~ vocabulary_size-1` 之间的整数。

**第 1 步：建一张「嵌入表」（定义处 `model.py:226`）**

```python
word_token_embedding = torch.nn.Embedding(config.vocabulary_size, config.embedding_dimension)
```

- `torch.nn.Embedding(N, D)` 内部就是**一个形状 `(N, D)` 的可学习权重矩阵**。这里 `N=vocabulary_size=50304`，`D=embedding_dimension=768`，是一张 `(50304, 768)` 的大表。
- 像一本**字典**：50304 行，第 i 行 = 「编号 i 的那个 token」专属的 768 维向量。
- 初始是**随机数**（`_initialize_weights` 里正态分布 `std=0.02`），此刻没有语义，语义靠训练学出来。

**第 2 步：查表（使用处 `model.py:291`）**

```python
token_embeddings = self.transformer.word_token_embedding(token_indices)  # (批, 序列, 嵌入维度)
```

- 做的事很朴素：**拿每个整数当行号，去表里把对应那一行抠出来**。
- 形状从 `(batch, sequence)` 变成 `(batch, sequence, embedding_dimension)`——给每个整数「贴上」768 维向量。
- **关键理解点**：查表在数学上**等价于「one-hot 向量 × 嵌入矩阵」**（把 id 变成只有第 i 位为 1 的 one-hot，再乘表，正好取第 i 行）。但真算会又慢又费内存，PyTorch 底层用**按行索引（gather）** 实现、跳过 one-hot。理解成 one-hot 乘法有助于想通「它是线性层、梯度怎么回传」，实现上则是纯查表。

**第 3 步：它是「可学习」的——反向传播只更新用到的行**

- 嵌入表是 `nn.Parameter`，随训练更新。
- 前向只取「本批次出现过的 token」对应行；反向时**梯度只回流到这些行**，朝减小 loss 的方向微调，没出现的行这步不动。
- 训练久了，表从随机数变成有语义的向量：意思相近的词向量彼此靠近。这正是「嵌入」得名的原因——把离散的词**嵌入**进连续向量空间。

**第 4 步：这张表还被复用为输出层（权重共享 `model.py:242`）**

```python
self.transformer.word_token_embedding.weight = self.language_model_head.weight
```

同一张 `(50304, 768)` 的表，**入口当查表、出口当分类权重**（weight tying），一套参数两头用。

**一句话总结**：词嵌入 = 一张 `(vocabulary_size, embedding_dimension)` 的可学习查找表（`nn.Embedding`）；前向就是**拿 token id 当行号抠出对应向量**（等价于 one-hot 乘矩阵，但实现为高效索引），形状 `(批, 序列)` → `(批, 序列, 维度)`；向量随训练更新、意思相近的词彼此靠近；在 nanoGPT 里还与输出头共享权重。

**点评**

回答抓住了本质——「嵌入 = 查表」，并讲清了三个高频追问点：**查表 ⇔ one-hot 乘矩阵的等价关系**（考点是理不理解它其实是个线性层）、**梯度只更新被用到的行**（考点是稀疏更新）、**初始随机→训练出语义**。全程对到 3 处代码行号（建表 226 / 查表 291 / 权重共享 242）。可延伸：`nn.Embedding` 的 `padding_idx` 参数、嵌入维度选多大、词嵌入与 word2vec 的异同。

#### 1.3 词嵌入表的「可学习」具体是怎么实现的（子问题）

**问题**

上一题第 3 步说嵌入表是「可学习」的、反向传播只更新用到的行。这里的「可学习」具体是怎么实现的？

**回答**

「可学习」不是一个魔法开关，而是**五个环节接力**：建表 → 初始化 → 登记为参数 → 前向建图求 loss → 反向求梯度 → 优化器更新。下面**每一步都贴出项目里对应的真实代码**。

**环节 1：建表——这张表天生就是「参数」（`model.py:226`）**

```python
# model.py:226
word_token_embedding = torch.nn.Embedding(config.vocabulary_size, config.embedding_dimension),
```

`torch.nn.Embedding` 内部的 `weight` 就是一个 `torch.nn.Parameter`。`Parameter` 是「可学习」的根源，它有两个天生属性：`requires_grad=True`（要算梯度）、自动被 `model.parameters()` / `model.named_parameters()` 收集（优化器才拿得到）。

**环节 2：初始化——把表填成随机数（`model.py:246` + `model.py:280-281`）**

```python
# model.py:246  —— 对所有子模块套用初始化函数
self.apply(self._initialize_weights)

# model.py:280-281  —— 嵌入表用正态分布 std=0.02 填充
elif isinstance(module, torch.nn.Embedding):
    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

所谓「学习」，就是把这堆随机数一步步改成有语义的数。

**环节 3：登记为「要优化的参数」——喂给优化器（`model.py:399-402` + `model.py:422`）**

```python
# model.py:399  —— 取出全部参数
parameter_dictionary = {parameter_name: parameter for parameter_name, parameter in self.named_parameters()}
# model.py:402  —— 只保留 requires_grad=True 的（可学习的）
parameter_dictionary = {name: p for name, p in parameter_dictionary.items() if p.requires_grad}
...
# model.py:422  —— 把这些参数交给 AdamW 优化器接管
optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate, betas=betas, **extra_arguments)
```

这一步明确回答了「谁来更新它」：正是靠 `requires_grad` 把嵌入表筛进 `optimizer` 的管辖范围。

**环节 4：前向传播——建计算图，一路算到 loss（`model.py:291` → `model.py:293` → `model.py:301-302`）**

```python
# model.py:291  —— 查表（gather），本步只取用「本批次出现过的行」
token_embeddings = self.transformer.word_token_embedding(token_indices)
# model.py:293  —— 参与后续运算
x = self.transformer.embedding_dropout(token_embeddings + position_embeddings)
# ...经过所有 Block、final LN 之后...
# model.py:301-302  —— 算出 loss
logits = self.language_model_head(x)
loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```

从查表到 loss 的每一步，autograd 都在背后记进**计算图**，作为反向求导的地图。查表（gather）是可微操作，也在图里。

**环节 5：反向传播——`backward()` 算梯度，存进 `.grad`（`train.py:441` + `train.py:448`）**

```python
# train.py:441  —— 前向，拿到 loss
logits, loss = gpt_model(input_batch, target_batch)
# train.py:448  —— 反向，沿计算图求出每个参数的梯度
gradient_scaler.scale(loss).backward()
```

`backward()` 用链式法则算出「loss 对每个参数的偏导数」= **梯度**，存进每个参数的 `.grad`。
**对嵌入表的特殊之处（= 上题「只更新用到的行」的来由）**：查表本质是「按行号取行」，loss 只依赖本批次取用过的行；反向时梯度只**累加回这些行**（底层用 `index_add` 散射回对应行号），没用到的行梯度为 0。

**环节 6：优化器更新——`step()` 真正改数值，再清空梯度（`train.py:456` + `train.py:460`）**

```python
# train.py:456  —— AdamW 按 参数 ← 参数 − lr × 更新量 就地修改参数
gradient_scaler.step(optimizer)
# train.py:460  —— 清空 .grad，否则下一轮会和这轮累加
optimizer.zero_grad(set_to_none=True)
```

嵌入表在这一步朝「减小 loss」的方向挪一点点；梯度为 0 的行几乎不动。

**一次「学习」= 环节 4→5→6 循环成千上万遍**：`前向(建图求loss) → backward(求梯度) → step(更新) → zero_grad(清空)`，每个 batch 走一遍，跑几十万 batch 后随机表就变成有语义的词向量。

**一句话总结**：「可学习」落地全程可在项目里逐行对上——`nn.Embedding` 建表即 `Parameter`（226）→ 随机初始化（246/280）→ `requires_grad` 筛进 `AdamW`（402/422）→ 前向建图算 loss（291/301）→ `backward()` 求梯度（448）→ `step()` 更新 + `zero_grad()` 清空（456/460）。

**点评**

按面试官要求改为「每一环节都贴出项目真实代码 + 行号」的形式，把「可学习」拆成 6 个环节、8 处代码引用（`model.py` 226/246/280/399/402/422/291/301，`train.py` 441/448/456/460），完整覆盖「建表→初始化→登记→前向→反向→更新」全链路，并保留了「嵌入表只更新用到的行（`index_add` 稀疏散射）」这一考点。可延伸：梯度累积、梯度裁剪、混合精度下 `GradScaler` 为什么 scale/unscale。

#### 1.4 MLP 子层为什么要先升维再降维，这一步在代码中怎么操作（子问题）

**问题**

MLP 子层为什么要先升维再降维？这一步具体在代码里是怎么操作的？

**回答**

先看实际代码，一共就是**两个 `Linear` 夹一个激活**：

```python
# 定义（model.py:165-171）
self.expansion_projection = torch.nn.Linear(config.embedding_dimension, 4 * config.embedding_dimension, bias=config.bias)  # 768 → 3072 升维
self.gelu_activation      = torch.nn.GELU()                                                                                # 非线性
self.output_projection    = torch.nn.Linear(4 * config.embedding_dimension, config.embedding_dimension, bias=config.bias)  # 3072 → 768 降维
self.dropout_layer        = torch.nn.Dropout(config.dropout)

# 前向（model.py:174-179）
def forward(self, x):
    x = self.expansion_projection(x)  # (批,序列,768) → (批,序列,3072)
    x = self.gelu_activation(x)       # 逐元素非线性，形状不变
    x = self.output_projection(x)     # (批,序列,3072) → (批,序列,768)
    x = self.dropout_layer(x)
    return x
```

默认 `embedding_dimension=768`，中间撑到 `4×768=3072`，再压回 768。

**为什么要「先升维再降维」**

**1. 核心目的：给非线性一个更大的舞台，换取更强表达能力。** 关键不在「升维」本身，而在「**升维 + 中间 GELU + 降维**」三件套。把 768 维向量投到 3072 维，相当于临时准备了 **3072 个「中间神经元」**，每个神经元用 `expansion_projection` 的一行权重检测某种输入模式；GELU 对它们做非线性筛选；`output_projection` 再把这些特征重新组合、压回 768。中间越宽、能同时检测的模式越多——对应「宽度决定容量」。

**2. 为什么中间必须夹一个 GELU（否则升维白做）。** 去掉 GELU，`output_projection(expansion_projection(x))` 就是「两个线性层连乘」，数学上等价于**一个** `Linear(768,768)`，升到 3072 毫无意义（线性套线性还是线性）。正是中间的非线性，让「升维→降维」成为一个有表达力的非线性变换。

**3. 为什么最后必须「降回」768。** 两个硬约束：

- **残差连接要求同形状**：Block 里是 `x = x + multi_layer_perceptron(...)`（`model.py:196`），加号两边形状必须一致（都是 768）。
- **下一个 Block 期待 768 维输入**：整条主干的「残差流」宽度统一为 `embedding_dimension`，MLP 只是临时「鼓出去加工再收回来」，不能改变主干宽度。

**4. 为什么是「4 倍」。** 4 倍是 2017 年 Transformer 原论文的经验值，兼顾表达力与算量，GPT 系列一直沿用；是工程惯例而非理论最优。

**5. 顺带：MLP 是「逐位置」加工（channel mixing）。** `torch.nn.Linear` 只作用在最后一维，对序列每个位置**独立**做同样的升维-激活-降维，位置间不交换信息；跨位置信息交换全靠注意力子层。所以 MLP「把每个词自己的特征加工深」，注意力「让词与词交换信息」，分工互补。

**一句话总结**：MLP 用 `expansion_projection` 升到 4 倍宽（`model.py:165`）→ 在更宽空间里用 **GELU** 施加非线性（`model.py:167`）→ 用 `output_projection` 降回原维度（`model.py:169`）；升维是为给非线性更大舞台，中间 GELU 是升维有意义的前提（否则退化成一个线性层），降回原维度是被残差连接与主干统一宽度这两个约束逼出来的。

**点评**

回答严格贴代码（`model.py:165/167/169/174-179`），并把「为什么」拆成 5 点，其中最硬的两个考点答到位：**中间非线性不可省**（否则两个 Linear 退化成一个）、**必须降回原维度是残差连接逼出来的**。还补了「逐位置 channel mixing」与注意力的分工对照。可延伸：GELU vs ReLU、SwiGLU/GeGLU 等 gated MLP 变体、MLP 占参数量约 2/3。

#### 1.5 注意力子层为什么先算 Q/K/V 再拆成多头，对应代码怎么操作（子问题）

**问题**

注意力子层为什么要先算 Q/K/V，再拆成多头？对应代码是怎么操作的？

**回答**

这是两件事：**（一）为什么先把 x 投影成 Q/K/V（还合并成一次矩阵乘法）**，**（二）为什么算完再拆头**。

**一、先算 Q/K/V：为什么 + 代码**

自注意力要让同一个输入 `x` 扮演三种角色：**Query**（我想找什么）、**Key**（我拿什么标签供匹配）、**Value**（匹配后要汇总的信息）。Q·K 点积得注意力权重，权重再加权 V。三者都是 `x` 经过**不同可学习线性投影**得来，所以必须先「算出 Q/K/V」。

为什么合并成一个 `Linear(768→2304)` 而非三个独立 Linear——**效率**：GPU 上一个大矩阵乘法比三个小的核函数利用率更高，结果等价：

```python
# 定义（model.py:100）
self.combined_query_key_value_projection = torch.nn.Linear(config.embedding_dimension, 3 * config.embedding_dimension, bias=config.bias)
# 前向（model.py:134）：一次算出 2304 维，再 split 成各 768 维
query, key, value = self.combined_query_key_value_projection(x).split(self.embedding_dimension, dim=2)
```

`.split(768, dim=2)` 把 `(批,序列,2304)` 沿最后一维每 768 切一段，得到 Q、K、V 各 `(批,序列,768)`。

**二、再拆成多头：为什么 + 代码**

只用「一个 768 维大注意力」全序列只能学**一种**关注模式；拆成 `12` 头 × `64` 维，**每头在自己 64 维子空间独立算一套注意力**，12 个头并行地从不同角度看关系（语法/指代/远距离依赖…），表达更多样。

关键：拆头**几乎不花钱**——Q/K/V 已是 768 维，拆头只是把它**重新解释**成「12×64」，是一次 `view`，不增参数、不增计算：

```python
# model.py:135-138
head_dimension = embedding_dimension // self.number_of_attention_heads          # 768 // 12 = 64
key   = key.view(batch_size, sequence_length, self.number_of_attention_heads, head_dimension).transpose(1, 2)   # (批,序列,768)→(批,序列,12,64)→(批,12,序列,64)
query = query.view(...).transpose(1, 2)
value = value.view(...).transpose(1, 2)
```

- **`.view(批,序列,12,64)`**：把 768 切成 `12×64`，即分头。
- **`.transpose(1,2)`**：把「头」维挪到第 1 维 → `(批,12,序列,64)`，让「头」像「批」一样成外层维度，后面注意力矩阵乘法对每个头**独立批量计算**（`model.py:145` Flash 或 `149-153` 手写），头间互不干扰。

算完拼回 + 输出投影：

```python
# model.py:154：transpose 挪回 + view 拼接，(批,12,序列,64)→(批,序列,768)
y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dimension)
# model.py:158：输出投影，让各头信息真正融合
y = self.residual_dropout(self.output_projection(y))
```

拼接只是把 12 头并排放好，`output_projection`（`model.py:104`）才让**头与头信息混合**。

**为什么是「先算 Q/K/V、再拆头」这个顺序**：「先全宽度投影、再按头切分」数学上**等价于**「每头各有一套 64 维 Q/K/V 投影」，但前者 = 一个大矩阵乘法 + 一次几乎免费的 reshape，后者 = 36 个小矩阵乘法。前者对 GPU 友好得多。所以投影必须在全宽度（768）上一次算完，拆头只能是算完后的廉价视图操作。

**一句话总结**：用合并 `Linear(768→2304)` 一次算出 Q/K/V（`model.py:100/134`，合并为 GPU 效率）→ `view+transpose` 把 768 维**免费**重解释成「12 头×64 维」并把头维提到批维旁（`model.py:135-138`），让每头在各自子空间并行做注意力 → `transpose+view` 拼回 768（`model.py:154`）→ `output_projection` 融合各头（`model.py:158`）。

**点评**

把问题拆成「Q/K/V 投影」和「拆头」两层，每层都对到代码，命中三个高频考点：**合并 QKV 投影是为 GPU 效率**（一个大 matmul vs 三个小的）、**拆头是零成本的 `view`（不增参数/计算）**、**`output_projection` 才真正融合多头**。还点明了「先投影后拆头」的顺序理由（等价性 + 效率）。可延伸：为什么除以 `sqrt(head_dimension)` 做缩放、因果掩码怎么加、MHA vs MQA/GQA。

---

## 附录：候选题库

> 围绕本项目由浅入深的常见面试问题清单，供面试官选用。`✅` 表示已问过（记录在正文对应题号）。

### 一、模型结构（`model.py`）

- ✅ 讲讲 GPT 整体结构（第 1 题）
- ✅ 为什么嵌入相加后过 dropout（1.1）
- ✅ 词嵌入的具体过程（1.2）
- ✅「可学习」具体怎么实现（1.3）
- ✅ MLP 为什么先升维再降维（1.4）
- ✅ 注意力为什么先算 Q/K/V 再拆多头（1.5）

- [ ] 注意力权重为什么除以 `√head_dimension`？不缩放会怎样？（`model.py:149`）
- [ ] **因果掩码**如何实现「只能看左边」？Flash 版与手写版各怎么加？（`model.py:145` / `150`）
- [ ] 为什么自定义 `LayerNorm` 而不用 `nn.LayerNorm`？`bias=False` 的好处？
- [ ] **权重共享**（`wte` 与 `lm_head`）为什么合理？省多少参数？
- [ ] 残差投影为什么用 `std=0.02/√(2·n_layer)` 初始化？（`model.py:251`）
- [ ] `vocab_size` 为什么从 50257 补到 50304？
- [ ] `forward` 训练/推理为什么两条路径？`x[:, [-1], :]` 的 `[-1]` 为什么用列表？（`model.py:306`）

### 二、损失与生成

- [ ] `cross_entropy` 的 `ignore_index=-1` 干嘛的？为什么先 `view(-1, ...)`？（`model.py:302`）
- [ ] `generate` 的 `temperature` 和 `top_k` 各怎么影响采样？
- [ ] 生成时为什么每步都把上下文裁剪到 `block_size`？
- [ ] `generate` 有没有 **KV cache**？没有的话慢在哪？

### 三、训练循环（`train.py`）

- [ ] `configure_optimizers` 为什么分「衰减/不衰减」两组参数？（`model.py:407`）
- [ ] 学习率 **warmup + 余弦衰减** 为什么这么设计？（`get_learning_rate`）
- [ ] **梯度累积** 怎么模拟大 batch？loss 为什么除以累积步数？（`train.py:442`）
- [ ] **混合精度**：`autocast`、`GradScaler` 的 scale/unscale 各解决什么问题？
- [ ] `zero_grad(set_to_none=True)` 里 `set_to_none` 有什么讲究？
- [ ] DDP 为什么只在最后一个 micro-step 同步梯度？（`train.py:439`）

### 四、数据管道

- [ ] 数据怎么用 `np.memmap` 读？为什么不一次性 load 进内存？
- [ ] `get_batch` 怎么随机采样 `(x, y)`？`y` 为什么相对 `x` 错开一位？

### 五、GPT-2 权重加载 & 工程

- [ ] `from_pretrained` 加载 HF 权重时，为什么有些权重要 **转置**？（Conv1D vs Linear）
- [ ] `torch.compile` 做了什么？为什么能加速？
- [ ] `estimate_mfu`（算力利用率）怎么估？为什么关心 MFU？

### 六、概念八股（不限本项目）

- [ ] 为什么 GPT 用 **decoder-only** 而非完整 encoder-decoder？
- [ ] 绝对位置嵌入 vs RoPE vs ALiBi，各自优劣？nanoGPT 用哪种、有何局限？
- [ ] 自注意力的时间/空间复杂度？为什么长序列贵？
- [ ] 为什么需要残差连接和 LayerNorm？去掉会怎样？
