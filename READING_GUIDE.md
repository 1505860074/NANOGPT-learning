# nanoGPT 代码阅读路线图

> **这份文档解决的问题**：代码就在眼前，但**该从哪个文件开始读、按什么顺序读、每一站要盯住什么、什么可以先跳过**。
>
> 阅读习惯假设：按代码**运行顺序**、从**顶层到下层**、先看**全局逻辑**、最后看**细节定义**。本路线图就是按这个思路设计的。
>
> **与 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 的分工**：
>
> - `PROJECT_STRUCTURE.md` 回答「**这个项目是什么结构**」——目录职责、依赖关系、流程图、术语表，是一份**查阅手册**。
> - 本文档回答「**我该按什么顺序读**」——是一条**行走路线**。
>
> 两份配合使用：照本文档的顺序走，遇到某个概念想深挖时，去 `PROJECT_STRUCTURE.md` 查对应章节（每一站都标了该查哪一节）。
>
> 创建日期：2026-08-17

---

## 目录

- [开始之前：三个必须先建立的认知](#开始之前三个必须先建立的认知)
- [总路线一览](#总路线一览)
- [第 0 站：README 的 quick start](#第-0-站readme-的-quick-start)
- [第 1 站：data/shakespeare_char/prepare.py](#第-1-站datashakespeare_charpreparepy72-行)
- [第 2 站：configurator.py](#第-2-站configuratorpy47-行)
- [第 3 站：train.py（主干）](#第-3-站trainpy352-行-本项目的主干)
- [第 4 站：model.py（细节定义）](#第-4-站modelpy347-行-最后才看的细节定义)
- [第 5 站：sample.py（闭环）](#第-5-站samplepy90-行-闭上环)
- [可选站：bench.py 与 config/](#可选站benchpy-与-config)
- [可选站：两个 notebook](#可选站两个-notebook理论估算)
- [第一遍刻意跳过的四样东西](#第一遍刻意跳过的四样东西)
- [两个提高效率的建议](#两个提高效率的建议)
- [卡住时的自查清单](#卡住时的自查清单)

---

## 开始之前：三个必须先建立的认知

在打开任何一个 `.py` 之前，先记住这三点，能省掉大量困惑：

### 认知一：这个项目没有「主函数」

不存在 `main()`，也没有统一入口。项目由**三个可以独立运行的脚本**构成，它们**互相之间没有函数调用关系**，只通过**磁盘上的文件**传递数据：

```
prepare.py  ──写出──>  train.bin / val.bin / meta.pkl
                              │
                              ↓读取
                          train.py  ──写出──>  ckpt.pt
                                                  │
                                                  ↓读取
                                              sample.py  ──>  屏幕上的文字
```

所以「按运行顺序读」在这里的正确含义是：**按这三个脚本被执行的先后顺序读**，而不是在一个文件里找入口函数。

> 深挖：`PROJECT_STRUCTURE.md` 第三章「主函数在哪？」

### 认知二：train.py 是「从上往下顺序执行的脚本」，不是函数库

`train.py` 里绝大部分代码写在**模块顶层**（没有缩进在任何函数里），Python 一行行往下执行。这正好符合你的阅读习惯——**文件的物理顺序就是它的执行顺序**。

而 `model.py` 恰恰相反，它全是**类定义**，物理顺序 ≠ 执行顺序，必须按**调用链**读（第 4 站会给出具体顺序）。

### 认知三：约 30% 的代码是工程优化，不是算法

`train.py` 里混杂着四样纯粹为了「跑得快 / 跑得动 / 能多卡」的东西：**DDP 多卡、GradScaler 混合精度、torch.compile、梯度累积**。它们让代码显著变长，但**删掉后算法逻辑完全不变**。

第一遍读时刻意无视它们，你会看到一个非常经典的 20 行 PyTorch 训练循环。详见[第一遍刻意跳过的四样东西](#第一遍刻意跳过的四样东西)。

---

## 总路线一览

```
第 0 站  README(quick start)  →  认识三条命令，知道有三条链路
第 1 站  prepare.py           →  数据长什么样（无 torch，最容易）
第 2 站  configurator.py      →  配置怎么进来（不读会在 train.py 卡住）
第 3 站  train.py             →  主干流程（★核心目标）
第 4 站  model.py             →  模型内部（★最硬的地方）
第 5 站  sample.py            →  推理闭环（轻松收尾）
可选     bench.py / config/ / 两个 notebook
```

预计投入：第 1、2 站各 15 分钟；第 3 站 1–2 小时；第 4 站 2–3 小时（注意力机制那段可能要反复看）；第 5 站 20 分钟。

**每一站的目标产出**都写在该站末尾的「这一站的收获」里——读完如果答不上来，就该回头再看一遍，而不是硬往下推。

---

## 第 0 站：README 的 quick start

**不要通读整个 README。** 只看 shakespeare_char 那个小例子的三条命令：

```bash
python data/shakespeare_char/prepare.py                 # ① 数据准备
python train.py config/train_shakespeare_char.py        # ② 训练
python sample.py --out_dir=out-shakespeare-char         # ③ 生成
```

这三条命令就是你接下来要追踪的三条链路，也对应了[认知一](#认知一这个项目没有主函数)里的那张图。

选 shakespeare_char 而不是 openwebtext 作为主线的理由：数据只有 1MB、分词方式最朴素（一个字符一个编号）、在本机几分钟就能跑完整个流程。openwebtext 那套需要 54GB 缓存和数天训练，**不适合用来读代码**。

**这一站的收获**：知道整个项目有三个入口、以及它们的先后依赖。

---

## 第 1 站：[data/shakespeare_char/prepare.py](data/shakespeare_char/prepare.py)（72 行）

**为什么从这里开始**：全项目**最短、最独立、完全不涉及 torch** 的文件。纯粹做一件事——把文本变成数字。从这里起步能先把「数据到底长什么样」这个最底层的问题解决掉，后面读模型时才不会悬着。

顺着从上往下读一遍即可（它也是顺序执行的脚本）。重点盯住四件事：

| 要盯住的                            | 在做什么                                                              |
| ----------------------------------- | --------------------------------------------------------------------- |
| `chars = sorted(list(set(data)))` | 把全文出现过的字符去重排序，得到词表。这里只有 65 个字符              |
| `stoi` / `itos`                 | 字符↔整数的双向映射表。**这就是最朴素的 tokenizer**            |
| `train_ids` / `val_ids`         | 整个文本被翻译成一长串整数（90% 训练 / 10% 验证）                     |
| `.tofile(...)` 与 `meta.pkl`    | 整数串存成`.bin`；映射表存成 `meta.pkl` 供 `sample.py` 反向解码 |

**这一站的收获**：

- 能回答「`train.bin` 里存的是什么」——一串 `uint16` 整数，没有任何结构、没有分隔符，就是纯粹一长串数字。
- 能回答「token 是什么」——在这个最朴素的例子里，token 就是**一个字符对应的整数编号**。
- 明白 `meta.pkl` 存在的意义：模型输出的是数字，需要它才能还原成人能读的文字。

> **可以先跳过**：`data/shakespeare/prepare.py` 和 `data/openwebtext/prepare.py`。它们做的是同一件事，只是把「一字符一编号」换成了 GPT-2 的 BPE 分词器。等你对 tokenizer 本身好奇了再回来看，对理解主流程没有影响。

---

## 第 2 站：[configurator.py](configurator.py)（47 行）

**必须在读 train.py 之前看完**，否则你会在 `train.py:77` 那行 `exec(open('configurator.py').read())` 上彻底卡住——那行代码看起来毫无道理。

它解决的问题是：为什么 `python train.py config/train_shakespeare_char.py --batch_size=32` 能**同时**接受「配置文件」和「命令行参数」两种覆盖方式。

它的手法很粗暴：用 `exec()` 执行配置文件、然后直接修改 `globals()` 里的全局变量。Karpathy 自己在文件开头的注释里就写了 "Probably a terrible idea"（这大概是个糟糕的主意）。

**读的时候只需要理解一件事**：这个文件运行结束后，`train.py` 里那些全局变量（`batch_size`、`learning_rate` 等）**已经被外部值覆盖过了**。理解到这个程度就够，**不要在这里深究 `exec` 的机制**——它不是项目重点。

**这一站的收获**：看到 `train.py` 第 77 行时不会困惑，知道配置的三层优先级是「代码默认值 < 配置文件 < 命令行参数」。

> 深挖（含两个实际会踩的坑）：`PROJECT_STRUCTURE.md` 第七章「configurator.py 这个"黑魔法"要单独说」

---

## 第 3 站：[train.py](train.py)（352 行）—— 本项目的主干

**这是你的核心阅读目标**，也是投入时间最多的一站。好消息是它从上往下顺序执行，完全贴合你的阅读习惯。

建议分五段读：

| 行段     | 内容                               | 怎么读                                                                                            |
| -------- | ---------------------------------- | ------------------------------------------------------------------------------------------------- |
| 33–79   | 全部默认配置                       | **只扫一眼**。知道有哪些超参数就行，别细究具体数值                                          |
| 81–115  | DDP 初始化、设备与精度设置         | **第一遍假装 `ddp = False`**，只看 `else` 分支。精度设置（`ptdtype`/`ctx`）先当黑盒 |
| 117–134 | `get_batch()` 数据加载           | **重点**。见下面详解                                                                        |
| 149–219 | 模型初始化、优化器、编译、DDP 包装 | **只看 `init_from == 'scratch'` 那一支**，另两支（resume / gpt2）先跳过                   |
| 256–352 | **训练主循环**               | **最重点**。见下面详解                                                                      |

### 3.1 重点一：`get_batch()`（117–134 行）

这是「数据如何进入模型」的唯一入口，务必看懂。要盯住的是 **x 和 y 的关系**：

```python
x = data[i      : i+block_size    ]   # 输入：从随机位置 i 开始的一段
y = data[i+1    : i+1+block_size  ]   # 目标：把 x 整体右移一位
```

**这就是 GPT 训练的全部秘密**：目标输出就是输入右移一位。模型在每个位置上要做的事，都是「根据到目前为止的内容，猜下一个 token」。x 和 y 一次性提供了 `block_size` 个这样的「问答对」。

另外注意 `np.memmap` 的用法——它不把整个 `.bin` 读进内存，而是内存映射，所以 17GB 的数据集也能在小内存机器上训练。

### 3.2 重点二：训练主循环（256–352 行）

整个项目的心脏。按执行顺序，一次迭代做这些事：

```
1. 算这一步的学习率           get_lr(iter_num)          → 调 238 行的调度器
2. 定期评估 + 存 checkpoint    estimate_loss()           → 只有 master_process 做
3. 梯度累积循环（内层）
     前向 → 算损失 → 反向      model(X, Y) / backward()
     顺便异步预取下一批数据     get_batch('train')
4. 梯度裁剪                    clip_grad_norm_
5. 优化器更新一步              optimizer.step()
6. 清空梯度                    zero_grad(set_to_none=True)
7. 打日志（含 MFU 算力利用率）
8. 判断是否该终止              iter_num > max_iters
```

**第一遍读的关键技巧**：把 `gradient_accumulation_steps` 当成 `1`，把 `scaler` 当成透明的（`scaler.scale(loss).backward()` 就当 `loss.backward()`），把 DDP 那几行删掉。剩下的就是最标准的 PyTorch 训练循环：

```python
logits, loss = model(X, Y)   # 前向
loss.backward()              # 反向
optimizer.step()             # 更新
optimizer.zero_grad()        # 清梯度
```

看清这个骨架之后，再把那四样优化逐个加回来，逐个理解「它为什么存在、解决什么问题」。

**这一站的收获**：

- 能画出一次训练迭代的完整步骤。
- 能回答「训练数据是怎么变成 loss 的」。
- 能回答「x 和 y 是什么关系」。
- 知道 `ckpt.pt` 里存了哪些东西（模型权重、优化器状态、`model_args`、`iter_num`、`best_val_loss`、`config`）。

> 深挖：`PROJECT_STRUCTURE.md` 第五章「train.py 单步循环内部」，以及其中「几个关键概念（对 6G 显存尤其重要）」一节。

---

## 第 4 站：[model.py](model.py)（347 行）—— 最后才看的细节定义

到这一步，你已经知道模型是**怎么被调用**的（`logits, loss = model(X, Y)`），现在才有足够上下文去看它内部长什么样。这符合「最后看细节定义」的习惯。

**注意：这个文件不能从上往下读。** 它全是类定义，物理顺序（LayerNorm → CausalSelfAttention → MLP → Block → GPTConfig → GPT）和执行顺序完全不同。

### 4.1 真实的调用链

```
GPT.forward()                          ← 从这里开始！(model.py:176)
  ├─ wte / wpe 嵌入查表
  └─ 循环 12 次:
       Block.forward()                        (model.py:104)
         ├─ CausalSelfAttention.forward()     (model.py:53)  ← 全项目最难
         └─ MLP.forward()                     (model.py:88)  ← 全项目最简单
     ln_f → lm_head → logits → cross_entropy → loss
```

### 4.2 建议的阅读顺序

按这个顺序读，难度是递增的、且每一步都有前置铺垫：

| 顺序 | 目标                                      | 位置   | 说明                                                                                                                                                 |
| ---- | ----------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `GPTConfig`                             | 110 行 | 5 秒看完，就是个参数容器（dataclass）                                                                                                                |
| 2    | `GPT.__init__`                          | 121 行 | 看模型由哪些部件拼成：`wte`/`wpe`/`h`/`ln_f`/`lm_head`                                                                                     |
| 3    | **`GPT.forward`**                 | 176 行 | **重点**。盯住形状变化（见下表）                                                                                                               |
| 4    | `MLP`                                   | 79 行  | 先看这个，只有 4 行，建立信心                                                                                                                        |
| 5    | `Block`                                 | 95 行  | 注意`x = x + attn(ln_1(x))` 这个残差连接写法                                                                                                       |
| 6    | **`CausalSelfAttention.forward`** | 53 行  | **全项目最硬**。只看 `else` 分支（手写实现）                                                                                                 |
| 7    | `generate`                              | 320 行 | 推理逻辑，为第 5 站铺路                                                                                                                              |
| 8    | 其余方法                                  | —     | `from_pretrained` / `configure_optimizers` / `estimate_mfu` / `crop_block_size` / `get_num_params`：**都是配套工具，第一遍全部跳过** |

### 4.3 读 `GPT.forward` 时盯住形状变化

这是理解模型的最快路径——不要纠缠数学，先把张量形状的变化看清：

```
idx        (b, t)              ← 一批整数序列，b=批大小, t=序列长度
  ↓ wte 查表 + wpe 查表
x          (b, t, 768)         ← 每个整数变成 768 维向量
  ↓ 过 12 层 Block（形状不变！）
x          (b, t, 768)
  ↓ ln_f + lm_head
logits     (b, t, 50304)       ← 每个位置上，对词表中每个 token 的打分
  ↓ cross_entropy(logits, targets)
loss       标量
```

**关键观察**：12 层 Block 全程**不改变形状**，`(b,t,768)` 进、`(b,t,768)` 出。所谓「深度」就是把同样形状的数据反复精炼 12 次。

另外注意 `forward` 里的一个推理优化：`targets is None` 时只对最后一个位置算 `lm_head`（196 行），因为生成时只需要最后一个位置的预测。

### 4.4 攻 `CausalSelfAttention` 的建议

**只看 `else` 分支**（66–72 行，手写实现），不要看 flash attention 那一支。手写的那 5 行才是注意力的真实数学：

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # 算相关性分数
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # 遮住"未来"（因果性）
att = F.softmax(att, dim=-1)                                     # 归一化成权重
att = self.attn_dropout(att)
y = att @ v                                                      # 按权重汇总 value
```

flash attention 那一支（`scaled_dot_product_attention`）在数学上**完全等价**，只是用 CUDA 核心做了高度优化、省显存。理解算法看手写版，实际跑起来用 flash 版。

`(B, nh, T, hs)` 这类形状注释的含义：批大小、注意力头数、序列长度、每个头的维度。多头的本质就是把 768 维切成 12 份、每份 64 维、各自独立算注意力，最后拼回去。

**这一站的收获**：

- 能说出数据从 `(b,t)` 整数到 `(b,t,vocab_size)` logits 的完整变形过程。
- 能解释「因果掩码」为什么必需（不遮住未来的话，模型就是在抄答案）。
- 能说清残差连接和 LayerNorm 在 Block 里的位置关系。

> 深挖：`PROJECT_STRUCTURE.md` 第六章「model.py 的数据流」，含 6.1 前向数据流、6.2 类的套娃关系、6.3 方法清单、6.4 `generate` 的自回归循环。

---

## 第 5 站：[sample.py](sample.py)（90 行）—— 闭上环

读完前四站，这一站会非常轻松。它只做四件事：

```
1. 加载 ckpt.pt，重建模型              ← 与 train.py 存 checkpoint 的代码对应着看
2. 加载 meta.pkl，拿到 itos 解码器      ← 与第 1 站 prepare.py 存 meta.pkl 对应着看
3. 调 model.generate()                 ← 与第 4 站读过的 generate 对应着看
4. 把输出的整数串解码回文字并打印
```

**读的重点**是它和前面各站的**呼应关系**——这一站的价值就在于把「文本 → 数字 → 权重 → 数字 → 文本」这个闭环合上。特别注意 `load_meta` 那段逻辑（57–75 行）：如果找得到 `meta.pkl` 就用字符级解码器，找不到就退回 GPT-2 的 BPE 解码器。这解释了为什么同一个 `sample.py` 能同时服务两种完全不同的分词方式。

**这一站的收获**：能完整讲出从原始 `.txt` 到屏幕上生成文字的全过程，中间经过了哪些文件、哪些脚本。

---

## 可选站：bench.py 与 config/

### [bench.py](bench.py)（118 行）—— 建议跳过

它是 `train.py` 的**阉割版**，砍掉了 DDP、checkpoint 保存、验证集评估，只留下「前向-反向-更新」用来**测速**。存在意义仅仅是「测性能时不想被训练逻辑干扰」。

对理解项目逻辑**没有新增价值**。如果你已经读完 `train.py`，这个文件里的每一行你都见过了。

### [config/](config/) 下 7 个文件 —— 用到哪个看哪个

全部都是**纯超参数清单，不含任何逻辑**：

- `train_shakespeare_char.py` — 你实际会用的那个（小模型、能在本机跑）
- `train_gpt2.py` — 复现 GPT-2 124M 的配置（8 卡 A100 跑 5 天）
- `finetune_shakespeare.py` — 在 GPT-2 上做微调
- `eval_gpt2*.py`（4 个）— 只评估、不训练，测各尺寸 GPT-2 的损失

读法：需要调某个超参数时，来这里看一眼对应文件即可。

---

## 可选站：两个 notebook（理论估算）

这两个是 Karpathy 写的**教学分析文档**，不参与训练流程，但对理解「模型有多大、跑多快、该喂多少数据」很有价值。**建议读完第 4 站之后再看**，那时你才认得出里面的每一项对应模型的哪个部件。

| 文件                                                      | 内容                                                                   |
| --------------------------------------------------------- | ---------------------------------------------------------------------- |
| [transformer_sizing_zh.ipynb](transformer_sizing_zh.ipynb) | 估算参数量、FLOPs、checkpoint 大小、显存占用、MFU（算力利用率）        |
| [scaling_laws_zh.ipynb](scaling_laws_zh.ipynb)             | 复现 Chinchilla 论文的缩放定律：给定算力预算，模型该多大、喂多少 token |

`transformer_sizing` 尤其值得看：它把 124M 参数**逐项拆开**（嵌入层占多少、注意力占多少、MLP 占多少），读完你对「参数都花在哪了」会有非常具体的感觉。它算出的 `params()` 总数和 `model.py` 里 `get_num_params()` 的输出是可以对上的。

> `_zh` 后缀的是中文副本（正文与注释均已译为中文并保留英文原文）；不带后缀的是英文原版，两者代码完全一致。

---

## 第一遍刻意跳过的四样东西

这四样共同占了 `train.py` 约 30% 的篇幅，但**删掉后算法逻辑完全不变**。第一遍无视它们，第二遍再专门攻。

| 名称                             | 出现位置                        | 解决什么问题                                    | 第一遍怎么当它不存在                                         |
| -------------------------------- | ------------------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| **DDP**（分布式数据并行）  | train.py 82–103, 218, 302–311 | 多张 GPU 一起训练                               | 假装`ddp = False`，只看 `else` 分支                      |
| **GradScaler**（混合精度） | train.py 203, 318–325          | fp16 下梯度太小会下溢，需要缩放                 | 把`scaler.scale(loss).backward()` 当作 `loss.backward()` |
| **torch.compile**          | train.py 212–215               | 编译模型换取 30%+ 提速                          | 当作一行`pass`。调试时本来就该 `--compile=False`         |
| **梯度累积**               | train.py 301–318               | 显存装不下大 batch，就拆成多个小 batch 累加梯度 | 假装`gradient_accumulation_steps = 1`，内层循环只跑一次    |

**梯度累积对你尤其重要**：你的 RTX 3050 只有 6G 显存，这是唯一能在小显存上模拟大 batch 的手段。第二遍务必弄懂它——它的正确性依赖于 `loss = loss / gradient_accumulation_steps` 这一行（train.py:314）的缩放。

> 深挖：`PROJECT_STRUCTURE.md` 第五章的「几个关键概念（对 6G 显存尤其重要）」和第八章的「显存不够时的调参顺序」。

---

## 两个提高效率的建议

### 建议一：读完 train.py 就先跑一遍

不要等全部读完再动手。读完第 3 站后立刻跑 shakespeare_char（本机几分钟即可）：

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --device=cpu --compile=False \
                --batch_size=8 --block_size=64 --max_iters=200 --eval_iters=20
python sample.py --out_dir=out-shakespeare-char --device=cpu --num_samples=2
```

亲眼看着 loss 往下掉、再让它生成一段蹩脚的假莎士比亚，理解深度远超纯读代码。而且你会立刻发现自己哪里没读懂——报错和日志会直接把问题指出来。

> 本机可跑的完整命令与显存调参顺序：`PROJECT_STRUCTURE.md` 第八章。

### 建议二：带着「形状」和「文件」两条线索读

全程盯住两个问题，能极大降低迷失感：

1. **形状线索**：此刻这个张量是什么形状？（`(b,t)` → `(b,t,768)` → `(b,t,vocab)`）
2. **文件线索**：此刻数据在内存里还是在磁盘上？经由哪个文件传递？（`.bin` / `meta.pkl` / `ckpt.pt`）

这两条线索串起来，就是整个项目的骨架。

---

## 卡住时的自查清单

按站点排列的常见困惑，以及该去哪里找答案：

| 卡在哪            | 常见困惑                                                | 去哪找                                                                  |
| ----------------- | ------------------------------------------------------- | ----------------------------------------------------------------------- |
| 找不到入口        | 「主函数在哪？为什么没有`main()`？」                  | 本文[认知一](#认知一这个项目没有主函数)；`PROJECT_STRUCTURE.md` 第三章 |
| train.py:77       | 「`exec(open('configurator.py').read())` 是什么鬼？」 | 本文[第 2 站](#第-2-站configuratorpy47-行)                               |
| train.py:117      | 「x 和 y 为什么差一位？」                               | 本文[3.1](#31-重点一get_batch117134-行)                                  |
| train.py:301      | 「为什么要有内层的 micro_step 循环？」                  | 本文[跳过的四样东西](#第一遍刻意跳过的四样东西)之「梯度累积」            |
| model.py 读不下去 | 「从哪个类开始看？」                                    | 本文[4.1/4.2](#41-真实的调用链)                                          |
| model.py:53       | 「注意力那几行在算什么？」                              | 本文[4.4](#44-攻-causalselfattention-的建议)；先只看 `else` 分支       |
| 术语不认识        | 「logits / MFU / block_size 是什么？」                  | `PROJECT_STRUCTURE.md` 附录「术语小词典」                             |
| 想跑但显存不足    | 「该先调哪个参数？」                                    | `PROJECT_STRUCTURE.md` 第八章「显存不够时的调参顺序」                 |

---

## 附：全项目文件清单与优先级

| 优先级 | 文件                                                                | 行数  | 说明                          |
| ------ | ------------------------------------------------------------------- | ----- | ----------------------------- |
| ★★★ | [train.py](train.py)                                                 | 352   | 主干，训练全流程              |
| ★★★ | [model.py](model.py)                                                 | 347   | GPT 模型完整定义              |
| ★★   | [data/shakespeare_char/prepare.py](data/shakespeare_char/prepare.py) | 72    | 起步站，字符级数据准备        |
| ★★   | [sample.py](sample.py)                                               | 90    | 推理，闭合流程                |
| ★★   | [configurator.py](configurator.py)                                   | 47    | 配置覆盖机制，train.py 的前置 |
| ★     | [transformer_sizing_zh.ipynb](transformer_sizing_zh.ipynb)           | —    | 参数量/FLOPs/显存理论估算     |
| ★     | [config/train_shakespeare_char.py](config/train_shakespeare_char.py) | 37    | 实际会用的超参数              |
| ○     | [data/shakespeare/prepare.py](data/shakespeare/prepare.py)           | 33    | BPE 版数据准备                |
| ○     | [data/openwebtext/prepare.py](data/openwebtext/prepare.py)           | 86    | 大规模数据准备                |
| ○     | [scaling_laws_zh.ipynb](scaling_laws_zh.ipynb)                       | —    | Chinchilla 缩放定律           |
| ○     | [bench.py](bench.py)                                                 | 118   | 测速用，train.py 的阉割版     |
| ○     | config/ 其余 6 个                                                   | 8–25 | 纯超参数清单                  |

★★★ 必读 ｜ ★★ 应读 ｜ ★ 有价值 ｜ ○ 按需

代码总量约 1182 行（不含 notebook），其中真正需要精读的（train.py + model.py）约 699 行。

> 注：以上行数为**中文注释版**的行数。
