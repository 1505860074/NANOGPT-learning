# torch 机制教学参考（给 opencode 用的教参）

> 用途：这份文档不是给用户阅读的学习材料，而是给我的讲解教案。
> 用户要求：围绕 nanoGPT 项目相关的 torch 知识，按「讲点」逐个讲解；
> 每讲完一个点就停下来，等用户实时提问后再进入下一个点。

---

## 讲解协议

- 每次只讲**一个讲点**，讲完立即停下，提示用户可提问或继续。
- 用户可以随时打断、追问；追问范围内可以发散，但主线以本文件的讲点顺序为准。
- 讲解语气：温柔大姐姐人设；技术内容保持准确；适当穿插安抚与夸奖，不喧宾夺主。
- 不必要对照代码时可以对照；必要时结合 model.py / train.py 具体行号。

## 分点计划（共 7 个讲点）

1. 张量与 Parameter —— 「旋钮」的身份从哪来
2. nn.Module 机制 —— 类为什么能像函数一样被调用
3. Autograd 计算图 —— 前向搭图、反向走图
4. backward → optimizer 完整链路 —— 一场标准参数更新
5. 参数分组与混合精度 —— 优化器怎么挑旋钮、fp16 的坑
6. 面试题准备清单 A / B / C 三档

---

## 点 1 · 张量与 Parameter

**知识骨架**

- `Tensor` = 多维数组 + dtype + device。`(B, T, C)` 形状、`float32/float16`、`cpu/gpu` 三要素决定一块数据怎么算。
- `Parameter` 是「可学习」的源头身份：
  - `torch.tensor([...])` → 中间结果，与训练无关；
  - `torch.nn.Parameter(torch.randn(...))` → 模型旋钮，训练要拧它。
- Parameter 与生俱来的三件套：
  1. `requires_grad=True`（要算梯度）；
  2. 放进 `Module` 后能被 `parameters()` 收集（优化器拿得到）；
  3. 参与 `backward()` 后 `.grad` 被填上，`optimizer.step()` 据此更新。
- 「需要梯度」从来不是手标出来的，而是 Parameter 这类张量的天生属性。
- LayerNorm 的 weight（缩放）与 bias（平移）只是 Parameter 的两个用例；嵌入表、Linear 的 W 也全是 Parameter，形状不限。

**常见追问**

- 为什么项目里找不到一行 `requires_grad=True`？→ 因为是 Parameter 类型自带的默认值。
- 哪些不是 Parameter？→ buffer（如因果掩码，`register_buffer` 注册），跟模型走但不学习。

---

## 点 2 · nn.Module 机制

**知识骨架**

- 模块树：`Module` 里塞子 `Module`，`nn.ModuleDict / ModuleList` 专门装一堆子模块；塞进去的瞬间子模块的参数与 buffer 自动挂到父模块名下。
- `__call__` → `forward`：`model(x)` 由 `Module.__call__` 兜底，内部转交 `forward()`。**不是类能返回东西，是 `__call__` 让实例能当函数用。**
- 魔法方法（dunder method）：名字固定带双下划线，Python 语法在对应场景自动触发。如 `obj(...)` → `__call__`、`len(obj)` → `__len__`、`x+y` → `__add__`。一个类只能有一个 `__call__`（同名覆盖），要多签名用 `*args/**kwargs` 自行分发。
- 两个遍历口子：
  - `parameters() / named_parameters()` → 要学的旋钮；
  - `buffers() / named_buffers()` → 跟模型走但不学的状态。
- `state_dict()`：参数+buffer 拍成字典，save/load 和优化器恢复状态的依据。
- `train() / eval()`：只切换**随机层（dropout/BatchNorm）**行为，**完全不碰梯度**；与 `no_grad` 是两条独立轴（面试陷阱）。

**常见追问**

- `module.forward(x)` 直接调用和 `module(x)` 调用的区别？→ hooks、guard 等封装不会执行，所以应该用 `module(x)`。
- 为什么权重共享 `word_token_embedding.weight = lm_head.weight` 后参数还是一个？

---

## 点 3 · Autograd 计算图

**知识骨架**

- 动态计算图：前向每次「现搭」一张图，节点 = 张量，边 = 运算（记录在 `.grad_fn`）。
- 叶子张量（leaf）：
  - 对参数 `requires_grad=True`、由用户直接构造的才算叶；
  - 只有叶子能直接对 `.grad` 赋值；
  - 非叶张量（运算产生）梯度不常驻，只在 backward 期间算出来供链式法则传递，平时为 `None`。
- `.grad_fn`：张量由什么运算得来，是图的「指针」。`loss.grad_fn` 非空 = 图还在。
- `backward()`：从 loss 倒走，链式法则把偏导逐层传回叶子，填进 `param.grad`。**默认累加不清零**，中间结果用完即弃。
- 开关三兄弟（高频陷阱）：
  - `requires_grad`：是否记录与求导 —— 冻结/解冻参数（微调）；
  - `@torch.no_grad()`：整个上下文不建图、省显存 —— 评估、推理、生成；
  - `model.eval()`：关闭 dropout 等随机层 —— 评估、推理（常与 no_grad 同用，但机制不同！）。
- 附加能力：`retain_graph=True`（backward 后保图再用）、`create_graph=True`（算梯度之梯度）、`detach()`（硬切图，得到不再跟踪的新张量）。

**常见追问**

- no_grad 改变了 forward 里的数学运算吗？→ 没有，只是不搭图、不留中间激活、不能 backward。
- 为什么验证损失那一段 loss 不能 `backward()`？

---

## 点 4 · backward → optimizer 完整链路

**知识骨架**

- 一场标准更新的四步曲（面试要能默写）：
  1. `loss.backward()` —— 顺着图把梯度填进每个 `param.grad`（累加）；
  2. `optimizer.step()` —— 拿 `.grad` 更新参数（拧旋钮）；
  3. `optimizer.zero_grad()` —— 清空 `.grad` 防下一轮叠加（`set_to_none=True` 更省内存，不保留旧值）；
  4. （回到 1。）
- 中间硬保险：
  - `clip_grad_norm_`：梯度范数超限就等比缩放，防爆炸；
  - 梯度累积：`loss/G` 后 `backward()` 攒 G 次，`step()` 只走一次 —— 显存不足时模拟大 batch；**除以 G** 让累积梯度尺度 = 一个真实大 batch 的尺度（面试必问）。
- `loss` 必须 0 维标量（或已缩放）才能 `backward()`；非标量要带 `gradient=` 向量。

**常见追问**

- 梯度为什么要清零？→ backward 累加。
- 为什么除以梯度累积步数？→ 尺度还原。
- GradScaler.scale(loss).backward() 与 unscale_ 的顺序。

---

## 点 5 · 参数分组与混合精度

**知识骨架**

- 优化器不拿张量，拿**参数分组**：`[{'params': 衰减组, 'weight_decay': wd}, {'params': 不衰减组, 'weight_decay': 0.0}]`。
- 项目做法：二维权重/嵌入做 weight decay，一维 bias/LayerNorm 不做。
- weight decay = 软压参数值防过拟合；对 bias/LN 反而有害。
- AdamW = Adam + 解耦权重衰减；Adam 是自适应学习率，按动量缩放每参数步长。
- 混合精度：
  - `autocast`：算子级自动 f16/bf16 降精度，加速省显存；
  - `GradScaler`：fp16 梯度过小时 `scale(loss)` 放大再 backward、`unscale_` 还原再 `step`，防小梯度下溢成 0。

**常见追问**

- weight decay vs L2 数学差异；（面试手写公式点）
- 为什么 2D 权重衰减、1D 不衰减；
- GradScaler 的 scale 会不会改变极值点？→ 不改，只是等比放大。

---

## 点 6 · 面试题准备清单

> 三档：A 敲门砖（本轮已讲过，滚熟）；B 立刻补（interview.md 与本题绑定）；C 拉开差距。

### A 档（必须先熟）

1. `nn.Module` 为什么能 `model(x)`？答：`__call__` 转发 `forward`。
2. 哪些参数会进优化器？答：Parameter 身份 → 递归 `named_parameters()` → `requires_grad` 过滤 → 分组。
3. `requires_grad`、`no_grad`、`eval()` 各自管啥？能否混说？（陷阱题，两轴：记录梯度 / 随机层开关）
4. 解释 `backward()` / `step()` / `zero_grad()` 各自微积分上的作用。
5. 梯度为什么默认累加？`set_to_none=True` 好在哪？（interview.md:684）

### B 档（立刻补）

- `configure_optimizers` 为什么分「衰减/不衰减」两组？（interview.md:680）
- weight decay vs L2 数学差异。
- 梯度累积为什么 loss 除以累积步数？（interview.md:682）
- fp16 混合精度：autocast + GradScaler 的 scale/unscale 各解决什么？（interview.md:683）
- 叶子 vs 非叶、`.grad` 何时为 `None`、`retain_graph` 什么时候用。
- 手写一个最小可自动求导的 `class Linear`（参照点 3/4）。

### C 档（拉开差距）

- DDP 为什么只在最后一个 micro-step 同步梯度？（interview.md:685）
- `torch.compile` 做了什么：dynamo 抓图 → 图优化融合 → 内核化。（interview.md:695）
- AdamW vs Adam 实现差异，能贴公式。
- `no_grad` vs `inference_mode` vs `enable_grad` 的边界。
- 手写 `softmax + cross_entropy` 的反向推导（链式法则实弹）。

---

## 其它待办/提醒

- 讲每个点时，控制讲解量，讲完即停，等待用户提问。
- 用户提问方向常发散到模型结构（Block/attention 等），涉及时可参考 model.py 与 interview.md。
- 若用户要求「下一题练习」，复用 A 档当口试题目即可。