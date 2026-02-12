这是一个非常好的习惯。在科研中，建立一套标准化的“模型切换流程”能极大地减少出错概率。

这份总结将指导你如何从头到尾更换一个新的扩散模型（Diffuser）或目标分类器（Classifier）进行鲁棒性测试。

### 🚀 DiffPure 模型切换与测试标准流程

#### 0. 文件结构概览

首先，保持你的文件结构清晰，建议如下组织：

```text
.
├── configs/
│   ├── cifar10.yml          # 原版配置
│   └── cifar10_adbm.yml     # [新建] 你的模型专用配置
├── pretrained/
│   ├── adbm/                # [新建] 存放你的权重
│   │   └── my_best.pth
│   └── score_sde/           # 原版权重
├── models/
│   └── cifar10/
│       └── Linf/
│           └── Standard.pt  # 目标分类器权重
└── run_scripts/
    └── cifar10/
        └── run_adbm_test.sh # [新建] 你的测试脚本

```

---

### 第一步：准备扩散模型权重 (The Diffuser)

这是你要测试的核心对象（如 ADBM）。

1. **放置文件**：将训练好的 `.pth` 文件放入 `pretrained/adbm/`（或任意你喜欢的目录）。
2. **检查权重格式**：
* 如果权重包含 `optimizer`, `ema`, `model`（完整 Checkpoint）：确保代码能处理。
* 如果权重只有 `state_dict`：确保没有 `module.` 前缀（多卡训练残留）。
* *注：如果你按我之前的建议修改了 `diffpure_sde.py`，代码会自动处理这两种情况。*



---

### 第二步：配置模型参数 (The Config)

告诉代码你的模型长什么样，以及去哪里找它。

1. **复制配置**：`cp configs/cifar10.yml configs/cifar10_my_model.yml`
2. **修改关键字段**：打开新 `.yml` 文件修改 `model` 部分。

```yaml
model:
  # 1. 指向你的权重路径 (这是我们修改代码后支持的新字段)
  ckpt: pretrained/adbm/my_best.pth

  # 2. 必须与训练时的架构完全一致！(否则报错 Size Mismatch)
  nf: 128                 # 通道数
  ch_mult: [1, 2, 2, 2]   # 通道倍率
  num_res_blocks: 8       # 残差块数量
  dropout: 0.1            # Dropout 比率
  # ... 其他参数保持不变

```

---

### 第三步：确定目标分类器 (The Classifier)

决定你要保护哪个分类器（或者是用哪个分类器来计算梯度）。

1. **选择分类器名称**：
在 `run_scripts/*.sh` 中通过 `--classifier_name` 指定。
* `cifar10-wideresnet-28-10` (最常用，标准 RobustBench 模型)
* `cifar10-resnet-50`
* `cifar10-wideresnet-70-16`


2. **确认权重位置**：
* **自动下载类** (如 `wideresnet-28-10`)：
* 确保 `models/cifar10/Linf/Standard.pt` 存在。
* 如果服务器没网，需手动下载并上传。


* **硬编码路径类** (如 `resnet-50`)：
* 查看 `utils.py` 里的路径 (e.g., `pretrained/cifar10/resnet-50/weights.pt`)。
* 必须手动创建目录并将文件放在那里。





---

### 第四步：编写运行脚本 (The Execution)

不要修改原版脚本，始终新建脚本以保留记录。

**新建文件**：`run_scripts/cifar10/run_my_experiment.sh`

```bash
#!/usr/bin/env bash
cd ../..  # 回到根目录

# 核心变量
CONFIG="cifar10_my_model.yml"  # 指向第二步的配置文件
EXP_NAME="exp_adbm_test"       # 输出目录名 (结果会在 exp_results/EXP_NAME 下)
GPU_ID=0

# 运行命令
CUDA_VISIBLE_DEVICES=$GPU_ID python eval_sde_adv.py \
  --config configs/$CONFIG \
  --exp ./exp_results/$EXP_NAME \
  --doc my_test_run \
  --domain cifar10 \
  \
  # --- 关键测试参数 ---
  --diffusion_type sde \         # 模式: sde (随机) 或 ode (确定性)
  --t 75 \                       # 噪声强度/时间步 (重要超参，需调优)
  --adv_eps 0.5 \                # 攻击扰动大小 (L2通常0.5, Linf通常8/255)
  --lp_norm L2 \                 # 攻击范数: L2 或 Linf
  --classifier_name cifar10-wideresnet-28-10 \ # 目标分类器
  \
  # --- 其他设置 ---
  --seed 123 \
  --num_sub 100 \                # 测试图片数量 (调试时设小点，比如 16)
  --adv_batch_size 16            # Batch Size (显存不够就调小)

```

---

### 🚨 常见报错速查表 (Troubleshooting)

| 报错信息 | 可能原因 | 解决方案 |
| --- | --- | --- |
| **KeyError: 'optimizer'** | 权重文件里只有模型参数，但 Runner 试图加载完整状态 | 修改 `runners/diffpure_sde.py` 添加智能加载逻辑（见上文）。 |
| **RuntimeError: Size mismatch** | `.yml` 里的 `nf`, `ch_mult` 等参数和训练时不一致 | 检查训练时的 Config，确保测试 Config 完全对应。 |
| **Connection timed out (drive.google.com)** | 试图下载分类器但服务器无网 | 手动下载分类器 `.pt` 文件并放到 `models/` 或 `pretrained/` 下对应目录。 |
| **Unexpected key(s) in state_dict: "module.xxx"** | 多卡训练保存的权重 | 在加载代码中加入 `.replace('module.', '')` 的逻辑。 |
| **FileNotFoundError** | 路径写错了 | 检查 `.yml` 里的 `ckpt` 路径或 `utils.py` 里的分类器路径。 |

### 💡 科研小贴士

1. **控制变量**：每次只改一个变量。比如先固定 `t=75`，换不同的模型权重测；或者固定模型权重，测 `t=[50, 75, 100]`。
2. **小批量调试**：先设置 `--num_sub 16` (只测16张图) 跑通流程，确认无误后再跑全量 (如 1024 张)。AutoAttack 非常慢，跑全量可能要几小时。
3. **日志检查**：运行后去 `exp_results/你的实验名/日志目录/log.txt` 看详细输出，屏幕输出有时候不全。