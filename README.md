# 深度学习训练框架

本框架基于 PyTorch Lightning，专为图像分类任务设计，提供灵活的配置和实验管理功能。
## 最新更改
- **[Jun 2025]** 添加推理示例代码和 ONNX 示例代码（推理代码支持单图预测和 val_dataloader 预测，具体的指标需要自己重新定义；ONNX 的导出和测试代码来源于 Pytorch 官方 tutorial）
- **[Jun 2025]** 在 train.py、pl_tool.py 和 datasets.py 添加注释和引导
- **[Jun 2025]** 添加模型注册机制（在 tools/models/test_model1.py 文件中定义自己的模型后，可以直接在 config.yaml 中导入并在 model_kwarg 添加模型的参数）

## 主要特性

- **结构清晰**：基于 PyTorch Lightning 构建，代码结构清晰易维护
- **实验跟踪**：支持 Weights & Biases (wandb) 实验记录和可视化
- **灵活配置**：支持 YAML 配置文件和命令行参数覆盖
- **断点续训**：支持模型断点保存与恢复，训练中断不丢失进度
- **自动实验管理**：自动创建带时间戳的实验目录，避免实验结果覆盖
- **混合精度训练**：支持 bf16/fp16 混合精度训练加速
- **EMA 优化**：内置指数移动平均(EMA)提升模型性能
- **多 GPU 训练**：支持单机多卡和分布式训练
- **支持ONNX**：添加了 ONNX 的导出和测试代码
- **模型自动注册与参数灵活传递**：自动发现 `tools/models/` 目录下的所有模型类，支持通过配置灵活传递模型参数

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

将数据集放置在指定路径，需要自行调整 dataset 的能够准确提取到需要的数据。

### 3. 配置参数

编辑 `configs/config.yaml` 文件或使用命令行参数：

```bash
# 使用默认配置训练
python train.py

# 使用命令行参数覆盖配置
python train.py --learning_rate 0.001 --epochs 200 --exp_name my_experiment
```

## 配置详解

主要配置参数位于 `configs/config.yaml`：

### 实验管理与日志

```yaml
project: Test # wandb项目名称
exp_name: baselinev1 # 实验名称，用于区分不同实验
seed: 42 # 随机种子，确保实验可重复
save_wandb: true # 是否保存到wandb
log_step: 50 # 每多少步记录一次日志
```

### 硬件与性能配置

```yaml
devices: 5 # 使用的GPU设备ID列表
num_workers: 8 # 数据加载器工作进程数
precision: bf16-mixed # 训练精度 (fp32/fp16/bf16-mixed)
prefetch_queue_size: 16 # 数据预取队列大小（如果使用 datasetv2）
```

### 数据管道

```yaml
data_path: "" # 数据集根路径
train_batch_size: 32 # 训练批次大小
valid_batch_size: 32 # 验证批次大小
image_size: 384 # 输入图像尺寸
```

### 模型定义与注册机制

```yaml
model:
  model_name: test_network # 模型名称，自动从 tools/models/ 目录下注册
  model_kwargs:
    in_channels: 3
    num_classes: 3
```

- **模型自动注册机制**：框架会自动扫描 `tools/models/` 目录下所有以 `nn.Module` 为基类的模型类，并注册到模型仓库，无需手动导入。
- **灵活参数传递**：通过 `model_kwargs` 可为模型构造函数传递任意参数，支持自定义模型结构和超参数。

### 训练策略

```yaml
epochs: 100 # 训练轮数
val_check: 1.0 # 验证频率 (1.0表示每个epoch验证一次)
accumulate_grad_batches: 1 # 梯度累积步数，默认不进行梯度累积
use_ema: true # 是否使用指数移动平均
ema_decay: 0.999 # EMA衰减率
```

### 优化器与学习率调度

```yaml
learning_rate: 0.0004 # 初始学习率
weight_decay: 0.05 # 权重衰减
gradient_clip_val: 1000000.0 # 梯度裁剪阈值（默认不进行裁剪）
pct_start: 0.1 # OneCycleLR调度器参数
```

### 模型保存与恢复

```yaml
resume: null # 恢复训练的checkpoint路径
save_checkpoint_num: 3 # 保存的checkpoint数量
save_metric: valid_loss # 保存checkpoint的监控指标
```

## 命令行参数

所有 YAML 配置参数都可以通过命令行覆盖，支持嵌套参数：

```bash
# 修改学习率和实验名称
python train.py --learning_rate 0.001 --exp_name high_lr_experiment

# 修改模型和批次大小
python train.py --model.model_name test_network --train_batch_size 64

# 禁用EMA和wandb
python train.py --use_ema --save_wandb

# 恢复训练
python train.py --resume /path/to/checkpoint.ckpt
```

## 实验目录结构

框架会自动创建带时间戳的实验目录：

```
experiments/
└── Test/                       # 项目名称
    └── exp_name_2024-01-15_14-30-00/  # 实验名称_时间戳
        ├── save_config.yaml         # 本次实验的有效配置
        └── checkpoints/                   # 模型检查点
            ├── epoch_10-loss_0.123.ckpt
            └── last.ckpt
```

## 项目结构

```
.
├── configs/                   # 配置文件目录
│   ├── config.yaml            # 主配置文件
│   └── option.py              # 配置加载和命令行解析
├── temp_data/                 # 存放临时文件
│   ├── cat.png                # 测试图片
│   ├── test.onnx              # 测试导出 ONNX 文件
├── tools/                     # 工具模块
│   ├── datasets/              # 数据集相关
│   │   ├── datasets.py        # 标准数据加载器
│   │   ├── datasetsv2.py      # 支持预取/多线程的数据加载器
│   │   └── augments.py        # 数据增强与变换
│   ├── models/                # 模型定义（自动注册）
│   │   ├── __init__.py
│   │   ├── test_model1.py
│   │   ├── test_model2.py
│   │   └── ...                # 其他自定义模型
│   ├── losses/                # 损失函数
│   │   └── losses.py
│   ├── example_predict.py     # 单图推理和 dataloader 示例代码
│   ├── example_export.py      # ONNX 导出示例代码
│   ├── utils.py               # 通用工具函数
│   ├── model_registry.py      # 模型注册与发现机制
│   └── pl_tool.py             # Lightning模块和EMA实现
├── experiments                # 保存训练参数以及权重
│   └── exp_name_timestamp     # 实验名称和当前时间
│       ├── save_config.yaml   # 结合 yaml 和命令行输入的配置文件
│       └── checkpoints/       # 保存权重文件夹（权重命名在 train.py 自行调整）
├── wandb/                     # 保存 wandb 数据文件夹
├── train.py                   # 训练脚本
├── test.py                    # 测试脚本
├── requirements.txt           # 依赖列表
└── README.md                  # 项目说明
```

## 高级功能

### EMA (指数移动平均)

框架内置 EMA 功能，在验证时自动使用 EMA 参数：

```python
# EMA会在训练过程中自动更新
# 验证时自动应用EMA参数
# checkpoint中自动保存EMA状态
```

### 多 GPU 训练

```yaml
# 使用多张GPU
devices:
  - 0
  - 1
  - 2
# 若网络中存在未使用的参数需要设置 strategy="ddp_find_unused_parameters_true"
```

### 混合精度训练

```bash
# 使用bf16混合精度
python train.py --precision bf16-mixed

# 使用fp16混合精度
python train.py --precision 16-mixed
```

### 梯度累积

```bash
# 累积4个batch的梯度再更新
python train.py --accumulate_grad_batches 4
```

## 扩展指南

### 添加新的模型（自动注册）

只需在 `tools/models/` 目录下添加新的模型类（继承自 `nn.Module`），框架会自动注册：

```python
# tools/models/my_custom_model.py
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        # ...模型结构...
```

然后在配置文件或命令行中指定：

```yaml
model:
  model_name: MyCustomModel
  model_kwargs:
    in_channels: 3
    num_classes: 10
```

### 添加新的数据集

在 `tools/datasets/datasets.py` 中添加新的数据集类：

```python
class CustomDataset(Dataset):
    def __init__(self, ...):
        # 初始化代码
        pass

    def __getitem__(self, idx):
        # 数据获取逻辑
        return {"image": image, "label": label}
```

### 自定义损失函数

在 `tools/losses/` 目录下添加自定义损失函数。

## 常见问题

### Q: 如何恢复中断的训练？

A: 使用 `--resume` 参数指定 checkpoint 路径：

```bash
python train.py --resume ./checkpoints/epoch_50-loss_0.123.ckpt
```

### Q: 如何禁用 wandb 日志？

A: 设置 `--save_wandb` 参数（注意：配置文件中默认为 true，命令行会切换为 false）：

```bash
python train.py --save_wandb
```

### Q: 内存不足怎么办？

A: 可以减少批次大小或使用梯度累积：

```bash
python train.py --train_batch_size 16 --accumulate_grad_batches 2
```

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目。
