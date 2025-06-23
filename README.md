# 深度学习训练框架

本框架基于 PyTorch Lightning，专为图像分类任务设计，提供灵活的配置和实验管理功能。

## 主要特性

- **结构清晰**：基于 PyTorch Lightning 构建。
- **实验跟踪**：支持 Weights & Biases (wandb)。
- **灵活配置**：支持 YAML 配置文件和命令行参数。
- **断点续训**：支持模型断点保存与恢复。
- **可视化**：内置数据可视化工具。
- **混合精度训练**：支持混合精度训练加速。
- **自动记录**：自动记录训练指标。

## 配置详解

主要配置参数位于 `configs/config.yaml`：

### 实验环境

- `seed`: 随机种子，用于实验的可重复性。
- `exp_name`: 实验名称，用于区分不同实验。
- `project`: wandb 项目名称，用于实验跟踪。

### 数据集

- `data_path`: 数据集路径。
- `image_size`: 输入图像大小。
- `num_classes`: 分类类别数。

### 模型

- `model_name`: 模型架构名称。
- `pretrained`: 是否使用预训练权重。

### 训练

- `learning_rate`: 学习率。
- `batch_size`: 批次大小。
- `epochs`: 训练轮数。
- `precision`: 训练精度模式。

## 使用指南

1. **安装依赖**：

   ```bash
   pip install -r requirements.txt
   ```

2. **启动训练**：
   ```bash
   python train.py
   ```

## 项目结构

```
.
├── configs/    # 配置文件
├── tools/      # 工具函数
│   ├── datasets/ # 数据集相关
│   ├── models/   # 模型定义
│   ├── losses/   # 损失函数定义
│   └── pl_tool.py# Lightning 模块
├── train.py    # 训练脚本
└── test.py     # 测试脚本
```
