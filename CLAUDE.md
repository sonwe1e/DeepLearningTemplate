# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

基于 PyTorch Lightning 的图像分类训练框架。单文件配置 `config.py`，自动模型注册，EMA，混合精度，wandb。

## Project Structure

```
├── config.py              # 唯一配置文件 — 修改此处默认值
├── pl_tool.py             # LightningModule + EMA（核心训练模块）
├── train.py               # 训练入口
├── predict.py             # 推理入口（含 checkpoint 加载）
├── tools/
│   ├── data/
│   │   ├── dataset.py     #   MockDataset + _DATASETS 路由 + get_dataloader
│   │   ├── cifar10.py     #   CIFAR10Dataset
│   │   ├── image_folder.py #  ImageFolderDataset
│   │   └── augment.py     #   build_transforms
│   ├── model/
│   │   ├── registry.py    #   ModelRegistry — 自动扫描本目录注册 nn.Module
│   │   └── simple2d_network.py  # Simple2DNetwork
│   └── loss/
│       ├── __init__.py    #   LOSS_REGISTRY + get_loss
│       └── focal_loss.py  #   FocalLoss
├── requirements-core.txt
└── requirements.txt
```

## Commands

```bash
pip install -r requirements-core.txt
python train.py                           # 默认模拟数据
python train.py --data_type cifar10       # CIFAR-10 真实数据
python train.py --use_ema --no-save_wandb # 布尔值用 --flag / --no-flag
python train.py --resume ./experiments/.../checkpoints/epoch_10-loss_0.123.ckpt
python predict.py
CKPT_PATH=./path/to/checkpoint.ckpt python predict.py
```

## Architecture

### 配置 (`config.py`)

不再是 YAML 文件。模块级变量作为默认值，`get_option()` 自动生成 argparse 并合并 CLI 覆盖。修改超参数只需编辑 `config.py` 顶部变量。dict/list 类型跳过 CLI 生成。

关键新增：`loss.pred_key` 和 `loss.target_key` 定义损失计算时匹配的字段。模型 `forward` 返回 `{'classes': logits}`，batch 含 `{'label': int}`，loss 配置 `pred_key: classes, target_key: label` 将它们连接起来。

### 训练入口 (`train.py`)

`import config` → `config.get_option()` → `seed_everything` → `get_model` → `get_dataloader` → `WandbLogger` → `pl.Trainer` → `trainer.fit`。

### 核心训练模块 (`pl_tool.py`)

`EMA` + `LightningModule`。损失计算：
```python
pred = self.forward(batch["image"])       # → {'classes': tensor}
loss = self.loss1(pred[self.pred_key], batch[self.target_key])
```

`pred_key`/`target_key` 从 `opt.loss` 读取，缺省 `"classes"` / `"label"`。

### 数据管道 (`tools/data/`)

每种数据集独立文件，`dataset.py` 的 `_DATASETS` 字典路由。新增数据集：创建 `xxx.py` 实现 `Dataset` 类，在 `_DATASETS` 中注册。`augment.py` 的 `build_transforms(opt)` 是纯函数。

### 模型管理 (`tools/model/`)

`ModelRegistry` 扫描 `Path(__file__).parent`（即 `tools/model/`）下的 `.py` 文件自动注册 `nn.Module` 子类。`forward()` 返回 dict（如 `{'classes': logits}`）。新增模型：在 `tools/model/` 下创建文件或直接在现有文件中添加类。

### 损失函数 (`tools/loss/`)

`__init__.py` 维护 `LOSS_REGISTRY` + `get_loss()`。每个损失类独立文件（如 `focal_loss.py`）。新增损失：创建文件实现类，在 `LOSS_REGISTRY` 中注册。

### 推理 (`predict.py`)

`evaluate()` 处理模型 dict 输出：`outputs["classes"] if isinstance(outputs, dict) else outputs`。
