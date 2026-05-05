# 深度学习训练框架

基于 PyTorch Lightning 的图像分类训练框架。

## 快速开始

```bash
pip install -r requirements-core.txt

# 默认模拟数据训练
python train.py

# CIFAR-10 真实数据
python train.py --data_type cifar10

# 推理
python predict.py
```

## 配置

编辑 `config.py` 顶部变量即可，无需修改其他文件。CLI 参数会自动覆盖默认值：

```bash
python train.py --learning_rate 0.001 --epochs 200 --exp_name my_experiment
python train.py --use_ema --no-save_wandb
python train.py --resume ./experiments/.../checkpoints/epoch_10-loss_0.123.ckpt
```

### 数据管道

```python
data_type = "mock"      # mock / cifar10 / image_folder
data_path = ""          # image_folder 时的数据根目录
image_size = 384
train_batch_size = 32
```

### 模型

```python
model = {
    "model_name": "Simple2DNetwork",
    "model_kwargs": {"in_channels": 3, "num_classes": 3},
}
```

模型自动从 `tools/model/` 目录注册。`forward()` 返回 dict（如 `{'classes': logits}`）。

### 损失函数

```python
loss = {
    "loss_type": "cross_entropy",  # cross_entropy / focal
    "loss_kwargs": {},             # focal: {"alpha": 0.25, "gamma": 2}
    "pred_key": "classes",         # 模型输出中用于计算损失的 key
    "target_key": "label",         # batch 中标签的 key
}
```

`pred_key` / `target_key` 指定损失计算匹配的字段。

## 项目结构

```
├── config.py                  # 唯一配置文件
├── train.py                   # 训练入口
├── predict.py                 # 推理入口
├── pl_tool.py                 # LightningModule + EMA
├── tools/
│   ├── data/
│   │   ├── dataset.py         # MockDataset + _DATASETS 路由
│   │   ├── cifar10.py         # CIFAR10Dataset
│   │   ├── image_folder.py    # ImageFolderDataset
│   │   └── augment.py         # albumentations 变换
│   ├── model/
│   │   ├── registry.py        # ModelRegistry 自动发现
│   │   └── simple2d_network.py # Simple2DNetwork
│   └── loss/
│       ├── __init__.py        # LOSS_REGISTRY + get_loss
│       └── focal_loss.py      # FocalLoss
├── requirements-core.txt
└── requirements.txt
```

## 扩展指南

### 添加新模型

在 `tools/model/` 下创建 `.py` 文件，继承 `nn.Module`，`forward` 返回 dict：

```python
class MyModel(nn.Module):
    def forward(self, x):
        ...
        return {"classes": logits}
```

自动注册，配置中指定 `model_name: MyModel`。

### 添加新数据集

在 `tools/data/` 下创建 `.py` 文件实现 `Dataset` 类，在 `dataset.py` 的 `_DATASETS` 字典中注册。

### 添加新损失函数

在 `tools/loss/` 下创建 `.py` 文件实现损失类，在 `__init__.py` 的 `LOSS_REGISTRY` 字典中注册。
