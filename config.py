"""实验配置 — 修改下方默认值后运行 python train.py"""
import argparse
import inspect
from pathlib import Path
from datetime import datetime

# ============================================================
# 实验管理与日志
# ============================================================
project: str = "Test"
exp_name: str = "baselinev1"
seed: int = 42
save_wandb: bool = True
log_step: int = 50

# ============================================================
# 硬件与性能
# ============================================================
devices: list = [0]
num_workers: int = 0  # Windows 用 0 避免多进程 pickle 问题，Linux 可调大
precision: str = "bf16-mixed"

# ============================================================
# 数据管道
# ============================================================
data_path: str = ""
data_type: str = "mock"  # mock, cifar10, image_folder
train_batch_size: int = 32
valid_batch_size: int = 32
image_size: int = 384

# ============================================================
# 模型定义
# ============================================================
model: dict = {
    "model_name": "Simple2DNetwork",
    "model_kwargs": {"in_channels": 3, "num_classes": 10},
}

# ============================================================
# 损失函数 — pred_key/target_key 指定匹配的模型输出和标签字段
# ============================================================
loss: dict = {
    "loss_type": "cross_entropy",
    "loss_kwargs": {},
    "pred_key": "classes",
    "target_key": "label",
}

# ============================================================
# 训练策略
# ============================================================
epochs: int = 100
val_check: float = 1.0
accumulate_grad_batches: int = 1
use_ema: bool = False
ema_decay: float = 0.999

# ============================================================
# 优化器与学习率
# ============================================================
learning_rate: float = 0.0004
weight_decay: float = 0.05
gradient_clip_val: float = 12.0
pct_start: float = 0.1

# ============================================================
# 模型保存与恢复
# ============================================================
resume: str = None
save_checkpoint_num: int = 3
save_metric: str = "train_loss"


def get_option(verbose: bool = True):
    """加载默认配置并合并 CLI 参数覆盖"""
    defaults = {
        k: v for k, v in globals().items()
        if not k.startswith("_") and not callable(v) and not isinstance(v, type)
        and not inspect.ismodule(v)
    }
    # 处理特殊的 None 默认值
    for k in list(defaults.keys()):
        if defaults[k] is None:
            defaults[k] = ""

    parser = argparse.ArgumentParser(description="实验配置")
    for key, value in defaults.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=None)
        elif isinstance(value, (dict, list)):
            pass
        else:
            parser.add_argument(f"--{key}", type=type(value), default=None)

    args = parser.parse_args()

    final = dict(defaults)
    for k, v in vars(args).items():
        if v is not None:
            final[k] = v

    # 恢复 resume 为 None
    if final.get("resume") == "":
        final["resume"] = None

    exp_path = None
    if verbose:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = final.get("exp_name", "default-exp")
        exp_path = Path("experiments") / f"{name}_{timestamp}"
        exp_path.mkdir(parents=True, exist_ok=True)
        final["exp_path"] = str(exp_path)

        print("=" * 40)
        print(f"实验输出目录: {exp_path.resolve()}")
        print("-" * 40)
        print("最终生效配置:")
        for k, v in final.items():
            print(f"  {k:<22}: {v}")
        print("=" * 40)

    return argparse.Namespace(**final), exp_path
