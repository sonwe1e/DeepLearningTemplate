import torch.nn as nn
from .focal_loss import FocalLoss

LOSS_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
    "focal": FocalLoss,
}


def get_loss(loss_type: str, **kwargs):
    key = loss_type.lower().replace(" ", "_")
    if key not in LOSS_REGISTRY:
        raise ValueError(f"未知损失函数: {loss_type}，可选: {list(LOSS_REGISTRY.keys())}")
    cls = LOSS_REGISTRY[key]
    try:
        return cls(**kwargs)
    except TypeError:
        return cls()
