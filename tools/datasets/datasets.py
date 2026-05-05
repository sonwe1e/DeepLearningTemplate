"""
数据集模块 — 深度学习训练数据管理

通过 data_type 配置键切换数据源:
- "mock": 模拟随机数据（默认，无需下载）
- "cifar10": CIFAR-10 自动下载
- "image_folder": 从 data_path/train 和 data_path/valid 目录加载

通过 use_prefetch 配置键切换预取加速（默认关闭）。
"""

import torch
from configs.option import get_option
from .augments import train_transform, valid_transform
from .dataset_base import get_dataset


def get_dataloader(opt):
    """创建训练和验证数据加载器

    Args:
        opt: 配置对象

    Returns:
        tuple: (train_dataloader, valid_dataloader)
    """
    train_dataset = get_dataset("train", opt, train_transform, valid_transform)
    valid_dataset = get_dataset("valid", opt, train_transform, valid_transform)

    use_prefetch = getattr(opt, "use_prefetch", False)

    if use_prefetch:
        from .datasetsv2 import (
            PrefetchDataLoader,
            CUDAPrefetcher,
            CPUPrefetcher,
            PrefetcherIterator,
        )

        train_dataloader = PrefetchDataLoader(
            num_prefetch_queue=getattr(opt, "prefetch_queue_size", 16),
            dataset=train_dataset,
            batch_size=opt.train_batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_dataloader = PrefetchDataLoader(
            num_prefetch_queue=getattr(opt, "prefetch_queue_size", 16),
            dataset=valid_dataset,
            batch_size=opt.valid_batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
        )

        if len(opt.devices) > 0 and torch.cuda.is_available():
            train_prefetcher = CUDAPrefetcher(train_dataloader, opt)
            valid_prefetcher = CUDAPrefetcher(valid_dataloader, opt)
        else:
            train_prefetcher = CPUPrefetcher(train_dataloader)
            valid_prefetcher = CPUPrefetcher(valid_dataloader)

        train_iterator = PrefetcherIterator(
            train_prefetcher,
            length=len(train_dataset) // opt.train_batch_size,
        )
        valid_iterator = PrefetcherIterator(
            valid_prefetcher,
            length=len(valid_dataset) // opt.valid_batch_size,
        )
        return train_iterator, valid_iterator

    # 标准 DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.train_batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.valid_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    opt = get_option()
    train_dataloader, valid_dataloader = get_dataloader(opt)

    print("=" * 50)
    print("数据加载器测试")
    print("=" * 50)
    print(f"训练数据集大小: {len(train_dataloader.dataset)}")
    print(f"验证数据集大小: {len(valid_dataloader.dataset)}")

    for i, batch in enumerate(train_dataloader):
        print(f"\n第 {i + 1} 个批次:")
        print(f"图像形状: {batch['image'].shape}")
        print(f"标签信息: {torch.unique(batch['label'])}")
        break

    print("\n数据加载器测试完成！")
