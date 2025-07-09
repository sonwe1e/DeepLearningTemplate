"""数据集模块 - 深度学习训练数据管理"""

import torch
import numpy as np
from configs.option import get_option
from .augments import train_transform, valid_transform


class Dataset(torch.utils.data.Dataset):
    """自定义数据集类"""

    def __init__(self, phase, opt, train_transform=None, valid_transform=None):
        self.phase = phase
        self.data_path = opt.data_path
        self.transform = train_transform if phase == "train" else valid_transform
        self.image_list = [0] * 100  # 模拟100个样本

    def __getitem__(self, index):
        """获取单个数据样本"""
        image = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)
        label = 1

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return {"image": image, "label": label}

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_list)

    def load_image(self, path):
        """加载单个图像文件"""
        return None


def get_dataloader(opt):
    """创建训练和验证数据加载器"""
    train_dataset = Dataset(
        phase="train",
        opt=opt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )

    valid_dataset = Dataset(
        phase="valid",
        opt=opt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )

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
    """数据集测试代码"""
    opt = get_option()
    train_dataloader, valid_dataloader = get_dataloader(opt)

    print("=" * 50)
    print("数据加载器测试")
    print("=" * 50)
    print(f"训练数据集大小: {len(train_dataloader.dataset)}")
    print(f"验证数据集大小: {len(valid_dataloader.dataset)}")
    print(f"训练批次数量: {len(train_dataloader)}")
    print(f"验证批次数量: {len(valid_dataloader)}")

    for i, batch in enumerate(train_dataloader):
        print(f"\n第 {i + 1} 个批次:")
        print(f"图像形状: {batch['image'].shape}")
        print(f"标签信息: {torch.unique(batch['label'])}")
        print(f"图像数据类型: {batch['image'].dtype}")
        print(f"图像数值范围: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        break

    print("\n数据加载器测试完成！")
