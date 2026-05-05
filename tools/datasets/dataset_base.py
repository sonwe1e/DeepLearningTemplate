"""
数据集基类 — 支持模拟数据、CIFAR-10 和 ImageFolder 三种数据源
"""
import os
import numpy as np
import torch
from .augments import train_transform, valid_transform


class MockDataset(torch.utils.data.Dataset):
    """模拟数据集 — 使用随机数据，无需下载，始终可用"""

    def __init__(self, phase, opt, train_transform=None, valid_transform=None):
        self.phase = phase
        self.data_path = opt.data_path
        self.transform = train_transform if phase == "train" else valid_transform
        self.image_list = [0] * 100

    def __getitem__(self, index):
        image = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)
        label = 1
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return {"image": image, "label": label}

    def __len__(self):
        return len(self.image_list)


class CIFAR10Dataset(torch.utils.data.Dataset):
    """CIFAR-10 数据集 — 首次使用时自动下载"""

    def __init__(self, phase, opt, train_transform=None, valid_transform=None):
        self.phase = phase
        self.transform = train_transform if phase == "train" else valid_transform

        from torchvision.datasets import CIFAR10

        self.cifar = CIFAR10(
            root=opt.data_path or "./data",
            train=(phase == "train"),
            download=True,
        )

    def __getitem__(self, index):
        image, label = self.cifar[index]
        image = np.array(image)
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return {"image": image, "label": label}

    def __len__(self):
        return len(self.cifar)


class ImageFolderDataset(torch.utils.data.Dataset):
    """ImageFolder 风格数据集 — 从目录结构加载，无数据时自动回退模拟数据

    期望目录结构:
        data_path/
            train/
                class_0/
                    img001.jpg
                class_1/
                    ...
            valid/
                class_0/
                    ...
    """

    def __init__(self, phase, opt, train_transform=None, valid_transform=None):
        self.phase = phase
        self.transform = train_transform if phase == "train" else valid_transform
        self.data_path = opt.data_path

        sub_dir = "train" if phase == "train" else "valid"
        data_dir = os.path.join(self.data_path, sub_dir)

        self._samples = []
        self._use_mock = False
        self._num_classes = 0

        if os.path.isdir(data_dir):
            classes = sorted(os.listdir(data_dir))
            self._num_classes = len(classes)
            for cls_idx, cls_name in enumerate(classes):
                cls_dir = os.path.join(data_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        self._samples.append((os.path.join(cls_dir, fname), cls_idx))

        if not self._samples:
            print(f"[警告] 在 {data_dir} 中未找到图像，回退到模拟数据")
            self._samples = [("__mock__", i % 3) for i in range(100)]
            self._use_mock = True
            self._num_classes = 3

    def __getitem__(self, index):
        path, label = self._samples[index]
        if self._use_mock:
            image = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)
        else:
            from PIL import Image as PILImage
            image = np.array(PILImage.open(path).convert("RGB"))
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return {"image": image, "label": label}

    def __len__(self):
        return len(self._samples)

    @property
    def num_classes(self):
        return self._num_classes


DATASET_REGISTRY = {
    "mock": MockDataset,
    "cifar10": CIFAR10Dataset,
    "image_folder": ImageFolderDataset,
}


def get_dataset(phase, opt, train_transform, valid_transform):
    data_type = getattr(opt, "data_type", "mock")
    cls = DATASET_REGISTRY.get(data_type)
    if cls is None:
        raise ValueError(f"未知数据集类型: {data_type}，可选: {list(DATASET_REGISTRY.keys())}")
    return cls(phase, opt, train_transform, valid_transform)
