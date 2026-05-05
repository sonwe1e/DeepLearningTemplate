import numpy as np
import torch


class CIFAR10Dataset(torch.utils.data.Dataset):
    """CIFAR-10 — 首次使用自动下载"""

    def __init__(self, phase, opt, train_transform, valid_transform):
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
