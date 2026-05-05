"""数据加载器入口"""
import torch
import numpy as np
from .augment import build_transforms
from .cifar10 import CIFAR10Dataset
from .image_folder import ImageFolderDataset


class MockDataset(torch.utils.data.Dataset):
    """模拟数据集 — 随机数据，始终可用"""

    def __init__(self, phase, opt, train_transform, valid_transform):
        self.transform = train_transform if phase == "train" else valid_transform
        self._len = 100

    def __getitem__(self, index):
        image = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)
        label = 1
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return {"image": image, "label": label}

    def __len__(self):
        return self._len


_DATASETS = {
    "mock": MockDataset,
    "cifar10": CIFAR10Dataset,
    "image_folder": ImageFolderDataset,
}


def get_dataloader(opt):
    train_transform, valid_transform = build_transforms(opt)

    data_type = getattr(opt, "data_type", "mock")
    cls = _DATASETS.get(data_type)
    if cls is None:
        raise ValueError(f"未知数据集类型: {data_type}，可选: {list(_DATASETS.keys())}")

    train_dataset = cls("train", opt, train_transform, valid_transform)
    valid_dataset = cls("valid", opt, train_transform, valid_transform)

    def _worker_init_fn(worker_id):
        seed = int(opt.seed) + worker_id
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))

    g = torch.Generator()
    g.manual_seed(int(opt.seed))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.train_batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=_worker_init_fn, generator=g,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.valid_batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True,
        worker_init_fn=_worker_init_fn, generator=g,
    )
    return train_dataloader, valid_dataloader
