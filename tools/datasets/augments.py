import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(opt):
    """基于传入的opt构建数据变换"""
    train_transform = A.Compose(
        [
            A.Resize(opt.image_size, opt.image_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    valid_transform = A.Compose(
        [
            A.Resize(opt.image_size, opt.image_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    return train_transform, valid_transform


class _TransformLoader:
    def __init__(self, opt):
        self.opt = opt
        self._train_transform = None
        self._valid_transform = None
        self._cached_size = None

    @property
    def train_transform(self):
        if self._train_transform is None or self._cached_size != self.opt.image_size:
            self._train_transform, self._valid_transform = build_transforms(self.opt)
            self._cached_size = self.opt.image_size
        return self._train_transform

    @property
    def valid_transform(self):
        if self._valid_transform is None or self._cached_size != self.opt.image_size:
            self._train_transform, self._valid_transform = build_transforms(self.opt)
            self._cached_size = self.opt.image_size
        return self._valid_transform
