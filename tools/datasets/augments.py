import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.option import get_option


def get_transforms():
    """获取数据变换，延迟加载配置"""
    opt = get_option(verbose=False)

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
    def __init__(self):
        self._train_transform = None
        self._valid_transform = None

    @property
    def train_transform(self):
        if self._train_transform is None:
            self._train_transform, self._valid_transform = get_transforms()
        return self._train_transform

    @property
    def valid_transform(self):
        if self._valid_transform is None:
            self._train_transform, self._valid_transform = get_transforms()
        return self._valid_transform


_loader = _TransformLoader()
train_transform = _loader.train_transform
valid_transform = _loader.valid_transform
