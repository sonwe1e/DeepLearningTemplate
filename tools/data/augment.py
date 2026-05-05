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
