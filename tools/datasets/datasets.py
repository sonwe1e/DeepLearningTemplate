import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from configs.option import get_option
from .augments import train_transform, valid_transform


class Dataset(torch.utils.data.Dataset):
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

    def load_image(self, path):
        return None

    def load_images_in_parallel(self):
        with ThreadPoolExecutor(max_workers=24) as executor:
            pass


def get_dataloader(opt):
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
    opt = get_option()
    train_dataloader, valid_dataloader = get_dataloader(opt)

    for i, batch in enumerate(train_dataloader):
        print(batch["image"].shape, torch.unique(batch["label"]))
        break
