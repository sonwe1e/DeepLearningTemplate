import os
import numpy as np
import torch


class ImageFolderDataset(torch.utils.data.Dataset):
    """ImageFolder — data_path/train|valid/class_name/*.jpg，无数据时回退模拟"""

    def __init__(self, phase, opt, train_transform, valid_transform):
        self.transform = train_transform if phase == "train" else valid_transform
        data_dir = os.path.join(opt.data_path, "train" if phase == "train" else "valid")
        self._samples = []
        self._mock = False

        if os.path.isdir(data_dir):
            for cls_idx, cls_name in enumerate(sorted(os.listdir(data_dir))):
                cls_dir = os.path.join(data_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        self._samples.append((os.path.join(cls_dir, fname), cls_idx))

        if not self._samples:
            print(f"[警告] {data_dir} 中未找到图像，回退模拟数据")
            self._samples = [("__mock__", i % 3) for i in range(100)]
            self._mock = True

    def __getitem__(self, index):
        path, label = self._samples[index]
        if self._mock:
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
