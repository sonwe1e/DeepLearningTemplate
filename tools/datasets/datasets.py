"""
数据集模块 - 深度学习训练数据管理

这个文件负责处理训练和验证数据的加载、预处理和批量管理，主要功能包括：

1. 数据集类定义 (Dataset):
   - 负责单个样本的加载和预处理
   - 支持训练和验证两种模式
   - 集成数据增强流水线
   - 支持并行数据加载优化

2. 数据加载器创建 (get_dataloader):
   - 创建训练和验证数据加载器
   - 配置批量大小、洗牌、多进程等参数
   - 优化GPU内存使用

当前实现：
- 使用随机生成的模拟数据进行测试
- 提供了完整的数据加载框架
- 可以通过修改 __getitem__ 方法适配真实数据

如何适配真实数据：
1. 修改 __init__ 方法：更新 image_list 为真实的文件路径列表
2. 修改 load_image 方法：实现图像文件的实际加载逻辑
3. 修改 __getitem__ 方法：加载真实图像和标签
4. 调整数据增强：根据任务需求修改 augments.py

支持的任务类型：
- 图像分类
- 语义分割（需要修改标签加载）
- 其他计算机视觉任务
"""

import os
import glob
import numpy as np
import tifffile
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import Dataset
from configs.option import get_option

# from .augments import train_transform, valid_transform
train_transform = None
valid_transform = None


class Dataset(Dataset):
    """
    自定义数据集类，处理SAR、OPT和Label融合数据集

    支持训练集、验证集和测试集的加载，其中：
    - 训练集和验证集从train目录划分，提供SAR、OPT和Label
    - 测试集从val目录加载，仅提供SAR和OPT
    - 使用并行加载优化初始化性能

    Args:
        phase (str): 数据集阶段，'train'、'valid' 或 'test'
        opt: 配置对象，包含数据路径等参数
        train_transform: 训练数据增强
        valid_transform: 验证/测试数据增强
        train_ratio (float): 训练集划分比例，默认为0.8
    """

    def __init__(
        self, phase, opt, train_transform=None, valid_transform=None, train_ratio=0.8
    ):
        # 基础配置
        self.phase = phase
        self.data_path = opt.data_path
        self.transform = train_transform if phase == "train" else valid_transform

        # 初始化数据列表
        self.sar_paths = []
        self.opt_paths = []
        self.label_paths = []  # 仅用于train和valid
        self.sar_mean = np.array([130.226])
        self.sar_std = np.array([58.866])
        self.opt_mean = np.array([68.901, 66.986, 59.681])
        self.opt_std = np.array([58.509, 54.889, 53.997])

        # 获取train目录下所有文件名（以Label为基准，确保一致性）
        if phase in ["train", "valid"]:
            label_dir = os.path.join(self.data_path, "train", "0_Label")
            label_files = sorted(glob.glob(os.path.join(label_dir, "*.tif")))
            base_names = [
                os.path.basename(f).replace("_Train.tif", "") for f in label_files
            ]

            # 构造对应的SAR和OPT路径
            sar_dir = os.path.join(self.data_path, "train", "1_SAR")
            opt_dir = os.path.join(self.data_path, "train", "2_Opt")
            self.sar_paths = [
                os.path.join(sar_dir, f"{name}_Train.tif") for name in base_names
            ]
            self.opt_paths = [
                os.path.join(opt_dir, f"{name}_Train.tif") for name in base_names
            ]
            self.label_paths = label_files

            # 划分训练集和验证集
            indices = list(range(len(self.sar_paths)))
            train_idx, valid_idx = train_test_split(
                indices, train_size=train_ratio, random_state=42
            )
            if phase == "train":
                self.sar_paths = [self.sar_paths[i] for i in train_idx]
                self.opt_paths = [self.opt_paths[i] for i in train_idx]
                self.label_paths = [self.label_paths[i] for i in train_idx]
            else:  # valid
                self.sar_paths = [self.sar_paths[i] for i in valid_idx]
                self.opt_paths = [self.opt_paths[i] for i in valid_idx]
                self.label_paths = [self.label_paths[i] for i in valid_idx]

        elif phase == "test":
            # 测试集从val目录加载，仅SAR和OPT
            sar_dir = os.path.join(self.data_path, "val", "1_SAR")
            sar_files = sorted(glob.glob(os.path.join(sar_dir, "*.tif")))
            base_names = [
                os.path.basename(f).replace("_Train.tif", "") for f in sar_files
            ]
            opt_dir = os.path.join(self.data_path, "val", "2_Opt")
            self.sar_paths = sar_files
            self.opt_paths = [
                os.path.join(opt_dir, f"{name}_Train.tif") for name in base_names
            ]

        # 并行预加载图像（可选，视内存情况）
        self.preloaded_images = None
        if opt.preload_images:
            self.load_images_in_parallel()

    def __getitem__(self, index):
        """
        获取单个数据样本

        Returns:
            dict: 训练/验证返回 {"sar": tensor, "opt": tensor, "label": tensor}
                  测试返回 {"sar": tensor, "opt": tensor}
        """
        # 加载图像
        if self.preloaded_images is not None:
            sar = self.preloaded_images["sar"][index]
            opt = self.preloaded_images["opt"][index]
            label = (
                self.preloaded_images["label"][index]
                if self.phase in ["train", "valid"]
                else None
            )
        else:
            sar = self.load_image(self.sar_paths[index])
            opt = self.load_image(self.opt_paths[index])
            label = (
                self.load_image(self.label_paths[index])
                if self.phase in ["train", "valid"]
                else None
            )

        # 应用数据增强
        if self.transform is not None:
            if self.phase in ["train", "valid"]:
                augmented = self.transform(image=sar, image1=opt, mask=label)
                sar, opt, label = (
                    augmented["image"],
                    augmented["image1"],
                    augmented["mask"],
                )
            else:  # test
                augmented = self.transform(image=sar, image1=opt)
                sar, opt = augmented["image"], augmented["image1"]

        # 转换为张量
        sar = (sar - self.sar_mean) / self.sar_std  # 标准化SAR图像
        opt = (opt - self.opt_mean) / self.opt_std  # 标准化OPT图像
        sar = torch.from_numpy(sar).unsqueeze(0).float()
        opt = torch.from_numpy(opt).permute(2, 0, 1).float()
        result = {"sar": sar, "opt": opt}
        if self.phase in ["train", "valid"]:
            label = torch.from_numpy(label).long()  # 假设Label是分割掩码
            result["label"] = label

        return result

    def __len__(self):
        """返回数据集大小"""
        return len(self.sar_paths)

    def load_image(self, path):
        """
        加载单个图像文件

        Args:
            path (str): 图像文件路径

        Returns:
            numpy.ndarray: 加载的图像数组 (H, W, C)
        """
        # 使用OpenCV加载.tif图像
        image = tifffile.imread(path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {path}")
        return image

    def load_images_in_parallel(self):
        """
        并行预加载图像以优化初始化性能
        """

        def load_single_image(path):
            return self.load_image(path)

        with ThreadPoolExecutor(max_workers=24) as executor:
            sar_images = list(executor.map(load_single_image, self.sar_paths))
            opt_images = list(executor.map(load_single_image, self.opt_paths))
            label_images = (
                list(executor.map(load_single_image, self.label_paths))
                if self.phase in ["train", "valid"]
                else []
            )

        self.preloaded_images = {
            "sar": sar_images,
            "opt": opt_images,
            "label": label_images,
        }


def get_dataloader(opt):
    """
    创建训练和验证数据加载器

    这个函数是数据流水线的入口点，负责：
    1. 创建训练和验证数据集实例
    2. 配置数据加载器的各项参数
    3. 返回可用于训练的数据加载器

    Args:
        opt: 配置对象，包含以下关键参数：
            - train_batch_size: 训练批量大小
            - valid_batch_size: 验证批量大小
            - num_workers: 数据加载的进程数
            - data_path: 数据根目录路径

    Returns:
        tuple: (train_dataloader, valid_dataloader)
            - train_dataloader: 训练数据加载器
            - valid_dataloader: 验证数据加载器

    参数调优建议：

    1. batch_size 选择：
       - 根据GPU显存调整，如果使用BN建议使用比较大的批量
       - 训练批量可以大一些，验证批量可以更大（因为不需要梯度）

    2. num_workers 选择：
       - 通常设置为GPU训练数量的4-8倍
       - 过多会导致内存消耗过大
       - 可以通过实验找到最优值

    3. pin_memory：
       - GPU训练时建议设为True，可以加速数据传输

    4. drop_last：
       - 训练时建议为True，确保每个批次大小一致
       - 验证时通常为False，避免丢失数据
    """
    # ==================== 数据集创建 ====================
    # 创建训练数据集
    train_dataset = Dataset(
        phase="train",  # 训练阶段
        opt=opt,  # 配置参数
        train_transform=train_transform,  # 训练数据增强
        valid_transform=valid_transform,  # 验证数据增强
    )

    # 创建验证数据集
    valid_dataset = Dataset(
        phase="valid",  # 验证阶段
        opt=opt,  # 配置参数
        train_transform=train_transform,  # 训练数据增强
        valid_transform=valid_transform,  # 验证数据增强
    )

    # ==================== 训练数据加载器配置 ====================
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,  # 训练数据集
        batch_size=opt.train_batch_size,  # 批量大小
        shuffle=True,  # 随机打乱，增加训练随机性
        num_workers=opt.num_workers,  # 多进程加载数据
        pin_memory=True,  # 固定内存，加速GPU传输
        drop_last=True,  # 丢弃最后不完整的批次
    )

    # ==================== 验证数据加载器配置 ====================
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,  # 验证数据集
        batch_size=opt.valid_batch_size,  # 验证批量大小（通常可以更大）
        shuffle=False,  # 验证时不需要打乱
        num_workers=opt.num_workers,  # 多进程加载数据
        pin_memory=True,  # 固定内存，加速GPU传输
        # drop_last=False (默认)             # 验证时保留所有数据
    )

    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    """
    数据集测试代码
    
    这个部分用于测试数据加载器是否正常工作，
    可以单独运行这个文件来验证数据流水线。
    
    测试内容：
    1. 创建数据加载器
    2. 加载第一个批次
    3. 打印图像尺寸和标签信息
    4. 验证数据格式是否正确
    
    运行方式：
    python -m tools.datasets.datasets
    
    或者直接：
    python tools/datasets/datasets.py
    """
    # 获取配置参数
    opt = get_option()

    # 创建数据加载器
    train_dataloader, valid_dataloader = get_dataloader(opt)

    # 测试训练数据加载器
    print("=" * 50)
    print("数据加载器测试")
    print("=" * 50)
    print(f"训练数据集大小: {len(train_dataloader.dataset)}")
    print(f"验证数据集大小: {len(valid_dataloader.dataset)}")
    print(f"训练批次数量: {len(train_dataloader)}")
    print(f"验证批次数量: {len(valid_dataloader)}")

    # 测试第一个训练批次
    for i, batch in enumerate(train_dataloader):
        print(f"\n第 {i + 1} 个批次:")
        print(
            f"图像形状: {batch['image'].shape}"
        )  # 期望: [batch_size, channels, height, width]
        print(f"标签信息: {torch.unique(batch['label'])}")  # 打印唯一标签值
        print(f"图像数据类型: {batch['image'].dtype}")
        print(f"图像数值范围: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        break  # 只测试第一个批次

    print("\n数据加载器测试完成！")
