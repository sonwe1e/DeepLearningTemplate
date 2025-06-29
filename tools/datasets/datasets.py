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

import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from configs.option import get_option
from .augments import train_transform, valid_transform


class Dataset(torch.utils.data.Dataset):
    """
    自定义数据集类

    这个类继承自PyTorch的Dataset，负责管理训练和验证数据。
    主要功能包括数据加载、预处理、标签处理等。

    当前实现使用随机生成的模拟数据，实际使用时需要根据具体任务进行修改。

    Args:
        phase (str): 数据集阶段，'train' 或 'valid'
        opt: 配置对象，包含数据路径、批量大小等参数
        train_transform: 训练时的数据增强变换
        valid_transform: 验证时的数据增强变换

    如何适配真实数据：
    1. 修改 __init__ 中的 image_list，从实际路径加载文件列表
    2. 实现 load_image 方法，加载真实图像文件
    3. 修改 __getitem__ 中的标签加载逻辑
    4. 根据任务调整返回的数据格式
    """

    def __init__(self, phase, opt, train_transform=None, valid_transform=None):
        # ==================== 基础配置 ====================
        self.phase = phase  # 训练或验证阶段
        self.data_path = opt.data_path  # 数据路径，从配置文件读取

        # ==================== 数据增强配置 ====================
        # 根据阶段选择对应的数据变换：训练时使用增强，验证时使用基础变换
        self.transform = train_transform if phase == "train" else valid_transform

        # ==================== 数据列表初始化 ====================
        # TODO: 这里需要根据实际数据修改
        # 当前使用模拟数据，实际应用时应该：
        # 1. 从 self.data_path 读取图像文件列表
        # 2. 根据标注文件加载对应的标签
        # 3. 支持不同的数据组织方式（如 ImageNet 风格的目录结构）
        self.image_list = [0] * 100  # 模拟100个样本

        # 示例：实际数据加载代码
        # import os
        # import glob
        # if phase == "train":
        #     self.image_list = glob.glob(os.path.join(self.data_path, "train", "*.jpg"))
        # else:
        #     self.image_list = glob.glob(os.path.join(self.data_path, "valid", "*.jpg"))

    def __getitem__(self, index):
        """
        获取单个数据样本

        这是PyTorch Dataset的核心方法，负责返回指定索引的数据样本。
        数据加载器会调用这个方法来获取训练或验证数据。

        Args:
            index (int): 样本索引，范围 [0, len(dataset))

        Returns:
            dict: 包含 'image' 和 'label' 的字典
                - image: 经过预处理的图像张量
                - label: 对应的标签（分类任务为类别索引）

        当前实现使用随机数据，实际使用时的修改示例：

        # 分类任务示例：
        # image_path = self.image_list[index]
        # image = self.load_image(image_path)
        # label = self.get_label_from_path(image_path)  # 从文件名或标注文件获取

        # 分割任务示例：
        # mask = self.load_mask(mask_path)
        # label = mask  # 分割掩码作为标签
        """
        # ==================== 数据生成（模拟） ====================
        # TODO: 替换为真实数据加载
        # 生成随机图像：256x256 RGB图像
        image = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)

        # 生成固定标签（实际应用中应该从标注文件读取）
        label = 1

        # ==================== 数据增强应用 ====================
        # 应用数据变换（增强、归一化等）
        if self.transform is not None:
            # 使用albumentations库进行数据增强
            # augmented 字典包含变换后的图像和其他可能的输出
            augmented = self.transform(image=image)
            image = augmented["image"]

        # ==================== 返回格式 ====================
        # 返回字典格式，便于扩展（如添加更多字段）
        return {"image": image, "label": label}

    def __len__(self):
        """
        返回数据集大小

        PyTorch会调用这个方法来确定数据集的总样本数，
        用于计算每个epoch的批次数量。

        Returns:
            int: 数据集中的样本总数
        """
        return len(self.image_list)

    def load_image(self, path):
        """
        加载单个图像文件

        这个方法负责从文件路径加载图像数据。
        当前为空实现，需要根据实际数据格式进行修改。

        Args:
            path (str): 图像文件路径

        Returns:
            numpy.ndarray: 加载的图像数组，格式通常为 (H, W, C)

        实现示例：

        # 使用PIL加载图像
        # from PIL import Image
        # image = Image.open(path).convert('RGB')
        # return np.array(image)

        # 使用OpenCV加载图像
        # import cv2
        # image = cv2.imread(path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # return image
        """
        # TODO: 实现真实的图像加载逻辑
        return None

    def load_images_in_parallel(self):
        """
        并行图像预加载优化方法

        这个方法可以用于在数据集初始化时预加载图像，
        利用多线程提升数据加载速度，特别适用于：
        - 图像文件较小且数量较多的情况
        - 需要减少训练时I/O等待的场景
        - 内存充足的环境

        使用建议：
        - 根据可用内存调整预加载的图像数量
        - 根据CPU核心数调整 max_workers 参数
        - 考虑使用缓存机制避免重复加载

        实现示例：

        # def preload_images(self):
        #     def load_single_image(path):
        #         return self.load_image(path)
        #
        #     with ThreadPoolExecutor(max_workers=8) as executor:
        #         self.preloaded_images = list(executor.map(
        #             load_single_image, self.image_list[:1000]  # 预加载前1000张
        #         ))
        # 其实也可以直接叫 GPT 写
        """
        # TODO: 实现并行图像加载逻辑
        with ThreadPoolExecutor(max_workers=24):
            # 这里可以实现具体的并行加载逻辑
            pass


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
