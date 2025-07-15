import os
import glob
import numpy as np
import tifffile
import cv2
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import Dataset
from configs.option import get_option
from scipy.ndimage import gaussian_filter
import albumentations as A

# from .augments import train_transform, valid_transform
train_transform = None
valid_transform = None


def augment_color_brightness_contrast(image: np.ndarray) -> np.ndarray:
    """
    对图像应用色彩、亮度和对比度的复合增强。
    使用 albumentations 库进行高效、标准化的实现。

    Args:
        image (np.ndarray): 输入图像，格式为 HWC, uint8, 0-255 BGR。
                            (注意：albumentations 内部处理颜色空间，输入输出保持一致)

    Returns:
        np.ndarray: 增强后的图像，格式为 HWC, uint8, 0-255 BGR。
    """
    transform = A.Compose(
        [
            A.ColorJitter(
                hue=[-0.1, 0.1],
                p=0.5,
            ),
        ]
    )

    # 应用变换并返回结果图像
    augmented_image = transform(image=image)["image"]
    return augmented_image


def add_random_patch(image: np.ndarray, probability: float = 0.8) -> np.ndarray:
    """
    Randomly adds a patch to the image with various textures (solid, gaussian, uniform).
    The patch has an irregular, organic shape.

    Args:
        image (np.ndarray): The input image as a NumPy array. Assumed to be in [0, 255] range.
        probability (float): The probability of applying this augmentation.

    Returns:
        np.ndarray: The augmented image or the original image if augmentation was not applied.
    """
    # 深拷贝输入图像，避免对原始数据进行意外修改。这是一个良好的编程实践。
    opt = image.copy()

    # 概率检查与阶段判断：确保此数据增强仅在训练阶段按指定概率触发。
    if np.random.random() >= probability:
        return opt

    h, w = opt.shape[:2]

    # === 1. 定义色块的几何属性 (位置、尺寸、形状) ===
    # 核心思想：通过随机化参数，确保每次生成的色块都独一无二。

    # 尺寸：在图像尺寸的一定比例范围内随机选择，避免过大或过小。
    patch_w = np.random.randint(int(0.1 * w), int(0.7 * w))
    patch_h = np.random.randint(int(0.1 * h), int(0.7 * h))

    # 位置：在保证色块完整位于图像内的前提下，随机选择左上角坐标。
    x1 = np.random.randint(0, w - patch_w + 1)
    y1 = np.random.randint(0, h - patch_h + 1)

    # === 2. 生成色块内容 (从三种类型中随机选择) ===
    # 核心思想：模拟不同类型的图像伪影或遮挡，增加模型的鲁棒性。
    patch_type = np.random.choice(["solid", "gaussian"])

    patch_content = np.zeros((patch_h, patch_w), dtype=np.float32)

    if patch_type == "solid":
        # 纯色块：模拟完全遮挡或标签。限制在灰度色，因为彩色会引入过强的先验。
        # 随机选择黑、灰、白三种基本颜色。
        color_value = np.random.choice([0, 125, 160, 195, 225, 255])
        patch_content.fill(color_value)

    elif patch_type == "gaussian":
        # 高斯噪声块：模拟更自然的污渍或纹理。
        # 1. 生成标准正态分布噪声。
        noise = np.random.randn(patch_h, patch_w)
        # 2. 通过高斯滤波使其平滑，产生连续、自然的纹理。sigma与尺寸关联，确保尺度适应性。
        noise = gaussian_filter(noise, sigma=min(patch_h, patch_w) * 0.1)
        # 3. 归一化到 [0, 1] 区间。
        #    为防止(max-min)为0的除零错误，增加一个极小值epsilon。
        min_val, max_val = noise.min(), noise.max()
        if max_val > min_val:
            noise = (noise - min_val) / (max_val - min_val)
        # 4. 映射到 [0, 255] 的像素值范围。
        patch_content = noise * 255

    # === 3. 创建不规则形状的软化掩码 (Alpha Channel) ===
    # 核心思想：避免硬边缘。使用一个平滑衰减的椭圆，并用噪声扰动其边界，以生成自然、不规则的形状。

    yy, xx = np.ogrid[:patch_h, :patch_w]
    center_y, center_x = patch_h / 2, patch_w / 2

    # 椭圆参数：随机化长短轴，增加形状的多样性。
    a = patch_h * np.random.uniform(0.3, 0.5)
    b = patch_w * np.random.uniform(0.35, 0.6)

    # 边界扰动噪声：用低频噪声对椭圆的距离场进行扰动。
    boundary_noise = gaussian_filter(
        np.random.random((patch_h, patch_w)), sigma=min(patch_h, patch_w) * 0.05
    )

    # 距离场：计算每个点到中心的归一化椭圆距离。
    distance_from_center = np.sqrt(
        ((yy - center_y) ** 2) / a**2 + ((xx - center_x) ** 2) / b**2
    )

    # 创建软化掩码：距离越远，透明度越高。噪声的加入使边界不规则。
    # np.clip确保alpha值在[0, 1]之间。
    soft_mask = np.clip(1.2 - distance_from_center - 0.5 * boundary_noise, 0, 1)

    # === 4. Alpha混合 ===
    # 核心思想：使用物理上正确的Alpha混合公式，将色块自然地叠加到原图上。
    # Result = alpha * Foreground + (1 - alpha) * Background

    # 最终的alpha通道，可以再乘以一个全局透明度因子增加随机性。
    alpha = soft_mask * np.random.uniform(0.7, 1.0)

    # 确保alpha和色块内容与原图通道数匹配，以便进行广播运算。
    if len(opt.shape) == 3:
        ch = opt.shape[2]
        patch_content = np.repeat(patch_content[:, :, np.newaxis], ch, axis=2)
        alpha = np.repeat(alpha[:, :, np.newaxis], ch, axis=2)

    # 提取目标区域，并转换为浮点数进行精确计算。
    original_region = opt[y1 : y1 + patch_h, x1 : x1 + patch_w].astype(np.float32)

    # 执行Alpha混合公式。
    blended = alpha * patch_content + (1 - alpha) * original_region

    # 将混合后的结果写回原图，并转换回原始数据类型。
    opt[y1 : y1 + patch_h, x1 : x1 + patch_w] = blended.astype(opt.dtype)

    return opt


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
        self, phase, opt, train_transform=None, valid_transform=None, train_ratio=0.92
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
                os.path.basename(f).replace("_Val.tif", "") for f in sar_files
            ]
            opt_dir = os.path.join(self.data_path, "val", "2_Opt")
            self.sar_paths = sar_files
            self.opt_paths = [
                os.path.join(opt_dir, f"{name}_Val.tif") for name in base_names
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

        if self.phase == "train":
            opt = add_random_patch(opt, probability=0.25)  # 添加随机补丁
            opt = augment_color_brightness_contrast(opt)  # 色彩增强
        # 转换为张量
        sar = (sar - self.sar_mean) / self.sar_std  # 标准化SAR图像
        opt = (opt - self.opt_mean) / self.opt_std  # 标准化OPT图像

        # 数据增强逻辑，应用于训练阶段
        if self.phase == "train":
            # 随机水平翻转
            flip_horizontal = np.random.random() > 0.5
            if flip_horizontal:
                sar = np.fliplr(sar)
                opt = np.fliplr(opt)
                if label is not None:
                    label = np.fliplr(label)

            # 随机垂直翻转
            flip_vertical = np.random.random() > 0.5
            if flip_vertical:
                sar = np.flipud(sar)
                opt = np.flipud(opt)
                if label is not None:
                    label = np.flipud(label)

            # 综合仿射变换（旋转、缩放和平移）
            affine = np.random.random() > 0.3  # 增加应用变换的概率为70%
            if affine:
                height, width = sar.shape[:2]
                center = (width // 2, height // 2)  # 旋转中心

                # 定义仿射变换参数
                angle = np.random.uniform(-90, 90)  # 随机旋转角度，范围 -45° 到 45°
                scale = np.random.uniform(0.8, 1.2)  # 随机缩放因子
                tx = np.random.uniform(-0.15 * width, 0.15 * width)  # 水平平移
                ty = np.random.uniform(-0.15 * height, 0.15 * height)  # 垂直平移

                # 创建旋转+缩放矩阵
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

                # 添加平移部分
                rotation_matrix[0, 2] += tx
                rotation_matrix[1, 2] += ty

                # 应用仿射变换
                sar = cv2.warpAffine(
                    sar,
                    rotation_matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                opt = cv2.warpAffine(
                    opt,
                    rotation_matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )

                if label is not None:
                    label = cv2.warpAffine(
                        label,
                        rotation_matrix,
                        (width, height),
                        flags=cv2.INTER_NEAREST,  # 对标签使用最近邻插值
                        borderMode=cv2.BORDER_REFLECT,
                    )

        sar = torch.from_numpy(sar.copy()).unsqueeze(0).float()
        opt = torch.from_numpy(opt.copy()).permute(2, 0, 1).float()

        result = {"sar": sar, "opt": opt}
        if self.phase in ["train", "valid"]:
            label = torch.from_numpy(label.copy()).long()  # 假设Label是分割掩码
            result["label"] = label
        result["name"] = os.path.basename(self.sar_paths[index])

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
    print("\n数据加载器测试完成！")
