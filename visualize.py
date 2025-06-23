import torch
import matplotlib.pyplot as plt
from typing import Union, Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader
from configs.option import get_option
from tools.datasets.datasets import *

torch.set_float32_matmul_precision("high")


class ImageVisualizer:
    def __init__(
        self,
        opt: Optional[Any] = None,
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225],
    ):
        """
        图像可视化工具类

        Args:
            mean: 图像归一化均值
            std: 图像归一化标准差
        """
        self.opt = opt
        self.mean = torch.tensor(mean).reshape(3, 1, 1)
        self.std = torch.tensor(std).reshape(3, 1, 1)

    def denormalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        反归一化图像

        Args:
            image: 输入图像张量

        Returns:
            反归一化后的图像张量
        """
        return image * self.std + self.mean

    def plot_grid(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        titles: list,
        nrow: int = 4,
        title: str = None,
    ):
        """
        绘制图像网格

        Args:
            images: 图像张量 [N, C, H, W]
            labels: 标签张量 [N]
            titles: 每个子图的标题列表
            nrow: 每行显示的图像数量
            title: 整个图表的标题
        """
        n = len(images)
        ncol = (n + nrow - 1) // nrow
        plt.rcParams.update({"font.size": 13})

        # 调整 figsize 以适应更多的图像
        fig, axes = plt.subplots(
            ncol, nrow, figsize=(nrow * 3, ncol * 3)
        )  # 根据需要调整 figsize

        # 如果只有一个子图 (axes 不是数组), 将其转换为数组
        if n == 1:
            axes = [axes]
        else:
            axes = axes.ravel()

        for idx in range(n):
            img = self.denormalize(images[idx]).permute(1, 2, 0).cpu().numpy()
            axes[idx].imshow(img)
            axes[idx].set_title(f"{titles[idx]}: {labels[idx].item()}")
            axes[idx].axis("off")

        # 隐藏空白子图
        for idx in range(n, len(axes)):
            axes[idx].axis("off")

        # 添加整个图表的标题
        if title:
            fig.suptitle(title, fontsize=20)

        # 调整子图之间的间距
        plt.subplots_adjust(wspace=0.02, hspace=0.18)  # 减小 wspace 和 hspace 的值
        fig.tight_layout()
        plt.savefig(f"./visualization/{self.opt.exp_name}.png", dpi=200)
        return fig


def get_batch_data(
    batch: Union[Tuple, Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从batch中提取图像和标签

    Args:
        batch: 数据批次

    Returns:
        图像和标签元组
    """
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]
    elif isinstance(batch, dict):
        return batch["image"], batch["label"]
    raise ValueError("Unsupported batch format. Expected tuple, list or dict.")


def visualize_datasets(
    opt: Any,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    num_images: int = 16,
) -> None:
    """
    数据集可视化

    Args:
        opt: 配置选项
        train_dataloader: 训练数据加载器
        valid_dataloader: 验证数据加载器
        num_images: 要可视化的图像数量 (每个数据集)
    """
    visualizer = ImageVisualizer(opt)

    # 获取训练集数据
    train_batch = next(iter(train_dataloader))
    train_images, train_labels = get_batch_data(train_batch)
    train_images = train_images[:num_images]
    train_labels = train_labels[:num_images]

    # 获取验证集数据
    valid_batch = next(iter(valid_dataloader))
    valid_images, valid_labels = get_batch_data(valid_batch)
    valid_images = valid_images[:num_images]
    valid_labels = valid_labels[:num_images]

    # 合并图像和标签
    combined_images = torch.cat([train_images, valid_images], dim=0)
    combined_labels = torch.cat([train_labels, valid_labels], dim=0)
    combined_titles = ["Train"] * num_images + ["Valid"] * num_images

    # 绘制合并后的图像
    fig = visualizer.plot_grid(
        combined_images,
        combined_labels,
        combined_titles,
        nrow=8,
        title="Data Visualization",
    )
    plt.close(fig)


def main():
    """主函数"""
    opt = get_option()
    train_dataloader, valid_dataloader = get_dataloader(opt)

    print("Starting data visualization...")
    visualize_datasets(opt, train_dataloader, valid_dataloader)
    print("Data visualization completed.")


if __name__ == "__main__":
    main()
