import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import rcParams

from configs.option import get_option, set_default_config_path
from tools.datasets.datasets import *
from tools.model_registry import get_model

# --- System & Style Configuration ---
# 核心：设定全局绘图风格，确保所有可视化输出的一致性和专业性。
# 使用'serif'字体族和更精细的参数，提升图像的出版质量。
# Core: Set a global plotting style for consistency and professionalism in all visualizations.
# Using 'serif' font family and fine-tuned parameters enhances the publication quality of figures.
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["axes.titlesize"] = 16
rcParams["axes.labelsize"] = 12
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 10

# --- Core Constants & Metadata ---
# 核心：将硬编码的路径和元数据（如类别名、颜色）集中管理，便于修改和维护。
# 这种做法避免了在代码各处散落“魔术数字”和字符串。
# Core: Centralize hardcoded paths and metadata (like class names, colors) for easy modification and maintenance.
# This practice avoids scattering "magic numbers" and strings throughout the code.
set_default_config_path(
    "/media/hdd/sonwe1e/DeepLearningTemplate/experiments/dual_rdnet_tiny-lr8e-5-fliprotate-diceloss-labelsmooth_2025-07-10_20-11-03/save_config.yaml"
)
CKPT_PATH = "/media/hdd/sonwe1e/DeepLearningTemplate/experiments/dual_rdnet_tiny-lr8e-5-fliprotate-diceloss-labelsmooth_2025-07-10_20-11-03/checkpoints/epoch_64-mIoU_0.522.ckpt"
VISUALIZATION_SAVE_PATH = (
    "/media/hdd/sonwe1e/DeepLearningTemplate/dataset/visualization_refined"
)
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
TTA_ENABLED = True

# --- Configuration Loading ---
opt, _ = get_option(verbose=False)
os.makedirs(VISUALIZATION_SAVE_PATH, exist_ok=True)

# --- Dataset & Model Metadata ---
NUM_CLASSES = opt.model["model_kwargs"]["num_classes"]
# 关键：为类别定义明确的名称和颜色，这对于生成可解释和美观的可视化至关重要。
# Key: Define explicit names and colors for classes, which is crucial for generating interpretable and aesthetic visualizations.
CLASS_NAMES = [
    f"Class {i}" for i in range(NUM_CLASSES)
]  # Replace with actual names if available, e.g., ["Water", "Forest", "Urban", ...]
CLASS_COLORS = sns.color_palette(
    "husl", NUM_CLASSES
)  # A good default, but can be manually defined for semantic meaning


# --- Metrics Calculation Class (Unchanged, already efficient) ---
class StreamMetrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = torch.zeros(
            (n_classes, n_classes), dtype=torch.int64, device=DEVICE
        )

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            lt_flat = lt.flatten()
            lp_flat = lp.flatten()
            k = (lt_flat >= 0) & (lt_flat < self.n_classes)
            inds = self.n_classes * lt_flat[k].to(torch.int64) + lp_flat[k].to(
                torch.int64
            )
            self.confusion_matrix += torch.bincount(
                inds, minlength=self.n_classes**2
            ).reshape(self.n_classes, self.n_classes)

    def get_results(self):
        hist = self.confusion_matrix.float()
        tp = torch.diag(hist)
        sum_a1 = hist.sum(dim=1)
        sum_a0 = hist.sum(dim=0)
        iou = tp / (sum_a1 + sum_a0 - tp + 1e-8)
        mean_iou = torch.nanmean(iou)
        # 增加Pixel Accuracy (PA) 和 Mean Pixel Accuracy (mPA)
        # Add Pixel Accuracy (PA) and Mean Pixel Accuracy (mPA)
        pixel_acc = tp.sum() / (hist.sum() + 1e-8)
        class_acc = tp / (sum_a1 + 1e-8)
        mean_pixel_acc = torch.nanmean(class_acc)
        return {
            "IoU": iou.cpu().numpy(),
            "mIoU": mean_iou.item(),
            "PA": pixel_acc.item(),
            "mPA": mean_pixel_acc.item(),
        }

    def reset(self):
        self.confusion_matrix.zero_()


# --- TTA Prediction Function ---
def predict_with_tta(model, sar_img, opt_img):
    """
    通过函数式方法封装TTA逻辑，提高代码的可读性和可扩展性。
    每种增强都被定义为一个前向变换和一个逆向变换，避免了代码重复。

    Encapsulates TTA logic using a functional approach to improve code readability and extensibility.
    Each augmentation is defined as a forward and an inverse transform, avoiding code repetition.
    """
    # 定义变换对：[ (图像变换, 输出变换), ... ]
    # Define transform pairs: [ (image_transform, output_transform), ... ]
    transforms = [
        (lambda x: x, lambda x: x),  # Identity
        (
            lambda x: torch.flip(x, [-1]),
            lambda x: torch.flip(x, [-1]),
        ),  # Horizontal Flip
        (lambda x: torch.flip(x, [-2]), lambda x: torch.flip(x, [-2])),  # Vertical Flip
        (
            lambda x: torch.rot90(x, 1, [-2, -1]),
            lambda x: torch.rot90(x, -1, [-2, -1]),
        ),  # Rotate 90
        (
            lambda x: torch.rot90(x, 2, [-2, -1]),
            lambda x: torch.rot90(x, -2, [-2, -1]),
        ),  # Rotate 180
        (
            lambda x: torch.rot90(x, 3, [-2, -1]),
            lambda x: torch.rot90(x, -3, [-2, -1]),
        ),  # Rotate 270
    ]

    logits_list = []
    for img_transform, out_transform in transforms:
        transformed_sar = img_transform(sar_img)
        transformed_opt = img_transform(opt_img)
        logits = model(transformed_sar, transformed_opt)
        logits_list.append(out_transform(logits))

    # 对 logits 求平均，这是 TTA 的标准做法，比对类别求平均更稳健
    # Averaging the logits is the standard and more robust practice for TTA than averaging final class predictions.
    final_logits = torch.mean(torch.stack(logits_list), dim=0)
    return final_logits


# --- Enhanced Visualization Function ---
def save_visualization(
    sar_img,
    opt_img,
    gt_mask,
    pred_mask,
    class_names,
    class_colors,
    iou_per_image,
    save_path,
):
    """
    生成出版质量的 2x2 对比图，增加了关键信息和美学优化。
    - 鲁棒的图像归一化 (Robust Image Normalization)
    - 在图上直接标注IoU (Direct IoU Annotation)
    - 精心设计的颜色和布局 (Refined Colors and Layout)
    """
    # 颜色和类别边界
    cmap = ListedColormap(class_colors)
    bounds = np.arange(-0.5, len(class_names), 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # 数据转换与鲁棒归一化
    # Data conversion and robust normalization
    sar_img = sar_img.cpu().numpy().transpose(1, 2, 0)
    opt_img = opt_img.cpu().numpy().transpose(1, 2, 0)
    gt_mask = gt_mask.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()

    # 核心优化：使用百分位裁剪进行归一化，能有效抵抗SAR图像中的极端亮点/暗点噪声。
    # Core Optimization: Use percentile clipping for normalization, which is robust against extreme bright/dark spot noise in SAR images.
    for img_slice in range(sar_img.shape[2]):
        p_low, p_high = np.percentile(sar_img[:, :, img_slice], [1, 99])
        sar_img[:, :, img_slice] = np.clip(sar_img[:, :, img_slice], p_low, p_high)
        sar_img[:, :, img_slice] = (sar_img[:, :, img_slice] - p_low) / (
            p_high - p_low + 1e-8
        )

    for img_slice in range(opt_img.shape[2]):
        p_low, p_high = np.percentile(opt_img[:, :, img_slice], [2, 98])
        opt_img[:, :, img_slice] = np.clip(opt_img[:, :, img_slice], p_low, p_high)
        opt_img[:, :, img_slice] = (opt_img[:, :, img_slice] - p_low) / (
            p_high - p_low + 1e-8
        )

    if sar_img.shape[2] == 1:
        sar_img = sar_img.squeeze(2)

    # 创建画布
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 12), dpi=200
    )  # Increased DPI for sharpness
    fig.suptitle(
        f"Qualitative Analysis: {os.path.basename(save_path)}",
        fontsize=18,
        y=0.96,
        weight="bold",
    )

    # 绘制输入图像
    axes[0, 0].imshow(sar_img, cmap="gray")
    axes[0, 0].set_title("SAR Input (Normalized)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(opt_img)
    axes[0, 1].set_title("Optical Input (Normalized)")
    axes[0, 1].axis("off")

    # 绘制分割结果
    axes[1, 0].imshow(gt_mask, cmap=cmap, norm=norm)
    axes[1, 0].set_title("Ground Truth")
    axes[1, 0].axis("off")

    im_pred = axes[1, 1].imshow(pred_mask, cmap=cmap, norm=norm)
    axes[1, 1].set_title("Model Prediction")
    axes[1, 1].axis("off")

    # 核心优化：在预测图上直接标注该样本的mIoU，将定性与定量分析结合。
    # Core Optimization: Annotate the image-specific mIoU directly on the prediction plot,
    # bridging qualitative and quantitative analysis.
    axes[1, 1].text(
        0.95,
        0.05,
        f"mIoU: {iou_per_image:.3f}",
        transform=axes[1, 1].transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8),
        color="black",
    )

    # 创建共享图例
    cbar = fig.colorbar(
        im_pred,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.04,
        pad=0.04,
        ticks=np.arange(len(class_names)),
    )
    cbar.ax.set_xticklabels(class_names, rotation=45, ha="right")
    cbar.set_label("Class Legend", size=12, weight="bold")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# --- Main Execution Logic ---
def main():
    # 1. Model Initialization
    model = get_model(opt.model["model_name"], **opt.model["model_kwargs"])
    ckpt = torch.load(CKPT_PATH, map_location="cpu")["ema_state_dict"]

    # 清理权重字典键的前缀
    clean_ckpt = {k.replace("model.", "", 1): v for k, v in ckpt.items()}
    model.load_state_dict(clean_ckpt)
    model.to(DEVICE)
    model.eval()

    # 2. Dataset and DataLoader
    valid_dataset = Dataset(
        phase="valid", opt=opt, train_transform=None, valid_transform=None
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.valid_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    # 3. Metrics Setup
    metrics = StreamMetrics(n_classes=NUM_CLASSES)

    # 4. Inference and Evaluation Loop
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="Validating & Visualizing"):
            sar, opt_img, label = (
                batch["sar"].to(DEVICE),
                batch["opt"].to(DEVICE),
                batch["label"].to(DEVICE),
            )

            # --- Prediction ---
            if TTA_ENABLED:
                final_logits = predict_with_tta(model, sar, opt_img)
            else:
                final_logits = model(sar, opt_img)

            predictions = torch.argmax(final_logits, dim=1)

            # --- Metrics Update ---
            metrics.update(label.long(), predictions.long())

            # --- Save Visualizations ---
            for i in range(predictions.shape[0]):
                # 计算单张图像的mIoU用于可视化
                # Calculate mIoU for this single image for visualization purposes
                single_metric = StreamMetrics(n_classes=NUM_CLASSES)
                single_metric.update(label[i].unsqueeze(0), predictions[i].unsqueeze(0))
                iou_per_image = single_metric.get_results()["mIoU"]

                img_name = batch["name"][i]
                save_name = os.path.join(
                    VISUALIZATION_SAVE_PATH, img_name.replace(".tif", ".png")
                )

                save_visualization(
                    sar_img=sar[i],
                    opt_img=opt_img[i],
                    gt_mask=label[i],
                    pred_mask=predictions[i],
                    class_names=CLASS_NAMES,
                    class_colors=CLASS_COLORS,
                    iou_per_image=iou_per_image,
                    save_path=save_name,
                )

    # 5. Final Report
    results = metrics.get_results()
    print("\n" + "=" * 30)
    print("      Validation Complete")
    print("=" * 30)
    print(f"Overall Pixel Accuracy (PA): {results['PA']:.4f}")
    print(f"Mean Pixel Accuracy (mPA):   {results['mPA']:.4f}")
    print(f"Mean IoU (mIoU):             {results['mIoU']:.4f}")
    print("\n--- Per-class IoU ---")
    for i, iou in enumerate(results["IoU"]):
        print(f"  {CLASS_NAMES[i]:<15}: {iou:.4f}")
    print("=" * 30)
    print(f"Visualizations saved to: {VISUALIZATION_SAVE_PATH}")


if __name__ == "__main__":
    main()
