import os
import os.path as osp
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import rcParams
import random
import shutil  # 添加shutil模块用于文件操作

# --- (Original Code - No changes needed for these sections) ---
from configs.option import get_option, set_default_config_path
from tools.datasets.datasets import *
from tools.model_registry import get_model

# --- 1. System & Style Configuration ---
# rcParams["font.family"] = "serif"
# rcParams["font.serif"] = ["Times New Roman"]
rcParams["axes.titlesize"] = 16
rcParams["axes.labelsize"] = 12
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 10

# --- 2. Core Constants & Metadata ---
# 首先定义 CKPT_PATH，这是用户唯一需要修改的部分
CKPT_PATH = "/media/hdd/sonwe1e/DeepLearningTemplate/experiments/baseline-epoch300-5diceloss-convnext-rot2affine-randomwhitev4_2025-07-12_23-44-16/checkpoints/epoch_249-mIoU_0.574.ckpt"

CONFIG_PATH = osp.join(osp.dirname(osp.dirname(CKPT_PATH)), "save_config.yaml")

# 使用推断的配置路径
set_default_config_path(CONFIG_PATH)

VISUALIZATION_SAVE_PATH = (
    "/media/hdd/sonwe1e/DeepLearningTemplate/dataset/visualization"
)
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
TTA_ENABLED = True

# --- ARCHITECTURAL CHOICE: Selective Visualization ---
# 核心优化：不再为每个样本生成可视化，而是选择性地保存。这避免了I/O成为瓶颈。
# 我们选择保存效果最好、最差和一些随机的样本，这对于分析模型行为更有价值。
# Core Optimization: Instead of generating visualizations for every sample, we save selectively. This prevents I/O from becoming the bottleneck.
# We choose to save the best, worst, and a few random samples, which is more valuable for analyzing model behavior.
NUM_BEST_SAMPLES = 3
NUM_WORST_SAMPLES = 20
NUM_RANDOM_SAMPLES = 3

# --- Configuration Loading ---
opt, _ = get_option(verbose=False)
os.makedirs(VISUALIZATION_SAVE_PATH, exist_ok=True)

# --- Dataset & Model Metadata ---
NUM_CLASSES = opt.model["model_kwargs"]["num_classes"]
CLASS_NAMES = [f"Class {i}" for i in range(NUM_CLASSES)]
CLASS_COLORS = sns.color_palette("husl", NUM_CLASSES)

# --- 3. Core Architectural Components ---


class StreamMetrics:
    # 备注：这个类已经很高效，因为它在GPU上累积混淆矩阵，无需修改。
    # Note: This class is already efficient as it accumulates the confusion matrix on the GPU. No changes needed.
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


def predict_with_tta(model, sar_img, opt_img):
    # 备注：TTA逻辑本身是计算密集型的，这是为了精度所做的权衡，我们保留它。
    # Note: The TTA logic is computationally intensive by nature; this is a trade-off for accuracy, and we will keep it.
    transforms = [
        (lambda x: x, lambda x: x),
        (lambda x: torch.flip(x, [-1]), lambda x: torch.flip(x, [-1])),
        (lambda x: torch.flip(x, [-2]), lambda x: torch.flip(x, [-2])),
        (lambda x: torch.rot90(x, 1, [-2, -1]), lambda x: torch.rot90(x, -1, [-2, -1])),
        (lambda x: torch.rot90(x, 2, [-2, -1]), lambda x: torch.rot90(x, -2, [-2, -1])),
        (lambda x: torch.rot90(x, 3, [-2, -1]), lambda x: torch.rot90(x, -3, [-2, -1])),
    ]
    logits_list = []
    for img_transform, out_transform in transforms:
        logits = model(img_transform(sar_img), img_transform(opt_img))
        logits_list.append(out_transform(logits))
    final_logits = torch.mean(torch.stack(logits_list), dim=0)
    return final_logits


# --- NEW: Efficient Per-Image Metric Calculation ---
def calculate_single_miou(pred, true, n_classes):
    """
    一个轻量级的、函数式的mIoU计算器，用于单张图像，避免了创建StreamMetrics对象的开销。
    A lightweight, functional mIoU calculator for a single image, avoiding the overhead of creating a StreamMetrics object.
    """
    iou_list = []
    for c in range(n_classes):
        true_c = true == c
        pred_c = pred == c
        intersection = torch.logical_and(true_c, pred_c).sum()
        union = torch.logical_or(true_c, pred_c).sum()
        iou = (intersection + 1e-8) / (union + 1e-8)
        iou_list.append(iou)
    return torch.stack(iou_list).mean().item()


def save_visualization(sample, save_dir, class_names, class_colors):
    """
    可视化函数现在接收一个包含所有必要数据的字典，保持其职责单一。
    The visualization function now receives a dictionary with all necessary data, keeping its responsibility singular.
    """
    sar_img, opt_img, gt_mask, pred_mask, iou_per_image, img_name = (
        sample["sar"],
        sample["opt"],
        sample["gt"],
        sample["pred"],
        sample["miou"],
        sample["name"],
    )
    save_path = os.path.join(save_dir, img_name.replace(".tif", ".png"))

    cmap = ListedColormap(class_colors)
    bounds = np.arange(-0.5, len(class_names), 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # 鲁棒归一化 (Robust Normalization)
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

    # --- CHANGE 1: Enable constrained_layout for robust automatic spacing. ---
    # 核心：使用 constrained_layout=True。这是 Matplotlib 中处理复杂布局（如图例、颜色条）的现代方法。
    # 它能自动调整元素间距以避免重叠，从而取代了功能相对有限的 tight_layout。
    # Core: Use constrained_layout=True. This is the modern way in Matplotlib to handle complex layouts (like legends, colorbars).
    # It automatically adjusts element spacing to avoid overlaps, replacing the more limited tight_layout.
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=200, constrained_layout=True)

    # --- CHANGE 2: Remove manual positioning from suptitle. ---
    # 备注：当使用 constrained_layout 时，不再需要手动调整标题的垂直位置（y=...），布局管理器会自动处理。
    # Note: When using constrained_layout, manually adjusting the title's vertical position (y=...) is no longer necessary; the layout manager handles it automatically.
    fig.suptitle(
        f"Qualitative Analysis: {os.path.basename(save_path)}",
        fontsize=18,
        weight="bold",
    )

    axes[0, 0].imshow(sar_img, cmap="gray")
    axes[0, 0].set_title("SAR Input (Normalized)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(opt_img)
    axes[0, 1].set_title("Optical Input (Normalized)")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(gt_mask, cmap=cmap, norm=norm)
    axes[1, 0].set_title("Ground Truth")
    axes[1, 0].axis("off")
    im_pred = axes[1, 1].imshow(pred_mask, cmap=cmap, norm=norm)
    axes[1, 1].set_title("Model Prediction")
    axes[1, 1].axis("off")
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

    # --- CHANGE 3: Remove the problematic tight_layout call. ---
    # 核心：constrained_layout 已经完成了布局工作，因此不再需要（也不兼容）tight_layout。
    # Core: constrained_layout has already handled the layout, so the call to tight_layout is no longer needed (and is incompatible).
    # plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # This line is removed.

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# --- 4. Main Execution Logic (Re-architected) ---
def main():
    # --- 清理旧的可视化结果 ---
    print("--- Cleaning previous visualization results ---")
    if os.path.exists(VISUALIZATION_SAVE_PATH):
        shutil.rmtree(VISUALIZATION_SAVE_PATH)
        print(f"Removed previous visualization directory: {VISUALIZATION_SAVE_PATH}")

    # 重新创建可视化目录
    os.makedirs(VISUALIZATION_SAVE_PATH, exist_ok=True)
    os.makedirs(os.path.join(VISUALIZATION_SAVE_PATH, "best"), exist_ok=True)
    os.makedirs(os.path.join(VISUALIZATION_SAVE_PATH, "worst"), exist_ok=True)
    os.makedirs(os.path.join(VISUALIZATION_SAVE_PATH, "random"), exist_ok=True)

    # --- Initialization ---
    model = get_model(opt.model["model_name"], **opt.model["model_kwargs"])
    ckpt = torch.load(CKPT_PATH, map_location="cpu")["ema_state_dict"]
    clean_ckpt = {k.replace("model.", "", 1): v for k, v in ckpt.items()}
    model.load_state_dict(clean_ckpt)
    model.to(DEVICE)
    model.eval()

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

    metrics = StreamMetrics(n_classes=NUM_CLASSES)

    # --- ARCHITECTURAL CHANGE: Data collection list ---
    # 核心：这是新架构的关键。我们创建一个列表来暂存用于可视化的数据。
    # 数据被转移到CPU并转换为numpy，以释放GPU显存，为下一批次推理做准备。
    # Core: This is the key to the new architecture. We create a list to temporarily store data for visualization.
    # The data is moved to CPU and converted to numpy to free up VRAM for the next inference batch.
    visualization_candidates = []

    # === PHASE 1: GPU-Centric Inference & Data Aggregation ===
    print("--- Phase 1: Running Inference and Aggregating Metrics ---")
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="Inference"):
            sar, opt_img, label, names = (
                batch["sar"].to(DEVICE),
                batch["opt"].to(DEVICE),
                batch["label"].to(DEVICE),
                batch["name"],
            )

            if TTA_ENABLED:
                final_logits = predict_with_tta(model, sar, opt_img)
            else:
                final_logits = model(sar, opt_img)

            predictions = torch.argmax(final_logits, dim=1)
            metrics.update(label.long(), predictions.long())

            # --- EFFICIENT DATA COLLECTION ---
            # 核心：我们不再在此处绘图。而是计算单张mIoU，并将所需数据打包存入列表。
            # 这是解耦GPU和CPU工作的关键一步。
            # Core: We no longer plot here. Instead, we calculate single-image mIoU and package the required data into a list.
            # This is the critical step to decouple GPU and CPU work.
            for i in range(predictions.shape[0]):
                miou = calculate_single_miou(predictions[i], label[i], NUM_CLASSES)
                visualization_candidates.append(
                    {
                        "sar": sar[i].cpu().numpy().transpose(1, 2, 0),
                        "opt": opt_img[i].cpu().numpy().transpose(1, 2, 0),
                        "gt": label[i].cpu().numpy(),
                        "pred": predictions[i].cpu().numpy(),
                        "miou": miou,
                        "name": names[i],
                    }
                )

    # === Final Metrics Report ===
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

    # === PHASE 2: CPU-Centric Visualization Generation ===
    print("\n--- Phase 2: Generating Curated Visualizations ---")

    # 核心：对收集到的所有样本按mIoU排序，以便挑选最好和最差的。
    # Core: Sort all collected samples by mIoU to select the best and worst.
    visualization_candidates.sort(key=lambda x: x["miou"], reverse=True)

    # 组合要可视化的样本列表
    # Combine the list of samples to visualize
    best_samples = visualization_candidates[:NUM_BEST_SAMPLES]
    worst_samples = visualization_candidates[-NUM_WORST_SAMPLES:]

    # 从中间部分随机抽取样本，避免与最好/最差的样本重叠
    # Randomly select samples from the middle part to avoid overlap with best/worst samples
    middle_pool = visualization_candidates[NUM_BEST_SAMPLES:-NUM_WORST_SAMPLES]
    random_samples = random.sample(
        middle_pool, min(len(middle_pool), NUM_RANDOM_SAMPLES)
    )

    final_vis_list = best_samples + worst_samples + random_samples

    # 创建子目录以便更好地组织输出
    # Create subdirectories for better output organization
    os.makedirs(os.path.join(VISUALIZATION_SAVE_PATH, "best"), exist_ok=True)
    os.makedirs(os.path.join(VISUALIZATION_SAVE_PATH, "worst"), exist_ok=True)
    os.makedirs(os.path.join(VISUALIZATION_SAVE_PATH, "random"), exist_ok=True)

    # 现在，我们只在一个小的、经过筛选的列表上运行缓慢的绘图代码。
    # Now, we run the slow plotting code only on a small, curated list.
    for sample in tqdm(best_samples, desc="Saving Best Samples"):
        save_visualization(
            sample,
            os.path.join(VISUALIZATION_SAVE_PATH, "best"),
            CLASS_NAMES,
            CLASS_COLORS,
        )

    for sample in tqdm(worst_samples, desc="Saving Worst Samples"):
        save_visualization(
            sample,
            os.path.join(VISUALIZATION_SAVE_PATH, "worst"),
            CLASS_NAMES,
            CLASS_COLORS,
        )

    for sample in tqdm(random_samples, desc="Saving Random Samples"):
        save_visualization(
            sample,
            os.path.join(VISUALIZATION_SAVE_PATH, "random"),
            CLASS_NAMES,
            CLASS_COLORS,
        )

    print("\n" + "=" * 30)
    print(f"Selective visualizations saved to: {VISUALIZATION_SAVE_PATH}")
    print("=" * 30)


if __name__ == "__main__":
    main()
