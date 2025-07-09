# -*- coding: utf-8 -*-
"""
Dataset Statistics Analyzer

This script performs a per-channel analysis of mean and standard deviation
for a dataset of SAR (single-channel) and Optical (multi-channel) TIFF images.
It is designed for efficiency and scalability using parallel processing and a
numerically stable one-pass algorithm.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import tifffile
from joblib import Parallel, delayed
from tqdm import tqdm

# ======================================================================================
# 核心计算单元 (Core Computation Unit)
# ======================================================================================


def _calculate_stats_for_image(
    file_path: Path,
) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
    """
    为单个图像文件计算其像素统计信息（像素总数、逐通道和、逐通道平方和）。
    这是并行处理中的最小工作单元。

    Args:
        file_path (Path): 图像文件的路径。

    Returns:
        Optional[Tuple[int, np.ndarray, np.ndarray]]: 一个元组，包含
            - pixel_count: 图像的像素总数 (H * W)。
            - channel_sum: 形状为 (C,) 的数组，表示每个通道的像素值总和。
            - channel_sum_sq: 形状为 (C,) 的数组，表示每个通道的像素值平方和。
        如果文件读取失败，则返回 None。
    """
    try:
        # 1. 读取图像数据
        img = tifffile.imread(file_path)

        # 2. 确保图像数据至少是三维的 (H, W, C)，以便统一处理单通道和多通道图像
        #    np.atleast_3d 会将 (H, W) -> (H, W, 1)，(H, W, C) -> (H, W, C)
        #    这是保证代码通用性的关键一步。
        img = np.atleast_3d(img)
        h, w, c = img.shape
        pixel_count = h * w

        # 3. 将图像数据类型转换为 float64 进行计算，以防止在计算平方和时发生溢出或精度损失。
        #    对于大型图像或高位深图像（如16-bit TIFF），这是一个至关重要的步骤。
        img_fp64 = img.astype(np.float64)

        # 4. 计算逐通道的和与平方和
        #    axis=(0, 1) 表示在高度和宽度维度上进行求和，保留通道维度。
        channel_sum = np.sum(img_fp64, axis=(0, 1))
        channel_sum_sq = np.sum(np.square(img_fp64), axis=(0, 1))

        return pixel_count, channel_sum, channel_sum_sq

    except Exception as e:
        # 错误处理：如果某个文件损坏或格式不正确，打印错误信息并跳过，不中断整个流程。
        print(f"\n[Warning] Could not process file {file_path}: {e}", file=sys.stderr)
        return None


# ======================================================================================
# 并行分析与聚合模块 (Parallel Analysis & Aggregation Module)
# ======================================================================================


def analyze_dataset(
    file_paths: List[Path], n_jobs: int = -1, desc: str = "Analyzing"
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    使用并行处理来分析整个数据集（由文件列表定义）的统计数据。

    Args:
        file_paths (List[Path]): 要分析的图像文件路径列表。
        n_jobs (int): joblib 使用的并行进程数。-1 表示使用所有可用的 CPU核心。
        desc (str): tqdm 进度条的描述文字。

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: 一个元组，包含
            - mean: 数据集的逐通道均值。
            - std: 数据集的逐通道标准差。
        如果文件列表为空或所有文件都无法处理，则返回 None。
    """
    if not file_paths:
        print(f"\n[Warning] No files found for group: {desc}", file=sys.stderr)
        return None

    # 1. 使用 joblib 实现高效的并行计算 (Map 步骤)
    #    delayed() 将函数调用包装成一个轻量级的对象，以便稍后执行。
    #    Parallel() 会将这些任务分发到多个进程中执行。
    #    tqdm 提供了一个清晰的进度条，方便监控大型数据集的处理进度。
    with tqdm(total=len(file_paths), desc=desc, unit="file") as pbar:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_stats_for_image)(p) for p in file_paths
        )

    # 2. 过滤掉处理失败的结果 (None)
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print(
            f"\n[Error] All files in group '{desc}' failed to process.", file=sys.stderr
        )
        return None

    # 3. 聚合所有进程的计算结果 (Reduce 步骤)
    #    这是数值稳定算法的核心：我们只聚合总数、和、平方和，而不是聚合均值/方差。
    total_pixel_count = 0

    # 从第一个有效结果中获取通道数，以初始化聚合数组
    num_channels = valid_results[0][1].shape[0]
    total_channel_sum = np.zeros(num_channels, dtype=np.float64)
    total_channel_sum_sq = np.zeros(num_channels, dtype=np.float64)

    for pixel_count, channel_sum, channel_sum_sq in valid_results:
        total_pixel_count += pixel_count
        total_channel_sum += channel_sum
        total_channel_sum_sq += channel_sum_sq

    # 4. 根据聚合结果计算最终的均值和标准差
    #    这个公式是 E[X^2] - (E[X])^2 的直接应用，数值上非常稳定。
    if total_pixel_count == 0:
        return None

    mean = total_channel_sum / total_pixel_count
    variance = (total_channel_sum_sq / total_pixel_count) - np.square(mean)
    # 处理由于浮点数精度问题可能导致的微小负方差
    variance[variance < 0] = 0
    std = np.sqrt(variance)

    return mean, std


# ======================================================================================
# 主程序入口 (Main Execution Block)
# ======================================================================================


def main():
    """
    主执行函数：解析参数，发现文件，并按组进行分析。
    """
    parser = argparse.ArgumentParser(
        description="Calculate per-channel mean and std for a dataset of SAR/Optical images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_root",
        default=Path("/media/hdd/sonwe1e/DeepLearningTemplate/dataset"),
        type=Path,
        help="Path to the root directory of the dataset (e.g., /path/to/DeepLearningTemplate/dataset).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run. -1 means using all available cores.",
    )
    args = parser.parse_args()

    root_dir = args.dataset_root
    if not root_dir.is_dir():
        print(
            f"[Error] The provided path '{root_dir}' is not a valid directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Analyzing dataset at: {root_dir}")
    print(f"Using {args.n_jobs if args.n_jobs != -1 else 'all'} CPU cores.\n")

    # 1. 使用 pathlib 和 glob 发现所有相关的图像文件
    #    这种方式比 os.path.join 和 os.listdir 更现代、更健壮。
    sar_train_files = sorted(list((root_dir / "train" / "1_SAR").glob("*.tif")))
    opt_train_files = sorted(list((root_dir / "train" / "2_Opt").glob("*.tif")))
    sar_val_files = sorted(list((root_dir / "val" / "1_SAR").glob("*.tif")))
    opt_val_files = sorted(list((root_dir / "val" / "2_Opt").glob("*.tif")))

    # 2. 定义需要分析的所有数据集组合
    analysis_groups = {
        "Train SAR": sar_train_files,
        "Train Opt": opt_train_files,
        "Val SAR": sar_val_files,
        "Val Opt": opt_val_files,
        "Train+Val SAR": sar_train_files + sar_val_files,
        "Train+Val Opt": opt_train_files + opt_val_files,
    }

    # 3. 遍历每个组合，执行分析并打印结果
    print("-" * 60)
    for name, files in analysis_groups.items():
        stats = analyze_dataset(files, n_jobs=args.n_jobs, desc=name)

        print(f"\n--- Results for: {name} ({len(files)} files) ---")
        if stats:
            mean, std = stats
            print(f"  Mean: {np.array2string(mean, precision=6, floatmode='fixed')}")
            print(f"  Std : {np.array2string(std, precision=6, floatmode='fixed')}")
        else:
            print("  Could not compute statistics (no valid files found).")
        print("-" * 60)


if __name__ == "__main__":
    main()
