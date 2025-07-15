#!/usr/bin/env python3
"""
数据集测试脚本 - test_dataset.py

这个脚本用于测试当前的 dataloader 迭代得到的数据是什么维度，是否能正常迭代。

主要功能：
1. 测试训练和验证数据集的创建
2. 检查 dataloader 能否正常迭代
3. 分析数据的维度、数据类型和数值范围
4. 验证 SAR、OPT 和 Label 数据的形状匹配
5. 检查数据增强是否正常工作
6. 测试 dataloader 的性能（加载时间）

使用方法：
    python test_dataset.py

    或者使用不同的配置文件：
    python test_dataset.py --config configs/custom_config.yaml
"""

import os
import sys
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.option import get_option
from tools.datasets.datasets import get_dataloader


def test_dataset_basic_info(train_dataloader, valid_dataloader):
    """测试数据集基本信息"""
    print("=" * 60)
    print("🔍 数据集基本信息测试")
    print("=" * 60)

    # 数据集大小信息
    print(f"📊 训练数据集大小: {len(train_dataloader.dataset)} 个样本")
    print(f"📊 验证数据集大小: {len(valid_dataloader.dataset)} 个样本")
    print(f"📦 训练批次数量: {len(train_dataloader)} 个批次")
    print(f"📦 验证批次数量: {len(valid_dataloader)} 个批次")
    print(f"🔢 训练批量大小: {train_dataloader.batch_size}")
    print(f"🔢 验证批量大小: {valid_dataloader.batch_size}")
    print()


def test_data_loading(dataloader, phase="train", max_batches=3):
    """测试数据加载功能"""
    print("=" * 60)
    print(f"🚀 {phase.upper()} 数据加载测试")
    print("=" * 60)

    try:
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n📦 第 {batch_idx + 1} 个批次:")

            # 检查必须的字段
            required_keys = ["sar", "opt"]
            if phase in ["train", "valid"]:
                required_keys.append("label")

            missing_keys = [key for key in required_keys if key not in batch]
            if missing_keys:
                print(f"❌ 缺少必需的键: {missing_keys}")
                return False

            # 检查 SAR 数据
            sar_data = batch["sar"]
            print("  🛰️  SAR 数据:")
            print(f"     形状: {sar_data.shape}")
            print(f"     数据类型: {sar_data.dtype}")
            print(f"     数值范围: [{sar_data.min():.4f}, {sar_data.max():.4f}]")
            print(f"     均值: {sar_data.mean():.4f}, 标准差: {sar_data.std():.4f}")

            # 检查 OPT 数据
            opt_data = batch["opt"]
            print("  🌍 OPT 数据:")
            print(f"     形状: {opt_data.shape}")
            print(f"     数据类型: {opt_data.dtype}")
            print(f"     数值范围: [{opt_data.min():.4f}, {opt_data.max():.4f}]")
            print(f"     均值: {opt_data.mean():.4f}, 标准差: {opt_data.std():.4f}")

            # 检查 Label 数据（如果存在）
            if "label" in batch:
                label_data = batch["label"]
                print("  🏷️  Label 数据:")
                print(f"     形状: {label_data.shape}")
                print(f"     数据类型: {label_data.dtype}")
                print(f"     唯一值: {torch.unique(label_data).tolist()}")
                print(f"     数值范围: [{label_data.min()}, {label_data.max()}]")

            # 检查维度匹配
            expected_batch_size = dataloader.batch_size
            if sar_data.shape[0] != expected_batch_size:
                print(
                    f"⚠️  批量大小不匹配: 期望 {expected_batch_size}, 实际 {sar_data.shape[0]}"
                )

            if sar_data.shape[2:] != opt_data.shape[2:]:
                print(
                    f"⚠️  SAR 和 OPT 空间维度不匹配: SAR {sar_data.shape[2:]}, OPT {opt_data.shape[2:]}"
                )

            if "label" in batch and label_data.shape[1:] != sar_data.shape[2:]:
                print(
                    f"⚠️  Label 和图像空间维度不匹配: Label {label_data.shape[1:]}, 图像 {sar_data.shape[2:]}"
                )

            # 限制测试批次数量
            if batch_idx + 1 >= max_batches:
                break

        print(f"\n✅ {phase.upper()} 数据加载测试完成!")
        return True

    except Exception as e:
        print(f"\n❌ {phase.upper()} 数据加载失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_data_consistency(dataloader, phase="train"):
    """测试数据一致性"""
    print("=" * 60)
    print(f"🔍 {phase.upper()} 数据一致性测试")
    print("=" * 60)

    try:
        # 获取同一个样本多次，检查是否一致
        dataset = dataloader.dataset
        sample_idx = 0

        sample1 = dataset[sample_idx]
        sample2 = dataset[sample_idx]

        # 检查是否完全相同（如果没有随机变换，应该相同）
        sar_diff = torch.abs(sample1["sar"] - sample2["sar"]).max()
        opt_diff = torch.abs(sample1["opt"] - sample2["opt"]).max()

        print("📊 相同样本的重复加载差异:")
        print(f"   SAR 最大差异: {sar_diff:.6f}")
        print(f"   OPT 最大差异: {opt_diff:.6f}")

        if sar_diff < 1e-6 and opt_diff < 1e-6:
            print("✅ 数据一致性测试通过（无随机变换）")
        else:
            print("ℹ️  数据存在差异，可能包含随机变换")

        return True

    except Exception as e:
        print(f"❌ 数据一致性测试失败: {str(e)}")
        return False


def test_loading_speed(dataloader, phase="train", num_batches=10):
    """测试数据加载速度"""
    print("=" * 60)
    print(f"⏱️  {phase.upper()} 数据加载速度测试")
    print("=" * 60)

    try:
        start_time = time.time()
        sample_count = 0

        print(f"📊 测试 {num_batches} 个批次的加载速度...")

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="加载进度")):
            sample_count += batch["sar"].shape[0]
            if batch_idx + 1 >= num_batches:
                break

        end_time = time.time()
        total_time = end_time - start_time

        print("\n📈 速度统计:")
        print(f"   总时间: {total_time:.2f} 秒")
        print(f"   平均每批次: {total_time / num_batches:.3f} 秒")
        print(f"   平均每样本: {total_time / sample_count:.4f} 秒")
        print(f"   吞吐量: {sample_count / total_time:.1f} 样本/秒")

        return True

    except Exception as e:
        print(f"❌ 速度测试失败: {str(e)}")
        return False


def test_memory_usage(dataloader, phase="train"):
    """测试内存使用情况"""
    print("=" * 60)
    print(f"💾 {phase.upper()} 内存使用测试")
    print("=" * 60)

    try:
        # 获取一个批次
        batch = next(iter(dataloader))

        # 计算内存使用
        sar_memory = (
            batch["sar"].element_size() * batch["sar"].nelement() / 1024 / 1024
        )  # MB
        opt_memory = (
            batch["opt"].element_size() * batch["opt"].nelement() / 1024 / 1024
        )  # MB

        total_memory = sar_memory + opt_memory

        if "label" in batch:
            label_memory = (
                batch["label"].element_size() * batch["label"].nelement() / 1024 / 1024
            )
            total_memory += label_memory
            print(f"💾 Label 内存使用: {label_memory:.2f} MB")

        print("💾 内存使用统计 (每批次):")
        print(f"   SAR 数据: {sar_memory:.2f} MB")
        print(f"   OPT 数据: {opt_memory:.2f} MB")
        print(f"   总计: {total_memory:.2f} MB")

        # 估算全部数据集的内存使用
        total_dataset_memory = total_memory * len(dataloader)
        print(
            f"📊 估计全数据集内存: {total_dataset_memory:.2f} MB ({total_dataset_memory / 1024:.2f} GB)"
        )

        return True

    except Exception as e:
        print(f"❌ 内存测试失败: {str(e)}")
        return False


def visualize_samples(dataloader, phase="train", num_samples=16):
    """可视化数据样本"""
    print("=" * 60)
    print(f"🎨 {phase.upper()} 数据可视化")
    print("=" * 60)

    try:
        batch = next(iter(dataloader))

        for i in range(min(num_samples, batch["sar"].shape[0])):
            plt.figure(figsize=(15, 5))

            # SAR 图像 (取第一个通道)
            plt.subplot(1, 3, 1)
            sar_img = batch["sar"][i, 0].cpu().numpy()  # 取第一个通道
            plt.imshow(sar_img, cmap="gray")
            # plt.title(f"SAR 图像 (样本 {i + 1})\n形状: {sar_img.shape}")
            plt.axis("off")

            # OPT 图像 (RGB)
            plt.subplot(1, 3, 2)
            if batch["opt"].shape[1] >= 3:
                # 如果有3个或更多通道，显示为RGB
                opt_img = batch["opt"][i, :3].cpu().numpy().transpose(1, 2, 0)
                # 标准化到 [0, 1]
                opt_img = (opt_img - opt_img.min()) / (opt_img.max() - opt_img.min())
                plt.imshow(opt_img)
                # plt.title(f"OPT 图像 (样本 {i + 1})\n形状: {opt_img.shape}")
            else:
                # 如果只有一个通道，显示为灰度图
                opt_img = batch["opt"][i, 0].cpu().numpy()
                plt.imshow(opt_img, cmap="gray")
                # plt.title(f"OPT 图像 (样本 {i + 1})\n形状: {opt_img.shape}")
            plt.axis("off")

            # Label 图像（如果存在）
            if "label" in batch:
                plt.subplot(1, 3, 3)
                label_img = batch["label"][i].cpu().numpy()
                plt.imshow(label_img, cmap="tab10")
                # plt.title(f"Label 图像 (样本 {i + 1})\n形状: {label_img.shape}")
                plt.colorbar()
                plt.axis("off")

            plt.tight_layout()

            # 保存图像
            save_path = f"sample_{phase}_{i + 1}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"📁 样本 {i + 1} 已保存到: {save_path}")
            plt.show()

        return True

    except Exception as e:
        print(f"❌ 可视化失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 开始数据集测试")
    print("=" * 80)

    # 获取配置
    try:
        opt, _ = get_option(verbose=False)
        print(f"✅ 配置加载成功")
        print(f"📁 数据路径: {opt.data_path}")
        print(f"🏷️  实验名称: {opt.exp_name}")
    except Exception as e:
        print(f"❌ 配置加载失败: {str(e)}")
        return

    # 创建数据加载器
    try:
        train_dataloader, valid_dataloader = get_dataloader(opt)
        print(f"✅ 数据加载器创建成功")
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return

    # 运行各项测试
    tests_passed = 0
    total_tests = 0

    # # 1. 基本信息测试
    # test_dataset_basic_info(train_dataloader, valid_dataloader)

    # # 2. 训练数据加载测试
    # total_tests += 1
    # if test_data_loading(train_dataloader, "train"):
    #     tests_passed += 1

    # # 3. 验证数据加载测试
    # total_tests += 1
    # if test_data_loading(valid_dataloader, "valid"):
    #     tests_passed += 1

    # # 4. 数据一致性测试
    # total_tests += 1
    # if test_data_consistency(train_dataloader, "train"):
    #     tests_passed += 1

    # # 5. 加载速度测试
    # total_tests += 1
    # if test_loading_speed(train_dataloader, "train"):
    #     tests_passed += 1

    # # 6. 内存使用测试
    # total_tests += 1
    # if test_memory_usage(train_dataloader, "train"):
    #     tests_passed += 1

    # 7. 数据可视化（可选）
    try:
        visualize_samples(train_dataloader, "train")
        print("✅ 数据可视化完成")
    except Exception as e:
        print(f"⚠️  数据可视化跳过: {str(e)}")

    # 总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    print(f"✅ 通过测试: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("🎉 所有测试都通过了！数据加载器工作正常。")
    else:
        print("⚠️  部分测试未通过，请检查数据集配置和文件路径。")

    print("\n🔍 如果需要更详细的调试信息，请检查以上输出。")
    print("📁 可视化样本已保存在当前目录。")


if __name__ == "__main__":
    main()
