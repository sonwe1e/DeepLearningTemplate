"""
模型预测示例 - 深度学习模型推理演示

这个文件演示如何使用训练好的模型进行推理预测，主要功能包括：

1. 模型检查点加载：
   - 从训练保存的 .ckpt 文件加载模型权重
   - 处理 PyTorch Lightning 的状态字典格式
   - 设置模型为评估模式

2. 单张图像预测：
   - 加载和预处理单张图像
   - 进行前向推理获取预测结果
   - 处理预测输出（如概率、类别等）

3. 验证集批量推理：
   - 在整个验证集上进行推理
   - 计算各种评估指标
   - 分析模型性能

使用场景：
- 测试训练好的模型效果
- 为生产环境准备推理代码
- 分析模型在特定数据上的表现
- 调试模型预测结果

=== 如何自定义预测 ===

1. 更换模型检查点：
   - 修改 ckpt_path 变量指向新的检查点文件
   - 确保模型配置与训练时一致

2. 处理不同任务类型：
   - 分类任务：返回类别概率或最可能的类别
   - 回归任务：直接返回预测数值
   - 分割任务：返回像素级别的预测掩码

3. 自定义输入数据：
   - 修改 predict_single_image 中的图像加载逻辑
   - 调整预处理方式以匹配训练时的设置
   - 支持不同格式的输入（PIL、OpenCV、numpy等）

4. 优化推理性能：
   - 启用模型编译：torch.compile(model)
   - 使用半精度推理：model.half()
   - 批量处理多张图像

5. 添加可视化：
   - 显示原图和预测结果
   - 绘制注意力图或特征图
   - 保存预测结果到文件
"""

import torch
import numpy as np
import os
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

# 导入项目相关模块
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.option import get_option, set_default_config_path
from tools.model_registry import get_model
from tools.utils import load_model


def _ensure_config_path():
    """确保配置文件路径已设置"""
    config_path = "/media/hdd/sonwe1e/Template/configs/config.yaml"
    set_default_config_path(config_path)


def _import_data_modules():
    """延迟导入需要配置文件的数据模块"""
    from tools.datasets.datasets import get_dataloader
    from tools.datasets.augments import valid_transform

    return get_dataloader, valid_transform


def load_checkpoint_model(ckpt_path, opt):
    """
    从检查点文件加载训练好的模型

    这个函数负责：
    1. 根据配置创建模型实例
    2. 从检查点文件加载权重
    3. 设置模型为评估模式

    Args:
        ckpt_path (str): 检查点文件路径（.ckpt 文件）
        opt: 配置对象，包含模型参数

    Returns:
        model: 加载好权重的模型，已设置为评估模式

    注意事项：
    - 确保检查点文件存在且格式正确
    - 模型配置必须与训练时完全一致
    - PyTorch Lightning 的检查点包含额外信息，需要正确提取模型权重
    """
    print(f"正在加载检查点: {ckpt_path}")

    # 检查文件是否存在
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点文件不存在: {ckpt_path}")

    # ==================== 创建模型实例 ====================
    # 根据配置创建与训练时相同的模型结构
    model = get_model(opt.model["model_name"], **opt.model["model_kwargs"])
    print(f"创建模型: {opt.model['model_name']}")

    # ==================== 加载权重 ====================
    # 使用工具函数加载模型权重
    # 这个函数会处理 PyTorch Lightning 的状态字典格式
    model = load_model(model, ckpt_path)
    print("模型权重加载成功")

    return model


def predict_single_image(model, image_path, device="cuda", return_probs=True):
    """
    对单张图像进行预测

    这个函数演示如何：
    1. 加载和预处理单张图像
    2. 进行模型推理
    3. 处理和返回预测结果

    Args:
        model: 已加载权重的模型
        image_path (str): 图像文件路径
        device (str): 推理设备 ('cuda' 或 'cpu')
        return_probs (bool): 是否返回概率分布

    Returns:
        dict: 包含预测结果的字典
            - prediction: 预测的类别索引
            - confidence: 预测置信度
            - probabilities: 所有类别的概率（如果 return_probs=True）

    如何适配不同任务：

    # 分类任务（当前实现）：
    # - 返回类别概率和最可能的类别

    # 回归任务示例：
    # output = model(image_tensor)
    # return {'prediction': output.item()}

    # 分割任务示例：
    # output = model(image_tensor)
    # prediction = torch.argmax(output, dim=1)
    # return {'segmentation_mask': prediction.cpu().numpy()}
    """
    print(f"正在预测图像: {image_path}")

    # 获取数据变换函数
    _, valid_transform = _import_data_modules()

    # ==================== 图像加载与预处理 ====================
    if os.path.exists(image_path):
        # 实际图像文件
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
    else:
        # 如果文件不存在，生成模拟图像用于演示
        print(f"图像文件不存在，使用模拟数据: {image_path}")
        image = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)

    # 应用与训练时相同的预处理
    if valid_transform is not None:
        augmented = valid_transform(image=image)
        image_tensor = augmented["image"]
    else:
        # 如果没有变换，手动转换为张量
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    # 添加批次维度并移动到指定设备
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # ==================== 模型推理 ====================
    model.eval()  # 确保模型处于评估模式
    with torch.no_grad():  # 禁用梯度计算，节省内存和加速
        # 前向传播获取模型输出
        output = model(image_tensor)

        # ==================== 结果处理 ====================
        # 根据不同任务类型处理输出

        # 分类任务处理（假设输出是 logits）
        if output.dim() == 2:  # [batch_size, num_classes]
            # 计算概率分布
            probabilities = F.softmax(output, dim=1)

            # 获取预测类别和置信度
            confidence, prediction = torch.max(probabilities, dim=1)

            result = {
                "prediction": prediction.item(),
                "confidence": confidence.item(),
            }

            if return_probs:
                result["probabilities"] = probabilities.cpu().numpy().flatten()

        else:
            # 其他任务类型的处理
            result = {
                "raw_output": output.cpu().numpy(),
                "prediction": output.cpu().numpy(),
            }

    print(
        f"预测完成 - 类别: {result.get('prediction', 'N/A')}, "
        f"置信度: {result.get('confidence', 0):.3f}"
    )

    return result


def evaluate_on_validation(model, valid_dataloader, device="cuda"):
    """
    在验证集上进行批量推理和评估

    这个函数演示如何：
    1. 在整个验证集上进行推理
    2. 收集预测结果和真实标签
    3. 计算各种评估指标
    4. 分析模型性能

    Args:
        model: 已加载权重的模型
        valid_dataloader: 验证数据加载器
        device (str): 推理设备

    Returns:
        dict: 包含评估结果的字典
            - accuracy: 准确率
            - predictions: 所有预测结果
            - labels: 所有真实标签
            - loss: 平均损失（如果适用）

    评估指标扩展：
    - 添加精确率、召回率、F1分数
    - 计算混淆矩阵
    - 分析每个类别的性能
    - 可视化预测分布
    """
    print("开始在验证集上进行推理...")

    model.eval()
    model.to(device)

    all_predictions = []
    all_labels = []
    total_loss = 0.0

    # ==================== 批量推理 ====================
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(valid_dataloader, desc="推理进度")):
            # 获取图像和标签
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失（可选）
            if outputs.dim() == 2:  # 分类任务
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()

                # 获取预测类别
                _, predicted = torch.max(outputs, 1)

            else:
                # 其他任务的处理
                predicted = outputs

            # 收集结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ==================== 评估指标计算 ====================
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # 计算准确率
    accuracy = np.mean(all_predictions == all_labels)
    avg_loss = total_loss / len(valid_dataloader) if total_loss > 0 else 0

    # ==================== 结果统计 ====================
    results = {
        "accuracy": accuracy,
        "average_loss": avg_loss,
        "total_samples": len(all_labels),
        "predictions": all_predictions,
        "labels": all_labels,
    }

    # 打印结果摘要
    print("\n" + "=" * 50)
    print("验证集推理结果摘要")
    print("=" * 50)
    print(f"总样本数: {results['total_samples']}")
    print(f"准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    if avg_loss > 0:
        print(f"平均损失: {avg_loss:.4f}")

    # 分析预测分布
    unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)

    print(f"\n预测类别分布: {dict(zip(unique_preds, pred_counts))}")
    print(f"真实类别分布: {dict(zip(unique_labels, label_counts))}")

    return results


def analyze_predictions(results):
    """
    深入分析预测结果

    这个函数提供更详细的性能分析，包括：
    1. 每个类别的性能指标
    2. 混淆矩阵分析
    3. 错误样本分析
    4. 置信度分布

    Args:
        results (dict): evaluate_on_validation 的返回结果

    如何扩展分析：
    - 添加可视化图表
    - 分析错误预测的模式
    - 计算更多评估指标
    - 生成分析报告
    """
    predictions = results["predictions"]
    labels = results["labels"]

    print("\n" + "=" * 50)
    print("详细性能分析")
    print("=" * 50)

    # 获取所有类别
    all_classes = np.unique(np.concatenate([predictions, labels]))

    # 计算每个类别的性能
    print("\n各类别性能:")
    print("类别\t准确数\t总数\t准确率")
    print("-" * 30)

    for class_id in all_classes:
        # 该类别的真实样本
        class_mask = labels == class_id
        class_total = np.sum(class_mask)

        # 该类别的正确预测
        class_correct = np.sum((predictions == class_id) & (labels == class_id))

        # 计算准确率
        class_accuracy = class_correct / class_total if class_total > 0 else 0

        print(f"{class_id}\t{class_correct}\t{class_total}\t{class_accuracy:.4f}")

    # 错误分析
    errors = predictions != labels
    error_count = np.sum(errors)

    print("\n错误分析:")
    print(f"错误预测数量: {error_count}")
    print(f"错误率: {error_count / len(labels):.4f}")

    if error_count > 0:
        print("\n错误预测示例（前10个）:")
        error_indices = np.where(errors)[0][:10]
        for idx in error_indices:
            print(f"样本 {idx}: 预测={predictions[idx]}, 真实={labels[idx]}")


def main():
    """
    主函数 - 演示完整的预测流程

    这个函数展示了：
    1. 配置加载
    2. 模型初始化和权重加载
    3. 单张图像预测
    4. 验证集批量推理
    5. 结果分析

    使用步骤：
    1. 修改配置文件路径（如需要）
    2. 设置正确的检查点路径
    3. 准备测试图像（可选）
    4. 运行脚本查看结果
    """
    # ==================== 配置初始化 ====================
    # 确保配置文件路径已设置
    _ensure_config_path()

    # 设置配置文件路径
    config_path = "/media/hdd/sonwe1e/Template/configs/config.yaml"
    set_default_config_path(config_path)

    # 加载配置
    opt, checkpoint_dir = get_option()

    # 获取数据模块
    get_dataloader, _ = _import_data_modules()

    # ==================== 检查点路径设置 ====================
    # TODO: 修改为你的实际检查点路径
    # 可以从 experiments/ 目录下找到训练好的检查点
    ckpt_path = "/media/hdd/sonwe1e/Template/experiments/baselinev1_2025-06-29_21-16-53/checkpoints/epoch_0-loss_1.133.ckpt"

    # 检查检查点是否存在
    if not os.path.exists(ckpt_path):
        print(f"检查点文件不存在: {ckpt_path}")
        print("请检查路径或先进行模型训练")
        return

    # 设置推理设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # ==================== 模型加载 ====================
    print("\n" + "=" * 50)
    print("步骤 1: 加载模型")
    print("=" * 50)

    model = load_checkpoint_model(ckpt_path, opt)
    model.to(device)

    # ==================== 单张图像预测示例 ====================
    print("\n" + "=" * 50)
    print("步骤 2: 单张图像预测")
    print("=" * 50)

    # 示例图像路径（可以修改为你的图像路径）
    test_image_path = "/media/hdd/sonwe1e/Template/temp_data/cat.png"

    # 进行预测
    single_result = predict_single_image(
        model=model, image_path=test_image_path, device=device, return_probs=True
    )

    # 打印结果
    print("\n单张图像预测结果:")
    print(f"预测类别: {single_result.get('prediction', 'N/A')}")
    print(f"置信度: {single_result.get('confidence', 0):.4f}")

    # ==================== 验证集推理 ====================
    print("\n" + "=" * 50)
    print("步骤 3: 验证集批量推理")
    print("=" * 50)

    # 创建验证数据加载器
    _, valid_dataloader = get_dataloader(opt)

    # 进行批量推理
    eval_results = evaluate_on_validation(
        model=model, valid_dataloader=valid_dataloader, device=device
    )

    # ==================== 详细分析 ====================
    print("\n" + "=" * 50)
    print("步骤 4: 结果分析")
    print("=" * 50)

    analyze_predictions(eval_results)

    print("\n" + "=" * 50)
    print("预测完成！")
    print("=" * 50)
    print("\n如何自定义这个脚本:")
    print("1. 修改 ckpt_path 指向你的检查点文件")
    print("2. 更换 test_image_path 为你的测试图像")
    print("3. 根据任务调整预测后处理逻辑")
    print("4. 添加更多评估指标和可视化")


if __name__ == "__main__":
    """
    脚本入口点
    
    直接运行这个文件来体验完整的预测流程：
    python tools/example_predict.py
    
    常见问题解决：
    
    1. 检查点加载失败：
       - 确认检查点文件路径正确
       - 检查模型配置是否与训练时一致
       
    2. CUDA内存不足：
       - 减小验证批量大小 valid_batch_size
       - 使用 CPU 推理：device='cpu'
       
    3. 预测结果异常：
       - 检查模型是否正确加载
       - 确认数据预处理与训练时一致
       - 验证模型是否已充分训练
       
    4. 图像加载错误：
       - 检查图像文件路径和格式
       - 确认图像预处理流程正确
    """
    main()
