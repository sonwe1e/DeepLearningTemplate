"""模型预测和推理工具"""

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
    """从检查点文件加载训练好的模型"""
    print(f"正在加载检查点: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点文件不存在: {ckpt_path}")

    model = get_model(opt.model["model_name"], **opt.model["model_kwargs"])
    print(f"创建模型: {opt.model['model_name']}")

    model = load_model(model, ckpt_path)
    print("模型权重加载成功")

    return model


def predict_single_image(model, image_path, device="cuda", return_probs=True):
    """对单张图像进行预测"""
    print(f"正在预测图像: {image_path}")

    _, valid_transform = _import_data_modules()

    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
    else:
        print(f"图像文件不存在，使用模拟数据: {image_path}")
        image = np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8)

    if valid_transform is not None:
        augmented = valid_transform(image=image)
        image_tensor = augmented["image"]
    else:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    image_tensor = image_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

        if output.dim() == 2:
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

            result = {
                "prediction": prediction.item(),
                "confidence": confidence.item(),
            }

            if return_probs:
                result["probabilities"] = probabilities.cpu().numpy().flatten()
        else:
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
    """在验证集上进行批量推理和评估"""
    print("开始在验证集上进行推理...")

    model.eval()
    model.to(device)

    all_predictions = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(valid_dataloader, desc="推理进度")):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            if outputs.dim() == 2:
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
            else:
                predicted = outputs

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_predictions == all_labels)
    avg_loss = total_loss / len(valid_dataloader) if total_loss > 0 else 0

    results = {
        "accuracy": accuracy,
        "average_loss": avg_loss,
        "total_samples": len(all_labels),
        "predictions": all_predictions,
        "labels": all_labels,
    }

    print("\n" + "=" * 50)
    print("验证集推理结果摘要")
    print("=" * 50)
    print(f"总样本数: {results['total_samples']}")
    print(f"准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    if avg_loss > 0:
        print(f"平均损失: {avg_loss:.4f}")

    unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)

    print(f"\n预测类别分布: {dict(zip(unique_preds, pred_counts))}")
    print(f"真实类别分布: {dict(zip(unique_labels, label_counts))}")

    return results


def analyze_predictions(results):
    """深入分析预测结果"""
    predictions = results["predictions"]
    labels = results["labels"]

    print("\n" + "=" * 50)
    print("详细性能分析")
    print("=" * 50)

    all_classes = np.unique(np.concatenate([predictions, labels]))

    print("\n各类别性能:")
    print("类别\t准确数\t总数\t准确率")
    print("-" * 30)

    for class_id in all_classes:
        class_mask = labels == class_id
        class_total = np.sum(class_mask)
        class_correct = np.sum((predictions == class_id) & (labels == class_id))
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        print(f"{class_id}\t{class_correct}\t{class_total}\t{class_accuracy:.4f}")

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
    """主函数 - 演示完整的预测流程"""
    _ensure_config_path()

    config_path = "/media/hdd/sonwe1e/Template/configs/config.yaml"
    set_default_config_path(config_path)

    opt, checkpoint_dir = get_option()
    get_dataloader, _ = _import_data_modules()

    ckpt_path = "/media/hdd/sonwe1e/Template/experiments/baselinev1_2025-06-29_21-16-53/checkpoints/epoch_0-loss_1.133.ckpt"

    if not os.path.exists(ckpt_path):
        print(f"检查点文件不存在: {ckpt_path}")
        print("请检查路径或先进行模型训练")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    print("\n" + "=" * 50)
    print("步骤 1: 加载模型")
    print("=" * 50)

    model = load_checkpoint_model(ckpt_path, opt)
    model.to(device)

    print("\n" + "=" * 50)
    print("步骤 2: 单张图像预测")
    print("=" * 50)

    test_image_path = "/media/hdd/sonwe1e/Template/temp_data/cat.png"

    single_result = predict_single_image(
        model=model, image_path=test_image_path, device=device, return_probs=True
    )

    print("\n单张图像预测结果:")
    print(f"预测类别: {single_result.get('prediction', 'N/A')}")
    print(f"置信度: {single_result.get('confidence', 0):.4f}")

    print("\n" + "=" * 50)
    print("步骤 3: 验证集批量推理")
    print("=" * 50)

    _, valid_dataloader = get_dataloader(opt)

    eval_results = evaluate_on_validation(
        model=model, valid_dataloader=valid_dataloader, device=device
    )

    print("\n" + "=" * 50)
    print("步骤 4: 结果分析")
    print("=" * 50)

    analyze_predictions(eval_results)

    print("\n" + "=" * 50)
    print("预测完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
