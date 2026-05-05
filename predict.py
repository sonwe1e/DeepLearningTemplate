"""模型推理 — 单图预测与验证集评估"""
import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

import config
from tools.model import get_model
from tools.data import get_dataloader
from tools.data.augment import build_transforms


def _load_model(model, model_path):
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    filtered_dict = {
        k.replace("model.", ""): v for k, v in state_dict.items() if "model" in k
    }
    model.load_state_dict(filtered_dict)
    model.eval()
    return model


def load_model_from_ckpt(ckpt_path, opt):
    model = get_model(opt.model["model_name"], **opt.model["model_kwargs"])
    return _load_model(model, ckpt_path)


def predict_image(model, image_path, transform, device="cuda"):
    """单图预测 — image_path 或 numpy array (HWC)"""
    from PIL import Image as PILImage
    model.eval()
    with torch.no_grad():
        if isinstance(image_path, str):
            image = np.array(PILImage.open(image_path).convert("RGB"))
        else:
            image = image_path
        augmented = transform(image=image)
        input_tensor = augmented["image"].unsqueeze(0).to(device)
        output = model(input_tensor)
        logits = output["classes"] if isinstance(output, dict) else output
        prob = F.softmax(logits, dim=1)
        conf, pred = torch.max(prob, dim=1)
        return pred.item(), conf.item()


def evaluate(model, dataloader, device="cuda"):
    """验证集评估，返回准确率和平均损失"""
    model.eval()
    model.to(device)
    correct = total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            logits = outputs["classes"] if isinstance(outputs, dict) else outputs
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total, total_loss / len(dataloader)


def main():
    opt, _ = config.get_option()

    ckpt_path = os.environ.get("CKPT_PATH")
    if not ckpt_path:
        import glob
        candidates = sorted(glob.glob("experiments/*/checkpoints/*.ckpt"), reverse=True)
        if not candidates:
            print("未找到检查点。设置 CKPT_PATH 环境变量或先训练模型。")
            return
        ckpt_path = candidates[0]
        print(f"自动选择检查点: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_from_ckpt(ckpt_path, opt)
    model.to(device)

    _, valid_dataloader = get_dataloader(opt)
    acc, avg_loss = evaluate(model, valid_dataloader, device)
    print(f"准确率: {acc:.4f} | 平均损失: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
