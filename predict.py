import os
import torch
from PIL import Image
import torch.nn as nn
from configs.option import get_option, set_default_config_path
from tools.datasets.datasets import *
from tools.model_registry import list_available_models, get_model

set_default_config_path(
    "/media/hdd/sonwe1e/DeepLearningTemplate/experiments/dual_rdnet_tiny-lr8e-5-fliprotate-diceloss-labelsmooth-92train_2025-07-10_22-28-03/save_config.yaml"
)
opt, checkpoint_path = get_option(verbose=False)


model = get_model(opt.model["model_name"], **opt.model["model_kwargs"])

# 创建验证数据集
valid_dataset = Dataset(
    phase="test",
    opt=opt,
    train_transform=None,
    valid_transform=None,
)


valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=opt.valid_batch_size,
    shuffle=False,
    num_workers=opt.num_workers,
    pin_memory=True,
)

ckpt_path = "/media/hdd/sonwe1e/DeepLearningTemplate/experiments/dual_rdnet_tiny-lr8e-5-fliprotate-diceloss-labelsmooth-92train_2025-07-10_22-28-03/checkpoints/epoch_149-mIoU_0.538.ckpt"
save_path = "/media/hdd/sonwe1e/DeepLearningTemplate/dataset/results"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)["ema_state_dict"]

for k, v in list(ckpt.items()):
    if k.startswith("model."):
        new_k = k[6:]  # 去掉前缀 "model."
        ckpt[new_k] = v
        del ckpt[k]

model.load_state_dict(ckpt)
model.cuda(3)
model.eval()

# l = len(os.listdir("/media/hdd/sonwe1e/DeepLearningTemplate/dataset/val/1_SAR"))
count = 0
TTA = True  # 是否使用测试时间增强（TTA）

for batch in valid_dataloader:
    sar = batch["sar"]
    opt = batch["opt"]

    # 确保输入数据在正确的设备上
    sar = sar.cuda(3)
    opt = opt.cuda(3)

    # 模型预测
    with torch.no_grad():
        if TTA:
            # TTA 推理
            outputs = []
            # 原始预测
            output = model(sar, opt)
            outputs.append(output)

            # 水平翻转
            sar_flip = torch.flip(sar, dims=[-1])
            opt_flip = torch.flip(opt, dims=[-1])
            output_flip = model(sar_flip, opt_flip)
            output_flip = torch.flip(output_flip, dims=[-1])
            outputs.append(output_flip)

            # 垂直翻转
            sar_vflip = torch.flip(sar, dims=[-2])
            opt_vflip = torch.flip(opt, dims=[-2])
            output_vflip = model(sar_vflip, opt_vflip)
            output_vflip = torch.flip(output_vflip, dims=[-2])
            outputs.append(output_vflip)

            # 90度旋转
            sar_rot90 = torch.rot90(sar, k=1, dims=[-2, -1])
            opt_rot90 = torch.rot90(opt, k=1, dims=[-2, -1])
            output_rot90 = model(sar_rot90, opt_rot90)
            output_rot90 = torch.rot90(output_rot90, k=-1, dims=[-2, -1])
            outputs.append(output_rot90)

            # 180度旋转
            sar_rot180 = torch.rot90(sar, k=2, dims=[-2, -1])
            opt_rot180 = torch.rot90(opt, k=2, dims=[-2, -1])
            output_rot180 = model(sar_rot180, opt_rot180)
            output_rot180 = torch.rot90(output_rot180, k=-2, dims=[-2, -1])
            outputs.append(output_rot180)

            # 270度旋转
            sar_rot270 = torch.rot90(sar, k=3, dims=[-2, -1])
            opt_rot270 = torch.rot90(opt, k=3, dims=[-2, -1])
            output_rot270 = model(sar_rot270, opt_rot270)
            output_rot270 = torch.rot90(output_rot270, k=-3, dims=[-2, -1])
            outputs.append(output_rot270)

            # 取平均
            output = torch.mean(torch.stack(outputs), dim=0)
        else:
            # 普通推理
            output = model(sar, opt)

    # 输出结果处理
    output = torch.argmax(output, dim=1)  # 假设是分类任务，取最大值索引作为预测类别
    output = output.cpu().numpy()  # 转换为 NumPy 数组

    # 保存预测结果 以 TIFF 图像形式保存
    for i in range(output.shape[0]):
        pred_image = output[i].astype("uint8")
        pred_image_path = f"{save_path}/{batch['name'][i]}"
        count += 1

        Image.fromarray(pred_image).save(pred_image_path, format="TIFF")
