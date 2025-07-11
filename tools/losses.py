import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C]  预测类别概率
        # targets: [N]     真实类别标签，需要是 one-hot 编码或者类别索引

        if inputs.size(-1) != 1:  # 判断是否为二元分类，为二元分类做一些调整
            # 非 one-hot 编码，转换为 one-hot 编码
            if len(targets.shape) == 1:
                targets = F.one_hot(targets, num_classes=inputs.size(-1)).float()
            ce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )
            p_t = torch.exp(-ce_loss)
        else:
            ce_loss = F.binary_cross_entropy(
                inputs.sigmoid(), targets.float(), reduction="none"
            )  # 这里要用 sigmoid
            p_t = torch.exp(-ce_loss)
        # p_t = (targets * inputs) + ((1 - targets) * (1 - inputs))  # 对于正样本 p_t = p, 对于负样本 p_t = 1 - p
        # ce_loss = -torch.log(p_t)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction="mean"):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C, H, W] 预测logits
        # targets: [N, H, W] 真实标签 或 [N, C, H, W] one-hot编码

        # 如果targets是类别索引，转换为one-hot编码
        if len(targets.shape) == 3:  # [N, H, W]
            targets = (
                F.one_hot(targets, num_classes=inputs.size(1))
                .permute(0, 3, 1, 2)
                .float()
            )

        # 对预测结果应用softmax
        inputs = F.softmax(inputs, dim=1)

        # 展平张量用于计算
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # [N, C, H*W]
        targets = targets.view(targets.size(0), targets.size(1), -1)  # [N, C, H*W]

        # 计算每个类别的dice系数
        intersection = (inputs * targets).sum(dim=2)  # [N, C]
        union = inputs.sum(dim=2) + targets.sum(dim=2)  # [N, C]

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss
