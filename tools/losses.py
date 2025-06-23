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
