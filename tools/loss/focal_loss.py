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
        if inputs.size(-1) != 1:
            if len(targets.shape) == 1:
                targets = F.one_hot(targets, num_classes=inputs.size(-1)).float()
            ce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )
            p_t = torch.exp(-ce_loss)
        else:
            ce_loss = F.binary_cross_entropy(
                inputs.sigmoid(), targets.float(), reduction="none"
            )
            p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
