import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class IdentityLoss(nn.Module):
    """
    一个占位符损失函数，其行为类似于 nn.Identity。
    主要目的是提供一个可替换的接口，在后续的开发中可以轻松地替换为
    实际的损失函数或更复杂的模型评估逻辑。

    它接收预测值 (predictions) 和真实值 (targets)，并返回一个标量值 (0.0)。
    """

    def __init__(self):
        super().__init__()
        # 初始化时可以不进行任何操作，因为其主要功能是作为占位符。
        pass

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算“损失”。
        作为一个占位符，它直接返回一个常数0.0，不进行实际的损失计算。
        这确保了它符合损失函数的输入/输出签名，但不会影响训练过程，
        因为其梯度为0。

        Args:
            predictions (torch.Tensor): 模型的预测输出。
            targets (torch.Tensor): 对应的真实标签或目标值。

        Returns:
            torch.Tensor: 一个标量张量 (torch.tensor(0.0))。
        """
        # 返回一个常数，例如0.0。这样在作为损失函数使用时，
        # 不会对模型的参数更新产生影响（梯度为0），
        # 但仍然满足损失函数需要返回一个标量值的约定。
        return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)


# PyTorch 中没有 isnan，但可以这样组合使用
def isnan(x):
    return x != x


def lovasz_grad(gt_sorted):
    """
    计算 Lovász 梯度的辅助函数.
    gt_sorted: [P] Tensor, 排序后误差对应的真实标签 (0/1)
    """
    p = len(gt_sorted)
    # cumsum: 计算累积和 [g_1, g_1+g_2, ...]
    # flip: 将序列反转
    # 这一系列操作高效地计算出论文中定义的梯度向量
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    # Jaccard index is prone to division by zero if union is 0.
    # The jaccard index becomes 0/0=NaN here. This is fine,
    # as the later multiplication by errors_sorted[j] will be 0.
    # But for a clean implementation, we can manually handle this.
    if p > 1:  # 避免单个像素的情况
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    计算单个图像/样本的扁平化 Lovász-Hinge 损失.
    logits: [P] a flattened float tensor with raw network outputs.
    labels: [P] a flattened long tensor with ground truth labels.
    """
    if len(labels) == 0:
        # 罕见情况：如果没有需要计算的像素，损失为0.
        return logits.sum() * 0.0

    # 核心步骤1: 计算误差.
    # 对于真实类别c, 误差为 p_i(c) (如果 i 不属于 c) 或 1-p_i(c) (如果 i 属于 c)
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]

    # 核心步骤2: 计算梯度并应用 Lovász 扩展
    grad = lovasz_grad(gt_sorted)

    # 计算最终损失. 使用 detach() 是因为梯度是通过数学推导预先计算好的，
    # 我们不希望PyTorch对排序操作本身进行反向传播.
    loss = torch.dot(F.relu(errors_sorted), grad.detach())

    return loss


def flatten_probas(probas, labels, ignore_index=None):
    """
    将概率和标签张量扁平化，为计算loss做准备.
    probas: [N, C, H, W]
    labels: [N, H, W]
    """
    # N: Batch size, C: Classes, H: Height, W: Width
    N, C, H, W = probas.shape
    # 变换维度以方便扁平化: [N, C, H, W] -> [N, H, W, C] -> [N*H*W, C]
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)

    # --- 处理 ignore_index ---
    # 这是健壮实现的关键部分，忽略不参与损失计算的像素
    if ignore_index is not None:
        valid = labels != ignore_index
        probas = probas[valid]
        labels = labels[valid]

    return probas, labels


class LovaszSoftmax(nn.Module):
    """
    多类别 Lovász-Softmax 损失.

    用法:
        criterion = LovaszSoftmax(ignore_index=255)
        loss = criterion(logits, labels)

    参数:
        ignore_index (int, optional): 指定一个在计算中被忽略的目标值，该值不贡献于输入梯度.
    """

    def __init__(self, ignore_index=None):
        super(LovaszSoftmax, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        计算损失.
        logits: [N, C, H, W] float tensor, 网络的原始输出.
        labels: [N, H, W] long tensor, 真实标签.
        """
        # 核心思想：Lovász-Softmax 是对每个类别分别计算 Lovász-Hinge 损失，然后取平均
        # 步骤 1: 将 logits 转换为概率
        probas = F.softmax(logits, dim=1)

        # 步骤 2: 将输入扁平化并处理 ignore_index
        probas, labels = flatten_probas(probas, labels, self.ignore_index)

        # 步骤 3: 迭代计算每个类别的 Lovász-Hinge 损失
        losses = []
        # 获取标签中所有出现过的类别
        present_classes = torch.unique(labels)

        for c in present_classes:
            # 针对类别 c 构造二元分类问题
            # foreground mask: 真实标签为 c 的像素
            fg_mask = labels == c
            # background mask: 真实标签不为 c 的像素
            bg_mask = labels != c

            # 仅在 fg 和 bg 都存在时才有意义
            if fg_mask.sum() == 0 or bg_mask.sum() == 0:
                continue

            # 提取对应类别的概率作为二元分类的 "logits"
            # 这里虽然变量名叫 logits_c, 但实际上是概率值，因为 lovasz_hinge_flat 期望输入范围在0-1之间
            logits_c = probas[:, c]

            # 构造二元标签
            labels_c = fg_mask.float()  # 1 for foreground, 0 for background

            loss_c = lovasz_hinge_flat(logits_c, labels_c)
            losses.append(loss_c)

        if len(losses) == 0:
            # 如果没有有效的类别可以计算损失（例如，所有像素都被忽略）
            # 返回一个零损失，并确保它在正确的设备上且需要梯度
            return logits.sum() * 0.0

        # 步骤 4: 对所有存在类别的损失取平均
        mean_loss = torch.stack(losses).mean()

        return mean_loss


class FocalLoss(nn.Module):
    """
    针对图像分割任务的Focal Loss实现。

    Focal Loss专门设计用于解决难易样本不平衡问题，通过降低易分类样本的权重，
    增加难分类样本的权重，从而使模型更关注于难以正确分类的样本。

    对于分割任务，它在像素级别应用这一策略。

    参数:
        alpha (float): 平衡正负样本的权重因子
        gamma (float): 聚焦参数，控制易分类样本权重降低的速度，越大降低越多
        reduction (str): 损失归约方式 ('mean', 'sum', 'none')
        label_smoothing (float): 标签平滑系数，介于0和1之间。当为0时不应用平滑
    """

    def __init__(self, alpha=0.25, gamma=2, reduction="mean", label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        计算Focal Loss

        参数:
            inputs: [N, C, H, W] 模型预测logits
            targets: [N, H, W] 类别索引 或 [N, C, H, W] one-hot编码

        返回:
            计算得到的Focal Loss
        """
        # 首先将inputs转换为概率形式
        if inputs.dim() == 4:  # [N, C, H, W] 图像分割格式
            N, C, H, W = inputs.size()

            # 如果targets是类别索引，转换为one-hot编码并应用标签平滑
            if targets.dim() == 3:  # [N, H, W]
                one_hot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

                # 应用标签平滑
                if self.label_smoothing > 0:
                    targets = (
                        one_hot * (1 - self.label_smoothing) + self.label_smoothing / C
                    )
                else:
                    targets = one_hot

            # 处理已经是one-hot格式的输入
            elif self.label_smoothing > 0:
                # 应用标签平滑，保留原始标签信息但增加不确定性
                targets = (
                    targets * (1 - self.label_smoothing) + self.label_smoothing / C
                )

            # 将输入展平为 [N*H*W, C]
            inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)
            targets = targets.permute(0, 2, 3, 1).contiguous().view(-1, C)

            # 计算交叉熵损失
            ce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )

            # 计算p_t
            p_t = torch.exp(-ce_loss)

            # 应用focal loss公式: -(1-p_t)^gamma * log(p_t)
            focal_weight = (1 - p_t) ** self.gamma

            # 应用alpha权重
            if self.alpha > 0:
                alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
                focal_weight = alpha_weight * focal_weight

            focal_loss = focal_weight * ce_loss

        else:
            # 处理其他形状的输入（如分类任务）
            if inputs.size(-1) != 1:  # 多类别情况
                if targets.dim() == 1:
                    one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()

                    # 应用标签平滑
                    if self.label_smoothing > 0:
                        targets = one_hot * (
                            1 - self.label_smoothing
                        ) + self.label_smoothing / inputs.size(-1)
                    else:
                        targets = one_hot
                elif self.label_smoothing > 0:
                    # 对已经是one-hot的输入应用标签平滑
                    targets = targets * (
                        1 - self.label_smoothing
                    ) + self.label_smoothing / inputs.size(-1)

                ce_loss = F.binary_cross_entropy_with_logits(
                    inputs, targets, reduction="none"
                )
                p_t = torch.exp(-ce_loss)
            else:
                # 二分类情况
                if self.label_smoothing > 0:
                    targets = (
                        targets * (1 - self.label_smoothing)
                        + self.label_smoothing * 0.5
                    )

                ce_loss = F.binary_cross_entropy(
                    inputs.sigmoid(), targets.float(), reduction="none"
                )
                p_t = torch.exp(-ce_loss)

            focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        # 根据reduction方式返回结果
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
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
