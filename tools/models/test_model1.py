import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    双卷积块：两个3x3卷积 + 批量归一化 + ReLU

    这是U-Net的基本构建块，用于特征提取。
    每个卷积保持空间尺寸（padding=1），批量归一化稳定训练，ReLU增加非线性。

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        dropout_rate (float): Dropout比例，默认为0（无Dropout）

    Invariants:
        - 输出空间尺寸与输入相同（H, W不变）
        - 通道数从in_channels转换为out_channels
    """

    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net架构：6通道输入，5类分割输出

    实现经典U-Net，包含编码器（下采样）、瓶颈和解码器（上采样）。
    - 编码器：4次下采样，每次通道数翻倍
    - 瓶颈：深层特征提取
    - 解码器：4次上采样，结合跳跃连接
    - 输出：1x1卷积映射到5类

    Args:
        in_channels (int): 输入通道数，默认为6（SAR+OPT）
        out_channels (int): 输出通道数，默认为5（5类分割）
        features (list): 编码器每层通道数，默认为[64, 128, 256, 512]

    Invariants:
        - 输入/输出空间尺寸相同（H, W）
        - 输出通道数为out_channels
        - 跳跃连接确保特征融合
    """

    def __init__(self, in_channels=4, num_classes=5, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器：构建下采样路径
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 瓶颈：最深层特征提取
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout_rate=0.3)

        # 解码器：构建上采样路径
        for feature in reversed(features):
            # 上采样：2x2转置卷积，通道数减半
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # 双卷积：融合跳跃连接和上采样特征
            self.ups.append(DoubleConv(feature * 2, feature, dropout_rate=0.3))

        # 最终输出层：1x1卷积映射到out_channels
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, sar, opt):
        """
        前向传播

        Args:
            sar (torch.Tensor): 输入张量，形状 (B, 3, H, W)
            opt (torch.Tensor): 输入张量，形状 (B, 1, H, W)

        Returns:
            torch.Tensor: 输出logits，形状 (B, 5, H, W)

        Trade-offs:
            - 跳跃连接增加内存占用，但保留高分辨率细节
            - Dropout在深层提高泛化能力，略增计算成本
        """
        x = torch.cat([sar, opt], dim=1)  # 合并SAR和OPT通道
        skip_connections = []

        # 编码器：下采样
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 瓶颈
        x = self.bottleneck(x)

        # 解码器：上采样
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            # 尺寸对齐（处理非2的幂次输入）
            if x.shape[2:] != skip.shape[2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            concat_skip = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](concat_skip)

        # 输出层
        return self.final_conv(x)


def test_unet():
    """测试U-Net，确保输入输出形状正确"""
    model = UNet(in_channels=6, out_channels=5)
    x = torch.randn(2, 6, 256, 256)
    preds = model(x)
    assert preds.shape == (2, 5, 256, 256), (
        f"Expected shape (2, 5, 256, 256), got {preds.shape}"
    )
    print("U-Net形状测试通过")


if __name__ == "__main__":
    test_unet()

"""
# 设计权衡说明（中文）
#
# 1. 通道数配置：
#   - 初始值为[64, 128, 256, 512]，平衡了模型容量和计算效率。
#     - 优点：适合大多数GPU内存（如8GB），可通过调整features扩展。
#     - 缺点：对于极高分辨率图像，可能需要更深层次或更大通道。
#
# 2. Dropout 使用：
#    - 在瓶颈和解码器中添加0.3的Dropout，增强泛化能力。
#    - 理由：融合数据集可能存在噪声，Dropout防止过拟合。
#    - 权衡：轻微增加训练时间，但提高模型鲁棒性。
#
# 3. 上采样策略：
#    - 转置卷积用于上采样，简单高效。
#    - 权衡：可能引入棋盘伪影，但通过后续卷积缓解。
#    - 替代方案：可考虑插值+卷积，需实验验证效果。
#
# 4. 跳跃连接处理：
#    - 为非2的幂次输入添加插值，确保尺寸对齐。
#    - 权衡：插值可能引入轻微信息损失，但保证通用性。
"""
