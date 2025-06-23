import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple2DNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(Simple2DNetwork, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 最终分类层
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # 卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4

        # 全局平均池化
        x = self.global_avg_pool(x)  # 4x4 -> 1x1

        # 扁平化
        x = x.view(-1, 128)

        # 分类层
        x = self.fc(x)

        return x


# 测试函数
def test_model():
    model = Simple2DNetwork(input_channels=3, num_classes=10)

    # 创建测试输入 (batch_size=1, channels=3, height=32, width=32)
    test_input = torch.randn(1, 3, 32, 32)

    # 前向传播
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    test_model()
