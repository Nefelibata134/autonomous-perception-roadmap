"""
resnet.py
手撕 ResNet18：理解残差连接（Skip Connection）
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    ResNet 基础残差块
    结构：Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+x) -> ReLU

    当 stride != 1 或 in_channels != out_channels 时，
    shortcut 路径需要 1x1 卷积调整维度
    """
    expansion = 1  # BasicBlock 输出通道数不变

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 主路径：第一个卷积可能下采样（stride=2）
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 主路径：第二个卷积不下采样
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut 路径（捷径/残差连接）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 维度不匹配时，用 1x1 卷积调整通道数和空间尺寸
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接：主路径输出 + shortcut 输出
        out += self.shortcut(x)
        out = torch.relu(out)

        return out


class ResNet18(nn.Module):
    """
    ResNet18 网络结构
    针对 224x224 输入设计（CIFAR-10 经 Resize 后可用）
    """

    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()

        # 初始卷积层：7x7 大卷积核，stride=2 下采样
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层：4 个 stage，每个 stage 包含 2 个 BasicBlock
        # Layer1: 64 -> 64,  不缩小尺寸（stride=1）
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        # Layer2: 64 -> 128, 缩小一半（stride=2）
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        # Layer3: 128 -> 256, 缩小一半
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        # Layer4: 256 -> 512, 缩小一半
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # 全局平均池化 + 全连接分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出固定 1x1，不管输入尺寸
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        构建一个残差层（stage）
        第一个 BasicBlock 可能下采样，后续保持尺寸
        """
        layers = []
        # 第一个 block 负责下采样（如果需要）
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # 后续 block 不下采样
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        # 4 个残差 stage
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局池化 + 分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平，保留 batch 维度
        x = self.fc(x)

        return x


def test_resnet():
    """快速测试网络结构"""
    model = ResNet18(num_classes=10)
    x = torch.randn(2, 3, 224, 224)  # 模拟 2 张 224x224 图片
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")  # [2, 10]
    print("✅ ResNet18 结构测试通过！")


if __name__ == "__main__":
    test_resnet()