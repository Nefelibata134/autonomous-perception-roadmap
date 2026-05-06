"""
train.py
验证 CustomImageDataset + DataLoader 的正确性
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomImageDataset
import matplotlib.pyplot as plt
import numpy as np


def get_train_transforms():
    """
    训练阶段的数据预处理管道
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def denormalize(tensor, mean, std):
    """
    反归一化，用于可视化时恢复原始颜色范围
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def visualize_batch(images, labels, class_names, save_path="batch_visualize.png"):
    """
    可视化一个 batch 的图片
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = denormalize(images, mean, std)

    num_show = min(8, len(images))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(num_show):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[i]], fontsize=12)
        axes[i].axis('off')

    for i in range(num_show, 8):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[可视化] 已保存到 {save_path}")


def main():
    # 配置
    DATA_PATH = "./data/cifar10_train"
    BATCH_SIZE = 8
    NUM_WORKERS = 4

    # 设备检查
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")

    # 1. 创建数据集
    transform = get_train_transforms()
    dataset = CustomImageDataset(root_dir=DATA_PATH, transform=transform)

    if len(dataset) == 0:
        print("[错误] 数据集为空！")
        return

    # 2. 创建 DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    print(f"[DataLoader] 批次数量: {len(dataloader)}")

    # 3. 取一个 batch 验证
    images, labels = next(iter(dataloader))

    print(f"[Batch] 图像形状: {images.shape}")
    print(f"[Batch] 标签形状: {labels.shape}")
    print(f"[Batch] 标签值: {labels.tolist()}")

    # 4. 可视化
    visualize_batch(images, labels, dataset.classes)

    # 5. 模拟 GPU 训练第一步
    images = images.to(device)
    labels = labels.to(device)
    print(f"[GPU] 数据已转移至 {device}, 形状: {images.shape}")
    print("\n✅ Day1 验证通过！")


if __name__ == "__main__":
    main()