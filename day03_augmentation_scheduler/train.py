"""
train.py
Day3: 数据增强 + 学习率调度 + TensorBoard 可视化
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import CustomImageDataset
from resnet import ResNet18


def get_train_transforms():
    """
    训练阶段：强数据增强
    模拟自动驾驶场景的光照变化、视角变化、颜色变化
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),  # 先放大到 256
        transforms.RandomCrop(224),  # 随机裁剪回 224（模拟不同拍摄距离）
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(15),  # 随机旋转 ±15 度（模拟相机倾斜）
        transforms.ColorJitter(  # 颜色抖动（模拟不同光照条件）
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms():
    """
    验证阶段：只做必要预处理，不做增强
    验证时必须用固定、确定性的预处理
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 每 10 个 batch 记录一次 loss（让 TensorBoard 曲线更密）
        if batch_idx % 10 == 0:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Batch/Loss", loss.item(), step)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    # ========== 配置 ==========
    DATA_PATH = "../day01_custom_dataset/data/cifar10_train"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10  # 增加 epoch，让学习率调度发挥作用
    LR = 0.001
    NUM_WORKERS = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")

    # ========== 数据划分（训练/验证用不同 transform）==========
    # 先创建一个临时数据集，用于获取索引划分
    temp_dataset = CustomImageDataset(root_dir=DATA_PATH, transform=None)
    train_size = int(0.8 * len(temp_dataset))
    val_size = len(temp_dataset) - train_size

    from torch.utils.data import random_split
    train_subset, val_subset = random_split(temp_dataset, [train_size, val_size])

    # 获取划分好的索引
    train_indices = train_subset.indices
    val_indices = val_subset.indices

    # 训练集用强增强，验证集用弱增强（保证验证公平）
    train_dataset = CustomImageDataset(root_dir=DATA_PATH, transform=get_train_transforms())
    val_dataset = CustomImageDataset(root_dir=DATA_PATH, transform=get_val_transforms())

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"[数据] 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # ========== 模型 ==========
    model = ResNet18(num_classes=10).to(device)

    # ========== 损失 & 优化器 & 学习率调度 ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 余弦退火调度器：学习率从 LR 平滑降到接近 0，周期 = NUM_EPOCHS
    # 帮助模型在训练后期精细调整，跳出局部最优
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ========== TensorBoard ==========
    writer = SummaryWriter(log_dir="./runs/day03_augmentation")

    # 可视化模型结构（只记录一次）
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    writer.add_graph(model, dummy_input)

    # ========== 训练循环 ==========
    best_acc = 0.0
    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}] LR: {current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # 学习率调度：每个 epoch 后更新
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # 记录到 TensorBoard
        writer.add_scalar("Epoch/Loss/train", train_loss, epoch)
        writer.add_scalar("Epoch/Loss/val", val_loss, epoch)
        writer.add_scalar("Epoch/Accuracy/train", train_acc, epoch)
        writer.add_scalar("Epoch/Accuracy/val", val_acc, epoch)
        writer.add_scalar("Epoch/LearningRate", current_lr, epoch)

        # 保存最佳模型（基于验证集准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_acc.pth")
            print(f"  -> 最佳准确率模型已保存 (val acc: {best_acc:.2f}%)")

        # 也保存基于 loss 最低的模型（防止过拟合）
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model_loss.pth")
            print(f"  -> 最佳损失模型已保存 (val loss: {best_loss:.4f})")

    writer.close()
    print(f"\n✅ 训练完成！最佳验证准确率: {best_acc:.2f}%, 最佳验证损失: {best_loss:.4f}")
    print("查看训练曲线: tensorboard --logdir=./runs/day03_augmentation")


if __name__ == "__main__":
    main()