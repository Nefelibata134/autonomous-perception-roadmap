"""
train.py
Day2: ResNet18 完整训练闭环
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import CustomImageDataset
from resnet import ResNet18


def get_transforms():
    """训练数据预处理"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个 epoch
    model.train() 启用 BatchNorm 和 Dropout 的训练模式
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # 1. 清空旧梯度（防止累加）
        optimizer.zero_grad()

        # 2. 前向传播
        outputs = model(images)

        # 3. 计算损失
        loss = criterion(outputs, labels)

        # 4. 反向传播（计算梯度）
        loss.backward()

        # 5. 更新权重
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # 取概率最大的类别
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    验证/测试
    model.eval() 关闭 BatchNorm 更新和 Dropout
    torch.no_grad() 不计算梯度，节省显存
    """
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
    # 跨目录引用 Day1 的数据（同级目录）
    DATA_PATH = "../day01_custom_dataset/data/cifar10_train"
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LR = 0.001
    NUM_WORKERS = 4

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")

    # ========== 数据 ==========
    transform = get_transforms()
    dataset = CustomImageDataset(root_dir=DATA_PATH, transform=transform)

    # 划分训练集(80%)和验证集(20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

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

    # ========== 损失函数 & 优化器 ==========
    criterion = nn.CrossEntropyLoss()  # 分类任务标准损失
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ========== TensorBoard ==========
    writer = SummaryWriter(log_dir="./runs/resnet18_baseline")

    # ========== 训练循环 ==========
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # 记录到 TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # 保存验证集上表现最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> 最佳模型已保存 (val acc: {best_acc:.2f}%)")

    writer.close()
    print(f"\n✅ 训练完成！最佳验证准确率: {best_acc:.2f}%")
    print("模型权重保存于: best_model.pth")
    print("查看训练曲线: tensorboard --logdir=./runs/resnet18_baseline")


if __name__ == "__main__":
    main()