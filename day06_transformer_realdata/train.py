"""
train.py
Day6: 真实 CIFAR-10 + Mixup + Cutout + ResNet18
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torch.nn as nn
import numpy as np
import sys


class Cutout:
    """
    Cutout 数据增强：随机遮挡图片的一个区域
    强迫模型关注全局特征，而不是 memorizing 局部纹理
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        # img: Tensor (C, H, W)
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask


def get_train_transforms():
    """训练：真实 CIFAR-10 是 32x32，放大到 224"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Cutout(n_holes=1, length=56)  # 224x224 图，遮挡 56x56
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def mixup_data(x, y, alpha=1.0):
    """
    Mixup：将两张图片按 lambda 比例混合，标签也混合
    降低 memorizing，提升泛化
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels)

        optimizer.zero_grad()
        outputs = model(images)

        if use_mixup:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)

        if use_mixup:
            correct += (lam * predicted.eq(labels_a).sum().float() +
                        (1 - lam) * predicted.eq(labels_b).sum().float()).item()
        else:
            correct += predicted.eq(labels).sum().item()

        if batch_idx % 50 == 0:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Batch/Loss", loss.item(), step)

    return running_loss / len(dataloader), 100. * correct / total


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

    return running_loss / len(dataloader), 100. * correct / total


def main():
    DATA_PATH = "./data"
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LR = 0.001
    NUM_WORKERS = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")

    # 加载真实 CIFAR-10
    print("[数据] 加载真实 CIFAR-10...")
    train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=False, transform=get_train_transforms())
    val_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=False, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"[数据] 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # 加载 Day2 手撕的 ResNet18
    sys.path.append('../day02_resnet_baseline')
    from resnet import ResNet18
    model = ResNet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    writer = SummaryWriter(log_dir="./runs/day06_real_cifar10")

    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        use_mixup = epoch < 5  # 前5轮用 Mixup，后5轮不用

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, use_mixup
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Mixup: {use_mixup} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        writer.add_scalar("Epoch/Loss/train", train_loss, epoch)
        writer.add_scalar("Epoch/Loss/val", val_loss, epoch)
        writer.add_scalar("Epoch/Accuracy/train", train_acc, epoch)
        writer.add_scalar("Epoch/Accuracy/val", val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> 最佳模型已保存 (val acc: {best_acc:.2f}%)")

    writer.close()
    print(f"\n✅ 训练完成！最佳验证准确率: {best_acc:.2f}%")
    print("查看曲线: tensorboard --logdir=./runs/day06_real_cifar10")


if __name__ == "__main__":
    main()