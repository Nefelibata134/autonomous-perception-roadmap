"""
train.py
Day4: 迁移学习 + 冻结层策略 + 分层学习率
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from dataset import CustomImageDataset


def get_transforms():
    """训练/验证通用预处理"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, phase="train"):
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

        if batch_idx % 10 == 0:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar(f"Batch/Loss_{phase}", loss.item(), step)

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
    NUM_WORKERS = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")

    # ========== 数据 ==========
    transform = get_transforms()
    dataset = CustomImageDataset(root_dir=DATA_PATH, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"[数据] 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # ========== 加载预训练 ResNet18 ==========
    print("[模型] 加载 ImageNet 预训练 ResNet18...")
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 查看原始 FC 层结构
    print(f"[模型] 原始 FC 层: {model.fc}")

    # 冻结 Backbone（所有卷积层）
    # requires_grad=False 表示不计算梯度，不参与反向传播
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后的 FC 层为 10 分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    model = model.to(device)
    print("[模型] Backbone 已冻结，FC 层已替换为 10 分类")

    # ========== 第一阶段：只训练 FC 层 ==========
    print("\n========== 阶段 1: 冻结 Backbone，只训练 FC 层 ==========")

    # 优化器只传入 FC 层的参数
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir="./runs/day04_transfer_learning")

    best_acc = 0.0
    for epoch in range(3):  # 少量 epoch 快速收敛
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, phase="stage1"
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Stage1 Epoch [{epoch + 1}/3] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        writer.add_scalar("Stage1/Loss/train", train_loss, epoch)
        writer.add_scalar("Stage1/Loss/val", val_loss, epoch)
        writer.add_scalar("Stage1/Accuracy/train", train_acc, epoch)
        writer.add_scalar("Stage1/Accuracy/val", val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_stage1.pth")

    print(f"阶段1 最佳验证准确率: {best_acc:.2f}%")

    # ========== 第二阶段：解冻 Backbone，分层学习率 ==========
    print("\n========== 阶段 2: 解冻 Backbone，分层学习率微调 ==========")

    # 解冻所有层
    for param in model.parameters():
        param.requires_grad = True

    # 分层学习率：Backbone 用小 lr（1e-5），FC 用稍大 lr（1e-3）
    # 这样 Backbone 微调幅度小，不破坏预训练特征；FC 快速适应新任务
    backbone_params = []
    fc_params = []
    for name, param in model.named_parameters():
        if 'fc' in name:
            fc_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': 1e-5},  # Backbone: 微调，幅度极小
        {'params': fc_params, 'lr': 1e-3}  # FC: 稍快调整
    ])

    # 也可以加 scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    for epoch in range(5):  # 更多 epoch 精细调整
        current_lr_backbone = optimizer.param_groups[0]['lr']
        current_lr_fc = optimizer.param_groups[1]['lr']
        print(f"Epoch [{epoch + 1}/5] LR backbone: {current_lr_backbone:.6f}, LR fc: {current_lr_fc:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 3, writer, phase="stage2"
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Stage2 Epoch [{epoch + 1}/5] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        writer.add_scalar("Stage2/Loss/train", train_loss, epoch + 3)
        writer.add_scalar("Stage2/Loss/val", val_loss, epoch + 3)
        writer.add_scalar("Stage2/Accuracy/train", train_acc, epoch + 3)
        writer.add_scalar("Stage2/Accuracy/val", val_acc, epoch + 3)
        writer.add_scalar("Stage2/LR/backbone", current_lr_backbone, epoch + 3)
        writer.add_scalar("Stage2/LR/fc", current_lr_fc, epoch + 3)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_stage2.pth")
            print(f"  -> 最佳模型已保存 (val acc: {best_acc:.2f}%)")

        scheduler.step()

    writer.close()
    print(f"\n✅ 训练完成！最佳验证准确率: {best_acc:.2f}%")
    print("阶段1 权重: best_stage1.pth")
    print("阶段2 权重: best_stage2.pth")
    print("查看曲线: tensorboard --logdir=./runs/day04_transfer_learning")


if __name__ == "__main__":
    main()