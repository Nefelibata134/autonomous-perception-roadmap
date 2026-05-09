"""
generate_detection_data.py
生成模拟目标检测数据集（YOLO 格式）
每张图包含 2-5 个随机彩色方块，模拟不同类别的目标
"""
import os
import random
from PIL import Image, ImageDraw

# 10 个类别的颜色（和之前一致，方便理解）
CLASS_COLORS = {
    0: (135, 206, 235),  # airplane - 天蓝
    1: (255, 99, 71),  # automobile - 番茄红
    2: (255, 215, 0),  # bird - 金黄
    3: (147, 112, 219),  # cat - 紫色
    4: (34, 139, 34),  # deer - 森林绿
    5: (210, 105, 30),  # dog - 巧克力色
    6: (50, 205, 50),  # frog - 酸橙绿
    7: (139, 69, 19),  # horse - 马鞍棕
    8: (70, 130, 180),  # ship - 钢蓝
    9: (255, 140, 0),  # truck - 深橙
}

NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def generate_image(img_size=640, min_objects=2, max_objects=5):
    """
    生成一张检测图片和对应的 YOLO 标签
    返回: (PIL Image, list of (class_id, x_center, y_center, width, height))
    """
    img = Image.new('RGB', (img_size, img_size), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    labels = []

    num_objects = random.randint(min_objects, max_objects)
    used_regions = []  # 记录已放置的区域，避免重叠

    for _ in range(num_objects):
        cls_id = random.randint(0, 9)
        color = CLASS_COLORS[cls_id]

        # 随机生成方块大小（占图片的 10%~30%）
        w = random.randint(int(img_size * 0.1), int(img_size * 0.3))
        h = random.randint(int(img_size * 0.1), int(img_size * 0.3))

        # 随机位置，确保不超出边界
        x1 = random.randint(0, img_size - w)
        y1 = random.randint(0, img_size - h)
        x2 = x1 + w
        y2 = y1 + h

        # 简单防重叠检查（如果重叠太严重就跳过）
        overlap = False
        for (rx1, ry1, rx2, ry2) in used_regions:
            if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2):
                overlap = True
                break

        if overlap:
            continue

        used_regions.append((x1, y1, x2, y2))

        # 画方块
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)

        # 计算 YOLO 格式（归一化到 0-1）
        cx = (x1 + x2) / 2.0 / img_size
        cy = (y1 + y2) / 2.0 / img_size
        nw = w / img_size
        nh = h / img_size

        labels.append((cls_id, cx, cy, nw, nh))

    return img, labels


def generate_dataset(root="./data/detection", num_train=800, num_val=200, img_size=640):
    """生成完整的数据集"""
    splits = {
        'train': num_train,
        'val': num_val
    }

    for split, num in splits.items():
        img_dir = os.path.join(root, 'images', split)
        lbl_dir = os.path.join(root, 'labels', split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(num):
            img, labels = generate_image(img_size)

            # 保存图片
            img_path = os.path.join(img_dir, f"{i:05d}.jpg")
            img.save(img_path)

            # 保存 YOLO 标签（每行: class x_center y_center width height）
            lbl_path = os.path.join(lbl_dir, f"{i:05d}.txt")
            with open(lbl_path, 'w') as f:
                for cls_id, cx, cy, nw, nh in labels:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        print(f"  {split}: {num} 张图片已生成")

    print(f"\n✅ 检测数据集生成完成！路径: {root}")


if __name__ == "__main__":
    generate_dataset()