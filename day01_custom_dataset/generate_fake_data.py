"""
generate_fake_data.py
生成模拟的 CIFAR-10 风格数据集（无需下载）
每张图片是随机颜色块 + 类别文字，用于测试 Dataset/DataLoader
"""
import os
import random
from PIL import Image, ImageDraw, ImageFont


def generate_fake_dataset(root="./data/cifar10_train", num_per_class=100, img_size=32):
    """
    生成假数据集，结构和 CIFAR-10 一致：
    data/cifar10_train/airplane/00000.png
    """
    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # 颜色映射（每个类别固定主色调，方便肉眼区分）
    color_map = {
        'airplane': (135, 206, 235),  # 天蓝
        'automobile': (255, 99, 71),  # 番茄红
        'bird': (255, 215, 0),  # 金黄
        'cat': (147, 112, 219),  # 紫色
        'deer': (34, 139, 34),  # 森林绿
        'dog': (210, 105, 30),  # 巧克力色
        'frog': (50, 205, 50),  # 酸橙绿
        'horse': (139, 69, 19),  # 马鞍棕
        'ship': (70, 130, 180),  # 钢蓝
        'truck': (255, 140, 0),  # 深橙
    }

    os.makedirs(root, exist_ok=True)
    total = 0

    for cls_name in classes:
        cls_dir = os.path.join(root, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        base_color = color_map[cls_name]

        for i in range(num_per_class):
            # 生成随机扰动的颜色
            r = min(255, max(0, base_color[0] + random.randint(-30, 30)))
            g = min(255, max(0, base_color[1] + random.randint(-30, 30)))
            b = min(255, max(0, base_color[2] + random.randint(-30, 30)))

            # 创建图片
            img = Image.new('RGB', (img_size, img_size), (r, g, b))
            draw = ImageDraw.Draw(img)

            # 尝试写文字（如果系统有字体），没有就算了
            try:
                # 尝试加载系统字体
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
            except:
                font = ImageFont.load_default()

            # 在图片上写类别名（作为"特征"）
            text = cls_name[:4]
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = (img_size - text_w) // 2
            y = (img_size - text_h) // 2
            draw.text((x, y), text, fill=(255, 255, 255), font=font)

            # 保存
            save_path = os.path.join(cls_dir, f"{i:05d}.png")
            img.save(save_path)
            total += 1

        print(f"  {cls_name}: {num_per_class} 张")

    print(f"\n✅ 完成！共生成 {total} 张图片到 {root}")
    print("（这些是模拟数据，仅用于测试 Dataset/DataLoader 流程）")


if __name__ == "__main__":
    generate_fake_dataset(num_per_class=100)