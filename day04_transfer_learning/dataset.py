"""
dataset.py
手撕 CustomImageDataset: 理解 Pytorch Dataset 的核心机制
"""
import os
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    """
    通用图像数据集类，支持任意文件夹结构的图片分类数据集
    
    期望的文件夹结构：
        root/
           airplane/
               00000.png
               00001.png
            automobile/
                ...     
    """

    def __init__(self,root_dir,transform=None):
        """
        Args
            root_dir (str): 数据集根目录路径
            transform (callable,optional):
        """
        self.root_dir = root_dir
        self.transform = transform

        # 1. 扫描根目录，获取所有子文件夹（即类别名），排序保证顺序稳定
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir,d))])
        # 2. 建立类别名到数字标签的映射字典
        # 例如: {'airplane': 0, 'automobile': 1, ...}
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

        # 3. 遍历每个类别文件夹，收集所有图片的 (路径, 标签)
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir,cls_name)

            # 遍历该类别下的所有文件
            for img_name in sorted(os.listdir(cls_dir)):
                # 只处理常见图片格式
                if img_name.lower().endswith(('.png','.jpg','.jpeg','.bmp','.gif')):
                    img_path = os.path.join(cls_dir,img_name)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((img_path,label))
        # 打印数据集信息
        print(f"[Dataset] 加载完成: {len(self.samples)} 张图片， {len(self.classes)}个类别")
        print(f"[Dataset] 类别映射：{self.class_to_idx}")

    def __len__(self):
        """
        返回数据集的总样本数
        DataLoader 需要知道一个 epoch 有多少样本
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引 idx 返回一个样本 (image, label)
        DataLoader 会通过索引调用这个方法来取数据

        Args:
            idx (int): 样本索引

        Returns:
            image: 经过 transform 后的图像 Tensor
            label (int): 类别标签
        """
        # 取出图片路径和对应标签
        img_path, label = self.samples[idx]

        # 用 PIL 打开图片，并转为 RGB（兼容透明通道的 PNG）
        image = Image.open(img_path).convert('RGB')

        # 应用预处理（如 Resize, ToTensor, Normalize 等）
        if self.transform:
            image = self.transform(image)

        return image, label

