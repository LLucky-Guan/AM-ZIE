import os
import torch
import torch.utils.data
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms


class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        # 加载低光照图像数据集
        train_dataset = LowLightDataset(
            self.config.train_dir,
            patch_size=self.config.patch_size,
            train=True,
        )
        val_dataset = LowLightDataset(
            self.config.val_dir,
            patch_size=self.config.patch_size,
            train=False,
        )

        # 创建 DataLoader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader


class LowLightDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, train):
        """
        初始化函数
        :param dir: 低光照图像的路径
        :param patch_size: 图像调整大小的尺寸
        :param train: 是否为训练集
        """
        super(LowLightDataset, self).__init__()

        self.img_dir = dir
        self.patch_size = patch_size
        self.train = train

        # 检查路径是否存在
        if not os.path.exists(self.img_dir):
            raise ValueError(f"低光照图像目录 {self.img_dir} 不存在。")

        # 获取所有图像文件
        self.img_names = sorted(
            [
                os.path.join(self.img_dir, img)
                for img in os.listdir(self.img_dir)
                if img.endswith(("png", "jpg", "jpeg", "bmp"))
            ]
        )

        if train:
            self.transforms = transforms.Compose(
                [  
                    # transforms.RandomCrop(self.patch_size),  # 随机裁剪
                    transforms.Resize((self.patch_size, self.patch_size)),  # 调整图像大小
                    transforms.RandomHorizontalFlip(),  # 随机水平翻转
                    # transforms.ColorJitter(
                    #     brightness=0,   # 亮度变化范围（0.8~1.2倍）
                    #     contrast=1,     # 对比度变化范围（0.7~1.3倍）
                    #     saturation=1    # 饱和度变化范围（0.7~1.3倍）
                    # ),  
                    transforms.ToTensor(),  # 将PIL Image或numpy数组转换为Tensor
                ]
            )

        else:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
    def get_image(self, index):

        img_name = self.img_names[index]
        img_id = os.path.basename(img_name)  # 获取文件名作为图像 ID

        # 加载低光照图像并应用变换
        img = Image.open(img_name)
        img = self.transforms(img)
        
        return img, img_id

    def __getitem__(self, index):
        """
        返回数据集中的一个样本
        """
        return self.get_image(index)

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.img_names)
