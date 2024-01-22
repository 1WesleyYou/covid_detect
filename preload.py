import os
from enum import Enum
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image
from PIL import Image


class Class(Enum):
    NON_COVID = 0
    COVID = 1


# 数据集路径 (指向 COVID 父文件夹)
data_root_path = './dataset'

transform = transforms.Compose([
    transforms.Resize([256, 256]),  # 调整图像大小
    transforms.Grayscale(),  # 变成灰度图，保险起见
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化，将像素的颜色值映射到 [-1, 1] 区间上
])


class CTScanDataset(Dataset):
    def __init__(self, transform_model=None):
        self.transform = transform_model
        self.data = self.load_data()

    def load_data(self):
        # 录入数据首先要清空内容
        data = []
        # COVID 部分数据集

        for filename in os.listdir('dataset/COVID'):
            # to get the id of the data figure
            # left = 6  # the position for the left part
            # right = filename.find(')')
            # index = filename[left + 1:right]
            data.append({'image_path': os.path.join(f'dataset/COVID/{filename}'), 'label': 1})
        for filename in os.listdir('dataset/non-COVID'):
            data.append({'image_path': os.path.join(f'dataset/non-COVID/{filename}'), 'label': 0})
        return data

    def __len__(self):
        # 获取数据集组数
        return len(self.data)

    # 重新定义 getitem, 这是 Dataset 的一个类方法，能从数据集中获得单个样本
    # 思路是通过访问路径获得图像张量和标签，这样就不需要一次性加载所有数据集了
    def __getitem__(self, idx):
        img_path = self.data[idx]['image_path']
        label = torch.tensor([self.data[idx]['label']], dtype=torch.long)
        image = Image.open(img_path).convert("L")  # 使用torchvision读取图像

        if self.transform:
            image = self.transform(image)

        return {'image_path': image, 'label': label}


# 初始化数据集
dataset = CTScanDataset(transform)

# 划分数据集：我们将 80% 的数据用于训练模型，10% 用于训练时测试评估，10% 用于最终评估
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
