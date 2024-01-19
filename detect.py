from model import UnetModel
from preload import Class, dataset
import torch
import torch.optim as optim
import torch.nn as nn

# 模型定义
model = UnetModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epoch
