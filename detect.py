from model import UnetModel
from preload import train_loader
import torch.optim as optim
import torch.nn as nn

# 模型定义
model = UnetModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
num_epoch = 10  # 训练次数
for epochs in range(num_epoch):
    model.train()
    # 读取数据集
    for idx, batch in enumerate(train_loader):
        input_image = batch['image_path']  # 尺寸是四维张量 [batch_size, channel, width, height]
        label = batch['label']  # 展平为 1 维, 这里应该输出一个批次 (batch) 的事实标签
        label = label.float()

        optimizer.zero_grad()
        output = model(input_image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        print(f'training case {idx+1}')

    print(f'epoch {epochs}, loss:{loss.item()}')
