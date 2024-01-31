import torch
from tqdm import tqdm
from model import UnetModel
from preload import train_loader, test_loader
import torch.optim as optim
import torch.nn as nn

# 模型定义
model = UnetModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
process_bar = tqdm(total=100, ncols=110, desc=f"Training process", position=0)
num_epoch = 10  # 训练次数

# 检查当前设备是否为 GPU
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Code is running on GPU: {device_name}")
else:
    print("Code is running on CPU.")

for epochs in range(num_epoch):
    model.train()
    loss = []
    # 读取数据集
    for idx, batch in enumerate(train_loader):
        tqdm_grow_scale = 100 / num_epoch / len(train_loader)
        process_bar.update(round(tqdm_grow_scale, 2))

        input_image = batch['image_path']  # 尺寸是四维张量 [batch_size, channel, width, height]
        label = batch['label']  # 展平为 1 维, 这里应该输出一个批次 (batch) 的事实标签
        label = label.float()

        optimizer.zero_grad()
        output = model(input_image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # print(f'training case {idx + 1}')

    print(f'epoch {epochs + 1}, loss:{loss.item()}')

# 检验训练效果
total = 0
correct = 0

# model.eval()
# with torch.no_grad():
#     for data, label in test_loader:
#         output = model(data)
#         _, predicted = torch.max(output, 1)  # 返回 (最大张量, 索引)； 1 表示维度； 整体表示最有可能的一个，这个适用于多元分类
#         total += label.size(0)  # 总的 case 数量
#         true_case = 0
#         for _ in label:
#             if predicted == label:
#                 true_case += 1
#         correct += true_case
# todo: 把这个检测搞定

# accuracy = correct / total
# print(f"the accuracy for the model is {accuracy * 100:.2f}%")  # 输出正确率

# trained_model = {"model": model, "accuracy": accuracy}

torch.save(model.state_dict(),"build/model.pth")