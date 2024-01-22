import torch.nn as nn


class UnetModel(nn.Module):
    def __init__(self):
        super(UnetModel, self).__init__()
        # 定义编码器（下采样部分，U的左半边）
        self.encoder = nn.Sequential(
            # CT 是灰度图, 输入维度是 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 定义中间连接
        self.middle = nn.Sequential(
            # padding 就是在图片周围加上一圈 0 防止边界特征被弱化
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 定义解码器（上采样部分）输出尺寸为 64,64 的子图像灰度图
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)  # 解卷积层, 获得分割之后的子图像
        )
        # 添加全连接层
        self.fc1 = nn.Linear(128 * 128, 32)  # 32 * 32是解码器输出的大小，根据实际情况调整
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        # 如果不展开,这里的 fc 会压缩一个维度,整个结果就是看不懂
        x = x.view(-1, 128 * 128)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
