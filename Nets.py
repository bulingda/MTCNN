import torch.nn as nn
import torch


class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.per_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # 10*10
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 5*5
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # 1*1
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.PReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.per_layer(x)
        cond = self.conv1(x)
        offset = self.conv2(x)
        ldmk_off = self.conv3(x)
        return cond, offset, ldmk_off


class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # 22
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 10
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # 8
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 3
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # 2
            nn.PReLU()
        )
        self.linear1 = nn.Linear(64 * 2 * 2, 128)
        self.relu = nn.PReLU()
        self.linear2 = nn.Linear(128, 1)
        self.linear3 = nn.Linear(128, 4)
        self.linear4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        # 讲多维度的图片转换成一维数据，-1是形状自动给，常放在卷积网络和线性网络中间
        x = self.linear1(x)
        x = self.relu(x)
        cond = torch.sigmoid(self.linear2(x))
        offset = self.linear3(x)
        ldmk_off = self.linear4(x)
        return cond, offset, ldmk_off


class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # 46
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 22
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # 20
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 9
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 7
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 3
            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # 2
            nn.PReLU()
        )
        self.linear1 = nn.Linear(128 * 2 * 2, 256)
        self.relu = nn.PReLU()
        self.linear2 = nn.Linear(256, 1)
        self.linear3 = nn.Linear(256, 4)
        self.linear4 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        cond = torch.sigmoid(self.linear2(x))
        offset = self.linear3(x)
        ldmk_off = self.linear4(x)
        return cond, offset, ldmk_off







