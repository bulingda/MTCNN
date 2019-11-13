import torch.nn as nn
import torch
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, k=3):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class PNet(nn.Module):
    cfg = [10, 16, 32]

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]  # isinstance为内联函数，判断一个对象是否是一个已知的类型。
            k = 3 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, k))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        out = self.layers(x)
        out = F.max_pool2d(out, 2)
        cond = self.conv1(out)
        offset = self.conv2(out)
        ldmk_off = self.conv3(out)
        return cond, offset, ldmk_off


class RNet(nn.Module):
    cfg = [28, 48, (64, 2)]

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(28)
        self.layers = self._make_layers(in_planes=28)
        self.linear1 = nn.Linear(64 * 2 * 2, 128)
        self.relu = nn.PReLU()
        self.linear2 = nn.Linear(128, 1)
        self.linear3 = nn.Linear(128, 4)
        self.linear4 = nn.Linear(128, 10)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]  # isinstance为内联函数，判断一个对象是否是一个已知的类型。
            k = 3 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, k))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.layers(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        # 讲多维度的图片转换成一维数据，-1是形状自动给，常放在卷积网络和线性网络中间
        x = self.linear1(x)
        x = self.relu(x)
        cond = torch.sigmoid(self.linear2(x))
        offset = self.linear3(x)
        ldmk_off = self.linear4(x)
        return cond, offset, ldmk_off


class ONet(nn.Module):
    cfg = [32, 64, 64, (128, 2)]

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear1 = nn.Linear(128 * 2 * 2, 256)
        self.relu = nn.PReLU()
        self.linear2 = nn.Linear(256, 1)
        self.linear3 = nn.Linear(256, 4)
        self.linear4 = nn.Linear(256, 10)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]  # isinstance为内联函数，判断一个对象是否是一个已知的类型。
            k = 3 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, k))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.layers(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        cond = torch.sigmoid(self.linear2(x))
        offset = self.linear3(x)
        ldmk_off = self.linear4(x)
        return cond, offset, ldmk_off







