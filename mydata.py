from torch.utils.data import Dataset
import os
import torch
import numpy as np
from PIL import Image


class Datasets (Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "landmark.txt")).readlines())

    def __len__(self):
        return len(self.dataset)

# 最终返回数据和标签
    def __getitem__(self, index):
        strs1 = self.dataset[index].split()
        # print(strs)
        img_path = os.path.join(self.path, "{0}".format(strs1[0]))
        c = torch.tensor(int(strs1[1])).float()
        offset = torch.tensor(np.array([float(strs1[2]), float(strs1[3]), float(strs1[4]), float(strs1[5])])).float()
        img = torch.Tensor(np.array(Image.open(img_path).convert('RGB'))/255.-0.5)
        img = img.permute(2, 0, 1)
        ldmk_off = torch.tensor(np.array([float(strs1[6]), float(strs1[7]), float(strs1[8]), float(strs1[9]), float(strs1[10]), float(strs1[11]), float(strs1[12]), float(strs1[13]), float(strs1[14]), float(strs1[15])])).float()
        # 换轴
        # print(img.shape)
        # print(c)
        # print(offset.shape)
        return img, c, offset, ldmk_off


if __name__ == '__main__':
    data = Datasets(r'F:\data\data\12')
    print(data[0])



