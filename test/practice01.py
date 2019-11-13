import os
from PIL import Image
# image_path = r'F:\data\imgs'
# label_path = r'F:\data\label.txt'
# for i, line in enumerate(open(label_path)):
#     strs = str(line).split(",")
#     image_file_path = os.path.join(image_path, "{0}.jpg".format(strs[0]))
#     print(image_file_path)
# from PIL import Image
# imagepath = '(1).png'
# with Image.open(imagepath) as img:
#     crop_box = [1, 10, 4000, 7000]
#     crop = img.crop(crop_box)
#     crop.show()
import numpy as np
# crop_box = [1, 1, 2, 2]
# crop_boxs = np.array([1, 1, 2, 2])
# box = [[1, 1, 2, 2]]
# print(crop_box)
# print(crop_boxs)
# print(box)
# c = [[1, 3], [2, 4]]
#
# print(c)
# print(np.stack(c))
# print(np.array(c))
# negative_path = r'F:\data\negative'
# negative_count = 1
#
# negative_path_ = os.path.join(negative_path, "pic{0}.jpg".format(negative_count))
# negative_list = os.listdir(negative_path)
# for i, line in enumerate(negative_list):
#     for _ in range(10):
#         print("negative/{0}.jpg 0 0 0 0 0\n".format(negative_count))
#         negative_count += 1
#         print(negative_count)
a = [1, 3, 345, 3]
b = [34, 56, 235, 72, 2]
print(a)
print(b)
np.array(a)
np.array(b)
c = min(a, b)
print(c)
d = np.min(1, 2)
print(d)



