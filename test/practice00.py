import os
from PIL import Image

before_path = r'F:\data\imgs'
image = os.listdir(before_path)
after_path = r'F:\data\img'
label_path = r'F:\data\label.txt'
part_label_file = open(label_path, "w")

for i, line in enumerate(open(label_path)):
    strs = str(line).strip()
    img_path = os.path.join(before_path, image[i])
    img = Image.open(img_path)
    img.save(os.path.join(after_path, strs[i] + '.jpg'))
    # 太复杂了，不改了名字了