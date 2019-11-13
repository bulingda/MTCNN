# 原始数据
box_label_path = r'E:\data\list_bbox_celeba.txt'
# 边框标签
landmark_label = r'E:\data\list_landmarks_celeba.txt'
with open(box_label_path) as line1s:
    with open(landmark_label) as line2s:
        for (line1, line2) in zip(list(line1s), list(line2s)):
            print(line1)
            print(line2)