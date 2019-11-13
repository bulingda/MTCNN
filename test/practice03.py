landmark_label_path = r'E:\data\list_landmarks_celeba.txt'
# landmark_label_file = open(landmark_label_path, 'rb')
# for j, line2 in enumerate(open(landmark_label_path).readlines()):
#     if j < 2:
#         continue
#     strs = str(line2).split()
#     print(strs)
# 这个操作结束后不关闭文件会有一些；灵异事件发生，用with打开代替
with open(landmark_label_path) as f:
    for i, line in enumerate(f.readlines()):
        print(line.split())