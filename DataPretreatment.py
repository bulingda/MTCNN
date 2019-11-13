import os
import traceback
from PIL import Image
import numpy as np
import utils

image_path = r'F:\data\img_celeba'
# 原始数据
box_label_path = r'F:\data\list_bbox_celeba.txt'
# 边框标签
landmark_label = r'F:\data\list_landmarks_celeba.txt'
# 关键点标签
negative_path = r'F:\data\negative'
save_path = r'E:\data\data'

for image_size in [12, 24, 48]:
    # 图片的尺寸是12，24，48的
    positive_img_path = os.path.join(save_path, str(image_size), "positive")
    part_img_path = os.path.join(save_path, str(image_size), "part")
    negative_img_path = os.path.join(save_path, str(image_size), "negative")
    landmark_img_path = os.path.join(save_path, str(image_size), "landmark")
    for paths in [positive_img_path, part_img_path, negative_img_path, landmark_img_path]:
        if not os.path.exists(paths):
            os.makedirs(paths)
    positive_label_path = os.path.join(save_path, str(image_size), "positive.txt")
    part_label_path = os.path.join(save_path, str(image_size), "part.txt")
    negative_label_path = os.path.join(save_path, str(image_size), "negative.txt")
    landmark_label_path = os.path.join(save_path, str(image_size), "landmark.txt")

    positive_count = 0
    part_count = 0
    negative_count = 1
    landmark_count = 0

    try:
        positive_label_file = open(positive_label_path, "w")
        part_label_file = open(part_label_path, "w")
        negative_label_file = open(negative_label_path, "w")
        landmark_label_file = open(landmark_label_path, "w")

        a = -1
        # 曾经的坑
        # for j, line2 in enumerate(open(landmark_label)):
        #     for i, line1 in enumerate(open(box_label_path)):

        with open(box_label_path) as line1s:
            with open(landmark_label) as line2s:
                for (line1, line2) in zip(line1s, line2s):
                    # 踩过的坑
                    # print(line1 + "line1")
                    # print(a)
                    # line1 = line1s.readlines()
                    # print(line1)
                    # line2 = line2s.readline()
                    # print(line2 + "line2")
                    # line1 = enumerate(line1s.readlines())
                    # line2 = enumerate(line2s.readlines())
                    a += 1
                    if a < 2:
                        continue
                    try:
                        strs1 = str(line1).strip().split()
                        # print(strs1)
                        # .strip()是去除头尾的多余空格
                        strs2 = str(line2).strip().split()
                        # # image_name_file = strs[0]
                        # strs = line.strip().split(",")
                        # strs = list(strs)
                        image_file_path = os.path.join(image_path, "{0}".format(strs1[0]))
                        # 根据标签文本里取到的图片名称找到路径，再找到图片和坐标点框出人脸
                        with Image.open(image_file_path) as img:
                            img_w, img_h = img.size
                            img = img.convert('RGB')
                            x1 = float(strs1[1])
                            y1 = float(strs1[2])
                            w = float(strs1[3])
                            h = float(strs1[4])
                            x2 = float(x1 + w)
                            y2 = float(y1 + h)
                            leye_x = float(strs2[1])
                            leye_y = float(strs2[2])
                            reye_x = float(strs2[3])
                            reye_y = float(strs2[4])
                            nose_x = float(strs2[5])
                            nose_y = float(strs2[6])
                            lmouth_x = float(strs2[7])
                            lmouth_y = float(strs2[8])
                            rmouth_x = float(strs2[9])
                            rmouth_y = float(strs2[10])

                            if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                                continue
                            boxes = [[x1, y1, x2, y2]]
                            cx = x1 + w / 2
                            cy = y1 + h / 2
                            # 计算人脸框中心点坐标
                            # 接下来就是创造部分样本的时刻了,不加倍了
                            for _ in range(1):
                                w_ = np.random.randint(-w * 0.2, w * 0.2)
                                h_ = np.random.randint(-h * 0.2, h * 0.2)
                                cx_ = cx + w_
                                cy_ = cy + h_
                                # 现在是让真实框的中心点坐标偏一下
                                # 下面，我们给他随机一个正方形边框
                                side_length = np.random.randint(int(0.8 * min(w, h)), np.ceil(1.25 * (max(w, h))))
                                # int 是向下取整，np.ceil 是向上取整。
                                # 边长取出来就可以计算坐标了
                                x1_ = np.max(cx_ - side_length / 2, 0)
                                # 坐标要是小于零就要跑到图片外面去了，这怎么可以呢
                                y1_ = np.max(cy_ - side_length / 2, 0)
                                x2_ = x1_ + side_length
                                y2_ = y1_ + side_length
                                # 存一下整理好的坐标吧
                                crop_box = np.array([x1_, y1_, x2_, y2_])

                                # 计算一下生成的坐标偏移量，这里跟老师的不一样，但自我感觉我的对啊
                                # 偏移量谁减谁都可以，只要反算的时候对应就好
                                # 左上角相对于左上角，右下角相对于右下角
                                offset_x1 = (x1_ - x1) / side_length
                                offset_y1 = (y1_ - y1) / side_length
                                offset_x2 = (x2_ - x2) / side_length
                                offset_y2 = (y2_ - y2) / side_length
                                # 关键点的偏移量都是相对于左上角的
                                ldmk_off1 = (leye_x - x1) / side_length
                                ldmk_off2 = (leye_y - y1) / side_length
                                ldmk_off3 = (reye_x - x1) / side_length
                                ldmk_off4 = (reye_y - y1) / side_length
                                ldmk_off5 = (nose_x - x1) / side_length
                                ldmk_off6 = (nose_y - y1) / side_length
                                ldmk_off7 = (lmouth_x - x1) / side_length
                                ldmk_off8 = (lmouth_y - y1) / side_length
                                ldmk_off9 = (rmouth_x - x1) / side_length
                                ldmk_off10 = (rmouth_y - y1) / side_length

                                # 位置找好了就可以切了
                                crop = img.crop(crop_box)
                                face_size = crop.resize((image_size, image_size))
                                # resize和reshape要用np.array型才能调用，list类型不能调用这个方法
                                # 做一下iou操作
                                ious = utils.ioufunction(crop_box, np.array(boxes))[0]
                                # 正样本
                                if ious > 0.65:
                                    positive_label_file.write(
                                        "positive/{0}.jpg {1} {2} {3} {4} {5} 0 0 0 0 0 0 0 0 0 0\n".format(positive_count, 1, offset_x1,
                                                                                    offset_y1, offset_x2, offset_y2))
                                    # 写标签到文本中
                                    positive_label_file.flush()
                                    # 刷新文件缓冲池，让文件更新更快
                                    # 将图片保存到对应文件里
                                    face_size.save(os.path.join(positive_img_path, "{0}.jpg".format(positive_count)))
                                    positive_count += 1
                                    for _ in range(2):  # 制作landmark
                                        landmark_label_file.write(
                                            "landmark/{0}.jpg {1} 0 0 0 0 {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n".format(
                                                landmark_count, 3, ldmk_off1, ldmk_off2, ldmk_off3, ldmk_off4, ldmk_off5,
                                                ldmk_off6, ldmk_off7, ldmk_off8, ldmk_off9, ldmk_off10))
                                        # 写标签到文本中
                                        landmark_label_file.flush()
                                        # 刷新文件缓冲池，让文件更新更快
                                        # 将图片保存到对应文件里
                                        face_size.save(os.path.join(landmark_img_path, "{0}.jpg".format(landmark_count)))
                                        landmark_count += 1
                                    # 部分样本
                                if ious > 0.4:
                                    part_label_file.write(
                                        "part/{0}.jpg {1} {2} {3} {4} {5} 0 0 0 0 0 0 0 0 0 0\n".format(part_count, 2, offset_x1, offset_y1,
                                                                                    offset_x2, offset_y2))
                                    # 写标签到文本中
                                    part_label_file.flush()
                                    # 刷新文件缓冲池，让文件更新更快
                                    # 将图片保存到对应文件里
                                    face_size.save(os.path.join(part_img_path, "{0}.jpg".format(part_count)))
                                    part_count += 1

                    except Exception as e:
                        traceback.print_exc()
        # 制作负样本
        negative_list = os.listdir(negative_path)
        for i, nimg_name in enumerate(negative_list):
            negative_path_ = os.path.join(negative_path, nimg_name)
            with Image.open(negative_path_) as nimg:
                nimg = nimg.convert('RGB')
                nimg_w, nimg_h = nimg.size
                if min(nimg_w, nimg_h) < 48:
                    # 这里会删除一部分图片
                    continue
                for _ in range(150):
                    fx = np.random.randint(0.5, nimg_w * 1.)
                    fy = fx
                    crop_box = np.array([0, 0, fx, fy])
                    if min(nimg_w, nimg_h) < 48:
                        continue
                    crop = nimg.crop(crop_box)
                    face_size = crop.resize((image_size, image_size))
                    negative_label_file.write("negative/{0}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count))
                    negative_label_file.flush()
                    face_size.save(os.path.join(negative_img_path, "{0}.jpg".format(negative_count)))
                    negative_count += 1
    finally:
        positive_label_file.close()
        negative_label_file.close()
        part_label_file.close()
        landmark_label_file.close()








