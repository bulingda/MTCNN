import numpy as np


def ioufunction(crop_box, boxes, isMin = False):
    # print("进入iou了")
    # 这样处理可以同时解决一组图片的面积
    # print(crop_box[0], "crop_box[0]")
    # print(boxes[:, 0], "boxes[:, 0]")
    # print(list(boxes[:, 0][0]), "list(boxes[:, 0][0]")
    # print(np.array(list(boxes[:, 2][0])), "np.array(list(boxes[:, 2][0]))")
    box_area = (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])  # 创的框的面积
    img_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # 原框的面积
    # 图片交集的面积
    x1 = np.maximum(boxes[:, 0], crop_box[0])
    y1 = np.maximum(boxes[:, 1], crop_box[1])
    x2 = np.minimum(boxes[:, 2], crop_box[2])
    y2 = np.minimum(boxes[:, 3], crop_box[3])
    sj = np.maximum(x2-x1, 0) * np.maximum(y2-y1, 0)
    # print(sj, "图片交集的面积")
    # 图片并集的面积
    sb = box_area+img_area-sj
    # print(sb, "图片并集的面积")
    if isMin is False:
        proportion = np.true_divide(sj, np.minimum(img_area, box_area))
        # true_divide函数与数学中的除法定义更为接近，即返回除法的浮点数结果而不作截断
        # print(proportion, "除数1")
    else:
        proportion = np.true_divide(sj, sb)
        # print("除数2")
    return proportion


# nms还有问题暂时还没调出来
def nmsfunction(boxes, thresh=0.3, isMin=False):
    # print("进入nms了")
    # print(boxes, "nmsboxes")
    if boxes.shape[0] == 0:
        return np.array([])
    # 没有框的时候就返回空
    _boxes = boxes[(-boxes[:, 4]).argsort()]
    # print(_boxes, "二维的")
    # 降序排序，排序的这个东西是最后网络输出出来的一堆框,是二维的n个4+1的值，此时格式是[x1,y1,x2,y2,c]
    c_boxes = []
    while _boxes.shape[0] > 1:

        a_boxes = _boxes[0]
        b_boxes = _boxes[1:]
        # 将置信度最大的和其他的比较
        c_boxes.append(a_boxes)
        # 剩下的保存下来,append没有返回值所以不需要接收
        # print(ioufunction(a_boxes, b_boxes, isMin), "ioufunction(a_boxes, b_boxes, isMin)")
        # print(ioufunction(a_boxes, b_boxes, isMin) < thresh, "ioufunction(a_boxes, b_boxes, isMin) < thresh")
        d_boxes = np.where(ioufunction(a_boxes, b_boxes, isMin) < thresh)
        _boxes = b_boxes[d_boxes]
    # print("循环出来了吗?")

    if _boxes.shape[0] > 0:
        c_boxes.append(_boxes[0])
    # print("结束nms了嘛?")
    return np.array(c_boxes)
    # 将list类型转为np中数组类型


def convert_to_squre(bboxes):
    box = bboxes.copy()
    if bboxes.shape[0] == 0:
        return np.array([])
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    # print(w, "w")
    # print(h, "h")
    max_side = np.maximum(w, h)

    box[:, 0] = bboxes[:, 0] + w * 0.5 - max_side * 0.5
    box[:, 1] = bboxes[:, 1] + w * 0.5 - max_side * 0.5
    box[:, 2] = box[:, 0] + max_side
    box[:, 3] = box[:, 1] + max_side
    # 这个地方是取最小的正方形的框，把图片放到正中心，不懂的话手画一下就懂了
    return box

# 测试使用


# a = np.array([1, 2, 3, 4, 1])
# b = np.array([[3, 4, 5, 6, 2], [2, 3, 3, 5, 3], [3, 2, 5, 4, 3]])
# # iou = ioufunction(a, b)
# nm = nmsfunction(b)
# # print(iou)
# print(nm)