from torchvision import transforms
import torch
import time
import numpy as np
import utils
import os


class Detector:
    def __init__(self, pnet_param=r"./param/pnet1.pth", rnet_param=r"./param/rnet1.pth", onet_param=r"./param/onet1.pth", isCuda=True):
        if os.path.exists(pnet_param):
            self.pnet = torch.load(pnet_param)  # , map_location='cpu'用cpu训练的时候要加这句话
            self.pnet.eval()
        if os.path.exists(rnet_param):
            self.rnet = torch.load(rnet_param)
            self.rnet.eval()
        if os.path.exists(onet_param):
            self.onet = torch.load(onet_param)
            self.onet.eval()
        # 保存的时候是把整个网络都保存下来的
        self.isCuda = isCuda
        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()
        # self.image_transform = transforms.Compose([transforms.ToTensor()])
        self.image_transform = lambda img: self.totensor(img)
        # 会将h，w，c的图片变成tensor的并且形状变为从c，h，w

    def totensor(self, img):
        img_data = torch.Tensor(np.array(img.convert('RGB'))/255.-0.5)
        return img_data.permute(2, 0, 1)
    """ *****五星级错误关键点
    训练的时候在mydata中将数据进行了归一化，使用的时候也要同样操作
    """

    def detect(self, image):
        start_time = time.time()
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        ptime = end_time-start_time
        # return pnet_boxes

        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        rtime = end_time - start_time
        # print("r网络结束了")

        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        otime = end_time - start_time
        # print("o网络结束了")

        sum_time = ptime + rtime + otime
        print("total time:{0} pnet time:{1} rnet time:{2} onet time:{3}".format(sum_time, ptime, rtime, otime))
        return onet_boxes
        # return pnet_boxes
        # return rnet_boxes

    def __pnet_detect(self, image):
        # print(image)
        # print(image.size)
        boxes = []
        # 图片的格式是:[x1,y1,x2,y2,c]
        img = image
        w, h = img.size
        minedge = min(w, h)
        scale = 1
        while minedge > 12:
            # print("p网络正在处理")
            img_data = self.image_transform(img)
            # print(img_data.shape)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)
            # print(img_data.shape)
            # 在第0个位置升一个维度，unsqueeze_是unsqueeze的就地版本，之前图片的形状是c,h，w，现在要加上n
            _cfd, _off, _ldmk = self.pnet(img_data)
            # print(_cfd.shape)
            # print(_off.shape)
            # print("进网络了")
            cfd, off, ldmk = _cfd[0][0].cpu().data.numpy(), _off[0].cpu().data.numpy(), _ldmk[0].cpu().data.numpy()
            # print(cfd.shape)
            # print(off.shape)
            # print(cfd)
            # print(ldmk[0].shape)
            """" 
            pnet 输出的图片结构是，N C H W 四维的，使用的时候只输入一张图片=>n=1，输出的是5个通道（1置+4偏）—>c=1，4
            取置信度的时候就要下到三维里去取三维里的第一个二维（一个平面，一组置信度），
            取偏移量的时候要取三维的，因 为一个平面放的是一组坐标值，所以取的是多组x1,y1,x2,y2.
            """
            mask = np.stack(np.nonzero(cfd > 0.4), axis=1)
            # print(mask.shape)
            # print(np.stack(mask, axis=1))
            for idx in mask:
                # print(self.box(idx, off, cfd[idx[0], idx[1]], scale))
                boxes.append(self.box(idx, off, cfd[idx[0], idx[1]], ldmk, scale))
                # print(boxes)
                # print("进盒子了")
                '''
                mask用nonzero取出来的是二维数组，一个坐标是两个值，在二维数组中要用两个数来索引到一个值。
                idx[0]取的是一系列的值。
                要先找到坐标的两个数才能找到这两个数对应坐标的置信度cfd
                '''
            # print(np.array(boxes), np.array(boxes).shape, "box->boxes.shape")
            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)
            img = img.resize((_w, _h))
            minedge = min(_w, _h)
            # print("进入nms")
            # print(utils.nmsfunction(np.array(boxes), 0.3).shape)
        return utils.nmsfunction(np.array(boxes), 0.45)

    def __rnet_detect(self, image, pnetboxes):
        # print(pnetboxes.shape)
        pboxes = utils.convert_to_squre(pnetboxes)
        # print(pboxes)
        # print(pboxes.shape)
        img_data = []
        # print(pboxes, pboxes.shape, "pboxes")
        # print(pboxes[:], "pboxes[:]")
        for _box in pboxes:
            x1 = int(_box[0])
            y1 = int(_box[1])
            x2 = int(_box[2])
            y2 = int(_box[3])

            # print(x1, "x1")
            pcrop_img = image.crop((x1, y1, x2, y2))
            img = pcrop_img.resize((24, 24))
            # print(img.size)

            img = self.image_transform(img)
            img_data.append(img)
        # print(type(img_data))
        # print(np.array(img_data).size)
        imgs = torch.stack(img_data)
        # print(imgs.shape)
        if self.isCuda:
            imgs = imgs.cuda()
        _cfd, _off, _ldmk = self.rnet(imgs)
        # print(_cfd.shape)
        # print(_off.shape)
        cfd = _cfd.data.cpu().numpy()[:, 0]
        off = _off.data.cpu().numpy()
        ldmk = _ldmk.data.cpu().numpy()
        # mask = np.stack(np.where(cfd > 0.6))[:, 0]
        # print(mask)
        # mask = np.stack(np.nonzero(cfd > 0.99), axis=1)
        mask = np.nonzero(cfd > 0.7)[0]
        # print(mask)
        # print(mask.shape)
        # print(np.stack(np.nonzero(cfd > 0.6), axis=1))
        # cfd = cfd[mask]
        boxes = []
        for idx in mask:
            boxes.append(self.box_(pboxes, idx, off, cfd[idx], ldmk))
        # print(utils.nmsfunction(np.array(boxes), 0.7).shape)
        return utils.nmsfunction(np.array(boxes), 0.5)

    def __onet_detect(self, image, rnetboxes):
        rboxes = utils.convert_to_squre(rnetboxes)
        img_data = []
        for _box in rboxes:
            x1 = int(_box[0])
            y1 = int(_box[1])
            x2 = int(_box[2])
            y2 = int(_box[3])
            rcrop = image.crop((x1, y1, x2, y2))
            img = rcrop.resize((48, 48))
            img = self.image_transform(img)
            img_data.append(img)
        imgs = torch.stack(img_data)
        if self.isCuda:
            imgs = imgs.cuda()
        _cfd, _off, _ldmk = self.onet(imgs)
        # print(_cfd)
        cfd, off, ldmk = _cfd.data.cpu().numpy()[:, 0], _off.data.cpu().numpy(), _ldmk.data.cpu().numpy()
        # print(cfd)
        mask = np.nonzero(cfd > 0.999)[0]
        boxes = []
        for idx in mask:
            boxes.append(self.box_(rboxes, idx, off, cfd[idx], ldmk))
        return utils.nmsfunction(np.array(boxes), 0.2, isMin=True)

    def box(self, index, offset, confidence, landmark, scale, stride=2, side_len=12):
        _x1 = (index[1] * stride) / scale
        _y1 = (index[0] * stride) / scale
        _x2 = (index[1] * stride + side_len) / scale
        _y2 = (index[0] * stride + side_len) / scale
        # 这里的坐标索引和常见的索引是反的，这里的索引是二维的、
        _w = _x2 - _x1
        _h = _y2 - _y1
        # print(offset.shape, "offset.shape")
        # print(landmark.shape, "landmark.shape")
        _offset = offset[:, index[0], index[1]]
        _landmark = landmark[:, index[0], index[1]]
        """
        偏移量是三维的，第一个维度全都取，第二个维度取置信度大于0.6的坐标的x，第三个维度就是y了，
        这样取出的就是置信度大于0.6的坐标值了
        """
        x1 = _x1 - _w * _offset[0]
        y1 = _y1 - _h * _offset[1]
        x2 = _x2 - _w * _offset[2]
        y2 = _y2 - _h * _offset[3]
        leye_x = x1 + _w * _landmark[0]
        leye_y = y1 + _h * _landmark[1]
        reye_x = x1 + _w * _landmark[2]
        reye_y = y1 + _h * _landmark[3]
        nose_x = x1 + _w * _landmark[4]
        nose_y = y1 + _h * _landmark[5]
        lmouth_x = x1 + _w * _landmark[6]
        lmouth_y = y1 + _h * _landmark[7]
        rmouth_x = x1 + _w * _landmark[8]
        rmouth_y = y1 + _h * _landmark[9]

        return [x1, y1, x2, y2, confidence, leye_x, leye_y, reye_x, reye_y, nose_x, nose_y, lmouth_x, lmouth_y, rmouth_x, rmouth_y]

    def box_(self, boxes, index, offset, cfd, landmark):
        box = boxes[index]
        _x1 = int(box[0])
        _y1 = int(box[1])
        _x2 = int(box[2])
        _y2 = int(box[3])
        ow = _x2 - _x1
        oh = _y2 - _y1
        x1 = _x1 - ow * offset[index][0]
        y1 = _y1 - oh * offset[index][1]
        x2 = _x2 - ow * offset[index][2]
        y2 = _y2 - oh * offset[index][3]
        leye_x = x1 + ow * landmark[index][0]
        leye_y = y1 + oh * landmark[index][1]
        reye_x = x1 + ow * landmark[index][2]
        reye_y = y1 + oh * landmark[index][3]
        nose_x = x1 + ow * landmark[index][4]
        nose_y = y1 + oh * landmark[index][5]
        lmouth_x = x1 + ow * landmark[index][6]
        lmouth_y = y1 + oh * landmark[index][7]
        rmouth_x = x1 + ow * landmark[index][8]
        rmouth_y = y1 + oh * landmark[index][9]
        return [x1, y1, x2, y2, cfd, leye_x, leye_y, reye_x, reye_y, nose_x, nose_y, lmouth_x, lmouth_y, rmouth_x, rmouth_y]























