import usingnet
import os
from PIL import Image
import cv2
import numpy as np

path = r"E:\using"
count = 0
for _ in os.listdir(path):
    count += 1
if __name__ == '__main__':
    for j in range(count):
        img = os.path.join(r"E:\using\{0}.jpg".format(j))
        imgs = Image.open(img).convert('RGB')
        test = usingnet.Detector()
        box = test.detect(imgs)
        # print(type(box))
        # print(box[1][4])
        boxs = box.astype(np.int)
        # print(box[1][0])
        # print(len(box))
        # print(box)
        image = cv2.imread(img)
        for i in range(len(box)):
            cv2.putText(image, '{0}'.format(round(box[i][4], 6)), (boxs[i][0], boxs[i][1] - 7),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 255, 0), thickness=1)
            cv2.rectangle(image, (boxs[i][0], boxs[i][1]), (boxs[i][2], boxs[i][3]), (0, 255, 0), 1, 0)
            cv2.circle(image, (boxs[i][5], boxs[i][6]), 1, (0, 0, 255), 4)
            cv2.circle(image, (boxs[i][7], boxs[i][8]), 1, (0, 0, 255), 4)
            cv2.circle(image, (boxs[i][9], boxs[i][10]), 1, (0, 0, 255), 4)
            cv2.circle(image, (boxs[i][11], boxs[i][12]), 1, (0, 0, 255), 4)
            cv2.circle(image, (boxs[i][13], boxs[i][14]), 1, (0, 0, 255), 4)
        img_p = Image.fromarray(np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), 'RGB')
        img_p.show()
        # cv2.namedWindow("image")
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
