import usingnet
import os
from PIL import Image
import cv2
import numpy as np
import time
path = r"D:\ChromeCoreDownloads\videoplayback.mp4"
if __name__ == '__main__':
    test = usingnet.Detector()
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        start_time = time.time()
        frames = frame[:, :, ::-1]
        image = Image.fromarray(frames, 'RGB')
        box = test.detect(image)
        # print(type(box))
        # print(box[1][4])
        boxs = box.astype(np.int)
        # print(box[1][0])
        # print(len(box))
        # print(box)
        for i in range(len(box)):
            cv2.putText(frame, '{0}'.format(round(box[i][4], 6)), (boxs[i][0], boxs[i][1] - 7),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 255, 0), thickness=1)
            cv2.rectangle(frame, (boxs[i][0], boxs[i][1]), (boxs[i][2], boxs[i][3]), (0, 255, 0), 1, 0)
            cv2.circle(frame, (boxs[i][5], boxs[i][6]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (boxs[i][7], boxs[i][8]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (boxs[i][9], boxs[i][10]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (boxs[i][11], boxs[i][12]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (boxs[i][13], boxs[i][14]), 1, (0, 0, 255), 4)
        end_time = time.time()
        print("2123132134", end_time - start_time)
        cv2.imshow("image", frame)
        cv2.waitKey(1)
        # if cv2.waitKey(40) & 0xFF == ord('q'):
        #     break


