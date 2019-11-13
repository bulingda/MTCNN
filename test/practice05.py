import os
import cv2
path = r'E:\data\data\48\landmark'
label = r'E:\data\data\48\landmark.txt'
count = 0
for _ in os.listdir(path):
    count += 1

with open(label) as lines:
    for _, line in enumerate(lines):
        strs1 = str(line).strip().split()
        for i in range(count):
            img = os.path.join(r"E:\data\data\48\landmark\{0}.jpg".format(i))
            image = cv2.imread(img)
            # cv2.putText(img, '{0}'.format(round(box[i][4], 6)), (boxs[i][0], boxs[i][1] - 7),
            #             cv2.FONT_HERSHEY_COMPLEX, 0.5,
            #             (0, 255, 0), thickness=1)
            # cv2.rectangle(img, (boxs[i][0], boxs[i][1]), (boxs[i][2], boxs[i][3]), (0, 255, 0), 1, 0)
            cv2.circle(image, (int(float(strs1[6])), int(float(strs1[7]))), 1, (0, 0, 255), 4)
            cv2.circle(image, (int(float(strs1[8])), int(float(strs1[9]))), 1, (0, 0, 255), 4)
            cv2.circle(image, (int(float(strs1[10])), int(float(strs1[11]))), 1, (0, 0, 255), 4)
            cv2.circle(image, (int(float(strs1[12])), int(float(strs1[13]))), 1, (0, 0, 255), 4)
            cv2.circle(image, (int(float(strs1[14])), int(float(strs1[15]))), 1, (0, 0, 255), 4)

        cv2.namedWindow("image")
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

