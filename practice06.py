import os
import cv2
path = r'F:\data\img_celeba'
label = r'F:\data\list_landmarks_celeba.txt'
count = 0
for _ in os.listdir(path):
    count += 1

with open(label) as lines:
    for a, line in enumerate(lines):
        if a < 2:
            continue
        strs1 = str(line).strip().split()
        img = os.path.join(r"F:\data\img_celeba\{0}".format(strs1[0]))
        image = cv2.imread(img)
        for i in range(count):
            cv2.circle(image, (int(float(strs1[1])), int(float(strs1[2]))), 1, (0, 0, 255), 4)
            cv2.circle(image, (int(float(strs1[3])), int(float(strs1[4]))), 1, (0, 0, 255), 4)
            cv2.circle(image, (int(float(strs1[5])), int(float(strs1[6]))), 1, (0, 0, 255), 4)
            cv2.circle(image, (int(float(strs1[7])), int(float(strs1[8]))), 1, (0, 0, 255), 4)
            cv2.circle(image, (int(float(strs1[9])), int(float(strs1[10]))), 1, (0, 0, 255), 4)

        cv2.namedWindow("image")
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

