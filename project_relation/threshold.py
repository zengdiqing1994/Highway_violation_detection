import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("D:/python_code/push_lane/good3.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# plt.subplot(131), plt.imshow(image, "gray")
# plt.title("source image"), plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.hist(image.ravel(), 256)
# plt.title("Histogram"), plt.xticks([]), plt.yticks([])
#ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
ret1,th1 = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
cv2.imshow("th1",th1)

cv2.waitKey(0)
cv2.imwrite("D:/python_code/push_lane/gray.jpg",gray)
