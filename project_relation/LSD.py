import numpy as np
import cv2
from matplotlib import pyplot as plt

#Read gray image
img = cv2.imread("C:/Users/zdq/Desktop/video process/DJI3.jpg",0)
#img = cv2.imread("C:/Users/zdq/Desktop/video process/DJI.jpg",0)
#img = cv2.imread("D:/python_code/goodimg.jpg",0)
#Create default parametrization LSD
lsd = cv2.createLineSegmentDetector(0)

#Detect lines in the image
lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines

#Draw detected lines in the image
drawn_img = lsd.drawSegments(img,lines)

# #Show image
cv2.imshow("LSD",drawn_img)
#
# image = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2GRAY)#将图像转化为灰度
# blurred = cv2.GaussianBlur(image, (5, 5), 0)#高斯滤波
#自适应阈值化处理
#cv2.ADAPTIVE_THRESH_MEAN_C：计算邻域均值作为阈值
# thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
# cv2.imshow("Mean Thresh", thresh)
# #cv2.ADAPTIVE_THRESH_GAUSSIAN_C：计算邻域加权平均作为阈值
# thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
# cv2.imshow("Gaussian Thresh", thresh)
cv2.waitKey(0)





# "C:/Users/zdq/Desktop/video process/videoframe/picturescqh745.jpg"
# "C:/Users/zdq/Desktop/video process/videoframe/picturescqh760.jpg"

#C:/Users/zdq/Desktop/video process/DJI1.jpg
#C:/Users/zdq/Desktop/video process/DJI.jpg
#D:/python_code/goodimg.jpg
