import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('C:/Users/zdq/Desktop/video process/videoframe/picturescqh745.jpg',0)          # queryImage
img2 = cv2.imread('C:/Users/zdq/Desktop/video process/videoframe/picturescqh760.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

plt.imshow(img3),plt.show()

# "C:/Users/zdq/Desktop/video process/videoframe/picturescqh745.jpg"
# "C:/Users/zdq/Desktop/video process/videoframe/picturescqh760.jpg"

#C:/Users/zdq/Desktop/video process/DJIcqh100.jpg
#C:/Users/zdq/Desktop/video process/DJIcqh115.jpg