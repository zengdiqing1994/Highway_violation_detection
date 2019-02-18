# import cv2
#
# img = cv2.imread('C:/Users/zdq/Desktop/video process/DJIcqh0.jpg')
#
# lower_reso = cv2.pyrDown(img)
#
# cv2.imshow('src',img)
# cv2.imshow('HigherReso',lower_reso)
# cv2.imwrite("C:/Users/zdq/Desktop/video process/DJI2.jpg",lower_reso)
#
# cv2.waitKey()
import cv2

im1 = cv2.imread('D:/python_code/pictures/person.jpg')
#cv2.imshow('image1', im1)
#cv2.waitKey(0)

im2 = cv2.resize(im1, (1280,1024), interpolation=cv2.INTER_CUBIC)
cv2.imshow('image2', im2)

cv2.imwrite('per.jpg', im2)
cv2.waitKey(0)
