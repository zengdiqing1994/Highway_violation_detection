# import cv2
#
# img = cv2.imread('D:/python_code/goodimg.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #cv2.imwrite("gray.jpg", gray)
# ret, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
# #cv2.imwrite("binary.jpg", binary)
# image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
# cv2.putText(img, "{:.3f}".format(len(contours)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)
# cv2.imshow("img", img)
# #cv2.imwrite("contours.jpg", img)
# cv2.waitKey(0)

import cv2

#img = cv2.imread('C:/Users/zdq/Desktop/video process/DJI1.jpg')
img = cv2.imread('C:/Users/zdq/Desktop/video process/DJI.jpg')
#img = cv2.imread('D:/python_code/goodimg.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

image,contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

cv2.imshow("img", img)
cv2.imwrite("C:/Users/zdq/Desktop/video process/DJI3.jpg",img)
cv2.waitKey(0)


# "C:/Users/zdq/Desktop/video process/videoframe/picturescqh745.jpg"
# "C:/Users/zdq/Desktop/video process/videoframe/picturescqh760.jpg"

#C:/Users/zdq/Desktop/video process/DJI1.jpg
#C:/Users/zdq/Desktop/video process/DJI.jpg

#D:/python_code/goodimg.jpg