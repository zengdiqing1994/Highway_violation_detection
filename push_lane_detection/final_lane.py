import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from fcm import centers
from carlane_detect import centers
import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
from carlane_detect import dataframe
from carlane_detect import image

ylist=[]
xlist = []
y=[]
new_k=[]

# dic_kb = centers
# print(dic_kb[0][0])

columns = list(dataframe.columns)
features = columns[-2:len(columns)]
df = dataframe[features]
print(columns)
print(df)
num_attr = len(df.columns) #2列
n = len(df) #4行
print(num_attr,n)
data_point = list()
for i in range(n):
    data_point.append(list(df.iloc[i]))
print(data_point)

for i in range(len(data_point)):
    if data_point[i][1] < 0:
        ylist.append(data_point[i][1])
        xlist.append(data_point[i][0])
print(ylist)
print((xlist))
# for k,v in dic_kb.items():
#     if v > 6000:
#         list(dic_kb.keys())[list(dic_kb.values()).index(v)]
#     print(k,v)
#
# for x in range((image.shape[1])):
#     for k,v in dic_kb.items():
#         if x == -v//k:
#             print(x)
# for v in dic_kb.values():
#     if v>6000:
#         dic_kb.pop(list(dic_kb.keys())[list(dic_kb.values()).index(v)])
# print(dic_kb)
# for k,v in dic_kb.items():
#     if v < 0:
#         ylist.append(v)
#         new_k = list(dic_kb.keys())[list(dic_kb.values()).index(v)]
#         xlist.append(new_k)
# print(ylist)
# print((xlist))

map_img = np.zeros((image.shape[0], image.shape[1],3), dtype=np.uint8)#生成一个空灰度图像
for i in range(len(data_point)):
    if data_point[i][1]>0:
        print(data_point[i][1])
        print(int(data_point[i][1]))
        cv2.line(map_img,((0,int(data_point[i][1]))),(int(-data_point[i][1]//data_point[i][0]),0),(0,255,0),2)
        print(data_point[i][1])
    else:
        for j in range(len(ylist)):
            cv2.line(map_img,(int((image.shape[1]-ylist[j])//xlist[j]),image.shape[1]),(int(-ylist[j]//xlist[j]),0),(0,255,0),2)

# for k,v in dic_kb.items():
#     if v>0:
#         cv2.line(map_img,(0,int(v)),(int(-v//k),0),(0,255,0),2)
#         print(int(v),int(-v//k))
#     else:
#         for i in range(len(ylist)):
#             cv2.line(map_img,((int((image.shape[1]-ylist[i])//xlist[i])),image.shape[1]),(int(-ylist[i]//xlist[i]),0),(0,255,0),2)
# cv2.imshow('gray',map_img)

def weighted_img(img, initial_img, α=0.5, β=0.5, λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(image, α, map_img, β, λ)

mix = weighted_img(image, map_img, α=0.5, β=0.5, λ=0.)
# cv2.imshow(mix,'mix')
cv2.imshow('mix',mix)
cv2.waitKey(0)
cv2.destroyAllWindows()
