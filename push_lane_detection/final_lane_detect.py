# import os
# os.environ['IMAGEIO_FFMPEG_EXE'] ='/home/zdq/anaconda3/lib/python3.6/site-packages'
import cv2
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from fcm import centers
# from carlane_detect import centers
# import matplotlib as mpl
# mpl.use('TKAgg')
# from matplotlib import pyplot as plt
from carlane_det import dataframe
from carlane_det import image
# from carlane_det import rho_inf
import carlane_det
# from moviepy.editor import VideoFileClip
from demo import img_detection
# global mix
global data_point

ylist=[]
xlist = []
y=[]
new_k=[]


# dic_kb = centers
# print(dic_kb[0][0])

#从已经得到的斜率，角度，截距数据中只留下k，b
dataframe = dataframe.drop(['1'],axis = 1)
columns = list(dataframe.columns)
features = columns[-2:len(columns)]
df = dataframe[features]
# print(columns)
# print(df)
num_attr = len(df.columns)
n = len(df)
# print(num_attr,n)

#把dataframe中的k，b都放入data_point列表中
data_point = list()
for i in range(n):
    data_point.append(list(df.iloc[i]))
# print(data_point)

#得到右车道线的数据放入xlist，ylist中
for i in range(len(data_point)):
    if data_point[i][1] < 0:
        ylist.append(data_point[i][1])
        xlist.append(data_point[i][0])
# print(ylist)
# print((xlist))

# FCM算法中得到的聚类中心的坐标的处理
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

#构造和原图一致的灰度图
map_img = np.zeros((image.shape[0], image.shape[1],3), dtype=np.uint8)#生成一个空灰度图像

# if not carlane_det.rho_inf:
#
#     cv2.line(map_img,(int(carlane_det.rho_inf),0),(int(carlane_det.rho_inf),image.shape[0]),(0,0,255),2)

# 画出相近角度平均算法下的最终车道线
for i in range(len(data_point)):
    if data_point[i][1]>0:
        # print(data_point[i][1])
        # print(int(data_point[i][1]))
        cv2.line(map_img,((0,int(data_point[i][1]))),(int(-data_point[i][1]//data_point[i][0]),0),(0,0,255),2)
        # print(data_point[i][1])
    else:
        # cv2.line(map_img, ((0, int(data_point[i][1]))), (int(-data_point[i][1] // data_point[i][0]), 0), (0, 0, 255), 2)
        for j in range(len(ylist)):
            # cv2.line(map_img,(int((image.shape[1]-ylist[j])//xlist[j]),image.shape[1]),(int(-ylist[j]//xlist[j]),0),(0,0,255),2)
            cv2.line(map_img,(0,int(ylist[j])),(int((image.shape[1]-ylist[j])//xlist[j]),image.shape[1]),(0,0,255),2)

# FCM算法聚类得到的点画出的最终车道线
# for k,v in dic_kb.items():
#     if v>0:
#         cv2.line(map_img,(0,int(v)),(int(-v//k),0),(0,255,0),2)
#         print(int(v),int(-v//k))
#     else:
#         for i in range(len(ylist)):
#             cv2.line(map_img,((int((image.shape[1]-ylist[i])//xlist[i])),image.shape[1]),(int(-ylist[i]//xlist[i]),0),(0,255,0),2)
# cv2.imshow('gray',map_img)

# 图像融合
def weighted_img(img, initial_img, α=0.5, β=0.5, λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    # return cv2.addWeighted(img_detection, α, map_img, β, λ)
    return cv2.addWeighted(img_detection, α, map_img, β, λ)

mix = weighted_img(img_detection, map_img, α=1, β=0.8, λ=0.)
cv2.imshow('mix',mix)
cv2.waitKey(0)
cv2.destroyAllWindows()
# return mix
# output = 'sol.MOV'
# clip = VideoFileClip("/home/zdq/tensorflow-yolov3-master/data/DJI_0008.MOV")
# out_clip = clip.fl_image(mix)
# out_clip.write_videofile(output, audio=False)