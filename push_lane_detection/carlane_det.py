import matplotlib as mpl
mpl.use('TKAgg')
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import math
from mpl_toolkits import mplot3d

kernel_size = 3

rho = 2    #1,2
theta = np.pi / 180
threshold = 200  #越大检测到的线可能越细
min_line_length = 200
max_line_gap = 50

k=[]
b=[]
dic = {}
dic_1 = {}
dic_2 = {}
dict_1={}
dict_2 = {}
dataframe=[]
theta_list = {}
theta_list1 = []
theta_axis = []

rho_axis = []
big_k = []
big_b = []
dic_big = {}
the = []
b_new = []
centers = []
dataframe1 = []

global image
global theta_inf
global rho_inf
image = cv2.imread('/home/zdq/tensorflow-yolov3-master/data/UAV_pic0/1100.jpg')
# print(image.shape)
# cv2.imshow('image',image)

#对图像进行预处理
def grayscale(img):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = grayscale(image)
# cv2.imshow("gray",gray)

#高斯平滑
def Gaussion_blur(gray,kernel_size):
    return cv2.GaussianBlur(gray,(kernel_size, kernel_size),1)
gaussion = Gaussion_blur(gray,kernel_size)
# cv2.imshow('gaussion',gaussion)

# 阈值分割
ret1,th1 = cv2.threshold(gaussion,175,255,cv2.THRESH_BINARY)
# cv2.imshow("th1",th1)
# cv2.waitKey(0)

# Canny边缘检测
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)
canny = canny(gaussion,455,150)
cv2.imshow('canny',canny)
cv2.waitKey(0)

#闭操作：先膨胀后腐蚀
kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

# 画线
def draw_lines(img, lines, color=[255, 0, 0], thickness=0):
    global dataframe
    global theta_axis
    global centers
    global rho_inf

    # 提取车道线参数后进行画线
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # 算出k，b斜率和截距，构成map：dic
            k.append((y2 - y1) / (x2 - x1))
            b.append(y1 - (k[len(k) - 1] * x1))
            dic = dict(zip(k, b))

            # 找出垂直的直线，先转换到极坐标空间
            theta_axis.append(math.atan((x1 - x2) / (y2 - y1)) * 180 / math.pi)
            rho_axis.append(
                x1 * math.cos(theta_axis[len(theta_axis) - 1]) + y1 * math.sin(theta_axis[len(theta_axis) - 1]))
            dic_2 = dict(zip(theta_axis, rho_axis))
    # theta_inf = []
    # rho_inf = []

    # 找出垂直直线就是角度theta为0的情况
    for i,j in dic_2.items():
        if abs(i) == 0.0:
            theta_inf = i
            rho_inf = j

            # 算出极坐标参数rho,theta
            # theta_axis.append(math.atan((x1 - x2) / (y2 - y1)) * 180 / math.pi)
            # #             theta_axis.append(math.atan2(x2,y2)*180/math.pi)
            # rho_axis.append(
            #     x1 * math.cos(theta_axis[len(theta_axis) - 1]) + y1 * math.sin(theta_axis[len(theta_axis) - 1]))
            # dic_1 = dict(zip(theta_axis, rho_axis))

    # 对于k，b空间，斜率太大的需要pop掉，因为会导致无穷大的情况无法画出垂直车道线
    for key in list(dic.keys()):
        if abs(key) > 6000:
            # theta_inf = math.atan((x1-x2))
            dic.pop(key)
            print(key)
        #水平的直线斜率为0,pop掉
        elif abs(key) == 0.0:
            dic.pop(key)
        #斜率很小的直线滤除
        elif abs(key) < 0.3:
            dic.pop(key)

    # 把斜率对应的角度给求出来，排序
    for key in dic.keys():
        the.append(math.atan(key) * 180 / math.pi)
        b_new.append(dic[key])
    dic_1 = dict(zip(the, b_new))

    #这里是想把三维图像画出来，k，theta，b
    dict_2 = {'0': k, '1': the, '2': b}
    # dataframe1 = pd.DataFrame(dict_2)
    #     # print(dataframe1)
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.set_xlabel('k')
    # ax.set_ylabel('theta')
    # ax.set_zlabel('b')
    # ax.scatter3D(dataframe1['0'], dataframe1['1'], dataframe1['2'], c=dataframe1['2'], cmap='Dark2')
    # plt.show()
    # print(dic_1)
        #dic.update(math.atan(key)*180/math.pi : dic.pop(key))
    # for value in dic_1.values():
    #     if abs(value) > 6000

    # 把斜率对应的角度给求出来，排序
    theta_list=sorted(dic_1.items(),key=lambda item:item[0])
    # print(theta_list)
    for i in range(len(theta_list)):
        theta_list1.append(list(theta_list[i]))
    print(theta_list1)

    dic_1 = dict(theta_list1)
    print(dic_1)

    # 把角度差的绝对值小于4°的角度单独放在一个列表当中
    fir_the = 0
    tmp = []
    tmp1 = []
    tmp_total = []
    tmp_total1 = []
    fir_the = list(dic_1.keys())[0]
    length = len(list(dic_1.keys()))
    print((fir_the))
    for i in range(0, length - 1):
        if abs(fir_the - list(dic_1.keys())[i+1]) < 4:
            tmp.append(list(dic_1.keys())[i+1])
            tmp1.append(dic_1[list(dic_1.keys())[i+1]])
        else:
            tmp.append(fir_the)
            tmp1.append(dic_1[fir_the])
            fir_the = list(dic_1.keys())[i+1]
            tmp_total.append(tmp)
            tmp_total1.append(tmp1)
            tmp = []
            tmp1 = []
            # if i+1 == len(theta_list)-1:
    tmp.append(fir_the)
    tmp_total.append(tmp)
    tmp1.append(dic_1[fir_the])
    tmp_total1.append(tmp1)
    print(tmp_total)
    print(tmp_total1)
        #     print((tmp_total))

    # 把相近角度的线进行平均合并，相对应的截距也平均
    tmp_final = []
    sum = 0
    for i in range(len(tmp_total)):
        for j in range(len(tmp_total[i])):
            sum += tmp_total[i][j]
        sum = sum / len(tmp_total[i])
        tmp_final.append(sum)
        sum = 0
    tmp_revertk = []
    for i in range(len(tmp_final)):
        tmp_revertk.append(math.tan(math.radians(tmp_final[i])))
    print(tmp_revertk)

    tmp_finalb = []
    sum1 = 0
    for i in range(len(tmp_total1)):
        for j in range(len(tmp_total1[i])):
            sum1 += tmp_total1[i][j]
        sum1 = sum1 / len(tmp_total1[i])
        tmp_finalb.append(sum1)
        sum1 = 0
    print(tmp_finalb)

    dic = dict(zip(tmp_revertk,tmp_finalb))
    centers = (list(zip(tmp_revertk,tmp_finalb)))

    # plt.xlim(-6, 6)
    # plt.ylim(-500, 2000)
    # plt.xlabel('k')
    # plt.ylabel('b')
    # plt.scatter(dic.keys(), dic.values(), c='r', marker='o')
    # plt.show()

    # 把最终合并后的车道线点的三维图像画出
    dict_1 = {'0': pd.Series(dic).index,'1':tmp_final, '2': pd.Series(dic).values}
    dataframe = pd.DataFrame(dict_1)
    print(dataframe)

    # plt.figure(2)
    # ax = plt.axes(projection = '3d')
    # ax.set_xlabel('k')
    # ax.set_ylabel('theta')
    # ax.set_zlabel('b')
    # ax.scatter3D(dataframe['0'],dataframe['1'],dataframe['2'],c = dataframe['2'],cmap = 'Dark2')
    # plt.show()
   # datafram = copy.deepcopy(pd.DataFrame(dict_1))


# 通过阈值分割的二值图像或者边缘检测的边缘特征来进行Hough变换，车道线参数提取
def hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(img,rho, theta, threshold,minLineLength=min_line_length,maxLineGap=max_line_gap)
    line_img = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    draw_lines(line_img,lines)
    return line_img

# line_img = hough_lines(th1,rho,theta,threshold,min_line_length,max_line_gap)
line_img = hough_lines(canny,rho,theta,threshold,min_line_length,max_line_gap)

# plt.figure(2)
# # # fig = plt.gcf()
# # # fig.set_size_inches(16.5, 12.5)
# # plt.imshow(line_img)
#
# cv2.imshow("line_img",line_img)
# cv2.waitKey(0)