import cv2
import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
# from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import math

kernel_size = 3

rho = 2    #1,2
theta = np.pi / 180
threshold = 160  #越大检测到的线可能越细
min_line_length = 90
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
rho_axis = []
big_k = []
big_b = []
dic_big = {}
the = []
b_new = []
centers = []
global image
image = cv2.imread('/home/zdq/tensorflow-yolov3-master/data/UAV_pic0/1100.jpg')
# print(image.shape)
# cv2.imshow('image',image)
#对图像进行预处理
def grayscale(img):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = grayscale(image)
# cv2.imshow("gray",gray)

def Gaussion_blur(gray,kernel_size):
    return cv2.GaussianBlur(gray,(kernel_size, kernel_size),1)
gaussion = Gaussion_blur(gray,kernel_size)
# cv2.imshow('gaussion',gaussion)

ret1,th1 = cv2.threshold(gaussion,140,255,cv2.THRESH_BINARY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)
canny = canny(gaussion,300,150)
# cv2.imshow('canny',canny)
#闭操作：先膨胀后腐蚀
kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

# dataframe = []
def draw_lines(img, lines, color=[255, 0, 0], thickness=0):
    global dataframe
    global theta_axis
    global centers
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # 算出k，b斜率和截距
            k.append((y2 - y1) / (x2 - x1))
            b.append(y1 - (k[len(k) - 1] * x1))
            dic = dict(zip(k, b))
    # plt.xlabel('k')
    # plt.ylabel('b')
    # plt.scatter(dic.keys(), dic.values(), c='r', marker='o')
    # plt.show()
            # 算出极坐标参数rho,theta
            # theta_axis.append(math.atan((x1 - x2) / (y2 - y1)) * 180 / math.pi)
            # #             theta_axis.append(math.atan2(x2,y2)*180/math.pi)
            # rho_axis.append(
            #     x1 * math.cos(theta_axis[len(theta_axis) - 1]) + y1 * math.sin(theta_axis[len(theta_axis) - 1]))
            # dic_1 = dict(zip(theta_axis, rho_axis))
        # 把斜率对应的角度给求出来，排序
    for key in list(dic.keys()):
        if abs(key) > 6000:
            dic.pop(key)
        elif abs(key) == 0.0:
            dic.pop(key)
        elif abs(key) < 0.3:
            dic.pop(key)


    for key in dic.keys():
        the.append(math.atan(key) * 180 / math.pi)
        b_new.append(dic[key])
    dic_1=dict(zip(the,b_new))

    # print(dic_1)
        #dic.update(math.atan(key)*180/math.pi : dic.pop(key))
    # for value in dic_1.values():
    #     if abs(value) > 6000

    theta_list=sorted(dic_1.items(),key=lambda item:item[0])
    # print(theta_list)
    for i in range(len(theta_list)):
        theta_list1.append(list(theta_list[i]))
    print(theta_list1)

    dic_1 = dict(theta_list1)
    print(dic_1)

    # 把角度差的绝对值小于3°的角度单独放在一个列表当中
    fir_the = 0
    tmp = []
    tmp1 = []
    tmp_total = []
    tmp_total1 = []
    fir_the = list(dic_1.keys())[0]
    length = len(list(dic_1.keys()))
    print((fir_the))
    for i in range(0, length - 1):
        if abs(fir_the - list(dic_1.keys())[i+1]) < 5:
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
    print(tmp_total1)
        #     print((tmp_total))

    # 把相近角度的线进行平均合并
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
    plt.xlabel('k')
    plt.ylabel('b')
    plt.scatter(dic.keys(), dic.values(), c='r', marker='o')
    plt.show()
    dict_1 = {'0': pd.Series(dic).index, '1': pd.Series(dic).values}
    dataframe = pd.DataFrame(dict_1)
    print(dataframe)
   # datafram = copy.deepcopy(pd.DataFrame(dict_1))



def hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(img,rho, theta, threshold,minLineLength=min_line_length,maxLineGap=max_line_gap)
    line_img = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    draw_lines(line_img,lines)
    return line_img

line_img = hough_lines(closed,rho,theta,threshold,min_line_length,max_line_gap)

# plt.figure(2)
# # # fig = plt.gcf()
# # # fig.set_size_inches(16.5, 12.5)
# # plt.imshow(line_img)
#
# cv2.imshow("line_img",line_img)
# cv2.waitKey(0)
