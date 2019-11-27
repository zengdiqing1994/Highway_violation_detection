# -*- coding: utf-8 -*-
# import matplotlib as mpl
# mpl.use('TKAgg')
from __future__ import division

from collections import OrderedDict
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import math
from mpl_toolkits import mplot3d
from moviepy.editor import VideoFileClip

kernel_size = 3

rho = 2    # 1, 2
theta = np.pi / 180
threshold = 140  # 越大检测到的线可能越细
min_line_length = 150
max_line_gap = 30

k = []
b = []
dic = {}
dic_1 = {}
dic_2 = {}
dict_1 = {}
dict_2 = {}
dataframe = []
theta_list = {}
theta_list1 = []
theta_axis = []

rho_axis = []
big_k = []
big_b = []
dic_big = {}
the = []
the1 = []
b_new = []
centers = []
dataframe1 = []

ylist =[]
xlist = []
y = []
new_k=[]


# 对图像进行预处理
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 高斯平滑
def Gaussion_blur(gray, kernel_size):
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 1)


# 阈值分割
# ret1,th1 = cv2.threshold(gaussion,175,255,cv2.THRESH_BINARY)


# Canny边缘检测
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)
# canny = canny(gaussion, 300,150)


# 闭操作：先膨胀后腐蚀
# kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


# 画线
def draw_lines(img, lines, color=(255, 0, 0), thickness=0):
    global dataframe
    global theta_axis
    global centers
    global rho_inf
    global tmp_revertk
    global tmp_finalb

    # 提取车道线参数后进行画线
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # 算出k，b斜率和截距，构成map：dic
            k.append((y2 - y1) / (x2 - x1))
            b.append(y1 - (k[len(k) - 1] * x1))
            dic = dict(zip(k, b))

            # 找出垂直的直线，先转换到极坐标空间
            theta_axis.append(math.atan((x1 - x2) / (y2 - y1)) * 180 / math.pi)
            rho_axis.append(
                x1 * math.cos(theta_axis[len(theta_axis) - 1]) + y1 * math.sin(theta_axis[len(theta_axis) - 1]))
            dic_2 = dict(zip(theta_axis, rho_axis))

    """反正切求出角度"""
    for key in dic.keys():
        the.append(math.atan(key) * 180 / math.pi)

    # 找出垂直直线就是角度theta为0的情况
    for i,j in dic_2.items():
        if abs(i) == 0.0:
            theta_inf = i
            rho_inf = j

    """对于k，b空间，斜率太大的需要pop掉，因为会导致无穷大的情况无法画出垂直车道线"""
    for key in list(dic.keys()):
        if abs(key) > 6000:
            # theta_inf = math.atan((x1-x2))
            dic.pop(key)
            # print(key)
        # 水平的直线斜率为0,pop掉
        elif abs(key) == 0.0:
            dic.pop(key)
        # 斜率很小的直线滤除
        elif abs(key) < 0.3:
            dic.pop(key)

    """把斜率对应的角度给求出来，排序"""
    for key in dic.keys():
        the1.append(math.atan(key) * 180 / math.pi)
        b_new.append(dic[key])

    dic_1 = dict(zip(the, b_new))

    theta_list=sorted(dic_1.items(), key=lambda item: item[0])
    for i in range(len(theta_list)):
        theta_list1.append(list(theta_list[i]))
    dic_1 = OrderedDict(theta_list1)

    """把角度差的绝对值小于4°的角度单独放在一个列表当中"""
    fir_the = 0
    tmp = []
    tmp1 = []
    tmp_total = []
    tmp_total1 = []
    fir_the = list(dic_1.keys())[0]
    length = len(list(dic_1.keys()))
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

    tmp.append(fir_the)
    tmp_total.append(tmp)
    tmp1.append(dic_1[fir_the])
    tmp_total1.append(tmp1)
    tmp = []
    tmp1 = []

    """把相近角度的线进行平均合并，相对应的截距也平均"""
    tmp_final = []
    sum = 0
    for i in range(len(tmp_total)):
        for j in range(len(tmp_total[i])):
            sum += tmp_total[i][j]
        sum = sum / len(tmp_total[i])
        tmp_final.append(sum)
        sum = 0

    tmp_revertk = []    # 这里把合并得到的角度正切成斜率k
    for i in range(len(tmp_final)):
        tmp_revertk.append(math.tan(math.radians(tmp_final[i])))
    # tmp_revertk = []

    tmp_finalb = []   # 合并后的截距
    sum1 = 0
    for i in range(len(tmp_total1)):
        for j in range(len(tmp_total1[i])):
            sum1 += tmp_total1[i][j]
        sum1 = sum1 / len(tmp_total1[i])
        tmp_finalb.append(sum1)
        sum1 = 0
    # tmp_finalb = []

    dic = dict(zip(tmp_revertk, tmp_finalb))
    centers = (list(zip(tmp_revertk, tmp_finalb)))

    dict_1 = {'0': pd.Series(dic).index, '1': pd.Series(dic).values}
    dataframe = pd.DataFrame(dict_1)
    tmp_final = []
    tmp_revertk = []
    tmp_finalb = []
    dic = {}
    dict_1 = {}
    tmp = []
    tmp1 = []
    tmp_total = []
    tmp_total1 = []
    dic_1 = {}
    return dataframe


"""通过阈值分割的二值图像或者边缘检测的边缘特征来进行Hough变换，车道线参数提取"""


def hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw = draw_lines(line_img, lines)
    return line_img, draw


def weighted_img(img, initial_img, a=0.5, b=0.5, c=0.):
    # return cv2.addWeighted(img_detection, α, map_img, β, λ)
    return cv2.addWeighted(img, a, initial_img, b, c)


def process_img(img):
    global dataframe
    global data_point
    global df
    global map_img

    gray = grayscale(img)
    gaussion = Gaussion_blur(gray, kernel_size)
    ret1, th1 = cv2.threshold(gaussion, 175, 255, cv2.THRESH_BINARY)
    vertices = np.array([[(100, img.shape[0]), (311, 0), (1500, 0), (1900, img.shape[0])]])
    roi = region_of_interest(th1, vertices)
    line_img, draw = hough_lines(roi, rho, theta, threshold, min_line_length, max_line_gap)
    # roi = 0
    columns = list(draw.columns)
    features = columns[-2:len(columns)]

    df = dataframe[features]
    dataframe = []
    draw = []
    n = len(df)

    """把dataframe中的k，b都放入data_point列表中"""
    data_point = []
    for i in range(n):
        data_point.append(list(df.iloc[i]))
    df = []

    """得到右车道线的数据放入xlist，ylist中"""
    for i in range(len(data_point)):
        if data_point[i][1] < 0:
            ylist.append(data_point[i][1])
            xlist.append(data_point[i][0])

    # 构造和原图一致的灰度图
    map_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # 生成一个空灰度图像

    """画出相近角度平均算法下的最终车道线"""
    for i in range(len(data_point)-1):
        """左车道线"""
        if data_point[i][1] > 0:
            # map_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # 生成一个空灰度图像
            cv2.line(map_img, ((0, int(data_point[i][1]))), (int(-data_point[i][1] // data_point[i][0]), 0),
                     (0, 0, 255), 2)

            """右车道线"""
        else:
            # map_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # 生成一个空灰度图像
            for j in range(len(ylist)):
                cv2.line(map_img, (0, int(ylist[j])), (int((img.shape[1] - ylist[j]) // xlist[j]), img.shape[1]),
                         (0, 0, 255), 2)

    res_img = weighted_img(img, map_img, a=0.7, b=1, c=0.)
    # res_img = []
    data_point = []
    return res_img

# img = cv2.imread('/home/zdq/PycharmProjects/deep_sort_yolov3-master/pic/013.jpg')
# main(img)


# output = 'output2.mp4'
# clip = VideoFileClip("/home/zdq/PycharmProjects/deep_sort_yolov3-master/DJI_cut.mp4")
# out_clip = clip.fl_image(process_img)
# out_clip.write_videofile(output, audio=False)