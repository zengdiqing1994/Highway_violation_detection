import cv2
import math
import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
from final_lane_detect import data_point
from carlane_det import image
# from carlane_det import rho_inf
import utils
global box
ZERO = 1e-9
minus_lane = {}
pos_lane = {}

# 点
class Point(object):
    def __init__(self,x,y):
        self.x, self.y = x, y
#向量
class Vector(object):
    def __init__(self, start_point, end_point):
        self.start, self.end = start_point, end_point
        self.x = end_point.x - start_point.x
        self.y = end_point.y - start_point.y

# def negative(vector):
#     return Vector(vector.end_point,vector.start_point)

def vector_product(vectorA, vectorB):
    '''计算 x_1 * y_2 - x_2 * y_1'''
    product =  vectorA.x * vectorB.y - vectorB.x * vectorA.y
    return  product

#判断是否有交点
def is_intersected(A, B, C, D):
    '''A, B, C, D 为 Point 类型'''
    AC = Vector(A, C)
    AD = Vector(A, D)
    BC = Vector(B, C)
    BD = Vector(B, D)
    # CA = negative(AC)
    # CB = negative(BC)
    # DA = negative(AD)
    # DB = negative(BD)
    res = (vector_product(AC, AD) * vector_product(BC, BD) <= ZERO) \
           and (vector_product(AC, BC) * vector_product(AD, BD) <= ZERO)
    return res

#主程序
def main():
    A = Point(utils.box[0],utils.box[1])
    B = Point(utils.box[2],utils.box[3])

    C = []
    C_list = []
    C_x = []
    C_y = []
    D_list = []
    D_x = []
    D_y = []
    result = []
    # print(data_point)

    #左车道线群
    for i in range(len(data_point)):
        if data_point[i][1] > 0:
            minus_lane[data_point[i][0]] = []
            minus_lane[data_point[i][0]].append([int(-data_point[i][1]//data_point[i][0]),0])
            minus_lane[data_point[i][0]].append([0,int(data_point[i][1])])

    #右车道线群
        elif data_point[i][1] < 0 :
            pos_lane[data_point[i][0]] = []
            # pos_lane[data_point[i][0]].append([int(rho_inf),0])
            # pos_lane[data_point[i][0]].append([int(rho_inf),image.shape(0)])
            pos_lane[data_point[i][0]].append([int(-data_point[i][1]//data_point[i][0]),0])
            pos_lane[data_point[i][0]].append([int((image.shape[0]-data_point[i][1])//data_point[i][0]),image.shape[0]])

    for j in minus_lane.values():
        C_list.append(j[0])
        D_list.append(j[1])
    for m in pos_lane.values():
        C_list.append(m[0])
        D_list.append(m[1])

    #得到车道线的所有线段坐标
    for c in C_list:
        # print(C[0])
        C_x.append(c[0])
        C_y.append(c[1])
    print("C_x :" + str(C_x),"C_y :" + str(C_y))
    for d in D_list:
        D_x.append(d[0])
        D_y.append(d[1])
    print("D_x :" + str(D_x),"D_y :" + str(D_y))

    #判断1辆车是否违章轧线
    for i in range(len(C_x)):
        # for c_y in C_y:
        C = Point(C_x[i],C_y[i])
        D = Point(D_x[i],D_y[i])
        res = is_intersected(A,B,C,D)
        result.append(res)

        if res == False:
            # print("车辆没有碾压第" + str(i+1) + "条车道线")
            continue

        elif res == True:
            print("\n")
            print("警告！1车辆违章碾轧第" + str(i+1) + "条车道线")
            break
    if result.count(True) == 0:
        print("\n")
        print("无车辆违章轧线")

main()

