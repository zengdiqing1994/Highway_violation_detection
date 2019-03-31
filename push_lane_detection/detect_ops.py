"""
Detection ops for Yolov2,解码darknet19网络得到的参数
"""

import tensorflow as tf
import numpy as np


def decode(detection_feat, feat_sizes=(13, 13), num_classes=80,
           anchors=None):
    """decode from the detection feature"""
    """
    model_output:darknet19网络输出额特征图
    output_size：darknet19网络输出的特征图大小，默认是13×13（默认输入是416×416，下采样32）
    """
    H, W = feat_sizes
    num_anchors = len(anchors) #这里的anchor是在configs文件中设置的
    #13×13×num_anchors*(num_class+5),第一个维度自适应batchsize
    detetion_results = tf.reshape(detection_feat, [-1, H * W, num_anchors,
                                        num_classes + 5])
    # darknet19网络输出转化——偏移值，置信度，类别概率
    bbox_xy = tf.nn.sigmoid(detetion_results[:, :, :, 0:2]) #中心坐标相对于cell左上角的偏移量，sigmoid函数归一化到0-1
    bbox_wh = tf.exp(detetion_results[:, :, :, 2:4])    #相对于anchor的wh比例，通过e指数解码
    obj_probs = tf.nn.sigmoid(detetion_results[:, :, :, 4]) #置信度，sigmoid函数归一化到0-1
    class_probs = tf.nn.softmax(detetion_results[:, :, :, 5:])  #网络回归的是”得分“，用softmax转变成类别概率

    anchors = tf.constant(anchors, dtype=tf.float32) #将传入的anchors转变成tf格式的常量列表

    # 构建特征图每个cell的左上角的xy坐标
    height_ind = tf.range(H, dtype=tf.float32)#range（0,13）
    width_ind = tf.range(W, dtype=tf.float32)#range（0,13）
    # 编程x_cell=[[0,1,....12],...[0,1,....12]]和y_cell=[[0,1,....12],...[0,1,....12]]
    x_offset, y_offset = tf.meshgrid(height_ind, width_ind)
    x_offset = tf.reshape(x_offset, [1, -1, 1]) #和上面[H*W,num_anchors,num_class+5]对应
    y_offset = tf.reshape(y_offset, [1, -1, 1])

    # decode
    bbox_x = (bbox_xy[:, :, :, 0] + x_offset) / W
    bbox_y = (bbox_xy[:, :, :, 1] + y_offset) / H
    bbox_w = bbox_wh[:, :, :, 0] * anchors[:, 0] / W * 0.5
    bbox_h = bbox_wh[:, :, :, 1] * anchors[:, 1] / H * 0.5
    #中心坐标+宽高box(x,y,w,h) -> xmin=x-w/2 -> 左上+右下box(xmin,ymin,xmax,ymax)
    bboxes = tf.stack([bbox_x - bbox_w, bbox_y - bbox_h,
                       bbox_x + bbox_w, bbox_y + bbox_h], axis=3)

    return bboxes, obj_probs, class_probs
