"""
Help functions for YOLOv2,功能函数，包含：预处理输入图片，筛选边界框NMS，绘制筛选后的边界框
"""
import random
import colorsys

import cv2
import numpy as np
global Xleft, Xright, YBottom,Yleft
global box
global bboxes
#Xleft,Xright,YBottom = 0

############## preprocess image ##################

#图像前期处理
def preprocess_image(image, image_size=(416, 416)):
    """Preprocess a image to inference"""
    # 赋值原图像
    image_cp = np.copy(image).astype(np.float32)
    # resize the image
    image_rgb = cv2.cvtColor(image_cp, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, image_size)
    # normalize归一化
    image_normalized = image_resized.astype(np.float32) / 255.0
    # expand the batch_size dim，增加一个维度在第零维——batchsize
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

# 筛选解码后的回归边界框——NMS(post process后期处理)
def postprocess(bboxes, obj_probs, class_probs, image_shape=(416, 416),
                threshold=0.5):
    """post process the detection results"""
    # bboxs表示为：图片中有多少box就有多少行;4列分别是box(xmin,ymin,xmax,ymax)
    bboxes = np.reshape(bboxes, [-1, 4])
    # 将所有box还原成图片中真实的位置
    bboxes[:, 0:1] *= float(image_shape[1]) # xmin×width
    bboxes[:, 1:2] *= float(image_shape[0]) # ymin*height
    bboxes[:, 2:3] *= float(image_shape[1]) # xmax*width
    bboxes[:, 3:4] *= float(image_shape[0]) # ymax*height
    bboxes = bboxes.astype(np.int32)

    # clip the bboxs:将边界框超出整张图片(0,0)-(415,415)的部分cut掉
    bbox_ref = [0, 0, image_shape[1] - 1, image_shape[0] - 1]
    bboxes = bboxes_clip(bbox_ref, bboxes)

    # 置信度×max类别概率=类别置信度scores
    obj_probs = np.reshape(obj_probs, [-1])
    class_probs = np.reshape(class_probs, [len(obj_probs), -1])
    class_inds = np.argmax(class_probs, axis=1)
    class_probs = class_probs[np.arange(len(obj_probs)), class_inds]
    scores = obj_probs * class_probs

    # filter bboxes with scores > threshold
    #类别置信度scores>threshold的边界框bboxes留下
    keep_inds = scores > threshold
    bboxes = bboxes[keep_inds]
    scores = scores[keep_inds]
    class_inds = class_inds[keep_inds]

    # sort top K，排序top_k(默认为400)
    class_inds, scores, bboxes = bboxes_sort(class_inds, scores, bboxes)
    # nms
    class_inds, scores, bboxes = bboxes_nms(class_inds, scores, bboxes)

    return bboxes, scores, class_inds

# 绘制筛选后的边界框
def draw_detection(im, bboxes, scores, cls_inds, labels, thr=0.3):
    # for display
    ############################
    # Generate colors for drawing bounding boxes.
    global box

    hsv_tuples = [(x / float(len(labels)), 1., 1.)
                  for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    # draw image
    imgcv = np.copy(im)             #image copy成
    h, w, _ = imgcv.shape   #图片的宽和高，几通道
    for i, box in enumerate(bboxes):      #对回归框进行枚举
        if scores[i] < thr:         #如果置信度小于设定的阈值
            continue                #那么就继续循环get
        cls_indx = cls_inds[i]      #类别的索引

        thick = int((h + w) / 800)  #框的线的宽度
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[cls_indx], thick)              #通过左上角坐标，右下角坐标来绘制矩形框
        cv2.line(imgcv,(box[0],box[1]),(box[2],box[3]),(0,255,0),2)
        Xleft = box[0]
        Yleft = box[1]
        Xright = box[2]
        YBottom = box[3]
        print('Xleft = '+ str(Xleft))
        print('Yleft = ' + str(Yleft))
        print('Xright = '+ str(Xright))
        print('YBottom = ' + str(YBottom))
        # mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        # if box[1] < 20:                                     #开始给文本框加标签说明
        #     text_loc = (box[0] + 2, box[1] + 15)
        # else:
        #     text_loc = (box[0], box[1] - 10)
        # cv2.putText(imgcv, mess, text_loc,
        #             cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * h, colors[cls_indx], thick // 3)

    return imgcv


############## process bboxes ##################
#（1）cut the box：将边界框超出整张图片(0,0)-(415-415)的部分cut掉
def bboxes_clip(bbox_ref, bboxes):
    """Clip bounding boxes with respect to reference bbox.
    """
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    # cut the box
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes

#（2）按类别置信度scores降序，对边界框进行排序并仅保留top_k
def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes

# (3)计算IOU+NMS
# 计算两个box的IOU
def bboxes_iou(bboxes1, bboxes2):
    """Computing iou between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])
    # 计算两个box交集的wh：如果两个box没有交集，那么wh为0（按照计算方式wh为负数，跟0比较取最大值）
    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    #计算IOU
    int_vol = int_h * int_w #交集面积
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1]) #bboxes1面积
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])  #bboxes2面积
    iou = int_vol / (vol1 + vol2 - int_vol) #IOU=交集/并集
    return iou

#NMS，或者用tf.image.non_max_suppression(boxes,scores,self.max_output_size,self.iou_threshod)
def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_iou(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]






