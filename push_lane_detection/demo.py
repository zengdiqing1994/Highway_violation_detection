"""
Demo for yolov2，主函数
"""

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from model import darknet
from detect_ops import decode
from utils import preprocess_image, postprocess, draw_detection
from config import anchors, class_names
# global box
global img_detection
from carlane_det import image

input_size = (416, 416)
# image_file = "/home/zdq/python_code/push_lane/good1.jpg"
# image_file = "/home/zdq/tensorflow-yolov3-master/data/UAV_pic0/1100.jpg"

# image = cv2.imread(image_file)
image_shape = image.shape[:2] #只能取wh，channel=3不取

# copy,resize 416*416,归一化，在第0维增加存放batchsize维度
image_cp = preprocess_image(image, input_size)
"""
image = Image.open(image_file)
image_cp = image.resize(input_size, Image.BICUBIC)
image_cp = np.array(image_cp, dtype=np.float32)/255.0
image_cp = np.expand_dims(image_cp, 0)
#print(image_cp)
"""

# （1）输入图片进入darknet19网络得到特征图，并进行解码得到：xmin xmax表示的边界框，置信度，类别概率
images = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3])
detection_feat = darknet(images) #darknet网络输出的特征图
feat_sizes = input_size[0] // 32, input_size[1] // 32   #特征图尺寸是图像下采样32倍
detection_results = decode(detection_feat, feat_sizes, len(class_names), anchors) #解码

checkpoint_path = "/home/zdq/YOLO/checkpoint_dir/yolo2_coco.ckpt"
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    bboxes, obj_probs, class_probs = sess.run(detection_results, feed_dict={images: image_cp})

# （2）筛选解码后回归边界框——NMS（post process后期处理）
bboxes, scores, class_inds = postprocess(bboxes, obj_probs, class_probs,
                                         image_shape=image_shape)

# （3）绘制筛选后的边界框
img_detection = draw_detection(image, bboxes, scores, class_inds, class_names)


#cv2.imwrite("/home/zdq/YOLO/detection.jpg", img_detection)
# cv2.imshow("detection results", img_detection)
#
# cv2.waitKey(0)




