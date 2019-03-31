import cv2
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import utils
from model import darknet
from detect_ops import decode
from utils import preprocess_image, postprocess, draw_detection
from config import anchors, class_names
# from utils import

global box
global Xright,Xleft,YBottom
global img_detection

kernel_size = 3

rho = 2
theta = np.pi/180
threshold = 119
min_line_length = 167
max_line_gap = 130

m = 1/18

image = cv2.imread('/home/zdq/python_code/push_lane/good3.jpg')
# cv2.imshow('image',image)


def grayscale(img):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = grayscale(image)
# cv2.imshow("gray",gray)

def Gaussion_blur(gray,kernel_size):
    return cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
gaussion = Gaussion_blur(gray,kernel_size)

ret1,th1 = cv2.threshold(gaussion,170,255,cv2.THRESH_BINARY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)
canny = canny(gaussion,230,300)

kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

def weight_add(img, initial_img, alpha = 0.5, belta = 0.5, gamma = 0.):
    return cv2.addWeighted(img,alpha,initial_img,belta,gamma)
mix = weight_add(th1,closed,alpha=0.5,belta = 0.5,gamma=0.)

def draw_lines(img, lines, color = [0, 0 ,255], thickness = 2):
    for line in lines:
        for x1,x2,y1,y2 in line:
            cv2.line(img, (x1, x2), (y1, y2), (0, 0, 255 ), 1)

def hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(img,rho, theta, threshold,minLineLength=min_line_length,maxLineGap=max_line_gap)
    line_img = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    draw_lines(line_img,lines)
    return line_img
line_img = hough_lines(mix,rho,theta,threshold,min_line_length,max_line_gap)
# cv2.imshow("line_img",line_img)

def region_of_interest(img,vertices):
    mask = np.zeros_like(img)
    if len(img.shape)>2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask,vertices,ignore_mask_color)
    masked_img = cv2.bitwise_and(img,mask)
    return masked_img
# vertices = np.array([[(110,194),(110,0),(150,0),(150,194)]],dtype = np.int32)
# roi = region_of_interest(line_img,vertices)


def main():
    input_size = (416, 416)
    # image_file = "/home/zdq/darknet/data/1.jpg"
    # image = cv2.imread(image_file)
    image_shape = image.shape[:2]
    image_cp = preprocess_image(image, input_size)

    images = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3])
    detection_feat = darknet(images)
    feat_sizes = input_size[0] // 32, input_size[1] // 32
    detection_results = decode(detection_feat, feat_sizes, len(class_names), anchors)

    checkpoint_path = "/home/zdq/YOLO/checkpoint_dir/yolo2_coco.ckpt"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        bboxes, obj_probs, class_probs = sess.run(detection_results, feed_dict={images: image_cp})

    bboxes, scores, class_inds = postprocess(bboxes, obj_probs, class_probs,
                                             image_shape=image_shape)
    img_detection = draw_detection(image, bboxes, scores, class_inds, class_names) #回归框，得到矩阵框的X左右，Y下像素坐标
    print('\n')

    # #检测车道线的像素，并放入字典中
    # lane_cor = {}
    # #手动选取ROI，待修改
    # vertices = np.array([[(110, 194), (110, 0), (150, 0), (150, 194)]], dtype=np.int32)
    # roi = region_of_interest(line_img, vertices)
    # # cv2.imshow("roi", roi)
    # for i in range(0, (roi.shape)[0]):
    #     for j in range(0, (roi.shape)[1]):
    #         if roi[i, j, 2] == 255:         #roi[i,j,num]这里num代表着BGR第几通道
    #             lane_cor[i] = j
    # print("The coodinate of the detected_lane y：x")
    #
    # print(lane_cor)
    #
    # global box
    # if (utils.box[0] + m * (utils.box[2] - utils.box[0])) <= lane_cor[utils.box[3]] <= (utils.box[2] - m * (utils.box[2] - utils.box[0])):
    #     print("The car is on the solid line!!!")
    # else:
    #     print("The car is permitted~")

    # mix1 = weight_add(img_detection, line_img, alpha=0.7, belta=1, gamma=0.)
    # mixed = weight_add(img_detection,roi , alpha=0.7, belta=1, gamma=0.)
    # cv2.imshow("mix1", mix1)
    # cv2.imshow("mixed",mixed)
    #
    cv2.imshow("detection results", img_detection)
    cv2.imwrite("/home/zdq/PycharmProjects/YOLOv2/detection.jpg", img_detection)
    cv2.waitKey(0)
    return img_detection


if __name__ == '__main__':
    main()