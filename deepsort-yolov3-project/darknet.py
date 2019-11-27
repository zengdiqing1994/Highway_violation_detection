from ctypes import *
import math
import random
import cv2
import time
# -*-coding:utf-8-*-


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/zdq/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

# make_boxes = lib.make_boxes
# make_boxes.argtypes = [c_void_p]
# make_boxes.restype = c_int

# num_boxes = lib.num_boxes
# num_boxes.argtypes = [c_void_p]
# num_boxes.restype = c_int
#
# make_probs = lib.make_probs
# make_probs.argtypes = [c_void_p]
# make_probs.restype = POINTER(POINTER(c_float))

#def nparray_to_image(img):
    #data = img.ctypes.data_as(POINTER(c_ubyte))
    #image = ndarray_image(data, img.ctype.shape, img.ctypes.strides)
    #return image
# class Darknet(object):
def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    #im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                new_x = int(b.x-(b.w/2))
                new_y = int(b.y-(b.h/2))
                new_w = int(b.w)
                new_h = int(b.h)
                res.append((new_x, new_y, new_w, new_h))
    res = sorted(res, key=lambda x: -x[1])
    # print()
    # free_image(im)
    #free_detections(dets, num)
    return res

#     def yolo_results_to_boxes(boxes):
#         boxs = [x[2] for x in boxes]
#         results = []
#         for box in boxs:
#             x, y, w, h = box
#             new_x = int(x-(w/2))
#             new_y = int(y-(h/2))
#             new_w = int(w)
#             new_h = int(h)
#             results.append([new_x, new_y, new_w, new_h])
#         return results

# def array_to_image(arr):
#     arr = arr.transpose(2, 0, 1)
#     c = arr.shape[0]
#     h = arr.shape[1]
#     w = arr.shape[2]
#     arr = (arr/255.0).flatten()
#     data = c_array(c_float, arr)
#     im = IMAGE(w,h,c,data)
#     return im

def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image

if __name__ == "__main__":
    net = load_net("/home/zdq/darknet/cfg/yolov3-voc.cfg", "/home/zdq/darknet/backup/yolov3-voc_20000.weights", 0)
    meta = load_meta("/home/zdq/darknet/cfg/voc.data")
    vid = cv2.VideoCapture('DJI_cut.mp4')

    while True:
        return_value,arr=vid.read()
        if not return_value:
            break
        im = nparray_to_image(arr)
        boxes = detect(net, meta, im)
    # def yolo_results_to_boxes(boxes):
    #     boxs = [x[2] for x in boxes]
    #     results = []
    #     for box in boxs:
    #         x, y, w, h = box
    #         new_x = int(x-(w/2))
    #         new_y = int(y-(h/2))
    #         new_w = int(w)
    #         new_h = int(h)
    #         results.append([new_x, new_y, new_w, new_h])
    #     return results

    #     for i in range(len(boxes)):
    #         score = boxes[i][1]
    #         label = boxes[i][0]
    #         xmin = boxes[i][2][0] - boxes[i][2][2]/2
    #         ymin = boxes[i][2][1] - boxes[i][2][3]/2
    #         xmax = boxes[i][2][0] + boxes[i][2][2]/2
    #         ymax = boxes[i][2][1] + boxes[i][2][3]/2
    #         cv2.rectangle(arr, (int(xmin), int(ymin)), (int(xmax),int(ymax)), (0,255,0), 1)
    #         cv2.putText(arr,str(label),(int(xmin),int(ymin)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,
    #                     color = (0,255,255),thickness=1)
    #     cv2.imshow("car",arr)
    #     cv2.waitKey(1)
    # #r = detect(net, meta, "data/dog.jpg")
    # #print r
    # cv2.destroyAllWindows()

