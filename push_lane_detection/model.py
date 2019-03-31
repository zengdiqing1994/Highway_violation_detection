"""
YOLOv2 implemented by Tensorflow, only for predicting，网络模型，darknet19
"""
import os

import numpy as np
import tensorflow as tf



######## basic layers 基础层：conv/pool/reorg(带passthrough的重组层) #######

#激活函数
def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu") #或者tf.maximum(0.1*x,x)

# Conv2d+BN:yolo2中每一个卷积层后面都有一个BN层
def conv2d(x, filters, size, pad=0, stride=1, batch_normalize=1,
           activation=leaky_relu, use_bias=False, name="conv2d"):
    # padding,注意：不用padding=“same”，否则可能会导致坐标计算错误
    if pad > 0:
        x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    # 有BN层，所以后面有BN层的conv就不用偏置bias，并先不经过激活函数activition
    out = tf.layers.conv2d(x, filters, size, strides=stride, padding="VALID",
                           activation=None, use_bias=use_bias, name=name)
    # BN，如果有，应该在卷积层conv和激活函数activation之间
    if batch_normalize == 1:
        out = tf.layers.batch_normalization(out, axis=-1, momentum=0.9,
                                            training=False, name=name+"_bn")
    if activation:
        out = activation(out)
    return out

# maxpool2d
def maxpool(x, size=2, stride=2, name="maxpool"):
    return tf.layers.max_pooling2d(x, size, stride)

# reorg layer（带passthrough的重组层）
def reorg(x, stride):
    return tf.extract_image_patches(x, [1, stride, stride, 1],
                        [1, stride, stride, 1], [1,1,1,1], padding="VALID")

###########Darknet19####################
# 默认是coco数据集，最后一层维度是anchor_num*(class_num+5)=5*(80+5)=425
# 如果训练数据是Pasca voc，最后一层维度就是5*(20+5)=125
# 如果只有一类car，最后一层就是5×（1+5）=30
def darknet(images, n_last_channels=425):
    """Darknet19 for YOLOv2"""
    net = conv2d(images, 32, 3, 1, name="conv1")
    net = maxpool(net, name="pool1")
    net = conv2d(net, 64, 3, 1, name="conv2")
    net = maxpool(net, name="pool2")
    net = conv2d(net, 128, 3, 1, name="conv3_1")
    net = conv2d(net, 64, 1, name="conv3_2")
    net = conv2d(net, 128, 3, 1, name="conv3_3")
    net = maxpool(net, name="pool3")
    net = conv2d(net, 256, 3, 1, name="conv4_1")
    net = conv2d(net, 128, 1, name="conv4_2")
    net = conv2d(net, 256, 3, 1, name="conv4_3")
    net = maxpool(net, name="pool4")
    net = conv2d(net, 512, 3, 1, name="conv5_1")
    net = conv2d(net, 256, 1, name="conv5_2")
    net = conv2d(net, 512, 3, 1, name="conv5_3")
    net = conv2d(net, 256, 1, name="conv5_4")
    net = conv2d(net, 512, 3, 1, name="conv5_5")
    shortcut = net #存储这一层呢个特征图。以便后面passthrough层
    net = maxpool(net, name="pool5")
    net = conv2d(net, 1024, 3, 1, name="conv6_1")
    net = conv2d(net, 512, 1, name="conv6_2")
    net = conv2d(net, 1024, 3, 1, name="conv6_3")
    net = conv2d(net, 512, 1, name="conv6_4")
    net = conv2d(net, 1024, 3, 1, name="conv6_5")
    # ---------
    net = conv2d(net, 1024, 3, 1, name="conv7_1")
    net = conv2d(net, 1024, 3, 1, name="conv7_2")
    # shortcut增加挨了一个中间卷积层，先采用64个1×1卷积核进行卷积，然后再进行passthrough处理
    # 这样26×26×512——》26×26×64——》13×13×256的特征图
    shortcut = conv2d(shortcut, 64, 1, name="conv_shortcut")
    shortcut = reorg(shortcut, 2)
    net = tf.concat([shortcut, net], axis=-1)   #channel整合在一起
    net = conv2d(net, 1024, 3, 1, name="conv8")

    # detection layer：最后用一个1×1卷积去调整channel，该层没有BN层和激活函数
    net = conv2d(net, n_last_channels, 1, batch_normalize=0,
                 activation=None, use_bias=True, name="conv_dec")
    return net



if __name__ == "__main__":
    x = tf.random_normal([1, 416, 416, 3])
    model = darknet(x)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 必须线restore模型才能打印shape;导入模型时，上面每层网络的那么不能修改，否则找不到
        saver.restore(sess, "./checkpoint_dir/yolo2_coco.ckpt")
        print(sess.run(model).shape)#（1,13,13,425）

