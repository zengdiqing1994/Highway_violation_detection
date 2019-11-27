wechat ： zengdiqing94

# Introduction

感谢大家关注这个项目，这个模块是在无人机航拍视频中识别路面车辆是否有轧线和违章越线的情况出现，轧线算法已经在上一级目录的push_lane文件夹中提出，现阶段是需要通过车辆轧线触发deepsort算法跟踪车辆，在时序的过程当中判断车辆是否有违章越线的行为，以下是参考的一些代码：

  https://github.com/nwojke/deep_sort
  
  https://github.com/qqwweee/keras-yolo3
  
  https://github.com/Qidian213/deep_sort_yolov3

# Quick Start

1. 因为Deepsort跟踪算法基于YOLOv3检测而来，使用的darknet53框架训练得到的detection框进行解码。

# Dependencies

代码基于Python 2.7，跟踪器需要下面几项

    NumPy
    sklean
    OpenCV
    Pillow

然后就是 TensorFlow-1.4.0.



# Note 

 model_data中veri.pb是deepsort中比较关键的模型，车辆特征模型，需要tensorflow-1.4.0
 
# Test only


 

