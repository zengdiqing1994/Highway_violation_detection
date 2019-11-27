# coding=utf-8

# 全局变量
VIDEO_PATH = '/home/zdq/PycharmProjects/deep_sort_yolov3-master/DJI_cut.mp4' # 视频地址
EXTRACT_FOLDER = '/home/zdq/PycharmProjects/deep_sort_yolov3-master/pic' # 存放帧图片的位置
EXTRACT_FREQUENCY = 15 # 帧提取频率


def extract_frames(video_path, dst_folder, index):
    # 主操作
    import cv2
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video")
        exit(1)
    count = 1
    while True:
        _, frame = video.read()
        if frame is None:
            break
        if count % EXTRACT_FREQUENCY == 0:
            save_path = "{}/{:>03d}.jpg".format(dst_folder, index)
            cv2.imwrite(save_path, frame)
            index += 1
        count += 1
    video.release()
    # 打印出所提取帧的总数
    print("Totally save {:d} pics".format(index-1))


def main():
    # 递归删除之前存放帧图片的文件夹，并新建一个
    import shutil
    try:
        shutil.rmtree(EXTRACT_FOLDER)
    except OSError:
        pass
    import os
    os.mkdir(EXTRACT_FOLDER)
    # 抽取帧图片，并保存到指定路径
    extract_frames(VIDEO_PATH, EXTRACT_FOLDER, 1)


if __name__ == '__main__':
    main()
