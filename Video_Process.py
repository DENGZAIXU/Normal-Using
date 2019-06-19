import os
import cv2
import numpy as np
import sys

from YOLO3 import YOLO3


import time

class Video(object):
    def __init__(self):
        self.vdo = cv2.VideoCapture()
        self.yolo3 = YOLO3("YOLO3/cfg/yolov3-tiny.cfg",
                           "YOLO3/yolov3-tiny.weights",
                           "YOLO3/cfg/coco.names",
                           is_xywh=True)
        self.write_video = True

    def open(self, video_path):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("demo.avi", fourcc, 20, (self.im_width, self.im_height))
        return self.vdo.isOpened()

    def process(self):

        while self.vdo.grab():
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            ## 处理图像
            bbox, cls_conf, cls_ids = self.yolo3(ori_im)

            end = time.time()
            print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))

            cv2.imshow("test", ori_im)
            cv2.waitKey(1)

            if self.write_video:  # 是否保存
                self.output.write(ori_im)


if __name__ == "__main__":

    if len(sys.argv) == 1: # 从摄像头获取
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("test", 800, 600)
        det = Video()
        det.open(0)
        det.process()

    else:   # 处理视频序列
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("test", 800, 600)
        det = Video()
        open_file = r"C:\Users\Xu\Desktop\demo.mp4"
        # open_file = sys.argv[1]

        det.open(open_file)
        det.process()