import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import *

import torch
import torch.optim as optim
import os

from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *
from fastai.core import *
from fastai.widgets import *

import threading
from threading import Thread, Lock
import cv2
import warnings
import datetime
import time

class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.capture.read()
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        # Start the thread to read frames from the video stream
       
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        # self.thread2 = Thread(target=self.fastai, args=())
        # self.thread2.start()
        # print(threading.active_count())
        # print(threading.enumerate())


    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                
            
            time.sleep(.001)

    def show_frame(self):
        
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def fastai(self):
        # Display frames in main program
        while True: 
            img_cv2 = self.frame
            self.show_frame()
            # ndarry to fastai.Image
            img_fastai = Image(pil2tensor(img_cv2, dtype=np.float32).div_(255))
            # now = time.time()
            pred,pred_idx,probs = learn_inf.predict(img_fastai)
            lbl_pred = widgets.Label()
            lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
            print(lbl_pred.value)

if __name__ == '__main__':
    base_dir = '/home/james/PolidaDemo'
    path = Path(base_dir + '/imgsA41')  # imgs
    # path = Path(base_dir + '/imgsA')  # imgs
    os.chdir(path)

    warnings.filterwarnings("ignore", category=UserWarning,module="torch.nn.functional")

    learn_inf = load_learner(path, 'ok_imgsA41_bs10_squeezenet1_0_512_J9.pkl')  # 啟用微調後的模型export.pkl # 3.2 sec loading
    # learn_inf = load_learner(path, 'export.pkl')  # 啟用微調後的模型export.pkl # 3.2 sec loading


    video_stream_widget = VideoStreamWidget()
    while True:
        try:
            video_stream_widget.fastai()
        except AttributeError:
            pass