#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#from ctypes.wintypes import PSIZE
from os.path import exists, join
import json
import numpy as np
import cv2
from torch.utils.data import Dataset

from .groupcolorjitter import GroupColorJitter

from tempatteditor.config import LABEL_FILE

"""SomethingSomethingV2 Dataset Class

Note:
    We assume that the channel order of loaded video frames is [B, G, R] based on
    the original implementation of ST-ABN and TPN repository. Especially, we use the
    color jitter implementation of the original TPN (mmaction), which assumes the
    input frame is BGR channel order.
"""

def crop_frames(frame, start_frame):
        # NOTE: expected input data shape: [H, W, C] ndarray
        #start_ = start_frame.shape
        return frame[start_frame:start_frame+32, start_frame:start_frame+32, :]


def normalize(frame):
    # NOTE: expected input data shape: [T, H, W, C] ndarray

    ### mean and std with RGB order
    #MEAN = [123.675, 116.28, 103.53]
    #STD = [58.395, 57.12, 57.375]
    ### mean and std with BGR order
    MEAN = [103.53, 116.28, 123.675]
    STD = [57.375, 57.12, 58.395]

    # frames = []
    # for img in frame:
    #     img = img.astype(np.float32)
    #     #cv2.imshow("aaa", img)
    #     #key = cv2.waitKey(1)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     #cv2.imshow("aaa", img)
    #     #key = cv2.waitKey(1)
    #     img = (img - MEAN) / STD
    #     frames.append(img)

    # frames = np.array(frames)

    frames = (frame - MEAN) / STD
    return frames


def augmentation(frame):
    # expected input data shape: [T, H, W, C] ndarray

    # horizontal flip
    if np.random.rand(1)[0] > 0.5:
        frame = np.flip(frame, axis=1)

    # Color Jitter
    colorjitter = GroupColorJitter(color_space_aug=True)
    frame = colorjitter(frame)

    return frame