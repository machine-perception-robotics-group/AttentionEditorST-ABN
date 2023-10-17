#!/usr/bin/env python
# -*- coding: utf-8 -*-

# utils.py
# Copyright (c) Tsubasa Hirakawa, 2020


import numpy as np
import cv2
from tempatteditor.config import IMAGE_SIZE


def center_crop(image, size=224):
    _h, _w, _c = image.shape
    _h_min = int(_h / 2) - int(size / 2)
    _h_max = int(_h / 2) + int(size / 2)
    _w_min = int(_w / 2) - int(size / 2)
    _w_max = int(_w / 2) + int(size / 2)
    _dst = image[_h_min:_h_max, _w_min:_w_max, :]
    return _dst


def load_video_frames(video_filename):
    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        return False

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    video_frames = np.zeros([count, IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)
    for i in range(count):
        _, frame = cap.read()
        video_frames[i, :] = center_crop(frame, size=IMAGE_SIZE)
    return video_frames


def softmax(x):
    x_max = np.max(x)
    x = np.exp(x - x_max)
    u = np.sum(x)
    return x / u


def min_max(x, axis=None):
    min = 0  #x.min(axis=axis, keepdims=True)
    max = 1  #x.max(axis=axis, keepdims=True)
    result = (x - min) / (max - min)
    return result
