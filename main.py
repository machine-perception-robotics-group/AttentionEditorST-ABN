#!/usr/bin/env python
# -*- coding: utf-8 -*-

# main.py
# Copyright (c) Tsubasa Hirakawa, 2021


import os
import cv2
from PyQt5 import QtWidgets, QtCore
from numpy.lib.npyio import save
from tempatteditor.config import LABEL_FILE, NUM_FRAMES, SAVE_DIR
from tempatteditor.widgets import *
from tempatteditor.utils import *

class MainWidget(QtWidgets.QWidget):

    def __init__(self):
        super(MainWidget, self).__init__()

        # variables -------------------
        self.video_filename = None
        self.video_frames = None
        self.correct_label = None

        # define widgets --------------
        self.file_view = FileView()
        self.video_view = VideoView()
        self.temp_att_view = TemporalAttentionView()
        self.network_view = NetworkView()

        # functions -------------------
        self.file_view.tree.doubleClicked.connect(self.load_video)
        self.network_view.button_infer.clicked.connect(self.inference)
        self.network_view.button_infer_whole.clicked.connect(self.inference_whole)
        self.temp_att_view.button_save.clicked.connect(self.save_attention)

        #load label
        with open(LABEL_FILE, "r") as f:
            data = f.readlines()
        self.labeldict = {}
        for d in data:
            name, _ , label = d.split(" ")
            self.labeldict[name] = int(label)

        # layout ----------------------
        # upper widgets
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addWidget(self.file_view)
        top_layout.addWidget(self.video_view)
        top_layout.addWidget(self.network_view)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.temp_att_view)
        self.setLayout(layout)

    def load_video(self):
        #get psth of select_file
        self.video_filename = self.file_view.select_file()
        if self.video_filename is not None:
            # get video file name
            self.video_view.open_file(self.video_filename)
            # TODO: how to decide trimmed frames ???
            # reset video view (load video frames; trimmed center)
            self.video_frames = load_video_frames(self.video_filename)
            # reset attention view
            self.temp_att_view.set_video_frames(self.video_frames)
            # reset network view
            self.network_view.clear_score()
            base_name, _ = os.path.splitext(os.path.basename(self.video_filename))
            self.correct_label = self.labeldict[base_name]

    def inference(self):
        if self.video_filename is not None:

            # inference
            _, _, t_att = self.network_view.infer(
                self.video_frames,
                self.temp_att_view.get_temporal_attention(),
                self.correct_label
            )

            # update temporal attention values
            if len(self.network_view.textbox.text()) is 0:
                start_frame = int((len(self.video_frames)- 32 + 1)/2 )
            else:
                start_frame = int(self.network_view.textbox.text())
            self.temp_att_view.set_temporal_attention(t_att.numpy().copy().squeeze(), start_frame)

    def inference_whole(self):
        if self.video_filename is not None:

            # inference
            t_att = self.network_view.infer_whole(
                self.video_frames,
                self.temp_att_view.get_temporal_attention(),
                self.correct_label
            )

            # update temporal attention values
            self.temp_att_view.set_temporal_attention(t_att.numpy().copy().squeeze())

    def save_attention(self):
        # load edited attention map
        _save_att = self.temp_att_view.get_temporal_attention()

        if _save_att is not None:
            # get save file name
            _basename, _ = os.path.splitext(os.path.basename(self.video_filename))
            class_name = str(self.correct_label)
            save_att_dir = os.path.join(SAVE_DIR, class_name)
            os.makedirs(save_att_dir, exist_ok=True)
            _default_save_filename = os.path.join(save_att_dir, _basename + ".npy")

            _selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                'Save Lattice', _default_save_filename, "Numpy Array (*.npy)"
            )

            # save attention map
            np.save(_selected_path, _save_att)


if __name__ == '__main__':

    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = MainWidget()
    widget.setGeometry(50, 50, 1000, 700)
    widget.show()
    sys.exit(app.exec_())
