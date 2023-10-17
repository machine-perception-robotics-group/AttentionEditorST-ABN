#!/usr/bin/env python
# -*- coding: utf-8 -*-

# network_view.py
# Copyright (c) Tsubasa Hirakawa, 2021


import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from sklearn.feature_extraction import img_to_graph

import torch

from config_files.stabn.models import MODEL_NAMES, load_model
from config_files.stabn.datasets.ssv2 import crop_frames, normalize, augmentation
from config_files.stabn.utils import load_checkpoint, load_args

from ..config import CHECKPOINT_FILE, CONFIG_FILE, DEFAULT_DATA_DIR, TRAIN_ARGS_FILE
from ..smth_cls import CLASS_LABEL


class NetworkView(QtWidgets.QWidget):
    """ST-ABNのネットワークモデルを準備して，推論しその認識結果を表示するウィジェット"""

    def __init__(self):
        """
        ウィンドウの初期化（クラスインスタンスの初期化）
        ネットワークモデルの準備
        分類結果を表示するためのウィンドウの準備
        """
        super(NetworkView, self).__init__()

        # クラスラベルの取得
        self.CLASS_LABEL = CLASS_LABEL
        self.n_class = len(self.CLASS_LABEL)

        # ネットワークモデルの準備 --------------

        # load train args
        self.TRAIN_ARGS_FILE = TRAIN_ARGS_FILE
        self.train_args = load_args(self.TRAIN_ARGS_FILE)

        # network model
        #_is_abn = True if "abn_" in self.train_args['model'] else False
        self.model = load_model(
            model_name = self.train_args['model'], num_classes=self.n_class,
            sample_size = self.train_args['frame_size'], sample_duration=self.train_args['frame_length'],
            dout_ratio = self.train_args['dout_ratio'], pretrain_2d=False
        )

        ### load checkpoint
        self.model.eval()
        self.CHECKPOINT_FILE = CHECKPOINT_FILE
        print("    load checkpont:", self.CHECKPOINT_FILE)
        try:
            self.model, _, _, _, _, _ = load_checkpoint(self.CHECKPOINT_FILE, self.model, None, device='cpu')
            _is_checkpoint_loaded = True
        except:
            print("      Failed to load checkpoint trained by DDP. Try to load after setting the network as Data Parallel mode.")
            _is_checkpoint_loaded = False

        if not _is_checkpoint_loaded:
            self.model, _, _, _, _, _ = load_checkpoint(self.CHECKPOINT_FILE, self.model, None, device='cpu')[0]
        # ---------------------------------------

        # button
        self.button_infer = QtWidgets.QPushButton("inference")
        self.button_infer.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                        QtWidgets.QSizePolicy.Fixed)
        self.button_infer_whole = QtWidgets.QPushButton("whole inference")
        self.button_infer_whole.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                        QtWidgets.QSizePolicy.Fixed)
        self.textbox = QtWidgets.QLineEdit()

        # 分類結果を表示するためのウィジェットの準備
        self.n_display = 100
        if self.n_display < self.n_class:
            self.n_display = self.n_class
        self.score_title = QtWidgets.QLabel("Top-%d results" % self.n_display)
        self.score_model = QtGui.QStandardItemModel(0, 4)
        self.score_model.setHeaderData(0, QtCore.Qt.Horizontal, 'rank')
        self.score_model.setHeaderData(1, QtCore.Qt.Horizontal, 'score')
        self.score_model.setHeaderData(2, QtCore.Qt.Horizontal, 'index')
        self.score_model.setHeaderData(3, QtCore.Qt.Horizontal, 'name')
        self.score = QtWidgets.QTreeView()
        self.score.setModel(self.score_model)
        self.score.setMinimumWidth(400)

        # layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.score_title)
        layout.addWidget(self.score)
        layout.addWidget(self.button_infer)
        layout.addWidget(self.button_infer_whole)
        layout.addWidget(self.textbox)

        self.setLayout(layout)


    def infer(self, video, temp_att, correct_label):
        """引数で入力された動画フレームとattention mapから推論"""

        #推論開始フレームの取得
        if len(self.textbox.text()) is 0:
            start_frame = int((len(video)- 32 + 1)/2 )
        else:
            start_frame = int(self.textbox.text())

        #32フレーム分の動画フレームを取得
        video = video[start_frame:start_frame+32, :, :, :]

        #動画フレームの前処理
        video = self.preprocess_video(video)

        if temp_att is not None:
            #32フレーム分のtemporal attentionの値を取得
            temp_att = torch.from_numpy(temp_att[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]).to(torch.float32)
            temp_att = temp_att[:, :, start_frame:start_frame+32, :, :]
            with torch.no_grad():
                    output_per, _, s_att, t_att = self.model(np.squeeze(video, 0), temp_att)
                    output_per = output_per.numpy()

        else:
            with torch.no_grad():
                    output_per, _, s_att, t_att = self.model(np.squeeze(video, 0))
                    output_per = output_per.numpy()

        self.update_score(output_per, correct_label)

        return output_per, s_att, t_att


    def infer_whole(self, video, temp_att, correct_label):
        """Temporal attentionを付与"""
        video = self.preprocess_video(video)

        if temp_att is not None:
            temp_att = torch.from_numpy(temp_att[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]).to(torch.float32)

        num_frames = video.size()[3]
        whole_frames = torch.zeros(1, 1, num_frames, 1, 1, dtype=torch.float32)

        for i in range(num_frames // 32):
            video_clip = video[:, :, :, i*32:i*32+32, :, :]

            ### evaluation
            with torch.no_grad():
                _, _, _, t_att = self.model(np.squeeze(video_clip, 0))

            whole_frames[:, :, i*32:i*32+32,: ,:] = t_att

        if num_frames % 32 != 0:
            video_clip = video[:, :, :, num_frames-32:num_frames, :, :]

            ### evaluation
            with torch.no_grad():
                _, _, _, t_att = self.model(np.squeeze(video_clip, 0))

            whole_frames[:, :,num_frames-32:num_frames, :, :] = t_att

        return whole_frames


    def preprocess_video(self, video):
        """入力された動画フレームの前処理"""
        # preprocessing for video frames
        img_group = video   #shape -> (T,H,W,C)

        # normalize (THWC)
        img_group = normalize(img_group)
        # transpose frames (THWC --> TCHW)
        img_group = img_group.transpose(0, 3, 1, 2)
        # Stack into numpy.array
        img_group = np.stack(img_group, axis=0)
        img_group = img_group.transpose(1, 0, 2, 3)
        img_group = img_group[np.newaxis, np.newaxis, :]

        # convert into Torch.Tensor
        img_group = torch.from_numpy(img_group).to(torch.float32)

        return img_group


    def update_score(self, input_score, correct_label):
        """認識結果のスコアの表示・更新"""
        _prob = np.squeeze(input_score)
        _ordered_index = np.argsort(-input_score)

        self.clear_score()
        for i in range(self.n_display):
            _index = _ordered_index[0, i]
            item_rank = QtGui.QStandardItem(str(i + 1))
            item_prob = QtGui.QStandardItem("%f" % _prob[_index])
            item_num = QtGui.QStandardItem("%04d" % _index)
            item_label = QtGui.QStandardItem(self.CLASS_LABEL[_index])
            if _index == correct_label:
                item_prob.setForeground(QtGui.QBrush(QtGui.QColor("#ff0000")))
                item_num.setForeground(QtGui.QBrush(QtGui.QColor("#ff0000")))
                item_label.setForeground(QtGui.QBrush(QtGui.QColor("#ff0000")))
            self.score_model.setItem(i, 0, item_rank)
            self.score_model.setItem(i, 1, item_prob)
            self.score_model.setItem(i, 2, item_num)
            self.score_model.setItem(i, 3, item_label)


    def clear_score(self):
        """現在表示されている認識結果のスコア部分の初期化"""
        self.score_model.removeRows(0, self.score_model.rowCount())


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = NetworkView()
    widget.setGeometry(100, 100, 300, 300)
    widget.show()
    sys.exit(app.exec_())
