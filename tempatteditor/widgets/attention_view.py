#!/usr/bin/env python
# -*- coding: utf-8 -*-

# attention_view.py
# Copyright (c) Tsubasa Hirakawa, 2021


import numpy as np
import cv2

from PyQt5 import QtWidgets, QtGui, QtCore

from ..config import IMAGE_SIZE, NUM_FRAMES, NUM_COLS, DISPLAY_IMG_SIZE


class TempAttImage(QtWidgets.QLabel):
    """1枚の動画フレームとそのTemporal Attentioinを表示するウィジェット"""

    def __init__(self, name):
        """
        ウィンドウの初期化（クラスインスタンスの初期化）
        動画フレームとその時のattentionの値を可視化した色を表示するように初期化
        最初は動画が選択されていないので，真っ黒の画像を作成して表示
        """
        super(TempAttImage, self).__init__()

        self.name = name             # この画像フレームウィンドウの名前（必要ないけど確認用に残している）
        self.ATT_BAR_SIZE = 20       # temporal attentionを表示するカラーバーの厚み（高さ）
        self.edit_value = 1.0        # ダブルクリック時にattentionを修正する時の値（デフォルト値．下の関数などで適宜変更）
        self.attention_value = None  # temporal attentionの初期値（最初はデータがないのでNoneで初期化）

        # 動画フレームを保存するNumpy配列
        self.image = np.ones([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8) * 255

        # 画像とtemporal attentionの最初の表示
        self.update()

    def update(self):
        """現在保有している動画フレームとtemporal attentionの値から画像を表示"""
        att_bar = self._make_attention_bar()
        _image = cv2.cvtColor(np.vstack((att_bar, self.image)), cv2.COLOR_BGR2RGB)

        qimg = QtGui.QImage(
            _image,
            IMAGE_SIZE,
            IMAGE_SIZE + self.ATT_BAR_SIZE,
            3 * IMAGE_SIZE,
            QtGui.QImage.Format_RGB888
        )

        self.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(DISPLAY_IMG_SIZE, DISPLAY_IMG_SIZE, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    def set_image(self, image, att_value=None):
        """動画フレームとtemporal attentionの値を書き換え（更新）--> その後updateで新しいものを表示"""
        self.image = image
        self.attention_value = att_value
        self.update()

    def set_attention(self, att_value=None):
        """temporal attentionの値を書き換え（更新）--> その後updateで新しいものを表示"""
        self.attention_value = att_value
        self.update()

    def set_edit_value(self, e_val):
        """temporal attentionを修正する際の値の変更"""
        self.edit_value = e_val

    def mouseDoubleClickEvent(self, QMouseEvent):
        """
        フレームをダブルクリックした際にtemporal attentionの値を修正
        self.edit_valueに指定している値に書き換え
        """
        self.attention_value = self.edit_value
        self.update()

    def _make_attention_bar(self):
        """動画フレームを表示する際のattenitonの値を可視化したカラーバーの作成（self.update()で使います）"""
        if self.attention_value is None:
            _bar = np.ones([self.ATT_BAR_SIZE, IMAGE_SIZE, 3], dtype=np.uint8) * 0
        else:
            _bar = np.ones([self.ATT_BAR_SIZE, IMAGE_SIZE], dtype=np.float) * self.attention_value * 255
            _bar = cv2.applyColorMap(_bar.astype(np.uint8), cv2.COLORMAP_JET)
        return _bar

    def clear_image(self):
        """別の動画を読み込んだ際に一度動画フレーム画像を真っ白に戻す"""
        self.image = np.ones([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8) * 255
        self.attention_value = None
        self.update()


class TemporalAttentionView(QtWidgets.QWidget):
    """Temporal Attentionを表示，編集するためのウィンドウウィジェット"""

    def __init__(self):
        """
        ウィンドウの初期化（クラスインスタンスの初期化）
        指定した数のフレーム数このTempAttImage（上のやつ）を準備
        その他のボタンなども準備して配置・表示
        """
        super(TemporalAttentionView, self).__init__()

        self.num_frames = None

        # TempAttImage（上のやつ）を必要なフレーム数分作成（リストに格納）
        self.att_img_list = [TempAttImage(name=str(i)) for i in range(NUM_FRAMES)]

        # radio button
        self.button_add = QtWidgets.QRadioButton("red_1.0")
        self.button_middle = QtWidgets.QRadioButton("green_0.5")
        self.button_remove = QtWidgets.QRadioButton("blue_0.0")
        self.button_none = QtWidgets.QRadioButton("black_None")
        self.button_add.toggled.connect(self.on_clicked)
        self.button_middle.toggled.connect(self.on_clicked)
        self.button_remove.toggled.connect(self.on_clicked)
        self.button_none.toggled.connect(self.on_clicked)
        self.button_add.setChecked(True)

        #add fill attention button
        self.button_fill = QtWidgets.QPushButton('Fill Attention')
        self.button_fill.clicked.connect(self.fill_attention)

        #add reset attention button
        self.button_reset = QtWidgets.QPushButton('Reset Attention')
        self.button_reset.clicked.connect(self.reset_attention)

        # save button
        self.button_save = QtWidgets.QPushButton('Save Attention')

        # layout (各ウィンドウやボタンの配置の指定)
        image_layout = QtWidgets.QGridLayout()
        for i in range(NUM_FRAMES):
            image_layout.addWidget(self.att_img_list[i], i // NUM_COLS, i % NUM_COLS)
        image_layout.setContentsMargins(0, 0, 0, 0)

        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addWidget(self.button_add)
        bottom_layout.addWidget(self.button_middle)
        bottom_layout.addWidget(self.button_remove)
        bottom_layout.addWidget(self.button_none)
        bottom_layout.addWidget(self.button_reset)
        self.button_reset.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                        QtWidgets.QSizePolicy.Fixed)
        bottom_layout.addWidget(self.button_fill)
        self.button_fill.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                        QtWidgets.QSizePolicy.Fixed)
        bottom_layout.addWidget(self.button_save)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(image_layout)
        layout.addLayout(bottom_layout)
        self.setLayout(layout)

    def set_video_frames(self, video_frames):
        """動画フレームを一度リセット"""
        for i in range (NUM_FRAMES):
            self.att_img_list[i].clear_image()
        """動画フレームをTempAttImageに送って，表示"""
        self.num_frames = video_frames.shape[0]
        for i in range(video_frames.shape[0]):
            self.att_img_list[i].set_image(video_frames[i], att_value=None)

    def set_temporal_attention(self, temp_atts, start_frame = 0):
        """Temporal AttentionをTempAttImageに送って，attenitonの値を更新・表示"""
        for i in range(temp_atts.shape[0]):
            self.att_img_list[start_frame+i].set_attention(temp_atts[i])

    def get_temporal_attention(self):
        """TempAttImageが持っているtemporal attentionの値を取得して返す"""
        _tmp_att = np.zeros(self.num_frames, dtype=np.float32)
        for i in range(self.num_frames):
            _ta = self.att_img_list[i].attention_value
            if _ta is None:
                return None
            _tmp_att[i] = _ta
        return _tmp_att

    def on_clicked(self):
        """
        add, removeのラジオボタンがクリックされたらattentionを修正する値を変更・更新
            add:    1.0に更新
            middle: 0.5に更新
            remove: 0.0に更新
            none:   Noneに更新
        """
        if self.button_add.isChecked():
            _e_val = 1.0
        elif self.button_middle.isChecked():
            _e_val = 0.5
        elif self.button_remove.isChecked():
            _e_val = 0.0
        else:
            _e_val = None
        for i in range(len(self.att_img_list)):
            self.att_img_list[i].set_edit_value(_e_val)

    def fill_attention(self):
        """attentionをまとめて付与"""
        self.first_att = None
        initial = True
        for i in range(self.num_frames):
            _ta = self.att_img_list[i].attention_value
            if _ta is not None:
                if initial:
                    self.first_att = _ta
                    self.first_frame = i
                    initial = False
                else:
                    self.second_att = _ta
                    self.second_frame = i
                    if self.first_att == self.second_att:
                        """同じ値のattentionで挟まれたフレームには同じ値を付与"""
                        for j in range(self.first_frame, self.second_frame):
                            self.att_img_list[j].set_attention(_ta)
                    self.first_att = self.second_att
                    self.first_frame = self.second_frame

    def reset_attention(self):
        """temoiral attentionの値をリセット"""
        for i in range(self.num_frames):
            self.att_img_list[i].set_attention(None)



if __name__ == '__main__':
    # デバッグ用の実行部分
    # デバッグ用実行方法: main.pyのあるフォルダに移動して次のコマンドを実行
    # python3 tempatteditor.widgets.attention_view

    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = TemporalAttentionView()
    widget.show()
    sys.exit(app.exec_())
