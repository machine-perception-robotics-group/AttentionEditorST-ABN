#!/usr/bin/env python
# -*- coding: utf-8 -*-

# video_view.py
# Copyright (c) Tsubasa Hirakawa, 2021


from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5 import QtWidgets
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget


class VideoView(QtWidgets.QWidget):
    """選択した動画を再生するウィジェット"""

    def __init__(self):
        """
        ウィンドウの初期化（クラスインスタンスの初期化）
        動画表示部分や再生ボタンを初期化して配置・表示
        """
        super(VideoView, self).__init__()

        # header (title)
        self.view_title = QtWidgets.QLabel("Video stream")

        # 動画表示部分
        self.media_playper = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video_widget = QVideoWidget()
        self.video_widget.setFixedSize(320, 240)

        # play button
        self.play_button = QtWidgets.QPushButton()
        self.play_button.setEnabled(False)
        self.play_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play)

        # slider
        self.position_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.setPosition)

        # layout (各ウィンドウやボタンの配置の指定)
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.position_slider)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.view_title)
        layout.addWidget(self.video_widget)
        layout.addLayout(control_layout)

        self.setLayout(layout)

        # 動画再生部分の機能の接続（再生ボタンを押すと再生開始など）
        self.media_playper.setVideoOutput(self.video_widget)
        self.media_playper.stateChanged.connect(self.media_state_changed)
        self.media_playper.positionChanged.connect(self.position_changed)
        self.media_playper.durationChanged.connect(self.duration_changed)
        self.media_playper.error.connect(self.handle_error)

    def open_file(self, video_filename):
        """動画ファイルの読み込みと再生準備"""
        if video_filename != '' or video_filename is not None:
            self.media_playper.setMedia(QMediaContent(QUrl.fromLocalFile(video_filename)))
            self.play_button.setEnabled(True)

    def play(self):
        """再生開始の機能"""
        if self.media_playper.state() == QMediaPlayer.PlayingState:
            self.media_playper.pause()
        else:
            self.media_playper.play()

    def media_state_changed(self, state):
        if self.media_playper.state() == QMediaPlayer.PlayingState:
            self.play_button.setIcon(
                    self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(
                    self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

    def position_changed(self, position):
        self.position_slider.setValue(position)

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)

    def setPosition(self, position):
        self.media_playper.setPosition(position)

    def handle_error(self):
        self.play_button.setEnabled(False)
        self.errorLabel.setText("Error: " + self.media_playper.errorString())


if __name__ == '__main__':
    # デバッグ用の実行部分
    # デバッグ用実行方法: main.pyのあるフォルダに移動して次のコマンドを実行
    # python3 tempatteditor.widgets.video_view

    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = VideoView()
    widget.setGeometry(100, 100, 500, 500)
    widget.show()
    sys.exit(app.exec_())
