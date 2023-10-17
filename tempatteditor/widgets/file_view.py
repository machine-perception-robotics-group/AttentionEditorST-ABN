#!/usr/bin/env python
# -*- coding: utf-8 -*-

# file_view.py
# Copyright (c) Tsubasa Hirakawa, 2021


from os.path import splitext
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from ..config import DEFAULT_DATA_DIR


# 画像ファイルの場合 ---------------------------
# FILE_FILTER = ['*.jpg', '*.png', '*.bmp']
# FILE_EXTENSION = ['.jpg', '.png', '.bmp']

# 動画ファイルの場合 ---------------------------
FILE_FILTER = ['*.mp4']
FILE_EXTENSION = ['.mp4']


class FileView(QtWidgets.QWidget):
    """動画（画像）ファイルを表示・選択するウィンドウウィジェット"""

    def __init__(self):
        """
        ウィンドウの初期化（クラスインスタンスの初期化）
        ディレクトリやファイルのTree表示，ボタンなどを初期化して配置・表示
        """
        super(FileView, self).__init__()

        # デフォルトで表示するディレクトリの指定
        self.current_path = DEFAULT_DATA_DIR

        # header (title)
        self.view_title = QtWidgets.QLabel("Video files")

        # file system model
        self.model = QtWidgets.QFileSystemModel()
        self.model.setNameFilters(FILE_FILTER)

        # tree view
        self.tree = QtWidgets.QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.setRootPath(self.current_path))
        self.tree.setColumnHidden(1, True)
        self.tree.setColumnHidden(2, True)
        self.tree.setColumnHidden(3, True)
        self.tree.setAnimated(False)
        self.tree.setIndentation(20)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.AscendingOrder)

        # buttons
        self.button_change = QtWidgets.QPushButton('change directory')
        self.button_change.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.button_change.clicked.connect(self.change_directory)

        # layout (各ウィンドウやボタンの配置の指定)
        _layout = QtWidgets.QVBoxLayout()
        _layout.addWidget(self.view_title)
        _layout.addWidget(self.tree)
        _layout.addWidget(self.button_change)
        self.setLayout(_layout)

    def change_directory(self):
        """
        ウィンドウに表示するディレクトリの変更
        """
        _selected_path = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                    caption="Open Image Directory",
                                                                    directory=self.current_path)
        self.current_path = _selected_path

        if _selected_path != '':
            self.tree.setRootIndex(self.model.setRootPath(_selected_path))

    def select_file(self):
        """
        現在選択しているファイルのファイルパスを取得して返す
        """
        _index = self.tree.currentIndex()
        path = self.model.filePath(_index)
        _, ext = splitext(path)
        if ext in FILE_EXTENSION:
            return path
        else:
            return None


if __name__ == '__main__':
    # デバッグ用の実行部分
    # デバッグ用実行方法: main.pyのあるフォルダに移動して次のコマンドを実行
    # python3 tempatteditor.widgets.file_view

    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = FileView()
    widget.setGeometry(100, 100, 500, 500)
    widget.show()
    sys.exit(app.exec_())
