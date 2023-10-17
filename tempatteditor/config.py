#!/usr/bin/env python
# -*- coding: utf-8 -*-

# config.py
# Copyright (c) Tsubasa Hirakawa, 2021


# File viewで最初に表示するフォルダのパス
DEFAULT_DATA_DIR = "./smth_smth_v2/video/"

# 修正したattention mapを保存するフォルダのパス
SAVE_DIR = "./saved_attention/"

# ST-ABNのネットワークの設定ファイル（configファイル）のパス（違う気がするけど...）
CONFIG_FILE = "./config_files/stabn/models/resnet_abn_v1.py"

# 学習済みモデルのパス
CHECKPOINT_FILE = "./checkpoints/abn_res50/checkpoint-best.pt"
TRAIN_ARGS_FILE = "./checkpoints/abn_res50/args.json"

#正解ラベルファイル
LABEL_FILE = "./smth_smth_v2/anno/train_videofolder.txt"

# ネットワークへ入力する動画フレームのサイズとフレーム数
# （読み込んだmp4の動画から自動でこのサイズに切り取ります）
IMAGE_SIZE = 224
NUM_FRAMES = 80

# Attention画像の表示サイズ（GUIへの表示だけ）
DISPLAY_IMG_SIZE = 76
# Attention Imageを横に並べる数
NUM_COLS = 20
