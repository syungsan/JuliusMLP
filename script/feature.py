#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import csv
import codecs
import subprocess
import math
import shutil
import numpy as np
from statistics import mean
import librosa
from dtw import dtw
import formant as fmt


# Training Only = 1, Training & Test = 2
METHOD_PATTERN = 2

# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
TEMP_DIR_PATH = BASE_ABSOLUTE_PATH + "temp"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
TRAINING_WAV_DIR_PATHS = [DATA_DIR_PATH + "/wavs/bad_wav", DATA_DIR_PATH + "/wavs/ok_wav"]
TEST_WAV_DIR_PATHS = [DATA_DIR_PATH + "/wavs/test/bad_wav", DATA_DIR_PATH + "/wavs/test/ok_wav"]
SETTING_FILE_PATH = DATA_DIR_PATH + "/setting.conf"
TRAINING_FILE_PATH = DATA_DIR_PATH + "/train.csv"
TEST_FILE_PATH = DATA_DIR_PATH + "/test.csv"
REFERENCE_WAV_PATH = DATA_DIR_PATH + "/wavs/reference.wav"

# Label
CLASS_LABELS = [0, 1]
WAV_LABELS = ["bad", "ok"]

# Command
SEGMENTATION_COMMAND = BASE_ABSOLUTE_PATH + "perl/bin/perl.exe " + BASE_ABSOLUTE_PATH + "script/segment_julius.pl"

# 正誤各waveファイルの最大保持設定数
MAX_WAV_COUNT = 3000

# 1: MFCCのフレーム平均, 2: スコア, 3: フレーム数, 4: DTW, 5: フォルマント, 6: セグメンテーション結果
feature_flag = [1, 0, 0, 0, 0, 1]


def read_csv(file_path, delimiter):

    lists = []
    file = codecs.open(file_path, 'r', 'utf-8')

    reader = csv.reader(file, delimiter=delimiter)

    for line in reader:
        lists.append(line)

    file.close
    return lists


def write_csv(file_path, list):

    try:
        # 書き込み UTF-8
        with open(file_path, 'w', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerows(list)

    # 起こりそうな例外をキャッチ
    except FileNotFoundError as e:
        print(e)
    except csv.Error as e:
        print(e)


def make_resource(word, wav_folder_paths):

    i = 0
    source_elements = []

    for wav_folder_path in wav_folder_paths:

        count = 1
        wav_file_paths = glob.glob("%s/*.wav" % wav_folder_path)

        for wav_file_path in wav_file_paths:

            base = os.path.basename(wav_file_path)
            name, ext = os.path.splitext(base)
            trans_name = name.replace(".", "-").replace(" ", "_")

            # 先頭に0を付ける工夫
            data_size = int(math.log10(MAX_WAV_COUNT) + 1)
            count_size = int(math.log10(count) + 1)
            new_count = "0" * (data_size - count_size) + str(count)

            new_name = "%s/segment_%s_%s_%s" % (TEMP_DIR_PATH, WAV_LABELS[i], new_count, trans_name)
            source_elements.append([new_name, CLASS_LABELS[i], wav_file_path])

            word_file = codecs.open(new_name + ".txt", 'w', 'utf-8')
            word_file.write(word + '\n')
            word_file.close

            shutil.copyfile(wav_file_path, new_name + ".wav")

            print("Copy source files " + new_name + ".etc ...")

            count += 1
        i += 1

    return source_elements


def execute_segmentation():

    try:
        subprocess.run(SEGMENTATION_COMMAND, shell=True, check=True)
    except subprocess.CalledProcessError:
        print("外部プログラムの実行に失敗しました", file=sys.stderr)


def pick_score(file_path, word_length):

    labs = read_csv(file_path=file_path, delimiter=" ")
    segments = []

    if len(labs) == 0:
        segments += [0] * word_length

    else:
        # 前後の消音のセグメントは抜かす
        for i in range(1, len(labs) - 1):

            # 特徴量とするセグメントの選抜
            segments.append(labs[i][2])

    print("Pick score in " + file_path + " ...")

    return segments


def pick_frame(file_path, word_length):

    labs = read_csv(file_path=file_path, delimiter=" ")
    segments = []

    if len(labs) == 0:
        segments = [0] * word_length * 2
    else:
        # 前後の消音のセグメントは抜かす
        for i in range(1, len(labs) - 1):

            # 特徴量とするセグメントの選抜
            segments.append(labs[i][0])
            segments.append(labs[i][1])

    print("Pick frame in " + file_path + " ...")

    return segments


def get_frame(wav_file_path):

    x, fs = librosa.load(wav_file_path, sr=16000, duration=2.5)
    return x.shape[0] / fs


def get_mfcc(wav_file_path):

    x, fs = librosa.load(wav_file_path, sr=16000, duration=2.5)

    # サンプリングレート: 1秒辺りのサンプル数
    print("sampling rate: {0}".format(fs))
    print("length: {0}[pt]={1}[s]".format(x.shape[0], x.shape[0] / fs))

    S = librosa.feature.melspectrogram(x, sr=fs, hop_length=160, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    mfccs = librosa.feature.mfcc(S=log_S, hop_length=160, n_mfcc=13)

    # Let's pad on the first and second deltas while we're at it
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    mfccs = np.append(mfccs, delta_mfccs, axis=0)
    mfccs = np.append(mfccs, delta2_mfccs, axis=0)

    print("Get MFCC in " + wav_file_path + " ...")

    return mfccs


def get_mfcc_mean(wav_file_path):

    x, fs = librosa.load(wav_file_path, sr=16000, duration=2.5)

    # サンプリングレート: 1秒辺りのサンプル数
    print("sampling rate: {0}".format(fs))
    print("length: {0}[pt]={1}[s]".format(x.shape[0], x.shape[0] / fs))

    S = librosa.feature.melspectrogram(x, sr=fs, hop_length=160, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    # Time shift 幅 1/100秒（Juliusと一致させる）
    mfccs = librosa.feature.mfcc(S=log_S, hop_length=160, n_mfcc=13)

    # Let's pad on the first and second deltas while we're at it
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    mfcc_means = []

    # numpyをforで回しで普通のListに入れているダメ
    for mfcc in mfccs:
        mfcc_means.append(mean(mfcc))

    for delta_mfcc in delta_mfccs:
        mfcc_means.append(mean(delta_mfcc))

    for delta2_mfcc in delta2_mfccs:
        mfcc_means.append(mean(delta2_mfcc))

    print("Get MFCC Mean in " + wav_file_path + " ...")

    return mfcc_means


def get_dtw(reference, target):

    manhattan_distance = lambda x, y: np.linalg.norm(x - y)

    dist, cost, acc, path = dtw(reference, target, dist=manhattan_distance)

    print("Minimal Distance => %f" % dist)

    return dist


def flatten_with_any_depth(nested_list):

    """深さ優先探索の要領で入れ子のリストをフラットにする関数"""
    # フラットなリストとフリンジを用意
    flat_list = []
    fringe = [nested_list]

    while len(fringe) > 0:
        node = fringe.pop(0)
        # ノードがリストであれば子要素をフリンジに追加
        # リストでなければそのままフラットリストに追加
        if isinstance(node, list):
            fringe = node + fringe
        else:
            flat_list.append(node)

    return flat_list


# フォルダ内の全ファイル消し
def clear_dir(dir_path):

    os.chdir(dir_path)
    files = glob.glob("*.*")

    for filename in files:
        os.unlink(filename)


if __name__ == '__main__':

    if not os.path.isdir(TEMP_DIR_PATH):
        os.makedirs(TEMP_DIR_PATH)

    clear_dir(TEMP_DIR_PATH)

    setting = read_csv(file_path=SETTING_FILE_PATH, delimiter=",")

    if feature_flag[3] == 1:
        references = get_mfcc(REFERENCE_WAV_PATH)

    # 訓練とテスト用のデータをまとめて作る
    for method in range(0, METHOD_PATTERN):

        # temp内の全ファイル消し
        clear_dir(TEMP_DIR_PATH)

        if method == 0:
            learning_file_path = TRAINING_FILE_PATH
            wav_folder_paths = TRAINING_WAV_DIR_PATHS
        else:
            learning_file_path = TEST_FILE_PATH
            wav_folder_paths = TEST_WAV_DIR_PATHS

        word = setting[0][1]
        source_elements = make_resource(word, wav_folder_paths)

        if feature_flag[1] == 1 or feature_flag[5] == 1 or feature_flag[4] == 1:
            execute_segmentation()

        word_length = len(setting[0][2])

        trains = []
        for source_element in source_elements:

            features = []

            # MFCC
            if feature_flag[0] == 1:

                mfccs = get_mfcc_mean(source_element[2])
                features.append(mfccs)

            # Score
            if feature_flag[1] == 1:

                scores = pick_score(source_element[0] + ".lab", word_length)
                scores = [float(s) for s in scores]
                features.append(mean(scores))

            # Frame
            if feature_flag[2] == 1:

                frame = get_frame(source_element[2])
                features.append(frame)

            # DTW
            if feature_flag[3] == 1:

                targets = get_mfcc(source_element[2])
                for i in range(len(references)):

                    dist = get_dtw(references[i], targets[i])
                    features.append(dist)

            # Formant
            if feature_flag[4] == 1:

                _frames = pick_frame(source_element[0] + ".lab", word_length)
                formants = fmt.get_formant(wav_file_path=source_element[2], frames=_frames)
                features.append(formants)

            # Segmentation
            if feature_flag[5] == 1:

                frames = pick_frame(source_element[0] + ".lab", word_length)
                scores = pick_score(source_element[0] + ".lab", word_length)
                scores = [float(s) for s in scores]

                features.append(frames)
                features.append(scores)

            features = flatten_with_any_depth(features)
            features.insert(0, source_element[1])

            trains.append(features)

        write_csv(learning_file_path, trains)

    clear_dir(TEMP_DIR_PATH)

    print("\nAll process completed...")
