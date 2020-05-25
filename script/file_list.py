#!/usr/bin/env python
# coding: utf-8

import os
import glob
import codecs


BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
TRAINING_BAD_WAV_DIR_PATH = DATA_DIR_PATH + "/wavs/test/bad_wav"
TRAINING_BAD_WAV_LIST = DATA_DIR_PATH + "/test_bad_wav_list.txt"


if __name__ == '__main__':

    wav_file_paths = glob.glob("%s/*.wav" % TRAINING_BAD_WAV_DIR_PATH)

    for wav_file_path in wav_file_paths:

        base = os.path.basename(wav_file_path)

        word_file = codecs.open(TRAINING_BAD_WAV_LIST, 'a', 'utf-8')
        word_file.write(base + '\n')
        word_file.close

    print("\nAll process completed...")

# 1個少なく記録される
