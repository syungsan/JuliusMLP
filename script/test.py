#!/usr/bin/env python
# coding: utf-8

import os
from keras.models import load_model

import pandas as pd
import scipy.stats as sp
import glob


# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
TEST_WAV_DIR_PATHS = [DATA_DIR_PATH + "/wavs/test/bad_wav", DATA_DIR_PATH + "/wavs/test/ok_wav"]
TEST_FILE_PATH = DATA_DIR_PATH + "/test.csv"
LOG_DIR_PATH = DATA_DIR_PATH + "/logs"

# Read data
test = pd.read_csv("../data/test.csv")
y = test.iloc[:, 0].values.astype("int32")
X = (test.iloc[:, 1:].values).astype("float32")

# Z-Score関数による正規化
X = sp.stats.zscore(X, axis=1)

os.chdir(LOG_DIR_PATH)
model_paths = glob.glob("*.*")

losss = []
corrects = []

for model_path in model_paths:

    model = load_model(model_path)

    model.summary()

    # we'll use binary xent for the loss, and AdaDelta as the optimizer
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=["accuracy"])

    score = model.evaluate(X, y, verbose=0)

    print('Test loss :', score[0])
    print('Test accuracy :', score[1])

    losss.append(score[0])
    corrects.append(score[1])

max_accuracy = max(corrects)
max_acc_indexs = [i for i, v in enumerate(corrects) if v == max_accuracy]

print("\n")
for max_acc_index in max_acc_indexs:
    print("最大Accuracy = %f ; at loss = %f => %s" % (max_accuracy*100, losss[max_acc_index], os.path.basename(model_paths[max_acc_index])))

print("")
print("\nAll process completed...")
