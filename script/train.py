#!/usr/bin/env python
# coding: utf-8

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np

import os
from sklearn.model_selection import StratifiedKFold
import sqlite3
import datetime
import shutil
import scipy.stats as sp

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

FEATURE = "mfcc+frame+score"

FOLDS_NUMBER = 10
BATCH_SIZE = 16 # [1, 8, 16, 32, 64, 128, 256, 512]
EPOCHS = 300

# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
TRAINING_WAV_DIR_PATHS = [DATA_DIR_PATH + "/wavs/bad_wav", DATA_DIR_PATH + "/wavs/ok_wav"]
TRAINING_FILE_PATH = DATA_DIR_PATH + "/train.csv"
LOG_DIR_PATH = DATA_DIR_PATH + "/logs"
DATABASE_PATH = DATA_DIR_PATH + "/evaluation.sqlite3"

# Read data
train = pd.read_csv(TRAINING_FILE_PATH)
X = (train.iloc[:, 1:].values).astype('float32')
y = train.iloc[:, 0].values.astype('int32')

# Z-Score関数による正規化
X = sp.stats.zscore(X, axis=1)

input_dim = X.shape[1]

# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# we'll use binary xent for the loss, and AdaDelta as the optimizer
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=["accuracy"])

# Cross Validation Evaluation Method
print("Training...")

if os.path.isdir(LOG_DIR_PATH):
    shutil.rmtree(LOG_DIR_PATH)

os.makedirs(LOG_DIR_PATH)

# define X-fold cross validation
kf = StratifiedKFold(n_splits=FOLDS_NUMBER, shuffle=True)

cvscores = []

fld = 0
_train_acc = [[0 for i in range(FOLDS_NUMBER)] for j in range(EPOCHS)]
_valid_acc = [[0 for i in range(FOLDS_NUMBER)] for j in range(EPOCHS)]
_train_loss = [[0 for i in range(FOLDS_NUMBER)] for j in range(EPOCHS)]
_valid_loss = [[0 for i in range(FOLDS_NUMBER)] for j in range(EPOCHS)]

# cross validation
for train, test in kf.split(X, y):

    print("\n")
    print("Running Fold", fld + 1, "/", FOLDS_NUMBER)

    now = datetime.datetime.now()
    hist = model.fit(X[train], y[train], validation_data=(X[test], y[test]), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

    # Evaluate
    scores = model.evaluate(X[test], y[test], verbose=0)

    print("\n")
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

    # Save the model
    model.save(LOG_DIR_PATH + "/" + FEATURE + "_mlp" + "_{0:%Y-%m-%d}".format(now) + "_final" + "{0:02d}".format(fld + 1) + "_" + str(round(scores[1]*100, 1)) + "%.h5")

    for epoch in range(0, EPOCHS):

        _train_acc[epoch][fld] = hist.history['accuracy'][epoch]
        _valid_acc[epoch][fld] = hist.history['val_accuracy'][epoch]
        _train_loss[epoch][fld] = hist.history['loss'][epoch]
        _valid_loss[epoch][fld] = hist.history['val_loss'][epoch]

    fld += 1

train_acc = []
valid_acc = []
train_loss = []
valid_loss = []

for epoch in range(0, EPOCHS):

    train_acc.append(np.mean(_train_acc[epoch]))
    valid_acc.append(np.mean(_valid_acc[epoch]))
    train_loss.append(np.mean(_train_loss[epoch]))
    valid_loss.append(np.mean(_valid_loss[epoch]))

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

if os.path.exists(DATABASE_PATH):
    os.remove(DATABASE_PATH)

db = sqlite3.connect(DATABASE_PATH)
cur = db.cursor()

sql = "CREATE TABLE IF NOT EXISTS learning (feature TEXT, epoch INTEGER, training_accuracy REAL, validation_accuracy REAL, training_loss REAL, validation_loss REAL);"

cur.execute(sql)
db.commit()

datas = []
for i in range(len(hist.epoch)):

    # なぜか2番目のtrain accuracyがバイナリーになるのでfloatにキャストし直し（sqliteのバージョンによるバグ？）
    datas.append([FEATURE, i + 1, float(train_acc[i]), valid_acc[i], train_loss[i], valid_loss[i]])

sql = "INSERT INTO learning (FEATURE, epoch, training_accuracy, validation_accuracy, training_loss, validation_loss) VALUES (?, ?, ?, ?, ?, ?);"

cur.executemany(sql, datas)
db.commit()

cur.close()
db.close()

print("\n")
print("All Proccess was completed.")
print("\n")
