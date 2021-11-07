import os
import gc

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau

from HW_base import evaluate_build, focus_build, plt_scatter
from HW_keras import hw_layer

if __name__ == '__main__':
    fname = 'F5_Keras_test'
    # 读取数据
    data_train_df = pd.read_csv('./Database/train.csv')
    data_test_df = pd.read_csv('./Database/test.csv')
    x_columns = ['R', 'C', 'time_step', 'u_in', 'u_out']
    y_columns = ['pressure']

    data_train = data_train_df[x_columns].values.astype(np.float32)
    data_train = data_train.reshape(-1, 80, data_train.shape[-1])

    target_train = data_train_df[y_columns].values.astype(np.float32)
    target_train = target_train.reshape(-1, 80, target_train.shape[-1])

    data_test = data_test_df[x_columns].values.astype(np.float32)
    data_test = data_test.reshape(-1, 80, data_test.shape[-1])

    del data_train_df
    del data_test_df
    gc.collect()
    # 数据分割
    np.random.seed(121212)
    data_idx = np.arange(len(data_train))
    np.random.shuffle(data_idx)

    train_index = data_idx[:int(len(data_idx)*0.8)]
    valid_index = data_idx[int(len(data_idx)*0.8):]

    x_train, y_train = data_train[train_index], target_train[train_index]
    x_valid, y_valid = data_train[valid_index], target_train[valid_index]
    x_test = data_test

    del data_train
    del data_test
    gc.collect()

    evaluate_list = [evaluate_build(x_test[..., i], 128) for i in range(x_test.shape[-1])]
    evaluate_focus_list = []
    for evaluate in evaluate_list:
        focus = 1 - (len(evaluate) - 1)/10
        if focus < 0.6:
            focus = 0.6
        evaluate_focus = focus_build(evaluate, focus)
        evaluate_focus_list.append(evaluate_focus)

    model = Sequential([
        hw_layer(evaluate_focus_list, input_shape=(80,5), name='hw_layer'),
        Dense(128, use_bias=False),
        Bidirectional(LSTM(128, use_bias=True, return_sequences=True)),
        Bidirectional(LSTM(128, use_bias=True, return_sequences=True)),
        Bidirectional(LSTM(128, use_bias=True, return_sequences=True)),
        Bidirectional(LSTM(128, use_bias=True, return_sequences=True)),
        Dense(32, use_bias=False, activation='selu'),
        Dense(1, name='hw_embedding')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01), loss='mae')
    # model.compile(optimizer='adam', loss='mse', metrics='mae')
    model.summary()

    callback_list = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.7,  patience=8, min_lr=1e-7, verbose=True),
        EarlyStopping(monitor='val_loss', patience=32),
        ModelCheckpoint(f'models/{fname}_best.h5', monitor='val_loss', save_best_only=True, verbose=False),
        TensorBoard(log_dir=f'./Log/{fname}')
    ]

    model.fit(x_train, y_train, epochs=1000, batch_size=100,
              validation_data=(x_valid, y_valid), validation_batch_size=100,
              callbacks=callback_list)