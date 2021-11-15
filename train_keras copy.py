import os
import gc

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import models
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau

from HW_base import evaluate_build, focus_build, plt_scatter
from HW_keras import hw_layer

def build_model(input_shape):
    
    inputs_x = tf.keras.layers.Input(shape=input_shape, name="input_x")

    x0 = Bidirectional(LSTM(512, return_sequences=True))(inputs_x)
    x = tf.keras.layers.Concatenate(axis=2)([inputs_x, x0])

    x1 = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = tf.keras.layers.Concatenate(axis=2)([x0, x1])

    x2 = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Concatenate(axis=2)([x1, x2])

    x3 = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = tf.keras.layers.Concatenate(axis=2)([x2, x3])

    x4 = Bidirectional(LSTM(512, return_sequences=True))(x)
    
    x = tf.keras.layers.Concatenate(axis=2)([x0, x1, x2, x3, x4])
    x = tf.keras.layers.Dense(64, activation="selu")(x)
    
    x = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs_x, outputs=x)

    model.compile(optimizer='adam', loss='mse', metrics='mae')
    return model

if __name__ == '__main__':
    model = build_model((80,230))
    print(model.summary())