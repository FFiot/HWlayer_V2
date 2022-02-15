import time
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy, Huber
from tensorflow.keras.metrics import AUC, Precision, Accuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

def min_max_normal(x):
    x = x.astype(np.float32)
    x_max = np.max(x, axis=(1, 2), keepdims=True)
    x_min = np.min(x, axis=(1, 2), keepdims=True)
    x = (x - x_min)/(x_max - x_min)
    return x

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    x_train = min_max_normal(x_train)
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    x_test = min_max_normal(x_test)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()
    model.compile(Adam(lr=0.01), loss=Huber(), metrics=['AUC', 'accuracy'])

    model_name = time.strftime('HW_Keras%Y%m%d%H%M%S')
    callback_list = [
        # ReduceLROnPlateau(monitor='loss', factor=0.7,  patience=8, min_lr=1e-7, verbose=True),
        EarlyStopping(monitor='loss', patience=32),
        ModelCheckpoint('models/%s_best.h5'%model_name, monitor='loss', save_best_only=True, verbose=False),
        TensorBoard(log_dir='./Log/%s'%model_name)
    ]

    model.fit(x_train, y_train, epochs=100, batch_size=100, callbacks=callback_list, 
              validation_data=(x_test, y_test))