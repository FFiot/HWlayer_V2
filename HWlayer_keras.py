import os
import time

import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau

from HWlayer_base import evaluate_build, focus_build, plt_scatter

class HWlayer(Layer):
    def __init__(self, evaluate_focus_list, **kwargs):
        super(HWlayer, self).__init__(**kwargs)
        self.evaluate_focus_list = evaluate_focus_list
        
    def build(self, input_shape):
        self.evaluate_list, self.focus_list = [], []
        for evaluate_focus in self.evaluate_focus_list:
            evaluate_focus = np.array(evaluate_focus)
            evaluate, focus = evaluate_focus[:,0], evaluate_focus[:,1]
            evaluate_shape = [1 for _ in range(len(input_shape))]
            evaluate_shape[-1] = len(evaluate)
            evaluate = np.reshape(evaluate, evaluate_shape)
            self.evaluate_list.append(K.variable(evaluate, dtype='float32', name='evaluate'))
            focus = np.expand_dims(focus, -1)
            self.focus_list.append(K.variable(focus, dtype='float32', name='focus'))
        
        super(HWlayer, self).build(input_shape)

    def call(self, x):
        y_list = []
        for i, (evaluate, focus) in enumerate(zip(self.evaluate_list, self.focus_list)):
            d = x[..., i]
            d = K.expand_dims(d, axis=-1)
            d = K.abs(d - evaluate)
            f_idx = K.argmin(d, axis=-1)
            f = tf.nn.embedding_lookup(focus, f_idx)
            d = d * f * -1.0
            d = K.softmax(d, axis=-1)
            y_list.append(d)
        y = K.concatenate(y_list, axis=-1)
        return y
    
    def get_config(self):
        config = super(HWlayer, self).get_config()
        return config

    def compute_output_shape(self, input_shape):        
        shape = list(input_shape)
        out_shape = sum([len(e) for e in self.evaluate_table])
        shape[-1] = out_shape
        return shape

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def test_fucntion(x):
        return np.sin(x**2 * np.pi * 8)/2 - (x)**2 + 0.5
    
    data_train = np.random.random(10000).reshape((-1,1))
    data_train = data_train.astype(np.float32)
    target_train = test_fucntion(data_train)
    plt_scatter(data_train, y_true=target_train)

    evaluate_list = [np.array(evaluate_build(data_train, num=128))]

    evaluate_focus_list = [focus_build(evaluate, 0.8) for evaluate in evaluate_list]

    model_name = time.strftime('HWlayer_Keras%Y%m%d%H%M%S')

    model = Sequential([
        HWlayer(evaluate_focus_list, input_shape=(1,), name='HWlayer'),
        Dense(1, name='HWlayer_embedding')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mae', metrics='mae')
    model.summary()

    callback_list = [
        ReduceLROnPlateau(monitor='loss', factor=0.7,  patience=4, min_lr=1e-7, verbose=True),
        EarlyStopping(monitor='loss', patience=32),
        ModelCheckpoint('models/%s_best.h5'%model_name, monitor='loss', save_best_only=True, verbose=False),
        TensorBoard(log_dir='./Log/%s'%model_name)
    ]

    model.fit(data_train, target_train, epochs=1000, batch_size=50, callbacks=callback_list)
    p = model.predict(data_train)

    plt_scatter(data_train, np.reshape(p, -1), target_train)