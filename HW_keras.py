import os
import time

import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau

from HW_base import evaluate_build, focus_build, plt_scatter

class hw_layer(Layer):
    def __init__(self, evaluate_focus_list, **kwargs):
        self.evaluate_focus_list = evaluate_focus_list
        super(hw_layer, self).__init__(**kwargs)

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
        
        super(hw_layer, self).build(input_shape)

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
        config = super(hw_layer, self).get_config()
        return config

    def compute_output_shape(self, input_shape):        
        shape = list(input_shape)
        out_shape = sum([len(e) for e in self.evaluate_table])
        shape[-1] = out_shape
        return shape

class DataGenerator(Sequence):
    def __init__(self, x, y=None, evaluate_focus_list=None, batch_size=100, shuffle=True) -> None:
        self.x = np.array(x) if x is not None else None
        self.y = np.array(y) if y is not None else None
        self.evaluate_focus_list = evaluate_focus_list
        self.batch_size = batch_size
        self.idx = np.arange(len(self.x))
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.idx)

        self.x_shape = list(self.x.shape)
        if self.evaluate_focus_list is not None:
            self.x_shape[-1] = sum([len(e) for e in self.evaluate_focus_list])

        self.steps_per_epoch = len(self.idx)//self.batch_size
        if len(self.idx) % self.batch_size > 0:
            self.steps_per_epoch += 1

    def __len__(self):
        data_size = len(self.idx)
        batch_num = data_size//self.batch_size
        if data_size % self.batch_size > 0:
            batch_num += 1
        return batch_num
    
    def __getitem__(self, batch_index):
        return self.__data_generation(self.idx[self.batch_size*batch_index:self.batch_size*(batch_index+1)])
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)
        return self.idx

    def __data_generation(self, idx_list):
        if self.y is None:
            return self.x[idx_list]
        else:
            return self.x[idx_list], self.y[idx_list]

def gpu_limit(gpu, MB):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[gpu], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MB)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

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

    model_name = time.strftime('HW_Keras%Y%m%d%H%M%S')

    model = Sequential([
        hw_layer(evaluate_focus_list, input_shape=(1,), name='hw_layer'),
        Dense(1, name='hw_embedding')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01), loss='mae', metrics='mae')
    model.summary()

    callback_list = [
        ReduceLROnPlateau(monitor='loss', factor=0.7,  patience=8, min_lr=1e-7, verbose=True),
        EarlyStopping(monitor='loss', patience=32),
        ModelCheckpoint('models/%s_best.h5'%model_name, monitor='loss', save_best_only=True, verbose=False),
        TensorBoard(log_dir='./Log/%s'%model_name)
    ]

    model.fit(data_train, target_train, epochs=1000, batch_size=50, callbacks=callback_list)
    p = model.predict(data_train)

    plt_scatter(data_train, np.reshape(p, -1), target_train)