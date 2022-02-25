import time

import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.layers import Softmax, Reshape
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Accuracy, Recall, Precision, CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau, Callback
from keras.preprocessing.image import ImageDataGenerator

from HWlayer_base import evaluate_build, focus_build
from HWlayer_keras import HWlayer

def evaluate_build_AveragePooling2D(data, evaluate_num=8, focus=0.8, deepth=3):
    output_list = []
    x = Input(data.shape[1:])
    x_list = [x]
    for idx in range(deepth):
        x_list += [MaxPooling2D(2)(x_list[-1])]
    
    model = Model(x, x_list)
    data_list = model.predict(data)
    
    evaluate_focus_table = []
    for data in data_list:
        print(data.shape)
        evaluate_list = [evaluate_build(data[..., i], evaluate_num) for i in range(data.shape[-1])]
        evaluate_focus_list = [focus_build(evaluate, focus) for evaluate in evaluate_list]
        evaluate_focus_table.append(evaluate_focus_list)
        
    return evaluate_focus_table

if __name__ == '__main__':
    evaluate_num = 8
    focus = 0.8
    deepth = 4
    evaluate_dims = evaluate_num * 3

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    x_test = x_test / 255
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    evaluate_focus_table = evaluate_build_AveragePooling2D(x_test, evaluate_num, focus, deepth)

    x = Input(shape=((32, 32, 3)))
    x_list = [x]
    for _ in range(deepth):
        x_list.append(MaxPooling2D(2)(x_list[-1]))
    hw_layer_list = [HWlayer(evaluate_focus_list)(x)  for (x, evaluate_focus_list) in zip(x_list, evaluate_focus_table)]
    y = hw_layer_list[0]
    hw_acitve_dic = {hw_layer.shape[1]:K.expand_dims(hw_layer, -2)  for hw_layer in hw_layer_list}    

    def hw_acitve(x, hw_acitve_dic):
        x_shape = list(x.shape)[1:]
        if x_shape[0] not in hw_acitve_dic:
            return x

        shape = x_shape[:-1] + [-1, evaluate_dims]
        y = Reshape(shape)(x)
        y = y * hw_acitve_dic[x_shape[0]]
        y = Reshape(x_shape)(y)
        return y

    VGG_cfg = [4, 4, 'M', 4, 4, 'M', 4, 4, 'M', 4, 4, 'M']

    for cfg in VGG_cfg:
        if cfg == 'M':
            y = MaxPooling2D(2)(y)
        else:
            y = Conv2D(cfg*evaluate_dims, 3, 1, 'same')(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = hw_acitve(y, hw_acitve_dic)
    y = Flatten()(y)
    y = Dense(10)(y)
    y = Softmax()(y)

    model = Model(x, y)
    model.summary()
        
    model_name = time.strftime('CIFAR10_VGG_KERAS_HWNET%Y%m%d%H%M%S')

    steps_per_epoch = len(x_train) // 100
    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=0.0001,
                                              maximal_learning_rate=0.01,
                                              scale_fn=lambda x: 1/(2.**(x-1)),
                                              step_size=2 * steps_per_epoch)
    optimizer = tf.keras.optimizers.SGD(clr)


    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])

    callback_list = [
        EarlyStopping(monitor='loss', patience=32),
        ModelCheckpoint('models/%s_best.h5'%model_name, monitor='loss', save_best_only=True, verbose=False),
        TensorBoard(log_dir='./Log/%s'%model_name)
    ]

    datagen = ImageDataGenerator(
        featurewise_center = False,  # 将整个数据集的均值设为 0
        samplewise_center = False,  # 将每个样本的均值设为 0
        featurewise_std_normalization = False,  # 将输入除以整个数据集的标准差
        samplewise_std_normalization = False,  # 将输入除以其标准差
        zca_whitening = False,  # 运用 ZCA 白化
        zca_epsilon = 1e-06,  # ZCA 白化的 epsilon值
        rotation_range = 0,  # 随机旋转图像范围 (角度, 0 to 180)
        width_shift_range = 0.1,  # 随机水平移动图像 (总宽度的百分比) 
        height_shift_range = 0.1,  # 随机垂直移动图像 (总高度的百分比)
        shear_range = 0.,  # 设置随机裁剪范围
        zoom_range = 0.,  # 设置随机放大范围
        channel_shift_range = 0.,  # 设置随机通道切换的范围
        fill_mode = 'nearest',  # 设置填充输入边界之外的点的模式
        cval = 0.,  # 在 fill_mode = "constant" 时使用的值
        horizontal_flip = True,  # 随机水平翻转图像
        vertical_flip = True,  # 随机垂直翻转图像
        rescale = None,  # 设置缩放因子 (在其他转换之前使用)
        preprocessing_function = None,  # 设置将应用于每一个输入的函数
        data_format = None,  # 图像数据格式，"channels_first" 或 "channels_last" 之一
        validation_split = 0.0)  # 保留用于验证的图像比例（严格在 0 和 1之间）

    datagen.fit(x_train)

    model.fit(datagen.flow(x_train, y_train, batch_size=100), epochs=200, validation_data=(x_test, y_test), validation_batch_size=1000, callbacks=callback_list)