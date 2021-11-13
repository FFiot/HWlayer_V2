import gc
import numpy as np
import pandas as pd

import torch

from fastai.callback.schedule import Learner
from fastai.data.core import DataLoaders
from fastai.losses import L1LossFlat
from fastai.callback.core import Callback
from fastai.callback.tracker import ReduceLROnPlateau, SaveModelCallback

from HW_base import evaluate_build, focus_build
from HW_torch import hw_layer, dataLoads_build, net_parameter_count, torch_valid, torch_predict

class net_test(torch.nn.Module):
    def __init__(self, evaluate_focus_list, **kwargs):
        super(net_test, self).__init__()
        self.hw_layer = hw_layer(evaluate_focus_list)
        self.embedding = torch.nn.Linear(self.hw_layer.channels, 128)

        self.lstm1 = torch.nn.LSTM(128, 128, num_layers=1, bias=False, bidirectional=True, batch_first=True)
        self.lstm2 = torch.nn.LSTM(128*3, 128, num_layers=1, bias=False, bidirectional=True, batch_first=True)
        self.lstm3 = torch.nn.LSTM(128*5, 128,  num_layers=1, bias=False, bidirectional=True, batch_first=True)
        self.lstm4 = torch.nn.LSTM(128*7, 128,  num_layers=1, bias=False, bidirectional=True, batch_first=True)

        self.fc1 = torch.nn.Linear(128*8, 32, bias=False)
        self.selu = torch.nn.SELU()
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.hw_layer(x)
        x = self.embedding(x)

        x1, _ = self.lstm1(x)
        x = torch.concat((x, x1), dim=-1)

        x2, _ = self.lstm2(x)
        x = torch.concat((x, x2), dim=-1)

        x3, _ = self.lstm3(x)
        x = torch.concat((x, x3), dim=-1)

        x4, _ = self.lstm4(x)
        x = torch.concat((x1, x2, x3, x4), dim=-1)

        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

    net = net_test(evaluate_focus_list, output_dims=1)
    print(net)
    print(net_parameter_count(net))

    dataLoads = dataLoads_build(x_train, y_train, x_valid, y_valid, 100)
    learn = Learner(dataLoads, net, loss_func=L1LossFlat())
    learn.lr_find()
    learn.fit_one_cycle(100, lr_max=2e-3, cbs=ReduceLROnPlateau(monitor='valid_loss', min_delta=0.5, patience=10))

    net = net.to(device)
    x_train = torch_predict([net.input_layer], x_train, 100, to_device=device)
    x_valid = torch_predict([net.input_layer], x_valid, 100, to_device=device)
    x_test  = torch_predict([net.input_layer], x_test, 100, to_device=device)