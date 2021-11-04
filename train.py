from warnings import simplefilter
import numpy as np
import pandas as pd

import torch

from HW_base import evaluate_build, focus_build
from HW_torch import hw_layer, torch_train, net_parameter_count

class net_test(torch.nn.Module):
    def __init__(self, evaluate_dic_list, output_dims=4, **kwargs):
        super(net_test, self).__init__()
        self.hw_layer = hw_layer(evaluate_dic_list)
        self.hw_dims = self.hw_layer.channels
        self.linear1 = torch.nn.Conv1d(self.hw_dims, self.hw_dims, 5, 1, 2, bias=False)
        self.linear2 = torch.nn.Linear(self.hw_dims, self.hw_dims, bias=False)
        self.linear3 = torch.nn.Linear(self.hw_dims, self.hw_dims, bias=False)
        self.fc1 = torch.nn.Linear(self.hw_dims, 32, bias=False)
        self.selu = torch.nn.SELU()
        self.fc2 = torch.nn.Linear(32, 1, bias=True)
    def forward(self, x):
        hw = self.hw_layer(x)
        #lstm1
        x = self.linear1(hw)
        x = x * hw
        #lstm2
        x = self.linear2(x)
        x = x * hw
        #lstm3
        x = self.linear3(x)
        x = x * hw
        #fc1
        x = self.fc1(x)
        x = self.selu(x)
        #fc2
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    data_train_df = pd.read_csv('./Database/train.csv')
    
    data_train = data_train_df[['R', 'C', 'time_step', 'u_in', 'u_out']].values.astype(np.float32)
    data_train = np.reshape(data_train, (-1, 80, data_train.shape[-1]))
    target_train = data_train_df[['pressure']].values.astype(np.float32)
    target_train = np.reshape(target_train, (-1, 80, target_train.shape[-1]))

    evaluate_list = [evaluate_build(data_train[..., i], 128) for i in range(data_train.shape[-1])]
    evaluate_focus_list = []
    for evaluate in evaluate_list:
        focus = 1 - (len(evaluate) - 1)/10
        if focus < 0.6:
            focus = 0.6
        evaluate_focus = focus_build(evaluate, focus)
        evaluate_focus_list.append(evaluate_focus)
    
    net = net_test(evaluate_focus_list, output_dims=1).to(device)
    print(net)
    print(net_parameter_count(net))

    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(1000):
        epoch += 1
        loss = torch_train([net], optimizer, loss_function, 
                           (data_train, target_train), 100, True,
                           (data_train, target_train), 1000,
                           desc=f'Epoch {epoch}/{1000}', to_device=device)

