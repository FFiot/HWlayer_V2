# coding=utf-8
# Copyright 2020 f.f.l.y@hotmail.com.
import numpy as np

import torch
import fastai

from tqdm import tqdm

from HW_base import evaluate_build, focus_build, plt_scatter, idx_build

def net_parameter_count(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad]), sum([p.numel() for p in model.parameters() if not p.requires_grad])

class VentilatorDataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = torch.from_numpy(data).float()
        if target is not None:
            self.targets = torch.from_numpy(target).float()
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if hasattr(self, 'targets'): 
            return self.data[idx], self.targets[idx]
        else: 
            return self.data[idx]

def dataLoads_build(x_train, y_train, x_valid=None, y_valid=None, batch_size=100, shuffle=True):
    train_loader = torch.utils.data.DataLoader(VentilatorDataset(x_train, y_train), batch_size=batch_size, shuffle=shuffle)
    if x_valid is not None and y_valid is not None:
        valid_loader = torch.utils.data.DataLoader(VentilatorDataset(x_valid, y_valid), batch_size=batch_size, shuffle=False)
        return fastai.data.core.DataLoaders(train_loader, valid_loader)
    else:
        return fastai.data.core.DataLoaders(train_loader) 

def torch_predict(net_list, data, batch_size, to_device=None, to_numpy=True):
    predict_list = []
    with torch.no_grad():
        for net in net_list:
            net.eval()
        idx_list = idx_build(len(data), batch_size, False)
        with tqdm(range(len(idx_list)), desc='prodict', ncols=0) as t:
            for i in t:
                x = data[idx_list[i]]
                if to_device is not None:
                    x = torch.from_numpy(x).to(to_device)
                for net in net_list:
                    x = net(x)
                p = x 
                if to_numpy:
                    predict_list.append(p.cpu().detach().numpy())
                else:
                    predict_list.append(p)
    if to_numpy:
        return np.concatenate(predict_list, axis=0)
    else:
        return torch.cat(predict_list, dim=0)

def torch_valid(net_list, loss_function, valid_data, batch_size, to_device=None):
    val_loss_list = []
    with torch.no_grad():
        for net in net_list:
            net.eval()
        for idx in idx_build(len(valid_data[0]), batch_size, True):
            x = valid_data[0][idx]
            y = valid_data[1][idx]
            if to_device is not None:
                x = torch.from_numpy(x).to(to_device)
                y = torch.from_numpy(y).to(to_device)   
            for net in net_list:
                x = net(x)
            p = x
            val_loss = loss_function(p, y)
            val_loss_list.append(float(val_loss.cpu().detach().numpy()))
        return np.array(val_loss_list).mean()

def torch_train(net_list, optimizer, loss_function, 
                train_data, batch_size, shulffe=True, 
                valid_data=None, valid_batch_size=None, desc='', to_device=None):
    loss_list = []
    idx_list = idx_build(len(train_data[0]), batch_size, shulffe)
    with tqdm(range(len(idx_list)), desc=desc, ncols=0) as t:
        for net in net_list:
            net.train()
        for i in t:
            x = train_data[0][idx_list[i]]
            y = train_data[1][idx_list[i]]
            
            if to_device is not None:
                x = torch.from_numpy(x).to(to_device)
                y = torch.from_numpy(y).to(to_device)

            for net in net_list:
                x = net(x)
            p = x

            loss = loss_function(p, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_list.append(float(loss.cpu().detach().numpy()))
            
            if i+1 < len(idx_list):
                t.set_postfix({'loss':'%0.8f'%loss_list[-1]})
            else:
                loss = np.array(loss_list).mean()
                if valid_data is not None:
                    val_loss = torch_valid(net_list, loss_function, valid_data, valid_batch_size, to_device=to_device)
                    t.set_postfix({'loss':'%0.8f'%loss, 'val_loss':'%0.8f'%val_loss})
                else:
                    t.set_postfix({'loss':'%0.8f'%loss})

    if valid_data is None:
        return loss
    else:
        return loss, val_loss

class hw_layer(torch.nn.Module):
    def __init__(self, evaluate_focus_list):
        super(hw_layer, self).__init__()
        self.evaluate_list = torch.nn.ModuleList()
        self.focus_list = torch.nn.ModuleList()

        def embedding_build(module_list, v):
            v = np.expand_dims(v, -1)
            v = torch.from_numpy(v)
            v = torch.nn.Embedding.from_pretrained(v, freeze=True)
            module_list.append(v)

        for evaluate_focus in evaluate_focus_list:
            evaluate_focus = np.array(evaluate_focus, dtype=np.float32)
            embedding_build(self.evaluate_list, evaluate_focus[:,0])
            embedding_build(self.focus_list, evaluate_focus[:,1])

        self.channels = sum([len(evaluate_focus) for evaluate_focus in evaluate_focus_list])
        
    def forward(self, x):
        output_list = []
        for i in range(list(x.shape)[-1]):
            data = x[..., i:i+1]

            evaluate = self.evaluate_list[i].weight
            evaluate_shape = np.ones_like(list(data.shape))
            evaluate_shape[-1] = evaluate.shape[0]
            evaluate = evaluate.reshape(tuple(evaluate_shape))

            distance = (data - evaluate).abs()

            focus = self.focus_list[i]       
            focus_idx = distance.argmin(axis=-1)
            focus = focus(focus_idx)

            s = distance * focus * -1.0
            s = s.softmax(dim=-1)

            output_list.append(s)
        return torch.cat(output_list, dim=-1)

class test_net(torch.nn.Module):
    def __init__(self, evaluate_focus_list, **kwargs):
        super(test_net, self).__init__()
        self.hw_evaluate = hw_layer(evaluate_focus_list)
        self.hw_embedding = torch.nn.Linear(self.hw_evaluate.channels, 1)

    def forward(self, x):
        x = self.hw_evaluate(x)
        x = self.hw_embedding(x)
        return x

if __name__ == '__main__':
    evaluate_num = 128
    focus_target = 0.8

    def test_fucntion(x):
        return np.sin(x**2 * np.pi * 8)/2 - (x)**2 + 0.5
    
    data_train = np.random.random(10000).reshape((-1 , 1)).astype(np.float32)
    target_train = test_fucntion(data_train)
    plt_scatter(data_train, y_true=target_train)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    evaluate_list = [evaluate_build(data_train, num=evaluate_num)]
    evaluate_dic_list = [focus_build(evaluate, focus_target) for evaluate in evaluate_list]
    net = test_net(evaluate_dic_list).to(device)
    print(net)
    print(net_parameter_count(net))
    
    if device.type == 'cuda':
        net = torch.nn.DataParallel(net)  
        torch.backends.cudnn.benchmark = True 

    x = torch.from_numpy(data_train).to(device)
    y = torch.from_numpy(target_train).to(device)

    optimizer = torch.optim.Adam(net.parameters())
    loss_function = torch.nn.MSELoss()

    for epoch in range(100):
        epoch += 1
        loss, val_loss = torch_train([net], optimizer, loss_function, 
                                     (x, y), 100, True,
                                     (x, y), 1000, 
                                     desc=f'Epoch {epoch}/{100}')