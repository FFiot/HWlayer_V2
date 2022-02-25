from matplotlib.pyplot import axis
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter
from torchsummary import summary

from tqdm import tqdm

from HWlayer_base import evaluate_build, focus_build

class HWlayer(nn.Module):
    def __init__(self, evaluate_focus):
        super(HWlayer, self).__init__()
        self.channels = len(evaluate_focus)
        evaluate_focus = np.array(evaluate_focus, dtype=np.float32)

        evaluate = np.expand_dims(evaluate_focus[:, 0], -1)
        evaluate = torch.from_numpy(evaluate)
        self.evaluate = torch.nn.Embedding.from_pretrained(evaluate, freeze=True)
        
        focus = np.expand_dims(evaluate_focus[:, 1], -1)
        focus = torch.from_numpy(focus)
        self.focus = torch.nn.Embedding.from_pretrained(focus, freeze=True)
        
        self.flatten = nn.Flatten()

    def forward(self, x):
        e = self.evaluate.weight.flatten()
        shape = list(x.shape)
        shape = [1]*len(shape) + [self.channels]
        e = e.reshape(shape)

        d = x.unsqueeze(-1)
        d = d - e
        d = d.abs()

        idx = d.argmin(axis=-1, keepdim=True)
        f = self.focus(idx).squeeze(-1)

        s = d * f * -1.0
        s = s.softmax(-1)

        return self.flatten(s)

class HWlayer2D(HWlayer):
    def __init__(self, evaluate_focus_list):
        super(HWlayer2D, self).__init__()
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
        for i in range(x.shape[1]):
            e = self.evaluate_list[i].weight.flatten()
            dims = e.shape[0]
            e = e.reshape((1, dims, 1, 1))
            
            d = x[:, i, :, :]
            d = d.unsqueeze(1)
            d = d - e
            d = d.abs()

            f = self.focus_list[i]
            idx = d.argmin(axis=1, keepdim=True)
            f = f(idx).squeeze(-1)

            s = d * f * -1.0
            s = s.softmax(1)

            output_list.append(s)
        y = torch.cat(output_list, dim=1)
        return y

cfg_dic = {
    'VGG': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_Backbone(nn.Module):
    def __init__(self, cfg='VGG') -> None:
        super(VGG_Backbone, self).__init__()
        self.backbone = self._make_layers(3, cfg_dic[cfg])

    def _make_layers(self, in_channels, cfg_list):
        layers = []
        for cfg in cfg_list:
            if cfg == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1),
                           nn.BatchNorm2d(cfg),
                           nn.ReLU(inplace=True)]
                in_channels = cfg
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.backbone(x)
        return x

class Linear_predict(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Linear_predict, self).__init__()
        self.line = nn.Sequential(nn.Flatten(),
                                  nn.Linear(in_channels, 64),
                                  nn.Linear(64, out_channels))
    def forward(self, x):
        return self.line(x)

class HWlayer_predict(nn.Module):
    def __init__(self, evaluate_focus, in_channels) -> None:
        super(HWlayer_predict, self).__init__()
        self.flatten = nn.Flatten()
        self.hw_layer = HWlayer(evaluate_focus)
        self.linear = nn.Linear(in_channels*self.hw_layer.channels, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hw_layer(x)
        x = self.linear(x)
        return x

def train(epoch, net_list, train_loader, criterion, optimizer, scheduler=None, valid_loader=None, device=None):
    for net in net_list:
        net.train()
    loss_list, p_list, y_list = [], [], []
    with tqdm((enumerate(train_loader)), desc='epoch%3d'%epoch, total=len(train_loader), ncols=0) as t:
        for idx, (x, y) in t:
            optimizer.zero_grad()
            if device is not None:
                x, y = x.to(device), y.to(device)
            p = x
            for net in net_list:
                p = net(p)
            
            loss = criterion(p, y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            _, p = p.max(axis=-1)
            y_list.append(y.detach().cpu().numpy())
            p_list.append(p.detach().cpu().numpy())
            if idx+1 < len(train_loader):
                t.set_postfix({'loss':'%0.4f'%loss_list[-1]})
            else:
                loss = np.array(loss_list).mean()
                acc = (np.array(p_list) == np.array(y_list)).astype(np.float32).mean()

                if scheduler is not None:
                    scheduler.step()

                if valid_loader is not None:
                    valid_loss, valid_acc = valid(net_list, criterion, valid_loader, device)
                    t.set_postfix({'loss':'%0.4f'%loss, 'acc':'%0.4f'%acc, 'valid_loss':'%0.4f'%valid_loss, 'valid_acc':'%0.4f'%valid_acc})
                    return loss, acc, valid_loss, valid_acc
                else:
                    return loss, acc

def valid(net_list, criterion, valid_loader, device=None):
    for net in net_list:
        net.eval()
    loss_list, p_list, y_list = [], [], []
    for x, y in valid_loader:
        if device is not None:
            x, y = x.to(device), y.to(device)
        p = x
        for net in net_list:
            p = net(p)

        loss = criterion(p, y)
        loss_list.append(loss.item())

        _, p = p.max(axis=-1)
        y_list.append(y.detach().cpu().numpy())
        p_list.append(p.detach().cpu().numpy())
    
    loss = np.array(loss_list).mean()
    acc = (np.array(p_list) == np.array(y_list)).astype(np.float32).mean()
    
    return loss, acc

def predict(net_list, data_loader, device=None):
    for net in net_list:
        net.eval()
    
    p_list = []
    for x, _ in data_loader:
        if device is not None:
            x = x.to(device)
        p = x
        for net in net_list:
            p = net(p)
        p_list.append(p.detach().cpu().numpy())
    
    return np.concatenate(p_list)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # train_data 
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    train_dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)    
    # test_data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    test_dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    def train_loop(net_backbone, net_predict, optimizer_params, epoch_number, device, name):
        if device == 'cuda':
            net_backbone = torch.nn.DataParallel(net_backbone)
            net_predict = torch.nn.DataParallel(net_predict)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(optimizer_params, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_number)

        writer = SummaryWriter(log_dir=f'./runs/{name}')
        acc_valid_list = []
        for epoch in range(epoch_number):
            train_loss, train_acc, valid_loss, valid_acc = train(epoch, [net_backbone, net_predict], train_loader, criterion, optimizer, scheduler, test_loader, device=device)
            
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('valid_loss', valid_loss, epoch)
            writer.add_scalar('valid_acc', valid_acc, epoch)

            acc_valid_list.append(valid_acc)
            max_idx = int(np.argmax(acc_valid_list, axis=0))
            if max_idx == len(acc_valid_list) - 1:
                all_states = {'epoch': epoch, 
                              'train_loss': train_loss,
                              'net_backbone': net_backbone.state_dict(),
                              'net_linear': net_predict.state_dict(),
                              'train_acc': train_acc,
                              'valid_loss': valid_loss,
                              'valid_acc': valid_acc}
                torch.save(obj=all_states, f=f'./models/{name}_best.pth')
            elif max_idx <= len(acc_valid_list) - 16:
                break

    print('Backbone and linear, train all!')

    net_backbone = VGG_Backbone(cfg='VGG').to(device)
    net_linear = Linear_predict(256*4, 10).to(device)

    # 这里读取模型

    optimizer_params = [{'params': net_backbone.parameters(), 'lr': 0.001}, {'params': net_linear.parameters(), 'lr': 0.001}]
    train_loop(net_backbone, net_linear, optimizer_params, 200, device, 'Init_train')
    
    for t in range(100):
        print('Backbone and hwlayer, train hwlayer!')

        x = predict([net_backbone], test_loader, device)
        evaluate = evaluate_build(x, 64)
        evaluate_focus = focus_build(evaluate, 0.8)
        net_hwlayer = HWlayer_predict(evaluate_focus, 256*4).to(device)
        
        optimizer_params = [{'params': net_hwlayer.parameters(), 'lr': 0.001}]
        train_loop(net_backbone, net_hwlayer, optimizer_params, 200, device, f'hwlayer_train_{t}')

        print('#3 step: backbone and hwlayer, train all!')

        optimizer_params = [{'params': net_backbone.parameters(), 'lr': 0.0001}, {'params': net_hwlayer.parameters(), 'lr': 0.0001}]
        train_loop(net_backbone, net_hwlayer, optimizer_params, 200, device, f'hwlayer_train_{t}')