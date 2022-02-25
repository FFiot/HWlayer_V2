from multiprocessing import cpu_count
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

class HWlayer2D(nn.Module):
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

class HWlayer_poolling2D(nn.Module):
    def __init__(self, deepth=2, kernel_size=2, stride=2):
        super(HWlayer_poolling2D, self).__init__()
        self.deepth = deepth
        self.poolling_list = torch.nn.ModuleList()
        for i in range(deepth):
            self.poolling_list.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

    def forward(self, x):
        output_list = [x]
        for poolling in self.poolling_list:
            output_list.append(poolling(output_list[-1]))

        return tuple(output_list)

class HWlayer_VGG(nn.Module):
    def __init__(self, evaluate_focus_table):
        super(HWlayer_VGG, self).__init__()
        self.HWlayer_list = torch.nn.ModuleList()
        for evaluate_focus_list in evaluate_focus_table:
            self.HWlayer_list.append(HWlayer2D(evaluate_focus_list))

        self.poolling_list = torch.nn.ModuleList()
        for _ in range(len(evaluate_focus_table)-1):
            self.poolling_list.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.s1 = nn.Sequential(
            nn.Conv2d(3, 24*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(24*4),
            nn.ReLU(inplace=True))

        self.s2 = nn.Sequential(
            nn.Conv2d(24*4, 24*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(24*4),
            nn.ReLU(inplace=True))

        self.s3 = nn.Sequential(
            nn.Conv2d(24*4, 24*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(24*4),
            nn.ReLU(inplace=True))

        self.s4 = nn.Sequential(
            nn.Conv2d(24*4, 24*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(24*4),
            nn.ReLU(inplace=True))
        
        self.s5 = nn.Sequential(
            nn.Conv2d(24*4, 24*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(24*4),
            nn.ReLU(inplace=True))

        self.s6 = nn.Sequential(
            nn.Conv2d(24*4, 24*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(24*4),
            nn.ReLU(inplace=True))

        self.s7 = nn.Sequential(
            nn.Conv2d(24*4, 24*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(24*4),
            nn.ReLU(inplace=True))

        self.s8 = nn.Sequential(
            nn.Conv2d(24*4, 24*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(24*4),
            nn.ReLU(inplace=True))
        
        self.s9 = nn.Sequential(
            nn.Conv2d(24*4, 24*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(24*4),
            nn.ReLU(inplace=True))

        self.s10 = nn.Sequential(
            nn.Conv2d(24*4, 24*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(24*4),
            nn.ReLU(inplace=True))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*24*4, 64),
            nn.Linear(64, 10))
    
    def forward(self, x):
        x_poolling_list = [x]
        for poolling in self.poolling_list:
            x_poolling_list.append(poolling(x_poolling_list[-1]))

        hw_list = [hw(p) for p, hw in zip(x_poolling_list, self.HWlayer_list)]
        hw_acitve_dic = {hw.shape[-1]:hw.unsqueeze(1) for hw in hw_list}

        # x = hw_list[0]

        def hw_acitve(x, hw_acitve_dic, res=None):
            dims = x.shape[-1]
            if dims not in hw_acitve_dic:
                return x
            hw_layer = hw_acitve_dic[dims]
            
            dims = hw_layer.shape[2]
            x_shape = list(x.shape)
            shape = [x_shape[0]] + [x_shape[1]//dims, dims] +x_shape[2:]
            y = x.reshape(shape) * hw_layer
            y = y.reshape(x_shape)
            if res is not None:
                return y + res
            else:
                return y

        x = self.s1(x)
        x = self.s2(x)

        x = self.p1(x)

        x = self.s3(x)
        x = self.s4(x)

        x = self.p2(x)

        x = self.s5(x)
        x = self.s6(x)
        

        x = self.p3(x)

        x = self.s7(x)
        x = self.s8(x)

        x = self.p4(x)

        x = self.fc(x)

        return x

if __name__ == '__main__':
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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deepth = 3

    output_list = [[] for _ in range(deepth+1)]
    net = HWlayer_poolling2D(deepth).to(device)
    for x, y in test_loader:
        x = x.to(device)
        y = net(x)
        for i, d in enumerate(y):
            output_list[i].append(d.detach().cpu().numpy())

    evaluate_focus_table = []
    for image_array in output_list:
        image_array = np.concatenate(image_array, axis=0)
        evaluate_list = [evaluate_build(image_array[:, i, :, :], 8) for i in range(d.shape[1])]
        focus_list = [0.9, 0.9, 0.9]
        evaluate_focus_list = [focus_build(evaluate, focus) for evaluate, focus in zip(evaluate_list, focus_list)]
        evaluate_focus_table.append(evaluate_focus_list)

    net = HWlayer_VGG(evaluate_focus_table).to(device)

    summary(net, input_size=(3, 32, 32))

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    def valid(epoch):
        net.eval()
        loss_list, p_list, y_list = [], [], []
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            p = net(x)

            loss = criterion(p, y)
            loss_list.append(loss.item())

            _, p = p.max(axis=-1)
            y_list.append(y.detach().cpu().numpy())
            p_list.append(p.detach().cpu().numpy())
        
        loss = np.array(loss_list).mean()
        acc = (np.array(p_list) == np.array(y_list)).astype(np.float32).mean()
        
        return loss, acc

    def train(epoch):
        net.train()        
        loss_list, p_list, y_list = [], [], []
        with tqdm((enumerate(train_loader)), desc='epoch%3d'%epoch, total=len(train_loader), ncols=0) as t:
            for idx, (x, y) in t:
                x, y = x.to(device), y.to(device)
                p = net(x)
                
                optimizer.zero_grad()
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
                    
                    valid_loss, valid_acc = valid(epoch)
                    t.set_postfix({'loss':'%0.4f'%loss, 'acc':'%0.4f'%acc, 'valid_loss':'%0.4f'%valid_loss, 'valid_acc':'%0.4f'%valid_acc})
            return loss, acc, valid_loss, valid_acc

    writer = SummaryWriter()
    # writer.add_graph(net, input_to_model=torch.from_numpy(np.random.randn(2,3,32,32).astype(np.float32)).to(device), verbose=False)
    for epoch in range(200):
        train_loss, train_acc, valid_loss, valid_acc = train(epoch)
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('acc/train', train_acc, epoch)
        writer.add_scalar('loss/valid', valid_loss, epoch)
        writer.add_scalar('acc/valid', valid_acc, epoch)
        if valid_acc / train_acc < 0.9:
            break