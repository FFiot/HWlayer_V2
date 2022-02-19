import numpy as np

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

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
        for i in range(x.shape[1]):
            evaluate = self.evaluate_list[i].weight
            dims = evaluate.shape[0]
            evaluate = evaluate.reshape((1, dims, 1, 1))
            
            d = x[:, i, :, :]
            d = d.unsqueeze(1)
            d = d - evaluate
            d = d.abs()

            focus = self.focus_list[i] 
            focus_idx = d.argmin(axis=1)
            focus = focus(focus_idx)

            pass
        return x

class HWlayer_poolling2D(nn.Module):
    def __init__(self, deepth, kernel_size=2, stride=2):
        super(HWlayer_poolling2D, self).__init__()
        self.deepth = deepth
        self.poolling_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    
    def forward(self, x):
        output_list = [x]
        for _ in range(self.deepth):
            y = self.poolling_layer(output_list[-1])
            output_list.append(y)
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

    def forward(self, x):
        x_poolling_list = [x]
        for poolling in self.poolling_list:
            x_poolling_list.append(poolling(x_poolling_list[-1]))

        x_hw_list = [hw(x) for x, hw in zip(x_poolling_list, self.HWlayer_list)]
        return x_hw_list


if __name__ == '__main__':
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    train_dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deepth = 2

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
        evaluate_focus_list = [focus_build(evaluate, 0.8) for evaluate in evaluate_list]
        evaluate_focus_table.append(evaluate_focus_list)

    net = HWlayer_VGG(evaluate_focus_table).to(device)

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        p = net(x)
