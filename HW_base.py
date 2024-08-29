import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

from matplotlib import pyplot as plt
def plt_scatter(x, y_predict=None, y_true=None, title=None):
    plt.figure(figsize=(8, 4))
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.1, 1.1)

    if y_true is not None:
        plt.scatter(x, y_true, s=8, marker='o', c='y')

    if y_predict is not None:
        plt.scatter(x, y_predict, s=2, marker='o', c='b')

    if title is not None:
        plt.title(title, fontsize='xx-large', fontweight='normal')

    plt.show()

def idx_build(size, batch_size, shuffle):
    idx = np.arange(size)
    if shuffle:
        np.random.shuffle(idx)
    batch_num = size // batch_size + (1 if size % batch_size > 0 else 0)
    return [idx[i * batch_size:(i + 1) * batch_size] for i in range(batch_num)]

def evaluate_build(data, num):
    data = np.reshape(data, -1)
    percentile = np.linspace(0.0, 100.0, num * 2 + 1)
    evaluate = np.percentile(data, percentile)[1::2]
    return np.unique(evaluate)

def focus_process(e, evaluate, focus_target):
    f_array = np.linspace(0, 100, 10001, dtype=np.float32)
    while True:
        distances = np.abs(evaluate - e)
        scaled_distances = -1.0 * np.expand_dims(distances, axis=-1) * f_array
        scores = np.exp(scaled_distances) / np.sum(np.exp(scaled_distances), axis=0)
        max_score = np.max(scores, 0)
        if max_score[-1] < focus_target:
            f_array *= 10
        else:
            index = np.argmin(np.abs(max_score - focus_target))
            return [float(e), float(f_array[index]), float(max_score[index])]

def focus_build(evaluate, focus_target=0.8):
    evaluate_range = np.abs(evaluate.max() - evaluate.min())
    if evaluate_range == 0:
        return {e: 1.0 for e in evaluate}
    if focus_target > 0.99:
        return {e: 1e8 for e in evaluate}

    evaluate_focus_list = []
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        task_list = [executor.submit(focus_process, e, evaluate, focus_target) for e in evaluate]
        for future in tqdm(as_completed(task_list), ncols=100, desc=f'evaluate_num:{len(evaluate):4d},focus:{focus_target:0.4f}'):
            e, f, v = future.result()
            evaluate_focus_list.append([e, f, v])
    evaluate_focus_list.sort(key=lambda x: x[0])
    return evaluate_focus_list

class hw_layer(torch.nn.Module):
    def __init__(self, evaluate_focus_list):
        super(hw_layer, self).__init__()
        self.evaluate_list = torch.nn.ModuleList()
        self.focus_list = torch.nn.ModuleList()

        for evaluate_focus in evaluate_focus_list:
            evaluate_focus = np.array(evaluate_focus, dtype=np.float32)
            self.add_embedding(self.evaluate_list, evaluate_focus[:, 0])
            self.add_embedding(self.focus_list, evaluate_focus[:, 1])

        self.channels = sum(len(evaluate_focus) for evaluate_focus in evaluate_focus_list)

    def add_embedding(self, module_list, values):
        values = np.expand_dims(values, -1)
        values = torch.from_numpy(values)
        embedding = torch.nn.Embedding.from_pretrained(values, freeze=True)
        module_list.append(embedding)

    def forward(self, x):
        output_list = []
        
        if x.ndim == 1:
            raise ValueError("Input tensor x must be more than 2-dimensional")

        for i in range(x.shape[-1]):
            data = x[..., i:i+1]

            evaluate = self.evaluate_list[i].weight
            evaluate_shape = np.ones_like(data.shape)
            evaluate_shape[-1] = evaluate.shape[0]
            evaluate = evaluate.reshape(evaluate_shape)

            distance = torch.abs(data - evaluate)

            focus = self.focus_list[i]
            focus_idx = torch.argmin(distance, axis=-1)
            focus = focus(focus_idx)

            s = torch.mul(torch.mul(distance, focus), -1.0)
            s = torch.nn.functional.softmax(s, dim=-1)

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

def net_parameter_count(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad]), sum([p.numel() for p in model.parameters() if not p.requires_grad])

def torch_train(net_list, optimizer, loss_function, 
                train_data, batch_size, shuffle=True, 
                valid_data=None, valid_batch_size=None, desc='', to_device=None):
    loss_list = []
    idx_list = idx_build(len(train_data[0]), batch_size, shuffle)
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

if __name__ == '__main__':
    # data = np.random.randn(10000, 5)
    # evaluate_list = [evaluate_build(data[..., i], 100) for i in range(data.shape[-1])]
    # evaluate_focus_list = [focus_build(evaluate, 0.8) for evaluate in evaluate_list]
    # pass
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