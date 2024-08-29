import torch
import numpy as np

import matplotlib.pyplot as plt

class HW_focus:
    def __init__(self):
        pass

    def __call__(self, x, target=0.6, resolution=1024):
        dim = x.shape[-1]

        d = x.unsqueeze(-1) - x.unsqueeze(-2)
        d = torch.square(d)

        f = torch.arange(0, torch.e * torch.sqrt(torch.tensor(dim, dtype=torch.float32)), torch.exp(torch.tensor(1.0)) / resolution)
        f = -1.0 * torch.exp(f) / torch.var(x)

        shape = d.shape

        s = d.unsqueeze(-2) * f.unsqueeze(0).unsqueeze(-1)
        s = torch.softmax(s, dim = -1)

        y, i = torch.max(s, dim = -1)
        
        y = torch.abs(y - target)

        y, i = torch.min(y, dim = -1)

        f = f[i]

        d = d * f.unsqueeze(-1)
        s = torch.softmax(d, dim = -1)

        s, i = torch.max(s, dim = -1)

        return f

def torch_test(dim = 16):
    x = np.random.randn(dim * 16).astype(np.float32)
    percentile = np.percentile(x, np.linspace(0, 100, dim * 2 + 1)[1::2])
    print("percentile:")
    for p in percentile:
        print(f"{p:.6f}", end=" ")
    print("")

    net = HW_focus()

    f = net(torch.tensor(percentile, dtype=torch.float32))
    f = f.detach().numpy()
    f = f.reshape((-1, 1))

    p = percentile.reshape((-1, 1)) - percentile.reshape((1, -1))
    d = np.square(p)

    print("softmax")
    s = d * f
    softmax = np.exp(s) / np.sum(np.exp(s), axis=-1, keepdims=True)

    plt.figure()  # Create a new figure for plotting
    for p, s in zip(percentile, softmax):
        plt.plot(percentile, s, label=f'Line {p:.8f}')  # Add a label for each line with 4 decimal places
    plt.title('Softmax Maximum vs k')
    plt.xlabel('k')
    plt.ylabel('Max Softmax Value')
    plt.grid()
    plt.legend()  # Show legend to identify lines
    plt.show()  # Show all plots on the same figure

def np_test(dim = 16):
    x = np.random.randn(dim * 16).astype(np.float32)
    percentile = np.percentile(x, np.linspace(0, 100, dim * 2 + 1)[1::2])

    variance = np.var(percentile)
    print(f"Mean Squared Deviation: {variance:.6f}")

    print("softmax")
    plt_list_dic = {p:[] for p in percentile}
    for k in np.arange(0, np.e * np.sqrt(dim), np.e / 64):
        focus = -1.0 * np.exp(k) / variance
        distance = np.reshape(percentile, (dim, 1)) - np.reshape(percentile, (1, dim))
        distance = np.square(distance) * focus
        soft_max = np.exp(distance) / np.sum(np.exp(distance), axis=-1, keepdims=True)

        for i, (p, s) in enumerate(zip(percentile, soft_max)):
            plt_list_dic[p].append([k, np.max(s)])

    plt.figure()  # Create a new figure for plotting
    for p, plt_list in plt_list_dic.items():
        plt_array = np.array(plt_list)
        plt.plot(plt_array[:, 0], plt_array[:, 1], label=f'Line {p:.8f}')  # Add a label for each line with 4 decimal places
    plt.title('Softmax Maximum vs k')
    plt.xlabel('k')
    plt.ylabel('Max Softmax Value')
    plt.grid()
    plt.legend()  # Show legend to identify lines
    plt.show()  # Show all plots on the same figure

if __name__ == '__main__':
    torch_test()
    
