import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

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
    batch_num = size//batch_size + 1 if size % batch_size > 0 else size//batch_size
    return [idx[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

def evaluate_build(data, num):
    data = np.reshape(data, -1)
    percentile = np.linspace(0.0, 100.0, num*2+1)
    evaluate = np.percentile(data, percentile)
    evaluate = [evaluate[i] for i in range(1, len(evaluate), 2)]
    evaluate = np.unique(evaluate)
    return evaluate

def focus_process(e, evaluate, focus_target):
    f_array = np.linspace(0, 100, 10000+1, dtype=np.float32)
    while True:
        d = np.abs(evaluate - e)
        d = np.expand_dims(d, axis=-1)
        s = d * -1.0 * f_array
        s = np.exp(s)
        s = s / np.sum(s, axis=0)
        s = np.max(s, 0)
        if s[-1] < focus_target:
            f_array *= 10
        else:
            i = np.abs(s - focus_target)
            i = np.argmin(i)
            v = s[i]
            f = f_array[i]
            return [float(e), float(f), float(v)]

def focus_build(evaluate, focus_target=0.8):
    evaluate_range = np.abs(evaluate.max() - evaluate.min())
    if evaluate_range == 0:
        return {e:1.0 for e in evaluate}

    if  focus_target > 0.99:
        return {e:1e8 for e in evaluate}
    
    evaluate_focus_list = []
    with ThreadPoolExecutor(max_workers=cpu_count()) as t:
        tast_list = [t.submit(focus_process, e, evaluate, focus_target) for e in evaluate]
        for future in tqdm(as_completed(tast_list), ncols=100, desc='evaluate_num:%4d,focus:%0.4f'%(len(evaluate), focus_target)):
            e, f, v = future.result()
            evaluate_focus_list.append([e, f, v])
    evaluate_focus_list.sort(key=lambda x:x[0])
    return evaluate_focus_list

def data_build(x, evaluate_focus_list):
    target_shape = list(x.shape)
    if target_shape[-1] == 1:
        target_shape[-1] = len(evaluate_focus_list)
    else:
        target_shape.append(len(evaluate_focus_list))

    x = np.reshape(x, (-1, 1))
    e = evaluate_focus_list[:,0]
    e = np.reshape(e, (1, -1))
    d = np.abs(x - e)
    i = np.argmin(d, -1)
    f = evaluate_focus_list[:,1]
    f = f[i]
    f = np.expand_dims(f, -1)
    e = np.exp(d * -1.0 * f)
    s = np.sum(e, axis=-1)
    s = np.expand_dims(s, -1)
    y = e / s
    y = y.astype(np.float32)
    y = np.reshape(y, target_shape)

    return y

if __name__ == '__main__':
    data = np.random.randn(10000, 5)
    evaluate_list = [evaluate_build(data[..., i], 100) for i in range(data.shape[-1])]
    evaluate_focus_list = [focus_build(evaluate, 0.8) for evaluate in evaluate_list]
    pass