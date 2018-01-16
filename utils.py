import numpy as np
import torch
from visdom import Visdom

USE_CUDA = torch.cuda.is_available()


def tfmt(s):
    m = s // 60
    s = s % 60

    h = m // 60
    m = m % 60

    if h != 0:
        f = '%d h %d m %d s' % (h, m, s)
    elif m != 0:
        f = '%d m %d s' % (m, s)
    else:
        f = '%d s' % s
    return f


def sentence_swap(batch_lengths, batch_sentences):
    for (length, sentence) in zip(batch_lengths, batch_sentences):
        swap_times = int(length / 2)
        for i in range(swap_times):
            random_idx = np.random.randint(length - 1)
            tmp = sentence[random_idx + 1].clone()
            sentence[random_idx + 1] = sentence[random_idx]
            sentence[random_idx] = tmp

    return batch_sentences


def shuffle(x, y):
    assert x.shape[0] == y.shape[0]
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    return x[indices], y[indices]

class VisdomPlotter(object):
    def __init__(self, env_name='UNMT'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, xlabel='iteration'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X = np.array([x, x]),
                Y = np.array([y, y]),
                env = self.env,
                opts = dict(
                    legend = [split_name],
                    title = var_name,
                    xlabel = xlabel,
                    ylabel = var_name
                )
            )
        else:
            self.viz.updateTrace(
                X = np.array([x]),
                Y = np.array([y]),
                env = self.env,
                win = self.plots[var_name],
                name = split_name
            )
