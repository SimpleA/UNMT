import numpy as np
import torch

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


def sentence_swap(index, sentence_embedding):
    for (b_idx, b_sent) in zip(index, sentence_embedding):
        swap_times = int((b_idx.size(0) - b_idx.nonzero().size(0)) / 2)
        for i in range(swap_times):
            random_idx = np.random.randint(b_idx.nonzero().size(0) - 1)
            tmp = b_sent[random_idx + 1].clone()
            b_sent[random_idx + 1] = b_sent[random_idx]
            b_sent[random_idx] = tmp

    return sentence_embedding


def shuffle(x, y):
    assert x.shape[0] == y.shape[0]
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    return x[indices], y[indices]
