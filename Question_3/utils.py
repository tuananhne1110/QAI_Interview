import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = np.sum((anchor - positive) ** 2, axis=-1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=-1)
    return np.maximum(pos_dist - neg_dist + margin, 0.0)

def shuffle(x, y):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    return x[idx,], y[idx,]
