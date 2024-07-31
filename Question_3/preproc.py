import numpy as np

class PreProc:
    def __init__(self):
        pass
    
    def one_hot_encode(self, y, levels):
        res = np.zeros((len(y), levels))
        for i in range(len(y)):
            res[i, y[i]] = 1
        return res

    def normalize(self, x):
        return x / np.max(x)

    def read_csv(self, fname):
        data = np.loadtxt(fname, skiprows=1, delimiter=',')
        y = data[:, :1]
        x = data[:, 1:]
        return x, y

    def load_data(self, fname):
        x, y = self.read_csv(fname)
        x = self.normalize(x)
        y = np.int16(y)
        y = self.one_hot_encode(y, levels=10)

        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        return x, y
