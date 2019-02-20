import numpy as np


class Loader(dict):
    """
    方法
    ========
    L 为该类的实例
    len(L)::返回 batch 的批数
    iter(L)::即为数据迭代器

    参数
    =============
    type: 'train', 'test'

    Return
    ========
    可迭代对象（numpy 对象）
    """

    def __init__(self, batch_size, X, Y=None, shuffle=True, *args, **kwargs):
        '''
        X, Y 均为类 numpy, 可以是 HDF5 
        '''
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.batch_size = batch_size
        if shuffle:
            self.type = 'train'
        else:
            self.type = 'test'

        if not hasattr(X, 'take'):
            self.X = X[:]
        else:
            self.X = X
        self.nrows = len(self.X)
        if Y is not None:
            if not hasattr(Y, 'take'):
                self.Y = Y[:]
        else:
            self.Y = None

    def __iter__(self):
        idx = np.arange(self.nrows)
        if self.type == 'train':
            np.random.shuffle(idx)

        for start in range(0, self.nrows, self.batch_size):
            end = min(start + self.batch_size, self.nrows)
            K = idx[start:end].tolist()
            if self.Y is None:
                yield self.X.take(K, axis=0)
            else:
                yield self.X.take(K, axis=0), self.Y.take(K, axis=0)

    def __len__(self):
        return round(self.nrows / self.batch_size)  # 向上取整