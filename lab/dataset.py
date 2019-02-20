import numpy as np

from mxnet import nd
from mxnet.gluon import data as gdata
from mxnet.gluon.data.vision import transforms as gtf

from datatools import Loader

# 数据增强
transform_train = gtf.Compose([
    # 随机对图像裁剪出面积为原图像面积0.08~1倍、且高和宽之比在3/4~4/3的图像，再放缩为高和
    # 宽都是为 224 的新图
    gtf.RandomResizedCrop(
        224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    gtf.RandomFlipLeftRight(),
    # 随机变化亮度、对比度和饱和度
    gtf.RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    # 随机加噪声
    gtf.RandomLighting(0.1),
    gtf.ToTensor(),
    # 对图像的每个通道做标准化
    gtf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = gtf.Compose([
    gtf.Resize(256),
    # 将图像中央的高和宽均为 224 的正方形区域裁剪出来
    gtf.CenterCrop(224),
    gtf.ToTensor(),
    gtf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class SimpleDataset:
    def __init__(self, name, numLabels, root='E:/xdata/X.h5'):
        import tables as tb
        h5 = tb.open_file(root)
        self.name = name
        self._dataset = h5.root[name]
        self.label_names = self._get_label_names(is_fine_labels=False)
        self._split(numLabels)
        self.testX, self.testY = self._dataset.testX[:], self._dataset.testY[:]
        h5.close()

    def _get_label_names(self, is_fine_labels=False):
        if self.name != 'cifar100':
            return np.asanyarray(self._dataset.label_names, dtype='U')
        elif is_fine_labels:
            return np.asanyarray(self._dataset.fine_label_names, dtype='U')
        else:
            return np.asanyarray(self._dataset.coarse_label_names, dtype='U')

    def _split(self, numLabels):
        from sklearn.model_selection import train_test_split
        xTr = self._dataset.trainX[:]
        yTr = self._dataset.trainY[:]
        test_size = xTr.shape[0] - numLabels
        # 数据集划分操作
        self.trainX, self.valX, self.trainY, self.valY = train_test_split(
            xTr, yTr, test_size=test_size, shuffle=True)


class AugLoader(Loader, gdata.Dataset):
    def __init__(self, batch_size, X, Y=None, shuffle=True, *args, **kwargs):
        super().__init__(batch_size, X, Y=None, shuffle=True, *args, **kwargs)
        self.X = nd.array(X[:])
        if not Y is None:
            self.Y = nd.array(Y[:])

    def aug_imgs(self, imgs):
        '''
        对 图像做数据增强 预处理
        dataset 需要有 type 属性（'train', 'test'）
        '''
        transforms_dict = {'train': transform_train, 'test': transform_test}
        return nd.stack(*[transforms_dict[self.type](img) for img in imgs])

    def __iter__(self):
        idx = np.arange(self.nrows)
        if self.type == 'train':
            np.random.shuffle(idx)
        for start in range(0, self.nrows, self.batch_size):
            end = min(start + self.batch_size, self.nrows)
            K = nd.array(idx[start:end])
            if self.Y is None:
                yield self.aug_imgs(self.X.take(K, 0))
            else:
                yield self.aug_imgs(self.X.take(K, 0)), self.Y.take(K, 0)