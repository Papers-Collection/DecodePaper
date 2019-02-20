import time

from mxnet import metric, autograd
from mxnet.gluon import loss as gloss, Trainer
from gluoncv.utils import TrainingHistory  # 可视化

from utils import make_dirs  # 创建多个目录


class Tools:
    def __init__(self, datasetName):
        self._get_result_dir(datasetName)

    def _get_result_dir(self, datasetName):
        self.modelDir = f'models/{datasetName}'
        self.resultDir = f'results/{datasetName}'
        make_dirs(self.modelDir, self.resultDir)


def evaluate_loss(data_iter, net, ctx, loss):
    l_sum, n = 0.0, 0
    #loss = gloss.SoftmaxCrossEntropyLoss()
    for X, y in data_iter:
        y = y.as_in_context(ctx).astype('float32')  # 模型的输出是 float32 类型数据
        outputs = net(X.as_in_context(ctx))  # 模型的输出
        l_sum += loss(outputs, y).sum().asscalar()  # 计算总损失
        n += y.size  # 计算样本数
    return l_sum / n  # 计算平均损失


def test(valid_iter, net, ctx):
    val_metric = metric.Accuracy()
    for X, y in valid_iter:
        X = X.as_in_context(ctx)
        y = y.as_in_context(ctx).astype('float32')  # 模型的输出是 float32 类型数据
        outputs = net(X)
        val_metric.update(y, outputs)
    return val_metric.get()


def get_result_dirs(datasetName):
    tools = Tools(datasetName)
    return tools.modelDir, tools.resultDir


def train(ctx,
          loss,
          trainer,
          datasetName,
          modelName,
          net,
          train_iter,
          valid_iter,
          num_epochs,
          n_retrain_epoch=0):
    '''
    n_retrain_epoch 是从第 n_retrain_epoch 次开始训练模型
    '''
    train_metric = metric.Accuracy()
    train_history = TrainingHistory(['training-error', 'validation-error'])
    best_val_score = 0
    modelDir, resultDir = get_result_dirs(datasetName)
    for epoch in range(num_epochs):
        train_l_sum, n, start = 0.0, 0, time.time()  # 计时开始
        train_metric.reset()
        for X, y in train_iter:
            X = X.as_in_context(ctx)
            y = y.as_in_context(ctx).astype('float32')  # 模型的输出是 float32 类型数据
            with autograd.record():  # 记录梯度信息
                outputs = net(X)  # 模型输出
                L = loss(outputs, y)
                l = L.mean()  # 计算总损失
            l.backward()  # 反向传播
            trainer.step(1)
            train_l_sum += L.sum().asscalar()  # 计算该批量的总损失
            train_metric.update(y, outputs)  # 计算训练精度
            n += y.size
        _, train_acc = train_metric.get()
        time_s = "time {:.2f} sec".format(time.time() - start)  # 计时结束
        valid_loss = evaluate_loss(valid_iter, net, ctx, loss)  # 计算验证集的平均损失
        _, val_acc = test(valid_iter, net, ctx)  # 计算验证集的精度
        epoch_s = (
            "epoch {:d}, train loss {:.5f}, valid loss {:.5f}, train acc {:.5f}, valid acc {:.5f}, ".
            format(n_retrain_epoch + epoch, train_l_sum / n, valid_loss,
                   train_acc, val_acc))
        print(epoch_s + time_s)
        train_history.update([1 - train_acc, 1 - val_acc])  # 更新图像的纵轴
        train_history.plot(save_path=f'{resultDir}/{modelName}_history.png')  # 实时更新图像
        if abs(train_acc-val_acc)>.2:  # 严重过拟合
            break
        if val_acc > best_val_score:  # 保存比较好的模型
            best_val_score = val_acc
            net.save_parameters('{}/{:.4f}-{}-{:d}-best.params'.format(
                modelDir, best_val_score, modelName, n_retrain_epoch + epoch))


def train_fine_tuning(datasetName,
                      modelName,
                      learning_rate,
                      net,
                      train_iter,
                      valid_iter,
                      num_epochs,
                      n_retrain_epoch=0):
    import d2lzh as d2l
    ctx = d2l.try_all_gpus()[0]
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate,
        'wd': 0.001
    })
    train(ctx, loss, trainer, datasetName, modelName, net, train_iter,
          valid_iter, num_epochs, n_retrain_epoch)