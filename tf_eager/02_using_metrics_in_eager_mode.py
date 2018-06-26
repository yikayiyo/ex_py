# -*-encoding='utf-8' -*-
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from sklearn.datasets import load_wine
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# from sklearn.decomposition import PCA

tfe.enable_eager_execution()


class two_layers_nn(tf.keras.Model):

    def __init__(self,output_size=2, loss_type='cross-entropy'):
        super(two_layers_nn, self).__init__()
        self.hidden_layer1 = tf.layers.Dense(units=20, activation=tf.nn.relu)
        self.hidden_layer2 = tf.layers.Dense(units=10, activation=tf.nn.relu)
        self.output_layer = tf.layers.Dense(output_size)
        self.loss_type = loss_type

    def predict(self, input_data):
        hidden1 = self.hidden_layer1(input_data)
        hidden2 = self.hidden_layer2(hidden1)
        out = self.output_layer(hidden2)
        return out

    def loss_fn(self, input_data, target):
        logits = self.predict(input_data)
        if self.loss_type == 'cross-entropy':
            loss = tf.losses.sparse_softmax_cross_entropy(labels=target, logits=logits)
        else:
            loss = tf.losses.mean_squared_error(target, preds)
        return loss

    def grads_fn(self, input_data, target):
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(input_data, target)
        return tape.gradient(loss, self.variables)

    def fit(self, input_data, target, optimizer, num_epochs=500, verbose=20, track_accuracy=True):

        if track_accuracy:
            # 保存模型准确率
            self.hist_accuracy = []
            # Object-oriented metrics
            accuracy = tfe.metrics.Accuracy()

        for i in range(num_epochs):
            grads = self.grads_fn(input_data, target)
            optimizer.apply_gradients(zip(grads, self.variables))
            if track_accuracy:
                logits = self.predict(input_data)
                #InvalidArgumentError: cannot compute Equal as input #0 was expected to be a int32 tensor but is a int64 tensor
                # preds = tf.argmax(logits, axis=1)

                # tf.int64 --> tf.int32
                preds = tf.cast(tf.argmax(logits, axis=1),dtype=tf.int32)
                # 计算accuracy
                accuracy(preds, target)
                # reslut放入列表将来用于可视化
                self.hist_accuracy.append(accuracy.result())
                # 一趟完成后 reset accuracy
                accuracy.init_variables()

def load_data(partname='part1'):
    if partname=='part1':
        wine_data = load_wine()
        # print(list(wine_data.keys())) # ['data', 'target', 'target_names', 'DESCR', 'feature_names']
        # print('类别数目: ', len(np.unique(wine_data.target)))# 3
        # print('类别分布: ', np.unique(wine_data.target, return_counts=True)[1])# [59 71 48]
        # print('特征数目: ', wine_data.data.shape[1])# 13

        # 标准化各个特征
        # print(wine_data.data)#[[ 1.51861254 -0.5622498   0.23205254 ...  0.36217728  1.84791957  1.01300893]...[...]]
        data = (wine_data.data - np.mean(wine_data.data, axis=0)) / np.std(wine_data.data, axis=0)
        target = wine_data.target
        # print(data)
        # print('标准化之后各个特征的mean: ',np.mean(data,axis=0))
        # print('标准化之后各个特征的std: ', np.std(data, axis=0))

        # pca降维到平面
        # plt.figure(figsize=(20, 10))
        # plt.subplot(121)
        # X_pca = PCA(n_components=2, random_state=2018).fit_transform(data)
        # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=wine_data.target, cmap=plt.cm.spring)
        # plt.xlabel('特征1', fontsize=15)
        # plt.ylabel('特征2', fontsize=15)
        # plt.title('PCA标准化数据-多分类', fontsize=15)
        #
        # plt.subplot(122)
        # ori_data_pca = PCA(n_components=2, random_state=2018).fit_transform(wine_data.data)
        # plt.scatter(ori_data_pca[:, 0], ori_data_pca[:, 1], c=wine_data.target, cmap=plt.cm.spring)
        # plt.xlabel('特征1', fontsize=15)
        # plt.ylabel('特征2', fontsize=15)
        # plt.title('PCA未标准化数据-多分类', fontsize=15)
        # plt.show()
        return data,target
    elif partname=='part2':
        pass
    else:
        pass
def precision(labels, predictions, conf_matrix=None, weights=None):

    # if not conf_matrix:  #error tenor--boolean
    if conf_matrix is None:
        conf_matrix = tf.confusion_matrix(labels, predictions, num_classes=3)
    tp_and_fp = tf.reduce_sum(conf_matrix, axis=0)
    tp = tf.diag_part(conf_matrix)
    precision_scores = tp/(tp_and_fp)
    if weights:
        precision_score = tf.multiply(precision_scores, weights)/tf.reduce_sum(weights)
    else:
        precision_score = tf.reduce_mean(precision_scores)
    return precision_score

def recall(labels, predictions, conf_matrix=None, weights=None):
    if conf_matrix is None:
        conf_matrix = tf.confusion_matrix(labels, predictions, num_classes=3)
    tp_and_fn = tf.reduce_sum(conf_matrix, axis=1)
    tp = tf.diag_part(conf_matrix)
    recall_scores = tp/(tp_and_fn)
    if weights:
        recall_score = tf.multiply(recall_scores, weights)/tf.reduce_sum(weights)
    else:
        recall_score = tf.reduce_mean(recall_scores)
    return recall_score

def part1():
    X, y = load_data('part1')
    X = tf.constant(X)
    y = tf.constant(y)
    optimizer = tf.train.GradientDescentOptimizer(5e-1)

    # Initialize model
    model = two_layers_nn(output_size=3)
    # Select here the number of epochs
    num_epochs = 10
    # Train the model with gradient descent
    model.fit(X, y, optimizer, num_epochs=num_epochs)
    # plt.figure(figsize=(10,8))
    # plt.plot(range(num_epochs), model.hist_accuracy)
    # plt.xlabel('训练趟数', fontsize=15)
    # plt.ylabel('准确度', fontsize=15)
    # plt.title('趋势图', fontsize=15)
    # plt.show()

    # model训练完后通过混淆矩阵观察其performance
    logits = model.predict(X)
    preds = tf.argmax(logits, axis=1)
    conf_matrix = tf.confusion_matrix(y, preds, num_classes=3)
    '''
    混淆矩阵: 主对角线是true positives
         [[58  1  0]
         [ 0 69  2]
         [ 1  0 47]]
    '''
    # print('混淆矩阵: \n', conf_matrix.numpy())

    # Average precision: 0.9794986487745782
    # Average recall: 0.978356011776876
    precision_score = precision(y, preds, conf_matrix, weights=None)
    print('Average precision: ', precision_score.numpy())
    recall_score = recall(y, preds, conf_matrix, weights=None)
    print('Average recall: ', recall_score.numpy())

def part2():
    load_data('part2')


def part3():
    pass

if __name__ == '__main__':
    part1()
    # part2()
    # part3()