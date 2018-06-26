import tensorflow as tf
import tensorflow.contrib.eager as tfe
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
tfe.enable_eager_execution()

class simple_nn(tf.keras.Model):

    def __init__(self):
        '''
        简单的网络，中间层包含10个units，输出为
        '''
        super().__init__()
        self.hidden_layer = tf.layers.Dense(units=10,activation=tf.nn.relu)
        self.output_layer = tf.layers.Dense(units=2)

    def predict(self,input_data):
        '''
        从前到后跑一遍模型
        :param input_data: 二维张量(n_samples, n_features)
        :return: 分类结果
        '''
        middle = self.hidden_layer(input_data)
        logits = self.output_layer(middle)
        return logits

    def loss_fn(self,input_data,target):
        '''
        定义损失函数
        :param input_data:
        :param target:
        :return:
        '''
        logits = self.predict(input_data)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=target, logits=logits)
        return loss

    def grads_fn(self,input_data,target):
        '''
        计算每一趟的梯度，return和with对齐
        :param input_data:
        :param target:
        :return:
        '''
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(input_data, target)
        return tape.gradient(loss, self.variables)

    def fit(self,input_data,target,optimizer,num_epochs=500, verbose=50):
        '''
        拟合数据和模型
        :param input_data:
        :param target:
        :param optimizer:
        :param num_epochs:
        :param verbose:
        :return:
        '''
        for i in range(num_epochs):
            grads = self.grads_fn(input_data, target)
            optimizer.apply_gradients(zip(grads, self.variables))
            if (i == 0) | ((i + 1) % verbose == 0):
                print('Loss at epoch %d: %f' % (i + 1, self.loss_fn(input_data, target).numpy()))


def plot_boundary(data, model):

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = np.argmax(model.predict(tf.constant(np.c_[xx.ravel(), yy.ravel()])).numpy(), axis=1)
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.autumn, alpha=0.8)
    plt.scatter(data[:, 0], data[:, 1], c=y, s=40, cmap=plt.cm.autumn, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('First feature', fontsize=15)
    plt.ylabel('Second feature', fontsize=15)
    plt.title('Toy classification problem', fontsize=15)
    plt.show()


if __name__ == '__main__':
    # 分类数据集来自sklearn.datasets
    # X矩阵(n_samples, n_features)
    # y是一个向量(n_samples,)
    X, y = make_moons(n_samples=100, noise=0.1, random_state=2018)
    print(X.shape,y.shape)
    print(type(X),type(y))

    # Loss at epoch 500: 0.098040
    # optimizer = tf.train.GradientDescentOptimizer(5e-1)
    # Loss at epoch 500: 0.000145  nb -0 -
    optimizer = tf.train.AdamOptimizer(5e-1)

    model = simple_nn()
    model.fit(tf.constant(X),tf.constant(y), optimizer, num_epochs=500, verbose=20)
    plot_boundary(X,model)
