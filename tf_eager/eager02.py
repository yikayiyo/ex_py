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

# tfe.enable_eager_execution()


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
        elif self.loss_type == 'regression':
            loss = tf.losses.mean_squared_error(target, logits)
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
        X = (wine_data.data - np.mean(wine_data.data, axis=0)) / np.std(wine_data.data, axis=0)
        y = wine_data.target
        # print(X)
        # print('标准化之后各个特征的mean: ',np.mean(X,axis=0))
        # print('标准化之后各个特征的std: ', np.std(X, axis=0))

        # pca降维到平面
        # plt.figure(figsize=(20, 10))
        # plt.subplot(121)
        # X_pca = PCA(n_components=2, random_state=2018).fit_transform(X)
        # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.spring)
        # plt.xlabel('特征1', fontsize=15)
        # plt.ylabel('特征2', fontsize=15)
        # plt.title('PCA标准化数据-多分类', fontsize=15)
        #
        # plt.subplot(122)
        # ori_data_pca = PCA(n_components=2, random_state=2018).fit_transform(wine_data.data)
        # plt.scatter(ori_data_pca[:, 0], ori_data_pca[:, 1], c=y, cmap=plt.cm.spring)
        # plt.xlabel('特征1', fontsize=15)
        # plt.ylabel('特征2', fontsize=15)
        # plt.title('PCA未标准化数据-多分类', fontsize=15)
        # plt.show()
        return X,y
    elif partname=='part2':
        X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                                   n_redundant=0, n_classes=2, n_clusters_per_class=1,
                                   flip_y=0.1, class_sep=4, hypercube=False,
                                   shift=0.0, scale=1.0, random_state=2018)

        # Reduce the number of samples with target 1
        X = np.vstack([X[y == 0], X[y == 1][:50]])
        y = np.hstack([y[y == 0], y[y == 1][:50]])
        # plt.figure(figsize=(10,8))
        # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.winter)
        # plt.xlabel('特征1', fontsize=15)
        # plt.ylabel('特征2', fontsize=15)
        # plt.title('不平衡分类问题', fontsize=15)
        # plt.show()

        return X,y
    else:
        X, y = make_regression(n_samples=100, n_features=1, n_informative=1, noise=30,
                               random_state=2018)
        return X,y

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


def roc_auc(labels, predictions, thresholds, get_fpr_tpr=True):
    tpr = []
    fpr = []
    for th in thresholds:
        # Compute number of true positives
        tp_cases = tf.where((tf.greater_equal(predictions, th)) &
                            (tf.equal(labels, 1)))
        tp = tf.size(tp_cases)
        # Compute number of true negatives
        tn_cases = tf.where((tf.less(predictions, th)) &
                            (tf.equal(labels, 0)))
        tn = tf.size(tn_cases)
        # Compute number of false positives
        fp_cases = tf.where((tf.greater_equal(predictions, th)) &
                            (tf.equal(labels, 0)))
        fp = tf.size(fp_cases)
        # Compute number of false negatives
        fn_cases = tf.where((tf.less(predictions, th)) &
                            (tf.equal(labels, 1)))
        fn = tf.size(fn_cases)
        # Compute True Positive Rate for this threshold
        tpr_th = tp / (tp + fn)
        # Compute the False Positive Rate for this threshold
        fpr_th = fp / (fp + tn)
        # Append to the entire True Positive Rate list
        tpr.append(tpr_th)
        # Append to the entire False Positive Rate list
        fpr.append(fpr_th)

    # Approximate area under the curve using Riemann sums and the trapezoidal rule
    auc_score = 0
    for i in range(0, len(thresholds) - 1):
        height_step = tf.abs(fpr[i + 1] - fpr[i])
        b1 = tpr[i]
        b2 = tpr[i + 1]
        step_area = height_step * (b1 + b2) / 2
        auc_score += step_area
    return auc_score, fpr, tpr

# Compute the R2 score
def r2(labels, predictions):
    mean_labels = tf.reduce_mean(labels)
    total_sum_squares = tf.reduce_sum((labels-mean_labels)**2)
    residual_sum_squares = tf.reduce_sum((labels-predictions)**2)
    r2_score = 1 - residual_sum_squares/total_sum_squares
    return r2_score

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
    plt.figure(figsize=(10,8))
    plt.plot(range(num_epochs), model.hist_accuracy)
    plt.xlabel('训练趟数', fontsize=15)
    plt.ylabel('准确率', fontsize=15)
    plt.title('多分类问题', fontsize=15)
    plt.show()

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
    X,y = load_data('part2')
    X = tf.constant(X)
    y = tf.constant(y)
    # Define optimizer
    optimizer = tf.train.GradientDescentOptimizer(5e-1)
    model = two_layers_nn(output_size=2)
    num_epochs = 5
    # Train the model with gradient descent
    model.fit(X, y, optimizer, num_epochs=num_epochs)
    # More thresholds means higher granularity for the area under the curve approximation
    # Feel free to experiment with the number of thresholds
    num_thresholds = 1000
    thresholds = tf.lin_space(0.0, 1.0, num_thresholds).numpy()

    # Apply Softmax on our predictions as the output of the model is unnormalized# Apply
    # Select the predictions of our positive class (the class with less samples)
    preds = tf.nn.softmax(model.predict(X))[:, 1]

    # Compute the ROC-AUC score and get the TPR and FPR of each threshold
    auc_score, fpr_list, tpr_list = roc_auc(y, preds, thresholds)
    print('模型的ROC-AUC : ', auc_score.numpy())
    plt.figure(figsize=(10,8))
    plt.plot(fpr_list, tpr_list, label='AUC score: %.2f' % auc_score);
    plt.xlabel('假正率', fontsize=15)
    plt.ylabel('真正率', fontsize=15)
    plt.title('ROC曲线')
    plt.legend(fontsize=15)
    plt.show()

def part3():
    X,y = load_data('part3')
    # plt.figure(figsize=(10, 8))
    # plt.scatter(X, y)
    # plt.xlabel('Input', fontsize=15)
    # plt.ylabel('Target', fontsize=15)
    # plt.title('Toy回归问题', fontsize=15)
    # plt.show()
    X = tf.constant(X)
    y = tf.constant(y)
    # y = tf.reshape(y,[tf.size(y),1])
    y = tf.reshape(y,[-1,1])
    # optimizer = tf.train.GradientDescentOptimizer(1e-4) #R2 score:  0.8193253447688115
    optimizer = tf.train.AdamOptimizer(0.5) # R2 score:  R2 score:  0.8516558598479806
    model = two_layers_nn(output_size=1,loss_type='regression')
    model.fit(X,y,optimizer,300,track_accuracy=False)
    preds = model.predict(X)
    r2_score = r2(y, preds)
    print('R2 score: ', r2_score.numpy())
    # Create datapoints between X_min and X_max to visualize the line of best fit
    X_best_fit = np.arange(X.numpy().min(), X.numpy().max(), 0.001)[:, None]
    # Predictions on X_best_fit
    preds_best_fit = model.predict(X_best_fit)
    plt.figure(figsize=(10,8))
    plt.scatter(X.numpy(), y.numpy())  # Original datapoints
    plt.plot(X_best_fit, preds_best_fit.numpy(), color='k',
             linewidth=6, label='$R^2$ score: %.2f' % r2_score)  # Our predictions
    plt.xlabel('Input', fontsize=15)
    plt.ylabel('Target', fontsize=15)
    plt.title('Toy回归问题', fontsize=15)
    plt.legend(fontsize=15)
    plt.show()

if __name__ == '__main__':
    tfe.enable_eager_execution()
    # part1()
    # part2()
    part3()