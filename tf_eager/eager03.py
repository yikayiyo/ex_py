import tensorflow as tf
import tensorflow.contrib.eager as tfe
from sklearn.datasets import make_moons
from eager02 import precision
from eager02 import recall


class simple_nn(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_layer = tf.layers.Dense(10, activation=tf.nn.relu)
        self.output_layer = tf.layers.Dense(2, activation=None)

    def predict(self, input_data):
        hidden_activations = self.dense_layer(input_data)
        logits = self.output_layer(hidden_activations)
        return logits

    def loss_fn(self, input_data, target):
        logits = self.predict(input_data)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=target, logits=logits)
        return loss

    def grads_fn(self, input_data, target):
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(input_data, target)
        return tape.gradient(loss, self.variables)

    def fit(self, input_data, target, optimizer, num_epochs=500, verbose=50):
        for i in range(num_epochs):
            grads = self.grads_fn(input_data, target)
            optimizer.apply_gradients(zip(grads, self.variables))
            if (i == 0) | ((i + 1) % verbose == 0):
                print('Loss at epoch %d: %f' % (i + 1, self.loss_fn(input_data, target).numpy()))
def fit_and_save(model,optimizer,input_data,target):
    model.fit(input_data, target, optimizer, num_epochs=500, verbose=50)
    # Specify checkpoint directory
    checkpoint_directory = 'models_checkpoints/SimpleNN/'
    # Create model checkpoint
    checkpoint = tfe.Checkpoint(optimizer=optimizer,
                                model=model,
                                optimizer_step=tf.train.get_or_create_global_step())
    # Save trained model
    checkpoint.save(file_prefix=checkpoint_directory)

def restore_model(model,optimizer):
    # model是重新初始化的
    # 指定checkpoint目录
    checkpoint_directory = 'models_checkpoints/SimpleNN/'
    # Create model checkpoint
    checkpoint = tfe.Checkpoint(optimizer=optimizer,
                                model=model,
                                optimizer_step=tf.train.get_or_create_global_step())
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
if __name__ == '__main__':
    tfe.enable_eager_execution()
    X, y = make_moons(n_samples=100, noise=0.1, random_state=2018)
    X_train, y_train = tf.constant(X[:80, :]), tf.constant(y[:80])
    X_test, y_test = tf.constant(X[80:, :]), tf.constant(y[80:])
    optimizer = tf.train.GradientDescentOptimizer(5e-1)
    model = simple_nn()
    # 执行各个步骤时，另一个步骤的代码要注释掉
    # 步骤一：训练模型并保存
    # fit_and_save(model,optimizer,X_train,y_train) #Loss at epoch 500: 0.035850
    # 步骤二：恢复模型
    restore_model(model,optimizer)
    # 运行一个批次检查loss是否和保存之前的结果近似
    # model.fit(X_train, y_train, optimizer, num_epochs=1)
    # test集合上测试
    logits = model.predict(X_test)
    preds = tf.argmax(logits, axis=1)
    conf_matrix = tf.confusion_matrix(y_test, preds, num_classes=2)
    print('混淆矩阵: \n', conf_matrix.numpy())
    precision_score = precision(y, preds, conf_matrix, weights=None)
    print('Average precision: ', precision_score.numpy())
    recall_score = recall(y, preds, conf_matrix, weights=None)
    print('Average recall: ', recall_score.numpy())
    # - 0- looks good
    # 混淆矩阵:
    # [[11  0]
    #  [0  9]]
    # Average
    # precision: 1.0
    # Average
    # recall: 1.0